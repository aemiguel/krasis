"""Simple request scheduler for the Krasis server.

Single-request-at-a-time for now. Handles the generation loop
and yields tokens as they are generated.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

import torch

from krasis.model import KrasisModel

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """A single generation request."""
    request_id: str
    prompt_tokens: List[int]
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_k: int = 50
    top_p: float = 0.95
    stop_token_ids: List[int] = field(default_factory=list)
    presence_penalty: float = 0.0


@dataclass
class GenerationOutput:
    """A single token output."""
    token_id: int
    text: str
    finish_reason: Optional[str] = None  # "stop", "length", "cancelled", or None


class Scheduler:
    """FIFO scheduler — one request at a time."""

    def __init__(self, model: KrasisModel):
        self.model = model
        self._lock = asyncio.Lock()

    async def generate_stream(
        self, request: GenerationRequest, disconnect_event: Optional[asyncio.Event] = None
    ) -> AsyncIterator[GenerationOutput]:
        """Generate tokens one at a time, yielding each as it's produced.

        Args:
            request: The generation request.
            disconnect_event: If set, signals that the client has disconnected
                and generation should be cancelled.
        """
        async with self._lock:
            loop = asyncio.get_event_loop()
            token_queue: asyncio.Queue[Optional[GenerationOutput]] = asyncio.Queue()

            async def _produce():
                try:
                    await loop.run_in_executor(
                        None, self._generate_blocking, request, token_queue, loop
                    )
                except Exception as e:
                    logger.error("Generation error: %s", e, exc_info=True)
                    await token_queue.put(GenerationOutput(
                        token_id=-1, text="", finish_reason="error"
                    ))

            # Start producer
            producer = asyncio.create_task(_produce())

            # If we have a disconnect event, start a monitor that cancels
            # the Rust generate_loop when the client disconnects.
            cancel_task = None
            if disconnect_event is not None:
                async def _cancel_monitor():
                    await disconnect_event.wait()
                    cpu_decoder = self.model._cpu_decoder
                    if cpu_decoder is not None and hasattr(cpu_decoder, '_store'):
                        cpu_decoder._store.cancel()
                        logger.info("Client disconnected — cancelled generation for %s", request.request_id)
                cancel_task = asyncio.create_task(_cancel_monitor())

            try:
                # Yield tokens as they arrive
                while True:
                    output = await token_queue.get()
                    if output is None:
                        break
                    yield output
                    if output.finish_reason is not None:
                        break
            finally:
                if cancel_task is not None:
                    cancel_task.cancel()

            await producer

    def _generate_blocking(
        self,
        request: GenerationRequest,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ):
        """Blocking generation — runs in thread pool."""
        model = self.model
        cfg = model.cfg
        tokenizer = model.tokenizer

        stop_ids = set(request.stop_token_ids or ([cfg.eos_token_id] + list(cfg.extra_stop_token_ids)))
        prompt_tokens = request.prompt_tokens
        device = torch.device(model.ranks[0].device)

        from krasis.kv_cache import SequenceKVState
        from krasis.sampler import sample

        # Create sequence states for each rank's KV cache
        seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

        # Reset cancel flag before starting
        cpu_decoder = model._cpu_decoder
        if cpu_decoder is not None and hasattr(cpu_decoder, '_store'):
            cpu_decoder._store.reset_cancel()

        start_time = time.perf_counter()
        generated_count = 0

        try:
            # Reset linear attention recurrent state between requests.
            # Without this, conv_state and recurrent_state persist from the
            # previous request, corrupting output (repetition / garbage).
            if cfg.is_hybrid:
                for layer in model.layers:
                    if layer.layer_type == "linear_attention":
                        layer.attention.reset_state()

            with torch.inference_mode():
                # ── Prefill (GPU) ──
                prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
                positions = torch.arange(len(prompt_tokens), dtype=torch.int32, device=device)

                logits = model.forward(prompt_tensor, positions, seq_states)
                next_logits = logits[-1:, :]

                prefill_time = time.perf_counter() - start_time
                logger.info(
                    "Prefill: %d tokens in %.2fs (%.0f tok/s)",
                    len(prompt_tokens), prefill_time,
                    len(prompt_tokens) / prefill_time if prefill_time > 0 else 0,
                )

                next_token = sample(
                    next_logits, request.temperature, request.top_k, request.top_p,
                    presence_penalty=request.presence_penalty,
                ).item()
                generated_count += 1

                text = tokenizer.decode_incremental(next_token)
                finish = "stop" if next_token in stop_ids else None
                if generated_count >= request.max_new_tokens and finish is None:
                    finish = "length"

                output = GenerationOutput(next_token, text, finish)
                loop.call_soon_threadsafe(queue.put_nowait, output)

                if finish:
                    loop.call_soon_threadsafe(queue.put_nowait, None)
                    return

            # ── Decode (CPU) — weights already initialized at load time ──
            cpu_decoder.prepare(seq_states, request.max_new_tokens)

            decode_start = time.perf_counter()

            import os
            tokenizer_path = os.path.join(cfg.model_path, "tokenizer.json")

            def _token_callback(token_id, text, finish_reason):
                nonlocal generated_count
                generated_count += 1
                out = GenerationOutput(token_id, text, finish_reason)
                loop.call_soon_threadsafe(queue.put_nowait, out)

            cpu_decoder._store.generate_loop(
                first_token=next_token,
                start_position=len(prompt_tokens),
                max_tokens=request.max_new_tokens - 1,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                stop_ids=list(stop_ids),
                tokenizer_path=tokenizer_path,
                callback=_token_callback,
                presence_penalty=request.presence_penalty,
            )

            decode_time = time.perf_counter() - decode_start
            if generated_count > 1:
                decode_tps = (generated_count - 1) / decode_time if decode_time > 0 else 0
                logger.info(
                    "Decode: %d tokens in %.2fs (%.1f tok/s)",
                    generated_count - 1, decode_time, decode_tps,
                )

        finally:
            for s in seq_states:
                s.free()
            loop.call_soon_threadsafe(queue.put_nowait, None)
