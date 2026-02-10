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


@dataclass
class GenerationOutput:
    """A single token output."""
    token_id: int
    text: str
    finish_reason: Optional[str] = None  # "stop", "length", or None


class Scheduler:
    """FIFO scheduler — one request at a time."""

    def __init__(self, model: KrasisModel):
        self.model = model
        self._lock = asyncio.Lock()

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[GenerationOutput]:
        """Generate tokens one at a time, yielding each as it's produced."""
        async with self._lock:
            # Run generation in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()

            # We use a queue to pass tokens from the generation thread
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

            # Yield tokens as they arrive
            while True:
                output = await token_queue.get()
                if output is None:
                    break
                yield output
                if output.finish_reason is not None:
                    break

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

        stop_ids = set(request.stop_token_ids or [cfg.eos_token_id])
        prompt_tokens = request.prompt_tokens
        device = torch.device(model.ranks[0].device)

        from krasis.kv_cache import SequenceKVState
        from krasis.sampler import sample

        # Create sequence states for each rank's KV cache
        seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

        start_time = time.perf_counter()
        generated_count = 0

        try:
            with torch.inference_mode():
                # ── Prefill ──
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
                    next_logits, request.temperature, request.top_k, request.top_p
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

                # ── Decode loop ──
                decode_start = time.perf_counter()

                for step in range(request.max_new_tokens - 1):
                    pos = len(prompt_tokens) + step
                    token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
                    pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)

                    logits = model.forward(token_tensor, pos_tensor, seq_states)
                    next_token = sample(
                        logits, request.temperature, request.top_k, request.top_p
                    ).item()
                    generated_count += 1

                    text = tokenizer.decode_incremental(next_token)
                    finish = "stop" if next_token in stop_ids else None
                    if generated_count >= request.max_new_tokens and finish is None:
                        finish = "length"

                    output = GenerationOutput(next_token, text, finish)
                    loop.call_soon_threadsafe(queue.put_nowait, output)

                    if finish:
                        break

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
