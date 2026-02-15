"""FastAPI HTTP server — OpenAI-compatible /v1/chat/completions.

Supports streaming (SSE) and blocking responses.

Usage:
    python -m krasis.server --model-path /path/to/Kimi-K2.5 --pp-partition 31,30
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from krasis.config import QuantConfig
from krasis.model import KrasisModel
from krasis.scheduler import GenerationRequest, Scheduler

logger = logging.getLogger("krasis.server")

app = FastAPI(title="Krasis", version="0.1.0")
_scheduler: Optional[Scheduler] = None
_model: Optional[KrasisModel] = None
_model_name: str = "unknown"


@app.get("/health")
async def health():
    if _model is None or not _model._loaded:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": _model_name,
            "object": "model",
            "owned_by": "krasis",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 256)
    temperature = body.get("temperature", 0.6)
    top_k = body.get("top_k", 50)
    top_p = body.get("top_p", 0.95)

    # Tokenize
    prompt_tokens = _model.tokenizer.apply_chat_template(messages)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    stop_ids = [_model.cfg.eos_token_id]
    # Handle custom stop tokens
    if "stop" in body:
        stop = body["stop"]
        if isinstance(stop, str):
            stop = [stop]
        for s in stop:
            ids = _model.tokenizer.encode(s, add_special_tokens=False)
            stop_ids.extend(ids)

    gen_request = GenerationRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_token_ids=stop_ids,
    )

    logger.info(
        "Request %s: %d prompt tokens, max_new=%d, stream=%s",
        request_id, len(prompt_tokens), max_tokens, stream,
    )

    if stream:
        return StreamingResponse(
            _stream_response(gen_request),
            media_type="text/event-stream",
        )
    else:
        return await _blocking_response(gen_request)


async def _stream_response(request: GenerationRequest):
    """SSE streaming response."""
    created = int(time.time())

    async for output in _scheduler.generate_stream(request):
        chunk = {
            "id": request.request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": _model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": output.text} if output.finish_reason is None else {},
                "finish_reason": output.finish_reason,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


async def _blocking_response(request: GenerationRequest):
    """Non-streaming response — collect all tokens then return."""
    created = int(time.time())
    chunks = []
    finish_reason = None

    async for output in _scheduler.generate_stream(request):
        chunks.append(output.text)
        if output.finish_reason:
            finish_reason = output.finish_reason

    full_text = "".join(chunks)

    return {
        "id": request.request_id,
        "object": "chat.completion",
        "created": created,
        "model": _model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text},
            "finish_reason": finish_reason or "stop",
        }],
        "usage": {
            "prompt_tokens": len(request.prompt_tokens),
            "completion_tokens": len(chunks),
            "total_tokens": len(request.prompt_tokens) + len(chunks),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Krasis standalone LLM server")
    parser.add_argument("--model-path", required=True, help="Path to HF model")
    parser.add_argument("--pp-partition", default=None,
                        help="Comma-separated layer counts per GPU (e.g. 31,30)")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs (auto-detected if omitted)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--krasis-threads", type=int, default=48,
                        help="CPU threads for expert computation")
    parser.add_argument("--kv-dtype", default="fp8_e4m3",
                        choices=["fp8_e4m3", "bf16"])
    parser.add_argument("--expert-divisor", type=int, default=1,
                        help="Expert loading: -3=hot_cached_static, -2=lru, -1=active-only, 0=chunked, 1=persistent, >=2=layer-grouped")
    parser.add_argument("--cache-strategy", default=None,
                        choices=["none", "active_only", "static_pin", "weighted_pin", "lru", "hybrid", "hot_cached_static"],
                        help="Expert caching strategy (overrides expert-divisor for cache modes)")
    parser.add_argument("--heatmap-path", default=None,
                        help="Path to expert_heatmap.json for hot_cached_static init")
    parser.add_argument("--cuda-graphs", action="store_true", default=False,
                        help="Enable CUDA graph capture for M=1 decode (hot_cached_static only)")
    parser.add_argument("--gpu-expert-bits", type=int, default=4, choices=[4, 8],
                        help="Marlin quantization bits for GPU prefill experts")
    parser.add_argument("--cpu-expert-bits", type=int, default=4, choices=[4, 8],
                        help="Quantization bits for CPU decode experts")
    parser.add_argument("--attention-quant", default="int8", choices=["bf16", "int8"],
                        help="Quantization for attention weights")
    parser.add_argument("--shared-expert-quant", default="int8", choices=["bf16", "int8"],
                        help="Quantization for shared expert weights")
    parser.add_argument("--dense-mlp-quant", default="int8", choices=["bf16", "int8"],
                        help="Quantization for dense MLP weights")
    parser.add_argument("--lm-head-quant", default="int8", choices=["bf16", "int8"],
                        help="Quantization for lm_head weights")
    parser.add_argument("--gpu-prefill-threshold", type=int, default=300,
                        help="Min tokens to trigger GPU prefill (default: 300)")
    parser.add_argument("--gpu-decode", action="store_true", default=None,
                        help="Route M=1 decode through GPU (default: True for active_only/static_pin)")
    parser.add_argument("--no-gpu-decode", dest="gpu_decode", action="store_false",
                        help="Disable GPU decode (use CPU for M=1)")
    parser.add_argument("--gguf-path", default=None,
                        help="Path to GGUF file for CPU experts")
    parser.add_argument("--force-load", action="store_true",
                        help="Force reload of cached weights")
    parser.add_argument("--strategy", default="manual",
                        choices=["auto", "manual"],
                        help="auto: auto-discover best strategy on first run; manual: use explicit flags")
    parser.add_argument("--force-optimize", action="store_true",
                        help="Force re-run auto-optimiser even if cached results exist (--strategy auto only)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run standardized benchmark before starting server")
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    global _model, _scheduler, _model_name
    import torch

    pp_partition = None
    if args.pp_partition:
        pp_partition = [int(x.strip()) for x in args.pp_partition.split(",")]

    kv_dtype = torch.float8_e4m3fn if args.kv_dtype == "fp8_e4m3" else torch.bfloat16

    quant_cfg = QuantConfig(
        lm_head=args.lm_head_quant,
        attention=args.attention_quant,
        shared_expert=args.shared_expert_quant,
        dense_mlp=args.dense_mlp_quant,
        gpu_expert_bits=args.gpu_expert_bits,
        cpu_expert_bits=args.cpu_expert_bits,
    )

    _model_name = args.model_path.rstrip("/").split("/")[-1]

    if args.strategy == "auto":
        # ── Auto strategy: neutral init → auto-optimise or load cached ──
        logger.info("Strategy: auto (expert_divisor=0 for neutral init)")
        _model = KrasisModel(
            model_path=args.model_path,
            pp_partition=pp_partition,
            num_gpus=args.num_gpus,
            kv_dtype=kv_dtype,
            krasis_threads=args.krasis_threads,
            quant_cfg=quant_cfg,
            expert_divisor=0,  # neutral chunked — cheapest init
            gguf_path=args.gguf_path,
            force_load=args.force_load,
            gpu_prefill_threshold=300,
        )

        logger.info("Loading model...")
        _model.load()

        from krasis.auto_optimise import auto_optimise_or_load
        # --benchmark always forces fresh auto-optimise (never use cache)
        force = args.force_optimize or args.benchmark
        auto_optimise_or_load(_model, force=force)

    else:
        # ── Manual strategy: existing behavior ──
        # Map cache-strategy to expert_divisor
        expert_divisor = args.expert_divisor
        if args.cache_strategy:
            strategy_map = {
                "active_only": -1,
                "static_pin": -1,  # active_only + static pinning
                "weighted_pin": -1,  # active_only + weighted pinning
                "lru": -2,  # LRU cross-layer caching
                "hybrid": -2,  # LRU + static pinning
                "hot_cached_static": -3,  # hot on GPU (static), cold on CPU (parallel)
                "none": expert_divisor,  # Keep existing divisor
            }
            expert_divisor = strategy_map.get(args.cache_strategy, expert_divisor)
            logger.info("Cache strategy '%s' → expert_divisor=%d", args.cache_strategy, expert_divisor)

        # GPU decode: auto-enable for active_only/static_pin, or respect explicit flag
        gpu_decode = args.gpu_decode
        if gpu_decode is None:
            # Auto-enable for cache strategies that use active-only mode
            gpu_decode = args.cache_strategy in ("active_only", "static_pin", "weighted_pin", "lru", "hybrid", "hot_cached_static")

        gpu_prefill_threshold = args.gpu_prefill_threshold
        if gpu_decode:
            gpu_prefill_threshold = 1
            logger.info("GPU decode enabled: gpu_prefill_threshold=1 (M=1 decode routes through GPU)")

        _model = KrasisModel(
            model_path=args.model_path,
            pp_partition=pp_partition,
            num_gpus=args.num_gpus,
            kv_dtype=kv_dtype,
            krasis_threads=args.krasis_threads,
            quant_cfg=quant_cfg,
            expert_divisor=expert_divisor,
            gguf_path=args.gguf_path,
            force_load=args.force_load,
            gpu_prefill_threshold=gpu_prefill_threshold,
        )

        logger.info("Loading model...")
        _model.load()

        # Configure expert pinning for static/weighted/hybrid strategies
        if args.cache_strategy in ("static_pin", "weighted_pin", "hybrid"):
            strategy = "weighted" if args.cache_strategy == "weighted_pin" else "uniform"
            for dev_str, manager in _model.gpu_prefill_managers.items():
                manager.configure_pinning(
                    budget_mb=0,  # auto-detect from free VRAM
                    warmup_requests=1,  # Pin after first request
                    strategy=strategy,
                )
            logger.info("Expert pinning configured: strategy=%s", strategy)

        # Initialize hot_cached_static strategy
        if args.cache_strategy == "hot_cached_static":
            for dev_str, manager in _model.gpu_prefill_managers.items():
                manager._init_hot_cached_static(heatmap_path=args.heatmap_path)
                if args.cuda_graphs:
                    manager._init_cuda_graphs()
            logger.info("hot_cached_static initialized (cuda_graphs=%s)", args.cuda_graphs)

    # Run benchmark if requested (after model load + strategy, before serving)
    if args.benchmark:
        from krasis.benchmark import KrasisBenchmark
        bench = KrasisBenchmark(_model)
        bench.run()

    logger.info("Model loaded, starting server on %s:%d", args.host, args.port)

    _scheduler = Scheduler(_model)

    # Use uvicorn.Server directly so we can patch handle_exit.
    # Default uvicorn graceful shutdown waits for active connections,
    # but generation threads block in Rust/CUDA and never finish.
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)

    def _handle_exit(sig, frame):
        if server.should_exit:
            # Second Ctrl-C — force kill immediately
            logger.info("Forcing exit...")
            os._exit(0)
        # First Ctrl-C — tell uvicorn to stop, skip waiting for connections
        server.should_exit = True
        server.force_exit = True
        logger.info("Shutting down (press Ctrl-C again to force)...")

    server.handle_exit = _handle_exit
    server.run()


if __name__ == "__main__":
    main()
