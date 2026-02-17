"""FastAPI HTTP server — OpenAI-compatible /v1/chat/completions.

Supports streaming (SSE) and blocking responses.

Usage:
    python -m krasis.server --model-path /path/to/Kimi-K2.5 --pp-partition 31,30
"""

import argparse
import atexit
import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from pathlib import Path
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


@app.post("/v1/timing")
async def toggle_timing(request: Request):
    """Toggle timing flags at runtime. POST with {"prefill": true/false, "decode": true/false}."""
    from krasis.timing import TIMING
    body = await request.json()
    result = {}
    if "prefill" in body:
        TIMING.prefill = bool(body["prefill"])
        result["prefill"] = TIMING.prefill
    if "decode" in body:
        TIMING.decode = bool(body["decode"])
        result["decode"] = TIMING.decode
    if "diag" in body:
        TIMING.diag = bool(body["diag"])
        result["diag"] = TIMING.diag
    if not result:
        result = {"prefill": TIMING.prefill, "decode": TIMING.decode, "diag": TIMING.diag}
    logger.info("Timing flags: prefill=%s decode=%s diag=%s", TIMING.prefill, TIMING.decode, TIMING.diag)
    return result


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


def _build_heatmap(model: KrasisModel, save_path: str) -> str:
    """Build expert activation heatmap by running active_only inference.

    Switches to active_only mode, runs a large prompt to gather activation
    counts, saves the heatmap, then switches back.  Returns path to saved file.
    """
    import gc, os, torch

    # Build a ~10K token prompt
    sections = [
        "Explain distributed consensus algorithms including Paxos, Raft, and PBFT. ",
        "Describe database transaction isolation levels and their trade-offs. ",
        "Discuss compiler optimization passes such as dead code elimination and loop unrolling. ",
        "Explain the CAP theorem and its practical implications for system design. ",
        "Describe memory management strategies in operating systems including paging and segmentation. ",
        "Discuss the principles of functional programming and category theory. ",
        "Explain how neural network backpropagation works with gradient descent. ",
        "Describe the architecture of modern CPUs including pipelining and branch prediction. ",
        "Discuss cryptographic primitives including AES, RSA, and elliptic curve cryptography. ",
        "Explain container orchestration with Kubernetes including pods, services, and deployments. ",
    ]
    content = ""
    while True:
        for section in sections:
            content += section
        tokens = model.tokenizer.apply_chat_template([{"role": "user", "content": content}])
        if len(tokens) >= 10000:
            tokens = tokens[:10000]
            break

    # Switch to active_only mode (tracks activations, cheapest GPU mode)
    for layer in model.layers:
        if hasattr(layer, 'gpu_prefill_manager'):
            layer.gpu_prefill_manager = None
    model.gpu_prefill_managers.clear()
    gc.collect()
    torch.cuda.empty_cache()

    model.gpu_prefill_enabled = True
    model.expert_divisor = -1  # active_only
    model._init_gpu_prefill()

    # Run inference to gather heatmap
    logger.info("Building heatmap with %d tokens...", len(tokens))
    with torch.inference_mode():
        model.generate(tokens, max_new_tokens=128, temperature=0.6)

    # Save heatmap
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for manager in model.gpu_prefill_managers.values():
        manager.save_heatmap(save_path)
        break
    logger.info("Heatmap saved to %s", save_path)

    # Switch back to HCS mode
    for layer in model.layers:
        if hasattr(layer, 'gpu_prefill_manager'):
            layer.gpu_prefill_manager = None
    model.gpu_prefill_managers.clear()
    gc.collect()
    torch.cuda.empty_cache()

    model.expert_divisor = -3  # hot_cached_static
    model._init_gpu_prefill()

    return save_path


def _warmup_model(model: KrasisModel):
    """Run a short generation to warm up GPU kernels, expert DMA, and CUDA caches.

    This ensures the first real user request runs at normal speed instead of
    paying cold-start penalties (kernel compilation, first DMA, etc.).
    """
    import torch

    logger.info("Warming up model (short generation)...")
    t0 = time.time()

    try:
        # Use generate() which handles all state setup/cleanup (KV, linear attn, etc.)
        warmup_tokens = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hi"}]
        )
        with torch.inference_mode():
            model.generate(
                warmup_tokens,
                max_new_tokens=5,
                temperature=0.6,
            )

        elapsed = time.time() - t0
        logger.info("Warmup complete (%.1fs) — server ready at full speed", elapsed)
    except Exception as e:
        logger.warning("Warmup failed (non-fatal): %s", e)
        # generate() cleans up its own state in its finally block


_registry_file: Optional[Path] = None


def _write_registry(host: str, port: int, model_name: str) -> None:
    """Write a server registry entry to ~/.krasis/servers/{pid}.json."""
    global _registry_file
    registry_dir = Path.home() / ".krasis" / "servers"
    registry_dir.mkdir(parents=True, exist_ok=True)
    _registry_file = registry_dir / f"{os.getpid()}.json"
    entry = {
        "pid": os.getpid(),
        "port": port,
        "host": host,
        "model": model_name,
        "started": int(time.time()),
    }
    _registry_file.write_text(json.dumps(entry))
    logger.info("Registry entry written: %s", _registry_file)


def _remove_registry() -> None:
    """Remove the server registry entry on shutdown."""
    global _registry_file
    if _registry_file is not None:
        try:
            _registry_file.unlink(missing_ok=True)
            logger.info("Registry entry removed: %s", _registry_file)
        except OSError:
            pass
        _registry_file = None


def main():
    parser = argparse.ArgumentParser(description="Krasis standalone LLM server")
    parser.add_argument("--model-path", required=True, help="Path to HF model")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs (auto-detected if omitted)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--krasis-threads", type=int, default=48,
                        help="CPU threads for expert computation")
    parser.add_argument("--kv-dtype", default="fp8_e4m3",
                        choices=["fp8_e4m3", "bf16"])
    parser.add_argument("--kv-cache-mb", type=int, default=2000,
                        help="KV cache size in MB (default: 2000)")
    parser.add_argument("--heatmap-path", default=None,
                        help="Path to expert_heatmap.json for HCS init")
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
    parser.add_argument("--gguf-path", default=None,
                        help="Path to GGUF file for CPU experts")
    parser.add_argument("--force-load", action="store_true",
                        help="Force reload of cached weights")
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

    # ── Load model with HCS strategy ──
    import os, json
    from krasis.config import ModelConfig

    cfg = ModelConfig.from_model_path(args.model_path)
    num_layers = cfg.num_hidden_layers
    num_gpus_available = args.num_gpus or torch.cuda.device_count()

    pp_partition = [num_layers]  # PP=1: all layers on primary GPU
    logger.info("HCS strategy: PP=1, %d GPUs available", num_gpus_available)

    _model = KrasisModel(
        model_path=args.model_path,
        pp_partition=pp_partition,
        num_gpus=num_gpus_available,
        kv_dtype=kv_dtype,
        krasis_threads=args.krasis_threads,
        quant_cfg=quant_cfg,
        expert_divisor=-3,  # hot_cached_static
        gguf_path=args.gguf_path,
        force_load=args.force_load,
        gpu_prefill_threshold=1,  # GPU decode always on for HCS
        kv_cache_mb=args.kv_cache_mb,
    )

    logger.info("Loading model...")
    _model.load()

    # Resolve heatmap: cached > build
    cache_dir = os.path.join(args.model_path, ".krasis_cache")
    heatmap_path = args.heatmap_path
    if not heatmap_path:
        heatmap_path = os.path.join(cache_dir, "auto_heatmap.json")

    if not os.path.exists(heatmap_path):
        logger.info("No heatmap found — building via calibration...")
        heatmap_path = _build_heatmap(_model, heatmap_path)
    else:
        logger.info("Using cached heatmap: %s", heatmap_path)

    # CUDA runtime warmup — triggers cuBLAS + Triton kernel compilation
    # before the allocation loop so VRAM measurements are accurate.
    # Full model warmup happens inside the allocation loop's first inference.
    _model.warmup_cuda_runtime()

    # ── HCS allocation loop: start small, step up until close to max VRAM ──
    import torch as _torch

    # Build 10K test prompt once — need enough text to exceed 10K tokens after tokenization
    _test_content = (
        "Explain distributed consensus algorithms including Paxos, Raft, and PBFT. "
        "Describe database transaction isolation levels and their trade-offs. "
        "Discuss compiler optimization passes such as dead code elimination. "
    ) * 300
    _test_tokens = _model.tokenizer.apply_chat_template(
        [{"role": "user", "content": _test_content}]
    )[:10000]
    logger.info("Test prompt: %d tokens", len(_test_tokens))

    for dev_str, manager in _model.gpu_prefill_managers.items():
        total_vram = _torch.cuda.get_device_properties(manager.device).total_memory
        initial_budget_mb = max(500, int(total_vram / (1024 * 1024) / 3))

        # Step 1: Load a conservative budget and run 10K inference to measure cost
        manager.clear_hcs()
        manager._init_hot_cached_static(
            heatmap_path=heatmap_path, expert_budget_mb=initial_budget_mb,
        )
        n_experts = sum(manager._hcs_num_pinned.values())
        free_after_load = _torch.cuda.mem_get_info(manager.device)[0]
        logger.info(
            "HCS calibration: budget=%d MB, %d experts, %d MB free after load on %s",
            initial_budget_mb, n_experts, free_after_load // (1024*1024), dev_str,
        )

        ok, info = manager.validate_gpu_allocation(_model, _test_tokens)
        if not ok:
            logger.error(
                "FATAL: HCS allocation failed on %s — even %d MB of experts OOMs. "
                "Not enough VRAM for inference. Try: smaller --kv-cache-mb or a smaller model.",
                dev_str, initial_budget_mb,
            )
            sys.exit(1)

        inference_cost_bytes = info["inference_cost_bytes"]
        logger.info("Measured inference cost: %d MB", inference_cost_bytes // (1024*1024))

        # Step 2: Calculate max expert budget from measured cost
        # free_after_load = free VRAM with initial_budget of experts loaded
        # Adding more experts reduces free VRAM 1:1
        # We need inference_cost * 1.2 headroom for safe operation
        headroom_needed = int(inference_cost_bytes * 1.2)
        extra_available = free_after_load - headroom_needed
        if extra_available > 0:
            max_budget_mb = initial_budget_mb + extra_available // (1024 * 1024)
        else:
            max_budget_mb = initial_budget_mb
        logger.info(
            "HCS calculated max budget: %d MB (%d MB initial + %d MB extra headroom)",
            max_budget_mb, initial_budget_mb, extra_available // (1024*1024),
        )

        # Step 3: Reload at max budget and verify with one final inference
        manager.clear_hcs()
        manager._init_hot_cached_static(
            heatmap_path=heatmap_path, expert_budget_mb=max_budget_mb,
        )
        n_experts = sum(manager._hcs_num_pinned.values())
        free_after_load = _torch.cuda.mem_get_info(manager.device)[0]
        logger.info(
            "HCS final load: budget=%d MB, %d experts, %d MB free on %s",
            max_budget_mb, n_experts, free_after_load // (1024*1024), dev_str,
        )

        ok, info = manager.validate_gpu_allocation(_model, _test_tokens)
        if not ok:
            # Final load OOM'd — fall back to calibration budget
            logger.warning(
                "Final load OOM at %d MB — falling back to calibration budget %d MB",
                max_budget_mb, initial_budget_mb,
            )
            manager.clear_hcs()
            manager._init_hot_cached_static(
                heatmap_path=heatmap_path, expert_budget_mb=initial_budget_mb,
            )
            n_experts = sum(manager._hcs_num_pinned.values())

        logger.info(
            "HCS allocation complete on %s: %d experts cached",
            dev_str, n_experts,
        )

    logger.info("HCS ready")

    # Run benchmark if requested (after model load + strategy, before serving)
    if args.benchmark:
        from krasis.benchmark import KrasisBenchmark
        bench = KrasisBenchmark(_model)
        bench.run()

    logger.info("Model loaded, starting server on %s:%d", args.host, args.port)

    _scheduler = Scheduler(_model)

    # ── Server registry: write entry + register cleanup ──
    _write_registry(args.host, args.port, _model_name)
    atexit.register(_remove_registry)

    # Use uvicorn.Server directly so we can patch handle_exit.
    # Default uvicorn graceful shutdown waits for active connections,
    # but generation threads block in Rust/CUDA and never finish.
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)

    def _handle_exit(sig, frame):
        if server.should_exit:
            # Second Ctrl-C — force kill immediately
            _remove_registry()
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
