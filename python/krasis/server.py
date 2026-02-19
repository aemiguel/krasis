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

# ANSI formatting for status output
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_DIM = "\033[2m"
_NC = "\033[0m"


def _status(label: str) -> None:
    """Print a highlighted status section header (also logged)."""
    print(f"\n{_BOLD}{_CYAN}▸ {label}{_NC}", flush=True)
    logger.info("── %s ──", label)

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


@app.post("/v1/hcs/reload")
async def reload_hcs(request: Request):
    """Reload HCS with a different allocation mode. POST with {"mode": "greedy"|"uniform"}."""
    body = await request.json()
    mode = body.get("mode", "uniform")
    if mode not in ("greedy", "uniform"):
        return {"error": f"Unknown mode: {mode}. Use 'greedy' or 'uniform'."}

    import os as _os
    import torch

    manager = list(_model.gpu_prefill_managers.values())[0]
    heatmap_path = _os.path.join(
        _model.cfg.model_path, ".krasis_cache", "auto_heatmap.json"
    )

    # Unified budget logic for every GPU
    devices = [hcs_dev.device for hcs_dev in manager._hcs_devices]
    device_budgets = {}
    for hcs_dev in manager._hcs_devices:
        free = torch.cuda.mem_get_info(hcs_dev.device)[0]
        # Add back current HCS usage on this device
        hcs_vram = sum(
            b["w13"].nbytes + b["w13_scale"].nbytes + b["w2"].nbytes + b["w2_scale"].nbytes
            for b in hcs_dev.buffers.values()
        )
        # Primary needs ~1 GB headroom for inference; others are pure storage
        headroom = 1000 if hcs_dev.is_primary else 300
        device_budgets[hcs_dev.device] = max(0, (free + hcs_vram) // (1024 * 1024) - headroom)

    manager.clear_hcs()
    torch.cuda.empty_cache()
    manager._init_hot_cached_static(
        heatmap_path=heatmap_path,
        allocation_mode=mode,
        devices=devices,
        device_budgets=device_budgets,
    )
    
    total_pinned = sum(sum(d.num_pinned.values()) for d in manager._hcs_devices)
    device_counts = {str(d.device): sum(d.num_pinned.values()) for d in manager._hcs_devices}
    logger.info(
        "HCS reloaded: mode=%s, total_experts=%d, counts=%s",
        mode, total_pinned, device_counts
    )
    return {"mode": mode, "total_experts": total_pinned}


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
    import os # Ensure os is in local scope
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True" # Mitigate fragmentation
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

    log_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    # Also log to krasis.log file (captures all logger output + uncaught exceptions)
    _log_file = os.path.join(os.getcwd(), "krasis.log")
    _file_handler = logging.FileHandler(_log_file, mode="a")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(_file_handler)

    # Capture uncaught exceptions to the log file
    _original_excepthook = sys.excepthook
    def _log_excepthook(exc_type, exc_value, exc_tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        _original_excepthook(exc_type, exc_value, exc_tb)
    sys.excepthook = _log_excepthook

    logger.info("Logging to %s", _log_file)

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

    _status("Loading model weights")
    _model.load()

    # Resolve heatmap: cached > build
    cache_dir = os.path.join(args.model_path, ".krasis_cache")
    heatmap_path = args.heatmap_path
    if not heatmap_path:
        heatmap_path = os.path.join(cache_dir, "auto_heatmap.json")

    if not os.path.exists(heatmap_path):
        _status("Building expert heatmap (calibration)")
        heatmap_path = _build_heatmap(_model, heatmap_path)
    else:
        _status("Loading cached heatmap")
        logger.info("Using cached heatmap: %s", heatmap_path)

    # CUDA runtime warmup — triggers cuBLAS + Triton kernel compilation
    # before the allocation loop so VRAM measurements are accurate.
    num_gpus_available = args.num_gpus or torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus_available)]
    _status("Warming up CUDA runtime")
    _model.warmup_cuda_runtime(devices)

    # ── Unified HCS allocation loop ──
    _status("Allocating HCS expert cache")
    logger.info("Unified HCS: initializing with devices: %s", [str(d) for d in devices])

    # Build 10K test prompt for validation — need enough text to exceed 10K tokens
    _test_content = (
        "Explain distributed consensus algorithms including Paxos, Raft, and PBFT. "
        "Describe database transaction isolation levels and their trade-offs. "
        "Discuss compiler optimization passes such as dead code elimination. "
    ) * 300
    _test_tokens = _model.tokenizer.apply_chat_template(
        [{"role": "user", "content": _test_content}]
    )[:10000]
    logger.info("Test prompt: %d tokens", len(_test_tokens))

    # Step 1: Measure inference cost on PRIMARY device only.
    # Secondary GPUs only need base model VRAM — inference runs on primary.
    manager = list(_model.gpu_prefill_managers.values())[0]
    inference_costs = {}
    free_after_cal = None
    initial_budget_mb = None

    primary_dev = devices[0]
    total_vram = torch.cuda.get_device_properties(primary_dev).total_memory
    initial_budget_mb = max(500, int(total_vram / (1024 * 1024) / 3))
    # With EP, validation runs forward_prefill_layer_grouped which allocates
    # DMA buffers (~415 MB) + workspace + activations on primary GPU.
    # Reduce calibration budget to leave room for these.
    if len(devices) > 1:
        initial_budget_mb = max(500, initial_budget_mb - 1024)
    manager.clear_hcs()
    manager._init_hot_cached_static(
        heatmap_path=heatmap_path,
        devices=[primary_dev],
        device_budgets={primary_dev: initial_budget_mb},
    )
    n_cal = sum(manager._hcs_devices[0].num_pinned.values())
    free_before_val = torch.cuda.mem_get_info(primary_dev)[0]
    logger.info(
        "HCS calibration: %d experts at %d MB, %d MB free on %s",
        n_cal, initial_budget_mb, free_before_val // (1024 * 1024), primary_dev,
    )
    ok, info = manager.validate_gpu_allocation(_model, _test_tokens)
    if not ok:
        logger.error("FATAL: calibration failed on %s.", primary_dev)
        sys.exit(1)
    inference_costs[primary_dev] = info["inference_cost_bytes"]
    # Measure free AFTER validation so AO buffer + decode allocations are captured
    free_after_cal = torch.cuda.mem_get_info(primary_dev)[0]
    logger.info(
        "Post-validation free on %s: %d MB (decode buffers allocated)",
        primary_dev, free_after_cal // (1024 * 1024),
    )
    manager.clear_hcs()
    torch.cuda.empty_cache()
    logger.info(
        "Inference cost on %s: %d MB",
        primary_dev, inference_costs[primary_dev] // (1024 * 1024),
    )

    # Secondary GPUs: inference cost is just base model VRAM (no compute overhead)
    for dev in devices[1:]:
        alloc = torch.cuda.memory_allocated(dev)
        inference_costs[dev] = alloc
        logger.info(
            "Inference cost on %s: %d MB (base model only, no compute)",
            dev, alloc // (1024 * 1024),
        )

    # Step 2: Calculate budget for EVERY GPU using measured costs
    # Minimum headroom: 1.2x inference cost OR 1 GB, whichever is larger.
    # Non-primary GPUs have low measured inference cost (no CUDA runtime, no
    # Marlin workspace) but still need headroom for DMA transfer buffers
    # (~500 MB), PyTorch allocator fragmentation, and activation intermediates
    # during HCS decode.
    MIN_HEADROOM_MB = 1024
    # EP adds DMA buffers (~500 MB) + activation copies on primary GPU during prefill
    EP_OVERHEAD_MB = 1024 if len(devices) > 1 else 0
    device_budgets = {}
    for i, dev in enumerate(devices):
        free_mb = torch.cuda.mem_get_info(dev)[0] // (1024 * 1024)
        if i == 0:
            headroom_mb = max(MIN_HEADROOM_MB, int(inference_costs[dev] * 1.2) // (1024 * 1024)) + EP_OVERHEAD_MB
        else:
            # Secondary: no decode (HCS is GPU-0-only), only need DMA buffer headroom
            headroom_mb = MIN_HEADROOM_MB

        if i == 0:
            # Primary: reclaim calibration budget since we cleared it
            budget = max(0, (free_after_cal // (1024 * 1024)) + initial_budget_mb - headroom_mb)
        else:
            budget = max(0, free_mb - headroom_mb)

        device_budgets[dev] = budget
        logger.info(
            "GPU %s: %d MB free, headroom=%d MB (measured, min %d), budget=%d MB",
            dev, free_mb, headroom_mb, MIN_HEADROOM_MB, budget,
        )

    # Step 3: Load HCS on primary GPU only (multi-GPU HCS decode is unstable;
    # both GPUs still used for EP prefill via gpu_prefill_managers)
    manager.clear_hcs()
    torch.cuda.empty_cache()
    manager._init_hot_cached_static(
        heatmap_path=heatmap_path,
        devices=[primary_dev],
        device_budgets={primary_dev: device_budgets[primary_dev]},
        allocation_mode="greedy",
    )


    # Step 5: Validate with real inference on primary
    ok, info = manager.validate_gpu_allocation(_model, _test_tokens)
    if not ok:
        logger.error("FATAL: HCS validation failed with final budgets: %s", device_budgets)
        sys.exit(1)

    total_pinned = sum(sum(d.num_pinned.values()) for d in manager._hcs_devices)
    device_counts = {str(d.device): sum(d.num_pinned.values()) for d in manager._hcs_devices}
    logger.info("HCS ready: %d experts across %d GPUs. Counts: %s", total_pinned, len(devices), device_counts)

    # Run benchmark if requested (after model load + strategy, before serving)
    if args.benchmark:
        from krasis.benchmark import KrasisBenchmark
        bench = KrasisBenchmark(_model)
        bench.run()
        sys.exit(0)

    _status(f"Server ready on {args.host}:{args.port}")
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
