"""Krasis LLM server — Rust HTTP server with Python GPU prefill.

Usage:
    python -m krasis.server --model-path /path/to/model
"""

import argparse
import atexit
import gc
import json
import logging
import os
import select
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

from krasis.config import QuantConfig, cache_dir_for_model
from krasis.model import KrasisModel

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

_model: Optional[KrasisModel] = None
_model_name: str = "unknown"


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
    model.layer_group_size = 1
    model._init_gpu_prefill()

    # Enable heatmap collection on all managers
    for manager in model.gpu_prefill_managers.values():
        manager.enable_heatmap()

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

    model.layer_group_size = 1
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
        except Exception:
            pass
        _registry_file = None


def _cleanup_cuda():
    """Release all CUDA contexts to prevent zombie GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
            torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    import os # Ensure os is in local scope
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True" # Mitigate fragmentation

    # Register cleanup early to prevent CUDA zombie processes
    atexit.register(_cleanup_cuda)
    def _force_exit_handler(sig, frame):
        _cleanup_cuda()
        os._exit(1)
    signal.signal(signal.SIGTERM, _force_exit_handler)
    # ── Pre-parse --config to load defaults from file ──
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None,
                     help="Path to config file (KEY=VALUE format). "
                          "CLI args override config file values.")
    pre_args, remaining_argv = pre.parse_known_args()

    config_defaults = {}
    if pre_args.config:
        config_path = pre_args.config
        if not os.path.isfile(config_path):
            print(f"Error: config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        # Mapping from CFG_* keys (used in ~/.krasis/config) to argparse dests
        _CFG_KEY_MAP = {
            "MODEL_PATH": "model_path",
            "CFG_SELECTED_GPUS": "_selected_gpus",  # special: comma list → num_gpus
            "CFG_PP_PARTITION": None,  # not used by server
            "CFG_LAYER_GROUP_SIZE": "layer_group_size",
            "CFG_KV_DTYPE": "kv_dtype",
            "CFG_GPU_EXPERT_BITS": "gpu_expert_bits",
            "CFG_CPU_EXPERT_BITS": "cpu_expert_bits",
            "CFG_ATTENTION_QUANT": "attention_quant",
            "CFG_SHARED_EXPERT_QUANT": "shared_expert_quant",
            "CFG_DENSE_MLP_QUANT": "dense_mlp_quant",
            "CFG_LM_HEAD_QUANT": "lm_head_quant",
            "CFG_KRASIS_THREADS": "krasis_threads",
            "CFG_HOST": "host",
            "CFG_PORT": "port",
            "CFG_GPU_PREFILL_THRESHOLD": "gpu_prefill_threshold",
            "CFG_GGUF_PATH": "gguf_path",
            "CFG_FORCE_LOAD": "force_load",
            "CFG_HCS": None,           # HCS disabled — ignore from config files
            "CFG_MULTI_GPU_HCS": None,  # HCS disabled — ignore from config files
            "CFG_KV_CACHE_MB": "kv_cache_mb",
        }
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # Determine argparse dest: check CFG_ map first, then fall back
                if key in _CFG_KEY_MAP:
                    dest = _CFG_KEY_MAP[key]
                    if dest is None:
                        continue  # skip keys not used by server
                    # Handle special cases for CFG_ format
                    if key == "CFG_SELECTED_GPUS":
                        # Convert comma-separated GPU indices to num_gpus count
                        gpu_list = [x.strip() for x in val.split(",") if x.strip()]
                        if gpu_list:
                            config_defaults["num_gpus"] = len(gpu_list)
                        continue
                    if key in ("CFG_FORCE_LOAD",):
                        # CFG_ format uses "1"/"" for booleans
                        config_defaults[dest] = val == "1"
                        continue
                else:
                    # Plain key format (key-name or key_name)
                    dest = key.replace("-", "_").lower()
                # Convert "true"/"false" strings for store_true args
                if isinstance(val, str) and val.lower() == "true":
                    config_defaults[dest] = True
                elif isinstance(val, str) and val.lower() == "false":
                    config_defaults[dest] = False
                else:
                    # Try int, then float, then string
                    try:
                        config_defaults[dest] = int(val)
                    except ValueError:
                        try:
                            config_defaults[dest] = float(val)
                        except ValueError:
                            config_defaults[dest] = val
        # Expand ~ in model_path
        if "model_path" in config_defaults and isinstance(config_defaults["model_path"], str):
            config_defaults["model_path"] = os.path.expanduser(config_defaults["model_path"])
        print(f"Loaded config from {config_path}: {config_defaults}")

    parser = argparse.ArgumentParser(description="Krasis standalone LLM server",
                                     parents=[pre])
    parser.add_argument("--model-path", required="model_path" not in config_defaults,
                        help="Path to HF model")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs (auto-detected if omitted)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--krasis-threads", type=int, default=40,
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
    parser.add_argument("--attention-quant", default="bf16", choices=["bf16"],
                        help="Quantization for attention weights (INT8 disabled — causes garbage output)")
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
    parser.add_argument("--no-stream-attention", action="store_true",
                        help="Disable streaming attention (load all layers persistently on GPU). "
                             "Only use if all layers fit in VRAM.")
    parser.add_argument("--layer-group-size", type=int, default=2,
                        help="Number of MoE layers to load per group during prefill (default: 2)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run standardized benchmark before starting server")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Run benchmark and exit (don't start server)")
    parser.add_argument("--stress-test", action="store_true",
                        help="Run stress test (diverse prompts) and exit")
    parser.add_argument("--perplexity", action="store_true",
                        help="Run perplexity evaluation and exit")
    parser.add_argument("--temperature", type=float, default=0.6)
    # Apply config file defaults, then parse CLI (CLI wins over config file)
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args(remaining_argv)

    # HCS is disabled — force off regardless of config file contents
    args.hcs = False
    args.multi_gpu_hcs = False

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

    global _model, _model_name
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

    # Expand ~ in paths (config files use ~/.krasis/...)
    args.model_path = os.path.expanduser(args.model_path)
    if args.heatmap_path:
        args.heatmap_path = os.path.expanduser(args.heatmap_path)
    if args.gguf_path:
        args.gguf_path = os.path.expanduser(args.gguf_path)

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
        layer_group_size=args.layer_group_size,
        gguf_path=args.gguf_path,
        force_load=args.force_load,
        gpu_prefill_threshold=1 if args.hcs else int(os.environ.get("KRASIS_PREFILL_THRESHOLD", "500")),
        kv_cache_mb=args.kv_cache_mb,
        stream_attention=not args.no_stream_attention,
    )

    _status("Loading model weights")
    _model.load()

    # Resolve heatmap: cached > build
    cache_dir = cache_dir_for_model(args.model_path)
    heatmap_path = args.heatmap_path
    if not heatmap_path:
        heatmap_path = os.path.join(cache_dir, "auto_heatmap.json")

    if args.hcs:
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
    if not args.hcs:
        _status("Pure CPU MoE decode")
        logger.info("CPU decode: M=1 via Rust engine")
        _model._hcs_device = None
        _model._multi_gpu_hcs = False
    else:
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

        # ── HCS allocation ──
        # Default: single-GPU HCS on the GPU with most free VRAM (typically GPU1).
        # --multi-gpu-hcs: pin experts on ALL GPUs (more experts, but CPU bounce hurts decode).
        primary_dev = devices[0]
        use_multi_gpu_hcs = args.multi_gpu_hcs and len(devices) > 1
        if use_multi_gpu_hcs:
            hcs_devices = devices
            cal_dev = max(devices, key=lambda d: torch.cuda.mem_get_info(d)[0])
        else:
            # Single-GPU HCS: pick the GPU with most free VRAM
            hcs_dev = max(devices, key=lambda d: torch.cuda.mem_get_info(d)[0]) if len(devices) > 1 else primary_dev
            hcs_devices = [hcs_dev]
            cal_dev = hcs_dev
        logger.info(
            "HCS %s: %d device(s) %s, calibration on %s (free: %s)",
            "multi-GPU" if use_multi_gpu_hcs else "single-GPU",
            len(hcs_devices), [str(d) for d in hcs_devices], cal_dev,
            ", ".join(f"{d}: {torch.cuda.mem_get_info(d)[0] // (1024*1024)} MB" for d in devices),
        )

        # Use the GPU0 (primary) manager to own HCS state — decode routing
        # happens on GPU0 so it needs the HCS device lookup tables.
        primary_manager = _model.gpu_prefill_managers.get(str(primary_dev))
        if primary_manager is None:
            logger.error("No GPU prefill manager for primary device %s", primary_dev)
            sys.exit(1)

        # For calibration, use the cal_dev manager (GPU1) so that _cross_device_moe
        # can correctly route to it during the validation forward pass.
        cal_manager = _model.gpu_prefill_managers.get(str(cal_dev)) if cal_dev != primary_dev else primary_manager
        if cal_manager is None:
            logger.error("No GPU prefill manager for calibration device %s", cal_dev)
            sys.exit(1)

        # Step 1: Calibration — measure per-expert total cost and inference cost.
        # Load a moderate number of experts on the calibration GPU, run a real
        # forward pass, and measure per-expert VRAM overhead.
        inference_costs = {}

        cal_manager.clear_hcs()
        torch.cuda.empty_cache()
        free_before_hcs = torch.cuda.mem_get_info(cal_dev)[0]
        initial_budget_mb = max(500, int(free_before_hcs / (1024 * 1024) / 3))

        # For calibration, set single-device HCS so decode forward uses cross-device MoE
        _model._hcs_device = cal_dev
        _model._multi_gpu_hcs = False

        MAX_CAL_RETRIES = 4
        for cal_attempt in range(MAX_CAL_RETRIES):
            cal_manager.clear_hcs()
            torch.cuda.empty_cache()
            free_before_hcs = torch.cuda.mem_get_info(cal_dev)[0]

            cal_manager._init_hot_cached_static(
                heatmap_path=heatmap_path,
                devices=[cal_dev],
                device_budgets={cal_dev: initial_budget_mb},
            )
            n_cal = sum(cal_manager._hcs_devices[0].num_pinned.values())
            free_after_hcs = torch.cuda.mem_get_info(cal_dev)[0]

            logger.info(
                "HCS calibration attempt %d on %s: %d experts, budget=%d MB, "
                "%d MB consumed (free: %d → %d MB)",
                cal_attempt + 1, cal_dev, n_cal, initial_budget_mb,
                (free_before_hcs - free_after_hcs) // (1024 * 1024),
                free_before_hcs // (1024 * 1024), free_after_hcs // (1024 * 1024),
            )

            if n_cal == 0:
                logger.error("FATAL: calibration loaded 0 experts on %s.", cal_dev)
                sys.exit(1)

            ok, info = cal_manager.validate_gpu_allocation(_model, _test_tokens)
            if ok:
                break
            # Validation OOMed — halve budget and retry
            initial_budget_mb = max(500, initial_budget_mb // 2)
            logger.warning(
                "Calibration validation OOM, retrying with budget=%d MB", initial_budget_mb,
            )
        else:
            logger.error("FATAL: calibration failed after %d attempts on %s.", MAX_CAL_RETRIES, cal_dev)
            sys.exit(1)
        inference_costs[cal_dev] = info["inference_cost_bytes"]
        logger.info(
            "Inference cost on %s: %d MB (transient)",
            cal_dev, inference_costs[cal_dev] // (1024 * 1024),
        )

        # Compute per-expert TOTAL cost: expert data + proportional share of
        # CUDA graphs, workspace, g_idx/sort_idx buffers, and other HCS overhead.
        hcs_total_consumed = free_before_hcs - free_after_hcs
        per_expert_total_bytes = hcs_total_consumed // max(1, n_cal)
        per_expert_data_bytes = cal_manager._per_expert_vram_bytes()
        hcs_overhead_mb = (hcs_total_consumed - n_cal * per_expert_data_bytes) // (1024 * 1024)
        logger.info(
            "Per-expert cost: %d bytes total (%d data + %d overhead), "
            "HCS overhead: %d MB (workspace + g_idx/sort_idx)",
            per_expert_total_bytes, per_expert_data_bytes,
            per_expert_total_bytes - per_expert_data_bytes, hcs_overhead_mb,
        )

        cal_manager.clear_hcs()
        torch.cuda.empty_cache()

        # Step 2: Calculate per-device HCS budgets.
        # Use per_expert_total_bytes (includes proportional graph/workspace overhead).
        #
        # Inference cost multiplier for GPU0 (primary):
        # - Without streaming: GPU0 holds ALL attention weights + KV + activations,
        #   so its transient peak is ~3x the calibration (GPU1) cost.
        # - With streaming: attention weights are streamed 2 layers at a time (~116 MB),
        #   so GPU0's transient cost is similar to GPU1 (just adds KV cache + small norms).
        #   Use 1.5x as a conservative safety factor.
        FRAG_MARGIN_MB = 512     # Base safety margin for prompt-length variation
        is_streaming = _model._stream_attn_enabled
        device_budgets = {}
        for dev in hcs_devices:
            free_mb = torch.cuda.mem_get_info(dev)[0] // (1024 * 1024)
            is_primary = (dev == primary_dev)
            inf_cost_mb = inference_costs[cal_dev] // (1024 * 1024)
            if is_primary:
                # Streaming: attention is 2 layers at a time, not all layers resident
                dev_inf_cost_mb = int(inf_cost_mb * 1.5) if is_streaming else inf_cost_mb * 3
            else:
                dev_inf_cost_mb = inf_cost_mb
            reserved_mb = dev_inf_cost_mb + FRAG_MARGIN_MB
            available_mb = max(0, free_mb - reserved_mb)
            # per_expert_total_bytes includes proportional CUDA graph + workspace overhead
            max_experts = available_mb * (1024 * 1024) // max(1, per_expert_total_bytes)
            budget_mb = max_experts * per_expert_data_bytes // (1024 * 1024)
            device_budgets[dev] = budget_mb
            logger.info(
                "HCS GPU %s: %d MB free, reserved=%d MB "
                "(inference=%d (%.1fx) + margin=%d), stream_attn=%s, "
                "budget=%d MB (%d max experts)",
                dev, free_mb, reserved_mb,
                dev_inf_cost_mb,
                1.5 if (is_primary and is_streaming) else (3.0 if is_primary else 1.0),
                FRAG_MARGIN_MB, is_streaming,
                budget_mb, max_experts,
            )

        # Step 3: Load HCS on selected GPUs, validate, retry with reduced budget on OOM
        #
        # For single-GPU cross-device mode (attention on GPU0, MoE on GPU1):
        #   Use the HCS device's own manager so _cross_device_moe finds it via
        #   gpu_prefill_managers[hcs_dev], and CUDA graphs can be captured (device match).
        # For multi-GPU mode: primary manager dispatches to all devices.
        if use_multi_gpu_hcs:
            hcs_load_manager = primary_manager
        else:
            hcs_load_manager = cal_manager  # Device-local manager (e.g. cuda:1)

        MAX_LOAD_RETRIES = 3
        for load_attempt in range(MAX_LOAD_RETRIES):
            hcs_load_manager.clear_hcs()
            torch.cuda.empty_cache()
            hcs_load_manager._init_hot_cached_static(
                heatmap_path=heatmap_path,
                devices=hcs_devices,
                device_budgets=device_budgets,
                allocation_mode="greedy",
            )

            # Set model flags for decode routing
            if use_multi_gpu_hcs:
                _model._hcs_device = None  # No single HCS device — multi-GPU mode
                _model._multi_gpu_hcs = True
            else:
                _model._hcs_device = hcs_devices[0]
                _model._multi_gpu_hcs = False

            ok, info = hcs_load_manager.validate_gpu_allocation(_model, _test_tokens)
            if ok:
                break

            # Validation OOMed — reduce the richest HCS GPU budget by 25% and retry.
            # Must fully clear HCS + graphs + CUDA cache to reset GPU state,
            # as OOM during graph-captured operations can corrupt GPU state.
            reduce_dev = max(hcs_devices, key=lambda d: device_budgets.get(d, 0))
            old_budget = device_budgets[reduce_dev]
            device_budgets[reduce_dev] = max(0, int(old_budget * 0.75))
            logger.warning(
                "HCS validation OOM (attempt %d/%d), reducing %s budget: %d → %d MB",
                load_attempt + 1, MAX_LOAD_RETRIES, reduce_dev,
                old_budget, device_budgets[reduce_dev],
            )
            hcs_load_manager.clear_hcs()
            gc.collect()
            torch.cuda.empty_cache()
        else:
            total_pinned = sum(sum(d.num_pinned.values()) for d in hcs_load_manager._hcs_devices)
            logger.error(
                "FATAL: HCS validation OOM after %d attempts with %d experts (budget %s). "
                "Reduce expert count or use more GPUs.",
                MAX_LOAD_RETRIES, total_pinned, {str(d): b for d, b in device_budgets.items()},
            )
            sys.exit(1)

        # Log margin for observability
        free_after = info["free_after_load_mb"]
        cost_mb = info["inference_cost_bytes"] // (1024 * 1024)
        margin = free_after - cost_mb
        total_pinned = sum(sum(d.num_pinned.values()) for d in hcs_load_manager._hcs_devices)
        device_counts = {str(d.device): sum(d.num_pinned.values()) for d in hcs_load_manager._hcs_devices}
        logger.info(
            "HCS ready: %d experts across %d GPU(s), margin=%d MB (free=%d, cost=%d). Counts: %s",
            total_pinned, len(hcs_devices), margin, free_after, cost_mb, device_counts,
        )

    # Run benchmark if requested (after model load + strategy, before serving)
    if args.benchmark or args.benchmark_only:
        from krasis.benchmark import KrasisBenchmark
        bench = KrasisBenchmark(_model)
        bench.run()
        if args.benchmark_only:
            sys.exit(0)

    # Run stress test if requested
    if args.stress_test:
        from krasis.stress_test import StressTest
        st = StressTest(_model)
        results = st.run()
        failed = sum(1 for r in results if r["status"].startswith("FAIL"))
        sys.exit(1 if failed > 0 else 0)

    # Run perplexity evaluation if requested
    if args.perplexity:
        _ppl_dir = os.path.join(os.path.dirname(__file__), "..", "..", "perplexity")
        sys.path.insert(0, os.path.dirname(_ppl_dir))
        from perplexity.measure_ppl import list_datasets, run_perplexity

        _status("Perplexity Evaluation")
        datasets = list_datasets()
        print("\nChoose dataset:")
        for i, ds in enumerate(datasets, 1):
            print(f"  {i}. {ds['name']:20s} ({ds['tokens_approx']} tokens)")
        print(f"  {len(datasets) + 1}. All datasets")

        choice = input(f"\nSelection [1]: ").strip() or "1"
        try:
            choice_idx = int(choice)
        except ValueError:
            print(f"Invalid selection: {choice}")
            sys.exit(1)

        if choice_idx == len(datasets) + 1:
            # Run all datasets
            selected = [ds["name"] for ds in datasets]
        elif 1 <= choice_idx <= len(datasets):
            selected = [datasets[choice_idx - 1]["name"]]
        else:
            print(f"Invalid selection: {choice_idx}")
            sys.exit(1)

        config = {
            "model_path": args.model_path,
            "gpu_expert_bits": args.gpu_expert_bits,
            "cpu_expert_bits": args.cpu_expert_bits,
            "attention_quant": args.attention_quant,
            "lm_head_quant": args.lm_head_quant,
            "layer_group_size": args.layer_group_size,
            "krasis_threads": args.krasis_threads,
            "kv_cache_mb": args.kv_cache_mb,
        }

        all_results = []
        for ds_name in selected:
            result = run_perplexity(model=_model, dataset_name=ds_name, config=config)
            all_results.append(result)

        # Print summary table if multiple datasets
        if len(all_results) > 1:
            print()
            bar = "\u2550" * 56
            print(bar)
            print("  PERPLEXITY SUMMARY")
            print(bar)
            print(f"  {'Dataset':20s} {'PPL':>10s} {'BPC':>8s} {'Tokens':>12s} {'Time':>8s}")
            print(f"  {'-' * 20} {'-' * 10} {'-' * 8} {'-' * 12} {'-' * 8}")
            for r in all_results:
                tok_s = r["num_tokens_scored"] / r["elapsed_s"] if r["elapsed_s"] > 0 else 0
                print(
                    f"  {r['dataset']:20s} {r['perplexity']:10.2f} {r['bits_per_char']:8.2f} "
                    f"{r['num_tokens_scored']:>12,} {r['elapsed_s']:7.1f}s"
                )
            print(bar)

        sys.exit(0)

    max_ctx = _model.get_max_context_tokens()
    _status(f"Server ready on {args.host}:{args.port}")
    print(f"  {_DIM}KV cache: {args.kv_cache_mb:,} MB → {max_ctx:,} max context tokens{_NC}", flush=True)
    print(f"  {_DIM}Press Q or Ctrl-C to stop{_NC}", flush=True)
    logger.info(
        "Model loaded, starting server on %s:%d (max context: %d tokens)",
        args.host, args.port, max_ctx,
    )

    # ── Server registry: write entry + register cleanup ──
    _write_registry(args.host, args.port, _model_name)
    atexit.register(_remove_registry)

    # ── Rust HTTP server — Python only for GPU prefill ──
    from krasis import RustServer

    tokenizer_path = os.path.join(args.model_path, "tokenizer.json")
    rust_server = RustServer(
        _model,
        args.host,
        args.port,
        _model_name,
        tokenizer_path,
        max_ctx,
    )

    def _handle_exit(sig, frame):
        rust_server.stop()
        try:
            sys.stderr = open(os.devnull, "w")
            sys.stdout = open(os.devnull, "w")
            logging.disable(logging.CRITICAL)
        except Exception:
            pass
        os.write(1, f"\n{_BOLD}{_GREEN}Server stopped.{_NC}\n".encode())

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    # Q to quit (background thread)
    def _stdin_listener():
        try:
            while rust_server.is_running():
                if select.select([sys.stdin], [], [], 0.5)[0]:
                    ch = sys.stdin.read(1)
                    if ch in ("q", "Q"):
                        _handle_exit(None, None)
                        break
        except (OSError, ValueError):
            pass
    if sys.stdin.isatty():
        t = threading.Thread(target=_stdin_listener, daemon=True)
        t.start()

    # run() releases the GIL and blocks until stop() is called
    rust_server.run()

    # ── Clean exit before Python teardown triggers cascading errors ──
    _remove_registry()
    _cleanup_cuda()
    os._exit(0)


if __name__ == "__main__":
    main()
