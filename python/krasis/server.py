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
import subprocess
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
_YELLOW = "\033[33m"
_RED = "\033[31m"
_DIM = "\033[2m"
_NC = "\033[0m"


def _status(label: str) -> None:
    """Print a highlighted status section header (also logged)."""
    print(f"\n{_BOLD}{_CYAN}▸ {label}{_NC}", flush=True)
    logger.info("── %s ──", label)


def _detail(text: str) -> None:
    """Print a detail line under a status header (green, indented)."""
    print(f"  {_GREEN}{text}{_NC}", flush=True)


def _dim(text: str) -> None:
    """Print a dim info line (secondary details)."""
    print(f"  {_DIM}{text}{_NC}", flush=True)


def _warn(text: str) -> None:
    """Print a warning line (yellow, indented)."""
    print(f"  {_YELLOW}{text}{_NC}", flush=True)

_model: Optional[KrasisModel] = None
_model_name: str = "unknown"


def _start_session_bridge(host: str, port: int) -> Optional[subprocess.Popen]:
    """Start the Session messenger bridge as a subprocess.

    Returns the Popen object, or None if startup fails.
    """
    import shutil

    krasis_dir = Path(__file__).resolve().parent.parent.parent  # krasis repo root
    bridge_script = krasis_dir / "session-bridge.mjs"

    if not bridge_script.is_file():
        _warn(f"Session bridge script not found: {bridge_script}")
        return None

    # Find bun or node (prefer bun, fall back to node)
    runtime_bin = shutil.which("bun")
    # Also check ~/.bun/bin which the installer uses
    if not runtime_bin:
        bun_home = Path.home() / ".bun" / "bin" / "bun"
        if bun_home.is_file():
            runtime_bin = str(bun_home)
    if not runtime_bin:
        runtime_bin = shutil.which("node")
    if not runtime_bin:
        _warn("Session bridge: neither bun nor node found in PATH.")
        _warn("Run 'krasis-setup' to install Session messenger dependencies.")
        return None

    # Check if @session.js/client is installed, auto-install if not
    node_modules = krasis_dir / "node_modules" / "@session.js" / "client"
    if not node_modules.is_dir():
        pkg_json = krasis_dir / "package.json"
        if not pkg_json.is_file():
            _warn("Session bridge: package.json not found, cannot install dependencies.")
            return None
        # Pick installer: prefer bun install, fall back to npm install
        installer = None
        if "bun" in str(runtime_bin):
            installer = [runtime_bin, "install"]
        else:
            npm_bin = shutil.which("npm")
            if npm_bin:
                installer = [npm_bin, "install"]
        if not installer:
            _warn("Session bridge: npm not found, cannot install dependencies.")
            _warn("Run 'krasis-setup' to install Session messenger dependencies.")
            return None
        _detail("Installing Session messenger dependencies...")
        try:
            ret = subprocess.run(
                installer, cwd=str(krasis_dir),
                capture_output=True, timeout=120,
            )
            if ret.returncode != 0:
                stderr = ret.stderr.decode("utf-8", errors="replace")[:500]
                _warn(f"Session dependency install failed: {stderr}")
                return None
            _detail("Session dependencies installed.")
        except subprocess.TimeoutExpired:
            _warn("Session dependency install timed out.")
            return None
        except Exception as e:
            _warn(f"Session dependency install error: {e}")
            return None

    identity_path = Path.home() / ".krasis" / "session_identity"
    session_id_path = Path.home() / ".krasis" / "session_id"

    env = os.environ.copy()
    env["KRASIS_HOST"] = "127.0.0.1"  # bridge always connects locally
    env["KRASIS_PORT"] = str(port)

    try:
        run_args = [runtime_bin, "run", str(bridge_script)] if "bun" in str(runtime_bin) else [runtime_bin, str(bridge_script)]
        proc = subprocess.Popen(
            run_args + ["--host", "127.0.0.1", "--port", str(port)],
            cwd=str(krasis_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Wait briefly for identity to be created and Session ID written
        for _ in range(30):  # up to 3 seconds
            time.sleep(0.1)
            if session_id_path.is_file():
                sid = session_id_path.read_text().strip()
                if sid:
                    _detail(f"Session messenger: ON")
                    print(f"  {_BOLD}Session ID: {sid}{_NC}", flush=True)
                    logger.info("Session bridge started, ID: %s", sid)
                    break
        else:
            _detail("Session messenger: starting (ID pending...)")

        # Log bridge output in background
        def _log_session_output():
            try:
                for line in proc.stdout:
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                    if decoded:
                        logger.info("[session-bridge] %s", decoded)
            except Exception:
                pass
        t = threading.Thread(target=_log_session_output, daemon=True)
        t.start()

        atexit.register(lambda: proc.terminate())
        return proc

    except Exception as e:
        _warn(f"Session bridge failed to start: {e}")
        return None


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


def _vram_snap(label: str):
    """Quick VRAM snapshot for server diagnostics."""
    import torch
    for i in range(torch.cuda.device_count()):
        dev = torch.device(f"cuda:{i}")
        alloc = torch.cuda.memory_allocated(dev) >> 20
        reserved = torch.cuda.memory_reserved(dev) >> 20
        free, total = torch.cuda.mem_get_info(dev)
        free_mb, total_mb = free >> 20, total >> 20
        used_mb = total_mb - free_mb
        print(
            f"  \033[33m[VRAM {label}]\033[0m cuda:{i}: "
            f"alloc={alloc} MB, reserved={reserved} MB, "
            f"used={used_mb} MB, free={free_mb} MB",
            flush=True,
        )
        logger.info(
            "VRAM_SNAP [%s] cuda:%d: alloc=%d MB, reserved=%d MB, used=%d MB, free=%d MB, total=%d MB",
            label, i, alloc, reserved, used_mb, free_mb, total_mb,
        )


def _warmup_prefill(model: KrasisModel):
    """Run a 50K-token prefill to warm up GPU kernels, CUDA caches, and lazy allocations.

    Uses a large prompt to trigger peak VRAM usage (all layer group buffers,
    FlashInfer workspace, KV cache pages). This is called BEFORE HCS allocation
    so the VRAM monitor captures realistic peak usage.
    """
    import json as _json

    _vram_snap("before-prefill-warmup")
    logger.info("Warming up prefill (50K tokens, GPU kernels + CUDA caches)...")
    t0 = time.time()

    try:
        # Build a ~50K token prompt by repeating text
        base_text = (
            "Explain the architecture of modern mixture-of-experts models "
            "including their routing mechanisms, load balancing strategies, "
            "and computational efficiency trade-offs in detail. "
        )
        # Each repetition is ~30 tokens, repeat enough for ~50K
        warmup_text = base_text * 1700  # ~51K tokens
        messages_json = _json.dumps([{"role": "user", "content": warmup_text}])

        # Use GPU decode mode for prefill so it doesn't try to set up CPU decoder
        result = model.server_prefill(
            messages_json, max_new_tokens=5, temperature=0.6, top_k=50,
            top_p=0.95, presence_penalty=0.0,
            enable_thinking=False, extra_stop_tokens=[],
            decode_mode="gpu",
        )
        _vram_snap("after-prefill-warmup-before-cleanup")
        logger.info("Prefill warmup: %d tokens processed", result.prompt_len)
        model.server_cleanup()
        _vram_snap("after-prefill-warmup-after-cleanup")

        elapsed = time.time() - t0
        logger.info("Prefill warmup complete (%.1fs, %d tokens)", elapsed, result.prompt_len)
    except Exception as e:
        logger.warning("Prefill warmup failed (non-fatal): %s", e)
        try:
            model.server_cleanup()
        except Exception:
            pass


def _capture_prefill(model: KrasisModel, num_repeats: int):
    """Run a prefill and return (prompt_len, prefill_result), WITHOUT cleanup.
    Leaves the model in prefilled state so decode can follow.
    Returns (None, None) on failure.
    """
    import json as _json
    base_text = (
        "Explain the architecture of modern mixture-of-experts models "
        "including their routing mechanisms, load balancing strategies, "
        "and computational efficiency trade-offs in detail. "
    )
    warmup_text = base_text * num_repeats
    messages_json = _json.dumps([{"role": "user", "content": warmup_text}])
    try:
        result = model.server_prefill(
            messages_json, max_new_tokens=10, temperature=0.6, top_k=50,
            top_p=0.95, presence_penalty=0.0,
            enable_thinking=False, extra_stop_tokens=[],
            decode_mode="gpu",
        )
        return result.prompt_len, result
    except Exception as e:
        logger.warning("Capture prefill failed: %s", e)
        try:
            model.server_cleanup()
        except Exception:
            pass
        return None, None


def _capture_decode_only(model: KrasisModel, prefill_result, num_steps: int = 4):
    """Run decode steps using an existing prefill result, then cleanup.
    The prefill must already have been done (model is in prefilled state).
    """
    try:
        gpu_store = getattr(model, '_gpu_decode_store', None)
        if gpu_store is None:
            raise RuntimeError("GPU decode store not configured")
        if prefill_result.first_token not in prefill_result.stop_ids:
            gpu_store.gpu_generate_batch(
                first_token=prefill_result.first_token,
                start_position=prefill_result.prompt_len,
                max_tokens=num_steps,
                temperature=0.6,
                top_k=50,
                top_p=0.95,
                stop_ids=prefill_result.stop_ids,
                presence_penalty=0.0,
            )
        model.server_cleanup()
    except Exception as e:
        logger.warning("Capture decode failed: %s", e)
        try:
            model.server_cleanup()
        except Exception:
            pass


def _warmup_decode(model: KrasisModel, num_steps: int = 4):
    """Run a short GPU decode warmup.

    Validates that GPU decode (with or without HCS) works correctly.
    Uses Rust GpuDecodeStore — zero Python in the decode loop.
    """
    import json as _json

    logger.info("Warming up GPU decode (%d steps)...", num_steps)
    _vram_snap("before-decode-warmup")
    t0 = time.time()

    try:
        messages_json = _json.dumps([{"role": "user", "content": "Hi"}])
        result = model.server_prefill(
            messages_json, max_new_tokens=num_steps + 1, temperature=0.6, top_k=50,
            top_p=0.95, presence_penalty=0.0,
            enable_thinking=False, extra_stop_tokens=[],
            decode_mode="gpu",
        )
        _vram_snap("decode-warmup-after-prefill")
        if result.first_token not in result.stop_ids:
            gpu_store = getattr(model, '_gpu_decode_store', None)
            if gpu_store is None:
                raise RuntimeError("GPU decode store not configured for warmup")
            gpu_store.gpu_generate_batch(
                first_token=result.first_token,
                start_position=result.prompt_len,
                max_tokens=num_steps,
                temperature=0.6,
                top_k=50,
                top_p=0.95,
                stop_ids=result.stop_ids,
                presence_penalty=0.0,
            )
            _vram_snap("decode-warmup-after-gpu-decode")
        model.server_cleanup()
        _vram_snap("decode-warmup-after-cleanup")

        elapsed = time.time() - t0
        logger.info("Decode warmup complete (%.1fs)", elapsed)
    except Exception as e:
        logger.warning("Decode warmup failed (non-fatal): %s", e)
        try:
            model.server_cleanup()
        except Exception:
            pass


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
            "CFG_HCS": "hcs",
            "CFG_MULTI_GPU_HCS": "multi_gpu_hcs",
            "CFG_KV_CACHE_MB": "kv_cache_mb",
            "CFG_VRAM_SAFETY_MARGIN": "vram_safety_margin",
            "CFG_ENABLE_THINKING": "enable_thinking",
            "CFG_SESSION_ENABLED": "session_enabled",
            "CFG_NUM_GPUS": "num_gpus",
            "CFG_CPU_DECODE": None,  # CPU decode removed, ignore config key
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
                    if key in ("CFG_FORCE_LOAD", "CFG_ENABLE_THINKING", "CFG_SESSION_ENABLED", "CFG_HCS", "CFG_MULTI_GPU_HCS", "CFG_CPU_DECODE"):
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
    parser.add_argument("--kv-cache-mb", type=int, default=1000,
                        help="KV cache size in MB (default: 1000)")
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
    parser.add_argument("--hcs", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Hot Cache Strategy (default: on for GPU decode, use --no-hcs to disable)")
    parser.add_argument("--multi-gpu-hcs", action="store_true", default=False,
                        help="Pin HCS experts across ALL GPUs (more capacity, but cross-device transfer)")
    parser.add_argument("--hcs-headroom-mb", type=int, default=1024,
                        help="VRAM headroom to reserve after warmup before HCS allocation (default: 1024 MB)")
    parser.add_argument("--vram-safety-margin", type=int, default=1000,
                        help="VRAM safety margin in MB — reserved free VRAM below which warnings fire (default: 1000)")
    parser.add_argument("--stream-attention", action="store_true",
                        help="Stream attention weights from CPU instead of keeping resident on GPU. "
                             "Use when attention weights don't fit in VRAM (e.g. very large models).")
    parser.add_argument("--no-stream-attention", action="store_true",
                        help="(deprecated, now the default) Attention is resident on GPU by default.")
    parser.add_argument("--layer-group-size", type=int, default=2,
                        help="Number of MoE layers to load per group during prefill (default: 2)")
    # GPU decode is the only mode — CPU decode has been removed.
    # Keep --gpu-decode as a no-op for config file compatibility.
    parser.add_argument("--gpu-decode", action="store_true", default=True,
                        help="(default, only mode) GPU decode via Rust GpuDecodeStore.")
    parser.add_argument("--draft-model", default=None,
                        help="Path to draft model for speculative decoding (e.g. ~/.krasis/models/Qwen3-0.6B)")
    parser.add_argument("--draft-k", type=int, default=3,
                        help="Number of tokens to draft per speculative round (default: 3)")
    parser.add_argument("--draft-context", type=int, default=512,
                        help="Context window for draft model warmup (default: 512)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run standardized benchmark via HTTP (same path as production)")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Run benchmark via HTTP and exit (don't keep server running)")
    parser.add_argument("--timing", action="store_true",
                        help="Enable decode timing instrumentation (per-layer breakdown)")
    parser.add_argument("--stress-test", action="store_true",
                        help="Run stress test (diverse prompts) and exit")
    parser.add_argument("--perplexity", action="store_true",
                        help="Run perplexity evaluation and exit")
    parser.add_argument("--note", default=None,
                        help="Description note written to the top of the log file for this run")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Enable thinking/reasoning mode (default: on)")
    parser.add_argument("--session-enabled", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Enable Session messenger bridge (default: off)")
    # Apply config file defaults, then parse CLI (CLI wins over config file)
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args(remaining_argv)

    log_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    # Archive previous krasis.log into logs/ with timestamp before overwriting
    _log_file = os.path.join(os.getcwd(), "krasis.log")
    _logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(_logs_dir, exist_ok=True)
    if os.path.isfile(_log_file) and os.path.getsize(_log_file) > 0:
        from datetime import datetime
        _mtime = os.path.getmtime(_log_file)
        _ts = datetime.fromtimestamp(_mtime).strftime("%Y%m%d_%H%M%S")
        _archive_name = f"krasis_{_ts}.log"
        _archive_path = os.path.join(_logs_dir, _archive_name)
        # Avoid overwriting an existing archive (e.g. rapid restarts)
        _counter = 1
        while os.path.exists(_archive_path):
            _archive_path = os.path.join(_logs_dir, f"krasis_{_ts}_{_counter}.log")
            _counter += 1
        import shutil
        shutil.move(_log_file, _archive_path)
        print(f"Archived previous log → logs/{os.path.basename(_archive_path)}")

    _file_handler = logging.FileHandler(_log_file, mode="w")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(_file_handler)

    # Write run note to top of log file if provided
    if args.note:
        with open(_log_file, "w") as _nf:
            _nf.write(f"=== RUN NOTE: {args.note} ===\n\n")
        # Re-open handler in append mode so logging doesn't overwrite the note
        logging.getLogger().removeHandler(_file_handler)
        _file_handler = logging.FileHandler(_log_file, mode="a")
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(_file_handler)

    # Capture uncaught exceptions to the log file (main thread)
    _original_excepthook = sys.excepthook
    def _log_excepthook(exc_type, exc_value, exc_tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        _original_excepthook(exc_type, exc_value, exc_tb)
    sys.excepthook = _log_excepthook

    # Capture thread exceptions to the log file too
    def _log_threading_excepthook(args):
        logger.critical("Exception in thread %s", args.thread.name if args.thread else "?",
                        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
    threading.excepthook = _log_threading_excepthook

    # Redirect stderr to log file so any stray prints/errors are captured
    class _StderrLogger:
        def __init__(self, original, log):
            self._original = original
            self._log = log
        def write(self, msg):
            if msg and msg.strip():
                self._log.error("[stderr] %s", msg.rstrip())
            if self._original:
                self._original.write(msg)
        def flush(self):
            if self._original:
                self._original.flush()
        def fileno(self):
            return self._original.fileno() if self._original else 2
    sys.stderr = _StderrLogger(sys.stderr, logger)

    logger.info("Logging to %s", _log_file)

    # Dump config to log for easier debugging
    if pre_args.config:
        logger.info("=== Config file: %s ===", pre_args.config)
        try:
            with open(pre_args.config) as _cf:
                for _line in _cf:
                    logger.info("  %s", _line.rstrip())
        except Exception as _e:
            logger.warning("Could not read config file: %s", _e)
    else:
        logger.info("No config file — using CLI args / defaults")
    logger.info("=== Resolved arguments ===")
    for _k, _v in sorted(vars(args).items()):
        logger.info("  %s = %r", _k, _v)

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

    # GPU decode is the only mode — skip CPU expert weights + CPU decoder
    gpu_only = True

    # ── Configuration summary ──
    _status(f"Krasis — {_model_name}")
    _detail(f"Decode: GPU  |  HCS: {'on' if args.hcs else 'off'}  |  GPUs: {num_gpus_available}")
    _detail(f"Experts: GPU INT{args.gpu_expert_bits}  |  Attention: {args.attention_quant}  |  KV: {args.kv_dtype}")
    _detail(f"Layer groups: {args.layer_group_size}  |  KV cache: {args.kv_cache_mb} MB  |  Threads: {args.krasis_threads}")
    _dim("GPU-only mode: CPU expert weights and CPU decoder skipped")

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
        gpu_prefill_threshold=1 if args.hcs else getattr(args, 'gpu_prefill_threshold', int(os.environ.get("KRASIS_PREFILL_THRESHOLD", "500"))),
        kv_cache_mb=args.kv_cache_mb,
        stream_attention=args.stream_attention,
    )

    _status("Loading model weights")
    _model.load(gpu_only=gpu_only)

    # Resolve heatmap: cached > build
    cache_dir = cache_dir_for_model(args.model_path)
    heatmap_path = args.heatmap_path
    if not heatmap_path:
        heatmap_path = os.path.join(cache_dir, "auto_heatmap.json")

    # CUDA runtime warmup — triggers cuBLAS + Triton kernel compilation
    # before any VRAM measurements.
    num_gpus_available = args.num_gpus or torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus_available)]
    device_indices = list(range(num_gpus_available))
    _status("CUDA runtime warmup")
    _model.warmup_cuda_runtime(devices)
    _detail("cuBLAS + Triton kernel compilation done")

    # ── Set decode mode (GPU only) ──
    _model.decode_mode = "gpu"

    # ── GPU decode store setup (before warmup so decode warmup can use it) ──
    _status("Setting up GPU decode store")
    gpu_store = _model.setup_gpu_decode_store()
    gpu_store_addr = gpu_store.gpu_store_addr()
    _detail(f"GPU decode store ready (addr={gpu_store_addr:#x})")

    # ── Load draft model BEFORE warmup/VRAM capture so HCS budget accounts for it ──
    if args.draft_model:
        import os
        draft_path = os.path.expanduser(args.draft_model)
        _status(f"Loading draft model from {draft_path}")
        gpu_store.load_draft_model(
            draft_path,
            max_seq=4096,
            draft_k=args.draft_k,
            context_window=args.draft_context,
        )
        _detail(f"Draft model loaded (k={args.draft_k}, context={args.draft_context})")

    # ── Start VRAM monitor before warmup for visibility ──
    from krasis import VramMonitor
    SAFETY_MARGIN_MB = args.vram_safety_margin
    vram_monitor = VramMonitor(device_indices, poll_interval_ms=50, safety_margin_mb=SAFETY_MARGIN_MB)
    vram_monitor.start()
    _dim("VRAM monitor started (tracking warmup)")
    for idx in device_indices:
        total = vram_monitor.total_mb(idx)
        _dim(f"cuda:{idx}: {total:,} MB total")

    # ── Phase 1: Warmup (trigger all lazy CUDA allocations) ──
    # torch.compile, KV cache, FlashInfer workspace, cuBLAS handles, decode buffers.
    # These cause a transient VRAM spike that is freed afterwards. The monitor
    # captures the spike for visibility but is reset before HCS budget measurement.
    _model._hcs_device = None
    _model._multi_gpu_hcs = False
    _status("Warmup (prefill + decode, no HCS)")
    _dim("Triggering lazy CUDA allocations (torch.compile, FlashInfer, cuBLAS)")
    t_warmup = time.time()
    _warmup_prefill(_model)
    _warmup_decode(_model, num_steps=1)
    warmup_elapsed = time.time() - t_warmup
    _detail(f"Warmup complete in {warmup_elapsed:.1f}s")

    # Log warmup VRAM impact before resetting
    for idx in device_indices:
        warmup_min_free = vram_monitor.min_free_mb(idx)
        warmup_peak_used = vram_monitor.peak_used_mb(idx)
        total = vram_monitor.total_mb(idx)
        _dim(f"cuda:{idx} warmup:  peak {warmup_peak_used:,} MB used / {total:,} MB total  (min free: {warmup_min_free:,} MB)")
        logger.info(
            "VRAM warmup cuda:%d: peak_used=%d MB, min_free=%d MB, total=%d MB",
            idx, warmup_peak_used, warmup_min_free, total,
        )

    # ── Phase 2: Four-point VRAM calibration ──
    # Measure min_free VRAM during: short prefill, short decode, long prefill, long decode.
    # This gives us two linear models (prefill VRAM/token, decode VRAM/token) for
    # proportional soft-tier HCS eviction at runtime.
    # Do NOT gc.collect/empty_cache here — persistent allocations stay resident.
    dev_idx = devices[0].index
    import torch

    _status("VRAM calibration (4-point measurement)")
    _detail(f"Safety margin: {SAFETY_MARGIN_MB:,} MB")

    # ── 2a: Short prompt prefill ──
    SHORT_REPEATS = 17   # ~500 tokens
    LONG_REPEATS = 1700  # ~51K tokens

    # ── 2a: Short prompt — prefill then decode (measured separately) ──
    vram_monitor.reset_min_free()
    _dim("Capture: short prompt prefill (~500 tokens)")
    short_prompt_len, short_result = _capture_prefill(_model, SHORT_REPEATS)
    torch.cuda.synchronize()
    time.sleep(0.1)  # let monitor poll
    prefill_short_free = vram_monitor.min_free_mb(dev_idx)
    short_tokens = short_prompt_len or 500
    logger.info("VRAM capture: short prefill: %d tokens, min_free=%d MB", short_tokens, prefill_short_free)
    _dim(f"  short prefill: {short_tokens} tokens, min_free={prefill_short_free:,} MB")

    # Now measure decode-only (prefill temporaries already freed, model in prefilled state)
    vram_monitor.reset_min_free()
    _dim("Capture: short prompt decode (decode-only, no prefill)")
    if short_result is not None:
        _capture_decode_only(_model, short_result, num_steps=8)
    else:
        _model.server_cleanup()
    torch.cuda.synchronize()
    time.sleep(0.1)
    decode_short_free = vram_monitor.min_free_mb(dev_idx)
    logger.info("VRAM capture: short decode-only: min_free=%d MB", decode_short_free)
    _dim(f"  short decode: min_free={decode_short_free:,} MB")

    # ── 2b: Long prompt — prefill then decode (measured separately) ──
    vram_monitor.reset_min_free()
    _dim("Capture: long prompt prefill (~50K tokens)")
    long_prompt_len, long_result = _capture_prefill(_model, LONG_REPEATS)
    torch.cuda.synchronize()
    time.sleep(0.1)
    prefill_long_free = vram_monitor.min_free_mb(dev_idx)
    long_tokens = long_prompt_len or 51000
    logger.info("VRAM capture: long prefill: %d tokens, min_free=%d MB", long_tokens, prefill_long_free)
    _dim(f"  long prefill: {long_tokens} tokens, min_free={prefill_long_free:,} MB")

    # Now measure decode-only after long prefill
    vram_monitor.reset_min_free()
    _dim("Capture: long prompt decode (decode-only, no prefill)")
    if long_result is not None:
        _capture_decode_only(_model, long_result, num_steps=8)
    else:
        _model.server_cleanup()
    torch.cuda.synchronize()
    time.sleep(0.1)
    decode_long_free = vram_monitor.min_free_mb(dev_idx)
    logger.info("VRAM capture: long decode-only: min_free=%d MB", decode_long_free)

    # ── Compute calibration ──
    if long_tokens > short_tokens:
        prefill_kb_per_tok = (prefill_short_free - prefill_long_free) / (long_tokens - short_tokens) * 1024
        decode_kb_per_tok = (decode_short_free - decode_long_free) / (long_tokens - short_tokens) * 1024
    else:
        prefill_kb_per_tok = 0
        decode_kb_per_tok = 0

    # Use the minimum of short/long prefill free — the first prefill allocates
    # one-time workspace (FlashInfer, torch.compile, fused_marlin_moe buffers)
    # which makes short_free < long_free. HCS must survive the worst case.
    prefill_min_free = min(prefill_short_free, prefill_long_free)
    hard_budget = max(0, int(prefill_min_free) - SAFETY_MARGIN_MB)
    decode_budget = max(0, int(decode_short_free) - SAFETY_MARGIN_MB)
    soft_budget = max(0, decode_budget - hard_budget)

    _status("VRAM calibration complete")
    _detail(f"Prefill: {prefill_kb_per_tok:.1f} KB/token ({prefill_short_free:,} MB @ {short_tokens} tok → {prefill_long_free:,} MB @ {long_tokens} tok)")
    _detail(f"Decode:  {decode_kb_per_tok:.1f} KB/token ({decode_short_free:,} MB @ short → {decode_long_free:,} MB @ long)")
    _detail(f"Hard HCS budget: {hard_budget:,} MB  |  Soft HCS budget: {soft_budget:,} MB  |  Total: {hard_budget + soft_budget:,} MB")
    logger.info("VRAM budget: hard=%d MB (prefill_min_free=%d - safety=%d), "
                "decode_budget=%d MB (decode_short_free=%d - safety=%d), "
                "soft=%d MB (decode_budget - hard)",
                hard_budget, prefill_min_free, SAFETY_MARGIN_MB,
                decode_budget, decode_short_free, SAFETY_MARGIN_MB,
                soft_budget)

    # Free PyTorch's cached CUDA blocks before HCS allocation.
    # After calibration, PyTorch holds freed KV/expert blocks in its caching allocator.
    # These are "allocated" from CUDA's perspective, reducing cudaMemGetInfo free.
    # HCS uses cuMemAlloc (not PyTorch), so it needs CUDA-level free memory.
    gc.collect()
    torch.cuda.empty_cache()

    if not args.hcs:
        _status("GPU decode (no HCS)")
        _warn("All experts streamed via DMA per token (slow for decode)")
    else:
        _status("Calculating HCS budget")

        # ── Device selection ──
        primary_dev = devices[0]
        total_experts = cfg.n_routed_experts * cfg.num_moe_layers

        # ── Load heatmap ──
        if not os.path.exists(heatmap_path):
            _status("Building expert heatmap (calibration)")
            heatmap_path = _build_heatmap(_model, heatmap_path)
        else:
            _dim(f"Using cached heatmap: {os.path.basename(heatmap_path)}")

        # ── Load heatmap and build sorted ranking ──
        with open(heatmap_path) as f:
            raw_heatmap = json.load(f)
        sorted_ranking = sorted(raw_heatmap.items(), key=lambda x: x[1], reverse=True)
        ranking = [(int(k.split(",")[0]), int(k.split(",")[1])) for k, _ in sorted_ranking]
        _detail(f"Heatmap: {len(ranking):,} experts ranked from {len(raw_heatmap):,} entries")

        # ── Pass calibration data to Rust ──
        if hasattr(_model, '_gpu_decode_store'):
            store = _model._gpu_decode_store

            cal_msg = store.set_vram_calibration(
                short_tokens, long_tokens,
                prefill_short_free, prefill_long_free,
                decode_short_free, decode_long_free,
                SAFETY_MARGIN_MB,
            )
            _dim(cal_msg)

            # ── Initialize tiered HCS: hard pool + soft pool ──
            t_hcs = time.time()
            _status("Loading HCS pool (hard + soft tier)")

            result = store.hcs_pool_init_tiered(
                ranking,
                hard_budget_mb=hard_budget,
                soft_budget_mb=soft_budget,
                window_size=10,
                replacement_pct=25,
            )
            hcs_elapsed = time.time() - t_hcs

            _status("HCS pool loaded")
            _detail(result)
            _dim(f"Loaded in {hcs_elapsed:.1f}s")
            logger.info("HCS pool: %s (%.1fs)", result, hcs_elapsed)

    # ── Decode validation (after HCS) ──
    # Must evict soft HCS before running prefill (layer groups need VRAM).
    _status("Validating decode" + (" with HCS" if args.hcs else ""))
    if hasattr(_model, '_gpu_decode_store'):
        store = _model._gpu_decode_store
        evicted, freed = store.py_hcs_evict_for_prefill(500)  # short warmup prompt
        if evicted > 0:
            _dim(f"Evicted {evicted} soft experts ({freed:.0f} MB) for warmup prefill")
    _warmup_decode(_model, num_steps=4)
    if hasattr(_model, '_gpu_decode_store'):
        store = _model._gpu_decode_store
        reloaded, reload_ms = store.py_hcs_reload_after_prefill()
        if reloaded > 0:
            _dim(f"Reloaded {reloaded} soft experts in {reload_ms:.0f}ms")
    _detail("Decode validation passed")

    # ── Enable VRAM monitor runtime warnings ──
    # enable_warnings() resets min-free tracking so the first poll captures
    # the post-HCS state. If free VRAM is already below the safety margin
    # (i.e. HCS was too aggressive), we get an immediate warning.
    # During runtime, every new low below the margin triggers another warning.
    _status("VRAM monitor: runtime warnings enabled")
    _detail(f"Safety margin: {SAFETY_MARGIN_MB:,} MB — warnings on every new low below this")
    vram_monitor.enable_warnings()
    logger.info("VRAM monitor: runtime warnings enabled (safety margin: %d MB)", SAFETY_MARGIN_MB)

    # Benchmark runs AFTER server starts (same HTTP path as production).
    # We set up the benchmark thread here; it launches after rust_server.run().
    _benchmark_requested = args.benchmark or args.benchmark_only
    _benchmark_only = args.benchmark_only

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
    _detail(f"Decode: GPU  |  HCS: {'on' if args.hcs else 'off'}  |  Max context: {max_ctx:,} tokens")
    _dim(f"KV cache: {args.kv_cache_mb:,} MB")

    # ── Session messenger bridge ──
    _session_proc = None
    if getattr(args, 'session_enabled', False):
        _session_proc = _start_session_bridge(args.host, args.port)

    _dim("Press Q or Ctrl-C to stop")
    logger.info(
        "Model loaded, starting server on %s:%d (max context: %d, decode: GPU)",
        args.host, args.port, max_ctx,
    )

    # ── Server registry: write entry + register cleanup ──
    _write_registry(args.host, args.port, _model_name)
    atexit.register(_remove_registry)

    # ── Rust HTTP server ──
    from krasis import RustServer

    tokenizer_path = os.path.join(args.model_path, "tokenizer.json")
    rust_server = RustServer(
        _model,
        args.host,
        args.port,
        _model_name,
        tokenizer_path,
        max_ctx,
        args.enable_thinking,
        gpu_store_addr,
    )

    def _handle_exit(sig, frame):
        if _session_proc and _session_proc.poll() is None:
            _session_proc.terminate()
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

    # Benchmark thread: engine benchmarks run immediately (no HTTP needed),
    # HTTP TTFT benchmark waits for server to be ready internally.
    if _benchmark_requested:
        def _run_benchmark():
            from krasis.benchmark import KrasisBenchmark
            bench = KrasisBenchmark(
                _model, rust_server=rust_server,
                host=args.host, port=args.port, timing=args.timing,
            )
            bench.run()

            if _benchmark_only:
                rust_server.stop()
            else:
                _status(f"Server ready on {args.host}:{args.port}")
                # Show Session ID again so it's visible after benchmark output
                if getattr(args, 'session_enabled', False):
                    sid_path = Path.home() / ".krasis" / "session_id"
                    if sid_path.is_file():
                        sid = sid_path.read_text().strip()
                        if sid:
                            _detail("Session messenger: ON")
                            print(f"  {_BOLD}Session ID: {sid}{_NC}", flush=True)
                _dim("Press Q or Ctrl-C to stop")

        bench_thread = threading.Thread(target=_run_benchmark, daemon=True)
        bench_thread.start()

    # run() releases the GIL and blocks until stop() is called
    rust_server.run()

    # ── Clean exit before Python teardown triggers cascading errors ──
    vram_monitor.stop()
    _remove_registry()
    _cleanup_cuda()
    os._exit(0)


if __name__ == "__main__":
    main()
