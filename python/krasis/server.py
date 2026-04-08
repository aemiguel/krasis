"""Krasis LLM server — Rust HTTP server with Rust prefill/decode.

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

from krasis.run_paths import get_run_dir

# Pre-scan config file for CFG_SELECTED_GPUS to set CUDA_VISIBLE_DEVICES
# BEFORE any torch/CUDA imports, since CUDA init happens at import time.
def _prescan_selected_gpus():
    import argparse as _ap
    _pre = _ap.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=None)
    _pre_args, _ = _pre.parse_known_args()
    if _pre_args.config and os.path.isfile(_pre_args.config):
        with open(_pre_args.config) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line.startswith("CFG_SELECTED_GPUS="):
                    _val = _line.split("=", 1)[1].strip().strip('"').strip("'")
                    _gpus = [x.strip() for x in _val.split(",") if x.strip()]
                    if _gpus:
                        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(_gpus)
                        print(f"Pre-scan: set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
                    break
_prescan_selected_gpus()

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

STARTUP_CALIBRATION_SHORT_TOKENS = int(os.environ.get("KRASIS_STARTUP_CALIBRATION_SHORT_TOKENS", "500"))
STARTUP_CALIBRATION_DECODE_TOKENS = int(os.environ.get("KRASIS_STARTUP_CALIBRATION_DECODE_TOKENS", "32"))
STARTUP_CALIBRATION_LONG_TOKENS_CAP = int(os.environ.get("KRASIS_STARTUP_CALIBRATION_LONG_TOKENS_CAP", "50000"))


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        _warn(f"Ignoring invalid {name}={raw!r}; using {default}")
        return default


def _startup_diag_enabled() -> bool:
    return os.environ.get("KRASIS_STARTUP_DIAG", "") == "1"


def _env_flag(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip() not in ("", "0", "false", "False")

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


def _validate_heatmap(heatmap_path: str, cfg) -> bool:
    """Check if a cached heatmap matches the current model config.

    Returns True if valid, False if it should be rebuilt.
    Only triggers rebuild on structural mismatches (wrong layer count or
    expert count), NOT on missing metadata or sparse coverage.
    """
    try:
        with open(heatmap_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Heatmap file corrupt or unreadable, will rebuild")
        return False

    meta = data.get("_metadata")
    if meta:
        # New format with embedded metadata -- exact check
        expected_moe = cfg.num_moe_layers
        expected_experts = cfg.n_routed_experts
        if meta.get("num_moe_layers") != expected_moe:
            logger.warning(
                "Heatmap stale: num_moe_layers=%s but model has %d, will rebuild",
                meta.get("num_moe_layers"), expected_moe,
            )
            return False
        if meta.get("n_routed_experts") != expected_experts:
            logger.warning(
                "Heatmap stale: n_routed_experts=%s but model has %d, will rebuild",
                meta.get("n_routed_experts"), expected_experts,
            )
            return False
        return True

    # Old format (no metadata) -- infer from data keys
    layers_seen = set()
    max_expert = -1
    for key in data:
        if key.startswith("_"):
            continue
        parts = key.split(",")
        if len(parts) == 2:
            layers_seen.add(int(parts[0]))
            max_expert = max(max_expert, int(parts[1]))

    expected_moe = cfg.num_moe_layers
    if len(layers_seen) < expected_moe:
        # Some layers have zero activations in a 10K sample -- only flag if
        # a significant number are missing (>20% of layers absent is suspicious)
        missing_pct = (expected_moe - len(layers_seen)) / expected_moe
        if missing_pct > 0.2:
            logger.warning(
                "Heatmap covers %d/%d MoE layers (%.0f%% missing), will rebuild",
                len(layers_seen), expected_moe, missing_pct * 100,
            )
            return False

    if max_expert >= cfg.n_routed_experts:
        logger.warning(
            "Heatmap references expert %d but model only has %d experts, will rebuild",
            max_expert, cfg.n_routed_experts,
        )
        return False

    return True


def _load_heatmap_prompts() -> list[str]:
    """Load heatmap calibration prompts from the prompts directory.

    Returns a list of prompt strings.  Users can edit heatmap_prompts.txt to
    match their typical workload.
    """
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    path = os.path.join(prompts_dir, "heatmap_prompts.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Heatmap prompt file not found: {path}\n"
            "This file is required for HCS calibration.  "
            "See python/krasis/prompts/heatmap_prompts.txt in the repo."
        )
    prompts = []
    current = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped == "" and current:
                prompts.append(" ".join(current))
                current = []
            elif stripped:
                current.append(stripped)
    if current:
        prompts.append(" ".join(current))
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _load_prompt_file(filename: str) -> str:
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    path = os.path.join(prompts_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path) as f:
        return f.read().strip()


def _discover_prefill_prompt_files() -> list[str]:
    files = []
    for i in range(1, 100):
        filename = f"prefill_prompt_{i}"
        prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        if os.path.isfile(os.path.join(prompts_dir, filename)):
            files.append(filename)
        else:
            break
    if not files:
        raise FileNotFoundError(
            "No prefill prompt files found. Expected prefill_prompt_1, prefill_prompt_2, etc."
        )
    return files


def _truncate_content_to_prompt_tokens(model: KrasisModel, content: str, max_tokens: int) -> tuple[str, list[int]]:
    tokens = _chat_prompt_tokens(model, content)
    if len(tokens) <= max_tokens:
        return content, tokens

    lo, hi = 0, len(content)
    best_content = ""
    best_tokens: list[int] = []
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = content[:mid]
        trial_tokens = _chat_prompt_tokens(model, trial)
        if len(trial_tokens) <= max_tokens:
            best_content = trial
            best_tokens = trial_tokens
            lo = mid + 1
        else:
            hi = mid - 1
    return best_content, best_tokens


def _kv_cache_max_tokens(model: KrasisModel) -> int:
    for cache in getattr(model, "kv_caches", []):
        if cache is not None:
            return cache.max_pages * cache.page_size
    return model.cfg.max_position_embeddings


def _make_startup_calibration_prompts(model: KrasisModel, lengths: list[int]) -> list[list[int]]:
    files = _discover_prefill_prompt_files()
    kv_limit = max(1, min(_kv_cache_max_tokens(model), model.cfg.max_position_embeddings) - 100)
    prompts: list[list[int]] = []
    for i, target in enumerate(lengths):
        content = _load_prompt_file(files[i % len(files)])
        _, tokens = _truncate_content_to_prompt_tokens(model, content, min(target, kv_limit))
        if not tokens:
            raise RuntimeError(f"Startup calibration prompt {files[i % len(files)]} produced no tokens")
        prompts.append(tokens)
    return prompts


# Decode tokens per heatmap prompt.  This is deliberately high relative to the
# short prompt length so that the heatmap is dominated by decode routing, which
# is where HCS cache effectiveness matters.
HEATMAP_DECODE_TOKENS = 256


def _chat_prompt_tokens(model: KrasisModel, prompt_text: str) -> list[int]:
    return model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
    )


def _default_stop_ids(model: KrasisModel) -> list[int]:
    return [model.cfg.eos_token_id] + list(model.cfg.extra_stop_token_ids)


def _build_heatmap(model: KrasisModel, save_path: str) -> str:
    """Build expert activation heatmap by running decode-heavy inference.

    Loads diverse short prompts from heatmap_prompts.txt, runs each with a
    long decode window (256 tokens), so the resulting heatmap reflects decode
    expert routing rather than prefill routing.  This runs on every startup
    to keep the heatmap current with the model and reference data.

    Uses the Rust decode engine's built-in heatmap collection — no
    GpuPrefillManager required.
    """
    import os, json

    prompts = _load_heatmap_prompts()

    gpu_store = getattr(model, '_gpu_decode_store', None)
    if gpu_store is None:
        raise RuntimeError("GPU decode store not configured — cannot build heatmap")

    cfg = model.cfg
    num_layers = cfg.num_hidden_layers
    num_experts = cfg.n_routed_experts

    # Init lightweight HCS state for collection only (no VRAM allocation)
    gpu_store.hcs_init_collection(num_layers, num_experts)
    gpu_store.hcs_start_collecting()

    # Run each prompt with long decode to build a decode-weighted heatmap
    total_decode_tokens = 0
    logger.info("Building heatmap from %d prompts (%d decode tokens each)...",
                len(prompts), HEATMAP_DECODE_TOKENS)
    stop_ids = _default_stop_ids(model)
    for i, prompt_text in enumerate(prompts):
        tokens = _chat_prompt_tokens(model, prompt_text)
        logger.info("  Heatmap prompt %d/%d: %d prefill tokens + %d decode tokens",
                    i + 1, len(prompts), len(tokens), HEATMAP_DECODE_TOKENS)
        first_token, prompt_len, kv_overflow = gpu_store.rust_prefill_tokens(tokens, temperature=0.6)
        if not kv_overflow and first_token not in stop_ids:
            generated = gpu_store.gpu_generate_batch(
                first_token=first_token,
                start_position=prompt_len,
                max_tokens=HEATMAP_DECODE_TOKENS,
                temperature=0.6,
                top_k=50,
                top_p=0.95,
                stop_ids=stop_ids,
                presence_penalty=0.0,
            )
            total_decode_tokens += 1 + len(generated)
        else:
            total_decode_tokens += 1

    logger.info("Heatmap collection complete: %d decode tokens across %d prompts",
                total_decode_tokens, len(prompts))

    # Export and save heatmap
    heatmap_dict = gpu_store.hcs_export_heatmap()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(heatmap_dict, f)
    logger.info("Heatmap saved to %s (%d entries)", save_path, len(heatmap_dict))

    # Tear down collection-only HCS so normal startup can re-init with real budget
    gpu_store.hcs_reset()

    return save_path


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

    # Note: WSL2 LD_LIBRARY_PATH fix for /usr/lib/wsl/lib is in launcher.py's
    # launch_server() — must be set BEFORE execvp because glibc caches
    # LD_LIBRARY_PATH at process startup (too late to set here for dlopen).

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
            "CFG_FORCE_REBUILD_CACHE": "force_rebuild_cache",
            "CFG_BUILD_CACHE": "build_cache",
            "CFG_HCS": "hcs",
            "CFG_MULTI_GPU_HCS": "multi_gpu_hcs",
            "CFG_KV_CACHE_MB": "kv_cache_mb",
            "CFG_VRAM_SAFETY_MARGIN": "vram_safety_margin",
            "CFG_ENABLE_THINKING": "enable_thinking",
            "CFG_SESSION_ENABLED": "session_enabled",
            "CFG_NUM_GPUS": "num_gpus",
            "CFG_CPU_DECODE": None,  # CPU decode removed, ignore config key
            "CFG_ATTN_SKIP_AFTER": "attn_skip_after",
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
                        # CUDA_VISIBLE_DEVICES is set earlier in _prescan_selected_gpus()
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
        if config_defaults.get("attention_quant") in ("int4", "int8"):
            raise ValueError(
                f"Unsupported attention_quant={config_defaults['attention_quant']} in {config_path}. "
                "Naive int4/int8 attention has been removed; use 'awq' or 'bf16'."
            )
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
    parser.add_argument("--kv-dtype", default="polar4",
                        choices=["polar4", "fp8_e4m3", "bf16"],
                        help="KV cache format: polar4 (4-bit PolarQuant, default), fp8_e4m3 (8-bit), bf16 (uncompressed)")
    parser.add_argument("--kv-cache-mb", type=int, default=1000,
                        help="KV cache size in MB (default: 1000)")
    parser.add_argument("--heatmap-path", default=None,
                        help="Path to expert_heatmap.json for HCS init")
    parser.add_argument("--gpu-expert-bits", type=int, default=4, choices=[4, 8],
                        help="Marlin quantization bits for GPU prefill experts")
    parser.add_argument("--cpu-expert-bits", type=int, default=4, choices=[4, 8],
                        help="Quantization bits for CPU decode experts")
    parser.add_argument("--attention-quant", default="bf16", choices=["bf16", "awq"],
                        help="Attention weight precision: bf16 (default), awq (calibrated per-tensor via AWQ)")
    parser.add_argument("--shared-expert-quant", default="int8", choices=["bf16", "int8"],
                        help="Quantization for shared expert weights")
    parser.add_argument("--dense-mlp-quant", default="int8", choices=["bf16", "int8"],
                        help="Quantization for dense MLP weights")
    parser.add_argument("--lm-head-quant", default="int8", choices=["bf16", "int8"],
                        help="Quantization for lm_head weights")
    parser.add_argument("--gguf-path", default=None,
                        help="Path to GGUF file for CPU experts")
    parser.add_argument("--force-load", action="store_true",
                        help="Override RAM safety checks and load anyway")
    parser.add_argument("--force-rebuild-cache", action="store_true",
                        help="Delete existing expert caches and rebuild from safetensors")
    parser.add_argument("--build-cache", action="store_true",
                        help="Build expert caches (if missing) and exit without starting server")
    parser.add_argument("--hcs", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Hot Cache Strategy (default: on for GPU decode, use --no-hcs to disable)")
    parser.add_argument("--multi-gpu-hcs", action="store_true", default=False,
                        help="Pin HCS experts across ALL GPUs (more capacity, but cross-device transfer)")
    # NOTE: --hcs-headroom-mb removed — HCS budget is computed from 4-point VRAM calibration, not a fixed headroom
    parser.add_argument("--vram-safety-margin", type=int, default=600,
                        help="VRAM safety margin in MB — reserved free VRAM for decode kernel intermediates "
                             "and CUDA allocator headroom (default: 600 MB)")
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
    parser.add_argument("--vram-report", action="store_true",
                        help="Generate VRAM report CSV (periodic readings + events) in the current run directory")
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
    parser.add_argument("--test-endpoints", action="store_true", default=False,
                        help="Enable test-only endpoints (/v1/internal/prefill_logits)")
    parser.add_argument("--wc-alloc", action="store_true", default=False,
                        help="Use WriteCombined host memory for expert DMA staging. "
                             "Bypasses CPU cache for ~64%% higher PCIe bandwidth "
                             "(~46 GB/s vs ~28 GB/s on Gen5). Requires NVMe swap on "
                             "RAM-constrained systems.")
    # Apply config file defaults, then parse CLI (CLI wins over config file)
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args(remaining_argv)
    _default_run_type = "server-run"
    if args.benchmark_only:
        _default_run_type = "server-benchmark"
    elif args.benchmark:
        _default_run_type = "server-run-benchmark"
    elif getattr(args, "stress_test", False):
        _default_run_type = "server-stress"
    _run_dir = get_run_dir(_default_run_type)

    log_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    _log_file = os.path.join(_run_dir, "krasis.log")

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

    kv_format_str = args.kv_dtype  # "fp8_e4m3", "bf16", or "polar4"
    if args.kv_dtype == "fp8_e4m3":
        kv_dtype = torch.float8_e4m3fn
    elif args.kv_dtype == "polar4":
        kv_dtype = torch.float8_e4m3fn  # base dtype for size calc; polar4 allocates its own tensors
    else:
        kv_dtype = torch.bfloat16

    quant_cfg = QuantConfig(
        lm_head=args.lm_head_quant,
        attention=args.attention_quant,
        shared_expert=args.shared_expert_quant,
        dense_mlp=args.dense_mlp_quant,
        gpu_expert_bits=args.gpu_expert_bits,
        cpu_expert_bits=args.cpu_expert_bits,
        kv_cache_format=args.kv_dtype,
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

    # GPU decode is the only supported serving mode here. Do not load CPU expert
    # weights or enable any CPU-side fallback path.
    gpu_only = True

    # ── Configuration summary ──
    _status(f"Krasis — {_model_name}")
    _detail(f"Decode: GPU  |  HCS: {'on' if args.hcs else 'off'}  |  GPUs: {num_gpus_available}")
    _detail(f"Experts: GPU INT{args.gpu_expert_bits}  |  Attention: {args.attention_quant}  |  KV: {args.kv_dtype}")
    _detail(f"Layer groups: {args.layer_group_size}  |  KV cache: {args.kv_cache_mb} MB  |  Threads: {args.krasis_threads}")
    _dim("GPU-only mode: CPU expert weights and CPU decoder skipped")

    # ── Force rebuild: delete existing expert caches before loading ──
    if getattr(args, 'force_rebuild_cache', False):
        _cache_dir = cache_dir_for_model(args.model_path)
        _deleted = []
        for pattern in ["experts_marlin_*.bin"]:
            import glob as _glob
            for f in _glob.glob(os.path.join(_cache_dir, pattern)):
                os.unlink(f)
                _deleted.append(os.path.basename(f))
        if _deleted:
            _status(f"Deleted {len(_deleted)} cache files: {', '.join(_deleted)}")
        else:
            _detail("No existing cache files to delete")

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

    # Set attention skip layer if configured
    attn_skip = getattr(args, 'attn_skip_after', None)
    if attn_skip is not None and attn_skip != '':
        from krasis.layer import TransformerLayer
        TransformerLayer.attn_skip_after = int(attn_skip)
        _detail(f"Attention skip: layers >= {attn_skip} will skip attention")

    _status("Loading model weights")
    if getattr(args, 'build_cache', False):
        # --build-cache: build GPU Marlin expert cache then exit (CPU cache no longer used)
        _detail("Build-cache mode: building/verifying GPU Marlin expert cache")
        _model.load(gpu_only=True)
        import glob as _glob
        _cache_dir = cache_dir_for_model(args.model_path)
        gpu_bits = args.gpu_expert_bits
        has_gpu = bool(_glob.glob(os.path.join(_cache_dir, f"experts_marlin_int{gpu_bits}_g*.bin")))
        _status("Cache build complete")
        _detail(f"GPU Marlin INT{gpu_bits}: {'exists' if has_gpu else 'MISSING'}")
        print("BUILD CACHE COMPLETE", flush=True)
        return
    _model.load(gpu_only=gpu_only)

    # Resolve heatmap save path (rebuilt fresh on every startup unless --heatmap-path)
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

    # ── WriteCombined DMA staging ──
    if getattr(args, 'wc_alloc', False):
        _model.wc_alloc = True
        _detail("WriteCombined expert DMA staging enabled (--wc-alloc)")

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
    if args.vram_safety_margin > 0:
        SAFETY_MARGIN_MB = args.vram_safety_margin
    else:
        SAFETY_MARGIN_MB = 600
    vram_monitor = VramMonitor(device_indices, poll_interval_ms=50, safety_margin_mb=SAFETY_MARGIN_MB)
    vram_monitor.start()
    _dim("VRAM monitor started (tracking warmup)")
    for idx in device_indices:
        total = vram_monitor.total_mb(idx)
        _dim(f"cuda:{idx}: {total:,} MB total")

    # ── Enable VRAM report if requested ──
    if getattr(args, 'vram_report', False):
        vram_monitor.enable_report()
        _dim("VRAM report enabled (periodic samples + events)")

    vram_monitor.report_event("model_loaded")

    # ── Phase 1: Warmup (trigger lazy Rust prefill allocations before HCS loading) ──
    # Run one real Rust prefill before budget measurement so first-request allocations
    # are not charged against the post-HCS steady-state budget.
    _model._hcs_device = None
    _model._multi_gpu_hcs = False
    _status("Pre-allocating Rust prefill engine")
    # Must happen BEFORE VRAM budget measurement and HCS pool loading.
    # The prefill engine allocates fixed scratch buffers on the GPU.
    # If we wait until after HCS, there may not be enough VRAM left.
    try:
        gpu_store.allocate_prefill_engine(_model.cfg.max_position_embeddings)
        _detail("Prefill engine scratch buffers allocated")
    except Exception as e:
        logger.error("Failed to pre-allocate prefill engine: %s", e)
        raise

    _status("Warmup (Rust prefill engine)")
    vram_monitor.report_event("warmup_start")
    t_warmup = time.time()
    try:
        warmup_token = 0
        warmup_probe = _model.tokenizer.encode(" hello") if _model.tokenizer is not None else []
        if warmup_probe:
            warmup_token = int(warmup_probe[0])
        warmup_len_default = min(25000, max(256, int(getattr(args, "gpu_prefill_threshold", 300))))
        startup_diag = _startup_diag_enabled()
        warmup_len = (
            _env_int("KRASIS_STARTUP_WARMUP_TOKENS", warmup_len_default, minimum=1)
            if startup_diag else warmup_len_default
        )
        if startup_diag:
            prefill_debug_enabled = os.environ.get("KRASIS_PREFILL_DEBUG", "") == "1"
            prefill_timing_enabled = _env_flag("KRASIS_PREFILL_TIMING")
            exit_after_calibration = _env_flag("KRASIS_STARTUP_EXIT_AFTER_CALIBRATION")
            _detail(
                f"Startup diag: warmup tokens={warmup_len:,} "
                f"(default {warmup_len_default:,}, token={warmup_token})"
            )
            logger.info(
                "Startup diag warmup: tokens=%d default_tokens=%d token_id=%d",
                warmup_len, warmup_len_default, warmup_token,
            )
            no_graph = _env_flag("KRASIS_NO_GRAPH")
            mapped_reads = _env_flag("KRASIS_MAPPED_READS")
            _detail(
                "Startup diag env: "
                f"no_graph={no_graph if no_graph is not None else 'unset'}, "
                f"mapped_reads={mapped_reads if mapped_reads is not None else 'unset'}, "
                f"prefill_debug={prefill_debug_enabled}, "
                f"prefill_timing={prefill_timing_enabled if prefill_timing_enabled is not None else 'unset'}, "
                f"exit_after_calibration={exit_after_calibration if exit_after_calibration is not None else 'unset'}"
            )
            logger.info(
                "Startup diag env: no_graph=%s mapped_reads=%s prefill_debug=%s "
                "prefill_timing=%s exit_after_calibration=%s",
                "unset" if no_graph is None else int(no_graph),
                "unset" if mapped_reads is None else int(mapped_reads),
                int(prefill_debug_enabled),
                "unset" if prefill_timing_enabled is None else int(prefill_timing_enabled),
                "unset" if exit_after_calibration is None else int(exit_after_calibration),
            )
        gpu_store.rust_prefill_tokens([warmup_token] * warmup_len, 0.0)
        _model.server_cleanup()
        _detail(f"Rust prefill warmed with {warmup_len:,} tokens before HCS budgeting")
    except Exception as e:
        logger.warning("Rust prefill warmup failed, continuing without it: %s", e)
        _warn(f"Rust prefill warmup failed: {e}")
    warmup_elapsed = time.time() - t_warmup
    _detail(f"Warmup complete in {warmup_elapsed:.1f}s")
    vram_monitor.report_event("warmup_end")

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

    # ── Phase 2: VRAM calibration ──
    # Measure real short/long prefill and decode VRAM minima with no HCS loaded,
    # then apply those transient deltas to the current post-calibration free VRAM.
    # This restores the measured decode model and avoids guessed headroom values.
    dev_idx = devices[0].index
    import torch
    tokenizer_path = os.path.join(args.model_path, "tokenizer.json")

    _status("VRAM calibration")
    vram_monitor.report_event("calibration_start")
    _detail(f"Safety margin: {SAFETY_MARGIN_MB:,} MB")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    _free_mb = torch.cuda.mem_get_info(dev_idx)[0] // (1024 * 1024)
    _total_mb = torch.cuda.mem_get_info(dev_idx)[1] // (1024 * 1024)
    _detail(f"Free VRAM after startup: {_free_mb:,} MB / {_total_mb:,} MB")
    logger.info("VRAM calibration baseline: free=%d MB, total=%d MB", _free_mb, _total_mb)

    max_calibration_tokens = max(1, min(
        _kv_cache_max_tokens(_model),
        _model.cfg.max_position_embeddings,
        STARTUP_CALIBRATION_LONG_TOKENS_CAP,
    ) - 100)
    short_target_default = min(STARTUP_CALIBRATION_SHORT_TOKENS, max_calibration_tokens)
    long_target_default = max(short_target_default, min(max_calibration_tokens, int(max_calibration_tokens * 0.8)))
    startup_diag = _startup_diag_enabled()
    short_target = short_target_default
    long_target = long_target_default
    if startup_diag:
        short_target = min(max_calibration_tokens, _env_int(
            "KRASIS_STARTUP_CAL_SHORT_TOKENS", short_target_default, minimum=1
        ))
        long_target = min(max_calibration_tokens, _env_int(
            "KRASIS_STARTUP_CAL_LONG_TOKENS", long_target_default, minimum=1
        ))
    long_target = max(short_target, long_target)
    if startup_diag:
        _detail(
            "Startup diag: calibration tokens "
            f"short={short_target:,} (default {short_target_default:,}), "
            f"long={long_target:,} (default {long_target_default:,}), "
            f"cap={max_calibration_tokens:,}"
        )
        logger.info(
            "Startup diag calibration: short_tokens=%d default_short=%d long_tokens=%d "
            "default_long=%d cap=%d decode_tokens=%d",
            short_target, short_target_default, long_target, long_target_default,
            max_calibration_tokens, STARTUP_CALIBRATION_DECODE_TOKENS,
        )
    calibration_prompts = _make_startup_calibration_prompts(_model, [short_target, long_target])
    calibration_stop_ids: list[int] = []

    def _measure_vram_probe(label: str, prompt_tokens: list[int]) -> tuple[int, int, int, int]:
        _detail(f"{label}: probing {len(prompt_tokens):,} prompt tokens + {STARTUP_CALIBRATION_DECODE_TOKENS} decode tokens")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        _model.server_cleanup()
        time.sleep(0.1)

        baseline_free = int(vram_monitor.current_free_mb(dev_idx))

        vram_monitor.reset_min_free()
        t_prefill = time.time()
        first_token, prompt_len, kv_overflow = gpu_store.rust_prefill_tokens(prompt_tokens, temperature=0.0)
        torch.cuda.synchronize()
        time.sleep(0.1)
        prefill_elapsed = time.time() - t_prefill
        prefill_min_free = int(vram_monitor.min_free_mb(dev_idx))
        if kv_overflow:
            _model.server_cleanup()
            raise RuntimeError(f"{label} VRAM calibration overflowed KV cache at {prompt_len:,} tokens")

        vram_monitor.reset_min_free()
        t_decode = time.time()
        gpu_store.gpu_generate_stream_probe(
            tokenizer_path=tokenizer_path,
            first_token=first_token,
            start_position=prompt_len,
            max_tokens=STARTUP_CALIBRATION_DECODE_TOKENS,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            stop_ids=calibration_stop_ids,
            presence_penalty=0.0,
        )
        torch.cuda.synchronize()
        time.sleep(0.1)
        decode_elapsed = time.time() - t_decode
        decode_min_free = int(gpu_store.get_last_min_free_vram_mb())

        _model.server_cleanup()
        torch.cuda.synchronize()
        time.sleep(0.1)
        post_cleanup_free = int(vram_monitor.current_free_mb(dev_idx))

        _detail(
            f"{label}: baseline={baseline_free:,} MB, "
            f"prefill min={prefill_min_free:,} MB, decode min={decode_min_free:,} MB, "
            f"post-cleanup={post_cleanup_free:,} MB"
        )
        if startup_diag:
            prefill_tps = prompt_len / prefill_elapsed if prefill_elapsed > 0 else 0.0
            decode_tps = STARTUP_CALIBRATION_DECODE_TOKENS / decode_elapsed if decode_elapsed > 0 else 0.0
            _detail(
                f"{label}: prefill {prefill_elapsed:.2f}s ({prefill_tps:.1f} tok/s), "
                f"decode {decode_elapsed:.2f}s ({decode_tps:.1f} tok/s)"
            )
            logger.info(
                "Startup diag probe %s: prompt_len=%d baseline_free=%d prefill_min=%d "
                "decode_min=%d post_cleanup=%d prefill_s=%.3f prefill_tps=%.2f "
                "decode_s=%.3f decode_tps=%.2f",
                label, prompt_len, baseline_free, prefill_min_free, decode_min_free,
                post_cleanup_free, prefill_elapsed, prefill_tps, decode_elapsed, decode_tps,
            )
        return prompt_len, baseline_free, prefill_min_free, decode_min_free

    short_tokens, short_baseline_free, short_prefill_min, short_decode_min = _measure_vram_probe(
        "Short calibration", calibration_prompts[0]
    )
    long_tokens, long_baseline_free, long_prefill_min, long_decode_min = _measure_vram_probe(
        "Long calibration", calibration_prompts[1]
    )

    # Compute max scratch: worst case prompt (50K tokens).
    # This determines how much VRAM prefill can claim via soft eviction.
    max_scratch_tokens = min(50000, _model.cfg.max_position_embeddings)
    max_scratch_mb = gpu_store.prefill_scratch_reservation_mb(max_scratch_tokens)
    _detail(f"Scratch reservation: max={max_scratch_mb:,} MB (at {max_scratch_tokens:,} tokens)")

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    _model.server_cleanup()
    time.sleep(0.1)
    post_calibration_free_mb = int(vram_monitor.current_free_mb(dev_idx))

    short_prefill_delta = max(0, short_baseline_free - short_prefill_min)
    long_prefill_delta = max(0, long_baseline_free - long_prefill_min)
    short_decode_delta = max(0, short_baseline_free - short_decode_min)
    long_decode_delta = max(0, long_baseline_free - long_decode_min)

    prefill_short_free = max(0, post_calibration_free_mb - short_prefill_delta)
    prefill_long_free = max(0, post_calibration_free_mb - long_prefill_delta)
    decode_short_free = max(0, post_calibration_free_mb - short_decode_delta)
    decode_long_free = max(0, post_calibration_free_mb - long_decode_delta)

    # GPU0 HCS is fully reclaimable. Load as much as measured short-prompt decode
    # can hold above the safety margin; longer prompts are handled by measured
    # per-request soft reload limits in Rust.
    reclaimable_hcs_budget = max(0, decode_short_free - SAFETY_MARGIN_MB)

    vram_monitor.report_event("calibration_end")
    _status("VRAM calibration complete")
    _detail(f"Post-calibration free VRAM: {post_calibration_free_mb:,} MB")
    _detail(
        f"Transient deltas: short prefill={short_prefill_delta:,} MB, "
        f"long prefill={long_prefill_delta:,} MB, short decode={short_decode_delta:,} MB, "
        f"long decode={long_decode_delta:,} MB"
    )
    _detail(f"GPU0 reclaimable HCS budget: {reclaimable_hcs_budget:,} MB")
    _detail(f"Worst-case prefill scratch reservation: {max_scratch_mb:,} MB at {max_scratch_tokens:,} tokens")
    logger.info(
        "VRAM calibration: post_free=%d MB, short=%dtok, long=%dtok, gpu0_reclaimable=%d MB, max_scratch=%d MB",
        post_calibration_free_mb, short_tokens, long_tokens, reclaimable_hcs_budget, max_scratch_mb,
    )
    if startup_diag and _env_flag("KRASIS_STARTUP_EXIT_AFTER_CALIBRATION"):
        _status("Startup diagnostic exit")
        _detail("Exiting after VRAM calibration by request")
        logger.info("Startup diagnostic exit after VRAM calibration requested")
        return

    # ── Pre-compute multi-GPU layer splits (before HCS, so we can filter rankings) ──
    # Splits are based on total HCS budget (hard+soft) on each GPU, so layers
    # are proportional to where experts can actually live.
    # _multi_gpu_splits: list of split points (one per aux GPU). Empty = single GPU.
    # _multi_gpu_gqa_offsets: GQA count before each split point.
    _multi_gpu_splits = []
    _multi_gpu_gqa_offsets = []
    # Legacy compat
    _multi_gpu_split = 0
    _multi_gpu_gqa_offset = 0
    if num_gpus_available > 1 and args.hcs:
        _status(f"Computing multi-GPU layer split ({num_gpus_available} GPUs)")
        num_layers = len(_model.layers)

        # Compute per-layer VRAM cost from actual loaded weights (not hardcoded estimates).
        _layer_vram_mb = []
        kv_cache = _model.kv_caches[0] if _model.kv_caches else None
        kv_total_mb = 0
        if kv_cache is not None:
            # k_cache and v_cache are single tensors (not lists); compute sizes individually
            # to avoid FP8 tensor addition (ufunc_add not implemented for Float8_e4m3fn)
            kv_total_bytes = 0
            for cache_tensor in (kv_cache.k_cache, kv_cache.v_cache):
                if cache_tensor is not None:
                    kv_total_bytes += cache_tensor.nelement() * cache_tensor.element_size()
            kv_total_mb = kv_total_bytes / (1024 * 1024)
            # num_layers is the first dim of the cache tensor
            num_kv_layers = kv_cache.k_cache.shape[0] if kv_cache.k_cache is not None else 0
            kv_per_layer_mb = kv_total_mb / num_kv_layers if num_kv_layers > 0 else 0
        else:
            kv_per_layer_mb = 0

        from krasis.attention import MarlinWeight as _MW
        for layer in _model.layers:
            layer_bytes = 0
            layer_bytes += layer.input_norm_weight.nelement() * layer.input_norm_weight.element_size()
            layer_bytes += layer.post_attn_norm_weight.nelement() * layer.post_attn_norm_weight.element_size()
            attn = layer.attention
            for attr_name in dir(attn):
                val = getattr(attn, attr_name, None)
                if isinstance(val, _MW):
                    # MarlinWeight: count packed + scales GPU tensors
                    if val.packed.is_cuda:
                        layer_bytes += val.packed.nelement() * val.packed.element_size()
                    if val.scales.is_cuda:
                        layer_bytes += val.scales.nelement() * val.scales.element_size()
                elif isinstance(val, torch.Tensor) and val.device.type == 'cuda':
                    layer_bytes += val.nelement() * val.element_size()
                elif isinstance(val, tuple) and len(val) == 2:
                    for t in val:
                        if isinstance(t, torch.Tensor) and t.device.type == 'cuda':
                            layer_bytes += t.nelement() * t.element_size()
            if layer.is_moe:
                for w in [layer.gate_weight, layer.gate_bias, layer.e_score_correction_bias]:
                    if w is not None:
                        layer_bytes += w.nelement() * w.element_size()
                if layer.shared_expert is not None:
                    for v in layer.shared_expert.values():
                        if isinstance(v, torch.Tensor):
                            layer_bytes += v.nelement() * v.element_size()
                        elif isinstance(v, tuple):
                            for t in v:
                                if isinstance(t, torch.Tensor):
                                    layer_bytes += t.nelement() * t.element_size()
            layer_mb = layer_bytes / (1024 * 1024)
            if layer.layer_type != "linear_attention":
                layer_mb += kv_per_layer_mb
            _layer_vram_mb.append(layer_mb)

        # Compute base overhead for the last aux GPU (embedding + lm_head + final_norm)
        # Only the last GPU needs the LM head; intermediate GPUs only need attention/norms.
        last_gpu_base_overhead_bytes = 0
        last_gpu_base_overhead_bytes += _model.embedding.nelement() * _model.embedding.element_size()
        last_gpu_base_overhead_bytes += _model.final_norm.nelement() * _model.final_norm.element_size()
        if isinstance(_model.lm_head_data, tuple):
            for t in _model.lm_head_data:
                if isinstance(t, torch.Tensor):
                    last_gpu_base_overhead_bytes += t.nelement() * t.element_size()
        elif isinstance(_model.lm_head_data, torch.Tensor):
            last_gpu_base_overhead_bytes += _model.lm_head_data.nelement() * _model.lm_head_data.element_size()
        last_gpu_base_overhead = last_gpu_base_overhead_bytes / (1024 * 1024)

        # Compute HCS budget for each GPU.
        # GPU0: total reclaimable resident HCS budget.
        # Aux GPUs: total VRAM - attention cost - overhead - safety margin.
        # We iterate to find self-consistent splits where each GPU's layer assignment
        # matches its available HCS budget proportionally.
        gpu0_hcs_total = reclaimable_hcs_budget
        num_aux = num_gpus_available - 1
        aux_totals = [vram_monitor.total_mb(device_indices[i + 1]) for i in range(num_aux)]

        # Initial guess: equal distribution
        initial_splits = [int(round(num_layers * (i + 1) / num_gpus_available))
                          for i in range(num_aux)]
        # Clamp each split to [2, num_layers - 2] and ensure monotonic
        for i in range(len(initial_splits)):
            initial_splits[i] = max(2, min(initial_splits[i], num_layers - 2))
        for i in range(1, len(initial_splits)):
            initial_splits[i] = max(initial_splits[i], initial_splits[i - 1] + 1)

        _multi_gpu_splits = list(initial_splits)
        gpu_hcs_budgets = [0.0] * num_gpus_available

        for _iter in range(5):
            prev_splits = list(_multi_gpu_splits)

            # Compute boundaries: [0, splits[0], splits[1], ..., num_layers]
            boundaries = [0] + _multi_gpu_splits + [num_layers]

            # Compute HCS budget for each GPU
            gpu_hcs_budgets[0] = gpu0_hcs_total
            for i in range(num_aux):
                gpu_idx_in_list = i + 1
                layer_start = boundaries[gpu_idx_in_list]
                layer_end_b = boundaries[gpu_idx_in_list + 1]
                attn_cost = sum(_layer_vram_mb[j] for j in range(layer_start, layer_end_b))
                # Last aux GPU has LM head overhead
                base_overhead = last_gpu_base_overhead if (i + 1 == num_aux) else 0
                gpu_hcs_budgets[gpu_idx_in_list] = max(0,
                    aux_totals[i] - base_overhead - attn_cost - SAFETY_MARGIN_MB)

            total_hcs = sum(gpu_hcs_budgets)
            if total_hcs <= 0:
                break

            # Redistribute layers proportionally to HCS budgets
            cumulative = 0.0
            new_splits = []
            for i in range(num_aux):
                cumulative += gpu_hcs_budgets[i]
                split_pos = int(round(num_layers * cumulative / total_hcs))
                split_pos = max(2, min(split_pos, num_layers - 2))
                new_splits.append(split_pos)

            # Ensure monotonic increasing
            for i in range(1, len(new_splits)):
                new_splits[i] = max(new_splits[i], new_splits[i - 1] + 1)

            _multi_gpu_splits = new_splits
            if _multi_gpu_splits == prev_splits:
                break  # converged

        # Compute GQA offsets for each split point
        _multi_gpu_gqa_offsets = []
        for split in _multi_gpu_splits:
            gqa_count = 0
            for i in range(min(split, num_layers)):
                if _model.layers[i].layer_type != "linear_attention":
                    gqa_count += 1
            _multi_gpu_gqa_offsets.append(gqa_count)

        # Recompute final boundaries and total HCS for display
        boundaries = [0] + _multi_gpu_splits + [num_layers]
        # Recompute HCS budgets for final splits (needed when loop was skipped)
        gpu_hcs_budgets[0] = gpu0_hcs_total
        for i in range(num_aux):
            gpu_idx_in_list = i + 1
            layer_start = boundaries[gpu_idx_in_list]
            layer_end_b = boundaries[gpu_idx_in_list + 1]
            attn_cost = sum(_layer_vram_mb[j] for j in range(layer_start, layer_end_b))
            base_overhead = last_gpu_base_overhead if (i + 1 == num_aux) else 0
            gpu_hcs_budgets[gpu_idx_in_list] = max(0,
                aux_totals[i] - base_overhead - attn_cost - SAFETY_MARGIN_MB)
        total_hcs = sum(gpu_hcs_budgets)
        _detail(f"HCS budgets: " + ", ".join(
            f"GPU{i} {gpu_hcs_budgets[i]:,.0f} MB" for i in range(num_gpus_available)
        ) + f" = {total_hcs:,.0f} MB total")
        for i in range(num_gpus_available):
            n_layers = boundaries[i + 1] - boundaries[i]
            ratio = gpu_hcs_budgets[i] / total_hcs if total_hcs > 0 else 0
            _detail(f"  GPU{i}: layers [{boundaries[i]}..{boundaries[i+1]}) = {n_layers} layers ({ratio:.1%})")
        logger.info("Multi-GPU splits: %s, gqa_offsets=%s", _multi_gpu_splits, _multi_gpu_gqa_offsets)

        # Legacy compat (used by HCS ranking filter below)
        _multi_gpu_split = _multi_gpu_splits[0] if _multi_gpu_splits else 0
        _multi_gpu_gqa_offset = _multi_gpu_gqa_offsets[0] if _multi_gpu_gqa_offsets else 0

    if not args.hcs:
        _status("GPU decode (no HCS)")
        _warn("All experts streamed via DMA per token (slow for decode)")
    else:
        _status("Calculating HCS budget")

        # ── Device selection ──
        primary_dev = devices[0]
        total_experts = cfg.n_routed_experts * cfg.num_moe_layers

        # ── Build heatmap ──
        # Always rebuild fresh on startup so the heatmap reflects current decode
        # routing.  A stale cached heatmap causes dramatically worse HCS hit
        # rates even with the same budget (March 2026: 69% coverage / 24% miss
        # with stale vs 54% coverage / 8% miss with fresh).
        #
        # Exception: if the user explicitly passed --heatmap-path, trust it.
        if args.heatmap_path:
            _dim(f"Using user-provided heatmap: {os.path.basename(heatmap_path)}")
            if not _validate_heatmap(heatmap_path, cfg):
                _warn("User-provided heatmap doesn't match model config — using it anyway")
        else:
            _status("Building expert heatmap (decode-weighted calibration)")
            heatmap_path = _build_heatmap(_model, heatmap_path)

        # ── Load heatmap and build sorted ranking ──
        with open(heatmap_path) as f:
            raw_heatmap = json.load(f)
        # Strip metadata before building ranking
        raw_heatmap.pop("_metadata", None)
        sorted_ranking = sorted(raw_heatmap.items(), key=lambda x: x[1], reverse=True)
        ranking = [(int(k.split(",")[0]), int(k.split(",")[1])) for k, _ in sorted_ranking]
        _detail(f"Heatmap: {len(ranking):,} experts ranked from {len(raw_heatmap):,} entries")

        # Build full ranking for a layer range: heatmap-ranked experts first,
        # then unranked experts to fill remaining VRAM (better than empty slots).
        def _full_ranking_for_layers(base_ranking, layer_start, layer_end):
            """Return ranking with heatmap experts first, then unranked experts for [layer_start, layer_end)."""
            # Filter base ranking to this layer range
            filtered = [(l, e) for l, e in base_ranking if layer_start <= l < layer_end]
            ranked_set = set(filtered)
            # Append all unranked experts from this layer range
            for i in range(layer_start, layer_end):
                layer = _model.layers[i]
                if not layer.is_moe:
                    continue
                n_experts = cfg.n_routed_experts
                for e in range(n_experts):
                    if (i, e) not in ranked_set:
                        filtered.append((i, e))
            return filtered

        # ── Pass calibration data to Rust ──
        if hasattr(_model, '_gpu_decode_store'):
            store = _model._gpu_decode_store

            cal_msg = store.set_vram_calibration(
                short_tokens, long_tokens,
                prefill_short_free, prefill_long_free,
                decode_short_free, decode_long_free,
                post_calibration_free_mb,
                SAFETY_MARGIN_MB,
            )
            _dim(cal_msg)

            # ── Set decode segment on primary store (for accurate HCS% reporting) ──
            gpu0_layer_end = _multi_gpu_split if _multi_gpu_split > 0 else len(_model.layers)
            store.set_decode_segment(0, gpu0_layer_end)

            # Multi-GPU: ensure all layers are in Marlin format BEFORE restricting swaps.
            # After calibration, weights may be in simple INT4. restrict_to_decode_segment
            # removes swap entries for layers outside GPU0's segment, so those layers would
            # be permanently stuck in simple INT4 if not swapped back to Marlin first.
            # Prefill needs ALL layers in Marlin format on GPU0.
            if _multi_gpu_split > 0:
                store.swap_to_marlin()
                _model.restrict_to_decode_segment(0, gpu0_layer_end)

            # ── Initialize GPU0 HCS as fully reclaimable on the primary GPU ──
            # In multi-GPU mode, filter ranking to GPU0's layer segment only
            # and include unranked experts to fill remaining VRAM (better than empty slots).
            gpu0_ranking = ranking
            gpu0_hard = 0
            gpu0_soft = reclaimable_hcs_budget
            if _multi_gpu_split > 0:
                gpu0_ranking = _full_ranking_for_layers(ranking, 0, _multi_gpu_split)
                num_ranked = sum(1 for l, e in gpu0_ranking if (l, e) in set(ranking))
                _dim(f"GPU0 HCS: {len(gpu0_ranking)} experts for layers [0..{_multi_gpu_split}) "
                     f"({num_ranked} ranked + {len(gpu0_ranking) - num_ranked} unranked), "
                     f"reclaimable: {gpu0_soft:,} MB")
            else:
                # Single GPU: include all unranked experts too
                gpu0_ranking = _full_ranking_for_layers(ranking, 0, len(_model.layers))
                num_ranked = len(ranking)
                _dim(f"GPU0 HCS: {len(gpu0_ranking)} experts "
                     f"({num_ranked} ranked + {len(gpu0_ranking) - num_ranked} unranked), "
                     f"reclaimable: {gpu0_soft:,} MB")

            t_hcs = time.time()
            vram_monitor.report_event("hcs_init_start")
            _status("Loading GPU0 HCS pool (fully reclaimable)")

            result = store.hcs_pool_init_tiered(
                gpu0_ranking,
                hard_budget_mb=gpu0_hard,
                soft_budget_mb=gpu0_soft,
                safety_margin_mb=SAFETY_MARGIN_MB,
            )
            hcs_elapsed = time.time() - t_hcs

            vram_monitor.report_event("hcs_init_end")
            _status("HCS pool loaded")
            _detail(result)
            _dim(f"Loaded in {hcs_elapsed:.1f}s")
            logger.info("HCS pool: %s (%.1fs)", result, hcs_elapsed)

    # ── Decode validation ──
    # Rust prefill warmup already ran before HCS budgeting.
    vram_monitor.report_event("validation_start")
    _status("Decode validation")
    _detail("Skipped: Rust prefill already warmed before HCS budgeting")
    vram_monitor.report_event("validation_end")

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

    # ── Multi-GPU decode setup ──
    # Lists passed to RustServer for N-GPU pipeline decode
    all_aux_gpu_store_addrs = []
    all_multi_gpu_split_layers = list(_multi_gpu_splits)
    all_multi_gpu_gqa_offsets = list(_multi_gpu_gqa_offsets)
    if _multi_gpu_splits and args.hcs:
        num_aux = len(_multi_gpu_splits)
        num_layers = len(_model.layers)
        boundaries = [0] + list(_multi_gpu_splits) + [num_layers]
        vram_monitor.report_event("multi_gpu_setup_start")
        _status(f"Multi-GPU decode setup ({num_aux + 1} GPUs)")
        gc.collect()
        torch.cuda.empty_cache()

        # Count layer types per GPU segment
        for gpu_i in range(num_aux + 1):
            seg_start, seg_end = boundaries[gpu_i], boundaries[gpu_i + 1]
            la, gqa, moe, dense = 0, 0, 0, 0
            for i in range(seg_start, seg_end):
                layer = _model.layers[i]
                if layer.layer_type == "linear_attention": la += 1
                else: gqa += 1
                if layer.is_moe: moe += 1
                elif layer.dense_mlp is not None: dense += 1
            _detail(f"  GPU{gpu_i}: layers [{seg_start}..{seg_end}) = {seg_end - seg_start} layers "
                    f"({la} LA + {gqa} GQA, {moe} MoE + {dense} dense)")
        _dim(f"  GQA cache offsets: {_multi_gpu_gqa_offsets}")
        _dim(f"  Attention layers are PERMANENT on each GPU (copied at setup, never evicted)")

        # Create aux stores for each aux GPU
        aux_stores = []
        for i in range(num_aux):
            seg_start = _multi_gpu_splits[i]
            seg_end = boundaries[i + 2]  # boundaries[i+1+1]
            gpu_idx = device_indices[i + 1]
            _dim(f"Creating aux decode store {i+1} on cuda:{gpu_idx} for layers [{seg_start}..{seg_end})...")
            aux_store = _model.setup_gpu_decode_store_aux(
                gpu_idx=gpu_idx,
                split_layer=seg_start,
                layer_end=seg_end,
            )
            aux_store.set_decode_segment(seg_start, seg_end)
            aux_stores.append(aux_store)
            all_aux_gpu_store_addrs.append(aux_store.gpu_store_addr())

        # Log post-setup VRAM
        for dev in devices:
            torch.cuda.synchronize(dev)
        gc.collect()
        torch.cuda.empty_cache()
        for idx in device_indices:
            post_free = vram_monitor.current_free_mb(idx)
            total_mb = vram_monitor.total_mb(idx)
            _dim(f"  cuda:{idx} after aux store setup: {post_free:,.0f} MB free / {total_mb:,} MB total")

        # Initialize HCS on each aux store if we have the heatmap
        if args.hcs and 'ranking' in locals():
            for i, aux_store in enumerate(aux_stores):
                seg_start = boundaries[i + 1]
                seg_end = boundaries[i + 2]
                gpu_idx = device_indices[i + 1]

                aux_ranking = _full_ranking_for_layers(ranking, seg_start, seg_end)
                num_aux_ranked = sum(1 for l, e in aux_ranking if (l, e) in set(ranking))
                _dim(f"  GPU{i+1} HCS: {len(aux_ranking)} experts for layers [{seg_start}..{seg_end}) "
                     f"({num_aux_ranked} ranked + {len(aux_ranking) - num_aux_ranked} unranked)")
                # Measure aux GPU free VRAM for HCS budget
                aux_free_mb = vram_monitor.current_free_mb(gpu_idx)
                aux_hcs_budget = max(0, int(aux_free_mb) - SAFETY_MARGIN_MB)
                # 100% hard tier — aux GPUs never do prefill so never need to evict
                aux_hard = aux_hcs_budget
                aux_soft = 0
                _detail(f"  GPU{i+1} HCS budget: {aux_hard:,} MB hard (aux_free={aux_free_mb:,.0f} MB)")

                if aux_ranking:
                    result = aux_store.hcs_pool_init_tiered(
                        aux_ranking,
                        hard_budget_mb=aux_hard,
                        soft_budget_mb=aux_soft,
                        safety_margin_mb=SAFETY_MARGIN_MB,
                    )
                    _detail(f"  GPU{i+1} HCS: {result}")

        # ── Multi-GPU decode validation ──
        _status("Validating multi-GPU decode")
        vram_monitor.reset_min_free()
        try:
            gpu_store = _model._gpu_decode_store
            evicted0, freed0 = gpu_store.py_hcs_evict_for_prefill(500)
            if evicted0 > 0:
                _dim(f"  Evicted {evicted0} soft experts for validation prefill")

            prompt_tokens = _chat_prompt_tokens(_model, "Hi")
            stop_ids = _default_stop_ids(_model)
            first_token, prompt_len, _kv_overflow = gpu_store.rust_prefill_tokens(
                prompt_tokens, temperature=0.6
            )

            # Copy KV cache and LA state to each aux store
            for i in range(num_aux):
                seg_start = boundaries[i + 1]
                seg_end = boundaries[i + 2]
                gpu_store.py_copy_kv_to_aux(
                    all_aux_gpu_store_addrs[i], seg_start, seg_end,
                    _multi_gpu_gqa_offsets[i], prompt_len)
                gpu_store.py_copy_la_states_to_aux(
                    all_aux_gpu_store_addrs[i], seg_start, seg_end)

            # Reload soft HCS on GPU0 only (aux GPUs have no soft tier)
            r0, _ = gpu_store.py_hcs_reload_after_prefill(prompt_len)
            if r0 > 0:
                _dim(f"  Reloaded {r0} soft experts after validation prefill")

            # Run multi-GPU decode
            if first_token not in stop_ids:
                tokens = gpu_store.gpu_generate_batch_multi(
                    aux_store_addrs=all_aux_gpu_store_addrs,
                    split_layers=all_multi_gpu_split_layers,
                    gqa_cache_offsets=all_multi_gpu_gqa_offsets,
                    first_token=first_token,
                    start_position=prompt_len,
                    max_tokens=4,
                    temperature=0.6,
                    top_k=50,
                    top_p=0.95,
                    stop_ids=stop_ids,
                    presence_penalty=0.0,
                )
                _detail(f"Multi-GPU decode validation: {len(tokens)} tokens generated OK")
            else:
                _detail("Multi-GPU decode validation: prefill hit stop token (OK)")

            # Log VRAM stats from all GPUs during multi-GPU decode
            torch.cuda.synchronize()
            time.sleep(0.1)  # let monitor poll
            for idx in device_indices:
                min_free = vram_monitor.min_free_mb(idx)
                current_free = vram_monitor.current_free_mb(idx)
                total_mb = vram_monitor.total_mb(idx)
                _dim(f"  cuda:{idx} during multi-GPU decode: min_free={min_free:,} MB, "
                     f"current_free={current_free:,.0f} MB / {total_mb:,} MB total")

            # ── Validate aux GPU VRAM safety during decode ──
            # Aux GPUs never run prefill, so decode transients are the only VRAM pressure.
            # Check that min_free stayed above 0 (budget was safe).
            for i in range(num_aux):
                gpu_idx = device_indices[i + 1]
                aux_min_free = vram_monitor.min_free_mb(gpu_idx)
                aux_current = vram_monitor.current_free_mb(gpu_idx)
                if aux_min_free < 50:
                    _warn(f"  GPU{i+1} (cuda:{gpu_idx}): min_free={aux_min_free} MB during decode — "
                          f"dangerously close to OOM, HCS budget may be too aggressive")
                else:
                    _dim(f"  GPU{i+1} (cuda:{gpu_idx}): min_free={aux_min_free} MB during decode, "
                         f"current_free={aux_current:.0f} MB (budget safe)")

        except Exception as e:
            raise RuntimeError(
                f"Multi-GPU decode validation failed: {e}\n"
                "Cannot start server with broken multi-GPU decode. "
                "Fix the underlying issue or disable multi-GPU."
            ) from e

        vram_monitor.report_event("multi_gpu_setup_end")
        _status(f"Multi-GPU decode ready ({num_aux + 1} GPUs, splits={_multi_gpu_splits})")

    # ── Final VRAM summary (all GPUs) ──
    if num_gpus_available > 1 and args.hcs and _multi_gpu_splits:
        gc.collect()
        torch.cuda.empty_cache()
        _status("VRAM allocation summary")
        boundaries_final = [0] + list(_multi_gpu_splits) + [len(_model.layers)]
        for gpu_i in range(num_gpus_available):
            idx = device_indices[gpu_i]
            total_vram = vram_monitor.total_mb(idx)
            current_free = vram_monitor.current_free_mb(idx)
            used = total_vram - current_free
            seg_start = boundaries_final[gpu_i]
            seg_end = boundaries_final[gpu_i + 1]
            n_layers = seg_end - seg_start
            # Count layer types
            la_count = sum(1 for i in range(seg_start, seg_end) if _model.layers[i].layer_type == "linear_attention")
            gqa_count = n_layers - la_count
            moe_count = sum(1 for i in range(seg_start, seg_end) if _model.layers[i].is_moe)
            # Estimate attention VRAM for this segment
            seg_attn_mb = sum(_layer_vram_mb[j] for j in range(seg_start, seg_end))
            # HCS info
            if gpu_i == 0:
                hcs_reclaimable = gpu0_soft
                hcs_total = hcs_reclaimable
                hcs_type = f"reclaimable={hcs_reclaimable:,}MB"
                # Prefill-only info
                prefill_only_entries = getattr(_model, '_prefill_only_attn', [])
                if prefill_only_entries:
                    po_bytes = 0
                    for _, _, _, mw in prefill_only_entries:
                        po_bytes += mw.packed.nelement() * mw.packed.element_size()
                        po_bytes += mw.scales.nelement() * mw.scales.element_size()
                    po_mb = po_bytes / (1024 * 1024)
                    _detail(f"  GPU{gpu_i} (cuda:{idx}): layers [{seg_start}..{seg_end}) = {n_layers} ({la_count}LA+{gqa_count}GQA, {moe_count}MoE)")
                    _detail(f"    Attention (decode segment): {seg_attn_mb:.0f} MB (permanent, AWQ simple INT4)")
                    _detail(f"    Attention (prefill-only):   {po_mb:.0f} MB (freed after prefill, reclaimed for HCS)")
                    _detail(f"    HCS:   {hcs_type} = {hcs_total:,} MB total")
                    _detail(f"    VRAM:  {used:,.0f} MB used / {total_vram:,} MB total ({current_free:,.0f} MB free)")
                else:
                    _detail(f"  GPU{gpu_i} (cuda:{idx}): layers [{seg_start}..{seg_end}) = {n_layers} ({la_count}LA+{gqa_count}GQA, {moe_count}MoE)")
                    _detail(f"    Attention: {seg_attn_mb:.0f} MB")
                    _detail(f"    HCS:   {hcs_type} = {hcs_total:,} MB total")
                    _detail(f"    VRAM:  {used:,.0f} MB used / {total_vram:,} MB total ({current_free:,.0f} MB free)")
            else:
                aux_hcs_budget = gpu_hcs_budgets[gpu_i]
                attn_type = "AWQ simple INT4" if args.attention_quant == "awq" else "BF16"
                _detail(f"  GPU{gpu_i} (cuda:{idx}): layers [{seg_start}..{seg_end}) = {n_layers} ({la_count}LA+{gqa_count}GQA, {moe_count}MoE)")
                _detail(f"    Attention: {seg_attn_mb:.0f} MB ({attn_type}, permanent)")
                _detail(f"    HCS:   {aux_hcs_budget:,.0f} MB (100% hard, no prefill)")
                _detail(f"    VRAM:  {used:,.0f} MB used / {total_vram:,} MB total ({current_free:,.0f} MB free)")

    # ── Server registry: write entry + register cleanup ──
    _write_registry(args.host, args.port, _model_name)
    atexit.register(_remove_registry)

    # ── Rust HTTP server ──
    from krasis import RustServer

    tokenizer_path = os.path.join(args.model_path, "tokenizer.json")

    # Look up </think> token ID for thinking budget tracking.
    # Only activate if the model's chat template actually supports enable_thinking
    # (i.e. the template contains <think> logic). Without this check, models
    # without thinking support would never emit </think>, breaking the budget.
    _hf_tok = _model.tokenizer.tokenizer  # unwrap Tokenizer → HF AutoTokenizer
    _template = getattr(_hf_tok, "chat_template", "") or ""
    think_end_id = 0
    if "enable_thinking" in _template:
        _raw_id = _hf_tok.convert_tokens_to_ids("</think>")
        if isinstance(_raw_id, int) and _raw_id != _hf_tok.unk_token_id:
            think_end_id = _raw_id
            logger.info("Thinking end token: </think> = %d", think_end_id)
        else:
            logger.info("Template has enable_thinking but no </think> token")
    else:
        logger.info("Model template does not support enable_thinking — thinking budget disabled")

    # Look up turn-boundary tokens to suppress during generation.
    # Models sometimes generate <|im_start|> during multi-turn thinking,
    # creating phantom new turns. Suppressing these prevents the issue.
    _suppress_tokens = []
    for special_tok in ["<|im_start|>", "<|start_header_id|>", "<|begin_of_text|>"]:
        _raw_id = _hf_tok.convert_tokens_to_ids(special_tok)
        if isinstance(_raw_id, int) and _raw_id != _hf_tok.unk_token_id:
            _suppress_tokens.append(_raw_id)
    if _suppress_tokens:
        _model._gpu_decode_store.set_suppress_tokens(_suppress_tokens)
        _model._suppress_tokens = _suppress_tokens
        logger.info("Suppress tokens: %s", {tok: _hf_tok.convert_ids_to_tokens(tok) for tok in _suppress_tokens})

    logger.info(
        "Model loaded, starting server on %s:%d (max context: %d, decode: GPU%s)",
        args.host, args.port, max_ctx,
        f"s ({len(all_aux_gpu_store_addrs)+1}-GPU)" if all_aux_gpu_store_addrs else "",
    )

    # ── Session messenger bridge ──
    _session_proc = None
    if getattr(args, 'session_enabled', False):
        _session_proc = _start_session_bridge(args.host, args.port)

    rust_server = RustServer(
        _model,
        args.host,
        args.port,
        _model_name,
        tokenizer_path,
        max_ctx,
        args.enable_thinking,
        think_end_id,
        gpu_store_addr,
        all_aux_gpu_store_addrs,
        all_multi_gpu_split_layers,
        all_multi_gpu_gqa_offsets,
        test_endpoints=getattr(args, 'test_endpoints', False),
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

    # ── Final ready banner (after all setup is done) ──
    # Brief pause so any async log output settles before the banner
    time.sleep(1.0)
    _decode_mode = f"{len(all_aux_gpu_store_addrs)+1}-GPU" if all_aux_gpu_store_addrs else "GPU"
    _hcs_str = "on" if args.hcs else "off"
    _think_str = "on" if think_end_id else "off"
    print(flush=True)
    print(f"  {_BOLD}{_GREEN}{'━' * 54}{_NC}", flush=True)
    vram_monitor.report_event("server_ready")
    print(f"  {_BOLD}{_GREEN}  KRASIS SERVER READY{_NC}", flush=True)
    print(f"  {_BOLD}{_GREEN}{'━' * 54}{_NC}", flush=True)
    print(f"  {_GREEN}Model:{_NC}    {_BOLD}{_model_name}{_NC}", flush=True)
    print(f"  {_GREEN}Address:{_NC}  {_BOLD}{args.host}:{args.port}{_NC}", flush=True)
    print(f"  {_GREEN}Context:{_NC}  {max_ctx:,} tokens  |  KV cache: {args.kv_cache_mb:,} MB", flush=True)
    print(f"  {_GREEN}Decode:{_NC}   {_decode_mode}  |  HCS: {_hcs_str}  |  Think: {_think_str}", flush=True)
    print(f"  {_BOLD}{_GREEN}{'━' * 54}{_NC}", flush=True)
    print(f"  {_DIM}Press Q or Ctrl-C to stop{_NC}", flush=True)
    print(flush=True)

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
                # Re-show ready banner after benchmark output
                print(flush=True)
                print(f"  {_BOLD}{_GREEN}{'═' * 50}{_NC}", flush=True)
                print(f"  {_BOLD}{_GREEN}  SERVER READY — {args.host}:{args.port}{_NC}", flush=True)
                print(f"  {_BOLD}{_GREEN}{'═' * 50}{_NC}", flush=True)
                # Show Session ID again so it's visible after benchmark output
                if getattr(args, 'session_enabled', False):
                    sid_path = Path.home() / ".krasis" / "session_id"
                    if sid_path.is_file():
                        sid = sid_path.read_text().strip()
                        if sid:
                            print(f"  {_BOLD}Session ID: {sid}{_NC}", flush=True)
                _dim("Press Q or Ctrl-C to stop")

        bench_thread = threading.Thread(target=_run_benchmark, daemon=True)
        bench_thread.start()

    # run() releases the GIL and blocks until stop() is called
    rust_server.run()

    # ── Write VRAM report if enabled ──
    vram_monitor.report_event("server_shutdown")
    if getattr(args, 'vram_report', False):
        _report_path = os.path.join(_run_dir, "vram_report.csv")
        os.makedirs(os.path.dirname(_report_path), exist_ok=True)
        try:
            vram_monitor.write_report(_report_path)
            # Print summary of key events
            summary = vram_monitor.report_summary()
            if summary:
                print(flush=True)
                print(f"  {_BOLD}VRAM Report Summary{_NC}", flush=True)
                print(f"  {'─' * 60}", flush=True)
                _hdr = f"  {'Event':<30s} {'Time':>8s}"
                for i in device_indices:
                    _hdr += f"  {'GPU' + str(i):>8s}"
                print(_hdr, flush=True)
                print(f"  {'─' * 60}", flush=True)
                for event, ts_ms, gpu_free in summary:
                    _line = f"  {event:<30s} {ts_ms/1000:>7.1f}s"
                    for mb in gpu_free:
                        _line += f"  {mb:>6d}MB"
                    print(_line, flush=True)
                print(f"  {'─' * 60}", flush=True)
                print(f"  {_DIM}Full report: {_report_path}{_NC}", flush=True)
        except Exception as e:
            logger.warning("Failed to write VRAM report: %s", e)

    # ── Clean exit before Python teardown triggers cascading errors ──
    vram_monitor.stop()
    _remove_registry()
    _cleanup_cuda()
    os._exit(0)


if __name__ == "__main__":
    main()
