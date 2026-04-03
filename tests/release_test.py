#!/usr/bin/env python3
"""Release test runner for Krasis.

Generates configs on the fly, launches the installed `krasis` command for each,
runs benchmark + sanity + large prompt tests, and produces an HTML report
for manual review.

The `krasis` command is used deliberately (not ./dev run or python -m krasis.server)
to validate the actual installed package. If the installed krasis is broken or
behind, the release test catches it.

Usage:
    ./dev release-test <model-name>

This script must be run via ./dev release-test, not directly.
"""

import argparse
import http.client
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from krasis.run_paths import create_run_dir, get_run_dir, slugify

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

MODELS_DIR = os.path.expanduser("~/.krasis/models")
CACHE_DIR = os.path.expanduser("~/.krasis/cache")
DEFAULT_PORT = 8012
DEFAULT_HOST = "127.0.0.1"
BENCHMARK_TIMEOUT = 1200  # 20 min for model load + warmup + benchmark
CACHE_BUILD_TIMEOUT = 1800  # 30 min for cache building (INT8 takes longer)
AWQ_CALIBRATE_TIMEOUT = 1800  # 30 min for AWQ calibration
PERPLEXITY_TIMEOUT = 1800  # 30 min for model load + 10K token eval
PERPLEXITY_MAX_TOKENS = 20000
PERPLEXITY_WARN_THRESHOLD = 0.05  # +5% PPL increase from baseline = WARN
PERPLEXITY_FAIL_THRESHOLD = 0.15  # +15% PPL increase from baseline = FAIL
PERPLEXITY_BASELINES_FILE = "perplexity_baselines.json"
HEALTH_POLL_INTERVAL = 2

SUPPORTED_MODELS = [
    "Qwen3-Coder-Next",
    "Qwen3.5-35B-A3B",
    "Qwen3.5-122B-A10B",
    "Qwen3-235B-A22B",
    "Qwen3.5-397B-A17B",
]

CONFIG_VARIANTS = [
    {"name": "INT4 BF16", "bits": 4, "attention": "bf16"},
    {"name": "INT4 AWQ", "bits": 4, "attention": "awq"},
    {"name": "INT8 AWQ", "bits": 8, "attention": "awq"},
    {"name": "INT4 AWQ Multi-GPU", "bits": 4, "attention": "awq", "multi_gpu": True},
]

REFERENCE_VALIDATE_MAX_PROMPTS = 4

# ANSI codes (for terminal output only — stripped from report)
BOLD = "\033[1m"
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
NC = "\033[0m"

ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')


def info(msg):
    print(f"{CYAN}{BOLD}=>{NC} {msg}")


def ok(msg):
    print(f"{GREEN}{BOLD}OK{NC} {msg}")


def warn(msg):
    print(f"{YELLOW}{BOLD}!!{NC} {msg}")


def die(msg):
    print(f"{RED}{BOLD}ERROR{NC} {msg}", file=sys.stderr)
    sys.exit(1)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return ANSI_ESCAPE.sub('', text)


SESSION_WS = "Release Test"


def session_notify(msg: str):
    """Send a notification to Session messenger (best-effort, never fails)."""
    try:
        subprocess.run(
            ["session-send", SESSION_WS, msg],
            timeout=10, capture_output=True,
        )
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# GPU Detection
# ═══════════════════════════════════════════════════════════════════

def detect_best_gpu() -> Tuple[int, str, int]:
    """Detect the GPU with the most VRAM. Returns (index, name, vram_mb)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        die("nvidia-smi failed. Are NVIDIA drivers installed?")

    if result.returncode != 0:
        die("nvidia-smi failed. Are NVIDIA drivers installed?")

    best = None
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            idx, name, vram = int(parts[0]), parts[1], int(parts[2])
            if best is None or vram > best[2]:
                best = (idx, name, vram)

    if best is None:
        die("No GPUs detected.")
    return best


def detect_all_gpus() -> List[Tuple[int, str, int]]:
    """Detect all GPUs. Returns list of (index, name, vram_mb), sorted by VRAM descending."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        die("nvidia-smi failed. Are NVIDIA drivers installed?")

    if result.returncode != 0:
        die("nvidia-smi failed. Are NVIDIA drivers installed?")

    gpus = []
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            idx, name, vram = int(parts[0]), parts[1], int(parts[2])
            gpus.append((idx, name, vram))

    if not gpus:
        die("No GPUs detected.")
    # Sort by VRAM descending so the biggest GPU is first (primary)
    gpus.sort(key=lambda g: g[2], reverse=True)
    return gpus


# ═══════════════════════════════════════════════════════════════════
# Config Generation
# ═══════════════════════════════════════════════════════════════════

def read_model_config(model_name: str) -> Dict:
    """Read config.json from the model directory."""
    model_path = os.path.join(MODELS_DIR, model_name)
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        die(f"Model config not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def get_num_layers(config: Dict) -> int:
    """Extract num_hidden_layers from model config."""
    text_config = config.get("text_config", config)
    return text_config.get("num_hidden_layers", 0)


def estimate_total_params_b(config: Dict) -> float:
    """Estimate total model parameters in billions from config.json dimensions.

    Computes from: attention params + MoE expert params + shared/dense MLP params.
    This is an approximation but accurate enough for the >200B threshold.
    """
    tc = config.get("text_config", config)
    layers = tc.get("num_hidden_layers", 0)
    hidden = tc.get("hidden_size", 0)
    num_experts = tc.get("num_experts", 0)
    moe_inter = tc.get("moe_intermediate_size", tc.get("intermediate_size", 0))
    num_heads = tc.get("num_attention_heads", 0)
    num_kv_heads = tc.get("num_key_value_heads", num_heads)
    head_dim = hidden // num_heads if num_heads > 0 else 0

    if layers == 0 or hidden == 0:
        return 0.0

    # Per-layer attention: Q + K + V + O projections
    attn_params = hidden * (num_heads * head_dim) + hidden * (num_kv_heads * head_dim) * 2 + (num_heads * head_dim) * hidden

    # Per-layer MoE experts: gate + up + down per expert
    expert_params = num_experts * 3 * hidden * moe_inter if num_experts > 0 else 0

    # Shared expert (if present, typically same size as one MoE expert)
    shared_inter = tc.get("shared_expert_intermediate_size", 0)
    shared_params = 3 * hidden * shared_inter if shared_inter > 0 else 0

    total_per_layer = attn_params + expert_params + shared_params
    total = total_per_layer * layers
    return total / 1e9


def generate_config(model_name: str, variant: Dict, gpu_idx: int,
                    num_layers: int, all_gpu_indices: Optional[List[int]] = None) -> str:
    """Generate a temporary config file. Returns path to temp file.
    """
    model_path = os.path.join(MODELS_DIR, model_name)
    bits = variant["bits"]
    if variant.get("multi_gpu") and all_gpu_indices and len(all_gpu_indices) > 1:
        gpu_selection = ",".join(str(i) for i in all_gpu_indices)
    else:
        gpu_selection = str(gpu_idx)

    lines = [
        f"# Release test config — {variant['name']}",
        f'MODEL_PATH="{model_path}"',
        f'CFG_SELECTED_GPUS="{gpu_selection}"',
        f'CFG_PP_PARTITION="{num_layers}"',
        f'CFG_LAYER_GROUP_SIZE="2"',
        f'CFG_KV_DTYPE="polar4"',
        f'CFG_GPU_EXPERT_BITS="{bits}"',
        f'CFG_CPU_EXPERT_BITS="{bits}"',
        f'CFG_ATTENTION_QUANT="{variant["attention"]}"',
        f'CFG_SHARED_EXPERT_QUANT="int8"',
        f'CFG_DENSE_MLP_QUANT="int8"',
        f'CFG_LM_HEAD_QUANT="int8"',
        f'CFG_HOST="0.0.0.0"',
        f'CFG_PORT="{DEFAULT_PORT}"',
        f'CFG_GPU_PREFILL_THRESHOLD="300"',
    ]

    fd, path = tempfile.mkstemp(prefix="krasis-release-", suffix=".conf")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ═══════════════════════════════════════════════════════════════════
# Cache Building (Pre-flight)
# ═══════════════════════════════════════════════════════════════════

def cache_exists(model_name: str, bits: int) -> bool:
    """Check if the Marlin expert cache exists for a given quantization.

    The user-facing surface exposes one expert quantization choice. Runtime
    still expects matching GPU/host bit-width config keys, but there is only
    one cache artifact per quantization level.
    Checks for any group size (g32, g64, g128) since the Rust engine tries
    multiple group sizes when building.
    """
    import glob as _glob
    model_cache = os.path.join(CACHE_DIR, model_name)
    has_gpu = bool(_glob.glob(os.path.join(model_cache, f"experts_marlin_int{bits}_g*.bin")))
    return has_gpu


def build_caches(model_name: str, gpu_idx: int, num_layers: int,
                 log_dir: str, force: bool = False,
                 skip_int8: bool = False) -> Dict[int, bool]:
    """Build expert caches for all required bit-widths before running tests.

    Launches krasis with --build-cache for each unique bit-width (INT4 and INT8).
    If cache already exists and force=False, skips building.
    If skip_int8=True, INT8 caches are not built.

    Returns dict mapping bits -> success.
    """
    # Collect unique bit-widths needed across all config variants
    needed_bits = set()
    for variant in CONFIG_VARIANTS:
        if variant.get("multi_gpu"):
            continue
        if skip_int8 and variant["bits"] == 8:
            continue
        needed_bits.add(variant["bits"])

    results = {}
    for bits in sorted(needed_bits):
        if not force and cache_exists(model_name, bits):
            ok(f"INT{bits} cache already exists — skipping build")
            results[bits] = True
            continue

        info(f"Building INT{bits} expert cache (this may take several minutes)...")
        config_path = None
        proc = None
        log_path = os.path.join(log_dir, f"cache_build_int{bits}.log")

        try:
            # Generate a minimal config just for cache building
            model_path = os.path.join(MODELS_DIR, model_name)
            lines = [
                f"# Cache build config — INT{bits}",
                f'MODEL_PATH="{model_path}"',
                f'CFG_SELECTED_GPUS="{gpu_idx}"',
                f'CFG_PP_PARTITION="{num_layers}"',
                f'CFG_LAYER_GROUP_SIZE="2"',
                f'CFG_KV_DTYPE="polar4"',
                f'CFG_GPU_EXPERT_BITS="{bits}"',
                f'CFG_CPU_EXPERT_BITS="{bits}"',
                f'CFG_ATTENTION_QUANT="bf16"',
                f'CFG_SHARED_EXPERT_QUANT="int8"',
                f'CFG_DENSE_MLP_QUANT="int8"',
                f'CFG_LM_HEAD_QUANT="int8"',
                f'CFG_HOST="0.0.0.0"',
                f'CFG_PORT="{DEFAULT_PORT}"',
                f'CFG_NUM_GPUS="1"',
            ]
            fd, config_path = tempfile.mkstemp(prefix="krasis-cache-", suffix=".conf")
            with os.fdopen(fd, "w") as f:
                f.write("\n".join(lines) + "\n")

            krasis = find_krasis_command()
            cmd = [krasis, "--config", config_path, "--build-cache"]
            if force:
                cmd.append("--force-rebuild-cache")

            log_file = open(log_path, "w")
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )

            # Wait for BUILD CACHE COMPLETE
            start = time.time()
            success = False
            while time.time() - start < CACHE_BUILD_TIMEOUT:
                if proc.poll() is not None:
                    # Process exited — check if it succeeded
                    try:
                        with open(log_path) as f:
                            if "BUILD CACHE COMPLETE" in f.read():
                                success = True
                    except (OSError, IOError):
                        pass
                    break
                try:
                    with open(log_path) as f:
                        if "BUILD CACHE COMPLETE" in f.read():
                            success = True
                            break
                except (OSError, IOError):
                    pass
                time.sleep(HEALTH_POLL_INTERVAL)

            if success:
                ok(f"INT{bits} cache built successfully")
                results[bits] = True
            else:
                if proc.poll() is not None:
                    warn(f"INT{bits} cache build failed (exit code {proc.returncode})")
                else:
                    warn(f"INT{bits} cache build timed out after {CACHE_BUILD_TIMEOUT}s")
                results[bits] = False

        except Exception as e:
            warn(f"INT{bits} cache build error: {e}")
            results[bits] = False

        finally:
            if proc and proc.poll() is None:
                kill_server(proc)
            try:
                log_file.close()
            except Exception:
                pass
            if config_path and os.path.isfile(config_path):
                os.unlink(config_path)
            # Wait for GPU to clear
            wait_for_gpu_clear(gpu_idx, timeout=60)

    return results


# ═══════════════════════════════════════════════════════════════════
# AWQ Template Building (Pre-flight)
# ═══════════════════════════════════════════════════════════════════

def awq_template_exists(model_name: str) -> bool:
    """Check if an AWQ calibration template exists for this model.

    Checks both the bundled templates directory and user cache.
    """
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_path):
        return False

    # Import compute_model_hash to get the model's hash
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(script_dir, "python"))
    from krasis.awq_calibrate import compute_model_hash

    model_hash = compute_model_hash(model_path)

    # Check bundled templates
    bundled = os.path.join(script_dir, "templates", "attention", model_hash, "template.json")
    if os.path.isfile(bundled):
        return True

    # Check user cache
    user_cache = os.path.expanduser(f"~/.krasis/templates/{model_hash}/template.json")
    if os.path.isfile(user_cache):
        return True

    return False


def build_awq_template(model_name: str, gpu_idx: int, num_layers: int,
                       log_dir: str) -> bool:
    """Build AWQ calibration template for a model if it doesn't exist.

    Runs ./dev awq-calibrate with a temporary config. Returns True on success.
    """
    if awq_template_exists(model_name):
        ok("AWQ template already exists — skipping calibration")
        return True

    info("Building AWQ template (this may take several minutes)...")
    config_path = None
    proc = None
    log_path = os.path.join(log_dir, "awq_calibrate.log")

    try:
        # Generate a minimal config for AWQ calibration (always uses BF16 attention)
        model_path = os.path.join(MODELS_DIR, model_name)
        lines = [
            f"# AWQ calibration config",
            f'MODEL_PATH="{model_path}"',
            f'CFG_SELECTED_GPUS="{gpu_idx}"',
            f'CFG_PP_PARTITION="{num_layers}"',
            f'CFG_LAYER_GROUP_SIZE="2"',
            f'CFG_KV_DTYPE="polar4"',
            f'CFG_GPU_EXPERT_BITS="4"',
            f'CFG_CPU_EXPERT_BITS="4"',
            f'CFG_ATTENTION_QUANT="bf16"',
            f'CFG_SHARED_EXPERT_QUANT="int8"',
            f'CFG_DENSE_MLP_QUANT="int8"',
            f'CFG_LM_HEAD_QUANT="int8"',
            f'CFG_HOST="0.0.0.0"',
            f'CFG_PORT="{DEFAULT_PORT}"',
            f'CFG_NUM_GPUS="1"',
        ]
        fd, config_path = tempfile.mkstemp(prefix="krasis-awq-", suffix=".conf")
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines) + "\n")

        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmd = ["./dev", "awq-calibrate", config_path]
        child_env = os.environ.copy()
        child_env["KRASIS_SKIP_GPU_CLEANUP"] = "1"

        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            cwd=script_dir,
            env=child_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        # Wait for completion (AWQ calibrate exits when done, unlike server)
        start = time.time()
        while time.time() - start < AWQ_CALIBRATE_TIMEOUT:
            if proc.poll() is not None:
                break
            time.sleep(HEALTH_POLL_INTERVAL)

        if proc.poll() is None:
            # Timed out
            warn(f"AWQ calibration timed out after {AWQ_CALIBRATE_TIMEOUT}s")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            return False

        if proc.returncode != 0:
            warn(f"AWQ calibration failed (exit code {proc.returncode})")
            # Show last few lines of log
            try:
                with open(log_path) as f:
                    content = f.read()
                tail = content[-2000:] if len(content) > 2000 else content
                warn(f"Log tail:\n{tail}")
            except (OSError, IOError):
                pass
            return False

        # Verify template was actually created
        if awq_template_exists(model_name):
            ok("AWQ template built successfully")
            return True
        else:
            warn("AWQ calibration completed but template not found")
            return False

    except Exception as e:
        warn(f"AWQ calibration error: {e}")
        return False

    finally:
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        try:
            log_file.close()
        except Exception:
            pass
        if config_path and os.path.isfile(config_path):
            os.unlink(config_path)
        # Wait for GPU to clear after calibration
        wait_for_gpu_clear(gpu_idx, timeout=60)


# ═══════════════════════════════════════════════════════════════════
# Server Management
# ═══════════════════════════════════════════════════════════════════

def find_krasis_command() -> str:
    """Find the installed krasis command. Fails if not found.

    Prefers the conda env krasis because that's the environment with GPU
    dependencies (PyTorch, sgl-kernel, etc). The ~/.local/bin/krasis may
    point to a bare venv without GPU deps.
    """
    # Prefer the conda env — this is the real production install
    conda_path = os.path.expanduser("~/miniconda3/envs/ktransformers/bin/krasis")
    if os.path.isfile(conda_path):
        return conda_path
    # Fall back to PATH
    path = shutil.which("krasis")
    if path:
        return path
    die("'krasis' command not found. Is Krasis installed? (pip install -e .)")


def launch_server(config_path: str, log_file, env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    """Launch krasis --config <path> --benchmark as a background process.

    Returns the Popen handle. stdout/stderr go to log_file.
    """
    krasis = find_krasis_command()
    cmd = [krasis, "--config", config_path, "--benchmark", "--vram-report"]

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=env,
    )
    return proc


def wait_for_benchmark_complete(log_path: str, proc: subprocess.Popen,
                                 timeout: int = BENCHMARK_TIMEOUT) -> bool:
    """Wait for BENCHMARK COMPLETE in the log. Returns True if found."""
    start = time.time()
    while time.time() - start < timeout:
        # Check if process died
        if proc.poll() is not None:
            return False
        try:
            with open(log_path) as f:
                if "BENCHMARK COMPLETE" in f.read():
                    return True
        except (OSError, IOError):
            pass
        time.sleep(HEALTH_POLL_INTERVAL)
    return False


def wait_for_health(port: int = DEFAULT_PORT, timeout: int = 30) -> bool:
    """Wait for the server health endpoint to respond."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            conn = http.client.HTTPConnection(DEFAULT_HOST, port, timeout=2)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            if resp.status == 200:
                conn.close()
                return True
            conn.close()
        except Exception:
            pass
        time.sleep(1)
    return False


def kill_server(proc: subprocess.Popen):
    """Kill the server process group cleanly."""
    if proc.poll() is not None:
        return

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass

    # Wait up to 10s for graceful exit
    for _ in range(10):
        if proc.poll() is not None:
            return
        time.sleep(1)

    # Force kill
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def kill_stale_krasis_processes():
    """Kill any lingering krasis Python processes (excluding ourselves)."""
    our_pid = os.getpid()
    try:
        result = subprocess.run(
            ["pgrep", "-f", "python.*krasis\\."],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return  # No matches

        pids = [int(p.strip()) for p in result.stdout.strip().split("\n")
                if p.strip() and int(p.strip()) != our_pid]

        if not pids:
            return

        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass

        # Wait up to 5s for graceful exit
        for _ in range(5):
            alive = [p for p in pids if _pid_alive(p)]
            if not alive:
                return
            time.sleep(1)

        # SIGKILL survivors
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass


def _pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def wait_for_gpu_clear(gpu_idx: int, timeout: int = 60) -> bool:
    """Wait for GPU memory to drop below 500 MB used.

    If memory doesn't clear within half the timeout, kills any stale
    krasis processes that may be holding GPU memory.
    """
    killed_stale = False
    for i in range(timeout):
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id={gpu_idx}", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                used = int(result.stdout.strip())
                if used < 500:
                    return True
        except (subprocess.TimeoutExpired, ValueError):
            pass

        # If we're past half the timeout and still stuck, kill stale processes
        if i == timeout // 2 and not killed_stale:
            warn("GPU not clearing — killing stale krasis processes...")
            kill_stale_krasis_processes()
            killed_stale = True

        time.sleep(1)
    return False


# ═══════════════════════════════════════════════════════════════════
# HTTP Client (streaming with TTFT measurement)
# ═══════════════════════════════════════════════════════════════════

def send_chat_streaming(prompt: str, port: int = DEFAULT_PORT,
                        max_tokens: int = 256, temperature: float = 0.3,
                        timeout: int = 120,
                        enable_thinking: bool = False) -> Dict:
    """Send a streaming chat completion request.

    Returns dict with: text, ttft_s, total_s, tokens, decode_tok_s,
    prefill_tok_s, prompt_tokens, overhead, finish_reason, error.

    Speed numbers come from krasis_timing (internal engine metrics), not
    from network-level measurement which suffers from TCP buffering artifacts.
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "enable_thinking": enable_thinking,
    }
    body = json.dumps(payload).encode()

    try:
        conn = http.client.HTTPConnection(DEFAULT_HOST, port, timeout=timeout)
        t_start = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()

        if resp.status != 200:
            error_body = resp.read().decode("utf-8", errors="replace")
            conn.close()
            return {"text": "", "ttft_s": 0, "total_s": 0, "tokens": 0,
                    "decode_tok_s": 0, "prefill_tok_s": 0, "prompt_tokens": 0,
                    "overhead": {}, "error": f"HTTP {resp.status}: {error_body[:200]}"}

        t_first = None
        token_count = 0
        text_parts = []
        finish_reason = None
        timing_data = None
        buf = b""

        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                line, buf = buf.split(b"\n\n", 1)
                line = line.strip()
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    try:
                        data = json.loads(line[6:])
                        if "krasis_timing" in data:
                            timing_data = data["krasis_timing"]
                            continue
                        choice = data.get("choices", [{}])[0]
                        fr = choice.get("finish_reason")
                        if fr is not None:
                            finish_reason = fr
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if t_first is None:
                                t_first = time.perf_counter()
                            token_count += 1
                            text_parts.append(content)
                    except json.JSONDecodeError:
                        pass

        conn.close()
        t_end = time.perf_counter()

        total_s = t_end - t_start
        ttft_s = (t_first - t_start) if t_first else total_s
        text = "".join(text_parts)

        # Use internal engine metrics from krasis_timing SSE event
        decode_tok_s = 0.0
        prefill_tok_s = 0.0
        prompt_tokens = 0
        overhead = {}
        if timing_data:
            decode_tok_s = timing_data.get("decode_tok_s", 0.0)
            prompt_tokens = timing_data.get("prompt_tokens", 0)
            overhead = timing_data.get("overhead", {})
            prefill_ms = overhead.get("prefill_ms", 0.0)
            if prefill_ms > 0 and prompt_tokens > 0:
                prefill_tok_s = prompt_tokens / (prefill_ms / 1000.0)

        return {
            "text": text,
            "ttft_s": round(ttft_s, 2),
            "total_s": round(total_s, 2),
            "tokens": token_count,
            "decode_tok_s": round(decode_tok_s, 2),
            "prefill_tok_s": round(prefill_tok_s, 1),
            "prompt_tokens": prompt_tokens,
            "overhead": overhead,
            "finish_reason": finish_reason,
            "error": None,
        }

    except Exception as e:
        return {"text": "", "ttft_s": 0, "total_s": 0, "tokens": 0,
                "decode_tok_s": 0, "prefill_tok_s": 0, "prompt_tokens": 0,
                "overhead": {}, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# Benchmark Extraction
# ═══════════════════════════════════════════════════════════════════

def extract_benchmark_section(log_path: str) -> str:
    """Extract the BENCHMARK COMPLETE summary block from the server log, ANSI-stripped.

    Only captures the final summary (model, config, speeds, HCS) — not the full
    benchmark run output with warmup, CUDA graph captures, etc.
    """
    with open(log_path) as f:
        content = f.read()

    lines = content.split("\n")
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        # Look for the "─" separator line immediately before BENCHMARK COMPLETE
        if "BENCHMARK COMPLETE" in line:
            # Walk backward to find the separator line (─────)
            for j in range(i - 1, max(0, i - 3), -1):
                if "─" in lines[j] or "────" in lines[j]:
                    start_idx = j
                    break
            if start_idx is None:
                start_idx = i
            # Walk forward to capture through the summary (ends at blank line or Log: line)
            for j in range(i + 1, min(len(lines), i + 15)):
                end_idx = j + 1
                if lines[j].strip() == "" and j > i + 3:
                    break
            break

    if start_idx is not None and end_idx is not None:
        section = "\n".join(lines[start_idx:end_idx])
        return strip_ansi(section)
    return "(benchmark output not found in log)"


def extract_vram_report(run_dir: str) -> Optional[List[Dict]]:
    """Extract VRAM report events from <run_dir>/vram_report.csv.

    Returns list of dicts: [{"event": str, "time_s": float, "gpus": [free_mb, ...]}, ...]
    Only includes event rows (not periodic samples). Returns None if CSV not found.
    """
    csv_path = os.path.join(run_dir, "vram_report.csv")
    if not os.path.isfile(csv_path):
        return None

    try:
        with open(csv_path) as f:
            lines = f.read().strip().split("\n")

        if len(lines) < 2:
            return None

        # Parse header to find GPU columns
        header = lines[0].split(",")
        gpu_cols = [i for i, h in enumerate(header) if h.startswith("gpu") and h.endswith("_free_mb")]

        events = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            event = parts[1].strip()
            if not event:
                continue  # Skip periodic samples
            ts_ms = int(parts[0])
            gpu_free = [int(parts[i]) if i < len(parts) else 0 for i in gpu_cols]
            events.append({
                "event": event,
                "time_s": ts_ms / 1000.0,
                "gpus": gpu_free,
            })

        return events if events else None
    except (OSError, IOError, ValueError):
        return None


def extract_safety_margin(log_path: str) -> Optional[int]:
    """Extract HCS safety margin from benchmark log output.

    Looks for: 'safety margin: NNN MB' in the log.
    Returns margin in MB, or None if not found.
    """
    try:
        with open(log_path) as f:
            for line in f:
                if "safety margin:" in line:
                    import re
                    m = re.search(r'safety margin:\s*(\d+)\s*MB', line)
                    if m:
                        return int(m.group(1))
        return None
    except (OSError, IOError):
        return None


def extract_server_error(log_path: str, max_chars: int = 3000) -> str:
    """Extract the tail of the server log for error reporting."""
    try:
        with open(log_path) as f:
            content = f.read()
        tail = content[-max_chars:] if len(content) > max_chars else content
        return strip_ansi(tail)
    except (OSError, IOError):
        return "(could not read log file)"


# ═══════════════════════════════════════════════════════════════════
# Test Phases
# ═══════════════════════════════════════════════════════════════════

def run_sanity_prompts(port: int = DEFAULT_PORT,
                       enable_thinking: bool = False) -> List[Dict]:
    """Phase 2: Run sanity test prompts via krasis chat sanity test.

    Uses the same multi-turn conversation engine as 'krasis sanity',
    which properly maintains conversation history for continuation prompts.
    """
    from krasis.chat import run_sanity_test

    server = {
        "host": DEFAULT_HOST,
        "port": port,
        "url": f"http://{DEFAULT_HOST}:{port}",
        "model": "unknown",
        "status": "ok",
    }
    # Try to fetch model name
    try:
        import urllib.request
        req = urllib.request.Request(f"http://{DEFAULT_HOST}:{port}/v1/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models_data = data.get("data", [])
            if models_data:
                server["model"] = models_data[0].get("id", "unknown")
    except Exception:
        pass

    # Thinking variant: use explicit enable_thinking=True, bump max_tokens
    # for longer answers (thinking tokens are exempt from max_tokens on the server,
    # but code answers like quicksort need more room)
    max_tokens = 512 if enable_thinking else 256
    thinking_arg = True if enable_thinking else None  # None = server default (off)

    return run_sanity_test(server, temperature=0.3, max_tokens=max_tokens,
                           enable_thinking=thinking_arg)


def run_large_prompt_tests(port: int = DEFAULT_PORT,
                           enable_thinking: bool = False) -> List[Dict]:
    """Phase 3: Run large prompt tests using canonical Gutenberg texts."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompts_dir = os.path.join(script_dir, "benchmarks", "prompts")

    if not os.path.isdir(prompts_dir):
        warn(f"Prompts directory not found: {prompts_dir}")
        return []

    prompt_files = sorted(f for f in os.listdir(prompts_dir) if f.endswith(".txt"))
    if not prompt_files:
        warn(f"No prompt files in {prompts_dir}")
        return []

    # (name, target_chars, file_index, question, timeout)
    tests = [
        ("~2K tokens",  7_000,  0,
         "Summarize the above text in one paragraph.", 120),
        ("~10K tokens", 35_000, 1,
         "What are the main themes in the text above?", 300),
        ("~25K tokens", 87_500, 2,
         "Summarize the key events described in the text above.", 600),
    ]

    results = []
    for name, target_chars, idx, question, timeout in tests:
        fname = prompt_files[idx % len(prompt_files)]
        path = os.path.join(prompts_dir, fname)
        with open(path) as f:
            text = f.read()
        if len(text) > target_chars:
            text = text[:target_chars]

        prompt = f"{text}\n\n{question}"
        info(f"  {name} ({len(prompt):,} chars, {fname})")

        r = send_chat_streaming(prompt, port=port, max_tokens=256, timeout=timeout,
                                enable_thinking=enable_thinking)
        r["prompt_name"] = name
        r["prompt_chars"] = len(prompt)
        r["prompt_preview"] = prompt[:200]
        r["book"] = fname
        results.append(r)

        if r["error"]:
            warn(f"    Error: {r['error']}")
        else:
            preview = r["text"][:120].replace("\n", " ")
            decode_str = f"decode {r['decode_tok_s']:.1f} tok/s" if r['decode_tok_s'] > 0 else "decode Not Received"
            prefill_str = f"prefill {r['prefill_tok_s']:.0f} tok/s" if r['prefill_tok_s'] > 0 else ""
            parts = [f"TTFT {r['ttft_s']:.1f}s"]
            if prefill_str:
                parts.append(prefill_str)
            parts.append(decode_str)
            parts.append(f"{r['total_s']:.1f}s")
            ok(f"    {' | '.join(parts)} — {preview}")

    return results


def run_reference_validation(config_path: str, port: int = DEFAULT_PORT) -> Dict:
    """Run a short reference-output validation pass against the live server."""
    from validate_model import validate_model

    return validate_model(
        config_path,
        no_server=True,
        port=port,
        enable_logprobs=False,
        enable_prefill=False,
        max_prompts=REFERENCE_VALIDATE_MAX_PROMPTS,
        return_summary=True,
    )


# ═══════════════════════════════════════════════════════════════════
# Sanity Validation
# ═══════════════════════════════════════════════════════════════════

# Each validator returns a list of (severity, message) tuples.
# severity: "FAIL" (wrong answer), "WARN" (quality issue), "PASS" (ok)

def _detect_repetition(text: str) -> List[Tuple[str, str]]:
    """Detect repeated n-grams (3+ word phrases repeated 3+ times)."""
    issues = []
    words = text.split()
    if len(words) < 9:
        return issues

    for n in (5, 4, 3):
        counts: Dict[str, int] = {}
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i:i + n])
            gram_lower = gram.lower()
            counts[gram_lower] = counts.get(gram_lower, 0) + 1

        worst = max(counts.values()) if counts else 0
        if worst >= 3:
            worst_gram = max(counts, key=counts.get)
            issues.append(("WARN", f"Repetition: \"{worst_gram}\" repeated {worst}x"))
            break  # Only report the longest repeating pattern

    return issues


def _detect_echo(prompt: str, response: str) -> List[Tuple[str, str]]:
    """Detect if response is just the prompt echoed back."""
    issues = []
    prompt_clean = prompt.strip().lower()
    response_clean = response.strip().lower()
    if len(response_clean) > 0 and len(prompt_clean) > 10:
        # Check if response starts with 80%+ of the prompt
        overlap = min(len(prompt_clean), len(response_clean))
        if overlap > 0 and response_clean[:overlap] == prompt_clean[:overlap]:
            if overlap >= len(prompt_clean) * 0.8:
                issues.append(("WARN", "Response echoes the prompt"))
    return issues


def _check_min_length(response: str, min_chars: int = 10) -> List[Tuple[str, str]]:
    """Check response meets minimum length."""
    if len(response.strip()) < min_chars:
        return [("WARN", f"Very short response ({len(response.strip())} chars)")]
    return []


# Prompt-specific validators: keyed by substring match on the prompt text.
# Each entry: (prompt_substring, validator_function)
# Validators take the response text and return list of (severity, message).

def _check_math_2plus2(response: str) -> List[Tuple[str, str]]:
    """2+2 should contain '4'."""
    if "4" not in response:
        return [("FAIL", "Expected answer '4' not found")]
    return []


def _check_math_multiply_10(response: str) -> List[Tuple[str, str]]:
    """'multiply that by 10' (following 2+2=4) should contain '40'."""
    if "40" not in response:
        return [("FAIL", "Expected answer '40' not found")]
    return []


def _check_math_divide_5(response: str) -> List[Tuple[str, str]]:
    """'divide the result by 5' (following 40) should contain '8'."""
    if "8" not in response:
        return [("FAIL", "Expected answer '8' not found")]
    return []


def _check_largest_animal(response: str) -> List[Tuple[str, str]]:
    """Should mention blue whale."""
    lower = response.lower()
    if "whale" not in lower:
        return [("FAIL", "Expected 'whale' not found — wrong answer")]
    return []


def _check_largest_water(response: str) -> List[Tuple[str, str]]:
    """Should mention Pacific Ocean."""
    lower = response.lower()
    if "pacific" not in lower and "ocean" not in lower:
        return [("FAIL", "Expected 'Pacific' or 'ocean' not found")]
    return []


def _check_binary_chop(response: str) -> List[Tuple[str, str]]:
    """Should be a substantive explanation mentioning search/binary/divide."""
    issues = []
    lower = response.lower()
    if len(response.strip()) < 200:
        issues.append(("WARN", f"Short explanation ({len(response.strip())} chars) for in-depth request"))
    keywords = ["search", "binary", "divid", "half", "midpoint", "sorted"]
    if not any(kw in lower for kw in keywords):
        issues.append(("FAIL", "Missing key terms (search/binary/divide/half) — likely off-topic"))
    return issues


def _check_towel_trick(response: str) -> List[Tuple[str, str]]:
    """Towels dry in parallel — answer is 4 hours, NOT 20 hours."""
    issues = []
    lower = response.lower()
    if "20 hour" in lower or "20hour" in lower:
        issues.append(("FAIL", "Says 20 hours — fell for the trick (answer is 4 hours)"))
    elif "4 hour" in lower or "4hour" in lower or "four hour" in lower:
        pass  # Correct
    elif "4" not in response:
        issues.append(("WARN", "Answer '4' not found — may have the wrong answer"))
    return issues


def _check_rust_quicksort(response: str) -> List[Tuple[str, str]]:
    """Should contain Rust code with fn and sorting-related terms."""
    issues = []
    if "fn " not in response and "fn(" not in response:
        issues.append(("FAIL", "No Rust function definition ('fn') found"))
    lower = response.lower()
    sort_terms = ["sort", "quicksort", "quick_sort", "partition", "pivot"]
    if not any(t in lower for t in sort_terms):
        issues.append(("FAIL", "No sorting-related terms found"))
    return issues


def _check_blue_whale_facts(response: str) -> List[Tuple[str, str]]:
    """Should mention whales."""
    if "whale" not in response.lower():
        return [("FAIL", "Expected 'whale' not found — off-topic response")]
    return []


def _check_whales_general(response: str) -> List[Tuple[str, str]]:
    """Should mention whales."""
    if "whale" not in response.lower():
        return [("FAIL", "Expected 'whale' not found — off-topic response")]
    return []


def _check_whales_geography(response: str) -> List[Tuple[str, str]]:
    """Should mention locations/oceans."""
    lower = response.lower()
    geo_terms = ["ocean", "pacific", "atlantic", "arctic", "antarctic", "sea",
                 "coast", "tropical", "temperate", "hemisphere", "water"]
    if not any(t in lower for t in geo_terms):
        return [("WARN", "No geographic terms found — may be off-topic")]
    return []


# Map prompt substrings to their specific validators
PROMPT_VALIDATORS = [
    ("what is 2+2", _check_math_2plus2),
    ("multiply that by 10", _check_math_multiply_10),
    ("divide the result by 5", _check_math_divide_5),
    ("largest animal", _check_largest_animal),
    ("largest body of water", _check_largest_water),
    ("binary chop", _check_binary_chop),
    ("towels to dry", _check_towel_trick),
    ("quicksort implementation in rust", _check_rust_quicksort),
    ("facts about the blue whale", _check_blue_whale_facts),
    ("more about whales in general", _check_whales_general),
    ("whales live geographically", _check_whales_geography),
]


def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks from response text.

    When thinking is enabled, the SSE stream includes thinking content wrapped
    in <think>...</think> tags. Validators should check the answer portion only.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def validate_sanity_result(prompt: str, response: str) -> List[Tuple[str, str]]:
    """Run all applicable validators on a sanity test result.

    Returns list of (severity, message) tuples. Empty list = PASS.
    Strips <think>...</think> blocks before validation so thinking content
    doesn't interfere with answer checking.
    """
    if not response or not response.strip():
        return [("FAIL", "Empty response")]

    # Strip thinking content — validate only the answer
    response = _strip_thinking(response)
    if not response:
        return [("FAIL", "Empty response after stripping thinking block")]

    issues = []

    # Universal checks
    issues.extend(_detect_repetition(response))
    issues.extend(_detect_echo(prompt, response))
    issues.extend(_check_min_length(response))

    # Prompt-specific checks
    prompt_lower = prompt.lower()
    for substring, validator in PROMPT_VALIDATORS:
        if substring in prompt_lower:
            issues.extend(validator(response))

    return issues


def validation_status(issues: List[Tuple[str, str]]) -> str:
    """Return overall status from a list of issues: FAIL > WARN > PASS."""
    if any(sev == "FAIL" for sev, _ in issues):
        return "FAIL"
    if any(sev == "WARN" for sev, _ in issues):
        return "WARN"
    return "PASS"


# ═══════════════════════════════════════════════════════════════════
# Perplexity Evaluation
# ═══════════════════════════════════════════════════════════════════

def run_perplexity_test(config_path: str, log_dir: str, timestamp: str,
                        variant_idx: int) -> Optional[Dict]:
    """Run perplexity evaluation for a config after server shutdown.

    Runs measure_ppl as a subprocess (loads model independently).
    Returns parsed results dict or None on failure.
    """
    log_path = os.path.join(log_dir, f"perplexity_config{variant_idx + 1}_{timestamp}.log")

    cmd = [
        sys.executable, "-m", "perplexity.measure_ppl",
        "--config", config_path,
        "--max-tokens", str(PERPLEXITY_MAX_TOKENS),
    ]

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    try:
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=script_dir,
            )

            # Poll with timeout
            start = time.time()
            while proc.poll() is None:
                elapsed = time.time() - start
                if elapsed > PERPLEXITY_TIMEOUT:
                    warn(f"Perplexity timed out after {PERPLEXITY_TIMEOUT}s")
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        proc.kill()
                    proc.wait(timeout=10)
                    return None
                time.sleep(5)

        if proc.returncode != 0:
            warn(f"Perplexity failed (exit code {proc.returncode})")
            try:
                with open(log_path) as f:
                    content = f.read()
                tail = content[-2000:] if len(content) > 2000 else content
                warn(f"Log tail:\n{tail}")
            except (OSError, IOError):
                pass
            return None

        with open(log_path) as f:
            output = f.read()

        result = _parse_perplexity_output(output, log_path)
        if result:
            ok(f"Perplexity: {result['perplexity']:.2f} "
               f"({result.get('tokens_scored', '?'):,} tokens, "
               f"{result.get('elapsed_s', 0):.0f}s)")
        else:
            warn("Could not parse perplexity from output")
        return result

    except Exception as e:
        warn(f"Perplexity error: {e}")
        return None


def _parse_perplexity_output(output: str, log_path: str) -> Optional[Dict]:
    """Parse perplexity results from measure_ppl output."""
    result = {}

    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("Perplexity:") and "COMPLETE" not in line:
            try:
                result["perplexity"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("BPC:"):
            try:
                result["bpc"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("Tokens scored:"):
            try:
                parts = line.split(":")[1].strip()
                result["tokens_scored"] = int(parts.split("/")[0].strip().replace(",", ""))
            except (ValueError, IndexError):
                pass
        elif line.startswith("Elapsed:"):
            try:
                elapsed_match = re.search(r'([\d.]+)s', line)
                if elapsed_match:
                    result["elapsed_s"] = float(elapsed_match.group(1))
                tok_match = re.search(r'\((\d+)\s*tok/s\)', line)
                if tok_match:
                    result["throughput_tok_s"] = int(tok_match.group(1))
            except (ValueError, AttributeError):
                pass

    if "perplexity" not in result:
        return None

    result["log_path"] = log_path
    return result


def load_perplexity_baselines(output_dir: str) -> Dict:
    """Load perplexity baselines from JSON file."""
    path = os.path.join(output_dir, PERPLEXITY_BASELINES_FILE)
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_perplexity_baselines(output_dir: str, baselines: Dict):
    """Save perplexity baselines to JSON file."""
    path = os.path.join(output_dir, PERPLEXITY_BASELINES_FILE)
    with open(path, "w") as f:
        json.dump(baselines, f, indent=2)


def check_perplexity_baseline(baselines: Dict, model_name: str, variant_name: str,
                               ppl: float) -> Tuple[str, Optional[float], Optional[float]]:
    """Compare perplexity against baseline.

    Returns (status, baseline_ppl, delta_pct).
    Status is one of: NEW (no baseline), PASS, WARN, FAIL.
    """
    model_baselines = baselines.get(model_name, {})
    baseline = model_baselines.get(variant_name)

    if baseline is None:
        return ("NEW", None, None)

    baseline_ppl = baseline["perplexity"]
    delta_pct = (ppl - baseline_ppl) / baseline_ppl

    if delta_pct > PERPLEXITY_FAIL_THRESHOLD:
        return ("FAIL", baseline_ppl, delta_pct)
    elif delta_pct > PERPLEXITY_WARN_THRESHOLD:
        return ("WARN", baseline_ppl, delta_pct)
    else:
        return ("PASS", baseline_ppl, delta_pct)


# ═══════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════

def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _generate_vram_svg(key_events: List[Dict], config_name: str, gpu_idx: int = 0) -> str:
    """Generate an inline SVG chart of VRAM usage over time for one config.

    Shows free VRAM as a line, with colored background bands for prefill (blue),
    decode (green), and HCS soft load (amber) phases. Includes a legend/key.
    """
    if not key_events:
        return ""

    # Extract GPU data points
    times = [ev["time_s"] for ev in key_events]
    vram_vals = [ev["gpus"][gpu_idx] if gpu_idx < len(ev["gpus"]) else 0 for ev in key_events]

    if not times:
        return ""

    t_min = min(times)
    t_max = max(times)
    v_min = 0
    v_max = max(vram_vals) * 1.1  # 10% headroom

    if t_max <= t_min:
        return ""
    if v_max <= 0:
        v_max = 1000

    # Chart dimensions
    width = 900
    height = 300
    margin_l = 70
    margin_r = 20
    margin_t = 30
    margin_b = 50
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def tx(t):
        return margin_l + (t - t_min) / (t_max - t_min) * plot_w

    def ty(v):
        return margin_t + plot_h - (v / v_max) * plot_h

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height + 40}" '
                 f'style="width:100%;max-width:{width}px;background:#1a1a2e;border-radius:8px;margin:10px 0">')

    # Phase colors
    phase_colors = {
        "prefill": ("rgba(59,130,246,0.25)", "#3b82f6", "Prefill"),
        "decode": ("rgba(34,197,94,0.25)", "#22c55e", "Decode"),
        "hcs_soft_load": ("rgba(251,191,36,0.25)", "#fbbf24", "HCS Soft Load"),
    }

    # Draw phase bands
    i = 0
    while i < len(key_events):
        ev = key_events[i]
        name = ev["event"]
        for phase, (fill, _stroke, _label) in phase_colors.items():
            if name == f"{phase}_start":
                # Find matching end
                end_ev = None
                for j in range(i + 1, len(key_events)):
                    if key_events[j]["event"] == f"{phase}_end":
                        end_ev = key_events[j]
                        break
                if end_ev:
                    x1 = tx(ev["time_s"])
                    x2 = tx(end_ev["time_s"])
                    if x2 - x1 < 1:
                        x2 = x1 + 1  # minimum visible width
                    parts.append(f'<rect x="{x1:.1f}" y="{margin_t}" '
                                 f'width="{x2 - x1:.1f}" height="{plot_h}" '
                                 f'fill="{fill}"/>')
        i += 1

    # Draw axes
    parts.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" '
                 f'y2="{margin_t + plot_h}" stroke="#555" stroke-width="1"/>')
    parts.append(f'<line x1="{margin_l}" y1="{margin_t + plot_h}" '
                 f'x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#555" stroke-width="1"/>')

    # Y-axis labels (VRAM MB)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        v = v_max * frac
        y = ty(v)
        label = f"{int(v):,}"
        parts.append(f'<text x="{margin_l - 5}" y="{y + 4}" text-anchor="end" '
                     f'fill="#999" font-size="11" font-family="monospace">{label}</text>')
        if frac > 0:
            parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{margin_l + plot_w}" '
                         f'y2="{y:.1f}" stroke="#333" stroke-width="0.5" stroke-dasharray="4,4"/>')

    # X-axis labels (time)
    duration = t_max - t_min
    if duration > 200:
        tick_step = 60
    elif duration > 100:
        tick_step = 30
    elif duration > 40:
        tick_step = 10
    else:
        tick_step = 5
    t = t_min
    while t <= t_max:
        x = tx(t)
        parts.append(f'<text x="{x:.1f}" y="{margin_t + plot_h + 18}" text-anchor="middle" '
                     f'fill="#999" font-size="11" font-family="monospace">{t:.0f}s</text>')
        parts.append(f'<line x1="{x:.1f}" y1="{margin_t + plot_h}" x2="{x:.1f}" '
                     f'y2="{margin_t + plot_h + 4}" stroke="#555" stroke-width="1"/>')
        t += tick_step

    # Axis titles
    parts.append(f'<text x="{margin_l + plot_w // 2}" y="{margin_t + plot_h + 38}" '
                 f'text-anchor="middle" fill="#ccc" font-size="12" font-family="sans-serif">Time</text>')
    parts.append(f'<text x="15" y="{margin_t + plot_h // 2}" text-anchor="middle" '
                 f'fill="#ccc" font-size="12" font-family="sans-serif" '
                 f'transform="rotate(-90,15,{margin_t + plot_h // 2})">Free VRAM (MB)</text>')

    # Draw VRAM line
    if len(times) > 1:
        line_points = " ".join(f"{tx(t):.1f},{ty(v):.1f}" for t, v in zip(times, vram_vals))
        parts.append(f'<polyline points="{line_points}" fill="none" stroke="#e2e8f0" '
                     f'stroke-width="1.5" stroke-linejoin="round"/>')

    # Draw data points
    for t, v in zip(times, vram_vals):
        parts.append(f'<circle cx="{tx(t):.1f}" cy="{ty(v):.1f}" r="2" fill="#e2e8f0"/>')

    # Legend
    legend_y = height + 8
    legend_items = [
        ("#3b82f6", "rgba(59,130,246,0.5)", "Prefill"),
        ("#22c55e", "rgba(34,197,94,0.5)", "Decode"),
        ("#fbbf24", "rgba(251,191,36,0.5)", "HCS Soft Load"),
        ("#e2e8f0", None, "Free VRAM"),
    ]
    lx = margin_l
    for stroke, fill, label in legend_items:
        if fill:
            parts.append(f'<rect x="{lx}" y="{legend_y}" width="16" height="12" '
                         f'fill="{fill}" stroke="{stroke}" stroke-width="1" rx="2"/>')
        else:
            parts.append(f'<line x1="{lx}" y1="{legend_y + 6}" x2="{lx + 16}" '
                         f'y2="{legend_y + 6}" stroke="{stroke}" stroke-width="2"/>')
            parts.append(f'<circle cx="{lx + 8}" cy="{legend_y + 6}" r="2.5" fill="{stroke}"/>')
        parts.append(f'<text x="{lx + 22}" y="{legend_y + 10}" fill="#ccc" font-size="11" '
                     f'font-family="sans-serif">{label}</text>')
        lx += len(label) * 7 + 45

    parts.append('</svg>')
    return "\n".join(parts)


def _extract_benchmark_numbers(benchmark_output: str) -> Dict:
    """Extract key numbers from benchmark output text for the summary dashboard."""
    nums = {}
    patterns = {
        "prefill": r"Prefill \(internal\):\s*([\d,]+\.?\d*)\s*tok/s",
        "decode": r"Decode \(internal\):\s*([\d,]+\.?\d*)\s*tok/s",
        "round_trip": r"Round trip \(network\):\s*([\d,]+\.?\d*)\s*tok/s",
        "hcs": r"HCS coverage:\s*[\d,]+/[\d,]+\s*\(([\d.]+)%\)",
        "min_free": r"Min free VRAM:\s*([\d,]+)\s*MB",
    }
    for key, pat in patterns.items():
        m = re.search(pat, benchmark_output)
        if m:
            nums[key] = m.group(1).replace(",", "")
    return nums


def generate_report(model_name: str, gpu_info: Tuple[int, str, int],
                    config_results: List[Dict]) -> str:
    """Generate an HTML release test report.

    Structure:
    1. Header with metadata
    2. Summary dashboard — all configs side-by-side in a colored table
    3. Benchmark details per config (collapsible)
    4. VRAM report — SVG chart with phase color bands + collapsible event table
    5. Sanity tests — collapsible per-config with full prompt/response
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Collect summary data ──────────────────────────────────────
    summary_rows = []
    for i, cr in enumerate(config_results):
        variant = cr["variant"]
        if cr.get("error"):
            summary_rows.append({
                "idx": i + 1, "name": variant["name"], "status": "FAILED",
                "error": cr["error"][:120],
            })
        else:
            nums = _extract_benchmark_numbers(cr.get("benchmark_output", ""))
            sanity = cr.get("sanity_results", [])
            # Run validation on each sanity result
            sanity_pass = 0
            sanity_warn = 0
            sanity_fail = 0
            for sr in sanity:
                if sr.get("error"):
                    sanity_fail += 1
                    sr["_validation"] = [("FAIL", sr["error"])]
                    sr["_validation_status"] = "FAIL"
                else:
                    issues = validate_sanity_result(sr.get("prompt", ""), sr.get("text", ""))
                    sr["_validation"] = issues
                    status = validation_status(issues)
                    sr["_validation_status"] = status
                    if status == "FAIL":
                        sanity_fail += 1
                    elif status == "WARN":
                        sanity_warn += 1
                    else:
                        sanity_pass += 1
            # Build summary string like "12 PASS, 1 WARN, 1 FAIL"
            parts = []
            if sanity_pass:
                parts.append(f"{sanity_pass} PASS")
            if sanity_warn:
                parts.append(f"{sanity_warn} WARN")
            if sanity_fail:
                parts.append(f"{sanity_fail} FAIL")
            sanity_summary = ", ".join(parts) if parts else f"0/{len(sanity)}"
            sanity_badge = "fail" if sanity_fail else ("warn" if sanity_warn else "pass")
            # Perplexity
            ppl_data = cr.get("perplexity")
            if ppl_data:
                ppl_str = f"{ppl_data['perplexity']:.2f}"
                bs = ppl_data.get("baseline_status", "NEW")
                if bs == "FAIL":
                    ppl_badge = "fail"
                elif bs == "WARN":
                    ppl_badge = "warn"
                elif bs == "NEW":
                    ppl_badge = "skip"
                else:
                    ppl_badge = "pass"
            else:
                ppl_str = "—"
                ppl_badge = "skip"
            validation = cr.get("reference_validation")
            if validation:
                if validation.get("fail_count", 0) > 0:
                    validate_badge = "fail"
                elif validation.get("warn_count", 0) > 0:
                    validate_badge = "warn"
                else:
                    validate_badge = "pass"
                validate_str = (f"{validation.get('match_count', 0)} MATCH, "
                                f"{validation.get('warn_count', 0)} WARN, "
                                f"{validation.get('fail_count', 0)} FAIL")
            else:
                validate_str = "—"
                validate_badge = "skip"
            summary_rows.append({
                "idx": i + 1, "name": variant["name"], "status": "PASS",
                "decode": nums.get("decode", "—"),
                "prefill": nums.get("prefill", "—"),
                "round_trip": nums.get("round_trip", "—"),
                "hcs": nums.get("hcs", "—"),
                "min_free": nums.get("min_free", "—"),
                "ppl": ppl_str,
                "ppl_badge": ppl_badge,
                "validate": validate_str,
                "validate_badge": validate_badge,
                "sanity": sanity_summary,
                "sanity_badge": sanity_badge,
            })

    # ── Start HTML ────────────────────────────────────────────────
    h = []
    h.append('<!DOCTYPE html>')
    h.append('<html lang="en"><head><meta charset="UTF-8">')
    h.append(f'<title>Release Test — {_html_escape(model_name)}</title>')
    h.append('<style>')
    h.append('''
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px 40px;
    line-height: 1.6;
}
h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
h2 { color: #79c0ff; margin-top: 32px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
h3 { color: #d2a8ff; }
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }
.meta { color: #8b949e; margin-bottom: 20px; }
.meta span { margin-right: 24px; }

/* Summary dashboard */
.dashboard { width: 100%; border-collapse: collapse; margin: 16px 0 32px; }
.dashboard th {
    background: #161b22; color: #8b949e; text-align: left;
    padding: 8px 12px; border-bottom: 2px solid #30363d; font-size: 13px;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.dashboard td { padding: 8px 12px; border-bottom: 1px solid #21262d; }
.dashboard tr:hover td { background: #161b22; }

/* Status badges */
.badge { padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; }
.badge-pass { background: #1a3a2a; color: #3fb950; }
.badge-fail { background: #3a1a1a; color: #f85149; }
.badge-warn { background: #3a2a1a; color: #d29922; }
.badge-skip { background: #1a1a2a; color: #8b949e; }

/* Collapsible sections */
details { margin: 8px 0; }
summary {
    cursor: pointer; padding: 8px 12px; background: #161b22;
    border: 1px solid #30363d; border-radius: 6px; color: #c9d1d9;
    font-weight: 600; user-select: none;
}
summary:hover { background: #1c2128; }
details[open] > summary { border-radius: 6px 6px 0 0; border-bottom: none; }
.detail-body {
    border: 1px solid #30363d; border-top: none; border-radius: 0 0 6px 6px;
    padding: 16px; background: #0d1117;
}

/* Benchmark output */
pre {
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 12px 16px; overflow-x: auto; font-size: 13px; color: #e6edf3;
}

/* Tables */
table.data { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 13px; }
table.data th {
    background: #161b22; color: #8b949e; text-align: left;
    padding: 6px 10px; border-bottom: 1px solid #30363d;
}
table.data td { padding: 6px 10px; border-bottom: 1px solid #21262d; }
table.data td.num { text-align: right; font-family: monospace; }

/* Sanity test cards */
.sanity-card { margin: 12px 0; padding: 12px 16px; background: #161b22; border-radius: 6px; border-left: 3px solid #30363d; }
.sanity-card.pass { border-left-color: #3fb950; }
.sanity-card.fail { border-left-color: #f85149; }
.sanity-card.warn { border-left-color: #d29922; }
.sanity-prompt { color: #d2a8ff; font-weight: 600; margin-bottom: 4px; }
.sanity-stats { color: #8b949e; font-size: 12px; margin-bottom: 8px; }
.sanity-response { color: #c9d1d9; white-space: pre-wrap; font-size: 13px; }
.sanity-issues { margin: 6px 0; padding: 6px 10px; border-radius: 4px; font-size: 12px; }
.sanity-issues.issues-fail { background: #3a1a1a; color: #f85149; }
.sanity-issues.issues-warn { background: #3a2a1a; color: #d29922; }
.turn-label { color: #8b949e; font-size: 11px; }

/* Warning box */
.warning { background: #2a1f0a; border: 1px solid #d29922; border-radius: 6px; padding: 10px 14px; color: #d29922; margin: 8px 0; }

/* Speed numbers */
.speed-good { color: #3fb950; }
.speed-ok { color: #d29922; }
.speed-bad { color: #f85149; }

/* TOC */
.toc { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 20px; margin: 16px 0; }
.toc ul { list-style: none; padding-left: 16px; margin: 4px 0; }
.toc > ul { padding-left: 0; }
''')
    h.append('</style></head><body>')

    # ── Header ────────────────────────────────────────────────────
    h.append(f'<h1>Krasis Release Test — {_html_escape(model_name)}</h1>')
    h.append('<div class="meta">')
    h.append(f'<span>Date: {now}</span>')
    h.append(f'<span>GPU: {_html_escape(gpu_info[1])} ({gpu_info[2]} MB)</span>')
    h.append(f'<span>Configs: {len(config_results)}</span>')
    h.append('</div>')

    # ── Table of Contents ─────────────────────────────────────────
    has_vram = any(cr.get("vram_report") for cr in config_results)
    h.append('<div class="toc"><ul>')
    h.append('<li><a href="#summary">Summary Dashboard</a></li>')
    h.append('<li><a href="#benchmarks">Benchmark Results</a>')
    h.append('<ul>')
    for i, cr in enumerate(config_results):
        variant = cr["variant"]
        status = "FAILED" if cr.get("error") else "PASS"
        badge_cls = "badge-fail" if cr.get("error") else "badge-pass"
        h.append(f'<li><a href="#config-{i+1}">{_html_escape(variant["name"])}</a> '
                 f'<span class="badge {badge_cls}">{status}</span></li>')
    h.append('</ul></li>')
    if has_vram:
        h.append('<li><a href="#vram">VRAM Report</a></li>')
    has_ppl = any(cr.get("perplexity") for cr in config_results)
    if has_ppl:
        h.append('<li><a href="#perplexity">Perplexity</a></li>')
    h.append('<li><a href="#sanity">Sanity Tests</a>')
    h.append('<ul>')
    for i, cr in enumerate(config_results):
        if cr.get("error"):
            continue
        variant = cr["variant"]
        sanity = cr.get("sanity_results", [])
        s_pass = sum(1 for s in sanity if s.get("_validation_status") == "PASS")
        s_warn = sum(1 for s in sanity if s.get("_validation_status") == "WARN")
        s_fail = sum(1 for s in sanity if s.get("_validation_status") == "FAIL")
        parts = []
        if s_pass:
            parts.append(f"{s_pass} PASS")
        if s_warn:
            parts.append(f"{s_warn} WARN")
        if s_fail:
            parts.append(f"{s_fail} FAIL")
        toc_summary = ", ".join(parts) if parts else f"0/{len(sanity)}"
        h.append(f'<li><a href="#sanity-{i+1}">{_html_escape(variant["name"])} ({toc_summary})</a></li>')
    h.append('</ul></li>')
    has_validation = any(cr.get("reference_validation") for cr in config_results)
    if has_validation:
        h.append('<li><a href="#reference-validation">Reference Validation</a></li>')
    h.append('</ul></div>')

    # ── Summary Dashboard ─────────────────────────────────────────
    h.append('<h2 id="summary">Summary Dashboard</h2>')
    h.append('<table class="dashboard"><thead><tr>')
    h.append('<th>#</th><th>Config</th><th>Status</th><th>Decode</th>'
             '<th>Prefill</th><th>Round Trip</th><th>HCS</th><th>Min Free</th>'
             '<th>PPL</th><th>Reference</th><th>Sanity</th>')
    h.append('</tr></thead><tbody>')
    for row in summary_rows:
        badge_cls = "badge-pass" if row["status"] == "PASS" else "badge-fail"
        if "decode" not in row:
            # Failed/skipped config
            h.append(f'<tr><td>{row["idx"]}</td><td>{_html_escape(row["name"])}</td>')
            h.append(f'<td><span class="badge {badge_cls}">{row["status"]}</span></td>')
            h.append(f'<td colspan="8" style="color:#8b949e">{_html_escape(row.get("error", ""))}</td></tr>')
        else:
            h.append(f'<tr><td>{row["idx"]}</td><td>{_html_escape(row["name"])}</td>')
            h.append(f'<td><span class="badge {badge_cls}">{row["status"]}</span></td>')
            h.append(f'<td class="num">{row["decode"]} tok/s</td>')
            h.append(f'<td class="num">{row["prefill"]} tok/s</td>')
            h.append(f'<td class="num">{row["round_trip"]} tok/s</td>')
            h.append(f'<td class="num">{row["hcs"]}%</td>')
            h.append(f'<td class="num">{row["min_free"]} MB</td>')
            pb = row.get("ppl_badge", "skip")
            h.append(f'<td><span class="badge badge-{pb}">{row.get("ppl", "—")}</span></td>')
            vb = row.get("validate_badge", "skip")
            h.append(f'<td><span class="badge badge-{vb}">{_html_escape(row.get("validate", "—"))}</span></td>')
            sb = row.get("sanity_badge", "pass")
            h.append(f'<td><span class="badge badge-{sb}">{row["sanity"]}</span></td></tr>')
    h.append('</tbody></table>')

    # ── Benchmark Results ─────────────────────────────────────────
    h.append('<h2 id="benchmarks">Benchmark Results</h2>')

    for i, cr in enumerate(config_results):
        variant = cr["variant"]
        status_text = "FAILED" if cr.get("error") else "PASS"
        badge_cls = "badge-fail" if cr.get("error") else "badge-pass"

        h.append(f'<h3 id="config-{i+1}">Config {i+1}: {_html_escape(variant["name"])} '
                 f'<span class="badge {badge_cls}">{status_text}</span></h3>')

        if cr.get("error"):
            h.append(f'<pre>{_html_escape(cr["error"])}</pre>')
            continue

        h.append(f'<pre>{_html_escape(cr.get("benchmark_output", "(no benchmark output)"))}</pre>')

        # Large prompt results
        large = cr.get("large_results", [])
        if large:
            h.append('<table class="data"><thead><tr>')
            h.append('<th>Size</th><th>Book</th><th>Decode</th><th>Prefill</th><th>TTFT</th>')
            h.append('</tr></thead><tbody>')
            for lr in large:
                book = _html_escape(lr.get('book', 'unknown'))
                if lr.get("error"):
                    h.append(f'<tr><td>{_html_escape(lr["prompt_name"])}</td>'
                             f'<td>{book}</td><td class="badge-fail">FAILED</td>'
                             f'<td>—</td><td>—</td></tr>')
                else:
                    decode_str = f"{lr['decode_tok_s']:.1f} tok/s" if lr.get('decode_tok_s', 0) > 0 else "N/A"
                    prefill_str = f"{lr['prefill_tok_s']:.0f} tok/s" if lr.get('prefill_tok_s', 0) > 0 else "N/A"
                    h.append(f'<tr><td>{_html_escape(lr["prompt_name"])}</td>'
                             f'<td>{book}</td><td class="num">{decode_str}</td>'
                             f'<td class="num">{prefill_str}</td>'
                             f'<td class="num">{lr["ttft_s"]:.2f}s</td></tr>')
            h.append('</tbody></table>')

    # ── VRAM Report ───────────────────────────────────────────────
    VRAM_KEY_EVENTS = {
        "prefill_start", "prefill_end",
        "hcs_soft_load_start", "hcs_soft_load_end",
        "decode_start", "decode_end",
    }

    if has_vram:
        h.append('<h2 id="vram">VRAM Report</h2>')
        h.append('<p style="color:#8b949e">Free VRAM (MB) at key lifecycle points. '
                 'Chart shows phase bands: '
                 '<span style="color:#3b82f6">&#9632; Prefill</span> '
                 '<span style="color:#22c55e">&#9632; Decode</span> '
                 '<span style="color:#fbbf24">&#9632; HCS Soft Load</span></p>')

        for i, cr in enumerate(config_results):
            vram = cr.get("vram_report")
            if not vram:
                continue

            key_events = [ev for ev in vram if ev["event"] in VRAM_KEY_EVENTS]
            if not key_events:
                continue

            variant = cr["variant"]
            num_gpus = len(key_events[0]["gpus"]) if key_events else 0

            h.append(f'<h3>Config {i+1}: {_html_escape(variant["name"])}</h3>')

            # SVG chart for each GPU
            for g in range(num_gpus):
                gpu_label = f"GPU{g}" if num_gpus > 1 else ""
                if gpu_label:
                    h.append(f'<p style="color:#8b949e;margin-bottom:0">{gpu_label}</p>')
                svg = _generate_vram_svg(key_events, variant["name"], gpu_idx=g)
                if svg:
                    h.append(svg)

            # Collapsible event table
            h.append(f'<details><summary>Event table ({len(key_events)} events)</summary>')
            h.append('<div class="detail-body">')
            hdr = '<table class="data"><thead><tr><th>Event</th><th>Time</th>'
            for g in range(num_gpus):
                hdr += f'<th>GPU{g} Free</th>'
            hdr += '</tr></thead><tbody>'
            h.append(hdr)

            for ev in key_events:
                phase = ev["event"].rsplit("_", 1)[0] if "_" in ev["event"] else ""
                color_map = {
                    "prefill": "#3b82f6",
                    "decode": "#22c55e",
                    "hcs_soft_load": "#fbbf24",
                }
                # Determine phase from event name
                ev_phase = ""
                for p in color_map:
                    if ev["event"].startswith(p):
                        ev_phase = p
                        break
                dot_color = color_map.get(ev_phase, "#8b949e")
                row = (f'<tr><td><span style="color:{dot_color}">&#9679;</span> '
                       f'{_html_escape(ev["event"])}</td>'
                       f'<td class="num">{ev["time_s"]:.1f}s</td>')
                for mb in ev["gpus"]:
                    row += f'<td class="num">{mb:,} MB</td>'
                row += '</tr>'
                h.append(row)

            h.append('</tbody></table>')
            h.append('</div></details>')

            # VRAM utilization warning
            safety_margin = cr.get("safety_margin_mb")
            soft_load_end = next(
                (ev for ev in key_events if ev["event"] == "hcs_soft_load_end"), None)
            if soft_load_end and safety_margin:
                gpu0_free = soft_load_end["gpus"][0] if soft_load_end["gpus"] else 0
                headroom = gpu0_free - safety_margin
                if headroom > 200:
                    h.append(f'<div class="warning">After HCS soft load, GPU0 has {gpu0_free} MB free '
                             f'({headroom} MB above safety margin of {safety_margin} MB). '
                             f'VRAM is underutilised — soft tier should load more experts.</div>')

    # ── Perplexity ────────────────────────────────────────────────
    if has_ppl:
        h.append('<h2 id="perplexity">Perplexity</h2>')
        h.append(f'<p style="color:#8b949e">WikiText-2, {PERPLEXITY_MAX_TOKENS:,} tokens, '
                 f'window=2048 stride=1024. Lower is better.</p>')

        h.append('<table class="data"><thead><tr>')
        h.append('<th>Config</th><th>PPL</th><th>BPC</th><th>Tokens</th>'
                 '<th>Throughput</th><th>Elapsed</th><th>Baseline</th><th>Status</th>')
        h.append('</tr></thead><tbody>')

        for i, cr in enumerate(config_results):
            ppl_data = cr.get("perplexity")
            if not ppl_data:
                continue

            variant = cr["variant"]
            ppl = ppl_data["perplexity"]
            bpc = ppl_data.get("bpc", 0)
            tokens = ppl_data.get("tokens_scored", 0)
            throughput = ppl_data.get("throughput_tok_s", 0)
            elapsed = ppl_data.get("elapsed_s", 0)
            bs = ppl_data.get("baseline_status", "NEW")
            bl_ppl = ppl_data.get("baseline_ppl")
            bl_delta = ppl_data.get("baseline_delta_pct")

            # Baseline column
            if bs == "NEW":
                bl_str = "no baseline"
            elif bl_ppl is not None and bl_delta is not None:
                sign = "+" if bl_delta >= 0 else ""
                bl_str = f"{bl_ppl:.2f} ({sign}{bl_delta * 100:.1f}%)"
            else:
                bl_str = "—"

            # Status badge
            if bs == "FAIL":
                badge = '<span class="badge badge-fail">FAIL</span>'
            elif bs == "WARN":
                badge = '<span class="badge badge-warn">WARN</span>'
            elif bs == "NEW":
                badge = '<span class="badge badge-skip">NEW</span>'
            else:
                badge = '<span class="badge badge-pass">PASS</span>'

            h.append(f'<tr><td>{_html_escape(variant["name"])}</td>')
            h.append(f'<td class="num">{ppl:.2f}</td>')
            h.append(f'<td class="num">{bpc:.2f}</td>')
            h.append(f'<td class="num">{tokens:,}</td>')
            h.append(f'<td class="num">{throughput:,} tok/s</td>')
            h.append(f'<td class="num">{elapsed:.0f}s</td>')
            h.append(f'<td class="num">{bl_str}</td>')
            h.append(f'<td>{badge}</td></tr>')

        h.append('</tbody></table>')

    if has_validation:
        h.append('<h2 id="reference-validation">Reference Validation</h2>')
        h.append(f'<p style="color:#8b949e">Short greedy-output validation against stored reference outputs. '
                 f'Each config runs Layer 1 token matching only, capped at {REFERENCE_VALIDATE_MAX_PROMPTS} prompts.</p>')
        h.append('<table class="data"><thead><tr>')
        h.append('<th>Config</th><th>Prompts</th><th>Match</th><th>Warn</th><th>Fail</th><th>Status</th>')
        h.append('</tr></thead><tbody>')
        for cr in config_results:
            validation = cr.get("reference_validation")
            if not validation:
                continue
            fail_count = validation.get("fail_count", 0)
            warn_count = validation.get("warn_count", 0)
            match_count = validation.get("match_count", 0)
            prompts_run = validation.get("prompts_run", 0)
            if fail_count:
                badge = '<span class="badge badge-fail">FAIL</span>'
            elif warn_count:
                badge = '<span class="badge badge-warn">WARN</span>'
            else:
                badge = '<span class="badge badge-pass">PASS</span>'
            h.append(f'<tr><td>{_html_escape(cr["variant"]["name"])}</td>')
            h.append(f'<td class="num">{prompts_run}</td>')
            h.append(f'<td class="num">{match_count}</td>')
            h.append(f'<td class="num">{warn_count}</td>')
            h.append(f'<td class="num">{fail_count}</td>')
            h.append(f'<td>{badge}</td></tr>')
        h.append('</tbody></table>')

    # ── Sanity Tests ──────────────────────────────────────────────
    h.append('<h2 id="sanity">Sanity Tests</h2>')

    for i, cr in enumerate(config_results):
        if cr.get("error"):
            continue

        variant = cr["variant"]
        sanity = cr.get("sanity_results", [])
        if not sanity:
            continue

        s_pass = sum(1 for s in sanity if s.get("_validation_status") == "PASS")
        s_warn = sum(1 for s in sanity if s.get("_validation_status") == "WARN")
        s_fail = sum(1 for s in sanity if s.get("_validation_status") == "FAIL")
        parts = []
        if s_pass:
            parts.append(f"{s_pass} PASS")
        if s_warn:
            parts.append(f"{s_warn} WARN")
        if s_fail:
            parts.append(f"{s_fail} FAIL")
        summary_text = ", ".join(parts) if parts else f"0/{len(sanity)}"
        badge_cls = "badge-fail" if s_fail else ("badge-warn" if s_warn else "badge-pass")

        h.append(f'<details id="sanity-{i+1}">')
        h.append(f'<summary>Config {i+1}: {_html_escape(variant["name"])} — '
                 f'<span class="badge {badge_cls}">{summary_text}</span></summary>')
        h.append('<div class="detail-body">')

        for sr in sanity:
            prompt = _html_escape(sr['prompt'])
            turn_label = ""
            if sr.get("turns_in_conversation", 1) > 1:
                turn_label = f' <span class="turn-label">[turn {sr.get("turn", 1)}/{sr["turns_in_conversation"]}]</span>'

            v_status = sr.get("_validation_status", "PASS")
            card_cls = "fail" if v_status == "FAIL" else ("warn" if v_status == "WARN" else "pass")
            h.append(f'<div class="sanity-card {card_cls}">')
            h.append(f'<div class="sanity-prompt">{prompt}{turn_label} '
                     f'<span class="badge badge-{card_cls}">{v_status}</span></div>')

            if sr.get("error"):
                h.append(f'<div style="color:#f85149">ERROR: {_html_escape(sr["error"])}</div>')
            else:
                decode_str = f"{sr['decode_tok_s']:.1f} tok/s" if sr.get('decode_tok_s', 0) > 0 else "N/A"
                prefill_str = f"{sr['prefill_tok_s']:.0f} tok/s" if sr.get('prefill_tok_s', 0) > 0 else "N/A"
                h.append(f'<div class="sanity-stats">Decode: {decode_str} · '
                         f'Prefill: {prefill_str} · TTFT: {sr["ttft_s"]:.2f}s</div>')

                # Show validation issues
                v_issues = sr.get("_validation", [])
                if v_issues:
                    issue_cls = "issues-fail" if v_status == "FAIL" else "issues-warn"
                    issue_lines = " / ".join(f"{sev}: {msg}" for sev, msg in v_issues)
                    h.append(f'<div class="sanity-issues {issue_cls}">{_html_escape(issue_lines)}</div>')

                response = _html_escape(sr.get('text', '(no response)'))
                h.append(f'<div class="sanity-response">{response}</div>')

            h.append('</div>')

        h.append('</div></details>')

    h.append('</body></html>')
    return "\n".join(h)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Krasis release test — runs all config variants and produces a report",
    )
    parser.add_argument("model", help="Model name (directory under ~/.krasis/models/)")
    parser.add_argument("--force-rebuild-cache", action="store_true",
                        help="Force rebuild of all expert caches even if they exist")
    args = parser.parse_args()

    model_name = args.model

    # Validate model exists
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_path):
        die(f"Model directory not found: {model_path}\n"
            f"  Available: {', '.join(os.listdir(MODELS_DIR)) if os.path.isdir(MODELS_DIR) else 'none'}")

    # Read model config
    model_config = read_model_config(model_name)
    num_layers = get_num_layers(model_config)
    if num_layers == 0:
        die("Could not determine num_hidden_layers from config.json")

    total_params_b = estimate_total_params_b(model_config)
    skip_int8 = total_params_b > 200.0

    # Detect GPUs
    gpu_idx, gpu_name, gpu_vram = detect_best_gpu()
    all_gpus = detect_all_gpus()
    all_gpu_indices = [g[0] for g in all_gpus]
    # Find krasis command
    krasis_cmd = find_krasis_command()

    info(f"Model: {model_name} ({num_layers} layers, ~{total_params_b:.0f}B params)")
    if skip_int8:
        info("INT8 configs skipped (model >200B params — INT8 cache too large)")
    info(f"GPU: {gpu_name} ({gpu_vram} MB, index {gpu_idx})")
    if len(all_gpus) > 1:
        gpu_summary = ", ".join(f"{g[1]} ({g[2]} MB)" for g in all_gpus)
        info(f"All GPUs: {gpu_summary}")
    info(f"Krasis: {krasis_cmd}")
    active_variants = [v for v in CONFIG_VARIANTS
                       if (not v.get("multi_gpu") or len(all_gpus) > 1)
                       and (not skip_int8 or v["bits"] != 8)]
    skip_reasons = []
    if len(all_gpus) <= 1 and any(v.get("multi_gpu") for v in CONFIG_VARIANTS):
        skip_reasons.append("multi-GPU skipped: only 1 GPU")
    if skip_int8:
        skip_reasons.append("INT8 skipped: >200B params")
    info(f"Configs: {len(active_variants)} variants"
         f"{' (' + ', '.join(skip_reasons) + ')' if skip_reasons else ''}")

    # Prepare output directory
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = str(get_run_dir("release-test"))
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, "release_test.html")
    info(f"Report: {report_path}")
    print()

    # ── Initial GPU cleanup ─────────────────────────────────────
    # Kill any stale krasis processes and wait for GPU memory to clear.
    # This is critical: build_caches and build_awq_template skip their
    # cleanup when caches/templates already exist, so without this,
    # Config 1 would start with stale GPU memory from whatever ran before.
    info("Cleaning GPU before starting...")
    kill_stale_krasis_processes()
    if not wait_for_gpu_clear(gpu_idx, timeout=30):
        warn("GPU memory not fully cleared — stale processes may affect Config 1")
    else:
        ok("GPU clear.")
    print()

    # ── Pre-flight: build expert caches ─────────────────────────

    print(f"{'=' * 64}")
    info("Phase 0: Building expert caches (if needed)")
    print(f"{'=' * 64}\n")

    cache_results = build_caches(model_name, gpu_idx, num_layers, output_dir,
                                 force=args.force_rebuild_cache,
                                 skip_int8=skip_int8)

    cache_ok = all(cache_results.values())
    if cache_ok:
        ok("All caches ready")
    else:
        failed_bits = [str(b) for b, ok_ in cache_results.items() if not ok_]
        die(f"Cache build failed for INT{', INT'.join(failed_bits)} — aborting release test")
    print()

    # ── Pre-flight: build AWQ template ──────────────────────────

    needs_awq = any(v["attention"] == "awq" and (not skip_int8 or v["bits"] != 8)
                    for v in CONFIG_VARIANTS)
    if needs_awq:
        print(f"{'=' * 64}")
        info("Phase 0b: Building AWQ template (if needed)")
        print(f"{'=' * 64}\n")

        awq_ok = build_awq_template(model_name, gpu_idx, num_layers, output_dir)
        if awq_ok:
            ok("AWQ template ready")
        else:
            warn("AWQ template build failed — AWQ configs will be skipped")
        print()

    # ── Run each config variant ──────────────────────────────────

    config_results = []

    for i, variant in enumerate(CONFIG_VARIANTS):
        print(f"\n{'=' * 64}")
        info(f"Config {i+1}/{len(CONFIG_VARIANTS)}: {variant['name']}")
        print(f"{'=' * 64}\n")

        result = {"variant": variant}
        config_path = None
        proc = None
        config_slug = slugify(f"config{i+1}-{variant['name']}")
        config_dir = str(create_run_dir(config_slug, parent=output_dir))
        log_path = os.path.join(config_dir, "server_stdout.log")

        # Skip multi-GPU configs if only 1 GPU available
        if variant.get("multi_gpu") and len(all_gpus) <= 1:
            result["error"] = "Multi-GPU skipped — only 1 GPU available"
            warn("Skipping (need 2+ GPUs for multi-GPU test)")
            session_notify(
                f"[{model_name}] Config {i+1}/{len(CONFIG_VARIANTS)} "
                f"SKIP: {variant['name']} (need 2+ GPUs)")
            config_results.append(result)
            continue

        # Skip INT8 configs for models >200B params
        if skip_int8 and variant["bits"] == 8:
            result["error"] = f"INT8 skipped — model is ~{total_params_b:.0f}B params (>200B)"
            warn("Skipping (INT8 too large for >200B model)")
            session_notify(
                f"[{model_name}] Config {i+1}/{len(CONFIG_VARIANTS)} "
                f"SKIP: {variant['name']} (INT8 too large)")
            config_results.append(result)
            continue

        # Skip AWQ configs if template wasn't built
        if variant["attention"] == "awq" and needs_awq and not awq_ok:
            result["error"] = "AWQ template not available — skipped"
            warn("Skipping (no AWQ template)")
            session_notify(
                f"[{model_name}] Config {i+1}/{len(CONFIG_VARIANTS)} "
                f"SKIP: {variant['name']} (no AWQ template)")
            config_results.append(result)
            continue

        try:
            # 1. Generate temp config
            config_path = generate_config(
                model_name, variant, gpu_idx, num_layers, all_gpu_indices=all_gpu_indices
            )
            info(f"Config: {config_path}")

            # 2. Launch krasis --config <file> --benchmark
            info(f"Launching: krasis --config ... --benchmark")
            log_file = open(log_path, "w")
            child_env = os.environ.copy()
            child_env["KRASIS_RUN_DIR"] = config_dir
            child_env["KRASIS_RUN_TYPE"] = "release-test-config"
            proc = launch_server(config_path, log_file, env=child_env)
            info(f"PID: {proc.pid} | Log: {log_path}")

            # 3. Wait for benchmark to complete (this includes model load + warmup)
            info("Waiting for benchmark to complete (may take several minutes)...")
            if not wait_for_benchmark_complete(log_path, proc, timeout=BENCHMARK_TIMEOUT):
                if proc.poll() is not None:
                    result["error"] = (
                        f"Server exited with code {proc.returncode}.\n\n"
                        f"Last output:\n{extract_server_error(log_path)}"
                    )
                    warn("Server crashed. Moving on.")
                    config_results.append(result)
                    continue
                else:
                    result["error"] = (
                        f"Benchmark timed out after {BENCHMARK_TIMEOUT}s.\n\n"
                        f"Last output:\n{extract_server_error(log_path)}"
                    )
                    kill_server(proc)
                    warn("Timed out. Moving on.")
                    config_results.append(result)
                    continue

            ok("Benchmark complete.")

            # 4. Extract benchmark output
            result["benchmark_output"] = extract_benchmark_section(log_path)

            # 5. Verify server is accepting HTTP requests
            if not wait_for_health(timeout=30):
                result["error"] = (
                    "Server not responding to health check after benchmark.\n\n"
                    f"Last output:\n{extract_server_error(log_path)}"
                )
                kill_server(proc)
                config_results.append(result)
                continue

            ok("Server healthy. Running HTTP tests.")

            # Warmup request (first HTTP request compiles CUDA graphs etc.)
            thinking = False
            info("Warmup HTTP request...")
            send_chat_streaming("Hi", port=DEFAULT_PORT, max_tokens=8, timeout=60,
                                enable_thinking=thinking)

            # 6. Phase 2: Sanity prompts
            info("Phase 2: Sanity test prompts")
            prev_run_dir = os.environ.get("KRASIS_RUN_DIR")
            prev_run_type = os.environ.get("KRASIS_RUN_TYPE")
            os.environ["KRASIS_RUN_DIR"] = config_dir
            os.environ["KRASIS_RUN_TYPE"] = "release-test-config"
            try:
                result["sanity_results"] = run_sanity_prompts(port=DEFAULT_PORT,
                                                              enable_thinking=thinking)
                # Run validation on sanity results immediately
                sanity_fail_count = 0
                sanity_warn_count = 0
                sanity_pass_count = 0
                for sr in result["sanity_results"]:
                    if sr.get("error"):
                        sanity_fail_count += 1
                        sr["_validation"] = [("FAIL", sr["error"])]
                        sr["_validation_status"] = "FAIL"
                    else:
                        issues = validate_sanity_result(sr.get("prompt", ""), sr.get("text", ""))
                        sr["_validation"] = issues
                        status = validation_status(issues)
                        sr["_validation_status"] = status
                        if status == "FAIL":
                            sanity_fail_count += 1
                        elif status == "WARN":
                            sanity_warn_count += 1
                        else:
                            sanity_pass_count += 1
                result["sanity_fail_count"] = sanity_fail_count
                result["sanity_warn_count"] = sanity_warn_count
                result["sanity_pass_count"] = sanity_pass_count

                # 7. Phase 3: Large prompts
                info("Phase 3: Large prompt validation")
                result["large_results"] = run_large_prompt_tests(port=DEFAULT_PORT,
                                                                  enable_thinking=thinking)

                # 8. Phase 4: Short reference validation
                info(f"Phase 4: Reference validation ({REFERENCE_VALIDATE_MAX_PROMPTS} prompts)")
                result["reference_validation"] = run_reference_validation(
                    config_path,
                    port=DEFAULT_PORT,
                )
            finally:
                if prev_run_dir is None:
                    os.environ.pop("KRASIS_RUN_DIR", None)
                else:
                    os.environ["KRASIS_RUN_DIR"] = prev_run_dir
                if prev_run_type is None:
                    os.environ.pop("KRASIS_RUN_TYPE", None)
                else:
                    os.environ["KRASIS_RUN_TYPE"] = prev_run_type

            ok(f"Config {i+1} complete.")

        except Exception as e:
            result["error"] = f"Unexpected error: {e}"
            warn(f"Error: {e}")

        finally:
            # Kill server
            if proc and proc.poll() is None:
                info("Stopping server...")
                kill_server(proc)

            # Close log file handle
            try:
                log_file.close()
            except Exception:
                pass

            # Extract VRAM report (CSV written by server at shutdown)
            vram_events = extract_vram_report(config_dir)
            if vram_events:
                result["vram_report"] = vram_events

            # Extract safety margin from benchmark log
            if log_path:
                margin = extract_safety_margin(log_path)
                if margin is not None:
                    result["safety_margin_mb"] = margin

            # Wait for GPU to clear before next config
            info("Waiting for GPU memory to clear...")
            if not wait_for_gpu_clear(gpu_idx):
                warn("GPU memory not fully cleared after 60s. Proceeding anyway.")
            else:
                ok("GPU clear.")

            # Clean up temp config
            if config_path and os.path.isfile(config_path):
                os.unlink(config_path)

        # Notify per-config status via Session
        if result.get("error"):
            session_notify(
                f"[{model_name}] Config {i+1}/{len(CONFIG_VARIANTS)} "
                f"FAIL: {variant['name']}")
        else:
            # Build a summary line with key benchmark numbers
            parts = []
            if result.get("benchmark_output"):
                nums = _extract_benchmark_numbers(result["benchmark_output"])
                if nums.get("decode"):
                    parts.append(f"{nums['decode']} tok/s decode")
                if nums.get("hcs"):
                    parts.append(f"{nums['hcs']}% HCS")
            # Include sanity validation summary
            sf = result.get("sanity_fail_count", 0)
            sw = result.get("sanity_warn_count", 0)
            sp = result.get("sanity_pass_count", 0)
            if sf:
                parts.append(f"sanity {sp}P/{sw}W/{sf}F")
            elif sw:
                parts.append(f"sanity {sp}P/{sw}W")
            validation = result.get("reference_validation")
            if validation:
                vf = validation.get("fail_count", 0)
                vw = validation.get("warn_count", 0)
                vm = validation.get("match_count", 0)
                if vf:
                    parts.append(f"ref {vm}M/{vw}W/{vf}F")
                elif vw:
                    parts.append(f"ref {vm}M/{vw}W")
                else:
                    parts.append(f"ref {vm}M")
            config_status = "FAIL" if sf or (validation and validation.get("fail_count", 0) > 0) else "PASS"
            detail = f" ({', '.join(parts)})" if parts else ""
            session_notify(
                f"[{model_name}] Config {i+1}/{len(CONFIG_VARIANTS)} "
                f"{config_status}: {variant['name']}{detail}")

        config_results.append(result)

    # ── Generate report ──────────────────────────────────────────

    print(f"\n{'=' * 64}")
    info("Generating report...")

    report = generate_report(model_name, (gpu_idx, gpu_name, gpu_vram), config_results)

    with open(report_path, "w") as f:
        f.write(report)

    ok(f"Report: {report_path}")

    # Summary
    print(f"\n{BOLD}Release Test Summary — {model_name}{NC}")
    print(f"{'─' * 48}")
    any_failed = False
    for i, cr in enumerate(config_results):
        if cr.get("error"):
            status = f"{RED}FAILED{NC}"
            any_failed = True
        elif cr.get("sanity_fail_count", 0) > 0:
            sf = cr["sanity_fail_count"]
            status = f"{RED}FAILED{NC} ({sf} sanity failures)"
            any_failed = True
        elif cr.get("reference_validation", {}).get("fail_count", 0) > 0:
            vf = cr["reference_validation"]["fail_count"]
            status = f"{RED}FAILED{NC} ({vf} reference failures)"
            any_failed = True
        else:
            status = f"{GREEN}OK{NC}"
        print(f"  Config {i+1} ({cr['variant']['name']}): {status}")
    print(f"{'─' * 48}")
    print(f"Report: {report_path}")
    print()

    # Exit with failure if any config errored or had sanity failures
    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    if not os.environ.get("KRASIS_DEV_SCRIPT"):
        print("ERROR: Do not run this script directly. Use: ./dev release-test <model-name>")
        sys.exit(1)
    main()
