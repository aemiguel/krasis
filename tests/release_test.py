#!/usr/bin/env python3
"""Release test runner for Krasis.

Generates configs on the fly, launches the installed `krasis` command for each,
runs benchmark + sanity + large prompt tests, and produces a markdown report
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
HEALTH_POLL_INTERVAL = 2

SUPPORTED_MODELS = [
    "Qwen3-Coder-Next",
    "Qwen3.5-35B-A3B",
    "Qwen3.5-122B-A10B",
    "Qwen3-235B-A22B",
    "Qwen3.5-397B-A17B",
]

CONFIG_VARIANTS = [
    {"name": "INT4/INT4 BF16",  "gpu_bits": 4, "cpu_bits": 4, "attention": "bf16"},
    {"name": "INT8/INT8 BF16",  "gpu_bits": 8, "cpu_bits": 8, "attention": "bf16"},
    {"name": "INT4/INT4 AWQ",   "gpu_bits": 4, "cpu_bits": 4, "attention": "awq"},
    {"name": "INT8/INT8 AWQ",   "gpu_bits": 8, "cpu_bits": 8, "attention": "awq"},
    # Multi-GPU variant — skipped automatically if only 1 GPU available
    {"name": "INT4/INT4 BF16 Multi-GPU", "gpu_bits": 4, "cpu_bits": 4, "attention": "bf16", "multi_gpu": True},
]

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


def generate_config(model_name: str, variant: Dict, gpu_idx: int,
                    num_layers: int, all_gpu_indices: Optional[List[int]] = None) -> str:
    """Generate a temporary config file. Returns path to temp file.

    For multi-GPU variants, all_gpu_indices is used for CFG_SELECTED_GPUS.
    """
    model_path = os.path.join(MODELS_DIR, model_name)

    if variant.get("multi_gpu") and all_gpu_indices and len(all_gpu_indices) > 1:
        gpu_str = ",".join(str(i) for i in all_gpu_indices)
    else:
        gpu_str = str(gpu_idx)

    lines = [
        f"# Release test config — {variant['name']}",
        f'MODEL_PATH="{model_path}"',
        f'CFG_SELECTED_GPUS="{gpu_str}"',
        f'CFG_PP_PARTITION="{num_layers}"',
        f'CFG_LAYER_GROUP_SIZE="2"',
        f'CFG_KV_DTYPE="fp8_e4m3"',
        f'CFG_GPU_EXPERT_BITS="{variant["gpu_bits"]}"',
        f'CFG_CPU_EXPERT_BITS="{variant["cpu_bits"]}"',
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

def cache_exists(model_name: str, gpu_bits: int, cpu_bits: int) -> bool:
    """Check if GPU Marlin expert cache exists for given bit-width.

    CPU expert caches are no longer used (GPU-only decode mode).
    Checks for any group size (g32, g64, g128) since the Rust engine tries
    multiple group sizes when building.
    """
    import glob as _glob
    model_cache = os.path.join(CACHE_DIR, model_name)
    has_gpu = bool(_glob.glob(os.path.join(model_cache, f"experts_marlin_int{gpu_bits}_g*.bin")))
    return has_gpu


def build_caches(model_name: str, gpu_idx: int, num_layers: int,
                 log_dir: str, force: bool = False) -> Dict[int, bool]:
    """Build expert caches for all required bit-widths before running tests.

    Launches krasis with --build-cache for each unique bit-width (INT4 and INT8).
    If cache already exists and force=False, skips building.

    Returns dict mapping bits -> success.
    """
    # Collect unique bit-widths needed across all config variants
    needed_bits = set()
    for variant in CONFIG_VARIANTS:
        needed_bits.add(variant["gpu_bits"])  # gpu_bits == cpu_bits in our configs

    results = {}
    for bits in sorted(needed_bits):
        if not force and cache_exists(model_name, bits, bits):
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
                f'CFG_KV_DTYPE="fp8_e4m3"',
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
            f'CFG_KV_DTYPE="fp8_e4m3"',
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

        # Run AWQ calibration directly via Python module.
        # We can't use ./dev awq-calibrate because its cleanup_gpu would kill
        # THIS release test process (pgrep matches 'python.*krasis\.').
        # We're already inside ./dev release-test which set up the conda env.
        cmd = [sys.executable, "-m", "krasis.awq_calibrate", "--config", config_path]

        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
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
    dependencies (PyTorch, flashinfer, etc). The ~/.local/bin/krasis may
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


def launch_server(config_path: str, log_file) -> subprocess.Popen:
    """Launch krasis --config <path> --benchmark as a background process.

    Returns the Popen handle. stdout/stderr go to log_file.
    """
    krasis = find_krasis_command()
    cmd = [krasis, "--config", config_path, "--benchmark"]

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
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
                        timeout: int = 120) -> Dict:
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
        "enable_thinking": False,
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

def run_sanity_prompts(port: int = DEFAULT_PORT) -> List[Dict]:
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

    return run_sanity_test(server, temperature=0.3, max_tokens=256)


def run_large_prompt_tests(port: int = DEFAULT_PORT) -> List[Dict]:
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

        r = send_chat_streaming(prompt, port=port, max_tokens=256, timeout=timeout)
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


# ═══════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════

def generate_report(model_name: str, gpu_info: Tuple[int, str, int],
                    config_results: List[Dict]) -> str:
    """Generate the markdown release test report.

    Structure:
    1. Header with metadata and links
    2. All benchmark result summaries grouped together
    3. All sanity test results in full (prompt + response) for manual verification
    """
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append(f"# Krasis Release Test — {model_name}")
    lines.append("")
    lines.append(f"**Date:** {now}")
    lines.append(f"**GPU:** {gpu_info[1]} ({gpu_info[2]} MB)")
    lines.append(f"**Model:** {model_name}")
    lines.append(f"**Configs tested:** {len(config_results)}")
    lines.append("")

    # Links
    lines.append("## Contents")
    lines.append("")
    lines.append("- [Benchmark Results](#benchmark-results)")
    for i, cr in enumerate(config_results):
        variant = cr["variant"]
        status = "FAILED" if cr.get("error") else "PASS"
        anchor = f"config-{i+1}-{variant['name'].lower().replace(' ', '-').replace('/', '')}"
        lines.append(f"  - [{variant['name']} — {status}](#{anchor})")
    lines.append("- [Sanity Tests](#sanity-tests)")
    for i, cr in enumerate(config_results):
        if cr.get("error"):
            continue
        variant = cr["variant"]
        sanity = cr.get("sanity_results", [])
        passed = sum(1 for s in sanity if not s.get("error"))
        anchor = f"sanity-config-{i+1}-{variant['name'].lower().replace(' ', '-').replace('/', '')}"
        lines.append(f"  - [{variant['name']} ({passed}/{len(sanity)})](#{anchor})")
    lines.append("")

    # ── Benchmark Results ──────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Benchmark Results")
    lines.append("")

    for i, cr in enumerate(config_results):
        variant = cr["variant"]
        lines.append(f"### Config {i+1}: {variant['name']}")
        lines.append("")

        if cr.get("error"):
            lines.append("**FAILED**")
            lines.append("")
            lines.append("```")
            lines.append(cr["error"])
            lines.append("```")
            lines.append("")
            continue

        lines.append("```")
        lines.append(cr.get("benchmark_output", "(no benchmark output)"))
        lines.append("```")
        lines.append("")

        # Large prompt results — compact table under benchmark
        large = cr.get("large_results", [])
        if large:
            lines.append("**Large Prompts:**")
            lines.append("")
            lines.append("| Size | Book | Decode | Prefill | TTFT |")
            lines.append("|------|------|--------|---------|------|")
            for lr in large:
                book = lr.get('book', 'unknown')
                if lr.get("error"):
                    lines.append(f"| {lr['prompt_name']} | {book} | FAILED | — | — |")
                else:
                    decode_str = f"{lr['decode_tok_s']:.1f} tok/s" if lr.get('decode_tok_s', 0) > 0 else "N/A"
                    prefill_str = f"{lr['prefill_tok_s']:.0f} tok/s" if lr.get('prefill_tok_s', 0) > 0 else "N/A"
                    lines.append(f"| {lr['prompt_name']} | {book} | {decode_str} | {prefill_str} | {lr['ttft_s']:.2f}s |")
            lines.append("")

    # ── Sanity Tests ───────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Sanity Tests")
    lines.append("")

    for i, cr in enumerate(config_results):
        if cr.get("error"):
            continue

        variant = cr["variant"]
        sanity = cr.get("sanity_results", [])
        if not sanity:
            continue

        passed = sum(1 for s in sanity if not s.get("error"))
        failed = len(sanity) - passed
        status = f"{passed}/{len(sanity)} passed" if failed == 0 else f"{passed}/{len(sanity)} passed, {failed} FAILED"

        lines.append(f"### Sanity Config {i+1}: {variant['name']}")
        lines.append("")
        lines.append(f"**{status}**")
        lines.append("")

        for sr in sanity:
            prompt = sr['prompt']
            turn_label = ""
            if sr.get("turns_in_conversation", 1) > 1:
                turn_label = f" [turn {sr.get('turn', 1)}/{sr['turns_in_conversation']}]"

            if sr.get("error"):
                lines.append(f"**Prompt:** {prompt}{turn_label}")
                lines.append(f"**FAILED:** {sr['error']}")
                lines.append("")
            else:
                decode_str = f"{sr['decode_tok_s']:.1f} tok/s" if sr.get('decode_tok_s', 0) > 0 else "N/A"
                prefill_str = f"{sr['prefill_tok_s']:.0f} tok/s" if sr.get('prefill_tok_s', 0) > 0 else "N/A"
                lines.append(f"**Prompt:** {prompt}{turn_label}  ")
                lines.append(f"*Decode: {decode_str} | Prefill: {prefill_str} | TTFT: {sr['ttft_s']:.2f}s*")
                lines.append("")
                response = sr.get('text', '(no response)')
                lines.append(f"> {response.replace(chr(10), chr(10) + '> ')}")
                lines.append("")

    return "\n".join(lines)


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

    # Detect GPUs
    gpu_idx, gpu_name, gpu_vram = detect_best_gpu()
    all_gpus = detect_all_gpus()
    all_gpu_indices = [g[0] for g in all_gpus]

    # Find krasis command
    krasis_cmd = find_krasis_command()

    info(f"Model: {model_name} ({num_layers} layers)")
    info(f"GPU: {gpu_name} ({gpu_vram} MB, index {gpu_idx})")
    if len(all_gpus) > 1:
        gpu_summary = ", ".join(f"{g[1]} ({g[2]} MB)" for g in all_gpus)
        info(f"All GPUs: {gpu_summary}")
    info(f"Krasis: {krasis_cmd}")
    # Count active variants (skip multi-GPU if only 1 GPU)
    active_variants = [v for v in CONFIG_VARIANTS
                       if not v.get("multi_gpu") or len(all_gpus) > 1]
    info(f"Configs: {len(active_variants)} variants"
         f"{' (multi-GPU skipped: only 1 GPU)' if len(active_variants) < len(CONFIG_VARIANTS) else ''}")

    # Prepare output directory
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(script_dir, "logs", "release-tests")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(
        output_dir,
        f"krasis_release_test_{model_name}_{timestamp}.md",
    )
    info(f"Report: {report_path}")
    print()

    # ── Pre-flight: build expert caches ─────────────────────────

    print(f"{'=' * 64}")
    info("Phase 0: Building expert caches (if needed)")
    print(f"{'=' * 64}\n")

    cache_results = build_caches(model_name, gpu_idx, num_layers, output_dir,
                                 force=args.force_rebuild_cache)

    cache_ok = all(cache_results.values())
    if cache_ok:
        ok("All caches ready")
    else:
        failed_bits = [str(b) for b, ok_ in cache_results.items() if not ok_]
        die(f"Cache build failed for INT{', INT'.join(failed_bits)} — aborting release test")
    print()

    # ── Pre-flight: build AWQ template ──────────────────────────

    needs_awq = any(v["attention"] == "awq" for v in CONFIG_VARIANTS)
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
        log_path = os.path.join(output_dir, f"server_config{i+1}_{timestamp}.log")

        # Skip multi-GPU configs if only 1 GPU available
        if variant.get("multi_gpu") and len(all_gpus) <= 1:
            result["error"] = "Multi-GPU skipped — only 1 GPU available"
            warn("Skipping (need 2+ GPUs for multi-GPU test)")
            config_results.append(result)
            continue

        # Skip AWQ configs if template wasn't built
        if variant["attention"] == "awq" and needs_awq and not awq_ok:
            result["error"] = "AWQ template not available — skipped"
            warn("Skipping (no AWQ template)")
            config_results.append(result)
            continue

        try:
            # 1. Generate temp config
            config_path = generate_config(model_name, variant, gpu_idx, num_layers,
                                          all_gpu_indices=all_gpu_indices)
            info(f"Config: {config_path}")

            # 2. Launch krasis --config <file> --benchmark
            info(f"Launching: krasis --config ... --benchmark")
            log_file = open(log_path, "w")
            proc = launch_server(config_path, log_file)
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
            info("Warmup HTTP request...")
            send_chat_streaming("Hi", port=DEFAULT_PORT, max_tokens=8, timeout=60)

            # 6. Phase 2: Sanity prompts
            info("Phase 2: Sanity test prompts")
            result["sanity_results"] = run_sanity_prompts(port=DEFAULT_PORT)

            # 7. Phase 3: Large prompts
            info("Phase 3: Large prompt validation")
            result["large_results"] = run_large_prompt_tests(port=DEFAULT_PORT)

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

            # Clean up temp config
            if config_path and os.path.isfile(config_path):
                os.unlink(config_path)

            # Wait for GPU to clear before next config
            info("Waiting for GPU memory to clear...")
            if not wait_for_gpu_clear(gpu_idx):
                warn("GPU memory not fully cleared after 60s. Proceeding anyway.")
            else:
                ok("GPU clear.")

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
    for i, cr in enumerate(config_results):
        status = f"{RED}FAILED{NC}" if cr.get("error") else f"{GREEN}OK{NC}"
        print(f"  Config {i+1} ({cr['variant']['name']}): {status}")
    print(f"{'─' * 48}")
    print(f"Report: {report_path}")
    print()

    # Exit with failure if any config errored
    if any(cr.get("error") for cr in config_results):
        sys.exit(1)


if __name__ == "__main__":
    if not os.environ.get("KRASIS_DEV_SCRIPT"):
        print("ERROR: Do not run this script directly. Use: ./dev release-test <model-name>")
        sys.exit(1)
    main()
