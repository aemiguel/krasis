"""Krasis TUI Launcher — arrow-key-driven interactive configuration.

Provides model selection, per-component quantization config, live VRAM/RAM
budget display, and server launch. Uses only stdlib + existing krasis modules.

Usage:
    python -m krasis.launcher                         # interactive TUI
    python -m krasis.launcher --non-interactive       # use saved config
    python -m krasis.launcher --model-path /path/to/model
"""

import argparse
import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

# Terminal handling imports — graceful fallback for non-Unix
try:
    import termios
    import tty
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False


# ═══════════════════════════════════════════════════════════════════════
# ANSI helpers
# ═══════════════════════════════════════════════════════════════════════

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
NC = "\033[0m"  # reset

import re
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _visible_len(s: str) -> int:
    """Length of string with ANSI escape codes stripped."""
    return len(_ANSI_RE.sub("", s))


def _clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def _hide_cursor():
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


def _show_cursor():
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════
# Terminal raw-mode key reading
# ═══════════════════════════════════════════════════════════════════════

KEY_UP = "UP"
KEY_DOWN = "DOWN"
KEY_LEFT = "LEFT"
KEY_RIGHT = "RIGHT"
KEY_ENTER = "ENTER"
KEY_ESCAPE = "ESC"
KEY_QUIT = "q"
KEY_SPACE = " "
KEY_BACKSPACE = "BACKSPACE"


def _read_key() -> str:
    """Read a single keypress in raw mode. Returns key constant or char."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)

        if ch == "\x1b":
            # Escape sequence
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                if ch3 == "A":
                    return KEY_UP
                elif ch3 == "B":
                    return KEY_DOWN
                elif ch3 == "C":
                    return KEY_RIGHT
                elif ch3 == "D":
                    return KEY_LEFT
            return KEY_ESCAPE
        elif ch in ("\r", "\n"):
            return KEY_ENTER
        elif ch == "\x7f" or ch == "\x08":
            return KEY_BACKSPACE
        elif ch == "\x03":  # Ctrl-C
            return KEY_ESCAPE
        else:
            return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ═══════════════════════════════════════════════════════════════════════
# Hardware detection
# ═══════════════════════════════════════════════════════════════════════

def detect_hardware() -> Dict[str, Any]:
    """Detect GPUs, CPU, RAM. Returns dict with hardware info.

    hw["gpus"] is a list of per-GPU dicts: {index, name, vram_mb}.
    hw["gpu_count"], ["gpu_model"], ["gpu_vram_mb"] reflect all GPUs / first GPU.
    """
    hw: Dict[str, Any] = {
        "gpus": [],           # per-GPU list
        "gpu_count": 0,
        "gpu_model": "unknown",
        "gpu_vram_mb": 0,
        "gpu_sm": (0, 0),     # compute capability of first GPU, e.g. (8, 9)
        "has_fp8": False,      # SM >= 8.9 supports FP8
        "cpu_model": "unknown",
        "cpu_cores": 0,
        "has_avx2": False,
        "total_ram_gb": 0,
    }

    # GPUs via nvidia-smi — per-GPU info + compute capability
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    sm_parts = parts[3].split(".")
                    sm = (int(sm_parts[0]), int(sm_parts[1])) if len(sm_parts) >= 2 else (0, 0)
                    hw["gpus"].append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "vram_mb": int(parts[2]),
                        "sm": sm,
                    })
                elif len(parts) >= 3:
                    hw["gpus"].append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "vram_mb": int(parts[2]),
                        "sm": (0, 0),
                    })
            hw["gpu_count"] = len(hw["gpus"])
            if hw["gpus"]:
                hw["gpu_model"] = hw["gpus"][0]["name"]
                hw["gpu_vram_mb"] = hw["gpus"][0]["vram_mb"]
                hw["gpu_sm"] = hw["gpus"][0]["sm"]
                hw["has_fp8"] = hw["gpu_sm"] >= (8, 9)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # CPU model
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    hw["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except FileNotFoundError:
        pass

    # Physical cores
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            cores = 0
            sockets = 1
            for line in result.stdout.split("\n"):
                if line.startswith("Core(s) per socket:"):
                    cores = int(line.split(":")[-1].strip())
                elif line.startswith("Socket(s):"):
                    sockets = int(line.split(":")[-1].strip())
            hw["cpu_cores"] = cores * sockets
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # AVX2
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()
            hw["has_avx2"] = " avx2 " in content
    except FileNotFoundError:
        pass

    # RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    hw["total_ram_gb"] = kb // (1024 * 1024)
                    break
    except (FileNotFoundError, ValueError):
        pass

    return hw


# ═══════════════════════════════════════════════════════════════════════
# Model scanning
# ═══════════════════════════════════════════════════════════════════════

def scan_models(search_dir: str, native_only: bool = False) -> List[Dict[str, Any]]:
    """Scan directory for HF model directories with config.json.

    Recurses into subdirectories to find models in org/repo folder structures
    (e.g. models/Qwen/Qwen3.5-122B-A10B/).

    Args:
        native_only: If True, only return models that have safetensors files.
    """
    models = []
    if not os.path.isdir(search_dir):
        return models

    # Walk the tree looking for directories containing config.json
    for dirpath, dirnames, filenames in os.walk(search_dir):
        if "config.json" not in filenames:
            continue

        model_dir = dirpath
        config_path = os.path.join(model_dir, "config.json")

        # Native-only filter: require at least one .safetensors file
        if native_only:
            if not any(f.endswith(".safetensors") for f in filenames):
                continue

        # Use relative path from search_dir as display name (e.g. "Qwen/Qwen3.5-122B-A10B")
        rel = os.path.relpath(model_dir, search_dir)
        name = rel if rel != "." else os.path.basename(model_dir)

        try:
            with open(config_path) as f:
                raw = json.load(f)
            cfg = raw.get("text_config", raw)

            arch = cfg.get("model_type", "?")
            layers = cfg.get("num_hidden_layers", 0)
            experts = cfg.get("n_routed_experts", cfg.get("num_experts", 0))
            shared = cfg.get("n_shared_experts", 0)
            hidden = cfg.get("hidden_size", 0)
            inter = cfg.get("moe_intermediate_size", cfg.get("intermediate_size", 0))

            # Estimate expert RAM (BF16)
            ram_gb = 0.0
            if experts and hidden and inter:
                ram_gb = (3 * hidden * inter * 2 * experts * (layers - cfg.get("first_k_dense_replace", 0))) / (1024**3)

            # Determine number of dense (non-MoE) layers
            if "first_k_dense_replace" in cfg:
                first_k_dense = cfg["first_k_dense_replace"]
            elif "decoder_sparse_step" in cfg:
                step = cfg["decoder_sparse_step"]
                first_k_dense = 0 if step <= 1 else step
            else:
                first_k_dense = 0

            models.append({
                "name": name,
                "path": model_dir,
                "arch": arch,
                "layers": layers,
                "experts": experts,
                "shared_experts": shared,
                "dense_layers": first_k_dense,
                "native_dtype": raw.get("torch_dtype", "bfloat16"),
                "ram_gb": ram_gb,
            })
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    # Sort by name for consistent display
    models.sort(key=lambda m: m["name"].lower())
    return models


def scan_gguf_files(search_dir: str) -> List[Dict[str, Any]]:
    """Scan for GGUF files across all subdirectories."""
    gguf_files = []
    if not os.path.isdir(search_dir):
        return gguf_files

    for name in sorted(os.listdir(search_dir)):
        model_dir = os.path.join(search_dir, name)
        if not os.path.isdir(model_dir):
            continue

        try:
            for fn in sorted(os.listdir(model_dir)):
                if fn.endswith(".gguf"):
                    full_path = os.path.join(model_dir, fn)
                    try:
                        size_bytes = os.path.getsize(full_path)
                    except OSError:
                        size_bytes = 0
                    gguf_files.append({
                        "name": fn,
                        "path": full_path,
                        "dir_name": name,
                        "size_gb": size_bytes / (1024**3),
                    })
        except OSError:
            continue

    return gguf_files


# ═══════════════════════════════════════════════════════════════════════
# Config file (KEY=VALUE format, backward-compatible with bash)
# ═══════════════════════════════════════════════════════════════════════

CONFIG_KEYS = [
    "MODEL_PATH", "CFG_SELECTED_GPUS", "CFG_PP_PARTITION", "CFG_LAYER_GROUP_SIZE",
    "CFG_KV_CACHE_MB", "CFG_KV_DTYPE", "CFG_GPU_EXPERT_BITS", "CFG_CPU_EXPERT_BITS",
    "CFG_ATTENTION_QUANT", "CFG_SHARED_EXPERT_QUANT", "CFG_DENSE_MLP_QUANT",
    "CFG_LM_HEAD_QUANT", "CFG_KRASIS_THREADS", "CFG_HOST", "CFG_PORT",
    "CFG_GPU_PREFILL_THRESHOLD", "CFG_GGUF_PATH", "CFG_VRAM_SAFETY_MARGIN",
    "CFG_FORCE_LOAD", "CFG_ENABLE_THINKING",
]


def _load_config(config_file: str) -> Dict[str, str]:
    """Load KEY=VALUE config file (bash-compatible)."""
    cfg = {}
    if not os.path.isfile(config_file):
        return cfg
    try:
        with open(config_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    cfg[key] = val
    except (OSError, UnicodeDecodeError):
        pass
    return cfg


def _save_config(config_file: str, values: Dict[str, Any]) -> None:
    """Save config to KEY=VALUE file (bash-compatible)."""
    import datetime
    with open(config_file, "w") as f:
        f.write(f"# Krasis saved configuration — {datetime.datetime.now().isoformat()}\n")
        f.write("# Re-generated by krasis launcher on each launch\n")
        for key in CONFIG_KEYS:
            val = values.get(key, "")
            f.write(f'{key}="{val}"\n')


# ═══════════════════════════════════════════════════════════════════════
# Launcher config (all tunable parameters)
# ═══════════════════════════════════════════════════════════════════════

class LauncherConfig:
    """Holds all launcher config values."""

    def __init__(self):
        self.model_path: str = ""
        self.selected_gpu_indices: List[int] = []  # empty = all GPUs
        self.pp_partition: str = ""
        self.layer_group_size: int = 2  # layers per group (must be even, min 2 for double buffering)
        self.kv_cache_mb: int = 1000
        self.kv_dtype: str = "fp8_e4m3"
        self.gpu_expert_bits: int = 4
        self.cpu_expert_bits: int = 4
        self.attention_quant: str = "bf16"
        self.shared_expert_quant: str = "int8"
        self.dense_mlp_quant: str = "int8"
        self.lm_head_quant: str = "int8"
        self.krasis_threads: int = 40
        self.host: str = "0.0.0.0"
        self.port: int = 8012
        self.gpu_prefill_threshold: int = 300
        self.gguf_path: str = ""
        self.vram_safety_margin: int = 600
        self.force_load: bool = False
        self.force_rebuild_cache: bool = False
        self.build_cache: bool = False
        self.enable_thinking: bool = True
        self.session_enabled: bool = False

    def apply_saved(self, saved: Dict[str, str]) -> None:
        """Apply loaded config values."""
        if "MODEL_PATH" in saved and saved["MODEL_PATH"]:
            self.model_path = saved["MODEL_PATH"]
        if "CFG_SELECTED_GPUS" in saved and saved["CFG_SELECTED_GPUS"]:
            try:
                self.selected_gpu_indices = [
                    int(x.strip()) for x in saved["CFG_SELECTED_GPUS"].split(",")
                    if x.strip()
                ]
            except ValueError:
                pass
        if "CFG_PP_PARTITION" in saved:
            self.pp_partition = saved["CFG_PP_PARTITION"]
        if "CFG_LAYER_GROUP_SIZE" in saved:
            val = saved["CFG_LAYER_GROUP_SIZE"]
            try:
                v = int(val)
                if v >= 2:
                    self.layer_group_size = v
            except ValueError:
                pass
        # Legacy compat
        elif "CFG_EXPERT_DIVISOR" in saved:
            val = saved["CFG_EXPERT_DIVISOR"]
            try:
                v = int(val)
                if v >= 2:
                    self.layer_group_size = v
            except ValueError:
                pass
        if "CFG_KV_CACHE_MB" in saved:
            try:
                self.kv_cache_mb = int(saved["CFG_KV_CACHE_MB"])
            except ValueError:
                pass
        if "CFG_KV_DTYPE" in saved:
            self.kv_dtype = saved["CFG_KV_DTYPE"]
        if "CFG_GPU_EXPERT_BITS" in saved:
            try:
                self.gpu_expert_bits = int(saved["CFG_GPU_EXPERT_BITS"])
            except ValueError:
                pass
        if "CFG_CPU_EXPERT_BITS" in saved:
            try:
                self.cpu_expert_bits = int(saved["CFG_CPU_EXPERT_BITS"])
            except ValueError:
                pass
        if "CFG_ATTENTION_QUANT" in saved:
            val = saved["CFG_ATTENTION_QUANT"]
            # Migrate legacy naive int4/int8 to AWQ (calibrated per-tensor)
            if val in ("int4", "int8"):
                val = "awq"
            self.attention_quant = val
        if "CFG_SHARED_EXPERT_QUANT" in saved:
            self.shared_expert_quant = saved["CFG_SHARED_EXPERT_QUANT"]
        if "CFG_DENSE_MLP_QUANT" in saved:
            self.dense_mlp_quant = saved["CFG_DENSE_MLP_QUANT"]
        if "CFG_LM_HEAD_QUANT" in saved:
            self.lm_head_quant = saved["CFG_LM_HEAD_QUANT"]
        if "CFG_KRASIS_THREADS" in saved:
            try:
                self.krasis_threads = int(saved["CFG_KRASIS_THREADS"])
            except ValueError:
                pass
        if "CFG_HOST" in saved:
            self.host = saved["CFG_HOST"]
        if "CFG_PORT" in saved:
            try:
                self.port = int(saved["CFG_PORT"])
            except ValueError:
                pass
        if "CFG_GPU_PREFILL_THRESHOLD" in saved:
            try:
                self.gpu_prefill_threshold = int(saved["CFG_GPU_PREFILL_THRESHOLD"])
            except ValueError:
                pass
        if "CFG_GGUF_PATH" in saved:
            self.gguf_path = saved["CFG_GGUF_PATH"]
        if "CFG_VRAM_SAFETY_MARGIN" in saved:
            try:
                self.vram_safety_margin = int(saved["CFG_VRAM_SAFETY_MARGIN"])
            except (ValueError, TypeError):
                pass
        if "CFG_FORCE_LOAD" in saved and saved["CFG_FORCE_LOAD"]:
            self.force_load = saved["CFG_FORCE_LOAD"] == "1"
        if "CFG_FORCE_REBUILD_CACHE" in saved and saved["CFG_FORCE_REBUILD_CACHE"]:
            self.force_rebuild_cache = saved["CFG_FORCE_REBUILD_CACHE"] == "1"
        if "CFG_BUILD_CACHE" in saved and saved["CFG_BUILD_CACHE"]:
            self.build_cache = saved["CFG_BUILD_CACHE"] == "1"
        if "CFG_ENABLE_THINKING" in saved:
            self.enable_thinking = saved["CFG_ENABLE_THINKING"] != "0"
        if "CFG_SESSION_ENABLED" in saved:
            self.session_enabled = saved["CFG_SESSION_ENABLED"] == "1"

    def to_save_dict(self) -> Dict[str, Any]:
        """Convert to dict for saving."""
        return {
            "MODEL_PATH": self.model_path,
            "CFG_SELECTED_GPUS": ",".join(str(i) for i in self.selected_gpu_indices),
            "CFG_PP_PARTITION": self.pp_partition,
            "CFG_LAYER_GROUP_SIZE": str(self.layer_group_size),
            "CFG_KV_CACHE_MB": str(self.kv_cache_mb),
            "CFG_KV_DTYPE": self.kv_dtype,
            "CFG_GPU_EXPERT_BITS": str(self.gpu_expert_bits),
            "CFG_CPU_EXPERT_BITS": str(self.cpu_expert_bits),
            "CFG_ATTENTION_QUANT": self.attention_quant,
            "CFG_SHARED_EXPERT_QUANT": self.shared_expert_quant,
            "CFG_DENSE_MLP_QUANT": self.dense_mlp_quant,
            "CFG_LM_HEAD_QUANT": self.lm_head_quant,
            "CFG_KRASIS_THREADS": str(self.krasis_threads),
            "CFG_HOST": self.host,
            "CFG_PORT": str(self.port),
            "CFG_GPU_PREFILL_THRESHOLD": str(self.gpu_prefill_threshold),
            "CFG_GGUF_PATH": self.gguf_path,
            "CFG_VRAM_SAFETY_MARGIN": str(self.vram_safety_margin),
            "CFG_FORCE_LOAD": "1" if self.force_load else "",
            "CFG_FORCE_REBUILD_CACHE": "1" if self.force_rebuild_cache else "",
            "CFG_BUILD_CACHE": "1" if self.build_cache else "",
            "CFG_ENABLE_THINKING": "1" if self.enable_thinking else "0",
            "CFG_SESSION_ENABLED": "1" if self.session_enabled else "0",
        }


# ═══════════════════════════════════════════════════════════════════════
# Config options for TUI
# ═══════════════════════════════════════════════════════════════════════

class ConfigOption:
    """One configurable item in the TUI."""

    def __init__(
        self,
        label: str,
        key: str,
        choices: Optional[List[Any]] = None,
        opt_type: str = "cycle",  # "cycle", "number", "text"
        affects_budget: bool = False,
        suffix: str = "",
        min_val: int = 1,
        max_val: int = 65536,
        step: int = 1,
    ):
        self.label = label
        self.key = key
        self.choices = choices
        self.opt_type = opt_type
        self.affects_budget = affects_budget
        self.suffix = suffix
        self.min_val = min_val
        self.max_val = max_val
        self.step = step


# Config options shown in TUI
OPTIONS = [
    ConfigOption("Layer group size", "layer_group_size",
                 choices=[2, 4, 6, 8, 10, 12], affects_budget=True),
    ConfigOption("KV cache (MB)", "kv_cache_mb",
                 opt_type="number", min_val=200, max_val=65500, step=100, affects_budget=True),
    ConfigOption("KV dtype", "kv_dtype",
                 choices=["fp8_e4m3", "bf16"], affects_budget=True),
    ConfigOption("GPU expert bits", "gpu_expert_bits",
                 choices=[4, 8], affects_budget=True),
    ConfigOption("Attention quant", "attention_quant",
                 choices=["bf16", "awq"], affects_budget=True),
    ConfigOption("Shared expert", "shared_expert_quant",
                 choices=["int8", "bf16"], affects_budget=True),
    ConfigOption("Dense MLP quant", "dense_mlp_quant",
                 choices=["int8", "bf16"], affects_budget=True),
    ConfigOption("LM head quant", "lm_head_quant",
                 choices=["int8", "bf16"], affects_budget=True),
    ConfigOption("VRAM safety margin", "vram_safety_margin",
                 opt_type="number", min_val=500, max_val=8000, step=100),
    ConfigOption("Host/Port", "host", opt_type="text"),
    ConfigOption("Enable thinking", "enable_thinking",
                 choices=[True, False]),
    ConfigOption("Session messenger", "session_enabled",
                 choices=[True, False]),
]

def _format_value(opt: ConfigOption, val: Any) -> str:
    """Format a config value for display."""
    if isinstance(val, bool):
        return f"{GREEN}ON{NC}" if val else f"{DIM}OFF{NC}"
    if opt.key == "layer_group_size":
        return f"{val} layers (double-buffered)"
    if opt.key == "kv_cache_mb":
        return f"{val:,} MB"
    if opt.key == "vram_safety_margin":
        return f"{val:,} MB"
    if opt.key == "attention_quant":
        labels = {
            "bf16": "BF16",
            "awq": "AWQ",
        }
        return labels.get(str(val), str(val))
    return str(val)


def _is_option_visible(opt: ConfigOption, model_info: Optional[Dict], cfg: Optional["LauncherConfig"] = None) -> bool:
    """Check if an option should be shown for the current model."""
    if opt.key == "dense_mlp_quant":
        # Only show if model has dense (non-MoE) layers
        if model_info and model_info.get("dense_layers", 0) == 0:
            return False
    return True


# Maps config key → short label for the native dtype display
_NATIVE_DTYPE_LABELS = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
}


def _quality_annotation(native_dtype: str, config_key: str, current_val: Any) -> str:
    """Return a quality annotation string like 'bf16 → int8, ~lossless'.

    Returns empty string for options that don't have a native/quality concept.
    """
    native_label = _NATIVE_DTYPE_LABELS.get(native_dtype, native_dtype)

    if config_key in ("attention_quant", "shared_expert_quant",
                      "dense_mlp_quant", "lm_head_quant"):
        current = str(current_val)  # "int8", "int4", "bf16", or "awq"
        if current == native_label or current == native_dtype:
            return f"{DIM}{native_label} \u2192 {current} \u2014 lossless{NC}"
        elif current == "int8":
            return f"{DIM}{native_label} \u2192 int8 \u2014 ~lossless{NC}"
        elif current == "int4":
            return f"{DIM}{native_label} \u2192 int4 \u2014 {YELLOW}slight loss{NC}{DIM}{NC}"
        elif current == "awq":
            return f"{DIM}{native_label} \u2192 AWQ \u2014 {GREEN}calibrated per-tensor{NC}{DIM}{NC}"
        return f"{DIM}{native_label} \u2192 {current}{NC}"

    elif config_key == "gpu_expert_bits":
        bits = int(current_val)
        if bits == 16:
            return f"{DIM}{native_label} \u2192 {native_label} \u2014 lossless{NC}"
        elif bits == 8:
            return f"{DIM}{native_label} \u2192 int8 \u2014 ~lossless{NC}"
        elif bits == 4:
            return f"{DIM}{native_label} \u2192 int4 \u2014 {YELLOW}slight loss{NC}{DIM}{NC}"
        return f"{DIM}{native_label} \u2192 int{bits}{NC}"

    elif config_key == "kv_dtype":
        current = str(current_val)
        if current == "fp8_e4m3":
            return f"{DIM}{native_label} \u2192 fp8 \u2014 ~lossless{NC}"
        elif current == "bf16":
            return f"{DIM}{native_label} \u2192 bf16 \u2014 lossless{NC}"
        return f"{DIM}{native_label} \u2192 {current}{NC}"

    return ""


def _format_tokens(n: int) -> str:
    """Format token count as e.g. '72K' or '1.2M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1000:
        return f"{n / 1000:.0f}K"
    return str(n)


# ═══════════════════════════════════════════════════════════════════════
# Model selection screen
# ═══════════════════════════════════════════════════════════════════════

def _model_selection_screen(
    models: List[Dict[str, Any]],
    preselected_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Arrow-key model picker. Returns selected model dict or None.

    If preselected_path is given, the cursor starts on that model.
    """
    if not models:
        return None

    cursor = 0
    if preselected_path:
        for i, m in enumerate(models):
            if m["path"] == preselected_path:
                cursor = i
                break

    while True:
        _clear_screen()
        lines = []
        lines.append(f"  {BOLD}Select a model:{NC}\n")

        for i, m in enumerate(models):
            prefix = f"  {CYAN}\u25b8{NC} " if i == cursor else "    "
            hl = BOLD if i == cursor else ""

            expert_str = ""
            if m["experts"]:
                expert_str = f"{m['experts']} experts"
                if m["shared_experts"]:
                    expert_str += f" (+{m['shared_experts']} shared)"
            else:
                expert_str = "dense"

            ram_str = f" | ~{m['ram_gb']:.0f} GB" if m["ram_gb"] > 0 else ""

            lines.append(f"{prefix}{hl}{m['name']}{NC}")
            lines.append(f"       {DIM}{m['arch']} | {m['layers']} layers | {expert_str}{ram_str}{NC}")

        lines.append(f"\n  {DIM}[\u2191\u2193] Select  [Enter] Confirm  [q] Quit{NC}")

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

        key = _read_key()
        if key == KEY_UP:
            cursor = (cursor - 1) % len(models)
        elif key == KEY_DOWN:
            cursor = (cursor + 1) % len(models)
        elif key == KEY_ENTER:
            return models[cursor]
        elif key == KEY_QUIT or key == KEY_ESCAPE:
            return None


# ═══════════════════════════════════════════════════════════════════════
# GPU selection screen
# ═══════════════════════════════════════════════════════════════════════

def _gpu_selection_screen(
    gpus: List[Dict[str, Any]],
    preselected: Optional[List[int]] = None,
) -> Optional[List[int]]:
    """Arrow-key GPU selector with toggle. Returns list of selected GPU indices, or None."""
    if not gpus:
        return None

    cursor = 0
    if preselected:
        selected = [g["index"] in preselected for g in gpus]
    else:
        # Default: only the GPU with the largest VRAM
        best_idx = max(gpus, key=lambda g: g["vram_mb"])["index"]
        selected = [g["index"] == best_idx for g in gpus]

    while True:
        _clear_screen()
        lines = []
        lines.append(f"  {BOLD}Select GPUs to use:{NC}\n")

        for i, gpu in enumerate(gpus):
            prefix = f"  {CYAN}\u25b8{NC} " if i == cursor else "    "
            hl = BOLD if i == cursor else ""
            check = f"{GREEN}\u2714{NC}" if selected[i] else f"{RED}\u2718{NC}"

            lines.append(
                f"{prefix}[{check}] {hl}GPU {gpu['index']}: {gpu['name']}{NC}"
                f"  {DIM}({gpu['vram_mb']:,} MB){NC}"
            )

        # Summary
        sel_count = sum(selected)
        total_vram = sum(g["vram_mb"] for g, s in zip(gpus, selected) if s)
        lines.append("")
        if sel_count > 0:
            lines.append(f"  Selected: {BOLD}{sel_count}{NC} GPU(s), {total_vram:,} MB total VRAM")
        else:
            lines.append(f"  {RED}No GPUs selected — at least one is required{NC}")

        lines.append(f"\n  {DIM}[\u2191\u2193] Navigate  [Space] Toggle  [Enter] Confirm  [q] Quit{NC}")

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

        key = _read_key()
        if key == KEY_UP:
            cursor = (cursor - 1) % len(gpus)
        elif key == KEY_DOWN:
            cursor = (cursor + 1) % len(gpus)
        elif key == KEY_SPACE:
            selected[cursor] = not selected[cursor]
        elif key == KEY_ENTER:
            indices = [gpus[i]["index"] for i in range(len(gpus)) if selected[i]]
            if not indices:
                continue  # don't allow empty selection
            return indices
        elif key == KEY_QUIT or key == KEY_ESCAPE:
            return None


# ═══════════════════════════════════════════════════════════════════════
# Launch mode selection screen
# ═══════════════════════════════════════════════════════════════════════

def _launch_mode_screen() -> Optional[str]:
    """Select launch mode: Launch, Benchmark, or Benchmark Suite.

    Returns "launch", "benchmark", "suite", or None if cancelled.
    """
    options = [
        ("Launch", "Start the server immediately", "launch"),
        ("Benchmark and Launch", "Run prefill + decode benchmark, then start server", "benchmark"),
        ("Benchmark Only", "Run benchmark and exit", "benchmark_only"),
        ("Stress Test", "Run diverse prompts to catch edge cases", "stress_test"),
        ("Benchmark Suite", "Run all model \u00d7 config combos from suite config", "suite"),
    ]
    cursor = 0

    while True:
        _clear_screen()
        lines = []
        lines.append(f"  {BOLD}Select launch mode:{NC}\n")

        for i, (label, desc, _mode) in enumerate(options):
            prefix = f"  {CYAN}\u25b8{NC} " if i == cursor else "    "
            hl = BOLD if i == cursor else ""
            lines.append(f"{prefix}{hl}{label}{NC}  {DIM}{desc}{NC}")

        lines.append(f"\n  {DIM}[\u2191\u2193] Select  [Enter] Confirm  [q] Quit{NC}")

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

        key = _read_key()
        if key == KEY_UP:
            cursor = (cursor - 1) % len(options)
        elif key == KEY_DOWN:
            cursor = (cursor + 1) % len(options)
        elif key == KEY_ENTER:
            return options[cursor][2]
        elif key == KEY_QUIT or key == KEY_ESCAPE:
            return None


# ═══════════════════════════════════════════════════════════════════════
# Text/number editing overlay
# ═══════════════════════════════════════════════════════════════════════

def _edit_value(label: str, current: str, is_number: bool = False) -> str:
    """Inline edit: shows current value, user types new one."""
    buf = ""

    while True:
        _clear_screen()
        lines = [
            f"  {BOLD}Edit: {label}{NC}\n",
            f"  Current: {DIM}{current}{NC}",
            f"  New:     {buf}\u2588\n",
            f"  {DIM}[Enter] Confirm  [Esc] Cancel{NC}",
        ]
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

        key = _read_key()
        if key == KEY_ENTER:
            if buf:
                if is_number:
                    try:
                        int(buf)
                    except ValueError:
                        continue  # reject non-numeric
                return buf
            return current  # empty = keep current
        elif key == KEY_ESCAPE:
            return current
        elif key == KEY_BACKSPACE:
            buf = buf[:-1]
        elif len(key) == 1 and key.isprintable():
            buf += key


# ═══════════════════════════════════════════════════════════════════════
# Main config screen with live budget
# ═══════════════════════════════════════════════════════════════════════

class Launcher:
    """Main TUI controller."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.script_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))  # krasis repo root
        self.krasis_home = os.environ.get(
            "KRASIS_HOME",
            os.path.join(os.path.expanduser("~"), ".krasis"),
        )
        os.makedirs(self.krasis_home, exist_ok=True)
        self.config_file = os.path.join(self.krasis_home, "config")
        self.models_dir = os.path.join(self.krasis_home, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.hw = detect_hardware()
        self.cfg = LauncherConfig()
        # Auto-downgrade KV dtype if GPU lacks FP8 support (needs SM >= 8.9)
        if not self.hw["has_fp8"]:
            self.cfg.kv_dtype = "bf16"
        self.model_info: Optional[Dict[str, Any]] = None
        self.budget: Optional[Dict[str, Any]] = None
        self.selected_gpus: List[Dict[str, Any]] = []  # subset of hw["gpus"]

    def _compute_default_pp(self, num_layers: int) -> str:
        """Compute PP partition — always PP=1 (all layers on primary GPU).

        Multi-GPU uses Expert Parallelism (EP), not Pipeline Parallelism.
        """
        return str(num_layers)

    def _resolve_selected_gpus(self) -> None:
        """Resolve selected GPU indices to GPU dicts from hardware info."""
        if self.cfg.selected_gpu_indices:
            # Match saved indices against detected GPUs
            hw_indices = {g["index"] for g in self.hw["gpus"]}
            valid = [i for i in self.cfg.selected_gpu_indices if i in hw_indices]
            if valid:
                self.selected_gpus = [
                    g for g in self.hw["gpus"] if g["index"] in valid
                ]
                return
        # Default: use the GPU with the largest VRAM
        best = max(self.hw["gpus"], key=lambda g: g["vram_mb"])
        self.selected_gpus = [best]
        self.cfg.selected_gpu_indices = [best["index"]]

    def _compute_budget(self) -> Optional[Dict[str, Any]]:
        """Compute VRAM/RAM budget from current config."""
        if not self.cfg.model_path:
            return None
        try:
            pp = [int(x.strip()) for x in self.cfg.pp_partition.split(",") if x.strip()]
            if not pp:
                return None
            # Use the smallest VRAM among selected GPUs for worst-case budgeting
            gpu_vram = self.hw["gpu_vram_mb"]
            if self.selected_gpus:
                gpu_vram = min(g["vram_mb"] for g in self.selected_gpus)
            from krasis.vram_budget import compute_launcher_budget
            lgs = self.cfg.layer_group_size
            return compute_launcher_budget(
                model_path=self.cfg.model_path,
                pp_partition=pp,
                layer_group_size=lgs,
                kv_dtype=self.cfg.kv_dtype,
                gpu_expert_bits=self.cfg.gpu_expert_bits,
                attention_quant=self.cfg.attention_quant,
                shared_expert_quant=self.cfg.shared_expert_quant,
                dense_mlp_quant=self.cfg.dense_mlp_quant,
                lm_head_quant=self.cfg.lm_head_quant,
                gpu_vram_mb=gpu_vram,
                total_ram_gb=self.hw["total_ram_gb"],
                kv_cache_mb=self.cfg.kv_cache_mb,
            )
        except Exception:
            return None

    def _render_config_screen(self, cursor: int, editing: bool = False) -> str:
        """Render the full config screen as a string."""
        lines = []

        # Header
        lines.append(f"{BOLD}\u2554{'═' * 71}\u2557{NC}")
        lines.append(f"{BOLD}\u2551{'':>27}{CYAN}Krasis{NC}{BOLD} MoE Server{'':>27}\u2551{NC}")
        lines.append(f"{BOLD}\u255a{'═' * 71}\u255d{NC}")
        lines.append("")

        # Model info
        model_name = os.path.basename(self.cfg.model_path) if self.cfg.model_path else "none"
        if self.model_info:
            mi = self.model_info
            expert_str = f"{mi['experts']} experts" if mi["experts"] else "dense"
            lines.append(f"  Model: {BOLD}{model_name}{NC} ({mi['arch']}, {mi['layers']} layers, {expert_str})")
        else:
            lines.append(f"  Model: {BOLD}{model_name}{NC}")

        # GPU info
        if self.selected_gpus:
            gpu_names = {}
            for g in self.selected_gpus:
                gpu_names[g["name"]] = gpu_names.get(g["name"], 0) + 1
            gpu_parts = []
            for name, count in gpu_names.items():
                vram = next(g["vram_mb"] for g in self.selected_gpus if g["name"] == name)
                gpu_parts.append(f"{count}x {name} ({vram:,} MB)")
            idx_str = ",".join(str(g["index"]) for g in self.selected_gpus)
            lines.append(f"  GPUs:  {' + '.join(gpu_parts)}  [{idx_str}]")
        elif self.hw["gpu_count"] > 0:
            lines.append(
                f"  GPUs:  {self.hw['gpu_count']}x {self.hw['gpu_model']} "
                f"({self.hw['gpu_vram_mb']} MB each)"
            )
        lines.append("")

        # Config options (filter to visible ones for current model)
        native_dtype = (self.model_info or {}).get("native_dtype", "bfloat16")
        visible_options = [opt for opt in OPTIONS if _is_option_visible(opt, self.model_info, self.cfg)]

        lines.append(f"  {DIM}\u2500\u2500\u2500 Configuration \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{NC}")

        for i, opt in enumerate(visible_options):
            val = getattr(self.cfg, opt.key)
            if opt.key == "host":
                display = f"{val}:{self.cfg.port}"
            else:
                display = _format_value(opt, val)

            if i == cursor:
                prefix = f"  {CYAN}\u25b8{NC} "
                label_style = BOLD
            else:
                prefix = "    "
                label_style = ""

            # Quality annotation for quant-related options
            annotation = _quality_annotation(native_dtype, opt.key, val)
            suffix = f"  {annotation}" if annotation else ""

            # KV cache: show token capacity and model's max context
            if opt.key == "kv_cache_mb" and self.model_info:
                mi = self.model_info
                kv_dim = mi.get("kv_dim", 0)
                num_kv_layers = mi.get("num_kv_layers", 0)
                max_ctx = mi.get("max_context", 0)
                if kv_dim > 0 and num_kv_layers > 0:
                    kv_bytes_per_token = kv_dim * num_kv_layers
                    alloc_tokens = (self.cfg.kv_cache_mb * 1024 * 1024) // kv_bytes_per_token if kv_bytes_per_token > 0 else 0
                    suffix = f"  {DIM}(~{_format_tokens(alloc_tokens)} tokens, model max {_format_tokens(max_ctx)}){NC}"

            # Build left part (prefix + label + value + annotation)
            label_part = f"{label_style}{opt.label:<20s}{NC}"
            left = f"{prefix}{label_part}{display}{suffix}"
            lines.append(left)

        lines.append("")

        # Budget display — VRAM and System RAM side by side
        if self.budget:
            b = self.budget
            wr = b["worst_rank"]
            rank = b["ranks"][wr]
            gpu_vram = b["gpu_vram_mb"]

            # Three main VRAM categories
            experts_mb = int(rank.get("expert_buffer_mb", 0))
            attention_mb = int(rank.get("attention_mb", 0))
            overhead_mb = int(
                rank.get("embedding_mb", 0) +
                rank.get("norms_gates_mb", 0) +
                rank.get("shared_expert_mb", 0) +
                rank.get("dense_mlp_mb", 0) +
                rank.get("lm_head_mb", 0) +
                rank.get("flashinfer_mb", 0) +
                rank.get("prefill_workspace_mb", 0) +
                rank.get("cuda_overhead_mb", 0)
            )

            # KV allocation (capped to available VRAM)
            free_before_kv = rank.get("free_mb", 0)
            kv_alloc = int(min(self.cfg.kv_cache_mb, max(0, free_before_kv)))
            kv_label = "fp8" if self.cfg.kv_dtype == "fp8_e4m3" else "bf16"
            kv_alloc_tokens = rank.get("kv_alloc_tokens", rank["kv_tokens"])

            total_used = experts_mb + attention_mb + overhead_mb + kv_alloc

            # HCS coverage: VRAM available for expert caching vs total expert cache
            permanent_mb = attention_mb + overhead_mb + kv_alloc
            free_for_hcs = max(0, int(gpu_vram - permanent_mb))
            total_expert_cache = b.get("ram_gpu_experts_mb", 0)
            hcs_pct = (free_for_hcs / total_expert_cache * 100) if total_expert_cache > 0 else 0

            # Labels
            expert_label = f"INT{self.cfg.gpu_expert_bits}"
            attn_label = "AWQ" if self.cfg.attention_quant == "awq" else "BF16"

            # RAM
            ram_experts_gb = b.get('ram_gpu_experts_mb', 0) / 1024
            ram_tot_gb = b.get('ram_total_mb', 0) / 1024
            sys_ram_gb = b['total_ram_gb']

            COL_W = 36  # visible width per column

            def _pad_col(text, width=COL_W):
                vl = _visible_len(text)
                return text + " " * max(0, width - vl)

            # Left column (VRAM) — 3 categories + KV + total + HCS
            left = [
                f"{DIM}\u2500\u2500 VRAM Budget \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{NC}",
                f"  Experts:     {CYAN}{experts_mb:>8,} MB{NC}  {DIM}({expert_label}){NC}",
                f"  Attention:   {CYAN}{attention_mb:>8,} MB{NC}  {DIM}({attn_label}){NC}",
                f"  Overhead:    {CYAN}{overhead_mb:>8,} MB{NC}",
                f"  KV cache:    {CYAN}{kv_alloc:>8,} MB{NC}  {DIM}({kv_label}){NC}",
                f"  {DIM}\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{NC}",
                f"  Total: {CYAN}{total_used:>8,}{NC} / {int(gpu_vram):,} MB",
                f"  HCS:   {CYAN}{free_for_hcs:>8,} MB{NC} {DIM}(~{hcs_pct:.0f}% coverage){NC}",
            ]
            # Right column (RAM) — expert cache
            right = [
                f"{DIM}\u2500\u2500 System RAM \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{NC}",
                f"  Expert cache:    {GREEN}{ram_experts_gb:>7.1f} GB{NC}",
                "",
                "",
                "",
                f"  {DIM}\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{NC}",
                f"  Total:   {GREEN}{ram_tot_gb:>7.1f}{NC} / {sys_ram_gb:.0f} GB",
                "",
            ]
            for l_line, r_line in zip(left, right):
                lines.append(f"  {_pad_col(l_line)}  {r_line}")

            # Token estimate + over-budget warning below tables
            lines.append(f"    {DIM}~{_format_tokens(kv_alloc_tokens)} tokens {kv_label} KV{NC}")
            free_after_kv = rank.get("free_after_kv_mb", rank["free_mb"])
            if free_after_kv < 0:
                over = int(-free_after_kv)
                lines.append(f"    {RED}\u26a0 OVER BUDGET by {over:,} MB!{NC}")
        else:
            lines.append(f"  {DIM}(budget unavailable){NC}")

        lines.append("")
        lines.append(f"  {DIM}[\u2191\u2193] Navigate  [\u2190\u2192] Change  [Enter] Launch  [L] Load  [S] Save  [q] Quit{NC}")

        return "\n".join(lines)

    def _load_config_screen(self) -> bool:
        """Show a scrolling list of .conf files in CWD. Returns True if a config was loaded."""
        cwd = os.getcwd()
        conf_files = sorted(
            f for f in os.listdir(cwd)
            if f.endswith(".conf") and os.path.isfile(os.path.join(cwd, f))
        )
        if not conf_files:
            # Show brief message then return
            _clear_screen()
            sys.stdout.write(f"\n  No .conf files found in {cwd}\n\n  {DIM}[Any key] Back{NC}\n")
            sys.stdout.flush()
            _read_key()
            return False

        cursor = 0
        while True:
            _clear_screen()
            lines = [f"  {BOLD}Load config from {cwd}:{NC}\n"]
            for i, name in enumerate(conf_files):
                prefix = f"  {CYAN}\u25b8{NC} " if i == cursor else "    "
                hl = BOLD if i == cursor else ""
                lines.append(f"{prefix}{hl}{name}{NC}")
            lines.append(f"\n  {DIM}[\u2191\u2193] Select  [Enter] Load  [Esc] Cancel{NC}")
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()

            key = _read_key()
            if key == KEY_UP:
                cursor = (cursor - 1) % len(conf_files)
            elif key == KEY_DOWN:
                cursor = (cursor + 1) % len(conf_files)
            elif key == KEY_ENTER:
                path = os.path.join(cwd, conf_files[cursor])
                saved = _load_config(path)
                if saved:
                    self.cfg.apply_saved(saved)
                    # Re-resolve GPUs and PP after loading
                    self._resolve_selected_gpus()
                    if self.model_info:
                        ngpus = len(self.selected_gpus) if self.selected_gpus else 1
                        pp_parts = [x.strip() for x in self.cfg.pp_partition.split(",") if x.strip()] if self.cfg.pp_partition else []
                        needs_recompute = not pp_parts or len(pp_parts) != ngpus
                        if needs_recompute:
                            self.cfg.pp_partition = self._compute_default_pp(self.model_info["layers"])
                    # Reload model info if model path changed
                    if self.cfg.model_path:
                        self._read_model_info()
                    self.budget = self._compute_budget()
                return True
            elif key == KEY_ESCAPE or key == KEY_QUIT:
                return False

    def _save_config_screen(self) -> None:
        """Prompt for filename and save current config to CWD."""
        _show_cursor()
        name = _edit_value("Config filename", "my-config.conf")
        _hide_cursor()
        if not name:
            return
        if "." not in os.path.basename(name):
            name += ".conf"
        save_path = os.path.join(os.getcwd(), name)
        _save_config(save_path, self.cfg.to_save_dict())
        # Show confirmation briefly
        _clear_screen()
        sys.stdout.write(f"\n  {GREEN}Saved: {save_path}{NC}\n\n  {DIM}[Any key] Back{NC}\n")
        sys.stdout.flush()
        _read_key()

    def _cycle_value(self, opt: ConfigOption, direction: int) -> None:
        """Cycle a config value left/right."""
        val = getattr(self.cfg, opt.key)

        if opt.opt_type == "cycle" and opt.choices:
            choices = opt.choices
            # Hide FP8 option if GPU doesn't support it
            if opt.key == "kv_dtype" and not self.hw.get("has_fp8", False):
                choices = [c for c in choices if c != "fp8_e4m3"]
            try:
                idx = choices.index(val)
            except ValueError:
                idx = 0
            idx = (idx + direction) % len(choices)
            setattr(self.cfg, opt.key, choices[idx])
        elif opt.opt_type == "number":
            new_val = int(val) + direction * opt.step
            new_val = max(opt.min_val, min(opt.max_val, new_val))
            setattr(self.cfg, opt.key, new_val)

    def run_interactive(self) -> bool:
        """Run the interactive TUI. Returns True if user chose to launch."""
        if not _HAS_TERMIOS:
            print("Error: interactive mode requires a Unix terminal", file=sys.stderr)
            return False

        print(f"Krasis home: {self.krasis_home}")
        print(f"Models dir:  {self.models_dir}")
        print(f"(Set KRASIS_HOME to change, caches can grow large)\n")

        # Step 1: Native model selection (safetensors only)
        # Skip if --model-path was explicitly provided via CLI
        if self.cfg.model_path and os.path.isdir(self.cfg.model_path):
            self._read_model_info()
            print(f"Model: {self.cfg.model_path} (from --model-path)")
        else:
            models = scan_models(self.models_dir, native_only=True)
            if not models:
                print(f"No native models found in {self.models_dir}", file=sys.stderr)
                print(f"Download a model with: huggingface-cli download <model> --local-dir {self.models_dir}/<name>",
                      file=sys.stderr)
                return False

            _hide_cursor()
            try:
                selected = _model_selection_screen(models, self.cfg.model_path)
            finally:
                _show_cursor()

            if selected is None:
                return False
            self.cfg.model_path = selected["path"]
            self.model_info = selected

        # Step 2: GPU selection (always shown, pre-selects saved GPUs)
        if self.hw["gpus"]:
            if self.args.selected_gpus is not None and self.selected_gpus:
                # CLI override — skip interactive GPU selection
                pass
            else:
                preselected = self.cfg.selected_gpu_indices or None
                _hide_cursor()
                try:
                    gpu_indices = _gpu_selection_screen(self.hw["gpus"], preselected)
                finally:
                    _show_cursor()

                if gpu_indices is None:
                    return False
                self.cfg.selected_gpu_indices = gpu_indices
                self._resolve_selected_gpus()
        if not self.selected_gpus:
            self._resolve_selected_gpus()

        # Set/recompute PP partition based on selected GPUs and model layer count
        ngpus = len(self.selected_gpus) if self.selected_gpus else 1
        pp_parts = [x.strip() for x in self.cfg.pp_partition.split(",") if x.strip()] if self.cfg.pp_partition else []
        needs_recompute = not pp_parts or len(pp_parts) != ngpus
        if not needs_recompute and self.model_info:
            # Also recompute if sum doesn't match model's actual layer count
            try:
                pp_sum = sum(int(p) for p in pp_parts)
                if pp_sum != self.model_info["layers"]:
                    needs_recompute = True
            except ValueError:
                needs_recompute = True
        if needs_recompute and self.model_info:
            self.cfg.pp_partition = self._compute_default_pp(self.model_info["layers"])
        if self.hw["cpu_cores"] > 0:
            self.cfg.krasis_threads = min(self.hw["cpu_cores"], 40)

        # Compute initial budget
        self.budget = self._compute_budget()

        cursor = 0
        _hide_cursor()
        try:
            while True:
                visible_options = [opt for opt in OPTIONS if _is_option_visible(opt, self.model_info, self.cfg)]
                n_visible = len(visible_options)

                _clear_screen()
                screen = self._render_config_screen(cursor)
                sys.stdout.write(screen + "\n")
                sys.stdout.flush()

                key = _read_key()

                if key == KEY_UP:
                    cursor = (cursor - 1) % n_visible
                elif key == KEY_DOWN:
                    cursor = (cursor + 1) % n_visible
                elif key in (KEY_LEFT, KEY_RIGHT):
                    opt = visible_options[cursor]
                    direction = -1 if key == KEY_LEFT else 1
                    if opt.opt_type == "cycle":
                        self._cycle_value(opt, direction)
                    elif opt.opt_type == "number":
                        self._cycle_value(opt, direction)
                    elif opt.opt_type == "text":
                        # Open inline editor for text fields on left/right
                        _show_cursor()
                        if opt.key == "host":
                            current = f"{self.cfg.host}:{self.cfg.port}"
                            new_val = _edit_value(opt.label, current)
                            if ":" in new_val:
                                h, p = new_val.rsplit(":", 1)
                                self.cfg.host = h
                                try:
                                    self.cfg.port = int(p)
                                except ValueError:
                                    pass
                            else:
                                self.cfg.host = new_val
                        else:
                            current = str(getattr(self.cfg, opt.key))
                            new_val = _edit_value(opt.label, current)
                            setattr(self.cfg, opt.key, new_val)
                        _hide_cursor()
                    if opt.affects_budget:
                        self.budget = self._compute_budget()
                elif key in ("l", "L"):
                    self._load_config_screen()
                elif key in ("s", "S"):
                    self._save_config_screen()
                elif key == KEY_ENTER:
                    return True
                elif key == KEY_QUIT or key == KEY_ESCAPE:
                    return False
        finally:
            _show_cursor()

    def _read_model_info(self) -> None:
        """Read model info from config.json for display."""
        config_path = os.path.join(self.cfg.model_path, "config.json")
        if not os.path.isfile(config_path):
            return
        try:
            with open(config_path) as f:
                raw = json.load(f)
            cfg = raw.get("text_config", raw)
            # Determine number of dense (non-MoE) layers
            if "first_k_dense_replace" in cfg:
                first_k_dense = cfg["first_k_dense_replace"]
            elif "decoder_sparse_step" in cfg:
                step = cfg["decoder_sparse_step"]
                first_k_dense = 0 if step <= 1 else step
            else:
                first_k_dense = 0

            # Compute KV cache dimensions for max-context estimate
            num_layers = cfg.get("num_hidden_layers", 0)
            is_mla = "kv_lora_rank" in cfg

            # Count layers that need KV cache (hybrid models skip linear attention layers)
            full_attn_interval = cfg.get("full_attention_interval", 0)
            if "layer_types" in cfg:
                num_kv_layers = sum(
                    1 for t in cfg["layer_types"]
                    if t in ("full_attention", "sliding_attention")
                )
            elif full_attn_interval > 0:
                num_kv_layers = sum(
                    1 for i in range(num_layers)
                    if (i + 1) % full_attn_interval == 0
                )
            else:
                num_kv_layers = num_layers

            # Bytes per token per layer in KV cache
            if is_mla:
                kv_dim = cfg.get("kv_lora_rank", 512) + cfg.get("qk_rope_head_dim", 64)
            else:
                num_kv_heads = cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 1))
                head_dim = cfg.get("head_dim", 128)
                kv_dim = num_kv_heads * head_dim * 2  # K + V

            max_context = cfg.get("max_position_embeddings", 131072)

            self.model_info = {
                "name": os.path.basename(self.cfg.model_path),
                "path": self.cfg.model_path,
                "arch": cfg.get("model_type", "?"),
                "layers": num_layers,
                "experts": cfg.get("n_routed_experts", cfg.get("num_experts", 0)),
                "shared_experts": cfg.get("n_shared_experts", 0),
                "dense_layers": first_k_dense,
                "native_dtype": raw.get("torch_dtype", "bfloat16"),
                "ram_gb": 0,
                "num_kv_layers": num_kv_layers,
                "kv_dim": kv_dim,
                "max_context": max_context,
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    def print_summary(self) -> None:
        """Print non-interactive launch summary."""
        model_name = os.path.basename(self.cfg.model_path)
        print(f"\n{BOLD}Krasis Launch Configuration{NC}")
        print(f"  Krasis home:     {self.krasis_home}")
        print(f"  Models dir:      {self.models_dir}")
        print(f"  Model:           {model_name}")
        print(f"  PP partition:    {self.cfg.pp_partition}")
        print(f"  Layer group:     {self.cfg.layer_group_size} layers (double-buffered)")
        print(f"  KV cache:        {self.cfg.kv_cache_mb:,} MB")
        print(f"  KV dtype:        {self.cfg.kv_dtype}")
        print(f"  GPU expert bits: {self.cfg.gpu_expert_bits}")
        print(f"  Attention quant: {self.cfg.attention_quant}")
        print(f"  Shared expert:   {self.cfg.shared_expert_quant}")
        dense_layers = (self.model_info or {}).get("dense_layers", 0)
        if dense_layers > 0:
            print(f"  Dense MLP quant: {self.cfg.dense_mlp_quant}")
        print(f"  LM head quant:   {self.cfg.lm_head_quant}")
        print(f"  VRAM safety:     {self.cfg.vram_safety_margin:,} MB")
        print(f"  Server:          {self.cfg.host}:{self.cfg.port}")
        if self.selected_gpus:
            idx_str = ",".join(str(g["index"]) for g in self.selected_gpus)
            print(f"  GPUs:            {len(self.selected_gpus)}x [{idx_str}]")
        elif self.hw["gpu_count"] > 0:
            print(f"  GPUs:            {self.hw['gpu_count']}x {self.hw['gpu_model']}")

        budget = self._compute_budget()
        if budget:
            wr = budget["worst_rank"]
            rank = budget["ranks"][wr]
            gpu_vram = budget["gpu_vram_mb"]
            experts_mb = int(rank.get("expert_buffer_mb", 0))
            attention_mb = int(rank.get("attention_mb", 0))
            overhead_mb = int(
                rank.get("embedding_mb", 0) + rank.get("norms_gates_mb", 0) +
                rank.get("shared_expert_mb", 0) + rank.get("dense_mlp_mb", 0) +
                rank.get("lm_head_mb", 0) + rank.get("flashinfer_mb", 0) +
                rank.get("prefill_workspace_mb", 0) + rank.get("cuda_overhead_mb", 0)
            )
            kv_label = "fp8" if self.cfg.kv_dtype == "fp8_e4m3" else "bf16"
            print(f"\n  Experts:     {experts_mb:>8,} MB  (INT{self.cfg.gpu_expert_bits})")
            attn_label = "AWQ" if self.cfg.attention_quant == "awq" else "BF16"
            print(f"  Attention:   {attention_mb:>8,} MB  ({attn_label})")
            print(f"  Overhead:    {overhead_mb:>8,} MB")
            print(f"  Total: {rank['total_mb']:>8,.0f} / {gpu_vram:,} MB (rank {wr})")
            permanent_mb = attention_mb + overhead_mb + min(self.cfg.kv_cache_mb, max(0, rank["free_mb"]))
            free_for_hcs = max(0, int(gpu_vram - permanent_mb))
            total_expert_cache = budget.get("ram_gpu_experts_mb", 0)
            hcs_pct = (free_for_hcs / total_expert_cache * 100) if total_expert_cache > 0 else 0
            print(f"  HCS:   {free_for_hcs:>8,} MB  (~{hcs_pct:.0f}% coverage)")
            if rank["free_mb"] > 0:
                print(f"  KV capacity: ~{_format_tokens(rank['kv_tokens'])} tokens ({kv_label})")
            else:
                print(f"  {RED}WARNING: OVER BUDGET by {-rank['free_mb']:,.0f} MB{NC}")
            ram_gb = budget.get('ram_total_mb', 0) / 1024
            print(f"  Expert cache: {ram_gb:.1f} GB / {budget['total_ram_gb']} GB RAM")
        print()

    def launch_server(self, benchmark: bool = False, benchmark_only: bool = False,
                      stress_test: bool = False) -> None:
        """Write a temp config file and exec the Krasis server with --config."""
        import tempfile

        # Write the full config to a temp file
        config_dict = self.cfg.to_save_dict()
        # Add num_gpus (derived from selected GPUs, not stored in LauncherConfig)
        num_gpus = len(self.selected_gpus) if self.selected_gpus else self.hw["gpu_count"]
        config_dict["CFG_NUM_GPUS"] = str(num_gpus)

        fd, config_path = tempfile.mkstemp(prefix="krasis-", suffix=".conf")
        with os.fdopen(fd, "w") as f:
            import datetime
            f.write(f"# Krasis launch config — {datetime.datetime.now().isoformat()}\n")
            for key, val in config_dict.items():
                f.write(f'{key}="{val}"\n')

        # Build minimal command: just --config plus any action flags
        cmd_args = [
            sys.executable, "-m", "krasis.server",
            "--config", config_path,
        ]

        if benchmark or benchmark_only:
            cmd_args.append("--benchmark")
        if benchmark_only:
            cmd_args.append("--benchmark-only")
        if stress_test:
            cmd_args.append("--stress-test")

        # Set CUDA_VISIBLE_DEVICES to selected GPUs
        if self.selected_gpus:
            cvd = ",".join(str(g["index"]) for g in self.selected_gpus)
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd
            print(f"  CUDA_VISIBLE_DEVICES={cvd}")

        # WSL2: LD_LIBRARY_PATH must be set BEFORE execvp so the new process
        # starts with it — glibc caches the library search path at startup,
        # so setting it inside the server's main() is too late for dlopen.
        _wsl_cuda = "/usr/lib/wsl/lib"
        if os.path.isdir(_wsl_cuda):
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if _wsl_cuda not in ld_path:
                os.environ["LD_LIBRARY_PATH"] = f"{_wsl_cuda}:{ld_path}" if ld_path else _wsl_cuda

        print(f"\n{GREEN}Starting Krasis server...{NC}\n")
        print(f"  Config: {config_path}")
        print(f"{DIM}$ {' '.join(cmd_args)}{NC}\n")

        os.execvp(cmd_args[0], cmd_args)


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    from krasis import __version__
    parser = argparse.ArgumentParser(
        description="Krasis — High-performance MoE inference engine",
        epilog=(
            "subcommands (use before any flags):\n"
            "  krasis                  Launch interactive TUI configurator\n"
            "  krasis chat [args]      Chat client (connect to running server)\n"
            "  krasis sanity           Run sanity test prompts against running server\n"
            "  krasis kill             Terminate all running krasis instances\n"
            "\n"
            "chat options:\n"
            "  krasis chat                         Interactive chat (default)\n"
            "  krasis chat --prompt \"question\"      Send prompt, print response, exit\n"
            "  krasis chat --prompt \"q1\" \"q2\"       Multiple prompts, run sequentially\n"
            "  krasis chat --file prompts.txt       Run prompts from file (multi-turn supported)\n"
            "  krasis chat --port 8013              Connect to non-default port\n"
            "  krasis chat --url http://host:port   Connect to remote server\n"
            "  krasis chat --system \"You are...\"    Set system prompt\n"
            "  krasis chat --temperature 0.8        Set temperature (default: 0.6)\n"
            "  krasis chat --max-tokens 32768       Set max tokens (default: 16384)\n"
            "\n"
            "examples:\n"
            "  krasis                              # interactive TUI\n"
            "  krasis --config model.conf          # non-interactive with config file\n"
            "  krasis chat                         # chat with running server\n"
            "  krasis chat --file test.txt         # run prompts from file\n"
            "  krasis sanity                       # run sanity test prompts\n"
            "  krasis kill                         # stop all krasis processes\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", default=None,
                        help="Path to config file (CFG_KEY=\"value\" format). "
                             "Implies --non-interactive.")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Use saved/default config without prompts")
    parser.add_argument("--model-path", default=None,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--pp-partition", default=None,
                        help="Comma-separated layer counts (e.g. '9,9,9')")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs to use")
    parser.add_argument("--selected-gpus", default=None,
                        help="Comma-separated GPU indices to use (e.g. '0,2')")
    parser.add_argument("--layer-group-size", type=int, default=None,
                        help="Layers per streaming group (even number, min 2 for double buffering)")
    parser.add_argument("--kv-cache-mb", type=int, default=None,
                        help="KV cache size in MB (default: 1000)")
    parser.add_argument("--vram-safety-margin", type=int, default=None,
                        help="VRAM safety margin in MB (default: 600)")
    parser.add_argument("--kv-dtype", default=None,
                        help="KV cache dtype: fp8_e4m3 or bf16")
    parser.add_argument("--gpu-expert-bits", type=int, default=None,
                        help="GPU Marlin expert bits: 4 or 8")
    parser.add_argument("--attention-quant", default=None,
                        help="Attention weight quant: bf16 or awq")
    parser.add_argument("--shared-expert-quant", default=None,
                        help="Shared expert quant: bf16 or int8")
    parser.add_argument("--dense-mlp-quant", default=None,
                        help="Dense MLP quant: bf16 or int8")
    parser.add_argument("--lm-head-quant", default=None,
                        help="LM head quant: bf16 or int8")
    parser.add_argument("--krasis-threads", type=int, default=None,
                        help="CPU threads for expert computation")
    parser.add_argument("--host", default=None,
                        help="Server bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None,
                        help="Server port (default: 8012)")
    parser.add_argument("--gguf-path", default=None,
                        help="Path to GGUF file for CPU experts")
    parser.add_argument("--gpu-prefill-threshold", type=int, default=None,
                        help="Min tokens for GPU prefill (default: 300)")
    parser.add_argument("--force-load", action="store_true",
                        help="Override RAM safety checks and load anyway")
    parser.add_argument("--force-rebuild-cache", action="store_true",
                        help="Delete existing expert caches and rebuild from safetensors")
    parser.add_argument("--build-cache", action="store_true",
                        help="Build expert caches (if missing) and exit without starting server")
    parser.add_argument("--session-enabled", action="store_true",
                        help="Enable Session messenger bridge")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run standardized benchmark before starting server")
    parser.add_argument("--benchmark-suite", nargs="?", const="", default=None,
                        help="Run benchmark suite (optional: path to TOML config)")
    parser.add_argument("--skip-setup", action="store_true",
                        help="(ignored — handled by bash wrapper)")
    parser.add_argument("--venv", default=None,
                        help="(ignored — handled by bash wrapper)")
    return parser.parse_args()


def _apply_cli_overrides(cfg: LauncherConfig, args: argparse.Namespace) -> None:
    """Apply CLI arguments as overrides (they take priority over saved/interactive)."""
    if args.model_path is not None:
        cfg.model_path = args.model_path
    if args.selected_gpus is not None:
        try:
            cfg.selected_gpu_indices = [
                int(x.strip()) for x in args.selected_gpus.split(",") if x.strip()
            ]
        except ValueError:
            pass
    if args.pp_partition is not None:
        cfg.pp_partition = args.pp_partition
    if args.layer_group_size is not None:
        cfg.layer_group_size = max(2, args.layer_group_size)
    if args.kv_cache_mb is not None:
        cfg.kv_cache_mb = max(200, args.kv_cache_mb)
    if args.vram_safety_margin is not None:
        cfg.vram_safety_margin = max(500, args.vram_safety_margin)
    if args.kv_dtype is not None:
        cfg.kv_dtype = args.kv_dtype
    if args.gpu_expert_bits is not None:
        cfg.gpu_expert_bits = args.gpu_expert_bits
    if args.attention_quant is not None:
        val = args.attention_quant
        if val in ("int4", "int8"):
            val = "awq"
        cfg.attention_quant = val
    if args.shared_expert_quant is not None:
        cfg.shared_expert_quant = args.shared_expert_quant
    if args.dense_mlp_quant is not None:
        cfg.dense_mlp_quant = args.dense_mlp_quant
    if args.lm_head_quant is not None:
        cfg.lm_head_quant = args.lm_head_quant
    if args.krasis_threads is not None:
        cfg.krasis_threads = args.krasis_threads
    if args.host is not None:
        cfg.host = args.host
    if args.port is not None:
        cfg.port = args.port
    if args.gpu_prefill_threshold is not None:
        cfg.gpu_prefill_threshold = args.gpu_prefill_threshold
    if args.gguf_path is not None:
        cfg.gguf_path = args.gguf_path
    if args.force_load:
        cfg.force_load = True
    if getattr(args, 'force_rebuild_cache', False):
        cfg.force_rebuild_cache = True
    if getattr(args, 'build_cache', False):
        cfg.build_cache = True
    if getattr(args, 'session_enabled', False):
        cfg.session_enabled = True


def _check_gpu_deps():
    """Quick check that GPU dependencies are present. Points to krasis-setup if not."""
    import shutil

    if not shutil.which("nvidia-smi"):
        print(f"{RED}No NVIDIA GPU detected. Krasis requires at least one NVIDIA GPU for prefill.{NC}")
        sys.exit(1)

    # Find nvcc: try multiple common locations so it works even if
    # the user's PATH doesn't include the CUDA toolkit yet.
    _cuda_search_dirs = [
        "/usr/local/cuda/bin",
        "/usr/local/cuda-12.8/bin",
        "/usr/local/cuda-12.6/bin",
        "/usr/local/cuda-12.4/bin",
        "/usr/local/cuda-12.1/bin",
        "/usr/local/cuda-11.8/bin",
        "/usr/bin",
    ]
    nvcc_path = shutil.which("nvcc")
    if not nvcc_path:
        for d in _cuda_search_dirs:
            candidate = os.path.join(d, "nvcc")
            if os.path.isfile(candidate):
                nvcc_path = candidate
                break
    if nvcc_path:
        cuda_bin = os.path.dirname(nvcc_path)
        cuda_home = os.path.dirname(cuda_bin)  # e.g. /usr/local/cuda-12.8
        # Add to PATH so subprocess/ninja can find nvcc
        if cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")
        # Set CUDA_HOME so PyTorch/FlashInfer JIT find the right toolkit
        os.environ["CUDA_HOME"] = cuda_home

        # Clear stale FlashInfer JIT cache if it references a missing nvcc.
        # FlashInfer caches ninja build files with absolute nvcc paths —
        # if nvcc moved (e.g. old /usr/bin/nvcc removed), the cache is broken.
        import glob
        fi_cache = os.path.expanduser("~/.cache/flashinfer")
        if os.path.isdir(fi_cache):
            for build_ninja in glob.glob(os.path.join(fi_cache, "**/build.ninja"), recursive=True):
                try:
                    with open(build_ninja) as f:
                        content = f.read()
                    # Check if the cached nvcc path still exists
                    if "/usr/bin/nvcc" in content and not os.path.isfile("/usr/bin/nvcc"):
                        # Stale cache — remove the containing directory
                        cache_dir = os.path.dirname(build_ninja)
                        import shutil as _shutil
                        _shutil.rmtree(cache_dir, ignore_errors=True)
                except OSError:
                    pass

    problems = []

    # Check nvcc
    has_nvcc = nvcc_path is not None
    if not has_nvcc:
        problems.append("CUDA toolkit (nvcc)")

    # Check ninja
    if not shutil.which("ninja") and not shutil.which("ninja-build"):
        problems.append("ninja")

    # Check CUDA torch
    try:
        import torch
        torch.set_float32_matmul_precision('high')
        if not torch.cuda.is_available():
            problems.append("CUDA-enabled PyTorch")
    except ImportError:
        problems.append("PyTorch")

    # Check GPU packages
    for pkg, import_name in [("flashinfer", "flashinfer"),
                              ("sgl-kernel", "sgl_kernel"),
                              ("sglang", "sglang")]:
        try:
            __import__(import_name)
        except ImportError:
            problems.append(pkg)

    # Check Triton has GPU support (not CPU fallback)
    try:
        import triton
        if not hasattr(triton, 'runtime') or not hasattr(triton.runtime, 'driver'):
            problems.append("triton (no GPU backend)")
        else:
            # Try to get a CUDA driver — this fails if Triton fell back to CPU
            try:
                driver = triton.runtime.driver.active
                if driver is None or 'cpu' in str(type(driver)).lower():
                    problems.append("triton (GPU backend unavailable — check CUDA)")
            except Exception:
                pass  # older triton versions may not have this
    except ImportError:
        problems.append("triton")

    # Check Python.h headers (needed for FlashInfer JIT compilation)
    import sysconfig
    inc = sysconfig.get_path("include")
    if not inc or not os.path.isfile(os.path.join(inc, "Python.h")):
        problems.append(f"python{sys.version_info.major}.{sys.version_info.minor}-dev (Python.h headers)")

    if problems:
        print(f"{RED}Missing GPU dependencies: {', '.join(problems)}{NC}")
        print(f"Run: {BOLD}krasis-setup{NC}")
        print()
        sys.exit(1)


def _do_kill():
    """Terminate all running krasis server processes (and only krasis)."""
    import signal
    import glob as globmod

    my_pid = os.getpid()
    my_ppid = os.getppid()
    killed = []

    # Scan /proc for krasis processes — more reliable than pgrep
    for proc_dir in globmod.glob("/proc/[0-9]*"):
        try:
            pid = int(os.path.basename(proc_dir))
        except ValueError:
            continue

        # Skip self and parent
        if pid == my_pid or pid == my_ppid:
            continue

        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().decode("utf-8", errors="replace").replace("\x00", " ").strip()
        except (OSError, PermissionError):
            continue

        if not cmdline:
            continue

        # Must be a krasis process — match server, chat, launcher, etc.
        # But not editors, shells, or this kill command
        if "krasis" not in cmdline:
            continue
        if "krasis kill" in cmdline:
            continue
        # Skip dev script itself (bash ./dev ...)
        if cmdline.startswith("bash") and "/dev " in cmdline:
            continue
        # Must be an actual krasis server/chat/launcher process
        # NOT anything that just has "krasis" in a file path (e.g. HF downloads to ~/.krasis/)
        is_krasis = False
        if "python" in cmdline and ("krasis.launcher" in cmdline or "krasis.server" in cmdline or "krasis.chat" in cmdline or "-m krasis" in cmdline):
            is_krasis = True
        elif cmdline.strip().startswith("krasis ") or cmdline.strip() == "krasis":
            is_krasis = True
        if not is_krasis:
            continue

        try:
            os.kill(pid, signal.SIGTERM)
            killed.append((pid, cmdline))
            print(f"  Terminated PID {pid}: {cmdline[:80]}")
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"  Permission denied for PID {pid}: {cmdline[:80]}")

    if not killed:
        print("No running krasis processes found.")
    else:
        print(f"\nTerminated {len(killed)} process(es).")

        # Give them a moment to exit, then SIGKILL stragglers
        import time
        time.sleep(2)
        for pid, cmdline in killed:
            try:
                os.kill(pid, 0)  # check if still alive
                os.kill(pid, signal.SIGKILL)
                print(f"  Force-killed PID {pid}")
            except (ProcessLookupError, PermissionError):
                pass


def main():
    # Handle subcommands early — before argparse, no GPU detection needed
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # strip "chat" from argv
        from krasis.chat import main as chat_main
        chat_main()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "sanity":
        sys.argv = [sys.argv[0], "--sanitytest"] + sys.argv[2:]
        from krasis.chat import main as chat_main
        chat_main()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "kill":
        _do_kill()
        return

    args = parse_args()

    # Check GPU dependencies are present
    _check_gpu_deps()

    # Handle --benchmark-suite early (no hardware detection or config needed)
    if args.benchmark_suite is not None:
        from krasis.suite import SuiteRunner
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        config_path = args.benchmark_suite if args.benchmark_suite else None
        if config_path is None:
            config_path = os.path.join(script_dir, "benchmarks", "benchmark_suite.toml")
        if not os.path.isfile(config_path):
            print(f"Error: suite config not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        runner = SuiteRunner(config_path)
        results = runner.run_all()
        if results:
            summary_path = runner.write_summary(results)
            passed = sum(1 for r in results if r.success)
            failed = len(results) - passed
            print(f"\n{BOLD}Suite complete: {passed} passed, {failed} failed{NC}")
            print(f"  Summary: {summary_path}")
        sys.exit(0)

    launcher = Launcher(args)

    # Load config from --config file (non-interactive only)
    if args.config:
        config_path = os.path.expanduser(args.config)
        if not os.path.isfile(config_path):
            print(f"Error: config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        saved = _load_config(config_path)
        if saved:
            launcher.cfg.apply_saved(saved)
        args.non_interactive = True  # --config implies non-interactive
    # (No auto-load from ~/.krasis/config — TUI always starts with hardcoded defaults.
    #  Use --config or the in-TUI "Load Config" option to load a config file.)

    # Force BF16 KV if GPU doesn't support FP8
    if not launcher.hw["has_fp8"] and launcher.cfg.kv_dtype == "fp8_e4m3":
        launcher.cfg.kv_dtype = "bf16"

    # Apply CLI overrides (take priority over saved config)
    _apply_cli_overrides(launcher.cfg, args)

    # Pre-resolve selected GPUs if set via CLI/saved config
    if launcher.cfg.selected_gpu_indices:
        launcher._resolve_selected_gpus()

    if args.non_interactive:
        # Non-interactive: use saved + CLI config, print summary, launch
        if not launcher.cfg.model_path:
            # Try first model in scan dir
            models = scan_models(launcher.models_dir)
            if models:
                launcher.cfg.model_path = models[0]["path"]
            else:
                print(f"Error: no --model-path given and no models in {launcher.models_dir}",
                      file=sys.stderr)
                print(f"Download a model with: huggingface-cli download <model> --local-dir {launcher.models_dir}/<name>",
                      file=sys.stderr)
                sys.exit(1)

        launcher._read_model_info()
        launcher._resolve_selected_gpus()

        # Set/recompute PP if not specified, GPU count mismatch, or layer sum mismatch
        ngpus = len(launcher.selected_gpus) if launcher.selected_gpus else 1
        pp_parts = [x.strip() for x in launcher.cfg.pp_partition.split(",") if x.strip()] if launcher.cfg.pp_partition else []
        needs_recompute = not pp_parts or len(pp_parts) != ngpus
        if not needs_recompute and launcher.model_info:
            try:
                pp_sum = sum(int(p) for p in pp_parts)
                if pp_sum != launcher.model_info["layers"]:
                    needs_recompute = True
            except ValueError:
                needs_recompute = True
        if needs_recompute and launcher.model_info:
            launcher.cfg.pp_partition = launcher._compute_default_pp(
                launcher.model_info["layers"]
            )

        launcher.print_summary()
        launcher.launch_server(benchmark=args.benchmark)
    else:
        # Interactive TUI
        if launcher.run_interactive():
            # Launch mode selection (skip if --benchmark on CLI)
            if args.benchmark:
                launch_mode = "benchmark"
            else:
                _hide_cursor()
                try:
                    launch_mode = _launch_mode_screen()
                finally:
                    _show_cursor()
                if launch_mode is None:
                    print("Aborted.")
                    sys.exit(0)

            if launch_mode == "suite":
                from krasis.suite import SuiteRunner
                config_path = os.path.join(launcher.script_dir, "benchmarks", "benchmark_suite.toml")
                if not os.path.isfile(config_path):
                    print(f"Error: suite config not found: {config_path}", file=sys.stderr)
                    sys.exit(1)
                runner = SuiteRunner(config_path)
                results = runner.run_all()
                if results:
                    summary_path = runner.write_summary(results)
                    passed = sum(1 for r in results if r.success)
                    failed = len(results) - passed
                    print(f"\n{BOLD}Suite complete: {passed} passed, {failed} failed{NC}")
                    print(f"  Summary: {summary_path}")
                sys.exit(0)

            launcher.print_summary()
            launcher.launch_server(
                benchmark=(launch_mode in ("benchmark", "benchmark_only")),
                benchmark_only=(launch_mode == "benchmark_only"),
                stress_test=(launch_mode == "stress_test"),
            )
        else:
            print("Aborted.")
            sys.exit(0)


if __name__ == "__main__":
    main()
