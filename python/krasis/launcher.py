"""Krasis TUI Launcher — arrow-key-driven interactive configuration.

Provides model selection, per-component quantization config, live VRAM/RAM
budget display, and server launch. Uses only stdlib + existing krasis modules.

Usage:
    python -m krasis.launcher                         # interactive TUI
    python -m krasis.launcher --non-interactive       # use saved config
    python -m krasis.launcher --model-path /path/to/model
"""

import argparse
from dataclasses import dataclass
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
        "cpu_model": "unknown",
        "cpu_cores": 0,
        "has_avx2": False,
        "total_ram_gb": 0,
    }

    # GPUs via nvidia-smi — per-GPU info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    hw["gpus"].append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "vram_mb": int(parts[2]),
                    })
            hw["gpu_count"] = len(hw["gpus"])
            if hw["gpus"]:
                hw["gpu_model"] = hw["gpus"][0]["name"]
                hw["gpu_vram_mb"] = hw["gpus"][0]["vram_mb"]
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

    Args:
        native_only: If True, only return models that have safetensors files.
    """
    models = []
    if not os.path.isdir(search_dir):
        return models

    for name in sorted(os.listdir(search_dir)):
        model_dir = os.path.join(search_dir, name)
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.isfile(config_path):
            continue

        # Native-only filter: require at least one .safetensors file
        if native_only:
            try:
                has_safetensors = any(
                    f.endswith(".safetensors") for f in os.listdir(model_dir)
                )
            except OSError:
                continue
            if not has_safetensors:
                continue

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

            models.append({
                "name": name,
                "path": model_dir,
                "arch": arch,
                "layers": layers,
                "experts": experts,
                "shared_experts": shared,
                "native_dtype": raw.get("torch_dtype", "bfloat16"),
                "ram_gb": ram_gb,
            })
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

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
    "MODEL_PATH", "CFG_SELECTED_GPUS", "CFG_PP_PARTITION", "CFG_EXPERT_DIVISOR",
    "CFG_KV_DTYPE", "CFG_GPU_EXPERT_BITS", "CFG_CPU_EXPERT_BITS",
    "CFG_ATTENTION_QUANT", "CFG_SHARED_EXPERT_QUANT", "CFG_DENSE_MLP_QUANT",
    "CFG_LM_HEAD_QUANT", "CFG_KRASIS_THREADS", "CFG_HOST", "CFG_PORT",
    "CFG_GPU_PREFILL_THRESHOLD", "CFG_GGUF_PATH", "CFG_FORCE_LOAD",
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
        self.expert_divisor = "auto"  # str "auto" or int
        self.kv_dtype: str = "fp8_e4m3"
        self.gpu_expert_bits: int = 4
        self.cpu_expert_bits: int = 4
        self.attention_quant: str = "int8"
        self.shared_expert_quant: str = "int8"
        self.dense_mlp_quant: str = "int8"
        self.lm_head_quant: str = "int8"
        self.krasis_threads: int = 48
        self.host: str = "0.0.0.0"
        self.port: int = 8080
        self.gpu_prefill_threshold: int = 300
        self.gguf_path: str = ""
        self.force_load: bool = False

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
        if "CFG_EXPERT_DIVISOR" in saved:
            val = saved["CFG_EXPERT_DIVISOR"]
            if val == "auto":
                self.expert_divisor = "auto"
            else:
                try:
                    self.expert_divisor = int(val)
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
            self.attention_quant = saved["CFG_ATTENTION_QUANT"]
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
        if "CFG_FORCE_LOAD" in saved and saved["CFG_FORCE_LOAD"]:
            self.force_load = saved["CFG_FORCE_LOAD"] == "1"

    def to_save_dict(self) -> Dict[str, Any]:
        """Convert to dict for saving."""
        return {
            "MODEL_PATH": self.model_path,
            "CFG_SELECTED_GPUS": ",".join(str(i) for i in self.selected_gpu_indices),
            "CFG_PP_PARTITION": self.pp_partition,
            "CFG_EXPERT_DIVISOR": str(self.expert_divisor),
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
            "CFG_FORCE_LOAD": "1" if self.force_load else "",
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
    ):
        self.label = label
        self.key = key
        self.choices = choices
        self.opt_type = opt_type
        self.affects_budget = affects_budget
        self.suffix = suffix
        self.min_val = min_val
        self.max_val = max_val


# The 13 config options
OPTIONS = [
    ConfigOption("PP partition", "pp_partition", opt_type="text", affects_budget=True),
    ConfigOption("Expert divisor", "expert_divisor",
                 choices=["auto", 1, 2, 3, 4, 8, 16, 32, 0], affects_budget=True,
                 suffix=" ({desc})"),
    ConfigOption("KV dtype", "kv_dtype",
                 choices=["fp8_e4m3", "bf16"], affects_budget=True),
    ConfigOption("GPU expert bits", "gpu_expert_bits",
                 choices=[4, 8], affects_budget=True),
    ConfigOption("CPU expert bits", "cpu_expert_bits",
                 choices=[4, 8], affects_budget=True),
    ConfigOption("Attention quant", "attention_quant",
                 choices=["int8", "bf16"], affects_budget=True),
    ConfigOption("Shared expert", "shared_expert_quant",
                 choices=["int8", "bf16"], affects_budget=True),
    ConfigOption("Dense MLP quant", "dense_mlp_quant",
                 choices=["int8", "bf16"], affects_budget=True),
    ConfigOption("LM head quant", "lm_head_quant",
                 choices=["int8", "bf16"], affects_budget=True),
    ConfigOption("CPU threads", "krasis_threads",
                 opt_type="number", min_val=1, max_val=256),
    ConfigOption("Host", "host", opt_type="text"),
    ConfigOption("Port", "port", opt_type="number", min_val=1, max_val=65535),
    ConfigOption("Prefill threshold", "gpu_prefill_threshold",
                 opt_type="number", min_val=0, max_val=100000),
]

DIVISOR_DESCRIPTIONS = {
    "auto": "auto-optimise",
    0: "chunked",
    1: "persistent",
    2: "grouped(2)",
    3: "grouped(3)",
    4: "grouped(4)",
    8: "grouped(8)",
    16: "grouped(16)",
    32: "grouped(32)",
}


def _format_value(opt: ConfigOption, val: Any) -> str:
    """Format a config value for display."""
    s = str(val)
    if opt.key == "expert_divisor":
        desc = DIVISOR_DESCRIPTIONS.get(val, f"grouped({val})")
        return f"{val} ({desc})"
    return s


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
        current = str(current_val)  # "int8" or "bf16"
        if current == native_label or current == native_dtype:
            return f"{DIM}{native_label} \u2192 {current} \u2014 lossless{NC}"
        elif current == "int8":
            return f"{DIM}{native_label} \u2192 int8 \u2014 ~lossless{NC}"
        elif current == "int4":
            return f"{DIM}{native_label} \u2192 int4 \u2014 {YELLOW}slight loss{NC}{DIM}{NC}"
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

    elif config_key == "cpu_expert_bits":
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
        selected = [True] * len(gpus)  # all enabled by default

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
# CPU expert source selection screen
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CpuExpertChoice:
    """Result from the CPU expert selection screen."""
    source: str      # "build" or "gguf"
    bits: int = 4    # 4 or 8 (for build mode)
    gguf_path: str = ""  # path to GGUF file (for gguf mode)


def _cpu_expert_selection_screen(
    gguf_files: List[Dict[str, Any]],
    preselected_gguf: str = "",
    preselected_bits: int = 4,
) -> Optional[CpuExpertChoice]:
    """Select CPU expert source: build from native or use GGUF.

    Returns CpuExpertChoice or None if cancelled.
    """
    # Build option list: build options first, then GGUF files
    options: List[Dict[str, Any]] = [
        {"source": "build", "bits": 4,
         "label": "Build INT4 cache from native model",
         "detail": "Recommended \u2014 best speed/quality tradeoff"},
        {"source": "build", "bits": 8,
         "label": "Build INT8 cache from native model",
         "detail": "Higher quality, ~2x RAM usage"},
    ]
    for gf in gguf_files:
        options.append({
            "source": "gguf", "gguf_path": gf["path"],
            "label": f"{gf['dir_name']}/{gf['name']}",
            "detail": f"{gf['size_gb']:.1f} GB",
        })

    # Pre-select based on saved config
    cursor = 0
    if preselected_gguf:
        for i, opt in enumerate(options):
            if opt.get("gguf_path") == preselected_gguf:
                cursor = i
                break
    elif preselected_bits == 8:
        cursor = 1

    while True:
        _clear_screen()
        lines = []
        lines.append(f"  {BOLD}Select CPU expert source:{NC}\n")

        for i, opt in enumerate(options):
            # Separator before GGUF section
            if i == 2:
                lines.append(f"\n  {DIM}\u2500\u2500\u2500 Available GGUF files \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{NC}")

            prefix = f"  {CYAN}\u25b8{NC} " if i == cursor else "    "
            hl = BOLD if i == cursor else ""
            lines.append(f"{prefix}{hl}{opt['label']}{NC}  {DIM}{opt['detail']}{NC}")

        if not gguf_files:
            lines.append(f"\n  {DIM}No GGUF files found in models/{NC}")

        lines.append(f"\n  {DIM}[\u2191\u2193] Select  [Enter] Confirm  [q] Quit{NC}")

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

        key = _read_key()
        if key == KEY_UP:
            cursor = (cursor - 1) % len(options)
        elif key == KEY_DOWN:
            cursor = (cursor + 1) % len(options)
        elif key == KEY_ENTER:
            opt = options[cursor]
            return CpuExpertChoice(
                source=opt["source"],
                bits=opt.get("bits", 4),
                gguf_path=opt.get("gguf_path", ""),
            )
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
        self.config_file = os.path.join(self.script_dir, ".krasis_config")
        self.models_dir = os.path.join(self.script_dir, "models")
        self.hw = detect_hardware()
        self.cfg = LauncherConfig()
        self.model_info: Optional[Dict[str, Any]] = None
        self.budget: Optional[Dict[str, Any]] = None
        self.selected_gpus: List[Dict[str, Any]] = []  # subset of hw["gpus"]

    def _compute_default_pp(self, num_layers: int) -> str:
        """Compute balanced PP partition string based on selected GPUs."""
        ngpus = len(self.selected_gpus) if self.selected_gpus else self.hw["gpu_count"]
        if ngpus <= 0:
            return str(num_layers)
        base = num_layers // ngpus
        remainder = num_layers % ngpus
        parts = []
        for i in range(ngpus):
            parts.append(base + (1 if i < remainder else 0))
        return ",".join(str(p) for p in parts)

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
        # Default: use all detected GPUs
        self.selected_gpus = list(self.hw["gpus"])
        self.cfg.selected_gpu_indices = [g["index"] for g in self.selected_gpus]

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
            # For "auto", use chunked (0) for budget estimate
            divisor = self.cfg.expert_divisor if isinstance(self.cfg.expert_divisor, int) else 0
            return compute_launcher_budget(
                model_path=self.cfg.model_path,
                pp_partition=pp,
                expert_divisor=divisor,
                kv_dtype=self.cfg.kv_dtype,
                gpu_expert_bits=self.cfg.gpu_expert_bits,
                cpu_expert_bits=self.cfg.cpu_expert_bits,
                attention_quant=self.cfg.attention_quant,
                shared_expert_quant=self.cfg.shared_expert_quant,
                dense_mlp_quant=self.cfg.dense_mlp_quant,
                lm_head_quant=self.cfg.lm_head_quant,
                gpu_vram_mb=gpu_vram,
                total_ram_gb=self.hw["total_ram_gb"],
            )
        except Exception:
            return None

    def _render_config_screen(self, cursor: int, editing: bool = False) -> str:
        """Render the full config screen as a string."""
        lines = []

        # Header
        lines.append(f"{BOLD}\u2554{'═' * 39}\u2557{NC}")
        lines.append(f"{BOLD}\u2551         {CYAN}Krasis{NC}{BOLD} MoE Server             \u2551{NC}")
        lines.append(f"{BOLD}\u255a{'═' * 39}\u255d{NC}")
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

        # Config options
        native_dtype = (self.model_info or {}).get("native_dtype", "bfloat16")
        lines.append(f"  {DIM}\u2500\u2500\u2500 Configuration \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{NC}")
        for i, opt in enumerate(OPTIONS):
            val = getattr(self.cfg, opt.key)
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

            # Right-align values
            label_part = f"{label_style}{opt.label:<20s}{NC}"
            lines.append(f"{prefix}{label_part}{display}{suffix}")

        lines.append("")

        # Budget display
        if self.budget:
            b = self.budget
            wr = b["worst_rank"]
            rank = b["ranks"][wr]
            gpu_vram = b["gpu_vram_mb"]

            lines.append(f"  {DIM}\u2500\u2500\u2500 VRAM Budget (worst: rank {wr}) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{NC}")
            lines.append(f"    Attention weights:  {rank['attention_mb']:>8,.0f} MB  ({self.cfg.attention_quant})")
            if rank["shared_expert_mb"] > 0:
                lines.append(f"    Shared experts:    {rank['shared_expert_mb']:>8,.0f} MB  ({self.cfg.shared_expert_quant})")
            if rank["dense_mlp_mb"] > 0:
                lines.append(f"    Dense MLP:         {rank['dense_mlp_mb']:>8,.0f} MB  ({self.cfg.dense_mlp_quant})")
            lines.append(f"    Expert buffers:    {rank['expert_buffer_mb']:>8,.0f} MB  (int{self.cfg.gpu_expert_bits}, {rank['expert_mode']})")
            lines.append(f"    Embedding/LM head: {rank['embed_lmhead_mb']:>8,.0f} MB")
            lines.append(f"    Other (norms/gates):{rank['norms_gates_mb']:>7,.0f} MB")
            lines.append(f"    CUDA overhead:     {rank['cuda_overhead_mb']:>8,} MB")
            lines.append(f"    {DIM}{'─' * 40}{NC}")
            lines.append(f"    Total:             {rank['total_mb']:>8,.0f} MB  / {gpu_vram:,} MB")

            if rank["free_mb"] > 0:
                kv_label = "fp8" if self.cfg.kv_dtype == "fp8_e4m3" else "bf16"
                lines.append(
                    f"    Free for KV:       {rank['free_mb']:>8,.0f} MB  "
                    f"{GREEN}\u2192 ~{_format_tokens(rank['kv_tokens'])} tokens ({kv_label}){NC}"
                )
            else:
                over = -rank["free_mb"]
                lines.append(f"    {RED}\u26a0 OVER BUDGET by {over:,.0f} MB!{NC}")

            # RAM
            lines.append("")
            lines.append(f"  {DIM}\u2500\u2500\u2500 System RAM \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{NC}")
            lines.append(
                f"    CPU experts (int{self.cfg.cpu_expert_bits}): "
                f"{b['cpu_expert_gb']:>8.1f} GB  / {b['total_ram_gb']} GB available"
            )
        else:
            lines.append(f"  {DIM}(budget unavailable){NC}")

        lines.append("")
        lines.append(f"  {DIM}[\u2191\u2193] Navigate  [\u2190\u2192] Change  [Enter] Launch  [q] Quit{NC}")

        return "\n".join(lines)

    def _cycle_value(self, opt: ConfigOption, direction: int) -> None:
        """Cycle a config value left/right."""
        val = getattr(self.cfg, opt.key)

        if opt.opt_type == "cycle" and opt.choices:
            try:
                idx = opt.choices.index(val)
            except ValueError:
                idx = 0
            idx = (idx + direction) % len(opt.choices)
            setattr(self.cfg, opt.key, opt.choices[idx])
        elif opt.opt_type == "number":
            new_val = int(val) + direction
            new_val = max(opt.min_val, min(opt.max_val, new_val))
            setattr(self.cfg, opt.key, new_val)

    def run_interactive(self) -> bool:
        """Run the interactive TUI. Returns True if user chose to launch."""
        if not _HAS_TERMIOS:
            print("Error: interactive mode requires a Unix terminal", file=sys.stderr)
            return False

        # Step 1: Native model selection (safetensors only)
        models = scan_models(self.models_dir, native_only=True)
        if not models:
            print(f"No native models found in {self.models_dir}", file=sys.stderr)
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

        # Step 2: CPU expert source selection (build INT4/INT8 or use GGUF)
        gguf_files = scan_gguf_files(self.models_dir)
        _hide_cursor()
        try:
            cpu_choice = _cpu_expert_selection_screen(
                gguf_files,
                preselected_gguf=self.cfg.gguf_path,
                preselected_bits=self.cfg.cpu_expert_bits,
            )
        finally:
            _show_cursor()

        if cpu_choice is None:
            return False
        if cpu_choice.source == "build":
            self.cfg.cpu_expert_bits = cpu_choice.bits
            self.cfg.gguf_path = ""
        else:
            self.cfg.gguf_path = cpu_choice.gguf_path

        # Step 3: GPU selection (always shown, pre-selects saved GPUs)
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

        # Set/recompute PP partition based on selected GPUs
        ngpus = len(self.selected_gpus) if self.selected_gpus else 1
        pp_parts = [x.strip() for x in self.cfg.pp_partition.split(",") if x.strip()] if self.cfg.pp_partition else []
        if (not pp_parts or len(pp_parts) != ngpus) and self.model_info:
            self.cfg.pp_partition = self._compute_default_pp(self.model_info["layers"])
        if self.hw["cpu_cores"] > 0:
            self.cfg.krasis_threads = min(self.hw["cpu_cores"], 16)

        # Compute initial budget
        self.budget = self._compute_budget()

        cursor = 0
        _hide_cursor()
        try:
            while True:
                _clear_screen()
                screen = self._render_config_screen(cursor)
                sys.stdout.write(screen + "\n")
                sys.stdout.flush()

                key = _read_key()

                if key == KEY_UP:
                    cursor = (cursor - 1) % len(OPTIONS)
                elif key == KEY_DOWN:
                    cursor = (cursor + 1) % len(OPTIONS)
                elif key == KEY_LEFT:
                    opt = OPTIONS[cursor]
                    if opt.opt_type == "cycle":
                        self._cycle_value(opt, -1)
                    elif opt.opt_type == "number":
                        self._cycle_value(opt, -1)
                    if opt.affects_budget:
                        self.budget = self._compute_budget()
                elif key == KEY_RIGHT:
                    opt = OPTIONS[cursor]
                    if opt.opt_type == "cycle":
                        self._cycle_value(opt, 1)
                    elif opt.opt_type == "number":
                        self._cycle_value(opt, 1)
                    if opt.affects_budget:
                        self.budget = self._compute_budget()
                elif key == "e" or (key == KEY_ENTER and OPTIONS[cursor].opt_type in ("text", "number")):
                    opt = OPTIONS[cursor]
                    if opt.opt_type in ("text", "number"):
                        _show_cursor()
                        current = str(getattr(self.cfg, opt.key))
                        new_val = _edit_value(
                            opt.label, current,
                            is_number=(opt.opt_type == "number"),
                        )
                        _hide_cursor()
                        if opt.opt_type == "number":
                            try:
                                setattr(self.cfg, opt.key, int(new_val))
                            except ValueError:
                                pass
                        else:
                            setattr(self.cfg, opt.key, new_val)
                        if opt.affects_budget:
                            self.budget = self._compute_budget()
                    elif key == KEY_ENTER:
                        # Enter on non-text option = launch
                        return True
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
            self.model_info = {
                "name": os.path.basename(self.cfg.model_path),
                "path": self.cfg.model_path,
                "arch": cfg.get("model_type", "?"),
                "layers": cfg.get("num_hidden_layers", 0),
                "experts": cfg.get("n_routed_experts", cfg.get("num_experts", 0)),
                "shared_experts": cfg.get("n_shared_experts", 0),
                "native_dtype": raw.get("torch_dtype", "bfloat16"),
                "ram_gb": 0,
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    def print_summary(self) -> None:
        """Print non-interactive launch summary."""
        model_name = os.path.basename(self.cfg.model_path)
        print(f"\n{BOLD}Krasis Launch Configuration{NC}")
        print(f"  Model:           {model_name}")
        if self.cfg.gguf_path:
            print(f"  CPU experts:     GGUF ({os.path.basename(self.cfg.gguf_path)})")
        else:
            print(f"  CPU experts:     Build INT{self.cfg.cpu_expert_bits} from native")
        print(f"  PP partition:    {self.cfg.pp_partition}")
        print(f"  Expert divisor:  {self.cfg.expert_divisor}")
        print(f"  KV dtype:        {self.cfg.kv_dtype}")
        print(f"  GPU expert bits: {self.cfg.gpu_expert_bits}")
        print(f"  CPU expert bits: {self.cfg.cpu_expert_bits}")
        print(f"  Attention quant: {self.cfg.attention_quant}")
        print(f"  Shared expert:   {self.cfg.shared_expert_quant}")
        print(f"  Dense MLP quant: {self.cfg.dense_mlp_quant}")
        print(f"  LM head quant:   {self.cfg.lm_head_quant}")
        print(f"  CPU threads:     {self.cfg.krasis_threads}")
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
            print(f"\n  VRAM: {rank['total_mb']:,.0f} MB / {gpu_vram:,} MB (rank {wr})")
            if rank["free_mb"] > 0:
                kv_label = "fp8" if self.cfg.kv_dtype == "fp8_e4m3" else "bf16"
                print(f"  KV capacity: ~{_format_tokens(rank['kv_tokens'])} tokens ({kv_label})")
            else:
                print(f"  {RED}WARNING: OVER BUDGET by {-rank['free_mb']:,.0f} MB{NC}")
            print(f"  CPU experts: {budget['cpu_expert_gb']:.1f} GB / {budget['total_ram_gb']} GB")
        print()

    def launch_server(self) -> None:
        """Build args and exec the Krasis server."""
        num_gpus = len(self.selected_gpus) if self.selected_gpus else self.hw["gpu_count"]

        is_auto = self.cfg.expert_divisor == "auto"

        cmd_args = [
            sys.executable, "-m", "krasis.server",
            "--model-path", self.cfg.model_path,
            "--pp-partition", self.cfg.pp_partition,
            "--num-gpus", str(num_gpus),
            "--kv-dtype", self.cfg.kv_dtype,
            "--gpu-expert-bits", str(self.cfg.gpu_expert_bits),
            "--cpu-expert-bits", str(self.cfg.cpu_expert_bits),
            "--attention-quant", self.cfg.attention_quant,
            "--shared-expert-quant", self.cfg.shared_expert_quant,
            "--dense-mlp-quant", self.cfg.dense_mlp_quant,
            "--lm-head-quant", self.cfg.lm_head_quant,
            "--krasis-threads", str(self.cfg.krasis_threads),
            "--host", self.cfg.host,
            "--port", str(self.cfg.port),
            "--gpu-prefill-threshold", str(self.cfg.gpu_prefill_threshold),
        ]

        if is_auto:
            cmd_args.extend(["--strategy", "auto"])
        else:
            cmd_args.extend(["--expert-divisor", str(self.cfg.expert_divisor)])

        if self.cfg.gguf_path:
            cmd_args.extend(["--gguf-path", self.cfg.gguf_path])
        if self.cfg.force_load:
            cmd_args.append("--force-load")

        # Set CUDA_VISIBLE_DEVICES to selected GPUs
        if self.selected_gpus:
            cvd = ",".join(str(g["index"]) for g in self.selected_gpus)
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd
            print(f"  CUDA_VISIBLE_DEVICES={cvd}")

        print(f"\n{GREEN}Starting Krasis server...{NC}\n")
        print(f"{DIM}$ {' '.join(cmd_args)}{NC}\n")

        os.execvp(cmd_args[0], cmd_args)


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Krasis — Interactive MoE inference server launcher",
    )
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
    parser.add_argument("--expert-divisor", default=None,
                        help="Expert loading: auto, 0=chunked, 1=persistent, >=2=layer-grouped")
    parser.add_argument("--kv-dtype", default=None,
                        help="KV cache dtype: fp8_e4m3 or bf16")
    parser.add_argument("--gpu-expert-bits", type=int, default=None,
                        help="GPU Marlin expert bits: 4 or 8")
    parser.add_argument("--cpu-expert-bits", type=int, default=None,
                        help="CPU expert bits: 4 or 8")
    parser.add_argument("--attention-quant", default=None,
                        help="Attention weight quant: bf16 or int8")
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
                        help="Server port (default: 8080)")
    parser.add_argument("--gguf-path", default=None,
                        help="Path to GGUF file for CPU experts")
    parser.add_argument("--gpu-prefill-threshold", type=int, default=None,
                        help="Min tokens for GPU prefill (default: 300)")
    parser.add_argument("--force-load", action="store_true",
                        help="Force reload cached weights")
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
    if args.expert_divisor is not None:
        if args.expert_divisor == "auto":
            cfg.expert_divisor = "auto"
        else:
            try:
                cfg.expert_divisor = int(args.expert_divisor)
            except ValueError:
                pass
    if args.kv_dtype is not None:
        cfg.kv_dtype = args.kv_dtype
    if args.gpu_expert_bits is not None:
        cfg.gpu_expert_bits = args.gpu_expert_bits
    if args.cpu_expert_bits is not None:
        cfg.cpu_expert_bits = args.cpu_expert_bits
    if args.attention_quant is not None:
        cfg.attention_quant = args.attention_quant
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


def main():
    args = parse_args()

    launcher = Launcher(args)

    # Load saved config
    saved = _load_config(launcher.config_file)
    if saved:
        launcher.cfg.apply_saved(saved)

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
                sys.exit(1)

        launcher._read_model_info()
        launcher._resolve_selected_gpus()

        # Set/recompute PP if not specified or GPU count mismatch
        ngpus = len(launcher.selected_gpus) if launcher.selected_gpus else 1
        pp_parts = [x.strip() for x in launcher.cfg.pp_partition.split(",") if x.strip()] if launcher.cfg.pp_partition else []
        if (not pp_parts or len(pp_parts) != ngpus) and launcher.model_info:
            launcher.cfg.pp_partition = launcher._compute_default_pp(
                launcher.model_info["layers"]
            )

        launcher.print_summary()
        _save_config(launcher.config_file, launcher.cfg.to_save_dict())
        launcher.launch_server()
    else:
        # Interactive TUI
        if launcher.run_interactive():
            _save_config(launcher.config_file, launcher.cfg.to_save_dict())
            launcher.print_summary()
            launcher.launch_server()
        else:
            print("Aborted.")
            sys.exit(0)


if __name__ == "__main__":
    main()
