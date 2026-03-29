"""Krasis — hybrid LLM MoE runtime."""

from importlib.metadata import version as _pkg_version, PackageNotFoundError
import os
from pathlib import Path


def _configure_vendored_sidecars() -> None:
    pkg_dir = Path(__file__).resolve().parent
    sidecars = {
        "KRASIS_MARLIN_SO": "libkrasis_marlin.so",
        "KRASIS_LIBKRASIS_FLASH_ATTN_SO": "libkrasis_flash_attn.so",
        "KRASIS_LIBKRASIS_FLA_SO": "libkrasis_fla.so",
    }
    search_roots = [
        pkg_dir,
        pkg_dir.parent / "krasis.libs",
    ]
    for env_key, filename in sidecars.items():
        if os.environ.get(env_key):
            continue
        for root in search_roots:
            candidate = root / filename
            if candidate.is_file():
                os.environ[env_key] = str(candidate)
                break


_configure_vendored_sidecars()

try:
    __version__ = _pkg_version("krasis")
except PackageNotFoundError:
    __version__ = "dev"

try:
    from krasis.krasis import KrasisEngine, WeightStore, RustServer, system_check, bench_decode_synthetic
    try:
        from krasis.krasis import GpuDecodeStore
    except ImportError:
        pass  # Built without CUDA feature
    try:
        from krasis.krasis import VramMonitor
    except ImportError:
        pass  # Built without CUDA feature
except ImportError:
    # Native module not built yet
    pass

# GpuPrefillManager removed — Rust prefill engine replaces it

try:
    from krasis.vram_budget import compute_vram_budget, compute_launcher_budget
except ImportError:
    pass

# Standalone server modules
try:
    from krasis.config import ModelConfig, QuantConfig
    from krasis.model import KrasisModel
except ImportError:
    pass
