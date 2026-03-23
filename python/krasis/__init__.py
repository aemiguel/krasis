"""Krasis — hybrid LLM MoE runtime."""

from importlib.metadata import version as _pkg_version, PackageNotFoundError
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
