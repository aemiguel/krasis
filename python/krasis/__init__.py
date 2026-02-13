"""Krasis — hybrid LLM MoE runtime."""

try:
    from krasis.krasis import KrasisEngine, WeightStore, system_check
except ImportError:
    # Native module not built yet
    pass

try:
    from krasis.sglang_bridge import KrasisMoEWrapper
except ImportError:
    # torch/numpy not available
    pass

try:
    from krasis.gpu_prefill import GpuPrefillManager
except ImportError:
    # SGLang/sgl_kernel not available
    pass

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
