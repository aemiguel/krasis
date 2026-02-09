"""Krasis — hybrid LLM MoE runtime."""

try:
    from krasis.krasis import KrasisEngine, WeightStore, system_check
except ImportError:
    # Native module not built yet
    pass
