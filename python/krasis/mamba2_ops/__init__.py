# Vendored Mamba2 SSD Triton kernels for parallel chunked prefill.
# Adapted from sglang/vllm, originally from Tri Dao's mamba_ssm package.
# License: Apache-2.0
#
# These kernels implement the Structured State Space Duality (SSD) algorithm
# for efficient parallel Mamba2 prefill on GPU. The same kernels used by
# vLLM and SGLang for production Mamba2 inference.

from .ssd_combined import mamba_chunk_scan_combined
from .selective_state_update import selective_state_update

__all__ = [
    "mamba_chunk_scan_combined",
    "selective_state_update",
]
