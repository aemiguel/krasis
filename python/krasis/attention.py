"""Attention weight types and linear dispatch.

GQA and MLA attention are handled by the Rust prefill engine.
This module only provides MarlinWeight and _linear for weight loading.
"""

import logging
from typing import Optional, Tuple

import torch

from krasis.weight_loader import int8_linear

logger = logging.getLogger(__name__)


class MarlinWeight:
    """Marlin-packed weight for use with gptq_marlin_gemm.

    Stores packed quantized weight, permuted scales, workspace buffer,
    scalar type, and original dimensions (N=output, K=input).
    """
    __slots__ = ('packed', 'scales', 'workspace', 'scalar_type', 'n', 'k')

    def __init__(self, packed, scales, workspace, scalar_type, n, k):
        self.packed = packed
        self.scales = scales
        self.workspace = workspace
        self.scalar_type = scalar_type
        self.n = n
        self.k = k


def _linear(x: torch.Tensor, weight_data) -> torch.Tensor:
    """Dispatch to Marlin GEMM, INT8, or BF16 linear based on weight type."""
    if isinstance(weight_data, MarlinWeight):
        from krasis.marlin_utils import gptq_marlin_gemm
        M = x.shape[0]
        return gptq_marlin_gemm(
            a=x.contiguous(),
            c=None,
            b_q_weight=weight_data.packed,
            b_scales=weight_data.scales,
            global_scale=None,
            b_zeros=None,
            g_idx=None,
            perm=None,
            workspace=weight_data.workspace,
            b_q_type=weight_data.scalar_type,
            size_m=M,
            size_n=weight_data.n,
            size_k=weight_data.k,
            is_k_full=True,
            use_fp32_reduce=True,
        )
    if isinstance(weight_data, tuple):
        return int8_linear(x, *weight_data)
    return torch.nn.functional.linear(x, weight_data)
