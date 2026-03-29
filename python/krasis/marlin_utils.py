"""Local Marlin GEMM utilities — replaces sgl_kernel and sglang dependencies.

Provides:
  - ScalarType: packed integer type descriptor matching sgl_kernel C++ layout
  - marlin_make_workspace: allocates Marlin GEMM workspace buffer
  - get_scalar_type: returns ScalarType for common quantization configs
  - gptq_marlin_gemm: calls vendored libkrasis_marlin.so via ctypes

These are used by AWQ calibration and weight loading only (one-time startup).
Runtime inference uses the Rust prefill engine which dlopens the .so directly.
"""

import ctypes
import functools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ── ScalarType (matches sgl_kernel C++ layout) ──────────────────────────────

@dataclass(frozen=True)
class ScalarType:
    """Packed scalar type descriptor matching sgl_kernel's ScalarType.

    The .id property produces the same int64 as sgl_kernel's ScalarType,
    used for FFI with Marlin GEMM kernels.
    """
    exponent: int
    mantissa: int
    signed: bool
    bias: int

    @functools.cached_property
    def id(self) -> int:
        """Pack fields into int64 matching sgl_kernel C++ ScalarType::from_id."""
        val = 0
        offset = 0
        def pack(member, bit_width):
            nonlocal val, offset
            val |= (int(member) & ((1 << bit_width) - 1)) << offset
            offset += bit_width
        pack(self.exponent, 8)
        pack(self.mantissa, 8)
        pack(self.signed, 1)
        pack(self.bias, 32)
        pack(False, 1)   # _finite_values_only
        pack(1, 8)       # nan_repr = IEEE_754 = 1
        return val

    @property
    def size_bits(self) -> int:
        return self.exponent + self.mantissa + int(self.signed)


# Pre-built scalar types matching sgl_kernel.scalar_type.scalar_types
_uint4b8 = ScalarType(exponent=0, mantissa=4, signed=False, bias=8)
_uint8b128 = ScalarType(exponent=0, mantissa=8, signed=False, bias=128)


def get_scalar_type(num_bits: int, has_zp: bool = False) -> ScalarType:
    """Get ScalarType for Marlin quantization config."""
    if has_zp:
        assert num_bits == 4
        return ScalarType(exponent=0, mantissa=4, signed=False, bias=0)
    return _uint4b8 if num_bits == 4 else _uint8b128


# ── Workspace ────────────────────────────────────────────────────────────────

def marlin_make_workspace(device: torch.device, max_blocks_per_sm: int = 1) -> torch.Tensor:
    """Allocate Marlin GEMM workspace (SM count * max_blocks_per_sm ints)."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm, dtype=torch.int, device=device,
                       requires_grad=False)


# ── Vendored Marlin GEMM via ctypes ─────────────────────────────────────────

_marlin_lib = None
_marlin_mm_fn = None


def _find_vendored_so() -> Optional[str]:
    """Find libkrasis_marlin.so in known locations."""
    candidates = []

    # 1. KRASIS_MARLIN_SO env override
    env_path = os.environ.get("KRASIS_MARLIN_SO")
    if env_path:
        candidates.append(env_path)

    # 2. Next to the krasis Python package (maturin build output)
    pkg_dir = Path(__file__).parent
    candidates.append(str(pkg_dir / "libkrasis_marlin.so"))
    candidates.append(str(pkg_dir.parent / "krasis.libs" / "libkrasis_marlin.so"))

    # 3. Cargo build output directories
    repo_root = pkg_dir.parent.parent
    for build_dir in sorted(repo_root.glob("target/release/build/krasis-*/out/libkrasis_marlin.so")):
        candidates.append(str(build_dir))

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _load_marlin_lib():
    """Load vendored libkrasis_marlin.so and resolve function pointer."""
    global _marlin_lib, _marlin_mm_fn

    so_path = _find_vendored_so()
    if so_path is None:
        raise RuntimeError(
            "Cannot find libkrasis_marlin.so in the installed package or repo build outputs.\n"
            "Set KRASIS_MARLIN_SO=/path/to/libkrasis_marlin.so to override."
        )

    _marlin_lib = ctypes.CDLL(so_path)
    _marlin_mm_fn = _marlin_lib.krasis_marlin_mm_bf16

    # void krasis_marlin_mm_bf16(
    #   A, B, C, C_tmp, b_bias, s, s2, zp, g_idx, perm, a_tmp, workspace,
    #   q_type_ptr, size_m, size_n, size_k, has_bias, has_act_order,
    #   is_k_full, has_zp, num_groups, group_size, dev, stream,
    #   thread_k, thread_n, sms, use_atomic, is_zp_float)
    _marlin_mm_fn.restype = None
    logger.info("Loaded vendored Marlin GEMM from %s", so_path)


def gptq_marlin_gemm(
    a: torch.Tensor,
    c: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    b_zeros: Optional[torch.Tensor],
    g_idx: Optional[torch.Tensor],
    perm: Optional[torch.Tensor],
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool,
    use_fp32_reduce: bool = True,
) -> torch.Tensor:
    """Call vendored Marlin GEMM kernel via ctypes.

    Matches sgl_kernel.gptq_marlin_gemm API for drop-in replacement.
    """
    global _marlin_lib, _marlin_mm_fn
    if _marlin_mm_fn is None:
        _load_marlin_lib()

    # Allocate output if not provided
    if c is None:
        c = torch.empty(size_m, size_n, dtype=torch.bfloat16, device=a.device)

    def _ptr(t):
        """Get raw CUDA pointer from tensor, or NULL."""
        if t is None:
            return ctypes.c_void_p(0)
        return ctypes.c_void_p(t.data_ptr())

    # ScalarType is passed as a pointer to the raw C struct matching scalar_type.hpp:
    #   uint8_t exponent;        // offset 0
    #   uint8_t mantissa;        // offset 1
    #   bool signed_;            // offset 2
    #   [pad 1 byte]             // offset 3
    #   int32_t bias;            // offset 4-7
    #   bool finite_values_only; // offset 8
    #   uint8_t nan_repr;        // offset 9 (NAN_IEEE_754=1)
    #   [pad 2 bytes]            // offset 10-11
    class _CScalarType(ctypes.Structure):
        _fields_ = [
            ("exponent", ctypes.c_uint8),
            ("mantissa", ctypes.c_uint8),
            ("signed_", ctypes.c_uint8),
            ("_pad0", ctypes.c_uint8),
            ("bias", ctypes.c_int32),
            ("finite_values_only", ctypes.c_uint8),
            ("nan_repr", ctypes.c_uint8),
            ("_pad1", ctypes.c_uint8 * 2),
        ]
    q_type_buf = _CScalarType(
        exponent=b_q_type.exponent,
        mantissa=b_q_type.mantissa,
        signed_=int(b_q_type.signed),
        _pad0=0,
        bias=b_q_type.bias,
        finite_values_only=0,
        nan_repr=1,  # NAN_IEEE_754
        _pad1=(ctypes.c_uint8 * 2)(0, 0),
    )

    device_idx = a.device.index if a.device.index is not None else 0
    stream = torch.cuda.current_stream(a.device).cuda_stream
    sms = torch.cuda.get_device_properties(a.device).multi_processor_count

    num_groups = b_scales.shape[-2] if b_scales.dim() >= 2 else 1
    group_size = size_k // num_groups if num_groups > 0 else size_k

    # C signature: (A, B, C, C_tmp, s, s2, zp, g_idx, perm, a_tmp,
    #               prob_m, prob_n, prob_k, lda, workspace, q_type_ptr,
    #               has_act_order, is_k_full, has_zp,
    #               num_groups, group_size, dev, stream_ptr,
    #               thread_k, thread_n, sms,
    #               use_atomic_add, use_fp32_reduce, is_zp_float)
    lda = size_k  # leading dimension of A (row-major, K columns)
    _marlin_mm_fn(
        _ptr(a),                           # A
        _ptr(b_q_weight),                  # B
        _ptr(c),                           # C
        ctypes.c_void_p(0),                # C_tmp
        _ptr(b_scales),                    # s (scales)
        ctypes.c_void_p(0),                # s2 (scale2, not used)
        _ptr(b_zeros),                     # zp (zeros)
        _ptr(g_idx),                       # g_idx
        _ptr(perm),                        # perm
        ctypes.c_void_p(0),                # a_tmp
        ctypes.c_int(size_m),              # prob_m
        ctypes.c_int(size_n),              # prob_n
        ctypes.c_int(size_k),              # prob_k
        ctypes.c_int(lda),                 # lda
        _ptr(workspace),                   # workspace
        ctypes.cast(ctypes.pointer(q_type_buf), ctypes.c_void_p),  # q_type_ptr
        ctypes.c_bool(g_idx is not None),  # has_act_order
        ctypes.c_bool(is_k_full),          # is_k_full
        ctypes.c_bool(b_zeros is not None), # has_zp
        ctypes.c_int(num_groups),          # num_groups
        ctypes.c_int(group_size),          # group_size
        ctypes.c_int(device_idx),          # dev
        ctypes.c_void_p(stream),           # stream_ptr
        ctypes.c_int(-1),                  # thread_k (auto)
        ctypes.c_int(-1),                  # thread_n (auto)
        ctypes.c_int(sms),                 # sms (SM count from device)
        ctypes.c_bool(use_fp32_reduce),    # use_atomic_add
        ctypes.c_bool(use_fp32_reduce),    # use_fp32_reduce
        ctypes.c_bool(False),              # is_zp_float
    )

    return c
