"""Shared control-plane metadata for attention backends.

This module centralizes the user-facing backend surface and cache layout rules
without forcing different backends into one runtime tensor contract.
"""

from dataclasses import dataclass
import json
import math
import os
import time
from typing import Dict, List, Optional

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from krasis.config import cache_dir_for_model
from krasis.krasis import (
    hqq4_init_group_ptr,
    hqq4_quantize_tensor_ptr,
    hqq4_rmse_group_ptr,
    hqq4_solve_group_ptr,
)


ATTENTION_QUANT_CHOICES = ("bf16", "awq", "hqq4")

HQQ_ATTENTION_CACHE_VERSION = 5
HQQ_ATTENTION_CACHE_DIRNAME = f"attention_hqq_v{HQQ_ATTENTION_CACHE_VERSION}"
HQQ_ATTENTION_MANIFEST = "manifest.json"
HQQ_ATTENTION_PENDING_MANIFEST = "manifest.build.json"
HQQ_DEFAULT_GROUP_SIZE = 128
HQQ_DEFAULT_AXIS = 1
HQQ_LAYOUT = "row_major_axis1_grouped_uint4_packed"
HQQ_ENABLED_NBITS = (4,)


@dataclass(frozen=True)
class AttentionBackendSpec:
    quant: str
    label: str
    quantized: bool
    requires_template: bool
    runtime_ready: bool
    nbits: Optional[int] = None


def get_attention_backend_spec(attention_quant: str) -> AttentionBackendSpec:
    if attention_quant == "bf16":
        return AttentionBackendSpec(
            quant="bf16",
            label="BF16",
            quantized=False,
            requires_template=False,
            runtime_ready=True,
        )
    if attention_quant == "awq":
        return AttentionBackendSpec(
            quant="awq",
            label="AWQ",
            quantized=True,
            requires_template=True,
            runtime_ready=True,
            nbits=4,
        )
    if attention_quant == "hqq4":
        return AttentionBackendSpec(
            quant="hqq4",
            label="HQQ4",
            quantized=True,
            requires_template=False,
            runtime_ready=True,
            nbits=4,
        )
    raise ValueError(
        f"Unsupported attention quant '{attention_quant}'. "
        f"Use one of: {', '.join(ATTENTION_QUANT_CHOICES)}."
    )


def attention_quant_label(attention_quant: str) -> str:
    return get_attention_backend_spec(attention_quant).label


def attention_quant_nbits(attention_quant: str) -> Optional[int]:
    return get_attention_backend_spec(attention_quant).nbits


def is_quantized_attention(attention_quant: str) -> bool:
    return get_attention_backend_spec(attention_quant).quantized


def is_hqq_attention(attention_quant: str) -> bool:
    return attention_quant.startswith("hqq")


def hqq_backend_name(nbits: int) -> str:
    return f"hqq{nbits}"


def validate_hqq_nbits(nbits: int) -> None:
    if nbits not in HQQ_ENABLED_NBITS:
        enabled = ", ".join(str(v) for v in HQQ_ENABLED_NBITS)
        raise ValueError(f"Unsupported HQQ nbits={nbits}. Enabled HQQ backends: {enabled}")


def _emit_real_model_timing(payload: dict) -> None:
    if os.environ.get("KRASIS_HQQ_REAL_MODEL_TIMING") == "1":
        print(json.dumps({"hqq_real_model_timing": payload}, sort_keys=True), flush=True)


def hqq_attention_cache_dir(model_path: str) -> str:
    """Return the native HQQ attention artifact directory under the model cache."""
    return os.path.join(cache_dir_for_model(model_path), HQQ_ATTENTION_CACHE_DIRNAME)


def hqq_attention_manifest_path(model_path: str) -> str:
    return os.path.join(hqq_attention_cache_dir(model_path), HQQ_ATTENTION_MANIFEST)


def hqq_attention_pending_manifest_path(model_path: str) -> str:
    return os.path.join(
        hqq_attention_cache_dir(model_path),
        HQQ_ATTENTION_PENDING_MANIFEST,
    )


def hqq_attention_cache_key(layer_idx: int, tensor_name: str, nbits: int = 4) -> str:
    """Stable filename stem for one HQQ attention tensor artifact."""
    return f"layer_{layer_idx:03d}_{tensor_name}_hqq{nbits}"


def hqq_attention_tensor_path(model_path: str, layer_idx: int, tensor_name: str, nbits: int = 4) -> str:
    return os.path.join(
        hqq_attention_cache_dir(model_path),
        f"{hqq_attention_cache_key(layer_idx, tensor_name, nbits)}.safetensors",
    )


def _dtype_nbytes(dtype: str) -> int:
    table = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E5M2": 1,
        "F8_E4M3": 1,
        "I16": 2,
        "U16": 2,
        "BF16": 2,
        "F16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    if dtype not in table:
        raise ValueError(f"Unsupported safetensors dtype {dtype!r}")
    return table[dtype]


def _artifact_tensor_bytes(path: str) -> int:
    total = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            sl = handle.get_slice(key)
            shape = sl.get_shape()
            dtype = str(sl.get_dtype())
            try:
                total += math.prod(shape) * _dtype_nbytes(dtype)
            except ValueError:
                tensor = handle.get_tensor(key)
                total += tensor.numel() * tensor.element_size()
    return total


def load_hqq_attention_manifest(model_path: str) -> Optional[dict]:
    path = hqq_attention_manifest_path(model_path)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_hqq_attention_manifest_to_path(path: str, manifest: dict) -> None:
    cache_dir = os.path.dirname(path)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def save_hqq_attention_manifest(model_path: str, manifest: dict) -> None:
    _save_hqq_attention_manifest_to_path(hqq_attention_manifest_path(model_path), manifest)


def load_hqq_attention_pending_manifest(model_path: str) -> Optional[dict]:
    path = hqq_attention_pending_manifest_path(model_path)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_hqq_attention_pending_manifest(model_path: str, manifest: dict) -> None:
    _save_hqq_attention_manifest_to_path(
        hqq_attention_pending_manifest_path(model_path),
        manifest,
    )


def delete_hqq_attention_pending_manifest(model_path: str) -> None:
    path = hqq_attention_pending_manifest_path(model_path)
    if os.path.isfile(path):
        os.remove(path)


def init_hqq_attention_manifest(model_path: str, num_hidden_layers: int, nbits: int = 4) -> dict:
    validate_hqq_nbits(nbits)
    return {
        "format_version": HQQ_ATTENTION_CACHE_VERSION,
        "backend": hqq_backend_name(nbits),
        "nbits": nbits,
        "group_size": HQQ_DEFAULT_GROUP_SIZE,
        "axis": HQQ_DEFAULT_AXIS,
        "layout": HQQ_LAYOUT,
        "num_hidden_layers": num_hidden_layers,
        "complete": False,
        "tensors": [],
        "totals": {
            "tensor_bytes": 0,
            "num_tensors": 0,
        },
    }


def _pack_uint4(q: torch.Tensor) -> torch.Tensor:
    if q.dtype != torch.uint8:
        q = q.to(torch.uint8)
    if q.shape[-1] % 2 != 0:
        q = torch.nn.functional.pad(q, (0, 1))
    lo = q[..., 0::2]
    hi = q[..., 1::2] << 4
    return (lo | hi).contiguous()


def _unpack_uint4(packed: torch.Tensor, cols: int) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    q = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    return q[:, :cols].contiguous()


def _hqq_num_groups(cols: int, group_size: int) -> int:
    return (cols + group_size - 1) // group_size


def _hqq_padded_cols(cols: int, group_size: int) -> int:
    return _hqq_num_groups(cols, group_size) * group_size


def _hqq_packed_cols(cols: int, group_size: int) -> int:
    return (_hqq_padded_cols(cols, group_size) + 1) // 2


def _hqq_tensor_shape_list(tensor: torch.Tensor) -> List[int]:
    return [int(v) for v in tensor.shape]


def validate_hqq_attention_tensors(
    tensors: Dict[str, torch.Tensor],
    *,
    expected_nbits: Optional[int] = None,
    expected_axis: int = HQQ_DEFAULT_AXIS,
    expected_layout: str = HQQ_LAYOUT,
) -> dict:
    required = ("packed", "scales", "zeros", "orig_shape", "group_size", "axis", "nbits")
    missing = [name for name in required if name not in tensors]
    if missing:
        raise RuntimeError(f"HQQ artifact missing required tensors: {', '.join(missing)}")

    packed = tensors["packed"]
    scales = tensors["scales"]
    zeros = tensors["zeros"]
    orig_shape = tensors["orig_shape"]
    group_size_t = tensors["group_size"]
    axis_t = tensors["axis"]
    nbits_t = tensors["nbits"]

    if packed.dtype != torch.uint8:
        raise RuntimeError(f"HQQ packed tensor must be uint8, found {packed.dtype}")
    if scales.dtype != torch.float32 or zeros.dtype != torch.float32:
        raise RuntimeError(
            f"HQQ scales/zeros must be float32, found {scales.dtype}/{zeros.dtype}"
        )
    for name, tensor in (
        ("orig_shape", orig_shape),
        ("group_size", group_size_t),
        ("axis", axis_t),
        ("nbits", nbits_t),
    ):
        if tensor.dtype != torch.int32:
            raise RuntimeError(f"HQQ {name} must be int32, found {tensor.dtype}")

    if orig_shape.numel() != 2:
        raise RuntimeError(f"HQQ orig_shape must have 2 elements, found {orig_shape.numel()}")
    if group_size_t.numel() != 1 or axis_t.numel() != 1 or nbits_t.numel() != 1:
        raise RuntimeError("HQQ group_size/axis/nbits tensors must be scalar length-1 tensors")

    rows, cols = (int(v) for v in orig_shape.tolist())
    group_size = int(group_size_t[0].item())
    axis = int(axis_t[0].item())
    nbits = int(nbits_t[0].item())
    if rows <= 0 or cols <= 0:
        raise RuntimeError(f"HQQ orig_shape must be positive, found {(rows, cols)}")
    if group_size <= 0:
        raise RuntimeError(f"HQQ group_size must be positive, found {group_size}")
    if axis != expected_axis:
        raise RuntimeError(f"HQQ axis mismatch: found {axis}, expected {expected_axis}")
    if expected_nbits is not None and nbits != expected_nbits:
        raise RuntimeError(f"HQQ nbits mismatch: found {nbits}, expected {expected_nbits}")
    if expected_layout != HQQ_LAYOUT:
        raise RuntimeError(
            f"HQQ validation only supports runtime layout {HQQ_LAYOUT}, got expected {expected_layout}"
        )

    groups = _hqq_num_groups(cols, group_size)
    padded_cols = _hqq_padded_cols(cols, group_size)
    packed_cols = _hqq_packed_cols(cols, group_size)
    if tuple(scales.shape) != (rows, groups):
        raise RuntimeError(
            f"HQQ scales shape mismatch: found {tuple(scales.shape)}, expected {(rows, groups)}"
        )
    if tuple(zeros.shape) != (rows, groups):
        raise RuntimeError(
            f"HQQ zeros shape mismatch: found {tuple(zeros.shape)}, expected {(rows, groups)}"
        )
    if tuple(packed.shape) != (rows, packed_cols):
        raise RuntimeError(
            f"HQQ packed shape mismatch: found {tuple(packed.shape)}, expected {(rows, packed_cols)}"
        )

    return {
        "rows": rows,
        "cols": cols,
        "group_size": group_size,
        "groups": groups,
        "padded_cols": padded_cols,
        "packed_cols": packed_cols,
        "axis": axis,
        "nbits": nbits,
        "layout": expected_layout,
        "packed_shape": _hqq_tensor_shape_list(packed),
        "scales_shape": _hqq_tensor_shape_list(scales),
        "zeros_shape": _hqq_tensor_shape_list(zeros),
    }


def _quantize_hqq4_group_current(chunk: torch.Tensor, qmax: float) -> tuple[torch.Tensor, float, float]:
    minv = chunk.amin(dim=1, keepdim=True)
    maxv = chunk.amax(dim=1, keepdim=True)
    scale = ((maxv - minv) / qmax).clamp(min=1e-8)
    zero = (-minv / scale).clamp_(0.0, qmax)
    q = ((chunk / scale) + zero).round().clamp_(0.0, qmax)
    return q.to(torch.uint8), scale.squeeze(1), zero.squeeze(1)


def quantize_hqq4_group_current_rust(chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rust shadow init path for one HQQ4 group chunk."""
    if chunk.ndim != 2:
        raise ValueError(f"HQQ group init expects 2D chunk, got shape {tuple(chunk.shape)}")

    rows, cols = chunk.shape
    w = chunk.detach().to(device="cpu", dtype=torch.float32).contiguous()
    q = torch.empty((rows, cols), dtype=torch.uint8)
    scale = torch.empty((rows,), dtype=torch.float32)
    zero = torch.empty((rows,), dtype=torch.float32)
    hqq4_init_group_ptr(
        int(w.data_ptr()),
        int(rows),
        int(cols),
        int(q.data_ptr()),
        int(scale.data_ptr()),
        int(zero.data_ptr()),
    )
    return q, scale, zero


def solve_hqq4_fixed_zero_rust(
    chunk: torch.Tensor,
    zero: torch.Tensor,
    scale_seed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rust shadow fixed-zero solve path for one HQQ4 group chunk."""
    if chunk.ndim != 2:
        raise ValueError(f"HQQ group solve expects 2D chunk, got shape {tuple(chunk.shape)}")
    rows, cols = chunk.shape
    w = chunk.detach().to(device="cpu", dtype=torch.float32).contiguous()
    zero_cpu = zero.detach().to(device="cpu", dtype=torch.float32).contiguous().view(rows)
    scale_seed_cpu = scale_seed.detach().to(device="cpu", dtype=torch.float32).contiguous().view(rows)
    q = torch.empty((rows, cols), dtype=torch.uint8)
    scale = torch.empty((rows,), dtype=torch.float32)
    hqq4_solve_group_ptr(
        int(w.data_ptr()),
        int(rows),
        int(cols),
        int(zero_cpu.data_ptr()),
        int(scale_seed_cpu.data_ptr()),
        int(q.data_ptr()),
        int(scale.data_ptr()),
    )
    return q, scale, zero_cpu


def compute_hqq4_rmse_rust(
    chunk: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    """Rust shadow RMSE path for one HQQ4 group chunk."""
    if chunk.ndim != 2:
        raise ValueError(f"HQQ group RMSE expects 2D chunk, got shape {tuple(chunk.shape)}")
    rows, cols = chunk.shape
    w = chunk.detach().to(device="cpu", dtype=torch.float32).contiguous()
    q_cpu = q.detach().to(device="cpu", dtype=torch.uint8).contiguous().view(rows, cols)
    scale_cpu = scale.detach().to(device="cpu", dtype=torch.float32).contiguous().view(rows)
    zero_cpu = zero.detach().to(device="cpu", dtype=torch.float32).contiguous().view(rows)
    rmse = torch.empty((rows,), dtype=torch.float32)
    hqq4_rmse_group_ptr(
        int(w.data_ptr()),
        int(rows),
        int(cols),
        int(q_cpu.data_ptr()),
        int(scale_cpu.data_ptr()),
        int(zero_cpu.data_ptr()),
        int(rmse.data_ptr()),
    )
    return rmse


def _quantize_hqq4_group_symmetric(chunk: torch.Tensor, qmax: float) -> tuple[torch.Tensor, float, float]:
    amax = chunk.abs().amax(dim=1, keepdim=True)
    scale = ((2.0 * amax) / qmax).clamp(min=1e-8)
    zero = torch.full_like(scale, qmax / 2.0)
    q = ((chunk / scale) + zero).round().clamp_(0.0, qmax)
    return q.to(torch.uint8), scale.squeeze(1), zero.squeeze(1)


def _solve_hqq4_fixed_zero(
    chunk: torch.Tensor,
    qmax: float,
    zero: torch.Tensor,
    scale_seed: torch.Tensor,
    iters: int = 6,
    timing_stats: Optional[dict] = None,
    timing_context: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    call_started = time.perf_counter() if timing_stats is not None else 0.0
    setup_started = time.perf_counter() if timing_stats is not None else 0.0
    scale = scale_seed.clone()
    if timing_stats is not None:
        _accumulate_hqq4_core_timing(timing_stats, "solve_alloc_setup_s", time.perf_counter() - setup_started)
        solve_alloc_elapsed = time.perf_counter() - setup_started
    else:
        solve_alloc_elapsed = 0.0

    iter_quant_elapsed = 0.0
    iter_reduce_elapsed = 0.0
    iter_update_elapsed = 0.0
    for _ in range(iters):
        quant_started = time.perf_counter() if timing_stats is not None else 0.0
        q = ((chunk / scale.unsqueeze(1)) + zero.unsqueeze(1)).round().clamp_(0.0, qmax)
        quant_elapsed = time.perf_counter() - quant_started if timing_stats is not None else 0.0
        reduction_started = time.perf_counter() if timing_stats is not None else 0.0
        centered = q - zero.unsqueeze(1)
        denom = torch.sum(centered * centered, dim=1)
        numer = torch.sum(chunk * centered, dim=1)
        reduction_elapsed = time.perf_counter() - reduction_started if timing_stats is not None else 0.0
        update_started = time.perf_counter() if timing_stats is not None else 0.0
        scale = torch.where(denom > 1e-12, numer / denom, scale).clamp_min_(1e-8)
        if timing_stats is not None:
            iter_quant_elapsed += quant_elapsed
            iter_reduce_elapsed += reduction_elapsed
            update_elapsed = time.perf_counter() - update_started
            iter_update_elapsed += update_elapsed
            _accumulate_hqq4_core_timing(timing_stats, "solve_iter_quantize_clamp_s", quant_elapsed)
            _accumulate_hqq4_core_timing(timing_stats, "solve_iter_reduction_s", reduction_elapsed)
            _accumulate_hqq4_core_timing(timing_stats, "solve_iter_update_s", update_elapsed)
    final_quant_started = time.perf_counter() if timing_stats is not None else 0.0
    q = ((chunk / scale.unsqueeze(1)) + zero.unsqueeze(1)).round().clamp_(0.0, qmax)
    if timing_stats is not None:
        final_quant_elapsed = time.perf_counter() - final_quant_started
        _accumulate_hqq4_core_timing(timing_stats, "solve_final_quantize_s", final_quant_elapsed)
        if timing_context is not None:
            rows, cols = chunk.shape
            zero_scalar = float(zero[0].item()) if zero.numel() > 0 else 0.0
            _emit_real_model_timing(
                {
                    **timing_context,
                    "phase": "artifact_pack_stage",
                    "stage": "solve_fixed_zero_s_call",
                    "rows": int(rows),
                    "cols": int(cols),
                    "iters": int(iters),
                    "elapsed_s": time.perf_counter() - call_started,
                    "solve_alloc_setup_s": solve_alloc_elapsed,
                    "solve_iter_quantize_clamp_s": iter_quant_elapsed,
                    "solve_iter_reduction_s": iter_reduce_elapsed,
                    "solve_iter_update_s": iter_update_elapsed,
                    "solve_final_quantize_s": final_quant_elapsed,
                    "zero_value": zero_scalar,
                }
            )
    return q.to(torch.uint8), scale, zero


def _accumulate_hqq4_core_timing(timing_stats: Optional[dict], key: str, elapsed_s: float) -> None:
    if timing_stats is None:
        return
    timing_stats[key] = timing_stats.get(key, 0.0) + elapsed_s


def _update_best_hqq4_group_fit(
    chunk: torch.Tensor,
    qmax: float,
    zero_grid: torch.Tensor,
    scale_seed: torch.Tensor,
    best_q: torch.Tensor,
    best_scale: torch.Tensor,
    best_zero: torch.Tensor,
    best_rmse: torch.Tensor,
    timing_stats: Optional[dict] = None,
    timing_context_base: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zero_grid_steps = int(zero_grid.numel())
    zero_grid_min = float(zero_grid[0].item()) if zero_grid_steps > 0 else 0.0
    zero_grid_max = float(zero_grid[-1].item()) if zero_grid_steps > 0 else 0.0
    rows, cols = chunk.shape
    for candidate_idx, zero_value in enumerate(zero_grid):
        loop_started = time.perf_counter() if timing_stats is not None else 0.0
        zero = torch.full(
            (chunk.shape[0],),
            float(zero_value),
            dtype=torch.float32,
        )
        solve_started = time.perf_counter() if timing_stats is not None else 0.0
        q, scale, zero = _solve_hqq4_fixed_zero(
            chunk,
            qmax,
            zero,
            scale_seed,
            timing_stats=timing_stats,
            timing_context={
                **(timing_context_base or {}),
                "rows": int(rows),
                "cols": int(cols),
                "candidate_idx": int(candidate_idx),
                "candidate_total": int(zero_grid_steps),
                "zero_grid_min": zero_grid_min,
                "zero_grid_max": zero_grid_max,
            },
        )
        solve_elapsed = time.perf_counter() - solve_started if timing_stats is not None else 0.0
        reconstruct_started = time.perf_counter() if timing_stats is not None else 0.0
        deq = (q.to(torch.float32) - zero.unsqueeze(1)) * scale.unsqueeze(1)
        rmse = torch.mean((deq - chunk) ** 2, dim=1)
        improved = rmse < best_rmse
        reconstruct_elapsed = time.perf_counter() - reconstruct_started if timing_stats is not None else 0.0
        update_started = time.perf_counter() if timing_stats is not None else 0.0
        if torch.any(improved):
            best_rmse = torch.where(improved, rmse, best_rmse)
            best_q = torch.where(improved.unsqueeze(1), q, best_q)
            best_scale = torch.where(improved, scale, best_scale)
            best_zero = torch.where(improved, zero, best_zero)
        if timing_stats is not None:
            update_elapsed = time.perf_counter() - update_started
            loop_elapsed = time.perf_counter() - loop_started
            _accumulate_hqq4_core_timing(
                timing_stats,
                "candidate_loop_overhead_s",
                loop_elapsed - solve_elapsed - reconstruct_elapsed - update_elapsed,
            )
            _accumulate_hqq4_core_timing(timing_stats, "solve_fixed_zero_s", solve_elapsed)
            _accumulate_hqq4_core_timing(timing_stats, "score_reconstruct_s", reconstruct_elapsed)
            _accumulate_hqq4_core_timing(timing_stats, "best_fit_update_s", update_elapsed)
            timing_stats["candidate_count"] = timing_stats.get("candidate_count", 0) + 1
    return best_q, best_scale, best_zero, best_rmse


def _run_hqq4_group_search_driver(
    chunk: torch.Tensor,
    qmax: float,
    search_specs: list[tuple[torch.Tensor, torch.Tensor]],
    best_q: torch.Tensor,
    best_scale: torch.Tensor,
    best_zero: torch.Tensor,
    best_rmse: torch.Tensor,
    timing_stats: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    search_spec_total = len(search_specs)
    for search_spec_idx, (zero_grid, scale_seed) in enumerate(search_specs):
        best_q, best_scale, best_zero, best_rmse = _update_best_hqq4_group_fit(
            chunk,
            qmax,
            zero_grid,
            scale_seed,
            best_q,
            best_scale,
            best_zero,
            best_rmse,
            timing_stats=timing_stats,
            timing_context_base={
                "search_spec_idx": int(search_spec_idx),
                "search_spec_total": int(search_spec_total),
            },
        )
    return best_q, best_scale, best_zero, best_rmse


def _refine_hqq4_group_fit(
    chunk: torch.Tensor,
    qmax: float,
    timing_stats: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_init, scale_init, zero_init = _quantize_hqq4_group_current(chunk, qmax)
    centered_init = q_init.to(torch.float32) - zero_init.unsqueeze(1)
    deq_init = centered_init * scale_init.unsqueeze(1)
    best_rmse = torch.mean((deq_init - chunk) ** 2, dim=1)
    best_q = q_init
    best_scale = scale_init
    best_zero = zero_init

    setup_started = time.perf_counter() if timing_stats is not None else 0.0
    range_scale = ((chunk.amax(dim=1) - chunk.amin(dim=1)) / qmax).clamp(min=1e-8)
    abs_scale = ((2.0 * chunk.abs().amax(dim=1)) / qmax).clamp(min=1e-8)

    global_zero_grid = torch.linspace(0.0, qmax, steps=129, dtype=torch.float32)
    if timing_stats is not None:
        _accumulate_hqq4_core_timing(timing_stats, "zero_grid_setup_s", time.perf_counter() - setup_started)
    best_q, best_scale, best_zero, best_rmse = _run_hqq4_group_search_driver(
        chunk,
        qmax,
        [
            (global_zero_grid, range_scale),
            (global_zero_grid, abs_scale),
        ],
        best_q,
        best_scale,
        best_zero,
        best_rmse,
        timing_stats=timing_stats,
    )

    local_setup_started = time.perf_counter() if timing_stats is not None else 0.0
    local_zero_min = max(0.0, float(best_zero.min().item()) - 0.5)
    local_zero_max = min(qmax, float(best_zero.max().item()) + 0.5)
    local_zero_grid = torch.linspace(local_zero_min, local_zero_max, steps=65, dtype=torch.float32)
    if timing_stats is not None:
        _accumulate_hqq4_core_timing(timing_stats, "zero_grid_setup_s", time.perf_counter() - local_setup_started)
    local_scale_seed = best_scale.clone()
    best_q, best_scale, best_zero, best_rmse = _run_hqq4_group_search_driver(
        chunk,
        qmax,
        [(local_zero_grid, local_scale_seed)],
        best_q,
        best_scale,
        best_zero,
        best_rmse,
        timing_stats=timing_stats,
    )

    return best_q, best_scale, best_zero


def quantize_hqq4_tensor_rust(
    weight: torch.Tensor,
    group_size: int = HQQ_DEFAULT_GROUP_SIZE,
    *,
    collect_stats: bool = False,
    worst_groups_limit: int = 8,
) -> Dict[str, torch.Tensor]:
    """Experimental Rust shadow implementation for HQQ4 tensor quantization."""
    if weight.ndim != 2:
        raise ValueError(f"HQQ attention expects 2D weights, got shape {tuple(weight.shape)}")
    if group_size <= 0:
        raise ValueError(f"Invalid HQQ group_size {group_size}")

    w = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    rows, cols = w.shape
    groups = _hqq_num_groups(cols, group_size)
    packed_cols = _hqq_packed_cols(cols, group_size)
    packed = torch.empty((rows, packed_cols), dtype=torch.uint8)
    scales = torch.empty((rows, groups), dtype=torch.float32)
    zeros = torch.empty((rows, groups), dtype=torch.float32)
    hqq4_quantize_tensor_ptr(
        int(w.data_ptr()),
        int(rows),
        int(cols),
        int(group_size),
        int(packed.data_ptr()),
        int(scales.data_ptr()),
        int(zeros.data_ptr()),
    )

    result = {
        "packed": packed,
        "scales": scales,
        "zeros": zeros,
        "orig_shape": torch.tensor([rows, cols], dtype=torch.int32),
        "group_size": torch.tensor([group_size], dtype=torch.int32),
        "axis": torch.tensor([HQQ_DEFAULT_AXIS], dtype=torch.int32),
        "nbits": torch.tensor([4], dtype=torch.int32),
    }
    if collect_stats:
        quant = _unpack_uint4(packed, _hqq_padded_cols(cols, group_size))[:, :cols]
        stats = {
            "numel": 0,
            "sum_abs": 0.0,
            "sum_sq": 0.0,
            "max_abs": 0.0,
            "worst_groups": [],
        }
        for g in range(groups):
            start = g * group_size
            end = min(start + group_size, cols)
            q = quant[:, start:end]
            scale = scales[:, g]
            zero = zeros[:, g]
            chunk = w[:, start:end]
            deq = (q.to(torch.float32) - zero.unsqueeze(1)) * scale.unsqueeze(1)
            diff = deq - chunk
            abs_diff = diff.abs()
            stats["numel"] += diff.numel()
            stats["sum_abs"] += float(abs_diff.sum().item())
            stats["sum_sq"] += float((diff * diff).sum().item())
            stats["max_abs"] = max(stats["max_abs"], float(abs_diff.max().item()))
            group_mean_abs = abs_diff.mean(dim=1)
            group_rmse = torch.sqrt(torch.mean(diff * diff, dim=1))
            if worst_groups_limit > 0:
                for row_idx in range(rows):
                    entry = {
                        "row": int(row_idx),
                        "group": int(g),
                        "start_col": int(start),
                        "end_col": int(end),
                        "mean_abs": float(group_mean_abs[row_idx].item()),
                        "rmse": float(group_rmse[row_idx].item()),
                        "max_abs": float(abs_diff[row_idx].max().item()),
                        "scale": float(scale[row_idx].item()),
                        "zero": float(zero[row_idx].item()),
                    }
                    if len(stats["worst_groups"]) < worst_groups_limit:
                        stats["worst_groups"].append(entry)
                    else:
                        min_idx = min(
                            range(len(stats["worst_groups"])),
                            key=lambda idx: stats["worst_groups"][idx]["mean_abs"],
                        )
                        if entry["mean_abs"] > stats["worst_groups"][min_idx]["mean_abs"]:
                            stats["worst_groups"][min_idx] = entry
        stats["worst_groups"].sort(key=lambda item: item["mean_abs"], reverse=True)
        denom = max(1, stats["numel"])
        result["stats"] = {
            "numel": int(stats["numel"]),
            "mean_abs": stats["sum_abs"] / denom,
            "rmse": math.sqrt(stats["sum_sq"] / denom),
            "max_abs": stats["max_abs"],
            "worst_groups": stats["worst_groups"],
        }
    return result


def quantize_hqq4_tensor(
    weight: torch.Tensor,
    group_size: int = HQQ_DEFAULT_GROUP_SIZE,
    *,
    collect_stats: bool = False,
    worst_groups_limit: int = 8,
    timing_context: Optional[dict] = None,
) -> Dict[str, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"HQQ attention expects 2D weights, got shape {tuple(weight.shape)}")
    if group_size <= 0:
        raise ValueError(f"Invalid HQQ group_size {group_size}")

    timing_enabled = os.environ.get("KRASIS_HQQ_REAL_MODEL_TIMING") == "1"
    timing_prefix = dict(timing_context or {})

    normalize_started = time.perf_counter() if timing_enabled else 0.0
    detached = weight.detach()
    normalized = detached.to(device="cpu", dtype=torch.float32)
    if timing_enabled:
        _emit_real_model_timing(
            {
                **timing_prefix,
                "phase": "artifact_pack_stage",
                "stage": "dtype_device_normalize",
                "elapsed_s": time.perf_counter() - normalize_started,
                "input_dtype": str(weight.dtype).replace("torch.", ""),
                "input_device": str(weight.device),
                "output_dtype": str(normalized.dtype).replace("torch.", ""),
                "output_device": str(normalized.device),
            }
        )

    layout_started = time.perf_counter() if timing_enabled else 0.0
    w = normalized.contiguous()
    rows, cols = w.shape
    groups = _hqq_num_groups(cols, group_size)
    qmax = 15.0
    if timing_enabled:
        _emit_real_model_timing(
            {
                **timing_prefix,
                "phase": "artifact_pack_stage",
                "stage": "reshape_contiguous_copy",
                "elapsed_s": time.perf_counter() - layout_started,
                "rows": int(rows),
                "cols": int(cols),
                "groups": int(groups),
            }
        )

    quant_chunks: List[torch.Tensor] = []
    scales = torch.empty((rows, groups), dtype=torch.float32)
    zeros = torch.empty((rows, groups), dtype=torch.float32)
    stats = {
        "numel": 0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "max_abs": 0.0,
        "worst_groups": [],
    } if collect_stats else None
    quantizer_core_timing = {
        "zero_grid_setup_s": 0.0,
        "candidate_loop_overhead_s": 0.0,
        "solve_fixed_zero_s": 0.0,
        "solve_alloc_setup_s": 0.0,
        "solve_iter_quantize_clamp_s": 0.0,
        "solve_iter_reduction_s": 0.0,
        "solve_iter_update_s": 0.0,
        "solve_final_quantize_s": 0.0,
        "score_reconstruct_s": 0.0,
        "best_fit_update_s": 0.0,
        "candidate_count": 0,
    } if timing_enabled else None

    quant_started = time.perf_counter() if timing_enabled else 0.0
    for g in range(groups):
        start = g * group_size
        end = min(start + group_size, cols)
        chunk = w[:, start:end]
        group_timing_before = None
        if quantizer_core_timing is not None:
            group_timing_before = dict(quantizer_core_timing)
        # Fit under the exact runtime contract: q in [0, 15], float zero-point,
        # dequant = (q - zero) * scale, same row-major axis-1 grouped packing.
        q, scale, zero = _refine_hqq4_group_fit(chunk, qmax, timing_stats=quantizer_core_timing)
        if timing_enabled:
            group_zero_grid_setup = quantizer_core_timing["zero_grid_setup_s"] - group_timing_before["zero_grid_setup_s"]
            group_candidate_loop = quantizer_core_timing["candidate_loop_overhead_s"] - group_timing_before["candidate_loop_overhead_s"]
            group_solve = quantizer_core_timing["solve_fixed_zero_s"] - group_timing_before["solve_fixed_zero_s"]
            group_solve_alloc = quantizer_core_timing["solve_alloc_setup_s"] - group_timing_before["solve_alloc_setup_s"]
            group_solve_iter_quant = quantizer_core_timing["solve_iter_quantize_clamp_s"] - group_timing_before["solve_iter_quantize_clamp_s"]
            group_solve_iter_reduce = quantizer_core_timing["solve_iter_reduction_s"] - group_timing_before["solve_iter_reduction_s"]
            group_solve_iter_update = quantizer_core_timing["solve_iter_update_s"] - group_timing_before["solve_iter_update_s"]
            group_solve_final_quant = quantizer_core_timing["solve_final_quantize_s"] - group_timing_before["solve_final_quantize_s"]
            group_score_reconstruct = quantizer_core_timing["score_reconstruct_s"] - group_timing_before["score_reconstruct_s"]
            group_best_fit = quantizer_core_timing["best_fit_update_s"] - group_timing_before["best_fit_update_s"]
            group_candidates = quantizer_core_timing["candidate_count"] - group_timing_before["candidate_count"]
            _emit_real_model_timing(
                {
                    **timing_prefix,
                    "phase": "artifact_pack_stage",
                    "stage": "quantize_pack_core_group",
                    "group_idx": int(g),
                    "start_col": int(start),
                    "end_col": int(end),
                    "elapsed_s": (
                        group_zero_grid_setup
                        + group_candidate_loop
                        + group_solve
                        + group_score_reconstruct
                        + group_best_fit
                    ),
                    "solve_alloc_setup_s": group_solve_alloc,
                    "solve_iter_quantize_clamp_s": group_solve_iter_quant,
                    "solve_iter_reduction_s": group_solve_iter_reduce,
                    "solve_iter_update_s": group_solve_iter_update,
                    "solve_final_quantize_s": group_solve_final_quant,
                    "zero_grid_setup_s": group_zero_grid_setup,
                    "candidate_loop_overhead_s": group_candidate_loop,
                    "solve_fixed_zero_s": group_solve,
                    "score_reconstruct_s": group_score_reconstruct,
                    "best_fit_update_s": group_best_fit,
                    "candidate_count": int(group_candidates),
                }
            )
        if collect_stats:
            deq = (q[:, : end - start].to(torch.float32) - zero.unsqueeze(1)) * scale.unsqueeze(1)
            diff = deq - chunk
            abs_diff = diff.abs()
            stats["numel"] += diff.numel()
            stats["sum_abs"] += float(abs_diff.sum().item())
            stats["sum_sq"] += float((diff * diff).sum().item())
            stats["max_abs"] = max(stats["max_abs"], float(abs_diff.max().item()))
            group_mean_abs = abs_diff.mean(dim=1)
            group_rmse = torch.sqrt(torch.mean(diff * diff, dim=1))
            if worst_groups_limit > 0:
                for row_idx in range(rows):
                    entry = {
                        "row": int(row_idx),
                        "group": int(g),
                        "start_col": int(start),
                        "end_col": int(end),
                        "mean_abs": float(group_mean_abs[row_idx].item()),
                        "rmse": float(group_rmse[row_idx].item()),
                        "max_abs": float(abs_diff[row_idx].max().item()),
                        "scale": float(scale[row_idx].item()),
                        "zero": float(zero[row_idx].item()),
                    }
                    if len(stats["worst_groups"]) < worst_groups_limit:
                        stats["worst_groups"].append(entry)
                    else:
                        min_idx = min(
                            range(len(stats["worst_groups"])),
                            key=lambda idx: stats["worst_groups"][idx]["mean_abs"],
                        )
                        if entry["mean_abs"] > stats["worst_groups"][min_idx]["mean_abs"]:
                            stats["worst_groups"][min_idx] = entry
        if end - start < group_size:
            q = torch.nn.functional.pad(q, (0, group_size - (end - start)))
        quant_chunks.append(q)
        scales[:, g] = scale
        zeros[:, g] = zero

    quant = torch.cat(quant_chunks, dim=1)
    packed = _pack_uint4(quant)
    if timing_enabled:
        _emit_real_model_timing(
            {
                **timing_prefix,
                "phase": "artifact_pack_stage",
                "stage": "quantize_pack",
                "elapsed_s": time.perf_counter() - quant_started,
                "packed_rows": int(packed.shape[0]),
                "packed_cols": int(packed.shape[1]),
            }
        )
        _emit_real_model_timing(
            {
                **timing_prefix,
                "phase": "artifact_pack_stage",
                "stage": "quantize_pack_core",
                "elapsed_s": quantizer_core_timing["zero_grid_setup_s"]
                + quantizer_core_timing["candidate_loop_overhead_s"]
                + quantizer_core_timing["solve_fixed_zero_s"]
                + quantizer_core_timing["score_reconstruct_s"]
                + quantizer_core_timing["best_fit_update_s"],
                "zero_grid_setup_s": quantizer_core_timing["zero_grid_setup_s"],
                "candidate_loop_overhead_s": quantizer_core_timing["candidate_loop_overhead_s"],
                "solve_fixed_zero_s": quantizer_core_timing["solve_fixed_zero_s"],
                "solve_alloc_setup_s": quantizer_core_timing["solve_alloc_setup_s"],
                "solve_iter_quantize_clamp_s": quantizer_core_timing["solve_iter_quantize_clamp_s"],
                "solve_iter_reduction_s": quantizer_core_timing["solve_iter_reduction_s"],
                "solve_iter_update_s": quantizer_core_timing["solve_iter_update_s"],
                "solve_final_quantize_s": quantizer_core_timing["solve_final_quantize_s"],
                "score_reconstruct_s": quantizer_core_timing["score_reconstruct_s"],
                "best_fit_update_s": quantizer_core_timing["best_fit_update_s"],
                "candidate_count": int(quantizer_core_timing["candidate_count"]),
            }
        )
    result = {
        "packed": packed,
        "scales": scales,
        "zeros": zeros,
        "orig_shape": torch.tensor([rows, cols], dtype=torch.int32),
        "group_size": torch.tensor([group_size], dtype=torch.int32),
        "axis": torch.tensor([HQQ_DEFAULT_AXIS], dtype=torch.int32),
        "nbits": torch.tensor([4], dtype=torch.int32),
    }
    if collect_stats:
        stats["worst_groups"].sort(key=lambda item: item["mean_abs"], reverse=True)
        denom = max(1, stats["numel"])
        result["stats"] = {
            "numel": int(stats["numel"]),
            "mean_abs": stats["sum_abs"] / denom,
            "rmse": math.sqrt(stats["sum_sq"] / denom),
            "max_abs": stats["max_abs"],
            "worst_groups": stats["worst_groups"],
        }
    return result


def write_hqq_attention_artifact(
    model_path: str,
    layer_idx: int,
    layer_type: str,
    tensor_name: str,
    weight: torch.Tensor,
    nbits: int = 4,
) -> dict:
    validate_hqq_nbits(nbits)

    path = hqq_attention_tensor_path(model_path, layer_idx, tensor_name, nbits=nbits)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    timing_context = {
        "layer_idx": int(layer_idx),
        "layer_type": layer_type,
        "tensor_name": tensor_name,
    }
    tensors = quantize_hqq4_tensor(
        weight,
        group_size=HQQ_DEFAULT_GROUP_SIZE,
        collect_stats=True,
        timing_context=timing_context,
    )
    metadata_started = time.perf_counter() if os.environ.get("KRASIS_HQQ_REAL_MODEL_TIMING") == "1" else 0.0
    payload_tensors = {k: v for k, v in tensors.items() if isinstance(v, torch.Tensor)}
    structure = validate_hqq_attention_tensors(
        payload_tensors,
        expected_nbits=nbits,
        expected_layout=HQQ_LAYOUT,
    )
    metadata = {
        "backend": hqq_backend_name(nbits),
        "format_version": str(HQQ_ATTENTION_CACHE_VERSION),
        "layer_idx": str(layer_idx),
        "layer_type": layer_type,
        "tensor_name": tensor_name,
        "layout": HQQ_LAYOUT,
        "dtype": str(weight.dtype).replace("torch.", ""),
    }
    if metadata_started:
        _emit_real_model_timing(
            {
                **timing_context,
                "phase": "artifact_pack_stage",
                "stage": "metadata_prepare",
                "elapsed_s": time.perf_counter() - metadata_started,
            }
        )
    file_started = time.perf_counter() if os.environ.get("KRASIS_HQQ_REAL_MODEL_TIMING") == "1" else 0.0
    save_file(payload_tensors, path, metadata=metadata)
    tensor_bytes = _artifact_tensor_bytes(path)
    if file_started:
        _emit_real_model_timing(
            {
                **timing_context,
                "phase": "artifact_pack_stage",
                "stage": "file_write",
                "elapsed_s": time.perf_counter() - file_started,
                "tensor_bytes": int(tensor_bytes),
            }
        )
    orig_shape = payload_tensors["orig_shape"].tolist()
    stats = tensors["stats"]
    return {
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "tensor_name": tensor_name,
        "file": os.path.basename(path),
        "path": path,
        "nbits": nbits,
        "group_size": int(tensors["group_size"][0].item()),
        "axis": int(tensors["axis"][0].item()),
        "layout": HQQ_LAYOUT,
        "original_shape": orig_shape,
        "original_dtype": metadata["dtype"],
        "tensor_bytes": tensor_bytes,
        "structure": structure,
        "quality": stats,
    }


def load_hqq_attention_artifact(
    model_path: str,
    entry: dict,
    expected_nbits: int,
    device: str = "cpu",
) -> dict:
    validate_hqq_nbits(expected_nbits)
    if entry.get("nbits") != expected_nbits:
        raise RuntimeError(
            f"HQQ artifact nbits mismatch for layer {entry.get('layer_idx')} "
            f"{entry.get('tensor_name')}: manifest has {entry.get('nbits')}, "
            f"expected {expected_nbits}"
        )

    path = os.path.join(hqq_attention_cache_dir(model_path), entry["file"])
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing HQQ artifact file: {path}")

    tensors = {}
    with safe_open(path, framework="pt", device=device) as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)

    expected_backend = hqq_backend_name(expected_nbits)
    if metadata.get("backend") != expected_backend:
        raise RuntimeError(
            f"HQQ artifact backend mismatch for {path}: "
            f"found {metadata.get('backend')}, expected {expected_backend}"
        )
    if metadata.get("layout") != entry.get("layout"):
        raise RuntimeError(
            f"HQQ artifact layout mismatch for {path}: "
            f"found {metadata.get('layout')}, expected {entry.get('layout')}"
        )
    structure = validate_hqq_attention_tensors(
        tensors,
        expected_nbits=expected_nbits,
        expected_layout=entry.get("layout", HQQ_LAYOUT),
    )

    return {
        "path": path,
        "metadata": metadata,
        "tensors": tensors,
        "tensor_bytes": _artifact_tensor_bytes(path),
        "structure": structure,
    }


def hqq_attention_cache_layer_bytes(model_path: str) -> Optional[Dict[int, int]]:
    manifest = load_hqq_attention_manifest(model_path)
    if not manifest:
        return None
    per_layer: Dict[int, int] = {}
    for entry in manifest.get("tensors", []):
        path = os.path.join(hqq_attention_cache_dir(model_path), entry["file"])
        if not os.path.isfile(path):
            return None
        per_layer[entry["layer_idx"]] = per_layer.get(entry["layer_idx"], 0) + _artifact_tensor_bytes(path)
    return per_layer


def hqq_attention_cache_total_bytes(model_path: str) -> Optional[int]:
    per_layer = hqq_attention_cache_layer_bytes(model_path)
    if per_layer is None:
        return None
    return sum(per_layer.values())
