"""Shared control-plane metadata for attention backends.

This module centralizes the user-facing backend surface and cache layout rules
without forcing different backends into one runtime tensor contract.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
import time
from typing import Dict, List, Optional

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from krasis.config import (
    HQQ_ATTENTION_DEFAULT_GROUP_SIZE,
    HQQ_ATTENTION_GROUP_SIZE_CHOICES,
    HQQ_CACHE_PROFILE_BASELINE,
    HQQ_CACHE_PROFILE_CHOICES,
    HQQ_CACHE_PROFILE_SELFCAL_V1,
    cache_dir_for_model,
)
from krasis.krasis import (
    hqq4_init_group_ptr,
    hqq4_quantize_tensor_ptr,
    hqq4_rmse_group_ptr,
    hqq4_solve_group_ptr,
)


ATTENTION_QUANT_CHOICES = ("bf16", "awq", "hqq4", "hqq46", "hqq46_auto", "hqq6", "hqq68_auto", "hqq8")

HQQ_ATTENTION_CACHE_VERSION = 5
HQQ_ATTENTION_CACHE_DIRNAME = f"attention_hqq_v{HQQ_ATTENTION_CACHE_VERSION}"
HQQ46_ATTENTION_CACHE_DIRNAME = f"attention_hqq46_v{HQQ_ATTENTION_CACHE_VERSION}"
HQQ46_AUTO_ATTENTION_CACHE_DIRNAME = f"attention_hqq46_auto_v{HQQ_ATTENTION_CACHE_VERSION}"
HQQ6_ATTENTION_CACHE_DIRNAME = f"attention_hqq6_v{HQQ_ATTENTION_CACHE_VERSION}"
HQQ68_AUTO_ATTENTION_CACHE_DIRNAME = f"attention_hqq68_auto_v{HQQ_ATTENTION_CACHE_VERSION}"
HQQ8_ATTENTION_CACHE_DIRNAME = f"attention_hqq8_v{HQQ_ATTENTION_CACHE_VERSION}"
HQQ_ATTENTION_MANIFEST = "manifest.json"
HQQ_ATTENTION_PENDING_MANIFEST = "manifest.build.json"
HQQ_DEFAULT_GROUP_SIZE = HQQ_ATTENTION_DEFAULT_GROUP_SIZE
HQQ_DEFAULT_AXIS = 1
HQQ_LAYOUT = "row_major_axis1_grouped_uint4_packed"
HQQ6_LAYOUT = "row_major_axis1_grouped_uint6_packed"
HQQ8_LAYOUT = "row_major_axis1_grouped_uint8"
HQQ_LAYOUT_BY_NBITS = {
    4: HQQ_LAYOUT,
    6: HQQ6_LAYOUT,
    8: HQQ8_LAYOUT,
}
HQQ_ENABLED_NBITS = (4, 6, 8)
HQQ_STORAGE_NBITS = (4, 6, 8)
HQQ_MIXED_46_CACHE_NBITS = 46
HQQ_MIXED_46_AUTO_CACHE_NBITS = 460
HQQ_MIXED_68_AUTO_CACHE_NBITS = 680
HQQ_CACHE_NBITS = (
    4,
    HQQ_MIXED_46_CACHE_NBITS,
    HQQ_MIXED_46_AUTO_CACHE_NBITS,
    6,
    HQQ_MIXED_68_AUTO_CACHE_NBITS,
    8,
)
HQQ46_LAYOUT = "mixed_hqq4_hqq6_axis1_grouped"
HQQ68_LAYOUT = "mixed_hqq6_hqq8_axis1_grouped"
HQQ46_PROMOTION_POLICY = {
    "name": "output_fused_v1",
    "description": "HQQ4 base with HQQ6 for fused_qkv, o_proj, and out_proj tensors.",
    "base_nbits": 4,
    "promoted_nbits": 6,
    "promoted_tensors": ("fused_qkv", "o_proj", "out_proj"),
}
HQQ46_AUTO_PROMOTION_POLICY = {
    "name": "weight_error_v1",
    "description": "HQQ4 base with per-tensor HQQ6 promotions selected by normalized error reduction per byte.",
    "base_nbits": 4,
    "promoted_nbits": 6,
}
HQQ68_AUTO_PROMOTION_POLICY = {
    "name": "weight_error_v1",
    "description": "HQQ6 base with per-tensor HQQ8 promotions selected by normalized error reduction per byte.",
    "base_nbits": 6,
    "promoted_nbits": 8,
}
HQQ_SIDECAR_MANIFEST_FORMAT = "krasis_hqq_selfcal_sidecar_manifest"
HQQ_SIDECAR_MANIFEST_FORMAT_VERSION = 1
HQQ_SIDECAR_MODES = ("int8_symmetric", "exact_bf16", "int8_exception")


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
    if attention_quant == "hqq8":
        return AttentionBackendSpec(
            quant="hqq8",
            label="HQQ8",
            quantized=True,
            requires_template=False,
            runtime_ready=True,
            nbits=8,
        )
    if attention_quant == "hqq46":
        return AttentionBackendSpec(
            quant="hqq46",
            label="HQQ4/6",
            quantized=True,
            requires_template=False,
            runtime_ready=True,
            nbits=HQQ_MIXED_46_CACHE_NBITS,
        )
    if attention_quant == "hqq46_auto":
        return AttentionBackendSpec(
            quant="hqq46_auto",
            label="HQQ4/6-auto",
            quantized=True,
            requires_template=False,
            runtime_ready=True,
            nbits=HQQ_MIXED_46_AUTO_CACHE_NBITS,
        )
    if attention_quant == "hqq68_auto":
        return AttentionBackendSpec(
            quant="hqq68_auto",
            label="HQQ6/8-auto",
            quantized=True,
            requires_template=False,
            runtime_ready=True,
            nbits=HQQ_MIXED_68_AUTO_CACHE_NBITS,
        )
    if attention_quant == "hqq6":
        return AttentionBackendSpec(
            quant="hqq6",
            label="HQQ6",
            quantized=True,
            requires_template=False,
            runtime_ready=True,
            nbits=6,
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
    if nbits == HQQ_MIXED_46_CACHE_NBITS:
        return "hqq46"
    if nbits == HQQ_MIXED_46_AUTO_CACHE_NBITS:
        return "hqq46_auto"
    if nbits == HQQ_MIXED_68_AUTO_CACHE_NBITS:
        return "hqq68_auto"
    return f"hqq{nbits}"


def validate_hqq_nbits(nbits: int) -> None:
    if nbits not in HQQ_ENABLED_NBITS:
        enabled = ", ".join(str(v) for v in HQQ_ENABLED_NBITS)
        raise ValueError(f"Unsupported HQQ nbits={nbits}. Enabled HQQ backends: {enabled}")


def validate_hqq_storage_nbits(nbits: int) -> None:
    if nbits not in HQQ_STORAGE_NBITS:
        enabled = ", ".join(str(v) for v in HQQ_STORAGE_NBITS)
        raise ValueError(f"Unsupported HQQ storage nbits={nbits}. Supported packed layouts: {enabled}")


def validate_hqq_cache_nbits(nbits: int) -> None:
    if nbits not in HQQ_CACHE_NBITS:
        enabled = ", ".join(str(v) for v in HQQ_CACHE_NBITS)
        raise ValueError(f"Unsupported HQQ cache nbits={nbits}. Enabled HQQ cache backends: {enabled}")


def attention_quant_cache_nbits(attention_quant: str) -> Optional[int]:
    return get_attention_backend_spec(attention_quant).nbits


def is_hqq_mixed_attention(attention_quant: str) -> bool:
    return attention_quant in ("hqq46", "hqq46_auto", "hqq68_auto")


def is_hqq46_auto_attention(attention_quant: str) -> bool:
    return attention_quant == "hqq46_auto"


def is_hqq_auto_attention(attention_quant: str) -> bool:
    return attention_quant in ("hqq46_auto", "hqq68_auto")


def hqq_auto_promotion_policy(attention_quant: str) -> dict:
    if attention_quant == "hqq46_auto":
        return HQQ46_AUTO_PROMOTION_POLICY
    if attention_quant == "hqq68_auto":
        return HQQ68_AUTO_PROMOTION_POLICY
    raise ValueError(f"attention_quant={attention_quant!r} is not an HQQ auto planner mode")


def hqq_auto_promotion_policy_for_cache_nbits(nbits: int) -> dict:
    if nbits == HQQ_MIXED_46_AUTO_CACHE_NBITS:
        return HQQ46_AUTO_PROMOTION_POLICY
    if nbits == HQQ_MIXED_68_AUTO_CACHE_NBITS:
        return HQQ68_AUTO_PROMOTION_POLICY
    raise ValueError(f"HQQ cache nbits={nbits} is not an auto planner cache")


def hqq_auto_direct_edge_nbits(attention_quant: str, budget_pct: Optional[float]) -> Optional[int]:
    if not is_hqq_auto_attention(attention_quant) or budget_pct is None:
        return None
    pct = normalize_hqq_auto_budget_pct(budget_pct, attention_quant)
    policy = hqq_auto_promotion_policy(attention_quant)
    if pct == 0.0:
        return int(policy["base_nbits"])
    if pct == 100.0:
        return int(policy["promoted_nbits"])
    return None


def hqq46_tensor_nbits(tensor_name: str) -> int:
    return 6 if tensor_name in HQQ46_PROMOTION_POLICY["promoted_tensors"] else 4


def hqq_layout_for_nbits(nbits: int) -> str:
    if nbits in (HQQ_MIXED_46_CACHE_NBITS, HQQ_MIXED_46_AUTO_CACHE_NBITS):
        return HQQ46_LAYOUT
    if nbits == HQQ_MIXED_68_AUTO_CACHE_NBITS:
        return HQQ68_LAYOUT
    validate_hqq_nbits(nbits)
    return HQQ_LAYOUT_BY_NBITS[nbits]


def _emit_real_model_timing(payload: dict) -> None:
    if os.environ.get("KRASIS_HQQ_REAL_MODEL_TIMING") == "1":
        print(json.dumps({"hqq_real_model_timing": payload}, sort_keys=True), flush=True)


def normalize_hqq_attention_cache_profile(cache_profile: Optional[str]) -> str:
    profile = str(cache_profile or HQQ_CACHE_PROFILE_BASELINE).strip().lower()
    if profile not in HQQ_CACHE_PROFILE_CHOICES:
        raise ValueError(
            f"Unsupported HQQ cache profile '{profile}'. "
            f"Use one of: {', '.join(HQQ_CACHE_PROFILE_CHOICES)}."
        )
    return profile


def normalize_hqq_attention_group_size(group_size: Optional[int]) -> int:
    value = HQQ_DEFAULT_GROUP_SIZE if group_size is None else int(group_size)
    if value not in HQQ_ATTENTION_GROUP_SIZE_CHOICES:
        choices = ", ".join(str(v) for v in HQQ_ATTENTION_GROUP_SIZE_CHOICES)
        raise ValueError(f"Unsupported HQQ attention group_size={value}. Use one of: {choices}.")
    return value


def _hqq_attention_cache_dirname(nbits: int, group_size: int) -> str:
    if nbits == 4:
        baseline = HQQ_ATTENTION_CACHE_DIRNAME
    elif nbits == HQQ_MIXED_46_CACHE_NBITS:
        baseline = HQQ46_ATTENTION_CACHE_DIRNAME
    elif nbits == HQQ_MIXED_46_AUTO_CACHE_NBITS:
        baseline = HQQ46_AUTO_ATTENTION_CACHE_DIRNAME
    elif nbits == 6:
        baseline = HQQ6_ATTENTION_CACHE_DIRNAME
    elif nbits == HQQ_MIXED_68_AUTO_CACHE_NBITS:
        baseline = HQQ68_AUTO_ATTENTION_CACHE_DIRNAME
    else:
        baseline = HQQ8_ATTENTION_CACHE_DIRNAME
    if group_size == HQQ_DEFAULT_GROUP_SIZE:
        return baseline
    return f"{baseline}_g{group_size}"


def hqq_attention_cache_dir(
    model_path: str,
    cache_profile: Optional[str] = HQQ_CACHE_PROFILE_BASELINE,
    nbits: int = 4,
    group_size: Optional[int] = HQQ_DEFAULT_GROUP_SIZE,
) -> str:
    """Return the native HQQ attention artifact directory under the model cache."""
    validate_hqq_cache_nbits(nbits)
    profile = normalize_hqq_attention_cache_profile(cache_profile)
    normalized_group_size = normalize_hqq_attention_group_size(group_size)
    baseline_dirname = _hqq_attention_cache_dirname(nbits, normalized_group_size)
    dirname = (
        baseline_dirname
        if profile == HQQ_CACHE_PROFILE_BASELINE
        else f"{baseline_dirname}_calib_selfcal_v1"
    )
    return os.path.join(cache_dir_for_model(model_path), dirname)


def hqq_attention_manifest_path(
    model_path: str,
    cache_profile: Optional[str] = HQQ_CACHE_PROFILE_BASELINE,
    nbits: int = 4,
    group_size: Optional[int] = HQQ_DEFAULT_GROUP_SIZE,
) -> str:
    return os.path.join(hqq_attention_cache_dir(model_path, cache_profile, nbits, group_size), HQQ_ATTENTION_MANIFEST)


def hqq_attention_pending_manifest_path(
    model_path: str,
    cache_profile: Optional[str] = HQQ_CACHE_PROFILE_BASELINE,
    nbits: int = 4,
    group_size: Optional[int] = HQQ_DEFAULT_GROUP_SIZE,
) -> str:
    return os.path.join(
        hqq_attention_cache_dir(model_path, cache_profile, nbits, group_size),
        HQQ_ATTENTION_PENDING_MANIFEST,
    )


def hqq_attention_cache_key(layer_idx: int, tensor_name: str, nbits: int = 4) -> str:
    """Stable filename stem for one HQQ attention tensor artifact."""
    return f"layer_{layer_idx:03d}_{tensor_name}_hqq{nbits}"


def hqq_attention_tensor_path(
    model_path: str,
    layer_idx: int,
    tensor_name: str,
    nbits: int = 4,
    cache_profile: Optional[str] = HQQ_CACHE_PROFILE_BASELINE,
    group_size: Optional[int] = HQQ_DEFAULT_GROUP_SIZE,
    cache_nbits: Optional[int] = None,
) -> str:
    actual_cache_nbits = nbits if cache_nbits is None else int(cache_nbits)
    return os.path.join(
        hqq_attention_cache_dir(model_path, cache_profile, actual_cache_nbits, group_size),
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


def hqq46_auto_budget_bytes_from_mib(value: Optional[int]) -> int:
    if value is None:
        raise ValueError("attention_quant=hqq46_auto requires hqq46_auto_budget_mib to be set.")
    budget_mib = int(value)
    if budget_mib <= 0:
        raise ValueError(
            f"attention_quant=hqq46_auto requires a positive hqq46_auto_budget_mib, got {budget_mib}."
        )
    return budget_mib * 1024 * 1024


def normalize_hqq_auto_budget_pct(value: Optional[float], attention_quant: str = "hqq_auto") -> float:
    if value is None:
        raise ValueError(f"attention_quant={attention_quant} requires hqq_auto_budget_pct to be set.")
    try:
        budget_pct = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"hqq_auto_budget_pct must be a percentage, got {value!r}") from exc
    if not math.isfinite(budget_pct) or budget_pct < 0.0 or budget_pct > 100.0:
        raise ValueError(
            f"attention_quant={attention_quant} requires 0 <= hqq_auto_budget_pct <= 100, "
            f"got {budget_pct!r}."
        )
    return budget_pct


def hqq_auto_budget_bytes_from_pct(value: float, promotion_span_bytes: int, attention_quant: str = "hqq_auto") -> int:
    budget_pct = normalize_hqq_auto_budget_pct(value, attention_quant)
    span = int(promotion_span_bytes)
    if span <= 0:
        raise ValueError(
            f"attention_quant={attention_quant} has non-positive promotion span: {promotion_span_bytes}"
        )
    if budget_pct == 0.0:
        return 0
    if budget_pct == 100.0:
        return span
    return max(1, math.floor(span * (budget_pct / 100.0)))


def _quality_relative_rmse(record: dict) -> float:
    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
    if "relative_rmse" in quality:
        return float(quality["relative_rmse"])
    rmse = float(quality.get("rmse", 0.0))
    source_rms = float(quality.get("source_rms", 0.0))
    return rmse / source_rms if source_rms > 0.0 else 0.0


def hqq_auto_candidate_from_records(base_record: dict, promoted_record: dict) -> dict:
    base_bytes = int(base_record["tensor_bytes"])
    promoted_bytes = int(promoted_record["tensor_bytes"])
    extra_bytes = promoted_bytes - base_bytes
    if extra_bytes <= 0:
        raise ValueError(
            f"HQQ auto candidate for layer {base_record.get('layer_idx')} "
            f"{base_record.get('tensor_name')} has non-positive promotion cost: {extra_bytes}"
        )
    base_rel_rmse = _quality_relative_rmse(base_record)
    promoted_rel_rmse = _quality_relative_rmse(promoted_record)
    gain = max(0.0, base_rel_rmse - promoted_rel_rmse)
    score = gain / float(extra_bytes)
    return {
        "layer_idx": int(base_record["layer_idx"]),
        "layer_type": base_record["layer_type"],
        "tensor_name": base_record["tensor_name"],
        "base_nbits": int(base_record["nbits"]),
        "promoted_nbits": int(promoted_record["nbits"]),
        "base_bytes": base_bytes,
        "promoted_bytes": promoted_bytes,
        "extra_bytes": extra_bytes,
        "base_relative_rmse": base_rel_rmse,
        "promoted_relative_rmse": promoted_rel_rmse,
        "relative_rmse_reduction": gain,
        "score": score,
        "base_record": base_record,
        "promoted_record": promoted_record,
    }


def hqq46_auto_candidate_from_records(base_record: dict, promoted_record: dict) -> dict:
    return hqq_auto_candidate_from_records(base_record, promoted_record)


def select_hqq_auto_promotions(
    candidates: List[dict],
    budget_bytes: int,
    *,
    max_dp_units: int = 4096,
) -> tuple[set[tuple[int, str]], dict]:
    """Select HQQ promotions by budgeted normalized-error reduction.

    Uses a bounded-resolution 0/1 knapsack. The unit size is derived from the
    requested budget so the planner remains cheap for large models while still
    optimizing globally instead of choosing broad tensor families.
    """
    budget_bytes = int(budget_bytes)
    if budget_bytes < 0:
        raise ValueError(f"HQQ auto promotion budget must be non-negative, got {budget_bytes}")
    if not candidates:
        return set(), {
            "budget_bytes": budget_bytes,
            "budget_used_bytes": 0,
            "budget_unit_bytes": 1,
            "selected_count": 0,
            "candidate_count": 0,
            "relative_rmse_reduction": 0.0,
            "selection_mode": "empty",
        }
    if budget_bytes == 0:
        return set(), {
            "budget_bytes": 0,
            "budget_used_bytes": 0,
            "budget_unit_bytes": 1,
            "selected_count": 0,
            "candidate_count": len(candidates),
            "relative_rmse_reduction": 0.0,
            "selection_mode": "zero_budget",
        }

    promotion_span_bytes = sum(int(candidate["extra_bytes"]) for candidate in candidates)
    if budget_bytes >= promotion_span_bytes:
        selected_keys = {
            (int(candidate["layer_idx"]), str(candidate["tensor_name"]))
            for candidate in candidates
        }
        reduction = sum(float(candidate["relative_rmse_reduction"]) for candidate in candidates)
        return selected_keys, {
            "budget_bytes": budget_bytes,
            "budget_used_bytes": int(promotion_span_bytes),
            "budget_unit_bytes": 1,
            "selected_count": len(candidates),
            "candidate_count": len(candidates),
            "relative_rmse_reduction": reduction,
            "selection_mode": "full_span",
        }

    unit_bytes = max(1, math.ceil(budget_bytes / max(1, int(max_dp_units))))
    capacity = max(1, budget_bytes // unit_bytes)
    dp_gain = [0.0] * (capacity + 1)
    dp_choice: list[list[int]] = [[] for _ in range(capacity + 1)]

    for idx, candidate in enumerate(candidates):
        gain = float(candidate["relative_rmse_reduction"])
        if gain <= 0.0:
            continue
        cost_units = max(1, math.ceil(int(candidate["extra_bytes"]) / unit_bytes))
        if cost_units > capacity:
            continue
        for cap in range(capacity, cost_units - 1, -1):
            new_gain = dp_gain[cap - cost_units] + gain
            if new_gain > dp_gain[cap]:
                dp_gain[cap] = new_gain
                dp_choice[cap] = dp_choice[cap - cost_units] + [idx]

    best_cap = max(range(capacity + 1), key=lambda cap: dp_gain[cap])
    selected_indices = set(dp_choice[best_cap])
    selected_keys = {
        (int(candidates[idx]["layer_idx"]), str(candidates[idx]["tensor_name"]))
        for idx in selected_indices
    }
    budget_used = sum(int(candidates[idx]["extra_bytes"]) for idx in selected_indices)
    reduction = sum(float(candidates[idx]["relative_rmse_reduction"]) for idx in selected_indices)
    summary = {
        "budget_bytes": budget_bytes,
        "budget_used_bytes": int(budget_used),
        "budget_unit_bytes": int(unit_bytes),
        "selected_count": len(selected_indices),
        "candidate_count": len(candidates),
        "relative_rmse_reduction": reduction,
        "selection_mode": "knapsack",
    }
    return selected_keys, summary


def select_hqq46_auto_promotions(
    candidates: List[dict],
    budget_bytes: int,
    *,
    max_dp_units: int = 4096,
) -> tuple[set[tuple[int, str]], dict]:
    return select_hqq_auto_promotions(candidates, budget_bytes, max_dp_units=max_dp_units)


def load_hqq_attention_manifest(
    model_path: str,
    cache_profile: Optional[str] = HQQ_CACHE_PROFILE_BASELINE,
    nbits: int = 4,
    group_size: Optional[int] = HQQ_DEFAULT_GROUP_SIZE,
) -> Optional[dict]:
    path = hqq_attention_manifest_path(model_path, cache_profile, nbits, group_size)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_hqq_sidecar_manifest(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    expanded = os.path.expanduser(str(path))
    if not os.path.isfile(expanded):
        return None
    with open(expanded) as f:
        manifest = json.load(f)
    manifest["_manifest_path"] = os.path.abspath(expanded)
    return manifest


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        data = cpu.view(torch.uint16).numpy().tobytes()
    else:
        data = cpu.numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _contract_shape(contract: dict) -> Optional[tuple[int, int]]:
    for key in ("reconstructed_source_shape", "source_shape", "target_shape"):
        shape = contract.get(key)
        if isinstance(shape, list) and len(shape) == 2:
            return (int(shape[0]), int(shape[1]))
    return None


def _validate_hqq_sidecar_artifact_entries(artifact: dict, *, mode: str, artifact_path: str) -> None:
    entries = artifact.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"HQQ sidecar artifact has no explicit entries: {artifact_path}")
    contract = artifact.get("source_contract") if isinstance(artifact.get("source_contract"), dict) else {}
    shape = _contract_shape(contract)
    expected_layer = int(artifact.get("layer", -1))
    expected_tensor = str(artifact.get("tensor") or "")
    total_rows = 0
    seen: set[tuple[int, str, int, int, int]] = set()
    for idx, entry in enumerate(entries):
        try:
            layer = int(entry.get("layer", expected_layer))
            tensor = str(entry.get("tensor", expected_tensor))
            group = int(entry["group"])
            start_col = int(entry["start_col"])
            end_col = int(entry["end_col"])
            width = int(entry.get("width", end_col - start_col))
            if "output_row" in entry:
                output_row_start = int(entry["output_row"])
                output_row_end = output_row_start + 1
                row_count = 1
            else:
                output_row_start = int(entry.get("output_row_start", 0))
                output_row_end = int(entry["output_row_end"])
                row_count = int(entry["row_count"])
        except Exception as exc:
            raise RuntimeError(f"HQQ sidecar artifact entry {idx} has invalid required fields: {artifact_path}") from exc
        if layer != expected_layer or tensor != expected_tensor:
            raise RuntimeError(
                f"HQQ sidecar artifact entry {idx} target mismatch: "
                f"{layer}:{tensor} != {expected_layer}:{expected_tensor}"
            )
        if group < 0 or start_col < 0 or end_col <= start_col or width != end_col - start_col:
            raise RuntimeError(f"HQQ sidecar artifact entry {idx} has invalid column bounds: {artifact_path}")
        if output_row_start < 0 or output_row_end <= output_row_start:
            raise RuntimeError(f"HQQ sidecar artifact entry {idx} has invalid output row bounds: {artifact_path}")
        if row_count != output_row_end - output_row_start:
            raise RuntimeError(f"HQQ sidecar artifact entry {idx} row_count does not match row bounds: {artifact_path}")
        if shape is not None:
            rows, cols = shape
            if output_row_end > rows or end_col > cols:
                raise RuntimeError(
                    f"HQQ sidecar artifact entry {idx} exceeds source shape {rows}x{cols}: {artifact_path}"
                )
        key = (layer, tensor, group, output_row_start, output_row_end)
        if mode == "int8_exception" and key in seen:
            raise RuntimeError(f"Duplicate INT8 exception entry for {key}: {artifact_path}")
        seen.add(key)
        total_rows += row_count
    if int(artifact.get("row_group_count", 0)) != total_rows:
        raise RuntimeError(
            f"HQQ sidecar artifact row_group_count mismatch: "
            f"{artifact.get('row_group_count')} != summed rows {total_rows}"
        )
    if mode == "int8_exception":
        if int(artifact.get("exception_group_count", 0)) != len(entries):
            raise RuntimeError(
                f"INT8 exception artifact exception_group_count mismatch: "
                f"{artifact.get('exception_group_count')} != {len(entries)}"
            )
        if artifact.get("hqq_artifact_sha256") and contract.get("artifact_sha256"):
            if artifact.get("hqq_artifact_sha256") != contract.get("artifact_sha256"):
                raise RuntimeError(f"INT8 exception artifact HQQ sha256 does not match source contract: {artifact_path}")


def require_complete_hqq_sidecar_manifest(
    path: str,
    *,
    model_path: str,
    source_cache_profile: Optional[str] = HQQ_CACHE_PROFILE_BASELINE,
) -> dict:
    """Validate an explicit HQQ sidecar manifest without treating it as a cache manifest."""
    expanded = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(expanded):
        raise RuntimeError(f"HQQ sidecar manifest not found: {expanded}")
    manifest = load_hqq_sidecar_manifest(expanded)
    if manifest is None:
        raise RuntimeError(f"HQQ sidecar manifest not found: {expanded}")
    if manifest.get("format") != HQQ_SIDECAR_MANIFEST_FORMAT:
        raise RuntimeError(
            f"Not an HQQ sidecar manifest: {expanded} format={manifest.get('format')!r}"
        )
    if int(manifest.get("format_version", -1)) != HQQ_SIDECAR_MANIFEST_FORMAT_VERSION:
        raise RuntimeError(
            "Unsupported HQQ sidecar manifest format_version="
            f"{manifest.get('format_version')} expected={HQQ_SIDECAR_MANIFEST_FORMAT_VERSION}"
        )
    if not manifest.get("complete"):
        raise RuntimeError(f"HQQ sidecar manifest is incomplete: {expanded}")
    mode = str(manifest.get("sidecar_mode", ""))
    if mode not in HQQ_SIDECAR_MODES:
        raise RuntimeError(
            f"Unsupported HQQ sidecar mode {mode!r}. Use one of: {', '.join(HQQ_SIDECAR_MODES)}."
        )
    expected_model = os.path.abspath(os.path.expanduser(model_path))
    manifest_model = os.path.abspath(os.path.expanduser(str(manifest.get("model_path", ""))))
    if manifest_model != expected_model:
        raise RuntimeError(
            f"HQQ sidecar model_path mismatch: manifest={manifest_model} expected={expected_model}"
        )
    expected_profile = normalize_hqq_attention_cache_profile(source_cache_profile)
    actual_profile = normalize_hqq_attention_cache_profile(
        manifest.get("source", {}).get("cache_profile")
    )
    if actual_profile != expected_profile:
        raise RuntimeError(
            f"HQQ sidecar source cache profile mismatch: manifest={actual_profile} expected={expected_profile}. "
            "No fallback to another HQQ profile is allowed."
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError(f"HQQ sidecar manifest has no artifacts: {expanded}")
    base_dir = os.path.dirname(expanded)
    source_cache_dir = hqq_attention_cache_dir(model_path, expected_profile)
    for artifact in artifacts:
        rel_file = artifact.get("file")
        if not rel_file:
            raise RuntimeError(f"HQQ sidecar artifact entry is missing file: {expanded}")
        artifact_path = os.path.join(base_dir, rel_file)
        if not os.path.isfile(artifact_path):
            raise RuntimeError(f"Missing HQQ sidecar artifact file: {artifact_path}")
        artifact_sha = artifact.get("sha256")
        if artifact_sha and _file_sha256(artifact_path) != artifact_sha:
            raise RuntimeError(f"HQQ sidecar artifact sha256 mismatch: {artifact_path}")
        if int(artifact.get("row_group_count", 0)) <= 0:
            raise RuntimeError(f"HQQ sidecar artifact has no row/groups: {artifact_path}")
        contract = artifact.get("source_contract")
        if isinstance(contract, dict):
            if bool(contract.get("fallback_allowed")):
                raise RuntimeError(f"HQQ sidecar source_contract fallback_allowed=true: {artifact_path}")
            if not bool(contract.get("ready")):
                raise RuntimeError(f"HQQ sidecar source_contract is not ready: {artifact_path}")
            if str(contract.get("target_tensor") or "") != str(artifact.get("tensor") or ""):
                raise RuntimeError(
                    "HQQ sidecar source_contract target tensor mismatch: "
                    f"{contract.get('target_tensor')} != {artifact.get('tensor')}"
                )
            if int(contract.get("layer", -1)) != int(artifact.get("layer", -2)):
                raise RuntimeError(
                    "HQQ sidecar source_contract layer mismatch: "
                    f"{contract.get('layer')} != {artifact.get('layer')}"
                )
            target_file = str(contract.get("target_file") or "")
            target_sha = str(contract.get("artifact_sha256") or "")
            if target_file and target_sha:
                target_path = os.path.join(source_cache_dir, target_file)
                if not os.path.isfile(target_path):
                    raise RuntimeError(f"HQQ sidecar source HQQ artifact is missing: {target_path}")
                if _file_sha256(target_path) != target_sha:
                    raise RuntimeError(f"HQQ sidecar source HQQ artifact sha256 mismatch: {target_path}")
            source_tensors = contract.get("source_tensors")
            if isinstance(source_tensors, list):
                for source in source_tensors:
                    source_path = str(source.get("source_path") or "")
                    source_name = str(source.get("name") or "")
                    source_sha = str(source.get("source_sha256") or "")
                    if not source_path or not source_name or not source_sha:
                        raise RuntimeError(
                            "HQQ sidecar source_contract source tensor lacks source_path/name/source_sha256"
                        )
                    if not os.path.isfile(source_path):
                        raise RuntimeError(f"HQQ sidecar source tensor file is missing: {source_path}")
                    with safe_open(source_path, framework="pt", device="cpu") as handle:
                        if source_name not in handle.keys():
                            raise RuntimeError(
                                f"HQQ sidecar source tensor {source_name!r} missing from {source_path}"
                            )
                        tensor_sha = _tensor_sha256(handle.get_tensor(source_name))
                    if tensor_sha != source_sha:
                        raise RuntimeError(
                            f"HQQ sidecar source tensor sha256 mismatch: {source_path}:{source_name}"
                        )
        _validate_hqq_sidecar_artifact_entries(artifact, mode=mode, artifact_path=artifact_path)
    return manifest


def _save_hqq_attention_manifest_to_path(path: str, manifest: dict) -> None:
    cache_dir = os.path.dirname(path)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def save_hqq_attention_manifest(
    model_path: str,
    manifest: dict,
    cache_profile: Optional[str] = HQQ_CACHE_PROFILE_BASELINE,
    nbits: int = 4,
    group_size: Optional[int] = HQQ_DEFAULT_GROUP_SIZE,
) -> None:
    _save_hqq_attention_manifest_to_path(
        hqq_attention_manifest_path(model_path, cache_profile, nbits, group_size),
        manifest,
    )


def load_hqq_attention_pending_manifest(
    model_path: str,
    cache_profile: Optional[str] = HQQ_CACHE_PROFILE_BASELINE,
    nbits: int = 4,
    group_size: Optional[int] = HQQ_DEFAULT_GROUP_SIZE,
) -> Optional[dict]:
    path = hqq_attention_pending_manifest_path(model_path, cache_profile, nbits, group_size)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_hqq_attention_pending_manifest(
    model_path: str,
    manifest: dict,
    cache_profile: Optional[str] = HQQ_CACHE_PROFILE_BASELINE,
    nbits: int = 4,
    group_size: Optional[int] = HQQ_DEFAULT_GROUP_SIZE,
) -> None:
    _save_hqq_attention_manifest_to_path(
        hqq_attention_pending_manifest_path(model_path, cache_profile, nbits, group_size),
        manifest,
    )


def delete_hqq_attention_pending_manifest(
    model_path: str,
    cache_profile: Optional[str] = HQQ_CACHE_PROFILE_BASELINE,
    nbits: int = 4,
    group_size: Optional[int] = HQQ_DEFAULT_GROUP_SIZE,
) -> None:
    path = hqq_attention_pending_manifest_path(model_path, cache_profile, nbits, group_size)
    if os.path.isfile(path):
        os.remove(path)


def init_hqq_attention_manifest(
    model_path: str,
    num_hidden_layers: int,
    nbits: int = 4,
    group_size: Optional[int] = HQQ_DEFAULT_GROUP_SIZE,
) -> dict:
    validate_hqq_cache_nbits(nbits)
    normalized_group_size = normalize_hqq_attention_group_size(group_size)
    layout = hqq_layout_for_nbits(nbits)
    manifest = {
        "format_version": HQQ_ATTENTION_CACHE_VERSION,
        "backend": hqq_backend_name(nbits),
        "nbits": nbits,
        "group_size": normalized_group_size,
        "axis": HQQ_DEFAULT_AXIS,
        "layout": layout,
        "num_hidden_layers": num_hidden_layers,
        "complete": False,
        "tensors": [],
        "totals": {
            "tensor_bytes": 0,
            "num_tensors": 0,
        },
    }
    if nbits == HQQ_MIXED_46_CACHE_NBITS:
        manifest["mixed_precision"] = {
            "base_nbits": HQQ46_PROMOTION_POLICY["base_nbits"],
            "promoted_nbits": HQQ46_PROMOTION_POLICY["promoted_nbits"],
            "policy": HQQ46_PROMOTION_POLICY["name"],
            "promoted_tensors": list(HQQ46_PROMOTION_POLICY["promoted_tensors"]),
            "description": HQQ46_PROMOTION_POLICY["description"],
        }
    elif nbits in (HQQ_MIXED_46_AUTO_CACHE_NBITS, HQQ_MIXED_68_AUTO_CACHE_NBITS):
        policy = hqq_auto_promotion_policy_for_cache_nbits(nbits)
        manifest["mixed_precision"] = {
            "base_nbits": policy["base_nbits"],
            "promoted_nbits": policy["promoted_nbits"],
            "policy": policy["name"],
            "description": policy["description"],
        }
        manifest["planner"] = {
            "name": policy["name"],
            "candidate_count": 0,
            "candidates": [],
        }
    return manifest


def require_complete_hqq_attention_manifest(
    model_path: str,
    cache_profile: Optional[str],
    expected_nbits: int,
    expected_num_hidden_layers: int,
    expected_group_size: Optional[int] = HQQ_DEFAULT_GROUP_SIZE,
) -> dict:
    """Load a selected complete HQQ manifest, failing visibly without fallback."""
    validate_hqq_cache_nbits(expected_nbits)
    normalized_group_size = normalize_hqq_attention_group_size(expected_group_size)
    profile = normalize_hqq_attention_cache_profile(cache_profile)
    path = hqq_attention_manifest_path(model_path, profile, expected_nbits, normalized_group_size)
    manifest = load_hqq_attention_manifest(model_path, profile, expected_nbits, normalized_group_size)
    if manifest is None:
        if profile == HQQ_CACHE_PROFILE_BASELINE:
            raise RuntimeError(
                f"attention_quant=hqq{expected_nbits} requested but no HQQ attention manifest exists. "
                f"Expected {path}"
            )
        raise RuntimeError(
            f"HQQ cache profile '{profile}' was explicitly selected, but no calibrated "
            f"HQQ attention manifest exists at {path}. No fallback to baseline is allowed."
        )

    expected_backend = hqq_backend_name(expected_nbits)
    errors = []
    if manifest.get("format_version") != HQQ_ATTENTION_CACHE_VERSION:
        errors.append(f"format_version={manifest.get('format_version')}")
    if manifest.get("backend") != expected_backend:
        errors.append(f"backend={manifest.get('backend')}")
    if manifest.get("nbits") != expected_nbits:
        errors.append(f"nbits={manifest.get('nbits')}")
    if manifest.get("num_hidden_layers") != expected_num_hidden_layers:
        errors.append(f"num_hidden_layers={manifest.get('num_hidden_layers')}")
    if manifest.get("group_size") != normalized_group_size:
        errors.append(f"group_size={manifest.get('group_size')}")
    if manifest.get("axis") != HQQ_DEFAULT_AXIS:
        errors.append(f"axis={manifest.get('axis')}")
    expected_layout = hqq_layout_for_nbits(expected_nbits)
    if manifest.get("layout") != expected_layout:
        errors.append(f"layout={manifest.get('layout')}")
    if not manifest.get("complete"):
        errors.append("complete=false")

    if profile != HQQ_CACHE_PROFILE_BASELINE:
        if manifest.get("cache_profile") != profile:
            errors.append(f"cache_profile={manifest.get('cache_profile')}")
        calibration = manifest.get("calibration")
        if not isinstance(calibration, dict):
            errors.append("calibration=<missing>")
        elif calibration.get("profile") != profile:
            errors.append(f"calibration.profile={calibration.get('profile')}")

    if errors:
        no_fallback = " No fallback to baseline is allowed." if profile != HQQ_CACHE_PROFILE_BASELINE else ""
        raise RuntimeError(
            f"HQQ attention cache manifest for profile '{profile}' is incomplete or "
            f"incompatible at {path}: {', '.join(errors)}.{no_fallback}"
        )

    return manifest


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


def _pack_uint6(q: torch.Tensor) -> torch.Tensor:
    if q.dtype != torch.uint8:
        q = q.to(torch.uint8)
    if q.shape[-1] % 4 != 0:
        q = torch.nn.functional.pad(q, (0, 4 - (q.shape[-1] % 4)))
    q = q.contiguous()
    a = q[..., 0::4] & 0x3F
    b = q[..., 1::4] & 0x3F
    c = q[..., 2::4] & 0x3F
    d = q[..., 3::4] & 0x3F
    packed = torch.empty((*q.shape[:-1], (q.shape[-1] // 4) * 3), dtype=torch.uint8)
    packed[..., 0::3] = a | ((b & 0x03) << 6)
    packed[..., 1::3] = (b >> 2) | ((c & 0x0F) << 4)
    packed[..., 2::3] = (c >> 4) | (d << 2)
    return packed.contiguous()


def _unpack_uint6(packed: torch.Tensor, cols: int) -> torch.Tensor:
    if packed.shape[-1] % 3 != 0:
        raise ValueError(f"Packed uint6 columns must be divisible by 3, got {packed.shape[-1]}")
    a = packed[..., 0::3]
    b = packed[..., 1::3]
    c = packed[..., 2::3]
    q = torch.empty((*packed.shape[:-1], (packed.shape[-1] // 3) * 4), dtype=torch.uint8)
    q[..., 0::4] = a & 0x3F
    q[..., 1::4] = ((a >> 6) | ((b & 0x0F) << 2)) & 0x3F
    q[..., 2::4] = ((b >> 4) | ((c & 0x03) << 4)) & 0x3F
    q[..., 3::4] = (c >> 2) & 0x3F
    return q[..., :cols].contiguous()


def _pack_hqq_quant(quant: torch.Tensor, nbits: int) -> torch.Tensor:
    if nbits == 4:
        return _pack_uint4(quant)
    if nbits == 6:
        return _pack_uint6(quant)
    if nbits == 8:
        if quant.dtype != torch.uint8:
            quant = quant.to(torch.uint8)
        return quant.contiguous()
    raise ValueError(f"Unsupported HQQ nbits={nbits}")


def _unpack_hqq_quant(packed: torch.Tensor, cols: int, nbits: int) -> torch.Tensor:
    if nbits == 4:
        return _unpack_uint4(packed, cols)
    if nbits == 6:
        return _unpack_uint6(packed, cols)
    if nbits == 8:
        return packed[:, :cols].contiguous()
    raise ValueError(f"Unsupported HQQ nbits={nbits}")


def _hqq_num_groups(cols: int, group_size: int) -> int:
    return (cols + group_size - 1) // group_size


def _hqq_padded_cols(cols: int, group_size: int) -> int:
    return _hqq_num_groups(cols, group_size) * group_size


def _hqq_packed_cols(cols: int, group_size: int) -> int:
    return (_hqq_padded_cols(cols, group_size) + 1) // 2


def _hqq_packed_cols_for_nbits(cols: int, group_size: int, nbits: int) -> int:
    validate_hqq_storage_nbits(nbits)
    padded_cols = _hqq_padded_cols(cols, group_size)
    if nbits == 4:
        return (padded_cols + 1) // 2
    if nbits == 6:
        return ((padded_cols + 3) // 4) * 3
    return padded_cols


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
    if expected_nbits is not None:
        layout_for_bits = hqq_layout_for_nbits(expected_nbits)
        if expected_layout != layout_for_bits:
            raise RuntimeError(
                f"HQQ validation expected layout {expected_layout}, but nbits={expected_nbits} requires {layout_for_bits}"
            )
    if expected_layout not in HQQ_LAYOUT_BY_NBITS.values():
        raise RuntimeError(
            f"HQQ validation only supports runtime layouts {sorted(HQQ_LAYOUT_BY_NBITS.values())}, got expected {expected_layout}"
        )

    groups = _hqq_num_groups(cols, group_size)
    padded_cols = _hqq_padded_cols(cols, group_size)
    if nbits == 4:
        packed_cols = _hqq_packed_cols(cols, group_size)
    elif nbits == 6:
        packed_cols = _hqq_packed_cols_for_nbits(cols, group_size, 6)
    elif nbits == 8:
        packed_cols = padded_cols
    else:
        raise RuntimeError(f"Unsupported HQQ nbits={nbits}")
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


def quantize_hqq8_tensor(
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
    started = time.perf_counter() if timing_enabled else 0.0

    w = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    rows, cols = w.shape
    groups = _hqq_num_groups(cols, group_size)
    padded_cols = _hqq_padded_cols(cols, group_size)
    quant = torch.zeros((rows, padded_cols), dtype=torch.uint8)
    scales = torch.empty((rows, groups), dtype=torch.float32)
    zeros = torch.empty((rows, groups), dtype=torch.float32)
    qmax = 255.0
    stats = {
        "numel": 0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "max_abs": 0.0,
        "worst_groups": [],
    } if collect_stats else None

    for g in range(groups):
        start = g * group_size
        end = min(start + group_size, cols)
        chunk = w[:, start:end]
        q, scale, zero = _quantize_hqq4_group_current(chunk, qmax)
        quant[:, start:end] = q[:, : end - start]
        scales[:, g] = scale
        zeros[:, g] = zero
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

    if timing_enabled:
        _emit_real_model_timing(
            {
                **timing_prefix,
                "phase": "artifact_pack_stage",
                "stage": "quantize_hqq8_affine_pack",
                "elapsed_s": time.perf_counter() - started,
                "rows": int(rows),
                "cols": int(cols),
                "groups": int(groups),
            }
        )

    result = {
        "packed": quant.contiguous(),
        "scales": scales,
        "zeros": zeros,
        "orig_shape": torch.tensor([rows, cols], dtype=torch.int32),
        "group_size": torch.tensor([group_size], dtype=torch.int32),
        "axis": torch.tensor([HQQ_DEFAULT_AXIS], dtype=torch.int32),
        "nbits": torch.tensor([8], dtype=torch.int32),
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


def quantize_hqq6_tensor(
    weight: torch.Tensor,
    group_size: int = HQQ_DEFAULT_GROUP_SIZE,
    *,
    collect_stats: bool = False,
    worst_groups_limit: int = 8,
) -> Dict[str, torch.Tensor]:
    """HQQ6 tensor quantizer with true packed 6-bit storage."""
    if weight.ndim != 2:
        raise ValueError(f"HQQ attention expects 2D weights, got shape {tuple(weight.shape)}")
    if group_size <= 0:
        raise ValueError(f"Invalid HQQ group_size {group_size}")

    w = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    rows, cols = w.shape
    groups = _hqq_num_groups(cols, group_size)
    padded_cols = _hqq_padded_cols(cols, group_size)
    quant = torch.zeros((rows, padded_cols), dtype=torch.uint8)
    scales = torch.empty((rows, groups), dtype=torch.float32)
    zeros = torch.empty((rows, groups), dtype=torch.float32)
    qmax = 63.0
    stats = {
        "numel": 0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "max_abs": 0.0,
        "worst_groups": [],
    } if collect_stats else None

    for g in range(groups):
        start = g * group_size
        end = min(start + group_size, cols)
        chunk = w[:, start:end]
        q, scale, zero = _quantize_hqq4_group_current(chunk, qmax)
        quant[:, start:end] = q[:, : end - start]
        scales[:, g] = scale
        zeros[:, g] = zero
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

    result = {
        "packed": _pack_uint6(quant),
        "scales": scales,
        "zeros": zeros,
        "orig_shape": torch.tensor([rows, cols], dtype=torch.int32),
        "group_size": torch.tensor([group_size], dtype=torch.int32),
        "axis": torch.tensor([HQQ_DEFAULT_AXIS], dtype=torch.int32),
        "nbits": torch.tensor([6], dtype=torch.int32),
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


def quantize_hqq6_tensor_probe(
    weight: torch.Tensor,
    group_size: int = HQQ_DEFAULT_GROUP_SIZE,
    *,
    collect_stats: bool = False,
    worst_groups_limit: int = 8,
) -> Dict[str, torch.Tensor]:
    return quantize_hqq6_tensor(
        weight,
        group_size=group_size,
        collect_stats=collect_stats,
        worst_groups_limit=worst_groups_limit,
    )


def write_hqq_attention_artifact(
    model_path: str,
    layer_idx: int,
    layer_type: str,
    tensor_name: str,
    weight: torch.Tensor,
    nbits: int = 4,
    cache_profile: Optional[str] = HQQ_CACHE_PROFILE_BASELINE,
    group_size: Optional[int] = HQQ_DEFAULT_GROUP_SIZE,
    cache_nbits: Optional[int] = None,
) -> dict:
    validate_hqq_nbits(nbits)
    normalized_group_size = normalize_hqq_attention_group_size(group_size)
    actual_cache_nbits = nbits if cache_nbits is None else int(cache_nbits)
    validate_hqq_cache_nbits(actual_cache_nbits)

    path = hqq_attention_tensor_path(
        model_path,
        layer_idx,
        tensor_name,
        nbits=nbits,
        cache_profile=cache_profile,
        group_size=normalized_group_size,
        cache_nbits=actual_cache_nbits,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    timing_context = {
        "layer_idx": int(layer_idx),
        "layer_type": layer_type,
        "tensor_name": tensor_name,
    }
    if nbits == 4:
        tensors = quantize_hqq4_tensor_rust(
            weight,
            group_size=normalized_group_size,
            collect_stats=True,
        )
    elif nbits == 6:
        tensors = quantize_hqq6_tensor(
            weight,
            group_size=normalized_group_size,
            collect_stats=True,
        )
    elif nbits == 8:
        tensors = quantize_hqq8_tensor(
            weight,
            group_size=normalized_group_size,
            collect_stats=True,
            timing_context=timing_context,
        )
    else:
        raise ValueError(f"Unsupported HQQ nbits={nbits}")
    metadata_started = time.perf_counter() if os.environ.get("KRASIS_HQQ_REAL_MODEL_TIMING") == "1" else 0.0
    payload_tensors = {k: v for k, v in tensors.items() if isinstance(v, torch.Tensor)}
    structure = validate_hqq_attention_tensors(
        payload_tensors,
        expected_nbits=nbits,
        expected_layout=hqq_layout_for_nbits(nbits),
    )
    metadata = {
        "backend": hqq_backend_name(nbits),
        "format_version": str(HQQ_ATTENTION_CACHE_VERSION),
        "layer_idx": str(layer_idx),
        "layer_type": layer_type,
        "tensor_name": tensor_name,
        "layout": hqq_layout_for_nbits(nbits),
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
    source_f32 = weight.detach().to(device="cpu", dtype=torch.float32)
    source_numel = max(1, source_f32.numel())
    source_rms = math.sqrt(float((source_f32 * source_f32).sum().item()) / source_numel)
    source_mean_abs = float(source_f32.abs().sum().item()) / source_numel
    stats["source_rms"] = source_rms
    stats["source_mean_abs"] = source_mean_abs
    stats["relative_rmse"] = stats["rmse"] / source_rms if source_rms > 0.0 else 0.0
    stats["relative_mean_abs"] = stats["mean_abs"] / source_mean_abs if source_mean_abs > 0.0 else 0.0
    return {
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "tensor_name": tensor_name,
        "file": os.path.basename(path),
        "path": path,
        "nbits": nbits,
        "group_size": int(tensors["group_size"][0].item()),
        "axis": int(tensors["axis"][0].item()),
        "layout": hqq_layout_for_nbits(nbits),
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
    cache_profile: Optional[str] = HQQ_CACHE_PROFILE_BASELINE,
    group_size: Optional[int] = None,
    cache_nbits: Optional[int] = None,
) -> dict:
    validate_hqq_nbits(expected_nbits)
    if entry.get("nbits") != expected_nbits:
        raise RuntimeError(
            f"HQQ artifact nbits mismatch for layer {entry.get('layer_idx')} "
            f"{entry.get('tensor_name')}: manifest has {entry.get('nbits')}, "
            f"expected {expected_nbits}"
        )

    artifact_group_size = normalize_hqq_attention_group_size(group_size or entry.get("group_size", HQQ_DEFAULT_GROUP_SIZE))
    actual_cache_nbits = expected_nbits if cache_nbits is None else int(cache_nbits)
    validate_hqq_cache_nbits(actual_cache_nbits)
    path = os.path.join(
        hqq_attention_cache_dir(model_path, cache_profile, actual_cache_nbits, artifact_group_size),
        entry["file"],
    )
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


def hqq_attention_cache_layer_bytes(
    model_path: str,
    cache_profile: Optional[str] = HQQ_CACHE_PROFILE_BASELINE,
    nbits: int = 4,
    group_size: Optional[int] = HQQ_DEFAULT_GROUP_SIZE,
) -> Optional[Dict[int, int]]:
    normalized_group_size = normalize_hqq_attention_group_size(group_size)
    manifest = load_hqq_attention_manifest(model_path, cache_profile, nbits, normalized_group_size)
    if not manifest:
        return None
    per_layer: Dict[int, int] = {}
    for entry in manifest.get("tensors", []):
        path = os.path.join(hqq_attention_cache_dir(model_path, cache_profile, nbits, normalized_group_size), entry["file"])
        if not os.path.isfile(path):
            return None
        per_layer[entry["layer_idx"]] = per_layer.get(entry["layer_idx"], 0) + _artifact_tensor_bytes(path)
    return per_layer


def hqq_attention_cache_total_bytes(
    model_path: str,
    cache_profile: Optional[str] = HQQ_CACHE_PROFILE_BASELINE,
    nbits: int = 4,
    group_size: Optional[int] = HQQ_DEFAULT_GROUP_SIZE,
) -> Optional[int]:
    per_layer = hqq_attention_cache_layer_bytes(model_path, cache_profile, nbits, group_size)
    if per_layer is None:
        return None
    return sum(per_layer.values())
