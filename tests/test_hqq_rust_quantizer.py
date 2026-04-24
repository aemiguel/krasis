import argparse
import copy
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path
from contextlib import contextmanager

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from krasis import GpuDecodeStore
import krasis.attention_backend as attention_backend_mod
from krasis.attention_backend import (
    HQQ_ATTENTION_CACHE_VERSION,
    HQQ_DEFAULT_GROUP_SIZE,
    HQQ_LAYOUT,
    _quantize_hqq4_group_current,
    _solve_hqq4_fixed_zero,
    compute_hqq4_rmse_rust,
    hqq_attention_tensor_path,
    load_hqq_attention_artifact,
    save_hqq_attention_manifest,
    validate_hqq_attention_tensors,
    quantize_hqq4_group_current_rust,
    quantize_hqq4_tensor,
    quantize_hqq4_tensor_rust,
    solve_hqq4_fixed_zero_rust,
    write_hqq_attention_artifact,
)
from krasis.config import ModelConfig
from krasis.model import KrasisModel
from krasis.weight_loader import WeightLoader
from tests.test_hqq_fused_branch_runtime import build_real_model_store
from tests.test_hqq_fused_branch_runtime import load_real_model_case
from tests.test_hqq_fused_branch_runtime import _real_model_quant_cfg


DEFAULT_BREADTH_MODEL_PATHS = {
    "qcn": os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next"),
    "q35": os.path.expanduser("~/.krasis/models/Qwen3.5-35B-A3B"),
    "q122": os.path.expanduser("~/.krasis/models/Qwen3.5-122B-A10B"),
    "q235": os.path.expanduser("~/.krasis/models/Qwen3-235B-A22B"),
    "qwen3-0.6b": os.path.expanduser("~/.krasis/models/Qwen3-0.6B"),
    "qwen3-8b": os.path.expanduser("~/.krasis/models/Qwen3-8B"),
}


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def _sequential_f32_sum(values: list[float]) -> tuple[float, tuple[int, float, float, float, float] | None]:
    acc = _f32(0.0)
    first_mismatch = None
    prefix_values: list[float] = []
    for idx, value in enumerate(values):
        value32 = _f32(value)
        new_acc = _f32(acc + value32)
        prefix_values.append(value32)
        torch_prefix = float(torch.tensor(prefix_values, dtype=torch.float32).sum().item())
        if first_mismatch is None and new_acc != torch_prefix:
            first_mismatch = (idx, acc, value32, new_acc, torch_prefix)
        acc = new_acc
    return acc, first_mismatch


def compare_rmse(name: str, chunk: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> None:
    py_rmse = torch.mean(
        (((q.to(torch.float32) - zero.unsqueeze(1)) * scale.unsqueeze(1)) - chunk) ** 2,
        dim=1,
    )
    rust_rmse = compute_hqq4_rmse_rust(chunk, q, scale, zero)
    print(
        {
            "case": name,
            "rmse_equal": torch.equal(py_rmse, rust_rmse),
            "rmse_max_abs_diff": float((py_rmse - rust_rmse).abs().max().item()),
        }
    )
    if not torch.equal(py_rmse, rust_rmse):
        raise SystemExit(f"Rust HQQ RMSE mismatch for {name}")


def compare_init(name: str, chunk: torch.Tensor) -> None:
    py_q, py_scale, py_zero = _quantize_hqq4_group_current(chunk, 15.0)
    rust_q, rust_scale, rust_zero = quantize_hqq4_group_current_rust(chunk)
    print(
        {
            "case": name,
            "init_q_equal": torch.equal(py_q, rust_q),
            "init_scale_equal": torch.equal(py_scale, rust_scale),
            "init_zero_equal": torch.equal(py_zero, rust_zero),
            "init_q_diff": int((py_q != rust_q).sum().item()),
            "init_scale_max_abs_diff": float((py_scale - rust_scale).abs().max().item()),
            "init_zero_max_abs_diff": float((py_zero - rust_zero).abs().max().item()),
        }
    )
    if not torch.equal(py_q, rust_q) or not torch.equal(py_scale, rust_scale) or not torch.equal(py_zero, rust_zero):
        raise SystemExit(f"Rust HQQ init mismatch for {name}")


def compare_solve(name: str, chunk: torch.Tensor, zero: torch.Tensor, scale_seed: torch.Tensor) -> None:
    py_q, py_scale, py_zero = _solve_hqq4_fixed_zero(chunk, 15.0, zero, scale_seed)
    rust_q, rust_scale, rust_zero = solve_hqq4_fixed_zero_rust(chunk, zero, scale_seed)
    print(
        {
            "case": name,
            "solve_q_equal": torch.equal(py_q, rust_q),
            "solve_scale_equal": torch.equal(py_scale, rust_scale),
            "solve_zero_equal": torch.equal(py_zero, rust_zero),
            "solve_q_diff": int((py_q != rust_q).sum().item()),
            "solve_scale_max_abs_diff": float((py_scale - rust_scale).abs().max().item()),
            "solve_zero_max_abs_diff": float((py_zero - rust_zero).abs().max().item()),
        }
    )
    if not torch.equal(py_q, rust_q) or not torch.equal(py_scale, rust_scale) or not torch.equal(py_zero, rust_zero):
        raise SystemExit(f"Rust HQQ solve mismatch for {name}")


def compare_quantized(name: str, weight: torch.Tensor) -> None:
    py = quantize_hqq4_tensor(weight)
    rust = quantize_hqq4_tensor_rust(weight)
    packed_equal = torch.equal(py["packed"], rust["packed"])
    scales_equal = torch.equal(py["scales"], rust["scales"])
    zeros_equal = torch.equal(py["zeros"], rust["zeros"])
    scales_max_abs_diff = float((py["scales"] - rust["scales"]).abs().max().item())
    print(
        {
            "case": name,
            "packed_equal": packed_equal,
            "scales_equal": scales_equal,
            "zeros_equal": zeros_equal,
            "packed_diff": int((py["packed"] != rust["packed"]).sum().item()),
            "scales_max_abs_diff": scales_max_abs_diff,
            "zeros_max_abs_diff": float((py["zeros"] - rust["zeros"]).abs().max().item()),
        }
    )
    if not packed_equal or not zeros_equal or scales_max_abs_diff > 1e-6:
        raise SystemExit(f"Rust HQQ quantizer mismatch for {name}")


def compare_real_case_tensors(
    name: str,
    real_case: dict,
    tensor_names: list[str],
    group_indices: list[int],
) -> None:
    tensor_map = {
        "q_proj": real_case["q_w"],
        "k_proj": real_case["k_w"],
        "v_proj": real_case["v_w"],
        "o_proj": real_case["o_w"],
        "fused_qkv": real_case["fused_w"],
    }
    print(
        {
            "case": name,
            "model_path": real_case["model_path"],
            "layer_idx": int(real_case["layer_idx"]),
            "tensor_names": tensor_names,
        }
    )
    for tensor_name in tensor_names:
        weight = tensor_map.get(tensor_name)
        if weight is None:
            raise SystemExit(f"Unsupported real tensor name for parity coverage: {tensor_name}")
        weight = weight.to(dtype=torch.float32)
        if group_indices:
            num_groups = weight.shape[1] // 128
            for group_idx in group_indices:
                if group_idx < 0 or group_idx >= num_groups:
                    raise SystemExit(
                        f"Requested group index {group_idx} out of range for {name}_{tensor_name} with {num_groups} groups"
                    )
                start = group_idx * 128
                end = start + 128
                compare_quantized(
                    f"{name}_{tensor_name}_group{group_idx}",
                    weight[:, start:end].contiguous(),
                )
        else:
            compare_quantized(
                f"{name}_{tensor_name}",
                weight,
            )


def _artifact_weights_from_real_case(real_case: dict) -> dict[str, torch.Tensor]:
    weights = {
        "q_proj": real_case["q_w"].cpu(),
        "k_proj": real_case["k_w"].cpu(),
        "v_proj": real_case["v_w"].cpu(),
        "o_proj": real_case["o_w"].cpu(),
    }
    q_norm = real_case["layer"].gqa_weights.get("q_norm")
    if q_norm is not None:
        weights["q_norm"] = q_norm.cpu()
    k_norm = real_case["layer"].gqa_weights.get("k_norm")
    if k_norm is not None:
        weights["k_norm"] = k_norm.cpu()
    return weights


def _dense_artifact_tensor_specs_from_real_case(
    real_case: dict,
    selected_tensor_names: list[str] | None = None,
) -> dict[str, tuple[str, torch.Tensor]]:
    layer = real_case["layer"]
    specs: dict[str, tuple[str, torch.Tensor]] = {}

    if layer.dense_mlp is not None:
        for tensor_name in ("gate_proj", "up_proj", "down_proj"):
            weight = layer.dense_mlp.get(tensor_name)
            if weight is None:
                continue
            specs[tensor_name] = (
                "dense_mlp",
                KrasisModel._dense_mlp_tensor(weight).cpu().contiguous(),
            )
        return specs

    shared_expert = getattr(layer, "shared_expert", None)
    if not shared_expert:
        return _routed_expert_tensor_specs_from_real_case(real_case, selected_tensor_names)

    gate_up = shared_expert.get("gate_up_proj")
    if gate_up is not None:
        specs["gate_up_proj"] = (
            "shared_expert",
            KrasisModel._dense_mlp_tensor(gate_up).cpu().contiguous(),
        )
        if isinstance(gate_up, tuple):
            gate_up_weight, gate_up_scale = gate_up
            mid_w = gate_up_weight.shape[0] // 2
            mid_s = gate_up_scale.shape[0] // 2
            gate_value = (gate_up_weight[:mid_w], gate_up_scale[:mid_s])
            up_value = (gate_up_weight[mid_w:], gate_up_scale[mid_s:])
        else:
            mid = gate_up.shape[0] // 2
            gate_value = gate_up[:mid]
            up_value = gate_up[mid:]
        specs["gate_proj"] = (
            "shared_expert",
            KrasisModel._dense_mlp_tensor(gate_value).cpu().contiguous(),
        )
        specs["up_proj"] = (
            "shared_expert",
            KrasisModel._dense_mlp_tensor(up_value).cpu().contiguous(),
        )

    down_proj = shared_expert.get("down_proj")
    if down_proj is not None:
        specs["down_proj"] = (
            "shared_expert",
            KrasisModel._dense_mlp_tensor(down_proj).cpu().contiguous(),
        )
    specs.update(_routed_expert_tensor_specs_from_real_case(real_case, selected_tensor_names))
    return specs


def _routed_expert_tensor_specs_from_real_case(
    real_case: dict,
    selected_tensor_names: list[str] | None = None,
) -> dict[str, tuple[str, torch.Tensor]]:
    cfg = real_case["cfg"]
    layer_idx = int(real_case["layer_idx"])
    prefix = f"{cfg.layers_prefix}.layers.{layer_idx}.mlp.experts"
    requested_expert_indices = sorted(
        {
            int(match.group(1))
            for tensor_name in (selected_tensor_names or [])
            if (
                match := re.match(
                    rf"^{re.escape(prefix)}\.(\d+)\.(gate_up_proj|gate_proj|up_proj|down_proj)$",
                    tensor_name,
                )
            )
        }
    )
    cache_key = tuple(requested_expert_indices)
    cache = real_case.setdefault("_routed_expert_tensor_specs_cache", {})
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    quant_cfg = _real_model_quant_cfg()
    loader = WeightLoader(cfg, quant_cfg)
    specs: dict[str, tuple[str, torch.Tensor]] = {}
    try:
        available_expert_indices = sorted(
            {
                int(key.split(".")[5])
                for key in loader._weight_map.keys()
                if key.startswith(prefix) and key.endswith(".gate_proj.weight")
            }
        )
        if not available_expert_indices:
            cache[cache_key] = specs
            return specs

        expert_indices = requested_expert_indices or [available_expert_indices[0]]
        for expert_idx in expert_indices:
            gate_name = f"{prefix}.{expert_idx}.gate_proj.weight"
            up_name = f"{prefix}.{expert_idx}.up_proj.weight"
            down_name = f"{prefix}.{expert_idx}.down_proj.weight"
            if gate_name not in loader._weight_map or up_name not in loader._weight_map or down_name not in loader._weight_map:
                continue

            gate = loader._load_bf16(gate_name, torch.device("cpu"))
            up = loader._load_bf16(up_name, torch.device("cpu"))
            down = loader._load_bf16(down_name, torch.device("cpu"))
            specs[f"{prefix}.{expert_idx}.gate_up_proj"] = (
                "routed_expert",
                KrasisModel._dense_mlp_tensor(torch.cat([gate, up], dim=0)).cpu().contiguous(),
            )
            specs[f"{prefix}.{expert_idx}.gate_proj"] = (
                "routed_expert",
                KrasisModel._dense_mlp_tensor(gate).cpu().contiguous(),
            )
            specs[f"{prefix}.{expert_idx}.up_proj"] = (
                "routed_expert",
                KrasisModel._dense_mlp_tensor(up).cpu().contiguous(),
            )
            specs[f"{prefix}.{expert_idx}.down_proj"] = (
                "routed_expert",
                KrasisModel._dense_mlp_tensor(down).cpu().contiguous(),
            )
    finally:
        loader.close()

    cache[cache_key] = specs
    return specs


def _build_shadow_real_artifact_model(real_case: dict, model_path: str) -> KrasisModel:
    model = KrasisModel(
        model_path=real_case["model_path"],
        num_gpus=1,
        gpu_prefill=False,
        krasis_threads=1,
        quant_cfg=_real_model_quant_cfg(),
        kv_dtype=torch.bfloat16,
    )
    model.cfg.model_path = model_path
    model.layers = [real_case["layer"]]
    model._hqq_attention_runtime = {}
    model._hqq_attention_runtime_nbits = None
    model._hqq_attention_cache_bytes = 0
    model._hqq_manifest = None
    model._hqq_rebuild = False
    return model


def _tensor_bytes(tensors: dict[str, torch.Tensor]) -> int:
    return sum(int(t.numel() * t.element_size()) for t in tensors.values())


def _write_hqq_attention_artifact_rust_shadow(
    model_path: str,
    layer_idx: int,
    layer_type: str,
    tensor_name: str,
    weight: torch.Tensor,
    nbits: int = 4,
) -> dict:
    path = hqq_attention_tensor_path(model_path, layer_idx, tensor_name, nbits=nbits)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensors = quantize_hqq4_tensor_rust(
        weight,
        group_size=HQQ_DEFAULT_GROUP_SIZE,
        collect_stats=True,
    )
    payload_tensors = {k: v for k, v in tensors.items() if isinstance(v, torch.Tensor)}
    structure = validate_hqq_attention_tensors(
        payload_tensors,
        expected_nbits=nbits,
        expected_layout=HQQ_LAYOUT,
    )
    metadata = {
        "backend": "hqq4",
        "format_version": str(HQQ_ATTENTION_CACHE_VERSION),
        "layer_idx": str(layer_idx),
        "layer_type": layer_type,
        "tensor_name": tensor_name,
        "layout": HQQ_LAYOUT,
        "dtype": str(weight.dtype).replace("torch.", ""),
    }
    save_file(payload_tensors, path, metadata=metadata)
    tensor_bytes = _tensor_bytes(payload_tensors)
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
        "original_shape": payload_tensors["orig_shape"].tolist(),
        "original_dtype": metadata["dtype"],
        "tensor_bytes": tensor_bytes,
        "structure": structure,
        "quality": tensors["stats"],
    }


def _write_real_artifact_layer_with_writer(
    model: KrasisModel,
    real_case: dict,
    *,
    writer,
    selected_tensor_names: list[str] | None = None,
    load_runtime: bool = True,
) -> None:
    weights = _artifact_weights_from_real_case(real_case)
    layer_idx = real_case["runtime_layer_idx"]
    layer_type = "full_attention"
    attention_tensor_map = model._hqq_attention_tensor_map(layer_type, weights)
    dense_tensor_specs = _dense_artifact_tensor_specs_from_real_case(
        real_case,
        selected_tensor_names=selected_tensor_names,
    )
    tensor_specs: dict[str, tuple[str, torch.Tensor]] = {
        tensor_name: (layer_type, tensor)
        for tensor_name, tensor in attention_tensor_map.items()
    }
    if selected_tensor_names is not None:
        tensor_specs = {}
        for tensor_name in selected_tensor_names:
            if tensor_name in attention_tensor_map:
                tensor_specs[tensor_name] = (layer_type, attention_tensor_map[tensor_name])
            elif tensor_name in dense_tensor_specs:
                tensor_specs[tensor_name] = dense_tensor_specs[tensor_name]
    fused_qkv = None
    if "fused_qkv" not in tensor_specs and (selected_tensor_names is None or "fused_qkv" in selected_tensor_names):
        fused_qkv = model._build_hqq_fused_qkv_artifact_weight(layer_type, weights)
    requested_tensor_names = selected_tensor_names or sorted(tensor_specs.keys())
    missing_tensor_names = [
        tensor_name
        for tensor_name in requested_tensor_names
        if tensor_name not in tensor_specs and not (tensor_name == "fused_qkv" and fused_qkv is not None)
    ]
    if missing_tensor_names:
        raise SystemExit(
            "Rust HQQ registration harness missing requested tensor(s) for "
            f"layer {layer_idx}: {missing_tensor_names!r}"
        )
    for tensor_name, (tensor_layer_type, tensor) in tensor_specs.items():
        record = writer(
            model.cfg.model_path,
            layer_idx=layer_idx,
            layer_type=tensor_layer_type,
            tensor_name=tensor_name,
            weight=tensor,
            nbits=4,
        )
        model._hqq_manifest["tensors"].append(record)
        model._hqq_manifest["totals"]["tensor_bytes"] += int(record["tensor_bytes"])
        model._hqq_manifest["totals"]["num_tensors"] += 1
    if fused_qkv is not None:
        record = writer(
            model.cfg.model_path,
            layer_idx=layer_idx,
            layer_type=layer_type,
            tensor_name="fused_qkv",
            weight=fused_qkv,
            nbits=4,
        )
        model._hqq_manifest["tensors"].append(record)
        model._hqq_manifest["totals"]["tensor_bytes"] += int(record["tensor_bytes"])
        model._hqq_manifest["totals"]["num_tensors"] += 1
    save_hqq_attention_manifest(model.cfg.model_path, model._hqq_manifest)
    if load_runtime:
        model._validate_hqq_attention_cache()
        model._load_hqq_attention_runtime_state()


def _write_hqq_attention_artifact_instrumented(
    case: str,
    variant: str,
    *,
    model_path: str,
    layer_idx: int,
    layer_type: str,
    tensor_name: str,
    weight: torch.Tensor,
    use_rust_shadow: bool,
    nbits: int = 4,
) -> tuple[dict, dict]:
    path = hqq_attention_tensor_path(model_path, layer_idx, tensor_name, nbits=nbits)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tensor_start = _stage_start(case, variant, "quantizer_tensor_total")

    prep_start = _stage_start(case, variant, "quantizer_tensor_prep")
    quantizer = quantize_hqq4_tensor_rust if use_rust_shadow else quantize_hqq4_tensor
    timing_context = {
        "case": case,
        "variant": variant,
        "tensor_name": tensor_name,
    }
    _stage_done(
        case,
        variant,
        "quantizer_tensor_prep",
        prep_start,
        tensor_name=tensor_name,
    )

    quantizer_start = _stage_start(case, variant, "quantizer_invocation")
    with _python_quantizer_timing(case, variant, tensor_name) as py_quantizer_stats:
        if use_rust_shadow:
            tensors = quantizer(
                weight,
                group_size=HQQ_DEFAULT_GROUP_SIZE,
                collect_stats=True,
            )
        else:
            tensors = quantizer(
                weight,
                group_size=HQQ_DEFAULT_GROUP_SIZE,
                collect_stats=True,
                timing_context=timing_context,
            )
    quantizer_elapsed = time.perf_counter() - quantizer_start
    _stage_done(
        case,
        variant,
        "quantizer_invocation",
        quantizer_start,
        tensor_name=tensor_name,
        rows=int(weight.shape[0]),
        cols=int(weight.shape[1]),
    )
    quantizer_summary = {
        "quantizer_invocation": float(quantizer_elapsed),
        "fixed_zero_solve_update": None,
        "elementwise_quantize_clamp": None,
        "reductions": None,
        "scale_zero_update_math": None,
        "allocation_setup_materialization": None,
        "unresolved_control_overhead": None,
        "measurement_limit_note": None,
    }
    if py_quantizer_stats is not None:
        orchestration_s = (
            py_quantizer_stats["zero_grid_setup_s"]
            + py_quantizer_stats["candidate_loop_overhead_s"]
            + py_quantizer_stats["score_reconstruct_s"]
            + py_quantizer_stats["best_fit_update_s"]
        )
        post_group_combine_s = (
            py_quantizer_stats["quantize_pack_s"] - py_quantizer_stats["group_loop_total_s"]
        )
        wrapper_overhead_s = (
            quantizer_elapsed
            - py_quantizer_stats["normalize_s"]
            - py_quantizer_stats["reshape_s"]
            - py_quantizer_stats["quantize_pack_s"]
        )
        local_pass_exposed = False
        solve_remaining_overhead_s = max(
            0.0,
            py_quantizer_stats["solve_fixed_zero_s"]
            - py_quantizer_stats["solve_alloc_setup_s"]
            - py_quantizer_stats["solve_iter_quantize_clamp_s"]
            - py_quantizer_stats["solve_iter_reduction_s"]
            - py_quantizer_stats["solve_iter_update_s"]
            - py_quantizer_stats["solve_final_quantize_s"],
        )
        elementwise_quantize_clamp_work_s = (
            py_quantizer_stats["solve_iter_quantize_clamp_s"]
            + py_quantizer_stats["solve_final_quantize_s"]
        )
        reductions_sum_accumulate_work_s = py_quantizer_stats["solve_iter_reduction_s"]
        allocation_setup_materialization_s = py_quantizer_stats["solve_alloc_setup_s"]
        remaining_control_loop_overhead_s = solve_remaining_overhead_s
        residual_solve_overhead_s = max(
            0.0,
            py_quantizer_stats["solve_fixed_zero_s"]
            - elementwise_quantize_clamp_work_s
            - reductions_sum_accumulate_work_s
            - py_quantizer_stats["solve_iter_update_s"]
            - allocation_setup_materialization_s
            - remaining_control_loop_overhead_s,
        )
        quantizer_summary.update(
            {
                "fixed_zero_solve_update": float(py_quantizer_stats["solve_fixed_zero_s"]),
                "elementwise_quantize_clamp": float(elementwise_quantize_clamp_work_s),
                "reductions": float(reductions_sum_accumulate_work_s),
                "scale_zero_update_math": float(py_quantizer_stats["solve_iter_update_s"]),
                "allocation_setup_materialization": float(allocation_setup_materialization_s),
                "unresolved_control_overhead": float(remaining_control_loop_overhead_s),
                "measurement_limit_note": "current product timing isolates solve_alloc_setup_s, but does not split the remaining untimed solve remainder into finer materialization vs control buckets",
            }
        )
        print(
            {
                "case": case,
                "variant": variant,
                "stage": "quantizer_group_loop_total",
                "tensor_name": tensor_name,
                "elapsed_s": float(py_quantizer_stats["group_loop_total_s"]),
                "group_count": int(py_quantizer_stats["group_count"]),
                "kernel_total_s": float(py_quantizer_stats["kernel_total_s"]),
                "normalize_s": float(py_quantizer_stats["normalize_s"]),
                "reshape_s": float(py_quantizer_stats["reshape_s"]),
                "quantize_pack_s": float(py_quantizer_stats["quantize_pack_s"]),
            }
        )
        print(
            {
                "case": case,
                "variant": variant,
                "stage": "quantizer_breakdown",
                "tensor_name": tensor_name,
                "fused_tensor_prep_s": 0.0,
                "python_group_orchestration_s": float(orchestration_s),
                "quantizer_core_call_s": float(py_quantizer_stats["solve_fixed_zero_s"]),
                "post_group_combine_s": float(post_group_combine_s),
                "normalize_s": float(py_quantizer_stats["normalize_s"]),
                "reshape_s": float(py_quantizer_stats["reshape_s"]),
                "wrapper_overhead_s": float(wrapper_overhead_s),
            }
        )
        print(
            {
                "case": case,
                "variant": variant,
                "stage": "quantizer_core_breakdown",
                "tensor_name": tensor_name,
                "fixed_zero_solve_update_s": float(py_quantizer_stats["solve_fixed_zero_s"]),
                "iterative_quantize_clamp_s": float(py_quantizer_stats["solve_iter_quantize_clamp_s"]),
                "iterative_reduction_s": float(py_quantizer_stats["solve_iter_reduction_s"]),
                "scale_zero_update_math_s": float(py_quantizer_stats["solve_iter_update_s"]),
                "candidate_loop_bookkeeping_s": float(
                    allocation_setup_materialization_s + remaining_control_loop_overhead_s
                ),
                "final_quantize_s": float(py_quantizer_stats["solve_final_quantize_s"]),
                "rmse_error_s": float(py_quantizer_stats["score_reconstruct_s"]),
                "best_state_retention_s": float(py_quantizer_stats["best_fit_update_s"]),
                "remaining_core_overhead_s": float(
                    py_quantizer_stats["zero_grid_setup_s"] + py_quantizer_stats["candidate_loop_overhead_s"]
                ),
                "solve_alloc_setup_s": float(py_quantizer_stats["solve_alloc_setup_s"]),
                "solve_iter_quantize_clamp_s": float(py_quantizer_stats["solve_iter_quantize_clamp_s"]),
                "solve_iter_reduction_s": float(py_quantizer_stats["solve_iter_reduction_s"]),
                "solve_iter_update_s": float(py_quantizer_stats["solve_iter_update_s"]),
                "solve_final_quantize_s": float(py_quantizer_stats["solve_final_quantize_s"]),
                "solve_remaining_overhead_s": float(solve_remaining_overhead_s),
                "local_pass_work_s": None,
                "local_pass_exposed": local_pass_exposed,
                "local_pass_note": "current product timing hook does not isolate local-pass time separately",
                "candidate_count": int(py_quantizer_stats["candidate_count"]),
            }
        )
        print(
            {
                "case": case,
                "variant": variant,
                "stage": "quantizer_solve_hotpath_breakdown",
                "tensor_name": tensor_name,
                "fixed_zero_solve_update_s": quantizer_summary["fixed_zero_solve_update"],
                "elementwise_quantize_clamp_work_s": quantizer_summary["elementwise_quantize_clamp"],
                "reductions_sum_accumulate_work_s": quantizer_summary["reductions"],
                "scale_zero_update_math_s": quantizer_summary["scale_zero_update_math"],
                "allocation_setup_materialization_s": quantizer_summary["allocation_setup_materialization"],
                "remaining_control_loop_overhead_s": quantizer_summary["unresolved_control_overhead"],
                "residual_solve_overhead_s": float(residual_solve_overhead_s),
                "quantize_clamp_note": "iterative plus final ((chunk / scale) + zero).round().clamp_()",
                "reduction_note": "includes centered materialization plus denom/numer sum reductions",
                "measurement_limit_note": quantizer_summary["measurement_limit_note"],
            }
        )
        for range_entry in py_quantizer_stats["group_ranges"]:
            print(
                {
                    "case": case,
                    "variant": variant,
                    "stage": "quantizer_group_range",
                    "tensor_name": tensor_name,
                    **range_entry,
                }
            )
    elif variant == "rust_shadow":
        quantizer_summary["measurement_limit_note"] = "current product timing does not expose Rust-shadow inner solve buckets; only outer quantizer_invocation is measured on this path"
        print(
            {
                "case": case,
                "variant": variant,
                "stage": "quantizer_breakdown",
                "tensor_name": tensor_name,
                "fused_tensor_prep_s": 0.0,
                "python_group_orchestration_s": 0.0,
                "quantizer_core_call_s": float(quantizer_elapsed),
                "post_group_combine_s": 0.0,
                "normalize_s": 0.0,
                "reshape_s": 0.0,
                "wrapper_overhead_s": 0.0,
            }
        )
    _stage_done(
        case,
        variant,
        "quantizer_tensor_total",
        tensor_start,
        tensor_name=tensor_name,
        rows=int(weight.shape[0]),
        cols=int(weight.shape[1]),
    )

    assembly_start = _stage_start(case, variant, "packed_scales_zeros_assembly")
    payload_tensors = {k: v for k, v in tensors.items() if isinstance(v, torch.Tensor)}
    payload_tensor_names = sorted(payload_tensors.keys())
    _stage_done(
        case,
        variant,
        "packed_scales_zeros_assembly",
        assembly_start,
        tensor_name=tensor_name,
        payload_tensor_names=payload_tensor_names,
        packed_shape=list(payload_tensors["packed"].shape),
        scales_shape=list(payload_tensors["scales"].shape),
        zeros_shape=list(payload_tensors["zeros"].shape),
    )

    payload_start = _stage_start(case, variant, "safetensors_payload_assembly")
    structure = validate_hqq_attention_tensors(
        payload_tensors,
        expected_nbits=nbits,
        expected_layout=HQQ_LAYOUT,
    )
    tensor_bytes = _tensor_bytes(payload_tensors)
    _stage_done(
        case,
        variant,
        "safetensors_payload_assembly",
        payload_start,
        tensor_name=tensor_name,
        tensor_bytes=int(tensor_bytes),
        groups=int(structure["groups"]),
    )

    metadata_start = _stage_start(case, variant, "metadata_header_assembly")
    metadata = {
        "backend": "hqq4",
        "format_version": str(HQQ_ATTENTION_CACHE_VERSION),
        "layer_idx": str(layer_idx),
        "layer_type": layer_type,
        "tensor_name": tensor_name,
        "layout": HQQ_LAYOUT,
        "dtype": str(weight.dtype).replace("torch.", ""),
    }
    _stage_done(
        case,
        variant,
        "metadata_header_assembly",
        metadata_start,
        tensor_name=tensor_name,
        metadata_keys=sorted(metadata.keys()),
    )

    file_start = _stage_start(case, variant, "file_write_flush")
    save_file(payload_tensors, path, metadata=metadata)
    _stage_done(
        case,
        variant,
        "file_write_flush",
        file_start,
        tensor_name=tensor_name,
        file=os.path.basename(path),
        tensor_bytes=int(tensor_bytes),
    )

    quantized_output_summary = {
        "tensor_name": tensor_name,
        "group_size": int(tensors["group_size"][0].item()),
        "axis": int(tensors["axis"][0].item()),
        "nbits": int(tensors["nbits"][0].item()),
        "layout": HQQ_LAYOUT,
        "orig_shape": tuple(int(v) for v in payload_tensors["orig_shape"].tolist()),
        "packed_dtype": str(payload_tensors["packed"].dtype).replace("torch.", ""),
        "scales_dtype": str(payload_tensors["scales"].dtype).replace("torch.", ""),
        "zeros_dtype": str(payload_tensors["zeros"].dtype).replace("torch.", ""),
        "packed_shape": tuple(int(v) for v in payload_tensors["packed"].shape),
        "scales_shape": tuple(int(v) for v in payload_tensors["scales"].shape),
        "zeros_shape": tuple(int(v) for v in payload_tensors["zeros"].shape),
        "tensor_bytes": int(tensor_bytes),
        "original_dtype": metadata["dtype"],
        "structure": copy.deepcopy(structure),
        "packed": payload_tensors["packed"].clone(),
        "scales": payload_tensors["scales"].clone(),
        "zeros": payload_tensors["zeros"].clone(),
    }

    return ({
        "layer_idx": layer_idx,
        "layer_type": layer_type,
        "tensor_name": tensor_name,
        "file": os.path.basename(path),
        "path": path,
        "nbits": nbits,
        "group_size": int(tensors["group_size"][0].item()),
        "axis": int(tensors["axis"][0].item()),
        "layout": HQQ_LAYOUT,
        "original_shape": payload_tensors["orig_shape"].tolist(),
        "original_dtype": metadata["dtype"],
        "tensor_bytes": tensor_bytes,
        "structure": structure,
        "quality": tensors["stats"],
    }, quantizer_summary, quantized_output_summary)


@contextmanager
def _python_quantizer_timing(case: str, variant: str, tensor_name: str):
    if variant != "python":
        yield None
        return

    previous_env = os.environ.get("KRASIS_HQQ_REAL_MODEL_TIMING")
    os.environ["KRASIS_HQQ_REAL_MODEL_TIMING"] = "1"
    original_emit = attention_backend_mod._emit_real_model_timing
    stats = {
        "normalize_s": 0.0,
        "reshape_s": 0.0,
        "quantize_pack_s": 0.0,
        "kernel_total_s": 0.0,
        "group_loop_total_s": 0.0,
        "group_count": 0,
        "group_ranges": [],
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
        "_range_elapsed_s": 0.0,
        "_range_start_group": None,
    }

    def _flush_range(end_group_idx: int) -> None:
        start_group_idx = stats["_range_start_group"]
        if start_group_idx is None:
            return
        stats["group_ranges"].append(
            {
                "group_start": int(start_group_idx),
                "group_end": int(end_group_idx),
                "elapsed_s": float(stats["_range_elapsed_s"]),
            }
        )
        stats["_range_start_group"] = None
        stats["_range_elapsed_s"] = 0.0

    def _instrumented_emit(payload: dict) -> None:
        if (
            payload.get("case") == case
            and payload.get("variant") == variant
            and payload.get("tensor_name") == tensor_name
            and payload.get("phase") == "artifact_pack_stage"
        ):
            stage = payload.get("stage")
            elapsed = float(payload.get("elapsed_s", 0.0))
            if stage == "dtype_device_normalize":
                stats["normalize_s"] += elapsed
            elif stage == "reshape_contiguous_copy":
                stats["reshape_s"] += elapsed
            elif stage == "quantize_pack":
                stats["quantize_pack_s"] += elapsed
            elif stage == "quantize_pack_core":
                stats["kernel_total_s"] += elapsed
                stats["zero_grid_setup_s"] += float(payload.get("zero_grid_setup_s", 0.0))
                stats["candidate_loop_overhead_s"] += float(payload.get("candidate_loop_overhead_s", 0.0))
                stats["solve_fixed_zero_s"] += float(payload.get("solve_fixed_zero_s", 0.0))
                stats["solve_alloc_setup_s"] += float(payload.get("solve_alloc_setup_s", 0.0))
                stats["solve_iter_quantize_clamp_s"] += float(payload.get("solve_iter_quantize_clamp_s", 0.0))
                stats["solve_iter_reduction_s"] += float(payload.get("solve_iter_reduction_s", 0.0))
                stats["solve_iter_update_s"] += float(payload.get("solve_iter_update_s", 0.0))
                stats["solve_final_quantize_s"] += float(payload.get("solve_final_quantize_s", 0.0))
                stats["score_reconstruct_s"] += float(payload.get("score_reconstruct_s", 0.0))
                stats["best_fit_update_s"] += float(payload.get("best_fit_update_s", 0.0))
                stats["candidate_count"] += int(payload.get("candidate_count", 0))
            elif stage == "quantize_pack_core_group":
                group_idx = int(payload["group_idx"])
                stats["group_count"] += 1
                stats["group_loop_total_s"] += elapsed
                if stats["_range_start_group"] is None:
                    stats["_range_start_group"] = group_idx
                stats["_range_elapsed_s"] += elapsed
                if (group_idx - stats["_range_start_group"] + 1) >= 16:
                    _flush_range(group_idx)
        original_emit(payload)

    attention_backend_mod._emit_real_model_timing = _instrumented_emit
    try:
        yield stats
    finally:
        _flush_range(int(stats["group_count"]) - 1)
        attention_backend_mod._emit_real_model_timing = original_emit
        if previous_env is None:
            os.environ.pop("KRASIS_HQQ_REAL_MODEL_TIMING", None)
        else:
            os.environ["KRASIS_HQQ_REAL_MODEL_TIMING"] = previous_env


def _write_real_artifact_layer_instrumented(
    case: str,
    variant: str,
    model: KrasisModel,
    real_case: dict,
    *,
    use_rust_shadow: bool,
    selected_tensor_names: list[str] | None = None,
    load_runtime: bool = False,
) -> dict[str, dict]:
    quantizer_summaries: dict[str, dict] = {}
    quantized_outputs: dict[str, dict] = {}
    traversal_start = _stage_start(case, variant, "tensor_traversal_discovery")
    attention_weights = _artifact_weights_from_real_case(real_case)
    dense_tensor_specs = _dense_artifact_tensor_specs_from_real_case(
        real_case,
        selected_tensor_names=selected_tensor_names,
    )
    layer_idx = real_case["runtime_layer_idx"]
    layer_type = "full_attention"
    tensor_map_start = time.perf_counter()
    attention_tensor_map = model._hqq_attention_tensor_map(layer_type, attention_weights)
    tensor_map_elapsed = time.perf_counter() - tensor_map_start
    tensor_specs: dict[str, tuple[str, torch.Tensor]] = {
        tensor_name: (layer_type, tensor)
        for tensor_name, tensor in attention_tensor_map.items()
    }
    if selected_tensor_names is not None:
        tensor_specs = {}
        for tensor_name in selected_tensor_names:
            if tensor_name in attention_tensor_map:
                tensor_specs[tensor_name] = (layer_type, attention_tensor_map[tensor_name])
            elif tensor_name in dense_tensor_specs:
                tensor_specs[tensor_name] = dense_tensor_specs[tensor_name]
    fused_qkv = None
    fused_prep_elapsed = 0.0
    if "fused_qkv" not in tensor_specs and (selected_tensor_names is None or "fused_qkv" in selected_tensor_names):
        fused_prep_start = time.perf_counter()
        fused_qkv = model._build_hqq_fused_qkv_artifact_weight(layer_type, attention_weights)
        fused_prep_elapsed = time.perf_counter() - fused_prep_start
    requested_tensor_names = selected_tensor_names or sorted(tensor_specs.keys())
    missing_tensor_names = [
        tensor_name
        for tensor_name in requested_tensor_names
        if tensor_name not in tensor_specs and not (tensor_name == "fused_qkv" and fused_qkv is not None)
    ]
    if missing_tensor_names:
        raise SystemExit(
            f"Rust HQQ dense artifact-write harness missing requested tensor(s) for {case}: "
            f"{missing_tensor_names!r}"
        )
    _stage_done(
        case,
        variant,
        "tensor_traversal_discovery",
        traversal_start,
        layer_idx=int(layer_idx),
        decode_tensors=sorted(tensor_specs.keys()),
        has_fused_qkv=bool(fused_qkv is not None),
        tensor_map_build_s=float(tensor_map_elapsed),
        fused_tensor_prep_s=float(fused_prep_elapsed),
        dense_tensor_names=sorted(dense_tensor_specs.keys()),
    )

    decode_start = _stage_start(case, variant, "decode_format_build")
    for tensor_name, (tensor_layer_type, tensor) in tensor_specs.items():
        record, quantizer_summary, quantized_output_summary = _write_hqq_attention_artifact_instrumented(
            case,
            variant,
            model_path=model.cfg.model_path,
            layer_idx=layer_idx,
            layer_type=tensor_layer_type,
            tensor_name=tensor_name,
            weight=tensor,
            use_rust_shadow=use_rust_shadow,
            nbits=4,
        )
        quantizer_summaries[tensor_name] = quantizer_summary
        quantized_outputs[tensor_name] = quantized_output_summary
        model._hqq_manifest["tensors"].append(record)
        model._hqq_manifest["totals"]["tensor_bytes"] += int(record["tensor_bytes"])
        model._hqq_manifest["totals"]["num_tensors"] += 1
    _stage_done(
        case,
        variant,
        "decode_format_build",
        decode_start,
        tensor_count=len(tensor_specs),
        tensor_names=sorted(tensor_specs.keys()),
    )

    prefill_start = _stage_start(case, variant, "prefill_format_build")
    if fused_qkv is not None:
        record, quantizer_summary, quantized_output_summary = _write_hqq_attention_artifact_instrumented(
            case,
            variant,
            model_path=model.cfg.model_path,
            layer_idx=layer_idx,
            layer_type=layer_type,
            tensor_name="fused_qkv",
            weight=fused_qkv,
            use_rust_shadow=use_rust_shadow,
            nbits=4,
        )
        quantizer_summaries["fused_qkv"] = quantizer_summary
        quantized_outputs["fused_qkv"] = quantized_output_summary
        model._hqq_manifest["tensors"].append(record)
        model._hqq_manifest["totals"]["tensor_bytes"] += int(record["tensor_bytes"])
        model._hqq_manifest["totals"]["num_tensors"] += 1
    _stage_done(
        case,
        variant,
        "prefill_format_build",
        prefill_start,
        built=bool(fused_qkv is not None),
        tensor_name="fused_qkv" if fused_qkv is not None else None,
    )

    manifest_start = _stage_start(case, variant, "metadata_header_assembly")
    save_hqq_attention_manifest(model.cfg.model_path, model._hqq_manifest)
    _stage_done(
        case,
        variant,
        "metadata_header_assembly",
        manifest_start,
        manifest_entries=len(model._hqq_manifest["tensors"]),
    )

    if load_runtime:
        model._validate_hqq_attention_cache()
        model._load_hqq_attention_runtime_state()
    return {
        "quantizer_summaries": quantizer_summaries,
        "quantized_outputs": quantized_outputs,
    }


def _normalized_manifest_entry(entry: dict) -> dict:
    normalized = dict(entry)
    normalized["path"] = os.path.basename(normalized["path"])
    return normalized


def _normalized_runtime_entry(entry: dict) -> dict:
    normalized = dict(entry)
    normalized["path"] = os.path.basename(normalized["path"])
    normalized.pop("packed", None)
    normalized.pop("scales", None)
    normalized.pop("zeros", None)
    return normalized


def _normalized_quantized_output_summary(entry: dict) -> dict:
    normalized = dict(entry)
    normalized.pop("packed", None)
    normalized.pop("scales", None)
    normalized.pop("zeros", None)
    return normalized


def _runtime_entry_from_artifact(entry: dict, artifact: dict) -> dict:
    tensors = artifact["tensors"]
    return {
        "backend": "hqq4",
        "format_version": int(HQQ_ATTENTION_CACHE_VERSION),
        "nbits": int(tensors["nbits"][0].item()),
        "layout": entry["layout"],
        "group_size": int(tensors["group_size"][0].item()),
        "axis": int(tensors["axis"][0].item()),
        "orig_shape": tuple(int(v) for v in tensors["orig_shape"].tolist()),
        "packed": tensors["packed"],
        "scales": tensors["scales"],
        "zeros": tensors["zeros"],
        "packed_dtype": str(tensors["packed"].dtype).replace("torch.", ""),
        "scales_dtype": str(tensors["scales"].dtype).replace("torch.", ""),
        "zeros_dtype": str(tensors["zeros"].dtype).replace("torch.", ""),
        "original_dtype": entry.get("original_dtype", artifact["metadata"].get("dtype", "")),
        "path": artifact["path"],
        "tensor_bytes": artifact["tensor_bytes"],
    }


def _normalized_staging_json(store: GpuDecodeStore) -> dict:
    def strip_volatile_timing_fields(value):
        if isinstance(value, dict):
            normalized = {}
            for key, child in value.items():
                if key.endswith("_ms"):
                    continue
                normalized[key] = strip_volatile_timing_fields(child)
            return normalized
        if isinstance(value, list):
            return [strip_volatile_timing_fields(child) for child in value]
        return value

    payload = json.loads(store.hqq_runtime_staging_json())
    for slot in payload.get("slots", []):
        for key in ("packed_slot_ptr", "scales_slot_ptr", "zeros_slot_ptr"):
            slot.pop(key, None)
    return strip_volatile_timing_fields(payload)


def _stage_start(case: str, variant: str, stage: str) -> float:
    print({"case": case, "variant": variant, "stage": stage, "event": "start"})
    return time.perf_counter()


def _stage_done(case: str, variant: str, stage: str, start: float, **extra) -> None:
    payload = {
        "case": case,
        "variant": variant,
        "stage": stage,
        "event": "done",
        "elapsed_s": time.perf_counter() - start,
    }
    payload.update(extra)
    print(payload)


def _first_nested_diff(left, right, path: str = "") -> tuple[str, object, object] | None:
    if type(left) is not type(right):
        return (path or "<root>", type(left).__name__, type(right).__name__)
    if isinstance(left, dict):
        left_keys = sorted(left.keys())
        right_keys = sorted(right.keys())
        if left_keys != right_keys:
            return (f"{path}.keys" if path else "keys", left_keys, right_keys)
        for key in left_keys:
            child = _first_nested_diff(left[key], right[key], f"{path}.{key}" if path else str(key))
            if child is not None:
                return child
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return (f"{path}.len" if path else "len", len(left), len(right))
        for idx, (l_item, r_item) in enumerate(zip(left, right)):
            child = _first_nested_diff(l_item, r_item, f"{path}[{idx}]")
            if child is not None:
                return child
        return None
    if left != right:
        return (path or "<root>", left, right)
    return None


def _first_runtime_mismatch(py_runtime: dict, rust_runtime: dict) -> tuple[str, object, object] | None:
    py_names = sorted(py_runtime.keys())
    rust_names = sorted(rust_runtime.keys())
    if py_names != rust_names:
        return ("runtime.tensor_names", py_names, rust_names)
    for tensor_name in py_names:
        py_desc = py_runtime[tensor_name]
        rust_desc = rust_runtime[tensor_name]
        desc_diff = _first_nested_diff(
            _normalized_runtime_entry(py_desc),
            _normalized_runtime_entry(rust_desc),
            f"{tensor_name}.descriptor",
        )
        if desc_diff is not None:
            return desc_diff
        for field in ("packed", "scales", "zeros"):
            if not torch.equal(py_desc[field], rust_desc[field]):
                diff_idx = (py_desc[field] != rust_desc[field]).nonzero(as_tuple=False)
                first_idx = diff_idx[0].tolist() if diff_idx.numel() else []
                py_value = py_desc[field][tuple(first_idx)].item() if first_idx else None
                rust_value = rust_desc[field][tuple(first_idx)].item() if first_idx else None
                return (f"{tensor_name}.{field}[{first_idx}]", py_value, rust_value)
    return None


def _first_quantized_output_mismatch(py_output: dict, rust_output: dict) -> tuple[str, object, object] | None:
    summary_diff = _first_nested_diff(
        _normalized_quantized_output_summary(py_output),
        _normalized_quantized_output_summary(rust_output),
        "quantized_output.summary",
    )
    if summary_diff is not None:
        return summary_diff
    for field in ("packed", "scales", "zeros"):
        py_tensor = py_output[field]
        rust_tensor = rust_output[field]
        if not torch.equal(py_tensor, rust_tensor):
            diff_idx = (py_tensor != rust_tensor).nonzero(as_tuple=False)
            first_idx = diff_idx[0].tolist() if diff_idx.numel() else []
            py_value = py_tensor[tuple(first_idx)].item() if first_idx else None
            rust_value = rust_tensor[tuple(first_idx)].item() if first_idx else None
            return (f"quantized_output.{field}[{first_idx}]", py_value, rust_value)
    return None


def compare_real_artifact_layer(name: str, real_case: dict) -> None:
    py_tmpdir = tempfile.mkdtemp(prefix="hqq-python-artifact-shadow-")
    rust_tmpdir = tempfile.mkdtemp(prefix="hqq-rust-artifact-shadow-")
    py_model_path = os.path.join(py_tmpdir, os.path.basename(real_case["model_path"]))
    rust_model_path = os.path.join(rust_tmpdir, os.path.basename(real_case["model_path"]))
    os.makedirs(py_model_path, exist_ok=True)
    os.makedirs(rust_model_path, exist_ok=True)

    py_model = _build_shadow_real_artifact_model(real_case, py_model_path)
    rust_model = _build_shadow_real_artifact_model(real_case, rust_model_path)
    old_home = os.environ.get("HOME")
    try:
        os.environ["HOME"] = py_tmpdir
        py_model._prepare_hqq_attention_cache()
        _write_real_artifact_layer_with_writer(
            py_model,
            real_case,
            writer=write_hqq_attention_artifact,
            load_runtime=False,
        )

        os.environ["HOME"] = rust_tmpdir
        rust_model._prepare_hqq_attention_cache()
        _write_real_artifact_layer_with_writer(
            rust_model,
            real_case,
            writer=_write_hqq_attention_artifact_rust_shadow,
            load_runtime=False,
        )

        py_entries = sorted(
            (_normalized_manifest_entry(entry) for entry in py_model._hqq_manifest["tensors"]),
            key=lambda item: (item["layer_idx"], item["tensor_name"]),
        )
        rust_entries = sorted(
            (_normalized_manifest_entry(entry) for entry in rust_model._hqq_manifest["tensors"]),
            key=lambda item: (item["layer_idx"], item["tensor_name"]),
        )
        if py_entries != rust_entries:
            raise SystemExit(f"Rust HQQ artifact manifest mismatch for {name}")

        for py_entry, rust_entry in zip(py_entries, rust_entries):
            py_artifact = load_hqq_attention_artifact(
                py_model.cfg.model_path,
                py_entry,
                expected_nbits=4,
                device="cpu",
            )
            rust_artifact = load_hqq_attention_artifact(
                rust_model.cfg.model_path,
                rust_entry,
                expected_nbits=4,
                device="cpu",
            )
            if py_artifact["metadata"] != rust_artifact["metadata"]:
                raise SystemExit(
                    f"Rust HQQ artifact metadata mismatch for {name} {py_entry['tensor_name']}"
                )
            if py_artifact["structure"] != rust_artifact["structure"]:
                raise SystemExit(
                    f"Rust HQQ artifact structure mismatch for {name} {py_entry['tensor_name']}"
                )
            for tensor_name in sorted(py_artifact["tensors"].keys()):
                if not torch.equal(
                    py_artifact["tensors"][tensor_name],
                    rust_artifact["tensors"][tensor_name],
                ):
                    raise SystemExit(
                        f"Rust HQQ artifact tensor mismatch for {name} "
                        f"{py_entry['tensor_name']} payload tensor {tensor_name}"
                    )

        py_runtime = py_model._hqq_attention_runtime[real_case["runtime_layer_idx"]]
        rust_runtime = rust_model._hqq_attention_runtime[real_case["runtime_layer_idx"]]
        runtime_tensor_names = sorted(py_runtime.keys())
        if runtime_tensor_names != sorted(rust_runtime.keys()):
            raise SystemExit(f"Rust HQQ runtime tensor-name mismatch for {name}")
        for tensor_name in runtime_tensor_names:
            py_desc = py_runtime[tensor_name]
            rust_desc = rust_runtime[tensor_name]
            if _normalized_runtime_entry(py_desc) != _normalized_runtime_entry(rust_desc):
                raise SystemExit(f"Rust HQQ runtime descriptor mismatch for {name} {tensor_name}")
            for field in ("packed", "scales", "zeros"):
                if not torch.equal(py_desc[field], rust_desc[field]):
                    raise SystemExit(f"Rust HQQ runtime tensor mismatch for {name} {tensor_name} {field}")

        print(
            {
                "case": name,
                "manifest_entries_equal": True,
                "runtime_descriptors_equal": True,
                "runtime_tensor_names": runtime_tensor_names,
                "layer_idx": int(real_case["runtime_layer_idx"]),
            }
        )
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        shutil.rmtree(py_tmpdir, ignore_errors=True)
        shutil.rmtree(rust_tmpdir, ignore_errors=True)


def diagnose_real_artifact_write_layer(name: str, real_case: dict) -> None:
    diagnose_real_artifact_write_layer_for_tensors(name, real_case, None)


def diagnose_real_artifact_write_layer_for_tensors(
    name: str,
    real_case: dict,
    selected_tensor_names: list[str] | None,
) -> None:
    py_tmpdir = tempfile.mkdtemp(prefix="hqq-python-artifact-write-shadow-")
    rust_tmpdir = tempfile.mkdtemp(prefix="hqq-rust-artifact-write-shadow-")
    py_model_path = os.path.join(py_tmpdir, os.path.basename(real_case["model_path"]))
    rust_model_path = os.path.join(rust_tmpdir, os.path.basename(real_case["model_path"]))
    os.makedirs(py_model_path, exist_ok=True)
    os.makedirs(rust_model_path, exist_ok=True)

    py_model = _build_shadow_real_artifact_model(real_case, py_model_path)
    rust_model = _build_shadow_real_artifact_model(real_case, rust_model_path)
    old_home = os.environ.get("HOME")
    try:
        os.environ["HOME"] = py_tmpdir
        py_model._prepare_hqq_attention_cache()
        py_outputs = _write_real_artifact_layer_instrumented(
            name,
            "python",
            py_model,
            real_case,
            use_rust_shadow=False,
            selected_tensor_names=selected_tensor_names,
            load_runtime=False,
        )

        os.environ["HOME"] = rust_tmpdir
        rust_model._prepare_hqq_attention_cache()
        rust_outputs = _write_real_artifact_layer_instrumented(
            name,
            "rust_shadow",
            rust_model,
            real_case,
            use_rust_shadow=True,
            selected_tensor_names=selected_tensor_names,
            load_runtime=False,
        )
        py_quantizer_summaries = py_outputs["quantizer_summaries"]
        rust_quantizer_summaries = rust_outputs["quantizer_summaries"]
        py_quantized_outputs = py_outputs["quantized_outputs"]
        rust_quantized_outputs = rust_outputs["quantized_outputs"]
        tensor_names = sorted(set(py_quantizer_summaries.keys()) | set(rust_quantizer_summaries.keys()))
        for tensor_name in tensor_names:
            py_summary = py_quantizer_summaries.get(tensor_name, {})
            rust_summary = rust_quantizer_summaries.get(tensor_name, {})
            bucket_names = [
                "quantizer_invocation",
                "fixed_zero_solve_update",
                "elementwise_quantize_clamp",
                "reductions",
                "scale_zero_update_math",
                "allocation_setup_materialization",
                "unresolved_control_overhead",
            ]
            delta = {}
            for bucket_name in bucket_names:
                py_value = py_summary.get(bucket_name)
                rust_value = rust_summary.get(bucket_name)
                delta[bucket_name] = (
                    None if py_value is None or rust_value is None else float(py_value - rust_value)
                )
            print(
                {
                    "case": name,
                    "stage": "quantizer_side_by_side_summary",
                    "tensor_name": tensor_name,
                    "python": py_summary,
                    "rust_shadow": rust_summary,
                    "delta_python_minus_rust_shadow": delta,
                }
            )

        quantized_compare_start = _stage_start(name, "compare", "quantized_tensor_outputs")
        for tensor_name in tensor_names:
            py_output = py_quantized_outputs.get(tensor_name)
            rust_output = rust_quantized_outputs.get(tensor_name)
            if py_output is None or rust_output is None:
                raise SystemExit(
                    f"Rust HQQ quantized-output presence mismatch for {name} {tensor_name}: "
                    f"python={py_output is not None} rust={rust_output is not None}"
                )
            quantized_diff = _first_quantized_output_mismatch(py_output, rust_output)
            if quantized_diff is not None:
                path, py_value, rust_value = quantized_diff
                raise SystemExit(
                    f"Rust HQQ quantized-output mismatch for {name} at {tensor_name}.{path}: "
                    f"python={py_value!r} rust={rust_value!r}"
                )
            _stage_done(
                name,
                "compare",
                "quantized_tensor_outputs",
                quantized_compare_start,
                tensor_name=tensor_name,
                packed_equal=True,
                scales_equal=True,
                zeros_equal=True,
                group_size=int(py_output["group_size"]),
                orig_shape=list(py_output["orig_shape"]),
            )
            quantized_compare_start = time.perf_counter()

        py_entries = sorted(
            (_normalized_manifest_entry(entry) for entry in py_model._hqq_manifest["tensors"]),
            key=lambda item: (item["layer_idx"], item["tensor_name"]),
        )
        rust_entries = sorted(
            (_normalized_manifest_entry(entry) for entry in rust_model._hqq_manifest["tensors"]),
            key=lambda item: (item["layer_idx"], item["tensor_name"]),
        )
        manifest_diff = _first_nested_diff(py_entries, rust_entries, "manifest")
        if manifest_diff is not None:
            path, py_value, rust_value = manifest_diff
            raise SystemExit(
                f"Rust HQQ artifact manifest mismatch for {name} at {path}: "
                f"python={py_value!r} rust={rust_value!r}"
            )

        artifact_compare_start = _stage_start(name, "compare", "prefill_format_view")
        for py_entry, rust_entry in zip(py_entries, rust_entries):
            py_artifact = load_hqq_attention_artifact(
                py_model.cfg.model_path,
                py_entry,
                expected_nbits=4,
                device="cpu",
            )
            rust_artifact = load_hqq_attention_artifact(
                rust_model.cfg.model_path,
                rust_entry,
                expected_nbits=4,
                device="cpu",
            )
            metadata_diff = _first_nested_diff(
                py_artifact["metadata"],
                rust_artifact["metadata"],
                f"{py_entry['tensor_name']}.metadata",
            )
            if metadata_diff is not None:
                path, py_value, rust_value = metadata_diff
                raise SystemExit(
                    f"Rust HQQ artifact metadata mismatch for {name} at {path}: "
                    f"python={py_value!r} rust={rust_value!r}"
                )
            structure_diff = _first_nested_diff(
                py_artifact["structure"],
                rust_artifact["structure"],
                f"{py_entry['tensor_name']}.structure",
            )
            if structure_diff is not None:
                path, py_value, rust_value = structure_diff
                raise SystemExit(
                    f"Rust HQQ artifact structure mismatch for {name} at {path}: "
                    f"python={py_value!r} rust={rust_value!r}"
                )
            for tensor_name in sorted(py_artifact["tensors"].keys()):
                py_tensor = py_artifact["tensors"][tensor_name]
                rust_tensor = rust_artifact["tensors"][tensor_name]
                if not torch.equal(py_tensor, rust_tensor):
                    diff_idx = (py_tensor != rust_tensor).nonzero(as_tuple=False)
                    first_idx = diff_idx[0].tolist() if diff_idx.numel() else []
                    py_value = py_tensor[tuple(first_idx)].item() if first_idx else None
                    rust_value = rust_tensor[tuple(first_idx)].item() if first_idx else None
                    raise SystemExit(
                        f"Rust HQQ artifact tensor mismatch for {name} at "
                        f"{py_entry['tensor_name']}.{tensor_name}[{first_idx}]: "
                        f"python={py_value!r} rust={rust_value!r}"
                    )
            _stage_done(
                name,
                "compare",
                "prefill_format_view",
                artifact_compare_start,
                tensor_name=py_entry["tensor_name"],
                metadata_equal=True,
                structure_equal=True,
                payload_equal=True,
            )
            artifact_compare_start = time.perf_counter()

        decode_compare_start = _stage_start(name, "compare", "decode_format_view")
        for py_entry, rust_entry in zip(py_entries, rust_entries):
            py_artifact = load_hqq_attention_artifact(
                py_model.cfg.model_path,
                py_entry,
                expected_nbits=4,
                device="cpu",
            )
            rust_artifact = load_hqq_attention_artifact(
                rust_model.cfg.model_path,
                rust_entry,
                expected_nbits=4,
                device="cpu",
            )
            py_desc = {py_entry["tensor_name"]: _runtime_entry_from_artifact(py_entry, py_artifact)}
            rust_desc = {rust_entry["tensor_name"]: _runtime_entry_from_artifact(rust_entry, rust_artifact)}
            runtime_diff = _first_runtime_mismatch(py_desc, rust_desc)
            if runtime_diff is not None:
                path, py_value, rust_value = runtime_diff
                raise SystemExit(
                    f"Rust HQQ runtime mismatch for {name} at {path}: "
                    f"python={py_value!r} rust={rust_value!r}"
                )
            _stage_done(
                name,
                "compare",
                "decode_format_view",
                decode_compare_start,
                tensor_name=py_entry["tensor_name"],
                descriptor_equal=True,
                packed_equal=True,
                scales_equal=True,
                zeros_equal=True,
            )
            decode_compare_start = time.perf_counter()

        print(
            {
                "case": name,
                "artifact_write_equal": True,
                "layer_idx": int(real_case["runtime_layer_idx"]),
                "tensor_names": [entry["tensor_name"] for entry in py_entries],
                "quantized_tensor_outputs_equal": True,
                "prefill_format_view_equal": True,
                "decode_format_view_equal": True,
            }
        )
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        shutil.rmtree(py_tmpdir, ignore_errors=True)
        shutil.rmtree(rust_tmpdir, ignore_errors=True)


def compare_real_artifact_registration_layer(
    name: str,
    real_case: dict,
    selected_tensor_names: list[str] | None = None,
) -> None:
    py_tmpdir = tempfile.mkdtemp(prefix="hqq-python-registration-shadow-")
    rust_tmpdir = tempfile.mkdtemp(prefix="hqq-rust-registration-shadow-")
    py_model_path = os.path.join(py_tmpdir, os.path.basename(real_case["model_path"]))
    rust_model_path = os.path.join(rust_tmpdir, os.path.basename(real_case["model_path"]))
    os.makedirs(py_model_path, exist_ok=True)
    os.makedirs(rust_model_path, exist_ok=True)

    py_model = _build_shadow_real_artifact_model(real_case, py_model_path)
    rust_model = _build_shadow_real_artifact_model(real_case, rust_model_path)
    py_store = None
    rust_store = None
    py_store_keepalive = None
    rust_store_keepalive = None
    py_runtime_keepalive: list[torch.Tensor] = []
    rust_runtime_keepalive: list[torch.Tensor] = []
    old_home = os.environ.get("HOME")
    try:
        os.environ["HOME"] = py_tmpdir
        py_model._prepare_hqq_attention_cache()
        py_write_start = _stage_start(name, "python", "artifact_write")
        _write_real_artifact_layer_with_writer(
            py_model,
            real_case,
            writer=write_hqq_attention_artifact,
            selected_tensor_names=selected_tensor_names,
            load_runtime=False,
        )
        _stage_done(
            name,
            "python",
            "artifact_write",
            py_write_start,
            tensor_count=len(py_model._hqq_manifest.get("tensors", [])),
        )
        py_discover_start = _stage_start(name, "python", "artifact_discovery")
        py_manifest = py_model._hqq_manifest or {}
        py_entries = sorted(
            ((entry["layer_idx"], entry["tensor_name"]) for entry in py_manifest.get("tensors", [])),
            key=lambda item: item,
        )
        _stage_done(
            name,
            "python",
            "artifact_discovery",
            py_discover_start,
            manifest_path=os.path.basename(py_model.cfg.model_path),
            tensor_count=len(py_entries),
            entries=py_entries,
        )
        py_validate_start = _stage_start(name, "python", "metadata_parse_validation")
        py_model._validate_hqq_attention_cache()
        _stage_done(
            name,
            "python",
            "metadata_parse_validation",
            py_validate_start,
            validated_tensor_bytes=int(py_model._hqq_attention_cache_bytes),
        )
        py_load_start = _stage_start(name, "python", "_load_hqq_attention_runtime_state")
        py_model._load_hqq_attention_runtime_state()
        _stage_done(
            name,
            "python",
            "_load_hqq_attention_runtime_state",
            py_load_start,
            loaded_tensors=int(getattr(py_model, "_hqq_attention_loaded_tensors", 0)),
        )

        os.environ["HOME"] = rust_tmpdir
        rust_model._prepare_hqq_attention_cache()
        rust_write_start = _stage_start(name, "rust_shadow", "artifact_write")
        _write_real_artifact_layer_with_writer(
            rust_model,
            real_case,
            writer=_write_hqq_attention_artifact_rust_shadow,
            selected_tensor_names=selected_tensor_names,
            load_runtime=False,
        )
        _stage_done(
            name,
            "rust_shadow",
            "artifact_write",
            rust_write_start,
            tensor_count=len(rust_model._hqq_manifest.get("tensors", [])),
        )
        rust_discover_start = _stage_start(name, "rust_shadow", "artifact_discovery")
        rust_manifest = rust_model._hqq_manifest or {}
        rust_entries = sorted(
            ((entry["layer_idx"], entry["tensor_name"]) for entry in rust_manifest.get("tensors", [])),
            key=lambda item: item,
        )
        _stage_done(
            name,
            "rust_shadow",
            "artifact_discovery",
            rust_discover_start,
            manifest_path=os.path.basename(rust_model.cfg.model_path),
            tensor_count=len(rust_entries),
            entries=rust_entries,
        )
        rust_validate_start = _stage_start(name, "rust_shadow", "metadata_parse_validation")
        rust_model._validate_hqq_attention_cache()
        _stage_done(
            name,
            "rust_shadow",
            "metadata_parse_validation",
            rust_validate_start,
            validated_tensor_bytes=int(rust_model._hqq_attention_cache_bytes),
        )
        rust_load_start = _stage_start(name, "rust_shadow", "_load_hqq_attention_runtime_state")
        rust_model._load_hqq_attention_runtime_state()
        _stage_done(
            name,
            "rust_shadow",
            "_load_hqq_attention_runtime_state",
            rust_load_start,
            loaded_tensors=int(getattr(rust_model, "_hqq_attention_loaded_tensors", 0)),
        )

        layer_idx = real_case["runtime_layer_idx"]
        py_runtime = py_model._hqq_attention_runtime[layer_idx]
        rust_runtime = rust_model._hqq_attention_runtime[layer_idx]
        runtime_mismatch = _first_runtime_mismatch(py_runtime, rust_runtime)
        runtime_tensor_names = sorted(py_runtime.keys())
        if runtime_mismatch is not None:
            mismatch_path, py_value, rust_value = runtime_mismatch
            raise SystemExit(
                f"Rust HQQ registration runtime mismatch for {name} at {mismatch_path}: "
                f"python={py_value!r} rust={rust_value!r}"
            )

        py_store_start = _stage_start(name, "python", "store_creation")
        py_store, py_store_keepalive = build_real_model_store(real_case, include_fused_config=True)
        _stage_done(name, "python", "store_creation", py_store_start)
        rust_store_start = _stage_start(name, "rust_shadow", "store_creation")
        rust_store, rust_store_keepalive = build_real_model_store(real_case, include_fused_config=True)
        _stage_done(name, "rust_shadow", "store_creation", rust_store_start)
        py_register_start = _stage_start(name, "python", "registration_total")
        py_registered = py_model._register_hqq_attention_layers_on_store(
            py_store,
            torch.device("cuda:0"),
            py_runtime_keepalive,
        )
        _stage_done(
            name,
            "python",
            "registration_total",
            py_register_start,
            registered_layers=int(py_registered),
        )
        rust_register_start = _stage_start(name, "rust_shadow", "registration_total")
        rust_registered = rust_model._register_hqq_attention_layers_on_store(
            rust_store,
            torch.device("cuda:0"),
            rust_runtime_keepalive,
        )
        _stage_done(
            name,
            "rust_shadow",
            "registration_total",
            rust_register_start,
            registered_layers=int(rust_registered),
        )
        if py_registered != 1 or rust_registered != 1:
            raise SystemExit(
                f"Rust HQQ registration expected one registered layer for {name}, "
                f"got python={py_registered} rust={rust_registered}"
            )

        py_staging_start = _stage_start(name, "python", "loaded_state_normalization")
        py_staging = _normalized_staging_json(py_store)
        _stage_done(name, "python", "loaded_state_normalization", py_staging_start)
        rust_staging_start = _stage_start(name, "rust_shadow", "loaded_state_normalization")
        rust_staging = _normalized_staging_json(rust_store)
        _stage_done(name, "rust_shadow", "loaded_state_normalization", rust_staging_start)
        staging_diff = _first_nested_diff(py_staging, rust_staging, "staging")
        if staging_diff is not None:
            mismatch_path, py_value, rust_value = staging_diff
            raise SystemExit(
                f"Rust HQQ registration staging mismatch for {name} at {mismatch_path}: "
                f"python={py_value!r} rust={rust_value!r}"
            )

        py_swap_prefill_start = _stage_start(name, "python", "swap_hqq_runtime_to_prefill")
        py_store.swap_hqq_runtime_to_prefill()
        py_swap_prefill = _normalized_staging_json(py_store)
        _stage_done(name, "python", "swap_hqq_runtime_to_prefill", py_swap_prefill_start)
        rust_swap_prefill_start = _stage_start(name, "rust_shadow", "swap_hqq_runtime_to_prefill")
        rust_store.swap_hqq_runtime_to_prefill()
        rust_swap_prefill = _normalized_staging_json(rust_store)
        _stage_done(name, "rust_shadow", "swap_hqq_runtime_to_prefill", rust_swap_prefill_start)
        swap_prefill_diff = _first_nested_diff(py_swap_prefill, rust_swap_prefill, "staging")
        if swap_prefill_diff is not None:
            mismatch_path, py_value, rust_value = swap_prefill_diff
            raise SystemExit(
                f"Rust HQQ prefill swap mismatch for {name} at {mismatch_path}: "
                f"python={py_value!r} rust={rust_value!r}"
            )

        py_swap_decode_start = _stage_start(name, "python", "swap_hqq_runtime_to_decode")
        py_store.swap_hqq_runtime_to_decode()
        py_swap_decode = _normalized_staging_json(py_store)
        _stage_done(name, "python", "swap_hqq_runtime_to_decode", py_swap_decode_start)
        rust_swap_decode_start = _stage_start(name, "rust_shadow", "swap_hqq_runtime_to_decode")
        rust_store.swap_hqq_runtime_to_decode()
        rust_swap_decode = _normalized_staging_json(rust_store)
        _stage_done(name, "rust_shadow", "swap_hqq_runtime_to_decode", rust_swap_decode_start)
        swap_decode_diff = _first_nested_diff(py_swap_decode, rust_swap_decode, "staging")
        if swap_decode_diff is not None:
            mismatch_path, py_value, rust_value = swap_decode_diff
            raise SystemExit(
                f"Rust HQQ decode swap mismatch for {name} at {mismatch_path}: "
                f"python={py_value!r} rust={rust_value!r}"
            )

        print(
            {
                "case": name,
                "registration_runtime_equal": True,
                "registration_staging_equal": True,
                "swap_prefill_equal": True,
                "swap_decode_equal": True,
                "runtime_tensor_names": runtime_tensor_names,
                "layer_idx": int(layer_idx),
            }
        )
    finally:
        py_store = None
        rust_store = None
        py_store_keepalive = None
        rust_store_keepalive = None
        py_runtime_keepalive.clear()
        rust_runtime_keepalive.clear()
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        shutil.rmtree(py_tmpdir, ignore_errors=True)
        shutil.rmtree(rust_tmpdir, ignore_errors=True)


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _resolve_breadth_model_specs(value: str) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for part in _parse_csv_strings(value):
        label = part
        path = DEFAULT_BREADTH_MODEL_PATHS.get(part, part)
        if "=" in part:
            label, raw_path = part.split("=", 1)
            label = label.strip()
            path = raw_path.strip()
        resolved = os.path.abspath(os.path.expanduser(path))
        specs.append((label, resolved))
    return specs


def _real_case_routed_expert_indices(real_case: dict) -> list[int]:
    layer_idx = int(real_case["layer_idx"])
    prefix = f"model.layers.{layer_idx}.mlp.experts."
    indices: set[int] = set()
    for tensor_name in _dense_artifact_tensor_specs_from_real_case(real_case):
        if not tensor_name.startswith(prefix):
            continue
        match = re.match(rf"{re.escape(prefix)}(\d+)\.", tensor_name)
        if match is not None:
            indices.add(int(match.group(1)))
    return sorted(indices)


def _select_breadth_tensor_names(real_case: dict) -> list[str]:
    selected = ["q_proj", "k_proj", "v_proj", "o_proj"]
    layer = real_case["layer"]
    tensor_specs = _dense_artifact_tensor_specs_from_real_case(real_case)
    if getattr(layer, "dense_mlp", None) is not None:
        for tensor_name in ("gate_proj", "up_proj", "down_proj"):
            if tensor_name in tensor_specs:
                selected.append(tensor_name)
        return selected

    for tensor_name in ("gate_up_proj", "down_proj"):
        if tensor_name in tensor_specs:
            selected.append(tensor_name)

    expert_indices = _real_case_routed_expert_indices(real_case)
    if expert_indices:
        expert_idx = expert_indices[-1]
        layer_idx = int(real_case["layer_idx"])
        for tensor_name in (
            f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_up_proj",
            f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj",
        ):
            if tensor_name in tensor_specs:
                selected.append(tensor_name)
    return selected


def _discover_breadth_registration_case(model_path: str) -> dict:
    cfg = ModelConfig.from_model_path(model_path)
    if not cfg.is_gqa:
        raise SystemExit(
            f"Breadth registration coverage currently requires a GQA model; "
            f"{model_path} resolved to attention_type={cfg.attention_type}"
        )

    first_attention_layer = None
    first_attention_moe_layer = None
    for layer_idx in range(cfg.num_hidden_layers):
        if cfg.is_mamba2_layer(layer_idx) or cfg.is_moe_only_layer(layer_idx):
            continue
        if not cfg.is_full_attention_layer(layer_idx):
            continue
        if first_attention_layer is None:
            first_attention_layer = layer_idx
        if cfg.is_moe_layer(layer_idx):
            first_attention_moe_layer = layer_idx
            break

    selected_layer_idx = first_attention_moe_layer
    if selected_layer_idx is None:
        selected_layer_idx = first_attention_layer
    if selected_layer_idx is None:
        raise SystemExit(f"No attention-capable layer found for breadth registration coverage in {model_path}")

    real_case = load_real_model_case(model_path, selected_layer_idx)
    selected_tensor_names = _select_breadth_tensor_names(real_case)
    routed_expert_indices = _real_case_routed_expert_indices(real_case)
    layer = real_case["layer"]
    if getattr(layer, "dense_mlp", None) is not None:
        coverage_kind = "dense_mlp"
    elif routed_expert_indices:
        coverage_kind = "shared_plus_routed_moe" if getattr(layer, "shared_expert", None) else "routed_moe"
    elif getattr(layer, "shared_expert", None):
        coverage_kind = "shared_expert_only"
    else:
        coverage_kind = "attention_only"
    return {
        "cfg": cfg,
        "real_case": real_case,
        "selected_tensor_names": selected_tensor_names,
        "selected_layer_idx": selected_layer_idx,
        "coverage_kind": coverage_kind,
        "routed_expert_indices": routed_expert_indices,
    }


def run_real_model_breadth_registration_case(case_label: str, model_path: str) -> None:
    if not os.path.isdir(model_path):
        raise SystemExit(f"Breadth registration model path unavailable for {case_label}: {model_path}")
    discovery = _discover_breadth_registration_case(model_path)
    print(
        {
            "case": f"breadth_{case_label}_selection",
            "model_path": model_path,
            "attention_type": discovery["cfg"].attention_type,
            "layer_idx": int(discovery["selected_layer_idx"]),
            "coverage_kind": discovery["coverage_kind"],
            "selected_tensor_names": discovery["selected_tensor_names"],
            "routed_expert_indices": discovery["routed_expert_indices"],
        }
    )
    compare_real_artifact_registration_layer(
        f"breadth_{case_label}_layer{discovery['selected_layer_idx']}_artifact_registration",
        discovery["real_case"],
        selected_tensor_names=discovery["selected_tensor_names"],
    )


def diagnose_solve_scale_drift(name: str, chunk: torch.Tensor, zero: torch.Tensor, scale_seed: torch.Tensor) -> None:
    py_q, py_scale, _ = _solve_hqq4_fixed_zero(chunk, 15.0, zero, scale_seed)
    rust_q, rust_scale, _ = solve_hqq4_fixed_zero_rust(chunk, zero, scale_seed)
    scale_diff = (py_scale - rust_scale).abs()
    row = int(scale_diff.argmax().item())
    row_chunk = chunk[row].detach().to(dtype=torch.float32).contiguous()
    row_scale_seed = torch.tensor(float(scale_seed[row]), dtype=torch.float32)
    row_zero = float(zero[row].item())
    row_q = ((row_chunk / row_scale_seed) + row_zero).round().clamp_(0.0, 15.0)
    row_centered = row_q - row_zero
    numer_terms = (row_chunk * row_centered).tolist()
    torch_numer = float(torch.tensor(numer_terms, dtype=torch.float32).sum().item())
    seq_numer, first_prefix_mismatch = _sequential_f32_sum(numer_terms)
    print(
        {
            "case": name,
            "row": row,
            "solve_q_equal": torch.equal(py_q[row], rust_q[row]),
            "scale_max_abs_diff": float(scale_diff[row].item()),
            "python_scale": float(py_scale[row].item()),
            "rust_scale": float(rust_scale[row].item()),
            "torch_numer_f32": torch_numer,
            "sequential_numer_f32": seq_numer,
            "first_prefix_mismatch": first_prefix_mismatch,
        }
    )


def diagnose_retained_best_state_drift(name: str, chunk: torch.Tensor) -> None:
    qmax = 15.0
    init_q, init_scale, init_zero = _quantize_hqq4_group_current(chunk, qmax)
    init_deq = (init_q.to(torch.float32) - init_zero.unsqueeze(1)) * init_scale.unsqueeze(1)
    py_best_rmse = torch.mean((init_deq - chunk) ** 2, dim=1)
    rust_best_rmse = compute_hqq4_rmse_rust(chunk, init_q, init_scale, init_zero)

    range_scale = ((chunk.amax(dim=1) - chunk.amin(dim=1)) / qmax).clamp(min=1e-8)
    zero = torch.zeros((chunk.shape[0],), dtype=torch.float32)
    py_q, py_scale, py_zero = _solve_hqq4_fixed_zero(chunk, qmax, zero, range_scale)
    rust_q, rust_scale, rust_zero = solve_hqq4_fixed_zero_rust(chunk, zero, range_scale)
    if not torch.equal(py_q, rust_q) or not torch.equal(py_scale, rust_scale) or not torch.equal(py_zero, rust_zero):
        raise SystemExit(f"Rust HQQ solve mismatch before retained-state diagnosis for {name}")

    py_rmse = torch.mean(
        (((py_q.to(torch.float32) - py_zero.unsqueeze(1)) * py_scale.unsqueeze(1)) - chunk) ** 2,
        dim=1,
    )
    rust_rmse = compute_hqq4_rmse_rust(chunk, rust_q, rust_scale, rust_zero)
    rmse_diff = (py_rmse - rust_rmse).abs()
    row = int(rmse_diff.argmax().item())
    row_chunk = chunk[row].detach().to(dtype=torch.float32).contiguous()
    row_q = py_q[row].to(torch.float32)
    row_scale = float(py_scale[row].item())
    row_zero = float(py_zero[row].item())
    rmse_terms = (((row_q - row_zero) * row_scale) - row_chunk).pow(2).tolist()
    rust_rmse_sum, first_prefix_mismatch = _sequential_f32_sum(rmse_terms)
    improved_py = py_rmse < py_best_rmse
    improved_rust = rust_rmse < rust_best_rmse
    print(
        {
            "case": name,
            "pass": "global_range",
            "candidate_idx": 0,
            "zero": 0.0,
            "row": row,
            "solve_q_equal": torch.equal(py_q[row], rust_q[row]),
            "solve_scale_equal": torch.equal(py_scale[row], rust_scale[row]),
            "solve_zero_equal": torch.equal(py_zero[row], rust_zero[row]),
            "py_best_before": float(py_best_rmse[row].item()),
            "rust_best_before": float(rust_best_rmse[row].item()),
            "py_rmse": float(py_rmse[row].item()),
            "rust_rmse": float(rust_rmse[row].item()),
            "rmse_abs_diff": float(rmse_diff[row].item()),
            "improved_py": bool(improved_py[row].item()),
            "improved_rust": bool(improved_rust[row].item()),
            "torch_rmse_sum": float(torch.tensor(rmse_terms, dtype=torch.float32).sum().item()),
            "rust_rmse_sum": rust_rmse_sum,
            "first_prefix_mismatch": first_prefix_mismatch,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next"))
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--include-real", action="store_true")
    parser.add_argument("--real-layers", default="")
    parser.add_argument("--real-tensors", default="q_proj,k_proj,v_proj,o_proj,fused_qkv")
    parser.add_argument("--group-indices", default="")
    parser.add_argument("--q35-model-path", default=os.path.expanduser("~/.krasis/models/Qwen3.5-35B-A3B"))
    parser.add_argument("--q35-layers", default="")
    parser.add_argument("--include-q35", action="store_true")
    parser.add_argument("--include-artifact-real", action="store_true")
    parser.add_argument("--artifact-real-layers", default="")
    parser.add_argument("--include-artifact-q35", action="store_true")
    parser.add_argument("--artifact-q35-layers", default="")
    parser.add_argument("--include-registration-real", action="store_true")
    parser.add_argument("--registration-real-layers", default="")
    parser.add_argument("--include-registration-q35", action="store_true")
    parser.add_argument("--registration-q35-layers", default="")
    parser.add_argument("--include-breadth-registration", action="store_true")
    parser.add_argument("--breadth-models", default="qwen3-0.6b,q35")
    args = parser.parse_args()

    torch.manual_seed(0)
    threshold_case = torch.tensor(
        [[-0.03173828125, 0.03173828125, -0.025390625, 0.0]],
        dtype=torch.float32,
    )
    compare_init("synthetic_threshold_init", threshold_case)
    ties_even_case = torch.tensor(
        [[-0.9375, 0.9375, 0.625, 0.0]],
        dtype=torch.float32,
    )
    compare_init("synthetic_ties_even_init", ties_even_case)
    solve_ties_even_case = torch.tensor([[0.0125, 0.0, -0.0125]], dtype=torch.float32)
    solve_zero = torch.zeros((1,), dtype=torch.float32)
    solve_scale_seed = torch.full((1,), 0.005, dtype=torch.float32)
    compare_solve("synthetic_ties_even_solve", solve_ties_even_case, solve_zero, solve_scale_seed)
    synthetic_scale_drift_case = torch.tensor(
        [[
            0.016357421875,
            -0.04541015625,
            0.042724609375,
            0.0225830078125,
            0.01336669921875,
            -0.00555419921875,
            -0.01397705078125,
            0.0242919921875,
            -0.01177978515625,
            0.00153350830078125,
            0.00811767578125,
            0.003662109375,
            -0.0027008056640625,
            0.010009765625,
            -0.006317138671875,
            -0.01519775390625,
            -0.06787109375,
            0.06787109375,
            0.00185394287109375,
            -0.0002880096435546875,
            -0.00494384765625,
            0.000415802001953125,
        ]],
        dtype=torch.float32,
    )
    synthetic_scale_zero = torch.tensor([0.1171875], dtype=torch.float32)
    synthetic_scale_seed = torch.tensor([0.03330077975988388], dtype=torch.float32)
    diagnose_solve_scale_drift(
        "synthetic_scale_update_reduction",
        synthetic_scale_drift_case,
        synthetic_scale_zero,
        synthetic_scale_seed,
    )
    synthetic_rmse_chunk = torch.tensor(
        [[0.25, -0.125, 0.0625, -0.03125, 0.015625, -0.0078125, 0.00390625, -0.001953125, 0.0, 0.125, -0.25]],
        dtype=torch.float32,
    )
    synthetic_rmse_scale = torch.tensor([0.0625], dtype=torch.float32)
    synthetic_rmse_zero = torch.tensor([4.0], dtype=torch.float32)
    synthetic_rmse_q = ((synthetic_rmse_chunk / synthetic_rmse_scale.unsqueeze(1)) + synthetic_rmse_zero.unsqueeze(1)).round().clamp_(0.0, 15.0).to(torch.uint8)
    compare_rmse("synthetic_squared_error_reduction", synthetic_rmse_chunk, synthetic_rmse_q, synthetic_rmse_scale, synthetic_rmse_zero)
    compare_quantized("synthetic_64x128", torch.randn(64, 128, dtype=torch.float32))
    compare_quantized("synthetic_96x192", torch.randn(96, 192, dtype=torch.float32) * 0.25)

    if args.include_real:
        real_tensor_names = _parse_csv_strings(args.real_tensors)
        group_indices = _parse_csv_ints(args.group_indices)
        qcn_layers = _parse_csv_ints(args.real_layers) if args.real_layers else [args.layer_idx]
        q35_layers = _parse_csv_ints(args.q35_layers) if args.q35_layers else [3]

        real_case = load_real_model_case(args.model_path, qcn_layers[0])
        q_chunk = real_case["q_w"].to(dtype=torch.float32)[:, :128]
        compare_init("real_q_proj_group0_init", q_chunk)
        range_scale = ((q_chunk.amax(dim=1) - q_chunk.amin(dim=1)) / 15.0).clamp(min=1e-8)
        zero = torch.zeros((q_chunk.shape[0],), dtype=torch.float32)
        compare_solve("real_q_proj_group0_global_candidate0_solve", q_chunk, zero, range_scale)
        candidate1_zero = torch.full((q_chunk.shape[0],), 0.1171875, dtype=torch.float32)
        diagnose_solve_scale_drift(
            "real_q_proj_group0_global_candidate1_scale",
            q_chunk,
            candidate1_zero,
            range_scale,
        )
        candidate0_q, candidate0_scale, candidate0_zero = _solve_hqq4_fixed_zero(q_chunk, 15.0, zero, range_scale)
        compare_rmse(
            "real_q_proj_group0_retained_best_state",
            q_chunk,
            candidate0_q,
            candidate0_scale,
            candidate0_zero,
        )
        diagnose_retained_best_state_drift("real_q_proj_group0_retained_best_state", q_chunk)
        for layer_idx in qcn_layers:
            compare_real_case_tensors(
                f"real_qcn_layer{layer_idx}",
                load_real_model_case(args.model_path, layer_idx),
                real_tensor_names,
                group_indices,
            )

        q35_available = os.path.isdir(os.path.abspath(os.path.expanduser(args.q35_model_path)))
        print(
            {
                "case": "real_q35_availability",
                "model_path": os.path.abspath(os.path.expanduser(args.q35_model_path)),
                "available": q35_available,
                "requested": bool(args.include_q35),
                "layers": q35_layers,
            }
        )
        if args.include_q35:
            if not q35_available:
                raise SystemExit(f"Requested Q35B parity coverage but model path is unavailable: {args.q35_model_path}")
            for layer_idx in q35_layers:
                compare_real_case_tensors(
                    f"real_q35_layer{layer_idx}",
                    load_real_model_case(args.q35_model_path, layer_idx),
                    real_tensor_names,
                    group_indices,
                )

    if args.include_artifact_real:
        qcn_artifact_layers = _parse_csv_ints(args.artifact_real_layers) if args.artifact_real_layers else [args.layer_idx]
        for layer_idx in qcn_artifact_layers:
            compare_real_artifact_layer(
                f"real_qcn_layer{layer_idx}_artifact_builder",
                load_real_model_case(args.model_path, layer_idx),
            )

        q35_artifact_layers = _parse_csv_ints(args.artifact_q35_layers) if args.artifact_q35_layers else [3]
        q35_artifact_available = os.path.isdir(os.path.abspath(os.path.expanduser(args.q35_model_path)))
        print(
            {
                "case": "real_q35_artifact_availability",
                "model_path": os.path.abspath(os.path.expanduser(args.q35_model_path)),
                "available": q35_artifact_available,
                "requested": bool(args.include_artifact_q35),
                "layers": q35_artifact_layers,
            }
        )
        if args.include_artifact_q35:
            if not q35_artifact_available:
                raise SystemExit(
                    f"Requested Q35B artifact parity coverage but model path is unavailable: {args.q35_model_path}"
                )
            for layer_idx in q35_artifact_layers:
                compare_real_artifact_layer(
                    f"real_q35_layer{layer_idx}_artifact_builder",
                    load_real_model_case(args.q35_model_path, layer_idx),
                )

    if args.include_registration_real:
        qcn_registration_layers = (
            _parse_csv_ints(args.registration_real_layers) if args.registration_real_layers else [args.layer_idx]
        )
        for layer_idx in qcn_registration_layers:
            compare_real_artifact_registration_layer(
                f"real_qcn_layer{layer_idx}_artifact_registration",
                load_real_model_case(args.model_path, layer_idx),
            )

        q35_registration_layers = (
            _parse_csv_ints(args.registration_q35_layers) if args.registration_q35_layers else [3]
        )
        q35_registration_available = os.path.isdir(os.path.abspath(os.path.expanduser(args.q35_model_path)))
        print(
            {
                "case": "real_q35_registration_availability",
                "model_path": os.path.abspath(os.path.expanduser(args.q35_model_path)),
                "available": q35_registration_available,
                "requested": bool(args.include_registration_q35),
                "layers": q35_registration_layers,
            }
        )
        if args.include_registration_q35:
            if not q35_registration_available:
                raise SystemExit(
                    f"Requested Q35B registration parity coverage but model path is unavailable: {args.q35_model_path}"
                )
            for layer_idx in q35_registration_layers:
                compare_real_artifact_registration_layer(
                    f"real_q35_layer{layer_idx}_artifact_registration",
                    load_real_model_case(args.q35_model_path, layer_idx),
                )

    if args.include_breadth_registration:
        for case_label, model_path in _resolve_breadth_model_specs(args.breadth_models):
            run_real_model_breadth_registration_case(case_label, model_path)


if __name__ == "__main__":
    main()
