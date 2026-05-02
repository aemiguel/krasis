#!/usr/bin/env python3
"""Prototype GPU HQQ candidate-search probe.

This diagnostic is intentionally runnable only through ./dev. It does not write
HQQ cache artifacts and does not change runtime defaults.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from krasis.attention_backend import (
    HQQ4_GLOBAL_STEPS,
    HQQ4_LOCAL_STEPS,
    HQQ_DEFAULT_GROUP_SIZE,
    _hqq_num_groups,
    _hqq_padded_cols,
    _pack_hqq_quant,
    _unpack_hqq_quant,
    quantize_hqq4_tensor_rust,
    quantize_hqq6_tensor,
    quantize_hqq8_tensor,
)
from krasis import hqq_search_cuda_tensor_ptr


MODEL_ALIASES = {
    "q122b": Path("~/.krasis/models/Qwen3.5-122B-A10B").expanduser(),
    "q35b": Path("~/.krasis/models/Qwen3.5-35B-A3B").expanduser(),
    "qcn": Path("~/.krasis/models/Qwen3-Coder-Next").expanduser(),
}


def require_dev_entrypoint() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("This script must be run via ./dev hqq-gpu-search-probe")


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"Missing safetensors index: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    return dict(index["weight_map"])


def find_layer_tensor_key(weight_map: dict[str, str], layer_idx: int, tensor_name: str) -> str:
    suffix = f".{tensor_name}.weight"
    layer_part = f".layers.{layer_idx}."
    matches = [key for key in weight_map if layer_part in key and key.endswith(suffix)]
    if not matches:
        raise RuntimeError(f"No tensor ending {suffix!r} found for layer {layer_idx}")
    self_attn = [key for key in matches if ".self_attn." in key]
    if self_attn:
        return sorted(self_attn)[0]
    return sorted(matches)[0]


def load_tensor(model_dir: Path, weight_map: dict[str, str], layer_idx: int, tensor_name: str) -> torch.Tensor:
    if tensor_name == "fused_qkv":
        parts = [
            load_tensor(model_dir, weight_map, layer_idx, name)
            for name in ("q_proj", "k_proj", "v_proj")
        ]
        return torch.cat(parts, dim=0).contiguous()

    key = find_layer_tensor_key(weight_map, layer_idx, tensor_name)
    shard_path = model_dir / weight_map[key]
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(key)
    return tensor.contiguous()


def qmax_for_nbits(nbits: int) -> float:
    if nbits not in (4, 6, 8):
        raise ValueError(f"Unsupported nbits={nbits}; expected one of 4,6,8")
    return float((1 << nbits) - 1)


def current_cpu_quantize(weight: torch.Tensor, nbits: int, group_size: int) -> dict[str, torch.Tensor]:
    if nbits == 4:
        return quantize_hqq4_tensor_rust(weight, group_size=group_size, collect_stats=False)
    if nbits == 6:
        return quantize_hqq6_tensor(weight, group_size=group_size, collect_stats=False)
    if nbits == 8:
        return quantize_hqq8_tensor(weight, group_size=group_size, collect_stats=False)
    raise ValueError(f"Unsupported nbits={nbits}")


def metrics_from_quant(
    source: torch.Tensor,
    quant: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> dict[str, float]:
    source_f32 = source.detach().to(dtype=torch.float32)
    if quant.device != source_f32.device:
        quant = quant.to(source_f32.device)
    if scales.device != source_f32.device:
        scales = scales.to(source_f32.device)
    if zeros.device != source_f32.device:
        zeros = zeros.to(source_f32.device)

    rows, cols = source_f32.shape
    groups = _hqq_num_groups(cols, group_size)
    sum_abs = torch.zeros((), device=source_f32.device, dtype=torch.float64)
    sum_sq = torch.zeros((), device=source_f32.device, dtype=torch.float64)
    max_abs = torch.zeros((), device=source_f32.device, dtype=torch.float32)
    source_sum_abs = torch.zeros((), device=source_f32.device, dtype=torch.float64)
    source_sum_sq = torch.zeros((), device=source_f32.device, dtype=torch.float64)

    for group_idx in range(groups):
        start = group_idx * group_size
        end = min(start + group_size, cols)
        src = source_f32[:, start:end]
        q = quant[:, start:end].to(torch.float32)
        scale = scales[:, group_idx].unsqueeze(1)
        zero = zeros[:, group_idx].unsqueeze(1)
        diff = ((q - zero) * scale) - src
        abs_diff = diff.abs()
        sum_abs += abs_diff.sum(dtype=torch.float64)
        sum_sq += (diff * diff).sum(dtype=torch.float64)
        max_abs = torch.maximum(max_abs, abs_diff.max())
        source_sum_abs += src.abs().sum(dtype=torch.float64)
        source_sum_sq += (src * src).sum(dtype=torch.float64)

    numel = max(1, rows * cols)
    mean_abs = float((sum_abs / numel).item())
    rmse = math.sqrt(float((sum_sq / numel).item()))
    source_mean_abs = float((source_sum_abs / numel).item())
    source_rms = math.sqrt(float((source_sum_sq / numel).item()))
    return {
        "mean_abs": mean_abs,
        "rmse": rmse,
        "max_abs": float(max_abs.item()),
        "source_mean_abs": source_mean_abs,
        "source_rms": source_rms,
        "relative_mean_abs": mean_abs / source_mean_abs if source_mean_abs > 0.0 else 0.0,
        "relative_rmse": rmse / source_rms if source_rms > 0.0 else 0.0,
    }


def unpack_cpu_quant(payload: dict[str, torch.Tensor], nbits: int, cols: int, group_size: int) -> torch.Tensor:
    padded_cols = _hqq_padded_cols(cols, group_size)
    return _unpack_hqq_quant(payload["packed"], padded_cols, nbits)[:, :cols].contiguous()


def solve_candidate_batch(
    chunk: torch.Tensor,
    zeros: torch.Tensor,
    scale_seed: torch.Tensor,
    qmax: float,
    iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    view = chunk.unsqueeze(0)
    zero_view = zeros.view(-1, 1, 1)
    scale = scale_seed.view(1, -1, 1).expand(zeros.numel(), -1, -1).clone()
    for _ in range(iters):
        q = torch.round(view / scale + zero_view).clamp_(0.0, qmax)
        centered = q - zero_view
        denom = (centered * centered).sum(dim=2)
        numer = (view * centered).sum(dim=2)
        next_scale = (numer / denom.clamp_min(1e-12)).clamp_min(1e-8)
        scale = torch.where((denom > 1e-12).unsqueeze(2), next_scale.unsqueeze(2), scale.clamp_min(1e-8))
    q = torch.round(view / scale + zero_view).clamp_(0.0, qmax)
    diff = ((q - zero_view) * scale) - view
    rmse = (diff * diff).mean(dim=2)
    return q.to(torch.uint8), scale.squeeze(2), rmse


def update_best_cuda(
    chunk: torch.Tensor,
    zero_grid: torch.Tensor,
    scale_seed: torch.Tensor,
    qmax: float,
    iters: int,
    candidate_batch: int,
    best_q: torch.Tensor,
    best_scale: torch.Tensor,
    best_zero: torch.Tensor,
    best_rmse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if candidate_batch <= 0:
        raise ValueError("candidate_batch must be positive")
    for offset in range(0, zero_grid.numel(), candidate_batch):
        zeros = zero_grid[offset : offset + candidate_batch]
        cand_q, cand_scale, cand_rmse = solve_candidate_batch(chunk, zeros, scale_seed, qmax, iters)
        for idx in range(zeros.numel()):
            improved = cand_rmse[idx] < best_rmse
            if torch.any(improved):
                best_rmse = torch.where(improved, cand_rmse[idx], best_rmse)
                best_scale = torch.where(improved, cand_scale[idx], best_scale)
                best_zero = torch.where(improved, zeros[idx].expand_as(best_zero), best_zero)
                best_q = torch.where(improved.unsqueeze(1), cand_q[idx], best_q)
    return best_q, best_scale, best_zero, best_rmse


def gpu_search_quantize(
    weight_cpu: torch.Tensor,
    *,
    nbits: int,
    group_size: int,
    global_steps: int,
    local_steps: int,
    iters: int,
    candidate_batch: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for HQQ GPU search probe")
    device_obj = torch.device(device)
    torch.cuda.set_device(device_obj)
    torch.cuda.reset_peak_memory_stats(device_obj)
    copy_start = torch.cuda.Event(enable_timing=True)
    copy_end = torch.cuda.Event(enable_timing=True)
    copy_start.record()
    weight = weight_cpu.detach().to(device=device_obj, dtype=torch.float32, non_blocking=False).contiguous()
    copy_end.record()
    torch.cuda.synchronize(device_obj)
    h2d_s = copy_start.elapsed_time(copy_end) / 1000.0

    rows, cols = weight.shape
    groups = _hqq_num_groups(cols, group_size)
    padded_cols = _hqq_padded_cols(cols, group_size)
    quant = torch.zeros((rows, padded_cols), device=device, dtype=torch.uint8)
    scales = torch.empty((rows, groups), device=device, dtype=torch.float32)
    zeros = torch.empty((rows, groups), device=device, dtype=torch.float32)
    qmax = qmax_for_nbits(nbits)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for group_idx in range(groups):
        start = group_idx * group_size
        end = min(start + group_size, cols)
        chunk = weight[:, start:end]
        minv = chunk.amin(dim=1)
        maxv = chunk.amax(dim=1)
        scale = ((maxv - minv) / qmax).clamp_min(1e-8)
        zero = (-minv / scale).clamp(0.0, qmax)
        best_q = torch.round(chunk / scale.unsqueeze(1) + zero.unsqueeze(1)).clamp_(0.0, qmax).to(torch.uint8)
        diff = ((best_q.to(torch.float32) - zero.unsqueeze(1)) * scale.unsqueeze(1)) - chunk
        best_rmse = (diff * diff).mean(dim=1)
        best_scale = scale
        best_zero = zero

        range_scale = ((maxv - minv) / qmax).clamp_min(1e-8)
        abs_scale = ((2.0 * chunk.abs().amax(dim=1)) / qmax).clamp_min(1e-8)
        global_zero_grid = torch.linspace(0.0, qmax, steps=global_steps, device=device, dtype=torch.float32)
        best_q, best_scale, best_zero, best_rmse = update_best_cuda(
            chunk,
            global_zero_grid,
            range_scale,
            qmax,
            iters,
            candidate_batch,
            best_q,
            best_scale,
            best_zero,
            best_rmse,
        )
        best_q, best_scale, best_zero, best_rmse = update_best_cuda(
            chunk,
            global_zero_grid,
            abs_scale,
            qmax,
            iters,
            candidate_batch,
            best_q,
            best_scale,
            best_zero,
            best_rmse,
        )
        local_min = max(0.0, float(best_zero.min().item()) - 0.5)
        local_max = min(qmax, float(best_zero.max().item()) + 0.5)
        local_zero_grid = torch.linspace(local_min, local_max, steps=local_steps, device=device, dtype=torch.float32)
        best_q, best_scale, best_zero, best_rmse = update_best_cuda(
            chunk,
            local_zero_grid,
            best_scale.clone(),
            qmax,
            iters,
            candidate_batch,
            best_q,
            best_scale,
            best_zero,
            best_rmse,
        )
        quant[:, start:end] = best_q[:, : end - start]
        scales[:, group_idx] = best_scale
        zeros[:, group_idx] = best_zero
    end_event.record()
    torch.cuda.synchronize(device_obj)
    search_s = start_event.elapsed_time(end_event) / 1000.0

    metrics = metrics_from_quant(weight, quant[:, :cols], scales, zeros, group_size)

    out_start = torch.cuda.Event(enable_timing=True)
    out_end = torch.cuda.Event(enable_timing=True)
    out_start.record()
    quant_cpu = quant.cpu()
    scales_cpu = scales.cpu()
    zeros_cpu = zeros.cpu()
    out_end.record()
    torch.cuda.synchronize(device_obj)
    d2h_s = out_start.elapsed_time(out_end) / 1000.0

    pack_started = time.perf_counter()
    packed = _pack_hqq_quant(quant_cpu, nbits)
    pack_s = time.perf_counter() - pack_started

    tensors = {
        "quant": quant_cpu[:, :cols].contiguous(),
        "packed": packed,
        "scales": scales_cpu,
        "zeros": zeros_cpu,
    }
    timing = {
        "h2d_s": h2d_s,
        "gpu_search_s": search_s,
        "d2h_s": d2h_s,
        "cpu_pack_s": pack_s,
        "total_with_transfer_pack_s": h2d_s + search_s + d2h_s + pack_s,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device_obj) / (1024 * 1024),
    }
    return {"timing": timing, "metrics": metrics}, tensors


def rust_cuda_search_quantize(
    weight_cpu: torch.Tensor,
    *,
    nbits: int,
    group_size: int,
    global_steps: int,
    local_steps: int,
    iters: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for HQQ Rust/CUDA search probe")
    device_obj = torch.device(device)
    torch.cuda.set_device(device_obj)
    torch.cuda.reset_peak_memory_stats(device_obj)
    copy_start = torch.cuda.Event(enable_timing=True)
    copy_end = torch.cuda.Event(enable_timing=True)
    copy_start.record()
    weight = weight_cpu.detach().to(device=device_obj, dtype=torch.float32, non_blocking=False).contiguous()
    copy_end.record()
    torch.cuda.synchronize(device_obj)
    h2d_s = copy_start.elapsed_time(copy_end) / 1000.0

    rows, cols = weight.shape
    groups = _hqq_num_groups(cols, group_size)
    padded_cols = _hqq_padded_cols(cols, group_size)
    quant = torch.empty((rows, padded_cols), device=device_obj, dtype=torch.uint8)
    scales = torch.empty((rows, groups), device=device_obj, dtype=torch.float32)
    zeros = torch.empty((rows, groups), device=device_obj, dtype=torch.float32)

    torch.cuda.synchronize(device_obj)
    search_started = time.perf_counter()
    hqq_search_cuda_tensor_ptr(
        int(weight.data_ptr()),
        int(rows),
        int(cols),
        int(group_size),
        int(nbits),
        int(global_steps),
        int(local_steps),
        int(iters),
        int(quant.data_ptr()),
        int(scales.data_ptr()),
        int(zeros.data_ptr()),
        int(device_obj.index or 0),
    )
    torch.cuda.synchronize(device_obj)
    search_s = time.perf_counter() - search_started

    metrics = metrics_from_quant(weight, quant[:, :cols], scales, zeros, group_size)

    out_start = torch.cuda.Event(enable_timing=True)
    out_end = torch.cuda.Event(enable_timing=True)
    out_start.record()
    quant_cpu = quant.cpu()
    scales_cpu = scales.cpu()
    zeros_cpu = zeros.cpu()
    out_end.record()
    torch.cuda.synchronize(device_obj)
    d2h_s = out_start.elapsed_time(out_end) / 1000.0

    pack_started = time.perf_counter()
    packed = _pack_hqq_quant(quant_cpu, nbits)
    pack_s = time.perf_counter() - pack_started

    tensors = {
        "quant": quant_cpu[:, :cols].contiguous(),
        "packed": packed,
        "scales": scales_cpu,
        "zeros": zeros_cpu,
    }
    timing = {
        "h2d_s": h2d_s,
        "rust_cuda_search_s": search_s,
        "d2h_s": d2h_s,
        "cpu_pack_s": pack_s,
        "total_with_transfer_pack_s": h2d_s + search_s + d2h_s + pack_s,
        "peak_torch_allocated_mb": torch.cuda.max_memory_allocated(device_obj) / (1024 * 1024),
        "note": "Scratch allocated through CUDA driver is not included in torch peak allocator stats.",
    }
    return {"timing": timing, "metrics": metrics}, tensors


def run_case(
    model_dir: Path,
    weight_map: dict[str, str],
    *,
    layer_idx: int,
    tensor_name: str,
    nbits: int,
    group_size: int,
    global_steps: int,
    local_steps: int,
    iters: int,
    candidate_batch: int,
    device: str,
    cpu_baseline: bool,
    backend: str,
) -> dict[str, Any]:
    weight = load_tensor(model_dir, weight_map, layer_idx, tensor_name)
    source = weight.to(dtype=torch.float32).contiguous()
    row_count, col_count = source.shape

    record: dict[str, Any] = {
        "layer_idx": layer_idx,
        "tensor_name": tensor_name,
        "nbits": nbits,
        "shape": [int(row_count), int(col_count)],
        "group_size": group_size,
        "global_steps": global_steps,
        "local_steps": local_steps,
        "iters": iters,
        "candidate_batch": candidate_batch,
    }

    if cpu_baseline:
        started = time.perf_counter()
        cpu_payload = current_cpu_quantize(source, nbits=nbits, group_size=group_size)
        cpu_s = time.perf_counter() - started
        cpu_quant = unpack_cpu_quant(cpu_payload, nbits, col_count, group_size)
        record["cpu_current"] = {
            "timing": {"quantize_s": cpu_s},
            "metrics": metrics_from_quant(source, cpu_quant, cpu_payload["scales"], cpu_payload["zeros"], group_size),
        }

    if backend == "torch":
        gpu_result, gpu_tensors = gpu_search_quantize(
            source,
            nbits=nbits,
            group_size=group_size,
            global_steps=global_steps,
            local_steps=local_steps,
            iters=iters,
            candidate_batch=candidate_batch,
            device=device,
        )
    elif backend == "rust-cuda":
        gpu_result, gpu_tensors = rust_cuda_search_quantize(
            source,
            nbits=nbits,
            group_size=group_size,
            global_steps=global_steps,
            local_steps=local_steps,
            iters=iters,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported backend={backend!r}")
    record["gpu_search"] = gpu_result
    record["gpu_search"]["backend"] = backend
    if cpu_baseline:
        cpu_rmse = record["cpu_current"]["metrics"]["rmse"]
        gpu_rmse = record["gpu_search"]["metrics"]["rmse"]
        record["gpu_vs_cpu_quality"] = {
            "rmse_delta": gpu_rmse - cpu_rmse,
            "rmse_ratio": gpu_rmse / cpu_rmse if cpu_rmse > 0.0 else None,
            "relative_rmse_delta": (
                record["gpu_search"]["metrics"]["relative_rmse"]
                - record["cpu_current"]["metrics"]["relative_rmse"]
            ),
            "packed_equal": bool(torch.equal(cpu_payload["packed"], gpu_tensors["packed"])),
            "scales_max_abs_diff": float((cpu_payload["scales"] - gpu_tensors["scales"]).abs().max().item()),
            "zeros_max_abs_diff": float((cpu_payload["zeros"] - gpu_tensors["zeros"]).abs().max().item()),
        }
        cpu_time = record["cpu_current"]["timing"]["quantize_s"]
        gpu_total = record["gpu_search"]["timing"]["total_with_transfer_pack_s"]
        record["speedup_vs_cpu_current"] = cpu_time / gpu_total if gpu_total > 0.0 else None
    return record


def main() -> None:
    require_dev_entrypoint()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="q122b", choices=sorted(MODEL_ALIASES))
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tensors", default="k_proj,o_proj,q_proj")
    parser.add_argument("--nbits", default="4", help="Comma-separated bit-widths: 4,6,8")
    parser.add_argument("--group-size", type=int, default=HQQ_DEFAULT_GROUP_SIZE)
    parser.add_argument("--global-steps", type=int, default=HQQ4_GLOBAL_STEPS)
    parser.add_argument("--local-steps", type=int, default=HQQ4_LOCAL_STEPS)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--candidate-batch", type=int, default=65)
    parser.add_argument("--backend", choices=["torch", "rust-cuda"], default="torch")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="logs/manual/phase2ff_hqq_gpu_search_probe_latest.json")
    parser.add_argument("--no-cpu-baseline", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser() if args.model_dir else MODEL_ALIASES[args.model]
    if not model_dir.is_dir():
        raise RuntimeError(f"Model directory not found: {model_dir}")
    weight_map = load_index(model_dir)

    results = {
        "model_dir": str(model_dir),
        "layer_idx": int(args.layer),
        "device": args.device,
        "prototype": (
            "rust_cuda_candidate_search_v1"
            if args.backend == "rust-cuda"
            else "torch_cuda_candidate_search_v1"
        ),
        "backend": args.backend,
        "note": "Diagnostic only; does not write HQQ cache artifacts or change runtime defaults.",
        "cases": [],
    }
    for nbits_raw in parse_csv(args.nbits):
        nbits = int(nbits_raw)
        for tensor_name in parse_csv(args.tensors):
            print(f"Running nbits={nbits} layer={args.layer} tensor={tensor_name}", flush=True)
            case = run_case(
                model_dir,
                weight_map,
                layer_idx=args.layer,
                tensor_name=tensor_name,
                nbits=nbits,
                group_size=args.group_size,
                global_steps=args.global_steps,
                local_steps=args.local_steps,
                iters=args.iters,
                candidate_batch=args.candidate_batch,
                device=args.device,
                cpu_baseline=not args.no_cpu_baseline,
                backend=args.backend,
            )
            results["cases"].append(case)
            print(json.dumps(case, sort_keys=True), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
