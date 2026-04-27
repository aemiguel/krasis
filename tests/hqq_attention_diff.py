#!/usr/bin/env python3
"""Offline HQQ attention artifact diff against BF16 safetensors source.

This script is intentionally runnable only through ./dev. It reads cached HQQ
attention artifacts and original local safetensors, then writes reporting-only
error metrics. It does not load Krasis runtime or modify artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


BEST_FIT_GRID_STEPS = 129
BEST_FIT_LOCAL_GRID_STEPS = 65


class StageTimer:
    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self.seconds: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        start = self._starts.pop(name)
        elapsed = time.perf_counter() - start
        self.seconds[name] = self.seconds.get(name, 0.0) + elapsed
        return elapsed

    def add(self, name: str, elapsed: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + float(elapsed)

    def snapshot(self) -> dict[str, float]:
        return {name: round(value, 6) for name, value in sorted(self.seconds.items())}


def require_dev_entrypoint() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("This script must be run via ./dev hqq-attn-diff")


def load_index_tensor(model_dir: Path, name: str) -> torch.Tensor:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"Missing safetensors index: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    try:
        shard_name = index["weight_map"][name]
    except KeyError as exc:
        raise RuntimeError(f"Tensor {name!r} not found in {index_path}") from exc
    shard_path = model_dir / shard_name
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name)
    return tensor.contiguous()


def load_safetensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    return tensors, metadata


def unpack_uint4(packed: torch.Tensor, cols: int) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    q = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    return q[:, :cols].contiguous()


def dequant_hqq4(tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    required = ("packed", "scales", "zeros", "orig_shape", "group_size", "axis", "nbits")
    missing = [name for name in required if name not in tensors]
    if missing:
        raise RuntimeError(f"HQQ artifact missing tensors: {', '.join(missing)}")

    rows, cols = [int(v) for v in tensors["orig_shape"].tolist()]
    group_size = int(tensors["group_size"][0].item())
    axis = int(tensors["axis"][0].item())
    nbits = int(tensors["nbits"][0].item())
    if axis != 1 or nbits != 4:
        raise RuntimeError(f"Unsupported HQQ artifact contract: axis={axis}, nbits={nbits}")

    packed = tensors["packed"].to(torch.uint8).contiguous()
    scales = tensors["scales"].to(torch.float32).contiguous()
    zeros = tensors["zeros"].to(torch.float32).contiguous()
    groups = math.ceil(cols / group_size)
    if tuple(scales.shape) != (rows, groups) or tuple(zeros.shape) != (rows, groups):
        raise RuntimeError(
            f"HQQ scale/zero shape mismatch: scales={tuple(scales.shape)} "
            f"zeros={tuple(zeros.shape)} expected={(rows, groups)}"
        )
    q = unpack_uint4(packed, groups * group_size)[:, :cols].to(torch.float32)
    scale_full = scales.repeat_interleave(group_size, dim=1)[:, :cols]
    zero_full = zeros.repeat_interleave(group_size, dim=1)[:, :cols]
    deq = (q - zero_full) * scale_full
    meta = {
        "rows": rows,
        "cols": cols,
        "group_size": group_size,
        "groups": groups,
        "axis": axis,
        "nbits": nbits,
        "packed_shape": list(packed.shape),
        "scales_shape": list(scales.shape),
        "zeros_shape": list(zeros.shape),
    }
    quant = {
        "qvalues": q.to(torch.uint8).contiguous(),
        "scales": scales,
        "zeros": zeros,
        "packed": packed,
    }
    return deq.contiguous(), meta, quant


def tensor_hash(t: torch.Tensor) -> str:
    cpu = t.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        data = cpu.view(torch.uint16).numpy().tobytes()
    else:
        data = cpu.numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bf16_word_to_f32(word: int) -> float:
    return struct.unpack(">f", ((word & 0xFFFF) << 16).to_bytes(4, "big"))[0]


def parse_input_row_positions(raw: str) -> set[int] | None:
    trimmed = raw.strip()
    if trimmed.lower() == "all" or trimmed == "*":
        return None
    return {int(part.strip()) for part in trimmed.split(",") if part.strip()}


def load_input_rows_from_trace(
    paths: list[Path],
    *,
    positions: set[int] | None,
    layer: int,
    tensor: str,
) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"event=input_row_full\b.*\bpos=(?P<pos>\d+)\b.*\blayer=(?P<layer>\d+)\b"
        r".*\btensor=(?P<tensor>\S+)\b.*\bn=(?P<n>\d+)\b.*\bhash64=(?P<hash>0x[0-9a-fA-F]+)\b"
        r".*\bbf16_hex=\[(?P<hex>[0-9a-fA-F,]*)\]"
    )
    matches: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                if "event=input_row_full" not in line:
                    continue
                match = pattern.search(line)
                if not match:
                    continue
                event_pos = int(match.group("pos"))
                if positions is not None and event_pos not in positions:
                    continue
                if int(match.group("layer")) != layer:
                    continue
                if match.group("tensor") != tensor:
                    continue
                raw_words = [item for item in match.group("hex").split(",") if item]
                words = [int(item, 16) for item in raw_words]
                expected_n = int(match.group("n"))
                if len(words) != expected_n:
                    raise RuntimeError(
                        f"Trace input row length mismatch at {path}:{line_no}: "
                        f"n={expected_n} words={len(words)}"
                    )
                matches.append({
                    "path": str(path),
                    "line": line_no,
                    "pos": event_pos,
                    "layer": layer,
                    "tensor": tensor,
                    "n": expected_n,
                    "hash64": match.group("hash"),
                    "values": torch.tensor([bf16_word_to_f32(word) for word in words], dtype=torch.float32),
                    "bf16_hex_sample": raw_words[:8],
                })
    if not matches:
        raise RuntimeError(
            f"No input_row_full event found for positions={positions or 'all'} layer={layer} tensor={tensor}"
        )
    collapsed: list[dict[str, Any]] = []
    for event in matches:
        if collapsed and collapsed[-1]["pos"] == event["pos"] and collapsed[-1]["hash64"] == event["hash64"]:
            collapsed[-1]["duplicate_count"] += 1
            collapsed[-1].setdefault("duplicate_locations", []).append({
                "path": event["path"],
                "line": event["line"],
            })
            continue
        event["duplicate_count"] = 1
        event["sequence_index"] = len(collapsed)
        collapsed.append(event)
    return collapsed


def parse_all_input_row_events(path: Path) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"event=input_row_full\b.*\bpos=(?P<pos>\d+)\b.*\blayer=(?P<layer>\d+)\b"
        r".*\btensor=(?P<tensor>\S+)\b.*\bn=(?P<n>\d+)\b.*\bhash64=(?P<hash>0x[0-9a-fA-F]+)\b"
        r".*\bbf16_hex=\[(?P<hex>[0-9a-fA-F,]*)\]"
    )
    events: list[dict[str, Any]] = []
    field_counts: dict[tuple[int, str], int] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if "event=input_row_full" not in line:
                continue
            match = pattern.search(line)
            if not match:
                continue
            raw_words = [item for item in match.group("hex").split(",") if item]
            expected_n = int(match.group("n"))
            if len(raw_words) != expected_n:
                raise RuntimeError(
                    f"Trace input row length mismatch at {path}:{line_no}: "
                    f"n={expected_n} words={len(raw_words)}"
                )
            layer = int(match.group("layer"))
            tensor = match.group("tensor")
            key = (layer, tensor)
            ordinal = field_counts.get(key, 0)
            field_counts[key] = ordinal + 1
            events.append({
                "path": str(path),
                "line": line_no,
                "raw_line": line.rstrip("\n"),
                "global_ordinal": len(events),
                "field_ordinal": ordinal,
                "pos": int(match.group("pos")),
                "layer": layer,
                "tensor": tensor,
                "n": expected_n,
                "hash64": match.group("hash"),
                "values": torch.tensor(
                    [bf16_word_to_f32(int(word, 16)) for word in raw_words],
                    dtype=torch.float32,
                ),
                "bf16_hex_sample": raw_words[:8],
            })
    return events


def parse_reference_case_spans(paths: list[Path]) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"event=reference_case_boundary\b.*\bphase=(?P<phase>\S+)\b"
        r".*\bcase_key=(?P<case_key>\S+)\b.*\bconv_idx=(?P<conv_idx>\d+)\b"
        r".*\bturn=(?P<turn>\d+)\b.*\bprompt_sequence_index=(?P<prompt_sequence_index>\d+)\b"
        r".*\binput_tokens_count=(?P<input_tokens_count>\d+)\b.*\bpositions=\[(?P<positions>[0-9,]*)\]"
    )
    spans: list[dict[str, Any]] = []
    open_spans: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                if "event=reference_case_boundary" not in line:
                    continue
                match = pattern.search(line)
                if not match:
                    continue
                phase = match.group("phase")
                case_key = match.group("case_key")
                prompt_sequence_index = int(match.group("prompt_sequence_index"))
                key = (str(path), case_key, prompt_sequence_index)
                positions = [
                    int(item)
                    for item in match.group("positions").split(",")
                    if item.strip()
                ]
                payload = {
                    "path": str(path),
                    "line": line_no,
                    "phase": phase,
                    "case_key": case_key,
                    "conv_idx": int(match.group("conv_idx")),
                    "turn": int(match.group("turn")),
                    "prompt_sequence_index": prompt_sequence_index,
                    "input_tokens_count": int(match.group("input_tokens_count")),
                    "positions": positions,
                }
                if phase == "start":
                    open_spans[key] = payload
                    continue
                start = open_spans.pop(key, None)
                if start is None:
                    continue
                spans.append({
                    **{k: v for k, v in start.items() if k not in {"line", "phase"}},
                    "start_line": start["line"],
                    "end_line": line_no,
                    "end_phase": phase,
                })
    return spans


def load_case_summary_doc(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, list):
        if not doc:
            raise RuntimeError(f"Case summary array is empty: {path}")
        doc = doc[0]
    if not isinstance(doc, dict) or not isinstance(doc.get("prompt_results"), list):
        raise RuntimeError(f"Case summary has no prompt_results array: {path}")
    return doc


def load_input_row_from_trace(path: Path, *, pos: int, layer: int, tensor: str) -> dict[str, Any]:
    rows = load_input_rows_from_trace([path], positions={pos}, layer=layer, tensor=tensor)
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one input row for pos={pos}, found {len(rows)}")
    return rows[0]


def load_case_map(path: Path) -> list[dict[str, Any]]:
    summary = load_case_summary_doc(path)
    out = []
    for index, result in enumerate(summary.get("prompt_results", [])):
        trace = result.get("trace_input_row") or {}
        positions = trace.get("positions") or [trace.get("pos")]
        for pos_index, pos in enumerate(positions):
            out.append({
                "sequence_index": len(out),
                "prompt_sequence_index": index,
                "position_sequence_index": pos_index,
                "case_key": trace.get("case_key", f"{result.get('conv_idx')}:{result.get('turn', 1) - 1}"),
                "conv_idx": result.get("conv_idx"),
                "turn": result.get("turn"),
                "pos": pos,
                "input_tokens_count": trace.get("input_tokens_count"),
                "prompt_prefix": trace.get("prompt_prefix", result.get("prompt", "")[:120]),
            })
    return out


def align_input_rows_to_cases_detailed(
    input_row_events: list[dict[str, Any]],
    case_map: list[dict[str, Any]],
) -> dict[str, Any]:
    aligned: list[dict[str, Any]] = []
    missing_cases: list[dict[str, Any]] = []
    unused_events: list[dict[str, Any]] = []
    if not case_map:
        return {
            "aligned": [
                {
                    "event": event,
                    "case": None,
                    "align_status": "no_case_summary",
                }
                for event in input_row_events
            ],
            "missing_cases": [],
            "unused_events": [],
        }

    event_index = 0
    used_event_indexes: set[int] = set()
    for case in case_map:
        expected_pos = case.get("pos")
        found_index = None
        for idx in range(event_index, len(input_row_events)):
            if input_row_events[idx].get("pos") == expected_pos:
                found_index = idx
                break
        if found_index is None:
            missing_cases.append(case)
            continue
        aligned.append(
            {
                "event": input_row_events[found_index],
                "case": case,
                "align_status": "matched_by_ordered_position",
            }
        )
        used_event_indexes.add(found_index)
        event_index = found_index + 1

    for idx, event in enumerate(input_row_events):
        if idx not in used_event_indexes:
            unused_events.append(event)
    return {
        "aligned": aligned,
        "missing_cases": missing_cases,
        "unused_events": unused_events,
    }


def align_input_rows_to_case_spans(
    input_row_events: list[dict[str, Any]],
    spans: list[dict[str, Any]],
) -> dict[str, Any]:
    aligned: list[dict[str, Any]] = []
    unused_events: list[dict[str, Any]] = []
    span_hits: dict[int, int] = {}
    for event in input_row_events:
        event_path = event.get("path")
        event_line = int(event.get("line", -1))
        event_pos = int(event.get("pos", -1))
        match_span = None
        for index, span in enumerate(spans):
            if span.get("path") != event_path:
                continue
            if int(span["start_line"]) < event_line < int(span["end_line"]):
                if event_pos in set(span.get("positions") or []):
                    match_span = (index, span)
                    break
        if match_span is None:
            unused_events.append(event)
            continue
        span_index, span = match_span
        span_hits[span_index] = span_hits.get(span_index, 0) + 1
        positions = list(span.get("positions") or [])
        try:
            position_sequence_index = positions.index(event_pos)
        except ValueError:
            position_sequence_index = None
        aligned.append({
            "event": event,
            "case": {
                "case_key": span.get("case_key"),
                "conv_idx": span.get("conv_idx"),
                "turn": span.get("turn"),
                "prompt_sequence_index": span.get("prompt_sequence_index"),
                "position_sequence_index": position_sequence_index,
                "pos": event_pos,
                "input_tokens_count": span.get("input_tokens_count"),
                "prompt_prefix": None,
            },
            "align_status": "matched_by_explicit_case_span",
            "span": span,
        })
    missing_spans = [
        span
        for index, span in enumerate(spans)
        if span_hits.get(index, 0) == 0
    ]
    return {
        "aligned": aligned,
        "missing_cases": missing_spans,
        "unused_events": unused_events,
        "spans": spans,
    }


def align_input_rows_to_cases(
    input_row_events: list[dict[str, Any]],
    case_map: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    if not case_map:
        return [(event, None) for event in input_row_events]
    aligned = []
    event_index = 0
    for case in case_map:
        expected_pos = case.get("pos")
        found_index = None
        for idx in range(event_index, len(input_row_events)):
            if input_row_events[idx].get("pos") == expected_pos:
                found_index = idx
                break
        if found_index is None:
            return [
                (event, case_map[idx] if idx < len(case_map) else None)
                for idx, event in enumerate(input_row_events)
            ]
        aligned.append((input_row_events[found_index], case))
        event_index = found_index + 1
    return aligned


def metrics(source: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    src = source.to(torch.float32).contiguous()
    got = actual.to(torch.float32).contiguous()
    if src.shape != got.shape:
        raise RuntimeError(f"Metric shape mismatch: source={tuple(src.shape)} actual={tuple(got.shape)}")
    diff = got - src
    abs_diff = diff.abs()
    sq = diff * diff
    src_sq = src * src
    rms = float(torch.sqrt(sq.mean()).item())
    source_rms = float(torch.sqrt(src_sq.mean()).item())
    return {
        "shape": list(src.shape),
        "numel": int(src.numel()),
        "source_sha256": tensor_hash(src),
        "actual_sha256": tensor_hash(got),
        "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_abs": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "rms": rms,
        "source_rms": source_rms,
        "relative_rms": rms / max(source_rms, 1e-12),
        "mean_signed": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def error_metrics_from_diff(source: torch.Tensor, diff: torch.Tensor) -> dict[str, Any]:
    src = source.to(torch.float32).contiguous()
    err = diff.to(torch.float32).contiguous()
    if src.shape != err.shape:
        raise RuntimeError(f"Error metric shape mismatch: source={tuple(src.shape)} diff={tuple(err.shape)}")
    abs_diff = err.abs()
    sq = err * err
    src_sq = src * src
    rms = float(torch.sqrt(sq.mean()).item())
    source_rms = float(torch.sqrt(src_sq.mean()).item())
    return {
        "shape": list(src.shape),
        "numel": int(src.numel()),
        "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_abs": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "rms": rms,
        "source_rms": source_rms,
        "relative_rms": rms / max(source_rms, 1e-12),
        "mean_signed": float(err.mean().item()) if err.numel() else 0.0,
        "error_sq_sum": float(sq.sum().item()),
        "source_sq_sum": float(src_sq.sum().item()),
    }


def qkvz_row_info(row: int, nk: int, dk: int, hr: int, dv: int) -> dict[str, int | str]:
    group_dim = (2 * dk) + (2 * hr * dv)
    key_head = row // group_dim
    offset = row % group_dim
    if offset < dk:
        return {
            "slice": "q",
            "key_head": int(key_head),
            "slice_local_row": int((key_head * dk) + offset),
            "row_in_key_head_slice": int(offset),
        }
    if offset < 2 * dk:
        return {
            "slice": "k",
            "key_head": int(key_head),
            "slice_local_row": int((key_head * dk) + (offset - dk)),
            "row_in_key_head_slice": int(offset - dk),
        }
    v_rows = hr * dv
    if offset < 2 * dk + v_rows:
        return {
            "slice": "v",
            "key_head": int(key_head),
            "slice_local_row": int((key_head * v_rows) + (offset - 2 * dk)),
            "row_in_key_head_slice": int(offset - 2 * dk),
        }
    return {
        "slice": "z",
        "key_head": int(key_head),
        "slice_local_row": int((key_head * v_rows) + (offset - 2 * dk - v_rows)),
        "row_in_key_head_slice": int(offset - 2 * dk - v_rows),
    }


def ba_row_info(row: int, nk: int, hr: int) -> dict[str, int | str]:
    group_dim = 2 * hr
    key_head = row // group_dim
    offset = row % group_dim
    if offset < hr:
        return {
            "slice": "b",
            "key_head": int(key_head),
            "slice_local_row": int((key_head * hr) + offset),
            "row_in_key_head_slice": int(offset),
        }
    return {
        "slice": "a",
        "key_head": int(key_head),
        "slice_local_row": int((key_head * hr) + (offset - hr)),
        "row_in_key_head_slice": int(offset - hr),
    }


def source_slice_names(source_meta: dict[str, Any]) -> list[str]:
    return [str(name) for name in source_meta.get("slice_names", ["q", "k", "v", "z"])]


def source_row_info(row: int, source_meta: dict[str, Any]) -> dict[str, int | str]:
    tensor_name = str(source_meta.get("tensor_name", "in_proj_qkvz"))
    if tensor_name == "in_proj_ba":
        return ba_row_info(
            row,
            int(source_meta["nk"]),
            int(source_meta["head_ratio"]),
        )
    if tensor_name == "out_proj":
        return {
            "slice": "out",
            "key_head": -1,
            "slice_local_row": int(row),
            "row_in_key_head_slice": int(row),
        }
    if tensor_name == "fused_qkv":
        offset = 0
        for slice_name, rows in (
            ("q", int(source_meta["q_rows"])),
            ("k", int(source_meta["k_rows"])),
            ("v", int(source_meta["v_rows"])),
        ):
            if row < offset + rows:
                return {
                    "slice": slice_name,
                    "key_head": -1,
                    "slice_local_row": int(row - offset),
                    "row_in_key_head_slice": int(row - offset),
                }
            offset += rows
    if tensor_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return {
            "slice": tensor_name.removesuffix("_proj"),
            "key_head": -1,
            "slice_local_row": int(row),
            "row_in_key_head_slice": int(row),
        }
    return qkvz_row_info(
        row,
        int(source_meta["nk"]),
        int(source_meta["dk"]),
        int(source_meta["head_ratio"]),
        int(source_meta["dv"]),
    )


def row_attribution(
    source: torch.Tensor,
    actual: torch.Tensor,
    source_meta: dict[str, Any],
    *,
    top_n: int,
    sample_cols: int,
) -> dict[str, Any]:
    src = source.to(torch.float32).contiguous()
    got = actual.to(torch.float32).contiguous()
    diff = got - src
    abs_diff = diff.abs()
    row_rms = torch.sqrt((diff * diff).mean(dim=1))
    row_source_rms = torch.sqrt((src * src).mean(dim=1))
    row_relative_rms = row_rms / torch.clamp(row_source_rms, min=1e-12)

    def make_row(row: int) -> dict[str, Any]:
        max_abs, max_col = abs_diff[row].max(dim=0)
        info = source_row_info(row, source_meta)
        return {
            "global_row": int(row),
            **info,
            "rms": float(row_rms[row].item()),
            "source_rms": float(row_source_rms[row].item()),
            "relative_rms": float(row_relative_rms[row].item()),
            "max_abs": float(max_abs.item()),
            "max_abs_col": int(max_col.item()),
            "mean_abs": float(abs_diff[row].mean().item()),
            "samples": {
                "source": [float(v) for v in src[row, :sample_cols].tolist()],
                "dequant": [float(v) for v in got[row, :sample_cols].tolist()],
                "diff": [float(v) for v in diff[row, :sample_cols].tolist()],
                "at_max_abs": {
                    "col": int(max_col.item()),
                    "source": float(src[row, max_col].item()),
                    "dequant": float(got[row, max_col].item()),
                    "diff": float(diff[row, max_col].item()),
                },
            },
        }

    n = min(top_n, src.shape[0])
    top_rms_rows = torch.topk(row_rms, k=n).indices.tolist()
    top_relative_rows = torch.topk(row_relative_rms, k=n).indices.tolist()
    return {
        "top_by_rms": [make_row(int(row)) for row in top_rms_rows],
        "top_by_relative_rms": [make_row(int(row)) for row in top_relative_rows],
    }


def group_attribution(
    source: torch.Tensor,
    actual: torch.Tensor,
    source_meta: dict[str, Any],
    hqq_meta: dict[str, Any],
    *,
    top_n: int,
    sample_cols: int,
) -> dict[str, Any]:
    src = source.to(torch.float32).contiguous()
    got = actual.to(torch.float32).contiguous()
    diff = got - src
    group_size = int(hqq_meta["group_size"])
    groups = int(hqq_meta["groups"])
    cols = int(hqq_meta["cols"])
    if int(hqq_meta["axis"]) != 1:
        raise RuntimeError(f"Group attribution only supports axis=1, got {hqq_meta['axis']}")
    if groups != math.ceil(cols / group_size):
        raise RuntimeError(f"Group metadata mismatch: groups={groups} cols={cols} group_size={group_size}")

    records: list[dict[str, Any]] = []
    for row in range(src.shape[0]):
        row_info = source_row_info(row, source_meta)
        for group_idx in range(groups):
            start = group_idx * group_size
            end = min(start + group_size, cols)
            src_group = src[row, start:end]
            got_group = got[row, start:end]
            diff_group = diff[row, start:end]
            metric = error_metrics_from_diff(src_group, diff_group)
            records.append({
                "global_row": int(row),
                **row_info,
                "hqq_group_index": int(group_idx),
                "col_start": int(start),
                "col_end": int(end),
                **metric,
            })

    by_rms = sorted(records, key=lambda item: item["rms"], reverse=True)[:top_n]
    by_relative = sorted(records, key=lambda item: item["relative_rms"], reverse=True)[:top_n]

    def add_samples(item: dict[str, Any]) -> dict[str, Any]:
        row = int(item["global_row"])
        start = int(item["col_start"])
        end = min(int(item["col_end"]), start + sample_cols)
        sampled = dict(item)
        sampled["samples"] = {
            "source": [float(v) for v in src[row, start:end].tolist()],
            "dequant": [float(v) for v in got[row, start:end].tolist()],
            "diff": [float(v) for v in diff[row, start:end].tolist()],
        }
        return sampled

    return {
        "axis": int(hqq_meta["axis"]),
        "group_size": group_size,
        "groups_per_row": groups,
        "top_by_rms": [add_samples(item) for item in by_rms],
        "top_by_relative_rms": [add_samples(item) for item in by_relative],
    }


def quantize_with_affine(
    source: torch.Tensor,
    scale: float,
    zero: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    safe_scale = max(float(scale), 1e-8)
    raw_q = (source.to(torch.float32) / safe_scale) + float(zero)
    q = torch.round(raw_q).clamp(0, 15).to(torch.uint8)
    dequant = (q.to(torch.float32) - float(zero)) * safe_scale
    return q, dequant, raw_q


def minmax_affine_params(source: torch.Tensor) -> tuple[float, float]:
    minv = float(source.min().item())
    maxv = float(source.max().item())
    scale = max((maxv - minv) / 15.0, 1e-8)
    zero = min(max(-minv / scale, 0.0), 15.0)
    return scale, zero


def symmetric_affine_params(source: torch.Tensor) -> tuple[float, float]:
    amax = float(source.abs().max().item())
    scale = max((2.0 * amax) / 15.0, 1e-8)
    return scale, 7.5


def solve_fixed_zero_vector(source: torch.Tensor, zero: float, scale_seed: float) -> tuple[torch.Tensor, float]:
    scale = max(float(scale_seed), 1e-8)
    zero_value = float(zero)
    src = source.to(torch.float32).contiguous()
    for _ in range(6):
        q = torch.round(src / scale + zero_value).clamp(0, 15)
        centered = q - zero_value
        denom = float((centered * centered).sum().item())
        numer = float((src * centered).sum().item())
        if denom > 1e-12:
            scale = max(numer / denom, 1e-8)
    q = torch.round(src / scale + zero_value).clamp(0, 15).to(torch.uint8)
    return q, scale


def solve_fixed_zero_weighted_vector(
    source: torch.Tensor,
    weights: torch.Tensor,
    zero: float,
    scale_seed: float,
    *,
    min_scale: float = 1e-8,
) -> tuple[torch.Tensor, float]:
    scale = max(float(scale_seed), float(min_scale), 1e-8)
    zero_value = float(zero)
    src = source.to(torch.float32).contiguous()
    w = weights.to(torch.float32).contiguous()
    for _ in range(8):
        q = torch.round(src / scale + zero_value).clamp(0, 15)
        centered = q - zero_value
        denom = float((w * centered * centered).sum().item())
        numer = float((w * src * centered).sum().item())
        if denom > 1e-12:
            scale = max(numer / denom, float(min_scale), 1e-8)
    q = torch.round(src / scale + zero_value).clamp(0, 15).to(torch.uint8)
    return q, scale


def best_fit_affine_params(source: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    src = source.to(torch.float32).contiguous()
    range_scale, _ = minmax_affine_params(src)
    abs_scale, _ = symmetric_affine_params(src)
    best_rmse = float("inf")
    best_q = torch.zeros_like(src, dtype=torch.uint8)
    best_scale = range_scale
    best_zero = 0.0

    def consider(zero_values: torch.Tensor, scale_seed: float) -> None:
        nonlocal best_rmse, best_q, best_scale, best_zero
        for zero_tensor in zero_values:
            zero = float(zero_tensor.item())
            q, scale = solve_fixed_zero_vector(src, zero, scale_seed)
            dequant = (q.to(torch.float32) - zero) * scale
            rmse = float(torch.sqrt(torch.mean((dequant - src) ** 2)).item())
            if rmse < best_rmse:
                best_rmse = rmse
                best_q = q
                best_scale = scale
                best_zero = zero

    global_grid = torch.linspace(0.0, 15.0, steps=BEST_FIT_GRID_STEPS, dtype=torch.float32)
    consider(global_grid, range_scale)
    consider(global_grid, abs_scale)

    local_min = max(0.0, best_zero - 0.5)
    local_max = min(15.0, best_zero + 0.5)
    local_grid = torch.linspace(local_min, local_max, steps=BEST_FIT_LOCAL_GRID_STEPS, dtype=torch.float32)
    consider(local_grid, best_scale)
    return best_scale, best_zero, best_q


def affine_candidate_score(source: torch.Tensor, input_group: torch.Tensor, scale: float, zero: float, q: torch.Tensor) -> float:
    dequant = (q.to(torch.float32) - float(zero)) * max(float(scale), 1e-8)
    contrib = input_group.to(torch.float32) * (dequant - source.to(torch.float32))
    return float(torch.sqrt(torch.mean(contrib * contrib)).item())


def activation_weighted_affine_params(source: torch.Tensor, input_group: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    src = source.to(torch.float32).contiguous()
    inp = input_group.to(torch.float32).contiguous()
    weights = inp * inp
    if float(weights.sum().item()) <= 1e-20:
        scale, zero, q = best_fit_affine_params(src)
        return scale, zero, q

    range_scale, _ = minmax_affine_params(src)
    abs_scale, _ = symmetric_affine_params(src)
    best_score = float("inf")
    best_q = torch.zeros_like(src, dtype=torch.uint8)
    best_scale = range_scale
    best_zero = 0.0

    def consider(zero_values: torch.Tensor, scale_seed: float) -> None:
        nonlocal best_score, best_q, best_scale, best_zero
        for zero_tensor in zero_values:
            zero = float(zero_tensor.item())
            q, scale = solve_fixed_zero_weighted_vector(src, weights, zero, scale_seed)
            score = affine_candidate_score(src, inp, scale, zero, q)
            if score < best_score:
                best_score = score
                best_q = q
                best_scale = scale
                best_zero = zero

    global_grid = torch.linspace(0.0, 15.0, steps=BEST_FIT_GRID_STEPS, dtype=torch.float32)
    consider(global_grid, range_scale)
    consider(global_grid, abs_scale)
    local_min = max(0.0, best_zero - 0.5)
    local_max = min(15.0, best_zero + 0.5)
    local_grid = torch.linspace(local_min, local_max, steps=BEST_FIT_LOCAL_GRID_STEPS, dtype=torch.float32)
    consider(local_grid, best_scale)
    return best_scale, best_zero, best_q


def protected_scale_floor(source: torch.Tensor, protected_indices: torch.Tensor, zero: float) -> float:
    src = source.to(torch.float32)
    zero_value = float(zero)
    min_scale = 1e-8
    for idx in protected_indices.to(torch.long).tolist():
        value = float(src[idx].item())
        if value > 0.0:
            denom = 15.0 - zero_value
            if denom <= 1e-8:
                return float("inf")
            min_scale = max(min_scale, value / denom)
        elif value < 0.0:
            denom = zero_value
            if denom <= 1e-8:
                return float("inf")
            min_scale = max(min_scale, -value / denom)
    return min_scale


def activation_clip_free_affine_params(
    source: torch.Tensor,
    input_group: torch.Tensor,
    *,
    protect_top_k: int = 8,
) -> tuple[float, float, torch.Tensor, dict[str, Any]]:
    src = source.to(torch.float32).contiguous()
    inp = input_group.to(torch.float32).contiguous()
    weights = inp * inp
    abs_input = inp.abs()
    max_input = float(abs_input.max().item()) if abs_input.numel() else 0.0
    threshold = max_input * 0.25
    protected_mask = abs_input >= threshold
    protected_indices = torch.nonzero(protected_mask, as_tuple=False).flatten()
    if protected_indices.numel() < min(protect_top_k, src.numel()):
        top = torch.topk(abs_input, k=min(protect_top_k, src.numel())).indices
        protected_indices = torch.unique(torch.cat([protected_indices, top])).to(torch.long)
    protected_indices = protected_indices.to(torch.long)

    range_scale, _ = minmax_affine_params(src)
    abs_scale, _ = symmetric_affine_params(src)
    best_score = float("inf")
    best_q = torch.zeros_like(src, dtype=torch.uint8)
    best_scale = range_scale
    best_zero = 0.0
    best_floor = 1e-8
    candidates_seen = 0

    def consider(zero_values: torch.Tensor, scale_seed: float) -> None:
        nonlocal best_score, best_q, best_scale, best_zero, best_floor, candidates_seen
        for zero_tensor in zero_values:
            zero = float(zero_tensor.item())
            floor = protected_scale_floor(src, protected_indices, zero)
            if not math.isfinite(floor):
                continue
            q, scale = solve_fixed_zero_weighted_vector(src, weights, zero, scale_seed, min_scale=floor)
            raw_q = src / max(scale, 1e-8) + zero
            if bool((raw_q.index_select(0, protected_indices) < -1e-6).any().item()):
                continue
            if bool((raw_q.index_select(0, protected_indices) > 15.0 + 1e-6).any().item()):
                continue
            candidates_seen += 1
            score = affine_candidate_score(src, inp, scale, zero, q)
            if score < best_score:
                best_score = score
                best_q = q
                best_scale = scale
                best_zero = zero
                best_floor = floor

    global_grid = torch.linspace(0.0, 15.0, steps=BEST_FIT_GRID_STEPS, dtype=torch.float32)
    consider(global_grid, range_scale)
    consider(global_grid, abs_scale)
    if candidates_seen == 0:
        scale, zero = minmax_affine_params(src)
        q, _, _ = quantize_with_affine(src, scale, zero)
        best_scale = scale
        best_zero = zero
        best_q = q
        best_floor = scale
    else:
        local_min = max(0.0, best_zero - 0.5)
        local_max = min(15.0, best_zero + 0.5)
        local_grid = torch.linspace(local_min, local_max, steps=BEST_FIT_LOCAL_GRID_STEPS, dtype=torch.float32)
        consider(local_grid, best_scale)

    return best_scale, best_zero, best_q, {
        "protected_local_cols": [int(v) for v in protected_indices.tolist()],
        "protected_count": int(protected_indices.numel()),
        "input_abs_threshold": float(threshold),
        "scale_floor": float(best_floor),
        "candidate_count": int(candidates_seen),
    }


def quantizer_candidate_metrics(
    name: str,
    source: torch.Tensor,
    input_group: torch.Tensor,
    *,
    scale: float,
    zero: float,
    q_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    if q_override is None:
        q, dequant, raw_q = quantize_with_affine(source, scale, zero)
    else:
        q = q_override.to(torch.uint8).contiguous()
        safe_scale = max(float(scale), 1e-8)
        raw_q = (source.to(torch.float32) / safe_scale) + float(zero)
        dequant = (q.to(torch.float32) - float(zero)) * safe_scale
    diff = dequant - source.to(torch.float32)
    contrib = input_group.to(torch.float32) * diff
    abs_diff = diff.abs()
    abs_contrib = contrib.abs()
    hist = torch.bincount(q.to(torch.int64), minlength=16)[:16]
    return {
        "name": name,
        "scale": float(scale),
        "zero": float(zero),
        "quant_step_size": float(scale),
        "weight_error_rms": float(torch.sqrt((diff * diff).mean()).item()),
        "weight_error_max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "weight_error_mean_abs": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "projection_contribution_signed_sum": float(contrib.sum().item()),
        "projection_contribution_abs_sum": float(abs_contrib.sum().item()),
        "projection_contribution_rms": float(torch.sqrt((contrib * contrib).mean()).item()),
        "projection_contribution_max_abs": float(abs_contrib.max().item()) if abs_contrib.numel() else 0.0,
        "qvalue_min": int(q.min().item()) if q.numel() else 0,
        "qvalue_max": int(q.max().item()) if q.numel() else 0,
        "qvalue_histogram": [int(v) for v in hist.tolist()],
        "qvalue_saturation_zero_count": int((q == 0).sum().item()),
        "qvalue_saturation_max_count": int((q == 15).sum().item()),
        "clip_low_count": int((raw_q < 0.0).sum().item()),
        "clip_high_count": int((raw_q > 15.0).sum().item()),
    }


def quantizer_choice_diagnostics(
    source: torch.Tensor,
    actual: torch.Tensor,
    input_row: torch.Tensor,
    source_meta: dict[str, Any],
    hqq_meta: dict[str, Any],
    hqq_quant: dict[str, torch.Tensor],
    group_records: list[dict[str, Any]],
    *,
    focus_row: int | None,
    focus_group: int | None,
    sample_cols: int,
    timer: StageTimer | None = None,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    src = source.to(torch.float32).contiguous()
    got = actual.to(torch.float32).contiguous()
    row = input_row.to(torch.float32).contiguous()
    group_size = int(hqq_meta["group_size"])
    qvalues = hqq_quant["qvalues"].to(torch.uint8).contiguous()
    scales = hqq_quant["scales"].to(torch.float32).contiguous()
    zeros = hqq_quant["zeros"].to(torch.float32).contiguous()

    seen: set[tuple[int, int]] = set()
    targets: list[tuple[int, int]] = []
    for item in group_records:
        key = (int(item["global_row"]), int(item["hqq_group_index"]))
        if key not in seen:
            seen.add(key)
            targets.append(key)
    if focus_row is not None and focus_group is not None and (focus_row, focus_group) not in seen:
        targets.insert(0, (focus_row, focus_group))

    records = []
    for row_idx, group_idx in targets:
        start = group_idx * group_size
        end = min(start + group_size, src.shape[1])
        src_group = src[row_idx, start:end]
        got_group = got[row_idx, start:end]
        input_group = row[start:end]
        cached_q = qvalues[row_idx, start:end]
        cached_scale = float(scales[row_idx, group_idx].item())
        cached_zero = float(zeros[row_idx, group_idx].item())
        expected_q, cached_formula_dequant, raw_q = quantize_with_affine(src_group, cached_scale, cached_zero)
        mismatch = expected_q.to(torch.int16) - cached_q.to(torch.int16)
        cached_dequant_from_q = (cached_q.to(torch.float32) - cached_zero) * max(cached_scale, 1e-8)
        minmax_scale, minmax_zero = minmax_affine_params(src_group)
        best_scale, best_zero, best_q = best_fit_affine_params(src_group)
        weighted_scale, weighted_zero, weighted_q = activation_weighted_affine_params(src_group, input_group)
        clip_free_scale, clip_free_zero, clip_free_q, clip_free_meta = activation_clip_free_affine_params(
            src_group,
            input_group,
        )
        cached_candidate = quantizer_candidate_metrics(
            "cached_hqq",
            src_group,
            input_group,
            scale=cached_scale,
            zero=cached_zero,
            q_override=cached_q,
        )
        minmax_candidate = quantizer_candidate_metrics(
            "minmax_affine",
            src_group,
            input_group,
            scale=minmax_scale,
            zero=minmax_zero,
        )
        best_candidate = quantizer_candidate_metrics(
            "offline_best_fit_affine",
            src_group,
            input_group,
            scale=best_scale,
            zero=best_zero,
            q_override=best_q,
        )
        weighted_candidate = quantizer_candidate_metrics(
            "activation_weighted_affine",
            src_group,
            input_group,
            scale=weighted_scale,
            zero=weighted_zero,
            q_override=weighted_q,
        )
        clip_free_candidate = quantizer_candidate_metrics(
            "activation_weighted_clip_free",
            src_group,
            input_group,
            scale=clip_free_scale,
            zero=clip_free_zero,
            q_override=clip_free_q,
        )
        clip_free_candidate["protected_columns"] = clip_free_meta
        candidates = [
            cached_candidate,
            minmax_candidate,
            best_candidate,
            weighted_candidate,
            clip_free_candidate,
        ]
        best_by_contribution = min(candidates, key=lambda item: abs(float(item["projection_contribution_signed_sum"])))
        best_by_weight_rms = min(candidates, key=lambda item: float(item["weight_error_rms"]))

        abs_contrib = (input_group * (got_group - src_group)).abs()
        order = torch.argsort(abs_contrib, descending=True)[:sample_cols]
        worst_columns = []
        for idx in order.tolist():
            col = start + int(idx)
            raw_value = float(raw_q[idx].item())
            cached_qv = int(cached_q[idx].item())
            expected_qv = int(expected_q[idx].item())
            worst_columns.append({
                "col": int(col),
                "group_local_col": int(idx),
                "input": float(row[col].item()),
                "source": float(src_group[idx].item()),
                "cached_dequant": float(got_group[idx].item()),
                "weight_error": float((got_group[idx] - src_group[idx]).item()),
                "contribution": float((row[col] * (got_group[idx] - src_group[idx])).item()),
                "abs_contribution": float(abs_contrib[idx].item()),
                "cached_qvalue": cached_qv,
                "expected_qvalue_from_cached_scale_zero": expected_qv,
                "raw_q_before_round_clamp": raw_value,
                "is_clipped_low": raw_value < 0.0,
                "is_clipped_high": raw_value > 15.0,
            })

        clip_count = int((raw_q < 0.0).sum().item() + (raw_q > 15.0).sum().item())
        mismatch_count = int((mismatch != 0).sum().item())
        cached_abs = abs(float(cached_candidate["projection_contribution_signed_sum"]))
        best_abs = abs(float(best_candidate["projection_contribution_signed_sum"]))
        minmax_abs = abs(float(minmax_candidate["projection_contribution_signed_sum"]))
        weighted_abs = abs(float(weighted_candidate["projection_contribution_signed_sum"]))
        clip_free_abs = abs(float(clip_free_candidate["projection_contribution_signed_sum"]))
        if mismatch_count:
            ranked_cause = "qvalue_packing_or_interpretation_mismatch"
        elif min(weighted_abs, clip_free_abs) < cached_abs * 0.5:
            ranked_cause = "activation_weighted_scale_zero_would_materially_reduce_output_error"
        elif clip_count and any(col["is_clipped_low"] or col["is_clipped_high"] for col in worst_columns[:3]):
            ranked_cause = "saturation_or_clipping_on_high_activation_columns"
        elif best_abs < cached_abs * 0.75:
            ranked_cause = "scale_zero_choice_suboptimal_for_this_input_row"
        elif minmax_abs < cached_abs * 0.75:
            ranked_cause = "current_hqq_fit_worse_than_simple_minmax_for_this_input_row"
        else:
            ranked_cause = "coarse_4bit_group_resolution_with_activation_alignment"

        records.append({
            "global_row": int(row_idx),
            **source_row_info(row_idx, source_meta),
            "hqq_group_index": int(group_idx),
            "col_start": int(start),
            "col_end": int(end),
            "source_min": float(src_group.min().item()),
            "source_max": float(src_group.max().item()),
            "source_range": float((src_group.max() - src_group.min()).item()),
            "cached_scale": cached_scale,
            "cached_zero": cached_zero,
            "cached_quant_step_size": cached_scale,
            "cached_qvalue_reconstruction": {
                "mismatch_count": mismatch_count,
                "max_abs_mismatch": int(mismatch.abs().max().item()) if mismatch.numel() else 0,
                "formula_dequant_max_abs_delta_vs_cached_dequant": float(
                    (cached_formula_dequant - got_group).abs().max().item()
                ),
                "cached_q_dequant_max_abs_delta_vs_cached_dequant": float(
                    (cached_dequant_from_q - got_group).abs().max().item()
                ),
            },
            "cached_raw_q_min": float(raw_q.min().item()),
            "cached_raw_q_max": float(raw_q.max().item()),
            "cached_clip_low_count": int((raw_q < 0.0).sum().item()),
            "cached_clip_high_count": int((raw_q > 15.0).sum().item()),
            "candidates": candidates,
            "best_candidate_by_abs_signed_output_contribution": best_by_contribution["name"],
            "best_candidate_by_weight_rms": best_by_weight_rms["name"],
            "ranked_primary_cause": ranked_cause,
            "worst_columns_by_abs_contribution": worst_columns,
        })
    if timer:
        timer.add("candidate_search", time.perf_counter() - start_time)

    cause_counts: dict[str, int] = {}
    aggregate: dict[str, dict[str, Any]] = {}
    for record in records:
        cause = str(record["ranked_primary_cause"])
        cause_counts[cause] = cause_counts.get(cause, 0) + 1
        for candidate in record["candidates"]:
            name = str(candidate["name"])
            slot = aggregate.setdefault(name, {
                "groups": 0,
                "signed_contribution_sum": 0.0,
                "abs_signed_contribution_sum": 0.0,
                "signed_contribution_sq_sum": 0.0,
                "max_abs_signed_contribution": 0.0,
                "weight_error_rms_sum": 0.0,
                "projection_contribution_rms_sum": 0.0,
                "projection_contribution_max_abs": 0.0,
                "clip_low_count": 0,
                "clip_high_count": 0,
                "qvalue_saturation_zero_count": 0,
                "qvalue_saturation_max_count": 0,
                "qvalue_histogram": [0] * 16,
            })
            signed = float(candidate["projection_contribution_signed_sum"])
            slot["groups"] += 1
            slot["signed_contribution_sum"] += signed
            slot["abs_signed_contribution_sum"] += abs(signed)
            slot["signed_contribution_sq_sum"] += signed * signed
            slot["max_abs_signed_contribution"] = max(slot["max_abs_signed_contribution"], abs(signed))
            slot["weight_error_rms_sum"] += float(candidate["weight_error_rms"])
            slot["projection_contribution_rms_sum"] += float(candidate["projection_contribution_rms"])
            slot["projection_contribution_max_abs"] = max(
                slot["projection_contribution_max_abs"],
                float(candidate["projection_contribution_max_abs"]),
            )
            slot["clip_low_count"] += int(candidate["clip_low_count"])
            slot["clip_high_count"] += int(candidate["clip_high_count"])
            slot["qvalue_saturation_zero_count"] += int(candidate["qvalue_saturation_zero_count"])
            slot["qvalue_saturation_max_count"] += int(candidate["qvalue_saturation_max_count"])
            for idx, count in enumerate(candidate["qvalue_histogram"]):
                slot["qvalue_histogram"][idx] += int(count)
    for slot in aggregate.values():
        groups_count = max(int(slot["groups"]), 1)
        slot["signed_contribution_rms_across_groups"] = math.sqrt(
            float(slot["signed_contribution_sq_sum"]) / groups_count
        )
        slot["mean_weight_error_rms"] = float(slot["weight_error_rms_sum"]) / groups_count
        slot["mean_projection_contribution_rms"] = float(slot["projection_contribution_rms_sum"]) / groups_count
        del slot["weight_error_rms_sum"]
        del slot["projection_contribution_rms_sum"]
    return {
        "computed": True,
        "top_groups_analyzed": len(records),
        "contract": {
            "q_range": "0..15",
            "axis": int(hqq_meta["axis"]),
            "group_size": int(hqq_meta["group_size"]),
            "dequant": "(q - zero) * scale",
            "cached_q_reconstruction": "round(source / scale + zero).clamp(0, 15)",
        },
        "cause_counts": cause_counts,
        "aggregate_by_candidate": aggregate,
        "best_aggregate_by_abs_signed_contribution_sum": min(
            aggregate.keys(),
            key=lambda name: abs(float(aggregate[name]["signed_contribution_sum"])),
        ) if aggregate else None,
        "best_aggregate_by_abs_signed_contribution_total": min(
            aggregate.keys(),
            key=lambda name: float(aggregate[name]["abs_signed_contribution_sum"]),
        ) if aggregate else None,
        "best_aggregate_by_mean_weight_rms": min(
            aggregate.keys(),
            key=lambda name: float(aggregate[name]["mean_weight_error_rms"]),
        ) if aggregate else None,
        "groups": records,
    }


def projection_sensitivity(
    source: torch.Tensor,
    actual: torch.Tensor,
    input_row: torch.Tensor,
    source_meta: dict[str, Any],
    hqq_meta: dict[str, Any],
    hqq_quant: dict[str, torch.Tensor],
    *,
    top_rows: int,
    top_groups: int,
    sample_cols: int,
    focus_row: int | None,
    focus_group: int | None,
    diff_cache: torch.Tensor | None = None,
    timer: StageTimer | None = None,
) -> dict[str, Any]:
    src = source.to(torch.float32).contiguous()
    got = actual.to(torch.float32).contiguous()
    row = input_row.to(torch.float32).contiguous()
    if row.numel() != src.shape[1]:
        raise RuntimeError(f"Input row length mismatch: row={row.numel()} weight_cols={src.shape[1]}")

    diff = diff_cache if diff_cache is not None else got - src
    stage_start = time.perf_counter()
    output_error = torch.mv(diff, row)
    if timer:
        timer.add("projection_sensitivity.output_error", time.perf_counter() - stage_start)
    abs_output_error = output_error.abs()
    ranges = source_slice_ranges(source_meta)

    slice_errors: dict[str, Any] = {}
    for name, row_ranges in ranges.items():
        indices = torch.tensor(
            [idx for start, end in row_ranges for idx in range(start, end)],
            dtype=torch.long,
        )
        err = output_error.index_select(0, indices)
        abs_err = err.abs()
        slice_errors[name] = {
            "rows": int(indices.numel()),
            "max_abs": float(abs_err.max().item()) if abs_err.numel() else 0.0,
            "mean_abs": float(abs_err.mean().item()) if abs_err.numel() else 0.0,
            "rms": float(torch.sqrt((err * err).mean()).item()) if err.numel() else 0.0,
            "error_sq_sum": float((err * err).sum().item()) if err.numel() else 0.0,
        }

    def make_row(row_idx: int) -> dict[str, Any]:
        info = source_row_info(row_idx, source_meta)
        max_abs_weight, max_col = diff[row_idx].abs().max(dim=0)
        row_weight_rms = float(torch.sqrt((diff[row_idx] * diff[row_idx]).mean()).item())
        row_source_rms = float(torch.sqrt((src[row_idx] * src[row_idx]).mean()).item())
        return {
            "global_row": int(row_idx),
            **info,
            "bf16_dot": float(torch.dot(src[row_idx], row).item()),
            "hqq_dot": float(torch.dot(got[row_idx], row).item()),
            "output_error": float(output_error[row_idx].item()),
            "abs_output_error": float(abs_output_error[row_idx].item()),
            "row_weight_rms": row_weight_rms,
            "row_weight_relative_rms": row_weight_rms / max(row_source_rms, 1e-12),
            "max_abs_weight_error": float(max_abs_weight.item()),
            "max_abs_weight_error_col": int(max_col.item()),
            "samples": {
                "input": [float(v) for v in row[:sample_cols].tolist()],
                "bf16_weight": [float(v) for v in src[row_idx, :sample_cols].tolist()],
                "hqq_weight": [float(v) for v in got[row_idx, :sample_cols].tolist()],
                "weight_diff": [float(v) for v in diff[row_idx, :sample_cols].tolist()],
            },
        }

    stage_start = time.perf_counter()
    n_rows = min(top_rows, src.shape[0])
    top_row_indices = torch.topk(abs_output_error, k=n_rows).indices.tolist()
    if timer:
        timer.add("projection_sensitivity.top_row_selection", time.perf_counter() - stage_start)

    group_size = int(hqq_meta["group_size"])
    groups = int(hqq_meta["groups"])
    qvalues = hqq_quant["qvalues"].to(torch.uint8).contiguous()
    scales = hqq_quant["scales"].to(torch.float32).contiguous()
    zeros = hqq_quant["zeros"].to(torch.float32).contiguous()

    def group_diagnostics(row_idx: int, group_idx: int) -> dict[str, Any]:
        start = group_idx * group_size
        end = min(start + group_size, src.shape[1])
        input_group = row[start:end]
        src_group = src[row_idx, start:end]
        got_group = got[row_idx, start:end]
        diff_group = diff[row_idx, start:end]
        contrib_vec = input_group * diff_group
        abs_input = input_group.abs()
        abs_diff = diff_group.abs()
        abs_contrib = contrib_vec.abs()
        q_group = qvalues[row_idx, start:end].to(torch.int64)
        hist = torch.bincount(q_group, minlength=16)[:16]
        signed_sum = float(contrib_vec.sum().item())
        input_norm = float(torch.linalg.vector_norm(input_group).item())
        diff_norm = float(torch.linalg.vector_norm(diff_group).item())
        alignment_cosine = signed_sum / max(input_norm * diff_norm, 1e-30)
        input_max, input_max_col = abs_input.max(dim=0)
        diff_max, diff_max_col = abs_diff.max(dim=0)
        contrib_max, contrib_max_col = abs_contrib.max(dim=0)
        scale_value = float(scales[row_idx, group_idx].item())
        zero_value = float(zeros[row_idx, group_idx].item())
        raw_q = (src_group / max(scale_value, 1e-30)) + zero_value
        expected_q = torch.round(raw_q).clamp(0, 15).to(torch.uint8)
        q_mismatch = expected_q.to(torch.int16) - q_group.to(torch.int16)
        clipped_low = raw_q < 0.0
        clipped_high = raw_q > 15.0
        quantization_residual = got_group - src_group
        return {
            "source_min": float(src_group.min().item()),
            "source_max": float(src_group.max().item()),
            "source_range": float((src_group.max() - src_group.min()).item()),
            "input_rms": float(torch.sqrt((input_group ** 2).mean()).item()),
            "input_max_abs": float(input_max.item()),
            "input_max_abs_col": int(start + input_max_col.item()),
            "weight_error_rms": float(torch.sqrt((diff_group ** 2).mean()).item()),
            "weight_error_max_abs": float(diff_max.item()),
            "weight_error_max_abs_col": int(start + diff_max_col.item()),
            "contribution_rms": float(torch.sqrt((contrib_vec ** 2).mean()).item()),
            "contribution_max_abs": float(contrib_max.item()),
            "contribution_max_abs_col": int(start + contrib_max_col.item()),
            "signed_sum": signed_sum,
            "abs_sum": float(abs_contrib.sum().item()),
            "alignment_cosine": alignment_cosine,
            "scale": scale_value,
            "zero": zero_value,
            "quant_step_size": scale_value,
            "qvalue_min": int(q_group.min().item()),
            "qvalue_max": int(q_group.max().item()),
            "qvalue_histogram": [int(v) for v in hist.tolist()],
            "qvalue_saturation_zero_count": int((q_group == 0).sum().item()),
            "qvalue_saturation_max_count": int((q_group == 15).sum().item()),
            "clip_low_count": int(clipped_low.sum().item()),
            "clip_high_count": int(clipped_high.sum().item()),
            "expected_qvalue_mismatch_count": int((q_mismatch != 0).sum().item()),
            "expected_qvalue_max_abs_mismatch": int(q_mismatch.abs().max().item()) if q_mismatch.numel() else 0,
            "raw_q_min": float(raw_q.min().item()),
            "raw_q_max": float(raw_q.max().item()),
            "raw_q_at_input_max_abs": float(raw_q[input_max_col].item()),
            "raw_q_at_weight_error_max_abs": float(raw_q[diff_max_col].item()),
            "raw_q_at_contribution_max_abs": float(raw_q[contrib_max_col].item()),
            "quantization_residual_mean_signed": float(quantization_residual.mean().item()),
        }

    def per_column_contributions(row_idx: int, group_idx: int, *, limit: int | None = None) -> dict[str, Any]:
        start = group_idx * group_size
        end = min(start + group_size, src.shape[1])
        input_group = row[start:end]
        diff_group = diff[row_idx, start:end]
        contrib_vec = input_group * diff_group
        order = torch.argsort(contrib_vec.abs(), descending=True)
        if limit is not None:
            order = order[:limit]
        columns = []
        for idx in order.tolist():
            col = start + int(idx)
            qv = int(qvalues[row_idx, col].item())
            columns.append({
                "col": int(col),
                "group_local_col": int(idx),
                "input": float(row[col].item()),
                "bf16_weight": float(src[row_idx, col].item()),
                "hqq_dequant_weight": float(got[row_idx, col].item()),
                "weight_error": float(diff[row_idx, col].item()),
                "contribution": float(contrib_vec[idx].item()),
                "abs_contribution": float(abs(contrib_vec[idx].item())),
                "qvalue": qv,
                "raw_q_before_round_clamp": float(
                    (src[row_idx, col] / max(float(scales[row_idx, group_idx].item()), 1e-30))
                    + float(zeros[row_idx, group_idx].item())
                ),
                "is_clipped_low": bool(
                    (
                        (src[row_idx, col] / max(float(scales[row_idx, group_idx].item()), 1e-30))
                        + float(zeros[row_idx, group_idx].item())
                    ).item() < 0.0
                ),
                "is_clipped_high": bool(
                    (
                        (src[row_idx, col] / max(float(scales[row_idx, group_idx].item()), 1e-30))
                        + float(zeros[row_idx, group_idx].item())
                    ).item() > 15.0
                ),
            })
        return {
            "global_row": int(row_idx),
            **source_row_info(row_idx, source_meta),
            "hqq_group_index": int(group_idx),
            "col_start": int(start),
            "col_end": int(end),
            "diagnostics": group_diagnostics(row_idx, group_idx),
            "columns_ranked_by_abs_contribution": columns,
        }

    stage_start = time.perf_counter()
    if src.shape[1] == groups * group_size:
        diff_grouped = diff.view(src.shape[0], groups, group_size)
        row_grouped = row.view(groups, group_size)
        contrib_matrix = torch.einsum("rgs,gs->rg", diff_grouped, row_grouped)
        abs_flat = contrib_matrix.abs().flatten()
        k = min(top_groups, int(abs_flat.numel()))
        values, flat_indices = torch.topk(abs_flat, k=k)
        top_group_records = []
        for value, flat_idx in zip(values.tolist(), flat_indices.tolist(), strict=False):
            row_idx = int(flat_idx // groups)
            group_idx = int(flat_idx % groups)
            start = group_idx * group_size
            end = min(start + group_size, src.shape[1])
            info = source_row_info(row_idx, source_meta)
            contrib = float(contrib_matrix[row_idx, group_idx].item())
            top_group_records.append({
                "global_row": int(row_idx),
                **info,
                "hqq_group_index": int(group_idx),
                "col_start": int(start),
                "col_end": int(end),
                "contribution": contrib,
                "abs_contribution": float(value),
                "weight_rms": float(torch.sqrt((diff[row_idx, start:end] ** 2).mean()).item()),
                "input_rms": float(torch.sqrt((row[start:end] ** 2).mean()).item()),
                "samples": {
                    "input": [float(v) for v in row[start:min(end, start + sample_cols)].tolist()],
                    "weight_diff": [
                        float(v) for v in diff[row_idx, start:min(end, start + sample_cols)].tolist()
                    ],
                },
            })
    else:
        group_records: list[dict[str, Any]] = []
        for row_idx in range(src.shape[0]):
            info = source_row_info(row_idx, source_meta)
            for group_idx in range(groups):
                start = group_idx * group_size
                end = min(start + group_size, src.shape[1])
                contrib_vec = diff[row_idx, start:end] * row[start:end]
                contrib = float(contrib_vec.sum().item())
                group_records.append({
                    "global_row": int(row_idx),
                    **info,
                    "hqq_group_index": int(group_idx),
                    "col_start": int(start),
                    "col_end": int(end),
                    "contribution": contrib,
                    "abs_contribution": abs(contrib),
                    "weight_rms": float(torch.sqrt((diff[row_idx, start:end] ** 2).mean()).item()),
                    "input_rms": float(torch.sqrt((row[start:end] ** 2).mean()).item()),
                    "samples": {
                        "input": [float(v) for v in row[start:min(end, start + sample_cols)].tolist()],
                        "weight_diff": [
                            float(v) for v in diff[row_idx, start:min(end, start + sample_cols)].tolist()
                        ],
                    },
                })
        top_group_records = sorted(group_records, key=lambda item: item["abs_contribution"], reverse=True)[:top_groups]
    if timer:
        timer.add("top_group_selection", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    for group in top_group_records:
        group["diagnostics"] = group_diagnostics(int(group["global_row"]), int(group["hqq_group_index"]))
    if timer:
        timer.add("top_group_diagnostics", time.perf_counter() - stage_start)
    quantizer_choice = quantizer_choice_diagnostics(
        src,
        got,
        row,
        source_meta,
        hqq_meta,
        hqq_quant,
        top_group_records,
        focus_row=focus_row,
        focus_group=focus_group,
        sample_cols=sample_cols,
        timer=timer,
    )
    total_error_sq = sum(float(item["error_sq_sum"]) for item in slice_errors.values())
    focus = None
    if focus_row is not None and focus_group is not None:
        if not (0 <= focus_row < src.shape[0]):
            raise RuntimeError(f"--focus-row out of range: {focus_row} for rows={src.shape[0]}")
        if not (0 <= focus_group < groups):
            raise RuntimeError(f"--focus-group out of range: {focus_group} for groups={groups}")
        focus = per_column_contributions(focus_row, focus_group)
    return {
        "computed": True,
        "tensor_name": str(source_meta.get("tensor_name", "unknown")),
        "input_row": {
            "n": int(row.numel()),
            "l2": float(torch.linalg.vector_norm(row).item()),
            "mean": float(row.mean().item()),
            "min": float(row.min().item()),
            "max": float(row.max().item()),
        },
        "overall": {
            "rows": int(output_error.numel()),
            "max_abs": float(abs_output_error.max().item()),
            "mean_abs": float(abs_output_error.mean().item()),
            "rms": float(torch.sqrt((output_error * output_error).mean()).item()),
        },
        "slices": {
            name: {
                **metric,
                "error_sq_fraction": float(metric["error_sq_sum"] / max(total_error_sq, 1e-30)),
            }
            for name, metric in slice_errors.items()
        },
        "dominant_slice_by_output_error_energy": max(
            slice_errors.keys(), key=lambda name: float(slice_errors[name]["error_sq_sum"])
        ),
        "top_rows_by_abs_output_error": [make_row(int(idx)) for idx in top_row_indices],
        "top_hqq_groups_by_abs_contribution": top_group_records,
        "focus_group_column_contributions": focus,
        "quantizer_choice": quantizer_choice,
    }


def aggregate_projection_sensitivities(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"computed": False, "reason": "no input rows"}
    slice_names = list(rows[0]["sensitivity"]["slices"].keys())
    slice_totals = {
        name: {"rows": 0, "error_sq_sum": 0.0, "max_abs": 0.0, "rms_values": []}
        for name in slice_names
    }
    candidate_totals: dict[str, dict[str, Any]] = {}
    regressions: list[dict[str, Any]] = []
    for row_record in rows:
        sensitivity = row_record["sensitivity"]
        for name in slice_names:
            metric = sensitivity["slices"][name]
            slot = slice_totals[name]
            slot["rows"] += int(metric["rows"])
            slot["error_sq_sum"] += float(metric["error_sq_sum"])
            slot["max_abs"] = max(float(slot["max_abs"]), float(metric["max_abs"]))
            slot["rms_values"].append(float(metric["rms"]))
        quantizer = sensitivity.get("quantizer_choice", {})
        for candidate_name, aggregate in quantizer.get("aggregate_by_candidate", {}).items():
            slot = candidate_totals.setdefault(candidate_name, {
                "groups": 0,
                "signed_contribution_sum": 0.0,
                "abs_signed_contribution_sum": 0.0,
                "signed_contribution_sq_sum": 0.0,
                "max_abs_signed_contribution": 0.0,
                "weight_rms_sum": 0.0,
                "clip_low_count": 0,
                "clip_high_count": 0,
                "qvalue_saturation_zero_count": 0,
                "qvalue_saturation_max_count": 0,
            })
            groups = int(aggregate["groups"])
            slot["groups"] += groups
            slot["signed_contribution_sum"] += float(aggregate["signed_contribution_sum"])
            slot["abs_signed_contribution_sum"] += float(aggregate["abs_signed_contribution_sum"])
            signed_rms = float(aggregate["signed_contribution_rms_across_groups"])
            slot["signed_contribution_sq_sum"] += signed_rms * signed_rms * groups
            slot["max_abs_signed_contribution"] = max(
                float(slot["max_abs_signed_contribution"]),
                float(aggregate["max_abs_signed_contribution"]),
            )
            slot["weight_rms_sum"] += float(aggregate["mean_weight_error_rms"]) * groups
            slot["clip_low_count"] += int(aggregate["clip_low_count"])
            slot["clip_high_count"] += int(aggregate["clip_high_count"])
            slot["qvalue_saturation_zero_count"] += int(aggregate["qvalue_saturation_zero_count"])
            slot["qvalue_saturation_max_count"] += int(aggregate["qvalue_saturation_max_count"])
        cached = quantizer.get("aggregate_by_candidate", {}).get("cached_hqq")
        weighted = quantizer.get("aggregate_by_candidate", {}).get("activation_weighted_affine")
        if cached and weighted:
            cached_abs = float(cached["abs_signed_contribution_sum"])
            weighted_abs = float(weighted["abs_signed_contribution_sum"])
            if weighted_abs > cached_abs:
                regressions.append({
                    "case": row_record.get("case"),
                    "cached_abs_signed_sum": cached_abs,
                    "activation_weighted_abs_signed_sum": weighted_abs,
                    "ratio": weighted_abs / max(cached_abs, 1e-30),
                })
    for slot in candidate_totals.values():
        groups = max(int(slot["groups"]), 1)
        slot["signed_contribution_rms_across_groups"] = math.sqrt(
            float(slot["signed_contribution_sq_sum"]) / groups
        )
        slot["mean_weight_error_rms"] = float(slot["weight_rms_sum"]) / groups
        del slot["signed_contribution_sq_sum"]
        del slot["weight_rms_sum"]
    total_error_sq = sum(float(slot["error_sq_sum"]) for slot in slice_totals.values())
    slices = {}
    for name, slot in slice_totals.items():
        row_count = max(int(slot["rows"]), 1)
        slices[name] = {
            "rows": int(slot["rows"]),
            "error_sq_sum": float(slot["error_sq_sum"]),
            "error_sq_fraction": float(slot["error_sq_sum"]) / max(total_error_sq, 1e-30),
            "rms": math.sqrt(float(slot["error_sq_sum"]) / row_count),
            "max_abs": float(slot["max_abs"]),
            "mean_row_rms": sum(slot["rms_values"]) / max(len(slot["rms_values"]), 1),
        }
    return {
        "computed": True,
        "input_rows": len(rows),
        "unique_hashes": sorted({row["input_row_trace"]["hash64"] for row in rows}),
        "slices": slices,
        "dominant_slice_by_output_error_energy": max(
            slices.keys(), key=lambda name: float(slices[name]["error_sq_sum"])
        ),
        "candidate_aggregate": candidate_totals,
        "best_candidate_by_abs_signed_contribution_total": min(
            candidate_totals.keys(),
            key=lambda name: float(candidate_totals[name]["abs_signed_contribution_sum"]),
        ) if candidate_totals else None,
        "activation_weighted_regressions": regressions,
    }


def empty_candidate_slot() -> dict[str, Any]:
    return {
        "groups": 0,
        "signed_contribution_sum": 0.0,
        "abs_signed_contribution_sum": 0.0,
        "signed_contribution_sq_sum": 0.0,
        "max_abs_signed_contribution": 0.0,
        "weight_rms_sum": 0.0,
        "weight_rms_sq_sum": 0.0,
        "projection_contribution_rms_sum": 0.0,
        "projection_contribution_max_abs": 0.0,
        "clip_low_count": 0,
        "clip_high_count": 0,
        "qvalue_saturation_zero_count": 0,
        "qvalue_saturation_max_count": 0,
    }


def add_candidate_metrics(slot: dict[str, Any], candidate: dict[str, Any]) -> None:
    signed = float(candidate["projection_contribution_signed_sum"])
    weight_rms = float(candidate["weight_error_rms"])
    slot["groups"] += 1
    slot["signed_contribution_sum"] += signed
    slot["abs_signed_contribution_sum"] += abs(signed)
    slot["signed_contribution_sq_sum"] += signed * signed
    slot["max_abs_signed_contribution"] = max(float(slot["max_abs_signed_contribution"]), abs(signed))
    slot["weight_rms_sum"] += weight_rms
    slot["weight_rms_sq_sum"] += weight_rms * weight_rms
    slot["projection_contribution_rms_sum"] += float(candidate["projection_contribution_rms"])
    slot["projection_contribution_max_abs"] = max(
        float(slot["projection_contribution_max_abs"]),
        float(candidate["projection_contribution_max_abs"]),
    )
    slot["clip_low_count"] += int(candidate["clip_low_count"])
    slot["clip_high_count"] += int(candidate["clip_high_count"])
    slot["qvalue_saturation_zero_count"] += int(candidate["qvalue_saturation_zero_count"])
    slot["qvalue_saturation_max_count"] += int(candidate["qvalue_saturation_max_count"])


def finalize_candidate_slots(slots: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized = {}
    cached_weight_rms = None
    if "cached_hqq" in slots and int(slots["cached_hqq"]["groups"]):
        cached_weight_rms = float(slots["cached_hqq"]["weight_rms_sum"]) / int(slots["cached_hqq"]["groups"])
    for name, slot in slots.items():
        groups = max(int(slot["groups"]), 1)
        mean_weight_rms = float(slot["weight_rms_sum"]) / groups
        finalized[name] = {
            "groups": int(slot["groups"]),
            "signed_contribution_sum": float(slot["signed_contribution_sum"]),
            "abs_signed_contribution_sum": float(slot["abs_signed_contribution_sum"]),
            "signed_contribution_rms_across_groups": math.sqrt(
                float(slot["signed_contribution_sq_sum"]) / groups
            ),
            "max_abs_signed_contribution": float(slot["max_abs_signed_contribution"]),
            "mean_weight_error_rms": mean_weight_rms,
            "weight_error_rms_across_groups": math.sqrt(float(slot["weight_rms_sq_sum"]) / groups),
            "mean_projection_contribution_rms": float(slot["projection_contribution_rms_sum"]) / groups,
            "projection_contribution_max_abs": float(slot["projection_contribution_max_abs"]),
            "clip_low_count": int(slot["clip_low_count"]),
            "clip_high_count": int(slot["clip_high_count"]),
            "qvalue_saturation_zero_count": int(slot["qvalue_saturation_zero_count"]),
            "qvalue_saturation_max_count": int(slot["qvalue_saturation_max_count"]),
        }
        if cached_weight_rms is not None:
            finalized[name]["mean_weight_error_rms_delta_vs_cached"] = mean_weight_rms - cached_weight_rms
    return finalized


def aggregate_candidate_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    slots: dict[str, dict[str, Any]] = {}
    for group in groups:
        for candidate in group.get("candidates", []):
            name = str(candidate["name"])
            add_candidate_metrics(slots.setdefault(name, empty_candidate_slot()), candidate)
    aggregate = finalize_candidate_slots(slots)
    return {
        "aggregate_by_candidate": aggregate,
        "best_candidate_by_abs_signed_contribution_total": min(
            aggregate.keys(),
            key=lambda name: float(aggregate[name]["abs_signed_contribution_sum"]),
        ) if aggregate else None,
        "best_candidate_by_signed_contribution_sum": min(
            aggregate.keys(),
            key=lambda name: abs(float(aggregate[name]["signed_contribution_sum"])),
        ) if aggregate else None,
        "best_candidate_by_mean_weight_rms": min(
            aggregate.keys(),
            key=lambda name: float(aggregate[name]["mean_weight_error_rms"]),
        ) if aggregate else None,
    }


def candidate_regression_record(
    *,
    scope: dict[str, Any],
    aggregate: dict[str, Any],
    candidate_name: str,
) -> dict[str, Any] | None:
    candidates = aggregate.get("aggregate_by_candidate", {})
    cached = candidates.get("cached_hqq")
    candidate = candidates.get(candidate_name)
    if not cached or not candidate:
        return None
    cached_abs = float(cached["abs_signed_contribution_sum"])
    candidate_abs = float(candidate["abs_signed_contribution_sum"])
    if candidate_abs <= cached_abs:
        return None
    return {
        **scope,
        "candidate": candidate_name,
        "cached_abs_signed_sum": cached_abs,
        "candidate_abs_signed_sum": candidate_abs,
        "ratio": candidate_abs / max(cached_abs, 1e-30),
    }


def build_decision_quality_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_names = [
        "cached_hqq",
        "minmax_affine",
        "activation_weighted_affine",
        "activation_weighted_clip_free",
        "offline_best_fit_affine",
    ]
    slice_names = list(rows[0]["sensitivity"]["slices"].keys()) if rows else []
    tensor_name = rows[0].get("sensitivity", {}).get("tensor_name", "unknown") if rows else "unknown"
    groups_by_slice = {name: [] for name in slice_names}
    groups_by_case: dict[str, list[dict[str, Any]]] = {}
    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    regressions: list[dict[str, Any]] = []

    for row_record in rows:
        sensitivity = row_record["sensitivity"]
        case = row_record.get("case") or {}
        case_key = str(case.get("case_key", f"row:{len(rows_by_case)}"))
        rows_by_case.setdefault(case_key, []).append(row_record)
        groups = sensitivity.get("quantizer_choice", {}).get("groups", [])
        groups_by_case.setdefault(case_key, []).extend(groups)
        for group in groups:
            groups_by_slice[str(group["slice"])].append(group)

    by_slice = {}
    for slice_name, groups in groups_by_slice.items():
        aggregate = aggregate_candidate_groups(groups)
        slice_regressions = [
            item
            for candidate in ("activation_weighted_affine", "activation_weighted_clip_free", "minmax_affine")
            if (
                item := candidate_regression_record(
                    scope={"slice": slice_name},
                    aggregate=aggregate,
                    candidate_name=candidate,
                )
            )
        ]
        regressions.extend(slice_regressions)
        by_slice[slice_name] = {
            "selected_groups": len(groups),
            **aggregate,
            "activation_candidate_regressions": slice_regressions,
        }

    by_case = []
    for case_key, groups in sorted(groups_by_case.items()):
        row_items = rows_by_case[case_key]
        case = row_items[0].get("case") or {}
        aggregate = aggregate_candidate_groups(groups)
        case_regressions = [
            item
            for candidate in ("activation_weighted_affine", "activation_weighted_clip_free", "minmax_affine")
            if (
                item := candidate_regression_record(
                    scope={"case_key": case_key, "prompt_prefix": case.get("prompt_prefix")},
                    aggregate=aggregate,
                    candidate_name=candidate,
                )
            )
        ]
        regressions.extend(case_regressions)
        cached_slice_totals = {name: {"error_sq_sum": 0.0, "rows": 0, "max_abs": 0.0} for name in slice_names}
        for row_record in row_items:
            for slice_name, metric in row_record["sensitivity"]["slices"].items():
                slot = cached_slice_totals[slice_name]
                slot["error_sq_sum"] += float(metric["error_sq_sum"])
                slot["rows"] += int(metric["rows"])
                slot["max_abs"] = max(float(slot["max_abs"]), float(metric["max_abs"]))
        cached_output_by_slice = {
            slice_name: {
                "rms": math.sqrt(float(slot["error_sq_sum"]) / max(int(slot["rows"]), 1)),
                "max_abs": float(slot["max_abs"]),
            }
            for slice_name, slot in cached_slice_totals.items()
        }
        by_case.append({
            "case_key": case_key,
            "conv_idx": case.get("conv_idx"),
            "turn": case.get("turn"),
            "prompt_prefix": case.get("prompt_prefix"),
            "positions": sorted({int(row["input_row_trace"]["pos"]) for row in row_items}),
            "rows": len(row_items),
            "unique_hashes": sorted({row["input_row_trace"]["hash64"] for row in row_items}),
            "cached_output_by_slice": cached_output_by_slice,
            "candidate_summary": aggregate,
            "activation_candidate_regressions": case_regressions,
        })

    all_groups = [group for groups in groups_by_slice.values() for group in groups]
    overall = aggregate_candidate_groups(all_groups)
    cached = overall["aggregate_by_candidate"].get("cached_hqq", {})
    weighted = overall["aggregate_by_candidate"].get("activation_weighted_affine", {})
    clip_free = overall["aggregate_by_candidate"].get("activation_weighted_clip_free", {})
    minmax = overall["aggregate_by_candidate"].get("minmax_affine", {})

    weighted_ratio = None
    if cached and weighted:
        weighted_ratio = float(weighted["abs_signed_contribution_sum"]) / max(
            float(cached["abs_signed_contribution_sum"]),
            1e-30,
        )
    clip_free_ratio = None
    if cached and clip_free:
        clip_free_ratio = float(clip_free["abs_signed_contribution_sum"]) / max(
            float(cached["abs_signed_contribution_sum"]),
            1e-30,
        )

    return {
        "computed": True,
        "scope": {
            "source": "selected top HQQ groups per captured row from offline hqq-attn-diff",
            "tensor": tensor_name,
            "rows": len(rows),
            "cases": len(by_case),
            "slices": slice_names,
            "candidate_names": candidate_names,
        },
        "overall": overall,
        "by_slice": by_slice,
        "by_case": by_case,
        "activation_candidate_regressions": regressions,
        "candidate_recommendation_evidence": {
            "proves": [
                f"For captured Qwen3.5 rows and selected top groups for {tensor_name}, activation-aware candidates can be compared against cached HQQ in the offline selected-group objective.",
                "The diagnostic can now identify per-case and per-slice regressions in the offline selected-group objective.",
            ],
            "does_not_prove": [
                "It does not prove end-to-end model quality improvement.",
                "It does not validate layers or tensors without matching activation vectors, other models, prompts outside the captured row set, or full-group quantizer replacement.",
                "It does not justify changing HQQ conversion, calibration, runtime defaults, or quantization math.",
            ],
            "additional_evidence_needed": [
                "Run the same offline summary across more layers and HQQ attention tensors when their required activation vectors have been captured.",
                "Design a calibration-row selection method that is independent of this one witness artifact and verify no regressions across QCN/Qwen3.5 lanes.",
                "Only after offline coverage is stable, test an explicitly approved experimental HQQ conversion artifact against llama-witness runtime comparisons.",
            ],
            "weighted_abs_signed_ratio_vs_cached": weighted_ratio,
            "clip_free_abs_signed_ratio_vs_cached": clip_free_ratio,
            "minmax_abs_signed_ratio_vs_cached": (
                float(minmax["abs_signed_contribution_sum"]) / max(float(cached["abs_signed_contribution_sum"]), 1e-30)
                if cached and minmax else None
            ),
        },
    }


def slice_dominance(slices: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = list(slices.keys())
    total_error_sq = sum(float(slices[name]["error_sq_sum"]) for name in names)
    ranked = []
    for name in names:
        error_sq = float(slices[name]["error_sq_sum"])
        ranked.append({
            "slice": name,
            "error_sq_sum": error_sq,
            "error_sq_fraction": error_sq / max(total_error_sq, 1e-30),
            "rms": float(slices[name]["rms"]),
            "relative_rms": float(slices[name]["relative_rms"]),
        })
    ranked.sort(key=lambda item: item["error_sq_sum"], reverse=True)
    return {
        "by_error_energy": ranked,
        "dominant_by_error_energy": ranked[0]["slice"] if ranked else None,
        "dominant_by_relative_rms": max(names, key=lambda name: float(slices[name]["relative_rms"])) if names else None,
    }


def sample_rows(t: torch.Tensor, rows: list[int], cols: int) -> list[dict[str, Any]]:
    out = []
    data = t.to(torch.float32)
    for row in rows:
        if 0 <= row < data.shape[0]:
            out.append({
                "row": int(row),
                "values": [float(v) for v in data[row, :cols].tolist()],
            })
    return out


def qkvz_slice_ranges(nk: int, dk: int, hr: int, dv: int) -> dict[str, list[tuple[int, int]]]:
    ranges: dict[str, list[tuple[int, int]]] = {"q": [], "k": [], "v": [], "z": []}
    group_dim = (2 * dk) + (2 * hr * dv)
    for head in range(nk):
        base = head * group_dim
        ranges["q"].append((base, base + dk))
        ranges["k"].append((base + dk, base + 2 * dk))
        ranges["v"].append((base + 2 * dk, base + 2 * dk + hr * dv))
        ranges["z"].append((base + 2 * dk + hr * dv, base + group_dim))
    return ranges


def ba_slice_ranges(nk: int, hr: int) -> dict[str, list[tuple[int, int]]]:
    ranges: dict[str, list[tuple[int, int]]] = {"b": [], "a": []}
    group_dim = 2 * hr
    for head in range(nk):
        base = head * group_dim
        ranges["b"].append((base, base + hr))
        ranges["a"].append((base + hr, base + group_dim))
    return ranges


def source_slice_ranges(source_meta: dict[str, Any]) -> dict[str, list[tuple[int, int]]]:
    tensor_name = str(source_meta.get("tensor_name", "in_proj_qkvz"))
    if tensor_name == "in_proj_ba":
        return ba_slice_ranges(int(source_meta["nk"]), int(source_meta["head_ratio"]))
    if tensor_name == "out_proj":
        return {"out": [(0, int(source_meta["rows"]))]}
    if tensor_name == "fused_qkv":
        q_rows = int(source_meta["q_rows"])
        k_rows = int(source_meta["k_rows"])
        v_rows = int(source_meta["v_rows"])
        return {
            "q": [(0, q_rows)],
            "k": [(q_rows, q_rows + k_rows)],
            "v": [(q_rows + k_rows, q_rows + k_rows + v_rows)],
        }
    if tensor_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return {tensor_name.removesuffix("_proj"): [(0, int(source_meta["rows"]))]}
    return qkvz_slice_ranges(
        int(source_meta["nk"]),
        int(source_meta["dk"]),
        int(source_meta["head_ratio"]),
        int(source_meta["dv"]),
    )


def gather_ranges(t: torch.Tensor, ranges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.cat([t[start:end] for start, end in ranges], dim=0).contiguous()


def build_qwen35_qkvz_source(model_dir: Path, layer: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    prefix = f"model.language_model.layers.{layer}.linear_attn"
    qkv_raw = load_index_tensor(model_dir, f"{prefix}.in_proj_qkv.weight")
    z_raw = load_index_tensor(model_dir, f"{prefix}.in_proj_z.weight")
    config_path = model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f).get("text_config", {})
    nk = int(cfg["linear_num_key_heads"])
    dk = int(cfg["linear_key_head_dim"])
    nv = int(cfg["linear_num_value_heads"])
    dv = int(cfg["linear_value_head_dim"])
    hr = nv // nk
    key_dim = nk * dk
    parts = []
    for head in range(nk):
        parts.append(qkv_raw[head * dk:(head + 1) * dk])
        parts.append(qkv_raw[key_dim + head * dk:key_dim + (head + 1) * dk])
        parts.append(qkv_raw[key_dim * 2 + head * hr * dv:key_dim * 2 + (head + 1) * hr * dv])
        parts.append(z_raw[head * hr * dv:(head + 1) * hr * dv])
    interleaved = torch.cat(parts, dim=0).contiguous()
    raw_concat = torch.cat([qkv_raw, z_raw], dim=0).contiguous()
    meta = {
        "tensor_name": "in_proj_qkvz",
        "source_qkv_tensor": f"{prefix}.in_proj_qkv.weight",
        "source_z_tensor": f"{prefix}.in_proj_z.weight",
        "qkv_raw_shape": list(qkv_raw.shape),
        "z_raw_shape": list(z_raw.shape),
        "fused_shape": list(interleaved.shape),
        "raw_concat_shape": list(raw_concat.shape),
        "nk": nk,
        "dk": dk,
        "nv": nv,
        "dv": dv,
        "head_ratio": hr,
        "group_dim": (2 * dk) + (2 * hr * dv),
        "slice_names": ["q", "k", "v", "z"],
        "row_order": "per_key_head:[q,k,v,z]",
    }
    return interleaved, raw_concat, meta


def build_qwen35_ba_source(model_dir: Path, layer: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    prefix = f"model.language_model.layers.{layer}.linear_attn"
    b_raw = load_index_tensor(model_dir, f"{prefix}.in_proj_b.weight")
    a_raw = load_index_tensor(model_dir, f"{prefix}.in_proj_a.weight")
    config_path = model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f).get("text_config", {})
    nk = int(cfg["linear_num_key_heads"])
    nv = int(cfg["linear_num_value_heads"])
    hr = nv // nk
    parts = []
    for head in range(nk):
        parts.append(b_raw[head * hr:(head + 1) * hr])
        parts.append(a_raw[head * hr:(head + 1) * hr])
    interleaved = torch.cat(parts, dim=0).contiguous()
    raw_concat = torch.cat([b_raw, a_raw], dim=0).contiguous()
    meta = {
        "tensor_name": "in_proj_ba",
        "source_b_tensor": f"{prefix}.in_proj_b.weight",
        "source_a_tensor": f"{prefix}.in_proj_a.weight",
        "b_raw_shape": list(b_raw.shape),
        "a_raw_shape": list(a_raw.shape),
        "fused_shape": list(interleaved.shape),
        "raw_concat_shape": list(raw_concat.shape),
        "rows": int(interleaved.shape[0]),
        "nk": nk,
        "nv": nv,
        "head_ratio": hr,
        "group_dim": 2 * hr,
        "slice_names": ["b", "a"],
        "row_order": "per_key_head:[b,a]",
    }
    return interleaved, raw_concat, meta


def build_qwen35_la_out_proj_source(model_dir: Path, layer: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    prefix = f"model.language_model.layers.{layer}.linear_attn"
    out = load_index_tensor(model_dir, f"{prefix}.out_proj.weight").contiguous()
    meta = {
        "tensor_name": "out_proj",
        "source_tensor": f"{prefix}.out_proj.weight",
        "fused_shape": list(out.shape),
        "raw_concat_shape": list(out.shape),
        "rows": int(out.shape[0]),
        "cols": int(out.shape[1]),
        "slice_names": ["out"],
        "row_order": "contiguous:out",
    }
    return out, out, meta


def build_qwen35_gqa_source(model_dir: Path, layer: int, tensor_name: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    prefix = f"model.language_model.layers.{layer}.self_attn"
    if tensor_name == "fused_qkv":
        q = load_index_tensor(model_dir, f"{prefix}.q_proj.weight")
        k = load_index_tensor(model_dir, f"{prefix}.k_proj.weight")
        v = load_index_tensor(model_dir, f"{prefix}.v_proj.weight")
        fused = torch.cat([q, k, v], dim=0).contiguous()
        meta = {
            "tensor_name": "fused_qkv",
            "source_q_tensor": f"{prefix}.q_proj.weight",
            "source_k_tensor": f"{prefix}.k_proj.weight",
            "source_v_tensor": f"{prefix}.v_proj.weight",
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "fused_shape": list(fused.shape),
            "raw_concat_shape": list(fused.shape),
            "rows": int(fused.shape[0]),
            "cols": int(fused.shape[1]),
            "q_rows": int(q.shape[0]),
            "k_rows": int(k.shape[0]),
            "v_rows": int(v.shape[0]),
            "slice_names": ["q", "k", "v"],
            "row_order": "contiguous:[q,k,v]",
        }
        return fused, fused, meta
    if tensor_name not in ("q_proj", "k_proj", "v_proj", "o_proj"):
        raise RuntimeError(f"Unsupported GQA tensor {tensor_name!r}")
    tensor = load_index_tensor(model_dir, f"{prefix}.{tensor_name}.weight").contiguous()
    slice_name = tensor_name.removesuffix("_proj")
    meta = {
        "tensor_name": tensor_name,
        "source_tensor": f"{prefix}.{tensor_name}.weight",
        "fused_shape": list(tensor.shape),
        "raw_concat_shape": list(tensor.shape),
        "rows": int(tensor.shape[0]),
        "cols": int(tensor.shape[1]),
        "slice_names": [slice_name],
        "row_order": f"contiguous:{slice_name}",
    }
    return tensor, tensor, meta


def build_qwen35_source(model_dir: Path, layer: int, tensor_name: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if tensor_name == "in_proj_qkvz":
        return build_qwen35_qkvz_source(model_dir, layer)
    if tensor_name == "in_proj_ba":
        return build_qwen35_ba_source(model_dir, layer)
    if tensor_name == "out_proj":
        return build_qwen35_la_out_proj_source(model_dir, layer)
    if tensor_name in ("fused_qkv", "q_proj", "k_proj", "v_proj", "o_proj"):
        return build_qwen35_gqa_source(model_dir, layer, tensor_name)
    raise RuntimeError(f"Source reconstruction for tensor {tensor_name!r} is not implemented")


def load_safetensors_index(model_dir: Path) -> dict[str, Any]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"Missing safetensors index: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index.get("weight_map"), dict):
        raise RuntimeError(f"Safetensors index has no weight_map: {index_path}")
    return index


def load_model_text_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config")
    return text_cfg if isinstance(text_cfg, dict) else cfg


def tensor_header(model_dir: Path, index: dict[str, Any], name: str) -> dict[str, Any]:
    shard_name = index.get("weight_map", {}).get(name)
    if not shard_name:
        return {"name": name, "exists": False}
    shard_path = model_dir / str(shard_name)
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(name)
        shape = [int(v) for v in tensor_slice.get_shape()]
        dtype = str(tensor_slice.get_dtype())
    return {
        "name": name,
        "exists": True,
        "shard": str(shard_name),
        "source_file": str(shard_name),
        "source_path": str(shard_path),
        "shape": shape,
        "dtype": dtype,
    }


def existing_tensor_header(model_dir: Path, index: dict[str, Any], name: str) -> dict[str, Any] | None:
    header = tensor_header(model_dir, index, name)
    return header if header.get("exists") else None


def source_prefix_candidates(layer: int, family: str) -> list[str]:
    submodule = "linear_attn" if family == "linear_attn" else "self_attn"
    return [
        f"model.layers.{layer}.{submodule}",
        f"model.language_model.layers.{layer}.{submodule}",
    ]


def merged_source_shape(headers: list[dict[str, Any]]) -> list[int] | None:
    if not headers or any(not item.get("exists") for item in headers):
        return None
    shapes = [item["shape"] for item in headers]
    if any(len(shape) != 2 for shape in shapes):
        return None
    cols = {int(shape[1]) for shape in shapes}
    if len(cols) != 1:
        return None
    return [sum(int(shape[0]) for shape in shapes), int(shapes[0][1])]


def dtype_summary(headers: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("dtype")) for item in headers if item.get("exists")})


def strict_resolver_contract(selected: dict[str, Any] | None) -> dict[str, Any]:
    if not selected:
        return {
            "resolver_kind": None,
            "ready": False,
            "reason": "source mapping is not uniquely resolved",
            "fallback_allowed": False,
        }

    names = [str(item.get("name", "")) for item in selected.get("source_tensors", [])]
    recipe = str(selected.get("recipe"))
    qcn_model_layers = bool(names) and all(name.startswith("model.layers.") for name in names)
    q35_language_model = bool(names) and all(name.startswith("model.language_model.") for name in names)

    if qcn_model_layers:
        if recipe in {"direct_fused_qkvz", "direct_fused_ba"}:
            resolver_kind = "qcn_model_layers_direct_fused"
        elif recipe == "self_attn_qkv_concat":
            resolver_kind = "qcn_model_layers_self_attn_qkv_concat"
        elif recipe.startswith("direct_"):
            resolver_kind = "qcn_model_layers_direct"
        else:
            resolver_kind = None
    elif q35_language_model:
        if recipe in {"qwen35_qkvz_split_interleave", "qwen35_ba_split_interleave"}:
            resolver_kind = "q35_language_model_split_interleave"
        elif recipe == "self_attn_qkv_concat":
            resolver_kind = "q35_language_model_self_attn_qkv_concat"
        elif recipe.startswith("direct_"):
            resolver_kind = "q35_language_model_direct"
        else:
            resolver_kind = None
    else:
        resolver_kind = None

    if not resolver_kind:
        return {
            "resolver_kind": None,
            "ready": False,
            "reason": f"unsupported explicit resolver recipe/prefix combination: {recipe}",
            "fallback_allowed": False,
        }
    return {
        "resolver_kind": resolver_kind,
        "ready": True,
        "reason": None,
        "fallback_allowed": False,
    }


def reconstruct_source_for_contract(
    model_dir: Path,
    layer: int,
    tensor_name: str,
    selected: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    recipe = str(selected.get("recipe"))
    source_names = [str(item["name"]) for item in selected.get("source_tensors", [])]
    if len(source_names) == 1:
        tensor = load_index_tensor(model_dir, source_names[0]).contiguous()
        return tensor, {
            "recipe": recipe,
            "row_order": selected.get("row_column_layout"),
            "source_tensors": source_names,
        }
    if recipe in {"qwen35_qkvz_split_interleave", "qwen35_ba_split_interleave"}:
        reconstructed, _raw_concat, meta = build_qwen35_source(model_dir, layer, tensor_name)
        return reconstructed.contiguous(), meta
    if recipe == "self_attn_qkv_concat":
        parts = [load_index_tensor(model_dir, name) for name in source_names]
        return torch.cat(parts, dim=0).contiguous(), {
            "recipe": recipe,
            "row_order": selected.get("row_column_layout"),
            "source_tensors": source_names,
        }
    raise RuntimeError(f"No strict source reconstruction implementation for recipe={recipe!r}")


def build_source_contract(
    *,
    model_dir: Path,
    cache_dir: Path,
    index: dict[str, Any],
    model_id: str,
    entry: dict[str, Any],
    selected: dict[str, Any] | None,
    shape_compatible: bool,
    dtype_ok: bool,
    source_hash_cache: dict[str, str],
    file_hash_cache: dict[str, str],
) -> dict[str, Any]:
    layer = int(entry["layer_idx"])
    tensor_name = str(entry["tensor_name"])
    hqq_rel_file = str(entry.get("file") or "")
    hqq_path = cache_dir / hqq_rel_file
    resolver = strict_resolver_contract(selected)

    source_items = []
    source_hash_ready = True
    if selected:
        for header in selected.get("source_tensors", []):
            item = dict(header)
            name = str(item.get("name"))
            if item.get("exists"):
                try:
                    if name not in source_hash_cache:
                        source_hash_cache[name] = tensor_hash(load_index_tensor(model_dir, name))
                    item["source_sha256"] = source_hash_cache[name]
                    item["source_sha256_unavailable_reason"] = None
                except Exception as exc:
                    item["source_sha256"] = None
                    item["source_sha256_unavailable_reason"] = str(exc)
                    source_hash_ready = False
            else:
                item["source_sha256"] = None
                item["source_sha256_unavailable_reason"] = "source tensor missing from safetensors index"
                source_hash_ready = False
            source_items.append(item)
    else:
        source_hash_ready = False

    artifact_hash = None
    artifact_unavailable_reason = None
    artifact_ready = False
    if hqq_rel_file:
        try:
            key = str(hqq_path)
            if key not in file_hash_cache:
                file_hash_cache[key] = file_sha256(hqq_path)
            artifact_hash = file_hash_cache[key]
            artifact_ready = True
        except Exception as exc:
            artifact_unavailable_reason = str(exc)

    reconstructed_hash = None
    reconstructed_hash_unavailable_reason = None
    reconstructed_shape = None
    if selected and resolver["ready"]:
        try:
            reconstructed, _meta = reconstruct_source_for_contract(model_dir, layer, tensor_name, selected)
            reconstructed_shape = [int(v) for v in reconstructed.shape]
            reconstructed_hash = tensor_hash(reconstructed)
        except Exception as exc:
            reconstructed_hash_unavailable_reason = str(exc)

    ready = bool(
        selected
        and resolver["ready"]
        and shape_compatible
        and dtype_ok
        and source_hash_ready
        and artifact_ready
        and reconstructed_hash
    )
    missing = []
    if not selected:
        missing.append("selected_source_mapping")
    if not resolver["ready"]:
        missing.append("strict_resolver")
    if not shape_compatible:
        missing.append("shape_compatibility")
    if not dtype_ok:
        missing.append("source_dtype")
    if not source_hash_ready:
        missing.append("source_sha256")
    if not artifact_ready:
        missing.append("artifact_sha256")
    if not reconstructed_hash:
        missing.append("reconstructed_source_sha256")

    return {
        "model_id": model_id,
        "layer": layer,
        "target_tensor": tensor_name,
        "target_file": hqq_rel_file,
        "target_path": str(hqq_path),
        "artifact_sha256": artifact_hash,
        "artifact_sha256_unavailable_reason": artifact_unavailable_reason,
        "resolver_kind": resolver["resolver_kind"],
        "resolver_ready": resolver["ready"],
        "resolver_unavailable_reason": resolver["reason"],
        "fallback_allowed": False,
        "source_tensors": source_items,
        "source_shape": selected.get("reconstructed_shape") if selected else None,
        "source_dtypes": selected.get("source_dtypes") if selected else [],
        "reconstructed_source_shape": reconstructed_shape,
        "reconstructed_source_sha256": reconstructed_hash,
        "reconstructed_source_sha256_unavailable_reason": reconstructed_hash_unavailable_reason,
        "row_column_layout": selected.get("row_column_layout") if selected else None,
        "fused_vs_split": selected.get("fused_vs_split") if selected else None,
        "recipe": selected.get("recipe") if selected else None,
        "ready": ready,
        "missing_contract_fields": missing,
    }


def build_source_mapping(
    model_dir: Path,
    index: dict[str, Any],
    layer: int,
    tensor_name: str,
) -> dict[str, Any]:
    weight_map = index.get("weight_map", {})
    candidates: list[dict[str, Any]] = []

    def add_candidate(recipe: str, names: list[str], row_column_layout: str, current_resolver_status: str) -> None:
        headers = [tensor_header(model_dir, index, name) for name in names]
        if not all(item.get("exists") for item in headers):
            return
        resolver_status = current_resolver_status
        if resolver_status == "supported_for_qwen35_prefix_only_or_requires_explicit_source_tensor_metadata":
            if all(str(name).startswith("model.language_model.") for name in names):
                resolver_status = "supported_by_current_qwen35_fallback"
            else:
                resolver_status = "requires_explicit_source_tensor_metadata_or_model_layers_direct_resolver"
        shape = headers[0]["shape"] if len(headers) == 1 else merged_source_shape(headers)
        candidates.append({
            "recipe": recipe,
            "source_tensors": headers,
            "reconstructed_shape": shape,
            "source_dtypes": dtype_summary(headers),
            "fused_vs_split": "direct_fused" if len(headers) == 1 else "split_source_reconstructed",
            "row_column_layout": row_column_layout,
            "current_sidecar_resolver_status": resolver_status,
        })

    if tensor_name == "in_proj_qkvz":
        for prefix in source_prefix_candidates(layer, "linear_attn"):
            add_candidate(
                "direct_fused_qkvz",
                [f"{prefix}.in_proj_qkvz.weight"],
                "direct source rows must already match cached qkvz row order; columns are axis-1 HQQ groups",
                "requires_explicit_source_tensor_metadata_or_direct_fused_resolver",
            )
            add_candidate(
                "qwen35_qkvz_split_interleave",
                [f"{prefix}.in_proj_qkv.weight", f"{prefix}.in_proj_z.weight"],
                "reconstruct per key head as [q,k,v,z]; columns are axis-1 HQQ groups",
                "supported_by_current_qwen35_fallback",
            )
    elif tensor_name == "in_proj_ba":
        for prefix in source_prefix_candidates(layer, "linear_attn"):
            add_candidate(
                "direct_fused_ba",
                [f"{prefix}.in_proj_ba.weight"],
                "direct source rows must already match cached ba row order; columns are axis-1 HQQ groups",
                "requires_explicit_source_tensor_metadata_or_direct_fused_resolver",
            )
            add_candidate(
                "qwen35_ba_split_interleave",
                [f"{prefix}.in_proj_b.weight", f"{prefix}.in_proj_a.weight"],
                "reconstruct per key head as [b,a]; columns are axis-1 HQQ groups",
                "supported_by_current_qwen35_fallback",
            )
    elif tensor_name == "out_proj":
        for prefix in source_prefix_candidates(layer, "linear_attn"):
            add_candidate(
                "direct_linear_attention_out_proj",
                [f"{prefix}.out_proj.weight"],
                "contiguous out_proj rows; columns are axis-1 HQQ groups",
                "supported_for_qwen35_prefix_only_or_requires_explicit_source_tensor_metadata",
            )
    elif tensor_name == "fused_qkv":
        for prefix in source_prefix_candidates(layer, "self_attn"):
            add_candidate(
                "self_attn_qkv_concat",
                [f"{prefix}.q_proj.weight", f"{prefix}.k_proj.weight", f"{prefix}.v_proj.weight"],
                "contiguous concat rows [q,k,v]; columns are axis-1 HQQ groups",
                "supported_for_qwen35_prefix_only_or_requires_explicit_source_tensor_metadata",
            )
    elif tensor_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        for prefix in source_prefix_candidates(layer, "self_attn"):
            add_candidate(
                f"direct_self_attn_{tensor_name}",
                [f"{prefix}.{tensor_name}.weight"],
                f"contiguous {tensor_name} rows; columns are axis-1 HQQ groups",
                "supported_for_qwen35_prefix_only_or_requires_explicit_source_tensor_metadata",
            )

    if len(candidates) == 1:
        status = "resolved"
    elif len(candidates) > 1:
        status = "ambiguous"
    else:
        status = "missing"

    missing_patterns = []
    if not candidates:
        for key in sorted(weight_map):
            if f"layers.{layer}." in key and any(part in key for part in ("linear_attn", "self_attn")):
                missing_patterns.append(key)
                if len(missing_patterns) >= 16:
                    break
    return {
        "status": status,
        "candidates": candidates,
        "missing_context_sample": missing_patterns,
    }


def load_hqq_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing HQQ manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest.get("tensors"), list):
        raise RuntimeError(f"HQQ manifest has no tensor list: {manifest_path}")
    return manifest


def source_provenance_fields(entry: dict[str, Any]) -> dict[str, Any]:
    source_keys = sorted(key for key in entry if key.startswith("source"))
    hash_keys = sorted(key for key in entry if "hash" in key.lower())
    return {
        "source_keys": source_keys,
        "hash_keys": hash_keys,
        "has_source_tensor": bool(entry.get("source_tensor")),
        "has_source_hash": any("source" in key.lower() and "hash" in key.lower() for key in hash_keys),
        "has_artifact_hash": any("artifact" in key.lower() and "hash" in key.lower() for key in hash_keys),
    }


def build_source_map_audit(model_dir: Path, cache_dir: Path) -> dict[str, Any]:
    index = load_safetensors_index(model_dir)
    cfg = load_model_text_config(model_dir)
    manifest = load_hqq_manifest(cache_dir)
    model_id = str(model_dir.name)
    shard_names = sorted(set(str(v) for v in index.get("weight_map", {}).values()))
    targets = []
    recipe_counts: dict[str, int] = {}
    resolver_status_counts: dict[str, int] = {}
    strict_resolver_counts: dict[str, int] = {}
    missing = []
    ambiguous = []
    invalid_shape = []
    dtype_issues = []
    metadata_gaps = []
    source_hash_cache: dict[str, str] = {}
    file_hash_cache: dict[str, str] = {}

    for entry in sorted(
        manifest["tensors"],
        key=lambda item: (int(item.get("layer_idx", -1)), str(item.get("tensor_name", ""))),
    ):
        layer = int(entry["layer_idx"])
        tensor_name = str(entry["tensor_name"])
        hqq_shape = [int(v) for v in entry.get("original_shape", [])]
        mapping = build_source_mapping(model_dir, index, layer, tensor_name)
        provenance = source_provenance_fields(entry)
        selected = mapping["candidates"][0] if len(mapping["candidates"]) == 1 else None
        shape_compatible = bool(
            selected
            and selected.get("reconstructed_shape") == hqq_shape
        )
        dtype_ok = bool(selected and set(selected.get("source_dtypes", [])) <= {"BF16", "F16", "F32"})
        current_resolver_ready = bool(
            selected
            and selected.get("current_sidecar_resolver_status") == "supported_by_current_qwen35_fallback"
        )
        source_contract = build_source_contract(
            model_dir=model_dir,
            cache_dir=cache_dir,
            index=index,
            model_id=model_id,
            entry=entry,
            selected=selected,
            shape_compatible=shape_compatible,
            dtype_ok=dtype_ok,
            source_hash_cache=source_hash_cache,
            file_hash_cache=file_hash_cache,
        )
        resolver_kind = source_contract.get("resolver_kind") or "unresolved"
        strict_resolver_counts[resolver_kind] = strict_resolver_counts.get(resolver_kind, 0) + 1
        if selected:
            recipe_counts[selected["recipe"]] = recipe_counts.get(selected["recipe"], 0) + 1
            status_name = selected["current_sidecar_resolver_status"]
            resolver_status_counts[status_name] = resolver_status_counts.get(status_name, 0) + 1
        if mapping["status"] == "missing":
            missing.append({"layer": layer, "tensor": tensor_name, "hqq_shape": hqq_shape})
        if mapping["status"] == "ambiguous":
            ambiguous.append({"layer": layer, "tensor": tensor_name, "candidate_count": len(mapping["candidates"])})
        if selected and not shape_compatible:
            invalid_shape.append({
                "layer": layer,
                "tensor": tensor_name,
                "hqq_shape": hqq_shape,
                "source_shape": selected.get("reconstructed_shape"),
                "recipe": selected.get("recipe"),
            })
        if selected and not dtype_ok:
            dtype_issues.append({
                "layer": layer,
                "tensor": tensor_name,
                "source_dtypes": selected.get("source_dtypes"),
            })
        if not provenance["has_source_tensor"] or not provenance["has_source_hash"] or not provenance["has_artifact_hash"]:
            metadata_gaps.append({
                "layer": layer,
                "tensor": tensor_name,
                "missing": [
                    name
                    for name, present in (
                        ("source_tensor", provenance["has_source_tensor"]),
                        ("source_hash", provenance["has_source_hash"]),
                        ("artifact_hash", provenance["has_artifact_hash"]),
                    )
                    if not present
                ],
            })
        targets.append({
            "layer": layer,
            "tensor": tensor_name,
            "layer_type": entry.get("layer_type"),
            "hqq": {
                "shape": hqq_shape,
                "dtype": entry.get("original_dtype"),
                "layout": entry.get("layout"),
                "axis": entry.get("axis"),
                "group_size": entry.get("group_size"),
                "nbits": entry.get("nbits"),
                "file": entry.get("file"),
            },
            "source_mapping": mapping,
            "selected_mapping": selected,
            "shape_compatible": shape_compatible,
            "source_dtype_ok": dtype_ok,
            "source_provenance": provenance,
            "source_contract": source_contract,
            "shape_dtype_mapping_ready": bool(selected and shape_compatible and dtype_ok and mapping["status"] == "resolved"),
            "current_sidecar_resolver_ready": current_resolver_ready,
            "sidecar_generation_ready": bool(
                selected
                and shape_compatible
                and dtype_ok
                and current_resolver_ready
                and provenance["has_source_tensor"]
                and provenance["has_source_hash"]
                and provenance["has_artifact_hash"]
            ),
        })

    mapping_ready = sum(1 for item in targets if item["shape_dtype_mapping_ready"])
    resolver_ready = sum(1 for item in targets if item["current_sidecar_resolver_ready"])
    strict_resolver_ready = sum(1 for item in targets if item["source_contract"]["resolver_ready"])
    source_contract_ready = sum(1 for item in targets if item["source_contract"]["ready"])
    artifact_hash_ready = sum(
        1 for item in targets if item["source_contract"].get("artifact_sha256")
    )
    source_hash_ready = sum(
        1
        for item in targets
        if item["source_contract"].get("reconstructed_source_sha256")
        and all(
            source.get("source_sha256")
            for source in item["source_contract"].get("source_tensors", [])
        )
    )
    sidecar_generation_ready = sum(1 for item in targets if item["sidecar_generation_ready"])
    linear_targets = [item for item in targets if item["layer_type"] == "linear_attention"]
    gqa_targets = [item for item in targets if item["layer_type"] == "full_attention"]
    return {
        "computed": True,
        "model_dir": str(model_dir),
        "cache_dir": str(cache_dir),
        "model_config": {
            "hidden_size": cfg.get("hidden_size"),
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "linear_num_key_heads": cfg.get("linear_num_key_heads"),
            "linear_num_value_heads": cfg.get("linear_num_value_heads"),
            "linear_key_head_dim": cfg.get("linear_key_head_dim"),
            "linear_value_head_dim": cfg.get("linear_value_head_dim"),
        },
        "source_index": {
            "path": str(model_dir / "model.safetensors.index.json"),
            "tensor_count": len(index.get("weight_map", {})),
            "shard_count": len(shard_names),
            "metadata": index.get("metadata", {}),
        },
        "hqq_manifest": {
            "path": str(cache_dir / "manifest.json"),
            "complete": manifest.get("complete"),
            "format_version": manifest.get("format_version"),
            "layout": manifest.get("layout"),
            "nbits": manifest.get("nbits"),
            "group_size": manifest.get("group_size"),
            "axis": manifest.get("axis"),
            "num_hidden_layers": manifest.get("num_hidden_layers"),
            "totals": manifest.get("totals"),
        },
        "summary": {
            "target_count": len(targets),
            "linear_attention_targets": len(linear_targets),
            "full_attention_targets": len(gqa_targets),
            "resolved_source_targets": sum(1 for item in targets if item["source_mapping"]["status"] == "resolved"),
            "shape_dtype_mapping_ready_targets": mapping_ready,
            "current_sidecar_resolver_ready_targets": resolver_ready,
            "strict_resolver_ready_targets": strict_resolver_ready,
            "source_hash_ready_targets": source_hash_ready,
            "artifact_hash_ready_targets": artifact_hash_ready,
            "source_contract_ready_targets": source_contract_ready,
            "sidecar_generation_ready_targets": sidecar_generation_ready,
            "missing_source_targets": len(missing),
            "ambiguous_source_targets": len(ambiguous),
            "shape_mismatch_targets": len(invalid_shape),
            "dtype_issue_targets": len(dtype_issues),
            "metadata_gap_targets": len(metadata_gaps),
            "recipe_counts": recipe_counts,
            "resolver_status_counts": resolver_status_counts,
            "strict_resolver_counts": strict_resolver_counts,
        },
        "decision": {
            "source_mapping_complete": mapping_ready == len(targets) and not ambiguous and not invalid_shape and not dtype_issues,
            "strict_source_contract_complete": source_contract_ready == len(targets),
            "sidecar_manifest_provenance_complete": len(metadata_gaps) == 0,
            "current_sidecar_generation_ready": sidecar_generation_ready == len(targets),
            "source_contract_generation_ready": source_contract_ready == len(targets),
            "runtime_behavior_changed": False,
            "fallback_added": False,
        },
        "missing_source_targets": missing,
        "ambiguous_source_targets": ambiguous,
        "shape_mismatch_targets": invalid_shape,
        "dtype_issue_targets": dtype_issues,
        "metadata_gaps": metadata_gaps[:64],
        "targets": targets,
    }


def audit_hqq_cache(
    cache_dir: Path,
    *,
    input_row_traces: list[Path],
    input_row_pos: str,
    input_row_tensor: str,
) -> dict[str, Any]:
    pattern = re.compile(r"layer_(?P<layer>\d+)_(?P<tensor>.+)_hqq4\.safetensors$")
    available_rows: dict[str, Any] | None = None
    if input_row_traces:
        rows = load_input_rows_from_trace(
            input_row_traces,
            positions=parse_input_row_positions(input_row_pos),
            layer=0,
            tensor=input_row_tensor,
        )
        available_rows = {
            "tensor": input_row_tensor,
            "layer": 0,
            "events": len(rows),
            "unique_hashes": len({row["hash64"] for row in rows}),
            "positions": sorted({int(row["pos"]) for row in rows}),
            "width": int(rows[0]["n"]) if rows else None,
        }

    tensors = []
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(cache_dir.glob("layer_*_*_hqq4.safetensors")):
        match = pattern.match(path.name)
        if not match:
            continue
        layer = int(match.group("layer"))
        tensor_name = match.group("tensor")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            orig_shape = [int(v) for v in handle.get_tensor("orig_shape").tolist()]
            group_size = int(handle.get_tensor("group_size")[0].item())
            axis = int(handle.get_tensor("axis")[0].item())
            nbits = int(handle.get_tensor("nbits")[0].item())
            scales_shape = list(handle.get_tensor("scales").shape)
            zeros_shape = list(handle.get_tensor("zeros").shape)
            packed_shape = list(handle.get_tensor("packed").shape)

        required_activation = None
        computable = False
        reason = ""
        if layer == 0 and tensor_name in ("in_proj_qkvz", "in_proj_ba"):
            required_activation = {
                "field": "event=input_row_full",
                "tensor": "la_input_row_for_qkvz_last",
                "layer": 0,
                "width": orig_shape[1],
                "why": "linear-attention qkvz and ba projections both consume the layer-0 normalized hidden row",
            }
            computable = bool(available_rows and available_rows.get("width") == orig_shape[1])
            reason = "matching captured layer-0 normalized input rows exist" if computable else (
                "missing full normalized layer-0 la_input_row_for_qkvz_last rows"
            )
        elif tensor_name == "out_proj":
            required_activation = {
                "field": "event=input_row_full",
                "tensor": "la_out_proj_input_last",
                "layer": layer,
                "width": orig_shape[1],
                "why": "linear-attention out_proj consumes the post-FLA/gated-RMSNorm attention row, not the normalized hidden row",
            }
            reason = "missing full post-linear-attention out_proj input row"
        elif tensor_name in ("in_proj_qkvz", "in_proj_ba"):
            required_activation = {
                "field": "event=input_row_full",
                "tensor": "la_input_row_for_qkvz_last",
                "layer": layer,
                "width": orig_shape[1],
                "why": "projection sensitivity needs the layer-specific normalized hidden row",
            }
            reason = f"missing full normalized layer-{layer} la_input_row_for_qkvz_last rows"
        elif tensor_name in ("q_proj", "k_proj", "v_proj", "fused_qkv"):
            required_activation = {
                "field": "event=input_row_full",
                "tensor": "gqa_input_norm_last",
                "layer": layer,
                "width": orig_shape[1],
                "why": "GQA q/k/v/fused_qkv projections consume the layer-specific attention input row",
            }
            reason = f"missing full GQA projection input row for layer {layer}"
        elif tensor_name == "o_proj":
            required_activation = {
                "field": "event=input_row_full",
                "tensor": "gqa_o_proj_input_last",
                "layer": layer,
                "width": orig_shape[1],
                "why": "GQA o_proj consumes the attention output row, not the q/k/v projection input row",
            }
            reason = f"missing full GQA o_proj input row for layer {layer}"
        else:
            reason = "unsupported tensor family for offline activation-sensitive projection diagnostics"

        entry = {
            "layer": layer,
            "tensor": tensor_name,
            "path": str(path),
            "layer_type": metadata.get("layer_type"),
            "backend": metadata.get("backend"),
            "layout": metadata.get("layout"),
            "orig_shape": orig_shape,
            "group_size": group_size,
            "axis": axis,
            "nbits": nbits,
            "packed_shape": packed_shape,
            "scales_shape": scales_shape,
            "zeros_shape": zeros_shape,
            "offline_decision_summary_computable": computable,
            "required_activation": required_activation,
            "not_computable_reason": None if computable else reason,
        }
        tensors.append(entry)
        by_layer.setdefault(layer, []).append(entry)

    layer0 = [entry for entry in tensors if entry["layer"] == 0]
    return {
        "computed": True,
        "cache_dir": str(cache_dir),
        "available_input_rows": available_rows,
        "summary": {
            "tensor_count": len(tensors),
            "layer_count": len(by_layer),
            "layer0_tensors": [entry["tensor"] for entry in layer0],
            "layer0_computable": [
                entry["tensor"] for entry in layer0 if entry["offline_decision_summary_computable"]
            ],
            "layer0_not_computable": [
                {
                    "tensor": entry["tensor"],
                    "missing_field": entry["required_activation"],
                    "reason": entry["not_computable_reason"],
                }
                for entry in layer0 if not entry["offline_decision_summary_computable"]
            ],
        },
        "tensors": tensors,
    }


def scan_input_row_events(paths: list[Path]) -> dict[str, Any]:
    pattern = re.compile(
        r"event=input_row_full\b.*\bpos=(?P<pos>\d+)\b.*\blayer=(?P<layer>\d+)\b"
        r".*\btensor=(?P<tensor>\S+)\b.*\bn=(?P<n>\d+)\b.*\bhash64=(?P<hash>0x[0-9a-fA-F]+)\b"
    )
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    total_events = 0
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                if "event=input_row_full" not in line:
                    continue
                match = pattern.search(line)
                if not match:
                    continue
                total_events += 1
                layer = int(match.group("layer"))
                tensor = match.group("tensor")
                pos = int(match.group("pos"))
                width = int(match.group("n"))
                key = (layer, tensor)
                slot = by_key.setdefault(key, {
                    "layer": layer,
                    "tensor": tensor,
                    "width": width,
                    "events": 0,
                    "positions": set(),
                    "hashes": set(),
                    "artifacts": set(),
                    "first_location": {"path": str(path), "line": line_no},
                })
                slot["events"] += 1
                slot["positions"].add(pos)
                slot["hashes"].add(match.group("hash"))
                slot["artifacts"].add(str(path))
                if slot["width"] != width:
                    slot["width_mismatch"] = sorted({int(slot["width"]), width})
    entries = []
    for slot in sorted(by_key.values(), key=lambda item: (int(item["layer"]), str(item["tensor"]))):
        entries.append({
            **{k: v for k, v in slot.items() if k not in ("positions", "hashes", "artifacts")},
            "positions": sorted(int(v) for v in slot["positions"]),
            "unique_hashes": len(slot["hashes"]),
            "artifacts": sorted(str(v) for v in slot["artifacts"]),
        })
    return {
        "paths": [str(path) for path in paths],
        "total_events": total_events,
        "entries": entries,
    }


def cache_entries(cache_dir: Path) -> list[dict[str, Any]]:
    pattern = re.compile(r"layer_(?P<layer>\d+)_(?P<tensor>.+)_hqq4\.safetensors$")
    entries = []
    for path in sorted(cache_dir.glob("layer_*_*_hqq4.safetensors")):
        match = pattern.match(path.name)
        if not match:
            continue
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            entries.append({
                "layer": int(match.group("layer")),
                "tensor": match.group("tensor"),
                "path": str(path),
                "layer_type": metadata.get("layer_type"),
                "backend": metadata.get("backend"),
                "layout": metadata.get("layout"),
                "orig_shape": [int(v) for v in handle.get_tensor("orig_shape").tolist()],
                "group_size": int(handle.get_tensor("group_size")[0].item()),
                "axis": int(handle.get_tensor("axis")[0].item()),
                "nbits": int(handle.get_tensor("nbits")[0].item()),
                "packed_shape": list(handle.get_tensor("packed").shape),
                "scales_shape": list(handle.get_tensor("scales").shape),
                "zeros_shape": list(handle.get_tensor("zeros").shape),
            })
    return entries


def activation_requirement_for(layer: int, tensor_name: str, shape: list[int]) -> dict[str, Any]:
    if tensor_name in ("in_proj_qkvz", "in_proj_ba"):
        return {
            "field": "event=input_row_full",
            "tensor": "la_input_row_for_qkvz_last",
            "layer": layer,
            "width": int(shape[1]),
            "producer_capture_site": {
                "file": "src/gpu_prefill.rs",
                "function": "PrefillRunner::forward_linear_attention",
                "current_reference": "before la_in_proj_qkvz GEMM, hidden row; existing trace_emit_bf16_row_full_words site near lines 9768-9780",
            },
            "why": "linear-attention in_proj_qkvz and in_proj_ba both consume the layer-specific normalized hidden row",
        }
    if tensor_name == "out_proj":
        return {
            "field": "event=input_row_full",
            "tensor": "la_out_proj_input_last",
            "layer": layer,
            "width": int(shape[1]),
            "producer_capture_site": {
                "file": "src/gpu_prefill.rs",
                "function": "PrefillRunner::forward_linear_attention",
                "current_reference": "before la_out_proj GEMM, rmsnorm_out row; current summary-only site near lines 12886-12896",
            },
            "why": "linear-attention out_proj consumes the post-FLA/gated-RMSNorm attention row, not the normalized hidden row",
        }
    if tensor_name in ("q_proj", "k_proj", "v_proj", "fused_qkv"):
        return {
            "field": "event=input_row_full",
            "tensor": "gqa_input_norm_last",
            "layer": layer,
            "width": int(shape[1]),
            "producer_capture_site": {
                "file": "src/gpu_prefill.rs",
                "function": "PrefillRunner::forward_gqa_chunked",
                "current_reference": "before q_proj/k_proj/v_proj GEMMs, hidden row; projection calls near lines 8377-8416",
            },
            "why": "GQA q/k/v/fused_qkv projections consume the layer-specific normalized attention input row",
        }
    if tensor_name == "o_proj":
        return {
            "field": "event=input_row_full",
            "tensor": "gqa_o_proj_input_last",
            "layer": layer,
            "width": int(shape[1]),
            "producer_capture_site": {
                "file": "src/gpu_prefill.rs",
                "function": "PrefillRunner::forward_gqa_chunked",
                "current_reference": "before GQA o_proj GEMM, attn_out row; o_proj call near lines 9385-9388",
            },
            "why": "GQA o_proj consumes the attention output row, not the q/k/v projection input row",
        }
    return {
        "field": "event=input_row_full",
        "tensor": "unknown",
        "layer": layer,
        "width": int(shape[1]),
        "producer_capture_site": None,
        "why": "unsupported tensor family for offline activation-sensitive projection diagnostics",
    }


def case_position_plan(case_summary_path: Path | None) -> dict[str, Any]:
    default_positions = [4, 8, 12, 16, 19, 24, 27, 30, 36, 64, 128, 255, 315, 429, 447, 509]
    if case_summary_path is None:
        return {
            "source": "default selector from prior 97-row Qwen3.5 trace",
            "positions_selector": ",".join(str(v) for v in default_positions),
            "case_count": None,
            "row_count": None,
            "cases": [],
        }
    with case_summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    cases = []
    all_positions: set[int] = set()
    for result in summary.get("prompt_results", []):
        trace = result.get("trace_input_row") or {}
        positions = [int(v) for v in trace.get("positions", [])]
        if not positions and trace.get("pos") is not None:
            positions = [int(trace["pos"])]
        for pos in positions:
            all_positions.add(pos)
        cases.append({
            "conv_idx": result.get("conv_idx"),
            "turn": result.get("turn"),
            "case_key": trace.get("case_key", f"{result.get('conv_idx')}:{result.get('turn', 1) - 1}"),
            "positions": positions,
            "row_count": len(positions),
            "prompt_prefix": (result.get("prompt") or "")[:120],
        })
    return {
        "source": str(case_summary_path),
        "positions_selector": ",".join(str(v) for v in sorted(all_positions)),
        "case_count": len(cases),
        "row_count": sum(int(case["row_count"]) for case in cases),
        "cases": cases,
    }


def build_capture_requirements(
    cache_dir: Path,
    *,
    input_row_traces: list[Path],
    case_summary_path: Path | None,
) -> dict[str, Any]:
    entries = cache_entries(cache_dir)
    by_key = {(int(entry["layer"]), str(entry["tensor"])): entry for entry in entries}
    existing = scan_input_row_events(input_row_traces) if input_row_traces else {
        "paths": [],
        "total_events": 0,
        "entries": [],
    }
    existing_by_key = {
        (int(entry["layer"]), str(entry["tensor"])): entry
        for entry in existing["entries"]
    }

    la_layers = sorted({int(entry["layer"]) for entry in entries if entry["layer_type"] == "linear_attention"})
    gqa_layers = sorted({int(entry["layer"]) for entry in entries if entry["layer_type"] == "full_attention"})
    deeper_la_layers = [layer for layer in (4, 12, 20, 28, 36) if layer in la_layers]
    representative_gqa_layers = [layer for layer in (3, 19, 39) if layer in gqa_layers]

    selected: list[tuple[int, str, str]] = []
    selected.append((0, "out_proj", "complete_layer0_linear_attention"))
    for layer in deeper_la_layers:
        for tensor_name in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
            selected.append((layer, tensor_name, "selected_deeper_linear_attention"))
    for layer in representative_gqa_layers:
        for tensor_name in ("fused_qkv", "q_proj", "k_proj", "v_proj", "o_proj"):
            selected.append((layer, tensor_name, "representative_gqa"))

    targets = []
    missing_by_activation: dict[str, dict[str, Any]] = {}
    for layer, tensor_name, reason in selected:
        entry = by_key.get((layer, tensor_name))
        if entry is None:
            continue
        required = activation_requirement_for(layer, tensor_name, entry["orig_shape"])
        existing_entry = existing_by_key.get((int(required["layer"]), str(required["tensor"])))
        has_existing = bool(existing_entry and int(existing_entry.get("width", -1)) == int(required["width"]))
        target = {
            "selection_reason": reason,
            "layer": layer,
            "tensor": tensor_name,
            "layer_type": entry["layer_type"],
            "hqq": {
                "shape": entry["orig_shape"],
                "group_size": entry["group_size"],
                "axis": entry["axis"],
                "nbits": entry["nbits"],
                "packed_shape": entry["packed_shape"],
                "scales_shape": entry["scales_shape"],
                "zeros_shape": entry["zeros_shape"],
                "path": entry["path"],
            },
            "required_activation": required,
            "existing_artifact": {
                "contains_required_row": has_existing,
                "matched_event_summary": existing_entry,
            },
            "offline_status": "computable_with_existing_artifact" if has_existing else "not_computable_offline",
        }
        targets.append(target)
        if not has_existing:
            key = f"L{layer}:{required['tensor']}:{required['width']}"
            slot = missing_by_activation.setdefault(key, {
                "layer": layer,
                "tensor": required["tensor"],
                "width": required["width"],
                "field": required["field"],
                "producer_capture_site": required["producer_capture_site"],
                "targets": [],
            })
            slot["targets"].append({
                "layer": layer,
                "tensor": tensor_name,
                "shape": entry["orig_shape"],
                "selection_reason": reason,
            })

    prompt_plan = case_position_plan(case_summary_path)
    return {
        "computed": True,
        "cache_dir": str(cache_dir),
        "scope": {
            "purpose": "minimal activation-row capture plan for offline HQQ decision-quality diagnostics",
            "selected_deeper_linear_attention_layers": deeper_la_layers,
            "representative_gqa_layers": representative_gqa_layers,
            "selection_policy": (
                "complete layer-0 linear attention by adding out_proj, then sample first/mid/late deeper "
                "linear-attention and GQA layers without broadening to all 140 tensors yet"
            ),
        },
        "existing_input_row_artifacts": existing,
        "minimal_prompt_case_position_set": {
            **prompt_plan,
            "comparability_note": (
                "Reuse the same case/absolute-position mapping as the current 97-row QKVZ/BA result so "
                "next summaries are comparable; positions outside a prompt are naturally absent."
            ),
        },
        "targets": targets,
        "missing_activation_rows": list(missing_by_activation.values()),
        "smallest_next_runtime_step": {
            "complete": bool(targets),
            "no_runtime_change_made_by_this_artifact": True,
            "why_bf16_and_hqq": (
                "Rows after layer-0 in_proj_qkvz/BA are not provably attention-quant independent. "
                "Capture BF16 and HQQ in the same narrow input-row-only mode and verify hashes per case/position."
            ),
            "commands_to_run_after_runtime_capture_selector_exists": [
                "KRASIS_TRACE=1 KRASIS_TRACE_COMPONENTS=prefill_layer_summary "
                "KRASIS_TRACE_INPUT_ROWS_ONLY=1 "
                f"KRASIS_TRACE_INPUT_ROW_POSITIONS={prompt_plan['positions_selector']} "
                "KRASIS_TRACE_INPUT_ROW_TENSORS=<planned tensor list> "
                "KRASIS_TRACE_INPUT_ROW_LAYERS=<planned layer list> "
                "./dev witness-compare tests/q35b-4-4-a16.conf --profile llama_witness_qwen35_expanded_thinking_off --startup-timeout 1200",
                "KRASIS_TRACE=1 KRASIS_TRACE_COMPONENTS=prefill_layer_summary "
                "KRASIS_TRACE_INPUT_ROWS_ONLY=1 "
                f"KRASIS_TRACE_INPUT_ROW_POSITIONS={prompt_plan['positions_selector']} "
                "KRASIS_TRACE_INPUT_ROW_TENSORS=<planned tensor list> "
                "KRASIS_TRACE_INPUT_ROW_LAYERS=<planned layer list> "
                "./dev witness-compare tests/q35b-4-4-hqq4.conf --profile llama_witness_qwen35_expanded_thinking_off --startup-timeout 1200",
            ],
        },
        "summary": {
            "target_count": len(targets),
            "missing_activation_row_kinds": len(missing_by_activation),
            "already_computable_targets": sum(
                1 for target in targets if target["existing_artifact"]["contains_required_row"]
            ),
            "not_computable_targets": sum(
                1 for target in targets if not target["existing_artifact"]["contains_required_row"]
            ),
        },
    }


def summarize_float_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "max": None, "min": None}
    return {
        "mean": sum(values) / len(values),
        "max": max(values),
        "min": min(values),
    }


def activation_row_pair_metrics(bf16: torch.Tensor, hqq: torch.Tensor) -> dict[str, float]:
    if bf16.numel() != hqq.numel():
        raise RuntimeError(f"Activation row width mismatch: bf16={bf16.numel()} hqq={hqq.numel()}")
    diff = hqq - bf16
    rms = float(torch.sqrt(torch.mean(diff * diff)).item()) if diff.numel() else 0.0
    linf = float(torch.max(torch.abs(diff)).item()) if diff.numel() else 0.0
    bf_norm = float(torch.linalg.vector_norm(bf16).item())
    hq_norm = float(torch.linalg.vector_norm(hqq).item())
    cosine = float(torch.dot(bf16, hqq).item() / max(bf_norm * hq_norm, 1e-30))
    return {
        "rms": rms,
        "linf": linf,
        "cosine": cosine,
        "bf16_l2": bf_norm,
        "hqq_l2": hq_norm,
    }


def build_case_candidate_index(case_summary_path: Path | None) -> dict[str, Any]:
    if case_summary_path is None:
        return {"cases": [], "by_pos": {}}
    summary = load_case_summary_doc(case_summary_path)
    cases = []
    by_pos: dict[int, list[dict[str, Any]]] = {}
    for index, result in enumerate(summary.get("prompt_results", [])):
        trace = result.get("trace_input_row") or {}
        positions = [int(pos) for pos in trace.get("positions", [])]
        if not positions and trace.get("pos") is not None:
            positions = [int(trace["pos"])]
        case = {
            "prompt_sequence_index": index,
            "conv_idx": result.get("conv_idx"),
            "turn": result.get("turn"),
            "case_key": trace.get("case_key", f"{result.get('conv_idx')}:{result.get('turn', 1) - 1}"),
            "prompt_prefix": trace.get("prompt_prefix", result.get("prompt", "")[:120]),
            "positions": positions,
        }
        cases.append(case)
        for pos in positions:
            by_pos.setdefault(pos, []).append(case)
    return {"cases": cases, "by_pos": by_pos}


def _row_group_entries_for_width(width: int, group_size: int = 128) -> list[dict[str, int]]:
    return [
        {
            "group": group,
            "start_col": start,
            "end_col": min(start + group_size, width),
        }
        for group, start in enumerate(range(0, width, group_size))
    ]


def _case_split_for_case(case: dict[str, Any]) -> str:
    conv_idx = case.get("conv_idx")
    if conv_idx is None:
        seq = int(case.get("prompt_sequence_index", 0))
        return "train" if seq % 2 == 0 else "heldout"
    return "train" if int(conv_idx) % 2 == 0 else "heldout"


def _event_public_meta(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": event.get("path"),
        "line": event.get("line"),
        "global_ordinal": event.get("global_ordinal"),
        "field_ordinal": event.get("field_ordinal"),
        "pos": event.get("pos"),
        "layer": event.get("layer"),
        "tensor": event.get("tensor"),
        "n": event.get("n"),
        "hash64": event.get("hash64"),
        "bf16_hex_sample": event.get("bf16_hex_sample"),
    }


def build_case_row_audit(
    *,
    model_dir: Path,
    input_row_traces: list[Path],
    case_summary_path: Path,
    positions: set[int] | None,
    layer: int,
    tensor: str,
    output_path: Path,
) -> dict[str, Any]:
    all_events: list[dict[str, Any]] = []
    for path in input_row_traces:
        all_events.extend(parse_all_input_row_events(path))
    matching_events = [
        event
        for event in all_events
        if int(event["layer"]) == int(layer)
        and str(event["tensor"]) == tensor
        and (positions is None or int(event["pos"]) in positions)
    ]
    case_map = load_case_map(case_summary_path)
    case_spans = parse_reference_case_spans(input_row_traces)
    if case_spans:
        aligned = align_input_rows_to_case_spans(matching_events, case_spans)
        align_method = "explicit_case_span"
        case_prefix_by_sequence = {
            int(case["prompt_sequence_index"]): case.get("prompt_prefix")
            for case in case_map
            if case.get("prompt_sequence_index") is not None
        }
    else:
        aligned = align_input_rows_to_cases_detailed(matching_events, case_map)
        align_method = "ordered_position"
        case_prefix_by_sequence = {}
    group_size = 128
    case_rows = []
    train_lines: list[str] = []
    heldout_lines: list[str] = []
    for item in aligned["aligned"]:
        event = item["event"]
        case = item.get("case") or {}
        if not case.get("prompt_prefix") and case.get("prompt_sequence_index") is not None:
            case["prompt_prefix"] = case_prefix_by_sequence.get(int(case["prompt_sequence_index"]))
        split = _case_split_for_case(case)
        raw_line = event.get("raw_line")
        if raw_line:
            if split == "train":
                train_lines.append(raw_line)
            else:
                heldout_lines.append(raw_line)
        row_groups = _row_group_entries_for_width(int(event["n"]), group_size=group_size)
        case_rows.append(
            {
                "model_id": model_dir.name,
                "split": split,
                "case_key": case.get("case_key"),
                "conv_idx": case.get("conv_idx"),
                "turn": case.get("turn"),
                "prompt_sequence_index": case.get("prompt_sequence_index"),
                "position_sequence_index": case.get("position_sequence_index"),
                "prompt_prefix": case.get("prompt_prefix"),
                "position": int(event["pos"]),
                "layer": int(event["layer"]),
                "tensor": str(event["tensor"]),
                "row_hash": event["hash64"],
                "row_width": int(event["n"]),
                "trace_path": event.get("path"),
                "trace_line": event.get("line"),
                "field_ordinal": event.get("field_ordinal"),
                "align_status": item.get("align_status"),
                "row_group_count": len(row_groups),
                "row_groups": row_groups,
            }
        )

    train_trace = output_path.with_suffix(".train.trace.log")
    heldout_trace = output_path.with_suffix(".heldout.trace.log")
    train_trace.write_text("\n".join(train_lines) + ("\n" if train_lines else ""), encoding="utf-8")
    heldout_trace.write_text("\n".join(heldout_lines) + ("\n" if heldout_lines else ""), encoding="utf-8")

    by_split: dict[str, int] = {}
    for row in case_rows:
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
    duplicate_positions: dict[int, int] = {}
    for case in case_map:
        pos = int(case["pos"])
        duplicate_positions[pos] = duplicate_positions.get(pos, 0) + 1
    duplicate_positions = {pos: count for pos, count in duplicate_positions.items() if count > 1}
    report = {
        "format": "krasis_hqq_case_row_audit",
        "format_version": 1,
        "model_dir": str(model_dir),
        "case_summary_path": str(case_summary_path),
        "input_row_traces": [str(path) for path in input_row_traces],
        "selection": {
            "layer": layer,
            "tensor": tensor,
            "positions": "all" if positions is None else sorted(positions),
            "split_rule": "conv_idx parity: even=train, odd=heldout",
            "row_group_size": group_size,
            "align_method": align_method,
        },
        "summary": {
            "case_count": len(case_map),
            "explicit_case_spans": len(case_spans),
            "matching_input_row_events": len(matching_events),
            "mapped_case_rows": len(case_rows),
            "train_rows": by_split.get("train", 0),
            "heldout_rows": by_split.get("heldout", 0),
            "missing_case_rows": len(aligned["missing_cases"]),
            "unused_matching_events": len(aligned["unused_events"]),
            "duplicate_case_positions": duplicate_positions,
            "case_identity": (
                "Rows are mapped by explicit reference_case_boundary spans emitted by the "
                "non-production witness harness."
                if case_spans
                else "Raw input_row_full trace lines do not carry case IDs. This audit maps rows to "
                "witness cases by ordered position using the witness summary's trace_input_row map; "
                "duplicate positions are accepted only by sequence order and are reported."
            ),
        },
        "case_rows": case_rows,
        "missing_cases": aligned["missing_cases"],
        "unused_events": [_event_public_meta(event) for event in aligned["unused_events"][:128]],
        "artifacts": {
            "train_trace": str(train_trace),
            "heldout_trace": str(heldout_trace),
        },
    }
    return report


def compact_case_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_key": case.get("case_key"),
            "conv_idx": case.get("conv_idx"),
            "turn": case.get("turn"),
            "prompt_prefix": case.get("prompt_prefix"),
        }
        for case in candidates
    ]


def field_stage(layer: int, tensor: str) -> str:
    if layer == 0 and tensor == "la_input_row_for_qkvz_last":
        return "layer0_pre_hqq_la_input"
    if layer == 0 and tensor == "la_out_proj_input_last":
        return "layer0_out_proj_input"
    if tensor.startswith("la_"):
        return "deeper_linear_attention"
    if tensor.startswith("gqa_"):
        return "gqa"
    return "other"


def field_order(layer: int, tensor: str) -> tuple[int, int, str]:
    stage_rank = {
        "la_input_row_for_qkvz_last": 0,
        "la_out_proj_input_last": 1,
        "gqa_input_norm_last": 0,
        "gqa_o_proj_input_last": 1,
    }.get(tensor, 9)
    return (int(layer), stage_rank, tensor)


def build_named_row_divergence_map(
    *,
    bf16_trace: Path,
    hqq_trace: Path,
    bf16_summary: Path | None,
    hqq_summary: Path | None,
) -> dict[str, Any]:
    bf16_events = parse_all_input_row_events(bf16_trace)
    hqq_events = parse_all_input_row_events(hqq_trace)
    case_index = build_case_candidate_index(bf16_summary)
    case_by_pos = case_index["by_pos"]

    def group_events(events: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for event in events:
            grouped.setdefault((int(event["layer"]), str(event["tensor"])), []).append(event)
        return grouped

    bf_by_field = group_events(bf16_events)
    hq_by_field = group_events(hqq_events)
    fields = sorted(set(bf_by_field) | set(hq_by_field), key=lambda item: field_order(item[0], item[1]))
    field_summaries = []
    pair_records = []
    position_matrix: dict[tuple[int, str, int], dict[str, Any]] = {}
    case_matrix: dict[tuple[int, str, str, int], dict[str, Any]] = {}
    stage_matrix: dict[str, dict[str, Any]] = {}
    first_mismatch: dict[str, Any] | None = None

    def update_slot(slot: dict[str, Any], record: dict[str, Any]) -> None:
        slot["pairs"] += 1
        slot["match_count"] += int(record["match"])
        slot["mismatch_count"] += int(not record["match"])
        slot["rms_values"].append(float(record["rms"]))
        slot["linf_values"].append(float(record["linf"]))
        slot["cosine_values"].append(float(record["cosine"]))
        slot["widths"].add(int(record["width"]))
        slot["bf16_hashes"].add(record["bf16_hash"])
        slot["hqq_hashes"].add(record["hqq_hash"])
        if not record["match"] and slot.get("first_mismatch") is None:
            slot["first_mismatch"] = {
                key: record[key]
                for key in (
                    "field_ordinal",
                    "pos",
                    "bf16_hash",
                    "hqq_hash",
                    "rms",
                    "linf",
                    "cosine",
                    "case_candidates",
                )
            }

    def finalize_slot(slot: dict[str, Any]) -> dict[str, Any]:
        rms = slot.pop("rms_values")
        linf = slot.pop("linf_values")
        cosine = slot.pop("cosine_values")
        bf_hashes = slot.pop("bf16_hashes")
        hq_hashes = slot.pop("hqq_hashes")
        widths = slot.pop("widths")
        return {
            **slot,
            "widths": sorted(widths),
            "bf16_unique_hashes": len(bf_hashes),
            "hqq_unique_hashes": len(hq_hashes),
            "rms": summarize_float_values(rms),
            "linf": summarize_float_values(linf),
            "cosine": summarize_float_values(cosine),
        }

    for layer, tensor in fields:
        bf_events = bf_by_field.get((layer, tensor), [])
        hq_events = hq_by_field.get((layer, tensor), [])
        pairs = min(len(bf_events), len(hq_events))
        summary_slot = {
            "layer": layer,
            "tensor": tensor,
            "stage": field_stage(layer, tensor),
            "bf16_events": len(bf_events),
            "hqq_events": len(hq_events),
            "paired_events": pairs,
            "unpaired_bf16_events": max(0, len(bf_events) - pairs),
            "unpaired_hqq_events": max(0, len(hq_events) - pairs),
            "pairs": 0,
            "match_count": 0,
            "mismatch_count": 0,
            "widths": set(),
            "bf16_hashes": set(),
            "hqq_hashes": set(),
            "rms_values": [],
            "linf_values": [],
            "cosine_values": [],
            "first_mismatch": None,
        }
        for idx in range(pairs):
            bf = bf_events[idx]
            hq = hq_events[idx]
            metrics_out = activation_row_pair_metrics(bf["values"], hq["values"])
            match = (
                int(bf["n"]) == int(hq["n"])
                and str(bf["hash64"]) == str(hq["hash64"])
            )
            pos = int(bf["pos"])
            candidates = compact_case_candidates(case_by_pos.get(pos, []))
            record = {
                "layer": layer,
                "tensor": tensor,
                "stage": field_stage(layer, tensor),
                "field_ordinal": idx,
                "bf16_line": bf["line"],
                "hqq_line": hq["line"],
                "pos": pos,
                "hqq_pos": int(hq["pos"]),
                "position_match": pos == int(hq["pos"]),
                "width": int(bf["n"]),
                "hqq_width": int(hq["n"]),
                "bf16_hash": bf["hash64"],
                "hqq_hash": hq["hash64"],
                "match": match,
                "case_candidates": candidates,
                **metrics_out,
            }
            pair_records.append(record)
            update_slot(summary_slot, record)

            pos_key = (layer, tensor, pos)
            pos_slot = position_matrix.setdefault(pos_key, {
                "layer": layer,
                "tensor": tensor,
                "pos": pos,
                "pairs": 0,
                "match_count": 0,
                "mismatch_count": 0,
                "widths": set(),
                "bf16_hashes": set(),
                "hqq_hashes": set(),
                "rms_values": [],
                "linf_values": [],
                "cosine_values": [],
                "first_mismatch": None,
                "case_candidates": candidates,
            })
            update_slot(pos_slot, record)

            for case in candidates:
                case_key = str(case.get("case_key"))
                case_slot = case_matrix.setdefault((layer, tensor, case_key, pos), {
                    "layer": layer,
                    "tensor": tensor,
                    "case_key": case_key,
                    "conv_idx": case.get("conv_idx"),
                    "turn": case.get("turn"),
                    "prompt_prefix": case.get("prompt_prefix"),
                    "pos": pos,
                    "pairs": 0,
                    "match_count": 0,
                    "mismatch_count": 0,
                    "widths": set(),
                    "bf16_hashes": set(),
                    "hqq_hashes": set(),
                    "rms_values": [],
                    "linf_values": [],
                    "cosine_values": [],
                    "first_mismatch": None,
                })
                update_slot(case_slot, record)

            stage = record["stage"]
            stage_slot = stage_matrix.setdefault(stage, {
                "stage": stage,
                "pairs": 0,
                "match_count": 0,
                "mismatch_count": 0,
                "widths": set(),
                "bf16_hashes": set(),
                "hqq_hashes": set(),
                "rms_values": [],
                "linf_values": [],
                "cosine_values": [],
                "first_mismatch": None,
            })
            update_slot(stage_slot, record)
            if first_mismatch is None and not match:
                first_mismatch = {
                    "stage": record["stage"],
                    "layer": layer,
                    "tensor": tensor,
                    "field_ordinal": idx,
                    "pos": pos,
                    "bf16_hash": bf["hash64"],
                    "hqq_hash": hq["hash64"],
                    "rms": record["rms"],
                    "linf": record["linf"],
                    "cosine": record["cosine"],
                    "case_candidates": candidates,
                }
        field_summaries.append(finalize_slot(summary_slot))

    position_entries = [
        finalize_slot(slot) for _, slot in sorted(position_matrix.items(), key=lambda item: (item[0][0], item[0][1], item[0][2]))
    ]
    case_entries = [
        finalize_slot(slot) for _, slot in sorted(case_matrix.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3]))
    ]
    stage_entries = [
        finalize_slot(slot) for _, slot in sorted(stage_matrix.items(), key=lambda item: item[0])
    ]
    matched = sum(int(record["match"]) for record in pair_records)
    mismatched = len(pair_records) - matched
    return {
        "computed": True,
        "bf16_trace": str(bf16_trace),
        "hqq_trace": str(hqq_trace),
        "bf16_summary": str(bf16_summary) if bf16_summary else None,
        "hqq_summary": str(hqq_summary) if hqq_summary else None,
        "summary": {
            "bf16_events": len(bf16_events),
            "hqq_events": len(hqq_events),
            "paired_events": len(pair_records),
            "match_count": matched,
            "mismatch_count": mismatched,
            "field_count": len(field_summaries),
            "case_identity": (
                "Trace events do not carry explicit case IDs. Case entries are mapped from the "
                "reference summary by absolute position and therefore use case_candidates when a "
                "position appears in multiple witness prompts."
            ),
        },
        "first_mismatch": first_mismatch,
        "divergence_attribution": {
            "first_visible_divergence": first_mismatch,
            "stage_summary": stage_entries,
            "interpretation": (
                "Layer-0 la_input_row_for_qkvz_last is the only shared BF16/HQQ activation field "
                "in the captured evidence. Divergence first appears at layer-0 la_out_proj_input_last; "
                "deeper LA and GQA fields are downstream of earlier HQQ differences."
            ),
        },
        "per_lane_evidence_design": {
            "shared_activation_safe": [
                {
                    "activation_field": "layer 0 la_input_row_for_qkvz_last",
                    "usable_for_tensors": ["layer 0 in_proj_qkvz", "layer 0 in_proj_ba"],
                    "why": "Captured BF16/HQQ rows match exactly before HQQ attention math is applied.",
                }
            ],
            "requires_per_lane_local_error": [
                {
                    "activation_field": "layer 0 la_out_proj_input_last",
                    "usable_for_tensors": ["layer 0 out_proj"],
                    "why": "Rows differ after the layer-0 linear-attention core; BF16 rows are not a shared input for HQQ out_proj.",
                },
                {
                    "activation_field": "deeper LA la_input_row_for_qkvz_last and la_out_proj_input_last",
                    "usable_for_tensors": ["selected deeper LA in_proj_qkvz", "selected deeper LA in_proj_ba", "selected deeper LA out_proj"],
                    "why": "Rows already include accumulated upstream BF16/HQQ divergence.",
                },
                {
                    "activation_field": "GQA gqa_input_norm_last and gqa_o_proj_input_last",
                    "usable_for_tensors": ["selected GQA q/k/v/fused_qkv/o_proj"],
                    "why": "Rows are downstream of prior HQQ attention differences and do not match BF16.",
                },
            ],
            "must_not_do": (
                "Do not use BF16 downstream/deeper/GQA activations as shared inputs for HQQ quantizer evidence; "
                "that would mix weight error with lane-specific activation drift."
            ),
        },
        "field_layer_matrix": field_summaries,
        "field_layer_position_matrix": position_entries,
        "field_layer_case_position_matrix": case_entries,
    }


def main() -> int:
    require_dev_entrypoint()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tensor", default="in_proj_qkvz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tsv-output")
    parser.add_argument("--sample-cols", type=int, default=8)
    parser.add_argument("--top-rows", type=int, default=12)
    parser.add_argument("--top-groups", type=int, default=12)
    parser.add_argument("--input-row-trace", action="append")
    parser.add_argument("--input-row-pos", default="36")
    parser.add_argument("--input-row-tensor", default="la_input_row_for_qkvz_last")
    parser.add_argument("--case-summary")
    parser.add_argument("--max-input-rows", type=int)
    parser.add_argument("--focus-row", type=int)
    parser.add_argument("--focus-group", type=int)
    parser.add_argument("--candidate-grid-steps", type=int, default=129)
    parser.add_argument("--candidate-local-grid-steps", type=int, default=65)
    parser.add_argument("--bf16-trace")
    parser.add_argument("--hqq-trace")
    parser.add_argument("--bf16-summary")
    parser.add_argument("--hqq-summary")
    args = parser.parse_args()
    global BEST_FIT_GRID_STEPS, BEST_FIT_LOCAL_GRID_STEPS
    BEST_FIT_GRID_STEPS = max(3, int(args.candidate_grid_steps))
    BEST_FIT_LOCAL_GRID_STEPS = max(3, int(args.candidate_local_grid_steps))
    timer = StageTimer()

    model_dir = Path(args.model_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    if args.tensor == "cache-audit":
        audit = audit_hqq_cache(
            cache_dir,
            input_row_traces=[Path(item).resolve() for item in (args.input_row_trace or [])],
            input_row_pos=args.input_row_pos,
            input_row_tensor=args.input_row_tensor,
        )
        report = {
            "format": "krasis_hqq_attention_cache_audit",
            "model_dir": str(model_dir),
            **audit,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        if args.tsv_output:
            tsv = Path(args.tsv_output)
            tsv.parent.mkdir(parents=True, exist_ok=True)
            with tsv.open("w", encoding="utf-8") as f:
                f.write("layer\ttensor\tshape\tgroup_size\taxis\tnbits\tcomputable\trequired_tensor\tmissing_reason\n")
                for entry in report["tensors"]:
                    required = entry.get("required_activation") or {}
                    f.write(
                        f"{entry['layer']}\t{entry['tensor']}\t{entry['orig_shape']}\t"
                        f"{entry['group_size']}\t{entry['axis']}\t{entry['nbits']}\t"
                        f"{int(bool(entry['offline_decision_summary_computable']))}\t"
                        f"{required.get('tensor')}\t{entry.get('not_computable_reason')}\n"
                    )
        print(json.dumps({
            "output": str(output),
            "tensor_count": report["summary"]["tensor_count"],
            "layer_count": report["summary"]["layer_count"],
            "layer0_computable": report["summary"]["layer0_computable"],
            "layer0_not_computable": report["summary"]["layer0_not_computable"],
        }, sort_keys=True))
        return 0

    if args.tensor == "source-map-audit":
        report = {
            "format": "krasis_hqq_source_map_audit",
            **build_source_map_audit(model_dir, cache_dir),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        if args.tsv_output:
            tsv = Path(args.tsv_output)
            tsv.parent.mkdir(parents=True, exist_ok=True)
            with tsv.open("w", encoding="utf-8") as f:
                f.write(
                    "layer\ttensor\tlayer_type\thqq_shape\thqq_dtype\tsource_status\t"
                    "recipe\tsource_shape\tsource_dtypes\tresolver_status\tshape_ok\t"
                    "dtype_ok\tshape_dtype_mapping_ready\tstrict_resolver_kind\t"
                    "source_contract_ready\tmetadata_missing\tsource_tensors\t"
                    "reconstructed_source_sha256\tartifact_sha256\n"
                )
                for entry in report["targets"]:
                    selected = entry.get("selected_mapping") or {}
                    provenance = entry.get("source_provenance") or {}
                    contract = entry.get("source_contract") or {}
                    missing = [
                        name
                        for name, present in (
                            ("source_tensor", provenance.get("has_source_tensor")),
                            ("source_hash", provenance.get("has_source_hash")),
                            ("artifact_hash", provenance.get("has_artifact_hash")),
                        )
                        if not present
                    ]
                    source_tensors = ",".join(
                        item.get("name", "")
                        for item in selected.get("source_tensors", [])
                    )
                    f.write(
                        f"{entry['layer']}\t{entry['tensor']}\t{entry.get('layer_type')}\t"
                        f"{entry['hqq']['shape']}\t{entry['hqq'].get('dtype')}\t"
                        f"{entry['source_mapping']['status']}\t{selected.get('recipe')}\t"
                        f"{selected.get('reconstructed_shape')}\t{selected.get('source_dtypes')}\t"
                        f"{selected.get('current_sidecar_resolver_status')}\t"
                        f"{int(bool(entry['shape_compatible']))}\t{int(bool(entry['source_dtype_ok']))}\t"
                        f"{int(bool(entry['shape_dtype_mapping_ready']))}\t"
                        f"{contract.get('resolver_kind')}\t{int(bool(contract.get('ready')))}\t"
                        f"{','.join(missing)}\t{source_tensors}\t"
                        f"{contract.get('reconstructed_source_sha256')}\t"
                        f"{contract.get('artifact_sha256')}\n"
                    )
        print(json.dumps({
            "output": str(output),
            "target_count": report["summary"]["target_count"],
            "resolved_source_targets": report["summary"]["resolved_source_targets"],
            "shape_dtype_mapping_ready_targets": report["summary"]["shape_dtype_mapping_ready_targets"],
            "current_sidecar_resolver_ready_targets": report["summary"]["current_sidecar_resolver_ready_targets"],
            "strict_resolver_ready_targets": report["summary"]["strict_resolver_ready_targets"],
            "source_hash_ready_targets": report["summary"]["source_hash_ready_targets"],
            "artifact_hash_ready_targets": report["summary"]["artifact_hash_ready_targets"],
            "source_contract_ready_targets": report["summary"]["source_contract_ready_targets"],
            "sidecar_generation_ready_targets": report["summary"]["sidecar_generation_ready_targets"],
            "metadata_gap_targets": report["summary"]["metadata_gap_targets"],
            "recipe_counts": report["summary"]["recipe_counts"],
        }, sort_keys=True))
        return 0

    if args.tensor == "capture-plan":
        plan = build_capture_requirements(
            cache_dir,
            input_row_traces=[Path(item).resolve() for item in (args.input_row_trace or [])],
            case_summary_path=Path(args.case_summary).resolve() if args.case_summary else None,
        )
        report = {
            "format": "krasis_hqq_activation_capture_requirements",
            "model_dir": str(model_dir),
            **plan,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        if args.tsv_output:
            tsv = Path(args.tsv_output)
            tsv.parent.mkdir(parents=True, exist_ok=True)
            with tsv.open("w", encoding="utf-8") as f:
                f.write(
                    "layer\ttensor\tselection_reason\tshape\tgroup_size\taxis\tnbits\t"
                    "required_tensor\trequired_width\tproducer_site\texisting_artifact\tstatus\n"
                )
                for target in report["targets"]:
                    required = target["required_activation"]
                    site = required["producer_capture_site"] or {}
                    f.write(
                        f"{target['layer']}\t{target['tensor']}\t{target['selection_reason']}\t"
                        f"{target['hqq']['shape']}\t{target['hqq']['group_size']}\t"
                        f"{target['hqq']['axis']}\t{target['hqq']['nbits']}\t"
                        f"{required['tensor']}\t{required['width']}\t"
                        f"{site.get('file')}::{site.get('function')} {site.get('current_reference')}\t"
                        f"{int(bool(target['existing_artifact']['contains_required_row']))}\t"
                        f"{target['offline_status']}\n"
                    )
                f.write("\n# missing_activation_rows\n")
                f.write("layer\trequired_tensor\twidth\tfield\tproducer_site\ttarget_count\ttargets\n")
                for missing in report["missing_activation_rows"]:
                    site = missing["producer_capture_site"] or {}
                    target_names = ",".join(
                        f"L{item['layer']}:{item['tensor']}" for item in missing["targets"]
                    )
                    f.write(
                        f"{missing['layer']}\t{missing['tensor']}\t{missing['width']}\t"
                        f"{missing['field']}\t"
                        f"{site.get('file')}::{site.get('function')} {site.get('current_reference')}\t"
                        f"{len(missing['targets'])}\t{target_names}\n"
                    )
                f.write("\n# minimal_prompt_case_position_set\n")
                f.write("conv_idx\tturn\tcase_key\trow_count\tpositions\tprompt_prefix\n")
                for case in report["minimal_prompt_case_position_set"]["cases"]:
                    f.write(
                        f"{case['conv_idx']}\t{case['turn']}\t{case['case_key']}\t"
                        f"{case['row_count']}\t{','.join(str(v) for v in case['positions'])}\t"
                        f"{case['prompt_prefix'].replace(chr(9), ' ')}\n"
                    )
        print(json.dumps({
            "output": str(output),
            "target_count": report["summary"]["target_count"],
            "missing_activation_row_kinds": report["summary"]["missing_activation_row_kinds"],
            "not_computable_targets": report["summary"]["not_computable_targets"],
            "positions": report["minimal_prompt_case_position_set"]["positions_selector"],
        }, sort_keys=True))
        return 0

    if args.tensor == "case-row-audit":
        if not args.input_row_trace:
            raise RuntimeError("case-row-audit requires --input-row-trace")
        if not args.case_summary:
            raise RuntimeError("case-row-audit requires --case-summary")
        report = build_case_row_audit(
            model_dir=model_dir,
            input_row_traces=[Path(item).resolve() for item in args.input_row_trace],
            case_summary_path=Path(args.case_summary).resolve(),
            positions=parse_input_row_positions(args.input_row_pos),
            layer=args.layer,
            tensor=args.input_row_tensor,
            output_path=Path(args.output),
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        if args.tsv_output:
            tsv = Path(args.tsv_output)
            tsv.parent.mkdir(parents=True, exist_ok=True)
            with tsv.open("w", encoding="utf-8") as f:
                f.write(
                    "model_id\tsplit\tcase_key\tconv_idx\tturn\tposition\tlayer\t"
                    "tensor\trow_hash\trow_width\trow_group_count\ttrace_line\tprompt_prefix\n"
                )
                for row in report["case_rows"]:
                    prefix = str(row.get("prompt_prefix") or "").replace("\t", " ").replace("\n", " ")
                    f.write(
                        f"{row['model_id']}\t{row['split']}\t{row['case_key']}\t"
                        f"{row['conv_idx']}\t{row['turn']}\t{row['position']}\t"
                        f"{row['layer']}\t{row['tensor']}\t{row['row_hash']}\t"
                        f"{row['row_width']}\t{row['row_group_count']}\t{row['trace_line']}\t"
                        f"{prefix}\n"
                    )
        print(json.dumps({
            "output": str(output),
            "mapped_case_rows": report["summary"]["mapped_case_rows"],
            "train_rows": report["summary"]["train_rows"],
            "heldout_rows": report["summary"]["heldout_rows"],
            "missing_case_rows": report["summary"]["missing_case_rows"],
            "unused_matching_events": report["summary"]["unused_matching_events"],
        }, sort_keys=True))
        return 0

    if args.tensor == "named-row-divergence":
        if not args.bf16_trace or not args.hqq_trace:
            raise RuntimeError("--tensor named-row-divergence requires --bf16-trace and --hqq-trace")
        divergence = build_named_row_divergence_map(
            bf16_trace=Path(args.bf16_trace).resolve(),
            hqq_trace=Path(args.hqq_trace).resolve(),
            bf16_summary=Path(args.bf16_summary).resolve() if args.bf16_summary else None,
            hqq_summary=Path(args.hqq_summary).resolve() if args.hqq_summary else None,
        )
        report = {
            "format": "krasis_hqq_named_row_divergence_map",
            "model_dir": str(model_dir),
            **divergence,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        if args.tsv_output:
            tsv = Path(args.tsv_output)
            tsv.parent.mkdir(parents=True, exist_ok=True)
            with tsv.open("w", encoding="utf-8") as f:
                f.write("# field_layer_matrix\n")
                f.write(
                    "layer\ttensor\tstage\tpairs\tmatch\tmismatch\twidths\t"
                    "rms_mean\trms_max\tlinf_max\tcosine_min\tfirst_mismatch_pos\tfirst_mismatch_rms\n"
                )
                for entry in report["field_layer_matrix"]:
                    first = entry.get("first_mismatch") or {}
                    f.write(
                        f"{entry['layer']}\t{entry['tensor']}\t{entry['stage']}\t"
                        f"{entry['pairs']}\t{entry['match_count']}\t{entry['mismatch_count']}\t"
                        f"{','.join(str(v) for v in entry['widths'])}\t"
                        f"{entry['rms']['mean'] if entry['rms']['mean'] is not None else ''}\t"
                        f"{entry['rms']['max'] if entry['rms']['max'] is not None else ''}\t"
                        f"{entry['linf']['max'] if entry['linf']['max'] is not None else ''}\t"
                        f"{entry['cosine']['min'] if entry['cosine']['min'] is not None else ''}\t"
                        f"{first.get('pos', '')}\t{first.get('rms', '')}\n"
                    )
                f.write("\n# field_layer_position_matrix\n")
                f.write(
                    "layer\ttensor\tpos\tpairs\tmatch\tmismatch\twidths\t"
                    "rms_mean\trms_max\tlinf_max\tcosine_min\tcase_candidates\n"
                )
                for entry in report["field_layer_position_matrix"]:
                    cases = ",".join(str(case.get("case_key")) for case in entry.get("case_candidates", []))
                    f.write(
                        f"{entry['layer']}\t{entry['tensor']}\t{entry['pos']}\t"
                        f"{entry['pairs']}\t{entry['match_count']}\t{entry['mismatch_count']}\t"
                        f"{','.join(str(v) for v in entry['widths'])}\t"
                        f"{entry['rms']['mean'] if entry['rms']['mean'] is not None else ''}\t"
                        f"{entry['rms']['max'] if entry['rms']['max'] is not None else ''}\t"
                        f"{entry['linf']['max'] if entry['linf']['max'] is not None else ''}\t"
                        f"{entry['cosine']['min'] if entry['cosine']['min'] is not None else ''}\t"
                        f"{cases}\n"
                    )
                f.write("\n# stage_summary\n")
                f.write("stage\tpairs\tmatch\tmismatch\trms_mean\trms_max\tlinf_max\tcosine_min\n")
                for entry in report["divergence_attribution"]["stage_summary"]:
                    f.write(
                        f"{entry['stage']}\t{entry['pairs']}\t{entry['match_count']}\t"
                        f"{entry['mismatch_count']}\t"
                        f"{entry['rms']['mean'] if entry['rms']['mean'] is not None else ''}\t"
                        f"{entry['rms']['max'] if entry['rms']['max'] is not None else ''}\t"
                        f"{entry['linf']['max'] if entry['linf']['max'] is not None else ''}\t"
                        f"{entry['cosine']['min'] if entry['cosine']['min'] is not None else ''}\n"
                    )
        print(json.dumps({
            "output": str(output),
            "paired_events": report["summary"]["paired_events"],
            "match_count": report["summary"]["match_count"],
            "mismatch_count": report["summary"]["mismatch_count"],
            "first_mismatch": report["first_mismatch"],
        }, sort_keys=True))
        return 0

    supported_tensors = (
        "in_proj_qkvz",
        "in_proj_ba",
        "out_proj",
        "fused_qkv",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    if args.tensor not in supported_tensors:
        raise RuntimeError(
            "Only --tensor in_proj_qkvz, in_proj_ba, out_proj, fused_qkv, q_proj, "
            "k_proj, v_proj, o_proj, cache-audit, source-map-audit, or capture-plan is supported"
        )

    hqq_path = cache_dir / f"layer_{args.layer:03d}_{args.tensor}_hqq4.safetensors"
    if not hqq_path.is_file():
        raise RuntimeError(f"Missing HQQ artifact: {hqq_path}")

    timer.start("load_source")
    source, raw_concat_source, source_meta = build_qwen35_source(model_dir, args.layer, args.tensor)
    timer.stop("load_source")
    timer.start("load_hqq_dequant")
    hqq_tensors, hqq_file_metadata = load_safetensors(hqq_path)
    deq, hqq_meta, hqq_quant = dequant_hqq4(hqq_tensors)
    timer.stop("load_hqq_dequant")
    if tuple(source.shape) != tuple(deq.shape):
        raise RuntimeError(f"Source/HQQ shape mismatch: source={tuple(source.shape)} hqq={tuple(deq.shape)}")
    timer.start("precompute_delta")
    diff_cache = (deq.to(torch.float32).contiguous() - source.to(torch.float32).contiguous()).contiguous()
    timer.stop("precompute_delta")

    ranges = source_slice_ranges(source_meta)
    slices = {}
    for name, row_ranges in ranges.items():
        src_slice = gather_ranges(source, row_ranges)
        deq_slice = gather_ranges(deq, row_ranges)
        sample_row_ids = [0, src_slice.shape[0] // 2, src_slice.shape[0] - 1]
        slices[name] = {
            **metrics(src_slice, deq_slice),
            **{
                key: value
                for key, value in error_metrics_from_diff(src_slice, deq_slice - src_slice).items()
                if key in ("error_sq_sum", "source_sq_sum")
            },
            "row_ranges": [[int(a), int(b)] for a, b in row_ranges],
            "source_samples": sample_rows(src_slice, sample_row_ids, args.sample_cols),
            "dequant_samples": sample_rows(deq_slice, sample_row_ids, args.sample_cols),
        }

    if args.input_row_trace:
        timer.start("row_parsing")
        input_row_events = load_input_rows_from_trace(
            [Path(item).resolve() for item in args.input_row_trace],
            positions=parse_input_row_positions(args.input_row_pos),
            layer=args.layer,
            tensor=args.input_row_tensor,
        )
        case_map = load_case_map(Path(args.case_summary).resolve()) if args.case_summary else []
        timer.stop("row_parsing")
        row_records = []
        aligned_rows = align_input_rows_to_cases(input_row_events, case_map)
        if args.max_input_rows is not None:
            aligned_rows = aligned_rows[:max(0, int(args.max_input_rows))]
        for input_row_event, case in aligned_rows:
            stage_start = time.perf_counter()
            sensitivity = projection_sensitivity(
                source,
                deq,
                input_row_event["values"],
                source_meta,
                hqq_meta,
                hqq_quant,
                top_rows=args.top_rows,
                top_groups=args.top_groups,
                sample_cols=args.sample_cols,
                focus_row=args.focus_row,
                focus_group=args.focus_group,
                diff_cache=diff_cache,
                timer=timer,
            )
            timer.add("projection_sensitivity.total", time.perf_counter() - stage_start)
            trace_meta = {
                key: value
                for key, value in input_row_event.items()
                if key != "values"
            }
            sensitivity["input_row_trace"] = trace_meta
            if case:
                sensitivity["case"] = case
            row_records.append({
                "case": case,
                "input_row_trace": trace_meta,
                "sensitivity": sensitivity,
            })
        output_sensitivity = row_records[0]["sensitivity"]
        if len(row_records) > 1:
            output_sensitivity["multi_input_rows"] = aggregate_projection_sensitivities(row_records)
            output_sensitivity["multi_input_rows"]["max_input_rows_applied"] = args.max_input_rows
            output_sensitivity["decision_quality_diagnostic"] = build_decision_quality_diagnostic(row_records)
            output_sensitivity["rows"] = [
                {
                    "case": row["case"],
                    "input_row_trace": row["input_row_trace"],
                    "overall": row["sensitivity"]["overall"],
                    "dominant_slice_by_output_error_energy": row["sensitivity"][
                        "dominant_slice_by_output_error_energy"
                    ],
                    "quantizer_choice": {
                        "aggregate_by_candidate": row["sensitivity"]
                        .get("quantizer_choice", {})
                        .get("aggregate_by_candidate", {}),
                        "best_aggregate_by_abs_signed_contribution_total": row["sensitivity"]
                        .get("quantizer_choice", {})
                        .get("best_aggregate_by_abs_signed_contribution_total"),
                    },
                }
                for row in row_records
            ]
    else:
        output_sensitivity = {
            "computed": False,
            "reason": (
                "Offline output sensitivity requires --input-row-trace with an input_row_full event "
                "for the target prefill position."
            ),
            "missing_field": "full normalized layer-0 post_input_norm/la_input_row_for_qkvz vector for pos=36",
            "no_runtime_probe_added": False,
        }

    report = {
        "format": "krasis_hqq_attention_diff",
        "model_dir": str(model_dir),
        "cache_dir": str(cache_dir),
        "hqq_artifact": str(hqq_path),
        "layer": args.layer,
        "tensor": args.tensor,
        "source": source_meta,
        "hqq": {
            **hqq_meta,
            "artifact_metadata": hqq_file_metadata,
            "tensor_dtypes": {name: str(tensor.dtype).replace("torch.", "") for name, tensor in hqq_tensors.items()},
        },
        "layout_checks": {
            "expected_interleaved_vs_hqq": metrics(source, deq),
            "raw_concat_vs_hqq": metrics(raw_concat_source, deq),
        },
        "slices": slices,
        "slice_dominance": slice_dominance(slices),
        "row_attribution": row_attribution(
            source,
            deq,
            source_meta,
            top_n=args.top_rows,
            sample_cols=args.sample_cols,
        ),
        "hqq_group_attribution": group_attribution(
            source,
            deq,
            source_meta,
            hqq_meta,
            top_n=args.top_groups,
            sample_cols=args.sample_cols,
        ),
        "candidate_search": {
            "grid_steps": BEST_FIT_GRID_STEPS,
            "local_grid_steps": BEST_FIT_LOCAL_GRID_STEPS,
        },
        "timings": {
            "seconds": timer.snapshot(),
            "input_rows_available": len(input_row_events) if args.input_row_trace else 0,
            "input_rows_analyzed": len(row_records) if args.input_row_trace else 0,
        },
        "output_sensitivity": output_sensitivity,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    if args.tsv_output:
        tsv = Path(args.tsv_output)
        tsv.parent.mkdir(parents=True, exist_ok=True)
        with tsv.open("w", encoding="utf-8") as f:
            f.write("slice\tmax_abs\tmean_abs\trms\trelative_rms\tsource_rms\tsource_sha256\tactual_sha256\n")
            for name in source_slice_names(source_meta):
                m = slices[name]
                f.write(
                    f"{name}\t{m['max_abs']:.9f}\t{m['mean_abs']:.9f}\t"
                    f"{m['rms']:.9f}\t{m['relative_rms']:.9f}\t{m['source_rms']:.9f}\t"
                    f"{m['source_sha256']}\t{m['actual_sha256']}\n"
                )
            f.write("\n# top_rows_by_rms\n")
            f.write("rank\tglobal_row\tslice\tkey_head\tslice_local_row\trms\trelative_rms\tmax_abs\tmax_abs_col\n")
            for rank, row in enumerate(report["row_attribution"]["top_by_rms"], 1):
                f.write(
                    f"{rank}\t{row['global_row']}\t{row['slice']}\t{row['key_head']}\t"
                    f"{row['slice_local_row']}\t{row['rms']:.9f}\t{row['relative_rms']:.9f}\t"
                    f"{row['max_abs']:.9f}\t{row['max_abs_col']}\n"
                )
            f.write("\n# top_hqq_groups_by_rms\n")
            f.write("rank\tglobal_row\tslice\tkey_head\thqq_group\tcol_start\tcol_end\trms\trelative_rms\tmax_abs\n")
            for rank, group in enumerate(report["hqq_group_attribution"]["top_by_rms"], 1):
                f.write(
                    f"{rank}\t{group['global_row']}\t{group['slice']}\t{group['key_head']}\t"
                    f"{group['hqq_group_index']}\t{group['col_start']}\t{group['col_end']}\t"
                    f"{group['rms']:.9f}\t{group['relative_rms']:.9f}\t{group['max_abs']:.9f}\n"
                )
            if report["output_sensitivity"]["computed"]:
                sensitivity = report["output_sensitivity"]
                f.write("\n# output_sensitivity_by_slice\n")
                f.write("slice\trows\tmax_abs\tmean_abs\trms\terror_sq_fraction\n")
                for name in source_slice_names(source_meta):
                    metric = sensitivity["slices"][name]
                    f.write(
                        f"{name}\t{metric['rows']}\t{metric['max_abs']:.9f}\t"
                        f"{metric['mean_abs']:.9f}\t{metric['rms']:.9f}\t"
                        f"{metric['error_sq_fraction']:.9f}\n"
                    )
                f.write("\n# top_rows_by_abs_output_error\n")
                f.write("rank\tglobal_row\tslice\tkey_head\tslice_local_row\tabs_output_error\toutput_error\tbf16_dot\thqq_dot\n")
                for rank, row in enumerate(sensitivity["top_rows_by_abs_output_error"], 1):
                    f.write(
                        f"{rank}\t{row['global_row']}\t{row['slice']}\t{row['key_head']}\t"
                        f"{row['slice_local_row']}\t{row['abs_output_error']:.9f}\t"
                        f"{row['output_error']:.9f}\t{row['bf16_dot']:.9f}\t{row['hqq_dot']:.9f}\n"
                    )
                f.write("\n# top_hqq_groups_by_abs_output_contribution\n")
                f.write("rank\tglobal_row\tslice\tkey_head\thqq_group\tcol_start\tcol_end\tabs_contribution\tcontribution\n")
                for rank, group in enumerate(sensitivity["top_hqq_groups_by_abs_contribution"], 1):
                    diag = group["diagnostics"]
                    f.write(
                        f"{rank}\t{group['global_row']}\t{group['slice']}\t{group['key_head']}\t"
                        f"{group['hqq_group_index']}\t{group['col_start']}\t{group['col_end']}\t"
                        f"{group['abs_contribution']:.9f}\t{group['contribution']:.9f}\n"
                    )
                f.write("\n# top_hqq_group_diagnostics\n")
                f.write(
                    "rank\tglobal_row\tslice\tkey_head\thqq_group\tinput_rms\tinput_max_abs\t"
                    "weight_error_rms\tweight_error_max_abs\tcontribution_rms\tcontribution_max_abs\t"
                    "signed_sum\talignment_cosine\tscale\tzero\tqvalue_min\tqvalue_max\tqzero\tqmax\n"
                )
                for rank, group in enumerate(sensitivity["top_hqq_groups_by_abs_contribution"], 1):
                    diag = group["diagnostics"]
                    f.write(
                        f"{rank}\t{group['global_row']}\t{group['slice']}\t{group['key_head']}\t"
                        f"{group['hqq_group_index']}\t{diag['input_rms']:.9f}\t{diag['input_max_abs']:.9f}\t"
                        f"{diag['weight_error_rms']:.9f}\t{diag['weight_error_max_abs']:.9f}\t"
                        f"{diag['contribution_rms']:.9f}\t{diag['contribution_max_abs']:.9f}\t"
                        f"{diag['signed_sum']:.9f}\t{diag['alignment_cosine']:.9f}\t"
                        f"{diag['scale']:.9f}\t{diag['zero']:.9f}\t{diag['qvalue_min']}\t"
                        f"{diag['qvalue_max']}\t{diag['qvalue_saturation_zero_count']}\t"
                        f"{diag['qvalue_saturation_max_count']}\n"
                    )
                focus = sensitivity.get("focus_group_column_contributions")
                if focus:
                    f.write("\n# focus_group_columns_by_abs_contribution\n")
                    f.write(
                        "rank\tglobal_row\tslice\tkey_head\thqq_group\tcol\tlocal_col\tinput\t"
                        "bf16_weight\thqq_dequant_weight\tweight_error\tcontribution\tabs_contribution\t"
                        "qvalue\traw_q\tclipped_low\tclipped_high\n"
                    )
                    for rank, col in enumerate(focus["columns_ranked_by_abs_contribution"], 1):
                        f.write(
                            f"{rank}\t{focus['global_row']}\t{focus['slice']}\t{focus['key_head']}\t"
                            f"{focus['hqq_group_index']}\t{col['col']}\t{col['group_local_col']}\t"
                            f"{col['input']:.9f}\t{col['bf16_weight']:.9f}\t"
                            f"{col['hqq_dequant_weight']:.9f}\t{col['weight_error']:.9f}\t"
                            f"{col['contribution']:.9f}\t{col['abs_contribution']:.9f}\t{col['qvalue']}\t"
                            f"{col['raw_q_before_round_clamp']:.9f}\t"
                            f"{int(col['is_clipped_low'])}\t{int(col['is_clipped_high'])}\n"
                        )
                quantizer = sensitivity.get("quantizer_choice")
                if quantizer and quantizer.get("computed"):
                    f.write("\n# quantizer_choice_top_groups\n")
                    f.write(
                        "rank\tglobal_row\tslice\tkey_head\thqq_group\tcol_start\tcol_end\t"
                        "source_min\tsource_max\tcached_scale\tcached_zero\tq_mismatch\tclip_low\tclip_high\t"
                        "cached_signed\tminmax_signed\tbestfit_signed\tweighted_signed\tclipfree_signed\t"
                        "cached_weight_rms\tminmax_weight_rms\tbestfit_weight_rms\tweighted_weight_rms\t"
                        "clipfree_weight_rms\tbest_by_contribution\tbest_by_weight_rms\tprimary_cause\n"
                    )
                    for rank, group in enumerate(quantizer["groups"], 1):
                        candidates = {item["name"]: item for item in group["candidates"]}
                        cached = candidates["cached_hqq"]
                        minmax = candidates["minmax_affine"]
                        bestfit = candidates["offline_best_fit_affine"]
                        weighted = candidates["activation_weighted_affine"]
                        clipfree = candidates["activation_weighted_clip_free"]
                        f.write(
                            f"{rank}\t{group['global_row']}\t{group['slice']}\t{group['key_head']}\t"
                            f"{group['hqq_group_index']}\t{group['col_start']}\t{group['col_end']}\t"
                            f"{group['source_min']:.9f}\t{group['source_max']:.9f}\t"
                            f"{group['cached_scale']:.9f}\t{group['cached_zero']:.9f}\t"
                            f"{group['cached_qvalue_reconstruction']['mismatch_count']}\t"
                            f"{group['cached_clip_low_count']}\t{group['cached_clip_high_count']}\t"
                            f"{cached['projection_contribution_signed_sum']:.9f}\t"
                            f"{minmax['projection_contribution_signed_sum']:.9f}\t"
                            f"{bestfit['projection_contribution_signed_sum']:.9f}\t"
                            f"{weighted['projection_contribution_signed_sum']:.9f}\t"
                            f"{clipfree['projection_contribution_signed_sum']:.9f}\t"
                            f"{cached['weight_error_rms']:.9f}\t{minmax['weight_error_rms']:.9f}\t"
                            f"{bestfit['weight_error_rms']:.9f}\t{weighted['weight_error_rms']:.9f}\t"
                            f"{clipfree['weight_error_rms']:.9f}\t"
                            f"{group['best_candidate_by_abs_signed_output_contribution']}\t"
                            f"{group['best_candidate_by_weight_rms']}\t{group['ranked_primary_cause']}\n"
                        )
                    f.write("\n# quantizer_choice_aggregate_by_candidate\n")
                    f.write(
                        "candidate\tgroups\tsigned_sum\tabs_signed_sum\tsigned_rms\tmax_abs_signed\t"
                        "mean_weight_rms\tmean_contribution_rms\tcontribution_max_abs\tclip_low\tclip_high\t"
                        "qzero\tqmax\n"
                    )
                    for name, aggregate in quantizer["aggregate_by_candidate"].items():
                        f.write(
                            f"{name}\t{aggregate['groups']}\t"
                            f"{aggregate['signed_contribution_sum']:.9f}\t"
                            f"{aggregate['abs_signed_contribution_sum']:.9f}\t"
                            f"{aggregate['signed_contribution_rms_across_groups']:.9f}\t"
                            f"{aggregate['max_abs_signed_contribution']:.9f}\t"
                            f"{aggregate['mean_weight_error_rms']:.9f}\t"
                            f"{aggregate['mean_projection_contribution_rms']:.9f}\t"
	                            f"{aggregate['projection_contribution_max_abs']:.9f}\t"
	                            f"{aggregate['clip_low_count']}\t{aggregate['clip_high_count']}\t"
	                            f"{aggregate['qvalue_saturation_zero_count']}\t"
	                            f"{aggregate['qvalue_saturation_max_count']}\n"
	                        )
                    multi = sensitivity.get("multi_input_rows")
                    if multi and multi.get("computed"):
                        f.write("\n# multi_input_row_candidate_aggregate\n")
                        f.write(
                            "candidate\tgroups\tsigned_sum\tabs_signed_sum\tsigned_rms\t"
                            "max_abs_signed\tmean_weight_rms\tclip_low\tclip_high\tqzero\tqmax\n"
                        )
                        for name, aggregate in multi["candidate_aggregate"].items():
                            f.write(
                                f"{name}\t{aggregate['groups']}\t"
                                f"{aggregate['signed_contribution_sum']:.9f}\t"
                                f"{aggregate['abs_signed_contribution_sum']:.9f}\t"
                                f"{aggregate['signed_contribution_rms_across_groups']:.9f}\t"
                                f"{aggregate['max_abs_signed_contribution']:.9f}\t"
                                f"{aggregate['mean_weight_error_rms']:.9f}\t"
                                f"{aggregate['clip_low_count']}\t{aggregate['clip_high_count']}\t"
                                f"{aggregate['qvalue_saturation_zero_count']}\t"
                                f"{aggregate['qvalue_saturation_max_count']}\n"
                            )
                        f.write("\n# multi_input_row_slices\n")
                        f.write("slice\trows\trms\tmax_abs\terror_sq_fraction\tmean_row_rms\n")
                        for name in multi["slices"].keys():
                            metric = multi["slices"][name]
                            f.write(
                                f"{name}\t{metric['rows']}\t{metric['rms']:.9f}\t"
                                f"{metric['max_abs']:.9f}\t{metric['error_sq_fraction']:.9f}\t"
                                f"{metric['mean_row_rms']:.9f}\n"
                            )
                    decision = sensitivity.get("decision_quality_diagnostic")
                    if decision and decision.get("computed"):
                        f.write("\n# decision_quality_by_slice\n")
                        f.write(
                            "slice\tcandidate\tgroups\tsigned_sum\tabs_signed_sum\tsigned_rms\t"
                            "max_abs_signed\tmean_weight_rms\tweight_rms_delta_vs_cached\tclip_low\tclip_high\tqzero\tqmax\n"
                        )
                        for slice_name in decision["by_slice"].keys():
                            candidates = decision["by_slice"][slice_name]["aggregate_by_candidate"]
                            for candidate_name, aggregate in candidates.items():
                                f.write(
                                    f"{slice_name}\t{candidate_name}\t{aggregate['groups']}\t"
                                    f"{aggregate['signed_contribution_sum']:.9f}\t"
                                    f"{aggregate['abs_signed_contribution_sum']:.9f}\t"
                                    f"{aggregate['signed_contribution_rms_across_groups']:.9f}\t"
                                    f"{aggregate['max_abs_signed_contribution']:.9f}\t"
                                    f"{aggregate['mean_weight_error_rms']:.9f}\t"
                                    f"{aggregate.get('mean_weight_error_rms_delta_vs_cached', 0.0):.9f}\t"
                                    f"{aggregate['clip_low_count']}\t{aggregate['clip_high_count']}\t"
                                    f"{aggregate['qvalue_saturation_zero_count']}\t"
                                    f"{aggregate['qvalue_saturation_max_count']}\n"
                                )
                        f.write("\n# decision_quality_by_case\n")
                        f.write(
                            "case_key\tpositions\tcandidate\tgroups\tsigned_sum\tabs_signed_sum\t"
                            "signed_rms\tmax_abs_signed\tmean_weight_rms\tweight_rms_delta_vs_cached\t"
                            "clip_low\tclip_high\tqzero\tqmax\tprompt_prefix\n"
                        )
                        for case in decision["by_case"]:
                            positions = ",".join(str(pos) for pos in case["positions"])
                            for candidate_name, aggregate in case["candidate_summary"]["aggregate_by_candidate"].items():
                                f.write(
                                    f"{case['case_key']}\t{positions}\t{candidate_name}\t{aggregate['groups']}\t"
                                    f"{aggregate['signed_contribution_sum']:.9f}\t"
                                    f"{aggregate['abs_signed_contribution_sum']:.9f}\t"
                                    f"{aggregate['signed_contribution_rms_across_groups']:.9f}\t"
                                    f"{aggregate['max_abs_signed_contribution']:.9f}\t"
                                    f"{aggregate['mean_weight_error_rms']:.9f}\t"
                                    f"{aggregate.get('mean_weight_error_rms_delta_vs_cached', 0.0):.9f}\t"
                                    f"{aggregate['clip_low_count']}\t{aggregate['clip_high_count']}\t"
                                    f"{aggregate['qvalue_saturation_zero_count']}\t"
                                    f"{aggregate['qvalue_saturation_max_count']}\t{case['prompt_prefix']}\n"
                                )
                        f.write("\n# decision_quality_regressions\n")
                        f.write("scope\tcandidate\tcached_abs_signed_sum\tcandidate_abs_signed_sum\tratio\n")
                        for item in decision["activation_candidate_regressions"]:
                            scope = item.get("case_key") or item.get("slice")
                            f.write(
                                f"{scope}\t{item['candidate']}\t{item['cached_abs_signed_sum']:.9f}\t"
                                f"{item['candidate_abs_signed_sum']:.9f}\t{item['ratio']:.9f}\n"
                            )
                    f.write("\n# quantizer_choice_focus_columns\n")
                    f.write(
                        "rank\tglobal_row\tslice\tkey_head\thqq_group\tcol\tlocal_col\tinput\tsource\t"
                        "cached_dequant\tweight_error\tcontribution\tcached_q\texpected_q\traw_q\t"
                        "clipped_low\tclipped_high\n"
                    )
                    for group in quantizer["groups"]:
                        if focus and group["global_row"] == focus["global_row"] and group["hqq_group_index"] == focus["hqq_group_index"]:
                            for rank, col in enumerate(group["worst_columns_by_abs_contribution"], 1):
                                f.write(
                                    f"{rank}\t{group['global_row']}\t{group['slice']}\t{group['key_head']}\t"
                                    f"{group['hqq_group_index']}\t{col['col']}\t{col['group_local_col']}\t"
                                    f"{col['input']:.9f}\t{col['source']:.9f}\t{col['cached_dequant']:.9f}\t"
                                    f"{col['weight_error']:.9f}\t{col['contribution']:.9f}\t"
                                    f"{col['cached_qvalue']}\t{col['expected_qvalue_from_cached_scale_zero']}\t"
                                    f"{col['raw_q_before_round_clamp']:.9f}\t"
                                    f"{int(col['is_clipped_low'])}\t{int(col['is_clipped_high'])}\n"
                                )
                            break

    print(json.dumps({
        "output": str(output),
        "hqq_artifact": str(hqq_path),
        "overall_relative_rms": report["layout_checks"]["expected_interleaved_vs_hqq"]["relative_rms"],
        "raw_concat_relative_rms": report["layout_checks"]["raw_concat_vs_hqq"]["relative_rms"],
        "slice_relative_rms": {name: slices[name]["relative_rms"] for name in source_slice_names(source_meta)},
        "dominant_slice_by_error_energy": report["slice_dominance"]["dominant_by_error_energy"],
        "dominant_slice_by_relative_rms": report["slice_dominance"]["dominant_by_relative_rms"],
        "output_sensitivity_computed": report["output_sensitivity"]["computed"],
        "dominant_slice_by_output_error_energy": report["output_sensitivity"].get("dominant_slice_by_output_error_energy"),
        "quantizer_choice_computed": report["output_sensitivity"].get("quantizer_choice", {}).get("computed"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
