"""HQQ self-calibration evidence and offline evaluation command.

This module captures and evaluates calibration evidence only. It must not write
HQQ tensor artifacts, qvalues, scales, zeros, or a loadable calibrated manifest.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from krasis.attention_backend import (
    HQQ_DEFAULT_GROUP_SIZE,
    hqq_attention_cache_dir,
    hqq_attention_manifest_path,
    load_hqq_attention_manifest,
    require_complete_hqq_attention_manifest,
)
from krasis.config import (
    HQQ_CACHE_PROFILE_BASELINE,
    HQQ_CACHE_PROFILE_CHOICES,
    HQQ_CACHE_PROFILE_SELFCAL_V1,
    cache_dir_for_model,
)


EVIDENCE_FORMAT = "krasis_hqq_selfcal_evidence"
EVIDENCE_FORMAT_VERSION = 1
CANDIDATES_FORMAT = "krasis_hqq_selfcal_sensitivity_candidates"
CANDIDATES_FORMAT_VERSION = 1
REFIT_EVAL_FORMAT = "krasis_hqq_selfcal_refit_evaluation"
REFIT_EVAL_FORMAT_VERSION = 1
EXPERIMENTAL_CACHE_WRITE_FORMAT = "krasis_hqq_selfcal_experimental_cache_write"
EXPERIMENTAL_CACHE_WRITE_FORMAT_VERSION = 1
ABLATION_REPORT_FORMAT = "krasis_hqq_selfcal_ablation_report"
ABLATION_REPORT_FORMAT_VERSION = 1
SIDECAR_SIM_FORMAT = "krasis_hqq_selfcal_sidecar_simulation"
SIDECAR_SIM_FORMAT_VERSION = 1
SIDECAR_MANIFEST_FORMAT = "krasis_hqq_selfcal_sidecar_manifest"
SIDECAR_MANIFEST_FORMAT_VERSION = 1
SIDECAR_WRITE_FORMAT = "krasis_hqq_selfcal_sidecar_write"
SIDECAR_WRITE_FORMAT_VERSION = 1
SIDECAR_CONTRACT_MANIFEST_WRITE_FORMAT = "krasis_hqq_selfcal_sidecar_contract_manifest_write"
SIDECAR_CONTRACT_MANIFEST_WRITE_FORMAT_VERSION = 1
SIDECAR_CONFLICT_SEARCH_FORMAT = "krasis_hqq_selfcal_sidecar_conflict_search"
SIDECAR_CONFLICT_SEARCH_FORMAT_VERSION = 1
INT8_EXCEPTION_CANDIDATES_FORMAT = "krasis_hqq_int8_exception_candidates"
INT8_EXCEPTION_CANDIDATES_FORMAT_VERSION = 1
INT8_EXCEPTION_WRITE_FORMAT = "krasis_hqq_int8_exception_manifest_write"
INT8_EXCEPTION_WRITE_FORMAT_VERSION = 1
DEFAULT_POSITIONS = "4,8,12,16,19,24,27,30,36,64,128,255,315,429,447,509"
DEFAULT_LAYERS = "all"
DEFAULT_TENSORS = (
    "la_input_row_for_qkvz_last,"
    "la_out_proj_input_last,"
    "gqa_input_norm_last,"
    "gqa_o_proj_input_last"
)
TRACE_FIELD_PATTERNS = {
    "layer": re.compile(r"\blayer=(\d+)\b"),
    "tensor": re.compile(r"\btensor=(\S+)\b"),
    "pos": re.compile(r"\bpos=(\d+)\b"),
    "width": re.compile(r"\b(?:width|n)=(\d+)\b"),
    "hash": re.compile(r"\bhash64=(\S+)\b"),
    "hex": re.compile(r"\bbf16_hex=(?:\[([0-9a-fA-F,\s]+)\]|([0-9a-fA-F]+))"),
}
ACTIVATION_TARGET_TENSORS = {
    "la_input_row_for_qkvz_last": ("in_proj_qkvz", "in_proj_ba"),
    "la_out_proj_input_last": ("out_proj",),
    "gqa_input_norm_last": ("fused_qkv", "q_proj", "k_proj", "v_proj"),
    "gqa_o_proj_input_last": ("o_proj",),
}
LINEAR_ATTENTION_HQQ_TENSORS = {"in_proj_qkvz", "in_proj_ba", "out_proj"}
GQA_SPLIT_HQQ_TENSORS = {"q_proj", "k_proj", "v_proj"}
GQA_DECODE_TENSORS = {"fused_qkv", "o_proj"}
MLA_HQQ_TENSORS = {"q_a_proj", "q_b_proj", "q_proj", "kv_a_proj_with_mqa", "o_proj"}
SHARED_ACTIVATION_FIELDS = {
    (0, "la_input_row_for_qkvz_last"),
}


def parse_config(path: str) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == "MODEL_PATH":
                key = "CFG_MODEL_PATH"
            cfg[key] = value
    return cfg


def write_config(path: str, cfg: Dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for key in sorted(cfg):
            output_key = "MODEL_PATH" if key == "CFG_MODEL_PATH" else key
            f.write(f'{output_key}="{cfg[key]}"\n')


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _bf16_hex_to_floats(hex_text: str, width: int) -> List[float]:
    hex_text = "".join(ch for ch in hex_text if ch in "0123456789abcdefABCDEF")
    if len(hex_text) < width * 4:
        raise ValueError(f"bf16_hex too short for width {width}: {len(hex_text)} hex chars")
    vals: List[float] = []
    for i in range(width):
        bits = int(hex_text[i * 4 : i * 4 + 4], 16)
        vals.append(struct.unpack(">f", struct.pack(">I", bits << 16))[0])
    return vals


def _tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        data = cpu.view(torch.uint16).numpy().tobytes()
    else:
        data = cpu.numpy().tobytes()
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_tensor_bytes(path: str) -> int:
    total = 0
    dtype_bytes = {
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
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            sl = handle.get_slice(key)
            shape = sl.get_shape()
            dtype = str(sl.get_dtype())
            total += math.prod(shape) * dtype_bytes[dtype]
    return total


def _parse_input_row_full_line(line: str) -> Optional[Dict[str, Any]]:
    if "event=input_row_full" not in line:
        return None
    parsed: Dict[str, Any] = {}
    for name, pattern in TRACE_FIELD_PATTERNS.items():
        match = pattern.search(line)
        if not match:
            return None
        if name == "hex":
            parsed[name] = match.group(1) or match.group(2)
        else:
            parsed[name] = match.group(1)
    return parsed


def _group_stats(values: List[float], group_size: int) -> List[Dict[str, Any]]:
    out = []
    for group_idx, start in enumerate(range(0, len(values), group_size)):
        chunk = values[start : start + group_size]
        abs_vals = [abs(v) for v in chunk]
        count = len(chunk)
        sum_abs = float(sum(abs_vals))
        sum_sq = float(sum(v * v for v in chunk))
        out.append(
            {
                "group": group_idx,
                "start_col": start,
                "end_col": start + count,
                "count": count,
                "max_abs": max(abs_vals) if abs_vals else 0.0,
                "mean_abs": sum_abs / count if count else 0.0,
                "rms": math.sqrt(sum_sq / count) if count else 0.0,
            }
        )
    return out


class FieldAccumulator:
    def __init__(self, layer: int, tensor: str, width: int, group_size: int) -> None:
        self.layer = layer
        self.tensor = tensor
        self.width = width
        self.group_size = group_size
        self.rows = 0
        self.hashes = set()
        self.positions = set()
        self.group_count = math.ceil(width / group_size)
        self.group_max_abs = [0.0] * self.group_count
        self.group_sum_abs = [0.0] * self.group_count
        self.group_sum_sq = [0.0] * self.group_count
        self.group_values = [0] * self.group_count

    def add(self, pos: int, row_hash: str, values: List[float]) -> None:
        if len(values) != self.width:
            raise ValueError(
                f"width changed for layer={self.layer} tensor={self.tensor}: "
                f"{len(values)} vs {self.width}"
            )
        self.rows += 1
        self.hashes.add(row_hash)
        self.positions.add(pos)
        for group in _group_stats(values, self.group_size):
            idx = group["group"]
            self.group_max_abs[idx] = max(self.group_max_abs[idx], float(group["max_abs"]))
            self.group_sum_abs[idx] += float(group["mean_abs"]) * int(group["count"])
            self.group_sum_sq[idx] += float(group["rms"]) * float(group["rms"]) * int(group["count"])
            self.group_values[idx] += int(group["count"])

    def to_json(self) -> Dict[str, Any]:
        groups = []
        for idx in range(self.group_count):
            count = self.group_values[idx]
            groups.append(
                {
                    "group": idx,
                    "start_col": idx * self.group_size,
                    "end_col": min((idx + 1) * self.group_size, self.width),
                    "sample_count": count,
                    "max_abs": self.group_max_abs[idx],
                    "mean_abs": self.group_sum_abs[idx] / count if count else 0.0,
                    "rms": math.sqrt(self.group_sum_sq[idx] / count) if count else 0.0,
                }
            )
        top_groups = sorted(groups, key=lambda g: (g["max_abs"], g["rms"]), reverse=True)[:16]
        return {
            "layer": self.layer,
            "tensor": self.tensor,
            "width": self.width,
            "group_size": self.group_size,
            "rows": self.rows,
            "unique_hashes": len(self.hashes),
            "positions": sorted(self.positions),
            "group_count": self.group_count,
            "top_groups_by_max_abs": top_groups,
            "groups": groups,
        }


def parse_trace_activation_stats(trace_path: str, group_size: int = HQQ_DEFAULT_GROUP_SIZE) -> Dict[str, Any]:
    fields: Dict[Tuple[int, str], FieldAccumulator] = {}
    total_events = 0
    with open(trace_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = _parse_input_row_full_line(line)
            if not parsed:
                continue
            total_events += 1
            layer = int(parsed["layer"])
            tensor = parsed["tensor"]
            pos = int(parsed["pos"])
            width = int(parsed["width"])
            row_hash = parsed["hash"]
            values = _bf16_hex_to_floats(parsed["hex"], width)
            key = (layer, tensor)
            if key not in fields:
                fields[key] = FieldAccumulator(layer, tensor, width, group_size)
            fields[key].add(pos, row_hash, values)

    field_stats = [fields[key].to_json() for key in sorted(fields)]
    return {
        "trace_path": trace_path,
        "input_row_full_events": total_events,
        "field_count": len(field_stats),
        "fields": field_stats,
    }


def calibration_cache_dir(model_path: str, target_profile: str) -> str:
    return hqq_attention_cache_dir(model_path, target_profile)


def evidence_default_path(model_path: str, target_profile: str) -> str:
    cache_dir = calibration_cache_dir(model_path, target_profile)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return os.path.join(cache_dir, "calibration", f"evidence_{stamp}.json")


def candidates_default_path(model_path: str, target_profile: str) -> str:
    cache_dir = calibration_cache_dir(model_path, target_profile)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return os.path.join(cache_dir, "calibration", f"sensitivity_candidates_{stamp}.json")


def refit_eval_default_path(model_path: str, target_profile: str) -> str:
    cache_dir = calibration_cache_dir(model_path, target_profile)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return os.path.join(cache_dir, "calibration", f"refit_evaluation_{stamp}.json")


def experimental_cache_write_default_path(model_path: str, target_profile: str) -> str:
    cache_dir = calibration_cache_dir(model_path, target_profile)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return os.path.join(cache_dir, "calibration", f"experimental_cache_write_{stamp}.json")


def ablation_report_default_path(model_path: str, target_profile: str) -> str:
    cache_dir = calibration_cache_dir(model_path, target_profile)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return os.path.join(cache_dir, "calibration", f"ablation_report_{stamp}.json")


def sidecar_simulation_default_path(model_path: str, target_profile: str) -> str:
    cache_dir = calibration_cache_dir(model_path, target_profile)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return os.path.join(cache_dir, "calibration", f"sidecar_simulation_{stamp}.json")


def sidecar_write_default_path(model_path: str, target_profile: str) -> str:
    cache_dir = calibration_cache_dir(model_path, target_profile)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return os.path.join(cache_dir, "calibration", f"sidecar_write_{stamp}.json")


def int8_exception_candidates_default_path(model_path: str, target_profile: str) -> str:
    cache_dir = calibration_cache_dir(model_path, target_profile)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return os.path.join(cache_dir, "calibration", f"int8_exception_candidates_{stamp}.json")


def int8_exception_write_default_path(model_path: str, target_profile: str) -> str:
    cache_dir = calibration_cache_dir(model_path, target_profile)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return os.path.join(cache_dir, "calibration", f"int8_exception_write_{stamp}.json")


def sidecar_variant_dir(model_path: str, target_profile: str, variant_name: str, mode: str) -> str:
    safe_variant = re.sub(r"[^A-Za-z0-9_.-]+", "_", variant_name).strip("._") or "unnamed"
    safe_mode = re.sub(r"[^A-Za-z0-9_.-]+", "_", mode).strip("._") or "unknown"
    return os.path.join(calibration_cache_dir(model_path, target_profile), "sidecars", safe_variant, safe_mode)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def evidence_trace_log_path(output_path: str) -> str:
    base = os.path.basename(output_path)
    stem, _ = os.path.splitext(base)
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "evidence"
    return os.path.join(os.path.dirname(output_path), f"{safe_stem}.trace.log")


def wait_for_server(port: int, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            resp.read()
            if resp.status == 200:
                return
            last_error = f"HTTP {resp.status}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"server did not become ready on port {port}: {last_error}")


def send_calibration_prompt(port: int, text: str, timeout: int) -> Dict[str, Any]:
    body = json.dumps(
        {
            "model": "krasis",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read().decode("utf-8", errors="replace")
    if resp.status != 200:
        raise RuntimeError(f"calibration request failed with HTTP {resp.status}: {data[:500]}")
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {"raw": data[:500]}


def prompt_files_for_dataset(repo_root: str, dataset: str, dataset_path: Optional[str]) -> List[str]:
    if dataset == "c4":
        if not dataset_path:
            raise ValueError("--dataset c4 requires --dataset-path pointing to local text/jsonl calibration data")
        path = Path(os.path.expanduser(dataset_path))
        if path.is_file():
            return [str(path)]
        if path.is_dir():
            files = sorted(str(p) for p in path.glob("*.txt")) + sorted(str(p) for p in path.glob("*.jsonl"))
            if files:
                return files
        raise FileNotFoundError(f"No local C4 text/jsonl files found at {path}")
    if dataset != "canonical":
        raise ValueError(f"Unsupported calibration dataset: {dataset}")
    prompt_dir = Path(repo_root) / "benchmarks" / "prompts"
    files = sorted(str(p) for p in prompt_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No canonical prompt files found in {prompt_dir}")
    return files


def read_prompt_excerpt(path: str, max_chars: int) -> str:
    p = Path(path)
    if p.suffix == ".jsonl":
        text_parts = []
        with p.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if sum(len(part) for part in text_parts) >= max_chars:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = obj.get("text") if isinstance(obj, dict) else None
                if isinstance(value, str) and value.strip():
                    text_parts.append(value)
        text = "\n\n".join(text_parts)
    else:
        text = p.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars] + "\n\nSummarize the above text briefly."


def model_num_hidden_layers(model_path: str) -> int:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    value = cfg.get("num_hidden_layers")
    if value is None and isinstance(cfg.get("text_config"), dict):
        value = cfg["text_config"].get("num_hidden_layers")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Cannot determine num_hidden_layers from {config_path}")
    return value


def run_calibration_requests(
    *,
    config_path: str,
    cfg: Dict[str, str],
    repo_root: str,
    duration_seconds: int,
    dataset: str,
    dataset_path: Optional[str],
    prompt_chars: int,
    startup_timeout: int,
    server_log_path: str,
    source_profile: str,
    positions: str,
    layers: str,
    tensors: str,
) -> Dict[str, Any]:
    port = int(cfg.get("CFG_PORT", "8012"))
    launch_cfg = dict(cfg)
    launch_cfg["CFG_ATTENTION_QUANT"] = "hqq4"
    launch_cfg["CFG_HQQ_CACHE_PROFILE"] = source_profile
    with tempfile.NamedTemporaryFile("w", suffix=".conf", prefix="krasis-hqq-selfcal-", delete=False) as tmp:
        launch_config_path = tmp.name
    write_config(launch_config_path, launch_cfg)

    env = os.environ.copy()
    env.update(
        {
            "KRASIS_TRACE": "1",
            "KRASIS_TRACE_COMPONENTS": "prefill_layer_summary",
            "KRASIS_TRACE_INPUT_ROWS_ONLY": "1",
            "KRASIS_TRACE_INPUT_ROW_POSITIONS": positions,
            "KRASIS_TRACE_INPUT_ROW_LAYERS": layers,
            "KRASIS_TRACE_INPUT_ROW_TENSORS": tensors,
        }
    )

    python = sys.executable
    command = [python, "-m", "krasis.server", "--config", launch_config_path]
    prompt_files = prompt_files_for_dataset(repo_root, dataset, dataset_path)
    requests: List[Dict[str, Any]] = []
    proc: Optional[subprocess.Popen] = None
    start = time.time()
    try:
        os.makedirs(os.path.dirname(server_log_path), exist_ok=True)
        with open(server_log_path, "w", encoding="utf-8") as server_log:
            proc = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, env=env)
            wait_for_server(port, startup_timeout)
            deadline = start + max(1, duration_seconds)
            idx = 0
            while time.time() < deadline:
                prompt_path = prompt_files[idx % len(prompt_files)]
                text = read_prompt_excerpt(prompt_path, prompt_chars)
                t0 = time.time()
                result = send_calibration_prompt(port, text, timeout=max(60, duration_seconds + 60))
                elapsed = time.time() - t0
                requests.append(
                    {
                        "prompt_file": prompt_path,
                        "prompt_chars": len(text),
                        "elapsed_seconds": elapsed,
                        "response_keys": sorted(result.keys()),
                    }
                )
                idx += 1
                if idx >= len(prompt_files):
                    break
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=20)
        try:
            os.remove(launch_config_path)
        except OSError:
            pass

    return {
        "server_command": command,
        "server_log": server_log_path,
        "duration_limit_seconds": duration_seconds,
        "elapsed_seconds": time.time() - start,
        "requests": requests,
        "request_count": len(requests),
        "trace_selectors": {
            "positions": positions,
            "layers": layers,
            "tensors": tensors,
        },
    }


def build_evidence_payload(
    *,
    config_path: str,
    cfg: Dict[str, str],
    source_profile: str,
    target_profile: str,
    duration_seconds: int,
    dataset: str,
    output_path: str,
    activation_stats: Dict[str, Any],
    run_info: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    model_path = os.path.expanduser(cfg.get("CFG_MODEL_PATH", ""))
    baseline_manifest = load_hqq_attention_manifest(model_path, HQQ_CACHE_PROFILE_BASELINE)
    source_manifest = load_hqq_attention_manifest(model_path, source_profile)
    return {
        "format": EVIDENCE_FORMAT,
        "format_version": EVIDENCE_FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "dry_run" if dry_run else "evidence_captured",
        "complete": False,
        "loadable_hqq_manifest": False,
        "not_loadable_reason": (
            "Phase 2A evidence only. Refit has not written calibrated HQQ tensor artifacts "
            "or a complete manifest."
        ),
        "config_path": config_path,
        "model_path": model_path,
        "cache": {
            "base_model_cache_dir": cache_dir_for_model(model_path),
            "baseline_cache_dir": hqq_attention_cache_dir(model_path, HQQ_CACHE_PROFILE_BASELINE),
            "source_cache_profile": source_profile,
            "source_cache_dir": hqq_attention_cache_dir(model_path, source_profile),
            "target_cache_profile": target_profile,
            "target_cache_dir": hqq_attention_cache_dir(model_path, target_profile),
            "target_manifest_path": hqq_attention_manifest_path(model_path, target_profile),
            "target_manifest_written": False,
            "evidence_path": output_path,
        },
        "source_manifest": {
            "present": source_manifest is not None,
            "complete": bool(source_manifest and source_manifest.get("complete")),
            "tensor_count": (
                source_manifest.get("totals", {}).get("num_tensors")
                if isinstance(source_manifest, dict)
                else None
            ),
        },
        "baseline_manifest": {
            "present": baseline_manifest is not None,
            "complete": bool(baseline_manifest and baseline_manifest.get("complete")),
        },
        "calibration": {
            "duration_limit_seconds": duration_seconds,
            "dataset": dataset,
            "phase": "2A_evidence_capture_only",
            "refit_started": False,
            "weights_modified": False,
            "qvalues_modified": False,
            "scale_zero_modified": False,
        },
        "run": run_info,
        "activation_stats": activation_stats,
    }


def _manifest_tensor_entries(manifest: Dict[str, Any]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    entries: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for entry in manifest.get("tensors", []):
        layer = int(entry.get("layer_idx", -1))
        tensor = str(entry.get("tensor_name", ""))
        if layer >= 0 and tensor:
            entries[(layer, tensor)] = entry
    return entries


def _quality_worst_by_group(entry: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    grouped: Dict[int, Dict[str, Any]] = {}
    for item in entry.get("quality", {}).get("worst_groups", []) or []:
        group = int(item.get("group", -1))
        if group < 0:
            continue
        current = grouped.setdefault(
            group,
            {
                "group": group,
                "rows": [],
                "max_mean_abs": 0.0,
                "max_rmse": 0.0,
                "max_abs": 0.0,
            },
        )
        current["rows"].append(int(item.get("row", -1)))
        current["max_mean_abs"] = max(current["max_mean_abs"], float(item.get("mean_abs", 0.0)))
        current["max_rmse"] = max(current["max_rmse"], float(item.get("rmse", 0.0)))
        current["max_abs"] = max(current["max_abs"], float(item.get("max_abs", 0.0)))
    return grouped


def _unpack_uint4(packed: torch.Tensor, cols: int) -> torch.Tensor:
    low = packed.bitwise_and(0x0F)
    high = packed.bitwise_right_shift(4).bitwise_and(0x0F)
    out = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8)
    out[:, 0::2] = low
    out[:, 1::2] = high
    return out[:, :cols].contiguous()


def _pack_uint4(qvalues: torch.Tensor, padded_cols: Optional[int] = None) -> torch.Tensor:
    q = qvalues.to(torch.uint8).contiguous()
    if padded_cols is not None and q.shape[1] < padded_cols:
        q = torch.nn.functional.pad(q, (0, padded_cols - q.shape[1]))
    if q.shape[1] % 2:
        q = torch.nn.functional.pad(q, (0, 1))
    low = q[:, 0::2].bitwise_and(0x0F)
    high = q[:, 1::2].bitwise_and(0x0F).bitwise_left_shift(4)
    return low.bitwise_or(high).contiguous()


def _load_index_tensor(model_dir: Path, name: str) -> torch.Tensor:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"Missing safetensors index: {index_path}")
    with index_path.open(encoding="utf-8") as f:
        index = json.load(f)
    shard_name = index.get("weight_map", {}).get(name)
    if not shard_name:
        raise RuntimeError(f"Tensor {name!r} not found in {index_path}")
    with safe_open(str(model_dir / shard_name), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name).contiguous()


def _load_qwen35_source_tensor(model_dir: Path, layer: int, tensor_name: str) -> torch.Tensor:
    if tensor_name in ("in_proj_qkvz", "in_proj_ba"):
        prefix = f"model.language_model.layers.{layer}.linear_attn"
        config_path = model_dir / "config.json"
        with config_path.open(encoding="utf-8") as f:
            cfg = json.load(f).get("text_config", {})
        nk = int(cfg["linear_num_key_heads"])
        nv = int(cfg["linear_num_value_heads"])
        hr = nv // nk
        if tensor_name == "in_proj_qkvz":
            dk = int(cfg["linear_key_head_dim"])
            dv = int(cfg["linear_value_head_dim"])
            qkv_raw = _load_index_tensor(model_dir, f"{prefix}.in_proj_qkv.weight")
            z_raw = _load_index_tensor(model_dir, f"{prefix}.in_proj_z.weight")
            key_dim = nk * dk
            parts = []
            for head in range(nk):
                parts.append(qkv_raw[head * dk : (head + 1) * dk])
                parts.append(qkv_raw[key_dim + head * dk : key_dim + (head + 1) * dk])
                parts.append(qkv_raw[key_dim * 2 + head * hr * dv : key_dim * 2 + (head + 1) * hr * dv])
                parts.append(z_raw[head * hr * dv : (head + 1) * hr * dv])
            return torch.cat(parts, dim=0).contiguous()
        b_raw = _load_index_tensor(model_dir, f"{prefix}.in_proj_b.weight")
        a_raw = _load_index_tensor(model_dir, f"{prefix}.in_proj_a.weight")
        parts = []
        for head in range(nk):
            parts.append(b_raw[head * hr : (head + 1) * hr])
            parts.append(a_raw[head * hr : (head + 1) * hr])
        return torch.cat(parts, dim=0).contiguous()
    if tensor_name == "out_proj":
        prefix = f"model.language_model.layers.{layer}.linear_attn"
        return _load_index_tensor(model_dir, f"{prefix}.out_proj.weight")
    if tensor_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        prefix = f"model.language_model.layers.{layer}.self_attn"
        return _load_index_tensor(model_dir, f"{prefix}.{tensor_name}.weight")
    if tensor_name == "fused_qkv":
        prefix = f"model.language_model.layers.{layer}.self_attn"
        return torch.cat(
            [
                _load_index_tensor(model_dir, f"{prefix}.q_proj.weight"),
                _load_index_tensor(model_dir, f"{prefix}.k_proj.weight"),
                _load_index_tensor(model_dir, f"{prefix}.v_proj.weight"),
            ],
            dim=0,
        ).contiguous()
    raise RuntimeError(f"No source tensor resolver for {tensor_name!r}")


def _load_source_tensor_from_contract(model_dir: Path, entry: Dict[str, Any]) -> torch.Tensor:
    contract = entry.get("source_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("HQQ source contract is missing")
    if not bool(contract.get("ready")):
        raise RuntimeError(f"HQQ source contract is not ready: {contract.get('missing_contract_fields')}")
    if bool(contract.get("fallback_allowed")):
        raise RuntimeError("HQQ source contract must be fail-closed; fallback_allowed=true is invalid")

    resolver_kind = str(contract.get("resolver_kind") or "")
    layer = int(entry["layer_idx"])
    tensor_name = str(entry["tensor_name"])
    source_items = contract.get("source_tensors")
    if not isinstance(source_items, list) or not source_items:
        raise RuntimeError("HQQ source contract has no source_tensors")
    source_names = [str(item.get("name") or "") for item in source_items]
    if any(not name for name in source_names):
        raise RuntimeError("HQQ source contract contains an empty source tensor name")

    for item, name in zip(source_items, source_names):
        expected_hash = item.get("source_sha256")
        if not isinstance(expected_hash, str) or not expected_hash:
            raise RuntimeError(f"HQQ source contract missing source_sha256 for {name}")
        actual_hash = _tensor_hash(_load_index_tensor(model_dir, name))
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"HQQ source tensor hash mismatch for {name}: {actual_hash} != {expected_hash}"
            )

    if resolver_kind == "qcn_model_layers_direct_fused":
        if len(source_names) != 1 or tensor_name not in ("in_proj_qkvz", "in_proj_ba"):
            raise RuntimeError(f"Invalid QCN direct-fused contract for {tensor_name}")
        source = _load_index_tensor(model_dir, source_names[0]).contiguous()
    elif resolver_kind == "qcn_model_layers_direct":
        if len(source_names) != 1:
            raise RuntimeError(f"Invalid QCN direct contract for {tensor_name}")
        source = _load_index_tensor(model_dir, source_names[0]).contiguous()
    elif resolver_kind == "qcn_model_layers_self_attn_qkv_concat":
        if len(source_names) != 3 or tensor_name != "fused_qkv":
            raise RuntimeError(f"Invalid QCN fused_qkv concat contract for {tensor_name}")
        source = torch.cat([_load_index_tensor(model_dir, name) for name in source_names], dim=0).contiguous()
    elif resolver_kind in {
        "q35_language_model_split_interleave",
        "q35_language_model_direct",
        "q35_language_model_self_attn_qkv_concat",
    }:
        source = _load_qwen35_source_tensor(model_dir, layer, tensor_name)
    else:
        raise RuntimeError(f"Unsupported HQQ source resolver_kind={resolver_kind!r}")

    expected_shape = contract.get("reconstructed_source_shape") or contract.get("source_shape")
    if expected_shape is not None and list(source.shape) != [int(v) for v in expected_shape]:
        raise RuntimeError(
            f"HQQ source contract shape mismatch for layer={layer} tensor={tensor_name}: "
            f"{list(source.shape)} != {expected_shape}"
        )
    expected_reconstructed_hash = contract.get("reconstructed_source_sha256")
    if not isinstance(expected_reconstructed_hash, str) or not expected_reconstructed_hash:
        raise RuntimeError("HQQ source contract missing reconstructed_source_sha256")
    actual_reconstructed_hash = _tensor_hash(source)
    if actual_reconstructed_hash != expected_reconstructed_hash:
        raise RuntimeError(
            f"HQQ reconstructed source hash mismatch for layer={layer} tensor={tensor_name}: "
            f"{actual_reconstructed_hash} != {expected_reconstructed_hash}"
        )
    return source.contiguous()


def _load_source_tensor(model_path: str, entry: Dict[str, Any]) -> torch.Tensor:
    model_dir = Path(model_path)
    if isinstance(entry.get("source_contract"), dict):
        return _load_source_tensor_from_contract(model_dir, entry)
    source_tensor = entry.get("source_tensor")
    if isinstance(source_tensor, str) and source_tensor:
        return _load_index_tensor(model_dir, source_tensor)
    return _load_qwen35_source_tensor(
        model_dir,
        int(entry["layer_idx"]),
        str(entry["tensor_name"]),
    )


def _load_hqq_tensor(cache_dir: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    path = os.path.join(cache_dir, entry["file"])
    contract = entry.get("source_contract")
    if isinstance(contract, dict):
        expected_artifact_hash = contract.get("artifact_sha256")
        if not isinstance(expected_artifact_hash, str) or not expected_artifact_hash:
            raise RuntimeError(
                f"HQQ source contract missing artifact_sha256 for layer={entry.get('layer_idx')} "
                f"tensor={entry.get('tensor_name')}"
            )
        actual_artifact_hash = _file_sha256(path)
        if actual_artifact_hash != expected_artifact_hash:
            raise RuntimeError(
                f"HQQ artifact hash mismatch for layer={entry.get('layer_idx')} "
                f"tensor={entry.get('tensor_name')}: {actual_artifact_hash} != {expected_artifact_hash}"
            )
    with safe_open(path, framework="pt", device="cpu") as handle:
        packed = handle.get_tensor("packed").to(torch.uint8).contiguous()
        scales = handle.get_tensor("scales").to(torch.float32).contiguous()
        zeros = handle.get_tensor("zeros").to(torch.float32).contiguous()
        orig_shape = handle.get_tensor("orig_shape")
        group_size = int(handle.get_tensor("group_size")[0].item())
        axis = int(handle.get_tensor("axis")[0].item())
        nbits = int(handle.get_tensor("nbits")[0].item())
    rows = int(orig_shape[0].item())
    cols = int(orig_shape[1].item())
    if axis != 1 or nbits != 4:
        raise RuntimeError(f"Unsupported HQQ artifact for refit evaluation: axis={axis}, nbits={nbits}")
    groups = math.ceil(cols / group_size)
    qvalues = _unpack_uint4(packed, cols)
    scale_full = scales.repeat_interleave(group_size, dim=1)[:, :cols]
    zero_full = zeros.repeat_interleave(group_size, dim=1)[:, :cols]
    dequant = (qvalues.to(torch.float32) - zero_full) * scale_full
    return {
        "path": path,
        "qvalues": qvalues,
        "scales": scales,
        "zeros": zeros,
        "dequant": dequant,
        "rows": rows,
        "cols": cols,
        "groups": groups,
        "group_size": group_size,
        "axis": axis,
        "nbits": nbits,
    }


def _load_source_contract_audit(source_contract_audit_path: str) -> Dict[str, Any]:
    with open(source_contract_audit_path, encoding="utf-8") as f:
        audit = json.load(f)
    if audit.get("format") != "krasis_hqq_source_map_audit":
        raise ValueError(f"Not a Krasis source-map audit report: {source_contract_audit_path}")
    targets = audit.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"Source-map audit has no targets: {source_contract_audit_path}")
    return audit


def _source_contract_lookup_from_audit(source_contract_audit_path: str) -> Dict[Tuple[int, str], Dict[str, Any]]:
    audit = _load_source_contract_audit(source_contract_audit_path)
    lookup: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for target in audit.get("targets", []) or []:
        layer = int(target.get("layer", -1))
        tensor_name = str(target.get("tensor") or "")
        if layer < 0 or not tensor_name:
            raise RuntimeError(f"Invalid source-contract audit target identity: {target}")
        contract = target.get("source_contract")
        if not isinstance(contract, dict):
            raise RuntimeError(f"Source-contract audit target missing source_contract: {layer}:{tensor_name}")
        if not bool(contract.get("ready")):
            raise RuntimeError(
                f"Source contract is not ready for {layer}:{tensor_name}: "
                f"{contract.get('missing_contract_fields')}"
            )
        if bool(contract.get("fallback_allowed")):
            raise RuntimeError(f"Source contract must be fail-closed for {layer}:{tensor_name}")
        key = (layer, tensor_name)
        if key in lookup:
            raise RuntimeError(f"Duplicate source-contract audit target: {layer}:{tensor_name}")
        lookup[key] = copy.deepcopy(contract)
    return lookup


def _attach_source_contract(
    entry: Dict[str, Any],
    source_contract_lookup: Dict[Tuple[int, str], Dict[str, Any]],
) -> Dict[str, Any]:
    layer = int(entry["layer_idx"])
    tensor_name = str(entry["tensor_name"])
    contract = source_contract_lookup.get((layer, tensor_name))
    if not isinstance(contract, dict):
        raise RuntimeError(f"Missing strict source_contract for layer={layer} tensor={tensor_name}")
    out = copy.deepcopy(entry)
    out["source_contract"] = copy.deepcopy(contract)
    return out


def _validate_source_contract_entry(
    *,
    model_path: str,
    cache_dir: str,
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    layer = int(entry["layer_idx"])
    tensor_name = str(entry["tensor_name"])
    contract = entry.get("source_contract")
    if not isinstance(contract, dict):
        raise RuntimeError(f"Missing source_contract for layer={layer} tensor={tensor_name}")
    if not bool(contract.get("ready")):
        raise RuntimeError(
            f"Source contract is not ready for layer={layer} tensor={tensor_name}: "
            f"{contract.get('missing_contract_fields')}"
        )
    if bool(contract.get("fallback_allowed")):
        raise RuntimeError(f"Source contract fallback_allowed=true for layer={layer} tensor={tensor_name}")
    source = _load_source_tensor(model_path, entry)
    hqq = _load_hqq_tensor(cache_dir, entry)
    expected_shape = [int(hqq["rows"]), int(hqq["cols"])]
    actual_shape = [int(v) for v in source.shape]
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"source/HQQ shape mismatch for layer={layer} tensor={tensor_name}: "
            f"{actual_shape} vs {expected_shape}"
        )
    if contract.get("target_tensor") != tensor_name:
        raise RuntimeError(
            f"Source contract target tensor mismatch for layer={layer}: "
            f"{contract.get('target_tensor')} != {tensor_name}"
        )
    if int(contract.get("layer", -1)) != layer:
        raise RuntimeError(
            f"Source contract layer mismatch for tensor={tensor_name}: {contract.get('layer')} != {layer}"
        )
    if str(contract.get("target_file") or "") != str(entry.get("file") or ""):
        raise RuntimeError(
            f"Source contract target file mismatch for layer={layer} tensor={tensor_name}: "
            f"{contract.get('target_file')} != {entry.get('file')}"
        )
    return {
        "contract_embedded": True,
        "resolver_valid": True,
        "source_hash_valid": True,
        "artifact_hash_valid": True,
        "shape_valid": True,
        "generation_ready": True,
        "runtime_ready": False,
        "runtime_ready_reason": "offline contract only; runtime sidecar enablement remains explicit and out of scope",
    }


def _source_contract_summary(validations: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(validations)
    total = len(items)
    return {
        "target_count": total,
        "contract_embedded": sum(1 for item in items if item.get("contract_embedded")),
        "resolver_valid": sum(1 for item in items if item.get("resolver_valid")),
        "source_hash_valid": sum(1 for item in items if item.get("source_hash_valid")),
        "artifact_hash_valid": sum(1 for item in items if item.get("artifact_hash_valid")),
        "shape_valid": sum(1 for item in items if item.get("shape_valid")),
        "generation_ready": sum(1 for item in items if item.get("generation_ready")),
        "runtime_ready": sum(1 for item in items if item.get("runtime_ready")),
        "runtime_enablement_changed": False,
        "fallback_allowed": False,
    }


def _qvalue_saturation_by_group(cache_dir: str, entry: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    path = os.path.join(cache_dir, entry["file"])
    if not os.path.isfile(path):
        return {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        packed = handle.get_tensor("packed")
        orig_shape = handle.get_tensor("orig_shape")
        group_size = int(handle.get_tensor("group_size")[0].item())
    cols = int(orig_shape[1].item())
    qvalues = _unpack_uint4(packed, cols)
    out: Dict[int, Dict[str, float]] = {}
    for group, start in enumerate(range(0, cols, group_size)):
        end = min(start + group_size, cols)
        chunk = qvalues[:, start:end]
        total = max(1, int(chunk.numel()))
        zeros = int((chunk == 0).sum().item())
        maxes = int((chunk == 15).sum().item())
        out[group] = {
            "zero_fraction": zeros / total,
            "max_fraction": maxes / total,
            "saturation_fraction": (zeros + maxes) / total,
        }
    return out


def _activation_evidence_policy(layer: int, activation_tensor: str) -> str:
    if (layer, activation_tensor) in SHARED_ACTIVATION_FIELDS:
        return "shared_bf16_hqq_activation"
    return "lane_local_only"


def _candidate_score(
    *,
    activation_rms: float,
    activation_max_abs: float,
    quality_rmse: float,
    quality_max_abs: float,
    saturation_fraction: float,
) -> float:
    # Ranking heuristic only: it identifies groups worth later refit evaluation.
    # It is not a refit objective and does not approve a calibrated cache.
    return (activation_rms * quality_rmse) + (0.25 * activation_max_abs * quality_max_abs) + (
        0.05 * saturation_fraction * activation_max_abs
    )


def build_sensitivity_candidates(
    *,
    cfg: Dict[str, str],
    evidence_path: str,
    output_path: str,
    top_candidates: int,
    source_contract_audit_path: Optional[str] = None,
) -> Dict[str, Any]:
    with open(evidence_path, encoding="utf-8") as f:
        evidence = json.load(f)
    if evidence.get("format") != EVIDENCE_FORMAT:
        raise ValueError(f"Not an HQQ self-calibration evidence file: {evidence_path}")

    model_path = os.path.expanduser(cfg.get("CFG_MODEL_PATH", evidence.get("model_path", "")))
    if not model_path:
        raise ValueError("Cannot determine model path for sensitivity candidate build")
    source_profile = (
        evidence.get("cache", {}).get("source_cache_profile")
        or cfg.get("CFG_HQQ_CACHE_PROFILE")
        or HQQ_CACHE_PROFILE_BASELINE
    )
    source_profile = str(source_profile).strip().lower()
    manifest_probe = load_hqq_attention_manifest(model_path, source_profile)
    if manifest_probe is None:
        raise RuntimeError(f"No HQQ source manifest found for profile {source_profile}")
    expected_layers = int(manifest_probe.get("num_hidden_layers", 0))
    manifest = require_complete_hqq_attention_manifest(
        model_path,
        source_profile,
        expected_nbits=4,
        expected_num_hidden_layers=expected_layers,
    )
    entries = _manifest_tensor_entries(manifest)
    tensor_names_by_layer = _manifest_tensor_names_by_layer(entries)
    cache_dir = hqq_attention_cache_dir(model_path, source_profile)
    source_contract_lookup = (
        _source_contract_lookup_from_audit(source_contract_audit_path)
        if source_contract_audit_path
        else {}
    )
    source_contract_validations: Dict[Tuple[int, str], Dict[str, Any]] = {}

    candidates: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    saturation_cache: Dict[Tuple[int, str], Dict[int, Dict[str, float]]] = {}
    fields = evidence.get("activation_stats", {}).get("fields", []) or []
    for field in fields:
        layer = int(field["layer"])
        activation_tensor = str(field["tensor"])
        width = int(field["width"])
        target_tensors = ACTIVATION_TARGET_TENSORS.get(activation_tensor, ())
        if not target_tensors:
            skipped.append(
                {
                    "layer": layer,
                    "activation_tensor": activation_tensor,
                    "reason": "no HQQ target tensor mapping for activation field",
                }
            )
            continue
        for tensor_name in target_tensors:
            key = (layer, tensor_name)
            entry = entries.get(key)
            if entry is None:
                skipped.append(
                    {
                        "layer": layer,
                        "activation_tensor": activation_tensor,
                        "target_tensor": tensor_name,
                        "reason": "HQQ tensor is not present in selected cache manifest",
                    }
                )
                continue
            if source_contract_lookup and key not in source_contract_validations:
                try:
                    entry = _attach_source_contract(entry, source_contract_lookup)
                    source_contract_validations[key] = _validate_source_contract_entry(
                        model_path=model_path,
                        cache_dir=cache_dir,
                        entry=entry,
                    )
                except Exception as exc:
                    skipped.append(
                        {
                            "layer": layer,
                            "activation_tensor": activation_tensor,
                            "target_tensor": tensor_name,
                            "reason": f"strict source-contract validation failed: {exc}",
                        }
                    )
                    continue
            shape = entry.get("original_shape") or []
            if len(shape) != 2 or int(shape[1]) != width:
                skipped.append(
                    {
                        "layer": layer,
                        "activation_tensor": activation_tensor,
                        "target_tensor": tensor_name,
                        "reason": "activation width does not match HQQ tensor input width",
                        "activation_width": width,
                        "hqq_shape": shape,
                    }
                )
                continue
            quality = entry.get("quality", {}) or {}
            global_rmse = float(quality.get("rmse", 0.0))
            global_max_abs = float(quality.get("max_abs", 0.0))
            worst_by_group = _quality_worst_by_group(entry)
            if key not in saturation_cache:
                saturation_cache[key] = _qvalue_saturation_by_group(cache_dir, entry)
            saturation_by_group = saturation_cache[key]
            for group in field.get("groups", []) or []:
                group_idx = int(group["group"])
                worst = worst_by_group.get(group_idx, {})
                saturation = saturation_by_group.get(group_idx, {})
                quality_rmse = float(worst.get("max_rmse", global_rmse))
                quality_max_abs = float(worst.get("max_abs", global_max_abs))
                activation_rms = float(group.get("rms", 0.0))
                activation_max_abs = float(group.get("max_abs", 0.0))
                score = _candidate_score(
                    activation_rms=activation_rms,
                    activation_max_abs=activation_max_abs,
                    quality_rmse=quality_rmse,
                    quality_max_abs=quality_max_abs,
                    saturation_fraction=float(saturation.get("saturation_fraction", 0.0)),
                )
                candidates.append(
                    {
                        "layer": layer,
                        "target_tensor": tensor_name,
                        "activation_tensor": activation_tensor,
                        "activation_policy": _activation_evidence_policy(layer, activation_tensor),
                        "group": group_idx,
                        "start_col": int(group.get("start_col", group_idx * int(entry.get("group_size", 0)))),
                        "end_col": int(group.get("end_col", 0)),
                        "score": score,
                        "activation": {
                            "rows": int(field.get("rows", 0)),
                            "unique_hashes": int(field.get("unique_hashes", 0)),
                            "max_abs": activation_max_abs,
                            "mean_abs": float(group.get("mean_abs", 0.0)),
                            "rms": activation_rms,
                        },
                        "hqq_quality": {
                            "global_rmse": global_rmse,
                            "global_max_abs": global_max_abs,
                            "group_rmse": quality_rmse,
                            "group_max_abs": quality_max_abs,
                            "worst_group_rows": sorted(set(worst.get("rows", []))),
                        },
                        "qvalue_saturation": {
                            "zero_fraction": float(saturation.get("zero_fraction", 0.0)),
                            "max_fraction": float(saturation.get("max_fraction", 0.0)),
                            "saturation_fraction": float(saturation.get("saturation_fraction", 0.0)),
                        },
                    }
                )

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    payload = {
        "format": CANDIDATES_FORMAT,
        "format_version": CANDIDATES_FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "sensitivity_candidates_built",
        "complete": False,
        "loadable_hqq_manifest": False,
        "not_loadable_reason": (
            "Phase 2B sensitivity candidates only. No refit has written calibrated HQQ "
            "tensor artifacts or a complete manifest."
        ),
        "model_path": model_path,
        "source_evidence_path": evidence_path,
        "cache": {
            "source_cache_profile": source_profile,
            "source_cache_dir": cache_dir,
            "target_cache_profile": HQQ_CACHE_PROFILE_SELFCAL_V1,
            "target_cache_dir": hqq_attention_cache_dir(model_path, HQQ_CACHE_PROFILE_SELFCAL_V1),
            "target_manifest_path": hqq_attention_manifest_path(model_path, HQQ_CACHE_PROFILE_SELFCAL_V1),
            "target_manifest_written": False,
            "candidate_path": output_path,
            "source_contract_audit_path": source_contract_audit_path,
            "source_contract_audit_sha256": _file_sha256(source_contract_audit_path)
            if source_contract_audit_path
            else None,
        },
        "sensitivity_model": {
            "purpose": "rank groups for later offline refit evaluation; not a refit decision",
            "inputs": [
                "activation coverage/intensity from Phase 2A evidence",
                "existing HQQ manifest reconstruction quality metadata",
                "packed qvalue saturation from the selected source cache when available",
            ],
            "score": "activation_rms * group_quality_rmse + 0.25 * activation_max_abs * group_quality_max_abs + 0.05 * saturation_fraction * activation_max_abs",
            "shared_activation_rule": (
                "Only layer-0 la_input_row_for_qkvz_last is currently proven shared between BF16 and HQQ. "
                "Downstream/deeper/GQA fields are lane-local evidence and must not be treated as shared BF16/HQQ activations."
            ),
        },
        "summary": {
            "activation_field_count": len(fields),
            "candidate_count": len(candidates),
            "top_candidate_count": min(top_candidates, len(candidates)),
            "skipped_count": len(skipped),
            "source_contract": _source_contract_summary(source_contract_validations.values())
            if source_contract_lookup
            else None,
        },
        "top_candidates": candidates[:top_candidates],
        "candidates": candidates,
        "skipped": skipped,
    }
    write_json(output_path, payload)
    latest_path = os.path.join(os.path.dirname(output_path), "sensitivity_candidates_latest.json")
    write_json(latest_path, payload)
    return payload


def _evidence_trace_path(evidence_path: str, evidence: Dict[str, Any]) -> str:
    trace_path = str(evidence.get("activation_stats", {}).get("trace_path") or "")
    if not trace_path:
        raise RuntimeError(f"Evidence file does not record activation_stats.trace_path: {evidence_path}")
    if os.path.isfile(trace_path):
        return trace_path
    base = os.path.dirname(evidence_path)
    candidate = os.path.join(base, trace_path)
    if os.path.isfile(candidate):
        return candidate
    raise RuntimeError(f"Evidence trace path is not readable: {trace_path}")


def _ensure_distinct_evidence_traces(
    *,
    train_evidence_path: str,
    train_trace_path: str,
    heldout_evidence_path: str,
    heldout_trace_path: str,
) -> None:
    if os.path.abspath(train_trace_path) == os.path.abspath(heldout_trace_path):
        raise RuntimeError(
            "Train and heldout evidence resolve to the same trace log. "
            "Evidence trace logs are mutable calibration outputs and must be unique per evidence file: "
            f"train={train_evidence_path} heldout={heldout_evidence_path} trace={train_trace_path}"
        )


def _activation_fields_by_key(evidence: Dict[str, Any]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    out: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for field in evidence.get("activation_stats", {}).get("fields", []) or []:
        key = (int(field["layer"]), str(field["tensor"]))
        if key in out:
            raise RuntimeError(f"Duplicate activation field in evidence: layer={key[0]} tensor={key[1]}")
        out[key] = field
    return out


def _manifest_tensor_names_by_layer(entries: Dict[Tuple[int, str], Dict[str, Any]]) -> Dict[int, set[str]]:
    by_layer: Dict[int, set[str]] = {}
    for layer_idx, tensor_name in entries:
        by_layer.setdefault(int(layer_idx), set()).add(str(tensor_name))
    return by_layer


def _int8_exception_runtime_activity(
    *,
    layer: int,
    tensor_name: str,
    tensor_names_by_layer: Dict[int, set[str]],
) -> Dict[str, Any]:
    layer_tensors = tensor_names_by_layer.get(int(layer), set())
    tensor_name = str(tensor_name)
    prefill_active = False
    decode_active = False
    inactive_reason: Optional[str] = None

    if tensor_name in LINEAR_ATTENTION_HQQ_TENSORS:
        prefill_active = True
        decode_active = True
    elif tensor_name == "fused_qkv":
        prefill_active = False
        decode_active = "fused_qkv" in layer_tensors
    elif tensor_name in GQA_SPLIT_HQQ_TENSORS:
        prefill_active = True
        if "fused_qkv" in layer_tensors:
            decode_active = False
            inactive_reason = "decode uses fused_qkv for this layer; split q/k/v sidecars are prefill-only"
        else:
            decode_active = True
    elif tensor_name == "o_proj":
        prefill_active = True
        decode_active = True
    elif tensor_name in MLA_HQQ_TENSORS:
        prefill_active = True
        decode_active = True
    else:
        inactive_reason = "tensor is not mapped to an active HQQ prefill/decode sidecar surface"

    if prefill_active and decode_active:
        surface = "both"
    elif decode_active:
        surface = "decode_only"
    elif prefill_active:
        surface = "prefill_only"
    else:
        surface = "inactive"

    return {
        "prefill_active": bool(prefill_active),
        "decode_active": bool(decode_active),
        "active": bool(prefill_active or decode_active),
        "surface": surface,
        "inactive_reason": inactive_reason,
    }


def _safe_ratio(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1e-30)


def _tensor_error_metrics(diff: torch.Tensor) -> Dict[str, Any]:
    values = diff.to(torch.float32).contiguous()
    abs_values = values.abs()
    sq = values * values
    return {
        "rms": float(torch.sqrt(torch.mean(sq)).item()) if values.numel() else 0.0,
        "mean_abs": float(abs_values.mean().item()) if values.numel() else 0.0,
        "max_abs": float(abs_values.max().item()) if values.numel() else 0.0,
    }


def _projection_reduction(int8_projection: Dict[str, Any], hqq_projection: Dict[str, Any]) -> Dict[str, Any]:
    rms_ratio = _safe_ratio(float(int8_projection.get("rms", 0.0)), float(hqq_projection.get("rms", 0.0)))
    abs_ratio = _safe_ratio(float(int8_projection.get("abs_sum", 0.0)), float(hqq_projection.get("abs_sum", 0.0)))
    max_ratio = _safe_ratio(float(int8_projection.get("max_abs", 0.0)), float(hqq_projection.get("max_abs", 0.0)))
    return {
        "rms_ratio": rms_ratio,
        "rms_reduction_fraction": 1.0 - rms_ratio,
        "abs_sum_ratio": abs_ratio,
        "abs_sum_reduction_fraction": 1.0 - abs_ratio,
        "max_abs_ratio": max_ratio,
        "max_abs_reduction_fraction": 1.0 - max_ratio,
    }


def _write_int8_exception_candidates_tsv(path: str, rows: List[Dict[str, Any]]) -> None:
    header = [
        "rank",
        "layer",
        "tensor",
        "group",
        "start_col",
        "end_col",
        "risk_score",
        "benefit_score",
        "train_rms_reduction",
        "heldout_rms_reduction",
        "heldout_hqq_rms",
        "heldout_int8_rms",
        "int8_total_bytes",
        "runtime_total_bytes",
        "runtime_surface",
        "decode_active",
        "prefill_active",
        "candidate_ready",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for idx, row in enumerate(rows, start=1):
            train = row["projection"]["train"]["reduction_vs_hqq"]
            heldout = row["projection"]["heldout"]["reduction_vs_hqq"]
            f.write(
                "\t".join(
                    [
                        str(idx),
                        str(row["layer"]),
                        str(row["target_tensor"]),
                        str(row["group"]),
                        str(row["start_col"]),
                        str(row["end_col"]),
                        f"{float(row['risk_score']):.10g}",
                        f"{float(row['benefit_score']):.10g}",
                        f"{float(train['rms_reduction_fraction']):.10g}",
                        f"{float(heldout['rms_reduction_fraction']):.10g}",
                        f"{float(row['projection']['heldout']['hqq']['rms']):.10g}",
                        f"{float(row['projection']['heldout']['int8_exception']['rms']):.10g}",
                        str(row["cost"]["int8_total_bytes"]),
                        str(row["cost"].get("runtime_total_bytes", "")),
                        str(row.get("runtime_activity", {}).get("surface", "")),
                        str(bool(row.get("runtime_activity", {}).get("decode_active", False))).lower(),
                        str(bool(row.get("runtime_activity", {}).get("prefill_active", False))).lower(),
                        str(bool(row["candidate_ready"])).lower(),
                    ]
                )
                + "\n"
            )


def build_int8_exception_candidates(
    *,
    cfg: Dict[str, str],
    train_evidence_path: str,
    heldout_evidence_path: str,
    output_path: str,
    top_candidates: int,
    patch_tensors: Optional[str],
    source_contract_audit_path: Optional[str],
) -> Dict[str, Any]:
    if not source_contract_audit_path:
        raise RuntimeError("--build-int8-exception-candidates requires --source-contract-audit")
    with open(train_evidence_path, encoding="utf-8") as f:
        train_evidence = json.load(f)
    with open(heldout_evidence_path, encoding="utf-8") as f:
        heldout_evidence = json.load(f)
    if train_evidence.get("format") != EVIDENCE_FORMAT:
        raise ValueError(f"Not an HQQ self-calibration evidence file: {train_evidence_path}")
    if heldout_evidence.get("format") != EVIDENCE_FORMAT:
        raise ValueError(f"Not an HQQ self-calibration evidence file: {heldout_evidence_path}")

    model_path = os.path.expanduser(cfg.get("CFG_MODEL_PATH", train_evidence.get("model_path", "")))
    if not model_path:
        raise ValueError("Cannot determine model path for INT8 exception candidate build")
    heldout_model_path = os.path.expanduser(str(heldout_evidence.get("model_path", "")))
    if heldout_model_path and heldout_model_path != model_path:
        raise RuntimeError(f"Train/heldout evidence model mismatch: {model_path} vs {heldout_model_path}")

    source_profile = (
        train_evidence.get("cache", {}).get("source_cache_profile")
        or cfg.get("CFG_HQQ_CACHE_PROFILE")
        or HQQ_CACHE_PROFILE_BASELINE
    )
    source_profile = str(source_profile).strip().lower()
    heldout_profile = (
        heldout_evidence.get("cache", {}).get("source_cache_profile")
        or cfg.get("CFG_HQQ_CACHE_PROFILE")
        or HQQ_CACHE_PROFILE_BASELINE
    )
    if str(heldout_profile).strip().lower() != source_profile:
        raise RuntimeError(f"Train/heldout source cache profile mismatch: {source_profile} vs {heldout_profile}")

    manifest_probe = load_hqq_attention_manifest(model_path, source_profile)
    if manifest_probe is None:
        raise RuntimeError(f"No HQQ source manifest found for profile {source_profile}")
    manifest = require_complete_hqq_attention_manifest(
        model_path,
        source_profile,
        expected_nbits=4,
        expected_num_hidden_layers=int(manifest_probe.get("num_hidden_layers", 0)),
    )
    entries = _manifest_tensor_entries(manifest)
    tensor_names_by_layer = _manifest_tensor_names_by_layer(entries)
    cache_dir = hqq_attention_cache_dir(model_path, source_profile)
    source_contract_lookup = _source_contract_lookup_from_audit(source_contract_audit_path)
    source_contract_validations: Dict[Tuple[int, str], Dict[str, Any]] = {}
    allowed_tensors = _csv_filter(patch_tensors)

    train_trace_path = _evidence_trace_path(train_evidence_path, train_evidence)
    heldout_trace_path = _evidence_trace_path(heldout_evidence_path, heldout_evidence)
    _ensure_distinct_evidence_traces(
        train_evidence_path=train_evidence_path,
        train_trace_path=train_trace_path,
        heldout_evidence_path=heldout_evidence_path,
        heldout_trace_path=heldout_trace_path,
    )
    train_fields = _activation_fields_by_key(train_evidence)
    heldout_fields = _activation_fields_by_key(heldout_evidence)

    tensor_cache: Dict[Tuple[int, str], Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]] = {}
    saturation_cache: Dict[Tuple[int, str], Dict[int, Dict[str, float]]] = {}
    row_cache: Dict[Tuple[str, int, str], torch.Tensor] = {}
    candidates: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for field_key, train_field in train_fields.items():
        layer, activation_tensor = field_key
        heldout_field = heldout_fields.get(field_key)
        if heldout_field is None:
            skipped.append(
                {
                    "layer": layer,
                    "activation_tensor": activation_tensor,
                    "reason": "heldout evidence missing matching activation field",
                }
            )
            continue
        target_tensors = ACTIVATION_TARGET_TENSORS.get(activation_tensor, ())
        if not target_tensors:
            skipped.append(
                {
                    "layer": layer,
                    "activation_tensor": activation_tensor,
                    "reason": "no HQQ target tensor mapping for activation field",
                }
            )
            continue
        row_key = (layer, activation_tensor)
        train_rows = row_cache.setdefault(
            ("train", *row_key),
            _load_activation_rows(train_trace_path, layer=layer, tensor=activation_tensor),
        )
        heldout_rows = row_cache.setdefault(
            ("heldout", *row_key),
            _load_activation_rows(heldout_trace_path, layer=layer, tensor=activation_tensor),
        )
        heldout_groups = {int(group["group"]): group for group in heldout_field.get("groups", []) or []}
        for tensor_name in target_tensors:
            if allowed_tensors is not None and tensor_name not in allowed_tensors:
                continue
            tensor_key = (layer, tensor_name)
            entry = entries.get(tensor_key)
            if entry is None:
                skipped.append(
                    {
                        "layer": layer,
                        "activation_tensor": activation_tensor,
                        "target_tensor": tensor_name,
                        "reason": "HQQ tensor is not present in selected cache manifest",
                    }
                )
                continue
            try:
                if tensor_key not in tensor_cache:
                    entry = _attach_source_contract(entry, source_contract_lookup)
                    source_contract_validations[tensor_key] = _validate_source_contract_entry(
                        model_path=model_path,
                        cache_dir=cache_dir,
                        entry=entry,
                    )
                    source = _load_source_tensor(model_path, entry).to(torch.float32).contiguous()
                    hqq = _load_hqq_tensor(cache_dir, entry)
                    if list(source.shape) != [int(hqq["rows"]), int(hqq["cols"])]:
                        raise RuntimeError(
                            f"source/HQQ shape mismatch for layer={layer} tensor={tensor_name}: "
                            f"{list(source.shape)} vs {[hqq['rows'], hqq['cols']]}"
                        )
                    tensor_cache[tensor_key] = (source, hqq, entry)
                source, hqq, entry = tensor_cache[tensor_key]
                if train_rows.shape[1] != int(hqq["cols"]) or heldout_rows.shape[1] != int(hqq["cols"]):
                    raise RuntimeError(
                        f"activation/HQQ width mismatch for layer={layer} tensor={activation_tensor}: "
                        f"train={train_rows.shape[1]} heldout={heldout_rows.shape[1]} hqq={hqq['cols']}"
                    )
                if tensor_key not in saturation_cache:
                    saturation_cache[tensor_key] = _qvalue_saturation_by_group(cache_dir, entry)
                for train_group in train_field.get("groups", []) or []:
                    group_idx = int(train_group["group"])
                    heldout_group = heldout_groups.get(group_idx)
                    if heldout_group is None:
                        skipped.append(
                            {
                                "layer": layer,
                                "activation_tensor": activation_tensor,
                                "target_tensor": tensor_name,
                                "group": group_idx,
                                "reason": "heldout evidence missing matching group",
                            }
                        )
                        continue
                    group_size = int(hqq["group_size"])
                    start = group_idx * group_size
                    end = min(start + group_size, int(hqq["cols"]))
                    train_slice = train_rows[:, start:end].contiguous()
                    heldout_slice = heldout_rows[:, start:end].contiguous()
                    source_group = source[:, start:end].contiguous()
                    hqq_group = hqq["dequant"][:, start:end].contiguous()
                    hqq_delta = hqq_group - source_group
                    q_int8, int8_group, int8_scales = _quantize_int8_symmetric_rows(source_group)
                    int8_delta = int8_group - source_group
                    hqq_train_projection = _projection_metrics(train_slice, hqq_delta.T)
                    int8_train_projection = _projection_metrics(train_slice, int8_delta.T)
                    hqq_heldout_projection = _projection_metrics(heldout_slice, hqq_delta.T)
                    int8_heldout_projection = _projection_metrics(heldout_slice, int8_delta.T)
                    train_reduction = _projection_reduction(int8_train_projection, hqq_train_projection)
                    heldout_reduction = _projection_reduction(int8_heldout_projection, hqq_heldout_projection)
                    hqq_weight_error = _tensor_error_metrics(hqq_delta)
                    int8_weight_error = _tensor_error_metrics(int8_delta)
                    saturation = saturation_cache[tensor_key].get(group_idx, {})
                    train_rms_reduction = float(train_reduction["rms_reduction_fraction"])
                    heldout_rms_reduction = float(heldout_reduction["rms_reduction_fraction"])
                    stability_gap = abs(train_rms_reduction - heldout_rms_reduction)
                    risk_score = (
                        float(hqq_heldout_projection["rms"])
                        * (1.0 + float(saturation.get("saturation_fraction", 0.0)))
                        * (1.0 + float(hqq_weight_error["rms"]))
                    )
                    benefit_score = max(0.0, heldout_rms_reduction) * float(hqq_heldout_projection["rms"])
                    candidate_ready = (
                        train_rms_reduction > 0.0
                        and heldout_rms_reduction > 0.0
                        and float(int8_weight_error["rms"]) < float(hqq_weight_error["rms"])
                    )
                    q_extreme = int(((q_int8 == -127) | (q_int8 == 127)).sum().item())
                    runtime_activity = _int8_exception_runtime_activity(
                        layer=layer,
                        tensor_name=tensor_name,
                        tensor_names_by_layer=tensor_names_by_layer,
                    )
                    candidates.append(
                        {
                            "layer": layer,
                            "target_tensor": tensor_name,
                            "activation_tensor": activation_tensor,
                            "activation_policy": _activation_evidence_policy(layer, activation_tensor),
                            "group": group_idx,
                            "start_col": start,
                            "end_col": end,
                            "layout": {
                                "kind": "full_output_rows_by_hqq_column_group",
                                "row_count": int(source_group.shape[0]),
                                "column_count": int(source_group.shape[1]),
                                "group_size": group_size,
                                "tile_aligned_to": "current HQQ group columns",
                                "row_layout": "source rows are output rows; columns are input columns",
                            },
                            "risk_score": risk_score,
                            "benefit_score": benefit_score,
                            "candidate_ready": candidate_ready,
                            "runtime_activity": runtime_activity,
                            "not_runtime_ready_reason": (
                                "Candidate report only. Use --write-int8-exception-manifest to create an explicit "
                                "opt-in runtime manifest; no default enablement is allowed."
                            ),
                            "train_heldout_stability": {
                                "rms_reduction_gap": stability_gap,
                                "train_rows": int(train_rows.shape[0]),
                                "heldout_rows": int(heldout_rows.shape[0]),
                                "train_positions": train_field.get("positions", []),
                                "heldout_positions": heldout_field.get("positions", []),
                            },
                            "projection": {
                                "train": {
                                    "hqq": hqq_train_projection,
                                    "int8_exception": int8_train_projection,
                                    "reduction_vs_hqq": train_reduction,
                                },
                                "heldout": {
                                    "hqq": hqq_heldout_projection,
                                    "int8_exception": int8_heldout_projection,
                                    "reduction_vs_hqq": heldout_reduction,
                                },
                            },
                            "weight_error": {
                                "hqq": hqq_weight_error,
                                "int8_exception": int8_weight_error,
                                "int8_vs_hqq_rms_ratio": _safe_ratio(
                                    float(int8_weight_error["rms"]),
                                    float(hqq_weight_error["rms"]),
                                ),
                            },
                            "activation": {
                                "train": {
                                    "rows": int(train_field.get("rows", 0)),
                                    "unique_hashes": int(train_field.get("unique_hashes", 0)),
                                    "group": copy.deepcopy(train_group),
                                },
                                "heldout": {
                                    "rows": int(heldout_field.get("rows", 0)),
                                    "unique_hashes": int(heldout_field.get("unique_hashes", 0)),
                                    "group": copy.deepcopy(heldout_group),
                                },
                            },
                            "qvalue_saturation": {
                                "hqq_zero_fraction": float(saturation.get("zero_fraction", 0.0)),
                                "hqq_max_fraction": float(saturation.get("max_fraction", 0.0)),
                                "hqq_saturation_fraction": float(saturation.get("saturation_fraction", 0.0)),
                                "int8_extreme_fraction": q_extreme / max(1, int(q_int8.numel())),
                            },
                            "cost": {
                                "int8_weight_bytes": int(source_group.numel()),
                                "int8_scale_bytes": int(int8_scales.numel()) * 4,
                                "runtime_base_f32_bytes": int(source_group.numel()) * 4,
                                "metadata_bytes_estimate": int(source_group.shape[0]) * 4 * 4,
                                "int8_total_bytes": (
                                    int(source_group.numel())
                                    + int(int8_scales.numel()) * 4
                                    + int(source_group.shape[0]) * 4 * 4
                                ),
                                "runtime_total_bytes": (
                                    int(source_group.numel())
                                    + int(int8_scales.numel()) * 4
                                    + int(source_group.shape[0]) * 4 * 4
                                    + int(source_group.numel()) * 4
                                ),
                                "prefill_extra_fma_estimate": int(source_group.numel()),
                            },
                            "source_contract": copy.deepcopy(entry.get("source_contract")),
                            "source_contract_validation": source_contract_validations.get(tensor_key),
                            "artifact_hashes": {
                                "source_group_sha256": _tensor_hash(source_group),
                                "hqq_dequant_group_sha256": _tensor_hash(hqq_group),
                                "int8_dequant_group_sha256": _tensor_hash(int8_group),
                            },
                        }
                    )
            except Exception as exc:
                skipped.append(
                    {
                        "layer": layer,
                        "activation_tensor": activation_tensor,
                        "target_tensor": tensor_name,
                        "reason": str(exc),
                    }
                )

    candidates.sort(
        key=lambda item: (
            bool(item["candidate_ready"]),
            float(item["benefit_score"]),
            float(item["risk_score"]),
        ),
        reverse=True,
    )
    selected = candidates[:top_candidates]
    total_exception_bytes = sum(int(item["cost"]["int8_total_bytes"]) for item in selected)
    total_runtime_sidecar_bytes = sum(int(item["cost"]["runtime_total_bytes"]) for item in selected)
    runtime_surface_counts: Dict[str, int] = {}
    for item in candidates:
        surface = str(item.get("runtime_activity", {}).get("surface", "missing"))
        runtime_surface_counts[surface] = runtime_surface_counts.get(surface, 0) + 1
    payload = {
        "format": INT8_EXCEPTION_CANDIDATES_FORMAT,
        "format_version": INT8_EXCEPTION_CANDIDATES_FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "int8_exception_candidates_built",
        "complete": False,
        "loadable_hqq_manifest": False,
        "runtime_application_implemented": True,
        "not_loadable_reason": (
            "Artifact-only INT8 exception candidate report. It is not a loadable runtime manifest; "
            "runtime use requires an explicit --write-int8-exception-manifest output."
        ),
        "model_path": model_path,
        "source_train_evidence_path": train_evidence_path,
        "source_heldout_evidence_path": heldout_evidence_path,
        "source_train_trace_path": train_trace_path,
        "source_heldout_trace_path": heldout_trace_path,
        "cache": {
            "source_cache_profile": source_profile,
            "source_cache_dir": cache_dir,
            "target_cache_profile": HQQ_CACHE_PROFILE_SELFCAL_V1,
            "target_cache_dir": hqq_attention_cache_dir(model_path, HQQ_CACHE_PROFILE_SELFCAL_V1),
            "target_manifest_path": hqq_attention_manifest_path(model_path, HQQ_CACHE_PROFILE_SELFCAL_V1),
            "target_manifest_written": False,
            "int8_exception_candidate_path": output_path,
            "source_contract_audit_path": source_contract_audit_path,
            "source_contract_audit_sha256": _file_sha256(source_contract_audit_path),
        },
        "exception_model": {
            "purpose": "witness-free HQQ risk scoring for possible block-regular INT8 exceptions",
            "runtime_dependency": "none",
            "witness_dependency": "none",
            "main_int4_path": "unchanged regular HQQ INT4",
            "exception_semantics": "replace the selected HQQ block with per-output-row symmetric INT8 dequantized source weights",
            "selection_caveat": (
                "This report identifies intrinsic candidates only. It is not a promotion decision and "
                "does not create a loadable runtime manifest."
            ),
        },
        "summary": {
            "train_activation_field_count": len(train_fields),
            "heldout_activation_field_count": len(heldout_fields),
            "candidate_count": len(candidates),
            "candidate_ready_count": sum(1 for item in candidates if item.get("candidate_ready")),
            "top_candidate_count": len(selected),
            "top_candidate_total_exception_bytes": total_exception_bytes,
            "top_candidate_runtime_sidecar_bytes": total_runtime_sidecar_bytes,
            "runtime_surface_counts": runtime_surface_counts,
            "skipped_count": len(skipped),
            "source_contract": _source_contract_summary(source_contract_validations.values()),
        },
        "top_candidates": selected,
        "candidates": candidates,
        "skipped": skipped,
    }
    write_json(output_path, payload)
    _write_int8_exception_candidates_tsv(os.path.splitext(output_path)[0] + ".tsv", selected)
    latest_path = os.path.join(os.path.dirname(output_path), "int8_exception_candidates_latest.json")
    write_json(latest_path, payload)
    return payload


def _parse_int8_exception_groups(value: Optional[str]) -> Optional[set[Tuple[int, str, int]]]:
    if value is None or not value.strip():
        return None
    out: set[Tuple[int, str, int]] = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(
                "INT8 exception group filters must use layer:tensor:group entries, "
                f"got {item!r}"
            )
        try:
            layer = int(parts[0])
            group = int(parts[2])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric field in INT8 exception group filter {item!r}") from exc
        out.add((layer, parts[1], group))
    return out


def _int8_exception_group_filter_to_list(entries: Optional[set[Tuple[int, str, int]]]) -> Optional[List[str]]:
    if entries is None:
        return None
    return [f"{layer}:{tensor}:{group}" for layer, tensor, group in sorted(entries)]


def _int8_exception_candidate_key(candidate: Dict[str, Any]) -> Tuple[int, str, int]:
    return (int(candidate["layer"]), str(candidate["target_tensor"]), int(candidate["group"]))


def _int8_exception_candidate_cost(candidate: Dict[str, Any]) -> int:
    try:
        cost = int(candidate["cost"]["runtime_total_bytes"])
    except Exception as exc:
        raise RuntimeError(f"INT8 exception candidate lacks cost.runtime_total_bytes: {candidate}") from exc
    if cost <= 0:
        raise RuntimeError(f"INT8 exception candidate has non-positive runtime sidecar cost: {candidate}")
    return cost


def _int8_exception_candidate_activity(candidate: Dict[str, Any]) -> Dict[str, Any]:
    activity = candidate.get("runtime_activity")
    if not isinstance(activity, dict):
        raise RuntimeError(
            "INT8 exception candidate lacks runtime_activity metadata; "
            "rebuild the candidate report with the current hqq-self-calibrate."
        )
    surface = str(activity.get("surface") or "")
    if surface not in {"both", "decode_only", "prefill_only", "inactive"}:
        raise RuntimeError(f"INT8 exception candidate has invalid runtime_activity surface: {activity}")
    return activity


def _int8_exception_candidate_is_active(candidate: Dict[str, Any]) -> bool:
    return bool(_int8_exception_candidate_activity(candidate).get("active"))


def _int8_exception_candidate_decode_active(candidate: Dict[str, Any]) -> bool:
    return bool(_int8_exception_candidate_activity(candidate).get("decode_active"))


def _int8_exception_candidate_prefill_only(candidate: Dict[str, Any]) -> bool:
    activity = _int8_exception_candidate_activity(candidate)
    return bool(activity.get("prefill_active")) and not bool(activity.get("decode_active"))


def _int8_exception_selection_score(candidate: Dict[str, Any]) -> Tuple[float, float, float]:
    cost = _int8_exception_candidate_cost(candidate)
    benefit = float(candidate.get("benefit_score", 0.0))
    risk = float(candidate.get("risk_score", 0.0))
    return (benefit / float(cost), benefit, risk)


def _validate_int8_exception_ratio(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    ratio = float(value)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("--int8-exception-max-sidecar-ratio must be finite and > 0")
    return ratio


def _validate_int8_exception_byte_cap(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    cap = int(value)
    if cap <= 0:
        raise ValueError("--int8-exception-max-sidecar-bytes must be positive")
    return cap


def _hqq_manifest_tensor_bytes(manifest: Dict[str, Any]) -> int:
    totals = manifest.get("totals")
    if isinstance(totals, dict):
        for key in ("tensor_bytes", "total_cache_bytes", "cache_total_bytes", "total_bytes"):
            raw = totals.get(key)
            if raw is None:
                continue
            value = int(raw)
            if value > 0:
                return value
    for key in ("total_cache_bytes", "cache_total_bytes", "total_bytes"):
        raw = manifest.get(key)
        if raw is None:
            continue
        value = int(raw)
        if value > 0:
            return value
    return 0


def _int8_exception_budget_bytes(
    *,
    source_hqq_bytes: int,
    max_sidecar_ratio: Optional[float],
    max_sidecar_bytes: Optional[int],
) -> Optional[int]:
    ratio = _validate_int8_exception_ratio(max_sidecar_ratio)
    byte_cap = _validate_int8_exception_byte_cap(max_sidecar_bytes)
    budget: Optional[int] = byte_cap
    if ratio is not None:
        if source_hqq_bytes <= 0:
            raise RuntimeError(
                "--int8-exception-max-sidecar-ratio requires positive source HQQ tensor bytes "
                "in the HQQ manifest totals"
            )
        ratio_budget = int(math.floor(float(source_hqq_bytes) * ratio))
        if ratio_budget <= 0:
            raise RuntimeError(
                "--int8-exception-max-sidecar-ratio produced a zero-byte sidecar budget; "
                "increase the ratio or check HQQ manifest totals"
            )
        budget = ratio_budget if budget is None else min(budget, ratio_budget)
    return budget


def _select_int8_exception_candidates(
    candidate_report: Dict[str, Any],
    *,
    requested_groups: Optional[set[Tuple[int, str, int]]],
    top_groups: Optional[int],
    max_sidecar_bytes: Optional[int],
    max_decode_sidecar_bytes: Optional[int] = None,
    max_prefill_sidecar_bytes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if (
        requested_groups is None
        and top_groups is None
        and max_sidecar_bytes is None
        and max_decode_sidecar_bytes is None
        and max_prefill_sidecar_bytes is None
    ):
        raise RuntimeError(
            "--write-int8-exception-manifest requires either --int8-exception-groups or "
            "--int8-exception-top-groups or an explicit sidecar byte/ratio cap. "
            "Implicit single-group writes are not allowed."
        )
    if top_groups is not None and top_groups <= 0:
        raise ValueError("--int8-exception-top-groups must be positive")
    max_sidecar_bytes = _validate_int8_exception_byte_cap(max_sidecar_bytes)

    all_candidates = candidate_report.get("candidates")
    if not isinstance(all_candidates, list) or not all_candidates:
        all_candidates = candidate_report.get("top_candidates", [])
    ready_candidates = [item for item in all_candidates or [] if item.get("candidate_ready")]

    seen: set[Tuple[int, str, int]] = set()
    deduped: List[Dict[str, Any]] = []
    for item in ready_candidates:
        key = _int8_exception_candidate_key(item)
        if key in seen:
            raise RuntimeError(f"Duplicate INT8 exception candidate key in report: {key}")
        seen.add(key)
        if not _int8_exception_candidate_is_active(item):
            continue
        deduped.append(item)

    if requested_groups is not None:
        by_key = {_int8_exception_candidate_key(item): item for item in deduped}
        missing = sorted(requested_groups.difference(by_key))
        if missing:
            missing_s = ", ".join(f"{layer}:{tensor}:{group}" for layer, tensor, group in missing)
            raise RuntimeError(f"Requested INT8 exception groups are absent or not candidate_ready: {missing_s}")
        selected = [by_key[key] for key in sorted(requested_groups)]
        if max_sidecar_bytes is not None:
            selected_bytes = sum(_int8_exception_candidate_cost(item) for item in selected)
            if selected_bytes > max_sidecar_bytes:
                raise RuntimeError(
                    "Requested INT8 exception groups exceed sidecar byte budget: "
                    f"{selected_bytes} > {max_sidecar_bytes}"
                )
        return selected

    max_decode_sidecar_bytes = _validate_int8_exception_byte_cap(max_decode_sidecar_bytes)
    max_prefill_sidecar_bytes = _validate_int8_exception_byte_cap(max_prefill_sidecar_bytes)
    if max_decode_sidecar_bytes is not None or max_prefill_sidecar_bytes is not None:
        decode_budget = max_decode_sidecar_bytes if max_decode_sidecar_bytes is not None else 0
        prefill_budget = max_prefill_sidecar_bytes if max_prefill_sidecar_bytes is not None else 0
        selected = []
        selected_keys: set[Tuple[int, str, int]] = set()
        selected_total_bytes = 0

        def add_ranked(bucket: List[Dict[str, Any]], bucket_budget: int) -> int:
            nonlocal selected_total_bytes
            used = 0
            for candidate in bucket:
                if top_groups is not None and len(selected) >= int(top_groups):
                    break
                key = _int8_exception_candidate_key(candidate)
                if key in selected_keys:
                    continue
                cost = _int8_exception_candidate_cost(candidate)
                if used + cost > bucket_budget:
                    continue
                if max_sidecar_bytes is not None and selected_total_bytes + cost > max_sidecar_bytes:
                    continue
                selected.append(candidate)
                selected_keys.add(key)
                used += cost
                selected_total_bytes += cost
            return used

        ranked_decode = sorted(
            [item for item in deduped if _int8_exception_candidate_decode_active(item)],
            key=_int8_exception_selection_score,
            reverse=True,
        )
        ranked_prefill_only = sorted(
            [item for item in deduped if _int8_exception_candidate_prefill_only(item)],
            key=_int8_exception_selection_score,
            reverse=True,
        )
        add_ranked(ranked_decode, decode_budget)
        add_ranked(ranked_prefill_only, prefill_budget)
        return selected

    ranked = sorted(deduped, key=_int8_exception_selection_score, reverse=True)
    selected = []
    selected_bytes = 0
    for candidate in ranked:
        if top_groups is not None and len(selected) >= int(top_groups):
            break
        cost = _int8_exception_candidate_cost(candidate)
        if max_sidecar_bytes is not None and selected_bytes + cost > max_sidecar_bytes:
            continue
        selected.append(candidate)
        selected_bytes += cost
    return selected


def _int8_exception_artifact_file(layer: int, tensor_name: str, variant_name: str) -> str:
    safe_tensor = re.sub(r"[^A-Za-z0-9_.-]+", "_", tensor_name).strip("._")
    safe_variant = re.sub(r"[^A-Za-z0-9_.-]+", "_", variant_name).strip("._") or "unnamed"
    return f"layer_{layer:03d}_{safe_tensor}_{safe_variant}_int8_exception.safetensors"


def write_int8_exception_manifest(
    *,
    cfg: Dict[str, str],
    candidates_path: str,
    output_path: str,
    variant_name: str,
    int8_exception_groups: Optional[str],
    int8_exception_top_groups: Optional[int],
    int8_exception_max_sidecar_ratio: Optional[float],
    int8_exception_max_sidecar_bytes: Optional[int],
    int8_exception_decode_sidecar_ratio: Optional[float],
    int8_exception_decode_sidecar_bytes: Optional[int],
    int8_exception_prefill_sidecar_ratio: Optional[float],
    int8_exception_prefill_sidecar_bytes: Optional[int],
) -> Dict[str, Any]:
    with open(candidates_path, encoding="utf-8") as f:
        candidate_report = json.load(f)
    if candidate_report.get("format") != INT8_EXCEPTION_CANDIDATES_FORMAT:
        raise ValueError(f"Not an INT8 exception candidate report: {candidates_path}")

    model_path = os.path.expanduser(cfg.get("CFG_MODEL_PATH", candidate_report.get("model_path", "")))
    if not model_path:
        raise ValueError("Cannot determine model path for INT8 exception manifest write")
    if os.path.abspath(os.path.expanduser(str(candidate_report.get("model_path", "")))) != os.path.abspath(model_path):
        raise RuntimeError("INT8 exception candidate model_path does not match selected config")

    source_profile = (
        candidate_report.get("cache", {}).get("source_cache_profile")
        or cfg.get("CFG_HQQ_CACHE_PROFILE")
        or HQQ_CACHE_PROFILE_BASELINE
    )
    source_profile = str(source_profile).strip().lower()
    manifest_probe = load_hqq_attention_manifest(model_path, str(source_profile))
    if manifest_probe is None:
        raise RuntimeError(f"No HQQ source manifest found for profile {source_profile}")
    source_manifest = require_complete_hqq_attention_manifest(
        model_path,
        str(source_profile),
        expected_nbits=4,
        expected_num_hidden_layers=int(manifest_probe.get("num_hidden_layers", 0)),
    )
    source_manifest_path = hqq_attention_manifest_path(model_path, str(source_profile))
    source_cache_dir = hqq_attention_cache_dir(model_path, str(source_profile))
    entries = _manifest_tensor_entries(source_manifest)
    source_hqq_bytes = _hqq_manifest_tensor_bytes(source_manifest)
    budget_bytes = _int8_exception_budget_bytes(
        source_hqq_bytes=source_hqq_bytes,
        max_sidecar_ratio=int8_exception_max_sidecar_ratio,
        max_sidecar_bytes=int8_exception_max_sidecar_bytes,
    )
    decode_budget_bytes = _int8_exception_budget_bytes(
        source_hqq_bytes=source_hqq_bytes,
        max_sidecar_ratio=int8_exception_decode_sidecar_ratio,
        max_sidecar_bytes=int8_exception_decode_sidecar_bytes,
    )
    prefill_budget_bytes = _int8_exception_budget_bytes(
        source_hqq_bytes=source_hqq_bytes,
        max_sidecar_ratio=int8_exception_prefill_sidecar_ratio,
        max_sidecar_bytes=int8_exception_prefill_sidecar_bytes,
    )

    requested_groups = _parse_int8_exception_groups(int8_exception_groups)
    candidates = _select_int8_exception_candidates(
        candidate_report,
        requested_groups=requested_groups,
        top_groups=int8_exception_top_groups,
        max_sidecar_bytes=budget_bytes,
        max_decode_sidecar_bytes=decode_budget_bytes,
        max_prefill_sidecar_bytes=prefill_budget_bytes,
    )
    if not candidates:
        raise RuntimeError("No INT8 exception candidates selected for manifest write under the requested limits")
    selected_runtime_sidecar_bytes = sum(_int8_exception_candidate_cost(item) for item in candidates)

    target_sidecar_dir = sidecar_variant_dir(
        model_path,
        HQQ_CACHE_PROFILE_SELFCAL_V1,
        variant_name,
        "int8_exception",
    )
    os.makedirs(target_sidecar_dir, exist_ok=True)
    manifest_path = os.path.join(target_sidecar_dir, "sidecar_manifest.json")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    selected_by_tensor: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for candidate in candidates:
        layout = candidate.get("layout") or {}
        if str(layout.get("kind")) != "full_output_rows_by_hqq_column_group":
            raise RuntimeError(f"Unsupported INT8 exception candidate layout: {layout}")
        selected_by_tensor.setdefault(
            (int(candidate["layer"]), str(candidate["target_tensor"])),
            [],
        ).append(candidate)

    artifact_summaries = []
    validations: Dict[Tuple[int, str], Dict[str, Any]] = {}
    total_weight_bytes = 0
    total_metadata_bytes = 0
    total_fmas = 0
    total_row_groups = 0

    for tensor_key in sorted(selected_by_tensor):
        layer, tensor_name = tensor_key
        entry = entries.get(tensor_key)
        if entry is None:
            raise RuntimeError(f"INT8 exception candidate references tensor absent from HQQ manifest: {tensor_key}")
        contract = selected_by_tensor[tensor_key][0].get("source_contract")
        if not isinstance(contract, dict):
            raise RuntimeError(f"INT8 exception candidate lacks source_contract: {tensor_key}")
        for candidate in selected_by_tensor[tensor_key][1:]:
            other_contract = candidate.get("source_contract")
            if not isinstance(other_contract, dict):
                raise RuntimeError(f"INT8 exception candidate lacks source_contract: {tensor_key}")
            for contract_key in (
                "artifact_sha256",
                "reconstructed_source_sha256",
                "target_file",
                "target_tensor",
                "layer",
            ):
                if other_contract.get(contract_key) != contract.get(contract_key):
                    raise RuntimeError(
                        f"INT8 exception candidates for {tensor_key} have mismatched source_contract "
                        f"field {contract_key}"
                    )
        entry_with_contract = copy.deepcopy(entry)
        entry_with_contract["source_contract"] = copy.deepcopy(contract)
        validations[tensor_key] = _validate_source_contract_entry(
            model_path=model_path,
            cache_dir=source_cache_dir,
            entry=entry_with_contract,
        )
        source = _load_source_tensor(model_path, entry_with_contract).to(torch.float32).contiguous()
        hqq = _load_hqq_tensor(source_cache_dir, entry_with_contract)
        if list(source.shape) != [int(hqq["rows"]), int(hqq["cols"])]:
            raise RuntimeError(
                f"source/HQQ shape mismatch for INT8 exception tensor {tensor_key}: "
                f"{list(source.shape)} vs {[hqq['rows'], hqq['cols']]}"
            )

        rows = int(hqq["rows"])
        cols = int(hqq["cols"])
        group_size = int(hqq["group_size"])
        artifact_rows: List[Dict[str, Any]] = []
        q_rows = []
        scale_rows = []
        max_width = 0
        for candidate in sorted(selected_by_tensor[tensor_key], key=lambda item: int(item["group"])):
            group = int(candidate["group"])
            start = group * group_size
            end = min(start + group_size, cols)
            if start < 0 or start >= cols or end <= start:
                raise RuntimeError(f"INT8 exception group out of bounds: {candidate}")
            if int(candidate["start_col"]) != start or int(candidate["end_col"]) != end:
                raise RuntimeError(f"INT8 exception candidate bounds do not match HQQ artifact: {candidate}")
            source_group = source[:, start:end].contiguous()
            hqq_group = hqq["dequant"][:, start:end].contiguous()
            if _tensor_hash(source_group) != candidate.get("artifact_hashes", {}).get("source_group_sha256"):
                raise RuntimeError(f"INT8 exception source group sha256 mismatch: {candidate}")
            if _tensor_hash(hqq_group) != candidate.get("artifact_hashes", {}).get("hqq_dequant_group_sha256"):
                raise RuntimeError(f"INT8 exception HQQ group sha256 mismatch: {candidate}")
            q, dequant, scales = _quantize_int8_symmetric_rows(source_group)
            if _tensor_hash(dequant) != candidate.get("artifact_hashes", {}).get("int8_dequant_group_sha256"):
                raise RuntimeError(f"INT8 exception dequant group sha256 mismatch: {candidate}")
            q_rows.append(q)
            scale_rows.append(scales)
            max_width = max(max_width, int(q.shape[1]))
            artifact_rows.append(
                {
                    "layer": layer,
                    "tensor": tensor_name,
                    "group": group,
                    "start_col": start,
                    "end_col": end,
                    "width": end - start,
                    "output_row_start": 0,
                    "output_row_end": rows,
                    "row_count": rows,
                    "source_group_sha256": candidate["artifact_hashes"]["source_group_sha256"],
                    "hqq_dequant_group_sha256": candidate["artifact_hashes"]["hqq_dequant_group_sha256"],
                    "int8_dequant_group_sha256": candidate["artifact_hashes"]["int8_dequant_group_sha256"],
                    "risk_score": float(candidate["risk_score"]),
                    "benefit_score": float(candidate["benefit_score"]),
                    "projection": copy.deepcopy(candidate.get("projection", {})),
                    "cost": copy.deepcopy(candidate.get("cost", {})),
                }
            )

        padded_q_rows = []
        for q in q_rows:
            if int(q.shape[1]) == max_width:
                padded_q_rows.append(q)
                continue
            padded = torch.zeros((int(q.shape[0]), max_width), dtype=torch.int8)
            padded[:, : int(q.shape[1])] = q
            padded_q_rows.append(padded)
        q_payload = torch.cat(padded_q_rows, dim=0).contiguous()
        scales_payload = torch.cat(scale_rows, dim=0).contiguous()
        output_rows = torch.arange(rows * len(artifact_rows), dtype=torch.int32) % rows
        groups = torch.cat(
            [
                torch.full((rows,), int(item["group"]), dtype=torch.int32)
                for item in artifact_rows
            ],
            dim=0,
        ).contiguous()
        start_cols = torch.cat(
            [
                torch.full((rows,), int(item["start_col"]), dtype=torch.int32)
                for item in artifact_rows
            ],
            dim=0,
        ).contiguous()
        widths = torch.cat(
            [
                torch.full((rows,), int(item["width"]), dtype=torch.int32)
                for item in artifact_rows
            ],
            dim=0,
        ).contiguous()
        rel_file = _int8_exception_artifact_file(layer, tensor_name, variant_name)
        path = os.path.join(target_sidecar_dir, rel_file)
        tmp_path = path + ".tmp"
        save_file(
            {
                "exception_qint8": q_payload,
                "scales": scales_payload,
                "output_rows": output_rows.contiguous(),
                "groups": groups,
                "start_cols": start_cols,
                "widths": widths,
            },
            tmp_path,
            metadata={
                "format": SIDECAR_MANIFEST_FORMAT,
                "mode": "int8_exception",
                "semantics": "replace_hqq_block_with_int8_dequant_source",
                "variant_name": variant_name,
                "layer": str(layer),
                "tensor": tensor_name,
            },
        )
        os.replace(tmp_path, path)
        weight_bytes = int(q_payload.numel())
        metadata_bytes = int(scales_payload.numel() * 4 + output_rows.numel() * 4 * 4)
        fmas = sum(int(item["cost"].get("prefill_extra_fma_estimate", 0)) for item in artifact_rows)
        total_weight_bytes += weight_bytes
        total_metadata_bytes += metadata_bytes
        total_fmas += fmas
        total_row_groups += int(output_rows.numel())
        artifact_summaries.append(
            {
                "layer": layer,
                "tensor": tensor_name,
                "file": rel_file,
                "sha256": _file_sha256(path),
                "hqq_artifact_file": entry_with_contract.get("file"),
                "hqq_artifact_sha256": contract.get("artifact_sha256"),
                "row_group_count": int(output_rows.numel()),
                "exception_group_count": len(artifact_rows),
                "weight_bytes": weight_bytes,
                "metadata_bytes": metadata_bytes,
                "decode_fma_per_token": fmas,
                "source_contract": copy.deepcopy(contract),
                "source_contract_validation": validations[tensor_key],
                "entries": artifact_rows,
            }
        )

    manifest = {
        "format": SIDECAR_MANIFEST_FORMAT,
        "format_version": SIDECAR_MANIFEST_FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "experimental_int8_exception_manifest",
        "complete": True,
        "loadable_hqq_manifest": False,
        "not_loadable_reason": "This is an explicit runtime sidecar manifest, not a calibrated HQQ cache manifest.",
        "model_path": model_path,
        "variant_name": variant_name,
        "sidecar_mode": "int8_exception",
        "exception_semantics": "replace selected HQQ-dequantized blocks with per-output-row symmetric INT8 dequantized source values on active prefill/decode sidecar surfaces",
        "main_int4_path": "unchanged regular HQQ INT4",
        "runtime_application_implemented": True,
        "runtime_ready": True,
        "decode_application_implemented": True,
        "source": {
            "cache_profile": source_profile,
            "baseline_manifest_path": source_manifest_path,
            "baseline_manifest_sha256": _file_sha256(source_manifest_path),
            "int8_exception_candidates_path": candidates_path,
            "int8_exception_candidates_sha256": _file_sha256(candidates_path),
        },
        "cache": {
            "target_cache_profile": HQQ_CACHE_PROFILE_SELFCAL_V1,
            "sidecar_dir": target_sidecar_dir,
            "sidecar_manifest_path": manifest_path,
        },
        "summary": {
            "artifact_count": len(artifact_summaries),
            "exception_group_count": sum(int(item["exception_group_count"]) for item in artifact_summaries),
            "row_group_count": total_row_groups,
            "sidecar_weight_bytes": total_weight_bytes,
            "sidecar_metadata_bytes": total_metadata_bytes,
            "sidecar_total_bytes": total_weight_bytes + total_metadata_bytes,
            "runtime_sidecar_total_bytes_estimate": selected_runtime_sidecar_bytes,
            "source_hqq_tensor_bytes": source_hqq_bytes,
            "sidecar_ratio_to_source_hqq": _safe_ratio(
                float(total_weight_bytes + total_metadata_bytes),
                float(source_hqq_bytes),
            )
            if source_hqq_bytes > 0
            else None,
            "runtime_sidecar_ratio_to_source_hqq": _safe_ratio(
                float(selected_runtime_sidecar_bytes),
                float(source_hqq_bytes),
            )
            if source_hqq_bytes > 0
            else None,
            "decode_fma_per_token": total_fmas,
            "selected_runtime_sidecar_bytes": selected_runtime_sidecar_bytes,
            "selected_decode_attached_groups": sum(
                1 for item in candidates if _int8_exception_candidate_decode_active(item)
            ),
            "selected_prefill_only_groups": sum(
                1 for item in candidates if _int8_exception_candidate_prefill_only(item)
            ),
            "source_contract": _source_contract_summary(validations.values()),
        },
        "selection": {
            "selected_groups": [
                {"layer": int(item["layer"]), "tensor": str(item["target_tensor"]), "group": int(item["group"])}
                for item in candidates
            ],
            "group_filter": _int8_exception_group_filter_to_list(requested_groups),
            "top_groups": int8_exception_top_groups,
            "max_sidecar_ratio": int8_exception_max_sidecar_ratio,
            "max_sidecar_bytes": int8_exception_max_sidecar_bytes,
            "decode_sidecar_ratio": int8_exception_decode_sidecar_ratio,
            "decode_sidecar_bytes": int8_exception_decode_sidecar_bytes,
            "prefill_sidecar_ratio": int8_exception_prefill_sidecar_ratio,
            "prefill_sidecar_bytes": int8_exception_prefill_sidecar_bytes,
            "effective_sidecar_budget_bytes": budget_bytes,
            "effective_decode_sidecar_budget_bytes": decode_budget_bytes,
            "effective_prefill_sidecar_budget_bytes": prefill_budget_bytes,
            "selection_score": "benefit_score_per_sidecar_byte_then_benefit_then_risk",
            "candidate_ready_count": sum(
                1
                for item in (candidate_report.get("candidates") or candidate_report.get("top_candidates") or [])
                if item.get("candidate_ready")
            ),
        },
        "artifacts": artifact_summaries,
        "switchability": {
            "fallback_to_baseline": False,
            "runtime_enabled_by_default": False,
            "selection": "loaded only when explicitly referenced by hqq_sidecar_manifest",
        },
    }
    write_json(manifest_path, manifest)
    report = {
        "format": INT8_EXCEPTION_WRITE_FORMAT,
        "format_version": INT8_EXCEPTION_WRITE_FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "int8_exception_manifest_written",
        "complete": True,
        "loadable_hqq_manifest": False,
        "runtime_ready": True,
        "model_path": model_path,
        "variant_name": variant_name,
        "cache": manifest["cache"],
        "source": manifest["source"],
        "summary": manifest["summary"],
        "sidecar_manifest": manifest,
    }
    write_json(output_path, report)
    latest_path = os.path.join(os.path.dirname(output_path), "int8_exception_write_latest.json")
    write_json(latest_path, report)
    return report


def _load_activation_rows(trace_path: str, *, layer: int, tensor: str) -> torch.Tensor:
    rows: List[List[float]] = []
    width: Optional[int] = None
    with open(trace_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = _parse_input_row_full_line(line)
            if not parsed:
                continue
            if int(parsed["layer"]) != layer or parsed["tensor"] != tensor:
                continue
            row_width = int(parsed["width"])
            values = _bf16_hex_to_floats(parsed["hex"], row_width)
            if width is None:
                width = row_width
            elif width != row_width:
                raise RuntimeError(
                    f"Activation width changed for layer={layer} tensor={tensor}: {row_width} vs {width}"
                )
            rows.append(values)
    if not rows:
        raise RuntimeError(f"No activation rows found in {trace_path} for layer={layer} tensor={tensor}")
    return torch.tensor(rows, dtype=torch.float32)


def _quantize_affine(source: torch.Tensor, scale: float, zero: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    safe_scale = max(float(scale), 1e-8)
    raw = source.to(torch.float32) / safe_scale + float(zero)
    q = torch.round(raw).clamp(0, 15).to(torch.uint8)
    dequant = (q.to(torch.float32) - float(zero)) * safe_scale
    return q, dequant, raw


def _minmax_params(source: torch.Tensor) -> Tuple[float, float]:
    src = source.to(torch.float32)
    minv = float(src.min().item())
    maxv = float(src.max().item())
    scale = max((maxv - minv) / 15.0, 1e-8)
    zero = min(max(-minv / scale, 0.0), 15.0)
    return scale, zero


def _symmetric_params(source: torch.Tensor) -> Tuple[float, float]:
    amax = float(source.to(torch.float32).abs().max().item())
    return max((2.0 * amax) / 15.0, 1e-8), 7.5


def _solve_fixed_zero(source: torch.Tensor, weights: torch.Tensor, zero: float, scale_seed: float, min_scale: float = 1e-8) -> Tuple[torch.Tensor, float]:
    src = source.to(torch.float32).contiguous()
    w = weights.to(torch.float32).contiguous()
    zero_value = float(zero)
    scale = max(float(scale_seed), float(min_scale), 1e-8)
    for _ in range(8):
        q = torch.round(src / scale + zero_value).clamp(0, 15)
        centered = q - zero_value
        denom = float((w * centered * centered).sum().item())
        numer = float((w * src * centered).sum().item())
        if denom > 1e-12:
            scale = max(numer / denom, float(min_scale), 1e-8)
    q = torch.round(src / scale + zero_value).clamp(0, 15).to(torch.uint8)
    return q, scale


def _weighted_affine_params(
    source: torch.Tensor,
    weights: torch.Tensor,
    *,
    grid_steps: int,
    local_grid_steps: int,
    min_scale: float = 1e-8,
) -> Tuple[float, float, torch.Tensor]:
    src = source.to(torch.float32).contiguous()
    w = weights.to(torch.float32).contiguous()
    if float(w.sum().item()) <= 1e-20:
        w = torch.ones_like(src, dtype=torch.float32)
    range_scale, _ = _minmax_params(src)
    abs_scale, _ = _symmetric_params(src)
    best_score = float("inf")
    best_q = torch.zeros_like(src, dtype=torch.uint8)
    best_scale = range_scale
    best_zero = 0.0

    def consider(zero_values: torch.Tensor, scale_seed: float) -> None:
        nonlocal best_score, best_q, best_scale, best_zero
        for zero_tensor in zero_values:
            zero = float(zero_tensor.item())
            q, scale = _solve_fixed_zero(src, w, zero, scale_seed, min_scale=min_scale)
            dequant = (q.to(torch.float32) - zero) * max(scale, 1e-8)
            diff = dequant - src
            score = float(torch.sqrt(torch.mean(w * diff * diff)).item())
            if score < best_score:
                best_score = score
                best_q = q
                best_scale = scale
                best_zero = zero

    global_grid = torch.linspace(0.0, 15.0, steps=max(3, grid_steps), dtype=torch.float32)
    consider(global_grid, range_scale)
    consider(global_grid, abs_scale)
    local_min = max(0.0, best_zero - 0.5)
    local_max = min(15.0, best_zero + 0.5)
    local_grid = torch.linspace(local_min, local_max, steps=max(3, local_grid_steps), dtype=torch.float32)
    consider(local_grid, best_scale)
    return best_scale, best_zero, best_q


def _protected_scale_floor(source: torch.Tensor, protected_indices: torch.Tensor, zero: float) -> float:
    src = source.to(torch.float32)
    min_scale = 1e-8
    for idx in protected_indices.to(torch.long).tolist():
        value = float(src[idx].item())
        if value > 0.0:
            denom = 15.0 - float(zero)
            if denom <= 1e-8:
                return float("inf")
            min_scale = max(min_scale, value / denom)
        elif value < 0.0:
            denom = float(zero)
            if denom <= 1e-8:
                return float("inf")
            min_scale = max(min_scale, -value / denom)
    return min_scale


def _clip_free_weighted_params(
    source: torch.Tensor,
    input_rows: torch.Tensor,
    *,
    grid_steps: int,
    local_grid_steps: int,
) -> Tuple[float, float, torch.Tensor, Dict[str, Any]]:
    src = source.to(torch.float32).contiguous()
    inp = input_rows.to(torch.float32).contiguous()
    weights = (inp * inp).mean(dim=0)
    abs_input = inp.abs().max(dim=0).values
    top_k = min(8, src.numel())
    threshold = float(abs_input.max().item()) * 0.25 if abs_input.numel() else 0.0
    protected = torch.nonzero(abs_input >= threshold, as_tuple=False).flatten()
    if protected.numel() < top_k:
        protected = torch.unique(torch.cat([protected, torch.topk(abs_input, k=top_k).indices])).to(torch.long)
    range_scale, _ = _minmax_params(src)
    abs_scale, _ = _symmetric_params(src)
    best_score = float("inf")
    best_scale = range_scale
    best_zero = 0.0
    best_q = torch.zeros_like(src, dtype=torch.uint8)
    best_floor = 1e-8
    seen = 0

    def consider(zero_values: torch.Tensor, scale_seed: float) -> None:
        nonlocal best_score, best_scale, best_zero, best_q, best_floor, seen
        for zero_tensor in zero_values:
            zero = float(zero_tensor.item())
            floor = _protected_scale_floor(src, protected, zero)
            if not math.isfinite(floor):
                continue
            q, scale = _solve_fixed_zero(src, weights, zero, scale_seed, min_scale=floor)
            raw = src / max(scale, 1e-8) + zero
            if bool((raw.index_select(0, protected) < -1e-6).any().item()):
                continue
            if bool((raw.index_select(0, protected) > 15.0 + 1e-6).any().item()):
                continue
            dequant = (q.to(torch.float32) - zero) * max(scale, 1e-8)
            projected = inp @ (dequant - src)
            score = float(torch.sqrt(torch.mean(projected * projected)).item())
            seen += 1
            if score < best_score:
                best_score = score
                best_scale = scale
                best_zero = zero
                best_q = q
                best_floor = floor

    grid = torch.linspace(0.0, 15.0, steps=max(3, grid_steps), dtype=torch.float32)
    consider(grid, range_scale)
    consider(grid, abs_scale)
    if seen:
        local_grid = torch.linspace(max(0.0, best_zero - 0.5), min(15.0, best_zero + 0.5), steps=max(3, local_grid_steps))
        consider(local_grid, best_scale)
    else:
        best_scale, best_zero = _minmax_params(src)
        best_q, _, _ = _quantize_affine(src, best_scale, best_zero)
        best_floor = best_scale
    return best_scale, best_zero, best_q, {
        "protected_local_cols": [int(v) for v in protected.tolist()],
        "protected_count": int(protected.numel()),
        "input_abs_threshold": threshold,
        "scale_floor": float(best_floor),
        "candidate_count": int(seen),
    }


def _projection_metrics(input_rows: torch.Tensor, diff_group: torch.Tensor) -> Dict[str, Any]:
    projected = input_rows.to(torch.float32) @ diff_group.to(torch.float32)
    abs_projected = projected.abs()
    sq = projected * projected
    return {
        "samples": int(input_rows.shape[0]),
        "rms": float(torch.sqrt(torch.mean(sq)).item()) if projected.numel() else 0.0,
        "mean_abs": float(abs_projected.mean().item()) if projected.numel() else 0.0,
        "max_abs": float(abs_projected.max().item()) if projected.numel() else 0.0,
        "signed_sum": float(projected.sum().item()) if projected.numel() else 0.0,
        "abs_sum": float(abs_projected.sum().item()) if projected.numel() else 0.0,
    }


def _candidate_metrics(
    name: str,
    source: torch.Tensor,
    input_rows: torch.Tensor,
    *,
    scale: float,
    zero: float,
    q: Optional[torch.Tensor] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if q is None:
        q, dequant, raw = _quantize_affine(source, scale, zero)
    else:
        q = q.to(torch.uint8).contiguous()
        raw = source.to(torch.float32) / max(float(scale), 1e-8) + float(zero)
        dequant = (q.to(torch.float32) - float(zero)) * max(float(scale), 1e-8)
    diff = dequant - source.to(torch.float32)
    weight_sq = diff * diff
    hist = torch.bincount(q.to(torch.int64), minlength=16)[:16]
    payload = {
        "name": name,
        "scale": float(scale),
        "zero": float(zero),
        "weight_error_rms": float(torch.sqrt(torch.mean(weight_sq)).item()) if diff.numel() else 0.0,
        "weight_error_max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "projection": _projection_metrics(input_rows, diff),
        "qvalue_histogram": [int(v) for v in hist.tolist()],
        "qvalue_saturation_zero_count": int((q == 0).sum().item()),
        "qvalue_saturation_max_count": int((q == 15).sum().item()),
        "clip_low_count": int((raw < 0.0).sum().item()),
        "clip_high_count": int((raw > 15.0).sum().item()),
    }
    if extra:
        payload.update(extra)
    return payload


def _aggregate_eval_candidates(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals: Dict[str, Dict[str, float]] = {}
    for evaluation in evaluations:
        for row in evaluation.get("selected_output_rows", []):
            for candidate in row.get("candidates", []):
                name = str(candidate["name"])
                slot = totals.setdefault(
                    name,
                    {
                        "row_count": 0.0,
                        "projection_abs_sum": 0.0,
                        "projection_signed_sum": 0.0,
                        "projection_rms_sum": 0.0,
                        "projection_max_abs": 0.0,
                        "weight_error_rms_sum": 0.0,
                    },
                )
                projection = candidate["projection"]
                slot["row_count"] += 1.0
                slot["projection_abs_sum"] += float(projection["abs_sum"])
                slot["projection_signed_sum"] += float(projection["signed_sum"])
                slot["projection_rms_sum"] += float(projection["rms"])
                slot["projection_max_abs"] = max(slot["projection_max_abs"], float(projection["max_abs"]))
                slot["weight_error_rms_sum"] += float(candidate["weight_error_rms"])
    out: Dict[str, Any] = {}
    for name, slot in totals.items():
        count = max(slot["row_count"], 1.0)
        out[name] = {
            "row_count": int(slot["row_count"]),
            "projection_abs_sum": slot["projection_abs_sum"],
            "projection_signed_sum": slot["projection_signed_sum"],
            "projection_rms_mean": slot["projection_rms_sum"] / count,
            "projection_max_abs": slot["projection_max_abs"],
            "weight_error_rms_mean": slot["weight_error_rms_sum"] / count,
        }
    return out


def _improvement_vs_cached(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    cached = aggregate.get("cached_hqq")
    if not cached:
        return {}
    cached_abs = max(float(cached.get("projection_abs_sum", 0.0)), 1e-30)
    cached_rms = max(float(cached.get("projection_rms_mean", 0.0)), 1e-30)
    out: Dict[str, Any] = {}
    for name, item in aggregate.items():
        if name == "cached_hqq":
            continue
        abs_ratio = float(item.get("projection_abs_sum", 0.0)) / cached_abs
        rms_ratio = float(item.get("projection_rms_mean", 0.0)) / cached_rms
        out[name] = {
            "projection_abs_sum_ratio": abs_ratio,
            "projection_abs_sum_reduction_fraction": 1.0 - abs_ratio,
            "projection_rms_mean_ratio": rms_ratio,
            "projection_rms_mean_reduction_fraction": 1.0 - rms_ratio,
            "weight_error_rms_mean_ratio": float(item.get("weight_error_rms_mean", 0.0))
            / max(float(cached.get("weight_error_rms_mean", 0.0)), 1e-30),
        }
    return out


def _reduction_vs_cached(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    cached = aggregate.get("cached_hqq")
    if not cached:
        return {}
    cached_abs = max(float(cached.get("projection_abs_sum", 0.0)), 1e-30)
    cached_rms = max(float(cached.get("projection_rms_mean", 0.0)), 1e-30)
    out: Dict[str, Any] = {}
    for name, item in aggregate.items():
        if name == "cached_hqq":
            continue
        abs_ratio = float(item.get("projection_abs_sum", 0.0)) / cached_abs
        rms_ratio = float(item.get("projection_rms_mean", 0.0)) / cached_rms
        out[name] = {
            "projection_abs_sum_ratio": abs_ratio,
            "projection_abs_sum_reduction_fraction": 1.0 - abs_ratio,
            "projection_rms_mean_ratio": rms_ratio,
            "projection_rms_mean_reduction_fraction": 1.0 - rms_ratio,
        }
    return out


def _sidecar_mode_set(value: str) -> set:
    modes = {part.strip() for part in value.split(",") if part.strip()}
    if not modes or modes == {"all"}:
        return {"exact_bf16", "int8_symmetric"}
    valid = {"exact_bf16", "int8_symmetric"}
    invalid = sorted(modes - valid)
    if invalid:
        raise ValueError(f"Unsupported sidecar modes: {', '.join(invalid)}")
    return modes


def _sidecar_artifact_mode(value: str) -> str:
    mode = value.strip()
    if mode not in {"exact_bf16", "int8_symmetric"}:
        raise ValueError(f"Unsupported sidecar artifact mode: {mode}")
    return mode


def _quantize_int8_symmetric(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    src = values.to(torch.float32).contiguous()
    max_abs = float(src.abs().max().item()) if src.numel() else 0.0
    scale = max(max_abs / 127.0, 1e-12)
    q = torch.clamp(torch.round(src / scale), -127, 127).to(torch.int8)
    dequant = q.to(torch.float32) * scale
    return q, dequant, scale


def _quantize_int8_symmetric_rows(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src = values.to(torch.float32).contiguous()
    if src.ndim != 2:
        raise RuntimeError(f"INT8 row quantization expects a 2D tensor, got shape={list(src.shape)}")
    max_abs = src.abs().amax(dim=1, keepdim=True) if src.numel() else torch.zeros((src.shape[0], 1))
    scales = torch.clamp(max_abs / 127.0, min=1e-12)
    q = torch.clamp(torch.round(src / scales), -127, 127).to(torch.int8)
    dequant = q.to(torch.float32) * scales
    return q.contiguous(), dequant.contiguous(), scales.flatten().to(torch.float32).contiguous()


def _sidecar_projection_metrics(
    input_rows: torch.Tensor,
    cached_diff: torch.Tensor,
    correction: torch.Tensor,
) -> Dict[str, Any]:
    final_diff = cached_diff.to(torch.float32) + correction.to(torch.float32)
    return _projection_metrics(input_rows, final_diff)


def _aggregate_sidecar_rows(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals: Dict[str, Dict[str, float]] = {}
    for evaluation in evaluations:
        for row in evaluation.get("selected_output_rows", []):
            for candidate in row.get("sidecar_modes", []):
                name = str(candidate["name"])
                slot = totals.setdefault(
                    name,
                    {
                        "row_count": 0.0,
                        "projection_abs_sum": 0.0,
                        "projection_signed_sum": 0.0,
                        "projection_rms_sum": 0.0,
                        "projection_max_abs": 0.0,
                        "sidecar_weight_bytes": 0.0,
                        "sidecar_metadata_bytes": 0.0,
                        "decode_fma_per_token": 0.0,
                    },
                )
                projection = candidate["projection"]
                slot["row_count"] += 1.0
                slot["projection_abs_sum"] += float(projection["abs_sum"])
                slot["projection_signed_sum"] += float(projection["signed_sum"])
                slot["projection_rms_sum"] += float(projection["rms"])
                slot["projection_max_abs"] = max(slot["projection_max_abs"], float(projection["max_abs"]))
                slot["sidecar_weight_bytes"] += float(candidate.get("sidecar_weight_bytes", 0))
                slot["sidecar_metadata_bytes"] += float(candidate.get("sidecar_metadata_bytes", 0))
                slot["decode_fma_per_token"] += float(candidate.get("decode_fma_per_token", 0))
    out: Dict[str, Any] = {}
    for name, slot in totals.items():
        count = max(slot["row_count"], 1.0)
        out[name] = {
            "row_group_count": int(slot["row_count"]),
            "projection_abs_sum": slot["projection_abs_sum"],
            "projection_signed_sum": slot["projection_signed_sum"],
            "projection_rms_mean": slot["projection_rms_sum"] / count,
            "projection_max_abs": slot["projection_max_abs"],
            "sidecar_weight_bytes": int(slot["sidecar_weight_bytes"]),
            "sidecar_metadata_bytes": int(slot["sidecar_metadata_bytes"]),
            "sidecar_total_bytes": int(slot["sidecar_weight_bytes"] + slot["sidecar_metadata_bytes"]),
            "decode_fma_per_token": int(slot["decode_fma_per_token"]),
        }
    return out


def simulate_sidecars(
    *,
    cfg: Dict[str, str],
    refit_eval_path: str,
    trace_path: str,
    output_path: str,
    sidecar_modes: str,
    patch_tensors: Optional[str],
    patch_top_groups: Optional[int],
    source_contract_audit_path: Optional[str] = None,
) -> Dict[str, Any]:
    with open(refit_eval_path, encoding="utf-8") as f:
        refit_eval = json.load(f)
    if refit_eval.get("format") != REFIT_EVAL_FORMAT:
        raise ValueError(f"Not an HQQ self-calibration refit-evaluation file: {refit_eval_path}")

    model_path = os.path.expanduser(cfg.get("CFG_MODEL_PATH", refit_eval.get("model_path", "")))
    if not model_path:
        raise ValueError("Cannot determine model path for sidecar simulation")
    source_profile = refit_eval.get("cache", {}).get("source_cache_profile") or HQQ_CACHE_PROFILE_BASELINE
    manifest_probe = load_hqq_attention_manifest(model_path, str(source_profile))
    if manifest_probe is None:
        raise RuntimeError(f"No HQQ source manifest found for profile {source_profile}")
    manifest = require_complete_hqq_attention_manifest(
        model_path,
        str(source_profile),
        expected_nbits=4,
        expected_num_hidden_layers=int(manifest_probe.get("num_hidden_layers", 0)),
    )
    entries = _manifest_tensor_entries(manifest)
    cache_dir = hqq_attention_cache_dir(model_path, str(source_profile))
    source_contract_lookup = (
        _source_contract_lookup_from_audit(source_contract_audit_path)
        if source_contract_audit_path
        else {}
    )
    source_contract_validations: Dict[Tuple[int, str], Dict[str, Any]] = {}
    modes = _sidecar_mode_set(sidecar_modes)
    allowed_tensors = _csv_filter(patch_tensors)
    patches, skipped, selection = _select_refit_patches(
        refit_eval,
        allowed_tensors=allowed_tensors,
        allowed_methods=None,
        top_groups=patch_top_groups,
    )
    selected_keys = {
        (int(patch["layer"]), str(patch["tensor"]), int(patch["group"]), int(patch["output_row"]))
        for patch in patches
    }
    selected_group_keys = {
        (int(patch["layer"]), str(patch["tensor"]), int(patch["group"]))
        for patch in patches
    }

    tensor_cache: Dict[Tuple[int, str], Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]] = {}
    row_cache: Dict[Tuple[int, str], torch.Tensor] = {}
    evaluations: List[Dict[str, Any]] = []
    for evaluation in refit_eval.get("evaluations", []) or []:
        layer = int(evaluation["layer"])
        target_tensor = str(evaluation["target_tensor"])
        activation_tensor = str(evaluation["activation_tensor"])
        group = int(evaluation["group"])
        group_key = (layer, target_tensor, group)
        if group_key not in selected_group_keys:
            continue
        entry = entries.get((layer, target_tensor))
        if entry is None:
            skipped.append(
                {
                    "layer": layer,
                    "tensor": target_tensor,
                    "group": group,
                    "reason": "target tensor missing from manifest during sidecar simulation",
                }
            )
            continue
        try:
            tensor_key = (layer, target_tensor)
            if tensor_key not in tensor_cache:
                if source_contract_lookup:
                    entry = _attach_source_contract(entry, source_contract_lookup)
                    source_contract_validations[tensor_key] = _validate_source_contract_entry(
                        model_path=model_path,
                        cache_dir=cache_dir,
                        entry=entry,
                    )
                source = _load_source_tensor(model_path, entry).to(torch.float32).contiguous()
                hqq = _load_hqq_tensor(cache_dir, entry)
                if list(source.shape) != [int(hqq["rows"]), int(hqq["cols"])]:
                    raise RuntimeError(
                        f"source/HQQ shape mismatch for layer={layer} tensor={target_tensor}: "
                        f"{list(source.shape)} vs {[hqq['rows'], hqq['cols']]}"
                    )
                tensor_cache[tensor_key] = (source, hqq, entry)
            source, hqq, entry = tensor_cache[tensor_key]
            row_key = (layer, activation_tensor)
            if row_key not in row_cache:
                row_cache[row_key] = _load_activation_rows(trace_path, layer=layer, tensor=activation_tensor)
            input_rows_full = row_cache[row_key]
            group_size = int(hqq["group_size"])
            start = group * group_size
            end = min(start + group_size, int(hqq["cols"]))
            input_rows = input_rows_full[:, start:end].contiguous()
            source_group = source[:, start:end].contiguous()
            cached_group = hqq["dequant"][:, start:end].contiguous()
            selected_rows = []
            for row in evaluation.get("selected_output_rows", []) or []:
                output_row = int(row["output_row"])
                if (layer, target_tensor, group, output_row) not in selected_keys:
                    continue
                src = source_group[output_row]
                cached = cached_group[output_row]
                cached_diff = cached - src
                residual = src - cached
                sidecar_rows = [
                    {
                        "name": "cached_hqq",
                        "projection": _projection_metrics(input_rows, cached_diff),
                        "sidecar_weight_bytes": 0,
                        "sidecar_metadata_bytes": 0,
                        "decode_fma_per_token": 0,
                    }
                ]
                if "exact_bf16" in modes:
                    exact_correction = residual.to(torch.float32)
                    sidecar_rows.append(
                        {
                            "name": "exact_bf16_sidecar",
                            "projection": _sidecar_projection_metrics(input_rows, cached_diff, exact_correction),
                            "sidecar_weight_bytes": int(residual.numel()) * 2,
                            "sidecar_metadata_bytes": 16,
                            "decode_fma_per_token": int(residual.numel()),
                            "correction_error_rms": 0.0,
                            "correction_error_max_abs": 0.0,
                        }
                    )
                if "int8_symmetric" in modes:
                    q, dequant, scale = _quantize_int8_symmetric(residual)
                    correction_error = dequant - residual
                    sidecar_rows.append(
                        {
                            "name": "int8_symmetric_sidecar",
                            "projection": _sidecar_projection_metrics(input_rows, cached_diff, dequant),
                            "sidecar_weight_bytes": int(q.numel()),
                            "sidecar_metadata_bytes": 20,
                            "decode_fma_per_token": int(q.numel()),
                            "scale": scale,
                            "qvalue_min": int(q.min().item()) if q.numel() else 0,
                            "qvalue_max": int(q.max().item()) if q.numel() else 0,
                            "correction_error_rms": float(
                                torch.sqrt(torch.mean(correction_error * correction_error)).item()
                            )
                            if correction_error.numel()
                            else 0.0,
                            "correction_error_max_abs": float(correction_error.abs().max().item())
                            if correction_error.numel()
                            else 0.0,
                        }
                    )
                selected_rows.append(
                    {
                        "output_row": output_row,
                        "sidecar_modes": sidecar_rows,
                    }
                )
            if selected_rows:
                evaluations.append(
                    {
                        "layer": layer,
                        "target_tensor": target_tensor,
                        "activation_tensor": activation_tensor,
                        "activation_policy": evaluation["activation_policy"],
                        "group": group,
                        "start_col": start,
                        "end_col": end,
                        "source_sha256": _tensor_hash(source_group),
                        "hqq_dequant_sha256": _tensor_hash(cached_group),
                        "source_contract": copy.deepcopy(entry.get("source_contract"))
                        if source_contract_lookup
                        else None,
                        "source_contract_validation": source_contract_validations.get(tensor_key)
                        if source_contract_lookup
                        else None,
                        "activation_rows": int(input_rows.shape[0]),
                        "selected_output_row_count": len(selected_rows),
                        "selected_output_rows": selected_rows,
                    }
                )
        except Exception as exc:
            skipped.append(
                {
                    "layer": layer,
                    "tensor": target_tensor,
                    "group": group,
                    "reason": str(exc),
                }
            )

    aggregate = _aggregate_sidecar_rows(evaluations)
    payload = {
        "format": SIDECAR_SIM_FORMAT,
        "format_version": SIDECAR_SIM_FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "sidecar_simulation_built",
        "complete": False,
        "loadable_hqq_manifest": False,
        "not_loadable_reason": (
            "Phase 2F sidecar simulation only. No sidecar runtime artifacts, calibrated HQQ "
            "manifest, HQQ tensor artifacts, qvalues, scales, or zeros were written."
        ),
        "model_path": model_path,
        "source_refit_evaluation_path": refit_eval_path,
        "source_trace_path": trace_path,
        "cache": {
            "source_cache_profile": source_profile,
            "source_cache_dir": cache_dir,
            "target_cache_profile": HQQ_CACHE_PROFILE_SELFCAL_V1,
            "target_cache_dir": hqq_attention_cache_dir(model_path, HQQ_CACHE_PROFILE_SELFCAL_V1),
            "target_manifest_path": hqq_attention_manifest_path(model_path, HQQ_CACHE_PROFILE_SELFCAL_V1),
            "target_manifest_written": False,
            "sidecar_simulation_path": output_path,
            "source_contract_audit_path": source_contract_audit_path,
            "source_contract_audit_sha256": _file_sha256(source_contract_audit_path)
            if source_contract_audit_path
            else None,
        },
        "sidecar_model": {
            "purpose": "simulate separate batched correction sidecars before any runtime format exists",
            "main_int4_path": "unchanged regular HQQ INT4",
            "sentinel_values": False,
            "main_kernel_branching": False,
            "correction_formula": "output = HQQ_INT4_output + activation_slice @ sidecar_correction",
            "modes": sorted(modes),
            "caveat": (
                "This is local projected-error evidence and byte/FMA estimation only. It does not prove "
                "end-to-end witness quality until a runtime sidecar path can be compared."
            ),
        },
        "selection": selection,
        "summary": {
            "selected_group_count": len(selected_group_keys),
            "selected_row_group_count": len(selected_keys),
            "simulated_group_count": len(evaluations),
            "simulated_row_group_count": sum(
                int(e.get("selected_output_row_count", 0)) for e in evaluations
            ),
            "skipped_count": len(skipped),
            "aggregate_by_mode": aggregate,
            "improvement_vs_cached": _reduction_vs_cached(aggregate),
            "source_contract": _source_contract_summary(source_contract_validations.values())
            if source_contract_lookup
            else None,
        },
        "evaluations": evaluations,
        "skipped": skipped,
    }
    write_json(output_path, payload)
    latest_path = os.path.join(os.path.dirname(output_path), "sidecar_simulation_latest.json")
    write_json(latest_path, payload)
    return payload


def _sidecar_artifact_file(layer: int, tensor_name: str, mode: str) -> str:
    safe_tensor = re.sub(r"[^A-Za-z0-9_.-]+", "_", tensor_name).strip("._")
    return f"layer_{layer:03d}_{safe_tensor}_{mode}_sidecar.safetensors"


def _parse_sidecar_entry_filter(value: Optional[str]) -> Optional[set[Tuple[int, str, int, int]]]:
    if value is None or not value.strip():
        return None
    out: set[Tuple[int, str, int, int]] = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(
                "Sidecar entry filters must use layer:tensor:group:output_row entries, "
                f"got {item!r}"
            )
        try:
            layer = int(parts[0])
            group = int(parts[2])
            output_row = int(parts[3])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric field in sidecar entry filter {item!r}") from exc
        out.add((layer, parts[1], group, output_row))
    return out


def _sidecar_entry_filter_to_list(entries: Optional[set[Tuple[int, str, int, int]]]) -> Optional[List[str]]:
    if entries is None:
        return None
    return [f"{layer}:{tensor}:{group}:{row}" for layer, tensor, group, row in sorted(entries)]


def _sidecar_entry_key_to_str(entry: Tuple[int, str, int, int]) -> str:
    layer, tensor, group, output_row = entry
    return f"{layer}:{tensor}:{group}:{output_row}"


def _validate_sidecar_correction_scale(value: float) -> float:
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0 or scale > 1.0:
        raise ValueError(
            "Sidecar correction scale must be finite and in the interval (0, 1]; "
            f"got {value!r}"
        )
    return scale


def _parse_sidecar_entry_scales(value: Optional[str]) -> Dict[Tuple[int, str, int, int], float]:
    if value is None or not value.strip():
        return {}
    out: Dict[Tuple[int, str, int, int], float] = {}
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        entry_text, sep, scale_text = item.partition("=")
        if not sep:
            raise ValueError(
                "Sidecar entry scales must use layer:tensor:group:output_row=scale entries, "
                f"got {item!r}"
            )
        parsed = _parse_sidecar_entry_filter(entry_text)
        if parsed is None or len(parsed) != 1:
            raise ValueError(f"Invalid sidecar entry-scale key: {entry_text!r}")
        key = next(iter(parsed))
        scale = _validate_sidecar_correction_scale(float(scale_text))
        if key in out:
            raise ValueError(f"Duplicate sidecar entry-scale key: {_sidecar_entry_key_to_str(key)}")
        out[key] = scale
    return out


def _sidecar_entry_scales_to_list(scales: Dict[Tuple[int, str, int, int], float]) -> List[Dict[str, Any]]:
    return [
        {"entry": _sidecar_entry_key_to_str(key), "correction_scale": scale}
        for key, scale in sorted(scales.items())
    ]


def _pad_rows(rows: List[torch.Tensor], *, dtype: torch.dtype, pad_value: float = 0.0) -> torch.Tensor:
    if not rows:
        return torch.empty((0, 0), dtype=dtype)
    max_width = max(int(row.numel()) for row in rows)
    out = torch.full((len(rows), max_width), pad_value, dtype=dtype)
    for idx, row in enumerate(rows):
        width = int(row.numel())
        if width:
            out[idx, :width] = row.to(dtype)
    return out.contiguous()


def write_sidecar_contract_manifest(
    *,
    cfg: Dict[str, str],
    source_contract_audit_path: str,
    output_path: str,
    variant_name: str,
) -> Dict[str, Any]:
    model_path = os.path.expanduser(cfg.get("CFG_MODEL_PATH", ""))
    if not model_path:
        raise ValueError("Cannot determine model path for source-contract sidecar manifest")
    source_profile = cfg.get("CFG_HQQ_CACHE_PROFILE", HQQ_CACHE_PROFILE_BASELINE) or HQQ_CACHE_PROFILE_BASELINE
    manifest_probe = load_hqq_attention_manifest(model_path, str(source_profile))
    if manifest_probe is None:
        raise RuntimeError(f"No HQQ source manifest found for profile {source_profile}")
    source_manifest = require_complete_hqq_attention_manifest(
        model_path,
        str(source_profile),
        expected_nbits=4,
        expected_num_hidden_layers=int(manifest_probe.get("num_hidden_layers", 0)),
    )
    source_manifest_path = hqq_attention_manifest_path(model_path, str(source_profile))
    source_cache_dir = hqq_attention_cache_dir(model_path, str(source_profile))
    target_sidecar_dir = sidecar_variant_dir(
        model_path,
        HQQ_CACHE_PROFILE_SELFCAL_V1,
        variant_name,
        "source_contract_manifest_only",
    )
    os.makedirs(target_sidecar_dir, exist_ok=True)
    manifest_path = os.path.join(target_sidecar_dir, "sidecar_manifest.json")
    lookup = _source_contract_lookup_from_audit(source_contract_audit_path)

    artifact_summaries: List[Dict[str, Any]] = []
    validations: List[Dict[str, Any]] = []
    for entry in sorted(
        source_manifest.get("tensors", []) or [],
        key=lambda item: (int(item.get("layer_idx", -1)), str(item.get("tensor_name", ""))),
    ):
        entry_with_contract = _attach_source_contract(entry, lookup)
        validation = _validate_source_contract_entry(
            model_path=model_path,
            cache_dir=source_cache_dir,
            entry=entry_with_contract,
        )
        validations.append(validation)
        layer = int(entry_with_contract["layer_idx"])
        tensor_name = str(entry_with_contract["tensor_name"])
        artifact_summaries.append(
            {
                "layer": layer,
                "tensor": tensor_name,
                "file": None,
                "hqq_artifact_file": entry_with_contract.get("file"),
                "hqq_artifact_sha256": entry_with_contract["source_contract"].get("artifact_sha256"),
                "row_group_count": 0,
                "weight_bytes": 0,
                "metadata_bytes": 0,
                "decode_fma_per_token": 0,
                "source_contract": copy.deepcopy(entry_with_contract["source_contract"]),
                "source_contract_validation": validation,
            }
        )

    if len(validations) != len(source_manifest.get("tensors", []) or []):
        raise RuntimeError("Source-contract manifest target count mismatch")
    summary = _source_contract_summary(validations)
    if summary["generation_ready"] != summary["target_count"]:
        raise RuntimeError(f"Not all source contracts are generation-ready: {summary}")

    manifest = {
        "format": SIDECAR_MANIFEST_FORMAT,
        "format_version": SIDECAR_MANIFEST_FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "source_contract_manifest_only",
        "complete": False,
        "loadable_hqq_manifest": False,
        "not_loadable_reason": (
            "Manifest-only source-contract artifact. It records strict offline provenance for future "
            "sidecar generation and intentionally contains no correction payload files."
        ),
        "model_path": model_path,
        "variant_name": variant_name,
        "sidecar_mode": "source_contract_manifest_only",
        "runtime_application_implemented": False,
        "runtime_ready": False,
        "source": {
            "cache_profile": source_profile,
            "baseline_manifest_path": source_manifest_path,
            "baseline_manifest_sha256": _file_sha256(source_manifest_path),
            "source_contract_audit_path": source_contract_audit_path,
            "source_contract_audit_sha256": _file_sha256(source_contract_audit_path),
        },
        "cache": {
            "target_cache_profile": HQQ_CACHE_PROFILE_SELFCAL_V1,
            "sidecar_dir": target_sidecar_dir,
            "sidecar_manifest_path": manifest_path,
        },
        "summary": {
            **summary,
            "artifact_count": len(artifact_summaries),
            "row_group_count": 0,
            "sidecar_weight_bytes": 0,
            "sidecar_metadata_bytes": 0,
            "sidecar_total_bytes": 0,
            "decode_fma_per_token": 0,
        },
        "artifacts": artifact_summaries,
        "switchability": {
            "fallback_to_baseline": False,
            "runtime_enabled": False,
            "selection": "manifest-only; no runtime sidecar selection or application",
        },
    }
    write_json(manifest_path, manifest)
    latest_manifest = os.path.join(target_sidecar_dir, "sidecar_manifest_latest.json")
    write_json(latest_manifest, manifest)
    report = {
        "format": SIDECAR_CONTRACT_MANIFEST_WRITE_FORMAT,
        "format_version": SIDECAR_CONTRACT_MANIFEST_WRITE_FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "source_contract_manifest_written",
        "complete": True,
        "loadable_hqq_manifest": False,
        "runtime_ready": False,
        "model_path": model_path,
        "variant_name": variant_name,
        "cache": manifest["cache"],
        "source": manifest["source"],
        "summary": manifest["summary"],
        "sidecar_manifest": manifest,
    }
    write_json(output_path, report)
    latest_report = os.path.join(os.path.dirname(output_path), "sidecar_contract_manifest_latest.json")
    write_json(latest_report, report)
    return report


def write_sidecar_artifacts(
    *,
    cfg: Dict[str, str],
    sidecar_simulation_path: str,
    output_path: str,
    variant_name: str,
    sidecar_artifact_mode: str,
    sidecar_correction_scale: float,
    sidecar_include_entries: Optional[str],
    sidecar_exclude_entries: Optional[str],
    sidecar_entry_scales: Optional[str],
    source_contract_audit_path: Optional[str],
) -> Dict[str, Any]:
    with open(sidecar_simulation_path, encoding="utf-8") as f:
        simulation = json.load(f)
    if simulation.get("format") != SIDECAR_SIM_FORMAT:
        raise ValueError(f"Not an HQQ self-calibration sidecar simulation file: {sidecar_simulation_path}")
    mode = _sidecar_artifact_mode(sidecar_artifact_mode)
    correction_scale = _validate_sidecar_correction_scale(sidecar_correction_scale)

    model_path = os.path.expanduser(cfg.get("CFG_MODEL_PATH", simulation.get("model_path", "")))
    if not model_path:
        raise ValueError("Cannot determine model path for sidecar artifact write")
    source_profile = simulation.get("cache", {}).get("source_cache_profile") or HQQ_CACHE_PROFILE_BASELINE
    manifest_probe = load_hqq_attention_manifest(model_path, str(source_profile))
    if manifest_probe is None:
        raise RuntimeError(f"No HQQ source manifest found for profile {source_profile}")
    source_manifest = require_complete_hqq_attention_manifest(
        model_path,
        str(source_profile),
        expected_nbits=4,
        expected_num_hidden_layers=int(manifest_probe.get("num_hidden_layers", 0)),
    )
    source_manifest_path = hqq_attention_manifest_path(model_path, str(source_profile))
    source_cache_dir = hqq_attention_cache_dir(model_path, str(source_profile))
    target_cache_dir = hqq_attention_cache_dir(model_path, HQQ_CACHE_PROFILE_SELFCAL_V1)
    hqq_target_manifest_path = hqq_attention_manifest_path(model_path, HQQ_CACHE_PROFILE_SELFCAL_V1)
    target_sidecar_dir = sidecar_variant_dir(
        model_path,
        HQQ_CACHE_PROFILE_SELFCAL_V1,
        variant_name,
        mode,
    )
    os.makedirs(target_sidecar_dir, exist_ok=True)
    manifest_path = os.path.join(target_sidecar_dir, "sidecar_manifest.json")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    include_entries = _parse_sidecar_entry_filter(sidecar_include_entries)
    exclude_entries = _parse_sidecar_entry_filter(sidecar_exclude_entries)
    entry_scales = _parse_sidecar_entry_scales(sidecar_entry_scales)
    source_contract_lookup = (
        _source_contract_lookup_from_audit(source_contract_audit_path)
        if source_contract_audit_path
        else {}
    )
    if include_entries is not None and exclude_entries is not None:
        overlap = include_entries & exclude_entries
        if overlap:
            raise ValueError(
                "Sidecar include/exclude filters overlap: "
                + ", ".join(_sidecar_entry_filter_to_list(overlap) or [])
            )
    if entry_scales:
        if include_entries is None:
            raise ValueError(
                "Sidecar entry scales require --sidecar-include-entries so mixed-scale writes are explicit"
            )
        extra_scale_keys = set(entry_scales) - include_entries
        if extra_scale_keys:
            raise ValueError(
                "Sidecar entry scales reference entries outside the include set: "
                + ", ".join(_sidecar_entry_filter_to_list(extra_scale_keys) or [])
            )
        if exclude_entries is not None:
            excluded_scale_keys = set(entry_scales) & exclude_entries
            if excluded_scale_keys:
                raise ValueError(
                    "Sidecar entry scales reference excluded entries: "
                    + ", ".join(_sidecar_entry_filter_to_list(excluded_scale_keys) or [])
                )

    entries = _manifest_tensor_entries(source_manifest)
    tensor_cache: Dict[Tuple[int, str], Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]] = {}
    row_entries_by_tensor: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    source_contract_validations: Dict[Tuple[int, str], Dict[str, Any]] = {}
    skipped: List[Dict[str, Any]] = []
    for evaluation in simulation.get("evaluations", []) or []:
        layer = int(evaluation["layer"])
        target_tensor = str(evaluation["target_tensor"])
        group = int(evaluation["group"])
        entry = entries.get((layer, target_tensor))
        if entry is None:
            skipped.append(
                {
                    "layer": layer,
                    "tensor": target_tensor,
                    "group": group,
                    "reason": "target tensor missing from source manifest during sidecar write",
                }
            )
            continue
        tensor_key = (layer, target_tensor)
        try:
            if tensor_key not in tensor_cache:
                if source_contract_lookup:
                    entry = _attach_source_contract(entry, source_contract_lookup)
                    source_contract_validations[tensor_key] = _validate_source_contract_entry(
                        model_path=model_path,
                        cache_dir=source_cache_dir,
                        entry=entry,
                    )
                source = _load_source_tensor(model_path, entry).to(torch.float32).contiguous()
                hqq = _load_hqq_tensor(source_cache_dir, entry)
                if list(source.shape) != [int(hqq["rows"]), int(hqq["cols"])]:
                    raise RuntimeError(
                        f"source/HQQ shape mismatch for layer={layer} tensor={target_tensor}: "
                        f"{list(source.shape)} vs {[hqq['rows'], hqq['cols']]}"
                    )
                tensor_cache[tensor_key] = (source, hqq, entry)
            source, hqq, entry = tensor_cache[tensor_key]
            group_size = int(hqq["group_size"])
            start = group * group_size
            end = min(start + group_size, int(hqq["cols"]))
            if int(evaluation["start_col"]) != start or int(evaluation["end_col"]) != end:
                raise RuntimeError(f"sidecar simulation group bounds do not match source artifact: {evaluation}")
            for row in evaluation.get("selected_output_rows", []) or []:
                output_row = int(row["output_row"])
                filter_key = (layer, target_tensor, group, output_row)
                if include_entries is not None and filter_key not in include_entries:
                    continue
                if exclude_entries is not None and filter_key in exclude_entries:
                    continue
                row_correction_scale = entry_scales.get(filter_key, correction_scale)
                sidecar_modes = {str(item.get("name")): item for item in row.get("sidecar_modes", []) or []}
                mode_name = "exact_bf16_sidecar" if mode == "exact_bf16" else "int8_symmetric_sidecar"
                if mode_name not in sidecar_modes:
                    skipped.append(
                        {
                            "layer": layer,
                            "tensor": target_tensor,
                            "group": group,
                            "output_row": output_row,
                            "reason": f"sidecar mode {mode_name!r} absent from simulation row",
                        }
                    )
                    continue
                residual = (
                    source[output_row, start:end].to(torch.float32)
                    - hqq["dequant"][output_row, start:end].to(torch.float32)
                ).contiguous() * row_correction_scale
                if mode == "int8_symmetric":
                    q, dequant, scale = _quantize_int8_symmetric(residual)
                    stored = q
                    scale_value = scale
                    correction_error = dequant - residual
                else:
                    stored = residual.to(torch.bfloat16)
                    scale_value = 1.0
                    correction_error = torch.zeros_like(residual)
                row_entries_by_tensor.setdefault(tensor_key, []).append(
                    {
                        "layer": layer,
                        "tensor": target_tensor,
                        "activation_tensor": str(evaluation["activation_tensor"]),
                        "activation_policy": str(evaluation["activation_policy"]),
                        "group": group,
                        "output_row": output_row,
                        "start_col": start,
                        "end_col": end,
                        "width": end - start,
                        "stored": stored,
                        "scale": scale_value,
                        "correction_scale": row_correction_scale,
                        "decode_fma_per_token": end - start,
                        "projection": sidecar_modes[mode_name].get("projection", {}),
                        "correction_error_rms": float(
                            torch.sqrt(torch.mean(correction_error * correction_error)).item()
                        )
                        if correction_error.numel()
                        else 0.0,
                        "correction_error_max_abs": float(correction_error.abs().max().item())
                        if correction_error.numel()
                        else 0.0,
                    }
                )
        except Exception as exc:
            skipped.append(
                {
                    "layer": layer,
                    "tensor": target_tensor,
                    "group": group,
                    "reason": str(exc),
                }
            )

    if not row_entries_by_tensor:
        raise RuntimeError("Sidecar simulation did not contain any writable sidecar rows")

    artifact_summaries = []
    total_weight_bytes = 0
    total_metadata_bytes = 0
    total_fmas = 0
    total_row_groups = 0
    for tensor_key in sorted(row_entries_by_tensor):
        layer, tensor_name = tensor_key
        rows = row_entries_by_tensor[tensor_key]
        rel_file = _sidecar_artifact_file(layer, tensor_name, mode)
        path = os.path.join(target_sidecar_dir, rel_file)
        output_rows = torch.tensor([int(item["output_row"]) for item in rows], dtype=torch.int32)
        groups = torch.tensor([int(item["group"]) for item in rows], dtype=torch.int32)
        start_cols = torch.tensor([int(item["start_col"]) for item in rows], dtype=torch.int32)
        end_cols = torch.tensor([int(item["end_col"]) for item in rows], dtype=torch.int32)
        widths = torch.tensor([int(item["width"]) for item in rows], dtype=torch.int32)
        scales = torch.tensor([float(item["scale"]) for item in rows], dtype=torch.float32)
        if mode == "int8_symmetric":
            tensor_payload = {
                "correction_qint8": _pad_rows([item["stored"] for item in rows], dtype=torch.int8),
                "scales": scales,
            }
            weight_bytes = sum(int(item["width"]) for item in rows)
        else:
            tensor_payload = {
                "correction_bf16": _pad_rows([item["stored"] for item in rows], dtype=torch.bfloat16),
                "scales": scales,
            }
            weight_bytes = sum(int(item["width"]) * 2 for item in rows)
        tensor_payload.update(
            {
                "output_rows": output_rows,
                "groups": groups,
                "start_cols": start_cols,
                "end_cols": end_cols,
                "widths": widths,
            }
        )
        tmp_path = path + ".tmp"
        save_file(
            tensor_payload,
            tmp_path,
            metadata={
                "format": SIDECAR_MANIFEST_FORMAT,
                "mode": mode,
                "variant_name": variant_name,
                "layer": str(layer),
                "tensor": tensor_name,
                "correction_scale": str(correction_scale),
                "entry_scale_overrides": str(len(entry_scales)),
            },
        )
        os.replace(tmp_path, path)
        metadata_bytes = len(rows) * 20
        fmas = sum(int(item["decode_fma_per_token"]) for item in rows)
        total_weight_bytes += weight_bytes
        total_metadata_bytes += metadata_bytes
        total_fmas += fmas
        total_row_groups += len(rows)
        artifact_summaries.append(
            {
                "layer": layer,
                "tensor": tensor_name,
                "file": rel_file,
                "sha256": _file_sha256(path),
                "row_group_count": len(rows),
                "weight_bytes": weight_bytes,
                "metadata_bytes": metadata_bytes,
                "decode_fma_per_token": fmas,
                "source_contract": copy.deepcopy(tensor_cache[tensor_key][2].get("source_contract"))
                if source_contract_lookup
                else None,
                "source_contract_validation": source_contract_validations.get(tensor_key)
                if source_contract_lookup
                else None,
                "entries": [
                    {
                        "output_row": int(item["output_row"]),
                        "group": int(item["group"]),
                        "start_col": int(item["start_col"]),
                        "end_col": int(item["end_col"]),
                        "activation_tensor": str(item["activation_tensor"]),
                        "activation_policy": str(item["activation_policy"]),
                        "correction_scale": float(item["correction_scale"]),
                        "correction_error_rms": float(item["correction_error_rms"]),
                        "correction_error_max_abs": float(item["correction_error_max_abs"]),
                    }
                    for item in rows
                ],
            }
        )

    manifest = {
        "format": SIDECAR_MANIFEST_FORMAT,
        "format_version": SIDECAR_MANIFEST_FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "experimental_sidecar_artifact",
        "complete": True,
        "loadable_hqq_manifest": False,
        "not_loadable_reason": (
            "Sidecar artifact only. This is not a calibrated HQQ manifest and does not make "
            "selfcal_v1 loadable without an explicit runtime sidecar application path."
        ),
        "model_path": model_path,
        "variant_name": variant_name,
        "sidecar_mode": mode,
        "correction_scale": correction_scale,
        "correction_scale_policy": "per_entry_overrides" if entry_scales else "global",
        "entry_scale_override_count": len(entry_scales),
        "main_int4_path": "unchanged regular HQQ INT4",
        "sentinel_values": False,
        "main_kernel_branching": False,
        "runtime_application_implemented": False,
        "source": {
            "cache_profile": source_profile,
            "baseline_manifest_path": source_manifest_path,
            "baseline_manifest_sha256": _file_sha256(source_manifest_path),
            "sidecar_simulation_path": sidecar_simulation_path,
            "sidecar_simulation_sha256": _file_sha256(sidecar_simulation_path),
            "source_contract_audit_path": source_contract_audit_path,
            "source_contract_audit_sha256": _file_sha256(source_contract_audit_path)
            if source_contract_audit_path
            else None,
            "refit_evaluation_path": simulation.get("source_refit_evaluation_path"),
            "refit_evaluation_sha256": _file_sha256(str(simulation["source_refit_evaluation_path"]))
            if simulation.get("source_refit_evaluation_path")
            and os.path.isfile(str(simulation["source_refit_evaluation_path"]))
            else None,
            "trace_path": simulation.get("source_trace_path"),
            "trace_sha256": _file_sha256(str(simulation["source_trace_path"]))
            if simulation.get("source_trace_path")
            and os.path.isfile(str(simulation["source_trace_path"]))
            else None,
        },
        "cache": {
            "target_cache_profile": HQQ_CACHE_PROFILE_SELFCAL_V1,
            "target_cache_dir": target_cache_dir,
            "hqq_target_manifest_path": hqq_target_manifest_path,
            "hqq_target_manifest_written": False,
            "sidecar_dir": target_sidecar_dir,
            "sidecar_manifest_path": manifest_path,
        },
        "summary": {
            "artifact_count": len(artifact_summaries),
            "row_group_count": total_row_groups,
            "sidecar_weight_bytes": total_weight_bytes,
            "sidecar_metadata_bytes": total_metadata_bytes,
            "sidecar_total_bytes": total_weight_bytes + total_metadata_bytes,
            "decode_fma_per_token": total_fmas,
            "skipped_count": len(skipped),
            "correction_scale": correction_scale,
            "entry_scale_override_count": len(entry_scales),
            "source_contract": _source_contract_summary(source_contract_validations.values())
            if source_contract_lookup
            else None,
        },
        "artifacts": artifact_summaries,
        "skipped": skipped,
        "switchability": {
            "comparison_target": "INT4 only versus INT4 + correction",
            "selection": "profile/run-level sidecar selection; no main INT4 kernel branch",
            "fallback_to_baseline": False,
        },
        "entry_filter": {
            "include_entries": _sidecar_entry_filter_to_list(include_entries),
            "exclude_entries": _sidecar_entry_filter_to_list(exclude_entries),
            "entry_scales": _sidecar_entry_scales_to_list(entry_scales),
        },
    }
    write_json(manifest_path, manifest)
    latest_manifest = os.path.join(target_sidecar_dir, "sidecar_manifest_latest.json")
    write_json(latest_manifest, manifest)
    report = {
        "format": SIDECAR_WRITE_FORMAT,
        "format_version": SIDECAR_WRITE_FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "sidecar_artifacts_written",
        "complete": True,
        "loadable_hqq_manifest": False,
        "model_path": model_path,
        "variant_name": variant_name,
        "sidecar_mode": mode,
        "correction_scale": correction_scale,
        "cache": manifest["cache"],
        "source": manifest["source"],
        "summary": manifest["summary"],
        "sidecar_manifest": manifest,
    }
    write_json(output_path, report)
    latest_report = os.path.join(os.path.dirname(output_path), "sidecar_write_latest.json")
    write_json(latest_report, report)
    return report


def evaluate_refit_candidates(
    *,
    cfg: Dict[str, str],
    candidates_path: str,
    trace_path: str,
    output_path: str,
    top_candidates: int,
    max_output_rows: int,
    grid_steps: int,
    local_grid_steps: int,
    activation_policy: str,
    source_contract_audit_path: Optional[str] = None,
) -> Dict[str, Any]:
    with open(candidates_path, encoding="utf-8") as f:
        candidates_doc = json.load(f)
    if candidates_doc.get("format") != CANDIDATES_FORMAT:
        raise ValueError(f"Not an HQQ self-calibration candidates file: {candidates_path}")
    model_path = os.path.expanduser(cfg.get("CFG_MODEL_PATH", candidates_doc.get("model_path", "")))
    if not model_path:
        raise ValueError("Cannot determine model path for refit evaluation")
    source_profile = candidates_doc.get("cache", {}).get("source_cache_profile") or HQQ_CACHE_PROFILE_BASELINE
    manifest = require_complete_hqq_attention_manifest(
        model_path,
        str(source_profile),
        expected_nbits=4,
        expected_num_hidden_layers=int(load_hqq_attention_manifest(model_path, str(source_profile)).get("num_hidden_layers", 0)),
    )
    entries = _manifest_tensor_entries(manifest)
    cache_dir = hqq_attention_cache_dir(model_path, str(source_profile))
    source_contract_lookup = (
        _source_contract_lookup_from_audit(source_contract_audit_path)
        if source_contract_audit_path
        else {}
    )
    source_contract_validations: Dict[Tuple[int, str], Dict[str, Any]] = {}
    selected = []
    for candidate in candidates_doc.get("candidates", []):
        if activation_policy != "all" and candidate.get("activation_policy") != activation_policy:
            continue
        selected.append(candidate)
        if len(selected) >= top_candidates:
            break

    tensor_cache: Dict[Tuple[int, str], Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]] = {}
    row_cache: Dict[Tuple[int, str], torch.Tensor] = {}
    evaluations: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for candidate in selected:
        layer = int(candidate["layer"])
        target_tensor = str(candidate["target_tensor"])
        activation_tensor = str(candidate["activation_tensor"])
        group = int(candidate["group"])
        key = (layer, target_tensor)
        entry = entries.get(key)
        if entry is None:
            skipped.append({"candidate": candidate, "reason": "target tensor missing from manifest"})
            continue
        try:
            if key not in tensor_cache:
                if source_contract_lookup:
                    entry = _attach_source_contract(entry, source_contract_lookup)
                    source_contract_validations[key] = _validate_source_contract_entry(
                        model_path=model_path,
                        cache_dir=cache_dir,
                        entry=entry,
                    )
                source = _load_source_tensor(model_path, entry).to(torch.float32).contiguous()
                hqq = _load_hqq_tensor(cache_dir, entry)
                if list(source.shape) != [int(hqq["rows"]), int(hqq["cols"])]:
                    raise RuntimeError(
                        f"source/HQQ shape mismatch for layer={layer} tensor={target_tensor}: "
                        f"{list(source.shape)} vs {[hqq['rows'], hqq['cols']]}"
                    )
                tensor_cache[key] = (source, hqq, entry)
            source, hqq, entry = tensor_cache[key]
            row_key = (layer, activation_tensor)
            if row_key not in row_cache:
                row_cache[row_key] = _load_activation_rows(trace_path, layer=layer, tensor=activation_tensor)
            input_rows_full = row_cache[row_key]
            group_size = int(hqq["group_size"])
            start = group * group_size
            end = min(start + group_size, int(hqq["cols"]))
            if input_rows_full.shape[1] != int(hqq["cols"]):
                raise RuntimeError(
                    f"activation/HQQ width mismatch for layer={layer} tensor={activation_tensor}: "
                    f"{input_rows_full.shape[1]} vs {hqq['cols']}"
                )
            input_rows = input_rows_full[:, start:end].contiguous()
            source_group = source[:, start:end].contiguous()
            cached_group = hqq["dequant"][:, start:end].contiguous()
            cached_delta = cached_group - source_group
            projected = input_rows @ cached_delta.T
            row_rms = torch.sqrt(torch.mean(projected * projected, dim=0))
            row_count = min(max_output_rows, int(row_rms.numel()))
            output_rows = torch.topk(row_rms, k=row_count).indices.tolist()
            selected_rows = []
            weights = (input_rows * input_rows).mean(dim=0)
            for row_idx in output_rows:
                row_idx = int(row_idx)
                src = source_group[row_idx]
                cached_q = hqq["qvalues"][row_idx, start:end]
                cached_scale = float(hqq["scales"][row_idx, group].item())
                cached_zero = float(hqq["zeros"][row_idx, group].item())
                minmax_scale, minmax_zero = _minmax_params(src)
                weighted_scale, weighted_zero, weighted_q = _weighted_affine_params(
                    src,
                    weights,
                    grid_steps=grid_steps,
                    local_grid_steps=local_grid_steps,
                )
                clip_scale, clip_zero, clip_q, clip_meta = _clip_free_weighted_params(
                    src,
                    input_rows,
                    grid_steps=grid_steps,
                    local_grid_steps=local_grid_steps,
                )
                candidates = [
                    _candidate_metrics(
                        "cached_hqq",
                        src,
                        input_rows,
                        scale=cached_scale,
                        zero=cached_zero,
                        q=cached_q,
                    ),
                    _candidate_metrics("minmax_affine", src, input_rows, scale=minmax_scale, zero=minmax_zero),
                    _candidate_metrics(
                        "activation_weighted_affine",
                        src,
                        input_rows,
                        scale=weighted_scale,
                        zero=weighted_zero,
                        q=weighted_q,
                    ),
                    _candidate_metrics(
                        "activation_weighted_clip_free",
                        src,
                        input_rows,
                        scale=clip_scale,
                        zero=clip_zero,
                        q=clip_q,
                        extra={"protected_columns": clip_meta},
                    ),
                ]
                cached = candidates[0]["projection"]
                best = min(candidates, key=lambda item: float(item["projection"]["rms"]))
                selected_rows.append(
                    {
                        "output_row": row_idx,
                        "cached_projection_rms_rank_score": float(row_rms[row_idx].item()),
                        "best_candidate_by_projection_rms": best["name"],
                        "best_vs_cached_projection_rms_ratio": float(best["projection"]["rms"])
                        / max(float(cached["rms"]), 1e-30),
                        "candidates": candidates,
                    }
                )
            evaluations.append(
                {
                    "layer": layer,
                    "target_tensor": target_tensor,
                    "activation_tensor": activation_tensor,
                    "activation_policy": candidate["activation_policy"],
                    "group": group,
                    "start_col": start,
                    "end_col": end,
                    "source_sha256": _tensor_hash(source_group),
                    "hqq_dequant_sha256": _tensor_hash(cached_group),
                    "source_contract": copy.deepcopy(entry.get("source_contract"))
                    if source_contract_lookup
                    else None,
                    "source_contract_validation": source_contract_validations.get(key)
                    if source_contract_lookup
                    else None,
                    "activation_rows": int(input_rows.shape[0]),
                    "selected_output_row_count": len(selected_rows),
                    "full_group_cached_projection": _projection_metrics(input_rows, cached_delta.T),
                    "selected_output_rows": selected_rows,
                }
            )
        except Exception as exc:
            skipped.append({"candidate": candidate, "reason": str(exc)})

    aggregate = _aggregate_eval_candidates(evaluations)
    improvement = _improvement_vs_cached(aggregate)
    payload = {
        "format": REFIT_EVAL_FORMAT,
        "format_version": REFIT_EVAL_FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "refit_evaluation_built",
        "complete": False,
        "loadable_hqq_manifest": False,
        "not_loadable_reason": (
            "Phase 2C refit evaluation only. No calibrated HQQ tensor artifacts, qvalues, "
            "scales, zeros, or complete manifest were written."
        ),
        "model_path": model_path,
        "source_candidates_path": candidates_path,
        "source_trace_path": trace_path,
        "cache": {
            "source_cache_profile": source_profile,
            "source_cache_dir": cache_dir,
            "target_cache_profile": HQQ_CACHE_PROFILE_SELFCAL_V1,
            "target_cache_dir": hqq_attention_cache_dir(model_path, HQQ_CACHE_PROFILE_SELFCAL_V1),
            "target_manifest_path": hqq_attention_manifest_path(model_path, HQQ_CACHE_PROFILE_SELFCAL_V1),
            "target_manifest_written": False,
            "refit_evaluation_path": output_path,
            "source_contract_audit_path": source_contract_audit_path,
            "source_contract_audit_sha256": _file_sha256(source_contract_audit_path)
            if source_contract_audit_path
            else None,
        },
        "refit_evaluation_model": {
            "purpose": "evaluate whether candidate affine refits reduce local projection error before any cache writer exists",
            "scope": "bounded top candidates and top cached-error output rows only",
            "activation_policy_filter": activation_policy,
            "max_output_rows_per_candidate": max_output_rows,
            "grid_steps": grid_steps,
            "local_grid_steps": local_grid_steps,
            "caveat": (
                "This artifact is offline local-error evidence. It does not prove end-to-end witness improvement "
                "until a calibrated cache variant is written and compared."
            ),
        },
        "summary": {
            "requested_top_candidates": top_candidates,
            "evaluated_candidate_count": len(evaluations),
            "skipped_count": len(skipped),
            "selected_output_row_total": sum(int(e.get("selected_output_row_count", 0)) for e in evaluations),
            "aggregate_by_candidate": aggregate,
            "improvement_vs_cached": improvement,
            "source_contract": _source_contract_summary(source_contract_validations.values())
            if source_contract_lookup
            else None,
        },
        "evaluations": evaluations,
        "skipped": skipped,
    }
    write_json(output_path, payload)
    latest_path = os.path.join(os.path.dirname(output_path), "refit_evaluation_latest.json")
    write_json(latest_path, payload)
    return payload


def _csv_filter(value: Optional[str]) -> Optional[set]:
    if value is None:
        return None
    parts = {part.strip() for part in value.split(",") if part.strip()}
    if not parts or parts == {"all"}:
        return None
    return parts


def _select_refit_patches(
    refit_eval: Dict[str, Any],
    *,
    allowed_tensors: Optional[set] = None,
    allowed_methods: Optional[set] = None,
    top_groups: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    patches: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    seen = set()
    eligible_group_count = 0
    selected_group_keys = []
    for evaluation in refit_eval.get("evaluations", []) or []:
        layer = int(evaluation["layer"])
        tensor = str(evaluation["target_tensor"])
        group = int(evaluation["group"])
        start_col = int(evaluation["start_col"])
        end_col = int(evaluation["end_col"])
        activation_tensor = str(evaluation["activation_tensor"])
        activation_policy = str(evaluation["activation_policy"])
        group_key = {"layer": layer, "tensor": tensor, "group": group}
        if allowed_tensors is not None and tensor not in allowed_tensors:
            skipped.append({**group_key, "reason": "tensor excluded by ablation filter"})
            continue
        if top_groups is not None and eligible_group_count >= top_groups:
            skipped.append({**group_key, "reason": "group excluded by ablation top-N filter"})
            continue
        eligible_group_count += 1
        selected_group_keys.append(group_key)
        for row in evaluation.get("selected_output_rows", []) or []:
            output_row = int(row["output_row"])
            best_name = str(row.get("best_candidate_by_projection_rms", ""))
            candidates = {str(c.get("name")): c for c in row.get("candidates", []) or []}
            candidate = candidates.get(best_name)
            if candidate is None:
                skipped.append(
                    {
                        "layer": layer,
                        "tensor": tensor,
                        "group": group,
                        "output_row": output_row,
                        "reason": f"best candidate {best_name!r} missing from row candidates",
                    }
                )
                continue
            if allowed_methods is not None and best_name not in allowed_methods:
                skipped.append(
                    {
                        "layer": layer,
                        "tensor": tensor,
                        "group": group,
                        "output_row": output_row,
                        "method": best_name,
                        "reason": "method excluded by ablation filter",
                    }
                )
                continue
            if best_name == "cached_hqq":
                skipped.append(
                    {
                        "layer": layer,
                        "tensor": tensor,
                        "group": group,
                        "output_row": output_row,
                        "reason": "cached_hqq already best; no patch written",
                    }
                )
                continue
            key = (layer, tensor, group, output_row)
            if key in seen:
                skipped.append(
                    {
                        "layer": layer,
                        "tensor": tensor,
                        "group": group,
                        "output_row": output_row,
                        "reason": "duplicate evaluated row/group; first patch kept",
                    }
                )
                continue
            seen.add(key)
            patches.append(
                {
                    "layer": layer,
                    "tensor": tensor,
                    "activation_tensor": activation_tensor,
                    "activation_policy": activation_policy,
                    "group": group,
                    "start_col": start_col,
                    "end_col": end_col,
                    "output_row": output_row,
                    "method": best_name,
                    "scale": float(candidate["scale"]),
                    "zero": float(candidate["zero"]),
                    "best_vs_cached_projection_rms_ratio": float(
                        row.get("best_vs_cached_projection_rms_ratio", 0.0)
                    ),
                    "cached_projection_rms_rank_score": float(
                        row.get("cached_projection_rms_rank_score", 0.0)
                    ),
                    "projection": candidate.get("projection", {}),
                    "weight_error_rms": float(candidate.get("weight_error_rms", 0.0)),
                    "weight_error_max_abs": float(candidate.get("weight_error_max_abs", 0.0)),
                }
            )
    selection = {
        "allowed_tensors": sorted(allowed_tensors) if allowed_tensors is not None else ["all"],
        "allowed_methods": sorted(allowed_methods) if allowed_methods is not None else ["all"],
        "top_groups": top_groups,
        "eligible_group_count": eligible_group_count,
        "selected_group_count": len(selected_group_keys),
        "selected_groups": selected_group_keys,
    }
    return patches, skipped, selection


def _copy_baseline_hqq_cache(
    *,
    baseline_manifest: Dict[str, Any],
    baseline_cache_dir: str,
    target_cache_dir: str,
) -> Dict[str, Any]:
    os.makedirs(target_cache_dir, exist_ok=True)
    for stale_name in ("manifest.json", "manifest.build.json"):
        stale_path = os.path.join(target_cache_dir, stale_name)
        if os.path.exists(stale_path):
            os.remove(stale_path)
    copied = []
    for entry in baseline_manifest.get("tensors", []) or []:
        rel = str(entry["file"])
        src = os.path.join(baseline_cache_dir, rel)
        dst = os.path.join(target_cache_dir, rel)
        if not os.path.isfile(src):
            raise RuntimeError(f"Baseline HQQ artifact missing: {src}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({"file": rel, "sha256": _file_sha256(dst), "bytes": os.path.getsize(dst)})
    return {
        "artifact_count": len(copied),
        "artifacts": copied,
    }


def _load_safetensors_payload(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        tensors = {key: handle.get_tensor(key).contiguous() for key in handle.keys()}
    return tensors, metadata


def _patch_hqq_artifact(
    *,
    model_path: str,
    entry: Dict[str, Any],
    target_cache_dir: str,
    patches: List[Dict[str, Any]],
) -> Dict[str, Any]:
    path = os.path.join(target_cache_dir, entry["file"])
    tensors, metadata = _load_safetensors_payload(path)
    orig_shape = tensors["orig_shape"]
    rows = int(orig_shape[0].item())
    cols = int(orig_shape[1].item())
    group_size = int(tensors["group_size"][0].item())
    qvalues = _unpack_uint4(tensors["packed"].to(torch.uint8), cols)
    scales = tensors["scales"].to(torch.float32).contiguous()
    zeros = tensors["zeros"].to(torch.float32).contiguous()
    source = _load_source_tensor(model_path, entry).to(torch.float32).contiguous()
    if list(source.shape) != [rows, cols]:
        raise RuntimeError(
            f"source/HQQ shape mismatch for experimental patch layer={entry.get('layer_idx')} "
            f"tensor={entry.get('tensor_name')}: {list(source.shape)} vs {[rows, cols]}"
        )

    applied = []
    for patch in patches:
        output_row = int(patch["output_row"])
        group = int(patch["group"])
        start = group * group_size
        end = min(start + group_size, cols)
        if output_row < 0 or output_row >= rows:
            raise RuntimeError(f"Patch output row out of range: {patch}")
        if int(patch["start_col"]) != start or int(patch["end_col"]) != end:
            raise RuntimeError(f"Patch group bounds do not match artifact shape: {patch}")
        scale = float(patch["scale"])
        zero = float(patch["zero"])
        src = source[output_row, start:end]
        q, _, _ = _quantize_affine(src, scale, zero)
        qvalues[output_row, start:end] = q
        scales[output_row, group] = scale
        zeros[output_row, group] = zero
        applied.append(
            {
                "layer": int(patch["layer"]),
                "tensor": str(patch["tensor"]),
                "group": group,
                "output_row": output_row,
                "start_col": start,
                "end_col": end,
                "method": str(patch["method"]),
                "scale": scale,
                "zero": zero,
                "activation_policy": str(patch["activation_policy"]),
                "best_vs_cached_projection_rms_ratio": float(
                    patch["best_vs_cached_projection_rms_ratio"]
                ),
            }
        )

    payload = dict(tensors)
    payload["packed"] = _pack_uint4(qvalues, padded_cols=scales.shape[1] * group_size)
    payload["scales"] = scales
    payload["zeros"] = zeros
    tmp = path + ".tmp"
    save_file(payload, tmp, metadata=metadata)
    os.replace(tmp, path)
    return {
        "layer": int(entry["layer_idx"]),
        "tensor": str(entry["tensor_name"]),
        "file": str(entry["file"]),
        "patched_rows": applied,
        "tensor_bytes": _artifact_tensor_bytes(path),
        "sha256": _file_sha256(path),
    }


def write_experimental_cache(
    *,
    cfg: Dict[str, str],
    refit_eval_path: str,
    output_path: str,
    variant_name: str,
    patch_tensors: Optional[str],
    patch_methods: Optional[str],
    patch_top_groups: Optional[int],
) -> Dict[str, Any]:
    with open(refit_eval_path, encoding="utf-8") as f:
        refit_eval = json.load(f)
    if refit_eval.get("format") != REFIT_EVAL_FORMAT:
        raise ValueError(f"Not an HQQ self-calibration refit-evaluation file: {refit_eval_path}")

    model_path = os.path.expanduser(cfg.get("CFG_MODEL_PATH", refit_eval.get("model_path", "")))
    if not model_path:
        raise ValueError("Cannot determine model path for experimental cache write")
    baseline_manifest_path = hqq_attention_manifest_path(model_path, HQQ_CACHE_PROFILE_BASELINE)
    baseline_manifest = load_hqq_attention_manifest(model_path, HQQ_CACHE_PROFILE_BASELINE)
    if baseline_manifest is None:
        raise RuntimeError(f"Missing baseline HQQ manifest: {baseline_manifest_path}")
    expected_layers = int(baseline_manifest.get("num_hidden_layers", 0))
    baseline_manifest = require_complete_hqq_attention_manifest(
        model_path,
        HQQ_CACHE_PROFILE_BASELINE,
        expected_nbits=4,
        expected_num_hidden_layers=expected_layers,
    )
    baseline_cache_dir = hqq_attention_cache_dir(model_path, HQQ_CACHE_PROFILE_BASELINE)
    target_cache_dir = hqq_attention_cache_dir(model_path, HQQ_CACHE_PROFILE_SELFCAL_V1)
    target_manifest_path = hqq_attention_manifest_path(model_path, HQQ_CACHE_PROFILE_SELFCAL_V1)

    allowed_tensors = _csv_filter(patch_tensors)
    allowed_methods = _csv_filter(patch_methods)
    patches, skipped, selection = _select_refit_patches(
        refit_eval,
        allowed_tensors=allowed_tensors,
        allowed_methods=allowed_methods,
        top_groups=patch_top_groups,
    )
    if not patches:
        raise RuntimeError("Refit evaluation did not contain any non-cached best refit patches")

    copy_summary = _copy_baseline_hqq_cache(
        baseline_manifest=baseline_manifest,
        baseline_cache_dir=baseline_cache_dir,
        target_cache_dir=target_cache_dir,
    )
    entries = _manifest_tensor_entries(baseline_manifest)
    patches_by_tensor: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for patch in patches:
        patches_by_tensor.setdefault((int(patch["layer"]), str(patch["tensor"])), []).append(patch)

    patched_artifacts = []
    for key in sorted(patches_by_tensor):
        entry = entries.get(key)
        if entry is None:
            raise RuntimeError(f"Refit patch references tensor absent from baseline manifest: {key}")
        patched_artifacts.append(
            _patch_hqq_artifact(
                model_path=model_path,
                entry=entry,
                target_cache_dir=target_cache_dir,
                patches=patches_by_tensor[key],
            )
        )

    manifest = copy.deepcopy(baseline_manifest)
    total_bytes = 0
    patch_lookup = {(int(a["layer"]), str(a["tensor"])): a for a in patched_artifacts}
    for entry in manifest.get("tensors", []) or []:
        key = (int(entry["layer_idx"]), str(entry["tensor_name"]))
        entry["path"] = os.path.join(target_cache_dir, entry["file"])
        actual_path = entry["path"]
        entry["tensor_bytes"] = _artifact_tensor_bytes(actual_path)
        total_bytes += int(entry["tensor_bytes"])
        if key in patch_lookup:
            entry["calibration_patch"] = {
                "profile": HQQ_CACHE_PROFILE_SELFCAL_V1,
                "status": "experimental",
                "patched_row_count": len(patch_lookup[key]["patched_rows"]),
                "patched_rows": patch_lookup[key]["patched_rows"],
                "artifact_sha256": patch_lookup[key]["sha256"],
            }
    manifest["cache_profile"] = HQQ_CACHE_PROFILE_SELFCAL_V1
    manifest["complete"] = True
    manifest["totals"] = {
        **dict(manifest.get("totals", {})),
        "tensor_bytes": total_bytes,
        "num_tensors": len(manifest.get("tensors", []) or []),
    }
    manifest["calibration"] = {
        "profile": HQQ_CACHE_PROFILE_SELFCAL_V1,
        "method": "selfcal_v1",
        "status": "experimental",
        "variant_name": variant_name,
        "created_at": _utc_now(),
        "source_cache_profile": HQQ_CACHE_PROFILE_BASELINE,
        "source_baseline_manifest_path": baseline_manifest_path,
        "source_baseline_manifest_sha256": _file_sha256(baseline_manifest_path),
        "source_refit_evaluation_path": refit_eval_path,
        "source_refit_evaluation_sha256": _file_sha256(refit_eval_path),
        "ablation_selection": selection,
        "patched_tensor_count": len(patched_artifacts),
        "patched_row_group_count": sum(len(a["patched_rows"]) for a in patched_artifacts),
        "patched_tensors": [
            {
                "file": item["file"],
                "sha256": item["sha256"],
                "patched_row_count": len(item["patched_rows"]),
                "methods": sorted({row["method"] for row in item["patched_rows"]}),
                "patched_rows": item["patched_rows"],
            }
            for item in patched_artifacts
        ],
        "skipped_patches": skipped,
        "load_time_selection": (
            "Only loaded when CFG_HQQ_CACHE_PROFILE/--hqq-cache-profile selfcal_v1 is explicitly selected. "
            "No fallback to baseline is allowed."
        ),
    }
    tmp_manifest = target_manifest_path + ".tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_manifest, target_manifest_path)

    require_complete_hqq_attention_manifest(
        model_path,
        HQQ_CACHE_PROFILE_SELFCAL_V1,
        expected_nbits=4,
        expected_num_hidden_layers=expected_layers,
    )
    report = {
        "format": EXPERIMENTAL_CACHE_WRITE_FORMAT,
        "format_version": EXPERIMENTAL_CACHE_WRITE_FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "experimental_cache_written",
        "complete": True,
        "loadable_hqq_manifest": True,
        "model_path": model_path,
        "source_refit_evaluation_path": refit_eval_path,
        "variant_name": variant_name,
        "cache": {
            "source_cache_profile": HQQ_CACHE_PROFILE_BASELINE,
            "source_cache_dir": baseline_cache_dir,
            "target_cache_profile": HQQ_CACHE_PROFILE_SELFCAL_V1,
            "target_cache_dir": target_cache_dir,
            "target_manifest_path": target_manifest_path,
            "target_manifest_written": True,
            "experimental_cache_write_path": output_path,
        },
        "copy_summary": {
            "artifact_count": copy_summary["artifact_count"],
        },
        "manifest_metadata": manifest["calibration"],
        "ablation_selection": selection,
        "summary": {
            "variant_name": variant_name,
            "patched_tensor_count": len(patched_artifacts),
            "patched_row_group_count": sum(len(a["patched_rows"]) for a in patched_artifacts),
            "skipped_patch_count": len(skipped),
            "baseline_manifest_sha256": manifest["calibration"]["source_baseline_manifest_sha256"],
            "refit_evaluation_sha256": manifest["calibration"]["source_refit_evaluation_sha256"],
        },
        "patched_artifacts": patched_artifacts,
        "skipped_patches": skipped,
    }
    write_json(output_path, report)
    latest_path = os.path.join(os.path.dirname(output_path), "experimental_cache_write_latest.json")
    write_json(latest_path, report)
    return report


def _summary_case_key(item: Dict[str, Any]) -> str:
    return f"conv{item.get('conv_idx')}_turn{item.get('turn')}"


def _witness_summary_metrics(summary_path: str) -> Dict[str, Any]:
    with open(summary_path, encoding="utf-8") as f:
        doc = json.load(f)
    cases = []
    for item in doc.get("prompt_results", []) or []:
        diag = item.get("metrics", {}).get("first_token_diagnostic", {}) or {}
        cases.append(
            {
                "case_key": _summary_case_key(item),
                "conv_idx": item.get("conv_idx"),
                "turn": item.get("turn"),
                "verdict": item.get("verdict"),
                "first_token_match": bool(diag.get("first_token_match")),
                "ref_first_token": diag.get("ref_first_token"),
                "our_first_token": diag.get("our_first_token"),
                "top_overlap_count": int(diag.get("top_overlap_count", 0)),
                "selected_logprob_delta": float(diag.get("selected_logprob_delta", 0.0)),
            }
        )
    return {
        "summary_path": summary_path,
        "overall": doc.get("overall"),
        "case_count": len(cases),
        "selected_logprob_delta_sum": sum(c["selected_logprob_delta"] for c in cases),
        "first_token_match_all": all(c["first_token_match"] for c in cases) if cases else False,
        "top_overlap_sum": sum(c["top_overlap_count"] for c in cases),
        "top_overlap_min": min((c["top_overlap_count"] for c in cases), default=0),
        "cases": cases,
    }


def _write_ablation_tsv(path: str, rows: List[Dict[str, Any]]) -> None:
    header = [
        "variant",
        "accepted",
        "overall",
        "patched_tensors",
        "patched_row_groups",
        "delta_sum",
        "delta_vs_baseline",
        "first_token_match_all",
        "top_overlap_sum",
        "top_overlap_min",
        "per_case_regressions",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write(
                "\t".join(
                    [
                        str(row["variant_name"]),
                        str(row["accepted"]),
                        str(row["overall"]),
                        str(row["patched_tensor_count"]),
                        str(row["patched_row_group_count"]),
                        f'{float(row["selected_logprob_delta_sum"]):.12g}',
                        f'{float(row["selected_logprob_delta_vs_baseline"]):.12g}',
                        str(row["first_token_match_all"]),
                        str(row["top_overlap_sum"]),
                        str(row["top_overlap_min"]),
                        str(row["per_case_regression_count"]),
                    ]
                )
                + "\n"
            )


def build_ablation_report(*, spec_path: str, output_path: str) -> Dict[str, Any]:
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    baseline_spec = spec.get("baseline", {})
    baseline = _witness_summary_metrics(str(baseline_spec["summary_path"]))
    baseline_cases = {case["case_key"]: case for case in baseline["cases"]}
    variants = []
    table_rows = []
    for variant_spec in spec.get("variants", []) or []:
        name = str(variant_spec["name"])
        metrics = _witness_summary_metrics(str(variant_spec["summary_path"]))
        write_report_path = variant_spec.get("write_report_path")
        patched_tensor_count = 0
        patched_row_group_count = 0
        write_report = None
        if write_report_path:
            with open(write_report_path, encoding="utf-8") as f:
                write_report = json.load(f)
            patched_tensor_count = int(write_report.get("summary", {}).get("patched_tensor_count", 0))
            patched_row_group_count = int(write_report.get("summary", {}).get("patched_row_group_count", 0))

        per_case = []
        regressions = []
        for case in metrics["cases"]:
            base_case = baseline_cases.get(case["case_key"])
            if base_case is None:
                regressions.append(
                    {
                        "case_key": case["case_key"],
                        "reason": "case missing from baseline",
                    }
                )
                continue
            delta_vs_baseline = case["selected_logprob_delta"] - base_case["selected_logprob_delta"]
            row = {
                **case,
                "baseline_selected_logprob_delta": base_case["selected_logprob_delta"],
                "selected_logprob_delta_vs_baseline": delta_vs_baseline,
                "regressed_vs_baseline": delta_vs_baseline > 0.0,
            }
            per_case.append(row)
            if row["regressed_vs_baseline"]:
                regressions.append(row)

        aggregate_delta_vs_baseline = (
            metrics["selected_logprob_delta_sum"] - baseline["selected_logprob_delta_sum"]
        )
        accepted = (
            metrics["overall"] == "PASS"
            and metrics["first_token_match_all"]
            and aggregate_delta_vs_baseline < 0.0
            and len(regressions) == 0
        )
        variant = {
            "variant_name": name,
            "summary_path": metrics["summary_path"],
            "write_report_path": write_report_path,
            "overall": metrics["overall"],
            "accepted": accepted,
            "acceptance": {
                "token_identity_preserved": metrics["first_token_match_all"],
                "aggregate_selected_logprob_delta_lower_than_baseline": aggregate_delta_vs_baseline < 0.0,
                "no_per_case_selected_logprob_regression": len(regressions) == 0,
                "strict_dominance_required": True,
            },
            "patched_tensor_count": patched_tensor_count,
            "patched_row_group_count": patched_row_group_count,
            "selected_logprob_delta_sum": metrics["selected_logprob_delta_sum"],
            "selected_logprob_delta_vs_baseline": aggregate_delta_vs_baseline,
            "first_token_match_all": metrics["first_token_match_all"],
            "top_overlap_sum": metrics["top_overlap_sum"],
            "top_overlap_min": metrics["top_overlap_min"],
            "per_case_regression_count": len(regressions),
            "per_case_regressions": regressions,
            "per_case": per_case,
            "write_report": write_report,
        }
        variants.append(variant)
        table_rows.append(
            {
                "variant_name": name,
                "accepted": accepted,
                "overall": metrics["overall"],
                "patched_tensor_count": patched_tensor_count,
                "patched_row_group_count": patched_row_group_count,
                "selected_logprob_delta_sum": metrics["selected_logprob_delta_sum"],
                "selected_logprob_delta_vs_baseline": aggregate_delta_vs_baseline,
                "first_token_match_all": metrics["first_token_match_all"],
                "top_overlap_sum": metrics["top_overlap_sum"],
                "top_overlap_min": metrics["top_overlap_min"],
                "per_case_regression_count": len(regressions),
            }
        )

    report = {
        "format": ABLATION_REPORT_FORMAT,
        "format_version": ABLATION_REPORT_FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "ablation_report_built",
        "acceptance_rule": {
            "name": "strict_dominance_v1",
            "requirements": [
                "first-token identity preserved for every case",
                "aggregate selected-logprob delta lower than unchanged baseline",
                "no per-case selected-logprob regression versus unchanged baseline",
            ],
        },
        "baseline": {
            "name": baseline_spec.get("name", "baseline"),
            **baseline,
        },
        "variant_count": len(variants),
        "accepted_variant_count": sum(1 for item in variants if item["accepted"]),
        "table": table_rows,
        "variants": variants,
    }
    write_json(output_path, report)
    tsv_path = os.path.splitext(output_path)[0] + ".tsv"
    _write_ablation_tsv(tsv_path, table_rows)
    return report


def _case_change(row: Dict[str, Any]) -> float:
    for key in (
        "change_vs_baseline",
        "delta_vs_baseline",
        "selected_logprob_delta_change",
        "selected_logprob_delta_vs_baseline",
    ):
        if key in row:
            return float(row[key])
    if "variant_selected_logprob_delta" in row and "baseline_selected_logprob_delta" in row:
        return float(row["variant_selected_logprob_delta"]) - float(row["baseline_selected_logprob_delta"])
    if "selected_logprob_delta" in row and "baseline_selected_logprob_delta" in row:
        return float(row["selected_logprob_delta"]) - float(row["baseline_selected_logprob_delta"])
    raise ValueError(f"Cannot determine per-case baseline change from row keys={sorted(row.keys())}")


def _case_key_from_report_row(row: Dict[str, Any]) -> str:
    conv = row.get("conv_idx")
    turn = row.get("turn")
    if conv is not None and turn is not None:
        return f"conv{conv}_turn{turn}"
    if "case_key" in row:
        return str(row["case_key"])
    return f"case{row.get('case_index')}"


def _variant_case_effects(variant: Dict[str, Any]) -> Dict[str, float]:
    return {_case_key_from_report_row(row): _case_change(row) for row in variant.get("per_case", []) or []}


def _entry_tuple_from_dict(entry: Dict[str, Any]) -> Tuple[int, str, int, int]:
    return (
        int(entry["layer"]),
        str(entry["tensor"]),
        int(entry["group"]),
        int(entry["output_row"]),
    )


def _entries_from_variant(variant: Dict[str, Any]) -> List[Tuple[int, str, int, int]]:
    return [_entry_tuple_from_dict(entry) for entry in variant.get("entries", []) or []]


def _variant_metric(variant: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in variant and variant[key] is not None:
            return float(variant[key])
    return default


def _unit_from_variant(
    variant: Dict[str, Any],
    *,
    source_report: str,
    source_kind: str,
) -> Optional[Dict[str, Any]]:
    entries = _entries_from_variant(variant)
    if not entries:
        return None
    scale = float(variant.get("correction_scale", 1.0))
    name = str(variant.get("variant") or variant.get("variant_name"))
    return {
        "name": name,
        "source_report": source_report,
        "source_kind": source_kind,
        "scope": variant.get("scope"),
        "layer": variant.get("layer"),
        "tensor": variant.get("tensor"),
        "group": variant.get("group"),
        "output_row": variant.get("output_row"),
        "correction_scale": scale,
        "entries": [_sidecar_entry_key_to_str(entry) for entry in entries],
        "entry_keys": entries,
        "case_effects": _variant_case_effects(variant),
        "aggregate_delta_vs_baseline": _variant_metric(
            variant,
            "aggregate_delta_vs_baseline",
            "selected_logprob_delta_sum_delta_vs_baseline",
            "selected_logprob_delta_sum_vs_baseline",
        ),
        "per_case_regression_count": int(variant.get("per_case_regression_count", 0)),
        "case9_delta_vs_baseline": _variant_metric(variant, "case9_delta_vs_baseline"),
        "qkvz_case2_delta_vs_baseline": _variant_metric(variant, "qkvz_case2_delta_vs_baseline"),
    }


def _attach_loo_hints(units: List[Dict[str, Any]], report_paths: Iterable[str]) -> None:
    by_row: Dict[Tuple[int, str, int, int], Dict[str, Any]] = {}
    for unit in units:
        if len(unit["entry_keys"]) == 1:
            by_row[unit["entry_keys"][0]] = unit
    for path in report_paths:
        if not path or not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        for variant in doc.get("variants", []) or []:
            excluded = variant.get("excluded_entry")
            if not excluded:
                continue
            key = _entry_tuple_from_dict(excluded)
            unit = by_row.get(key)
            if unit is None:
                continue
            effects = _variant_case_effects(variant)
            case9_effect = effects.get("conv8_turn1", effects.get("case8"))
            unit["loo_excluding_entry_variant"] = variant.get("variant")
            unit["loo_excluding_entry_case9_delta_vs_baseline"] = case9_effect


def _candidate_from_units(
    units: Tuple[Dict[str, Any], ...],
    *,
    baseline_cases: List[str],
    regression_tolerance: float,
    candidate_index: int,
) -> Optional[Dict[str, Any]]:
    entry_scales: Dict[Tuple[int, str, int, int], float] = {}
    entries: List[str] = []
    for unit in units:
        for key in unit["entry_keys"]:
            scale = float(unit["correction_scale"])
            if key in entry_scales and abs(entry_scales[key] - scale) > 1e-12:
                return None
            entry_scales[key] = scale
    for key in sorted(entry_scales):
        entries.append(_sidecar_entry_key_to_str(key))

    case_effects = {}
    regressions = []
    for case_key in baseline_cases:
        value = sum(float(unit["case_effects"].get(case_key, 0.0)) for unit in units)
        case_effects[case_key] = value
        if value > regression_tolerance:
            regressions.append({"case_key": case_key, "predicted_delta_vs_baseline": value})
    aggregate = sum(case_effects.values())
    worst = max((item["predicted_delta_vs_baseline"] for item in regressions), default=0.0)
    case9 = case_effects.get("conv8_turn1", case_effects.get("case8", 0.0))
    qkvz_case2 = case_effects.get("conv1_turn1", case_effects.get("case1", 0.0))
    accepted = aggregate < 0.0 and not regressions
    label = "phase2l_search_%02d" % candidate_index
    return {
        "candidate_name": label,
        "unit_names": [unit["name"] for unit in units],
        "unit_count": len(units),
        "entry_count": len(entries),
        "entries": entries,
        "entry_scales": [
            {"entry": _sidecar_entry_key_to_str(key), "correction_scale": entry_scales[key]}
            for key in sorted(entry_scales)
        ],
        "include_entries_arg": ",".join(entries),
        "entry_scales_arg": ",".join(
            f"{_sidecar_entry_key_to_str(key)}={entry_scales[key]:.12g}" for key in sorted(entry_scales)
        ),
        "predicted_selected_logprob_delta_change": aggregate,
        "predicted_per_case_regression_count": len(regressions),
        "predicted_worst_per_case_regression": worst,
        "predicted_case9_delta_vs_baseline": case9,
        "predicted_qkvz_case2_delta_vs_baseline": qkvz_case2,
        "predicted_accepted_strict": accepted,
        "predicted_per_case": [
            {
                "case_key": key,
                "predicted_delta_vs_baseline": case_effects[key],
                "regressed_vs_baseline": case_effects[key] > regression_tolerance,
            }
            for key in baseline_cases
        ],
    }


def _write_conflict_search_tsv(path: str, candidates: List[Dict[str, Any]]) -> None:
    header = [
        "candidate",
        "predicted_accepted",
        "units",
        "entries",
        "predicted_delta",
        "predicted_regressions",
        "predicted_worst",
        "case9_delta",
        "qkvz_case2_delta",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for candidate in candidates:
            f.write(
                "\t".join(
                    [
                        candidate["candidate_name"],
                        str(candidate["predicted_accepted_strict"]),
                        ",".join(candidate["unit_names"]),
                        str(candidate["entry_count"]),
                        f'{candidate["predicted_selected_logprob_delta_change"]:.12g}',
                        str(candidate["predicted_per_case_regression_count"]),
                        f'{candidate["predicted_worst_per_case_regression"]:.12g}',
                        f'{candidate["predicted_case9_delta_vs_baseline"]:.12g}',
                        f'{candidate["predicted_qkvz_case2_delta_vs_baseline"]:.12g}',
                    ]
                )
                + "\n"
            )


def build_sidecar_conflict_search(*, spec_path: str, output_path: str) -> Dict[str, Any]:
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    report_paths = spec.get("reports", {}) or {}
    singleton_paths = []
    for key in ("singleton075", "singleton05"):
        path = report_paths.get(key)
        if path:
            singleton_paths.append((key, str(path)))

    units: List[Dict[str, Any]] = []
    baseline = None
    baseline_cases: List[str] = []
    for source_kind, path in singleton_paths:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if baseline is None:
            baseline = doc.get("baseline", {})
        for variant in doc.get("variants", []) or []:
            unit = _unit_from_variant(variant, source_report=path, source_kind=source_kind)
            if unit is not None:
                units.append(unit)
                if not baseline_cases:
                    baseline_cases = list(unit["case_effects"].keys())
    if not units:
        raise ValueError("Conflict search spec did not provide any singleton units with entries")
    _attach_loo_hints(
        units,
        [
            str(path)
            for key, path in sorted(report_paths.items())
            if "loo" in key.lower() or "attribution" in key.lower()
        ],
    )

    max_units = int(spec.get("max_units", 3))
    min_units = int(spec.get("min_units", 2))
    max_row_groups = int(spec.get("max_row_groups", 32))
    max_candidates = int(spec.get("max_candidates", 4))
    regression_tolerance = float(spec.get("regression_tolerance", 0.0))

    raw_candidates: List[Dict[str, Any]] = []
    candidate_counter = 1
    for size in range(min_units, max_units + 1):
        for combo in itertools.combinations(units, size):
            if sum(len(unit["entry_keys"]) for unit in combo) > max_row_groups:
                continue
            candidate = _candidate_from_units(
                combo,
                baseline_cases=baseline_cases,
                regression_tolerance=regression_tolerance,
                candidate_index=candidate_counter,
            )
            if candidate is None:
                continue
            candidate_counter += 1
            raw_candidates.append(candidate)

    def sort_key(candidate: Dict[str, Any]) -> Tuple[int, float, float, int]:
        return (
            int(candidate["predicted_per_case_regression_count"]),
            float(candidate["predicted_worst_per_case_regression"]),
            float(candidate["predicted_selected_logprob_delta_change"]),
            int(candidate["entry_count"]),
        )

    raw_candidates.sort(key=sort_key)
    selected = raw_candidates[:max_candidates]
    for idx, candidate in enumerate(selected, 1):
        candidate["candidate_name"] = f"phase2l_search_{idx:02d}"

    report = {
        "format": SIDECAR_CONFLICT_SEARCH_FORMAT,
        "format_version": SIDECAR_CONFLICT_SEARCH_FORMAT_VERSION,
        "created_at": _utc_now(),
        "status": "sidecar_conflict_search_built",
        "objective": {
            "name": "strict_per_case_non_regression_then_aggregate_gain",
            "regression_tolerance": regression_tolerance,
            "requirements": [
                "zero per-case selected-logprob regression versus unchanged baseline first",
                "aggregate selected-logprob delta improvement second",
                "case 9 and qkvz case 2 tracked explicitly",
            ],
        },
        "input_spec_path": spec_path,
        "input_reports": report_paths,
        "baseline": baseline,
        "search": {
            "unit_count": len(units),
            "raw_candidate_count": len(raw_candidates),
            "selected_candidate_count": len(selected),
            "min_units": min_units,
            "max_units": max_units,
            "max_row_groups": max_row_groups,
            "max_candidates": max_candidates,
            "model": "additive per-case witness-effect model from singleton and scaled singleton reports, with LOO reports attached as conflict context",
        },
        "units": [
            {
                key: value
                for key, value in unit.items()
                if key not in {"entry_keys", "case_effects"}
            }
            for unit in units
        ],
        "candidates": selected,
    }
    write_json(output_path, report)
    _write_conflict_search_tsv(os.path.splitext(output_path)[0] + ".tsv", selected)
    return report


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = parse_config(args.config)
    model_path = os.path.expanduser(cfg.get("CFG_MODEL_PATH", ""))
    if not model_path:
        raise ValueError(f"Config has no MODEL_PATH/CFG_MODEL_PATH: {args.config}")
    attention_quant = cfg.get("CFG_ATTENTION_QUANT", "bf16")
    if attention_quant != "hqq4":
        raise ValueError(
            f"hqq-self-calibrate requires CFG_ATTENTION_QUANT=\"hqq4\"; found {attention_quant!r}"
        )
    source_profile = cfg.get("CFG_HQQ_CACHE_PROFILE", HQQ_CACHE_PROFILE_BASELINE) or HQQ_CACHE_PROFILE_BASELINE
    source_profile = source_profile.strip().lower()
    if source_profile not in HQQ_CACHE_PROFILE_CHOICES:
        raise ValueError(f"Unsupported source HQQ cache profile: {source_profile}")
    target_profile = args.profile.strip().lower()
    if target_profile != HQQ_CACHE_PROFILE_SELFCAL_V1:
        raise ValueError("Phase 2A only supports --profile selfcal_v1")
    if source_profile != HQQ_CACHE_PROFILE_BASELINE:
        expected_layers = model_num_hidden_layers(model_path)
        require_complete_hqq_attention_manifest(
            model_path,
            source_profile,
            expected_nbits=4,
            expected_num_hidden_layers=expected_layers,
        )

    if args.build_candidates:
        output_path = args.output or candidates_default_path(model_path, target_profile)
        return build_sensitivity_candidates(
            cfg=cfg,
            evidence_path=args.build_candidates,
            output_path=output_path,
            top_candidates=args.top_candidates,
            source_contract_audit_path=args.source_contract_audit,
        )
    if args.build_int8_exception_candidates:
        if not args.heldout_evidence:
            raise ValueError("--build-int8-exception-candidates requires --heldout-evidence")
        output_path = args.output or int8_exception_candidates_default_path(model_path, target_profile)
        return build_int8_exception_candidates(
            cfg=cfg,
            train_evidence_path=args.build_int8_exception_candidates,
            heldout_evidence_path=args.heldout_evidence,
            output_path=output_path,
            top_candidates=args.top_candidates,
            patch_tensors=args.patch_tensors,
            source_contract_audit_path=args.source_contract_audit,
        )
    if args.evaluate_refits:
        if not args.trace_log:
            raise ValueError("--evaluate-refits requires --trace-log with the activation rows used for evaluation")
        output_path = args.output or refit_eval_default_path(model_path, target_profile)
        return evaluate_refit_candidates(
            cfg=cfg,
            candidates_path=args.evaluate_refits,
            trace_path=args.trace_log,
            output_path=output_path,
            top_candidates=args.top_candidates,
            max_output_rows=args.max_output_rows,
            grid_steps=args.refit_grid_steps,
            local_grid_steps=args.refit_local_grid_steps,
            activation_policy=args.activation_policy,
            source_contract_audit_path=args.source_contract_audit,
        )
    if args.simulate_sidecars:
        if not args.trace_log:
            raise ValueError("--simulate-sidecars requires --trace-log with the activation rows used for evaluation")
        output_path = args.output or sidecar_simulation_default_path(model_path, target_profile)
        return simulate_sidecars(
            cfg=cfg,
            refit_eval_path=args.simulate_sidecars,
            trace_path=args.trace_log,
            output_path=output_path,
            sidecar_modes=args.sidecar_modes,
            patch_tensors=args.patch_tensors,
            patch_top_groups=args.patch_top_groups,
            source_contract_audit_path=args.source_contract_audit,
        )
    if args.write_sidecar_artifact:
        output_path = args.output or sidecar_write_default_path(model_path, target_profile)
        return write_sidecar_artifacts(
            cfg=cfg,
            sidecar_simulation_path=args.write_sidecar_artifact,
            output_path=output_path,
            variant_name=args.variant_name,
            sidecar_artifact_mode=args.sidecar_artifact_mode,
            sidecar_correction_scale=args.sidecar_correction_scale,
            sidecar_include_entries=args.sidecar_include_entries,
            sidecar_exclude_entries=args.sidecar_exclude_entries,
            sidecar_entry_scales=args.sidecar_entry_scales,
            source_contract_audit_path=args.source_contract_audit,
        )
    if args.write_int8_exception_manifest:
        output_path = args.output or int8_exception_write_default_path(model_path, target_profile)
        return write_int8_exception_manifest(
            cfg=cfg,
            candidates_path=args.write_int8_exception_manifest,
            output_path=output_path,
            variant_name=args.variant_name,
            int8_exception_groups=args.int8_exception_groups,
            int8_exception_top_groups=args.int8_exception_top_groups,
            int8_exception_max_sidecar_ratio=args.int8_exception_max_sidecar_ratio,
            int8_exception_max_sidecar_bytes=args.int8_exception_max_sidecar_bytes,
            int8_exception_decode_sidecar_ratio=args.int8_exception_decode_sidecar_ratio,
            int8_exception_decode_sidecar_bytes=args.int8_exception_decode_sidecar_bytes,
            int8_exception_prefill_sidecar_ratio=args.int8_exception_prefill_sidecar_ratio,
            int8_exception_prefill_sidecar_bytes=args.int8_exception_prefill_sidecar_bytes,
        )
    if args.write_sidecar_contract_manifest:
        output_path = args.output or sidecar_write_default_path(model_path, target_profile)
        return write_sidecar_contract_manifest(
            cfg=cfg,
            source_contract_audit_path=args.write_sidecar_contract_manifest,
            output_path=output_path,
            variant_name=args.variant_name,
        )
    if args.write_experimental_cache:
        output_path = args.output or experimental_cache_write_default_path(model_path, target_profile)
        return write_experimental_cache(
            cfg=cfg,
            refit_eval_path=args.write_experimental_cache,
            output_path=output_path,
            variant_name=args.variant_name,
            patch_tensors=args.patch_tensors,
            patch_methods=args.patch_methods,
            patch_top_groups=args.patch_top_groups,
        )
    if args.summarize_ablation:
        output_path = args.output or ablation_report_default_path(model_path, target_profile)
        return build_ablation_report(spec_path=args.summarize_ablation, output_path=output_path)
    if args.search_sidecar_conflicts:
        if args.output is None:
            raise ValueError("--search-sidecar-conflicts requires --output")
        return build_sidecar_conflict_search(spec_path=args.search_sidecar_conflicts, output_path=args.output)

    output_path = args.output or evidence_default_path(model_path, target_profile)
    trace_path = args.trace_log
    run_info: Dict[str, Any]
    if args.dry_run:
        activation_stats = {
            "trace_path": trace_path,
            "input_row_full_events": 0,
            "field_count": 0,
            "fields": [],
        }
        run_info = {
            "dry_run": True,
            "server_started": False,
            "request_count": 0,
            "trace_selectors": {
                "positions": args.positions,
                "layers": args.layers,
                "tensors": args.tensors,
            },
        }
    else:
        if trace_path:
            run_info = {
                "dry_run": False,
                "server_started": False,
                "request_count": 0,
                "trace_log_source": "provided",
            }
        else:
            server_log_path = evidence_trace_log_path(output_path)
            run_info = run_calibration_requests(
                config_path=args.config,
                cfg=cfg,
                repo_root=args.repo_root,
                duration_seconds=args.duration_seconds,
                dataset=args.dataset,
                dataset_path=args.dataset_path,
                prompt_chars=args.prompt_chars,
                startup_timeout=args.startup_timeout,
                server_log_path=server_log_path,
                source_profile=source_profile,
                positions=args.positions,
                layers=args.layers,
                tensors=args.tensors,
            )
            trace_path = server_log_path
        activation_stats = parse_trace_activation_stats(trace_path)
        if os.path.isfile(trace_path):
            activation_stats["trace_sha256"] = _file_sha256(trace_path)

    payload = build_evidence_payload(
        config_path=args.config,
        cfg=cfg,
        source_profile=source_profile,
        target_profile=target_profile,
        duration_seconds=args.duration_seconds,
        dataset=args.dataset,
        output_path=output_path,
        activation_stats=activation_stats,
        run_info=run_info,
        dry_run=args.dry_run,
    )
    write_json(output_path, payload)
    latest_path = os.path.join(os.path.dirname(output_path), "evidence_latest.json")
    write_json(latest_path, payload)
    return payload


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture HQQ self-calibration evidence and offline experiments")
    parser.add_argument("--config", required=True, help="Krasis config with CFG_ATTENTION_QUANT=hqq4")
    parser.add_argument("--profile", default=HQQ_CACHE_PROFILE_SELFCAL_V1, choices=[HQQ_CACHE_PROFILE_SELFCAL_V1])
    parser.add_argument("--duration-seconds", type=int, default=180)
    parser.add_argument("--dataset", default="canonical", choices=["canonical", "c4"])
    parser.add_argument("--dataset-path", default=None, help="Required for --dataset c4; local text/jsonl path")
    parser.add_argument("--prompt-chars", type=int, default=24000)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--positions", default=DEFAULT_POSITIONS)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--tensors", default=DEFAULT_TENSORS)
    parser.add_argument("--trace-log", default=None, help="Reduce an existing trace instead of launching a server")
    parser.add_argument("--build-candidates", default=None, metavar="EVIDENCE_JSON")
    parser.add_argument("--build-int8-exception-candidates", default=None, metavar="TRAIN_EVIDENCE_JSON")
    parser.add_argument(
        "--heldout-evidence",
        default=None,
        metavar="HELDOUT_EVIDENCE_JSON",
        help="Heldout evidence JSON for --build-int8-exception-candidates",
    )
    parser.add_argument("--evaluate-refits", default=None, metavar="CANDIDATES_JSON")
    parser.add_argument("--simulate-sidecars", default=None, metavar="REFIT_EVAL_JSON")
    parser.add_argument("--write-sidecar-artifact", default=None, metavar="SIDECAR_SIM_JSON")
    parser.add_argument("--write-int8-exception-manifest", default=None, metavar="CANDIDATES_JSON")
    parser.add_argument(
        "--write-sidecar-contract-manifest",
        default=None,
        metavar="SOURCE_MAP_AUDIT_JSON",
        help="Write a manifest-only offline sidecar source-contract artifact from source-map-audit JSON",
    )
    parser.add_argument("--write-experimental-cache", default=None, metavar="REFIT_EVAL_JSON")
    parser.add_argument("--summarize-ablation", default=None, metavar="SPEC_JSON")
    parser.add_argument("--search-sidecar-conflicts", default=None, metavar="SPEC_JSON")
    parser.add_argument("--variant-name", default="selfcal_v1_experimental")
    parser.add_argument("--patch-tensors", default=None, help="Comma-separated tensor filter for experimental writes")
    parser.add_argument("--patch-methods", default=None, help="Comma-separated refit method filter for experimental writes")
    parser.add_argument("--patch-top-groups", type=int, default=None, help="Patch only the first N candidate groups")
    parser.add_argument(
        "--sidecar-modes",
        default="exact_bf16,int8_symmetric",
        help="Comma-separated sidecar simulation modes: exact_bf16,int8_symmetric",
    )
    parser.add_argument(
        "--sidecar-artifact-mode",
        default="int8_symmetric",
        choices=["exact_bf16", "int8_symmetric"],
        help="Sidecar artifact storage mode for --write-sidecar-artifact",
    )
    parser.add_argument(
        "--sidecar-correction-scale",
        type=float,
        default=1.0,
        help="Scale explicit sidecar corrections for --write-sidecar-artifact; must be finite and in (0, 1]",
    )
    parser.add_argument(
        "--sidecar-include-entries",
        default=None,
        help="Comma-separated layer:tensor:group:output_row entries to include in sidecar artifact writes",
    )
    parser.add_argument(
        "--sidecar-exclude-entries",
        default=None,
        help="Comma-separated layer:tensor:group:output_row entries to exclude from sidecar artifact writes",
    )
    parser.add_argument(
        "--sidecar-entry-scales",
        default=None,
        help=(
            "Comma-separated layer:tensor:group:output_row=scale overrides for sidecar artifact writes; "
            "requires --sidecar-include-entries"
        ),
    )
    parser.add_argument(
        "--int8-exception-groups",
        default=None,
        help="Comma-separated layer:tensor:group entries for --write-int8-exception-manifest",
    )
    parser.add_argument(
        "--int8-exception-top-groups",
        type=int,
        default=None,
        help=(
            "Write the first N ready groups from --write-int8-exception-manifest when no explicit group list "
            "is supplied. Use this with optional sidecar byte/ratio caps for HQQ4SC budget ladders."
        ),
    )
    parser.add_argument(
        "--int8-exception-max-sidecar-ratio",
        type=float,
        default=None,
        help=(
            "Limit --write-int8-exception-manifest runtime sidecar bytes to this fraction of the "
            "source HQQ4 attention cache tensor bytes, e.g. 0.10 for a +10%% HQQ4SC memory budget."
        ),
    )
    parser.add_argument(
        "--int8-exception-max-sidecar-bytes",
        type=int,
        default=None,
        help=(
            "Hard runtime-sidecar byte cap for --write-int8-exception-manifest. When combined with "
            "--int8-exception-max-sidecar-ratio, the lower cap wins."
        ),
    )
    parser.add_argument(
        "--int8-exception-decode-sidecar-ratio",
        type=float,
        default=None,
        help=(
            "Runtime-sidecar byte budget for decode-attached INT8 exception candidates as a fraction "
            "of source HQQ4 attention cache tensor bytes."
        ),
    )
    parser.add_argument(
        "--int8-exception-decode-sidecar-bytes",
        type=int,
        default=None,
        help="Hard runtime-sidecar byte cap for decode-attached INT8 exception candidates.",
    )
    parser.add_argument(
        "--int8-exception-prefill-sidecar-ratio",
        type=float,
        default=None,
        help=(
            "Runtime-sidecar byte budget for prefill-only INT8 exception candidates as a fraction "
            "of source HQQ4 attention cache tensor bytes."
        ),
    )
    parser.add_argument(
        "--int8-exception-prefill-sidecar-bytes",
        type=int,
        default=None,
        help="Hard runtime-sidecar byte cap for prefill-only INT8 exception candidates.",
    )
    parser.add_argument(
        "--source-contract-audit",
        default=None,
        help="Strict source-map-audit JSON to embed and validate during --write-sidecar-artifact",
    )
    parser.add_argument("--top-candidates", type=int, default=64)
    parser.add_argument("--max-output-rows", type=int, default=32)
    parser.add_argument("--refit-grid-steps", type=int, default=65)
    parser.add_argument("--refit-local-grid-steps", type=int, default=33)
    parser.add_argument(
        "--activation-policy",
        default="all",
        choices=["all", "shared_bf16_hqq_activation", "lane_local_only"],
        help="Candidate policy filter for --evaluate-refits",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--dry-run", action="store_true", help="Write non-loadable evidence skeleton without launching")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    if payload.get("format") == CANDIDATES_FORMAT:
        print(json.dumps({
            "status": payload["status"],
            "candidate_path": payload["cache"]["candidate_path"],
            "loadable_hqq_manifest": payload["loadable_hqq_manifest"],
            "candidate_count": payload["summary"]["candidate_count"],
            "top_candidate_count": payload["summary"]["top_candidate_count"],
        }, sort_keys=True))
    elif payload.get("format") == INT8_EXCEPTION_CANDIDATES_FORMAT:
        print(json.dumps({
            "status": payload["status"],
            "candidate_path": payload["cache"]["int8_exception_candidate_path"],
            "loadable_hqq_manifest": payload["loadable_hqq_manifest"],
            "candidate_count": payload["summary"]["candidate_count"],
            "candidate_ready_count": payload["summary"]["candidate_ready_count"],
            "top_candidate_count": payload["summary"]["top_candidate_count"],
        }, sort_keys=True))
    elif payload.get("format") == INT8_EXCEPTION_WRITE_FORMAT:
        print(json.dumps({
            "status": payload["status"],
            "manifest_path": payload["cache"]["sidecar_manifest_path"],
            "runtime_ready": payload["runtime_ready"],
            "exception_group_count": payload["summary"]["exception_group_count"],
            "row_group_count": payload["summary"]["row_group_count"],
            "sidecar_total_bytes": payload["summary"]["sidecar_total_bytes"],
        }, sort_keys=True))
    elif payload.get("format") == REFIT_EVAL_FORMAT:
        print(json.dumps({
            "status": payload["status"],
            "refit_evaluation_path": payload["cache"]["refit_evaluation_path"],
            "loadable_hqq_manifest": payload["loadable_hqq_manifest"],
            "evaluated_candidate_count": payload["summary"]["evaluated_candidate_count"],
            "selected_output_row_total": payload["summary"]["selected_output_row_total"],
        }, sort_keys=True))
    elif payload.get("format") == SIDECAR_SIM_FORMAT:
        print(json.dumps({
            "status": payload["status"],
            "sidecar_simulation_path": payload["cache"]["sidecar_simulation_path"],
            "loadable_hqq_manifest": payload["loadable_hqq_manifest"],
            "simulated_group_count": payload["summary"]["simulated_group_count"],
            "simulated_row_group_count": payload["summary"]["simulated_row_group_count"],
        }, sort_keys=True))
    elif payload.get("format") == SIDECAR_WRITE_FORMAT:
        print(json.dumps({
            "status": payload["status"],
            "sidecar_manifest_path": payload["cache"]["sidecar_manifest_path"],
            "loadable_hqq_manifest": payload["loadable_hqq_manifest"],
            "variant_name": payload["variant_name"],
            "sidecar_mode": payload["sidecar_mode"],
            "correction_scale": payload["correction_scale"],
            "row_group_count": payload["summary"]["row_group_count"],
            "sidecar_total_bytes": payload["summary"]["sidecar_total_bytes"],
            "decode_fma_per_token": payload["summary"]["decode_fma_per_token"],
        }, sort_keys=True))
    elif payload.get("format") == SIDECAR_CONTRACT_MANIFEST_WRITE_FORMAT:
        print(json.dumps({
            "status": payload["status"],
            "sidecar_manifest_path": payload["cache"]["sidecar_manifest_path"],
            "loadable_hqq_manifest": payload["loadable_hqq_manifest"],
            "variant_name": payload["variant_name"],
            "contract_embedded": payload["summary"]["contract_embedded"],
            "generation_ready": payload["summary"]["generation_ready"],
            "runtime_ready": payload["summary"]["runtime_ready"],
        }, sort_keys=True))
    elif payload.get("format") == EXPERIMENTAL_CACHE_WRITE_FORMAT:
        print(json.dumps({
            "status": payload["status"],
            "experimental_cache_write_path": payload["cache"]["experimental_cache_write_path"],
            "target_manifest_path": payload["cache"]["target_manifest_path"],
            "loadable_hqq_manifest": payload["loadable_hqq_manifest"],
            "variant_name": payload["summary"]["variant_name"],
            "patched_tensor_count": payload["summary"]["patched_tensor_count"],
            "patched_row_group_count": payload["summary"]["patched_row_group_count"],
        }, sort_keys=True))
    elif payload.get("format") == ABLATION_REPORT_FORMAT:
        print(json.dumps({
            "status": payload["status"],
            "variant_count": payload["variant_count"],
            "accepted_variant_count": payload["accepted_variant_count"],
        }, sort_keys=True))
    elif payload.get("format") == SIDECAR_CONFLICT_SEARCH_FORMAT:
        print(json.dumps({
            "status": payload["status"],
            "raw_candidate_count": payload["search"]["raw_candidate_count"],
            "selected_candidate_count": payload["search"]["selected_candidate_count"],
            "predicted_accepted_count": sum(
                1 for item in payload["candidates"] if item["predicted_accepted_strict"]
            ),
        }, sort_keys=True))
    else:
        print(json.dumps({
            "status": payload["status"],
            "evidence_path": payload["cache"]["evidence_path"],
            "loadable_hqq_manifest": payload["loadable_hqq_manifest"],
            "input_row_full_events": payload["activation_stats"]["input_row_full_events"],
            "field_count": payload["activation_stats"]["field_count"],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        print("ERROR: run via ./dev hqq-self-calibrate, not python directly", file=sys.stderr)
        sys.exit(2)
    sys.exit(main())
