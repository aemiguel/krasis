#!/usr/bin/env python3
"""Generate reference greedy decode outputs for deterministic model validation.

Loads a model via HuggingFace transformers in BF16, runs the sanity prompts
with greedy decoding (do_sample=False), and stores output token IDs and text
as JSON. This provides the ground truth for validating Krasis decode correctness.

Usage:
    ./dev generate-reference <model-name> [--max-tokens N]

This script must be run via ./dev generate-reference, not directly.
"""

import argparse
import hashlib
import inspect
import importlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import re
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from reference_contract import (
        REFERENCE_PROFILES,
        apply_capture_template,
        build_contract,
        build_reference_sanity_report,
        canonical_profile_id,
        collect_capture_stop_ids,
        capture_settings_for_profile,
        emit_reference_generation_trace,
        load_tokenizer_with_compat,
        profile_filename,
    )
except ModuleNotFoundError:
    from tests.reference_contract import (
        REFERENCE_PROFILES,
        apply_capture_template,
        build_contract,
        build_reference_sanity_report,
        canonical_profile_id,
        collect_capture_stop_ids,
        capture_settings_for_profile,
        emit_reference_generation_trace,
        load_tokenizer_with_compat,
        profile_filename,
    )

# Guard: must be run via ./dev
if not os.environ.get("KRASIS_DEV_SCRIPT"):
    print("ERROR: This script must be run via ./dev generate-reference, not directly.")
    print("  Usage: ./dev generate-reference <model-name>")
    sys.exit(1)

MODELS_DIR = os.environ.get("KRASIS_REFERENCE_CAPTURE_MODELS_DIR", os.path.expanduser("~/.krasis/models"))
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
REFERENCE_DIR = SCRIPT_DIR / "reference_outputs"
PROMPTS_FILE = REPO_DIR / "benchmarks" / "sanity_test_prompts.txt"
DEFAULT_DIAGNOSTIC_STEPS = 4
DEFAULT_DIAGNOSTIC_TOPK = 10

MODEL_ALIASES = {
    "qcn": "Qwen3-Coder-Next",
    "qwen3-coder-next": "Qwen3-Coder-Next",
    "qwen35": "Qwen3.5-35B-A3B",
    "q35b": "Qwen3.5-35B-A3B",
    "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B",
    "q122b": "Qwen3.5-122B-A10B",
    "qwen122": "Qwen3.5-122B-A10B",
    "qwen3.5-122b-a10b": "Qwen3.5-122B-A10B",
    "gemma": "gemma-4-26B-A4B-it",
    "gemma26": "gemma-4-26B-A4B-it",
    "gemma-4-26b-a4b-it": "gemma-4-26B-A4B-it",
    "minimax": "MiniMax-M2.5",
    "minimax25": "MiniMax-M2.5",
    "minimax-m2.5": "MiniMax-M2.5",
    "nemotron-nano": "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nemotronnano": "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia-nemotron-3-nano-30b-a3b-bf16": "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nemotron-super": "NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nemotronsuper": "NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia-nemotron-3-super-120b-a12b-bf16": "NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "q235": "Qwen3-235B-A22B",
    "qwen235": "Qwen3-235B-A22B",
    "qwen3-235b-a22b": "Qwen3-235B-A22B",
    "q397": "Qwen3.5-397B-A17B",
    "qwen397": "Qwen3.5-397B-A17B",
    "qwen3.5-397b-a17b": "Qwen3.5-397B-A17B",
}

BOLD = "\033[1m"
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
NC = "\033[0m"


def info(msg: str):
    print(f"{CYAN}{BOLD}=>{NC} {msg}")

def ok(msg: str):
    print(f"{GREEN}{BOLD}OK{NC} {msg}")

def warn(msg: str):
    print(f"{YELLOW}{BOLD}!!{NC} {msg}")

def die(msg: str):
    print(f"{RED}{BOLD}ERROR{NC} {msg}", file=sys.stderr)
    sys.exit(1)


def _package_version(package_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_file_audit(model_path: str) -> Dict[str, Any]:
    root = Path(model_path)
    filenames = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]
    files: Dict[str, Any] = {}
    for name in filenames:
        path = root / name
        if path.is_file():
            files[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    remote_code_files = []
    for path in sorted(root.glob("*.py")):
        remote_code_files.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "path": str(root.resolve()),
        "files": files,
        "remote_code_files": remote_code_files,
    }


def build_invocation_metadata(model_name: str, profile_id: str, max_new_tokens: int) -> Dict[str, Any]:
    run_dir = os.environ.get("KRASIS_RUN_DIR")
    script_argv = [str(Path(arg)) if idx == 0 else arg for idx, arg in enumerate(sys.argv)]
    dev_command = os.environ.get("KRASIS_DEV_CMD")
    if not dev_command:
        dev_command = " ".join(shlex.quote(arg) for arg in (["./dev", "generate-reference"] + sys.argv[1:]))
    metadata: Dict[str, Any] = {
        "captured_at": datetime.now().isoformat(),
        "dev_command": dev_command,
        "script_argv": script_argv,
        "cwd": str(Path.cwd().resolve()),
        "run_type": os.environ.get("KRASIS_RUN_TYPE"),
        "model": model_name,
        "profile_id": profile_id,
        "max_new_tokens": max_new_tokens,
    }
    if run_dir:
        metadata["run_dir"] = str(Path(run_dir).resolve())
    return metadata


def resolve_model_name(raw_model: str) -> str:
    raw = raw_model.strip()
    if not raw:
        die("Model name cannot be empty")

    candidate = Path(raw)
    if candidate.name != raw:
        raw = candidate.name

    if os.path.isdir(os.path.join(MODELS_DIR, raw)):
        return raw

    lowered = raw.lower()
    resolved = MODEL_ALIASES.get(lowered)
    if resolved and os.path.isdir(os.path.join(MODELS_DIR, resolved)):
        return resolved

    if "/" in raw_model:
        basename = Path(raw_model).name
        if os.path.isdir(os.path.join(MODELS_DIR, basename)):
            return basename

    if resolved:
        return resolved
    return raw


def write_run_manifest(
    invocation: Dict[str, Any],
    output_path: Path,
    model_name: str,
    profile_id: str,
    max_new_tokens: int,
    diagnostic_steps: int,
    diagnostic_top_k: int,
    prompt_subset: Optional[Dict[str, Any]] = None,
    attn_implementation: Optional[str] = None,
    sanity: Optional[Dict[str, Any]] = None,
) -> None:
    run_dir = invocation.get("run_dir")
    if not run_dir:
        return
    manifest_path = Path(run_dir) / "generate_reference_manifest.json"
    manifest = {
        "kind": "generate_reference_manifest",
        "captured_at": invocation.get("captured_at"),
        "dev_command": invocation.get("dev_command"),
        "script_argv": invocation.get("script_argv"),
        "cwd": invocation.get("cwd"),
        "run_dir": run_dir,
        "run_type": invocation.get("run_type"),
        "model": model_name,
        "profile_id": profile_id,
        "max_new_tokens": max_new_tokens,
        "decode_diagnostics": {
            "captured_steps_per_turn": diagnostic_steps,
            "top_k": diagnostic_top_k,
        },
        "prompt_subset": prompt_subset,
        "attn_implementation": attn_implementation,
        "reference_output_path": str(output_path.resolve()),
        "sanity": sanity,
        "follow_up_commands": [
            "./dev reference-inventory",
            "./dev validate <config> --max-prompts 0 --no-server --port 1",
            "./dev reference-test <config>",
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    ok(f"Run manifest saved: {manifest_path}")


def write_generation_failure_manifest(
    invocation: Dict[str, Any],
    *,
    intended_output_path: Path,
    model_name: str,
    profile_id: str,
    sanity: Dict[str, Any],
    prompt_subset: Optional[Dict[str, Any]] = None,
    attn_implementation: Optional[str] = None,
    turn_summaries: Optional[List[Dict[str, Any]]] = None,
    diagnostic_controls: Optional[Dict[str, Any]] = None,
    environment_audit: Optional[Dict[str, Any]] = None,
    token_audit_turns: Optional[List[Dict[str, Any]]] = None,
) -> None:
    run_dir = invocation.get("run_dir")
    if not run_dir:
        return
    manifest_path = Path(run_dir) / "generate_reference_failure.json"
    manifest = {
        "kind": "generate_reference_failure",
        "captured_at": invocation.get("captured_at"),
        "dev_command": invocation.get("dev_command"),
        "script_argv": invocation.get("script_argv"),
        "cwd": invocation.get("cwd"),
        "run_dir": run_dir,
        "run_type": invocation.get("run_type"),
        "model": model_name,
        "profile_id": profile_id,
        "prompt_subset": prompt_subset,
        "attn_implementation": attn_implementation,
        "diagnostic_controls": diagnostic_controls or {},
        "intended_reference_output_path": str(intended_output_path.resolve()),
        "sanity": sanity,
        "environment_audit": environment_audit or {},
        "turn_summaries": turn_summaries or [],
        "token_audit_turns": token_audit_turns or [],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    warn(f"Failure manifest saved: {manifest_path}")


def write_generation_diagnostic_manifest(
    invocation: Dict[str, Any],
    *,
    intended_output_path: Path,
    model_name: str,
    profile_id: str,
    sanity: Dict[str, Any],
    prompt_subset: Optional[Dict[str, Any]] = None,
    attn_implementation: Optional[str] = None,
    turn_summaries: Optional[List[Dict[str, Any]]] = None,
    diagnostic_controls: Optional[Dict[str, Any]] = None,
    environment_audit: Optional[Dict[str, Any]] = None,
    token_audit_turns: Optional[List[Dict[str, Any]]] = None,
) -> None:
    run_dir = invocation.get("run_dir")
    if not run_dir:
        return
    manifest_path = Path(run_dir) / "generate_reference_diagnostic.json"
    manifest = {
        "kind": "generate_reference_diagnostic",
        "captured_at": invocation.get("captured_at"),
        "dev_command": invocation.get("dev_command"),
        "script_argv": invocation.get("script_argv"),
        "cwd": invocation.get("cwd"),
        "run_dir": run_dir,
        "run_type": invocation.get("run_type"),
        "model": model_name,
        "profile_id": profile_id,
        "prompt_subset": prompt_subset,
        "attn_implementation": attn_implementation,
        "diagnostic_controls": diagnostic_controls or {},
        "intended_reference_output_path": str(intended_output_path.resolve()),
        "reference_artifact_written": False,
        "sanity": sanity,
        "environment_audit": environment_audit or {},
        "turn_summaries": turn_summaries or [],
        "token_audit_turns": token_audit_turns or [],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    ok(f"Diagnostic manifest saved: {manifest_path}")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_float_or_none(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _tensor_finiteness_stats(tensor: Any) -> Dict[str, Any]:
    import torch

    detached = tensor.detach()
    flat = detached.reshape(-1)
    finite_mask = torch.isfinite(flat)
    nan_mask = torch.isnan(flat)
    inf_mask = torch.isinf(flat)
    finite_values = flat[finite_mask]
    stats: Dict[str, Any] = {
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "shape": [int(dim) for dim in detached.shape],
        "numel": int(flat.numel()),
        "finite_count": int(finite_mask.sum().item()),
        "nan_count": int(nan_mask.sum().item()),
        "inf_count": int(inf_mask.sum().item()),
        "all_finite": bool(finite_mask.all().item()) if flat.numel() else True,
    }
    if finite_values.numel() > 0:
        finite_float = finite_values.float()
        stats["finite_min"] = float(finite_float.min().item())
        stats["finite_max"] = float(finite_float.max().item())
    else:
        stats["finite_min"] = None
        stats["finite_max"] = None
    return stats


def _cache_state_snapshot(model: Any, config: Any) -> Dict[str, Any]:
    generation_config = getattr(model, "generation_config", None)
    snapshot: Dict[str, Any] = {
        "config_use_cache": getattr(config, "use_cache", None),
        "generation_config_use_cache": getattr(generation_config, "use_cache", None),
        "model_training": bool(getattr(model, "training", False)),
        "model_eval": not bool(getattr(model, "training", False)),
        "known_cache_attrs": {},
    }
    for attr_name in (
        "past_key_values",
        "_past_key_values",
        "cache_params",
        "_cache",
        "cache",
        "_seen_tokens",
        "seen_tokens",
    ):
        if not hasattr(model, attr_name):
            continue
        try:
            value = getattr(model, attr_name)
        except Exception as exc:
            snapshot["known_cache_attrs"][attr_name] = {"present": True, "read_error": repr(exc)}
            continue
        entry: Dict[str, Any] = {
            "present": True,
            "is_none": value is None,
            "type": type(value).__name__,
        }
        try:
            entry["past_seen_tokens"] = _past_seen_tokens(value)
        except Exception:
            pass
        snapshot["known_cache_attrs"][attr_name] = entry
    return snapshot


def _cache_mode_to_value(cache_mode: str) -> Optional[bool]:
    if cache_mode not in ("auto", "on", "off"):
        die(f"Unknown cache mode: {cache_mode}")
    if cache_mode == "auto":
        return None
    return cache_mode == "on"


def _set_cache_mode(model: Any, config: Any, use_cache_mode: str) -> Dict[str, Any]:
    if use_cache_mode not in ("auto", "on", "off"):
        die(f"Unknown use-cache mode: {use_cache_mode}")
    generation_config = getattr(model, "generation_config", None)
    before = _cache_state_snapshot(model, config)
    if use_cache_mode == "auto":
        return {"mode": use_cache_mode, "before": before, "after": before}

    enabled = use_cache_mode == "on"
    if hasattr(config, "use_cache"):
        setattr(config, "use_cache", enabled)
    if generation_config is not None and hasattr(generation_config, "use_cache"):
        setattr(generation_config, "use_cache", enabled)
    after = _cache_state_snapshot(model, config)
    return {"mode": use_cache_mode, "before": before, "after": after}


def _cleanup_between_turns(model: Any, config: Any, mode: str) -> Dict[str, Any]:
    if mode not in ("none", "gc", "reset"):
        die(f"Unknown cleanup mode: {mode}")
    before = _cache_state_snapshot(model, config)
    actions: List[str] = []
    if mode in ("gc", "reset"):
        import gc
        import torch

        gc.collect()
        actions.append("gc.collect")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            actions.extend(["torch.cuda.empty_cache", "torch.cuda.synchronize"])
    if mode == "reset":
        model.eval()
        actions.append("model.eval")
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            try:
                generation_config.validate()
                actions.append("generation_config.validate")
            except Exception as exc:
                actions.append(f"generation_config.validate_error:{exc!r}")
        for attr_name in ("past_key_values", "_past_key_values", "cache_params", "_cache", "cache"):
            if hasattr(model, attr_name):
                try:
                    setattr(model, attr_name, None)
                    actions.append(f"clear_model_attr:{attr_name}")
                except Exception as exc:
                    actions.append(f"clear_model_attr_error:{attr_name}:{exc!r}")
    after = _cache_state_snapshot(model, config)
    return {"mode": mode, "actions": actions, "before": before, "after": after}


def summarize_reference_turns(reference: Dict[str, Any], expected_top_k: int) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for conv_idx, conv in enumerate(reference.get("conversations", [])):
        for turn_idx, turn in enumerate(conv.get("turns", [])):
            input_ids = turn.get("input_token_ids") if isinstance(turn.get("input_token_ids"), list) else []
            token_ids = turn.get("token_ids") if isinstance(turn.get("token_ids"), list) else []
            per_token = turn.get("per_token_data") if isinstance(turn.get("per_token_data"), list) else []
            first_diag = per_token[0] if per_token and isinstance(per_token[0], dict) else None
            top_k = first_diag.get("top_k") if first_diag and isinstance(first_diag.get("top_k"), list) else []
            top_k_log_probs = [
                entry.get("log_prob")
                for entry in top_k
                if isinstance(entry, dict)
            ]
            prompt = turn.get("prompt") if isinstance(turn.get("prompt"), str) else ""
            source_conv = turn.get("source_conversation_index", conv_idx)
            source_turn = turn.get("source_turn_index", turn_idx)
            diag_token_id = first_diag.get("token_id") if first_diag else None
            generated_first_token = token_ids[0] if token_ids else None
            top_k0 = top_k[0] if top_k and isinstance(top_k[0], dict) else None
            logits_stats = first_diag.get("logits_stats") if first_diag and isinstance(first_diag.get("logits_stats"), dict) else {}
            log_probs_stats = first_diag.get("log_probs_stats") if first_diag and isinstance(first_diag.get("log_probs_stats"), dict) else {}
            generation_state = turn.get("generation_state") if isinstance(turn.get("generation_state"), dict) else {}
            pre_generate = turn.get("pre_generate_diagnostic") if isinstance(turn.get("pre_generate_diagnostic"), list) else []
            post_generate = turn.get("post_generate_diagnostic") if isinstance(turn.get("post_generate_diagnostic"), list) else []
            postponed_prefill = (
                turn.get("postponed_prefill_diagnostic")
                if isinstance(turn.get("postponed_prefill_diagnostic"), list)
                else []
            )
            pre_first = pre_generate[0] if pre_generate and isinstance(pre_generate[0], dict) else None
            post_first = post_generate[0] if post_generate and isinstance(post_generate[0], dict) else None
            postponed_first = (
                postponed_prefill[0]
                if postponed_prefill and isinstance(postponed_prefill[0], dict)
                else None
            )
            summaries.append(
                {
                    "conversation_index": conv_idx,
                    "turn_index": turn_idx,
                    "source_conversation_index": source_conv,
                    "source_turn_index": source_turn,
                    "label": f"conv{source_conv}_t{source_turn}",
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt_preview": prompt[:120].replace("\n", " "),
                    "input_token_count": len(input_ids),
                    "input_token_sha256": _sha256_json(input_ids),
                    "generated_token_count": len(token_ids),
                    "generated_first_token_id": generated_first_token,
                    "generated_first_token_text": turn.get("text", "")[:40] if isinstance(turn.get("text"), str) else "",
                    "diagnostic_present": first_diag is not None,
                    "diagnostic_token_id": diag_token_id,
                    "diagnostic_token_matches_generated_first": (
                        diag_token_id == generated_first_token
                        if first_diag is not None and generated_first_token is not None
                        else False
                    ),
                    "diagnostic_logit_finite": _finite_float_or_none(first_diag.get("logit") if first_diag else None) is not None,
                    "diagnostic_log_prob_finite": _finite_float_or_none(first_diag.get("log_prob") if first_diag else None) is not None,
                    "diagnostic_logit": _finite_float_or_none(first_diag.get("logit") if first_diag else None),
                    "diagnostic_log_prob": _finite_float_or_none(first_diag.get("log_prob") if first_diag else None),
                    "logits_dtype": logits_stats.get("dtype"),
                    "logits_device": logits_stats.get("device"),
                    "logits_finite_count": logits_stats.get("finite_count"),
                    "logits_numel": logits_stats.get("numel"),
                    "logits_nan_count": logits_stats.get("nan_count"),
                    "logits_inf_count": logits_stats.get("inf_count"),
                    "log_probs_finite_count": log_probs_stats.get("finite_count"),
                    "log_probs_numel": log_probs_stats.get("numel"),
                    "log_probs_nan_count": log_probs_stats.get("nan_count"),
                    "log_probs_inf_count": log_probs_stats.get("inf_count"),
                    "top_k_length": len(top_k),
                    "top_k_coverage_ok": len(top_k) == expected_top_k,
                    "top_k_log_probs_all_finite": all(
                        _finite_float_or_none(value) is not None for value in top_k_log_probs
                    ) if top_k else False,
                    "top_k0_token_id": top_k0.get("token_id") if top_k0 else None,
                    "top_k0_log_prob_finite": (
                        _finite_float_or_none(top_k0.get("log_prob")) is not None
                        if top_k0
                        else False
                    ),
                    "top_k0_log_prob": _finite_float_or_none(top_k0.get("log_prob") if top_k0 else None),
                    "model_training": generation_state.get("model_training"),
                    "model_eval": generation_state.get("model_eval"),
                    "attn_implementation": generation_state.get("attn_implementation"),
                    "use_cache_mode": generation_state.get("use_cache_mode"),
                    "generate_use_cache_mode": generation_state.get("generate_use_cache_mode"),
                    "diagnostic_use_cache_mode": generation_state.get("diagnostic_use_cache_mode"),
                    "diagnostic_scope": generation_state.get("diagnostic_scope"),
                    "use_cache_generate_arg": generation_state.get("use_cache_generate_arg"),
                    "use_cache_forward_arg": generation_state.get("use_cache_forward_arg"),
                    "config_use_cache": generation_state.get("config_use_cache"),
                    "generation_config_use_cache": generation_state.get("generation_config_use_cache"),
                    "cache_snapshot_before_turn": generation_state.get("cache_snapshot_before_turn"),
                    "cache_snapshot_before_generate": generation_state.get("cache_snapshot_before_generate"),
                    "cache_snapshot_after_generate": generation_state.get("cache_snapshot_after_generate"),
                    "cache_snapshot_after_diagnostics": generation_state.get("cache_snapshot_after_diagnostics"),
                    "between_turn_cleanup": generation_state.get("between_turn_cleanup"),
                    "reloaded_model_before_turn": generation_state.get("reloaded_model_before_turn"),
                    "explicit_past_key_values_supplied": generation_state.get("explicit_past_key_values_supplied"),
                    "explicit_cache_object_supplied": generation_state.get("explicit_cache_object_supplied"),
                    "past_key_values_reused_from_prior_turn": generation_state.get("past_key_values_reused_from_prior_turn"),
                    "diagnostics_captured": generation_state.get("diagnostics_captured"),
                    "diagnostics_postponed": generation_state.get("diagnostics_postponed"),
                    "diagnostics_skipped_reason": generation_state.get("diagnostics_skipped_reason"),
                    "pre_generate_diagnostic_present": pre_first is not None,
                    "pre_generate_logits_finite_count": (
                        pre_first.get("logits_stats", {}).get("finite_count") if pre_first else None
                    ),
                    "pre_generate_logits_numel": (
                        pre_first.get("logits_stats", {}).get("numel") if pre_first else None
                    ),
                    "pre_generate_logits_nan_count": (
                        pre_first.get("logits_stats", {}).get("nan_count") if pre_first else None
                    ),
                    "pre_generate_top_k_log_probs_all_finite": all(
                        _finite_float_or_none(entry.get("log_prob")) is not None
                        for entry in (pre_first.get("top_k", []) if pre_first else [])
                        if isinstance(entry, dict)
                    ) if pre_first else None,
                    "post_generate_diagnostic_present": post_first is not None,
                    "post_generate_logits_finite_count": (
                        post_first.get("logits_stats", {}).get("finite_count") if post_first else None
                    ),
                    "post_generate_logits_numel": (
                        post_first.get("logits_stats", {}).get("numel") if post_first else None
                    ),
                    "post_generate_logits_nan_count": (
                        post_first.get("logits_stats", {}).get("nan_count") if post_first else None
                    ),
                    "post_generate_top_k_log_probs_all_finite": all(
                        _finite_float_or_none(entry.get("log_prob")) is not None
                        for entry in (post_first.get("top_k", []) if post_first else [])
                        if isinstance(entry, dict)
                    ) if post_first else None,
                    "postponed_prefill_diagnostic_present": postponed_first is not None,
                    "postponed_prefill_logits_finite_count": (
                        postponed_first.get("logits_stats", {}).get("finite_count") if postponed_first else None
                    ),
                    "postponed_prefill_logits_numel": (
                        postponed_first.get("logits_stats", {}).get("numel") if postponed_first else None
                    ),
                    "postponed_prefill_logits_nan_count": (
                        postponed_first.get("logits_stats", {}).get("nan_count") if postponed_first else None
                    ),
                    "postponed_prefill_top_k_log_probs_all_finite": all(
                        _finite_float_or_none(entry.get("log_prob")) is not None
                        for entry in (postponed_first.get("top_k", []) if postponed_first else [])
                        if isinstance(entry, dict)
                    ) if postponed_first else None,
                }
            )
    return summaries


def build_token_audit_turns(reference: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for conv_idx, conv in enumerate(reference.get("conversations", [])):
        for turn_idx, turn in enumerate(conv.get("turns", [])):
            input_ids = turn.get("input_token_ids") if isinstance(turn.get("input_token_ids"), list) else []
            token_ids = turn.get("token_ids") if isinstance(turn.get("token_ids"), list) else []
            prompt = turn.get("prompt") if isinstance(turn.get("prompt"), str) else ""
            source_conv = turn.get("source_conversation_index", conv_idx)
            source_turn = turn.get("source_turn_index", turn_idx)
            turns.append(
                {
                    "conversation_index": conv_idx,
                    "turn_index": turn_idx,
                    "source_conversation_index": source_conv,
                    "source_turn_index": source_turn,
                    "label": f"conv{source_conv}_t{source_turn}",
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "input_token_ids": input_ids,
                    "input_token_count": len(input_ids),
                    "input_token_sha256": _sha256_json(input_ids),
                    "generated_token_ids": token_ids,
                    "generated_first_token_id": token_ids[0] if token_ids else None,
                    "generated_text": turn.get("text") if isinstance(turn.get("text"), str) else "",
                    "generation_state": turn.get("generation_state") if isinstance(turn.get("generation_state"), dict) else {},
                }
            )
    return turns


def parse_prompt_conversations(lines: List[str]) -> List[List[str]]:
    """Parse prompt lines into conversations.

    Lines starting with '- ' are continuations of the previous conversation.
    All other non-empty lines start a new conversation.
    """
    conversations: List[List[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            prompt_text = stripped[2:].strip()
            if not prompt_text:
                continue
            if conversations:
                conversations[-1].append(prompt_text)
            else:
                conversations.append([prompt_text])
        else:
            conversations.append([stripped])
    return conversations


def _build_prompt_subset(
    conversations: List[List[str]],
    conversation_index: Optional[int],
    turn_index: Optional[int],
    turn_start_index: Optional[int],
    turn_end_index: Optional[int],
    first_stored_turns: Optional[int],
) -> Tuple[List[List[str]], Dict[str, Any]]:
    if first_stored_turns is not None and (
        conversation_index is not None
        or turn_index is not None
        or turn_start_index is not None
        or turn_end_index is not None
    ):
        die("--first-stored-turns cannot be combined with conversation/turn subset selectors")
    if first_stored_turns is not None:
        if first_stored_turns <= 0:
            die("--first-stored-turns must be > 0")
        selected: List[List[str]] = []
        source_indices: List[int] = []
        remaining = first_stored_turns
        total_turns = sum(len(conv) for conv in conversations)
        if first_stored_turns > total_turns:
            die(f"--first-stored-turns out of range: {first_stored_turns} (total={total_turns})")
        for source_conv_idx, conversation in enumerate(conversations):
            if remaining <= 0:
                break
            take = min(remaining, len(conversation))
            selected.append(conversation[:take])
            source_indices.append(source_conv_idx)
            remaining -= take
        return selected, {
            "enabled": True,
            "conversation_index": None,
            "turn_index": None,
            "turn_start_index": None,
            "turn_end_index": None,
            "first_stored_turns": first_stored_turns,
            "mode": "first_stored_turns",
            "source_conversation_indices": source_indices,
            "recorded_turns": first_stored_turns,
        }
    if turn_index is not None and (turn_start_index is not None or turn_end_index is not None):
        die("--turn-index cannot be combined with --turn-start-index or --turn-end-index")
    if turn_index is not None and conversation_index is None:
        die("--turn-index requires --conversation-index")
    if (turn_start_index is not None or turn_end_index is not None) and conversation_index is None:
        die("--turn-start-index/--turn-end-index require --conversation-index")
    if conversation_index is None:
        return conversations, {
            "enabled": False,
            "conversation_index": None,
            "turn_index": None,
            "turn_start_index": None,
            "turn_end_index": None,
            "first_stored_turns": None,
            "mode": "all",
        }
    if conversation_index < 0 or conversation_index >= len(conversations):
        die(f"--conversation-index out of range: {conversation_index} (count={len(conversations)})")

    selected_conversation = conversations[conversation_index]
    if turn_start_index is not None or turn_end_index is not None:
        start_idx = 0 if turn_start_index is None else turn_start_index
        end_idx = len(selected_conversation) - 1 if turn_end_index is None else turn_end_index
        if start_idx < 0 or start_idx >= len(selected_conversation):
            die(
                f"--turn-start-index out of range: {start_idx} "
                f"(conversation_index={conversation_index} turns={len(selected_conversation)})"
            )
        if end_idx < 0 or end_idx >= len(selected_conversation):
            die(
                f"--turn-end-index out of range: {end_idx} "
                f"(conversation_index={conversation_index} turns={len(selected_conversation)})"
            )
        if start_idx > end_idx:
            die(f"--turn-start-index must be <= --turn-end-index ({start_idx} > {end_idx})")
        return [selected_conversation[: end_idx + 1]], {
            "enabled": True,
            "conversation_index": conversation_index,
            "turn_index": None,
            "turn_start_index": start_idx,
            "turn_end_index": end_idx,
            "first_stored_turns": None,
            "mode": "turn_range_with_prior_history",
            "source_conversation_turns": len(selected_conversation),
            "recorded_turns": end_idx - start_idx + 1,
        }
    if turn_index is None:
        return [selected_conversation], {
            "enabled": True,
            "conversation_index": conversation_index,
            "turn_index": None,
            "turn_start_index": None,
            "turn_end_index": None,
            "first_stored_turns": None,
            "mode": "conversation",
            "source_conversation_turns": len(selected_conversation),
        }
    if turn_index < 0 or turn_index >= len(selected_conversation):
        die(
            f"--turn-index out of range: {turn_index} "
            f"(conversation_index={conversation_index} turns={len(selected_conversation)})"
        )
    return [selected_conversation[: turn_index + 1]], {
        "enabled": True,
        "conversation_index": conversation_index,
        "turn_index": turn_index,
        "turn_start_index": turn_index,
        "turn_end_index": turn_index,
        "first_stored_turns": None,
        "mode": "single_turn_with_prior_history",
        "source_conversation_turns": len(selected_conversation),
        "recorded_turns": 1,
    }


def _should_record_turn(prompt_subset: Dict[str, Any], turn_idx: int) -> bool:
    if not prompt_subset.get("enabled"):
        return True
    if prompt_subset.get("turn_index") is not None:
        return turn_idx == prompt_subset.get("turn_index")
    start_idx = prompt_subset.get("turn_start_index")
    end_idx = prompt_subset.get("turn_end_index")
    if start_idx is not None or end_idx is not None:
        start = int(start_idx or 0)
        end = int(end_idx if end_idx is not None else turn_idx)
        return start <= turn_idx <= end
    return True


def _is_final_recorded_turn(prompt_subset: Dict[str, Any], turn_idx: int) -> bool:
    if not _should_record_turn(prompt_subset, turn_idx):
        return False
    if prompt_subset.get("enabled"):
        if prompt_subset.get("turn_index") is not None:
            return turn_idx == int(prompt_subset["turn_index"])
        end_idx = prompt_subset.get("turn_end_index")
        if end_idx is not None:
            return turn_idx == int(end_idx)
    return True


def _should_capture_immediate_diagnostics(
    prompt_subset: Dict[str, Any],
    turn_idx: int,
    diagnostic_scope: str,
) -> bool:
    if diagnostic_scope == "all":
        return True
    if diagnostic_scope == "final":
        return _is_final_recorded_turn(prompt_subset, turn_idx)
    if diagnostic_scope == "postponed":
        return False
    die(f"Unknown diagnostic scope: {diagnostic_scope}")


def _diagnostic_output_path(
    model_name: str,
    profile_id: str,
    prompt_subset: Dict[str, Any],
    attn_implementation: Optional[str],
    explicit_output: Optional[str],
) -> Path:
    if explicit_output:
        return Path(explicit_output).expanduser().resolve()

    base = profile_filename(profile_id)
    if not prompt_subset.get("enabled") and not attn_implementation:
        return REFERENCE_DIR / model_name / base

    stem = Path(base).stem
    suffix_parts: List[str] = []
    if prompt_subset.get("enabled"):
        conv_idx = prompt_subset.get("conversation_index")
        first_stored_turns = prompt_subset.get("first_stored_turns")
        turn_idx = prompt_subset.get("turn_index")
        turn_start_idx = prompt_subset.get("turn_start_index")
        turn_end_idx = prompt_subset.get("turn_end_index")
        if first_stored_turns is not None:
            suffix_parts.append(f"first{first_stored_turns}turns")
        else:
            suffix_parts.append(f"conv{conv_idx}" if conv_idx is not None else "subset")
        if turn_idx is not None:
            suffix_parts.append(f"turn{turn_idx}")
        elif turn_start_idx is not None or turn_end_idx is not None:
            suffix_parts.append(f"turns{turn_start_idx}-{turn_end_idx}")
    if attn_implementation:
        suffix_parts.append(f"attn-{attn_implementation}")
    suffix = "__" + "__".join(suffix_parts) if suffix_parts else ""
    return REFERENCE_DIR / model_name / f"{stem}{suffix}.json"


def _decode_token_text(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return ""


def build_token_diagnostic_entry(
    tokenizer: Any,
    step_logits: Any,
    *,
    expected_token_id: int,
    logit_pos: int,
    prev_token_id: int,
    diagnostic_top_k: int,
) -> Dict[str, Any]:
    import torch

    logits_stats = _tensor_finiteness_stats(step_logits)
    log_probs = torch.log_softmax(step_logits.float(), dim=-1)
    log_probs_stats = _tensor_finiteness_stats(log_probs)
    expected_log_prob = float(log_probs[expected_token_id].item())
    expected_logit = float(step_logits[expected_token_id].float().item())
    expected_rank = int(torch.count_nonzero(step_logits > step_logits[expected_token_id]).item()) + 1
    vocab_size = step_logits.shape[-1]
    top_k = min(int(diagnostic_top_k), int(vocab_size))

    top_vals, top_ids = torch.topk(log_probs, k=top_k)
    top_k_entries = []
    for tok_id, log_prob in zip(top_ids.tolist(), top_vals.tolist()):
        top_k_entries.append(
            {
                "token_id": int(tok_id),
                "text": _decode_token_text(tokenizer, int(tok_id)),
                "log_prob": float(log_prob),
            }
        )

    return {
        "position": int(logit_pos),
        "previous_token_id": int(prev_token_id),
        "previous_token_text": _decode_token_text(tokenizer, int(prev_token_id)),
        "token_id": int(expected_token_id),
        "text": _decode_token_text(tokenizer, int(expected_token_id)),
        "log_prob": expected_log_prob,
        "logit": expected_logit,
        "rank": expected_rank,
        "logits_stats": logits_stats,
        "log_probs_stats": log_probs_stats,
        "top_k": top_k_entries,
    }


def build_teacher_forced_per_token_data(
    model: Any,
    tokenizer: Any,
    prompt_input_ids: Any,
    generated_token_ids: List[int],
    *,
    diagnostic_steps: int,
    diagnostic_top_k: int,
    use_cache: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    if diagnostic_steps <= 0 or diagnostic_top_k <= 0 or not generated_token_ids:
        return []

    import torch

    steps_to_capture = min(len(generated_token_ids), diagnostic_steps)
    teacher_suffix = generated_token_ids[: max(0, steps_to_capture - 1)]
    teacher_ids = prompt_input_ids[0].tolist() + teacher_suffix
    teacher_tensor = torch.tensor([teacher_ids], dtype=prompt_input_ids.dtype, device=model.device)

    forward_kwargs: Dict[str, Any] = {"attention_mask": torch.ones_like(teacher_tensor)}
    if use_cache is not None:
        forward_kwargs["use_cache"] = use_cache
    with torch.no_grad():
        logits = model(teacher_tensor, **forward_kwargs).logits[0].float()

    prompt_len = prompt_input_ids.shape[1]
    per_token_data: List[Dict[str, Any]] = []

    for step in range(steps_to_capture):
        logit_pos = prompt_len - 1 + step
        step_logits = logits[logit_pos]
        expected_token_id = int(generated_token_ids[step])
        prev_token_id = int(prompt_input_ids[0, -1].item()) if step == 0 else int(generated_token_ids[step - 1])
        per_token_data.append(
            build_token_diagnostic_entry(
                tokenizer,
                step_logits,
                expected_token_id=expected_token_id,
                logit_pos=logit_pos,
                prev_token_id=prev_token_id,
                diagnostic_top_k=diagnostic_top_k,
            )
        )

    return per_token_data


def capture_prefill_logits(model: Any, prompt_input_ids: Any, *, use_cache: Optional[bool] = None) -> Any:
    import torch

    forward_kwargs: Dict[str, Any] = {"attention_mask": torch.ones_like(prompt_input_ids)}
    if use_cache is not None:
        forward_kwargs["use_cache"] = use_cache
    with torch.no_grad():
        return model(prompt_input_ids, **forward_kwargs).logits[0, -1].detach()


def build_prefill_diagnostic_from_logits(
    tokenizer: Any,
    prompt_input_ids: Any,
    generated_token_ids: List[int],
    step_logits: Any,
    *,
    diagnostic_top_k: int,
) -> List[Dict[str, Any]]:
    if diagnostic_top_k <= 0 or not generated_token_ids:
        return []
    return [
        build_token_diagnostic_entry(
            tokenizer,
            step_logits,
            expected_token_id=int(generated_token_ids[0]),
            logit_pos=int(prompt_input_ids.shape[1]) - 1,
            prev_token_id=int(prompt_input_ids[0, -1].item()),
            diagnostic_top_k=diagnostic_top_k,
        )
    ]


def build_prefill_next_token_data(
    model: Any,
    tokenizer: Any,
    prompt_input_ids: Any,
    generated_token_ids: List[int],
    *,
    diagnostic_top_k: int,
    use_cache: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Capture first-token diagnostics from HF generate scores for exact raw input ids."""
    if diagnostic_top_k <= 0 or not generated_token_ids:
        return []

    import torch

    generate_kwargs = {
        "max_new_tokens": 1,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
        "attention_mask": torch.ones_like(prompt_input_ids),
    }
    if use_cache is not None:
        generate_kwargs["use_cache"] = use_cache
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id

    with torch.no_grad():
        generated = model.generate(prompt_input_ids, **generate_kwargs)

    scores = getattr(generated, "scores", None)
    if not scores:
        raise RuntimeError("HF generate returned no scores for first-token diagnostic")

    prompt_len = prompt_input_ids.shape[1]
    return [
        build_token_diagnostic_entry(
            tokenizer,
            scores[0][0],
            expected_token_id=int(generated_token_ids[0]),
            logit_pos=prompt_len - 1,
            prev_token_id=int(prompt_input_ids[0, -1].item()),
            diagnostic_top_k=diagnostic_top_k,
        )
    ]


def _past_seen_tokens(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except Exception:
            return 0
    if isinstance(past_key_values, tuple) and past_key_values:
        try:
            return int(past_key_values[0][0].shape[2])
        except Exception:
            return 0
    return 0


def maybe_patch_legacy_remote_cache_position(model: Any) -> None:
    """Backfill cache_position for legacy remote-code generation hooks."""
    import torch
    from transformers.cache_utils import DynamicCache

    prepare = getattr(model, "prepare_inputs_for_generation", None)
    if prepare is None:
        return

    try:
        prepare_sig = inspect.signature(prepare)
        forward_sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return

    if "cache_position" not in prepare_sig.parameters:
        return
    if "cache_position" not in forward_sig.parameters:
        return

    module_name = getattr(model.__class__, "__module__", "")
    if "transformers_modules" not in module_name:
        return
    model_module = importlib.import_module(module_name)
    hybrid_cache_cls = (
        getattr(model_module, "HybridMambaAttentionDynamicCache", None)
        or getattr(model_module, "NemotronHHybridDynamicCache", None)
    )

    original_prepare = prepare

    def patched_prepare_inputs_for_generation(*args: Any, **kwargs: Any) -> Any:
        cache_state = kwargs.get("cache_params", kwargs.get("past_key_values"))

        cache_position = kwargs.get("cache_position")
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        inputs_embeds = kwargs.get("inputs_embeds")
        if input_ids is not None:
            sequence_length = int(input_ids.shape[1])
            batch_size = int(input_ids.shape[0])
            device = input_ids.device
        elif inputs_embeds is not None:
            sequence_length = int(inputs_embeds.shape[1])
            batch_size = int(inputs_embeds.shape[0])
            device = inputs_embeds.device
        else:
            sequence_length = 0
            batch_size = 1
            device = model.device

        if hybrid_cache_cls is not None and (
            cache_state is None or isinstance(cache_state, DynamicCache) and not hasattr(cache_state, "conv_kernel_size")
        ):
            cache_state = hybrid_cache_cls(model.config, batch_size, model.dtype, device=device)

        if cache_state is not None and kwargs.get("past_key_values") is None:
            kwargs["past_key_values"] = cache_state

        if cache_position is None and cache_state is not None:
            kwargs["cache_position"] = torch.arange(
                sequence_length,
                device=device,
                dtype=torch.long,
            ) + _past_seen_tokens(cache_state)

        model_inputs = original_prepare(*args, **kwargs)
        if "cache_params" in forward_sig.parameters and "cache_params" not in model_inputs:
            cache_for_forward = model_inputs.pop("past_key_values", None)
            if cache_for_forward is not None:
                model_inputs["cache_params"] = cache_for_forward
        return model_inputs

    setattr(model, "prepare_inputs_for_generation", patched_prepare_inputs_for_generation)


def apply_model_config_compat(config: Any) -> None:
    """Backfill config aliases expected by newer HF integration code."""

    def _patch_one(target: Any) -> None:
        if target is None or hasattr(target, "num_experts"):
            return

        expert_count = None
        if hasattr(target, "num_local_experts"):
            expert_count = getattr(target, "num_local_experts")
        elif hasattr(target, "n_routed_experts"):
            expert_count = getattr(target, "n_routed_experts")

        if expert_count is not None:
            setattr(target, "num_experts", expert_count)

    _patch_one(config)
    for attr_name in ("text_config", "language_config", "llm_config"):
        _patch_one(getattr(config, attr_name, None))


def _target_pattern_matches_param_name(pattern: str, param_name: str) -> bool:
    regex = re.escape(pattern)
    regex = regex.replace(r"\.\*\.", r"\..*\.")
    regex = regex.replace(r"\*", r".*")
    return re.search(regex, param_name) is not None


def estimate_conversion_workspace_bytes(model_loader: Any, config: Any) -> int:
    """Estimate the largest temporary tensor created by checkpoint conversion ops."""
    import torch
    from accelerate import init_empty_weights

    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import Concatenate, WeightConverter
    except ImportError as exc:
        warn(f"Could not inspect conversion mapping on this transformers version: {exc}")
        return 0

    try:
        with init_empty_weights(include_buffers=True):
            empty_model = model_loader.from_config(config, trust_remote_code=True)
    except Exception as exc:
        warn(f"Could not build empty model for conversion workspace estimate: {exc}")
        return 0

    try:
        conversions = get_model_conversion_mapping(empty_model)
        max_bytes = 0
        seen_param_names = set()
        for conversion in conversions:
            if not isinstance(conversion, WeightConverter):
                continue
            if not any(isinstance(op, Concatenate) for op in conversion.operations):
                continue
            for param_name, param in empty_model.named_parameters():
                if param_name in seen_param_names:
                    continue
                if not any(
                    _target_pattern_matches_param_name(target_pattern, param_name)
                    for target_pattern in conversion.target_patterns
                ):
                    continue
                seen_param_names.add(param_name)
                max_bytes = max(max_bytes, int(param.numel()) * torch.empty((), dtype=param.dtype).element_size())
        return max_bytes
    finally:
        del empty_model


def load_reference_model_and_tokenizer(
    model_name: str,
    *,
    attn_implementation: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the BF16 HF reference model through the shared capture path."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    model_name = resolve_model_name(model_name)
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_path):
        die(f"Model not found: {model_path}")

    info(f"Model: {model_name}")
    info(f"Path: {model_path}")
    t0 = time.time()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    apply_model_config_compat(config)
    if attn_implementation:
        try:
            config._attn_implementation = attn_implementation
        except Exception:
            pass
    info(f"Config loaded ({time.time() - t0:.1f}s)")

    architectures = list(getattr(config, "architectures", []) or [])
    use_conditional_loader = any("ConditionalGeneration" in arch for arch in architectures)
    model_loader = AutoModelForImageTextToText if use_conditional_loader else AutoModelForCausalLM
    info(
        "HF model loader: "
        f"{model_loader.__name__} (architectures={architectures if architectures else ['unknown']})"
    )

    t0 = time.time()
    tokenizer = load_tokenizer_with_compat(model_path)
    info(f"Tokenizer loaded ({time.time() - t0:.1f}s)")

    conversion_workspace_bytes = estimate_conversion_workspace_bytes(model_loader, config)
    conversion_workspace_gib = conversion_workspace_bytes / (1024 ** 3)
    if conversion_workspace_bytes > 0:
        info(f"Estimated conversion workspace reserve: {conversion_workspace_gib:.2f}GiB")

    num_gpus = torch.cuda.device_count()
    max_memory = {"cpu": "200GiB"}
    base_headroom_mb = 2048
    conversion_headroom_mb = (conversion_workspace_bytes + (1024 * 1024 - 1)) // (1024 * 1024)
    total_headroom_mb = base_headroom_mb + conversion_headroom_mb
    for i in range(num_gpus):
        free_mb = torch.cuda.mem_get_info(i)[0] // (1024 * 1024)
        alloc_gb = max(1, (free_mb - total_headroom_mb)) / 1024
        max_memory[i] = f"{alloc_gb:.1f}GiB"
        info(
            f"GPU {i}: {free_mb}MB free, allocating {alloc_gb:.1f}GiB "
            f"(headroom {total_headroom_mb}MB)"
        )
    info(f"Loading model in BF16 ({num_gpus} GPUs + CPU offload)...")

    t0 = time.time()
    load_kwargs = {
        "config": config,
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "max_memory": max_memory,
        "trust_remote_code": True,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
        info(f"Requested HF attention implementation: {attn_implementation}")
    model = model_loader.from_pretrained(model_path, **load_kwargs)
    maybe_patch_legacy_remote_cache_position(model)
    model.eval()
    load_time = time.time() - t0
    info(f"Model loaded ({load_time:.1f}s)")

    runtime_version = "unknown"
    try:
        import transformers
        runtime_version = transformers.__version__
    except Exception:
        pass

    return {
        "model_name": model_name,
        "model_path": model_path,
        "config": config,
        "model": model,
        "tokenizer": tokenizer,
        "runtime_version": runtime_version,
        "num_gpus": num_gpus,
        "load_kwargs_audit": {
            "dtype": str(load_kwargs.get("dtype")),
            "device_map": load_kwargs.get("device_map"),
            "max_memory": max_memory,
            "trust_remote_code": load_kwargs.get("trust_remote_code"),
            "attn_implementation": load_kwargs.get("attn_implementation"),
        },
    }


def build_environment_audit(
    loaded: Dict[str, Any],
    *,
    attn_implementation: Optional[str],
    diagnostic_controls: Dict[str, Any],
) -> Dict[str, Any]:
    import torch

    model = loaded["model"]
    config = loaded["config"]
    tokenizer = loaded["tokenizer"]
    generation_config = getattr(model, "generation_config", None)
    first_param_dtype = None
    first_param_device = None
    try:
        first_param = next(model.parameters())
        first_param_dtype = str(first_param.dtype)
        first_param_device = str(first_param.device)
    except Exception:
        pass

    cuda_devices = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            cuda_devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "capability": [props.major, props.minor],
                    "total_memory_bytes": int(total_bytes),
                    "free_memory_bytes": int(free_bytes),
                }
            )

    return {
        "python_executable": sys.executable,
        "packages": {
            "transformers": _package_version("transformers"),
            "torch": getattr(torch, "__version__", None),
            "accelerate": _package_version("accelerate"),
            "tokenizers": _package_version("tokenizers"),
            "safetensors": _package_version("safetensors"),
        },
        "cuda": {
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "devices": cuda_devices,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "model": {
            "name": loaded.get("model_name"),
            "path": loaded.get("model_path"),
            "class": type(model).__name__,
            "config_class": type(config).__name__,
            "architectures": list(getattr(config, "architectures", []) or []),
            "hf_device_map": _json_safe(getattr(model, "hf_device_map", None)),
            "first_parameter_dtype": first_param_dtype,
            "first_parameter_device": first_param_device,
            "load_kwargs": loaded.get("load_kwargs_audit"),
            "files": _model_file_audit(str(loaded.get("model_path"))),
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "name_or_path": getattr(tokenizer, "name_or_path", None),
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "model_max_length": getattr(tokenizer, "model_max_length", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "chat_template_sha256": hashlib.sha256(
                str(getattr(tokenizer, "chat_template", "") or "").encode("utf-8")
            ).hexdigest(),
        },
        "generation": {
            "config": _json_safe(generation_config.to_dict() if hasattr(generation_config, "to_dict") else None),
            "config_use_cache": getattr(config, "use_cache", None),
            "generation_config_use_cache": getattr(generation_config, "use_cache", None),
            "attn_implementation_arg": attn_implementation,
            "config_attn_implementation": getattr(config, "_attn_implementation", None),
            "diagnostic_controls": diagnostic_controls,
        },
    }


def generate_reference(
    model_name: str,
    max_new_tokens: int = 200,
    profile: str = "auto",
    diagnostic_steps: int = DEFAULT_DIAGNOSTIC_STEPS,
    diagnostic_top_k: int = DEFAULT_DIAGNOSTIC_TOPK,
    conversation_index: Optional[int] = None,
    turn_index: Optional[int] = None,
    turn_start_index: Optional[int] = None,
    turn_end_index: Optional[int] = None,
    first_stored_turns: Optional[int] = None,
    attn_implementation: Optional[str] = None,
    output: Optional[str] = None,
    use_cache_mode: str = "auto",
    generate_use_cache_mode: Optional[str] = None,
    diagnostic_use_cache_mode: Optional[str] = None,
    diagnostic_scope: str = "all",
    diagnose_pre_post_forward: bool = False,
    between_turn_cleanup: str = "none",
    reload_model_between_turns: bool = False,
    diagnostic_only: bool = False,
):
    """Generate reference outputs using HuggingFace transformers."""
    import torch

    if diagnostic_scope not in ("all", "final", "postponed"):
        die("--diagnostic-scope must be one of: all, final, postponed")
    generate_cache_mode = generate_use_cache_mode or use_cache_mode
    diagnostic_cache_mode = diagnostic_use_cache_mode or use_cache_mode

    model_name = resolve_model_name(model_name)
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_path):
        die(f"Model not found: {model_path}")

    if not PROMPTS_FILE.is_file():
        die(f"Prompts file not found: {PROMPTS_FILE}")

    # Parse prompts
    with open(PROMPTS_FILE) as f:
        lines = f.readlines()
    source_conversations = parse_prompt_conversations(lines)
    conversations, prompt_subset = _build_prompt_subset(
        source_conversations,
        conversation_index,
        turn_index,
        turn_start_index,
        turn_end_index,
        first_stored_turns,
    )
    total_prompts = sum(len(c) for c in conversations)
    recorded_turns = int(prompt_subset.get("recorded_turns") or total_prompts)

    info(f"Model: {model_name}")
    info(f"Path: {model_path}")
    info(f"Conversations: {len(conversations)} ({total_prompts} prompts, {recorded_turns} recorded)")
    info(f"Max new tokens: {max_new_tokens}")
    if prompt_subset.get("enabled"):
        info(f"Prompt subset: {prompt_subset}")
    if attn_implementation:
        info(f"HF attention implementation override: {attn_implementation}")
    diagnostic_controls = {
        "use_cache_mode": use_cache_mode,
        "generate_use_cache_mode": generate_cache_mode,
        "diagnostic_use_cache_mode": diagnostic_cache_mode,
        "diagnostic_scope": diagnostic_scope,
        "diagnose_pre_post_forward": diagnose_pre_post_forward,
        "between_turn_cleanup": between_turn_cleanup,
        "reload_model_between_turns": reload_model_between_turns,
        "diagnostic_only": diagnostic_only,
    }
    info(f"Diagnostic controls: {diagnostic_controls}")
    loaded = load_reference_model_and_tokenizer(model_name, attn_implementation=attn_implementation)
    model_name = loaded["model_name"]
    model_path = loaded["model_path"]
    config = loaded["config"]
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]
    cache_mode_state = _set_cache_mode(model, config, generate_cache_mode)
    generate_use_cache_value = _cache_mode_to_value(generate_cache_mode)
    diagnostic_use_cache_value = _cache_mode_to_value(diagnostic_cache_mode)
    environment_audit = build_environment_audit(
        loaded,
        attn_implementation=attn_implementation,
        diagnostic_controls=diagnostic_controls,
    )

    eos_token_ids = collect_capture_stop_ids(
        tokenizer,
        model_path=model_path,
        config_json=config.to_dict() if hasattr(config, "to_dict") else None,
    )
    info(f"EOS token IDs: {eos_token_ids}")

    profile_id = canonical_profile_id(tokenizer, profile)
    capture_settings = capture_settings_for_profile(profile_id)
    capture_settings["source"] = "local_generate_reference"
    if prompt_subset.get("enabled"):
        capture_settings["prompt_subset"] = prompt_subset
    if attn_implementation:
        capture_settings["attn_implementation"] = attn_implementation
    capture_settings["diagnostic_controls"] = diagnostic_controls
    info(f"Reference profile: {profile_id}")
    invocation = build_invocation_metadata(model_name, profile_id, max_new_tokens)
    invocation["prompt_subset"] = prompt_subset
    if attn_implementation:
        invocation["attn_implementation"] = attn_implementation
    invocation["diagnostic_controls"] = diagnostic_controls
    invocation["environment_audit"] = environment_audit

    # Generate reference outputs
    result = {
        "format_version": 5,
        "model": model_name,
        "model_path": model_path,
        "generated_at": datetime.now().isoformat(),
        "runtime": "transformers",
        "profile_id": profile_id,
        "max_new_tokens": max_new_tokens,
        "eos_token_ids": eos_token_ids,
        "capture_settings": capture_settings,
        "capture_invocation": invocation,
        "decode_diagnostics": {
            "schema_version": 1,
            "coverage": "teacher_forced_first_generated_steps",
            "captured_steps_per_turn": diagnostic_steps,
            "top_k": diagnostic_top_k,
            "log_prob_base": "natural_log",
        },
        "prompt_subset": prompt_subset,
        "attn_implementation": attn_implementation,
        "environment_audit": environment_audit,
        "conversations": [],
    }

    result["runtime_version"] = loaded["runtime_version"]

    prompt_num = 0
    for conv_idx, conversation in enumerate(conversations):
        conv_result: Dict[str, Any] = {"turns": []}
        messages: List[Dict[str, str]] = []
        is_multi = len(conversation) > 1

        if is_multi:
            info(f"Conversation {conv_idx + 1} ({len(conversation)} turns)")

        for turn_idx, prompt in enumerate(conversation):
            cleanup_state: Optional[Dict[str, Any]] = None
            reloaded_before_turn = False
            if turn_idx > 0:
                cleanup_state = _cleanup_between_turns(model, config, between_turn_cleanup)
                if reload_model_between_turns:
                    info(f"Reloading HF model before turn {turn_idx} as requested")
                    del model
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    loaded = load_reference_model_and_tokenizer(model_name, attn_implementation=attn_implementation)
                    model_name = loaded["model_name"]
                    model_path = loaded["model_path"]
                    config = loaded["config"]
                    model = loaded["model"]
                    tokenizer = loaded["tokenizer"]
                    cache_mode_state = _set_cache_mode(model, config, generate_cache_mode)
                    reloaded_before_turn = True

            prompt_num += 1
            turn_label = f" [turn {turn_idx + 1}/{len(conversation)}]" if is_multi else ""
            print(f"  {GREEN}[{prompt_num}/{total_prompts}]{NC} {prompt}{turn_label}")

            messages.append({"role": "user", "content": prompt})

            # Apply chat template — disable thinking if the model supports it
            # We want the "normal" response for reference, not thinking tokens
            try:
                template_out = apply_capture_template(tokenizer, messages, capture_settings)
            except TypeError:
                die(
                    "Requested reference profile is incompatible with this tokenizer/chat template: "
                    f"profile={profile_id}"
                )

            # apply_chat_template may return a tensor or a BatchEncoding
            if hasattr(template_out, "input_ids"):
                input_ids = template_out.input_ids
            elif isinstance(template_out, torch.Tensor):
                input_ids = template_out
            else:
                input_ids = torch.tensor([template_out], dtype=torch.long)

            input_ids = input_ids.to(model.device)
            input_len = input_ids.shape[1]
            cache_snapshot_before_turn = _cache_state_snapshot(model, config)
            pre_generate_logits = None
            should_record = _should_record_turn(prompt_subset, turn_idx)
            should_capture_diagnostics = should_record and _should_capture_immediate_diagnostics(
                prompt_subset,
                turn_idx,
                diagnostic_scope,
            )
            if diagnose_pre_post_forward and should_capture_diagnostics:
                pre_generate_logits = capture_prefill_logits(
                    model,
                    input_ids,
                    use_cache=diagnostic_use_cache_value,
                )

            t0 = time.time()
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,  # Greedy decoding
                "eos_token_id": eos_token_ids if eos_token_ids else None,
                "attention_mask": torch.ones_like(input_ids),
            }
            if generate_use_cache_value is not None:
                generate_kwargs["use_cache"] = generate_use_cache_value
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is not None:
                generate_kwargs["pad_token_id"] = pad_token_id
            cache_snapshot_before_generate = _cache_state_snapshot(model, config)
            with torch.no_grad():
                output_ids = model.generate(input_ids, **generate_kwargs)
            cache_snapshot_after_generate = _cache_state_snapshot(model, config)
            gen_time = time.time() - t0

            # Extract only new tokens
            new_token_ids = output_ids[0][input_len:].tolist()

            # Strip trailing EOS/pad tokens
            while new_token_ids and new_token_ids[-1] in eos_token_ids:
                new_token_ids.pop()

            text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
            tok_s = len(new_token_ids) / gen_time if gen_time > 0 else 0

            preview = text[:120].replace("\n", " ")
            print(f"    {DIM}{len(new_token_ids)} tokens, {gen_time:.1f}s ({tok_s:.1f} tok/s){NC}")
            print(f"    {DIM}{preview}{'...' if len(text) > 120 else ''}{NC}")

            if should_record:
                source_conversation_indices = prompt_subset.get("source_conversation_indices")
                if isinstance(source_conversation_indices, list) and conv_idx < len(source_conversation_indices):
                    source_conversation_index = source_conversation_indices[conv_idx]
                elif prompt_subset.get("conversation_index") is not None:
                    source_conversation_index = prompt_subset.get("conversation_index")
                else:
                    source_conversation_index = conv_idx
                turn_result = {
                    "prompt": prompt,
                    "source_conversation_index": source_conversation_index,
                    "source_turn_index": turn_idx,
                    "input_token_ids": input_ids[0].tolist(),
                    "token_ids": new_token_ids,
                    "text": text,
                    "num_tokens": len(new_token_ids),
                    "generation_state": {
                        "model_training": bool(getattr(model, "training", False)),
                        "model_eval": not bool(getattr(model, "training", False)),
                        "attn_implementation": attn_implementation or getattr(config, "_attn_implementation", None),
                        "diagnostic_controls": diagnostic_controls,
                        "use_cache_mode": use_cache_mode,
                        "generate_use_cache_mode": generate_cache_mode,
                        "diagnostic_use_cache_mode": diagnostic_cache_mode,
                        "diagnostic_scope": diagnostic_scope,
                        "use_cache_generate_arg": generate_use_cache_value,
                        "use_cache_forward_arg": diagnostic_use_cache_value,
                        "use_cache_config": bool(getattr(config, "use_cache", True)),
                        "config_use_cache": getattr(config, "use_cache", None),
                        "generation_config_use_cache": getattr(getattr(model, "generation_config", None), "use_cache", None),
                        "cache_mode_state": cache_mode_state,
                        "between_turn_cleanup": cleanup_state,
                        "reloaded_model_before_turn": reloaded_before_turn,
                        "cache_snapshot_before_turn": cache_snapshot_before_turn,
                        "cache_snapshot_before_generate": cache_snapshot_before_generate,
                        "cache_snapshot_after_generate": cache_snapshot_after_generate,
                        "generate_kwargs_has_past_key_values": "past_key_values" in generate_kwargs,
                        "generate_kwargs_has_cache_params": "cache_params" in generate_kwargs,
                        "explicit_past_key_values_supplied": False,
                        "explicit_cache_object_supplied": False,
                        "past_key_values_reused_from_prior_turn": False,
                        "diagnostics_captured": should_capture_diagnostics,
                        "diagnostics_postponed": diagnostic_scope == "postponed",
                        "diagnostics_skipped_reason": (
                            None
                            if should_capture_diagnostics
                            else (
                                "postponed"
                                if diagnostic_scope == "postponed"
                                else "not_final_recorded_turn"
                            )
                        ),
                    },
                }
                if pre_generate_logits is not None:
                    turn_result["pre_generate_diagnostic"] = build_prefill_diagnostic_from_logits(
                        tokenizer,
                        input_ids,
                        new_token_ids,
                        pre_generate_logits,
                        diagnostic_top_k=diagnostic_top_k,
                    )
                if should_capture_diagnostics:
                    per_token_data = build_teacher_forced_per_token_data(
                        model,
                        tokenizer,
                        input_ids,
                        new_token_ids,
                        diagnostic_steps=diagnostic_steps,
                        diagnostic_top_k=diagnostic_top_k,
                        use_cache=diagnostic_use_cache_value,
                    )
                    if per_token_data:
                        turn_result["per_token_data"] = per_token_data
                if diagnose_pre_post_forward and should_capture_diagnostics:
                    post_generate_logits = capture_prefill_logits(
                        model,
                        input_ids,
                        use_cache=diagnostic_use_cache_value,
                    )
                    turn_result["post_generate_diagnostic"] = build_prefill_diagnostic_from_logits(
                        tokenizer,
                        input_ids,
                        new_token_ids,
                        post_generate_logits,
                        diagnostic_top_k=diagnostic_top_k,
                    )
                    turn_result["generation_state"]["cache_snapshot_after_diagnostics"] = _cache_state_snapshot(model, config)
                conv_result["turns"].append(turn_result)

            # Add assistant response to history for multi-turn
            messages.append({"role": "assistant", "content": text})

        if conv_result["turns"]:
            result["conversations"].append(conv_result)

    if diagnostic_scope == "postponed":
        info("Running postponed diagnostics after all selected turns were generated")
        for conv in result["conversations"]:
            for turn in conv["turns"]:
                input_ids = torch.tensor([turn["input_token_ids"]], dtype=torch.long, device=model.device)
                new_token_ids = turn.get("token_ids", [])
                turn["generation_state"]["cache_snapshot_before_postponed_diagnostics"] = _cache_state_snapshot(model, config)
                per_token_data = build_teacher_forced_per_token_data(
                    model,
                    tokenizer,
                    input_ids,
                    new_token_ids,
                    diagnostic_steps=diagnostic_steps,
                    diagnostic_top_k=diagnostic_top_k,
                    use_cache=diagnostic_use_cache_value,
                )
                if per_token_data:
                    turn["per_token_data"] = per_token_data
                if diagnose_pre_post_forward:
                    postponed_logits = capture_prefill_logits(
                        model,
                        input_ids,
                        use_cache=diagnostic_use_cache_value,
                    )
                    turn["postponed_prefill_diagnostic"] = build_prefill_diagnostic_from_logits(
                        tokenizer,
                        input_ids,
                        new_token_ids,
                        postponed_logits,
                        diagnostic_top_k=diagnostic_top_k,
                    )
                turn["generation_state"]["diagnostics_captured"] = True
                turn["generation_state"]["diagnostics_postponed"] = True
                turn["generation_state"]["diagnostics_skipped_reason"] = None
                turn["generation_state"]["cache_snapshot_after_diagnostics"] = _cache_state_snapshot(model, config)

    result["contract"] = build_contract(
        model_name=model_name,
        model_path=model_path,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        add_generation_prompt=True,
        enable_thinking=capture_settings["enable_thinking"],
        profile_id=capture_settings["profile_id"],
        prompt_source_path=str(PROMPTS_FILE),
        runtime_name="transformers",
        runtime_version=result.get("runtime_version"),
        torch_dtype="bfloat16",
        extra={
            "capture_invocation": invocation,
            "prompt_subset": prompt_subset,
            "attn_implementation": attn_implementation,
            "diagnostic_controls": diagnostic_controls,
        },
    )
    result["sanity"] = build_reference_sanity_report(
        result,
        expected_conversations=len(conversations),
        expected_turns=recorded_turns,
    )
    output_path = _diagnostic_output_path(model_name, profile_id, prompt_subset, attn_implementation, output)
    sanity = result["sanity"]
    if sanity.get("status") != "pass":
        warn(
            "Reference sanity gate: FAIL "
            + ", ".join(sanity.get("failed_checks", []))
        )
        for check in sanity.get("checks", []):
            if not check.get("ok"):
                warn(f"  {check.get('name')}: {check.get('detail')}")
        for sample in sanity.get("thought_leakage", {}).get("samples", []):
            warn(
                "  thought sample "
                f"conv={sample.get('conversation_index')} turn={sample.get('turn_index')}: "
                f"{sample.get('preview')}"
            )
        write_generation_failure_manifest(
            invocation,
            intended_output_path=output_path,
            model_name=model_name,
            profile_id=profile_id,
            sanity=sanity,
            prompt_subset=prompt_subset,
            attn_implementation=attn_implementation,
            turn_summaries=summarize_reference_turns(result, diagnostic_top_k),
            diagnostic_controls=diagnostic_controls,
            environment_audit=environment_audit,
            token_audit_turns=build_token_audit_turns(result),
        )
        die("Reference capture failed strict sanity gates; no reference artifact was written.")

    if diagnostic_only:
        write_generation_diagnostic_manifest(
            invocation,
            intended_output_path=output_path,
            model_name=model_name,
            profile_id=profile_id,
            sanity=sanity,
            prompt_subset=prompt_subset,
            attn_implementation=attn_implementation,
            turn_summaries=summarize_reference_turns(result, diagnostic_top_k),
            diagnostic_controls=diagnostic_controls,
            environment_audit=environment_audit,
            token_audit_turns=build_token_audit_turns(result),
        )
        ok("Diagnostic-only capture complete; no reference artifact was written.")
        return

    # Save reference only after strict sanity gates pass.
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)

    emit_reference_generation_trace(
        "generate_reference",
        result["contract"],
        str(output_path),
        max_new_tokens=max_new_tokens,
    )
    write_run_manifest(
        invocation,
        output_path,
        model_name,
        profile_id,
        max_new_tokens,
        diagnostic_steps,
        diagnostic_top_k,
        prompt_subset,
        attn_implementation,
        result["sanity"],
    )

    ok(f"Reference saved: {output_path}")
    ok(
        "Reference metadata: "
        f"profile={result['contract'].get('profile_id')} "
        f"template_hash={result['contract'].get('tokenizer', {}).get('chat_template_hash')} "
        f"prompt_sha256={result['contract'].get('prompt_source', {}).get('sha256')}"
    )
    ok(f"Total: {total_prompts} prompts, {sum(len(c['turns']) for c in result['conversations'])} recorded turns")
    ok("Reference sanity gate: PASS")


def main():
    if os.environ.get("KRASIS_ALLOW_ARCHIVED_HF_REFERENCE") != "1":
        print("ERROR: ./dev generate-reference is archived.")
        print("  HF/Transformers reference capture is no longer trusted by default.")
        print("  Use llama-witness for new reference authority.")
        print("  For forensic reruns only, set KRASIS_ALLOW_ARCHIVED_HF_REFERENCE=1 and document the reason.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Generate BF16 HuggingFace greedy reference outputs with stored contract metadata"
    )
    parser.add_argument("model", help="Model directory name under the active capture models dir")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max new tokens per turn (default: 200)")
    parser.add_argument(
        "--profile",
        default="auto",
        choices=("auto",) + REFERENCE_PROFILES,
        help=(
            "Reference capture profile. auto chooses greedy_chat_thinking_off when "
            "the tokenizer chat template supports enable_thinking, otherwise greedy_chat_default"
        ),
    )
    parser.add_argument(
        "--diag-steps",
        type=int,
        default=DEFAULT_DIAGNOSTIC_STEPS,
        help=f"Teacher-forced decode diagnostic coverage per turn (default: {DEFAULT_DIAGNOSTIC_STEPS})",
    )
    parser.add_argument(
        "--diag-topk",
        type=int,
        default=DEFAULT_DIAGNOSTIC_TOPK,
        help=f"Top-k entries to store for each diagnostic step (default: {DEFAULT_DIAGNOSTIC_TOPK})",
    )
    parser.add_argument(
        "--conversation-index",
        type=int,
        default=None,
        help=(
            "Restrict capture to one zero-based conversation from benchmarks/sanity_test_prompts.txt. "
            "Use with --turn-index for a single recorded turn."
        ),
    )
    parser.add_argument(
        "--turn-index",
        type=int,
        default=None,
        help=(
            "Restrict capture to one zero-based turn within --conversation-index. "
            "Prior turns in that conversation are generated only to preserve chat history."
        ),
    )
    parser.add_argument(
        "--turn-start-index",
        type=int,
        default=None,
        help=(
            "Record a zero-based turn range within --conversation-index. "
            "Prior turns before the range are generated only to preserve chat history."
        ),
    )
    parser.add_argument(
        "--turn-end-index",
        type=int,
        default=None,
        help=(
            "End of the zero-based turn range recorded by --turn-start-index. "
            "If omitted with --turn-start-index, records through the end of the conversation."
        ),
    )
    parser.add_argument(
        "--first-stored-turns",
        type=int,
        default=None,
        help=(
            "Record the first N stored turns in global prompt order while preserving conversation grouping. "
            "Used for global-order diagnostics; cannot be combined with conversation/turn selectors."
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
        help="Optional HuggingFace attention implementation override for this reference capture",
    )
    parser.add_argument(
        "--use-cache",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Legacy cache mode applied to generate and diagnostics unless the split cache flags are supplied. "
            "auto leaves HF defaults unchanged; on/off explicitly sets model/generation config."
        ),
    )
    parser.add_argument(
        "--generate-use-cache",
        choices=("auto", "on", "off"),
        default=None,
        help="Cache mode passed to model.generate only. Defaults to --use-cache.",
    )
    parser.add_argument(
        "--diagnostic-use-cache",
        choices=("auto", "on", "off"),
        default=None,
        help="Cache mode passed to diagnostic forward calls only. Defaults to --use-cache.",
    )
    parser.add_argument(
        "--diagnostic-scope",
        choices=("all", "final", "postponed"),
        default="all",
        help=(
            "Diagnostic capture scope for recorded turns: all, final recorded turn only, "
            "or postponed until after all selected turns are generated."
        ),
    )
    parser.add_argument(
        "--diagnose-pre-post-forward",
        action="store_true",
        help="For recorded turns, capture first-token diagnostic forward before and after model.generate.",
    )
    parser.add_argument(
        "--between-turn-cleanup",
        choices=("none", "gc", "reset"),
        default="none",
        help="Cleanup action before each turn after the first: none, gc/CUDA cache cleanup, or reset config/model cache attrs.",
    )
    parser.add_argument(
        "--reload-model-between-turns",
        action="store_true",
        help="Reload the HF model before each turn after the first while preserving generated chat history.",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Run generation and strict gates but write only a diagnostic manifest, not a reference JSON artifact.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional explicit output JSON path. By default, subset or attention-override captures "
            "write a diagnostic non-canonical filename under tests/reference_outputs/<model>/."
        ),
    )
    args = parser.parse_args()

    if args.diag_steps < 0:
        die("--diag-steps must be >= 0")
    if args.diag_topk < 0:
        die("--diag-topk must be >= 0")

    generate_reference(
        args.model,
        args.max_tokens,
        args.profile,
        diagnostic_steps=args.diag_steps,
        diagnostic_top_k=args.diag_topk,
        conversation_index=args.conversation_index,
        turn_index=args.turn_index,
        turn_start_index=args.turn_start_index,
        turn_end_index=args.turn_end_index,
        first_stored_turns=args.first_stored_turns,
        attn_implementation=args.attn_implementation,
        output=args.output,
        use_cache_mode=args.use_cache,
        generate_use_cache_mode=args.generate_use_cache,
        diagnostic_use_cache_mode=args.diagnostic_use_cache,
        diagnostic_scope=args.diagnostic_scope,
        diagnose_pre_post_forward=args.diagnose_pre_post_forward,
        between_turn_cleanup=args.between_turn_cleanup,
        reload_model_between_turns=args.reload_model_between_turns,
        diagnostic_only=args.diagnostic_only,
    )


if __name__ == "__main__":
    main()
