#!/usr/bin/env python3
"""Backfill exact raw-token top-k/logprob diagnostics into a reference artifact.

Usage:
    ./dev reference-backfill-topk <model> --reference greedy_reference.json

This script must be run via ./dev reference-backfill-topk, not directly.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from generate_reference import (
        build_invocation_metadata,
        build_prefill_next_token_data,
        build_teacher_forced_per_token_data,
        die,
        emit_reference_generation_trace,
        info,
        load_reference_model_and_tokenizer,
        ok,
        resolve_model_name,
        warn,
    )
    from reference_contract import build_reference_sanity_report, validate_contracts, build_contract
except ModuleNotFoundError:
    from tests.generate_reference import (
        build_invocation_metadata,
        build_prefill_next_token_data,
        build_teacher_forced_per_token_data,
        die,
        emit_reference_generation_trace,
        info,
        load_reference_model_and_tokenizer,
        ok,
        resolve_model_name,
        warn,
    )
    from tests.reference_contract import build_reference_sanity_report, validate_contracts, build_contract


if not os.environ.get("KRASIS_DEV_SCRIPT"):
    print("ERROR: This script must be run via ./dev reference-backfill-topk, not directly.")
    print("  Usage: ./dev reference-backfill-topk <model> --reference <greedy_reference.json>")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill teacher-forced top-k/logprob diagnostics from exact stored raw input_token_ids"
    )
    parser.add_argument("model", help="Model directory name or alias under the active capture models dir")
    parser.add_argument("--reference", required=True, help="Existing reference JSON with input_token_ids and token_ids")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <reference stem>_backfilled_topk.json beside the input.",
    )
    parser.add_argument("--diag-steps", type=int, default=1, help="Generated-token diagnostic steps to backfill")
    parser.add_argument("--diag-topk", type=int, default=10, help="Top-k logprob entries per diagnostic step")
    parser.add_argument(
        "--diagnose-turn",
        action="append",
        default=[],
        metavar="CONV:TURN",
        help="Diagnose one exact turn without writing output. May be repeated.",
    )
    parser.add_argument(
        "--diagnose-prefix-search",
        action="store_true",
        help="For diagnosed turns, probe exact raw-token prefixes to locate the first NaN-producing prefix.",
    )
    parser.add_argument(
        "--diagnose-prefix-lengths",
        default=None,
        help="Comma-separated exact raw-token prefix lengths to probe for diagnosed turns.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=("eager", "sdpa", "flash_attention_2"),
        help="Request a specific HF attention implementation for diagnostic/backfill model loading.",
    )
    return parser.parse_args()


def _default_output_path(reference_path: Path) -> Path:
    return reference_path.with_name(f"{reference_path.stem}_backfilled_topk{reference_path.suffix}")


def _turns(reference: Dict[str, Any]) -> List[Tuple[int, int, Dict[str, Any]]]:
    turns: List[Tuple[int, int, Dict[str, Any]]] = []
    for conv_idx, conv in enumerate(reference.get("conversations", [])):
        for turn_idx, turn in enumerate(conv.get("turns", [])):
            turns.append((conv_idx, turn_idx, turn))
    return turns


def _require_reference_schema(reference: Dict[str, Any], diag_steps: int) -> None:
    if not isinstance(reference.get("contract"), dict):
        die("Reference artifact is missing a top-level contract")
    if not isinstance(reference.get("profile_id"), str) or not reference["profile_id"]:
        die("Reference artifact is missing profile_id")
    if not isinstance(reference.get("model"), str) or not reference["model"]:
        die("Reference artifact is missing model")
    turns = _turns(reference)
    if not turns:
        die("Reference artifact contains no turns")
    missing_raw = []
    missing_generated = []
    too_short = []
    for conv_idx, turn_idx, turn in turns:
        input_token_ids = turn.get("input_token_ids")
        token_ids = turn.get("token_ids")
        if not isinstance(input_token_ids, list) or not input_token_ids:
            missing_raw.append((conv_idx, turn_idx))
        if not isinstance(token_ids, list) or not token_ids:
            missing_generated.append((conv_idx, turn_idx))
        elif diag_steps > len(token_ids):
            too_short.append((conv_idx, turn_idx, len(token_ids)))
    if missing_raw:
        die(f"Turns missing raw input_token_ids: {missing_raw[:5]} count={len(missing_raw)}")
    if missing_generated:
        die(f"Turns missing generated token_ids: {missing_generated[:5]} count={len(missing_generated)}")
    if too_short:
        die(f"--diag-steps exceeds generated token count for turns: {too_short[:5]} count={len(too_short)}")


def _reference_model_names(reference: Dict[str, Any]) -> List[str]:
    names = []
    for value in (
        reference.get("model"),
        Path(str(reference.get("model_path", ""))).name if reference.get("model_path") else None,
        reference.get("contract", {}).get("model", {}).get("name"),
        reference.get("contract", {}).get("model", {}).get("path_basename"),
    ):
        if isinstance(value, str) and value and value not in names:
            names.append(value)
    return names


def _validate_contract_identity(reference: Dict[str, Any], current_contract: Dict[str, Any], cli_model: str) -> None:
    resolved_cli = resolve_model_name(cli_model)
    ref_names = _reference_model_names(reference)
    if resolved_cli not in ref_names:
        die(f"CLI model {resolved_cli!r} does not match reference model names {ref_names}")

    stored_contract = reference["contract"]
    validation = validate_contracts(stored_contract, current_contract)
    if validation.get("errors"):
        die("Reference contract does not match local model: " + "; ".join(validation["errors"]))

    top_profile = reference.get("profile_id")
    contract_profile = stored_contract.get("profile_id")
    if top_profile != contract_profile:
        die(f"profile mismatch inside reference: top-level={top_profile} contract={contract_profile}")


def _validate_per_token_data(
    turn: Dict[str, Any],
    per_token_data: List[Dict[str, Any]],
    diag_steps: int,
    diag_topk: int,
) -> None:
    token_ids = turn["token_ids"]
    if len(per_token_data) != diag_steps:
        raise ValueError(f"expected {diag_steps} per_token_data entries, got {len(per_token_data)}")
    for idx, entry in enumerate(per_token_data):
        if entry.get("token_id") != token_ids[idx]:
            raise ValueError(
                f"diagnostic token mismatch at step {idx}: "
                f"entry={entry.get('token_id')} token_ids={token_ids[idx]}"
            )
        for key in ("log_prob", "logit"):
            value = entry.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"non-finite {key} at step {idx}: {value!r}")
        top_k = entry.get("top_k")
        if not isinstance(top_k, list) or len(top_k) != diag_topk:
            raise ValueError(f"expected top_k length {diag_topk} at step {idx}, got {len(top_k) if isinstance(top_k, list) else 'missing'}")
        for top_idx, top_entry in enumerate(top_k):
            value = top_entry.get("log_prob") if isinstance(top_entry, dict) else None
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"non-finite top_k[{top_idx}].log_prob at step {idx}: {value!r}")


def _write_run_manifest(invocation: Dict[str, Any], output_path: Path, summary: Dict[str, Any]) -> None:
    run_dir = invocation.get("run_dir")
    if not run_dir:
        return
    manifest_path = Path(run_dir) / "reference_backfill_topk_manifest.json"
    manifest = {
        "kind": "reference_backfill_topk_manifest",
        "captured_at": invocation.get("captured_at"),
        "dev_command": invocation.get("dev_command"),
        "script_argv": invocation.get("script_argv"),
        "cwd": invocation.get("cwd"),
        "run_dir": run_dir,
        "run_type": invocation.get("run_type"),
        "model": invocation.get("model"),
        "profile_id": invocation.get("profile_id"),
        "reference_output_path": str(output_path.resolve()),
        "summary": summary,
        "follow_up_commands": [
            "./dev reference-inventory",
            "./dev validate <config> --max-prompts 0 --no-server --port 1",
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    ok(f"Run manifest saved: {manifest_path}")


def _write_failure_manifest(invocation: Dict[str, Any], summary: Dict[str, Any]) -> None:
    run_dir = invocation.get("run_dir")
    if not run_dir:
        return
    manifest_path = Path(run_dir) / "reference_backfill_topk_failure.json"
    manifest = {
        "kind": "reference_backfill_topk_failure",
        "captured_at": invocation.get("captured_at"),
        "dev_command": invocation.get("dev_command"),
        "script_argv": invocation.get("script_argv"),
        "cwd": invocation.get("cwd"),
        "run_dir": run_dir,
        "run_type": invocation.get("run_type"),
        "model": invocation.get("model"),
        "profile_id": invocation.get("profile_id"),
        "summary": summary,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    warn(f"Failure manifest saved: {manifest_path}")


def _parse_turn_selector(selector: str) -> Tuple[int, int]:
    parts = selector.split(":", 1)
    if len(parts) != 2:
        die(f"Invalid --diagnose-turn {selector!r}; expected CONV:TURN")
    try:
        conv_idx = int(parts[0])
        turn_idx = int(parts[1])
    except ValueError:
        die(f"Invalid --diagnose-turn {selector!r}; indices must be integers")
    if conv_idx < 0 or turn_idx < 0:
        die(f"Invalid --diagnose-turn {selector!r}; indices must be non-negative")
    return conv_idx, turn_idx


def _find_turn(reference: Dict[str, Any], conv_idx: int, turn_idx: int) -> Dict[str, Any]:
    try:
        turn = reference["conversations"][conv_idx]["turns"][turn_idx]
    except (KeyError, IndexError, TypeError):
        die(f"Requested turn not found: conv={conv_idx} turn={turn_idx}")
    if not isinstance(turn, dict):
        die(f"Requested turn is not an object: conv={conv_idx} turn={turn_idx}")
    return turn


def _tensor_stats(tensor: Any) -> Dict[str, Any]:
    import torch

    detached = tensor.detach()
    flat = detached.reshape(-1)
    finite_mask = torch.isfinite(flat)
    finite_count = int(finite_mask.count_nonzero().item())
    total = int(flat.numel())
    nan_count = int(torch.isnan(flat).count_nonzero().item())
    pos_inf_count = int(torch.isposinf(flat).count_nonzero().item())
    neg_inf_count = int(torch.isneginf(flat).count_nonzero().item())
    finite_values = flat[finite_mask]
    return {
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "shape": [int(dim) for dim in detached.shape],
        "total": total,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "pos_inf_count": pos_inf_count,
        "neg_inf_count": neg_inf_count,
        "finite_min": float(finite_values.min().item()) if finite_count else None,
        "finite_max": float(finite_values.max().item()) if finite_count else None,
        "all_finite": finite_count == total,
    }


def _compact_logits_stats(logits: Any) -> Dict[str, Any]:
    stats = _tensor_stats(logits)
    return {
        "dtype": stats["dtype"],
        "device": stats["device"],
        "shape": stats["shape"],
        "total": stats["total"],
        "finite_count": stats["finite_count"],
        "nan_count": stats["nan_count"],
        "pos_inf_count": stats["pos_inf_count"],
        "neg_inf_count": stats["neg_inf_count"],
        "finite_min": stats["finite_min"],
        "finite_max": stats["finite_max"],
        "all_finite": stats["all_finite"],
    }


def _decode_piece(tokenizer: Any, token_id: int) -> Dict[str, Any]:
    token = None
    decoded = ""
    try:
        token = tokenizer.convert_ids_to_tokens(int(token_id))
    except Exception:
        token = None
    try:
        decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        decoded = ""
    return {
        "token_id": int(token_id),
        "token": token,
        "decoded": decoded,
        "is_special": int(token_id) in set(int(x) for x in (getattr(tokenizer, "all_special_ids", []) or [])),
    }


def _tokenizer_metadata(tokenizer: Any) -> Dict[str, Any]:
    special_ids = [int(x) for x in (getattr(tokenizer, "all_special_ids", []) or [])]
    special_tokens = list(getattr(tokenizer, "all_special_tokens", []) or [])
    metadata = {
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "len_tokenizer": int(len(tokenizer)) if hasattr(tokenizer, "__len__") else None,
        "special_ids": special_ids,
        "special_tokens": special_tokens,
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if metadata[key] is not None:
            metadata[key] = int(metadata[key])
    return metadata


def _prefix_token_audit(tokenizer: Any, token_ids: List[int]) -> Dict[str, Any]:
    metadata = _tokenizer_metadata(tokenizer)
    vocab_limit = metadata["len_tokenizer"] or metadata["vocab_size"]
    invalid_ids = [
        int(token_id)
        for token_id in token_ids
        if int(token_id) < 0 or (vocab_limit and int(token_id) >= int(vocab_limit))
    ]
    decoded = tokenizer.decode([int(x) for x in token_ids], skip_special_tokens=False)
    try:
        reencoded = tokenizer.encode(decoded, add_special_tokens=False)
    except Exception:
        reencoded = []
    return {
        "token_ids": [int(x) for x in token_ids],
        "token_pieces": [_decode_piece(tokenizer, int(x)) for x in token_ids],
        "validity": {
            "all_integer_ids": all(isinstance(x, int) for x in token_ids),
            "all_ids_in_tokenizer_range": not invalid_ids,
            "invalid_ids": invalid_ids,
            "vocab_limit_used": int(vocab_limit) if vocab_limit else None,
        },
        "decoded_text": decoded,
        "round_trip": {
            "reencoded_token_ids": [int(x) for x in reencoded],
            "exact_match": [int(x) for x in reencoded] == [int(x) for x in token_ids],
            "source_len": len(token_ids),
            "reencoded_len": len(reencoded),
        },
    }


def _turn_source_text_audit(tokenizer: Any, turn: Dict[str, Any]) -> Dict[str, Any]:
    source = turn.get("prompt")
    if not isinstance(source, str):
        source = turn.get("text") if isinstance(turn.get("text"), str) else None
    if source is None:
        return {"source_text_available": False}
    try:
        encoded = tokenizer.encode(source, add_special_tokens=False)
    except Exception as exc:
        return {"source_text_available": True, "source_text": source, "encode_error": str(exc)}
    return {
        "source_text_available": True,
        "source_text": source,
        "source_text_token_ids": [int(x) for x in encoded],
        "source_text_token_pieces": [_decode_piece(tokenizer, int(x)) for x in encoded],
    }


def _model_attention_implementation(model: Any, requested: Optional[str]) -> Dict[str, Any]:
    config = getattr(model, "config", None)
    return {
        "requested": requested,
        "config_attn_implementation": getattr(config, "_attn_implementation", None),
        "config_attn_implementation_internal": getattr(config, "_attn_implementation_internal", None),
        "model_class": model.__class__.__name__,
        "model_module": model.__class__.__module__,
    }


def _expected_token_stats(logits: Any, expected_token_id: int) -> Dict[str, Any]:
    import torch

    logits_f32 = logits.float()
    log_probs = torch.log_softmax(logits_f32, dim=-1)
    expected_logit = logits_f32[expected_token_id]
    expected_log_prob = log_probs[expected_token_id]
    return {
        "expected_token_id": int(expected_token_id),
        "expected_logit": float(expected_logit.item()) if torch.isfinite(expected_logit).item() else str(expected_logit.item()),
        "expected_logit_finite": bool(torch.isfinite(expected_logit).item()),
        "expected_log_prob": float(expected_log_prob.item()) if torch.isfinite(expected_log_prob).item() else str(expected_log_prob.item()),
        "expected_log_prob_finite": bool(torch.isfinite(expected_log_prob).item()),
        "rank": int(torch.count_nonzero(logits_f32 > expected_logit).item()) + 1
        if torch.isfinite(expected_logit).item()
        else None,
        "raw_logits": _tensor_stats(logits),
        "float_logits": _tensor_stats(logits_f32),
        "log_softmax": _tensor_stats(log_probs),
        "nan_stage": "raw_logits"
        if not _tensor_stats(logits)["all_finite"]
        else ("log_softmax" if not _tensor_stats(log_probs)["all_finite"] else "none"),
    }


def _generate_first_token_logits(
    model: Any,
    tokenizer: Any,
    input_ids: Any,
) -> Tuple[Any, Optional[int]]:
    import torch

    generate_kwargs = {
        "max_new_tokens": 1,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
        "attention_mask": torch.ones_like(input_ids),
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id
    with torch.no_grad():
        generated = model.generate(input_ids, **generate_kwargs)
    scores = getattr(generated, "scores", None)
    if not scores:
        raise RuntimeError("HF generate returned no scores for first-token diagnostic")
    model_generated_token_id: Optional[int] = None
    sequence = getattr(generated, "sequences", None)
    if sequence is not None and sequence.shape[1] > input_ids.shape[1]:
        model_generated_token_id = int(sequence[0, input_ids.shape[1]].item())
    return scores[0][0], model_generated_token_id


def _teacher_forced_next_token_logits(model: Any, input_ids: Any) -> Any:
    import torch

    with torch.no_grad():
        output = model(input_ids, attention_mask=torch.ones_like(input_ids))
    logits = output.logits[0]
    return logits[int(input_ids.shape[1]) - 1]


def _diagnose_generate_scores(
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    generated_token_ids: List[int],
    *,
    diagnostic_top_k: int,
) -> Dict[str, Any]:
    step_logits, model_generated_token_id = _generate_first_token_logits(model, tokenizer, input_ids)
    expected_token_id = int(generated_token_ids[0])
    stats = _expected_token_stats(step_logits, expected_token_id)
    stats.update(
        {
            "method": "generate_scores",
            "prompt_length": int(input_ids.shape[1]),
            "expected_first_token_id": expected_token_id,
            "model_generated_first_token_id": model_generated_token_id,
            "model_matches_reference_first_token": model_generated_token_id == expected_token_id,
            "diag_topk": int(diagnostic_top_k),
        }
    )
    return stats


def _diagnose_teacher_forced(
    model: Any,
    input_ids: Any,
    generated_token_ids: List[int],
) -> Dict[str, Any]:
    expected_token_id = int(generated_token_ids[0])
    step_logits = _teacher_forced_next_token_logits(model, input_ids)
    stats = _expected_token_stats(step_logits, expected_token_id)
    stats.update(
        {
            "method": "teacher_forced_forward",
            "prompt_length": int(input_ids.shape[1]),
            "expected_first_token_id": expected_token_id,
            "model_generated_first_token_id": None,
            "model_matches_reference_first_token": None,
        }
    )
    return stats


def _prefix_probe(
    method: str,
    model: Any,
    tokenizer: Any,
    full_input_ids: Any,
    prefix_len: int,
    *,
    include_token_audit: bool = False,
) -> Dict[str, Any]:
    import torch

    prefix_len = int(prefix_len)
    prompt_len = int(full_input_ids.shape[1])
    if prefix_len <= 0 or prefix_len > prompt_len:
        raise ValueError(f"invalid prefix_len={prefix_len} for prompt_len={prompt_len}")
    prefix_input_ids = full_input_ids[:, :prefix_len]
    if method == "generate_scores":
        logits, generated_token_id = _generate_first_token_logits(model, tokenizer, prefix_input_ids)
    elif method == "teacher_forced_forward":
        logits = _teacher_forced_next_token_logits(model, prefix_input_ids)
        generated_token_id = None
    else:
        raise ValueError(f"unknown diagnostic method {method!r}")

    stats = _compact_logits_stats(logits)
    top_token_id: Optional[int] = None
    if stats["all_finite"]:
        top_token_id = int(torch.argmax(logits.float()).item())
    result = {
        "method": method,
        "prefix_length": prefix_len,
        "prefix_last_token_id": int(prefix_input_ids[0, -1].item()),
        "prefix_tail_token_ids": [int(x) for x in prefix_input_ids[0, max(0, prefix_len - 8):].tolist()],
        "logits": stats,
        "nan_stage": "raw_logits" if not stats["all_finite"] else "none",
        "model_generated_first_token_id": generated_token_id,
        "top_logit_token_id": top_token_id,
    }
    if include_token_audit:
        result["token_audit"] = _prefix_token_audit(tokenizer, [int(x) for x in prefix_input_ids[0].tolist()])
    return result


def _parse_prefix_lengths(value: Optional[str]) -> List[int]:
    if not value:
        return []
    lengths: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            prefix_len = int(item)
        except ValueError:
            die(f"Invalid --diagnose-prefix-lengths value {item!r}; expected comma-separated integers")
        if prefix_len <= 0:
            die(f"Invalid --diagnose-prefix-lengths value {prefix_len}; lengths must be > 0")
        if prefix_len not in lengths:
            lengths.append(prefix_len)
    return lengths


def _run_exact_prefix_diagnostics(
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    prefix_lengths: List[int],
) -> Dict[str, Any]:
    prompt_len = int(input_ids.shape[1])
    exact: Dict[str, Any] = {"prompt_length": prompt_len, "requested_prefix_lengths": prefix_lengths, "methods": []}
    valid_lengths = [length for length in prefix_lengths if 1 <= length <= prompt_len]
    skipped_lengths = [length for length in prefix_lengths if length not in valid_lengths]
    exact["skipped_prefix_lengths"] = skipped_lengths
    for method in ("generate_scores", "teacher_forced_forward"):
        probes = [
            _prefix_probe(method, model, tokenizer, input_ids, length, include_token_audit=True)
            for length in valid_lengths
        ]
        exact["methods"].append({"method": method, "probes": probes})
    return exact


def _sample_prefix_lengths(prompt_len: int) -> List[int]:
    values = {1, prompt_len}
    length = 2
    while length < prompt_len:
        values.add(length)
        length *= 2
    return sorted(v for v in values if 1 <= v <= prompt_len)


def _binary_search_nan_prefix(
    method: str,
    model: Any,
    tokenizer: Any,
    input_ids: Any,
) -> Dict[str, Any]:
    prompt_len = int(input_ids.shape[1])
    probes: List[Dict[str, Any]] = []

    full_probe = _prefix_probe(method, model, tokenizer, input_ids, prompt_len)
    probes.append(full_probe)
    if full_probe["logits"]["all_finite"]:
        return {
            "method": method,
            "prompt_length": prompt_len,
            "full_prompt_all_finite": True,
            "first_nan_prefix_length": None,
            "last_finite_prefix_length_before_nan": prompt_len,
            "binary_steps": probes,
            "binary_search_assumption": "full prompt is finite; no monotonic NaN transition to search",
        }

    lo = 1
    hi = prompt_len
    while lo < hi:
        mid = (lo + hi) // 2
        probe = _prefix_probe(method, model, tokenizer, input_ids, mid)
        probes.append(probe)
        if probe["logits"]["all_finite"]:
            lo = mid + 1
        else:
            hi = mid

    first_nan = lo
    boundary: List[Dict[str, Any]] = []
    if first_nan > 1:
        boundary.append(_prefix_probe(method, model, tokenizer, input_ids, first_nan - 1))
    boundary.append(_prefix_probe(method, model, tokenizer, input_ids, first_nan))
    return {
        "method": method,
        "prompt_length": prompt_len,
        "full_prompt_all_finite": False,
        "first_nan_prefix_length": first_nan,
        "last_finite_prefix_length_before_nan": first_nan - 1 if first_nan > 1 else None,
        "binary_steps": probes,
        "boundary_checks": boundary,
        "binary_search_assumption": "assumes non-finite state is monotonic with increasing prefix length; boundary checks included",
    }


def _run_prefix_search(
    model: Any,
    tokenizer: Any,
    input_ids: Any,
) -> Dict[str, Any]:
    prompt_len = int(input_ids.shape[1])
    sample_lengths = _sample_prefix_lengths(prompt_len)
    results: Dict[str, Any] = {
        "prompt_length": prompt_len,
        "sample_prefix_lengths": sample_lengths,
        "methods": [],
    }
    for method in ("generate_scores", "teacher_forced_forward"):
        method_result = _binary_search_nan_prefix(method, model, tokenizer, input_ids)
        sample_probes = []
        for prefix_len in sample_lengths:
            if prefix_len == prompt_len:
                continue
            sample_probes.append(_prefix_probe(method, model, tokenizer, input_ids, prefix_len))
        method_result["sample_prefix_probes"] = sample_probes
        all_observed = sample_probes + method_result.get("binary_steps", []) + method_result.get("boundary_checks", [])
        by_prefix: Dict[int, Dict[str, Any]] = {}
        for probe in all_observed:
            by_prefix[int(probe["prefix_length"])] = probe
        ordered_samples = [by_prefix[length] for length in sorted(by_prefix)]
        transition_scans = []
        for left, right in zip(ordered_samples, ordered_samples[1:]):
            left_finite = bool(left["logits"]["all_finite"])
            right_finite = bool(right["logits"]["all_finite"])
            if left_finite == right_finite:
                continue
            scanned = []
            boundary_prefix = int(right["prefix_length"])
            lo = int(left["prefix_length"]) + 1
            hi = int(right["prefix_length"])
            target_finite = right_finite
            while lo < hi:
                mid = (lo + hi) // 2
                probe = _prefix_probe(method, model, tokenizer, input_ids, mid)
                scanned.append(probe)
                if bool(probe["logits"]["all_finite"]) == target_finite:
                    hi = mid
                    boundary_prefix = mid
                else:
                    lo = mid + 1
            transition_scans.append(
                {
                    "from_prefix_length": int(left["prefix_length"]),
                    "from_all_finite": left_finite,
                    "to_prefix_length": int(right["prefix_length"]),
                    "to_all_finite": right_finite,
                    "boundary_prefix_length": boundary_prefix,
                    "boundary_type": "first_nan" if not right_finite else "first_finite_after_nan",
                    "first_nan_prefix_length_in_interval": boundary_prefix if not right_finite else None,
                    "first_finite_prefix_length_in_interval": boundary_prefix if right_finite else None,
                    "binary_steps": scanned,
                }
            )
            for probe in scanned:
                by_prefix[int(probe["prefix_length"])] = probe

        observed_nan_prefixes = [
            prefix_len
            for prefix_len, probe in by_prefix.items()
            if not bool(probe["logits"]["all_finite"])
        ]
        method_result["transition_scans"] = transition_scans
        method_result["observed_first_nan_prefix_length"] = min(observed_nan_prefixes) if observed_nan_prefixes else None
        method_result["observed_nan_prefix_lengths"] = sorted(observed_nan_prefixes)
        method_result["observed_non_monotonic"] = any(
            scan["boundary_type"] == "first_nan" for scan in transition_scans
        ) and any(
            scan["boundary_type"] == "first_finite_after_nan" for scan in transition_scans
        )
        results["methods"].append(method_result)
    return results


def _run_turn_diagnostics(
    reference: Dict[str, Any],
    selectors: List[str],
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    invocation: Dict[str, Any],
) -> int:
    import torch

    parsed_selectors = [_parse_turn_selector(selector) for selector in selectors]
    exact_prefix_lengths = _parse_prefix_lengths(args.diagnose_prefix_lengths)
    diagnostics: List[Dict[str, Any]] = []
    failures = 0
    for conv_idx, turn_idx in parsed_selectors:
        turn = _find_turn(reference, conv_idx, turn_idx)
        input_ids = torch.tensor([turn["input_token_ids"]], dtype=torch.long, device=model.device)
        generated_token_ids = turn["token_ids"]
        item: Dict[str, Any] = {
            "conversation_index": conv_idx,
            "turn_index": turn_idx,
            "prompt_length": int(input_ids.shape[1]),
            "generated_first_token_id": int(generated_token_ids[0]),
            "generated_token_count": int(len(generated_token_ids)),
            "attention_implementation": _model_attention_implementation(model, args.attn_implementation),
            "tokenizer": _tokenizer_metadata(tokenizer),
            "source_text_audit": _turn_source_text_audit(tokenizer, turn),
            "methods": [],
        }
        for method in ("generate_scores", "teacher_forced_forward"):
            try:
                if method == "generate_scores":
                    method_stats = _diagnose_generate_scores(
                        model,
                        tokenizer,
                        input_ids,
                        generated_token_ids,
                        diagnostic_top_k=args.diag_topk,
                    )
                else:
                    method_stats = _diagnose_teacher_forced(model, input_ids, generated_token_ids)
                item["methods"].append(method_stats)
                if method_stats["nan_stage"] != "none":
                    failures += 1
            except Exception as exc:
                failures += 1
                item["methods"].append({"method": method, "error": str(exc), "nan_stage": "exception"})
        if args.diagnose_prefix_search:
            try:
                item["prefix_search"] = _run_prefix_search(model, tokenizer, input_ids)
            except Exception as exc:
                failures += 1
                item["prefix_search"] = {"error": str(exc)}
        if exact_prefix_lengths:
            try:
                item["exact_prefix_diagnostics"] = _run_exact_prefix_diagnostics(
                    model,
                    tokenizer,
                    input_ids,
                    exact_prefix_lengths,
                )
            except Exception as exc:
                failures += 1
                item["exact_prefix_diagnostics"] = {"error": str(exc)}
        diagnostics.append(item)
        method_summary = ", ".join(
            f"{m.get('method')}:{m.get('nan_stage')}" for m in item["methods"]
        )
        info(
            f"Diagnosed conv={conv_idx} turn={turn_idx} "
            f"prompt_len={item['prompt_length']} first_token={item['generated_first_token_id']} "
            f"stages={method_summary}"
        )

    summary = {
        "diagnostic_turns": diagnostics,
        "failed_method_checks": failures,
        "strict_result": "fail" if failures else "pass",
    }
    run_dir = invocation.get("run_dir")
    if run_dir:
        diag_path = Path(run_dir) / "reference_backfill_topk_diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(
                {
                    "kind": "reference_backfill_topk_diagnostics",
                    "captured_at": datetime.now().isoformat(),
                    "summary": summary,
                },
                f,
                indent=2,
            )
        ok(f"Diagnostic manifest saved: {diag_path}")
    print(json.dumps(summary, indent=2))
    if failures:
        die(f"Diagnostic strict gate failed for {failures} method check(s)")
    return 0


def backfill_topk(args: argparse.Namespace) -> int:
    import torch

    if args.diag_steps <= 0:
        die("--diag-steps must be > 0")
    if args.diag_topk <= 0:
        die("--diag-topk must be > 0")

    reference_path = Path(args.reference).expanduser().resolve()
    if not reference_path.is_file():
        die(f"Reference artifact not found: {reference_path}")
    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(reference_path)
    if output_path == reference_path:
        die("--output must be a new artifact path; in-place backfill is not allowed")

    with open(reference_path) as f:
        source_reference = json.load(f)
    _require_reference_schema(source_reference, args.diag_steps)

    loaded = load_reference_model_and_tokenizer(args.model, attn_implementation=args.attn_implementation)
    model_name = loaded["model_name"]
    model_path = loaded["model_path"]
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]

    profile_id = source_reference["profile_id"]
    max_new_tokens = int(source_reference.get("max_new_tokens") or source_reference["contract"].get("generation", {}).get("max_new_tokens") or 0)
    if max_new_tokens <= 0:
        die("Reference artifact has no positive max_new_tokens value")

    current_contract = build_contract(
        model_name=model_name,
        model_path=model_path,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        add_generation_prompt=bool(source_reference["contract"].get("request", {}).get("add_generation_prompt", True)),
        enable_thinking=source_reference["contract"].get("request", {}).get("enable_thinking"),
        profile_id=profile_id,
        prompt_source_path=None,
        runtime_name="transformers",
        runtime_version=loaded["runtime_version"],
        torch_dtype="bfloat16",
    )
    _validate_contract_identity(source_reference, current_contract, args.model)

    output = copy.deepcopy(source_reference)
    output["format_version"] = 5
    output["backfilled_at"] = datetime.now().isoformat()
    output["backfill_source_reference_path"] = str(reference_path)
    output["runtime_version"] = loaded["runtime_version"]
    output["decode_diagnostics"] = {
        "schema_version": 1,
        "coverage": "teacher_forced_first_generated_steps",
        "captured_steps_per_turn": args.diag_steps,
        "top_k": args.diag_topk,
        "log_prob_base": "natural_log",
        "source": "reference_backfill_topk",
    }

    invocation = build_invocation_metadata(model_name, profile_id, max_new_tokens)
    invocation["source_reference_path"] = str(reference_path)
    invocation["diagnostic_steps"] = args.diag_steps
    invocation["diagnostic_top_k"] = args.diag_topk

    if args.diagnose_turn:
        return _run_turn_diagnostics(
            source_reference,
            args.diagnose_turn,
            model,
            tokenizer,
            args,
            invocation,
        )

    output["backfill_invocation"] = invocation

    failures: List[Dict[str, Any]] = []
    processed = 0
    for conv_idx, conv in enumerate(output.get("conversations", [])):
        for turn_idx, turn in enumerate(conv.get("turns", [])):
            input_ids = torch.tensor([turn["input_token_ids"]], dtype=torch.long, device=model.device)
            try:
                if args.diag_steps == 1:
                    per_token_data = build_prefill_next_token_data(
                        model,
                        tokenizer,
                        input_ids,
                        turn["token_ids"],
                        diagnostic_top_k=args.diag_topk,
                    )
                else:
                    per_token_data = build_teacher_forced_per_token_data(
                        model,
                        tokenizer,
                        input_ids,
                        turn["token_ids"],
                        diagnostic_steps=args.diag_steps,
                        diagnostic_top_k=args.diag_topk,
                    )
                _validate_per_token_data(turn, per_token_data, args.diag_steps, args.diag_topk)
            except Exception as exc:
                failures.append(
                    {
                        "conversation_index": conv_idx,
                        "turn_index": turn_idx,
                        "error": str(exc),
                    }
                )
                continue
            turn["per_token_data"] = per_token_data
            processed += 1
            first = per_token_data[0]
            info(
                f"Backfilled conv={conv_idx} turn={turn_idx} "
                f"token={first['token_id']} rank={first['rank']} log_prob={first['log_prob']:.6f}"
            )

    total_turns = len(_turns(output))
    summary = {
        "source_reference_path": str(reference_path),
        "output_path": str(output_path),
        "eligible_turns": total_turns,
        "processed_turns": processed,
        "failed_turns": len(failures),
        "diag_steps": args.diag_steps,
        "diag_topk": args.diag_topk,
        "failures": failures[:5],
    }
    output["backfill_summary"] = summary
    if failures:
        _write_failure_manifest(invocation, summary)
        die(f"Failed to backfill {len(failures)} turns; first failure: {failures[0]}")
    if processed != total_turns:
        die(f"Processed {processed}/{total_turns} turns")

    output["sanity"] = build_reference_sanity_report(
        output,
        expected_conversations=len(output.get("conversations", [])),
        expected_turns=total_turns,
    )
    if output["sanity"].get("status") != "pass":
        die("Backfilled artifact failed sanity gate: " + ", ".join(output["sanity"].get("failed_checks", [])))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    emit_reference_generation_trace(
        "reference_backfill_topk",
        output["contract"],
        str(output_path),
        max_new_tokens=max_new_tokens,
    )
    _write_run_manifest(invocation, output_path, summary)
    ok(f"Backfilled reference saved: {output_path}")
    ok(f"Backfilled turns: {processed}/{total_turns}, diag_steps={args.diag_steps}, top_k={args.diag_topk}")
    return 0


def main() -> int:
    if os.environ.get("KRASIS_ALLOW_ARCHIVED_HF_REFERENCE") != "1":
        print("ERROR: ./dev reference-backfill-topk is archived.")
        print("  HF/Transformers reference backfill is no longer trusted by default.")
        print("  Use llama-witness for new reference authority.")
        print("  For forensic reruns only, set KRASIS_ALLOW_ARCHIVED_HF_REFERENCE=1 and document the reason.")
        return 1
    return backfill_topk(parse_args())


if __name__ == "__main__":
    sys.exit(main())
