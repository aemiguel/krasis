#!/usr/bin/env python3
"""Offline diff tool for KRASIS_TRACE logs.

Usage:
    ./dev trace-diff <expected-trace> <actual-trace>

This script must be run via ./dev trace-diff, not directly.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
    print("ERROR: This script must be run via ./dev trace-diff, not directly.")
    print("  Usage: ./dev trace-diff <expected-trace> <actual-trace>")
    sys.exit(1)


TRACE_PREFIX = "[KRASIS-TRACE]"

DEFAULT_KEY_FIELDS = (
    "event",
    "scope",
    "context",
    "phase",
    "step",
    "pos",
    "tok",
    "layer",
    "component",
    "tensor",
    "label",
)

DEFAULT_IGNORE_FIELDS = (
    "path",
    "file",
)

FAMILY_IDENTITY_FIELDS = (
    "context",
    "phase",
    "step",
    "pos",
    "tok",
    "input_tok",
    "layer",
    "scope",
)

FAMILY_MEMBER_FIELDS = (
    "event",
    "component",
    "tensor",
    "label",
    "dtype",
)

PACKET_SIGNATURE_COUNT_FIELDS = (
    "event",
    "component",
    "tensor",
    "label",
    "dtype",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff two KRASIS_TRACE logs")
    parser.add_argument("expected", help="Expected/reference trace log path")
    parser.add_argument("actual", help="Actual trace log path")
    parser.add_argument(
        "--events",
        default="all",
        help="Comma-separated event names to include (default: all)",
    )
    parser.add_argument(
        "--components",
        default="all",
        help="Comma-separated components to include (default: all)",
    )
    parser.add_argument(
        "--contexts",
        default="all",
        help="Comma-separated contexts to include (default: all)",
    )
    parser.add_argument(
        "--phases",
        default="all",
        help="Comma-separated phases to include (default: all)",
    )
    parser.add_argument(
        "--ignore-fields",
        default=",".join(DEFAULT_IGNORE_FIELDS),
        help="Comma-separated fields to ignore in value comparison",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Limit comparison to the first N matching events from each log",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON report path",
    )
    parser.add_argument(
        "--show-window",
        type=int,
        default=2,
        help="Show this many events before and after the divergence (default: 2)",
    )
    parser.add_argument(
        "--resync-window",
        type=int,
        default=4,
        help=(
            "Look ahead up to this many events on each side for a bounded "
            "resync hint after the first divergence (default: 4)"
        ),
    )
    parser.add_argument(
        "--family-resync-window",
        type=int,
        default=3,
        help=(
            "Look ahead up to this many family packets on each side for a "
            "bounded family-aware resync hint around the first divergence "
            "(default: 3)"
        ),
    )
    return parser.parse_args()


def _csv_set(raw: str) -> Optional[set[str]]:
    raw = raw.strip()
    if not raw or raw.lower() == "all":
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def _parse_trace_line(line: str, lineno: int) -> Optional[Dict[str, Any]]:
    if TRACE_PREFIX not in line:
        return None
    text = line[line.index(TRACE_PREFIX) + len(TRACE_PREFIX):].strip()
    fields: Dict[str, Any] = {"_line": line.rstrip("\n"), "_lineno": lineno}
    idx = 0
    length = len(text)
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break

        key_start = idx
        while idx < length and (text[idx].isalnum() or text[idx] == "_"):
            idx += 1
        if idx >= length or idx == key_start or text[idx] != "=":
            while idx < length and not text[idx].isspace():
                idx += 1
            continue

        key = text[key_start:idx]
        idx += 1
        if idx >= length:
            fields[key] = ""
            break

        if text[idx] in "[{(":
            closer = { "[": "]", "{": "}", "(": ")" }[text[idx]]
            depth = 0
            value_start = idx
            while idx < length:
                ch = text[idx]
                if ch == text[value_start]:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        idx += 1
                        break
                idx += 1
            value = text[value_start:idx]
        else:
            value_start = idx
            while idx < length and not text[idx].isspace():
                idx += 1
            value = text[value_start:idx]
        fields[key] = value
    if "event" not in fields:
        return None
    return fields


def _parse_float(text: Any) -> Optional[float]:
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_bool(text: Any) -> Optional[bool]:
    if text is None:
        return None
    if isinstance(text, bool):
        return text
    raw = str(text).strip().lower()
    if raw == "true":
        return True
    if raw == "false":
        return False
    return None


def _strip_brackets(text: str) -> str:
    if len(text) >= 2 and text[0] in "[{(" and text[-1] in "]})":
        return text[1:-1]
    return text


def _parse_float_list(text: Any) -> Optional[List[float]]:
    if text is None:
        return None
    raw = _strip_brackets(str(text).strip())
    if not raw:
        return []
    values: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = _parse_float(part)
        if value is None:
            return None
        values.append(value)
    return values


def _parse_int_list(text: Any) -> Optional[List[int]]:
    if text is None:
        return None
    raw = _strip_brackets(str(text).strip())
    if not raw:
        return []
    values: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            return None
    return values


def _parse_ranked_pairs(text: Any) -> Optional[List[Tuple[int, float]]]:
    if text is None:
        return None
    raw = _strip_brackets(str(text).strip())
    if not raw:
        return []
    pairs: List[Tuple[int, float]] = []
    for item in raw.split(","):
        piece = item.strip()
        if not piece or ":" not in piece:
            return None
        token_text, score_text = piece.split(":", 1)
        try:
            token = int(token_text.strip())
        except ValueError:
            return None
        score = _parse_float(score_text.strip())
        if score is None:
            return None
        pairs.append((token, score))
    return pairs


def _float_delta_summary(expected: Optional[float], actual: Optional[float]) -> Optional[Dict[str, Any]]:
    if expected is None or actual is None:
        return None
    abs_delta = abs(actual - expected)
    rel_delta = None
    if expected != 0:
        rel_delta = abs_delta / abs(expected)
    return {
        "expected": expected,
        "actual": actual,
        "abs_delta": abs_delta,
        "rel_delta": rel_delta,
    }


def _vector_delta_summary(
    expected: Optional[Sequence[float]],
    actual: Optional[Sequence[float]],
) -> Optional[Dict[str, Any]]:
    if expected is None or actual is None:
        return None
    if len(expected) != len(actual):
        return {
            "expected_len": len(expected),
            "actual_len": len(actual),
            "reason": "length_mismatch",
        }
    if not expected:
        return {
            "expected_len": 0,
            "actual_len": 0,
            "max_abs_delta": 0.0,
            "mean_abs_delta": 0.0,
            "first_delta_index": None,
            "l2_delta": 0.0,
            "sign_flip_count": 0,
        }
    abs_deltas = [abs(a - e) for e, a in zip(expected, actual)]
    sq_deltas = [(a - e) * (a - e) for e, a in zip(expected, actual)]
    first_delta_index = None
    for idx, delta in enumerate(abs_deltas):
        if delta > 0:
            first_delta_index = idx
            break
    sign_flip_count = 0
    for exp, act in zip(expected, actual):
        if exp == 0.0 or act == 0.0:
            continue
        if math.copysign(1.0, exp) != math.copysign(1.0, act):
            sign_flip_count += 1
    summary: Dict[str, Any] = {
        "expected_len": len(expected),
        "actual_len": len(actual),
        "max_abs_delta": max(abs_deltas),
        "mean_abs_delta": sum(abs_deltas) / len(abs_deltas),
        "first_delta_index": first_delta_index,
        "l2_delta": math.sqrt(sum(sq_deltas)),
        "sign_flip_count": sign_flip_count,
    }
    if first_delta_index is not None:
        summary["first_expected"] = expected[first_delta_index]
        summary["first_actual"] = actual[first_delta_index]
    return summary


def _ranked_pair_summary(
    expected: Optional[Sequence[Tuple[int, float]]],
    actual: Optional[Sequence[Tuple[int, float]]],
) -> Optional[Dict[str, Any]]:
    if expected is None or actual is None:
        return None
    expected_ids = [token for token, _ in expected]
    actual_ids = [token for token, _ in actual]
    shared_ids = [token for token in expected_ids if token in set(actual_ids)]
    expected_scores = {token: score for token, score in expected}
    actual_scores = {token: score for token, score in actual}
    score_deltas = []
    for token in shared_ids:
        score_deltas.append(
            {
                "token": token,
                "expected": expected_scores[token],
                "actual": actual_scores[token],
                "abs_delta": abs(actual_scores[token] - expected_scores[token]),
            }
        )
    score_deltas.sort(key=lambda item: item["abs_delta"], reverse=True)
    first_rank_mismatch = None
    for idx, (exp_id, act_id) in enumerate(zip(expected_ids, actual_ids)):
        if exp_id != act_id:
            first_rank_mismatch = idx
            break
    return {
        "expected_ids": expected_ids,
        "actual_ids": actual_ids,
        "shared_id_count": len(shared_ids),
        "shared_ids": shared_ids,
        "missing_from_actual": [token for token in expected_ids if token not in set(actual_ids)],
        "unexpected_in_actual": [token for token in actual_ids if token not in set(expected_ids)],
        "first_rank_mismatch": first_rank_mismatch,
        "top1_changed": bool(expected_ids and actual_ids and expected_ids[0] != actual_ids[0]),
        "largest_score_deltas": score_deltas[:3],
    }


def _analyze_tensor_mismatch(
    expected_event: Dict[str, Any],
    actual_event: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "kind": "tensor",
        "tensor": expected_event.get("tensor"),
        "dtype_expected": expected_event.get("dtype"),
        "dtype_actual": actual_event.get("dtype"),
        "sample": _vector_delta_summary(
            _parse_float_list(expected_event.get("sample")),
            _parse_float_list(actual_event.get("sample")),
        ),
        "l2": _float_delta_summary(
            _parse_float(expected_event.get("l2")),
            _parse_float(actual_event.get("l2")),
        ),
        "mean": _float_delta_summary(
            _parse_float(expected_event.get("mean")),
            _parse_float(actual_event.get("mean")),
        ),
        "min": _float_delta_summary(
            _parse_float(expected_event.get("min")),
            _parse_float(actual_event.get("min")),
        ),
        "max": _float_delta_summary(
            _parse_float(expected_event.get("max")),
            _parse_float(actual_event.get("max")),
        ),
        "nan_count": _float_delta_summary(
            _parse_float(expected_event.get("nan")),
            _parse_float(actual_event.get("nan")),
        ),
        "inf_count": _float_delta_summary(
            _parse_float(expected_event.get("inf")),
            _parse_float(actual_event.get("inf")),
        ),
    }


def _analyze_values_mismatch(
    expected_event: Dict[str, Any],
    actual_event: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "kind": "values",
        "label": expected_event.get("label"),
        "sample": _vector_delta_summary(
            _parse_float_list(expected_event.get("sample")),
            _parse_float_list(actual_event.get("sample")),
        ),
    }


def _analyze_logits_mismatch(
    expected_event: Dict[str, Any],
    actual_event: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "kind": "logits",
        "topk": _ranked_pair_summary(
            _parse_ranked_pairs(expected_event.get("topk")),
            _parse_ranked_pairs(actual_event.get("topk")),
        ),
    }


def _analyze_python_compare_mismatch(
    expected_event: Dict[str, Any],
    actual_event: Dict[str, Any],
) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {
        "kind": "python_compare",
        "phase": expected_event.get("phase"),
    }
    top5 = _ranked_pair_summary(
        _parse_ranked_pairs(expected_event.get("top5")),
        _parse_ranked_pairs(actual_event.get("top5")),
    )
    if top5 is not None:
        analysis["top5"] = top5
    for field in ("logit_min", "logit_max", "spread"):
        delta = _float_delta_summary(
            _parse_float(expected_event.get(field)),
            _parse_float(actual_event.get(field)),
        )
        if delta is not None:
            analysis[field] = delta
    return analysis


def _analyze_route_mismatch(
    expected_event: Dict[str, Any],
    actual_event: Dict[str, Any],
) -> Dict[str, Any]:
    expected_ids = _parse_int_list(expected_event.get("ids"))
    actual_ids = _parse_int_list(actual_event.get("ids"))
    expected_weights = _parse_float_list(expected_event.get("weights"))
    actual_weights = _parse_float_list(actual_event.get("weights"))
    analysis: Dict[str, Any] = {
        "kind": "route",
        "ids": None,
        "weights": _vector_delta_summary(expected_weights, actual_weights),
        "sum": _float_delta_summary(
            _parse_float(expected_event.get("sum")),
            _parse_float(actual_event.get("sum")),
        ),
    }
    if expected_ids is not None and actual_ids is not None:
        first_id_mismatch = None
        for idx, (exp_id, act_id) in enumerate(zip(expected_ids, actual_ids)):
            if exp_id != act_id:
                first_id_mismatch = idx
                break
        analysis["ids"] = {
            "expected": expected_ids,
            "actual": actual_ids,
            "shared": [token for token in expected_ids if token in set(actual_ids)],
            "missing_from_actual": [token for token in expected_ids if token not in set(actual_ids)],
            "unexpected_in_actual": [token for token in actual_ids if token not in set(expected_ids)],
            "first_rank_mismatch": first_id_mismatch,
            "top1_changed": bool(expected_ids and actual_ids and expected_ids[0] != actual_ids[0]),
        }
    for field in ("norm_topk_prob",):
        expected_value = _parse_bool(expected_event.get(field))
        actual_value = _parse_bool(actual_event.get(field))
        if expected_value is not None or actual_value is not None:
            analysis[field] = {
                "expected": expected_value,
                "actual": actual_value,
                "changed": expected_value != actual_value,
            }
    for field in ("topk", "scoring_func", "num_experts"):
        expected_value = expected_event.get(field)
        actual_value = actual_event.get(field)
        if expected_value is not None or actual_value is not None:
            analysis[field] = {
                "expected": expected_value,
                "actual": actual_value,
                "changed": expected_value != actual_value,
            }
    return analysis


def _semantic_mismatch_analysis(
    comparison: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if comparison.get("reason") != "event_value_mismatch":
        return None
    expected_event = comparison.get("expected_event")
    actual_event = comparison.get("actual_event")
    if not isinstance(expected_event, dict) or not isinstance(actual_event, dict):
        return None
    event_kind = expected_event.get("event")
    if event_kind == "tensor":
        return _analyze_tensor_mismatch(expected_event, actual_event)
    if event_kind == "values":
        return _analyze_values_mismatch(expected_event, actual_event)
    if event_kind == "logits":
        return _analyze_logits_mismatch(expected_event, actual_event)
    if event_kind == "route":
        return _analyze_route_mismatch(expected_event, actual_event)
    if event_kind == "python_compare":
        return _analyze_python_compare_mismatch(expected_event, actual_event)
    return None


def _print_semantic_analysis(analysis: Dict[str, Any]) -> None:
    print("Typed mismatch analysis:")
    kind = analysis.get("kind", "unknown")
    print(f"  kind={kind}")
    if kind in {"tensor", "values"} and analysis.get("sample") is not None:
        sample = analysis["sample"]
        print(
            "  sample:"
            f" len={sample.get('expected_len')}->{sample.get('actual_len')}"
            f" max_abs_delta={sample.get('max_abs_delta')}"
            f" mean_abs_delta={sample.get('mean_abs_delta')}"
            f" first_delta_index={sample.get('first_delta_index')}"
            f" sign_flips={sample.get('sign_flip_count')}"
        )
    if kind == "tensor":
        for field in ("l2", "mean", "min", "max", "nan_count", "inf_count"):
            delta = analysis.get(field)
            if delta is None:
                continue
            print(
                f"  {field}: expected={delta.get('expected')} actual={delta.get('actual')} "
                f"abs_delta={delta.get('abs_delta')}"
            )
    if kind in {"logits", "python_compare"}:
        pairs = analysis.get("topk") or analysis.get("top5")
        if pairs is not None:
            print(
                "  ranked_ids:"
                f" top1_changed={pairs.get('top1_changed')}"
                f" shared={pairs.get('shared_id_count')}"
                f" first_rank_mismatch={pairs.get('first_rank_mismatch')}"
            )
            if pairs.get("missing_from_actual"):
                print(f"  missing_from_actual={pairs.get('missing_from_actual')}")
            if pairs.get("unexpected_in_actual"):
                print(f"  unexpected_in_actual={pairs.get('unexpected_in_actual')}")
            if pairs.get("largest_score_deltas"):
                print(f"  largest_score_deltas={pairs.get('largest_score_deltas')}")
    if kind == "python_compare":
        for field in ("logit_min", "logit_max", "spread"):
            delta = analysis.get(field)
            if delta is None:
                continue
            print(
                f"  {field}: expected={delta.get('expected')} actual={delta.get('actual')} "
                f"abs_delta={delta.get('abs_delta')}"
            )
    if kind == "route":
        ids = analysis.get("ids")
        if ids is not None:
            print(
                "  ids:"
                f" top1_changed={ids.get('top1_changed')}"
                f" first_rank_mismatch={ids.get('first_rank_mismatch')}"
                f" shared={len(ids.get('shared', []))}"
            )
            if ids.get("missing_from_actual"):
                print(f"  missing_from_actual={ids.get('missing_from_actual')}")
            if ids.get("unexpected_in_actual"):
                print(f"  unexpected_in_actual={ids.get('unexpected_in_actual')}")
        weights = analysis.get("weights")
        if weights is not None:
            print(
                "  weights:"
                f" max_abs_delta={weights.get('max_abs_delta')}"
                f" mean_abs_delta={weights.get('mean_abs_delta')}"
                f" first_delta_index={weights.get('first_delta_index')}"
            )
        sum_delta = analysis.get("sum")
        if sum_delta is not None:
            print(
                f"  sum: expected={sum_delta.get('expected')} actual={sum_delta.get('actual')} "
                f"abs_delta={sum_delta.get('abs_delta')}"
            )
        for field in ("norm_topk_prob", "topk", "scoring_func", "num_experts"):
            delta = analysis.get(field)
            if delta is not None and delta.get("changed"):
                print(f"  {field}: expected={delta.get('expected')} actual={delta.get('actual')}")


def _load_trace_events(
    path: Path,
    *,
    include_events: Optional[set[str]],
    include_components: Optional[set[str]],
    include_contexts: Optional[set[str]],
    include_phases: Optional[set[str]],
    max_events: int,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            parsed = _parse_trace_line(line, lineno)
            if parsed is None:
                continue
            if include_events is not None and parsed.get("event") not in include_events:
                continue
            component = parsed.get("component")
            if include_components is not None and component not in include_components:
                continue
            context = parsed.get("context")
            if include_contexts is not None and context not in include_contexts:
                continue
            phase = parsed.get("phase")
            if include_phases is not None and phase not in include_phases:
                continue
            events.append(parsed)
            if max_events > 0 and len(events) >= max_events:
                break
    return events


def _event_key(event: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    pairs: List[Tuple[str, str]] = []
    for field in DEFAULT_KEY_FIELDS:
        value = event.get(field)
        if value is not None:
            pairs.append((field, str(value)))
    return tuple(pairs)


def _event_values(
    event: Dict[str, Any],
    ignore_fields: set[str],
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for key, value in event.items():
        if key.startswith("_"):
            continue
        if key in ignore_fields:
            continue
        if key in DEFAULT_KEY_FIELDS:
            continue
        values[key] = value
    return values


def _format_pairs(pairs: Sequence[Tuple[str, Any]]) -> str:
    return ", ".join(f"{key}={value}" for key, value in pairs)


def _event_brief(event: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if event is None:
        return None
    brief = {
        "lineno": event.get("_lineno"),
        "event": event.get("event"),
        "component": event.get("component"),
        "context": event.get("context"),
        "phase": event.get("phase"),
        "step": event.get("step"),
        "layer": event.get("layer"),
        "tensor": event.get("tensor"),
    }
    for key in ("status", "state", "reason", "profile", "model", "file"):
        if key in event:
            brief[key] = event.get(key)
    return {key: value for key, value in brief.items() if value is not None}


def _contract_event_brief(event: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if event is None:
        return None
    brief = {
        "lineno": event.get("_lineno"),
        "event": event.get("event"),
        "context": event.get("context"),
        "phase": event.get("phase"),
        "reason": event.get("reason"),
        "status": event.get("status"),
        "state": event.get("state"),
        "profile": event.get("profile"),
        "model": event.get("model"),
        "file": event.get("file"),
        "errors": event.get("errors"),
        "warnings": event.get("warnings"),
    }
    return {key: value for key, value in brief.items() if value is not None}


def _family_identity(event: Optional[Dict[str, Any]]) -> Optional[Tuple[Tuple[str, str], ...]]:
    if event is None:
        return None
    pairs: List[Tuple[str, str]] = []
    for field in FAMILY_IDENTITY_FIELDS:
        value = event.get(field)
        if value is not None:
            pairs.append((field, str(value)))
    return tuple(pairs) if pairs else None


def _family_member_key(event: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    pairs: List[Tuple[str, str]] = []
    for field in FAMILY_MEMBER_FIELDS:
        value = event.get(field)
        if value is not None:
            pairs.append((field, str(value)))
    return tuple(pairs)


def _family_member_brief(event: Dict[str, Any]) -> Dict[str, Any]:
    brief = {
        "lineno": event.get("_lineno"),
        "event": event.get("event"),
        "component": event.get("component"),
        "tensor": event.get("tensor"),
        "label": event.get("label"),
        "dtype": event.get("dtype"),
    }
    return {key: value for key, value in brief.items() if value is not None}


def _counter_entries(
    counts: collections.Counter[str],
) -> List[Dict[str, Any]]:
    return [
        {"value": key, "count": count}
        for key, count in sorted(counts.items())
    ]


def _packet_signature(
    packet: Dict[str, Any],
) -> Dict[str, Any]:
    events = packet.get("events", [])
    member_counts = _family_counts(events)
    field_counts = {field: collections.Counter() for field in PACKET_SIGNATURE_COUNT_FIELDS}
    ordered_members: List[List[Tuple[str, str]]] = []

    for event in events:
        member_key = _family_member_key(event)
        ordered_members.append(list(member_key))
        for field in PACKET_SIGNATURE_COUNT_FIELDS:
            value = event.get(field)
            if value is not None:
                field_counts[field][str(value)] += 1

    return {
        "identity": packet.get("identity"),
        "event_count": len(events),
        "unique_member_count": len(member_counts),
        "member_counts": [
            {"member": list(member_key), "count": count}
            for member_key, count in sorted(member_counts.items())
        ],
        "ordered_members": ordered_members,
        "event_counts": _counter_entries(field_counts["event"]),
        "component_counts": _counter_entries(field_counts["component"]),
        "tensor_counts": _counter_entries(field_counts["tensor"]),
        "label_counts": _counter_entries(field_counts["label"]),
        "dtype_counts": _counter_entries(field_counts["dtype"]),
    }


def _packet_signature_key(packet: Optional[Dict[str, Any]]) -> Optional[str]:
    if packet is None:
        return None
    signature = packet.get("signature")
    if signature is None:
        return None
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _family_slice(
    events: Sequence[Dict[str, Any]],
    *,
    index: int,
) -> Dict[str, Any]:
    if not events:
        return {
            "identity": None,
            "start_index": None,
            "end_index": None,
            "members": [],
        }
    clamped_index = min(max(index, 0), len(events) - 1)
    identity = _family_identity(events[clamped_index])
    if identity is None:
        return {
            "identity": None,
            "start_index": clamped_index,
            "end_index": clamped_index + 1,
            "members": [_family_member_brief(events[clamped_index])],
        }

    start = clamped_index
    while start > 0 and _family_identity(events[start - 1]) == identity:
        start -= 1
    end = clamped_index + 1
    while end < len(events) and _family_identity(events[end]) == identity:
        end += 1

    return {
        "identity": list(identity),
        "start_index": start,
        "end_index": end,
        "members": [_family_member_brief(event) for event in events[start:end]],
    }


def _family_counts(
    events: Sequence[Dict[str, Any]],
) -> collections.Counter[Tuple[Tuple[str, str], ...]]:
    counts: collections.Counter[Tuple[Tuple[str, str], ...]] = collections.Counter()
    for event in events:
        counts[_family_member_key(event)] += 1
    return counts


def _family_analysis(
    expected: Sequence[Dict[str, Any]],
    actual: Sequence[Dict[str, Any]],
    *,
    index: int,
    ignore_fields: set[str],
) -> Optional[Dict[str, Any]]:
    if not expected and not actual:
        return None

    expected_slice = _family_slice(expected, index=index) if expected else {
        "identity": None,
        "start_index": None,
        "end_index": None,
        "members": [],
    }
    actual_slice = _family_slice(actual, index=index) if actual else {
        "identity": None,
        "start_index": None,
        "end_index": None,
        "members": [],
    }

    expected_start = expected_slice["start_index"]
    expected_end = expected_slice["end_index"]
    actual_start = actual_slice["start_index"]
    actual_end = actual_slice["end_index"]
    expected_events = (
        list(expected[expected_start:expected_end])
        if expected_start is not None and expected_end is not None
        else []
    )
    actual_events = (
        list(actual[actual_start:actual_end])
        if actual_start is not None and actual_end is not None
        else []
    )

    if not expected_events and not actual_events:
        return None

    expected_counts = _family_counts(expected_events)
    actual_counts = _family_counts(actual_events)
    shared_keys = set(expected_counts) & set(actual_counts)
    missing_members = []
    unexpected_members = []
    multiplicity_mismatches = []

    for member_key, count in sorted(expected_counts.items()):
        actual_count = actual_counts.get(member_key, 0)
        if actual_count == 0:
            missing_members.append(
                {
                    "member": list(member_key),
                    "count": count,
                }
            )
        elif actual_count != count:
            multiplicity_mismatches.append(
                {
                    "member": list(member_key),
                    "expected_count": count,
                    "actual_count": actual_count,
                }
            )
    for member_key, count in sorted(actual_counts.items()):
        if expected_counts.get(member_key, 0) == 0:
            unexpected_members.append(
                {
                    "member": list(member_key),
                    "count": count,
                }
            )

    first_shared_value_mismatch = None
    for member_key in sorted(shared_keys):
        exp_candidates = [event for event in expected_events if _family_member_key(event) == member_key]
        act_candidates = [event for event in actual_events if _family_member_key(event) == member_key]
        shared_count = min(len(exp_candidates), len(act_candidates))
        for idx_in_member in range(shared_count):
            exp_event = exp_candidates[idx_in_member]
            act_event = act_candidates[idx_in_member]
            if _event_values(exp_event, ignore_fields) != _event_values(act_event, ignore_fields):
                mismatches = []
                differing_fields = sorted(
                    set(_event_values(exp_event, ignore_fields))
                    | set(_event_values(act_event, ignore_fields))
                )
                exp_values = _event_values(exp_event, ignore_fields)
                act_values = _event_values(act_event, ignore_fields)
                for field in differing_fields:
                    if exp_values.get(field) != act_values.get(field):
                        mismatches.append(
                            {
                                "field": field,
                                "expected": exp_values.get(field),
                                "actual": act_values.get(field),
                            }
                        )
                first_shared_value_mismatch = {
                    "member": list(member_key),
                    "expected": _family_member_brief(exp_event),
                    "actual": _family_member_brief(act_event),
                    "mismatches": mismatches,
                    "semantic_analysis": _semantic_mismatch_analysis(
                        {
                            "reason": "event_value_mismatch",
                            "expected_event": exp_event,
                            "actual_event": act_event,
                        }
                    ),
                }
                break
        if first_shared_value_mismatch is not None:
            break

    return {
        "expected_identity": expected_slice["identity"],
        "actual_identity": actual_slice["identity"],
        "expected_member_count": len(expected_events),
        "actual_member_count": len(actual_events),
        "expected_range": {
            "start_index": expected_start,
            "end_index": expected_end,
        },
        "actual_range": {
            "start_index": actual_start,
            "end_index": actual_end,
        },
        "expected_members": expected_slice["members"],
        "actual_members": actual_slice["members"],
        "missing_members": missing_members,
        "unexpected_members": unexpected_members,
        "multiplicity_mismatches": multiplicity_mismatches,
        "first_shared_value_mismatch": first_shared_value_mismatch,
    }


def _family_packets(
    events: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    packets: List[Dict[str, Any]] = []
    index = 0
    while index < len(events):
        family = _family_slice(events, index=index)
        start = family.get("start_index")
        end = family.get("end_index")
        if start is None or end is None or end <= start:
            start = index
            end = index + 1
            family = {
                "identity": _family_identity(events[index]),
                "start_index": start,
                "end_index": end,
                "members": [_family_member_brief(events[index])],
            }
        packet_events = list(events[start:end])
        packets.append(
            {
                "identity": family.get("identity"),
                "start_index": start,
                "end_index": end,
                "event_count": len(packet_events),
                "members": family.get("members", []),
                "events": packet_events,
            }
        )
        packets[-1]["signature"] = _packet_signature(packets[-1])
        packets[-1]["signature_key"] = _packet_signature_key(packets[-1])
        index = end
    return packets


def _packet_brief(packet: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if packet is None:
        return None
    signature = packet.get("signature") or {}
    return {
        "identity": packet.get("identity"),
        "start_index": packet.get("start_index"),
        "end_index": packet.get("end_index"),
        "event_count": packet.get("event_count"),
        "signature_key": packet.get("signature_key"),
        "signature": {
            "unique_member_count": signature.get("unique_member_count"),
            "member_counts": signature.get("member_counts"),
            "event_counts": signature.get("event_counts"),
            "component_counts": signature.get("component_counts"),
        },
        "members": packet.get("members"),
    }


def _packet_identity_equal(expected_packet: Dict[str, Any], actual_packet: Dict[str, Any]) -> bool:
    return (expected_packet.get("identity") or []) == (actual_packet.get("identity") or [])


def _packet_structure_equal(expected_packet: Dict[str, Any], actual_packet: Dict[str, Any]) -> bool:
    return expected_packet.get("signature_key") == actual_packet.get("signature_key")


def _packet_equal(
    expected_packet: Dict[str, Any],
    actual_packet: Dict[str, Any],
    *,
    ignore_fields: set[str],
) -> bool:
    if not _packet_structure_equal(expected_packet, actual_packet):
        return False
    expected_events = expected_packet.get("events", [])
    actual_events = actual_packet.get("events", [])
    if len(expected_events) != len(actual_events):
        return False
    for expected_event, actual_event in zip(expected_events, actual_events):
        if not _events_equal(expected_event, actual_event, ignore_fields=ignore_fields):
            return False
    return True


def _packet_index_for_event_index(
    packets: Sequence[Dict[str, Any]],
    event_index: int,
) -> Optional[int]:
    if not packets:
        return None
    if event_index < 0:
        return 0
    for idx, packet in enumerate(packets):
        start = packet.get("start_index")
        end = packet.get("end_index")
        if start is None or end is None:
            continue
        if start <= event_index < end:
            return idx
    if event_index >= packets[-1].get("end_index", 0):
        return len(packets) - 1
    return None


def _packet_window(
    packets: Sequence[Dict[str, Any]],
    packet_index: Optional[int],
    *,
    radius: int,
) -> List[Dict[str, Any]]:
    if packet_index is None or not packets:
        return []
    start = max(0, packet_index - radius)
    end = min(len(packets), packet_index + radius + 1)
    window: List[Dict[str, Any]] = []
    for idx in range(start, end):
        entry = _packet_brief(packets[idx]) or {}
        entry["packet_index"] = idx
        entry["relative"] = idx - packet_index
        window.append(entry)
    return window


def _family_region_resync_hint(
    expected_packets: Sequence[Dict[str, Any]],
    actual_packets: Sequence[Dict[str, Any]],
    *,
    expected_packet_index: Optional[int],
    actual_packet_index: Optional[int],
    window: int,
    ignore_fields: set[str],
) -> Optional[Dict[str, Any]]:
    if (
        window <= 0
        or expected_packet_index is None
        or actual_packet_index is None
        or not expected_packets
        or not actual_packets
    ):
        return None

    current_expected_packet = expected_packets[expected_packet_index]
    current_actual_packet = actual_packets[actual_packet_index]
    if _packet_structure_equal(current_expected_packet, current_actual_packet):
        return None

    best: Optional[Dict[str, Any]] = None
    max_expected_skip = min(window, max(0, len(expected_packets) - expected_packet_index - 1))
    max_actual_skip = min(window, max(0, len(actual_packets) - actual_packet_index - 1))

    for expected_skip in range(0, max_expected_skip + 1):
        for actual_skip in range(0, max_actual_skip + 1):
            if expected_skip == 0 and actual_skip == 0:
                continue
            exp_resume = expected_packet_index + expected_skip
            act_resume = actual_packet_index + actual_skip
            if exp_resume >= len(expected_packets) or act_resume >= len(actual_packets):
                continue
            if not _packet_structure_equal(
                expected_packets[exp_resume],
                actual_packets[act_resume],
            ):
                continue

            resumed_packet_signature_match_count = 0
            resumed_packet_exact_match_count = 0
            probe_expected = exp_resume
            probe_actual = act_resume
            while probe_expected < len(expected_packets) and probe_actual < len(actual_packets):
                if not _packet_structure_equal(
                    expected_packets[probe_expected],
                    actual_packets[probe_actual],
                ):
                    break
                resumed_packet_signature_match_count += 1
                if _packet_equal(
                    expected_packets[probe_expected],
                    actual_packets[probe_actual],
                    ignore_fields=ignore_fields,
                ):
                    resumed_packet_exact_match_count += 1
                probe_expected += 1
                probe_actual += 1

            if expected_skip > 0 and actual_skip == 0:
                resync_type = "missing_expected_families"
            elif actual_skip > 0 and expected_skip == 0:
                resync_type = "unexpected_actual_families"
            else:
                resync_type = "shifted_family_region"

            candidate = {
                "type": resync_type,
                "expected_skip": expected_skip,
                "actual_skip": actual_skip,
                "expected_resume_packet_index": exp_resume,
                "actual_resume_packet_index": act_resume,
                "resumed_packet_signature_match_count": resumed_packet_signature_match_count,
                "resumed_packet_exact_match_count": resumed_packet_exact_match_count,
                "expected_resume_packet": _packet_brief(expected_packets[exp_resume]),
                "actual_resume_packet": _packet_brief(actual_packets[act_resume]),
                "expected_skipped_packets": [
                    _packet_brief(packet)
                    for packet in expected_packets[expected_packet_index:exp_resume]
                ],
                "actual_skipped_packets": [
                    _packet_brief(packet)
                    for packet in actual_packets[actual_packet_index:act_resume]
                ],
            }
            score = (
                expected_skip + actual_skip,
                max(expected_skip, actual_skip),
                -resumed_packet_signature_match_count,
                -resumed_packet_exact_match_count,
            )
            if best is None or score < best["_score"]:
                candidate["_score"] = score
                best = candidate

    if best is None:
        return None
    best.pop("_score", None)
    return best


def _family_region_analysis(
    expected: Sequence[Dict[str, Any]],
    actual: Sequence[Dict[str, Any]],
    *,
    event_index: int,
    ignore_fields: set[str],
    packet_window: int,
) -> Optional[Dict[str, Any]]:
    expected_packets = _family_packets(expected)
    actual_packets = _family_packets(actual)
    if not expected_packets and not actual_packets:
        return None

    expected_packet_index = _packet_index_for_event_index(expected_packets, event_index)
    actual_packet_index = _packet_index_for_event_index(actual_packets, event_index)
    expected_focus_packet = (
        expected_packets[expected_packet_index]
        if expected_packet_index is not None and expected_packet_index < len(expected_packets)
        else None
    )
    actual_focus_packet = (
        actual_packets[actual_packet_index]
        if actual_packet_index is not None and actual_packet_index < len(actual_packets)
        else None
    )

    return {
        "expected_packet_index": expected_packet_index,
        "actual_packet_index": actual_packet_index,
        "expected_packet_count": len(expected_packets),
        "actual_packet_count": len(actual_packets),
        "expected_focus_packet": _packet_brief(expected_focus_packet),
        "actual_focus_packet": _packet_brief(actual_focus_packet),
        "focus_packet_structure_match": (
            _packet_structure_equal(expected_focus_packet, actual_focus_packet)
            if expected_focus_packet is not None and actual_focus_packet is not None
            else None
        ),
        "focus_packet_exact_match": (
            _packet_equal(
                expected_focus_packet,
                actual_focus_packet,
                ignore_fields=ignore_fields,
            )
            if expected_focus_packet is not None and actual_focus_packet is not None
            else None
        ),
        "expected_packet_window": _packet_window(
            expected_packets,
            expected_packet_index,
            radius=max(packet_window, 0),
        ),
        "actual_packet_window": _packet_window(
            actual_packets,
            actual_packet_index,
            radius=max(packet_window, 0),
        ),
        "resync_hint": _family_region_resync_hint(
            expected_packets,
            actual_packets,
            expected_packet_index=expected_packet_index,
            actual_packet_index=actual_packet_index,
            window=max(packet_window, 0),
            ignore_fields=ignore_fields,
        ),
    }


def _nearest_contract_state(
    events: Sequence[Dict[str, Any]],
    *,
    before_lineno: Optional[int],
) -> Optional[Dict[str, Any]]:
    if before_lineno is None:
        return None
    for event in reversed(events):
        lineno = event.get("_lineno")
        if lineno is None or lineno >= before_lineno:
            continue
        if event.get("component") != "config_contract":
            continue
        return _contract_event_brief(event)
    return None


def _summarize_events(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    event_counts = collections.Counter()
    component_counts = collections.Counter()
    context_counts = collections.Counter()
    phase_counts = collections.Counter()
    context_phase_counts = collections.Counter()

    for event in events:
        event_counts[str(event.get("event", "unknown"))] += 1
        component_counts[str(event.get("component", "unknown"))] += 1
        context = str(event.get("context", "none"))
        phase = str(event.get("phase", "none"))
        context_counts[context] += 1
        phase_counts[phase] += 1
        context_phase_counts[f"{context}:{phase}"] += 1

    return {
        "events": dict(sorted(event_counts.items())),
        "components": dict(sorted(component_counts.items())),
        "contexts": dict(sorted(context_counts.items())),
        "phases": dict(sorted(phase_counts.items())),
        "context_phases": dict(sorted(context_phase_counts.items())),
    }


def _matching_prefix_length(
    expected: Sequence[Dict[str, Any]],
    actual: Sequence[Dict[str, Any]],
    *,
    ignore_fields: set[str],
) -> int:
    compared = min(len(expected), len(actual))
    for idx in range(compared):
        exp_event = expected[idx]
        act_event = actual[idx]
        if _event_key(exp_event) != _event_key(act_event):
            return idx
        if _event_values(exp_event, ignore_fields) != _event_values(act_event, ignore_fields):
            return idx
    return compared


def _divergence_window(
    events: Sequence[Dict[str, Any]],
    index: int,
    *,
    radius: int,
) -> List[Dict[str, Any]]:
    if not events:
        return []
    start = max(0, index - radius)
    end = min(len(events), index + radius + 1)
    window = []
    for idx in range(start, end):
        entry = _event_brief(events[idx]) or {}
        entry["index"] = idx
        entry["relative"] = idx - index
        window.append(entry)
    return window


def _diff_events(
    expected: Sequence[Dict[str, Any]],
    actual: Sequence[Dict[str, Any]],
    *,
    ignore_fields: set[str],
) -> Dict[str, Any]:
    compared = min(len(expected), len(actual))
    for idx in range(compared):
        exp_event = expected[idx]
        act_event = actual[idx]
        exp_key = _event_key(exp_event)
        act_key = _event_key(act_event)
        if exp_key != act_key:
            return {
                "status": "different",
                "reason": "event_key_mismatch",
                "index": idx,
                "expected_event": exp_event,
                "actual_event": act_event,
                "expected_key": list(exp_key),
                "actual_key": list(act_key),
            }

        exp_values = _event_values(exp_event, ignore_fields)
        act_values = _event_values(act_event, ignore_fields)
        if exp_values != act_values:
            differing_fields = sorted(set(exp_values) | set(act_values))
            mismatches = []
            for field in differing_fields:
                if exp_values.get(field) != act_values.get(field):
                    mismatches.append(
                        {
                            "field": field,
                            "expected": exp_values.get(field),
                            "actual": act_values.get(field),
                        }
                    )
            return {
                "status": "different",
                "reason": "event_value_mismatch",
                "index": idx,
                "expected_event": exp_event,
                "actual_event": act_event,
                "mismatches": mismatches,
            }

    if len(expected) != len(actual):
        return {
            "status": "different",
            "reason": "event_count_mismatch",
            "index": compared,
            "expected_count": len(expected),
            "actual_count": len(actual),
            "expected_event": expected[compared] if compared < len(expected) else None,
            "actual_event": actual[compared] if compared < len(actual) else None,
        }

    return {
        "status": "match",
        "reason": "all_events_match",
        "index": compared,
        "expected_count": len(expected),
        "actual_count": len(actual),
    }


def _events_equal(
    expected_event: Dict[str, Any],
    actual_event: Dict[str, Any],
    *,
    ignore_fields: set[str],
) -> bool:
    return (
        _event_key(expected_event) == _event_key(actual_event)
        and _event_values(expected_event, ignore_fields)
        == _event_values(actual_event, ignore_fields)
    )


def _resync_hint(
    expected: Sequence[Dict[str, Any]],
    actual: Sequence[Dict[str, Any]],
    *,
    start_index: int,
    window: int,
    ignore_fields: set[str],
) -> Optional[Dict[str, Any]]:
    if window <= 0:
        return None

    best: Optional[Dict[str, Any]] = None
    max_expected_skip = min(window, max(0, len(expected) - start_index - 1))
    max_actual_skip = min(window, max(0, len(actual) - start_index - 1))

    for expected_skip in range(0, max_expected_skip + 1):
        for actual_skip in range(0, max_actual_skip + 1):
            if expected_skip == 0 and actual_skip == 0:
                continue
            expected_index = start_index + expected_skip
            actual_index = start_index + actual_skip
            if expected_index >= len(expected) or actual_index >= len(actual):
                continue
            if not _events_equal(
                expected[expected_index],
                actual[actual_index],
                ignore_fields=ignore_fields,
            ):
                continue

            resumed_match_count = 0
            probe_expected = expected_index
            probe_actual = actual_index
            while probe_expected < len(expected) and probe_actual < len(actual):
                if not _events_equal(
                    expected[probe_expected],
                    actual[probe_actual],
                    ignore_fields=ignore_fields,
                ):
                    break
                resumed_match_count += 1
                probe_expected += 1
                probe_actual += 1

            if expected_skip > 0 and actual_skip == 0:
                resync_type = "missing_expected_events"
            elif actual_skip > 0 and expected_skip == 0:
                resync_type = "unexpected_actual_events"
            else:
                resync_type = "shifted_both_streams"

            candidate = {
                "type": resync_type,
                "divergence_index": start_index,
                "expected_skip": expected_skip,
                "actual_skip": actual_skip,
                "expected_resume_index": expected_index,
                "actual_resume_index": actual_index,
                "resumed_match_count": resumed_match_count,
                "expected_resume_event": _event_brief(expected[expected_index]),
                "actual_resume_event": _event_brief(actual[actual_index]),
                "expected_skipped": [
                    _event_brief(event)
                    for event in expected[start_index:expected_index]
                ],
                "actual_skipped": [
                    _event_brief(event)
                    for event in actual[start_index:actual_index]
                ],
            }
            score = (
                expected_skip + actual_skip,
                max(expected_skip, actual_skip),
                -resumed_match_count,
            )
            if best is None or score < best["_score"]:
                candidate["_score"] = score
                best = candidate

    if best is None:
        return None
    best.pop("_score", None)
    return best


def _classify_divergence(
    comparison: Dict[str, Any],
    *,
    semantic_analysis: Optional[Dict[str, Any]],
    family_analysis: Optional[Dict[str, Any]],
    family_region_analysis: Optional[Dict[str, Any]],
    resync_hint: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if comparison.get("status") == "match":
        return {
            "classification": "trace_match",
            "summary": "All compared trace events matched",
            "scope": "trace",
            "confidence": "high",
        }

    expected_event = comparison.get("expected_event") or {}
    actual_event = comparison.get("actual_event") or {}
    component = expected_event.get("component") or actual_event.get("component")
    event_name = expected_event.get("event") or actual_event.get("event")
    reason = comparison.get("reason")

    if component == "config_contract":
        return {
            "classification": "config_contract_divergence",
            "summary": "Reference/runtime contract state diverged before or during validation",
            "scope": "contract",
            "confidence": "high",
        }

    same_family_identity = (
        family_analysis is not None
        and (family_analysis.get("expected_identity") or []) == (family_analysis.get("actual_identity") or [])
    )

    if reason == "event_value_mismatch":
        if same_family_identity and family_analysis and family_analysis.get("first_shared_value_mismatch") is not None:
            kind = semantic_analysis.get("kind") if semantic_analysis is not None else None
            classification = f"same_family_{kind}_value_mismatch" if kind else "same_family_value_mismatch"
            summary = (
                f"Same-family {kind} value drift inside one decode packet"
                if kind
                else "Same-family value drift inside one decode packet"
            )
            return {
                "classification": classification,
                "summary": summary,
                "scope": "decode_family",
                "confidence": "high",
            }
        kind = semantic_analysis.get("kind") if semantic_analysis is not None else event_name
        return {
            "classification": f"same_position_{kind}_value_mismatch" if kind else "same_position_value_mismatch",
            "summary": (
                f"Same-position {kind} value mismatch"
                if kind
                else "Same-position value mismatch"
            ),
            "scope": "event",
            "confidence": "high",
        }

    if same_family_identity and family_analysis:
        if family_analysis.get("missing_members"):
            return {
                "classification": "decode_family_missing_member",
                "summary": "Decode family is missing one or more expected sub-events",
                "scope": "decode_family",
                "confidence": "high",
            }
        if family_analysis.get("unexpected_members"):
            return {
                "classification": "decode_family_unexpected_member",
                "summary": "Decode family contains unexpected extra sub-events",
                "scope": "decode_family",
                "confidence": "high",
            }
        if family_analysis.get("multiplicity_mismatches"):
            return {
                "classification": "decode_family_multiplicity_mismatch",
                "summary": "Decode family contains the right member type but wrong multiplicity",
                "scope": "decode_family",
                "confidence": "high",
            }

    family_region_resync = None
    if family_region_analysis is not None:
        family_region_resync = family_region_analysis.get("resync_hint")
    if family_region_resync is not None:
        label_map = {
            "missing_expected_families": "missing expected family packets",
            "unexpected_actual_families": "unexpected actual family packets",
            "shifted_family_region": "shifted family region with later realignment",
        }
        return {
            "classification": family_region_resync.get("type", "family_region_shift"),
            "summary": (
                f"Family-region structural shift: {label_map.get(family_region_resync.get('type'), family_region_resync.get('type'))}"
            ),
            "scope": "decode_region",
            "confidence": "medium" if family_region_resync.get("resumed_packet_exact_match_count", 0) == 0 else "high",
        }

    if resync_hint is not None:
        label_map = {
            "missing_expected_events": "missing expected events",
            "unexpected_actual_events": "unexpected actual events",
            "shifted_both_streams": "shifted event streams with later realignment",
        }
        return {
            "classification": resync_hint.get("type", "event_stream_shift"),
            "summary": f"Event-stream shift: {label_map.get(resync_hint.get('type'), resync_hint.get('type'))}",
            "scope": "event_stream",
            "confidence": "medium",
        }

    return {
        "classification": reason or "trace_divergence",
        "summary": f"Unclassified divergence at event={event_name} component={component}",
        "scope": "generic",
        "confidence": "low",
    }


def _print_summary(report: Dict[str, Any]) -> None:
    print("Trace diff")
    print(f"Expected: {report['expected_path']}")
    print(f"Actual: {report['actual_path']}")
    print(f"Expected events: {report['expected_event_count']}")
    print(f"Actual events: {report['actual_event_count']}")
    print(f"Matching prefix: {report['matching_prefix_count']}")
    print(f"Status: {report['comparison']['status']}")
    print(f"Reason: {report['comparison']['reason']}")
    classification = report.get("classification")
    if classification is not None:
        print(
            "Classification: "
            f"{classification.get('classification')} "
            f"scope={classification.get('scope')} "
            f"confidence={classification.get('confidence')}"
        )
        print(f"Classification summary: {classification.get('summary')}")

    if report.get("matching_prefix_last_event") is not None:
        print(f"Last matching event: {_format_pairs(sorted(report['matching_prefix_last_event'].items()))}")

    comparison = report["comparison"]
    if comparison["status"] == "match":
        print("First divergence: none")
        return

    print(f"First divergence index: {comparison.get('index')}")
    resync_hint = report.get("resync_hint")
    if resync_hint is not None:
        print(
            "Resync hint: "
            f"type={resync_hint['type']} expected_skip={resync_hint['expected_skip']} "
            f"actual_skip={resync_hint['actual_skip']} "
            f"resumed_match_count={resync_hint['resumed_match_count']}"
        )
    expected_event = comparison.get("expected_event")
    actual_event = comparison.get("actual_event")

    if comparison["reason"] == "event_key_mismatch":
        print("Expected event key:")
        print(f"  {_format_pairs(comparison.get('expected_key', []))}")
        print("Actual event key:")
        print(f"  {_format_pairs(comparison.get('actual_key', []))}")
    elif comparison["reason"] == "event_value_mismatch":
        print("Mismatched fields:")
        for mismatch in comparison.get("mismatches", []):
            print(
                f"  {mismatch['field']}: "
                f"expected={mismatch['expected']} actual={mismatch['actual']}"
            )
    elif comparison["reason"] == "event_count_mismatch":
        print(
            f"Event count mismatch: expected={comparison.get('expected_count')} "
            f"actual={comparison.get('actual_count')}"
        )

    if expected_event is not None:
        print(
            f"Expected location: line {expected_event.get('_lineno')} "
            f"event={expected_event.get('event')} component={expected_event.get('component')}"
        )
        print(f"Expected raw: {expected_event.get('_line')}")
    if actual_event is not None:
        print(
            f"Actual location: line {actual_event.get('_lineno')} "
            f"event={actual_event.get('event')} component={actual_event.get('component')}"
        )
        print(f"Actual raw: {actual_event.get('_line')}")

    expected_window = report.get("expected_window", [])
    actual_window = report.get("actual_window", [])
    if expected_window:
        print("Expected window:")
        for entry in expected_window:
            print(f"  rel={entry['relative']} idx={entry['index']} {_format_pairs(sorted((k, v) for k, v in entry.items() if k not in ('relative', 'index')))}")
    if actual_window:
        print("Actual window:")
        for entry in actual_window:
            print(f"  rel={entry['relative']} idx={entry['index']} {_format_pairs(sorted((k, v) for k, v in entry.items() if k not in ('relative', 'index')))}")
    expected_contract_state = report.get("expected_contract_state")
    actual_contract_state = report.get("actual_contract_state")
    if expected_contract_state is not None:
        print(
            "Expected preceding contract state: "
            f"{_format_pairs(sorted(expected_contract_state.items()))}"
        )
    if actual_contract_state is not None:
        print(
            "Actual preceding contract state: "
            f"{_format_pairs(sorted(actual_contract_state.items()))}"
        )
    semantic_analysis = report.get("semantic_analysis")
    if semantic_analysis is not None:
        _print_semantic_analysis(semantic_analysis)
    family_analysis = report.get("family_analysis")
    if family_analysis is not None:
        print("Family analysis:")
        print(
            "  expected_identity="
            f"{_format_pairs(family_analysis.get('expected_identity') or [])}"
        )
        print(
            "  actual_identity="
            f"{_format_pairs(family_analysis.get('actual_identity') or [])}"
        )
        print(
            "  member_count:"
            f" expected={family_analysis.get('expected_member_count')}"
            f" actual={family_analysis.get('actual_member_count')}"
        )
        if family_analysis.get("missing_members"):
            print("  missing_members:")
            for item in family_analysis["missing_members"]:
                print(
                    f"    count={item.get('count')} "
                    f"{_format_pairs(item.get('member', []))}"
                )
        if family_analysis.get("unexpected_members"):
            print("  unexpected_members:")
            for item in family_analysis["unexpected_members"]:
                print(
                    f"    count={item.get('count')} "
                    f"{_format_pairs(item.get('member', []))}"
                )
        if family_analysis.get("multiplicity_mismatches"):
            print("  multiplicity_mismatches:")
            for item in family_analysis["multiplicity_mismatches"]:
                print(
                    "    "
                    f"expected_count={item.get('expected_count')} "
                    f"actual_count={item.get('actual_count')} "
                    f"{_format_pairs(item.get('member', []))}"
                )
        first_family_value_mismatch = family_analysis.get("first_shared_value_mismatch")
        if first_family_value_mismatch is not None:
            print(
                "  first_shared_value_mismatch:"
                f" {_format_pairs(first_family_value_mismatch.get('member', []))}"
            )
            for mismatch in first_family_value_mismatch.get("mismatches", []):
                print(
                    f"    {mismatch['field']}: "
                    f"expected={mismatch['expected']} actual={mismatch['actual']}"
                )
            if first_family_value_mismatch.get("semantic_analysis") is not None:
                _print_semantic_analysis(first_family_value_mismatch["semantic_analysis"])
    family_region_analysis = report.get("family_region_analysis")
    if family_region_analysis is not None:
        print("Family region analysis:")
        print(
            "  packet_index:"
            f" expected={family_region_analysis.get('expected_packet_index')}"
            f" actual={family_region_analysis.get('actual_packet_index')}"
        )
        print(
            "  packet_count:"
            f" expected={family_region_analysis.get('expected_packet_count')}"
            f" actual={family_region_analysis.get('actual_packet_count')}"
        )
        if family_region_analysis.get("expected_packet_window"):
            print("  expected_packet_window:")
            for packet in family_region_analysis["expected_packet_window"]:
                print(
                    f"    rel={packet.get('relative')} packet_idx={packet.get('packet_index')} "
                    f"events={packet.get('event_count')} "
                    f"{_format_pairs(packet.get('identity') or [])}"
                )
        if family_region_analysis.get("actual_packet_window"):
            print("  actual_packet_window:")
            for packet in family_region_analysis["actual_packet_window"]:
                print(
                    f"    rel={packet.get('relative')} packet_idx={packet.get('packet_index')} "
                    f"events={packet.get('event_count')} "
                    f"{_format_pairs(packet.get('identity') or [])}"
                )
        family_region_resync = family_region_analysis.get("resync_hint")
        if family_region_resync is not None:
            print(
                "  family_resync_hint:"
                f" type={family_region_resync.get('type')}"
                f" expected_skip={family_region_resync.get('expected_skip')}"
                f" actual_skip={family_region_resync.get('actual_skip')}"
                f" resumed_packet_signature_match_count={family_region_resync.get('resumed_packet_signature_match_count')}"
                f" resumed_packet_exact_match_count={family_region_resync.get('resumed_packet_exact_match_count')}"
            )
            if family_region_resync.get("expected_skipped_packets"):
                print("  expected_skipped_packets:")
                for packet in family_region_resync["expected_skipped_packets"]:
                    print(
                        f"    events={packet.get('event_count')} "
                        f"unique_members={packet.get('signature', {}).get('unique_member_count')} "
                        f"{_format_pairs(packet.get('identity') or [])}"
                    )
            if family_region_resync.get("actual_skipped_packets"):
                print("  actual_skipped_packets:")
                for packet in family_region_resync["actual_skipped_packets"]:
                    print(
                        f"    events={packet.get('event_count')} "
                        f"unique_members={packet.get('signature', {}).get('unique_member_count')} "
                        f"{_format_pairs(packet.get('identity') or [])}"
                    )
    if resync_hint is not None:
        if resync_hint.get("expected_skipped"):
            print("Expected skipped events:")
            for entry in resync_hint["expected_skipped"]:
                print(f"  {_format_pairs(sorted(entry.items()))}")
        if resync_hint.get("actual_skipped"):
            print("Actual skipped events:")
            for entry in resync_hint["actual_skipped"]:
                print(f"  {_format_pairs(sorted(entry.items()))}")
        if resync_hint.get("expected_resume_event") is not None:
            print(
                "Expected resume event: "
                f"{_format_pairs(sorted(resync_hint['expected_resume_event'].items()))}"
            )
        if resync_hint.get("actual_resume_event") is not None:
            print(
                "Actual resume event: "
                f"{_format_pairs(sorted(resync_hint['actual_resume_event'].items()))}"
            )

    expected_summary = report.get("expected_summary", {})
    actual_summary = report.get("actual_summary", {})
    if expected_summary or actual_summary:
        print("Expected phase counts:")
        print(f"  {_format_pairs(sorted(expected_summary.get('context_phases', {}).items()))}")
        print("Actual phase counts:")
        print(f"  {_format_pairs(sorted(actual_summary.get('context_phases', {}).items()))}")


def _json_out_path(args: argparse.Namespace) -> Optional[Path]:
    if args.json_out:
        return Path(args.json_out).expanduser().resolve()
    run_dir = os.environ.get("KRASIS_RUN_DIR")
    if run_dir:
        return Path(run_dir).resolve() / "trace_diff.json"
    return None


def main() -> int:
    args = parse_args()
    expected_path = Path(args.expected).expanduser().resolve()
    actual_path = Path(args.actual).expanduser().resolve()
    if not expected_path.is_file():
        print(f"ERROR: Expected trace file not found: {expected_path}", file=sys.stderr)
        return 2
    if not actual_path.is_file():
        print(f"ERROR: Actual trace file not found: {actual_path}", file=sys.stderr)
        return 2

    include_events = _csv_set(args.events)
    include_components = _csv_set(args.components)
    include_contexts = _csv_set(args.contexts)
    include_phases = _csv_set(args.phases)
    ignore_fields = _csv_set(args.ignore_fields) or set()

    expected_events = _load_trace_events(
        expected_path,
        include_events=include_events,
        include_components=include_components,
        include_contexts=include_contexts,
        include_phases=include_phases,
        max_events=args.max_events,
    )
    actual_events = _load_trace_events(
        actual_path,
        include_events=include_events,
        include_components=include_components,
        include_contexts=include_contexts,
        include_phases=include_phases,
        max_events=args.max_events,
    )
    expected_all_events = _load_trace_events(
        expected_path,
        include_events=None,
        include_components=None,
        include_contexts=None,
        include_phases=None,
        max_events=0,
    )
    actual_all_events = _load_trace_events(
        actual_path,
        include_events=None,
        include_components=None,
        include_contexts=None,
        include_phases=None,
        max_events=0,
    )

    comparison = _diff_events(expected_events, actual_events, ignore_fields=ignore_fields)
    matching_prefix_count = _matching_prefix_length(expected_events, actual_events, ignore_fields=ignore_fields)
    matching_prefix_last_event = None
    if matching_prefix_count > 0:
        matching_prefix_last_event = _event_brief(expected_events[matching_prefix_count - 1])
    report = {
        "expected_path": str(expected_path),
        "actual_path": str(actual_path),
        "expected_event_count": len(expected_events),
        "actual_event_count": len(actual_events),
        "matching_prefix_count": matching_prefix_count,
        "matching_prefix_last_event": matching_prefix_last_event,
        "filters": {
            "events": sorted(include_events) if include_events is not None else "all",
            "components": sorted(include_components) if include_components is not None else "all",
            "contexts": sorted(include_contexts) if include_contexts is not None else "all",
            "phases": sorted(include_phases) if include_phases is not None else "all",
            "ignore_fields": sorted(ignore_fields),
            "max_events": args.max_events,
            "show_window": args.show_window,
            "resync_window": args.resync_window,
            "family_resync_window": args.family_resync_window,
        },
        "expected_summary": _summarize_events(expected_events),
        "actual_summary": _summarize_events(actual_events),
        "comparison": comparison,
    }
    divergence_index = comparison.get("index", matching_prefix_count)
    report["expected_window"] = _divergence_window(expected_events, divergence_index, radius=max(args.show_window, 0))
    report["actual_window"] = _divergence_window(actual_events, divergence_index, radius=max(args.show_window, 0))
    report["expected_contract_state"] = _nearest_contract_state(
        expected_all_events,
        before_lineno=comparison.get("expected_event", {}).get("_lineno")
        if isinstance(comparison.get("expected_event"), dict)
        else None,
    )
    report["actual_contract_state"] = _nearest_contract_state(
        actual_all_events,
        before_lineno=comparison.get("actual_event", {}).get("_lineno")
        if isinstance(comparison.get("actual_event"), dict)
        else None,
    )
    report["resync_hint"] = None
    report["semantic_analysis"] = _semantic_mismatch_analysis(comparison)
    report["family_analysis"] = _family_analysis(
        expected_events,
        actual_events,
        index=divergence_index,
        ignore_fields=ignore_fields,
    )
    report["family_region_analysis"] = _family_region_analysis(
        expected_events,
        actual_events,
        event_index=divergence_index,
        ignore_fields=ignore_fields,
        packet_window=max(args.family_resync_window, 0),
    )
    if comparison["status"] != "match" and comparison.get("reason") in {
        "event_key_mismatch",
        "event_count_mismatch",
    }:
        report["resync_hint"] = _resync_hint(
            expected_events,
            actual_events,
            start_index=divergence_index,
            window=max(args.resync_window, 0),
            ignore_fields=ignore_fields,
        )
    report["classification"] = _classify_divergence(
        comparison,
        semantic_analysis=report["semantic_analysis"],
        family_analysis=report["family_analysis"],
        family_region_analysis=report["family_region_analysis"],
        resync_hint=report["resync_hint"],
    )

    _print_summary(report)

    output_path = _json_out_path(args)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"JSON report: {output_path}")

    return 0 if comparison["status"] == "match" else 1


if __name__ == "__main__":
    sys.exit(main())
