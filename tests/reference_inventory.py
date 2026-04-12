#!/usr/bin/env python3
"""Inventory stored reference artifacts and classify their contract state.

Usage:
    ./dev reference-inventory

This script must be run via ./dev reference-inventory, not directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from reference_contract import (
        build_reference_artifact_metadata,
        emit_contract_failure_trace,
        emit_reference_inventory_trace,
        load_tokenizer_with_compat,
        load_or_infer_reference_contract,
        resolve_local_reference_model_path,
        resolve_reference_model_name,
    )
except ModuleNotFoundError:
    from tests.reference_contract import (
        build_reference_artifact_metadata,
        emit_contract_failure_trace,
        emit_reference_inventory_trace,
        load_tokenizer_with_compat,
        load_or_infer_reference_contract,
        resolve_local_reference_model_path,
        resolve_reference_model_name,
    )


if not os.environ.get("KRASIS_DEV_SCRIPT"):
    print("ERROR: This script must be run via ./dev reference-inventory, not directly.")
    print("  Usage: ./dev reference-inventory")
    sys.exit(1)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_REFERENCE_ROOT = REPO_DIR.parent / "krasis-internal" / "reference-outputs" / "output"
LOCAL_REFERENCE_ROOT = SCRIPT_DIR / "reference_outputs"


def info(msg: str) -> None:
    print(f"=> {msg}")


def warn(msg: str) -> None:
    print(f"!! {msg}")


def die(msg: str) -> None:
    print(f"ERROR {msg}", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory stored reference artifacts")
    parser.add_argument(
        "--reference-root",
        default=str(DEFAULT_REFERENCE_ROOT),
        help="Directory containing per-model reference output folders",
    )
    parser.add_argument(
        "--model-root",
        default=os.path.expanduser("~/.krasis/models"),
        help="Directory containing local HF model folders for tokenizer loading",
    )
    parser.add_argument(
        "--include-local",
        action="store_true",
        help="Also scan krasis/tests/reference_outputs if present",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional explicit JSON output path",
    )
    return parser.parse_args()


def _candidate_reference_files(root: Path) -> List[Path]:
    return sorted(root.glob("*/**/*.json"))


def _reference_paths(args: argparse.Namespace) -> List[Path]:
    roots = [Path(args.reference_root).expanduser()]
    if args.include_local:
        roots.append(LOCAL_REFERENCE_ROOT)

    paths: List[Path] = []
    for root in roots:
        if not root.is_dir():
            warn(f"Reference root not found, skipping: {root}")
            continue
        for path in _candidate_reference_files(root):
            if path.name.endswith(".json") and "greedy" in path.name:
                paths.append(path.resolve())
    return sorted(set(paths))


def _output_path(args: argparse.Namespace) -> Path:
    if args.json_out:
        return Path(args.json_out).expanduser().resolve()
    run_dir = os.environ.get("KRASIS_RUN_DIR")
    if run_dir:
        return Path(run_dir).resolve() / "reference_inventory.json"
    return REPO_DIR / "logs" / "reference_inventory.json"


def _load_tokenizer(model_path: str):
    return load_tokenizer_with_compat(model_path)


def _inventory_entry(reference_path: Path, reference_root: Path, model_root: str) -> Dict[str, Any]:
    with open(reference_path) as f:
        reference = json.load(f)

    try:
        reference_dir = str(reference_path.parent.relative_to(reference_root))
    except ValueError:
        reference_dir = reference_path.parent.name

    model_name = resolve_reference_model_name(reference) or reference_path.parent.name
    model_path = resolve_local_reference_model_path(reference, model_root=model_root)

    entry: Dict[str, Any] = {
        "reference_path": str(reference_path),
        "reference_dir": reference_dir,
        "model_name": model_name,
        "format_version": reference.get("format_version"),
        "generated_at": reference.get("generated_at"),
        "model_path_resolved": bool(model_path),
    }

    if not model_path:
        entry["status"] = "missing_model"
        entry["error"] = (
            f"Local model directory not found for reference model {model_name}. "
            f"Expected under {Path(model_root).expanduser()}."
        )
        return entry

    tokenizer = _load_tokenizer(model_path)
    reference_contract = load_or_infer_reference_contract(reference, model_path, tokenizer)
    artifact = build_reference_artifact_metadata(reference, reference_contract, str(reference_path))
    validation_errors = []
    if artifact.get("state") == "legacy_invalid":
        validation_errors.append(
            "legacy reference profile remains unknown; regenerate with an explicit profile"
        )

    entry.update(
        {
            "status": "ok",
            "resolved_model_path": model_path,
            "profile_id": reference_contract.get("profile_id"),
            "legacy_inferred": bool(reference_contract.get("legacy_inferred", False)),
            "capture_inference_source": reference_contract.get("extra", {}).get("capture_inference_source"),
            "reference_artifact": artifact,
            "error_count": len(validation_errors),
            "warning_count": 1 if reference_contract.get("legacy_inferred") else 0,
            "errors": validation_errors,
        }
    )
    if reference_contract.get("legacy_inferred"):
        entry["warnings"] = [
            "reference contract was inferred from legacy metadata; file-level fingerprint checks are unavailable"
        ]
    return entry


def build_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    state_counts = {
        "explicit_valid": 0,
        "legacy_inferable": 0,
        "legacy_invalid": 0,
    }
    missing_model = 0
    scan_errors = 0
    invalid_models: List[str] = []
    inferable_models: List[str] = []

    for entry in entries:
        status = entry.get("status")
        if status == "missing_model":
            missing_model += 1
            continue
        if status != "ok":
            scan_errors += 1
            continue
        state = entry.get("reference_artifact", {}).get("state")
        if state in state_counts:
            state_counts[state] += 1
        if state == "legacy_invalid":
            invalid_models.append(entry.get("model_name", "unknown"))
        elif state == "legacy_inferable":
            inferable_models.append(entry.get("model_name", "unknown"))

    return {
        "total": len(entries),
        "state_counts": state_counts,
        "missing_model_count": missing_model,
        "scan_error_count": scan_errors,
        "legacy_invalid_models": invalid_models,
        "legacy_inferable_models": inferable_models,
    }


def print_summary(summary: Dict[str, Any], entries: List[Dict[str, Any]]) -> None:
    print("Reference inventory summary")
    print(f"Total artifacts: {summary['total']}")
    print(f"Explicit valid: {summary['state_counts']['explicit_valid']}")
    print(f"Legacy inferable: {summary['state_counts']['legacy_inferable']}")
    print(f"Legacy invalid: {summary['state_counts']['legacy_invalid']}")
    print(f"Missing local model: {summary['missing_model_count']}")
    print(f"Scan errors: {summary['scan_error_count']}")

    invalid_entries = [
        entry for entry in entries
        if entry.get("reference_artifact", {}).get("state") == "legacy_invalid"
    ]
    if invalid_entries:
        print("")
        print("Legacy invalid references")
        for entry in invalid_entries:
            print(
                f"{entry['reference_dir']}: {Path(entry['reference_path']).name} "
                f"[model={entry['model_name']}, source={entry.get('capture_inference_source')}]"
            )

    inferable_entries = [
        entry for entry in entries
        if entry.get("reference_artifact", {}).get("state") == "legacy_inferable"
    ]
    if inferable_entries:
        print("")
        print("Legacy inferable references")
        for entry in inferable_entries:
            print(
                f"{entry['reference_dir']}: {Path(entry['reference_path']).name} "
                f"[model={entry['model_name']}, source={entry.get('capture_inference_source')}]"
            )

    missing_entries = [entry for entry in entries if entry.get("status") == "missing_model"]
    if missing_entries:
        print("")
        print("Missing local model paths")
        for entry in missing_entries:
            print(f"{entry['model_name']}: {entry['error']}")


def main() -> int:
    args = parse_args()
    reference_paths = _reference_paths(args)
    if not reference_paths:
        emit_contract_failure_trace(
            "reference_inventory",
            "missing_reference_root",
            reference_root=str(Path(args.reference_root).expanduser()),
        )
        die("No reference artifacts found")

    info(f"Scanning {len(reference_paths)} reference artifacts")
    entries: List[Dict[str, Any]] = []
    for reference_path in reference_paths:
        info(f"Classifying {reference_path}")
        try:
            entries.append(
                _inventory_entry(
                    reference_path,
                    Path(args.reference_root).expanduser().resolve(),
                    args.model_root,
                )
            )
        except Exception as exc:
            entries.append(
                {
                    "reference_path": str(reference_path),
                    "reference_dir": reference_path.parent.name,
                    "model_name": reference_path.parent.name,
                    "status": "scan_error",
                    "error": str(exc),
                }
            )

    summary = build_summary(entries)
    print("")
    print_summary(summary, entries)

    output = {
        "reference_root": str(Path(args.reference_root).expanduser().resolve()),
        "model_root": str(Path(args.model_root).expanduser().resolve()),
        "summary": summary,
        "entries": entries,
    }
    output_path = _output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print("")
    print(f"Inventory report: {output_path}")

    emit_reference_inventory_trace("reference_inventory", summary, entries)
    return 0 if summary["scan_error_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
