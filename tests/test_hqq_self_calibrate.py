import json
import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from safetensors import safe_open
import torch
from safetensors.torch import save_file

from krasis.attention_backend import (
    HQQ_ATTENTION_CACHE_DIRNAME,
    HQQ_ATTENTION_CACHE_VERSION,
    HQQ_DEFAULT_AXIS,
    HQQ_DEFAULT_GROUP_SIZE,
    HQQ_LAYOUT,
    hqq_attention_cache_dir,
    hqq_attention_manifest_path,
    load_hqq_attention_artifact,
    require_complete_hqq_attention_manifest,
    require_complete_hqq_sidecar_manifest,
)
from krasis.config import HQQ_CACHE_PROFILE_BASELINE, HQQ_CACHE_PROFILE_SELFCAL_V1
from krasis.hqq_self_calibrate import main as hqq_self_calibrate_main
from krasis.hqq_self_calibrate import _int8_exception_budget_bytes
from krasis.hqq_self_calibrate import _select_int8_exception_candidates
from krasis.hqq_self_calibrate import evidence_trace_log_path


@contextmanager
def isolated_home():
    old_home = os.environ.get("HOME")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["HOME"] = tmp
        try:
            yield Path(tmp)
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home


def write_baseline_manifest(model_path: Path) -> Path:
    manifest_path = Path(hqq_attention_manifest_path(str(model_path), HQQ_CACHE_PROFILE_BASELINE))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": HQQ_ATTENTION_CACHE_VERSION,
        "backend": "hqq4",
        "nbits": 4,
        "group_size": HQQ_DEFAULT_GROUP_SIZE,
        "axis": HQQ_DEFAULT_AXIS,
        "layout": HQQ_LAYOUT,
        "num_hidden_layers": 40,
        "complete": True,
        "tensors": [
            {
                "layer_idx": 0,
                "layer_type": "linear_attention",
                "tensor_name": "in_proj_qkvz",
                "source_tensor": "toy.attn.weight",
                "file": "layer_000_in_proj_qkvz_hqq4.safetensors",
                "nbits": 4,
                "group_size": HQQ_DEFAULT_GROUP_SIZE,
                "axis": HQQ_DEFAULT_AXIS,
                "layout": HQQ_LAYOUT,
                "original_shape": [1, 4],
                "quality": {
                    "rmse": 0.25,
                    "max_abs": 0.5,
                    "worst_groups": [
                        {
                            "row": 0,
                            "group": 0,
                            "start_col": 0,
                            "end_col": 4,
                            "mean_abs": 0.2,
                            "rmse": 0.25,
                            "max_abs": 0.5,
                        }
                    ],
                },
            }
        ],
        "totals": {"tensor_bytes": 0, "num_tensors": 1},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    save_file(
        {
            "packed": torch.tensor([[0xF0, 0x71] + [0x00] * 62], dtype=torch.uint8),
            "scales": torch.ones((1, 1), dtype=torch.float32),
            "zeros": torch.zeros((1, 1), dtype=torch.float32),
            "orig_shape": torch.tensor([1, 4], dtype=torch.int32),
            "group_size": torch.tensor([HQQ_DEFAULT_GROUP_SIZE], dtype=torch.int32),
            "axis": torch.tensor([HQQ_DEFAULT_AXIS], dtype=torch.int32),
            "nbits": torch.tensor([4], dtype=torch.int32),
        },
        str(manifest_path.parent / "layer_000_in_proj_qkvz_hqq4.safetensors"),
        metadata={"backend": "hqq4", "layout": HQQ_LAYOUT},
    )
    save_file(
        {"toy.attn.weight": torch.tensor([[0.0, 14.0, 2.0, 6.0]], dtype=torch.bfloat16)},
        str(model_path / "model-00001-of-00001.safetensors"),
    )
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"toy.attn.weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    return manifest_path


def write_config(path: Path, model_path: Path, attention_quant: str = "hqq4") -> None:
    path.write_text(
        "\n".join(
            [
                f'MODEL_PATH="{model_path}"',
                f'CFG_ATTENTION_QUANT="{attention_quant}"',
                'CFG_HQQ_CACHE_PROFILE="baseline"',
                'CFG_PORT="18099"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_trace(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                (
                    "event=input_row_full layer=0 tensor=la_input_row_for_qkvz_last "
                    "pos=4 width=4 hash64=0x1 bf16_hex=3f8000004000c000"
                ),
                (
                    "event=input_row_full layer=0 tensor=la_input_row_for_qkvz_last "
                    "pos=8 width=4 hash64=0x2 bf16_hex=408000004100c080"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )


def assert_raises_contains(fn, text: str) -> None:
    try:
        fn()
    except Exception as exc:
        if text not in str(exc):
            raise AssertionError(f"Expected {text!r} in {str(exc)!r}") from exc
        return
    raise AssertionError(f"Expected exception containing {text!r}")


def make_int8_exception_candidate(
    *,
    layer: int,
    tensor: str,
    group: int,
    benefit: float,
    cost: int,
    ready: bool = True,
    surface: str = "both",
) -> dict:
    prefill_active = surface in ("both", "prefill_only")
    decode_active = surface in ("both", "decode_only")
    return {
        "layer": layer,
        "target_tensor": tensor,
        "group": group,
        "candidate_ready": ready,
        "benefit_score": benefit,
        "risk_score": benefit,
        "cost": {"int8_total_bytes": cost, "runtime_total_bytes": cost},
        "runtime_activity": {
            "prefill_active": prefill_active,
            "decode_active": decode_active,
            "active": prefill_active or decode_active,
            "surface": surface,
            "inactive_reason": None if prefill_active or decode_active else "test inactive",
        },
    }


def test_int8_exception_budget_selection_uses_ratio_and_top_limit() -> None:
    report = {
        "candidates": [
            make_int8_exception_candidate(
                layer=0,
                tensor="in_proj_qkvz",
                group=0,
                benefit=90.0,
                cost=60,
            ),
            make_int8_exception_candidate(
                layer=0,
                tensor="out_proj",
                group=0,
                benefit=30.0,
                cost=10,
            ),
            make_int8_exception_candidate(
                layer=1,
                tensor="fused_qkv",
                group=0,
                benefit=10.0,
                cost=10,
            ),
        ]
    }
    budget = _int8_exception_budget_bytes(
        source_hqq_bytes=1000,
        max_sidecar_ratio=0.07,
        max_sidecar_bytes=None,
    )
    selected = _select_int8_exception_candidates(
        report,
        requested_groups=None,
        top_groups=2,
        max_sidecar_bytes=budget,
    )
    assert [(item["target_tensor"], item["group"]) for item in selected] == [
        ("out_proj", 0),
        ("in_proj_qkvz", 0),
    ]
    assert sum(item["cost"]["runtime_total_bytes"] for item in selected) <= 70


def test_int8_exception_budget_rejects_explicit_groups_over_cap() -> None:
    report = {
        "candidates": [
            make_int8_exception_candidate(
                layer=0,
                tensor="in_proj_qkvz",
                group=0,
                benefit=90.0,
                cost=60,
            )
        ]
    }
    assert_raises_contains(
        lambda: _select_int8_exception_candidates(
            report,
            requested_groups={(0, "in_proj_qkvz", 0)},
            top_groups=None,
            max_sidecar_bytes=50,
        ),
        "exceed sidecar byte budget",
    )
    assert_raises_contains(
        lambda: _int8_exception_budget_bytes(
            source_hqq_bytes=0,
            max_sidecar_ratio=0.10,
            max_sidecar_bytes=None,
        ),
        "requires positive source HQQ tensor bytes",
    )


def test_int8_exception_split_budget_prefers_decode_and_excludes_inactive() -> None:
    report = {
        "candidates": [
            make_int8_exception_candidate(
                layer=0,
                tensor="q_proj",
                group=0,
                benefit=500.0,
                cost=20,
                surface="prefill_only",
            ),
            make_int8_exception_candidate(
                layer=0,
                tensor="fused_qkv",
                group=0,
                benefit=100.0,
                cost=60,
                surface="decode_only",
            ),
            make_int8_exception_candidate(
                layer=1,
                tensor="out_proj",
                group=0,
                benefit=70.0,
                cost=30,
                surface="both",
            ),
            make_int8_exception_candidate(
                layer=2,
                tensor="unused_alias",
                group=0,
                benefit=999.0,
                cost=1,
                surface="inactive",
            ),
        ]
    }
    selected = _select_int8_exception_candidates(
        report,
        requested_groups=None,
        top_groups=None,
        max_sidecar_bytes=100,
        max_decode_sidecar_bytes=80,
        max_prefill_sidecar_bytes=20,
    )
    assert [(item["target_tensor"], item["group"]) for item in selected] == [
        ("out_proj", 0),
        ("q_proj", 0),
    ]
    assert all(item["target_tensor"] != "unused_alias" for item in selected)
    assert sum(item["cost"]["runtime_total_bytes"] for item in selected) <= 100


def test_hqq8_rejects_self_calibration_and_sidecar_workflows() -> None:
    with isolated_home() as home:
        model_path = home / "models" / "TinyModel"
        model_path.mkdir(parents=True)
        config_path = home / "tiny_hqq8.conf"
        trace_path = home / "trace.log"
        output_path = home / "hqq8_selfcal_rejected.json"

        write_baseline_manifest(model_path)
        write_config(config_path, model_path, attention_quant="hqq8")
        write_trace(trace_path)

        assert_raises_contains(
            lambda: hqq_self_calibrate_main(
                [
                    "--config",
                    str(config_path),
                    "--trace-log",
                    str(trace_path),
                    "--output",
                    str(output_path),
                    "--duration-seconds",
                    "1",
                ]
            ),
            'hqq-self-calibrate requires CFG_ATTENTION_QUANT="hqq4"',
        )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        data = cpu.view(torch.uint16).numpy().tobytes()
    else:
        data = cpu.numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def write_source_contract_audit(path: Path, model_path: Path) -> None:
    manifest_path = Path(hqq_attention_manifest_path(str(model_path), HQQ_CACHE_PROFILE_BASELINE))
    artifact_path = manifest_path.parent / "layer_000_in_proj_qkvz_hqq4.safetensors"
    with safe_open(str(model_path / "model-00001-of-00001.safetensors"), framework="pt", device="cpu") as handle:
        source = handle.get_tensor("toy.attn.weight").contiguous()
    path.write_text(
        json.dumps(
            {
                "format": "krasis_hqq_source_map_audit",
                "targets": [
                    {
                        "layer": 0,
                        "tensor": "in_proj_qkvz",
                        "source_contract": {
                            "ready": True,
                            "fallback_allowed": False,
                            "resolver_kind": "qcn_model_layers_direct",
                            "layer": 0,
                            "target_tensor": "in_proj_qkvz",
                            "target_file": "layer_000_in_proj_qkvz_hqq4.safetensors",
                            "target_shape": [1, 4],
                            "source_shape": [1, 4],
                            "reconstructed_source_shape": [1, 4],
                            "reconstructed_source_sha256": tensor_sha256(source),
                            "artifact_sha256": file_sha256(artifact_path),
                            "row_column_layout": "source rows are output rows; columns are input columns",
                            "source_tensors": [
                                {
                                    "name": "toy.attn.weight",
                                    "shape": [1, 4],
                                    "dtype": "BF16",
                                    "source_file": "model-00001-of-00001.safetensors",
                                    "source_path": str(model_path / "model-00001-of-00001.safetensors"),
                                    "source_sha256": tensor_sha256(source),
                                }
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def write_witness_summary(path: Path, deltas, *, first_match=True, top_overlap=9) -> None:
    prompt_results = []
    for idx, delta in enumerate(deltas):
        prompt_results.append(
            {
                "conv_idx": idx,
                "turn": 1,
                "verdict": "PASS" if first_match else "FAIL",
                "metrics": {
                    "first_match": first_match,
                    "first_token_diagnostic": {
                        "first_token_match": first_match,
                        "ref_first_token": 100 + idx,
                        "our_first_token": 100 + idx if first_match else 200 + idx,
                        "top_overlap_count": top_overlap,
                        "selected_logprob_delta": float(delta),
                    },
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "overall": "PASS" if first_match else "FAIL",
                "prompt_results": prompt_results,
            }
        ),
        encoding="utf-8",
    )


def test_evidence_file_is_not_loadable_calibrated_manifest() -> None:
    with isolated_home() as home:
        model_path = home / "models" / "TinyModel"
        model_path.mkdir(parents=True)
        config_path = home / "tiny.conf"
        trace_path = home / "trace.log"
        output_path = (
            Path(hqq_attention_cache_dir(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1))
            / "calibration"
            / "evidence_test.json"
        )

        write_baseline_manifest(model_path)
        write_config(config_path, model_path)
        write_trace(trace_path)

        rc = hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--trace-log",
                str(trace_path),
                "--output",
                str(output_path),
                "--duration-seconds",
                "1",
            ]
        )
        assert rc == 0
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["format"] == "krasis_hqq_selfcal_evidence"
        assert payload["complete"] is False
        assert payload["loadable_hqq_manifest"] is False
        assert payload["activation_stats"]["input_row_full_events"] == 2
        field = payload["activation_stats"]["fields"][0]
        assert field["rows"] == 2
        assert field["width"] == 4
        assert field["top_groups_by_max_abs"][0]["max_abs"] == 8.0

        calibrated_manifest = Path(hqq_attention_manifest_path(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1))
        assert not calibrated_manifest.exists()
        assert_raises_contains(
            lambda: require_complete_hqq_attention_manifest(
                str(model_path),
                HQQ_CACHE_PROFILE_SELFCAL_V1,
                expected_nbits=4,
                expected_num_hidden_layers=40,
            ),
            "No fallback to baseline is allowed",
        )


def test_sensitivity_candidates_are_not_loadable_calibrated_manifest() -> None:
    with isolated_home() as home:
        model_path = home / "models" / "TinyModel"
        model_path.mkdir(parents=True)
        config_path = home / "tiny.conf"
        trace_path = home / "trace.log"
        calib_dir = Path(hqq_attention_cache_dir(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1)) / "calibration"
        evidence_path = calib_dir / "evidence_test.json"
        candidate_path = calib_dir / "sensitivity_candidates_test.json"

        write_baseline_manifest(model_path)
        write_config(config_path, model_path)
        write_trace(trace_path)

        rc = hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--trace-log",
                str(trace_path),
                "--output",
                str(evidence_path),
                "--duration-seconds",
                "1",
            ]
        )
        assert rc == 0
        rc = hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--build-candidates",
                str(evidence_path),
                "--output",
                str(candidate_path),
                "--top-candidates",
                "4",
            ]
        )
        assert rc == 0
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        assert payload["format"] == "krasis_hqq_selfcal_sensitivity_candidates"
        assert payload["complete"] is False
        assert payload["loadable_hqq_manifest"] is False
        assert payload["summary"]["candidate_count"] == 1
        top = payload["top_candidates"][0]
        assert top["target_tensor"] == "in_proj_qkvz"
        assert top["activation_policy"] == "shared_bf16_hqq_activation"
        assert top["qvalue_saturation"]["saturation_fraction"] == 0.5

        calibrated_manifest = Path(hqq_attention_manifest_path(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1))
        assert not calibrated_manifest.exists()
        assert_raises_contains(
            lambda: require_complete_hqq_attention_manifest(
                str(model_path),
                HQQ_CACHE_PROFILE_SELFCAL_V1,
                expected_nbits=4,
                expected_num_hidden_layers=40,
            ),
            "No fallback to baseline is allowed",
        )


def test_int8_exception_candidates_are_intrinsic_and_non_loadable() -> None:
    with isolated_home() as home:
        model_path = home / "models" / "TinyModel"
        model_path.mkdir(parents=True)
        config_path = home / "tiny.conf"
        train_trace_path = home / "train.trace.log"
        heldout_trace_path = home / "heldout.trace.log"
        calib_dir = Path(hqq_attention_cache_dir(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1)) / "calibration"
        train_evidence_path = calib_dir / "train_evidence_test.json"
        heldout_evidence_path = calib_dir / "heldout_evidence_test.json"
        exception_path = calib_dir / "int8_exception_candidates_test.json"
        write_path = calib_dir / "int8_exception_write_test.json"
        source_contract_path = home / "source_contract_audit.json"

        write_baseline_manifest(model_path)
        write_config(config_path, model_path)
        write_trace(train_trace_path)
        write_trace(heldout_trace_path)
        write_source_contract_audit(source_contract_path, model_path)

        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--trace-log",
                str(train_trace_path),
                "--output",
                str(train_evidence_path),
                "--duration-seconds",
                "1",
            ]
        ) == 0
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--trace-log",
                str(heldout_trace_path),
                "--output",
                str(heldout_evidence_path),
                "--duration-seconds",
                "1",
            ]
        ) == 0
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--build-int8-exception-candidates",
                str(train_evidence_path),
                "--heldout-evidence",
                str(heldout_evidence_path),
                "--source-contract-audit",
                str(source_contract_path),
                "--output",
                str(exception_path),
                "--patch-tensors",
                "in_proj_qkvz",
                "--top-candidates",
                "1",
            ]
        ) == 0

        payload = json.loads(exception_path.read_text(encoding="utf-8"))
        assert payload["format"] == "krasis_hqq_int8_exception_candidates"
        assert payload["complete"] is False
        assert payload["loadable_hqq_manifest"] is False
        assert payload["runtime_application_implemented"] is True
        assert payload["summary"]["candidate_count"] == 1
        assert payload["summary"]["candidate_ready_count"] == 1
        top = payload["top_candidates"][0]
        assert top["target_tensor"] == "in_proj_qkvz"
        assert top["layout"]["kind"] == "full_output_rows_by_hqq_column_group"
        assert top["candidate_ready"] is True
        assert top["projection"]["heldout"]["reduction_vs_hqq"]["rms_reduction_fraction"] > 0
        assert top["cost"]["int8_total_bytes"] > 0
        assert (exception_path.parent / "int8_exception_candidates_test.tsv").is_file()

        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--write-int8-exception-manifest",
                str(exception_path),
                "--variant-name",
                "toy_int8_exception",
                "--int8-exception-groups",
                "0:in_proj_qkvz:0",
                "--output",
                str(write_path),
            ]
        ) == 0
        write_payload = json.loads(write_path.read_text(encoding="utf-8"))
        assert write_payload["format"] == "krasis_hqq_int8_exception_manifest_write"
        assert write_payload["runtime_ready"] is True
        assert write_payload["summary"]["exception_group_count"] == 1
        assert write_payload["summary"]["row_group_count"] == 1
        manifest_path = Path(write_payload["cache"]["sidecar_manifest_path"])
        manifest = require_complete_hqq_sidecar_manifest(
            str(manifest_path),
            model_path=str(model_path),
            source_cache_profile=HQQ_CACHE_PROFILE_BASELINE,
        )
        assert manifest["sidecar_mode"] == "int8_exception"
        artifact = manifest["artifacts"][0]
        with safe_open(str(manifest_path.parent / artifact["file"]), framework="pt", device="cpu") as handle:
            q = handle.get_tensor("exception_qint8").to(torch.float32)
            scales = handle.get_tensor("scales").to(torch.float32)
            replacement = q * scales[:, None]
        with safe_open(str(model_path / "model-00001-of-00001.safetensors"), framework="pt", device="cpu") as handle:
            source = handle.get_tensor("toy.attn.weight").to(torch.float32)
        expected_scale = torch.clamp(source.abs().amax(dim=1, keepdim=True) / 127.0, min=1e-12)
        expected_q = torch.clamp(torch.round(source / expected_scale), -127, 127).to(torch.int8)
        expected_replacement = expected_q.to(torch.float32) * expected_scale
        assert torch.equal(replacement, expected_replacement)
        assert torch.max(torch.abs(replacement - source)).item() < 0.06

        assert_raises_contains(
            lambda: hqq_self_calibrate_main(
                [
                    "--config",
                    str(config_path),
                    "--write-int8-exception-manifest",
                    str(exception_path),
                    "--variant-name",
                    "toy_implicit_rejected",
                    "--output",
                    str(calib_dir / "int8_exception_implicit_rejected.json"),
                ]
            ),
            "requires either --int8-exception-groups or --int8-exception-top-groups",
        )
        assert_raises_contains(
            lambda: hqq_self_calibrate_main(
                [
                    "--config",
                    str(config_path),
                    "--write-int8-exception-manifest",
                    str(exception_path),
                    "--variant-name",
                    "toy_missing_group",
                    "--int8-exception-groups",
                    "0:in_proj_qkvz:99",
                    "--output",
                    str(calib_dir / "int8_exception_missing_group.json"),
                ]
            ),
            "absent or not candidate_ready",
        )

        bad_hash_report_path = calib_dir / "int8_exception_candidates_bad_hash_test.json"
        bad_hash = json.loads(exception_path.read_text(encoding="utf-8"))
        bad_hash["top_candidates"][0]["artifact_hashes"]["source_group_sha256"] = "0" * 64
        bad_hash["candidates"][0]["artifact_hashes"]["source_group_sha256"] = "0" * 64
        bad_hash_report_path.write_text(json.dumps(bad_hash), encoding="utf-8")
        assert_raises_contains(
            lambda: hqq_self_calibrate_main(
                [
                    "--config",
                    str(config_path),
                    "--write-int8-exception-manifest",
                    str(bad_hash_report_path),
                    "--variant-name",
                    "toy_bad_hash",
                    "--int8-exception-top-groups",
                    "1",
                    "--output",
                    str(calib_dir / "int8_exception_bad_hash_write.json"),
                ]
            ),
            "source group sha256 mismatch",
        )

        bad_bounds_report_path = calib_dir / "int8_exception_candidates_bad_bounds_test.json"
        bad_bounds = json.loads(exception_path.read_text(encoding="utf-8"))
        bad_bounds["top_candidates"][0]["end_col"] = 5
        bad_bounds["candidates"][0]["end_col"] = 5
        bad_bounds_report_path.write_text(json.dumps(bad_bounds), encoding="utf-8")
        assert_raises_contains(
            lambda: hqq_self_calibrate_main(
                [
                    "--config",
                    str(config_path),
                    "--write-int8-exception-manifest",
                    str(bad_bounds_report_path),
                    "--variant-name",
                    "toy_bad_bounds",
                    "--int8-exception-top-groups",
                    "1",
                    "--output",
                    str(calib_dir / "int8_exception_bad_bounds_write.json"),
                ]
            ),
            "bounds do not match",
        )

        bad_manifest_path = manifest_path.parent / "sidecar_manifest_bad_hash.json"
        bad_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bad_manifest["artifacts"][0]["sha256"] = "0" * 64
        bad_manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
        assert_raises_contains(
            lambda: require_complete_hqq_sidecar_manifest(
                str(bad_manifest_path),
                model_path=str(model_path),
                source_cache_profile=HQQ_CACHE_PROFILE_BASELINE,
            ),
            "artifact sha256 mismatch",
        )

        calibrated_manifest = Path(hqq_attention_manifest_path(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1))
        assert not calibrated_manifest.exists()
        assert_raises_contains(
            lambda: require_complete_hqq_attention_manifest(
                str(model_path),
                HQQ_CACHE_PROFILE_SELFCAL_V1,
                expected_nbits=4,
                expected_num_hidden_layers=40,
            ),
            "No fallback to baseline is allowed",
        )


def test_int8_exception_candidate_build_rejects_shared_train_heldout_trace() -> None:
    with isolated_home() as home:
        model_path = home / "models" / "TinyModel"
        model_path.mkdir(parents=True)
        config_path = home / "tiny.conf"
        trace_path = home / "shared.trace.log"
        calib_dir = Path(hqq_attention_cache_dir(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1)) / "calibration"
        train_evidence_path = calib_dir / "train_evidence_test.json"
        heldout_evidence_path = calib_dir / "heldout_evidence_test.json"
        exception_path = calib_dir / "int8_exception_candidates_test.json"
        source_contract_path = home / "source_contract_audit.json"

        write_baseline_manifest(model_path)
        write_config(config_path, model_path)
        write_trace(trace_path)
        write_source_contract_audit(source_contract_path, model_path)

        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--trace-log",
                str(trace_path),
                "--output",
                str(train_evidence_path),
                "--duration-seconds",
                "1",
            ]
        ) == 0
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--trace-log",
                str(trace_path),
                "--output",
                str(heldout_evidence_path),
                "--duration-seconds",
                "1",
            ]
        ) == 0

        assert_raises_contains(
            lambda: hqq_self_calibrate_main(
                [
                    "--config",
                    str(config_path),
                    "--build-int8-exception-candidates",
                    str(train_evidence_path),
                    "--heldout-evidence",
                    str(heldout_evidence_path),
                    "--source-contract-audit",
                    str(source_contract_path),
                    "--output",
                    str(exception_path),
                    "--patch-tensors",
                    "in_proj_qkvz",
                    "--top-candidates",
                    "1",
                ]
            ),
            "same trace log",
        )


def test_evidence_trace_log_path_is_unique_per_output() -> None:
    first = evidence_trace_log_path("/tmp/calibration/train evidence.json")
    second = evidence_trace_log_path("/tmp/calibration/heldout_evidence.json")
    assert first.endswith("train_evidence.trace.log")
    assert second.endswith("heldout_evidence.trace.log")
    assert first != second


def test_refit_evaluation_is_not_loadable_calibrated_manifest() -> None:
    with isolated_home() as home:
        model_path = home / "models" / "TinyModel"
        model_path.mkdir(parents=True)
        config_path = home / "tiny.conf"
        trace_path = home / "trace.log"
        calib_dir = Path(hqq_attention_cache_dir(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1)) / "calibration"
        evidence_path = calib_dir / "evidence_test.json"
        candidate_path = calib_dir / "sensitivity_candidates_test.json"
        eval_path = calib_dir / "refit_evaluation_test.json"

        write_baseline_manifest(model_path)
        write_config(config_path, model_path)
        write_trace(trace_path)

        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--trace-log",
                str(trace_path),
                "--output",
                str(evidence_path),
                "--duration-seconds",
                "1",
            ]
        ) == 0
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--build-candidates",
                str(evidence_path),
                "--output",
                str(candidate_path),
                "--top-candidates",
                "4",
            ]
        ) == 0
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--evaluate-refits",
                str(candidate_path),
                "--trace-log",
                str(trace_path),
                "--output",
                str(eval_path),
                "--top-candidates",
                "1",
                "--max-output-rows",
                "1",
                "--refit-grid-steps",
                "17",
                "--refit-local-grid-steps",
                "9",
            ]
        ) == 0
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        assert payload["format"] == "krasis_hqq_selfcal_refit_evaluation"
        assert payload["complete"] is False
        assert payload["loadable_hqq_manifest"] is False
        assert payload["summary"]["evaluated_candidate_count"] == 1
        row = payload["evaluations"][0]["selected_output_rows"][0]
        assert row["best_candidate_by_projection_rms"] in {
            "cached_hqq",
            "minmax_affine",
            "activation_weighted_affine",
            "activation_weighted_clip_free",
        }

        calibrated_manifest = Path(hqq_attention_manifest_path(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1))
        assert not calibrated_manifest.exists()
        assert_raises_contains(
            lambda: require_complete_hqq_attention_manifest(
                str(model_path),
                HQQ_CACHE_PROFILE_SELFCAL_V1,
                expected_nbits=4,
                expected_num_hidden_layers=40,
            ),
            "No fallback to baseline is allowed",
        )


def test_sidecar_simulation_is_non_loadable_and_reports_estimates() -> None:
    with isolated_home() as home:
        model_path = home / "models" / "TinyModel"
        model_path.mkdir(parents=True)
        config_path = home / "tiny.conf"
        trace_path = home / "trace.log"
        calib_dir = Path(hqq_attention_cache_dir(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1)) / "calibration"
        evidence_path = calib_dir / "evidence_test.json"
        candidate_path = calib_dir / "sensitivity_candidates_test.json"
        eval_path = calib_dir / "refit_evaluation_test.json"
        sidecar_path = calib_dir / "sidecar_simulation_test.json"

        write_baseline_manifest(model_path)
        write_config(config_path, model_path)
        write_trace(trace_path)

        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--trace-log",
                str(trace_path),
                "--output",
                str(evidence_path),
                "--duration-seconds",
                "1",
            ]
        ) == 0
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--build-candidates",
                str(evidence_path),
                "--output",
                str(candidate_path),
                "--top-candidates",
                "4",
            ]
        ) == 0
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--evaluate-refits",
                str(candidate_path),
                "--trace-log",
                str(trace_path),
                "--output",
                str(eval_path),
                "--top-candidates",
                "1",
                "--max-output-rows",
                "1",
                "--refit-grid-steps",
                "17",
                "--refit-local-grid-steps",
                "9",
            ]
        ) == 0
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--simulate-sidecars",
                str(eval_path),
                "--trace-log",
                str(trace_path),
                "--output",
                str(sidecar_path),
                "--patch-top-groups",
                "1",
            ]
        ) == 0

        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert payload["format"] == "krasis_hqq_selfcal_sidecar_simulation"
        assert payload["complete"] is False
        assert payload["loadable_hqq_manifest"] is False
        assert payload["sidecar_model"]["main_int4_path"] == "unchanged regular HQQ INT4"
        assert payload["sidecar_model"]["sentinel_values"] is False
        assert payload["summary"]["simulated_row_group_count"] >= 1
        aggregate = payload["summary"]["aggregate_by_mode"]
        assert aggregate["exact_bf16_sidecar"]["sidecar_weight_bytes"] > 0
        assert aggregate["int8_symmetric_sidecar"]["sidecar_total_bytes"] > 0
        assert payload["summary"]["improvement_vs_cached"]["exact_bf16_sidecar"][
            "projection_abs_sum_reduction_fraction"
        ] >= 0.999

        calibrated_manifest = Path(hqq_attention_manifest_path(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1))
        assert not calibrated_manifest.exists()
        assert_raises_contains(
            lambda: require_complete_hqq_attention_manifest(
                str(model_path),
                HQQ_CACHE_PROFILE_SELFCAL_V1,
                expected_nbits=4,
                expected_num_hidden_layers=40,
            ),
            "No fallback to baseline is allowed",
        )


def test_sidecar_artifact_writer_stays_separate_from_hqq_manifest() -> None:
    with isolated_home() as home:
        model_path = home / "models" / "TinyModel"
        model_path.mkdir(parents=True)
        config_path = home / "tiny.conf"
        trace_path = home / "trace.log"
        calib_dir = Path(hqq_attention_cache_dir(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1)) / "calibration"
        evidence_path = calib_dir / "evidence_test.json"
        candidate_path = calib_dir / "sensitivity_candidates_test.json"
        eval_path = calib_dir / "refit_evaluation_test.json"
        sidecar_sim_path = calib_dir / "sidecar_simulation_test.json"
        sidecar_write_path = calib_dir / "sidecar_write_test.json"

        write_baseline_manifest(model_path)
        write_config(config_path, model_path)
        write_trace(trace_path)
        baseline_manifest_path = Path(hqq_attention_manifest_path(str(model_path), HQQ_CACHE_PROFILE_BASELINE))
        baseline_artifact_path = baseline_manifest_path.parent / "layer_000_in_proj_qkvz_hqq4.safetensors"
        baseline_manifest_hash = file_sha256(baseline_manifest_path)
        baseline_artifact_hash = file_sha256(baseline_artifact_path)

        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--trace-log",
                str(trace_path),
                "--output",
                str(evidence_path),
                "--duration-seconds",
                "1",
            ]
        ) == 0
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--build-candidates",
                str(evidence_path),
                "--output",
                str(candidate_path),
                "--top-candidates",
                "4",
            ]
        ) == 0
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--evaluate-refits",
                str(candidate_path),
                "--trace-log",
                str(trace_path),
                "--output",
                str(eval_path),
                "--top-candidates",
                "1",
                "--max-output-rows",
                "1",
                "--refit-grid-steps",
                "17",
                "--refit-local-grid-steps",
                "9",
            ]
        ) == 0
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--simulate-sidecars",
                str(eval_path),
                "--trace-log",
                str(trace_path),
                "--output",
                str(sidecar_sim_path),
                "--sidecar-modes",
                "int8_symmetric",
                "--patch-top-groups",
                "1",
            ]
        ) == 0
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--write-sidecar-artifact",
                str(sidecar_sim_path),
                "--output",
                str(sidecar_write_path),
                "--variant-name",
                "toy_sidecar",
                "--sidecar-artifact-mode",
                "int8_symmetric",
            ]
        ) == 0
        filtered_write_path = calib_dir / "sidecar_write_filtered_test.json"
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--write-sidecar-artifact",
                str(sidecar_sim_path),
                "--output",
                str(filtered_write_path),
                "--variant-name",
                "toy_sidecar_filtered",
                "--sidecar-artifact-mode",
                "int8_symmetric",
                "--sidecar-include-entries",
                "0:in_proj_qkvz:0:0",
            ]
        ) == 0
        filtered_report = json.loads(filtered_write_path.read_text(encoding="utf-8"))
        assert filtered_report["summary"]["row_group_count"] == 1
        assert filtered_report["sidecar_manifest"]["entry_filter"]["include_entries"] == [
            "0:in_proj_qkvz:0:0"
        ]
        scaled_write_path = calib_dir / "sidecar_write_scaled_test.json"
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--write-sidecar-artifact",
                str(sidecar_sim_path),
                "--output",
                str(scaled_write_path),
                "--variant-name",
                "toy_sidecar_scaled",
                "--sidecar-artifact-mode",
                "int8_symmetric",
                "--sidecar-correction-scale",
                "0.5",
            ]
        ) == 0
        scaled_report = json.loads(scaled_write_path.read_text(encoding="utf-8"))
        assert scaled_report["correction_scale"] == 0.5
        assert scaled_report["sidecar_manifest"]["correction_scale"] == 0.5
        assert scaled_report["summary"]["correction_scale"] == 0.5
        assert scaled_report["sidecar_manifest"]["artifacts"][0]["entries"][0]["correction_scale"] == 0.5
        entry_scaled_write_path = calib_dir / "sidecar_write_entry_scaled_test.json"
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--write-sidecar-artifact",
                str(sidecar_sim_path),
                "--output",
                str(entry_scaled_write_path),
                "--variant-name",
                "toy_sidecar_entry_scaled",
                "--sidecar-artifact-mode",
                "int8_symmetric",
                "--sidecar-correction-scale",
                "0.75",
                "--sidecar-include-entries",
                "0:in_proj_qkvz:0:0",
                "--sidecar-entry-scales",
                "0:in_proj_qkvz:0:0=0.25",
            ]
        ) == 0
        entry_scaled_report = json.loads(entry_scaled_write_path.read_text(encoding="utf-8"))
        assert entry_scaled_report["summary"]["entry_scale_override_count"] == 1
        assert (
            entry_scaled_report["sidecar_manifest"]["artifacts"][0]["entries"][0]["correction_scale"]
            == 0.25
        )
        assert_raises_contains(
            lambda: hqq_self_calibrate_main(
                [
                    "--config",
                    str(config_path),
                    "--write-sidecar-artifact",
                    str(sidecar_sim_path),
                    "--output",
                    str(calib_dir / "sidecar_write_entry_scale_without_include_test.json"),
                    "--variant-name",
                    "toy_sidecar_bad_entry_scale",
                    "--sidecar-artifact-mode",
                    "int8_symmetric",
                    "--sidecar-entry-scales",
                    "0:in_proj_qkvz:0:0=0.25",
                ]
            ),
            "Sidecar entry scales require --sidecar-include-entries",
        )
        assert_raises_contains(
            lambda: hqq_self_calibrate_main(
                [
                    "--config",
                    str(config_path),
                    "--write-sidecar-artifact",
                    str(sidecar_sim_path),
                    "--output",
                    str(calib_dir / "sidecar_write_bad_scale_test.json"),
                    "--variant-name",
                    "toy_sidecar_bad_scale",
                    "--sidecar-artifact-mode",
                    "int8_symmetric",
                    "--sidecar-correction-scale",
                    "0",
                ]
            ),
            "Sidecar correction scale must be finite",
        )
        assert_raises_contains(
            lambda: hqq_self_calibrate_main(
                [
                    "--config",
                    str(config_path),
                    "--write-sidecar-artifact",
                    str(sidecar_sim_path),
                    "--output",
                    str(calib_dir / "sidecar_write_excluded_test.json"),
                    "--variant-name",
                    "toy_sidecar_excluded",
                    "--sidecar-artifact-mode",
                    "int8_symmetric",
                    "--sidecar-exclude-entries",
                    "0:in_proj_qkvz:0:0",
                ]
            ),
            "Sidecar simulation did not contain any writable sidecar rows",
        )

        assert file_sha256(baseline_manifest_path) == baseline_manifest_hash
        assert file_sha256(baseline_artifact_path) == baseline_artifact_hash

        report = json.loads(sidecar_write_path.read_text(encoding="utf-8"))
        assert report["format"] == "krasis_hqq_selfcal_sidecar_write"
        assert report["complete"] is True
        assert report["loadable_hqq_manifest"] is False
        assert report["sidecar_mode"] == "int8_symmetric"
        assert report["summary"]["row_group_count"] >= 1
        assert report["summary"]["sidecar_total_bytes"] > 0
        assert report["cache"]["hqq_target_manifest_written"] is False
        sidecar_manifest_path = Path(report["cache"]["sidecar_manifest_path"])
        assert sidecar_manifest_path.is_file()
        assert sidecar_manifest_path != Path(
            hqq_attention_manifest_path(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1)
        )

        sidecar_manifest = json.loads(sidecar_manifest_path.read_text(encoding="utf-8"))
        assert sidecar_manifest["format"] == "krasis_hqq_selfcal_sidecar_manifest"
        assert sidecar_manifest["loadable_hqq_manifest"] is False
        assert sidecar_manifest["runtime_application_implemented"] is False
        validated_sidecar = require_complete_hqq_sidecar_manifest(
            str(sidecar_manifest_path),
            model_path=str(model_path),
            source_cache_profile=HQQ_CACHE_PROFILE_BASELINE,
        )
        assert validated_sidecar["variant_name"] == "toy_sidecar"
        assert_raises_contains(
            lambda: require_complete_hqq_sidecar_manifest(
                str(sidecar_manifest_path),
                model_path=str(model_path),
                source_cache_profile=HQQ_CACHE_PROFILE_SELFCAL_V1,
            ),
            "source cache profile mismatch",
        )
        sidecar_file = sidecar_manifest_path.parent / sidecar_manifest["artifacts"][0]["file"]
        with safe_open(str(sidecar_file), framework="pt", device="cpu") as handle:
            assert "correction_qint8" in handle.keys()
            assert "output_rows" in handle.keys()
            assert "groups" in handle.keys()
        sidecar_file.unlink()
        assert_raises_contains(
            lambda: require_complete_hqq_sidecar_manifest(
                str(sidecar_manifest_path),
                model_path=str(model_path),
                source_cache_profile=HQQ_CACHE_PROFILE_BASELINE,
            ),
            "Missing HQQ sidecar artifact file",
        )

        calibrated_manifest = Path(hqq_attention_manifest_path(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1))
        assert not calibrated_manifest.exists()
        assert_raises_contains(
            lambda: require_complete_hqq_attention_manifest(
                str(model_path),
                HQQ_CACHE_PROFILE_SELFCAL_V1,
                expected_nbits=4,
                expected_num_hidden_layers=40,
            ),
            "No fallback to baseline is allowed",
        )


def test_experimental_cache_writer_creates_loadable_manifest_without_touching_baseline() -> None:
    with isolated_home() as home:
        model_path = home / "models" / "TinyModel"
        model_path.mkdir(parents=True)
        config_path = home / "tiny.conf"
        trace_path = home / "trace.log"
        calib_dir = Path(hqq_attention_cache_dir(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1)) / "calibration"
        evidence_path = calib_dir / "evidence_test.json"
        candidate_path = calib_dir / "sensitivity_candidates_test.json"
        eval_path = calib_dir / "refit_evaluation_test.json"
        write_path = calib_dir / "experimental_cache_write_test.json"

        write_baseline_manifest(model_path)
        write_config(config_path, model_path)
        write_trace(trace_path)
        baseline_manifest_path = Path(hqq_attention_manifest_path(str(model_path), HQQ_CACHE_PROFILE_BASELINE))
        baseline_artifact_path = baseline_manifest_path.parent / "layer_000_in_proj_qkvz_hqq4.safetensors"
        baseline_manifest_hash = file_sha256(baseline_manifest_path)
        baseline_artifact_hash = file_sha256(baseline_artifact_path)

        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--trace-log",
                str(trace_path),
                "--output",
                str(evidence_path),
                "--duration-seconds",
                "1",
            ]
        ) == 0
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--build-candidates",
                str(evidence_path),
                "--output",
                str(candidate_path),
                "--top-candidates",
                "4",
            ]
        ) == 0
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--evaluate-refits",
                str(candidate_path),
                "--trace-log",
                str(trace_path),
                "--output",
                str(eval_path),
                "--top-candidates",
                "1",
                "--max-output-rows",
                "1",
                "--refit-grid-steps",
                "17",
                "--refit-local-grid-steps",
                "9",
            ]
        ) == 0
        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--write-experimental-cache",
                str(eval_path),
                "--output",
                str(write_path),
                "--variant-name",
                "toy_top1",
                "--patch-top-groups",
                "1",
            ]
        ) == 0

        assert file_sha256(baseline_manifest_path) == baseline_manifest_hash
        assert file_sha256(baseline_artifact_path) == baseline_artifact_hash

        report = json.loads(write_path.read_text(encoding="utf-8"))
        assert report["format"] == "krasis_hqq_selfcal_experimental_cache_write"
        assert report["complete"] is True
        assert report["loadable_hqq_manifest"] is True
        assert report["summary"]["variant_name"] == "toy_top1"
        assert report["summary"]["patched_tensor_count"] == 1
        assert report["summary"]["patched_row_group_count"] >= 1
        assert report["ablation_selection"]["top_groups"] == 1

        calibrated_manifest = require_complete_hqq_attention_manifest(
            str(model_path),
            HQQ_CACHE_PROFILE_SELFCAL_V1,
            expected_nbits=4,
            expected_num_hidden_layers=40,
        )
        assert calibrated_manifest["cache_profile"] == "selfcal_v1"
        assert calibrated_manifest["calibration"]["status"] == "experimental"
        assert calibrated_manifest["calibration"]["source_baseline_manifest_sha256"] == baseline_manifest_hash
        entry = calibrated_manifest["tensors"][0]
        artifact = load_hqq_attention_artifact(
            str(model_path),
            entry,
            expected_nbits=4,
            cache_profile=HQQ_CACHE_PROFILE_SELFCAL_V1,
        )
        assert artifact["tensor_bytes"] == entry["tensor_bytes"]
        assert "calibration_patch" in entry

        Path(artifact["path"]).unlink()
        assert_raises_contains(
            lambda: load_hqq_attention_artifact(
                str(model_path),
                entry,
                expected_nbits=4,
                cache_profile=HQQ_CACHE_PROFILE_SELFCAL_V1,
            ),
            "Missing HQQ artifact file",
        )


def test_ablation_report_applies_strict_dominance() -> None:
    with isolated_home() as home:
        model_path = home / "models" / "TinyModel"
        model_path.mkdir(parents=True)
        config_path = home / "tiny.conf"
        calib_dir = Path(hqq_attention_cache_dir(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1)) / "calibration"
        baseline_summary = home / "baseline_summary.json"
        better_summary = home / "better_summary.json"
        mixed_summary = home / "mixed_summary.json"
        write_report = calib_dir / "experimental_cache_write_test.json"
        spec_path = home / "ablation_spec.json"
        report_path = calib_dir / "ablation_report_test.json"

        write_baseline_manifest(model_path)
        write_config(config_path, model_path)
        write_witness_summary(baseline_summary, [0.5, 0.5])
        write_witness_summary(better_summary, [0.4, 0.4])
        write_witness_summary(mixed_summary, [0.3, 0.6])
        write_report.parent.mkdir(parents=True, exist_ok=True)
        write_report.write_text(
            json.dumps(
                {
                    "format": "krasis_hqq_selfcal_experimental_cache_write",
                    "summary": {
                        "patched_tensor_count": 1,
                        "patched_row_group_count": 2,
                    },
                }
            ),
            encoding="utf-8",
        )
        spec_path.write_text(
            json.dumps(
                {
                    "baseline": {"name": "baseline", "summary_path": str(baseline_summary)},
                    "variants": [
                        {
                            "name": "better",
                            "summary_path": str(better_summary),
                            "write_report_path": str(write_report),
                        },
                        {
                            "name": "mixed",
                            "summary_path": str(mixed_summary),
                            "write_report_path": str(write_report),
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--summarize-ablation",
                str(spec_path),
                "--output",
                str(report_path),
            ]
        ) == 0
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["format"] == "krasis_hqq_selfcal_ablation_report"
        assert payload["accepted_variant_count"] == 1
        by_name = {item["variant_name"]: item for item in payload["variants"]}
        assert by_name["better"]["accepted"] is True
        assert by_name["mixed"]["accepted"] is False
        assert by_name["mixed"]["per_case_regression_count"] == 1
        assert (report_path.parent / "ablation_report_test.tsv").is_file()


def test_sidecar_conflict_search_prefers_zero_regression_then_gain() -> None:
    with isolated_home() as home:
        model_path = home / "models" / "TinyModel"
        model_path.mkdir(parents=True)
        config_path = home / "tiny.conf"
        work = home / "conflict_search"
        work.mkdir()
        singleton_path = work / "singletons.json"
        spec_path = work / "spec.json"
        report_path = work / "conflict_search.json"

        write_baseline_manifest(model_path)
        write_config(config_path, model_path)

        def variant(name, row, effects):
            return {
                "variant": name,
                "correction_scale": 0.75,
                "entries": [
                    {
                        "layer": 0,
                        "tensor": "in_proj_qkvz",
                        "group": 0,
                        "output_row": row,
                        "start_col": 0,
                        "end_col": 4,
                    }
                ],
                "per_case": [
                    {
                        "conv_idx": conv,
                        "turn": 1,
                        "change_vs_baseline": change,
                        "first_token_match": True,
                    }
                    for conv, change in effects.items()
                ],
                "aggregate_delta_vs_baseline": sum(effects.values()),
                "per_case_regression_count": sum(1 for value in effects.values() if value > 0),
            }

        singleton_path.write_text(
            json.dumps(
                {
                    "format": "manual_test_singleton_report",
                    "baseline": {
                        "summary_path": "baseline.json",
                        "selected_logprob_delta_sum": 1.0,
                        "first_token_matches": 3,
                        "prompt_count": 3,
                    },
                    "variants": [
                        variant("unit_a", 0, {0: -0.10, 1: -0.05, 8: 0.02}),
                        variant("unit_b", 1, {0: -0.02, 1: 0.0, 8: -0.03}),
                        variant("unit_c", 2, {0: -0.20, 1: 0.04, 8: 0.04}),
                    ],
                }
            ),
            encoding="utf-8",
        )
        spec_path.write_text(
            json.dumps(
                {
                    "reports": {"singleton075": str(singleton_path)},
                    "min_units": 2,
                    "max_units": 2,
                    "max_row_groups": 4,
                    "max_candidates": 2,
                }
            ),
            encoding="utf-8",
        )

        assert hqq_self_calibrate_main(
            [
                "--config",
                str(config_path),
                "--search-sidecar-conflicts",
                str(spec_path),
                "--output",
                str(report_path),
            ]
        ) == 0
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["format"] == "krasis_hqq_selfcal_sidecar_conflict_search"
        assert payload["search"]["selected_candidate_count"] == 2
        best = payload["candidates"][0]
        assert best["predicted_accepted_strict"] is True
        assert best["predicted_per_case_regression_count"] == 0
        assert best["unit_names"] == ["unit_a", "unit_b"]
        assert "0:in_proj_qkvz:0:0" in best["include_entries_arg"]
        assert (report_path.parent / "conflict_search.tsv").is_file()


if __name__ == "__main__":
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("ERROR: run via ./dev hqq-self-calibrate-test")
    test_int8_exception_budget_selection_uses_ratio_and_top_limit()
    test_int8_exception_budget_rejects_explicit_groups_over_cap()
    test_int8_exception_split_budget_prefers_decode_and_excludes_inactive()
    test_evidence_file_is_not_loadable_calibrated_manifest()
    test_sensitivity_candidates_are_not_loadable_calibrated_manifest()
    test_int8_exception_candidates_are_intrinsic_and_non_loadable()
    test_int8_exception_candidate_build_rejects_shared_train_heldout_trace()
    test_evidence_trace_log_path_is_unique_per_output()
    test_refit_evaluation_is_not_loadable_calibrated_manifest()
    test_sidecar_simulation_is_non_loadable_and_reports_estimates()
    test_sidecar_artifact_writer_stays_separate_from_hqq_manifest()
    test_experimental_cache_writer_creates_loadable_manifest_without_touching_baseline()
    test_ablation_report_applies_strict_dominance()
    test_sidecar_conflict_search_prefers_zero_regression_then_gain()
    print("ok")
