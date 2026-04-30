import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch

from krasis.attention_backend import (
    HQQ_ATTENTION_CACHE_DIRNAME,
    HQQ_ATTENTION_CACHE_VERSION,
    HQQ46_ATTENTION_CACHE_DIRNAME,
    HQQ46_AUTO_ATTENTION_CACHE_DIRNAME,
    HQQ46_LAYOUT,
    HQQ68_AUTO_ATTENTION_CACHE_DIRNAME,
    HQQ68_LAYOUT,
    HQQ_MIXED_46_AUTO_CACHE_NBITS,
    HQQ_MIXED_46_CACHE_NBITS,
    HQQ_MIXED_68_AUTO_CACHE_NBITS,
    HQQ_DEFAULT_AXIS,
    HQQ_DEFAULT_GROUP_SIZE,
    HQQ_LAYOUT,
    _hqq_padded_cols,
    _hqq_packed_cols_for_nbits,
    _pack_hqq_quant,
    _unpack_hqq_quant,
    hqq_attention_cache_dir,
    hqq_attention_manifest_path,
    normalize_hqq_attention_cache_profile,
    hqq_auto_budget_bytes_from_pct,
    quantize_hqq6_tensor_probe,
    require_complete_hqq_attention_manifest,
    select_hqq_auto_promotions,
)
from krasis.config import (
    HQQ_CACHE_PROFILE_BASELINE,
    HQQ_CACHE_PROFILE_SELFCAL_V1,
    QuantConfig,
)


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


def minimal_manifest(complete=True, profile=HQQ_CACHE_PROFILE_BASELINE):
    manifest = {
        "format_version": HQQ_ATTENTION_CACHE_VERSION,
        "backend": "hqq4",
        "nbits": 4,
        "group_size": HQQ_DEFAULT_GROUP_SIZE,
        "axis": HQQ_DEFAULT_AXIS,
        "layout": HQQ_LAYOUT,
        "num_hidden_layers": 2,
        "complete": complete,
        "tensors": [],
        "totals": {
            "tensor_bytes": 0,
            "num_tensors": 0,
        },
    }
    if profile != HQQ_CACHE_PROFILE_BASELINE:
        manifest["cache_profile"] = profile
        manifest["calibration"] = {
            "profile": profile,
            "method": "selfcal_v1",
        }
    return manifest


def write_manifest(model_path: Path, profile: str, manifest: dict, nbits: int = 4) -> Path:
    path = Path(hqq_attention_manifest_path(str(model_path), profile, nbits=nbits))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def assert_raises_contains(fn, text: str) -> None:
    try:
        fn()
    except Exception as exc:
        message = str(exc)
        if text not in message:
            raise AssertionError(f"Expected {text!r} in {message!r}") from exc
        return
    raise AssertionError(f"Expected exception containing {text!r}")


def test_profile_path_resolution() -> None:
    with isolated_home() as home:
        model_path = home / "models" / "TinyModel"
        baseline_dir = Path(hqq_attention_cache_dir(str(model_path)))
        selfcal_dir = Path(hqq_attention_cache_dir(str(model_path), HQQ_CACHE_PROFILE_SELFCAL_V1))
        g32_dir = Path(hqq_attention_cache_dir(str(model_path), group_size=32))
        mixed_dir = Path(hqq_attention_cache_dir(str(model_path), nbits=HQQ_MIXED_46_CACHE_NBITS))
        auto_dir = Path(hqq_attention_cache_dir(str(model_path), nbits=HQQ_MIXED_46_AUTO_CACHE_NBITS))
        auto68_dir = Path(hqq_attention_cache_dir(str(model_path), nbits=HQQ_MIXED_68_AUTO_CACHE_NBITS))
        assert baseline_dir == home / ".krasis" / "cache" / "TinyModel" / HQQ_ATTENTION_CACHE_DIRNAME
        assert selfcal_dir == (
            home / ".krasis" / "cache" / "TinyModel" / f"{HQQ_ATTENTION_CACHE_DIRNAME}_calib_selfcal_v1"
        )
        assert g32_dir == home / ".krasis" / "cache" / "TinyModel" / f"{HQQ_ATTENTION_CACHE_DIRNAME}_g32"
        assert mixed_dir == home / ".krasis" / "cache" / "TinyModel" / HQQ46_ATTENTION_CACHE_DIRNAME
        assert auto_dir == home / ".krasis" / "cache" / "TinyModel" / HQQ46_AUTO_ATTENTION_CACHE_DIRNAME
        assert auto68_dir == home / ".krasis" / "cache" / "TinyModel" / HQQ68_AUTO_ATTENTION_CACHE_DIRNAME
        assert normalize_hqq_attention_cache_profile(None) == HQQ_CACHE_PROFILE_BASELINE
        assert normalize_hqq_attention_cache_profile("SELFCAL_V1") == HQQ_CACHE_PROFILE_SELFCAL_V1


def test_quant_config_profile_validation() -> None:
    assert QuantConfig().hqq_cache_profile == HQQ_CACHE_PROFILE_BASELINE
    assert QuantConfig(attention="hqq4", hqq_group_size=32).hqq_group_size == 32
    assert QuantConfig(attention="hqq46", hqq_group_size=128).attention == "hqq46"
    assert QuantConfig(attention="hqq46_auto", hqq_auto_budget_pct=35.5).attention == "hqq46_auto"
    assert QuantConfig(attention="hqq68_auto", hqq_auto_budget_pct=50.0).attention == "hqq68_auto"
    assert QuantConfig(attention="hqq4", hqq_cache_profile="selfcal_v1").hqq_cache_profile == "selfcal_v1"
    assert QuantConfig(attention="hqq6", hqq_cache_profile="selfcal_v1").hqq_cache_profile == "selfcal_v1"
    assert QuantConfig(attention="hqq8", hqq_cache_profile="selfcal_v1").hqq_cache_profile == "selfcal_v1"
    assert QuantConfig(attention="hqq4", hqq_sidecar_manifest="~/sidecar.json").hqq_sidecar_manifest.endswith(
        "sidecar.json"
    )
    assert_raises_contains(
        lambda: QuantConfig(attention="bf16", hqq_cache_profile="selfcal_v1"),
        "requires attention='hqq4'",
    )
    assert_raises_contains(
        lambda: QuantConfig(attention="bf16", hqq_sidecar_manifest="/tmp/sidecar.json"),
        "hqq_sidecar_manifest requires attention='hqq4'",
    )
    assert_raises_contains(
        lambda: QuantConfig(attention="hqq8", hqq_sidecar_manifest="/tmp/sidecar.json"),
        "HQQ4/6, HQQ6, HQQ6/8, and HQQ8 are clean higher-precision attention modes",
    )
    assert_raises_contains(
        lambda: QuantConfig(attention="hqq46_auto"),
        "requires hqq_auto_budget_pct",
    )
    assert_raises_contains(
        lambda: QuantConfig(attention="hqq6", hqq_auto_budget_pct=50.0),
        "hqq_auto_budget_pct is only valid",
    )
    assert_raises_contains(
        lambda: QuantConfig(attention="hqq68_auto"),
        "requires hqq_auto_budget_pct",
    )
    assert_raises_contains(
        lambda: QuantConfig(attention="hqq68_auto", hqq_auto_budget_pct=50.0, hqq46_auto_budget_mib=150),
        "hqq46_auto_budget_mib is not valid",
    )
    assert_raises_contains(
        lambda: QuantConfig(attention="hqq4", hqq_group_size=32, hqq_sidecar_manifest="/tmp/sidecar.json"),
        "requires hqq_group_size=128",
    )
    assert_raises_contains(
        lambda: QuantConfig(attention="hqq4", hqq_cache_profile="unknown"),
        "Unsupported hqq_cache_profile",
    )
    assert_raises_contains(
        lambda: QuantConfig(attention="hqq4", hqq_group_size=16),
        "Unsupported hqq_group_size",
    )
    assert_raises_contains(
        lambda: QuantConfig(attention="bf16", hqq_group_size=32),
        "requires attention='hqq4'",
    )


def test_selected_calibrated_profile_requires_complete_manifest_without_fallback() -> None:
    with isolated_home() as home:
        model_path = home / "models" / "TinyModel"
        assert_raises_contains(
            lambda: require_complete_hqq_attention_manifest(
                str(model_path),
                HQQ_CACHE_PROFILE_SELFCAL_V1,
                expected_nbits=4,
                expected_num_hidden_layers=2,
            ),
            "No fallback to baseline is allowed",
        )

        write_manifest(model_path, HQQ_CACHE_PROFILE_SELFCAL_V1, minimal_manifest(complete=False, profile="selfcal_v1"))
        assert_raises_contains(
            lambda: require_complete_hqq_attention_manifest(
                str(model_path),
                HQQ_CACHE_PROFILE_SELFCAL_V1,
                expected_nbits=4,
                expected_num_hidden_layers=2,
            ),
            "complete=false",
        )

        write_manifest(model_path, HQQ_CACHE_PROFILE_SELFCAL_V1, minimal_manifest(complete=True, profile="baseline"))
        assert_raises_contains(
            lambda: require_complete_hqq_attention_manifest(
                str(model_path),
                HQQ_CACHE_PROFILE_SELFCAL_V1,
                expected_nbits=4,
                expected_num_hidden_layers=2,
            ),
            "cache_profile=None",
        )

        write_manifest(model_path, HQQ_CACHE_PROFILE_SELFCAL_V1, minimal_manifest(complete=True, profile="selfcal_v1"))
        manifest = require_complete_hqq_attention_manifest(
            str(model_path),
            HQQ_CACHE_PROFILE_SELFCAL_V1,
            expected_nbits=4,
            expected_num_hidden_layers=2,
        )
        assert manifest["cache_profile"] == "selfcal_v1"


def test_baseline_manifest_loading_remains_unchanged() -> None:
    with isolated_home() as home:
        model_path = home / "models" / "TinyModel"
        write_manifest(model_path, HQQ_CACHE_PROFILE_BASELINE, minimal_manifest(complete=True))
        manifest = require_complete_hqq_attention_manifest(
            str(model_path),
            HQQ_CACHE_PROFILE_BASELINE,
            expected_nbits=4,
            expected_num_hidden_layers=2,
        )
        assert manifest["complete"] is True
        assert "cache_profile" not in manifest


def test_hqq46_manifest_validation_accepts_mixed_cache_metadata() -> None:
    with isolated_home() as home:
        model_path = home / "models" / "TinyModel"
        manifest = minimal_manifest(complete=True)
        manifest["backend"] = "hqq46"
        manifest["nbits"] = HQQ_MIXED_46_CACHE_NBITS
        manifest["layout"] = HQQ46_LAYOUT
        manifest["mixed_precision"] = {
            "base_nbits": 4,
            "promoted_nbits": 6,
            "policy": "output_fused_v1",
            "promoted_tensors": ["fused_qkv", "o_proj", "out_proj"],
        }
        write_manifest(model_path, HQQ_CACHE_PROFILE_BASELINE, manifest, nbits=HQQ_MIXED_46_CACHE_NBITS)
        loaded = require_complete_hqq_attention_manifest(
            str(model_path),
            HQQ_CACHE_PROFILE_BASELINE,
            expected_nbits=HQQ_MIXED_46_CACHE_NBITS,
            expected_num_hidden_layers=2,
        )
        assert loaded["backend"] == "hqq46"
        assert loaded["layout"] == HQQ46_LAYOUT


def test_hqq46_auto_planner_selects_by_global_budget() -> None:
    candidates = [
        {
            "layer_idx": 0,
            "tensor_name": "large",
            "extra_bytes": 10,
            "relative_rmse_reduction": 9.0,
        },
        {
            "layer_idx": 0,
            "tensor_name": "small_a",
            "extra_bytes": 6,
            "relative_rmse_reduction": 6.0,
        },
        {
            "layer_idx": 1,
            "tensor_name": "small_b",
            "extra_bytes": 6,
            "relative_rmse_reduction": 6.0,
        },
    ]
    selected, summary = select_hqq_auto_promotions(candidates, budget_bytes=12, max_dp_units=12)
    assert selected == {(0, "small_a"), (1, "small_b")}
    assert summary["budget_used_bytes"] == 12
    assert summary["selected_count"] == 2


def test_hqq_auto_budget_pct_scales_from_promotion_span() -> None:
    assert hqq_auto_budget_bytes_from_pct(25.0, 400, "hqq68_auto") == 100
    assert hqq_auto_budget_bytes_from_pct(100.0, 400, "hqq68_auto") == 400
    assert_raises_contains(
        lambda: hqq_auto_budget_bytes_from_pct(0.0, 400, "hqq68_auto"),
        "0 < hqq_auto_budget_pct <= 100",
    )


def test_hqq_auto_full_budget_selects_every_promotion() -> None:
    candidates = [
        {
            "layer_idx": 0,
            "tensor_name": "high_gain",
            "extra_bytes": 10,
            "relative_rmse_reduction": 5.0,
            "base_nbits": 4,
            "promoted_nbits": 6,
        },
        {
            "layer_idx": 1,
            "tensor_name": "zero_gain",
            "extra_bytes": 7,
            "relative_rmse_reduction": 0.0,
            "base_nbits": 4,
            "promoted_nbits": 6,
        },
        {
            "layer_idx": 2,
            "tensor_name": "hqq68",
            "extra_bytes": 3,
            "relative_rmse_reduction": 0.5,
            "base_nbits": 6,
            "promoted_nbits": 8,
        },
    ]
    selected, summary = select_hqq_auto_promotions(candidates, budget_bytes=20, max_dp_units=2)
    assert selected == {(0, "high_gain"), (1, "zero_gain"), (2, "hqq68")}
    assert summary["selection_mode"] == "full_span"
    assert summary["budget_used_bytes"] == 20
    assert summary["selected_count"] == 3


def test_hqq6_packed_roundtrip_contract() -> None:
    quant = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.uint8)
    packed = _pack_hqq_quant(quant, 6)
    assert packed.shape == (1, 6)
    assert packed.tolist() == [[64, 32, 12, 68, 97, 28]]
    unpacked = _unpack_hqq_quant(packed, quant.shape[1], 6)
    assert torch.equal(unpacked, quant)
    assert _hqq_packed_cols_for_nbits(cols=5, group_size=4, nbits=6) == 6


def test_hqq6_probe_quantizer_uses_true_packed_storage() -> None:
    weight = torch.tensor(
        [
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            [0.125, -0.25, 0.375, -0.5, 0.625],
        ],
        dtype=torch.float32,
    )
    tensors = quantize_hqq6_tensor_probe(weight, group_size=4, collect_stats=True)
    rows, cols = weight.shape
    padded_cols = _hqq_padded_cols(cols, 4)
    assert int(tensors["nbits"][0].item()) == 6
    assert tensors["packed"].shape == (rows, 6)
    assert tensors["scales"].shape == (rows, 2)

    quant = _unpack_hqq_quant(tensors["packed"], padded_cols, 6)[:, :cols]
    reconstructed = torch.empty_like(weight)
    for group_idx in range(2):
        start = group_idx * 4
        end = min(start + 4, cols)
        reconstructed[:, start:end] = (
            quant[:, start:end].to(torch.float32) - tensors["zeros"][:, group_idx].unsqueeze(1)
        ) * tensors["scales"][:, group_idx].unsqueeze(1)
    rmse = torch.sqrt(torch.mean((reconstructed - weight) ** 2)).item()
    assert abs(rmse - tensors["stats"]["rmse"]) < 1e-7
    assert tensors["packed"].numel() < padded_cols * rows


if __name__ == "__main__":
    test_profile_path_resolution()
    test_quant_config_profile_validation()
    test_selected_calibrated_profile_requires_complete_manifest_without_fallback()
    test_baseline_manifest_loading_remains_unchanged()
    test_hqq46_manifest_validation_accepts_mixed_cache_metadata()
    test_hqq46_auto_planner_selects_by_global_budget()
    test_hqq_auto_budget_pct_scales_from_promotion_span()
    test_hqq_auto_full_budget_selects_every_promotion()
    test_hqq6_packed_roundtrip_contract()
    test_hqq6_probe_quantizer_uses_true_packed_storage()
    print("ok")
