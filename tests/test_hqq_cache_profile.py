import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from krasis.attention_backend import (
    HQQ_ATTENTION_CACHE_DIRNAME,
    HQQ_ATTENTION_CACHE_VERSION,
    HQQ_DEFAULT_AXIS,
    HQQ_DEFAULT_GROUP_SIZE,
    HQQ_LAYOUT,
    hqq_attention_cache_dir,
    hqq_attention_manifest_path,
    normalize_hqq_attention_cache_profile,
    require_complete_hqq_attention_manifest,
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


def write_manifest(model_path: Path, profile: str, manifest: dict) -> Path:
    path = Path(hqq_attention_manifest_path(str(model_path), profile))
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
        assert baseline_dir == home / ".krasis" / "cache" / "TinyModel" / HQQ_ATTENTION_CACHE_DIRNAME
        assert selfcal_dir == (
            home / ".krasis" / "cache" / "TinyModel" / f"{HQQ_ATTENTION_CACHE_DIRNAME}_calib_selfcal_v1"
        )
        assert normalize_hqq_attention_cache_profile(None) == HQQ_CACHE_PROFILE_BASELINE
        assert normalize_hqq_attention_cache_profile("SELFCAL_V1") == HQQ_CACHE_PROFILE_SELFCAL_V1


def test_quant_config_profile_validation() -> None:
    assert QuantConfig().hqq_cache_profile == HQQ_CACHE_PROFILE_BASELINE
    assert QuantConfig(attention="hqq4", hqq_cache_profile="selfcal_v1").hqq_cache_profile == "selfcal_v1"
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
        "HQQ8 is a clean higher-precision attention mode",
    )
    assert_raises_contains(
        lambda: QuantConfig(attention="hqq4", hqq_cache_profile="unknown"),
        "Unsupported hqq_cache_profile",
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


if __name__ == "__main__":
    test_profile_path_resolution()
    test_quant_config_profile_validation()
    test_selected_calibrated_profile_requires_complete_manifest_without_fallback()
    test_baseline_manifest_loading_remains_unchanged()
    print("ok")
