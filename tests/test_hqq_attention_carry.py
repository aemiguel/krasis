#!/usr/bin/env python3
"""Verify HQQ attention cache carry preserves fused_qkv only when present."""

import os
import shutil
import tempfile
from types import MethodType, SimpleNamespace

import torch

from krasis.attention_backend import load_hqq_attention_manifest
from krasis.model import KrasisModel


def make_tensor(rows: int, cols: int, start: int) -> torch.Tensor:
    values = torch.arange(start, start + rows * cols, dtype=torch.float32)
    return values.reshape(rows, cols) / 97.0


def build_minimal_model(model_path: str, weights: dict) -> KrasisModel:
    model = KrasisModel.__new__(KrasisModel)
    model.cfg = SimpleNamespace(
        is_mla=False,
        model_path=model_path,
        num_hidden_layers=1,
    )
    model.quant_cfg = SimpleNamespace(attention="hqq4")
    model.layers = [SimpleNamespace(device=torch.device("cpu"))]
    model._hqq_attention_runtime = {}
    model._hqq_attention_runtime_nbits = None
    model._hqq_attention_cache_bytes = 0
    model._hqq_manifest = None
    model._hqq_rebuild = False

    def extract_layer_weights(self, layer, device):
        return {
            "layer_type": "full_attention",
            "attention": weights,
        }

    model._extract_layer_weights = MethodType(extract_layer_weights, model)
    return model


def run_case(case_name: str, include_fused_qkv: bool) -> None:
    tmpdir = tempfile.mkdtemp(prefix=f"hqq-attn-carry-{case_name}-")
    old_home = os.environ.get("HOME")
    try:
        os.environ["HOME"] = tmpdir
        model_path = os.path.join(tmpdir, "model")
        os.makedirs(model_path, exist_ok=True)

        weights = {
            "q_proj": make_tensor(8, 16, 0),
            "k_proj": make_tensor(4, 16, 1000),
            "v_proj": make_tensor(4, 16, 2000),
            "o_proj": make_tensor(16, 8, 3000),
        }
        if include_fused_qkv:
            weights["fused_qkv"] = make_tensor(16, 16, 4000)

        model = build_minimal_model(model_path, weights)

        tensor_map = model._hqq_attention_tensor_map("full_attention", weights)
        expected_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if include_fused_qkv:
            expected_names.append("fused_qkv")
        assert list(tensor_map.keys()) == expected_names, (
            f"{case_name}: tensor map mismatch: got {list(tensor_map.keys())}, "
            f"expected {expected_names}"
        )

        model._prepare_hqq_attention_cache()
        model._maybe_write_hqq_attention_artifacts(0, "full_attention", weights)
        model._validate_hqq_attention_cache()
        model._load_hqq_attention_runtime_state()

        manifest = load_hqq_attention_manifest(model_path)
        assert manifest is not None, f"{case_name}: manifest missing"
        manifest_names = [
            entry["tensor_name"]
            for entry in manifest["tensors"]
            if entry["layer_idx"] == 0
        ]
        assert manifest_names == expected_names, (
            f"{case_name}: manifest names mismatch: got {manifest_names}, "
            f"expected {expected_names}"
        )

        runtime_names = list(model._hqq_attention_runtime[0].keys())
        assert runtime_names == expected_names, (
            f"{case_name}: runtime names mismatch: got {runtime_names}, "
            f"expected {expected_names}"
        )

        has_fused_manifest = "fused_qkv" in manifest_names
        has_fused_runtime = "fused_qkv" in runtime_names
        assert has_fused_manifest == include_fused_qkv, (
            f"{case_name}: fused_qkv manifest presence mismatch"
        )
        assert has_fused_runtime == include_fused_qkv, (
            f"{case_name}: fused_qkv runtime presence mismatch"
        )
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> None:
    run_case("without-fused", include_fused_qkv=False)
    run_case("with-fused", include_fused_qkv=True)
    print("PASS: HQQ attention cache carry preserves fused_qkv presence exactly")


if __name__ == "__main__":
    main()
