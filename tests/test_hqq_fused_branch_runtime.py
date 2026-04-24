#!/usr/bin/env python3
"""Verify split/fused HQQ single-token GQA decode branch behavior."""

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from types import MethodType, SimpleNamespace

import torch

from krasis import GpuDecodeStore
from krasis.attention_backend import (
    HQQ_ATTENTION_CACHE_VERSION,
    HQQ_LAYOUT,
    load_hqq_attention_manifest,
    quantize_hqq4_tensor,
    save_hqq_attention_manifest,
)
from krasis.config import QuantConfig
from krasis.layer import TransformerLayer
from krasis.model import KrasisModel
from krasis.weight_loader import WeightLoader


DEFAULT_REAL_MODEL_PATH = os.path.expanduser("~/.krasis/models/Qwen3-0.6B")
_REAL_MODEL_CASES: dict[tuple[str, int], dict] = {}


def dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _bf16_words_to_f32(words: list[int]) -> torch.Tensor:
    return torch.tensor(words, dtype=torch.uint16).view(torch.bfloat16).float()


def _similarity_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference = reference.reshape(-1).float().cpu()
    candidate = candidate.reshape(-1).float().cpu()
    if reference.shape != candidate.shape:
        raise AssertionError(
            f"similarity shape mismatch: reference={tuple(reference.shape)} candidate={tuple(candidate.shape)}"
        )
    diff = candidate - reference
    ref_norm = float(reference.norm().item())
    cand_norm = float(candidate.norm().item())
    if ref_norm == 0.0 and cand_norm == 0.0:
        cosine = 1.0
    elif ref_norm == 0.0 or cand_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(torch.nn.functional.cosine_similarity(reference, candidate, dim=0).item())
    return {
        "numel": int(reference.numel()),
        "cosine": cosine,
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rmse": float(diff.square().mean().sqrt().item()),
    }


def _diag_vector(diag: dict, key: str) -> torch.Tensor:
    return torch.tensor(diag[key], dtype=torch.float32)


def _real_model_bf16_projection_reference(real_case: dict, diag: dict) -> dict:
    pre_attn_hidden = _diag_vector(diag, "pre_attn_hidden")
    o_proj_input = _diag_vector(diag, "o_proj_input")
    return {
        "q_proj_raw": torch.mv(real_case["q_w"].float().cpu(), pre_attn_hidden),
        "k_proj_raw": torch.mv(real_case["k_w"].float().cpu(), pre_attn_hidden),
        "v_proj_raw": torch.mv(real_case["v_w"].float().cpu(), pre_attn_hidden),
        "o_proj_out": torch.mv(real_case["o_w"].float().cpu(), o_proj_input),
    }


def _projection_similarity_from_diag(real_case: dict, diag: dict) -> dict:
    reference = _real_model_bf16_projection_reference(real_case, diag)
    return {
        "q_proj_raw": _similarity_stats(reference["q_proj_raw"], _diag_vector(diag, "q_proj_raw")),
        "k_proj_raw": _similarity_stats(reference["k_proj_raw"], _diag_vector(diag, "k_proj_raw")),
        "v_proj_raw": _similarity_stats(reference["v_proj_raw"], _diag_vector(diag, "v_proj_raw")),
        "o_proj_out": _similarity_stats(reference["o_proj_out"], _diag_vector(diag, "o_proj_out")),
    }


def _unpack_uint4(packed: torch.Tensor, cols: int) -> torch.Tensor:
    packed = packed.to(torch.uint8).cpu()
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    quant = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)
    return quant[:, :cols].to(torch.float32)


def _dequantize_hqq_runtime_desc(desc: dict) -> torch.Tensor:
    rows, cols = (int(v) for v in desc["orig_shape"])
    group_size = int(desc["group_size"])
    groups = (cols + group_size - 1) // group_size
    quant = _unpack_uint4(desc["packed"], cols)
    scales = desc["scales"].float().cpu()
    zeros = desc["zeros"].float().cpu()
    dequant = torch.empty((rows, cols), dtype=torch.float32)
    for g in range(groups):
        start = g * group_size
        end = min(start + group_size, cols)
        dequant[:, start:end] = (
            quant[:, start:end] - zeros[:, g].unsqueeze(1)
        ) * scales[:, g].unsqueeze(1)
    return dequant


def _reference_probe_inputs(cols: int) -> torch.Tensor:
    positions = torch.arange(cols, dtype=torch.float32)
    return torch.stack(
        (
            torch.linspace(-1.0, 1.0, cols, dtype=torch.float32),
            torch.cos(positions / 17.0),
            torch.sin(positions / 29.0),
        ),
        dim=0,
    )


def _runtime_tensor_similarity(reference_weight: torch.Tensor, runtime_desc: dict) -> dict:
    reference_weight = reference_weight.float().cpu()
    dequant_weight = _dequantize_hqq_runtime_desc(runtime_desc)
    probe_inputs = _reference_probe_inputs(reference_weight.shape[1])
    reference_outputs = probe_inputs @ reference_weight.t()
    dequant_outputs = probe_inputs @ dequant_weight.t()
    return {
        "weight": _similarity_stats(reference_weight, dequant_weight),
        "probe_output": _similarity_stats(reference_outputs, dequant_outputs),
    }


def make_weight(rows: int, cols: int, *, scale: float, offset: float = 0.0, flip_cols: bool = False) -> torch.Tensor:
    weight = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    if flip_cols:
        weight = weight.flip(1)
    return (weight / scale + offset).contiguous()


def stage_hqq(store: GpuDecodeStore, layer_idx: int, tensor_name: str, weight: torch.Tensor) -> dict:
    quantized = quantize_hqq4_tensor(weight)
    packed = quantized["packed"].contiguous()
    scales = quantized["scales"].contiguous()
    zeros = quantized["zeros"].contiguous()
    store.stage_hqq_runtime_tensor_formats(
        layer_idx=layer_idx,
        tensor_name=tensor_name,
        backend="hqq4",
        nbits=4,
        format_version=HQQ_ATTENTION_CACHE_VERSION,
        packed_ptr=int(packed.data_ptr()),
        packed_bytes=int(packed.numel() * packed.element_size()),
        scales_ptr=int(scales.data_ptr()),
        scales_bytes=int(scales.numel() * scales.element_size()),
        zeros_ptr=int(zeros.data_ptr()),
        zeros_bytes=int(zeros.numel() * zeros.element_size()),
        rows=int(quantized["orig_shape"][0]),
        cols=int(quantized["orig_shape"][1]),
        group_size=int(quantized["group_size"][0]),
        axis=int(quantized["axis"][0]),
        layout=HQQ_LAYOUT,
        packed_dtype=dtype_name(packed),
        scales_dtype=dtype_name(scales),
        zeros_dtype=dtype_name(zeros),
    )
    return {"packed": packed, "scales": scales, "zeros": zeros}


def build_base_store(
    *,
    include_fused_config: bool,
):
    device = torch.device("cuda:0")
    hidden = 8
    vocab = 4
    heads = 1
    kv_heads = 1
    head_dim = 4
    kv_stride = kv_heads * head_dim
    max_seq = 4
    layer_idx = 0

    store = GpuDecodeStore(0)
    store.configure(hidden, 1, vocab, 1e-6)

    embedding = torch.arange(vocab * hidden, dtype=torch.bfloat16, device=device).reshape(vocab, hidden) / 32
    final_norm = torch.ones(hidden, dtype=torch.float32, device=device)
    lm_head = torch.eye(vocab, hidden, dtype=torch.bfloat16, device=device).contiguous()

    store.set_embedding(int(embedding.data_ptr()))
    store.set_final_norm(int(final_norm.data_ptr()), hidden)
    lm_head_wid = store.register_weight(int(lm_head.data_ptr()), lm_head.shape[0], lm_head.shape[1], 0)
    store.set_lm_head(lm_head_wid)

    cos = torch.ones(max_seq, head_dim // 2, dtype=torch.float32, device=device)
    sin = torch.zeros(max_seq, head_dim // 2, dtype=torch.float32, device=device)
    store.set_rope_tables(int(cos.data_ptr()), int(sin.data_ptr()), head_dim // 2, max_seq)

    kv_k = torch.zeros(max_seq, kv_stride, dtype=torch.bfloat16, device=device)
    kv_v = torch.zeros(max_seq, kv_stride, dtype=torch.bfloat16, device=device)

    q_w = make_weight(heads * head_dim, hidden, scale=97.0)
    k_w = make_weight(kv_stride, hidden, scale=89.0, flip_cols=True)
    v_w = make_weight(kv_stride, hidden, scale=83.0, offset=-0.1)
    o_w = make_weight(hidden, heads * head_dim, scale=71.0)
    fused_w = torch.cat([q_w, k_w, v_w], dim=0).contiguous()

    q_bf16 = q_w.to(device=device, dtype=torch.bfloat16)
    k_bf16 = k_w.to(device=device, dtype=torch.bfloat16)
    v_bf16 = v_w.to(device=device, dtype=torch.bfloat16)
    o_bf16 = o_w.to(device=device, dtype=torch.bfloat16)
    fused_bf16 = fused_w.to(device=device, dtype=torch.bfloat16)
    inp_norm = torch.ones(hidden, dtype=torch.float32, device=device)
    post_norm = torch.ones(hidden, dtype=torch.float32, device=device)

    q_wid = store.register_weight(int(q_bf16.data_ptr()), q_bf16.shape[0], q_bf16.shape[1], 0)
    k_wid = store.register_weight(int(k_bf16.data_ptr()), k_bf16.shape[0], k_bf16.shape[1], 0)
    v_wid = store.register_weight(int(v_bf16.data_ptr()), v_bf16.shape[0], v_bf16.shape[1], 0)
    o_wid = store.register_weight(int(o_bf16.data_ptr()), o_bf16.shape[0], o_bf16.shape[1], 0)
    fused_wid = store.register_weight(int(fused_bf16.data_ptr()), fused_bf16.shape[0], fused_bf16.shape[1], 0)

    store.register_gqa_layer(
        layer_idx=layer_idx,
        input_norm_ptr=int(inp_norm.data_ptr()),
        input_norm_size=hidden,
        post_attn_norm_ptr=int(post_norm.data_ptr()),
        post_attn_norm_size=hidden,
        q_proj_wid=q_wid,
        k_proj_wid=k_wid,
        v_proj_wid=v_wid,
        o_proj_wid=o_wid,
        fused_qkv_wid=(fused_wid if include_fused_config else None),
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        sm_scale=1.0 / math.sqrt(head_dim),
        q_norm_ptr=0,
        k_norm_ptr=0,
        gated=False,
    )
    store.set_kv_cache_ptrs_bf16([(layer_idx, int(kv_k.data_ptr()), int(kv_v.data_ptr()))], max_seq)
    store.set_timing(True)

    keepalive = {
        "embedding": embedding,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "cos": cos,
        "sin": sin,
        "kv_k": kv_k,
        "kv_v": kv_v,
        "q_bf16": q_bf16,
        "k_bf16": k_bf16,
        "v_bf16": v_bf16,
        "o_bf16": o_bf16,
        "fused_bf16": fused_bf16,
        "inp_norm": inp_norm,
        "post_norm": post_norm,
        "q_w": q_w,
        "k_w": k_w,
        "v_w": v_w,
        "o_w": o_w,
        "fused_w": fused_w,
    }
    return store, keepalive


def build_minimal_model(model_path: str, weights: dict) -> KrasisModel:
    model = KrasisModel.__new__(KrasisModel)
    model.cfg = SimpleNamespace(
        is_mla=False,
        model_path=model_path,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        gqa_head_dim=4,
        gated_attention=False,
    )
    model.quant_cfg = SimpleNamespace(attention="hqq4")
    model._hqq_attention_runtime = {}
    model._hqq_attention_runtime_nbits = None
    model._hqq_attention_cache_bytes = 0
    model._hqq_manifest = None
    model._hqq_rebuild = False

    layer = SimpleNamespace(
        device=torch.device("cpu"),
        layer_type="full_attention",
        attention=SimpleNamespace(),
        input_norm_weight=torch.ones(8, dtype=torch.float32),
        post_attn_norm_weight=torch.ones(8, dtype=torch.float32),
        gqa_weights={},
    )
    model.layers = [layer]

    def extract_layer_weights(self, _layer, _device):
        return {
            "layer_type": "full_attention",
            "attention": weights,
        }

    model._extract_layer_weights = MethodType(extract_layer_weights, model)
    return model


def register_hqq_direct(store: GpuDecodeStore, keepalive: dict, *, include_fused_desc: bool) -> None:
    layer_idx = 0
    runtime_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    keepalive["q_stage"] = stage_hqq(store, layer_idx, "q_proj", keepalive["q_w"])
    keepalive["k_stage"] = stage_hqq(store, layer_idx, "k_proj", keepalive["k_w"])
    keepalive["v_stage"] = stage_hqq(store, layer_idx, "v_proj", keepalive["v_w"])
    keepalive["o_stage"] = stage_hqq(store, layer_idx, "o_proj", keepalive["o_w"])
    if include_fused_desc:
        runtime_names.insert(0, "fused_qkv")
        keepalive["fused_stage"] = stage_hqq(store, layer_idx, "fused_qkv", keepalive["fused_w"])

    store.register_hqq_runtime_slots()
    store.swap_hqq_runtime_to_prefill()
    store.swap_hqq_runtime_to_decode()
    store.register_hqq_runtime_gqa_layer(
        layer_idx=layer_idx,
        input_norm_ptr=int(keepalive["inp_norm"].data_ptr()),
        input_norm_size=keepalive["inp_norm"].numel(),
        post_attn_norm_ptr=int(keepalive["post_norm"].data_ptr()),
        post_attn_norm_size=keepalive["post_norm"].numel(),
        backend="hqq4",
        nbits=4,
        format_version=HQQ_ATTENTION_CACHE_VERSION,
        tensor_names=runtime_names,
        num_heads=1,
        num_kv_heads=1,
        head_dim=4,
        sm_scale=1.0 / math.sqrt(4),
        q_norm_ptr=0,
        k_norm_ptr=0,
        gated=False,
    )


def register_hqq_from_artifacts(
    store: GpuDecodeStore,
    keepalive: dict,
    *,
    include_fused_artifact: bool,
) -> KrasisModel:
    tmpdir = tempfile.mkdtemp(prefix="hqq-fused-branch-artifacts-")
    model_path = os.path.join(tmpdir, "model")
    os.makedirs(model_path, exist_ok=True)
    old_home = os.environ.get("HOME")

    weights = {
        "q_proj": keepalive["q_w"].cpu(),
        "k_proj": keepalive["k_w"].cpu(),
        "v_proj": keepalive["v_w"].cpu(),
        "o_proj": keepalive["o_w"].cpu(),
    }
    if include_fused_artifact:
        weights["fused_qkv"] = keepalive["fused_w"].cpu()

    model = build_minimal_model(model_path, weights)
    try:
        os.environ["HOME"] = tmpdir
        model._prepare_hqq_attention_cache()
        model._maybe_write_hqq_attention_artifacts(0, "full_attention", weights)
        model._validate_hqq_attention_cache()
        model._load_hqq_attention_runtime_state()
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home

    runtime_keepalive = []
    registered = model._register_hqq_attention_layers_on_store(
        store,
        torch.device("cuda:0"),
        runtime_keepalive,
    )
    if registered != 1:
        raise AssertionError(f"artifact registration expected 1 layer, got {registered}")
    keepalive["artifact_model"] = model
    keepalive["artifact_runtime_keepalive"] = runtime_keepalive
    keepalive["artifact_tmpdir"] = tmpdir
    return model


def _real_model_quant_cfg() -> QuantConfig:
    return QuantConfig(
        lm_head="bf16",
        attention="hqq4",
        shared_expert="bf16",
        dense_mlp="bf16",
        gpu_expert_bits=4,
        cpu_expert_bits=4,
        kv_cache_format="bf16",
    )


def load_real_model_case(model_path: str, layer_idx: int) -> dict:
    key = (os.path.abspath(os.path.expanduser(model_path)), layer_idx)
    cached = _REAL_MODEL_CASES.get(key)
    if cached is not None:
        return cached

    resolved_model_path = key[0]
    if not os.path.isdir(resolved_model_path):
        raise RuntimeError(f"real-model path not found: {resolved_model_path}")

    model = KrasisModel(
        model_path=resolved_model_path,
        num_gpus=1,
        gpu_prefill=False,
        krasis_threads=1,
        quant_cfg=_real_model_quant_cfg(),
        kv_dtype=torch.bfloat16,
    )
    loader = WeightLoader(model.cfg, model.quant_cfg)
    load_layer_started = time.perf_counter()
    try:
        weights = loader.load_layer(
            layer_idx,
            torch.device("cpu"),
            attn_device=torch.device("cpu"),
        )
    finally:
        loader.close()
    load_layer_elapsed = time.perf_counter() - load_layer_started

    layer_build_started = time.perf_counter()
    layer = TransformerLayer(
        model.cfg,
        layer_idx,
        weights,
        torch.device("cpu"),
        krasis_engine=None,
        gpu_prefill_manager=None,
        gpu_prefill_threshold=model.gpu_prefill_threshold,
    )
    layer_build_elapsed = time.perf_counter() - layer_build_started
    if not getattr(layer, "gqa_weights", None):
        raise RuntimeError("real-model case requires populated GQA weights")

    q_w = layer.gqa_weights["q_proj"].contiguous()
    k_w = layer.gqa_weights["k_proj"].contiguous()
    v_w = layer.gqa_weights["v_proj"].contiguous()
    o_w = layer.gqa_weights["o_proj"].contiguous()
    fused_w = torch.cat([q_w, k_w, v_w], dim=0).contiguous()

    cached = {
        "model_path": resolved_model_path,
        "layer_idx": layer_idx,
        "runtime_layer_idx": 0,
        "cfg": model.cfg,
        "weights": weights,
        "layer": layer,
        "timing": {
            "load_layer_s": load_layer_elapsed,
            "transformer_layer_s": layer_build_elapsed,
        },
        "q_w": q_w,
        "k_w": k_w,
        "v_w": v_w,
        "o_w": o_w,
        "fused_w": fused_w,
    }
    _REAL_MODEL_CASES[key] = cached
    return cached


def build_real_model_store(
    real_case: dict,
    *,
    include_fused_config: bool,
) -> tuple[GpuDecodeStore, dict]:
    started = time.perf_counter()
    device = torch.device("cuda:0")
    cfg = real_case["cfg"]
    layer_idx = real_case["runtime_layer_idx"]
    hidden = int(cfg.hidden_size)
    head_dim = int(cfg.gqa_head_dim or cfg.head_dim)
    kv_heads = int(cfg.num_key_value_heads)
    kv_stride = kv_heads * head_dim
    max_seq = 4

    store = GpuDecodeStore(0)
    store.configure(hidden, 1, 1, float(cfg.rms_norm_eps))

    embedding = torch.linspace(-0.25, 0.25, hidden, dtype=torch.bfloat16, device=device).reshape(1, hidden)
    final_norm = torch.ones(hidden, dtype=torch.float32, device=device)
    lm_head = torch.zeros(1, hidden, dtype=torch.bfloat16, device=device)
    lm_head[0, 0] = 1.0

    store.set_embedding(int(embedding.data_ptr()))
    store.set_final_norm(int(final_norm.data_ptr()), hidden)
    lm_head_wid = store.register_weight(int(lm_head.data_ptr()), lm_head.shape[0], lm_head.shape[1], 0)
    store.set_lm_head(lm_head_wid)

    rope_half = int(cfg.rotary_dim // 2)
    inv_freq = 1.0 / (
        cfg.rope_theta
        ** (torch.arange(0, rope_half * 2, 2, dtype=torch.float32, device=device) / (rope_half * 2))
    )
    positions = torch.arange(max_seq, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().contiguous()
    sin = freqs.sin().contiguous()
    store.set_rope_tables(int(cos.data_ptr()), int(sin.data_ptr()), cos.shape[1], max_seq)

    kv_k = torch.zeros(max_seq, kv_stride, dtype=torch.bfloat16, device=device)
    kv_v = torch.zeros(max_seq, kv_stride, dtype=torch.bfloat16, device=device)

    q_bf16 = real_case["q_w"].to(device=device, dtype=torch.bfloat16)
    k_bf16 = real_case["k_w"].to(device=device, dtype=torch.bfloat16)
    v_bf16 = real_case["v_w"].to(device=device, dtype=torch.bfloat16)
    o_bf16 = real_case["o_w"].to(device=device, dtype=torch.bfloat16)
    fused_bf16 = real_case["fused_w"].to(device=device, dtype=torch.bfloat16)

    input_norm = real_case["layer"].input_norm_weight.float().contiguous().to(device)
    post_norm = real_case["layer"].post_attn_norm_weight
    if post_norm is None:
        post_norm = torch.zeros(0, dtype=torch.float32, device=device)
        post_norm_ptr = 0
        post_norm_size = 0
    else:
        post_norm = post_norm.float().contiguous().to(device)
        post_norm_ptr = int(post_norm.data_ptr())
        post_norm_size = int(post_norm.numel())
    q_norm = real_case["layer"].gqa_weights.get("q_norm")
    if q_norm is not None:
        q_norm = q_norm.float().contiguous().to(device)
        q_norm_ptr = int(q_norm.data_ptr())
    else:
        q_norm_ptr = 0
    k_norm = real_case["layer"].gqa_weights.get("k_norm")
    if k_norm is not None:
        k_norm = k_norm.float().contiguous().to(device)
        k_norm_ptr = int(k_norm.data_ptr())
    else:
        k_norm_ptr = 0

    q_wid = store.register_weight(int(q_bf16.data_ptr()), q_bf16.shape[0], q_bf16.shape[1], 0)
    k_wid = store.register_weight(int(k_bf16.data_ptr()), k_bf16.shape[0], k_bf16.shape[1], 0)
    v_wid = store.register_weight(int(v_bf16.data_ptr()), v_bf16.shape[0], v_bf16.shape[1], 0)
    o_wid = store.register_weight(int(o_bf16.data_ptr()), o_bf16.shape[0], o_bf16.shape[1], 0)
    fused_wid = store.register_weight(int(fused_bf16.data_ptr()), fused_bf16.shape[0], fused_bf16.shape[1], 0)

    store.register_gqa_layer(
        layer_idx=layer_idx,
        input_norm_ptr=int(input_norm.data_ptr()),
        input_norm_size=int(input_norm.numel()),
        post_attn_norm_ptr=post_norm_ptr,
        post_attn_norm_size=post_norm_size,
        q_proj_wid=q_wid,
        k_proj_wid=k_wid,
        v_proj_wid=v_wid,
        o_proj_wid=o_wid,
        fused_qkv_wid=(fused_wid if include_fused_config else None),
        num_heads=int(cfg.num_attention_heads),
        num_kv_heads=int(cfg.num_key_value_heads),
        head_dim=head_dim,
        sm_scale=1.0 / math.sqrt(head_dim),
        q_norm_ptr=q_norm_ptr,
        k_norm_ptr=k_norm_ptr,
        gated=bool(cfg.gated_attention),
    )
    store.set_kv_cache_ptrs_bf16([(layer_idx, int(kv_k.data_ptr()), int(kv_v.data_ptr()))], max_seq)
    store.set_timing(True)

    keepalive = {
        "embedding": embedding,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "cos": cos,
        "sin": sin,
        "kv_k": kv_k,
        "kv_v": kv_v,
        "q_bf16": q_bf16,
        "k_bf16": k_bf16,
        "v_bf16": v_bf16,
        "o_bf16": o_bf16,
        "fused_bf16": fused_bf16,
        "inp_norm": input_norm,
        "post_norm": post_norm,
        "q_norm": q_norm,
        "k_norm": k_norm,
        "q_w": real_case["q_w"],
        "k_w": real_case["k_w"],
        "v_w": real_case["v_w"],
        "o_w": real_case["o_w"],
        "fused_w": real_case["fused_w"],
        "store_build_timing": {
            "elapsed_s": time.perf_counter() - started,
            "include_fused_config": bool(include_fused_config),
        },
    }
    print(
        json.dumps(
            {
                "hqq_real_model_timing": {
                    "phase": "build_real_model_store",
                    "layer_idx": int(real_case["layer_idx"]),
                    "runtime_layer_idx": int(real_case["runtime_layer_idx"]),
                    "include_fused_config": bool(include_fused_config),
                    "elapsed_s": keepalive["store_build_timing"]["elapsed_s"],
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return store, keepalive


def register_hqq_from_real_model_artifacts(
    store: GpuDecodeStore,
    keepalive: dict,
    real_case: dict,
    *,
    drop_fused_artifact: bool,
) -> KrasisModel:
    tmpdir = tempfile.mkdtemp(prefix="hqq-real-model-artifacts-")
    model_path = os.path.join(tmpdir, os.path.basename(real_case["model_path"]))
    os.makedirs(model_path, exist_ok=True)
    old_home = os.environ.get("HOME")

    model = KrasisModel(
        model_path=real_case["model_path"],
        num_gpus=1,
        gpu_prefill=False,
        krasis_threads=1,
        quant_cfg=_real_model_quant_cfg(),
        kv_dtype=torch.bfloat16,
    )
    model.cfg.model_path = model_path
    model.layers = [real_case["layer"]]
    model._hqq_attention_runtime = {}
    model._hqq_attention_runtime_nbits = None
    model._hqq_attention_cache_bytes = 0
    model._hqq_manifest = None
    model._hqq_rebuild = False

    weights = {
        "q_proj": real_case["q_w"].cpu(),
        "k_proj": real_case["k_w"].cpu(),
        "v_proj": real_case["v_w"].cpu(),
        "o_proj": real_case["o_w"].cpu(),
    }
    q_norm = real_case["layer"].gqa_weights.get("q_norm")
    if q_norm is not None:
        weights["q_norm"] = q_norm.cpu()
    k_norm = real_case["layer"].gqa_weights.get("k_norm")
    if k_norm is not None:
        weights["k_norm"] = k_norm.cpu()

    try:
        os.environ["HOME"] = tmpdir
        prepare_started = time.perf_counter()
        model._prepare_hqq_attention_cache()
        prepare_elapsed = time.perf_counter() - prepare_started
        write_started = time.perf_counter()
        model._maybe_write_hqq_attention_artifacts(real_case["runtime_layer_idx"], "full_attention", weights)
        write_elapsed = time.perf_counter() - write_started
        if drop_fused_artifact:
            manifest = load_hqq_attention_manifest(model.cfg.model_path)
            if manifest is None:
                raise AssertionError("expected HQQ manifest after artifact write")
            removed_entry = None
            kept_entries = []
            for entry in manifest.get("tensors", []):
                if (
                    removed_entry is None
                    and entry.get("layer_idx") == real_case["runtime_layer_idx"]
                    and entry.get("tensor_name") == "fused_qkv"
                ):
                    removed_entry = entry
                    continue
                kept_entries.append(entry)
            if removed_entry is None:
                raise AssertionError("expected product artifact write to emit fused_qkv")
            manifest["tensors"] = kept_entries
            manifest["totals"]["tensor_bytes"] -= int(removed_entry["tensor_bytes"])
            manifest["totals"]["num_tensors"] = len(kept_entries)
            save_hqq_attention_manifest(model.cfg.model_path, manifest)
            os.remove(removed_entry["path"])
        validate_started = time.perf_counter()
        model._validate_hqq_attention_cache()
        validate_elapsed = time.perf_counter() - validate_started
        load_runtime_started = time.perf_counter()
        model._load_hqq_attention_runtime_state()
        load_runtime_elapsed = time.perf_counter() - load_runtime_started
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home

    runtime_keepalive = []
    register_started = time.perf_counter()
    registered = model._register_hqq_attention_layers_on_store(
        store,
        torch.device("cuda:0"),
        runtime_keepalive,
    )
    register_elapsed = time.perf_counter() - register_started
    if registered != 1:
        raise AssertionError(f"real-model artifact registration expected 1 layer, got {registered}")
    keepalive["artifact_model"] = model
    keepalive["artifact_runtime_keepalive"] = runtime_keepalive
    keepalive["artifact_tmpdir"] = tmpdir
    keepalive["artifact_timing"] = {
        "prepare_cache_s": prepare_elapsed,
        "artifact_write_s": write_elapsed,
        "validate_cache_s": validate_elapsed,
        "load_runtime_state_s": load_runtime_elapsed,
        "register_layers_s": register_elapsed,
    }
    return model


def collect_success_case(
    *,
    path_name: str,
    include_fused_desc: bool,
    include_fused_config: bool,
    expected_mode: str,
    expected_tensors: list[str],
    use_artifacts: bool,
) -> dict:
    store, keepalive = build_base_store(include_fused_config=include_fused_config)
    if use_artifacts:
        model = register_hqq_from_artifacts(
            store,
            keepalive,
            include_fused_artifact=include_fused_desc,
        )
    else:
        register_hqq_direct(
            store,
            keepalive,
            include_fused_desc=include_fused_desc,
        )
        model = None

    try:
        exec_json = json.loads(store.hqq_execution_json(0))
        projection_mode = exec_json["metadata"]["projection_mode"]
        if projection_mode != expected_mode:
            raise AssertionError(
                f"{path_name}: projection_mode={projection_mode} expected {expected_mode}"
            )
        store.debug_gpu_decode_step(0, 0)
        timing = json.loads(store.hqq_decode_timing_json())
        got_tensors = [entry["tensor_name"] for entry in timing["entries"]]
        if got_tensors != expected_tensors:
            raise AssertionError(
                f"{path_name}: timing tensors={got_tensors} expected {expected_tensors}"
            )
        return {
            "projection_mode": projection_mode,
            "timing_tensors": got_tensors,
            "timing_output_modes": [entry["output_mode"] for entry in timing["entries"]],
            "logits": store.download_logits_f32(),
        }
    finally:
        if model is not None:
            shutil.rmtree(keepalive["artifact_tmpdir"], ignore_errors=True)


def collect_failure_case(*, path_name: str, use_artifacts: bool) -> dict:
    store, keepalive = build_base_store(include_fused_config=True)
    if use_artifacts:
        model = register_hqq_from_artifacts(
            store,
            keepalive,
            include_fused_artifact=False,
        )
    else:
        register_hqq_direct(
            store,
            keepalive,
            include_fused_desc=False,
        )
        model = None

    try:
        exec_json = json.loads(store.hqq_execution_json(0))
        if exec_json["metadata"]["projection_mode"] != "split_qkv":
            raise AssertionError(
                f"{path_name}: mismatch case should retain split execution descriptor truth"
            )
        try:
            store.debug_gpu_decode_step(0, 0)
        except RuntimeError as exc:
            message = str(exc)
            if "fused GQA path" not in message:
                raise AssertionError(f"{path_name}: unexpected rejection text: {message}")
            if "fused_qkv_present=false" not in message:
                raise AssertionError(f"{path_name}: rejection text missing fused presence truth: {message}")
            timing = json.loads(store.hqq_decode_timing_json())
            if timing["entries"]:
                raise AssertionError(f"{path_name}: expected no timing entries on rejection, got {timing}")
            return {
                "projection_mode": exec_json["metadata"]["projection_mode"],
                "error": message,
            }
        raise AssertionError(f"{path_name}: expected fused mismatch case to hard-fail")
    finally:
        if model is not None:
            shutil.rmtree(keepalive["artifact_tmpdir"], ignore_errors=True)


def run_runtime_cases() -> dict:
    return {
        "split": collect_success_case(
            path_name="runtime/split",
            include_fused_desc=False,
            include_fused_config=False,
            expected_mode="split_qkv",
            expected_tensors=["q_proj", "k_proj", "v_proj", "o_proj"],
            use_artifacts=False,
        ),
        "fused": collect_success_case(
            path_name="runtime/fused",
            include_fused_desc=True,
            include_fused_config=True,
            expected_mode="fused_qkv",
            expected_tensors=["fused_qkv", "o_proj"],
            use_artifacts=False,
        ),
        "mismatch": collect_failure_case(
            path_name="runtime/mismatch",
            use_artifacts=False,
        ),
    }


def run_artifact_cases() -> dict:
    return {
        "split": collect_success_case(
            path_name="artifact/split",
            include_fused_desc=False,
            include_fused_config=False,
            expected_mode="split_qkv",
            expected_tensors=["q_proj", "k_proj", "v_proj", "o_proj"],
            use_artifacts=True,
        ),
        "fused": collect_success_case(
            path_name="artifact/fused",
            include_fused_desc=True,
            include_fused_config=True,
            expected_mode="fused_qkv",
            expected_tensors=["fused_qkv", "o_proj"],
            use_artifacts=True,
        ),
        "mismatch": collect_failure_case(
            path_name="artifact/mismatch",
            use_artifacts=True,
        ),
    }


def collect_real_model_success_case(
    *,
    real_case: dict,
    path_name: str,
    include_fused_desc: bool,
    include_fused_config: bool,
    expected_mode: str,
    expected_tensors: list[str],
) -> dict:
    store, keepalive = build_real_model_store(
        real_case,
        include_fused_config=include_fused_config,
    )
    model = register_hqq_from_real_model_artifacts(
        store,
        keepalive,
        real_case,
        drop_fused_artifact=not include_fused_desc,
    )
    try:
        exec_json = json.loads(store.hqq_execution_json(real_case["runtime_layer_idx"]))
        projection_mode = exec_json["metadata"]["projection_mode"]
        if projection_mode != expected_mode:
            raise AssertionError(
                f"{path_name}: projection_mode={projection_mode} expected {expected_mode}"
            )
        store.debug_gpu_decode_step(0, 0)
        timing = json.loads(store.hqq_decode_timing_json())
        got_tensors = [entry["tensor_name"] for entry in timing["entries"]]
        if got_tensors != expected_tensors:
            raise AssertionError(
                f"{path_name}: timing tensors={got_tensors} expected {expected_tensors}"
            )
        return {
            "projection_mode": projection_mode,
            "timing_tensors": got_tensors,
            "timing_output_modes": [entry["output_mode"] for entry in timing["entries"]],
            "layer_idx": real_case["layer_idx"],
            "runtime_layer_idx": real_case["runtime_layer_idx"],
            "model_path": real_case["model_path"],
            "artifact_timing": keepalive.get("artifact_timing", {}),
            "runtime_tensors": {
                name: model._hqq_attention_runtime[real_case["runtime_layer_idx"]][name]
                for name in expected_tensors
            },
        }
    finally:
        shutil.rmtree(keepalive["artifact_tmpdir"], ignore_errors=True)


def collect_real_model_failure_case(*, real_case: dict, path_name: str) -> dict:
    store, keepalive = build_real_model_store(
        real_case,
        include_fused_config=True,
    )
    model = register_hqq_from_real_model_artifacts(
        store,
        keepalive,
        real_case,
        drop_fused_artifact=True,
    )
    try:
        exec_json = json.loads(store.hqq_execution_json(real_case["runtime_layer_idx"]))
        if exec_json["metadata"]["projection_mode"] != "split_qkv":
            raise AssertionError(
                f"{path_name}: mismatch case should retain split execution descriptor truth"
            )
        try:
            store.debug_gpu_decode_step(0, 0)
        except RuntimeError as exc:
            message = str(exc)
            if "fused GQA path" not in message:
                raise AssertionError(f"{path_name}: unexpected rejection text: {message}")
            if "fused_qkv_present=false" not in message:
                raise AssertionError(f"{path_name}: rejection text missing fused presence truth: {message}")
            timing = json.loads(store.hqq_decode_timing_json())
            if timing["entries"]:
                raise AssertionError(f"{path_name}: expected no timing entries on rejection, got {timing}")
            return {
                "projection_mode": exec_json["metadata"]["projection_mode"],
                "error": message,
                "layer_idx": real_case["layer_idx"],
                "runtime_layer_idx": real_case["runtime_layer_idx"],
                "model_path": real_case["model_path"],
                "artifact_timing": keepalive.get("artifact_timing", {}),
            }
        raise AssertionError(f"{path_name}: expected fused mismatch case to hard-fail")
    finally:
        shutil.rmtree(keepalive["artifact_tmpdir"], ignore_errors=True)


def run_real_model_artifact_cases(model_path: str, layer_idx: int) -> dict:
    real_case = load_real_model_case(model_path, layer_idx)
    result = {
        "bf16_reference": {
            "model_path": real_case["model_path"],
            "layer_idx": real_case["layer_idx"],
            "runtime_layer_idx": real_case["runtime_layer_idx"],
            "source": "artifact_runtime_dequantized_weights_vs_real_bf16_weights",
            "probes": ["linear_ramp", "cos(i/17)", "sin(i/29)"],
        },
        "split": collect_real_model_success_case(
            real_case=real_case,
            path_name="real-model-artifact/split",
            include_fused_desc=False,
            include_fused_config=False,
            expected_mode="split_qkv",
            expected_tensors=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        "fused": collect_real_model_success_case(
            real_case=real_case,
            path_name="real-model-artifact/fused",
            include_fused_desc=True,
            include_fused_config=True,
            expected_mode="fused_qkv",
            expected_tensors=["fused_qkv", "o_proj"],
        ),
        "mismatch": collect_real_model_failure_case(
            real_case=real_case,
            path_name="real-model-artifact/mismatch",
        ),
    }
    for mode_name in ("split", "fused"):
        mode_result = result[mode_name]
        if mode_name == "split":
            reference_map = {
                "q_proj": real_case["q_w"],
                "k_proj": real_case["k_w"],
                "v_proj": real_case["v_w"],
                "o_proj": real_case["o_w"],
            }
        else:
            reference_map = {
                "fused_qkv": real_case["fused_w"],
                "o_proj": real_case["o_w"],
            }
        mode_result["similarity_vs_bf16"] = {
            tensor_name: _runtime_tensor_similarity(reference_map[tensor_name], runtime_desc)
            for tensor_name, runtime_desc in mode_result["runtime_tensors"].items()
        }
        mode_result.pop("runtime_tensors")
    print(
        json.dumps(
            {
                "hqq_real_model_timing": {
                    "phase": "real_model_artifact",
                    "model_path": real_case["model_path"],
                    "layer_idx": int(real_case["layer_idx"]),
                    "runtime_layer_idx": int(real_case["runtime_layer_idx"]),
                    "load_layer_s": real_case["timing"]["load_layer_s"],
                    "transformer_layer_s": real_case["timing"]["transformer_layer_s"],
                    "artifact_timing": result["fused"].get("artifact_timing", {}),
                    "similarity_vs_bf16": {
                        "split": result["split"]["similarity_vs_bf16"],
                        "fused": result["fused"]["similarity_vs_bf16"],
                    },
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("runtime", "artifact", "real-model-artifact", "all"),
        default="all",
        help="Choose direct runtime verification, artifact-load verification, or both.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_REAL_MODEL_PATH,
        help="Real model path for --mode real-model-artifact.",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=0,
        help="Layer index for --mode real-model-artifact.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HQQ fused-branch runtime verification")

    result = {}
    if args.mode in ("runtime", "all"):
        result["runtime"] = run_runtime_cases()
    if args.mode in ("artifact", "all"):
        result["artifact"] = run_artifact_cases()
    if args.mode in ("real-model-artifact", "all"):
        result["real_model_artifact"] = run_real_model_artifact_cases(
            args.model_path,
            args.layer_idx,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
