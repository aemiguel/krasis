#!/usr/bin/env python3
"""
Generate ground truth test data for Krasis CUDA kernel unit tests.

This script runs OFFLINE, not at runtime. It creates test fixtures
(raw binary + JSON metadata) that the Rust test harness compares
against our CUDA kernel outputs.

Usage:
    python generate_kernel_test_data.py              # generate all
    python generate_kernel_test_data.py --kernel rmsnorm  # one kernel
    python generate_kernel_test_data.py --list       # list available

Requirements: torch (always available in krasis conda env)
"""

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SEED = 42
OUT_DIR = Path(__file__).parent / "kernel_test_data"


def bf16_bytes(t: torch.Tensor) -> bytes:
    """Convert tensor to raw BF16 bytes (little-endian unsigned short)."""
    return t.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().tobytes()


def fp32_bytes(t: torch.Tensor) -> bytes:
    """Convert tensor to raw FP32 bytes."""
    return t.to(torch.float32).contiguous().numpy().tobytes()


def i32_bytes(t: torch.Tensor) -> bytes:
    """Convert tensor to raw INT32 bytes."""
    return t.to(torch.int32).contiguous().numpy().tobytes()


def save_test(name: str, inputs: dict, outputs: dict,
              atol: float = 1e-3, rtol: float = 1e-3, note: str = ""):
    """Save a test case as raw binary files + metadata JSON."""
    d = OUT_DIR / name
    d.mkdir(parents=True, exist_ok=True)

    meta = {"atol": atol, "rtol": rtol, "note": note, "inputs": {}, "outputs": {}}

    for k, (data, shape, dtype) in inputs.items():
        fname = f"input_{k}.bin"
        (d / fname).write_bytes(data)
        meta["inputs"][k] = {"file": fname, "shape": list(shape), "dtype": dtype}

    for k, (data, shape, dtype) in outputs.items():
        fname = f"expected_{k}.bin"
        (d / fname).write_bytes(data)
        meta["outputs"][k] = {"file": fname, "shape": list(shape), "dtype": dtype}

    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  saved {name} ({sum(len(v[0]) for v in inputs.values()) + sum(len(v[0]) for v in outputs.values())} bytes)")


# ════════════════════════════════════════════════════════════════════════
#  Test generators
# ════════════════════════════════════════════════════════════════════════

def gen_rmsnorm():
    """RMSNorm: out = x * rsqrt(mean(x^2) + eps) * weight"""
    torch.manual_seed(SEED)
    eps = 1e-6

    for M, H in [(1, 2048), (128, 2048), (5000, 2048), (128, 4096)]:
        x = torch.randn(M, H, dtype=torch.bfloat16)
        w = torch.randn(H, dtype=torch.bfloat16)

        # Reference: compute in FP32 from BF16 inputs (matching what kernel does)
        xf = x.float()
        wf = w.float()
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        out = (xf * rms * wf).to(torch.bfloat16)

        save_test(f"rmsnorm_m{M}_h{H}", {
            "x": (bf16_bytes(x), (M, H), "bf16"),
            "weight": (bf16_bytes(w), (H,), "bf16"),
        }, {
            "out": (bf16_bytes(out), (M, H), "bf16"),
        }, atol=1e-3, rtol=1e-3,
        note=f"RMSNorm M={M} H={H} eps={eps}")


def gen_fused_add_rmsnorm():
    """Fused residual add + RMSNorm: residual += x; out = rmsnorm(residual) * w"""
    torch.manual_seed(SEED)
    eps = 1e-6

    for M, H in [(128, 2048), (5000, 2048)]:
        residual = torch.randn(M, H, dtype=torch.bfloat16)
        x = torch.randn(M, H, dtype=torch.bfloat16)
        w = torch.randn(H, dtype=torch.bfloat16)

        # Reference — matches kernel behavior:
        # Pass 1: r = res + x (FP32), write BF16, accumulate r*r in FP32
        # Pass 2: read BF16 back, multiply by rms_inv * weight
        res_f = residual.float() + x.float()
        res_out = res_f.to(torch.bfloat16)  # updated residual (BF16 written back)
        # Kernel accumulates variance from FP32 r, not BF16-truncated value
        rms = torch.rsqrt(res_f.pow(2).mean(-1, keepdim=True) + eps)
        # But second pass reads BF16 residual back
        res_f2 = res_out.float()
        out = (res_f2 * rms * w.float()).to(torch.bfloat16)

        save_test(f"fused_add_rmsnorm_m{M}_h{H}", {
            "residual": (bf16_bytes(residual), (M, H), "bf16"),
            "x": (bf16_bytes(x), (M, H), "bf16"),
            "weight": (bf16_bytes(w), (H,), "bf16"),
        }, {
            "residual_out": (bf16_bytes(res_out), (M, H), "bf16"),
            "out": (bf16_bytes(out), (M, H), "bf16"),
        }, atol=3e-3, rtol=5e-3,
        note=f"Fused add+RMSNorm M={M} H={H} (parallel reduction rounding)")


def gen_rope():
    """RoPE (rotary position embedding) with partial rotary support."""
    torch.manual_seed(SEED)

    for M, num_q, num_kv, hd, partial in [
        (256, 16, 4, 128, 0.25),   # QCN-like: partial rotary
        (256, 16, 4, 128, 1.0),    # full rotary
        (1, 32, 8, 128, 0.5),      # single token, half rotary
    ]:
        half_dim = int(hd * partial) // 2  # rope applies to first half_dim pairs

        q = torch.randn(M, num_q, hd, dtype=torch.bfloat16)
        k = torch.randn(M, num_kv, hd, dtype=torch.bfloat16)
        positions = torch.arange(M, dtype=torch.int32)

        # Generate cos/sin tables in FP32 (matching decode store format)
        max_pos = M + 100
        freqs = 1.0 / (10000.0 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        t = torch.arange(max_pos, dtype=torch.float32)
        angles = torch.outer(t, freqs)  # [max_pos, half_dim]
        cos_cache = angles.cos()
        sin_cache = angles.sin()

        # Reference: apply rotary to first half_dim pairs of each head
        q_out = q.clone()
        k_out = k.clone()
        for i in range(M):
            pos = positions[i].item()
            c = cos_cache[pos]  # [half_dim]
            s = sin_cache[pos]
            for h in range(num_q):
                q0 = q_out[i, h, :half_dim].float()
                q1 = q_out[i, h, half_dim:2*half_dim].float()
                q_out[i, h, :half_dim] = (q0 * c - q1 * s).to(torch.bfloat16)
                q_out[i, h, half_dim:2*half_dim] = (q1 * c + q0 * s).to(torch.bfloat16)
            for h in range(num_kv):
                k0 = k_out[i, h, :half_dim].float()
                k1 = k_out[i, h, half_dim:2*half_dim].float()
                k_out[i, h, :half_dim] = (k0 * c - k1 * s).to(torch.bfloat16)
                k_out[i, h, half_dim:2*half_dim] = (k1 * c + k0 * s).to(torch.bfloat16)

        pct = int(partial * 100)
        save_test(f"rope_m{M}_h{hd}_partial{pct}", {
            "q": (bf16_bytes(q), (M, num_q, hd), "bf16"),
            "k": (bf16_bytes(k), (M, num_kv, hd), "bf16"),
            "positions": (i32_bytes(positions), (M,), "i32"),
            "cos_cache": (fp32_bytes(cos_cache), (max_pos, half_dim), "f32"),
            "sin_cache": (fp32_bytes(sin_cache), (max_pos, half_dim), "f32"),
        }, {
            "q_out": (bf16_bytes(q_out), (M, num_q, hd), "bf16"),
            "k_out": (bf16_bytes(k_out), (M, num_kv, hd), "bf16"),
        }, atol=1e-3, rtol=1e-3,
        note=f"RoPE M={M} num_q={num_q} num_kv={num_kv} hd={hd} partial={partial} half_dim={half_dim}")


def gen_silu_mul():
    """SiLU + mul: out = silu(gate) * up, from interleaved [M, 2*N]."""
    torch.manual_seed(SEED)

    for M, N in [(128, 512), (128, 2048), (5000, 512)]:
        gate_up = torch.randn(M, 2 * N, dtype=torch.bfloat16)

        # Reference: split gate and up, apply silu to gate, multiply
        gate = gate_up[:, :N].float()
        up = gate_up[:, N:].float()
        silu_gate = gate * torch.sigmoid(gate)
        out = (silu_gate * up).to(torch.bfloat16)

        save_test(f"silu_mul_m{M}_n{N}", {
            "gate_up": (bf16_bytes(gate_up), (M, 2 * N), "bf16"),
        }, {
            "out": (bf16_bytes(out), (M, N), "bf16"),
        }, atol=5e-3, rtol=5e-3,
        note=f"SiLU+Mul M={M} N={N} (sigmoid chain has extra BF16 rounding)")


def gen_relu2():
    """ReLU squared: out = relu(x)^2."""
    torch.manual_seed(SEED)

    for M, N in [(128, 512), (128, 2048)]:
        x = torch.randn(M, N, dtype=torch.bfloat16)
        out = F.relu(x.float()).pow(2).to(torch.bfloat16)

        save_test(f"relu2_m{M}_n{N}", {
            "x": (bf16_bytes(x), (M, N), "bf16"),
        }, {
            "out": (bf16_bytes(out), (M, N), "bf16"),
        }, atol=1e-3, rtol=1e-3,
        note=f"ReLU² M={M} N={N}")


def gen_transpose_3d():
    """3D transpose [A,B,C] -> [B,A,C]."""
    torch.manual_seed(SEED)

    for A, B, C in [(32, 500, 128), (8, 1000, 64)]:
        # FP32 version
        x_f32 = torch.randn(A, B, C, dtype=torch.float32)
        out_f32 = x_f32.permute(1, 0, 2).contiguous()

        save_test(f"transpose_3d_f32_{A}_{B}_{C}", {
            "x": (fp32_bytes(x_f32), (A, B, C), "f32"),
        }, {
            "out": (fp32_bytes(out_f32), (B, A, C), "f32"),
        }, atol=0.0, rtol=0.0,
        note=f"3D transpose FP32 [{A},{B},{C}] -> [{B},{A},{C}] (exact)")

        # BF16 version
        x_bf16 = torch.randn(A, B, C, dtype=torch.bfloat16)
        out_bf16 = x_bf16.permute(1, 0, 2).contiguous()

        save_test(f"transpose_3d_bf16_{A}_{B}_{C}", {
            "x": (bf16_bytes(x_bf16), (A, B, C), "bf16"),
        }, {
            "out": (bf16_bytes(out_bf16), (B, A, C), "bf16"),
        }, atol=0.0, rtol=0.0,
        note=f"3D transpose BF16 [{A},{B},{C}] -> [{B},{A},{C}] (exact)")


def gen_sigmoid_topk():
    """Sigmoid top-K routing: sigmoid(gate), select top-K experts."""
    torch.manual_seed(SEED)

    for M, E, topk in [(128, 512, 10), (1, 512, 10), (5000, 512, 6)]:
        # Gate logits as FP32 (matching cuBLAS output)
        gate = torch.randn(M, E, dtype=torch.float32)

        # Reference
        scores = torch.sigmoid(gate)
        topk_vals, topk_ids = scores.topk(topk, dim=-1)

        save_test(f"sigmoid_topk_m{M}_e{E}_k{topk}", {
            "gate": (fp32_bytes(gate), (M, E), "f32"),
        }, {
            "topk_weights": (fp32_bytes(topk_vals), (M, topk), "f32"),
            "topk_ids": (i32_bytes(topk_ids), (M, topk), "i32"),
        }, atol=1e-5, rtol=1e-5,
        note=f"Sigmoid top-K M={M} E={E} topk={topk}. IDs must match exactly, weights within FP32 tolerance.")


def gen_softmax_topk():
    """Softmax top-K routing: softmax(gate), select top-K experts."""
    torch.manual_seed(SEED + 1)  # different seed to get different values

    for M, E, topk in [(128, 512, 10), (1, 64, 6)]:
        gate = torch.randn(M, E, dtype=torch.float32)

        # Reference
        scores = torch.softmax(gate, dim=-1)
        topk_vals, topk_ids = scores.topk(topk, dim=-1)

        save_test(f"softmax_topk_m{M}_e{E}_k{topk}", {
            "gate": (fp32_bytes(gate), (M, E), "f32"),
        }, {
            "topk_weights": (fp32_bytes(topk_vals), (M, topk), "f32"),
            "topk_ids": (i32_bytes(topk_ids), (M, topk), "i32"),
        }, atol=1e-5, rtol=1e-5,
        note=f"Softmax top-K M={M} E={E} topk={topk}")


def gen_concat_3_bf16():
    """Concatenate 3 BF16 arrays along last dim."""
    torch.manual_seed(SEED)

    M = 500
    d1, d2, d3 = 128, 128, 128

    a = torch.randn(M, d1, dtype=torch.bfloat16)
    b = torch.randn(M, d2, dtype=torch.bfloat16)
    c = torch.randn(M, d3, dtype=torch.bfloat16)
    out = torch.cat([a, b, c], dim=-1)

    save_test("concat_3_bf16_m500", {
        "a": (bf16_bytes(a), (M, d1), "bf16"),
        "b": (bf16_bytes(b), (M, d2), "bf16"),
        "c": (bf16_bytes(c), (M, d3), "bf16"),
    }, {
        "out": (bf16_bytes(out), (M, d1 + d2 + d3), "bf16"),
    }, atol=0.0, rtol=0.0,
    note=f"Concat3 BF16 M={M} dims=[{d1},{d2},{d3}] (exact)")


def gen_gated_q_split():
    """Split interleaved [M, H, 2*D] into Q [M, H, D] and gate [M, H, D]."""
    torch.manual_seed(SEED)

    M, H, D = 256, 16, 128
    # Interleaved: for each head, first D elements are Q, next D are gate
    qg = torch.randn(M, H, 2 * D, dtype=torch.bfloat16)

    q = qg[:, :, :D].contiguous()
    gate = qg[:, :, D:].contiguous()

    save_test("gated_q_split_m256_h16_d128", {
        "qg": (bf16_bytes(qg), (M, H, 2 * D), "bf16"),
    }, {
        "q": (bf16_bytes(q), (M, H, D), "bf16"),
        "gate": (bf16_bytes(gate), (M, H, D), "bf16"),
    }, atol=0.0, rtol=0.0,
    note=f"Gated Q split M={M} H={H} D={D} (exact)")


def gen_fp8_kv_roundtrip():
    """FP8 E4M3 round-trip: BF16 -> FP8 -> BF16. Tests quantization bounds."""
    torch.manual_seed(SEED)

    M, D = 1000, 128
    # Use values in a reasonable activation range (not too extreme)
    k = torch.randn(M, D, dtype=torch.bfloat16) * 2.0
    v = torch.randn(M, D, dtype=torch.bfloat16) * 2.0

    # Reference: BF16 -> FP8 E4M3 -> BF16
    # torch.float8_e4m3fn is the dtype for E4M3
    k_fp8 = k.to(torch.float8_e4m3fn)
    v_fp8 = v.to(torch.float8_e4m3fn)
    k_rt = k_fp8.to(torch.bfloat16)
    v_rt = v_fp8.to(torch.bfloat16)

    save_test("fp8_kv_roundtrip_m1000_d128", {
        "k": (bf16_bytes(k), (M, D), "bf16"),
        "v": (bf16_bytes(v), (M, D), "bf16"),
    }, {
        "k_roundtrip": (bf16_bytes(k_rt), (M, D), "bf16"),
        "v_roundtrip": (bf16_bytes(v_rt), (M, D), "bf16"),
    }, atol=5e-2, rtol=5e-2,
    note=f"FP8 E4M3 round-trip M={M} D={D}. Tests quantization bounds, not exact match.")


def gen_sigmoid_mul():
    """Sigmoid-gated multiply: out = attn * sigmoid(gate)."""
    torch.manual_seed(SEED)

    M, N = 256, 2048
    attn = torch.randn(M, N, dtype=torch.bfloat16)
    gate = torch.randn(M, N, dtype=torch.bfloat16)

    out = (attn.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)

    save_test("sigmoid_mul_m256_n2048", {
        "attn": (bf16_bytes(attn), (M, N), "bf16"),
        "gate": (bf16_bytes(gate), (M, N), "bf16"),
    }, {
        "out": (bf16_bytes(out), (M, N), "bf16"),
    }, atol=1e-3, rtol=1e-3,
    note=f"Sigmoid-gated multiply M={M} N={N}")


def gen_gqa_attention():
    """GQA causal attention: single-chunk and cross-chunk with FP8 cache."""
    torch.manual_seed(SEED)

    # --- Test 1: single-chunk, no FP8 cache (start_pos=0) ---
    M, num_q, num_kv, hd = 64, 16, 4, 128
    scale = 1.0 / math.sqrt(hd)

    q = torch.randn(M, num_q, hd, dtype=torch.bfloat16)
    k = torch.randn(M, num_kv, hd, dtype=torch.bfloat16)
    v = torch.randn(M, num_kv, hd, dtype=torch.bfloat16)

    # Reference: expand KV heads for GQA, then standard SDPA
    heads_per_group = num_q // num_kv
    k_exp = k.unsqueeze(2).expand(-1, num_kv, heads_per_group, -1).reshape(M, num_q, hd)
    v_exp = v.unsqueeze(2).expand(-1, num_kv, heads_per_group, -1).reshape(M, num_q, hd)

    # [M, num_q, hd] -> [num_q, M, hd] for SDPA
    q_t = q.float().transpose(0, 1)
    k_t = k_exp.float().transpose(0, 1)
    v_t = v_exp.float().transpose(0, 1)

    out_t = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
    out = out_t.transpose(0, 1).to(torch.bfloat16)  # back to [M, num_q, hd]

    save_test("gqa_attention_m64_h16_kv4_d128", {
        "q": (bf16_bytes(q), (M, num_q, hd), "bf16"),
        "k": (bf16_bytes(k), (M, num_kv, hd), "bf16"),
        "v": (bf16_bytes(v), (M, num_kv, hd), "bf16"),
    }, {
        "out": (bf16_bytes(out), (M, num_q, hd), "bf16"),
    }, atol=1e-2, rtol=1e-2,
    note=f"GQA attention M={M} num_q={num_q} num_kv={num_kv} hd={hd} scale={scale:.6f}. Uses MMA so tolerance is wider.")

    # --- Test 2: cross-chunk attention (FP8 cache for prior tokens) ---
    # Simulate: 32 tokens already in FP8 cache, 32 new tokens in BF16
    # The new tokens should attend to ALL 64 positions
    cached_len = 32
    new_len = 32
    total = cached_len + new_len

    q_full = torch.randn(total, num_q, hd, dtype=torch.bfloat16)
    k_full = torch.randn(total, num_kv, hd, dtype=torch.bfloat16)
    v_full = torch.randn(total, num_kv, hd, dtype=torch.bfloat16)

    # Full-sequence reference (FP32 SDPA)
    k_exp_full = k_full.unsqueeze(2).expand(-1, num_kv, heads_per_group, -1).reshape(total, num_q, hd)
    v_exp_full = v_full.unsqueeze(2).expand(-1, num_kv, heads_per_group, -1).reshape(total, num_q, hd)
    q_t2 = q_full.float().transpose(0, 1)
    k_t2 = k_exp_full.float().transpose(0, 1)
    v_t2 = v_exp_full.float().transpose(0, 1)
    out_full = F.scaled_dot_product_attention(q_t2, k_t2, v_t2, is_causal=True, scale=scale)
    out_full = out_full.transpose(0, 1).to(torch.bfloat16)

    # The output we care about is the NEW tokens only (positions 32-63)
    out_new = out_full[cached_len:]

    # FP8 cached K/V (simulate round-trip through FP8 E4M3)
    kv_stride = num_kv * hd
    k_cached_bf16 = k_full[:cached_len].reshape(cached_len, kv_stride)
    v_cached_bf16 = v_full[:cached_len].reshape(cached_len, kv_stride)
    k_cached_fp8 = k_cached_bf16.to(torch.float8_e4m3fn)
    v_cached_fp8 = v_cached_bf16.to(torch.float8_e4m3fn)

    # Recompute reference with FP8 cached values for accuracy comparison
    k_full_dequant = k_full.clone()
    v_full_dequant = v_full.clone()
    k_full_dequant[:cached_len] = k_cached_fp8.to(torch.bfloat16).reshape(cached_len, num_kv, hd)
    v_full_dequant[:cached_len] = v_cached_fp8.to(torch.bfloat16).reshape(cached_len, num_kv, hd)

    k_exp_dq = k_full_dequant.unsqueeze(2).expand(-1, num_kv, heads_per_group, -1).reshape(total, num_q, hd)
    v_exp_dq = v_full_dequant.unsqueeze(2).expand(-1, num_kv, heads_per_group, -1).reshape(total, num_q, hd)
    q_t3 = q_full.float().transpose(0, 1)
    k_t3 = k_exp_dq.float().transpose(0, 1)
    v_t3 = v_exp_dq.float().transpose(0, 1)
    out_dq = F.scaled_dot_product_attention(q_t3, k_t3, v_t3, is_causal=True, scale=scale)
    out_new_dq = out_dq.transpose(0, 1)[cached_len:].to(torch.bfloat16)

    # Pack FP8 K/V cache as raw bytes (each element is 1 byte)
    k_fp8_bytes = k_cached_fp8.view(torch.uint8).numpy().tobytes()
    v_fp8_bytes = v_cached_fp8.view(torch.uint8).numpy().tobytes()

    save_test("gqa_cross_chunk_m32_cached32_h16_kv4_d128", {
        "q_new": (bf16_bytes(q_full[cached_len:]), (new_len, num_q, hd), "bf16"),
        "k_new": (bf16_bytes(k_full[cached_len:]), (new_len, num_kv, hd), "bf16"),
        "v_new": (bf16_bytes(v_full[cached_len:]), (new_len, num_kv, hd), "bf16"),
        "k_cache_fp8": (k_fp8_bytes, (cached_len, kv_stride), "fp8e4m3"),
        "v_cache_fp8": (v_fp8_bytes, (cached_len, kv_stride), "fp8e4m3"),
    }, {
        "out_new": (bf16_bytes(out_new_dq), (new_len, num_q, hd), "bf16"),
    }, atol=5e-2, rtol=5e-2,
    note=f"Cross-chunk GQA: {cached_len} cached (FP8) + {new_len} new (BF16). Tolerance wider due to FP8 cache + MMA.")


def gen_fp8_kv_roundtrip():
    """FP8 E4M3 KV cache write + read: BF16 -> FP8 -> dequant back."""
    torch.manual_seed(SEED)

    M, D = 256, 512  # num_kv_heads * head_dim = 4*128
    k = torch.randn(M, D, dtype=torch.bfloat16) * 2.0
    v = torch.randn(M, D, dtype=torch.bfloat16) * 2.0

    # Reference: BF16 -> FP8 E4M3 -> BF16
    k_fp8 = k.to(torch.float8_e4m3fn)
    v_fp8 = v.to(torch.float8_e4m3fn)
    k_rt = k_fp8.to(torch.bfloat16)
    v_rt = v_fp8.to(torch.bfloat16)

    save_test("fp8_kv_roundtrip_m256_d512", {
        "k": (bf16_bytes(k), (M, D), "bf16"),
        "v": (bf16_bytes(v), (M, D), "bf16"),
    }, {
        "k_roundtrip": (bf16_bytes(k_rt), (M, D), "bf16"),
        "v_roundtrip": (bf16_bytes(v_rt), (M, D), "bf16"),
    }, atol=5e-2, rtol=5e-2,
    note=f"FP8 E4M3 round-trip M={M} D={D}. Write BF16 through kv_cache_append_fp8, read back, compare.")


def gen_la_l2norm():
    """L2 norm per head with scale."""
    torch.manual_seed(SEED)

    M, num_heads, dim = 128, 32, 128
    scale = math.sqrt(dim)

    x = torch.randn(M, num_heads, dim, dtype=torch.float32)
    # Reference: F.normalize then scale
    out = F.normalize(x, dim=-1) * scale

    save_test("la_l2norm_m128_h32_d128", {
        "x": (fp32_bytes(x), (M, num_heads, dim), "f32"),
    }, {
        "out": (fp32_bytes(out), (M, num_heads, dim), "f32"),
    }, atol=1e-4, rtol=1e-4,
    note=f"L2 norm per head M={M} num_heads={num_heads} dim={dim} scale={scale:.2f}")


def gen_la_gate_beta():
    """Compute gating params: beta=sigmoid(b), gate=-exp(A_log)*softplus(a+dt_bias)."""
    torch.manual_seed(SEED)

    M, nv = 128, 32
    b = torch.randn(M, nv, dtype=torch.bfloat16)
    a = torch.randn(M, nv, dtype=torch.bfloat16)
    A_log = torch.randn(nv, dtype=torch.float32) * 0.5
    dt_bias = torch.randn(nv, dtype=torch.float32)

    # Reference
    beta = torch.sigmoid(b.float())
    a_val = torch.exp(A_log)
    x_sp = a.float() + dt_bias.unsqueeze(0)
    sp = torch.where(x_sp > 20.0, x_sp, torch.log1p(torch.exp(x_sp)))
    gate = -a_val.unsqueeze(0) * sp

    save_test("la_gate_beta_m128_nv32", {
        "b": (bf16_bytes(b), (M, nv), "bf16"),
        "a": (bf16_bytes(a), (M, nv), "bf16"),
        "A_log": (fp32_bytes(A_log), (nv,), "f32"),
        "dt_bias": (fp32_bytes(dt_bias), (nv,), "f32"),
    }, {
        "beta": (fp32_bytes(beta), (M, nv), "f32"),
        "gate": (fp32_bytes(gate), (M, nv), "f32"),
    }, atol=1e-4, rtol=1e-4,
    note=f"LA gating M={M} nv={nv}")


def gen_la_gated_rmsnorm():
    """Gated RMSNorm: out = rmsnorm(x) * weight * silu(gate)."""
    torch.manual_seed(SEED)

    M, nv, dv = 64, 32, 128
    eps = 1e-6

    x = torch.randn(M, nv, dv, dtype=torch.float32)
    gate = torch.randn(M, nv, dv, dtype=torch.bfloat16)
    weight = torch.randn(dv, dtype=torch.bfloat16)

    # Reference
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    normed = x * rms * weight.float().unsqueeze(0).unsqueeze(0)
    g = gate.float()
    silu_g = g * torch.sigmoid(g)
    out = (normed * silu_g).to(torch.bfloat16)

    save_test("la_gated_rmsnorm_m64_nv32_dv128", {
        "x": (fp32_bytes(x), (M, nv, dv), "f32"),
        "gate": (bf16_bytes(gate), (M, nv, dv), "bf16"),
        "weight": (bf16_bytes(weight), (dv,), "bf16"),
    }, {
        "out": (bf16_bytes(out), (M * nv * dv,), "bf16"),
    }, atol=3e-3, rtol=3e-3,
    note=f"Gated RMSNorm M={M} nv={nv} dv={dv}")


def gen_la_conv1d():
    """Causal depthwise conv1d + SiLU over full sequence."""
    torch.manual_seed(SEED)

    M, conv_dim, kernel_dim = 64, 384, 4  # conv_dim = nk*dk + nv*dv for QCN-like
    # Input: [M, conv_dim] BF16, row-major
    x = torch.randn(M, conv_dim, dtype=torch.bfloat16)
    weight = torch.randn(conv_dim, kernel_dim, dtype=torch.float32)
    conv_state = torch.randn(conv_dim, kernel_dim, dtype=torch.float32)

    # Reference: causal depthwise conv1d + SiLU
    # Pad with conv_state on the left
    x_f = x.float()
    # Build full sequence: state (kernel_dim cols) + input (M cols)
    # For each channel, convolve over the sequence
    out_ref = torch.zeros(conv_dim, M, dtype=torch.float32)
    state_out = conv_state.clone()

    for ch in range(conv_dim):
        for t in range(M):
            acc = 0.0
            for w in range(kernel_dim):
                src_pos = t + w - (kernel_dim - 1)
                if src_pos < 0:
                    val = conv_state[ch, kernel_dim + src_pos].item()
                else:
                    val = x_f[src_pos, ch].item()
                acc += val * weight[ch, w].item()
            # SiLU
            sig = 1.0 / (1.0 + math.exp(-acc))
            out_ref[ch, t] = acc * sig

    # Update state with last kernel_dim tokens
    for ch in range(conv_dim):
        for w in range(kernel_dim):
            src_pos = M - kernel_dim + w
            if src_pos < 0:
                state_out[ch, w] = conv_state[ch, kernel_dim + src_pos]
            else:
                state_out[ch, w] = x_f[src_pos, ch]

    save_test("la_conv1d_m64_cd384_k4", {
        "x": (bf16_bytes(x), (M, conv_dim), "bf16"),
        "weight": (fp32_bytes(weight), (conv_dim, kernel_dim), "f32"),
        "conv_state": (fp32_bytes(conv_state), (conv_dim, kernel_dim), "f32"),
    }, {
        "out": (fp32_bytes(out_ref), (conv_dim, M), "f32"),
        "state_out": (fp32_bytes(state_out), (conv_dim, kernel_dim), "f32"),
    }, atol=1e-4, rtol=1e-4,
    note=f"Causal conv1d+SiLU M={M} conv_dim={conv_dim} kernel={kernel_dim}")


def gen_la_uninterleave_qkvz():
    """Uninterleave head-grouped QKVZ projection output."""
    torch.manual_seed(SEED)

    M, nk, dk, hr, dv = 32, 8, 128, 4, 128
    nv = nk * hr  # 32
    group_dim = 2 * dk + 2 * hr * dv  # 256 + 1024 = 1280
    total_dim = nk * group_dim  # 10240

    qkvz = torch.randn(M, total_dim, dtype=torch.bfloat16)

    # Reference: uninterleave
    q_out = torch.zeros(M, nk * dk, dtype=torch.bfloat16)
    k_out = torch.zeros(M, nk * dk, dtype=torch.bfloat16)
    v_out = torch.zeros(M, nv * dv, dtype=torch.bfloat16)
    z_out = torch.zeros(M, nv * dv, dtype=torch.bfloat16)

    for t in range(M):
        for hg in range(nk):
            base = hg * group_dim
            # q: first dk
            q_out[t, hg*dk:(hg+1)*dk] = qkvz[t, base:base+dk]
            # k: next dk
            k_out[t, hg*dk:(hg+1)*dk] = qkvz[t, base+dk:base+2*dk]
            # v: next hr*dv
            for s in range(hr):
                v_head = hg * hr + s
                v_start = base + 2*dk + s*dv
                v_out[t, v_head*dv:(v_head+1)*dv] = qkvz[t, v_start:v_start+dv]
            # z: last hr*dv
            for s in range(hr):
                z_head = hg * hr + s
                z_start = base + 2*dk + hr*dv + s*dv
                z_out[t, z_head*dv:(z_head+1)*dv] = qkvz[t, z_start:z_start+dv]

    save_test("la_uninterleave_qkvz_m32_nk8_dk128_hr4_dv128", {
        "qkvz": (bf16_bytes(qkvz), (M, total_dim), "bf16"),
    }, {
        "q": (bf16_bytes(q_out), (M, nk * dk), "bf16"),
        "k": (bf16_bytes(k_out), (M, nk * dk), "bf16"),
        "v": (bf16_bytes(v_out), (M, nv * dv), "bf16"),
        "z": (bf16_bytes(z_out), (M, nv * dv), "bf16"),
    }, atol=0.0, rtol=0.0,
    note=f"Uninterleave QKVZ M={M} nk={nk} dk={dk} hr={hr} dv={dv} (exact permutation)")


def gen_embedding():
    """Embedding lookup: out[t] = table[token_ids[t]]."""
    torch.manual_seed(SEED)

    M, D, vocab = 64, 2048, 1000
    table = torch.randn(vocab, D, dtype=torch.bfloat16)
    token_ids = torch.randint(0, vocab, (M,), dtype=torch.int32)

    out = table[token_ids.long()]

    save_test("embedding_m64_d2048_v1000", {
        "table": (bf16_bytes(table), (vocab, D), "bf16"),
        "token_ids": (i32_bytes(token_ids), (M,), "i32"),
    }, {
        "out": (bf16_bytes(out), (M, D), "bf16"),
    }, atol=0.0, rtol=0.0,
    note=f"Embedding lookup M={M} D={D} vocab={vocab} (exact)")


def gen_la_triangular_solve():
    """Forward substitution on strictly lower-triangular A: x += A @ x (row by row)."""
    torch.manual_seed(SEED)

    nv, num_chunks, cs, dim = 4, 2, 8, 16  # small for tractable reference

    # b is the input, A is strictly lower triangular (zero on and above diagonal)
    b = torch.randn(nv, num_chunks, cs, dim, dtype=torch.float32)
    A_full = torch.randn(nv, num_chunks, cs, cs, dtype=torch.float32)
    # Zero upper triangular + diagonal
    A = torch.tril(A_full, diagonal=-1)

    # Reference: forward substitution x[i] += sum_j<i A[i,j] * x[j]
    x = b.clone()
    for h in range(nv):
        for c in range(num_chunks):
            for i in range(1, cs):
                for j in range(i):
                    x[h, c, i] += A[h, c, i, j] * x[h, c, j]

    save_test("la_tri_solve_nv4_nc2_cs8_d16", {
        "b": (fp32_bytes(b), (nv, num_chunks, cs, dim), "f32"),
        "A": (fp32_bytes(A), (nv, num_chunks, cs, cs), "f32"),
    }, {
        "x_out": (fp32_bytes(x), (nv, num_chunks, cs, dim), "f32"),
    }, atol=1e-5, rtol=1e-5,
    note=f"LA triangular solve nv={nv} chunks={num_chunks} cs={cs} dim={dim}")


def gen_la_chunk_recurrence():
    """Full chunk recurrence: compute_v_new + chunk_output + state_update."""
    torch.manual_seed(SEED)

    nv, cs, dk, dv = 4, 8, 16, 16  # small for tractable reference

    # Inputs for one chunk
    q = torch.randn(nv, cs, dk, dtype=torch.float32)
    k = torch.randn(nv, cs, dk, dtype=torch.float32)
    v_corr = torch.randn(nv, cs, dv, dtype=torch.float32)
    k_cumd = torch.randn(nv, cs, dk, dtype=torch.float32)
    g_cum = torch.randn(nv, cs, dtype=torch.float32) * 0.5  # moderate gate values
    state = torch.randn(nv, dk, dv, dtype=torch.float32) * 0.1

    # --- Step 1: compute_v_new = v_corr - k_cumd @ state ---
    # k_cumd @ state: [nv, cs, dk] @ [nv, dk, dv] = [nv, cs, dv]
    v_prime = torch.bmm(k_cumd, state)
    v_new = v_corr - v_prime

    # --- Step 2: chunk_output ---
    # inter = (q * exp(g)) @ state
    # intra = tril(q @ k^T * decay, -1) @ v_new
    output = torch.zeros(nv, cs, dv, dtype=torch.float32)
    for h in range(nv):
        for t in range(cs):
            exp_g_t = math.exp(g_cum[h, t].item())
            # Inter-chunk: (q[t] * exp(g[t])) @ state
            for d in range(dv):
                inter = 0.0
                for j in range(dk):
                    inter += q[h, t, j].item() * state[h, j, d].item()
                inter *= exp_g_t

                # Intra-chunk: sum_{s<t} (q[t]@k[s]) * decay(t,s) * v_new[s,d]
                intra = 0.0
                for s in range(t):
                    qk = sum(q[h, t, j].item() * k[h, s, j].item() for j in range(dk))
                    decay = math.exp(g_cum[h, t].item() - g_cum[h, s].item())
                    intra += qk * decay * v_new[h, s, d].item()

                output[h, t, d] = inter + intra

    # --- Step 3: state_update ---
    # state = state * exp(g_last) + (k * k_decay)^T @ v_new
    state_out = state.clone()
    for h in range(nv):
        g_last = g_cum[h, cs - 1].item()
        g_last_exp = math.exp(g_last)
        for j in range(dk):
            for d in range(dv):
                s = state[h, j, d].item() * g_last_exp
                for t in range(cs):
                    k_decay = math.exp(g_last - g_cum[h, t].item())
                    s += k[h, t, j].item() * k_decay * v_new[h, t, d].item()
                state_out[h, j, d] = s

    save_test("la_chunk_recurrence_nv4_cs8_dk16_dv16", {
        "q": (fp32_bytes(q), (nv, cs, dk), "f32"),
        "k": (fp32_bytes(k), (nv, cs, dk), "f32"),
        "v_corr": (fp32_bytes(v_corr), (nv, cs, dv), "f32"),
        "k_cumd": (fp32_bytes(k_cumd), (nv, cs, dk), "f32"),
        "g_cum": (fp32_bytes(g_cum), (nv, cs), "f32"),
        "state": (fp32_bytes(state), (nv, dk, dv), "f32"),
    }, {
        "v_new": (fp32_bytes(v_new), (nv, cs, dv), "f32"),
        "output": (fp32_bytes(output), (nv, cs, dv), "f32"),
        "state_out": (fp32_bytes(state_out), (nv, dk, dv), "f32"),
    }, atol=1e-4, rtol=1e-4,
    note=f"LA chunk recurrence nv={nv} cs={cs} dk={dk} dv={dv}. Tests compute_v_new + chunk_output + state_update.")


def gen_moe_gather_scatter():
    """MoE gather (reorder tokens) and scatter-add (accumulate back)."""
    torch.manual_seed(SEED)

    M, hidden, topk = 32, 64, 4  # small for tractable reference
    E = 8  # number of experts

    # Generate random routing: each token goes to topk experts
    topk_ids = torch.stack([torch.randperm(E)[:topk] for _ in range(M)]).to(torch.int32)

    # Build sorted_token_ids: for each expert, list all tokens routed to it
    sorted_ids = []
    for e in range(E):
        for t in range(M):
            if e in topk_ids[t]:
                sorted_ids.append(t)
    # Pad to block_size=128 boundary per expert
    block_size = 128
    sorted_padded = []
    expert_offsets = [0]
    idx = 0
    for e in range(E):
        count = sum(1 for t in range(M) for _ in range(1) if e in topk_ids[t])
        padded = ((count + block_size - 1) // block_size) * block_size
        for i in range(padded):
            if i < count:
                sorted_padded.append(sorted_ids[idx])
                idx += 1
            else:
                sorted_padded.append(M)  # padding (out of bounds)
        expert_offsets.append(expert_offsets[-1] + padded)
    total_padded = len(sorted_padded)
    sorted_ids_tensor = torch.tensor(sorted_padded, dtype=torch.int32)

    # Input hidden states
    src = torch.randn(M, hidden, dtype=torch.bfloat16)

    # --- gather: src[M, hidden] -> out[total_padded, hidden] using sorted_ids ---
    gather_out = torch.zeros(total_padded, hidden, dtype=torch.bfloat16)
    for pos in range(total_padded):
        tid = sorted_padded[pos]
        if tid < M:
            gather_out[pos] = src[tid]

    # --- scatter_fused: expert_out[total_padded, hidden] -> accum[M, hidden] ---
    # Simulate expert outputs (just use gather_out * 0.5 as proxy)
    expert_out = (gather_out.float() * 0.5).to(torch.bfloat16)
    scale_factor = 1.0

    accum_ref = torch.zeros(M, hidden, dtype=torch.float32)
    for pos in range(total_padded):
        tid = sorted_padded[pos]
        if tid < M:
            accum_ref[tid] += expert_out[pos].float() * scale_factor

    save_test("moe_gather_scatter_m32_h64_k4", {
        "src": (bf16_bytes(src), (M, hidden), "bf16"),
        "sorted_ids": (i32_bytes(sorted_ids_tensor), (total_padded,), "i32"),
        "expert_out": (bf16_bytes(expert_out), (total_padded, hidden), "bf16"),
    }, {
        "gather_out": (bf16_bytes(gather_out), (total_padded, hidden), "bf16"),
        "accum": (fp32_bytes(accum_ref), (M, hidden), "f32"),
    }, atol=1e-4, rtol=1e-4,
    note=f"MoE gather+scatter M={M} hidden={hidden} topk={topk} E={E} total_padded={total_padded}")


# ════════════════════════════════════════════════════════════════════════

ALL_GENERATORS = {
    "rmsnorm": gen_rmsnorm,
    "fused_add_rmsnorm": gen_fused_add_rmsnorm,
    "rope": gen_rope,
    "silu_mul": gen_silu_mul,
    "relu2": gen_relu2,
    "transpose_3d": gen_transpose_3d,
    "sigmoid_topk": gen_sigmoid_topk,
    "softmax_topk": gen_softmax_topk,
    "concat_3_bf16": gen_concat_3_bf16,
    "gated_q_split": gen_gated_q_split,
    "fp8_kv_roundtrip": gen_fp8_kv_roundtrip,
    "sigmoid_mul": gen_sigmoid_mul,
    "gqa_attention": gen_gqa_attention,
    # Tier 2: LA sub-components
    "la_l2norm": gen_la_l2norm,
    "la_gate_beta": gen_la_gate_beta,
    "la_gated_rmsnorm": gen_la_gated_rmsnorm,
    "la_conv1d": gen_la_conv1d,
    "la_uninterleave_qkvz": gen_la_uninterleave_qkvz,
    # Tier 2: LA complex ops
    "la_triangular_solve": gen_la_triangular_solve,
    "la_chunk_recurrence": gen_la_chunk_recurrence,
    # Tier 2: other pipeline kernels
    "embedding": gen_embedding,
    "moe_gather_scatter": gen_moe_gather_scatter,
}


def main():
    parser = argparse.ArgumentParser(description="Generate kernel test data")
    parser.add_argument("--kernel", type=str, help="Generate only this kernel's test data")
    parser.add_argument("--list", action="store_true", help="List available kernels")
    args = parser.parse_args()

    if args.list:
        for name in sorted(ALL_GENERATORS):
            print(f"  {name}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.kernel:
        if args.kernel not in ALL_GENERATORS:
            print(f"Unknown kernel: {args.kernel}")
            print(f"Available: {', '.join(sorted(ALL_GENERATORS))}")
            sys.exit(1)
        print(f"Generating test data for: {args.kernel}")
        ALL_GENERATORS[args.kernel]()
    else:
        print(f"Generating all kernel test data ({len(ALL_GENERATORS)} kernels)")
        for name, gen_fn in ALL_GENERATORS.items():
            print(f"\n[{name}]")
            gen_fn()

    total_bytes = sum(
        f.stat().st_size
        for f in OUT_DIR.rglob("*.bin")
    )
    print(f"\nTotal test data: {total_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
