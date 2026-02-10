#!/usr/bin/env python3
"""Direct comparison: Rust engine MoE output vs Python BF16 reference.

Loads 1 MoE layer in the Rust engine, sends a known activation through,
and compares against a Python-computed BF16 reference using the same
compressed-tensors weights.

This tests the FULL Rust pipeline: weight loading, kernel execution,
shared expert, routed_scaling_factor, etc.
"""

import numpy as np
import torch
import json
import os
import time
import struct

MODEL_PATH = "/home/main/Documents/Claude/hf-models/Kimi-K2.5"
GROUP_SIZE = 32


def load_safetensors_tensor(path, name):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        t = f.get_tensor(name)
    if t.dtype == torch.bfloat16:
        return t.float().numpy()
    return t.numpy()


def get_shard_for_tensor(model_path, tensor_name):
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    shard = index["weight_map"][tensor_name]
    return os.path.join(model_path, shard)


def dequant_ct(packed_int32, scales_f32, rows, cols, gs=32):
    """Vectorized dequantize compressed-tensors INT4 to FP32."""
    packed = packed_int32.astype(np.uint32).reshape(rows, cols // 8)
    shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    nibbles = (packed[:, :, None] >> shifts[None, None, :]) & 0xF
    signed = nibbles.reshape(rows, cols).astype(np.float32) - 8.0
    num_groups = cols // gs
    scales_expanded = np.repeat(scales_f32.reshape(rows, num_groups), gs, axis=1)
    return signed * scales_expanded


def load_expert_proj(layer_idx, expert_idx, proj_name):
    prefix = f"language_model.model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}"
    packed = load_safetensors_tensor(
        get_shard_for_tensor(MODEL_PATH, f"{prefix}.weight_packed"),
        f"{prefix}.weight_packed"
    )
    scales = load_safetensors_tensor(
        get_shard_for_tensor(MODEL_PATH, f"{prefix}.weight_scale"),
        f"{prefix}.weight_scale"
    )
    shape = load_safetensors_tensor(
        get_shard_for_tensor(MODEL_PATH, f"{prefix}.weight_shape"),
        f"{prefix}.weight_shape"
    )
    r, c = int(shape[0]), int(shape[1])
    return packed, scales, r, c


def load_shared_expert_proj(layer_idx, proj_name):
    """Load shared expert projection as BF16 weight tensor."""
    name = f"language_model.model.layers.{layer_idx}.mlp.shared_experts.{proj_name}.weight"
    path = get_shard_for_tensor(MODEL_PATH, name)
    data = load_safetensors_tensor(path, name)
    return data


def expert_forward_bf16(gate_w, up_w, down_w, activation):
    """BF16 reference expert forward: SiLU(gate(x)) * up(x) → down()."""
    gate_out = gate_w @ activation
    up_out = up_w @ activation
    silu = gate_out / (1.0 + np.exp(-gate_out))
    hidden = silu * up_out
    return down_w @ hidden


def main():
    print("=== Rust Engine vs Python BF16 Reference ===")
    print(f"Model: Kimi K2.5")

    layer_idx = 1  # First MoE layer (moe_layer_idx=0)
    hidden_size = 7168
    intermediate = 2048
    n_experts = 384
    topk = 8
    routed_scaling_factor = 2.827

    # Create reproducible activation (BF16 range)
    np.random.seed(42)
    activation_f32 = np.random.randn(hidden_size).astype(np.float32) * 0.05
    # Convert to BF16 and back to simulate actual precision
    activation_bf16 = torch.tensor(activation_f32).bfloat16()
    activation_f32 = activation_bf16.float().numpy()

    # Use specific expert IDs
    expert_ids = [10, 42, 100, 150, 200, 250, 300, 350]
    # Equal weights (after normalization)
    expert_weights = [1.0 / topk] * topk

    # === Part 1: Python BF16 Reference ===
    print("\n--- Part 1: Python BF16 reference ---")

    # Load and dequantize all 8 experts
    routed_output = np.zeros(hidden_size, dtype=np.float32)
    for i, eid in enumerate(expert_ids):
        t0 = time.time()
        gp, gs, gr, gc = load_expert_proj(layer_idx, eid, "gate_proj")
        up, us, ur, uc = load_expert_proj(layer_idx, eid, "up_proj")
        dp, ds, dr, dc = load_expert_proj(layer_idx, eid, "down_proj")

        gate_w = dequant_ct(gp, gs, gr, gc)
        up_w = dequant_ct(up, us, ur, uc)
        down_w = dequant_ct(dp, ds, dr, dc)

        expert_out = expert_forward_bf16(gate_w, up_w, down_w, activation_f32)
        routed_output += expert_weights[i] * expert_out
        print(f"  Expert {eid}: {time.time()-t0:.1f}s, out_RMS={np.sqrt(np.mean(expert_out**2)):.6f}")

    # Load and compute shared expert
    print("  Loading shared expert...")
    t0 = time.time()
    sh_gate = load_shared_expert_proj(layer_idx, "gate_proj")
    sh_up = load_shared_expert_proj(layer_idx, "up_proj")
    sh_down = load_shared_expert_proj(layer_idx, "down_proj")
    shared_out = expert_forward_bf16(sh_gate, sh_up, sh_down, activation_f32)
    print(f"  Shared expert: {time.time()-t0:.1f}s, out_RMS={np.sqrt(np.mean(shared_out**2)):.6f}")
    print(f"  Shared gate shape: {sh_gate.shape}, up: {sh_up.shape}, down: {sh_down.shape}")

    # Final: output = routed_scaling_factor * routed + shared
    ref_output = routed_scaling_factor * routed_output + shared_out

    ref_rms = np.sqrt(np.mean(ref_output**2))
    print(f"\n  Python reference output:")
    print(f"    RMS={ref_rms:.6f}, mean={ref_output.mean():.6f}")
    print(f"    Routed RMS={np.sqrt(np.mean(routed_output**2)):.6f}")
    print(f"    Shared RMS={np.sqrt(np.mean(shared_out**2)):.6f}")

    # === Part 2: Rust Engine ===
    print("\n--- Part 2: Rust engine (loading 1 MoE layer) ---")
    from krasis import KrasisEngine

    t0 = time.time()
    engine = KrasisEngine(parallel=True, num_threads=16)
    # Load just 1 layer (the first MoE layer, which is layer 1 in the model)
    engine.load(MODEL_PATH, max_layers=1, start_layer=0, num_bits=4)
    print(f"  Engine loaded in {time.time()-t0:.1f}s")
    print(f"  MoE layers: {engine.num_moe_layers()}, experts: {engine.num_experts()}, "
          f"hidden: {engine.hidden_size()}, topk: {engine.top_k()}")

    # Call synchronous moe_forward
    act_bytes = activation_bf16.view(torch.uint16).numpy().view(np.uint8).tobytes()

    t0 = time.time()
    result_bytes = engine.moe_forward(
        0,  # moe_layer_idx
        act_bytes,
        expert_ids,
        expert_weights,
    )
    print(f"  Synchronous moe_forward: {(time.time()-t0)*1000:.1f} ms")

    # Parse result (f32 bytes → numpy)
    rust_output_f32 = np.frombuffer(result_bytes, dtype=np.float32)
    print(f"  Output shape: {rust_output_f32.shape}, expected: ({hidden_size},)")

    # Wait — the sync moe_forward returns f32, but we need to check if shared expert
    # is included (it should be, since the engine loaded shared experts too)

    # === Part 3: Compare ===
    print("\n--- Part 3: Comparison ---")
    diff = np.abs(ref_output - rust_output_f32)
    cos = np.dot(ref_output, rust_output_f32) / (
        np.linalg.norm(ref_output) * np.linalg.norm(rust_output_f32) + 1e-10)

    print(f"  Python ref RMS: {ref_rms:.6f}")
    print(f"  Rust RMS:       {np.sqrt(np.mean(rust_output_f32**2)):.6f}")
    print(f"  Max diff:       {diff.max():.6f}")
    print(f"  Mean diff:      {diff.mean():.6f}")
    print(f"  Cosine sim:     {cos:.6f}")

    # Also test the async path (submit/sync)
    print("\n--- Part 4: Async path (submit_forward / sync_forward) ---")
    ids_i32 = np.array(expert_ids, dtype=np.int32)
    wts_f32 = np.array(expert_weights, dtype=np.float32)
    ids_bytes = ids_i32.view(np.uint8).tobytes()
    wts_bytes = wts_f32.view(np.uint8).tobytes()

    t0 = time.time()
    engine.submit_forward(0, act_bytes, ids_bytes, wts_bytes, 1)
    async_result_bytes = engine.sync_forward()
    print(f"  Async round-trip: {(time.time()-t0)*1000:.1f} ms")

    # Async returns BF16 (not f32!)
    async_bf16 = torch.frombuffer(bytearray(async_result_bytes), dtype=torch.bfloat16)
    async_f32 = async_bf16.float().numpy()

    diff_async = np.abs(ref_output - async_f32)
    cos_async = np.dot(ref_output, async_f32) / (
        np.linalg.norm(ref_output) * np.linalg.norm(async_f32) + 1e-10)

    print(f"  Async RMS:      {np.sqrt(np.mean(async_f32**2)):.6f}")
    print(f"  Max diff:       {diff_async.max():.6f}")
    print(f"  Cosine sim:     {cos_async:.6f}")

    # Compare sync vs async (should be very close, just BF16 rounding)
    diff_sa = np.abs(rust_output_f32 - async_f32)
    print(f"  Sync vs Async max diff: {diff_sa.max():.6f}")

    if cos > 0.99:
        print("\n  PASS: Rust engine output closely matches Python BF16 reference")
    elif cos > 0.9:
        print("\n  WARNING: Moderate divergence — may cause degraded generation quality")
    else:
        print("\n  FAIL: Rust engine output significantly diverges from reference")

    print("\n=== Test complete ===")


if __name__ == "__main__":
    main()
