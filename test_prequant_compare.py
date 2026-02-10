#!/usr/bin/env python3
"""Diagnostic: Compare pre-quantized compressed-tensors weights against our convention.

For Kimi K2.5 (compressed-tensors pack-quantized INT4):
  1. Load weight_packed, weight_scale from safetensors
  2. Dequantize using offset-8 convention (vectorized numpy)
  3. Compare INT4 matmul vs BF16 dequantized matmul
  4. Full expert forward: gate → up → SiLU → down

If the INT4 matmul closely matches the BF16 reference, the format is correct
and the Rust kernel should produce the same result. If not, the nibble packing
or scale layout is wrong.
"""

import numpy as np
import torch
import json
import os
import time

MODEL_PATH = "/home/main/Documents/Claude/hf-models/Kimi-K2.5"
GROUP_SIZE = 32


def load_safetensors_tensor(path, name):
    from safetensors import safe_open
    # Use torch for bfloat16 support, convert to numpy
    with safe_open(path, framework="pt", device="cpu") as f:
        t = f.get_tensor(name)
    # Convert bfloat16 → float32 for numpy
    if t.dtype == torch.bfloat16:
        return t.float().numpy()
    return t.numpy()


def get_shard_for_tensor(model_path, tensor_name):
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    shard = index["weight_map"][tensor_name]
    return os.path.join(model_path, shard)


def dequant_ct_vectorized(packed_int32, scales_f32, rows, cols, group_size=32):
    """Vectorized dequantization of compressed-tensors INT4.

    Each int32 word contains 8 nibbles: nibble_j at bits [4j, 4j+3].
    Dequantized value = (nibble - 8) * scale_for_group.
    """
    packed = packed_int32.astype(np.uint32).reshape(rows, cols // 8)

    # Extract all 8 nibbles from each word
    # shape: [rows, cols//8, 8]
    shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    nibbles = (packed[:, :, None] >> shifts[None, None, :]) & 0xF
    # Reshape to [rows, cols]
    nibbles = nibbles.reshape(rows, cols).astype(np.float32)

    # Apply offset: unsigned [0,15] → signed [-8,7]
    signed_vals = nibbles - 8.0

    # Broadcast scales: [rows, num_groups] → [rows, cols]
    num_groups = cols // group_size
    scales_expanded = np.repeat(scales_f32.reshape(rows, num_groups), group_size, axis=1)

    return signed_vals * scales_expanded


def matmul_int4_vectorized(packed_int32, scales_f32, activation, rows, cols, group_size=32):
    """Vectorized INT4 matmul simulating our Rust kernel.

    For each row, for each group:
      acc += sum(signed_nibble * activation) * scale
    """
    packed = packed_int32.astype(np.uint32).reshape(rows, cols // 8)
    num_groups = cols // group_size

    # Extract nibbles: [rows, cols]
    shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    nibbles = (packed[:, :, None] >> shifts[None, None, :]) & 0xF
    signed_vals = nibbles.reshape(rows, cols).astype(np.float32) - 8.0

    # Group-wise matmul: for each group, dot product of signed vals with activation, then scale
    output = np.zeros(rows, dtype=np.float32)
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        # [rows, group_size] @ [group_size] → [rows]
        group_dot = signed_vals[:, start:end] @ activation[start:end]
        # Scale per row for this group
        output += group_dot * scales_f32[:, g]

    return output


def load_expert_proj(layer_idx, expert_idx, proj_name):
    """Load one expert projection from compressed-tensors safetensors."""
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
    return packed, scales.astype(np.float32), r, c


def main():
    print("=== Pre-quantized Weight Format Diagnostic ===")
    print(f"Model: Kimi K2.5, group_size={GROUP_SIZE}")

    layer_idx = 1  # First MoE layer
    expert_idx = 0

    # Load gate_proj
    print(f"\n--- Loading expert {expert_idx}, layer {layer_idx} ---")
    t0 = time.time()
    gate_packed, gate_scales, gate_rows, gate_cols = load_expert_proj(layer_idx, expert_idx, "gate_proj")
    up_packed, up_scales, up_rows, up_cols = load_expert_proj(layer_idx, expert_idx, "up_proj")
    down_packed, down_scales, down_rows, down_cols = load_expert_proj(layer_idx, expert_idx, "down_proj")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print(f"  gate_proj: ({gate_rows}, {gate_cols})  packed={gate_packed.shape}  scales={gate_scales.shape}")
    print(f"  up_proj:   ({up_rows}, {up_cols})  packed={up_packed.shape}  scales={up_scales.shape}")
    print(f"  down_proj: ({down_rows}, {down_cols})  packed={down_packed.shape}  scales={down_scales.shape}")

    hidden_size = gate_cols  # 7168
    intermediate = gate_rows  # 2048

    # === Test 1: Nibble statistics ===
    print(f"\n--- Test 1: Nibble statistics ---")
    packed_u32 = gate_packed.astype(np.uint32).ravel()[:1000]
    shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    nibbles = (packed_u32[:, None] >> shifts[None, :]) & 0xF
    nibbles_flat = nibbles.ravel()
    signed_flat = nibbles_flat.astype(np.int32) - 8
    print(f"  Unsigned nibbles: mean={nibbles_flat.mean():.2f}, std={nibbles_flat.std():.2f}")
    print(f"  Signed (offset-8): mean={signed_flat.mean():.4f}, std={signed_flat.std():.2f}")
    print(f"  Distribution: {dict(zip(*np.unique(nibbles_flat, return_counts=True)))}")

    # === Test 2: Dequantize and check weight statistics ===
    print(f"\n--- Test 2: Dequantized weight statistics ---")
    t0 = time.time()
    gate_dequant = dequant_ct_vectorized(gate_packed, gate_scales, gate_rows, gate_cols, GROUP_SIZE)
    print(f"  Dequantized in {time.time()-t0:.2f}s")
    print(f"  Gate weights: mean={gate_dequant.mean():.6f}, std={gate_dequant.std():.6f}, "
          f"min={gate_dequant.min():.4f}, max={gate_dequant.max():.4f}")

    # === Test 3: INT4 matmul vs BF16 matmul ===
    print(f"\n--- Test 3: INT4 matmul vs BF16 dequantized matmul ---")
    np.random.seed(42)
    activation = np.random.randn(hidden_size).astype(np.float32) * 0.1

    # BF16 reference: dequantized_weights @ activation
    ref_gate = gate_dequant @ activation  # [intermediate]

    # INT4 matmul with our kernel logic
    int4_gate = matmul_int4_vectorized(gate_packed, gate_scales, activation,
                                        gate_rows, gate_cols, GROUP_SIZE)

    diff = np.abs(ref_gate - int4_gate)
    cos = np.dot(ref_gate, int4_gate) / (np.linalg.norm(ref_gate) * np.linalg.norm(int4_gate))
    print(f"  BF16 ref:  RMS={np.sqrt(np.mean(ref_gate**2)):.6f}")
    print(f"  INT4:      RMS={np.sqrt(np.mean(int4_gate**2)):.6f}")
    print(f"  Max diff:  {diff.max():.8f}")
    print(f"  Cosine:    {cos:.8f}")
    assert cos > 0.9999, f"INT4 matmul should match BF16 dequantized: cosine={cos}"
    print(f"  PASS: INT4 matmul matches BF16 reference (cosine={cos:.6f})")

    # === Test 4: Full expert forward ===
    print(f"\n--- Test 4: Full expert forward (BF16 ref vs INT4) ---")

    # Dequantize up and down
    t0 = time.time()
    up_dequant = dequant_ct_vectorized(up_packed, up_scales, up_rows, up_cols, GROUP_SIZE)
    down_dequant = dequant_ct_vectorized(down_packed, down_scales, down_rows, down_cols, GROUP_SIZE)
    print(f"  Dequantized all projections in {time.time()-t0:.2f}s")

    # BF16 reference expert forward
    gate_out = gate_dequant @ activation  # [intermediate]
    up_out = up_dequant @ activation      # [intermediate]
    silu = gate_out / (1.0 + np.exp(-gate_out))
    hidden_act = silu * up_out
    expert_ref = down_dequant @ hidden_act  # [hidden_size]

    # INT4 expert forward
    gate_int4 = matmul_int4_vectorized(gate_packed, gate_scales, activation,
                                        gate_rows, gate_cols, GROUP_SIZE)
    up_int4 = matmul_int4_vectorized(up_packed, up_scales, activation,
                                      up_rows, up_cols, GROUP_SIZE)
    silu_int4 = gate_int4 / (1.0 + np.exp(-gate_int4))
    hidden_int4 = silu_int4 * up_int4
    expert_int4 = matmul_int4_vectorized(down_packed, down_scales, hidden_int4,
                                          down_rows, down_cols, GROUP_SIZE)

    diff_expert = np.abs(expert_ref - expert_int4)
    cos_expert = np.dot(expert_ref, expert_int4) / (
        np.linalg.norm(expert_ref) * np.linalg.norm(expert_int4) + 1e-10)

    print(f"  BF16 ref:  RMS={np.sqrt(np.mean(expert_ref**2)):.6f}, mean={expert_ref.mean():.6f}")
    print(f"  INT4:      RMS={np.sqrt(np.mean(expert_int4**2)):.6f}, mean={expert_int4.mean():.6f}")
    print(f"  Max diff:  {diff_expert.max():.6f}")
    print(f"  Cosine:    {cos_expert:.6f}")

    # === Test 5: Compare integer kernel (INT16 activation) ===
    print(f"\n--- Test 5: Integer kernel (INT16 quantized activation) ---")

    # Simulate INT16 activation quantization
    num_groups = hidden_size // GROUP_SIZE
    act_int16 = np.zeros(hidden_size, dtype=np.int16)
    act_scales = np.zeros(num_groups, dtype=np.float32)

    for g in range(num_groups):
        start = g * GROUP_SIZE
        end = start + GROUP_SIZE
        group = activation[start:end]
        max_abs = np.max(np.abs(group))
        if max_abs > 0:
            scale = max_abs / 32767.0
            inv_scale = 32767.0 / max_abs
            act_scales[g] = scale
            act_int16[start:end] = np.clip(np.round(group * inv_scale), -32768, 32767).astype(np.int16)
        else:
            act_scales[g] = 1.0

    # Integer kernel: INT4_weight × INT16_activation × (weight_scale × act_scale)
    def matmul_int4_integer(packed_int32, weight_scales, act_i16, a_scales, rows_, cols_, gs=32):
        packed = packed_int32.astype(np.uint32).reshape(rows_, cols_ // 8)
        shifts_arr = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
        nibbles = (packed[:, :, None] >> shifts_arr[None, None, :]) & 0xF
        signed_vals = nibbles.reshape(rows_, cols_).astype(np.int32) - 8
        n_groups = cols_ // gs
        output = np.zeros(rows_, dtype=np.float32)
        for g_idx in range(n_groups):
            s = g_idx * gs
            e = s + gs
            # INT4 × INT16 → INT32 sum
            group_sum = (signed_vals[:, s:e].astype(np.int32) @
                         act_i16[s:e].astype(np.int32))  # [rows]
            combined_scale = weight_scales[:, g_idx] * a_scales[g_idx]
            output += group_sum.astype(np.float32) * combined_scale
        return output

    gate_int_out = matmul_int4_integer(gate_packed, gate_scales, act_int16, act_scales,
                                        gate_rows, gate_cols, GROUP_SIZE)

    diff_int = np.abs(ref_gate - gate_int_out)
    cos_int = np.dot(ref_gate, gate_int_out) / (
        np.linalg.norm(ref_gate) * np.linalg.norm(gate_int_out) + 1e-10)
    print(f"  Gate BF16 ref RMS:     {np.sqrt(np.mean(ref_gate**2)):.6f}")
    print(f"  Gate INT4×INT16 RMS:   {np.sqrt(np.mean(gate_int_out**2)):.6f}")
    print(f"  Max diff vs BF16 ref:  {diff_int.max():.8f}")
    print(f"  Cosine vs BF16 ref:    {cos_int:.8f}")

    # Full expert with integer kernel
    up_int_out = matmul_int4_integer(up_packed, up_scales, act_int16, act_scales,
                                      up_rows, up_cols, GROUP_SIZE)
    silu_int = gate_int_out / (1.0 + np.exp(-gate_int_out))
    hidden_int = silu_int * up_int_out

    # For down_proj, need to quantize the intermediate hidden state
    down_groups = intermediate // GROUP_SIZE
    hidden_int16 = np.zeros(intermediate, dtype=np.int16)
    hidden_scales = np.zeros(down_groups, dtype=np.float32)
    for g in range(down_groups):
        s = g * GROUP_SIZE
        e = s + GROUP_SIZE
        group = hidden_int[s:e]
        max_abs = np.max(np.abs(group))
        if max_abs > 0:
            sc = max_abs / 32767.0
            inv_sc = 32767.0 / max_abs
            hidden_scales[g] = sc
            hidden_int16[s:e] = np.clip(np.round(group * inv_sc), -32768, 32767).astype(np.int16)
        else:
            hidden_scales[g] = 1.0

    expert_int_out = matmul_int4_integer(down_packed, down_scales, hidden_int16, hidden_scales,
                                          down_rows, down_cols, GROUP_SIZE)

    diff_expert_int = np.abs(expert_ref - expert_int_out)
    cos_expert_int = np.dot(expert_ref, expert_int_out) / (
        np.linalg.norm(expert_ref) * np.linalg.norm(expert_int_out) + 1e-10)
    print(f"\n  Full expert (integer kernel):")
    print(f"  BF16 ref RMS:   {np.sqrt(np.mean(expert_ref**2)):.6f}")
    print(f"  Integer RMS:    {np.sqrt(np.mean(expert_int_out**2)):.6f}")
    print(f"  Max diff:       {diff_expert_int.max():.6f}")
    print(f"  Cosine:         {cos_expert_int:.6f}")

    # === Test 6: Sensitivity check — what if nibble order is reversed? ===
    print(f"\n--- Test 6: Nibble order sensitivity ---")

    # Try reversed nibble order: nibble_j at bits [(7-j)*4, (7-j)*4+3]
    packed_u32_gate = gate_packed.astype(np.uint32).reshape(gate_rows, gate_cols // 8)
    shifts_rev = np.array([28, 24, 20, 16, 12, 8, 4, 0], dtype=np.uint32)
    nibbles_rev = (packed_u32_gate[:, :, None] >> shifts_rev[None, None, :]) & 0xF
    signed_rev = nibbles_rev.reshape(gate_rows, gate_cols).astype(np.float32) - 8.0
    num_groups_gate = gate_cols // GROUP_SIZE
    scales_expanded = np.repeat(gate_scales.reshape(gate_rows, num_groups_gate), GROUP_SIZE, axis=1)
    gate_dequant_rev = signed_rev * scales_expanded

    ref_gate_rev = gate_dequant_rev @ activation

    cos_rev = np.dot(ref_gate, ref_gate_rev) / (
        np.linalg.norm(ref_gate) * np.linalg.norm(ref_gate_rev) + 1e-10)
    print(f"  Normal order cosine with itself: 1.000000")
    print(f"  Reversed order cosine with normal: {cos_rev:.6f}")
    if abs(cos_rev - 1.0) < 0.01:
        print("  NOTE: Reversed order gives similar results (symmetric weight distribution)")
    else:
        print(f"  Nibble order matters! Difference: {abs(cos_rev - 1.0):.6f}")

    # === Test 7: Two's complement convention test ===
    print(f"\n--- Test 7: Convention comparison (offset-8 vs two's complement) ---")

    packed_u32_flat = gate_packed.astype(np.uint32).ravel()[:1000]
    shifts_arr = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    nibbles_all = (packed_u32_flat[:, None] >> shifts_arr[None, :]) & 0xF

    # Offset-8: signed = nibble - 8 → range [-8, 7]
    offset8 = nibbles_all.astype(np.int32) - 8
    # Two's complement: if nibble >= 8, signed = nibble - 16 → range [-8, 7]
    twos_comp = np.where(nibbles_all >= 8, nibbles_all.astype(np.int32) - 16, nibbles_all.astype(np.int32))

    # Compare
    match = np.sum(offset8 == twos_comp)
    total = offset8.size
    print(f"  Offset-8 vs two's complement: {match}/{total} nibbles agree ({100*match/total:.1f}%)")
    print(f"  Offset-8 mean: {offset8.mean():.4f}, two's complement mean: {twos_comp.mean():.4f}")

    # They differ for nibbles 1-7 and 9-15:
    # offset-8: nibble 8 → 0, nibble 9 → 1, ..., nibble 15 → 7
    # two's comp: nibble 8 → -8, nibble 9 → -7, ..., nibble 15 → -1
    # So nibbles 8-15 are OPPOSITE sign between conventions!
    # Only nibble 0 (both -8) and nibble 8 (offset:0, tc:-8) match partially

    if match < total * 0.9:
        print("  The two conventions are DIFFERENT — choosing wrong one would corrupt output!")

        # Test matmul with two's complement
        nib_gate = (gate_packed.astype(np.uint32).reshape(gate_rows, gate_cols // 8)[:, :, None] >>
                    np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)[None, None, :]) & 0xF
        signed_tc = np.where(nib_gate >= 8, nib_gate.astype(np.int32) - 16,
                             nib_gate.astype(np.int32)).reshape(gate_rows, gate_cols).astype(np.float32)
        gate_dequant_tc = signed_tc * scales_expanded
        ref_gate_tc = gate_dequant_tc @ activation

        cos_tc = np.dot(ref_gate, ref_gate_tc) / (
            np.linalg.norm(ref_gate) * np.linalg.norm(ref_gate_tc) + 1e-10)
        print(f"  Matmul cosine (offset-8 vs two's complement): {cos_tc:.6f}")

    print("\n=== Diagnostic complete ===")


if __name__ == "__main__":
    main()
