#!/usr/bin/env python3
"""Compare Rust engine output: cache-loaded (full model) vs safetensors-loaded (partial).

If the cache is corrupted, these will differ for the same layer/activation.
"""

import numpy as np
import torch
import time

MODEL_PATH = "/home/main/Documents/Claude/hf-models/Kimi-K2.5"

def main():
    from krasis import KrasisEngine

    # Create reproducible activation
    np.random.seed(42)
    activation_f32 = np.random.randn(7168).astype(np.float32) * 0.05
    activation_bf16 = torch.tensor(activation_f32).bfloat16()
    act_bytes = activation_bf16.view(torch.uint16).numpy().view(np.uint8).tobytes()

    expert_ids = [10, 42, 100, 150, 200, 250, 300, 350]
    expert_weights = [1.0 / 8] * 8

    # ── Test 1: Load 1 layer from safetensors (partial, bypasses cache) ──
    print("--- Loading 1 layer from safetensors (partial, no cache) ---")
    t0 = time.time()
    engine_partial = KrasisEngine(parallel=True, num_threads=16)
    engine_partial.load(MODEL_PATH, max_layers=1, start_layer=0, num_bits=4)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    result_partial = engine_partial.moe_forward(0, act_bytes, expert_ids, expert_weights)
    out_partial = np.frombuffer(result_partial, dtype=np.float32)
    print(f"  Partial output: RMS={np.sqrt(np.mean(out_partial**2)):.6f}")

    del engine_partial

    # ── Test 2: Load full model from cache ──
    print("\n--- Loading FULL model from cache ---")
    t0 = time.time()
    engine_full = KrasisEngine(parallel=True, num_threads=16)
    engine_full.load(MODEL_PATH, num_bits=4)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  MoE layers: {engine_full.num_moe_layers()}")

    # Test layer 0 (same as partial)
    result_full_l0 = engine_full.moe_forward(0, act_bytes, expert_ids, expert_weights)
    out_full_l0 = np.frombuffer(result_full_l0, dtype=np.float32)
    print(f"  Full L0 output: RMS={np.sqrt(np.mean(out_full_l0**2)):.6f}")

    # Compare partial vs full (same layer 0)
    diff = np.abs(out_partial - out_full_l0)
    cos = np.dot(out_partial, out_full_l0) / (
        np.linalg.norm(out_partial) * np.linalg.norm(out_full_l0) + 1e-10)
    print(f"\n  Layer 0 comparison (partial vs cache-loaded):")
    print(f"    Max diff:   {diff.max():.8f}")
    print(f"    Mean diff:  {diff.mean():.8f}")
    print(f"    Cosine sim: {cos:.8f}")

    if cos > 0.9999:
        print("    MATCH: Cache layer 0 matches safetensors layer 0")
    elif cos > 0.99:
        print("    CLOSE: Small differences (quantization rounding?)")
    else:
        print("    MISMATCH: Cache data DIFFERS from safetensors!")

    # ── Test 3: Test a later layer (layer 30, mid-model) ──
    print("\n--- Testing later layers from cache ---")
    for test_layer in [0, 10, 29, 59]:
        result = engine_full.moe_forward(test_layer, act_bytes, expert_ids, expert_weights)
        out = np.frombuffer(result, dtype=np.float32)
        rms = np.sqrt(np.mean(out**2))
        has_nan = np.any(np.isnan(out))
        has_inf = np.any(np.isinf(out))
        print(f"  Layer {test_layer:2d}: RMS={rms:.6f}, nan={has_nan}, inf={has_inf}, "
              f"mean={out.mean():.6f}, max_abs={np.abs(out).max():.6f}")

    # ── Test 4: Run 60 layers sequentially (simulate decode) ──
    print("\n--- Simulating decode: 60 MoE layers sequentially ---")
    # Start with a BF16 hidden state
    hidden_f32 = activation_f32.copy()

    for layer in range(engine_full.num_moe_layers()):
        # Convert to BF16 bytes
        hidden_bf16 = torch.tensor(hidden_f32).bfloat16()
        h_bytes = hidden_bf16.view(torch.uint16).numpy().view(np.uint8).tobytes()

        result = engine_full.moe_forward(layer, h_bytes, expert_ids, expert_weights)
        moe_out = np.frombuffer(result, dtype=np.float32)

        # In real model: residual += hidden; hidden = norm(residual); moe_out added to residual
        # Here we just track the MoE output magnitude
        hidden_f32 = moe_out  # simplified: pass output to next layer

        rms = np.sqrt(np.mean(moe_out**2))
        has_nan = np.any(np.isnan(moe_out))
        if layer % 10 == 0 or layer == engine_full.num_moe_layers() - 1 or has_nan:
            print(f"  Layer {layer:2d}: RMS={rms:.6f}, nan={has_nan}")

    print("\nDone.")


if __name__ == "__main__":
    main()
