#!/usr/bin/env python3
"""Quick MoE sanity check: verify Rust engine output is reasonable."""

import logging
import time
import sys
import struct
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("moe_sanity")

MODEL_PATH = "/home/main/Documents/Claude/hf-models/Kimi-K2.5"

def bf16_to_float(bf16_bytes):
    """Convert BF16 bytes to float32 numpy array."""
    raw = np.frombuffer(bf16_bytes, dtype=np.uint16)
    # BF16 -> FP32: shift left 16 bits
    fp32_int = raw.astype(np.uint32) << 16
    return fp32_int.view(np.float32)

def main():
    import torch

    logger.info("Loading Krasis engine...")
    from krasis import KrasisEngine
    engine = KrasisEngine(parallel=True, num_threads=16)

    t0 = time.time()
    engine.load(MODEL_PATH, num_bits=4)
    logger.info("Engine loaded in %.1fs", time.time() - t0)
    logger.info("MoE layers: %d, experts: %d, hidden: %d",
                engine.num_moe_layers(), engine.num_experts(), engine.hidden_size())

    hidden = engine.hidden_size()

    # Test 1: Zero activation — output should be near zero
    logger.info("--- Test 1: Zero activation ---")
    zero_act = b'\x00' * (hidden * 2)
    zero_ids = struct.pack("i" * 8, *range(8))  # expert IDs 0-7
    zero_wts = struct.pack("f" * 8, *([0.125] * 8))  # equal weights

    engine.submit_forward(0, zero_act, zero_ids, zero_wts, 1)
    out = engine.sync_forward()
    out_arr = bf16_to_float(out)
    logger.info("Zero input: mean=%.6f std=%.6f max=%.6f", out_arr.mean(), out_arr.std(), np.abs(out_arr).max())

    # Test 2: Small constant activation — check output magnitude
    logger.info("--- Test 2: Small constant activation ---")
    val = 0.01
    bf16_bytes = struct.pack("e", val)  # Python 'e' = IEEE 754 half-float, NOT bf16
    # Actually let's use torch for BF16
    act_tensor = torch.full((1, hidden), val, dtype=torch.bfloat16)
    act_bytes = act_tensor.view(torch.uint16).numpy().view(np.uint8).tobytes()

    ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int32)
    wts = torch.tensor([[0.125] * 8], dtype=torch.float32)
    ids_bytes = ids.numpy().view(np.uint8).tobytes()
    wts_bytes = wts.numpy().view(np.uint8).tobytes()

    engine.submit_forward(0, act_bytes, ids_bytes, wts_bytes, 1)
    out = engine.sync_forward()
    out_arr = bf16_to_float(out)
    logger.info("Constant %.3f input: mean=%.6f std=%.6f max=%.6f rms=%.6f",
                val, out_arr.mean(), out_arr.std(), np.abs(out_arr).max(),
                np.sqrt(np.mean(out_arr**2)))

    # Test 3: Random activation (normal distribution, typical scale)
    logger.info("--- Test 3: Random normal activation (std=0.003) ---")
    act_tensor = torch.randn(1, hidden, dtype=torch.float32) * 0.003
    act_tensor = act_tensor.to(torch.bfloat16)
    act_bytes = act_tensor.view(torch.uint16).numpy().view(np.uint8).tobytes()

    # Test across multiple layers
    for layer_idx in [0, 1, 29, 59]:
        if layer_idx >= engine.num_moe_layers():
            continue
        engine.submit_forward(layer_idx, act_bytes, ids_bytes, wts_bytes, 1)
        out = engine.sync_forward()
        out_arr = bf16_to_float(out)
        logger.info("  Layer %d: mean=%.6f std=%.6f max=%.6f rms=%.6f",
                    layer_idx, out_arr.mean(), out_arr.std(), np.abs(out_arr).max(),
                    np.sqrt(np.mean(out_arr**2)))

    # Test 4: Determinism — same input twice should give same output
    logger.info("--- Test 4: Determinism check ---")
    engine.submit_forward(0, act_bytes, ids_bytes, wts_bytes, 1)
    out1 = engine.sync_forward()
    engine.submit_forward(0, act_bytes, ids_bytes, wts_bytes, 1)
    out2 = engine.sync_forward()
    arr1 = bf16_to_float(out1)
    arr2 = bf16_to_float(out2)
    diff = np.abs(arr1 - arr2)
    logger.info("Determinism: max_diff=%.8f, mean_diff=%.8f", diff.max(), diff.mean())

    # Test 5: Weight sensitivity — different experts should give different outputs
    logger.info("--- Test 5: Expert sensitivity ---")
    for exp_id in [0, 1, 100, 383]:
        ids_one = struct.pack("i", exp_id)
        wts_one = struct.pack("f", 1.0)
        engine.submit_forward(0, act_bytes, ids_one, wts_one, 1)
        out = engine.sync_forward()
        out_arr = bf16_to_float(out)
        logger.info("  Expert %d: mean=%.6f std=%.6f rms=%.6f",
                    exp_id, out_arr.mean(), out_arr.std(), np.sqrt(np.mean(out_arr**2)))

    # Test 6: Check that shared expert is included (output should differ from no-shared-expert)
    logger.info("--- Test 6: Shared expert contribution ---")
    # Shared expert is always included by the engine
    # Compare output to see it's not just the routed expert
    engine.submit_forward(0, act_bytes, ids_bytes, wts_bytes, 1)
    out_full = engine.sync_forward()
    full_arr = bf16_to_float(out_full)

    # Single expert with weight 1.0 — output = scale * 1.0 * expert_0(x) + shared(x)
    ids_single = struct.pack("i", 0)
    wts_single = struct.pack("f", 1.0)
    engine.submit_forward(0, act_bytes, ids_single, wts_single, 1)
    out_single = engine.sync_forward()
    single_arr = bf16_to_float(out_single)

    diff = np.abs(full_arr - single_arr)
    logger.info("Full (8 experts) vs single (expert 0): max_diff=%.6f, mean_diff=%.6f",
                diff.max(), diff.mean())
    logger.info("  Full rms=%.6f, Single rms=%.6f",
                np.sqrt(np.mean(full_arr**2)), np.sqrt(np.mean(single_arr**2)))

    logger.info("=== Sanity checks complete ===")

if __name__ == "__main__":
    main()
