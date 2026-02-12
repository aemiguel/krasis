#!/usr/bin/env python3
"""Integration test for Krasis PyO3 bindings."""

import struct
import time
import os

# Test 1: Import
print("=== Test 1: Import ===")
from krasis import KrasisEngine
print(f"  KrasisEngine imported OK: {KrasisEngine}")

# Test 2: Create engine
print("\n=== Test 2: Create engine ===")
engine = KrasisEngine(parallel=True)
print(f"  Created engine, parallel={engine.is_parallel()}")

# Test 3: Error before loading
print("\n=== Test 3: Error handling ===")
try:
    engine.hidden_size()
    print("  ERROR: should have raised")
except RuntimeError as e:
    print(f"  Correct error: {e}")

# Test 4: Load V2-Lite model
MODEL_DIR = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"
if not os.path.exists(MODEL_DIR):
    print(f"\nSkipping model tests — {MODEL_DIR} not found")
    exit(0)

print(f"\n=== Test 4: Load model ===")
start = time.time()
engine.load(MODEL_DIR, group_size=128)
elapsed = time.time() - start
print(f"  Loaded in {elapsed:.1f}s")
print(f"  hidden_size={engine.hidden_size()}")
print(f"  num_experts={engine.num_experts()}")
print(f"  top_k={engine.top_k()}")
print(f"  num_moe_layers={engine.num_moe_layers()}")

assert engine.hidden_size() == 2048
assert engine.num_experts() == 64
assert engine.top_k() == 6
assert engine.num_moe_layers() == 26

# Test 5: MoE forward (single token)
print("\n=== Test 5: MoE forward ===")

# Create synthetic BF16 activation [2048]
hidden_size = engine.hidden_size()
activation_bf16 = bytearray(hidden_size * 2)
for i in range(hidden_size):
    # BF16 encoding: small random-ish values
    val = ((i * 7 + 13) / hidden_size - 0.5) * 0.1
    # Convert f32 to BF16: take upper 16 bits of f32
    f32_bits = struct.pack('f', val)
    bf16_bits = f32_bits[2:4]  # upper 2 bytes
    activation_bf16[i * 2] = bf16_bits[0]
    activation_bf16[i * 2 + 1] = bf16_bits[1]

expert_indices = [0, 1, 2, 3, 4, 5]  # top-6
expert_weights = [1.0/6] * 6  # uniform

start = time.time()
output_bytes = engine.moe_forward(0, bytes(activation_bf16), expert_indices, expert_weights)
elapsed_us = (time.time() - start) * 1e6

# Parse f32 output
assert len(output_bytes) == hidden_size * 4, f"Expected {hidden_size * 4} bytes, got {len(output_bytes)}"
output = struct.unpack(f'{hidden_size}f', output_bytes)

# Check output statistics
rms = (sum(v**2 for v in output) / hidden_size) ** 0.5
nonzero = sum(1 for v in output if abs(v) > 1e-10)
print(f"  Output: {len(output)} floats")
print(f"  RMS: {rms:.6f}")
print(f"  Non-zero: {nonzero}/{hidden_size}")
print(f"  Time: {elapsed_us:.0f} μs ({elapsed_us/1000:.1f} ms)")
print(f"  First 5 values: {[f'{v:.6f}' for v in output[:5]]}")

assert rms > 1e-5, f"Output RMS too small: {rms}"
assert nonzero > hidden_size / 2, f"Too many zeros: {nonzero}/{hidden_size}"

# Test 6: Validation errors
print("\n=== Test 6: Input validation ===")
try:
    engine.moe_forward(0, b"too_short", [0], [1.0])
    print("  ERROR: should have raised ValueError")
except ValueError as e:
    print(f"  Correct size error: {e}")

try:
    engine.moe_forward(0, bytes(activation_bf16), [0, 1], [1.0])
    print("  ERROR: should have raised ValueError")
except ValueError as e:
    print(f"  Correct mismatch error: {e}")

# Test 7: Benchmark multiple forward passes
print("\n=== Test 7: Benchmark ===")
warmup = 3
iters = 20

for _ in range(warmup):
    engine.moe_forward(0, bytes(activation_bf16), expert_indices, expert_weights)

start = time.time()
for _ in range(iters):
    engine.moe_forward(0, bytes(activation_bf16), expert_indices, expert_weights)
total = time.time() - start

per_call_ms = total / iters * 1000
print(f"  {iters} iterations: {total:.3f}s total")
print(f"  Per MoE forward: {per_call_ms:.1f} ms")
print(f"  Per expert: {per_call_ms / 6:.1f} ms")

print("\n=== All tests passed! ===")
