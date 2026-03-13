#!/usr/bin/env python3
"""
Analyze expert weight deduplication potential in the CPU expert cache.

Scans the experts_cpu_int4_g128.bin file directly via mmap and computes:
1. Exact duplicate experts (bitwise identical)
2. Scale-only deduplication (identical scale vectors across experts)
3. Cosine similarity distribution between expert pairs (sampled)
4. Row-level scale sharing potential
"""

import mmap
import struct
import numpy as np
import hashlib
import sys
import time
from collections import defaultdict

CACHE_DIR = "/home/main/.krasis/cache/Qwen3-Coder-Next"
CACHE_FILE = f"{CACHE_DIR}/experts_cpu_int4_g128.bin"

# QCN dimensions
HIDDEN = 2048
INTERMEDIATE = 512
GROUP_SIZE = 128
NUM_EXPERTS = 512
NUM_BITS = 4

# Compute per-expert sizes (INT4, group_size=128)
# w13 = gate+up fused: [2*INTERMEDIATE, HIDDEN] = [1024, 2048]
W13_N = 2 * INTERMEDIATE  # 1024
W13_K = HIDDEN             # 2048
W13_PACKED_U32 = (W13_K // 8) * W13_N  # 256 * 1024 = 262144 u32 = 1,048,576 bytes
W13_SCALES_U16 = (W13_K // GROUP_SIZE) * W13_N  # 16 * 1024 = 16384 u16 = 32,768 bytes

# w2 = down: [HIDDEN, INTERMEDIATE] = [2048, 512]
W2_N = HIDDEN       # 2048
W2_K = INTERMEDIATE  # 512
W2_PACKED_U32 = (W2_K // 8) * W2_N  # 64 * 2048 = 131072 u32 = 524,288 bytes
W2_SCALES_U16 = (W2_K // GROUP_SIZE) * W2_N  # 4 * 2048 = 8192 u16 = 16,384 bytes

W13_PACKED_BYTES = W13_PACKED_U32 * 4
W13_SCALES_BYTES = W13_SCALES_U16 * 2
W2_PACKED_BYTES = W2_PACKED_U32 * 4
W2_SCALES_BYTES = W2_SCALES_U16 * 2

EXPERT_BYTES = W13_PACKED_BYTES + W13_SCALES_BYTES + W2_PACKED_BYTES + W2_SCALES_BYTES
HEADER_SIZE = 64

print(f"Expert sizes:")
print(f"  w13_packed: {W13_PACKED_BYTES:,} bytes ({W13_PACKED_BYTES/1024:.1f} KB)")
print(f"  w13_scales: {W13_SCALES_BYTES:,} bytes ({W13_SCALES_BYTES/1024:.1f} KB)")
print(f"  w2_packed:  {W2_PACKED_BYTES:,} bytes ({W2_PACKED_BYTES/1024:.1f} KB)")
print(f"  w2_scales:  {W2_SCALES_BYTES:,} bytes ({W2_SCALES_BYTES/1024:.1f} KB)")
print(f"  TOTAL:      {EXPERT_BYTES:,} bytes ({EXPERT_BYTES/1024:.1f} KB)")
print(f"  All experts: {NUM_EXPERTS * 48 * EXPERT_BYTES / 1e9:.2f} GB")
print()


def expert_offset(layer, expert):
    """Byte offset of expert data in the mmap'd file."""
    return HEADER_SIZE + (layer * NUM_EXPERTS + expert) * EXPERT_BYTES


def get_expert_components(mm, layer, expert):
    """Return (w13_packed, w13_scales, w2_packed, w2_scales) as bytes views."""
    base = expert_offset(layer, expert)
    o = base
    w13p = mm[o:o+W13_PACKED_BYTES]; o += W13_PACKED_BYTES
    w13s = mm[o:o+W13_SCALES_BYTES]; o += W13_SCALES_BYTES
    w2p = mm[o:o+W2_PACKED_BYTES]; o += W2_PACKED_BYTES
    w2s = mm[o:o+W2_SCALES_BYTES]; o += W2_SCALES_BYTES
    return w13p, w13s, w2p, w2s


def hash_bytes(data):
    return hashlib.md5(data).digest()


def main():
    print(f"Opening {CACHE_FILE}...")
    with open(CACHE_FILE, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    # Verify header
    magic = mm[0:8]
    version = struct.unpack('<Q', mm[8:16])[0]
    hidden = struct.unpack('<Q', mm[16:24])[0]
    intermediate = struct.unpack('<Q', mm[24:32])[0]
    n_experts = struct.unpack('<Q', mm[32:40])[0]
    n_layers = struct.unpack('<Q', mm[40:48])[0]
    gs = struct.unpack('<Q', mm[48:56])[0]

    print(f"Header: version={version}, hidden={hidden}, intermediate={intermediate}")
    print(f"  experts={n_experts}, moe_layers={n_layers}, group_size={gs}")
    print(f"  Expected file size: {HEADER_SIZE + n_layers * n_experts * EXPERT_BYTES:,} bytes")
    print(f"  Actual file size: {mm.size():,} bytes")
    print()

    NUM_LAYERS = int(n_layers)

    # ── Analysis 1: Exact expert duplicates ──
    print("=" * 70)
    print("ANALYSIS 1: Exact expert duplicates (bitwise identical)")
    print("=" * 70)
    start = time.time()

    exact_dupes_total = 0
    exact_dupes_by_layer = []

    for layer in range(NUM_LAYERS):
        hashes = {}
        dupes_in_layer = 0
        for e in range(NUM_EXPERTS):
            base = expert_offset(layer, e)
            h = hash_bytes(mm[base:base+EXPERT_BYTES])
            if h in hashes:
                dupes_in_layer += 1
            else:
                hashes[h] = e
        exact_dupes_by_layer.append(dupes_in_layer)
        exact_dupes_total += dupes_in_layer
        if (layer + 1) % 12 == 0 or layer == NUM_LAYERS - 1:
            print(f"  Layer {layer}: {len(hashes)} unique / {NUM_EXPERTS} total ({dupes_in_layer} dupes)")

    print(f"\nTotal exact duplicates: {exact_dupes_total} / {NUM_LAYERS * NUM_EXPERTS}")
    savings_bytes = exact_dupes_total * EXPERT_BYTES
    print(f"Potential savings: {savings_bytes / 1e9:.2f} GB ({100*exact_dupes_total/(NUM_LAYERS*NUM_EXPERTS):.1f}%)")
    print(f"Time: {time.time()-start:.1f}s\n")

    # ── Analysis 2: Scale deduplication ──
    print("=" * 70)
    print("ANALYSIS 2: Scale vector deduplication")
    print("=" * 70)
    start = time.time()

    # Check w13_scales and w2_scales separately
    w13s_unique_by_layer = []
    w2s_unique_by_layer = []
    total_w13s_unique = 0
    total_w2s_unique = 0

    # Also check scale row dedup (each row's scale vector)
    w13_scale_rows_total = 0
    w13_scale_rows_unique = 0
    w2_scale_rows_total = 0
    w2_scale_rows_unique = 0

    for layer in range(NUM_LAYERS):
        w13s_hashes = set()
        w2s_hashes = set()

        # Row-level dedup for this layer
        w13_row_hashes = set()
        w2_row_hashes = set()

        for e in range(NUM_EXPERTS):
            _, w13s, _, w2s = get_expert_components(mm, layer, e)
            w13s_hashes.add(hash_bytes(w13s))
            w2s_hashes.add(hash_bytes(w2s))

            # Row-level: w13 scales are [K/gs, 2*N] = [16, 1024] in u16
            # Each row is 1024 * 2 = 2048 bytes
            w13_row_size = W13_N * 2  # 2048 bytes per scale row
            for r in range(W13_K // GROUP_SIZE):  # 16 rows
                row_data = w13s[r*w13_row_size:(r+1)*w13_row_size]
                w13_row_hashes.add(hash_bytes(row_data))

            w2_row_size = W2_N * 2  # 4096 bytes per scale row
            for r in range(W2_K // GROUP_SIZE):  # 4 rows
                row_data = w2s[r*w2_row_size:(r+1)*w2_row_size]
                w2_row_hashes.add(hash_bytes(row_data))

        w13s_unique_by_layer.append(len(w13s_hashes))
        w2s_unique_by_layer.append(len(w2s_hashes))

        w13_rows_this = NUM_EXPERTS * (W13_K // GROUP_SIZE)  # 512 * 16 = 8192
        w2_rows_this = NUM_EXPERTS * (W2_K // GROUP_SIZE)    # 512 * 4 = 2048

        w13_scale_rows_total += w13_rows_this
        w13_scale_rows_unique += len(w13_row_hashes)
        w2_scale_rows_total += w2_rows_this
        w2_scale_rows_unique += len(w2_row_hashes)

        if (layer + 1) % 12 == 0 or layer == NUM_LAYERS - 1:
            print(f"  Layer {layer}: w13_scales {len(w13s_hashes)} unique, w2_scales {len(w2s_hashes)} unique / {NUM_EXPERTS}")
            print(f"    w13 scale rows: {len(w13_row_hashes)} unique / {w13_rows_this}")
            print(f"    w2 scale rows:  {len(w2_row_hashes)} unique / {w2_rows_this}")

    total_scale_bytes = NUM_LAYERS * NUM_EXPERTS * (W13_SCALES_BYTES + W2_SCALES_BYTES)
    avg_w13s_unique = np.mean(w13s_unique_by_layer)
    avg_w2s_unique = np.mean(w2s_unique_by_layer)

    # Per-expert scale savings: if we share across experts in same layer
    w13s_savings = NUM_LAYERS * (NUM_EXPERTS - avg_w13s_unique) * W13_SCALES_BYTES
    w2s_savings = NUM_LAYERS * (NUM_EXPERTS - avg_w2s_unique) * W2_SCALES_BYTES

    print(f"\nPer-expert scale dedup:")
    print(f"  w13 scales: avg {avg_w13s_unique:.0f} unique per layer / {NUM_EXPERTS} ({100*(1-avg_w13s_unique/NUM_EXPERTS):.1f}% dedup)")
    print(f"  w2 scales:  avg {avg_w2s_unique:.0f} unique per layer / {NUM_EXPERTS} ({100*(1-avg_w2s_unique/NUM_EXPERTS):.1f}% dedup)")
    print(f"  Scale savings: {(w13s_savings + w2s_savings)/1e9:.2f} GB")

    print(f"\nRow-level scale dedup (across all experts in a layer):")
    print(f"  w13 scale rows: {w13_scale_rows_unique:,} unique / {w13_scale_rows_total:,} total ({100*(1-w13_scale_rows_unique/w13_scale_rows_total):.1f}% dedup)")
    print(f"  w2 scale rows:  {w2_scale_rows_unique:,} unique / {w2_scale_rows_total:,} total ({100*(1-w2_scale_rows_unique/w2_scale_rows_total):.1f}% dedup)")

    # Compute savings from row-level dedup
    w13_row_bytes = W13_N * 2  # 2048
    w2_row_bytes = W2_N * 2    # 4096
    w13_row_savings = (w13_scale_rows_total - w13_scale_rows_unique) * w13_row_bytes
    w2_row_savings = (w2_scale_rows_total - w2_scale_rows_unique) * w2_row_bytes
    print(f"  Row-level scale savings: {(w13_row_savings + w2_row_savings)/1e9:.2f} GB")

    print(f"\nTotal scales: {total_scale_bytes/1e9:.2f} GB ({100*total_scale_bytes/(NUM_LAYERS*NUM_EXPERTS*EXPERT_BYTES):.1f}% of total)")
    print(f"Time: {time.time()-start:.1f}s\n")

    # ── Analysis 3: Cosine similarity between experts (sampled) ──
    print("=" * 70)
    print("ANALYSIS 3: Cosine similarity between expert pairs (sampled)")
    print("=" * 70)
    start = time.time()

    # Sample a few layers and compare random pairs
    sample_layers = [0, 11, 23, 35, 47]
    num_pairs = 200

    for layer in sample_layers:
        sims = []
        rng = np.random.RandomState(42 + layer)

        for _ in range(num_pairs):
            e1, e2 = rng.choice(NUM_EXPERTS, 2, replace=False)
            base1 = expert_offset(layer, e1)
            base2 = expert_offset(layer, e2)

            # Compare packed weights as flat byte arrays
            # Use w13_packed (largest component, 1MB)
            d1 = np.frombuffer(mm[base1:base1+W13_PACKED_BYTES], dtype=np.uint8).astype(np.float32)
            d2 = np.frombuffer(mm[base2:base2+W13_PACKED_BYTES], dtype=np.uint8).astype(np.float32)

            dot = np.dot(d1, d2)
            n1 = np.linalg.norm(d1)
            n2 = np.linalg.norm(d2)
            if n1 > 0 and n2 > 0:
                sim = dot / (n1 * n2)
                sims.append(sim)

        sims = np.array(sims)
        print(f"  Layer {layer:2d}: mean={sims.mean():.4f}, std={sims.std():.4f}, "
              f"min={sims.min():.4f}, max={sims.max():.4f}, "
              f">0.99={np.sum(sims>0.99)}, >0.95={np.sum(sims>0.95)}, >0.90={np.sum(sims>0.90)}")

    print(f"Time: {time.time()-start:.1f}s\n")

    # ── Analysis 4: Weight value distribution ──
    print("=" * 70)
    print("ANALYSIS 4: Weight packed value entropy (compression potential)")
    print("=" * 70)

    # Sample a few experts and measure byte-level entropy
    sample_experts = [(0, 0), (0, 255), (0, 511), (23, 0), (23, 255), (47, 0), (47, 511)]

    for layer, expert in sample_experts:
        base = expert_offset(layer, expert)
        data = np.frombuffer(mm[base:base+EXPERT_BYTES], dtype=np.uint8)

        # Byte-level entropy
        counts = np.bincount(data, minlength=256)
        probs = counts / len(data)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))

        # Theoretical compression ratio
        comp_ratio = entropy / 8.0

        print(f"  Expert [{layer:2d},{expert:3d}]: entropy={entropy:.3f} bits/byte, "
              f"theoretical compression={comp_ratio:.1%}, "
              f"compressed={EXPERT_BYTES*comp_ratio/1024:.1f} KB / {EXPERT_BYTES/1024:.1f} KB")

    print()

    # ── Analysis 5: Scale value statistics ──
    print("=" * 70)
    print("ANALYSIS 5: Scale value range and precision analysis")
    print("=" * 70)

    # Check if scales could fit in FP8
    all_w13_scales = []
    all_w2_scales = []

    for layer in [0, 23, 47]:
        for e in range(0, NUM_EXPERTS, 64):  # Sample every 64th expert
            _, w13s, _, w2s = get_expert_components(mm, layer, e)
            w13_arr = np.frombuffer(w13s, dtype=np.uint16).copy()
            w2_arr = np.frombuffer(w2s, dtype=np.uint16).copy()

            # Convert BF16 to float32 for analysis
            # BF16: sign(1) + exp(8) + mantissa(7)
            # Just shift left by 16 to get FP32
            w13_f32 = np.frombuffer((w13_arr.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
            w2_f32 = np.frombuffer((w2_arr.astype(np.uint32) << 16).tobytes(), dtype=np.float32)

            all_w13_scales.append(w13_f32)
            all_w2_scales.append(w2_f32)

    w13_all = np.concatenate(all_w13_scales)
    w2_all = np.concatenate(all_w2_scales)

    print(f"  w13 scales (sampled {len(w13_all):,} values):")
    print(f"    range: [{w13_all.min():.6f}, {w13_all.max():.6f}]")
    print(f"    mean: {np.mean(np.abs(w13_all)):.6f}, std: {np.std(w13_all):.6f}")
    print(f"    zeros: {np.sum(w13_all == 0):,} ({100*np.sum(w13_all==0)/len(w13_all):.2f}%)")

    # Check unique BF16 values (how many distinct scale values exist)
    w13_unique_u16 = len(set(np.concatenate([np.frombuffer(s, dtype=np.uint16) for s in
                    [get_expert_components(mm, 0, e)[1] for e in range(0, NUM_EXPERTS, 64)]])))
    print(f"    unique BF16 values (layer 0 sampled): {w13_unique_u16:,}")

    print(f"\n  w2 scales (sampled {len(w2_all):,} values):")
    print(f"    range: [{w2_all.min():.6f}, {w2_all.max():.6f}]")
    print(f"    mean: {np.mean(np.abs(w2_all)):.6f}, std: {np.std(w2_all):.6f}")
    print(f"    zeros: {np.sum(w2_all == 0):,} ({100*np.sum(w2_all==0)/len(w2_all):.2f}%)")

    # FP8 E4M3 range: [-448, 448], ~3 decimal digits precision
    # FP8 E5M2 range: [-57344, 57344], ~2 decimal digits
    w13_abs = np.abs(w13_all[w13_all != 0])
    w2_abs = np.abs(w2_all[w2_all != 0])
    print(f"\n  FP8 E4M3 feasibility (range [-448, 448]):")
    print(f"    w13 out of range: {np.sum(w13_abs > 448):,} / {len(w13_abs):,}")
    print(f"    w2 out of range:  {np.sum(w2_abs > 448):,} / {len(w2_abs):,}")

    # Check if scale values cluster (could use codebook)
    print(f"\n  Scale clustering potential:")
    w13_u16_all = np.concatenate([np.frombuffer(get_expert_components(mm, 0, e)[1], dtype=np.uint16)
                                   for e in range(NUM_EXPERTS)])
    unique_vals = np.unique(w13_u16_all)
    print(f"    Layer 0 w13: {len(unique_vals):,} unique BF16 values across all {NUM_EXPERTS} experts")
    print(f"    Max possible BF16 values: 65536")
    print(f"    If 256 codebook entries: {100*256/len(unique_vals):.1f}% coverage (8-bit index)")
    print(f"    If 4096 codebook entries: {100*min(4096,len(unique_vals))/len(unique_vals):.1f}% coverage (12-bit index)")

    mm.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
