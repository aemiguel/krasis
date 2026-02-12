//! AVX2 INT4 expert matmul kernel for Zen 2 (EPYC 7742).
//!
//! Three implementations for correctness verification:
//!   1. `matmul_bf16_scalar` — BF16 reference (ground truth)
//!   2. `matmul_int4_scalar` — scalar INT4 (correctness baseline)
//!   3. `expert_matmul_int4` — AVX2 vectorized INT4 (production kernel)
//!
//! For M=1 decode: computes y[N] = x[K] @ W[N,K]^T
//! Weight matrix W stored as packed INT4 with per-group BF16 scales.
//!
//! AVX2 strategy: process 8 INT4 values per iteration.
//!   - Broadcast packed u32 word → 8 lanes
//!   - Variable shift + mask to extract 8 nibbles
//!   - Subtract offset → signed [-8, 7]
//!   - Convert to f32, FMA against BF16 activations
//!   - Per-group scale applied once, not per-element
//!   - Horizontal sum at end of each output row

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::weights::marlin::{bf16_to_f32, QuantizedInt4, QuantizedInt8};

/// Scalar BF16 matrix-vector multiply (reference).
///
/// Computes output[n] = sum_k(weight[n, k] * activation[k]).
pub fn matmul_bf16_scalar(
    weight_bf16: &[u16], // [N, K] row-major BF16
    activation: &[u16],  // [K] BF16
    output: &mut [f32],  // [N]
    n: usize,
    k: usize,
) {
    assert_eq!(weight_bf16.len(), n * k);
    assert_eq!(activation.len(), k);
    assert_eq!(output.len(), n);

    for row in 0..n {
        let mut acc: f32 = 0.0;
        let row_offset = row * k;
        for col in 0..k {
            let w = bf16_to_f32(weight_bf16[row_offset + col]);
            let x = bf16_to_f32(activation[col]);
            acc += w * x;
        }
        output[row] = acc;
    }
}

/// Scalar INT4 matrix-vector multiply.
///
/// Matches AVX2 computation semantics: scale applied per-group, not per-element.
pub fn matmul_int4_scalar(
    q: &QuantizedInt4,
    activation: &[u16], // [K] BF16
    output: &mut [f32],  // [N]
) {
    assert_eq!(activation.len(), q.cols);
    assert_eq!(output.len(), q.rows);

    let num_groups = q.cols / q.group_size;
    let packed_k = q.cols / 8;

    for row in 0..q.rows {
        let mut acc: f32 = 0.0;

        for g in 0..num_groups {
            let scale = bf16_to_f32(q.scales[row * num_groups + g]);
            let mut group_acc: f32 = 0.0;

            for pack in 0..(q.group_size / 8) {
                let k_base = g * q.group_size + pack * 8;
                let word = q.packed[row * packed_k + k_base / 8];

                for j in 0..8u32 {
                    let u4 = ((word >> (j * 4)) & 0xF) as i32;
                    let q_val = (u4 - 8) as f32;
                    let x_val = bf16_to_f32(activation[k_base + j as usize]);
                    group_acc += q_val * x_val;
                }
            }

            acc += group_acc * scale;
        }

        output[row] = acc;
    }
}

/// AVX2 + FMA INT4 matrix-vector multiply.
///
/// Per iteration (8 INT4 values from one packed u32):
///   1. Broadcast word → 8 lanes via `_mm256_set1_epi32`
///   2. Variable right-shift by [0,4,8,...,28] → isolate each nibble
///   3. AND 0xF, subtract 8 → signed [-8, 7] as i32
///   4. Convert to f32
///   5. Load 8 BF16 activations → f32 via zero-extend + shift
///   6. FMA: group_acc += weight * activation
///   7. After group: FMA acc += group_acc * scale
///   8. After row: horizontal sum → output
///
/// # Safety
/// Requires AVX2 + FMA. All pointers must be valid for their respective lengths.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn expert_matmul_int4(
    packed: *const u32,     // [N, K/8] packed INT4
    scales: *const u16,     // [N, K/group_size] BF16 scales
    activation: *const u16, // [K] BF16
    output: *mut f32,       // [N]
    k: usize,
    n: usize,
    group_size: usize,
) {
    let num_groups = k / group_size;
    let packed_k = k / 8;
    let packs_per_group = group_size / 8;

    // Constants
    let shift_amounts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    let mask_0f = _mm256_set1_epi32(0xF);
    let offset_8 = _mm256_set1_epi32(8);

    for row in 0..n {
        let mut acc = _mm256_setzero_ps();

        for g in 0..num_groups {
            let scale_f32 = bf16_to_f32(*scales.add(row * num_groups + g));
            let scale_vec = _mm256_set1_ps(scale_f32);
            let mut group_acc = _mm256_setzero_ps();

            for pack in 0..packs_per_group {
                let k_base = g * group_size + pack * 8;

                // Load packed u32, broadcast to all 8 lanes
                let word = *packed.add(row * packed_k + k_base / 8);
                let word_vec = _mm256_set1_epi32(word as i32);

                // Extract 8 nibbles via variable shift + mask
                let shifted = _mm256_srlv_epi32(word_vec, shift_amounts);
                let masked = _mm256_and_si256(shifted, mask_0f);

                // Unsigned [0,15] → signed [-8,7]
                let signed_i32 = _mm256_sub_epi32(masked, offset_8);
                let weights_f32 = _mm256_cvtepi32_ps(signed_i32);

                // Load 8 BF16 activations → f32
                // BF16 = upper 16 bits of f32, so: zero-extend u16→u32, shift left 16
                let act_bf16 = _mm_loadu_si128(activation.add(k_base) as *const __m128i);
                let act_u32 = _mm256_cvtepu16_epi32(act_bf16);
                let act_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(act_u32, 16));

                // FMA: group_acc += weight * activation
                group_acc = _mm256_fmadd_ps(weights_f32, act_f32, group_acc);
            }

            // acc += group_acc * scale
            acc = _mm256_fmadd_ps(group_acc, scale_vec, acc);
        }

        // Horizontal sum of 8 lanes → single f32
        *output.add(row) = hsum_avx2(acc);
    }
}

/// Horizontal sum of 8 f32 lanes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let hi64 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, hi64);
    _mm_cvtss_f32(sum32)
}

/// Safe wrapper for the AVX2 INT4 matmul kernel.
pub fn matmul_int4_avx2(q: &QuantizedInt4, activation: &[u16], output: &mut [f32]) {
    assert_eq!(activation.len(), q.cols);
    assert_eq!(output.len(), q.rows);
    assert!(q.cols % q.group_size == 0);
    assert!(q.cols % 8 == 0);

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("AVX2 + FMA required");
    }

    unsafe {
        expert_matmul_int4(
            q.packed.as_ptr(),
            q.scales.as_ptr(),
            activation.as_ptr(),
            output.as_mut_ptr(),
            q.cols,
            q.rows,
            q.group_size,
        );
    }
}

// ── Integer kernel (INT16 × INT4 → INT32 accumulation) ─────────────
//
// Instead of converting INT4 weights and BF16 activations to f32 and
// using FMA, this kernel:
//   1. Pre-quantizes BF16 activations to INT16 (once per forward call)
//   2. Unpacks INT4 weights to INT16 (in registers, no memory writes)
//   3. Uses _mm256_madd_epi16 to multiply 16 INT16 pairs → 8 INT32
//   4. Accumulates in INT32, converts to f32 only at group boundaries
//
// Advantages over FMA kernel:
//   - 16 values per instruction vs 8 → 2x instruction throughput
//   - 3-cycle latency vs 5 on Zen 2
//   - No f32 conversion in inner loop (only at group boundaries)

/// Quantize BF16 activation vector to INT16 with per-group scales.
///
/// For each group: scale = max(|x|) / 32767, int16[i] = round(x[i] / scale).
/// INT16 per-group quantization has 15 bits of precision — more than BF16's
/// 7-bit mantissa — so this does NOT degrade quality.
pub fn quantize_activation_int16(
    activation_bf16: &[u16],
    group_size: usize,
    output_int16: &mut [i16],
    output_scales: &mut [f32],
) {
    let k = activation_bf16.len();
    let num_groups = k / group_size;
    assert_eq!(output_int16.len(), k);
    assert_eq!(output_scales.len(), num_groups);
    assert!(k % group_size == 0);

    for g in 0..num_groups {
        let start = g * group_size;

        // Find max abs value in group
        let mut max_abs: f32 = 0.0;
        for i in 0..group_size {
            let val = bf16_to_f32(activation_bf16[start + i]);
            max_abs = max_abs.max(val.abs());
        }

        // Compute scale (handle zero group)
        let scale = if max_abs > 0.0 { max_abs / 32767.0 } else { 1.0 };
        let inv_scale = if max_abs > 0.0 { 32767.0 / max_abs } else { 0.0 };
        output_scales[g] = scale;

        // Quantize to INT16
        for i in 0..group_size {
            let val = bf16_to_f32(activation_bf16[start + i]);
            let quantized = (val * inv_scale).round() as i32;
            output_int16[start + i] = quantized.clamp(-32768, 32767) as i16;
        }
    }
}

/// Scalar integer-path INT4 matmul (correctness reference).
///
/// Same computation as AVX2 integer kernel but without SIMD.
pub fn matmul_int4_integer_scalar(
    q: &QuantizedInt4,
    act_int16: &[i16],
    act_scales: &[f32],
    output: &mut [f32],
) {
    assert_eq!(act_int16.len(), q.cols);
    assert_eq!(act_scales.len(), q.cols / q.group_size);
    assert_eq!(output.len(), q.rows);

    let num_groups = q.cols / q.group_size;
    let packed_k = q.cols / 8;

    for row in 0..q.rows {
        let mut acc: f32 = 0.0;

        for g in 0..num_groups {
            let w_scale = bf16_to_f32(q.scales[row * num_groups + g]);
            let a_scale = act_scales[g];
            let combined = w_scale * a_scale;
            let mut group_sum: i32 = 0;

            for pack in 0..(q.group_size / 8) {
                let k_base = g * q.group_size + pack * 8;
                let word = q.packed[row * packed_k + k_base / 8];

                for j in 0..8u32 {
                    let u4 = ((word >> (j * 4)) & 0xF) as i32;
                    let w_val = u4 - 8;
                    let a_val = act_int16[k_base + j as usize] as i32;
                    group_sum += w_val * a_val;
                }
            }

            acc += group_sum as f32 * combined;
        }

        output[row] = acc;
    }
}

/// AVX2 integer INT4 matmul using `_mm256_madd_epi16`.
///
/// Per iteration (16 INT4 values from 8 packed bytes):
///   1. Load 8 bytes (2 u32 words = 16 nibbles)
///   2. Separate low/high nibbles via mask, interleave to sequential order
///   3. Subtract offset (unsigned [0,15] → signed [-8,7])
///   4. Sign-extend INT8 → INT16 via `_mm256_cvtepi8_epi16`
///   5. `_mm256_madd_epi16` against pre-quantized INT16 activations → 8 INT32
///   6. Accumulate INT32 partial sums per group
///   7. At group boundary: convert to f32, apply weight_scale × act_scale
///   8. After all groups: horizontal sum → output scalar
///
/// # Safety
/// Requires AVX2 + FMA. All pointers must be valid for their respective lengths.
/// `group_size` must be divisible by 16.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn expert_matmul_int4_integer(
    packed: *const u32,        // [N, K/8] packed INT4
    weight_scales: *const u16, // [N, K/group_size] BF16 weight scales
    act_int16: *const i16,     // [K] quantized INT16 activations
    act_scales: *const f32,    // [K/group_size] activation scales
    output: *mut f32,          // [N]
    k: usize,
    n: usize,
    group_size: usize,
) {
    let num_groups = k / group_size;
    let packs_per_group = group_size / 16; // 16 values per iteration

    // Constants for nibble unpacking
    let mask_0f = _mm_set1_epi8(0x0F);
    let offset_8 = _mm_set1_epi8(8);

    for row in 0..n {
        let mut float_acc = _mm256_setzero_ps();
        let packed_base = packed.add(row * (k / 8));

        for g in 0..num_groups {
            let mut int_acc = _mm256_setzero_si256();
            let w_scale_f32 = bf16_to_f32(*weight_scales.add(row * num_groups + g));
            let a_scale_f32 = *act_scales.add(g);
            let combined_scale = w_scale_f32 * a_scale_f32;

            for p in 0..packs_per_group {
                let k_base = g * group_size + p * 16;

                // Load 8 bytes of packed INT4 (2 u32 words = 16 nibbles)
                let load_ptr = packed_base.add(k_base / 8) as *const __m128i;
                let raw = _mm_loadl_epi64(load_ptr);

                // Separate low nibbles (even values) and high nibbles (odd values)
                let lo = _mm_and_si128(raw, mask_0f);
                let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);

                // Interleave to sequential order: [v0,v1,v2,...,v15]
                let interleaved = _mm_unpacklo_epi8(lo, hi);

                // Unsigned [0,15] → signed [-8,7]
                let signed = _mm_sub_epi8(interleaved, offset_8);

                // Sign-extend INT8 → INT16 (16 values in one YMM register)
                let w16 = _mm256_cvtepi8_epi16(signed);

                // Load 16 INT16 activations
                let a16 = _mm256_loadu_si256(act_int16.add(k_base) as *const __m256i);

                // Multiply-accumulate: 16 INT16 × INT16 → 8 INT32 partial sums
                let dot = _mm256_madd_epi16(w16, a16);
                int_acc = _mm256_add_epi32(int_acc, dot);
            }

            // Convert INT32 partial sums to f32 and apply combined scale
            let group_f32 = _mm256_cvtepi32_ps(int_acc);
            float_acc = _mm256_fmadd_ps(group_f32, _mm256_set1_ps(combined_scale), float_acc);
        }

        // Horizontal sum → single f32 output
        *output.add(row) = hsum_avx2(float_acc);
    }
}

/// Safe wrapper for the AVX2 integer INT4 matmul kernel.
pub fn matmul_int4_integer(
    q: &QuantizedInt4,
    act_int16: &[i16],
    act_scales: &[f32],
    output: &mut [f32],
) {
    assert_eq!(act_int16.len(), q.cols);
    assert_eq!(act_scales.len(), q.cols / q.group_size);
    assert_eq!(output.len(), q.rows);
    assert!(q.group_size % 16 == 0, "Integer kernel requires group_size divisible by 16");

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("AVX2 + FMA required");
    }

    unsafe {
        expert_matmul_int4_integer(
            q.packed.as_ptr(),
            q.scales.as_ptr(),
            act_int16.as_ptr(),
            act_scales.as_ptr(),
            output.as_mut_ptr(),
            q.cols,
            q.rows,
            q.group_size,
        );
    }
}

/// Parallel AVX2 integer INT4 matmul — splits output rows across rayon threads.
///
/// The integer kernel is ~2x faster per-element than FMA, so parallelism only
/// helps for larger matrices. Threshold: rows×cols > 8M elements (~500 μs serial).
/// Below that, rayon overhead (~150 μs) exceeds the benefit.
pub fn matmul_int4_integer_parallel(
    q: &QuantizedInt4,
    act_int16: &[i16],
    act_scales: &[f32],
    output: &mut [f32],
) {
    use rayon::prelude::*;

    assert_eq!(act_int16.len(), q.cols);
    assert_eq!(act_scales.len(), q.cols / q.group_size);
    assert_eq!(output.len(), q.rows);
    assert!(q.group_size % 16 == 0, "Integer kernel requires group_size divisible by 16");

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("AVX2 + FMA required");
    }

    // For small matrices, single-thread is faster (rayon overhead > kernel benefit).
    // Threshold: ~8M elements ≈ 500 μs serial at ~0.067 ns/element.
    if q.rows * q.cols <= 8_000_000 {
        unsafe {
            expert_matmul_int4_integer(
                q.packed.as_ptr(),
                q.scales.as_ptr(),
                act_int16.as_ptr(),
                act_scales.as_ptr(),
                output.as_mut_ptr(),
                q.cols,
                q.rows,
                q.group_size,
            );
        }
        return;
    }

    let packed_k = q.cols / 8;
    let num_groups = q.cols / q.group_size;
    let chunk_size = 32;

    let packed_addr = q.packed.as_ptr() as usize;
    let scales_addr = q.scales.as_ptr() as usize;
    let act_addr = act_int16.as_ptr() as usize;
    let act_scales_addr = act_scales.as_ptr() as usize;
    let cols = q.cols;
    let group_size = q.group_size;

    output.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk)| {
        let start_row = chunk_idx * chunk_size;
        let chunk_rows = chunk.len();

        unsafe {
            expert_matmul_int4_integer(
                (packed_addr as *const u32).add(start_row * packed_k),
                (scales_addr as *const u16).add(start_row * num_groups),
                act_addr as *const i16,
                act_scales_addr as *const f32,
                chunk.as_mut_ptr(),
                cols,
                chunk_rows,
                group_size,
            );
        }
    });
}

/// Parallel AVX2 INT4 matmul — splits output rows across rayon threads.
///
/// Each thread processes a chunk of rows independently. The activation
/// vector is shared (read-only, cached in L1/L2 across all threads).
/// Weight rows are accessed sequentially within each chunk for optimal
/// prefetcher behavior.
pub fn matmul_int4_parallel(q: &QuantizedInt4, activation: &[u16], output: &mut [f32]) {
    use rayon::prelude::*;

    assert_eq!(activation.len(), q.cols);
    assert_eq!(output.len(), q.rows);
    assert!(q.cols % q.group_size == 0);
    assert!(q.cols % 8 == 0);

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("AVX2 + FMA required");
    }

    // For small matrices, avoid rayon overhead
    if q.rows <= 64 {
        unsafe {
            expert_matmul_int4(
                q.packed.as_ptr(),
                q.scales.as_ptr(),
                activation.as_ptr(),
                output.as_mut_ptr(),
                q.cols,
                q.rows,
                q.group_size,
            );
        }
        return;
    }

    let packed_k = q.cols / 8;
    let num_groups = q.cols / q.group_size;
    let chunk_size = 32; // rows per chunk — balances parallelism vs overhead

    // Safety: each chunk writes to disjoint output rows and reads from
    // disjoint weight rows + shared activation. No data races.
    // Convert to usize for Send+Sync in rayon closure.
    // Safety: all threads read disjoint weight rows + shared activation.
    let packed_addr = q.packed.as_ptr() as usize;
    let scales_addr = q.scales.as_ptr() as usize;
    let act_addr = activation.as_ptr() as usize;
    let cols = q.cols;
    let group_size = q.group_size;

    output.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk)| {
        let start_row = chunk_idx * chunk_size;
        let chunk_rows = chunk.len();

        unsafe {
            expert_matmul_int4(
                (packed_addr as *const u32).add(start_row * packed_k),
                (scales_addr as *const u16).add(start_row * num_groups),
                act_addr as *const u16,
                chunk.as_mut_ptr(),
                cols,
                chunk_rows,
                group_size,
            );
        }
    });
}

// ── INT8 integer kernel (INT16 × INT8 → INT32 accumulation) ──────────
//
// Simpler than INT4: no nibble extraction needed. Load 16 raw i8 values,
// sign-extend to INT16, then _mm256_madd_epi16 against pre-quantized
// INT16 activations. Same throughput as INT4 integer kernel but 2x memory
// bandwidth (16 bytes per 16 values vs 8 bytes).

/// Scalar integer-path INT8 matmul (correctness reference).
pub fn matmul_int8_integer_scalar(
    q: &QuantizedInt8,
    act_int16: &[i16],
    act_scales: &[f32],
    output: &mut [f32],
) {
    assert_eq!(act_int16.len(), q.cols);
    assert_eq!(act_scales.len(), q.cols / q.group_size);
    assert_eq!(output.len(), q.rows);

    let num_groups = q.cols / q.group_size;

    for row in 0..q.rows {
        let mut acc: f32 = 0.0;

        for g in 0..num_groups {
            let w_scale = bf16_to_f32(q.scales[row * num_groups + g]);
            let a_scale = act_scales[g];
            let combined = w_scale * a_scale;
            let mut group_sum: i32 = 0;

            let group_start = g * q.group_size;
            for i in 0..q.group_size {
                let k_idx = row * q.cols + group_start + i;
                let w_val = q.data[k_idx] as i32;
                let a_val = act_int16[group_start + i] as i32;
                group_sum += w_val * a_val;
            }

            acc += group_sum as f32 * combined;
        }

        output[row] = acc;
    }
}

/// AVX2 integer INT8 matmul using `_mm256_madd_epi16`.
///
/// Per iteration (16 INT8 values):
///   1. Load 16 bytes (16 INT8 weights) into __m128i
///   2. Sign-extend INT8 → INT16 via `_mm256_cvtepi8_epi16` (16 values in __m256i)
///   3. Load 16 INT16 activations
///   4. `_mm256_madd_epi16` against activations → 8 INT32 partial sums
///   5. Accumulate INT32 per group
///   6. At group boundary: convert to f32, apply weight_scale × act_scale
///   7. After all groups: horizontal sum → output scalar
///
/// # Safety
/// Requires AVX2 + FMA. All pointers must be valid for their respective lengths.
/// `group_size` must be divisible by 16.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn expert_matmul_int8_integer(
    data: *const i8,           // [N, K] raw INT8 weights
    weight_scales: *const u16, // [N, K/group_size] BF16 weight scales
    act_int16: *const i16,     // [K] quantized INT16 activations
    act_scales: *const f32,    // [K/group_size] activation scales
    output: *mut f32,          // [N]
    k: usize,
    n: usize,
    group_size: usize,
) {
    let num_groups = k / group_size;
    let vals_per_iter = 16; // 16 INT8 values per iteration
    let iters_per_group = group_size / vals_per_iter;

    for row in 0..n {
        let mut float_acc = _mm256_setzero_ps();
        let data_base = data.add(row * k);

        for g in 0..num_groups {
            let mut int_acc = _mm256_setzero_si256();
            let w_scale_f32 = bf16_to_f32(*weight_scales.add(row * num_groups + g));
            let a_scale_f32 = *act_scales.add(g);
            let combined_scale = w_scale_f32 * a_scale_f32;

            for p in 0..iters_per_group {
                let k_base = g * group_size + p * vals_per_iter;

                // Load 16 INT8 weights
                let raw = _mm_loadu_si128(data_base.add(k_base) as *const __m128i);

                // Sign-extend INT8 → INT16 (16 values in one YMM register)
                let w16 = _mm256_cvtepi8_epi16(raw);

                // Load 16 INT16 activations
                let a16 = _mm256_loadu_si256(act_int16.add(k_base) as *const __m256i);

                // Multiply-accumulate: 16 INT16 × INT16 → 8 INT32 partial sums
                let dot = _mm256_madd_epi16(w16, a16);
                int_acc = _mm256_add_epi32(int_acc, dot);
            }

            // Convert INT32 partial sums to f32 and apply combined scale
            let group_f32 = _mm256_cvtepi32_ps(int_acc);
            float_acc = _mm256_fmadd_ps(group_f32, _mm256_set1_ps(combined_scale), float_acc);
        }

        // Horizontal sum → single f32 output
        *output.add(row) = hsum_avx2(float_acc);
    }
}

/// Safe wrapper for the AVX2 integer INT8 matmul kernel.
pub fn matmul_int8_integer(
    q: &QuantizedInt8,
    act_int16: &[i16],
    act_scales: &[f32],
    output: &mut [f32],
) {
    assert_eq!(act_int16.len(), q.cols);
    assert_eq!(act_scales.len(), q.cols / q.group_size);
    assert_eq!(output.len(), q.rows);
    assert!(q.group_size % 16 == 0, "Integer kernel requires group_size divisible by 16");

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("AVX2 + FMA required");
    }

    unsafe {
        expert_matmul_int8_integer(
            q.data.as_ptr(),
            q.scales.as_ptr(),
            act_int16.as_ptr(),
            act_scales.as_ptr(),
            output.as_mut_ptr(),
            q.cols,
            q.rows,
            q.group_size,
        );
    }
}

/// Parallel AVX2 integer INT8 matmul — splits output rows across rayon threads.
pub fn matmul_int8_integer_parallel(
    q: &QuantizedInt8,
    act_int16: &[i16],
    act_scales: &[f32],
    output: &mut [f32],
) {
    use rayon::prelude::*;

    assert_eq!(act_int16.len(), q.cols);
    assert_eq!(act_scales.len(), q.cols / q.group_size);
    assert_eq!(output.len(), q.rows);
    assert!(q.group_size % 16 == 0, "Integer kernel requires group_size divisible by 16");

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("AVX2 + FMA required");
    }

    // For small matrices, single-thread is faster
    if q.rows * q.cols <= 8_000_000 {
        unsafe {
            expert_matmul_int8_integer(
                q.data.as_ptr(),
                q.scales.as_ptr(),
                act_int16.as_ptr(),
                act_scales.as_ptr(),
                output.as_mut_ptr(),
                q.cols,
                q.rows,
                q.group_size,
            );
        }
        return;
    }

    let num_groups = q.cols / q.group_size;
    let chunk_size = 32;

    let data_addr = q.data.as_ptr() as usize;
    let scales_addr = q.scales.as_ptr() as usize;
    let act_addr = act_int16.as_ptr() as usize;
    let act_scales_addr = act_scales.as_ptr() as usize;
    let cols = q.cols;
    let group_size = q.group_size;

    output.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk)| {
        let start_row = chunk_idx * chunk_size;
        let chunk_rows = chunk.len();

        unsafe {
            expert_matmul_int8_integer(
                (data_addr as *const i8).add(start_row * cols),
                (scales_addr as *const u16).add(start_row * num_groups),
                act_addr as *const i16,
                act_scales_addr as *const f32,
                chunk.as_mut_ptr(),
                cols,
                chunk_rows,
                group_size,
            );
        }
    });
}

// ── Transposed layout INT4 kernels (for unified weight format) ────────
//
// Weight layout: [K/8, N] — K is reduction dimension (outer), N is output
// dimension (inner, contiguous). Enables SIMD across N outputs without
// horizontal sum. Used for unified w13 (gate+up) and w2 (down) weights.
//
// Strategy: outer loop over K groups, inner SIMD over 8 N outputs at a time.
//   - Load 8 packed u32s (one per N output) containing 8 K-dim INT4 values each
//   - For each K position j: extract nibble j from all 8 words, multiply by
//     broadcast activation[k], accumulate
//   - At group end: scale and add to accumulator
//   - Store 8 output values directly (no horizontal sum!)

/// Scalar INT4 transposed matmul (correctness reference).
///
/// Weight layout: packed[K/8, N], scales[K/group_size, N].
/// Computes output[n] = sum_k(activation[k] * weight[k, n]) for each n.
pub fn matmul_int4_transposed_scalar(
    packed: &[u32],      // [K/8, N]
    scales: &[u16],      // [K/group_size, N]
    activation: &[u16],  // [K] BF16
    output: &mut [f32],  // [N]
    k: usize,
    n: usize,
    group_size: usize,
) {
    let num_groups = k / group_size;
    let packs_per_group = group_size / 8;

    output[..n].fill(0.0);

    for n_pos in 0..n {
        let mut acc: f32 = 0.0;

        for g in 0..num_groups {
            let scale = bf16_to_f32(scales[g * n + n_pos]);
            let mut group_acc: f32 = 0.0;

            for pack in 0..packs_per_group {
                let k_row = g * packs_per_group + pack;
                let k_base = k_row * 8;
                let word = packed[k_row * n + n_pos];

                for j in 0..8u32 {
                    let u4 = ((word >> (j * 4)) & 0xF) as i32;
                    let q_val = (u4 - 8) as f32;
                    let x_val = bf16_to_f32(activation[k_base + j as usize]);
                    group_acc += q_val * x_val;
                }
            }

            acc += group_acc * scale;
        }

        output[n_pos] = acc;
    }
}

/// AVX2 FMA INT4 transposed matmul.
///
/// Processes 8 N outputs at a time. No horizontal sum needed.
/// `n_stride` is the total N (row stride); kernel processes N positions
/// `[n_start .. n_start + n_out)` and writes `n_out` values to output.
///
/// # Safety
/// Requires AVX2 + FMA. All pointers must be valid for their respective lengths.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn expert_matmul_int4_transposed(
    packed: *const u32,     // [K/8, n_stride]
    scales: *const u16,     // [K/group_size, n_stride]
    activation: *const u16, // [K] BF16
    output: *mut f32,       // [n_out]
    k: usize,
    n_stride: usize,
    n_start: usize,
    n_out: usize,
    group_size: usize,
) {
    let num_groups = k / group_size;
    let packs_per_group = group_size / 8;

    let mask_0f = _mm256_set1_epi32(0xF);
    let offset_8 = _mm256_set1_epi32(8);

    let n_blocks = n_out / 8;
    let n_rem = n_out % 8;

    for nb in 0..n_blocks {
        let n_base = n_start + nb * 8;
        let mut acc = _mm256_setzero_ps();

        for g in 0..num_groups {
            // Load 8 BF16 scales for this group → f32
            let scales_bf16 = _mm_loadu_si128(
                scales.add(g * n_stride + n_base) as *const __m128i,
            );
            let scales_u32 = _mm256_cvtepu16_epi32(scales_bf16);
            let scale_vec = _mm256_castsi256_ps(_mm256_slli_epi32(scales_u32, 16));

            let mut group_acc = _mm256_setzero_ps();

            for pack in 0..packs_per_group {
                let k_row = g * packs_per_group + pack;
                let k_base = k_row * 8;

                // Load 8 packed u32s (8 N positions, each containing 8 K-dim INT4 values)
                let words = _mm256_loadu_si256(
                    packed.add(k_row * n_stride + n_base) as *const __m256i,
                );

                // Pre-load 8 BF16 activations for k_base..k_base+8 → f32
                let act_bf16 = _mm_loadu_si128(
                    activation.add(k_base) as *const __m128i,
                );
                let act_u32 = _mm256_cvtepu16_epi32(act_bf16);
                let act_f32_all = _mm256_castsi256_ps(_mm256_slli_epi32(act_u32, 16));

                // Process 8 nibbles (8 K positions)
                for j in 0..8i32 {
                    // Extract nibble j from all 8 u32s → 8 INT4 weights (one per N)
                    let shift = _mm256_set1_epi32(j * 4);
                    let shifted = _mm256_srlv_epi32(words, shift);
                    let masked = _mm256_and_si256(shifted, mask_0f);
                    let signed_i32 = _mm256_sub_epi32(masked, offset_8);
                    let w_f32 = _mm256_cvtepi32_ps(signed_i32);

                    // Broadcast activation[k_base+j] to all 8 lanes
                    let perm = _mm256_set1_epi32(j);
                    let act_broadcast = _mm256_permutevar8x32_ps(act_f32_all, perm);

                    group_acc = _mm256_fmadd_ps(w_f32, act_broadcast, group_acc);
                }
            }

            // acc += group_acc * scale_vec
            acc = _mm256_fmadd_ps(group_acc, scale_vec, acc);
        }

        // Store 8 output values — no horizontal sum!
        _mm256_storeu_ps(output.add(nb * 8), acc);
    }

    // Handle remainder N positions with scalar code
    if n_rem > 0 {
        let rem_start = n_blocks * 8;
        for r in 0..n_rem {
            let n_pos = n_start + rem_start + r;
            let mut acc: f32 = 0.0;
            for g in 0..num_groups {
                let scale = bf16_to_f32(*scales.add(g * n_stride + n_pos));
                let mut group_acc: f32 = 0.0;
                for pack in 0..packs_per_group {
                    let k_row = g * packs_per_group + pack;
                    let k_base = k_row * 8;
                    let word = *packed.add(k_row * n_stride + n_pos);
                    for j in 0..8u32 {
                        let u4 = ((word >> (j * 4)) & 0xF) as i32;
                        let q_val = (u4 - 8) as f32;
                        let x_val = bf16_to_f32(*activation.add(k_base + j as usize));
                        group_acc += q_val * x_val;
                    }
                }
                acc += group_acc * scale;
            }
            *output.add(rem_start + r) = acc;
        }
    }
}

/// Safe wrapper for AVX2 FMA transposed INT4 matmul.
pub fn matmul_int4_transposed_avx2(
    packed: &[u32],
    scales: &[u16],
    activation: &[u16],
    output: &mut [f32],
    k: usize,
    n: usize,
    group_size: usize,
) {
    assert_eq!(activation.len(), k);
    assert_eq!(output.len(), n);
    assert_eq!(packed.len(), (k / 8) * n);
    assert_eq!(scales.len(), (k / group_size) * n);
    assert!(k % group_size == 0);
    assert!(k % 8 == 0);

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("AVX2 + FMA required");
    }

    unsafe {
        expert_matmul_int4_transposed(
            packed.as_ptr(), scales.as_ptr(), activation.as_ptr(),
            output.as_mut_ptr(), k, n, 0, n, group_size,
        );
    }
}

/// Parallel AVX2 FMA transposed INT4 matmul — splits N outputs across rayon threads.
pub fn matmul_int4_transposed_parallel(
    packed: &[u32],
    scales: &[u16],
    activation: &[u16],
    output: &mut [f32],
    k: usize,
    n: usize,
    group_size: usize,
) {
    use rayon::prelude::*;

    assert_eq!(activation.len(), k);
    assert_eq!(output.len(), n);
    assert!(k % group_size == 0);
    assert!(k % 8 == 0);

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("AVX2 + FMA required");
    }

    if n <= 64 {
        unsafe {
            expert_matmul_int4_transposed(
                packed.as_ptr(), scales.as_ptr(), activation.as_ptr(),
                output.as_mut_ptr(), k, n, 0, n, group_size,
            );
        }
        return;
    }

    let chunk_n = 256; // N outputs per chunk (should be multiple of 8)
    let packed_addr = packed.as_ptr() as usize;
    let scales_addr = scales.as_ptr() as usize;
    let act_addr = activation.as_ptr() as usize;

    output.par_chunks_mut(chunk_n).enumerate().for_each(|(chunk_idx, chunk)| {
        let n_start = chunk_idx * chunk_n;
        let n_count = chunk.len();

        unsafe {
            expert_matmul_int4_transposed(
                packed_addr as *const u32,
                scales_addr as *const u16,
                act_addr as *const u16,
                chunk.as_mut_ptr(),
                k, n, n_start, n_count, group_size,
            );
        }
    });
}

/// AVX2 integer transposed INT4 matmul using `_mm256_mullo_epi32`.
///
/// Uses INT32 accumulation for the inner loop. At group boundaries,
/// converts to f32 and applies combined weight_scale × activation_scale.
/// The combined scale is a vector of 8 values (one per N output) because
/// each N output has a different weight scale for the same K group.
///
/// # Safety
/// Requires AVX2 + FMA. All pointers must be valid.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn expert_matmul_int4_transposed_integer(
    packed: *const u32,        // [K/8, n_stride]
    weight_scales: *const u16, // [K/group_size, n_stride] BF16
    act_int16: *const i16,     // [K] quantized INT16 activations
    act_scales: *const f32,    // [K/group_size] activation scales
    output: *mut f32,          // [n_out]
    k: usize,
    n_stride: usize,
    n_start: usize,
    n_out: usize,
    group_size: usize,
) {
    let num_groups = k / group_size;
    let packs_per_group = group_size / 8;

    let mask_0f = _mm256_set1_epi32(0xF);
    let offset_8 = _mm256_set1_epi32(8);

    let n_blocks = n_out / 8;
    let n_rem = n_out % 8;

    for nb in 0..n_blocks {
        let n_base = n_start + nb * 8;
        let mut float_acc = _mm256_setzero_ps();

        for g in 0..num_groups {
            let mut int_acc = _mm256_setzero_si256();

            for pack in 0..packs_per_group {
                let k_row = g * packs_per_group + pack;
                let k_base = k_row * 8;

                let words = _mm256_loadu_si256(
                    packed.add(k_row * n_stride + n_base) as *const __m256i,
                );

                for j in 0..8i32 {
                    let shift = _mm256_set1_epi32(j * 4);
                    let shifted = _mm256_srlv_epi32(words, shift);
                    let masked = _mm256_and_si256(shifted, mask_0f);
                    let signed_i32 = _mm256_sub_epi32(masked, offset_8);

                    // Broadcast INT16 activation → INT32, multiply, accumulate
                    let act_val = *act_int16.add(k_base + j as usize) as i32;
                    let act_broadcast = _mm256_set1_epi32(act_val);

                    int_acc = _mm256_add_epi32(
                        int_acc,
                        _mm256_mullo_epi32(signed_i32, act_broadcast),
                    );
                }
            }

            // Convert INT32 → f32, apply combined weight × activation scale
            let group_f32 = _mm256_cvtepi32_ps(int_acc);
            let w_scales_bf16 = _mm_loadu_si128(
                weight_scales.add(g * n_stride + n_base) as *const __m128i,
            );
            let w_scales_u32 = _mm256_cvtepu16_epi32(w_scales_bf16);
            let w_scale_vec = _mm256_castsi256_ps(_mm256_slli_epi32(w_scales_u32, 16));
            let a_scale = _mm256_set1_ps(*act_scales.add(g));
            let combined = _mm256_mul_ps(w_scale_vec, a_scale);

            float_acc = _mm256_fmadd_ps(group_f32, combined, float_acc);
        }

        _mm256_storeu_ps(output.add(nb * 8), float_acc);
    }

    // Handle remainder N positions with scalar code
    if n_rem > 0 {
        let rem_start = n_blocks * 8;
        for r in 0..n_rem {
            let n_pos = n_start + rem_start + r;
            let mut acc: f32 = 0.0;
            for g in 0..num_groups {
                let w_scale = bf16_to_f32(*weight_scales.add(g * n_stride + n_pos));
                let a_scale = *act_scales.add(g);
                let combined = w_scale * a_scale;
                let mut group_sum: i32 = 0;
                for pack in 0..packs_per_group {
                    let k_row = g * packs_per_group + pack;
                    let k_base = k_row * 8;
                    let word = *packed.add(k_row * n_stride + n_pos);
                    for j in 0..8u32 {
                        let u4 = ((word >> (j * 4)) & 0xF) as i32;
                        let w_val = u4 - 8;
                        let a_val = *act_int16.add(k_base + j as usize) as i32;
                        group_sum += w_val * a_val;
                    }
                }
                acc += group_sum as f32 * combined;
            }
            *output.add(rem_start + r) = acc;
        }
    }
}

/// Safe wrapper for AVX2 integer transposed INT4 matmul.
pub fn matmul_int4_transposed_integer(
    packed: &[u32],
    scales: &[u16],
    act_int16: &[i16],
    act_scales: &[f32],
    output: &mut [f32],
    k: usize,
    n: usize,
    group_size: usize,
) {
    assert_eq!(act_int16.len(), k);
    assert_eq!(act_scales.len(), k / group_size);
    assert_eq!(output.len(), n);
    assert_eq!(packed.len(), (k / 8) * n);
    assert_eq!(scales.len(), (k / group_size) * n);
    assert!(k % group_size == 0);
    assert!(group_size % 8 == 0);

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("AVX2 + FMA required");
    }

    unsafe {
        expert_matmul_int4_transposed_integer(
            packed.as_ptr(), scales.as_ptr(), act_int16.as_ptr(),
            act_scales.as_ptr(), output.as_mut_ptr(),
            k, n, 0, n, group_size,
        );
    }
}

/// Parallel AVX2 integer transposed INT4 matmul.
pub fn matmul_int4_transposed_integer_parallel(
    packed: &[u32],
    scales: &[u16],
    act_int16: &[i16],
    act_scales: &[f32],
    output: &mut [f32],
    k: usize,
    n: usize,
    group_size: usize,
) {
    use rayon::prelude::*;

    assert_eq!(act_int16.len(), k);
    assert_eq!(act_scales.len(), k / group_size);
    assert_eq!(output.len(), n);
    assert!(k % group_size == 0);
    assert!(group_size % 8 == 0);

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("AVX2 + FMA required");
    }

    if n <= 64 {
        unsafe {
            expert_matmul_int4_transposed_integer(
                packed.as_ptr(), scales.as_ptr(), act_int16.as_ptr(),
                act_scales.as_ptr(), output.as_mut_ptr(),
                k, n, 0, n, group_size,
            );
        }
        return;
    }

    let chunk_n = 256;
    let packed_addr = packed.as_ptr() as usize;
    let scales_addr = scales.as_ptr() as usize;
    let act_addr = act_int16.as_ptr() as usize;
    let act_scales_addr = act_scales.as_ptr() as usize;

    output.par_chunks_mut(chunk_n).enumerate().for_each(|(chunk_idx, chunk)| {
        let n_start = chunk_idx * chunk_n;
        let n_count = chunk.len();

        unsafe {
            expert_matmul_int4_transposed_integer(
                packed_addr as *const u32,
                scales_addr as *const u16,
                act_addr as *const i16,
                act_scales_addr as *const f32,
                chunk.as_mut_ptr(),
                k, n, n_start, n_count, group_size,
            );
        }
    });
}

// ── Transposed layout INT8 kernels (for CPU-optimized weight format) ────
//
// Weight layout: [K, N] — K is reduction dimension (outer), N is output
// dimension (inner, contiguous). Same SIMD-across-N approach as INT4 transposed.
//
// INT8 is simpler than INT4: no nibble extraction, direct i8 loads.
// Uses _mm256_madd_epi16 to process 2 K positions per iteration by
// interleaving weights from adjacent K rows.

/// AVX2 integer transposed INT8 matmul using `_mm256_madd_epi16`.
///
/// Weight layout: data[K, N] as i8, scales[K/group_size, N] as BF16.
/// Processes 8 N-outputs at a time, 2 K positions per iteration.
///
/// # Safety
/// Requires AVX2 + FMA. All pointers must be valid.
/// `group_size` must be even and divisible by 2.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn expert_matmul_int8_transposed_integer(
    data: *const i8,           // [K, n_stride] raw INT8 weights
    weight_scales: *const u16, // [K/group_size, n_stride] BF16 scales
    act_int16: *const i16,     // [K] quantized INT16 activations
    act_scales: *const f32,    // [K/group_size] activation scales
    output: *mut f32,          // [n_out]
    k: usize,
    n_stride: usize,
    n_start: usize,
    n_out: usize,
    group_size: usize,
) {
    let num_groups = k / group_size;
    let pairs_per_group = group_size / 2;

    let n_blocks = n_out / 8;
    let n_rem = n_out % 8;

    for nb in 0..n_blocks {
        let n_base = n_start + nb * 8;
        let mut float_acc = _mm256_setzero_ps();

        for g in 0..num_groups {
            let mut int_acc = _mm256_setzero_si256();

            for p in 0..pairs_per_group {
                let k0 = g * group_size + p * 2;
                let k1 = k0 + 1;

                // Load 8 i8 values from each K row (64 bits each)
                let row0 = _mm_loadl_epi64(
                    data.add(k0 * n_stride + n_base) as *const __m128i,
                );
                let row1 = _mm_loadl_epi64(
                    data.add(k1 * n_stride + n_base) as *const __m128i,
                );

                // Interleave bytes: [w[k0,0], w[k1,0], w[k0,1], w[k1,1], ...]
                let interleaved = _mm_unpacklo_epi8(row0, row1);

                // Sign-extend to INT16: 16 values in 256-bit register
                let w16 = _mm256_cvtepi8_epi16(interleaved);

                // Activation pair broadcast: [act[k0], act[k1]] repeated 8 times
                let a0 = *act_int16.add(k0) as u16;
                let a1 = *act_int16.add(k1) as u16;
                let combined = (a0 as u32) | ((a1 as u32) << 16);
                let act_pair = _mm256_set1_epi32(combined as i32);

                // madd_epi16: 16 (INT16 × INT16) → 8 INT32 partial sums
                let dot = _mm256_madd_epi16(w16, act_pair);
                int_acc = _mm256_add_epi32(int_acc, dot);
            }

            // Convert INT32 → f32, apply combined weight × activation scale
            let group_f32 = _mm256_cvtepi32_ps(int_acc);
            let w_scales_bf16 = _mm_loadu_si128(
                weight_scales.add(g * n_stride + n_base) as *const __m128i,
            );
            let w_scales_u32 = _mm256_cvtepu16_epi32(w_scales_bf16);
            let w_scale_vec = _mm256_castsi256_ps(_mm256_slli_epi32(w_scales_u32, 16));
            let a_scale = _mm256_set1_ps(*act_scales.add(g));
            let combined = _mm256_mul_ps(w_scale_vec, a_scale);

            float_acc = _mm256_fmadd_ps(group_f32, combined, float_acc);
        }

        _mm256_storeu_ps(output.add(nb * 8), float_acc);
    }

    // Handle remainder N positions with scalar code
    if n_rem > 0 {
        let rem_start = n_blocks * 8;
        for r in 0..n_rem {
            let n_pos = n_start + rem_start + r;
            let mut acc: f32 = 0.0;
            for g in 0..num_groups {
                let w_scale = bf16_to_f32(*weight_scales.add(g * n_stride + n_pos));
                let a_scale = *act_scales.add(g);
                let combined = w_scale * a_scale;
                let mut group_sum: i32 = 0;
                for ki in 0..group_size {
                    let k_pos = g * group_size + ki;
                    let w_val = *data.add(k_pos * n_stride + n_pos) as i32;
                    let a_val = *act_int16.add(k_pos) as i32;
                    group_sum += w_val * a_val;
                }
                acc += group_sum as f32 * combined;
            }
            *output.add(rem_start + r) = acc;
        }
    }
}

/// Safe wrapper for AVX2 integer transposed INT8 matmul.
///
/// Weight layout: data as i8 packed into u32 in [K, N] layout, scales [K/gs, N] as BF16.
/// The u32 vec is just a byte container — actual data is i8.
pub fn matmul_int8_transposed_integer(
    data_u32: &[u32],     // [K, N] as i8 packed into u32 (byte container)
    scales: &[u16],        // [K/group_size, N] BF16
    act_int16: &[i16],     // [K]
    act_scales: &[f32],    // [K/group_size]
    output: &mut [f32],    // [N]
    k: usize,
    n: usize,
    group_size: usize,
) {
    assert_eq!(act_int16.len(), k);
    assert_eq!(act_scales.len(), k / group_size);
    assert_eq!(output.len(), n);
    // INT8: K*N bytes = K*N/4 u32s
    assert_eq!(data_u32.len(), (k * n + 3) / 4, "data_u32 len mismatch: expected {}, got {}", (k * n + 3) / 4, data_u32.len());
    assert_eq!(scales.len(), (k / group_size) * n);
    assert!(k % group_size == 0);
    assert!(group_size % 2 == 0, "Transposed INT8 kernel requires even group_size");

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("AVX2 + FMA required");
    }

    unsafe {
        expert_matmul_int8_transposed_integer(
            data_u32.as_ptr() as *const i8,
            scales.as_ptr(), act_int16.as_ptr(),
            act_scales.as_ptr(), output.as_mut_ptr(),
            k, n, 0, n, group_size,
        );
    }
}

/// Parallel AVX2 integer transposed INT8 matmul.
///
/// Splits work by N columns across rayon threads (same strategy as INT4 transposed).
pub fn matmul_int8_transposed_integer_parallel(
    data_u32: &[u32],
    scales: &[u16],
    act_int16: &[i16],
    act_scales: &[f32],
    output: &mut [f32],
    k: usize,
    n: usize,
    group_size: usize,
) {
    use rayon::prelude::*;

    assert_eq!(act_int16.len(), k);
    assert_eq!(act_scales.len(), k / group_size);
    assert_eq!(output.len(), n);
    assert!(k % group_size == 0);
    assert!(group_size % 2 == 0);

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("AVX2 + FMA required");
    }

    if n <= 64 {
        unsafe {
            expert_matmul_int8_transposed_integer(
                data_u32.as_ptr() as *const i8,
                scales.as_ptr(), act_int16.as_ptr(),
                act_scales.as_ptr(), output.as_mut_ptr(),
                k, n, 0, n, group_size,
            );
        }
        return;
    }

    let chunk_n = 256;
    let data_addr = data_u32.as_ptr() as usize;
    let scales_addr = scales.as_ptr() as usize;
    let act_addr = act_int16.as_ptr() as usize;
    let act_scales_addr = act_scales.as_ptr() as usize;

    output.par_chunks_mut(chunk_n).enumerate().for_each(|(chunk_idx, chunk)| {
        let n_start = chunk_idx * chunk_n;
        let n_count = chunk.len();

        unsafe {
            expert_matmul_int8_transposed_integer(
                data_addr as *const i8,
                scales_addr as *const u16,
                act_addr as *const i16,
                act_scales_addr as *const f32,
                chunk.as_mut_ptr(),
                k, n, n_start, n_count, group_size,
            );
        }
    });
}

// ============================================================================
// Marlin-native INT4 kernel — reads GPU-native Marlin-packed data directly
// ============================================================================

/// Precomputed mapping from Marlin packed positions to tile coordinates.
///
/// For each of the 1024 positions in a permutation chunk (128 u32 words × 8 INT4 values):
/// - `k_off[i]`: K-offset within the 16-value tile (0..15)
/// - `n_off[i]`: N-offset within the 64-value chunk (0..63)
pub struct MarlinTileMap {
    pub k_off: [u8; 1024],
    pub n_off: [u8; 1024],
}

/// Precomputed inverse scale permutation for reading Marlin-permuted scales.
pub struct MarlinScaleMap {
    pub inv_perm: [usize; 64],
}

/// Build the Marlin tile map from the weight permutation table.
///
/// The weight permutation (from `generate_weight_perm_int4()`) maps packed positions
/// to tile-transposed positions. Each entry `perm[i]` encodes:
///   - `perm[i] / 256` = which N-tile within the 4-tile chunk (0..3)
///   - `(perm[i] % 256) / 16` = K-offset within tile (0..15)
///   - `perm[i] % 16` = N-offset within N-tile (0..15)
pub fn build_marlin_tile_map() -> MarlinTileMap {
    use crate::weights::marlin::generate_weight_perm_int4;
    let perm = generate_weight_perm_int4();
    let mut map = MarlinTileMap {
        k_off: [0u8; 1024],
        n_off: [0u8; 1024],
    };
    for i in 0..1024 {
        let p = perm[i];
        let nt_local = p / 256;        // which N-tile (0..3)
        let tk = (p % 256) / 16;       // K-offset in tile (0..15)
        let tn = p % 16;               // N-offset in N-tile (0..15)
        map.k_off[i] = tk as u8;
        map.n_off[i] = (nt_local * 16 + tn) as u8;
    }
    map
}

/// Build the inverse scale permutation for grouped quantization.
pub fn build_marlin_scale_map() -> MarlinScaleMap {
    use crate::weights::marlin::generate_scale_perms;
    let (scale_perm, _) = generate_scale_perms();
    let mut inv = [0usize; 64];
    for i in 0..64 {
        inv[scale_perm[i]] = i;
    }
    MarlinScaleMap { inv_perm: inv }
}

/// Read a single BF16 scale value from Marlin-permuted scale array.
///
/// Scales are stored as flat `[K/gs, N]` with 64-element permutation chunks.
#[inline]
fn read_marlin_scale(scales: &[u16], group: usize, n_pos: usize, n_total: usize, inv_sperm: &[usize; 64]) -> f32 {
    let flat = group * n_total + n_pos;
    let chunk = flat / 64;
    let local = flat % 64;
    bf16_to_f32(scales[chunk * 64 + inv_sperm[local]])
}

/// Scalar Marlin-native INT4 matmul for correctness verification.
///
/// Reads Marlin-packed `[K/16, 2*N]` weights and Marlin-permuted `[K/gs, N]` scales.
/// Computes output[n] = Σ_k weight[k,n] × act[k] with per-group scaling.
pub fn matmul_int4_marlin_scalar(
    packed: &[u32],        // [K/16, 2*N] Marlin-packed
    scales: &[u16],        // [K/gs, N] Marlin-permuted BF16
    act_int16: &[i16],     // [K] INT16 activations
    act_scales: &[f32],    // [K/gs] activation scales
    output: &mut [f32],    // [N]
    k: usize,
    n: usize,
    group_size: usize,
) {
    assert_eq!(act_int16.len(), k);
    assert_eq!(act_scales.len(), k / group_size);
    assert_eq!(output.len(), n);
    assert!(k % 16 == 0, "K must be divisible by 16 (Marlin tile)");
    assert!(n % 64 == 0, "N must be divisible by 64 (Marlin constraint)");
    assert!(group_size % 16 == 0);

    assert!(group_size < k, "Channelwise (group_size == K) not supported; Marlin uses different scale permutation");

    let tile_map = build_marlin_tile_map();
    let scale_map = build_marlin_scale_map();

    let out_cols = 2 * n;  // packed row width in u32
    let n_chunks = n / 64;
    let tiles_per_group = group_size / 16;
    let num_groups = k / group_size;

    output.fill(0.0);

    for nc in 0..n_chunks {
        let n_base = nc * 64;

        for group in 0..num_groups {
            let mut int_acc = [0i32; 64];

            for tile_in_group in 0..tiles_per_group {
                let kt = group * tiles_per_group + tile_in_group;
                let word_base = kt * out_cols + nc * 128;

                for w in 0..128usize {
                    let word = packed[word_base + w];
                    for b in 0..8u32 {
                        let idx = w * 8 + b as usize;
                        let val = ((word >> (b * 4)) & 0xF) as i32 - 8;
                        let tk = tile_map.k_off[idx] as usize;
                        let n_off = tile_map.n_off[idx] as usize;
                        let k_idx = kt * 16 + tk;
                        int_acc[n_off] += val * act_int16[k_idx] as i32;
                    }
                }
            }

            // Apply scales: weight_scale * act_scale
            let a_scale = act_scales[group];
            for i in 0..64 {
                let w_scale = read_marlin_scale(scales, group, n_base + i, n, &scale_map.inv_perm);
                output[n_base + i] += int_acc[i] as f32 * w_scale * a_scale;
            }
        }
    }
}

/// AVX2 Marlin-native INT4 matmul — production kernel.
///
/// Processes Marlin tiles: for each (K-group × N-chunk-of-64), unpacks 1024 INT4 values
/// from 128 u32 words, applies the inverse permutation to get a clean [16, 64] tile,
/// then accumulates integer products. Scale applied once per group.
///
/// # Arguments
/// * `packed` - Marlin-packed weights `[K/16, 2*N]` u32
/// * `scales` - Marlin-permuted scales `[K/gs, N]` BF16
/// * `act_int16` - INT16-quantized activations `[K]`
/// * `act_scales` - Per-group activation scales `[K/gs]`
/// * `output` - Output buffer `[n_out]`
/// * `k` - Reduction dimension (K)
/// * `n_stride` - Full N dimension of the weight matrix
/// * `n_start` - Starting N offset for this chunk
/// * `n_out` - Number of N outputs to compute
/// * `group_size` - Quantization group size
/// * `tile_map` - Precomputed Marlin tile map
/// * `scale_map` - Precomputed inverse scale permutation
pub unsafe fn expert_matmul_int4_marlin(
    packed: *const u32,
    scales: *const u16,
    act_int16: *const i16,
    act_scales: *const f32,
    output: *mut f32,
    k: usize,
    n_stride: usize,
    n_start: usize,
    n_out: usize,
    group_size: usize,
    tile_map: &MarlinTileMap,
    scale_map: &MarlinScaleMap,
) {
    let out_cols = 2 * n_stride;
    let tiles_per_group = group_size / 16;
    let num_groups = k / group_size;

    // Process 64 N-positions at a time (one permutation chunk)
    let n_full_chunks = n_out / 64;
    let n_rem = n_out % 64;

    for nc_local in 0..n_full_chunks {
        let n_base = n_start + nc_local * 64;
        let nc_in_full = n_base / 64; // chunk index in the full N dimension

        // Float accumulators for this N-chunk (8 × __m256 = 64 f32)
        let mut float_acc = [_mm256_setzero_ps(); 8];

        for group in 0..num_groups {
            // Integer accumulators: 8 × __m256i = 64 i32 values
            let mut int_acc = [_mm256_setzero_si256(); 8];

            for tile_in_group in 0..tiles_per_group {
                let kt = group * tiles_per_group + tile_in_group;
                let word_base = kt * out_cols + nc_in_full * 128;

                // Unpack + accumulate: process 128 u32 words for this tile
                // Each word has 8 INT4 values mapped via tile_map to (k_off, n_off)
                for w in 0..128usize {
                    let word = *packed.add(word_base + w);
                    for b in 0..8u32 {
                        let idx = w * 8 + b as usize;
                        let val = ((word >> (b * 4)) & 0xF) as i32 - 8;
                        let tk = *tile_map.k_off.get_unchecked(idx) as usize;
                        let n_off = *tile_map.n_off.get_unchecked(idx) as usize;
                        let k_idx = kt * 16 + tk;
                        let act_val = *act_int16.add(k_idx) as i32;

                        // Accumulate into the right position in int_acc
                        let acc_idx = n_off / 8;
                        let lane = n_off % 8;
                        // Scalar accumulation into the SIMD accumulator's lane
                        let acc_arr = &mut int_acc[acc_idx] as *mut __m256i as *mut i32;
                        *acc_arr.add(lane) += val * act_val;
                    }
                }
            }

            // Apply scales: weight_scale[group, n] * act_scale[group]
            let a_scale = *act_scales.add(group);
            let a_scale_vec = _mm256_set1_ps(a_scale);

            for i in 0..8 {
                let n_pos_base = n_base + i * 8;

                // Load 8 weight scales for this group and N-positions
                let mut w_scales = [0.0f32; 8];
                for j in 0..8 {
                    let n_pos = n_pos_base + j;
                    let flat = group * n_stride + n_pos;
                    let chunk = flat / 64;
                    let local = flat % 64;
                    let perm_idx = *scale_map.inv_perm.get_unchecked(local);
                    w_scales[j] = bf16_to_f32(*scales.add(chunk * 64 + perm_idx));
                }
                let w_scale_vec = _mm256_loadu_ps(w_scales.as_ptr());
                let combined = _mm256_mul_ps(w_scale_vec, a_scale_vec);

                // Convert int_acc to f32 and FMA into float_acc
                let int_f32 = _mm256_cvtepi32_ps(int_acc[i]);
                float_acc[i] = _mm256_fmadd_ps(int_f32, combined, float_acc[i]);
            }

            // Reset integer accumulators for next group
            for i in 0..8 {
                int_acc[i] = _mm256_setzero_si256();
            }
        }

        // Store results
        for i in 0..8 {
            _mm256_storeu_ps(output.add(nc_local * 64 + i * 8), float_acc[i]);
        }
    }

    // Handle remainder N positions (if N not divisible by 64 — shouldn't happen for Marlin)
    if n_rem > 0 {
        let n_base = n_start + n_full_chunks * 64;
        for i in 0..n_rem {
            let n_pos = n_base + i;
            let mut acc = 0.0f32;

            for group in 0..num_groups {
                // Fall back to looking up individual values via dequantize logic
                let a_scale = *act_scales.add(group);
                let flat_scale = group * n_stride + n_pos;
                let chunk = flat_scale / 64;
                let local = flat_scale % 64;
                let perm_idx = scale_map.inv_perm[local];
                let w_scale = bf16_to_f32(*scales.add(chunk * 64 + perm_idx));

                let mut group_sum = 0i32;
                for ki in 0..group_size {
                    let k_abs = group * group_size + ki;
                    // Need to find this (k_abs, n_pos) in Marlin layout — expensive, only for remainder
                    // Use dequantize_marlin logic
                    let kt = k_abs / 16;
                    let _tk = k_abs % 16;
                    // This is complex for remainder — but N % 64 == 0 for Marlin, so this never runs
                    let _ = kt;
                    group_sum += 0; // placeholder — Marlin requires N % 64 == 0
                }
                acc += group_sum as f32 * w_scale * a_scale;
            }
            *output.add(n_full_chunks * 64 + i) = acc;
        }
    }
}

/// Safe wrapper for Marlin-native INT4 matmul (sequential).
pub fn matmul_int4_marlin(
    packed: &[u32],
    scales: &[u16],
    act_int16: &[i16],
    act_scales: &[f32],
    output: &mut [f32],
    k: usize,
    n: usize,
    group_size: usize,
    tile_map: &MarlinTileMap,
    scale_map: &MarlinScaleMap,
) {
    assert_eq!(act_int16.len(), k);
    assert_eq!(act_scales.len(), k / group_size);
    assert_eq!(output.len(), n);
    assert!(k % 16 == 0);
    assert!(n % 64 == 0);
    assert!(group_size % 16 == 0);

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("AVX2 + FMA required");
    }

    unsafe {
        expert_matmul_int4_marlin(
            packed.as_ptr(), scales.as_ptr(), act_int16.as_ptr(),
            act_scales.as_ptr(), output.as_mut_ptr(),
            k, n, 0, n, group_size, tile_map, scale_map,
        );
    }
}

/// Safe wrapper for Marlin-native INT4 matmul (parallel across N dimension).
pub fn matmul_int4_marlin_parallel(
    packed: &[u32],
    scales: &[u16],
    act_int16: &[i16],
    act_scales: &[f32],
    output: &mut [f32],
    k: usize,
    n: usize,
    group_size: usize,
    tile_map: &MarlinTileMap,
    scale_map: &MarlinScaleMap,
) {
    use rayon::prelude::*;

    assert_eq!(act_int16.len(), k);
    assert_eq!(act_scales.len(), k / group_size);
    assert_eq!(output.len(), n);
    assert!(k % 16 == 0);
    assert!(n % 64 == 0);
    assert!(group_size % 16 == 0);

    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("AVX2 + FMA required");
    }

    if n <= 64 {
        unsafe {
            expert_matmul_int4_marlin(
                packed.as_ptr(), scales.as_ptr(), act_int16.as_ptr(),
                act_scales.as_ptr(), output.as_mut_ptr(),
                k, n, 0, n, group_size, tile_map, scale_map,
            );
        }
        return;
    }

    let chunk_n = 256; // Must be multiple of 64
    let packed_addr = packed.as_ptr() as usize;
    let scales_addr = scales.as_ptr() as usize;
    let act_addr = act_int16.as_ptr() as usize;
    let act_scales_addr = act_scales.as_ptr() as usize;

    output.par_chunks_mut(chunk_n).enumerate().for_each(|(chunk_idx, chunk)| {
        let n_start = chunk_idx * chunk_n;
        let n_count = chunk.len();
        // n_count should always be multiple of 64 since N % 64 == 0 and chunk_n % 64 == 0
        assert!(n_count % 64 == 0);

        unsafe {
            expert_matmul_int4_marlin(
                packed_addr as *const u32,
                scales_addr as *const u16,
                act_addr as *const i16,
                act_scales_addr as *const f32,
                chunk.as_mut_ptr(),
                k, n, n_start, n_count, group_size,
                tile_map, scale_map,
            );
        }
    });
}

/// Transpose a QuantizedInt4 from [N, K/8] layout to [K/8, N] layout.
///
/// Used to convert from the original row-major format to the transposed format
/// needed by the unified weight kernels. Pure rearrangement of u32/u16 elements.
pub fn transpose_int4(q: &QuantizedInt4) -> (Vec<u32>, Vec<u16>) {
    let n = q.rows;
    let packed_k = q.cols / 8;
    let num_groups = q.cols / q.group_size;

    // Transpose packed: [N, K/8] → [K/8, N]
    let mut t_packed = vec![0u32; packed_k * n];
    for row in 0..n {
        for col in 0..packed_k {
            t_packed[col * n + row] = q.packed[row * packed_k + col];
        }
    }

    // Transpose scales: [N, K/group_size] → [K/group_size, N]
    let mut t_scales = vec![0u16; num_groups * n];
    for row in 0..n {
        for col in 0..num_groups {
            t_scales[col * n + row] = q.scales[row * num_groups + col];
        }
    }

    (t_packed, t_scales)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::marlin::{f32_to_bf16, quantize_int4, DEFAULT_GROUP_SIZE};
    use crate::weights::safetensors_io::MmapSafetensors;
    use std::path::Path;

    #[test]
    fn test_avx2_available() {
        assert!(is_x86_feature_detected!("avx2"), "AVX2 required");
        assert!(is_x86_feature_detected!("fma"), "FMA required");
    }

    #[test]
    fn test_scalar_int4_vs_avx2_synthetic() {
        let n = 16;
        let k = 128;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i as f32 / weight_bf16.len() as f32) - 0.5) * 0.2;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i as f32 / k as f32) - 0.5) * 2.0;
            activation[i] = f32_to_bf16(val);
        }

        // BF16 reference
        let mut ref_output = vec![0.0f32; n];
        matmul_bf16_scalar(&weight_bf16, &activation, &mut ref_output, n, k);

        // Quantize
        let q = quantize_int4(&weight_bf16, n, k, group_size);

        // Scalar INT4
        let mut scalar_output = vec![0.0f32; n];
        matmul_int4_scalar(&q, &activation, &mut scalar_output);

        // AVX2 INT4
        let mut avx2_output = vec![0.0f32; n];
        matmul_int4_avx2(&q, &activation, &mut avx2_output);

        // Scalar vs AVX2: may differ slightly due to FMA rounding
        let mut max_diff: f32 = 0.0;
        for i in 0..n {
            let diff = (scalar_output[i] - avx2_output[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("Synthetic {n}×{k}: scalar vs AVX2 max_diff={max_diff:.8}");
        assert!(max_diff < 1e-3, "Scalar vs AVX2 diverged: {max_diff}");

        // INT4 vs BF16: quantization error
        let mut max_err: f32 = 0.0;
        for i in 0..n {
            let err = (ref_output[i] - avx2_output[i]).abs();
            max_err = max_err.max(err);
        }
        eprintln!("Synthetic {n}×{k}: INT4 vs BF16 max_err={max_err:.6}");
        assert!(max_err < 1.0, "Quantization error too large: {max_err}");
    }

    #[test]
    fn test_avx2_multi_group() {
        let n = 4;
        let k = 256;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 3 + 7) as f32 / weight_bf16.len() as f32 - 0.5) * 0.4;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 11 + 3) as f32 / k as f32 - 0.5) * 0.5;
            activation[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&weight_bf16, n, k, group_size);

        let mut scalar_output = vec![0.0f32; n];
        matmul_int4_scalar(&q, &activation, &mut scalar_output);

        let mut avx2_output = vec![0.0f32; n];
        matmul_int4_avx2(&q, &activation, &mut avx2_output);

        let mut max_diff: f32 = 0.0;
        for i in 0..n {
            let diff = (scalar_output[i] - avx2_output[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("Multi-group {n}×{k} (2 groups): scalar vs AVX2 max_diff={max_diff:.8}");
        assert!(max_diff < 1e-3, "Multi-group mismatch: {max_diff}");
    }

    #[test]
    fn test_v2_lite_expert_matmul() {
        let path = Path::new(
            "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite/model-00001-of-000004.safetensors",
        );
        if !path.exists() {
            eprintln!("Skipping — V2-Lite not downloaded");
            return;
        }

        let st = MmapSafetensors::open(path).expect("Failed to open");
        let gate_name = "model.layers.1.mlp.experts.0.gate_proj.weight";
        let bf16_data: &[u16] = st.tensor_as_slice(gate_name).expect("Failed to read");
        let info = st.tensor_info(gate_name).unwrap();

        let n = info.shape[0]; // 1408
        let k = info.shape[1]; // 2048

        // Synthetic activation with moderate magnitudes
        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 7 + 13) as f32 / k as f32 - 0.5) * 0.1;
            activation[i] = f32_to_bf16(val);
        }

        // BF16 reference
        let mut ref_output = vec![0.0f32; n];
        matmul_bf16_scalar(bf16_data, &activation, &mut ref_output, n, k);

        // Quantize to INT4
        let q = quantize_int4(bf16_data, n, k, DEFAULT_GROUP_SIZE);

        // Scalar INT4
        let mut scalar_output = vec![0.0f32; n];
        matmul_int4_scalar(&q, &activation, &mut scalar_output);

        // AVX2 INT4
        let mut avx2_output = vec![0.0f32; n];
        matmul_int4_avx2(&q, &activation, &mut avx2_output);

        // Scalar vs AVX2
        let mut max_diff_sa: f32 = 0.0;
        for i in 0..n {
            max_diff_sa = max_diff_sa.max((scalar_output[i] - avx2_output[i]).abs());
        }

        // INT4 vs BF16 reference
        let mut max_err: f32 = 0.0;
        let mut sum_sq_err: f64 = 0.0;
        let mut sum_sq_ref: f64 = 0.0;
        for i in 0..n {
            let err = (ref_output[i] - avx2_output[i]).abs();
            max_err = max_err.max(err);
            sum_sq_err += (err as f64).powi(2);
            sum_sq_ref += (ref_output[i] as f64).powi(2);
        }
        let rmse = (sum_sq_err / n as f64).sqrt();
        let rms_ref = (sum_sq_ref / n as f64).sqrt();
        let snr_db = 20.0 * (rms_ref / rmse).log10();

        eprintln!("V2-Lite gate_proj [{n}, {k}] matmul:");
        eprintln!("  Scalar vs AVX2 max_diff: {max_diff_sa:.8}");
        eprintln!("  INT4 vs BF16: max_err={max_err:.6}, RMSE={rmse:.6}, SNR={snr_db:.1} dB");

        assert!(max_diff_sa < 0.01, "Scalar vs AVX2 diverged: {max_diff_sa}");
        assert!(snr_db > 10.0, "SNR too low: {snr_db:.1} dB");
    }

    #[test]
    fn test_avx2_throughput() {
        // Benchmark: time the AVX2 kernel on V2-Lite-sized expert
        let n = 1408;
        let k = 2048;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 37 + 11) as f32 / weight_bf16.len() as f32 - 0.5) * 0.1;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 7 + 3) as f32 / k as f32 - 0.5) * 0.2;
            activation[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&weight_bf16, n, k, group_size);
        let mut output = vec![0.0f32; n];

        // Warmup
        matmul_int4_avx2(&q, &activation, &mut output);

        let iters = 100;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            matmul_int4_avx2(&q, &activation, &mut output);
        }
        let elapsed = start.elapsed();

        let us_per_call = elapsed.as_micros() as f64 / iters as f64;
        let weight_bytes = q.packed.len() * 4 + q.scales.len() * 2;
        let gb_per_sec = (weight_bytes as f64 / 1e9) / (us_per_call / 1e6);

        eprintln!(
            "AVX2 INT4 single-thread [{n}×{k}]: {us_per_call:.0} μs/call, {gb_per_sec:.1} GB/s"
        );

        // Parallel version
        matmul_int4_parallel(&q, &activation, &mut output);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            matmul_int4_parallel(&q, &activation, &mut output);
        }
        let elapsed = start.elapsed();

        let us_par = elapsed.as_micros() as f64 / iters as f64;
        let gb_par = (weight_bytes as f64 / 1e9) / (us_par / 1e6);
        let speedup = us_per_call / us_par;

        eprintln!(
            "AVX2 INT4 parallel      [{n}×{k}]: {us_par:.0} μs/call, {gb_par:.1} GB/s, {speedup:.1}x speedup"
        );
        eprintln!(
            "  Weight data: {:.1} KB, rayon threads: {}",
            weight_bytes as f64 / 1024.0,
            rayon::current_num_threads(),
        );
    }

    #[test]
    fn test_parallel_correctness() {
        let n = 1408;
        let k = 2048;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 37 + 11) as f32 / weight_bf16.len() as f32 - 0.5) * 0.1;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 7 + 3) as f32 / k as f32 - 0.5) * 0.2;
            activation[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&weight_bf16, n, k, group_size);

        let mut serial_out = vec![0.0f32; n];
        let mut parallel_out = vec![0.0f32; n];

        matmul_int4_avx2(&q, &activation, &mut serial_out);
        matmul_int4_parallel(&q, &activation, &mut parallel_out);

        let mut max_diff: f32 = 0.0;
        for i in 0..n {
            max_diff = max_diff.max((serial_out[i] - parallel_out[i]).abs());
        }
        eprintln!("Parallel vs serial max_diff: {max_diff:.8}");
        assert!(max_diff == 0.0, "Parallel should be bit-identical to serial");
    }

    // ── Integer kernel tests ──────────────────────────────────────────

    #[test]
    fn test_activation_int16_quantization() {
        let k = 256;
        let group_size = 128;

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 7 + 3) as f32 / k as f32 - 0.5) * 2.0;
            activation[i] = f32_to_bf16(val);
        }

        let mut int16_out = vec![0i16; k];
        let mut scales = vec![0.0f32; k / group_size];
        quantize_activation_int16(&activation, group_size, &mut int16_out, &mut scales);

        // Check round-trip: dequantize and compare
        let mut max_err: f32 = 0.0;
        for g in 0..(k / group_size) {
            let scale = scales[g];
            for i in 0..group_size {
                let orig = bf16_to_f32(activation[g * group_size + i]);
                let reconstructed = int16_out[g * group_size + i] as f32 * scale;
                let err = (orig - reconstructed).abs();
                max_err = max_err.max(err);
            }
        }
        eprintln!("INT16 activation round-trip max error: {max_err:.8}");
        // INT16 has 15 bits of precision, BF16 has 7 — so error should be tiny
        assert!(max_err < 0.001, "INT16 round-trip error too large: {max_err}");
    }

    #[test]
    fn test_integer_scalar_vs_avx2_synthetic() {
        let n = 16;
        let k = 128;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i as f32 / weight_bf16.len() as f32) - 0.5) * 0.2;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i as f32 / k as f32) - 0.5) * 2.0;
            activation[i] = f32_to_bf16(val);
        }

        // Quantize weights
        let q = quantize_int4(&weight_bf16, n, k, group_size);

        // Quantize activations to INT16
        let mut act_int16 = vec![0i16; k];
        let mut act_scales = vec![0.0f32; k / group_size];
        quantize_activation_int16(&activation, group_size, &mut act_int16, &mut act_scales);

        // Scalar integer
        let mut scalar_out = vec![0.0f32; n];
        matmul_int4_integer_scalar(&q, &act_int16, &act_scales, &mut scalar_out);

        // AVX2 integer
        let mut avx2_out = vec![0.0f32; n];
        matmul_int4_integer(&q, &act_int16, &act_scales, &mut avx2_out);

        let mut max_diff: f32 = 0.0;
        for i in 0..n {
            let diff = (scalar_out[i] - avx2_out[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("Integer scalar vs AVX2 [{n}×{k}]: max_diff={max_diff:.8}");
        assert!(max_diff < 1e-3, "Integer scalar vs AVX2 diverged: {max_diff}");
    }

    #[test]
    fn test_integer_vs_fma_synthetic() {
        let n = 32;
        let k = 256;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 3 + 7) as f32 / weight_bf16.len() as f32 - 0.5) * 0.4;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 11 + 3) as f32 / k as f32 - 0.5) * 0.5;
            activation[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&weight_bf16, n, k, group_size);

        // FMA kernel
        let mut fma_out = vec![0.0f32; n];
        matmul_int4_avx2(&q, &activation, &mut fma_out);

        // Integer kernel
        let mut act_int16 = vec![0i16; k];
        let mut act_scales = vec![0.0f32; k / group_size];
        quantize_activation_int16(&activation, group_size, &mut act_int16, &mut act_scales);

        let mut int_out = vec![0.0f32; n];
        matmul_int4_integer(&q, &act_int16, &act_scales, &mut int_out);

        // Compare: both are INT4 quantized, so difference is from
        // activation quantization (BF16→INT16) and accumulation method
        let mut max_diff: f32 = 0.0;
        let mut sum_sq_diff: f64 = 0.0;
        let mut sum_sq_fma: f64 = 0.0;
        for i in 0..n {
            let diff = (fma_out[i] - int_out[i]).abs();
            max_diff = max_diff.max(diff);
            sum_sq_diff += (diff as f64).powi(2);
            sum_sq_fma += (fma_out[i] as f64).powi(2);
        }
        let rmse = (sum_sq_diff / n as f64).sqrt();
        let rms_fma = (sum_sq_fma / n as f64).sqrt();
        let rel_err = rmse / rms_fma;

        eprintln!("Integer vs FMA [{n}×{k}]: max_diff={max_diff:.6}, relative RMSE={rel_err:.6}");
        // Both use same INT4 weights; difference is only from activation quantization
        assert!(rel_err < 0.01, "Integer vs FMA relative error too large: {rel_err}");
    }

    #[test]
    fn test_integer_parallel_correctness() {
        let n = 1408;
        let k = 2048;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 37 + 11) as f32 / weight_bf16.len() as f32 - 0.5) * 0.1;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 7 + 3) as f32 / k as f32 - 0.5) * 0.2;
            activation[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&weight_bf16, n, k, group_size);

        let mut act_int16 = vec![0i16; k];
        let mut act_scales = vec![0.0f32; k / group_size];
        quantize_activation_int16(&activation, group_size, &mut act_int16, &mut act_scales);

        let mut serial_out = vec![0.0f32; n];
        let mut parallel_out = vec![0.0f32; n];

        matmul_int4_integer(&q, &act_int16, &act_scales, &mut serial_out);
        matmul_int4_integer_parallel(&q, &act_int16, &act_scales, &mut parallel_out);

        let mut max_diff: f32 = 0.0;
        for i in 0..n {
            max_diff = max_diff.max((serial_out[i] - parallel_out[i]).abs());
        }
        eprintln!("Integer parallel vs serial max_diff: {max_diff:.8}");
        assert!(max_diff == 0.0, "Integer parallel should be bit-identical to serial");
    }

    #[test]
    fn test_integer_throughput() {
        // Benchmark: compare integer vs FMA kernel throughput
        let n = 1408;
        let k = 2048;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 37 + 11) as f32 / weight_bf16.len() as f32 - 0.5) * 0.1;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 7 + 3) as f32 / k as f32 - 0.5) * 0.2;
            activation[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&weight_bf16, n, k, group_size);
        let weight_bytes = q.packed.len() * 4 + q.scales.len() * 2;
        let iters = 100;

        // Quantize activation once (amortized)
        let mut act_int16 = vec![0i16; k];
        let mut act_scales = vec![0.0f32; k / group_size];
        quantize_activation_int16(&activation, group_size, &mut act_int16, &mut act_scales);

        let mut output = vec![0.0f32; n];

        // ── FMA single-thread ──
        matmul_int4_avx2(&q, &activation, &mut output);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            matmul_int4_avx2(&q, &activation, &mut output);
        }
        let fma_st_us = start.elapsed().as_micros() as f64 / iters as f64;
        let fma_st_gb = (weight_bytes as f64 / 1e9) / (fma_st_us / 1e6);

        // ── Integer single-thread ──
        matmul_int4_integer(&q, &act_int16, &act_scales, &mut output);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            matmul_int4_integer(&q, &act_int16, &act_scales, &mut output);
        }
        let int_st_us = start.elapsed().as_micros() as f64 / iters as f64;
        let int_st_gb = (weight_bytes as f64 / 1e9) / (int_st_us / 1e6);

        // ── FMA parallel ──
        matmul_int4_parallel(&q, &activation, &mut output);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            matmul_int4_parallel(&q, &activation, &mut output);
        }
        let fma_par_us = start.elapsed().as_micros() as f64 / iters as f64;
        let fma_par_gb = (weight_bytes as f64 / 1e9) / (fma_par_us / 1e6);

        // ── Integer parallel ──
        matmul_int4_integer_parallel(&q, &act_int16, &act_scales, &mut output);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            matmul_int4_integer_parallel(&q, &act_int16, &act_scales, &mut output);
        }
        let int_par_us = start.elapsed().as_micros() as f64 / iters as f64;
        let int_par_gb = (weight_bytes as f64 / 1e9) / (int_par_us / 1e6);

        eprintln!("╔══════════════════════════════════════════════════╗");
        eprintln!("║  Kernel Throughput [{n}×{k}]                      ║");
        eprintln!("╠══════════════════════════════════════════════════╣");
        eprintln!("║  FMA single-thread:     {fma_st_us:>6.0} μs  {fma_st_gb:>5.1} GB/s ║");
        eprintln!("║  Integer single-thread: {int_st_us:>6.0} μs  {int_st_gb:>5.1} GB/s ║");
        eprintln!("║  FMA parallel:          {fma_par_us:>6.0} μs  {fma_par_gb:>5.1} GB/s ║");
        eprintln!("║  Integer parallel:      {int_par_us:>6.0} μs  {int_par_gb:>5.1} GB/s ║");
        eprintln!("╠══════════════════════════════════════════════════╣");
        eprintln!("║  Integer speedup (ST): {:.2}x                      ║", fma_st_us / int_st_us);
        eprintln!("║  Integer speedup (MT): {:.2}x                      ║", fma_par_us / int_par_us);
        eprintln!("╚══════════════════════════════════════════════════╝");
    }

    // ── Transposed layout kernel tests ──────────────────────────────

    #[test]
    fn test_transposed_scalar_matches_original() {
        // Verify that transposing weights and using the transposed scalar kernel
        // produces the same output as the original scalar kernel.
        let n = 16;
        let k = 128;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i as f32 / weight_bf16.len() as f32) - 0.5) * 0.2;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i as f32 / k as f32) - 0.5) * 2.0;
            activation[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&weight_bf16, n, k, group_size);

        // Original scalar output
        let mut orig_output = vec![0.0f32; n];
        matmul_int4_scalar(&q, &activation, &mut orig_output);

        // Transpose and compute with transposed scalar
        let (t_packed, t_scales) = transpose_int4(&q);
        let mut trans_output = vec![0.0f32; n];
        matmul_int4_transposed_scalar(
            &t_packed, &t_scales, &activation, &mut trans_output, k, n, group_size,
        );

        let mut max_diff: f32 = 0.0;
        for i in 0..n {
            max_diff = max_diff.max((orig_output[i] - trans_output[i]).abs());
        }
        eprintln!("Transposed scalar vs original scalar [{n}×{k}]: max_diff={max_diff:.8}");
        assert!(max_diff == 0.0, "Transposed scalar should be bit-identical: max_diff={max_diff}");
    }

    #[test]
    fn test_transposed_avx2_matches_scalar() {
        let n = 32;
        let k = 256;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 3 + 7) as f32 / weight_bf16.len() as f32 - 0.5) * 0.4;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 11 + 3) as f32 / k as f32 - 0.5) * 0.5;
            activation[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&weight_bf16, n, k, group_size);
        let (t_packed, t_scales) = transpose_int4(&q);

        let mut scalar_out = vec![0.0f32; n];
        matmul_int4_transposed_scalar(
            &t_packed, &t_scales, &activation, &mut scalar_out, k, n, group_size,
        );

        let mut avx2_out = vec![0.0f32; n];
        matmul_int4_transposed_avx2(
            &t_packed, &t_scales, &activation, &mut avx2_out, k, n, group_size,
        );

        let mut max_diff: f32 = 0.0;
        for i in 0..n {
            max_diff = max_diff.max((scalar_out[i] - avx2_out[i]).abs());
        }
        eprintln!("Transposed AVX2 vs scalar [{n}×{k}]: max_diff={max_diff:.8}");
        assert!(max_diff < 1e-3, "Transposed AVX2 vs scalar diverged: {max_diff}");
    }

    #[test]
    fn test_transposed_avx2_matches_original_avx2() {
        // End-to-end: transposed AVX2 should match original AVX2 (same weights, same activation)
        let n = 1408;
        let k = 2048;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 37 + 11) as f32 / weight_bf16.len() as f32 - 0.5) * 0.1;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 7 + 3) as f32 / k as f32 - 0.5) * 0.2;
            activation[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&weight_bf16, n, k, group_size);
        let (t_packed, t_scales) = transpose_int4(&q);

        let mut orig_out = vec![0.0f32; n];
        matmul_int4_avx2(&q, &activation, &mut orig_out);

        let mut trans_out = vec![0.0f32; n];
        matmul_int4_transposed_avx2(
            &t_packed, &t_scales, &activation, &mut trans_out, k, n, group_size,
        );

        let mut max_diff: f32 = 0.0;
        for i in 0..n {
            max_diff = max_diff.max((orig_out[i] - trans_out[i]).abs());
        }
        eprintln!("Transposed AVX2 vs original AVX2 [{n}×{k}]: max_diff={max_diff:.8}");
        assert!(max_diff < 0.01, "Transposed vs original diverged: {max_diff}");
    }

    #[test]
    fn test_transposed_integer_matches_transposed_fma() {
        let n = 32;
        let k = 256;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 3 + 7) as f32 / weight_bf16.len() as f32 - 0.5) * 0.4;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 11 + 3) as f32 / k as f32 - 0.5) * 0.5;
            activation[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&weight_bf16, n, k, group_size);
        let (t_packed, t_scales) = transpose_int4(&q);

        // FMA transposed
        let mut fma_out = vec![0.0f32; n];
        matmul_int4_transposed_avx2(
            &t_packed, &t_scales, &activation, &mut fma_out, k, n, group_size,
        );

        // Integer transposed
        let mut act_int16 = vec![0i16; k];
        let mut act_scales = vec![0.0f32; k / group_size];
        quantize_activation_int16(&activation, group_size, &mut act_int16, &mut act_scales);

        let mut int_out = vec![0.0f32; n];
        matmul_int4_transposed_integer(
            &t_packed, &t_scales, &act_int16, &act_scales, &mut int_out, k, n, group_size,
        );

        let mut max_diff: f32 = 0.0;
        let mut sum_sq_diff: f64 = 0.0;
        let mut sum_sq_fma: f64 = 0.0;
        for i in 0..n {
            let diff = (fma_out[i] - int_out[i]).abs();
            max_diff = max_diff.max(diff);
            sum_sq_diff += (diff as f64).powi(2);
            sum_sq_fma += (fma_out[i] as f64).powi(2);
        }
        let rmse = (sum_sq_diff / n as f64).sqrt();
        let rms_fma = (sum_sq_fma / n as f64).sqrt();
        let rel_err = if rms_fma > 0.0 { rmse / rms_fma } else { 0.0 };

        eprintln!(
            "Transposed integer vs FMA [{n}×{k}]: max_diff={max_diff:.6}, rel_err={rel_err:.6}"
        );
        assert!(rel_err < 0.01, "Transposed integer vs FMA relative error too large: {rel_err}");
    }

    #[test]
    fn test_transposed_parallel_correctness() {
        let n = 1408;
        let k = 2048;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 37 + 11) as f32 / weight_bf16.len() as f32 - 0.5) * 0.1;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 7 + 3) as f32 / k as f32 - 0.5) * 0.2;
            activation[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&weight_bf16, n, k, group_size);
        let (t_packed, t_scales) = transpose_int4(&q);

        // FMA: serial vs parallel
        let mut serial_out = vec![0.0f32; n];
        let mut parallel_out = vec![0.0f32; n];

        matmul_int4_transposed_avx2(
            &t_packed, &t_scales, &activation, &mut serial_out, k, n, group_size,
        );
        matmul_int4_transposed_parallel(
            &t_packed, &t_scales, &activation, &mut parallel_out, k, n, group_size,
        );

        let max_diff: f32 = (0..n)
            .map(|i| (serial_out[i] - parallel_out[i]).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Transposed FMA parallel vs serial: max_diff={max_diff:.8}");
        assert!(max_diff == 0.0, "Transposed parallel should be bit-identical");

        // Integer: serial vs parallel
        let mut act_int16 = vec![0i16; k];
        let mut act_scales = vec![0.0f32; k / group_size];
        quantize_activation_int16(&activation, group_size, &mut act_int16, &mut act_scales);

        let mut serial_int = vec![0.0f32; n];
        let mut parallel_int = vec![0.0f32; n];

        matmul_int4_transposed_integer(
            &t_packed, &t_scales, &act_int16, &act_scales, &mut serial_int, k, n, group_size,
        );
        matmul_int4_transposed_integer_parallel(
            &t_packed, &t_scales, &act_int16, &act_scales, &mut parallel_int, k, n, group_size,
        );

        let max_diff_int: f32 = (0..n)
            .map(|i| (serial_int[i] - parallel_int[i]).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Transposed integer parallel vs serial: max_diff={max_diff_int:.8}");
        assert!(max_diff_int == 0.0, "Transposed integer parallel should be bit-identical");
    }

    #[test]
    fn test_transposed_int8_matches_nontransposed() {
        use crate::weights::marlin::quantize_int8;

        // Test transposed INT8 kernel against non-transposed INT8 kernel
        let n = 32;
        let k = 256;
        let group_size = 128;

        // Generate BF16 weights [N, K]
        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 3 + 7) as f32 / weight_bf16.len() as f32 - 0.5) * 0.4;
            weight_bf16[i] = f32_to_bf16(val);
        }

        // Quantize to INT8 [N, K]
        let q8 = quantize_int8(&weight_bf16, n, k, group_size);

        // Non-transposed INT8 kernel (reference)
        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 11 + 3) as f32 / k as f32 - 0.5) * 0.5;
            activation[i] = f32_to_bf16(val);
        }

        let mut act_int16 = vec![0i16; k];
        let mut act_scales = vec![0.0f32; k / group_size];
        quantize_activation_int16(&activation, group_size, &mut act_int16, &mut act_scales);

        let mut nontransposed_out = vec![0.0f32; n];
        matmul_int8_integer(&q8, &act_int16, &act_scales, &mut nontransposed_out);

        // Transpose INT8 data: [N, K] → [K, N]
        let mut transposed_i8 = vec![0i8; k * n];
        for row in 0..n {
            for col in 0..k {
                transposed_i8[col * n + row] = q8.data[row * k + col];
            }
        }
        // Transpose scales: [N, K/gs] → [K/gs, N]
        let num_groups = k / group_size;
        let mut transposed_scales = vec![0u16; num_groups * n];
        for row in 0..n {
            for g in 0..num_groups {
                transposed_scales[g * n + row] = q8.scales[row * num_groups + g];
            }
        }

        // Pack i8 data into u32 container
        let u32_count = (k * n + 3) / 4;
        let mut data_u32 = vec![0u32; u32_count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                transposed_i8.as_ptr() as *const u8,
                data_u32.as_mut_ptr() as *mut u8,
                k * n,
            );
        }

        let mut transposed_out = vec![0.0f32; n];
        matmul_int8_transposed_integer(
            &data_u32, &transposed_scales, &act_int16, &act_scales,
            &mut transposed_out, k, n, group_size,
        );

        let mut max_diff: f32 = 0.0;
        for i in 0..n {
            let diff = (nontransposed_out[i] - transposed_out[i]).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!(
            "Transposed INT8 vs non-transposed [{n}×{k}]: max_diff={max_diff:.8}"
        );
        // Small difference expected from different FMA accumulation ordering
        assert!(max_diff < 0.001, "Transposed INT8 should match non-transposed closely: {max_diff}");
    }

    #[test]
    fn test_transposed_int8_parallel_correctness() {
        use crate::weights::marlin::quantize_int8;

        let n = 1408;
        let k = 2048;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 37 + 11) as f32 / weight_bf16.len() as f32 - 0.5) * 0.1;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let q8 = quantize_int8(&weight_bf16, n, k, group_size);

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 7 + 3) as f32 / k as f32 - 0.5) * 0.2;
            activation[i] = f32_to_bf16(val);
        }

        let mut act_int16 = vec![0i16; k];
        let mut act_scales = vec![0.0f32; k / group_size];
        quantize_activation_int16(&activation, group_size, &mut act_int16, &mut act_scales);

        // Transpose to [K, N]
        let mut transposed_i8 = vec![0i8; k * n];
        for row in 0..n {
            for col in 0..k {
                transposed_i8[col * n + row] = q8.data[row * k + col];
            }
        }
        let num_groups = k / group_size;
        let mut transposed_scales = vec![0u16; num_groups * n];
        for row in 0..n {
            for g in 0..num_groups {
                transposed_scales[g * n + row] = q8.scales[row * num_groups + g];
            }
        }
        let u32_count = (k * n + 3) / 4;
        let mut data_u32 = vec![0u32; u32_count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                transposed_i8.as_ptr() as *const u8,
                data_u32.as_mut_ptr() as *mut u8,
                k * n,
            );
        }

        let mut serial_out = vec![0.0f32; n];
        matmul_int8_transposed_integer(
            &data_u32, &transposed_scales, &act_int16, &act_scales,
            &mut serial_out, k, n, group_size,
        );

        let mut parallel_out = vec![0.0f32; n];
        matmul_int8_transposed_integer_parallel(
            &data_u32, &transposed_scales, &act_int16, &act_scales,
            &mut parallel_out, k, n, group_size,
        );

        let max_diff: f32 = (0..n)
            .map(|i| (serial_out[i] - parallel_out[i]).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Transposed INT8 parallel vs serial [{n}×{k}]: max_diff={max_diff:.8}");
        assert!(max_diff == 0.0, "Transposed INT8 parallel should be bit-identical");
    }

    #[test]
    fn test_marlin_kernel_matches_transposed() {
        use crate::weights::marlin::marlin_repack;

        // N=64 (minimum Marlin), K=256 (two groups — ensures is_grouped=true for Marlin scale_perm)
        // NOTE: group_size must be < K for grouped quantization (scale_perm_64),
        // otherwise Marlin uses scale_perm_single_32 which the kernel handles separately.
        let n = 64;
        let k = 256;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 37 + 11) as f32 / weight_bf16.len() as f32 - 0.5) * 0.2;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 7 + 3) as f32 / k as f32 - 0.5) * 0.3;
            activation[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&weight_bf16, n, k, group_size);

        // Transposed kernel (existing, known-correct)
        let (t_packed, t_scales) = transpose_int4(&q);
        let mut act_int16 = vec![0i16; k];
        let mut act_scales_vec = vec![0.0f32; k / group_size];
        quantize_activation_int16(&activation, group_size, &mut act_int16, &mut act_scales_vec);

        let mut transposed_out = vec![0.0f32; n];
        matmul_int4_transposed_integer(
            &t_packed, &t_scales, &act_int16, &act_scales_vec, &mut transposed_out, k, n, group_size,
        );

        // Scalar Marlin kernel (simpler, for debugging)
        let m = marlin_repack(&q);

        let mut scalar_out = vec![0.0f32; n];
        matmul_int4_marlin_scalar(
            &m.packed, &m.scales, &act_int16, &act_scales_vec,
            &mut scalar_out, k, n, group_size,
        );

        eprintln!("ref[0..5]:    {:?}", &transposed_out[0..5]);
        eprintln!("scalar[0..5]: {:?}", &scalar_out[0..5]);

        let mut max_diff_scalar: f32 = 0.0;
        for i in 0..n {
            let diff = (transposed_out[i] - scalar_out[i]).abs();
            max_diff_scalar = max_diff_scalar.max(diff);
        }
        eprintln!("Marlin SCALAR vs transposed [{n}×{k}]: max_diff={max_diff_scalar:.8}");
        assert!(max_diff_scalar < 0.001, "Scalar Marlin should match transposed: max_diff={max_diff_scalar}");
    }

    #[test]
    fn test_marlin_kernel_scalar_matches_avx2() {
        use crate::weights::marlin::marlin_repack;

        let n = 64;
        let k = 256;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 13 + 5) as f32 / weight_bf16.len() as f32 - 0.5) * 0.3;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 19 + 7) as f32 / k as f32 - 0.5) * 0.4;
            activation[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&weight_bf16, n, k, group_size);
        let m = marlin_repack(&q);

        let mut act_int16 = vec![0i16; k];
        let mut act_scales_vec = vec![0.0f32; k / group_size];
        quantize_activation_int16(&activation, group_size, &mut act_int16, &mut act_scales_vec);

        let tile_map = build_marlin_tile_map();
        let scale_map = build_marlin_scale_map();

        // Scalar Marlin
        let mut scalar_out = vec![0.0f32; n];
        matmul_int4_marlin_scalar(
            &m.packed, &m.scales, &act_int16, &act_scales_vec,
            &mut scalar_out, k, n, group_size,
        );

        // AVX2 Marlin
        let mut avx2_out = vec![0.0f32; n];
        matmul_int4_marlin(
            &m.packed, &m.scales, &act_int16, &act_scales_vec,
            &mut avx2_out, k, n, group_size, &tile_map, &scale_map,
        );

        let mut max_diff: f32 = 0.0;
        for i in 0..n {
            let diff = (scalar_out[i] - avx2_out[i]).abs();
            max_diff = max_diff.max(diff);
        }

        eprintln!("Marlin scalar vs AVX2 [{n}×{k}]: max_diff={max_diff:.8}");
        // Precision difference from FMA ordering: scalar does (x * ws) * as,
        // AVX2 does x * (ws * as) + acc via FMA. ~0.0003 is expected.
        assert!(max_diff < 0.001, "Marlin AVX2 should match scalar closely: max_diff={max_diff}");
    }

    #[test]
    fn test_marlin_kernel_large() {
        use crate::weights::marlin::marlin_repack;

        // Realistic size: V2-Lite gate_proj dimensions
        let n = 1408;
        let k = 2048;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 37 + 11) as f32 / weight_bf16.len() as f32 - 0.5) * 0.1;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 7 + 3) as f32 / k as f32 - 0.5) * 0.2;
            activation[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&weight_bf16, n, k, group_size);

        // Transposed kernel (reference)
        let (t_packed, t_scales) = transpose_int4(&q);
        let mut act_int16 = vec![0i16; k];
        let mut act_scales_vec = vec![0.0f32; k / group_size];
        quantize_activation_int16(&activation, group_size, &mut act_int16, &mut act_scales_vec);

        let mut ref_out = vec![0.0f32; n];
        matmul_int4_transposed_integer(
            &t_packed, &t_scales, &act_int16, &act_scales_vec, &mut ref_out, k, n, group_size,
        );

        // Marlin kernel
        let m = marlin_repack(&q);
        let tile_map = build_marlin_tile_map();
        let scale_map = build_marlin_scale_map();

        let mut marlin_out = vec![0.0f32; n];
        matmul_int4_marlin(
            &m.packed, &m.scales, &act_int16, &act_scales_vec,
            &mut marlin_out, k, n, group_size, &tile_map, &scale_map,
        );

        let mut max_diff: f32 = 0.0;
        for i in 0..n {
            let diff = (ref_out[i] - marlin_out[i]).abs();
            max_diff = max_diff.max(diff);
        }

        eprintln!("Marlin kernel large [{n}×{k}]: max_diff={max_diff:.8}");
        assert!(max_diff == 0.0, "Marlin kernel large should match transposed exactly: max_diff={max_diff}");
    }

    #[test]
    fn test_marlin_kernel_parallel() {
        use crate::weights::marlin::marlin_repack;

        let n = 1408;
        let k = 2048;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i * 37 + 11) as f32 / weight_bf16.len() as f32 - 0.5) * 0.1;
            weight_bf16[i] = f32_to_bf16(val);
        }

        let mut activation = vec![0u16; k];
        for i in 0..k {
            let val = ((i * 7 + 3) as f32 / k as f32 - 0.5) * 0.2;
            activation[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&weight_bf16, n, k, group_size);
        let m = marlin_repack(&q);

        let mut act_int16 = vec![0i16; k];
        let mut act_scales_vec = vec![0.0f32; k / group_size];
        quantize_activation_int16(&activation, group_size, &mut act_int16, &mut act_scales_vec);

        let tile_map = build_marlin_tile_map();
        let scale_map = build_marlin_scale_map();

        let mut serial_out = vec![0.0f32; n];
        matmul_int4_marlin(
            &m.packed, &m.scales, &act_int16, &act_scales_vec,
            &mut serial_out, k, n, group_size, &tile_map, &scale_map,
        );

        let mut parallel_out = vec![0.0f32; n];
        matmul_int4_marlin_parallel(
            &m.packed, &m.scales, &act_int16, &act_scales_vec,
            &mut parallel_out, k, n, group_size, &tile_map, &scale_map,
        );

        let max_diff: f32 = (0..n)
            .map(|i| (serial_out[i] - parallel_out[i]).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Marlin parallel vs serial [{n}×{k}]: max_diff={max_diff:.8}");
        assert!(max_diff == 0.0, "Marlin parallel should be bit-identical to serial");
    }
}
