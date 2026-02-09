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

use crate::weights::marlin::{bf16_to_f32, QuantizedInt4};

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
}
