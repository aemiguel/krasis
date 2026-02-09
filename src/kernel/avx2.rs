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
}
