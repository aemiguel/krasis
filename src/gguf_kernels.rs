//! Native GGUF matmul kernels for CPU decode.
//!
//! Primary path: INT16-quantized activations × GGUF weights via AVX2 SIMD.
//! - Q4_K: AVX2 INT16 × unsigned INT4 with asymmetric min correction
//! - Q8_0: AVX2 INT16 × signed INT8
//! - Q4_0: AVX2 INT16 × unsigned INT4 with fixed offset-8 correction
//! Fallback: scalar f32 for Q5_0, Q6_K (and all formats on non-AVX2).
//!
//! For M=1 decode: output[N] = input[K] @ weights[N, K] (row-major in GGUF).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::gguf::GgmlType;
use crate::weights::GgufExpertWeights;

// ── Block layout constants ──────────────────────────────────────────

const Q4_K_BLOCK_SIZE: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;

const Q5_0_BLOCK_SIZE: usize = 32;
const Q5_0_BLOCK_BYTES: usize = 22;

const Q4_0_BLOCK_SIZE: usize = 32;
const Q4_0_BLOCK_BYTES: usize = 18;

const Q6_K_BLOCK_SIZE: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 210;

const Q8_0_BLOCK_SIZE: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;

/// Activation quantization group size (matches Q4_K sub-block and Q8_0 block).
pub const GGUF_GROUP_SIZE: usize = 32;

// ── GgufScratch ─────────────────────────────────────────────────────

/// Scratch buffers for GGUF expert computation (reused across calls).
pub struct GgufScratch {
    // INT16 quantized input activation (for gate/up)
    pub input_int16: Vec<i16>,
    pub input_scales: Vec<f32>,
    pub input_sums: Vec<i32>,
    // f32 input (fallback for non-INT formats)
    pub input_f32: Vec<f32>,
    // Intermediate outputs
    pub gate_out: Vec<f32>,
    pub up_out: Vec<f32>,
    pub hidden_f32: Vec<f32>,
    // INT16 quantized hidden state (for down projection)
    pub hidden_int16: Vec<i16>,
    pub hidden_scales: Vec<f32>,
    pub hidden_sums: Vec<i32>,
    // Final expert output
    pub expert_out: Vec<f32>,
}

impl GgufScratch {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        let h_groups = (hidden_size + GGUF_GROUP_SIZE - 1) / GGUF_GROUP_SIZE;
        let n_groups = (intermediate_size + GGUF_GROUP_SIZE - 1) / GGUF_GROUP_SIZE;
        GgufScratch {
            input_int16: vec![0i16; hidden_size],
            input_scales: vec![0.0f32; h_groups],
            input_sums: vec![0i32; h_groups],
            input_f32: vec![0.0f32; hidden_size],
            gate_out: vec![0.0f32; intermediate_size],
            up_out: vec![0.0f32; intermediate_size],
            hidden_f32: vec![0.0f32; intermediate_size],
            hidden_int16: vec![0i16; intermediate_size],
            hidden_scales: vec![0.0f32; n_groups],
            hidden_sums: vec![0i32; n_groups],
            expert_out: vec![0.0f32; hidden_size],
        }
    }

    /// Resize all buffers if needed (for shared experts with different sizes).
    pub fn ensure_size(&mut self, hidden_size: usize, intermediate_size: usize) {
        let h_groups = (hidden_size + GGUF_GROUP_SIZE - 1) / GGUF_GROUP_SIZE;
        let n_groups = (intermediate_size + GGUF_GROUP_SIZE - 1) / GGUF_GROUP_SIZE;
        self.input_int16.resize(hidden_size, 0);
        self.input_scales.resize(h_groups, 0.0);
        self.input_sums.resize(h_groups, 0);
        self.input_f32.resize(hidden_size, 0.0);
        self.gate_out.resize(intermediate_size, 0.0);
        self.up_out.resize(intermediate_size, 0.0);
        self.hidden_f32.resize(intermediate_size, 0.0);
        self.hidden_int16.resize(intermediate_size, 0);
        self.hidden_scales.resize(n_groups, 0.0);
        self.hidden_sums.resize(n_groups, 0);
        self.expert_out.resize(hidden_size, 0.0);
    }
}

// ── Activation quantization ─────────────────────────────────────────

/// Whether a GGUF type has an AVX2 INT16 kernel.
fn supports_int_path(dtype: GgmlType) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return matches!(dtype, GgmlType::Q4_K | GgmlType::Q8_0 | GgmlType::Q4_0);
        }
    }
    false
}

/// Quantize BF16 activation to INT16 with per-32-element group scales and sums.
pub fn quantize_bf16_to_int16(
    input_bf16: &[u16],
    output_int16: &mut [i16],
    output_scales: &mut [f32],
    output_sums: &mut [i32],
) {
    let k = input_bf16.len();
    let gs = GGUF_GROUP_SIZE;
    let num_groups = k / gs;

    for g in 0..num_groups {
        let start = g * gs;
        let mut max_abs = 0.0f32;
        for i in 0..gs {
            max_abs = max_abs.max(bf16_to_f32(input_bf16[start + i]).abs());
        }
        let scale = if max_abs > 0.0 { max_abs / 32767.0 } else { 1.0 };
        let inv_scale = if max_abs > 0.0 { 32767.0 / max_abs } else { 0.0 };
        output_scales[g] = scale;

        let mut sum = 0i32;
        for i in 0..gs {
            let val = bf16_to_f32(input_bf16[start + i]);
            let q = (val * inv_scale).round() as i32;
            let qc = q.clamp(-32768, 32767) as i16;
            output_int16[start + i] = qc;
            sum += qc as i32;
        }
        output_sums[g] = sum;
    }
}

/// Quantize f32 hidden state to INT16 with per-32-element group scales and sums.
pub fn quantize_f32_to_int16(
    input_f32: &[f32],
    output_int16: &mut [i16],
    output_scales: &mut [f32],
    output_sums: &mut [i32],
) {
    let k = input_f32.len();
    let gs = GGUF_GROUP_SIZE;
    let num_groups = k / gs;

    for g in 0..num_groups {
        let start = g * gs;
        let mut max_abs = 0.0f32;
        for i in 0..gs {
            max_abs = max_abs.max(input_f32[start + i].abs());
        }
        let scale = if max_abs > 0.0 { max_abs / 32767.0 } else { 1.0 };
        let inv_scale = if max_abs > 0.0 { 32767.0 / max_abs } else { 0.0 };
        output_scales[g] = scale;

        let mut sum = 0i32;
        for i in 0..gs {
            let q = (input_f32[start + i] * inv_scale).round() as i32;
            let qc = q.clamp(-32768, 32767) as i16;
            output_int16[start + i] = qc;
            sum += qc as i32;
        }
        output_sums[g] = sum;
    }
}

// ── Public dispatch ─────────────────────────────────────────────────

/// INT16-activation matvec dispatch (uses AVX2 for Q4_K, Q8_0, Q4_0).
fn gguf_matvec_int(
    dtype: GgmlType,
    weights: &[u8],
    act_int16: &[i16],
    act_scales: &[f32],
    act_sums: &[i32],
    n: usize,
    k: usize,
    output: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            match dtype {
                GgmlType::Q4_K => {
                    unsafe {
                        matvec_q4_k_avx2(
                            weights.as_ptr(), act_int16.as_ptr(),
                            act_scales.as_ptr(), act_sums.as_ptr(),
                            n, k, output.as_mut_ptr(),
                        );
                    }
                    return;
                }
                GgmlType::Q8_0 => {
                    unsafe {
                        matvec_q8_0_avx2(
                            weights.as_ptr(), act_int16.as_ptr(),
                            act_scales.as_ptr(),
                            n, k, output.as_mut_ptr(),
                        );
                    }
                    return;
                }
                GgmlType::Q4_0 => {
                    unsafe {
                        matvec_q4_0_avx2(
                            weights.as_ptr(), act_int16.as_ptr(),
                            act_scales.as_ptr(), act_sums.as_ptr(),
                            n, k, output.as_mut_ptr(),
                        );
                    }
                    return;
                }
                _ => {}
            }
        }
    }

    // Fallback: reconstruct f32 from INT16 and use scalar path
    let mut input_f32 = vec![0.0f32; k];
    let gs = GGUF_GROUP_SIZE;
    for g in 0..k / gs {
        let s = act_scales[g];
        for i in 0..gs {
            input_f32[g * gs + i] = act_int16[g * gs + i] as f32 * s;
        }
    }
    gguf_matvec_f32(dtype, weights, &input_f32, n, k, output);
}

/// f32-activation matvec dispatch (scalar fallback for all formats).
fn gguf_matvec_f32(
    dtype: GgmlType,
    weights: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
    output: &mut [f32],
) {
    match dtype {
        GgmlType::Q4_K => matvec_q4_k_scalar(weights, input, n, k, output),
        GgmlType::Q5_0 => matvec_q5_0_scalar(weights, input, n, k, output),
        GgmlType::Q4_0 => matvec_q4_0_scalar(weights, input, n, k, output),
        GgmlType::Q6_K => matvec_q6_k_scalar(weights, input, n, k, output),
        GgmlType::Q8_0 => matvec_q8_0_scalar(weights, input, n, k, output),
        _ => panic!("GGUF matvec not implemented for {:?}", dtype),
    }
}

// ── AVX2 Q4_K kernel (INT16 × unsigned INT4, asymmetric) ────────────
//
// Q4_K block: 256 elements, 144 bytes
//   [0..2]    fp16 d (super-block scale)
//   [2..4]    fp16 dmin (super-block minimum)
//   [4..16]   12 bytes packed 6-bit scales/mins (8 sub-blocks)
//   [16..144] 128 bytes 4-bit quants (4 groups of 32 bytes = 256 nibbles)
//
// Math per sub-block (32 elements, unsigned q4 in [0,15]):
//   dot = d * sc * act_scale * Σ(q4 * act_int16) - dmin * mn * act_scale * Σ(act_int16)
//   First term: AVX2 madd_epi16, second term: precomputed act_sums

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_q4_k_avx2(
    weights: *const u8,
    act_int16: *const i16,
    act_scales: *const f32,
    act_sums: *const i32,
    n: usize,
    k: usize,
    output: *mut f32,
) {
    let blocks_per_row = k / Q4_K_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
    let mask_0f = _mm_set1_epi8(0x0F);

    for row in 0..n {
        let row_data = weights.add(row * row_bytes);
        let mut float_acc = _mm256_setzero_ps();
        let mut correction = 0.0f32;

        for b in 0..blocks_per_row {
            let block = row_data.add(b * Q4_K_BLOCK_BYTES);
            let d = f16_to_f32_ptr(block);
            let dmin = f16_to_f32_ptr(block.add(2));
            let scales_packed = block.add(4);
            let quants = block.add(16);
            let act_base = b * Q4_K_BLOCK_SIZE;
            let group_base = b * 8; // 8 groups of 32 per Q4_K block

            for j in 0..4usize {
                let (sc_lo, mn_lo) = get_scale_min_k4_ptr(2 * j, scales_packed);
                let (sc_hi, mn_hi) = get_scale_min_k4_ptr(2 * j + 1, scales_packed);

                let qs = quants.add(j * 32);
                let lo_group = group_base + j * 2;
                let hi_group = group_base + j * 2 + 1;
                let a_scale_lo = *act_scales.add(lo_group);
                let a_scale_hi = *act_scales.add(hi_group);

                // Load 32 quant bytes in two halves
                let raw0 = _mm_loadu_si128(qs as *const __m128i);
                let raw1 = _mm_loadu_si128(qs.add(16) as *const __m128i);

                // Extract nibbles
                let lo_nib_0 = _mm_and_si128(raw0, mask_0f);
                let lo_nib_1 = _mm_and_si128(raw1, mask_0f);
                let hi_nib_0 = _mm_and_si128(_mm_srli_epi16(raw0, 4), mask_0f);
                let hi_nib_1 = _mm_and_si128(_mm_srli_epi16(raw1, 4), mask_0f);

                // --- Low nibble sub-block: 32 elements [j*64 .. j*64+31] ---
                let w16_lo0 = _mm256_cvtepu8_epi16(lo_nib_0);
                let a16_lo0 = _mm256_loadu_si256(
                    act_int16.add(act_base + j * 64) as *const __m256i
                );
                let mut int_acc_lo = _mm256_madd_epi16(w16_lo0, a16_lo0);

                let w16_lo1 = _mm256_cvtepu8_epi16(lo_nib_1);
                let a16_lo1 = _mm256_loadu_si256(
                    act_int16.add(act_base + j * 64 + 16) as *const __m256i
                );
                int_acc_lo = _mm256_add_epi32(
                    int_acc_lo, _mm256_madd_epi16(w16_lo1, a16_lo1)
                );

                let combined_lo = d * sc_lo as f32 * a_scale_lo;
                float_acc = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(int_acc_lo),
                    _mm256_set1_ps(combined_lo),
                    float_acc,
                );
                correction += dmin * mn_lo as f32 * a_scale_lo
                    * *act_sums.add(lo_group) as f32;

                // --- High nibble sub-block: 32 elements [j*64+32 .. j*64+63] ---
                let w16_hi0 = _mm256_cvtepu8_epi16(hi_nib_0);
                let a16_hi0 = _mm256_loadu_si256(
                    act_int16.add(act_base + j * 64 + 32) as *const __m256i
                );
                let mut int_acc_hi = _mm256_madd_epi16(w16_hi0, a16_hi0);

                let w16_hi1 = _mm256_cvtepu8_epi16(hi_nib_1);
                let a16_hi1 = _mm256_loadu_si256(
                    act_int16.add(act_base + j * 64 + 48) as *const __m256i
                );
                int_acc_hi = _mm256_add_epi32(
                    int_acc_hi, _mm256_madd_epi16(w16_hi1, a16_hi1)
                );

                let combined_hi = d * sc_hi as f32 * a_scale_hi;
                float_acc = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(int_acc_hi),
                    _mm256_set1_ps(combined_hi),
                    float_acc,
                );
                correction += dmin * mn_hi as f32 * a_scale_hi
                    * *act_sums.add(hi_group) as f32;
            }
        }

        *output.add(row) = hsum_avx(float_acc) - correction;
    }
}

// ── AVX2 Q8_0 kernel (INT16 × signed INT8, symmetric) ──────────────
//
// Q8_0 block: 32 elements, 34 bytes = fp16 d + 32 int8 values
// Math: dot = d * act_scale * Σ(q8 * act_int16)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_q8_0_avx2(
    weights: *const u8,
    act_int16: *const i16,
    act_scales: *const f32,
    n: usize,
    k: usize,
    output: *mut f32,
) {
    let blocks_per_row = k / Q8_0_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;

    for row in 0..n {
        let row_data = weights.add(row * row_bytes);
        let mut float_acc = _mm256_setzero_ps();

        for b in 0..blocks_per_row {
            let block = row_data.add(b * Q8_0_BLOCK_BYTES);
            let d = f16_to_f32_ptr(block);
            let qs = block.add(2);
            let a_scale = *act_scales.add(b);
            let combined = d * a_scale;

            // First 16 int8 weights → sign-extend to int16
            let raw0 = _mm_loadu_si128(qs as *const __m128i);
            let w16_0 = _mm256_cvtepi8_epi16(raw0);
            let a16_0 = _mm256_loadu_si256(
                act_int16.add(b * 32) as *const __m256i
            );
            let mut int_acc = _mm256_madd_epi16(w16_0, a16_0);

            // Second 16 int8 weights
            let raw1 = _mm_loadu_si128(qs.add(16) as *const __m128i);
            let w16_1 = _mm256_cvtepi8_epi16(raw1);
            let a16_1 = _mm256_loadu_si256(
                act_int16.add(b * 32 + 16) as *const __m256i
            );
            int_acc = _mm256_add_epi32(int_acc, _mm256_madd_epi16(w16_1, a16_1));

            float_acc = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(int_acc),
                _mm256_set1_ps(combined),
                float_acc,
            );
        }

        *output.add(row) = hsum_avx(float_acc);
    }
}

// ── AVX2 Q4_0 kernel (INT16 × unsigned INT4, offset-8) ─────────────
//
// Q4_0 block: 32 elements, 18 bytes = fp16 d + 16 bytes quants
// Elements 0..15 = low nibbles, 16..31 = high nibbles
// Math: dot = d * act_scale * Σ(q4 * act_int16) - d * 8 * act_scale * Σ(act_int16)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_q4_0_avx2(
    weights: *const u8,
    act_int16: *const i16,
    act_scales: *const f32,
    act_sums: *const i32,
    n: usize,
    k: usize,
    output: *mut f32,
) {
    let blocks_per_row = k / Q4_0_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q4_0_BLOCK_BYTES;
    let mask_0f = _mm_set1_epi8(0x0F);

    for row in 0..n {
        let row_data = weights.add(row * row_bytes);
        let mut float_acc = _mm256_setzero_ps();
        let mut correction = 0.0f32;

        for b in 0..blocks_per_row {
            let block = row_data.add(b * Q4_0_BLOCK_BYTES);
            let d = f16_to_f32_ptr(block);
            let qs = block.add(2); // 16 bytes of quant data
            let a_scale = *act_scales.add(b);
            let a_sum = *act_sums.add(b);

            // Load 16 quant bytes
            let raw = _mm_loadu_si128(qs as *const __m128i);

            // Low nibbles → elements 0..15
            let lo = _mm_and_si128(raw, mask_0f);
            let w16_lo = _mm256_cvtepu8_epi16(lo);
            let a16_lo = _mm256_loadu_si256(
                act_int16.add(b * 32) as *const __m256i
            );
            let mut int_acc = _mm256_madd_epi16(w16_lo, a16_lo);

            // High nibbles → elements 16..31
            let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);
            let w16_hi = _mm256_cvtepu8_epi16(hi);
            let a16_hi = _mm256_loadu_si256(
                act_int16.add(b * 32 + 16) as *const __m256i
            );
            int_acc = _mm256_add_epi32(int_acc, _mm256_madd_epi16(w16_hi, a16_hi));

            let combined = d * a_scale;
            float_acc = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(int_acc),
                _mm256_set1_ps(combined),
                float_acc,
            );
            correction += d * 8.0 * a_scale * a_sum as f32;
        }

        *output.add(row) = hsum_avx(float_acc) - correction;
    }
}

// ── Scalar fallback kernels ─────────────────────────────────────────

fn matvec_q4_k_scalar(weights: &[u8], input: &[f32], n: usize, k: usize, output: &mut [f32]) {
    let blocks_per_row = k / Q4_K_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
    for row in 0..n {
        let row_data = &weights[row * row_bytes..(row + 1) * row_bytes];
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &row_data[b * Q4_K_BLOCK_BYTES..];
            sum += q4_k_block_dot_scalar(block, &input[b * Q4_K_BLOCK_SIZE..]);
        }
        output[row] = sum;
    }
}

#[inline]
fn q4_k_block_dot_scalar(block: &[u8], input: &[f32]) -> f32 {
    let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let dmin = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
    let scales_packed = &block[4..16];
    let quants = &block[16..144];
    let mut sum = 0.0f32;
    for j in 0..4 {
        let (sc_lo, mn_lo) = get_scale_min_k4(2 * j, scales_packed);
        let (sc_hi, mn_hi) = get_scale_min_k4(2 * j + 1, scales_packed);
        let d_lo = d * sc_lo as f32;
        let min_lo = dmin * mn_lo as f32;
        let d_hi = d * sc_hi as f32;
        let min_hi = dmin * mn_hi as f32;
        let qs = &quants[j * 32..];
        for l in 0..32 {
            sum += (d_lo * (qs[l] & 0xF) as f32 - min_lo) * input[j * 64 + l];
        }
        for l in 0..32 {
            sum += (d_hi * ((qs[l] >> 4) & 0xF) as f32 - min_hi) * input[j * 64 + 32 + l];
        }
    }
    sum
}

fn matvec_q8_0_scalar(weights: &[u8], input: &[f32], n: usize, k: usize, output: &mut [f32]) {
    let blocks_per_row = k / Q8_0_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;
    for row in 0..n {
        let row_data = &weights[row * row_bytes..(row + 1) * row_bytes];
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &row_data[b * Q8_0_BLOCK_BYTES..];
            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let qs = &block[2..34];
            for j in 0..32 {
                sum += d * (qs[j] as i8) as f32 * input[b * 32 + j];
            }
        }
        output[row] = sum;
    }
}

fn matvec_q4_0_scalar(weights: &[u8], input: &[f32], n: usize, k: usize, output: &mut [f32]) {
    let blocks_per_row = k / Q4_0_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q4_0_BLOCK_BYTES;
    for row in 0..n {
        let row_data = &weights[row * row_bytes..(row + 1) * row_bytes];
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &row_data[b * Q4_0_BLOCK_BYTES..];
            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let qs = &block[2..18];
            for j in 0..32 {
                let nibble = if j < 16 { qs[j] & 0x0F } else { (qs[j - 16] >> 4) & 0x0F };
                let q = nibble as i32 - 8;
                sum += d * q as f32 * input[b * 32 + j];
            }
        }
        output[row] = sum;
    }
}

fn matvec_q5_0_scalar(weights: &[u8], input: &[f32], n: usize, k: usize, output: &mut [f32]) {
    let blocks_per_row = k / Q5_0_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q5_0_BLOCK_BYTES;
    for row in 0..n {
        let row_data = &weights[row * row_bytes..(row + 1) * row_bytes];
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &row_data[b * Q5_0_BLOCK_BYTES..];
            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
            let qs = &block[6..22];
            for j in 0..32 {
                let q4 = if j < 16 { qs[j] & 0x0F } else { (qs[j - 16] >> 4) & 0x0F };
                let q5bit = ((qh >> j) & 1) as u8;
                let q = (q4 | (q5bit << 4)) as i32 - 16;
                sum += d * q as f32 * input[b * 32 + j];
            }
        }
        output[row] = sum;
    }
}

fn matvec_q6_k_scalar(weights: &[u8], input: &[f32], n: usize, k: usize, output: &mut [f32]) {
    let blocks_per_row = k / Q6_K_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q6_K_BLOCK_BYTES;
    for row in 0..n {
        let row_data = &weights[row * row_bytes..(row + 1) * row_bytes];
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &row_data[b * Q6_K_BLOCK_BYTES..];
            let ql = &block[0..128];
            let qh = &block[128..192];
            let scales = &block[192..208];
            let d = half::f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
            let inp = &input[b * Q6_K_BLOCK_SIZE..];
            for n_half in 0..2usize {
                let ql_off = n_half * 64;
                let qh_off = n_half * 32;
                let sc_off = n_half * 8;
                let inp_off = n_half * 128;
                for l in 0..32usize {
                    let is = l / 16;
                    let q0 = ((ql[ql_off + l] & 0xF) as i32)
                        | ((((qh[qh_off + l] >> 0) & 3) as i32) << 4);
                    let q1 = ((ql[ql_off + 32 + l] & 0xF) as i32)
                        | ((((qh[qh_off + l] >> 2) & 3) as i32) << 4);
                    let q2 = (((ql[ql_off + l] >> 4) & 0xF) as i32)
                        | ((((qh[qh_off + l] >> 4) & 3) as i32) << 4);
                    let q3 = (((ql[ql_off + 32 + l] >> 4) & 0xF) as i32)
                        | ((((qh[qh_off + l] >> 6) & 3) as i32) << 4);
                    let s0 = d * (scales[sc_off + is + 0] as i8 as f32);
                    let s1 = d * (scales[sc_off + is + 2] as i8 as f32);
                    let s2 = d * (scales[sc_off + is + 4] as i8 as f32);
                    let s3 = d * (scales[sc_off + is + 6] as i8 as f32);
                    sum += s0 * (q0 - 32) as f32 * inp[inp_off + l];
                    sum += s1 * (q1 - 32) as f32 * inp[inp_off + 32 + l];
                    sum += s2 * (q2 - 32) as f32 * inp[inp_off + 64 + l];
                    sum += s3 * (q3 - 32) as f32 * inp[inp_off + 96 + l];
                }
            }
        }
        output[row] = sum;
    }
}

// ── Scale/min extraction helpers ────────────────────────────────────

#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, mn)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn get_scale_min_k4_ptr(j: usize, scales: *const u8) -> (u8, u8) {
    if j < 4 {
        (*scales.add(j) & 63, *scales.add(j + 4) & 63)
    } else {
        let sc = (*scales.add(j + 4) & 0xF) | ((*scales.add(j - 4) >> 6) << 4);
        let mn = (*scales.add(j + 4) >> 4) | ((*scales.add(j) >> 6) << 4);
        (sc, mn)
    }
}

// ── AVX2 utility ────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let hi32 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn f16_to_f32_ptr(ptr: *const u8) -> f32 {
    let bits = u16::from_le_bytes([*ptr, *ptr.add(1)]);
    half::f16::from_bits(bits).to_f32()
}

// ── Expert forward ──────────────────────────────────────────────────

/// Full expert forward with native GGUF weights using optimal precision.
///
/// Pipeline: quantize input → gate/up matvec → SiLU → quantize hidden → down matvec.
/// Q4_K/Q8_0/Q4_0: AVX2 INT16×INTn SIMD. Others: scalar f32 fallback.
pub fn expert_forward_gguf(
    expert: &GgufExpertWeights,
    activation_bf16: &[u16],
    scratch: &mut GgufScratch,
) {
    let k = expert.hidden_size;
    let n = expert.intermediate_size;
    scratch.ensure_size(k, n);

    let use_int_gate_up = supports_int_path(expert.gate_up_type) && k % GGUF_GROUP_SIZE == 0;
    let use_int_down = supports_int_path(expert.down_type) && n % GGUF_GROUP_SIZE == 0;

    // --- Gate and Up projections ---
    if use_int_gate_up {
        quantize_bf16_to_int16(
            activation_bf16,
            &mut scratch.input_int16, &mut scratch.input_scales, &mut scratch.input_sums,
        );
        gguf_matvec_int(
            expert.gate_up_type, &expert.gate_data,
            &scratch.input_int16, &scratch.input_scales, &scratch.input_sums,
            n, k, &mut scratch.gate_out,
        );
        gguf_matvec_int(
            expert.gate_up_type, &expert.up_data,
            &scratch.input_int16, &scratch.input_scales, &scratch.input_sums,
            n, k, &mut scratch.up_out,
        );
    } else {
        for i in 0..k {
            scratch.input_f32[i] = bf16_to_f32(activation_bf16[i]);
        }
        gguf_matvec_f32(
            expert.gate_up_type, &expert.gate_data,
            &scratch.input_f32[..k], n, k, &mut scratch.gate_out,
        );
        gguf_matvec_f32(
            expert.gate_up_type, &expert.up_data,
            &scratch.input_f32[..k], n, k, &mut scratch.up_out,
        );
    }

    // --- SiLU(gate) * up ---
    for i in 0..n {
        let g = scratch.gate_out[i];
        let silu = g / (1.0 + (-g).exp());
        scratch.hidden_f32[i] = silu * scratch.up_out[i];
    }

    // --- Down projection ---
    if use_int_down {
        quantize_f32_to_int16(
            &scratch.hidden_f32[..n],
            &mut scratch.hidden_int16, &mut scratch.hidden_scales, &mut scratch.hidden_sums,
        );
        gguf_matvec_int(
            expert.down_type, &expert.down_data,
            &scratch.hidden_int16[..n], &scratch.hidden_scales, &scratch.hidden_sums,
            k, n, &mut scratch.expert_out,
        );
    } else {
        gguf_matvec_f32(
            expert.down_type, &expert.down_data,
            &scratch.hidden_f32[..n], k, n, &mut scratch.expert_out,
        );
    }
}

#[inline]
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_scalar_roundtrip() {
        let mut block = vec![0u8; Q4_0_BLOCK_BYTES];
        let d_bits = half::f16::from_f32(1.0).to_bits();
        block[0..2].copy_from_slice(&d_bits.to_le_bytes());
        for i in 0..16 { block[2 + i] = 0x88; } // all nibbles = 8 → q=0
        let input = vec![1.0f32; 32];
        let mut output = vec![0.0f32; 1];
        matvec_q4_0_scalar(&block, &input, 1, 32, &mut output);
        assert!((output[0] - 0.0).abs() < 1e-6, "expected 0, got {}", output[0]);
    }

    #[test]
    fn test_q8_0_scalar_simple() {
        let mut block = vec![0u8; Q8_0_BLOCK_BYTES];
        let d_bits = half::f16::from_f32(0.1).to_bits();
        block[0..2].copy_from_slice(&d_bits.to_le_bytes());
        for i in 0..32 { block[2 + i] = 10u8; } // q8 = 10
        let input = vec![1.0f32; 32];
        let mut output = vec![0.0f32; 1];
        matvec_q8_0_scalar(&block, &input, 1, 32, &mut output);
        // Expected: 32 * 0.1 * 10 = 32.0
        assert!((output[0] - 32.0).abs() < 0.5, "expected ~32, got {}", output[0]);
    }

    #[test]
    fn test_quantize_bf16_roundtrip() {
        let k = 64;
        let mut bf16 = vec![0u16; k];
        // Store some BF16 values: 1.0 = 0x3F80
        for i in 0..k {
            let val = (i as f32 - 32.0) * 0.1;
            bf16[i] = (val.to_bits() >> 16) as u16;
        }

        let mut int16 = vec![0i16; k];
        let mut scales = vec![0.0f32; k / GGUF_GROUP_SIZE];
        let mut sums = vec![0i32; k / GGUF_GROUP_SIZE];

        quantize_bf16_to_int16(&bf16, &mut int16, &mut scales, &mut sums);

        // Verify roundtrip: int16 * scale ≈ original f32
        for g in 0..k / GGUF_GROUP_SIZE {
            for i in 0..GGUF_GROUP_SIZE {
                let idx = g * GGUF_GROUP_SIZE + i;
                let original = bf16_to_f32(bf16[idx]);
                let reconstructed = int16[idx] as f32 * scales[g];
                let err = (original - reconstructed).abs();
                assert!(err < 0.01, "g={g} i={i}: orig={original} recon={reconstructed} err={err}");
            }
        }

        // Verify sums
        for g in 0..k / GGUF_GROUP_SIZE {
            let expected: i32 = (0..GGUF_GROUP_SIZE)
                .map(|i| int16[g * GGUF_GROUP_SIZE + i] as i32)
                .sum();
            assert_eq!(sums[g], expected, "sum mismatch at group {g}");
        }
    }
}
