//! Marlin INT4 format conversion and handling.
//!
//! Converts HF model weights (BF16) to INT4 packed format.
//! Two stages:
//!   1. quantize_int4() — symmetric INT4 quantization + packing (CPU-friendly layout)
//!   2. marlin_repack() — permute for GPU warp coalescing (added later)
//!
//! The CPU reads the packed INT4 + BF16 scales directly.
//! The GPU applies marlin_repack on-the-fly per layer during prefill.

/// Default quantization group size.
pub const DEFAULT_GROUP_SIZE: usize = 128;

/// Number of INT4 values packed per u32.
const PACK_FACTOR: usize = 8;

/// Convert a raw BF16 u16 to f32.
#[inline]
pub fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// Convert f32 to raw BF16 u16 (round to nearest even).
#[inline]
pub fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    // Round to nearest even: add 0x7FFF + bit[16] for tie-breaking
    let round = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (round >> 16) as u16
}

/// Symmetric INT4 quantization result for a single weight matrix.
pub struct QuantizedInt4 {
    /// Packed INT4 weights: 8 values per u32, row-major.
    /// Shape: [rows, cols / 8]
    pub packed: Vec<u32>,
    /// Per-group BF16 scales. Shape: [rows, cols / group_size]
    pub scales: Vec<u16>,
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
}

/// Quantize a BF16 weight matrix to symmetric INT4 with per-group scales.
///
/// Symmetric INT4: values in [-8, 7], scale chosen so that
///   max(abs(group)) maps to 7.
///
/// # Arguments
/// * `weight_bf16` - row-major BF16 weight data (as raw u16), length = rows * cols
/// * `rows` - number of rows (output dimension)
/// * `cols` - number of columns (input dimension), must be divisible by group_size
/// * `group_size` - quantization group size (typically 128)
pub fn quantize_int4(
    weight_bf16: &[u16],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> QuantizedInt4 {
    assert_eq!(weight_bf16.len(), rows * cols);
    assert!(cols % group_size == 0, "cols ({cols}) must be divisible by group_size ({group_size})");
    assert!(cols % PACK_FACTOR == 0, "cols ({cols}) must be divisible by {PACK_FACTOR}");

    let num_groups_per_row = cols / group_size;
    let packed_cols = cols / PACK_FACTOR;

    let mut scales = vec![0u16; rows * num_groups_per_row];
    let mut packed = vec![0u32; rows * packed_cols];

    for row in 0..rows {
        let row_offset = row * cols;

        // Pass 1: compute per-group scales
        for g in 0..num_groups_per_row {
            let group_start = row_offset + g * group_size;
            let mut amax: f32 = 0.0;
            for i in 0..group_size {
                let val = bf16_to_f32(weight_bf16[group_start + i]);
                amax = amax.max(val.abs());
            }
            // scale = amax / 7.0 (map max abs value to INT4 range [-8, 7])
            // Use 7.0 not 8.0 so positive range is fully used
            let scale = if amax == 0.0 { 1.0 } else { amax / 7.0 };
            scales[row * num_groups_per_row + g] = f32_to_bf16(scale);
        }

        // Pass 2: quantize and pack
        for g in 0..num_groups_per_row {
            let group_start = row_offset + g * group_size;
            let scale = bf16_to_f32(scales[row * num_groups_per_row + g]);
            let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

            for i in (0..group_size).step_by(PACK_FACTOR) {
                let mut word: u32 = 0;
                for j in 0..PACK_FACTOR {
                    let val = bf16_to_f32(weight_bf16[group_start + i + j]);
                    // Quantize: round to nearest, clamp to [-8, 7]
                    let q = (val * inv_scale).round().clamp(-8.0, 7.0) as i8;
                    // Store as unsigned 4-bit (0..15): q + 8
                    let u4 = (q + 8) as u8 & 0xF;
                    word |= (u4 as u32) << (j * 4);
                }
                let col_in_row = g * group_size + i;
                packed[row * packed_cols + col_in_row / PACK_FACTOR] = word;
            }
        }
    }

    QuantizedInt4 {
        packed,
        scales,
        rows,
        cols,
        group_size,
    }
}

/// Dequantize INT4 packed weights back to f32 for verification.
pub fn dequantize_int4(q: &QuantizedInt4) -> Vec<f32> {
    let num_groups_per_row = q.cols / q.group_size;
    let packed_cols = q.cols / PACK_FACTOR;
    let mut output = vec![0.0f32; q.rows * q.cols];

    for row in 0..q.rows {
        for g in 0..num_groups_per_row {
            let scale = bf16_to_f32(q.scales[row * num_groups_per_row + g]);

            for i in (0..q.group_size).step_by(PACK_FACTOR) {
                let col_in_row = g * q.group_size + i;
                let word = q.packed[row * packed_cols + col_in_row / PACK_FACTOR];

                for j in 0..PACK_FACTOR {
                    let u4 = ((word >> (j * 4)) & 0xF) as i8;
                    let q_val = u4 - 8; // back to signed [-8, 7]
                    let val = q_val as f32 * scale;
                    output[row * q.cols + col_in_row + j] = val;
                }
            }
        }
    }

    output
}

/// Apply Marlin permutation to already-packed INT4 weights.
/// This reorders the packed int32 words for GPU warp-level memory coalescing.
pub fn marlin_repack(_packed: &[u32], _k: usize, _n: usize) -> Vec<u32> {
    // TODO: implement the Marlin repack permutation
    // This is the same permutation as gptq_marlin_moe_repack in vLLM
    todo!("Marlin repack permutation")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::safetensors_io::MmapSafetensors;
    use std::path::Path;

    #[test]
    fn test_quantize_roundtrip_synthetic() {
        // Small synthetic test: 4 rows, 128 cols (one group)
        let rows = 4;
        let cols = 128;
        let group_size = 128;

        // Create synthetic BF16 data: values in [-0.1, 0.1]
        let mut bf16_data = vec![0u16; rows * cols];
        for i in 0..bf16_data.len() {
            let val = (i as f32 / bf16_data.len() as f32 - 0.5) * 0.2;
            bf16_data[i] = f32_to_bf16(val);
        }

        let q = quantize_int4(&bf16_data, rows, cols, group_size);

        assert_eq!(q.packed.len(), rows * cols / PACK_FACTOR);
        assert_eq!(q.scales.len(), rows * (cols / group_size));

        // Dequantize and check error
        let deq = dequantize_int4(&q);
        let mut max_err: f32 = 0.0;
        let mut sum_sq_err: f64 = 0.0;
        for i in 0..bf16_data.len() {
            let orig = bf16_to_f32(bf16_data[i]);
            let err = (orig - deq[i]).abs();
            max_err = max_err.max(err);
            sum_sq_err += (err as f64) * (err as f64);
        }
        let rmse = (sum_sq_err / bf16_data.len() as f64).sqrt();

        eprintln!("Synthetic roundtrip: max_err={:.6}, rmse={:.6}", max_err, rmse);
        // INT4 with 16 levels over [-0.1, 0.1] → step ~0.013, max_err should be < step/2
        assert!(max_err < 0.02, "Max error too large: {max_err}");
    }

    #[test]
    fn test_quantize_roundtrip_v2_lite() {
        let path = Path::new("/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite/model-00001-of-000004.safetensors");
        if !path.exists() {
            eprintln!("Skipping — V2-Lite not downloaded");
            return;
        }

        let st = MmapSafetensors::open(path).expect("Failed to open");
        let gate_name = "model.layers.1.mlp.experts.0.gate_proj.weight";
        let info = st.tensor_info(gate_name).expect("Not found");
        let bf16_data: &[u16] = st.tensor_as_slice(gate_name).expect("Failed to read");

        let rows = info.shape[0]; // 1408
        let cols = info.shape[1]; // 2048

        let q = quantize_int4(bf16_data, rows, cols, DEFAULT_GROUP_SIZE);
        let deq = dequantize_int4(&q);

        // Compute error stats vs original BF16
        let mut max_err: f32 = 0.0;
        let mut sum_sq_err: f64 = 0.0;
        let mut sum_sq_orig: f64 = 0.0;
        for i in 0..bf16_data.len() {
            let orig = bf16_to_f32(bf16_data[i]);
            let err = (orig - deq[i]).abs();
            max_err = max_err.max(err);
            sum_sq_err += (err as f64) * (err as f64);
            sum_sq_orig += (orig as f64) * (orig as f64);
        }
        let rmse = (sum_sq_err / bf16_data.len() as f64).sqrt();
        let rms_orig = (sum_sq_orig / bf16_data.len() as f64).sqrt();
        let snr_db = 20.0 * (rms_orig / rmse).log10();

        eprintln!(
            "V2-Lite gate_proj [{rows}, {cols}] INT4 roundtrip: max_err={:.6}, rmse={:.6}, SNR={:.1} dB",
            max_err, rmse, snr_db
        );
        eprintln!(
            "  Packed size: {} KB (was {} KB BF16) — {:.1}x compression",
            q.packed.len() * 4 / 1024,
            bf16_data.len() * 2 / 1024,
            (bf16_data.len() * 2) as f64 / (q.packed.len() * 4 + q.scales.len() * 2) as f64,
        );

        // INT4 SNR should be > 20 dB for well-distributed weights
        assert!(snr_db > 15.0, "SNR too low: {snr_db:.1} dB");
    }
}
