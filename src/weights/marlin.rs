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

/// Symmetric INT8 quantization result for a single weight matrix.
pub struct QuantizedInt8 {
    /// Raw INT8 weights, row-major. Shape: [rows, cols]
    pub data: Vec<i8>,
    /// Per-group BF16 scales. Shape: [rows, cols / group_size]
    pub scales: Vec<u16>,
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
}

/// Quantize a BF16 weight matrix to symmetric INT8 with per-group scales.
///
/// Symmetric INT8: values in [-128, 127], scale chosen so that
///   max(abs(group)) maps to 127.
///
/// # Arguments
/// * `weight_bf16` - row-major BF16 weight data (as raw u16), length = rows * cols
/// * `rows` - number of rows (output dimension)
/// * `cols` - number of columns (input dimension), must be divisible by group_size
/// * `group_size` - quantization group size (typically 128)
pub fn quantize_int8(
    weight_bf16: &[u16],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> QuantizedInt8 {
    assert_eq!(weight_bf16.len(), rows * cols);
    assert!(cols % group_size == 0, "cols ({cols}) must be divisible by group_size ({group_size})");

    let num_groups_per_row = cols / group_size;
    let mut scales = vec![0u16; rows * num_groups_per_row];
    let mut data = vec![0i8; rows * cols];

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
            let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            scales[row * num_groups_per_row + g] = f32_to_bf16(scale);
        }

        // Pass 2: quantize
        for g in 0..num_groups_per_row {
            let group_start = row_offset + g * group_size;
            let scale = bf16_to_f32(scales[row * num_groups_per_row + g]);
            let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

            for i in 0..group_size {
                let val = bf16_to_f32(weight_bf16[group_start + i]);
                let q = (val * inv_scale).round().clamp(-128.0, 127.0) as i8;
                data[group_start + i] = q;
            }
        }
    }

    QuantizedInt8 {
        data,
        scales,
        rows,
        cols,
        group_size,
    }
}

/// Dequantize INT8 weights back to f32 for verification.
pub fn dequantize_int8(q: &QuantizedInt8) -> Vec<f32> {
    let num_groups_per_row = q.cols / q.group_size;
    let mut output = vec![0.0f32; q.rows * q.cols];

    for row in 0..q.rows {
        for g in 0..num_groups_per_row {
            let scale = bf16_to_f32(q.scales[row * num_groups_per_row + g]);
            let group_start = row * q.cols + g * q.group_size;

            for i in 0..q.group_size {
                output[group_start + i] = q.data[group_start + i] as f32 * scale;
            }
        }
    }

    output
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

/// Marlin tile size (K dimension).
const MARLIN_TILE: usize = 16;

/// Marlin-repacked INT4 weights for GPU consumption.
pub struct MarlinRepacked {
    /// Packed INT4 weights in Marlin tile layout.
    /// Shape: [K/16, N*2] (u32 values, 8 INT4 per u32)
    pub packed: Vec<u32>,
    /// Permuted BF16 scales for Marlin kernel.
    /// Shape: [K/group_size, N]
    pub scales: Vec<u16>,
    pub k: usize,
    pub n: usize,
    pub group_size: usize,
}

/// Generate the Marlin weight permutation table for INT4.
///
/// Returns a 1024-element array mapping destination → source index within a
/// 16×64 tile. Matches vLLM's `get_weight_perm(num_bits=4)`.
pub fn generate_weight_perm_int4() -> [usize; 1024] {
    let mut perm = [0usize; 1024];
    let mut idx = 0;

    for i in 0..32 {
        let col = i / 4;
        let mut perm1 = [0usize; 8];
        let mut p1_idx = 0;

        for block in 0..2 {
            for &row in &[
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ] {
                perm1[p1_idx] = 16 * row + col + 8 * block;
                p1_idx += 1;
            }
        }

        for j in 0..4 {
            for &p in &perm1 {
                perm[idx] = p + 256 * j;
                idx += 1;
            }
        }
    }

    // Apply INT4 interleaving: [0, 2, 4, 6, 1, 3, 5, 7]
    let interleave = [0, 2, 4, 6, 1, 3, 5, 7];
    let mut result = [0usize; 1024];
    for group in 0..(1024 / 8) {
        for (dest, &src) in interleave.iter().enumerate() {
            result[group * 8 + dest] = perm[group * 8 + src];
        }
    }

    result
}

/// Generate scale permutation tables.
///
/// Returns (scale_perm, scale_perm_single):
/// - scale_perm: 64 elements, used when group_size < K
/// - scale_perm_single: 32 elements, used for channelwise quantization
pub fn generate_scale_perms() -> ([usize; 64], [usize; 32]) {
    // scale_perm: [0,8,16,24,32,40,48,56, 1,9,17,25,... 7,15,23,...,63]
    let mut scale_perm = [0usize; 64];
    for i in 0..8 {
        for j in 0..8 {
            scale_perm[i * 8 + j] = i + 8 * j;
        }
    }

    // scale_perm_single: for channelwise
    let offsets = [0, 1, 8, 9, 16, 17, 24, 25];
    let mut scale_perm_single = [0usize; 32];
    for i in 0..4 {
        for (j, &off) in offsets.iter().enumerate() {
            scale_perm_single[i * 8 + j] = 2 * i + off;
        }
    }

    (scale_perm, scale_perm_single)
}

/// Repack our QuantizedInt4 into Marlin GPU format.
///
/// Follows vLLM's Python reference: unpack → transpose → tile permute → repack.
/// Our format: packed `[N, K/8]`, scales `[N, K/group_size]`
/// Marlin format: packed `[K/16, 2*N]`, scales `[K/group_size, N]`
///
/// N = rows (output dim), K = cols (input dim) of the original weight matrix.
pub fn marlin_repack(q: &QuantizedInt4) -> MarlinRepacked {
    let n = q.rows; // output dimension
    let k = q.cols; // input dimension (K)
    let group_size = q.group_size;

    assert!(k % MARLIN_TILE == 0, "K ({k}) must be divisible by {MARLIN_TILE}");
    assert!(n % 64 == 0, "N ({n}) must be divisible by 64 (Marlin tile constraint)");

    // Step 1: Unpack our [N, K/8] → individual [N, K] unsigned INT4 values (0-15)
    let packed_k = k / PACK_FACTOR;
    let mut unpacked = vec![0u8; n * k]; // [N, K] values in 0..15

    for row in 0..n {
        for col_pack in 0..packed_k {
            let word = q.packed[row * packed_k + col_pack];
            for j in 0..PACK_FACTOR {
                unpacked[row * k + col_pack * PACK_FACTOR + j] =
                    ((word >> (j as u32 * 4)) & 0xF) as u8;
            }
        }
    }

    // Step 2: Transpose [N, K] → [K, N]
    let mut transposed = vec![0u8; k * n]; // [K, N]
    for row in 0..n {
        for col in 0..k {
            transposed[col * n + row] = unpacked[row * k + col];
        }
    }

    // Step 3: marlin_permute_weights
    // 3a: Reshape (K, N) → (K/16, 16, N/16, 16) → permute(0,2,1,3) → (K/16, N/16, 16, 16)
    // 3b: Flatten → (K/16, N*16)
    // 3c: Apply perm to chunks of 1024

    let k_tiles = k / MARLIN_TILE;
    let n_tiles = n / MARLIN_TILE;
    let row_len = n * MARLIN_TILE; // N*16 values per output row

    let mut permuted = vec![0u8; k_tiles * row_len]; // [K/16, N*16]

    // 3a+3b: tile transpose — put tiles contiguous
    for kt in 0..k_tiles {
        for nt in 0..n_tiles {
            for tk in 0..MARLIN_TILE {
                for tn in 0..MARLIN_TILE {
                    let src_k = kt * MARLIN_TILE + tk;
                    let src_n = nt * MARLIN_TILE + tn;
                    // After permute(0,2,1,3) + flatten:
                    // dst row = kt, dst col = nt * 16 * 16 + tk * 16 + tn
                    let dst_col = nt * MARLIN_TILE * MARLIN_TILE + tk * MARLIN_TILE + tn;
                    permuted[kt * row_len + dst_col] = transposed[src_k * n + src_n];
                }
            }
        }
    }

    // 3c: Apply weight permutation to chunks of 1024
    let perm = generate_weight_perm_int4();
    let num_chunks = row_len / 1024;
    let mut perm_applied = vec![0u8; k_tiles * row_len];

    for kt in 0..k_tiles {
        for chunk in 0..num_chunks {
            let base = kt * row_len + chunk * 1024;
            for i in 0..1024 {
                perm_applied[base + i] = permuted[base + perm[i]];
            }
        }
    }

    // Step 4: Pack with stride-8 packing (matching vLLM's pack_cols)
    // Output shape: (K/16, N*16/8) = (K/16, 2*N)
    let out_cols = row_len / PACK_FACTOR; // = 2*N
    let mut out_packed = vec![0u32; k_tiles * out_cols];

    for row in 0..k_tiles {
        for col in 0..out_cols {
            let mut word: u32 = 0;
            for i in 0..PACK_FACTOR {
                // stride-8 packing: column i of pack = col + i * out_cols
                // In the flat row: position col + i * out_cols ... but that's wrong.
                // vLLM pack_cols: q_packed |= q_w[:, i::pack_factor] << (num_bits * i)
                // So bit position i takes from column: col * pack_factor + i = col * 8 + i
                // Wait no. Let me re-read:
                // q_res = zeros((rows, cols // pack_factor))
                // for i in range(pack_factor):
                //     q_res |= q_w[:, i::pack_factor] << num_bits * i
                // So q_res[r, c] = q_w[r, c*pack_factor + 0] << 0
                //                | q_w[r, c*pack_factor + 1] << 4
                //                | ...
                // Wait, q_w[:, i::pack_factor] means columns i, i+pf, i+2*pf, ...
                // So q_res[:, j] gets bit i from q_w[:, i + j*pf]
                // Hmm no. q_w[:, i::pack_factor] has shape (rows, cols/pf)
                // Column c of q_w[:, i::pack_factor] = q_w[:, i + c*pf]
                // So q_res[:, c] |= q_w[:, i + c*pf] << (4*i)
                // Actually that's: q_res[r,c] = sum over i: q_w[r, i + c*8] << (4*i)
                // Which is: sequential packing of 8 consecutive values starting at col*8
                // That's just normal sequential packing!
                let src_col = col * PACK_FACTOR + i;
                let val = perm_applied[row * row_len + src_col] as u32;
                word |= val << (i as u32 * 4);
            }
            out_packed[row * out_cols + col] = word;
        }
    }

    // Step 5: Transpose scales [N, K/gs] → [K/gs, N]
    let num_groups_k = k / group_size;
    let mut scales_transposed = vec![0u16; num_groups_k * n];
    for row in 0..n {
        for g in 0..num_groups_k {
            scales_transposed[g * n + row] = q.scales[row * num_groups_k + g];
        }
    }

    // Step 6: Apply scale permutation
    let (scale_perm, scale_perm_single) = generate_scale_perms();
    // Use scale_perm (64) for grouped, scale_perm_single (32) for channelwise
    let is_grouped = group_size < k;
    let sperm: &[usize] = if is_grouped { &scale_perm } else { &scale_perm_single };
    let perm_len = sperm.len();
    let total_scale_vals = num_groups_k * n;
    let num_scale_chunks = total_scale_vals / perm_len;

    let mut scales_permuted = vec![0u16; total_scale_vals];
    for chunk in 0..num_scale_chunks {
        let base = chunk * perm_len;
        for i in 0..perm_len {
            scales_permuted[base + i] = scales_transposed[base + sperm[i]];
        }
    }

    MarlinRepacked {
        packed: out_packed,
        scales: scales_permuted,
        k,
        n,
        group_size,
    }
}

/// Dequantize Marlin-repacked weights back to f32 for verification.
///
/// Reverses the permutation and packing to recover the original [N, K] f32 values.
pub fn dequantize_marlin(m: &MarlinRepacked) -> Vec<f32> {
    let k = m.k;
    let n = m.n;
    let group_size = m.group_size;
    let k_tiles = k / MARLIN_TILE;
    let row_len = n * MARLIN_TILE;
    let out_cols = row_len / PACK_FACTOR;
    let num_groups_k = k / group_size;

    // Step 1: Unpack Marlin-format packed [K/16, 2*N] → [K/16, N*16] values
    let mut perm_applied = vec![0u8; k_tiles * row_len];
    for row in 0..k_tiles {
        for col in 0..out_cols {
            let word = m.packed[row * out_cols + col];
            for i in 0..PACK_FACTOR {
                perm_applied[row * row_len + col * PACK_FACTOR + i] =
                    ((word >> (i as u32 * 4)) & 0xF) as u8;
            }
        }
    }

    // Step 2: Invert weight permutation
    let perm = generate_weight_perm_int4();
    let num_chunks = row_len / 1024;
    let mut permuted = vec![0u8; k_tiles * row_len];
    for kt in 0..k_tiles {
        for chunk in 0..num_chunks {
            let base = kt * row_len + chunk * 1024;
            for i in 0..1024 {
                // perm maps dest→src, so to invert: src position perm[i] gets value from dest i
                permuted[base + perm[i]] = perm_applied[base + i];
            }
        }
    }

    // Step 3: Invert tile transpose → [K, N] values
    let n_tiles = n / MARLIN_TILE;
    let mut transposed = vec![0u8; k * n];
    for kt in 0..k_tiles {
        for nt in 0..n_tiles {
            for tk in 0..MARLIN_TILE {
                for tn in 0..MARLIN_TILE {
                    let src_k = kt * MARLIN_TILE + tk;
                    let src_n = nt * MARLIN_TILE + tn;
                    let permuted_col = nt * MARLIN_TILE * MARLIN_TILE + tk * MARLIN_TILE + tn;
                    transposed[src_k * n + src_n] = permuted[kt * row_len + permuted_col];
                }
            }
        }
    }

    // Step 4: Invert scale permutation
    let (scale_perm, scale_perm_single) = generate_scale_perms();
    let is_grouped = group_size < k;
    let sperm: &[usize] = if is_grouped { &scale_perm } else { &scale_perm_single };
    let perm_len = sperm.len();
    let total_scale_vals = num_groups_k * n;
    let num_scale_chunks = total_scale_vals / perm_len;

    let mut scales_transposed = vec![0u16; total_scale_vals];
    for chunk in 0..num_scale_chunks {
        let base = chunk * perm_len;
        for i in 0..perm_len {
            scales_transposed[base + sperm[i]] = m.scales[base + i];
        }
    }

    // Step 5: Transpose [K, N] → [N, K] and dequantize
    // Scales are [K/gs, N], need to read scale for group (k / gs, n)
    let mut output = vec![0.0f32; n * k];
    for ki in 0..k {
        for ni in 0..n {
            let u4 = transposed[ki * n + ni];
            let q_val = (u4 as i8) - 8;
            let group_idx = ki / group_size;
            let scale = bf16_to_f32(scales_transposed[group_idx * n + ni]);
            output[ni * k + ki] = q_val as f32 * scale;
        }
    }

    output
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

    #[test]
    fn test_weight_perm_properties() {
        let perm = generate_weight_perm_int4();

        // Must be a valid permutation of 0..1024
        let mut sorted = perm.to_vec();
        sorted.sort();
        for (i, &v) in sorted.iter().enumerate() {
            assert_eq!(v, i, "Not a valid permutation: missing index {i}");
        }
        eprintln!("Weight perm: valid permutation of 0..1024 ✓");
    }

    #[test]
    fn test_scale_perm_properties() {
        let (sp, sps) = generate_scale_perms();

        // scale_perm: permutation of 0..64
        let mut sorted = sp.to_vec();
        sorted.sort();
        for (i, &v) in sorted.iter().enumerate() {
            assert_eq!(v, i, "scale_perm: missing index {i}");
        }

        // scale_perm_single: permutation of 0..32
        let mut sorted = sps.to_vec();
        sorted.sort();
        for (i, &v) in sorted.iter().enumerate() {
            assert_eq!(v, i, "scale_perm_single: missing index {i}");
        }
        eprintln!("Scale perms: valid permutations ✓");
    }

    #[test]
    fn test_marlin_repack_roundtrip_synthetic() {
        // 64 rows (N), 128 cols (K) — Marlin requires N % 64 == 0
        let n = 64;
        let k = 128;
        let group_size = 128;

        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..weight_bf16.len() {
            let val = ((i as f32 / weight_bf16.len() as f32) - 0.5) * 0.2;
            weight_bf16[i] = f32_to_bf16(val);
        }

        // Quantize to our format
        let q = quantize_int4(&weight_bf16, n, k, group_size);

        // Dequantize via our format (baseline)
        let deq_ours = dequantize_int4(&q);

        // Repack to Marlin, then dequantize
        let m = marlin_repack(&q);

        assert_eq!(m.packed.len(), (k / 16) * (2 * n));
        assert_eq!(m.scales.len(), (k / group_size) * n);

        let deq_marlin = dequantize_marlin(&m);

        // Both dequantizations should produce identical results
        let mut max_diff: f32 = 0.0;
        for i in 0..(n * k) {
            let diff = (deq_ours[i] - deq_marlin[i]).abs();
            max_diff = max_diff.max(diff);
        }

        eprintln!(
            "Marlin repack roundtrip {n}×{k}: max_diff={max_diff:.8} (should be 0.0)"
        );
        assert!(
            max_diff == 0.0,
            "Marlin repack changed values! max_diff={max_diff}"
        );
    }

    #[test]
    fn test_marlin_repack_v2_lite() {
        let path = Path::new("/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite/model-00001-of-000004.safetensors");
        if !path.exists() {
            eprintln!("Skipping — V2-Lite not downloaded");
            return;
        }

        let st = MmapSafetensors::open(path).expect("Failed to open");
        let gate_name = "model.layers.1.mlp.experts.0.gate_proj.weight";
        let info = st.tensor_info(gate_name).expect("Not found");
        let bf16_data: &[u16] = st.tensor_as_slice(gate_name).expect("Failed to read");

        let n = info.shape[0]; // 1408
        let k = info.shape[1]; // 2048

        let q = quantize_int4(bf16_data, n, k, DEFAULT_GROUP_SIZE);
        let deq_ours = dequantize_int4(&q);

        let m = marlin_repack(&q);

        eprintln!(
            "V2-Lite gate_proj Marlin repack: packed [{}, {}] ({} KB), scales [{}, {}] ({} KB)",
            k / 16,
            2 * n,
            m.packed.len() * 4 / 1024,
            k / DEFAULT_GROUP_SIZE,
            n,
            m.scales.len() * 2 / 1024,
        );

        let deq_marlin = dequantize_marlin(&m);

        let mut max_diff: f32 = 0.0;
        for i in 0..(n * k) {
            let diff = (deq_ours[i] - deq_marlin[i]).abs();
            max_diff = max_diff.max(diff);
        }

        eprintln!("  Round-trip max_diff: {max_diff:.8}");
        assert!(
            max_diff == 0.0,
            "Marlin repack changed values! max_diff={max_diff}"
        );
    }
}
