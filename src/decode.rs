//! CPU decode compute kernels for non-MoE layers.
//!
//! Provides quantized INT4/INT8 matmul, RMSNorm, and SiLU for CPU-only decode.
//! Weights are quantized once at prepare() time, then reused for all decode steps.
//! Activations are f32, quantized to INT16 per-call before each matmul.

use crate::kernel::avx2::{
    matmul_int4_transposed_integer, matmul_int4_transposed_integer_parallel,
    matmul_int4_transposed_integer_tiled, matmul_int4_transposed_integer_parallel_tiled,
    matmul_int8_transposed_integer, matmul_int8_transposed_integer_parallel,
    matmul_int8_transposed_integer_tiled, matmul_int8_transposed_integer_parallel_tiled,
    repack_tiled_int4_packed, repack_tiled_int8_packed, repack_tiled_scales,
    quantize_activation_int16_f32,
};
use crate::moe::{ExpertScratch, moe_forward_unified, moe_forward_flattened};
use crate::weights::marlin::f32_to_bf16;
use crate::weights::WeightStore;
use pyo3::prelude::*;
use std::sync::Arc;

/// A single quantized weight matrix in transposed format for CPU decode.
struct TransposedWeight {
    /// Packed weight data (transposed).
    /// INT4: [K/8, N] as u32 (8 nibbles per u32)
    /// INT8: [K, N] as i8 packed into u32 container
    /// When tiled: [N/TILE, K/8, TILE] (INT4) or [N/TILE, K, TILE] (INT8)
    packed: Vec<u32>,
    /// Per-group scales in BF16 (transposed). [K/group_size, N]
    /// When tiled: [N/TILE, K/group_size, TILE]
    scales: Vec<u16>,
    /// Output dimension (N = rows of original weight).
    rows: usize,
    /// Input dimension (K = cols of original weight).
    cols: usize,
    group_size: usize,
    num_bits: u8,
    /// Whether data is in tiled layout (TILE_N=256 wide tiles).
    tiled: bool,
}

/// Quantize f32 weight matrix [N, K] to transposed INT4 format.
///
/// INT4 symmetric: values mapped to [-8, 7], 8 packed per u32.
/// Output layout: packed [K/8, N], scales [K/gs, N] (both transposed).
fn quantize_f32_to_transposed_int4(
    weight: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> TransposedWeight {
    assert_eq!(weight.len(), rows * cols);
    assert!(cols % group_size == 0, "cols {} must be divisible by group_size {}", cols, group_size);
    assert!(cols % 8 == 0, "cols {} must be divisible by 8 for INT4", cols);
    assert!(group_size % 8 == 0);

    let packed_k = cols / 8;
    let num_groups = cols / group_size;

    // Quantize in row-major [N, K/8] packed, [N, K/gs] scales
    let mut packed_rm = vec![0u32; rows * packed_k];
    let mut scales_rm = vec![0u16; rows * num_groups];

    for row in 0..rows {
        let row_base = row * cols;
        for g in 0..num_groups {
            let g_start = g * group_size;

            let mut max_abs: f32 = 0.0;
            for i in 0..group_size {
                max_abs = max_abs.max(weight[row_base + g_start + i].abs());
            }

            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            let inv_scale = if max_abs > 0.0 { 7.0 / max_abs } else { 0.0 };
            scales_rm[row * num_groups + g] = f32_to_bf16(scale);

            for pack in 0..(group_size / 8) {
                let base = g_start + pack * 8;
                let mut word: u32 = 0;
                for j in 0..8u32 {
                    let val = weight[row_base + base + j as usize];
                    let q = ((val * inv_scale).round() as i32).clamp(-8, 7);
                    let u4 = (q + 8) as u32;
                    word |= u4 << (j * 4);
                }
                packed_rm[row * packed_k + g * (group_size / 8) + pack] = word;
            }
        }
    }

    // Transpose packed: [N, K/8] -> [K/8, N]
    let mut packed = vec![0u32; packed_k * rows];
    for k in 0..packed_k {
        for n in 0..rows {
            packed[k * rows + n] = packed_rm[n * packed_k + k];
        }
    }

    // Transpose scales: [N, K/gs] -> [K/gs, N]
    let mut scales = vec![0u16; num_groups * rows];
    for g in 0..num_groups {
        for n in 0..rows {
            scales[g * rows + n] = scales_rm[n * num_groups + g];
        }
    }

    TransposedWeight { packed, scales, rows, cols, group_size, num_bits: 4, tiled: false }
}

/// Quantize f32 weight matrix [N, K] to transposed INT8 format.
///
/// INT8 symmetric: values mapped to [-127, 127], stored as i8 in u32 container.
/// Output layout: data [K, N] as i8 in u32, scales [K/gs, N] (both transposed).
fn quantize_f32_to_transposed_int8(
    weight: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> TransposedWeight {
    assert_eq!(weight.len(), rows * cols);
    assert!(cols % group_size == 0, "cols {} must be divisible by group_size {}", cols, group_size);
    assert!(group_size % 2 == 0);

    let num_groups = cols / group_size;

    // Quantize in row-major
    let mut data_rm = vec![0i8; rows * cols];
    let mut scales_rm = vec![0u16; rows * num_groups];

    for row in 0..rows {
        let row_base = row * cols;
        for g in 0..num_groups {
            let g_start = g * group_size;
            let mut max_abs: f32 = 0.0;
            for i in 0..group_size {
                max_abs = max_abs.max(weight[row_base + g_start + i].abs());
            }
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = if max_abs > 0.0 { 127.0 / max_abs } else { 0.0 };
            scales_rm[row * num_groups + g] = f32_to_bf16(scale);
            for i in 0..group_size {
                let val = weight[row_base + g_start + i];
                data_rm[row_base + g_start + i] =
                    ((val * inv_scale).round() as i32).clamp(-128, 127) as i8;
            }
        }
    }

    // Transpose data: [N, K] -> [K, N] as i8, packed into Vec<u32>
    let byte_count = cols * rows;
    let u32_count = (byte_count + 3) / 4;
    let mut transposed_bytes = vec![0i8; u32_count * 4];
    for k in 0..cols {
        for n in 0..rows {
            transposed_bytes[k * rows + n] = data_rm[n * cols + k];
        }
    }
    let packed: Vec<u32> = unsafe {
        let mut v = vec![0u32; u32_count];
        std::ptr::copy_nonoverlapping(
            transposed_bytes.as_ptr() as *const u8,
            v.as_mut_ptr() as *mut u8,
            u32_count * 4,
        );
        v
    };

    // Transpose scales: [N, K/gs] -> [K/gs, N]
    let mut scales = vec![0u16; num_groups * rows];
    for g in 0..num_groups {
        for n in 0..rows {
            scales[g * rows + n] = scales_rm[n * num_groups + g];
        }
    }

    TransposedWeight { packed, scales, rows, cols, group_size, num_bits: 8, tiled: false }
}

/// A single MoE routing weight stored as float32 (small, accuracy-critical).
struct RouteWeight {
    /// Gate weight [num_experts, hidden_dim] stored row-major.
    data: Vec<f32>,
    /// Optional bias [num_experts].
    bias: Option<Vec<f32>>,
    /// Optional e_score_correction [num_experts].
    e_score_corr: Option<Vec<f32>>,
    num_experts: usize,
    hidden_dim: usize,
}

/// CPU decode weight store — holds quantized non-MoE weights for fast matmul.
#[pyclass]
pub struct CpuDecodeStore {
    weights: Vec<TransposedWeight>,
    /// Scratch buffer for INT16 activation quantization (reused across calls).
    act_int16: Vec<i16>,
    act_scales: Vec<f32>,
    /// Current scratch size (max K seen so far).
    scratch_k: usize,
    group_size: usize,
    /// Whether to use parallel (multi-threaded) matmul for large outputs.
    parallel: bool,
    /// Whether norms use (1+w)*x instead of w*x (Qwen3-Next).
    norm_bias_one: bool,
    /// MoE routing weights (float32, per-layer). Indexed by route_id.
    route_weights: Vec<RouteWeight>,
    /// Norm weights stored in Rust for zero-overhead access. Indexed by norm_id.
    norm_weights: Vec<Vec<f32>>,
    /// Pre-allocated scratch for MoE routing (max_experts floats).
    route_logits: Vec<f32>,
    route_scores: Vec<f32>,
    route_corrected: Vec<f32>,
    /// Full decode graph for single-call decode_step (optional, built by configure_decode).
    decode_graph: Option<Box<DecodeGraph>>,
    /// Contiguous mmap regions backing TransposedWeight data (for THP).
    /// Each entry is (base_ptr as usize, byte_len) for munmap on drop.
    /// Stored as usize instead of *mut u8 to satisfy Send/Sync requirements.
    mmap_regions: Vec<(usize, usize)>,
}

#[pymethods]
impl CpuDecodeStore {
    #[new]
    #[pyo3(signature = (group_size=128, parallel=true, norm_bias_one=false))]
    pub fn new(group_size: usize, parallel: bool, norm_bias_one: bool) -> Self {
        CpuDecodeStore {
            weights: Vec::new(),
            act_int16: Vec::new(),
            act_scales: Vec::new(),
            scratch_k: 0,
            group_size,
            parallel,
            norm_bias_one,
            route_weights: Vec::new(),
            norm_weights: Vec::new(),
            route_logits: Vec::new(),
            route_scores: Vec::new(),
            route_corrected: Vec::new(),
            decode_graph: None,
            mmap_regions: Vec::new(),
        }
    }

    /// Store a weight matrix from f32 data. Returns weight ID.
    ///
    /// Args:
    ///   data_ptr: pointer to f32 [rows, cols] row-major
    ///   rows: output dimension (N)
    ///   cols: input dimension (K)
    ///   num_bits: 4 or 8
    pub fn store_weight_f32(
        &mut self,
        data_ptr: usize,
        rows: usize,
        cols: usize,
        num_bits: u8,
    ) -> PyResult<usize> {
        if num_bits != 4 && num_bits != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("num_bits must be 4 or 8, got {}", num_bits)));
        }
        if cols % self.group_size != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("cols {} must be divisible by group_size {}", cols, self.group_size)));
        }
        if num_bits == 4 && cols % 8 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("cols {} must be divisible by 8 for INT4", cols)));
        }

        let data: &[f32] = unsafe {
            std::slice::from_raw_parts(data_ptr as *const f32, rows * cols)
        };

        let weight = match num_bits {
            4 => quantize_f32_to_transposed_int4(data, rows, cols, self.group_size),
            8 => quantize_f32_to_transposed_int8(data, rows, cols, self.group_size),
            _ => unreachable!(),
        };

        // Grow scratch buffers if needed
        if cols > self.scratch_k {
            self.scratch_k = cols;
            self.act_int16 = vec![0i16; cols];
            self.act_scales = vec![0f32; cols / self.group_size];
        }

        let id = self.weights.len();
        let bytes = weight.packed.len() * 4 + weight.scales.len() * 2;
        self.weights.push(weight);
        log::debug!("Stored weight {}: [{}x{}] INT{} transposed, {:.1} KB",
            id, rows, cols, num_bits, bytes as f64 / 1024.0);
        Ok(id)
    }

    /// Matrix-vector multiply: output[N] = W[N,K] @ input[K]
    ///
    /// Input is f32, internally quantized to INT16. Output is f32.
    pub fn matmul(
        &mut self,
        weight_id: usize,
        input_ptr: usize,
        output_ptr: usize,
    ) -> PyResult<()> {
        if weight_id >= self.weights.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("weight_id {} out of range ({})", weight_id, self.weights.len())));
        }
        let w = &self.weights[weight_id];
        let k = w.cols;
        let n = w.rows;
        let gs = w.group_size;

        let input: &[f32] = unsafe {
            std::slice::from_raw_parts(input_ptr as *const f32, k)
        };
        let output: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(output_ptr as *mut f32, n)
        };

        // Quantize input to INT16
        quantize_activation_int16_f32(
            input, gs, &mut self.act_int16[..k], &mut self.act_scales[..k / gs]);

        self.dispatch_matmul(weight_id, &self.act_int16[..k], &self.act_scales[..k / gs], output);
        Ok(())
    }

    /// Batch matmul: quantize input once, run multiple matmuls.
    ///
    /// All weights must have the same input dimension (K).
    /// weight_ids: list of weight IDs
    /// input_ptr: f32 [K]
    /// output_ptrs: list of f32 output pointers
    pub fn matmul_batch(
        &mut self,
        weight_ids: Vec<usize>,
        input_ptr: usize,
        output_ptrs: Vec<usize>,
    ) -> PyResult<()> {
        if weight_ids.len() != output_ptrs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "weight_ids and output_ptrs must have same length"));
        }
        if weight_ids.is_empty() {
            return Ok(());
        }

        let k = self.weights[weight_ids[0]].cols;
        let gs = self.weights[weight_ids[0]].group_size;

        let input: &[f32] = unsafe {
            std::slice::from_raw_parts(input_ptr as *const f32, k)
        };

        // Quantize input once
        quantize_activation_int16_f32(
            input, gs, &mut self.act_int16[..k], &mut self.act_scales[..k / gs]);

        for i in 0..weight_ids.len() {
            let wid = weight_ids[i];
            let w = &self.weights[wid];
            assert_eq!(w.cols, k, "All weights in batch must have same K");
            let n = w.rows;
            let output: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(output_ptrs[i] as *mut f32, n)
            };
            self.dispatch_matmul(wid, &self.act_int16[..k], &self.act_scales[..k / gs], output);
        }
        Ok(())
    }

    /// Fused add + RMSNorm (in-place on both buffers).
    ///
    /// If first_call: residual = hidden, hidden = rmsnorm(residual)
    /// Else: residual += hidden, hidden = rmsnorm(residual)
    pub fn fused_add_rmsnorm(
        &self,
        hidden_ptr: usize,
        residual_ptr: usize,
        weight_ptr: usize,
        eps: f32,
        size: usize,
        first_call: bool,
    ) -> PyResult<()> {
        let hidden: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(hidden_ptr as *mut f32, size)
        };
        let residual: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(residual_ptr as *mut f32, size)
        };
        let weight: &[f32] = unsafe {
            std::slice::from_raw_parts(weight_ptr as *const f32, size)
        };

        unsafe { fused_add_rmsnorm_avx2(hidden, residual, weight, eps, first_call, self.norm_bias_one) };
        Ok(())
    }

    /// Store norm weight in Rust for zero-overhead access. Returns norm_id.
    pub fn store_norm_weight(
        &mut self,
        data_ptr: usize,
        size: usize,
    ) -> PyResult<usize> {
        let data: &[f32] = unsafe {
            std::slice::from_raw_parts(data_ptr as *const f32, size)
        };
        let id = self.norm_weights.len();
        self.norm_weights.push(data.to_vec());
        Ok(id)
    }

    /// Fused add + RMSNorm using stored norm weight (zero Python overhead for weight access).
    ///
    /// Same as fused_add_rmsnorm but takes norm_id instead of weight_ptr,
    /// avoiding Python dict lookup and .data_ptr() per call.
    pub fn fused_add_rmsnorm_id(
        &self,
        hidden_ptr: usize,
        residual_ptr: usize,
        norm_id: usize,
        eps: f32,
        size: usize,
        first_call: bool,
    ) -> PyResult<()> {
        if norm_id >= self.norm_weights.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("norm_id {} out of range ({})", norm_id, self.norm_weights.len())));
        }
        let hidden: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(hidden_ptr as *mut f32, size)
        };
        let residual: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(residual_ptr as *mut f32, size)
        };
        let weight = &self.norm_weights[norm_id];

        unsafe { fused_add_rmsnorm_avx2(hidden, residual, weight, eps, first_call, self.norm_bias_one) };
        Ok(())
    }

    /// Standalone RMSNorm (non-fused).
    pub fn rmsnorm(
        &self,
        input_ptr: usize,
        weight_ptr: usize,
        eps: f32,
        output_ptr: usize,
        size: usize,
    ) -> PyResult<()> {
        let input: &[f32] = unsafe {
            std::slice::from_raw_parts(input_ptr as *const f32, size)
        };
        let weight: &[f32] = unsafe {
            std::slice::from_raw_parts(weight_ptr as *const f32, size)
        };
        let output: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(output_ptr as *mut f32, size)
        };

        let mut sum_sq: f32 = 0.0;
        for i in 0..size {
            sum_sq += input[i] * input[i];
        }
        let rms = (sum_sq / size as f32 + eps).sqrt().recip();

        if self.norm_bias_one {
            for i in 0..size {
                output[i] = input[i] * rms * (1.0 + weight[i]);
            }
        } else {
            for i in 0..size {
                output[i] = input[i] * rms * weight[i];
            }
        }

        Ok(())
    }

    /// SiLU(gate) * up -> output, elementwise.
    pub fn silu_mul(
        &self,
        gate_ptr: usize,
        up_ptr: usize,
        output_ptr: usize,
        size: usize,
    ) -> PyResult<()> {
        let gate: &[f32] = unsafe {
            std::slice::from_raw_parts(gate_ptr as *const f32, size)
        };
        let up: &[f32] = unsafe {
            std::slice::from_raw_parts(up_ptr as *const f32, size)
        };
        let output: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(output_ptr as *mut f32, size)
        };

        for i in 0..size {
            let x = gate[i];
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            output[i] = x * sigmoid * up[i];
        }

        Ok(())
    }

    /// Fused shared expert: gate_up_matmul → SiLU*mul → down_matmul.
    ///
    /// Does the full shared expert MLP in one Rust call, avoiding 3 FFI round-trips.
    /// input: f32 [K], gate_up_wid: fused [2*intermediate, K], down_wid: [K, intermediate]
    /// output: f32 [K] (same dim as input, since down_proj maps back to hidden)
    pub fn fused_shared_expert(
        &mut self,
        gate_up_wid: usize,
        down_wid: usize,
        input_ptr: usize,
        output_ptr: usize,
    ) -> PyResult<()> {
        // gate_up matmul: [2*intermediate] = gate_up_W @ input
        let gu_w = &self.weights[gate_up_wid];
        let k_in = gu_w.cols;
        let n_gu = gu_w.rows; // 2 * intermediate
        let gs = gu_w.group_size;
        let intermediate = n_gu / 2;

        let input: &[f32] = unsafe {
            std::slice::from_raw_parts(input_ptr as *const f32, k_in)
        };

        // Quantize input once for gate_up
        quantize_activation_int16_f32(
            input, gs, &mut self.act_int16[..k_in], &mut self.act_scales[..k_in / gs]);

        // gate_up matmul
        let mut gate_up = vec![0f32; n_gu];
        self.dispatch_matmul_ext(gate_up_wid, &self.act_int16[..k_in], &self.act_scales[..k_in / gs], &mut gate_up);

        // SiLU(gate) * up → hidden
        let mut se_hidden = vec![0f32; intermediate];
        for i in 0..intermediate {
            let x = gate_up[i];
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            se_hidden[i] = x * sigmoid * gate_up[intermediate + i];
        }

        // down matmul: quantize se_hidden, then matmul
        let d_w = &self.weights[down_wid];
        let k_down = d_w.cols;
        let n_down = d_w.rows;
        let gs_down = d_w.group_size;

        // Grow scratch if needed for down proj input
        if k_down > self.scratch_k {
            self.scratch_k = k_down;
            self.act_int16 = vec![0i16; k_down];
            self.act_scales = vec![0f32; k_down / gs_down];
        }

        quantize_activation_int16_f32(
            &se_hidden, gs_down, &mut self.act_int16[..k_down], &mut self.act_scales[..k_down / gs_down]);

        let output: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(output_ptr as *mut f32, n_down)
        };
        self.dispatch_matmul_ext(down_wid, &self.act_int16[..k_down], &self.act_scales[..k_down / gs_down], output);

        Ok(())
    }

    /// Gated DeltaNet recurrent state update + query output.
    ///
    /// state: [nv, dk, dv] f32 (modified in-place)
    /// q: [nv, dk] f32 (already L2-normalized and scaled, with heads expanded)
    /// k: [nv, dk] f32 (already L2-normalized, with heads expanded)
    /// v: [nv, dv] f32
    /// g: [nv] f32 (decay = exp(-A * softplus(a + dt_bias)), already computed)
    /// beta: [nv] f32 (sigmoid already applied)
    /// output: [nv, dv] f32 (query @ state result)
    pub fn linear_attention_recurrent(
        &self,
        state_ptr: usize,
        q_ptr: usize,
        k_ptr: usize,
        v_ptr: usize,
        g_ptr: usize,
        beta_ptr: usize,
        output_ptr: usize,
        nv: usize,
        dk: usize,
        dv: usize,
    ) -> PyResult<()> {
        let state: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(state_ptr as *mut f32, nv * dk * dv)
        };
        let q: &[f32] = unsafe { std::slice::from_raw_parts(q_ptr as *const f32, nv * dk) };
        let k: &[f32] = unsafe { std::slice::from_raw_parts(k_ptr as *const f32, nv * dk) };
        let v: &[f32] = unsafe { std::slice::from_raw_parts(v_ptr as *const f32, nv * dv) };
        let g: &[f32] = unsafe { std::slice::from_raw_parts(g_ptr as *const f32, nv) };
        let beta: &[f32] = unsafe { std::slice::from_raw_parts(beta_ptr as *const f32, nv) };
        let output: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(output_ptr as *mut f32, nv * dv)
        };

        // Dispatch to AVX2 implementation
        unsafe {
            linear_attention_recurrent_avx2(state, q, k, v, g, beta, output, nv, dk, dv);
        }
        Ok(())
    }

    /// Gated RMSNorm + SiLU gate: out = SiLU(z) * RMSNorm(x, weight)
    ///
    /// x: [nv * dv] f32 (recurrent output)
    /// z: [nv * dv] f32 (gate signal from projection)
    /// norm_weight: [nv, dv] or [nv * dv] f32
    /// output: [nv * dv] f32
    /// eps: RMSNorm epsilon
    /// nv: number of value heads (norm is per-head)
    /// dv: value head dimension
    pub fn gated_rmsnorm_silu(
        &self,
        x_ptr: usize,
        z_ptr: usize,
        norm_weight_ptr: usize,
        output_ptr: usize,
        eps: f32,
        nv: usize,
        dv: usize,
    ) -> PyResult<()> {
        let size = nv * dv;
        let x: &[f32] = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, size) };
        let z: &[f32] = unsafe { std::slice::from_raw_parts(z_ptr as *const f32, size) };
        let norm_weight: &[f32] = unsafe {
            std::slice::from_raw_parts(norm_weight_ptr as *const f32, size)
        };
        let output: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(output_ptr as *mut f32, size)
        };

        // Per-head RMSNorm: for each head h, norm over dv dimensions
        for h in 0..nv {
            let base = h * dv;
            let mut sum_sq = 0.0f32;
            for j in 0..dv {
                sum_sq += x[base + j] * x[base + j];
            }
            let rms = (sum_sq / dv as f32 + eps).sqrt().recip();

            for j in 0..dv {
                let normed = x[base + j] * rms * norm_weight[base + j];
                // SiLU(z) * normed
                let zval = z[base + j];
                let silu_z = zval / (1.0 + (-zval).exp());
                output[base + j] = silu_z * normed;
            }
        }

        Ok(())
    }

    /// Fused linear attention conv: un-interleave + conv1d state update + depthwise conv +
    /// SiLU + gate parameters + head expansion + L2 normalize.
    ///
    /// Replaces ~15 Python tensor ops per layer with a single Rust call.
    ///
    /// Inputs:
    ///   qkvz_ptr: [nk * (2*dk + 2*dv*hr)] f32 — projection output (interleaved)
    ///   ba_ptr: [nk * 2*hr] f32 — beta/alpha projection output (interleaved)
    ///   conv_state_ptr: [conv_dim, kernel_dim] f32 — modified in-place (shift + append)
    ///   conv_weight_ptr: [conv_dim, kernel_dim] f32 — immutable conv1d weights
    ///   a_log_ptr: [nv] f32 — log decay (immutable)
    ///   dt_bias_ptr: [nv] f32 — dt bias (immutable)
    ///   scale: f32 — query scale factor
    ///
    /// Outputs:
    ///   q_out_ptr: [nv * dk] f32 — L2-normalized, scaled, head-expanded query
    ///   k_out_ptr: [nv * dk] f32 — L2-normalized, head-expanded key
    ///   v_out_ptr: [nv * dv] f32 — value (after conv+SiLU, not normalized)
    ///   z_out_ptr: [nv * dv] f32 — gate signal (un-interleaved, no processing)
    ///   g_out_ptr: [nv] f32 — decay gate (raw, not exp'd)
    ///   beta_out_ptr: [nv] f32 — beta gate (sigmoid applied)
    #[allow(clippy::too_many_arguments)]
    pub fn linear_attention_conv(
        &self,
        qkvz_ptr: usize,
        ba_ptr: usize,
        conv_state_ptr: usize,
        conv_weight_ptr: usize,
        a_log_ptr: usize,
        dt_bias_ptr: usize,
        scale: f32,
        q_out_ptr: usize,
        k_out_ptr: usize,
        v_out_ptr: usize,
        z_out_ptr: usize,
        g_out_ptr: usize,
        beta_out_ptr: usize,
        nk: usize,
        nv: usize,
        dk: usize,
        dv: usize,
        hr: usize,
        kernel_dim: usize,
    ) -> PyResult<()> {
        let conv_dim = nk * dk * 2 + nv * dv;  // q_flat + k_flat + v_flat
        let group_dim = 2 * dk + 2 * dv * hr;

        let qkvz: &[f32] = unsafe {
            std::slice::from_raw_parts(qkvz_ptr as *const f32, nk * group_dim)
        };
        let ba: &[f32] = unsafe {
            std::slice::from_raw_parts(ba_ptr as *const f32, nk * 2 * hr)
        };
        let conv_state: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(conv_state_ptr as *mut f32, conv_dim * kernel_dim)
        };
        let conv_weight: &[f32] = unsafe {
            std::slice::from_raw_parts(conv_weight_ptr as *const f32, conv_dim * kernel_dim)
        };
        let a_log: &[f32] = unsafe {
            std::slice::from_raw_parts(a_log_ptr as *const f32, nv)
        };
        let dt_bias: &[f32] = unsafe {
            std::slice::from_raw_parts(dt_bias_ptr as *const f32, nv)
        };
        let q_out: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(q_out_ptr as *mut f32, nv * dk)
        };
        let k_out: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(k_out_ptr as *mut f32, nv * dk)
        };
        let v_out: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(v_out_ptr as *mut f32, nv * dv)
        };
        let z_out: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(z_out_ptr as *mut f32, nv * dv)
        };
        let g_out: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(g_out_ptr as *mut f32, nv)
        };
        let beta_out: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(beta_out_ptr as *mut f32, nv)
        };

        // Step 1: Un-interleave qkvz [nk, group_dim] into q[nk,dk], k[nk,dk], v[nv,dv], z[nv,dv]
        // and ba [nk, 2*hr] into b[nv], a_param[nv]
        // Layout: per key-head group: [q_dk, k_dk, v_dv*hr, z_dv*hr]
        // mixed_qkv = [q_flat(nk*dk), k_flat(nk*dk), v_flat(nv*dv)]
        let key_dim = nk * dk;
        let mut mixed_qkv = vec![0.0f32; conv_dim];

        // Un-interleave qkvz into mixed_qkv and z_out
        for h in 0..nk {
            let src = h * group_dim;
            // q: mixed_qkv[h*dk .. (h+1)*dk]
            mixed_qkv[h * dk..(h + 1) * dk].copy_from_slice(&qkvz[src..src + dk]);
            // k: mixed_qkv[key_dim + h*dk .. key_dim + (h+1)*dk]
            mixed_qkv[key_dim + h * dk..key_dim + (h + 1) * dk]
                .copy_from_slice(&qkvz[src + dk..src + 2 * dk]);
            // v: goes to mixed_qkv[2*key_dim + h*hr*dv .. ] and z: goes to z_out
            for r in 0..hr {
                let v_head = h * hr + r;
                let v_src = src + 2 * dk + r * dv;
                let z_src = src + 2 * dk + hr * dv + r * dv;
                mixed_qkv[2 * key_dim + v_head * dv..2 * key_dim + (v_head + 1) * dv]
                    .copy_from_slice(&qkvz[v_src..v_src + dv]);
                z_out[v_head * dv..(v_head + 1) * dv]
                    .copy_from_slice(&qkvz[z_src..z_src + dv]);
            }
        }

        // Un-interleave ba into b, a_param
        let mut b_raw = vec![0.0f32; nv];
        let mut a_param = vec![0.0f32; nv];
        for h in 0..nk {
            let src = h * 2 * hr;
            for r in 0..hr {
                b_raw[h * hr + r] = ba[src + r];
                a_param[h * hr + r] = ba[src + hr + r];
            }
        }

        // Step 2: Conv state update — shift left by 1, append mixed_qkv
        // conv_state is [conv_dim, kernel_dim] row-major
        for ch in 0..conv_dim {
            let base = ch * kernel_dim;
            // Shift left by 1
            for t in 0..kernel_dim - 1 {
                conv_state[base + t] = conv_state[base + t + 1];
            }
            // Append new value
            conv_state[base + kernel_dim - 1] = mixed_qkv[ch];
        }

        // Step 3: Depthwise conv1d + SiLU (one dot product per channel)
        let mut conv_out = vec![0.0f32; conv_dim];
        for ch in 0..conv_dim {
            let s_base = ch * kernel_dim;
            let w_base = ch * kernel_dim;
            let mut dot = 0.0f32;
            for t in 0..kernel_dim {
                dot += conv_state[s_base + t] * conv_weight[w_base + t];
            }
            // SiLU
            let sigmoid = 1.0 / (1.0 + (-dot).exp());
            conv_out[ch] = dot * sigmoid;
        }

        // Step 4: Split conv_out back to q_conv[nk,dk], k_conv[nk,dk], v_conv[nv,dv]
        // Then expand key heads (nk→nv) and L2 normalize

        // Expand + normalize q
        for vh in 0..nv {
            let kh = vh / hr;  // source key head
            let src_base = kh * dk;  // q is first key_dim elements
            let dst_base = vh * dk;
            // L2 norm
            let mut sum_sq = 0.0f32;
            for i in 0..dk {
                let val = conv_out[src_base + i];
                sum_sq += val * val;
            }
            let inv_norm = if sum_sq > 0.0 { 1.0 / sum_sq.sqrt() } else { 0.0 };
            for i in 0..dk {
                q_out[dst_base + i] = conv_out[src_base + i] * inv_norm * scale;
            }
        }

        // Expand + normalize k
        for vh in 0..nv {
            let kh = vh / hr;
            let src_base = key_dim + kh * dk;  // k starts at key_dim
            let dst_base = vh * dk;
            let mut sum_sq = 0.0f32;
            for i in 0..dk {
                let val = conv_out[src_base + i];
                sum_sq += val * val;
            }
            let inv_norm = if sum_sq > 0.0 { 1.0 / sum_sq.sqrt() } else { 0.0 };
            for i in 0..dk {
                k_out[dst_base + i] = conv_out[src_base + i] * inv_norm;
            }
        }

        // v: no expansion needed (already nv*dv), no normalization
        v_out.copy_from_slice(&conv_out[2 * key_dim..2 * key_dim + nv * dv]);

        // Step 5: Gate parameters
        for h in 0..nv {
            beta_out[h] = 1.0 / (1.0 + (-b_raw[h]).exp());  // sigmoid(b)
            // g = -exp(A_log) * softplus(a_param + dt_bias)
            let ap_dt = a_param[h] + dt_bias[h];
            let softplus = if ap_dt > 20.0 { ap_dt } else { (1.0 + ap_dt.exp()).ln() };
            g_out[h] = -(a_log[h].exp()) * softplus;
        }

        Ok(())
    }

    /// Store MoE routing weight (float32). Returns route_id.
    ///
    /// Gate weight is [num_experts, hidden_dim] float32.
    /// Optional bias [num_experts] and e_score_correction [num_experts].
    #[pyo3(signature = (data_ptr, num_experts, hidden_dim, bias_ptr=None, bias_len=0, e_score_corr_ptr=None, e_score_corr_len=0))]
    pub fn store_route_weight(
        &mut self,
        data_ptr: usize,
        num_experts: usize,
        hidden_dim: usize,
        bias_ptr: Option<usize>,
        bias_len: usize,
        e_score_corr_ptr: Option<usize>,
        e_score_corr_len: usize,
    ) -> PyResult<usize> {
        let data: &[f32] = unsafe {
            std::slice::from_raw_parts(data_ptr as *const f32, num_experts * hidden_dim)
        };
        let mut rw = RouteWeight {
            data: data.to_vec(),
            bias: None,
            e_score_corr: None,
            num_experts,
            hidden_dim,
        };
        if let Some(bp) = bias_ptr {
            if bias_len > 0 {
                let b: &[f32] = unsafe {
                    std::slice::from_raw_parts(bp as *const f32, bias_len)
                };
                rw.bias = Some(b.to_vec());
            }
        }
        if let Some(ep) = e_score_corr_ptr {
            if e_score_corr_len > 0 {
                let e: &[f32] = unsafe {
                    std::slice::from_raw_parts(ep as *const f32, e_score_corr_len)
                };
                rw.e_score_corr = Some(e.to_vec());
            }
        }
        let id = self.route_weights.len();
        let bytes = num_experts * hidden_dim * 4
            + rw.bias.as_ref().map_or(0, |b| b.len() * 4)
            + rw.e_score_corr.as_ref().map_or(0, |e| e.len() * 4);
        log::debug!("Stored route weight {}: [{}x{}] f32, {:.1} KB",
            id, num_experts, hidden_dim, bytes as f64 / 1024.0);
        self.route_weights.push(rw);
        // Pre-allocate scratch buffers for max expert count seen
        if num_experts > self.route_logits.len() {
            self.route_logits.resize(num_experts, 0.0);
            self.route_scores.resize(num_experts, 0.0);
            self.route_corrected.resize(num_experts, 0.0);
        }
        Ok(id)
    }

    /// MoE routing: AVX2 matmul + scoring + topk, all in Rust.
    ///
    /// hidden_ptr: [hidden_dim] f32
    /// topk_ids_out_ptr: [topk] i32 output
    /// topk_weights_out_ptr: [topk] f32 output
    /// scoring_func: 0=sigmoid, 1=softmax, 2=swiglu (topk-then-softmax)
    /// norm_topk_prob: whether to normalize topk weights
    #[allow(clippy::too_many_arguments)]
    pub fn moe_route(
        &mut self,
        route_id: usize,
        hidden_ptr: usize,
        topk_ids_out_ptr: usize,
        topk_weights_out_ptr: usize,
        topk: usize,
        scoring_func: u8,
        norm_topk_prob: bool,
    ) -> PyResult<()> {
        if route_id >= self.route_weights.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("route_id {} out of range ({})", route_id, self.route_weights.len())));
        }
        let rw = &self.route_weights[route_id];
        let ne = rw.num_experts;
        let hd = rw.hidden_dim;

        let hidden: &[f32] = unsafe {
            std::slice::from_raw_parts(hidden_ptr as *const f32, hd)
        };
        let topk_ids: &mut [i32] = unsafe {
            std::slice::from_raw_parts_mut(topk_ids_out_ptr as *mut i32, topk)
        };
        let topk_weights: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(topk_weights_out_ptr as *mut f32, topk)
        };

        // Use pre-allocated scratch buffers
        let logits = &mut self.route_logits[..ne];
        let scores = &mut self.route_scores[..ne];

        // Step 1: AVX2 matmul — logits[e] = gate_weight[e, :] @ hidden (parallel)
        unsafe { moe_route_matmul_avx2_parallel(&rw.data, hidden, logits, ne, hd) };

        // Add bias if present
        if let Some(ref bias) = rw.bias {
            for e in 0..ne {
                logits[e] += bias[e];
            }
        }

        // Step 2: Scoring + topk
        match scoring_func {
            0 => {
                // sigmoid scoring
                for e in 0..ne {
                    scores[e] = 1.0 / (1.0 + (-logits[e]).exp());
                }

                // topk on (scores + e_score_corr) if present, but weights from raw scores
                if let Some(ref esc) = rw.e_score_corr {
                    let corrected = &mut self.route_corrected[..ne];
                    for e in 0..ne {
                        corrected[e] = scores[e] + esc[e];
                    }
                    topk_indices(corrected, topk, topk_ids);
                } else {
                    topk_indices(scores, topk, topk_ids);
                }
                for i in 0..topk {
                    topk_weights[i] = scores[topk_ids[i] as usize];
                }

                if norm_topk_prob {
                    let sum: f32 = topk_weights[..topk].iter().sum();
                    if sum > 0.0 {
                        for w in topk_weights[..topk].iter_mut() {
                            *w /= sum;
                        }
                    }
                }
            }
            1 => {
                // softmax scoring
                let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for e in 0..ne {
                    scores[e] = (logits[e] - max_logit).exp();
                    sum_exp += scores[e];
                }
                let inv_sum = 1.0 / sum_exp;
                for e in 0..ne {
                    scores[e] *= inv_sum;
                }

                if let Some(ref esc) = rw.e_score_corr {
                    let corrected = &mut self.route_corrected[..ne];
                    for e in 0..ne {
                        corrected[e] = scores[e] + esc[e];
                    }
                    topk_indices(corrected, topk, topk_ids);
                } else {
                    topk_indices(scores, topk, topk_ids);
                }
                for i in 0..topk {
                    topk_weights[i] = scores[topk_ids[i] as usize];
                }

                if norm_topk_prob {
                    let sum: f32 = topk_weights[..topk].iter().sum();
                    if sum > 0.0 {
                        for w in topk_weights[..topk].iter_mut() {
                            *w /= sum;
                        }
                    }
                }
            }
            2 => {
                // swiglu: topk on raw logits, then softmax on topk values
                topk_indices(logits, topk, topk_ids);
                let max_l = (0..topk).map(|i| logits[topk_ids[i] as usize])
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for i in 0..topk {
                    let v = (logits[topk_ids[i] as usize] - max_l).exp();
                    topk_weights[i] = v;
                    sum_exp += v;
                }
                let inv_sum = 1.0 / sum_exp;
                for i in 0..topk {
                    topk_weights[i] *= inv_sum;
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unknown scoring_func: {}", scoring_func)));
            }
        }

        Ok(())
    }

    /// Number of stored weights.
    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }

    /// Total bytes used by stored weights (quantized + routing).
    pub fn total_bytes(&self) -> usize {
        let quant: usize = self.weights.iter().map(|w| {
            w.packed.len() * 4 + w.scales.len() * 2
        }).sum();
        let route: usize = self.route_weights.iter().map(|rw| {
            rw.data.len() * 4
            + rw.bias.as_ref().map_or(0, |b| b.len() * 4)
            + rw.e_score_corr.as_ref().map_or(0, |e| e.len() * 4)
        }).sum();
        quant + route
    }

    /// Bytes used by a single weight matrix.
    pub fn weight_bytes(&self, weight_id: usize) -> usize {
        let w = &self.weights[weight_id];
        w.packed.len() * 4 + w.scales.len() * 2
    }

    /// Number of stored route weights.
    pub fn num_route_weights(&self) -> usize {
        self.route_weights.len()
    }
}

impl Drop for CpuDecodeStore {
    fn drop(&mut self) {
        if !self.mmap_regions.is_empty() {
            // Defuse mmap-backed weight Vecs to prevent dealloc on mmap'd memory
            for w in self.weights.iter_mut() {
                std::mem::forget(std::mem::take(&mut w.packed));
                std::mem::forget(std::mem::take(&mut w.scales));
            }
            // Unmap the contiguous regions
            for &(base_usize, len) in &self.mmap_regions {
                unsafe { libc::munmap(base_usize as *mut libc::c_void, len); }
            }
        }
    }
}

// Private helper (not exposed to Python)
impl CpuDecodeStore {
    /// Dispatch matmul to correct INT4/INT8 kernel (uses provided buffers).
    fn dispatch_matmul_ext(&self, weight_id: usize, act_int16: &[i16], act_scales: &[f32], output: &mut [f32]) {
        self.dispatch_matmul(weight_id, act_int16, act_scales, output);
    }

    /// Dispatch matmul to correct INT4/INT8 kernel.
    fn dispatch_matmul(&self, weight_id: usize, act_int16: &[i16], act_scales: &[f32], output: &mut [f32]) {
        let w = &self.weights[weight_id];
        let k = w.cols;
        let n = w.rows;
        let gs = w.group_size;

        match (w.num_bits, w.tiled) {
            (4, true) => {
                if self.parallel && n > 64 {
                    matmul_int4_transposed_integer_parallel_tiled(
                        &w.packed, &w.scales, act_int16, act_scales, output, k, n, gs);
                } else {
                    matmul_int4_transposed_integer_tiled(
                        &w.packed, &w.scales, act_int16, act_scales, output, k, n, gs);
                }
            }
            (4, false) => {
                if self.parallel && n > 64 {
                    matmul_int4_transposed_integer_parallel(
                        &w.packed, &w.scales, act_int16, act_scales, output, k, n, gs);
                } else {
                    matmul_int4_transposed_integer(
                        &w.packed, &w.scales, act_int16, act_scales, output, k, n, gs);
                }
            }
            (8, true) => {
                if self.parallel && n > 64 {
                    matmul_int8_transposed_integer_parallel_tiled(
                        &w.packed, &w.scales, act_int16, act_scales, output, k, n, gs);
                } else {
                    matmul_int8_transposed_integer_tiled(
                        &w.packed, &w.scales, act_int16, act_scales, output, k, n, gs);
                }
            }
            (8, false) => {
                if self.parallel && n > 64 {
                    matmul_int8_transposed_integer_parallel(
                        &w.packed, &w.scales, act_int16, act_scales, output, k, n, gs);
                } else {
                    matmul_int8_transposed_integer(
                        &w.packed, &w.scales, act_int16, act_scales, output, k, n, gs);
                }
            }
            _ => unreachable!(),
        }
    }
}

/// AVX2 fused add + RMSNorm.
///
/// Vectorized: 8 floats per iteration for sum_sq, residual update, and norm output.
/// ~8x faster than scalar for hidden_size=2048 (256 iterations vs 2048).
///
/// # Safety
/// Requires AVX2 + FMA. Slices must have matching lengths.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_add_rmsnorm_avx2(
    hidden: &mut [f32],
    residual: &mut [f32],
    weight: &[f32],
    eps: f32,
    first_call: bool,
    norm_bias_one: bool,
) {
    use std::arch::x86_64::*;

    let size = hidden.len();
    let n8 = size / 8;
    let n_rem = size % 8;

    // Step 1: residual update (copy or add)
    if first_call {
        for b in 0..n8 {
            let h = _mm256_loadu_ps(hidden.as_ptr().add(b * 8));
            _mm256_storeu_ps(residual.as_mut_ptr().add(b * 8), h);
        }
        for r in 0..n_rem {
            residual[n8 * 8 + r] = hidden[n8 * 8 + r];
        }
    } else {
        for b in 0..n8 {
            let h = _mm256_loadu_ps(hidden.as_ptr().add(b * 8));
            let r = _mm256_loadu_ps(residual.as_ptr().add(b * 8));
            _mm256_storeu_ps(residual.as_mut_ptr().add(b * 8), _mm256_add_ps(h, r));
        }
        for r in 0..n_rem {
            let idx = n8 * 8 + r;
            residual[idx] += hidden[idx];
        }
    }

    // Step 2: sum of squares
    let mut sum_acc = _mm256_setzero_ps();
    for b in 0..n8 {
        let v = _mm256_loadu_ps(residual.as_ptr().add(b * 8));
        sum_acc = _mm256_fmadd_ps(v, v, sum_acc);
    }
    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum_acc, 1);
    let lo = _mm256_castps256_ps128(sum_acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let hi64 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, hi64);
    let mut sum_sq = _mm_cvtss_f32(sum32);
    for r in 0..n_rem {
        let v = residual[n8 * 8 + r];
        sum_sq += v * v;
    }

    let rms = (sum_sq / size as f32 + eps).sqrt().recip();
    let rms_vec = _mm256_set1_ps(rms);

    // Step 3: output = residual * rms * weight
    if norm_bias_one {
        let ones = _mm256_set1_ps(1.0);
        for b in 0..n8 {
            let res = _mm256_loadu_ps(residual.as_ptr().add(b * 8));
            let w = _mm256_loadu_ps(weight.as_ptr().add(b * 8));
            let result = _mm256_mul_ps(_mm256_mul_ps(res, rms_vec), _mm256_add_ps(w, ones));
            _mm256_storeu_ps(hidden.as_mut_ptr().add(b * 8), result);
        }
        for r in 0..n_rem {
            let idx = n8 * 8 + r;
            hidden[idx] = residual[idx] * rms * (1.0 + weight[idx]);
        }
    } else {
        for b in 0..n8 {
            let res = _mm256_loadu_ps(residual.as_ptr().add(b * 8));
            let w = _mm256_loadu_ps(weight.as_ptr().add(b * 8));
            let result = _mm256_mul_ps(_mm256_mul_ps(res, rms_vec), w);
            _mm256_storeu_ps(hidden.as_mut_ptr().add(b * 8), result);
        }
        for r in 0..n_rem {
            let idx = n8 * 8 + r;
            hidden[idx] = residual[idx] * rms * weight[idx];
        }
    }
}

/// AVX2-optimized linear attention recurrent update.
///
/// For each value head h:
///   1. Decay: state[h, :, :] *= exp(g[h])
///   2. kv_mem[dv] = state[h, :, :].T @ k[h, :] (sum over dk)
///   3. delta[dv] = (v[h, :] - kv_mem) * beta[h]
///   4. state[h, :, :] += k[h, :].outer(delta)
///   5. output[h, dv] = state[h, :, :].T @ q[h, :] (sum over dk)
#[target_feature(enable = "avx2,fma")]
unsafe fn linear_attention_recurrent_avx2(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    output: &mut [f32],
    nv: usize,
    dk: usize,
    dv: usize,
) {
    use std::arch::x86_64::*;
    let dv8 = dv / 8;
    // Stack scratch buffers (dv <= 256 typically)
    let mut kv_mem = vec![0.0f32; dv];
    let mut delta = vec![0.0f32; dv];
    let mut out_buf = vec![0.0f32; dv];

    for h in 0..nv {
        let g_exp = g[h].exp();
        let beta_h = beta[h];
        let s_base = h * dk * dv;
        let q_base = h * dk;
        let k_base = h * dk;
        let v_base = h * dv;
        let o_base = h * dv;

        // Zero scratch
        for j in (0..dv).step_by(8) {
            _mm256_storeu_ps(kv_mem.as_mut_ptr().add(j), _mm256_setzero_ps());
            _mm256_storeu_ps(out_buf.as_mut_ptr().add(j), _mm256_setzero_ps());
        }

        let g_exp_v = _mm256_set1_ps(g_exp);

        // Pass 1: Decay state + compute kv_mem (cache-friendly: row-major)
        for i in 0..dk {
            let row_ptr = state.as_mut_ptr().add(s_base + i * dv);
            let k_v = _mm256_set1_ps(k[k_base + i]);
            for j in 0..dv8 {
                let j8 = j * 8;
                let s = _mm256_loadu_ps(row_ptr.add(j8));
                let s_decayed = _mm256_mul_ps(s, g_exp_v);
                _mm256_storeu_ps(row_ptr.add(j8), s_decayed);
                let km = _mm256_loadu_ps(kv_mem.as_ptr().add(j8));
                _mm256_storeu_ps(kv_mem.as_mut_ptr().add(j8),
                    _mm256_fmadd_ps(s_decayed, k_v, km));
            }
        }

        // Compute delta[j] = (v[j] - kv_mem[j]) * beta
        let beta_v = _mm256_set1_ps(beta_h);
        for j in 0..dv8 {
            let j8 = j * 8;
            let vv = _mm256_loadu_ps(v.as_ptr().add(v_base + j8));
            let km = _mm256_loadu_ps(kv_mem.as_ptr().add(j8));
            let d = _mm256_mul_ps(_mm256_sub_ps(vv, km), beta_v);
            _mm256_storeu_ps(delta.as_mut_ptr().add(j8), d);
        }

        // Pass 2: State update + output accumulation (cache-friendly)
        for i in 0..dk {
            let row_ptr = state.as_mut_ptr().add(s_base + i * dv);
            let k_v = _mm256_set1_ps(k[k_base + i]);
            let q_v = _mm256_set1_ps(q[q_base + i]);
            for j in 0..dv8 {
                let j8 = j * 8;
                let s = _mm256_loadu_ps(row_ptr.add(j8));
                let d = _mm256_loadu_ps(delta.as_ptr().add(j8));
                let s_new = _mm256_fmadd_ps(k_v, d, s);
                _mm256_storeu_ps(row_ptr.add(j8), s_new);
                let ob = _mm256_loadu_ps(out_buf.as_ptr().add(j8));
                _mm256_storeu_ps(out_buf.as_mut_ptr().add(j8),
                    _mm256_fmadd_ps(s_new, q_v, ob));
            }
        }

        // Write output
        for j in 0..dv8 {
            let j8 = j * 8;
            _mm256_storeu_ps(output.as_mut_ptr().add(o_base + j8),
                _mm256_loadu_ps(out_buf.as_ptr().add(j8)));
        }
    }
}

/// AVX2-optimized matmul for MoE routing: logits[e] = gate[e, :] @ hidden
///
/// gate: [ne * hd] f32 row-major, hidden: [hd] f32, logits: [ne] f32 output.
/// hd must be divisible by 8 (guaranteed for hidden_dim=2048).
#[target_feature(enable = "avx2,fma")]
unsafe fn moe_route_matmul_avx2(
    gate: &[f32],
    hidden: &[f32],
    logits: &mut [f32],
    ne: usize,
    hd: usize,
) {
    use std::arch::x86_64::*;
    let chunks = hd / 8;
    let hidden_ptr = hidden.as_ptr();
    let gate_ptr = gate.as_ptr();
    for e in 0..ne {
        let row = gate_ptr.add(e * hd);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        // Unroll by 2 to hide FMA latency
        let chunks2 = chunks / 2;
        let mut i = 0usize;
        for _ in 0..chunks2 {
            let h0 = _mm256_loadu_ps(hidden_ptr.add(i));
            let g0 = _mm256_loadu_ps(row.add(i));
            acc0 = _mm256_fmadd_ps(g0, h0, acc0);
            let h1 = _mm256_loadu_ps(hidden_ptr.add(i + 8));
            let g1 = _mm256_loadu_ps(row.add(i + 8));
            acc1 = _mm256_fmadd_ps(g1, h1, acc1);
            i += 16;
        }
        // Handle odd chunk
        if chunks % 2 != 0 {
            let h0 = _mm256_loadu_ps(hidden_ptr.add(i));
            let g0 = _mm256_loadu_ps(row.add(i));
            acc0 = _mm256_fmadd_ps(g0, h0, acc0);
        }
        // Horizontal sum of acc0 + acc1
        let sum8 = _mm256_add_ps(acc0, acc1);
        let hi128 = _mm256_extractf128_ps(sum8, 1);
        let lo128 = _mm256_castps256_ps128(sum8);
        let sum4 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum4);
        let sum2 = _mm_add_ps(sum4, shuf);
        let shuf2 = _mm_movehl_ps(sum2, sum2);
        let sum1 = _mm_add_ss(sum2, shuf2);
        logits[e] = _mm_cvtss_f32(sum1);
    }
}

/// Parallel AVX2-optimized matmul for MoE routing.
///
/// Splits ne experts across rayon threads. Each thread computes a chunk of
/// dot products independently. For 512 experts @ hd=2048, this is 4MB of
/// gate data that benefits from parallel DRAM channel access.
unsafe fn moe_route_matmul_avx2_parallel(
    gate: &[f32],
    hidden: &[f32],
    logits: &mut [f32],
    ne: usize,
    hd: usize,
) {
    use rayon::prelude::*;
    // SAFETY: Each rayon thread writes to a disjoint slice of logits[].
    // gate[] and hidden[] are read-only.
    let gate_addr = gate.as_ptr() as usize;
    let hidden_addr = hidden.as_ptr() as usize;
    let logits_addr = logits.as_mut_ptr() as usize;

    (0..ne).into_par_iter().for_each(|e| {
        #[target_feature(enable = "avx2,fma")]
        unsafe fn dot_avx2(gate_row: *const f32, hidden_ptr: *const f32, hd: usize) -> f32 {
            use std::arch::x86_64::*;
            let chunks = hd / 8;
            let chunks2 = chunks / 2;
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0usize;
            for _ in 0..chunks2 {
                let h0 = _mm256_loadu_ps(hidden_ptr.add(i));
                let g0 = _mm256_loadu_ps(gate_row.add(i));
                acc0 = _mm256_fmadd_ps(g0, h0, acc0);
                let h1 = _mm256_loadu_ps(hidden_ptr.add(i + 8));
                let g1 = _mm256_loadu_ps(gate_row.add(i + 8));
                acc1 = _mm256_fmadd_ps(g1, h1, acc1);
                i += 16;
            }
            if chunks % 2 != 0 {
                let h0 = _mm256_loadu_ps(hidden_ptr.add(i));
                let g0 = _mm256_loadu_ps(gate_row.add(i));
                acc0 = _mm256_fmadd_ps(g0, h0, acc0);
            }
            let sum8 = _mm256_add_ps(acc0, acc1);
            let hi128 = _mm256_extractf128_ps(sum8, 1);
            let lo128 = _mm256_castps256_ps128(sum8);
            let sum4 = _mm_add_ps(lo128, hi128);
            let shuf = _mm_movehdup_ps(sum4);
            let sum2 = _mm_add_ps(sum4, shuf);
            let shuf2 = _mm_movehl_ps(sum2, sum2);
            let sum1 = _mm_add_ss(sum2, shuf2);
            _mm_cvtss_f32(sum1)
        }
        unsafe {
            let gate_ptr = gate_addr as *const f32;
            let hidden_ptr = hidden_addr as *const f32;
            let logits_ptr = logits_addr as *mut f32;
            let row = gate_ptr.add(e * hd);
            *logits_ptr.add(e) = dot_avx2(row, hidden_ptr, hd);
        }
    });
}

/// Find indices of top-k largest values via partial selection sort.
/// For small k (e.g., 10) and moderate n (e.g., 512), this is faster than full sort.
fn topk_indices(values: &[f32], k: usize, out: &mut [i32]) {
    let n = values.len();
    assert!(k <= n);
    assert!(out.len() >= k);

    // Initialize with first k indices
    let mut heap: Vec<(f32, usize)> = (0..k).map(|i| (values[i], i)).collect();
    // Build min-heap by value (we want to evict the smallest of our top-k candidates)
    heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Scan remaining elements
    for i in k..n {
        if values[i] > heap[0].0 {
            heap[0] = (values[i], i);
            // Sift down to restore min-heap
            let mut pos = 0;
            loop {
                let left = 2 * pos + 1;
                let right = 2 * pos + 2;
                let mut smallest = pos;
                if left < k && heap[left].0 < heap[smallest].0 {
                    smallest = left;
                }
                if right < k && heap[right].0 < heap[smallest].0 {
                    smallest = right;
                }
                if smallest == pos {
                    break;
                }
                heap.swap(pos, smallest);
                pos = smallest;
            }
        }
    }

    // Sort by value descending (to match torch.topk ordering)
    heap.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for i in 0..k {
        out[i] = heap[i].1 as i32;
    }
}

// ════════════════════════════════════════════════════════════════════
// FULL DECODE GRAPH — single-call decode_step replaces Python loop
// ════════════════════════════════════════════════════════════════════

/// Free-function dispatch for quantized matmul (avoids &self borrow conflict).
fn dispatch_matmul_free(
    w: &TransposedWeight,
    act_int16: &[i16],
    act_scales: &[f32],
    output: &mut [f32],
    parallel: bool,
) {
    match (w.num_bits, w.tiled) {
        (4, true) => {
            if parallel && w.rows > 64 {
                matmul_int4_transposed_integer_parallel_tiled(
                    &w.packed, &w.scales, act_int16, act_scales, output,
                    w.cols, w.rows, w.group_size);
            } else {
                matmul_int4_transposed_integer_tiled(
                    &w.packed, &w.scales, act_int16, act_scales, output,
                    w.cols, w.rows, w.group_size);
            }
        }
        (4, false) => {
            if parallel && w.rows > 64 {
                matmul_int4_transposed_integer_parallel(
                    &w.packed, &w.scales, act_int16, act_scales, output,
                    w.cols, w.rows, w.group_size);
            } else {
                matmul_int4_transposed_integer(
                    &w.packed, &w.scales, act_int16, act_scales, output,
                    w.cols, w.rows, w.group_size);
            }
        }
        (8, true) => {
            if parallel && w.rows > 64 {
                matmul_int8_transposed_integer_parallel_tiled(
                    &w.packed, &w.scales, act_int16, act_scales, output,
                    w.cols, w.rows, w.group_size);
            } else {
                matmul_int8_transposed_integer_tiled(
                    &w.packed, &w.scales, act_int16, act_scales, output,
                    w.cols, w.rows, w.group_size);
            }
        }
        (8, false) => {
            if parallel && w.rows > 64 {
                matmul_int8_transposed_integer_parallel(
                    &w.packed, &w.scales, act_int16, act_scales, output,
                    w.cols, w.rows, w.group_size);
            } else {
                matmul_int8_transposed_integer(
                    &w.packed, &w.scales, act_int16, act_scales, output,
                    w.cols, w.rows, w.group_size);
            }
        }
        _ => unreachable!(),
    }
}

/// Prefetch a TransposedWeight's packed data and scales into L3 cache using NTA hints.
/// Call this before a compute-bound phase so the weight data is warm when the matmul starts.
#[cfg(target_arch = "x86_64")]
#[inline]
fn prefetch_weight_nta(w: &TransposedWeight) {
    use std::arch::x86_64::{_mm_prefetch, _MM_HINT_NTA};
    const STRIDE: usize = 512; // 8 cache lines per prefetch
    let packed_bytes = w.packed.len() * 4;
    let scales_bytes = w.scales.len() * 2;
    unsafe {
        let p = w.packed.as_ptr() as *const i8;
        let mut off = 0;
        while off < packed_bytes {
            _mm_prefetch(p.add(off), _MM_HINT_NTA);
            off += STRIDE;
        }
        let s = w.scales.as_ptr() as *const i8;
        off = 0;
        while off < scales_bytes {
            _mm_prefetch(s.add(off), _MM_HINT_NTA);
            off += STRIDE;
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn prefetch_weight_nta(_w: &TransposedWeight) {}

/// BF16 → f32 conversion.
#[inline]
fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// AVX2 fast SiLU: output[i] = x[i] * sigmoid(x[i]) for n elements.
///
/// Uses polynomial sigmoid approximation (same as silu_quantize_int16_avx2).
/// Replaces scalar exp() which costs ~50ns each.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fast_silu_avx2(input: &mut [f32], n: usize) {
    use std::arch::x86_64::*;

    // Fast exp AVX2 (copy from kernel/avx2.rs inline)
    #[inline(always)]
    unsafe fn fast_exp_avx2_inline(x: __m256) -> __m256 {
        let log2e = _mm256_set1_ps(1.4426950408889634);
        let t = _mm256_mul_ps(x, log2e);
        let n = _mm256_floor_ps(t);
        let ni = _mm256_cvtps_epi32(n);
        let f = _mm256_sub_ps(t, n);
        let c5 = _mm256_set1_ps(0.0013333558);
        let c4 = _mm256_set1_ps(0.009618129);
        let c3 = _mm256_set1_ps(0.0555041);
        let c2 = _mm256_set1_ps(0.2402265);
        let c1 = _mm256_set1_ps(0.6931472);
        let one = _mm256_set1_ps(1.0);
        let poly = _mm256_fmadd_ps(c5, f, c4);
        let poly = _mm256_fmadd_ps(poly, f, c3);
        let poly = _mm256_fmadd_ps(poly, f, c2);
        let poly = _mm256_fmadd_ps(poly, f, c1);
        let poly = _mm256_fmadd_ps(poly, f, one);
        let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23));
        _mm256_mul_ps(poly, pow2n)
    }

    let n8 = n / 8;
    let ptr = input.as_mut_ptr();
    for i in 0..n8 {
        let x = _mm256_loadu_ps(ptr.add(i * 8));
        let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        let clamped = _mm256_max_ps(
            _mm256_min_ps(neg_x, _mm256_set1_ps(20.0)),
            _mm256_set1_ps(-20.0));
        let exp_neg_x = fast_exp_avx2_inline(clamped);
        let denom = _mm256_add_ps(_mm256_set1_ps(1.0), exp_neg_x);
        let rcp = _mm256_rcp_ps(denom);
        let two = _mm256_set1_ps(2.0);
        let sigmoid = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(denom, rcp, two));
        let silu = _mm256_mul_ps(x, sigmoid);
        _mm256_storeu_ps(ptr.add(i * 8), silu);
    }
    // Scalar remainder
    for i in (n8 * 8)..n {
        let x = input[i];
        let s = 1.0 / (1.0 + (-x).exp());
        input[i] = x * s;
    }
}

/// AVX2 fast SiLU + multiply: output[i] = SiLU(gate[i]) * up[i].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fast_silu_mul_avx2(gate: &[f32], up: &[f32], output: &mut [f32], n: usize) {
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn fast_exp_avx2_inline(x: __m256) -> __m256 {
        let log2e = _mm256_set1_ps(1.4426950408889634);
        let t = _mm256_mul_ps(x, log2e);
        let n = _mm256_floor_ps(t);
        let ni = _mm256_cvtps_epi32(n);
        let f = _mm256_sub_ps(t, n);
        let c5 = _mm256_set1_ps(0.0013333558);
        let c4 = _mm256_set1_ps(0.009618129);
        let c3 = _mm256_set1_ps(0.0555041);
        let c2 = _mm256_set1_ps(0.2402265);
        let c1 = _mm256_set1_ps(0.6931472);
        let one = _mm256_set1_ps(1.0);
        let poly = _mm256_fmadd_ps(c5, f, c4);
        let poly = _mm256_fmadd_ps(poly, f, c3);
        let poly = _mm256_fmadd_ps(poly, f, c2);
        let poly = _mm256_fmadd_ps(poly, f, c1);
        let poly = _mm256_fmadd_ps(poly, f, one);
        let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23));
        _mm256_mul_ps(poly, pow2n)
    }

    let n8 = n / 8;
    for i in 0..n8 {
        let g = _mm256_loadu_ps(gate.as_ptr().add(i * 8));
        let u = _mm256_loadu_ps(up.as_ptr().add(i * 8));
        let neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
        let clamped = _mm256_max_ps(
            _mm256_min_ps(neg_g, _mm256_set1_ps(20.0)),
            _mm256_set1_ps(-20.0));
        let exp_neg_g = fast_exp_avx2_inline(clamped);
        let denom = _mm256_add_ps(_mm256_set1_ps(1.0), exp_neg_g);
        let rcp = _mm256_rcp_ps(denom);
        let two = _mm256_set1_ps(2.0);
        let sigmoid = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(denom, rcp, two));
        let silu = _mm256_mul_ps(g, sigmoid);
        let result = _mm256_mul_ps(silu, u);
        _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), result);
    }
    for i in (n8 * 8)..n {
        let x = gate[i];
        let s = 1.0 / (1.0 + (-x).exp());
        output[i] = x * s * up[i];
    }
}

/// Per-layer attention configuration.
enum DecodeAttnConfig {
    LinearAttention {
        in_proj_qkvz_wid: usize,
        in_proj_ba_wid: usize,
        out_proj_wid: usize,
        conv_weight: Vec<f32>,    // [conv_dim * kernel_dim]
        a_log: Vec<f32>,          // [nv]
        dt_bias: Vec<f32>,        // [nv]
        norm_weight: Vec<f32>,    // [nv * dv] (expanded)
        nk: usize, nv: usize, dk: usize, dv: usize, hr: usize,
        kernel_dim: usize, conv_dim: usize,
        scale: f32,
    },
    GQA {
        q_proj_wid: usize,
        k_proj_wid: usize,
        v_proj_wid: usize,
        o_proj_wid: usize,
        q_norm: Option<Vec<f32>>,
        k_norm: Option<Vec<f32>>,
        gated: bool,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sm_scale: f32,
        fused_qkv_wid: Option<usize>,  // fused Q+K+V weight for single dispatch
    },
    /// Multi-head Latent Attention (DeepSeek V2/V3, Kimi K2.5).
    /// KV is compressed into a low-rank latent vector + rope embedding.
    MLA {
        // Quantized projection weights (stored as TransposedWeight IDs)
        kv_a_proj_wid: usize,  // [kv_lora_rank + rope_dim, hidden_size]
        o_proj_wid: usize,     // [hidden_size, num_heads * v_head_dim]
        // Q path: either direct or LoRA
        q_proj_wid: Option<usize>,      // [num_heads * head_dim, hidden_size]
        q_a_proj_wid: Option<usize>,    // [q_lora_rank, hidden_size]
        q_b_proj_wid: Option<usize>,    // [num_heads * head_dim, q_lora_rank]
        // BF16 per-head projection matrices (stored as f32 for compute)
        w_kc: Vec<f32>,         // [num_heads, qk_nope_dim, kv_lora_rank]
        w_vc: Vec<f32>,         // [num_heads, v_head_dim, kv_lora_rank]
        // Norm weights
        kv_a_norm: Vec<f32>,    // [kv_lora_rank]
        q_a_norm: Option<Vec<f32>>,  // [q_lora_rank] (only if LoRA)
        // MLA-specific RoPE (YaRN) — separate from GQA rope
        rope_cos: Vec<f32>,     // [max_seq, rope_dim/2]
        rope_sin: Vec<f32>,     // [max_seq, rope_dim/2]
        // Dimensions
        num_heads: usize,
        kv_lora_rank: usize,
        qk_nope_dim: usize,
        qk_rope_dim: usize,
        v_head_dim: usize,
        sm_scale: f32,
    },
}

/// Per-layer MLP configuration.
enum DecodeMlpConfig {
    MoE {
        route_id: usize,
        moe_layer_idx: usize,
        shared_gate_up_wid: Option<usize>,
        shared_down_wid: Option<usize>,
        shared_gate_wid: Option<usize>,  // shared_expert_gate
    },
    Dense {
        gate_proj_wid: usize,
        up_proj_wid: usize,
        down_proj_wid: usize,
    },
    None,
}

/// Per-layer config.
struct DecodeLayer {
    input_norm_id: usize,
    post_attn_norm_id: usize,
    attn: DecodeAttnConfig,
    mlp: DecodeMlpConfig,
}

/// Full decode graph — owns all config + scratch for single-call decode.
struct DecodeGraph {
    // Global
    hidden_size: usize,
    eps: f32,
    final_norm_id: usize,
    lm_head_wid: usize,
    vocab_size: usize,
    routed_scaling_factor: f32,
    scoring_func: u8,
    topk: usize,
    norm_topk_prob: bool,
    parallel: bool,

    // Layers
    layers: Vec<DecodeLayer>,

    // Embedding
    embedding_ptr: usize,

    // RoPE (GQA)
    rope_cos_ptr: usize,
    rope_sin_ptr: usize,
    rope_half_dim: usize,
    max_rope_seq: usize,

    // Per-request state (updated by set_decode_state)
    seq_len: usize,
    kv_max_seq: usize,
    kv_k_ptrs: Vec<usize>,
    kv_v_ptrs: Vec<usize>,
    conv_state_ptrs: Vec<usize>,
    recur_state_ptrs: Vec<usize>,

    // Main buffers
    hidden: Vec<f32>,
    residual: Vec<f32>,

    // LA scratch
    la_qkvz_buf: Vec<f32>,
    la_ba_buf: Vec<f32>,
    la_q_buf: Vec<f32>,
    la_k_buf: Vec<f32>,
    la_v_buf: Vec<f32>,
    la_z_buf: Vec<f32>,
    la_g_buf: Vec<f32>,
    la_beta_buf: Vec<f32>,
    la_recur_out: Vec<f32>,
    la_gated_out: Vec<f32>,
    la_mixed_qkv: Vec<f32>,  // scratch for decode_la_conv (avoids heap alloc per call)
    la_conv_out: Vec<f32>,   // scratch for decode_la_conv

    // GQA scratch
    gqa_q_buf: Vec<f32>,
    gqa_k_buf: Vec<f32>,
    gqa_v_buf: Vec<f32>,
    gqa_qkv_buf: Vec<f32>,  // fused Q+K+V output buffer
    gqa_scores: Vec<f32>,
    gqa_attn_out: Vec<f32>,

    // MLA scratch
    mla_kv_out: Vec<f32>,           // kv_a_proj output [kv_lora_rank + rope_dim]
    mla_kv_compressed: Vec<f32>,    // normed [kv_lora_rank]
    mla_q_full: Vec<f32>,           // [num_heads * head_dim]
    mla_q_compressed: Vec<f32>,     // q_a_proj output [q_lora_rank] (LoRA path)
    mla_q_absorbed: Vec<f32>,       // [num_heads * kv_lora_rank] after w_kc absorption
    mla_attn_scores: Vec<f32>,      // [num_heads * kv_max_seq]
    mla_attn_out: Vec<f32>,         // [num_heads * kv_lora_rank]
    mla_v_projected: Vec<f32>,      // [num_heads * v_head_dim]
    // MLA KV cache: per-layer pointers to flat [max_seq, dim] f32 arrays
    mla_ckv_ptrs: Vec<usize>,       // compressed KV [max_seq * kv_lora_rank] per layer
    mla_kpe_ptrs: Vec<usize>,       // rope K position [max_seq * rope_dim] per layer

    // MLP scratch
    mlp_gate_up: Vec<f32>,
    mlp_hidden_buf: Vec<f32>,

    // MoE integration
    moe_store: Option<Arc<WeightStore>>,
    moe_scratch: Option<ExpertScratch>,
    moe_scratch_pool: Vec<ExpertScratch>,
    moe_output: Vec<f32>,
    moe_act_bf16: Vec<u16>,
    shared_out: Vec<f32>,
    moe_topk_ids: Vec<i32>,
    moe_topk_weights: Vec<f32>,
    moe_parallel: bool,

    // MoE routing scratch (pre-allocated, sized by max_experts)
    route_logits: Vec<f32>,
    route_scores: Vec<f32>,
    route_corrected: Vec<f32>,

    // Quantization scratch (separate from CpuDecodeStore's)
    act_int16: Vec<i16>,
    act_scales: Vec<f32>,
    group_size: usize,

    // Timing (enabled by KRASIS_DECODE_TIMING=1)
    timing_enabled: bool,
    timing_step_count: u64,
    timing_report_interval: u64,
    t_norm: f64,
    t_la_proj: f64,
    t_la_conv: f64,
    t_la_recur: f64,
    t_la_gate_norm: f64,
    t_la_out_proj: f64,
    t_gqa_proj: f64,
    t_gqa_rope: f64,
    t_gqa_attn: f64,
    t_gqa_o_proj: f64,
    t_mla_proj: f64,
    t_mla_rope: f64,
    t_mla_attn: f64,
    t_mla_o_proj: f64,
    t_moe_route: f64,
    t_moe_experts: f64,
    t_moe_shared: f64,
    t_dense_mlp: f64,
    t_lm_head: f64,
    t_total: f64,
}

// ── Configuration methods ─────────────────────────────────────────

#[pymethods]
impl CpuDecodeStore {
    /// Initialize the full decode graph with global model config.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (hidden_size, num_layers, eps, final_norm_id, lm_head_wid, vocab_size, topk, scoring_func, norm_topk_prob, routed_scaling_factor, embedding_ptr))]
    pub fn configure_decode(
        &mut self,
        hidden_size: usize,
        num_layers: usize,
        eps: f32,
        final_norm_id: usize,
        lm_head_wid: usize,
        vocab_size: usize,
        topk: usize,
        scoring_func: u8,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        embedding_ptr: usize,
    ) -> PyResult<()> {
        let gs = self.group_size;
        self.decode_graph = Some(Box::new(DecodeGraph {
            hidden_size,
            eps,
            final_norm_id,
            lm_head_wid,
            vocab_size,
            routed_scaling_factor,
            scoring_func,
            topk,
            norm_topk_prob,
            parallel: self.parallel,
            layers: Vec::with_capacity(num_layers),
            embedding_ptr,
            rope_cos_ptr: 0, rope_sin_ptr: 0, rope_half_dim: 0, max_rope_seq: 0,
            seq_len: 0, kv_max_seq: 0,
            kv_k_ptrs: vec![0; num_layers],
            kv_v_ptrs: vec![0; num_layers],
            conv_state_ptrs: vec![0; num_layers],
            recur_state_ptrs: vec![0; num_layers],
            hidden: vec![0.0; hidden_size],
            residual: vec![0.0; hidden_size],
            // Scratch — sized during finalize
            la_qkvz_buf: Vec::new(), la_ba_buf: Vec::new(),
            la_q_buf: Vec::new(), la_k_buf: Vec::new(),
            la_v_buf: Vec::new(), la_z_buf: Vec::new(),
            la_g_buf: Vec::new(), la_beta_buf: Vec::new(),
            la_recur_out: Vec::new(), la_gated_out: Vec::new(),
            la_mixed_qkv: Vec::new(), la_conv_out: Vec::new(),
            gqa_q_buf: Vec::new(), gqa_k_buf: Vec::new(), gqa_v_buf: Vec::new(),
            gqa_qkv_buf: Vec::new(),
            gqa_scores: Vec::new(), gqa_attn_out: Vec::new(),
            mla_kv_out: Vec::new(), mla_kv_compressed: Vec::new(),
            mla_q_full: Vec::new(), mla_q_compressed: Vec::new(),
            mla_q_absorbed: Vec::new(), mla_attn_scores: Vec::new(),
            mla_attn_out: Vec::new(), mla_v_projected: Vec::new(),
            mla_ckv_ptrs: vec![0; num_layers], mla_kpe_ptrs: vec![0; num_layers],
            mlp_gate_up: Vec::new(), mlp_hidden_buf: Vec::new(),
            moe_store: None, moe_scratch: None, moe_scratch_pool: Vec::new(),
            moe_output: vec![0.0; hidden_size],
            moe_act_bf16: vec![0u16; hidden_size],
            shared_out: vec![0.0; hidden_size],
            moe_topk_ids: vec![0i32; topk.max(1)],
            moe_topk_weights: vec![0.0f32; topk.max(1)],
            moe_parallel: true,
            route_logits: Vec::new(), route_scores: Vec::new(), route_corrected: Vec::new(),
            act_int16: Vec::new(), act_scales: Vec::new(),
            group_size: gs,
            timing_enabled: std::env::var("KRASIS_CPU_DECODE_TIMING").map(|v| v == "1").unwrap_or(false),
            timing_step_count: 0,
            timing_report_interval: std::env::var("KRASIS_TIMING_INTERVAL")
                .ok().and_then(|v| v.parse().ok()).unwrap_or(20),
            t_norm: 0.0, t_la_proj: 0.0, t_la_conv: 0.0, t_la_recur: 0.0,
            t_la_gate_norm: 0.0, t_la_out_proj: 0.0,
            t_gqa_proj: 0.0, t_gqa_rope: 0.0, t_gqa_attn: 0.0, t_gqa_o_proj: 0.0,
            t_mla_proj: 0.0, t_mla_rope: 0.0, t_mla_attn: 0.0, t_mla_o_proj: 0.0,
            t_moe_route: 0.0, t_moe_experts: 0.0, t_moe_shared: 0.0,
            t_dense_mlp: 0.0, t_lm_head: 0.0, t_total: 0.0,
        }));
        log::info!("DecodeGraph configured: hidden={}, layers={}, vocab={}, topk={}",
            hidden_size, num_layers, vocab_size, topk);
        Ok(())
    }

    /// Add a linear attention layer to the decode graph.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (input_norm_id, post_attn_norm_id, in_proj_qkvz_wid, in_proj_ba_wid, out_proj_wid, conv_weight_ptr, a_log_ptr, dt_bias_ptr, norm_weight_ptr, nk, nv, dk, dv, hr, kernel_dim, scale))]
    pub fn add_decode_la_layer(
        &mut self,
        input_norm_id: usize,
        post_attn_norm_id: usize,
        in_proj_qkvz_wid: usize,
        in_proj_ba_wid: usize,
        out_proj_wid: usize,
        conv_weight_ptr: usize,
        a_log_ptr: usize,
        dt_bias_ptr: usize,
        norm_weight_ptr: usize,
        nk: usize, nv: usize, dk: usize, dv: usize, hr: usize,
        kernel_dim: usize,
        scale: f32,
    ) -> PyResult<()> {
        let g = self.decode_graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure_decode first"))?;
        let conv_dim = nk * dk * 2 + nv * dv;
        let conv_weight: Vec<f32> = unsafe {
            std::slice::from_raw_parts(conv_weight_ptr as *const f32, conv_dim * kernel_dim).to_vec()
        };
        let a_log: Vec<f32> = unsafe {
            std::slice::from_raw_parts(a_log_ptr as *const f32, nv).to_vec()
        };
        let dt_bias: Vec<f32> = unsafe {
            std::slice::from_raw_parts(dt_bias_ptr as *const f32, nv).to_vec()
        };
        let norm_weight: Vec<f32> = unsafe {
            std::slice::from_raw_parts(norm_weight_ptr as *const f32, nv * dv).to_vec()
        };
        g.layers.push(DecodeLayer {
            input_norm_id,
            post_attn_norm_id,
            attn: DecodeAttnConfig::LinearAttention {
                in_proj_qkvz_wid, in_proj_ba_wid, out_proj_wid,
                conv_weight, a_log, dt_bias, norm_weight,
                nk, nv, dk, dv, hr, kernel_dim, conv_dim, scale,
            },
            mlp: DecodeMlpConfig::None,
        });
        Ok(())
    }

    /// Add a GQA attention layer to the decode graph.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (input_norm_id, post_attn_norm_id, q_proj_wid, k_proj_wid, v_proj_wid, o_proj_wid, q_norm_ptr, q_norm_len, k_norm_ptr, k_norm_len, gated, num_heads, num_kv_heads, head_dim, sm_scale))]
    pub fn add_decode_gqa_layer(
        &mut self,
        input_norm_id: usize,
        post_attn_norm_id: usize,
        q_proj_wid: usize,
        k_proj_wid: usize,
        v_proj_wid: usize,
        o_proj_wid: usize,
        q_norm_ptr: usize,
        q_norm_len: usize,
        k_norm_ptr: usize,
        k_norm_len: usize,
        gated: bool,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sm_scale: f32,
    ) -> PyResult<()> {
        let g = self.decode_graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure_decode first"))?;
        let q_norm = if q_norm_len > 0 {
            Some(unsafe { std::slice::from_raw_parts(q_norm_ptr as *const f32, q_norm_len).to_vec() })
        } else { None };
        let k_norm = if k_norm_len > 0 {
            Some(unsafe { std::slice::from_raw_parts(k_norm_ptr as *const f32, k_norm_len).to_vec() })
        } else { None };
        g.layers.push(DecodeLayer {
            input_norm_id,
            post_attn_norm_id,
            attn: DecodeAttnConfig::GQA {
                q_proj_wid, k_proj_wid, v_proj_wid, o_proj_wid,
                q_norm, k_norm, gated, num_heads, num_kv_heads, head_dim, sm_scale,
                fused_qkv_wid: None,
            },
            mlp: DecodeMlpConfig::None,
        });
        Ok(())
    }

    /// Add an MLA (Multi-head Latent Attention) layer to the decode graph.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (input_norm_id, post_attn_norm_id, kv_a_proj_wid, o_proj_wid,
        q_proj_wid, q_a_proj_wid, q_b_proj_wid,
        w_kc_ptr, w_kc_len, w_vc_ptr, w_vc_len,
        kv_a_norm_ptr, kv_a_norm_len,
        q_a_norm_ptr, q_a_norm_len,
        rope_cos_ptr, rope_sin_ptr, rope_len, rope_max_seq,
        num_heads, kv_lora_rank, qk_nope_dim, qk_rope_dim, v_head_dim, sm_scale))]
    pub fn add_decode_mla_layer(
        &mut self,
        input_norm_id: usize,
        post_attn_norm_id: usize,
        kv_a_proj_wid: usize,
        o_proj_wid: usize,
        q_proj_wid: Option<usize>,
        q_a_proj_wid: Option<usize>,
        q_b_proj_wid: Option<usize>,
        w_kc_ptr: usize, w_kc_len: usize,
        w_vc_ptr: usize, w_vc_len: usize,
        kv_a_norm_ptr: usize, kv_a_norm_len: usize,
        q_a_norm_ptr: usize, q_a_norm_len: usize,
        rope_cos_ptr: usize, rope_sin_ptr: usize,
        rope_len: usize, rope_max_seq: usize,
        num_heads: usize,
        kv_lora_rank: usize,
        qk_nope_dim: usize,
        qk_rope_dim: usize,
        v_head_dim: usize,
        sm_scale: f32,
    ) -> PyResult<()> {
        let g = self.decode_graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure_decode first"))?;

        // Copy BF16 w_kc/w_vc to f32
        let w_kc_bf16 = unsafe { std::slice::from_raw_parts(w_kc_ptr as *const u16, w_kc_len) };
        let w_kc: Vec<f32> = w_kc_bf16.iter().map(|&b| crate::weights::marlin::bf16_to_f32(b)).collect();
        let w_vc_bf16 = unsafe { std::slice::from_raw_parts(w_vc_ptr as *const u16, w_vc_len) };
        let w_vc: Vec<f32> = w_vc_bf16.iter().map(|&b| crate::weights::marlin::bf16_to_f32(b)).collect();

        // Copy norm weights
        let kv_a_norm = unsafe {
            std::slice::from_raw_parts(kv_a_norm_ptr as *const f32, kv_a_norm_len).to_vec()
        };
        let q_a_norm = if q_a_norm_len > 0 {
            Some(unsafe { std::slice::from_raw_parts(q_a_norm_ptr as *const f32, q_a_norm_len).to_vec() })
        } else { None };

        // Copy RoPE tables (already f32 from Python)
        let rope_half = qk_rope_dim / 2;
        let rope_cos = unsafe {
            std::slice::from_raw_parts(rope_cos_ptr as *const f32, rope_max_seq * rope_half).to_vec()
        };
        let rope_sin = unsafe {
            std::slice::from_raw_parts(rope_sin_ptr as *const f32, rope_max_seq * rope_half).to_vec()
        };

        g.layers.push(DecodeLayer {
            input_norm_id,
            post_attn_norm_id,
            attn: DecodeAttnConfig::MLA {
                kv_a_proj_wid, o_proj_wid,
                q_proj_wid, q_a_proj_wid, q_b_proj_wid,
                w_kc, w_vc, kv_a_norm, q_a_norm,
                rope_cos, rope_sin,
                num_heads, kv_lora_rank, qk_nope_dim, qk_rope_dim, v_head_dim, sm_scale,
            },
            mlp: DecodeMlpConfig::None,
        });
        Ok(())
    }

    /// Set MoE config for a layer (call after add_decode_*_layer).
    #[pyo3(signature = (layer_idx, route_id, moe_layer_idx, shared_gate_up_wid=None, shared_down_wid=None, shared_gate_wid=None))]
    pub fn set_decode_layer_moe(
        &mut self,
        layer_idx: usize,
        route_id: usize,
        moe_layer_idx: usize,
        shared_gate_up_wid: Option<usize>,
        shared_down_wid: Option<usize>,
        shared_gate_wid: Option<usize>,
    ) -> PyResult<()> {
        let g = self.decode_graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure_decode first"))?;
        g.layers[layer_idx].mlp = DecodeMlpConfig::MoE {
            route_id, moe_layer_idx,
            shared_gate_up_wid, shared_down_wid, shared_gate_wid,
        };
        Ok(())
    }

    /// Set dense MLP config for a layer.
    #[pyo3(signature = (layer_idx, gate_proj_wid, up_proj_wid, down_proj_wid))]
    pub fn set_decode_layer_dense(
        &mut self,
        layer_idx: usize,
        gate_proj_wid: usize,
        up_proj_wid: usize,
        down_proj_wid: usize,
    ) -> PyResult<()> {
        let g = self.decode_graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure_decode first"))?;
        g.layers[layer_idx].mlp = DecodeMlpConfig::Dense {
            gate_proj_wid, up_proj_wid, down_proj_wid,
        };
        Ok(())
    }

    /// Set RoPE tables for GQA layers.
    #[pyo3(signature = (cos_ptr, sin_ptr, half_dim, max_seq))]
    pub fn set_decode_rope(
        &mut self,
        cos_ptr: usize,
        sin_ptr: usize,
        half_dim: usize,
        max_seq: usize,
    ) -> PyResult<()> {
        let g = self.decode_graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure_decode first"))?;
        g.rope_cos_ptr = cos_ptr;
        g.rope_sin_ptr = sin_ptr;
        g.rope_half_dim = half_dim;
        g.max_rope_seq = max_seq;
        Ok(())
    }

    /// Share MoE weight store from KrasisEngine.
    pub fn set_moe_store(&mut self, engine: PyRefMut<'_, crate::moe::KrasisEngine>) -> PyResult<()> {
        let store = engine.get_weight_store()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Engine has no weight store"))?;
        let g = self.decode_graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure_decode first"))?;
        let cfg = &store.config;
        let hidden = cfg.hidden_size;
        let intermediate = cfg.moe_intermediate_size;
        let gs = g.group_size;
        let topk = g.topk;
        g.moe_parallel = engine.get_parallel();
        g.moe_scratch = Some(ExpertScratch::new(hidden, intermediate, gs));
        g.moe_scratch_pool = (0..topk).map(|_| ExpertScratch::new(hidden, intermediate, gs)).collect();
        g.moe_store = Some(store);
        log::info!("DecodeGraph MoE store set: hidden={}, intermediate={}, topk={}", hidden, intermediate, topk);
        Ok(())
    }

    /// Repack all weights (non-expert + expert) to tiled layout for better memory access.
    /// Call after all weights are loaded but before running decode.
    pub fn repack_to_tiled(&mut self) -> PyResult<()> {
        use crate::kernel::avx2::{repack_tiled_int4_packed, repack_tiled_int8_packed, repack_tiled_scales};
        let t0 = std::time::Instant::now();

        // Repack non-expert TransposedWeights
        let mut n_weights = 0usize;
        for w in self.weights.iter_mut() {
            if w.tiled { continue; }
            let num_groups = w.cols / w.group_size;
            let new_packed = if w.num_bits == 4 {
                repack_tiled_int4_packed(&w.packed, w.cols, w.rows)
            } else {
                repack_tiled_int8_packed(&w.packed, w.cols, w.rows)
            };
            let new_scales = repack_tiled_scales(&w.scales, num_groups, w.rows);
            w.packed = new_packed;
            w.scales = new_scales;
            w.tiled = true;
            n_weights += 1;
        }

        // Repack expert weights if moe_store is available
        let mut n_experts = 0usize;
        if let Some(ref mut g) = self.decode_graph {
            if let Some(ref mut arc) = g.moe_store {
                if let Some(ws) = Arc::get_mut(arc) {
                    for layer in ws.experts_cpu.iter_mut() {
                        for expert in layer.iter_mut() {
                            if expert.tiled { continue; }
                            let h = expert.hidden_size;
                            let m = expert.intermediate_size;
                            let gs = expert.group_size;
                            let two_m = 2 * m;

                            // w13: K=hidden_size, N=2*intermediate_size
                            expert.w13_packed = if expert.num_bits == 4 {
                                repack_tiled_int4_packed(&expert.w13_packed, h, two_m)
                            } else {
                                repack_tiled_int8_packed(&expert.w13_packed, h, two_m)
                            };
                            expert.w13_scales = repack_tiled_scales(&expert.w13_scales, h / gs, two_m);

                            // w2: K=intermediate_size, N=hidden_size
                            expert.w2_packed = if expert.w2_bits == 4 {
                                repack_tiled_int4_packed(&expert.w2_packed, m, h)
                            } else {
                                repack_tiled_int8_packed(&expert.w2_packed, m, h)
                            };
                            expert.w2_scales = repack_tiled_scales(&expert.w2_scales, m / gs, h);

                            expert.tiled = true;
                            n_experts += 1;
                        }
                    }
                } else {
                    log::warn!("Cannot repack experts: Arc has multiple owners");
                }
            }
        }

        log::info!("Repacked to tiled layout: {} weights + {} experts in {:.1}s",
            n_weights, n_experts, t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Consolidate all TransposedWeight data into contiguous mmap regions with MADV_HUGEPAGE.
    ///
    /// After tiling, each weight's packed/scales data lives in separate heap Vecs.
    /// This method allocates two large contiguous mmap regions (one for packed, one for scales),
    /// copies all weight data into them, and replaces the heap Vecs with mmap-backed slices.
    /// The contiguous layout enables transparent huge pages (2MB), dramatically reducing TLB misses.
    pub fn consolidate_weights_mmap(&mut self) -> PyResult<()> {
        if self.weights.is_empty() {
            return Ok(());
        }
        if !self.mmap_regions.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Weights already consolidated into mmap"));
        }

        let t0 = std::time::Instant::now();

        // Calculate total bytes needed
        let total_packed_bytes: usize = self.weights.iter().map(|w| w.packed.len() * 4).sum();
        let total_scales_bytes: usize = self.weights.iter().map(|w| w.scales.len() * 2).sum();

        if total_packed_bytes == 0 {
            return Ok(());
        }

        log::info!("Consolidating {} weights into contiguous mmap: {:.1} MB packed + {:.1} MB scales",
            self.weights.len(), total_packed_bytes as f64 / 1e6, total_scales_bytes as f64 / 1e6);

        // Allocate contiguous mmap regions
        let packed_base = unsafe {
            libc::mmap(std::ptr::null_mut(), total_packed_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1, 0)
        };
        if packed_base == libc::MAP_FAILED {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "mmap failed for packed weight consolidation"));
        }

        let scales_base = unsafe {
            libc::mmap(std::ptr::null_mut(), total_scales_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1, 0)
        };
        if scales_base == libc::MAP_FAILED {
            unsafe { libc::munmap(packed_base, total_packed_bytes); }
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "mmap failed for scales consolidation"));
        }

        // Request huge pages
        unsafe {
            libc::madvise(packed_base, total_packed_bytes, libc::MADV_HUGEPAGE);
            libc::madvise(scales_base, total_scales_bytes, libc::MADV_HUGEPAGE);
        }

        // Copy data from heap Vecs into mmap, replace Vecs with mmap-backed ones
        let mut p_off: usize = 0; // byte offset into packed mmap
        let mut s_off: usize = 0; // byte offset into scales mmap

        for w in self.weights.iter_mut() {
            let pk_bytes = w.packed.len() * 4;
            let sc_bytes = w.scales.len() * 2;
            let pk_len = w.packed.len();
            let sc_len = w.scales.len();

            // Copy packed data into mmap
            unsafe {
                std::ptr::copy_nonoverlapping(
                    w.packed.as_ptr() as *const u8,
                    (packed_base as *mut u8).add(p_off),
                    pk_bytes,
                );
            }
            // Drop old heap Vec, replace with mmap-backed Vec
            let old_packed = std::mem::take(&mut w.packed);
            drop(old_packed);
            w.packed = unsafe {
                Vec::from_raw_parts(
                    (packed_base as *mut u32).add(p_off / 4),
                    pk_len, pk_len,
                )
            };
            p_off += pk_bytes;

            // Copy scales data into mmap
            unsafe {
                std::ptr::copy_nonoverlapping(
                    w.scales.as_ptr() as *const u8,
                    (scales_base as *mut u8).add(s_off),
                    sc_bytes,
                );
            }
            let old_scales = std::mem::take(&mut w.scales);
            drop(old_scales);
            w.scales = unsafe {
                Vec::from_raw_parts(
                    (scales_base as *mut u16).add(s_off / 2),
                    sc_len, sc_len,
                )
            };
            s_off += sc_bytes;
        }

        assert_eq!(p_off, total_packed_bytes);
        assert_eq!(s_off, total_scales_bytes);

        // Track mmap regions for cleanup (stored as usize for Send/Sync)
        self.mmap_regions.push((packed_base as usize, total_packed_bytes));
        self.mmap_regions.push((scales_base as usize, total_scales_bytes));

        log::info!("Consolidated weights into mmap with MADV_HUGEPAGE in {:.1}ms",
            t0.elapsed().as_secs_f64() * 1000.0);
        Ok(())
    }

    /// Finalize decode graph — allocate scratch buffers based on layer configs.
    pub fn finalize_decode(&mut self) -> PyResult<()> {
        let g = self.decode_graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure_decode first"))?;
        let hs = g.hidden_size;
        let gs = g.group_size;
        // Find max dimensions across layers
        let mut max_qkvz = 0usize;
        let mut max_ba = 0usize;
        let mut max_nv_dk = 0usize;
        let mut max_nv_dv = 0usize;
        let mut max_nv = 0usize;
        let mut max_q_proj = 0usize;
        let mut max_kv_proj = 0usize;
        let mut max_heads = 0usize;
        let mut max_heads_hd = 0usize;
        let mut max_intermediate = 0usize;
        let mut max_conv_dim = 0usize;
        let mut max_k = 0usize; // for quantization scratch

        for layer in &g.layers {
            match &layer.attn {
                DecodeAttnConfig::LinearAttention { nk, nv, dk, dv, hr, kernel_dim: _, conv_dim,
                    in_proj_qkvz_wid, in_proj_ba_wid, out_proj_wid, .. } => {
                    let group_dim = 2 * dk + 2 * dv * hr;
                    max_qkvz = max_qkvz.max(nk * group_dim);
                    max_ba = max_ba.max(nk * 2 * hr);
                    max_nv_dk = max_nv_dk.max(nv * dk);
                    max_nv_dv = max_nv_dv.max(nv * dv);
                    max_nv = max_nv.max(*nv);
                    max_conv_dim = max_conv_dim.max(*conv_dim);
                    max_k = max_k.max(self.weights[*in_proj_qkvz_wid].cols);
                    max_k = max_k.max(self.weights[*in_proj_ba_wid].cols);
                    max_k = max_k.max(self.weights[*out_proj_wid].cols);
                }
                DecodeAttnConfig::GQA { num_heads, num_kv_heads, head_dim, gated,
                    q_proj_wid, k_proj_wid, v_proj_wid, o_proj_wid, fused_qkv_wid, .. } => {
                    let q_size = if *gated { num_heads * head_dim * 2 } else { num_heads * head_dim };
                    max_q_proj = max_q_proj.max(q_size);
                    max_kv_proj = max_kv_proj.max(num_kv_heads * head_dim);
                    max_heads = max_heads.max(*num_heads);
                    max_heads_hd = max_heads_hd.max(num_heads * head_dim);
                    if let Some(fid) = fused_qkv_wid {
                        max_k = max_k.max(self.weights[*fid].cols);
                    } else {
                        max_k = max_k.max(self.weights[*q_proj_wid].cols);
                        max_k = max_k.max(self.weights[*k_proj_wid].cols);
                        max_k = max_k.max(self.weights[*v_proj_wid].cols);
                    }
                    max_k = max_k.max(self.weights[*o_proj_wid].cols);
                }
                DecodeAttnConfig::MLA { kv_a_proj_wid, o_proj_wid,
                    q_proj_wid, q_a_proj_wid, q_b_proj_wid,
                    num_heads, kv_lora_rank, qk_nope_dim, qk_rope_dim, v_head_dim, .. } => {
                    max_heads = max_heads.max(*num_heads);
                    max_heads_hd = max_heads_hd.max(num_heads * v_head_dim);
                    max_k = max_k.max(self.weights[*kv_a_proj_wid].cols);
                    max_k = max_k.max(self.weights[*o_proj_wid].cols);
                    if let Some(wid) = q_proj_wid {
                        max_k = max_k.max(self.weights[*wid].cols);
                    }
                    if let Some(wid) = q_a_proj_wid {
                        max_k = max_k.max(self.weights[*wid].cols);
                    }
                    if let Some(wid) = q_b_proj_wid {
                        max_k = max_k.max(self.weights[*wid].cols);
                    }
                }
            }
            match &layer.mlp {
                DecodeMlpConfig::MoE { shared_gate_up_wid, shared_down_wid, .. } => {
                    if let Some(wid) = shared_gate_up_wid {
                        max_intermediate = max_intermediate.max(self.weights[*wid].rows / 2);
                        max_k = max_k.max(self.weights[*wid].cols);
                    }
                    if let Some(wid) = shared_down_wid {
                        max_k = max_k.max(self.weights[*wid].cols);
                    }
                }
                DecodeMlpConfig::Dense { gate_proj_wid, up_proj_wid, down_proj_wid } => {
                    max_intermediate = max_intermediate.max(self.weights[*gate_proj_wid].rows);
                    // down_proj.cols may be padded larger than gate_proj.rows
                    max_intermediate = max_intermediate.max(self.weights[*down_proj_wid].cols);
                    max_k = max_k.max(self.weights[*gate_proj_wid].cols);
                    max_k = max_k.max(self.weights[*up_proj_wid].cols);
                    max_k = max_k.max(self.weights[*down_proj_wid].cols);
                }
                DecodeMlpConfig::None => {}
            }
        }
        // lm_head K
        max_k = max_k.max(self.weights[g.lm_head_wid].cols);

        // Allocate scratch
        g.la_qkvz_buf = vec![0.0; max_qkvz.max(1)];
        g.la_ba_buf = vec![0.0; max_ba.max(1)];
        g.la_q_buf = vec![0.0; max_nv_dk.max(1)];
        g.la_k_buf = vec![0.0; max_nv_dk.max(1)];
        g.la_v_buf = vec![0.0; max_nv_dv.max(1)];
        g.la_z_buf = vec![0.0; max_nv_dv.max(1)];
        g.la_g_buf = vec![0.0; max_nv.max(1)];
        g.la_beta_buf = vec![0.0; max_nv.max(1)];
        g.la_recur_out = vec![0.0; max_nv_dv.max(1)];
        g.la_gated_out = vec![0.0; max_nv_dv.max(1)];
        g.la_mixed_qkv = vec![0.0; max_conv_dim.max(1)];
        g.la_conv_out = vec![0.0; max_conv_dim.max(1)];
        g.gqa_q_buf = vec![0.0; max_q_proj.max(1)];
        g.gqa_k_buf = vec![0.0; max_kv_proj.max(1)];
        g.gqa_v_buf = vec![0.0; max_kv_proj.max(1)];
        g.gqa_qkv_buf = vec![0.0; (max_q_proj + 2 * max_kv_proj).max(1)];
        // scores buffer sized for max_heads * max_kv_seq — will be set after set_decode_state
        g.gqa_scores = Vec::new(); // deferred
        g.gqa_attn_out = vec![0.0; max_heads_hd.max(1)];

        // MLA scratch — sized by scanning MLA layers
        {
            let mut max_kv_out = 0usize;
            let mut max_kv_lora = 0usize;
            let mut max_q_full = 0usize;
            let mut max_q_compressed = 0usize;
            let mut max_q_absorbed = 0usize;
            let mut max_v_projected = 0usize;
            for layer in &g.layers {
                if let DecodeAttnConfig::MLA { num_heads, kv_lora_rank, qk_nope_dim, qk_rope_dim,
                    v_head_dim, q_a_proj_wid, .. } = &layer.attn {
                    let head_dim = qk_nope_dim + qk_rope_dim;
                    max_kv_out = max_kv_out.max(kv_lora_rank + qk_rope_dim);
                    max_kv_lora = max_kv_lora.max(*kv_lora_rank);
                    max_q_full = max_q_full.max(num_heads * head_dim);
                    max_q_absorbed = max_q_absorbed.max(num_heads * kv_lora_rank);
                    max_v_projected = max_v_projected.max(num_heads * v_head_dim);
                    if let Some(wid) = q_a_proj_wid {
                        max_q_compressed = max_q_compressed.max(self.weights[*wid].rows);
                    }
                }
            }
            if max_kv_out > 0 {
                g.mla_kv_out = vec![0.0; max_kv_out];
                g.mla_kv_compressed = vec![0.0; max_kv_lora];
                g.mla_q_full = vec![0.0; max_q_full];
                g.mla_q_compressed = vec![0.0; max_q_compressed.max(1)];
                g.mla_q_absorbed = vec![0.0; max_q_absorbed];
                // mla_attn_scores deferred until set_decode_state (needs kv_max_seq)
                g.mla_attn_scores = Vec::new();
                g.mla_attn_out = vec![0.0; max_q_absorbed];
                g.mla_v_projected = vec![0.0; max_v_projected];
            }
        }

        g.mlp_gate_up = vec![0.0; (max_intermediate * 2).max(1)];
        g.mlp_hidden_buf = vec![0.0; max_intermediate.max(1)];
        g.act_int16 = vec![0i16; max_k];
        g.act_scales = vec![0.0f32; max_k / gs];

        // MoE routing scratch (sized by max expert count)
        let mut max_ne = 0usize;
        for layer in &g.layers {
            if let DecodeMlpConfig::MoE { route_id, .. } = &layer.mlp {
                max_ne = max_ne.max(self.route_weights[*route_id].num_experts);
            }
        }
        g.route_logits = vec![0.0f32; max_ne.max(1)];
        g.route_scores = vec![0.0f32; max_ne.max(1)];
        g.route_corrected = vec![0.0f32; max_ne.max(1)];

        log::info!("DecodeGraph finalized: {} layers, max_k={}, scratch allocated", g.layers.len(), max_k);
        Ok(())
    }

    /// Update per-request state pointers (call after prepare).
    #[pyo3(signature = (seq_len, kv_max_seq, kv_k_ptrs, kv_v_ptrs, conv_state_ptrs, recur_state_ptrs, mla_ckv_ptrs=None, mla_kpe_ptrs=None))]
    pub fn set_decode_state(
        &mut self,
        seq_len: usize,
        kv_max_seq: usize,
        kv_k_ptrs: Vec<usize>,
        kv_v_ptrs: Vec<usize>,
        conv_state_ptrs: Vec<usize>,
        recur_state_ptrs: Vec<usize>,
        mla_ckv_ptrs: Option<Vec<usize>>,
        mla_kpe_ptrs: Option<Vec<usize>>,
    ) -> PyResult<()> {
        let g = self.decode_graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure_decode first"))?;
        g.seq_len = seq_len;
        g.kv_max_seq = kv_max_seq;
        g.kv_k_ptrs = kv_k_ptrs;
        g.kv_v_ptrs = kv_v_ptrs;
        g.conv_state_ptrs = conv_state_ptrs;
        g.recur_state_ptrs = recur_state_ptrs;
        if let Some(ptrs) = mla_ckv_ptrs { g.mla_ckv_ptrs = ptrs; }
        if let Some(ptrs) = mla_kpe_ptrs { g.mla_kpe_ptrs = ptrs; }
        // Allocate/resize GQA scores buffer for current max_seq
        let mut max_heads = 0;
        for layer in &g.layers {
            match &layer.attn {
                DecodeAttnConfig::GQA { num_heads, .. } => {
                    max_heads = max_heads.max(*num_heads);
                }
                DecodeAttnConfig::MLA { num_heads, .. } => {
                    max_heads = max_heads.max(*num_heads);
                }
                _ => {}
            }
        }
        let needed = max_heads * kv_max_seq;
        if g.gqa_scores.len() < needed {
            g.gqa_scores.resize(needed, 0.0);
        }
        // Also resize MLA attention scores buffer
        if g.mla_attn_scores.len() < needed {
            g.mla_attn_scores.resize(needed, 0.0);
        }
        Ok(())
    }

    /// Full decode step — runs entire layer loop in Rust.
    ///
    /// Replaces the Python step() method. One Python call per token.
    /// output_ptr: *mut f32 [vocab_size] — logits written here.
    pub fn decode_step(
        &mut self,
        token_id: usize,
        position: usize,
        output_ptr: usize,
    ) -> PyResult<()> {
        use std::time::Instant;

        // Split borrows: graph is mutable, weights/norms/routes are read-only
        let graph = self.decode_graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure_decode first"))?;
        let weights = &self.weights;
        let norm_weights = &self.norm_weights;
        let route_weights = &self.route_weights;
        let norm_bias_one = self.norm_bias_one;

        let hs = graph.hidden_size;
        let eps = graph.eps;
        let parallel = graph.parallel;
        let timing = graph.timing_enabled;
        let t_step_start = if timing { Instant::now() } else { Instant::now() };

        // ── Embedding lookup ──
        let emb: &[f32] = unsafe {
            std::slice::from_raw_parts(
                (graph.embedding_ptr as *const f32).add(token_id * hs), hs)
        };
        graph.hidden[..hs].copy_from_slice(emb);

        let mut first_residual = true;

        // ── Layer loop ──
        for layer_idx in 0..graph.layers.len() {
            // Pre-attention norm
            let t0 = if timing { Instant::now() } else { t_step_start };
            unsafe {
                fused_add_rmsnorm_avx2(
                    &mut graph.hidden, &mut graph.residual,
                    &norm_weights[graph.layers[layer_idx].input_norm_id],
                    eps, first_residual, norm_bias_one);
            }
            if timing { graph.t_norm += t0.elapsed().as_secs_f64(); }
            first_residual = false;

            // Attention
            match &graph.layers[layer_idx].attn {
                DecodeAttnConfig::LinearAttention {
                    in_proj_qkvz_wid, in_proj_ba_wid, out_proj_wid,
                    conv_weight, a_log, dt_bias, norm_weight,
                    nk, nv, dk, dv, hr, kernel_dim, conv_dim, scale,
                } => {
                    let nk = *nk; let nv = *nv; let dk = *dk; let dv = *dv;
                    let hr = *hr; let kd = *kernel_dim; let cd = *conv_dim;

                    // Projections
                    let t0 = if timing { Instant::now() } else { t_step_start };
                    let k_in = weights[*in_proj_qkvz_wid].cols;
                    quantize_activation_int16_f32(
                        &graph.hidden[..k_in], graph.group_size,
                        &mut graph.act_int16[..k_in],
                        &mut graph.act_scales[..k_in / graph.group_size]);
                    dispatch_matmul_free(
                        &weights[*in_proj_qkvz_wid],
                        &graph.act_int16[..k_in],
                        &graph.act_scales[..k_in / graph.group_size],
                        &mut graph.la_qkvz_buf[..weights[*in_proj_qkvz_wid].rows],
                        parallel);
                    dispatch_matmul_free(
                        &weights[*in_proj_ba_wid],
                        &graph.act_int16[..k_in],
                        &graph.act_scales[..k_in / graph.group_size],
                        &mut graph.la_ba_buf[..weights[*in_proj_ba_wid].rows],
                        parallel);
                    if timing { graph.t_la_proj += t0.elapsed().as_secs_f64(); }

                    // Conv + gate params
                    let t0 = if timing { Instant::now() } else { t_step_start };
                    let conv_state: &mut [f32] = unsafe {
                        std::slice::from_raw_parts_mut(
                            graph.conv_state_ptrs[layer_idx] as *mut f32, cd * kd)
                    };
                    decode_la_conv(
                        &graph.la_qkvz_buf, &graph.la_ba_buf,
                        conv_state, conv_weight, a_log, dt_bias, *scale,
                        &mut graph.la_q_buf, &mut graph.la_k_buf,
                        &mut graph.la_v_buf, &mut graph.la_z_buf,
                        &mut graph.la_g_buf, &mut graph.la_beta_buf,
                        &mut graph.la_mixed_qkv, &mut graph.la_conv_out,
                        nk, nv, dk, dv, hr, kd, cd);
                    if timing { graph.t_la_conv += t0.elapsed().as_secs_f64(); }

                    // Recurrent state update
                    let t0 = if timing { Instant::now() } else { t_step_start };
                    let state: &mut [f32] = unsafe {
                        std::slice::from_raw_parts_mut(
                            graph.recur_state_ptrs[layer_idx] as *mut f32, nv * dk * dv)
                    };
                    unsafe {
                        linear_attention_recurrent_avx2(
                            state,
                            &graph.la_q_buf[..nv * dk],
                            &graph.la_k_buf[..nv * dk],
                            &graph.la_v_buf[..nv * dv],
                            &graph.la_g_buf[..nv],
                            &graph.la_beta_buf[..nv],
                            &mut graph.la_recur_out[..nv * dv],
                            nv, dk, dv);
                    }
                    if timing { graph.t_la_recur += t0.elapsed().as_secs_f64(); }

                    // Gated RMSNorm + SiLU (AVX2)
                    let t0 = if timing { Instant::now() } else { t_step_start };
                    unsafe {
                        gated_rmsnorm_silu_avx2(
                            &graph.la_recur_out, &graph.la_z_buf, norm_weight,
                            &mut graph.la_gated_out, nv, dv, eps);
                    }
                    if timing { graph.t_la_gate_norm += t0.elapsed().as_secs_f64(); }

                    // Out projection
                    let t0 = if timing { Instant::now() } else { t_step_start };
                    let k_out = weights[*out_proj_wid].cols;
                    quantize_activation_int16_f32(
                        &graph.la_gated_out[..k_out], graph.group_size,
                        &mut graph.act_int16[..k_out],
                        &mut graph.act_scales[..k_out / graph.group_size]);
                    dispatch_matmul_free(
                        &weights[*out_proj_wid],
                        &graph.act_int16[..k_out],
                        &graph.act_scales[..k_out / graph.group_size],
                        &mut graph.hidden[..hs],
                        parallel);
                    if timing { graph.t_la_out_proj += t0.elapsed().as_secs_f64(); }
                }

                DecodeAttnConfig::GQA {
                    q_proj_wid, k_proj_wid, v_proj_wid, o_proj_wid,
                    q_norm, k_norm, gated, num_heads, num_kv_heads, head_dim, sm_scale,
                    fused_qkv_wid,
                } => {
                    let nh = *num_heads; let nkv = *num_kv_heads;
                    let hd = *head_dim;

                    // Q/K/V projections
                    let t0 = if timing { Instant::now() } else { t_step_start };
                    let q_rows = weights[*q_proj_wid].rows;
                    let k_rows = weights[*k_proj_wid].rows;
                    let v_rows = weights[*v_proj_wid].rows;
                    if let Some(fid) = fused_qkv_wid {
                        // Fused Q+K+V: one dispatch, output split afterward
                        let fw = &weights[*fid];
                        let k_in = fw.cols;
                        let total_rows = fw.rows;
                        quantize_activation_int16_f32(
                            &graph.hidden[..k_in], graph.group_size,
                            &mut graph.act_int16[..k_in],
                            &mut graph.act_scales[..k_in / graph.group_size]);
                        dispatch_matmul_free(fw,
                            &graph.act_int16[..k_in], &graph.act_scales[..k_in / graph.group_size],
                            &mut graph.gqa_qkv_buf[..total_rows], parallel);
                        // Split fused output into Q, K, V
                        graph.gqa_q_buf[..q_rows].copy_from_slice(&graph.gqa_qkv_buf[..q_rows]);
                        graph.gqa_k_buf[..k_rows].copy_from_slice(&graph.gqa_qkv_buf[q_rows..q_rows+k_rows]);
                        graph.gqa_v_buf[..v_rows].copy_from_slice(&graph.gqa_qkv_buf[q_rows+k_rows..q_rows+k_rows+v_rows]);
                    } else {
                        // Separate Q, K, V dispatches
                        let k_in = weights[*q_proj_wid].cols;
                        quantize_activation_int16_f32(
                            &graph.hidden[..k_in], graph.group_size,
                            &mut graph.act_int16[..k_in],
                            &mut graph.act_scales[..k_in / graph.group_size]);
                        dispatch_matmul_free(&weights[*q_proj_wid],
                            &graph.act_int16[..k_in], &graph.act_scales[..k_in / graph.group_size],
                            &mut graph.gqa_q_buf[..q_rows], parallel);
                        dispatch_matmul_free(&weights[*k_proj_wid],
                            &graph.act_int16[..k_in], &graph.act_scales[..k_in / graph.group_size],
                            &mut graph.gqa_k_buf[..k_rows], parallel);
                        dispatch_matmul_free(&weights[*v_proj_wid],
                            &graph.act_int16[..k_in], &graph.act_scales[..k_in / graph.group_size],
                            &mut graph.gqa_v_buf[..v_rows], parallel);
                    }
                    if timing { graph.t_gqa_proj += t0.elapsed().as_secs_f64(); }

                    // Gated attention rearrange + QK norm + RoPE
                    let t0 = if timing { Instant::now() } else { t_step_start };
                    if *gated {
                        for h in 0..nh {
                            for d in 0..hd {
                                graph.gqa_attn_out[h * hd + d] = graph.gqa_q_buf[h * hd * 2 + hd + d];
                            }
                        }
                        for h in (1..nh).rev() {
                            for d in 0..hd {
                                graph.gqa_q_buf[h * hd + d] = graph.gqa_q_buf[h * hd * 2 + d];
                            }
                        }
                    }
                    if let Some(qn) = q_norm {
                        for h in 0..nh {
                            let base = h * hd;
                            let mut sum_sq = 0.0f32;
                            for d in 0..hd {
                                sum_sq += graph.gqa_q_buf[base + d] * graph.gqa_q_buf[base + d];
                            }
                            let rms = (sum_sq / hd as f32 + eps).sqrt().recip();
                            let w_offset = if qn.len() == nh * hd { base } else { 0 };
                            for d in 0..hd {
                                graph.gqa_q_buf[base + d] *= rms * qn[w_offset + d];
                            }
                        }
                    }
                    if let Some(kn) = k_norm {
                        for h in 0..nkv {
                            let base = h * hd;
                            let mut sum_sq = 0.0f32;
                            for d in 0..hd {
                                sum_sq += graph.gqa_k_buf[base + d] * graph.gqa_k_buf[base + d];
                            }
                            let rms = (sum_sq / hd as f32 + eps).sqrt().recip();
                            let w_offset = if kn.len() == nkv * hd { base } else { 0 };
                            for d in 0..hd {
                                graph.gqa_k_buf[base + d] *= rms * kn[w_offset + d];
                            }
                        }
                    }
                    // RoPE
                    let d2 = graph.rope_half_dim;
                    let cos: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            (graph.rope_cos_ptr as *const f32).add(position * d2), d2)
                    };
                    let sin: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            (graph.rope_sin_ptr as *const f32).add(position * d2), d2)
                    };
                    for h in 0..nh {
                        let base = h * hd;
                        for i in 0..d2 {
                            let x1 = graph.gqa_q_buf[base + i];
                            let x2 = graph.gqa_q_buf[base + d2 + i];
                            graph.gqa_q_buf[base + i] = x1 * cos[i] - x2 * sin[i];
                            graph.gqa_q_buf[base + d2 + i] = x2 * cos[i] + x1 * sin[i];
                        }
                    }
                    for h in 0..nkv {
                        let base = h * hd;
                        for i in 0..d2 {
                            let x1 = graph.gqa_k_buf[base + i];
                            let x2 = graph.gqa_k_buf[base + d2 + i];
                            graph.gqa_k_buf[base + i] = x1 * cos[i] - x2 * sin[i];
                            graph.gqa_k_buf[base + d2 + i] = x2 * cos[i] + x1 * sin[i];
                        }
                    }
                    if timing { graph.t_gqa_rope += t0.elapsed().as_secs_f64(); }

                    // KV cache write + Attention compute
                    let t0 = if timing { Instant::now() } else { t_step_start };
                    let kv_stride = nkv * hd;
                    let k_cache: &mut [f32] = unsafe {
                        std::slice::from_raw_parts_mut(
                            graph.kv_k_ptrs[layer_idx] as *mut f32,
                            graph.kv_max_seq * kv_stride)
                    };
                    let v_cache: &mut [f32] = unsafe {
                        std::slice::from_raw_parts_mut(
                            graph.kv_v_ptrs[layer_idx] as *mut f32,
                            graph.kv_max_seq * kv_stride)
                    };
                    let write_offset = position * kv_stride;
                    k_cache[write_offset..write_offset + kv_stride]
                        .copy_from_slice(&graph.gqa_k_buf[..kv_stride]);
                    v_cache[write_offset..write_offset + kv_stride]
                        .copy_from_slice(&graph.gqa_v_buf[..kv_stride]);
                    let seq_len = position + 1;
                    unsafe {
                        gqa_attention_compute_avx2(
                            &graph.gqa_q_buf, k_cache, v_cache,
                            &mut graph.gqa_scores, &mut graph.gqa_attn_out,
                            nh, nkv, hd, graph.kv_max_seq, seq_len, *sm_scale,
                            *gated);
                    }
                    if timing { graph.t_gqa_attn += t0.elapsed().as_secs_f64(); }

                    // O projection
                    let t0 = if timing { Instant::now() } else { t_step_start };
                    let o_k = weights[*o_proj_wid].cols;
                    quantize_activation_int16_f32(
                        &graph.gqa_attn_out[..o_k], graph.group_size,
                        &mut graph.act_int16[..o_k],
                        &mut graph.act_scales[..o_k / graph.group_size]);
                    dispatch_matmul_free(
                        &weights[*o_proj_wid],
                        &graph.act_int16[..o_k],
                        &graph.act_scales[..o_k / graph.group_size],
                        &mut graph.hidden[..hs],
                        parallel);
                    if timing { graph.t_gqa_o_proj += t0.elapsed().as_secs_f64(); }
                }

                DecodeAttnConfig::MLA {
                    kv_a_proj_wid, o_proj_wid,
                    q_proj_wid, q_a_proj_wid, q_b_proj_wid,
                    w_kc, w_vc, kv_a_norm, q_a_norm,
                    rope_cos, rope_sin,
                    num_heads, kv_lora_rank, qk_nope_dim, qk_rope_dim, v_head_dim, sm_scale,
                } => {
                    let nh = *num_heads;
                    let klr = *kv_lora_rank;
                    let qk_nd = *qk_nope_dim;
                    let qk_rd = *qk_rope_dim;
                    let vhd = *v_head_dim;
                    let head_dim = qk_nd + qk_rd;
                    let rope_half = qk_rd / 2;

                    // ── Step 1: KV compressed projection ──
                    // hidden → kv_out[kv_lora_rank + qk_rope_dim]
                    let t0 = if timing { Instant::now() } else { t_step_start };
                    let k_in = weights[*kv_a_proj_wid].cols;
                    quantize_activation_int16_f32(
                        &graph.hidden[..k_in], graph.group_size,
                        &mut graph.act_int16[..k_in],
                        &mut graph.act_scales[..k_in / graph.group_size]);
                    dispatch_matmul_free(
                        &weights[*kv_a_proj_wid],
                        &graph.act_int16[..k_in],
                        &graph.act_scales[..k_in / graph.group_size],
                        &mut graph.mla_kv_out[..klr + qk_rd],
                        parallel);

                    // Split: kv_compressed = kv_out[..klr], k_pe = kv_out[klr..]
                    // RMSNorm on kv_compressed
                    graph.mla_kv_compressed[..klr].copy_from_slice(&graph.mla_kv_out[..klr]);
                    {
                        let mut sum_sq = 0.0f32;
                        for i in 0..klr {
                            sum_sq += graph.mla_kv_compressed[i] * graph.mla_kv_compressed[i];
                        }
                        let rms = (sum_sq / klr as f32 + eps).sqrt().recip();
                        for i in 0..klr {
                            graph.mla_kv_compressed[i] *= rms * kv_a_norm[i];
                        }
                    }

                    // ── Step 2: Query projection ──
                    if let Some(qa_wid) = q_a_proj_wid {
                        // LoRA path: q_a_proj → RMSNorm → q_b_proj
                        let qa_rows = weights[*qa_wid].rows;
                        let qa_k = weights[*qa_wid].cols;
                        quantize_activation_int16_f32(
                            &graph.hidden[..qa_k], graph.group_size,
                            &mut graph.act_int16[..qa_k],
                            &mut graph.act_scales[..qa_k / graph.group_size]);
                        dispatch_matmul_free(
                            &weights[*qa_wid],
                            &graph.act_int16[..qa_k],
                            &graph.act_scales[..qa_k / graph.group_size],
                            &mut graph.mla_q_compressed[..qa_rows],
                            parallel);
                        // RMSNorm on q_compressed
                        if let Some(qa_norm) = q_a_norm {
                            let mut sum_sq = 0.0f32;
                            for i in 0..qa_rows {
                                sum_sq += graph.mla_q_compressed[i] * graph.mla_q_compressed[i];
                            }
                            let rms = (sum_sq / qa_rows as f32 + eps).sqrt().recip();
                            for i in 0..qa_rows {
                                graph.mla_q_compressed[i] *= rms * qa_norm[i];
                            }
                        }
                        // q_b_proj: q_compressed → q_full
                        let qb_wid = q_b_proj_wid.unwrap();
                        let qb_k = weights[qb_wid].cols;
                        quantize_activation_int16_f32(
                            &graph.mla_q_compressed[..qb_k], graph.group_size,
                            &mut graph.act_int16[..qb_k],
                            &mut graph.act_scales[..qb_k / graph.group_size]);
                        dispatch_matmul_free(
                            &weights[qb_wid],
                            &graph.act_int16[..qb_k],
                            &graph.act_scales[..qb_k / graph.group_size],
                            &mut graph.mla_q_full[..nh * head_dim],
                            parallel);
                    } else {
                        // Direct path: q_proj
                        let qw = q_proj_wid.as_ref().unwrap();
                        let q_k = weights[*qw].cols;
                        quantize_activation_int16_f32(
                            &graph.hidden[..q_k], graph.group_size,
                            &mut graph.act_int16[..q_k],
                            &mut graph.act_scales[..q_k / graph.group_size]);
                        dispatch_matmul_free(
                            &weights[*qw],
                            &graph.act_int16[..q_k],
                            &graph.act_scales[..q_k / graph.group_size],
                            &mut graph.mla_q_full[..nh * head_dim],
                            parallel);
                    }
                    if timing { graph.t_mla_proj += t0.elapsed().as_secs_f64(); }

                    // ── Step 3: De-interleave + RoPE ──
                    let t0 = if timing { Instant::now() } else { t_step_start };

                    // De-interleave k_pe in-place: [re0,im0,re1,im1,...] → [re0,re1,...,im0,im1,...]
                    // Work on mla_kv_out[klr..klr+qk_rd]
                    {
                        // Use temporary storage (stack for small dims)
                        let mut tmp = [0.0f32; 256]; // qk_rope_dim is typically 64
                        let src = &graph.mla_kv_out[klr..klr + qk_rd];
                        for i in 0..rope_half {
                            tmp[i] = src[i * 2];               // real parts
                            tmp[rope_half + i] = src[i * 2 + 1]; // imag parts
                        }
                        graph.mla_kv_out[klr..klr + qk_rd].copy_from_slice(&tmp[..qk_rd]);
                    }

                    // De-interleave q_pe per head and apply RoPE
                    let cos = &rope_cos[position * rope_half..(position + 1) * rope_half];
                    let sin = &rope_sin[position * rope_half..(position + 1) * rope_half];

                    for h in 0..nh {
                        let base = h * head_dim + qk_nd; // start of q_pe for this head
                        // De-interleave q_pe for this head
                        let mut tmp = [0.0f32; 256];
                        for i in 0..rope_half {
                            tmp[i] = graph.mla_q_full[base + i * 2];
                            tmp[rope_half + i] = graph.mla_q_full[base + i * 2 + 1];
                        }
                        // Apply RoPE to q_pe
                        for i in 0..rope_half {
                            let x1 = tmp[i];
                            let x2 = tmp[rope_half + i];
                            graph.mla_q_full[base + i] = x1 * cos[i] - x2 * sin[i];
                            graph.mla_q_full[base + rope_half + i] = x2 * cos[i] + x1 * sin[i];
                        }
                    }

                    // Apply RoPE to k_pe (single, shared across heads)
                    {
                        let kpe = &mut graph.mla_kv_out[klr..klr + qk_rd];
                        for i in 0..rope_half {
                            let x1 = kpe[i];
                            let x2 = kpe[rope_half + i];
                            kpe[i] = x1 * cos[i] - x2 * sin[i];
                            kpe[rope_half + i] = x2 * cos[i] + x1 * sin[i];
                        }
                    }

                    // ── Step 4: Absorb w_kc into query ──
                    // For each head: q_absorbed[h, :klr] = q_nope[h, :qk_nd] @ w_kc[h, :qk_nd, :klr]
                    // w_kc layout: [num_heads, qk_nope_dim, kv_lora_rank]
                    for h in 0..nh {
                        let q_base = h * head_dim; // q_nope starts here
                        let wkc_base = h * qk_nd * klr;
                        let out_base = h * klr;
                        for j in 0..klr {
                            let mut acc = 0.0f32;
                            for i in 0..qk_nd {
                                acc += graph.mla_q_full[q_base + i] * w_kc[wkc_base + i * klr + j];
                            }
                            graph.mla_q_absorbed[out_base + j] = acc;
                        }
                    }
                    if timing { graph.t_mla_rope += t0.elapsed().as_secs_f64(); }

                    // ── Step 5: KV cache write + Attention ──
                    let t0 = if timing { Instant::now() } else { t_step_start };

                    // Write compressed KV to cache
                    let ckv_cache: &mut [f32] = unsafe {
                        std::slice::from_raw_parts_mut(
                            graph.mla_ckv_ptrs[layer_idx] as *mut f32,
                            graph.kv_max_seq * klr)
                    };
                    let kpe_cache: &mut [f32] = unsafe {
                        std::slice::from_raw_parts_mut(
                            graph.mla_kpe_ptrs[layer_idx] as *mut f32,
                            graph.kv_max_seq * qk_rd)
                    };
                    let ckv_offset = position * klr;
                    ckv_cache[ckv_offset..ckv_offset + klr]
                        .copy_from_slice(&graph.mla_kv_compressed[..klr]);
                    let kpe_offset = position * qk_rd;
                    kpe_cache[kpe_offset..kpe_offset + qk_rd]
                        .copy_from_slice(&graph.mla_kv_out[klr..klr + qk_rd]);

                    // Attention: per head
                    // score[h,t] = dot(q_absorbed[h], ckv[t]) + dot(q_pe[h], kpe[t])
                    let seq_len = position + 1;
                    for h in 0..nh {
                        let qa_base = h * klr;
                        let qpe_base = h * head_dim + qk_nd; // q_pe in mla_q_full after RoPE
                        let score_base = h * seq_len;
                        let mut max_score = f32::NEG_INFINITY;

                        for t in 0..seq_len {
                            let mut s = 0.0f32;
                            // dot(q_absorbed, ckv)
                            let ckv_t = t * klr;
                            for i in 0..klr {
                                s += graph.mla_q_absorbed[qa_base + i] * ckv_cache[ckv_t + i];
                            }
                            // dot(q_pe, kpe)
                            let kpe_t = t * qk_rd;
                            for i in 0..qk_rd {
                                s += graph.mla_q_full[qpe_base + i] * kpe_cache[kpe_t + i];
                            }
                            s *= sm_scale;
                            graph.mla_attn_scores[score_base + t] = s;
                            if s > max_score { max_score = s; }
                        }

                        // Softmax
                        let mut sum_exp = 0.0f32;
                        for t in 0..seq_len {
                            let e = (graph.mla_attn_scores[score_base + t] - max_score).exp();
                            graph.mla_attn_scores[score_base + t] = e;
                            sum_exp += e;
                        }
                        let inv_sum = 1.0 / sum_exp;
                        for t in 0..seq_len {
                            graph.mla_attn_scores[score_base + t] *= inv_sum;
                        }

                        // Weighted sum: attn_out[h] = sum_t(weight[t] * ckv[t])
                        let out_base = h * klr;
                        for j in 0..klr {
                            let mut acc = 0.0f32;
                            for t in 0..seq_len {
                                acc += graph.mla_attn_scores[score_base + t]
                                    * ckv_cache[t * klr + j];
                            }
                            graph.mla_attn_out[out_base + j] = acc;
                        }
                    }
                    if timing { graph.t_mla_attn += t0.elapsed().as_secs_f64(); }

                    // ── Step 6: w_vc projection + o_proj ──
                    let t0 = if timing { Instant::now() } else { t_step_start };

                    // For each head: v_projected[h] = w_vc[h] @ attn_out[h]
                    // w_vc: [num_heads, v_head_dim, kv_lora_rank]
                    for h in 0..nh {
                        let wvc_base = h * vhd * klr;
                        let ao_base = h * klr;
                        let vp_base = h * vhd;
                        for o in 0..vhd {
                            let mut acc = 0.0f32;
                            for d in 0..klr {
                                acc += w_vc[wvc_base + o * klr + d]
                                    * graph.mla_attn_out[ao_base + d];
                            }
                            graph.mla_v_projected[vp_base + o] = acc;
                        }
                    }

                    // o_proj: [num_heads * v_head_dim] → [hidden_size]
                    let o_k = weights[*o_proj_wid].cols;
                    quantize_activation_int16_f32(
                        &graph.mla_v_projected[..o_k], graph.group_size,
                        &mut graph.act_int16[..o_k],
                        &mut graph.act_scales[..o_k / graph.group_size]);
                    dispatch_matmul_free(
                        &weights[*o_proj_wid],
                        &graph.act_int16[..o_k],
                        &graph.act_scales[..o_k / graph.group_size],
                        &mut graph.hidden[..hs],
                        parallel);
                    if timing { graph.t_mla_o_proj += t0.elapsed().as_secs_f64(); }
                }
            }

            // Post-attention norm
            let t0 = if timing { Instant::now() } else { t_step_start };
            unsafe {
                fused_add_rmsnorm_avx2(
                    &mut graph.hidden, &mut graph.residual,
                    &norm_weights[graph.layers[layer_idx].post_attn_norm_id],
                    eps, false, norm_bias_one);
            }
            if timing { graph.t_norm += t0.elapsed().as_secs_f64(); }

            // MLP
            match &graph.layers[layer_idx].mlp {
                DecodeMlpConfig::MoE {
                    route_id, moe_layer_idx,
                    shared_gate_up_wid, shared_down_wid, shared_gate_wid,
                } => {
                    let route_id = *route_id;
                    let moe_layer_idx = *moe_layer_idx;
                    let sgu_wid = *shared_gate_up_wid;
                    let sd_wid = *shared_down_wid;
                    let sg_wid = *shared_gate_wid;
                    let topk = graph.topk;
                    let sf = graph.scoring_func;
                    let ntp = graph.norm_topk_prob;
                    let rsf = graph.routed_scaling_factor;

                    // Routing
                    let t0 = if timing { Instant::now() } else { t_step_start };
                    let rw = &route_weights[route_id];
                    let ne = rw.num_experts;
                    let hd = rw.hidden_dim;
                    let logits_buf = &mut graph.route_logits[..ne];
                    let scores_buf = &mut graph.route_scores[..ne];
                    let corrected_buf = &mut graph.route_corrected[..ne];
                    unsafe { moe_route_matmul_avx2_parallel(&rw.data, &graph.hidden[..hd], logits_buf, ne, hd) };
                    if let Some(ref bias) = rw.bias {
                        for e in 0..ne { logits_buf[e] += bias[e]; }
                    }
                    moe_route_score_topk(
                        logits_buf, scores_buf, corrected_buf,
                        &rw.e_score_corr, sf, ntp, topk,
                        &mut graph.moe_topk_ids, &mut graph.moe_topk_weights);
                    if timing { graph.t_moe_route += t0.elapsed().as_secs_f64(); }

                    // Routed + shared experts overlapped via rayon::join.
                    // Shared expert runs on current thread while routed experts use pool.
                    let t0 = if timing { Instant::now() } else { t_step_start };
                    let has_shared = sgu_wid.is_some() && sd_wid.is_some();
                    let moe_store_arc = graph.moe_store.clone();
                    if let Some(moe_store) = moe_store_arc {
                        for j in 0..hs {
                            graph.moe_act_bf16[j] = f32_to_bf16(graph.hidden[j]);
                        }
                        let mut expert_indices = [0usize; 32];
                        let mut expert_weights_arr = [0.0f32; 32];
                        let mut n_exp = 0;
                        for i in 0..topk {
                            if graph.moe_topk_ids[i] >= 0 {
                                expert_indices[n_exp] = graph.moe_topk_ids[i] as usize;
                                expert_weights_arr[n_exp] = graph.moe_topk_weights[i];
                                n_exp += 1;
                            }
                        }
                        // SAFETY: rayon::join closures access disjoint graph fields.
                        // Routed: moe_output(w), moe_scratch(w), moe_scratch_pool(w), moe_act_bf16(r)
                        // Shared: act_int16(w), act_scales(w), mlp_gate_up(w), mlp_hidden_buf(w), shared_out(w), hidden(r)
                        let gp = &mut **graph as *mut DecodeGraph as usize;
                        let moe_par = graph.moe_parallel;
                        let gs = graph.group_size;
                        rayon::join(
                            move || unsafe {
                                let g = &mut *(gp as *mut DecodeGraph);
                                g.moe_output.fill(0.0);
                                if n_exp > 0 {
                                    let scratch = g.moe_scratch.as_mut().unwrap();
                                    let pool = &mut g.moe_scratch_pool;
                                    let mut no_shared: Option<ExpertScratch> = None;
                                    moe_forward_unified(
                                        &*moe_store, moe_layer_idx,
                                        &g.moe_act_bf16[..hs],
                                        &expert_indices[..n_exp],
                                        &expert_weights_arr[..n_exp],
                                        &mut g.moe_output,
                                        scratch, pool, &mut no_shared,
                                        moe_par, None);
                                }
                                if rsf != 1.0 {
                                    for j in 0..hs { g.moe_output[j] *= rsf; }
                                }
                            },
                            move || unsafe {
                                if !has_shared { return; }
                                let g = &mut *(gp as *mut DecodeGraph);
                                let gu_wid = sgu_wid.unwrap();
                                let dn_wid = sd_wid.unwrap();
                                let gu_w = &weights[gu_wid];
                                let k_in = gu_w.cols;
                                let n_gu = gu_w.rows;
                                let intermediate = n_gu / 2;
                                quantize_activation_int16_f32(
                                    &g.hidden[..k_in], gs,
                                    &mut g.act_int16[..k_in],
                                    &mut g.act_scales[..k_in / gs]);
                                dispatch_matmul_free(gu_w,
                                    &g.act_int16[..k_in],
                                    &g.act_scales[..k_in / gs],
                                    &mut g.mlp_gate_up[..n_gu], parallel);
                                fast_silu_mul_avx2(
                                    &g.mlp_gate_up[..intermediate],
                                    &g.mlp_gate_up[intermediate..n_gu],
                                    &mut g.mlp_hidden_buf[..intermediate],
                                    intermediate);
                                let dn_w = &weights[dn_wid];
                                let k_dn = dn_w.cols;
                                quantize_activation_int16_f32(
                                    &g.mlp_hidden_buf[..k_dn], gs,
                                    &mut g.act_int16[..k_dn],
                                    &mut g.act_scales[..k_dn / gs]);
                                dispatch_matmul_free(dn_w,
                                    &g.act_int16[..k_dn],
                                    &g.act_scales[..k_dn / gs],
                                    &mut g.shared_out[..hs], parallel);
                                if let Some(sg) = sg_wid {
                                    let sg_w = &weights[sg];
                                    let sg_k = sg_w.cols;
                                    quantize_activation_int16_f32(
                                        &g.hidden[..sg_k], gs,
                                        &mut g.act_int16[..sg_k],
                                        &mut g.act_scales[..sg_k / gs]);
                                    let mut gate_val = [0.0f32; 1];
                                    dispatch_matmul_free(sg_w,
                                        &g.act_int16[..sg_k],
                                        &g.act_scales[..sg_k / gs],
                                        &mut gate_val, parallel);
                                    let gate_sigmoid = 1.0 / (1.0 + (-gate_val[0]).exp());
                                    for j in 0..hs { g.shared_out[j] *= gate_sigmoid; }
                                }
                            },
                        );
                        if has_shared {
                            for j in 0..hs {
                                graph.hidden[j] = graph.moe_output[j] + graph.shared_out[j];
                            }
                        } else {
                            graph.hidden[..hs].copy_from_slice(&graph.moe_output[..hs]);
                        }
                    }
                    if timing { graph.t_moe_experts += t0.elapsed().as_secs_f64(); }
                }

                DecodeMlpConfig::Dense { gate_proj_wid, up_proj_wid, down_proj_wid } => {
                    let t0 = if timing { Instant::now() } else { t_step_start };
                    let gw = &weights[*gate_proj_wid];
                    let uw = &weights[*up_proj_wid];
                    let dw = &weights[*down_proj_wid];
                    let k_in = gw.cols;
                    let intermediate = gw.rows;
                    quantize_activation_int16_f32(
                        &graph.hidden[..k_in], graph.group_size,
                        &mut graph.act_int16[..k_in],
                        &mut graph.act_scales[..k_in / graph.group_size]);
                    dispatch_matmul_free(gw,
                        &graph.act_int16[..k_in],
                        &graph.act_scales[..k_in / graph.group_size],
                        &mut graph.mlp_gate_up[..intermediate], parallel);
                    dispatch_matmul_free(uw,
                        &graph.act_int16[..k_in],
                        &graph.act_scales[..k_in / graph.group_size],
                        &mut graph.mlp_gate_up[intermediate..2*intermediate], parallel);
                    // AVX2 fused SiLU(gate) * up
                    unsafe {
                        fast_silu_mul_avx2(
                            &graph.mlp_gate_up[..intermediate],
                            &graph.mlp_gate_up[intermediate..2*intermediate],
                            &mut graph.mlp_hidden_buf[..intermediate],
                            intermediate);
                    }
                    let k_dn = dw.cols;
                    quantize_activation_int16_f32(
                        &graph.mlp_hidden_buf[..k_dn], graph.group_size,
                        &mut graph.act_int16[..k_dn],
                        &mut graph.act_scales[..k_dn / graph.group_size]);
                    dispatch_matmul_free(dw,
                        &graph.act_int16[..k_dn],
                        &graph.act_scales[..k_dn / graph.group_size],
                        &mut graph.hidden[..hs], parallel);
                    if timing { graph.t_dense_mlp += t0.elapsed().as_secs_f64(); }
                }

                DecodeMlpConfig::None => {}
            }
        }

        // ── Final norm ──
        let t0 = if timing { Instant::now() } else { t_step_start };
        unsafe {
            fused_add_rmsnorm_avx2(
                &mut graph.hidden, &mut graph.residual,
                &norm_weights[graph.final_norm_id],
                eps, false, norm_bias_one);
        }
        if timing { graph.t_norm += t0.elapsed().as_secs_f64(); }

        // ── LM head ──
        let t0 = if timing { Instant::now() } else { t_step_start };
        let lm_k = weights[graph.lm_head_wid].cols;
        quantize_activation_int16_f32(
            &graph.hidden[..lm_k], graph.group_size,
            &mut graph.act_int16[..lm_k],
            &mut graph.act_scales[..lm_k / graph.group_size]);
        let output: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(output_ptr as *mut f32, graph.vocab_size)
        };
        dispatch_matmul_free(
            &weights[graph.lm_head_wid],
            &graph.act_int16[..lm_k],
            &graph.act_scales[..lm_k / graph.group_size],
            output, parallel);
        if timing { graph.t_lm_head += t0.elapsed().as_secs_f64(); }

        // ── Timing report ──
        if timing {
            graph.t_total += t_step_start.elapsed().as_secs_f64();
            graph.timing_step_count += 1;
            let n = graph.timing_step_count;
            if n % graph.timing_report_interval == 0 {
                let nf = n as f64;
                let total_ms = graph.t_total / nf * 1000.0;
                log::info!("=== CPU DECODE TIMING ({} steps, avg {:.1} ms/tok, {:.2} tok/s) ===",
                    n, total_ms, 1000.0 / total_ms);
                log::info!("  norm:         {:6.1} ms ({:4.1}%)", graph.t_norm / nf * 1000.0, graph.t_norm / graph.t_total * 100.0);
                log::info!("  la_proj:      {:6.1} ms ({:4.1}%)", graph.t_la_proj / nf * 1000.0, graph.t_la_proj / graph.t_total * 100.0);
                log::info!("  la_conv:      {:6.1} ms ({:4.1}%)", graph.t_la_conv / nf * 1000.0, graph.t_la_conv / graph.t_total * 100.0);
                log::info!("  la_recur:     {:6.1} ms ({:4.1}%)", graph.t_la_recur / nf * 1000.0, graph.t_la_recur / graph.t_total * 100.0);
                log::info!("  la_gate_norm: {:6.1} ms ({:4.1}%)", graph.t_la_gate_norm / nf * 1000.0, graph.t_la_gate_norm / graph.t_total * 100.0);
                log::info!("  la_out_proj:  {:6.1} ms ({:4.1}%)", graph.t_la_out_proj / nf * 1000.0, graph.t_la_out_proj / graph.t_total * 100.0);
                log::info!("  gqa_proj:     {:6.1} ms ({:4.1}%)", graph.t_gqa_proj / nf * 1000.0, graph.t_gqa_proj / graph.t_total * 100.0);
                log::info!("  gqa_rope:     {:6.1} ms ({:4.1}%)", graph.t_gqa_rope / nf * 1000.0, graph.t_gqa_rope / graph.t_total * 100.0);
                log::info!("  gqa_attn:     {:6.1} ms ({:4.1}%)", graph.t_gqa_attn / nf * 1000.0, graph.t_gqa_attn / graph.t_total * 100.0);
                log::info!("  gqa_o_proj:   {:6.1} ms ({:4.1}%)", graph.t_gqa_o_proj / nf * 1000.0, graph.t_gqa_o_proj / graph.t_total * 100.0);
                if graph.t_mla_proj > 0.0 {
                    log::info!("  mla_proj:     {:6.1} ms ({:4.1}%)", graph.t_mla_proj / nf * 1000.0, graph.t_mla_proj / graph.t_total * 100.0);
                    log::info!("  mla_rope:     {:6.1} ms ({:4.1}%)", graph.t_mla_rope / nf * 1000.0, graph.t_mla_rope / graph.t_total * 100.0);
                    log::info!("  mla_attn:     {:6.1} ms ({:4.1}%)", graph.t_mla_attn / nf * 1000.0, graph.t_mla_attn / graph.t_total * 100.0);
                    log::info!("  mla_o_proj:   {:6.1} ms ({:4.1}%)", graph.t_mla_o_proj / nf * 1000.0, graph.t_mla_o_proj / graph.t_total * 100.0);
                }
                log::info!("  moe_route:    {:6.1} ms ({:4.1}%)", graph.t_moe_route / nf * 1000.0, graph.t_moe_route / graph.t_total * 100.0);
                log::info!("  moe_experts:  {:6.1} ms ({:4.1}%)", graph.t_moe_experts / nf * 1000.0, graph.t_moe_experts / graph.t_total * 100.0);
                log::info!("  moe_shared:   {:6.1} ms ({:4.1}%)", graph.t_moe_shared / nf * 1000.0, graph.t_moe_shared / graph.t_total * 100.0);
                log::info!("  dense_mlp:    {:6.1} ms ({:4.1}%)", graph.t_dense_mlp / nf * 1000.0, graph.t_dense_mlp / graph.t_total * 100.0);
                log::info!("  lm_head:      {:6.1} ms ({:4.1}%)", graph.t_lm_head / nf * 1000.0, graph.t_lm_head / graph.t_total * 100.0);
                let accounted = graph.t_norm + graph.t_la_proj + graph.t_la_conv + graph.t_la_recur
                    + graph.t_la_gate_norm + graph.t_la_out_proj + graph.t_gqa_proj + graph.t_gqa_rope
                    + graph.t_gqa_attn + graph.t_gqa_o_proj
                    + graph.t_mla_proj + graph.t_mla_rope + graph.t_mla_attn + graph.t_mla_o_proj
                    + graph.t_moe_route + graph.t_moe_experts
                    + graph.t_moe_shared + graph.t_dense_mlp + graph.t_lm_head;
                let overhead = graph.t_total - accounted;
                log::info!("  overhead:     {:6.1} ms ({:4.1}%)", overhead / nf * 1000.0, overhead / graph.t_total * 100.0);
            }
        }

        Ok(())
    }

    /// Run the full decode loop in Rust: decode_step -> sample -> tokenize -> callback.
    ///
    /// This eliminates all per-token Python overhead. The only Python interaction
    /// is the callback invocation to stream tokens back (one GIL acquisition per token).
    ///
    /// Args:
    ///   first_token: Initial token ID (from prefill sampling, already emitted by caller)
    ///   start_position: Sequence position for the first decode step
    ///   max_tokens: Maximum tokens to generate
    ///   temperature: Sampling temperature (0 = greedy)
    ///   top_k: Top-k filtering (0 = disabled)
    ///   top_p: Top-p nucleus filtering (1.0 = disabled)
    ///   stop_ids: List of stop token IDs
    ///   tokenizer_path: Path to tokenizer.json
    ///   callback: Python callable(token_id: int, text: str, finish_reason: Optional[str])
    ///
    /// Returns: total tokens generated
    #[pyo3(signature = (first_token, start_position, max_tokens, temperature, top_k, top_p, stop_ids, tokenizer_path, callback))]
    pub fn generate_loop(
        &mut self,
        py: pyo3::Python<'_>,
        first_token: usize,
        start_position: usize,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        stop_ids: Vec<usize>,
        tokenizer_path: &str,
        callback: pyo3::PyObject,
    ) -> PyResult<usize> {
        use std::time::Instant;
        use tokenizers::Tokenizer;

        let graph = self.decode_graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure_decode first"))?;
        let vocab_size = graph.vocab_size;

        // Load tokenizer (one-time cost ~10ms, happens once per generation)
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to load tokenizer {}: {}", tokenizer_path, e)))?;

        // Pre-allocate logits buffer
        let mut logits = vec![0.0f32; vocab_size];
        let output_ptr = logits.as_mut_ptr() as usize;

        // Build stop set for O(1) lookup
        let stop_set: std::collections::HashSet<usize> = stop_ids.into_iter().collect();

        // RNG for sampling (xorshift64)
        let mut rng_state: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        if rng_state == 0 { rng_state = 0xDEADBEEF; }
        let mut rng_next = move || -> u64 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            rng_state
        };

        let decode_start = Instant::now();
        let mut next_token = first_token;
        let mut generated = 0usize;

        for step in 0..max_tokens {
            let pos = start_position + step;

            // decode_step: all Rust compute, no Python involvement (~133ms)
            self.decode_step(next_token, pos, output_ptr)?;

            // Sample next token (pure Rust, ~0.1ms on 152K vocab)
            next_token = sample_from_logits(
                &mut logits, vocab_size, temperature, top_k, top_p, &mut rng_next);
            generated += 1;

            // Decode token to text (pure Rust tokenizer)
            let text = tokenizer.decode(&[next_token as u32], true)
                .unwrap_or_default();

            // Check stop conditions
            let finish_reason = if stop_set.contains(&next_token) {
                Some("stop")
            } else if generated >= max_tokens {
                Some("length")
            } else {
                None
            };

            // Callback to Python (GIL already held here)
            let finished = finish_reason.is_some();
            callback.call1(py, (next_token, &text, finish_reason))?;

            if finished {
                break;
            }
        }

        let elapsed = decode_start.elapsed().as_secs_f64();
        if generated > 0 {
            let tps = generated as f64 / elapsed;
            log::info!("generate_loop: {} tokens in {:.2}s ({:.1} tok/s)",
                generated, elapsed, tps);
        }

        Ok(generated)
    }

    /// Generate tokens in a tight Rust loop, returning all token IDs.
    /// No callback, no tokenizer — pure compute. Zero Python involvement per token.
    #[pyo3(signature = (first_token, start_position, max_tokens, temperature, top_k, top_p, stop_ids))]
    pub fn generate_batch(
        &mut self,
        first_token: usize,
        start_position: usize,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        stop_ids: Vec<usize>,
    ) -> PyResult<Vec<usize>> {
        use std::time::Instant;

        let graph = self.decode_graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure_decode first"))?;
        let vocab_size = graph.vocab_size;

        // Pre-allocate logits buffer
        let mut logits = vec![0.0f32; vocab_size];
        let output_ptr = logits.as_mut_ptr() as usize;

        // Build stop set for O(1) lookup
        let stop_set: std::collections::HashSet<usize> = stop_ids.into_iter().collect();

        // RNG for sampling (xorshift64)
        let mut rng_state: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        if rng_state == 0 { rng_state = 0xDEADBEEF; }
        let mut rng_next = move || -> u64 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            rng_state
        };

        let decode_start = Instant::now();
        let mut next_token = first_token;
        let mut result = Vec::with_capacity(max_tokens);

        for step in 0..max_tokens {
            let pos = start_position + step;

            self.decode_step(next_token, pos, output_ptr)?;

            next_token = sample_from_logits(
                &mut logits, vocab_size, temperature, top_k, top_p, &mut rng_next);
            result.push(next_token);

            if stop_set.contains(&next_token) {
                break;
            }
        }

        let elapsed = decode_start.elapsed().as_secs_f64();
        if !result.is_empty() {
            let tps = result.len() as f64 / elapsed;
            log::info!("generate_batch: {} tokens in {:.2}s ({:.1} tok/s)",
                result.len(), elapsed, tps);
        }

        Ok(result)
    }
}

// ── Sampling (pure Rust, no PyTorch) ──

/// Sample a token from logits using temperature, top-k, and top-p.
fn sample_from_logits(
    logits: &mut [f32],
    vocab_size: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng_next: &mut dyn FnMut() -> u64,
) -> usize {
    // Greedy
    if temperature == 0.0 {
        let mut best_idx = 0usize;
        let mut best_val = logits[0];
        for i in 1..vocab_size {
            if logits[i] > best_val {
                best_val = logits[i];
                best_idx = i;
            }
        }
        return best_idx;
    }

    // Apply temperature
    let inv_temp = 1.0 / temperature;
    for i in 0..vocab_size {
        logits[i] *= inv_temp;
    }

    // Top-k: find k-th largest value, mask everything below it
    let effective_k = if top_k > 0 && top_k < vocab_size { top_k } else { vocab_size };

    // Build index array for sorting (only if we need top-k or top-p)
    let mut indices: Vec<usize> = (0..vocab_size).collect();
    // Partial sort: we only need top-k elements
    indices.select_nth_unstable_by(effective_k.min(vocab_size) - 1, |&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Sort the top-k portion for top-p filtering
    let top_slice = &mut indices[..effective_k];
    top_slice.sort_unstable_by(|&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Softmax over top-k (numerically stable)
    let max_logit = logits[top_slice[0]];
    let mut probs = Vec::with_capacity(effective_k);
    let mut sum = 0.0f32;
    for &idx in top_slice.iter() {
        let p = (logits[idx] - max_logit).exp();
        probs.push(p);
        sum += p;
    }
    let inv_sum = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }

    // Top-p (nucleus) filtering
    let mut cutoff = effective_k;
    if top_p < 1.0 {
        let mut cum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
    }

    // Renormalize after top-p
    if cutoff < effective_k {
        let mut new_sum = 0.0f32;
        for i in 0..cutoff {
            new_sum += probs[i];
        }
        let inv_new_sum = 1.0 / new_sum;
        for i in 0..cutoff {
            probs[i] *= inv_new_sum;
        }
    }

    // Weighted random sampling
    let r = (rng_next() as f64 / u64::MAX as f64) as f32;
    let mut cum = 0.0f32;
    for i in 0..cutoff {
        cum += probs[i];
        if r < cum {
            return top_slice[i];
        }
    }
    // Fallback (rounding)
    top_slice[cutoff - 1]
}

// ── Helper: LA conv (factored out for clarity) ──

fn decode_la_conv(
    qkvz: &[f32], ba: &[f32],
    conv_state: &mut [f32], conv_weight: &[f32],
    a_log: &[f32], dt_bias: &[f32], scale: f32,
    q_out: &mut [f32], k_out: &mut [f32],
    v_out: &mut [f32], z_out: &mut [f32],
    g_out: &mut [f32], beta_out: &mut [f32],
    mixed_qkv: &mut [f32], conv_out: &mut [f32],
    nk: usize, nv: usize, dk: usize, dv: usize, hr: usize,
    kernel_dim: usize, conv_dim: usize,
) {
    let group_dim = 2 * dk + 2 * dv * hr;
    let key_dim = nk * dk;

    // Un-interleave qkvz → mixed_qkv + z_out
    for h in 0..nk {
        let src = h * group_dim;
        mixed_qkv[h * dk..(h + 1) * dk].copy_from_slice(&qkvz[src..src + dk]);
        mixed_qkv[key_dim + h * dk..key_dim + (h + 1) * dk]
            .copy_from_slice(&qkvz[src + dk..src + 2 * dk]);
        for r in 0..hr {
            let v_head = h * hr + r;
            let v_src = src + 2 * dk + r * dv;
            let z_src = src + 2 * dk + hr * dv + r * dv;
            mixed_qkv[2 * key_dim + v_head * dv..2 * key_dim + (v_head + 1) * dv]
                .copy_from_slice(&qkvz[v_src..v_src + dv]);
            z_out[v_head * dv..(v_head + 1) * dv]
                .copy_from_slice(&qkvz[z_src..z_src + dv]);
        }
    }

    // Conv state update + depthwise conv1d (dot product only, defer SiLU)
    // Optimized for kernel_dim=4: explicit unroll avoids serial dependencies
    if kernel_dim == 4 {
        for ch in 0..conv_dim {
            let base = ch * 4;
            let s1 = conv_state[base + 1];
            let s2 = conv_state[base + 2];
            let s3 = conv_state[base + 3];
            let s4 = mixed_qkv[ch];
            conv_state[base] = s1;
            conv_state[base + 1] = s2;
            conv_state[base + 2] = s3;
            conv_state[base + 3] = s4;
            conv_out[ch] = s1 * conv_weight[base]
                + s2 * conv_weight[base + 1]
                + s3 * conv_weight[base + 2]
                + s4 * conv_weight[base + 3];
        }
    } else {
        for ch in 0..conv_dim {
            let base = ch * kernel_dim;
            for t in 0..kernel_dim - 1 {
                conv_state[base + t] = conv_state[base + t + 1];
            }
            conv_state[base + kernel_dim - 1] = mixed_qkv[ch];
            let mut dot = 0.0f32;
            for t in 0..kernel_dim {
                dot += conv_state[base + t] * conv_weight[base + t];
            }
            conv_out[ch] = dot;
        }
    }
    // Apply SiLU in bulk using AVX2
    unsafe { fast_silu_avx2(conv_out, conv_dim); }

    // Expand + L2 normalize q/k using AVX2
    unsafe {
        l2_normalize_expand_avx2(conv_out, q_out, 0, dk, nv, hr, scale);
        l2_normalize_expand_avx2(conv_out, k_out, key_dim, dk, nv, hr, 1.0);
    }

    // v: no expansion, no normalization
    v_out[..nv * dv].copy_from_slice(&conv_out[2 * key_dim..2 * key_dim + nv * dv]);

    // Gate parameters: un-interleave ba inline and compute gates
    for h in 0..nk {
        let src = h * 2 * hr;
        for r in 0..hr {
            let vh = h * hr + r;
            let b_raw = ba[src + r];
            let a_p = ba[src + hr + r];
            beta_out[vh] = 1.0 / (1.0 + (-b_raw).exp());
            let ap_dt = a_p + dt_bias[vh];
            let softplus = if ap_dt > 20.0 { ap_dt } else { (1.0 + ap_dt.exp()).ln() };
            g_out[vh] = -(a_log[vh].exp()) * softplus;
        }
    }
}

/// AVX2 L2 normalize + expand: for each value head, read from key head's src,
/// normalize, and write scaled result to dst.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_normalize_expand_avx2(
    src: &[f32], dst: &mut [f32],
    src_offset: usize, dk: usize, nv: usize, hr: usize,
    scale: f32,
) {
    use std::arch::x86_64::*;
    let dk8 = dk / 8;
    let scale_v = _mm256_set1_ps(scale);

    for vh in 0..nv {
        let kh = vh / hr;
        let s_base = src_offset + kh * dk;
        let d_base = vh * dk;

        // Sum of squares
        let mut sum_acc = _mm256_setzero_ps();
        for i in 0..dk8 {
            let v = _mm256_loadu_ps(src.as_ptr().add(s_base + i * 8));
            sum_acc = _mm256_fmadd_ps(v, v, sum_acc);
        }
        // Horizontal sum
        let hi = _mm256_extractf128_ps(sum_acc, 1);
        let lo = _mm256_castps256_ps128(sum_acc);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let s1 = _mm_add_ps(sum128, shuf);
        let s2 = _mm_add_ps(s1, _mm_movehl_ps(shuf, s1));
        let sum_sq = _mm_cvtss_f32(s2);
        // Scalar remainder
        let mut sum_sq = sum_sq;
        for i in (dk8 * 8)..dk {
            let v = src[s_base + i];
            sum_sq += v * v;
        }

        let inv_norm = if sum_sq > 0.0 { 1.0 / sum_sq.sqrt() } else { 0.0 };
        let inv_v = _mm256_mul_ps(_mm256_set1_ps(inv_norm), scale_v);

        for i in 0..dk8 {
            let v = _mm256_loadu_ps(src.as_ptr().add(s_base + i * 8));
            let scaled = _mm256_mul_ps(v, inv_v);
            _mm256_storeu_ps(dst.as_mut_ptr().add(d_base + i * 8), scaled);
        }
        for i in (dk8 * 8)..dk {
            dst[d_base + i] = src[s_base + i] * inv_norm * scale;
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn l2_normalize_expand_avx2(
    src: &[f32], dst: &mut [f32],
    src_offset: usize, dk: usize, nv: usize, hr: usize,
    scale: f32,
) {
    for vh in 0..nv {
        let kh = vh / hr;
        let s_base = src_offset + kh * dk;
        let d_base = vh * dk;
        let mut sum_sq = 0.0f32;
        for i in 0..dk { sum_sq += src[s_base + i] * src[s_base + i]; }
        let inv_norm = if sum_sq > 0.0 { 1.0 / sum_sq.sqrt() } else { 0.0 };
        for i in 0..dk { dst[d_base + i] = src[s_base + i] * inv_norm * scale; }
    }
}

/// AVX2 Gated RMSNorm + SiLU: out[i] = SiLU(z[i]) * RMSNorm(recur[i]) * weight[i]
/// Processes per-head blocks of dv elements. Eliminates scalar exp() calls.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gated_rmsnorm_silu_avx2(
    recur_out: &[f32], z_buf: &[f32], norm_weight: &[f32],
    gated_out: &mut [f32], nv: usize, dv: usize, eps: f32,
) {
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn fast_exp_avx2_inline(x: __m256) -> __m256 {
        let log2e = _mm256_set1_ps(1.4426950408889634);
        let t = _mm256_mul_ps(x, log2e);
        let n = _mm256_floor_ps(t);
        let ni = _mm256_cvtps_epi32(n);
        let f = _mm256_sub_ps(t, n);
        let c5 = _mm256_set1_ps(0.0013333558);
        let c4 = _mm256_set1_ps(0.009618129);
        let c3 = _mm256_set1_ps(0.0555041);
        let c2 = _mm256_set1_ps(0.2402265);
        let c1 = _mm256_set1_ps(0.6931472);
        let one = _mm256_set1_ps(1.0);
        let poly = _mm256_fmadd_ps(c5, f, c4);
        let poly = _mm256_fmadd_ps(poly, f, c3);
        let poly = _mm256_fmadd_ps(poly, f, c2);
        let poly = _mm256_fmadd_ps(poly, f, c1);
        let poly = _mm256_fmadd_ps(poly, f, one);
        let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23));
        _mm256_mul_ps(poly, pow2n)
    }

    let dv8 = dv / 8;

    for h in 0..nv {
        let base = h * dv;

        // RMSNorm: sum of squares
        let mut sum_acc = _mm256_setzero_ps();
        for i in 0..dv8 {
            let v = _mm256_loadu_ps(recur_out.as_ptr().add(base + i * 8));
            sum_acc = _mm256_fmadd_ps(v, v, sum_acc);
        }
        let hi = _mm256_extractf128_ps(sum_acc, 1);
        let lo = _mm256_castps256_ps128(sum_acc);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let s1 = _mm_add_ps(sum128, shuf);
        let s2 = _mm_add_ps(s1, _mm_movehl_ps(shuf, s1));
        let mut sum_sq = _mm_cvtss_f32(s2);
        for i in (dv8 * 8)..dv { sum_sq += recur_out[base + i] * recur_out[base + i]; }

        let rms = (sum_sq / dv as f32 + eps).sqrt().recip();
        let rms_v = _mm256_set1_ps(rms);

        // Fused: gated_out = SiLU(z) * (recur * rms * weight)
        let clamp_hi = _mm256_set1_ps(20.0);
        let clamp_lo = _mm256_set1_ps(-20.0);
        let one = _mm256_set1_ps(1.0);
        let two = _mm256_set1_ps(2.0);

        for i in 0..dv8 {
            let off = base + i * 8;
            let r = _mm256_loadu_ps(recur_out.as_ptr().add(off));
            let w = _mm256_loadu_ps(norm_weight.as_ptr().add(off));
            let z = _mm256_loadu_ps(z_buf.as_ptr().add(off));

            let normed = _mm256_mul_ps(_mm256_mul_ps(r, rms_v), w);

            // SiLU(z) = z * sigmoid(z)
            let neg_z = _mm256_sub_ps(_mm256_setzero_ps(), z);
            let clamped = _mm256_max_ps(_mm256_min_ps(neg_z, clamp_hi), clamp_lo);
            let exp_neg_z = fast_exp_avx2_inline(clamped);
            let denom = _mm256_add_ps(one, exp_neg_z);
            let rcp = _mm256_rcp_ps(denom);
            let sigmoid = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(denom, rcp, two));
            let silu_z = _mm256_mul_ps(z, sigmoid);

            let result = _mm256_mul_ps(silu_z, normed);
            _mm256_storeu_ps(gated_out.as_mut_ptr().add(off), result);
        }
        // Scalar remainder
        for i in (dv8 * 8)..dv {
            let j = base + i;
            let normed = recur_out[j] * rms * norm_weight[j];
            let zval = z_buf[j];
            let silu_z = zval / (1.0 + (-zval).exp());
            gated_out[j] = silu_z * normed;
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn gated_rmsnorm_silu_avx2(
    recur_out: &[f32], z_buf: &[f32], norm_weight: &[f32],
    gated_out: &mut [f32], nv: usize, dv: usize, eps: f32,
) {
    for h in 0..nv {
        let base = h * dv;
        let mut sum_sq = 0.0f32;
        for j in 0..dv { sum_sq += recur_out[base + j] * recur_out[base + j]; }
        let rms = (sum_sq / dv as f32 + eps).sqrt().recip();
        for j in 0..dv {
            let normed = recur_out[base + j] * rms * norm_weight[base + j];
            let zval = z_buf[base + j];
            let silu_z = zval / (1.0 + (-zval).exp());
            gated_out[base + j] = silu_z * normed;
        }
    }
}

/// MoE routing: score + topk (factored out from decode_step).
fn moe_route_score_topk(
    logits: &mut [f32],
    scores: &mut [f32],
    corrected: &mut [f32],
    e_score_corr: &Option<Vec<f32>>,
    scoring_func: u8,
    norm_topk_prob: bool,
    topk: usize,
    topk_ids: &mut [i32],
    topk_weights: &mut [f32],
) {
    let ne = logits.len();
    match scoring_func {
        0 => {
            // sigmoid
            for e in 0..ne { scores[e] = 1.0 / (1.0 + (-logits[e]).exp()); }
            if let Some(ref esc) = e_score_corr {
                for e in 0..ne { corrected[e] = scores[e] + esc[e]; }
                topk_indices(&corrected[..ne], topk, topk_ids);
            } else {
                topk_indices(scores, topk, topk_ids);
            }
            for i in 0..topk { topk_weights[i] = scores[topk_ids[i] as usize]; }
            if norm_topk_prob {
                let sum: f32 = topk_weights[..topk].iter().sum();
                if sum > 0.0 { for w in topk_weights[..topk].iter_mut() { *w /= sum; } }
            }
        }
        1 => {
            let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for e in 0..ne { scores[e] = (logits[e] - max_l).exp(); sum_exp += scores[e]; }
            let inv = 1.0 / sum_exp;
            for e in 0..ne { scores[e] *= inv; }
            if let Some(ref esc) = e_score_corr {
                for e in 0..ne { corrected[e] = scores[e] + esc[e]; }
                topk_indices(&corrected[..ne], topk, topk_ids);
            } else {
                topk_indices(scores, topk, topk_ids);
            }
            for i in 0..topk { topk_weights[i] = scores[topk_ids[i] as usize]; }
            if norm_topk_prob {
                let sum: f32 = topk_weights[..topk].iter().sum();
                if sum > 0.0 { for w in topk_weights[..topk].iter_mut() { *w /= sum; } }
            }
        }
        2 => {
            topk_indices(logits, topk, topk_ids);
            let max_l = (0..topk).map(|i| logits[topk_ids[i] as usize])
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for i in 0..topk {
                let v = (logits[topk_ids[i] as usize] - max_l).exp();
                topk_weights[i] = v;
                sum_exp += v;
            }
            let inv = 1.0 / sum_exp;
            for i in 0..topk { topk_weights[i] *= inv; }
        }
        _ => {}
    }
}

/// AVX2 GQA attention compute (M=1 decode).
///
/// For each query head: compute dot-product scores against KV cache,
/// softmax, weighted sum of values.
///
/// If gated=true, the gate values are expected in gqa_attn_out[nh*hd..2*nh*hd]
/// (caller places them there). The function writes attention output to
/// gqa_attn_out[0..nh*hd] and applies sigmoid gating.
#[target_feature(enable = "avx2,fma")]
unsafe fn gqa_attention_compute_avx2(
    q: &[f32],              // [num_heads * head_dim]
    k_cache: &[f32],        // [max_seq * kv_heads * head_dim]
    v_cache: &[f32],        // [max_seq * kv_heads * head_dim]
    scores: &mut [f32],     // scratch [num_heads * seq_len]
    attn_out: &mut [f32],   // output [num_heads * head_dim * (2 if gated)]
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    seq_len: usize,
    sm_scale: f32,
    gated: bool,
) {
    use std::arch::x86_64::*;
    let kv_stride = num_kv_heads * head_dim;
    let num_groups = num_heads / num_kv_heads;
    let hd8 = head_dim / 8;

    // Save gate values before overwriting attn_out (for gated attention)
    // Gate is in attn_out[0..num_heads*head_dim] from the caller's rearrangement
    let gate_buf: Vec<f32> = if gated {
        attn_out[..num_heads * head_dim].to_vec()
    } else {
        Vec::new()
    };

    for h in 0..num_heads {
        let kv_h = h / num_groups;
        let q_base = h * head_dim;
        let s_base = h * seq_len;
        let o_base = h * head_dim;

        // Compute scores
        for s in 0..seq_len {
            let k_offset = s * kv_stride + kv_h * head_dim;
            let mut acc = _mm256_setzero_ps();
            for b in 0..hd8 {
                let qv = _mm256_loadu_ps(q.as_ptr().add(q_base + b * 8));
                let kv = _mm256_loadu_ps(k_cache.as_ptr().add(k_offset + b * 8));
                acc = _mm256_fmadd_ps(qv, kv, acc);
            }
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let s4 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(s4);
            let s2 = _mm_add_ps(s4, shuf);
            let hi2 = _mm_movehl_ps(s2, s2);
            let s1 = _mm_add_ss(s2, hi2);
            scores[s_base + s] = _mm_cvtss_f32(s1) * sm_scale;
        }

        // Softmax
        let sc = &mut scores[s_base..s_base + seq_len];
        let max_s = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for v in sc.iter_mut() {
            *v = (*v - max_s).exp();
            sum_exp += *v;
        }
        let inv = 1.0 / sum_exp;
        for v in sc.iter_mut() { *v *= inv; }

        // Weighted sum
        for b in 0..hd8 {
            _mm256_storeu_ps(attn_out.as_mut_ptr().add(o_base + b * 8), _mm256_setzero_ps());
        }
        for s in 0..seq_len {
            let w = _mm256_set1_ps(scores[s_base + s]);
            let v_offset = s * kv_stride + kv_h * head_dim;
            for b in 0..hd8 {
                let vv = _mm256_loadu_ps(v_cache.as_ptr().add(v_offset + b * 8));
                let out_p = attn_out.as_mut_ptr().add(o_base + b * 8);
                let cur = _mm256_loadu_ps(out_p);
                _mm256_storeu_ps(out_p, _mm256_fmadd_ps(w, vv, cur));
            }
        }
    }

    // Gated attention: attn_out *= sigmoid(gate)
    if gated {
        let size = num_heads * head_dim;
        for i in 0..size {
            let g = gate_buf[i];
            let sig = 1.0 / (1.0 + (-g).exp());
            attn_out[i] *= sig;
        }
    }
}

// ── Synthetic decode benchmark ──

/// Fast deterministic PRNG for filling benchmark buffers with varied data.
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self { Self(if seed == 0 { 0xDEADBEEF } else { seed }) }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    #[inline]
    fn next_u32(&mut self) -> u32 { self.next_u64() as u32 }
    /// Random f32 in [-scale, scale]
    #[inline]
    fn next_f32(&mut self, scale: f32) -> f32 {
        let bits = self.next_u64();
        // Map to [-1, 1] then scale
        (bits as i64 as f64 / i64::MAX as f64) as f32 * scale
    }
}

/// Fill u32 slice with random data (every element, no CoW zero pages).
fn fill_random_u32(v: &mut [u32], rng: &mut Xorshift64) {
    for val in v.iter_mut() {
        *val = rng.next_u32();
    }
}

/// Fill u16 slice with random BF16-range scale values (~0.001 to ~0.1).
fn fill_random_scales_u16(v: &mut [u16], rng: &mut Xorshift64) {
    for val in v.iter_mut() {
        // Generate random f32 in [0.005, 0.05], convert to BF16 (upper 16 bits)
        let f = 0.005 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.045;
        *val = (f.to_bits() >> 16) as u16;
    }
}

/// Fill f32 slice with random values in [-scale, scale].
fn fill_random_f32(v: &mut [f32], rng: &mut Xorshift64, scale: f32) {
    for val in v.iter_mut() {
        *val = rng.next_f32(scale);
    }
}

/// Hint the OS to use transparent huge pages for a large allocation.
#[cfg(target_os = "linux")]
fn hint_hugepages<T>(v: &mut [T]) {
    if v.is_empty() { return; }
    let ptr = v.as_mut_ptr() as *mut libc::c_void;
    let len = v.len() * std::mem::size_of::<T>();
    unsafe { libc::madvise(ptr, len, libc::MADV_HUGEPAGE); }
}

fn fake_transposed_weight(rows: usize, cols: usize, group_size: usize, num_bits: u8, rng: &mut Xorshift64) -> TransposedWeight {
    let mut packed = if num_bits == 4 {
        let pk = cols / 8;
        vec![0u32; pk * rows]
    } else {
        let byte_count = cols * rows;
        let u32_count = (byte_count + 3) / 4;
        vec![0u32; u32_count]
    };
    fill_random_u32(&mut packed, rng);
    let num_groups = cols / group_size;
    let mut scales = vec![0u16; num_groups * rows];
    fill_random_scales_u16(&mut scales, rng);
    TransposedWeight { packed, scales, rows, cols, group_size, num_bits, tiled: false }
}

/// Synthetic decode benchmark — measures decode_step speed without loading a real model.
///
/// Reads config.json to get model dimensions, allocates fake weights matching the
/// exact layout and sizes, then runs decode_step in a loop. Memory access patterns
/// are identical to real inference (same buffer sizes, same DRAM access).
///
/// Args:
///   config_path: Path to model's config.json
///   num_steps: Number of decode steps to run
///   warmup: Number of warmup steps before timing
///   timing: Enable per-component timing (KRASIS_CPU_DECODE_TIMING)
///   num_bits: Weight quantization (4 or 8)
#[pyfunction]
#[pyo3(signature = (config_path, num_steps=100, warmup=5, timing=false, num_bits=4, max_experts=0, num_threads=20, tiled=true))]
pub fn bench_decode_synthetic(
    config_path: &str,
    num_steps: usize,
    warmup: usize,
    timing: bool,
    num_bits: u8,
    max_experts: usize,
    num_threads: usize,
    tiled: bool,
) -> PyResult<()> {
    use std::time::Instant;
    use crate::weights::{ModelConfig, UnifiedExpertWeights};

    // Enable transparent huge pages for this process.
    // Some environments (tmux, systemd) inherit PR_SET_THP_DISABLE=1
    // which silently prevents all THP allocation despite madvise succeeding.
    #[cfg(target_os = "linux")]
    unsafe {
        if libc::prctl(42, 0, 0, 0, 0) == 1 { // PR_GET_THP_DISABLE
            libc::prctl(41, 0, 0, 0, 0); // PR_SET_THP_DISABLE = 0
            eprintln!("THP re-enabled (was inherited disabled)");
        }
    }

    // Configure rayon thread pool (must match real model's thread count)
    // NOTE: Both sequential pinning (cores 0-19) and CCD-spread pinning tested.
    // Both HURT performance. Sequential concentrates on 2-3 MCs. Spread causes
    // cross-fabric latency. Linux scheduler's natural balancing is best.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global();
    eprintln!("=== Synthetic Decode Benchmark (rayon: {} threads) ===", num_threads);

    // ── 1. Parse config.json ──
    let config_str = std::fs::read_to_string(config_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to read {}: {}", config_path, e)))?;
    let raw: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

    // Use text_config if present (VL wrapper)
    let cfg = if let Some(tc) = raw.get("text_config") { tc } else { &raw };

    let hidden_size = cfg["hidden_size"].as_u64().unwrap() as usize;
    let num_layers = cfg["num_hidden_layers"].as_u64().unwrap() as usize;
    let vocab_size = cfg["vocab_size"].as_u64().unwrap() as usize;
    let eps = cfg.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;
    let full_attn_interval = cfg.get("full_attention_interval").and_then(|v| v.as_u64()).unwrap_or(4) as usize;

    // Linear attention params
    let nk = cfg.get("linear_num_key_heads").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let nv = cfg.get("linear_num_value_heads").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
    let dk = cfg.get("linear_key_head_dim").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
    let dv = cfg.get("linear_value_head_dim").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
    let kernel_dim = cfg.get("linear_conv_kernel_dim").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
    let hr = nv / nk;

    // GQA params
    let gqa_num_heads = cfg.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
    let gqa_num_kv_heads = cfg.get("num_key_value_heads").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
    let gqa_head_dim = cfg.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(256) as usize;

    // MoE params
    let num_experts = cfg.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(512) as usize;
    let topk = cfg.get("num_experts_per_tok").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let moe_intermediate = cfg.get("moe_intermediate_size").and_then(|v| v.as_u64()).unwrap_or(512) as usize;
    let shared_intermediate = cfg.get("shared_expert_intermediate_size").and_then(|v| v.as_u64()).unwrap_or(moe_intermediate as u64) as usize;
    let norm_topk_prob = cfg.get("norm_topk_prob").and_then(|v| v.as_bool()).unwrap_or(true);

    // Detect gated attention
    let partial_rotary = cfg.get("partial_rotary_factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let gqa_gated = partial_rotary < 1.0; // QCN uses 0.25 → gated

    // Norm bias one (Qwen3Next uses (1+w)*x)
    let model_type = cfg.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    let norm_bias_one = model_type.contains("qwen3_next");

    let group_size = 128usize;
    let scale = 1.0 / (dk as f32).sqrt();

    // Derived LA dimensions
    let key_dim = nk * dk;
    let value_dim = nv * dv;
    let conv_dim = key_dim * 2 + value_dim;
    let group_dim = 2 * dk + 2 * dv * hr;
    let qkvz_dim = nk * group_dim;
    let ba_dim = nk * 2 * hr;

    // max_experts=0 means use full count from config
    let alloc_experts = if max_experts > 0 && max_experts < num_experts {
        max_experts
    } else {
        num_experts
    };

    eprintln!("Model: hidden={}, layers={}, vocab={}", hidden_size, num_layers, vocab_size);
    eprintln!("LA: nk={}, nv={}, dk={}, dv={}, hr={}, conv_dim={}, qkvz_dim={}", nk, nv, dk, dv, hr, conv_dim, qkvz_dim);
    eprintln!("GQA: heads={}, kv_heads={}, head_dim={}, gated={}", gqa_num_heads, gqa_num_kv_heads, gqa_head_dim, gqa_gated);
    eprintln!("MoE: experts={}, topk={}, intermediate={}, shared={} (alloc={})", num_experts, topk, moe_intermediate, shared_intermediate, alloc_experts);
    eprintln!("Quantization: INT{}, group_size={}", num_bits, group_size);

    // ── 2. Allocate ALL weights in contiguous mmap (matches real model's mmap'd layout) ──
    let gen_start = Instant::now();

    // Pre-calculate total packed u32 and scales u16 for ALL weights (non-expert + expert)
    let packed_count_for = |rows: usize, cols: usize| -> usize {
        if num_bits == 4 { (cols / 8) * rows } else { (cols * rows + 3) / 4 }
    };
    let scales_count_for = |rows: usize, cols: usize| -> usize {
        (cols / group_size) * rows
    };
    let mut total_packed_u32 = 0usize;
    let mut total_scales_u16 = 0usize;

    // lm_head
    total_packed_u32 += packed_count_for(vocab_size, hidden_size);
    total_scales_u16 += scales_count_for(vocab_size, hidden_size);
    // Per-layer projection weights
    for layer_idx in 0..num_layers {
        let is_gqa = full_attn_interval > 0 && (layer_idx + 1) % full_attn_interval == 0;
        if is_gqa {
            let q_out = if gqa_gated { gqa_num_heads * gqa_head_dim * 2 } else { gqa_num_heads * gqa_head_dim };
            let kv_out = gqa_num_kv_heads * gqa_head_dim;
            for &(r, c) in &[(q_out, hidden_size), (kv_out, hidden_size), (kv_out, hidden_size),
                             (hidden_size, gqa_num_heads * gqa_head_dim),
                             (q_out + kv_out + kv_out, hidden_size)] { // fused QKV
                total_packed_u32 += packed_count_for(r, c);
                total_scales_u16 += scales_count_for(r, c);
            }
        } else {
            for &(r, c) in &[(qkvz_dim, hidden_size), (ba_dim, hidden_size), (hidden_size, nv * dv)] {
                total_packed_u32 += packed_count_for(r, c);
                total_scales_u16 += scales_count_for(r, c);
            }
        }
        for &(r, c) in &[(2 * shared_intermediate, hidden_size), (hidden_size, shared_intermediate), (1, hidden_size)] {
            total_packed_u32 += packed_count_for(r, c);
            total_scales_u16 += scales_count_for(r, c);
        }
    }
    // Expert weights
    let w13_packed_per = packed_count_for(2 * moe_intermediate, hidden_size);
    let w13_scales_per = scales_count_for(2 * moe_intermediate, hidden_size);
    let w2_packed_per = packed_count_for(hidden_size, moe_intermediate);
    let w2_scales_per = scales_count_for(hidden_size, moe_intermediate);
    total_packed_u32 += num_layers * alloc_experts * (w13_packed_per + w2_packed_per);
    total_scales_u16 += num_layers * alloc_experts * (w13_scales_per + w2_scales_per);

    // Allocate two contiguous mmap regions (like real model's mmap'd safetensors)
    let packed_mmap_bytes = total_packed_u32 * 4;
    let scales_mmap_bytes = total_scales_u16 * 2;
    eprintln!("Allocating contiguous mmap: {:.1} GB packed + {:.1} MB scales",
        packed_mmap_bytes as f64 / 1e9, scales_mmap_bytes as f64 / 1e6);

    // NOTE: Do NOT use MAP_POPULATE here — it pre-faults all pages as 4K before
    // madvise can request huge pages. Instead: mmap lazy, madvise HUGEPAGE, then
    // the random data fill below faults pages in as 2MB transparent huge pages.
    let packed_base = unsafe {
        libc::mmap(std::ptr::null_mut(), packed_mmap_bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1, 0)
    };
    if packed_base == libc::MAP_FAILED {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("mmap failed for packed weights"));
    }
    let scales_base = unsafe {
        libc::mmap(std::ptr::null_mut(), scales_mmap_bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1, 0)
    };
    if scales_base == libc::MAP_FAILED {
        unsafe { libc::munmap(packed_base, packed_mmap_bytes); }
        return Err(pyo3::exceptions::PyRuntimeError::new_err("mmap failed for scales"));
    }
    unsafe {
        libc::madvise(packed_base, packed_mmap_bytes, libc::MADV_HUGEPAGE);
        libc::madvise(scales_base, scales_mmap_bytes, libc::MADV_HUGEPAGE);
    }

    // Fill entire mmap regions with random data before slicing
    let mut rng = Xorshift64::new(0x12345678ABCDEF01);
    eprintln!("Pre-generating random weight data...");
    let packed_slice = unsafe { std::slice::from_raw_parts_mut(packed_base as *mut u32, total_packed_u32) };
    fill_random_u32(packed_slice, &mut rng);
    let scales_slice = unsafe { std::slice::from_raw_parts_mut(scales_base as *mut u16, total_scales_u16) };
    fill_random_scales_u16(scales_slice, &mut rng);

    // Offset trackers into the mmap regions
    let mut p_off = 0usize; // packed offset in u32 units
    let mut s_off = 0usize; // scales offset in u16 units

    let mut store = CpuDecodeStore::new(group_size, true, norm_bias_one);

    // Helper: create TransposedWeight from mmap slice (zero-copy).
    // SAFETY: All returned Vecs MUST be defused (forgotten) before munmap.
    let mut mmap_weight = |rows: usize, cols: usize| -> usize {
        let pk = if num_bits == 4 { (cols / 8) * rows } else { (cols * rows + 3) / 4 };
        let sc = (cols / group_size) * rows;
        let packed = unsafe { Vec::from_raw_parts((packed_base as *mut u32).add(p_off), pk, pk) };
        p_off += pk;
        let scales = unsafe { Vec::from_raw_parts((scales_base as *mut u16).add(s_off), sc, sc) };
        s_off += sc;
        let id = store.weights.len();
        store.weights.push(TransposedWeight { packed, scales, rows, cols, group_size, num_bits, tiled: false });
        id
    };

    // Norm weights (2 per layer + 1 final) — small random perturbations around 1.0
    let mut norm_ids = Vec::new();
    for _ in 0..(num_layers * 2 + 1) {
        let id = store.norm_weights.len();
        // Norm weights: small values around 0 for (1+w)*x style, or around 1 for w*x style
        let mut nw = vec![0.0f32; hidden_size];
        if norm_bias_one {
            fill_random_f32(&mut nw, &mut rng, 0.02); // small perturbations around 0
        } else {
            for v in nw.iter_mut() { *v = 1.0 + rng.next_f32(0.02); }
        }
        store.norm_weights.push(nw);
        norm_ids.push(id);
    }
    let final_norm_id = norm_ids[num_layers * 2];

    // lm_head weight
    let lm_head_wid = mmap_weight(vocab_size, hidden_size);

    // Count layer types
    let mut n_la = 0usize;
    let mut n_gqa = 0usize;

    // ── 3. Configure decode graph ──
    // Scoring func: 0 = softmax (QCN uses softmax)
    store.decode_graph = Some(Box::new(DecodeGraph {
        hidden_size,
        eps,
        final_norm_id,
        lm_head_wid,
        vocab_size,
        routed_scaling_factor: 1.0,
        scoring_func: 0,
        topk,
        norm_topk_prob,
        parallel: true,
        layers: Vec::with_capacity(num_layers),
        embedding_ptr: 0, // set below
        rope_cos_ptr: 0,
        rope_sin_ptr: 0,
        rope_half_dim: 0,
        max_rope_seq: 0,
        seq_len: 0,
        kv_max_seq: 0,
        kv_k_ptrs: vec![0; num_layers],
        kv_v_ptrs: vec![0; num_layers],
        conv_state_ptrs: vec![0; num_layers],
        recur_state_ptrs: vec![0; num_layers],
        hidden: vec![0.0; hidden_size],
        residual: vec![0.0; hidden_size],
        la_qkvz_buf: Vec::new(),
        la_ba_buf: Vec::new(),
        la_q_buf: Vec::new(),
        la_k_buf: Vec::new(),
        la_v_buf: Vec::new(),
        la_z_buf: Vec::new(),
        la_g_buf: Vec::new(),
        la_beta_buf: Vec::new(),
        la_recur_out: Vec::new(),
        la_gated_out: Vec::new(),
        la_mixed_qkv: Vec::new(),
        la_conv_out: Vec::new(),
        gqa_q_buf: Vec::new(),
        gqa_k_buf: Vec::new(),
        gqa_v_buf: Vec::new(),
        gqa_qkv_buf: Vec::new(),
        gqa_scores: Vec::new(),
        gqa_attn_out: Vec::new(),
        mla_kv_out: Vec::new(), mla_kv_compressed: Vec::new(),
        mla_q_full: Vec::new(), mla_q_compressed: Vec::new(),
        mla_q_absorbed: Vec::new(), mla_attn_scores: Vec::new(),
        mla_attn_out: Vec::new(), mla_v_projected: Vec::new(),
        mla_ckv_ptrs: vec![0; num_layers], mla_kpe_ptrs: vec![0; num_layers],
        mlp_gate_up: Vec::new(),
        mlp_hidden_buf: Vec::new(),
        moe_store: None,
        moe_scratch: None,
        moe_scratch_pool: Vec::new(),
        moe_output: vec![0.0; hidden_size],
        moe_act_bf16: vec![0u16; hidden_size],
        shared_out: vec![0.0; hidden_size],
        moe_topk_ids: vec![0i32; topk.max(1)],
        moe_topk_weights: vec![0.0f32; topk.max(1)],
        moe_parallel: true,
        route_logits: Vec::new(),
        route_scores: Vec::new(),
        route_corrected: Vec::new(),
        act_int16: Vec::new(),
        act_scales: Vec::new(),
        group_size,
        timing_enabled: timing,
        timing_step_count: 0,
        timing_report_interval: 20,
        t_norm: 0.0, t_la_proj: 0.0, t_la_conv: 0.0, t_la_recur: 0.0,
        t_la_gate_norm: 0.0, t_la_out_proj: 0.0,
        t_gqa_proj: 0.0, t_gqa_rope: 0.0, t_gqa_attn: 0.0, t_gqa_o_proj: 0.0,
        t_mla_proj: 0.0, t_mla_rope: 0.0, t_mla_attn: 0.0, t_mla_o_proj: 0.0,
        t_moe_route: 0.0, t_moe_experts: 0.0, t_moe_shared: 0.0,
        t_dense_mlp: 0.0, t_lm_head: 0.0, t_total: 0.0,
    }));

    // ── 4. Add layers ──
    for layer_idx in 0..num_layers {
        let input_norm_id = norm_ids[layer_idx * 2];
        let post_attn_norm_id = norm_ids[layer_idx * 2 + 1];
        let is_gqa = full_attn_interval > 0 && (layer_idx + 1) % full_attn_interval == 0;

        if is_gqa {
            // GQA layer
            let q_out = if gqa_gated { gqa_num_heads * gqa_head_dim * 2 } else { gqa_num_heads * gqa_head_dim };
            let kv_out = gqa_num_kv_heads * gqa_head_dim;
            let q_proj_wid = mmap_weight(q_out, hidden_size);
            let k_proj_wid = mmap_weight(kv_out, hidden_size);
            let v_proj_wid = mmap_weight(kv_out, hidden_size);
            let o_proj_wid = mmap_weight(hidden_size, gqa_num_heads * gqa_head_dim);
            // Fused Q+K+V weight: one big [q_out+kv_out+kv_out, hidden_size] matrix
            let fused_qkv_wid = mmap_weight(q_out + kv_out + kv_out, hidden_size);
            let sm_scale = 1.0 / (gqa_head_dim as f32).sqrt();

            // Random norm weights around 1.0
            let mut qn = vec![0.0f32; gqa_num_heads * gqa_head_dim];
            for v in qn.iter_mut() { *v = 1.0 + rng.next_f32(0.02); }
            let mut kn = vec![0.0f32; gqa_num_kv_heads * gqa_head_dim];
            for v in kn.iter_mut() { *v = 1.0 + rng.next_f32(0.02); }

            let g = store.decode_graph.as_mut().unwrap();
            g.layers.push(DecodeLayer {
                input_norm_id,
                post_attn_norm_id,
                attn: DecodeAttnConfig::GQA {
                    q_proj_wid,
                    k_proj_wid,
                    v_proj_wid,
                    o_proj_wid,
                    q_norm: Some(qn),
                    k_norm: Some(kn),
                    gated: gqa_gated,
                    num_heads: gqa_num_heads,
                    num_kv_heads: gqa_num_kv_heads,
                    head_dim: gqa_head_dim,
                    sm_scale,
                    fused_qkv_wid: Some(fused_qkv_wid),
                },
                mlp: DecodeMlpConfig::None, // set below
            });
            n_gqa += 1;
        } else {
            // Linear attention layer
            let in_proj_qkvz_wid = mmap_weight(qkvz_dim, hidden_size);
            let in_proj_ba_wid = mmap_weight(ba_dim, hidden_size);
            let out_proj_wid = mmap_weight(hidden_size, nv * dv);

            // Random conv/attention parameters
            let mut cw = vec![0.0f32; conv_dim * kernel_dim];
            fill_random_f32(&mut cw, &mut rng, 0.05);
            let mut al = vec![0.0f32; nv];
            for v in al.iter_mut() { *v = -8.0 + rng.next_f32(0.5); }
            let mut db = vec![0.0f32; nv];
            for v in db.iter_mut() { *v = 0.1 + rng.next_f32(0.02); }
            let mut nw = vec![0.0f32; nv * dv];
            for v in nw.iter_mut() { *v = 1.0 + rng.next_f32(0.02); }

            let g = store.decode_graph.as_mut().unwrap();
            g.layers.push(DecodeLayer {
                input_norm_id,
                post_attn_norm_id,
                attn: DecodeAttnConfig::LinearAttention {
                    in_proj_qkvz_wid,
                    in_proj_ba_wid,
                    out_proj_wid,
                    conv_weight: cw,
                    a_log: al,
                    dt_bias: db,
                    norm_weight: nw,
                    nk, nv, dk, dv, hr, kernel_dim, conv_dim, scale,
                },
                mlp: DecodeMlpConfig::None, // set below
            });
            n_la += 1;
        }

        // MoE on every layer (decoder_sparse_step=1 for QCN)
        let route_id = store.route_weights.len();
        // Route weight covers alloc_experts (not num_experts) so selected IDs stay in bounds
        let mut route_data = vec![0.0f32; alloc_experts * hidden_size];
        fill_random_f32(&mut route_data, &mut rng, 0.02);
        store.route_weights.push(RouteWeight {
            data: route_data,
            bias: None,
            e_score_corr: None,
            num_experts: alloc_experts,
            hidden_dim: hidden_size,
        });

        let sgu_wid = mmap_weight(2 * shared_intermediate, hidden_size);
        let sd_wid = mmap_weight(hidden_size, shared_intermediate);
        let sg_wid = mmap_weight(1, hidden_size);

        let g = store.decode_graph.as_mut().unwrap();
        g.layers[layer_idx].mlp = DecodeMlpConfig::MoE {
            route_id,
            moe_layer_idx: layer_idx,
            shared_gate_up_wid: Some(sgu_wid),
            shared_down_wid: Some(sd_wid),
            shared_gate_wid: Some(sg_wid),
        };
    }

    // Release mmap_weight closure so p_off/s_off are available for expert allocation
    drop(mmap_weight);

    eprintln!("Layers: {} LA + {} GQA = {} total", n_la, n_gqa, num_layers);

    // ── 5. Create fake MoE weight store ──
    eprintln!("Allocating {} expert weights ({} experts x {} layers)...",
        alloc_experts * num_layers, alloc_experts, num_layers);
    let alloc_start = Instant::now();

    let mut moe_store = WeightStore::new();
    moe_store.config = ModelConfig {
        hidden_size,
        moe_intermediate_size: moe_intermediate,
        n_routed_experts: alloc_experts,
        num_experts_per_tok: topk,
        num_hidden_layers: num_layers,
        first_k_dense_replace: 0,
        n_shared_experts: 1,
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        activation_alpha: 0.0,
    };
    moe_store.group_size = group_size;
    moe_store.cpu_num_bits = num_bits;

    // Build expert Vecs directly from mmap regions (zero-copy, contiguous layout)
    // SAFETY: All expert Vecs MUST be defused before munmap.
    for _layer in 0..num_layers {
        let mut layer_experts = Vec::with_capacity(alloc_experts);
        for _e in 0..alloc_experts {
            let w13_packed = unsafe { Vec::from_raw_parts((packed_base as *mut u32).add(p_off), w13_packed_per, w13_packed_per) };
            p_off += w13_packed_per;
            let w13_scales = unsafe { Vec::from_raw_parts((scales_base as *mut u16).add(s_off), w13_scales_per, w13_scales_per) };
            s_off += w13_scales_per;
            let w2_packed = unsafe { Vec::from_raw_parts((packed_base as *mut u32).add(p_off), w2_packed_per, w2_packed_per) };
            p_off += w2_packed_per;
            let w2_scales = unsafe { Vec::from_raw_parts((scales_base as *mut u16).add(s_off), w2_scales_per, w2_scales_per) };
            s_off += w2_scales_per;
            layer_experts.push(UnifiedExpertWeights {
                w13_packed, w13_scales, w2_packed, w2_scales,
                hidden_size,
                intermediate_size: moe_intermediate,
                group_size,
                num_bits,
                w2_bits: num_bits,
                gate_bias: None,
                up_bias: None,
                down_bias: None,
                tiled: false,
            });
        }
        moe_store.experts_cpu.push(layer_experts);
    }
    assert_eq!(p_off, total_packed_u32, "mmap packed offset mismatch");
    assert_eq!(s_off, total_scales_u16, "mmap scales offset mismatch");

    let expert_bytes: usize = moe_store.experts_cpu.iter()
        .flat_map(|layer| layer.iter())
        .map(|e| e.w13_packed.len() * 4 + e.w13_scales.len() * 2 + e.w2_packed.len() * 4 + e.w2_scales.len() * 2)
        .sum();
    eprintln!("Expert weights allocated: {:.1} GB in {:.1}s",
        expert_bytes as f64 / 1e9, alloc_start.elapsed().as_secs_f64());

    // ── 5b. Repack to tiled layout if requested ──
    if tiled {
        let tile_start = Instant::now();
        eprintln!("Repacking to tiled layout (TILE_N=256)...");

        // Repack non-expert TransposedWeights
        for w in store.weights.iter_mut() {
            let num_groups = w.cols / w.group_size;
            let new_packed = if w.num_bits == 4 {
                repack_tiled_int4_packed(&w.packed, w.cols, w.rows)
            } else {
                repack_tiled_int8_packed(&w.packed, w.cols, w.rows)
            };
            let new_scales = repack_tiled_scales(&w.scales, num_groups, w.rows);
            // Defuse old mmap-backed vecs before replacing
            std::mem::forget(std::mem::take(&mut w.packed));
            std::mem::forget(std::mem::take(&mut w.scales));
            w.packed = new_packed;
            w.scales = new_scales;
            w.tiled = true;
        }

        // Repack expert weights
        for layer in moe_store.experts_cpu.iter_mut() {
            for expert in layer.iter_mut() {
                let h = expert.hidden_size;
                let m = expert.intermediate_size;
                let gs = expert.group_size;
                let two_m = 2 * m;

                // w13: K=hidden_size, N=2*intermediate_size
                let new_w13_packed = if expert.num_bits == 4 {
                    repack_tiled_int4_packed(&expert.w13_packed, h, two_m)
                } else {
                    repack_tiled_int8_packed(&expert.w13_packed, h, two_m)
                };
                let w13_groups = h / gs;
                let new_w13_scales = repack_tiled_scales(&expert.w13_scales, w13_groups, two_m);
                std::mem::forget(std::mem::take(&mut expert.w13_packed));
                std::mem::forget(std::mem::take(&mut expert.w13_scales));
                expert.w13_packed = new_w13_packed;
                expert.w13_scales = new_w13_scales;

                // w2: K=intermediate_size, N=hidden_size
                let new_w2_packed = if expert.w2_bits == 4 {
                    repack_tiled_int4_packed(&expert.w2_packed, m, h)
                } else {
                    repack_tiled_int8_packed(&expert.w2_packed, m, h)
                };
                let w2_groups = m / gs;
                let new_w2_scales = repack_tiled_scales(&expert.w2_scales, w2_groups, h);
                std::mem::forget(std::mem::take(&mut expert.w2_packed));
                std::mem::forget(std::mem::take(&mut expert.w2_scales));
                expert.w2_packed = new_w2_packed;
                expert.w2_scales = new_w2_scales;

                expert.tiled = true;
            }
        }

        eprintln!("Tiled repack done in {:.1}s", tile_start.elapsed().as_secs_f64());

        // NOTE: consolidate_weights_mmap() was tested here but caused a 15% regression.
        // Non-expert weights (1.1 GB) are too small relative to expert data (40 GB)
        // for mmap+MADV_HUGEPAGE to help. Tiled layout alone provides good locality.
    }

    // Set MoE store on graph directly (bypass PyO3 interface)
    {
        let g = store.decode_graph.as_mut().unwrap();
        g.moe_scratch = Some(ExpertScratch::new(hidden_size, moe_intermediate, group_size));
        g.moe_scratch_pool = (0..topk).map(|_| ExpertScratch::new(hidden_size, moe_intermediate, group_size)).collect();
        g.moe_store = Some(Arc::new(moe_store));
    }

    // ── 6. Finalize decode graph (allocate scratch buffers) ──
    store.finalize_decode()?;

    // ── 7. Allocate and pre-fill state buffers with random data ──
    // Embedding table — random values like real learned embeddings
    let mut embedding = vec![0.0f32; vocab_size * hidden_size];
    fill_random_f32(&mut embedding, &mut rng, 0.1);
    let embedding_ptr = embedding.as_ptr() as usize;

    // Conv state: [conv_dim * kernel_dim] per LA layer
    let mut conv_states: Vec<Vec<f32>> = Vec::new();
    // Recurrent state: [nv * dk * dv] per LA layer
    let mut recur_states: Vec<Vec<f32>> = Vec::new();
    // KV cache: for GQA layers
    let kv_max_seq = 256usize; // small for bench
    let kv_stride = gqa_num_kv_heads * gqa_head_dim;
    let mut kv_k_caches: Vec<Vec<f32>> = Vec::new();
    let mut kv_v_caches: Vec<Vec<f32>> = Vec::new();
    // RoPE tables — realistic cos/sin values
    let rope_half_dim = gqa_head_dim / 2;
    let mut rope_cos = vec![0.0f32; kv_max_seq * rope_half_dim];
    let mut rope_sin = vec![0.0f32; kv_max_seq * rope_half_dim];
    for pos in 0..kv_max_seq {
        for d in 0..rope_half_dim {
            let freq = 1.0 / (10000.0f32.powf(2.0 * d as f32 / gqa_head_dim as f32));
            let angle = pos as f32 * freq;
            rope_cos[pos * rope_half_dim + d] = angle.cos();
            rope_sin[pos * rope_half_dim + d] = angle.sin();
        }
    }

    for layer_idx in 0..num_layers {
        let is_gqa = full_attn_interval > 0 && (layer_idx + 1) % full_attn_interval == 0;
        if is_gqa {
            conv_states.push(Vec::new());
            recur_states.push(Vec::new());
            let mut kk = vec![0.0f32; kv_max_seq * kv_stride];
            fill_random_f32(&mut kk, &mut rng, 0.1);
            let mut kv = vec![0.0f32; kv_max_seq * kv_stride];
            fill_random_f32(&mut kv, &mut rng, 0.1);
            kv_k_caches.push(kk);
            kv_v_caches.push(kv);
        } else {
            let mut cs = vec![0.0f32; conv_dim * kernel_dim];
            fill_random_f32(&mut cs, &mut rng, 0.1);
            let mut rs = vec![0.0f32; nv * dk * dv];
            fill_random_f32(&mut rs, &mut rng, 0.01);
            conv_states.push(cs);
            recur_states.push(rs);
            kv_k_caches.push(Vec::new());
            kv_v_caches.push(Vec::new());
        }
    }

    // Pre-fill hidden state with realistic values (as if after embedding lookup)
    {
        let g = store.decode_graph.as_mut().unwrap();
        fill_random_f32(&mut g.hidden, &mut rng, 0.5);
        fill_random_f32(&mut g.residual, &mut rng, 0.5);
    }

    // Set pointers on graph
    {
        let g = store.decode_graph.as_mut().unwrap();
        g.embedding_ptr = embedding_ptr;
        g.rope_cos_ptr = rope_cos.as_ptr() as usize;
        g.rope_sin_ptr = rope_sin.as_ptr() as usize;
        g.rope_half_dim = rope_half_dim;
        g.max_rope_seq = kv_max_seq;
        g.seq_len = 10; // simulate position 10
        g.kv_max_seq = kv_max_seq;

        for layer_idx in 0..num_layers {
            let is_gqa = full_attn_interval > 0 && (layer_idx + 1) % full_attn_interval == 0;
            if is_gqa {
                g.kv_k_ptrs[layer_idx] = kv_k_caches[layer_idx].as_ptr() as usize;
                g.kv_v_ptrs[layer_idx] = kv_v_caches[layer_idx].as_ptr() as usize;
            } else {
                g.conv_state_ptrs[layer_idx] = conv_states[layer_idx].as_ptr() as usize;
                g.recur_state_ptrs[layer_idx] = recur_states[layer_idx].as_ptr() as usize;
            }
        }

        // Resize GQA scores buffer
        let max_heads = gqa_num_heads;
        g.gqa_scores = vec![0.0f32; max_heads * kv_max_seq];
    }

    // Output buffer for logits
    let mut logits = vec![0.0f32; vocab_size];
    let output_ptr = logits.as_mut_ptr() as usize;

    // Non-MoE weight bytes
    let non_moe_bytes: usize = store.weights.iter()
        .map(|w| w.packed.len() * 4 + w.scales.len() * 2)
        .sum();
    eprintln!("Non-MoE weights: {:.1} MB", non_moe_bytes as f64 / 1e6);
    eprintln!("Total weight memory: {:.1} GB", (expert_bytes + non_moe_bytes) as f64 / 1e9);
    eprintln!("Data pre-generation: {:.1}s", gen_start.elapsed().as_secs_f64());

    // ── 8. Run benchmark ──
    eprintln!("\nRunning {} warmup + {} timed steps...", warmup, num_steps);

    // Warmup
    for i in 0..warmup {
        let position = 10 + i;
        if position >= kv_max_seq { break; }
        store.decode_step(0, position, output_ptr)?;
    }

    // Reset timing accumulators
    if timing {
        let g = store.decode_graph.as_mut().unwrap();
        g.timing_step_count = 0;
        g.t_norm = 0.0; g.t_la_proj = 0.0; g.t_la_conv = 0.0; g.t_la_recur = 0.0;
        g.t_la_gate_norm = 0.0; g.t_la_out_proj = 0.0;
        g.t_gqa_proj = 0.0; g.t_gqa_rope = 0.0; g.t_gqa_attn = 0.0; g.t_gqa_o_proj = 0.0;
        g.t_mla_proj = 0.0; g.t_mla_rope = 0.0; g.t_mla_attn = 0.0; g.t_mla_o_proj = 0.0;
        g.t_moe_route = 0.0; g.t_moe_experts = 0.0; g.t_moe_shared = 0.0;
        g.t_dense_mlp = 0.0; g.t_lm_head = 0.0; g.t_total = 0.0;
    }

    // Timed run
    let t_start = Instant::now();
    for i in 0..num_steps {
        let position = (10 + warmup + i) % (kv_max_seq - 1);
        store.decode_step(0, position, output_ptr)?;
    }
    let elapsed = t_start.elapsed().as_secs_f64();

    // ── 9. Report ──
    let ms_per_tok = elapsed / num_steps as f64 * 1000.0;
    let tok_per_sec = num_steps as f64 / elapsed;

    eprintln!("\n=== RESULTS ({} steps) ===", num_steps);
    eprintln!("Total: {:.2}s", elapsed);
    eprintln!("Per token: {:.1} ms", ms_per_tok);
    eprintln!("Speed: {:.2} tok/s", tok_per_sec);

    if timing {
        let g = store.decode_graph.as_ref().unwrap();
        let n = g.timing_step_count as f64;
        if n > 0.0 {
            let total = g.t_total / n * 1000.0;
            eprintln!("\nPer-component average (ms):");
            eprintln!("  norm:         {:6.1}", g.t_norm / n * 1000.0);
            eprintln!("  la_proj:      {:6.1}", g.t_la_proj / n * 1000.0);
            eprintln!("  la_conv:      {:6.1}", g.t_la_conv / n * 1000.0);
            eprintln!("  la_recur:     {:6.1}", g.t_la_recur / n * 1000.0);
            eprintln!("  la_gate_norm: {:6.1}", g.t_la_gate_norm / n * 1000.0);
            eprintln!("  la_out_proj:  {:6.1}", g.t_la_out_proj / n * 1000.0);
            eprintln!("  gqa_proj:     {:6.1}", g.t_gqa_proj / n * 1000.0);
            eprintln!("  gqa_rope:     {:6.1}", g.t_gqa_rope / n * 1000.0);
            eprintln!("  gqa_attn:     {:6.1}", g.t_gqa_attn / n * 1000.0);
            eprintln!("  gqa_o_proj:   {:6.1}", g.t_gqa_o_proj / n * 1000.0);
            if g.t_mla_proj > 0.0 {
                eprintln!("  mla_proj:     {:6.1}", g.t_mla_proj / n * 1000.0);
                eprintln!("  mla_rope:     {:6.1}", g.t_mla_rope / n * 1000.0);
                eprintln!("  mla_attn:     {:6.1}", g.t_mla_attn / n * 1000.0);
                eprintln!("  mla_o_proj:   {:6.1}", g.t_mla_o_proj / n * 1000.0);
            }
            eprintln!("  moe_route:    {:6.1}", g.t_moe_route / n * 1000.0);
            eprintln!("  moe_experts:  {:6.1}", g.t_moe_experts / n * 1000.0);
            eprintln!("  moe_shared:   {:6.1}", g.t_moe_shared / n * 1000.0);
            eprintln!("  dense_mlp:    {:6.1}", g.t_dense_mlp / n * 1000.0);
            eprintln!("  lm_head:      {:6.1}", g.t_lm_head / n * 1000.0);
            let accounted = g.t_norm + g.t_la_proj + g.t_la_conv + g.t_la_recur
                + g.t_la_gate_norm + g.t_la_out_proj + g.t_gqa_proj + g.t_gqa_rope
                + g.t_gqa_attn + g.t_gqa_o_proj
                + g.t_mla_proj + g.t_mla_rope + g.t_mla_attn + g.t_mla_o_proj
                + g.t_moe_route + g.t_moe_experts
                + g.t_moe_shared + g.t_dense_mlp + g.t_lm_head;
            let overhead = g.t_total - accounted;
            eprintln!("  overhead:     {:6.1}", overhead / n * 1000.0);
            eprintln!("  TOTAL:        {:6.1}", total);
        }
    }

    // ── 10. Defuse mmap-backed Vecs before munmap ──
    // All TransposedWeight and UnifiedExpertWeights Vecs point into the mmap regions.
    // We must prevent their Drop from calling dealloc on mmap'd memory.

    // Defuse non-expert weight Vecs (skip tiled — they're heap-backed, normal drop is fine)
    for w in store.weights.iter_mut() {
        if !w.tiled {
            std::mem::forget(std::mem::take(&mut w.packed));
            std::mem::forget(std::mem::take(&mut w.scales));
        }
    }

    // Defuse expert weight Vecs (inside Arc<WeightStore>)
    if let Some(ref mut g) = store.decode_graph {
        if let Some(arc) = g.moe_store.take() {
            match Arc::try_unwrap(arc) {
                Ok(mut ws) => {
                    for layer in ws.experts_cpu.iter_mut() {
                        for expert in layer.iter_mut() {
                            if !expert.tiled {
                                std::mem::forget(std::mem::take(&mut expert.w13_packed));
                                std::mem::forget(std::mem::take(&mut expert.w13_scales));
                                std::mem::forget(std::mem::take(&mut expert.w2_packed));
                                std::mem::forget(std::mem::take(&mut expert.w2_scales));
                            }
                        }
                    }
                }
                Err(_arc) => {
                    eprintln!("WARNING: Arc has multiple owners, leaking mmap to avoid UB");
                    return Ok(());
                }
            }
        }
    }

    // Keep non-mmap buffers alive until after defuse
    drop(logits);
    drop(embedding);
    drop(conv_states);
    drop(recur_states);
    drop(kv_k_caches);
    drop(kv_v_caches);
    drop(rope_cos);
    drop(rope_sin);

    // Now safe to unmap — all Vecs pointing into these regions have been defused
    unsafe {
        libc::munmap(packed_base, packed_mmap_bytes);
        libc::munmap(scales_base, scales_mmap_bytes);
    }

    Ok(())
}
