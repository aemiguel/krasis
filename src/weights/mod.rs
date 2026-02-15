//! Weight loading and format management.
//!
//! Loads expert weights from HF safetensors format, quantizes to INT4,
//! and stores in memory for CPU inference and GPU prefill.
//!
//! Disk cache: after first quantization, saves packed INT4 + scales to
//! `.krasis_cache/experts_int4_g{group_size}.bin` for instant loading.

pub mod marlin;
pub mod safetensors_io;

use crate::weights::marlin::{quantize_int4, quantize_int8, QuantizedInt4, QuantizedInt8, DEFAULT_GROUP_SIZE};
use crate::weights::safetensors_io::MmapSafetensors;
use memmap2::Mmap;
use pyo3::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use rayon::prelude::*;

/// Map GGUF quantization type to target CPU bit width for AVX2 transposed format.
///
/// Returns (target_bits, is_exact) where is_exact indicates whether the conversion
/// is lossless (exact match) or requires rounding.
pub fn gguf_type_to_cpu_bits(dtype: crate::gguf::GgmlType) -> (u8, bool) {
    use crate::gguf::GgmlType;
    match dtype {
        // Exact 4-bit matches
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q4_K => (4, true),
        // 5-bit → round down to 4 (we have no 5-bit kernel)
        GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q5_K => (4, false),
        // 6-bit → round up to 8 (lossless)
        GgmlType::Q6_K => (8, false),
        // Exact 8-bit matches
        GgmlType::Q8_0 | GgmlType::Q8_1 | GgmlType::Q8_K => (8, true),
        // High precision → best available (INT8)
        GgmlType::F16 | GgmlType::BF16 | GgmlType::F32 => (8, false),
        // Low precision → INT4
        GgmlType::Q2_K | GgmlType::Q3_K => (4, false),
    }
}

/// Model configuration (subset of config.json relevant to MoE).
///
/// Supports multiple architectures:
/// - DeepSeek V2/V3: `n_routed_experts`, `first_k_dense_replace` (flat)
/// - Kimi K2.5: same keys but nested under `text_config`
/// - Qwen3-MoE: `num_experts`, `decoder_sparse_step` (flat)
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_hidden_layers: usize,
    pub first_k_dense_replace: usize,
    /// Number of shared (always-active) experts per MoE layer. 0 = none.
    /// Shared expert intermediate_size = n_shared_experts × moe_intermediate_size.
    pub n_shared_experts: usize,
    /// Scaling factor applied to routed expert output before adding shared expert output.
    /// DeepSeek V2-Lite: 1.0, Kimi K2.5: 2.827, Qwen3: N/A (no shared experts).
    pub routed_scaling_factor: f32,
}

impl ModelConfig {
    /// Parse config.json with support for multiple MoE architectures.
    pub fn from_json(raw: &serde_json::Value) -> Result<Self, String> {
        // If there's a text_config (VL wrapper like Kimi K2.5), use that
        let cfg = if let Some(tc) = raw.get("text_config") {
            log::info!("Found text_config wrapper (VL model), using inner config");
            tc
        } else {
            raw
        };

        let hidden_size = cfg.get("hidden_size")
            .and_then(|v| v.as_u64())
            .ok_or("Missing hidden_size")? as usize;

        let moe_intermediate_size = cfg.get("moe_intermediate_size")
            .and_then(|v| v.as_u64())
            .ok_or("Missing moe_intermediate_size")? as usize;

        // n_routed_experts (DeepSeek/Kimi) OR num_experts (Qwen3)
        let n_routed_experts = cfg.get("n_routed_experts")
            .or_else(|| cfg.get("num_experts"))
            .and_then(|v| v.as_u64())
            .ok_or("Missing n_routed_experts or num_experts")? as usize;

        let num_experts_per_tok = cfg.get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .ok_or("Missing num_experts_per_tok")? as usize;

        let num_hidden_layers = cfg.get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .ok_or("Missing num_hidden_layers")? as usize;

        // first_k_dense_replace (DeepSeek/Kimi) OR derive from decoder_sparse_step (Qwen3)
        let first_k_dense_replace = if let Some(v) = cfg.get("first_k_dense_replace") {
            v.as_u64().ok_or("first_k_dense_replace not a number")? as usize
        } else if let Some(step) = cfg.get("decoder_sparse_step") {
            let step = step.as_u64().ok_or("decoder_sparse_step not a number")? as usize;
            if step <= 1 {
                // step=1 means every layer is MoE → no dense prefix
                0
            } else {
                // step>1 means interleaved MoE/dense — not yet supported
                return Err(format!(
                    "decoder_sparse_step={step} (interleaved MoE) not yet supported"
                ));
            }
        } else {
            return Err("Missing first_k_dense_replace or decoder_sparse_step".to_string());
        };

        // Shared experts (optional — Qwen3 has none)
        let n_shared_experts = cfg.get("n_shared_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let routed_scaling_factor = cfg.get("routed_scaling_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        Ok(ModelConfig {
            hidden_size,
            moe_intermediate_size,
            n_routed_experts,
            num_experts_per_tok,
            num_hidden_layers,
            first_k_dense_replace,
            n_shared_experts,
            routed_scaling_factor,
        })
    }
}

/// Quantized weight matrix — either INT4 or INT8.
pub enum QuantWeight {
    Int4(QuantizedInt4),
    Int8(QuantizedInt8),
}

impl QuantWeight {
    pub fn rows(&self) -> usize {
        match self {
            QuantWeight::Int4(q) => q.rows,
            QuantWeight::Int8(q) => q.rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            QuantWeight::Int4(q) => q.cols,
            QuantWeight::Int8(q) => q.cols,
        }
    }

    pub fn group_size(&self) -> usize {
        match self {
            QuantWeight::Int4(q) => q.group_size,
            QuantWeight::Int8(q) => q.group_size,
        }
    }

    /// Total bytes of weight data (packed + scales).
    pub fn data_bytes(&self) -> usize {
        match self {
            QuantWeight::Int4(q) => q.packed.len() * 4 + q.scales.len() * 2,
            QuantWeight::Int8(q) => q.data.len() + q.scales.len() * 2,
        }
    }

    /// Return as INT4 ref (panics if INT8).
    pub fn as_int4(&self) -> &QuantizedInt4 {
        match self {
            QuantWeight::Int4(q) => q,
            QuantWeight::Int8(_) => panic!("Expected INT4 weight, got INT8"),
        }
    }

    /// Number of bits per weight value.
    pub fn num_bits(&self) -> u8 {
        match self {
            QuantWeight::Int4(_) => 4,
            QuantWeight::Int8(_) => 8,
        }
    }
}

/// Quantized weights for a single expert (gate + up + down projections).
/// Legacy format — separate projections in [N, K/8] layout.
pub struct ExpertWeights {
    /// gate_proj: [moe_intermediate_size, hidden_size]
    pub gate: QuantWeight,
    /// up_proj: [moe_intermediate_size, hidden_size]
    pub up: QuantWeight,
    /// down_proj: [hidden_size, moe_intermediate_size]
    pub down: QuantWeight,
}

/// Raw GGUF expert weights — stored as-is from the GGUF file, no conversion.
///
/// Gate, up, down projections stored as raw GGUF block data (Q4_K, Q5_K, Q6_K, etc.).
/// The matmul kernels consume these blocks directly.
/// Note: gate/up and down may use DIFFERENT quantization types (e.g. Q4_K vs Q6_K in Q4_K_M).
pub struct GgufExpertWeights {
    /// gate_proj raw GGUF data (Q4_K blocks etc.)
    pub gate_data: Vec<u8>,
    /// up_proj raw GGUF data
    pub up_data: Vec<u8>,
    /// down_proj raw GGUF data
    pub down_data: Vec<u8>,
    /// GGML quantization type for gate/up projections
    pub gate_up_type: crate::gguf::GgmlType,
    /// GGML quantization type for down projection (may differ from gate/up)
    pub down_type: crate::gguf::GgmlType,
    /// gate/up: [intermediate_size, hidden_size]
    pub intermediate_size: usize,
    pub hidden_size: usize,
}

impl GgufExpertWeights {
    /// Total bytes of raw weight data.
    pub fn data_bytes(&self) -> usize {
        self.gate_data.len() + self.up_data.len() + self.down_data.len()
    }
}

/// Unified expert weights with combined w13 (gate+up) in a packed layout.
///
/// Used for both CPU (transposed) and GPU (Marlin) weight formats.
/// The actual data layout depends on how the weights were created:
///
/// **CPU transposed format** (`from_expert_weights` / `from_expert_weights_int8`):
///   INT4: [K/8, N] packed u32 — K outer, N contiguous → SIMD across N
///   INT8: [K, N] as i8 packed into u32 — K outer, N contiguous → SIMD across N
///
/// **GPU Marlin format** (`from_expert_weights_marlin`):
///   INT4: Marlin tile-permuted [K/8, N] → optimized for fused_marlin_moe CUDA kernel
pub struct UnifiedExpertWeights {
    /// w13 (gate+up concatenated): packed data as u32.
    /// CPU INT4: [K/8, 2*N] transposed packed. CPU INT8: [K, 2*N] i8 in u32.
    /// GPU Marlin: [K/8, 2*N] Marlin tile-permuted.
    pub w13_packed: Vec<u32>,
    /// w13 scales: [K/group_size, 2*N] as BF16.
    pub w13_scales: Vec<u16>,

    /// w2 (down): packed data as u32.
    /// CPU INT4: [K_down/8, N_down] transposed. CPU INT8: [K_down, N_down] i8 in u32.
    /// GPU Marlin: [K_down/8, N_down] Marlin tile-permuted.
    pub w2_packed: Vec<u32>,
    /// w2 scales: [K_down/group_size, N_down] as BF16.
    pub w2_scales: Vec<u16>,

    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub group_size: usize,
    /// Quantization bit width for w13 (gate+up): 4 or 8.
    pub num_bits: u8,
    /// Quantization bit width for w2 (down): 4 or 8.
    /// Usually same as num_bits, but may differ for GGUF-sourced mixed precision
    /// (e.g. Q4_K gate/up → INT4, Q6_K down → INT8).
    pub w2_bits: u8,
}

impl UnifiedExpertWeights {
    /// Convert from separate gate/up/down ExpertWeights (INT4 only) to unified transposed format.
    ///
    /// Concatenates gate+up into w13, transposes both w13 and w2 from [N, K/8] to [K/8, N].
    /// This is a pure rearrangement of packed u32 and scale u16 values — no re-quantization.
    pub fn from_expert_weights(ew: &ExpertWeights) -> Self {
        let gate = ew.gate.as_int4();
        let up = ew.up.as_int4();
        let down = ew.down.as_int4();

        let hidden = gate.cols;       // K for w13
        let intermediate = gate.rows; // N for w13 (per gate/up)
        let group_size = gate.group_size;

        let packed_k = hidden / 8;
        let num_groups = hidden / group_size;
        let two_n = 2 * intermediate;

        // w13: concatenate gate[N, K/8] + up[N, K/8] → [2*N, K/8], then transpose → [K/8, 2*N]
        let mut w13_packed = vec![0u32; packed_k * two_n];
        for k in 0..packed_k {
            for n in 0..intermediate {
                // Gate weights: first N columns
                w13_packed[k * two_n + n] = gate.packed[n * packed_k + k];
                // Up weights: next N columns
                w13_packed[k * two_n + intermediate + n] = up.packed[n * packed_k + k];
            }
        }

        // w13 scales: [2*N, K/gs] → transpose → [K/gs, 2*N]
        let mut w13_scales = vec![0u16; num_groups * two_n];
        for g in 0..num_groups {
            for n in 0..intermediate {
                w13_scales[g * two_n + n] = gate.scales[n * num_groups + g];
                w13_scales[g * two_n + intermediate + n] = up.scales[n * num_groups + g];
            }
        }

        // w2 (down): [hidden, intermediate/8] → transpose → [intermediate/8, hidden]
        let down_k = down.cols;        // intermediate_size (reduction for down)
        let down_n = down.rows;        // hidden_size (output for down)
        let down_packed_k = down_k / 8;
        let down_num_groups = down_k / group_size;

        let mut w2_packed = vec![0u32; down_packed_k * down_n];
        for k in 0..down_packed_k {
            for n in 0..down_n {
                w2_packed[k * down_n + n] = down.packed[n * down_packed_k + k];
            }
        }

        let mut w2_scales = vec![0u16; down_num_groups * down_n];
        for g in 0..down_num_groups {
            for n in 0..down_n {
                w2_scales[g * down_n + n] = down.scales[n * down_num_groups + g];
            }
        }

        UnifiedExpertWeights {
            w13_packed,
            w13_scales,
            w2_packed,
            w2_scales,
            hidden_size: hidden,
            intermediate_size: intermediate,
            group_size,
            num_bits: 4,
            w2_bits: 4,
        }
    }

    /// Convert from separate gate/up/down ExpertWeights (INT8) to unified transposed format.
    ///
    /// Concatenates gate+up into w13, transposes from [N, K] to [K, N].
    /// i8 data packed into Vec<u32> as byte container.
    pub fn from_expert_weights_int8(ew: &ExpertWeights) -> Self {
        let gate = match &ew.gate { QuantWeight::Int8(q) => q, _ => panic!("Expected INT8 gate weight") };
        let up = match &ew.up { QuantWeight::Int8(q) => q, _ => panic!("Expected INT8 up weight") };
        let down = match &ew.down { QuantWeight::Int8(q) => q, _ => panic!("Expected INT8 down weight") };

        let hidden = gate.cols;       // K for w13
        let intermediate = gate.rows; // N for w13 (per gate/up)
        let group_size = gate.group_size;
        let num_groups = hidden / group_size;
        let two_n = 2 * intermediate;

        // w13: concatenate gate[N, K] + up[N, K] → [2*N, K], then transpose → [K, 2*N]
        let w13_byte_count = hidden * two_n;
        let w13_u32_count = (w13_byte_count + 3) / 4;
        let mut w13_bytes = vec![0i8; w13_u32_count * 4]; // pad to u32 boundary

        for k in 0..hidden {
            for n in 0..intermediate {
                w13_bytes[k * two_n + n] = gate.data[n * hidden + k];
                w13_bytes[k * two_n + intermediate + n] = up.data[n * hidden + k];
            }
        }

        let w13_packed: Vec<u32> = unsafe {
            let mut v = vec![0u32; w13_u32_count];
            std::ptr::copy_nonoverlapping(
                w13_bytes.as_ptr() as *const u8,
                v.as_mut_ptr() as *mut u8,
                w13_u32_count * 4,
            );
            v
        };

        // w13 scales: [2*N, K/gs] → transpose → [K/gs, 2*N]
        let mut w13_scales = vec![0u16; num_groups * two_n];
        for g in 0..num_groups {
            for n in 0..intermediate {
                w13_scales[g * two_n + n] = gate.scales[n * num_groups + g];
                w13_scales[g * two_n + intermediate + n] = up.scales[n * num_groups + g];
            }
        }

        // w2 (down): [hidden, intermediate] → transpose → [intermediate, hidden]
        let down_k = down.cols;        // intermediate_size
        let down_n = down.rows;        // hidden_size
        let down_num_groups = down_k / group_size;

        let w2_byte_count = down_k * down_n;
        let w2_u32_count = (w2_byte_count + 3) / 4;
        let mut w2_bytes = vec![0i8; w2_u32_count * 4];

        for k in 0..down_k {
            for n in 0..down_n {
                w2_bytes[k * down_n + n] = down.data[n * down_k + k];
            }
        }

        let w2_packed: Vec<u32> = unsafe {
            let mut v = vec![0u32; w2_u32_count];
            std::ptr::copy_nonoverlapping(
                w2_bytes.as_ptr() as *const u8,
                v.as_mut_ptr() as *mut u8,
                w2_u32_count * 4,
            );
            v
        };

        let mut w2_scales = vec![0u16; down_num_groups * down_n];
        for g in 0..down_num_groups {
            for n in 0..down_n {
                w2_scales[g * down_n + n] = down.scales[n * down_num_groups + g];
            }
        }

        UnifiedExpertWeights {
            w13_packed,
            w13_scales,
            w2_packed,
            w2_scales,
            hidden_size: hidden,
            intermediate_size: intermediate,
            group_size,
            num_bits: 8,
            w2_bits: 8,
        }
    }

    /// Convert from separate gate/up/down ExpertWeights to GPU-native Marlin format.
    ///
    /// Combines gate+up into w13 [2*N, K], then Marlin-repacks both w13 and w2.
    /// Result is GPU-native Marlin INT4: same bytes on disk, in RAM, on GPU.
    pub fn from_expert_weights_marlin(ew: &ExpertWeights) -> Self {
        use crate::weights::marlin::marlin_repack;

        let gate = ew.gate.as_int4();
        let up = ew.up.as_int4();
        let down = ew.down.as_int4();

        let hidden = gate.cols;       // K for w13
        let intermediate = gate.rows; // N per gate/up
        let group_size = gate.group_size;

        // Combine gate+up into single QuantizedInt4 [2*N, K]
        let mut combined_packed = Vec::with_capacity(gate.packed.len() + up.packed.len());
        combined_packed.extend_from_slice(&gate.packed);
        combined_packed.extend_from_slice(&up.packed);

        let mut combined_scales = Vec::with_capacity(gate.scales.len() + up.scales.len());
        combined_scales.extend_from_slice(&gate.scales);
        combined_scales.extend_from_slice(&up.scales);

        let combined = QuantizedInt4 {
            packed: combined_packed,
            scales: combined_scales,
            rows: 2 * intermediate,
            cols: hidden,
            group_size,
        };

        let w13 = marlin_repack(&combined);
        let w2 = marlin_repack(down);

        UnifiedExpertWeights {
            w13_packed: w13.packed,
            w13_scales: w13.scales,
            w2_packed: w2.packed,
            w2_scales: w2.scales,
            hidden_size: hidden,
            intermediate_size: intermediate,
            group_size,
            num_bits: 4,
            w2_bits: 4,
        }
    }

    /// Convert from separate gate/up/down ExpertWeights with mixed precision.
    ///
    /// gate/up use `w13_bits`, down uses `w2_bits`. This handles GGUF models where
    /// gate/up and down projections use different quantization types (e.g. Q4_K_M:
    /// gate/up=Q4_K → INT4, down=Q6_K → INT8).
    pub fn from_expert_weights_mixed(ew: &ExpertWeights, w13_bits: u8, w2_bits: u8) -> Self {
        // Build w13 (gate+up) at w13_bits precision
        let mut result = if w13_bits == 4 {
            let gate = ew.gate.as_int4();
            let up = ew.up.as_int4();

            let hidden = gate.cols;
            let intermediate = gate.rows;
            let group_size = gate.group_size;
            let packed_k = hidden / 8;
            let num_groups = hidden / group_size;
            let two_n = 2 * intermediate;

            let mut w13_packed = vec![0u32; packed_k * two_n];
            for k in 0..packed_k {
                for n in 0..intermediate {
                    w13_packed[k * two_n + n] = gate.packed[n * packed_k + k];
                    w13_packed[k * two_n + intermediate + n] = up.packed[n * packed_k + k];
                }
            }

            let mut w13_scales = vec![0u16; num_groups * two_n];
            for g in 0..num_groups {
                for n in 0..intermediate {
                    w13_scales[g * two_n + n] = gate.scales[n * num_groups + g];
                    w13_scales[g * two_n + intermediate + n] = up.scales[n * num_groups + g];
                }
            }

            UnifiedExpertWeights {
                w13_packed,
                w13_scales,
                w2_packed: Vec::new(),
                w2_scales: Vec::new(),
                hidden_size: hidden,
                intermediate_size: intermediate,
                group_size,
                num_bits: 4,
                w2_bits,
            }
        } else {
            // INT8 gate/up
            let gate = match &ew.gate { QuantWeight::Int8(q) => q, _ => panic!("Expected INT8 gate for w13_bits=8") };
            let up = match &ew.up { QuantWeight::Int8(q) => q, _ => panic!("Expected INT8 up for w13_bits=8") };

            let hidden = gate.cols;
            let intermediate = gate.rows;
            let group_size = gate.group_size;
            let num_groups = hidden / group_size;
            let two_n = 2 * intermediate;

            let w13_byte_count = hidden * two_n;
            let w13_u32_count = (w13_byte_count + 3) / 4;
            let mut w13_bytes = vec![0i8; w13_u32_count * 4];
            for k in 0..hidden {
                for n in 0..intermediate {
                    w13_bytes[k * two_n + n] = gate.data[n * hidden + k];
                    w13_bytes[k * two_n + intermediate + n] = up.data[n * hidden + k];
                }
            }
            let w13_packed: Vec<u32> = unsafe {
                let mut v = vec![0u32; w13_u32_count];
                std::ptr::copy_nonoverlapping(
                    w13_bytes.as_ptr() as *const u8,
                    v.as_mut_ptr() as *mut u8,
                    w13_u32_count * 4,
                );
                v
            };

            let mut w13_scales = vec![0u16; num_groups * two_n];
            for g in 0..num_groups {
                for n in 0..intermediate {
                    w13_scales[g * two_n + n] = gate.scales[n * num_groups + g];
                    w13_scales[g * two_n + intermediate + n] = up.scales[n * num_groups + g];
                }
            }

            UnifiedExpertWeights {
                w13_packed,
                w13_scales,
                w2_packed: Vec::new(),
                w2_scales: Vec::new(),
                hidden_size: hidden,
                intermediate_size: intermediate,
                group_size,
                num_bits: 8,
                w2_bits,
            }
        };

        // Build w2 (down) at w2_bits precision
        if w2_bits == 4 {
            let down = ew.down.as_int4();
            let down_k = down.cols;
            let down_n = down.rows;
            let down_packed_k = down_k / 8;
            let down_num_groups = down_k / result.group_size;

            let mut w2_packed = vec![0u32; down_packed_k * down_n];
            for k in 0..down_packed_k {
                for n in 0..down_n {
                    w2_packed[k * down_n + n] = down.packed[n * down_packed_k + k];
                }
            }
            let mut w2_scales = vec![0u16; down_num_groups * down_n];
            for g in 0..down_num_groups {
                for n in 0..down_n {
                    w2_scales[g * down_n + n] = down.scales[n * down_num_groups + g];
                }
            }
            result.w2_packed = w2_packed;
            result.w2_scales = w2_scales;
        } else {
            // INT8 down
            let down = match &ew.down { QuantWeight::Int8(q) => q, _ => panic!("Expected INT8 down for w2_bits=8") };
            let down_k = down.cols;
            let down_n = down.rows;
            let down_num_groups = down_k / result.group_size;

            let w2_byte_count = down_k * down_n;
            let w2_u32_count = (w2_byte_count + 3) / 4;
            let mut w2_bytes = vec![0i8; w2_u32_count * 4];
            for k in 0..down_k {
                for n in 0..down_n {
                    w2_bytes[k * down_n + n] = down.data[n * down_k + k];
                }
            }
            let w2_packed: Vec<u32> = unsafe {
                let mut v = vec![0u32; w2_u32_count];
                std::ptr::copy_nonoverlapping(
                    w2_bytes.as_ptr() as *const u8,
                    v.as_mut_ptr() as *mut u8,
                    w2_u32_count * 4,
                );
                v
            };
            let mut w2_scales = vec![0u16; down_num_groups * down_n];
            for g in 0..down_num_groups {
                for n in 0..down_n {
                    w2_scales[g * down_n + n] = down.scales[n * down_num_groups + g];
                }
            }
            result.w2_packed = w2_packed;
            result.w2_scales = w2_scales;
        }

        result
    }

    /// Total bytes of weight data (packed + scales for w13 + w2).
    pub fn data_bytes(&self) -> usize {
        self.w13_packed.len() * 4 + self.w13_scales.len() * 2
            + self.w2_packed.len() * 4 + self.w2_scales.len() * 2
    }
}

/// Manages loaded expert weights for all MoE layers.
#[pyclass]
pub struct WeightStore {
    /// Expert weights indexed as [moe_layer_index][expert_index].
    /// moe_layer_index is 0-based within MoE layers only (skips dense layers).
    /// Legacy format (separate gate/up/down). Used for INT8 fallback path.
    pub experts: Vec<Vec<ExpertWeights>>,
    /// Shared expert weights (legacy format).
    pub shared_experts: Vec<ExpertWeights>,

    /// CPU decode weights — transposed layout, optimized for sequential access.
    /// INT4: [K/8, N] packed. INT8: [K, N] as i8 in u32.
    pub experts_cpu: Vec<Vec<UnifiedExpertWeights>>,
    /// CPU shared expert weights (transposed).
    pub shared_experts_cpu: Vec<UnifiedExpertWeights>,

    /// GPU prefill weights — Marlin tile-permuted layout for fused_marlin_moe.
    /// Always INT4 Marlin format. Empty if GPU prefill not enabled.
    pub experts_gpu: Vec<Vec<UnifiedExpertWeights>>,
    /// GPU shared expert weights (Marlin).
    pub shared_experts_gpu: Vec<UnifiedExpertWeights>,

    /// Raw GGUF expert weights — loaded as-is from GGUF file, no conversion.
    /// When populated, used for CPU decode INSTEAD of experts_cpu.
    pub experts_gguf: Vec<Vec<GgufExpertWeights>>,
    /// GGUF shared expert weights.
    pub shared_experts_gguf: Vec<GgufExpertWeights>,

    /// Model configuration.
    pub config: ModelConfig,
    /// Group size used for quantization.
    pub group_size: usize,
    /// CPU expert quantization bit width (4 or 8).
    pub cpu_num_bits: u8,
    /// GPU expert quantization bit width (4 for Marlin).
    pub gpu_num_bits: u8,
}

/// Safetensors shard index: maps tensor names to shard filenames.
#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

// ── Disk cache format ────────────────────────────────────────────────
//
// Header (64 bytes):
//   [0..4]   magic "KRAS"
//   [4..8]   version (u32 LE) — currently 1
//   [8..16]  hidden_size (u64 LE)
//   [16..24] moe_intermediate_size (u64 LE)
//   [24..32] n_routed_experts (u64 LE)
//   [32..40] num_moe_layers (u64 LE)
//   [40..48] group_size (u64 LE)
//   [48..56] config_hash (u64 LE) — FNV-1a of config.json
//   [56..64] reserved (must be 0)
//
// Body: for each (layer, expert) sequentially:
//   gate_packed [N_gate * K_gate/8 u32s as bytes]
//   gate_scales [N_gate * K_gate/group_size u16s as bytes]
//   up_packed   [same dims as gate]
//   up_scales   [same dims as gate]
//   down_packed [N_down * K_down/8 u32s as bytes]
//   down_scales [N_down * K_down/group_size u16s as bytes]

const CACHE_MAGIC: &[u8; 4] = b"KRAS";
#[allow(dead_code)]
const CACHE_VERSION: u32 = 1;
const CACHE_VERSION_MARLIN: u32 = 3;
const CACHE_VERSION_CPU: u32 = 4;
const CACHE_VERSION_CPU_GGUF: u32 = 5;
const CACHE_HEADER_SIZE: usize = 64;

/// FNV-1a hash for cache invalidation.
fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Cache file path for v1 format (separate gate/up/down, [N, K/8] layout).
#[allow(dead_code)]
fn cache_path(model_dir: &Path, num_bits: u8, group_size: usize) -> PathBuf {
    model_dir
        .join(".krasis_cache")
        .join(format!("experts_int{num_bits}_g{group_size}.bin"))
}

/// Cache file path for v3 Marlin format (GPU-native Marlin INT4).
fn cache_path_marlin(model_dir: &Path, group_size: usize) -> PathBuf {
    model_dir
        .join(".krasis_cache")
        .join(format!("experts_marlin_g{group_size}.bin"))
}

/// Cache file path for CPU-optimized transposed format (INT4 or INT8).
fn cache_path_cpu(model_dir: &Path, num_bits: u8, group_size: usize) -> PathBuf {
    model_dir
        .join(".krasis_cache")
        .join(format!("experts_cpu_int{num_bits}_g{group_size}.bin"))
}

/// Cache file path for GGUF-sourced AVX2 transposed CPU cache.
fn cache_path_gguf_avx2(model_dir: &Path, group_size: usize) -> PathBuf {
    model_dir
        .join(".krasis_cache")
        .join(format!("experts_gguf_avx2_g{group_size}.bin"))
}

/// Compute per-expert byte sizes for unified format.
/// Returns (w13_packed_bytes, w13_scales_bytes, w2_packed_bytes, w2_scales_bytes).
fn unified_expert_byte_sizes(config: &ModelConfig, group_size: usize) -> (usize, usize, usize, usize) {
    let h = config.hidden_size;
    let m = config.moe_intermediate_size;
    // w13 (gate+up concat, transposed): [K/8, 2*N] as u32
    let w13_packed_bytes = (h / 8) * (2 * m) * 4;
    let w13_scales_bytes = (h / group_size) * (2 * m) * 2;
    // w2 (down, transposed): [K_down/8, N_down] as u32
    let w2_packed_bytes = (m / 8) * h * 4;
    let w2_scales_bytes = (m / group_size) * h * 2;
    (w13_packed_bytes, w13_scales_bytes, w2_packed_bytes, w2_scales_bytes)
}

/// Compute per-expert byte sizes for CPU transposed format.
/// INT4 has same sizes as Marlin (same u32 packing, different layout).
/// INT8 has larger packed data (1 byte per element vs 0.5 for INT4).
/// Returns (w13_packed_bytes, w13_scales_bytes, w2_packed_bytes, w2_scales_bytes).
fn cpu_expert_byte_sizes(config: &ModelConfig, group_size: usize, num_bits: u8) -> (usize, usize, usize, usize) {
    let h = config.hidden_size;
    let m = config.moe_intermediate_size;
    let two_n = 2 * m;

    if num_bits == 4 {
        // INT4 transposed: same byte counts as Marlin (u32-packed nibbles)
        unified_expert_byte_sizes(config, group_size)
    } else {
        // INT8 transposed: [K, N] as i8 packed into u32 (1 byte per element)
        let w13_byte_count = h * two_n;
        let w13_packed_bytes = ((w13_byte_count + 3) / 4) * 4; // round up to u32 boundary
        let w13_scales_bytes = (h / group_size) * two_n * 2;
        let w2_byte_count = m * h;
        let w2_packed_bytes = ((w2_byte_count + 3) / 4) * 4;
        let w2_scales_bytes = (m / group_size) * h * 2;
        (w13_packed_bytes, w13_scales_bytes, w2_packed_bytes, w2_scales_bytes)
    }
}

/// Compute per-expert byte sizes for mixed-precision CPU transposed format.
/// w13_bits for gate/up, w2_bits for down — may differ.
/// Returns (w13_packed_bytes, w13_scales_bytes, w2_packed_bytes, w2_scales_bytes).
fn cpu_expert_byte_sizes_mixed(h: usize, m: usize, group_size: usize, w13_bits: u8, w2_bits: u8) -> (usize, usize, usize, usize) {
    let two_n = 2 * m;
    let w13_packed_bytes = if w13_bits == 4 {
        (h / 8) * two_n * 4
    } else {
        (((h * two_n) + 3) / 4) * 4
    };
    let w13_scales_bytes = (h / group_size) * two_n * 2;

    let w2_packed_bytes = if w2_bits == 4 {
        (m / 8) * h * 4
    } else {
        (((m * h) + 3) / 4) * 4
    };
    let w2_scales_bytes = (m / group_size) * h * 2;

    (w13_packed_bytes, w13_scales_bytes, w2_packed_bytes, w2_scales_bytes)
}

/// Expected total v5 GGUF-sourced CPU cache file size (mixed precision).
fn expected_gguf_cpu_cache_size(
    config: &ModelConfig, group_size: usize, w13_bits: u8, w2_bits: u8,
    num_moe_layers: usize, n_shared_experts: usize, shared_intermediate: usize,
) -> usize {
    let h = config.hidden_size;
    let m = config.moe_intermediate_size;

    let (w13pb, w13sb, w2pb, w2sb) = cpu_expert_byte_sizes_mixed(h, m, group_size, w13_bits, w2_bits);
    let per_routed_expert = w13pb + w13sb + w2pb + w2sb;
    let routed_total = num_moe_layers * config.n_routed_experts * per_routed_expert;

    let shared_total = if n_shared_experts > 0 {
        let (s13p, s13s, s2p, s2s) = cpu_expert_byte_sizes_mixed(
            h, shared_intermediate, group_size, w13_bits, w2_bits,
        );
        num_moe_layers * (s13p + s13s + s2p + s2s)
    } else {
        0
    };

    CACHE_HEADER_SIZE + routed_total + shared_total
}

/// Expected total CPU transposed cache file size.
fn expected_cpu_cache_size(
    config: &ModelConfig, group_size: usize, num_bits: u8, num_moe_layers: usize,
    n_shared_experts: usize, shared_intermediate: usize,
) -> usize {
    let (w13pb, w13sb, w2pb, w2sb) = cpu_expert_byte_sizes(config, group_size, num_bits);
    let per_routed_expert = w13pb + w13sb + w2pb + w2sb;
    let routed_total = num_moe_layers * config.n_routed_experts * per_routed_expert;

    let shared_total = if n_shared_experts > 0 {
        let shared_m = shared_intermediate;
        let h = config.hidden_size;
        let two_shared_n = 2 * shared_m;
        let (s_w13pb, s_w13sb, s_w2pb, s_w2sb) = if num_bits == 4 {
            (
                (h / 8) * two_shared_n * 4,
                (h / group_size) * two_shared_n * 2,
                (shared_m / 8) * h * 4,
                (shared_m / group_size) * h * 2,
            )
        } else {
            let s_w13_bytes = h * two_shared_n;
            let s_w2_bytes = shared_m * h;
            (
                ((s_w13_bytes + 3) / 4) * 4,
                (h / group_size) * two_shared_n * 2,
                ((s_w2_bytes + 3) / 4) * 4,
                (shared_m / group_size) * h * 2,
            )
        };
        num_moe_layers * (s_w13pb + s_w13sb + s_w2pb + s_w2sb)
    } else {
        0
    };

    CACHE_HEADER_SIZE + routed_total + shared_total
}

/// Expected total v2 unified cache file size.
fn expected_unified_cache_size(
    config: &ModelConfig, group_size: usize, num_moe_layers: usize,
    n_shared_experts: usize, shared_intermediate: usize,
) -> usize {
    let (w13pb, w13sb, w2pb, w2sb) = unified_expert_byte_sizes(config, group_size);
    let per_routed_expert = w13pb + w13sb + w2pb + w2sb;
    let routed_total = num_moe_layers * config.n_routed_experts * per_routed_expert;

    // Shared experts (may have different intermediate size)
    let shared_total = if n_shared_experts > 0 {
        let shared_m = shared_intermediate;
        let h = config.hidden_size;
        let s_w13p = (h / 8) * (2 * shared_m) * 4;
        let s_w13s = (h / group_size) * (2 * shared_m) * 2;
        let s_w2p = (shared_m / 8) * h * 4;
        let s_w2s = (shared_m / group_size) * h * 2;
        num_moe_layers * (s_w13p + s_w13s + s_w2p + s_w2s)
    } else {
        0
    };

    CACHE_HEADER_SIZE + routed_total + shared_total
}

/// Compute per-expert byte sizes from config (legacy v1 format).
/// Returns (gate_data_bytes, gate_scales_bytes, down_data_bytes, down_scales_bytes).
#[allow(dead_code)]
fn expert_byte_sizes(config: &ModelConfig, group_size: usize, num_bits: u8) -> (usize, usize, usize, usize) {
    let h = config.hidden_size;
    let m = config.moe_intermediate_size;

    let (gate_data_bytes, down_data_bytes) = if num_bits == 4 {
        // INT4: gate/up: [m, h] → packed [m, h/8] as u32
        // down: [h, m] → packed [h, m/8] as u32
        (m * (h / 8) * 4, h * (m / 8) * 4)
    } else {
        // INT8: gate/up: [m, h] → raw i8 [m, h]
        // down: [h, m] → raw i8 [h, m]
        (m * h, h * m)
    };

    let gate_scales_bytes = m * (h / group_size) * 2;
    let down_scales_bytes = h * (m / group_size) * 2;

    (gate_data_bytes, gate_scales_bytes, down_data_bytes, down_scales_bytes)
}

/// Expected total cache file size (legacy v1 format).
#[allow(dead_code)]
fn expected_cache_size(config: &ModelConfig, group_size: usize, num_bits: u8, num_moe_layers: usize) -> usize {
    let (gpb, gsb, dpb, dsb) = expert_byte_sizes(config, group_size, num_bits);
    let per_expert = gpb + gsb + gpb + gsb + dpb + dsb; // gate + up + down
    CACHE_HEADER_SIZE + num_moe_layers * config.n_routed_experts * per_expert
}

#[pymethods]
impl WeightStore {
    #[new]
    pub fn new() -> Self {
        WeightStore {
            experts: Vec::new(),
            shared_experts: Vec::new(),
            experts_cpu: Vec::new(),
            shared_experts_cpu: Vec::new(),
            experts_gpu: Vec::new(),
            shared_experts_gpu: Vec::new(),
            experts_gguf: Vec::new(),
            shared_experts_gguf: Vec::new(),
            config: ModelConfig {
                hidden_size: 0,
                moe_intermediate_size: 0,
                n_routed_experts: 0,
                num_experts_per_tok: 0,
                num_hidden_layers: 0,
                first_k_dense_replace: 0,
                n_shared_experts: 0,
                routed_scaling_factor: 1.0,
            },
            group_size: DEFAULT_GROUP_SIZE,
            cpu_num_bits: 4,
            gpu_num_bits: 4,
        }
    }
}

impl WeightStore {
    /// Load expert weights from a HF model directory, using disk cache if available.
    ///
    /// Loads DUAL format caches:
    ///   - GPU: Marlin INT4 cache → `experts_gpu` (for GPU prefill)
    ///   - CPU: Transposed INT4/INT8 cache → `experts_cpu` (for CPU decode)
    ///
    /// If `max_layers` is Some(n), only load n MoE layers.
    /// If `start_layer` is Some(s), start loading from MoE layer s (0-based).
    /// `cpu_num_bits`: 4 or 8 for CPU decode format.
    /// `gpu_num_bits`: 4 (Marlin, always INT4).
    pub fn load_from_hf(
        model_dir: &Path,
        group_size: usize,
        max_layers: Option<usize>,
        start_layer: Option<usize>,
        cpu_num_bits: u8,
        gpu_num_bits: u8,
    ) -> Result<Self, String> {
        let start = std::time::Instant::now();

        // Parse config.json (supports multiple MoE architectures)
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let raw_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        let config = ModelConfig::from_json(&raw_json)
            .map_err(|e| format!("Failed to extract MoE config: {e}"))?;

        log::info!(
            "Model config: hidden={}, moe_intermediate={}, experts={}, top-{}, layers={}, first_dense={}, cpu_bits={}, gpu_bits={}",
            config.hidden_size, config.moe_intermediate_size, config.n_routed_experts,
            config.num_experts_per_tok, config.num_hidden_layers, config.first_k_dense_replace,
            cpu_num_bits, gpu_num_bits,
        );

        let total_moe_layers = config.num_hidden_layers - config.first_k_dense_replace;
        let moe_start = start_layer.unwrap_or(0);
        if moe_start >= total_moe_layers {
            return Err(format!(
                "start_layer={moe_start} >= total MoE layers={total_moe_layers}"
            ));
        }
        let remaining = total_moe_layers - moe_start;
        let num_moe_layers = match max_layers {
            Some(n) => {
                let capped = n.min(remaining);
                log::info!(
                    "Partial load: MoE layers [{moe_start}..{}), {capped}/{total_moe_layers} total",
                    moe_start + capped,
                );
                capped
            }
            None => remaining,
        };
        let config_hash = fnv1a(config_str.as_bytes());

        // Detect effective group_size for pre-quantized models (needed for correct cache path)
        let effective_gs_hint = Self::detect_group_size_hint(model_dir, &config);
        let cache_gs = effective_gs_hint.unwrap_or(group_size);

        // ── Phase 1: Load/build GPU Marlin cache → experts_gpu ──
        let mut experts_gpu: Vec<Vec<UnifiedExpertWeights>> = Vec::new();
        let mut shared_experts_gpu: Vec<UnifiedExpertWeights> = Vec::new();
        let mut effective_gs = cache_gs;

        // Try loading existing Marlin cache
        let mut gpu_loaded = false;
        for try_gs in &[cache_gs, group_size, 32, 64, 128] {
            if gpu_loaded { break; }
            let try_path = cache_path_marlin(model_dir, *try_gs);
            if try_path.exists() {
                match Self::load_marlin_cache(
                    &try_path, &config, *try_gs, total_moe_layers, config_hash,
                    moe_start, num_moe_layers,
                ) {
                    Ok(store) => {
                        log::info!(
                            "Loaded GPU Marlin cache in {:.1}s (gs={}): {} layers, {} experts (+ {} shared)",
                            start.elapsed().as_secs_f64(), try_gs,
                            num_moe_layers, config.n_routed_experts, store.shared_experts_gpu.len(),
                        );
                        experts_gpu = store.experts_gpu;
                        shared_experts_gpu = store.shared_experts_gpu;
                        effective_gs = *try_gs;
                        gpu_loaded = true;
                    }
                    Err(e) => {
                        if *try_gs == cache_gs {
                            log::warn!("Marlin cache invalid (gs={}): {e}", try_gs);
                        }
                    }
                }
            }
        }

        // Build Marlin cache if not found
        if !gpu_loaded {
            let mpath = cache_path_marlin(model_dir, cache_gs);
            log::info!("No v3 Marlin cache found, building from safetensors...");
            let built_gs = Self::build_marlin_cache_locked(
                model_dir, &config, group_size, total_moe_layers, &mpath, config_hash,
            )?;
            effective_gs = built_gs;

            // Load the just-built cache
            for try_gs in &[built_gs, cache_gs, group_size, 32, 64, 128] {
                if gpu_loaded { break; }
                let try_path = cache_path_marlin(model_dir, *try_gs);
                if try_path.exists() {
                    match Self::load_marlin_cache(
                        &try_path, &config, *try_gs, total_moe_layers, config_hash,
                        moe_start, num_moe_layers,
                    ) {
                        Ok(store) => {
                            log::info!(
                                "Loaded GPU Marlin cache in {:.1}s (built gs={})",
                                start.elapsed().as_secs_f64(), try_gs,
                            );
                            experts_gpu = store.experts_gpu;
                            shared_experts_gpu = store.shared_experts_gpu;
                            effective_gs = *try_gs;
                            gpu_loaded = true;
                        }
                        Err(_) => {}
                    }
                }
            }

            if !gpu_loaded {
                log::warn!("All Marlin cache attempts failed — GPU prefill will not be available");
            }
        }

        // ── Phase 2: Load/build CPU transposed cache → experts_cpu ──
        let mut experts_cpu: Vec<Vec<UnifiedExpertWeights>> = Vec::new();
        let mut shared_experts_cpu: Vec<UnifiedExpertWeights> = Vec::new();
        let mut cpu_loaded = false;

        // Try loading existing CPU cache
        let cpu_path = cache_path_cpu(model_dir, cpu_num_bits, effective_gs);
        if cpu_path.exists() {
            match Self::load_cpu_cache(
                &cpu_path, &config, effective_gs, total_moe_layers, config_hash,
                moe_start, num_moe_layers, cpu_num_bits,
            ) {
                Ok((cpu_exp, cpu_shared)) => {
                    log::info!(
                        "Loaded CPU INT{} cache in {:.1}s: {} layers, {} experts (+ {} shared)",
                        cpu_num_bits, start.elapsed().as_secs_f64(),
                        num_moe_layers, config.n_routed_experts, cpu_shared.len(),
                    );
                    experts_cpu = cpu_exp;
                    shared_experts_cpu = cpu_shared;
                    cpu_loaded = true;
                }
                Err(e) => log::warn!("CPU cache invalid: {e}"),
            }
        }

        // Build CPU cache if not found
        if !cpu_loaded {
            log::info!("No CPU INT{} cache found, building from safetensors...", cpu_num_bits);
            let built_gs = Self::streaming_build_cpu_cache(
                model_dir, &config, group_size, total_moe_layers,
                0, &cpu_path, config_hash, cpu_num_bits,
            )?;

            // effective_gs may have been updated by the CPU build
            let actual_cpu_path = cache_path_cpu(model_dir, cpu_num_bits, built_gs);
            if built_gs != effective_gs && cpu_path != actual_cpu_path {
                // CPU build detected a different group_size — rename cache
                if cpu_path.exists() {
                    let _ = std::fs::rename(&cpu_path, &actual_cpu_path);
                }
            }
            let load_path = if actual_cpu_path.exists() { &actual_cpu_path } else { &cpu_path };

            match Self::load_cpu_cache(
                load_path, &config, built_gs, total_moe_layers, config_hash,
                moe_start, num_moe_layers, cpu_num_bits,
            ) {
                Ok((cpu_exp, cpu_shared)) => {
                    log::info!(
                        "Loaded CPU INT{} cache after build in {:.1}s",
                        cpu_num_bits, start.elapsed().as_secs_f64(),
                    );
                    experts_cpu = cpu_exp;
                    shared_experts_cpu = cpu_shared;
                    cpu_loaded = true;
                    if built_gs != effective_gs {
                        effective_gs = built_gs;
                    }
                }
                Err(e) => log::warn!("Failed to load built CPU cache: {e}"),
            }
        }

        if !cpu_loaded {
            log::warn!("CPU cache not loaded — CPU decode will use legacy path if available");
        }

        // ── Build final WeightStore ──
        let store = WeightStore {
            experts: Vec::new(),
            shared_experts: Vec::new(),
            experts_cpu,
            shared_experts_cpu,
            experts_gpu,
            shared_experts_gpu,
            experts_gguf: Vec::new(),
            shared_experts_gguf: Vec::new(),
            config: config.clone(),
            group_size: effective_gs,
            cpu_num_bits,
            gpu_num_bits,
        };

        let total_elapsed = start.elapsed();
        log::info!(
            "Dual cache loaded in {:.1}s: {} MoE layers, GPU={} CPU=INT{}{}, gs={}",
            total_elapsed.as_secs_f64(),
            num_moe_layers,
            if gpu_loaded { "Marlin" } else { "none" },
            cpu_num_bits,
            if cpu_loaded { "" } else { "(none)" },
            effective_gs,
        );

        Ok(store)
    }

    /// Load from safetensors shards and quantize to INT4/INT8 (or load pre-quantized).
    /// Returns (routed_experts, shared_experts, effective_group_size).
    /// Legacy function — used by save_cache/load_cache paths.
    ///
    /// `start_moe_layer`: 0-based offset into MoE layers (skips first N MoE layers).
    /// `num_moe_layers`: how many MoE layers to load starting from `start_moe_layer`.
    #[allow(dead_code)]
    /// `num_bits`: 4 for INT4, 8 for INT8.
    fn load_and_quantize_all(
        model_dir: &Path,
        config: &ModelConfig,
        group_size: usize,
        num_bits: u8,
        num_moe_layers: usize,
        start_moe_layer: usize,
    ) -> Result<(Vec<Vec<ExpertWeights>>, Vec<ExpertWeights>, usize), String> {
        // Parse safetensors index
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_str = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("Failed to read safetensors index: {e}"))?;
        let index: SafetensorsIndex = serde_json::from_str(&index_str)
            .map_err(|e| format!("Failed to parse safetensors index: {e}"))?;

        // Determine which shard files we actually need for our layer range.
        // Only open shards containing expert weights for layers in [start_moe_layer, start_moe_layer + num_moe_layers).
        // This avoids mmapping all 64 shards when each PP rank only needs ~20.
        let first_abs_layer = start_moe_layer + config.first_k_dense_replace;
        let last_abs_layer = first_abs_layer + num_moe_layers; // exclusive
        let mut needed_shards: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (tensor_name, shard_name) in &index.weight_map {
            // Check if this tensor belongs to a layer in our range
            if let Some(layer_num) = parse_layer_number(tensor_name) {
                if layer_num >= first_abs_layer && layer_num < last_abs_layer {
                    needed_shards.insert(shard_name.clone());
                }
            }
        }
        let mut shard_names: Vec<String> = needed_shards.into_iter().collect();
        shard_names.sort();

        let all_shard_count: std::collections::HashSet<&String> = index.weight_map.values().collect();
        log::info!(
            "[DIAG-RUST] Filtered shards: {}/{} needed for layers [{first_abs_layer}..{last_abs_layer})",
            shard_names.len(), all_shard_count.len(),
        );
        crate::syscheck::log_memory_usage("[DIAG-RUST] before mmap shards");

        // Open only needed shards
        let mut shards: HashMap<String, MmapSafetensors> = HashMap::new();
        for (i, name) in shard_names.iter().enumerate() {
            let path = model_dir.join(name);
            let st = MmapSafetensors::open(&path)
                .map_err(|e| format!("Failed to open {name}: {e}"))?;
            shards.insert(name.clone(), st);
            if (i + 1) % 10 == 0 || i + 1 == shard_names.len() {
                log::info!("[DIAG-RUST] Opened {}/{} shards", i + 1, shard_names.len());
            }
        }
        crate::syscheck::log_memory_usage("[DIAG-RUST] after mmap filtered shards");

        // Auto-detect expert weight prefix pattern
        let layers_prefix = detect_expert_prefix(&index.weight_map)?;
        log::info!("Detected expert prefix: {layers_prefix}");

        // Detect pre-quantized vs BF16 weights
        let prequantized = is_prequantized(&index.weight_map);
        let effective_group_size = if prequantized {
            // Use the first layer THIS rank owns (not global first_moe) since we
            // only opened shards for our layer range
            let probe_layer = start_moe_layer + config.first_k_dense_replace;
            let native_gs = detect_prequant_group_size(
                &index.weight_map, &shards, &layers_prefix, probe_layer,
            )?;
            if native_gs != group_size {
                log::info!(
                    "Pre-quantized model has group_size={native_gs}, overriding requested {group_size}"
                );
            }
            log::info!("Using pre-quantized INT4 weights (group_size={native_gs})");
            native_gs
        } else {
            log::info!("Using BF16 weights → quantizing to INT4 (group_size={group_size})");
            group_size
        };

        let mut experts: Vec<Vec<ExpertWeights>> = Vec::with_capacity(num_moe_layers);
        log::info!(
            "[DIAG-RUST] Starting expert loading: {} layers × {} experts (MoE layers [{start_moe_layer}..{}))",
            num_moe_layers, config.n_routed_experts, start_moe_layer + num_moe_layers,
        );

        for moe_idx in start_moe_layer..(start_moe_layer + num_moe_layers) {
            let layer_idx = moe_idx + config.first_k_dense_replace;
            let layer_start = std::time::Instant::now();
            let mut layer_experts = Vec::with_capacity(config.n_routed_experts);

            for eidx in 0..config.n_routed_experts {
                let prefix = format!("{layers_prefix}.layers.{layer_idx}.mlp.experts.{eidx}");

                let (gate, up, down) = if prequantized {
                    // Pre-quantized models are always INT4 (compressed-tensors format)
                    let g = QuantWeight::Int4(load_prequantized_weight(
                        &prefix, "gate_proj", &index.weight_map, &shards, effective_group_size,
                    )?);
                    let u = QuantWeight::Int4(load_prequantized_weight(
                        &prefix, "up_proj", &index.weight_map, &shards, effective_group_size,
                    )?);
                    let d = QuantWeight::Int4(load_prequantized_weight(
                        &prefix, "down_proj", &index.weight_map, &shards, effective_group_size,
                    )?);
                    (g, u, d)
                } else {
                    load_and_quantize_expert(
                        &prefix, &index.weight_map, &shards, effective_group_size, num_bits,
                    )?
                };

                layer_experts.push(ExpertWeights { gate, up, down });
            }

            let layer_elapsed = layer_start.elapsed();
            let action = if prequantized { "loaded" } else { "quantized" };
            let layers_done = experts.len() + 1;
            log::info!(
                "Layer {layer_idx}: {action} {} experts in {:.1}s [{layers_done}/{num_moe_layers}]",
                config.n_routed_experts,
                layer_elapsed.as_secs_f64(),
            );
            experts.push(layer_experts);
            // Log memory every 5 layers
            if layers_done % 5 == 0 || layers_done == num_moe_layers {
                crate::syscheck::log_memory_usage(&format!("[DIAG-RUST] after loading {layers_done}/{num_moe_layers} layers"));
            }
        }

        // Load shared experts (always BF16, quantized to INT4/INT8 like routed)
        let shared_experts = if config.n_shared_experts > 0 {
            let shared_intermediate = config.n_shared_experts * config.moe_intermediate_size;
            log::info!(
                "Loading shared experts: n_shared={}, intermediate_size={}",
                config.n_shared_experts, shared_intermediate,
            );
            let mut shared = Vec::with_capacity(num_moe_layers);
            for moe_idx in start_moe_layer..(start_moe_layer + num_moe_layers) {
                let layer_idx = moe_idx + config.first_k_dense_replace;
                let prefix = format!("{layers_prefix}.layers.{layer_idx}.mlp.shared_experts");
                let (gate, up, down) = load_and_quantize_expert(
                    &prefix, &index.weight_map, &shards, effective_group_size, num_bits,
                )?;
                shared.push(ExpertWeights { gate, up, down });
            }
            log::info!("Loaded {} shared expert layers", shared.len());
            shared
        } else {
            Vec::new()
        };

        let total_bytes: usize = experts.iter().flat_map(|layer| {
            layer.iter().map(|e| {
                e.gate.data_bytes() + e.up.data_bytes() + e.down.data_bytes()
            })
        }).sum();
        let shared_bytes: usize = shared_experts.iter().map(|e| {
            e.gate.data_bytes() + e.up.data_bytes() + e.down.data_bytes()
        }).sum();

        log::info!(
            "Loaded {} MoE layers × {} experts = {:.1} GB INT{num_bits} (group_size={effective_group_size}), shared={:.1} MB",
            num_moe_layers,
            config.n_routed_experts,
            total_bytes as f64 / 1e9,
            shared_bytes as f64 / 1e6,
        );

        Ok((experts, shared_experts, effective_group_size))
    }

    /// Write INT4 expert weights to a cache file (legacy v1 format).
    #[allow(dead_code)]
    fn save_cache(&self, path: &Path, config_hash: u64) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create cache dir: {e}"))?;
        }

        let num_moe_layers = self.experts.len();

        // Write to a temp file then rename (atomic)
        let tmp_path = path.with_extension("bin.tmp");
        let file = std::fs::File::create(&tmp_path)
            .map_err(|e| format!("Failed to create cache file: {e}"))?;
        let mut w = std::io::BufWriter::with_capacity(4 * 1024 * 1024, file);

        // Header (64 bytes)
        w.write_all(CACHE_MAGIC)
            .map_err(|e| format!("Write error: {e}"))?;
        w.write_all(&CACHE_VERSION.to_le_bytes())
            .map_err(|e| format!("Write error: {e}"))?;
        w.write_all(&(self.config.hidden_size as u64).to_le_bytes())
            .map_err(|e| format!("Write error: {e}"))?;
        w.write_all(&(self.config.moe_intermediate_size as u64).to_le_bytes())
            .map_err(|e| format!("Write error: {e}"))?;
        w.write_all(&(self.config.n_routed_experts as u64).to_le_bytes())
            .map_err(|e| format!("Write error: {e}"))?;
        w.write_all(&(num_moe_layers as u64).to_le_bytes())
            .map_err(|e| format!("Write error: {e}"))?;
        w.write_all(&(self.group_size as u64).to_le_bytes())
            .map_err(|e| format!("Write error: {e}"))?;
        w.write_all(&config_hash.to_le_bytes())
            .map_err(|e| format!("Write error: {e}"))?;
        w.write_all(&0u64.to_le_bytes()) // reserved
            .map_err(|e| format!("Write error: {e}"))?;

        // Expert data
        let write_start = std::time::Instant::now();
        for (layer_idx, layer) in self.experts.iter().enumerate() {
            for expert in layer {
                write_quantized(&mut w, &expert.gate)?;
                write_quantized(&mut w, &expert.up)?;
                write_quantized(&mut w, &expert.down)?;
            }
            if (layer_idx + 1) % 10 == 0 {
                log::info!("  Cache write: {}/{} layers", layer_idx + 1, num_moe_layers);
            }
        }

        w.flush().map_err(|e| format!("Flush error: {e}"))?;
        drop(w);

        // Atomic rename
        std::fs::rename(&tmp_path, path)
            .map_err(|e| format!("Failed to rename cache file: {e}"))?;

        let elapsed = write_start.elapsed();
        let size = std::fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or(0);
        log::info!(
            "Cache written: {:.1} GB in {:.1}s ({:.1} GB/s)",
            size as f64 / 1e9,
            elapsed.as_secs_f64(),
            size as f64 / 1e9 / elapsed.as_secs_f64(),
        );

        Ok(())
    }

    /// Load expert weights from cache file via mmap (legacy v1 format).
    #[allow(dead_code)]
    fn load_cache(
        path: &Path,
        config: &ModelConfig,
        group_size: usize,
        num_bits: u8,
        num_moe_layers: usize,
        config_hash: u64,
    ) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open cache: {e}"))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("Failed to mmap cache: {e}"))?;

        // Validate size
        let expected = expected_cache_size(config, group_size, num_bits, num_moe_layers);
        if mmap.len() != expected {
            return Err(format!(
                "Cache size mismatch: expected {} bytes, got {}",
                expected, mmap.len()
            ));
        }

        // Validate header
        if &mmap[0..4] != CACHE_MAGIC {
            return Err("Bad magic".to_string());
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != CACHE_VERSION {
            return Err(format!("Cache version {version}, expected {CACHE_VERSION}"));
        }

        let h_hidden = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let h_intermediate = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let h_n_experts = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;
        let h_num_layers = u64::from_le_bytes(mmap[32..40].try_into().unwrap()) as usize;
        let h_group_size = u64::from_le_bytes(mmap[40..48].try_into().unwrap()) as usize;
        let h_config_hash = u64::from_le_bytes(mmap[48..56].try_into().unwrap());

        if h_hidden != config.hidden_size
            || h_intermediate != config.moe_intermediate_size
            || h_n_experts != config.n_routed_experts
            || h_num_layers != num_moe_layers
            || h_group_size != group_size
        {
            return Err("Cache header dimensions don't match config".to_string());
        }

        if h_config_hash != config_hash {
            return Err("Config hash mismatch — model config.json changed".to_string());
        }

        // Read expert data from mmap
        log::info!("Loading from cache: {} (INT{})", path.display(), num_bits);
        let (gpb, gsb, dpb, dsb) = expert_byte_sizes(config, group_size, num_bits);
        let h = config.hidden_size;
        let m = config.moe_intermediate_size;
        let mut offset = CACHE_HEADER_SIZE;

        let mut experts: Vec<Vec<ExpertWeights>> = Vec::with_capacity(num_moe_layers);
        let load_start = std::time::Instant::now();

        for layer_idx in 0..num_moe_layers {
            let mut layer_experts = Vec::with_capacity(config.n_routed_experts);
            for _eidx in 0..config.n_routed_experts {
                let gate = read_quantized(&mmap, &mut offset, m, h, group_size, num_bits, gpb, gsb);
                let up = read_quantized(&mmap, &mut offset, m, h, group_size, num_bits, gpb, gsb);
                let down = read_quantized(&mmap, &mut offset, h, m, group_size, num_bits, dpb, dsb);
                layer_experts.push(ExpertWeights { gate, up, down });
            }
            experts.push(layer_experts);

            if (layer_idx + 1) % 10 == 0 {
                log::info!("  Cache read: {}/{} layers", layer_idx + 1, num_moe_layers);
            }
        }

        // Evict page cache — data is now copied into heap Vecs
        let cache_bytes = mmap.len();
        let _ = unsafe { mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) };
        drop(mmap);
        drop(file);

        let elapsed = load_start.elapsed();
        log::info!(
            "Cache loaded: {:.1} GB in {:.1}s ({:.1} GB/s)",
            cache_bytes as f64 / 1e9,
            elapsed.as_secs_f64(),
            cache_bytes as f64 / 1e9 / elapsed.as_secs_f64(),
        );

        Ok(WeightStore {
            experts,
            shared_experts: Vec::new(), // loaded separately after cache
            experts_cpu: Vec::new(),
            shared_experts_cpu: Vec::new(),
            experts_gpu: Vec::new(),
            shared_experts_gpu: Vec::new(),
            experts_gguf: Vec::new(),
            shared_experts_gguf: Vec::new(),
            config: config.clone(),
            group_size,
            cpu_num_bits: num_bits,
            gpu_num_bits: 4,
        })
    }

    /// Migrate expert weights to NUMA nodes according to the given map.
    /// Uses mbind(MPOL_MF_MOVE) to move physical pages without changing virtual addresses.
    /// Returns the number of successfully migrated experts.
    pub fn migrate_numa(&mut self, map: &crate::numa::NumaExpertMap) -> usize {
        use crate::numa::migrate_vec_to_node;

        fn migrate_quant_weight(w: &mut QuantWeight, node: usize) -> bool {
            match w {
                QuantWeight::Int4(q) => {
                    migrate_vec_to_node(&mut q.packed, node)
                        && migrate_vec_to_node(&mut q.scales, node)
                }
                QuantWeight::Int8(q) => {
                    migrate_vec_to_node(&mut q.data, node)
                        && migrate_vec_to_node(&mut q.scales, node)
                }
            }
        }

        let start = std::time::Instant::now();
        let mut migrated = 0;
        let mut failed = 0;

        for (layer_idx, layer) in self.experts.iter_mut().enumerate() {
            for (expert_idx, expert) in layer.iter_mut().enumerate() {
                let node = map.node_for(layer_idx, expert_idx);

                // Migrate all weight buffers for gate, up, down
                let ok = migrate_quant_weight(&mut expert.gate, node)
                    && migrate_quant_weight(&mut expert.up, node)
                    && migrate_quant_weight(&mut expert.down, node);

                if ok {
                    migrated += 1;
                } else {
                    failed += 1;
                }
            }
        }

        let elapsed = start.elapsed();
        log::info!(
            "NUMA migration: {migrated} experts migrated, {failed} failed, in {:.1}s",
            elapsed.as_secs_f64(),
        );

        migrated
    }

    /// Load shared expert weights from safetensors (BF16, quantized to INT4/INT8). Legacy.
    #[allow(dead_code)]
    fn load_shared_experts(
        model_dir: &Path,
        config: &ModelConfig,
        group_size: usize,
        num_bits: u8,
        num_moe_layers: usize,
    ) -> Result<Vec<ExpertWeights>, String> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_str = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("Failed to read safetensors index: {e}"))?;
        let index: SafetensorsIndex = serde_json::from_str(&index_str)
            .map_err(|e| format!("Failed to parse safetensors index: {e}"))?;
        let layers_prefix = detect_expert_prefix(&index.weight_map)?;

        // Collect shard names needed for shared experts
        let mut shard_names: std::collections::HashSet<String> = std::collections::HashSet::new();
        for moe_idx in 0..num_moe_layers {
            let layer_idx = moe_idx + config.first_k_dense_replace;
            let prefix = format!("{layers_prefix}.layers.{layer_idx}.mlp.shared_experts");
            for proj in &["gate_proj", "up_proj", "down_proj"] {
                let name = format!("{prefix}.{proj}.weight");
                if let Some(shard) = index.weight_map.get(&name) {
                    shard_names.insert(shard.clone());
                }
            }
        }

        let mut shards: HashMap<String, MmapSafetensors> = HashMap::new();
        for name in &shard_names {
            let path = model_dir.join(name);
            let st = MmapSafetensors::open(&path)
                .map_err(|e| format!("Failed to open {name}: {e}"))?;
            shards.insert(name.clone(), st);
        }

        let shared_intermediate = config.n_shared_experts * config.moe_intermediate_size;
        log::info!(
            "Loading shared experts: n_shared={}, intermediate_size={}, {} layers",
            config.n_shared_experts, shared_intermediate, num_moe_layers,
        );

        let start = std::time::Instant::now();
        let mut shared = Vec::with_capacity(num_moe_layers);
        for moe_idx in 0..num_moe_layers {
            let layer_idx = moe_idx + config.first_k_dense_replace;
            let prefix = format!("{layers_prefix}.layers.{layer_idx}.mlp.shared_experts");
            let (gate, up, down) = load_and_quantize_expert(
                &prefix, &index.weight_map, &shards, group_size, num_bits,
            )?;
            shared.push(ExpertWeights { gate, up, down });
        }
        log::info!(
            "Loaded {} shared expert layers in {:.1}s",
            shared.len(), start.elapsed().as_secs_f64(),
        );
        Ok(shared)
    }

    fn streaming_build_marlin_cache(
        model_dir: &Path,
        config: &ModelConfig,
        group_size: usize,
        num_moe_layers: usize,
        start_moe_layer: usize,
        cache_path: &Path,
        config_hash: u64,
    ) -> Result<usize, String> {
        log::info!(
            "Streaming build MARLIN cache: {} MoE layers from safetensors → {}",
            num_moe_layers, cache_path.display(),
        );
        crate::syscheck::log_memory_usage("before streaming_build_marlin_cache");

        // Parse safetensors index
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_str = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("Failed to read safetensors index: {e}"))?;
        let index: SafetensorsIndex = serde_json::from_str(&index_str)
            .map_err(|e| format!("Failed to parse safetensors index: {e}"))?;

        // Determine which shard files we need
        let first_abs_layer = start_moe_layer + config.first_k_dense_replace;
        let last_abs_layer = first_abs_layer + num_moe_layers;
        let mut needed_shards: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (tensor_name, shard_name) in &index.weight_map {
            if let Some(layer_num) = parse_layer_number(tensor_name) {
                if layer_num >= first_abs_layer && layer_num < last_abs_layer {
                    needed_shards.insert(shard_name.clone());
                }
            }
        }
        let mut shard_names: Vec<String> = needed_shards.into_iter().collect();
        shard_names.sort();

        log::info!(
            "Opening {}/{} safetensors shards (mmap, near-zero RAM)",
            shard_names.len(),
            index.weight_map.values().collect::<std::collections::HashSet<_>>().len(),
        );

        // Open shards via mmap
        let mut shards: HashMap<String, MmapSafetensors> = HashMap::new();
        for (i, name) in shard_names.iter().enumerate() {
            let path = model_dir.join(name);
            let st = MmapSafetensors::open(&path)
                .map_err(|e| format!("Failed to open {name}: {e}"))?;
            shards.insert(name.clone(), st);
            if (i + 1) % 10 == 0 || i + 1 == shard_names.len() {
                log::info!("  Opened {}/{} shards", i + 1, shard_names.len());
            }
        }

        // Detect prefix and quantization format
        let layers_prefix = detect_expert_prefix(&index.weight_map)?;
        let prequantized = is_prequantized(&index.weight_map);
        let effective_group_size = if prequantized {
            let probe_layer = start_moe_layer + config.first_k_dense_replace;
            let native_gs = detect_prequant_group_size(
                &index.weight_map, &shards, &layers_prefix, probe_layer,
            )?;
            if native_gs != group_size {
                log::info!(
                    "Pre-quantized model has group_size={native_gs}, overriding requested {group_size}"
                );
            }
            native_gs
        } else {
            group_size
        };

        // Create cache directory + temp file
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create cache dir: {e}"))?;
        }
        let tmp_path = cache_path.with_extension("bin.tmp");
        let file = std::fs::File::create(&tmp_path)
            .map_err(|e| format!("Failed to create cache file: {e}"))?;
        let mut w = std::io::BufWriter::with_capacity(4 * 1024 * 1024, file);

        // Write header (version 3 = Marlin format)
        write_marlin_cache_header(&mut w, config, effective_group_size, num_moe_layers, config_hash)?;

        // Configure rayon to use physical cores only (hyperthreads hurt on EPYC)
        let physical_cores = detect_physical_cores();
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(physical_cores)
            .build_global();
        log::info!("Rayon thread pool: {physical_cores} physical cores (cache build)");

        let overall_start = std::time::Instant::now();

        // Prefetch first layer's expert data asynchronously
        {
            let first_layer_idx = start_moe_layer + config.first_k_dense_replace;
            let proj_names = if prequantized {
                vec!["gate_proj.weight_packed", "gate_proj.weight_scale",
                     "up_proj.weight_packed", "up_proj.weight_scale",
                     "down_proj.weight_packed", "down_proj.weight_scale"]
            } else {
                vec!["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
            };
            for eidx in 0..config.n_routed_experts {
                for proj in &proj_names {
                    let tensor_name = format!(
                        "{layers_prefix}.layers.{first_layer_idx}.mlp.experts.{eidx}.{proj}"
                    );
                    if let Some(shard_name) = index.weight_map.get(&tensor_name) {
                        if let Some(shard) = shards.get(shard_name) {
                            shard.prefetch_tensor(&tensor_name);
                        }
                    }
                }
            }
            log::info!("Issued MADV_WILLNEED prefetch for layer {first_layer_idx} experts");
        }

        // Stream routed experts layer by layer
        for moe_idx in start_moe_layer..(start_moe_layer + num_moe_layers) {
            let layer_idx = moe_idx + config.first_k_dense_replace;
            let layer_start = std::time::Instant::now();

            // 2-phase pipeline: sequential I/O then parallel compute.
            // Phase 1 does sequential mmap reads (optimal for kernel readahead).
            // Phase 2 does parallel Marlin repack (CPU-bound, scales with cores).
            let io_start = std::time::Instant::now();
            let mut expert_data: Vec<ExpertWeights> = Vec::with_capacity(config.n_routed_experts);
            for eidx in 0..config.n_routed_experts {
                let prefix = format!("{layers_prefix}.layers.{layer_idx}.mlp.experts.{eidx}");
                let (gate, up, down) = if prequantized {
                    let g = QuantWeight::Int4(load_prequantized_weight(
                        &prefix, "gate_proj", &index.weight_map, &shards, effective_group_size,
                    )?);
                    let u = QuantWeight::Int4(load_prequantized_weight(
                        &prefix, "up_proj", &index.weight_map, &shards, effective_group_size,
                    )?);
                    let d = QuantWeight::Int4(load_prequantized_weight(
                        &prefix, "down_proj", &index.weight_map, &shards, effective_group_size,
                    )?);
                    (g, u, d)
                } else {
                    load_and_quantize_expert(
                        &prefix, &index.weight_map, &shards, effective_group_size, 4,
                    )?
                };
                expert_data.push(ExpertWeights { gate, up, down });
            }
            let io_elapsed = io_start.elapsed();

            // Prefetch next layer's data while we do CPU-bound Marlin repack
            let next_moe_idx = moe_idx + 1;
            if next_moe_idx < start_moe_layer + num_moe_layers {
                let next_layer_idx = next_moe_idx + config.first_k_dense_replace;
                let proj_names: &[&str] = if prequantized {
                    &["gate_proj.weight_packed", "gate_proj.weight_scale",
                      "up_proj.weight_packed", "up_proj.weight_scale",
                      "down_proj.weight_packed", "down_proj.weight_scale"]
                } else {
                    &["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
                };
                for eidx in 0..config.n_routed_experts {
                    for proj in proj_names {
                        let tensor_name = format!(
                            "{layers_prefix}.layers.{next_layer_idx}.mlp.experts.{eidx}.{proj}"
                        );
                        if let Some(shard_name) = index.weight_map.get(&tensor_name) {
                            if let Some(shard) = shards.get(shard_name) {
                                shard.prefetch_tensor(&tensor_name);
                            }
                        }
                    }
                }
            }

            // Phase 2: Parallel Marlin repack across all CPU cores
            let repack_start = std::time::Instant::now();
            let expert_results: Vec<UnifiedExpertWeights> = expert_data
                .into_par_iter()
                .map(|ew| UnifiedExpertWeights::from_expert_weights_marlin(&ew))
                .collect();
            let repack_elapsed = repack_start.elapsed();

            // Phase 3: Sequential write (file format requires expert ordering)
            for marlin in &expert_results {
                write_vec_u32(&mut w, &marlin.w13_packed)?;
                write_vec_u16(&mut w, &marlin.w13_scales)?;
                write_vec_u32(&mut w, &marlin.w2_packed)?;
                write_vec_u16(&mut w, &marlin.w2_scales)?;
            }
            drop(expert_results); // Free Marlin results immediately

            let layers_done = moe_idx - start_moe_layer + 1;
            let layer_elapsed = layer_start.elapsed();
            if layers_done % 5 == 0 || layers_done == num_moe_layers {
                crate::syscheck::log_memory_usage(&format!(
                    "Marlin cache: {layers_done}/{num_moe_layers} layers ({:.1}s/layer, io={:.1}s repack={:.1}s)",
                    layer_elapsed.as_secs_f64(),
                    io_elapsed.as_secs_f64(),
                    repack_elapsed.as_secs_f64(),
                ));
            } else {
                log::info!(
                    "  Layer {layer_idx}: {} experts in {:.1}s (io={:.1}s repack={:.1}s) [{layers_done}/{num_moe_layers}]",
                    config.n_routed_experts,
                    layer_elapsed.as_secs_f64(),
                    io_elapsed.as_secs_f64(),
                    repack_elapsed.as_secs_f64(),
                );
            }
        }

        // Stream shared experts
        if config.n_shared_experts > 0 {
            log::info!("Streaming shared experts ({} layers)...", num_moe_layers);
            for moe_idx in start_moe_layer..(start_moe_layer + num_moe_layers) {
                let layer_idx = moe_idx + config.first_k_dense_replace;
                let prefix = format!("{layers_prefix}.layers.{layer_idx}.mlp.shared_experts");
                let (gate, up, down) = load_and_quantize_expert(
                    &prefix, &index.weight_map, &shards, effective_group_size, 4,
                )?;
                let ew = ExpertWeights { gate, up, down };
                let marlin = UnifiedExpertWeights::from_expert_weights_marlin(&ew);

                write_vec_u32(&mut w, &marlin.w13_packed)?;
                write_vec_u16(&mut w, &marlin.w13_scales)?;
                write_vec_u32(&mut w, &marlin.w2_packed)?;
                write_vec_u16(&mut w, &marlin.w2_scales)?;
            }
        }

        // Flush + atomic rename
        w.flush().map_err(|e| format!("Flush error: {e}"))?;
        drop(w);
        std::fs::rename(&tmp_path, cache_path)
            .map_err(|e| format!("Failed to rename cache file: {e}"))?;

        // Evict safetensors page cache, then free mmaps and reclaim RAM
        for shard in shards.values() {
            shard.evict_page_cache();
        }
        drop(shards);
        #[cfg(target_os = "linux")]
        unsafe { libc::malloc_trim(0); }

        let elapsed = overall_start.elapsed();
        let size = std::fs::metadata(cache_path).map(|m| m.len()).unwrap_or(0);
        log::info!(
            "Marlin cache built: {:.1} GB in {:.1}s ({:.1} GB/s)",
            size as f64 / 1e9,
            elapsed.as_secs_f64(),
            size as f64 / 1e9 / elapsed.as_secs_f64(),
        );
        crate::syscheck::log_memory_usage("after streaming_build_marlin_cache");

        Ok(effective_group_size)
    }

    /// Build Marlin cache with a file lock for multi-process safety.
    fn build_marlin_cache_locked(
        model_dir: &Path,
        config: &ModelConfig,
        group_size: usize,
        total_moe_layers: usize,
        cache_path: &Path,
        config_hash: u64,
    ) -> Result<usize, String> {
        use std::fs::OpenOptions;

        if cache_path.exists() {
            log::info!("Marlin cache appeared while preparing to build (another rank finished)");
            return Ok(group_size);
        }

        let lock_path = cache_path.with_extension("bin.lock");

        // Ensure cache directory exists
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create cache directory {}: {e}", parent.display()))?;
        }

        match OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&lock_path)
        {
            Ok(mut lock_file) => {
                log::info!(
                    "Acquired Marlin cache build lock, building {} MoE layers...",
                    total_moe_layers,
                );
                let _ = write!(lock_file, "{}", std::process::id());
                drop(lock_file);

                let result = Self::streaming_build_marlin_cache(
                    model_dir, config, group_size, total_moe_layers,
                    0, cache_path, config_hash,
                );

                let _ = std::fs::remove_file(&lock_path);

                match result {
                    Ok(effective_gs) => {
                        let expected_path = cache_path_marlin(model_dir, effective_gs);
                        if expected_path != *cache_path {
                            std::fs::rename(cache_path, &expected_path)
                                .map_err(|e| format!("Failed to rename cache: {e}"))?;
                            log::info!(
                                "Renamed Marlin cache to {} (effective gs={})",
                                expected_path.display(), effective_gs,
                            );
                        }
                        Ok(effective_gs)
                    }
                    Err(e) => Err(e),
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                log::info!("Another process is building Marlin cache, waiting...");
                let wait_start = std::time::Instant::now();
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(5));

                    for try_gs in &[group_size, 32, 64, 128] {
                        let try_path = cache_path_marlin(model_dir, *try_gs);
                        if try_path.exists() {
                            let waited = wait_start.elapsed();
                            log::info!(
                                "Marlin cache ready after {:.0}s wait (gs={})",
                                waited.as_secs_f64(), try_gs,
                            );
                            return Ok(*try_gs);
                        }
                    }

                    if !lock_path.exists() && !cache_path.exists() {
                        return Err("Marlin cache build by another process failed".to_string());
                    }

                    let waited = wait_start.elapsed();
                    if waited > std::time::Duration::from_secs(7200) {
                        return Err("Timed out waiting for Marlin cache build (2 hours)".to_string());
                    }
                    if waited.as_secs() % 60 < 5 {
                        log::info!("Still waiting for Marlin cache build ({:.0}s)...", waited.as_secs_f64());
                    }
                }
            }
            Err(e) => {
                Err(format!("Failed to create cache lock file: {e}"))
            }
        }
    }

    /// Load v3 Marlin cache from disk.
    ///
    /// Same file structure as v2 unified (identical byte counts per expert),
    /// but data is GPU-native Marlin format (tile-permuted, scale-permuted).
    fn load_marlin_cache(
        path: &Path,
        config: &ModelConfig,
        group_size: usize,
        total_moe_layers: usize,
        config_hash: u64,
        start_moe_layer: usize,
        num_layers_to_load: usize,
    ) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open Marlin cache: {e}"))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("Failed to mmap Marlin cache: {e}"))?;

        // Validate header
        if mmap.len() < CACHE_HEADER_SIZE {
            return Err("Marlin cache too small for header".to_string());
        }
        if &mmap[0..4] != CACHE_MAGIC {
            return Err("Bad magic in Marlin cache".to_string());
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != CACHE_VERSION_MARLIN {
            return Err(format!("Cache version {version}, expected {CACHE_VERSION_MARLIN} (Marlin)"));
        }

        let h_hidden = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let h_intermediate = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let h_n_experts = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;
        let h_num_layers = u64::from_le_bytes(mmap[32..40].try_into().unwrap()) as usize;
        let h_group_size = u64::from_le_bytes(mmap[40..48].try_into().unwrap()) as usize;
        let h_config_hash = u64::from_le_bytes(mmap[48..56].try_into().unwrap());
        let h_n_shared = u64::from_le_bytes(mmap[56..64].try_into().unwrap()) as usize;

        if h_hidden != config.hidden_size
            || h_intermediate != config.moe_intermediate_size
            || h_n_experts != config.n_routed_experts
            || h_num_layers != total_moe_layers
            || h_group_size != group_size
        {
            return Err(format!(
                "Marlin cache header mismatch: file has {}h/{}m/{}e/{}L/g{}, expected {}h/{}m/{}e/{}L/g{}",
                h_hidden, h_intermediate, h_n_experts, h_num_layers, h_group_size,
                config.hidden_size, config.moe_intermediate_size, config.n_routed_experts,
                total_moe_layers, group_size,
            ));
        }
        if h_config_hash != config_hash {
            return Err("Config hash mismatch in Marlin cache".to_string());
        }
        if h_n_shared != config.n_shared_experts {
            return Err(format!(
                "Shared expert count mismatch: cache={h_n_shared}, config={}",
                config.n_shared_experts,
            ));
        }

        if start_moe_layer + num_layers_to_load > total_moe_layers {
            return Err(format!(
                "Range [{}, {}) exceeds total MoE layers {}",
                start_moe_layer, start_moe_layer + num_layers_to_load, total_moe_layers,
            ));
        }

        // Validate file size (byte counts identical to v2 unified)
        let shared_intermediate = config.n_shared_experts * config.moe_intermediate_size;
        let expected = expected_unified_cache_size(
            config, group_size, total_moe_layers, config.n_shared_experts, shared_intermediate,
        );
        if mmap.len() != expected {
            return Err(format!(
                "Marlin cache size mismatch: expected {} bytes, got {}",
                expected, mmap.len(),
            ));
        }

        let is_partial = start_moe_layer > 0 || num_layers_to_load < total_moe_layers;
        if is_partial {
            log::info!(
                "Loading MARLIN cache (partial): layers [{}-{}), {} of {} ({})",
                start_moe_layer, start_moe_layer + num_layers_to_load,
                num_layers_to_load, total_moe_layers, path.display(),
            );
        } else {
            log::info!("Loading MARLIN cache: {} (all {} layers)", path.display(), total_moe_layers);
        }
        let load_start = std::time::Instant::now();

        let h = config.hidden_size;
        let m = config.moe_intermediate_size;

        // Per-expert byte sizes (identical to v2 unified)
        let (w13pb, w13sb, w2pb, w2sb) = unified_expert_byte_sizes(config, group_size);
        let per_routed_expert = w13pb + w13sb + w2pb + w2sb;
        let per_routed_layer = config.n_routed_experts * per_routed_expert;

        let mut offset = CACHE_HEADER_SIZE + start_moe_layer * per_routed_layer;

        // Load routed experts
        let mut experts_gpu = Vec::with_capacity(num_layers_to_load);
        for layer_idx in 0..num_layers_to_load {
            let mut layer_experts = Vec::with_capacity(config.n_routed_experts);
            for _eidx in 0..config.n_routed_experts {
                layer_experts.push(read_unified_expert(&mmap, &mut offset, h, m, group_size));
            }
            experts_gpu.push(layer_experts);

            if (layer_idx + 1) % 10 == 0 || layer_idx + 1 == num_layers_to_load {
                log::info!(
                    "  Marlin cache loaded: {}/{} layers ({:.1} GB)",
                    layer_idx + 1, num_layers_to_load,
                    offset as f64 / 1e9,
                );
            }
        }

        // Load shared experts
        let mut shared_experts_gpu = Vec::new();
        if config.n_shared_experts > 0 {
            let routed_total = total_moe_layers * per_routed_layer;
            let shared_m = config.n_shared_experts * config.moe_intermediate_size;
            let (s_w13pb, s_w13sb, s_w2pb, s_w2sb) = (
                (h / 8) * (2 * shared_m) * 4,
                (h / group_size) * (2 * shared_m) * 2,
                (shared_m / 8) * h * 4,
                (shared_m / group_size) * h * 2,
            );
            let per_shared = s_w13pb + s_w13sb + s_w2pb + s_w2sb;

            let shared_base = CACHE_HEADER_SIZE + routed_total + start_moe_layer * per_shared;
            offset = shared_base;

            for _i in 0..num_layers_to_load {
                shared_experts_gpu.push(
                    read_unified_expert(&mmap, &mut offset, h, shared_m, group_size),
                );
            }
            log::info!("  Loaded {} shared experts (Marlin)", num_layers_to_load);
        }

        // Evict page cache — data is now copied into heap Vecs
        let _ = unsafe { mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) };
        drop(mmap);
        drop(file);

        let elapsed = load_start.elapsed();
        log::info!(
            "MARLIN cache loaded in {:.1}s: {} layers × {} experts (+ {} shared), {:.1} GB",
            elapsed.as_secs_f64(),
            num_layers_to_load, config.n_routed_experts,
            shared_experts_gpu.len(),
            offset as f64 / 1e9,
        );

        Ok(WeightStore {
            experts: Vec::new(),
            shared_experts: Vec::new(),
            experts_cpu: Vec::new(),
            shared_experts_cpu: Vec::new(),
            experts_gpu,
            shared_experts_gpu,
            experts_gguf: Vec::new(),
            shared_experts_gguf: Vec::new(),
            config: config.clone(),
            group_size,
            cpu_num_bits: 4,
            gpu_num_bits: 4,
        })
    }


    /// Streaming build CPU transposed cache from safetensors.
    ///
    /// Reads expert weights layer by layer, transposes to CPU-optimized format,
    /// writes to disk cache. Supports both INT4 and INT8 via `cpu_num_bits`.
    fn streaming_build_cpu_cache(
        model_dir: &Path,
        config: &ModelConfig,
        group_size: usize,
        num_moe_layers: usize,
        start_moe_layer: usize,
        cache_path: &Path,
        config_hash: u64,
        cpu_num_bits: u8,
    ) -> Result<usize, String> {
        log::info!(
            "Streaming build CPU INT{} cache: {} MoE layers from safetensors → {}",
            cpu_num_bits, num_moe_layers, cache_path.display(),
        );
        crate::syscheck::log_memory_usage("before streaming_build_cpu_cache");

        // Parse safetensors index
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_str = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("Failed to read safetensors index: {e}"))?;
        let index: SafetensorsIndex = serde_json::from_str(&index_str)
            .map_err(|e| format!("Failed to parse safetensors index: {e}"))?;

        // Determine needed shards
        let first_abs_layer = start_moe_layer + config.first_k_dense_replace;
        let last_abs_layer = first_abs_layer + num_moe_layers;
        let mut needed_shards: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (tensor_name, shard_name) in &index.weight_map {
            if let Some(layer_num) = parse_layer_number(tensor_name) {
                if layer_num >= first_abs_layer && layer_num < last_abs_layer {
                    needed_shards.insert(shard_name.clone());
                }
            }
        }
        let mut shard_names: Vec<String> = needed_shards.into_iter().collect();
        shard_names.sort();

        log::info!(
            "Opening {}/{} safetensors shards for CPU cache build",
            shard_names.len(),
            index.weight_map.values().collect::<std::collections::HashSet<_>>().len(),
        );

        let mut shards: HashMap<String, MmapSafetensors> = HashMap::new();
        for (i, name) in shard_names.iter().enumerate() {
            let path = model_dir.join(name);
            let st = MmapSafetensors::open(&path)
                .map_err(|e| format!("Failed to open {name}: {e}"))?;
            shards.insert(name.clone(), st);
            if (i + 1) % 10 == 0 || i + 1 == shard_names.len() {
                log::info!("  Opened {}/{} shards", i + 1, shard_names.len());
            }
        }

        // Detect prefix and quantization format
        let layers_prefix = detect_expert_prefix(&index.weight_map)?;
        let prequantized = is_prequantized(&index.weight_map);
        let effective_group_size = if prequantized {
            let probe_layer = start_moe_layer + config.first_k_dense_replace;
            let native_gs = detect_prequant_group_size(
                &index.weight_map, &shards, &layers_prefix, probe_layer,
            )?;
            if native_gs != group_size {
                log::info!(
                    "Pre-quantized model has group_size={native_gs}, overriding requested {group_size}"
                );
            }
            native_gs
        } else {
            group_size
        };

        // Create cache directory + temp file
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create cache dir: {e}"))?;
        }
        let tmp_path = cache_path.with_extension("bin.tmp");
        let file = std::fs::File::create(&tmp_path)
            .map_err(|e| format!("Failed to create CPU cache file: {e}"))?;
        let mut w = std::io::BufWriter::with_capacity(4 * 1024 * 1024, file);

        // Write header (version 4 = CPU transposed format)
        write_cpu_cache_header(&mut w, config, effective_group_size, num_moe_layers, config_hash, cpu_num_bits)?;

        let overall_start = std::time::Instant::now();

        // Stream routed experts layer by layer
        for moe_idx in start_moe_layer..(start_moe_layer + num_moe_layers) {
            let layer_idx = moe_idx + config.first_k_dense_replace;
            let layer_start = std::time::Instant::now();

            // Phase 1: Sequential I/O — load expert weights from safetensors
            let io_start = std::time::Instant::now();
            let mut expert_data: Vec<ExpertWeights> = Vec::with_capacity(config.n_routed_experts);
            for eidx in 0..config.n_routed_experts {
                let prefix = format!("{layers_prefix}.layers.{layer_idx}.mlp.experts.{eidx}");
                let (gate, up, down) = if prequantized {
                    // Pre-quantized models are always INT4 — if cpu_num_bits=8, we'd need
                    // to dequantize and re-quantize. For now, only support INT4 for pre-quantized.
                    if cpu_num_bits != 4 {
                        return Err(format!(
                            "CPU INT{cpu_num_bits} cache not supported for pre-quantized INT4 models (would need dequant+requant)"
                        ));
                    }
                    let g = QuantWeight::Int4(load_prequantized_weight(
                        &prefix, "gate_proj", &index.weight_map, &shards, effective_group_size,
                    )?);
                    let u = QuantWeight::Int4(load_prequantized_weight(
                        &prefix, "up_proj", &index.weight_map, &shards, effective_group_size,
                    )?);
                    let d = QuantWeight::Int4(load_prequantized_weight(
                        &prefix, "down_proj", &index.weight_map, &shards, effective_group_size,
                    )?);
                    (g, u, d)
                } else {
                    load_and_quantize_expert(
                        &prefix, &index.weight_map, &shards, effective_group_size, cpu_num_bits,
                    )?
                };
                expert_data.push(ExpertWeights { gate, up, down });
            }
            let io_elapsed = io_start.elapsed();

            // Phase 2: Parallel CPU transpose across all cores
            let repack_start = std::time::Instant::now();
            let expert_results: Vec<UnifiedExpertWeights> = expert_data
                .into_par_iter()
                .map(|ew| {
                    if cpu_num_bits == 8 {
                        UnifiedExpertWeights::from_expert_weights_int8(&ew)
                    } else {
                        UnifiedExpertWeights::from_expert_weights(&ew)
                    }
                })
                .collect();
            let repack_elapsed = repack_start.elapsed();

            // Phase 3: Sequential write
            for cpu_exp in &expert_results {
                write_vec_u32(&mut w, &cpu_exp.w13_packed)?;
                write_vec_u16(&mut w, &cpu_exp.w13_scales)?;
                write_vec_u32(&mut w, &cpu_exp.w2_packed)?;
                write_vec_u16(&mut w, &cpu_exp.w2_scales)?;
            }
            drop(expert_results);

            let layers_done = moe_idx - start_moe_layer + 1;
            let layer_elapsed = layer_start.elapsed();
            if layers_done % 5 == 0 || layers_done == num_moe_layers {
                crate::syscheck::log_memory_usage(&format!(
                    "CPU cache: {layers_done}/{num_moe_layers} layers ({:.1}s/layer, io={:.1}s transpose={:.1}s)",
                    layer_elapsed.as_secs_f64(),
                    io_elapsed.as_secs_f64(),
                    repack_elapsed.as_secs_f64(),
                ));
            } else {
                log::info!(
                    "  Layer {layer_idx}: {} experts in {:.1}s (io={:.1}s transpose={:.1}s) [{layers_done}/{num_moe_layers}]",
                    config.n_routed_experts,
                    layer_elapsed.as_secs_f64(),
                    io_elapsed.as_secs_f64(),
                    repack_elapsed.as_secs_f64(),
                );
            }
        }

        // Stream shared experts
        if config.n_shared_experts > 0 {
            log::info!("Streaming shared experts for CPU cache ({} layers)...", num_moe_layers);
            for moe_idx in start_moe_layer..(start_moe_layer + num_moe_layers) {
                let layer_idx = moe_idx + config.first_k_dense_replace;
                let prefix = format!("{layers_prefix}.layers.{layer_idx}.mlp.shared_experts");
                let (gate, up, down) = load_and_quantize_expert(
                    &prefix, &index.weight_map, &shards, effective_group_size, cpu_num_bits,
                )?;
                let ew = ExpertWeights { gate, up, down };
                let cpu_exp = if cpu_num_bits == 8 {
                    UnifiedExpertWeights::from_expert_weights_int8(&ew)
                } else {
                    UnifiedExpertWeights::from_expert_weights(&ew)
                };

                write_vec_u32(&mut w, &cpu_exp.w13_packed)?;
                write_vec_u16(&mut w, &cpu_exp.w13_scales)?;
                write_vec_u32(&mut w, &cpu_exp.w2_packed)?;
                write_vec_u16(&mut w, &cpu_exp.w2_scales)?;
            }
        }

        // Flush + atomic rename
        w.flush().map_err(|e| format!("Flush error: {e}"))?;
        drop(w);
        std::fs::rename(&tmp_path, cache_path)
            .map_err(|e| format!("Failed to rename CPU cache file: {e}"))?;

        // Evict safetensors page cache, then free mmaps and reclaim RAM
        for shard in shards.values() {
            shard.evict_page_cache();
        }
        drop(shards);
        #[cfg(target_os = "linux")]
        unsafe { libc::malloc_trim(0); }

        let elapsed = overall_start.elapsed();
        let size = std::fs::metadata(cache_path).map(|m| m.len()).unwrap_or(0);
        log::info!(
            "CPU INT{} cache built: {:.1} GB in {:.1}s ({:.1} GB/s)",
            cpu_num_bits,
            size as f64 / 1e9,
            elapsed.as_secs_f64(),
            size as f64 / 1e9 / elapsed.as_secs_f64(),
        );
        crate::syscheck::log_memory_usage("after streaming_build_cpu_cache");

        Ok(effective_group_size)
    }

    /// Load v4 CPU transposed cache from disk.
    fn load_cpu_cache(
        path: &Path,
        config: &ModelConfig,
        group_size: usize,
        total_moe_layers: usize,
        config_hash: u64,
        start_moe_layer: usize,
        num_layers_to_load: usize,
        expected_bits: u8,
    ) -> Result<(Vec<Vec<UnifiedExpertWeights>>, Vec<UnifiedExpertWeights>), String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open CPU cache: {e}"))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("Failed to mmap CPU cache: {e}"))?;

        // Validate header
        if mmap.len() < CACHE_HEADER_SIZE {
            return Err("CPU cache too small for header".to_string());
        }
        if &mmap[0..4] != CACHE_MAGIC {
            return Err("Bad magic in CPU cache".to_string());
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != CACHE_VERSION_CPU {
            return Err(format!("Cache version {version}, expected {CACHE_VERSION_CPU} (CPU)"));
        }

        let h_hidden = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let h_intermediate = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let h_n_experts = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;
        let h_num_layers = u64::from_le_bytes(mmap[32..40].try_into().unwrap()) as usize;
        let h_group_size = u64::from_le_bytes(mmap[40..48].try_into().unwrap()) as usize;
        let h_config_hash = u64::from_le_bytes(mmap[48..56].try_into().unwrap());
        let packed_meta = u64::from_le_bytes(mmap[56..64].try_into().unwrap());
        let h_n_shared = (packed_meta & 0xFFFFFFFF) as usize;
        let h_num_bits = ((packed_meta >> 32) & 0xFF) as u8;

        if h_hidden != config.hidden_size
            || h_intermediate != config.moe_intermediate_size
            || h_n_experts != config.n_routed_experts
            || h_num_layers != total_moe_layers
            || h_group_size != group_size
        {
            return Err(format!(
                "CPU cache header mismatch: file has {}h/{}m/{}e/{}L/g{}, expected {}h/{}m/{}e/{}L/g{}",
                h_hidden, h_intermediate, h_n_experts, h_num_layers, h_group_size,
                config.hidden_size, config.moe_intermediate_size, config.n_routed_experts,
                total_moe_layers, group_size,
            ));
        }
        if h_config_hash != config_hash {
            return Err("Config hash mismatch in CPU cache".to_string());
        }
        if h_n_shared != config.n_shared_experts {
            return Err(format!(
                "Shared expert count mismatch: cache={h_n_shared}, config={}",
                config.n_shared_experts,
            ));
        }
        if h_num_bits != expected_bits {
            return Err(format!(
                "CPU cache num_bits mismatch: cache=INT{h_num_bits}, expected INT{expected_bits}",
            ));
        }

        if start_moe_layer + num_layers_to_load > total_moe_layers {
            return Err(format!(
                "Range [{}, {}) exceeds total MoE layers {}",
                start_moe_layer, start_moe_layer + num_layers_to_load, total_moe_layers,
            ));
        }

        // Validate file size
        let shared_intermediate = config.n_shared_experts * config.moe_intermediate_size;
        let expected = expected_cpu_cache_size(
            config, group_size, expected_bits, total_moe_layers,
            config.n_shared_experts, shared_intermediate,
        );
        if mmap.len() != expected {
            return Err(format!(
                "CPU cache size mismatch: expected {} bytes, got {}",
                expected, mmap.len(),
            ));
        }

        let is_partial = start_moe_layer > 0 || num_layers_to_load < total_moe_layers;
        if is_partial {
            log::info!(
                "Loading CPU INT{} cache (partial): layers [{}-{}), {} of {} ({})",
                expected_bits, start_moe_layer, start_moe_layer + num_layers_to_load,
                num_layers_to_load, total_moe_layers, path.display(),
            );
        } else {
            log::info!("Loading CPU INT{} cache: {} (all {} layers)", expected_bits, path.display(), total_moe_layers);
        }
        let load_start = std::time::Instant::now();

        let h = config.hidden_size;
        let m = config.moe_intermediate_size;

        // Per-expert byte sizes for this cpu_num_bits
        let (w13pb, w13sb, w2pb, w2sb) = cpu_expert_byte_sizes(config, group_size, expected_bits);
        let per_routed_expert = w13pb + w13sb + w2pb + w2sb;
        let per_routed_layer = config.n_routed_experts * per_routed_expert;

        let mut offset = CACHE_HEADER_SIZE + start_moe_layer * per_routed_layer;

        // Load routed experts
        let mut experts_cpu = Vec::with_capacity(num_layers_to_load);
        for layer_idx in 0..num_layers_to_load {
            let mut layer_experts = Vec::with_capacity(config.n_routed_experts);
            for _eidx in 0..config.n_routed_experts {
                layer_experts.push(read_unified_expert_cpu(
                    &mmap, &mut offset, h, m, group_size, expected_bits,
                ));
            }
            experts_cpu.push(layer_experts);

            if (layer_idx + 1) % 10 == 0 || layer_idx + 1 == num_layers_to_load {
                log::info!(
                    "  CPU cache loaded: {}/{} layers ({:.1} GB)",
                    layer_idx + 1, num_layers_to_load,
                    offset as f64 / 1e9,
                );
            }
        }

        // Load shared experts
        let mut shared_experts_cpu = Vec::new();
        if config.n_shared_experts > 0 {
            let routed_total = total_moe_layers * per_routed_layer;
            let shared_m = config.n_shared_experts * config.moe_intermediate_size;
            let (s_w13pb, s_w13sb, s_w2pb, s_w2sb) = if expected_bits == 4 {
                (
                    (h / 8) * (2 * shared_m) * 4,
                    (h / group_size) * (2 * shared_m) * 2,
                    (shared_m / 8) * h * 4,
                    (shared_m / group_size) * h * 2,
                )
            } else {
                let s_w13_bytes = h * (2 * shared_m);
                let s_w2_bytes = shared_m * h;
                (
                    ((s_w13_bytes + 3) / 4) * 4,
                    (h / group_size) * (2 * shared_m) * 2,
                    ((s_w2_bytes + 3) / 4) * 4,
                    (shared_m / group_size) * h * 2,
                )
            };
            let per_shared = s_w13pb + s_w13sb + s_w2pb + s_w2sb;

            let shared_base = CACHE_HEADER_SIZE + routed_total + start_moe_layer * per_shared;
            offset = shared_base;

            for _i in 0..num_layers_to_load {
                shared_experts_cpu.push(
                    read_unified_expert_cpu(&mmap, &mut offset, h, shared_m, group_size, expected_bits),
                );
            }
            log::info!("  Loaded {} shared experts (CPU INT{})", num_layers_to_load, expected_bits);
        }

        // Evict page cache — data is now copied into heap Vecs
        let _ = unsafe { mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) };
        drop(mmap);
        drop(file);

        let elapsed = load_start.elapsed();
        log::info!(
            "CPU INT{} cache loaded in {:.1}s: {} layers × {} experts (+ {} shared), {:.1} GB",
            expected_bits,
            elapsed.as_secs_f64(),
            num_layers_to_load, config.n_routed_experts,
            shared_experts_cpu.len(),
            offset as f64 / 1e9,
        );

        Ok((experts_cpu, shared_experts_cpu))
    }

    /// Quick check for pre-quantized group_size without loading full weights.
    /// Returns Some(group_size) if model has pre-quantized experts, None otherwise.
    fn detect_group_size_hint(model_dir: &Path, config: &ModelConfig) -> Option<usize> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_str = std::fs::read_to_string(&index_path).ok()?;
        let index: SafetensorsIndex = serde_json::from_str(&index_str).ok()?;
        let layers_prefix = detect_expert_prefix(&index.weight_map).ok()?;

        let first_moe_layer = config.first_k_dense_replace;
        let packed_name = format!(
            "{layers_prefix}.layers.{first_moe_layer}.mlp.experts.0.gate_proj.weight_packed"
        );

        // If weight_packed exists, model is pre-quantized — detect group_size
        let _shard_name = index.weight_map.get(&packed_name)?;

        let scale_name = format!(
            "{layers_prefix}.layers.{first_moe_layer}.mlp.experts.0.gate_proj.weight_scale"
        );
        let shape_name = format!(
            "{layers_prefix}.layers.{first_moe_layer}.mlp.experts.0.gate_proj.weight_shape"
        );

        // Open just the shard(s) needed for scale and shape
        let scale_shard_name = index.weight_map.get(&scale_name)?;
        let shape_shard_name = index.weight_map.get(&shape_name)?;

        let mut shards: HashMap<String, MmapSafetensors> = HashMap::new();
        for name in [scale_shard_name, shape_shard_name] {
            if !shards.contains_key(name) {
                let path = model_dir.join(name);
                let st = MmapSafetensors::open(&path).ok()?;
                shards.insert(name.clone(), st);
            }
        }

        match detect_prequant_group_size(&index.weight_map, &shards, &layers_prefix, first_moe_layer) {
            Ok(gs) => {
                log::info!("Detected pre-quantized group_size={gs} for cache path");
                Some(gs)
            }
            Err(e) => {
                log::warn!("Failed to detect pre-quantized group_size: {e}");
                None
            }
        }
    }

    /// Get expert weights for a given MoE layer index and expert index.
    /// moe_layer_idx is 0-based within MoE layers (not absolute layer index).
    pub fn get_expert(&self, moe_layer_idx: usize, expert_idx: usize) -> &ExpertWeights {
        &self.experts[moe_layer_idx][expert_idx]
    }

    /// Get shared expert weights for a given MoE layer index.
    /// Returns None if no shared experts.
    pub fn get_shared_expert(&self, moe_layer_idx: usize) -> Option<&ExpertWeights> {
        self.shared_experts.get(moe_layer_idx)
    }

    /// Number of MoE layers loaded.
    pub fn num_moe_layers(&self) -> usize {
        if !self.experts_gguf.is_empty() {
            self.experts_gguf.len()
        } else if !self.experts_cpu.is_empty() {
            self.experts_cpu.len()
        } else if !self.experts_gpu.is_empty() {
            self.experts_gpu.len()
        } else {
            self.experts.len()
        }
    }

    /// Whether CPU decode weights (transposed format) have been populated.
    pub fn has_cpu_weights(&self) -> bool {
        !self.experts_cpu.is_empty()
    }

    /// Whether GPU prefill weights (Marlin format) have been populated.
    pub fn has_gpu_weights(&self) -> bool {
        !self.experts_gpu.is_empty()
    }

    /// Backward compat: `has_unified()` returns true when either CPU or GPU weights exist.
    pub fn has_unified(&self) -> bool {
        self.has_cpu_weights() || self.has_gpu_weights()
    }

    /// Get CPU decode expert weights for a given MoE layer and expert index.
    pub fn get_expert_cpu(&self, moe_layer_idx: usize, expert_idx: usize) -> &UnifiedExpertWeights {
        &self.experts_cpu[moe_layer_idx][expert_idx]
    }

    /// Get GPU prefill expert weights for a given MoE layer and expert index.
    pub fn get_expert_gpu(&self, moe_layer_idx: usize, expert_idx: usize) -> &UnifiedExpertWeights {
        &self.experts_gpu[moe_layer_idx][expert_idx]
    }

    /// Get CPU decode shared expert weights for a given MoE layer index.
    pub fn get_shared_expert_cpu(&self, moe_layer_idx: usize) -> Option<&UnifiedExpertWeights> {
        self.shared_experts_cpu.get(moe_layer_idx)
    }

    /// Get GPU prefill shared expert weights for a given MoE layer index.
    pub fn get_shared_expert_gpu(&self, moe_layer_idx: usize) -> Option<&UnifiedExpertWeights> {
        self.shared_experts_gpu.get(moe_layer_idx)
    }

    /// Backward compat: returns CPU expert ref (used by moe_forward_unified).
    pub fn get_expert_unified(&self, moe_layer_idx: usize, expert_idx: usize) -> &UnifiedExpertWeights {
        if self.has_cpu_weights() {
            self.get_expert_cpu(moe_layer_idx, expert_idx)
        } else {
            self.get_expert_gpu(moe_layer_idx, expert_idx)
        }
    }

    /// Backward compat: returns CPU shared expert ref.
    pub fn get_shared_expert_unified(&self, moe_layer_idx: usize) -> Option<&UnifiedExpertWeights> {
        if self.has_cpu_weights() {
            self.get_shared_expert_cpu(moe_layer_idx)
        } else {
            self.get_shared_expert_gpu(moe_layer_idx)
        }
    }

    /// Whether native GGUF expert weights are loaded (for CPU decode).
    pub fn has_gguf(&self) -> bool {
        !self.experts_gguf.is_empty()
    }

    /// Get native GGUF expert weights for a given MoE layer and expert index.
    pub fn get_expert_gguf(&self, moe_layer_idx: usize, expert_idx: usize) -> &GgufExpertWeights {
        &self.experts_gguf[moe_layer_idx][expert_idx]
    }

    /// Get native GGUF shared expert weights for a given MoE layer index.
    pub fn get_shared_expert_gguf(&self, moe_layer_idx: usize) -> Option<&GgufExpertWeights> {
        self.shared_experts_gguf.get(moe_layer_idx)
    }

    /// Migrate CPU expert weights to NUMA nodes.
    /// Returns the number of successfully migrated experts.
    pub fn migrate_numa_unified(&mut self, map: &crate::numa::NumaExpertMap) -> usize {
        use crate::numa::migrate_vec_to_node;

        let start = std::time::Instant::now();
        let mut migrated = 0;
        let mut failed = 0;

        for (layer_idx, layer) in self.experts_cpu.iter_mut().enumerate() {
            for (expert_idx, expert) in layer.iter_mut().enumerate() {
                let node = map.node_for(layer_idx, expert_idx);

                let ok = migrate_vec_to_node(&mut expert.w13_packed, node)
                    && migrate_vec_to_node(&mut expert.w13_scales, node)
                    && migrate_vec_to_node(&mut expert.w2_packed, node)
                    && migrate_vec_to_node(&mut expert.w2_scales, node);

                if ok {
                    migrated += 1;
                } else {
                    failed += 1;
                }
            }
        }

        let elapsed = start.elapsed();
        log::info!(
            "NUMA migration (CPU experts): {migrated} experts migrated, {failed} failed, in {:.1}s",
            elapsed.as_secs_f64(),
        );

        migrated
    }

    /// Load CPU expert weights from a GGUF file (dequant → BF16 → re-quantize to our format).
    ///
    /// The GGUF file provides pre-quantized expert weights (Q4_K, Q5_K, Q6_K, etc.)
    /// which we dequantize to FP32, convert to BF16, then re-quantize to our INT4/INT8
    /// CPU transposed format. GPU Marlin cache is NOT loaded here — it's still built
    /// from BF16 safetensors by the normal `load_from_hf` path.
    ///
    /// This populates `experts_cpu` and `shared_experts_cpu`.
    ///
    /// `model_dir`: path to HF model directory (for config.json)
    /// `gguf_path`: path to the GGUF file
    /// `group_size`: quantization group size (128 default)
    /// `cpu_num_bits`: 4 or 8 for CPU decode format
    /// `gpu_num_bits`: for GPU (Marlin), still loaded from safetensors
    /// `max_layers`: optional limit on number of MoE layers to load
    /// `start_layer`: optional start MoE layer index
    pub fn load_from_gguf(
        model_dir: &Path,
        gguf_path: &Path,
        group_size: usize,
        max_layers: Option<usize>,
        start_layer: Option<usize>,
        cpu_num_bits: u8,
        gpu_num_bits: u8,
        gguf_native: bool,
    ) -> Result<Self, String> {
        let start = std::time::Instant::now();

        // Parse config.json from HF model dir
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let raw_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        let config = ModelConfig::from_json(&raw_json)
            .map_err(|e| format!("Failed to extract MoE config: {e}"))?;

        log::info!(
            "GGUF loading: hidden={}, intermediate={}, experts={}, top-{}, layers={}, cpu_bits={}",
            config.hidden_size, config.moe_intermediate_size, config.n_routed_experts,
            config.num_experts_per_tok, config.num_hidden_layers, cpu_num_bits,
        );

        let total_moe_layers = config.num_hidden_layers - config.first_k_dense_replace;
        let moe_start = start_layer.unwrap_or(0);
        let remaining = total_moe_layers - moe_start;
        let num_moe_layers = match max_layers {
            Some(n) => n.min(remaining),
            None => remaining,
        };
        let config_hash = fnv1a(config_str.as_bytes());

        // Open GGUF file
        log::info!("Opening GGUF: {}", gguf_path.display());
        let gguf = crate::gguf::GgufFile::open(gguf_path)?;

        let merged = gguf.has_merged_experts();
        log::info!(
            "GGUF expert format: {}",
            if merged { "merged (ffn_gate_exps)" } else { "per-expert (ffn_gate.E)" },
        );

        // ── Phase 1: Load GPU Marlin cache from safetensors (unchanged) ──
        let effective_gs_hint = Self::detect_group_size_hint(model_dir, &config);
        let cache_gs = effective_gs_hint.unwrap_or(group_size);
        let mut experts_gpu: Vec<Vec<UnifiedExpertWeights>> = Vec::new();
        let mut shared_experts_gpu: Vec<UnifiedExpertWeights> = Vec::new();
        let mut effective_gs = cache_gs;
        let mut gpu_loaded = false;

        // Try loading existing Marlin cache
        for try_gs in &[cache_gs, group_size, 32, 64, 128] {
            if gpu_loaded { break; }
            let try_path = cache_path_marlin(model_dir, *try_gs);
            if try_path.exists() {
                match Self::load_marlin_cache(
                    &try_path, &config, *try_gs, total_moe_layers, config_hash,
                    moe_start, num_moe_layers,
                ) {
                    Ok(store) => {
                        log::info!(
                            "Loaded GPU Marlin cache in {:.1}s (gs={})",
                            start.elapsed().as_secs_f64(), try_gs,
                        );
                        experts_gpu = store.experts_gpu;
                        shared_experts_gpu = store.shared_experts_gpu;
                        effective_gs = *try_gs;
                        gpu_loaded = true;
                    }
                    Err(e) => {
                        if *try_gs == cache_gs {
                            log::warn!("Marlin cache invalid (gs={}): {e}", try_gs);
                        }
                    }
                }
            }
        }

        // Build Marlin cache if not found
        if !gpu_loaded {
            let mpath = cache_path_marlin(model_dir, cache_gs);
            log::info!("No Marlin cache found, building from safetensors...");
            let built_gs = Self::build_marlin_cache_locked(
                model_dir, &config, group_size, total_moe_layers, &mpath, config_hash,
            )?;
            effective_gs = built_gs;

            for try_gs in &[built_gs, cache_gs, group_size, 32, 64, 128] {
                if gpu_loaded { break; }
                let try_path = cache_path_marlin(model_dir, *try_gs);
                if try_path.exists() {
                    if let Ok(store) = Self::load_marlin_cache(
                        &try_path, &config, *try_gs, total_moe_layers, config_hash,
                        moe_start, num_moe_layers,
                    ) {
                        experts_gpu = store.experts_gpu;
                        shared_experts_gpu = store.shared_experts_gpu;
                        effective_gs = *try_gs;
                        gpu_loaded = true;
                    }
                }
            }
        }

        // ── Phase 2: CPU experts — try AVX2 cache first, then build from GGUF ──
        let mut experts_cpu: Vec<Vec<UnifiedExpertWeights>> = Vec::new();
        let mut shared_experts_cpu: Vec<UnifiedExpertWeights> = Vec::new();
        let mut experts_gguf: Vec<Vec<GgufExpertWeights>> = Vec::new();
        let mut shared_experts_gguf: Vec<GgufExpertWeights> = Vec::new();
        let mut cpu_loaded = false;

        if gguf_native {
            log::info!("GGUF native mode — bypassing AVX2 cache, loading raw GGUF blocks");
        }

        // Step 1: Try loading existing GGUF→AVX2 cache (unless gguf_native)
        if !gguf_native {
            let avx2_cache_path = cache_path_gguf_avx2(model_dir, effective_gs);
            if avx2_cache_path.exists() {
                match Self::load_gguf_cpu_cache(
                    &avx2_cache_path, &config, effective_gs, total_moe_layers, config_hash,
                    moe_start, num_moe_layers,
                ) {
                    Ok((cpu_exp, cpu_shared, w13b, w2b)) => {
                        log::info!(
                            "Loaded GGUF→AVX2 CPU cache in {:.1}s: w13=INT{}, w2=INT{}, {} layers",
                            start.elapsed().as_secs_f64(), w13b, w2b, num_moe_layers,
                        );
                        experts_cpu = cpu_exp;
                        shared_experts_cpu = cpu_shared;
                        cpu_loaded = true;
                    }
                    Err(e) => log::warn!("GGUF AVX2 cache invalid: {e}"),
                }
            }
        }

        // Step 2: Build AVX2 cache from GGUF if needed (unless gguf_native)
        if !cpu_loaded && !gguf_native {
            let avx2_cache_path = cache_path_gguf_avx2(model_dir, effective_gs);
            log::info!("No GGUF→AVX2 cache found, building from GGUF...");
            let (w13b, w2b) = Self::streaming_build_cpu_cache_from_gguf(
                model_dir, gguf_path, &config, effective_gs,
                total_moe_layers, &avx2_cache_path, config_hash,
            )?;

            // Load the just-built cache
            match Self::load_gguf_cpu_cache(
                &avx2_cache_path, &config, effective_gs, total_moe_layers, config_hash,
                moe_start, num_moe_layers,
            ) {
                Ok((cpu_exp, cpu_shared, _, _)) => {
                    log::info!(
                        "Loaded GGUF→AVX2 cache after build in {:.1}s: w13=INT{}, w2=INT{}",
                        start.elapsed().as_secs_f64(), w13b, w2b,
                    );
                    experts_cpu = cpu_exp;
                    shared_experts_cpu = cpu_shared;
                    cpu_loaded = true;
                }
                Err(e) => log::warn!("Failed to load built GGUF AVX2 cache: {e}"),
            }
        }

        // Step 3: Fall back to raw GGUF native if requested or if AVX2 cache failed
        if !cpu_loaded {
            log::info!(
                "Loading CPU experts from GGUF native ({} layers × {} experts)...",
                num_moe_layers, config.n_routed_experts,
            );

            let h = config.hidden_size;
            let m = config.moe_intermediate_size;
            let n_experts = config.n_routed_experts;

            for moe_idx in moe_start..(moe_start + num_moe_layers) {
                let abs_layer = moe_idx + config.first_k_dense_replace;
                let layer_start = std::time::Instant::now();
                let mut layer_experts = Vec::with_capacity(n_experts);

                if merged {
                    let gate_name = format!("blk.{abs_layer}.ffn_gate_exps.weight");
                    let up_name = format!("blk.{abs_layer}.ffn_up_exps.weight");
                    let down_name = format!("blk.{abs_layer}.ffn_down_exps.weight");

                    let gate_info = gguf.tensors.get(&gate_name)
                        .ok_or_else(|| format!("Missing tensor: {gate_name}"))?;
                    let up_info = gguf.tensors.get(&up_name)
                        .ok_or_else(|| format!("Missing tensor: {up_name}"))?;
                    let down_info = gguf.tensors.get(&down_name)
                        .ok_or_else(|| format!("Missing tensor: {down_name}"))?;

                    let gate_data = gguf.tensor_data(gate_info)?;
                    let up_data = gguf.tensor_data(up_info)?;
                    let down_data = gguf.tensor_data(down_info)?;

                    let gate_type = gate_info.dtype;
                    let gate_expert_elements = m * h;
                    let gate_expert_blocks = gate_expert_elements / gate_type.block_size();
                    let gate_expert_bytes = gate_expert_blocks * gate_type.block_bytes();

                    let down_type = down_info.dtype;
                    let down_expert_elements = h * m;
                    let down_expert_blocks = down_expert_elements / down_type.block_size();
                    let down_expert_bytes = down_expert_blocks * down_type.block_bytes();

                    let up_expert_bytes = gate_expert_bytes;

                    if moe_idx == moe_start {
                        log::info!(
                            "GGUF expert layout: gate/up={} ({} bytes/expert), down={} ({} bytes/expert)",
                            gate_type.name(), gate_expert_bytes, down_type.name(), down_expert_bytes,
                        );
                    }

                    for eidx in 0..n_experts {
                        let gate_start = eidx * gate_expert_bytes;
                        let up_start = eidx * up_expert_bytes;
                        let down_start = eidx * down_expert_bytes;

                        layer_experts.push(GgufExpertWeights {
                            gate_data: gate_data[gate_start..gate_start + gate_expert_bytes].to_vec(),
                            up_data: up_data[up_start..up_start + up_expert_bytes].to_vec(),
                            down_data: down_data[down_start..down_start + down_expert_bytes].to_vec(),
                            gate_up_type: gate_type,
                            down_type,
                            intermediate_size: m,
                            hidden_size: h,
                        });
                    }
                } else {
                    for eidx in 0..n_experts {
                        let gate_name = format!("blk.{abs_layer}.ffn_gate.{eidx}.weight");
                        let up_name = format!("blk.{abs_layer}.ffn_up.{eidx}.weight");
                        let down_name = format!("blk.{abs_layer}.ffn_down.{eidx}.weight");

                        let gate_info = gguf.tensors.get(&gate_name)
                            .ok_or_else(|| format!("Missing tensor: {gate_name}"))?;
                        let up_info = gguf.tensors.get(&up_name)
                            .ok_or_else(|| format!("Missing tensor: {up_name}"))?;
                        let down_info = gguf.tensors.get(&down_name)
                            .ok_or_else(|| format!("Missing tensor: {down_name}"))?;

                        layer_experts.push(GgufExpertWeights {
                            gate_data: gguf.tensor_data(gate_info)?.to_vec(),
                            up_data: gguf.tensor_data(up_info)?.to_vec(),
                            down_data: gguf.tensor_data(down_info)?.to_vec(),
                            gate_up_type: gate_info.dtype,
                            down_type: down_info.dtype,
                            intermediate_size: m,
                            hidden_size: h,
                        });
                    }
                }

                let elapsed = layer_start.elapsed();
                let layers_done = experts_gguf.len() + 1;
                log::info!(
                    "GGUF layer {abs_layer}: {} experts copied in {:.1}s [{layers_done}/{num_moe_layers}]",
                    n_experts, elapsed.as_secs_f64(),
                );
                experts_gguf.push(layer_experts);

                if layers_done % 5 == 0 || layers_done == num_moe_layers {
                    crate::syscheck::log_memory_usage(
                        &format!("[GGUF] after {layers_done}/{num_moe_layers} layers"),
                    );
                }
            }

            // Load shared experts from GGUF (if present)
            if config.n_shared_experts > 0 {
                let shared_intermediate = config.n_shared_experts * config.moe_intermediate_size;
                log::info!(
                    "Loading shared experts from GGUF: n_shared={}, intermediate={}",
                    config.n_shared_experts, shared_intermediate,
                );

                for moe_idx in moe_start..(moe_start + num_moe_layers) {
                    let abs_layer = moe_idx + config.first_k_dense_replace;

                    if let Some((gate_name, up_name, down_name)) = gguf.find_shared_expert_tensors(abs_layer) {
                        let gate_info = gguf.tensors.get(&gate_name)
                            .ok_or_else(|| format!("Missing shared tensor: {gate_name}"))?;
                        let up_info = gguf.tensors.get(&up_name)
                            .ok_or_else(|| format!("Missing shared tensor: {up_name}"))?;
                        let down_info = gguf.tensors.get(&down_name)
                            .ok_or_else(|| format!("Missing shared tensor: {down_name}"))?;

                        shared_experts_gguf.push(GgufExpertWeights {
                            gate_data: gguf.tensor_data(gate_info)?.to_vec(),
                            up_data: gguf.tensor_data(up_info)?.to_vec(),
                            down_data: gguf.tensor_data(down_info)?.to_vec(),
                            gate_up_type: gate_info.dtype,
                            down_type: down_info.dtype,
                            intermediate_size: shared_intermediate,
                            hidden_size: h,
                        });
                    }
                }
                log::info!("Loaded {} shared expert layers from GGUF", shared_experts_gguf.len());
            }
        }

        let total_elapsed = start.elapsed();
        let mode = if cpu_loaded { "AVX2" } else { "native" };
        log::info!(
            "GGUF loading done ({mode}) in {:.1}s: {} MoE layers, GPU={}",
            total_elapsed.as_secs_f64(), num_moe_layers,
            if gpu_loaded { "Marlin" } else { "none" },
        );

        Ok(WeightStore {
            experts: Vec::new(),
            shared_experts: Vec::new(),
            experts_cpu,
            shared_experts_cpu,
            experts_gpu,
            shared_experts_gpu,
            experts_gguf,
            shared_experts_gguf,
            config: config.clone(),
            group_size: effective_gs,
            cpu_num_bits,
            gpu_num_bits,
        })
    }

    /// Build GGUF-sourced AVX2 transposed CPU cache (v5).
    ///
    /// Reads GGUF file, dequantizes each expert to FP32, re-quantizes to AVX2
    /// transposed format at the same bit width as the GGUF source, and writes
    /// to a disk cache. Returns (w13_bits, w2_bits).
    fn streaming_build_cpu_cache_from_gguf(
        _model_dir: &Path,
        gguf_path: &Path,
        config: &ModelConfig,
        group_size: usize,
        total_moe_layers: usize,
        cache_path: &Path,
        config_hash: u64,
    ) -> Result<(u8, u8), String> {
        use crate::gguf;

        log::info!(
            "Building GGUF→AVX2 CPU cache: {} MoE layers → {}",
            total_moe_layers, cache_path.display(),
        );
        crate::syscheck::log_memory_usage("before streaming_build_cpu_cache_from_gguf");

        // Open GGUF
        let gguf_file = gguf::GgufFile::open(gguf_path)?;
        let merged = gguf_file.has_merged_experts();
        let h = config.hidden_size;
        let m = config.moe_intermediate_size;
        let n_experts = config.n_routed_experts;

        // Scan ALL layers to determine target precision (Q4_K_M has mixed types across layers)
        let mut w13_bits: u8 = 4;
        let mut w2_bits: u8 = 4;
        let mut gate_types = std::collections::BTreeSet::new();
        let mut down_types = std::collections::BTreeSet::new();
        for moe_idx in 0..total_moe_layers {
            let abs_layer = moe_idx + config.first_k_dense_replace;
            if merged {
                let gate_name = format!("blk.{abs_layer}.ffn_gate_exps.weight");
                let down_name = format!("blk.{abs_layer}.ffn_down_exps.weight");
                if let Some(gt) = gguf_file.tensors.get(&gate_name) {
                    let (bits, _) = gguf_type_to_cpu_bits(gt.dtype);
                    w13_bits = w13_bits.max(bits);
                    gate_types.insert(gt.dtype.name().to_string());
                }
                if let Some(dt) = gguf_file.tensors.get(&down_name) {
                    let (bits, _) = gguf_type_to_cpu_bits(dt.dtype);
                    w2_bits = w2_bits.max(bits);
                    down_types.insert(dt.dtype.name().to_string());
                }
            } else {
                let gate_name = format!("blk.{abs_layer}.ffn_gate.0.weight");
                let down_name = format!("blk.{abs_layer}.ffn_down.0.weight");
                if let Some(gt) = gguf_file.tensors.get(&gate_name) {
                    let (bits, _) = gguf_type_to_cpu_bits(gt.dtype);
                    w13_bits = w13_bits.max(bits);
                    gate_types.insert(gt.dtype.name().to_string());
                }
                if let Some(dt) = gguf_file.tensors.get(&down_name) {
                    let (bits, _) = gguf_type_to_cpu_bits(dt.dtype);
                    w2_bits = w2_bits.max(bits);
                    down_types.insert(dt.dtype.name().to_string());
                }
            }
        }

        // Warn about non-exact conversions
        let gate_types_str: Vec<_> = gate_types.iter().collect();
        let down_types_str: Vec<_> = down_types.iter().collect();
        for gt_name in &gate_types_str {
            // Find this type and check exactness
            for moe_idx in 0..total_moe_layers {
                let abs_layer = moe_idx + config.first_k_dense_replace;
                let name = if merged {
                    format!("blk.{abs_layer}.ffn_gate_exps.weight")
                } else {
                    format!("blk.{abs_layer}.ffn_gate.0.weight")
                };
                if let Some(t) = gguf_file.tensors.get(&name) {
                    if t.dtype.name() == gt_name.as_str() {
                        let (_, exact) = gguf_type_to_cpu_bits(t.dtype);
                        if !exact {
                            log::warn!(
                                "GGUF gate/up type {} will be rounded to INT{} (not an exact match)",
                                gt_name, w13_bits,
                            );
                        }
                        break;
                    }
                }
            }
        }
        for dt_name in &down_types_str {
            for moe_idx in 0..total_moe_layers {
                let abs_layer = moe_idx + config.first_k_dense_replace;
                let name = if merged {
                    format!("blk.{abs_layer}.ffn_down_exps.weight")
                } else {
                    format!("blk.{abs_layer}.ffn_down.0.weight")
                };
                if let Some(t) = gguf_file.tensors.get(&name) {
                    if t.dtype.name() == dt_name.as_str() {
                        let (_, exact) = gguf_type_to_cpu_bits(t.dtype);
                        if !exact {
                            log::warn!(
                                "GGUF down type {} will be rounded to INT{} (not an exact match)",
                                dt_name, w2_bits,
                            );
                        }
                        break;
                    }
                }
            }
        }

        log::info!(
            "GGUF types: gate/up=[{}] → INT{}, down=[{}] → INT{}{}",
            gate_types_str.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
            w13_bits,
            down_types_str.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
            w2_bits,
            if w13_bits == w2_bits { String::new() } else { " (mixed precision)".to_string() },
        );

        // Create cache directory + temp file
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create cache dir: {e}"))?;
        }
        let tmp_path = cache_path.with_extension("bin.tmp");
        let file = std::fs::File::create(&tmp_path)
            .map_err(|e| format!("Failed to create GGUF CPU cache file: {e}"))?;
        let mut w = std::io::BufWriter::with_capacity(4 * 1024 * 1024, file);

        // Write v5 header
        write_cpu_cache_header_v5(&mut w, config, group_size, total_moe_layers, config_hash, w13_bits, w2_bits)?;

        let overall_start = std::time::Instant::now();

        // Stream routed experts layer by layer
        for moe_idx in 0..total_moe_layers {
            let abs_layer = moe_idx + config.first_k_dense_replace;
            let layer_start = std::time::Instant::now();

            if merged {
                // Merged expert tensors: dequant whole tensor, slice per-expert
                let gate_name = format!("blk.{abs_layer}.ffn_gate_exps.weight");
                let up_name = format!("blk.{abs_layer}.ffn_up_exps.weight");
                let down_name = format!("blk.{abs_layer}.ffn_down_exps.weight");

                let gate_info = gguf_file.tensors.get(&gate_name)
                    .ok_or_else(|| format!("Missing tensor: {gate_name}"))?;
                let up_info = gguf_file.tensors.get(&up_name)
                    .ok_or_else(|| format!("Missing tensor: {up_name}"))?;
                let down_info = gguf_file.tensors.get(&down_name)
                    .ok_or_else(|| format!("Missing tensor: {down_name}"))?;

                let gate_data = gguf_file.tensor_data(gate_info)?;
                let up_data = gguf_file.tensor_data(up_info)?;
                let down_data = gguf_file.tensor_data(down_info)?;

                // Use per-tensor dtypes (Q4_K_M has mixed types across layers)
                let layer_gate_type = gate_info.dtype;
                let layer_up_type = up_info.dtype;
                let layer_down_type = down_info.dtype;

                let gate_expert_elements = m * h;
                let gate_expert_blocks = gate_expert_elements / layer_gate_type.block_size();
                let gate_expert_bytes = gate_expert_blocks * layer_gate_type.block_bytes();

                let up_expert_elements = m * h;
                let up_expert_blocks = up_expert_elements / layer_up_type.block_size();
                let up_expert_bytes = up_expert_blocks * layer_up_type.block_bytes();

                let down_expert_elements = h * m;
                let down_expert_blocks = down_expert_elements / layer_down_type.block_size();
                let down_expert_bytes = down_expert_blocks * layer_down_type.block_bytes();

                // Parallel: dequant + requant each expert
                let expert_results: Vec<UnifiedExpertWeights> = (0..n_experts)
                    .into_par_iter()
                    .map(|eidx| {
                        let gate_slice = &gate_data[eidx * gate_expert_bytes..(eidx + 1) * gate_expert_bytes];
                        let up_slice = &up_data[eidx * up_expert_bytes..(eidx + 1) * up_expert_bytes];
                        let down_slice = &down_data[eidx * down_expert_bytes..(eidx + 1) * down_expert_bytes];

                        let gate_f32 = gguf::dequantize_raw_data(layer_gate_type, gate_slice, gate_expert_elements)
                            .expect("Failed to dequant gate");
                        let up_f32 = gguf::dequantize_raw_data(layer_up_type, up_slice, up_expert_elements)
                            .expect("Failed to dequant up");
                        let down_f32 = gguf::dequantize_raw_data(layer_down_type, down_slice, down_expert_elements)
                            .expect("Failed to dequant down");

                        Self::gguf_expert_from_f32(
                            &gate_f32, &up_f32, &down_f32,
                            m, h, group_size, w13_bits, w2_bits,
                        )
                    })
                    .collect();

                for cpu_exp in &expert_results {
                    write_vec_u32(&mut w, &cpu_exp.w13_packed)?;
                    write_vec_u16(&mut w, &cpu_exp.w13_scales)?;
                    write_vec_u32(&mut w, &cpu_exp.w2_packed)?;
                    write_vec_u16(&mut w, &cpu_exp.w2_scales)?;
                }
            } else {
                // Per-expert tensors
                let expert_results: Vec<UnifiedExpertWeights> = (0..n_experts)
                    .into_par_iter()
                    .map(|eidx| {
                        let gate_name = format!("blk.{abs_layer}.ffn_gate.{eidx}.weight");
                        let up_name = format!("blk.{abs_layer}.ffn_up.{eidx}.weight");
                        let down_name = format!("blk.{abs_layer}.ffn_down.{eidx}.weight");

                        let gate_info = gguf_file.tensors.get(&gate_name)
                            .unwrap_or_else(|| panic!("Missing tensor: {gate_name}"));
                        let up_info = gguf_file.tensors.get(&up_name)
                            .unwrap_or_else(|| panic!("Missing tensor: {up_name}"));
                        let down_info = gguf_file.tensors.get(&down_name)
                            .unwrap_or_else(|| panic!("Missing tensor: {down_name}"));

                        let gate_f32 = gguf_file.dequantize_tensor(gate_info)
                            .expect("Failed to dequant gate");
                        let up_f32 = gguf_file.dequantize_tensor(up_info)
                            .expect("Failed to dequant up");
                        let down_f32 = gguf_file.dequantize_tensor(down_info)
                            .expect("Failed to dequant down");

                        Self::gguf_expert_from_f32(
                            &gate_f32, &up_f32, &down_f32,
                            m, h, group_size, w13_bits, w2_bits,
                        )
                    })
                    .collect();

                for cpu_exp in &expert_results {
                    write_vec_u32(&mut w, &cpu_exp.w13_packed)?;
                    write_vec_u16(&mut w, &cpu_exp.w13_scales)?;
                    write_vec_u32(&mut w, &cpu_exp.w2_packed)?;
                    write_vec_u16(&mut w, &cpu_exp.w2_scales)?;
                }
            }

            let layers_done = moe_idx + 1;
            let layer_elapsed = layer_start.elapsed();
            if layers_done % 5 == 0 || layers_done == total_moe_layers {
                crate::syscheck::log_memory_usage(&format!(
                    "GGUF→AVX2 cache: {layers_done}/{total_moe_layers} layers ({:.1}s/layer)",
                    layer_elapsed.as_secs_f64(),
                ));
            } else {
                log::info!(
                    "  Layer {abs_layer}: {} experts in {:.1}s [{layers_done}/{total_moe_layers}]",
                    n_experts, layer_elapsed.as_secs_f64(),
                );
            }
        }

        // Stream shared experts
        if config.n_shared_experts > 0 {
            let shared_intermediate = config.n_shared_experts * config.moe_intermediate_size;
            log::info!("Streaming shared experts for GGUF cache ({} layers)...", total_moe_layers);

            for moe_idx in 0..total_moe_layers {
                let abs_layer = moe_idx + config.first_k_dense_replace;

                if let Some((gate_name, up_name, down_name)) = gguf_file.find_shared_expert_tensors(abs_layer) {
                    let gate_info = gguf_file.tensors.get(&gate_name)
                        .ok_or_else(|| format!("Missing shared tensor: {gate_name}"))?;
                    let up_info = gguf_file.tensors.get(&up_name)
                        .ok_or_else(|| format!("Missing shared tensor: {up_name}"))?;
                    let down_info = gguf_file.tensors.get(&down_name)
                        .ok_or_else(|| format!("Missing shared tensor: {down_name}"))?;

                    let gate_f32 = gguf_file.dequantize_tensor(gate_info)?;
                    let up_f32 = gguf_file.dequantize_tensor(up_info)?;
                    let down_f32 = gguf_file.dequantize_tensor(down_info)?;

                    let cpu_exp = Self::gguf_expert_from_f32(
                        &gate_f32, &up_f32, &down_f32,
                        shared_intermediate, h, group_size, w13_bits, w2_bits,
                    );

                    write_vec_u32(&mut w, &cpu_exp.w13_packed)?;
                    write_vec_u16(&mut w, &cpu_exp.w13_scales)?;
                    write_vec_u32(&mut w, &cpu_exp.w2_packed)?;
                    write_vec_u16(&mut w, &cpu_exp.w2_scales)?;
                } else {
                    return Err(format!(
                        "Missing shared expert tensors for layer {abs_layer}"
                    ));
                }
            }
        }

        // Flush + atomic rename
        w.flush().map_err(|e| format!("Flush error: {e}"))?;
        drop(w);
        std::fs::rename(&tmp_path, cache_path)
            .map_err(|e| format!("Failed to rename GGUF CPU cache file: {e}"))?;

        // Evict GGUF page cache, then free mmap and reclaim RAM
        gguf_file.evict_page_cache();
        drop(gguf_file);
        #[cfg(target_os = "linux")]
        unsafe { libc::malloc_trim(0); }

        let elapsed = overall_start.elapsed();
        let size = std::fs::metadata(cache_path).map(|m| m.len()).unwrap_or(0);
        log::info!(
            "GGUF→AVX2 cache built: {:.1} GB in {:.1}s ({:.1} GB/s), w13=INT{}, w2=INT{}",
            size as f64 / 1e9,
            elapsed.as_secs_f64(),
            size as f64 / 1e9 / elapsed.as_secs_f64(),
            w13_bits, w2_bits,
        );
        crate::syscheck::log_memory_usage("after streaming_build_cpu_cache_from_gguf");

        Ok((w13_bits, w2_bits))
    }

    /// Load v5 GGUF-sourced CPU cache from disk.
    fn load_gguf_cpu_cache(
        path: &Path,
        config: &ModelConfig,
        group_size: usize,
        total_moe_layers: usize,
        config_hash: u64,
        start_moe_layer: usize,
        num_layers_to_load: usize,
    ) -> Result<(Vec<Vec<UnifiedExpertWeights>>, Vec<UnifiedExpertWeights>, u8, u8), String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open GGUF CPU cache: {e}"))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("Failed to mmap GGUF CPU cache: {e}"))?;

        // Validate header
        if mmap.len() < CACHE_HEADER_SIZE {
            return Err("GGUF CPU cache too small for header".to_string());
        }
        if &mmap[0..4] != CACHE_MAGIC {
            return Err("Bad magic in GGUF CPU cache".to_string());
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != CACHE_VERSION_CPU_GGUF {
            return Err(format!("Cache version {version}, expected {CACHE_VERSION_CPU_GGUF} (GGUF CPU)"));
        }

        let h_hidden = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let h_intermediate = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let h_n_experts = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;
        let h_num_layers = u64::from_le_bytes(mmap[32..40].try_into().unwrap()) as usize;
        let h_group_size = u64::from_le_bytes(mmap[40..48].try_into().unwrap()) as usize;
        let h_config_hash = u64::from_le_bytes(mmap[48..56].try_into().unwrap());
        let packed_meta = u64::from_le_bytes(mmap[56..64].try_into().unwrap());
        let h_n_shared = (packed_meta & 0xFFFF) as usize;
        let h_w13_bits = ((packed_meta >> 48) & 0xFF) as u8;
        let h_w2_bits = ((packed_meta >> 56) & 0xFF) as u8;

        if h_hidden != config.hidden_size
            || h_intermediate != config.moe_intermediate_size
            || h_n_experts != config.n_routed_experts
            || h_num_layers != total_moe_layers
            || h_group_size != group_size
        {
            return Err(format!(
                "GGUF CPU cache header mismatch: file has {}h/{}m/{}e/{}L/g{}, expected {}h/{}m/{}e/{}L/g{}",
                h_hidden, h_intermediate, h_n_experts, h_num_layers, h_group_size,
                config.hidden_size, config.moe_intermediate_size, config.n_routed_experts,
                total_moe_layers, group_size,
            ));
        }
        if h_config_hash != config_hash {
            return Err("Config hash mismatch in GGUF CPU cache".to_string());
        }
        if h_n_shared != config.n_shared_experts {
            return Err(format!(
                "Shared expert count mismatch: cache={h_n_shared}, config={}",
                config.n_shared_experts,
            ));
        }
        if h_w13_bits != 4 && h_w13_bits != 8 {
            return Err(format!("Invalid w13_bits in cache: {h_w13_bits}"));
        }
        if h_w2_bits != 4 && h_w2_bits != 8 {
            return Err(format!("Invalid w2_bits in cache: {h_w2_bits}"));
        }

        // Validate file size
        let shared_intermediate = config.n_shared_experts * config.moe_intermediate_size;
        let expected = expected_gguf_cpu_cache_size(
            config, group_size, h_w13_bits, h_w2_bits,
            total_moe_layers, config.n_shared_experts, shared_intermediate,
        );
        if mmap.len() != expected {
            return Err(format!(
                "GGUF CPU cache size mismatch: expected {} bytes, got {}",
                expected, mmap.len(),
            ));
        }

        log::info!(
            "Loading GGUF→AVX2 CPU cache: w13=INT{}, w2=INT{}, {} layers ({})",
            h_w13_bits, h_w2_bits, num_layers_to_load, path.display(),
        );
        let load_start = std::time::Instant::now();

        let h = config.hidden_size;
        let m = config.moe_intermediate_size;

        // Compute per-expert byte sizes
        let (w13pb, w13sb, w2pb, w2sb) = cpu_expert_byte_sizes_mixed(h, m, group_size, h_w13_bits, h_w2_bits);
        let per_routed_expert = w13pb + w13sb + w2pb + w2sb;
        let per_routed_layer = config.n_routed_experts * per_routed_expert;

        let mut offset = CACHE_HEADER_SIZE + start_moe_layer * per_routed_layer;

        // Load routed experts
        let mut experts_cpu = Vec::with_capacity(num_layers_to_load);
        for layer_idx in 0..num_layers_to_load {
            let mut layer_experts = Vec::with_capacity(config.n_routed_experts);
            for _eidx in 0..config.n_routed_experts {
                layer_experts.push(read_unified_expert_cpu_mixed(
                    &mmap, &mut offset, h, m, group_size, h_w13_bits, h_w2_bits,
                ));
            }
            experts_cpu.push(layer_experts);

            if (layer_idx + 1) % 10 == 0 || layer_idx + 1 == num_layers_to_load {
                log::info!(
                    "  GGUF CPU cache loaded: {}/{} layers ({:.1} GB)",
                    layer_idx + 1, num_layers_to_load, offset as f64 / 1e9,
                );
            }
        }

        // Load shared experts
        let mut shared_experts_cpu = Vec::new();
        if config.n_shared_experts > 0 {
            let routed_total = total_moe_layers * per_routed_layer;
            let shared_m = config.n_shared_experts * config.moe_intermediate_size;
            let (s13p, s13s, s2p, s2s) = cpu_expert_byte_sizes_mixed(h, shared_m, group_size, h_w13_bits, h_w2_bits);
            let per_shared = s13p + s13s + s2p + s2s;

            let shared_base = CACHE_HEADER_SIZE + routed_total + start_moe_layer * per_shared;
            offset = shared_base;

            for _i in 0..num_layers_to_load {
                shared_experts_cpu.push(
                    read_unified_expert_cpu_mixed(&mmap, &mut offset, h, shared_m, group_size, h_w13_bits, h_w2_bits),
                );
            }
            log::info!("  Loaded {} shared experts (GGUF→AVX2)", num_layers_to_load);
        }

        // Evict page cache — data is now copied into heap Vecs
        let _ = unsafe { mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) };
        drop(mmap);
        drop(file);

        let elapsed = load_start.elapsed();
        log::info!(
            "GGUF→AVX2 CPU cache loaded in {:.1}s: {} layers × {} experts (+ {} shared), {:.1} GB",
            elapsed.as_secs_f64(),
            num_layers_to_load, config.n_routed_experts,
            shared_experts_cpu.len(),
            offset as f64 / 1e9,
        );

        Ok((experts_cpu, shared_experts_cpu, h_w13_bits, h_w2_bits))
    }

    /// Convert FP32 gate/up/down expert data to our CPU-optimized UnifiedExpertWeights.
    ///
    /// FP32 → BF16 → quantize to INT4/INT8 → transpose to CPU format.
    /// Supports per-projection precision: `w13_bits` for gate/up, `w2_bits` for down.
    fn gguf_expert_from_f32(
        gate_f32: &[f32],
        up_f32: &[f32],
        down_f32: &[f32],
        intermediate_size: usize,
        hidden_size: usize,
        group_size: usize,
        w13_bits: u8,
        w2_bits: u8,
    ) -> UnifiedExpertWeights {
        use crate::weights::marlin::{f32_to_bf16, quantize_int4, quantize_int8};

        let m = intermediate_size;
        let h = hidden_size;

        // Convert FP32 → BF16
        let gate_bf16: Vec<u16> = gate_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let up_bf16: Vec<u16> = up_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let down_bf16: Vec<u16> = down_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        // Quantize gate/up at w13_bits precision
        let gate_q = if w13_bits == 4 {
            QuantWeight::Int4(quantize_int4(&gate_bf16, m, h, group_size))
        } else {
            QuantWeight::Int8(quantize_int8(&gate_bf16, m, h, group_size))
        };
        let up_q = if w13_bits == 4 {
            QuantWeight::Int4(quantize_int4(&up_bf16, m, h, group_size))
        } else {
            QuantWeight::Int8(quantize_int8(&up_bf16, m, h, group_size))
        };
        // Quantize down at w2_bits precision
        let down_q = if w2_bits == 4 {
            QuantWeight::Int4(quantize_int4(&down_bf16, h, m, group_size))
        } else {
            QuantWeight::Int8(quantize_int8(&down_bf16, h, m, group_size))
        };

        let ew = ExpertWeights { gate: gate_q, up: up_q, down: down_q };

        // Use mixed-precision constructor if bits differ, otherwise fast path
        if w13_bits == w2_bits {
            if w13_bits == 4 {
                UnifiedExpertWeights::from_expert_weights(&ew)
            } else {
                UnifiedExpertWeights::from_expert_weights_int8(&ew)
            }
        } else {
            UnifiedExpertWeights::from_expert_weights_mixed(&ew, w13_bits, w2_bits)
        }
    }
}

/// Write v3 Marlin cache header (same layout as v2, version=3).
fn write_marlin_cache_header<W: Write>(
    w: &mut W,
    config: &ModelConfig,
    group_size: usize,
    num_moe_layers: usize,
    config_hash: u64,
) -> Result<(), String> {
    w.write_all(CACHE_MAGIC)
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&CACHE_VERSION_MARLIN.to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(config.hidden_size as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(config.moe_intermediate_size as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(config.n_routed_experts as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(num_moe_layers as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(group_size as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&config_hash.to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(config.n_shared_experts as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    Ok(())
}

/// Write v4 CPU cache header (same layout, version=4, reserved[56..64] encodes num_bits).
fn write_cpu_cache_header<W: Write>(
    w: &mut W,
    config: &ModelConfig,
    group_size: usize,
    num_moe_layers: usize,
    config_hash: u64,
    num_bits: u8,
) -> Result<(), String> {
    w.write_all(CACHE_MAGIC)
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&CACHE_VERSION_CPU.to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(config.hidden_size as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(config.moe_intermediate_size as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(config.n_routed_experts as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(num_moe_layers as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(group_size as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&config_hash.to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    // Byte 56..64: pack n_shared_experts (low 32) + num_bits (high 32)
    let packed_meta = (config.n_shared_experts as u64) | ((num_bits as u64) << 32);
    w.write_all(&packed_meta.to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    Ok(())
}

/// Write v5 GGUF-sourced CPU cache header.
///
/// Same 64-byte layout, version=5.
/// Byte 56..64 packs: n_shared_experts (low 16) | w13_bits (byte 6) | w2_bits (byte 7).
fn write_cpu_cache_header_v5<W: Write>(
    w: &mut W,
    config: &ModelConfig,
    group_size: usize,
    num_moe_layers: usize,
    config_hash: u64,
    w13_bits: u8,
    w2_bits: u8,
) -> Result<(), String> {
    w.write_all(CACHE_MAGIC)
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&CACHE_VERSION_CPU_GGUF.to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(config.hidden_size as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(config.moe_intermediate_size as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(config.n_routed_experts as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(num_moe_layers as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&(group_size as u64).to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&config_hash.to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    // Byte 56..64: n_shared_experts (low 16) | w13_bits (byte 6) | w2_bits (byte 7) | reserved (byte 7)
    let packed_meta = (config.n_shared_experts as u64)
        | ((w13_bits as u64) << 48)
        | ((w2_bits as u64) << 56);
    w.write_all(&packed_meta.to_le_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    Ok(())
}

/// Write a Vec<u32> as raw bytes to a writer.
fn write_vec_u32<W: Write>(w: &mut W, data: &[u32]) -> Result<(), String> {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    w.write_all(bytes).map_err(|e| format!("Write u32 error: {e}"))
}

/// Write a Vec<u16> as raw bytes to a writer.
fn write_vec_u16<W: Write>(w: &mut W, data: &[u16]) -> Result<(), String> {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
    };
    w.write_all(bytes).map_err(|e| format!("Write u16 error: {e}"))
}

/// Read a UnifiedExpertWeights from mmap'd cache data at the given offset.
fn read_unified_expert(
    data: &[u8],
    offset: &mut usize,
    hidden_size: usize,
    intermediate_size: usize,
    group_size: usize,
) -> UnifiedExpertWeights {
    let h = hidden_size;
    let m = intermediate_size;
    let packed_k = h / 8;
    let num_groups = h / group_size;
    let two_n = 2 * m;

    // w13_packed: [K/8, 2*N] as u32
    let w13_packed_count = packed_k * two_n;
    let mut w13_packed = vec![0u32; w13_packed_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(*offset),
            w13_packed.as_mut_ptr() as *mut u8,
            w13_packed_count * 4,
        );
    }
    *offset += w13_packed_count * 4;

    // w13_scales: [K/gs, 2*N] as u16
    let w13_scales_count = num_groups * two_n;
    let mut w13_scales = vec![0u16; w13_scales_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(*offset),
            w13_scales.as_mut_ptr() as *mut u8,
            w13_scales_count * 2,
        );
    }
    *offset += w13_scales_count * 2;

    // w2_packed: [K_down/8, N_down] = [m/8, h] as u32
    let down_packed_k = m / 8;
    let w2_packed_count = down_packed_k * h;
    let mut w2_packed = vec![0u32; w2_packed_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(*offset),
            w2_packed.as_mut_ptr() as *mut u8,
            w2_packed_count * 4,
        );
    }
    *offset += w2_packed_count * 4;

    // w2_scales: [K_down/gs, N_down] = [m/gs, h] as u16
    let down_num_groups = m / group_size;
    let w2_scales_count = down_num_groups * h;
    let mut w2_scales = vec![0u16; w2_scales_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(*offset),
            w2_scales.as_mut_ptr() as *mut u8,
            w2_scales_count * 2,
        );
    }
    *offset += w2_scales_count * 2;

    UnifiedExpertWeights {
        w13_packed,
        w13_scales,
        w2_packed,
        w2_scales,
        hidden_size,
        intermediate_size,
        group_size,
        num_bits: 4, // Marlin cache is always INT4
        w2_bits: 4,
    }
}

/// Read a UnifiedExpertWeights from mmap'd CPU cache data at the given offset.
/// Supports both INT4 and INT8 transposed formats.
fn read_unified_expert_cpu(
    data: &[u8],
    offset: &mut usize,
    hidden_size: usize,
    intermediate_size: usize,
    group_size: usize,
    num_bits: u8,
) -> UnifiedExpertWeights {
    let h = hidden_size;
    let m = intermediate_size;
    let two_n = 2 * m;
    let num_groups = h / group_size;

    let (w13_packed_count, w2_packed_count) = if num_bits == 4 {
        // INT4: [K/8, N] as u32
        ((h / 8) * two_n, (m / 8) * h)
    } else {
        // INT8: [K, N] as i8 packed into u32 → ceil(bytes/4) u32s
        (((h * two_n) + 3) / 4, ((m * h) + 3) / 4)
    };

    // w13_packed
    let mut w13_packed = vec![0u32; w13_packed_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(*offset),
            w13_packed.as_mut_ptr() as *mut u8,
            w13_packed_count * 4,
        );
    }
    *offset += w13_packed_count * 4;

    // w13_scales: [K/gs, 2*N] as u16
    let w13_scales_count = num_groups * two_n;
    let mut w13_scales = vec![0u16; w13_scales_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(*offset),
            w13_scales.as_mut_ptr() as *mut u8,
            w13_scales_count * 2,
        );
    }
    *offset += w13_scales_count * 2;

    // w2_packed
    let mut w2_packed = vec![0u32; w2_packed_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(*offset),
            w2_packed.as_mut_ptr() as *mut u8,
            w2_packed_count * 4,
        );
    }
    *offset += w2_packed_count * 4;

    // w2_scales: [K_down/gs, N_down] = [m/gs, h] as u16
    let down_num_groups = m / group_size;
    let w2_scales_count = down_num_groups * h;
    let mut w2_scales = vec![0u16; w2_scales_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(*offset),
            w2_scales.as_mut_ptr() as *mut u8,
            w2_scales_count * 2,
        );
    }
    *offset += w2_scales_count * 2;

    UnifiedExpertWeights {
        w13_packed,
        w13_scales,
        w2_packed,
        w2_scales,
        hidden_size,
        intermediate_size,
        group_size,
        num_bits,
        w2_bits: num_bits,
    }
}

/// Read a UnifiedExpertWeights from mmap'd v5 GGUF cache data with mixed precision.
/// w13_bits may differ from w2_bits (e.g. Q4_K gate/up → INT4, Q6_K down → INT8).
fn read_unified_expert_cpu_mixed(
    data: &[u8],
    offset: &mut usize,
    hidden_size: usize,
    intermediate_size: usize,
    group_size: usize,
    w13_bits: u8,
    w2_bits: u8,
) -> UnifiedExpertWeights {
    let h = hidden_size;
    let m = intermediate_size;
    let two_n = 2 * m;
    let num_groups = h / group_size;

    // w13 packed size depends on w13_bits
    let w13_packed_count = if w13_bits == 4 {
        (h / 8) * two_n
    } else {
        ((h * two_n) + 3) / 4
    };

    let mut w13_packed = vec![0u32; w13_packed_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(*offset),
            w13_packed.as_mut_ptr() as *mut u8,
            w13_packed_count * 4,
        );
    }
    *offset += w13_packed_count * 4;

    let w13_scales_count = num_groups * two_n;
    let mut w13_scales = vec![0u16; w13_scales_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(*offset),
            w13_scales.as_mut_ptr() as *mut u8,
            w13_scales_count * 2,
        );
    }
    *offset += w13_scales_count * 2;

    // w2 packed size depends on w2_bits
    let down_num_groups = m / group_size;
    let w2_packed_count = if w2_bits == 4 {
        (m / 8) * h
    } else {
        ((m * h) + 3) / 4
    };

    let mut w2_packed = vec![0u32; w2_packed_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(*offset),
            w2_packed.as_mut_ptr() as *mut u8,
            w2_packed_count * 4,
        );
    }
    *offset += w2_packed_count * 4;

    let w2_scales_count = down_num_groups * h;
    let mut w2_scales = vec![0u16; w2_scales_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(*offset),
            w2_scales.as_mut_ptr() as *mut u8,
            w2_scales_count * 2,
        );
    }
    *offset += w2_scales_count * 2;

    UnifiedExpertWeights {
        w13_packed,
        w13_scales,
        w2_packed,
        w2_scales,
        hidden_size,
        intermediate_size,
        group_size,
        num_bits: w13_bits,
        w2_bits,
    }
}

/// Write a QuantWeight's data + scales to a writer (legacy v1 format).
#[allow(dead_code)]
fn write_quantized<W: Write>(w: &mut W, q: &QuantWeight) -> Result<(), String> {
    match q {
        QuantWeight::Int4(q4) => {
            let packed_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    q4.packed.as_ptr() as *const u8,
                    q4.packed.len() * 4,
                )
            };
            w.write_all(packed_bytes)
                .map_err(|e| format!("Write packed error: {e}"))?;
            let scales_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    q4.scales.as_ptr() as *const u8,
                    q4.scales.len() * 2,
                )
            };
            w.write_all(scales_bytes)
                .map_err(|e| format!("Write scales error: {e}"))?;
        }
        QuantWeight::Int8(q8) => {
            let data_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    q8.data.as_ptr() as *const u8,
                    q8.data.len(),
                )
            };
            w.write_all(data_bytes)
                .map_err(|e| format!("Write data error: {e}"))?;
            let scales_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    q8.scales.as_ptr() as *const u8,
                    q8.scales.len() * 2,
                )
            };
            w.write_all(scales_bytes)
                .map_err(|e| format!("Write scales error: {e}"))?;
        }
    }
    Ok(())
}

/// Read a QuantWeight from mmap'd cache data at the given offset (legacy v1 format).
///
/// Uses direct memcpy — safe on x86_64 (little-endian, unaligned loads OK).
#[allow(dead_code)]
fn read_quantized(
    data: &[u8],
    offset: &mut usize,
    rows: usize,
    cols: usize,
    group_size: usize,
    num_bits: u8,
    data_bytes: usize,
    scales_bytes: usize,
) -> QuantWeight {
    let scales_count = scales_bytes / 2;

    if num_bits == 4 {
        let packed_count = data_bytes / 4;
        let mut packed = vec![0u32; packed_count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().add(*offset),
                packed.as_mut_ptr() as *mut u8,
                data_bytes,
            );
        }
        *offset += data_bytes;

        let mut scales = vec![0u16; scales_count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().add(*offset),
                scales.as_mut_ptr() as *mut u8,
                scales_bytes,
            );
        }
        *offset += scales_bytes;

        QuantWeight::Int4(QuantizedInt4 {
            packed,
            scales,
            rows,
            cols,
            group_size,
        })
    } else {
        let mut weight_data = vec![0i8; data_bytes];
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().add(*offset),
                weight_data.as_mut_ptr() as *mut u8,
                data_bytes,
            );
        }
        *offset += data_bytes;

        let mut scales = vec![0u16; scales_count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().add(*offset),
                scales.as_mut_ptr() as *mut u8,
                scales_bytes,
            );
        }
        *offset += scales_bytes;

        QuantWeight::Int8(QuantizedInt8 {
            data: weight_data,
            scales,
            rows,
            cols,
            group_size,
        })
    }
}

/// Extract the layer number from a tensor name like "model.layers.42.mlp.experts.0.gate_proj.weight".
/// Returns None if no ".layers.N." pattern is found.
fn parse_layer_number(tensor_name: &str) -> Option<usize> {
    let parts: Vec<&str> = tensor_name.split('.').collect();
    for i in 0..parts.len().saturating_sub(1) {
        if parts[i] == "layers" {
            return parts[i + 1].parse().ok();
        }
    }
    None
}

/// Detect the number of physical CPU cores (excluding hyperthreads).
fn detect_physical_cores() -> usize {
    // Try reading thread siblings to determine threads-per-core
    if let Ok(siblings) = std::fs::read_to_string(
        "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list"
    ) {
        let threads_per_core = siblings.trim().split(',').count();
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(64);
        let physical = logical / threads_per_core.max(1);
        if physical > 0 {
            return physical;
        }
    }
    // Fallback: assume no HT
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(64)
}

/// Auto-detect the expert weight prefix from the weight map.
/// Returns "model" for Qwen3/V2-Lite or "language_model.model" for Kimi K2.5.
fn detect_expert_prefix(weight_map: &HashMap<String, String>) -> Result<String, String> {
    for key in weight_map.keys() {
        if let Some(pos) = key.find(".layers.") {
            if key.contains(".mlp.experts.") {
                return Ok(key[..pos].to_string());
            }
        }
    }
    Err("Could not detect expert weight prefix from safetensors index".to_string())
}

/// Detect whether the model uses BF16 weights or pre-quantized compressed-tensors INT4.
/// Returns true if pre-quantized (weight_packed tensors found).
fn is_prequantized(weight_map: &HashMap<String, String>) -> bool {
    weight_map.keys().any(|k| k.ends_with(".weight_packed"))
}

/// Detect the native group_size from a pre-quantized model's weight_scale dimensions.
fn detect_prequant_group_size(
    weight_map: &HashMap<String, String>,
    shards: &HashMap<String, MmapSafetensors>,
    layers_prefix: &str,
    first_moe_layer: usize,
) -> Result<usize, String> {
    let scale_name = format!(
        "{layers_prefix}.layers.{first_moe_layer}.mlp.experts.0.gate_proj.weight_scale"
    );
    let shape_name = format!(
        "{layers_prefix}.layers.{first_moe_layer}.mlp.experts.0.gate_proj.weight_shape"
    );

    // Read weight_shape to get original cols
    let shape_shard_name = weight_map.get(&shape_name)
        .ok_or_else(|| format!("Tensor not found: {shape_name}"))?;
    let shape_shard = shards.get(shape_shard_name)
        .ok_or_else(|| format!("Shard not loaded: {shape_shard_name}"))?;
    let shape_data: &[i32] = shape_shard.tensor_as_slice(&shape_name)
        .map_err(|e| format!("Failed to read {shape_name}: {e}"))?;
    let orig_cols = shape_data[1] as usize;

    // Read weight_scale shape to get scale columns
    let scale_shard_name = weight_map.get(&scale_name)
        .ok_or_else(|| format!("Tensor not found: {scale_name}"))?;
    let scale_shard = shards.get(scale_shard_name)
        .ok_or_else(|| format!("Shard not loaded: {scale_shard_name}"))?;
    let scale_info = scale_shard.tensor_info(&scale_name)
        .ok_or_else(|| format!("Tensor not in shard: {scale_name}"))?;
    let scale_cols = scale_info.shape[1];

    let group_size = orig_cols / scale_cols;
    log::info!(
        "Detected pre-quantized INT4: orig_cols={orig_cols}, scale_cols={scale_cols}, group_size={group_size}"
    );
    Ok(group_size)
}

/// Load a pre-quantized INT4 weight directly (compressed-tensors format).
/// Reads weight_packed (I32), weight_scale (BF16), weight_shape (I32[2]).
fn load_prequantized_weight(
    prefix: &str,
    proj_name: &str,
    weight_map: &HashMap<String, String>,
    shards: &HashMap<String, MmapSafetensors>,
    group_size: usize,
) -> Result<QuantizedInt4, String> {
    let packed_name = format!("{prefix}.{proj_name}.weight_packed");
    let scale_name = format!("{prefix}.{proj_name}.weight_scale");
    let shape_name = format!("{prefix}.{proj_name}.weight_shape");

    // Read weight_shape to get [rows, cols]
    let shape_shard_name = weight_map.get(&shape_name)
        .ok_or_else(|| format!("Tensor not found: {shape_name}"))?;
    let shape_shard = shards.get(shape_shard_name)
        .ok_or_else(|| format!("Shard not loaded: {shape_shard_name}"))?;
    let shape_data: &[i32] = shape_shard.tensor_as_slice(&shape_name)
        .map_err(|e| format!("Failed to read {shape_name}: {e}"))?;
    let rows = shape_data[0] as usize;
    let cols = shape_data[1] as usize;

    // Read weight_packed — I32 [rows, cols/8], directly compatible with our u32 packed format
    let packed_shard_name = weight_map.get(&packed_name)
        .ok_or_else(|| format!("Tensor not found: {packed_name}"))?;
    let packed_shard = shards.get(packed_shard_name)
        .ok_or_else(|| format!("Shard not loaded: {packed_shard_name}"))?;
    let packed_data: &[i32] = packed_shard.tensor_as_slice(&packed_name)
        .map_err(|e| format!("Failed to read {packed_name}: {e}"))?;
    // Reinterpret i32 as u32 (same bit pattern)
    let packed: Vec<u32> = packed_data.iter().map(|&v| v as u32).collect();

    // Read weight_scale — BF16 [rows, cols/group_size], directly compatible with our u16 scales
    let scale_shard_name = weight_map.get(&scale_name)
        .ok_or_else(|| format!("Tensor not found: {scale_name}"))?;
    let scale_shard = shards.get(scale_shard_name)
        .ok_or_else(|| format!("Shard not loaded: {scale_shard_name}"))?;
    let scales_data: &[u16] = scale_shard.tensor_as_slice(&scale_name)
        .map_err(|e| format!("Failed to read {scale_name}: {e}"))?;
    let scales: Vec<u16> = scales_data.to_vec();

    // Validate dimensions
    assert_eq!(packed.len(), rows * (cols / 8),
        "Packed size mismatch: expected {}×{}/8={}, got {}",
        rows, cols, rows * (cols / 8), packed.len());
    assert_eq!(scales.len(), rows * (cols / group_size),
        "Scale size mismatch: expected {}×{}/{}={}, got {}",
        rows, cols, group_size, rows * (cols / group_size), scales.len());

    Ok(QuantizedInt4 {
        packed,
        scales,
        rows,
        cols,
        group_size,
    })
}

/// Load a BF16 weight tensor and quantize it to INT4.
fn load_and_quantize_weight(
    prefix: &str,
    proj_name: &str,
    weight_map: &HashMap<String, String>,
    shards: &HashMap<String, MmapSafetensors>,
    group_size: usize,
) -> Result<QuantizedInt4, String> {
    let tensor_name = format!("{prefix}.{proj_name}.weight");
    let shard_name = weight_map.get(&tensor_name)
        .ok_or_else(|| format!("Tensor not found in index: {tensor_name}"))?;
    let shard = shards.get(shard_name)
        .ok_or_else(|| format!("Shard not loaded: {shard_name}"))?;

    let info = shard.tensor_info(&tensor_name)
        .ok_or_else(|| format!("Tensor not in shard: {tensor_name}"))?;

    let rows = info.shape[0];
    let cols = info.shape[1];

    let bf16_data: &[u16] = shard.tensor_as_slice(&tensor_name)
        .map_err(|e| format!("Failed to read {tensor_name}: {e}"))?;

    Ok(quantize_int4(bf16_data, rows, cols, group_size))
}

/// Load a BF16 expert's gate/up/down projections and quantize to INT4 or INT8.
fn load_and_quantize_expert(
    prefix: &str,
    weight_map: &HashMap<String, String>,
    shards: &HashMap<String, MmapSafetensors>,
    group_size: usize,
    num_bits: u8,
) -> Result<(QuantWeight, QuantWeight, QuantWeight), String> {
    if num_bits == 4 {
        let g = QuantWeight::Int4(load_and_quantize_weight(
            prefix, "gate_proj", weight_map, shards, group_size,
        )?);
        let u = QuantWeight::Int4(load_and_quantize_weight(
            prefix, "up_proj", weight_map, shards, group_size,
        )?);
        let d = QuantWeight::Int4(load_and_quantize_weight(
            prefix, "down_proj", weight_map, shards, group_size,
        )?);
        Ok((g, u, d))
    } else {
        // INT8 path: load BF16 and quantize to INT8
        let load_int8 = |proj_name: &str| -> Result<QuantWeight, String> {
            let tensor_name = format!("{prefix}.{proj_name}.weight");
            let shard_name = weight_map.get(&tensor_name)
                .ok_or_else(|| format!("Tensor not found in index: {tensor_name}"))?;
            let shard = shards.get(shard_name)
                .ok_or_else(|| format!("Shard not loaded: {shard_name}"))?;
            let info = shard.tensor_info(&tensor_name)
                .ok_or_else(|| format!("Tensor not in shard: {tensor_name}"))?;
            let rows = info.shape[0];
            let cols = info.shape[1];
            let bf16_data: &[u16] = shard.tensor_as_slice(&tensor_name)
                .map_err(|e| format!("Failed to read {tensor_name}: {e}"))?;
            Ok(QuantWeight::Int8(quantize_int8(bf16_data, rows, cols, group_size)))
        };

        let g = load_int8("gate_proj")?;
        let u = load_int8("up_proj")?;
        let d = load_int8("down_proj")?;
        Ok((g, u, d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_v2_lite() {
        let _ = env_logger::try_init();
        let model_dir = Path::new("/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite");
        if !model_dir.exists() {
            eprintln!("Skipping — V2-Lite not downloaded");
            return;
        }

        let store = WeightStore::load_from_hf(model_dir, DEFAULT_GROUP_SIZE, None, None, 4, 4)
            .expect("Failed to load V2-Lite");

        // V2-Lite: 27 layers, layer 0 dense, layers 1-26 MoE = 26 MoE layers
        assert_eq!(store.num_moe_layers(), 26);
        assert_eq!(store.config.n_routed_experts, 64);
        assert_eq!(store.config.hidden_size, 2048);
        assert_eq!(store.config.moe_intermediate_size, 1408);

        eprintln!(
            "V2-Lite loaded: {} MoE layers × {} experts, unified={}",
            store.num_moe_layers(),
            store.config.n_routed_experts,
            store.has_unified(),
        );

        // Check expert dimensions via unified format
        if store.has_unified() {
            let expert = store.get_expert_unified(0, 0);
            assert_eq!(expert.hidden_size, 2048);
            assert_eq!(expert.intermediate_size, 1408);
            // w13_packed: [K/8, 2*N] = [256, 2816]
            assert_eq!(expert.w13_packed.len(), (2048 / 8) * (2 * 1408));
            // w2_packed: [K_down/8, N_down] = [176, 2048]
            assert_eq!(expert.w2_packed.len(), (1408 / 8) * 2048);

            // Spot-check: non-zero weights
            assert!(
                expert.w13_packed.iter().any(|&v| v != 0),
                "Expert 0 w13_packed all zeros"
            );
            assert!(
                expert.w13_scales.iter().any(|&v| v != 0),
                "Expert 0 w13_scales all zeros"
            );
        } else {
            let expert = store.get_expert(0, 0);
            assert_eq!(expert.gate.rows(), 1408);
            assert_eq!(expert.gate.cols(), 2048);
            assert_eq!(expert.up.rows(), 1408);
            assert_eq!(expert.up.cols(), 2048);
            assert_eq!(expert.down.rows(), 2048);
            assert_eq!(expert.down.cols(), 1408);

            let deq = marlin::dequantize_int4(expert.gate.as_int4());
            let mut sum_sq: f64 = 0.0;
            for &v in &deq {
                sum_sq += (v as f64).powi(2);
            }
            let rms = (sum_sq / deq.len() as f64).sqrt();
            eprintln!("  Expert 0 gate_proj RMS: {rms:.6}");
            assert!(rms > 0.001, "Expert weights look empty");
        }
    }

    #[test]
    fn test_cache_bit_exact() {
        let _ = env_logger::try_init();
        let model_dir = Path::new("/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite");
        if !model_dir.exists() {
            eprintln!("Skipping — V2-Lite not downloaded");
            return;
        }

        // Load (will use v2 unified cache if available, or v1→convert, or quantize)
        let store = WeightStore::load_from_hf(model_dir, DEFAULT_GROUP_SIZE, None, None, 4, 4)
            .expect("Failed to load V2-Lite");

        // Verify Marlin cache file exists (v3 format)
        let mpath = cache_path_marlin(model_dir, store.group_size);
        assert!(mpath.exists(), "Marlin cache file should exist after load");

        let size = std::fs::metadata(&mpath).unwrap().len();
        let shared_intermediate = store.config.n_shared_experts * store.config.moe_intermediate_size;
        let expected = expected_unified_cache_size(
            &store.config, store.group_size, store.num_moe_layers(),
            store.config.n_shared_experts, shared_intermediate,
        );
        assert_eq!(size as usize, expected, "Marlin cache file size mismatch");

        // Store should have unified weights
        assert!(store.has_unified(), "Store should have unified format");

        // Spot-check multiple experts across layers for non-zero data
        for layer in [0, 12, 25] {
            for eidx in [0, 31, 63] {
                let expert = store.get_expert_unified(layer, eidx);
                assert!(
                    expert.w13_packed.iter().any(|&v| v != 0),
                    "Layer {layer} expert {eidx} w13_packed all zeros"
                );
                assert!(
                    expert.w13_scales.iter().any(|&v| v != 0),
                    "Layer {layer} expert {eidx} w13_scales all zeros"
                );
                assert!(
                    expert.w2_packed.iter().any(|&v| v != 0),
                    "Layer {layer} expert {eidx} w2_packed all zeros"
                );
            }
        }

        eprintln!("Unified cache verified: {:.1} GB", size as f64 / 1e9);
    }

    #[test]
    fn test_config_deepseek_v2() {
        let json: serde_json::Value = serde_json::from_str(r#"{
            "hidden_size": 2048,
            "moe_intermediate_size": 1408,
            "n_routed_experts": 64,
            "num_experts_per_tok": 6,
            "num_hidden_layers": 27,
            "first_k_dense_replace": 1,
            "n_shared_experts": 2,
            "routed_scaling_factor": 1.0
        }"#).unwrap();
        let config = ModelConfig::from_json(&json).unwrap();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.moe_intermediate_size, 1408);
        assert_eq!(config.n_routed_experts, 64);
        assert_eq!(config.num_experts_per_tok, 6);
        assert_eq!(config.num_hidden_layers, 27);
        assert_eq!(config.first_k_dense_replace, 1);
        assert_eq!(config.n_shared_experts, 2);
        assert_eq!(config.routed_scaling_factor, 1.0);
    }

    #[test]
    fn test_config_kimi_k25_text_config() {
        let json: serde_json::Value = serde_json::from_str(r#"{
            "model_type": "kimi_k25",
            "text_config": {
                "hidden_size": 7168,
                "moe_intermediate_size": 2048,
                "n_routed_experts": 384,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 61,
                "first_k_dense_replace": 1,
                "n_shared_experts": 1,
                "routed_scaling_factor": 2.827
            },
            "vision_config": {}
        }"#).unwrap();
        let config = ModelConfig::from_json(&json).unwrap();
        assert_eq!(config.hidden_size, 7168);
        assert_eq!(config.moe_intermediate_size, 2048);
        assert_eq!(config.n_routed_experts, 384);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.num_hidden_layers, 61);
        assert_eq!(config.first_k_dense_replace, 1);
        assert_eq!(config.n_shared_experts, 1);
        assert!((config.routed_scaling_factor - 2.827).abs() < 0.001);
    }

    #[test]
    fn test_config_qwen3_moe() {
        let json: serde_json::Value = serde_json::from_str(r#"{
            "hidden_size": 4096,
            "moe_intermediate_size": 1536,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 94,
            "decoder_sparse_step": 1
        }"#).unwrap();
        let config = ModelConfig::from_json(&json).unwrap();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.moe_intermediate_size, 1536);
        assert_eq!(config.n_routed_experts, 128);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.num_hidden_layers, 94);
        // decoder_sparse_step=1 → all layers are MoE → first_k_dense_replace=0
        assert_eq!(config.first_k_dense_replace, 0);
        // No shared experts in Qwen3
        assert_eq!(config.n_shared_experts, 0);
        assert_eq!(config.routed_scaling_factor, 1.0);
    }

    #[test]
    fn test_load_kimi_k25_single_expert() {
        let _ = env_logger::try_init();
        let model_dir = Path::new("/home/main/Documents/Claude/hf-models/Kimi-K2.5");
        if !model_dir.exists() {
            eprintln!("Skipping — Kimi K2.5 not downloaded");
            return;
        }

        // Parse config
        let config_str = std::fs::read_to_string(model_dir.join("config.json")).unwrap();
        let raw_json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
        let config = ModelConfig::from_json(&raw_json).unwrap();
        assert_eq!(config.hidden_size, 7168);
        assert_eq!(config.moe_intermediate_size, 2048);
        assert_eq!(config.n_routed_experts, 384);
        assert_eq!(config.first_k_dense_replace, 1);

        // Parse safetensors index and open needed shard
        let index_str = std::fs::read_to_string(
            model_dir.join("model.safetensors.index.json")
        ).unwrap();
        let index: SafetensorsIndex = serde_json::from_str(&index_str).unwrap();

        // Verify pre-quantized detection
        assert!(is_prequantized(&index.weight_map));

        let layers_prefix = detect_expert_prefix(&index.weight_map).unwrap();
        assert_eq!(layers_prefix, "language_model.model");

        // Open only the shards needed for layer 1, expert 0
        let prefix = format!("{layers_prefix}.layers.1.mlp.experts.0");
        let mut needed_shards: std::collections::HashSet<String> = std::collections::HashSet::new();
        for proj in &["gate_proj", "up_proj", "down_proj"] {
            for suffix in &["weight_packed", "weight_scale", "weight_shape"] {
                let name = format!("{prefix}.{proj}.{suffix}");
                if let Some(shard) = index.weight_map.get(&name) {
                    needed_shards.insert(shard.clone());
                }
            }
        }
        let mut shards: HashMap<String, MmapSafetensors> = HashMap::new();
        for name in &needed_shards {
            let path = model_dir.join(name);
            let st = MmapSafetensors::open(&path).unwrap();
            shards.insert(name.clone(), st);
        }

        // Detect group_size
        let gs = detect_prequant_group_size(&index.weight_map, &shards, &layers_prefix, 1).unwrap();
        assert_eq!(gs, 32, "Kimi K2.5 should have group_size=32");

        // Load one expert's weights
        let gate = load_prequantized_weight(
            &prefix, "gate_proj", &index.weight_map, &shards, gs,
        ).unwrap();
        let up = load_prequantized_weight(
            &prefix, "up_proj", &index.weight_map, &shards, gs,
        ).unwrap();
        let down = load_prequantized_weight(
            &prefix, "down_proj", &index.weight_map, &shards, gs,
        ).unwrap();

        // Verify dimensions: gate/up=[2048, 7168], down=[7168, 2048]
        assert_eq!(gate.rows, 2048);
        assert_eq!(gate.cols, 7168);
        assert_eq!(up.rows, 2048);
        assert_eq!(up.cols, 7168);
        assert_eq!(down.rows, 7168);
        assert_eq!(down.cols, 2048);
        assert_eq!(gate.group_size, 32);

        // Verify packed sizes
        assert_eq!(gate.packed.len(), 2048 * (7168 / 8));
        assert_eq!(gate.scales.len(), 2048 * (7168 / 32));

        // Verify non-zero data
        assert!(gate.packed.iter().any(|&v| v != 0), "gate packed all zeros");
        assert!(gate.scales.iter().any(|&v| v != 0), "gate scales all zeros");

        // Dequantize and check RMS
        let deq = marlin::dequantize_int4(&gate);
        let rms = (deq.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / deq.len() as f64).sqrt();
        eprintln!(
            "Kimi K2.5 layer 1 expert 0 gate_proj: [{}, {}] group_size={}, RMS={rms:.6}",
            gate.rows, gate.cols, gate.group_size
        );
        assert!(rms > 0.001, "Expert weights look empty (RMS={rms})");
        assert!(rms < 10.0, "Expert weights look corrupted (RMS={rms})");
    }
}
