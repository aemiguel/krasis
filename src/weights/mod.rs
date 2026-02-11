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

/// Unified expert weights with combined w13 (gate+up) in transposed layout.
///
/// Layout: [K/8, N] — K (reduction dim) is outer, N (output dim) is contiguous.
/// This enables SIMD across the output dimension (no horizontal sum needed).
///
/// Single copy in RAM shared by CPU decode and GPU prefill.
pub struct UnifiedExpertWeights {
    /// w13 (gate+up concatenated, transposed): packed INT4 in [K/8, 2*N] layout.
    /// K = hidden_size (reduction dim), N = intermediate_size (output dim per gate/up).
    /// First N columns are gate, next N columns are up.
    pub w13_packed: Vec<u32>,
    /// w13 scales: [K/group_size, 2*N] as BF16.
    pub w13_scales: Vec<u16>,

    /// w2 (down, transposed): packed INT4 in [K_down/8, N_down] layout.
    /// K_down = intermediate_size (reduction dim), N_down = hidden_size (output dim).
    pub w2_packed: Vec<u32>,
    /// w2 scales: [K_down/group_size, N_down] as BF16.
    pub w2_scales: Vec<u16>,

    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub group_size: usize,
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
        }
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
    pub experts: Vec<Vec<ExpertWeights>>,
    /// Shared expert weights indexed as [moe_layer_index].
    /// Empty if n_shared_experts == 0.
    /// Shared expert is a single concatenated MLP with
    /// intermediate_size = n_shared_experts × moe_intermediate_size.
    pub shared_experts: Vec<ExpertWeights>,
    /// Unified expert weights (transposed, combined w13 format).
    /// Primary format for CPU decode. Populated after load via convert_to_unified().
    pub experts_unified: Vec<Vec<UnifiedExpertWeights>>,
    /// Unified shared expert weights.
    pub shared_experts_unified: Vec<UnifiedExpertWeights>,
    /// Model configuration.
    pub config: ModelConfig,
    /// Group size used for quantization.
    pub group_size: usize,
    /// Quantization bit width (4 or 8).
    pub num_bits: u8,
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
const CACHE_VERSION: u32 = 1;
const CACHE_VERSION_UNIFIED: u32 = 2;
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
fn cache_path(model_dir: &Path, num_bits: u8, group_size: usize) -> PathBuf {
    model_dir
        .join(".krasis_cache")
        .join(format!("experts_int{num_bits}_g{group_size}.bin"))
}

/// Cache file path for v2 unified format (combined w13, transposed [K/8, N] layout).
fn cache_path_unified(model_dir: &Path, num_bits: u8, group_size: usize) -> PathBuf {
    model_dir
        .join(".krasis_cache")
        .join(format!("experts_unified_int{num_bits}_g{group_size}.bin"))
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

/// Compute per-expert byte sizes from config.
/// Returns (gate_data_bytes, gate_scales_bytes, down_data_bytes, down_scales_bytes).
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

/// Expected total cache file size.
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
            experts_unified: Vec::new(),
            shared_experts_unified: Vec::new(),
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
            num_bits: 4,
        }
    }
}

impl WeightStore {
    /// Load expert weights from a HF model directory, using disk cache if available.
    ///
    /// First checks for a cached `.krasis_cache/experts_int{bits}_g{group_size}.bin`.
    /// If valid, loads directly from cache (mmap + copy, ~1-2s for V2-Lite).
    /// Otherwise, reads BF16 safetensors, quantizes, and writes cache.
    ///
    /// If `max_layers` is Some(n), only load n MoE layers (skips cache).
    /// If `start_layer` is Some(s), start loading from MoE layer s (0-based, skips cache).
    /// Combined: loads MoE layers [start_layer .. start_layer + max_layers).
    /// `num_bits`: 4 for INT4 (default), 8 for INT8.
    pub fn load_from_hf(
        model_dir: &Path,
        group_size: usize,
        max_layers: Option<usize>,
        start_layer: Option<usize>,
        num_bits: u8,
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
            "Model config: hidden={}, moe_intermediate={}, experts={}, top-{}, layers={}, first_dense={}",
            config.hidden_size, config.moe_intermediate_size, config.n_routed_experts,
            config.num_experts_per_tok, config.num_hidden_layers, config.first_k_dense_replace,
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

        // Try loading from cache (only for full model loads — no start_layer or max_layers)
        let partial = max_layers.is_some() || start_layer.is_some();
        if !partial && num_bits == 4 {
            // Try v2 unified cache first (directly loads transposed + combined w13 format)
            let upath = cache_path_unified(model_dir, num_bits, cache_gs);
            if upath.exists() {
                match Self::load_cache_unified(&upath, &config, cache_gs, num_moe_layers, config_hash) {
                    Ok(store) => {
                        let elapsed = start.elapsed();
                        log::info!(
                            "Loaded from unified cache in {:.1}s: {} MoE layers × {} experts (+ {} shared)",
                            elapsed.as_secs_f64(),
                            num_moe_layers,
                            config.n_routed_experts,
                            store.shared_experts_unified.len(),
                        );
                        return Ok(store);
                    }
                    Err(e) => {
                        log::warn!("Unified cache invalid: {e}");
                    }
                }
            }

            // Try v1 cache — stream-convert to v2 on disk, then load v2
            let v1path = cache_path(model_dir, num_bits, cache_gs);
            if v1path.exists() {
                match Self::streaming_v1_to_unified_cache(
                    model_dir, &config, cache_gs, num_moe_layers, &v1path, &upath, config_hash,
                ) {
                    Ok(()) => {
                        match Self::load_cache_unified(&upath, &config, cache_gs, num_moe_layers, config_hash) {
                            Ok(store) => {
                                let elapsed = start.elapsed();
                                log::info!(
                                    "Loaded (v1→v2 streaming) in {:.1}s: {} MoE layers × {} experts",
                                    elapsed.as_secs_f64(),
                                    num_moe_layers,
                                    config.n_routed_experts,
                                );
                                return Ok(store);
                            }
                            Err(e) => {
                                log::warn!("Failed to load freshly-built v2 cache: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("v1→v2 streaming conversion failed: {e}");
                    }
                }
            }
        } else if !partial {
            // INT8: use v1 cache path (unified is INT4-only)
            let cpath = cache_path(model_dir, num_bits, cache_gs);
            if cpath.exists() {
                match Self::load_cache(&cpath, &config, cache_gs, num_bits, num_moe_layers, config_hash) {
                    Ok(mut store) => {
                        if config.n_shared_experts > 0 {
                            store.shared_experts = Self::load_shared_experts(
                                model_dir, &config, store.group_size, num_bits, num_moe_layers,
                            )?;
                        }
                        let elapsed = start.elapsed();
                        log::info!(
                            "Loaded from cache in {:.1}s: {} MoE layers × {} experts",
                            elapsed.as_secs_f64(),
                            num_moe_layers,
                            config.n_routed_experts,
                        );
                        return Ok(store);
                    }
                    Err(e) => {
                        log::warn!("Cache invalid, re-quantizing: {e}");
                    }
                }
            }
        }

        // Full INT4 load (no cache available): stream-build unified cache, then load
        if !partial && num_bits == 4 {
            let upath = cache_path_unified(model_dir, num_bits, cache_gs);
            let effective_group_size = Self::streaming_build_unified_cache(
                model_dir, &config, group_size, num_moe_layers, moe_start, &upath, config_hash,
            )?;
            // The cache was built with effective_group_size (may differ from cache_gs for pre-quantized)
            // If it differs, rename the file to the correct path
            let final_upath = cache_path_unified(model_dir, num_bits, effective_group_size);
            if final_upath != upath {
                std::fs::rename(&upath, &final_upath)
                    .map_err(|e| format!("Failed to rename cache for correct group_size: {e}"))?;
            }
            let store = Self::load_cache_unified(&final_upath, &config, effective_group_size, num_moe_layers, config_hash)?;
            let total_elapsed = start.elapsed();
            log::info!(
                "Loaded {} MoE layers (streaming build + load) in {:.1}s",
                num_moe_layers, total_elapsed.as_secs_f64(),
            );
            return Ok(store);
        }

        // Fallback: partial loads or INT8 — use original load_and_quantize_all path
        log::info!("[DIAG-RUST] Starting load_and_quantize_all: {} MoE layers, start={}, bits={}", num_moe_layers, moe_start, num_bits);
        crate::syscheck::log_memory_usage("[DIAG-RUST] before load_and_quantize_all");
        let (experts, shared_experts, effective_group_size) =
            Self::load_and_quantize_all(model_dir, &config, group_size, num_bits, num_moe_layers, moe_start)?;
        log::info!("[DIAG-RUST] load_and_quantize_all completed OK");
        crate::syscheck::log_memory_usage("[DIAG-RUST] after load_and_quantize_all");

        let store = WeightStore {
            experts,
            shared_experts,
            experts_unified: Vec::new(),
            shared_experts_unified: Vec::new(),
            config: config.clone(),
            group_size: effective_group_size,
            num_bits,
        };

        // Save cache for next time (only for full model loads)
        if !partial {
            // INT8: save v1 cache
            let cpath = cache_path(model_dir, num_bits, effective_group_size);
            match store.save_cache(&cpath, config_hash) {
                Ok(()) => log::info!("Saved INT{num_bits} cache to {}", cpath.display()),
                Err(e) => log::warn!("Failed to save cache: {e}"),
            }
        }

        let total_elapsed = start.elapsed();
        log::info!(
            "Loaded {} MoE layers in {:.1}s",
            num_moe_layers,
            total_elapsed.as_secs_f64(),
        );

        Ok(store)
    }

    /// Load from safetensors shards and quantize to INT4/INT8 (or load pre-quantized).
    /// Returns (routed_experts, shared_experts, effective_group_size).
    ///
    /// `start_moe_layer`: 0-based offset into MoE layers (skips first N MoE layers).
    /// `num_moe_layers`: how many MoE layers to load starting from `start_moe_layer`.
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

    /// Write INT4 expert weights to a cache file.
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

    /// Load expert weights from cache file via mmap.
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

        let elapsed = load_start.elapsed();
        log::info!(
            "Cache loaded: {:.1} GB in {:.1}s ({:.1} GB/s)",
            mmap.len() as f64 / 1e9,
            elapsed.as_secs_f64(),
            mmap.len() as f64 / 1e9 / elapsed.as_secs_f64(),
        );

        Ok(WeightStore {
            experts,
            shared_experts: Vec::new(), // loaded separately after cache
            experts_unified: Vec::new(),
            shared_experts_unified: Vec::new(),
            config: config.clone(),
            group_size,
            num_bits,
        })
    }

    /// Write unified expert weights to a v2 cache file.
    /// Includes both routed and shared experts.
    fn save_cache_unified(&self, path: &Path, config_hash: u64) -> Result<(), String> {
        if self.experts_unified.is_empty() {
            return Err("No unified weights to save".to_string());
        }

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create cache dir: {e}"))?;
        }

        let num_moe_layers = self.experts_unified.len();

        // Write to temp file then rename (atomic)
        let tmp_path = path.with_extension("bin.tmp");
        let file = std::fs::File::create(&tmp_path)
            .map_err(|e| format!("Failed to create cache file: {e}"))?;
        let mut w = std::io::BufWriter::with_capacity(4 * 1024 * 1024, file);

        // Header (64 bytes) — same layout as v1, version=2
        write_unified_cache_header(&mut w, &self.config, self.group_size, num_moe_layers, config_hash)?;

        let write_start = std::time::Instant::now();

        // Write routed experts: for each (layer, expert): w13_packed, w13_scales, w2_packed, w2_scales
        for (layer_idx, layer) in self.experts_unified.iter().enumerate() {
            for expert in layer {
                write_vec_u32(&mut w, &expert.w13_packed)?;
                write_vec_u16(&mut w, &expert.w13_scales)?;
                write_vec_u32(&mut w, &expert.w2_packed)?;
                write_vec_u16(&mut w, &expert.w2_scales)?;
            }
            if (layer_idx + 1) % 10 == 0 {
                log::info!("  Unified cache write: {}/{} layers", layer_idx + 1, num_moe_layers);
            }
        }

        // Write shared experts (one per layer)
        for shared in &self.shared_experts_unified {
            write_vec_u32(&mut w, &shared.w13_packed)?;
            write_vec_u16(&mut w, &shared.w13_scales)?;
            write_vec_u32(&mut w, &shared.w2_packed)?;
            write_vec_u16(&mut w, &shared.w2_scales)?;
        }

        w.flush().map_err(|e| format!("Flush error: {e}"))?;
        drop(w);

        // Atomic rename
        std::fs::rename(&tmp_path, path)
            .map_err(|e| format!("Failed to rename cache file: {e}"))?;

        let elapsed = write_start.elapsed();
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        log::info!(
            "Unified cache written: {:.1} GB in {:.1}s ({:.1} GB/s)",
            size as f64 / 1e9,
            elapsed.as_secs_f64(),
            size as f64 / 1e9 / elapsed.as_secs_f64(),
        );

        Ok(())
    }

    /// Load unified expert weights from v2 cache file via mmap.
    /// Returns a WeightStore with experts_unified populated directly (no conversion needed).
    fn load_cache_unified(
        path: &Path,
        config: &ModelConfig,
        group_size: usize,
        num_moe_layers: usize,
        config_hash: u64,
    ) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open unified cache: {e}"))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("Failed to mmap unified cache: {e}"))?;

        // Validate header
        if mmap.len() < CACHE_HEADER_SIZE {
            return Err("Unified cache too small for header".to_string());
        }
        if &mmap[0..4] != CACHE_MAGIC {
            return Err("Bad magic in unified cache".to_string());
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != CACHE_VERSION_UNIFIED {
            return Err(format!("Cache version {version}, expected {CACHE_VERSION_UNIFIED}"));
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
            || h_num_layers != num_moe_layers
            || h_group_size != group_size
        {
            return Err("Unified cache header dimensions don't match config".to_string());
        }
        if h_config_hash != config_hash {
            return Err("Config hash mismatch in unified cache".to_string());
        }
        if h_n_shared != config.n_shared_experts {
            return Err(format!(
                "Shared expert count mismatch: cache={h_n_shared}, config={}",
                config.n_shared_experts,
            ));
        }

        // Validate total file size
        let shared_intermediate = config.n_shared_experts * config.moe_intermediate_size;
        let expected = expected_unified_cache_size(
            config, group_size, num_moe_layers, config.n_shared_experts, shared_intermediate,
        );
        if mmap.len() != expected {
            return Err(format!(
                "Unified cache size mismatch: expected {} bytes, got {}",
                expected, mmap.len(),
            ));
        }

        log::info!("Loading unified cache: {} (INT4)", path.display());
        let load_start = std::time::Instant::now();
        let mut offset = CACHE_HEADER_SIZE;

        let h = config.hidden_size;
        let m = config.moe_intermediate_size;

        // Read routed experts
        let mut experts_unified: Vec<Vec<UnifiedExpertWeights>> = Vec::with_capacity(num_moe_layers);
        for layer_idx in 0..num_moe_layers {
            let mut layer_experts = Vec::with_capacity(config.n_routed_experts);
            for _eidx in 0..config.n_routed_experts {
                let expert = read_unified_expert(&mmap, &mut offset, h, m, group_size);
                layer_experts.push(expert);
            }
            experts_unified.push(layer_experts);
            if (layer_idx + 1) % 10 == 0 {
                log::info!("  Unified cache read: {}/{} layers", layer_idx + 1, num_moe_layers);
            }
        }

        // Read shared experts
        let mut shared_experts_unified = Vec::with_capacity(config.n_shared_experts.min(1) * num_moe_layers);
        for _layer_idx in 0..num_moe_layers {
            if config.n_shared_experts > 0 {
                let shared = read_unified_expert(&mmap, &mut offset, h, shared_intermediate, group_size);
                shared_experts_unified.push(shared);
            }
        }

        let elapsed = load_start.elapsed();
        log::info!(
            "Unified cache loaded: {:.1} GB in {:.1}s ({:.1} GB/s)",
            mmap.len() as f64 / 1e9,
            elapsed.as_secs_f64(),
            mmap.len() as f64 / 1e9 / elapsed.as_secs_f64(),
        );

        Ok(WeightStore {
            experts: Vec::new(),
            shared_experts: Vec::new(),
            experts_unified,
            shared_experts_unified,
            config: config.clone(),
            group_size,
            num_bits: 4,
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

    /// Load shared expert weights from safetensors (BF16, quantized to INT4/INT8).
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

    /// Stream-convert from safetensors (or pre-quantized) to a v2 unified cache on disk.
    ///
    /// Processes one MoE layer at a time: load expert weights → convert to unified → write
    /// to disk → drop. Peak RAM is ~16 GB (one layer of routed + shared experts) instead
    /// of ~488 GB (all layers).
    ///
    /// Returns the effective group_size (may differ from requested if model is pre-quantized).
    fn streaming_build_unified_cache(
        model_dir: &Path,
        config: &ModelConfig,
        group_size: usize,
        num_moe_layers: usize,
        start_moe_layer: usize,
        cache_path: &Path,
        config_hash: u64,
    ) -> Result<usize, String> {
        log::info!(
            "Streaming build unified cache: {} MoE layers from safetensors → {}",
            num_moe_layers, cache_path.display(),
        );
        crate::syscheck::log_memory_usage("before streaming_build_unified_cache");

        // Parse safetensors index
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_str = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("Failed to read safetensors index: {e}"))?;
        let index: SafetensorsIndex = serde_json::from_str(&index_str)
            .map_err(|e| format!("Failed to parse safetensors index: {e}"))?;

        // Determine which shard files we need for our layer range
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

        // Open shards via mmap (near-zero RSS)
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

        // Write header
        write_unified_cache_header(&mut w, config, effective_group_size, num_moe_layers, config_hash)?;

        let overall_start = std::time::Instant::now();

        // Stream routed experts layer by layer
        for moe_idx in start_moe_layer..(start_moe_layer + num_moe_layers) {
            let layer_idx = moe_idx + config.first_k_dense_replace;
            let layer_start = std::time::Instant::now();

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

                let ew = ExpertWeights { gate, up, down };
                let unified = UnifiedExpertWeights::from_expert_weights(&ew);

                // Write unified data to cache
                write_vec_u32(&mut w, &unified.w13_packed)?;
                write_vec_u16(&mut w, &unified.w13_scales)?;
                write_vec_u32(&mut w, &unified.w2_packed)?;
                write_vec_u16(&mut w, &unified.w2_scales)?;

                // ew + unified dropped here → ~0 persistent RAM per expert
            }

            let layers_done = moe_idx - start_moe_layer + 1;
            let layer_elapsed = layer_start.elapsed();
            if layers_done % 5 == 0 || layers_done == num_moe_layers {
                crate::syscheck::log_memory_usage(&format!(
                    "streaming convert: {layers_done}/{num_moe_layers} layers ({:.1}s/layer)",
                    layer_elapsed.as_secs_f64(),
                ));
            } else {
                log::info!(
                    "  Layer {layer_idx}: converted {} experts in {:.1}s [{layers_done}/{num_moe_layers}]",
                    config.n_routed_experts,
                    layer_elapsed.as_secs_f64(),
                );
            }
        }

        // Stream shared experts (one per MoE layer)
        if config.n_shared_experts > 0 {
            log::info!("Streaming shared experts ({} layers)...", num_moe_layers);
            for moe_idx in start_moe_layer..(start_moe_layer + num_moe_layers) {
                let layer_idx = moe_idx + config.first_k_dense_replace;
                let prefix = format!("{layers_prefix}.layers.{layer_idx}.mlp.shared_experts");
                let (gate, up, down) = load_and_quantize_expert(
                    &prefix, &index.weight_map, &shards, effective_group_size, 4,
                )?;
                let ew = ExpertWeights { gate, up, down };
                let unified = UnifiedExpertWeights::from_expert_weights(&ew);

                write_vec_u32(&mut w, &unified.w13_packed)?;
                write_vec_u16(&mut w, &unified.w13_scales)?;
                write_vec_u32(&mut w, &unified.w2_packed)?;
                write_vec_u16(&mut w, &unified.w2_scales)?;
            }
        }

        // Flush + atomic rename
        w.flush().map_err(|e| format!("Flush error: {e}"))?;
        drop(w);
        std::fs::rename(&tmp_path, cache_path)
            .map_err(|e| format!("Failed to rename cache file: {e}"))?;

        // Return freed pages to OS
        #[cfg(target_os = "linux")]
        unsafe { libc::malloc_trim(0); }

        let elapsed = overall_start.elapsed();
        let size = std::fs::metadata(cache_path).map(|m| m.len()).unwrap_or(0);
        log::info!(
            "Streaming unified cache built: {:.1} GB in {:.1}s ({:.1} GB/s)",
            size as f64 / 1e9,
            elapsed.as_secs_f64(),
            size as f64 / 1e9 / elapsed.as_secs_f64(),
        );
        crate::syscheck::log_memory_usage("after streaming_build_unified_cache");

        Ok(effective_group_size)
    }

    /// Stream-convert from a v1 cache file to a v2 unified cache on disk.
    ///
    /// The v1 file is mmap'd (zero-copy read), and each layer's experts are read,
    /// converted to unified format, written to the v2 temp file, then freed.
    /// Peak RAM is ~16 GB (one layer) instead of loading the entire v1 cache.
    ///
    /// Shared experts are NOT stored in v1 cache, so they must be loaded from safetensors.
    fn streaming_v1_to_unified_cache(
        model_dir: &Path,
        config: &ModelConfig,
        group_size: usize,
        num_moe_layers: usize,
        v1_path: &Path,
        v2_path: &Path,
        config_hash: u64,
    ) -> Result<(), String> {
        log::info!(
            "Streaming v1→v2 conversion: {} → {}",
            v1_path.display(), v2_path.display(),
        );
        crate::syscheck::log_memory_usage("before streaming_v1_to_unified_cache");

        // mmap the v1 cache (zero-copy)
        let v1_file = std::fs::File::open(v1_path)
            .map_err(|e| format!("Failed to open v1 cache: {e}"))?;
        let v1_mmap = unsafe { Mmap::map(&v1_file) }
            .map_err(|e| format!("Failed to mmap v1 cache: {e}"))?;

        // Validate v1 header
        let expected_v1 = expected_cache_size(config, group_size, 4, num_moe_layers);
        if v1_mmap.len() != expected_v1 {
            return Err(format!(
                "v1 cache size mismatch: expected {expected_v1}, got {}", v1_mmap.len(),
            ));
        }
        if &v1_mmap[0..4] != CACHE_MAGIC {
            return Err("Bad magic in v1 cache".to_string());
        }
        let version = u32::from_le_bytes(v1_mmap[4..8].try_into().unwrap());
        if version != CACHE_VERSION {
            return Err(format!("v1 cache version {version}, expected {CACHE_VERSION}"));
        }
        let h_config_hash = u64::from_le_bytes(v1_mmap[48..56].try_into().unwrap());
        if h_config_hash != config_hash {
            return Err("Config hash mismatch in v1 cache".to_string());
        }

        // Create v2 temp file
        if let Some(parent) = v2_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create cache dir: {e}"))?;
        }
        let tmp_path = v2_path.with_extension("bin.tmp");
        let file = std::fs::File::create(&tmp_path)
            .map_err(|e| format!("Failed to create v2 cache file: {e}"))?;
        let mut w = std::io::BufWriter::with_capacity(4 * 1024 * 1024, file);

        // Write v2 header
        write_unified_cache_header(&mut w, config, group_size, num_moe_layers, config_hash)?;

        let overall_start = std::time::Instant::now();
        let h = config.hidden_size;
        let m = config.moe_intermediate_size;
        let (gpb, gsb, dpb, dsb) = expert_byte_sizes(config, group_size, 4);
        let mut v1_offset = CACHE_HEADER_SIZE;

        // Stream layer by layer from v1 mmap
        for layer_idx in 0..num_moe_layers {
            for _eidx in 0..config.n_routed_experts {
                // Read old-format expert from v1 mmap
                let gate = read_quantized(&v1_mmap, &mut v1_offset, m, h, group_size, 4, gpb, gsb);
                let up = read_quantized(&v1_mmap, &mut v1_offset, m, h, group_size, 4, gpb, gsb);
                let down = read_quantized(&v1_mmap, &mut v1_offset, h, m, group_size, 4, dpb, dsb);
                let ew = ExpertWeights { gate, up, down };

                // Convert + write
                let unified = UnifiedExpertWeights::from_expert_weights(&ew);
                write_vec_u32(&mut w, &unified.w13_packed)?;
                write_vec_u16(&mut w, &unified.w13_scales)?;
                write_vec_u32(&mut w, &unified.w2_packed)?;
                write_vec_u16(&mut w, &unified.w2_scales)?;

                // ew + unified dropped here
            }

            let layers_done = layer_idx + 1;
            if layers_done % 5 == 0 || layers_done == num_moe_layers {
                crate::syscheck::log_memory_usage(&format!(
                    "v1→v2 streaming: {layers_done}/{num_moe_layers} layers",
                ));
            }
        }

        // Shared experts: load from safetensors (v1 doesn't store them)
        if config.n_shared_experts > 0 {
            log::info!("Loading shared experts from safetensors for v2 cache...");
            let index_path = model_dir.join("model.safetensors.index.json");
            let index_str = std::fs::read_to_string(&index_path)
                .map_err(|e| format!("Failed to read safetensors index: {e}"))?;
            let index: SafetensorsIndex = serde_json::from_str(&index_str)
                .map_err(|e| format!("Failed to parse safetensors index: {e}"))?;
            let layers_prefix = detect_expert_prefix(&index.weight_map)?;

            // Open only needed shards for shared experts
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

            for moe_idx in 0..num_moe_layers {
                let layer_idx = moe_idx + config.first_k_dense_replace;
                let prefix = format!("{layers_prefix}.layers.{layer_idx}.mlp.shared_experts");
                let (gate, up, down) = load_and_quantize_expert(
                    &prefix, &index.weight_map, &shards, group_size, 4,
                )?;
                let ew = ExpertWeights { gate, up, down };
                let unified = UnifiedExpertWeights::from_expert_weights(&ew);

                write_vec_u32(&mut w, &unified.w13_packed)?;
                write_vec_u16(&mut w, &unified.w13_scales)?;
                write_vec_u32(&mut w, &unified.w2_packed)?;
                write_vec_u16(&mut w, &unified.w2_scales)?;
            }
        }

        // Flush + atomic rename
        w.flush().map_err(|e| format!("Flush error: {e}"))?;
        drop(w);
        std::fs::rename(&tmp_path, v2_path)
            .map_err(|e| format!("Failed to rename v2 cache file: {e}"))?;

        // Drop v1 mmap, delete v1 file
        drop(v1_mmap);
        drop(v1_file);
        if let Err(e) = std::fs::remove_file(v1_path) {
            log::warn!("Failed to delete v1 cache: {e}");
        } else {
            log::info!("Deleted v1 cache: {}", v1_path.display());
        }

        // Return freed pages to OS
        #[cfg(target_os = "linux")]
        unsafe { libc::malloc_trim(0); }

        let elapsed = overall_start.elapsed();
        let size = std::fs::metadata(v2_path).map(|m| m.len()).unwrap_or(0);
        log::info!(
            "v1→v2 streaming conversion done: {:.1} GB in {:.1}s",
            size as f64 / 1e9, elapsed.as_secs_f64(),
        );
        crate::syscheck::log_memory_usage("after streaming_v1_to_unified_cache");

        Ok(())
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
        if !self.experts_unified.is_empty() {
            self.experts_unified.len()
        } else {
            self.experts.len()
        }
    }

    /// Whether unified weights have been populated.
    pub fn has_unified(&self) -> bool {
        !self.experts_unified.is_empty()
    }

    /// Get unified expert weights for a given MoE layer and expert index.
    /// Panics if convert_to_unified() has not been called.
    pub fn get_expert_unified(&self, moe_layer_idx: usize, expert_idx: usize) -> &UnifiedExpertWeights {
        &self.experts_unified[moe_layer_idx][expert_idx]
    }

    /// Get unified shared expert weights for a given MoE layer index.
    /// Returns None if no shared experts or unified conversion not done.
    pub fn get_shared_expert_unified(&self, moe_layer_idx: usize) -> Option<&UnifiedExpertWeights> {
        self.shared_experts_unified.get(moe_layer_idx)
    }

    /// Convert all experts from separate gate/up/down format to unified w13+w2 transposed format.
    ///
    /// After conversion, drops the old `experts` and `shared_experts` to free ~50% RAM
    /// (unified format is same size but replaces old format rather than duplicating).
    ///
    /// This is INT4-only. INT8 experts are not converted (unified kernel only supports INT4).
    pub fn convert_to_unified(&mut self) {
        if self.num_bits != 4 {
            log::warn!("convert_to_unified() only supports INT4, skipping (num_bits={})", self.num_bits);
            return;
        }
        if self.has_unified() {
            log::info!("Already converted to unified format, skipping");
            return;
        }

        let start = std::time::Instant::now();
        let num_layers = self.experts.len();
        let mut old_bytes: usize = 0;
        let mut new_bytes: usize = 0;

        // Convert layer by layer, freeing old format after each layer to avoid
        // holding both copies in RAM simultaneously (~507 GB each for Kimi K2.5).
        // Peak RAM = old format (remaining layers) + new format (converted layers)
        // ≈ old format + one layer overhead during conversion.
        self.experts_unified = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            // Take ownership of this layer's old weights (replaces with empty Vec)
            let old_layer = std::mem::take(&mut self.experts[layer_idx]);

            let mut unified_experts = Vec::with_capacity(old_layer.len());
            for expert in &old_layer {
                old_bytes += expert.gate.data_bytes() + expert.up.data_bytes() + expert.down.data_bytes();
                let unified = UnifiedExpertWeights::from_expert_weights(expert);
                new_bytes += unified.data_bytes();
                unified_experts.push(unified);
            }
            // old_layer dropped here, freeing ~8.3 GB (Kimi K2.5) immediately
            drop(old_layer);
            self.experts_unified.push(unified_experts);

            if (layer_idx + 1) % 10 == 0 || layer_idx + 1 == num_layers {
                log::info!("  Unified conversion: {}/{} layers", layer_idx + 1, num_layers);
            }
        }
        // All old layer data is freed; clear the outer Vec
        self.experts = Vec::new();

        // Convert shared experts (small, no layer-by-layer needed)
        self.shared_experts_unified = Vec::with_capacity(self.shared_experts.len());
        for shared in &self.shared_experts {
            old_bytes += shared.gate.data_bytes() + shared.up.data_bytes() + shared.down.data_bytes();
            let unified = UnifiedExpertWeights::from_expert_weights(shared);
            new_bytes += unified.data_bytes();
            self.shared_experts_unified.push(unified);
        }
        self.shared_experts = Vec::new();

        // Force glibc to return freed pages to OS. Without this, glibc's allocator
        // retains the ~572 GB of freed old-format memory in its arenas, causing RSS
        // to be ~850 GB instead of ~572 GB.
        #[cfg(target_os = "linux")]
        {
            let rss_before = crate::syscheck::get_rss_gib();
            unsafe { libc::malloc_trim(0); }
            let rss_after = crate::syscheck::get_rss_gib();
            log::info!(
                "malloc_trim: RSS {:.1} GiB → {:.1} GiB (reclaimed {:.1} GiB)",
                rss_before, rss_after, rss_before - rss_after
            );
        }

        let elapsed = start.elapsed();
        log::info!(
            "Converted to unified format in {:.1}s: old={:.1} GB → new={:.1} GB (freed {:.1} GB)",
            elapsed.as_secs_f64(),
            old_bytes as f64 / 1e9,
            new_bytes as f64 / 1e9,
            old_bytes as f64 / 1e9,
        );
    }

    /// Migrate unified expert weights to NUMA nodes.
    /// Returns the number of successfully migrated experts.
    pub fn migrate_numa_unified(&mut self, map: &crate::numa::NumaExpertMap) -> usize {
        use crate::numa::migrate_vec_to_node;

        let start = std::time::Instant::now();
        let mut migrated = 0;
        let mut failed = 0;

        for (layer_idx, layer) in self.experts_unified.iter_mut().enumerate() {
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
            "NUMA migration (unified): {migrated} experts migrated, {failed} failed, in {:.1}s",
            elapsed.as_secs_f64(),
        );

        migrated
    }
}

/// Write the 64-byte unified (v2) cache header.
fn write_unified_cache_header<W: Write>(
    w: &mut W,
    config: &ModelConfig,
    group_size: usize,
    num_moe_layers: usize,
    config_hash: u64,
) -> Result<(), String> {
    w.write_all(CACHE_MAGIC)
        .map_err(|e| format!("Write error: {e}"))?;
    w.write_all(&CACHE_VERSION_UNIFIED.to_le_bytes())
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
    }
}

/// Write a QuantWeight's data + scales to a writer.
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

/// Read a QuantWeight from mmap'd cache data at the given offset.
///
/// Uses direct memcpy — safe on x86_64 (little-endian, unaligned loads OK).
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

        let store = WeightStore::load_from_hf(model_dir, DEFAULT_GROUP_SIZE, None, None, 4)
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
        let store = WeightStore::load_from_hf(model_dir, DEFAULT_GROUP_SIZE, None, None, 4)
            .expect("Failed to load V2-Lite");

        // Verify unified cache file exists (v2 format)
        let upath = cache_path_unified(model_dir, 4, DEFAULT_GROUP_SIZE);
        assert!(upath.exists(), "Unified cache file should exist after load");

        let size = std::fs::metadata(&upath).unwrap().len();
        let shared_intermediate = store.config.n_shared_experts * store.config.moe_intermediate_size;
        let expected = expected_unified_cache_size(
            &store.config, store.group_size, store.num_moe_layers(),
            store.config.n_shared_experts, shared_intermediate,
        );
        assert_eq!(size as usize, expected, "Unified cache file size mismatch");

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
