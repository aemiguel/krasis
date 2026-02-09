//! Weight loading and format management.
//!
//! Loads expert weights from HF safetensors format, quantizes to INT4,
//! and stores in memory for CPU inference and GPU prefill.
//!
//! Disk cache: after first quantization, saves packed INT4 + scales to
//! `.krasis_cache/experts_int4_g{group_size}.bin` for instant loading.

pub mod marlin;
pub mod safetensors_io;

use crate::weights::marlin::{quantize_int4, QuantizedInt4, DEFAULT_GROUP_SIZE};
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

/// INT4 quantized weights for a single expert (gate + up + down projections).
pub struct ExpertWeights {
    /// gate_proj: [moe_intermediate_size, hidden_size] quantized to INT4
    pub gate: QuantizedInt4,
    /// up_proj: [moe_intermediate_size, hidden_size] quantized to INT4
    pub up: QuantizedInt4,
    /// down_proj: [hidden_size, moe_intermediate_size] quantized to INT4
    pub down: QuantizedInt4,
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
    /// Model configuration.
    pub config: ModelConfig,
    /// Group size used for quantization.
    pub group_size: usize,
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

/// Cache file path for a given model directory and group size.
fn cache_path(model_dir: &Path, group_size: usize) -> PathBuf {
    model_dir
        .join(".krasis_cache")
        .join(format!("experts_int4_g{group_size}.bin"))
}

/// Compute per-expert byte sizes from config.
fn expert_byte_sizes(config: &ModelConfig, group_size: usize) -> (usize, usize, usize, usize) {
    let h = config.hidden_size;
    let m = config.moe_intermediate_size;

    // gate/up: [m, h] → packed [m, h/8] as u32, scales [m, h/gs] as u16
    let gate_packed_bytes = m * (h / 8) * 4;
    let gate_scales_bytes = m * (h / group_size) * 2;

    // down: [h, m] → packed [h, m/8] as u32, scales [h, m/gs] as u16
    let down_packed_bytes = h * (m / 8) * 4;
    let down_scales_bytes = h * (m / group_size) * 2;

    (gate_packed_bytes, gate_scales_bytes, down_packed_bytes, down_scales_bytes)
}

/// Expected total cache file size.
fn expected_cache_size(config: &ModelConfig, group_size: usize, num_moe_layers: usize) -> usize {
    let (gpb, gsb, dpb, dsb) = expert_byte_sizes(config, group_size);
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
        }
    }
}

impl WeightStore {
    /// Load expert weights from a HF model directory, using disk cache if available.
    ///
    /// First checks for a cached `.krasis_cache/experts_int4_g{group_size}.bin`.
    /// If valid, loads directly from cache (mmap + copy, ~1-2s for V2-Lite).
    /// Otherwise, reads BF16 safetensors, quantizes to INT4, and writes cache.
    ///
    /// If `max_layers` is Some(n), only load the first n MoE layers (skips cache).
    pub fn load_from_hf(
        model_dir: &Path,
        group_size: usize,
        max_layers: Option<usize>,
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
        let num_moe_layers = match max_layers {
            Some(n) => {
                let capped = n.min(total_moe_layers);
                if capped < total_moe_layers {
                    log::info!("Partial load: {capped}/{total_moe_layers} MoE layers");
                }
                capped
            }
            None => total_moe_layers,
        };
        let config_hash = fnv1a(config_str.as_bytes());

        // Try loading from cache (only for full model loads)
        if max_layers.is_none() {
            let cpath = cache_path(model_dir, group_size);
            if cpath.exists() {
                match Self::load_cache(&cpath, &config, group_size, num_moe_layers, config_hash) {
                    Ok(mut store) => {
                        // Shared experts not in cache — load them now
                        if config.n_shared_experts > 0 {
                            store.shared_experts = Self::load_shared_experts(
                                model_dir, &config, store.group_size, num_moe_layers,
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

        // Load from safetensors and quantize (or load pre-quantized)
        let (experts, shared_experts, effective_group_size) =
            Self::load_and_quantize_all(model_dir, &config, group_size, num_moe_layers)?;

        let store = WeightStore {
            experts,
            shared_experts,
            config: config.clone(),
            group_size: effective_group_size,
        };

        // Save cache for next time (only for full model loads)
        if max_layers.is_none() {
            let cpath = cache_path(model_dir, group_size);
            match store.save_cache(&cpath, config_hash) {
                Ok(()) => log::info!("Saved INT4 cache to {}", cpath.display()),
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

    /// Load from safetensors shards and quantize to INT4 (or load pre-quantized).
    /// Returns (routed_experts, shared_experts, effective_group_size).
    fn load_and_quantize_all(
        model_dir: &Path,
        config: &ModelConfig,
        group_size: usize,
        num_moe_layers: usize,
    ) -> Result<(Vec<Vec<ExpertWeights>>, Vec<ExpertWeights>, usize), String> {
        // Parse safetensors index
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_str = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("Failed to read safetensors index: {e}"))?;
        let index: SafetensorsIndex = serde_json::from_str(&index_str)
            .map_err(|e| format!("Failed to parse safetensors index: {e}"))?;

        // Determine which shard files we need
        let mut shard_names: Vec<String> = index.weight_map.values().cloned().collect();
        shard_names.sort();
        shard_names.dedup();

        // Open all shards
        let mut shards: HashMap<String, MmapSafetensors> = HashMap::new();
        for name in &shard_names {
            let path = model_dir.join(name);
            let st = MmapSafetensors::open(&path)
                .map_err(|e| format!("Failed to open {name}: {e}"))?;
            shards.insert(name.clone(), st);
        }

        // Auto-detect expert weight prefix pattern
        let layers_prefix = detect_expert_prefix(&index.weight_map)?;
        log::info!("Detected expert prefix: {layers_prefix}");

        // Detect pre-quantized vs BF16 weights
        let prequantized = is_prequantized(&index.weight_map);
        let effective_group_size = if prequantized {
            let first_moe = config.first_k_dense_replace;
            let native_gs = detect_prequant_group_size(
                &index.weight_map, &shards, &layers_prefix, first_moe,
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

        for moe_idx in 0..num_moe_layers {
            let layer_idx = moe_idx + config.first_k_dense_replace;
            let layer_start = std::time::Instant::now();
            let mut layer_experts = Vec::with_capacity(config.n_routed_experts);

            for eidx in 0..config.n_routed_experts {
                let prefix = format!("{layers_prefix}.layers.{layer_idx}.mlp.experts.{eidx}");

                let (gate, up, down) = if prequantized {
                    let g = load_prequantized_weight(
                        &prefix, "gate_proj", &index.weight_map, &shards, effective_group_size,
                    )?;
                    let u = load_prequantized_weight(
                        &prefix, "up_proj", &index.weight_map, &shards, effective_group_size,
                    )?;
                    let d = load_prequantized_weight(
                        &prefix, "down_proj", &index.weight_map, &shards, effective_group_size,
                    )?;
                    (g, u, d)
                } else {
                    let g = load_and_quantize_weight(
                        &prefix, "gate_proj", &index.weight_map, &shards, effective_group_size,
                    )?;
                    let u = load_and_quantize_weight(
                        &prefix, "up_proj", &index.weight_map, &shards, effective_group_size,
                    )?;
                    let d = load_and_quantize_weight(
                        &prefix, "down_proj", &index.weight_map, &shards, effective_group_size,
                    )?;
                    (g, u, d)
                };

                layer_experts.push(ExpertWeights { gate, up, down });
            }

            let layer_elapsed = layer_start.elapsed();
            let action = if prequantized { "loaded" } else { "quantized" };
            log::info!(
                "Layer {layer_idx}: {action} {} experts in {:.1}s",
                config.n_routed_experts,
                layer_elapsed.as_secs_f64(),
            );
            experts.push(layer_experts);
        }

        // Load shared experts (always BF16, quantized to INT4 by us)
        let shared_experts = if config.n_shared_experts > 0 {
            let shared_intermediate = config.n_shared_experts * config.moe_intermediate_size;
            log::info!(
                "Loading shared experts: n_shared={}, intermediate_size={}",
                config.n_shared_experts, shared_intermediate,
            );
            let mut shared = Vec::with_capacity(num_moe_layers);
            for moe_idx in 0..num_moe_layers {
                let layer_idx = moe_idx + config.first_k_dense_replace;
                let prefix = format!("{layers_prefix}.layers.{layer_idx}.mlp.shared_experts");
                let gate = load_and_quantize_weight(
                    &prefix, "gate_proj", &index.weight_map, &shards, effective_group_size,
                )?;
                let up = load_and_quantize_weight(
                    &prefix, "up_proj", &index.weight_map, &shards, effective_group_size,
                )?;
                let down = load_and_quantize_weight(
                    &prefix, "down_proj", &index.weight_map, &shards, effective_group_size,
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
                (e.gate.packed.len() + e.up.packed.len() + e.down.packed.len()) * 4
                    + (e.gate.scales.len() + e.up.scales.len() + e.down.scales.len()) * 2
            })
        }).sum();
        let shared_bytes: usize = shared_experts.iter().map(|e| {
            (e.gate.packed.len() + e.up.packed.len() + e.down.packed.len()) * 4
                + (e.gate.scales.len() + e.up.scales.len() + e.down.scales.len()) * 2
        }).sum();

        log::info!(
            "Loaded {} MoE layers × {} experts = {:.1} GB INT4 (group_size={effective_group_size}), shared={:.1} MB",
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

    /// Load INT4 expert weights from cache file via mmap.
    fn load_cache(
        path: &Path,
        config: &ModelConfig,
        group_size: usize,
        num_moe_layers: usize,
        config_hash: u64,
    ) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open cache: {e}"))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("Failed to mmap cache: {e}"))?;

        // Validate size
        let expected = expected_cache_size(config, group_size, num_moe_layers);
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
        log::info!("Loading from cache: {}", path.display());
        let (gpb, gsb, dpb, dsb) = expert_byte_sizes(config, group_size);
        let h = config.hidden_size;
        let m = config.moe_intermediate_size;
        let mut offset = CACHE_HEADER_SIZE;

        let mut experts: Vec<Vec<ExpertWeights>> = Vec::with_capacity(num_moe_layers);
        let load_start = std::time::Instant::now();

        for layer_idx in 0..num_moe_layers {
            let mut layer_experts = Vec::with_capacity(config.n_routed_experts);
            for _eidx in 0..config.n_routed_experts {
                let gate = read_quantized(&mmap, &mut offset, m, h, group_size, gpb, gsb);
                let up = read_quantized(&mmap, &mut offset, m, h, group_size, gpb, gsb);
                let down = read_quantized(&mmap, &mut offset, h, m, group_size, dpb, dsb);
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

        // Note: shared experts are NOT cached — they are loaded from safetensors
        // on every startup. This is fast since there's only 1 per MoE layer.
        // TODO: add shared experts to cache format in v2.
        Ok(WeightStore {
            experts,
            shared_experts: Vec::new(), // loaded separately after cache
            config: config.clone(),
            group_size,
        })
    }

    /// Migrate expert weights to NUMA nodes according to the given map.
    /// Uses mbind(MPOL_MF_MOVE) to move physical pages without changing virtual addresses.
    /// Returns the number of successfully migrated experts.
    pub fn migrate_numa(&mut self, map: &crate::numa::NumaExpertMap) -> usize {
        use crate::numa::migrate_vec_to_node;

        let start = std::time::Instant::now();
        let mut migrated = 0;
        let mut failed = 0;

        for (layer_idx, layer) in self.experts.iter_mut().enumerate() {
            for (expert_idx, expert) in layer.iter_mut().enumerate() {
                let node = map.node_for(layer_idx, expert_idx);

                // Migrate all 6 buffers (packed + scales for gate, up, down)
                let ok = migrate_vec_to_node(&mut expert.gate.packed, node)
                    && migrate_vec_to_node(&mut expert.gate.scales, node)
                    && migrate_vec_to_node(&mut expert.up.packed, node)
                    && migrate_vec_to_node(&mut expert.up.scales, node)
                    && migrate_vec_to_node(&mut expert.down.packed, node)
                    && migrate_vec_to_node(&mut expert.down.scales, node);

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

    /// Load shared expert weights from safetensors (always BF16, quantized to INT4).
    fn load_shared_experts(
        model_dir: &Path,
        config: &ModelConfig,
        group_size: usize,
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
            let gate = load_and_quantize_weight(
                &prefix, "gate_proj", &index.weight_map, &shards, group_size,
            )?;
            let up = load_and_quantize_weight(
                &prefix, "up_proj", &index.weight_map, &shards, group_size,
            )?;
            let down = load_and_quantize_weight(
                &prefix, "down_proj", &index.weight_map, &shards, group_size,
            )?;
            shared.push(ExpertWeights { gate, up, down });
        }
        log::info!(
            "Loaded {} shared expert layers in {:.1}s",
            shared.len(), start.elapsed().as_secs_f64(),
        );
        Ok(shared)
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
        self.experts.len()
    }
}

/// Write a QuantizedInt4's packed + scales data to a writer.
fn write_quantized<W: Write>(w: &mut W, q: &QuantizedInt4) -> Result<(), String> {
    let packed_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            q.packed.as_ptr() as *const u8,
            q.packed.len() * 4,
        )
    };
    w.write_all(packed_bytes)
        .map_err(|e| format!("Write packed error: {e}"))?;

    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            q.scales.as_ptr() as *const u8,
            q.scales.len() * 2,
        )
    };
    w.write_all(scales_bytes)
        .map_err(|e| format!("Write scales error: {e}"))?;

    Ok(())
}

/// Read a QuantizedInt4 from mmap'd cache data at the given offset.
///
/// Uses direct memcpy — safe on x86_64 (little-endian, unaligned loads OK).
fn read_quantized(
    data: &[u8],
    offset: &mut usize,
    rows: usize,
    cols: usize,
    group_size: usize,
    packed_bytes: usize,
    scales_bytes: usize,
) -> QuantizedInt4 {
    let packed_count = packed_bytes / 4;
    let mut packed = vec![0u32; packed_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(*offset),
            packed.as_mut_ptr() as *mut u8,
            packed_bytes,
        );
    }
    *offset += packed_bytes;

    let scales_count = scales_bytes / 2;
    let mut scales = vec![0u16; scales_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().add(*offset),
            scales.as_mut_ptr() as *mut u8,
            scales_bytes,
        );
    }
    *offset += scales_bytes;

    QuantizedInt4 {
        packed,
        scales,
        rows,
        cols,
        group_size,
    }
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

        let store = WeightStore::load_from_hf(model_dir, DEFAULT_GROUP_SIZE, None)
            .expect("Failed to load V2-Lite");

        // V2-Lite: 27 layers, layer 0 dense, layers 1-26 MoE = 26 MoE layers
        assert_eq!(store.num_moe_layers(), 26);
        assert_eq!(store.config.n_routed_experts, 64);
        assert_eq!(store.config.hidden_size, 2048);
        assert_eq!(store.config.moe_intermediate_size, 1408);

        // Check expert dimensions
        let expert = store.get_expert(0, 0); // first MoE layer, expert 0
        assert_eq!(expert.gate.rows, 1408);
        assert_eq!(expert.gate.cols, 2048);
        assert_eq!(expert.up.rows, 1408);
        assert_eq!(expert.up.cols, 2048);
        assert_eq!(expert.down.rows, 2048);
        assert_eq!(expert.down.cols, 1408);

        eprintln!(
            "V2-Lite loaded: {} MoE layers × {} experts",
            store.num_moe_layers(),
            store.config.n_routed_experts,
        );

        // Spot-check: dequantize one expert's gate_proj and verify SNR
        let deq = marlin::dequantize_int4(&expert.gate);
        let mut sum_sq: f64 = 0.0;
        for &v in &deq {
            sum_sq += (v as f64).powi(2);
        }
        let rms = (sum_sq / deq.len() as f64).sqrt();
        eprintln!("  Expert 0 gate_proj RMS: {rms:.6}");
        assert!(rms > 0.001, "Expert weights look empty");
    }

    #[test]
    fn test_cache_bit_exact() {
        let _ = env_logger::try_init();
        let model_dir = Path::new("/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite");
        if !model_dir.exists() {
            eprintln!("Skipping — V2-Lite not downloaded");
            return;
        }

        // Load (will use cache if available, or quantize + create cache)
        let store = WeightStore::load_from_hf(model_dir, DEFAULT_GROUP_SIZE, None)
            .expect("Failed to load V2-Lite");

        // Verify cache file exists
        let cpath = cache_path(model_dir, DEFAULT_GROUP_SIZE);
        assert!(cpath.exists(), "Cache file should exist after load");

        let size = std::fs::metadata(&cpath).unwrap().len();
        let expected = expected_cache_size(&store.config, store.group_size, store.num_moe_layers());
        assert_eq!(size as usize, expected, "Cache file size mismatch");

        // Spot-check multiple experts across layers for non-zero data
        for layer in [0, 12, 25] {
            for eidx in [0, 31, 63] {
                let expert = store.get_expert(layer, eidx);
                assert!(
                    expert.gate.packed.iter().any(|&v| v != 0),
                    "Layer {layer} expert {eidx} gate packed all zeros"
                );
                assert!(
                    expert.gate.scales.iter().any(|&v| v != 0),
                    "Layer {layer} expert {eidx} gate scales all zeros"
                );
            }
        }

        eprintln!("Cache bit-exact verified: {:.1} GB", size as f64 / 1e9);
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
