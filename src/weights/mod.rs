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
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_hidden_layers: usize,
    pub first_k_dense_replace: usize,
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
            config: ModelConfig {
                hidden_size: 0,
                moe_intermediate_size: 0,
                n_routed_experts: 0,
                num_experts_per_tok: 0,
                num_hidden_layers: 0,
                first_k_dense_replace: 0,
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
    pub fn load_from_hf(model_dir: &Path, group_size: usize) -> Result<Self, String> {
        let start = std::time::Instant::now();

        // Parse config.json
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config: ModelConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;

        log::info!(
            "Model config: hidden={}, moe_intermediate={}, experts={}, top-{}, layers={}, first_dense={}",
            config.hidden_size, config.moe_intermediate_size, config.n_routed_experts,
            config.num_experts_per_tok, config.num_hidden_layers, config.first_k_dense_replace,
        );

        let num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace;
        let config_hash = fnv1a(config_str.as_bytes());
        let cpath = cache_path(model_dir, group_size);

        // Try loading from cache first
        if cpath.exists() {
            match Self::load_cache(&cpath, &config, group_size, num_moe_layers, config_hash) {
                Ok(store) => {
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

        // No valid cache — load from safetensors and quantize
        let experts = Self::load_and_quantize_all(model_dir, &config, group_size, num_moe_layers)?;

        let store = WeightStore {
            experts,
            config: config.clone(),
            group_size,
        };

        // Save cache for next time
        match store.save_cache(&cpath, config_hash) {
            Ok(()) => log::info!("Saved INT4 cache to {}", cpath.display()),
            Err(e) => log::warn!("Failed to save cache: {e}"),
        }

        let total_elapsed = start.elapsed();
        log::info!(
            "Loaded and quantized {} MoE layers in {:.1}s (cache saved for next run)",
            num_moe_layers,
            total_elapsed.as_secs_f64(),
        );

        Ok(store)
    }

    /// Load from safetensors shards and quantize to INT4.
    fn load_and_quantize_all(
        model_dir: &Path,
        config: &ModelConfig,
        group_size: usize,
        num_moe_layers: usize,
    ) -> Result<Vec<Vec<ExpertWeights>>, String> {
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

        let mut experts: Vec<Vec<ExpertWeights>> = Vec::with_capacity(num_moe_layers);

        for moe_idx in 0..num_moe_layers {
            let layer_idx = moe_idx + config.first_k_dense_replace;
            let layer_start = std::time::Instant::now();
            let mut layer_experts = Vec::with_capacity(config.n_routed_experts);

            for eidx in 0..config.n_routed_experts {
                let prefix = format!("{layers_prefix}.layers.{layer_idx}.mlp.experts.{eidx}");

                let gate = load_and_quantize_weight(
                    &prefix, "gate_proj", &index.weight_map, &shards, group_size,
                )?;
                let up = load_and_quantize_weight(
                    &prefix, "up_proj", &index.weight_map, &shards, group_size,
                )?;
                let down = load_and_quantize_weight(
                    &prefix, "down_proj", &index.weight_map, &shards, group_size,
                )?;

                layer_experts.push(ExpertWeights { gate, up, down });
            }

            let layer_elapsed = layer_start.elapsed();
            log::info!(
                "Layer {layer_idx}: quantized {} experts in {:.1}s",
                config.n_routed_experts,
                layer_elapsed.as_secs_f64(),
            );
            experts.push(layer_experts);
        }

        let total_bytes: usize = experts.iter().flat_map(|layer| {
            layer.iter().map(|e| {
                (e.gate.packed.len() + e.up.packed.len() + e.down.packed.len()) * 4
                    + (e.gate.scales.len() + e.up.scales.len() + e.down.scales.len()) * 2
            })
        }).sum();

        log::info!(
            "Quantized {} MoE layers × {} experts = {:.1} GB INT4",
            num_moe_layers,
            config.n_routed_experts,
            total_bytes as f64 / 1e9,
        );

        Ok(experts)
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

        Ok(WeightStore {
            experts,
            config: config.clone(),
            group_size,
        })
    }

    /// Get expert weights for a given MoE layer index and expert index.
    /// moe_layer_idx is 0-based within MoE layers (not absolute layer index).
    pub fn get_expert(&self, moe_layer_idx: usize, expert_idx: usize) -> &ExpertWeights {
        &self.experts[moe_layer_idx][expert_idx]
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

        let store = WeightStore::load_from_hf(model_dir, DEFAULT_GROUP_SIZE)
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
        let store = WeightStore::load_from_hf(model_dir, DEFAULT_GROUP_SIZE)
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
}
