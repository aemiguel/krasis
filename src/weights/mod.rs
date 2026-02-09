//! Weight loading and format management.
//!
//! Loads expert weights from HF safetensors format, quantizes to INT4,
//! and stores in memory for CPU inference and GPU prefill.

pub mod marlin;
pub mod safetensors_io;

use crate::weights::marlin::{quantize_int4, QuantizedInt4, DEFAULT_GROUP_SIZE};
use crate::weights::safetensors_io::MmapSafetensors;
use pyo3::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

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
    /// Load all expert weights from a HF model directory.
    ///
    /// Reads config.json for model dimensions, opens all safetensors shards,
    /// loads each expert's BF16 weights and quantizes to INT4.
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

        // Load MoE layers
        let num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace;
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
                "Layer {layer_idx}: loaded {} experts in {:.1}s",
                config.n_routed_experts,
                layer_elapsed.as_secs_f64(),
            );
            experts.push(layer_experts);
        }

        let total_elapsed = start.elapsed();
        let total_bytes: usize = experts.iter().flat_map(|layer| {
            layer.iter().map(|e| {
                (e.gate.packed.len() + e.up.packed.len() + e.down.packed.len()) * 4
                    + (e.gate.scales.len() + e.up.scales.len() + e.down.scales.len()) * 2
            })
        }).sum();

        log::info!(
            "Loaded {} MoE layers × {} experts = {:.1} GB INT4 in {:.1}s",
            num_moe_layers,
            config.n_routed_experts,
            total_bytes as f64 / 1e9,
            total_elapsed.as_secs_f64(),
        );

        Ok(WeightStore {
            experts,
            config,
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
}
