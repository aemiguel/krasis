//! MoE forward dispatch — runs expert computation on CPU for decode.
//!
//! For each token during decode:
//!   1. SGLang computes router logits on GPU, selects top-k experts
//!   2. Krasis receives activation + expert indices + weights
//!   3. For each selected expert: gate+up matmul → SiLU → down matmul
//!   4. Weighted sum of expert outputs returned to SGLang

use crate::kernel::avx2::{matmul_int4_avx2, matmul_int4_parallel};
use crate::weights::marlin::f32_to_bf16;
use crate::weights::{ExpertWeights, WeightStore};
use pyo3::prelude::*;

/// Scratch buffers for expert computation (reused across calls).
pub struct ExpertScratch {
    /// SiLU(gate) * up result as BF16 for down_proj input.
    pub hidden_bf16: Vec<u16>,
    /// gate_proj output (f32).
    pub gate_out: Vec<f32>,
    /// up_proj output (f32).
    pub up_out: Vec<f32>,
    /// Single expert output (f32).
    pub expert_out: Vec<f32>,
}

impl ExpertScratch {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        ExpertScratch {
            hidden_bf16: vec![0u16; intermediate_size],
            gate_out: vec![0.0f32; intermediate_size],
            up_out: vec![0.0f32; intermediate_size],
            expert_out: vec![0.0f32; hidden_size],
        }
    }
}

/// Compute a single expert's output: SiLU(x @ gate^T) * (x @ up^T) @ down^T
///
/// Result is written to `scratch.expert_out`.
///
/// # Arguments
/// * `expert` - INT4 quantized gate/up/down projections
/// * `activation` - Input activation vector [hidden_size] as BF16
/// * `scratch` - Reusable intermediate buffers (result in scratch.expert_out)
/// * `parallel` - Use multi-threaded matmul (for large experts)
pub fn expert_forward(
    expert: &ExpertWeights,
    activation: &[u16],
    scratch: &mut ExpertScratch,
    parallel: bool,
) {
    let matmul = if parallel { matmul_int4_parallel } else { matmul_int4_avx2 };

    // gate_out = activation @ gate_proj^T → [intermediate_size]
    matmul(&expert.gate, activation, &mut scratch.gate_out);

    // up_out = activation @ up_proj^T → [intermediate_size]
    matmul(&expert.up, activation, &mut scratch.up_out);

    // hidden = SiLU(gate_out) * up_out → BF16 [intermediate_size]
    for i in 0..scratch.gate_out.len() {
        let x = scratch.gate_out[i];
        let silu = x * fast_sigmoid(x);
        let hidden = silu * scratch.up_out[i];
        scratch.hidden_bf16[i] = f32_to_bf16(hidden);
    }

    // expert_out = hidden @ down_proj^T → [hidden_size]
    matmul(&expert.down, &scratch.hidden_bf16, &mut scratch.expert_out);
}

/// Full MoE forward for a single token on one layer.
///
/// Runs the selected experts and combines their outputs with routing weights.
///
/// # Arguments
/// * `store` - Loaded expert weights
/// * `moe_layer_idx` - MoE layer index (0-based, skipping dense layers)
/// * `activation` - Input activation [hidden_size] as BF16
/// * `expert_indices` - Selected expert indices from router
/// * `expert_weights` - Routing weights (softmax scores) for selected experts
/// * `output` - Output buffer [hidden_size] as f32 (accumulated weighted sum)
/// * `scratch` - Reusable intermediate buffers
/// * `parallel` - Use multi-threaded matmul
pub fn moe_forward(
    store: &WeightStore,
    moe_layer_idx: usize,
    activation: &[u16],
    expert_indices: &[usize],
    expert_weights: &[f32],
    output: &mut [f32],
    scratch: &mut ExpertScratch,
    parallel: bool,
) {
    assert_eq!(expert_indices.len(), expert_weights.len());
    assert_eq!(activation.len(), store.config.hidden_size);
    assert_eq!(output.len(), store.config.hidden_size);

    // Zero output before accumulation
    output.fill(0.0);

    for (&eidx, &weight) in expert_indices.iter().zip(expert_weights.iter()) {
        let expert = store.get_expert(moe_layer_idx, eidx);
        expert_forward(expert, activation, scratch, parallel);

        // Accumulate: output += weight * expert_out
        for i in 0..output.len() {
            output[i] += weight * scratch.expert_out[i];
        }
    }
}

/// Fast sigmoid approximation: 1 / (1 + exp(-x))
#[inline]
fn fast_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Main MoE runner that SGLang calls as a FusedMoE replacement.
#[pyclass]
pub struct MoERunner {
    // TODO: weight store reference, thread pool config
}

#[pymethods]
impl MoERunner {
    #[new]
    pub fn new() -> Self {
        MoERunner {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::marlin::{quantize_int4, DEFAULT_GROUP_SIZE};
    use crate::weights::WeightStore;
    use std::path::Path;

    #[test]
    fn test_expert_forward_synthetic() {
        let hidden = 256;
        let intermediate = 128;
        let group_size = 128;

        // Create synthetic expert weights
        let mut gate_bf16 = vec![0u16; intermediate * hidden];
        let mut up_bf16 = vec![0u16; intermediate * hidden];
        let mut down_bf16 = vec![0u16; hidden * intermediate];

        for i in 0..gate_bf16.len() {
            gate_bf16[i] = f32_to_bf16((i as f32 / gate_bf16.len() as f32 - 0.5) * 0.1);
            up_bf16[i] = f32_to_bf16((i as f32 / up_bf16.len() as f32 - 0.3) * 0.1);
        }
        for i in 0..down_bf16.len() {
            down_bf16[i] = f32_to_bf16((i as f32 / down_bf16.len() as f32 - 0.5) * 0.1);
        }

        let expert = ExpertWeights {
            gate: quantize_int4(&gate_bf16, intermediate, hidden, group_size),
            up: quantize_int4(&up_bf16, intermediate, hidden, group_size),
            down: quantize_int4(&down_bf16, hidden, intermediate, group_size),
        };

        // Synthetic activation
        let mut activation = vec![0u16; hidden];
        for i in 0..hidden {
            activation[i] = f32_to_bf16(((i * 3 + 1) as f32 / hidden as f32 - 0.5) * 0.2);
        }

        let mut scratch = ExpertScratch::new(hidden, intermediate);

        expert_forward(&expert, &activation, &mut scratch, false);

        // Check output is non-zero and reasonable
        let mut sum_sq: f64 = 0.0;
        let mut nonzero = 0;
        for &v in &scratch.expert_out {
            if v != 0.0 { nonzero += 1; }
            sum_sq += (v as f64).powi(2);
        }
        let rms = (sum_sq / scratch.expert_out.len() as f64).sqrt();

        eprintln!("Synthetic expert forward: RMS={rms:.6}, nonzero={nonzero}/{hidden}");
        assert!(nonzero > hidden / 2, "Too many zero outputs");
        assert!(rms > 1e-6, "Output RMS too small");
        assert!(rms < 10.0, "Output RMS suspiciously large");
    }

    #[test]
    fn test_v2_lite_single_expert() {
        let model_dir = Path::new("/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite");
        if !model_dir.exists() {
            eprintln!("Skipping — V2-Lite not downloaded");
            return;
        }

        // Load just enough to test one expert
        let store = WeightStore::load_from_hf(model_dir, DEFAULT_GROUP_SIZE)
            .expect("Failed to load");

        let hidden = store.config.hidden_size;
        let intermediate = store.config.moe_intermediate_size;

        // Create a reasonable activation vector
        let mut activation = vec![0u16; hidden];
        for i in 0..hidden {
            activation[i] = f32_to_bf16(((i * 7 + 13) as f32 / hidden as f32 - 0.5) * 0.1);
        }

        let mut scratch = ExpertScratch::new(hidden, intermediate);

        // Test single expert forward (serial)
        let start = std::time::Instant::now();
        let expert = store.get_expert(0, 0);
        expert_forward(expert, &activation, &mut scratch, false);
        let serial_us = start.elapsed().as_micros();
        let output_serial: Vec<f32> = scratch.expert_out.clone();

        // Test single expert forward (parallel)
        let start = std::time::Instant::now();
        expert_forward(expert, &activation, &mut scratch, true);
        let parallel_us = start.elapsed().as_micros();
        let output_parallel: Vec<f32> = scratch.expert_out.clone();

        // Results should be very close (FP ordering may differ slightly)
        let mut max_diff: f32 = 0.0;
        for i in 0..hidden {
            max_diff = max_diff.max((output_serial[i] - output_parallel[i]).abs());
        }

        let mut rms: f64 = 0.0;
        for &v in &output_serial {
            rms += (v as f64).powi(2);
        }
        rms = (rms / hidden as f64).sqrt();

        eprintln!("V2-Lite expert 0 forward:");
        eprintln!("  Serial:   {serial_us} μs");
        eprintln!("  Parallel: {parallel_us} μs ({:.1}x speedup)", serial_us as f64 / parallel_us as f64);
        eprintln!("  Output RMS: {rms:.6}, serial vs parallel max_diff: {max_diff:.8}");

        assert!(rms > 1e-4, "Output too small");
    }

    #[test]
    fn test_v2_lite_moe_forward() {
        let model_dir = Path::new("/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite");
        if !model_dir.exists() {
            eprintln!("Skipping — V2-Lite not downloaded");
            return;
        }

        let store = WeightStore::load_from_hf(model_dir, DEFAULT_GROUP_SIZE)
            .expect("Failed to load");

        let hidden = store.config.hidden_size;
        let intermediate = store.config.moe_intermediate_size;
        let top_k = store.config.num_experts_per_tok;

        let mut activation = vec![0u16; hidden];
        for i in 0..hidden {
            activation[i] = f32_to_bf16(((i * 7 + 13) as f32 / hidden as f32 - 0.5) * 0.1);
        }

        let mut scratch = ExpertScratch::new(hidden, intermediate);
        let mut output = vec![0.0f32; hidden];

        // Simulate router output: top-6 experts with softmax weights
        let expert_indices: Vec<usize> = (0..top_k).collect();
        let expert_weights: Vec<f32> = vec![1.0 / top_k as f32; top_k];

        // Time full MoE forward
        let start = std::time::Instant::now();
        moe_forward(
            &store, 0, &activation, &expert_indices, &expert_weights,
            &mut output, &mut scratch, true,
        );
        let moe_us = start.elapsed().as_micros();

        let mut rms: f64 = 0.0;
        for &v in &output {
            rms += (v as f64).powi(2);
        }
        rms = (rms / hidden as f64).sqrt();

        eprintln!("V2-Lite MoE forward (layer 0, top-{top_k}):");
        eprintln!("  Time: {moe_us} μs ({:.1} ms)", moe_us as f64 / 1000.0);
        eprintln!("  Output RMS: {rms:.6}");
        eprintln!(
            "  Per-expert: {:.0} μs ({} matmuls × 3 per expert)",
            moe_us as f64 / top_k as f64,
            top_k,
        );

        assert!(rms > 1e-5, "MoE output too small");
    }
}
