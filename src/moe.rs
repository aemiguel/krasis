//! MoE forward dispatch — runs expert computation on CPU for decode.
//!
//! For each token during decode:
//!   1. SGLang computes router logits on GPU, selects top-k experts
//!   2. Krasis receives activation + expert indices + weights
//!   3. For each selected expert: gate+up matmul → SiLU → down matmul
//!   4. Weighted sum of expert outputs returned to SGLang

use crate::kernel::avx2::{
    matmul_int4_avx2, matmul_int4_parallel,
    matmul_int4_integer, matmul_int4_integer_parallel,
    quantize_activation_int16,
};
use crate::weights::marlin::{f32_to_bf16, DEFAULT_GROUP_SIZE};
use crate::weights::{ExpertWeights, WeightStore};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::Path;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
    /// INT16-quantized input activation for gate/up projections.
    pub input_act_int16: Vec<i16>,
    /// Per-group scales for input activation INT16 quantization.
    pub input_act_scales: Vec<f32>,
    /// INT16-quantized hidden state for down projection.
    pub hidden_int16: Vec<i16>,
    /// Per-group scales for hidden state INT16 quantization.
    pub hidden_scales: Vec<f32>,
    /// Quantization group size.
    pub group_size: usize,
}

impl ExpertScratch {
    pub fn new(hidden_size: usize, intermediate_size: usize, group_size: usize) -> Self {
        ExpertScratch {
            hidden_bf16: vec![0u16; intermediate_size],
            gate_out: vec![0.0f32; intermediate_size],
            up_out: vec![0.0f32; intermediate_size],
            expert_out: vec![0.0f32; hidden_size],
            input_act_int16: vec![0i16; hidden_size],
            input_act_scales: vec![0.0f32; hidden_size / group_size],
            hidden_int16: vec![0i16; intermediate_size],
            hidden_scales: vec![0.0f32; intermediate_size / group_size],
            group_size,
        }
    }
}

/// Compute a single expert's output using the FMA kernel: SiLU(x @ gate^T) * (x @ up^T) @ down^T
///
/// Result is written to `scratch.expert_out`.
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

/// Compute a single expert's output using the integer kernel (_mm256_madd_epi16).
///
/// Uses pre-quantized INT16 activations for gate/up projections (shared across
/// all experts in a layer). The intermediate hidden state is quantized to INT16
/// per-expert before the down projection.
///
/// # Arguments
/// * `expert` - INT4 quantized gate/up/down projections
/// * `act_int16` - Pre-quantized input activation [hidden_size] as INT16
/// * `act_scales` - Per-group scales for the input activation
/// * `scratch` - Reusable intermediate buffers (result in scratch.expert_out)
/// * `parallel` - Use multi-threaded matmul (for large experts)
pub fn expert_forward_integer(
    expert: &ExpertWeights,
    act_int16: &[i16],
    act_scales: &[f32],
    scratch: &mut ExpertScratch,
    parallel: bool,
) {
    let matmul = if parallel { matmul_int4_integer_parallel } else { matmul_int4_integer };

    // gate_out = integer_matmul(gate_proj, act_int16) → f32 [intermediate_size]
    matmul(&expert.gate, act_int16, act_scales, &mut scratch.gate_out);

    // up_out = integer_matmul(up_proj, act_int16) → f32 [intermediate_size]
    matmul(&expert.up, act_int16, act_scales, &mut scratch.up_out);

    // hidden = SiLU(gate_out) * up_out → BF16 [intermediate_size]
    for i in 0..scratch.gate_out.len() {
        let x = scratch.gate_out[i];
        let silu = x * fast_sigmoid(x);
        let hidden = silu * scratch.up_out[i];
        scratch.hidden_bf16[i] = f32_to_bf16(hidden);
    }

    // Quantize intermediate hidden state to INT16 for down projection
    quantize_activation_int16(
        &scratch.hidden_bf16,
        scratch.group_size,
        &mut scratch.hidden_int16,
        &mut scratch.hidden_scales,
    );

    // expert_out = integer_matmul(down_proj, hidden_int16) → f32 [hidden_size]
    matmul(&expert.down, &scratch.hidden_int16, &scratch.hidden_scales, &mut scratch.expert_out);
}

/// Full MoE forward for a single token on one layer.
///
/// Pre-quantizes the BF16 activation to INT16 once, then runs all selected
/// experts using the integer kernel (_mm256_madd_epi16) for ~2x throughput
/// over the FMA kernel. While computing expert[i], prefetches expert[i+1]'s
/// weights into L3 cache using NTA hints.
///
/// # Arguments
/// * `store` - Loaded expert weights
/// * `moe_layer_idx` - MoE layer index (0-based, skipping dense layers)
/// * `activation` - Input activation [hidden_size] as BF16
/// * `expert_indices` - Selected expert indices from router
/// * `expert_weights` - Routing weights (softmax scores) for selected experts
/// * `output` - Output buffer [hidden_size] as f32 (accumulated weighted sum)
/// * `scratch` - Reusable intermediate buffers (includes INT16 activation buffers)
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

    // Pre-quantize input activation to INT16 (shared across all experts).
    // Take buffers out of scratch to split the borrow: act_int16 is read-only
    // during expert_forward while scratch is written to.
    let mut act_int16 = std::mem::take(&mut scratch.input_act_int16);
    let mut act_scales = std::mem::take(&mut scratch.input_act_scales);
    quantize_activation_int16(
        activation,
        scratch.group_size,
        &mut act_int16,
        &mut act_scales,
    );

    // Zero output before accumulation
    output.fill(0.0);

    let n = expert_indices.len();

    for i in 0..n {
        let eidx = expert_indices[i];
        let weight = expert_weights[i];
        let expert = store.get_expert(moe_layer_idx, eidx);

        // Prefetch next expert's weights into L3 while we compute this one
        if i + 1 < n {
            let next_expert = store.get_expert(moe_layer_idx, expert_indices[i + 1]);
            prefetch_expert_nta(next_expert);
        }

        expert_forward_integer(expert, &act_int16, &act_scales, scratch, parallel);

        // Accumulate: output += weight * expert_out
        for j in 0..output.len() {
            output[j] += weight * scratch.expert_out[j];
        }
    }

    // Return buffers to scratch for reuse
    scratch.input_act_int16 = act_int16;
    scratch.input_act_scales = act_scales;
}

/// Prefetch an expert's weight data into L3 cache using NTA (non-temporal) hints.
///
/// NTA brings data to L3 without displacing L1/L2 contents, keeping the
/// activation vector (which is reused across all experts) hot in L1/L2.
/// Prefetches every 512 bytes (8 cache lines) — enough coverage without
/// overwhelming the prefetch buffers.
#[cfg(target_arch = "x86_64")]
fn prefetch_expert_nta(expert: &ExpertWeights) {
    const STRIDE: usize = 512; // 8 cache lines per prefetch

    unsafe {
        // Prefetch gate_proj packed weights
        let gate_bytes = expert.gate.packed.len() * 4;
        let gate_ptr = expert.gate.packed.as_ptr() as *const i8;
        let mut off = 0;
        while off < gate_bytes {
            _mm_prefetch(gate_ptr.add(off), _MM_HINT_NTA);
            off += STRIDE;
        }

        // Prefetch up_proj packed weights
        let up_bytes = expert.up.packed.len() * 4;
        let up_ptr = expert.up.packed.as_ptr() as *const i8;
        off = 0;
        while off < up_bytes {
            _mm_prefetch(up_ptr.add(off), _MM_HINT_NTA);
            off += STRIDE;
        }

        // Prefetch down_proj packed weights
        let down_bytes = expert.down.packed.len() * 4;
        let down_ptr = expert.down.packed.as_ptr() as *const i8;
        off = 0;
        while off < down_bytes {
            _mm_prefetch(down_ptr.add(off), _MM_HINT_NTA);
            off += STRIDE;
        }

        // Prefetch scales (smaller, but still worth it)
        let gate_scales_bytes = expert.gate.scales.len() * 2;
        let gs_ptr = expert.gate.scales.as_ptr() as *const i8;
        off = 0;
        while off < gate_scales_bytes {
            _mm_prefetch(gs_ptr.add(off), _MM_HINT_NTA);
            off += STRIDE;
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn prefetch_expert_nta(_expert: &ExpertWeights) {
    // No-op on non-x86 — hardware prefetcher will handle it
}

/// Fast sigmoid approximation: 1 / (1 + exp(-x))
#[inline]
fn fast_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Main Krasis MoE engine — drop-in replacement for KTransformers CPU expert dispatch.
///
/// Usage from Python:
/// ```python
/// engine = KrasisEngine()
/// engine.load("/path/to/model", group_size=128)
/// output = engine.moe_forward(0, activation_bf16_bytes, [1, 5, 12], [0.3, 0.5, 0.2])
/// ```
#[pyclass]
pub struct KrasisEngine {
    store: Option<WeightStore>,
    scratch: Option<ExpertScratch>,
    output_buf: Vec<f32>,
    parallel: bool,
}

#[pymethods]
impl KrasisEngine {
    #[new]
    #[pyo3(signature = (parallel=true))]
    pub fn new(parallel: bool) -> Self {
        KrasisEngine {
            store: None,
            scratch: None,
            output_buf: Vec::new(),
            parallel,
        }
    }

    /// Load expert weights from a HuggingFace model directory.
    ///
    /// Reads config.json, opens safetensors shards, quantizes BF16 → INT4.
    /// Runs startup system checks (CPU governor, hugepages, memory budget).
    #[pyo3(signature = (model_dir, group_size=None))]
    pub fn load(&mut self, model_dir: &str, group_size: Option<usize>) -> PyResult<()> {
        let gs = group_size.unwrap_or(DEFAULT_GROUP_SIZE);
        let path = Path::new(model_dir);

        // Estimate memory footprint before loading (read config.json first)
        let config_path = path.join("config.json");
        if let Ok(config_str) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) {
                let h = config["hidden_size"].as_u64().unwrap_or(0) as f64;
                let m = config["moe_intermediate_size"].as_u64().unwrap_or(0) as f64;
                let n_exp = config["n_routed_experts"].as_u64().unwrap_or(0) as f64;
                let n_layers = config["num_hidden_layers"].as_u64().unwrap_or(0) as f64;
                let first_dense = config["first_k_dense_replace"].as_u64().unwrap_or(0) as f64;
                let moe_layers = n_layers - first_dense;

                // INT4 packed: (h/8)*m*4 + (h/gs)*m*2 per gate/up, similar for down
                let per_expert_bytes = (m * (h / 8.0) * 4.0 + m * (h / gs as f64) * 2.0) * 2.0
                    + h * (m / 8.0) * 4.0 + h * (m / gs as f64) * 2.0;
                let total_gb = moe_layers * n_exp * per_expert_bytes / 1e9;

                crate::syscheck::run_startup_checks(total_gb);
            }
        }

        let store = WeightStore::load_from_hf(path, gs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        let scratch = ExpertScratch::new(
            store.config.hidden_size,
            store.config.moe_intermediate_size,
            gs,
        );
        self.output_buf = vec![0.0f32; store.config.hidden_size];
        self.store = Some(store);
        self.scratch = Some(scratch);
        Ok(())
    }

    /// Run MoE forward for a single token on one layer.
    ///
    /// Args:
    ///   moe_layer_idx: 0-based MoE layer index (skipping dense layers)
    ///   activation_bf16: BF16 activation as bytes [hidden_size × 2 bytes]
    ///   expert_indices: Selected expert indices from router
    ///   expert_weights: Routing weights (softmax scores) for selected experts
    ///
    /// Returns: f32 output as bytes [hidden_size × 4 bytes]
    pub fn moe_forward<'py>(
        &mut self,
        py: Python<'py>,
        moe_layer_idx: usize,
        activation_bf16: &[u8],
        expert_indices: Vec<usize>,
        expert_weights: Vec<f32>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let store = self.store.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "Model not loaded — call load() first"))?;

        let hidden_size = store.config.hidden_size;

        // Validate activation size
        if activation_bf16.len() != hidden_size * 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Expected {} bytes (hidden_size={} × 2), got {}",
                    hidden_size * 2, hidden_size, activation_bf16.len())
            ));
        }

        // Validate expert args
        if expert_indices.len() != expert_weights.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("expert_indices len ({}) != expert_weights len ({})",
                    expert_indices.len(), expert_weights.len())
            ));
        }

        // Reinterpret BF16 bytes as &[u16]
        let activation: &[u16] = unsafe {
            std::slice::from_raw_parts(
                activation_bf16.as_ptr() as *const u16,
                hidden_size,
            )
        };

        // Split borrows: store (immutable), scratch + output_buf (mutable)
        let scratch = self.scratch.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not loaded"))?;
        let parallel = self.parallel;

        moe_forward(
            store,
            moe_layer_idx,
            activation,
            &expert_indices,
            &expert_weights,
            &mut self.output_buf,
            scratch,
            parallel,
        );

        // Return output as f32 bytes
        let output_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.output_buf.as_ptr() as *const u8,
                self.output_buf.len() * 4,
            )
        };
        Ok(PyBytes::new(py, output_bytes))
    }

    /// Number of MoE layers loaded.
    pub fn num_moe_layers(&self) -> PyResult<usize> {
        let store = self.store.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not loaded"))?;
        Ok(store.num_moe_layers())
    }

    /// Hidden size of the model.
    pub fn hidden_size(&self) -> PyResult<usize> {
        let store = self.store.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not loaded"))?;
        Ok(store.config.hidden_size)
    }

    /// Total number of routed experts per layer.
    pub fn num_experts(&self) -> PyResult<usize> {
        let store = self.store.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not loaded"))?;
        Ok(store.config.n_routed_experts)
    }

    /// Number of experts selected per token (top-k).
    pub fn top_k(&self) -> PyResult<usize> {
        let store = self.store.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not loaded"))?;
        Ok(store.config.num_experts_per_tok)
    }

    /// Whether parallel matmul is enabled.
    pub fn is_parallel(&self) -> bool {
        self.parallel
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

        let mut scratch = ExpertScratch::new(hidden, intermediate, group_size);

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
        let group_size = DEFAULT_GROUP_SIZE;

        // Create a reasonable activation vector
        let mut activation = vec![0u16; hidden];
        for i in 0..hidden {
            activation[i] = f32_to_bf16(((i * 7 + 13) as f32 / hidden as f32 - 0.5) * 0.1);
        }

        let mut scratch = ExpertScratch::new(hidden, intermediate, group_size);

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
        let group_size = DEFAULT_GROUP_SIZE;

        let mut activation = vec![0u16; hidden];
        for i in 0..hidden {
            activation[i] = f32_to_bf16(((i * 7 + 13) as f32 / hidden as f32 - 0.5) * 0.1);
        }

        let mut scratch = ExpertScratch::new(hidden, intermediate, group_size);
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
