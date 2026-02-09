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
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

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
/// over the FMA kernel.
///
/// When a NUMA expert map is provided, experts are grouped by NUMA node and
/// processed in node order (all node-0 experts first, then node-1, etc.).
/// Each group's threads are pinned to the target node for data locality.
///
/// If the model has shared experts, the shared expert forward is computed
/// and the final output is: `routed_scaling_factor * routed_output + shared_output`.
///
/// # Arguments
/// * `store` - Loaded expert weights
/// * `moe_layer_idx` - MoE layer index (0-based, skipping dense layers)
/// * `activation` - Input activation [hidden_size] as BF16
/// * `expert_indices` - Selected expert indices from router
/// * `expert_weights` - Routing weights (softmax scores) for selected experts
/// * `output` - Output buffer [hidden_size] as f32 (accumulated weighted sum)
/// * `scratch` - Reusable intermediate buffers (includes INT16 activation buffers)
/// * `scratch_pool` - Pre-allocated per-expert scratch buffers
/// * `shared_scratch` - Scratch buffer for shared expert (may differ in intermediate size)
/// * `parallel` - Use multi-threaded matmul
/// * `numa_map` - Optional NUMA expert map for node-aware dispatch
pub fn moe_forward(
    store: &WeightStore,
    moe_layer_idx: usize,
    activation: &[u16],
    expert_indices: &[usize],
    expert_weights: &[f32],
    output: &mut [f32],
    scratch: &mut ExpertScratch,
    scratch_pool: &mut [ExpertScratch],
    shared_scratch: &mut Option<ExpertScratch>,
    parallel: bool,
    numa_map: Option<&crate::numa::NumaExpertMap>,
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
    let hidden = store.config.hidden_size;

    if parallel && n > 1 && scratch_pool.len() >= n {
        use rayon::prelude::*;

        if let Some(nmap) = numa_map {
            if nmap.num_nodes > 1 {
                // NUMA-aware dispatch: group experts by node, process each group
                // with threads pinned to that node for data locality.
                let groups = nmap.group_by_node(moe_layer_idx, expert_indices);
                let mut node_order: Vec<usize> = groups.keys().cloned().collect();
                node_order.sort();

                for node in &node_order {
                    let expert_group = &groups[node];
                    // Dispatch this node's experts in parallel, pinned to the node
                    let pool_slices: Vec<(usize, usize)> = expert_group
                        .iter()
                        .map(|&(pos, eidx)| (pos, eidx))
                        .collect();

                    // Use rayon with thread pinning.
                    // SAFETY: each pos indexes a unique scratch buffer, so
                    // concurrent mutable access is safe. We use usize to
                    // avoid Send/Sync issues with raw pointers.
                    let pool_base = scratch_pool.as_mut_ptr() as usize;
                    let target_node = *node;
                    pool_slices.par_iter().for_each(|&(pos, eidx)| {
                        crate::numa::pin_thread_to_node(target_node);
                        let expert = store.get_expert(moe_layer_idx, eidx);
                        let local_scratch = unsafe {
                            &mut *(pool_base as *mut ExpertScratch).add(pos)
                        };
                        expert_forward_integer(
                            expert, &act_int16, &act_scales, local_scratch, true,
                        );
                        crate::numa::unpin_thread();
                    });
                }

                // Weighted sum (sequential)
                for i in 0..n {
                    let weight = expert_weights[i];
                    let expert_out = &scratch_pool[i].expert_out;
                    for j in 0..hidden {
                        output[j] += weight * expert_out[j];
                    }
                }

                // Restore activation buffers and return early
                scratch.input_act_int16 = act_int16;
                scratch.input_act_scales = act_scales;
                return;
            }
        }

        // Non-NUMA parallel path: run all experts concurrently
        let pool = &mut scratch_pool[..n];
        pool.par_iter_mut().enumerate().for_each(|(i, local_scratch)| {
            let eidx = expert_indices[i];
            let expert = store.get_expert(moe_layer_idx, eidx);
            expert_forward_integer(expert, &act_int16, &act_scales, local_scratch, true);
        });

        // Weighted sum (sequential — fast since it's just addition)
        for i in 0..n {
            let weight = expert_weights[i];
            let expert_out = &scratch_pool[i].expert_out;
            for j in 0..hidden {
                output[j] += weight * expert_out[j];
            }
        }
    } else {
        // Sequential: single expert at a time with NTA prefetch
        for i in 0..n {
            let eidx = expert_indices[i];
            let weight = expert_weights[i];
            let expert = store.get_expert(moe_layer_idx, eidx);

            // Prefetch next expert's weights into L3 while we compute this one
            if i + 1 < n {
                let next_expert = store.get_expert(moe_layer_idx, expert_indices[i + 1]);
                prefetch_expert_nta(next_expert);
            }

            expert_forward_integer(expert, &act_int16, &act_scales, scratch, false);

            // Accumulate: output += weight * expert_out
            for j in 0..output.len() {
                output[j] += weight * scratch.expert_out[j];
            }
        }
    }

    // Apply shared expert if present
    if let Some(shared_expert) = store.get_shared_expert(moe_layer_idx) {
        if let Some(ss) = shared_scratch.as_mut() {
            // Reuse the same quantized activation (same hidden_size, same group_size).
            // Use take pattern to split borrows, same as for routed experts.
            let mut ss_act_int16 = std::mem::take(&mut ss.input_act_int16);
            let mut ss_act_scales = std::mem::take(&mut ss.input_act_scales);

            // Copy quantized activation from routed path
            ss_act_int16.clear();
            ss_act_int16.extend_from_slice(&act_int16);
            ss_act_scales.clear();
            ss_act_scales.extend_from_slice(&act_scales);

            expert_forward_integer(shared_expert, &ss_act_int16, &ss_act_scales, ss, parallel);

            // output = routed_scaling_factor * routed_output + shared_output
            let scale = store.config.routed_scaling_factor;
            for j in 0..hidden {
                output[j] = scale * output[j] + ss.expert_out[j];
            }

            // Return buffers
            ss.input_act_int16 = ss_act_int16;
            ss.input_act_scales = ss_act_scales;
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

/// Work unit for the async MoE worker thread.
struct MoeWork {
    moe_layer_idx: usize,
    /// BF16 activations [batch_size, hidden_size] flattened.
    activations: Vec<u16>,
    /// Expert indices [batch_size, topk] flattened. -1 = skip (GPU-handled).
    topk_ids: Vec<i32>,
    /// Expert routing weights [batch_size, topk] flattened.
    topk_weights: Vec<f32>,
    batch_size: usize,
    topk: usize,
}

/// Background worker for async MoE computation.
///
/// Owns its own scratch buffers. Processes batches of tokens sequentially,
/// with expert-level parallelism within each token via rayon.
fn moe_worker(
    store: Arc<WeightStore>,
    work_rx: mpsc::Receiver<MoeWork>,
    result_tx: mpsc::SyncSender<Vec<u16>>,
    parallel: bool,
    numa_map: Option<crate::numa::NumaExpertMap>,
) {
    let hidden = store.config.hidden_size;
    let intermediate = store.config.moe_intermediate_size;
    let group_size = store.group_size;
    let topk_max = store.config.num_experts_per_tok;

    let mut scratch = ExpertScratch::new(hidden, intermediate, group_size);
    let mut scratch_pool: Vec<ExpertScratch> = (0..topk_max)
        .map(|_| ExpertScratch::new(hidden, intermediate, group_size))
        .collect();
    let mut output_f32 = vec![0.0f32; hidden];

    // Shared expert scratch (different intermediate size: n_shared * moe_intermediate)
    let mut shared_scratch = if store.config.n_shared_experts > 0 {
        let shared_intermediate = store.config.n_shared_experts * intermediate;
        Some(ExpertScratch::new(hidden, shared_intermediate, group_size))
    } else {
        None
    };

    while let Ok(work) = work_rx.recv() {
        let batch = work.batch_size;
        let topk = work.topk;
        let mut output_bf16 = vec![0u16; batch * hidden];

        for b in 0..batch {
            let act = &work.activations[b * hidden..(b + 1) * hidden];
            let ids_raw = &work.topk_ids[b * topk..(b + 1) * topk];
            let weights_raw = &work.topk_weights[b * topk..(b + 1) * topk];

            // Filter out masked experts (id == -1 means GPU-handled)
            let mut expert_indices = Vec::with_capacity(topk);
            let mut expert_weights = Vec::with_capacity(topk);
            for j in 0..topk {
                if ids_raw[j] >= 0 {
                    expert_indices.push(ids_raw[j] as usize);
                    expert_weights.push(weights_raw[j]);
                }
            }

            if expert_indices.is_empty() {
                continue; // output_bf16 already zero-initialized
            }

            output_f32.fill(0.0);
            moe_forward(
                &store, work.moe_layer_idx, act,
                &expert_indices, &expert_weights,
                &mut output_f32, &mut scratch, &mut scratch_pool,
                &mut shared_scratch,
                parallel, numa_map.as_ref(),
            );

            // Convert f32 → BF16
            let out_slice = &mut output_bf16[b * hidden..(b + 1) * hidden];
            for j in 0..hidden {
                out_slice[j] = f32_to_bf16(output_f32[j]);
            }
        }

        if result_tx.send(output_bf16).is_err() {
            break; // Engine dropped
        }
    }
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
    store: Option<Arc<WeightStore>>,
    scratch: Option<ExpertScratch>,
    /// Pre-allocated scratch buffers for expert-level parallelism (one per top-k expert).
    scratch_pool: Vec<ExpertScratch>,
    /// Scratch buffer for shared expert (different intermediate size). None if no shared experts.
    shared_scratch: Option<ExpertScratch>,
    output_buf: Vec<f32>,
    parallel: bool,
    /// NUMA expert placement map (None if single-node or NUMA disabled).
    numa_map: Option<crate::numa::NumaExpertMap>,
    /// Async worker: channel to send work.
    work_tx: Option<mpsc::SyncSender<MoeWork>>,
    /// Async worker: channel to receive BF16 results (Mutex for Sync).
    result_rx: Option<Mutex<mpsc::Receiver<Vec<u16>>>>,
    /// Async worker: thread handle (Mutex for Sync).
    worker_handle: Option<Mutex<JoinHandle<()>>>,
}

impl Drop for KrasisEngine {
    fn drop(&mut self) {
        // Drop sender first to signal worker to exit
        self.work_tx.take();
        if let Some(h_mutex) = self.worker_handle.take() {
            if let Ok(h) = h_mutex.into_inner() {
                let _ = h.join();
            }
        }
    }
}

#[pymethods]
impl KrasisEngine {
    #[new]
    #[pyo3(signature = (parallel=true, num_threads=None))]
    pub fn new(parallel: bool, num_threads: Option<usize>) -> Self {
        // Configure rayon thread pool (once, globally)
        if let Some(n) = num_threads {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global();
            log::info!("Rayon thread pool: {n} threads");
        }
        KrasisEngine {
            store: None,
            scratch: None,
            scratch_pool: Vec::new(),
            shared_scratch: None,
            output_buf: Vec::new(),
            parallel,
            numa_map: None,
            work_tx: None,
            result_rx: None,
            worker_handle: None,
        }
    }

    /// Load expert weights from a HuggingFace model directory.
    ///
    /// Reads config.json, opens safetensors shards, quantizes BF16 → INT4.
    /// Runs startup system checks (CPU governor, hugepages, memory budget).
    #[pyo3(signature = (model_dir, group_size=None, max_layers=None))]
    pub fn load(&mut self, model_dir: &str, group_size: Option<usize>, max_layers: Option<usize>) -> PyResult<()> {
        let gs = group_size.unwrap_or(DEFAULT_GROUP_SIZE);
        let path = Path::new(model_dir);

        // Estimate memory footprint before loading (read config.json first)
        let config_path = path.join("config.json");
        if let Ok(config_str) = std::fs::read_to_string(&config_path) {
            if let Ok(raw_json) = serde_json::from_str::<serde_json::Value>(&config_str) {
                if let Ok(config) = crate::weights::ModelConfig::from_json(&raw_json) {
                    let h = config.hidden_size as f64;
                    let m = config.moe_intermediate_size as f64;
                    let n_exp = config.n_routed_experts as f64;
                    let moe_layers = (config.num_hidden_layers - config.first_k_dense_replace) as f64;

                    // INT4 packed: (h/8)*m*4 + (h/gs)*m*2 per gate/up, similar for down
                    let per_expert_bytes = (m * (h / 8.0) * 4.0 + m * (h / gs as f64) * 2.0) * 2.0
                        + h * (m / 8.0) * 4.0 + h * (m / gs as f64) * 2.0;
                    let total_gb = moe_layers * n_exp * per_expert_bytes / 1e9;

                    crate::syscheck::run_startup_checks(total_gb);
                }
            }
        }

        let mut store = WeightStore::load_from_hf(path, gs, max_layers)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // Use the effective group_size from the loaded store (may differ for pre-quantized models)
        let effective_gs = store.group_size;
        let scratch = ExpertScratch::new(
            store.config.hidden_size,
            store.config.moe_intermediate_size,
            effective_gs,
        );
        // Pre-allocate scratch pool for expert-level parallelism
        let top_k = store.config.num_experts_per_tok;
        let scratch_pool: Vec<ExpertScratch> = (0..top_k)
            .map(|_| ExpertScratch::new(
                store.config.hidden_size,
                store.config.moe_intermediate_size,
                effective_gs,
            ))
            .collect();

        // Shared expert scratch (different intermediate size)
        let shared_scratch = if store.config.n_shared_experts > 0 {
            let shared_intermediate = store.config.n_shared_experts * store.config.moe_intermediate_size;
            log::info!(
                "Shared expert scratch: intermediate_size={}, routed_scaling_factor={}",
                shared_intermediate, store.config.routed_scaling_factor,
            );
            Some(ExpertScratch::new(store.config.hidden_size, shared_intermediate, effective_gs))
        } else {
            None
        };

        // Detect NUMA topology and migrate expert weights if multi-node
        let topo = crate::numa::NumaTopology::detect();
        let numa_map = if topo.is_numa() {
            let num_moe_layers = store.num_moe_layers();
            let num_experts = store.config.n_routed_experts;
            let map = crate::numa::NumaExpertMap::round_robin(
                num_moe_layers, num_experts, topo.num_nodes,
            );
            let migrated = store.migrate_numa(&map);
            log::info!(
                "NUMA: {} nodes, migrated {}/{} experts",
                topo.num_nodes, migrated, num_moe_layers * num_experts,
            );
            Some(map)
        } else {
            log::info!("NUMA: single node, no migration needed");
            None
        };

        // Wrap store in Arc for sharing with worker thread
        let store = Arc::new(store);

        // Stop existing worker if any (e.g. on re-load)
        self.work_tx.take();
        if let Some(h_mutex) = self.worker_handle.take() {
            if let Ok(h) = h_mutex.into_inner() {
                let _ = h.join();
            }
        }

        // Spawn async worker thread
        let (work_tx, work_rx) = mpsc::sync_channel(1);
        let (result_tx, result_rx) = mpsc::sync_channel(1);
        let worker_store = store.clone();
        let worker_parallel = self.parallel;
        let worker_numa = numa_map.clone();
        let handle = std::thread::Builder::new()
            .name("krasis-moe-worker".to_string())
            .spawn(move || {
                moe_worker(worker_store, work_rx, result_tx, worker_parallel, worker_numa);
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to spawn worker: {e}")
            ))?;

        log::info!("Async MoE worker thread started");

        self.output_buf = vec![0.0f32; store.config.hidden_size];
        self.scratch_pool = scratch_pool;
        self.shared_scratch = shared_scratch;
        self.work_tx = Some(work_tx);
        self.result_rx = Some(Mutex::new(result_rx));
        self.worker_handle = Some(Mutex::new(handle));
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
            &mut self.scratch_pool,
            &mut self.shared_scratch,
            parallel,
            self.numa_map.as_ref(),
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

    /// Submit asynchronous MoE forward for a batch of tokens.
    ///
    /// Returns immediately — call `sync_forward()` to get results.
    /// Expert indices of -1 are skipped (used for GPU-handled experts).
    ///
    /// Args:
    ///   moe_layer_idx: 0-based MoE layer index
    ///   activation_bf16: BF16 activations as bytes [batch_size × hidden_size × 2]
    ///   topk_ids_i32: Expert indices as i32 bytes [batch_size × topk × 4]
    ///   topk_weights_f32: Routing weights as f32 bytes [batch_size × topk × 4]
    ///   batch_size: Number of tokens in the batch
    #[pyo3(signature = (moe_layer_idx, activation_bf16, topk_ids_i32, topk_weights_f32, batch_size))]
    pub fn submit_forward(
        &self,
        _py: Python<'_>,
        moe_layer_idx: usize,
        activation_bf16: &[u8],
        topk_ids_i32: &[u8],
        topk_weights_f32: &[u8],
        batch_size: usize,
    ) -> PyResult<()> {
        let store = self.store.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "Model not loaded — call load() first"))?;
        let work_tx = self.work_tx.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "Worker not started"))?;

        let hidden = store.config.hidden_size;

        // Validate input sizes
        let expected_act = batch_size * hidden * 2;
        if activation_bf16.len() != expected_act {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Expected {expected_act} activation bytes, got {}", activation_bf16.len())
            ));
        }
        if topk_ids_i32.len() % 4 != 0 || topk_weights_f32.len() % 4 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "topk_ids must be i32 bytes (multiple of 4), topk_weights must be f32 bytes (multiple of 4)"
            ));
        }

        // Reinterpret bytes to typed slices (safe: x86_64 is little-endian)
        let activations: &[u16] = unsafe {
            std::slice::from_raw_parts(
                activation_bf16.as_ptr() as *const u16,
                batch_size * hidden,
            )
        };
        let topk_ids: &[i32] = unsafe {
            std::slice::from_raw_parts(
                topk_ids_i32.as_ptr() as *const i32,
                topk_ids_i32.len() / 4,
            )
        };
        let topk_weights: &[f32] = unsafe {
            std::slice::from_raw_parts(
                topk_weights_f32.as_ptr() as *const f32,
                topk_weights_f32.len() / 4,
            )
        };

        if topk_ids.len() != topk_weights.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("topk_ids ({}) and topk_weights ({}) count mismatch",
                    topk_ids.len(), topk_weights.len())
            ));
        }
        if batch_size == 0 || topk_ids.len() % batch_size != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("topk_ids len ({}) not divisible by batch_size ({batch_size})",
                    topk_ids.len())
            ));
        }

        let topk = topk_ids.len() / batch_size;

        let work = MoeWork {
            moe_layer_idx,
            activations: activations.to_vec(),
            topk_ids: topk_ids.to_vec(),
            topk_weights: topk_weights.to_vec(),
            batch_size,
            topk,
        };

        work_tx.send(work)
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("Worker thread died"))?;

        Ok(())
    }

    /// Wait for async MoE forward to complete and return BF16 output.
    ///
    /// Must be called after `submit_forward()`. Blocks until the worker finishes.
    ///
    /// Returns: BF16 output as bytes [batch_size × hidden_size × 2]
    pub fn sync_forward<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let rx_mutex = self.result_rx.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "Worker not started"))?;

        let rx = rx_mutex.lock()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("Worker mutex poisoned"))?;

        let output_bf16 = rx.recv()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("Worker thread died"))?;

        drop(rx); // release mutex before creating PyBytes

        let output_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                output_bf16.as_ptr() as *const u8,
                output_bf16.len() * 2,
            )
        };

        Ok(PyBytes::new(py, output_bytes))
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
        let store = WeightStore::load_from_hf(model_dir, DEFAULT_GROUP_SIZE, None)
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

        let store = WeightStore::load_from_hf(model_dir, DEFAULT_GROUP_SIZE, None)
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
        let mut scratch_pool: Vec<ExpertScratch> = (0..top_k)
            .map(|_| ExpertScratch::new(hidden, intermediate, group_size))
            .collect();
        let mut output = vec![0.0f32; hidden];

        // Simulate router output: top-6 experts with softmax weights
        let expert_indices: Vec<usize> = (0..top_k).collect();
        let expert_weights: Vec<f32> = vec![1.0 / top_k as f32; top_k];

        // Shared expert scratch for V2-Lite (2 shared experts)
        let shared_intermediate = store.config.n_shared_experts * store.config.moe_intermediate_size;
        let mut shared_scratch = if store.config.n_shared_experts > 0 {
            Some(ExpertScratch::new(hidden, shared_intermediate, group_size))
        } else {
            None
        };

        // Time full MoE forward
        let start = std::time::Instant::now();
        moe_forward(
            &store, 0, &activation, &expert_indices, &expert_weights,
            &mut output, &mut scratch, &mut scratch_pool, &mut shared_scratch, true, None,
        );
        let moe_us = start.elapsed().as_micros();

        let mut rms: f64 = 0.0;
        for &v in &output {
            rms += (v as f64).powi(2);
        }
        rms = (rms / hidden as f64).sqrt();

        eprintln!("V2-Lite MoE forward (layer 0, top-{top_k}, shared={}):", store.config.n_shared_experts);
        eprintln!("  Time: {moe_us} μs ({:.1} ms)", moe_us as f64 / 1000.0);
        eprintln!("  Output RMS: {rms:.6}");
        eprintln!(
            "  Per-expert: {:.0} μs ({} matmuls × 3 per expert)",
            moe_us as f64 / top_k as f64,
            top_k,
        );

        assert!(rms > 1e-5, "MoE output too small");
    }

    #[test]
    fn test_kimi_k25_moe_forward() {
        let model_dir = Path::new("/home/main/Documents/Claude/hf-models/Kimi-K2.5");
        if !model_dir.exists() {
            eprintln!("Skipping — Kimi K2.5 not downloaded");
            return;
        }

        // Load just 1 MoE layer (384 experts × 1 layer ≈ 9.5 GB)
        let store = WeightStore::load_from_hf(model_dir, DEFAULT_GROUP_SIZE, Some(1))
            .expect("Failed to load Kimi K2.5");

        assert_eq!(store.num_moe_layers(), 1);
        assert_eq!(store.config.n_routed_experts, 384);
        assert_eq!(store.config.hidden_size, 7168);
        assert_eq!(store.config.moe_intermediate_size, 2048);
        // Pre-quantized with group_size=32
        assert_eq!(store.group_size, 32);

        let hidden = store.config.hidden_size;
        let intermediate = store.config.moe_intermediate_size;
        let top_k = store.config.num_experts_per_tok; // 8

        // Create activation vector
        let mut activation = vec![0u16; hidden];
        for i in 0..hidden {
            activation[i] = f32_to_bf16(((i * 7 + 13) as f32 / hidden as f32 - 0.5) * 0.1);
        }

        let group_size = store.group_size;
        let mut scratch = ExpertScratch::new(hidden, intermediate, group_size);
        let mut scratch_pool: Vec<ExpertScratch> = (0..top_k)
            .map(|_| ExpertScratch::new(hidden, intermediate, group_size))
            .collect();
        let mut output = vec![0.0f32; hidden];

        // Use top-8 experts (indices 0..7)
        let expert_indices: Vec<usize> = (0..top_k).collect();
        let expert_weights: Vec<f32> = vec![1.0 / top_k as f32; top_k];

        // Shared expert scratch for Kimi K2.5 (1 shared expert)
        // Note: partial load (max_layers=1) may not load shared experts from cache path,
        // so we test with None here. Full test below tests shared.
        let mut shared_scratch: Option<ExpertScratch> = None;

        // Time MoE forward
        let start = std::time::Instant::now();
        moe_forward(
            &store, 0, &activation, &expert_indices, &expert_weights,
            &mut output, &mut scratch, &mut scratch_pool, &mut shared_scratch, true, None,
        );
        let moe_us = start.elapsed().as_micros();

        let rms = (output.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / hidden as f64).sqrt();
        let nonzero = output.iter().filter(|&&v| v.abs() > 1e-10).count();

        eprintln!("Kimi K2.5 MoE forward (1 layer, top-{top_k}, group_size={group_size}):");
        eprintln!("  Time: {moe_us} μs ({:.1} ms)", moe_us as f64 / 1000.0);
        eprintln!("  Output RMS: {rms:.6}, nonzero: {nonzero}/{hidden}");
        eprintln!("  Per-expert: {:.0} μs", moe_us as f64 / top_k as f64);

        assert!(rms > 1e-5, "MoE output too small (RMS={rms})");
        assert!(rms < 100.0, "MoE output too large (RMS={rms})");
        assert!(nonzero > hidden / 2, "Too many zeros ({nonzero}/{hidden})");
    }

    #[test]
    fn test_async_submit_sync() {
        let _ = env_logger::try_init();
        let model_dir = Path::new("/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite");
        if !model_dir.exists() {
            eprintln!("Skipping — V2-Lite not downloaded");
            return;
        }

        let store = WeightStore::load_from_hf(model_dir, DEFAULT_GROUP_SIZE, None)
            .expect("Failed to load");

        let hidden = store.config.hidden_size;
        let top_k = store.config.num_experts_per_tok;
        let group_size = store.group_size;

        // First: compute reference output via synchronous moe_forward
        let mut activation = vec![0u16; hidden];
        for i in 0..hidden {
            activation[i] = f32_to_bf16(((i * 7 + 13) as f32 / hidden as f32 - 0.5) * 0.1);
        }
        let expert_indices: Vec<usize> = (0..top_k).collect();
        let expert_weights: Vec<f32> = vec![1.0 / top_k as f32; top_k];

        let mut scratch = ExpertScratch::new(hidden, store.config.moe_intermediate_size, group_size);
        let mut scratch_pool: Vec<ExpertScratch> = (0..top_k)
            .map(|_| ExpertScratch::new(hidden, store.config.moe_intermediate_size, group_size))
            .collect();
        let shared_intermediate = store.config.n_shared_experts * store.config.moe_intermediate_size;
        let mut shared_scratch = if store.config.n_shared_experts > 0 {
            Some(ExpertScratch::new(hidden, shared_intermediate, group_size))
        } else {
            None
        };
        let mut ref_output = vec![0.0f32; hidden];
        moe_forward(
            &store, 0, &activation, &expert_indices, &expert_weights,
            &mut ref_output, &mut scratch, &mut scratch_pool, &mut shared_scratch, true, None,
        );

        // Convert reference to BF16 for comparison
        let ref_bf16: Vec<u16> = ref_output.iter().map(|&v| f32_to_bf16(v)).collect();

        // Now: test async worker
        let store_arc = Arc::new(store);
        let (work_tx, work_rx) = mpsc::sync_channel(1);
        let (result_tx, result_rx) = mpsc::sync_channel(1);
        let worker_store = store_arc.clone();
        let handle = std::thread::Builder::new()
            .name("test-worker".to_string())
            .spawn(move || {
                moe_worker(worker_store, work_rx, result_tx, true, None);
            })
            .expect("Failed to spawn worker");

        // Submit batch=1 work
        let topk_ids: Vec<i32> = expert_indices.iter().map(|&e| e as i32).collect();
        let topk_weights_f32: Vec<f32> = expert_weights.clone();

        let work = MoeWork {
            moe_layer_idx: 0,
            activations: activation.clone(),
            topk_ids,
            topk_weights: topk_weights_f32,
            batch_size: 1,
            topk: top_k,
        };

        let start = std::time::Instant::now();
        work_tx.send(work).expect("Send failed");
        let async_output = result_rx.recv().expect("Recv failed");
        let async_us = start.elapsed().as_micros();

        assert_eq!(async_output.len(), hidden, "Async output wrong size");

        // Compare async output with reference
        let mut max_diff: f32 = 0.0;
        for i in 0..hidden {
            let a = crate::weights::marlin::bf16_to_f32(async_output[i]);
            let b = crate::weights::marlin::bf16_to_f32(ref_bf16[i]);
            max_diff = max_diff.max((a - b).abs());
        }

        eprintln!("Async submit/sync test (V2-Lite, batch=1, top-{top_k}):");
        eprintln!("  Round-trip: {async_us} μs ({:.1} ms)", async_us as f64 / 1000.0);
        eprintln!("  Max diff vs sync: {max_diff:.8}");
        assert!(max_diff < 0.01, "Async/sync output mismatch: max_diff={max_diff}");

        // Test batch=2 (two identical tokens → should get identical outputs)
        let mut batch_act = vec![0u16; 2 * hidden];
        batch_act[..hidden].copy_from_slice(&activation);
        batch_act[hidden..].copy_from_slice(&activation);
        let batch_ids: Vec<i32> = expert_indices.iter().map(|&e| e as i32)
            .chain(expert_indices.iter().map(|&e| e as i32)).collect();
        let batch_weights: Vec<f32> = expert_weights.iter().chain(expert_weights.iter()).cloned().collect();

        let work2 = MoeWork {
            moe_layer_idx: 0,
            activations: batch_act,
            topk_ids: batch_ids,
            topk_weights: batch_weights,
            batch_size: 2,
            topk: top_k,
        };
        work_tx.send(work2).expect("Send failed");
        let batch_output = result_rx.recv().expect("Recv failed");
        assert_eq!(batch_output.len(), 2 * hidden);

        // Both tokens should produce identical output
        let max_inter_diff: f32 = (0..hidden)
            .map(|i| {
                let a = crate::weights::marlin::bf16_to_f32(batch_output[i]);
                let b = crate::weights::marlin::bf16_to_f32(batch_output[hidden + i]);
                (a - b).abs()
            })
            .fold(0.0f32, f32::max);
        eprintln!("  Batch=2 inter-token diff: {max_inter_diff:.8}");
        assert!(max_inter_diff < 1e-6, "Batch tokens should be identical: diff={max_inter_diff}");

        // Test with masked experts (id=-1)
        let masked_ids: Vec<i32> = vec![-1; top_k]; // All masked
        let work3 = MoeWork {
            moe_layer_idx: 0,
            activations: activation.clone(),
            topk_ids: masked_ids,
            topk_weights: expert_weights.clone(),
            batch_size: 1,
            topk: top_k,
        };
        work_tx.send(work3).expect("Send failed");
        let masked_output = result_rx.recv().expect("Recv failed");
        let masked_nonzero = masked_output.iter().filter(|&&v| v != 0).count();
        eprintln!("  Masked experts: {masked_nonzero}/{hidden} nonzero (should be 0)");
        assert_eq!(masked_nonzero, 0, "All-masked experts should produce zero output");

        // Clean up
        drop(work_tx);
        handle.join().expect("Worker panicked");
        eprintln!("  Worker thread joined cleanly");
    }

    #[test]
    fn test_shared_expert_v2_lite() {
        let _ = env_logger::try_init();
        let model_dir = Path::new("/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite");
        if !model_dir.exists() {
            eprintln!("Skipping — V2-Lite not downloaded");
            return;
        }

        let store = WeightStore::load_from_hf(model_dir, DEFAULT_GROUP_SIZE, None)
            .expect("Failed to load");

        // V2-Lite has 2 shared experts with routed_scaling_factor=1.0
        assert_eq!(store.config.n_shared_experts, 2);
        assert_eq!(store.config.routed_scaling_factor, 1.0);
        assert_eq!(store.shared_experts.len(), store.num_moe_layers());

        let hidden = store.config.hidden_size;
        let intermediate = store.config.moe_intermediate_size;
        let shared_intermediate = store.config.n_shared_experts * intermediate;
        let top_k = store.config.num_experts_per_tok;
        let group_size = store.group_size;

        // Check shared expert dimensions
        let shared_exp = &store.shared_experts[0];
        assert_eq!(shared_exp.gate.rows, shared_intermediate); // 2816
        assert_eq!(shared_exp.gate.cols, hidden);               // 2048
        assert_eq!(shared_exp.down.rows, hidden);               // 2048
        assert_eq!(shared_exp.down.cols, shared_intermediate); // 2816

        // Activation
        let mut activation = vec![0u16; hidden];
        for i in 0..hidden {
            activation[i] = f32_to_bf16(((i * 7 + 13) as f32 / hidden as f32 - 0.5) * 0.1);
        }

        let mut scratch = ExpertScratch::new(hidden, intermediate, group_size);
        let mut scratch_pool: Vec<ExpertScratch> = (0..top_k)
            .map(|_| ExpertScratch::new(hidden, intermediate, group_size))
            .collect();

        // First: MoE forward WITHOUT shared experts
        let mut output_no_shared = vec![0.0f32; hidden];
        let expert_indices: Vec<usize> = (0..top_k).collect();
        let expert_weights: Vec<f32> = vec![1.0 / top_k as f32; top_k];
        let mut no_shared: Option<ExpertScratch> = None;
        moe_forward(
            &store, 0, &activation, &expert_indices, &expert_weights,
            &mut output_no_shared, &mut scratch, &mut scratch_pool,
            &mut no_shared, true, None,
        );

        // Second: MoE forward WITH shared experts
        let mut output_with_shared = vec![0.0f32; hidden];
        let mut shared_scratch = Some(ExpertScratch::new(hidden, shared_intermediate, group_size));
        moe_forward(
            &store, 0, &activation, &expert_indices, &expert_weights,
            &mut output_with_shared, &mut scratch, &mut scratch_pool,
            &mut shared_scratch, true, None,
        );

        // Outputs should differ (shared expert adds to routed output)
        let mut max_diff: f32 = 0.0;
        for j in 0..hidden {
            max_diff = max_diff.max((output_with_shared[j] - output_no_shared[j]).abs());
        }

        let rms_no_shared = (output_no_shared.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / hidden as f64).sqrt();
        let rms_with_shared = (output_with_shared.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / hidden as f64).sqrt();

        eprintln!("V2-Lite shared expert test:");
        eprintln!("  n_shared={}, shared_intermediate={}", store.config.n_shared_experts, shared_intermediate);
        eprintln!("  RMS without shared: {rms_no_shared:.6}");
        eprintln!("  RMS with shared:    {rms_with_shared:.6}");
        eprintln!("  Max diff:           {max_diff:.6}");

        assert!(max_diff > 1e-4, "Shared expert should change output (max_diff={max_diff})");
        assert!(rms_with_shared > 1e-5, "Output with shared too small");
    }
}
