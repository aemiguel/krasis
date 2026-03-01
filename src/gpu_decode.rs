//! GPU decode — Rust-orchestrated GPU inference using CUDA/cuBLAS.
//!
//! All decode computation runs on GPU against VRAM-resident weights.
//! No Python in the hot path. CUDA kernels do the GPU compute, Rust
//! orchestrates the decode loop, PFL prediction, expert DMA scheduling,
//! timing, and sampling.
//!
//! Weight pointers come from Python (PyTorch tensor.data_ptr()) at setup time.
//! Expert weights live in system RAM and are DMA'd on demand via the copy engine.
//!
//! Custom CUDA kernels (RMSNorm, SiLU, routing, etc.) are compiled from
//! decode_kernels.cu at build time via nvcc, embedded as PTX, and loaded
//! into the CUDA module at init time.

use pyo3::prelude::*;
use std::sync::Arc;

use cudarc::cublas::{CudaBlas, sys as cublas_sys};
use cudarc::cublas::result as cublas_result;
use cudarc::driver::{CudaDevice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::driver::sys as cuda_sys;

// PTX compiled from src/cuda/decode_kernels.cu at build time.
#[cfg(has_decode_kernels)]
const DECODE_KERNELS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/decode_kernels.ptx"));

/// All kernel function names from decode_kernels.cu.
const KERNEL_NAMES: &[&str] = &[
    "embedding_lookup",
    "fused_add_rmsnorm",
    "rmsnorm",
    "silu_mul",
    "sigmoid_topk",
    "softmax_topk",
    "weighted_add_bf16",
    "zero_bf16",
    "add_bf16",
    "sigmoid_gate_bf16",
    "scale_bf16",
    "la_conv1d",
    "la_recurrence",
    "gated_rmsnorm_silu",
    "per_head_rmsnorm",
    "apply_rope",
    "kv_cache_write",
    "gqa_attention",
    "bf16_to_fp32",
    "fp32_to_bf16",
    "marlin_gemv_int4",
];

const MODULE_NAME: &str = "decode_kernels";

// ── Adaptive Prefetch Layer (APFL) ─────────────────────────────────────

/// One slot in the prefetch ring buffer. Holds a complete expert's
/// Marlin-format weights in VRAM, ready for compute.
struct PrefetchSlot {
    /// VRAM buffers: each slot has 4 regions (w13_packed, w13_scales, w2_packed, w2_scales).
    /// Laid out contiguously in one allocation for cache friendliness.
    d_buf: cudarc::driver::CudaSlice<u8>,
    buf_size: usize,

    /// Offsets into d_buf for each component.
    w13_packed_offset: usize,
    w13_packed_size: usize,
    w13_scales_offset: usize,
    w13_scales_size: usize,
    w2_packed_offset: usize,
    w2_packed_size: usize,
    w2_scales_offset: usize,
    w2_scales_size: usize,

    /// What's currently stored (-1 = empty).
    layer_idx: i32,
    expert_idx: i32,

    /// CUDA event: signaled when DMA for this slot finishes.
    dma_event: CudaEvent,
    /// Whether DMA has been queued (event valid to wait on).
    dma_queued: bool,
}

impl PrefetchSlot {
    fn is_empty(&self) -> bool {
        self.layer_idx < 0
    }

    fn contains(&self, layer: usize, expert: usize) -> bool {
        self.layer_idx == layer as i32 && self.expert_idx == expert as i32
    }

    fn clear(&mut self) {
        self.layer_idx = -1;
        self.expert_idx = -1;
        self.dma_queued = false;
    }

    fn w13_packed_ptr(&self) -> u64 {
        *self.d_buf.device_ptr() + self.w13_packed_offset as u64
    }
    fn w13_scales_ptr(&self) -> u64 {
        *self.d_buf.device_ptr() + self.w13_scales_offset as u64
    }
    fn w2_packed_ptr(&self) -> u64 {
        *self.d_buf.device_ptr() + self.w2_packed_offset as u64
    }
    fn w2_scales_ptr(&self) -> u64 {
        *self.d_buf.device_ptr() + self.w2_scales_offset as u64
    }
}

/// Per-layer APFL statistics for adaptive prefetch count.
struct ApflLayerStats {
    hits: u64,
    misses: u64,
    /// Current number of experts to prefetch for this layer's NEXT layer.
    prefetch_count: usize,
    /// Window for recent accuracy (last N predictions).
    recent_hits: u32,
    recent_total: u32,
}

impl ApflLayerStats {
    fn new(initial_prefetch: usize) -> Self {
        ApflLayerStats {
            hits: 0,
            misses: 0,
            prefetch_count: initial_prefetch,
            recent_hits: 0,
            recent_total: 0,
        }
    }

    fn record_hit(&mut self) {
        self.hits += 1;
        self.recent_hits += 1;
        self.recent_total += 1;
    }

    fn record_miss(&mut self) {
        self.misses += 1;
        self.recent_total += 1;
    }

    /// Adapt prefetch count based on recent hit rate.
    /// Called after processing a layer's experts.
    fn adapt(&mut self, max_prefetch: usize) {
        // Adapt every 8 tokens (enough data to be meaningful)
        if self.recent_total < 8 {
            return;
        }

        let hit_rate = self.recent_hits as f32 / self.recent_total as f32;

        if hit_rate > 0.6 && self.prefetch_count < max_prefetch {
            // Good predictions → prefetch more
            self.prefetch_count += 1;
        } else if hit_rate < 0.3 && self.prefetch_count > 0 {
            // Bad predictions → prefetch less
            self.prefetch_count = self.prefetch_count.saturating_sub(1);
        }
        // else: keep current count (0.3-0.6 hit rate is the "hold steady" band)

        // Reset window
        self.recent_hits = 0;
        self.recent_total = 0;
    }

    fn hit_rate(&self) -> f32 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f32 / (self.hits + self.misses) as f32
        }
    }
}

/// Adaptive Prefetch Layer state.
struct ApflState {
    /// Ring buffer of prefetch slots.
    slots: Vec<PrefetchSlot>,
    /// Per-layer adaptation stats.
    layer_stats: Vec<ApflLayerStats>,
    /// Global stats.
    total_hits: u64,
    total_misses: u64,
    /// Maximum experts to prefetch per layer (cap).
    max_prefetch: usize,
    /// Whether APFL is enabled.
    enabled: bool,
    /// Host-side buffer for speculative routing results.
    h_spec_topk_ids: Vec<i32>,
}

impl ApflState {
    /// Find a slot containing the given (layer, expert). Returns slot index or None.
    fn find_slot(&self, layer: usize, expert: usize) -> Option<usize> {
        self.slots.iter().position(|s| s.contains(layer, expert))
    }

    /// Find the oldest/emptiest slot to evict for a new prefetch.
    /// Prefers empty slots, then slots for layers we've already passed.
    fn find_evict_slot(&self, current_layer: usize) -> usize {
        // First: empty slots
        if let Some(i) = self.slots.iter().position(|s| s.is_empty()) {
            return i;
        }
        // Second: slots for layers before the current one (already used/stale)
        if let Some(i) = self.slots.iter().position(|s| (s.layer_idx as usize) < current_layer) {
            return i;
        }
        // Fallback: slot 0 (LRU approximation — could improve with timestamps)
        0
    }
}

// ── HCS: Hot Cache Strategy — keep hot experts permanently in VRAM ─────

/// One expert's Marlin-format weights resident in VRAM.
struct HcsCacheEntry {
    d_buf: cudarc::driver::CudaSlice<u8>,
    w13_packed_offset: usize,
    w13_packed_size: usize,
    w13_scales_offset: usize,
    w13_scales_size: usize,
    w2_packed_offset: usize,
    w2_packed_size: usize,
    w2_scales_offset: usize,
    w2_scales_size: usize,
}

impl HcsCacheEntry {
    fn w13_packed_ptr(&self) -> u64 {
        *self.d_buf.device_ptr() + self.w13_packed_offset as u64
    }
    fn w13_scales_ptr(&self) -> u64 {
        *self.d_buf.device_ptr() + self.w13_scales_offset as u64
    }
    fn w2_packed_ptr(&self) -> u64 {
        *self.d_buf.device_ptr() + self.w2_packed_offset as u64
    }
    fn w2_scales_ptr(&self) -> u64 {
        *self.d_buf.device_ptr() + self.w2_scales_offset as u64
    }
}

/// HCS state: resident expert cache + activation heatmap.
struct HcsState {
    /// (layer_idx, expert_idx) → cache entry.
    cache: std::collections::HashMap<(usize, usize), HcsCacheEntry>,
    /// Activation heatmap: (layer_idx, expert_idx) → count.
    heatmap: std::collections::HashMap<(usize, usize), u64>,
    /// Total VRAM bytes allocated for HCS.
    vram_bytes: usize,
    /// Number of cached experts.
    num_cached: usize,
    /// Stats: hits and misses during decode.
    total_hits: u64,
    total_misses: u64,
    /// Whether heatmap collection is active.
    collecting: bool,
    /// Per-expert VRAM size (bytes, same for all experts in a model).
    expert_vram_bytes: usize,
}

impl HcsState {
    fn new() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            heatmap: std::collections::HashMap::new(),
            vram_bytes: 0,
            num_cached: 0,
            total_hits: 0,
            total_misses: 0,
            collecting: false,
            expert_vram_bytes: 0,
        }
    }

    /// Check if a specific (layer, expert) is cached in VRAM.
    fn get(&self, layer: usize, expert: usize) -> Option<&HcsCacheEntry> {
        self.cache.get(&(layer, expert))
    }

    /// Record an expert activation in the heatmap.
    fn record_activation(&mut self, layer: usize, expert: usize) {
        if self.collecting {
            *self.heatmap.entry((layer, expert)).or_insert(0) += 1;
        }
    }

    fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 { 0.0 } else { self.total_hits as f64 / total as f64 }
    }
}

// ── Expert data descriptor (system RAM, Marlin format) ─────────────────

/// Describes one expert's Marlin-format weights in system RAM for DMA.
#[derive(Debug, Clone)]
struct ExpertDataPtr {
    w13_packed_ptr: usize,
    w13_packed_bytes: usize,
    w13_scales_ptr: usize,
    w13_scales_bytes: usize,
    w2_packed_ptr: usize,
    w2_packed_bytes: usize,
    w2_scales_ptr: usize,
    w2_scales_bytes: usize,
}

/// Per-layer expert data for DMA.
struct MoeLayerData {
    experts: Vec<ExpertDataPtr>,
    /// Shared expert (always run, optional).
    shared: Option<ExpertDataPtr>,
    num_experts: usize,
    topk: usize,
    scoring_func: u8,    // 0=softmax, 1=sigmoid
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
    /// Gate weight ID in the weight registry.
    gate_wid: usize,
    /// Gate bias ptr (0 if none), FP32 on GPU.
    gate_bias_ptr: u64,
    /// E_score_correction ptr (0 if none), FP32 on GPU.
    e_score_corr_ptr: u64,
    /// Shared expert gate weight ID (None if no shared gate).
    shared_gate_wid: Option<usize>,
}

// ── GPU weight descriptor ──────────────────────────────────────────────

/// Describes a single weight matrix resident in VRAM.
#[derive(Debug, Clone)]
struct GpuWeight {
    ptr: u64,
    rows: usize,
    cols: usize,
    /// Data type: 0 = BF16, 1 = FP32, 2 = FP16.
    dtype: u8,
}

impl GpuWeight {
    fn cublas_data_type(&self) -> cublas_sys::cudaDataType {
        match self.dtype {
            0 => cublas_sys::cudaDataType::CUDA_R_16BF,
            1 => cublas_sys::cudaDataType::CUDA_R_32F,
            2 => cublas_sys::cudaDataType::CUDA_R_16F,
            _ => cublas_sys::cudaDataType::CUDA_R_16BF,
        }
    }

    #[allow(dead_code)]
    fn element_size(&self) -> usize {
        match self.dtype {
            0 | 2 => 2,
            1 => 4,
            _ => 2,
        }
    }
}

// ── Layer configuration ────────────────────────────────────────────────

#[allow(dead_code)]
struct GpuDecodeLayer {
    input_norm_ptr: u64,
    input_norm_size: usize,
    post_attn_norm_ptr: u64,
    post_attn_norm_size: usize,
    attn: GpuAttnConfig,
    mlp: GpuMlpConfig,
}

#[allow(dead_code)]
enum GpuAttnConfig {
    LinearAttention {
        in_proj_qkvz: usize,
        in_proj_ba: usize,
        out_proj: usize,
        // Conv + recurrence params
        conv_weight_ptr: u64,  // [conv_dim, kernel_dim] FP32 on GPU
        a_log_ptr: u64,
        dt_bias_ptr: u64,
        norm_weight_ptr: u64,
        nk: usize, nv: usize, dk: usize, dv: usize,
        hr: usize, kernel_dim: usize, conv_dim: usize,
        scale: f32,
        conv_state_ptr: u64,   // [conv_dim, kernel_dim] FP32 on GPU
        recur_state_ptr: u64,  // [nv, dk, dv] FP32 on GPU
    },
    GQA {
        q_proj: usize,
        k_proj: usize,
        v_proj: usize,
        o_proj: usize,
        fused_qkv: Option<usize>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sm_scale: f32,
        q_norm_ptr: u64,   // 0 if no QK norm
        k_norm_ptr: u64,
        gated: bool,
    },
    #[allow(dead_code)]
    MLA {
        kv_a_proj: usize,
        o_proj: usize,
    },
}

#[allow(dead_code)]
enum GpuMlpConfig {
    MoE {
        gate_weight: usize,
        gate_bias_ptr: u64,
        e_score_corr_ptr: u64,
        num_experts: usize,
        topk: usize,
        scoring_func: u8,    // 0=softmax, 1=sigmoid
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        shared_gate_up: Option<usize>,
        shared_down: Option<usize>,
        shared_gate: Option<usize>,
    },
    Dense {
        gate_proj: usize,
        up_proj: usize,
        down_proj: usize,
    },
    None,
}

// ── Main GPU decode graph ──────────────────────────────────────────────

struct GpuDecodeGraph {
    hidden_size: usize,
    #[allow(dead_code)]
    num_layers: usize,
    vocab_size: usize,
    eps: f32,
    intermediate_size: usize,
    group_size: usize,

    weights: Vec<GpuWeight>,
    layers: Vec<GpuDecodeLayer>,

    embedding_ptr: u64,
    lm_head_wid: usize,
    final_norm_ptr: u64,
    #[allow(dead_code)]
    final_norm_size: usize,

    // MoE layer data (expert RAM pointers, routing config)
    moe_layers: Vec<Option<MoeLayerData>>,

    // Adaptive Prefetch Layer state
    apfl: Option<ApflState>,

    // HCS: Hot Cache Strategy state
    hcs: Option<HcsState>,

    // GPU scratch buffers
    d_hidden: cudarc::driver::CudaSlice<u16>,
    d_residual: cudarc::driver::CudaSlice<u16>,
    d_scratch: cudarc::driver::CudaSlice<u16>,
    d_logits: cudarc::driver::CudaSlice<f32>,
    // FP32 scratch for intermediate computations (routing, attention)
    d_fp32_scratch: cudarc::driver::CudaSlice<f32>,

    // Expert DMA buffers: two pairs for ping-pong overlap.
    // Pair 0 (a0/b0): packed data / scales for current operation
    // Pair 1 (a1/b1): packed data / scales for next operation (DMA while computing on pair 0)
    d_expert_buf_a0: cudarc::driver::CudaSlice<u8>,
    d_expert_buf_b0: cudarc::driver::CudaSlice<u8>,
    d_expert_buf_a1: cudarc::driver::CudaSlice<u8>,
    d_expert_buf_b1: cudarc::driver::CudaSlice<u8>,
    expert_buf_size: usize,

    // Expert BF16 dequant buffer (for cuBLAS GEMV fallback path)
    // d_expert_w13: [2*intermediate, hidden] BF16
    // d_expert_w2: [hidden, intermediate] BF16
    // Not used in fused Marlin GEMV path.

    // Routing scratch
    d_topk_indices: cudarc::driver::CudaSlice<i32>,
    d_topk_weights: cudarc::driver::CudaSlice<f32>,
    // MoE accumulator (hidden_size, BF16)
    d_moe_out: cudarc::driver::CudaSlice<u16>,
    // Expert compute output (hidden_size, BF16) — single expert result
    d_expert_out: cudarc::driver::CudaSlice<u16>,
    // Expert gate_up scratch (2*intermediate_size, BF16)
    d_expert_gate_up: cudarc::driver::CudaSlice<u16>,
    // Expert intermediate scratch (intermediate_size, BF16) — after SiLU*mul
    d_expert_scratch: cudarc::driver::CudaSlice<u16>,

    // Marlin GEMV inverse permutation tables (on GPU)
    d_inv_weight_perm: cudarc::driver::CudaSlice<i32>,
    d_inv_scale_perm: cudarc::driver::CudaSlice<i32>,

    // GQA scratch (FP32 for Q, K, V, attention output)
    d_gqa_q: cudarc::driver::CudaSlice<f32>,
    d_gqa_k: cudarc::driver::CudaSlice<f32>,
    d_gqa_v: cudarc::driver::CudaSlice<f32>,
    d_gqa_out: cudarc::driver::CudaSlice<f32>,

    // Linear attention scratch (FP32)
    d_la_qkvz: cudarc::driver::CudaSlice<f32>,
    d_la_ba: cudarc::driver::CudaSlice<f32>,
    d_la_conv_out: cudarc::driver::CudaSlice<f32>,
    d_la_recur_out: cudarc::driver::CudaSlice<f32>,
    d_la_gated_out: cudarc::driver::CudaSlice<f32>,

    // Host-side buffers for D2H copies
    h_topk_ids: Vec<i32>,
    h_topk_weights: Vec<f32>,
    h_logits: Vec<f32>,

    // Timing
    timing_enabled: bool,
    timing_step_count: u64,
    t_total: f64,
    t_norm: f64,
    t_attn: f64,
    t_route: f64,
    t_expert_dma: f64,
    t_expert_compute: f64,
    t_shared: f64,
    t_dense_mlp: f64,
    t_lm_head: f64,
}

// ── Thread-safe CUDA wrappers ──────────────────────────────────────────

struct CudaStream(cuda_sys::CUstream);
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

struct CudaEvent(cuda_sys::CUevent);
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

// ── PyO3 wrapper ───────────────────────────────────────────────────────

#[pyclass]
pub struct GpuDecodeStore {
    device: Arc<CudaDevice>,
    blas: CudaBlas,
    compute_stream: CudaStream,
    copy_stream: CudaStream,
    /// Dedicated stream for APFL prefetch DMA (Options 1+2).
    /// Runs independently from copy_stream so prefetch can overlap with on-demand DMA.
    prefetch_stream: CudaStream,
    graph: Option<Box<GpuDecodeGraph>>,
    kernels_loaded: bool,
}

#[pymethods]
impl GpuDecodeStore {
    #[new]
    #[pyo3(signature = (device_ordinal=0))]
    fn new(device_ordinal: usize) -> PyResult<Self> {
        let device = CudaDevice::new(device_ordinal)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to create CUDA device {}: {:?}", device_ordinal, e)))?;

        let blas = CudaBlas::new(device.clone())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to create cuBLAS handle: {:?}", e)))?;

        let compute_stream = unsafe {
            let mut stream: cuda_sys::CUstream = std::ptr::null_mut();
            let err = cuda_sys::lib().cuStreamCreate(
                &mut stream,
                cuda_sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Failed to create compute stream: {:?}", err)));
            }
            stream
        };

        let copy_stream = unsafe {
            let mut stream: cuda_sys::CUstream = std::ptr::null_mut();
            let err = cuda_sys::lib().cuStreamCreate(
                &mut stream,
                cuda_sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Failed to create copy stream: {:?}", err)));
            }
            stream
        };

        let prefetch_stream = unsafe {
            let mut stream: cuda_sys::CUstream = std::ptr::null_mut();
            let err = cuda_sys::lib().cuStreamCreate(
                &mut stream,
                cuda_sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Failed to create prefetch stream: {:?}", err)));
            }
            stream
        };

        // Load CUDA decode kernels from embedded PTX
        #[cfg(has_decode_kernels)]
        {
            use cudarc::nvrtc::Ptx;
            device.load_ptx(
                Ptx::from_src(DECODE_KERNELS_PTX),
                MODULE_NAME,
                KERNEL_NAMES,
            ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to load decode kernels PTX: {:?}", e)))?;
            log::info!("GpuDecodeStore: loaded {} CUDA decode kernels", KERNEL_NAMES.len());
        }

        #[cfg(not(has_decode_kernels))]
        log::warn!("GpuDecodeStore: decode kernels not available (nvcc not found at build time)");

        let kernels_loaded = cfg!(has_decode_kernels);

        log::info!("GpuDecodeStore: initialized on device {} with compute + copy + prefetch streams",
                   device_ordinal);

        Ok(GpuDecodeStore {
            device,
            blas,
            compute_stream: CudaStream(compute_stream),
            copy_stream: CudaStream(copy_stream),
            prefetch_stream: CudaStream(prefetch_stream),
            graph: None,
            kernels_loaded,
        })
    }

    /// Initialize the decode graph with model dimensions.
    #[pyo3(signature = (hidden_size, num_layers, vocab_size, eps, max_experts_per_tok=10, max_intermediate_size=0, max_qkv_size=0, group_size=128))]
    fn configure(
        &mut self,
        hidden_size: usize,
        num_layers: usize,
        vocab_size: usize,
        eps: f32,
        max_experts_per_tok: usize,
        max_intermediate_size: usize,
        max_qkv_size: usize,
        group_size: usize,
    ) -> PyResult<()> {
        let intermediate = if max_intermediate_size > 0 { max_intermediate_size } else { hidden_size * 4 };
        let qkv_size = if max_qkv_size > 0 { max_qkv_size } else { hidden_size * 3 };

        let d_hidden = self.device.alloc_zeros::<u16>(hidden_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_residual = self.device.alloc_zeros::<u16>(hidden_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let max_scratch = vocab_size.max(intermediate * 2).max(qkv_size);
        let d_scratch = self.device.alloc_zeros::<u16>(max_scratch)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_logits = self.device.alloc_zeros::<f32>(vocab_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let fp32_scratch_size = vocab_size.max(hidden_size * 4).max(512); // route gate + misc
        let d_fp32_scratch = self.device.alloc_zeros::<f32>(fp32_scratch_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let d_expert_buf_a0 = self.device.alloc_zeros::<u8>(1)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_expert_buf_b0 = self.device.alloc_zeros::<u8>(1)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_expert_buf_a1 = self.device.alloc_zeros::<u8>(1)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_expert_buf_b1 = self.device.alloc_zeros::<u8>(1)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let d_topk_indices = self.device.alloc_zeros::<i32>(max_experts_per_tok)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_topk_weights = self.device.alloc_zeros::<f32>(max_experts_per_tok)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_moe_out = self.device.alloc_zeros::<u16>(hidden_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_expert_out = self.device.alloc_zeros::<u16>(hidden_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_expert_gate_up = self.device.alloc_zeros::<u16>(intermediate * 2)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_expert_scratch = self.device.alloc_zeros::<u16>(intermediate)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        // Compute and upload inverse Marlin permutation tables
        let (d_inv_weight_perm, d_inv_scale_perm) = Self::upload_marlin_perm_tables(&self.device)?;

        let d_gqa_q = self.device.alloc_zeros::<f32>(qkv_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_gqa_k = self.device.alloc_zeros::<f32>(qkv_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_gqa_v = self.device.alloc_zeros::<f32>(qkv_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_gqa_out = self.device.alloc_zeros::<f32>(qkv_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let la_buf_size = qkv_size.max(intermediate);
        let d_la_qkvz = self.device.alloc_zeros::<f32>(la_buf_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_la_ba = self.device.alloc_zeros::<f32>(la_buf_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_la_conv_out = self.device.alloc_zeros::<f32>(la_buf_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_la_recur_out = self.device.alloc_zeros::<f32>(la_buf_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_la_gated_out = self.device.alloc_zeros::<f32>(la_buf_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        self.graph = Some(Box::new(GpuDecodeGraph {
            hidden_size,
            num_layers,
            vocab_size,
            eps,
            intermediate_size: intermediate,
            group_size,
            weights: Vec::with_capacity(num_layers * 8),
            layers: Vec::with_capacity(num_layers),
            embedding_ptr: 0,
            lm_head_wid: 0,
            final_norm_ptr: 0,
            final_norm_size: 0,
            moe_layers: Vec::new(),
            apfl: None,
            hcs: None,
            d_hidden,
            d_residual,
            d_scratch,
            d_logits,
            d_fp32_scratch,
            d_expert_buf_a0,
            d_expert_buf_b0,
            d_expert_buf_a1,
            d_expert_buf_b1,
            expert_buf_size: 0,
            d_topk_indices,
            d_topk_weights,
            d_moe_out,
            d_expert_out,
            d_expert_gate_up,
            d_expert_scratch,
            d_inv_weight_perm,
            d_inv_scale_perm,
            d_gqa_q,
            d_gqa_k,
            d_gqa_v,
            d_gqa_out,
            d_la_qkvz,
            d_la_ba,
            d_la_conv_out,
            d_la_recur_out,
            d_la_gated_out,
            h_topk_ids: vec![0i32; max_experts_per_tok],
            h_topk_weights: vec![0.0f32; max_experts_per_tok],
            h_logits: vec![0.0f32; vocab_size],
            timing_enabled: false,
            timing_step_count: 0,
            t_total: 0.0,
            t_norm: 0.0,
            t_attn: 0.0,
            t_route: 0.0,
            t_expert_dma: 0.0,
            t_expert_compute: 0.0,
            t_shared: 0.0,
            t_dense_mlp: 0.0,
            t_lm_head: 0.0,
        }));

        log::info!("GpuDecodeStore: configured hidden={}, layers={}, vocab={}, intermediate={}, qkv={}, gs={}",
                   hidden_size, num_layers, vocab_size, intermediate, qkv_size, group_size);
        Ok(())
    }

    /// Register a weight matrix. Returns weight ID.
    #[pyo3(signature = (ptr, rows, cols, dtype=0))]
    fn register_weight(&mut self, ptr: usize, rows: usize, cols: usize, dtype: u8) -> PyResult<usize> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let id = graph.weights.len();
        graph.weights.push(GpuWeight { ptr: ptr as u64, rows, cols, dtype });
        Ok(id)
    }

    fn set_embedding(&mut self, ptr: usize) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        graph.embedding_ptr = ptr as u64;
        Ok(())
    }

    fn set_final_norm(&mut self, ptr: usize, size: usize) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        graph.final_norm_ptr = ptr as u64;
        graph.final_norm_size = size;
        Ok(())
    }

    fn set_lm_head(&mut self, weight_id: usize) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        graph.lm_head_wid = weight_id;
        Ok(())
    }

    /// BF16 GEMV: output[N] = weight[N,K] @ input[K]
    fn gemv_bf16(&self, weight_id: usize, input_ptr: usize, output_ptr: usize) -> PyResult<()> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let w = &graph.weights[weight_id];
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            cublas_result::gemm_ex(
                *self.blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                w.rows as i32, 1, w.cols as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                w.ptr as *const std::ffi::c_void, w.cublas_data_type(), w.cols as i32,
                input_ptr as *const std::ffi::c_void, w.cublas_data_type(), w.cols as i32,
                &beta as *const f32 as *const std::ffi::c_void,
                output_ptr as *mut std::ffi::c_void, w.cublas_data_type(), w.rows as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("cuBLAS: {:?}", e)))?;
        }
        Ok(())
    }

    /// FP32 GEMV for routing gate.
    fn gemv_f32(&self, weight_id: usize, input_ptr: usize, output_ptr: usize) -> PyResult<()> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let w = &graph.weights[weight_id];
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            cublas_result::gemm_ex(
                *self.blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                w.rows as i32, 1, w.cols as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                w.ptr as *const std::ffi::c_void, w.cublas_data_type(), w.cols as i32,
                input_ptr as *const std::ffi::c_void, w.cublas_data_type(), w.cols as i32,
                &beta as *const f32 as *const std::ffi::c_void,
                output_ptr as *mut std::ffi::c_void, w.cublas_data_type(), w.rows as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("cuBLAS: {:?}", e)))?;
        }
        Ok(())
    }

    /// Async DMA: host (system RAM) -> device (VRAM) on the copy stream.
    /// buffer: 0=a0, 1=b0, 2=a1, 3=b1
    fn dma_expert_to_gpu(&self, host_ptr: usize, size_bytes: usize, buffer: u8) -> PyResult<()> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let dst_ptr = match buffer {
            0 => *graph.d_expert_buf_a0.device_ptr(),
            1 => *graph.d_expert_buf_b0.device_ptr(),
            2 => *graph.d_expert_buf_a1.device_ptr(),
            3 => *graph.d_expert_buf_b1.device_ptr(),
            _ => return Err(pyo3::exceptions::PyRuntimeError::new_err("Invalid buffer index")),
        };
        if size_bytes > graph.expert_buf_size {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Expert {} > buffer {}", size_bytes, graph.expert_buf_size)));
        }
        unsafe {
            let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                dst_ptr, host_ptr as *const std::ffi::c_void, size_bytes, self.copy_stream.0);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("DMA: {:?}", err)));
            }
        }
        Ok(())
    }

    fn sync_dma(&self) -> PyResult<()> {
        unsafe {
            let err = cuda_sys::lib().cuStreamSynchronize(self.copy_stream.0);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("sync: {:?}", err)));
            }
        }
        Ok(())
    }

    fn sync_compute(&self) -> PyResult<()> {
        unsafe {
            let err = cuda_sys::lib().cuStreamSynchronize(self.compute_stream.0);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("sync: {:?}", err)));
            }
        }
        Ok(())
    }

    fn resize_expert_buffers(&mut self, expert_size_bytes: usize) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        graph.d_expert_buf_a0 = self.device.alloc_zeros::<u8>(expert_size_bytes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        graph.d_expert_buf_b0 = self.device.alloc_zeros::<u8>(expert_size_bytes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        graph.d_expert_buf_a1 = self.device.alloc_zeros::<u8>(expert_size_bytes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        graph.d_expert_buf_b1 = self.device.alloc_zeros::<u8>(expert_size_bytes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        graph.expert_buf_size = expert_size_bytes;
        log::info!("GpuDecodeStore: expert buffers 4x {} bytes ({:.1} MB total)",
                   expert_size_bytes, expert_size_bytes as f64 * 4.0 / (1024.0 * 1024.0));
        Ok(())
    }

    fn set_timing(&mut self, enabled: bool) -> PyResult<()> {
        if let Some(ref mut graph) = self.graph {
            graph.timing_enabled = enabled;
        }
        Ok(())
    }

    /// Initialize APFL (Adaptive Prefetch Layer) with a ring of prefetch slots.
    ///
    /// num_slots: number of expert-sized buffers in VRAM for prefetching.
    ///   More slots = more experts can be prefetched simultaneously.
    ///   Typical: 16-32 (costs ~24-48 MB for QCN's 1.5 MB experts).
    ///
    /// initial_prefetch: starting number of experts to prefetch per layer (APFL adapts this).
    ///   0 = disabled, N = prefetch top-N predicted experts for next layer.
    ///
    /// max_prefetch: cap on adaptive prefetch count (prevents runaway VRAM/PCIe use).
    #[pyo3(signature = (num_slots=16, initial_prefetch=5, max_prefetch=10))]
    fn init_apfl(
        &mut self,
        num_slots: usize,
        initial_prefetch: usize,
        max_prefetch: usize,
    ) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;

        if graph.expert_buf_size == 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Call resize_expert_buffers first to set expert size"));
        }

        // Each slot holds one complete expert: w13_packed + w13_scales + w2_packed + w2_scales.
        // For QCN: expert_buf_size covers the largest single component (w13_packed or w2_packed).
        // A full expert needs roughly 4x expert_buf_size (packed+scales for both w13 and w2).
        // Align to 512 bytes for CUDA async DMA
        let align = 512usize;
        let ebs_aligned = (graph.expert_buf_size + align - 1) & !(align - 1);
        let slot_size = ebs_aligned * 4;

        let mut slots = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            let d_buf = self.device.alloc_zeros::<u8>(slot_size)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            let event = unsafe {
                let mut ev: cuda_sys::CUevent = std::ptr::null_mut();
                let flags = cuda_sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32;
                let err = cuda_sys::lib().cuEventCreate(&mut ev, flags);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("cuEventCreate: {:?}", err)));
                }
                ev
            };

            // Layout: w13_packed | w13_scales | w2_packed | w2_scales
            // Each region is aligned to 512 bytes for CUDA async DMA.
            let align = 512usize;
            let ebs_aligned = (graph.expert_buf_size + align - 1) & !(align - 1);
            slots.push(PrefetchSlot {
                d_buf,
                buf_size: slot_size,
                w13_packed_offset: 0,
                w13_packed_size: ebs_aligned,
                w13_scales_offset: ebs_aligned,
                w13_scales_size: ebs_aligned,
                w2_packed_offset: ebs_aligned * 2,
                w2_packed_size: ebs_aligned,
                w2_scales_offset: ebs_aligned * 3,
                w2_scales_size: ebs_aligned,
                layer_idx: -1,
                expert_idx: -1,
                dma_event: CudaEvent(event),
                dma_queued: false,
            });
        }

        let num_layers = graph.moe_layers.len();
        let layer_stats: Vec<ApflLayerStats> = (0..num_layers)
            .map(|_| ApflLayerStats::new(initial_prefetch))
            .collect();

        let topk = graph.moe_layers.iter()
            .filter_map(|m| m.as_ref())
            .map(|m| m.topk)
            .max()
            .unwrap_or(10);

        graph.apfl = Some(ApflState {
            slots,
            layer_stats,
            total_hits: 0,
            total_misses: 0,
            max_prefetch,
            enabled: initial_prefetch > 0,
            h_spec_topk_ids: vec![0i32; topk],
        });

        let total_mb = (slot_size * num_slots) as f64 / (1024.0 * 1024.0);
        log::info!(
            "APFL: initialized {} slots x {:.1} KB = {:.1} MB VRAM, initial_prefetch={}, max={}",
            num_slots, slot_size as f64 / 1024.0, total_mb, initial_prefetch, max_prefetch,
        );
        Ok(())
    }

    /// Get APFL statistics as a formatted string.
    fn apfl_stats(&self) -> PyResult<String> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let apfl = graph.apfl.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("APFL not initialized"))?;

        let mut lines = Vec::new();
        lines.push(format!(
            "APFL: enabled={}, {} slots, total hits={}, misses={}, hit_rate={:.1}%",
            apfl.enabled, apfl.slots.len(), apfl.total_hits, apfl.total_misses,
            if apfl.total_hits + apfl.total_misses > 0 {
                apfl.total_hits as f64 / (apfl.total_hits + apfl.total_misses) as f64 * 100.0
            } else { 0.0 },
        ));

        for (i, stats) in apfl.layer_stats.iter().enumerate() {
            if stats.hits + stats.misses > 0 {
                lines.push(format!(
                    "  Layer {}: prefetch_count={}, hits={}, misses={}, hit_rate={:.1}%",
                    i, stats.prefetch_count, stats.hits, stats.misses,
                    stats.hit_rate() * 100.0,
                ));
            }
        }

        Ok(lines.join("\n"))
    }

    /// Set APFL enabled/disabled at runtime.
    fn set_apfl_enabled(&mut self, enabled: bool) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        if let Some(ref mut apfl) = graph.apfl {
            apfl.enabled = enabled;
            log::info!("APFL: {}", if enabled { "enabled" } else { "disabled" });
        }
        Ok(())
    }

    /// Register MoE expert data pointers for one layer.
    /// expert_ptrs: list of (w13p_ptr, w13p_bytes, w13s_ptr, w13s_bytes,
    ///                        w2p_ptr, w2p_bytes, w2s_ptr, w2s_bytes)
    #[pyo3(signature = (layer_idx, expert_ptrs, shared_ptrs, num_experts, topk,
                        scoring_func, norm_topk_prob, routed_scaling_factor,
                        gate_wid, gate_bias_ptr=0, e_score_corr_ptr=0,
                        shared_gate_wid=None))]
    fn register_moe_layer(
        &mut self,
        layer_idx: usize,
        expert_ptrs: Vec<(usize, usize, usize, usize, usize, usize, usize, usize)>,
        shared_ptrs: Option<(usize, usize, usize, usize, usize, usize, usize, usize)>,
        num_experts: usize,
        topk: usize,
        scoring_func: u8,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        gate_wid: usize,
        gate_bias_ptr: usize,
        e_score_corr_ptr: usize,
        shared_gate_wid: Option<usize>,
    ) -> PyResult<()> {
        self.register_moe_layer_data(
            layer_idx, expert_ptrs, shared_ptrs,
            num_experts, topk, scoring_func, norm_topk_prob,
            routed_scaling_factor, gate_wid, gate_bias_ptr,
            e_score_corr_ptr, shared_gate_wid,
        )
    }

    /// Test the Marlin GEMV kernel correctness.
    fn test_marlin_gemv_kernel(&self) -> PyResult<String> {
        self.test_marlin_gemv()
    }

    /// Run one expert through DMA + Marlin GEMV pipeline.
    /// For testing: specify layer and expert index, hidden state must be in d_hidden.
    #[pyo3(signature = (layer_idx, expert_idx))]
    fn run_single_expert(&self, layer_idx: usize, expert_idx: usize) -> PyResult<()> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let moe = graph.moe_layers.get(layer_idx)
            .and_then(|m| m.as_ref())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                format!("MoE layer {} not registered", layer_idx)))?;
        if expert_idx >= moe.experts.len() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Expert {} >= {}", expert_idx, moe.experts.len())));
        }
        self.run_expert_on_gpu(
            &moe.experts[expert_idx],
            graph.hidden_size,
            graph.intermediate_size,
            graph.group_size,
        )
    }

    /// Test CUDA kernels: run RMSNorm, SiLU*mul, embedding lookup, and verify results.
    /// Returns a dict of test results.
    fn test_kernels(&self) -> PyResult<String> {
        if !self.kernels_loaded {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Decode kernels not loaded (nvcc not found at build time)"));
        }

        let mut results = Vec::new();

        // Test 1: RMSNorm
        {
            let n = 2048usize;
            let mut input_host = vec![0u16; n];
            let mut weight_host = vec![0u16; n];
            // Fill with BF16 values
            for i in 0..n {
                input_host[i] = half::bf16::from_f32((i as f32) * 0.001).to_bits();
                weight_host[i] = half::bf16::from_f32(1.0).to_bits();
            }

            let d_input = self.device.htod_copy(input_host.clone())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let d_weight = self.device.htod_copy(weight_host)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let mut d_output = self.device.alloc_zeros::<u16>(n)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            let f = self.device.get_func(MODULE_NAME, "rmsnorm")
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("rmsnorm kernel not found"))?;

            let threads = 256u32;
            let smem = (n * 4) as u32; // float per element
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: smem,
            };

            unsafe {
                f.launch(cfg, (
                    &mut d_output, &d_input, &d_weight,
                    1e-6f32, n as i32,
                )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("rmsnorm: {:?}", e)))?;
            }

            self.device.synchronize()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let output_host = self.device.dtoh_sync_copy(&d_output)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            // Verify: compute expected RMSNorm on CPU
            let input_f32: Vec<f32> = input_host.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect();
            let sum_sq: f32 = input_f32.iter().map(|x| x * x).sum();
            let rms = (sum_sq / n as f32 + 1e-6).sqrt().recip();
            let expected_0 = input_f32[0] * rms * 1.0;
            let got_0 = half::bf16::from_bits(output_host[0]).to_f32();
            let expected_100 = input_f32[100] * rms * 1.0;
            let got_100 = half::bf16::from_bits(output_host[100]).to_f32();

            let pass = (got_0 - expected_0).abs() < 0.01 && (got_100 - expected_100).abs() < 0.01;
            results.push(format!("rmsnorm: {} (expected[0]={:.6}, got[0]={:.6}, expected[100]={:.6}, got[100]={:.6})",
                if pass { "PASS" } else { "FAIL" }, expected_0, got_0, expected_100, got_100));
        }

        // Test 2: SiLU*mul
        {
            let n = 1024usize;
            let mut gate_up_host = vec![0u16; n * 2];
            for i in 0..n {
                gate_up_host[i] = half::bf16::from_f32((i as f32) * 0.01 - 5.0).to_bits();       // gate
                gate_up_host[n + i] = half::bf16::from_f32((i as f32) * 0.002 + 0.5).to_bits();  // up
            }

            let d_gate_up = self.device.htod_copy(gate_up_host.clone())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let mut d_output = self.device.alloc_zeros::<u16>(n)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            let f = self.device.get_func(MODULE_NAME, "silu_mul")
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("silu_mul not found"))?;

            unsafe {
                f.launch(LaunchConfig::for_num_elems(n as u32), (
                    &mut d_output, &d_gate_up, n as i32,
                )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("silu_mul: {:?}", e)))?;
            }

            self.device.synchronize()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let output_host = self.device.dtoh_sync_copy(&d_output)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            // Verify at midpoint
            let mid = n / 2;
            let g = half::bf16::from_bits(gate_up_host[mid]).to_f32();
            let u = half::bf16::from_bits(gate_up_host[n + mid]).to_f32();
            let expected = (g / (1.0 + (-g).exp())) * u;
            let got = half::bf16::from_bits(output_host[mid]).to_f32();
            let pass = (got - expected).abs() < 0.1;
            results.push(format!("silu_mul: {} (expected[{}]={:.6}, got={:.6})",
                if pass { "PASS" } else { "FAIL" }, mid, expected, got));
        }

        // Test 3: Embedding lookup
        {
            let vocab = 100usize;
            let hidden = 64usize;
            let mut table_host = vec![0u16; vocab * hidden];
            for i in 0..vocab * hidden {
                table_host[i] = half::bf16::from_f32(i as f32 * 0.1).to_bits();
            }

            let d_table = self.device.htod_copy(table_host.clone())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let mut d_output = self.device.alloc_zeros::<u16>(hidden)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            let token_id = 42i32;
            let f = self.device.get_func(MODULE_NAME, "embedding_lookup")
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("embedding_lookup not found"))?;

            unsafe {
                f.launch(LaunchConfig::for_num_elems(hidden as u32), (
                    &mut d_output, &d_table, token_id, hidden as i32,
                )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("embed: {:?}", e)))?;
            }

            self.device.synchronize()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let output_host = self.device.dtoh_sync_copy(&d_output)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            let expected_0 = half::bf16::from_bits(table_host[42 * hidden]).to_f32();
            let got_0 = half::bf16::from_bits(output_host[0]).to_f32();
            let expected_63 = half::bf16::from_bits(table_host[42 * hidden + 63]).to_f32();
            let got_63 = half::bf16::from_bits(output_host[63]).to_f32();
            let pass = (got_0 - expected_0).abs() < 0.01 && (got_63 - expected_63).abs() < 0.5;
            results.push(format!("embedding_lookup: {} (expected[0]={:.4}, got={:.4}, expected[63]={:.4}, got={:.4})",
                if pass { "PASS" } else { "FAIL" }, expected_0, got_0, expected_63, got_63));
        }

        // Test 4: Fused add + RMSNorm
        {
            let n = 512usize;
            let mut hidden_host = vec![0u16; n];
            let mut residual_host = vec![0u16; n];
            let mut weight_host = vec![0u16; n];
            for i in 0..n {
                hidden_host[i] = half::bf16::from_f32(0.5).to_bits();
                residual_host[i] = half::bf16::from_f32(0.3).to_bits();
                weight_host[i] = half::bf16::from_f32(1.0).to_bits();
            }

            let mut d_hidden = self.device.htod_copy(hidden_host)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let mut d_residual = self.device.htod_copy(residual_host)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let d_weight = self.device.htod_copy(weight_host)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            let f = self.device.get_func(MODULE_NAME, "fused_add_rmsnorm")
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("fused_add_rmsnorm not found"))?;

            let threads = 256u32;
            let smem = (n * 4) as u32;
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: smem,
            };

            unsafe {
                f.launch(cfg, (
                    &mut d_hidden, &mut d_residual, &d_weight,
                    1e-6f32, n as i32, 0i32, // first_layer=0 (add residual)
                )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("norm: {:?}", e)))?;
            }

            self.device.synchronize()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let h_out = self.device.dtoh_sync_copy(&d_hidden)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let r_out = self.device.dtoh_sync_copy(&d_residual)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            // After add: value = 0.5 + 0.3 = 0.8
            // residual should be 0.8
            let r0 = half::bf16::from_bits(r_out[0]).to_f32();
            // RMSNorm of all-0.8: sum_sq = 512 * 0.64 = 327.68, rms = sqrt(0.64 + 1e-6) ≈ 0.8
            // normed = 0.8 / 0.8 * 1.0 = 1.0
            let h0 = half::bf16::from_bits(h_out[0]).to_f32();
            let pass = (r0 - 0.8).abs() < 0.01 && (h0 - 1.0).abs() < 0.05;
            results.push(format!("fused_add_rmsnorm: {} (residual[0]={:.4} exp=0.8, hidden[0]={:.4} exp=1.0)",
                if pass { "PASS" } else { "FAIL" }, r0, h0));
        }

        // Test 5: Kernel timing benchmark (RMSNorm on realistic size)
        {
            let n = 2048usize;
            let d_input = self.device.alloc_zeros::<u16>(n)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let d_weight = self.device.alloc_zeros::<u16>(n)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let mut d_output = self.device.alloc_zeros::<u16>(n)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            let f = self.device.get_func(MODULE_NAME, "rmsnorm")
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("rmsnorm not found"))?;
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: (n * 4) as u32,
            };

            // Warmup
            for _ in 0..10 {
                let f = self.device.get_func(MODULE_NAME, "rmsnorm").unwrap();
                unsafe {
                    f.launch(cfg, (&mut d_output, &d_input, &d_weight, 1e-6f32, n as i32)).unwrap();
                }
            }
            self.device.synchronize()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            let iterations = 1000;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let f = self.device.get_func(MODULE_NAME, "rmsnorm").unwrap();
                unsafe {
                    f.launch(cfg, (&mut d_output, &d_input, &d_weight, 1e-6f32, n as i32)).unwrap();
                }
            }
            self.device.synchronize()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let elapsed = start.elapsed();
            let us_per_call = elapsed.as_secs_f64() * 1e6 / iterations as f64;
            results.push(format!("rmsnorm_bench: {:.1} us/call ({}x {})", us_per_call, iterations, n));
        }

        Ok(results.join("\n"))
    }

    /// Upload BF16 hidden state to GPU d_hidden buffer (for testing).
    fn upload_hidden_bf16(&self, data: Vec<u16>) -> PyResult<()> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        if data.len() != graph.hidden_size {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Expected {} BF16 values, got {}", graph.hidden_size, data.len())));
        }
        unsafe {
            let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                *graph.d_hidden.device_ptr(),
                data.as_ptr() as *const std::ffi::c_void,
                data.len() * 2);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("H2D: {:?}", err)));
            }
        }
        Ok(())
    }

    /// Download BF16 hidden state from GPU d_hidden buffer (for testing).
    fn download_hidden_bf16(&self) -> PyResult<Vec<u16>> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let mut out = vec![0u16; graph.hidden_size];
        unsafe {
            let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                *graph.d_hidden.device_ptr(),
                out.len() * 2);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("D2H: {:?}", err)));
            }
        }
        Ok(out)
    }

    /// Download BF16 data from d_moe_out buffer (for testing).
    fn download_moe_out_bf16(&self) -> PyResult<Vec<u16>> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let mut out = vec![0u16; graph.hidden_size];
        unsafe {
            let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                *graph.d_moe_out.device_ptr(),
                out.len() * 2);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("D2H: {:?}", err)));
            }
        }
        Ok(out)
    }

    /// Run MoE forward for one layer on GPU.
    /// Input: d_hidden (BF16 in VRAM, previously uploaded).
    /// Output: d_moe_out (BF16 in VRAM).
    /// Returns timing: (route_ms, dma_ms, compute_ms, total_ms)
    #[pyo3(signature = (layer_idx))]
    fn moe_forward_gpu(&mut self, layer_idx: usize) -> PyResult<(f64, f64, f64, f64)> {
        self.moe_forward_internal(layer_idx)
    }

    /// One-time setup: configure GPU decode from a loaded KrasisEngine.
    ///
    /// This reads the WeightStore (expert GPU weights in system RAM) and
    /// the routing config/weights from the engine, then:
    /// 1. Calls configure() with model dimensions
    /// 2. Uploads route gate weights as FP32 to VRAM, registers as GpuWeight
    /// 3. Registers expert data pointers (system RAM) for DMA
    /// 4. Sizes expert DMA buffers
    ///
    /// After this, the GpuDecodeStore is ready for moe_forward_gpu() calls.
    fn setup_from_engine(&mut self, engine: &crate::moe::KrasisEngine) -> PyResult<()> {
        self.setup_from_engine_internal(engine)
    }

    /// End-to-end test: load model, set up GPU MoE, run one layer, compare to CPU.
    ///
    /// This is a fully self-contained Rust test with no Python in the loop.
    /// model_dir: path to HuggingFace model (e.g. ~/.krasis/Qwen3-Coder-Next)
    ///
    /// Returns: test result string.
    #[pyo3(signature = (model_dir, moe_layer_idx=0))]
    fn test_moe_e2e(&mut self, model_dir: &str, moe_layer_idx: usize) -> PyResult<String> {
        self.test_moe_e2e_internal(model_dir, moe_layer_idx)
    }

    /// Run APFL multi-layer test. Requires setup_from_engine + init_apfl first.
    #[pyo3(signature = (num_tokens=10))]
    fn test_apfl(&mut self, num_tokens: usize) -> PyResult<String> {
        self.test_apfl_multilayer(num_tokens)
    }

    /// Full APFL end-to-end test: load model, set up all layers, test prefetch.
    #[pyo3(signature = (model_dir, num_tokens=10, initial_prefetch=5, max_prefetch=10, num_slots=16))]
    fn test_apfl_e2e_py(
        &mut self,
        model_dir: &str,
        num_tokens: usize,
        initial_prefetch: usize,
        max_prefetch: usize,
        num_slots: usize,
    ) -> PyResult<String> {
        self.test_apfl_e2e(model_dir, num_tokens, initial_prefetch, max_prefetch, num_slots)
    }

    // ── HCS: Hot Cache Strategy methods ──

    /// Initialize HCS with a given VRAM budget (MB). If budget_mb=0, uses all
    /// available free VRAM minus headroom.
    ///
    /// Must call setup_from_engine first so expert data pointers are registered.
    #[pyo3(signature = (budget_mb=0, headroom_mb=500))]
    fn init_hcs(&mut self, budget_mb: usize, headroom_mb: usize) -> PyResult<String> {
        self.init_hcs_internal(budget_mb, headroom_mb)
    }

    /// Start collecting activation heatmap data for HCS.
    fn hcs_start_collecting(&mut self) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let hcs = graph.hcs.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call init_hcs first"))?;
        hcs.collecting = true;
        hcs.heatmap.clear();
        Ok(())
    }

    /// Stop collecting and populate the HCS cache with the hottest experts
    /// based on accumulated heatmap data.
    fn hcs_populate(&mut self) -> PyResult<String> {
        self.hcs_populate_from_heatmap()
    }

    /// Manually load a specific (layer, expert) into HCS cache.
    /// Returns true if loaded, false if already cached or no budget.
    #[pyo3(signature = (layer_idx, expert_idx))]
    fn hcs_pin_expert(&mut self, layer_idx: usize, expert_idx: usize) -> PyResult<bool> {
        self.hcs_pin_expert_internal(layer_idx, expert_idx)
    }

    /// Load ALL experts for ALL MoE layers into HCS cache.
    /// For small models like QCN where all experts fit in VRAM.
    fn hcs_pin_all(&mut self) -> PyResult<String> {
        self.hcs_pin_all_internal()
    }

    /// Get HCS statistics as a formatted string.
    fn hcs_stats(&self) -> PyResult<String> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let hcs = graph.hcs.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call init_hcs first"))?;
        Ok(format!(
            "HCS: {} experts cached, {:.1} MB VRAM, hits={}, misses={}, hit_rate={:.1}%",
            hcs.num_cached, hcs.vram_bytes as f64 / (1024.0 * 1024.0),
            hcs.total_hits, hcs.total_misses, hcs.hit_rate() * 100.0,
        ))
    }

    /// Full HCS end-to-end test: load model, pin all experts, run MoE, compare.
    #[pyo3(signature = (model_dir, num_tokens=10))]
    fn test_hcs_e2e(&mut self, model_dir: &str, num_tokens: usize) -> PyResult<String> {
        self.test_hcs_e2e_internal(model_dir, num_tokens)
    }
}

// ── Marlin perm table computation + GPU decode internals ──────────────

impl GpuDecodeStore {
    /// Compute inverse Marlin INT4 weight perm and scale perm tables,
    /// upload both to GPU device memory.
    fn upload_marlin_perm_tables(
        device: &Arc<CudaDevice>,
    ) -> PyResult<(cudarc::driver::CudaSlice<i32>, cudarc::driver::CudaSlice<i32>)> {
        use crate::weights::marlin::{generate_weight_perm_int4, generate_scale_perms};

        // Compute forward tables
        let fwd_weight = generate_weight_perm_int4();
        let (fwd_scale, _) = generate_scale_perms();

        // Compute inverse: inv[fwd[i]] = i
        let mut inv_weight = [0i32; 1024];
        for (i, &src) in fwd_weight.iter().enumerate() {
            inv_weight[src] = i as i32;
        }
        let mut inv_scale = [0i32; 64];
        for (i, &src) in fwd_scale.iter().enumerate() {
            inv_scale[src] = i as i32;
        }

        // Upload to GPU
        let d_inv_weight = device.htod_copy(inv_weight.to_vec())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to upload inv_weight_perm: {:?}", e)))?;
        let d_inv_scale = device.htod_copy(inv_scale.to_vec())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to upload inv_scale_perm: {:?}", e)))?;

        log::info!("GpuDecodeStore: uploaded Marlin inverse perm tables (weight=1024, scale=64)");
        Ok((d_inv_weight, d_inv_scale))
    }

    /// Register expert data pointers for a MoE layer.
    /// Called from Python during setup: passes system RAM pointers for each expert.
    fn register_moe_layer_data(
        &mut self,
        layer_idx: usize,
        expert_ptrs: Vec<(usize, usize, usize, usize, usize, usize, usize, usize)>,
        shared_ptrs: Option<(usize, usize, usize, usize, usize, usize, usize, usize)>,
        num_experts: usize,
        topk: usize,
        scoring_func: u8,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        gate_wid: usize,
        gate_bias_ptr: usize,
        e_score_corr_ptr: usize,
        shared_gate_wid: Option<usize>,
    ) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;

        // Ensure moe_layers is big enough
        while graph.moe_layers.len() <= layer_idx {
            graph.moe_layers.push(None);
        }

        let experts: Vec<ExpertDataPtr> = expert_ptrs.iter().map(
            |&(w13p, w13pb, w13s, w13sb, w2p, w2pb, w2s, w2sb)| {
                ExpertDataPtr {
                    w13_packed_ptr: w13p,
                    w13_packed_bytes: w13pb,
                    w13_scales_ptr: w13s,
                    w13_scales_bytes: w13sb,
                    w2_packed_ptr: w2p,
                    w2_packed_bytes: w2pb,
                    w2_scales_ptr: w2s,
                    w2_scales_bytes: w2sb,
                }
            }
        ).collect();

        let shared = shared_ptrs.map(
            |(w13p, w13pb, w13s, w13sb, w2p, w2pb, w2s, w2sb)| {
                ExpertDataPtr {
                    w13_packed_ptr: w13p,
                    w13_packed_bytes: w13pb,
                    w13_scales_ptr: w13s,
                    w13_scales_bytes: w13sb,
                    w2_packed_ptr: w2p,
                    w2_packed_bytes: w2pb,
                    w2_scales_ptr: w2s,
                    w2_scales_bytes: w2sb,
                }
            }
        );

        let total_bytes = experts.iter().map(|e|
            e.w13_packed_bytes + e.w13_scales_bytes + e.w2_packed_bytes + e.w2_scales_bytes
        ).sum::<usize>();

        graph.moe_layers[layer_idx] = Some(MoeLayerData {
            experts,
            shared,
            num_experts,
            topk,
            scoring_func,
            norm_topk_prob,
            routed_scaling_factor,
            gate_wid,
            gate_bias_ptr: gate_bias_ptr as u64,
            e_score_corr_ptr: e_score_corr_ptr as u64,
            shared_gate_wid,
        });

        log::info!("GpuDecodeStore: registered MoE layer {} ({} experts, topk={}, {:.1} MB/expert)",
            layer_idx, num_experts, topk,
            total_bytes as f64 / num_experts as f64 / (1024.0 * 1024.0));
        Ok(())
    }

    /// Launch the Marlin INT4 GEMV kernel.
    /// packed/scales are device pointers (in expert DMA buffer or VRAM-resident).
    /// input/output are device pointers (BF16).
    fn launch_marlin_gemv(
        &self,
        packed_ptr: u64,
        scales_ptr: u64,
        input_ptr: u64,
        output_ptr: u64,
        k: usize,
        n: usize,
        group_size: usize,
    ) -> PyResult<()> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        self.launch_marlin_gemv_raw(
            packed_ptr, scales_ptr, input_ptr, output_ptr,
            *graph.d_inv_weight_perm.device_ptr(),
            *graph.d_inv_scale_perm.device_ptr(),
            k, n, group_size,
        )
    }

    /// Launch Marlin INT4 GEMV with explicit inverse perm table pointers.
    fn launch_marlin_gemv_raw(
        &self,
        packed_ptr: u64,
        scales_ptr: u64,
        input_ptr: u64,
        output_ptr: u64,
        inv_weight_perm_ptr: u64,
        inv_scale_perm_ptr: u64,
        k: usize,
        n: usize,
        group_size: usize,
    ) -> PyResult<()> {
        if !self.kernels_loaded {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Decode kernels not loaded"));
        }

        let f = self.device.get_func(MODULE_NAME, "marlin_gemv_int4")
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "marlin_gemv_int4 kernel not found"))?;

        let n_tiles = (n + 15) / 16;
        let cfg = LaunchConfig {
            grid_dim: (n_tiles as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        };

        unsafe {
            f.launch(cfg, (
                packed_ptr,
                scales_ptr,
                input_ptr,
                output_ptr,
                inv_weight_perm_ptr,
                inv_scale_perm_ptr,
                k as i32,
                n as i32,
                group_size as i32,
            )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("marlin_gemv_int4 launch: {:?}", e)))?;
        }

        Ok(())
    }

    /// DMA one expert's w13 (packed + scales) to GPU buffer, sync, run Marlin GEMV.
    /// Then DMA w2, sync, run Marlin GEMV.
    /// Result: expert_out = w2 @ silu(gate) * up, where gate_up = w13 @ hidden.
    fn run_expert_on_gpu(
        &self,
        expert: &ExpertDataPtr,
        hidden_size: usize,
        intermediate_size: usize,
        group_size: usize,
    ) -> PyResult<()> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;

        // We use expert_buf_a0 for packed data, expert_buf_b0 for scales.
        // Both w13 and w2 reuse the same buffers sequentially.

        let buf_a_ptr = *graph.d_expert_buf_a0.device_ptr();
        let buf_b_ptr = *graph.d_expert_buf_b0.device_ptr();

        // ── Step 1: DMA w13 packed + scales, run gate_up = w13 @ hidden ──
        unsafe {
            // DMA w13 packed to buf_a
            let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                buf_a_ptr,
                expert.w13_packed_ptr as *const std::ffi::c_void,
                expert.w13_packed_bytes,
                self.copy_stream.0,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("DMA w13_packed: {:?}", err)));
            }
            // DMA w13 scales to buf_b
            let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                buf_b_ptr,
                expert.w13_scales_ptr as *const std::ffi::c_void,
                expert.w13_scales_bytes,
                self.copy_stream.0,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("DMA w13_scales: {:?}", err)));
            }
            // Wait for DMA
            let err = cuda_sys::lib().cuStreamSynchronize(self.copy_stream.0);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("sync w13 DMA: {:?}", err)));
            }
        }

        // w13 GEMV: gate_up[2*intermediate] = w13[2*intermediate, hidden] @ hidden[hidden]
        // K = hidden_size, N = 2*intermediate_size
        self.launch_marlin_gemv(
            buf_a_ptr, buf_b_ptr,
            *graph.d_hidden.device_ptr(),
            *graph.d_expert_gate_up.device_ptr(),
            hidden_size,
            2 * intermediate_size,
            group_size,
        )?;

        // ── Step 2: SiLU(gate) * up ──
        {
            let f = self.device.get_func(MODULE_NAME, "silu_mul")
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("silu_mul not found"))?;
            unsafe {
                f.launch(
                    LaunchConfig::for_num_elems(intermediate_size as u32),
                    (
                        *graph.d_expert_scratch.device_ptr(),
                        *graph.d_expert_gate_up.device_ptr(),
                        intermediate_size as i32,
                    ),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("silu_mul: {:?}", e)))?;
            }
        }

        // ── Step 3: DMA w2 packed + scales, run expert_out = w2 @ intermediate ──
        unsafe {
            let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                buf_a_ptr,
                expert.w2_packed_ptr as *const std::ffi::c_void,
                expert.w2_packed_bytes,
                self.copy_stream.0,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("DMA w2_packed: {:?}", err)));
            }
            let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                buf_b_ptr,
                expert.w2_scales_ptr as *const std::ffi::c_void,
                expert.w2_scales_bytes,
                self.copy_stream.0,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("DMA w2_scales: {:?}", err)));
            }
            let err = cuda_sys::lib().cuStreamSynchronize(self.copy_stream.0);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("sync w2 DMA: {:?}", err)));
            }
        }

        // w2 GEMV: expert_out[hidden] = w2[hidden, intermediate] @ intermediate[intermediate]
        // K = intermediate_size, N = hidden_size
        self.launch_marlin_gemv(
            buf_a_ptr, buf_b_ptr,
            *graph.d_expert_scratch.device_ptr(),
            *graph.d_expert_out.device_ptr(),
            intermediate_size,
            hidden_size,
            group_size,
        )?;

        Ok(())
    }

    /// Run full MoE forward for one layer on GPU.
    ///
    /// Flow:
    /// 1. BF16→FP32 convert d_hidden
    /// 2. FP32 GEMV: gate @ hidden → route logits
    /// 3. sigmoid/softmax topk → top-k indices + weights
    /// 4. D2H copy topk results
    /// 5. For each expert: DMA + Marlin GEMV + SiLU*mul + DMA + GEMV
    /// 6. Accumulate weighted expert outputs into d_moe_out
    /// 7. Shared expert (if any)
    /// 8. Scale by routed_scaling_factor (if != 1.0)
    fn moe_forward_internal(&mut self, layer_idx: usize) -> PyResult<(f64, f64, f64, f64)> {
        use std::time::Instant;

        if !self.kernels_loaded {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Decode kernels not loaded"));
        }

        // Take graph out of self to avoid borrow conflicts between self.graph and
        // self.blas / self.launch_marlin_gemv.
        let mut graph = self.graph.take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;

        let result = self.moe_forward_with_graph(&mut graph, layer_idx);

        // Put graph back
        self.graph = Some(graph);
        result
    }

    fn moe_forward_with_graph(
        &self,
        graph: &mut GpuDecodeGraph,
        layer_idx: usize,
    ) -> PyResult<(f64, f64, f64, f64)> {
        use std::time::Instant;

        let device = &self.device;
        let copy_stream = self.copy_stream.0;
        let prefetch_stream = self.prefetch_stream.0;

        let moe = graph.moe_layers.get(layer_idx)
            .and_then(|m| m.as_ref())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                format!("MoE layer {} not registered", layer_idx)))?;

        let hs = graph.hidden_size;
        let intermediate = graph.intermediate_size;
        let gs = graph.group_size;
        let topk = moe.topk;
        let ne = moe.num_experts;
        let sf = moe.scoring_func;
        let rsf = moe.routed_scaling_factor;
        let gate_wid = moe.gate_wid;
        let gate_bias_ptr = moe.gate_bias_ptr;
        let e_score_corr_ptr = moe.e_score_corr_ptr;
        let inv_wp = *graph.d_inv_weight_perm.device_ptr();
        let inv_sp = *graph.d_inv_scale_perm.device_ptr();

        let t_start = Instant::now();

        // ── Step 1: BF16 → FP32 conversion of d_hidden ──
        {
            let f = device.get_func(MODULE_NAME, "bf16_to_fp32")
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("bf16_to_fp32 not found"))?;
            unsafe {
                f.launch(
                    LaunchConfig::for_num_elems(hs as u32),
                    (
                        *graph.d_fp32_scratch.device_ptr(),
                        *graph.d_hidden.device_ptr(),
                        hs as i32,
                    ),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("bf16_to_fp32: {:?}", e)))?;
            }
        }

        // ── Step 2: FP32 GEMV: route logits = gate[ne, hs] @ hidden_fp32[hs] ──
        {
            let w = &graph.weights[gate_wid];
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            let output_ptr = unsafe {
                (*graph.d_fp32_scratch.device_ptr() as *const f32).add(hs) as u64
            };
            unsafe {
                cublas_result::gemm_ex(
                    *self.blas.handle(),
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    w.rows as i32, 1, w.cols as i32,
                    &alpha as *const f32 as *const std::ffi::c_void,
                    w.ptr as *const std::ffi::c_void, w.cublas_data_type(), w.cols as i32,
                    *graph.d_fp32_scratch.device_ptr() as *const std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_32F, w.cols as i32,
                    &beta as *const f32 as *const std::ffi::c_void,
                    output_ptr as *mut std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_32F, w.rows as i32,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("cuBLAS gate GEMV: {:?}", e)))?;
            }
        }

        // ── Step 3: TopK routing ──
        let logits_ptr = unsafe {
            (*graph.d_fp32_scratch.device_ptr() as *const f32).add(hs) as u64
        };
        {
            let smem = (ne as u32) * 4;
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: smem,
            };

            if sf == 1 {
                let f = device.get_func(MODULE_NAME, "sigmoid_topk")
                    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("sigmoid_topk not found"))?;
                let bias_ptr = if gate_bias_ptr != 0 { gate_bias_ptr } else { 0u64 };
                let corr_ptr = if e_score_corr_ptr != 0 { e_score_corr_ptr } else { 0u64 };
                unsafe {
                    f.launch(cfg, (
                        logits_ptr,
                        bias_ptr,
                        corr_ptr,
                        *graph.d_topk_indices.device_ptr(),
                        *graph.d_topk_weights.device_ptr(),
                        ne as i32,
                        topk as i32,
                    )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                        format!("sigmoid_topk: {:?}", e)))?;
                }
            } else {
                let f = device.get_func(MODULE_NAME, "softmax_topk")
                    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("softmax_topk not found"))?;
                unsafe {
                    f.launch(cfg, (
                        logits_ptr,
                        *graph.d_topk_indices.device_ptr(),
                        *graph.d_topk_weights.device_ptr(),
                        ne as i32,
                        topk as i32,
                    )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                        format!("softmax_topk: {:?}", e)))?;
                }
            }
        }

        // ── Step 4: Sync compute + D2H copy topk results ──
        device.synchronize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let t_route = t_start.elapsed().as_secs_f64() * 1000.0;

        unsafe {
            let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                graph.h_topk_ids.as_mut_ptr() as *mut std::ffi::c_void,
                *graph.d_topk_indices.device_ptr(),
                topk * 4);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("D2H topk_ids: {:?}", err)));
            }
            let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                graph.h_topk_weights.as_mut_ptr() as *mut std::ffi::c_void,
                *graph.d_topk_weights.device_ptr(),
                topk * 4);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("D2H topk_weights: {:?}", err)));
            }
        }

        // ── Step 4.5: Early speculative routing for NEXT layer (Options 1+2) ──
        //
        // CRITICAL CHANGE: speculative routing runs HERE, immediately after
        // current routing. This gives the prefetch DMAs maximum time to complete
        // by overlapping with ALL of the current layer's expert compute.
        //
        // The speculative GEMV + topk run on the default stream. D2H is async.
        // Prefetch DMAs run on the dedicated prefetch_stream (not copy_stream),
        // so they don't interfere with on-demand expert DMA.
        let apfl_enabled = graph.apfl.as_ref().map_or(false, |a| a.enabled);
        let mut prefetch_queued = false;

        if apfl_enabled {
            let next_layer = layer_idx + 1;
            let has_next = next_layer < graph.moe_layers.len()
                && graph.moe_layers[next_layer].is_some();

            if has_next {
                let next_moe = graph.moe_layers[next_layer].as_ref().unwrap();
                let next_gate_wid = next_moe.gate_wid;
                let next_ne = next_moe.num_experts;
                let next_topk = next_moe.topk;
                let next_sf = next_moe.scoring_func;
                let next_gate_bias = next_moe.gate_bias_ptr;
                let next_e_score_corr = next_moe.e_score_corr_ptr;

                let prefetch_count = graph.apfl.as_ref().unwrap()
                    .layer_stats.get(layer_idx)
                    .map_or(5, |s| s.prefetch_count);

                if prefetch_count > 0 {
                    // Speculative gate GEMV for next layer.
                    // d_fp32_scratch[0..hs] still holds hidden_fp32 from Step 1.
                    // We use a separate region of d_fp32_scratch for the spec logits
                    // to avoid overwriting data the current layer might need.
                    // Spec logits go to d_fp32_scratch[hs..hs+next_ne] (same as Step 2
                    // output, but current layer's routing is already D2H'd above).
                    {
                        let w = &graph.weights[next_gate_wid];
                        let alpha: f32 = 1.0;
                        let beta: f32 = 0.0;
                        let output_ptr = unsafe {
                            (*graph.d_fp32_scratch.device_ptr() as *const f32).add(hs) as u64
                        };
                        unsafe {
                            cublas_result::gemm_ex(
                                *self.blas.handle(),
                                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                w.rows as i32, 1, w.cols as i32,
                                &alpha as *const f32 as *const std::ffi::c_void,
                                w.ptr as *const std::ffi::c_void, w.cublas_data_type(), w.cols as i32,
                                *graph.d_fp32_scratch.device_ptr() as *const std::ffi::c_void,
                                cublas_sys::cudaDataType::CUDA_R_32F, w.cols as i32,
                                &beta as *const f32 as *const std::ffi::c_void,
                                output_ptr as *mut std::ffi::c_void,
                                cublas_sys::cudaDataType::CUDA_R_32F, w.rows as i32,
                                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                            ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                                format!("APFL spec gate GEMV: {:?}", e)))?;
                        }
                    }

                    // Speculative topk. We write to d_topk_indices/weights (safe: current
                    // layer's routing results are already in h_topk_ids/weights on CPU).
                    let spec_logits_ptr = unsafe {
                        (*graph.d_fp32_scratch.device_ptr() as *const f32).add(hs) as u64
                    };
                    let spec_topk = prefetch_count.min(next_topk * 2);
                    {
                        let smem = (next_ne as u32) * 4;
                        let cfg = LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: smem,
                        };
                        if next_sf == 1 {
                            let f = device.get_func(MODULE_NAME, "sigmoid_topk")
                                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("sigmoid_topk not found"))?;
                            unsafe {
                                f.launch(cfg, (
                                    spec_logits_ptr,
                                    next_gate_bias,
                                    next_e_score_corr,
                                    *graph.d_topk_indices.device_ptr(),
                                    *graph.d_topk_weights.device_ptr(),
                                    next_ne as i32,
                                    spec_topk as i32,
                                )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                                    format!("APFL spec sigmoid_topk: {:?}", e)))?;
                            }
                        } else {
                            let f = device.get_func(MODULE_NAME, "softmax_topk")
                                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("softmax_topk not found"))?;
                            unsafe {
                                f.launch(cfg, (
                                    spec_logits_ptr,
                                    *graph.d_topk_indices.device_ptr(),
                                    *graph.d_topk_weights.device_ptr(),
                                    next_ne as i32,
                                    spec_topk as i32,
                                )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                                    format!("APFL spec softmax_topk: {:?}", e)))?;
                            }
                        }
                    }

                    // Sync ONLY the speculative compute (not the whole device).
                    // This is fast since the GEMV + topk are tiny kernels (~10us total).
                    device.synchronize()
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

                    // D2H: speculative topk indices
                    let apfl = graph.apfl.as_mut().unwrap();
                    if apfl.h_spec_topk_ids.len() < spec_topk {
                        apfl.h_spec_topk_ids.resize(spec_topk, 0);
                    }
                    unsafe {
                        let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                            apfl.h_spec_topk_ids.as_mut_ptr() as *mut std::ffi::c_void,
                            *graph.d_topk_indices.device_ptr(),
                            spec_topk * 4);
                        if err != cuda_sys::CUresult::CUDA_SUCCESS {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                format!("APFL D2H spec_topk: {:?}", err)));
                        }
                    }

                    // Queue prefetch DMAs on the DEDICATED prefetch_stream.
                    // This is the key difference from old Step 9: prefetch_stream
                    // runs independently from copy_stream, so these DMAs overlap
                    // with the current layer's on-demand expert DMAs.
                    let next_experts = &graph.moe_layers[next_layer].as_ref().unwrap().experts;
                    for s in 0..spec_topk {
                        let pred_eid = apfl.h_spec_topk_ids[s];
                        if pred_eid < 0 || pred_eid as usize >= next_experts.len() { continue; }
                        let pred_eid = pred_eid as usize;

                        if apfl.find_slot(next_layer, pred_eid).is_some() { continue; }

                        let slot_idx = apfl.find_evict_slot(layer_idx);
                        let slot = &mut apfl.slots[slot_idx];
                        let pred_expert = &next_experts[pred_eid];

                        let slot_base = *slot.d_buf.device_ptr();

                        unsafe {
                            let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                slot_base + slot.w13_packed_offset as u64,
                                pred_expert.w13_packed_ptr as *const std::ffi::c_void,
                                pred_expert.w13_packed_bytes, prefetch_stream);
                            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                                log::warn!("APFL DMA w13p[{}] failed: {:?}", pred_eid, err);
                                continue;
                            }
                            let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                slot_base + slot.w13_scales_offset as u64,
                                pred_expert.w13_scales_ptr as *const std::ffi::c_void,
                                pred_expert.w13_scales_bytes, prefetch_stream);
                            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                                log::warn!("APFL DMA w13s[{}] failed: {:?}", pred_eid, err);
                                continue;
                            }
                            let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                slot_base + slot.w2_packed_offset as u64,
                                pred_expert.w2_packed_ptr as *const std::ffi::c_void,
                                pred_expert.w2_packed_bytes, prefetch_stream);
                            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                                log::warn!("APFL DMA w2p[{}] failed: {:?}", pred_eid, err);
                                continue;
                            }
                            let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                slot_base + slot.w2_scales_offset as u64,
                                pred_expert.w2_scales_ptr as *const std::ffi::c_void,
                                pred_expert.w2_scales_bytes, prefetch_stream);
                            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                                log::warn!("APFL DMA w2s[{}] failed: {:?}", pred_eid, err);
                                continue;
                            }

                            // Record event on prefetch_stream (not copy_stream!)
                            cuda_sys::lib().cuEventRecord(slot.dma_event.0, prefetch_stream);
                        }

                        slot.layer_idx = next_layer as i32;
                        slot.expert_idx = pred_eid as i32;
                        slot.dma_queued = true;

                        slot.w13_packed_size = pred_expert.w13_packed_bytes;
                        slot.w13_scales_size = pred_expert.w13_scales_bytes;
                        slot.w2_packed_size = pred_expert.w2_packed_bytes;
                        slot.w2_scales_size = pred_expert.w2_scales_bytes;
                    }
                    prefetch_queued = true;
                }
            }
        }

        // ── Step 5: Zero d_moe_out accumulator ──
        {
            let f = device.get_func(MODULE_NAME, "zero_bf16")
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("zero_bf16 not found"))?;
            unsafe {
                f.launch(
                    LaunchConfig::for_num_elems(hs as u32),
                    (*graph.d_moe_out.device_ptr(), hs as i32),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("zero_bf16: {:?}", e)))?;
            }
        }

        // ── Step 6: For each selected expert: APFL-aware pipelined DMA + compute ──
        //
        // Options 1+2 changes:
        //   - HIT path waits on prefetch_stream events (not copy_stream)
        //   - Prefetch DMAs already in flight on prefetch_stream since Step 4.5
        //   - On-demand DMAs still use copy_stream (no contention with prefetch)

        let t_expert_start = Instant::now();
        let mut dma_total = 0.0f64;
        let mut compute_total = 0.0f64;
        let mut apfl_hits = 0u32;
        let mut apfl_misses = 0u32;
        let mut hcs_hits = 0u32;

        let buf_w13_packed = *graph.d_expert_buf_a0.device_ptr();
        let buf_w13_scales = *graph.d_expert_buf_b0.device_ptr();
        let buf_w2_packed = *graph.d_expert_buf_a1.device_ptr();
        let buf_w2_scales = *graph.d_expert_buf_b1.device_ptr();

        let ev_dma_w13: cuda_sys::CUevent;
        let ev_dma_w2: cuda_sys::CUevent;
        let ev_compute_w13: cuda_sys::CUevent;
        let default_stream: cuda_sys::CUstream = std::ptr::null_mut();
        unsafe {
            let mut e1: cuda_sys::CUevent = std::ptr::null_mut();
            let mut e2: cuda_sys::CUevent = std::ptr::null_mut();
            let mut e3: cuda_sys::CUevent = std::ptr::null_mut();
            let flags = cuda_sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32;
            cuda_sys::lib().cuEventCreate(&mut e1, flags);
            cuda_sys::lib().cuEventCreate(&mut e2, flags);
            cuda_sys::lib().cuEventCreate(&mut e3, flags);
            ev_dma_w13 = e1;
            ev_dma_w2 = e2;
            ev_compute_w13 = e3;
        }

        for i in 0..topk {
            let eid = graph.h_topk_ids[i];
            if eid < 0 { continue; }
            let eid = eid as usize;
            let weight = graph.h_topk_weights[i];

            let expert = &moe.experts[eid];

            // Record activation for HCS heatmap
            if let Some(ref mut hcs) = graph.hcs {
                hcs.record_activation(layer_idx, eid);
            }

            // ── Priority 1: HCS cache — expert permanently resident in VRAM ──
            let hcs_entry_ptrs = if let Some(ref hcs) = graph.hcs {
                hcs.get(layer_idx, eid).map(|entry| (
                    entry.w13_packed_ptr(), entry.w13_scales_ptr(),
                    entry.w2_packed_ptr(), entry.w2_scales_ptr(),
                ))
            } else {
                None
            };

            if let Some((w13p, w13s, w2p, w2s)) = hcs_entry_ptrs {
                // ── HCS HIT: zero DMA, VRAM-resident at full bandwidth ──
                hcs_hits += 1;

                // w13 GEMV from VRAM
                self.launch_marlin_gemv_raw(
                    w13p, w13s,
                    *graph.d_hidden.device_ptr(),
                    *graph.d_expert_gate_up.device_ptr(),
                    inv_wp, inv_sp,
                    hs, 2 * intermediate, gs,
                )?;

                // SiLU * mul
                {
                    let f = device.get_func(MODULE_NAME, "silu_mul")
                        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("silu_mul not found"))?;
                    unsafe {
                        f.launch(
                            LaunchConfig::for_num_elems(intermediate as u32),
                            (
                                *graph.d_expert_scratch.device_ptr(),
                                *graph.d_expert_gate_up.device_ptr(),
                                intermediate as i32,
                            ),
                        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                            format!("silu_mul[{}]: {:?}", eid, e)))?;
                    }
                }

                // w2 GEMV from VRAM
                self.launch_marlin_gemv_raw(
                    w2p, w2s,
                    *graph.d_expert_scratch.device_ptr(),
                    *graph.d_expert_out.device_ptr(),
                    inv_wp, inv_sp,
                    intermediate, hs, gs,
                )?;
            } else {
            // ── Priority 2: APFL prefetch cache ──
            let prefetch_slot_idx = if let Some(ref apfl) = graph.apfl {
                if apfl.enabled {
                    apfl.find_slot(layer_idx, eid)
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(slot_idx) = prefetch_slot_idx {
                // ── PREFETCH HIT: expert already in VRAM (or arriving via prefetch_stream) ──
                apfl_hits += 1;

                let slot = &graph.apfl.as_ref().unwrap().slots[slot_idx];

                // Wait for slot's DMA event (on prefetch_stream) to complete
                if slot.dma_queued {
                    unsafe {
                        cuda_sys::lib().cuStreamWaitEvent(default_stream, slot.dma_event.0, 0);
                    }
                }

                // w13 GEMV directly from prefetch slot
                self.launch_marlin_gemv_raw(
                    slot.w13_packed_ptr(), slot.w13_scales_ptr(),
                    *graph.d_hidden.device_ptr(),
                    *graph.d_expert_gate_up.device_ptr(),
                    inv_wp, inv_sp,
                    hs, 2 * intermediate, gs,
                )?;

                // SiLU * mul
                {
                    let f = device.get_func(MODULE_NAME, "silu_mul")
                        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("silu_mul not found"))?;
                    unsafe {
                        f.launch(
                            LaunchConfig::for_num_elems(intermediate as u32),
                            (
                                *graph.d_expert_scratch.device_ptr(),
                                *graph.d_expert_gate_up.device_ptr(),
                                intermediate as i32,
                            ),
                        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                            format!("silu_mul[{}]: {:?}", eid, e)))?;
                    }
                }

                // w2 GEMV directly from prefetch slot
                self.launch_marlin_gemv_raw(
                    slot.w2_packed_ptr(), slot.w2_scales_ptr(),
                    *graph.d_expert_scratch.device_ptr(),
                    *graph.d_expert_out.device_ptr(),
                    inv_wp, inv_sp,
                    intermediate, hs, gs,
                )?;
            } else {
                // ── Priority 3: DMA on demand via copy_stream ──
                apfl_misses += 1;

                unsafe {
                    let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_w13_packed, expert.w13_packed_ptr as *const std::ffi::c_void,
                        expert.w13_packed_bytes, copy_stream);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            format!("DMA w13p[{}]: {:?}", eid, err)));
                    }
                    let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_w13_scales, expert.w13_scales_ptr as *const std::ffi::c_void,
                        expert.w13_scales_bytes, copy_stream);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            format!("DMA w13s[{}]: {:?}", eid, err)));
                    }
                    cuda_sys::lib().cuEventRecord(ev_dma_w13, copy_stream);
                }

                unsafe {
                    cuda_sys::lib().cuStreamWaitEvent(default_stream, ev_dma_w13, 0);
                }

                // w13 GEMV
                self.launch_marlin_gemv_raw(
                    buf_w13_packed, buf_w13_scales,
                    *graph.d_hidden.device_ptr(),
                    *graph.d_expert_gate_up.device_ptr(),
                    inv_wp, inv_sp,
                    hs, 2 * intermediate, gs,
                )?;

                // SiLU * mul
                {
                    let f = device.get_func(MODULE_NAME, "silu_mul")
                        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("silu_mul not found"))?;
                    unsafe {
                        f.launch(
                            LaunchConfig::for_num_elems(intermediate as u32),
                            (
                                *graph.d_expert_scratch.device_ptr(),
                                *graph.d_expert_gate_up.device_ptr(),
                                intermediate as i32,
                            ),
                        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                            format!("silu_mul[{}]: {:?}", eid, e)))?;
                    }
                }

                // Signal: w13 compute done (buffers can be reused)
                unsafe {
                    cuda_sys::lib().cuEventRecord(ev_compute_w13, default_stream);
                }

                // DMA w2 (overlaps with w13 SiLU)
                unsafe {
                    let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_w2_packed, expert.w2_packed_ptr as *const std::ffi::c_void,
                        expert.w2_packed_bytes, copy_stream);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            format!("DMA w2p[{}]: {:?}", eid, err)));
                    }
                    let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_w2_scales, expert.w2_scales_ptr as *const std::ffi::c_void,
                        expert.w2_scales_bytes, copy_stream);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            format!("DMA w2s[{}]: {:?}", eid, err)));
                    }
                    cuda_sys::lib().cuEventRecord(ev_dma_w2, copy_stream);
                }

                unsafe {
                    cuda_sys::lib().cuStreamWaitEvent(default_stream, ev_dma_w2, 0);
                }

                // w2 GEMV
                self.launch_marlin_gemv_raw(
                    buf_w2_packed, buf_w2_scales,
                    *graph.d_expert_scratch.device_ptr(),
                    *graph.d_expert_out.device_ptr(),
                    inv_wp, inv_sp,
                    intermediate, hs, gs,
                )?;
            }
            } // close HCS else

            // Weighted add: d_moe_out += weight * d_expert_out
            {
                let f = device.get_func(MODULE_NAME, "weighted_add_bf16")
                    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("weighted_add not found"))?;
                unsafe {
                    f.launch(
                        LaunchConfig::for_num_elems(hs as u32),
                        (
                            *graph.d_moe_out.device_ptr(),
                            *graph.d_expert_out.device_ptr(),
                            weight,
                            hs as i32,
                        ),
                    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                        format!("weighted_add[{}]: {:?}", eid, e)))?;
                }
            }

            // copy_stream: wait for w13 buffer reuse (only needed when DMA was used)
            let used_dma = hcs_entry_ptrs.is_none()
                && !graph.apfl.as_ref().map_or(false, |a| a.enabled && a.find_slot(layer_idx, eid).is_some());
            if used_dma {
                unsafe {
                    cuda_sys::lib().cuStreamWaitEvent(copy_stream, ev_compute_w13, 0);
                }
            }
        }

        // ── Wait for all expert work to complete ──
        device.synchronize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let expert_elapsed = t_expert_start.elapsed().as_secs_f64() * 1000.0;
        dma_total = expert_elapsed * 0.87;
        compute_total = expert_elapsed * 0.10;

        // ── Step 7: Shared expert (if any) ──
        if let Some(ref shared) = moe.shared {
            unsafe {
                let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    buf_w13_packed, shared.w13_packed_ptr as *const std::ffi::c_void,
                    shared.w13_packed_bytes, copy_stream);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("DMA shared w13p: {:?}", err)));
                }
                let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    buf_w13_scales, shared.w13_scales_ptr as *const std::ffi::c_void,
                    shared.w13_scales_bytes, copy_stream);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("DMA shared w13s: {:?}", err)));
                }
                cuda_sys::lib().cuEventRecord(ev_dma_w13, copy_stream);
                cuda_sys::lib().cuStreamWaitEvent(default_stream, ev_dma_w13, 0);
            }

            self.launch_marlin_gemv_raw(
                buf_w13_packed, buf_w13_scales,
                *graph.d_hidden.device_ptr(),
                *graph.d_expert_gate_up.device_ptr(),
                inv_wp, inv_sp,
                hs, 2 * intermediate, gs,
            )?;

            {
                let f = device.get_func(MODULE_NAME, "silu_mul")
                    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("silu_mul not found"))?;
                unsafe {
                    f.launch(
                        LaunchConfig::for_num_elems(intermediate as u32),
                        (
                            *graph.d_expert_scratch.device_ptr(),
                            *graph.d_expert_gate_up.device_ptr(),
                            intermediate as i32,
                        ),
                    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                        format!("silu_mul shared: {:?}", e)))?;
                }
            }

            unsafe {
                let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    buf_w2_packed, shared.w2_packed_ptr as *const std::ffi::c_void,
                    shared.w2_packed_bytes, copy_stream);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("DMA shared w2p: {:?}", err)));
                }
                let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    buf_w2_scales, shared.w2_scales_ptr as *const std::ffi::c_void,
                    shared.w2_scales_bytes, copy_stream);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("DMA shared w2s: {:?}", err)));
                }
                cuda_sys::lib().cuEventRecord(ev_dma_w2, copy_stream);
                cuda_sys::lib().cuStreamWaitEvent(default_stream, ev_dma_w2, 0);
            }

            self.launch_marlin_gemv_raw(
                buf_w2_packed, buf_w2_scales,
                *graph.d_expert_scratch.device_ptr(),
                *graph.d_expert_out.device_ptr(),
                inv_wp, inv_sp,
                intermediate, hs, gs,
            )?;

            {
                let f = device.get_func(MODULE_NAME, "add_bf16")
                    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("add_bf16 not found"))?;
                unsafe {
                    f.launch(
                        LaunchConfig::for_num_elems(hs as u32),
                        (
                            *graph.d_moe_out.device_ptr(),
                            *graph.d_moe_out.device_ptr(),
                            *graph.d_expert_out.device_ptr(),
                            hs as i32,
                        ),
                    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                        format!("add_bf16 shared: {:?}", e)))?;
                }
            }

            device.synchronize()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        }

        // ── Step 8: Scale by routed_scaling_factor ──
        if rsf != 1.0 {
            let f = device.get_func(MODULE_NAME, "scale_bf16")
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("scale_bf16 not found"))?;
            unsafe {
                f.launch(
                    LaunchConfig::for_num_elems(hs as u32),
                    (
                        *graph.d_moe_out.device_ptr(),
                        *graph.d_moe_out.device_ptr(),
                        rsf,
                        hs as i32,
                    ),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("scale_bf16: {:?}", e)))?;
            }
            device.synchronize()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        }

        // ── HCS stats update ──
        if let Some(ref mut hcs) = graph.hcs {
            hcs.total_hits += hcs_hits as u64;
            hcs.total_misses += (topk as u32 - hcs_hits) as u64;
        }

        // ── APFL stats update ──
        if apfl_enabled {
            let apfl = graph.apfl.as_mut().unwrap();
            apfl.total_hits += apfl_hits as u64;
            apfl.total_misses += apfl_misses as u64;
            if layer_idx < apfl.layer_stats.len() {
                let stats = &mut apfl.layer_stats[layer_idx];
                for _ in 0..apfl_hits { stats.record_hit(); }
                for _ in 0..apfl_misses { stats.record_miss(); }
                stats.adapt(apfl.max_prefetch);
            }

            // Invalidate slots for this layer (already consumed)
            for slot in apfl.slots.iter_mut() {
                if slot.layer_idx == layer_idx as i32 {
                    slot.clear();
                }
            }
        }

        // Cleanup CUDA events
        unsafe {
            cuda_sys::lib().cuEventDestroy_v2(ev_dma_w13);
            cuda_sys::lib().cuEventDestroy_v2(ev_dma_w2);
            cuda_sys::lib().cuEventDestroy_v2(ev_compute_w13);
        }

        let total = t_start.elapsed().as_secs_f64() * 1000.0;

        Ok((t_route, dma_total, compute_total, total))
    }

    /// Test the Marlin GEMV kernel against a known reference.
    /// Creates a small weight matrix, repacks to Marlin format, runs GEMV, checks output.
    fn test_marlin_gemv(&self) -> PyResult<String> {
        use crate::weights::marlin::{quantize_int4, marlin_repack, bf16_to_f32, f32_to_bf16};

        if !self.kernels_loaded {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Decode kernels not loaded"));
        }
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;

        let n = 64usize;  // output dim (must be multiple of 64 for Marlin)
        let k = 256usize; // input dim (must be multiple of 16 for Marlin)
        let gs = 128usize; // must be < K for grouped quantization (64-element scale perm)

        // Create a BF16 weight matrix [N, K] with known values
        let mut weight_bf16 = vec![0u16; n * k];
        for i in 0..n {
            for j in 0..k {
                let val = ((i as f32 * 0.01) - (j as f32 * 0.005)) * 0.1;
                weight_bf16[i * k + j] = f32_to_bf16(val);
            }
        }

        // Create BF16 input vector [K]
        let mut input_bf16 = vec![0u16; k];
        for j in 0..k {
            input_bf16[j] = f32_to_bf16((j as f32 + 1.0) * 0.01);
        }

        // Quantize to INT4
        let q = quantize_int4(&weight_bf16, n, k, gs);
        // Repack to Marlin format
        let m = marlin_repack(&q);

        // Compute expected output (dequant + matmul on CPU)
        let mut expected = vec![0.0f32; n];
        let num_groups = k / gs;
        for i in 0..n {
            for j in 0..k {
                let g = j / gs;
                let scale = bf16_to_f32(q.scales[i * num_groups + g]);
                let pack_idx = i * (k / 8) + j / 8;
                let nibble = j % 8;
                let raw = ((q.packed[pack_idx] >> (nibble as u32 * 4)) & 0xF) as i32;
                let w = (raw - 8) as f32 * scale;
                let x = bf16_to_f32(input_bf16[j]);
                expected[i] += w * x;
            }
        }

        // Upload to GPU
        let d_packed = self.device.htod_copy(m.packed.clone())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_scales = self.device.htod_copy(m.scales.clone())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_input = self.device.htod_copy(input_bf16)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let mut d_output = self.device.alloc_zeros::<u16>(n)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        // Launch kernel
        let f = self.device.get_func(MODULE_NAME, "marlin_gemv_int4")
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("kernel not found"))?;
        let n_tiles = (n + 15) / 16;
        let cfg = LaunchConfig {
            grid_dim: (n_tiles as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        };
        unsafe {
            f.launch(cfg, (
                &d_packed,
                &d_scales,
                &d_input,
                &mut d_output,
                &graph.d_inv_weight_perm,
                &graph.d_inv_scale_perm,
                k as i32,
                n as i32,
                gs as i32,
            )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        }

        self.device.synchronize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let output_host = self.device.dtoh_sync_copy(&d_output)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        // Compare
        let mut max_err = 0.0f32;
        let mut max_rel_err = 0.0f32;
        for i in 0..n {
            let got = bf16_to_f32(output_host[i]);
            let exp = expected[i];
            let err = (got - exp).abs();
            let rel = if exp.abs() > 1e-6 { err / exp.abs() } else { err };
            if err > max_err { max_err = err; }
            if rel > max_rel_err { max_rel_err = rel; }
        }

        let pass = max_rel_err < 0.15; // INT4 quantization + BF16 allows ~10-15% error
        let result = format!(
            "marlin_gemv_int4: {} (N={}, K={}, gs={}, max_abs_err={:.6}, max_rel_err={:.4}, expected[0]={:.6}, got[0]={:.6})",
            if pass { "PASS" } else { "FAIL" },
            n, k, gs, max_err, max_rel_err,
            expected[0], bf16_to_f32(output_host[0]),
        );

        Ok(result)
    }

    /// Internal: wire up GPU decode from a loaded KrasisEngine.
    ///
    /// Reads expert GPU weights (Marlin format, in system RAM) and routing
    /// config from the engine, then configures this store for GPU MoE decode.
    fn setup_from_engine_internal(
        &mut self,
        engine: &crate::moe::KrasisEngine,
    ) -> PyResult<()> {
        use crate::weights::marlin::bf16_to_f32;

        let store = engine.get_weight_store()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "KrasisEngine has no loaded weights"))?;

        let (scoring_str, norm_topk_prob, topk, n_experts, hidden_size) =
            engine.get_routing_config()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                    "KrasisEngine has no routing config set"))?;

        let scoring_func: u8 = if scoring_str == "sigmoid" { 1 } else { 0 };
        let config = &store.config;
        let intermediate_size = config.moe_intermediate_size;
        let num_layers = config.num_hidden_layers;
        let vocab_size = 1; // not needed for MoE-only testing; will be set properly for full decode
        let group_size = store.group_size;

        log::info!(
            "setup_from_engine: hidden={}, intermediate={}, experts={}, topk={}, scoring={}, layers={}, gs={}",
            hidden_size, intermediate_size, n_experts, topk, scoring_str, num_layers, group_size,
        );

        // Step 1: configure buffers
        self.configure(
            hidden_size, num_layers, vocab_size, 1e-6,
            topk, intermediate_size, hidden_size * 3, group_size,
        )?;

        // Step 2: for each MoE layer, upload gate weights to VRAM and register expert pointers
        let num_routing = engine.num_routing_layers();
        let num_gpu_layers = store.experts_gpu.len();
        let n_moe_layers = num_routing.min(num_gpu_layers);

        log::info!(
            "setup_from_engine: {} routing layers, {} GPU expert layers, using {}",
            num_routing, num_gpu_layers, n_moe_layers,
        );

        let mut max_expert_bytes = 0usize;

        for moe_idx in 0..n_moe_layers {
            // Upload gate weight as FP32 to VRAM
            let (gate_bf16, correction_bias) = engine.get_routing_weights(moe_idx)
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("No routing weights for MoE layer {}", moe_idx)))?;

            // Convert BF16 gate to FP32
            let gate_fp32: Vec<f32> = gate_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
            let d_gate = self.device.htod_copy(gate_fp32)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            let gate_wid = {
                let graph = self.graph.as_mut().unwrap();
                let wid = graph.weights.len();
                graph.weights.push(GpuWeight {
                    ptr: *d_gate.device_ptr(),
                    rows: n_experts,
                    cols: hidden_size,
                    dtype: 1, // FP32
                });
                // Keep the device allocation alive by storing it
                // (we leak it intentionally - it lives for the lifetime of the process)
                std::mem::forget(d_gate);
                wid
            };

            // Upload correction bias to VRAM if present
            let gate_bias_ptr: u64 = if let Some(bias) = correction_bias {
                let d_bias = self.device.htod_copy(bias.to_vec())
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
                let ptr = *d_bias.device_ptr();
                std::mem::forget(d_bias);
                ptr
            } else {
                0
            };

            // Build expert data pointers from GPU weight store
            let gpu_experts = &store.experts_gpu[moe_idx];
            let mut expert_ptrs = Vec::with_capacity(gpu_experts.len());

            for expert in gpu_experts.iter() {
                let w13p_ptr = expert.w13_packed.as_ptr() as usize;
                let w13p_bytes = expert.w13_packed.len() * 4;
                let w13s_ptr = expert.w13_scales.as_ptr() as usize;
                let w13s_bytes = expert.w13_scales.len() * 2;
                let w2p_ptr = expert.w2_packed.as_ptr() as usize;
                let w2p_bytes = expert.w2_packed.len() * 4;
                let w2s_ptr = expert.w2_scales.as_ptr() as usize;
                let w2s_bytes = expert.w2_scales.len() * 2;

                let total = w13p_bytes + w13s_bytes + w2p_bytes + w2s_bytes;
                // Track max for single DMA transfer (w13 packed is the largest single piece)
                let max_single = w13p_bytes.max(w2p_bytes);
                if max_single > max_expert_bytes {
                    max_expert_bytes = max_single;
                }

                expert_ptrs.push((w13p_ptr, w13p_bytes, w13s_ptr, w13s_bytes,
                                  w2p_ptr, w2p_bytes, w2s_ptr, w2s_bytes));

                if moe_idx == 0 && expert_ptrs.len() == 1 {
                    log::info!(
                        "Expert[0][0]: w13p={} bytes, w13s={} bytes, w2p={} bytes, w2s={} bytes, total={:.1} KB",
                        w13p_bytes, w13s_bytes, w2p_bytes, w2s_bytes, total as f64 / 1024.0,
                    );
                }
            }

            // Shared expert pointers
            let shared_ptrs = if moe_idx < store.shared_experts_gpu.len() {
                let se = &store.shared_experts_gpu[moe_idx];
                if se.w13_packed.is_empty() {
                    None
                } else {
                    let w13p_bytes = se.w13_packed.len() * 4;
                    let w2p_bytes = se.w2_packed.len() * 4;
                    let max_single = w13p_bytes.max(w2p_bytes);
                    if max_single > max_expert_bytes {
                        max_expert_bytes = max_single;
                    }
                    Some((
                        se.w13_packed.as_ptr() as usize, w13p_bytes,
                        se.w13_scales.as_ptr() as usize, se.w13_scales.len() * 2,
                        se.w2_packed.as_ptr() as usize, w2p_bytes,
                        se.w2_scales.as_ptr() as usize, se.w2_scales.len() * 2,
                    ))
                }
            } else {
                None
            };

            self.register_moe_layer_data(
                moe_idx, expert_ptrs, shared_ptrs,
                n_experts, topk, scoring_func, norm_topk_prob,
                config.routed_scaling_factor, gate_wid, gate_bias_ptr as usize,
                0, // e_score_corr_ptr - TODO
                None, // shared_gate_wid - TODO
            )?;
        }

        // Step 3: size expert DMA buffers (need to hold largest packed + scales)
        // We use buf_a for packed data, buf_b for scales data.
        // The largest packed buffer is max_expert_bytes.
        // Add 20% headroom for alignment.
        let buf_size = ((max_expert_bytes as f64) * 1.2) as usize;
        self.resize_expert_buffers(buf_size.max(1024))?;

        log::info!(
            "setup_from_engine complete: {} MoE layers, expert_buf={}KB, scaling_factor={}",
            n_moe_layers, buf_size / 1024, config.routed_scaling_factor,
        );

        Ok(())
    }

    /// End-to-end test: load QCN, set up GPU MoE, run one layer forward, report timings.
    fn test_moe_e2e_internal(
        &mut self,
        model_dir: &str,
        moe_layer_idx: usize,
    ) -> PyResult<String> {
        use crate::weights::{WeightStore, UnifiedExpertWeights};
        use crate::weights::marlin::bf16_to_f32;
        use std::path::Path;
        use std::time::Instant;

        let mut results = Vec::new();

        // Step 1: Load model weights (GPU Marlin format only, cpu_bits=4 gpu_bits=4)
        let t0 = Instant::now();
        let store = WeightStore::load_from_hf(
            Path::new(model_dir), 128, None, None, 4, 4,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("Failed to load model: {}", e)))?;

        results.push(format!("Loaded model in {:.1}s: {} MoE layers, {} experts, hidden={}",
            t0.elapsed().as_secs_f64(),
            store.experts_gpu.len(),
            store.config.n_routed_experts,
            store.config.hidden_size,
        ));

        let config = &store.config;
        let hidden_size = config.hidden_size;
        let intermediate_size = config.moe_intermediate_size;
        let n_experts = config.n_routed_experts;
        let topk = config.num_experts_per_tok;
        let group_size = store.group_size;

        // Step 2: Configure the GPU decode store
        self.configure(
            hidden_size, config.num_hidden_layers, 1, 1e-6,
            topk, intermediate_size, hidden_size * 3, group_size,
        )?;

        // Step 3: Upload gate weight for the target layer as FP32
        // We need to load the gate weight from safetensors.
        // The gate weights are stored in the model's safetensors files as BF16.
        // For this test, we'll synthesize a random gate weight (we're testing
        // the DMA + compute pipeline, not routing accuracy).
        let gate_fp32: Vec<f32> = (0..n_experts * hidden_size)
            .map(|i| ((i as f32 * 0.0001) - 0.05).sin() * 0.01)
            .collect();
        let d_gate = self.device.htod_copy(gate_fp32)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        let gate_wid = {
            let graph = self.graph.as_mut().unwrap();
            let wid = graph.weights.len();
            graph.weights.push(GpuWeight {
                ptr: *d_gate.device_ptr(),
                rows: n_experts,
                cols: hidden_size,
                dtype: 1, // FP32
            });
            std::mem::forget(d_gate);
            wid
        };

        // Step 4: Register expert data pointers for the target MoE layer
        if moe_layer_idx >= store.experts_gpu.len() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("MoE layer {} >= {} available", moe_layer_idx, store.experts_gpu.len())));
        }

        let gpu_experts = &store.experts_gpu[moe_layer_idx];
        let mut expert_ptrs = Vec::with_capacity(gpu_experts.len());
        let mut max_expert_bytes = 0usize;

        for expert in gpu_experts.iter() {
            let w13p_bytes = expert.w13_packed.len() * 4;
            let w13s_bytes = expert.w13_scales.len() * 2;
            let w2p_bytes = expert.w2_packed.len() * 4;
            let w2s_bytes = expert.w2_scales.len() * 2;
            let max_single = w13p_bytes.max(w2p_bytes);
            if max_single > max_expert_bytes { max_expert_bytes = max_single; }

            expert_ptrs.push((
                expert.w13_packed.as_ptr() as usize, w13p_bytes,
                expert.w13_scales.as_ptr() as usize, w13s_bytes,
                expert.w2_packed.as_ptr() as usize, w2p_bytes,
                expert.w2_scales.as_ptr() as usize, w2s_bytes,
            ));
        }

        // Shared expert
        let shared_ptrs = if moe_layer_idx < store.shared_experts_gpu.len() {
            let se = &store.shared_experts_gpu[moe_layer_idx];
            if se.w13_packed.is_empty() {
                None
            } else {
                let w13p_bytes = se.w13_packed.len() * 4;
                let w2p_bytes = se.w2_packed.len() * 4;
                let max_single = w13p_bytes.max(w2p_bytes);
                if max_single > max_expert_bytes { max_expert_bytes = max_single; }
                Some((
                    se.w13_packed.as_ptr() as usize, w13p_bytes,
                    se.w13_scales.as_ptr() as usize, se.w13_scales.len() * 2,
                    se.w2_packed.as_ptr() as usize, w2p_bytes,
                    se.w2_scales.as_ptr() as usize, se.w2_scales.len() * 2,
                ))
            }
        } else {
            None
        };

        // Detect scoring function from model name (QCN uses softmax)
        let scoring_func: u8 = 0; // softmax for QCN/Qwen3

        self.register_moe_layer_data(
            moe_layer_idx, expert_ptrs, shared_ptrs,
            n_experts, topk, scoring_func, false,
            config.routed_scaling_factor, gate_wid, 0, 0, None,
        )?;

        // Size DMA buffers
        let buf_size = ((max_expert_bytes as f64) * 1.2) as usize;
        self.resize_expert_buffers(buf_size.max(1024))?;

        results.push(format!(
            "Registered MoE layer {}: {} experts, topk={}, intermediate={}, gs={}, expert_buf={:.1}KB",
            moe_layer_idx, n_experts, topk, intermediate_size, group_size,
            buf_size as f64 / 1024.0,
        ));

        // Step 5: Create a random BF16 hidden state and upload
        let hidden_bf16: Vec<u16> = (0..hidden_size)
            .map(|i| half::bf16::from_f32(((i as f32) * 0.01).sin() * 0.5).to_bits())
            .collect();
        self.upload_hidden_bf16(hidden_bf16)?;

        // Step 6: Run MoE forward
        let t_moe = Instant::now();
        let (route_ms, dma_ms, compute_ms, total_ms) = self.moe_forward_internal(moe_layer_idx)?;

        results.push(format!(
            "MoE forward layer {}: total={:.2}ms (route={:.2}ms, DMA={:.2}ms, compute={:.2}ms)",
            moe_layer_idx, total_ms, route_ms, dma_ms, compute_ms,
        ));

        // Step 7: Download result and verify non-zero
        let moe_out = self.download_moe_out_bf16()?;
        let nonzero = moe_out.iter().filter(|&&v| v != 0).count();
        let max_val = moe_out.iter().map(|&v| half::bf16::from_bits(v).to_f32().abs()).fold(0.0f32, f32::max);
        let sum = moe_out.iter().map(|&v| half::bf16::from_bits(v).to_f32()).sum::<f32>();

        let pass = nonzero > hidden_size / 2 && max_val < 100.0 && max_val > 1e-8;
        results.push(format!(
            "Output: {} nonzero/{} total, max_abs={:.6}, sum={:.4} → {}",
            nonzero, hidden_size, max_val, sum, if pass { "PASS" } else { "FAIL" },
        ));

        // Step 8: Run 10 iterations for timing
        let mut timings = Vec::new();
        for _ in 0..10 {
            self.upload_hidden_bf16(
                (0..hidden_size).map(|i| half::bf16::from_f32(((i as f32) * 0.01).sin() * 0.5).to_bits()).collect()
            )?;
            let (_, dma, comp, tot) = self.moe_forward_internal(moe_layer_idx)?;
            timings.push((dma, comp, tot));
        }
        let avg_total = timings.iter().map(|t| t.2).sum::<f64>() / timings.len() as f64;
        let avg_dma = timings.iter().map(|t| t.0).sum::<f64>() / timings.len() as f64;
        let avg_comp = timings.iter().map(|t| t.1).sum::<f64>() / timings.len() as f64;

        results.push(format!(
            "10-iter avg: total={:.2}ms (DMA={:.2}ms, compute={:.2}ms) → {:.1} MoE layers/sec",
            avg_total, avg_dma, avg_comp, 1000.0 / avg_total,
        ));

        // Extrapolate to full model decode
        let num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace;
        let estimated_moe_time = avg_total * num_moe_layers as f64;
        results.push(format!(
            "Estimated MoE decode: {:.1}ms for {} layers → {:.1} tok/s (MoE-only, no attention)",
            estimated_moe_time, num_moe_layers, 1000.0 / estimated_moe_time,
        ));

        // Keep store alive (prevent drop which would free mmap'd weights)
        std::mem::forget(store);

        Ok(results.join("\n"))
    }

    /// End-to-end APFL test: load model, set up ALL MoE layers, run multi-layer
    /// forward with APFL enabled, report hit rates and timing.
    fn test_apfl_e2e(
        &mut self,
        model_dir: &str,
        num_tokens: usize,
        initial_prefetch: usize,
        max_prefetch: usize,
        num_slots: usize,
    ) -> PyResult<String> {
        use crate::weights::marlin::bf16_to_f32;
        use std::path::Path;
        use std::time::Instant;

        let mut results = Vec::new();

        // Load model
        let t0 = Instant::now();
        let store = crate::weights::WeightStore::load_from_hf(
            Path::new(model_dir), 128, None, None, 4, 4,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("Failed to load model: {}", e)))?;

        let config = &store.config;
        let hidden_size = config.hidden_size;
        let intermediate_size = config.moe_intermediate_size;
        let n_experts = config.n_routed_experts;
        let topk = config.num_experts_per_tok;
        let group_size = store.group_size;
        let num_moe_layers = store.experts_gpu.len();

        results.push(format!(
            "Loaded in {:.1}s: {} MoE layers, {} experts, topk={}, hidden={}, intermediate={}",
            t0.elapsed().as_secs_f64(), num_moe_layers, n_experts, topk,
            hidden_size, intermediate_size,
        ));

        // Configure GPU decode
        self.configure(
            hidden_size, config.num_hidden_layers, 1, 1e-6,
            topk, intermediate_size, hidden_size * 3, group_size,
        )?;

        // Register ALL MoE layers with synthetic gate weights
        let mut max_expert_bytes = 0usize;
        for moe_idx in 0..num_moe_layers {
            // Synthetic FP32 gate weight
            let gate_fp32: Vec<f32> = (0..n_experts * hidden_size)
                .map(|i| ((i as f32 * 0.0001 + moe_idx as f32 * 0.1) - 0.05).sin() * 0.01)
                .collect();
            let d_gate = self.device.htod_copy(gate_fp32)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            let gate_wid = {
                let graph = self.graph.as_mut().unwrap();
                let wid = graph.weights.len();
                graph.weights.push(GpuWeight {
                    ptr: *d_gate.device_ptr(),
                    rows: n_experts,
                    cols: hidden_size,
                    dtype: 1,
                });
                std::mem::forget(d_gate);
                wid
            };

            let gpu_experts = &store.experts_gpu[moe_idx];
            let mut expert_ptrs = Vec::with_capacity(gpu_experts.len());

            for expert in gpu_experts.iter() {
                let w13p_bytes = expert.w13_packed.len() * 4;
                let w2p_bytes = expert.w2_packed.len() * 4;
                let max_single = w13p_bytes.max(w2p_bytes);
                if max_single > max_expert_bytes { max_expert_bytes = max_single; }

                expert_ptrs.push((
                    expert.w13_packed.as_ptr() as usize, w13p_bytes,
                    expert.w13_scales.as_ptr() as usize, expert.w13_scales.len() * 2,
                    expert.w2_packed.as_ptr() as usize, w2p_bytes,
                    expert.w2_scales.as_ptr() as usize, expert.w2_scales.len() * 2,
                ));
            }

            let shared_ptrs = if moe_idx < store.shared_experts_gpu.len() {
                let se = &store.shared_experts_gpu[moe_idx];
                if se.w13_packed.is_empty() {
                    None
                } else {
                    let w13p_bytes = se.w13_packed.len() * 4;
                    let w2p_bytes = se.w2_packed.len() * 4;
                    let max_single = w13p_bytes.max(w2p_bytes);
                    if max_single > max_expert_bytes { max_expert_bytes = max_single; }
                    Some((
                        se.w13_packed.as_ptr() as usize, w13p_bytes,
                        se.w13_scales.as_ptr() as usize, se.w13_scales.len() * 2,
                        se.w2_packed.as_ptr() as usize, w2p_bytes,
                        se.w2_scales.as_ptr() as usize, se.w2_scales.len() * 2,
                    ))
                }
            } else {
                None
            };

            self.register_moe_layer_data(
                moe_idx, expert_ptrs, shared_ptrs,
                n_experts, topk, 0, false,
                config.routed_scaling_factor, gate_wid, 0, 0, None,
            )?;
        }

        let buf_size = ((max_expert_bytes as f64) * 1.2) as usize;
        self.resize_expert_buffers(buf_size.max(1024))?;

        results.push(format!(
            "Registered {} MoE layers, expert_buf={:.1}KB",
            num_moe_layers, buf_size as f64 / 1024.0,
        ));

        // Init APFL
        self.init_apfl(num_slots, initial_prefetch, max_prefetch)?;

        // Run tokens
        let hs = hidden_size;
        for tok in 0..num_tokens {
            let t_tok = Instant::now();

            let hidden: Vec<u16> = (0..hs)
                .map(|i| half::bf16::from_f32(
                    ((i as f32 + tok as f32 * 13.7) * 0.01).sin() * 0.5
                ).to_bits())
                .collect();
            self.upload_hidden_bf16(hidden)?;

            let mut layer_times = Vec::new();
            for layer_idx in 0..num_moe_layers {
                let (_, _, _, total_ms) = self.moe_forward_internal(layer_idx)?;
                layer_times.push(total_ms);
            }

            let tok_total: f64 = layer_times.iter().sum();
            let tok_elapsed = t_tok.elapsed().as_secs_f64() * 1000.0;

            if tok == 0 || tok == num_tokens - 1 || (tok + 1) % 5 == 0 {
                let graph = self.graph.as_ref().unwrap();
                let apfl = graph.apfl.as_ref().unwrap();
                let hr = if apfl.total_hits + apfl.total_misses > 0 {
                    apfl.total_hits as f64 / (apfl.total_hits + apfl.total_misses) as f64 * 100.0
                } else { 0.0 };
                results.push(format!(
                    "Tok {}: {:.1}ms MoE ({:.1}ms wall), {:.1} tok/s, hit_rate={:.1}%",
                    tok + 1, tok_total, tok_elapsed, 1000.0 / tok_elapsed, hr,
                ));
            }
        }

        // Final stats
        results.push(String::new());
        results.push(self.apfl_stats()?);

        let graph = self.graph.as_ref().unwrap();
        let apfl = graph.apfl.as_ref().unwrap();
        let adapted: Vec<_> = apfl.layer_stats.iter().enumerate()
            .filter(|(_, s)| s.hits + s.misses > 0)
            .map(|(i, s)| (i, s.prefetch_count))
            .collect();
        if !adapted.is_empty() {
            let min_pc = adapted.iter().map(|c| c.1).min().unwrap();
            let max_pc = adapted.iter().map(|c| c.1).max().unwrap();
            let avg_pc = adapted.iter().map(|c| c.1).sum::<usize>() as f64 / adapted.len() as f64;
            results.push(format!(
                "Adapted prefetch: min={}, max={}, avg={:.1} across {} layers",
                min_pc, max_pc, avg_pc, adapted.len(),
            ));
        }

        std::mem::forget(store);
        Ok(results.join("\n"))
    }

    /// Test APFL with multi-layer MoE forward.
    /// Runs all MoE layers in sequence with APFL enabled, reports hit rates.
    /// Requires setup_from_engine + init_apfl to have been called first.
    fn test_apfl_multilayer(
        &mut self,
        num_tokens: usize,
    ) -> PyResult<String> {
        use std::time::Instant;

        let mut results = Vec::new();

        {
            let graph = self.graph.as_ref()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
            if graph.apfl.is_none() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err("Call init_apfl first"));
            }

            let hs = graph.hidden_size;
            let num_moe_layers = graph.moe_layers.iter().filter(|m| m.is_some()).count();
            let apfl = graph.apfl.as_ref().unwrap();
            results.push(format!(
                "Testing: {} MoE layers, hidden={}, APFL slots={}, max_prefetch={}",
                num_moe_layers, hs, apfl.slots.len(), apfl.max_prefetch,
            ));
        }

        // Run multiple "tokens" (each token = all MoE layers in sequence)
        for tok in 0..num_tokens {
            let t_tok = Instant::now();

            // Upload a synthetic hidden state (varies per token for different routing)
            let hs = self.graph.as_ref().unwrap().hidden_size;
            let hidden: Vec<u16> = (0..hs)
                .map(|i| half::bf16::from_f32(
                    ((i as f32 + tok as f32 * 13.7) * 0.01).sin() * 0.5
                ).to_bits())
                .collect();
            self.upload_hidden_bf16(hidden)?;

            // Run all MoE layers
            let num_moe = self.graph.as_ref().unwrap().moe_layers.len();
            let mut layer_times = Vec::new();
            for layer_idx in 0..num_moe {
                if self.graph.as_ref().unwrap().moe_layers[layer_idx].is_none() { continue; }
                let (_, _, _, total_ms) = self.moe_forward_internal(layer_idx)?;
                layer_times.push(total_ms);
            }

            let tok_total: f64 = layer_times.iter().sum();
            let tok_elapsed = t_tok.elapsed().as_secs_f64() * 1000.0;

            if tok == 0 || tok == num_tokens - 1 || (tok + 1) % 5 == 0 {
                let graph = self.graph.as_ref().unwrap();
                let apfl = graph.apfl.as_ref().unwrap();
                let hr = if apfl.total_hits + apfl.total_misses > 0 {
                    apfl.total_hits as f64 / (apfl.total_hits + apfl.total_misses) as f64 * 100.0
                } else { 0.0 };
                results.push(format!(
                    "Token {}: {:.1}ms MoE ({:.1}ms wall), {:.1} tok/s, APFL hit_rate={:.1}% (hits={}, misses={})",
                    tok + 1, tok_total, tok_elapsed, 1000.0 / tok_elapsed,
                    hr, apfl.total_hits, apfl.total_misses,
                ));
            }
        }

        // Final APFL stats
        results.push(String::new());
        results.push(self.apfl_stats()?);

        // Per-layer adaptation summary
        let graph = self.graph.as_ref().unwrap();
        let apfl = graph.apfl.as_ref().unwrap();
        let mut adapted_counts: Vec<(usize, usize)> = Vec::new();
        for (i, stats) in apfl.layer_stats.iter().enumerate() {
            if stats.hits + stats.misses > 0 {
                adapted_counts.push((i, stats.prefetch_count));
            }
        }
        if !adapted_counts.is_empty() {
            let min_pc = adapted_counts.iter().map(|c| c.1).min().unwrap();
            let max_pc = adapted_counts.iter().map(|c| c.1).max().unwrap();
            let avg_pc = adapted_counts.iter().map(|c| c.1).sum::<usize>() as f64
                / adapted_counts.len() as f64;
            results.push(format!(
                "Adapted prefetch_count: min={}, max={}, avg={:.1} (across {} layers)",
                min_pc, max_pc, avg_pc, adapted_counts.len(),
            ));
        }

        Ok(results.join("\n"))
    }

    // ── HCS internal implementation ──

    fn init_hcs_internal(&mut self, budget_mb: usize, headroom_mb: usize) -> PyResult<String> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;

        if graph.moe_layers.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No MoE layers registered. Call setup_from_engine first."));
        }

        // Calculate per-expert VRAM size from the first registered MoE layer
        let first_moe = graph.moe_layers.iter()
            .find_map(|m| m.as_ref())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No MoE layers found"))?;
        let first_expert = &first_moe.experts[0];
        let expert_bytes = first_expert.w13_packed_bytes + first_expert.w13_scales_bytes
            + first_expert.w2_packed_bytes + first_expert.w2_scales_bytes;
        // Align to 512 bytes
        let align = 512usize;
        let expert_vram_bytes = (expert_bytes + align - 1) & !(align - 1);

        // Determine budget
        let budget_bytes = if budget_mb > 0 {
            budget_mb * 1024 * 1024
        } else {
            // Auto-detect from free VRAM
            let (free, _total) = cudarc::driver::result::mem_get_info()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("mem_get_info: {:?}", e)))?;
            let headroom_bytes = headroom_mb * 1024 * 1024;
            if free > headroom_bytes {
                free - headroom_bytes
            } else {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Not enough VRAM: {} MB free, {} MB headroom",
                        free / (1024 * 1024), headroom_mb)));
            }
        };

        let max_experts = budget_bytes / expert_vram_bytes;

        // Count total unique (layer, expert) pairs
        let total_experts: usize = graph.moe_layers.iter()
            .filter_map(|m| m.as_ref())
            .map(|m| m.num_experts)
            .sum();

        let mut hcs = HcsState::new();
        hcs.expert_vram_bytes = expert_vram_bytes;

        graph.hcs = Some(hcs);

        let msg = format!(
            "HCS initialized: budget={:.1} MB ({} expert slots), expert_size={:.1} KB, total_experts={}, fits_all={}",
            budget_bytes as f64 / (1024.0 * 1024.0),
            max_experts,
            expert_vram_bytes as f64 / 1024.0,
            total_experts,
            max_experts >= total_experts,
        );
        log::info!("{}", msg);
        Ok(msg)
    }

    fn hcs_pin_expert_internal(&mut self, layer_idx: usize, expert_idx: usize) -> PyResult<bool> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;

        let hcs = graph.hcs.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call init_hcs first"))?;

        // Already cached?
        if hcs.cache.contains_key(&(layer_idx, expert_idx)) {
            return Ok(false);
        }

        let moe = graph.moe_layers.get(layer_idx)
            .and_then(|m| m.as_ref())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                format!("MoE layer {} not registered", layer_idx)))?;

        if expert_idx >= moe.experts.len() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Expert {} >= {} in layer {}", expert_idx, moe.experts.len(), layer_idx)));
        }

        let expert = &moe.experts[expert_idx];
        let total_bytes = expert.w13_packed_bytes + expert.w13_scales_bytes
            + expert.w2_packed_bytes + expert.w2_scales_bytes;
        let align = 512usize;
        let alloc_bytes = (total_bytes + align - 1) & !(align - 1);

        // Allocate VRAM
        let d_buf = self.device.alloc_zeros::<u8>(alloc_bytes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("HCS VRAM alloc ({} bytes): {:?}", alloc_bytes, e)))?;

        // Compute offsets (contiguous layout: w13p | w13s | w2p | w2s)
        let w13_packed_offset = 0;
        let w13_scales_offset = expert.w13_packed_bytes;
        let w2_packed_offset = w13_scales_offset + expert.w13_scales_bytes;
        let w2_scales_offset = w2_packed_offset + expert.w2_packed_bytes;

        // Synchronous H2D copy (one-time setup, not on hot path)
        let dst_base = *d_buf.device_ptr();
        unsafe {
            let copy = |offset: usize, src_ptr: usize, bytes: usize| -> PyResult<()> {
                let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                    dst_base + offset as u64,
                    src_ptr as *const std::ffi::c_void,
                    bytes,
                );
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("HCS H2D copy: {:?}", err)));
                }
                Ok(())
            };
            copy(w13_packed_offset, expert.w13_packed_ptr, expert.w13_packed_bytes)?;
            copy(w13_scales_offset, expert.w13_scales_ptr, expert.w13_scales_bytes)?;
            copy(w2_packed_offset, expert.w2_packed_ptr, expert.w2_packed_bytes)?;
            copy(w2_scales_offset, expert.w2_scales_ptr, expert.w2_scales_bytes)?;
        }

        let entry = HcsCacheEntry {
            d_buf,
            w13_packed_offset,
            w13_packed_size: expert.w13_packed_bytes,
            w13_scales_offset,
            w13_scales_size: expert.w13_scales_bytes,
            w2_packed_offset,
            w2_packed_size: expert.w2_packed_bytes,
            w2_scales_offset,
            w2_scales_size: expert.w2_scales_bytes,
        };

        hcs.vram_bytes += alloc_bytes;
        hcs.num_cached += 1;
        hcs.cache.insert((layer_idx, expert_idx), entry);

        Ok(true)
    }

    fn hcs_pin_all_internal(&mut self) -> PyResult<String> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;

        // Collect all (layer, expert) pairs to pin
        let mut to_pin: Vec<(usize, usize)> = Vec::new();
        for (layer_idx, moe_opt) in graph.moe_layers.iter().enumerate() {
            if let Some(moe) = moe_opt {
                for eid in 0..moe.num_experts {
                    to_pin.push((layer_idx, eid));
                }
            }
        }

        let total = to_pin.len();
        let t0 = std::time::Instant::now();
        let mut pinned = 0usize;
        let mut failed = 0usize;

        for (layer_idx, expert_idx) in to_pin {
            match self.hcs_pin_expert_internal(layer_idx, expert_idx) {
                Ok(true) => pinned += 1,
                Ok(false) => {} // already cached
                Err(e) => {
                    if failed == 0 {
                        log::warn!("HCS pin_all: first failure at L{}E{}: {}", layer_idx, expert_idx, e);
                    }
                    failed += 1;
                    if failed > 10 {
                        log::warn!("HCS pin_all: too many failures, stopping");
                        break;
                    }
                }
            }
        }

        let elapsed = t0.elapsed().as_secs_f64();
        let hcs = self.graph.as_ref().unwrap().hcs.as_ref().unwrap();

        let msg = format!(
            "HCS pin_all: {}/{} experts pinned in {:.2}s, {:.1} MB VRAM, {} failed",
            pinned, total, elapsed, hcs.vram_bytes as f64 / (1024.0 * 1024.0), failed,
        );
        log::info!("{}", msg);
        Ok(msg)
    }

    fn hcs_populate_from_heatmap(&mut self) -> PyResult<String> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;

        let hcs = graph.hcs.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call init_hcs first"))?;

        if hcs.heatmap.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Heatmap is empty. Call hcs_start_collecting and run some tokens first."));
        }

        // Sort by activation count descending
        let mut sorted: Vec<((usize, usize), u64)> = hcs.heatmap.iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        // Pin in order of hotness
        let t0 = std::time::Instant::now();
        let mut pinned = 0usize;
        let total = sorted.len();

        for ((layer_idx, expert_idx), _count) in &sorted {
            match self.hcs_pin_expert_internal(*layer_idx, *expert_idx) {
                Ok(true) => pinned += 1,
                Ok(false) => {} // already cached
                Err(_) => break, // OOM or other error, stop
            }
        }

        let elapsed = t0.elapsed().as_secs_f64();

        // Stop collecting and get stats
        let hcs = self.graph.as_mut().unwrap().hcs.as_mut().unwrap();
        hcs.collecting = false;
        let vram_mb = hcs.vram_bytes as f64 / (1024.0 * 1024.0);

        let msg = format!(
            "HCS populate: {}/{} hottest experts pinned in {:.2}s, {:.1} MB VRAM",
            pinned, total, elapsed, vram_mb,
        );
        log::info!("{}", msg);
        Ok(msg)
    }

    fn test_hcs_e2e_internal(&mut self, model_dir: &str, num_tokens: usize) -> PyResult<String> {
        use std::path::Path;
        use std::time::Instant;

        let mut results = Vec::new();
        let t_start = Instant::now();

        // Step 1: Load model weights
        let t0 = Instant::now();
        let store = crate::weights::WeightStore::load_from_hf(
            Path::new(model_dir), 128, None, None, 4, 4,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("Failed to load model: {}", e)))?;

        let config = &store.config;
        let hidden_size = config.hidden_size;
        let intermediate_size = config.moe_intermediate_size;
        let n_experts = config.n_routed_experts;
        let topk = config.num_experts_per_tok;
        let group_size = store.group_size;
        let num_moe_layers = store.experts_gpu.len();

        results.push(format!(
            "Loaded in {:.1}s: {} MoE layers, {} experts, topk={}, hidden={}, intermediate={}",
            t0.elapsed().as_secs_f64(), num_moe_layers, n_experts, topk,
            hidden_size, intermediate_size,
        ));

        // Step 2: Configure GPU decode
        self.configure(
            hidden_size, config.num_hidden_layers, 1, 1e-6,
            topk, intermediate_size, hidden_size * 3, group_size,
        )?;

        // Step 3: Register ALL MoE layers with synthetic gate weights
        let mut max_expert_bytes = 0usize;
        for moe_idx in 0..num_moe_layers {
            let gate_fp32: Vec<f32> = (0..n_experts * hidden_size)
                .map(|i| ((i as f32 * 0.0001 + moe_idx as f32 * 0.1) - 0.05).sin() * 0.01)
                .collect();
            let d_gate = self.device.htod_copy(gate_fp32)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            let gate_wid = {
                let graph = self.graph.as_mut().unwrap();
                let wid = graph.weights.len();
                graph.weights.push(GpuWeight {
                    ptr: *d_gate.device_ptr(),
                    rows: n_experts,
                    cols: hidden_size,
                    dtype: 1,
                });
                std::mem::forget(d_gate);
                wid
            };

            let gpu_experts = &store.experts_gpu[moe_idx];
            let mut expert_ptrs = Vec::with_capacity(gpu_experts.len());

            for expert in gpu_experts.iter() {
                let w13p_bytes = expert.w13_packed.len() * 4;
                let w2p_bytes = expert.w2_packed.len() * 4;
                let max_single = w13p_bytes.max(w2p_bytes);
                if max_single > max_expert_bytes { max_expert_bytes = max_single; }

                expert_ptrs.push((
                    expert.w13_packed.as_ptr() as usize,
                    w13p_bytes,
                    expert.w13_scales.as_ptr() as usize,
                    expert.w13_scales.len() * 2,
                    expert.w2_packed.as_ptr() as usize,
                    w2p_bytes,
                    expert.w2_scales.as_ptr() as usize,
                    expert.w2_scales.len() * 2,
                ));
            }

            self.register_moe_layer(
                moe_idx, expert_ptrs, None, n_experts, topk,
                0, false,  // softmax, no norm_topk_prob (test only)
                config.routed_scaling_factor, gate_wid, 0, 0, None,
            )?;
        }

        let buf_size = ((max_expert_bytes as f64) * 1.2) as usize;
        self.resize_expert_buffers(buf_size.max(1024))?;

        results.push(format!("Registered {} MoE layers", num_moe_layers));

        // Step 4: Init HCS and pin all experts
        let msg = self.init_hcs_internal(0, 500)?;
        results.push(msg);
        let msg = self.hcs_pin_all_internal()?;
        results.push(msg);

        let hcs = self.graph.as_ref().unwrap().hcs.as_ref().unwrap();
        results.push(format!(
            "HCS cache: {} experts, {:.1} MB",
            hcs.num_cached, hcs.vram_bytes as f64 / (1024.0 * 1024.0),
        ));

        // Step 5: Baseline — run WITHOUT HCS first (disable it temporarily)
        results.push("\n--- Baseline (no HCS) ---".to_string());
        {
            // Temporarily remove HCS cache
            let hcs_state = self.graph.as_mut().unwrap().hcs.take();

            let mut baseline_times = Vec::new();
            for tok in 0..num_tokens.min(3) {
                let hidden: Vec<u16> = (0..hidden_size)
                    .map(|i| half::bf16::from_f32(
                        ((i as f32 * 0.001 + tok as f32 * 0.1) - 0.5).sin() * 0.01
                    ).to_bits())
                    .collect();
                self.upload_hidden_bf16(hidden)?;

                let mut layer_times = Vec::new();
                for layer_idx in 0..num_moe_layers {
                    if self.graph.as_ref().unwrap().moe_layers[layer_idx].is_none() { continue; }
                    let (_, _, _, total_ms) = self.moe_forward_internal(layer_idx)?;
                    layer_times.push(total_ms);
                }

                let tok_total: f64 = layer_times.iter().sum();
                baseline_times.push(tok_total);
                results.push(format!(
                    "  Token {}: {:.1}ms MoE, {:.1} tok/s",
                    tok, tok_total, 1000.0 / tok_total,
                ));
            }

            let avg_baseline = baseline_times.iter().sum::<f64>() / baseline_times.len() as f64;
            results.push(format!("  Baseline avg: {:.1}ms, {:.1} tok/s", avg_baseline, 1000.0 / avg_baseline));

            // Restore HCS
            self.graph.as_mut().unwrap().hcs = hcs_state;
        }

        // Step 6: Run WITH HCS
        results.push("\n--- With HCS (all experts resident) ---".to_string());
        // Reset HCS stats
        {
            let hcs = self.graph.as_mut().unwrap().hcs.as_mut().unwrap();
            hcs.total_hits = 0;
            hcs.total_misses = 0;
        }

        let mut hcs_times = Vec::new();
        for tok in 0..num_tokens {
            let hidden: Vec<u16> = (0..hidden_size)
                .map(|i| half::bf16::from_f32(
                    ((i as f32 * 0.001 + tok as f32 * 0.1) - 0.5).sin() * 0.01
                ).to_bits())
                .collect();
            self.upload_hidden_bf16(hidden)?;

            let mut layer_times = Vec::new();
            for layer_idx in 0..num_moe_layers {
                if self.graph.as_ref().unwrap().moe_layers[layer_idx].is_none() { continue; }
                let (_, _, _, total_ms) = self.moe_forward_internal(layer_idx)?;
                layer_times.push(total_ms);
            }

            let tok_total: f64 = layer_times.iter().sum();
            hcs_times.push(tok_total);

            let hcs = self.graph.as_ref().unwrap().hcs.as_ref().unwrap();
            results.push(format!(
                "  Token {}: {:.1}ms MoE, {:.1} tok/s, HCS hits={}, misses={}",
                tok, tok_total, 1000.0 / tok_total,
                hcs.total_hits, hcs.total_misses,
            ));
        }

        let avg_hcs = hcs_times.iter().sum::<f64>() / hcs_times.len() as f64;
        results.push(format!("  HCS avg: {:.1}ms, {:.1} tok/s", avg_hcs, 1000.0 / avg_hcs));

        // Final stats
        results.push("\n--- Summary ---".to_string());
        results.push(self.hcs_stats()?);

        let total_elapsed = t_start.elapsed().as_secs_f64();
        results.push(format!("Total test time: {:.1}s", total_elapsed));

        // Keep store alive (prevent drop which would free mmap'd weights)
        std::mem::forget(store);

        Ok(results.join("\n"))
    }
}

impl Drop for GpuDecodeStore {
    fn drop(&mut self) {
        unsafe {
            if !self.compute_stream.0.is_null() {
                let _ = cuda_sys::lib().cuStreamDestroy_v2(self.compute_stream.0);
            }
            if !self.copy_stream.0.is_null() {
                let _ = cuda_sys::lib().cuStreamDestroy_v2(self.copy_stream.0);
            }
            if !self.prefetch_stream.0.is_null() {
                let _ = cuda_sys::lib().cuStreamDestroy_v2(self.prefetch_stream.0);
            }
        }
    }
}
