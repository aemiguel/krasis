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
    "uninterleave_qkvz",
    "compute_gate_beta",
    "repeat_interleave_heads",
    "l2norm_scale_per_head",
    "gated_delta_net_step",
    "la_recurrence",
    "gated_rmsnorm_silu",
    "per_head_rmsnorm",
    "apply_rope",
    "kv_cache_write",
    "gqa_attention",
    "gqa_attention_tiled",
    "gqa_attention_reduce",
    "split_gated_q",
    "apply_gated_attn",
    "bf16_to_fp32",
    "fp32_to_bf16",
    "marlin_gemv_int4",
    "marlin_gemv_int4_fused_silu_accum",
    "marlin_gemv_int4_v2",
    "reduce_ksplits_bf16",
    "marlin_gemv_int4_fused_silu_accum_v2",
    "reduce_ksplits_weighted_accum_bf16",
    "sigmoid_gate_inplace_bf16",
    // "fused_gate_topk" exists in .cu but is not loaded (single-block = slow on multi-SM GPUs)
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
    d_buf: Option<cudarc::driver::CudaSlice<u8>>,  // owned buffer (None for external/pool)
    w13_packed_offset: usize,
    w13_packed_size: usize,
    w13_scales_offset: usize,
    w13_scales_size: usize,
    w2_packed_offset: usize,
    w2_packed_size: usize,
    w2_scales_offset: usize,
    w2_scales_size: usize,
    // Raw pointers for externally-owned VRAM (Python HCS buffers) or pool entries
    ext_w13_packed: u64,
    ext_w13_scales: u64,
    ext_w2_packed: u64,
    ext_w2_scales: u64,
    /// Pool slot index (Some = pool entry, can be evicted/reused; None = external or individual alloc)
    pool_slot: Option<usize>,
}

impl HcsCacheEntry {
    fn w13_packed_ptr(&self) -> u64 {
        if let Some(ref buf) = self.d_buf {
            *buf.device_ptr() + self.w13_packed_offset as u64
        } else { self.ext_w13_packed }
    }
    fn w13_scales_ptr(&self) -> u64 {
        if let Some(ref buf) = self.d_buf {
            *buf.device_ptr() + self.w13_scales_offset as u64
        } else { self.ext_w13_scales }
    }
    fn w2_packed_ptr(&self) -> u64 {
        if let Some(ref buf) = self.d_buf {
            *buf.device_ptr() + self.w2_packed_offset as u64
        } else { self.ext_w2_packed }
    }
    fn w2_scales_ptr(&self) -> u64 {
        if let Some(ref buf) = self.d_buf {
            *buf.device_ptr() + self.w2_scales_offset as u64
        } else { self.ext_w2_scales }
    }
}

/// HCS state: resident expert cache + activation heatmap + dynamic eviction.
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

    // ── Pool-based VRAM for dynamic eviction ──
    /// One contiguous VRAM allocation divided into equal-sized expert slots.
    pool_buf: Option<cudarc::driver::CudaSlice<u8>>,
    /// Bytes per slot (aligned).
    pool_slot_size: usize,
    /// Total number of slots in the pool.
    pool_num_slots: usize,
    /// Stack of available slot indices (pop to allocate, push to free).
    pool_free_slots: Vec<usize>,
    /// Reverse mapping: slot index → (layer, expert) currently occupying it.
    pool_slot_to_expert: Vec<Option<(usize, usize)>>,

    // ── Dynamic eviction: sliding window activation tracking ──
    /// Bitset for current prompt: 1 bit per (layer_idx * num_experts + expert_idx).
    current_activations: Vec<u64>,
    /// Max experts per layer (stride for bit indexing).
    num_experts_per_layer: usize,
    /// Sliding window of recent prompt activation bitsets.
    prompt_history: std::collections::VecDeque<Vec<u64>>,
    /// Window size (number of prompts to keep).
    window_size: usize,
    /// Fraction of pool experts to consider replacing per rebalance (0.0-1.0).
    replacement_pct: f32,
    /// Whether dynamic rebalancing is enabled.
    rebalance_enabled: bool,
    /// Cumulative stats.
    total_evictions: u64,
    total_promotions: u64,
    total_rebalances: u64,
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
            pool_buf: None,
            pool_slot_size: 0,
            pool_num_slots: 0,
            pool_free_slots: Vec::new(),
            pool_slot_to_expert: Vec::new(),
            current_activations: Vec::new(),
            num_experts_per_layer: 0,
            prompt_history: std::collections::VecDeque::new(),
            window_size: 10,
            replacement_pct: 0.25,
            rebalance_enabled: false,
            total_evictions: 0,
            total_promotions: 0,
            total_rebalances: 0,
        }
    }

    /// Check if a specific (layer, expert) is cached in VRAM.
    fn get(&self, layer: usize, expert: usize) -> Option<&HcsCacheEntry> {
        self.cache.get(&(layer, expert))
    }

    /// Record an expert activation in the heatmap and dynamic eviction bitset.
    fn record_activation(&mut self, layer: usize, expert: usize) {
        if self.collecting {
            *self.heatmap.entry((layer, expert)).or_insert(0) += 1;
        }
        // Set bit in current prompt activation bitset
        if self.rebalance_enabled && self.num_experts_per_layer > 0 {
            let bit_idx = layer * self.num_experts_per_layer + expert;
            let word = bit_idx / 64;
            let bit = bit_idx % 64;
            if word < self.current_activations.len() {
                self.current_activations[word] |= 1u64 << bit;
            }
        }
    }

    /// Clear current prompt activation bitset (call at start of each prompt).
    fn begin_prompt(&mut self) {
        if self.rebalance_enabled {
            for w in self.current_activations.iter_mut() {
                *w = 0;
            }
        }
    }

    /// Push current prompt's activations into the sliding window.
    fn finish_prompt(&mut self) {
        if !self.rebalance_enabled || self.current_activations.is_empty() {
            return;
        }
        let snapshot = self.current_activations.clone();
        self.prompt_history.push_back(snapshot);
        while self.prompt_history.len() > self.window_size {
            self.prompt_history.pop_front();
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

// ── Cached kernel function handles ─────────────────────────────────────
// Avoids HashMap lookup per kernel call (23+ lookups per MoE forward).

#[derive(Clone)]
struct CachedKernels {
    bf16_to_fp32: cudarc::driver::CudaFunction,
    fp32_to_bf16: cudarc::driver::CudaFunction,
    rmsnorm: cudarc::driver::CudaFunction,
    fused_add_rmsnorm: cudarc::driver::CudaFunction,
    silu_mul: cudarc::driver::CudaFunction,
    sigmoid_topk: cudarc::driver::CudaFunction,
    softmax_topk: cudarc::driver::CudaFunction,
    zero_bf16: cudarc::driver::CudaFunction,
    add_bf16: cudarc::driver::CudaFunction,
    weighted_add_bf16: cudarc::driver::CudaFunction,
    scale_bf16: cudarc::driver::CudaFunction,
    embedding_lookup: cudarc::driver::CudaFunction,
    marlin_gemv_int4: cudarc::driver::CudaFunction,
    fused_silu_accum: cudarc::driver::CudaFunction,
    // v2 kernels with K-splitting for better SM occupancy
    marlin_gemv_int4_v2: cudarc::driver::CudaFunction,
    reduce_ksplits_bf16: cudarc::driver::CudaFunction,
    fused_silu_accum_v2: cudarc::driver::CudaFunction,
    reduce_ksplits_weighted_accum_bf16: cudarc::driver::CudaFunction,
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

    // Shared expert weights permanently resident in VRAM (one per MoE layer).
    // Indexed by MoE layer index. None = no shared expert or not yet pinned.
    shared_expert_vram: Vec<Option<HcsCacheEntry>>,

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

    // Expert DMA double-buffer: two contiguous buffers for ping-pong overlap.
    // While expert N computes from buf[N%2], expert N+1 DMAs into buf[(N+1)%2].
    // Each buffer holds one full expert: w13_packed | w13_scales | w2_packed | w2_scales.
    d_expert_buf: [cudarc::driver::CudaSlice<u8>; 2],
    /// Size of each contiguous expert buffer (bytes).
    expert_buf_total_size: usize,
    /// Offsets within each contiguous buffer for the 4 weight components.
    expert_buf_w13p_offset: usize,
    expert_buf_w13s_offset: usize,
    expert_buf_w2p_offset: usize,
    expert_buf_w2s_offset: usize,

    // Legacy fields kept for compatibility with existing resize_expert_buffers callers.
    // TODO: remove once all callers updated.
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

    // v2 K-split partial sum buffer: [max_k_splits * max_N] FP32
    // max_N = max(2*intermediate_size, hidden_size), max_k_splits = 8
    d_v2_partial: cudarc::driver::CudaSlice<f32>,
    num_sms: usize,

    // GQA scratch (FP32 for Q, K, V, attention output)
    d_gqa_q: cudarc::driver::CudaSlice<f32>,
    d_gqa_k: cudarc::driver::CudaSlice<f32>,
    d_gqa_v: cudarc::driver::CudaSlice<f32>,
    d_gqa_out: cudarc::driver::CudaSlice<f32>,

    // FlashDecoding tiled attention partial buffers (allocated lazily after kv_max_seq is known)
    d_gqa_tiled_o: Option<cudarc::driver::CudaSlice<f32>>,   // [num_q_heads, max_tiles, head_dim]
    d_gqa_tiled_lse: Option<cudarc::driver::CudaSlice<f32>>, // [num_q_heads, max_tiles, 2]
    gqa_tile_size: usize,
    gqa_max_tiles: usize,
    gqa_num_q_heads: usize,
    gqa_head_dim: usize,

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

    // Cached kernel function handles (populated after configure)
    kernels: Option<CachedKernels>,

    // Pre-allocated CUDA events for MoE forward (avoid create/destroy per layer)
    // [0..1] for DMA done, [2..3] for compute done on double-buffer slots
    pre_events: Option<[CudaEvent; 4]>,

    // ── Full decode step state ──

    /// Whether model norms use (1+w)*x instead of w*x.
    norm_bias_one: bool,

    /// GQA KV cache: raw device pointers to FP8 E4M3 [max_seq, kv_stride] per layer.
    /// Memory is owned by Python (PagedKVCache tensors). Prefill writes FP8 via
    /// FlashInfer, decode writes FP8 via kv_cache_write kernel. Shared buffer,
    /// no export copy needed.
    kv_k_ptrs: Vec<u64>,  // device pointers, one per layer (indexed by layer_idx)
    kv_v_ptrs: Vec<u64>,
    kv_max_seq: usize,
    kv_current_pos: usize,

    /// RoPE tables in VRAM: cos[max_seq * half_dim], sin[max_seq * half_dim]
    d_rope_cos: Option<cudarc::driver::CudaSlice<f32>>,
    d_rope_sin: Option<cudarc::driver::CudaSlice<f32>>,
    rope_half_dim: usize,

    /// Gated attention flag per GQA layer (QCN has gated GQA).
    /// Stored as BF16 scratch for gated Q rearrangement.
    d_gqa_gate_buf: Option<cudarc::driver::CudaSlice<f32>>,

    /// Max dynamic shared memory per block (bytes) for GQA attention.
    /// Default 48KB, can be increased to ~99KB via opt-in on Blackwell+.
    gqa_max_smem_bytes: u32,

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
    // Sub-MoE timing breakdown (accumulated across all layers per token, then across tokens)
    t_moe_route_sync: f64,   // time waiting for routing sync (device.synchronize in Step 4)
    t_moe_expert_loop: f64,  // total expert loop time (HCS compute + DMA + cold compute)
    t_moe_shared: f64,       // shared expert DMA + compute + gate
    t_moe_overhead: f64,     // bf16->fp32 conv, zero, scale, etc.
    // DMA instrumentation (accumulated across all layers per token, then across tokens)
    dma_bytes_total: u64,    // total bytes DMA'd (cold experts only)
    dma_call_count: u64,     // number of cuMemcpyHtoDAsync calls
    dma_cold_experts: u64,   // number of cold (DMA'd) experts
    dma_hcs_experts: u64,    // number of HCS-hit experts
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
    /// Max opt-in shared memory for GQA attention (bytes). Set during PTX load.
    gqa_max_smem_bytes: u32,
    last_decode_elapsed: f64,
    #[cfg(feature = "gpu-debug")]
    debug_stop_layer: usize,
    #[cfg(feature = "gpu-debug")]
    debug_capture_layers: bool,
    #[cfg(feature = "gpu-debug")]
    debug_layer_captures: Vec<Vec<u16>>,
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

        let mut gqa_smem_limit: u32 = 48 * 1024; // default

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

            // Opt-in to extended shared memory for gqa_attention kernel.
            // RTX 5090 supports 99KB opt-in (vs 48KB default), allowing the fast
            // shared-memory attention path for up to ~25K tokens instead of ~12K.
            if let Some(attn_func) = device.get_func(MODULE_NAME, "gqa_attention") {
                // Extract the raw CUfunction handle from cudarc's CudaFunction.
                // CudaFunction is { cu_function: *mut CUfunc_st, device: Arc<CudaDevice> }
                // Both fields are pointer-sized (8 bytes on x86_64).
                // Since #[repr(Rust)] doesn't guarantee field order, we try both offsets
                // and validate by calling cuFuncGetAttribute on each candidate.
                let struct_ptr = &attn_func as *const _ as *const u8;
                let word0: cuda_sys::CUfunction = unsafe {
                    std::ptr::read(struct_ptr as *const cuda_sys::CUfunction)
                };
                let word1: cuda_sys::CUfunction = unsafe {
                    std::ptr::read(struct_ptr.add(8) as *const cuda_sys::CUfunction)
                };
                // Validate: cuFuncGetAttribute succeeds only on a real CUfunction
                let mut dummy = 0i32;
                let w0_valid = unsafe {
                    cuda_sys::lib().cuFuncGetAttribute(
                        &mut dummy,
                        cuda_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS,
                        word0,
                    ) == cuda_sys::CUresult::CUDA_SUCCESS
                };
                let raw_fn = if w0_valid { word0 } else { word1 };
                log::info!("GQA attention: CUfunction at offset {} (w0_valid={})",
                           if w0_valid { 0 } else { 8 }, w0_valid);
                // Query device max opt-in shared memory
                let mut max_smem_i32 = 0i32;
                unsafe {
                    let _ = cuda_sys::lib().cuDeviceGetAttribute(
                        &mut max_smem_i32,
                        cuda_sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                        device_ordinal as i32,
                    );
                }
                if max_smem_i32 > 49152 {
                    let result = unsafe {
                        cuda_sys::lib().cuFuncSetAttribute(
                            raw_fn,
                            cuda_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            max_smem_i32,
                        )
                    };
                    if result == cuda_sys::CUresult::CUDA_SUCCESS {
                        gqa_smem_limit = max_smem_i32 as u32;
                        let max_tokens = (gqa_smem_limit - 128) / 4;
                        log::info!("GQA attention: opt-in shared memory = {} KB, fast path up to {} tokens",
                                   gqa_smem_limit / 1024, max_tokens);
                    } else {
                        log::warn!("GQA attention: failed to set extended shared memory ({} bytes), result={:?}",
                                   max_smem_i32, result);
                    }
                } else {
                    log::info!("GQA attention: device max shared memory = {} KB (no opt-in needed)", max_smem_i32 / 1024);
                }
            }
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
            gqa_max_smem_bytes: gqa_smem_limit,
            last_decode_elapsed: 0.0,
            #[cfg(feature = "gpu-debug")]
            debug_stop_layer: 0,
            #[cfg(feature = "gpu-debug")]
            debug_capture_layers: false,
            #[cfg(feature = "gpu-debug")]
            debug_layer_captures: Vec::new(),
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

        // Double-buffer for ping-pong expert DMA (initialized to 1 byte, resized later)
        let d_expert_buf_0 = self.device.alloc_zeros::<u8>(1)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_expert_buf_1 = self.device.alloc_zeros::<u8>(1)
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

        // v2 K-split partial sum buffer: max_k_splits=8, max_N = max(2*intermediate, hidden_size)
        let max_n_v2 = (intermediate * 2).max(hidden_size);
        let max_k_splits = 8;
        let d_v2_partial = self.device.alloc_zeros::<f32>(max_k_splits * max_n_v2)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        // Query SM count for auto K-split calculation
        let num_sms = unsafe {
            let mut dev: i32 = 0;
            cuda_sys::lib().cuCtxGetDevice(&mut dev);
            let mut count: i32 = 0;
            cuda_sys::lib().cuDeviceGetAttribute(
                &mut count,
                cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                dev,
            );
            count.max(1) as usize
        };
        log::info!("GpuDecodeStore: GPU has {} SMs", num_sms);

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
            shared_expert_vram: Vec::new(),
            apfl: None,
            hcs: None,
            d_hidden,
            d_residual,
            d_scratch,
            d_logits,
            d_fp32_scratch,
            d_expert_buf: [d_expert_buf_0, d_expert_buf_1],
            expert_buf_total_size: 0,
            expert_buf_w13p_offset: 0,
            expert_buf_w13s_offset: 0,
            expert_buf_w2p_offset: 0,
            expert_buf_w2s_offset: 0,
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
            d_v2_partial,
            num_sms,
            d_gqa_q,
            d_gqa_k,
            d_gqa_v,
            d_gqa_out,
            d_gqa_tiled_o: None,
            d_gqa_tiled_lse: None,
            gqa_tile_size: 0,
            gqa_max_tiles: 0,
            gqa_num_q_heads: 0,
            gqa_head_dim: 0,
            d_la_qkvz,
            d_la_ba,
            d_la_conv_out,
            d_la_recur_out,
            d_la_gated_out,
            h_topk_ids: vec![0i32; max_experts_per_tok],
            h_topk_weights: vec![0.0f32; max_experts_per_tok],
            h_logits: vec![0.0f32; vocab_size],
            kernels: None,
            pre_events: None,
            norm_bias_one: false,
            kv_k_ptrs: Vec::new(),
            kv_v_ptrs: Vec::new(),
            kv_max_seq: 0,
            kv_current_pos: 0,
            d_rope_cos: None,
            d_rope_sin: None,
            rope_half_dim: 0,
            d_gqa_gate_buf: None,
            gqa_max_smem_bytes: self.gqa_max_smem_bytes,
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
            t_moe_route_sync: 0.0,
            t_moe_expert_loop: 0.0,
            t_moe_shared: 0.0,
            t_moe_overhead: 0.0,
            dma_bytes_total: 0,
            dma_call_count: 0,
            dma_cold_experts: 0,
            dma_hcs_experts: 0,
        }));

        // Cache kernel function handles (avoid HashMap lookup per call)
        if self.kernels_loaded {
            let get = |name: &str| -> PyResult<cudarc::driver::CudaFunction> {
                self.device.get_func(MODULE_NAME, name)
                    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                        format!("Kernel '{}' not found", name)))
            };
            let kernels = CachedKernels {
                bf16_to_fp32: get("bf16_to_fp32")?,
                fp32_to_bf16: get("fp32_to_bf16")?,
                rmsnorm: get("rmsnorm")?,
                fused_add_rmsnorm: get("fused_add_rmsnorm")?,
                silu_mul: get("silu_mul")?,
                sigmoid_topk: get("sigmoid_topk")?,
                softmax_topk: get("softmax_topk")?,
                zero_bf16: get("zero_bf16")?,
                add_bf16: get("add_bf16")?,
                weighted_add_bf16: get("weighted_add_bf16")?,
                scale_bf16: get("scale_bf16")?,
                embedding_lookup: get("embedding_lookup")?,
                marlin_gemv_int4: get("marlin_gemv_int4")?,
                fused_silu_accum: get("marlin_gemv_int4_fused_silu_accum")?,
                marlin_gemv_int4_v2: get("marlin_gemv_int4_v2")?,
                reduce_ksplits_bf16: get("reduce_ksplits_bf16")?,
                fused_silu_accum_v2: get("marlin_gemv_int4_fused_silu_accum_v2")?,
                reduce_ksplits_weighted_accum_bf16: get("reduce_ksplits_weighted_accum_bf16")?,
            };
            self.graph.as_mut().unwrap().kernels = Some(kernels);
            log::info!("GpuDecodeStore: cached 18 kernel function handles");
        }

        // Pre-allocate CUDA events (reuse across MoE forward calls)
        {
            let mut raw_events = [std::ptr::null_mut(); 4];
            unsafe {
                let flags = cuda_sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32;
                for e in raw_events.iter_mut() {
                    let err = cuda_sys::lib().cuEventCreate(e, flags);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            format!("cuEventCreate: {:?}", err)));
                    }
                }
            }
            self.graph.as_mut().unwrap().pre_events = Some([
                CudaEvent(raw_events[0]),
                CudaEvent(raw_events[1]),
                CudaEvent(raw_events[2]),
                CudaEvent(raw_events[3]),
            ]);
            log::info!("GpuDecodeStore: pre-allocated 4 CUDA events");
        }

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
        // Legacy 4-buffer allocation (kept for compatibility)
        graph.d_expert_buf_a0 = self.device.alloc_zeros::<u8>(expert_size_bytes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        graph.d_expert_buf_b0 = self.device.alloc_zeros::<u8>(expert_size_bytes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        graph.d_expert_buf_a1 = self.device.alloc_zeros::<u8>(expert_size_bytes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        graph.d_expert_buf_b1 = self.device.alloc_zeros::<u8>(expert_size_bytes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        graph.expert_buf_size = expert_size_bytes;

        // Compute proper double-buffer layout from the first registered MoE layer's expert sizes.
        // All experts in a model have identical weight dimensions.
        if let Some(first_moe) = graph.moe_layers.iter().find_map(|m| m.as_ref()) {
            let e = &first_moe.experts[0];
            let align = 256usize; // CUDA DMA alignment
            let w13p_aligned = (e.w13_packed_bytes + align - 1) & !(align - 1);
            let w13s_aligned = (e.w13_scales_bytes + align - 1) & !(align - 1);
            let w2p_aligned = (e.w2_packed_bytes + align - 1) & !(align - 1);
            let w2s_aligned = (e.w2_scales_bytes + align - 1) & !(align - 1);
            let total = w13p_aligned + w13s_aligned + w2p_aligned + w2s_aligned;

            graph.expert_buf_w13p_offset = 0;
            graph.expert_buf_w13s_offset = w13p_aligned;
            graph.expert_buf_w2p_offset = w13p_aligned + w13s_aligned;
            graph.expert_buf_w2s_offset = w13p_aligned + w13s_aligned + w2p_aligned;
            graph.expert_buf_total_size = total;

            graph.d_expert_buf[0] = self.device.alloc_zeros::<u8>(total)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            graph.d_expert_buf[1] = self.device.alloc_zeros::<u8>(total)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            log::info!(
                "GpuDecodeStore: double-buffer 2x {:.1} KB = {:.1} MB (w13p={}, w13s={}, w2p={}, w2s={})",
                total as f64 / 1024.0,
                total as f64 * 2.0 / (1024.0 * 1024.0),
                e.w13_packed_bytes, e.w13_scales_bytes,
                e.w2_packed_bytes, e.w2_scales_bytes,
            );
        }

        log::info!("GpuDecodeStore: expert buffers 4x {} bytes ({:.1} MB total)",
                   expert_size_bytes, expert_size_bytes as f64 * 4.0 / (1024.0 * 1024.0));
        Ok(())
    }

    fn set_timing(&mut self, enabled: bool) -> PyResult<()> {
        if let Some(ref mut graph) = self.graph {
            graph.timing_enabled = enabled;
            if enabled {
                // Reset accumulators
                graph.timing_step_count = 0;
                graph.t_total = 0.0;
                graph.t_norm = 0.0;
                graph.t_attn = 0.0;
                graph.t_route = 0.0;
                graph.t_expert_dma = 0.0;
                graph.t_expert_compute = 0.0;
                graph.t_shared = 0.0;
                graph.t_dense_mlp = 0.0;
                graph.t_lm_head = 0.0;
                graph.t_moe_route_sync = 0.0;
                graph.t_moe_expert_loop = 0.0;
                graph.t_moe_shared = 0.0;
                graph.t_moe_overhead = 0.0;
                graph.dma_bytes_total = 0;
                graph.dma_call_count = 0;
                graph.dma_cold_experts = 0;
                graph.dma_hcs_experts = 0;
            }
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

    /// Set the shared expert gate weight ID for a MoE layer.
    /// Called from Python after setup_from_engine to wire in the sigmoid gate.
    fn set_moe_shared_gate_wid(&mut self, layer_idx: usize, wid: usize) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let moe = graph.moe_layers.get_mut(layer_idx)
            .and_then(|m| m.as_mut())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                format!("MoE layer {} not registered", layer_idx)))?;
        moe.shared_gate_wid = Some(wid);
        log::info!("Set shared_gate_wid={} for MoE layer {}", wid, layer_idx);
        Ok(())
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

    #[cfg(feature = "gpu-debug")]
    fn set_debug_stop_layer(&mut self, n: usize) {
        self.debug_stop_layer = n;
    }

    #[cfg(feature = "gpu-debug")]
    fn set_debug_capture_layers(&mut self, enable: bool) {
        self.debug_capture_layers = enable;
        self.debug_layer_captures.clear();
    }

    #[cfg(feature = "gpu-debug")]
    fn download_layer_captures(&self) -> PyResult<Vec<Vec<u16>>> {
        Ok(self.debug_layer_captures.clone())
    }

    /// Download BF16 residual state from GPU d_residual buffer (for testing).
    fn download_residual_bf16(&self) -> PyResult<Vec<u16>> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let mut out = vec![0u16; graph.hidden_size];
        unsafe {
            let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                *graph.d_residual.device_ptr(),
                out.len() * 2);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("D2H residual: {:?}", err)));
            }
        }
        Ok(out)
    }

    /// Download FP32 data from an LA intermediate buffer (for testing/debugging).
    /// buffer_name: "qkvz", "ba", "conv_out", "recur_out", "gated_out"
    /// size: number of f32 elements to read
    #[pyo3(signature = (buffer_name, size))]
    fn download_la_buffer_f32(&self, buffer_name: &str, size: usize) -> PyResult<Vec<f32>> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let ptr = match buffer_name {
            "qkvz" => *graph.d_la_qkvz.device_ptr(),
            "ba" => *graph.d_la_ba.device_ptr(),
            "conv_out" => *graph.d_la_conv_out.device_ptr(),
            "recur_out" => *graph.d_la_recur_out.device_ptr(),
            "gated_out" => *graph.d_la_gated_out.device_ptr(),
            _ => return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Unknown LA buffer: {}", buffer_name))),
        };
        let mut out = vec![0.0f32; size];
        unsafe {
            let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                ptr,
                size * 4);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("D2H LA buffer '{}': {:?}", buffer_name, err)));
            }
        }
        Ok(out)
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

    /// Debug: Run a single expert's w13 GEMV and return gate_up as BF16 u16 list.
    /// Does NOT do routing/accumulation — just DMA one expert and run w13 GEMV.
    /// Returns: Vec<u16> of length 2*intermediate_size (gate_up BF16).
    #[pyo3(signature = (layer_idx, expert_id))]
    fn test_single_expert_w13(&mut self, layer_idx: usize, expert_id: usize) -> PyResult<Vec<u16>> {
        let graph = self.graph.take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;

        let result = self.test_single_expert_w13_impl(&graph, layer_idx, expert_id);

        self.graph = Some(graph);
        result
    }

    /// Debug: CPU reference dequant + matmul for a single expert's w13.
    /// Reads the Marlin-packed weights from system RAM, dequantizes with inverse
    /// perm + scales, and does FP32 matmul against d_hidden.
    /// Returns: Vec<f32> of length 2*intermediate_size.
    #[pyo3(signature = (layer_idx, expert_id))]
    fn test_cpu_reference_w13(&self, layer_idx: usize, expert_id: usize) -> PyResult<Vec<f32>> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;

        let moe = graph.moe_layers.get(layer_idx)
            .and_then(|m| m.as_ref())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                format!("MoE layer {} not registered", layer_idx)))?;

        let expert = &moe.experts[expert_id];
        let hs = graph.hidden_size;
        let intermediate = graph.intermediate_size;
        let gs = graph.group_size;
        let n = 2 * intermediate;  // w13 output dim
        let k = hs;                // w13 input dim

        // Download current d_hidden to CPU
        let mut hidden_bf16 = vec![0u16; hs];
        unsafe {
            let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                hidden_bf16.as_mut_ptr() as *mut std::ffi::c_void,
                *graph.d_hidden.device_ptr(),
                hs * 2);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("D2H hidden: {:?}", err)));
            }
        }
        let hidden_f32: Vec<f32> = hidden_bf16.iter().map(|&bits| {
            let full = (bits as u32) << 16;
            f32::from_bits(full)
        }).collect();

        // Get w13 weight pointers from expert data (system RAM)
        let w13_packed_ptr = expert.w13_packed_ptr as *const u32;
        let w13_scales_ptr = expert.w13_scales_ptr as *const u16;
        let k_tiles = k / 16;
        let out_cols = 2 * n;  // u32 cols in packed
        let num_groups_k = k / gs;

        // Read packed weights and scales from system RAM
        let packed_len = k_tiles * out_cols;
        let scales_len = num_groups_k * n;
        let packed: Vec<u32> = unsafe {
            std::slice::from_raw_parts(w13_packed_ptr, packed_len).to_vec()
        };
        let scales_raw: Vec<u16> = unsafe {
            std::slice::from_raw_parts(w13_scales_ptr, scales_len).to_vec()
        };

        // Generate inverse perm tables
        let fwd_perm = crate::weights::marlin::generate_weight_perm_int4();
        let (fwd_scale, _) = crate::weights::marlin::generate_scale_perms();
        let mut inv_wperm = [0usize; 1024];
        for (i, &src) in fwd_perm.iter().enumerate() {
            inv_wperm[src] = i;
        }
        let mut inv_sperm = [0usize; 64];
        for (i, &src) in fwd_scale.iter().enumerate() {
            inv_sperm[src] = i;
        }

        // CPU GEMV: for each output n, accumulate over k
        let mut output = vec![0.0f32; n];
        let row_len = n * 16;

        for out_n in 0..n {
            let n_tile = out_n / 16;
            let tn = out_n % 16;
            let mut acc = 0.0f32;

            for kt in 0..k_tiles {
                for tk in 0..16 {
                    let kk = kt * 16 + tk;

                    // Scale lookup (same logic as kernel)
                    let sg = kk / gs;
                    let scale_flat = sg * n + out_n;
                    let schunk = scale_flat / 64;
                    let slocal = scale_flat % 64;
                    let sperm_pos = schunk * 64 + inv_sperm[slocal];
                    let scale_bits = scales_raw[sperm_pos];
                    let scale = f32::from_bits((scale_bits as u32) << 16);

                    // Weight lookup (same logic as kernel)
                    let tile_pos = n_tile * 256 + tk * 16 + tn;
                    let chunk = tile_pos / 1024;
                    let local_idx = tile_pos % 1024;
                    let perm_pos = chunk * 1024 + inv_wperm[local_idx];
                    let u32_col = perm_pos / 8;
                    let nibble = perm_pos % 8;
                    let word = packed[kt * out_cols + u32_col];
                    let raw = ((word >> (nibble * 4)) & 0xF) as i32;
                    let w_val = (raw - 8) as f32;

                    acc += w_val * scale * hidden_f32[kk];
                }
            }
            output[out_n] = acc;
        }

        Ok(output)
    }

    /// Download d_expert_gate_up buffer as BF16 u16 values.
    fn download_gate_up_bf16(&self) -> PyResult<Vec<u16>> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let size = graph.intermediate_size * 2;
        let mut out = vec![0u16; size];
        unsafe {
            let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                *graph.d_expert_gate_up.device_ptr(),
                size * 2);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("D2H: {:?}", err)));
            }
        }
        Ok(out)
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

    /// Register external VRAM pointers as HCS entries (no allocation).
    /// Used to share Python HCS buffers with Rust decode without copying.
    /// w13p/w13s/w2p/w2s are raw GPU pointers (from tensor.data_ptr()).
    #[pyo3(signature = (layer_idx, expert_idx, w13p, w13s, w2p, w2s))]
    fn hcs_register_external(
        &mut self, layer_idx: usize, expert_idx: usize,
        w13p: u64, w13s: u64, w2p: u64, w2s: u64,
    ) -> PyResult<bool> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let hcs = graph.hcs.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call init_hcs first"))?;
        if hcs.cache.contains_key(&(layer_idx, expert_idx)) {
            return Ok(false);
        }
        let entry = HcsCacheEntry {
            d_buf: None,
            w13_packed_offset: 0, w13_packed_size: 0,
            w13_scales_offset: 0, w13_scales_size: 0,
            w2_packed_offset: 0, w2_packed_size: 0,
            w2_scales_offset: 0, w2_scales_size: 0,
            ext_w13_packed: w13p, ext_w13_scales: w13s,
            ext_w2_packed: w2p, ext_w2_scales: w2s,
            pool_slot: None,
        };
        hcs.num_cached += 1;
        hcs.cache.insert((layer_idx, expert_idx), entry);
        Ok(true)
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

    /// Initialize pool-based HCS with dynamic eviction.
    ///
    /// Allocates a contiguous VRAM pool and fills it with the hottest experts
    /// from the provided ranking (list of (layer_idx, expert_idx) pairs, sorted
    /// hottest-first). Enables sliding-window activation tracking for between-prompt
    /// rebalancing.
    ///
    /// Args:
    ///   ranking: list of (layer_idx, expert_idx) tuples, hottest first
    ///   budget_mb: VRAM budget for pool (0 = auto from free VRAM)
    ///   headroom_mb: VRAM to keep free (only used when budget_mb=0)
    ///   window_size: number of recent prompts to track (default 10)
    ///   replacement_pct: fraction of pool to consider replacing per rebalance (default 25)
    #[pyo3(signature = (ranking, budget_mb=0, headroom_mb=500, window_size=10, replacement_pct=25))]
    fn hcs_pool_init(
        &mut self,
        ranking: Vec<(usize, usize)>,
        budget_mb: usize,
        headroom_mb: usize,
        window_size: usize,
        replacement_pct: usize,
    ) -> PyResult<String> {
        self.hcs_pool_init_internal(ranking, budget_mb, headroom_mb, window_size, replacement_pct)
    }

    /// Get dynamic HCS eviction statistics.
    fn hcs_dynamic_stats(&self) -> PyResult<String> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let hcs = graph.hcs.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call init_hcs first"))?;
        Ok(format!(
            "HCS dynamic: pool={}/{} slots, window={}/{} prompts, rebalances={}, evictions={}, promotions={}, hit_rate={:.1}%",
            hcs.pool_num_slots - hcs.pool_free_slots.len(), hcs.pool_num_slots,
            hcs.prompt_history.len(), hcs.window_size,
            hcs.total_rebalances, hcs.total_evictions, hcs.total_promotions,
            hcs.hit_rate() * 100.0,
        ))
    }

    /// Full HCS end-to-end test: load model, pin all experts, run MoE, compare.
    #[pyo3(signature = (model_dir, num_tokens=10))]
    fn test_hcs_e2e(&mut self, model_dir: &str, num_tokens: usize) -> PyResult<String> {
        self.test_hcs_e2e_internal(model_dir, num_tokens)
    }

    /// Optimization pass benchmark: measures MoE forward with shared experts.
    /// Runs with and without shared expert VRAM residency to measure delta.
    #[pyo3(signature = (model_dir, num_tokens=5))]
    fn bench_shared_expert_residency(&mut self, model_dir: &str, num_tokens: usize) -> PyResult<String> {
        self.bench_shared_expert_residency_internal(model_dir, num_tokens)
    }

    /// Benchmark raw PCIe DMA bandwidth + pure HCS compute speed.
    /// Tests: (1) H2D DMA at various transfer sizes, (2) pure GEMV compute
    /// on VRAM-resident experts, (3) full MoE forward breakdown.
    #[pyo3(signature = (model_dir, num_tokens=10))]
    fn bench_pcie_and_compute(&mut self, model_dir: &str, num_tokens: usize) -> PyResult<String> {
        self.bench_pcie_and_compute_internal(model_dir, num_tokens)
    }

    // ── Full GPU Decode: attention, KV cache, decode_step, generate_stream ──

    /// Register a Linear Attention layer's weights and state pointers for GPU decode.
    ///
    /// All pointers are to VRAM-resident data (BF16 weights, FP32 state).
    /// Called once during setup from Python after model load.
    #[pyo3(signature = (layer_idx,
                        input_norm_ptr, input_norm_size,
                        post_attn_norm_ptr, post_attn_norm_size,
                        in_proj_qkvz_wid, in_proj_ba_wid, out_proj_wid,
                        conv_weight_ptr, a_log_ptr, dt_bias_ptr, norm_weight_ptr,
                        conv_state_ptr, recur_state_ptr,
                        nk, nv, dk, dv, hr, kernel_dim, conv_dim, scale))]
    #[allow(clippy::too_many_arguments)]
    fn register_la_layer(
        &mut self,
        layer_idx: usize,
        input_norm_ptr: usize, input_norm_size: usize,
        post_attn_norm_ptr: usize, post_attn_norm_size: usize,
        in_proj_qkvz_wid: usize, in_proj_ba_wid: usize, out_proj_wid: usize,
        conv_weight_ptr: usize, a_log_ptr: usize, dt_bias_ptr: usize,
        norm_weight_ptr: usize,
        conv_state_ptr: usize, recur_state_ptr: usize,
        nk: usize, nv: usize, dk: usize, dv: usize,
        hr: usize, kernel_dim: usize, conv_dim: usize, scale: f32,
    ) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        while graph.layers.len() <= layer_idx {
            graph.layers.push(GpuDecodeLayer {
                input_norm_ptr: 0,
                input_norm_size: 0,
                post_attn_norm_ptr: 0,
                post_attn_norm_size: 0,
                attn: GpuAttnConfig::GQA {
                    q_proj: 0, k_proj: 0, v_proj: 0, o_proj: 0, fused_qkv: None,
                    num_heads: 0, num_kv_heads: 0, head_dim: 0, sm_scale: 0.0,
                    q_norm_ptr: 0, k_norm_ptr: 0, gated: false,
                },
                mlp: GpuMlpConfig::None,
            });
        }
        graph.layers[layer_idx].input_norm_ptr = input_norm_ptr as u64;
        graph.layers[layer_idx].input_norm_size = input_norm_size;
        graph.layers[layer_idx].post_attn_norm_ptr = post_attn_norm_ptr as u64;
        graph.layers[layer_idx].post_attn_norm_size = post_attn_norm_size;
        graph.layers[layer_idx].attn = GpuAttnConfig::LinearAttention {
            in_proj_qkvz: in_proj_qkvz_wid,
            in_proj_ba: in_proj_ba_wid,
            out_proj: out_proj_wid,
            conv_weight_ptr: conv_weight_ptr as u64,
            a_log_ptr: a_log_ptr as u64,
            dt_bias_ptr: dt_bias_ptr as u64,
            norm_weight_ptr: norm_weight_ptr as u64,
            nk, nv, dk, dv, hr, kernel_dim, conv_dim, scale,
            conv_state_ptr: conv_state_ptr as u64,
            recur_state_ptr: recur_state_ptr as u64,
        };
        log::info!("GpuDecodeStore: registered LA layer {} (conv_dim={}, nk={}, nv={}), total_layers={}",
            layer_idx, conv_dim, nk, nv, graph.layers.len());
        Ok(())
    }

    /// Register a GQA layer's weights and config for GPU decode.
    #[pyo3(signature = (layer_idx,
                        input_norm_ptr, input_norm_size,
                        post_attn_norm_ptr, post_attn_norm_size,
                        q_proj_wid, k_proj_wid, v_proj_wid, o_proj_wid,
                        fused_qkv_wid,
                        num_heads, num_kv_heads, head_dim, sm_scale,
                        q_norm_ptr=0, k_norm_ptr=0, gated=false))]
    #[allow(clippy::too_many_arguments)]
    fn register_gqa_layer(
        &mut self,
        layer_idx: usize,
        input_norm_ptr: usize, input_norm_size: usize,
        post_attn_norm_ptr: usize, post_attn_norm_size: usize,
        q_proj_wid: usize, k_proj_wid: usize, v_proj_wid: usize, o_proj_wid: usize,
        fused_qkv_wid: Option<usize>,
        num_heads: usize, num_kv_heads: usize, head_dim: usize, sm_scale: f32,
        q_norm_ptr: usize, k_norm_ptr: usize, gated: bool,
    ) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        while graph.layers.len() <= layer_idx {
            graph.layers.push(GpuDecodeLayer {
                input_norm_ptr: 0,
                input_norm_size: 0,
                post_attn_norm_ptr: 0,
                post_attn_norm_size: 0,
                attn: GpuAttnConfig::GQA {
                    q_proj: 0, k_proj: 0, v_proj: 0, o_proj: 0, fused_qkv: None,
                    num_heads: 0, num_kv_heads: 0, head_dim: 0, sm_scale: 0.0,
                    q_norm_ptr: 0, k_norm_ptr: 0, gated: false,
                },
                mlp: GpuMlpConfig::None,
            });
        }
        graph.layers[layer_idx].input_norm_ptr = input_norm_ptr as u64;
        graph.layers[layer_idx].input_norm_size = input_norm_size;
        graph.layers[layer_idx].post_attn_norm_ptr = post_attn_norm_ptr as u64;
        graph.layers[layer_idx].post_attn_norm_size = post_attn_norm_size;
        graph.layers[layer_idx].attn = GpuAttnConfig::GQA {
            q_proj: q_proj_wid,
            k_proj: k_proj_wid,
            v_proj: v_proj_wid,
            o_proj: o_proj_wid,
            fused_qkv: fused_qkv_wid,
            num_heads,
            num_kv_heads,
            head_dim,
            sm_scale,
            q_norm_ptr: q_norm_ptr as u64,
            k_norm_ptr: k_norm_ptr as u64,
            gated,
        };
        log::info!("GpuDecodeStore: registered GQA layer {} (heads={}, kv_heads={}, hd={}), total_layers={}",
            layer_idx, num_heads, num_kv_heads, head_dim, graph.layers.len());
        Ok(())
    }

    /// Register MLP config for a layer (MoE, Dense, or None).
    /// For MoE layers, the expert data should already be registered via register_moe_layer.
    #[pyo3(signature = (layer_idx, mlp_type, gate_proj_wid=None, up_proj_wid=None, down_proj_wid=None))]
    fn register_mlp(
        &mut self,
        layer_idx: usize,
        mlp_type: &str,
        gate_proj_wid: Option<usize>,
        up_proj_wid: Option<usize>,
        down_proj_wid: Option<usize>,
    ) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        if layer_idx >= graph.layers.len() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Layer {} not registered", layer_idx)));
        }
        match mlp_type {
            "moe" => {
                // MoE config is read from graph.moe_layers during decode_step
                graph.layers[layer_idx].mlp = GpuMlpConfig::None; // Placeholder; MoE data is in moe_layers
            }
            "dense" => {
                graph.layers[layer_idx].mlp = GpuMlpConfig::Dense {
                    gate_proj: gate_proj_wid.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("gate_proj_wid required"))?,
                    up_proj: up_proj_wid.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("up_proj_wid required"))?,
                    down_proj: down_proj_wid.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("down_proj_wid required"))?,
                };
            }
            _ => {
                graph.layers[layer_idx].mlp = GpuMlpConfig::None;
            }
        }
        Ok(())
    }

    /// Set up RoPE tables in VRAM for GQA attention.
    /// cos_ptr, sin_ptr: device pointers to FP32 [max_seq, half_dim] on GPU.
    #[pyo3(signature = (cos_ptr, sin_ptr, half_dim, max_seq))]
    fn set_rope_tables(
        &mut self,
        cos_ptr: usize,
        sin_ptr: usize,
        half_dim: usize,
        max_seq: usize,
    ) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        // Copy from the PyTorch tensors into our own VRAM allocations
        let total = max_seq * half_dim;
        let d_cos = self.device.alloc_zeros::<f32>(total)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_sin = self.device.alloc_zeros::<f32>(total)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        // D2D copy
        unsafe {
            let err = cuda_sys::lib().cuMemcpyDtoD_v2(
                *d_cos.device_ptr(), cos_ptr as u64, total * 4);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("D2D rope cos: {:?}", err)));
            }
            let err = cuda_sys::lib().cuMemcpyDtoD_v2(
                *d_sin.device_ptr(), sin_ptr as u64, total * 4);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("D2D rope sin: {:?}", err)));
            }
        }
        graph.d_rope_cos = Some(d_cos);
        graph.d_rope_sin = Some(d_sin);
        graph.rope_half_dim = half_dim;
        log::info!("GpuDecodeStore: RoPE tables set ({} half_dim, {} max_seq)", half_dim, max_seq);
        Ok(())
    }

    /// Register shared FP8 KV cache pointers from Python's PagedKVCache.
    /// Python owns the memory (FP8 E4M3 contiguous tensors). Both FlashInfer
    /// prefill and Rust decode read/write the same buffers — no export copy.
    ///
    /// kv_ptrs: list of (layer_idx, k_data_ptr, v_data_ptr) device pointers.
    /// max_seq: maximum sequence length the buffers can hold.
    #[pyo3(signature = (kv_ptrs, max_seq))]
    fn set_kv_cache_ptrs(
        &mut self,
        kv_ptrs: Vec<(usize, usize, usize)>,
        max_seq: usize,
    ) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let num_layers = graph.layers.len();
        graph.kv_k_ptrs = vec![0u64; num_layers];
        graph.kv_v_ptrs = vec![0u64; num_layers];
        graph.kv_max_seq = max_seq;
        let mut registered = 0usize;
        for (layer_idx, k_ptr, v_ptr) in kv_ptrs {
            if layer_idx >= num_layers {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Layer {} out of range ({})", layer_idx, num_layers)));
            }
            graph.kv_k_ptrs[layer_idx] = k_ptr as u64;
            graph.kv_v_ptrs[layer_idx] = v_ptr as u64;
            registered += 1;
        }
        log::info!("GpuDecodeStore: KV cache shared FP8 pointers set ({} GQA layers, max_seq={})",
            registered, max_seq);

        // Allocate FlashDecoding tiled attention buffers.
        // Find max num_q_heads and head_dim across all GQA layers.
        let mut max_nh: usize = 0;
        let mut max_hd: usize = 0;
        for layer in &graph.layers {
            if let GpuAttnConfig::GQA { num_heads, head_dim, .. } = &layer.attn {
                max_nh = max_nh.max(*num_heads);
                max_hd = max_hd.max(*head_dim);
            }
        }
        if max_nh > 0 && max_hd > 0 {
            let tile_size: usize = 256;
            let max_tiles = (max_seq + tile_size - 1) / tile_size;
            let partial_o_size = max_nh * max_tiles * max_hd;  // floats
            let partial_lse_size = max_nh * max_tiles * 2;     // floats
            let d_tiled_o = self.device.alloc_zeros::<f32>(partial_o_size)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let d_tiled_lse = self.device.alloc_zeros::<f32>(partial_lse_size)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let o_mb = (partial_o_size * 4) as f64 / (1024.0 * 1024.0);
            let lse_kb = (partial_lse_size * 4) as f64 / 1024.0;
            log::info!("GpuDecodeStore: tiled GQA buffers allocated (tile_size={}, max_tiles={}, \
                        partial_o={:.1} MB, partial_lse={:.1} KB)",
                       tile_size, max_tiles, o_mb, lse_kb);
            graph.d_gqa_tiled_o = Some(d_tiled_o);
            graph.d_gqa_tiled_lse = Some(d_tiled_lse);
            graph.gqa_tile_size = tile_size;
            graph.gqa_max_tiles = max_tiles;
            graph.gqa_num_q_heads = max_nh;
            graph.gqa_head_dim = max_hd;
        }

        Ok(())
    }

    /// Set KV cache position after prefill. Called once per request.
    /// No data copy needed — prefill already wrote into the shared buffer.
    #[pyo3(signature = (seq_len))]
    fn set_kv_position(&mut self, seq_len: usize) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        if seq_len > graph.kv_max_seq {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("seq_len {} exceeds KV max_seq {}", seq_len, graph.kv_max_seq)));
        }
        graph.kv_current_pos = seq_len;
        Ok(())
    }

    /// Set norm_bias_one flag (Qwen3-Next uses (1+w)*x norms).
    fn set_norm_bias_one(&mut self, flag: bool) -> PyResult<()> {
        if let Some(ref mut graph) = self.graph {
            graph.norm_bias_one = flag;
        }
        Ok(())
    }

    /// Get self pointer for Rust-side access (same pattern as CpuDecodeStore).
    fn gpu_store_addr(&self) -> usize {
        self as *const GpuDecodeStore as usize
    }

    /// Run a single GPU decode step (for testing). Fills d_hidden and h_logits.
    fn py_gpu_decode_step(&mut self, token_id: usize, position: usize) -> PyResult<()> {
        self.gpu_decode_step(token_id, position)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Batch GPU decode: generate tokens without streaming. Returns list of token IDs.
    /// Used by benchmark engine path and decode warmup.
    #[pyo3(signature = (first_token, start_position, max_tokens, temperature, top_k, top_p, stop_ids, presence_penalty=0.0))]
    fn gpu_generate_batch(
        &mut self,
        first_token: usize,
        start_position: usize,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        stop_ids: Vec<usize>,
        presence_penalty: f32,
    ) -> PyResult<Vec<usize>> {
        let vocab_size = match self.graph.as_ref() {
            Some(g) => g.vocab_size,
            None => return Err(pyo3::exceptions::PyRuntimeError::new_err("graph not configured")),
        };

        let stop_set: std::collections::HashSet<usize> = stop_ids.iter().copied().collect();

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

        let mut next_token = first_token;
        let mut tokens = Vec::with_capacity(max_tokens);
        let mut seen_tokens: std::collections::HashSet<usize> = std::collections::HashSet::new();
        seen_tokens.insert(first_token);

        let decode_start = std::time::Instant::now();

        for step in 0..max_tokens {
            let pos = start_position + step;
            self.gpu_decode_step(next_token, pos)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("gpu_decode_step error: {}", e)))?;

            let logits = &mut self.graph.as_mut().unwrap().h_logits;
            if presence_penalty != 0.0 {
                for &tok in &seen_tokens {
                    if tok < vocab_size {
                        logits[tok] -= presence_penalty;
                    }
                }
            }

            next_token = crate::decode::sample_from_logits_pub(
                logits, vocab_size, temperature, top_k, top_p, &mut rng_next);
            seen_tokens.insert(next_token);
            tokens.push(next_token);

            if stop_set.contains(&next_token) {
                break;
            }
        }

        let elapsed = decode_start.elapsed().as_secs_f64();
        if !tokens.is_empty() {
            let tps = tokens.len() as f64 / elapsed;
            log::info!("gpu_generate_batch: {} tokens in {:.2}s ({:.1} tok/s)",
                tokens.len(), elapsed, tps);
        }
        self.last_decode_elapsed = elapsed;

        Ok(tokens)
    }

    /// Get elapsed time of last decode run (seconds).
    #[getter]
    fn last_decode_elapsed_s(&self) -> f64 {
        self.last_decode_elapsed
    }

    /// Get max sequence length of the KV cache.
    #[getter]
    fn kv_max_seq(&self) -> usize {
        self.graph.as_ref().map_or(0, |g| g.kv_max_seq)
    }
}

// ── Pure-Rust methods for GPU decode (no PyO3, used by Rust HTTP server) ──

impl GpuDecodeStore {
    /// Full GPU decode step: embedding → layer loop → final norm → LM head → logits.
    ///
    /// All computation on GPU via CUDA kernels. Zero Python, zero GIL.
    /// The MoE forward uses the fast Marlin GEMV path with HCS.
    pub fn gpu_decode_step(
        &mut self,
        token_id: usize,
        position: usize,
    ) -> Result<(), String> {
        if !self.kernels_loaded {
            return Err("Decode kernels not loaded".to_string());
        }

        let mut graph = self.graph.take()
            .ok_or_else(|| "Call configure first".to_string())?;

        let result = self.gpu_decode_step_with_graph(&mut graph, token_id, position);

        self.graph = Some(graph);
        result
    }

    fn gpu_decode_step_with_graph(
        &mut self,
        graph: &mut GpuDecodeGraph,
        token_id: usize,
        position: usize,
    ) -> Result<(), String> {
        use cudarc::driver::LaunchConfig;
        use std::time::Instant;

        let hs = graph.hidden_size;
        let eps = graph.eps;
        let timing = graph.timing_enabled;

        // Validate position doesn't exceed KV cache
        if position >= graph.kv_max_seq {
            return Err(format!(
                "position {} exceeds kv_max_seq {} — KV cache too small for this decode",
                position, graph.kv_max_seq));
        }

        // Timing: sync and take initial timestamp
        let t0 = if timing {
            self.device.synchronize().map_err(|e| format!("timing sync: {:?}", e))?;
            Instant::now()
        } else {
            Instant::now() // cheap, no sync
        };

        // Clone kernel handles to avoid holding an immutable borrow on graph
        // (moe_forward_with_graph needs &mut graph)
        let k = graph.kernels.as_ref()
            .ok_or_else(|| "Kernels not cached".to_string())?
            .clone();

        #[cfg(feature = "gpu-debug")]
        let debug_peek_bf16 = |label: &str, ptr: u64, n: usize| {
            let mut buf = vec![0u16; n];
            unsafe {
                let _ = cuda_sys::lib().cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut std::ffi::c_void,
                    ptr, n * 2);
            }
            let vals: Vec<f32> = buf.iter().map(|&b| {
                let bits = (b as u32) << 16;
                f32::from_bits(bits)
            }).collect();
            log::info!("DBG {} [{:.4}, {:.4}, {:.4}, {:.4}]", label, vals[0], vals[1], vals[2], vals[3]);
        };
        #[cfg(feature = "gpu-debug")]
        let debug_peek_f32 = |label: &str, ptr: u64, n: usize| {
            let mut buf = vec![0f32; n];
            unsafe {
                let _ = cuda_sys::lib().cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut std::ffi::c_void,
                    ptr, n * 4);
            }
            log::info!("DBG {} [{:.4}, {:.4}, {:.4}, {:.4}]", label, buf[0], buf[1], buf[2], buf[3]);
        };

        // ── 1. Embedding lookup ──
        #[cfg(feature = "gpu-debug")]
        log::info!("gpu_decode_step: token={}, pos={}", token_id, position);
        {
            let threads = 256u32;
            let blocks = ((hs as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                k.embedding_lookup.clone().launch(cfg, (
                    *graph.d_hidden.device_ptr(),
                    graph.embedding_ptr,
                    token_id as i32,
                    hs as i32,
                )).map_err(|e| format!("embedding_lookup: {:?}", e))?;
            }
        }

        #[cfg(feature = "gpu-debug")]
        {
            self.device.synchronize().map_err(|e| format!("sync after emb: {:?}", e))?;
            debug_peek_bf16("after_embedding d_hidden", *graph.d_hidden.device_ptr(), 4);
        }

        let mut first_residual = true;
        let num_layers = graph.layers.len();
        let mut gqa_cache_idx = 0usize; // Track which GQA cache slot we're on

        // Timing accumulators for this token
        let mut tt_attn = 0.0f64;
        let mut tt_moe = 0.0f64;
        let mut tt_norm = 0.0f64;
        let mut tt_shared = 0.0f64;
        let mut tt_dense_mlp = 0.0f64;

        // ── 2. Layer loop ──
        for layer_idx in 0..num_layers {
            let layer = &graph.layers[layer_idx];

            // ── Pre-attention norm (fused residual add + RMSNorm) ──
            {
                let smem = (hs as u32) * 4; // FP32 per element
                let threads = 256u32.min(hs as u32);
                let cfg = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: smem,
                };
                unsafe {
                    k.fused_add_rmsnorm.clone().launch(cfg, (
                        *graph.d_hidden.device_ptr(),
                        *graph.d_residual.device_ptr(),
                        layer.input_norm_ptr,
                        eps,
                        hs as i32,
                        if first_residual { 1i32 } else { 0i32 },
                    )).map_err(|e| format!("fused_add_rmsnorm[{}]: {:?}", layer_idx, e))?;
                }
            }
            first_residual = false;

            // Timing: after pre-attn norm
            let t_attn_start = if timing {
                self.device.synchronize().map_err(|e| format!("timing sync: {:?}", e))?;
                Instant::now()
            } else { Instant::now() };

            // ── Attention ──
            match &layer.attn {
                GpuAttnConfig::LinearAttention {
                    in_proj_qkvz, in_proj_ba, out_proj,
                    conv_weight_ptr, a_log_ptr, dt_bias_ptr, norm_weight_ptr,
                    nk, nv, dk, dv, hr, kernel_dim, conv_dim, scale,
                    conv_state_ptr, recur_state_ptr,
                } => {
                    let nk_ = *nk; let nv_ = *nv; let dk_ = *dk; let dv_ = *dv;
                    let hr_ = *hr; let cd = *conv_dim; let kd = *kernel_dim;
                    let key_dim = nk_ * dk_;

                    // ── LA Step 1: Projections (cuBLAS GEMV) ──
                    let qkvz_w = &graph.weights[*in_proj_qkvz];
                    let ba_w = &graph.weights[*in_proj_ba];
                    self.gemv_bf16_to_f32(
                        qkvz_w, *graph.d_hidden.device_ptr(),
                        *graph.d_la_qkvz.device_ptr())?;
                    self.gemv_bf16_to_f32(
                        ba_w, *graph.d_hidden.device_ptr(),
                        *graph.d_la_ba.device_ptr())?;

                    // ── LA Step 2: Un-interleave QKVZ ──
                    // Interleaved: [h0_q(dk), h0_k(dk), h0_v(hr*dv), h0_z(hr*dv), h1_q, ...]
                    // → conv_input [q_flat(key_dim), k_flat(key_dim), v_flat(nv*dv)] in d_la_conv_out
                    // → z[nv*dv] in d_la_recur_out (temp, will be overwritten after recurrence)
                    {
                        let group_dim = 2 * dk_ + 2 * hr_ * dv_;
                        let total = nk_ * group_dim;
                        let threads = 256u32;
                        let blocks = ((total as u32) + threads - 1) / threads;
                        unsafe {
                            let unint_fn = self.device.get_func(MODULE_NAME, "uninterleave_qkvz")
                                .ok_or_else(|| "uninterleave_qkvz not found".to_string())?;
                            unint_fn.launch(
                                LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                (
                                    *graph.d_la_conv_out.device_ptr(),  // conv_input output
                                    *graph.d_la_recur_out.device_ptr(), // z output (temp)
                                    *graph.d_la_qkvz.device_ptr(),     // interleaved input
                                    nk_ as i32, dk_ as i32, hr_ as i32, dv_ as i32,
                                ),
                            ).map_err(|e| format!("uninterleave_qkvz[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    // Save z values from d_la_recur_out to d_la_gated_out before recurrence overwrites it
                    {
                        let z_size = nv_ * dv_;
                        unsafe {
                            let err = cuda_sys::lib().cuMemcpyDtoD_v2(
                                *graph.d_la_gated_out.device_ptr(),
                                *graph.d_la_recur_out.device_ptr(),
                                z_size * 4);
                            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                                return Err(format!("D2D z save[{}]: {:?}", layer_idx, err));
                            }
                        }
                    }

                    // ── LA Step 3: Conv1d (with SiLU) ──
                    // Input: d_la_conv_out [conv_dim] = [q_flat, k_flat, v_flat]
                    // This reads from d_la_conv_out and writes to d_la_qkvz (reuse as conv output buffer)
                    {
                        let threads = 256u32;
                        let blocks = ((cd as u32) + threads - 1) / threads;
                        unsafe {
                            let la_conv1d_fn = self.device.get_func(MODULE_NAME, "la_conv1d")
                                .ok_or_else(|| "la_conv1d kernel not found".to_string())?;
                            la_conv1d_fn.launch(
                                LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                (
                                    *conv_state_ptr,
                                    *graph.d_la_conv_out.device_ptr(), // un-interleaved conv input
                                    *graph.d_la_qkvz.device_ptr(),    // reuse as conv output (SiLU applied)
                                    *conv_weight_ptr,
                                    cd as i32,
                                    kd as i32,
                                ),
                            ).map_err(|e| format!("la_conv1d[{}]: {:?}", layer_idx, e))?;
                        }
                    }
                    // Now d_la_qkvz has conv output [q(key_dim), k(key_dim), v(nv*dv)] with SiLU

                    // ── LA Step 4: Compute gate and beta from BA ──
                    // BA is interleaved: [h0_b(ratio), h0_a(ratio), h1_b(ratio), h1_a(ratio), ...]
                    // beta = sigmoid(b), gate = exp(-exp(A_log) * softplus(a + dt_bias))
                    // Store in d_la_conv_out (reuse: [gate(nv), beta(nv)] at start)
                    let gate_ptr_local: u64;
                    let beta_ptr_local: u64;
                    {
                        let threads = 256u32;
                        let blocks = ((nv_ as u32) + threads - 1) / threads;
                        gate_ptr_local = *graph.d_la_conv_out.device_ptr();
                        beta_ptr_local = unsafe { (*graph.d_la_conv_out.device_ptr() as *const f32).add(nv_) as u64 };
                        unsafe {
                            let gb_fn = self.device.get_func(MODULE_NAME, "compute_gate_beta")
                                .ok_or_else(|| "compute_gate_beta not found".to_string())?;
                            gb_fn.launch(
                                LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                (
                                    gate_ptr_local,
                                    beta_ptr_local,
                                    *graph.d_la_ba.device_ptr(),
                                    *a_log_ptr,
                                    *dt_bias_ptr,
                                    nv_ as i32,
                                    hr_ as i32,
                                ),
                            ).map_err(|e| format!("compute_gate_beta[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    // ── LA Step 5: Head repeat-interleave (if hr > 1) ──
                    // q and k are [nk, dk]. Need to expand to [nv, dk] for the recurrence.
                    // Conv output in d_la_qkvz: [q(key_dim), k(key_dim), v(nv*dv)]
                    // After repeat-interleave: q[nv*dk], k[nv*dk] in d_la_recur_out (temp)
                    let q_ptr_for_recur: u64;
                    let k_ptr_for_recur: u64;
                    if hr_ > 1 {
                        // q: d_la_qkvz[0..key_dim] → d_la_recur_out[0..nv*dk]
                        let total_q = (nv_ * dk_) as u32;
                        let threads = 256u32;
                        let blocks = (total_q + threads - 1) / threads;
                        unsafe {
                            let ri_fn = self.device.get_func(MODULE_NAME, "repeat_interleave_heads")
                                .ok_or_else(|| "repeat_interleave_heads not found".to_string())?;
                            // Q
                            ri_fn.clone().launch(
                                LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                (
                                    *graph.d_la_recur_out.device_ptr(),  // output [nv*dk]
                                    *graph.d_la_qkvz.device_ptr(),      // input q [nk*dk]
                                    nk_ as i32, dk_ as i32, hr_ as i32,
                                ),
                            ).map_err(|e| format!("repeat_interleave q[{}]: {:?}", layer_idx, e))?;
                            // K: input at offset key_dim, output at offset nv*dk
                            let k_in = (*graph.d_la_qkvz.device_ptr() as *const f32).add(key_dim) as u64;
                            let k_out = (*graph.d_la_recur_out.device_ptr() as *const f32).add(nv_ * dk_) as u64;
                            ri_fn.launch(
                                LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                (
                                    k_out, k_in,
                                    nk_ as i32, dk_ as i32, hr_ as i32,
                                ),
                            ).map_err(|e| format!("repeat_interleave k[{}]: {:?}", layer_idx, e))?;
                        }
                        q_ptr_for_recur = *graph.d_la_recur_out.device_ptr();
                        k_ptr_for_recur = unsafe { (*graph.d_la_recur_out.device_ptr() as *const f32).add(nv_ * dk_) as u64 };
                    } else {
                        q_ptr_for_recur = *graph.d_la_qkvz.device_ptr();
                        k_ptr_for_recur = unsafe { (*graph.d_la_qkvz.device_ptr() as *const f32).add(key_dim) as u64 };
                    }

                    // ── LA Step 6: L2 normalize + scale Q and K ──
                    {
                        let threads = 256u32;
                        unsafe {
                            let l2_fn = self.device.get_func(MODULE_NAME, "l2norm_scale_per_head")
                                .ok_or_else(|| "l2norm_scale_per_head not found".to_string())?;
                            // Q: normalize with scale
                            l2_fn.clone().launch(
                                LaunchConfig { grid_dim: (nv_ as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                (q_ptr_for_recur, *scale, nv_ as i32, dk_ as i32),
                            ).map_err(|e| format!("l2norm q[{}]: {:?}", layer_idx, e))?;
                            // K: normalize without scale (scale=1.0)
                            l2_fn.launch(
                                LaunchConfig { grid_dim: (nv_ as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                (k_ptr_for_recur, 1.0f32, nv_ as i32, dk_ as i32),
                            ).map_err(|e| format!("l2norm k[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    // ── LA Step 7: Gated delta net recurrence ──
                    // v is at offset 2*key_dim in d_la_qkvz (conv output)
                    // Output to d_la_ba (free after step 4) to avoid overlap with q/k in d_la_recur_out
                    let v_ptr = unsafe { (*graph.d_la_qkvz.device_ptr() as *const f32).add(2 * key_dim) as u64 };
                    {
                        let threads = 256u32;
                        unsafe {
                            let delta_fn = self.device.get_func(MODULE_NAME, "gated_delta_net_step")
                                .ok_or_else(|| "gated_delta_net_step not found".to_string())?;
                            delta_fn.launch(
                                LaunchConfig { grid_dim: (nv_ as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                (
                                    *recur_state_ptr,
                                    q_ptr_for_recur,
                                    k_ptr_for_recur,
                                    v_ptr,
                                    gate_ptr_local,
                                    beta_ptr_local,
                                    *graph.d_la_ba.device_ptr(),  // output: use d_la_ba (free after step 4)
                                    nv_ as i32, dk_ as i32, dv_ as i32,
                                ),
                            ).map_err(|e| format!("gated_delta_net_step[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    // ── LA Step 8: Gated RMSNorm + SiLU ──
                    // z was saved in d_la_gated_out earlier
                    // recurrence output is in d_la_ba
                    {
                        let threads = 256u32;
                        let smem = (dv_ as u32 + 32) * 4;
                        unsafe {
                            let gated_rmsnorm_fn = self.device.get_func(MODULE_NAME, "gated_rmsnorm_silu")
                                .ok_or_else(|| "gated_rmsnorm_silu not found".to_string())?;
                            gated_rmsnorm_fn.launch(
                                LaunchConfig { grid_dim: (nv_ as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: smem },
                                (
                                    *graph.d_la_conv_out.device_ptr(),  // output (reuse buffer)
                                    *graph.d_la_ba.device_ptr(),        // recurrence output (from step 7)
                                    *graph.d_la_gated_out.device_ptr(), // z (saved earlier)
                                    *norm_weight_ptr,
                                    eps,
                                    nv_ as i32, dv_ as i32,
                                ),
                            ).map_err(|e| format!("gated_rmsnorm_silu[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    // ── LA Step 9: Output projection ──
                    let out_w = &graph.weights[*out_proj];
                    // d_la_conv_out (FP32 nv*dv, gated rmsnorm output) → BF16 → GEMV → d_hidden
                    let gated_size = nv_ * dv_;
                    {
                        unsafe {
                            let fp32_to_bf16_fn = self.device.get_func(MODULE_NAME, "fp32_to_bf16")
                                .ok_or_else(|| "fp32_to_bf16 not found".to_string())?;
                            fp32_to_bf16_fn.launch(
                                LaunchConfig::for_num_elems(gated_size as u32),
                                (
                                    *graph.d_scratch.device_ptr(),
                                    *graph.d_la_conv_out.device_ptr(),
                                    gated_size as i32,
                                ),
                            ).map_err(|e| format!("fp32_to_bf16 la out[{}]: {:?}", layer_idx, e))?;
                        }
                    }
                    self.gemv_bf16_internal(
                        out_w,
                        *graph.d_scratch.device_ptr(),
                        *graph.d_hidden.device_ptr(),
                    )?;
                }

                GpuAttnConfig::GQA {
                    q_proj, k_proj, v_proj, o_proj,
                    fused_qkv,
                    num_heads, num_kv_heads, head_dim, sm_scale,
                    q_norm_ptr, k_norm_ptr, gated,
                } => {
                    let nh = *num_heads;
                    let nkv = *num_kv_heads;
                    let hd = *head_dim;
                    let kv_stride = nkv * hd;

                    // ── GQA: Q/K/V projections ──
                    if let Some(fid) = fused_qkv {
                        let fw = &graph.weights[*fid];
                        self.gemv_bf16_to_f32(fw, *graph.d_hidden.device_ptr(),
                            *graph.d_gqa_q.device_ptr())?;
                        let q_size = if *gated { nh * hd * 2 } else { nh * hd };
                        let k_offset = q_size;
                        let v_offset = k_offset + kv_stride;
                        unsafe {
                            let err = cuda_sys::lib().cuMemcpyDtoD_v2(
                                *graph.d_gqa_k.device_ptr(),
                                (*graph.d_gqa_q.device_ptr() as *const f32).add(k_offset) as u64,
                                kv_stride * 4);
                            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                                return Err(format!("D2D K split[{}]: {:?}", layer_idx, err));
                            }
                            let err = cuda_sys::lib().cuMemcpyDtoD_v2(
                                *graph.d_gqa_v.device_ptr(),
                                (*graph.d_gqa_q.device_ptr() as *const f32).add(v_offset) as u64,
                                kv_stride * 4);
                            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                                return Err(format!("D2D V split[{}]: {:?}", layer_idx, err));
                            }
                        }
                    } else {
                        let qw = &graph.weights[*q_proj];
                        let kw = &graph.weights[*k_proj];
                        let vw = &graph.weights[*v_proj];
                        self.gemv_bf16_to_f32(qw, *graph.d_hidden.device_ptr(),
                            *graph.d_gqa_q.device_ptr())?;
                        self.gemv_bf16_to_f32(kw, *graph.d_hidden.device_ptr(),
                            *graph.d_gqa_k.device_ptr())?;
                        self.gemv_bf16_to_f32(vw, *graph.d_hidden.device_ptr(),
                            *graph.d_gqa_v.device_ptr())?;
                    }

                    // ── GQA: Split gated Q into Q[nh*hd] and gate[nh*hd] ──
                    // Q proj output for gated attn is [nh, 2*hd] = [head0_q(hd), head0_gate(hd), ...]
                    // Must split before QK norm/RoPE which expect [nh, hd] layout.
                    // Gate stored in d_la_qkvz (unused during GQA layers).
                    if *gated {
                        let total = (nh * hd) as u32;
                        let threads = 256u32;
                        let blocks = (total + threads - 1) / threads;
                        unsafe {
                            let split_fn = self.device.get_func(MODULE_NAME, "split_gated_q")
                                .ok_or_else(|| "split_gated_q not found".to_string())?;
                            split_fn.launch(
                                LaunchConfig {
                                    grid_dim: (blocks, 1, 1),
                                    block_dim: (threads, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                (
                                    *graph.d_gqa_q.device_ptr(),      // q_out (in-place safe)
                                    *graph.d_la_qkvz.device_ptr(),    // gate_out
                                    *graph.d_gqa_q.device_ptr(),      // qg_in
                                    nh as i32,
                                    hd as i32,
                                ),
                            ).map_err(|e| format!("split_gated_q[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    // ── GQA: QK norm (if enabled) ──
                    // q_norm/k_norm are [head_dim] shared across all heads (weight_per_head=0)
                    if *q_norm_ptr != 0 {
                        let threads = 256u32;
                        let cfg = LaunchConfig {
                            grid_dim: (nh as u32, 1, 1),
                            block_dim: (threads, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        unsafe {
                            let norm_fn = self.device.get_func(MODULE_NAME, "per_head_rmsnorm")
                                .ok_or_else(|| "per_head_rmsnorm not found".to_string())?;
                            norm_fn.launch(cfg, (
                                *graph.d_gqa_q.device_ptr(),
                                *q_norm_ptr,
                                eps,
                                nh as i32,
                                hd as i32,
                                0i32, // weight shared across heads
                            )).map_err(|e| format!("per_head_rmsnorm Q[{}]: {:?}", layer_idx, e))?;
                        }
                    }
                    if *k_norm_ptr != 0 {
                        let threads = 256u32;
                        let cfg = LaunchConfig {
                            grid_dim: (nkv as u32, 1, 1),
                            block_dim: (threads, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        unsafe {
                            let norm_fn = self.device.get_func(MODULE_NAME, "per_head_rmsnorm")
                                .ok_or_else(|| "per_head_rmsnorm not found".to_string())?;
                            norm_fn.launch(cfg, (
                                *graph.d_gqa_k.device_ptr(),
                                *k_norm_ptr,
                                eps,
                                nkv as i32,
                                hd as i32,
                                0i32, // weight shared across heads
                            )).map_err(|e| format!("per_head_rmsnorm K[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    // ── GQA: RoPE ──
                    if let Some(ref d_cos) = graph.d_rope_cos {
                        if let Some(ref d_sin) = graph.d_rope_sin {
                            let half_dim = graph.rope_half_dim;
                            let total_heads = nh + nkv;
                            let total_work = total_heads * half_dim;
                            let threads = 256u32;
                            let blocks = ((total_work as u32) + threads - 1) / threads;
                            let cfg = LaunchConfig {
                                grid_dim: (blocks, 1, 1),
                                block_dim: (threads, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            unsafe {
                                let rope_fn = self.device.get_func(MODULE_NAME, "apply_rope")
                                    .ok_or_else(|| "apply_rope not found".to_string())?;
                                rope_fn.launch(cfg, (
                                    *graph.d_gqa_q.device_ptr(),
                                    *graph.d_gqa_k.device_ptr(),
                                    *d_cos.device_ptr(),
                                    *d_sin.device_ptr(),
                                    position as i32,
                                    nh as i32,
                                    nkv as i32,
                                    hd as i32,
                                    half_dim as i32,
                                )).map_err(|e| format!("apply_rope[{}]: {:?}", layer_idx, e))?;
                            }
                        }
                    }

                    // ── GQA: KV cache write ──
                    {
                        let threads = 256u32;
                        let blocks = ((kv_stride as u32) + threads - 1) / threads;
                        let cfg = LaunchConfig {
                            grid_dim: (blocks, 1, 1),
                            block_dim: (threads, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        if graph.kv_k_ptrs[layer_idx] == 0 || graph.kv_v_ptrs[layer_idx] == 0 {
                            return Err(format!(
                                "kv_cache_write[{}]: null KV pointer (k={:#x}, v={:#x})",
                                layer_idx, graph.kv_k_ptrs[layer_idx], graph.kv_v_ptrs[layer_idx]));
                        }
                        unsafe {
                            let kv_write_fn = self.device.get_func(MODULE_NAME, "kv_cache_write")
                                .ok_or_else(|| "kv_cache_write not found".to_string())?;
                            kv_write_fn.launch(cfg, (
                                graph.kv_k_ptrs[layer_idx],
                                graph.kv_v_ptrs[layer_idx],
                                *graph.d_gqa_k.device_ptr(),
                                *graph.d_gqa_v.device_ptr(),
                                position as i32,
                                kv_stride as i32,
                            )).map_err(|e| format!("kv_cache_write[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    // ── GQA: Attention compute ──
                    // For short sequences: single-block shared-memory kernel (one block per Q head).
                    // For long sequences: FlashDecoding tiled kernel (splits seq across blocks)
                    //   + lightweight reduce kernel. Threshold: use tiled when seq_len > tile_size.
                    {
                        let threads = 256u32;
                        let seq_len = (position + 1) as u32;
                        let tile_size = graph.gqa_tile_size;
                        let num_tiles_candidate = if tile_size > 0 {
                            ((seq_len as usize) + tile_size - 1) / tile_size
                        } else { 0 };
                        // Use tiled when total blocks (tiles * heads) >= num_sms.
                        // Below that, the original single-block-per-head kernel is faster
                        // because the per-tile overhead isn't worth it.
                        let use_tiled = tile_size > 0
                            && graph.d_gqa_tiled_o.is_some()
                            && (num_tiles_candidate * nh) >= graph.num_sms;

                        if use_tiled {
                            // FlashDecoding: tiled attention + reduce
                            let num_tiles = ((seq_len as usize) + tile_size - 1) / tile_size;
                            let tile_smem = (tile_size as u32) * 4 + 128;
                            let tiled_o = graph.d_gqa_tiled_o.as_ref().unwrap();
                            let tiled_lse = graph.d_gqa_tiled_lse.as_ref().unwrap();
                            unsafe {
                                let tiled_fn = self.device.get_func(MODULE_NAME, "gqa_attention_tiled")
                                    .ok_or_else(|| "gqa_attention_tiled not found".to_string())?;
                                tiled_fn.launch(
                                    LaunchConfig {
                                        grid_dim: (nh as u32, num_tiles as u32, 1),
                                        block_dim: (threads, 1, 1),
                                        shared_mem_bytes: tile_smem,
                                    },
                                    (
                                        *tiled_o.device_ptr(),
                                        *tiled_lse.device_ptr(),
                                        *graph.d_gqa_q.device_ptr(),
                                        graph.kv_k_ptrs[layer_idx],
                                        graph.kv_v_ptrs[layer_idx],
                                        *sm_scale,
                                        nh as i32,
                                        nkv as i32,
                                        hd as i32,
                                        seq_len as i32,
                                        tile_size as i32,
                                    ),
                                ).map_err(|e| format!("gqa_attention_tiled[{}]: {:?}", layer_idx, e))?;

                                let reduce_fn = self.device.get_func(MODULE_NAME, "gqa_attention_reduce")
                                    .ok_or_else(|| "gqa_attention_reduce not found".to_string())?;
                                let reduce_smem = (num_tiles as u32) * 4;
                                reduce_fn.launch(
                                    LaunchConfig {
                                        grid_dim: (nh as u32, 1, 1),
                                        block_dim: (threads, 1, 1),
                                        shared_mem_bytes: reduce_smem,
                                    },
                                    (
                                        *graph.d_gqa_out.device_ptr(),
                                        *tiled_o.device_ptr(),
                                        *tiled_lse.device_ptr(),
                                        nh as i32,
                                        hd as i32,
                                        num_tiles as i32,
                                    ),
                                ).map_err(|e| format!("gqa_attention_reduce[{}]: {:?}", layer_idx, e))?;
                            }
                        } else {
                            // Original single-block kernel for short sequences
                            let smem_threshold = (graph.gqa_max_smem_bytes - 128) / 4;
                            let use_smem = seq_len <= smem_threshold;
                            let shared_mem_bytes = if use_smem {
                                seq_len * 4 + 128
                            } else {
                                128
                            };
                            let cfg = LaunchConfig {
                                grid_dim: (nh as u32, 1, 1),
                                block_dim: (threads, 1, 1),
                                shared_mem_bytes,
                            };
                            unsafe {
                                let attn_fn = self.device.get_func(MODULE_NAME, "gqa_attention")
                                    .ok_or_else(|| "gqa_attention not found".to_string())?;
                                attn_fn.launch(cfg, (
                                    *graph.d_gqa_out.device_ptr(),
                                    *graph.d_gqa_q.device_ptr(),
                                    graph.kv_k_ptrs[layer_idx],
                                    graph.kv_v_ptrs[layer_idx],
                                    *sm_scale,
                                    nh as i32,
                                    nkv as i32,
                                    hd as i32,
                                    seq_len as i32,
                                    graph.kv_max_seq as i32,
                                    if use_smem { 1i32 } else { 0i32 },
                                )).map_err(|e| format!("gqa_attention[{}]: {:?}", layer_idx, e))?;
                            }
                        }
                    }

                    // ── GQA: Apply gated attention ──
                    // d_gqa_out *= sigmoid(gate) where gate is in d_la_qkvz
                    if *gated {
                        let total = (nh * hd) as u32;
                        let threads = 256u32;
                        let blocks = (total + threads - 1) / threads;
                        unsafe {
                            let gate_fn = self.device.get_func(MODULE_NAME, "apply_gated_attn")
                                .ok_or_else(|| "apply_gated_attn not found".to_string())?;
                            gate_fn.launch(
                                LaunchConfig {
                                    grid_dim: (blocks, 1, 1),
                                    block_dim: (threads, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                (
                                    *graph.d_gqa_out.device_ptr(),
                                    *graph.d_la_qkvz.device_ptr(),
                                    (nh * hd) as i32,
                                ),
                            ).map_err(|e| format!("apply_gated_attn[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    // ── GQA: O projection ──
                    // d_gqa_out is FP32 [nh * hd]. Convert to BF16, then GEMV.
                    let o_size = nh * hd;
                    {
                        unsafe {
                            let fp32_to_bf16_fn = self.device.get_func(MODULE_NAME, "fp32_to_bf16")
                                .ok_or_else(|| "fp32_to_bf16 not found".to_string())?;
                            fp32_to_bf16_fn.launch(
                                LaunchConfig::for_num_elems(o_size as u32),
                                (
                                    *graph.d_scratch.device_ptr(),
                                    *graph.d_gqa_out.device_ptr(),
                                    o_size as i32,
                                ),
                            ).map_err(|e| format!("fp32_to_bf16 gqa out[{}]: {:?}", layer_idx, e))?;
                        }
                    }
                    let ow = &graph.weights[*o_proj];
                    self.gemv_bf16_internal(
                        ow,
                        *graph.d_scratch.device_ptr(),
                        *graph.d_hidden.device_ptr(),
                    )?;

                    gqa_cache_idx += 1;
                }

                GpuAttnConfig::MLA { .. } => {
                    return Err("MLA attention not implemented for GPU decode".to_string());
                }
            }

            // Timing: after attention
            if timing {
                self.device.synchronize().map_err(|e| format!("timing sync: {:?}", e))?;
                tt_attn += t_attn_start.elapsed().as_secs_f64();
            }

            // ── Post-attention norm (fused residual add + RMSNorm) ──
            {
                let smem = (hs as u32) * 4;
                let threads = 256u32.min(hs as u32);
                let cfg = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: smem,
                };
                unsafe {
                    k.fused_add_rmsnorm.clone().launch(cfg, (
                        *graph.d_hidden.device_ptr(),
                        *graph.d_residual.device_ptr(),
                        layer.post_attn_norm_ptr,
                        eps,
                        hs as i32,
                        0i32, // not first layer
                    )).map_err(|e| format!("post_attn_norm[{}]: {:?}", layer_idx, e))?;
                }
            }

            // Timing: after post-attn norm, before MLP/MoE
            let t_mlp_start = if timing {
                self.device.synchronize().map_err(|e| format!("timing sync: {:?}", e))?;
                Instant::now()
            } else { Instant::now() };

            // ── MLP / MoE ──
            // Check if this layer has MoE data registered
            let has_moe = layer_idx < graph.moe_layers.len()
                && graph.moe_layers[layer_idx].is_some();
            #[cfg(feature = "gpu-debug")]
            {
                self.device.synchronize().map_err(|e| format!("sync before mlp[{}]: {:?}", layer_idx, e))?;
                self.device.synchronize().map_err(|e| format!("sync norm dbg: {:?}", e))?;
                let mut buf = vec![0u16; 4];
                unsafe {
                    let _ = cuda_sys::lib().cuMemcpyDtoH_v2(
                        buf.as_mut_ptr() as *mut std::ffi::c_void,
                        *graph.d_hidden.device_ptr(), 8);
                }
                let v0 = f32::from_bits((buf[0] as u32) << 16);
                if v0.is_nan() || position < 30 {
                    debug_peek_bf16(&format!("L{} post_attn_norm d_hidden", layer_idx),
                        *graph.d_hidden.device_ptr(), 4);
                }
            }
            log::trace!("gpu_decode_step: layer {} mlp/moe (has_moe={})", layer_idx, has_moe);
            if has_moe {
                // Use the fast Rust MoE forward path (HCS + double-buffered DMA)
                // d_hidden → MoE → d_moe_out, then add d_moe_out to d_hidden
                self.moe_forward_with_graph(graph, layer_idx)
                    .map_err(|e| format!("moe_forward[{}]: {}", layer_idx, e))?;

                // Add MoE output to hidden state: d_hidden = d_moe_out
                // (MoE output replaces hidden, residual stream continues)
                unsafe {
                    let err = cuda_sys::lib().cuMemcpyDtoD_v2(
                        *graph.d_hidden.device_ptr(),
                        *graph.d_moe_out.device_ptr(),
                        hs * 2); // BF16
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(format!("D2D moe_out->hidden[{}]: {:?}", layer_idx, err));
                    }
                }
                #[cfg(feature = "gpu-debug")]
                {
                    self.device.synchronize().map_err(|e| format!("sync moe dbg: {:?}", e))?;
                    let mut buf = vec![0u16; 4];
                    unsafe {
                        let _ = cuda_sys::lib().cuMemcpyDtoH_v2(
                            buf.as_mut_ptr() as *mut std::ffi::c_void,
                            *graph.d_hidden.device_ptr(), 8);
                    }
                    let v0 = f32::from_bits((buf[0] as u32) << 16);
                    if v0.is_nan() || position < 30 {
                        debug_peek_bf16(&format!("L{} after_moe d_hidden", layer_idx),
                            *graph.d_hidden.device_ptr(), 4);
                    }
                }
            } else if let GpuMlpConfig::Dense { gate_proj, up_proj, down_proj } = &layer.mlp {
                // Dense MLP: gate_up = [gate(hidden), up(hidden)], silu(gate)*up, down
                let gw = &graph.weights[*gate_proj];
                let uw = &graph.weights[*up_proj];
                let intermediate = gw.rows;

                // Gate GEMV: hidden → d_expert_gate_up[0..intermediate]
                self.gemv_bf16_internal(
                    gw, *graph.d_hidden.device_ptr(),
                    *graph.d_expert_gate_up.device_ptr())?;
                // Up GEMV: hidden → d_expert_gate_up[intermediate..2*intermediate]
                let up_out_ptr = unsafe {
                    (*graph.d_expert_gate_up.device_ptr() as *const u16).add(intermediate) as u64
                };
                self.gemv_bf16_internal(uw, *graph.d_hidden.device_ptr(), up_out_ptr)?;

                // Fused SiLU*mul
                unsafe {
                    k.silu_mul.clone().launch(
                        LaunchConfig::for_num_elems(intermediate as u32),
                        (
                            *graph.d_expert_scratch.device_ptr(),
                            *graph.d_expert_gate_up.device_ptr(),
                            intermediate as i32,
                        ),
                    ).map_err(|e| format!("silu_mul dense[{}]: {:?}", layer_idx, e))?;
                }

                // Down GEMV: d_expert_scratch → d_hidden
                let dw = &graph.weights[*down_proj];
                self.gemv_bf16_internal(
                    dw, *graph.d_expert_scratch.device_ptr(),
                    *graph.d_hidden.device_ptr())?;
            }
            // GpuMlpConfig::None → skip (layer 0 in QCN is dense but registered separately)

            // Timing: after MLP/MoE
            if timing {
                self.device.synchronize().map_err(|e| format!("timing sync: {:?}", e))?;
                let mlp_elapsed = t_mlp_start.elapsed().as_secs_f64();
                if has_moe {
                    tt_moe += mlp_elapsed;
                } else {
                    tt_dense_mlp += mlp_elapsed;
                }
            }

            #[cfg(feature = "gpu-debug")]
            {
                if self.debug_capture_layers {
                    self.device.synchronize().map_err(|e| format!("sync capture: {:?}", e))?;
                    let mut buf = vec![0u16; hs];
                    unsafe {
                        let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                            buf.as_mut_ptr() as *mut std::ffi::c_void,
                            *graph.d_hidden.device_ptr(),
                            hs * 2);
                        if err != cuda_sys::CUresult::CUDA_SUCCESS {
                            return Err(format!("capture D2H layer {}: {:?}", layer_idx, err));
                        }
                    }
                    self.debug_layer_captures.push(buf);
                }

                if self.debug_stop_layer > 0 && layer_idx + 1 >= self.debug_stop_layer {
                    self.device.synchronize().map_err(|e| format!("sync debug_stop: {:?}", e))?;
                    log::warn!("DEBUG: stopped after layer {} (debug_stop_layer={})", layer_idx, self.debug_stop_layer);
                    return Ok(());
                }
            }
        }

        // ── 3. Final norm ──
        let t_lmhead_start = if timing {
            self.device.synchronize().map_err(|e| format!("timing sync: {:?}", e))?;
            Instant::now()
        } else { Instant::now() };

        #[cfg(feature = "gpu-debug")]
        {
            self.device.synchronize().map_err(|e| format!("sync after all layers: {:?}", e))?;
            debug_peek_bf16("before_final_norm d_hidden", *graph.d_hidden.device_ptr(), 4);
        }
        {
            let smem = (hs as u32) * 4;
            let threads = 256u32.min(hs as u32);
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: smem,
            };
            unsafe {
                k.fused_add_rmsnorm.clone().launch(cfg, (
                    *graph.d_hidden.device_ptr(),
                    *graph.d_residual.device_ptr(),
                    graph.final_norm_ptr,
                    eps,
                    hs as i32,
                    0i32,
                )).map_err(|e| format!("final_norm: {:?}", e))?;
            }
        }

        // ── 4. LM head GEMV → logits ──
        {
            let lm_w = &graph.weights[graph.lm_head_wid];
            // d_hidden (BF16) → d_logits (FP32) via cuBLAS GEMV
            self.gemv_bf16_to_f32_internal(
                lm_w, *graph.d_hidden.device_ptr(),
                *graph.d_logits.device_ptr())?;
        }

        // ── 5. Sync + D2H logits ──
        self.device.synchronize()
            .map_err(|e| format!("sync: {:?}", e))?;

        // Timing: accumulate per-component times
        if timing {
            let tt_lmhead = t_lmhead_start.elapsed().as_secs_f64();
            let tt_total = t0.elapsed().as_secs_f64();
            graph.t_attn += tt_attn;
            graph.t_route += tt_moe; // MoE includes routing + DMA + compute
            graph.t_shared += tt_shared;
            graph.t_dense_mlp += tt_dense_mlp;
            graph.t_lm_head += tt_lmhead;
            graph.t_total += tt_total;
            graph.t_norm += tt_total - tt_attn - tt_moe - tt_shared - tt_dense_mlp - tt_lmhead;
            graph.timing_step_count += 1;
        }

        #[cfg(feature = "gpu-debug")]
        debug_peek_f32("logits[0..4]", *graph.d_logits.device_ptr(), 4);

        unsafe {
            let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                graph.h_logits.as_mut_ptr() as *mut std::ffi::c_void,
                *graph.d_logits.device_ptr(),
                graph.vocab_size * 4);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("D2H logits: {:?}", err));
            }
        }

        #[cfg(feature = "gpu-debug")]
        {
            let mut top3: Vec<(usize, f32)> = graph.h_logits.iter().enumerate()
                .map(|(i, &v)| (i, v)).collect();
            top3.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            log::info!("DBG logits top3: [{}: {:.2}, {}: {:.2}, {}: {:.2}]",
                top3[0].0, top3[0].1, top3[1].0, top3[1].1, top3[2].0, top3[2].1);
        }

        Ok(())
    }

    /// BF16 GEMV: output_bf16[N] = weight_bf16[N,K] @ input_bf16[K]
    fn gemv_bf16_internal(&self, w: &GpuWeight, input_ptr: u64, output_ptr: u64) -> Result<(), String> {
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
            ).map_err(|e| format!("cuBLAS gemv_bf16: {:?}", e))?;
        }
        Ok(())
    }

    /// BF16 input GEMV with FP32 output: output_f32[N] = weight_bf16[N,K] @ input_bf16[K]
    fn gemv_bf16_to_f32(&self, w: &GpuWeight, input_ptr: u64, output_ptr: u64) -> Result<(), String> {
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
                output_ptr as *mut std::ffi::c_void,
                cublas_sys::cudaDataType::CUDA_R_32F, w.rows as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            ).map_err(|e| format!("cuBLAS gemv_bf16_to_f32: {:?}", e))?;
        }
        Ok(())
    }

    /// Same as gemv_bf16_to_f32 but takes raw pointers (for use in decode_step_with_graph).
    fn gemv_bf16_to_f32_internal(&self, w: &GpuWeight, input_ptr: u64, output_ptr: u64) -> Result<(), String> {
        self.gemv_bf16_to_f32(w, input_ptr, output_ptr)
    }

    /// Generate tokens in a tight Rust loop via GPU decode.
    /// No Python, no GIL. Same interface as CpuDecodeStore.generate_stream.
    pub fn gpu_generate_stream<F>(
        &mut self,
        first_token: usize,
        start_position: usize,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        stop_ids: &[usize],
        tokenizer: &tokenizers::Tokenizer,
        presence_penalty: f32,
        mut on_token: F,
    ) -> usize
    where
        F: FnMut(usize, &str, Option<&str>) -> bool,
    {
        use std::time::Instant;

        // Bind CUDA context to this thread. Required when called from
        // the server thread (which differs from the setup thread).
        if let Err(e) = self.device.bind_to_thread() {
            log::error!("gpu_generate_stream: failed to bind CUDA context: {:?}", e);
            return 0;
        }

        let vocab_size = match self.graph.as_ref() {
            Some(g) => g.vocab_size,
            None => { log::error!("gpu_generate_stream: graph not configured"); return 0; }
        };

        let stop_set: std::collections::HashSet<usize> = stop_ids.iter().copied().collect();

        // RNG
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

        // Begin activation tracking for this prompt's decode
        if let Some(ref mut hcs) = self.graph.as_mut().unwrap().hcs {
            hcs.begin_prompt();
        }

        // Reset per-prompt timing accumulators
        {
            let g = self.graph.as_mut().unwrap();
            if g.timing_enabled {
                g.timing_step_count = 0;
                g.t_total = 0.0; g.t_norm = 0.0; g.t_attn = 0.0;
                g.t_route = 0.0; g.t_expert_dma = 0.0; g.t_expert_compute = 0.0;
                g.t_shared = 0.0; g.t_dense_mlp = 0.0; g.t_lm_head = 0.0;
                g.t_moe_route_sync = 0.0; g.t_moe_expert_loop = 0.0;
                g.t_moe_shared = 0.0; g.t_moe_overhead = 0.0;
                g.dma_bytes_total = 0; g.dma_call_count = 0;
                g.dma_cold_experts = 0; g.dma_hcs_experts = 0;
            }
        }

        let decode_start = Instant::now();
        let mut next_token = first_token;
        let mut generated = 0usize;
        let mut seen_tokens: std::collections::HashSet<usize> = std::collections::HashSet::new();
        seen_tokens.insert(first_token);

        #[cfg(feature = "gpu-debug")]
        let debug_logits = std::env::var("KRASIS_DEBUG_LOGITS").ok()
            .map(|v| v.parse::<usize>().unwrap_or(0)).unwrap_or(0);

        for step in 0..max_tokens {
            let pos = start_position + step;
            if let Err(e) = self.gpu_decode_step(next_token, pos) {
                log::error!("gpu_generate_stream: decode_step error: {}", e);
                break;
            }

            // Logits are now in graph.h_logits (host-side)
            let logits = &mut self.graph.as_mut().unwrap().h_logits;

            #[cfg(feature = "gpu-debug")]
            if debug_logits > 0 && step < debug_logits {
                let mut indexed: Vec<(usize, f32)> = logits.iter().copied()
                    .enumerate().take(vocab_size).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top5: Vec<String> = indexed[..5.min(indexed.len())].iter()
                    .map(|(idx, val)| {
                        let tok_str = tokenizer.decode(&[*idx as u32], true).unwrap_or_default();
                        format!("{}({:.3})\"{}\"", idx, val, tok_str.replace('\n', "\\n"))
                    }).collect();
                log::warn!("LOGITS step={} pos={} input_tok={} top5=[{}]",
                    step, pos, next_token, top5.join(", "));
            }

            if presence_penalty != 0.0 {
                for &tok in &seen_tokens {
                    if tok < vocab_size {
                        logits[tok] -= presence_penalty;
                    }
                }
            }

            next_token = crate::decode::sample_from_logits_pub(
                logits, vocab_size, temperature, top_k, top_p, &mut rng_next);
            seen_tokens.insert(next_token);
            generated += 1;

            let text = tokenizer.decode(&[next_token as u32], true)
                .unwrap_or_default();

            let finish_reason = if stop_set.contains(&next_token) {
                Some("stop")
            } else if generated >= max_tokens {
                Some("length")
            } else {
                None
            };

            let finished = finish_reason.is_some();
            let cont = on_token(next_token, &text, finish_reason);
            if finished || !cont {
                break;
            }
        }

        let elapsed = decode_start.elapsed().as_secs_f64();
        if generated > 0 {
            let tps = generated as f64 / elapsed;
            log::info!("gpu_generate_stream: {} tokens in {:.2}s ({:.1} tok/s)",
                generated, elapsed, tps);
        }

        // Print timing summary if enabled
        if let Some(ref mut graph) = self.graph {
            if graph.timing_enabled && graph.timing_step_count > 0 {
                let n = graph.timing_step_count as f64;
                let avg_total = graph.t_total / n * 1000.0;
                let avg_attn = graph.t_attn / n * 1000.0;
                let avg_moe = graph.t_route / n * 1000.0;
                let avg_norm = graph.t_norm / n * 1000.0;
                let avg_dense = graph.t_dense_mlp / n * 1000.0;
                let avg_lm = graph.t_lm_head / n * 1000.0;

                log::info!("┌─────────────────────────────────────────────────┐");
                log::info!("│  GPU DECODE TIMING ({} tokens avg)             │", graph.timing_step_count);
                log::info!("├─────────────────────────────────────────────────┤");
                log::info!("│  Total:       {:7.2} ms/tok  ({:5.1} tok/s)    │", avg_total, 1000.0 / avg_total);
                log::info!("│  Attention:   {:7.2} ms  ({:4.1}%)              │", avg_attn, avg_attn / avg_total * 100.0);
                log::info!("│  MoE:         {:7.2} ms  ({:4.1}%)              │", avg_moe, avg_moe / avg_total * 100.0);
                log::info!("│  Norms+Emb:   {:7.2} ms  ({:4.1}%)              │", avg_norm, avg_norm / avg_total * 100.0);
                log::info!("│  Dense MLP:   {:7.2} ms  ({:4.1}%)              │", avg_dense, avg_dense / avg_total * 100.0);
                log::info!("│  LM Head:     {:7.2} ms  ({:4.1}%)              │", avg_lm, avg_lm / avg_total * 100.0);
                log::info!("│  Other:       {:7.2} ms  ({:4.1}%)              │",
                    avg_total - avg_attn - avg_moe - avg_norm - avg_dense - avg_lm,
                    (avg_total - avg_attn - avg_moe - avg_norm - avg_dense - avg_lm) / avg_total * 100.0);
                log::info!("├─────────────────────────────────────────────────┤");
                let avg_route_sync = graph.t_moe_route_sync / n * 1000.0;
                let avg_expert_loop = graph.t_moe_expert_loop / n * 1000.0;
                let avg_shared = graph.t_moe_shared / n * 1000.0;
                let avg_moe_other = avg_moe - avg_route_sync - avg_expert_loop - avg_shared;
                log::info!("│  MoE breakdown (of {:.2} ms):                    │", avg_moe);
                log::info!("│    Route sync:  {:7.2} ms  ({:4.1}%)             │", avg_route_sync, avg_route_sync / avg_moe * 100.0);
                log::info!("│    Expert loop: {:7.2} ms  ({:4.1}%)             │", avg_expert_loop, avg_expert_loop / avg_moe * 100.0);
                log::info!("│    Shared exp:  {:7.2} ms  ({:4.1}%)             │", avg_shared, avg_shared / avg_moe * 100.0);
                log::info!("│    MoE other:   {:7.2} ms  ({:4.1}%)             │", avg_moe_other, avg_moe_other / avg_moe * 100.0);
                log::info!("└─────────────────────────────────────────────────┘");

                // Measured PCIe DMA stats
                let (hcs_cached, hcs_hits, hcs_misses) = if let Some(ref hcs) = graph.hcs {
                    (hcs.num_cached, hcs.total_hits, hcs.total_misses)
                } else { (0, 0, 0) };
                let avg_dma_bytes = graph.dma_bytes_total as f64 / n;
                let avg_dma_calls = graph.dma_call_count as f64 / n;
                let avg_cold = graph.dma_cold_experts as f64 / n;
                let avg_hcs = graph.dma_hcs_experts as f64 / n;
                let dma_mb = avg_dma_bytes / (1024.0 * 1024.0);
                // PCIe bandwidth: bytes DMA'd / expert loop wall time
                // This is a lower bound (expert loop includes compute too)
                let min_pcie_bw = if avg_expert_loop > 0.001 {
                    dma_mb / (avg_expert_loop / 1000.0) / 1024.0
                } else { 0.0 };
                // Upper bound: assume cold expert time is purely DMA-bound
                // cold_expert_fraction * expert_loop_time = DMA time
                let cold_frac = avg_cold / (avg_cold + avg_hcs).max(1.0);
                let est_dma_time_ms = avg_expert_loop * cold_frac;
                let est_pcie_bw = if est_dma_time_ms > 0.001 {
                    dma_mb / (est_dma_time_ms / 1000.0) / 1024.0
                } else { 0.0 };
                log::info!("├─────────────────────────────────────────────────┤");
                log::info!("│  PCIe DMA (non-serialized):                     │");
                log::info!("│    Cold experts/tok: {:.1} ({:.0} DMA calls)      │", avg_cold, avg_dma_calls);
                log::info!("│    HCS experts/tok:  {:.1} ({} cached)            │", avg_hcs, hcs_cached);
                log::info!("│    DMA bytes/tok:    {:.2} MB                     │", dma_mb);
                log::info!("│    HCS hit/miss:     {}/{}                        │", hcs_hits, hcs_misses);
                let bytes_per_call = if avg_dma_calls > 0.0 { avg_dma_bytes / avg_dma_calls } else { 0.0 };
                log::info!("│    Avg DMA call size: {:.1} KB                   │", bytes_per_call / 1024.0);
                log::info!("│    Min PCIe BW:      {:.1} GB/s (bytes/loop_time)│", min_pcie_bw);
                log::info!("│    Est PCIe BW:      {:.1} GB/s (cold fraction)  │", est_pcie_bw);
                log::info!("└─────────────────────────────────────────────────┘");
            }
        }

        // Finish activation tracking and run dynamic rebalance
        if let Some(ref mut hcs) = self.graph.as_mut().unwrap().hcs {
            hcs.finish_prompt();
        }
        if generated > 0 {
            let (swapped, rebalance_ms) = self.hcs_rebalance_internal();
            if swapped > 0 {
                log::info!("HCS rebalance after decode: {} experts swapped in {:.1}ms", swapped, rebalance_ms);
            }
        }

        generated
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

        // Pin shared expert in VRAM if present (Certainty Rule: always accessed, zero DMA at runtime)
        while graph.shared_expert_vram.len() <= layer_idx {
            graph.shared_expert_vram.push(None);
        }
        if let Some(ref se) = graph.moe_layers[layer_idx].as_ref().unwrap().shared {
            let total_bytes_se = se.w13_packed_bytes + se.w13_scales_bytes
                + se.w2_packed_bytes + se.w2_scales_bytes;
            let align = 512usize;
            let alloc_bytes = (total_bytes_se + align - 1) & !(align - 1);

            let d_buf = self.device.alloc_zeros::<u8>(alloc_bytes)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Shared expert VRAM alloc ({} bytes): {:?}", alloc_bytes, e)))?;

            let w13_packed_offset = 0;
            let w13_scales_offset = se.w13_packed_bytes;
            let w2_packed_offset = w13_scales_offset + se.w13_scales_bytes;
            let w2_scales_offset = w2_packed_offset + se.w2_packed_bytes;

            // Synchronous H2D copy (one-time setup)
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
                            format!("Shared expert H2D: {:?}", err)));
                    }
                    Ok(())
                };
                copy(w13_packed_offset, se.w13_packed_ptr, se.w13_packed_bytes)?;
                copy(w13_scales_offset, se.w13_scales_ptr, se.w13_scales_bytes)?;
                copy(w2_packed_offset, se.w2_packed_ptr, se.w2_packed_bytes)?;
                copy(w2_scales_offset, se.w2_scales_ptr, se.w2_scales_bytes)?;
            }

            log::info!("Shared expert layer {} pinned in VRAM: {:.1} KB",
                layer_idx, alloc_bytes as f64 / 1024.0);

            graph.shared_expert_vram[layer_idx] = Some(HcsCacheEntry {
                d_buf: Some(d_buf),
                w13_packed_offset,
                w13_packed_size: se.w13_packed_bytes,
                w13_scales_offset,
                w13_scales_size: se.w13_scales_bytes,
                w2_packed_offset,
                w2_packed_size: se.w2_packed_bytes,
                w2_scales_offset,
                w2_scales_size: se.w2_scales_bytes,
                ext_w13_packed: 0, ext_w13_scales: 0, ext_w2_packed: 0, ext_w2_scales: 0,
                pool_slot: None,
            });
        }

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
        // Shared memory: input BF16 [K*2] + inv_weight_perm [1024*4] + inv_scale_perm [64*4]
        let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_tiles as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem_bytes,
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

    /// Launch fused silu_mul + w2 GEMV + weighted_add.
    /// Replaces 3 separate kernel launches (silu_mul, w2 GEMV, weighted_add) with 1.
    /// gate_up_ptr: [2*K] BF16 output from w13 GEMV
    /// accum_ptr: [N] BF16 moe_out accumulator (read-modify-write)
    fn launch_fused_silu_accum(
        &self,
        w2_packed_ptr: u64,
        w2_scales_ptr: u64,
        gate_up_ptr: u64,
        accum_ptr: u64,
        inv_weight_perm_ptr: u64,
        inv_scale_perm_ptr: u64,
        k: usize,       // intermediate_size
        n: usize,        // hidden_size
        group_size: usize,
        weight: f32,
        weight_ptr: u64, // optional device ptr: if non-zero, kernel reads sigmoid(*weight_ptr) instead of weight
        kernels: &CachedKernels,
    ) -> PyResult<()> {
        let n_tiles = (n + 15) / 16;
        let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_tiles as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem_bytes,
        };

        unsafe {
            kernels.fused_silu_accum.clone().launch(cfg, (
                w2_packed_ptr,
                w2_scales_ptr,
                gate_up_ptr,
                accum_ptr,
                inv_weight_perm_ptr,
                inv_scale_perm_ptr,
                k as i32,
                n as i32,
                group_size as i32,
                weight,
                weight_ptr,
            )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("fused_silu_accum launch: {:?}", e)))?;
        }

        Ok(())
    }

    /// Calculate optimal K_SPLITS for v2 kernels based on problem size and GPU SM count.
    fn calc_k_splits(&self, k: usize, n: usize) -> usize {
        let graph = match self.graph.as_ref() {
            Some(g) => g,
            None => return 1,
        };
        let num_sms = graph.num_sms;
        let k_tiles = k / 16;
        // Maximum K_SPLITS: each k_slice (16 per block) needs at least 1 tile
        let max_ksplits = k_tiles / 16;
        if max_ksplits <= 1 { return 1; }

        let n_tiles = (n + 15) / 16;
        // Target: 4 blocks per SM for good occupancy
        let target_blocks = num_sms * 4;
        let desired = (target_blocks + n_tiles - 1) / n_tiles;
        desired.clamp(1, max_ksplits.min(8))
    }

    /// Launch Marlin GEMV v2 with K-splitting.
    /// Output goes to d_v2_partial as FP32 [k_splits, N].
    /// Caller must then launch reduce_ksplits_bf16 to get final BF16 output.
    fn launch_marlin_gemv_v2(
        &self,
        packed_ptr: u64,
        scales_ptr: u64,
        input_ptr: u64,
        partial_out_ptr: u64,
        inv_weight_perm_ptr: u64,
        inv_scale_perm_ptr: u64,
        k: usize,
        n: usize,
        group_size: usize,
        k_splits: usize,
        kernels: &CachedKernels,
    ) -> PyResult<()> {
        let n_tiles = (n + 15) / 16;
        // Shared mem: input BF16 [K*2] + inv_wperm [1024*4] + inv_sperm [64*4] + reduce [16*16*4]
        let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4 + 16 * 16 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_tiles as u32, k_splits as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem_bytes,
        };

        unsafe {
            kernels.marlin_gemv_int4_v2.clone().launch(cfg, (
                packed_ptr,
                scales_ptr,
                input_ptr,
                partial_out_ptr,
                inv_weight_perm_ptr,
                inv_scale_perm_ptr,
                k as i32,
                n as i32,
                group_size as i32,
                k_splits as i32,
            )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("marlin_gemv_int4_v2 launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Launch reduce kernel to sum K-split partial sums to BF16 output.
    fn launch_reduce_ksplits_bf16(
        &self,
        output_ptr: u64,
        partial_ptr: u64,
        n: usize,
        k_splits: usize,
        kernels: &CachedKernels,
    ) -> PyResult<()> {
        let cfg = LaunchConfig {
            grid_dim: (((n + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            kernels.reduce_ksplits_bf16.clone().launch(cfg, (
                output_ptr,
                partial_ptr,
                n as i32,
                k_splits as i32,
            )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("reduce_ksplits_bf16 launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Launch fused silu+w2+accum v2 with K-splitting.
    /// Outputs FP32 partial sums to d_v2_partial.
    fn launch_fused_silu_accum_v2(
        &self,
        w2_packed_ptr: u64,
        w2_scales_ptr: u64,
        gate_up_ptr: u64,
        partial_out_ptr: u64,
        inv_weight_perm_ptr: u64,
        inv_scale_perm_ptr: u64,
        k: usize,
        n: usize,
        group_size: usize,
        k_splits: usize,
        kernels: &CachedKernels,
    ) -> PyResult<()> {
        let n_tiles = (n + 15) / 16;
        let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4 + 16 * 16 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_tiles as u32, k_splits as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem_bytes,
        };

        unsafe {
            kernels.fused_silu_accum_v2.clone().launch(cfg, (
                w2_packed_ptr,
                w2_scales_ptr,
                gate_up_ptr,
                partial_out_ptr,
                inv_weight_perm_ptr,
                inv_scale_perm_ptr,
                k as i32,
                n as i32,
                group_size as i32,
                k_splits as i32,
            )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("fused_silu_accum_v2 launch: {:?}", e)))?;
        }
        Ok(())
    }

    /// Launch reduce kernel with weighted accumulation to BF16 accum buffer.
    fn launch_reduce_ksplits_weighted_accum(
        &self,
        accum_ptr: u64,
        partial_ptr: u64,
        n: usize,
        k_splits: usize,
        weight: f32,
        kernels: &CachedKernels,
    ) -> PyResult<()> {
        let cfg = LaunchConfig {
            grid_dim: (((n + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            kernels.reduce_ksplits_weighted_accum_bf16.clone().launch(cfg, (
                accum_ptr,
                partial_ptr,
                n as i32,
                k_splits as i32,
                weight,
            )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("reduce_ksplits_weighted_accum launch: {:?}", e)))?;
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
    /// Debug: run single expert w13 GEMV, return gate_up BF16.
    fn test_single_expert_w13_impl(
        &self,
        graph: &GpuDecodeGraph,
        layer_idx: usize,
        expert_id: usize,
    ) -> PyResult<Vec<u16>> {
        let moe = graph.moe_layers.get(layer_idx)
            .and_then(|m| m.as_ref())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                format!("MoE layer {} not registered", layer_idx)))?;

        let expert = &moe.experts[expert_id];
        let hs = graph.hidden_size;
        let intermediate = graph.intermediate_size;
        let gs = graph.group_size;
        let w13_n = 2 * intermediate;
        let inv_wp = *graph.d_inv_weight_perm.device_ptr();
        let inv_sp = *graph.d_inv_scale_perm.device_ptr();
        let copy_stream = self.copy_stream.0;
        let default_stream: cuda_sys::CUstream = std::ptr::null_mut();

        // Check HCS first
        let hcs_ptrs = if let Some(ref hcs) = graph.hcs {
            hcs.get(layer_idx, expert_id).map(|entry| (
                entry.w13_packed_ptr(), entry.w13_scales_ptr(),
            ))
        } else {
            None
        };

        let (w13p, w13s) = if let Some((p, s)) = hcs_ptrs {
            (p, s)
        } else {
            // DMA from system RAM to buf[0]
            let base = *graph.d_expert_buf[0].device_ptr();
            let w13p_off = graph.expert_buf_w13p_offset;
            let w13s_off = graph.expert_buf_w13s_offset;
            unsafe {
                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    base + w13p_off as u64, expert.w13_packed_ptr as *const std::ffi::c_void,
                    expert.w13_packed_bytes, copy_stream);
                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    base + w13s_off as u64, expert.w13_scales_ptr as *const std::ffi::c_void,
                    expert.w13_scales_bytes, copy_stream);
                let mut ev: cuda_sys::CUevent = std::ptr::null_mut();
                cuda_sys::lib().cuEventCreate(&mut ev,
                    cuda_sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32);
                cuda_sys::lib().cuEventRecord(ev, copy_stream);
                cuda_sys::lib().cuStreamWaitEvent(default_stream, ev, 0);
                cuda_sys::lib().cuEventDestroy_v2(ev);
            }
            (base + w13p_off as u64, base + w13s_off as u64)
        };

        // Always use v1 kernel for debug (simpler, no K-split)
        self.launch_marlin_gemv_raw(
            w13p, w13s,
            *graph.d_hidden.device_ptr(),
            *graph.d_expert_gate_up.device_ptr(),
            inv_wp, inv_sp,
            hs, w13_n, gs,
        )?;
        self.device.synchronize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        // Download gate_up
        let size = w13_n;
        let mut out = vec![0u16; size];
        unsafe {
            let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                *graph.d_expert_gate_up.device_ptr(),
                size * 2);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("D2H gate_up: {:?}", err)));
            }
        }
        Ok(out)
    }

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
        let timing = graph.timing_enabled;

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

        // v2 K-split config for w13 GEMV (only use v2 if k_splits > 1)
        let w13_n = 2 * intermediate;
        let w13_k_tiles = hs / 16;
        let w13_max_ksplits = w13_k_tiles / 16;
        let w13_ksplits = if w13_max_ksplits > 1 {
            let n_tiles = (w13_n + 15) / 16;
            let target = graph.num_sms * 4;
            let desired = (target + n_tiles - 1) / n_tiles;
            desired.clamp(1, w13_max_ksplits.min(8))
        } else {
            1
        };
        let use_v2_w13 = w13_ksplits > 1;
        let partial_ptr = *graph.d_v2_partial.device_ptr();

        #[cfg(feature = "gpu-debug")]
        let t_start = Instant::now();

        // Get cached kernel handles (avoids HashMap lookup per call)
        let k = graph.kernels.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Kernels not cached"))?;

        // Use pre-allocated events if available, otherwise create on demand
        let pre_ev = &graph.pre_events;

        // ── Step 1+2: Gate GEMV (BF16 gate × BF16 hidden → FP32 logits) ──
        // Gate weights stored as BF16 (saves 50% VRAM vs FP32), hidden is already BF16.
        // cuBLAS gemm_ex with BF16×BF16→FP32 + COMPUTE_32F eliminates the bf16_to_fp32 step.
        let logits_ptr = unsafe {
            (*graph.d_fp32_scratch.device_ptr() as *const f32).add(hs) as u64
        };
        {
            let w = &graph.weights[gate_wid];
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            unsafe {
                cublas_result::gemm_ex(
                    *self.blas.handle(),
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    w.rows as i32, 1, w.cols as i32,
                    &alpha as *const f32 as *const std::ffi::c_void,
                    w.ptr as *const std::ffi::c_void, cublas_sys::cudaDataType::CUDA_R_16BF, w.cols as i32,
                    *graph.d_hidden.device_ptr() as *const std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_16BF, hs as i32,
                    &beta as *const f32 as *const std::ffi::c_void,
                    logits_ptr as *mut std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_32F, w.rows as i32,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("cuBLAS gate GEMV (bf16): {:?}", e)))?;
            }
        }

        // ── Step 3: TopK routing ──
        {
            let smem = (ne as u32) * 4;
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: smem,
            };

            if sf == 1 {
                let bias_ptr = if gate_bias_ptr != 0 { gate_bias_ptr } else { 0u64 };
                let corr_ptr = if e_score_corr_ptr != 0 { e_score_corr_ptr } else { 0u64 };
                unsafe {
                    k.sigmoid_topk.clone().launch(cfg, (
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
                unsafe {
                    k.softmax_topk.clone().launch(cfg, (
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
        let t_route_start = Instant::now();
        device.synchronize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let t_after_route_sync = Instant::now();
        if timing { graph.t_moe_route_sync += (t_after_route_sync - t_route_start).as_secs_f64(); }

        #[cfg(feature = "gpu-debug")]
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
                    // Speculative gate GEMV for next layer (BF16 gate × BF16 hidden → FP32).
                    // Matches the main routing path.
                    let output_ptr = unsafe {
                        (*graph.d_fp32_scratch.device_ptr() as *const f32).add(hs) as u64
                    };
                    {
                        let w = &graph.weights[next_gate_wid];
                        let alpha: f32 = 1.0;
                        let beta: f32 = 0.0;
                        unsafe {
                            cublas_result::gemm_ex(
                                *self.blas.handle(),
                                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                w.rows as i32, 1, w.cols as i32,
                                &alpha as *const f32 as *const std::ffi::c_void,
                                w.ptr as *const std::ffi::c_void, cublas_sys::cudaDataType::CUDA_R_16BF, w.cols as i32,
                                *graph.d_hidden.device_ptr() as *const std::ffi::c_void,
                                cublas_sys::cudaDataType::CUDA_R_16BF, hs as i32,
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
            unsafe {
                k.zero_bf16.clone().launch(
                    LaunchConfig::for_num_elems(hs as u32),
                    (*graph.d_moe_out.device_ptr(), hs as i32),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("zero_bf16: {:?}", e)))?;
            }
        }

        // ── Step 6: Double-buffered expert loop with DMA/compute overlap ──
        //
        // True ping-pong: expert N computes from buf[N%2] while expert N+1
        // DMAs into buf[(N+1)%2]. The DMA engine and compute SMs run in
        // parallel on separate hardware. HCS and APFL experts skip DMA entirely.

        #[cfg(feature = "gpu-debug")]
        let t_expert_start = Instant::now();
        #[cfg(feature = "gpu-debug")]
        let mut dma_total = 0.0f64;
        #[cfg(feature = "gpu-debug")]
        let mut compute_total = 0.0f64;
        let mut apfl_hits = 0u32;
        let mut apfl_misses = 0u32;
        let mut hcs_hits = 0u32;

        let default_stream: cuda_sys::CUstream = std::ptr::null_mut();

        // Extract raw event pointers from pre-allocated CudaEvent wrappers
        let ev_dma: [cuda_sys::CUevent; 2];
        let ev_compute: [cuda_sys::CUevent; 2];
        if let Some(ref pe) = pre_ev {
            ev_dma = [pe[0].0, pe[1].0];
            ev_compute = [pe[2].0, pe[3].0];
        } else {
            // Fallback: create on demand
            unsafe {
                let flags = cuda_sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32;
                let mut events = [std::ptr::null_mut(); 4];
                for e in events.iter_mut() {
                    cuda_sys::lib().cuEventCreate(e, flags);
                }
                ev_dma = [events[0], events[1]];
                ev_compute = [events[2], events[3]];
            }
        }

        // Double-buffer base pointers and offsets
        let use_double_buf = graph.expert_buf_total_size > 0;
        let buf_base = [
            *graph.d_expert_buf[0].device_ptr(),
            *graph.d_expert_buf[1].device_ptr(),
        ];
        let w13p_off = graph.expert_buf_w13p_offset;
        let w13s_off = graph.expert_buf_w13s_offset;
        let w2p_off = graph.expert_buf_w2p_offset;
        let w2s_off = graph.expert_buf_w2s_offset;

        // Legacy single-buffer pointers (fallback if double-buffer not sized)
        let buf_w13_packed = *graph.d_expert_buf_a0.device_ptr();
        let buf_w13_scales = *graph.d_expert_buf_b0.device_ptr();
        let buf_w2_packed = *graph.d_expert_buf_a1.device_ptr();
        let buf_w2_scales = *graph.d_expert_buf_b1.device_ptr();

        // Track which ping-pong slot was last used for DMA (for compute/DMA overlap)
        let mut dma_expert_count = 0u32;

        #[cfg(feature = "gpu-debug")]
        if layer_idx == 0 {
            let weight_sum: f32 = (0..topk).map(|j| graph.h_topk_weights[j]).sum();
            log::info!("DBG MoE L{} routing: weight_sum={:.4}, ids={:?}, weights={:?}",
                layer_idx, weight_sum,
                &graph.h_topk_ids[..topk], &graph.h_topk_weights[..topk]);
        }

        let t_expert_loop_start = Instant::now();

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
                if timing { graph.dma_hcs_experts += 1; }

                // w13 GEMV: hidden -> gate_up (use v2 K-split if beneficial)
                if use_v2_w13 {
                    self.launch_marlin_gemv_v2(
                        w13p, w13s,
                        *graph.d_hidden.device_ptr(),
                        partial_ptr, inv_wp, inv_sp,
                        hs, w13_n, gs, w13_ksplits, k,
                    )?;
                    self.launch_reduce_ksplits_bf16(
                        *graph.d_expert_gate_up.device_ptr(),
                        partial_ptr,
                        w13_n, w13_ksplits, k,
                    )?;
                } else {
                    self.launch_marlin_gemv_raw(
                        w13p, w13s,
                        *graph.d_hidden.device_ptr(),
                        *graph.d_expert_gate_up.device_ptr(),
                        inv_wp, inv_sp,
                        hs, w13_n, gs,
                    )?;
                }

                // Fused: silu_mul + w2 GEMV + weighted_add (3 launches -> 1)
                self.launch_fused_silu_accum(
                    w2p, w2s,
                    *graph.d_expert_gate_up.device_ptr(),
                    *graph.d_moe_out.device_ptr(),
                    inv_wp, inv_sp,
                    intermediate, hs, gs,
                    weight, 0u64,
                    k,
                )?;
            } else if use_double_buf {
                // ── Priority 3: Double-buffered DMA with ping-pong overlap ──
                //
                // Expert N DMAs to buf[slot], expert N-1 computes from buf[prev_slot].
                // The copy engine and compute SMs run concurrently on different buffers.
                apfl_misses += 1;
                if timing {
                    let dma_bytes = expert.w13_packed_bytes + expert.w13_scales_bytes
                                  + expert.w2_packed_bytes + expert.w2_scales_bytes;
                    graph.dma_bytes_total += dma_bytes as u64;
                    graph.dma_call_count += 4;
                    graph.dma_cold_experts += 1;
                }

                let slot = (dma_expert_count % 2) as usize;

                // Wait for this buffer's previous compute to finish (free the buffer)
                if dma_expert_count >= 2 {
                    unsafe {
                        cuda_sys::lib().cuStreamWaitEvent(copy_stream, ev_compute[slot], 0);
                    }
                }

                // DMA all 4 weight arrays to contiguous buffer[slot]
                unsafe {
                    let base = buf_base[slot];
                    let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        base + w13p_off as u64, expert.w13_packed_ptr as *const std::ffi::c_void,
                        expert.w13_packed_bytes, copy_stream);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            format!("DMA w13p[{}]: {:?}", eid, err)));
                    }
                    let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        base + w13s_off as u64, expert.w13_scales_ptr as *const std::ffi::c_void,
                        expert.w13_scales_bytes, copy_stream);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            format!("DMA w13s[{}]: {:?}", eid, err)));
                    }
                    let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        base + w2p_off as u64, expert.w2_packed_ptr as *const std::ffi::c_void,
                        expert.w2_packed_bytes, copy_stream);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            format!("DMA w2p[{}]: {:?}", eid, err)));
                    }
                    let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        base + w2s_off as u64, expert.w2_scales_ptr as *const std::ffi::c_void,
                        expert.w2_scales_bytes, copy_stream);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            format!("DMA w2s[{}]: {:?}", eid, err)));
                    }
                    cuda_sys::lib().cuEventRecord(ev_dma[slot], copy_stream);
                }

                // Wait for THIS expert's DMA to complete before computing
                // GPU-side only: default_stream waits for copy_stream's DMA event.
                // No host sync here -- that would serialize DMA and compute, destroying overlap.
                unsafe {
                    cuda_sys::lib().cuStreamWaitEvent(default_stream, ev_dma[slot], 0);
                }

                // Compute from buf[slot]
                let base = buf_base[slot];
                // w13 GEMV: hidden -> gate_up (v2 K-split if beneficial)
                if use_v2_w13 {
                    self.launch_marlin_gemv_v2(
                        base + w13p_off as u64, base + w13s_off as u64,
                        *graph.d_hidden.device_ptr(),
                        partial_ptr, inv_wp, inv_sp,
                        hs, w13_n, gs, w13_ksplits, k,
                    )?;
                    self.launch_reduce_ksplits_bf16(
                        *graph.d_expert_gate_up.device_ptr(),
                        partial_ptr,
                        w13_n, w13_ksplits, k,
                    )?;
                } else {
                    self.launch_marlin_gemv_raw(
                        base + w13p_off as u64, base + w13s_off as u64,
                        *graph.d_hidden.device_ptr(),
                        *graph.d_expert_gate_up.device_ptr(),
                        inv_wp, inv_sp,
                        hs, w13_n, gs,
                    )?;
                }

                #[cfg(feature = "gpu-debug")]
                if layer_idx == 0 && i == 0 {
                    device.synchronize().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
                    let mut gu = vec![0u16; 4];
                    unsafe {
                        let _ = cuda_sys::lib().cuMemcpyDtoH_v2(
                            gu.as_mut_ptr() as *mut std::ffi::c_void,
                            *graph.d_expert_gate_up.device_ptr(), 8);
                    }
                    let vals: Vec<f32> = gu.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
                    log::info!("DBG MoE L0 expert[{}] gate_up[0..4] = [{:.4}, {:.4}, {:.4}, {:.4}], w={:.4}",
                        eid, vals[0], vals[1], vals[2], vals[3], weight);
                }

                // Fused: silu_mul + w2 GEMV + weighted_add (3 launches -> 1)
                self.launch_fused_silu_accum(
                    base + w2p_off as u64, base + w2s_off as u64,
                    *graph.d_expert_gate_up.device_ptr(),
                    *graph.d_moe_out.device_ptr(),
                    inv_wp, inv_sp,
                    intermediate, hs, gs,
                    weight, 0u64,
                    k,
                )?;

                #[cfg(feature = "gpu-debug")]
                if layer_idx == 0 && i == 0 {
                    device.synchronize().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
                    let mut mo = vec![0u16; 4];
                    unsafe {
                        let _ = cuda_sys::lib().cuMemcpyDtoH_v2(
                            mo.as_mut_ptr() as *mut std::ffi::c_void,
                            *graph.d_moe_out.device_ptr(), 8);
                    }
                    let vals: Vec<f32> = mo.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
                    log::info!("DBG MoE L0 moe_out[0..4] after expert[{}] = [{:.6}, {:.6}, {:.6}, {:.6}]",
                        eid, vals[0], vals[1], vals[2], vals[3]);
                }

                // Signal: compute done on this buffer (copy_stream can reuse it)
                unsafe {
                    cuda_sys::lib().cuEventRecord(ev_compute[slot], default_stream);
                }

                dma_expert_count += 1;
            } else {
                // ── Fallback: legacy single-buffer DMA (no ping-pong) ──
                apfl_misses += 1;
                if timing {
                    let dma_bytes = expert.w13_packed_bytes + expert.w13_scales_bytes
                                  + expert.w2_packed_bytes + expert.w2_scales_bytes;
                    graph.dma_bytes_total += dma_bytes as u64;
                    graph.dma_call_count += 4;
                    graph.dma_cold_experts += 1;
                }

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
                    let ev_dma_w13 = ev_dma[0];
                    cuda_sys::lib().cuEventRecord(ev_dma_w13, copy_stream);
                    cuda_sys::lib().cuStreamWaitEvent(default_stream, ev_dma_w13, 0);
                }

                // w13 GEMV: hidden -> gate_up (v2 K-split if beneficial)
                if use_v2_w13 {
                    self.launch_marlin_gemv_v2(
                        buf_w13_packed, buf_w13_scales,
                        *graph.d_hidden.device_ptr(),
                        partial_ptr, inv_wp, inv_sp,
                        hs, w13_n, gs, w13_ksplits, k,
                    )?;
                    self.launch_reduce_ksplits_bf16(
                        *graph.d_expert_gate_up.device_ptr(),
                        partial_ptr,
                        w13_n, w13_ksplits, k,
                    )?;
                } else {
                    self.launch_marlin_gemv_raw(
                        buf_w13_packed, buf_w13_scales,
                        *graph.d_hidden.device_ptr(),
                        *graph.d_expert_gate_up.device_ptr(),
                        inv_wp, inv_sp,
                        hs, w13_n, gs,
                    )?;
                }

                // DMA w2 weights while w13 GEMV runs
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
                    let ev_dma_w2 = ev_dma[1];
                    cuda_sys::lib().cuEventRecord(ev_dma_w2, copy_stream);
                    cuda_sys::lib().cuStreamWaitEvent(default_stream, ev_dma_w2, 0);
                }

                // Fused: silu_mul + w2 GEMV + weighted_add (3 launches -> 1)
                self.launch_fused_silu_accum(
                    buf_w2_packed, buf_w2_scales,
                    *graph.d_expert_gate_up.device_ptr(),
                    *graph.d_moe_out.device_ptr(),
                    inv_wp, inv_sp,
                    intermediate, hs, gs,
                    weight, 0u64,
                    k,
                )?;
            }
        }

        // Wait for all expert work to complete
        if timing {
            device.synchronize()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            graph.t_moe_expert_loop += (Instant::now() - t_expert_loop_start).as_secs_f64();
        }
        #[cfg(feature = "gpu-debug")]
        {
            if !timing {
                device.synchronize()
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            }
            let expert_elapsed = t_expert_start.elapsed().as_secs_f64() * 1000.0;
            dma_total = expert_elapsed * 0.87;
            compute_total = expert_elapsed * 0.10;
        }

        let t_shared_start = Instant::now();

        // ── Step 7: Shared expert (if any) ──
        // Priority: VRAM-resident (pinned at registration) > DMA fallback
        if moe.shared.is_some() {
            let se_vram = graph.shared_expert_vram.get(layer_idx).and_then(|e| e.as_ref());

            let (w13p, w13s, w2p, w2s) = if let Some(entry) = se_vram {
                // VRAM-resident: zero DMA, full bandwidth
                (entry.w13_packed_ptr(), entry.w13_scales_ptr(),
                 entry.w2_packed_ptr(), entry.w2_scales_ptr())
            } else {
                // Fallback: DMA from system RAM (should not happen if registration worked)
                let shared = moe.shared.as_ref().unwrap();
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
                    cuda_sys::lib().cuEventRecord(ev_dma[0], copy_stream);
                    cuda_sys::lib().cuStreamWaitEvent(default_stream, ev_dma[0], 0);
                }
                (buf_w13_packed, buf_w13_scales, buf_w2_packed, buf_w2_scales)
            };

            // w13 GEMV: hidden -> gate_up
            self.launch_marlin_gemv_raw(
                w13p, w13s,
                *graph.d_hidden.device_ptr(),
                *graph.d_expert_gate_up.device_ptr(),
                inv_wp, inv_sp,
                hs, 2 * intermediate, gs,
            )?;

            // w2: DMA fallback path needs separate DMA for w2
            if se_vram.is_none() {
                let shared = moe.shared.as_ref().unwrap();
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
                    cuda_sys::lib().cuEventRecord(ev_dma[1], copy_stream);
                    cuda_sys::lib().cuStreamWaitEvent(default_stream, ev_dma[1], 0);
                }
            }

            // Compute sigmoid gate weight on GPU (no D2H sync needed).
            // The gate must only scale the shared expert, not the routed experts.
            // Python does: output = routed + sigmoid(gate) * shared
            // Gate GEMV produces FP32 logit on device; the fused kernel reads it
            // and applies sigmoid internally.
            let gate_weight_ptr = if let Some(sg_wid) = moe.shared_gate_wid {
                let sg_w = &graph.weights[sg_wid];
                // GEMV: gate_weight[1, hs] @ d_hidden[hs] -> d_scratch[0] (1 FP32 scalar)
                self.gemv_bf16_to_f32(
                    sg_w,
                    *graph.d_hidden.device_ptr(),
                    *graph.d_scratch.device_ptr(),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("shared gate GEMV: {}", e)))?;
                *graph.d_scratch.device_ptr()  // FP32 gate logit on device
            } else {
                0u64  // no gate -> kernel uses weight arg directly
            };

            let shared_weight = if gate_weight_ptr != 0 { 0.0f32 } else { 1.0f32 };

            // Fused: silu_mul + w2 GEMV + add to accumulator
            // When gate_weight_ptr != 0, kernel reads sigmoid(*gate_weight_ptr) as weight
            self.launch_fused_silu_accum(
                w2p, w2s,
                *graph.d_expert_gate_up.device_ptr(),
                *graph.d_moe_out.device_ptr(),
                inv_wp, inv_sp,
                intermediate, hs, gs,
                shared_weight, gate_weight_ptr,
                k,
            )?;
        }

        // ── Step 8: Scale by routed_scaling_factor ──
        if rsf != 1.0 {
            unsafe {
                k.scale_bf16.clone().launch(
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
            // No separate sync -- combined sync at end
        }

        // Final sync: ensure shared expert + scale complete (debug builds only;
        // in release, same-stream ordering guarantees correctness without sync)
        #[cfg(feature = "gpu-debug")]
        device.synchronize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        if timing {
            graph.t_moe_shared += (Instant::now() - t_shared_start).as_secs_f64();
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

        // Events are pre-allocated and reused across calls (no cleanup needed)

        #[cfg(feature = "gpu-debug")]
        {
            let total = t_start.elapsed().as_secs_f64() * 1000.0;
            return Ok((t_route, dma_total, compute_total, total));
        }
        #[cfg(not(feature = "gpu-debug"))]
        Ok((0.0, 0.0, 0.0, 0.0))
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
        // Shared memory: input BF16 [K*2] + inv_weight_perm [1024*4] + inv_scale_perm [64*4]
        let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_tiles as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem_bytes,
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

        // Step 1: configure buffers (only if not already configured by Python setup)
        if self.graph.is_none() {
            self.configure(
                hidden_size, num_layers, vocab_size, 1e-6,
                topk, intermediate_size, hidden_size * 3, group_size,
            )?;
        }

        // Step 2: for each MoE layer, upload gate weights to VRAM and register expert pointers
        let num_routing = engine.num_routing_layers();
        let num_gpu_layers = store.experts_gpu.len();
        let n_moe_layers = num_routing.min(num_gpu_layers);

        log::info!(
            "setup_from_engine: {} routing layers, {} GPU expert layers, using {}",
            num_routing, num_gpu_layers, n_moe_layers,
        );

        let mut max_expert_bytes = 0usize;
        let first_k_dense = config.first_k_dense_replace;

        for moe_idx in 0..n_moe_layers {
            // Map MoE layer index to absolute layer index
            // (e.g. QCN has first_k_dense_replace=1, so moe_idx 0 = abs layer 1)
            let abs_layer_idx = moe_idx + first_k_dense;

            // Upload gate weight as FP32 to VRAM
            let (gate_bf16, correction_bias) = engine.get_routing_weights(moe_idx)
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("No routing weights for MoE layer {}", moe_idx)))?;

            // Upload gate as BF16 directly (saves VRAM, enables bf16*bf16->fp32 GEMV
            // which eliminates the separate bf16_to_fp32 conversion step)
            let d_gate = self.device.htod_copy(gate_bf16.to_vec())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            let gate_wid = {
                let graph = self.graph.as_mut().unwrap();
                let wid = graph.weights.len();
                graph.weights.push(GpuWeight {
                    ptr: *d_gate.device_ptr(),
                    rows: n_experts,
                    cols: hidden_size,
                    dtype: 0, // BF16
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
                log::info!("MoE layer {} shared expert: w13p={} w13s={} w2p={} w2s={}",
                    abs_layer_idx, se.w13_packed.len(), se.w13_scales.len(),
                    se.w2_packed.len(), se.w2_scales.len());
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
                abs_layer_idx, expert_ptrs, shared_ptrs,
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

        // Step 4: Pin expert weight memory for async DMA (page-lock for full PCIe bandwidth)
        // Without pinning, CUDA must bounce through a staging buffer, halving effective bandwidth.
        let t_pin = std::time::Instant::now();
        let mut pinned_regions = 0usize;
        let mut pinned_bytes = 0usize;
        let mut pin_failures = 0usize;

        for moe_idx in 0..n_moe_layers {
            let gpu_experts = &store.experts_gpu[moe_idx];
            for expert in gpu_experts.iter() {
                // Pin each weight buffer (w13_packed, w13_scales, w2_packed, w2_scales)
                let regions: [(usize, usize); 4] = [
                    (expert.w13_packed.as_ptr() as usize, expert.w13_packed.len() * 4),
                    (expert.w13_scales.as_ptr() as usize, expert.w13_scales.len() * 2),
                    (expert.w2_packed.as_ptr() as usize, expert.w2_packed.len() * 4),
                    (expert.w2_scales.as_ptr() as usize, expert.w2_scales.len() * 2),
                ];
                for (ptr, size) in regions {
                    if size == 0 { continue; }
                    let err = unsafe {
                        cuda_sys::lib().cuMemHostRegister_v2(
                            ptr as *mut std::ffi::c_void,
                            size,
                            0, // CU_MEMHOSTREGISTER_DEFAULT
                        )
                    };
                    if err == cuda_sys::CUresult::CUDA_SUCCESS {
                        pinned_regions += 1;
                        pinned_bytes += size;
                    } else {
                        pin_failures += 1;
                        if pin_failures == 1 {
                            log::warn!("First pin failure at moe_idx={}: {:?} (size={})",
                                moe_idx, err, size);
                        }
                    }
                }
            }
            // Also pin shared expert buffers
            if moe_idx < store.shared_experts_gpu.len() {
                let se = &store.shared_experts_gpu[moe_idx];
                let regions: [(usize, usize); 4] = [
                    (se.w13_packed.as_ptr() as usize, se.w13_packed.len() * 4),
                    (se.w13_scales.as_ptr() as usize, se.w13_scales.len() * 2),
                    (se.w2_packed.as_ptr() as usize, se.w2_packed.len() * 4),
                    (se.w2_scales.as_ptr() as usize, se.w2_scales.len() * 2),
                ];
                for (ptr, size) in regions {
                    if size == 0 { continue; }
                    let err = unsafe {
                        cuda_sys::lib().cuMemHostRegister_v2(
                            ptr as *mut std::ffi::c_void,
                            size,
                            0,
                        )
                    };
                    if err == cuda_sys::CUresult::CUDA_SUCCESS {
                        pinned_regions += 1;
                        pinned_bytes += size;
                    } else {
                        pin_failures += 1;
                    }
                }
            }
        }

        let pin_elapsed = t_pin.elapsed().as_secs_f64();
        log::info!(
            "Expert memory pinning: {} regions ({:.1} GB) pinned in {:.1}s, {} failures",
            pinned_regions, pinned_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            pin_elapsed, pin_failures,
        );

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
            Path::new(model_dir), 128, None, None, 4, 4, false,
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
            Path::new(model_dir), 128, None, None, 4, 4, false,
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

    /// Initialize pool-based HCS with dynamic eviction support.
    fn hcs_pool_init_internal(
        &mut self,
        ranking: Vec<(usize, usize)>,
        budget_mb: usize,
        headroom_mb: usize,
        window_size: usize,
        replacement_pct: usize,
    ) -> PyResult<String> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;

        if graph.moe_layers.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No MoE layers registered. Call setup_from_engine first."));
        }

        // Calculate per-expert VRAM size
        let first_moe = graph.moe_layers.iter()
            .find_map(|m| m.as_ref())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No MoE layers found"))?;
        let first_expert = &first_moe.experts[0];
        let expert_bytes = first_expert.w13_packed_bytes + first_expert.w13_scales_bytes
            + first_expert.w2_packed_bytes + first_expert.w2_scales_bytes;
        let align = 512usize;
        let slot_size = (expert_bytes + align - 1) & !(align - 1);

        // Determine budget
        let budget_bytes = if budget_mb > 0 {
            budget_mb * 1024 * 1024
        } else {
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

        let num_slots = budget_bytes / slot_size;
        if num_slots == 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Budget too small for even one expert ({} bytes need {} bytes)",
                    budget_bytes, slot_size)));
        }

        let pool_alloc_bytes = num_slots * slot_size;

        // Determine max experts per layer for bitset indexing
        let num_experts_per_layer = graph.moe_layers.iter()
            .filter_map(|m| m.as_ref())
            .map(|m| m.num_experts)
            .max()
            .unwrap_or(0);
        let num_layers = graph.moe_layers.len();
        let total_bits = num_layers * num_experts_per_layer;
        let bitset_words = (total_bits + 63) / 64;

        // Total unique experts
        let total_experts: usize = graph.moe_layers.iter()
            .filter_map(|m| m.as_ref())
            .map(|m| m.num_experts)
            .sum();

        log::info!("HCS pool: allocating {:.1} MB ({} slots x {:.1} KB/slot)",
            pool_alloc_bytes as f64 / (1024.0 * 1024.0),
            num_slots, slot_size as f64 / 1024.0);

        // Allocate the pool
        let pool_buf = self.device.alloc_zeros::<u8>(pool_alloc_bytes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("HCS pool alloc ({} MB): {:?}", pool_alloc_bytes / (1024 * 1024), e)))?;
        let pool_base = *pool_buf.device_ptr();

        // Build free slot stack (all slots initially free)
        let mut free_slots: Vec<usize> = (0..num_slots).rev().collect();
        let mut slot_to_expert: Vec<Option<(usize, usize)>> = vec![None; num_slots];

        // Fill slots from ranking via H2D DMA
        let t0 = std::time::Instant::now();
        let mut loaded = 0usize;
        let mut cache = std::collections::HashMap::new();

        for &(layer_idx, expert_idx) in &ranking {
            if free_slots.is_empty() {
                break;
            }
            // Validate the (layer, expert) pair
            let moe = match graph.moe_layers.get(layer_idx).and_then(|m| m.as_ref()) {
                Some(m) => m,
                None => continue,
            };
            if expert_idx >= moe.experts.len() {
                continue;
            }

            let slot = free_slots.pop().unwrap();
            let expert = &moe.experts[expert_idx];
            let dst = pool_base + (slot as u64 * slot_size as u64);

            // Contiguous layout: w13p | w13s | w2p | w2s
            let w13p_off = 0u64;
            let w13s_off = expert.w13_packed_bytes as u64;
            let w2p_off = w13s_off + expert.w13_scales_bytes as u64;
            let w2s_off = w2p_off + expert.w2_packed_bytes as u64;

            unsafe {
                let mut ok = true;
                for &(off, src_ptr, bytes) in &[
                    (w13p_off, expert.w13_packed_ptr, expert.w13_packed_bytes),
                    (w13s_off, expert.w13_scales_ptr, expert.w13_scales_bytes),
                    (w2p_off, expert.w2_packed_ptr, expert.w2_packed_bytes),
                    (w2s_off, expert.w2_scales_ptr, expert.w2_scales_bytes),
                ] {
                    let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                        dst + off,
                        src_ptr as *const std::ffi::c_void,
                        bytes,
                    );
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        log::warn!("HCS pool H2D copy failed for L{}E{}: {:?}", layer_idx, expert_idx, err);
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    free_slots.push(slot);
                    continue;
                }
            }

            let entry = HcsCacheEntry {
                d_buf: None,
                w13_packed_offset: 0, w13_packed_size: 0,
                w13_scales_offset: 0, w13_scales_size: 0,
                w2_packed_offset: 0, w2_packed_size: 0,
                w2_scales_offset: 0, w2_scales_size: 0,
                ext_w13_packed: dst + w13p_off,
                ext_w13_scales: dst + w13s_off,
                ext_w2_packed: dst + w2p_off,
                ext_w2_scales: dst + w2s_off,
                pool_slot: Some(slot),
            };
            cache.insert((layer_idx, expert_idx), entry);
            slot_to_expert[slot] = Some((layer_idx, expert_idx));
            loaded += 1;
        }

        let load_elapsed = t0.elapsed().as_secs_f64();
        let pct = if total_experts > 0 { loaded as f64 / total_experts as f64 * 100.0 } else { 0.0 };

        // Create HCS state with pool
        let mut hcs = HcsState::new();
        hcs.cache = cache;
        hcs.expert_vram_bytes = slot_size;
        hcs.vram_bytes = pool_alloc_bytes;
        hcs.num_cached = loaded;
        hcs.pool_buf = Some(pool_buf);
        hcs.pool_slot_size = slot_size;
        hcs.pool_num_slots = num_slots;
        hcs.pool_free_slots = free_slots;
        hcs.pool_slot_to_expert = slot_to_expert;
        hcs.current_activations = vec![0u64; bitset_words];
        hcs.num_experts_per_layer = num_experts_per_layer;
        hcs.window_size = window_size;
        hcs.replacement_pct = replacement_pct as f32 / 100.0;
        hcs.rebalance_enabled = true;

        graph.hcs = Some(hcs);

        let msg = format!(
            "HCS pool: {}/{} experts loaded in {:.2}s ({:.1}% coverage), {:.1} MB VRAM, \
             {} free slots, window={}, replace={:.0}%",
            loaded, total_experts, load_elapsed, pct,
            pool_alloc_bytes as f64 / (1024.0 * 1024.0),
            num_slots - loaded, window_size, replacement_pct,
        );
        log::info!("{}", msg);
        Ok(msg)
    }

    /// Rebalance HCS pool based on sliding window activation scores.
    /// Evicts low-scoring pool entries and promotes high-scoring cold experts.
    /// Called automatically after each prompt's decode completes.
    fn hcs_rebalance_internal(&mut self) -> (usize, f64) {
        // Take HCS out of graph to avoid borrow conflicts with moe_layers
        let graph = match self.graph.as_mut() {
            Some(g) => g,
            None => return (0, 0.0),
        };
        let mut hcs = match graph.hcs.take() {
            Some(h) => h,
            None => return (0, 0.0),
        };

        if !hcs.rebalance_enabled || hcs.prompt_history.len() < 2 || hcs.pool_buf.is_none() {
            graph.hcs = Some(hcs);
            return (0, 0.0);
        }

        let t0 = std::time::Instant::now();
        let num_experts = hcs.num_experts_per_layer;
        let window = &hcs.prompt_history;

        // Score all experts: count prompts in window where expert was active
        let mut scored_cached: Vec<(usize, usize, u32)> = Vec::new();
        let mut scored_cold: Vec<(usize, usize, u32)> = Vec::new();

        for (layer_idx, moe_opt) in graph.moe_layers.iter().enumerate() {
            if let Some(moe) = moe_opt {
                for eid in 0..moe.num_experts {
                    let bit_idx = layer_idx * num_experts + eid;
                    let word = bit_idx / 64;
                    let bit = bit_idx % 64;

                    let score: u32 = window.iter()
                        .map(|bits| {
                            if word < bits.len() { ((bits[word] >> bit) & 1) as u32 } else { 0 }
                        })
                        .sum();

                    if let Some(entry) = hcs.cache.get(&(layer_idx, eid)) {
                        if entry.pool_slot.is_some() {
                            scored_cached.push((layer_idx, eid, score));
                        }
                        // External entries are never evicted
                    } else {
                        scored_cold.push((layer_idx, eid, score));
                    }
                }
            }
        }

        // Sort: cached worst-first, cold best-first
        scored_cached.sort_by_key(|x| x.2);
        scored_cold.sort_by(|a, b| b.2.cmp(&a.2));

        let max_replace = ((hcs.pool_num_slots as f32 * hcs.replacement_pct) as usize)
            .min(scored_cached.len())
            .min(scored_cold.len());

        // Only replace where cold expert has strictly higher score
        let mut actual = 0usize;
        for i in 0..max_replace {
            if scored_cold[i].2 > scored_cached[i].2 {
                actual += 1;
            } else {
                break;
            }
        }

        if actual == 0 {
            hcs.total_rebalances += 1;
            graph.hcs = Some(hcs);
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            return (0, elapsed);
        }

        // Perform swaps
        let pool_base = *hcs.pool_buf.as_ref().unwrap().device_ptr();
        let slot_size = hcs.pool_slot_size;

        for i in 0..actual {
            let (ev_layer, ev_eid, _ev_score) = scored_cached[i];
            let (pr_layer, pr_eid, _pr_score) = scored_cold[i];

            // Evict: remove from cache, reclaim slot
            let evicted = hcs.cache.remove(&(ev_layer, ev_eid)).unwrap();
            let slot = evicted.pool_slot.unwrap();
            hcs.pool_slot_to_expert[slot] = None;
            hcs.num_cached -= 1;

            // Promote: DMA new expert into the freed slot
            let moe = graph.moe_layers[pr_layer].as_ref().unwrap();
            let expert = &moe.experts[pr_eid];
            let dst = pool_base + (slot as u64 * slot_size as u64);

            let w13p_off = 0u64;
            let w13s_off = expert.w13_packed_bytes as u64;
            let w2p_off = w13s_off + expert.w13_scales_bytes as u64;
            let w2s_off = w2p_off + expert.w2_packed_bytes as u64;

            let mut dma_ok = true;
            unsafe {
                for &(off, src_ptr, bytes) in &[
                    (w13p_off, expert.w13_packed_ptr, expert.w13_packed_bytes),
                    (w13s_off, expert.w13_scales_ptr, expert.w13_scales_bytes),
                    (w2p_off, expert.w2_packed_ptr, expert.w2_packed_bytes),
                    (w2s_off, expert.w2_scales_ptr, expert.w2_scales_bytes),
                ] {
                    let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                        dst + off,
                        src_ptr as *const std::ffi::c_void,
                        bytes,
                    );
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        log::warn!("HCS rebalance H2D failed for L{}E{}: {:?}", pr_layer, pr_eid, err);
                        dma_ok = false;
                        break;
                    }
                }
            }

            if dma_ok {
                let entry = HcsCacheEntry {
                    d_buf: None,
                    w13_packed_offset: 0, w13_packed_size: 0,
                    w13_scales_offset: 0, w13_scales_size: 0,
                    w2_packed_offset: 0, w2_packed_size: 0,
                    w2_scales_offset: 0, w2_scales_size: 0,
                    ext_w13_packed: dst + w13p_off,
                    ext_w13_scales: dst + w13s_off,
                    ext_w2_packed: dst + w2p_off,
                    ext_w2_scales: dst + w2s_off,
                    pool_slot: Some(slot),
                };
                hcs.cache.insert((pr_layer, pr_eid), entry);
                hcs.pool_slot_to_expert[slot] = Some((pr_layer, pr_eid));
                hcs.num_cached += 1;
            } else {
                // DMA failed: return slot to free list
                hcs.pool_free_slots.push(slot);
            }
        }

        hcs.total_evictions += actual as u64;
        hcs.total_promotions += actual as u64;
        hcs.total_rebalances += 1;

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::info!(
            "HCS rebalance #{}: swapped {}/{} experts in {:.1}ms (window={} prompts, {} cached)",
            hcs.total_rebalances, actual, max_replace, elapsed_ms,
            hcs.prompt_history.len(), hcs.num_cached,
        );

        graph.hcs = Some(hcs);
        (actual, elapsed_ms)
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
            d_buf: Some(d_buf),
            w13_packed_offset,
            w13_packed_size: expert.w13_packed_bytes,
            w13_scales_offset,
            w13_scales_size: expert.w13_scales_bytes,
            w2_packed_offset,
            w2_packed_size: expert.w2_packed_bytes,
            w2_scales_offset,
            w2_scales_size: expert.w2_scales_bytes,
            ext_w13_packed: 0, ext_w13_scales: 0, ext_w2_packed: 0, ext_w2_scales: 0,
            pool_slot: None,
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
            Path::new(model_dir), 128, None, None, 4, 4, false,
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

            // Shared expert pointers (if available)
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

            self.register_moe_layer(
                moe_idx, expert_ptrs, shared_ptrs, n_experts, topk,
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

    /// Benchmark: shared expert VRAM residency vs DMA.
    /// Loads QCN, registers ALL MoE layers WITH shared experts, runs multi-layer
    /// forward twice: once with shared experts DMA'd (baseline), once with VRAM-resident.
    fn bench_shared_expert_residency_internal(
        &mut self,
        model_dir: &str,
        num_tokens: usize,
    ) -> PyResult<String> {
        use std::path::Path;
        use std::time::Instant;

        let mut results = Vec::new();
        let t_start = Instant::now();

        // Step 1: Load model
        let t0 = Instant::now();
        let store = crate::weights::WeightStore::load_from_hf(
            Path::new(model_dir), 128, None, None, 4, 4, false,
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

        // Count shared experts
        let num_shared = store.shared_experts_gpu.iter()
            .filter(|se| !se.w13_packed.is_empty())
            .count();
        results.push(format!("Shared experts available: {}/{}", num_shared, num_moe_layers));

        // Step 2: Configure and register WITH shared experts pinned in VRAM
        self.configure(
            hidden_size, config.num_hidden_layers, 1, 1e-6,
            topk, intermediate_size, hidden_size * 3, group_size,
        )?;

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
                    rows: n_experts, cols: hidden_size, dtype: 1,
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
                if se.w13_packed.is_empty() { None }
                else {
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
            } else { None };

            self.register_moe_layer_data(
                moe_idx, expert_ptrs, shared_ptrs, n_experts, topk,
                0, false, config.routed_scaling_factor, gate_wid, 0, 0, None,
            )?;
        }

        let buf_size = ((max_expert_bytes as f64) * 1.2) as usize;
        self.resize_expert_buffers(buf_size.max(1024))?;

        // Check how many shared experts got pinned
        let pinned_count = self.graph.as_ref().unwrap().shared_expert_vram.iter()
            .filter(|e| e.is_some()).count();
        let pinned_bytes: usize = self.graph.as_ref().unwrap().shared_expert_vram.iter()
            .filter_map(|e| e.as_ref())
            .map(|e| e.w13_packed_size + e.w13_scales_size + e.w2_packed_size + e.w2_scales_size)
            .sum();
        results.push(format!(
            "Shared experts pinned in VRAM: {}, total {:.1} MB",
            pinned_count, pinned_bytes as f64 / (1024.0 * 1024.0),
        ));

        // ── Run WITH shared expert VRAM residency ──
        results.push("\n--- With shared expert VRAM residency ---".to_string());
        let mut resident_times = Vec::new();
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
            resident_times.push(tok_total);
            results.push(format!(
                "  Token {}: {:.1}ms MoE, {:.1} tok/s",
                tok, tok_total, 1000.0 / tok_total,
            ));
        }
        let avg_resident = resident_times.iter().sum::<f64>() / resident_times.len() as f64;
        results.push(format!("  Resident avg: {:.1}ms, {:.1} tok/s", avg_resident, 1000.0 / avg_resident));

        // ── Run WITHOUT shared expert VRAM residency (DMA fallback) ──
        results.push("\n--- Without shared expert VRAM residency (DMA) ---".to_string());
        {
            // Temporarily remove shared expert VRAM entries
            let saved_vram = std::mem::take(&mut self.graph.as_mut().unwrap().shared_expert_vram);

            let mut dma_times = Vec::new();
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
                dma_times.push(tok_total);
                results.push(format!(
                    "  Token {}: {:.1}ms MoE, {:.1} tok/s",
                    tok, tok_total, 1000.0 / tok_total,
                ));
            }
            let avg_dma = dma_times.iter().sum::<f64>() / dma_times.len() as f64;
            results.push(format!("  DMA avg: {:.1}ms, {:.1} tok/s", avg_dma, 1000.0 / avg_dma));

            // Restore
            self.graph.as_mut().unwrap().shared_expert_vram = saved_vram;

            // Summary
            results.push("\n--- Pass 1 Summary: Shared Expert Residency ---".to_string());
            let delta = avg_dma - avg_resident;
            let pct = delta / avg_dma * 100.0;
            results.push(format!(
                "VRAM resident: {:.1}ms ({:.1} tok/s)",
                avg_resident, 1000.0 / avg_resident,
            ));
            results.push(format!(
                "DMA fallback:  {:.1}ms ({:.1} tok/s)",
                avg_dma, 1000.0 / avg_dma,
            ));
            results.push(format!(
                "Delta: {:.1}ms saved ({:.1}% improvement)",
                delta, pct,
            ));
            results.push(format!(
                "VRAM cost: {:.1} MB for {} shared experts",
                pinned_bytes as f64 / (1024.0 * 1024.0), pinned_count,
            ));
        }

        let total_elapsed = t_start.elapsed().as_secs_f64();
        results.push(format!("Total bench time: {:.1}s", total_elapsed));

        std::mem::forget(store);
        Ok(results.join("\n"))
    }

    /// Benchmark PCIe DMA bandwidth and pure HCS compute speed.
    fn bench_pcie_and_compute_internal(
        &mut self,
        model_dir: &str,
        num_tokens: usize,
    ) -> PyResult<String> {
        use std::path::Path;
        use std::time::Instant;

        let mut results = Vec::new();
        let t_start = Instant::now();

        // ═══════════════════════════════════════════════════════════════════
        // PART 1: Raw PCIe DMA bandwidth test (no model needed)
        // ═══════════════════════════════════════════════════════════════════
        results.push("=== PART 1: Raw PCIe H2D DMA Bandwidth ===".to_string());

        // Test various transfer sizes from 1 KB to 64 MB
        let test_sizes: Vec<(usize, &str)> = vec![
            (1024, "1 KB"),
            (4 * 1024, "4 KB"),
            (16 * 1024, "16 KB"),
            (64 * 1024, "64 KB"),
            (256 * 1024, "256 KB"),
            (512 * 1024, "512 KB"),
            (1024 * 1024, "1 MB"),
            (2 * 1024 * 1024, "2 MB"),
            (4 * 1024 * 1024, "4 MB"),
            (8 * 1024 * 1024, "8 MB"),
            (16 * 1024 * 1024, "16 MB"),
            (32 * 1024 * 1024, "32 MB"),
            (64 * 1024 * 1024, "64 MB"),
        ];

        for &(size, label) in &test_sizes {
            // Allocate pinned host memory + device memory
            let mut h_buf: Vec<u8> = vec![0xABu8; size];
            let d_buf = self.device.alloc_zeros::<u8>(size)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            // Pin host memory for async DMA
            unsafe {
                cuda_sys::lib().cuMemHostRegister_v2(
                    h_buf.as_mut_ptr() as *mut std::ffi::c_void,
                    size,
                    0, // CU_MEMHOSTREGISTER_DEFAULT
                );
            }

            let copy_stream = self.copy_stream.0;

            // Warmup: 3 transfers
            for _ in 0..3 {
                unsafe {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        *d_buf.device_ptr(),
                        h_buf.as_ptr() as *const std::ffi::c_void,
                        size,
                        copy_stream,
                    );
                }
            }
            unsafe {
                cuda_sys::lib().cuStreamSynchronize(copy_stream);
            }

            // Timed: N iterations (more for small sizes to get stable timing)
            let iters = if size < 64 * 1024 { 200 } else if size < 1024 * 1024 { 100 } else { 50 };

            let t0 = Instant::now();
            for _ in 0..iters {
                unsafe {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        *d_buf.device_ptr(),
                        h_buf.as_ptr() as *const std::ffi::c_void,
                        size,
                        copy_stream,
                    );
                }
            }
            unsafe {
                cuda_sys::lib().cuStreamSynchronize(copy_stream);
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let total_bytes = size as f64 * iters as f64;
            let bw_gbs = total_bytes / elapsed / 1e9;
            let per_xfer_us = elapsed * 1e6 / iters as f64;

            results.push(format!(
                "  {:>6}: {:.2} GB/s  ({:.1} us/xfer, {} iters)",
                label, bw_gbs, per_xfer_us, iters,
            ));

            // Unpin
            unsafe {
                cuda_sys::lib().cuMemHostUnregister(h_buf.as_mut_ptr() as *mut std::ffi::c_void);
            }
        }

        // Also test unpinned (pageable) DMA for comparison at a few sizes
        results.push("".to_string());
        results.push("  -- Unpinned (pageable) for comparison --".to_string());
        for &(size, label) in &[(1024 * 1024, "1 MB"), (16 * 1024 * 1024, "16 MB")] {
            let h_buf: Vec<u8> = vec![0xABu8; size];
            let d_buf = self.device.alloc_zeros::<u8>(size)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            // Warmup
            for _ in 0..3 {
                unsafe {
                    cuda_sys::lib().cuMemcpyHtoD_v2(
                        *d_buf.device_ptr(),
                        h_buf.as_ptr() as *const std::ffi::c_void,
                        size,
                    );
                }
            }

            let iters = 30;
            let t0 = Instant::now();
            for _ in 0..iters {
                unsafe {
                    cuda_sys::lib().cuMemcpyHtoD_v2(
                        *d_buf.device_ptr(),
                        h_buf.as_ptr() as *const std::ffi::c_void,
                        size,
                    );
                }
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let bw_gbs = (size as f64 * iters as f64) / elapsed / 1e9;
            let per_xfer_us = elapsed * 1e6 / iters as f64;
            results.push(format!(
                "  {:>6}: {:.2} GB/s  ({:.1} us/xfer, {} iters) [pageable]",
                label, bw_gbs, per_xfer_us, iters,
            ));
        }

        // ═══════════════════════════════════════════════════════════════════
        // PART 2: Load model and measure expert sizes
        // ═══════════════════════════════════════════════════════════════════
        results.push("".to_string());
        results.push("=== PART 2: Model Expert Sizes ===".to_string());

        let t0 = Instant::now();
        let store = crate::weights::WeightStore::load_from_hf(
            Path::new(model_dir), 128, None, None, 4, 4, false,
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

        // Measure actual expert sizes
        if !store.experts_gpu.is_empty() && !store.experts_gpu[0].is_empty() {
            let e0 = &store.experts_gpu[0][0];
            let w13p = e0.w13_packed.len() * 4;
            let w13s = e0.w13_scales.len() * 2;
            let w2p = e0.w2_packed.len() * 4;
            let w2s = e0.w2_scales.len() * 2;
            let total = w13p + w13s + w2p + w2s;
            results.push(format!(
                "Expert size: w13_packed={} B, w13_scales={} B, w2_packed={} B, w2_scales={} B, total={} B ({:.1} KB)",
                w13p, w13s, w2p, w2s, total, total as f64 / 1024.0,
            ));
            results.push(format!(
                "Per-layer DMA (topk={}): {} experts x {} B = {} B ({:.1} KB, {:.2} MB)",
                topk, topk, total, topk * total, (topk * total) as f64 / 1024.0,
                (topk * total) as f64 / (1024.0 * 1024.0),
            ));
            results.push(format!(
                "Per-token DMA (all layers): {} layers x {:.2} MB = {:.1} MB",
                num_moe_layers,
                (topk * total) as f64 / (1024.0 * 1024.0),
                (num_moe_layers * topk * total) as f64 / (1024.0 * 1024.0),
            ));
        }

        // ═══════════════════════════════════════════════════════════════════
        // PART 3: Configure GPU and register layers
        // ═══════════════════════════════════════════════════════════════════
        self.configure(
            hidden_size, config.num_hidden_layers, 1, 1e-6,
            topk, intermediate_size, hidden_size * 3, group_size,
        )?;

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
                    rows: n_experts, cols: hidden_size, dtype: 1,
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
                if se.w13_packed.is_empty() { None }
                else {
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

            self.register_moe_layer(
                moe_idx, expert_ptrs, shared_ptrs, n_experts, topk,
                0, false, config.routed_scaling_factor, gate_wid, 0, 0, None,
            )?;
        }

        let buf_size = ((max_expert_bytes as f64) * 1.2) as usize;
        self.resize_expert_buffers(buf_size.max(1024))?;

        // ═══════════════════════════════════════════════════════════════════
        // PART 4: Pure DMA test with REAL expert weights
        // ═══════════════════════════════════════════════════════════════════
        results.push("".to_string());
        results.push("=== PART 3: DMA with Real Expert Weights ===".to_string());
        {
            let graph = self.graph.as_ref().unwrap();
            let moe = graph.moe_layers[0].as_ref().unwrap();
            let expert = &moe.experts[0];
            let total_bytes = expert.w13_packed_bytes + expert.w13_scales_bytes
                + expert.w2_packed_bytes + expert.w2_scales_bytes;

            // Single expert DMA (4 separate calls, as current code does)
            let copy_stream = self.copy_stream.0;
            let buf_base = *graph.d_expert_buf[0].device_ptr();
            let w13p_off = graph.expert_buf_w13p_offset;
            let w13s_off = graph.expert_buf_w13s_offset;
            let w2p_off = graph.expert_buf_w2p_offset;
            let w2s_off = graph.expert_buf_w2s_offset;

            // Warmup
            for _ in 0..5 {
                unsafe {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_base + w13p_off as u64, expert.w13_packed_ptr as *const std::ffi::c_void,
                        expert.w13_packed_bytes, copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_base + w13s_off as u64, expert.w13_scales_ptr as *const std::ffi::c_void,
                        expert.w13_scales_bytes, copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_base + w2p_off as u64, expert.w2_packed_ptr as *const std::ffi::c_void,
                        expert.w2_packed_bytes, copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_base + w2s_off as u64, expert.w2_scales_ptr as *const std::ffi::c_void,
                        expert.w2_scales_bytes, copy_stream);
                }
            }
            unsafe { cuda_sys::lib().cuStreamSynchronize(copy_stream); }

            // Time single expert DMA (4 calls)
            let iters = 200;
            let t0 = Instant::now();
            for _ in 0..iters {
                unsafe {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_base + w13p_off as u64, expert.w13_packed_ptr as *const std::ffi::c_void,
                        expert.w13_packed_bytes, copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_base + w13s_off as u64, expert.w13_scales_ptr as *const std::ffi::c_void,
                        expert.w13_scales_bytes, copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_base + w2p_off as u64, expert.w2_packed_ptr as *const std::ffi::c_void,
                        expert.w2_packed_bytes, copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_base + w2s_off as u64, expert.w2_scales_ptr as *const std::ffi::c_void,
                        expert.w2_scales_bytes, copy_stream);
                }
            }
            unsafe { cuda_sys::lib().cuStreamSynchronize(copy_stream); }
            let elapsed = t0.elapsed().as_secs_f64();
            let per_xfer_us = elapsed * 1e6 / iters as f64;
            let bw = total_bytes as f64 * iters as f64 / elapsed / 1e9;

            results.push(format!(
                "  Single expert (4 calls, {} B): {:.1} us/expert, {:.2} GB/s effective",
                total_bytes, per_xfer_us, bw,
            ));

            // Time 10-expert sequence (simulates one layer's DMA)
            let t0 = Instant::now();
            let layer_iters = 100;
            for _ in 0..layer_iters {
                for eid in 0..topk.min(n_experts) {
                    let exp = &moe.experts[eid];
                    unsafe {
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            buf_base + w13p_off as u64, exp.w13_packed_ptr as *const std::ffi::c_void,
                            exp.w13_packed_bytes, copy_stream);
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            buf_base + w13s_off as u64, exp.w13_scales_ptr as *const std::ffi::c_void,
                            exp.w13_scales_bytes, copy_stream);
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            buf_base + w2p_off as u64, exp.w2_packed_ptr as *const std::ffi::c_void,
                            exp.w2_packed_bytes, copy_stream);
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            buf_base + w2s_off as u64, exp.w2_scales_ptr as *const std::ffi::c_void,
                            exp.w2_scales_bytes, copy_stream);
                    }
                }
            }
            unsafe { cuda_sys::lib().cuStreamSynchronize(copy_stream); }
            let elapsed = t0.elapsed().as_secs_f64();
            let per_layer_us = elapsed * 1e6 / layer_iters as f64;
            let layer_bytes = total_bytes * topk.min(n_experts);
            let bw = layer_bytes as f64 * layer_iters as f64 / elapsed / 1e9;

            results.push(format!(
                "  Full layer ({} experts, {} B): {:.1} us/layer, {:.2} GB/s effective",
                topk.min(n_experts), layer_bytes, per_layer_us, bw,
            ));
            results.push(format!(
                "  Projected per-token DMA ({} layers): {:.1} ms",
                num_moe_layers, per_layer_us * num_moe_layers as f64 / 1000.0,
            ));
        }

        // ═══════════════════════════════════════════════════════════════════
        // PART 5: Pure HCS compute (VRAM-resident, zero DMA)
        // ═══════════════════════════════════════════════════════════════════
        results.push("".to_string());
        results.push("=== PART 4: Pure HCS Compute (zero DMA) ===".to_string());

        // Pin all experts for full HCS
        let msg = self.init_hcs_internal(0, 500)?;
        results.push(format!("  {}", msg));
        let msg = self.hcs_pin_all_internal()?;
        results.push(format!("  {}", msg));

        let graph = self.graph.as_ref().unwrap();
        let hcs = graph.hcs.as_ref().unwrap();
        results.push(format!(
            "  HCS cache: {} experts, {:.1} MB",
            hcs.num_cached, hcs.vram_bytes as f64 / (1024.0 * 1024.0),
        ));

        // Test A: Single expert GEMV (w13 + fused silu+w2+accum) — pure compute, no routing
        results.push("".to_string());
        results.push("  -- Single Expert Compute (w13 GEMV + fused silu+w2+accum) --".to_string());
        {
            let graph = self.graph.as_ref().unwrap();
            let inv_wp = *graph.d_inv_weight_perm.device_ptr();
            let inv_sp = *graph.d_inv_scale_perm.device_ptr();
            let hs = graph.hidden_size;
            let intermediate = graph.intermediate_size;
            let gs = graph.group_size;
            let k = graph.kernels.as_ref().unwrap();

            // Pick first HCS expert
            let hcs = graph.hcs.as_ref().unwrap();
            let first_key = hcs.cache.keys().next().unwrap();
            let entry = hcs.cache.get(first_key).unwrap();
            let (w13p, w13s, w2p, w2s) = (
                entry.w13_packed_ptr(), entry.w13_scales_ptr(),
                entry.w2_packed_ptr(), entry.w2_scales_ptr(),
            );

            // Warmup
            for _ in 0..10 {
                self.launch_marlin_gemv_raw(
                    w13p, w13s,
                    *graph.d_hidden.device_ptr(),
                    *graph.d_expert_gate_up.device_ptr(),
                    inv_wp, inv_sp, hs, 2 * intermediate, gs,
                )?;
                self.launch_fused_silu_accum(
                    w2p, w2s,
                    *graph.d_expert_gate_up.device_ptr(),
                    *graph.d_moe_out.device_ptr(),
                    inv_wp, inv_sp,
                    intermediate, hs, gs, 0.1f32, 0u64, k,
                )?;
            }
            self.device.synchronize()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            // Benchmark single expert compute
            let iters = 500;
            let t0 = Instant::now();
            for _ in 0..iters {
                self.launch_marlin_gemv_raw(
                    w13p, w13s,
                    *graph.d_hidden.device_ptr(),
                    *graph.d_expert_gate_up.device_ptr(),
                    inv_wp, inv_sp, hs, 2 * intermediate, gs,
                )?;
                self.launch_fused_silu_accum(
                    w2p, w2s,
                    *graph.d_expert_gate_up.device_ptr(),
                    *graph.d_moe_out.device_ptr(),
                    inv_wp, inv_sp,
                    intermediate, hs, gs, 0.1f32, 0u64, k,
                )?;
            }
            self.device.synchronize()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let elapsed = t0.elapsed().as_secs_f64();
            let per_expert_us = elapsed * 1e6 / iters as f64;

            results.push(format!(
                "  Per expert compute: {:.1} us ({} iters)",
                per_expert_us, iters,
            ));

            // Benchmark w13 GEMV alone
            let t0 = Instant::now();
            for _ in 0..iters {
                self.launch_marlin_gemv_raw(
                    w13p, w13s,
                    *graph.d_hidden.device_ptr(),
                    *graph.d_expert_gate_up.device_ptr(),
                    inv_wp, inv_sp, hs, 2 * intermediate, gs,
                )?;
            }
            self.device.synchronize()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let w13_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            // Benchmark fused silu+w2+accum alone
            let t0 = Instant::now();
            for _ in 0..iters {
                self.launch_fused_silu_accum(
                    w2p, w2s,
                    *graph.d_expert_gate_up.device_ptr(),
                    *graph.d_moe_out.device_ptr(),
                    inv_wp, inv_sp,
                    intermediate, hs, gs, 0.1f32, 0u64, k,
                )?;
            }
            self.device.synchronize()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let fused_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            results.push(format!(
                "    w13 GEMV [{},{}]: {:.1} us",
                hs, 2 * intermediate, w13_us,
            ));
            results.push(format!(
                "    fused silu+w2+accum [{},{}]: {:.1} us",
                intermediate, hs, fused_us,
            ));

            // ── v2 K-split benchmark ──
            let w13_ksplits = self.calc_k_splits(hs, 2 * intermediate);
            let w2_ksplits = self.calc_k_splits(intermediate, hs);
            results.push(format!(
                "  -- v2 K-split: w13 k_splits={}, w2 k_splits={}, {} SMs --",
                w13_ksplits, w2_ksplits, graph.num_sms,
            ));

            // v2 w13 GEMV + reduce
            let partial_ptr = *graph.d_v2_partial.device_ptr();
            // Warmup
            for _ in 0..10 {
                self.launch_marlin_gemv_v2(
                    w13p, w13s,
                    *graph.d_hidden.device_ptr(),
                    partial_ptr, inv_wp, inv_sp,
                    hs, 2 * intermediate, gs, w13_ksplits, k,
                )?;
                self.launch_reduce_ksplits_bf16(
                    *graph.d_expert_gate_up.device_ptr(),
                    partial_ptr,
                    2 * intermediate, w13_ksplits, k,
                )?;
            }
            self.device.synchronize()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

            let t0 = Instant::now();
            for _ in 0..iters {
                self.launch_marlin_gemv_v2(
                    w13p, w13s,
                    *graph.d_hidden.device_ptr(),
                    partial_ptr, inv_wp, inv_sp,
                    hs, 2 * intermediate, gs, w13_ksplits, k,
                )?;
                self.launch_reduce_ksplits_bf16(
                    *graph.d_expert_gate_up.device_ptr(),
                    partial_ptr,
                    2 * intermediate, w13_ksplits, k,
                )?;
            }
            self.device.synchronize()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let w13_v2_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            // v2 fused silu+w2+accum + reduce (only if k_splits > 1)
            let fused_v2_us = if w2_ksplits > 1 {
                for _ in 0..10 {
                    self.launch_fused_silu_accum_v2(
                        w2p, w2s,
                        *graph.d_expert_gate_up.device_ptr(),
                        partial_ptr, inv_wp, inv_sp,
                        intermediate, hs, gs, w2_ksplits, k,
                    )?;
                    self.launch_reduce_ksplits_weighted_accum(
                        *graph.d_moe_out.device_ptr(),
                        partial_ptr,
                        hs, w2_ksplits, 0.1f32, k,
                    )?;
                }
                self.device.synchronize()
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

                let t0 = Instant::now();
                for _ in 0..iters {
                    self.launch_fused_silu_accum_v2(
                        w2p, w2s,
                        *graph.d_expert_gate_up.device_ptr(),
                        partial_ptr, inv_wp, inv_sp,
                        intermediate, hs, gs, w2_ksplits, k,
                    )?;
                    self.launch_reduce_ksplits_weighted_accum(
                        *graph.d_moe_out.device_ptr(),
                        partial_ptr,
                        hs, w2_ksplits, 0.1f32, k,
                    )?;
                }
                self.device.synchronize()
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
                t0.elapsed().as_secs_f64() * 1e6 / iters as f64
            } else {
                fused_us // v1 is better for k_splits=1
            };

            // Best combo: v2 w13 + v1 fused (when fused v2 is slower)
            let best_fused = if fused_v2_us < fused_us { fused_v2_us } else { fused_us };
            let best_fused_label = if fused_v2_us < fused_us { "v2" } else { "v1" };
            let per_expert_best = w13_v2_us + best_fused;

            results.push(format!(
                "    v2 w13 [{},{}]: {:.1} us (v1: {:.1} us, {:.1}x)",
                hs, 2 * intermediate, w13_v2_us, w13_us, w13_us / w13_v2_us,
            ));
            results.push(format!(
                "    v2 fused [{},{}]: {:.1} us (v1: {:.1} us, {:.1}x)",
                intermediate, hs, fused_v2_us, fused_us, fused_us / fused_v2_us,
            ));
            results.push(format!(
                "    BEST combo: v2 w13 + {} fused = {:.1} us/expert (v1: {:.1} us, {:.1}x)",
                best_fused_label, per_expert_best, per_expert_us, per_expert_us / per_expert_best,
            ));
            results.push(format!(
                "    BEST per token ({} layers x {} experts): {:.1} ms = {:.1} tok/s",
                num_moe_layers, topk,
                per_expert_best * topk as f64 * num_moe_layers as f64 / 1000.0,
                1000.0 / (per_expert_best * topk as f64 * num_moe_layers as f64 / 1000.0),
            ));
        }

        // Test B: Full layer compute — 10 experts sequential, all HCS (zero DMA)
        // Uses v2 w13 + v1 fused (best combo from above)
        results.push("".to_string());
        results.push(format!("  -- Full Layer Compute ({} experts, all HCS, v2 w13) --", topk));
        {
            let graph = self.graph.as_ref().unwrap();
            let inv_wp = *graph.d_inv_weight_perm.device_ptr();
            let inv_sp = *graph.d_inv_scale_perm.device_ptr();
            let hs = graph.hidden_size;
            let intermediate = graph.intermediate_size;
            let gs = graph.group_size;
            let k = graph.kernels.as_ref().unwrap();
            let hcs = graph.hcs.as_ref().unwrap();
            let partial_ptr = *graph.d_v2_partial.device_ptr();

            // Calculate w13 K-splits for v2
            let w13_n = 2 * intermediate;
            let w13_k_tiles = hs / 16;
            let w13_max_ksplits = w13_k_tiles / 16;
            let w13_ksplits = if w13_max_ksplits > 1 {
                let n_tiles = (w13_n + 15) / 16;
                let target = graph.num_sms * 4;
                let desired = (target + n_tiles - 1) / n_tiles;
                desired.clamp(1, w13_max_ksplits.min(8))
            } else {
                1
            };

            // Collect first topk HCS entries from layer 0
            let mut entries: Vec<(u64, u64, u64, u64)> = Vec::new();
            for eid in 0..topk.min(n_experts) {
                if let Some(entry) = hcs.get(0, eid) {
                    entries.push((
                        entry.w13_packed_ptr(), entry.w13_scales_ptr(),
                        entry.w2_packed_ptr(), entry.w2_scales_ptr(),
                    ));
                }
            }
            let num_cached = entries.len();

            if num_cached > 0 {
                // Warmup
                for _ in 0..5 {
                    for (w13p, w13s, w2p, w2s) in &entries {
                        self.launch_marlin_gemv_v2(
                            *w13p, *w13s,
                            *graph.d_hidden.device_ptr(),
                            partial_ptr, inv_wp, inv_sp,
                            hs, 2 * intermediate, gs, w13_ksplits, k,
                        )?;
                        self.launch_reduce_ksplits_bf16(
                            *graph.d_expert_gate_up.device_ptr(),
                            partial_ptr,
                            2 * intermediate, w13_ksplits, k,
                        )?;
                        self.launch_fused_silu_accum(
                            *w2p, *w2s,
                            *graph.d_expert_gate_up.device_ptr(),
                            *graph.d_moe_out.device_ptr(),
                            inv_wp, inv_sp,
                            intermediate, hs, gs, 0.1f32, 0u64, k,
                        )?;
                    }
                }
                self.device.synchronize()
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

                let iters = 200;
                let t0 = Instant::now();
                for _ in 0..iters {
                    for (w13p, w13s, w2p, w2s) in &entries {
                        self.launch_marlin_gemv_v2(
                            *w13p, *w13s,
                            *graph.d_hidden.device_ptr(),
                            partial_ptr, inv_wp, inv_sp,
                            hs, 2 * intermediate, gs, w13_ksplits, k,
                        )?;
                        self.launch_reduce_ksplits_bf16(
                            *graph.d_expert_gate_up.device_ptr(),
                            partial_ptr,
                            2 * intermediate, w13_ksplits, k,
                        )?;
                        self.launch_fused_silu_accum(
                            *w2p, *w2s,
                            *graph.d_expert_gate_up.device_ptr(),
                            *graph.d_moe_out.device_ptr(),
                            inv_wp, inv_sp,
                            intermediate, hs, gs, 0.1f32, 0u64, k,
                        )?;
                    }
                }
                self.device.synchronize()
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
                let elapsed = t0.elapsed().as_secs_f64();
                let per_layer_us = elapsed * 1e6 / iters as f64;
                let per_layer_ms = per_layer_us / 1000.0;

                results.push(format!(
                    "  Per layer ({} experts): {:.1} us ({:.3} ms)",
                    num_cached, per_layer_us, per_layer_ms,
                ));
                results.push(format!(
                    "  Per token ({} layers): {:.1} ms = {:.1} tok/s (MoE compute only)",
                    num_moe_layers,
                    per_layer_ms * num_moe_layers as f64,
                    1000.0 / (per_layer_ms * num_moe_layers as f64),
                ));
            } else {
                results.push("  No HCS entries for layer 0!".to_string());
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // PART 6: Full MoE forward comparison (baseline vs HCS)
        // ═══════════════════════════════════════════════════════════════════
        results.push("".to_string());
        results.push("=== PART 5: Full MoE Forward (routing + compute + DMA) ===".to_string());

        // Baseline (no HCS)
        {
            let hcs_state = self.graph.as_mut().unwrap().hcs.take();

            let mut times = Vec::new();
            for tok in 0..num_tokens.min(5) {
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
                times.push(tok_total);
            }

            // Use last 3 for avg (skip warmup)
            let skip = times.len().saturating_sub(3);
            let avg: f64 = times[skip..].iter().sum::<f64>() / times[skip..].len() as f64;
            results.push(format!(
                "  Baseline (no HCS): {:.1} ms avg = {:.1} tok/s",
                avg, 1000.0 / avg,
            ));

            self.graph.as_mut().unwrap().hcs = hcs_state;
        }

        // With HCS
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
        }

        let skip = hcs_times.len().saturating_sub(5);
        let avg_hcs: f64 = hcs_times[skip..].iter().sum::<f64>() / hcs_times[skip..].len() as f64;
        let hcs = self.graph.as_ref().unwrap().hcs.as_ref().unwrap();
        results.push(format!(
            "  With HCS ({} cached, {:.0}% hit): {:.1} ms avg = {:.1} tok/s",
            hcs.num_cached,
            hcs.hit_rate() * 100.0,
            avg_hcs, 1000.0 / avg_hcs,
        ));

        // ═══════════════════════════════════════════════════════════════════
        // PART 7: Summary projections
        // ═══════════════════════════════════════════════════════════════════
        results.push("".to_string());
        results.push("=== PART 6: Summary & Projections ===".to_string());

        if !store.experts_gpu.is_empty() && !store.experts_gpu[0].is_empty() {
            let e0 = &store.experts_gpu[0][0];
            let expert_bytes = e0.w13_packed.len() * 4 + e0.w13_scales.len() * 2
                + e0.w2_packed.len() * 4 + e0.w2_scales.len() * 2;

            // Use measured DMA and compute from parts 3 and 4
            results.push(format!("  Expert size: {} B ({:.1} KB)", expert_bytes, expert_bytes as f64 / 1024.0));
            results.push(format!("  topk={}, {} MoE layers, {} total experts/layer",
                topk, num_moe_layers, n_experts));
            results.push(format!("  HCS cached: {}/{} ({:.0}%)",
                hcs.num_cached,
                num_moe_layers * n_experts,
                hcs.num_cached as f64 / (num_moe_layers * n_experts) as f64 * 100.0,
            ));
        }

        let total_elapsed = t_start.elapsed().as_secs_f64();
        results.push(format!("\nTotal bench time: {:.1}s", total_elapsed));

        std::mem::forget(store);
        Ok(results.join("\n"))
    }
}

impl Drop for GpuDecodeStore {
    fn drop(&mut self) {
        // Destroy pre-allocated events
        if let Some(ref graph) = self.graph {
            if let Some(ref events) = graph.pre_events {
                unsafe {
                    for e in events.iter() {
                        if !e.0.is_null() {
                            let _ = cuda_sys::lib().cuEventDestroy_v2(e.0);
                        }
                    }
                }
            }
        }
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
