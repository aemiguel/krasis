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
use std::io::Write;
use std::sync::Arc;

use cudarc::cublas::{CudaBlas, sys as cublas_sys};
use cudarc::cublas::result as cublas_result;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::driver::sys as cuda_sys;

fn stderr_debug_enabled() -> bool {
    std::env::var("KRASIS_DEBUG_STDERR")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn prefill_debug_enabled() -> bool {
    std::env::var("KRASIS_PREFILL_DEBUG")
        .map(|v| v == "1")
        .unwrap_or(false)
}

// PTX compiled from src/cuda/decode_kernels.cu at build time.
#[cfg(has_decode_kernels)]
const DECODE_KERNELS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/decode_kernels.ptx"));

// CUDA graph types from cudarc's sys bindings (dynamically loaded via cuda_sys::lib())
type CUgraph = cuda_sys::CUgraph;
type CUgraphExec = cuda_sys::CUgraphExec;

/// Extract raw CUfunction handle from cudarc's CudaFunction.
/// CudaFunction has two pointer-sized fields (CUfunction + Arc<CudaDevice>);
/// since #[repr(Rust)] doesn't guarantee field order, we try both offsets
/// and validate via cuFuncGetAttribute.
fn extract_cu_function(func: &cudarc::driver::CudaFunction) -> cuda_sys::CUfunction {
    unsafe {
        let struct_ptr = func as *const _ as *const u8;
        let word0: cuda_sys::CUfunction = std::ptr::read(struct_ptr as *const _);
        let mut dummy = 0i32;
        let w0_valid = cuda_sys::lib().cuFuncGetAttribute(
            &mut dummy,
            cuda_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS,
            word0,
        ) == cuda_sys::CUresult::CUDA_SUCCESS;
        if w0_valid { word0 } else { std::ptr::read(struct_ptr.add(8) as *const _) }
    }
}

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
    "reduce_ksplits_f32",
    "marlin_gemv_int4_fused_silu_accum_v2",
    "marlin_gemv_int8_fused_silu_accum",
    "marlin_gemv_int8_fused_silu_accum_v2",
    "reduce_ksplits_weighted_accum_bf16",
    "sigmoid_gate_inplace_bf16",
    "simple_int4_gemv_f32",
    "simple_int4_gemv_bf16",
    "marlin_gemv_int4_v2_batched",
    "marlin_gemv_int8_v2_batched",
    "reduce_ksplits_bf16_batched",
    "fused_silu_w2_batched",
    "fused_silu_w2_int8_batched",
    "multi_expert_weighted_add_bf16",
    "marlin_gemv_int4_f32",
    "marlin_gemv_int8",
    "marlin_gemv_int8_f32",
    "marlin_gemv_int4_v2_fused_f32",
    "marlin_gemv_int8_v2",
    "marlin_gemv_int8_v2_fused_f32",
    // Graphable kernel variants (read position/token from GPU pointers for CUDA graph capture)
    "embedding_lookup_g",
    "apply_rope_g",
    "kv_cache_write_g",
    "gqa_attention_g",
    "gated_rmsnorm_silu_bf16",
    "gqa_attention_g_bf16",
    "apply_gated_attn_bf16",
    "gqa_attention_tiled_g",
    "gqa_attention_reduce_g",
    "la_fused_post_proj",
    "expert_classify_prepare",
    // MLA (Multi-head Latent Attention) kernels
    "mla_kv_cache_write_g",
    "mla_kv_cache_write",
    "mla_attention_g",
    "mla_attention",
    "mla_deinterleave",
    "mla_split_q",
    "mla_absorb_wkc",
    "mla_apply_wvc",
    // Mamba2 SSM kernels (Nemotron-H hybrid models)
    "mamba2_conv1d",
    "mamba2_ssm_step",
    "mamba2_discretize",
    "mamba2_gate_output",
    // relu2 expert activation (Nemotron LatentMoE)
    "relu2_w2_batched",
    "relu2_w2_int8_batched",
    // 4-bit PolarQuant kernels
    "kv_cache_write_polar4",
    "gqa_attention_polar4",
    "kv_cache_write_polar4_g",
    "gqa_attention_polar4_g",
    "gqa_attention_polar4_tiled_g",
    "gqa_attention_polar4_reduce_g",
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

/// Four-point VRAM calibration data from startup measurements.
/// Used to interpolate expected free VRAM at any prompt length.
#[derive(Clone, Copy)]
struct VramCalibration {
    short_tokens: usize,
    long_tokens: usize,
    /// Post-calibration no-HCS idle free VRAM (MB).
    baseline_free_mb: u64,
    /// min_free VRAM (MB) during short prompt prefill (no HCS loaded)
    prefill_short_free_mb: u64,
    /// min_free VRAM (MB) during long prompt prefill (no HCS loaded)
    prefill_long_free_mb: u64,
    /// min_free VRAM (MB) during short prompt decode (no HCS loaded)
    decode_short_free_mb: u64,
    /// min_free VRAM (MB) during long prompt decode (no HCS loaded)
    decode_long_free_mb: u64,
    safety_margin_mb: u64,
}

impl VramCalibration {
    /// Interpolate expected min_free VRAM (MB) during prefill for a given prompt length.
    fn prefill_free_mb(&self, tokens: usize) -> u64 {
        if self.long_tokens <= self.short_tokens {
            return self.prefill_long_free_mb;
        }
        let t = (tokens.saturating_sub(self.short_tokens) as f64)
            / (self.long_tokens - self.short_tokens) as f64;
        let t = t.clamp(0.0, 1.5); // allow slight extrapolation
        let free = self.prefill_short_free_mb as f64
            - t * (self.prefill_short_free_mb as f64 - self.prefill_long_free_mb as f64);
        (free.max(0.0)) as u64
    }

    /// Interpolate expected min_free VRAM (MB) during decode for a given prompt length.
    fn decode_free_mb(&self, tokens: usize) -> u64 {
        if self.long_tokens <= self.short_tokens {
            return self.decode_long_free_mb;
        }
        let t = (tokens.saturating_sub(self.short_tokens) as f64)
            / (self.long_tokens - self.short_tokens) as f64;
        let t = t.clamp(0.0, 1.5);
        let free = self.decode_short_free_mb as f64
            - t * (self.decode_short_free_mb as f64 - self.decode_long_free_mb as f64);
        (free.max(0.0)) as u64
    }

    /// Max HCS budget for prefill of N tokens (what survives prefill).
    fn prefill_hcs_budget_mb(&self, tokens: usize) -> u64 {
        self.prefill_free_mb(tokens).saturating_sub(self.safety_margin_mb)
    }

    /// Minimum idle free VRAM required while HCS is resident so the measured
    /// prefill transient still leaves the configured safety margin.
    fn required_prefill_idle_free_mb(&self, tokens: usize) -> u64 {
        self.baseline_free_mb
            .saturating_sub(self.prefill_hcs_budget_mb(tokens))
    }

    /// Max HCS budget for decode after prefill of N tokens.
    fn decode_hcs_budget_mb(&self, tokens: usize) -> u64 {
        self.decode_free_mb(tokens).saturating_sub(self.safety_margin_mb)
    }

    /// Minimum idle free VRAM required while HCS is resident so the measured
    /// decode transient still leaves the configured safety margin.
    fn required_idle_free_mb(&self, tokens: usize) -> u64 {
        self.baseline_free_mb
            .saturating_sub(self.decode_hcs_budget_mb(tokens))
    }

    /// Decode VRAM consumption per token in KB (linear rate).
    fn decode_kb_per_tok(&self) -> f64 {
        if self.long_tokens > self.short_tokens {
            ((self.decode_short_free_mb as f64 - self.decode_long_free_mb as f64)
                / (self.long_tokens - self.short_tokens) as f64) * 1024.0
        } else {
            0.0
        }
    }

    /// Largest prefill chunk length whose measured idle-floor requirement fits
    /// within the currently available idle free VRAM.
    fn max_safe_prefill_tokens(&self, free_mb: u64, prompt_tokens: usize) -> usize {
        let max_tokens = prompt_tokens.clamp(128, 50_000);
        let fits = |tokens: usize| {
            self.required_prefill_idle_free_mb(tokens) <= free_mb
                && self.prefill_free_mb(tokens) >= self.safety_margin_mb
        };

        if !fits(128) {
            return 128;
        }
        if fits(max_tokens) {
            return max_tokens;
        }

        let mut lo = 128usize;
        let mut hi = max_tokens;
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if fits(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// Per-layer APFL statistics.
struct ApflLayerStats {
    hits: u64,
    misses: u64,
}

impl ApflLayerStats {
    fn new() -> Self {
        ApflLayerStats { hits: 0, misses: 0 }
    }

    fn record_hit(&mut self) {
        self.hits += 1;
    }

    fn record_miss(&mut self) {
        self.misses += 1;
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
    /// Per-layer stats.
    layer_stats: Vec<ApflLayerStats>,
    /// Global stats.
    total_hits: u64,
    total_misses: u64,
    /// Fixed number of experts to prefetch per layer.
    prefetch_count: usize,
    /// Whether APFL is enabled.
    enabled: bool,
    /// Host-side buffer for speculative routing results.
    h_spec_topk_ids: Vec<i32>,
    /// Deferred spec routing: pending results from previous layer's spec.
    /// When true, spec_stream has completed GEMV+topk and h_spec_topk_ids
    /// contains the predicted expert IDs for pending_next_layer.
    pending_prefetch: bool,
    pending_next_layer: usize,
    pending_spec_count: usize,
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

/// Pinned (page-locked) host memory chunk for truly async H2D DMA.
/// cuMemcpyHtoDAsync with pinned source returns immediately; the DMA engine
/// handles the transfer without CPU involvement. Without pinning, CUDA
/// internally stages through a small pinned buffer, blocking the CPU.
struct PinnedHostChunk {
    data: Vec<u8>,
    registered: bool,
}

impl PinnedHostChunk {
    fn new(size: usize) -> Self {
        let data = vec![0u8; size];
        let registered = unsafe {
            let err = cuda_sys::lib().cuMemHostRegister_v2(
                data.as_ptr() as *mut std::ffi::c_void,
                size,
                0, // CU_MEMHOSTREGISTER_DEFAULT
            );
            err == cuda_sys::CUresult::CUDA_SUCCESS
        };
        if !registered {
            log::warn!("PinnedHostChunk: cuMemHostRegister failed for {} MB, falling back to unpinned",
                size / (1024 * 1024));
        }
        Self { data, registered }
    }

    fn as_ptr(&self) -> *const u8 { self.data.as_ptr() }
    fn as_mut_ptr(&mut self) -> *mut u8 { self.data.as_mut_ptr() }
}

impl Drop for PinnedHostChunk {
    fn drop(&mut self) {
        if self.registered {
            unsafe {
                cuda_sys::lib().cuMemHostUnregister(self.data.as_ptr() as *mut std::ffi::c_void);
            }
        }
    }
}

/// HCS state: resident expert cache + activation heatmap + dynamic eviction.
struct HcsState {
    /// (layer_idx, expert_idx) → cache entry (for management operations).
    cache: std::collections::HashMap<(usize, usize), HcsCacheEntry>,
    /// Fast flat lookup: [num_layers * num_experts_per_layer] x 4 pointers.
    /// Each entry: [w13_packed, w13_scales, w2_packed, w2_scales]. All zero = not cached.
    cache_fast: Vec<[u64; 4]>,
    /// Number of layers for flat indexing.
    cache_fast_num_layers: usize,
    /// Activation heatmap: flat [num_layers * num_experts_per_layer] → count.
    heatmap_flat: Vec<u64>,
    /// Legacy heatmap for populate_from_heatmap (only used during collection).
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

    // ── Soft-tier HCS (evicted during prefill, reloaded before decode) ──
    /// Chunked VRAM allocations for soft-tier experts.
    /// Each chunk is ~1 GB (proportional to total VRAM). Chunk 0 = hottest experts.
    /// Eviction drops chunks from the tail (coldest first), proportional to scratch needs.
    soft_chunks: Vec<cudarc::driver::CudaSlice<u8>>,
    /// Number of expert slots per chunk (all chunks same size except possibly the last).
    soft_slots_per_chunk: usize,
    /// Total number of chunks when fully loaded.
    soft_total_chunks: usize,
    /// Number of chunks currently loaded (always a prefix: chunks 0..loaded-1).
    soft_chunks_loaded: usize,
    /// Number of slots in the soft pool.
    soft_num_slots: usize,
    /// Bytes per soft slot (same as pool_slot_size).
    soft_slot_size: usize,
    /// Reverse mapping: soft slot index → (layer, expert).
    soft_slot_to_expert: Vec<Option<(usize, usize)>>,
    /// Ordered list of experts in the soft tier (for reload after eviction).
    /// Stored in ranking order so reload is deterministic.
    soft_ranking: Vec<(usize, usize)>,
    /// Number of soft experts currently loaded.
    soft_num_cached: usize,
    /// Whether soft tier is currently loaded (false during prefill).
    soft_loaded: bool,
    /// Whether an async soft-tier reload is in progress.
    soft_reload_pending: bool,
    /// CUDA event signaling async soft-tier DMA completion.
    soft_reload_event: Option<CudaEvent>,
    /// CUDA stream for async soft-tier reload DMA.
    soft_reload_stream: Option<CudaStream>,
    /// Pending cache entries to activate when async reload completes.
    soft_reload_entries: Vec<(usize, usize, HcsCacheEntry)>,
    /// Pre-packed pinned host-side mirrors of GPU chunks for batch DMA reload.
    /// Each chunk is page-locked via cuMemHostRegister so cuMemcpyHtoDAsync
    /// returns immediately (truly async). Layout mirrors GPU chunks:
    /// expert data packed as [w13_packed|w13_scales|w2_packed|w2_scales] per slot.
    /// One cuMemcpyHtoDAsync per chunk instead of 4 calls per expert.
    soft_host_chunks: Vec<PinnedHostChunk>,
    /// Safety margin in MB — minimum free VRAM to maintain.
    safety_margin_mb: usize,
    /// Hard tier budget in MB (set at init, survives worst-case prefill).
    hard_budget_mb: usize,
    /// Maximum soft tier budget in MB (set at init, sized for decode after short prompt).
    soft_max_mb: usize,

    /// Max experts per layer (stride for flat indexing in cache_fast, heatmap, etc.).
    num_experts_per_layer: usize,

    // ── GPU-side expert pointer table (for CUDA graph capture) ──
    /// Flat GPU buffer: [num_layers * num_experts * 4] u64 pointers.
    /// Layout per expert: [w13_packed, w13_scales, w2_packed, w2_scales].
    /// Zero = not cached. Updated on load/evict.
    d_expert_ptrs: Option<cudarc::driver::CudaSlice<u64>>,
    /// Number of experts per layer for d_expert_ptrs indexing.
    d_expert_ptrs_ne: usize,
    /// Number of layers for d_expert_ptrs indexing.
    d_expert_ptrs_nl: usize,

    // ── Mapped host memory fallback pointers ──
    /// Mapped device pointers for ALL experts (GPU-accessible host memory via PCIe).
    /// Same shape as cache_fast: [num_layers * num_experts_per_layer] x 4 pointers.
    /// When an expert is evicted from HCS, d_expert_ptrs reverts to these instead of zero.
    /// This allows GPU to read cold expert data directly over PCIe without CPU-initiated DMA.
    mapped_fallback: Vec<[u64; 4]>,
    /// Whether mapped reads are available (all experts have mapped device pointers).
    mapped_reads_available: bool,
}

impl HcsState {
    fn new() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            cache_fast: Vec::new(),
            cache_fast_num_layers: 0,
            heatmap_flat: Vec::new(),
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
            soft_chunks: Vec::new(),
            soft_slots_per_chunk: 0,
            soft_total_chunks: 0,
            soft_chunks_loaded: 0,
            soft_num_slots: 0,
            soft_slot_size: 0,
            soft_slot_to_expert: Vec::new(),
            soft_ranking: Vec::new(),
            soft_num_cached: 0,
            soft_loaded: false,
            soft_reload_pending: false,
            soft_reload_event: None,  // CudaEvent
            soft_reload_stream: None, // CudaStream
            soft_reload_entries: Vec::new(),
            soft_host_chunks: Vec::<PinnedHostChunk>::new(),
            safety_margin_mb: 600,
            hard_budget_mb: 0,
            soft_max_mb: 0,
            num_experts_per_layer: 0,
            d_expert_ptrs: None,
            d_expert_ptrs_ne: 0,
            d_expert_ptrs_nl: 0,
            mapped_fallback: Vec::new(),
            mapped_reads_available: false,
        }
    }

    /// Get the GPU base pointer for a soft slot, resolving chunk index.
    #[inline]
    fn soft_slot_ptr(&self, slot: usize) -> u64 {
        let spc = self.soft_slots_per_chunk;
        debug_assert!(spc > 0);
        let chunk_idx = slot / spc;
        let offset = slot % spc;
        let base = *self.soft_chunks[chunk_idx].device_ptr();
        base + (offset as u64 * self.soft_slot_size as u64)
    }

    /// Drop loaded soft chunks down to target_chunks, freeing VRAM and clearing cache entries.
    fn trim_soft_chunks_to(&mut self, target_chunks: usize) -> (usize, usize) {
        if target_chunks >= self.soft_chunks_loaded {
            return (0, 0);
        }

        let spc = self.soft_slots_per_chunk;
        let mut evicted = 0usize;
        let mut freed_bytes = 0usize;

        for drop_idx in target_chunks..self.soft_chunks_loaded {
            let slots_this = if drop_idx == self.soft_total_chunks.saturating_sub(1) {
                self.soft_num_slots.saturating_sub(drop_idx * spc)
            } else {
                spc
            };
            freed_bytes += slots_this * self.soft_slot_size;

            let slot_start = drop_idx * spc;
            let slot_end = std::cmp::min(slot_start + spc, self.soft_num_slots);
            for slot in slot_start..slot_end {
                if let Some((layer_idx, expert_idx)) = self.soft_slot_to_expert[slot].take() {
                    self.cache_fast_clear(layer_idx, expert_idx);
                    if self.cache.remove(&(layer_idx, expert_idx)).is_some() {
                        evicted += 1;
                    }
                }
            }
        }

        self.soft_chunks.truncate(target_chunks);
        self.soft_chunks_loaded = target_chunks;
        self.soft_num_cached = self.soft_num_cached.saturating_sub(evicted);
        self.num_cached = self.num_cached.saturating_sub(evicted);
        self.vram_bytes = self.vram_bytes.saturating_sub(freed_bytes);
        self.soft_loaded = self.soft_chunks_loaded == self.soft_total_chunks;

        (evicted, freed_bytes)
    }

    /// Check if a specific (layer, expert) is cached in VRAM.
    /// Returns the 4 pointers directly, avoiding HashMap lookup.
    #[inline(always)]
    fn get_fast(&self, layer: usize, expert: usize) -> Option<(u64, u64, u64, u64)> {
        if self.num_experts_per_layer == 0 { return None; }
        let idx = layer * self.num_experts_per_layer + expert;
        if idx < self.cache_fast.len() {
            let ptrs = unsafe { *self.cache_fast.get_unchecked(idx) };
            if ptrs[0] != 0 {
                Some((ptrs[0], ptrs[1], ptrs[2], ptrs[3]))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Legacy get for management operations (returns full entry).
    fn get(&self, layer: usize, expert: usize) -> Option<&HcsCacheEntry> {
        self.cache.get(&(layer, expert))
    }

    /// Initialize the flat cache array. Must be called after num_experts_per_layer is set.
    fn init_cache_fast(&mut self, num_layers: usize) {
        self.cache_fast_num_layers = num_layers;
        let total = num_layers * self.num_experts_per_layer;
        self.cache_fast = vec![[0u64; 4]; total];
        self.heatmap_flat = vec![0u64; total];
    }

    /// Update flat cache when an entry is inserted.
    /// Also updates the GPU-side expert pointer table if initialized.
    fn cache_fast_set(&mut self, layer: usize, expert: usize, entry: &HcsCacheEntry) {
        if self.num_experts_per_layer == 0 { return; }
        let idx = layer * self.num_experts_per_layer + expert;
        if idx < self.cache_fast.len() {
            self.cache_fast[idx] = [
                entry.w13_packed_ptr(), entry.w13_scales_ptr(),
                entry.w2_packed_ptr(), entry.w2_scales_ptr(),
            ];
        }
        self.gpu_expert_ptrs_set(layer, expert, entry);
    }

    /// Clear flat cache when an entry is removed.
    /// Also clears the GPU-side expert pointer table if initialized.
    fn cache_fast_clear(&mut self, layer: usize, expert: usize) {
        if self.num_experts_per_layer == 0 { return; }
        let idx = layer * self.num_experts_per_layer + expert;
        if idx < self.cache_fast.len() {
            self.cache_fast[idx] = [0u64; 4];
        }
        self.gpu_expert_ptrs_clear(layer, expert);
    }

    /// Initialize the GPU-side expert pointer table.
    /// Must be called after init_cache_fast and before any CUDA graph capture.
    fn init_gpu_expert_ptrs(&mut self, device: &std::sync::Arc<cudarc::driver::CudaDevice>,
                            num_layers: usize, num_experts: usize) {
        let total = num_layers * num_experts * 4;
        self.d_expert_ptrs_nl = num_layers;
        self.d_expert_ptrs_ne = num_experts;
        match device.alloc_zeros::<u64>(total) {
            Ok(buf) => {
                log::info!("HCS: allocated GPU expert pointer table: {} entries ({:.1} MB)",
                    total, (total * 8) as f64 / 1048576.0);
                self.d_expert_ptrs = Some(buf);
            }
            Err(e) => {
                log::warn!("HCS: failed to allocate GPU expert pointer table: {:?}", e);
            }
        }
    }

    /// Update the GPU-side expert pointer table when an expert is loaded.
    fn gpu_expert_ptrs_set(&self, layer: usize, expert: usize, entry: &HcsCacheEntry) {
        if let Some(ref buf) = self.d_expert_ptrs {
            let idx = (layer * self.d_expert_ptrs_ne + expert) * 4;
            unsafe {
                let dst = *buf.device_ptr() + (idx * 8) as u64;
                // Source must stay alive after this function returns because the
                // driver copy is async. Use the cache_fast backing storage rather
                // than a stack-local array so request-boundary HCS updates cannot
                // race a dead host pointer.
                let src = if self.num_experts_per_layer > 0 {
                    let cache_idx = layer * self.num_experts_per_layer + expert;
                    if cache_idx < self.cache_fast.len() {
                        self.cache_fast[cache_idx].as_ptr() as *const std::ffi::c_void
                    } else {
                        return;
                    }
                } else {
                    return;
                };
                let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    dst,
                    src,
                    32, // 4 * u64
                    std::ptr::null_mut(),
                );
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    log::warn!("gpu_expert_ptrs_set[{},{}]: {:?}", layer, expert, err);
                }
            }
        }
    }

    /// Clear the GPU-side expert pointer table when an expert is evicted.
    /// If mapped fallback is available, writes mapped host device pointers instead of zeros
    /// so the GPU can still read this expert directly over PCIe.
    fn gpu_expert_ptrs_clear(&self, layer: usize, expert: usize) {
        if let Some(ref buf) = self.d_expert_ptrs {
            let idx = (layer * self.d_expert_ptrs_ne + expert) * 4;
            unsafe {
                let dst = *buf.device_ptr() + (idx * 8) as u64;
                // As above, use stable heap-backed host storage for async pointer
                // table updates instead of a stack-local temporary.
                let fallback_idx = if self.num_experts_per_layer > 0 {
                    layer * self.num_experts_per_layer + expert
                } else {
                    return;
                };
                let src = if self.mapped_reads_available && fallback_idx < self.mapped_fallback.len() {
                    self.mapped_fallback[fallback_idx].as_ptr() as *const std::ffi::c_void
                } else if fallback_idx < self.cache_fast.len() {
                    self.cache_fast[fallback_idx].as_ptr() as *const std::ffi::c_void
                } else {
                    return;
                };
                let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    dst,
                    src,
                    32,
                    std::ptr::null_mut(),
                );
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    log::warn!("gpu_expert_ptrs_clear[{},{}]: {:?}", layer, expert, err);
                }
            }
        }
    }

    /// Log HCS state after reload: total experts loaded, cache_fast vs counters,
    /// and check for duplicates (same expert in both hard and soft, or duplicated within soft).
    fn log_hcs_post_reload(&self) {
        let nep = self.num_experts_per_layer;
        if nep == 0 { return; }

        // Count non-zero entries in cache_fast (the actual source of truth for decode lookups)
        let cf_nonzero = self.cache_fast.iter().filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0 || p[3] != 0).count();

        // Count entries in the HashMap cache
        let hm_count = self.cache.len();

        // Check for duplicates: scan cache_fast for entries that appear more than once
        // (shouldn't happen -- each (layer, expert) should appear at most once)
        let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        let mut duplicates = 0usize;
        let num_layers = self.cache_fast_num_layers;
        for layer in 0..num_layers {
            for expert in 0..nep {
                let idx = layer * nep + expert;
                if idx >= self.cache_fast.len() { break; }
                let ptrs = &self.cache_fast[idx];
                if ptrs[0] != 0 || ptrs[1] != 0 || ptrs[2] != 0 || ptrs[3] != 0 {
                    if !seen.insert((layer, expert)) {
                        duplicates += 1;
                    }
                }
            }
        }

        // Check soft_slot_to_expert for duplicates within the soft tier
        let mut soft_seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        let mut soft_dupes = 0usize;
        for slot_opt in &self.soft_slot_to_expert {
            if let Some((l, e)) = slot_opt {
                if !soft_seen.insert((*l, *e)) {
                    soft_dupes += 1;
                }
            }
        }

        let total_possible = num_layers * nep;
        let pct = if total_possible > 0 { cf_nonzero as f64 / total_possible as f64 * 100.0 } else { 0.0 };

        if stderr_debug_enabled() {
            eprintln!("  \x1b[35mHCS post-reload: cache_fast={} ({:.1}%), hashmap={}, num_cached={}, soft_num_cached={}, dupes={}, soft_dupes={}\x1b[0m",
                cf_nonzero, pct, hm_count, self.num_cached, self.soft_num_cached, duplicates, soft_dupes);
        }
        log::info!("HCS post-reload: cache_fast={} ({:.1}%), hashmap={}, num_cached={}, soft_num_cached={}, dupes={}, soft_dupes={}",
            cf_nonzero, pct, hm_count, self.num_cached, self.soft_num_cached, duplicates, soft_dupes);

        // Warn if counters are inconsistent
        if cf_nonzero != hm_count {
            if stderr_debug_enabled() {
                eprintln!("  \x1b[31mHCS WARNING: cache_fast count ({}) != hashmap count ({})\x1b[0m", cf_nonzero, hm_count);
            }
            log::warn!("HCS post-reload: cache_fast count ({}) != hashmap count ({})", cf_nonzero, hm_count);
        }
        if cf_nonzero != self.num_cached {
            if stderr_debug_enabled() {
                eprintln!("  \x1b[31mHCS WARNING: cache_fast count ({}) != num_cached counter ({})\x1b[0m", cf_nonzero, self.num_cached);
            }
            log::warn!("HCS post-reload: cache_fast count ({}) != num_cached counter ({})", cf_nonzero, self.num_cached);
        }
        if duplicates > 0 {
            if stderr_debug_enabled() {
                eprintln!("  \x1b[31mHCS WARNING: {} duplicate entries in cache_fast!\x1b[0m", duplicates);
            }
            log::warn!("HCS post-reload: {} duplicate entries in cache_fast!", duplicates);
        }
        if soft_dupes > 0 {
            if stderr_debug_enabled() {
                eprintln!("  \x1b[31mHCS WARNING: {} duplicate entries in soft_slot_to_expert!\x1b[0m", soft_dupes);
            }
            log::warn!("HCS post-reload: {} duplicate entries in soft_slot_to_expert!", soft_dupes);
        }
    }

    /// Record an expert activation in the heatmap and dynamic eviction bitset.
    fn record_activation(&mut self, layer: usize, expert: usize) {
        if self.collecting {
            let idx = layer * self.num_experts_per_layer + expert;
            if idx < self.heatmap_flat.len() {
                self.heatmap_flat[idx] += 1;
            }
        }
    }

    fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 { 0.0 } else { self.total_hits as f64 / total as f64 }
    }

    fn validation_counts(&self) -> (usize, usize, usize, usize) {
        let cf_nonzero = self.cache_fast.iter()
            .filter(|p| p[0] != 0 || p[1] != 0 || p[2] != 0 || p[3] != 0)
            .count();

        let mut soft_seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        let mut soft_dupes = 0usize;
        for slot_opt in &self.soft_slot_to_expert {
            if let Some((l, e)) = slot_opt {
                if !soft_seen.insert((*l, *e)) {
                    soft_dupes += 1;
                }
            }
        }

        (cf_nonzero, self.cache.len(), 0, soft_dupes)
    }

    fn validation_sorted_resident_experts(&self) -> Vec<(usize, usize)> {
        let mut resident: Vec<(usize, usize)> = self.cache.keys().copied().collect();
        resident.sort_unstable();
        resident
    }

    fn validation_slot_entries(&self) -> Vec<(&'static str, usize, usize, usize)> {
        let mut entries = Vec::with_capacity(self.pool_slot_to_expert.len() + self.soft_slot_to_expert.len());
        for (slot_idx, slot_opt) in self.pool_slot_to_expert.iter().enumerate() {
            if let Some((layer_idx, expert_idx)) = slot_opt {
                entries.push(("hard", slot_idx, *layer_idx, *expert_idx));
            }
        }
        for (slot_idx, slot_opt) in self.soft_slot_to_expert.iter().enumerate() {
            if let Some((layer_idx, expert_idx)) = slot_opt {
                entries.push(("soft", slot_idx, *layer_idx, *expert_idx));
            }
        }
        entries
    }
}

fn validation_fnv1a_u64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn validation_hash_pairs(pairs: &[(usize, usize)]) -> String {
    let mut buf = Vec::with_capacity(pairs.len() * 16);
    for &(layer, expert) in pairs {
        buf.extend_from_slice(&(layer as u64).to_le_bytes());
        buf.extend_from_slice(&(expert as u64).to_le_bytes());
    }
    format!("{:016x}", validation_fnv1a_u64(&buf))
}

fn validation_artifact_paths(prompt_len: usize) -> (String, String, String, String) {
    let ts_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let base = format!("logs/state-validation/prompt{}_{}", prompt_len, ts_ms);
    (
        format!("{}_decode_start_hcs.txt", base),
        format!("{}_decode_start_slots.txt", base),
        format!("{}_decode_cold_experts.txt", base),
        format!("{}_decode_cold_loads.txt", base),
    )
}

fn write_validation_text_file(path: &str, body: &str) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            log::warn!("STATE_VALIDATION failed to create {}: {}", parent.display(), e);
            return;
        }
    }
    match std::fs::File::create(path) {
        Ok(mut file) => {
            if let Err(e) = file.write_all(body.as_bytes()) {
                log::warn!("STATE_VALIDATION failed to write {}: {}", path, e);
            }
        }
        Err(e) => {
            log::warn!("STATE_VALIDATION failed to create {}: {}", path, e);
        }
    }
}

fn validation_record_cold_hist(
    hist: &mut std::collections::BTreeMap<(usize, usize), (u64, usize)>,
    layer_idx: usize,
    expert_idx: usize,
    token_idx: usize,
) {
    let entry = hist.entry((layer_idx, expert_idx)).or_insert((0, token_idx));
    entry.0 += 1;
    if token_idx < entry.1 {
        entry.1 = token_idx;
    }
}

fn validation_record_cold_load(
    hist: &mut std::collections::BTreeMap<(usize, usize), (u64, usize)>,
    events: &mut Vec<(usize, usize, usize)>,
    layer_idx: usize,
    expert_idx: usize,
    token_idx: usize,
) {
    validation_record_cold_hist(hist, layer_idx, expert_idx, token_idx);
    events.push((token_idx, layer_idx, expert_idx));
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
    /// Contiguous pinned buffer: [w13_packed | w13_scales | w2_packed | w2_scales]
    /// When set, DMA can use a single cuMemcpyHtoDAsync call.
    contiguous_ptr: usize,
    contiguous_bytes: usize,
    /// GPU-accessible device pointers to mapped host memory (for zero-copy GPU reads).
    /// Set when cuMemHostRegister with CU_MEMHOSTREGISTER_DEVICEMAP succeeds.
    /// GPU SMs can read this data directly over PCIe without CPU-initiated DMA.
    mapped_w13_packed_dptr: u64,
    mapped_w13_scales_dptr: u64,
    mapped_w2_packed_dptr: u64,
    mapped_w2_scales_dptr: u64,
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
    /// Expert activation type: 0=silu_gated (gate_proj+up_proj with SiLU), 1=relu2 (up_proj only, relu^2).
    activation_type: u8,
    /// Whether experts have a gate projection (w13 = [gate|up]) or just up_proj (w13 = [up]).
    gated_experts: bool,
    /// Latent down projection weight ID (hidden -> latent, e.g. 4096->1024). None for standard MoE.
    latent_down_wid: Option<usize>,
    /// Latent up projection weight ID (latent -> hidden, e.g. 1024->4096). None for standard MoE.
    latent_up_wid: Option<usize>,
    /// Expert input/output size for LatentMoE (e.g. 1024). 0 = use hidden_size (standard MoE).
    /// This is the K dimension for w13 GEMV and N dimension for w2 GEMV.
    moe_input_size: usize,
}

// ── GPU weight descriptor ──────────────────────────────────────────────

/// Describes a single weight matrix resident in VRAM.
#[derive(Debug, Clone)]
struct GpuWeight {
    ptr: u64,
    rows: usize,
    cols: usize,
    /// Data type: 0 = BF16, 1 = FP32, 2 = FP16, 4 = Marlin INT8, 5 = Marlin INT4.
    dtype: u8,
    /// For Marlin INT8/INT4 (dtype=4/5): pointer to BF16 scales on GPU.
    /// Layout: [K/group_size, N] where N=rows, K=cols.
    scales_ptr: u64,
    /// Quantization group size for Marlin INT8/INT4 (typically 128).
    group_size: usize,
    /// Simple INT4 decode format: [rows, cols/2] packed u8 on GPU.
    /// When non-zero, decode GEMV uses this instead of Marlin (faster at M=1).
    simple_packed_ptr: u64,
    /// Simple INT4 decode format: [rows, cols/group_size] FP32 scales on GPU.
    simple_scales_f32_ptr: u64,
}

impl GpuWeight {
    /// Create a standard weight (BF16/FP32/FP16/FP8).
    fn new(ptr: u64, rows: usize, cols: usize, dtype: u8) -> Self {
        Self { ptr, rows, cols, dtype, scales_ptr: 0, group_size: 0,
               simple_packed_ptr: 0, simple_scales_f32_ptr: 0 }
    }

    /// Create a Marlin INT8 weight with packed data and scales.
    fn new_marlin_int8(packed_ptr: u64, scales_ptr: u64, rows: usize, cols: usize, group_size: usize) -> Self {
        Self { ptr: packed_ptr, rows, cols, dtype: 4, scales_ptr, group_size,
               simple_packed_ptr: 0, simple_scales_f32_ptr: 0 }
    }

    /// Create a Marlin INT4 weight with packed data and scales.
    fn new_marlin_int4(packed_ptr: u64, scales_ptr: u64, rows: usize, cols: usize, group_size: usize) -> Self {
        Self { ptr: packed_ptr, rows, cols, dtype: 5, scales_ptr, group_size,
               simple_packed_ptr: 0, simple_scales_f32_ptr: 0 }
    }

    /// Whether this weight has a simple INT4 decode format available.
    fn has_simple_int4(&self) -> bool {
        self.simple_packed_ptr != 0
    }

    fn cublas_data_type(&self) -> cublas_sys::cudaDataType {
        match self.dtype {
            0 => cublas_sys::cudaDataType::CUDA_R_16BF,
            1 => cublas_sys::cudaDataType::CUDA_R_32F,
            2 => cublas_sys::cudaDataType::CUDA_R_16F,
            _ => cublas_sys::cudaDataType::CUDA_R_16BF,
        }
    }

    fn is_marlin_int8(&self) -> bool {
        self.dtype == 4
    }

    fn is_marlin_int4(&self) -> bool {
        self.dtype == 5
    }

    fn is_marlin(&self) -> bool {
        self.dtype == 4 || self.dtype == 5
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
    MLA {
        // Q projection: q_lora path (q_a_proj → layernorm → q_b_proj) or direct q_proj
        q_a_proj: Option<usize>,   // None for direct q_proj
        q_b_proj: Option<usize>,
        q_a_norm_ptr: u64,         // FP32 layernorm weight for q_a (0 if direct)
        q_proj: Option<usize>,     // Some for direct, None for q_lora
        // KV projection
        kv_a_proj: usize,
        kv_a_norm_ptr: u64,        // FP32 layernorm weight for kv_a
        // Absorbed weights (BF16, kept on GPU permanently)
        w_kc_ptr: u64,             // [num_heads, qk_nope_dim, ckv_cache_dim] BF16
        w_vc_ptr: u64,             // [num_heads, v_head_dim, ckv_cache_dim] BF16
        // Output
        o_proj: usize,
        // Dimensions
        num_heads: usize,
        kv_lora_rank: usize,       // real kv_lora_rank from model (e.g. 256 for Mistral)
        ckv_cache_dim: usize,      // effective ckv dim in cache (≥512 for MLA decode)
        qk_nope_dim: usize,
        qk_rope_dim: usize,
        v_head_dim: usize,
        q_lora_rank: usize,        // 0 if direct q_proj
        sm_scale: f32,
        rope_interleave: bool,
        // MLA KV cache (separate ckv and kpe, FP8)
        ckv_cache_ptr: u64,        // [max_seq, ckv_cache_dim] FP8
        kpe_cache_ptr: u64,        // [max_seq, qk_rope_dim] FP8
    },
    /// Mamba2 selective state space layer (Nemotron-H hybrid models).
    /// O(1) per-token decode, no KV cache. State is always GPU-resident.
    Mamba2 {
        in_proj: usize,            // weight ID: hidden_size -> expand * (head_dim + 2*state_size + 1) * num_heads
        out_proj: usize,           // weight ID: expand * head_dim * num_heads -> hidden_size
        conv_weight_ptr: u64,      // [conv_dim, conv_kernel] FP32 on GPU
        a_ptr: u64,                // [num_heads] FP32 — discretization parameter A
        d_ptr: u64,                // [num_heads] FP32 — skip connection D
        dt_bias_ptr: u64,          // [num_heads] FP32
        norm_weight_ptr: u64,      // [expand * head_dim * num_heads] FP32 — output RMSNorm
        num_heads: usize,          // 128 for Nemotron
        head_dim: usize,           // 64
        state_size: usize,         // 128
        expand: usize,             // 2
        conv_kernel: usize,        // 4
        conv_dim: usize,           // expand * head_dim * num_heads (conv input dim)
        conv_state_ptr: u64,       // [conv_dim, conv_kernel] FP32 on GPU (per-sequence)
        ssm_state_ptr: u64,        // [num_heads, head_dim, state_size] FP32 on GPU (per-sequence)
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

// ── Pinned mapped memory for zero-copy D2H ────────────────────────────
// GPU writes directly to host memory via PCIe BAR. No explicit D2H copy needed.
// Used for topk routing results (tiny: 80 bytes, but called 96x/token = 0.77ms overhead).
struct PinnedMapped {
    host_ptr: *mut u8,
    device_ptr: u64,
    size: usize,
}

impl PinnedMapped {
    fn new(size: usize) -> Result<Self, String> {
        let mut host_ptr: *mut u8 = std::ptr::null_mut();
        let flags = 0x02; // CU_MEMHOSTALLOC_DEVICEMAP
        unsafe {
            let err = cuda_sys::lib().cuMemHostAlloc(
                &mut host_ptr as *mut *mut u8 as *mut *mut std::ffi::c_void,
                size, flags);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuMemHostAlloc({} bytes): {:?}", size, err));
            }
            let mut dptr: u64 = 0;
            let err = cuda_sys::lib().cuMemHostGetDevicePointer_v2(
                &mut dptr, host_ptr as *mut std::ffi::c_void, 0);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                cuda_sys::lib().cuMemFreeHost(host_ptr as *mut std::ffi::c_void);
                return Err(format!("cuMemHostGetDevicePointer: {:?}", err));
            }
            Ok(Self { host_ptr, device_ptr: dptr, size })
        }
    }
}

impl Drop for PinnedMapped {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            unsafe { cuda_sys::lib().cuMemFreeHost(self.host_ptr as *mut std::ffi::c_void); }
        }
    }
}

// Safety: host_ptr points to pinned memory accessible from any thread
unsafe impl Send for PinnedMapped {}
unsafe impl Sync for PinnedMapped {}

// ── Cached kernel function handles ─────────────────────────────────────
// Avoids HashMap lookup per kernel call (~470 lookups per token eliminated).

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
    fused_silu_accum_int8: cudarc::driver::CudaFunction,
    // v2 kernels with K-splitting for better SM occupancy
    marlin_gemv_int4_v2: cudarc::driver::CudaFunction,
    reduce_ksplits_bf16: cudarc::driver::CudaFunction,
    fused_silu_accum_v2: cudarc::driver::CudaFunction,
    fused_silu_accum_v2_int8: cudarc::driver::CudaFunction,
    reduce_ksplits_weighted_accum_bf16: cudarc::driver::CudaFunction,
    // Attention kernels (LA + GQA) — eliminates ~470 HashMap lookups per token
    uninterleave_qkvz: cudarc::driver::CudaFunction,
    la_conv1d: cudarc::driver::CudaFunction,
    compute_gate_beta: cudarc::driver::CudaFunction,
    repeat_interleave_heads: cudarc::driver::CudaFunction,
    l2norm_scale_per_head: cudarc::driver::CudaFunction,
    gated_delta_net_step: cudarc::driver::CudaFunction,
    gated_rmsnorm_silu: cudarc::driver::CudaFunction,
    split_gated_q: cudarc::driver::CudaFunction,
    per_head_rmsnorm: cudarc::driver::CudaFunction,
    apply_rope: cudarc::driver::CudaFunction,
    kv_cache_write: cudarc::driver::CudaFunction,
    gqa_attention: cudarc::driver::CudaFunction,
    gqa_attention_tiled: cudarc::driver::CudaFunction,
    gqa_attention_reduce: cudarc::driver::CudaFunction,
    apply_gated_attn: cudarc::driver::CudaFunction,
    // Fused v2 kernels with inline atomic reduction (no separate reduce kernel)
    marlin_gemv_int4_v2_fused_f32: cudarc::driver::CudaFunction,
    marlin_gemv_int8_v2: cudarc::driver::CudaFunction,
    marlin_gemv_int8_v2_fused_f32: cudarc::driver::CudaFunction,
    // Batched expert kernels — reduce launch overhead from 30/layer to 4/layer
    marlin_gemv_int4_v2_batched: cudarc::driver::CudaFunction,
    marlin_gemv_int8_v2_batched: cudarc::driver::CudaFunction,
    reduce_ksplits_bf16_batched: cudarc::driver::CudaFunction,
    fused_silu_w2_batched: cudarc::driver::CudaFunction,
    fused_silu_w2_int8_batched: cudarc::driver::CudaFunction,
    multi_expert_weighted_add_bf16: cudarc::driver::CudaFunction,
    // Graphable kernel variants (read position/token from GPU pointers)
    embedding_lookup_g: cudarc::driver::CudaFunction,
    apply_rope_g: cudarc::driver::CudaFunction,
    kv_cache_write_g: cudarc::driver::CudaFunction,
    gqa_attention_g: cudarc::driver::CudaFunction,
    // BF16-output variants (eliminate fp32_to_bf16 conversion kernel)
    gated_rmsnorm_silu_bf16: cudarc::driver::CudaFunction,
    gqa_attention_g_bf16: cudarc::driver::CudaFunction,
    apply_gated_attn_bf16: cudarc::driver::CudaFunction,
    // Tiled GQA for CUDA graph capture (reads d_seq_len from GPU pointer)
    gqa_attention_tiled_g: cudarc::driver::CudaFunction,
    gqa_attention_reduce_g: cudarc::driver::CudaFunction,
    // Fused LA post-projection: repeat_interleave + l2norm + delta_net + rmsnorm → BF16
    la_fused_post_proj: cudarc::driver::CudaFunction,
    // GPU-side expert classification (eliminates cuStreamSynchronize in route sync)
    expert_classify_prepare: cudarc::driver::CudaFunction,
    // MLA (Multi-head Latent Attention) kernels
    mla_kv_cache_write_g: cudarc::driver::CudaFunction,
    mla_kv_cache_write: cudarc::driver::CudaFunction,
    mla_attention_g: cudarc::driver::CudaFunction,
    mla_attention: cudarc::driver::CudaFunction,
    mla_deinterleave: cudarc::driver::CudaFunction,
    mla_split_q: cudarc::driver::CudaFunction,
    mla_absorb_wkc: cudarc::driver::CudaFunction,
    mla_apply_wvc: cudarc::driver::CudaFunction,
    // 4-bit PolarQuant kernels
    kv_cache_write_polar4: cudarc::driver::CudaFunction,
    gqa_attention_polar4: cudarc::driver::CudaFunction,
    kv_cache_write_polar4_g: cudarc::driver::CudaFunction,
    gqa_attention_polar4_g: cudarc::driver::CudaFunction,
    gqa_attention_polar4_tiled_g: cudarc::driver::CudaFunction,
    gqa_attention_polar4_reduce_g: cudarc::driver::CudaFunction,
    // Mamba2 SSM kernels (Nemotron-H)
    mamba2_conv1d: cudarc::driver::CudaFunction,
    mamba2_ssm_step: cudarc::driver::CudaFunction,
    mamba2_discretize: cudarc::driver::CudaFunction,
    mamba2_gate_output: cudarc::driver::CudaFunction,
    // relu2 expert activation (Nemotron LatentMoE)
    relu2_w2_batched: cudarc::driver::CudaFunction,
    relu2_w2_int8_batched: cudarc::driver::CudaFunction,
}

// ── Main GPU decode graph ──────────────────────────────────────────────

struct GpuDecodeGraph {
    hidden_size: usize,
    #[allow(dead_code)]
    num_layers: usize,
    /// Decode segment: the range of layers this store is responsible for during multi-GPU decode.
    /// Default: [0..num_layers) for single-GPU or primary store, narrowed for aux stores.
    decode_layer_start: usize,
    decode_layer_end: usize,
    vocab_size: usize,
    eps: f32,
    intermediate_size: usize,         // max intermediate (for buffer allocation)
    moe_intermediate_size: usize,     // MoE expert intermediate (for expert kernels)
    group_size: usize,
    /// Expert quantization bits: 4 (INT4 Marlin) or 8 (INT8 Marlin).
    expert_bits: u8,
    /// Shared expert quantization bits (may differ from expert_bits, e.g. INT8 shared with INT4 routed).
    shared_expert_bits: u8,

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

    // Per-component expert buffers (w13_packed, w13_scales, w2_packed, w2_scales).
    // Used by DMA upload and GEMV kernel dispatch.
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

    // Marlin GEMV inverse permutation tables (on GPU) — INT4
    d_inv_weight_perm: cudarc::driver::CudaSlice<i32>,
    d_inv_scale_perm: cudarc::driver::CudaSlice<i32>,
    // Marlin GEMV inverse permutation tables (on GPU) — INT8
    d_inv_weight_perm_int8: cudarc::driver::CudaSlice<i32>,

    // v2 K-split partial sum buffer: [max_k_splits * max_N] FP32
    // max_N = max(2*intermediate_size, hidden_size), max_k_splits = 8
    d_v2_partial: cudarc::driver::CudaSlice<f32>,
    num_sms: usize,

    // ── Batched expert buffers (reduce kernel launch overhead) ──
    // Per-expert gate_up outputs: [max_experts_per_tok * 2 * intermediate_size] BF16
    d_batch_gate_ups: cudarc::driver::CudaSlice<u16>,
    // Per-expert v2 partial sums: [max_experts_per_tok * max_k_splits * max_N] FP32
    d_batch_partials: cudarc::driver::CudaSlice<f32>,
    // Per-expert w2 GEMV outputs: [max_experts_per_tok * hidden_size] BF16
    d_batch_expert_outs: cudarc::driver::CudaSlice<u16>,
    // Device arrays for batched kernel pointer arguments (uploaded each layer)
    d_batch_w13_packed_ptrs: cudarc::driver::CudaSlice<u64>,
    d_batch_w13_scales_ptrs: cudarc::driver::CudaSlice<u64>,
    d_batch_w2_packed_ptrs: cudarc::driver::CudaSlice<u64>,
    d_batch_w2_scales_ptrs: cudarc::driver::CudaSlice<u64>,
    // Device array for batched routing weights: [max_experts_per_tok] FP32
    d_batch_weights: cudarc::driver::CudaSlice<f32>,
    // Host staging buffers for pointer upload
    h_batch_w13_packed_ptrs: Vec<u64>,
    h_batch_w13_scales_ptrs: Vec<u64>,
    h_batch_w2_packed_ptrs: Vec<u64>,
    h_batch_w2_scales_ptrs: Vec<u64>,
    h_batch_weights: Vec<f32>,
    max_experts_per_tok: usize,
    // Contiguous upload buffer (1 H2D instead of 5): [4*max*8 + max*4] bytes
    d_batch_upload: cudarc::driver::CudaSlice<u8>,
    h_batch_upload: Vec<u8>,
    batch_upload_ptrs_bytes: usize, // 4 * max * 8
    batch_upload_total_bytes: usize, // 4 * max * 8 + max * 4

    // GQA scratch (FP32 for Q, K, V, attention output)
    d_gqa_q: cudarc::driver::CudaSlice<f32>,
    d_gqa_k: cudarc::driver::CudaSlice<f32>,
    d_gqa_v: cudarc::driver::CudaSlice<f32>,
    d_gqa_out: cudarc::driver::CudaSlice<f32>,

    // MLA scratch (FP32, allocated on demand)
    d_mla_q_absorbed: cudarc::driver::CudaSlice<f32>,   // [num_heads * kv_lora_rank] for absorbed query
    d_mla_kv: cudarc::driver::CudaSlice<f32>,           // [kv_lora_rank + qk_rope_dim] for KV projection
    d_mla_attn_out: cudarc::driver::CudaSlice<f32>,     // [num_heads * kv_lora_rank] attention output

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

    // Mamba2 SSM scratch (FP32)
    d_mamba2_conv_out: Option<cudarc::driver::CudaSlice<f32>>,  // [conv_dim] FP32
    mamba2_conv_out_size: usize,  // track allocated size (CudaSlice has no .len())
    /// Number of B/C groups for Mamba2 (n_groups from config, e.g. 8)
    mamba2_n_groups: usize,
    /// Conv bias pointers per Mamba2 layer (FP32 on GPU, keyed by layer_idx)
    mamba2_conv_bias_ptrs: std::collections::HashMap<usize, u64>,

    /// LatentMoE: override pointer for expert w13 input. 0 = use d_hidden (standard MoE).
    /// Set to the latent buffer ptr before moe_forward for LatentMoE layers.
    moe_input_override_ptr: u64,

    // Host-side buffers for D2H copies
    h_topk_ids: Vec<i32>,
    h_topk_weights: Vec<f32>,
    h_logits: Vec<f32>,

    // Pinned mapped memory for zero-copy topk (replaces d_topk_indices/weights + D2H)
    pinned_topk_ids: Option<PinnedMapped>,
    pinned_topk_weights: Option<PinnedMapped>,

    // GPU-side route sync: mapped memory for cold expert communication
    // Layout: [cold_count(i32), ready_flag(i32), cold_ids[topk](i32), cold_slots[topk](i32)]
    mapped_cold_buf: Option<PinnedMapped>,
    /// Whether GPU-side route sync is active (classify kernel available + HCS has d_expert_ptrs)
    gpu_route_sync: bool,
    /// Whether mapped host memory reads are active (GPU reads cold experts directly over PCIe).
    /// When true, d_expert_ptrs contains valid pointers for ALL experts (VRAM for cached,
    /// mapped host for cold), and no CPU sync/DMA is needed between graph replays.
    mapped_reads_active: bool,
    /// Mapped activation buffer for recording topk IDs when mapped reads bypass CPU sync.
    /// Layout: [num_moe_layers * topk] i32 — classify kernel writes per-layer topk IDs here.
    mapped_activations: Option<PinnedMapped>,

    // Cached kernel function handles (populated after configure)
    kernels: Option<CachedKernels>,

    // Pre-allocated CUDA events for MoE forward (avoid create/destroy per layer)
    // [0..1] for DMA done, [2..3] for compute done on double-buffer slots
    pre_events: Option<[CudaEvent; 4]>,

    // ── Full decode step state ──

    /// Whether model norms use (1+w)*x instead of w*x.
    norm_bias_one: bool,

    /// GQA KV cache: raw device pointers to FP8 E4M3 [max_seq, kv_stride] per layer.
    /// Memory is owned by Python (PagedKVCache tensors). Rust prefill writes FP8
    /// via kv_cache_append kernel, decode writes FP8 via kv_cache_write kernel.
    /// Shared buffer, no export copy needed.
    kv_k_ptrs: Vec<u64>,  // device pointers, one per layer (indexed by layer_idx)
    kv_v_ptrs: Vec<u64>,
    kv_max_seq: usize,
    kv_current_pos: usize,
    /// KV cache format: 0=bf16, 1=fp8, 2=polar4
    kv_format: u32,
    /// Polar4 KV cache pointers (only used when kv_format==2)
    kv_k_radius_ptrs: Vec<u64>,
    kv_v_radius_ptrs: Vec<u64>,
    kv_k_angles_ptrs: Vec<u64>,
    kv_v_angles_ptrs: Vec<u64>,
    /// Number of 16-element blocks per KV stride (for polar4)
    kv_num_blocks: usize,

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
    // Fine-grained MoE timing (within "MoE other")
    t_moe_gate_gemv: f64,    // gate GEMV + topk kernel launch (pre-sync)
    t_moe_d2h_topk: f64,     // D2H copy of topk indices/weights
    t_moe_apfl: f64,         // APFL speculative routing for next layer
    t_moe_padding_setup: f64, // replay batch pointer/weight padding + upload staging
    t_moe_d2d_copy: f64,     // D2D moe_out -> hidden copy
    t_moe_accum: f64,        // weighted accumulation into moe_out
    // Attention breakdown
    t_attn_la: f64,          // linear attention layers
    t_attn_gqa: f64,         // GQA (full attention) layers
    // LA sub-component timing
    t_la_proj: f64,          // LA projections (2 cuBLAS GEMVs)
    t_la_conv: f64,          // LA conv1d + gate/beta
    t_la_recur: f64,         // LA recurrence (repeat-interleave + l2norm + delta net)
    t_la_out: f64,           // LA gated rmsnorm + output projection
    // GQA sub-component timing
    t_gqa_proj: f64,         // GQA QKV projections + split + norm + RoPE + KV write
    t_gqa_attn: f64,         // GQA attention kernel
    t_gqa_out: f64,          // GQA gated + O projection
    // Expert loop sub-component timing
    t_expert_w13: f64,       // w13 GEMV (gate+up projection)
    t_expert_silu_w2: f64,   // fused silu_mul + w2 GEMV + weighted_add
    // DMA expert sub-timing (Phase 3 cold experts)
    t_dma_expert_wait: f64,  // time waiting for DMA events (cuStreamWaitEvent + actual DMA)
    t_dma_expert_compute: f64, // compute time for DMA'd experts (w13+silu+w2)
    // DMA instrumentation (accumulated across all layers per token, then across tokens)
    dma_bytes_total: u64,    // total bytes DMA'd (cold experts only)
    dma_call_count: u64,     // number of cuMemcpyHtoDAsync calls
    dma_cold_experts: u64,   // number of cold (DMA'd) experts
    dma_hcs_experts: u64,    // number of HCS-hit experts
    validation_decode_steps: u64,
    validation_per_layer_steps: u64,
    validation_ungraphed_steps: u64,
    validation_decode_start_num_cached: usize,
    validation_decode_start_soft_num_cached: usize,
    validation_decode_start_hard_num_cached: usize,
    validation_decode_start_dupes: usize,
    validation_decode_start_soft_dupes: usize,
    validation_decode_start_hash: String,
    validation_decode_start_resident: Vec<(usize, usize)>,
    validation_decode_start_hcs_file: String,
    validation_decode_start_slots: Vec<(&'static str, usize, usize, usize)>,
    validation_decode_start_slots_file: String,
    validation_decode_cold_hist: std::collections::BTreeMap<(usize, usize), (u64, usize)>,
    validation_decode_cold_events: Vec<(usize, usize, usize)>,
    validation_decode_cold_file: String,
    validation_decode_cold_events_file: String,

    // ── Speculative decode batch buffers (allocated when draft model loaded) ──
    /// Max batch size for speculative decode (draft_k + 1).
    batch_max: usize,
    /// [batch_max * hidden_size] BF16 — per-token hidden states during batch decode.
    d_batch_hidden: Option<cudarc::driver::CudaSlice<u16>>,
    /// [batch_max * hidden_size] BF16 — per-token residual states during batch decode.
    d_batch_residual: Option<cudarc::driver::CudaSlice<u16>>,
    /// [batch_max * hidden_size] BF16 — per-token MoE output accumulator.
    d_batch_moe_out: Option<cudarc::driver::CudaSlice<u16>>,
    /// [batch_max * vocab_size] FP32 — per-token logits from LM head.
    d_batch_logits: Option<cudarc::driver::CudaSlice<f32>>,
    /// Host-side copy of batch logits.
    h_batch_logits: Vec<f32>,
    /// Per-token routing results: [batch_max * max_topk] on host.
    h_batch_topk_ids: Vec<i32>,
    h_batch_topk_weights: Vec<f32>,
    /// Per-token routing results on GPU: [batch_max * max_topk].
    d_batch_topk_ids: Option<cudarc::driver::CudaSlice<i32>>,
    d_batch_topk_wts: Option<cudarc::driver::CudaSlice<f32>>,
    /// Per-token gate logits on GPU: [batch_max * num_experts] FP32.
    d_batch_gate_logits: Option<cudarc::driver::CudaSlice<f32>>,

    /// LA state backup for rollback after rejected draft tokens.
    /// Each entry: (conv_state_backup, recur_state_backup) for one LA layer.
    /// Allocated once when batch buffers are allocated.
    la_backup: Vec<LaStateBackup>,
    /// Hidden states saved at each LA layer entry for each batch token,
    /// used during LA replay after rollback. [num_la_layers * batch_max * hidden_size] BF16.
    d_la_hidden_stack: Option<cudarc::driver::CudaSlice<u16>>,

    // ── Batched GEMM projection buffers (allocated with batch buffers) ──
    /// [batch_max * max_proj_dim] FP32 — primary batch projection output (qkvz / fused_qkv / Q).
    d_batch_proj_a: Option<cudarc::driver::CudaSlice<f32>>,
    /// [batch_max * max_proj_dim] FP32 — secondary batch projection output (ba / K / V).
    d_batch_proj_b: Option<cudarc::driver::CudaSlice<f32>>,
    /// [batch_max * max_attn_out_dim] BF16 — gathered attention outputs for batched O projection.
    d_batch_attn_out: Option<cudarc::driver::CudaSlice<u16>>,
    /// Maximum projection output dimension across all layers.
    batch_max_proj_dim: usize,
    /// Maximum attention output dimension across all layers (for O projection input).
    batch_max_attn_out_dim: usize,

    // ── Multi-GPU segment config ──
    /// When set, restricts the layer loop to [segment_layer_start..segment_layer_end).
    /// Default: 0 and num_layers (full decode).
    segment_layer_start: usize,
    segment_layer_end: usize, // 0 = use num_layers
    /// Skip embedding lookup (hidden state uploaded from another GPU).
    segment_skip_embedding: bool,
    /// Skip final norm + LM head (hidden state will be downloaded for transfer).
    segment_skip_final: bool,
    /// GQA cache index offset (number of GQA layers before segment_layer_start).
    segment_gqa_cache_offset: usize,

    // ── CUDA Graph capture state ──
    /// GPU-side token ID for graphable embedding lookup (1 element).
    d_graph_token_id: Option<cudarc::driver::CudaSlice<i32>>,
    /// GPU-side decode position for graphable attention kernels (1 element).
    d_graph_pos: Option<cudarc::driver::CudaSlice<i32>>,
    /// GPU-side sequence length (pos+1) for graphable GQA attention (1 element).
    d_graph_seq_len: Option<cudarc::driver::CudaSlice<i32>>,
    /// Dummy expert buffer (zeros) for cold experts during CUDA graph replay.
    /// Sized to hold one full expert (w13_packed + w13_scales + w2_packed + w2_scales).
    d_dummy_expert: Option<cudarc::driver::CudaSlice<u8>>,
    /// GPU-side array of 4 u64 dummy pointers [w13p, w13s, w2p, w2s] for unused batch slots.
    d_dummy_ptrs: Option<cudarc::driver::CudaSlice<u64>>,
    /// Host-side copy of dummy expert pointers for zero-weight padding during replay.
    h_dummy_ptrs: [u64; 4],

    // ── Per-layer CUDA graph capture (49 graphs for 48 MoE layers) ──
    /// Per-layer graph executables: graph[0] = routing for first MoE layer,
    /// graph[1..N-1] = experts(N-1) + routing(N), graph[N] = experts(last) + final.
    per_layer_graphs: Vec<CudaGraphExecPtr>,
    /// Whether per-layer graphs are valid for replay.
    per_layer_graphs_valid: bool,
    /// Whether graphs have ever been successfully captured (for cross-request reuse).
    /// After the first capture, KV cache and LA state buffer addresses are stable
    /// (KV is allocated once at model load; LA Rust-side FP32 buffers use copy_()
    /// in-place after first allocation). So graphs can be reused across requests
    /// without recapture -- only the scalar params (token_id, pos, seq_len) and
    /// LA state contents change, both handled via GPU-side indirection buffers.
    graphs_ever_captured: bool,
    /// Snapshot of LA state pointers at capture time, for verifying address stability.
    /// Vec of (layer_idx, conv_state_ptr, recur_state_ptr).
    captured_la_ptrs: Vec<(usize, u64, u64)>,
    /// Snapshot of KV cache pointers at capture time, for verifying address stability.
    /// Vec of (layer_idx, k_ptr, v_ptr).
    captured_kv_ptrs: Vec<(usize, u64, u64)>,
    /// Snapshot of MLA KV cache pointers at capture time (ckv_cache + kpe_cache).
    /// Vec of (layer_idx, ckv_cache_ptr, kpe_cache_ptr).
    captured_mla_ptrs: Vec<(usize, u64, u64)>,
    /// Snapshot of Polar4 KV cache pointers at capture time (k_radius, v_radius, k_angles, v_angles).
    /// Vec of (layer_idx, k_radius_ptr, v_radius_ptr, k_angles_ptr, v_angles_ptr).
    captured_polar4_ptrs: Vec<(usize, u64, u64, u64, u64)>,
    /// MoE layer indices (which layers have MoE data).
    per_layer_moe_indices: Vec<usize>,
    /// Persistent cuBLAS workspace for CUDA graph capture (prevents internal cudaMalloc).
    d_cublas_workspace: Option<cudarc::driver::CudaSlice<u8>>,
}

/// Backup storage for one LA layer's mutable state.
/// Pointers are NOT cached — they're read dynamically from the graph's layer config
/// because prefill re-registers LA layers with new tensor pointers.
struct LaStateBackup {
    layer_idx: usize,
    conv_state_bytes: usize,
    recur_state_bytes: usize,
    d_conv_backup: cudarc::driver::CudaSlice<u8>,
    d_recur_backup: cudarc::driver::CudaSlice<u8>,
}

// ── Thread-safe CUDA wrappers ──────────────────────────────────────────

struct CudaStream(cuda_sys::CUstream);
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

struct CudaEvent(cuda_sys::CUevent);
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

/// Wrapper for cublas handle to allow Send+Sync (CUDA handles are thread-safe).
struct CublasHandle(cublas_sys::cublasHandle_t);
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

/// Wrapper for CUfunction to allow Send+Sync (CUDA function handles are thread-safe).
struct CudaFunc(cuda_sys::CUfunction);
unsafe impl Send for CudaFunc {}
unsafe impl Sync for CudaFunc {}

struct CudaGraphExecPtr(CUgraphExec);
unsafe impl Send for CudaGraphExecPtr {}
unsafe impl Sync for CudaGraphExecPtr {}

/// Single-slot AWQ: one GPU allocation per weight, shared between Marlin (prefill) and
/// simple INT4 (decode). Host copies of both formats are kept for DMA swaps.
struct SingleSlotSwapEntry {
    weight_id: usize,
    /// Fixed GPU address for packed weights (never changes — both formats share this slot).
    packed_slot_ptr: u64,
    /// Fixed GPU address for scales (never changes — sized to max of BF16 and FP32).
    scales_slot_ptr: u64,
    /// Marlin packed data (tile-permuted INT4) saved from GPU during registration.
    marlin_packed_host: Vec<u8>,
    /// Marlin BF16 scales saved from GPU during registration.
    marlin_scales_host: Vec<u8>,
    /// Simple INT4 packed data (sequential layout) from repack.
    simple_packed_host: Vec<u8>,
    /// Simple INT4 FP32 scales from repack (as raw bytes).
    simple_scales_host: Vec<u8>,
}

// ── PyO3 wrapper ───────────────────────────────────────────────────────

#[pyclass]
pub struct GpuDecodeStore {
    device: Arc<CudaDevice>,
    blas: CudaBlas,
    compute_stream: CudaStream,
    copy_stream: CudaStream,
    /// Dedicated stream for APFL prefetch DMA.
    /// Runs independently from copy_stream so prefetch can overlap with on-demand DMA.
    prefetch_stream: CudaStream,
    /// Dedicated stream for APFL speculative routing (gate GEMV + topk).
    /// Runs independently from default stream so spec routing overlaps with expert loop.
    spec_stream: CudaStream,
    /// cuBLAS handle bound to spec_stream for speculative gate GEMV.
    spec_blas_handle: CublasHandle,
    /// Raw CUfunction handles for launching topk kernels on spec_stream via cuLaunchKernel.
    raw_sigmoid_topk: CudaFunc,
    raw_softmax_topk: CudaFunc,
    graph: Option<Box<GpuDecodeGraph>>,
    kernels_loaded: bool,
    /// Cached Marlin perm table device pointers (set during configure, never change).
    /// Used by attention Marlin GEMV launchers which can't access graph (it's .take()d).
    perm_inv_weight_int4: u64,
    perm_inv_weight_int8: u64,
    perm_inv_scale: u64,
    /// Cached v2 partial sum buffer pointer (for attention Marlin GEMV v2).
    /// Set during configure, never changes. Used by gemv_bf16_internal.
    v2_partial_ptr: u64,
    v2_num_sms: usize,
    /// Max opt-in shared memory for GQA attention (bytes). Set during PTX load.
    gqa_max_smem_bytes: u32,
    last_decode_elapsed: f64,
    /// Draft model for speculative decoding (None = disabled).
    draft: Option<crate::draft_model::DraftModel>,
    /// Number of tokens to draft per speculative round.
    draft_k: usize,
    /// Context window for draft model warmup (last N tokens of prompt).
    draft_context_window: usize,
    /// Jaccard similarity threshold for fail-fast expert divergence detection.
    /// At each MoE layer during batched verification, if a draft token's expert
    /// routing has Jaccard similarity < this threshold vs token[0], that token
    /// and all subsequent draft tokens are dropped from the batch.
    /// Lower = more lenient (fewer bailouts), higher = stricter (more bailouts).
    /// Default 0.15 ≈ at least 2-3 shared experts out of topk=10.
    spec_jaccard_threshold: f32,
    /// Min free VRAM (MB) observed during the most recent decode run.
    last_min_free_vram_mb: usize,
    last_soft_evict_experts: usize,
    last_soft_evict_freed_mb: f64,
    last_soft_reload_queued: usize,
    last_soft_reload_alloc_mb: f64,
    last_soft_reload_activated: usize,
    /// Four-point VRAM calibration data for runtime budget decisions.
    vram_calibration: Option<VramCalibration>,
    /// Token IDs to suppress during sampling (logit set to -inf).
    /// Used to prevent the model from generating turn-boundary tokens
    /// (e.g. <|im_start|>) which cause phantom new turns in multi-turn chat.
    suppress_tokens: Vec<usize>,
    /// Minimum tokens before stop tokens are honored. When > 0, stop token IDs
    /// are added to suppress_tokens for the first min_new_tokens decode steps.
    /// Prevents models from bailing out with EOS on early multi-turn tokens.
    min_new_tokens: usize,
    /// Stop token IDs (for min_new_tokens suppression). Set per-request.
    stop_token_ids_for_suppress: Vec<usize>,
    /// When set, suppress stop tokens until this token ID has been generated.
    /// Used to force models to close thinking blocks (</think>) before terminating.
    /// Set per-request; None = disabled.
    think_end_token_for_suppress: Option<usize>,
    /// Whether the think_end_token has been seen in the current decode.
    think_end_seen: bool,
    /// Max thinking tokens before stop-token suppression is lifted.
    /// Prevents infinite generation when models fail to close </think>.
    /// 0 = unlimited (no cap). Set per-request.
    think_suppress_budget: usize,
    /// Count of tokens generated while in thinking mode (before </think>).
    think_suppress_count: usize,
    /// Pending simple INT4 data from repack_marlin_int4_cpu, consumed by register_marlin_int4_weight.
    /// FIFO queue: repack pushes, register pops. Ensures each weight gets its matching simple format.
    pending_simple_int4: Vec<crate::weights::marlin::SimpleInt4>,
    /// Single-slot AWQ: swap entries for each attention weight. Host copies of both
    /// Marlin and simple INT4 formats, plus the fixed GPU slot addresses.
    single_slot_swaps: Vec<SingleSlotSwapEntry>,
    #[cfg(feature = "gpu-debug")]
    debug_stop_layer: usize,
    #[cfg(feature = "gpu-debug")]
    debug_capture_layers: bool,
    #[cfg(feature = "gpu-debug")]
    debug_layer_captures: Vec<Vec<u16>>,
    /// Pre-allocated Rust prefill engine.
    /// Created early (before HCS) to claim scratch VRAM while it's still available.
    prefill_engine_slot: Option<crate::gpu_prefill::PrefillEngine>,
    /// Cached scratch VRAM computation: (fixed_bytes, per_token_bytes).
    /// Stored when the prefill engine is created so it's available for eviction checks
    /// even after the engine is taken by the RustServer.
    prefill_scratch_info: Option<(usize, usize)>,
}

#[pymethods]
impl GpuDecodeStore {
    pub fn max_safe_prefill_chunk_tokens(&self, prompt_tokens: usize, free_mb: u64) -> Option<usize> {
        self.vram_calibration
            .map(|cal| cal.max_safe_prefill_tokens(free_mb, prompt_tokens))
    }

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

        let spec_stream = unsafe {
            let mut stream: cuda_sys::CUstream = std::ptr::null_mut();
            let err = cuda_sys::lib().cuStreamCreate(
                &mut stream,
                cuda_sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Failed to create spec stream: {:?}", err)));
            }
            stream
        };

        // Create a dedicated cuBLAS handle for speculative routing on spec_stream.
        let spec_blas_handle = cublas_result::create_handle()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to create spec cuBLAS handle: {:?}", e)))?;
        unsafe {
            cublas_result::set_stream(spec_blas_handle, spec_stream as cublas_sys::cudaStream_t)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Failed to set spec cuBLAS stream: {:?}", e)))?;
        }

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
                        // Max tokens depends on head_dim (Q preload takes head_dim*4 bytes)
                        // but head_dim not known here; log raw limit, actual threshold at dispatch
                        log::info!("GQA attention: opt-in shared memory = {} KB",
                                   gqa_smem_limit / 1024);
                    } else {
                        log::warn!("GQA attention: failed to set extended shared memory ({} bytes), result={:?}",
                                   max_smem_i32, result);
                    }
                } else {
                    log::info!("GQA attention: device max shared memory = {} KB (no opt-in needed)", max_smem_i32 / 1024);
                }
            }

            // Also opt-in for gated_delta_net_step which needs ~65KB shared memory
            if let Some(delta_func) = device.get_func(MODULE_NAME, "gated_delta_net_step") {
                let struct_ptr = &delta_func as *const _ as *const u8;
                let word0: cuda_sys::CUfunction = unsafe {
                    std::ptr::read(struct_ptr as *const cuda_sys::CUfunction)
                };
                let word1: cuda_sys::CUfunction = unsafe {
                    std::ptr::read(struct_ptr.add(8) as *const cuda_sys::CUfunction)
                };
                let mut dummy = 0i32;
                let w0_valid = unsafe {
                    cuda_sys::lib().cuFuncGetAttribute(
                        &mut dummy,
                        cuda_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS,
                        word0,
                    ) == cuda_sys::CUresult::CUDA_SUCCESS
                };
                let raw_fn = if w0_valid { word0 } else { word1 };
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
                        log::info!("gated_delta_net_step: opt-in shared memory = {} KB", max_smem_i32 / 1024);
                    } else {
                        log::warn!("gated_delta_net_step: failed to set extended shared memory, result={:?}", result);
                    }
                }
            }
        }

        #[cfg(not(has_decode_kernels))]
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Decode kernels not available (nvcc not found at build time). \
             GPU decode requires compiled CUDA kernels. Rebuild with nvcc in PATH."
        ));

        let kernels_loaded = cfg!(has_decode_kernels);

        log::info!("GpuDecodeStore: initialized on device {} with compute + copy + prefetch + spec streams",
                   device_ordinal);

        Ok(GpuDecodeStore {
            device,
            blas,
            compute_stream: CudaStream(compute_stream),
            copy_stream: CudaStream(copy_stream),
            prefetch_stream: CudaStream(prefetch_stream),
            spec_stream: CudaStream(spec_stream),
            spec_blas_handle: CublasHandle(spec_blas_handle),
            raw_sigmoid_topk: CudaFunc(std::ptr::null_mut()),
            raw_softmax_topk: CudaFunc(std::ptr::null_mut()),
            graph: None,
            kernels_loaded,
            perm_inv_weight_int4: 0,
            perm_inv_weight_int8: 0,
            perm_inv_scale: 0,
            v2_partial_ptr: 0,
            v2_num_sms: 0,
            gqa_max_smem_bytes: gqa_smem_limit,
            last_decode_elapsed: 0.0,
            draft: None,
            draft_k: 3,
            draft_context_window: 512,
            spec_jaccard_threshold: 0.15,
            last_min_free_vram_mb: 0,
            last_soft_evict_experts: 0,
            last_soft_evict_freed_mb: 0.0,
            last_soft_reload_queued: 0,
            last_soft_reload_alloc_mb: 0.0,
            last_soft_reload_activated: 0,
            vram_calibration: None,
            suppress_tokens: Vec::new(),
            min_new_tokens: 0,
            stop_token_ids_for_suppress: Vec::new(),
            think_end_token_for_suppress: None,
            think_end_seen: false,
            think_suppress_budget: 0,
            think_suppress_count: 0,
            pending_simple_int4: Vec::new(),
            single_slot_swaps: Vec::new(),
            #[cfg(feature = "gpu-debug")]
            debug_stop_layer: 0,
            #[cfg(feature = "gpu-debug")]
            debug_capture_layers: false,
            #[cfg(feature = "gpu-debug")]
            debug_layer_captures: Vec::new(),
            prefill_engine_slot: None,
            prefill_scratch_info: None,
        })
    }

    /// Pre-allocate the Rust prefill engine.
    /// Must be called BEFORE HCS pool loading so scratch buffers claim VRAM
    /// while it's still available. The engine is stored internally and
    /// taken by the Rust server via take_prefill_engine().
    #[pyo3(signature = (max_context_tokens))]
    fn allocate_prefill_engine(&mut self, max_context_tokens: usize) -> PyResult<()> {
        match self.create_prefill_engine(max_context_tokens) {
            Ok(engine) => {
                log::info!("Prefill engine pre-allocated (max_tokens={})", max_context_tokens);
                // Cache scratch VRAM computation for eviction checks (survives take_prefill_engine)
                self.prefill_scratch_info = Some(crate::gpu_prefill::compute_scratch_vram(&engine.config));
                self.prefill_engine_slot = Some(engine);
                Ok(())
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to allocate prefill engine: {}", e)
            ))
        }
    }

    /// Return the max VRAM (MB) that prefill scratch will need for a given prompt size.
    /// Used by the VRAM budget calculator to reserve space for dynamic scratch growth.
    #[pyo3(signature = (max_tokens))]
    fn prefill_scratch_reservation_mb(&self, max_tokens: usize) -> PyResult<usize> {
        let engine = self.prefill_engine_slot.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "Prefill engine not allocated yet"))?;
        let (fixed_bytes, per_token_bytes) = crate::gpu_prefill::compute_scratch_vram(&engine.config);
        let total_bytes = fixed_bytes + per_token_bytes * max_tokens;
        Ok(total_bytes / (1024 * 1024))
    }

    /// Run Rust prefill directly from token IDs and prepare the decode store
    /// to continue generation from the returned first token.
    #[pyo3(signature = (token_ids, temperature=0.6))]
    fn rust_prefill_tokens(
        &mut self,
        token_ids: Vec<u32>,
        temperature: f32,
    ) -> PyResult<(usize, usize, bool)> {
        let prompt_len = token_ids.len();
        let prefill_debug = prefill_debug_enabled();
        let (cache_fast_snapshot, ne) = {
            let (cache_fast, ne) = self.export_hcs_snapshot();
            (cache_fast.to_vec(), ne)
        };
        self.swap_to_marlin_rust().map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        let prefill_hcs_guard_store_addr = self as *mut Self as usize;
        let (first_token, prompt_len, kv_overflow) = {
            let engine = self.prefill_engine_slot.as_mut().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "Rust prefill engine not allocated. Call allocate_prefill_engine() first."
                )
            })?;

            let kv_overflow = prompt_len > engine.kv_max_seq;
            engine.update_hcs_snapshot(&cache_fast_snapshot, ne);
            engine.set_prefill_hcs_guard_store_addr(prefill_hcs_guard_store_addr);

            if let Err(e) = engine.prepare_for_prefill(prompt_len) {
                engine.clear_prefill_hcs_guard_store_addr();
                return Err(pyo3::exceptions::PyRuntimeError::new_err(e));
            }

            let prefill_result = match engine.run_prefill(&token_ids, temperature, &[]) {
                Ok(r) => r,
                Err(e) => {
                    engine.clear_prefill_hcs_guard_store_addr();
                    let _ = engine.release_scratch();
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(e));
                }
            };
            if prefill_debug {
                eprintln!(
                    "[PREFILL-DEBUG] bridge prompt_tokens={} prefill_ms={:.1} chunk_size={} scratch_tokens={}",
                    prompt_len,
                    prefill_result.prefill_time_ms,
                    engine.config.prefill_chunk_size,
                    engine.scratch.max_tokens,
                );
            }

            if let Err(e) = engine.release_scratch() {
                log::error!("rust_prefill_tokens: failed to release scratch: {}", e);
            }
            engine.clear_prefill_hcs_guard_store_addr();

            (
                prefill_result.first_token as usize,
                prefill_result.prompt_len,
                kv_overflow,
            )
        };

        self.set_kv_position_rust(prompt_len);
        self.swap_to_simple_int4_rust()
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        Ok((first_token, prompt_len, kv_overflow))
    }

    /// Initialize the decode graph with model dimensions.
    #[pyo3(signature = (hidden_size, num_layers, vocab_size, eps, max_experts_per_tok=10, max_intermediate_size=0, max_qkv_size=0, group_size=128, expert_bits=4, moe_intermediate_size=0))]
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
        expert_bits: u8,
        moe_intermediate_size: usize,
    ) -> PyResult<()> {
        let intermediate = if max_intermediate_size > 0 { max_intermediate_size } else { hidden_size * 4 };
        let moe_inter = if moe_intermediate_size > 0 { moe_intermediate_size } else { intermediate };
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

        // v2 K-split partial sum buffer: max_k_splits=8, max_N = max(2*intermediate, hidden_size, qkv_size)
        // qkv_size included so attention projections can also use v2 GEMV
        let max_n_v2 = (intermediate * 2).max(hidden_size).max(qkv_size);
        let max_k_splits = 8;
        let d_v2_partial = self.device.alloc_zeros::<f32>(max_k_splits * max_n_v2)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        // Batched expert buffers: per-expert scratch for batched kernel launches
        let max_ept = max_experts_per_tok;
        let d_batch_gate_ups = self.device.alloc_zeros::<u16>(max_ept * intermediate * 2)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_batch_partials = self.device.alloc_zeros::<f32>(max_ept * max_k_splits * max_n_v2)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_batch_expert_outs = self.device.alloc_zeros::<u16>(max_ept * hidden_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_batch_w13_packed_ptrs = self.device.alloc_zeros::<u64>(max_ept)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_batch_w13_scales_ptrs = self.device.alloc_zeros::<u64>(max_ept)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_batch_w2_packed_ptrs = self.device.alloc_zeros::<u64>(max_ept)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_batch_w2_scales_ptrs = self.device.alloc_zeros::<u64>(max_ept)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_batch_weights = self.device.alloc_zeros::<f32>(max_ept)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        // Contiguous upload buffer: [4*max_ept*8 + max_ept*4] bytes → single H2D copy per layer
        let batch_upload_ptrs_bytes = 4 * max_ept * 8; // 4 pointer arrays x max_ept x 8 bytes
        let batch_upload_total_bytes = batch_upload_ptrs_bytes + max_ept * 4; // + weights (f32)
        let d_batch_upload = self.device.alloc_zeros::<u8>(batch_upload_total_bytes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let h_batch_upload = vec![0u8; batch_upload_total_bytes];
        log::info!("GpuDecodeStore: batched expert buffers allocated ({:.1} KB gate_ups + {:.1} KB partials + {:.1} KB outs)",
            (max_ept * intermediate * 2 * 2) as f64 / 1024.0,
            (max_ept * max_k_splits * max_n_v2 * 4) as f64 / 1024.0,
            (max_ept * hidden_size * 2) as f64 / 1024.0);

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

        // Compute and upload inverse Marlin permutation tables (INT4 + INT8)
        let (d_inv_weight_perm, d_inv_scale_perm, d_inv_weight_perm_int8) = Self::upload_marlin_perm_tables(&self.device)?;

        // Cache perm table pointers on self so Marlin attention GEMV launchers
        // can access them even when graph is .take()d during decode_step.
        self.perm_inv_weight_int4 = *d_inv_weight_perm.device_ptr();
        self.perm_inv_weight_int8 = *d_inv_weight_perm_int8.device_ptr();
        self.perm_inv_scale = *d_inv_scale_perm.device_ptr();

        let d_gqa_q = self.device.alloc_zeros::<f32>(qkv_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_gqa_k = self.device.alloc_zeros::<f32>(qkv_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_gqa_v = self.device.alloc_zeros::<f32>(qkv_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_gqa_out = self.device.alloc_zeros::<f32>(qkv_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;

        // MLA buffers: sized to hold max possible MLA dimensions
        // These are reused across all MLA layers (allocated once)
        let mla_buf_size = qkv_size; // qkv_size already accounts for MLA dimension needs
        let d_mla_q_absorbed = self.device.alloc_zeros::<f32>(mla_buf_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_mla_kv = self.device.alloc_zeros::<f32>(mla_buf_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
        let d_mla_attn_out = self.device.alloc_zeros::<f32>(mla_buf_size)
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
            decode_layer_start: 0,
            decode_layer_end: num_layers,
            vocab_size,
            eps,
            intermediate_size: intermediate,
            moe_intermediate_size: moe_inter,
            group_size,
            expert_bits,
            shared_expert_bits: expert_bits, // default: same as routed; overridden when shared expert is registered
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
            d_inv_weight_perm_int8,
            d_v2_partial,
            num_sms,
            d_batch_gate_ups,
            d_batch_partials,
            d_batch_expert_outs,
            d_batch_w13_packed_ptrs,
            d_batch_w13_scales_ptrs,
            d_batch_w2_packed_ptrs,
            d_batch_w2_scales_ptrs,
            d_batch_weights,
            h_batch_w13_packed_ptrs: vec![0u64; max_ept],
            h_batch_w13_scales_ptrs: vec![0u64; max_ept],
            h_batch_w2_packed_ptrs: vec![0u64; max_ept],
            h_batch_w2_scales_ptrs: vec![0u64; max_ept],
            h_batch_weights: vec![0.0f32; max_ept],
            max_experts_per_tok: max_ept,
            d_batch_upload,
            h_batch_upload,
            batch_upload_ptrs_bytes,
            batch_upload_total_bytes,
            d_gqa_q,
            d_gqa_k,
            d_gqa_v,
            d_gqa_out,
            d_mla_q_absorbed,
            d_mla_kv,
            d_mla_attn_out,
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
            d_mamba2_conv_out: None,
            mamba2_conv_out_size: 0,
            mamba2_n_groups: 1,
            mamba2_conv_bias_ptrs: std::collections::HashMap::new(),
            moe_input_override_ptr: 0,
            h_topk_ids: vec![0i32; max_experts_per_tok],
            h_topk_weights: vec![0.0f32; max_experts_per_tok],
            h_logits: vec![0.0f32; vocab_size],
            pinned_topk_ids: PinnedMapped::new(max_experts_per_tok * 4).ok(),
            pinned_topk_weights: PinnedMapped::new(max_experts_per_tok * 4).ok(),
            // GPU-side route sync: [cold_count, ready_flag, cold_ids[topk], cold_slots[topk]]
            mapped_cold_buf: PinnedMapped::new((2 + max_experts_per_tok * 2) * 4).ok(),
            gpu_route_sync: false,
            mapped_reads_active: false,
            mapped_activations: None,
            kernels: None,
            pre_events: None,
            norm_bias_one: false,
            kv_k_ptrs: Vec::new(),
            kv_v_ptrs: Vec::new(),
            kv_max_seq: 0,
            kv_current_pos: 0,
            kv_format: 1, // default FP8
            kv_k_radius_ptrs: Vec::new(),
            kv_v_radius_ptrs: Vec::new(),
            kv_k_angles_ptrs: Vec::new(),
            kv_v_angles_ptrs: Vec::new(),
            kv_num_blocks: 0,
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
            t_moe_gate_gemv: 0.0,
            t_moe_d2h_topk: 0.0,
            t_moe_apfl: 0.0,
            t_moe_padding_setup: 0.0,
            t_moe_d2d_copy: 0.0,
            t_moe_accum: 0.0,
            t_attn_la: 0.0,
            t_attn_gqa: 0.0,
            t_la_proj: 0.0,
            t_la_conv: 0.0,
            t_la_recur: 0.0,
            t_la_out: 0.0,
            t_gqa_proj: 0.0,
            t_gqa_attn: 0.0,
            t_gqa_out: 0.0,
            t_expert_w13: 0.0,
            t_expert_silu_w2: 0.0,
            t_dma_expert_wait: 0.0,
            t_dma_expert_compute: 0.0,
            dma_bytes_total: 0,
            dma_call_count: 0,
            dma_cold_experts: 0,
            dma_hcs_experts: 0,
            validation_decode_steps: 0,
            validation_per_layer_steps: 0,
            validation_ungraphed_steps: 0,
            validation_decode_start_num_cached: 0,
            validation_decode_start_soft_num_cached: 0,
            validation_decode_start_hard_num_cached: 0,
            validation_decode_start_dupes: 0,
            validation_decode_start_soft_dupes: 0,
            validation_decode_start_hash: String::new(),
            validation_decode_start_resident: Vec::new(),
            validation_decode_start_hcs_file: String::new(),
            validation_decode_start_slots: Vec::new(),
            validation_decode_start_slots_file: String::new(),
            validation_decode_cold_hist: std::collections::BTreeMap::new(),
            validation_decode_cold_events: Vec::new(),
            validation_decode_cold_file: String::new(),
            validation_decode_cold_events_file: String::new(),
            batch_max: 0,
            d_batch_hidden: None,
            d_batch_residual: None,
            d_batch_moe_out: None,
            d_batch_logits: None,
            h_batch_logits: Vec::new(),
            h_batch_topk_ids: Vec::new(),
            h_batch_topk_weights: Vec::new(),
            d_batch_topk_ids: None,
            d_batch_topk_wts: None,
            d_batch_gate_logits: None,
            d_batch_proj_a: None,
            d_batch_proj_b: None,
            d_batch_attn_out: None,
            batch_max_proj_dim: 0,
            batch_max_attn_out_dim: 0,
            segment_layer_start: 0,
            segment_layer_end: 0,
            segment_skip_embedding: false,
            segment_skip_final: false,
            segment_gqa_cache_offset: 0,
            la_backup: Vec::new(),
            d_la_hidden_stack: None,
            // CUDA graph state
            d_graph_token_id: None,
            d_graph_pos: None,
            d_graph_seq_len: None,
            d_dummy_expert: None,
            d_dummy_ptrs: None,
            h_dummy_ptrs: [0u64; 4],
            per_layer_graphs: Vec::new(),
            per_layer_graphs_valid: false,
            graphs_ever_captured: false,
            captured_la_ptrs: Vec::new(),
            captured_kv_ptrs: Vec::new(),
            captured_mla_ptrs: Vec::new(),
            captured_polar4_ptrs: Vec::new(),
            per_layer_moe_indices: Vec::new(),
            d_cublas_workspace: None,
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
                fused_silu_accum_int8: get("marlin_gemv_int8_fused_silu_accum")?,
                marlin_gemv_int4_v2: get("marlin_gemv_int4_v2")?,
                reduce_ksplits_bf16: get("reduce_ksplits_bf16")?,
                fused_silu_accum_v2: get("marlin_gemv_int4_fused_silu_accum_v2")?,
                fused_silu_accum_v2_int8: get("marlin_gemv_int8_fused_silu_accum_v2")?,
                reduce_ksplits_weighted_accum_bf16: get("reduce_ksplits_weighted_accum_bf16")?,
                // Attention kernels (LA + GQA)
                uninterleave_qkvz: get("uninterleave_qkvz")?,
                la_conv1d: get("la_conv1d")?,
                compute_gate_beta: get("compute_gate_beta")?,
                repeat_interleave_heads: get("repeat_interleave_heads")?,
                l2norm_scale_per_head: get("l2norm_scale_per_head")?,
                gated_delta_net_step: get("gated_delta_net_step")?,
                gated_rmsnorm_silu: get("gated_rmsnorm_silu")?,
                split_gated_q: get("split_gated_q")?,
                per_head_rmsnorm: get("per_head_rmsnorm")?,
                apply_rope: get("apply_rope")?,
                kv_cache_write: get("kv_cache_write")?,
                gqa_attention: get("gqa_attention")?,
                gqa_attention_tiled: get("gqa_attention_tiled")?,
                gqa_attention_reduce: get("gqa_attention_reduce")?,
                apply_gated_attn: get("apply_gated_attn")?,
                // Fused v2 kernels (inline atomic reduction)
                marlin_gemv_int4_v2_fused_f32: get("marlin_gemv_int4_v2_fused_f32")?,
                marlin_gemv_int8_v2: get("marlin_gemv_int8_v2")?,
                marlin_gemv_int8_v2_fused_f32: get("marlin_gemv_int8_v2_fused_f32")?,
                marlin_gemv_int4_v2_batched: get("marlin_gemv_int4_v2_batched")?,
                marlin_gemv_int8_v2_batched: get("marlin_gemv_int8_v2_batched")?,
                reduce_ksplits_bf16_batched: get("reduce_ksplits_bf16_batched")?,
                fused_silu_w2_batched: get("fused_silu_w2_batched")?,
                fused_silu_w2_int8_batched: get("fused_silu_w2_int8_batched")?,
                multi_expert_weighted_add_bf16: get("multi_expert_weighted_add_bf16")?,
                // Graphable variants
                embedding_lookup_g: get("embedding_lookup_g")?,
                apply_rope_g: get("apply_rope_g")?,
                kv_cache_write_g: get("kv_cache_write_g")?,
                gqa_attention_g: get("gqa_attention_g")?,
                // BF16-output variants
                gated_rmsnorm_silu_bf16: get("gated_rmsnorm_silu_bf16")?,
                gqa_attention_g_bf16: get("gqa_attention_g_bf16")?,
                apply_gated_attn_bf16: get("apply_gated_attn_bf16")?,
                gqa_attention_tiled_g: get("gqa_attention_tiled_g")?,
                gqa_attention_reduce_g: get("gqa_attention_reduce_g")?,
                la_fused_post_proj: get("la_fused_post_proj")?,
                expert_classify_prepare: get("expert_classify_prepare")?,
                // MLA kernels
                mla_kv_cache_write_g: get("mla_kv_cache_write_g")?,
                mla_kv_cache_write: get("mla_kv_cache_write")?,
                mla_attention_g: get("mla_attention_g")?,
                mla_attention: get("mla_attention")?,
                mla_deinterleave: get("mla_deinterleave")?,
                mla_split_q: get("mla_split_q")?,
                mla_absorb_wkc: get("mla_absorb_wkc")?,
                mla_apply_wvc: get("mla_apply_wvc")?,
                // 4-bit PolarQuant kernels
                kv_cache_write_polar4: get("kv_cache_write_polar4")?,
                gqa_attention_polar4: get("gqa_attention_polar4")?,
                kv_cache_write_polar4_g: get("kv_cache_write_polar4_g")?,
                gqa_attention_polar4_g: get("gqa_attention_polar4_g")?,
                gqa_attention_polar4_tiled_g: get("gqa_attention_polar4_tiled_g")?,
                gqa_attention_polar4_reduce_g: get("gqa_attention_polar4_reduce_g")?,
                // Mamba2 SSM kernels
                mamba2_conv1d: get("mamba2_conv1d")?,
                mamba2_ssm_step: get("mamba2_ssm_step")?,
                mamba2_discretize: get("mamba2_discretize")?,
                mamba2_gate_output: get("mamba2_gate_output")?,
                // relu2 expert activation
                relu2_w2_batched: get("relu2_w2_batched")?,
                relu2_w2_int8_batched: get("relu2_w2_int8_batched")?,
            };
            // Extract raw CUfunction handles for spec routing on spec_stream.
            self.raw_sigmoid_topk = CudaFunc(extract_cu_function(&kernels.sigmoid_topk));
            self.raw_softmax_topk = CudaFunc(extract_cu_function(&kernels.softmax_topk));
            self.graph.as_mut().unwrap().kernels = Some(kernels);
            log::info!("GpuDecodeStore: cached kernel function handles");
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

        // Cache v2 partial buffer pointer on self for attention Marlin GEMV v2
        {
            let g = self.graph.as_ref().unwrap();
            self.v2_partial_ptr = *g.d_v2_partial.device_ptr();
            self.v2_num_sms = g.num_sms;
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
        graph.weights.push(GpuWeight::new(ptr as u64, rows, cols, dtype));
        Ok(id)
    }

    /// Register a Marlin INT8 weight (packed + scales already on GPU).
    /// Returns the weight ID.
    fn register_marlin_int8_weight(
        &mut self, packed_ptr: usize, scales_ptr: usize,
        rows: usize, cols: usize, group_size: usize,
    ) -> PyResult<usize> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let id = graph.weights.len();
        graph.weights.push(GpuWeight::new_marlin_int8(packed_ptr as u64, scales_ptr as u64, rows, cols, group_size));
        Ok(id)
    }

    /// Register a Marlin INT4 weight (packed + scales already on GPU).
    /// Returns the weight ID. Used when Python does the quantization/repacking
    /// so both prefill (gptq_marlin_gemm) and decode (Rust GEMV) share the same data.
    ///
    /// Single-slot AWQ: the GPU addresses from PyTorch become the permanent "slot".
    /// Both Marlin and simple INT4 data are DMA'd into the same addresses on swap.
    /// Marlin data is copied from GPU to host during registration (for later restore).
    /// Simple INT4 host data is stashed from repack_marlin_int4_cpu.
    /// simple_packed_ptr and simple_scales_f32_ptr are set to the SAME addresses as
    /// the Marlin pointers — CUDA graphs see stable pointers across all requests.
    fn register_marlin_int4_weight(
        &mut self, packed_ptr: usize, scales_ptr: usize,
        rows: usize, cols: usize, group_size: usize,
    ) -> PyResult<usize> {
        // Pop pending simple INT4 (if available from repack_marlin_int4_cpu)
        let simple = if !self.pending_simple_int4.is_empty() {
            Some(self.pending_simple_int4.remove(0))
        } else {
            None
        };

        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let id = graph.weights.len();

        let packed_gpu = packed_ptr as u64;
        let scales_gpu = scales_ptr as u64;

        // Create weight with BOTH Marlin and simple ptrs pointing to the same slot
        let mut w = GpuWeight::new_marlin_int4(packed_gpu, scales_gpu, rows, cols, group_size);
        // Single-slot: simple INT4 decode reads from the same GPU addresses
        w.simple_packed_ptr = packed_gpu;
        w.simple_scales_f32_ptr = scales_gpu;

        if let Some(s) = simple {
            let packed_bytes = s.packed.len();
            let marlin_scales_bytes = rows * (cols / group_size) * 2; // BF16

            // Copy Marlin data from GPU to host (for later restore after decode)
            let mut marlin_packed_host = vec![0u8; packed_bytes];
            let mut marlin_scales_host = vec![0u8; marlin_scales_bytes];
            unsafe {
                let r = cuda_sys::lib().cuMemcpyDtoH_v2(
                    marlin_packed_host.as_mut_ptr() as *mut std::ffi::c_void,
                    packed_gpu, packed_bytes);
                if r != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("cuMemcpyDtoH Marlin packed: {:?}", r)));
                }
                let r = cuda_sys::lib().cuMemcpyDtoH_v2(
                    marlin_scales_host.as_mut_ptr() as *mut std::ffi::c_void,
                    scales_gpu, marlin_scales_bytes);
                if r != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("cuMemcpyDtoH Marlin scales: {:?}", r)));
                }
            }

            // Convert simple INT4 FP32 scales to raw bytes
            let simple_scales_host: Vec<u8> = s.scales.iter()
                .flat_map(|f| f.to_bits().to_ne_bytes())
                .collect();

            let packed_mb = packed_bytes as f64 / 1024.0 / 1024.0;
            log::info!(
                "  + single-slot AWQ: packed={:.1} MB, Marlin scales={} B, simple scales={} B",
                packed_mb, marlin_scales_bytes, simple_scales_host.len(),
            );

            self.single_slot_swaps.push(SingleSlotSwapEntry {
                weight_id: id,
                packed_slot_ptr: packed_gpu,
                scales_slot_ptr: scales_gpu,
                marlin_packed_host,
                marlin_scales_host,
                simple_packed_host: s.packed,
                simple_scales_host,
            });
        }

        graph.weights.push(w);
        Ok(id)
    }

    /// Single-slot AWQ: DMA simple INT4 data into the GPU slots for decode.
    /// Overwrites Marlin data at the same addresses. Pointers never change,
    /// so CUDA graphs remain valid without recapture.
    fn swap_to_simple_int4(&mut self) -> PyResult<()> {
        if self.single_slot_swaps.is_empty() {
            return Ok(());
        }
        let mut total_bytes: usize = 0;
        for entry in &self.single_slot_swaps {
            unsafe {
                let r = cuda_sys::lib().cuMemcpyHtoD_v2(
                    entry.packed_slot_ptr,
                    entry.simple_packed_host.as_ptr() as *const std::ffi::c_void,
                    entry.simple_packed_host.len());
                if r != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("swap_to_simple_int4 packed: {:?}", r)));
                }
                let r = cuda_sys::lib().cuMemcpyHtoD_v2(
                    entry.scales_slot_ptr,
                    entry.simple_scales_host.as_ptr() as *const std::ffi::c_void,
                    entry.simple_scales_host.len());
                if r != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("swap_to_simple_int4 scales: {:?}", r)));
                }
            }
            total_bytes += entry.simple_packed_host.len() + entry.simple_scales_host.len();
        }
        log::info!("Single-slot AWQ: swapped to simple INT4 ({:.1} MB DMA)",
                   total_bytes as f64 / 1024.0 / 1024.0);
        Ok(())
    }

    /// Single-slot AWQ: DMA Marlin data back into the GPU slots for prefill.
    /// Restores original Marlin-permuted data at the same addresses.
    fn swap_to_marlin(&mut self) -> PyResult<()> {
        if self.single_slot_swaps.is_empty() {
            return Ok(());
        }
        let mut total_bytes: usize = 0;
        for entry in &self.single_slot_swaps {
            unsafe {
                let r = cuda_sys::lib().cuMemcpyHtoD_v2(
                    entry.packed_slot_ptr,
                    entry.marlin_packed_host.as_ptr() as *const std::ffi::c_void,
                    entry.marlin_packed_host.len());
                if r != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("swap_to_marlin packed: {:?}", r)));
                }
                let r = cuda_sys::lib().cuMemcpyHtoD_v2(
                    entry.scales_slot_ptr,
                    entry.marlin_scales_host.as_ptr() as *const std::ffi::c_void,
                    entry.marlin_scales_host.len());
                if r != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("swap_to_marlin scales: {:?}", r)));
                }
            }
            total_bytes += entry.marlin_packed_host.len() + entry.marlin_scales_host.len();
        }
        log::info!("Single-slot AWQ: swapped to Marlin ({:.1} MB DMA)",
                   total_bytes as f64 / 1024.0 / 1024.0);
        Ok(())
    }

    /// Register a simple INT4 weight directly (no Marlin, no single-slot swap).
    /// Used for aux GPU stores that never do prefill — they only need decode format.
    /// Pops pending simple INT4 from repack_marlin_int4_cpu, allocates GPU memory,
    /// uploads packed data and FP32 scales. CUDA graphs see stable pointers.
    fn register_simple_int4_only(
        &mut self, rows: usize, cols: usize, group_size: usize,
    ) -> PyResult<usize> {
        let simple = if !self.pending_simple_int4.is_empty() {
            self.pending_simple_int4.remove(0)
        } else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "register_simple_int4_only: no pending simple INT4 data (call repack_marlin_int4_cpu first)"));
        };

        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let id = graph.weights.len();

        // Allocate GPU memory for packed data and FP32 scales
        let packed_bytes = simple.packed.len();
        let scales_bytes = simple.scales.len() * 4; // f32 = 4 bytes each
        let packed_gpu: u64;
        let scales_gpu: u64;
        unsafe {
            let mut p: u64 = 0;
            let r = cuda_sys::lib().cuMemAlloc_v2(&mut p, packed_bytes);
            if r != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("cuMemAlloc packed: {:?}", r)));
            }
            packed_gpu = p;
            let r = cuda_sys::lib().cuMemAlloc_v2(&mut p, scales_bytes);
            if r != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("cuMemAlloc scales: {:?}", r)));
            }
            scales_gpu = p;

            // Upload simple INT4 packed data
            let r = cuda_sys::lib().cuMemcpyHtoD_v2(
                packed_gpu,
                simple.packed.as_ptr() as *const std::ffi::c_void,
                packed_bytes);
            if r != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("cuMemcpyHtoD packed: {:?}", r)));
            }

            // Upload FP32 scales
            let scales_host: Vec<u8> = simple.scales.iter()
                .flat_map(|f| f.to_bits().to_ne_bytes())
                .collect();
            let r = cuda_sys::lib().cuMemcpyHtoD_v2(
                scales_gpu,
                scales_host.as_ptr() as *const std::ffi::c_void,
                scales_bytes);
            if r != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("cuMemcpyHtoD scales: {:?}", r)));
            }
        }

        // Create weight with simple INT4 ptrs only — dtype=5 (Marlin INT4) so
        // GEMV dispatch enters the is_marlin_int4 branch, then has_simple_int4
        // returns true and uses the fast simple kernel.
        let mut w = GpuWeight::new_marlin_int4(packed_gpu, scales_gpu, rows, cols, group_size);
        w.simple_packed_ptr = packed_gpu;
        w.simple_scales_f32_ptr = scales_gpu;

        let packed_mb = packed_bytes as f64 / 1024.0 / 1024.0;
        log::info!("  + simple INT4 only: wid={}, {:.1} MB packed + {:.1} MB scales",
                   id, packed_mb, scales_bytes as f64 / 1024.0 / 1024.0);

        graph.weights.push(w);
        Ok(id)
    }

    /// Remove single-slot swap entries for weights NOT used by the decode segment.
    /// Called after set_decode_segment() in multi-GPU mode. Returns count of entries removed.
    /// After this, swap_to_simple_int4/swap_to_marlin only touch decode-segment weights.
    fn restrict_swaps_to_decode_segment(&mut self) -> PyResult<usize> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;

        let start = graph.decode_layer_start;
        let end = graph.decode_layer_end;

        // Collect weight IDs referenced by layers in the decode segment
        let mut decode_wids = std::collections::HashSet::new();
        for i in start..end.min(graph.layers.len()) {
            match &graph.layers[i].attn {
                GpuAttnConfig::GQA { q_proj, k_proj, v_proj, o_proj, fused_qkv, .. } => {
                    decode_wids.insert(*q_proj);
                    decode_wids.insert(*k_proj);
                    decode_wids.insert(*v_proj);
                    decode_wids.insert(*o_proj);
                    if let Some(fused) = fused_qkv {
                        decode_wids.insert(*fused);
                    }
                }
                GpuAttnConfig::LinearAttention { in_proj_qkvz, in_proj_ba, out_proj, .. } => {
                    decode_wids.insert(*in_proj_qkvz);
                    decode_wids.insert(*in_proj_ba);
                    decode_wids.insert(*out_proj);
                }
                GpuAttnConfig::MLA { kv_a_proj, o_proj, q_a_proj, q_b_proj, q_proj, .. } => {
                    decode_wids.insert(*kv_a_proj);
                    decode_wids.insert(*o_proj);
                    if let Some(w) = q_a_proj { decode_wids.insert(*w); }
                    if let Some(w) = q_b_proj { decode_wids.insert(*w); }
                    if let Some(w) = q_proj { decode_wids.insert(*w); }
                }
                GpuAttnConfig::Mamba2 { in_proj, out_proj, .. } => {
                    decode_wids.insert(*in_proj);
                    decode_wids.insert(*out_proj);
                }
            }
        }

        let before = self.single_slot_swaps.len();
        self.single_slot_swaps.retain(|entry| decode_wids.contains(&entry.weight_id));
        let removed = before - self.single_slot_swaps.len();

        if removed > 0 {
            log::info!("restrict_swaps_to_decode_segment: removed {} swap entries for layers outside [{}, {}), {} remain",
                       removed, start, end, self.single_slot_swaps.len());
        }

        Ok(removed)
    }

    /// Quantize a BF16 weight tensor to Marlin INT8 format and register it.
    /// Takes a GPU BF16 tensor, copies to CPU, quantizes to INT8, repacks to
    /// Marlin format, uploads packed + scales back to GPU, and registers.
    /// Returns the weight ID.
    ///
    /// This is the main entry point for converting attention weights to Marlin INT8.
    /// N = rows (output dim), K = cols (input dim).
    fn quantize_and_register_marlin_int8(
        &mut self, gpu_bf16_ptr: usize, rows: usize, cols: usize, group_size: usize,
    ) -> PyResult<usize> {
        use crate::weights::marlin::{quantize_int8, marlin_repack_int8};

        let n = rows;
        let k = cols;
        let total_elements = n * k;

        // Step 1: Copy BF16 data from GPU to CPU
        let bf16_data: Vec<u16> = unsafe {
            let mut buf = vec![0u16; total_elements];
            let result = cuda_sys::lib().cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                gpu_bf16_ptr as cuda_sys::CUdeviceptr,
                (total_elements * 2) as usize,
            );
            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Failed to copy BF16 from GPU: {:?}", result)));
            }
            buf
        };

        // Step 2: Quantize INT8 + Marlin repack (CPU)
        let q = quantize_int8(&bf16_data, n, k, group_size);
        let m = marlin_repack_int8(&q);

        self._finish_register_marlin_int8(m, n, k, group_size)
    }

    /// Quantize a CPU BF16 weight tensor to Marlin INT8 format and register it.
    /// Takes a CPU BF16 pointer directly — no GPU→CPU copy needed.
    /// This avoids ever putting the full BF16 weight on GPU.
    fn quantize_cpu_and_register_marlin_int8(
        &mut self, cpu_bf16_ptr: usize, rows: usize, cols: usize, group_size: usize,
    ) -> PyResult<usize> {
        use crate::weights::marlin::{quantize_int8, marlin_repack_int8};

        let n = rows;
        let k = cols;
        let total_elements = n * k;

        // Read BF16 data directly from CPU memory
        let bf16_data: Vec<u16> = unsafe {
            std::slice::from_raw_parts(cpu_bf16_ptr as *const u16, total_elements).to_vec()
        };

        let q = quantize_int8(&bf16_data, n, k, group_size);
        let m = marlin_repack_int8(&q);

        self._finish_register_marlin_int8(m, n, k, group_size)
    }

    /// Quantize a BF16 weight tensor to Marlin INT4 format and register it.
    /// Takes a GPU BF16 tensor, copies to CPU, quantizes to INT4, repacks to
    /// Marlin format, uploads packed + scales back to GPU, and registers.
    /// Returns the weight ID.
    fn quantize_and_register_marlin_int4(
        &mut self, gpu_bf16_ptr: usize, rows: usize, cols: usize, group_size: usize,
    ) -> PyResult<usize> {
        use crate::weights::marlin::{quantize_int4, marlin_repack};

        let n = rows;
        let k = cols;
        let total_elements = n * k;

        // Step 1: Copy BF16 data from GPU to CPU
        let bf16_data: Vec<u16> = unsafe {
            let mut buf = vec![0u16; total_elements];
            let result = cuda_sys::lib().cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                gpu_bf16_ptr as cuda_sys::CUdeviceptr,
                (total_elements * 2) as usize,
            );
            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Failed to copy BF16 from GPU: {:?}", result)));
            }
            buf
        };

        // Step 2: Quantize INT4 + Marlin repack (CPU)
        let q = quantize_int4(&bf16_data, n, k, group_size);
        let m = marlin_repack(&q);

        self._finish_register_marlin_int4(m, n, k, group_size)
    }

    /// Quantize a CPU BF16 weight tensor to Marlin INT4 format and register it.
    /// Takes a CPU BF16 pointer directly — no GPU→CPU copy needed.
    /// This avoids ever putting the full BF16 weight on GPU.
    fn quantize_cpu_and_register_marlin_int4(
        &mut self, cpu_bf16_ptr: usize, rows: usize, cols: usize, group_size: usize,
    ) -> PyResult<usize> {
        use crate::weights::marlin::{quantize_int4, marlin_repack};

        let n = rows;
        let k = cols;
        let total_elements = n * k;

        // Read BF16 data directly from CPU memory
        let bf16_data: Vec<u16> = unsafe {
            std::slice::from_raw_parts(cpu_bf16_ptr as *const u16, total_elements).to_vec()
        };

        let q = quantize_int4(&bf16_data, n, k, group_size);
        let m = marlin_repack(&q);

        self._finish_register_marlin_int4(m, n, k, group_size)
    }

    /// Quantize a CPU BF16 weight to Marlin INT4 format using Rust repack.
    /// Returns (packed_bytes, scales_bytes, n, k) — caller uploads to GPU and registers.
    /// This ensures decode GEMV and prefill GEMM use the same Marlin repack format.
    ///
    /// Also produces and stashes the simple INT4 format (for fast decode GEMV).
    /// The stashed data is consumed by the next register_marlin_int4_weight call.
    fn repack_marlin_int4_cpu(
        &mut self, cpu_bf16_ptr: usize, rows: usize, cols: usize, group_size: usize,
    ) -> PyResult<(pyo3::Py<pyo3::types::PyBytes>, pyo3::Py<pyo3::types::PyBytes>, usize, usize)> {
        use crate::weights::marlin::{quantize_int4, marlin_repack, simple_int4_from_quantized};

        let n = rows;
        let k = cols;
        let total_elements = n * k;

        let bf16_data: Vec<u16> = unsafe {
            std::slice::from_raw_parts(cpu_bf16_ptr as *const u16, total_elements).to_vec()
        };

        let q = quantize_int4(&bf16_data, n, k, group_size);
        let m = marlin_repack(&q);

        // Also create simple INT4 format for decode GEMV and stash it
        let simple = simple_int4_from_quantized(&q);
        self.pending_simple_int4.push(simple);

        // Convert packed u32 vec to bytes
        let packed_bytes: Vec<u8> = m.packed.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // Convert scales u16 vec to bytes
        let scales_bytes: Vec<u8> = m.scales.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        pyo3::Python::with_gil(|py| {
            let pb = pyo3::types::PyBytes::new(py, &packed_bytes);
            let sb = pyo3::types::PyBytes::new(py, &scales_bytes);
            Ok((pb.into(), sb.into(), n, k))
        })
    }

    /// Quantize a CPU BF16 weight to Marlin INT4 format without producing decode-side
    /// simple-INT4 state. Use this for synthetic/offline packing where we only need
    /// the Marlin-format bytes and must not grow pending_simple_int4.
    fn repack_marlin_int4_cpu_no_simple(
        &self, cpu_bf16_ptr: usize, rows: usize, cols: usize, group_size: usize,
    ) -> PyResult<(pyo3::Py<pyo3::types::PyBytes>, pyo3::Py<pyo3::types::PyBytes>, usize, usize)> {
        use crate::weights::marlin::{quantize_int4, marlin_repack};

        let n = rows;
        let k = cols;
        let total_elements = n * k;

        let bf16_data: Vec<u16> = unsafe {
            std::slice::from_raw_parts(cpu_bf16_ptr as *const u16, total_elements).to_vec()
        };

        let q = quantize_int4(&bf16_data, n, k, group_size);
        let m = marlin_repack(&q);

        let packed_bytes: Vec<u8> = m.packed.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let scales_bytes: Vec<u8> = m.scales.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        pyo3::Python::with_gil(|py| {
            let pb = pyo3::types::PyBytes::new(py, &packed_bytes);
            let sb = pyo3::types::PyBytes::new(py, &scales_bytes);
            Ok((pb.into(), sb.into(), n, k))
        })
    }

    /// Quantize a CPU BF16 weight to Marlin INT8 format using Rust repack.
    /// Returns (packed_bytes, scales_bytes, n, k) — caller uploads to GPU and registers.
    fn repack_marlin_int8_cpu(
        &self, cpu_bf16_ptr: usize, rows: usize, cols: usize, group_size: usize,
    ) -> PyResult<(pyo3::Py<pyo3::types::PyBytes>, pyo3::Py<pyo3::types::PyBytes>, usize, usize)> {
        use crate::weights::marlin::{quantize_int8, marlin_repack_int8};

        let n = rows;
        let k = cols;
        let total_elements = n * k;

        let bf16_data: Vec<u16> = unsafe {
            std::slice::from_raw_parts(cpu_bf16_ptr as *const u16, total_elements).to_vec()
        };

        let q = quantize_int8(&bf16_data, n, k, group_size);
        let m = marlin_repack_int8(&q);

        let packed_bytes: Vec<u8> = m.packed.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let scales_bytes: Vec<u8> = m.scales.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        pyo3::Python::with_gil(|py| {
            let pb = pyo3::types::PyBytes::new(py, &packed_bytes);
            let sb = pyo3::types::PyBytes::new(py, &scales_bytes);
            Ok((pb.into(), sb.into(), n, k))
        })
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
    /// Supports BF16, FP8, and Marlin INT8 weights.
    fn gemv_bf16(&self, weight_id: usize, input_ptr: usize, output_ptr: usize) -> PyResult<()> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let w = &graph.weights[weight_id];
        self.gemv_bf16_internal(w, input_ptr as u64, output_ptr as u64)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
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
                graph.t_moe_gate_gemv = 0.0;
                graph.t_moe_d2h_topk = 0.0;
                graph.t_moe_apfl = 0.0;
                graph.t_moe_padding_setup = 0.0;
                graph.t_moe_d2d_copy = 0.0;
                graph.t_moe_accum = 0.0;
                graph.t_attn_la = 0.0;
                graph.t_attn_gqa = 0.0;
                graph.t_la_proj = 0.0;
                graph.t_la_conv = 0.0;
                graph.t_la_recur = 0.0;
                graph.t_la_out = 0.0;
                graph.t_gqa_proj = 0.0;
                graph.t_gqa_attn = 0.0;
                graph.t_gqa_out = 0.0;
                graph.t_expert_w13 = 0.0;
                graph.t_expert_silu_w2 = 0.0;
                graph.t_dma_expert_wait = 0.0;
                graph.t_dma_expert_compute = 0.0;
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
    /// prefetch_count: fixed number of experts to prefetch per layer.
    ///   0 = disabled, N = prefetch top-N predicted experts for next layer.
    #[pyo3(signature = (num_slots=16, prefetch_count=1))]
    fn init_apfl(
        &mut self,
        num_slots: usize,
        prefetch_count: usize,
    ) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;

        if graph.expert_buf_size == 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Call resize_expert_buffers first to set expert size"));
        }

        // Each slot holds one complete expert: w13_packed + w13_scales + w2_packed + w2_scales.
        // Layout matches the GPU double-buffer (expert_buf_* offsets) so that contiguous
        // host-to-device DMA works with identical data placement.
        let slot_size = if graph.expert_buf_total_size > 0 {
            graph.expert_buf_total_size
        } else {
            // Fallback: equal-sized regions (legacy)
            let align = 512usize;
            let ebs = (graph.expert_buf_size + align - 1) & !(align - 1);
            ebs * 4
        };

        let w13p_off = graph.expert_buf_w13p_offset;
        let w13s_off = graph.expert_buf_w13s_offset;
        let w2p_off = graph.expert_buf_w2p_offset;
        let w2s_off = graph.expert_buf_w2s_offset;

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

            // Layout matches GPU double-buffer for contiguous DMA compatibility
            slots.push(PrefetchSlot {
                d_buf,
                buf_size: slot_size,
                w13_packed_offset: w13p_off,
                w13_packed_size: w13s_off - w13p_off,
                w13_scales_offset: w13s_off,
                w13_scales_size: w2p_off - w13s_off,
                w2_packed_offset: w2p_off,
                w2_packed_size: w2s_off - w2p_off,
                w2_scales_offset: w2s_off,
                w2_scales_size: slot_size - w2s_off,
                layer_idx: -1,
                expert_idx: -1,
                dma_event: CudaEvent(event),
                dma_queued: false,
            });
        }

        let num_layers = graph.moe_layers.len();
        let layer_stats: Vec<ApflLayerStats> = (0..num_layers)
            .map(|_| ApflLayerStats::new())
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
            prefetch_count,
            enabled: prefetch_count > 0,
            h_spec_topk_ids: vec![0i32; topk],
            pending_prefetch: false,
            pending_next_layer: 0,
            pending_spec_count: 0,
        });

        let total_mb = (slot_size * num_slots) as f64 / (1024.0 * 1024.0);
        log::info!(
            "APFL: initialized {} slots x {:.1} KB = {:.1} MB VRAM, prefetch_count={}",
            num_slots, slot_size as f64 / 1024.0, total_mb, prefetch_count,
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
            "APFL: enabled={}, {} slots, prefetch_count={}, total hits={}, misses={}, hit_rate={:.1}%",
            apfl.enabled, apfl.slots.len(), apfl.prefetch_count,
            apfl.total_hits, apfl.total_misses,
            if apfl.total_hits + apfl.total_misses > 0 {
                apfl.total_hits as f64 / (apfl.total_hits + apfl.total_misses) as f64 * 100.0
            } else { 0.0 },
        ));

        for (i, stats) in apfl.layer_stats.iter().enumerate() {
            if stats.hits + stats.misses > 0 {
                lines.push(format!(
                    "  Layer {}: hits={}, misses={}, hit_rate={:.1}%",
                    i, stats.hits, stats.misses,
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

    /// Configure Nemotron-specific MoE fields: relu2 activation, ungated experts, latent projections.
    #[pyo3(signature = (layer_idx, activation_type, gated_experts, latent_down_wid, latent_up_wid, moe_input_size))]
    fn set_moe_nemotron_config(
        &mut self,
        layer_idx: usize,
        activation_type: u8,    // 0=silu_gated, 1=relu2
        gated_experts: bool,    // false for Nemotron (up_proj only, no gate_proj)
        latent_down_wid: Option<usize>,  // hidden->latent projection
        latent_up_wid: Option<usize>,    // latent->hidden projection
        moe_input_size: usize,  // expert I/O dimension (latent_size), 0=use hidden_size
    ) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let moe = graph.moe_layers.get_mut(layer_idx)
            .and_then(|m| m.as_mut())
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                format!("MoE layer {} not registered", layer_idx)))?;
        moe.activation_type = activation_type;
        moe.gated_experts = gated_experts;
        moe.latent_down_wid = latent_down_wid;
        moe.latent_up_wid = latent_up_wid;
        moe.moe_input_size = moe_input_size;
        log::info!("Set Nemotron MoE config for layer {}: activation={}, gated={}, latent_down={:?}, latent_up={:?}, input_size={}",
                   layer_idx, activation_type, gated_experts, latent_down_wid, latent_up_wid, moe_input_size);
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
            graph.moe_intermediate_size,
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
        let intermediate = graph.moe_intermediate_size;
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
        let size = graph.moe_intermediate_size * 2;
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
    #[pyo3(signature = (model_dir, num_tokens=10, prefetch_count=1, num_slots=16))]
    fn test_apfl_e2e_py(
        &mut self,
        model_dir: &str,
        num_tokens: usize,
        prefetch_count: usize,
        num_slots: usize,
    ) -> PyResult<String> {
        self.test_apfl_e2e(model_dir, num_tokens, prefetch_count, num_slots)
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

    /// Initialize a lightweight HCS state for heatmap collection only.
    /// No VRAM is allocated for expert caching — only the flat heatmap array.
    /// Call hcs_start_collecting() after this, run inference, then hcs_export_heatmap().
    fn hcs_init_collection(&mut self, num_layers: usize, num_experts_per_layer: usize) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let mut hcs = HcsState::new();
        hcs.num_experts_per_layer = num_experts_per_layer;
        let total = num_layers * num_experts_per_layer;
        hcs.cache_fast = vec![[0u64; 4]; total];
        hcs.cache_fast_num_layers = num_layers;
        hcs.heatmap_flat = vec![0u64; total];
        graph.hcs = Some(hcs);
        log::info!("HCS collection-only init: {} layers × {} experts = {} slots",
            num_layers, num_experts_per_layer, total);
        Ok(())
    }

    /// Start collecting activation heatmap data for HCS.
    fn hcs_start_collecting(&mut self) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let hcs = graph.hcs.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call init_hcs first"))?;
        hcs.collecting = true;
        hcs.heatmap.clear();
        // Clear flat heatmap too
        for v in hcs.heatmap_flat.iter_mut() { *v = 0; }
        Ok(())
    }

    /// Export the collected heatmap as a Python dict {"layer,expert": count}.
    fn hcs_export_heatmap(&self, py: pyo3::Python<'_>) -> PyResult<PyObject> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let hcs = graph.hcs.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No HCS state"))?;
        let nep = hcs.num_experts_per_layer;
        if nep == 0 || hcs.heatmap_flat.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Heatmap is empty"));
        }
        let dict = pyo3::types::PyDict::new(py);
        for (idx, &count) in hcs.heatmap_flat.iter().enumerate() {
            if count > 0 {
                let layer = idx / nep;
                let expert = idx % nep;
                dict.set_item(format!("{},{}", layer, expert), count)?;
            }
        }
        Ok(dict.into())
    }

    /// Destroy HCS state so it can be re-initialized with a real budget.
    fn hcs_reset(&mut self) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        graph.hcs = None;
        log::info!("HCS state reset");
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
    /// hottest-first).
    ///
    /// Args:
    ///   ranking: list of (layer_idx, expert_idx) tuples, hottest first
    ///   budget_mb: VRAM budget for pool (0 = auto from free VRAM)
    ///   headroom_mb: VRAM to keep free (only used when budget_mb=0)
    #[pyo3(signature = (ranking, budget_mb=0, headroom_mb=500))]
    fn hcs_pool_init(
        &mut self,
        ranking: Vec<(usize, usize)>,
        budget_mb: usize,
        headroom_mb: usize,
    ) -> PyResult<String> {
        // budget_mb=0 means auto-detect from free VRAM (legacy non-tiered API)
        let actual_budget_mb = if budget_mb == 0 {
            let (free, _total) = cudarc::driver::result::mem_get_info()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("mem_get_info: {:?}", e)))?;
            let headroom_bytes = headroom_mb * 1024 * 1024;
            if free > headroom_bytes {
                (free - headroom_bytes) / (1024 * 1024)
            } else {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Not enough VRAM: {} MB free, {} MB headroom",
                        free / (1024 * 1024), headroom_mb)));
            }
        } else {
            budget_mb
        };
        self.hcs_pool_init_internal(ranking, actual_budget_mb)
    }

    /// Store four-point VRAM calibration data from startup measurement.
    /// Called once during server init, before hcs_pool_init.
    #[pyo3(signature = (short_tokens, long_tokens,
                        prefill_short_free_mb, prefill_long_free_mb,
                        decode_short_free_mb, decode_long_free_mb,
                        baseline_free_mb,
                        safety_margin_mb))]
    fn set_vram_calibration(
        &mut self,
        short_tokens: usize,
        long_tokens: usize,
        prefill_short_free_mb: u64,
        prefill_long_free_mb: u64,
        decode_short_free_mb: u64,
        decode_long_free_mb: u64,
        baseline_free_mb: u64,
        safety_margin_mb: u64,
    ) -> PyResult<String> {
        let cal = VramCalibration {
            short_tokens,
            long_tokens,
            baseline_free_mb,
            prefill_short_free_mb,
            prefill_long_free_mb,
            decode_short_free_mb,
            decode_long_free_mb,
            safety_margin_mb,
        };
        let prefill_kb_per_tok = if long_tokens > short_tokens {
            ((prefill_short_free_mb as f64 - prefill_long_free_mb as f64)
                / (long_tokens - short_tokens) as f64) * 1024.0
        } else { 0.0 };
        let decode_kb_per_tok = if long_tokens > short_tokens {
            ((decode_short_free_mb as f64 - decode_long_free_mb as f64)
                / (long_tokens - short_tokens) as f64) * 1024.0
        } else { 0.0 };

        let short_prefill_budget = cal.prefill_hcs_budget_mb(short_tokens);
        let long_prefill_budget = cal.prefill_hcs_budget_mb(long_tokens);
        let short_decode_budget = cal.decode_hcs_budget_mb(short_tokens);
        let long_decode_budget = cal.decode_hcs_budget_mb(long_tokens);
        let short_prefill_idle_floor = cal.required_prefill_idle_free_mb(short_tokens);
        let short_idle_floor = cal.required_idle_free_mb(short_tokens);

        let msg = format!(
            "VRAM calibration: baseline {} MB | prefill {:.1} KB/tok, decode {:.1} KB/tok | \
             HCS allowance prefill: short {} MB, long {} MB | decode: short {} MB, long {} MB | \
             idle floors prefill={} MB decode={} MB",
            baseline_free_mb,
            prefill_kb_per_tok, decode_kb_per_tok,
            short_prefill_budget, long_prefill_budget,
            short_decode_budget, long_decode_budget,
            short_prefill_idle_floor, short_idle_floor,
        );
        log::info!("{}", msg);
        log::info!("  prefill: short={}tok/{}MB, long={}tok/{}MB",
            short_tokens, prefill_short_free_mb, long_tokens, prefill_long_free_mb);
        log::info!("  decode:  short={}tok/{}MB, long={}tok/{}MB",
            short_tokens, decode_short_free_mb, long_tokens, decode_long_free_mb);
        log::info!("  prefill HCS allowance: short={}MB, long={}MB",
            short_prefill_budget, long_prefill_budget);
        log::info!("  decode HCS allowance: short={}MB, long={}MB",
            short_decode_budget, long_decode_budget);
        log::info!("  baseline idle free: {}MB, safety margin: {}MB",
            baseline_free_mb, safety_margin_mb);

        if let Some(engine) = self.prefill_engine_slot.as_mut() {
            engine.set_safety_margin_mb(safety_margin_mb as usize);
        }
        self.vram_calibration = Some(cal);
        Ok(msg)
    }

    /// Initialize HCS with hard + soft tiers based on VRAM calibration.
    /// hard_budget_mb: experts that survive worst-case prefill
    /// soft_budget_mb: additional experts loaded during decode, evicted for prefill.
    /// safety_margin_mb: minimum free VRAM to maintain (default 600).
    #[pyo3(signature = (ranking, hard_budget_mb, soft_budget_mb, safety_margin_mb=600))]
    fn hcs_pool_init_tiered(
        &mut self,
        ranking: Vec<(usize, usize)>,
        hard_budget_mb: usize,
        soft_budget_mb: usize,
        safety_margin_mb: usize,
    ) -> PyResult<String> {
        // First, init the hard pool using existing logic
        let result = self.hcs_pool_init_internal(
            ranking.clone(), hard_budget_mb,
        )?;

        {
            let graph = self.graph.as_mut()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
            let hcs = graph.hcs.as_mut()
                .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("HCS not initialized"))?;
            hcs.safety_margin_mb = safety_margin_mb;
            hcs.hard_budget_mb = hard_budget_mb;
            hcs.soft_max_mb = soft_budget_mb;
        }

        // Now allocate and fill the soft tier
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let hcs = graph.hcs.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("HCS not initialized"))?;

        let slot_size = hcs.pool_slot_size;
        if slot_size == 0 {
            return Ok(result);
        }

        let soft_budget_bytes = soft_budget_mb * 1024 * 1024;
        let soft_num_slots = soft_budget_bytes / slot_size;
        if soft_num_slots == 0 {
            return Ok(format!("{} | soft: 0 slots (budget too small)", result));
        }

        // Count how many experts actually need soft tier (not already in hard pool)
        let soft_experts_available: usize = ranking.iter()
            .filter(|&&(l, e)| !hcs.cache.contains_key(&(l, e)))
            .count();
        if soft_experts_available == 0 {
            log::info!("HCS soft tier: skipping allocation — all experts already in hard pool (100% coverage)");
            hcs.soft_loaded = true;
            hcs.soft_num_slots = 0;
            hcs.soft_num_cached = 0;
            let total_experts: usize = graph.moe_layers.iter()
                .filter_map(|m| m.as_ref())
                .map(|m| m.num_experts)
                .sum();
            let total_cached = hcs.num_cached;
            let total_pct = if total_experts > 0 {
                total_cached as f64 / total_experts as f64 * 100.0
            } else { 0.0 };
            return Ok(format!(
                "{} | soft: 0 experts (all in hard tier, {:.1} MB saved) | total: {}/{} ({:.1}%) coverage",
                result, soft_budget_mb as f64, total_cached, total_experts, total_pct,
            ));
        }

        // Only allocate for the experts that actually need soft slots
        let mut soft_num_slots = std::cmp::min(soft_num_slots, soft_experts_available);

        // Query actual VRAM and cap allocation to what's really available.
        // The Python-side budget was measured before reaching this point;
        // VRAM may have drifted due to CUDA context growth, allocator overhead,
        // or stale processes.  Always check reality before allocating.
        let safety_bytes = safety_margin_mb * 1024 * 1024;
        if let Ok((actual_free, _)) = cudarc::driver::result::mem_get_info() {
            let usable = actual_free.saturating_sub(safety_bytes);
            let max_slots = usable / slot_size;
            if max_slots < soft_num_slots {
                log::warn!(
                    "HCS soft tier: capping from {} to {} slots ({:.0} MB free, {:.0} MB safety)",
                    soft_num_slots, max_slots,
                    actual_free as f64 / (1024.0 * 1024.0),
                    safety_bytes as f64 / (1024.0 * 1024.0),
                );
                soft_num_slots = max_slots;
            }
        }

        if soft_num_slots == 0 {
            log::warn!("HCS soft tier: no VRAM available for soft experts after safety margin");
            hcs.soft_loaded = true;
            hcs.soft_num_slots = 0;
            hcs.soft_num_cached = 0;
            return Ok(format!("{} | soft: 0 experts (insufficient VRAM)", result));
        }

        // Compute chunk size for granular reclaim and load/reload guardrails.
        // Target ~128 MB on 32 GB, ~64 MB on 16 GB, clamped to [64 MB, 256 MB].
        // Smaller chunks let residency stay closer to the decode-derived HCS bound.
        let chunk_target_bytes = {
            let (_, total_vram) = cudarc::driver::result::mem_get_info()
                .unwrap_or((0, 32 * 1024 * 1024 * 1024));
            let target = total_vram / 256;
            target.max(64 * 1024 * 1024).min(256 * 1024 * 1024)
        };
        let slots_per_chunk = (chunk_target_bytes / slot_size).max(1);
        let planned_num_chunks = (soft_num_slots + slots_per_chunk - 1) / slots_per_chunk;

        let soft_alloc_bytes = soft_num_slots * slot_size;
        log::info!("HCS soft tier: allocating {:.1} MB in {} chunks of {:.0} MB ({} slots/chunk, {} slots for {} available experts)",
            soft_alloc_bytes as f64 / (1024.0 * 1024.0), planned_num_chunks,
            (slots_per_chunk * slot_size) as f64 / (1024.0 * 1024.0),
            slots_per_chunk, soft_num_slots, soft_experts_available);

        // Allocate GPU chunks + pinned host mirrors for batch DMA reload
        let mut soft_chunks: Vec<cudarc::driver::CudaSlice<u8>> = Vec::with_capacity(planned_num_chunks);
        let mut soft_host_chunks: Vec<PinnedHostChunk> = Vec::with_capacity(planned_num_chunks);
        let startup_idle_floor_mb = self.vram_calibration
            .map(|cal| cal.required_idle_free_mb(cal.short_tokens) as usize)
            .unwrap_or(safety_margin_mb);
        let mut actual_soft_slots = 0usize;
        for c in 0..planned_num_chunks {
            let slots_this_chunk = if c == planned_num_chunks - 1 {
                soft_num_slots - c * slots_per_chunk
            } else {
                slots_per_chunk
            };
            let chunk_bytes = slots_this_chunk * slot_size;
            if let Ok((actual_free, _)) = cudarc::driver::result::mem_get_info() {
                let actual_free_mb = actual_free / (1024 * 1024);
                let chunk_mb = (chunk_bytes + (1024 * 1024) - 1) / (1024 * 1024);
                if actual_free_mb <= startup_idle_floor_mb
                    || actual_free_mb.saturating_sub(chunk_mb) < startup_idle_floor_mb
                {
                    log::warn!(
                        "HCS soft tier: stopping at chunk {} — free={} MB, chunk={} MB, idle floor={} MB",
                        c, actual_free_mb, chunk_mb, startup_idle_floor_mb,
                    );
                    break;
                }
            }
            let chunk_buf = self.device.alloc_zeros::<u8>(chunk_bytes)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("HCS soft chunk {} alloc ({} MB): {:?}",
                        c, chunk_bytes / (1024 * 1024), e)))?;
            if let Ok((actual_free, _)) = cudarc::driver::result::mem_get_info() {
                let actual_free_mb = actual_free / (1024 * 1024);
                if actual_free_mb < startup_idle_floor_mb {
                    log::warn!(
                        "HCS soft tier: dropping chunk {} after alloc — free={} MB below idle floor={} MB",
                        c, actual_free_mb, startup_idle_floor_mb,
                    );
                    drop(chunk_buf);
                    break;
                }
            }
            soft_chunks.push(chunk_buf);
            soft_host_chunks.push(PinnedHostChunk::new(chunk_bytes));
            actual_soft_slots += slots_this_chunk;
        }
        soft_num_slots = actual_soft_slots;
        let num_chunks = soft_chunks.len();
        if num_chunks == 0 || soft_num_slots == 0 {
            log::warn!("HCS soft tier: no chunks loaded after decode guardrail");
            hcs.soft_loaded = true;
            hcs.soft_num_slots = 0;
            hcs.soft_num_cached = 0;
            hcs.soft_max_mb = 0;
            return Ok(format!("{} | soft: 0 experts (decode guardrail stopped load)", result));
        }
        let pinned_count = soft_host_chunks.iter().filter(|c| c.registered).count();
        log::info!("HCS soft tier: {}/{} host chunks pinned (page-locked for async DMA)",
            pinned_count, num_chunks);

        let mut soft_slot_to_expert: Vec<Option<(usize, usize)>> = vec![None; soft_num_slots];
        let mut soft_ranking: Vec<(usize, usize)> = Vec::new();
        let mut soft_loaded = 0usize;
        let mut soft_slot = 0usize;

        // Fill soft slots: pack into host chunks, then batch DMA per chunk
        let t0 = std::time::Instant::now();
        // First pass: pack all expert data into host chunk buffers
        let mut experts_per_slot: Vec<Option<(usize, usize, u64, u64, u64, u64)>> = vec![None; soft_num_slots];
        for &(layer_idx, expert_idx) in &ranking {
            if soft_slot >= soft_num_slots {
                break;
            }
            if hcs.cache.contains_key(&(layer_idx, expert_idx)) {
                continue;
            }
            let moe = match graph.moe_layers.get(layer_idx).and_then(|m| m.as_ref()) {
                Some(m) => m,
                None => continue,
            };
            if expert_idx >= moe.experts.len() {
                continue;
            }

            let expert = &moe.experts[expert_idx];
            let chunk_idx = soft_slot / slots_per_chunk;
            let offset_in_chunk = soft_slot % slots_per_chunk;
            let host_offset = offset_in_chunk * slot_size;

            let w13p_off = 0usize;
            let w13s_off = expert.w13_packed_bytes;
            let w2p_off = w13s_off + expert.w13_scales_bytes;
            let w2s_off = w2p_off + expert.w2_packed_bytes;

            // Pack into pinned host chunk buffer
            unsafe {
                let dst = soft_host_chunks[chunk_idx].as_mut_ptr().add(host_offset);
                std::ptr::copy_nonoverlapping(
                    expert.w13_packed_ptr as *const u8, dst.add(w13p_off),
                    expert.w13_packed_bytes);
                std::ptr::copy_nonoverlapping(
                    expert.w13_scales_ptr as *const u8, dst.add(w13s_off),
                    expert.w13_scales_bytes);
                std::ptr::copy_nonoverlapping(
                    expert.w2_packed_ptr as *const u8, dst.add(w2p_off),
                    expert.w2_packed_bytes);
                std::ptr::copy_nonoverlapping(
                    expert.w2_scales_ptr as *const u8, dst.add(w2s_off),
                    expert.w2_scales_bytes);
            }

            experts_per_slot[soft_slot] = Some((
                layer_idx, expert_idx,
                w13p_off as u64, w13s_off as u64, w2p_off as u64, w2s_off as u64,
            ));
            soft_slot_to_expert[soft_slot] = Some((layer_idx, expert_idx));
            soft_ranking.push((layer_idx, expert_idx));
            soft_slot += 1;
            soft_loaded += 1;
        }

        // Second pass: batch DMA each host chunk to GPU in one call
        let mut dma_ok = true;
        for c in 0..num_chunks {
            let slots_this_chunk = if c == num_chunks - 1 {
                soft_num_slots - c * slots_per_chunk
            } else {
                slots_per_chunk
            };
            let chunk_bytes = slots_this_chunk * slot_size;
            unsafe {
                let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                    *soft_chunks[c].device_ptr(),
                    soft_host_chunks[c].as_ptr() as *const std::ffi::c_void,
                    chunk_bytes,
                );
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    log::warn!("HCS soft build: chunk {} batch DMA failed: {:?}", c, err);
                    dma_ok = false;
                    break;
                }
            }
        }

        // Build cache entries from GPU pointers
        if dma_ok {
            for slot in 0..soft_slot {
                if let Some((layer_idx, expert_idx, w13p_off, w13s_off, w2p_off, w2s_off)) = experts_per_slot[slot] {
                    let chunk_idx = slot / slots_per_chunk;
                    let offset_in_chunk = slot % slots_per_chunk;
                    let dst = *soft_chunks[chunk_idx].device_ptr()
                        + (offset_in_chunk as u64 * slot_size as u64);
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
                        pool_slot: None,
                    };
                    hcs.cache_fast_set(layer_idx, expert_idx, &entry);
                    hcs.cache.insert((layer_idx, expert_idx), entry);
                }
            }
        }
        let load_elapsed = t0.elapsed().as_secs_f64();

        hcs.soft_chunks = soft_chunks;
        hcs.soft_host_chunks = soft_host_chunks;
        hcs.soft_slots_per_chunk = slots_per_chunk;
        hcs.soft_total_chunks = num_chunks;
        hcs.soft_chunks_loaded = num_chunks;
        hcs.soft_max_mb = (soft_num_slots * slot_size) / (1024 * 1024);
        hcs.soft_num_slots = soft_num_slots;
        hcs.soft_slot_size = slot_size;
        hcs.soft_slot_to_expert = soft_slot_to_expert;
        hcs.soft_ranking = soft_ranking;
        hcs.soft_num_cached = soft_loaded;
        hcs.soft_loaded = true;
        hcs.num_cached += soft_loaded;
        hcs.vram_bytes += soft_alloc_bytes;

        let total_experts: usize = graph.moe_layers.iter()
            .filter_map(|m| m.as_ref())
            .map(|m| m.num_experts)
            .sum();
        let total_cached = hcs.num_cached;
        let total_pct = if total_experts > 0 {
            total_cached as f64 / total_experts as f64 * 100.0
        } else { 0.0 };

        let msg = format!(
            "{} | soft: {} experts in {:.2}s ({:.1} MB) | \
             total: {}/{} ({:.1}%) coverage",
            result, soft_loaded, load_elapsed,
            soft_alloc_bytes as f64 / (1024.0 * 1024.0),
            total_cached, total_experts, total_pct,
        );
        log::info!("{}", msg);
        Ok(msg)
    }

    /// Get HCS pool statistics.
    fn hcs_dynamic_stats(&self) -> PyResult<String> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let hcs = graph.hcs.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call init_hcs first"))?;
        Ok(format!(
            "HCS: pool={}/{} slots, hit_rate={:.1}%",
            hcs.pool_num_slots - hcs.pool_free_slots.len(), hcs.pool_num_slots,
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

    /// Update LA layer conv/recur state pointers (after prefill reallocates them).
    fn update_la_state_ptrs(
        &mut self,
        layer_idx: usize,
        new_conv_state_ptr: usize,
        new_recur_state_ptr: usize,
    ) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        if layer_idx >= graph.layers.len() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Layer {} out of range", layer_idx)));
        }
        match &mut graph.layers[layer_idx].attn {
            GpuAttnConfig::LinearAttention { conv_state_ptr, recur_state_ptr, .. } => {
                *conv_state_ptr = new_conv_state_ptr as u64;
                *recur_state_ptr = new_recur_state_ptr as u64;
            }
            _ => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Layer {} is not LinearAttention", layer_idx)));
            }
        }
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

    /// Register an MLA (Multi-head Latent Attention) layer for GPU decode.
    #[pyo3(signature = (
        layer_idx, input_norm_ptr, input_norm_size,
        post_attn_norm_ptr, post_attn_norm_size,
        kv_a_proj_wid, o_proj_wid,
        kv_a_norm_ptr, w_kc_ptr, w_vc_ptr,
        num_heads, kv_lora_rank, qk_nope_dim, qk_rope_dim, v_head_dim,
        sm_scale, rope_interleave,
        ckv_cache_ptr, kpe_cache_ptr,
        q_a_proj_wid=None, q_b_proj_wid=None, q_a_norm_ptr=0,
        q_proj_wid=None, q_lora_rank=0, ckv_cache_dim=0
    ))]
    fn register_mla_layer(
        &mut self,
        layer_idx: usize,
        input_norm_ptr: usize, input_norm_size: usize,
        post_attn_norm_ptr: usize, post_attn_norm_size: usize,
        kv_a_proj_wid: usize, o_proj_wid: usize,
        kv_a_norm_ptr: usize, w_kc_ptr: usize, w_vc_ptr: usize,
        num_heads: usize, kv_lora_rank: usize,
        qk_nope_dim: usize, qk_rope_dim: usize, v_head_dim: usize,
        sm_scale: f32, rope_interleave: bool,
        ckv_cache_ptr: usize, kpe_cache_ptr: usize,
        q_a_proj_wid: Option<usize>, q_b_proj_wid: Option<usize>, q_a_norm_ptr: usize,
        q_proj_wid: Option<usize>, q_lora_rank: usize,
        ckv_cache_dim: usize,
    ) -> PyResult<()> {
        // ckv_cache_dim: effective dimension in cache (≥512 for MLA decode).
        // If 0 (default), use kv_lora_rank directly.
        let ckv_cache_dim = if ckv_cache_dim == 0 { kv_lora_rank } else { ckv_cache_dim };

        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        while graph.layers.len() <= layer_idx {
            graph.layers.push(GpuDecodeLayer {
                input_norm_ptr: 0,
                input_norm_size: 0,
                post_attn_norm_ptr: 0,
                post_attn_norm_size: 0,
                attn: GpuAttnConfig::MLA {
                    q_a_proj: None, q_b_proj: None, q_a_norm_ptr: 0,
                    q_proj: None, kv_a_proj: 0, kv_a_norm_ptr: 0,
                    w_kc_ptr: 0, w_vc_ptr: 0, o_proj: 0,
                    num_heads: 0, kv_lora_rank: 0, ckv_cache_dim: 0,
                    qk_nope_dim: 0, qk_rope_dim: 0, v_head_dim: 0,
                    q_lora_rank: 0, sm_scale: 0.0, rope_interleave: true,
                    ckv_cache_ptr: 0, kpe_cache_ptr: 0,
                },
                mlp: GpuMlpConfig::None,
            });
        }
        graph.layers[layer_idx].input_norm_ptr = input_norm_ptr as u64;
        graph.layers[layer_idx].input_norm_size = input_norm_size;
        graph.layers[layer_idx].post_attn_norm_ptr = post_attn_norm_ptr as u64;
        graph.layers[layer_idx].post_attn_norm_size = post_attn_norm_size;
        graph.layers[layer_idx].attn = GpuAttnConfig::MLA {
            q_a_proj: q_a_proj_wid,
            q_b_proj: q_b_proj_wid,
            q_a_norm_ptr: q_a_norm_ptr as u64,
            q_proj: q_proj_wid,
            kv_a_proj: kv_a_proj_wid,
            kv_a_norm_ptr: kv_a_norm_ptr as u64,
            w_kc_ptr: w_kc_ptr as u64,
            w_vc_ptr: w_vc_ptr as u64,
            o_proj: o_proj_wid,
            num_heads,
            kv_lora_rank,
            ckv_cache_dim,
            qk_nope_dim,
            qk_rope_dim,
            v_head_dim,
            q_lora_rank,
            sm_scale,
            rope_interleave,
            ckv_cache_ptr: ckv_cache_ptr as u64,
            kpe_cache_ptr: kpe_cache_ptr as u64,
        };
        log::info!("GpuDecodeStore: registered MLA layer {} (heads={}, kv_lora_rank={}, ckv_cache_dim={}, nope={}, rope={}, v_hd={})",
            layer_idx, num_heads, kv_lora_rank, ckv_cache_dim, qk_nope_dim, qk_rope_dim, v_head_dim);
        Ok(())
    }

    /// Register a Mamba2 SSM layer for GPU decode (Nemotron-H hybrid models).
    /// Mamba2 layers have O(1) per-token state, no KV cache, always GPU-resident.
    #[allow(clippy::too_many_arguments)]
    fn register_mamba2_layer(
        &mut self,
        layer_idx: usize,
        input_norm_ptr: usize, input_norm_size: usize,
        post_attn_norm_ptr: usize, post_attn_norm_size: usize,
        in_proj_wid: usize, out_proj_wid: usize,
        conv_weight_ptr: usize, a_ptr: usize, d_ptr: usize,
        dt_bias_ptr: usize, norm_weight_ptr: usize,
        conv_state_ptr: usize, ssm_state_ptr: usize,
        num_heads: usize, head_dim: usize, state_size: usize,
        expand: usize, conv_kernel: usize,
    ) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let conv_dim = expand * head_dim * num_heads;
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
        graph.layers[layer_idx].attn = GpuAttnConfig::Mamba2 {
            in_proj: in_proj_wid,
            out_proj: out_proj_wid,
            conv_weight_ptr: conv_weight_ptr as u64,
            a_ptr: a_ptr as u64,
            d_ptr: d_ptr as u64,
            dt_bias_ptr: dt_bias_ptr as u64,
            norm_weight_ptr: norm_weight_ptr as u64,
            num_heads,
            head_dim,
            state_size,
            expand,
            conv_kernel,
            conv_dim,
            conv_state_ptr: conv_state_ptr as u64,
            ssm_state_ptr: ssm_state_ptr as u64,
        };
        // Allocate mamba2 conv_out buffer if not yet allocated (sized to max conv_dim)
        if graph.mamba2_conv_out_size < conv_dim {
            graph.d_mamba2_conv_out = Some(self.device.alloc_zeros::<f32>(conv_dim)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("alloc mamba2_conv_out: {:?}", e)))?);
            graph.mamba2_conv_out_size = conv_dim;
        }
        log::info!("GpuDecodeStore: registered Mamba2 layer {} (heads={}, hd={}, state={}, conv_dim={}), total_layers={}",
            layer_idx, num_heads, head_dim, state_size, conv_dim, graph.layers.len());
        Ok(())
    }

    /// Set Mamba2 n_groups parameter (shared across all Mamba2 layers).
    fn set_mamba2_n_groups(&mut self, n_groups: usize) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        graph.mamba2_n_groups = n_groups;
        log::info!("GpuDecodeStore: set mamba2_n_groups={}", n_groups);
        Ok(())
    }

    /// Register conv bias pointer for a Mamba2 layer (FP32 on GPU).
    fn set_mamba2_conv_bias(&mut self, layer_idx: usize, conv_bias_ptr: usize) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        graph.mamba2_conv_bias_ptrs.insert(layer_idx, conv_bias_ptr as u64);
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
    /// Python owns the memory (FP8 E4M3 contiguous tensors). Both Rust
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

    /// Register Polar4 KV cache pointers from Python's PagedKVCache.
    /// polar4_ptrs: list of (layer_idx, k_radius_ptr, v_radius_ptr, k_angles_ptr, v_angles_ptr)
    #[pyo3(signature = (polar4_ptrs, max_seq, num_blocks))]
    fn set_kv_cache_ptrs_polar4(
        &mut self,
        polar4_ptrs: Vec<(usize, usize, usize, usize, usize)>,
        max_seq: usize,
        num_blocks: usize,
    ) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let num_layers = graph.layers.len();
        graph.kv_k_radius_ptrs = vec![0u64; num_layers];
        graph.kv_v_radius_ptrs = vec![0u64; num_layers];
        graph.kv_k_angles_ptrs = vec![0u64; num_layers];
        graph.kv_v_angles_ptrs = vec![0u64; num_layers];
        graph.kv_k_ptrs = vec![0u64; num_layers]; // clear FP8 ptrs
        graph.kv_v_ptrs = vec![0u64; num_layers];
        graph.kv_max_seq = max_seq;
        graph.kv_format = 2;
        graph.kv_num_blocks = num_blocks;
        let mut registered = 0usize;
        for (layer_idx, kr_ptr, vr_ptr, ka_ptr, va_ptr) in polar4_ptrs {
            if layer_idx >= num_layers {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Layer {} out of range ({})", layer_idx, num_layers)));
            }
            graph.kv_k_radius_ptrs[layer_idx] = kr_ptr as u64;
            graph.kv_v_radius_ptrs[layer_idx] = vr_ptr as u64;
            graph.kv_k_angles_ptrs[layer_idx] = ka_ptr as u64;
            graph.kv_v_angles_ptrs[layer_idx] = va_ptr as u64;
            registered += 1;
        }
        log::info!("GpuDecodeStore: Polar4 KV cache pointers set ({} GQA layers, max_seq={}, blocks={})",
            registered, max_seq, num_blocks);

        // Allocate FlashDecoding tiled attention buffers (same as FP8 path).
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
            let partial_o_size = max_nh * max_tiles * max_hd;
            let partial_lse_size = max_nh * max_tiles * 2;
            let d_tiled_o = self.device.alloc_zeros::<f32>(partial_o_size)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
            let d_tiled_lse = self.device.alloc_zeros::<f32>(partial_lse_size)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
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

    /// Load a draft model for speculative decoding.
    /// model_dir: path to the model directory (e.g. ~/.krasis/models/Qwen3-0.6B)
    /// max_seq: max KV cache length for draft model (default 4096)
    /// draft_k: number of tokens to draft per round (default 8)
    /// context_window: how many prompt tokens to feed the draft model for warmup (default 512)
    #[pyo3(signature = (model_dir, max_seq=4096, draft_k=3, context_window=512))]
    fn load_draft_model(
        &mut self,
        model_dir: &str,
        max_seq: usize,
        draft_k: usize,
        context_window: usize,
    ) -> PyResult<()> {
        if !self.kernels_loaded {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Decode kernels must be loaded before draft model"));
        }
        let draft = crate::draft_model::DraftModel::load(&self.device, model_dir, max_seq)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        log::info!("Draft model loaded: {:.1} MB VRAM, draft_k={}, context_window={}",
            draft.vram_bytes as f64 / 1e6, draft_k, context_window);
        self.draft = Some(draft);
        self.draft_k = draft_k;
        self.draft_context_window = context_window;

        // Allocate batch buffers for batched speculative verification
        self.allocate_batch_buffers(draft_k + 1)?;

        Ok(())
    }

    /// Check if a draft model is loaded.
    #[getter]
    fn has_draft_model(&self) -> bool {
        self.draft.is_some()
    }

    /// Set the Jaccard similarity threshold for fail-fast expert divergence.
    /// Lower = more lenient (fewer bailouts), higher = stricter (more bailouts).
    /// Default 0.15. Set to 0.0 to disable fail-fast.
    #[pyo3(signature = (threshold=0.15))]
    fn set_spec_jaccard_threshold(&mut self, threshold: f32) {
        self.spec_jaccard_threshold = threshold.clamp(0.0, 1.0);
        log::info!("Speculative Jaccard threshold set to {:.3}", self.spec_jaccard_threshold);
    }

    /// Set token IDs to suppress during sampling (logit → -inf).
    /// Prevents the model from generating turn-boundary tokens like <|im_start|>.
    fn set_suppress_tokens(&mut self, token_ids: Vec<usize>) {
        if !token_ids.is_empty() {
            log::info!("Suppress tokens set: {:?}", token_ids);
        }
        self.suppress_tokens = token_ids;
    }

    /// Return the currently configured suppress-token list for reuse by Rust prefill.
    pub fn suppress_tokens_clone(&self) -> Vec<u32> {
        self.suppress_tokens.iter().map(|&t| t as u32).collect()
    }

    /// Set min_new_tokens: stop tokens are suppressed for the first N decode steps.
    /// Set stop_ids each request via set_min_new_tokens_stop_ids before decode.
    fn set_min_new_tokens(&mut self, n: usize) {
        self.min_new_tokens = n;
    }

    /// Set stop token IDs for min_new_tokens suppression (per-request).
    fn set_min_new_tokens_stop_ids(&mut self, stop_ids: Vec<usize>) {
        self.stop_token_ids_for_suppress = stop_ids;
    }

    /// Get self pointer for Rust-side access (same pattern as CpuDecodeStore).
    fn gpu_store_addr(&self) -> usize {
        self as *const GpuDecodeStore as usize
    }

    /// Get the CUDA device ordinal this store was created on.
    fn gpu_index(&self) -> usize {
        self.device.ordinal()
    }

    /// Return the min free VRAM (MB) recorded by the most recent batch/stream decode run.
    fn get_last_min_free_vram_mb(&self) -> usize {
        self.last_min_free_vram_mb
    }

    /// Return benchmark stats: (min_free_vram_mb, hcs_loaded, hcs_total, hcs_pct)
    fn get_benchmark_stats(&self) -> (usize, usize, usize, f64) {
        self.benchmark_stats()
    }

    /// Clamp the soft-tier resident budget to a measured target.
    /// Used by startup decode residency calibration after HCS load.
    fn hcs_clamp_soft_budget_mb(&mut self, target_soft_mb: usize) -> PyResult<(usize, usize)> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let hcs = graph.hcs.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("HCS not initialized"))?;

        if hcs.soft_slot_size == 0 || hcs.soft_num_slots == 0 || hcs.soft_slots_per_chunk == 0 {
            hcs.soft_max_mb = 0;
            return Ok((0, 0));
        }

        let slot_size = hcs.soft_slot_size;
        let spc = hcs.soft_slots_per_chunk;
        let target_soft_bytes = target_soft_mb.saturating_mul(1024 * 1024);
        let target_soft_slots = (target_soft_bytes / slot_size).min(hcs.soft_num_slots);
        let target_chunks = if target_soft_slots == 0 {
            0
        } else {
            ((target_soft_slots + spc - 1) / spc).min(hcs.soft_total_chunks)
        };

        let (_evicted, _freed_bytes) = hcs.trim_soft_chunks_to(target_chunks);

        let loaded_slots = if hcs.soft_chunks_loaded >= hcs.soft_total_chunks {
            hcs.soft_num_slots
        } else {
            (hcs.soft_chunks_loaded * spc).min(hcs.soft_num_slots)
        };
        let loaded_soft_mb = (loaded_slots * slot_size) / (1024 * 1024);
        hcs.soft_max_mb = loaded_soft_mb;
        hcs.soft_loaded = hcs.soft_chunks_loaded == hcs.soft_total_chunks;

        Ok((hcs.soft_max_mb, loaded_soft_mb))
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
        let mut min_vram_free_bytes: usize = usize::MAX;

        {
            let mut free: usize = 0;
            let mut _total: usize = 0;
            unsafe { let _ = cuda_sys::lib().cuMemGetInfo_v2(&mut free, &mut _total); }
            if free < min_vram_free_bytes {
                min_vram_free_bytes = free;
            }
        }

        for step in 0..max_tokens {
            let pos = start_position + step;
            self.gpu_decode_step(next_token, pos)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("gpu_decode_step error: {}", e)))?;

            let mut free: usize = 0;
            let mut _total: usize = 0;
            unsafe { let _ = cuda_sys::lib().cuMemGetInfo_v2(&mut free, &mut _total); }
            if free < min_vram_free_bytes {
                min_vram_free_bytes = free;
            }

            let logits = &mut self.graph.as_mut().unwrap().h_logits;
            if presence_penalty != 0.0 {
                for &tok in &seen_tokens {
                    if tok < vocab_size {
                        logits[tok] -= presence_penalty;
                    }
                }
            }
            for &tok in &self.suppress_tokens {
                if tok < vocab_size { logits[tok] = f32::NEG_INFINITY; }
            }
            let think_end_active = self.think_end_token_for_suppress.is_some() && !self.think_end_seen;
            let think_within_budget = self.think_suppress_budget == 0
                || self.think_suppress_count < self.think_suppress_budget;
            if step < self.min_new_tokens || (think_end_active && think_within_budget) {
                for &tok in &self.stop_token_ids_for_suppress {
                    if tok < vocab_size { logits[tok] = f32::NEG_INFINITY; }
                }
            }

            // Force-inject </think> when thinking budget is exhausted
            next_token = if think_end_active && !think_within_budget {
                if let Some(te) = self.think_end_token_for_suppress {
                    eprintln!("[krasis] Thinking budget exhausted ({} tokens) — force-injecting </think>",
                        self.think_suppress_count);
                    te
                } else {
                    crate::decode::sample_from_logits_pub(
                        logits, vocab_size, temperature, top_k, top_p, &mut rng_next)
                }
            } else {
                crate::decode::sample_from_logits_pub(
                    logits, vocab_size, temperature, top_k, top_p, &mut rng_next)
            };
            self.notify_token_generated(next_token);
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
        if min_vram_free_bytes < usize::MAX {
            self.last_min_free_vram_mb = min_vram_free_bytes / (1024 * 1024);
        }
        self.last_decode_elapsed = elapsed;
        self.emit_batch_decode_timing_summary();

        Ok(tokens)
    }

    /// Startup calibration probe: run the production decode stream path
    /// without Python-side streaming, so CUDA-graph capture and steady-state
    /// decode allocations are included in VRAM measurement.
    #[pyo3(signature = (tokenizer_path, first_token, start_position, max_tokens, temperature, top_k, top_p, stop_ids, presence_penalty=0.0))]
    fn gpu_generate_stream_probe(
        &mut self,
        tokenizer_path: String,
        first_token: usize,
        start_position: usize,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        stop_ids: Vec<usize>,
        presence_penalty: f32,
    ) -> PyResult<Vec<usize>> {
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to load tokenizer for decode probe: {}", e)))?;
        let mut tokens = Vec::with_capacity(max_tokens);
        self.gpu_generate_stream(
            first_token,
            start_position,
            max_tokens,
            temperature,
            top_k,
            top_p,
            &stop_ids,
            &tokenizer,
            presence_penalty,
            0,
            |token_id, _text, _finish_reason, _logprobs| {
                tokens.push(token_id);
                true
            },
        );
        Ok(tokens)
    }

    /// Multi-GPU batch generate (for warmup/validation from Python).
    ///
    /// Like gpu_generate_batch but uses gpu_generate_stream_multi under the hood.
    fn gpu_generate_batch_multi(
        &mut self,
        aux_store_addrs: Vec<usize>,
        split_layers: Vec<usize>,
        gqa_cache_offsets: Vec<usize>,
        first_token: usize,
        start_position: usize,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        stop_ids: Vec<usize>,
        presence_penalty: f32,
    ) -> PyResult<Vec<usize>> {
        let num_aux = aux_store_addrs.len();
        if num_aux == 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("no aux stores provided"));
        }
        if split_layers.len() != num_aux || gqa_cache_offsets.len() != num_aux {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "aux_store_addrs, split_layers, and gqa_cache_offsets must have same length"));
        }

        let num_layers = match self.graph.as_ref() {
            Some(g) => g.layers.len(),
            None => return Err(pyo3::exceptions::PyRuntimeError::new_err("primary graph not configured")),
        };

        // Layer boundaries: [0, split_layers[0], split_layers[1], ..., num_layers]
        let mut boundaries = Vec::with_capacity(num_aux + 2);
        boundaries.push(0usize);
        boundaries.extend_from_slice(&split_layers);
        boundaries.push(num_layers);

        // Vocab size from last aux store (has LM head)
        let last_aux = unsafe { &mut *(aux_store_addrs[num_aux - 1] as *mut GpuDecodeStore) };
        let vocab_size = match last_aux.graph.as_ref() {
            Some(g) => g.vocab_size,
            None => return Err(pyo3::exceptions::PyRuntimeError::new_err("last aux graph not configured")),
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

            // GPU0: embedding + layers [0..boundaries[1])
            if let Err(e) = self.device.bind_to_thread() {
                log::error!("batch_multi: bind GPU0 failed: {:?}", e);
                break;
            }
            let gpu0_is_last = num_aux == 0;
            if let Err(e) = self.gpu_decode_segment(
                next_token, pos, true, 0, boundaries[1], gpu0_is_last, 0,
            ) {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("GPU0 segment error: {}", e)));
            }

            // Pipeline through aux GPUs
            for i in 0..num_aux {
                // Download hidden state from previous GPU
                let (h_hidden, h_residual) = if i == 0 {
                    self.download_hidden_state()
                } else {
                    let prev = unsafe { &mut *(aux_store_addrs[i - 1] as *mut GpuDecodeStore) };
                    prev.download_hidden_state()
                }.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("download_hidden from GPU{} error: {}", i, e)))?;

                // Upload to current aux GPU
                let aux = unsafe { &mut *(aux_store_addrs[i] as *mut GpuDecodeStore) };
                if let Err(e) = aux.device.bind_to_thread() {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("bind GPU{} failed: {:?}", i + 1, e)));
                }
                aux.upload_hidden_state(&h_hidden, &h_residual)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                        format!("upload_hidden to GPU{} error: {}", i + 1, e)))?;

                let is_last = i + 1 == num_aux;
                if let Err(e) = aux.gpu_decode_segment(
                    next_token, pos, false,
                    boundaries[i + 1], boundaries[i + 2],
                    is_last, gqa_cache_offsets[i],
                ) {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("GPU{} segment error: {}", i + 1, e)));
                }
            }

            // Logits from last aux store
            let last_aux = unsafe { &mut *(aux_store_addrs[num_aux - 1] as *mut GpuDecodeStore) };
            let logits = &mut last_aux.graph.as_mut().unwrap().h_logits;
            if presence_penalty != 0.0 {
                for &tok in &seen_tokens {
                    if tok < vocab_size { logits[tok] -= presence_penalty; }
                }
            }
            self.apply_suppress_at_step(logits, vocab_size, step);

            next_token = crate::decode::sample_from_logits_pub(
                logits, vocab_size, temperature, top_k, top_p, &mut rng_next);
            self.notify_token_generated(next_token);
            seen_tokens.insert(next_token);
            tokens.push(next_token);

            if stop_set.contains(&next_token) { break; }
        }

        let elapsed = decode_start.elapsed().as_secs_f64();
        if !tokens.is_empty() {
            let tps = tokens.len() as f64 / elapsed;
            log::info!("gpu_generate_batch_multi: {} tokens in {:.2}s ({:.1} tok/s)",
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

    /// Get total number of layers in the model.
    pub fn num_layers(&self) -> usize {
        self.graph.as_ref().map_or(0, |g| g.layers.len())
    }

    /// Evict soft-tier HCS experts before prefill (PyO3 wrapper).
    /// Returns (evicted_count, freed_mb).
    #[pyo3(signature = (estimated_tokens))]
    fn py_hcs_evict_for_prefill(&mut self, estimated_tokens: usize) -> (usize, f64) {
        self.hcs_evict_for_prefill(estimated_tokens)
    }

    /// Reload soft-tier HCS experts after prefill (PyO3 wrapper).
    /// actual_tokens: prompt length for adaptive sizing (0 = use full soft budget).
    /// Returns (loaded_count, reload_ms).
    #[pyo3(signature = (actual_tokens=0))]
    fn py_hcs_reload_after_prefill(&mut self, actual_tokens: usize) -> (usize, f64) {
        self.hcs_reload_after_prefill(actual_tokens)
    }

    /// Copy KV cache from this store to an aux store (PyO3 wrapper for validation).
    #[pyo3(signature = (aux_store_addr, layer_start, layer_end, gqa_offset, prompt_len))]
    fn py_copy_kv_to_aux(&self, aux_store_addr: usize, layer_start: usize, layer_end: usize, gqa_offset: usize, prompt_len: usize) -> PyResult<()> {
        let aux_store = unsafe { &mut *(aux_store_addr as *mut GpuDecodeStore) };
        self.copy_kv_to_aux(aux_store, layer_start, layer_end, gqa_offset, prompt_len)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Fill HCS-resident experts into a GPU buffer for prefill via D2D copy.
    /// For each expert in the given MoE layer that's in HCS VRAM, copies its
    /// weights into the correct position in the destination GPU tensors.
    ///
    /// Args:
    ///   moe_layer_idx: MoE layer index
    ///   d_w13p, d_w13s, d_w2p, d_w2s: GPU destination tensor data_ptr() values
    ///   stride_w13p, stride_w13s, stride_w2p, stride_w2s: per-expert byte stride in each tensor
    ///   num_experts: total experts in this layer
    ///
    /// Returns: list of expert indices that were filled from HCS (cold experts need H2D DMA).
    #[pyo3(signature = (moe_layer_idx, d_w13p, d_w13s, d_w2p, d_w2s,
                         stride_w13p, stride_w13s, stride_w2p, stride_w2s,
                         num_experts))]
    fn py_hcs_fill_layer_for_prefill(
        &self,
        moe_layer_idx: usize,
        d_w13p: u64, d_w13s: u64, d_w2p: u64, d_w2s: u64,
        stride_w13p: usize, stride_w13s: usize,
        stride_w2p: usize, stride_w2s: usize,
        num_experts: usize,
    ) -> Vec<usize> {
        let graph = match self.graph.as_ref() {
            Some(g) => g,
            None => return Vec::new(),
        };
        let hcs = match graph.hcs.as_ref() {
            Some(h) => h,
            None => return Vec::new(),
        };

        let nep = hcs.num_experts_per_layer;
        if nep == 0 {
            return Vec::new();
        }

        let mut filled = Vec::new();
        for expert_idx in 0..num_experts {
            let idx = moe_layer_idx * nep + expert_idx;
            if idx >= hcs.cache_fast.len() {
                continue;
            }
            let ptrs = &hcs.cache_fast[idx];
            if ptrs[0] == 0 {
                continue; // not in HCS
            }
            // D2D copy from HCS VRAM to destination tensor slot
            let expert_off = expert_idx as u64;
            unsafe {
                let _ = cuda_sys::lib().cuMemcpyDtoD_v2(
                    d_w13p + expert_off * stride_w13p as u64,
                    ptrs[0], stride_w13p,
                );
                let _ = cuda_sys::lib().cuMemcpyDtoD_v2(
                    d_w13s + expert_off * stride_w13s as u64,
                    ptrs[1], stride_w13s,
                );
                let _ = cuda_sys::lib().cuMemcpyDtoD_v2(
                    d_w2p + expert_off * stride_w2p as u64,
                    ptrs[2], stride_w2p,
                );
                let _ = cuda_sys::lib().cuMemcpyDtoD_v2(
                    d_w2s + expert_off * stride_w2s as u64,
                    ptrs[3], stride_w2s,
                );
            }
            filled.push(expert_idx);
        }

        filled
    }

    /// Fill a prefill layer's GPU tensors using D2D for HCS-resident experts
    /// and H2D for cold experts, all in Rust with no Python per-expert overhead.
    ///
    /// Args:
    ///   moe_layer_idx: the MoE layer index
    ///   d_w13p, d_w13s, d_w2p, d_w2s: GPU destination pointers (empty tensors)
    ///   h_w13p, h_w13s, h_w2p, h_w2s: pinned host source pointers
    ///   stride_w13p, stride_w13s, stride_w2p, stride_w2s: per-expert byte stride
    ///   num_experts: total experts in this layer
    ///   cuda_stream: raw CUDA stream pointer for async H2D (0 = default stream)
    ///
    /// Returns: number of HCS-hit experts (D2D'd).
    #[pyo3(signature = (moe_layer_idx, d_w13p, d_w13s, d_w2p, d_w2s,
                         h_w13p, h_w13s, h_w2p, h_w2s,
                         stride_w13p, stride_w13s, stride_w2p, stride_w2s,
                         num_experts, cuda_stream))]
    fn py_hcs_fill_layer_selective(
        &self,
        moe_layer_idx: usize,
        d_w13p: u64, d_w13s: u64, d_w2p: u64, d_w2s: u64,
        h_w13p: u64, h_w13s: u64, h_w2p: u64, h_w2s: u64,
        stride_w13p: usize, stride_w13s: usize,
        stride_w2p: usize, stride_w2s: usize,
        num_experts: usize,
        cuda_stream: u64,
    ) -> usize {
        let graph = match self.graph.as_ref() {
            Some(g) => g,
            None => return 0,
        };
        let hcs = match graph.hcs.as_ref() {
            Some(h) => h,
            None => return 0,
        };

        let nep = hcs.num_experts_per_layer;
        if nep == 0 {
            return 0;
        }

        let stream = cuda_stream as cuda_sys::CUstream;
        let mut hcs_hits = 0usize;

        for expert_idx in 0..num_experts {
            let idx = moe_layer_idx * nep + expert_idx;
            let expert_off = expert_idx as u64;

            let in_hcs = idx < hcs.cache_fast.len() && hcs.cache_fast[idx][0] != 0;

            if in_hcs {
                // D2D from HCS VRAM (synchronous, ~700 GB/s)
                let ptrs = &hcs.cache_fast[idx];
                unsafe {
                    let _ = cuda_sys::lib().cuMemcpyDtoD_v2(
                        d_w13p + expert_off * stride_w13p as u64,
                        ptrs[0], stride_w13p,
                    );
                    let _ = cuda_sys::lib().cuMemcpyDtoD_v2(
                        d_w13s + expert_off * stride_w13s as u64,
                        ptrs[1], stride_w13s,
                    );
                    let _ = cuda_sys::lib().cuMemcpyDtoD_v2(
                        d_w2p + expert_off * stride_w2p as u64,
                        ptrs[2], stride_w2p,
                    );
                    let _ = cuda_sys::lib().cuMemcpyDtoD_v2(
                        d_w2s + expert_off * stride_w2s as u64,
                        ptrs[3], stride_w2s,
                    );
                }
                hcs_hits += 1;
            } else {
                // H2D from pinned host (async on provided stream)
                unsafe {
                    let _ = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        d_w13p + expert_off * stride_w13p as u64,
                        (h_w13p + expert_off * stride_w13p as u64) as *const std::ffi::c_void,
                        stride_w13p,
                        stream,
                    );
                    let _ = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        d_w13s + expert_off * stride_w13s as u64,
                        (h_w13s + expert_off * stride_w13s as u64) as *const std::ffi::c_void,
                        stride_w13s,
                        stream,
                    );
                    let _ = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        d_w2p + expert_off * stride_w2p as u64,
                        (h_w2p + expert_off * stride_w2p as u64) as *const std::ffi::c_void,
                        stride_w2p,
                        stream,
                    );
                    let _ = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        d_w2s + expert_off * stride_w2s as u64,
                        (h_w2s + expert_off * stride_w2s as u64) as *const std::ffi::c_void,
                        stride_w2s,
                        stream,
                    );
                }
            }
        }

        hcs_hits
    }

    /// Test KV cache kernels in isolation.
    #[pyo3(signature = (q_ptr, k_ptr, v_ptr, out_ptr, cache_k_ptr, cache_v_ptr, cache_k_angles_ptr=0, cache_v_angles_ptr=0, nh=0, nkv=0, hd=0, pos=0, max_seq=0, kv_stride=0, sm_sc=0.0, format=0))]
    fn test_kv_cache(
        &self,
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        out_ptr: u64,
        cache_k_ptr: u64,
        cache_v_ptr: u64,
        cache_k_angles_ptr: u64,
        cache_v_angles_ptr: u64,
        nh: i32,
        nkv: i32,
        hd: i32,
        pos: i32,
        max_seq: i32,
        kv_stride: i32,
        sm_sc: f32,
        format: i32, // 0=FP16, 1=FP8, 2=POLAR4
    ) -> PyResult<()> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let k = graph.kernels.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Kernels not loaded (check configure)"))?;

        let threads = 256u32;
        let blocks_write = ((kv_stride as u32) + threads - 1) / threads;

        unsafe {
            // 1. Write to cache
            if format == 2 {
                // Polar4 uses different packing: radius and angles
                k.kv_cache_write_polar4.clone().launch(
                    LaunchConfig { grid_dim: (blocks_write, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                    (cache_k_ptr, cache_v_ptr, cache_k_angles_ptr, cache_v_angles_ptr, k_ptr, v_ptr, pos, kv_stride),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("kv_cache_write_polar4 failed: {:?}", e)))?;
            } else {
                k.kv_cache_write.clone().launch(
                    LaunchConfig { grid_dim: (blocks_write, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                    (cache_k_ptr, cache_v_ptr, k_ptr, v_ptr, pos, kv_stride),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("kv_cache_write failed: {:?}", e)))?;
            }

            // 2. Run attention
            let seq_len = pos + 1;
            if format == 2 {
                let num_warps = threads / 32;
                let shared_mem_bytes = ((hd as u32) * (num_warps + 1) + 2 * num_warps) * 4 + 128;
                k.gqa_attention_polar4.clone().launch(
                    LaunchConfig {
                        grid_dim: (nh as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes,
                    },
                    (out_ptr, q_ptr, cache_k_ptr, cache_v_ptr, cache_k_angles_ptr, cache_v_angles_ptr, sm_sc, nh, nkv, hd, seq_len, max_seq),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("gqa_attention_polar4 failed: {:?}", e)))?;
            } else {
                let q_smem = (hd as u32) * 4;
                let shared_mem_bytes = q_smem + (seq_len as u32) * 4 + 128;
                k.gqa_attention.clone().launch(
                    LaunchConfig {
                        grid_dim: (nh as u32, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes,
                    },
                    (out_ptr, q_ptr, cache_k_ptr, cache_v_ptr, sm_sc, nh, nkv, hd, seq_len, max_seq, 1i32),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("gqa_attention failed: {:?}", e)))?;
            }
        }

        Ok(())
    }

    /// Set the decode segment for this store (which layers it handles during multi-GPU decode).
    /// This affects HCS% reporting — only experts in this segment are counted.
    #[pyo3(signature = (layer_start, layer_end))]
    fn set_decode_segment(&mut self, layer_start: usize, layer_end: usize) -> PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        graph.decode_layer_start = layer_start;
        graph.decode_layer_end = layer_end;
        log::info!("GpuDecodeStore: decode segment set to [{}, {})", layer_start, layer_end);
        Ok(())
    }
}

// ── Pure-Rust methods for GPU decode (no PyO3, used by Rust HTTP server) ──

impl GpuDecodeStore {
    /// Set KV cache position after Rust prefill (no GIL needed).
    /// Equivalent to the pyo3 `set_kv_position` but callable from Rust directly.
    pub fn set_kv_position_rust(&mut self, seq_len: usize) {
        if let Some(ref mut graph) = self.graph {
            graph.kv_current_pos = seq_len;
            log::info!("Rust prefill: set KV position to {}", seq_len);
        }
    }

    /// Take the pre-allocated prefill engine from the slot.
    /// Called by the Rust server during startup. Returns None if not pre-allocated.
    pub fn take_prefill_engine(&mut self) -> Option<crate::gpu_prefill::PrefillEngine> {
        self.prefill_engine_slot.take()
    }

    /// Export HCS cache_fast snapshot for the Rust prefill engine.
    /// Returns (cache_fast_slice, num_experts_per_layer). Empty if no HCS.
    pub fn export_hcs_snapshot(&self) -> (&[[u64; 4]], usize) {
        if let Some(ref graph) = self.graph {
            if let Some(ref hcs) = graph.hcs {
                return (&hcs.cache_fast, hcs.num_experts_per_layer);
            }
        }
        (&[], 0)
    }

    /// Swap GPU weight slots from Marlin format to simple INT4 format for decode.
    /// Must be called after prefill and before decode when using AWQ single-slot weights.
    pub fn swap_to_simple_int4_rust(&mut self) -> Result<(), String> {
        if self.single_slot_swaps.is_empty() {
            return Ok(());
        }
        let mut total_bytes: usize = 0;
        for entry in &self.single_slot_swaps {
            unsafe {
                let r = cuda_sys::lib().cuMemcpyHtoD_v2(
                    entry.packed_slot_ptr,
                    entry.simple_packed_host.as_ptr() as *const std::ffi::c_void,
                    entry.simple_packed_host.len());
                if r != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("swap_to_simple_int4 packed: {:?}", r));
                }
                let r = cuda_sys::lib().cuMemcpyHtoD_v2(
                    entry.scales_slot_ptr,
                    entry.simple_scales_host.as_ptr() as *const std::ffi::c_void,
                    entry.simple_scales_host.len());
                if r != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("swap_to_simple_int4 scales: {:?}", r));
                }
            }
            total_bytes += entry.simple_packed_host.len() + entry.simple_scales_host.len();
        }
        if stderr_debug_enabled() {
            eprintln!("[krasis] Swapped to simple INT4 for decode ({:.1} MB DMA, {} weights)",
                total_bytes as f64 / 1024.0 / 1024.0, self.single_slot_swaps.len());
        }
        Ok(())
    }

    /// Swap GPU weight slots from simple INT4 format back to Marlin format for prefill.
    /// Must be called before prefill when using AWQ single-slot weights.
    pub fn swap_to_marlin_rust(&mut self) -> Result<(), String> {
        if self.single_slot_swaps.is_empty() {
            return Ok(());
        }
        let mut total_bytes: usize = 0;
        for entry in &self.single_slot_swaps {
            unsafe {
                let r = cuda_sys::lib().cuMemcpyHtoD_v2(
                    entry.packed_slot_ptr,
                    entry.marlin_packed_host.as_ptr() as *const std::ffi::c_void,
                    entry.marlin_packed_host.len());
                if r != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("swap_to_marlin packed: {:?}", r));
                }
                let r = cuda_sys::lib().cuMemcpyHtoD_v2(
                    entry.scales_slot_ptr,
                    entry.marlin_scales_host.as_ptr() as *const std::ffi::c_void,
                    entry.marlin_scales_host.len());
                if r != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("swap_to_marlin scales: {:?}", r));
                }
            }
            total_bytes += entry.marlin_packed_host.len() + entry.marlin_scales_host.len();
        }
        if stderr_debug_enabled() {
            eprintln!("[krasis] Swapped to Marlin for prefill ({:.1} MB DMA, {} weights)",
                total_bytes as f64 / 1024.0 / 1024.0, self.single_slot_swaps.len());
        }
        Ok(())
    }

    /// Set min_new_tokens and stop_ids for suppression (called from Rust server code).
    pub fn set_min_new_tokens_ext(&mut self, n: usize, stop_ids: Vec<usize>) {
        self.min_new_tokens = n;
        self.stop_token_ids_for_suppress = stop_ids;
    }

    /// Configure think-end suppression: suppress stop tokens until think_end_id is generated.
    /// Call before each decode when thinking is enabled. Pass None to disable.
    /// `budget` = max thinking tokens before suppression is lifted (0 = unlimited).
    pub fn set_think_end_suppress(&mut self, think_end_id: Option<usize>, budget: usize) {
        self.think_end_token_for_suppress = think_end_id;
        self.think_end_seen = false;
        self.think_suppress_budget = budget;
        self.think_suppress_count = 0;
    }

    /// Notify that a token was generated — checks for think_end_token and counts thinking tokens.
    #[inline]
    pub fn notify_token_generated(&mut self, token_id: usize) {
        if let Some(te) = self.think_end_token_for_suppress {
            if token_id == te {
                self.think_end_seen = true;
                log::info!("Think end token seen after {} thinking tokens (budget: {})",
                    self.think_suppress_count, self.think_suppress_budget);
            } else if !self.think_end_seen {
                // Still in thinking mode — count toward budget
                self.think_suppress_count += 1;
                if self.think_suppress_budget > 0 && self.think_suppress_count == self.think_suppress_budget {
                    log::warn!("Thinking budget exhausted ({} tokens) without </think> — stop-token suppression lifted",
                        self.think_suppress_budget);
                }
            }
        }
    }

    /// Apply token suppression to logits (set to -inf).
    /// Called before sampling to prevent the model from generating turn-boundary tokens.
    /// `step` is the decode step number (0-based) — used for min_new_tokens.
    #[inline]
    fn apply_suppress_at_step(&self, logits: &mut [f32], vocab_size: usize, step: usize) {
        for &tok in &self.suppress_tokens {
            if tok < vocab_size {
                logits[tok] = f32::NEG_INFINITY;
            }
        }
        // Suppress stop tokens for the first min_new_tokens steps
        if step < self.min_new_tokens {
            for &tok in &self.stop_token_ids_for_suppress {
                if tok < vocab_size {
                    logits[tok] = f32::NEG_INFINITY;
                }
            }
        }
        // Suppress stop tokens until </think> has been generated,
        // but only up to the thinking budget (if set). After the budget is exhausted,
        // stop suppression is lifted so the model can terminate — prevents infinite
        // thinking loops from running for minutes.
        if self.think_end_token_for_suppress.is_some() && !self.think_end_seen {
            let within_budget = self.think_suppress_budget == 0
                || self.think_suppress_count < self.think_suppress_budget;
            if within_budget {
                for &tok in &self.stop_token_ids_for_suppress {
                    if tok < vocab_size {
                        logits[tok] = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }

    /// Create a Rust PrefillEngine from the registered weights in this decode store.
    /// This extracts all layer weight pointers, MoE expert data, and allocates prefill scratch.
    /// The returned engine can run prefill without any Python/GIL involvement.
    #[allow(dead_code)]
    pub fn create_prefill_engine(
        &self,
        max_tokens: usize,
    ) -> Result<crate::gpu_prefill::PrefillEngine, String> {
        use crate::gpu_prefill::*;

        let graph = self.graph.as_ref()
            .ok_or("GpuDecodeStore not configured")?;

        // Load prefill kernels
        let kernels = PrefillKernels::load(self.device.clone())?;

        // Create streams
        let prefill_stream = unsafe {
            let mut stream: cuda_sys::CUstream = std::ptr::null_mut();
            let err = cuda_sys::lib().cuStreamCreate(
                &mut stream,
                cuda_sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("Create prefill stream: {:?}", err));
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
                return Err(format!("Create prefill copy stream: {:?}", err));
            }
            stream
        };

        // Create cuBLAS handle
        let cublas_handle = cudarc::cublas::result::create_handle()
            .map_err(|e| format!("Create cuBLAS handle: {:?}", e))?;
        // Build model config
        let num_layers = graph.layers.len();
        let mut layer_types = vec![0u8; num_layers];
        let mut num_q_heads = 0usize;
        let mut num_kv_heads = 0usize;
        let mut head_dim = 0usize;

        // Detect dimensions from first GQA layer
        for l in &graph.layers {
            match &l.attn {
                GpuAttnConfig::GQA { num_heads, num_kv_heads: nkv, head_dim: hd, .. } => {
                    num_q_heads = *num_heads;
                    num_kv_heads = *nkv;
                    head_dim = *hd;
                    break;
                }
                _ => {}
            }
        }

        // Detect Mamba2 dimensions
        let mut mamba_d_inner = 0;
        let mut mamba_d_state = 0;
        let mut mamba_num_heads = 0;
        let mut mamba_head_dim = 0;
        let mut mamba_conv_dim = 0;
        let mut mamba_conv_kernel = 0;
        let mut mamba_n_groups = 1;

        // Detect linear attention dimensions
        let mut la_nk = 0usize; let mut la_nv = 0usize;
        let mut la_dk = 0usize; let mut la_dv = 0usize;
        let mut la_hr = 0usize; let mut la_kd = 0usize;
        let mut la_cd = 0usize; let mut la_scale = 0.0f32;

        for (i, l) in graph.layers.iter().enumerate() {
            match &l.attn {
                GpuAttnConfig::GQA { .. } => { layer_types[i] = 0; }
                GpuAttnConfig::Mamba2 { num_heads, head_dim: hd, state_size,
                                        expand, conv_kernel, conv_dim, .. } => {
                    layer_types[i] = 1;
                    mamba_d_inner = expand * hd * num_heads;
                    mamba_d_state = *state_size;
                    mamba_num_heads = *num_heads;
                    mamba_head_dim = *hd;
                    mamba_conv_dim = *conv_dim;
                    mamba_conv_kernel = *conv_kernel;
                    // n_groups: derive from Mamba2 config or default to 1
                    mamba_n_groups = 1; // TODO: should be stored in config
                }
                GpuAttnConfig::LinearAttention { nk, nv: nv_val, dk: dk_val, dv: dv_val,
                    hr, kernel_dim, conv_dim, scale, .. } =>
                {
                    layer_types[i] = 3;
                    la_nk = *nk; la_nv = *nv_val;
                    la_dk = *dk_val; la_dv = *dv_val;
                    la_hr = *hr; la_kd = *kernel_dim;
                    la_cd = *conv_dim; la_scale = *scale;
                }
                GpuAttnConfig::MLA { .. } => { layer_types[i] = 0; }
            }
        }

        // Detect MoE config from first MoE layer
        let mut n_routed = 0usize;
        let mut n_topk = 0usize;
        let mut scoring_func = 0u8;
        let mut norm_topk = false;
        let mut routed_sf = 1.0f32;
        let mut moe_gated = true;
        let mut moe_activation = 0u8;
        for ml in &graph.moe_layers {
            if let Some(m) = ml {
                n_routed = m.num_experts;
                n_topk = m.topk;
                scoring_func = m.scoring_func;
                norm_topk = m.norm_topk_prob;
                routed_sf = m.routed_scaling_factor;
                moe_gated = m.gated_experts;
                moe_activation = m.activation_type;
                break;
            }
        }

        let has_linear_attention = layer_types.iter().any(|&t| t == 3);

        let mut config = PrefillModelConfig {
            hidden_size: graph.hidden_size,
            intermediate_size: graph.moe_intermediate_size.max(graph.intermediate_size),
            moe_intermediate_size: graph.moe_intermediate_size,
            num_hidden_layers: num_layers,
            num_q_heads,
            num_kv_heads,
            head_dim,
            vocab_size: graph.vocab_size,
            rms_norm_eps: graph.eps,
            max_seq_len: max_tokens,
            n_routed_experts: n_routed,
            num_experts_per_tok: n_topk,
            expert_bits: graph.expert_bits,
            shared_expert_bits: graph.shared_expert_bits,
            group_size: graph.group_size,
            sms: graph.num_sms,
            device_ordinal: self.gpu_index(),
            layer_types,
            first_k_dense: 0, // TODO: from model config
            scoring_func,
            norm_topk_prob: norm_topk,
            routed_scaling_factor: routed_sf,
            moe_gated,
            moe_activation,
            mamba_d_inner,
            mamba_d_state,
            mamba_num_heads,
            mamba_head_dim,
            mamba_conv_dim,
            mamba_conv_kernel,
            mamba_n_groups,
            tie_word_embeddings: false,
            la_num_k_heads: la_nk,
            la_num_v_heads: la_nv,
            la_k_head_dim: la_dk,
            la_v_head_dim: la_dv,
            la_head_ratio: la_hr.max(1),
            la_conv_kernel_dim: la_kd,
            la_conv_dim: la_cd,
            la_scale,
            la_chunk_size: 64,
            rope_half_dim: graph.rope_half_dim,
            prefill_chunk_size: 0, // computed dynamically below
            layer_group_size: 4,      // group MoE layers for expert DMA pipelining
            fused_moe_w1_ctmp_floats: 0,
            fused_moe_w2_ctmp_floats: 0,
        };
        let (fused_moe_w1_ctmp_floats, fused_moe_w2_ctmp_floats) =
            fused_moe_ctmp_floats_for_config(&config);
        config.fused_moe_w1_ctmp_floats = fused_moe_w1_ctmp_floats;
        config.fused_moe_w2_ctmp_floats = fused_moe_w2_ctmp_floats;

        // Scratch allocation is DYNAMIC per-prompt via prepare_for_prefill/release_scratch.
        // At init, allocate minimal 1-token scratch so HCS gets maximum VRAM.
        // Before each prefill, prepare_for_prefill() reallocates to the actual prompt size.
        // After each prefill, release_scratch() shrinks back and frees VRAM for decode HCS.
        config.prefill_chunk_size = 0; // Set dynamically per prompt
        if stderr_debug_enabled() {
            eprintln!("[PREFILL] Dynamic scratch allocation enabled (per-prompt sizing)");
        }

        // Build per-layer weights
        let mut layer_weights = Vec::with_capacity(num_layers);
        let extract_marlin = |wid: usize| -> Option<MarlinWeight> {
            let w = &graph.weights[wid];
            if w.dtype == 4 || w.dtype == 5 {
                Some(MarlinWeight {
                    packed: w.ptr,
                    scales: w.scales_ptr,
                    n: w.rows,
                    k: w.cols,
                    num_groups: w.cols / w.group_size.max(1),
                    group_size: w.group_size,
                    num_bits: if w.dtype == 4 { 8 } else { 4 },
                })
            } else {
                None
            }
        };
        let extract_bf16 = |wid: usize| -> Option<Bf16Weight> {
            let w = &graph.weights[wid];
            if w.dtype == 0 {  // BF16
                Some(Bf16Weight { ptr: w.ptr, n: w.rows, k: w.cols })
            } else {
                None
            }
        };

        for (i, l) in graph.layers.iter().enumerate() {
            let mut lw = PrefillLayerWeights {
                input_norm: l.input_norm_ptr,
                post_attn_norm: l.post_attn_norm_ptr,
                q_proj: None, k_proj: None, v_proj: None, o_proj: None,
                q_proj_bf16: None, k_proj_bf16: None, v_proj_bf16: None, o_proj_bf16: None,
                gqa_gated: false,
                mamba2_in_proj: None, mamba2_out_proj: None,
                mamba2_conv_weight: 0, mamba2_conv_bias: 0,
                mamba2_A: 0, mamba2_D: 0, mamba2_dt_bias: 0, mamba2_norm: 0,
                moe_gate_ptr: 0, moe_gate_rows: 0, moe_gate_cols: 0,
                moe_gate_bias_ptr: 0, moe_e_score_corr_ptr: 0,
                moe_num_experts: 0, moe_topk: 0,
                moe_scoring_func: 0, moe_norm_topk_prob: false,
                moe_routed_scaling_factor: 1.0,
                moe_gated: true, moe_activation: 0,
                shared_w1: None, shared_w2: None,
                shared_w1_bf16: None, shared_w2_bf16: None,
                shared_gate_ptr: 0, shared_gate_rows: 0, shared_gate_cols: 0,
                layer_type: config.layer_types[i],
                moe_layer_idx: None,
                la_in_proj_qkvz: None, la_in_proj_ba: None, la_out_proj: None,
                la_in_proj_qkvz_bf16: None, la_in_proj_ba_bf16: None, la_out_proj_bf16: None,
                la_conv_weight_ptr: 0, la_a_log_ptr: 0, la_dt_bias_ptr: 0,
                la_norm_weight_ptr: 0, la_conv_state_ptr: 0, la_recur_state_ptr: 0,
                q_norm_ptr: 0, k_norm_ptr: 0,
            };

            match &l.attn {
                GpuAttnConfig::GQA { q_proj, k_proj, v_proj, o_proj, gated,
                                     q_norm_ptr, k_norm_ptr, .. } => {
                    lw.q_proj = extract_marlin(*q_proj);
                    lw.k_proj = extract_marlin(*k_proj);
                    lw.v_proj = extract_marlin(*v_proj);
                    lw.o_proj = extract_marlin(*o_proj);
                    // BF16 fallback for attention_quant="bf16"
                    if lw.q_proj.is_none() { lw.q_proj_bf16 = extract_bf16(*q_proj); }
                    if lw.k_proj.is_none() { lw.k_proj_bf16 = extract_bf16(*k_proj); }
                    if lw.v_proj.is_none() { lw.v_proj_bf16 = extract_bf16(*v_proj); }
                    if lw.o_proj.is_none() { lw.o_proj_bf16 = extract_bf16(*o_proj); }
                    lw.gqa_gated = *gated;
                    lw.q_norm_ptr = *q_norm_ptr;
                    lw.k_norm_ptr = *k_norm_ptr;
                }
                GpuAttnConfig::Mamba2 { in_proj, out_proj,
                    conv_weight_ptr, a_ptr, d_ptr, dt_bias_ptr, norm_weight_ptr, .. } =>
                {
                    lw.mamba2_in_proj = extract_marlin(*in_proj);
                    lw.mamba2_out_proj = extract_marlin(*out_proj);
                    lw.mamba2_conv_weight = *conv_weight_ptr;
                    lw.mamba2_A = *a_ptr;
                    lw.mamba2_D = *d_ptr;
                    lw.mamba2_dt_bias = *dt_bias_ptr;
                    lw.mamba2_norm = *norm_weight_ptr;
                }
                GpuAttnConfig::LinearAttention { in_proj_qkvz, in_proj_ba, out_proj,
                    conv_weight_ptr, a_log_ptr, dt_bias_ptr, norm_weight_ptr,
                    conv_state_ptr, recur_state_ptr, .. } =>
                {
                    lw.la_in_proj_qkvz = extract_marlin(*in_proj_qkvz);
                    lw.la_in_proj_ba = extract_marlin(*in_proj_ba);
                    lw.la_out_proj = extract_marlin(*out_proj);
                    // BF16 fallback for attention_quant="bf16"
                    if lw.la_in_proj_qkvz.is_none() {
                        lw.la_in_proj_qkvz_bf16 = extract_bf16(*in_proj_qkvz);
                    }
                    if lw.la_in_proj_ba.is_none() {
                        lw.la_in_proj_ba_bf16 = extract_bf16(*in_proj_ba);
                    }
                    if lw.la_out_proj.is_none() {
                        lw.la_out_proj_bf16 = extract_bf16(*out_proj);
                    }
                    lw.la_conv_weight_ptr = *conv_weight_ptr;
                    lw.la_a_log_ptr = *a_log_ptr;
                    lw.la_dt_bias_ptr = *dt_bias_ptr;
                    lw.la_norm_weight_ptr = *norm_weight_ptr;
                    lw.la_conv_state_ptr = *conv_state_ptr;
                    lw.la_recur_state_ptr = *recur_state_ptr;
                }
                _ => {} // MLA: TODO
            }

            // MoE config
            if i < 3 && stderr_debug_enabled() {
                eprintln!("[PREFILL-INIT] layer {} mlp={}, moe_layers.len()={}, moe_layers[{}]={:?}",
                    i, match &l.mlp { GpuMlpConfig::MoE{..} => "MoE", GpuMlpConfig::Dense{..} => "Dense", GpuMlpConfig::None => "None" },
                    graph.moe_layers.len(), i,
                    graph.moe_layers.get(i).map(|x| x.is_some()));
            }
            match &l.mlp {
                GpuMlpConfig::MoE { gate_weight, gate_bias_ptr, e_score_corr_ptr,
                    num_experts, topk, scoring_func: sf, norm_topk_prob: ntp,
                    routed_scaling_factor: rsf,
                    shared_gate_up, shared_down, .. } =>
                {
                    let gw = &graph.weights[*gate_weight];
                    lw.moe_gate_ptr = gw.ptr;
                    lw.moe_gate_rows = gw.rows;
                    lw.moe_gate_cols = gw.cols;
                    lw.moe_gate_bias_ptr = *gate_bias_ptr;
                    lw.moe_e_score_corr_ptr = *e_score_corr_ptr;
                    lw.moe_num_experts = *num_experts;
                    lw.moe_topk = *topk;
                    lw.moe_scoring_func = *sf;
                    lw.moe_norm_topk_prob = *ntp;
                    lw.moe_routed_scaling_factor = *rsf;
                    lw.moe_gated = moe_gated;
                    lw.moe_activation = moe_activation;
                    lw.moe_layer_idx = Some(i);
                    if let Some(wid) = shared_gate_up {
                        lw.shared_w1 = extract_marlin(*wid);
                    }
                    if let Some(wid) = shared_down {
                        lw.shared_w2 = extract_marlin(*wid);
                    }
                }
                GpuMlpConfig::Dense { gate_proj, up_proj: _, down_proj } => {
                    // For dense MLP, gate_proj serves as w1 (gate_up)
                    lw.shared_w1 = extract_marlin(*gate_proj);
                    lw.shared_w2 = extract_marlin(*down_proj);
                }
                GpuMlpConfig::None => {
                    // MoE layers use GpuMlpConfig::None as placeholder — actual MoE data
                    // is in graph.moe_layers[i]. Bridge the gap here.
                    if let Some(Some(moe_data)) = graph.moe_layers.get(i) {
                        let gw = &graph.weights[moe_data.gate_wid];
                        lw.moe_gate_ptr = gw.ptr;
                        lw.moe_gate_rows = gw.rows;
                        lw.moe_gate_cols = gw.cols;
                        lw.moe_gate_bias_ptr = moe_data.gate_bias_ptr;
                        lw.moe_e_score_corr_ptr = moe_data.e_score_corr_ptr;
                        lw.moe_num_experts = moe_data.num_experts;
                        lw.moe_topk = moe_data.topk;
                        lw.moe_scoring_func = moe_data.scoring_func;
                        lw.moe_norm_topk_prob = moe_data.norm_topk_prob;
                        lw.moe_routed_scaling_factor = moe_data.routed_scaling_factor;
                        lw.moe_gated = moe_data.gated_experts;
                        lw.moe_activation = moe_data.activation_type;
                        lw.moe_layer_idx = Some(i);
                        // Shared expert gate weight (sigmoid gating of shared expert output)
                        if let Some(sg_wid) = moe_data.shared_gate_wid {
                            let sg_w = &graph.weights[sg_wid];
                            lw.shared_gate_ptr = sg_w.ptr;
                            lw.shared_gate_rows = sg_w.rows;
                            lw.shared_gate_cols = sg_w.cols;
                        }
                        // Shared expert weights: look for pinned VRAM copies first
                        if let Some(Some(se_vram)) = graph.shared_expert_vram.get(i) {
                            let bits = graph.shared_expert_bits;
                            if bits == 16 {
                                // BF16 validation mode: raw BF16 data, use cuBLAS
                                // For BF16: w13_packed is [w13_n, hidden] raw BF16, total bytes = w13_n * hidden * 2
                                // So w13_n = w13_packed_bytes / (hidden * 2)
                                let shared_inter_w13_n = se_vram.w13_packed_size / (graph.hidden_size * 2);
                                let w13_n = shared_inter_w13_n;
                                let shared_inter = if moe_data.gated_experts { w13_n / 2 } else { w13_n };
                                lw.shared_w1_bf16 = Some(Bf16Weight {
                                    ptr: se_vram.w13_packed_ptr(),
                                    n: w13_n,
                                    k: graph.hidden_size,
                                });
                                lw.shared_w2_bf16 = Some(Bf16Weight {
                                    ptr: se_vram.w2_packed_ptr(),
                                    n: graph.hidden_size,
                                    k: shared_inter,
                                });
                            } else {
                                lw.shared_w1 = Some(MarlinWeight {
                                    packed: se_vram.w13_packed_ptr(),
                                    scales: se_vram.w13_scales_ptr(),
                                    n: graph.moe_intermediate_size * 2,
                                    k: graph.hidden_size,
                                    num_groups: graph.hidden_size / graph.group_size.max(1),
                                    group_size: graph.group_size,
                                    num_bits: bits,
                                });
                                lw.shared_w2 = Some(MarlinWeight {
                                    packed: se_vram.w2_packed_ptr(),
                                    scales: se_vram.w2_scales_ptr(),
                                    n: graph.hidden_size,
                                    k: graph.moe_intermediate_size,
                                    num_groups: graph.moe_intermediate_size / graph.group_size.max(1),
                                    group_size: graph.group_size,
                                    num_bits: bits,
                                });
                            }
                        }
                    }
                }
            }

            if i < 3 && stderr_debug_enabled() {
                eprintln!("[PREFILL-INIT] layer {} -> moe_gate_ptr={:#x}, shared_w1={}, shared_w2={}, moe_layer_idx={:?}",
                    i, lw.moe_gate_ptr, lw.shared_w1.is_some(), lw.shared_w2.is_some(), lw.moe_layer_idx);
            }
            layer_weights.push(lw);
        }

        // Convert QK norm weights from FP32 (decode format) to BF16 (prefill kernel format).
        // The decode path stores these as FP32 on GPU, but the prefill rmsnorm kernel expects BF16.
        let mut qk_norm_bf16_bufs: Vec<CudaSlice<u16>> = Vec::new();
        for lw in layer_weights.iter_mut() {
            for norm_ptr in [&mut lw.q_norm_ptr, &mut lw.k_norm_ptr] {
                if *norm_ptr != 0 {
                    let d = head_dim;
                    // Download FP32 from GPU
                    let mut h_fp32 = vec![0.0f32; d];
                    unsafe {
                        let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                            h_fp32.as_mut_ptr() as *mut std::ffi::c_void,
                            *norm_ptr, (d * 4) as usize,
                        );
                        if err != cuda_sys::CUresult::CUDA_SUCCESS {
                            return Err(format!("QK norm FP32 download: {:?}", err));
                        }
                    }
                    // Convert FP32 -> BF16 on CPU
                    let h_bf16: Vec<u16> = h_fp32.iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect();
                    // Upload BF16 to GPU
                    let d_bf16 = self.device.htod_copy(h_bf16)
                        .map_err(|e| format!("QK norm BF16 upload: {e}"))?;
                    *norm_ptr = *d_bf16.device_ptr();
                    qk_norm_bf16_bufs.push(d_bf16);
                }
            }
        }

        // Build MoE expert data
        let mut moe_layers: Vec<Option<PrefillMoeLayerData>> = Vec::new();
        for ml in &graph.moe_layers {
            moe_layers.push(ml.as_ref().map(|m| {
                let experts: Vec<ExpertWeightPtrs> = m.experts.iter().map(|e| ExpertWeightPtrs {
                    w13_packed_ptr: e.w13_packed_ptr,
                    w13_packed_bytes: e.w13_packed_bytes,
                    w13_scales_ptr: e.w13_scales_ptr,
                    w13_scales_bytes: e.w13_scales_bytes,
                    w2_packed_ptr: e.w2_packed_ptr,
                    w2_packed_bytes: e.w2_packed_bytes,
                    w2_scales_ptr: e.w2_scales_ptr,
                    w2_scales_bytes: e.w2_scales_bytes,
                    contiguous_ptr: e.contiguous_ptr,
                    contiguous_bytes: e.contiguous_bytes,
                }).collect();
                let shared = m.shared.as_ref().map(|e| ExpertWeightPtrs {
                    w13_packed_ptr: e.w13_packed_ptr,
                    w13_packed_bytes: e.w13_packed_bytes,
                    w13_scales_ptr: e.w13_scales_ptr,
                    w13_scales_bytes: e.w13_scales_bytes,
                    w2_packed_ptr: e.w2_packed_ptr,
                    w2_packed_bytes: e.w2_packed_bytes,
                    w2_scales_ptr: e.w2_scales_ptr,
                    w2_scales_bytes: e.w2_scales_bytes,
                    contiguous_ptr: e.contiguous_ptr,
                    contiguous_bytes: e.contiguous_bytes,
                });
                // Compute bulk DMA pointers from first expert + count.
                // The LayerExpertBacking stores all experts contiguously per component,
                // so expert[0]'s ptr is the start and n * per_expert is the total.
                let ne = experts.len();
                let (bulk_w13p, bulk_w13s, bulk_w2p, bulk_w2s) = if ne > 0 {
                    let e0 = &experts[0];
                    // Verify contiguity: last expert's ptr should be at expected offset
                    let contiguous = if ne > 1 {
                        let el = &experts[ne - 1];
                        el.w13_packed_ptr == e0.w13_packed_ptr + (ne - 1) * e0.w13_packed_bytes
                        && el.w13_scales_ptr == e0.w13_scales_ptr + (ne - 1) * e0.w13_scales_bytes
                        && el.w2_packed_ptr == e0.w2_packed_ptr + (ne - 1) * e0.w2_packed_bytes
                        && el.w2_scales_ptr == e0.w2_scales_ptr + (ne - 1) * e0.w2_scales_bytes
                    } else { true };
                    if contiguous {
                        (
                            (e0.w13_packed_ptr, ne * e0.w13_packed_bytes),
                            (e0.w13_scales_ptr, ne * e0.w13_scales_bytes),
                            (e0.w2_packed_ptr, ne * e0.w2_packed_bytes),
                            (e0.w2_scales_ptr, ne * e0.w2_scales_bytes),
                        )
                    } else {
                        log::warn!("MoE layer experts not contiguous — bulk DMA disabled");
                        ((0, 0), (0, 0), (0, 0), (0, 0))
                    }
                } else {
                    ((0, 0), (0, 0), (0, 0), (0, 0))
                };
                PrefillMoeLayerData { experts, shared, bulk_w13p, bulk_w13s, bulk_w2p, bulk_w2s }
            }));
        }

        // Allocate minimal scratch at init (0 tokens = just fixed-size buffers).
        // prepare_for_prefill() dynamically resizes before each prompt.
        // release_scratch() frees back to minimal after prefill for max HCS.
        // GpuBuf uses raw cuMemAlloc_v2 (no cudarc pool) so alloc/free is
        // synchronous and deterministic — no pool interaction with HCS.
        let scratch = allocate_scratch(&self.device, &config, 0)?;
        config.prefill_chunk_size = 0;
        if stderr_debug_enabled() {
            eprintln!("[PREFILL] Dynamic scratch: allocated minimal at init (sized per-prompt)");
        }

        // Create CUDA events for double-buffer synchronization
        let dma_event = unsafe {
            let mut event: cuda_sys::CUevent = std::ptr::null_mut();
            let err = cuda_sys::lib().cuEventCreate(
                &mut event,
                cuda_sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("Create DMA event: {:?}", err));
            }
            event
        };
        let compute_event = unsafe {
            let mut event: cuda_sys::CUevent = std::ptr::null_mut();
            let err = cuda_sys::lib().cuEventCreate(
                &mut event,
                cuda_sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("Create compute event: {:?}", err));
            }
            event
        };

        // Snapshot current HCS state (if available)
        let (hcs_cache_fast, hcs_ne) = if let Some(ref hcs) = graph.hcs {
            (hcs.cache_fast.clone(), hcs.num_experts_per_layer)
        } else {
            (Vec::new(), 0)
        };

        // Create shared expert stream and event (gap 4: always-async shared expert)
        let shared_stream = unsafe {
            let mut stream: cuda_sys::CUstream = std::ptr::null_mut();
            let err = cuda_sys::lib().cuStreamCreate(
                &mut stream,
                cuda_sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("Create shared expert stream: {:?}", err));
            }
            stream
        };
        let shared_cublas_handle = cudarc::cublas::result::create_handle()
            .map_err(|e| format!("Create shared cuBLAS handle: {:?}", e))?;
        unsafe {
            cudarc::cublas::result::set_stream(
                shared_cublas_handle,
                shared_stream as cudarc::cublas::sys::cudaStream_t,
            ).map_err(|e| format!("Set shared cuBLAS stream: {:?}", e))?;
        }
        let shared_event = unsafe {
            let mut event: cuda_sys::CUevent = std::ptr::null_mut();
            let err = cuda_sys::lib().cuEventCreate(
                &mut event,
                cuda_sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("Create shared event: {:?}", err));
            }
            event
        };

        // Shared expert scratch and workspace are allocated dynamically in
        // prepare_for_prefill and freed in release_scratch. This keeps ~400 MB
        // of VRAM available for HCS during decode instead of wasting it on
        // buffers that are only used during prefill.
        let max_possible_chunk: usize = 50000;

        // Compute per-expert byte sizes for fused MoE
        // Uses moe_intermediate_size (per-expert), not intermediate_size (dense MLP max)
        let moe_inter = config.moe_intermediate_size;
        let h = graph.hidden_size;
        let gs = config.group_size;
        let bits = config.expert_bits as usize;

        let w1_n = if config.moe_gated { 2 * moe_inter } else { moe_inter };
        // Marlin packed layout: [K/16, N, 16*bits/8] bytes per expert
        // Total = K * N * bits / 8 bytes
        let w1_packed_per_expert = if bits == 16 { h * w1_n * 2 } else { h * w1_n * bits / 8 };
        let w1_scales_per_expert = if gs == 0 { 0 } else { (h / gs) * w1_n * 2 };
        let w2_packed_per_expert = if bits == 16 { moe_inter * h * 2 } else { moe_inter * h * bits / 8 };
        let w2_scales_per_expert = if gs == 0 { 0 } else { (moe_inter / gs) * h * 2 };

        // Fused MoE contiguous buffers: [E, per_expert_bytes] for all experts
        // Only allocate if we have MoE layers and the fused kernel is available
        // AND there is enough free VRAM. Otherwise fall back to per-expert sequential dispatch.
        let has_moe = n_routed > 0;
        let has_fused_kernel = kernels.fused_moe_fn.is_some();
        let can_fused = has_moe && has_fused_kernel;

        // Cold staging + pointer tables are the PRIMARY approach for fused MoE.
        // They allow zero-copy HCS expert access and H2D staging without contiguous buffers.
        // Fused expert weight buffers (A/B sets) are SECONDARY — only needed as overflow
        // when cold staging is full, or for the legacy non-pointer-table path.
        let total_w1_packed = n_routed * w1_packed_per_expert;
        let total_w1_scales = n_routed * w1_scales_per_expert;
        let total_w2_packed = n_routed * w2_packed_per_expert;
        let total_w2_scales = n_routed * w2_scales_per_expert;
        let cold_per_expert = w1_packed_per_expert + w1_scales_per_expert
            + w2_packed_per_expert + w2_scales_per_expert;

        let max_sorted = scratch.fused_sorted_count;

        // Cold staging is allocated dynamically in prepare_for_prefill and freed in
        // release_scratch. This keeps ~792 MB (QCN) of VRAM available for HCS during
        // decode. The cost is budgeted in compute_scratch_vram as a fixed cost.
        let actual_max_cold = if can_fused { n_routed } else { 0 };

        // Fused expert weight buffers: not needed when cold staging covers all experts
        // (pointer table mode). Cold staging is dynamically allocated in prepare_for_prefill.
        let (d_fused_expert_w1_a, d_fused_expert_w1s_a, d_fused_expert_w2_a, d_fused_expert_w2s_a,
         d_fused_expert_w1_b, d_fused_expert_w1s_b, d_fused_expert_w2_b, d_fused_expert_w2s_b) =
            (None, None, None, None, None, None, None, None);

        // Build engine
        let topk = config.num_experts_per_tok.max(1);
        let q_type = if config.expert_bits == 8 {
            crate::gpu_prefill::ScalarType::U8B128
        } else {
            crate::gpu_prefill::ScalarType::U4B8
        };
        Ok(PrefillEngine {
            device: self.device.clone(),
            kernels,
            config,
            scratch,
            layer_weights,
            moe_layers,
            embedding_ptr: graph.embedding_ptr,
            final_norm_ptr: graph.final_norm_ptr,
            lm_head: extract_marlin(graph.lm_head_wid),
            lm_head_bf16_ptr: {
                let w = &graph.weights[graph.lm_head_wid];
                if w.dtype == 0 { w.ptr } else { 0 }
            },
            lm_head_bf16_rows: {
                let w = &graph.weights[graph.lm_head_wid];
                if w.dtype == 0 { w.rows } else { 0 }
            },
            lm_head_bf16_cols: {
                let w = &graph.weights[graph.lm_head_wid];
                if w.dtype == 0 { w.cols } else { 0 }
            },
            rope_cos_ptr: graph.d_rope_cos.as_ref().map_or(0, |c| *c.device_ptr()),
            rope_sin_ptr: graph.d_rope_sin.as_ref().map_or(0, |c| *c.device_ptr()),
            kv_k_ptrs: graph.kv_k_ptrs.clone(),
            kv_v_ptrs: graph.kv_v_ptrs.clone(),
            kv_max_seq: graph.kv_max_seq,
            kv_format: graph.kv_format,
            kv_k_radius_ptrs: graph.kv_k_radius_ptrs.clone(),
            kv_v_radius_ptrs: graph.kv_v_radius_ptrs.clone(),
            kv_k_angles_ptrs: graph.kv_k_angles_ptrs.clone(),
            kv_v_angles_ptrs: graph.kv_v_angles_ptrs.clone(),
            kv_num_blocks: graph.kv_num_blocks,
            stream: prefill_stream,
            copy_stream,
            cublas_handle,
            shared_cublas_handle,
            h_logits: Vec::new(),
            h_topk_ids: vec![0i32; max_possible_chunk * topk],
            h_topk_weights: vec![0.0f32; max_possible_chunk * topk],
            h_gather_src_map: Vec::new(),
            h_gather_weight_map: Vec::new(),
            hcs_cache_fast,
            hcs_num_experts_per_layer: hcs_ne,
            dma_event,
            compute_event,
            attn_weight_bufs: None,
            shared_stream,
            shared_event,
            d_shared_fp32_scratch: None,  // allocated dynamically in prepare_for_prefill
            d_shared_workspace: None,     // allocated dynamically in prepare_for_prefill
            d_fused_expert_w1_a,
            d_fused_expert_w1s_a,
            d_fused_expert_w2_a,
            d_fused_expert_w2s_a,
            d_fused_expert_w1_b,
            d_fused_expert_w1s_b,
            d_fused_expert_w2_b,
            d_fused_expert_w2s_b,
            fused_expert_buf_cur: 0,
            w1_packed_per_expert,
            w1_scales_per_expert,
            w2_packed_per_expert,
            w2_scales_per_expert,
            max_sorted,
            preloaded_moe_layer: None,
            pinning_pool_ptr: 0,
            pinning_pool_bytes: 0,
            pinned_expert_offsets: Vec::new(),
            pinning_pool_expert_bytes: 0,
            pinning_active: false,
            prescan_active_experts: Vec::new(),
            // Expert pointer table for zero-copy MoE
            d_expert_w1_ptrs: if can_fused {
                Some(self.device.alloc_zeros::<u64>(n_routed.max(1))
                    .map_err(|e| format!("alloc expert_w1_ptrs: {e}"))?)
            } else { None },
            d_expert_w1s_ptrs: if can_fused {
                Some(self.device.alloc_zeros::<u64>(n_routed.max(1))
                    .map_err(|e| format!("alloc expert_w1s_ptrs: {e}"))?)
            } else { None },
            d_expert_w2_ptrs: if can_fused {
                Some(self.device.alloc_zeros::<u64>(n_routed.max(1))
                    .map_err(|e| format!("alloc expert_w2_ptrs: {e}"))?)
            } else { None },
            d_expert_w2s_ptrs: if can_fused {
                Some(self.device.alloc_zeros::<u64>(n_routed.max(1))
                    .map_err(|e| format!("alloc expert_w2s_ptrs: {e}"))?)
            } else { None },
            h_expert_w1_ptrs: vec![0u64; n_routed.max(1)],
            h_expert_w1s_ptrs: vec![0u64; n_routed.max(1)],
            h_expert_w2_ptrs: vec![0u64; n_routed.max(1)],
            h_expert_w2s_ptrs: vec![0u64; n_routed.max(1)],
            d_cold_staging: None,  // allocated dynamically in prepare_for_prefill
            cold_expert_bytes: cold_per_expert,
            max_cold_experts: actual_max_cold,
            q_type,
            qk_norm_bf16_bufs,
            gqa_timing_enabled: std::cell::Cell::new(false),
            t_gqa_proj: std::cell::Cell::new(0.0),
            t_gqa_norm: std::cell::Cell::new(0.0),
            t_gqa_rope: std::cell::Cell::new(0.0),
            t_gqa_kv_prep: std::cell::Cell::new(0.0),
            t_gqa_fa2: std::cell::Cell::new(0.0),
            t_gqa_gate: std::cell::Cell::new(0.0),
            t_gqa_oproj: std::cell::Cell::new(0.0),
            gqa_fa2_calls: std::cell::Cell::new(0),
            gqa_fp8_calls: std::cell::Cell::new(0),
            gqa_bf16_calls: std::cell::Cell::new(0),
            t_la_proj: std::cell::Cell::new(0.0),
            t_la_uninterleave: std::cell::Cell::new(0.0),
            t_la_conv: std::cell::Cell::new(0.0),
            t_la_prep: std::cell::Cell::new(0.0),
            t_la_convert: std::cell::Cell::new(0.0),
            t_la_fla: std::cell::Cell::new(0.0),
            t_la_postfla: std::cell::Cell::new(0.0),
            t_la_norm: std::cell::Cell::new(0.0),
            t_la_oproj: std::cell::Cell::new(0.0),
            t_moe_gate: std::cell::Cell::new(0.0),
            t_moe_dma: std::cell::Cell::new(0.0),
            t_moe_w1: std::cell::Cell::new(0.0),
            t_moe_w2: std::cell::Cell::new(0.0),
            t_moe_scatter: std::cell::Cell::new(0.0),
            t_moe_shared: std::cell::Cell::new(0.0),
            fla: crate::gpu_prefill::load_fla(has_linear_attention)?,
            d_fla_g_cumsum: None,
            d_fla_a: None,
            d_fla_ai: None,
            d_fla_w: None,
            d_fla_u: None,
            d_fla_h: None,
            d_fla_final_state: None,
            d_fla_v_new: None,
            d_fla_o: None,
            safety_margin_mb: self.vram_calibration
                .as_ref()
                .map(|cal| cal.safety_margin_mb as usize)
                .unwrap_or(crate::gpu_prefill::PREFILL_SAFETY_MARGIN_MB),
            prefill_hcs_store_addr: 0,
        })
    }

    /// Upload Marlin INT8 repacked weights to GPU and register.
    fn _finish_register_marlin_int8(
        &mut self, m: crate::weights::marlin::MarlinRepacked,
        n: usize, k: usize, group_size: usize,
    ) -> pyo3::PyResult<usize> {
        let d_packed = self.device.htod_copy(m.packed)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to upload Marlin INT8 packed: {:?}", e)))?;
        let packed_ptr = *d_packed.device_ptr();

        let d_scales = self.device.htod_copy(m.scales)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to upload Marlin INT8 scales: {:?}", e)))?;
        let scales_ptr = *d_scales.device_ptr();

        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let id = graph.weights.len();
        graph.weights.push(GpuWeight::new_marlin_int8(packed_ptr, scales_ptr, n, k, group_size));

        std::mem::forget(d_packed);
        std::mem::forget(d_scales);

        log::info!(
            "Registered Marlin INT8 weight id={}: [{n}×{k}] gs={group_size}, packed={:.1} MB, scales={:.1} KB",
            id,
            (n * k / 4) as f64 * 4.0 / 1024.0 / 1024.0,
            (k / group_size * n) as f64 * 2.0 / 1024.0,
        );

        Ok(id)
    }

    /// Upload Marlin INT4 repacked weights to GPU and register.
    fn _finish_register_marlin_int4(
        &mut self, m: crate::weights::marlin::MarlinRepacked,
        n: usize, k: usize, group_size: usize,
    ) -> pyo3::PyResult<usize> {
        let d_packed = self.device.htod_copy(m.packed)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to upload Marlin INT4 packed: {:?}", e)))?;
        let packed_ptr = *d_packed.device_ptr();

        let d_scales = self.device.htod_copy(m.scales)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to upload Marlin INT4 scales: {:?}", e)))?;
        let scales_ptr = *d_scales.device_ptr();

        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Call configure first"))?;
        let id = graph.weights.len();
        graph.weights.push(GpuWeight::new_marlin_int4(packed_ptr, scales_ptr, n, k, group_size));

        std::mem::forget(d_packed);
        std::mem::forget(d_scales);

        log::info!(
            "Registered Marlin INT4 weight id={}: [{n}×{k}] gs={group_size}, packed={:.1} MB, scales={:.1} KB",
            id,
            (n * k / 8) as f64 * 4.0 / 1024.0 / 1024.0,
            (k / group_size * n) as f64 * 2.0 / 1024.0,
        );

        Ok(id)
    }

    /// Query current VRAM free in MB via cuMemGetInfo_v2.
    pub fn query_vram_free_mb(&self) -> usize {
        let mut free: usize = 0;
        let mut _total: usize = 0;
        unsafe { let _ = cuda_sys::lib().cuMemGetInfo_v2(&mut free, &mut _total); }
        free / (1024 * 1024)
    }

    /// Return benchmark stats: (min_free_vram_mb, hcs_loaded, hcs_total, hcs_pct)
    /// Only counts MoE experts within this store's decode segment [decode_layer_start..decode_layer_end).
    pub fn benchmark_stats(&self) -> (usize, usize, usize, f64) {
        let min_free = self.last_min_free_vram_mb;
        let (loaded, total, pct) = if let Some(graph) = self.graph.as_ref() {
            if let Some(hcs) = graph.hcs.as_ref() {
                let seg_start = graph.decode_layer_start;
                let seg_end = graph.decode_layer_end;
                let total: usize = graph.moe_layers.iter()
                    .enumerate()
                    .filter(|(i, _)| *i >= seg_start && *i < seg_end)
                    .filter_map(|(_, m)| m.as_ref())
                    .map(|m| m.num_experts)
                    .sum();
                let loaded = hcs.num_cached;
                let pct = if total > 0 { loaded as f64 / total as f64 * 100.0 } else { 0.0 };
                (loaded, total, pct)
            } else {
                (0, 0, 0.0)
            }
        } else {
            (0, 0, 0.0)
        };
        (min_free, loaded, total, pct)
    }

    pub fn hcs_soft_reload_pending(&self) -> bool {
        self.graph.as_ref()
            .and_then(|g| g.hcs.as_ref())
            .map(|h| h.soft_reload_pending)
            .unwrap_or(false)
    }

    /// Return HCS safety margin in MB (0 if HCS not configured).
    pub fn hcs_safety_margin_mb(&self) -> usize {
        self.graph.as_ref()
            .and_then(|g| g.hcs.as_ref())
            .map(|hcs| hcs.safety_margin_mb)
            .unwrap_or(0)
    }

    fn validation_capture_decode_start(&mut self, prompt_len: usize) {
        let Some(graph) = self.graph.as_mut() else { return; };
        graph.validation_decode_start_num_cached = 0;
        graph.validation_decode_start_soft_num_cached = 0;
        graph.validation_decode_start_hard_num_cached = 0;
        graph.validation_decode_start_dupes = 0;
        graph.validation_decode_start_soft_dupes = 0;
        graph.validation_decode_start_hash.clear();
        graph.validation_decode_start_resident.clear();
        graph.validation_decode_start_slots.clear();
        graph.validation_decode_cold_hist.clear();
        graph.validation_decode_cold_events.clear();

        let (hcs_file, slots_file, cold_file, cold_events_file) = validation_artifact_paths(prompt_len);
        graph.validation_decode_start_hcs_file = hcs_file;
        graph.validation_decode_start_slots_file = slots_file;
        graph.validation_decode_cold_file = cold_file;
        graph.validation_decode_cold_events_file = cold_events_file;

        if let Some(hcs) = graph.hcs.as_ref() {
            let resident = hcs.validation_sorted_resident_experts();
            let (_, _, dupes, soft_dupes) = hcs.validation_counts();
            graph.validation_decode_start_num_cached = hcs.num_cached;
            graph.validation_decode_start_soft_num_cached = hcs.soft_num_cached;
            graph.validation_decode_start_hard_num_cached = hcs.num_cached.saturating_sub(hcs.soft_num_cached);
            graph.validation_decode_start_dupes = dupes;
            graph.validation_decode_start_soft_dupes = soft_dupes;
            graph.validation_decode_start_hash = validation_hash_pairs(&resident);
            graph.validation_decode_start_resident = resident;
            graph.validation_decode_start_slots = hcs.validation_slot_entries();
        }
    }

    pub fn config_validation_snapshot_json(
        &self,
        prompt_len: usize,
        sync_reload: bool,
        reload_pending_at_decode_start: bool,
    ) -> String {
        let mut payload = serde_json::json!({
            "prompt_len": prompt_len,
            "sync_reload": sync_reload,
            "soft_evict_experts": self.last_soft_evict_experts,
            "soft_evict_freed_mb": self.last_soft_evict_freed_mb,
            "soft_reload_queued": self.last_soft_reload_queued,
            "soft_reload_alloc_mb": self.last_soft_reload_alloc_mb,
            "soft_reload_activated": self.last_soft_reload_activated,
            "reload_pending_at_decode_start": reload_pending_at_decode_start,
        });

        if let Some(graph) = self.graph.as_ref() {
            let resident_body = {
                let mut s = String::new();
                for (layer_idx, expert_idx) in &graph.validation_decode_start_resident {
                    s.push_str(&format!("{},{}\n", layer_idx, expert_idx));
                }
                s
            };
            if !graph.validation_decode_start_hcs_file.is_empty() {
                write_validation_text_file(&graph.validation_decode_start_hcs_file, &resident_body);
            }

            let slots_body = {
                let mut s = String::new();
                for (tier, slot_idx, layer_idx, expert_idx) in &graph.validation_decode_start_slots {
                    s.push_str(&format!("{},{},{},{}\n", tier, slot_idx, layer_idx, expert_idx));
                }
                s
            };
            if !graph.validation_decode_start_slots_file.is_empty() {
                write_validation_text_file(&graph.validation_decode_start_slots_file, &slots_body);
            }

            let cold_body = {
                let mut s = String::new();
                for (&(layer_idx, expert_idx), &(count, first_token)) in &graph.validation_decode_cold_hist {
                    s.push_str(&format!("{},{},{},{}\n", layer_idx, expert_idx, count, first_token));
                }
                s
            };
            if !graph.validation_decode_cold_file.is_empty() {
                write_validation_text_file(&graph.validation_decode_cold_file, &cold_body);
            }

            let cold_events_body = {
                let mut s = String::new();
                for (token_idx, layer_idx, expert_idx) in &graph.validation_decode_cold_events {
                    s.push_str(&format!("{},{},{}\n", token_idx, layer_idx, expert_idx));
                }
                s
            };
            if !graph.validation_decode_cold_events_file.is_empty() {
                write_validation_text_file(&graph.validation_decode_cold_events_file, &cold_events_body);
            }

            payload["decode_path"] = serde_json::json!({
                "per_layer_graphs_valid_end": graph.per_layer_graphs_valid,
                "per_layer_steps": graph.validation_per_layer_steps,
                "ungraphed_steps": graph.validation_ungraphed_steps,
                "timing_enabled": graph.timing_enabled,
                "gpu_route_sync": graph.gpu_route_sync,
            });
            payload["decode_experts"] = serde_json::json!({
                "cold_total": graph.dma_cold_experts,
                "hcs_total": graph.dma_hcs_experts,
                "total_per_token": if graph.validation_decode_steps > 0 {
                    (graph.dma_cold_experts + graph.dma_hcs_experts) as f64 / graph.validation_decode_steps as f64
                } else { 0.0 },
                "cold_per_token": if graph.validation_decode_steps > 0 {
                    graph.dma_cold_experts as f64 / graph.validation_decode_steps as f64
                } else { 0.0 },
                "hcs_per_token": if graph.validation_decode_steps > 0 {
                    graph.dma_hcs_experts as f64 / graph.validation_decode_steps as f64
                } else { 0.0 },
                "dma_bytes_total": graph.dma_bytes_total,
                "dma_calls_total": graph.dma_call_count,
                "tokens_timed": graph.validation_decode_steps,
            });
            payload["decode_start_hcs"] = serde_json::json!({
                "num_cached": graph.validation_decode_start_num_cached,
                "soft_num_cached": graph.validation_decode_start_soft_num_cached,
                "hard_num_cached": graph.validation_decode_start_hard_num_cached,
                "dupes": graph.validation_decode_start_dupes,
                "soft_dupes": graph.validation_decode_start_soft_dupes,
                "resident_hash": graph.validation_decode_start_hash,
                "resident_count": graph.validation_decode_start_resident.len(),
                "resident_file": graph.validation_decode_start_hcs_file,
                "slot_count": graph.validation_decode_start_slots.len(),
                "slots_file": graph.validation_decode_start_slots_file,
            });
            payload["decode_cold_experts"] = serde_json::json!({
                "unique_cold_experts": graph.validation_decode_cold_hist.len(),
                "cold_histogram_file": graph.validation_decode_cold_file,
                "cold_load_count": graph.validation_decode_cold_events.len(),
                "cold_loads_file": graph.validation_decode_cold_events_file,
            });

            if let Some(hcs) = graph.hcs.as_ref() {
                let (cache_fast_nonzero, hashmap_count, dupes, soft_dupes) = hcs.validation_counts();
                payload["hcs"] = serde_json::json!({
                    "num_cached": hcs.num_cached,
                    "soft_num_cached": hcs.soft_num_cached,
                    "hard_num_cached": hcs.num_cached.saturating_sub(hcs.soft_num_cached),
                    "cache_fast_nonzero": cache_fast_nonzero,
                    "hashmap_count": hashmap_count,
                    "dupes": dupes,
                    "soft_dupes": soft_dupes,
                    "soft_loaded": hcs.soft_loaded,
                    "soft_reload_pending": hcs.soft_reload_pending,
                    "soft_slots": hcs.soft_num_slots,
                    "soft_slot_size_mb": hcs.soft_slot_size as f64 / (1024.0 * 1024.0),
                    "soft_ranking_len": hcs.soft_ranking.len(),
                    "mapped_reads_available": hcs.mapped_reads_available,
                    "hit_rate": hcs.hit_rate(),
                    "total_hits": hcs.total_hits,
                    "total_misses": hcs.total_misses,
                });
            }
        }

        payload.to_string()
    }

    /// Allocate batch buffers for speculative decode batched verification.
    /// Called when draft model is loaded. batch_max = draft_k + 1.
    fn allocate_batch_buffers(&mut self, batch_max: usize) -> pyo3::PyResult<()> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("graph not configured"))?;

        let hs = graph.hidden_size;
        let vs = graph.vocab_size;
        let mut vram_total: usize = 0;

        // Per-token hidden/residual/moe_out: [batch_max * hidden_size] BF16
        let bh = self.device.alloc_zeros::<u16>(batch_max * hs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("batch_hidden: {:?}", e)))?;
        let br = self.device.alloc_zeros::<u16>(batch_max * hs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("batch_residual: {:?}", e)))?;
        let bmo = self.device.alloc_zeros::<u16>(batch_max * hs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("batch_moe_out: {:?}", e)))?;
        vram_total += batch_max * hs * 2 * 3;

        // Per-token logits: [batch_max * vocab_size] FP32
        let bl = self.device.alloc_zeros::<f32>(batch_max * vs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("batch_logits: {:?}", e)))?;
        vram_total += batch_max * vs * 4;

        // Host logits + routing buffers
        let h_bl = vec![0.0f32; batch_max * vs];
        // Find max topk across all MoE layers
        let max_topk = graph.moe_layers.iter()
            .filter_map(|m| m.as_ref())
            .map(|m| m.topk)
            .max()
            .unwrap_or(16);
        let h_bt_ids = vec![0i32; batch_max * max_topk];
        let h_bt_wts = vec![0.0f32; batch_max * max_topk];

        // GPU topk routing buffers (for batched MoE routing without per-token sync)
        let d_bt_ids = self.device.alloc_zeros::<i32>(batch_max * max_topk)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("d_batch_topk_ids: {:?}", e)))?;
        let d_bt_wts = self.device.alloc_zeros::<f32>(batch_max * max_topk)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("d_batch_topk_wts: {:?}", e)))?;
        vram_total += batch_max * max_topk * 4 * 2;

        // GPU gate logits buffer (batch_max * max_num_experts)
        let max_num_experts = graph.moe_layers.iter()
            .filter_map(|m| m.as_ref())
            .map(|m| m.num_experts)
            .max()
            .unwrap_or(256);
        let d_gate_logits = self.device.alloc_zeros::<f32>(batch_max * max_num_experts)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("d_batch_gate_logits: {:?}", e)))?;
        vram_total += batch_max * max_num_experts * 4;

        // LA state backups: one backup per LA layer
        let mut la_backup = Vec::new();
        let mut num_la_layers = 0usize;
        for (li, layer) in graph.layers.iter().enumerate() {
            if let GpuAttnConfig::LinearAttention {
                conv_state_ptr, recur_state_ptr,
                nk: _, nv, dk, dv, conv_dim, kernel_dim, ..
            } = &layer.attn {
                let conv_bytes = *conv_dim * *kernel_dim * 4;  // FP32
                let recur_bytes = *nv * *dk * *dv * 4;         // FP32
                let d_conv = self.device.alloc_zeros::<u8>(conv_bytes)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("la_conv_backup: {:?}", e)))?;
                let d_recur = self.device.alloc_zeros::<u8>(recur_bytes)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("la_recur_backup: {:?}", e)))?;
                vram_total += conv_bytes + recur_bytes;
                la_backup.push(LaStateBackup {
                    layer_idx: li,
                    conv_state_bytes: conv_bytes,
                    recur_state_bytes: recur_bytes,
                    d_conv_backup: d_conv,
                    d_recur_backup: d_recur,
                });
                num_la_layers += 1;
            }
        }

        // Hidden state stack for LA replay: [num_la_layers * batch_max * hidden_size] BF16
        let stack_size = num_la_layers * batch_max * hs;
        let d_stack = if stack_size > 0 {
            let s = self.device.alloc_zeros::<u16>(stack_size)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("la_hidden_stack: {:?}", e)))?;
            vram_total += stack_size * 2;
            Some(s)
        } else {
            None
        };

        // ── Compute max projection dimensions across all layers ──
        let mut max_proj_dim: usize = 0;
        let mut max_attn_out_dim: usize = 0;
        for layer in graph.layers.iter() {
            match &layer.attn {
                GpuAttnConfig::LinearAttention { in_proj_qkvz, in_proj_ba, nv, dv, .. } => {
                    let qkvz_w = &graph.weights[*in_proj_qkvz];
                    let ba_w = &graph.weights[*in_proj_ba];
                    max_proj_dim = max_proj_dim.max(qkvz_w.rows).max(ba_w.rows);
                    max_attn_out_dim = max_attn_out_dim.max(nv * dv); // gated_size for output proj
                }
                GpuAttnConfig::GQA { q_proj, fused_qkv, num_heads, head_dim, .. } => {
                    if let Some(fid) = fused_qkv {
                        let fw = &graph.weights[*fid];
                        max_proj_dim = max_proj_dim.max(fw.rows);
                    } else {
                        let qw = &graph.weights[*q_proj];
                        max_proj_dim = max_proj_dim.max(qw.rows);
                    }
                    max_attn_out_dim = max_attn_out_dim.max(num_heads * head_dim);
                }
                _ => {}
            }
        }

        // Allocate batch projection buffers
        let proj_a_size = batch_max * max_proj_dim;
        let proj_b_size = batch_max * max_proj_dim;
        let attn_out_size = batch_max * max_attn_out_dim;

        let d_proj_a = if proj_a_size > 0 {
            let buf = self.device.alloc_zeros::<f32>(proj_a_size)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("batch_proj_a: {:?}", e)))?;
            vram_total += proj_a_size * 4;
            Some(buf)
        } else { None };

        let d_proj_b = if proj_b_size > 0 {
            let buf = self.device.alloc_zeros::<f32>(proj_b_size)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("batch_proj_b: {:?}", e)))?;
            vram_total += proj_b_size * 4;
            Some(buf)
        } else { None };

        let d_attn_out = if attn_out_size > 0 {
            let buf = self.device.alloc_zeros::<u16>(attn_out_size)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("batch_attn_out: {:?}", e)))?;
            vram_total += attn_out_size * 2;
            Some(buf)
        } else { None };

        log::info!("Batch GEMM buffers: max_proj_dim={}, max_attn_out_dim={}, {:.1} MB VRAM",
            max_proj_dim, max_attn_out_dim,
            (proj_a_size * 4 + proj_b_size * 4 + attn_out_size * 2) as f64 / 1e6);

        graph.batch_max = batch_max;
        graph.d_batch_hidden = Some(bh);
        graph.d_batch_residual = Some(br);
        graph.d_batch_moe_out = Some(bmo);
        graph.d_batch_logits = Some(bl);
        graph.h_batch_logits = h_bl;
        graph.h_batch_topk_ids = h_bt_ids;
        graph.h_batch_topk_weights = h_bt_wts;
        graph.d_batch_topk_ids = Some(d_bt_ids);
        graph.d_batch_topk_wts = Some(d_bt_wts);
        graph.d_batch_gate_logits = Some(d_gate_logits);
        graph.la_backup = la_backup;
        graph.d_la_hidden_stack = d_stack;
        graph.d_batch_proj_a = d_proj_a;
        graph.d_batch_proj_b = d_proj_b;
        graph.d_batch_attn_out = d_attn_out;
        graph.batch_max_proj_dim = max_proj_dim;
        graph.batch_max_attn_out_dim = max_attn_out_dim;

        log::info!("Batch buffers allocated: batch_max={}, {:.1} MB VRAM ({} LA layers backed up)",
            batch_max, vram_total as f64 / 1e6, num_la_layers);
        Ok(())
    }

    /// Save all LA layer states to backup buffers (D2D copy in VRAM).
    fn save_la_states(&self) -> Result<(), String> {
        let graph = self.graph.as_ref().ok_or("graph not configured")?;
        for backup in &graph.la_backup {
            // Read current pointers from graph's layer config (may change after prefill)
            let (conv_ptr, recur_ptr) = match &graph.layers[backup.layer_idx].attn {
                GpuAttnConfig::LinearAttention { conv_state_ptr, recur_state_ptr, .. } => {
                    (*conv_state_ptr, *recur_state_ptr)
                }
                _ => return Err(format!("LA backup[{}]: layer is not LA", backup.layer_idx)),
            };
            unsafe {
                let err = cuda_sys::lib().cuMemcpyDtoD_v2(
                    *backup.d_conv_backup.device_ptr(),
                    conv_ptr,
                    backup.conv_state_bytes);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("save LA conv[{}]: {:?}", backup.layer_idx, err));
                }
                let err = cuda_sys::lib().cuMemcpyDtoD_v2(
                    *backup.d_recur_backup.device_ptr(),
                    recur_ptr,
                    backup.recur_state_bytes);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("save LA recur[{}]: {:?}", backup.layer_idx, err));
                }
            }
        }
        Ok(())
    }

    /// Restore all LA layer states from backup buffers.
    fn restore_la_states(&self) -> Result<(), String> {
        let graph = self.graph.as_ref().ok_or("graph not configured")?;
        for backup in &graph.la_backup {
            let (conv_ptr, recur_ptr) = match &graph.layers[backup.layer_idx].attn {
                GpuAttnConfig::LinearAttention { conv_state_ptr, recur_state_ptr, .. } => {
                    (*conv_state_ptr, *recur_state_ptr)
                }
                _ => return Err(format!("LA backup[{}]: layer is not LA", backup.layer_idx)),
            };
            unsafe {
                let err = cuda_sys::lib().cuMemcpyDtoD_v2(
                    conv_ptr,
                    *backup.d_conv_backup.device_ptr(),
                    backup.conv_state_bytes);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("restore LA conv[{}]: {:?}", backup.layer_idx, err));
                }
                let err = cuda_sys::lib().cuMemcpyDtoD_v2(
                    recur_ptr,
                    *backup.d_recur_backup.device_ptr(),
                    backup.recur_state_bytes);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("restore LA recur[{}]: {:?}", backup.layer_idx, err));
                }
            }
        }
        Ok(())
    }

    /// Replay LA layers for tokens 0..num_tokens using saved hidden states from the stack.
    /// This correctly updates conv_state and recur_state for the accepted tokens.
    fn replay_la_states(
        &mut self,
        num_tokens: usize,
        positions: &[usize],
    ) -> Result<(), String> {
        let mut graph = self.graph.take().ok_or("graph not configured")?;
        let result = self.replay_la_states_inner(&mut graph, num_tokens, positions);
        self.graph = Some(graph);
        result
    }

    fn replay_la_states_inner(
        &self,
        graph: &mut GpuDecodeGraph,
        num_tokens: usize,
        positions: &[usize],
    ) -> Result<(), String> {
        let hs = graph.hidden_size;
        let eps = graph.eps;
        let d_stack_ptr = graph.d_la_hidden_stack.as_ref()
            .ok_or("LA hidden stack not allocated")?.device_ptr();
        let batch_max = graph.batch_max;
        let k = graph.kernels.as_ref().ok_or("kernels not cached")?.clone();

        // For each LA layer, replay tokens in order
        let mut la_idx = 0usize;
        for layer_idx in 0..graph.layers.len() {
            let is_la = matches!(&graph.layers[layer_idx].attn, GpuAttnConfig::LinearAttention { .. });
            if !is_la { continue; }

            for t in 0..num_tokens {
                // Load saved hidden state for this token at this LA layer
                let stack_offset = (la_idx * batch_max + t) * hs;
                unsafe {
                    let err = cuda_sys::lib().cuMemcpyDtoD_v2(
                        *graph.d_hidden.device_ptr(),
                        (*d_stack_ptr as *const u16).add(stack_offset) as u64,
                        hs * 2);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(format!("replay LA load hidden[{}][{}]: {:?}", layer_idx, t, err));
                    }
                }

                // Run the LA forward pass (this updates conv_state and recur_state)
                // We reuse the full attention code path - it writes output to d_hidden
                // which we don't need, but the side effects on conv/recur state are what matter.
                let layer = &graph.layers[layer_idx];
                match &layer.attn {
                    GpuAttnConfig::LinearAttention {
                        in_proj_qkvz, in_proj_ba, out_proj: _,
                        conv_weight_ptr, a_log_ptr, dt_bias_ptr, norm_weight_ptr: _,
                        nk, nv, dk, dv, hr, kernel_dim, conv_dim, scale,
                        conv_state_ptr, recur_state_ptr,
                    } => {
                        let nk_ = *nk; let nv_ = *nv; let dk_ = *dk; let dv_ = *dv;
                        let hr_ = *hr; let cd = *conv_dim; let kd = *kernel_dim;
                        let key_dim = nk_ * dk_;

                        // Projections
                        let qkvz_w = &graph.weights[*in_proj_qkvz];
                        let ba_w = &graph.weights[*in_proj_ba];
                        self.gemv_bf16_to_f32(qkvz_w, *graph.d_hidden.device_ptr(), *graph.d_la_qkvz.device_ptr())?;
                        self.gemv_bf16_to_f32(ba_w, *graph.d_hidden.device_ptr(), *graph.d_la_ba.device_ptr())?;

                        // Uninterleave
                        {
                            let group_dim = 2 * dk_ + 2 * hr_ * dv_;
                            let total = nk_ * group_dim;
                            let threads = 256u32;
                            let blocks = ((total as u32) + threads - 1) / threads;
                            let unint_fn = self.device.get_func(MODULE_NAME, "uninterleave_qkvz")
                                .ok_or_else(|| "uninterleave_qkvz not found".to_string())?;
                            unsafe {
                                unint_fn.launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (*graph.d_la_conv_out.device_ptr(), *graph.d_la_recur_out.device_ptr(),
                                     *graph.d_la_qkvz.device_ptr(), nk_ as i32, dk_ as i32, hr_ as i32, dv_ as i32),
                                ).map_err(|e| format!("replay uninterleave[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // Conv1d (updates conv_state)
                        {
                            let threads = 256u32;
                            let blocks = ((cd as u32) + threads - 1) / threads;
                            let la_conv1d_fn = self.device.get_func(MODULE_NAME, "la_conv1d")
                                .ok_or_else(|| "la_conv1d kernel not found".to_string())?;
                            unsafe {
                                la_conv1d_fn.launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (*conv_state_ptr, *graph.d_la_conv_out.device_ptr(),
                                     *graph.d_la_qkvz.device_ptr(), *conv_weight_ptr, cd as i32, kd as i32),
                                ).map_err(|e| format!("replay la_conv1d[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // Gate/beta
                        let gate_ptr_local = *graph.d_la_conv_out.device_ptr();
                        let beta_ptr_local = unsafe { (*graph.d_la_conv_out.device_ptr() as *const f32).add(nv_) as u64 };
                        {
                            let threads = 256u32;
                            let blocks = ((nv_ as u32) + threads - 1) / threads;
                            let gb_fn = self.device.get_func(MODULE_NAME, "compute_gate_beta")
                                .ok_or_else(|| "compute_gate_beta not found".to_string())?;
                            unsafe {
                                gb_fn.launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (gate_ptr_local, beta_ptr_local, *graph.d_la_ba.device_ptr(),
                                     *a_log_ptr, *dt_bias_ptr, nv_ as i32, hr_ as i32),
                                ).map_err(|e| format!("replay gate_beta[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // Head repeat-interleave
                        let q_ptr_for_recur: u64;
                        let k_ptr_for_recur: u64;
                        if hr_ > 1 {
                            let total_q = (nv_ * dk_) as u32;
                            let threads = 256u32;
                            let blocks = (total_q + threads - 1) / threads;
                            let ri_fn = self.device.get_func(MODULE_NAME, "repeat_interleave_heads")
                                .ok_or_else(|| "repeat_interleave_heads not found".to_string())?;
                            unsafe {
                                ri_fn.clone().launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (*graph.d_la_recur_out.device_ptr(), *graph.d_la_qkvz.device_ptr(),
                                     nk_ as i32, dk_ as i32, hr_ as i32),
                                ).map_err(|e| format!("replay repeat_interleave q[{}]: {:?}", layer_idx, e))?;
                                let k_in = (*graph.d_la_qkvz.device_ptr() as *const f32).add(key_dim) as u64;
                                let k_out = (*graph.d_la_recur_out.device_ptr() as *const f32).add(nv_ * dk_) as u64;
                                ri_fn.launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (k_out, k_in, nk_ as i32, dk_ as i32, hr_ as i32),
                                ).map_err(|e| format!("replay repeat_interleave k[{}]: {:?}", layer_idx, e))?;
                            }
                            q_ptr_for_recur = *graph.d_la_recur_out.device_ptr();
                            k_ptr_for_recur = unsafe { (*graph.d_la_recur_out.device_ptr() as *const f32).add(nv_ * dk_) as u64 };
                        } else {
                            q_ptr_for_recur = *graph.d_la_qkvz.device_ptr();
                            k_ptr_for_recur = unsafe { (*graph.d_la_qkvz.device_ptr() as *const f32).add(key_dim) as u64 };
                        }

                        // L2 norm + scale
                        {
                            let threads = 256u32;
                            let l2_fn = self.device.get_func(MODULE_NAME, "l2norm_scale_per_head")
                                .ok_or_else(|| "l2norm_scale_per_head not found".to_string())?;
                            unsafe {
                                l2_fn.clone().launch(
                                    LaunchConfig { grid_dim: (nv_ as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (q_ptr_for_recur, *scale, nv_ as i32, dk_ as i32),
                                ).map_err(|e| format!("replay l2norm q[{}]: {:?}", layer_idx, e))?;
                                l2_fn.launch(
                                    LaunchConfig { grid_dim: (nv_ as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (k_ptr_for_recur, 1.0f32, nv_ as i32, dk_ as i32),
                                ).map_err(|e| format!("replay l2norm k[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // Recurrence (updates recur_state — the key side effect we need)
                        let v_ptr = unsafe { (*graph.d_la_qkvz.device_ptr() as *const f32).add(2 * key_dim) as u64 };
                        {
                            let threads = 256u32;
                            let delta_fn = self.device.get_func(MODULE_NAME, "gated_delta_net_step")
                                .ok_or_else(|| "gated_delta_net_step not found".to_string())?;
                            unsafe {
                                delta_fn.launch(
                                    LaunchConfig { grid_dim: (nv_ as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (*recur_state_ptr, q_ptr_for_recur, k_ptr_for_recur, v_ptr,
                                     gate_ptr_local, beta_ptr_local, *graph.d_la_ba.device_ptr(),
                                     nv_ as i32, dk_ as i32, dv_ as i32),
                                ).map_err(|e| format!("replay gated_delta_net[{}]: {:?}", layer_idx, e))?;
                            }
                        }
                        // Skip steps 8-9 (gated_rmsnorm_silu, output projection) — not needed for state replay
                    }
                    _ => {} // Not LA layer — shouldn't happen due to is_la check
                }
            }
            la_idx += 1;
        }
        Ok(())
    }

    /// Initialize CUDA graph infrastructure: allocate GPU-side buffers for
    /// graphable kernels and dummy expert buffer for cold expert handling.
    /// Must be called after configure() and HCS pool init.
    fn init_cuda_graph_buffers(&mut self) -> Result<(), String> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| "Call configure first".to_string())?;

        // Allocate GPU-side scalar buffers for graphable kernels
        graph.d_graph_token_id = Some(self.device.alloc_zeros::<i32>(1)
            .map_err(|e| format!("alloc d_graph_token_id: {:?}", e))?);
        graph.d_graph_pos = Some(self.device.alloc_zeros::<i32>(1)
            .map_err(|e| format!("alloc d_graph_pos: {:?}", e))?);
        graph.d_graph_seq_len = Some(self.device.alloc_zeros::<i32>(1)
            .map_err(|e| format!("alloc d_graph_seq_len: {:?}", e))?);
        // Allocate dummy expert buffer (all zeros, used for padded expert slots during
        // graph replay). Allocated unconditionally from model config so it exists before
        // HCS loads — prevents GEMV on arbitrary data producing NaN for dummy experts.
        {
            let hs = graph.hidden_size;
            let intermediate = graph.moe_intermediate_size;
            let gs = graph.group_size;
            let expert_bits = graph.expert_bits;
            if intermediate > 0 && hs > 0 && expert_bits > 0 {
                let w13_packed_bytes = if expert_bits == 16 {
                    (2 * intermediate * hs * 2) as u64
                } else {
                    (2 * intermediate * hs * expert_bits as usize / 8) as u64
                };
                // Scales are raw BF16 (2 bytes each), NOT packed like weights.
                // Formula: (K / group_size) * N * 2 bytes
                let w13_scales_bytes = if gs == 0 { 0u64 } else {
                    (hs / gs * (2 * intermediate) * 2) as u64
                };
                let w2_packed_bytes = if expert_bits == 16 {
                    (hs * intermediate * 2) as u64
                } else {
                    (hs * intermediate * expert_bits as usize / 8) as u64
                };
                let w2_scales_bytes = if gs == 0 { 0u64 } else {
                    (intermediate / gs * hs * 2) as u64
                };
                let total_bytes = (w13_packed_bytes + w13_scales_bytes + w2_packed_bytes
                    + w2_scales_bytes) as usize;
                let align = 512usize;
                let alloc_bytes = (total_bytes + align - 1) & !(align - 1);

                graph.d_dummy_expert = Some(self.device.alloc_zeros::<u8>(alloc_bytes)
                    .map_err(|e| format!("alloc d_dummy_expert: {:?}", e))?);
                let dummy_base = *graph.d_dummy_expert.as_ref().unwrap().device_ptr();
                let dummy_vals: [u64; 4] = [
                    dummy_base,
                    dummy_base + w13_packed_bytes,
                    dummy_base + w13_packed_bytes + w13_scales_bytes,
                    dummy_base + w13_packed_bytes + w13_scales_bytes + w2_packed_bytes,
                ];
                graph.h_dummy_ptrs = dummy_vals;
                let d_dummy_ptrs = self.device.htod_copy(dummy_vals.to_vec())
                    .map_err(|e| format!("alloc d_dummy_ptrs: {:?}", e))?;
                graph.d_dummy_ptrs = Some(d_dummy_ptrs);
                log::info!("CUDA graph: allocated dummy expert buffer ({} KB)",
                    alloc_bytes / 1024);
            }
        }

        // Allocate persistent cuBLAS workspace (required for CUDA graph capture —
        // cuBLAS must not do internal cudaMalloc during capture).
        // 4 MB workspace prevents cuBLAS internal allocation during graph capture.
        let workspace_bytes = 4 * 1024 * 1024;
        let d_workspace = self.device.alloc_zeros::<u8>(workspace_bytes)
            .map_err(|e| format!("alloc d_cublas_workspace: {:?}", e))?;
        unsafe {
            let ws_ptr = *d_workspace.device_ptr() as *mut std::ffi::c_void;
            let lib = cublas_sys::lib();
            let err = lib.cublasSetWorkspace_v2(*self.blas.handle(), ws_ptr, workspace_bytes);
            if err != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(format!("cublasSetWorkspace: {:?}", err));
            }
        }
        graph.d_cublas_workspace = Some(d_workspace);

        if stderr_debug_enabled() {
            eprintln!("[krasis] CUDA graph infrastructure initialized (has_dummy={}, workspace=32KB)",
                graph.d_dummy_expert.is_some());
        }
        Ok(())
    }

    /// Invalidate per-layer CUDA graphs (e.g. when HCS layout changes around prefill).
    fn invalidate_cuda_graph(&mut self) {
        if let Some(ref mut graph) = self.graph {
            for exec in graph.per_layer_graphs.drain(..) {
                unsafe { cuda_sys::lib().cuGraphExecDestroy(exec.0); }
            }
            graph.per_layer_graphs_valid = false;
            // NOTE: graphs_ever_captured stays true -- we keep the snapshot for diagnostics
        }
    }

    /// Verify that all LA and KV pointers baked into captured CUDA graphs are still valid.
    /// Returns true if graphs can be safely reused, false if recapture is needed.
    fn verify_graph_pointers(&self) -> bool {
        let graph = match self.graph.as_ref() {
            Some(g) => g,
            None => return false,
        };
        if !graph.graphs_ever_captured || graph.per_layer_graphs.is_empty() {
            return false;
        }

        // Check LA state pointers
        for &(li, captured_conv, captured_recur) in &graph.captured_la_ptrs {
            if li >= graph.layers.len() {
                if stderr_debug_enabled() {
                    eprintln!("[krasis] graph-reuse: LA layer {} out of range", li);
                }
                return false;
            }
            match &graph.layers[li].attn {
                GpuAttnConfig::LinearAttention { conv_state_ptr, recur_state_ptr, .. } => {
                    if *conv_state_ptr != captured_conv || *recur_state_ptr != captured_recur {
                        if stderr_debug_enabled() {
                            eprintln!("[krasis] graph-reuse: LA layer {} ptrs changed! \
                                conv: 0x{:x}→0x{:x}, recur: 0x{:x}→0x{:x}",
                                li, captured_conv, *conv_state_ptr,
                                captured_recur, *recur_state_ptr);
                        }
                        return false;
                    }
                }
                _ => {
                    if stderr_debug_enabled() {
                        eprintln!("[krasis] graph-reuse: layer {} no longer LA", li);
                    }
                    return false;
                }
            }
        }

        // Check KV cache pointers
        for &(li, captured_k, captured_v) in &graph.captured_kv_ptrs {
            if li >= graph.kv_k_ptrs.len() {
                if stderr_debug_enabled() {
                    eprintln!("[krasis] graph-reuse: KV layer {} out of range", li);
                }
                return false;
            }
            if graph.kv_k_ptrs[li] != captured_k || graph.kv_v_ptrs[li] != captured_v {
                if stderr_debug_enabled() {
                    eprintln!("[krasis] graph-reuse: KV layer {} ptrs changed! \
                        k: 0x{:x}→0x{:x}, v: 0x{:x}→0x{:x}",
                        li, captured_k, graph.kv_k_ptrs[li],
                        captured_v, graph.kv_v_ptrs[li]);
                }
                return false;
            }
        }

        // Check MLA cache pointers
        for &(li, captured_ckv, captured_kpe) in &graph.captured_mla_ptrs {
            if li >= graph.layers.len() {
                if stderr_debug_enabled() {
                    eprintln!("[krasis] graph-reuse: MLA layer {} out of range", li);
                }
                return false;
            }
            match &graph.layers[li].attn {
                GpuAttnConfig::MLA { ckv_cache_ptr, kpe_cache_ptr, .. } => {
                    if *ckv_cache_ptr != captured_ckv || *kpe_cache_ptr != captured_kpe {
                        if stderr_debug_enabled() {
                            eprintln!("[krasis] graph-reuse: MLA layer {} ptrs changed! \
                                ckv: 0x{:x}→0x{:x}, kpe: 0x{:x}→0x{:x}",
                                li, captured_ckv, *ckv_cache_ptr,
                                captured_kpe, *kpe_cache_ptr);
                        }
                        return false;
                    }
                }
                _ => {
                    if stderr_debug_enabled() {
                        eprintln!("[krasis] graph-reuse: layer {} no longer MLA", li);
                    }
                    return false;
                }
            }
        }

        // Check Polar4 KV cache pointers
        for &(li, cap_kr, cap_vr, cap_ka, cap_va) in &graph.captured_polar4_ptrs {
            if li >= graph.kv_k_radius_ptrs.len() {
                if stderr_debug_enabled() {
                    eprintln!("[krasis] graph-reuse: Polar4 layer {} out of range", li);
                }
                return false;
            }
            if graph.kv_k_radius_ptrs[li] != cap_kr
                || graph.kv_v_radius_ptrs[li] != cap_vr
                || graph.kv_k_angles_ptrs[li] != cap_ka
                || graph.kv_v_angles_ptrs[li] != cap_va
            {
                if stderr_debug_enabled() {
                    eprintln!("[krasis] graph-reuse: Polar4 layer {} ptrs changed!", li);
                }
                return false;
            }
        }

        true
    }

    // ════════════════════════════════════════════════════════════════════
    // Per-layer CUDA graph capture & replay (49 graphs for 48 MoE layers)
    // ════════════════════════════════════════════════════════════════════

    /// Capture 49 per-layer CUDA graphs:
    ///   Graph 0: embedding + layer 0 routing
    ///   Graph 1..N-1: layer K experts + layer K+1 routing
    ///   Graph N: last layer experts + final norm + LM head
    ///
    /// Between segments, CPU reads routing results, DMAs cold experts, and
    /// populates d_batch_upload with expert pointers. The first decode token
    /// is computed correctly during capture (real execution on step 0 provides
    /// correct routing data for the CPU-side work between captures).
    /// Capture per-layer CUDA graphs for a segment of layers.
    /// - layer_range: if Some((start, end)), only capture graphs for MoE layers in [start..end).
    ///   If None, capture for all layers (single-GPU mode).
    /// - do_embedding: include embedding lookup in the first graph.
    /// - do_final: include final norm + LM head in the last graph.
    /// - gqa_cache_offset: number of GQA layers before the segment start (for multi-GPU).
    fn capture_per_layer_graphs(
        &mut self,
        layer_range: Option<(usize, usize)>,
        do_embedding: bool,
        do_final: bool,
        gqa_cache_offset: usize,
    ) -> Result<(), String> {
        let mut graph = self.graph.take()
            .ok_or_else(|| "Call configure first".to_string())?;

        if graph.d_graph_token_id.is_none() {
            self.graph = Some(graph);
            return Err("Call init_cuda_graph_buffers first".to_string());
        }

        // Set segment config so run_segment_kernels uses correct GQA cache indices
        if let Some((start, _end)) = layer_range {
            graph.segment_layer_start = start;
            graph.segment_gqa_cache_offset = gqa_cache_offset;
        } else {
            graph.segment_layer_start = 0;
            graph.segment_gqa_cache_offset = 0;
        }

        // Invalidate any existing per-layer graphs
        for exec in graph.per_layer_graphs.drain(..) {
            unsafe { cuda_sys::lib().cuGraphExecDestroy(exec.0); }
        }
        graph.per_layer_graphs_valid = false;

        // Identify which layers have MoE data (within our range)
        let num_layers = graph.layers.len();
        let (range_start, range_end) = layer_range.unwrap_or((0, num_layers));
        let mut moe_indices: Vec<usize> = Vec::new();
        for i in range_start..range_end {
            if i < graph.moe_layers.len() && graph.moe_layers[i].is_some() {
                moe_indices.push(i);
            }
        }
        graph.per_layer_moe_indices = moe_indices.clone();
        let num_moe = moe_indices.len();
        let num_graphs = num_moe + 1; // routing(0) + (num_moe-1) combined + final

        if stderr_debug_enabled() {
            eprintln!("[krasis] Capturing {} per-layer graphs ({} MoE layers, range {}-{}, emb={}, final={})",
                num_graphs, num_moe, range_start, range_end, do_embedding, do_final);
        }

        // Create capture stream
        let mut capture_stream: cuda_sys::CUstream = std::ptr::null_mut();
        unsafe {
            let err = cuda_sys::lib().cuStreamCreate(
                &mut capture_stream,
                cuda_sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                self.graph = Some(graph);
                return Err(format!("cuStreamCreate for capture: {:?}", err));
            }
        }

        // Save original stream and swap to capture stream
        let stream_ptr: *mut cuda_sys::CUstream;
        let orig_stream: cuda_sys::CUstream;
        unsafe {
            self.device.synchronize().map_err(|e| format!("sync before capture: {:?}", e))?;
            let stream_ref = self.device.cu_stream();
            stream_ptr = stream_ref as *const cuda_sys::CUstream as *mut cuda_sys::CUstream;
            orig_stream = *stream_ptr;
            *stream_ptr = capture_stream;

            // Switch cuBLAS handle to capture stream so GEMM/GEMV ops are recorded in the graph
            cublas_result::set_stream(*self.blas.handle(), capture_stream as cublas_sys::cudaStream_t)
                .map_err(|e| format!("cublasSetStream to capture: {:?}", e))?;
        }

        // Shared memory opt-in for gated_delta_net_step (same as monolithic capture)
        {
            let k = graph.kernels.as_ref().unwrap();
            let struct_ptr = &k.gated_delta_net_step as *const _ as *const u8;
            let word0: cuda_sys::CUfunction = unsafe {
                std::ptr::read(struct_ptr as *const cuda_sys::CUfunction)
            };
            let word1: cuda_sys::CUfunction = unsafe {
                std::ptr::read(struct_ptr.add(8) as *const cuda_sys::CUfunction)
            };
            let mut dummy = 0i32;
            let w0_valid = unsafe {
                cuda_sys::lib().cuFuncGetAttribute(
                    &mut dummy,
                    cuda_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS,
                    word0,
                ) == cuda_sys::CUresult::CUDA_SUCCESS
            };
            let raw_fn = if w0_valid { word0 } else { word1 };
            let mut cur_smem = 0i32;
            unsafe {
                cuda_sys::lib().cuFuncGetAttribute(
                    &mut cur_smem,
                    cuda_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    raw_fn,
                );
            }
            if cur_smem < 66560 {
                let mut max_smem_i32 = 0i32;
                unsafe {
                    cuda_sys::lib().cuDeviceGetAttribute(
                        &mut max_smem_i32,
                        cuda_sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                        self.device.ordinal() as i32,
                    );
                    cuda_sys::lib().cuFuncSetAttribute(
                        raw_fn,
                        cuda_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        max_smem_i32,
                    );
                }
            }
        }

        let mut captured_graphs: Vec<CudaGraphExecPtr> = Vec::with_capacity(num_graphs);
        let mut capture_err: Option<String> = None;

        // Helper: begin capture on stream
        let begin_capture = |stream: cuda_sys::CUstream| -> Result<(), String> {
            let err = unsafe {
                cuda_sys::lib().cuStreamBeginCapture_v2(
                    stream,
                    cuda_sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED,
                )
            };
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuStreamBeginCapture: {:?}", err));
            }
            Ok(())
        };

        // Helper: end capture and instantiate
        let end_capture_and_instantiate = |stream: cuda_sys::CUstream, label: &str|
            -> Result<CudaGraphExecPtr, String>
        {
            let mut cu_graph: CUgraph = std::ptr::null_mut();
            let err = unsafe { cuda_sys::lib().cuStreamEndCapture(stream, &mut cu_graph) };
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuStreamEndCapture[{}]: {:?}", label, err));
            }
            if cu_graph.is_null() {
                return Err(format!("cuStreamEndCapture[{}]: null graph", label));
            }
            let mut graph_exec: CUgraphExec = std::ptr::null_mut();
            let err = unsafe {
                cuda_sys::lib().cuGraphInstantiateWithFlags(&mut graph_exec, cu_graph, 0)
            };
            unsafe { cuda_sys::lib().cuGraphDestroy(cu_graph); }
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuGraphInstantiate[{}]: {:?}", label, err));
            }
            Ok(CudaGraphExecPtr(graph_exec))
        };

        // ── Capture all graphs ──
        // Each graph captures kernels on the capture stream (operations are recorded, not executed).
        // The buffer addresses are fixed, so the graph structure is valid regardless of data content.

        for graph_idx in 0..num_graphs {
            if capture_err.is_some() { break; }

            let label = if graph_idx == 0 {
                format!("routing-L{}", moe_indices[0])
            } else if graph_idx < num_moe {
                format!("experts-L{}-routing-L{}", moe_indices[graph_idx - 1], moe_indices[graph_idx])
            } else {
                format!("experts-L{}-final", moe_indices[num_moe - 1])
            };

            if let Err(e) = begin_capture(capture_stream) {
                capture_err = Some(format!("begin[{}]: {}", label, e));
                break;
            }

            // Run the segment kernels
            let seg_result = self.run_graph_segment(&mut graph, graph_idx, &moe_indices, do_embedding, do_final);
            if let Err(e) = seg_result {
                // Must still end capture to restore stream state
                let mut dummy: CUgraph = std::ptr::null_mut();
                unsafe { cuda_sys::lib().cuStreamEndCapture(capture_stream, &mut dummy); }
                if !dummy.is_null() { unsafe { cuda_sys::lib().cuGraphDestroy(dummy); } }
                capture_err = Some(format!("segment[{}]: {}", label, e));
                break;
            }

            match end_capture_and_instantiate(capture_stream, &label) {
                Ok(exec) => {
                    captured_graphs.push(exec);
                    if stderr_debug_enabled() {
                        eprintln!("[krasis] Captured graph {} ({})", graph_idx, label);
                    }
                }
                Err(e) => {
                    capture_err = Some(e);
                    break;
                }
            }
        }

        // Restore original stream, cuBLAS handle, and destroy capture stream
        unsafe {
            *stream_ptr = orig_stream;
            cublas_result::set_stream(*self.blas.handle(), orig_stream as cublas_sys::cudaStream_t)
                .map_err(|e| format!("cublasSetStream restore: {:?}", e))?;
            cuda_sys::lib().cuStreamDestroy_v2(capture_stream);
        }

        if let Some(e) = capture_err {
            // Clean up any graphs we did capture
            for exec in captured_graphs {
                unsafe { cuda_sys::lib().cuGraphExecDestroy(exec.0); }
            }
            self.graph = Some(graph);
            return Err(format!("Per-layer graph capture failed: {}", e));
        }

        graph.per_layer_graphs = captured_graphs;
        graph.per_layer_graphs_valid = true;
        graph.graphs_ever_captured = true;

        // Snapshot LA and KV pointers at capture time for cross-request reuse verification.
        // If these pointers are stable across requests (they should be: KV is allocated once,
        // LA Rust-side FP32 buffers use copy_() in-place), we can skip recapture.
        graph.captured_la_ptrs.clear();
        graph.captured_kv_ptrs.clear();
        graph.captured_mla_ptrs.clear();
        graph.captured_polar4_ptrs.clear();
        for (li, layer) in graph.layers.iter().enumerate() {
            match &layer.attn {
                GpuAttnConfig::LinearAttention { conv_state_ptr, recur_state_ptr, .. } => {
                    graph.captured_la_ptrs.push((li, *conv_state_ptr, *recur_state_ptr));
                }
                GpuAttnConfig::GQA { .. } => {
                    if graph.kv_format == 2 {
                        // Polar4: track radius/angles pointers
                        if li < graph.kv_k_radius_ptrs.len() && graph.kv_k_radius_ptrs[li] != 0 {
                            graph.captured_polar4_ptrs.push((
                                li,
                                graph.kv_k_radius_ptrs[li],
                                graph.kv_v_radius_ptrs[li],
                                graph.kv_k_angles_ptrs[li],
                                graph.kv_v_angles_ptrs[li],
                            ));
                        }
                    } else if li < graph.kv_k_ptrs.len() && graph.kv_k_ptrs[li] != 0 {
                        graph.captured_kv_ptrs.push((li, graph.kv_k_ptrs[li], graph.kv_v_ptrs[li]));
                    }
                }
                GpuAttnConfig::MLA { ckv_cache_ptr, kpe_cache_ptr, .. } => {
                    if *ckv_cache_ptr != 0 {
                        graph.captured_mla_ptrs.push((li, *ckv_cache_ptr, *kpe_cache_ptr));
                    }
                }
                GpuAttnConfig::Mamba2 { .. } => {
                    // Mamba2 state is always GPU-resident, no pointer tracking needed for graph capture
                }
            }
        }
        if stderr_debug_enabled() {
            eprintln!("[krasis] Pointer snapshot: {} LA ptrs, {} KV ptrs, {} MLA ptrs, {} Polar4 ptrs",
                graph.captured_la_ptrs.len(), graph.captured_kv_ptrs.len(),
                graph.captured_mla_ptrs.len(), graph.captured_polar4_ptrs.len());
        }

        self.graph = Some(graph);
        if stderr_debug_enabled() {
            eprintln!("[krasis] Per-layer graphs captured: {} graphs", num_graphs);
        }
        Ok(())
    }

    /// Run one graph segment's kernels (for capture or debug).
    /// graph_idx 0: embedding + routing for moe_indices[0]
    /// graph_idx 1..N-1: experts for moe_indices[idx-1] + routing for moe_indices[idx]
    /// graph_idx N: experts for moe_indices[N-1] + final norm + LM head
    ///
    /// seg_do_embedding/seg_do_final control whether the first/last graph includes
    /// embedding and final norm+LM head (set false for multi-GPU segments that
    /// don't own those parts).
    fn run_graph_segment(
        &self,
        graph: &mut GpuDecodeGraph,
        graph_idx: usize,
        moe_indices: &[usize],
        seg_do_embedding: bool,
        seg_do_final: bool,
    ) -> Result<(), String> {
        let num_moe = moe_indices.len();
        let is_first = graph_idx == 0;
        let is_last = graph_idx == num_moe;

        // What this segment contains:
        // - Expert compute for a MoE layer (if not first graph)
        // - Routing for next layer group (if not last graph)
        //   "Routing" = all layers from prev_moe+1 through this_moe (inclusive),
        //   covering any non-MoE layers in between + the MoE layer's routing portion.

        // Expert layer index (layer whose experts we compute)
        let expert_layer = if !is_first { Some(moe_indices[graph_idx - 1]) } else { None };

        // Range of layers whose routing (norm+attn+post-norm+gate+topk) we capture
        let routing_range = if !is_last {
            let moe_layer = moe_indices[graph_idx];
            // Start after the previous MoE layer's experts (or from layer 0 for graph 0)
            let start = if is_first {
                // For multi-GPU, the first MoE index might not be 0
                // Use the layer range start or 0
                if !moe_indices.is_empty() {
                    // Start from the beginning of the segment or 0
                    let seg_start = graph.segment_layer_start;
                    if seg_start > 0 { seg_start } else { 0 }
                } else { 0 }
            } else {
                moe_indices[graph_idx - 1] + 1
            };
            // Include non-MoE layers between start and moe_layer, plus moe_layer itself
            Some((start, moe_layer))
        } else {
            None
        };

        // Include final norm + LM head only in last graph AND if segment owns it
        let include_final = is_last && seg_do_final;
        // Include embedding only in first graph AND if segment owns it
        let include_embedding = is_first && seg_do_embedding;

        // moe_seq_idx: the index of the MoE layer being routed in this segment
        let moe_seq_idx = if !is_last { graph_idx } else { 0 };
        self.run_segment_kernels(graph, expert_layer, routing_range, include_embedding, include_final, moe_seq_idx)
    }

    /// Run the actual kernels for one graph segment.
    /// - expert_layer: if Some, run MoE expert compute for this layer (from d_batch_upload)
    /// - routing_range: if Some((start, end)), run routing for layers start..=end
    /// - include_embedding: run embedding lookup
    /// - include_final: run final norm + LM head
    /// - moe_seq_idx: sequential index of the MoE layer being routed (for mapped activations)
    fn run_segment_kernels(
        &self,
        graph: &mut GpuDecodeGraph,
        expert_layer: Option<usize>,
        routing_range: Option<(usize, usize)>,
        include_embedding: bool,
        include_final: bool,
        moe_seq_idx: usize,
    ) -> Result<(), String> {
        use cudarc::driver::LaunchConfig;

        let cu_stream: cuda_sys::CUstream = *self.device.cu_stream();
        let hs = graph.hidden_size;
        let eps = graph.eps;
        let k = graph.kernels.as_ref()
            .ok_or_else(|| "Kernels not cached".to_string())?.clone();

        let intermediate = graph.moe_intermediate_size;
        let gs = graph.group_size;
        // Select inverse weight permutation based on expert quantization bits
        let inv_wp = if graph.expert_bits == 8 {
            *graph.d_inv_weight_perm_int8.device_ptr()
        } else {
            *graph.d_inv_weight_perm.device_ptr()
        };
        let inv_sp = *graph.d_inv_scale_perm.device_ptr();
        let is_int8 = graph.expert_bits == 8;

        let d_pos_ptr = *graph.d_graph_pos.as_ref().unwrap().device_ptr();
        let d_seq_len_ptr = *graph.d_graph_seq_len.as_ref().unwrap().device_ptr();

        // v2 K-split config
        let w13_n = 2 * intermediate;
        let w13_k_tiles = hs / 16;
        let w13_max_ksplits = w13_k_tiles / 16;
        let w13_ksplits = if w13_max_ksplits > 1 {
            let n_tiles = (w13_n + 15) / 16;
            let target = graph.num_sms * 4;
            let desired = (target + n_tiles - 1) / n_tiles;
            desired.clamp(1, w13_max_ksplits.min(8))
        } else { 1 };
        let use_v2_w13 = w13_ksplits > 1;
        let partial_ptr = *graph.d_v2_partial.device_ptr();

        // ── Expert compute (if this segment includes it) ──
        if let Some(layer_idx) = expert_layer {
            if let Some(ref moe) = graph.moe_layers.get(layer_idx).and_then(|m| m.as_ref()) {
                let topk = moe.topk;
                let rsf = moe.routed_scaling_factor;

                let max_ept = graph.max_experts_per_tok;
                let d_upload_base = *graph.d_batch_upload.device_ptr();
                let ptr_stride = max_ept * 8;
                let d_w13p = d_upload_base;
                let d_w13s = d_upload_base + ptr_stride as u64;
                let d_w2p = d_upload_base + (ptr_stride * 2) as u64;
                let d_w2s = d_upload_base + (ptr_stride * 3) as u64;
                let d_wts = d_upload_base + (ptr_stride * 4) as u64;

                // Zero MoE output accumulator
                unsafe {
                    k.zero_bf16.clone().launch(
                        LaunchConfig::for_num_elems(hs as u32),
                        (*graph.d_moe_out.device_ptr(), hs as i32),
                    ).map_err(|e| format!("zero_bf16[{}]: {:?}", layer_idx, e))?;
                }

                // Batched w13 GEMV v2
                if use_v2_w13 {
                    let w13_n_tiles = (w13_n + 15) / 16;
                    let w13_smem = (hs * 2 + 1024 * 4 + 64 * 4 + 16 * 16 * 4) as u32;
                    let w13_kernel = if is_int8 {
                        k.marlin_gemv_int8_v2_batched.clone()
                    } else {
                        k.marlin_gemv_int4_v2_batched.clone()
                    };
                    unsafe {
                        w13_kernel.launch(
                            LaunchConfig {
                                grid_dim: (w13_n_tiles as u32, w13_ksplits as u32, topk as u32),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: w13_smem,
                            },
                            (
                                d_w13p, d_w13s,
                                *graph.d_hidden.device_ptr(),
                                *graph.d_batch_partials.device_ptr(),
                                inv_wp, inv_sp,
                                hs as i32, w13_n as i32, gs as i32, w13_ksplits as i32,
                            ),
                        ).map_err(|e| format!("batched w13 v2[{}]: {:?}", layer_idx, e))?;
                    }

                    // Batched reduce
                    unsafe {
                        k.reduce_ksplits_bf16_batched.clone().launch(
                            LaunchConfig {
                                grid_dim: (((w13_n + 255) / 256) as u32, 1, topk as u32),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                *graph.d_batch_gate_ups.device_ptr(),
                                *graph.d_batch_partials.device_ptr(),
                                w13_n as i32, w13_ksplits as i32,
                            ),
                        ).map_err(|e| format!("batched reduce[{}]: {:?}", layer_idx, e))?;
                    }
                }

                // Batched activation + w2
                // Select kernel based on activation type: 0=silu_gated, 1=relu2
                let is_relu2 = moe.activation_type == 1;
                let w2_n_tiles = (hs + 15) / 16;
                // relu2 input is [K] (up_proj only), silu is [2*K] (gate+up)
                let w2_input_k = if is_relu2 { intermediate } else { intermediate };
                let w2_smem = (w2_input_k * 2 + 1024 * 4 + 64 * 4) as u32;
                let w2_kernel = if is_relu2 {
                    if is_int8 { k.relu2_w2_int8_batched.clone() }
                    else { k.relu2_w2_batched.clone() }
                } else {
                    if is_int8 { k.fused_silu_w2_int8_batched.clone() }
                    else { k.fused_silu_w2_batched.clone() }
                };
                unsafe {
                    w2_kernel.launch(
                        LaunchConfig {
                            grid_dim: (w2_n_tiles as u32, 1, topk as u32),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: w2_smem,
                        },
                        (
                            d_w2p, d_w2s,
                            *graph.d_batch_gate_ups.device_ptr(),
                            *graph.d_batch_expert_outs.device_ptr(),
                            inv_wp, inv_sp,
                            intermediate as i32, hs as i32, gs as i32,
                        ),
                    ).map_err(|e| format!("batched silu_w2[{}]: {:?}", layer_idx, e))?;
                }

                // Multi-expert weighted add
                unsafe {
                    k.multi_expert_weighted_add_bf16.clone().launch(
                        LaunchConfig {
                            grid_dim: (((hs + 255) / 256) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            *graph.d_moe_out.device_ptr(),
                            *graph.d_batch_expert_outs.device_ptr(),
                            d_wts,
                            hs as i32, topk as i32,
                        ),
                    ).map_err(|e| format!("multi_expert_weighted_add[{}]: {:?}", layer_idx, e))?;
                }

                // Shared expert (if any)
                if let Some(ref shared_vram) = graph.shared_expert_vram.get(layer_idx).and_then(|s| s.as_ref()) {
                    let sw13p = shared_vram.w13_packed_ptr();
                    let sw13s = shared_vram.w13_scales_ptr();
                    let sw2p = shared_vram.w2_packed_ptr();
                    let sw2s = shared_vram.w2_scales_ptr();

                    // Shared gate (if applicable)
                    if let Some(ref moe_data) = graph.moe_layers[layer_idx] {
                        if let Some(sg_wid) = moe_data.shared_gate_wid {
                            let sg_w = &graph.weights[sg_wid];
                            let alpha: f32 = 1.0;
                            let beta: f32 = 0.0;
                            unsafe {
                                cublas_result::gemm_ex(
                                    *self.blas.handle(),
                                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                    sg_w.rows as i32, 1, sg_w.cols as i32,
                                    &alpha as *const f32 as *const std::ffi::c_void,
                                    sg_w.ptr as *const std::ffi::c_void,
                                    cublas_sys::cudaDataType::CUDA_R_16BF, sg_w.cols as i32,
                                    *graph.d_hidden.device_ptr() as *const std::ffi::c_void,
                                    cublas_sys::cudaDataType::CUDA_R_16BF, hs as i32,
                                    &beta as *const f32 as *const std::ffi::c_void,
                                    *graph.d_fp32_scratch.device_ptr() as *mut std::ffi::c_void,
                                    cublas_sys::cudaDataType::CUDA_R_32F, sg_w.rows as i32,
                                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                                ).map_err(|e| format!("shared gate GEMV[{}]: {:?}", layer_idx, e))?;
                            }
                        }
                    }

                    // w13 GEMV for shared expert
                    if use_v2_w13 {
                        self.launch_marlin_gemv_v2(
                            sw13p, sw13s,
                            *graph.d_hidden.device_ptr(),
                            partial_ptr, inv_wp, inv_sp,
                            hs, w13_n, gs, w13_ksplits, &k, is_int8,
                        ).map_err(|e| format!("shared w13 v2: {:?}", e))?;
                        self.launch_reduce_ksplits_bf16(
                            *graph.d_expert_gate_up.device_ptr(),
                            partial_ptr, w13_n, w13_ksplits, &k,
                        ).map_err(|e| format!("shared reduce: {:?}", e))?;
                    } else {
                        self.launch_marlin_gemv_raw(
                            sw13p, sw13s,
                            *graph.d_hidden.device_ptr(),
                            *graph.d_expert_gate_up.device_ptr(),
                            inv_wp, inv_sp, hs, w13_n, gs, is_int8,
                        ).map_err(|e| format!("shared w13 raw: {:?}", e))?;
                    }

                    let shared_gate_ptr = if graph.moe_layers[layer_idx].as_ref()
                        .and_then(|m| m.shared_gate_wid).is_some() {
                        *graph.d_fp32_scratch.device_ptr()
                    } else { 0 };

                    self.launch_fused_silu_accum(
                        sw2p, sw2s,
                        *graph.d_expert_gate_up.device_ptr(),
                        *graph.d_moe_out.device_ptr(),
                        inv_wp, inv_sp, intermediate, hs, gs,
                        1.0, shared_gate_ptr, &k, is_int8,
                    ).map_err(|e| format!("shared silu_accum: {:?}", e))?;
                }

                // Scale by routed_scaling_factor
                if rsf != 1.0 {
                    let threads = 256u32;
                    let blocks = ((hs as u32) + threads - 1) / threads;
                    unsafe {
                        k.scale_bf16.clone().launch(
                            LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                            (*graph.d_moe_out.device_ptr(), *graph.d_moe_out.device_ptr(), rsf, hs as i32),
                        ).map_err(|e| format!("scale_bf16[{}]: {:?}", layer_idx, e))?;
                    }
                }

                // Copy MoE output to hidden state
                unsafe {
                    cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                        *graph.d_hidden.device_ptr(),
                        *graph.d_moe_out.device_ptr(),
                        hs * 2, cu_stream);
                }
            }
        }

        // Track first_residual state for routing layers
        let seg_start = graph.segment_layer_start;
        let seg_gqa_offset = graph.segment_gqa_cache_offset;
        let mut first_residual = include_embedding && seg_start == 0;
        let mut gqa_cache_idx = seg_gqa_offset;

        // Count GQA layers before our routing range start (from segment start, not 0)
        if let Some((start, _)) = routing_range {
            gqa_cache_idx = seg_gqa_offset;
            for i in seg_start..start {
                if let GpuAttnConfig::GQA { .. } = &graph.layers[i].attn {
                    gqa_cache_idx += 1;
                }
            }
            if !include_embedding || start > 0 {
                first_residual = false;
            }
        }

        // ── Embedding lookup ──
        if include_embedding {
            let d_token_id_ptr = *graph.d_graph_token_id.as_ref().unwrap().device_ptr();
            let threads = 256u32;
            let blocks = ((hs as u32) + threads - 1) / threads;
            unsafe {
                k.embedding_lookup_g.clone().launch(
                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                    (
                        *graph.d_hidden.device_ptr(),
                        graph.embedding_ptr,
                        d_token_id_ptr,
                        hs as i32,
                    ),
                ).map_err(|e| format!("embedding_lookup_g: {:?}", e))?;
            }
        }

        // ── Routing layers (norm + attention + post-norm + gate + topk for MoE layers) ──
        if let Some((range_start, range_end)) = routing_range {
            for layer_idx in range_start..=range_end {
                let layer = &graph.layers[layer_idx];

                // Pre-attention norm
                {
                    let smem = (hs as u32) * 4;
                    let threads = 256u32.min(hs as u32);
                    unsafe {
                        k.fused_add_rmsnorm.clone().launch(
                            LaunchConfig { grid_dim: (1, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: smem },
                            (
                                *graph.d_hidden.device_ptr(),
                                *graph.d_residual.device_ptr(),
                                layer.input_norm_ptr,
                                eps, hs as i32,
                                if first_residual { 1i32 } else { 0i32 },
                            ),
                        ).map_err(|e| format!("fused_add_rmsnorm[{}]: {:?}", layer_idx, e))?;
                    }
                }
                first_residual = false;

                // Attention (LA or GQA)
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

                        // LA projections (cuBLAS)
                        let qkvz_w = &graph.weights[*in_proj_qkvz];
                        let ba_w = &graph.weights[*in_proj_ba];
                        self.gemv_bf16_to_f32(qkvz_w, *graph.d_hidden.device_ptr(),
                            *graph.d_la_qkvz.device_ptr())?;
                        self.gemv_bf16_to_f32(ba_w, *graph.d_hidden.device_ptr(),
                            *graph.d_la_ba.device_ptr())?;

                        // Un-interleave QKVZ
                        {
                            let group_dim = 2 * dk_ + 2 * hr_ * dv_;
                            let total = nk_ * group_dim;
                            let threads = 256u32;
                            let blocks = ((total as u32) + threads - 1) / threads;
                            unsafe {
                                k.uninterleave_qkvz.clone().launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (
                                        *graph.d_la_conv_out.device_ptr(),
                                        *graph.d_la_recur_out.device_ptr(),
                                        *graph.d_la_qkvz.device_ptr(),
                                        nk_ as i32, dk_ as i32, hr_ as i32, dv_ as i32,
                                    ),
                                ).map_err(|e| format!("uninterleave_qkvz[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // Save z
                        {
                            let z_size = nv_ * dv_;
                            unsafe {
                                cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                                    *graph.d_la_gated_out.device_ptr(),
                                    *graph.d_la_recur_out.device_ptr(),
                                    z_size * 4, cu_stream);
                            }
                        }

                        // Conv1d (with SiLU)
                        {
                            let threads = 256u32;
                            let blocks = ((cd as u32) + threads - 1) / threads;
                            unsafe {
                                k.la_conv1d.clone().launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (
                                        *conv_state_ptr,
                                        *graph.d_la_conv_out.device_ptr(),
                                        *graph.d_la_qkvz.device_ptr(),
                                        *conv_weight_ptr,
                                        cd as i32, kd as i32,
                                    ),
                                ).map_err(|e| format!("la_conv1d[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // Gate and beta from BA
                        let gate_ptr_local: u64;
                        let beta_ptr_local: u64;
                        {
                            let threads = 256u32;
                            let blocks = ((nv_ as u32) + threads - 1) / threads;
                            gate_ptr_local = *graph.d_la_conv_out.device_ptr();
                            beta_ptr_local = unsafe { (*graph.d_la_conv_out.device_ptr() as *const f32).add(nv_) as u64 };
                            unsafe {
                                k.compute_gate_beta.clone().launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (
                                        gate_ptr_local, beta_ptr_local,
                                        *graph.d_la_ba.device_ptr(),
                                        *a_log_ptr, *dt_bias_ptr,
                                        nv_ as i32, hr_ as i32,
                                    ),
                                ).map_err(|e| format!("compute_gate_beta[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // Fused: repeat-interleave + l2norm + delta_net + rmsnorm → BF16
                        {
                            let q_conv_ptr = *graph.d_la_qkvz.device_ptr();
                            let k_conv_ptr = unsafe { (*graph.d_la_qkvz.device_ptr() as *const f32).add(key_dim) as u64 };
                            let v_conv_ptr = unsafe { (*graph.d_la_qkvz.device_ptr() as *const f32).add(2 * key_dim) as u64 };
                            let threads = 256u32;
                            let smem = ((dk_ * 2 + dv_ + 32) as u32) * 4;
                            unsafe {
                                k.la_fused_post_proj.clone().launch(
                                    LaunchConfig { grid_dim: (nv_ as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: smem },
                                    (
                                        *recur_state_ptr,
                                        q_conv_ptr, k_conv_ptr, v_conv_ptr,
                                        gate_ptr_local, beta_ptr_local,
                                        *graph.d_la_gated_out.device_ptr(),
                                        *norm_weight_ptr,
                                        *graph.d_scratch.device_ptr(),
                                        *scale, eps,
                                        ((((nv_ << 16) | dk_) as i64) << 32) | (((dv_ << 16) | hr_) as i64 & 0xFFFFFFFF_i64),
                                    ),
                                ).map_err(|e| format!("la_fused_post_proj[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // Output projection
                        let o_w = &graph.weights[*out_proj];
                        self.gemv_bf16_internal(o_w, *graph.d_scratch.device_ptr(),
                            *graph.d_hidden.device_ptr())?;
                    }

                    GpuAttnConfig::GQA {
                        q_proj, k_proj, v_proj, o_proj,
                        fused_qkv,
                        num_heads, num_kv_heads, head_dim, sm_scale,
                        q_norm_ptr, k_norm_ptr, gated,
                    } => {
                        let nh = *num_heads; let nkv = *num_kv_heads;
                        let hd = *head_dim; let half_dim = graph.rope_half_dim;
                        let kv_stride = nkv * hd;

                        // QKV projection
                        if let Some(fid) = fused_qkv {
                            let fw = &graph.weights[*fid];
                            self.gemv_bf16_to_f32(fw, *graph.d_hidden.device_ptr(),
                                *graph.d_gqa_q.device_ptr())?;
                            let q_size = if *gated { nh * hd * 2 } else { nh * hd };
                            let k_offset = q_size;
                            let v_offset = k_offset + kv_stride;
                            unsafe {
                                cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                                    *graph.d_gqa_k.device_ptr(),
                                    (*graph.d_gqa_q.device_ptr() as *const f32).add(k_offset) as u64,
                                    kv_stride * 4, cu_stream);
                                cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                                    *graph.d_gqa_v.device_ptr(),
                                    (*graph.d_gqa_q.device_ptr() as *const f32).add(v_offset) as u64,
                                    kv_stride * 4, cu_stream);
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

                        // Split gated Q
                        if *gated {
                            let total = (nh * hd) as u32;
                            let threads = 256u32;
                            let blocks = (total + threads - 1) / threads;
                            unsafe {
                                k.split_gated_q.clone().launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (
                                        *graph.d_gqa_q.device_ptr(),
                                        *graph.d_la_qkvz.device_ptr(),
                                        *graph.d_gqa_q.device_ptr(),
                                        nh as i32, hd as i32,
                                    ),
                                ).map_err(|e| format!("split_gated_q[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // QK norm
                        if *q_norm_ptr != 0 {
                            let threads = 256u32;
                            let cfg = LaunchConfig { grid_dim: (nh as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
                            unsafe {
                                k.per_head_rmsnorm.clone().launch(cfg, (
                                    *graph.d_gqa_q.device_ptr(), *q_norm_ptr, eps,
                                    nh as i32, hd as i32, 0i32,
                                )).map_err(|e| format!("per_head_rmsnorm Q[{}]: {:?}", layer_idx, e))?;
                            }
                        }
                        if *k_norm_ptr != 0 {
                            let threads = 256u32;
                            let cfg = LaunchConfig { grid_dim: (nkv as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
                            unsafe {
                                k.per_head_rmsnorm.clone().launch(cfg, (
                                    *graph.d_gqa_k.device_ptr(), *k_norm_ptr, eps,
                                    nkv as i32, hd as i32, 0i32,
                                )).map_err(|e| format!("per_head_rmsnorm K[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // RoPE
                        if let Some(ref d_cos) = graph.d_rope_cos {
                            let total_work = (nh + nkv) * half_dim;
                            let threads = 256u32;
                            let blocks = ((total_work as u32) + threads - 1) / threads;
                            unsafe {
                                k.apply_rope_g.clone().launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (
                                        *graph.d_gqa_q.device_ptr(),
                                        *graph.d_gqa_k.device_ptr(),
                                        *d_cos.device_ptr(),
                                        *graph.d_rope_sin.as_ref().unwrap().device_ptr(),
                                        d_pos_ptr,
                                        nh as i32, nkv as i32, hd as i32, half_dim as i32,
                                    ),
                                ).map_err(|e| format!("apply_rope_g[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // KV cache write
                        if graph.kv_format == 2 {
                            // Polar4: structured rotation + 4-bit quantization
                            let num_blocks = graph.kv_num_blocks;
                            let threads = 256u32;
                            let blocks = ((num_blocks as u32) + threads - 1) / threads;
                            unsafe {
                                k.kv_cache_write_polar4_g.clone().launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (
                                        graph.kv_k_radius_ptrs[layer_idx],
                                        graph.kv_v_radius_ptrs[layer_idx],
                                        graph.kv_k_angles_ptrs[layer_idx],
                                        graph.kv_v_angles_ptrs[layer_idx],
                                        *graph.d_gqa_k.device_ptr(),
                                        *graph.d_gqa_v.device_ptr(),
                                        d_pos_ptr, kv_stride as i32,
                                    ),
                                ).map_err(|e| format!("kv_cache_write_polar4_g[{}]: {:?}", layer_idx, e))?;
                            }
                        } else {
                            let threads = 256u32;
                            let blocks = ((kv_stride as u32) + threads - 1) / threads;
                            unsafe {
                                k.kv_cache_write_g.clone().launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (
                                        graph.kv_k_ptrs[layer_idx],
                                        graph.kv_v_ptrs[layer_idx],
                                        *graph.d_gqa_k.device_ptr(),
                                        *graph.d_gqa_v.device_ptr(),
                                        d_pos_ptr, kv_stride as i32,
                                    ),
                                ).map_err(|e| format!("kv_cache_write_g[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // GQA attention
                        if graph.kv_format == 2 {
                            // Polar4 attention: tiled + reduce (same SM parallelism as FP8)
                            let threads = 256u32;
                            let tile_size = graph.gqa_tile_size;
                            let max_tiles = graph.gqa_max_tiles;

                            let tiled_o = graph.d_gqa_tiled_o.as_ref()
                                .ok_or_else(|| format!("gqa_attention_polar4_tiled_g[{}]: tiled buffers not allocated", layer_idx))?;
                            let tiled_lse = graph.d_gqa_tiled_lse.as_ref().unwrap();
                            if tile_size == 0 || max_tiles == 0 {
                                return Err(format!("gqa_attention_polar4_tiled_g[{}]: tile_size={} max_tiles={} invalid", layer_idx, tile_size, max_tiles));
                            }
                            let tile_smem = ((tile_size + hd) as u32) * 4 + 128;
                            unsafe {
                                k.gqa_attention_polar4_tiled_g.clone().launch(
                                    LaunchConfig {
                                        grid_dim: (nh as u32, max_tiles as u32, 1),
                                        block_dim: (threads, 1, 1),
                                        shared_mem_bytes: tile_smem,
                                    },
                                    (
                                        *tiled_o.device_ptr(),
                                        *tiled_lse.device_ptr(),
                                        *graph.d_gqa_q.device_ptr(),
                                        graph.kv_k_radius_ptrs[layer_idx],
                                        graph.kv_v_radius_ptrs[layer_idx],
                                        graph.kv_k_angles_ptrs[layer_idx],
                                        graph.kv_v_angles_ptrs[layer_idx],
                                        *sm_scale,
                                        nkv as i32, hd as i32,
                                        d_seq_len_ptr,
                                        tile_size as i32,
                                    ),
                                ).map_err(|e| format!("gqa_attention_polar4_tiled_g[{}]: {:?}", layer_idx, e))?;

                                let reduce_smem = ((max_tiles + hd) as u32) * 4;
                                k.gqa_attention_polar4_reduce_g.clone().launch(
                                    LaunchConfig {
                                        grid_dim: (nh as u32, 1, 1),
                                        block_dim: (threads, 1, 1),
                                        shared_mem_bytes: reduce_smem,
                                    },
                                    (
                                        *graph.d_gqa_out.device_ptr(),
                                        *tiled_o.device_ptr(),
                                        *tiled_lse.device_ptr(),
                                        nh as i32, hd as i32,
                                        d_seq_len_ptr,
                                        tile_size as i32,
                                        max_tiles as i32,
                                    ),
                                ).map_err(|e| format!("gqa_attention_polar4_reduce_g[{}]: {:?}", layer_idx, e))?;
                            }
                        } else {
                            // FP8 attention (tiled + reduce for SM utilization + single K read)
                            let threads = 256u32;
                            let tile_size = graph.gqa_tile_size;
                            let max_tiles = graph.gqa_max_tiles;

                            let tiled_o = graph.d_gqa_tiled_o.as_ref()
                                .ok_or_else(|| format!("gqa_attention_tiled_g[{}]: tiled buffers not allocated", layer_idx))?;
                            let tiled_lse = graph.d_gqa_tiled_lse.as_ref().unwrap();
                            if tile_size == 0 || max_tiles == 0 {
                                return Err(format!("gqa_attention_tiled_g[{}]: tile_size={} max_tiles={} invalid", layer_idx, tile_size, max_tiles));
                            }
                            let tile_smem = ((tile_size + hd) as u32) * 4 + 128;
                            unsafe {
                                k.gqa_attention_tiled_g.clone().launch(
                                    LaunchConfig {
                                        grid_dim: (nh as u32, max_tiles as u32, 1),
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
                                        nh as i32, nkv as i32, hd as i32,
                                        d_seq_len_ptr,
                                        tile_size as i32,
                                        max_tiles as i32,
                                    ),
                                ).map_err(|e| format!("gqa_attention_tiled_g[{}]: {:?}", layer_idx, e))?;

                                let reduce_smem = (max_tiles as u32) * 4;
                                k.gqa_attention_reduce_g.clone().launch(
                                    LaunchConfig {
                                        grid_dim: (nh as u32, 1, 1),
                                        block_dim: (threads, 1, 1),
                                        shared_mem_bytes: reduce_smem,
                                    },
                                    (
                                        *graph.d_gqa_out.device_ptr(),
                                        *tiled_o.device_ptr(),
                                        *tiled_lse.device_ptr(),
                                        nh as i32, hd as i32,
                                        d_seq_len_ptr,
                                        tile_size as i32,
                                        max_tiles as i32,
                                    ),
                                ).map_err(|e| format!("gqa_attention_reduce_g[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // Gated attention + BF16 conversion (or just BF16 conversion if non-gated)
                        let attn_out_dim = nh * hd;
                        if *gated {
                            let total = attn_out_dim as u32;
                            let threads = 256u32;
                            let blocks = (total + threads - 1) / threads;
                            unsafe {
                                k.apply_gated_attn_bf16.clone().launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (
                                        *graph.d_scratch.device_ptr(),
                                        *graph.d_gqa_out.device_ptr(),
                                        *graph.d_la_qkvz.device_ptr(),
                                        attn_out_dim as i32,
                                    ),
                                ).map_err(|e| format!("apply_gated_attn_bf16[{}]: {:?}", layer_idx, e))?;
                            }
                        } else {
                            unsafe {
                                k.fp32_to_bf16.clone().launch(
                                    LaunchConfig::for_num_elems(attn_out_dim as u32),
                                    (
                                        *graph.d_scratch.device_ptr(),
                                        *graph.d_gqa_out.device_ptr(),
                                        attn_out_dim as i32,
                                    ),
                                ).map_err(|e| format!("fp32_to_bf16[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // O projection
                        let o_w = &graph.weights[*o_proj];
                        self.gemv_bf16_internal(o_w, *graph.d_scratch.device_ptr(),
                            *graph.d_hidden.device_ptr())?;

                        gqa_cache_idx += 1;
                    }

                    GpuAttnConfig::MLA {
                        q_a_proj, q_b_proj, q_a_norm_ptr, q_proj,
                        kv_a_proj, kv_a_norm_ptr,
                        w_kc_ptr, w_vc_ptr, o_proj,
                        num_heads, kv_lora_rank, ckv_cache_dim, qk_nope_dim, qk_rope_dim, v_head_dim,
                        q_lora_rank, sm_scale, rope_interleave,
                        ckv_cache_ptr, kpe_cache_ptr,
                    } => {
                        let nh = *num_heads;
                        let klr = *kv_lora_rank;
                        let ccd = *ckv_cache_dim;
                        let nope = *qk_nope_dim;
                        let rope = *qk_rope_dim;
                        let vhd = *v_head_dim;
                        let q_head_dim = nope + rope;

                        // ── MLA Step 1: Q projection ──
                        if let (Some(qa_id), Some(qb_id)) = (q_a_proj, q_b_proj) {
                            let qa_w = &graph.weights[*qa_id];
                            self.gemv_bf16_to_f32(qa_w, *graph.d_hidden.device_ptr(),
                                *graph.d_gqa_q.device_ptr())?;
                            {
                                let threads = 256u32;
                                let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
                                unsafe {
                                    k.per_head_rmsnorm.clone().launch(cfg, (
                                        *graph.d_gqa_q.device_ptr(), *q_a_norm_ptr, eps,
                                        1i32, *q_lora_rank as i32, 0i32,
                                    )).map_err(|e| format!("mla q_a_norm[{}]: {:?}", layer_idx, e))?;
                                }
                            }
                            unsafe {
                                k.fp32_to_bf16.clone().launch(
                                    LaunchConfig::for_num_elems(*q_lora_rank as u32),
                                    (
                                        *graph.d_scratch.device_ptr(),
                                        *graph.d_gqa_q.device_ptr(),
                                        *q_lora_rank as i32,
                                    ),
                                ).map_err(|e| format!("mla q_a_to_bf16[{}]: {:?}", layer_idx, e))?;
                            }
                            let qb_w = &graph.weights[*qb_id];
                            self.gemv_bf16_to_f32(qb_w, *graph.d_scratch.device_ptr(),
                                *graph.d_gqa_q.device_ptr())?;
                        } else if let Some(qid) = q_proj {
                            let qw = &graph.weights[*qid];
                            self.gemv_bf16_to_f32(qw, *graph.d_hidden.device_ptr(),
                                *graph.d_gqa_q.device_ptr())?;
                        } else {
                            return Err(format!("MLA layer {} has no Q projection", layer_idx));
                        }

                        // ── MLA Step 2: KV projection + norm ──
                        let kva_w = &graph.weights[*kv_a_proj];
                        self.gemv_bf16_to_f32(kva_w, *graph.d_hidden.device_ptr(),
                            *graph.d_mla_kv.device_ptr())?;
                        {
                            let threads = 256u32;
                            let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
                            unsafe {
                                k.per_head_rmsnorm.clone().launch(cfg, (
                                    *graph.d_mla_kv.device_ptr(), *kv_a_norm_ptr, eps,
                                    1i32, klr as i32, 0i32,
                                )).map_err(|e| format!("mla kv_a_norm[{}]: {:?}", layer_idx, e))?;
                            }
                        }
                        let ckv_ptr = *graph.d_mla_kv.device_ptr();
                        let k_pe_ptr = unsafe { (*graph.d_mla_kv.device_ptr() as *const f32).add(klr) as u64 };

                        // ── MLA Step 3: Split Q → q_nope + q_pe ──
                        {
                            let total = (nh * q_head_dim) as u32;
                            let threads = 256u32;
                            let blocks = (total + threads - 1) / threads;
                            unsafe {
                                k.mla_split_q.clone().launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (
                                        *graph.d_gqa_k.device_ptr(),
                                        *graph.d_gqa_v.device_ptr(),
                                        *graph.d_gqa_q.device_ptr(),
                                        nh as i32, nope as i32, rope as i32,
                                    ),
                                ).map_err(|e| format!("mla_split_q[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // ── MLA Step 4: De-interleave (conditional) ──
                        if *rope_interleave {
                            {
                                let total_q = (nh * rope) as u32;
                                let threads = 256u32;
                                let blocks = (total_q + threads - 1) / threads;
                                unsafe {
                                    k.mla_deinterleave.clone().launch(
                                        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                        (
                                            *graph.d_gqa_v.device_ptr(),
                                            total_q as i32,
                                            rope as i32,
                                        ),
                                    ).map_err(|e| format!("mla deinterleave q_pe[{}]: {:?}", layer_idx, e))?;
                                }
                            }
                            {
                                let threads = 256u32;
                                let blocks = ((rope as u32) + threads - 1) / threads;
                                unsafe {
                                    k.mla_deinterleave.clone().launch(
                                        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                        (
                                            k_pe_ptr,
                                            rope as i32,
                                            rope as i32,
                                        ),
                                    ).map_err(|e| format!("mla deinterleave k_pe[{}]: {:?}", layer_idx, e))?;
                                }
                            }
                        }

                        // ── MLA Step 5: RoPE (graphable: reads position from GPU buffer) ──
                        if let Some(ref d_cos) = graph.d_rope_cos {
                            if let Some(ref d_sin) = graph.d_rope_sin {
                                let half_dim = rope / 2;
                                let total_work = (nh + 1) * half_dim;
                                let threads = 256u32;
                                let blocks = ((total_work as u32) + threads - 1) / threads;
                                unsafe {
                                    k.apply_rope_g.clone().launch(
                                        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                        (
                                            *graph.d_gqa_v.device_ptr(),
                                            k_pe_ptr,
                                            *d_cos.device_ptr(),
                                            *d_sin.device_ptr(),
                                            d_pos_ptr,
                                            nh as i32, 1i32, rope as i32, half_dim as i32,
                                        ),
                                    ).map_err(|e| format!("mla apply_rope_g[{}]: {:?}", layer_idx, e))?;
                                }
                            }
                        }

                        // ── MLA Step 6: Absorb w_kc ──
                        {
                            let threads = 256u32;
                            unsafe {
                                k.mla_absorb_wkc.clone().launch(
                                    LaunchConfig { grid_dim: (nh as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (
                                        *graph.d_mla_q_absorbed.device_ptr(),
                                        *graph.d_gqa_k.device_ptr(),
                                        *w_kc_ptr,
                                        nh as i32, nope as i32, ccd as i32,
                                    ),
                                ).map_err(|e| format!("mla_absorb_wkc[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // ── MLA Step 7: Write to FP8 cache (graphable) ──
                        {
                            let total = (ccd + rope) as u32;
                            let threads = 256u32;
                            let blocks = (total + threads - 1) / threads;
                            unsafe {
                                k.mla_kv_cache_write_g.clone().launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (
                                        *ckv_cache_ptr,
                                        *kpe_cache_ptr,
                                        ckv_ptr,
                                        k_pe_ptr,
                                        d_pos_ptr,
                                        klr as i32,
                                        ccd as i32,
                                        rope as i32,
                                    ),
                                ).map_err(|e| format!("mla_kv_cache_write_g[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // ── MLA Step 8: Attention (graphable: reads seq_len from GPU buffer) ──
                        {
                            let threads = 256u32;
                            let num_warps = (threads + 31) / 32;
                            let tile_size = 4096u32;
                            let shared_mem = (ccd as u32 + rope as u32 + num_warps + tile_size) * 4;
                            unsafe {
                                k.mla_attention_g.clone().launch(
                                    LaunchConfig {
                                        grid_dim: (nh as u32, 1, 1),
                                        block_dim: (threads, 1, 1),
                                        shared_mem_bytes: shared_mem,
                                    },
                                    (
                                        *graph.d_mla_attn_out.device_ptr(),
                                        *graph.d_mla_q_absorbed.device_ptr(),
                                        *graph.d_gqa_v.device_ptr(),
                                        *ckv_cache_ptr,
                                        *kpe_cache_ptr,
                                        *sm_scale,
                                        nh as i32,
                                        ccd as i32,
                                        rope as i32,
                                        d_seq_len_ptr,
                                        graph.kv_max_seq as i32,
                                    ),
                                ).map_err(|e| format!("mla_attention_g[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // ── MLA Step 9: Apply w_vc ──
                        {
                            let threads = 256u32;
                            unsafe {
                                k.mla_apply_wvc.clone().launch(
                                    LaunchConfig { grid_dim: (nh as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (
                                        *graph.d_scratch.device_ptr(),
                                        *graph.d_mla_attn_out.device_ptr(),
                                        *w_vc_ptr,
                                        nh as i32, vhd as i32, ccd as i32,
                                    ),
                                ).map_err(|e| format!("mla_apply_wvc[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // ── MLA Step 10: O projection ──
                        let ow = &graph.weights[*o_proj];
                        self.gemv_bf16_internal(ow, *graph.d_scratch.device_ptr(),
                            *graph.d_hidden.device_ptr())?;
                    }

                    GpuAttnConfig::Mamba2 { .. } => {
                        // Mamba2 cannot use CUDA graph capture (SSM state changes per token).
                        // This arm should not be reached in graph-captured decode.
                        return Err("Mamba2 layers are not compatible with CUDA graph capture. \
                            Use non-graph decode path.".to_string());
                    }
                }

                // Post-attention norm
                {
                    let smem = (hs as u32) * 4;
                    let threads = 256u32.min(hs as u32);
                    unsafe {
                        k.fused_add_rmsnorm.clone().launch(
                            LaunchConfig { grid_dim: (1, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: smem },
                            (
                                *graph.d_hidden.device_ptr(),
                                *graph.d_residual.device_ptr(),
                                layer.post_attn_norm_ptr,
                                eps, hs as i32, 0i32,
                            ),
                        ).map_err(|e| format!("post_attn_norm[{}]: {:?}", layer_idx, e))?;
                    }
                }

                // Dense MLP (for non-MoE layers in the routing range)
                let is_moe_layer = layer_idx < graph.moe_layers.len()
                    && graph.moe_layers[layer_idx].is_some();
                if !is_moe_layer {
                    if let GpuMlpConfig::Dense { gate_proj, up_proj, down_proj } = &layer.mlp {
                        let gw = &graph.weights[*gate_proj];
                        let uw = &graph.weights[*up_proj];
                        let dw = &graph.weights[*down_proj];
                        let inter = gw.rows;

                        self.gemv_bf16_internal(gw, *graph.d_hidden.device_ptr(),
                            *graph.d_expert_gate_up.device_ptr())?;
                        let up_out_ptr = unsafe {
                            (*graph.d_expert_gate_up.device_ptr() as *const u16).add(inter) as u64
                        };
                        self.gemv_bf16_internal(uw, *graph.d_hidden.device_ptr(), up_out_ptr)?;
                        unsafe {
                            k.silu_mul.clone().launch(
                                LaunchConfig::for_num_elems(inter as u32),
                                (*graph.d_expert_scratch.device_ptr(), *graph.d_expert_gate_up.device_ptr(), inter as i32),
                            ).map_err(|e| format!("silu_mul dense[{}]: {:?}", layer_idx, e))?;
                        }
                        self.gemv_bf16_internal(dw, *graph.d_expert_scratch.device_ptr(),
                            *graph.d_hidden.device_ptr())?;
                    }
                }

                // For MoE layers at the end of the routing range: gate GEMV + topk
                if is_moe_layer && layer_idx == range_end {
                    let moe = graph.moe_layers[layer_idx].as_ref().unwrap();
                    let topk = moe.topk;
                    let ne = moe.num_experts;
                    let sf = moe.scoring_func;
                    let gate_wid = moe.gate_wid;
                    let gate_bias_ptr = moe.gate_bias_ptr;
                    let e_score_corr_ptr = moe.e_score_corr_ptr;

                    // Gate GEMV
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
                                w.ptr as *const std::ffi::c_void,
                                cublas_sys::cudaDataType::CUDA_R_16BF, w.cols as i32,
                                *graph.d_hidden.device_ptr() as *const std::ffi::c_void,
                                cublas_sys::cudaDataType::CUDA_R_16BF, hs as i32,
                                &beta as *const f32 as *const std::ffi::c_void,
                                logits_ptr as *mut std::ffi::c_void,
                                cublas_sys::cudaDataType::CUDA_R_32F, w.rows as i32,
                                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                            ).map_err(|e| format!("gate GEMV[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    // TopK routing
                    let topk_ids_dptr = if let Some(ref pm) = graph.pinned_topk_ids {
                        pm.device_ptr
                    } else {
                        *graph.d_topk_indices.device_ptr()
                    };
                    let topk_wts_dptr = if let Some(ref pm) = graph.pinned_topk_weights {
                        pm.device_ptr
                    } else {
                        *graph.d_topk_weights.device_ptr()
                    };
                    {
                        let smem = (ne as u32) * 4;
                        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: smem };
                        if sf == 1 {
                            let bias_ptr = if gate_bias_ptr != 0 { gate_bias_ptr } else { 0u64 };
                            let corr_ptr = if e_score_corr_ptr != 0 { e_score_corr_ptr } else { 0u64 };
                            unsafe {
                                k.sigmoid_topk.clone().launch(cfg, (
                                    logits_ptr, bias_ptr, corr_ptr,
                                    topk_ids_dptr, topk_wts_dptr,
                                    ne as i32, topk as i32,
                                )).map_err(|e| format!("sigmoid_topk[{}]: {:?}", layer_idx, e))?;
                            }
                        } else {
                            unsafe {
                                k.softmax_topk.clone().launch(cfg, (
                                    logits_ptr, topk_ids_dptr, topk_wts_dptr,
                                    ne as i32, topk as i32,
                                )).map_err(|e| format!("softmax_topk[{}]: {:?}", layer_idx, e))?;
                            }
                        }
                    }

                    // GPU-side expert classification: check HCS table and prepare batch upload
                    if graph.gpu_route_sync {
                        if let Some(ref hcs) = graph.hcs {
                            if let Some(ref d_eptrs) = hcs.d_expert_ptrs {
                                let d_eptrs_ptr = *d_eptrs.device_ptr();
                                let d_upload_ptr = *graph.d_batch_upload.device_ptr();
                                let cold_buf_dptr = graph.mapped_cold_buf.as_ref().unwrap().device_ptr;
                                let max_ept = graph.max_experts_per_tok;

                                // Get dummy expert pointers
                                let dummy_base = graph.d_dummy_expert.as_ref()
                                    .map(|b| *b.device_ptr()).unwrap_or(0);

                                // Validate: layer_idx must be within d_expert_ptrs bounds
                                let table_entries = hcs.d_expert_ptrs_nl * hcs.d_expert_ptrs_ne * 4;
                                let max_access = (layer_idx * ne + (ne - 1)) * 4 + 3;
                                if max_access >= table_entries {
                                    return Err(format!(
                                        "expert_classify[{}]: OOB layer_idx={} ne={} max_access={} table={}",
                                        layer_idx, layer_idx, ne, max_access, table_entries));
                                }

                                // Mapped activations buffer device pointer (0 if not available)
                                let act_dptr = graph.mapped_activations.as_ref()
                                    .map(|m| m.device_ptr).unwrap_or(0);
                                unsafe {
                                    k.expert_classify_prepare.clone().launch(
                                        LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 },
                                        (
                                            topk_ids_dptr,
                                            topk_wts_dptr,
                                            d_eptrs_ptr,
                                            d_upload_ptr,
                                            cold_buf_dptr,
                                            layer_idx as i32,
                                            ne as i32,
                                            topk as i32,
                                            max_ept as i32,
                                            dummy_base,
                                            act_dptr,
                                            moe_seq_idx as i32,
                                        ),
                                    ).map_err(|e| format!("expert_classify[{}]: ptrs=[ids={:#x} wts={:#x} eptrs={:#x} upload={:#x} cold={:#x} dummy={:#x}] err={:?}",
                                        layer_idx, topk_ids_dptr, topk_wts_dptr, d_eptrs_ptr, d_upload_ptr, cold_buf_dptr, dummy_base, e))?;
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Final norm + LM head ──
        if include_final {
            // Final RMSNorm
            {
                let smem = (hs as u32) * 4;
                let threads = 256u32.min(hs as u32);
                unsafe {
                    k.fused_add_rmsnorm.clone().launch(
                        LaunchConfig { grid_dim: (1, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: smem },
                        (
                            *graph.d_hidden.device_ptr(),
                            *graph.d_residual.device_ptr(),
                            graph.final_norm_ptr,
                            eps, hs as i32, 0i32,
                        ),
                    ).map_err(|e| format!("final_norm: {:?}", e))?;
                }
            }

            // LM head GEMV
            {
                let w = &graph.weights[graph.lm_head_wid];
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;
                unsafe {
                    cublas_result::gemm_ex(
                        *self.blas.handle(),
                        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        w.rows as i32, 1, w.cols as i32,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        w.ptr as *const std::ffi::c_void,
                        cublas_sys::cudaDataType::CUDA_R_16BF, w.cols as i32,
                        *graph.d_hidden.device_ptr() as *const std::ffi::c_void,
                        cublas_sys::cudaDataType::CUDA_R_16BF, hs as i32,
                        &beta as *const f32 as *const std::ffi::c_void,
                        *graph.d_logits.device_ptr() as *mut std::ffi::c_void,
                        cublas_sys::cudaDataType::CUDA_R_32F, w.rows as i32,
                        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    ).map_err(|e| format!("lm_head GEMV: {:?}", e))?;
                }
            }
        }

        Ok(())
    }

    /// Replay per-layer CUDA graphs for one decode token.
    /// Between each pair of graphs: sync, read routing, DMA cold experts, populate pointer table.
    /// Replay per-layer CUDA graphs for one decode step.
    /// - do_final: if true, D2H logits after the last graph (single-GPU or final segment).
    ///   If false, skip logits D2H (hidden state will be transferred to next GPU).
    fn replay_per_layer_graphs(
        &mut self,
        token_id: usize,
        position: usize,
        do_final: bool,
    ) -> Result<(), String> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| "No graph configured".to_string())?;
        if !graph.per_layer_graphs_valid || graph.per_layer_graphs.is_empty() {
            return Err("Per-layer graphs not captured".to_string());
        }

        let num_graphs = graph.per_layer_graphs.len();
        let moe_indices = graph.per_layer_moe_indices.clone();
        let replay_stream: cuda_sys::CUstream = *self.device.cu_stream();
        let copy_stream = self.copy_stream.0;

        // Update scalar params (token_id, pos, seq_len)
        {
            let token_id_i32 = token_id as i32;
            let pos_i32 = position as i32;
            let seq_len_i32 = (position + 1) as i32;
            unsafe {
                if let Some(ref buf) = graph.d_graph_token_id {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        *buf.device_ptr(),
                        &token_id_i32 as *const i32 as *const std::ffi::c_void,
                        4, replay_stream);
                }
                if let Some(ref buf) = graph.d_graph_pos {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        *buf.device_ptr(),
                        &pos_i32 as *const i32 as *const std::ffi::c_void,
                        4, replay_stream);
                }
                if let Some(ref buf) = graph.d_graph_seq_len {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        *buf.device_ptr(),
                        &seq_len_i32 as *const i32 as *const std::ffi::c_void,
                        4, replay_stream);
                }
            }
        }

        let hs = graph.hidden_size;
        let max_ept = graph.max_experts_per_tok;
        let use_pinned = graph.pinned_topk_ids.is_some() && graph.pinned_topk_weights.is_some();
        let mapped_reads = graph.mapped_reads_active;

        // Double-buffer base pointers for cold expert DMA
        let buf_base = [
            *graph.d_expert_buf[0].device_ptr(),
            *graph.d_expert_buf[1].device_ptr(),
        ];
        let w13p_off = graph.expert_buf_w13p_offset;
        let w13s_off = graph.expert_buf_w13s_offset;
        let w2p_off = graph.expert_buf_w2p_offset;
        let w2s_off = graph.expert_buf_w2s_offset;

        // GPU-side route sync state
        let gpu_rs = graph.gpu_route_sync;
        let cold_buf_host = if gpu_rs && !mapped_reads {
            graph.mapped_cold_buf.as_ref().map(|m| m.host_ptr as *mut i32)
        } else { None };

        // Pre-clear ready flag before the first graph (only needed for non-mapped GPU route sync)
        if gpu_rs && !mapped_reads {
            unsafe {
                let cold_ptr = cold_buf_host.unwrap();
                std::ptr::write_volatile(cold_ptr.add(1), 0i32);
            }
        }

        if mapped_reads {
            // ═══ MAPPED READS FAST PATH ═══
            // All experts (hot + cold) have valid pointers in d_expert_ptrs.
            // The classify kernel writes pointers to d_batch_upload for ALL experts.
            // No CPU sync, no DMA, no classification between graphs.
            // Just launch all graphs back-to-back on the same stream.
            for graph_idx in 0..num_graphs {
                let exec = &graph.per_layer_graphs[graph_idx];
                let err = unsafe { cuda_sys::lib().cuGraphLaunch(exec.0, replay_stream) };
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("cuGraphLaunch[{}]: {:?}", graph_idx, err));
                }
            }

            // Single final sync — all 49 graphs execute sequentially on the stream
            unsafe {
                let err = cuda_sys::lib().cuStreamSynchronize(replay_stream);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("mapped reads final sync: {:?}", err));
                }
            }


            // D2H logits
            if do_final {
                unsafe {
                    let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                        graph.h_logits.as_mut_ptr() as *mut std::ffi::c_void,
                        *graph.d_logits.device_ptr(),
                        graph.vocab_size * 4);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(format!("D2H logits: {:?}", err));
                    }
                }
            }

            return Ok(());
        }

        // ═══ LEGACY PATH (non-mapped reads) ═══

        for graph_idx in 0..num_graphs {
            // ── If not first graph: populate d_batch_upload with expert pointers ──
            if graph_idx > 0 {
                let moe_layer_idx = moe_indices[graph_idx - 1];
                let topk = graph.moe_layers[moe_layer_idx].as_ref()
                    .map(|m| m.topk).unwrap_or(10);

                if gpu_rs {
                    // ── GPU-side route sync path ──
                    // The classify kernel (part of previous graph) already:
                    // 1. Checked d_expert_ptrs for each topk expert
                    // 2. Wrote HCS-hit pointers directly to d_batch_upload
                    // 3. Wrote cold expert IDs + count to mapped_cold_buf
                    // 4. Filled unused slots with dummy pointers
                    // We just need to wait for it and handle cold experts.

                    let cold_ptr = cold_buf_host.unwrap();

                    // Poll mapped memory ready_flag (CPU spin-wait, ~microseconds)
                    unsafe {
                        let ready_ptr = cold_ptr.add(1) as *const std::sync::atomic::AtomicI32;
                        loop {
                            let val = (*ready_ptr).load(std::sync::atomic::Ordering::Acquire);
                            if val != 0 { break; }
                            std::hint::spin_loop();
                        }
                        // Clear ready flag for next layer
                        (*ready_ptr).store(0, std::sync::atomic::Ordering::Release);
                    }

                    let cold_count = unsafe { std::ptr::read_volatile(cold_ptr) } as usize;

                    // Count hot experts (topk - cold)
                    graph.dma_hcs_experts += (topk - cold_count) as u64;

                    // DMA cold experts into VRAM buffers and update their d_batch_upload slots.
                    // All DMAs are queued on copy_stream without per-expert sync, then a single
                    // event wait ensures all transfers complete before batch pointer updates.
                    if cold_count > 0 {
                        graph.dma_cold_experts += cold_count as u64;
                        let moe_data = graph.moe_layers[moe_layer_idx].as_ref().unwrap();

                        // Collect VRAM buffer destinations for each cold expert
                        let mut cold_bufs: [(u64, u64, u64, u64, usize); 10] = [(0,0,0,0,0); 10];
                        let mut actual_cold = 0usize;

                        for ci in 0..cold_count {
                            let eid = unsafe { std::ptr::read_volatile(cold_ptr.add(2 + ci)) } as usize;
                            let batch_slot = unsafe { std::ptr::read_volatile(cold_ptr.add(2 + topk + ci)) } as usize;
                            let token_idx = graph.validation_decode_steps as usize + 1;
                            validation_record_cold_load(
                                &mut graph.validation_decode_cold_hist,
                                &mut graph.validation_decode_cold_events,
                                moe_layer_idx,
                                eid,
                                token_idx,
                            );
                            let expert = &moe_data.experts[eid];

                            let (w13p, w13s, w2p, w2s) = if ci < 2 {
                                let base = buf_base[ci];
                                (base + w13p_off as u64, base + w13s_off as u64,
                                 base + w2p_off as u64, base + w2s_off as u64)
                            } else if let Some(ref apfl) = graph.apfl {
                                let apfl_idx = ci - 2;
                                if apfl_idx < apfl.slots.len() {
                                    let slot = &apfl.slots[apfl_idx];
                                    (slot.w13_packed_ptr(), slot.w13_scales_ptr(),
                                     slot.w2_packed_ptr(), slot.w2_scales_ptr())
                                } else {
                                    continue;
                                }
                            } else {
                                continue;
                            };

                            // Queue DMA (contiguous or 4-call, NO per-expert sync)
                            unsafe {
                                if expert.contiguous_ptr != 0 && ci < 2 {
                                    // Contiguous DMA: single call for entire expert
                                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                        buf_base[ci], expert.contiguous_ptr as *const std::ffi::c_void,
                                        expert.contiguous_bytes, copy_stream);
                                    graph.dma_bytes_total += expert.contiguous_bytes as u64;
                                    graph.dma_call_count += 1;
                                } else {
                                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                        w13p, expert.w13_packed_ptr as *const std::ffi::c_void,
                                        expert.w13_packed_bytes, copy_stream);
                                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                        w13s, expert.w13_scales_ptr as *const std::ffi::c_void,
                                        expert.w13_scales_bytes, copy_stream);
                                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                        w2p, expert.w2_packed_ptr as *const std::ffi::c_void,
                                        expert.w2_packed_bytes, copy_stream);
                                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                        w2s, expert.w2_scales_ptr as *const std::ffi::c_void,
                                        expert.w2_scales_bytes, copy_stream);
                                    let dma_bytes = expert.w13_packed_bytes + expert.w13_scales_bytes
                                        + expert.w2_packed_bytes + expert.w2_scales_bytes;
                                    graph.dma_bytes_total += dma_bytes as u64;
                                    graph.dma_call_count += 4;
                                }
                            }

                            cold_bufs[actual_cold] = (w13p, w13s, w2p, w2s, batch_slot);
                            actual_cold += 1;
                        }

                        // Single sync after all DMAs complete (replaces N per-expert syncs)
                        unsafe {
                            cuda_sys::lib().cuStreamSynchronize(copy_stream);
                        }

                        // Update batch_upload slots for all cold experts
                        let d_upload_base = *graph.d_batch_upload.device_ptr();
                        for ci in 0..actual_cold {
                            let (w13p, w13s, w2p, w2s, batch_slot) = cold_bufs[ci];
                            if batch_slot < max_ept {
                                let ptrs: [u64; 4] = [w13p, w13s, w2p, w2s];
                                unsafe {
                                    for (arr_idx, &ptr_val) in ptrs.iter().enumerate() {
                                        let dst = d_upload_base + ((arr_idx * max_ept + batch_slot) * 8) as u64;
                                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                            dst,
                                            &ptr_val as *const u64 as *const std::ffi::c_void,
                                            8, replay_stream);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // ── Legacy CPU-side route sync path ──

                    // Sync to get routing results
                    unsafe {
                        let err = cuda_sys::lib().cuStreamSynchronize(replay_stream);
                        if err != cuda_sys::CUresult::CUDA_SUCCESS {
                            return Err(format!("sync routing[{}]: {:?}", moe_layer_idx, err));
                        }
                    }

                    // Read topk indices/weights from GPU
                    if use_pinned {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                graph.pinned_topk_ids.as_ref().unwrap().host_ptr as *const i32,
                                graph.h_topk_ids.as_mut_ptr(), topk);
                            std::ptr::copy_nonoverlapping(
                                graph.pinned_topk_weights.as_ref().unwrap().host_ptr as *const f32,
                                graph.h_topk_weights.as_mut_ptr(), topk);
                        }
                    } else {
                        unsafe {
                            cuda_sys::lib().cuMemcpyDtoH_v2(
                                graph.h_topk_ids.as_mut_ptr() as *mut std::ffi::c_void,
                                *graph.d_topk_indices.device_ptr(), topk * 4);
                            cuda_sys::lib().cuMemcpyDtoH_v2(
                                graph.h_topk_weights.as_mut_ptr() as *mut std::ffi::c_void,
                                *graph.d_topk_weights.device_ptr(), topk * 4);
                        }
                    }

                    // Classify experts: HCS hit vs cold (need DMA)
                    let mut batch_count = 0usize;
                    let mut cold_experts: Vec<(usize, usize, f32)> = Vec::new();
                    let moe_data = graph.moe_layers[moe_layer_idx].as_ref().unwrap();
                    let t_padding_setup = if graph.timing_enabled {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };

                    for i in 0..topk {
                        let eid = graph.h_topk_ids[i];
                        if eid < 0 { continue; }
                        let eid = eid as usize;
                        let weight = graph.h_topk_weights[i];

                        let hcs_ptrs = if let Some(ref hcs) = graph.hcs {
                            hcs.get_fast(moe_layer_idx, eid)
                        } else { None };

                        if let Some((w13p, w13s, w2p, w2s)) = hcs_ptrs {
                            if batch_count < max_ept {
                                graph.h_batch_w13_packed_ptrs[batch_count] = w13p;
                                graph.h_batch_w13_scales_ptrs[batch_count] = w13s;
                                graph.h_batch_w2_packed_ptrs[batch_count] = w2p;
                                graph.h_batch_w2_scales_ptrs[batch_count] = w2s;
                                graph.h_batch_weights[batch_count] = weight;
                                batch_count += 1;
                                graph.dma_hcs_experts += 1;
                            }
                        } else {
                            let token_idx = graph.validation_decode_steps as usize + 1;
                            validation_record_cold_load(
                                &mut graph.validation_decode_cold_hist,
                                &mut graph.validation_decode_cold_events,
                                moe_layer_idx,
                                eid,
                                token_idx,
                            );
                            cold_experts.push((i, eid, weight));
                        }
                    }

                    graph.dma_cold_experts += cold_experts.len() as u64;

                    // Queue ALL cold expert DMAs on copy_stream (no per-expert sync)
                    let mut cold_ptrs_list: Vec<(u64, u64, u64, u64)> = Vec::with_capacity(cold_experts.len());
                    for (ci, &(_topk_pos, eid, _weight)) in cold_experts.iter().enumerate() {
                        let expert = &moe_data.experts[eid];

                        let (_base, w13p, w13s, w2p, w2s) = if ci < 2 {
                            let base = buf_base[ci];
                            (base,
                             base + w13p_off as u64, base + w13s_off as u64,
                             base + w2p_off as u64, base + w2s_off as u64)
                        } else if let Some(ref apfl) = graph.apfl {
                            let apfl_idx = ci - 2;
                            if apfl_idx < apfl.slots.len() {
                                let slot = &apfl.slots[apfl_idx];
                                let base = *slot.d_buf.device_ptr();
                                (base,
                                 slot.w13_packed_ptr(), slot.w13_scales_ptr(),
                                 slot.w2_packed_ptr(), slot.w2_scales_ptr())
                            } else {
                                cold_ptrs_list.push((0, 0, 0, 0));
                                continue;
                            }
                        } else {
                            cold_ptrs_list.push((0, 0, 0, 0));
                            continue;
                        };

                        unsafe {
                            if expert.contiguous_ptr != 0 && ci < 2 {
                                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                    buf_base[ci], expert.contiguous_ptr as *const std::ffi::c_void,
                                    expert.contiguous_bytes, copy_stream);
                                graph.dma_bytes_total += expert.contiguous_bytes as u64;
                                graph.dma_call_count += 1;
                            } else {
                                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                    w13p, expert.w13_packed_ptr as *const std::ffi::c_void,
                                    expert.w13_packed_bytes, copy_stream);
                                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                    w13s, expert.w13_scales_ptr as *const std::ffi::c_void,
                                    expert.w13_scales_bytes, copy_stream);
                                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                    w2p, expert.w2_packed_ptr as *const std::ffi::c_void,
                                    expert.w2_packed_bytes, copy_stream);
                                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                    w2s, expert.w2_scales_ptr as *const std::ffi::c_void,
                                    expert.w2_scales_bytes, copy_stream);
                                let dma_bytes = expert.w13_packed_bytes + expert.w13_scales_bytes
                                    + expert.w2_packed_bytes + expert.w2_scales_bytes;
                                graph.dma_bytes_total += dma_bytes as u64;
                                graph.dma_call_count += 4;
                            }
                        }
                        cold_ptrs_list.push((w13p, w13s, w2p, w2s));
                    }

                    // Single sync after all DMAs (replaces N per-expert syncs)
                    if !cold_experts.is_empty() {
                        unsafe { cuda_sys::lib().cuStreamSynchronize(copy_stream); }
                    }

                    // Add cold experts to batch with their VRAM pointers
                    for (ci, &(_topk_pos, _eid, weight)) in cold_experts.iter().enumerate() {
                        let (w13p, w13s, w2p, w2s) = cold_ptrs_list[ci];
                        if w13p == 0 { continue; } // skipped expert
                        if batch_count < max_ept {
                            graph.h_batch_w13_packed_ptrs[batch_count] = w13p;
                            graph.h_batch_w13_scales_ptrs[batch_count] = w13s;
                            graph.h_batch_w2_packed_ptrs[batch_count] = w2p;
                            graph.h_batch_w2_scales_ptrs[batch_count] = w2s;
                            graph.h_batch_weights[batch_count] = weight;
                            batch_count += 1;
                        }
                    }

                    if batch_count < topk {
                        // Zero-weight padding should not alias real experts in the replayed batch.
                        // Use the prevalidated dummy expert layout cached at graph init so replay
                        // stays deterministic and keeps the replayed expert mix stable across
                        // decode steps without any per-step DtoH traffic.
                        let fill_ptrs = if graph.h_dummy_ptrs[0] != 0 {
                            graph.h_dummy_ptrs
                        } else {
                            let dummy_base = graph.d_dummy_expert.as_ref()
                                .map(|b| *b.device_ptr()).unwrap_or(buf_base[0]);
                            [dummy_base, dummy_base, dummy_base, dummy_base]
                        };
                        for i in batch_count..topk.min(max_ept) {
                            graph.h_batch_w13_packed_ptrs[i] = fill_ptrs[0];
                            graph.h_batch_w13_scales_ptrs[i] = fill_ptrs[1];
                            graph.h_batch_w2_packed_ptrs[i] = fill_ptrs[2];
                            graph.h_batch_w2_scales_ptrs[i] = fill_ptrs[3];
                            graph.h_batch_weights[i] = 0.0;
                        }
                    }

                    let ptr_stride = max_ept * 8;
                    let fill_count = topk.min(max_ept);
                    unsafe {
                        let h = graph.h_batch_upload.as_mut_ptr();
                        std::ptr::copy_nonoverlapping(
                            graph.h_batch_w13_packed_ptrs.as_ptr() as *const u8, h, fill_count * 8);
                        std::ptr::copy_nonoverlapping(
                            graph.h_batch_w13_scales_ptrs.as_ptr() as *const u8, h.add(ptr_stride), fill_count * 8);
                        std::ptr::copy_nonoverlapping(
                            graph.h_batch_w2_packed_ptrs.as_ptr() as *const u8, h.add(ptr_stride * 2), fill_count * 8);
                        std::ptr::copy_nonoverlapping(
                            graph.h_batch_w2_scales_ptrs.as_ptr() as *const u8, h.add(ptr_stride * 3), fill_count * 8);
                        std::ptr::copy_nonoverlapping(
                            graph.h_batch_weights.as_ptr() as *const u8, h.add(ptr_stride * 4), fill_count * 4);

                        let upload_bytes = ptr_stride * 4 + max_ept * 4;
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            *graph.d_batch_upload.device_ptr(),
                            h as *const std::ffi::c_void,
                            upload_bytes, replay_stream);
                    }
                    if let Some(t_padding_setup) = t_padding_setup {
                        graph.t_moe_padding_setup += t_padding_setup.elapsed().as_secs_f64();
                    }
                }
            }

            // ── Pre-graph: clear mapped cold buffer ready flag for GPU classify kernel ──
            if gpu_rs && graph_idx < num_graphs - 1 {
                // The graph about to be replayed contains a classify kernel at the end.
                // Clear ready_flag so the classify kernel can set it when done.
                unsafe {
                    let cold_ptr = cold_buf_host.unwrap();
                    std::ptr::write_volatile(cold_ptr.add(1), 0i32); // clear ready_flag
                }
            }

            // ── Replay graph ──
            let exec = &graph.per_layer_graphs[graph_idx];
            let err = unsafe { cuda_sys::lib().cuGraphLaunch(exec.0, replay_stream) };
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuGraphLaunch[{}]: {:?}", graph_idx, err));
            }
            // DEBUG: sync after each graph to pinpoint crashes
            if std::env::var("KRASIS_GRAPH_DEBUG").is_ok() {
                let sync_err = unsafe { cuda_sys::lib().cuStreamSynchronize(replay_stream) };
                if sync_err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("graph[{}] post-launch sync: {:?}", graph_idx, sync_err));
                }
            }
        }

        // Final sync
        unsafe {
            let err = cuda_sys::lib().cuStreamSynchronize(replay_stream);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("final sync: {:?}", err));
            }
        }

        // D2H logits (only if this is the final segment)
        if do_final {
            unsafe {
                let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                    graph.h_logits.as_mut_ptr() as *mut std::ffi::c_void,
                    *graph.d_logits.device_ptr(),
                    graph.vocab_size * 4);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("D2H logits: {:?}", err));
                }
            }
        }

        Ok(())
    }

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

        // Decode NaN diagnostic: check d_hidden for NaN after each component
        let decode_nan_diag = std::env::var("KRASIS_DECODE_DIAG").map(|v| v == "1").unwrap_or(false);
        let check_nan_bf16 = |label: &str, ptr: u64, n: usize, device: &std::sync::Arc<cudarc::driver::CudaDevice>| {
            if !decode_nan_diag { return; }
            let _ = device.synchronize();
            let mut buf = vec![0u16; n.min(64)];
            let dl = buf.len();
            unsafe {
                let _ = cuda_sys::lib().cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut std::ffi::c_void,
                    ptr, dl * 2);
            }
            let has_nan = buf.iter().any(|&b| {
                let bits = (b as u32) << 16;
                f32::from_bits(bits).is_nan()
            });
            if has_nan {
                let vals: Vec<f32> = buf[..4.min(dl)].iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
                eprintln!("[DECODE-NaN] {} NaN detected! first4=[{:.4},{:.4},{:.4},{:.4}]",
                    label, vals[0], vals[1], vals[2], vals[3]);
            }
        };
        let check_nan_f32 = |label: &str, ptr: u64, n: usize, device: &std::sync::Arc<cudarc::driver::CudaDevice>| {
            if !decode_nan_diag { return; }
            let _ = device.synchronize();
            let mut buf = vec![0.0f32; n.min(64)];
            let dl = buf.len();
            unsafe {
                let _ = cuda_sys::lib().cuMemcpyDtoH_v2(
                    buf.as_mut_ptr() as *mut std::ffi::c_void,
                    ptr, dl * 4);
            }
            let has_nan = buf.iter().any(|f| f.is_nan());
            if has_nan {
                eprintln!("[DECODE-NaN] {} NaN detected! first4=[{:.4},{:.4},{:.4},{:.4}]",
                    label, buf[0], buf[1], buf[2], buf[3]);
            }
        };

        // ── Multi-GPU segment config ──
        let seg_skip_emb = graph.segment_skip_embedding;
        let seg_skip_final = graph.segment_skip_final;
        let seg_start = graph.segment_layer_start;
        let seg_end = if graph.segment_layer_end > 0 { graph.segment_layer_end } else { graph.layers.len() };
        let seg_gqa_offset = graph.segment_gqa_cache_offset;

        // ── 1. Embedding lookup (skip if this segment receives uploaded hidden state) ──
        if !seg_skip_emb {
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
            check_nan_bf16("after_embedding", *graph.d_hidden.device_ptr(), hs, &self.device);
        }

        // first_residual: true only when embedding was done (layer 0 initializes residual stream)
        let mut first_residual = !seg_skip_emb && seg_start == 0;
        let num_layers = graph.layers.len();
        let mut gqa_cache_idx = seg_gqa_offset; // Start at offset for multi-GPU segments

        // Timing accumulators for this token
        let mut tt_attn = 0.0f64;
        let mut tt_moe = 0.0f64;
        let mut tt_norm = 0.0f64;
        let mut tt_shared = 0.0f64;
        let mut tt_dense_mlp = 0.0f64;

        // ── 2. Layer loop (respects multi-GPU segment bounds) ──
        for layer_idx in seg_start..seg_end.min(num_layers) {
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
            check_nan_bf16(&format!("L{} post_pre_attn_norm", layer_idx), *graph.d_hidden.device_ptr(), hs, &self.device);

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
                    let t_la_s1 = Instant::now();
                    let qkvz_w = &graph.weights[*in_proj_qkvz];
                    let ba_w = &graph.weights[*in_proj_ba];
                    self.gemv_bf16_to_f32(
                        qkvz_w, *graph.d_hidden.device_ptr(),
                        *graph.d_la_qkvz.device_ptr())?;
                    self.gemv_bf16_to_f32(
                        ba_w, *graph.d_hidden.device_ptr(),
                        *graph.d_la_ba.device_ptr())?;
                    if layer_idx < 2 {
                        check_nan_f32(&format!("L{} la_qkvz_proj", layer_idx), *graph.d_la_qkvz.device_ptr(), 64, &self.device);
                        check_nan_f32(&format!("L{} la_ba_proj", layer_idx), *graph.d_la_ba.device_ptr(), 64, &self.device);
                    }

                    if timing {
                        self.device.synchronize().map_err(|e| format!("la proj sync: {:?}", e))?;
                        graph.t_la_proj += (Instant::now() - t_la_s1).as_secs_f64();
                    }
                    let t_la_s2 = Instant::now();

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
                            k.uninterleave_qkvz.clone().launch(
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
                            k.la_conv1d.clone().launch(
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
                    if layer_idx < 2 {
                        check_nan_f32(&format!("L{} la_conv1d_out", layer_idx), *graph.d_la_qkvz.device_ptr(), 64, &self.device);
                    }

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
                            k.compute_gate_beta.clone().launch(
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

                    if layer_idx < 2 {
                        check_nan_f32(&format!("L{} la_gate_beta", layer_idx), gate_ptr_local, 32, &self.device);
                    }

                    if timing {
                        self.device.synchronize().map_err(|e| format!("la conv sync: {:?}", e))?;
                        graph.t_la_conv += (Instant::now() - t_la_s2).as_secs_f64();
                    }
                    let t_la_s5 = Instant::now();

                    // ── LA Steps 5-8 FUSED: repeat-interleave + l2norm + delta_net + rmsnorm → BF16 ──
                    // Conv output in d_la_qkvz: [q(key_dim), k(key_dim), v(nv*dv)]
                    // Gate/beta in d_la_conv_out (from step 4)
                    // Z saved in d_la_gated_out (from step 2)
                    // Output: BF16 in d_scratch (ready for output projection)
                    {
                        let q_conv_ptr = *graph.d_la_qkvz.device_ptr();
                        let k_conv_ptr = unsafe { (*graph.d_la_qkvz.device_ptr() as *const f32).add(key_dim) as u64 };
                        let v_conv_ptr = unsafe { (*graph.d_la_qkvz.device_ptr() as *const f32).add(2 * key_dim) as u64 };
                        let threads = 256u32;
                        // Shared memory: dk*2 (Q+K) + dv + 32 (warp scratch) floats
                        let smem = ((dk_ * 2 + dv_ + 32) as u32) * 4;
                        unsafe {
                            k.la_fused_post_proj.clone().launch(
                                LaunchConfig { grid_dim: (nv_ as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: smem },
                                (
                                    *recur_state_ptr,                   // state [nv, dk, dv]
                                    q_conv_ptr,                         // q from conv output [nk*dk]
                                    k_conv_ptr,                         // k from conv output [nk*dk]
                                    v_conv_ptr,                         // v from conv output [nv*dv]
                                    gate_ptr_local,                     // gate [nv]
                                    beta_ptr_local,                     // beta [nv]
                                    *graph.d_la_gated_out.device_ptr(), // z [nv*dv]
                                    *norm_weight_ptr,                   // norm weight [dv]
                                    *graph.d_scratch.device_ptr(),      // BF16 output
                                    *scale,                             // q_scale
                                    eps,
                                    ((((nv_ << 16) | dk_) as i64) << 32) | (((dv_ << 16) | hr_) as i64 & 0xFFFFFFFF_i64),
                                ),
                            ).map_err(|e| format!("la_fused_post_proj[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    if layer_idx < 2 {
                        check_nan_bf16(&format!("L{} la_fused_post_proj (BF16 out)", layer_idx), *graph.d_scratch.device_ptr(), 64, &self.device);
                    }

                    if timing {
                        self.device.synchronize().map_err(|e| format!("la recur sync: {:?}", e))?;
                        graph.t_la_recur += (Instant::now() - t_la_s5).as_secs_f64();
                    }
                    let t_la_s8 = Instant::now();

                    // ── LA Step 9: Output projection ──
                    let out_w = &graph.weights[*out_proj];
                    if layer_idx < 2 && decode_nan_diag {
                        let _ = self.device.synchronize();
                        eprintln!("[DECODE-DIAG] L{} out_proj: dtype={} rows={} cols={} marlin_int4={} simple_int4={} ptr=0x{:x}",
                            layer_idx, out_w.dtype, out_w.rows, out_w.cols,
                            out_w.is_marlin_int4(), out_w.has_simple_int4(), out_w.ptr);
                        // Check d_scratch input
                        let mut ibuf = vec![0u16; 32];
                        unsafe {
                            let _ = cuda_sys::lib().cuMemcpyDtoH_v2(
                                ibuf.as_mut_ptr() as *mut std::ffi::c_void,
                                *graph.d_scratch.device_ptr(), 64);
                        }
                        let ivals: Vec<f32> = ibuf[..4].iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
                        let i_any_nan = ibuf.iter().any(|&b| f32::from_bits((b as u32) << 16).is_nan());
                        eprintln!("[DECODE-DIAG] L{} out_proj input (d_scratch BF16): [{:.4},{:.4},{:.4},{:.4}] any_nan={}",
                            layer_idx, ivals[0], ivals[1], ivals[2], ivals[3], i_any_nan);
                    }
                    self.gemv_bf16_internal(
                        out_w,
                        *graph.d_scratch.device_ptr(),
                        *graph.d_hidden.device_ptr(),
                    )?;
                    if layer_idx < 2 {
                        check_nan_bf16(&format!("L{} la_out_proj_output", layer_idx), *graph.d_hidden.device_ptr(), hs, &self.device);
                    }
                    if timing {
                        self.device.synchronize().map_err(|e| format!("la out sync: {:?}", e))?;
                        graph.t_la_out += (Instant::now() - t_la_s8).as_secs_f64();
                    }
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

                    // MoE-only layers (Nemotron): 0 heads means skip attention entirely.
                    // The hidden state passes through norm → MoE without attention.
                    if nh == 0 {
                        // Skip attention: attn_out stays zero (via norm, no attn_out added to residual).
                        // hidden is already in d_hidden after norm, ready for MoE.
                    } else {

                    let kv_stride = nkv * hd;
                    let t_gqa_s1 = Instant::now();

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
                            k.split_gated_q.clone().launch(
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
                            k.per_head_rmsnorm.clone().launch(cfg, (
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
                            k.per_head_rmsnorm.clone().launch(cfg, (
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
                                k.apply_rope.clone().launch(cfg, (
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
                    if graph.kv_format == 2 {
                        // Polar4: structured rotation + 4-bit quantization
                        let num_blocks = graph.kv_num_blocks;
                        let threads = 256u32;
                        let blocks = ((num_blocks as u32) + threads - 1) / threads;
                        let cfg = LaunchConfig {
                            grid_dim: (blocks, 1, 1),
                            block_dim: (threads, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        if graph.kv_k_radius_ptrs[layer_idx] == 0 {
                            return Err(format!(
                                "kv_cache_write_polar4[{}]: null polar4 pointer", layer_idx));
                        }
                        unsafe {
                            k.kv_cache_write_polar4.clone().launch(cfg, (
                                graph.kv_k_radius_ptrs[layer_idx],
                                graph.kv_v_radius_ptrs[layer_idx],
                                graph.kv_k_angles_ptrs[layer_idx],
                                graph.kv_v_angles_ptrs[layer_idx],
                                *graph.d_gqa_k.device_ptr(),
                                *graph.d_gqa_v.device_ptr(),
                                position as i32,
                                kv_stride as i32,
                            )).map_err(|e| format!("kv_cache_write_polar4[{}]: {:?}", layer_idx, e))?;
                        }
                    } else {
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
                            k.kv_cache_write.clone().launch(cfg, (
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
                    if timing {
                        self.device.synchronize().map_err(|e| format!("gqa proj sync: {:?}", e))?;
                        graph.t_gqa_proj += (Instant::now() - t_gqa_s1).as_secs_f64();
                    }
                    let t_gqa_attn_start = Instant::now();
                    if graph.kv_format == 2 {
                        // Polar4 attention: single-block-per-head kernel
                        let threads = 256u32;
                        let seq_len = (position + 1) as u32;
                        let num_blocks = graph.kv_num_blocks;
                        // Shared memory: Q scratch + softmax reduction + per-warp partial V sums
                        let num_warps = threads / 32;
                        let shared_mem_bytes = ((hd as u32) * (num_warps + 1) + 2 * num_warps) * 4 + 128;
                        let cfg = LaunchConfig {
                            grid_dim: (nh as u32, 1, 1),
                            block_dim: (threads, 1, 1),
                            shared_mem_bytes,
                        };
                        unsafe {
                            k.gqa_attention_polar4.clone().launch(cfg, (
                                *graph.d_gqa_out.device_ptr(),
                                *graph.d_gqa_q.device_ptr(),
                                graph.kv_k_radius_ptrs[layer_idx],
                                graph.kv_v_radius_ptrs[layer_idx],
                                graph.kv_k_angles_ptrs[layer_idx],
                                graph.kv_v_angles_ptrs[layer_idx],
                                *sm_scale,
                                nh as i32,
                                nkv as i32,
                                hd as i32,
                                seq_len as i32,
                                graph.kv_max_seq as i32,
                            )).map_err(|e| format!("gqa_attention_polar4[{}]: {:?}", layer_idx, e))?;
                        }
                    } else {
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
                            let tile_smem = (tile_size as u32 + hd as u32) * 4 + 128;
                            let tiled_o = graph.d_gqa_tiled_o.as_ref().unwrap();
                            let tiled_lse = graph.d_gqa_tiled_lse.as_ref().unwrap();
                            unsafe {
                                k.gqa_attention_tiled.clone().launch(
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

                                let reduce_smem = (num_tiles as u32) * 4;
                                k.gqa_attention_reduce.clone().launch(
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
                            let q_smem = (hd as u32) * 4;
                            let smem_threshold = graph.gqa_max_smem_bytes.saturating_sub(128 + q_smem) / 4;
                            let use_smem = seq_len <= smem_threshold;
                            let shared_mem_bytes = if use_smem {
                                q_smem + seq_len * 4 + 128
                            } else {
                                q_smem + 128
                            };
                            let cfg = LaunchConfig {
                                grid_dim: (nh as u32, 1, 1),
                                block_dim: (threads, 1, 1),
                                shared_mem_bytes,
                            };
                            unsafe {
                                k.gqa_attention.clone().launch(cfg, (
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
                    } // end else (non-polar4)

                    if timing {
                        self.device.synchronize().map_err(|e| format!("gqa attn sync: {:?}", e))?;
                        graph.t_gqa_attn += (Instant::now() - t_gqa_attn_start).as_secs_f64();
                    }
                    let t_gqa_out_start = Instant::now();

                    // ── GQA: Apply gated attention + convert to BF16 for O projection ──
                    let o_size = nh * hd;
                    if *gated {
                        // Fused: sigmoid(gate) * attn_out → BF16 (eliminates separate fp32_to_bf16)
                        let total = o_size as u32;
                        let threads = 256u32;
                        let blocks = (total + threads - 1) / threads;
                        unsafe {
                            k.apply_gated_attn_bf16.clone().launch(
                                LaunchConfig {
                                    grid_dim: (blocks, 1, 1),
                                    block_dim: (threads, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                (
                                    *graph.d_scratch.device_ptr(),      // BF16 output
                                    *graph.d_gqa_out.device_ptr(),      // FP32 attention output
                                    *graph.d_la_qkvz.device_ptr(),      // FP32 gate values
                                    o_size as i32,
                                ),
                            ).map_err(|e| format!("apply_gated_attn_bf16[{}]: {:?}", layer_idx, e))?;
                        }
                    } else {
                        // Non-gated: just convert FP32 → BF16
                        unsafe {
                            k.fp32_to_bf16.clone().launch(
                                LaunchConfig::for_num_elems(o_size as u32),
                                (
                                    *graph.d_scratch.device_ptr(),
                                    *graph.d_gqa_out.device_ptr(),
                                    o_size as i32,
                                ),
                            ).map_err(|e| format!("fp32_to_bf16 gqa out[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    // ── GQA: O projection ──
                    let ow = &graph.weights[*o_proj];
                    self.gemv_bf16_internal(
                        ow,
                        *graph.d_scratch.device_ptr(),
                        *graph.d_hidden.device_ptr(),
                    )?;
                    if timing {
                        self.device.synchronize().map_err(|e| format!("gqa out sync: {:?}", e))?;
                        graph.t_gqa_out += (Instant::now() - t_gqa_out_start).as_secs_f64();
                    }

                    gqa_cache_idx += 1;

                    } // end of `if nh > 0` else block (skip MoE-only layers)
                }

                GpuAttnConfig::MLA {
                    q_a_proj, q_b_proj, q_a_norm_ptr, q_proj,
                    kv_a_proj, kv_a_norm_ptr,
                    w_kc_ptr, w_vc_ptr, o_proj,
                    num_heads, kv_lora_rank, ckv_cache_dim, qk_nope_dim, qk_rope_dim, v_head_dim,
                    q_lora_rank, sm_scale, rope_interleave,
                    ckv_cache_ptr, kpe_cache_ptr,
                } => {
                    let nh = *num_heads;
                    let klr = *kv_lora_rank;     // real kv_lora_rank (e.g. 256 for Mistral)
                    let ccd = *ckv_cache_dim;    // padded ckv dim in cache (≥512)
                    let nope = *qk_nope_dim;
                    let rope = *qk_rope_dim;
                    let vhd = *v_head_dim;
                    let q_head_dim = nope + rope;

                    // ── MLA Step 1: Q projection ──
                    if let (Some(qa_id), Some(qb_id)) = (q_a_proj, q_b_proj) {
                        // q_lora path: q_a_proj → RMSNorm → q_b_proj
                        let qa_w = &graph.weights[*qa_id];
                        self.gemv_bf16_to_f32(qa_w, *graph.d_hidden.device_ptr(),
                            *graph.d_gqa_q.device_ptr())?;

                        // RMSNorm on q_a output (1 head of size q_lora_rank)
                        {
                            let threads = 256u32;
                            let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
                            unsafe {
                                k.per_head_rmsnorm.clone().launch(cfg, (
                                    *graph.d_gqa_q.device_ptr(), *q_a_norm_ptr, eps,
                                    1i32, *q_lora_rank as i32, 0i32,
                                )).map_err(|e| format!("mla q_a_norm[{}]: {:?}", layer_idx, e))?;
                            }
                        }

                        // Convert FP32 → BF16 for q_b input
                        unsafe {
                            k.fp32_to_bf16.clone().launch(
                                LaunchConfig::for_num_elems(*q_lora_rank as u32),
                                (
                                    *graph.d_scratch.device_ptr(),
                                    *graph.d_gqa_q.device_ptr(),
                                    *q_lora_rank as i32,
                                ),
                            ).map_err(|e| format!("mla q_a_to_bf16[{}]: {:?}", layer_idx, e))?;
                        }

                        // q_b_proj: BF16 q_lora_rank → FP32 num_heads*(nope+rope)
                        let qb_w = &graph.weights[*qb_id];
                        self.gemv_bf16_to_f32(qb_w, *graph.d_scratch.device_ptr(),
                            *graph.d_gqa_q.device_ptr())?;
                    } else if let Some(qid) = q_proj {
                        let qw = &graph.weights[*qid];
                        self.gemv_bf16_to_f32(qw, *graph.d_hidden.device_ptr(),
                            *graph.d_gqa_q.device_ptr())?;
                    } else {
                        return Err(format!("MLA layer {} has no Q projection", layer_idx));
                    }
                    // d_gqa_q = [num_heads * (nope + rope)] FP32

                    // ── MLA Step 2: KV projection ──
                    let kva_w = &graph.weights[*kv_a_proj];
                    self.gemv_bf16_to_f32(kva_w, *graph.d_hidden.device_ptr(),
                        *graph.d_mla_kv.device_ptr())?;

                    // RMSNorm on ckv portion only (first kv_lora_rank elements)
                    {
                        let threads = 256u32;
                        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
                        unsafe {
                            k.per_head_rmsnorm.clone().launch(cfg, (
                                *graph.d_mla_kv.device_ptr(), *kv_a_norm_ptr, eps,
                                1i32, klr as i32, 0i32,
                            )).map_err(|e| format!("mla kv_a_norm[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    let ckv_ptr = *graph.d_mla_kv.device_ptr();
                    let k_pe_ptr = unsafe { (*graph.d_mla_kv.device_ptr() as *const f32).add(klr) as u64 };

                    // ── MLA Step 3: Split Q → q_nope + q_pe ──
                    // d_gqa_q: [h0_nope|h0_rope|h1_nope|h1_rope|...]
                    // → d_gqa_k: q_nope [nh*nope], d_gqa_v: q_pe [nh*rope]
                    {
                        let total = (nh * q_head_dim) as u32;
                        let threads = 256u32;
                        let blocks = (total + threads - 1) / threads;
                        unsafe {
                            k.mla_split_q.clone().launch(
                                LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                (
                                    *graph.d_gqa_k.device_ptr(),
                                    *graph.d_gqa_v.device_ptr(),
                                    *graph.d_gqa_q.device_ptr(),
                                    nh as i32,
                                    nope as i32,
                                    rope as i32,
                                ),
                            ).map_err(|e| format!("mla_split_q[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    // ── MLA Step 4: De-interleave q_pe and k_pe (conditional) ──
                    if *rope_interleave {
                        // q_pe: [nh * rope] in d_gqa_v
                        {
                            let total_q = (nh * rope) as u32;
                            let threads = 256u32;
                            let blocks = (total_q + threads - 1) / threads;
                            unsafe {
                                k.mla_deinterleave.clone().launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (
                                        *graph.d_gqa_v.device_ptr(),
                                        total_q as i32,
                                        rope as i32,
                                    ),
                                ).map_err(|e| format!("mla deinterleave q_pe[{}]: {:?}", layer_idx, e))?;
                            }
                        }
                        // k_pe: [rope] at k_pe_ptr
                        {
                            let threads = 256u32;
                            let blocks = ((rope as u32) + threads - 1) / threads;
                            unsafe {
                                k.mla_deinterleave.clone().launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (
                                        k_pe_ptr,
                                        rope as i32,
                                        rope as i32,
                                    ),
                                ).map_err(|e| format!("mla deinterleave k_pe[{}]: {:?}", layer_idx, e))?;
                            }
                        }
                    }

                    // ── MLA Step 5: RoPE on q_pe and k_pe ──
                    if let Some(ref d_cos) = graph.d_rope_cos {
                        if let Some(ref d_sin) = graph.d_rope_sin {
                            let half_dim = rope / 2;
                            let total_work = (nh + 1) * half_dim; // nh q heads + 1 k "head"
                            let threads = 256u32;
                            let blocks = ((total_work as u32) + threads - 1) / threads;
                            unsafe {
                                k.apply_rope.clone().launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (
                                        *graph.d_gqa_v.device_ptr(),  // q_pe [nh * rope]
                                        k_pe_ptr,                     // k_pe [rope]
                                        *d_cos.device_ptr(),
                                        *d_sin.device_ptr(),
                                        position as i32,
                                        nh as i32, 1i32, rope as i32, half_dim as i32,
                                    ),
                                ).map_err(|e| format!("mla apply_rope[{}]: {:?}", layer_idx, e))?;
                            }
                        }
                    }

                    // ── MLA Step 6: Absorb w_kc (q_nope @ w_kc → q_absorbed) ──
                    // w_kc is [H, nope, ccd] (padded to ≥512 for MLA decode)
                    {
                        let threads = 256u32;
                        unsafe {
                            k.mla_absorb_wkc.clone().launch(
                                LaunchConfig { grid_dim: (nh as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                (
                                    *graph.d_mla_q_absorbed.device_ptr(),
                                    *graph.d_gqa_k.device_ptr(),
                                    *w_kc_ptr,
                                    nh as i32,
                                    nope as i32,
                                    ccd as i32,  // padded ckv dim
                                ),
                            ).map_err(|e| format!("mla_absorb_wkc[{}]: {:?}", layer_idx, e))?;
                        }
                    }
                    // ── MLA Step 7: Write to FP8 cache ──
                    // Cache has ccd-dim entries; real data is klr, rest zero-padded by kernel
                    {
                        let total = (ccd + rope) as u32;  // ccd for ckv, rope for kpe
                        let threads = 256u32;
                        let blocks = (total + threads - 1) / threads;
                        unsafe {
                            k.mla_kv_cache_write.clone().launch(
                                LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                (
                                    *ckv_cache_ptr,
                                    *kpe_cache_ptr,
                                    ckv_ptr,
                                    k_pe_ptr,
                                    position as i32,
                                    klr as i32,   // real data size
                                    ccd as i32,   // cache stride (padded)
                                    rope as i32,
                                ),
                            ).map_err(|e| format!("mla_kv_cache_write[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    // ── MLA Step 8: MLA attention ──
                    // Uses ccd (padded) for cache reads and q_absorbed dimensions
                    {
                        let threads = 256u32;
                        let num_warps = (threads + 31) / 32;
                        let tile_size = 4096u32; // MLA_TILE_SIZE in CUDA
                        let shared_mem = (ccd as u32 + rope as u32 + num_warps + tile_size) * 4;
                        let attn_seq_len = (position + 1) as i32;
                        unsafe {
                            k.mla_attention.clone().launch(
                                LaunchConfig {
                                    grid_dim: (nh as u32, 1, 1),
                                    block_dim: (threads, 1, 1),
                                    shared_mem_bytes: shared_mem,
                                },
                                (
                                    *graph.d_mla_attn_out.device_ptr(),
                                    *graph.d_mla_q_absorbed.device_ptr(),
                                    *graph.d_gqa_v.device_ptr(),
                                    *ckv_cache_ptr,
                                    *kpe_cache_ptr,
                                    *sm_scale,
                                    nh as i32,
                                    ccd as i32,   // padded ckv dim
                                    rope as i32,
                                    attn_seq_len,
                                    graph.kv_max_seq as i32,
                                ),
                            ).map_err(|e| format!("mla_attention[{}]: {:?}", layer_idx, e))?;
                        }
                    }
                    // ── MLA Step 9: Apply w_vc (attn_out @ w_vc → BF16) ──
                    // w_vc is [H, v_head, ccd] (padded)
                    {
                        let threads = 256u32;
                        unsafe {
                            k.mla_apply_wvc.clone().launch(
                                LaunchConfig { grid_dim: (nh as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                (
                                    *graph.d_scratch.device_ptr(),
                                    *graph.d_mla_attn_out.device_ptr(),
                                    *w_vc_ptr,
                                    nh as i32,
                                    vhd as i32,
                                    ccd as i32,   // padded ckv dim
                                ),
                            ).map_err(|e| format!("mla_apply_wvc[{}]: {:?}", layer_idx, e))?;
                        }
                    }

                    // ── MLA Step 10: O projection ──
                    let ow = &graph.weights[*o_proj];
                    self.gemv_bf16_internal(
                        ow,
                        *graph.d_scratch.device_ptr(),
                        *graph.d_hidden.device_ptr(),
                    )?;
                }

                GpuAttnConfig::Mamba2 {
                    in_proj, out_proj,
                    conv_weight_ptr, a_ptr, d_ptr, dt_bias_ptr, norm_weight_ptr,
                    num_heads, head_dim, state_size, expand: _, conv_kernel, conv_dim,
                    conv_state_ptr, ssm_state_ptr,
                } => {
                    // ── Mamba2 Decode Step ──
                    // 1. in_proj GEMV: hidden -> [z, x_inner, dt, B, C]
                    let in_w = &graph.weights[*in_proj];
                    let in_proj_dim = in_w.rows; // total in_proj output dim
                    let d_inner = num_heads * head_dim;
                    let n_groups = graph.mamba2_n_groups.max(1);

                    // Output goes to d_scratch (large enough for in_proj output)
                    self.gemv_bf16_internal(in_w, *graph.d_hidden.device_ptr(),
                        *graph.d_scratch.device_ptr())?;

                    // Convert in_proj output from BF16 to FP32 in d_fp32_scratch
                    unsafe {
                        k.bf16_to_fp32.clone().launch(
                            LaunchConfig::for_num_elems(in_proj_dim as u32),
                            (*graph.d_fp32_scratch.device_ptr(),
                             *graph.d_scratch.device_ptr(),
                             in_proj_dim as i32),
                        ).map_err(|e| format!("bf16_to_fp32 mamba2 in_proj[{}]: {:?}", layer_idx, e))?;
                    }

                    // Split in_proj output: [z(d_inner), xBC(conv_dim), dt(num_heads)]
                    // xBC = [x(d_inner), B(n_groups*state_size), C(n_groups*state_size)]
                    let fp32_base = *graph.d_fp32_scratch.device_ptr();
                    let z_ptr = fp32_base;
                    let xbc_ptr = fp32_base + (d_inner * 4) as u64;
                    let dt_ptr = fp32_base + ((d_inner + *conv_dim) * 4) as u64;
                    // B and C are after x in the xBC segment
                    let b_ptr = xbc_ptr + (d_inner * 4) as u64;
                    let c_ptr = b_ptr + (n_groups * state_size * 4) as u64;
                    // 2. Conv1d: update conv_state, convolve + SiLU
                    // conv operates on xBC (x + B + C concatenated)
                    // Output overwrites xBC in-place (same buffer, safe since we read first)
                    let conv_out_ptr = graph.d_mamba2_conv_out.as_ref()
                        .map(|b| *b.device_ptr())
                        .unwrap_or(xbc_ptr); // fallback to xBC
                    unsafe {
                        let conv_bias_ptr = graph.mamba2_conv_bias_ptrs
                            .get(&layer_idx).copied().unwrap_or(0u64);
                        k.mamba2_conv1d.clone().launch(
                            LaunchConfig {
                                grid_dim: ((*conv_dim as u32 + 255) / 256, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                *conv_state_ptr,
                                xbc_ptr,
                                *conv_weight_ptr,
                                conv_bias_ptr,
                                conv_out_ptr,
                                *conv_dim as i32,
                                *conv_kernel as i32,
                            ),
                        ).map_err(|e| format!("mamba2_conv1d[{}]: {:?}", layer_idx, e))?;
                    }

                    // After conv1d, the convolved output contains x_conv, B_conv, C_conv
                    // x_conv is the first d_inner elements of conv_out
                    let x_conv_ptr = conv_out_ptr;
                    let b_conv_ptr = conv_out_ptr + (d_inner * 4) as u64;
                    let c_conv_ptr = b_conv_ptr + (n_groups * state_size * 4) as u64;

                    // 3. Discretize: dt, A_bar, B_bar
                    // Use tail of fp32_scratch for temporaries
                    let scratch_offset = in_proj_dim;
                    let dt_out_ptr = fp32_base + (scratch_offset * 4) as u64;
                    let a_bar_ptr = dt_out_ptr + (*num_heads * 4) as u64;
                    let b_bar_ptr = a_bar_ptr + (*num_heads * 4) as u64;
                    unsafe {
                        k.mamba2_discretize.clone().launch(
                            LaunchConfig {
                                grid_dim: (1, 1, 1),
                                block_dim: ((*num_heads as u32).max(1), 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                dt_ptr,
                                *dt_bias_ptr,
                                *a_ptr,
                                b_conv_ptr,
                                dt_out_ptr,
                                a_bar_ptr,
                                b_bar_ptr,
                                *num_heads as i32,
                                *state_size as i32,
                                n_groups as i32,
                            ),
                        ).map_err(|e| format!("mamba2_discretize[{}]: {:?}", layer_idx, e))?;
                    }

                    // 4. SSM step: h = A_bar*h + B_bar*x, y = C*h
                    let y_ptr = dt_out_ptr + ((*num_heads + *num_heads + *num_heads * state_size) * 4) as u64;
                    unsafe {
                        k.mamba2_ssm_step.clone().launch(
                            LaunchConfig {
                                grid_dim: (*num_heads as u32, 1, 1),
                                block_dim: ((*head_dim as u32).min(256), 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                *ssm_state_ptr,
                                x_conv_ptr,
                                a_bar_ptr,
                                b_bar_ptr,
                                c_conv_ptr,
                                y_ptr,
                                *num_heads as i32,
                                *head_dim as i32,
                                *state_size as i32,
                                n_groups as i32,
                            ),
                        ).map_err(|e| format!("mamba2_ssm_step[{}]: {:?}", layer_idx, e))?;
                    }

                    // 5. Gate + skip + per-group RMSNorm:
                    // output = norm_weight * groupRMSNorm(silu(z) * (y + D*x_conv)) -> BF16
                    // x_conv (post-conv) is the correct x for D skip, NOT x_inner (pre-conv)
                    let group_size = d_inner / n_groups;
                    unsafe {
                        k.mamba2_gate_output.clone().launch(
                            LaunchConfig {
                                grid_dim: (n_groups as u32, 1, 1),  // one block per group
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: (256 + group_size) as u32 * 4,  // reduction + gated values
                            },
                            (
                                z_ptr,
                                y_ptr,
                                *d_ptr,
                                x_conv_ptr,  // post-conv x, NOT pre-conv x_inner
                                *norm_weight_ptr,
                                *graph.d_scratch.device_ptr(),  // BF16 output
                                d_inner as i32,
                                *head_dim as i32,
                                n_groups as i32,
                                eps,
                            ),
                        ).map_err(|e| format!("mamba2_gate_output[{}]: {:?}", layer_idx, e))?;
                    }

                    // 6. out_proj GEMV: d_inner -> hidden_size
                    let out_w = &graph.weights[*out_proj];
                    self.gemv_bf16_internal(out_w, *graph.d_scratch.device_ptr(),
                        *graph.d_hidden.device_ptr())?;
                }
            }

            // Timing: after attention
            if timing {
                self.device.synchronize().map_err(|e| format!("timing sync: {:?}", e))?;
                let attn_elapsed = t_attn_start.elapsed().as_secs_f64();
                tt_attn += attn_elapsed;
                match &layer.attn {
                    GpuAttnConfig::LinearAttention { .. } => graph.t_attn_la += attn_elapsed,
                    GpuAttnConfig::GQA { .. } => graph.t_attn_gqa += attn_elapsed,
                    GpuAttnConfig::Mamba2 { .. } => graph.t_attn_la += attn_elapsed, // reuse LA timing
                    _ => {}
                }
            }

            check_nan_bf16(&format!("L{} post_attn", layer_idx), *graph.d_hidden.device_ptr(), hs, &self.device);

            // ── Post-attention norm (fused residual add + RMSNorm) ──
            // Nemotron layers: post_attn_norm_size==0 means skip (single-sublayer blocks).
            // The mixer output in d_hidden will be added to d_residual by the NEXT layer's pre-norm.
            if layer.post_attn_norm_size > 0 {
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

            check_nan_bf16(&format!("L{} post_post_attn_norm", layer_idx), *graph.d_hidden.device_ptr(), hs, &self.device);

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
                // Check for LatentMoE: apply fc1_latent_proj before MoE dispatch
                let has_latent = graph.moe_layers.get(layer_idx)
                    .and_then(|m| m.as_ref())
                    .map(|m| m.latent_down_wid.is_some())
                    .unwrap_or(false);

                if has_latent {
                    let moe = graph.moe_layers[layer_idx].as_ref().unwrap();
                    let ld_wid = moe.latent_down_wid.unwrap();
                    // fc1_latent_proj: d_hidden -> d_scratch (latent input in BF16)
                    // d_scratch is safe here: not used by gate GEMV (which uses d_fp32_scratch)
                    // and not read again until after all w13 GEMVs complete.
                    let ld_w = &graph.weights[ld_wid];
                    self.gemv_bf16_internal(ld_w, *graph.d_hidden.device_ptr(),
                        *graph.d_scratch.device_ptr())?;
                    // Tell moe_forward to use d_scratch as expert input instead of d_hidden
                    graph.moe_input_override_ptr = *graph.d_scratch.device_ptr();
                }

                // MoE forward: gate reads d_hidden (always), experts read override ptr (if set)
                self.moe_forward_with_graph(graph, layer_idx)
                    .map_err(|e| format!("moe_forward[{}]: {}", layer_idx, e))?;

                // Reset override for next layer
                graph.moe_input_override_ptr = 0;

                if has_latent {
                    let moe = graph.moe_layers[layer_idx].as_ref().unwrap();
                    let lu_wid = moe.latent_up_wid.unwrap();
                    // fc2_latent_proj: d_moe_out(latent_size) -> d_hidden(hidden_size)
                    let lu_w = &graph.weights[lu_wid];
                    self.gemv_bf16_internal(lu_w, *graph.d_moe_out.device_ptr(),
                        *graph.d_hidden.device_ptr())?;
                }

                // Copy MoE output to hidden state (skip if latent already wrote to d_hidden)
                if !has_latent {
                    let t_d2d = Instant::now();
                    unsafe {
                        let err = cuda_sys::lib().cuMemcpyDtoD_v2(
                            *graph.d_hidden.device_ptr(),
                            *graph.d_moe_out.device_ptr(),
                            hs * 2); // BF16
                        if err != cuda_sys::CUresult::CUDA_SUCCESS {
                            return Err(format!("D2D moe_out->hidden[{}]: {:?}", layer_idx, err));
                        }
                    }
                    if timing { graph.t_moe_d2d_copy += (Instant::now() - t_d2d).as_secs_f64(); }
                }
                check_nan_bf16(&format!("L{} post_moe", layer_idx), *graph.d_hidden.device_ptr(), hs, &self.device);
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

        // ── 3. Final norm (skip if this segment doesn't produce logits) ──
        if seg_skip_final {
            // Segment ends here — hidden+residual ready for download to next GPU.
            // Just sync to ensure all layer kernels are done.
            self.device.synchronize()
                .map_err(|e| format!("sync segment end: {:?}", e))?;
            if timing {
                let tt_total = t0.elapsed().as_secs_f64();
                graph.t_attn += tt_attn;
                graph.t_route += tt_moe;
                graph.t_shared += tt_shared;
                graph.t_dense_mlp += tt_dense_mlp;
                graph.t_total += tt_total;
                graph.timing_step_count += 1;
            }
            return Ok(());
        }

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

    // ── Multi-GPU layer-split decode ───────────────────────────────────

    /// Download hidden + residual state from GPU to host buffer.
    /// Returns (hidden_bf16, residual_bf16) as byte vectors.
    pub fn download_hidden_state(&self) -> Result<(Vec<u8>, Vec<u8>), String> {
        let graph = self.graph.as_ref()
            .ok_or("graph not configured")?;
        let hs = graph.hidden_size;
        let bytes = hs * 2; // BF16

        let mut h_hidden = vec![0u8; bytes];
        let mut h_residual = vec![0u8; bytes];

        self.device.synchronize()
            .map_err(|e| format!("sync before download_hidden: {:?}", e))?;

        unsafe {
            let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                h_hidden.as_mut_ptr() as *mut std::ffi::c_void,
                *graph.d_hidden.device_ptr(),
                bytes);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("D2H hidden: {:?}", err));
            }
            let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                h_residual.as_mut_ptr() as *mut std::ffi::c_void,
                *graph.d_residual.device_ptr(),
                bytes);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("D2H residual: {:?}", err));
            }
        }

        Ok((h_hidden, h_residual))
    }

    /// Upload hidden + residual state from host buffer to GPU.
    pub fn upload_hidden_state(&self, hidden: &[u8], residual: &[u8]) -> Result<(), String> {
        let graph = self.graph.as_ref()
            .ok_or("graph not configured")?;
        let hs = graph.hidden_size;
        let bytes = hs * 2;

        if hidden.len() != bytes || residual.len() != bytes {
            return Err(format!(
                "upload_hidden_state: expected {} bytes, got hidden={} residual={}",
                bytes, hidden.len(), residual.len()));
        }

        unsafe {
            let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                *graph.d_hidden.device_ptr(),
                hidden.as_ptr() as *const std::ffi::c_void,
                bytes);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("H2D hidden: {:?}", err));
            }
            let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                *graph.d_residual.device_ptr(),
                residual.as_ptr() as *const std::ffi::c_void,
                bytes);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("H2D residual: {:?}", err));
            }
        }

        Ok(())
    }

    /// Copy KV cache data from this store (primary GPU) to an auxiliary store
    /// for GQA layers within [layer_start..layer_end).
    ///
    /// Called once per request per aux GPU after prefill, before decode begins.
    /// Only copies positions [0..prompt_len) — decode will write new positions
    /// directly on each GPU.
    pub fn copy_kv_to_aux(
        &self,
        aux_store: &mut GpuDecodeStore,
        layer_start: usize,
        layer_end: usize,
        _gqa_offset: usize,
        prompt_len: usize,
    ) -> Result<(), String> {
        let graph = self.graph.as_ref().ok_or("primary graph not configured")?;
        let aux_graph = aux_store.graph.as_ref().ok_or("aux graph not configured")?;

        if graph.kv_format != aux_graph.kv_format {
            return Err(format!(
                "KV format mismatch: primary={} aux={}",
                graph.kv_format, aux_graph.kv_format
            ));
        }

        let mut copied = 0usize;
        let mut total_bytes = 0usize;

        if graph.kv_format == 2 {
            // Polar4: 4 tensors per GQA layer (k_radius, v_radius, k_angles, v_angles)
            let num_blocks = graph.kv_num_blocks;
            if num_blocks == 0 {
                return Err("Polar4 copy but kv_num_blocks=0".to_string());
            }
            // Per position: radius = num_blocks * 2 bytes (BF16), angles = num_blocks * 8 bytes (uint8)
            let radius_bytes_per_pos = num_blocks * 2;
            let angles_bytes_per_pos = num_blocks * 8;

            for layer_idx in layer_start..layer_end {
                let kr_src = graph.kv_k_radius_ptrs[layer_idx];
                let vr_src = graph.kv_v_radius_ptrs[layer_idx];
                let ka_src = graph.kv_k_angles_ptrs[layer_idx];
                let va_src = graph.kv_v_angles_ptrs[layer_idx];
                if kr_src == 0 { continue; } // Not a GQA layer

                let kr_dst = aux_graph.kv_k_radius_ptrs[layer_idx];
                let vr_dst = aux_graph.kv_v_radius_ptrs[layer_idx];
                let ka_dst = aux_graph.kv_k_angles_ptrs[layer_idx];
                let va_dst = aux_graph.kv_v_angles_ptrs[layer_idx];
                if kr_dst == 0 {
                    return Err(format!("Aux store missing Polar4 KV cache for GQA layer {}", layer_idx));
                }

                let r_bytes = prompt_len * radius_bytes_per_pos;
                let a_bytes = prompt_len * angles_bytes_per_pos;
                if r_bytes == 0 { continue; }

                // Use a single host buffer for the largest tensor (angles)
                let max_bytes = r_bytes.max(a_bytes);
                let mut host_buf = vec![0u8; max_bytes];

                // Copy all 4 tensors: k_radius, v_radius, k_angles, v_angles
                let copies: [(u64, u64, usize, &str); 4] = [
                    (kr_src, kr_dst, r_bytes, "k_radius"),
                    (vr_src, vr_dst, r_bytes, "v_radius"),
                    (ka_src, ka_dst, a_bytes, "k_angles"),
                    (va_src, va_dst, a_bytes, "v_angles"),
                ];

                for (src, dst, nbytes, name) in &copies {
                    self.device.bind_to_thread()
                        .map_err(|e| format!("bind GPU0 for {} copy: {:?}", name, e))?;
                    if copies[0].0 == *src {
                        // Sync once before first D2H per layer
                        self.device.synchronize()
                            .map_err(|e| format!("sync GPU0 before KV copy: {:?}", e))?;
                    }
                    unsafe {
                        let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                            host_buf.as_mut_ptr() as *mut std::ffi::c_void,
                            *src, *nbytes);
                        if err != cuda_sys::CUresult::CUDA_SUCCESS {
                            return Err(format!("KV copy D2H {} layer {}: {:?}", name, layer_idx, err));
                        }
                    }
                    aux_store.device.bind_to_thread()
                        .map_err(|e| format!("bind GPU1 for {} copy: {:?}", name, e))?;
                    unsafe {
                        let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                            *dst,
                            host_buf.as_ptr() as *const std::ffi::c_void,
                            *nbytes);
                        if err != cuda_sys::CUresult::CUDA_SUCCESS {
                            return Err(format!("KV copy H2D {} layer {}: {:?}", name, layer_idx, err));
                        }
                    }
                    total_bytes += nbytes;
                }
                copied += 1;
            }
        } else {
            // FP8/BF16: 2 tensors per GQA layer (K and V)
            for layer_idx in layer_start..layer_end {
                let k_src = graph.kv_k_ptrs[layer_idx];
                let v_src = graph.kv_v_ptrs[layer_idx];
                if k_src == 0 || v_src == 0 { continue; } // Not a GQA layer

                let k_dst = aux_graph.kv_k_ptrs[layer_idx];
                let v_dst = aux_graph.kv_v_ptrs[layer_idx];
                if k_dst == 0 || v_dst == 0 {
                    return Err(format!("Aux store missing KV cache for GQA layer {}", layer_idx));
                }

                // Get KV stride from GQA config
                let kv_stride = if let GpuAttnConfig::GQA { num_kv_heads, head_dim, .. } = &graph.layers[layer_idx].attn {
                    num_kv_heads * head_dim
                } else {
                    continue; // Not GQA, skip
                };

                // FP8 = 1 byte per element, BF16 = 2 bytes per element
                let elem_size = if graph.kv_format == 0 { 2usize } else { 1usize };
                let bytes = prompt_len * kv_stride * elem_size;
                if bytes == 0 { continue; }

                let mut host_buf = vec![0u8; bytes];

                // K: D2H from GPU0, H2D to GPU1
                self.device.bind_to_thread()
                    .map_err(|e| format!("bind GPU0 for KV copy: {:?}", e))?;
                self.device.synchronize()
                    .map_err(|e| format!("sync GPU0 before KV copy: {:?}", e))?;

                unsafe {
                    let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                        host_buf.as_mut_ptr() as *mut std::ffi::c_void,
                        k_src, bytes);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(format!("KV copy D2H K layer {}: {:?}", layer_idx, err));
                    }
                }

                aux_store.device.bind_to_thread()
                    .map_err(|e| format!("bind GPU1 for KV copy: {:?}", e))?;
                unsafe {
                    let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                        k_dst,
                        host_buf.as_ptr() as *const std::ffi::c_void,
                        bytes);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(format!("KV copy H2D K layer {}: {:?}", layer_idx, err));
                    }
                }

                // V: D2H from GPU0, H2D to GPU1
                self.device.bind_to_thread()
                    .map_err(|e| format!("bind GPU0 for KV V copy: {:?}", e))?;
                unsafe {
                    let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                        host_buf.as_mut_ptr() as *mut std::ffi::c_void,
                        v_src, bytes);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(format!("KV copy D2H V layer {}: {:?}", layer_idx, err));
                    }
                }

                aux_store.device.bind_to_thread()
                    .map_err(|e| format!("bind GPU1 for KV V copy: {:?}", e))?;
                unsafe {
                    let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                        v_dst,
                        host_buf.as_ptr() as *const std::ffi::c_void,
                        bytes);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(format!("KV copy H2D V layer {}: {:?}", layer_idx, err));
                    }
                }
                copied += 1;
                total_bytes += bytes * 2; // K + V
            }
        }

        // Set KV position on aux store
        if let Some(ref mut ag) = aux_store.graph.as_mut() {
            ag.kv_current_pos = prompt_len;
        }

        log::info!("copy_kv_to_aux: copied {} GQA layers, {} bytes total ({} positions, format={})",
            copied, total_bytes, prompt_len, graph.kv_format);
        Ok(())
    }

    /// Decode a segment of layers (for multi-GPU layer split).
    ///
    /// - `do_embedding`: if true, performs embedding lookup for token_id
    /// - `layer_start..layer_end`: which layers to execute
    /// - `do_final`: if true, performs final norm + LM head + logits D2H
    /// - `gqa_cache_offset`: number of GQA layers before layer_start (for KV cache indexing)
    pub fn gpu_decode_segment(
        &mut self,
        token_id: usize,
        position: usize,
        do_embedding: bool,
        layer_start: usize,
        layer_end: usize,
        do_final: bool,
        gqa_cache_offset: usize,
    ) -> Result<(), String> {
        if !self.kernels_loaded {
            return Err("Decode kernels not loaded".to_string());
        }

        // Set segment config on graph, run, then restore defaults
        let mut graph = self.graph.take()
            .ok_or_else(|| "Call configure first".to_string())?;

        graph.segment_skip_embedding = !do_embedding;
        graph.segment_skip_final = !do_final;
        graph.segment_layer_start = layer_start;
        graph.segment_layer_end = layer_end;
        graph.segment_gqa_cache_offset = gqa_cache_offset;

        let result = self.gpu_decode_step_with_graph(&mut graph, token_id, position);

        // Restore defaults
        graph.segment_skip_embedding = false;
        graph.segment_skip_final = false;
        graph.segment_layer_start = 0;
        graph.segment_layer_end = 0;
        graph.segment_gqa_cache_offset = 0;

        self.graph = Some(graph);
        result
    }

    /// Multi-GPU streaming decode: coordinates N GpuDecodeStore instances in a pipeline.
    ///
    /// `self` owns layers [0..split_layers[0]) (GPU0).
    /// Each aux store owns a segment of layers defined by consecutive split_layers entries.
    /// The last aux store does final norm + LM head.
    /// Hidden state transfers via host memory (D2H + H2D, ~8KB per transfer).
    ///
    /// The streaming callback `on_token` is the same as gpu_generate_stream.
    pub fn gpu_generate_stream_multi<F>(
        &mut self,
        aux_store_addrs: &[usize],
        split_layers: &[usize],
        gqa_cache_offsets: &[usize],
        first_token: usize,
        start_position: usize,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        stop_ids: &[usize],
        tokenizer: &tokenizers::Tokenizer,
        presence_penalty: f32,
        logprobs_top_n: usize,
        mut on_token: F,
    ) -> usize
    where
        F: FnMut(usize, &str, Option<&str>, Option<&[(u32, f32)]>) -> bool,
    {
        use std::time::Instant;

        let num_aux = aux_store_addrs.len();
        if num_aux == 0 || split_layers.len() != num_aux || gqa_cache_offsets.len() != num_aux {
            log::error!("gpu_generate_stream_multi: invalid args (num_aux={}, splits={}, offsets={})",
                num_aux, split_layers.len(), gqa_cache_offsets.len());
            return 0;
        }

        // Bind primary GPU context
        if let Err(e) = self.device.bind_to_thread() {
            log::error!("gpu_generate_stream_multi: failed to bind primary CUDA context: {:?}", e);
            return 0;
        }

        let num_layers = match self.graph.as_ref() {
            Some(g) => g.layers.len(),
            None => { log::error!("primary store graph not configured"); return 0; }
        };

        // Layer boundaries: [0, split_layers[0], split_layers[1], ..., num_layers]
        let num_gpus = num_aux + 1;
        let mut boundaries = Vec::with_capacity(num_gpus + 1);
        boundaries.push(0usize);
        boundaries.extend_from_slice(split_layers);
        boundaries.push(num_layers);

        // Vocab size from last aux store (has LM head)
        let last_aux = unsafe { &mut *(aux_store_addrs[num_aux - 1] as *mut GpuDecodeStore) };
        let vocab_size = match last_aux.graph.as_ref() {
            Some(g) => g.vocab_size,
            None => { log::error!("last aux store graph not configured"); return 0; }
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

        // CUDA graph reuse: verify pointer stability on all GPUs before reusing.
        let mut all_reuse = self.verify_graph_pointers();
        for &addr in aux_store_addrs.iter() {
            let s = unsafe { &mut *(addr as *mut GpuDecodeStore) };
            if !s.verify_graph_pointers() {
                all_reuse = false;
            }
        }
        if all_reuse {
            if stderr_debug_enabled() {
                eprintln!("[krasis] CUDA graphs reused on all {} GPUs (ptrs verified stable)", num_gpus);
            }
        } else {
            if !self.verify_graph_pointers() {
                let was = self.graph.as_ref().map(|g| g.graphs_ever_captured).unwrap_or(false);
                if was && stderr_debug_enabled() { eprintln!("[krasis] GPU0 graph ptrs changed — invalidating"); }
                self.invalidate_cuda_graph();
            }
            for (i, &addr) in aux_store_addrs.iter().enumerate() {
                let s = unsafe { &mut *(addr as *mut GpuDecodeStore) };
                if !s.verify_graph_pointers() {
                    let was = s.graph.as_ref().map(|g| g.graphs_ever_captured).unwrap_or(false);
                    if was && stderr_debug_enabled() { eprintln!("[krasis] GPU{} graph ptrs changed — invalidating", i + 1); }
                    s.invalidate_cuda_graph();
                }
            }
        }

        let mut detok = crate::server::StreamDetokenizer::new(tokenizer);
        let mut seen_tokens: std::collections::HashSet<usize> = std::collections::HashSet::new();
        seen_tokens.insert(first_token);

        let mut next_token = first_token;
        let mut generated = 0usize;
        let decode_start = Instant::now();

        log::info!("gpu_generate_stream_multi: {} GPUs, boundaries={:?}, gqa_offsets={:?}",
            num_gpus, boundaries, gqa_cache_offsets);

        // Per-GPU timing accumulators
        let timing = self.graph.as_ref().map_or(false, |g| g.timing_enabled);
        let mut t_gpu_totals: Vec<f64> = vec![0.0; num_gpus];
        let mut t_transfer_total = 0.0f64;
        let mut t_sample_total = 0.0f64;

        // Per-GPU VRAM tracking: snapshot min free on each GPU during decode
        let mut min_vram_free: Vec<usize> = vec![usize::MAX; num_gpus];
        // Initial snapshot on GPU0
        {
            let mut free: usize = 0;
            let mut _total: usize = 0;
            unsafe { let _ = cuda_sys::lib().cuMemGetInfo_v2(&mut free, &mut _total); }
            min_vram_free[0] = free;
        }

        // Initialize CUDA graph buffers on all stores
        let no_graph = std::env::var("KRASIS_NO_GRAPH").map(|v| v != "0").unwrap_or(false);
        {
            // GPU0 graph buffers
            if let Err(e) = self.device.bind_to_thread() {
                log::error!("multi-gpu: bind GPU0 for graph init: {:?}", e);
            } else {
                let needs_init = self.graph.as_ref().map(|g| g.d_graph_token_id.is_none()).unwrap_or(false);
                if needs_init {
                    match self.init_cuda_graph_buffers() {
                        Ok(()) => {
                            if stderr_debug_enabled() { eprintln!("[krasis] GPU0 CUDA graph buffers initialized"); }
                        }
                        Err(e) => eprintln!("[krasis] GPU0 graph init failed: {}", e),
                    }
                }
            }
            // Aux GPU graph buffers
            for (i, &addr) in aux_store_addrs.iter().enumerate() {
                let s = unsafe { &mut *(addr as *mut GpuDecodeStore) };
                if let Err(e) = s.device.bind_to_thread() {
                    log::error!("multi-gpu: bind GPU{} for graph init: {:?}", i + 1, e);
                } else {
                    let needs_init = s.graph.as_ref().map(|g| g.d_graph_token_id.is_none()).unwrap_or(false);
                    if needs_init {
                        match s.init_cuda_graph_buffers() {
                            Ok(()) => {
                                if stderr_debug_enabled() { eprintln!("[krasis] GPU{} CUDA graph buffers initialized", i + 1); }
                            }
                            Err(e) => eprintln!("[krasis] GPU{} graph init failed: {}", i + 1, e),
                        }
                    }
                }
            }
        }

        for step in 0..max_tokens {
            let pos = start_position + step;

            // Check if per-layer graphs are available on ALL GPUs
            let mut use_graphs = self.graph.as_ref()
                .map(|g| g.per_layer_graphs_valid).unwrap_or(false) && !no_graph;
            if use_graphs {
                for &addr in aux_store_addrs.iter() {
                    let s = unsafe { &*(addr as *const GpuDecodeStore) };
                    if !s.graph.as_ref().map(|g| g.per_layer_graphs_valid).unwrap_or(false) {
                        use_graphs = false;
                        break;
                    }
                }
            }

            if use_graphs {
                // ── Per-layer graph replay path ──
                let mut step_ok = true;

                // GPU0: replay graphs for its layer segment
                if let Err(e) = self.device.bind_to_thread() {
                    log::error!("multi-gpu: bind GPU0 failed: {:?}", e);
                    break;
                }
                let t_start = Instant::now();
                if let Err(e) = self.replay_per_layer_graphs(next_token, pos, false) {
                    log::error!("multi-gpu: GPU0 graph replay failed (no ungraphed fallback): {}", e);
                    break;
                }
                t_gpu_totals[0] += Instant::now().duration_since(t_start).as_secs_f64();

                // Pipeline through aux GPUs: transfer + replay
                for i in 0..num_aux {
                    // Transfer hidden state from previous GPU
                    let t_xfer_start = Instant::now();
                    let (h_hidden, h_residual) = if i == 0 {
                        match self.download_hidden_state() {
                            Ok(v) => v,
                            Err(e) => { log::error!("multi-gpu download_hidden from GPU0: {}", e); step_ok = false; break; }
                        }
                    } else {
                        let prev = unsafe { &mut *(aux_store_addrs[i - 1] as *mut GpuDecodeStore) };
                        match prev.download_hidden_state() {
                            Ok(v) => v,
                            Err(e) => { log::error!("multi-gpu download_hidden from GPU{}: {}", i, e); step_ok = false; break; }
                        }
                    };
                    let aux = unsafe { &mut *(aux_store_addrs[i] as *mut GpuDecodeStore) };
                    if let Err(e) = aux.device.bind_to_thread() {
                        log::error!("multi-gpu: bind GPU{} failed: {:?}", i + 1, e);
                        step_ok = false; break;
                    }
                    if let Err(e) = aux.upload_hidden_state(&h_hidden, &h_residual) {
                        log::error!("multi-gpu upload_hidden to GPU{}: {}", i + 1, e);
                        step_ok = false; break;
                    }
                    t_transfer_total += Instant::now().duration_since(t_xfer_start).as_secs_f64();

                    // Replay graphs on this aux GPU
                    let is_last = i + 1 == num_aux;
                    let t_gpu_start = Instant::now();
                    if let Err(e) = aux.replay_per_layer_graphs(next_token, pos, is_last) {
                        log::error!("multi-gpu: GPU{} graph replay failed (no ungraphed fallback): {}", i + 1, e);
                        step_ok = false; break;
                    }
                    t_gpu_totals[i + 1] += Instant::now().duration_since(t_gpu_start).as_secs_f64();
                }
                if !step_ok { if step == 0 { continue; } else { break; } }

            } else {
                // ── Ungraphed path (step 0, or after invalidation) ──
                let mut step_ok = true;

                // GPU0: embedding + layers [0..boundaries[1])
                if let Err(e) = self.device.bind_to_thread() {
                    log::error!("multi-gpu: bind GPU0 failed: {:?}", e);
                    break;
                }
                let t_gpu0_start = if timing {
                    self.device.synchronize().ok();
                    Instant::now()
                } else { Instant::now() };

                if let Err(e) = self.gpu_decode_segment(
                    next_token, pos,
                    true,              // do_embedding
                    0,                 // layer_start
                    boundaries[1],     // layer_end
                    false,             // do_final (not last segment)
                    0,                 // gqa_cache_offset (starts at 0)
                ) {
                    log::error!("multi-gpu GPU0 segment error: {}", e);
                    break;
                }

                if timing { self.device.synchronize().ok(); }
                t_gpu_totals[0] += Instant::now().duration_since(t_gpu0_start).as_secs_f64();

                // Pipeline through aux GPUs: transfer + decode segment
                for i in 0..num_aux {
                    // Transfer hidden state from previous GPU
                    let t_xfer_start = Instant::now();
                    let (h_hidden, h_residual) = if i == 0 {
                        match self.download_hidden_state() {
                            Ok(v) => v,
                            Err(e) => { log::error!("multi-gpu download_hidden from GPU0: {}", e); step_ok = false; break; }
                        }
                    } else {
                        let prev = unsafe { &mut *(aux_store_addrs[i - 1] as *mut GpuDecodeStore) };
                        match prev.download_hidden_state() {
                            Ok(v) => v,
                            Err(e) => { log::error!("multi-gpu download_hidden from GPU{}: {}", i, e); step_ok = false; break; }
                        }
                    };

                    let aux = unsafe { &mut *(aux_store_addrs[i] as *mut GpuDecodeStore) };
                    if let Err(e) = aux.device.bind_to_thread() {
                        log::error!("multi-gpu: bind GPU{} failed: {:?}", i + 1, e);
                        step_ok = false; break;
                    }
                    if let Err(e) = aux.upload_hidden_state(&h_hidden, &h_residual) {
                        log::error!("multi-gpu upload_hidden to GPU{}: {}", i + 1, e);
                        step_ok = false; break;
                    }
                    if timing { aux.device.synchronize().ok(); }
                    t_transfer_total += Instant::now().duration_since(t_xfer_start).as_secs_f64();

                    let is_last = i + 1 == num_aux;
                    let t_gpu_start = Instant::now();
                    if let Err(e) = aux.gpu_decode_segment(
                        next_token, pos,
                        false,                 // no embedding
                        boundaries[i + 1],     // layer_start
                        boundaries[i + 2],     // layer_end
                        is_last,               // do_final only for last GPU
                        gqa_cache_offsets[i],   // GQA layers before this segment
                    ) {
                        log::error!("multi-gpu GPU{} segment error: {}", i + 1, e);
                        step_ok = false; break;
                    }
                    if timing { aux.device.synchronize().ok(); }
                    t_gpu_totals[i + 1] += Instant::now().duration_since(t_gpu_start).as_secs_f64();
                }
                if !step_ok { break; }

                // Capture per-layer graphs after step 0
                if step == 0 && !no_graph {
                    let gpu0_has_bufs = self.graph.as_ref()
                        .map(|g| g.d_graph_token_id.is_some()).unwrap_or(false);
                    let all_have_bufs = gpu0_has_bufs && aux_store_addrs.iter().all(|&addr| {
                        let s = unsafe { &*(addr as *const GpuDecodeStore) };
                        s.graph.as_ref().map(|g| g.d_graph_token_id.is_some()).unwrap_or(false)
                    });

                    if all_have_bufs {
                        // Capture GPU0 graphs: layers [0..boundaries[1]), embedding, no final
                        if let Err(e) = self.device.bind_to_thread() {
                            log::error!("multi-gpu: bind GPU0 for capture: {:?}", e);
                        } else {
                            match self.capture_per_layer_graphs(
                                Some((0, boundaries[1])), true, false, 0,
                            ) {
                                Ok(()) => {
                                    if stderr_debug_enabled() { eprintln!("[krasis] GPU0 per-layer graphs captured"); }
                                }
                                Err(e) => {
                                    log::error!("multi-gpu: GPU0 graph capture failed (no ungraphed fallback): {}", e);
                                    break;
                                }
                            }
                        }

                        // Capture graphs for each aux GPU
                        for (i, &addr) in aux_store_addrs.iter().enumerate() {
                            let s = unsafe { &mut *(addr as *mut GpuDecodeStore) };
                            let is_last = i + 1 == num_aux;
                            if let Err(e) = s.device.bind_to_thread() {
                                log::error!("multi-gpu: bind GPU{} for capture: {:?}", i + 1, e);
                            } else {
                                match s.capture_per_layer_graphs(
                                    Some((boundaries[i + 1], boundaries[i + 2])), false, is_last, gqa_cache_offsets[i],
                                ) {
                                    Ok(()) => {
                                        if stderr_debug_enabled() { eprintln!("[krasis] GPU{} per-layer graphs captured", i + 1); }
                                    }
                                    Err(e) => {
                                        log::error!("multi-gpu: GPU{} graph capture failed (no ungraphed fallback): {}", i + 1, e);
                                        step_ok = false; break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let t_sample_start = Instant::now();
            // Logits are now in last aux store's h_logits
            let last_aux = unsafe { &mut *(aux_store_addrs[num_aux - 1] as *mut GpuDecodeStore) };
            let logits = &mut last_aux.graph.as_mut().unwrap().h_logits;

            if presence_penalty != 0.0 {
                for &tok in &seen_tokens {
                    if tok < vocab_size { logits[tok] -= presence_penalty; }
                }
            }
            self.apply_suppress_at_step(logits, vocab_size, generated);

            // Force-inject </think> when thinking budget is exhausted
            let think_end_active = self.think_end_token_for_suppress.is_some() && !self.think_end_seen;
            let think_within_budget = self.think_suppress_budget == 0
                || self.think_suppress_count < self.think_suppress_budget;
            let target_pred = if think_end_active && !think_within_budget {
                if let Some(te) = self.think_end_token_for_suppress {
                    eprintln!("[krasis] Thinking budget exhausted ({} tokens) — force-injecting </think>",
                        self.think_suppress_count);
                    te
                } else {
                    crate::decode::sample_from_logits_pub(
                        logits, vocab_size, temperature, top_k, top_p, &mut rng_next)
                }
            } else {
                crate::decode::sample_from_logits_pub(
                    logits, vocab_size, temperature, top_k, top_p, &mut rng_next)
            };
            self.notify_token_generated(target_pred);
            seen_tokens.insert(target_pred);
            generated += 1;
            next_token = target_pred;

            let mut text = detok.add(target_pred as u32);
            let finish_reason = if stop_set.contains(&target_pred) { Some("stop") }
                else if generated >= max_tokens { Some("length") }
                else { None };
            if finish_reason.is_some() { text.push_str(&detok.flush()); }
            t_sample_total += Instant::now().duration_since(t_sample_start).as_secs_f64();

            let finished = finish_reason.is_some();
            let token_lp = if logprobs_top_n > 0 {
                let last_aux = unsafe { &*(aux_store_addrs[num_aux - 1] as *const GpuDecodeStore) };
                let logits_ref = &last_aux.graph.as_ref().unwrap().h_logits;
                Some(crate::decode::extract_top_logprobs(logits_ref, vocab_size, logprobs_top_n))
            } else { None };
            let cont = on_token(target_pred, &text, finish_reason, token_lp.as_deref());
            if finished || !cont { break; }
        }

        let elapsed = decode_start.elapsed().as_secs_f64();
        if generated > 0 {
            let tps = generated as f64 / elapsed;

            // Post-decode VRAM snapshot on all GPUs
            // GPU0 (already bound from last step)
            if let Err(_) = self.device.bind_to_thread() {} else {
                let mut free: usize = 0;
                let mut total: usize = 0;
                unsafe { let _ = cuda_sys::lib().cuMemGetInfo_v2(&mut free, &mut total); }
                if free < min_vram_free[0] { min_vram_free[0] = free; }
            }
            // Aux GPUs
            for i in 0..num_aux {
                let aux = unsafe { &mut *(aux_store_addrs[i] as *mut GpuDecodeStore) };
                if let Err(_) = aux.device.bind_to_thread() {} else {
                    let mut free: usize = 0;
                    let mut total: usize = 0;
                    unsafe { let _ = cuda_sys::lib().cuMemGetInfo_v2(&mut free, &mut total); }
                    if free < min_vram_free[i + 1] { min_vram_free[i + 1] = free; }
                }
            }
            // Re-bind GPU0 as the primary context
            let _ = self.device.bind_to_thread();

            // Update last_min_free on each store
            let gpu0_min_mb = min_vram_free[0] / (1024 * 1024);
            self.last_min_free_vram_mb = gpu0_min_mb;
            for i in 0..num_aux {
                let aux = unsafe { &mut *(aux_store_addrs[i] as *mut GpuDecodeStore) };
                aux.last_min_free_vram_mb = min_vram_free[i + 1] / (1024 * 1024);
            }

            // Print per-GPU VRAM summary
            let mut vram_parts: Vec<String> = Vec::with_capacity(num_gpus);
            for g in 0..num_gpus {
                let min_mb = min_vram_free[g] / (1024 * 1024);
                vram_parts.push(format!("GPU{} {}MB min", g, min_mb));
            }
            eprintln!("  \x1b[32mdecode: {} tokens in {:.2}s ({:.1} tok/s)  VRAM: {}\x1b[0m",
                generated, elapsed, tps, vram_parts.join(", "));

            log::info!("gpu_generate_stream_multi: {} tokens in {:.2}s ({:.1} tok/s)",
                generated, elapsed, tps);
            for g in 0..num_gpus {
                log::info!("  GPU{} VRAM: min_free={} MB during decode", g, min_vram_free[g] / (1024 * 1024));
            }
        }
        self.last_decode_elapsed = elapsed;

        // Print multi-GPU timing summary
        if timing && generated > 0 {
            let n = generated as f64;
            let avg_xfer = t_transfer_total / n * 1000.0;
            let avg_sample = t_sample_total / n * 1000.0;
            let avg_total = elapsed / n * 1000.0;
            let avg_gpu_sum: f64 = t_gpu_totals.iter().map(|t| t / n * 1000.0).sum();
            let avg_other = avg_total - avg_gpu_sum - avg_xfer - avg_sample;

            eprintln!();
            eprintln!("  \x1b[33m┌───────────────────────────────────────────────────────┐\x1b[0m");
            eprintln!("  \x1b[33m│\x1b[0m  MULTI-GPU DECODE TIMING ({} tokens, {} GPUs)          \x1b[33m│\x1b[0m", generated, num_gpus);
            eprintln!("  \x1b[33m├───────────────────────────────────────────────────────┤\x1b[0m");
            eprintln!("  \x1b[33m│\x1b[0m  Total:        {:7.2} ms/tok  ({:5.1} tok/s)         \x1b[33m│\x1b[0m", avg_total, 1000.0 / avg_total);

            // Per-GPU timing
            for gpu_idx in 0..num_gpus {
                let avg_gpu = t_gpu_totals[gpu_idx] / n * 1000.0;
                let layer_start = boundaries[gpu_idx];
                let layer_end = boundaries[gpu_idx + 1];
                let n_layers = layer_end - layer_start;
                eprintln!("  \x1b[33m│\x1b[0m  GPU{} (L{}-{}): {:7.2} ms  ({:4.1}%)  [{} layers]    \x1b[33m│\x1b[0m",
                    gpu_idx, layer_start, layer_end - 1, avg_gpu, avg_gpu / avg_total * 100.0, n_layers);
            }
            eprintln!("  \x1b[33m│\x1b[0m  Transfer:     {:7.2} ms  ({:4.1}%)  [{} hops]        \x1b[33m│\x1b[0m",
                avg_xfer, avg_xfer / avg_total * 100.0, num_aux);
            eprintln!("  \x1b[33m│\x1b[0m  Sample:       {:7.2} ms  ({:4.1}%)                   \x1b[33m│\x1b[0m",
                avg_sample, avg_sample / avg_total * 100.0);
            if avg_other.abs() > 0.01 {
                eprintln!("  \x1b[33m│\x1b[0m  Other:        {:7.2} ms  ({:4.1}%)                   \x1b[33m│\x1b[0m",
                    avg_other, avg_other / avg_total * 100.0);
            }
            eprintln!("  \x1b[33m├───────────────────────────────────────────────────────┤\x1b[0m");

            // Per-layer rate comparison
            for gpu_idx in 0..num_gpus {
                let avg_gpu = t_gpu_totals[gpu_idx] / n * 1000.0;
                let n_layers = boundaries[gpu_idx + 1] - boundaries[gpu_idx];
                let ms_per_layer = if n_layers > 0 { avg_gpu / n_layers as f64 } else { 0.0 };
                eprintln!("  \x1b[33m│\x1b[0m  GPU{} per-layer: {:5.3} ms                           \x1b[33m│\x1b[0m", gpu_idx, ms_per_layer);
            }

            // HCS stats for all GPUs
            eprintln!("  \x1b[33m├───────────────────────────────────────────────────────┤\x1b[0m");
            if let Some(ref graph0) = self.graph {
                if let Some(ref hcs) = graph0.hcs {
                    let total = (hcs.total_hits + hcs.total_misses).max(1) as f64;
                    let hit_pct = hcs.total_hits as f64 / total * 100.0;
                    eprintln!("  \x1b[33m│\x1b[0m  GPU0 HCS: {} cached, {:.1}% hit ({}/{})           \x1b[33m│\x1b[0m",
                        hcs.num_cached, hit_pct, hcs.total_hits, hcs.total_misses);
                }
            }
            for (i, &addr) in aux_store_addrs.iter().enumerate() {
                let s = unsafe { &*(addr as *const GpuDecodeStore) };
                if let Some(ref graph_i) = s.graph {
                    if let Some(ref hcs) = graph_i.hcs {
                        let total = (hcs.total_hits + hcs.total_misses).max(1) as f64;
                        let hit_pct = hcs.total_hits as f64 / total * 100.0;
                        eprintln!("  \x1b[33m│\x1b[0m  GPU{} HCS: {} cached, {:.1}% hit ({}/{})          \x1b[33m│\x1b[0m",
                            i + 1, hcs.num_cached, hit_pct, hcs.total_hits, hcs.total_misses);
                    }
                }
            }
            eprintln!("  \x1b[33m└───────────────────────────────────────────────────────┘\x1b[0m");

            // Per-GPU detailed timing
            let all_stores: Vec<(usize, &GpuDecodeStore)> = std::iter::once((0usize, &*self as &GpuDecodeStore))
                .chain(aux_store_addrs.iter().enumerate().map(|(i, &addr)| {
                    (i + 1, unsafe { &*(addr as *const GpuDecodeStore) })
                }))
                .collect();

            for (gpu_idx, store_ref) in &all_stores {
                let layer_start = boundaries[*gpu_idx];
                let layer_end = boundaries[*gpu_idx + 1];
                eprintln!();
                eprintln!("  \x1b[36m=== GPU{} Detail (layers {}-{}) ===\x1b[0m", gpu_idx, layer_start, layer_end - 1);
                if let Some(ref g) = store_ref.graph {
                    if g.timing_step_count > 0 {
                        let ns = g.timing_step_count as f64;
                        eprintln!("    Attention:    {:6.2} ms (LA {:5.2}, GQA {:5.2})",
                            g.t_attn / ns * 1000.0, g.t_attn_la / ns * 1000.0, g.t_attn_gqa / ns * 1000.0);
                        eprintln!("    MoE:          {:6.2} ms (route {:5.2}, experts {:5.2}, shared {:5.2})",
                            g.t_route / ns * 1000.0, g.t_moe_route_sync / ns * 1000.0,
                            g.t_moe_expert_loop / ns * 1000.0, g.t_moe_shared / ns * 1000.0);
                        eprintln!("      w13: {:5.2} ms, silu+w2: {:5.2} ms",
                            g.t_expert_w13 / ns * 1000.0, g.t_expert_silu_w2 / ns * 1000.0);
                        eprintln!("    Norms+Emb:    {:6.2} ms", g.t_norm / ns * 1000.0);
                        if *gpu_idx == num_gpus - 1 {
                            eprintln!("    LM Head:      {:6.2} ms", g.t_lm_head / ns * 1000.0);
                        }
                        eprintln!("    Cold/HCS:     {:.1}/{:.1} experts/tok",
                            g.dma_cold_experts as f64 / ns, g.dma_hcs_experts as f64 / ns);
                    }
                }
            }
            eprintln!();

            // Structured log
            let mut log_parts: Vec<String> = Vec::new();
            log_parts.push(format!("MULTI-GPU DECODE TIMING ({} tokens, {} GPUs): Total: {:.2} ms/tok ({:.1} tok/s)",
                generated, num_gpus, avg_total, 1000.0 / avg_total));
            for gpu_idx in 0..num_gpus {
                let avg_gpu = t_gpu_totals[gpu_idx] / n * 1000.0;
                log_parts.push(format!("GPU{} (L{}-{}): {:.2} ms ({:.1}%)",
                    gpu_idx, boundaries[gpu_idx], boundaries[gpu_idx + 1] - 1,
                    avg_gpu, avg_gpu / avg_total * 100.0));
            }
            log_parts.push(format!("Transfer: {:.2} ms ({:.1}%)", avg_xfer, avg_xfer / avg_total * 100.0));
            log_parts.push(format!("Sample: {:.2} ms ({:.1}%)", avg_sample, avg_sample / avg_total * 100.0));
            log::info!("{}", log_parts.join(" | "));
        }

        generated
    }

    /// Batched GPU decode step: process multiple tokens through all layers.
    /// Used for speculative decode verification: tokens[0] is the real token,
    /// tokens[1..] are draft tokens verified alongside.
    ///
    /// At MoE layers, routes all tokens, takes the expert union, and DMAs each
    /// expert once — the key optimization over sequential verification.
    ///
    /// Returns the number of valid logit positions in h_batch_logits.
    /// May be less than tokens.len() if fail-fast expert divergence detected.
    /// Position 0 always has valid logits (the real token).
    pub fn gpu_decode_step_batched(
        &mut self,
        tokens: &[usize],
        positions: &[usize],
    ) -> Result<usize, String> {
        let batch_size = tokens.len();
        if batch_size == 0 { return Ok(0); }
        if batch_size == 1 {
            self.gpu_decode_step(tokens[0], positions[0])?;
            return Ok(1);
        }

        let mut graph = self.graph.take()
            .ok_or_else(|| "Call configure first".to_string())?;

        let result = self.gpu_decode_step_batched_inner(&mut graph, tokens, positions);

        self.graph = Some(graph);
        result
    }

    fn gpu_decode_step_batched_inner(
        &mut self,
        graph: &mut GpuDecodeGraph,
        tokens: &[usize],
        positions: &[usize],
    ) -> Result<usize, String> {
        use cudarc::driver::LaunchConfig;

        let mut batch_size = tokens.len();
        let orig_batch_size = batch_size;
        let hs = graph.hidden_size;
        let eps = graph.eps;
        let do_timing = std::env::var("KRASIS_SPEC_DEBUG").is_ok();
        let mut tt_norm: f64 = 0.0;
        let mut tt_proj: f64 = 0.0;
        let mut tt_attn: f64 = 0.0;
        let mut tt_moe: f64 = 0.0;
        let mut tt_lmhead: f64 = 0.0;

        if batch_size > graph.batch_max {
            return Err(format!("batch_size {} > batch_max {}", batch_size, graph.batch_max));
        }

        let d_bh_ptr = *graph.d_batch_hidden.as_ref()
            .ok_or("batch buffers not allocated")?.device_ptr();
        let d_br_ptr = *graph.d_batch_residual.as_ref()
            .ok_or("batch buffers not allocated")?.device_ptr();
        let d_bmo_ptr = *graph.d_batch_moe_out.as_ref()
            .ok_or("batch buffers not allocated")?.device_ptr();
        let d_bpa_ptr = *graph.d_batch_proj_a.as_ref()
            .ok_or("batch proj buffers not allocated")?.device_ptr();
        let d_bpb_ptr = *graph.d_batch_proj_b.as_ref()
            .ok_or("batch proj buffers not allocated")?.device_ptr();
        let d_bao_ptr = *graph.d_batch_attn_out.as_ref()
            .ok_or("batch attn_out buffers not allocated")?.device_ptr();

        let k = graph.kernels.as_ref()
            .ok_or_else(|| "Kernels not cached".to_string())?
            .clone();

        // ── 1. Embedding lookup for all tokens → d_batch_hidden ──
        for t in 0..batch_size {
            let out_ptr = d_bh_ptr + (t * hs * 2) as u64;
            let threads = 256u32;
            let blocks = ((hs as u32) + threads - 1) / threads;
            unsafe {
                k.embedding_lookup.clone().launch(
                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                    (out_ptr, graph.embedding_ptr, tokens[t] as i32, hs as i32),
                ).map_err(|e| format!("batch embedding[{}]: {:?}", t, e))?;
            }
        }

        let num_layers = graph.layers.len();
        let mut la_stack_idx = 0usize;

        // ── 2. Layer loop — batched GEMM for projections, per-token for attention ──
        for layer_idx in 0..num_layers {
            let is_la = matches!(&graph.layers[layer_idx].attn, GpuAttnConfig::LinearAttention { .. });
            let has_moe = layer_idx < graph.moe_layers.len()
                && graph.moe_layers[layer_idx].is_some();
            let first_residual = layer_idx == 0;

            let t_norm_start = std::time::Instant::now();

            // ── A. Pre-attention norm (in-place on batch arrays, no D2D swap) ──
            {
                let smem = (hs as u32) * 4;
                let threads = 256u32.min(hs as u32);
                for t in 0..batch_size {
                    let h_ptr = d_bh_ptr + (t * hs * 2) as u64;
                    let r_ptr = d_br_ptr + (t * hs * 2) as u64;
                    unsafe {
                        k.fused_add_rmsnorm.clone().launch(
                            LaunchConfig { grid_dim: (1, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: smem },
                            (h_ptr, r_ptr, graph.layers[layer_idx].input_norm_ptr, eps, hs as i32,
                             if first_residual { 1i32 } else { 0i32 }),
                        ).map_err(|e| format!("batch norm[{}][{}]: {:?}", layer_idx, t, e))?;
                    }
                }
            }

            // ── B. Save LA hidden stack (after norm, before projection) ──
            if is_la {
                if let Some(ref d_stack) = graph.d_la_hidden_stack {
                    for t in 0..batch_size {
                        let stack_offset = (la_stack_idx * graph.batch_max + t) * hs;
                        unsafe {
                            cuda_sys::lib().cuMemcpyDtoD_v2(
                                (*d_stack.device_ptr() as *const u16).add(stack_offset) as u64,
                                d_bh_ptr + (t * hs * 2) as u64,
                                hs * 2);
                        }
                    }
                }
            }

            if do_timing {
                self.device.synchronize().map_err(|e| format!("timing: {:?}", e))?;
                tt_norm += t_norm_start.elapsed().as_secs_f64();
            }
            let t_attn_start = std::time::Instant::now();

            // ── C. Batch input projection GEMM + D. per-token attention + E. batch output GEMM ──
            match &graph.layers[layer_idx].attn {
                GpuAttnConfig::LinearAttention {
                    in_proj_qkvz, in_proj_ba, out_proj,
                    conv_weight_ptr, a_log_ptr, dt_bias_ptr, norm_weight_ptr,
                    nk, nv, dk, dv, hr, kernel_dim, conv_dim, scale,
                    conv_state_ptr, recur_state_ptr,
                } => {
                    let nk_ = *nk; let nv_ = *nv; let dk_ = *dk; let dv_ = *dv;
                    let hr_ = *hr; let cd = *conv_dim; let kd = *kernel_dim;
                    let key_dim = nk_ * dk_;
                    let gated_size = nv_ * dv_;

                    // C1. Batch GEMM: qkvz_w × batch_hidden → batch_proj_a (weights loaded ONCE)
                    let qkvz_w = &graph.weights[*in_proj_qkvz];
                    let qkvz_dim = qkvz_w.rows;
                    self.gemm_bf16_to_f32_batch(
                        qkvz_w, d_bh_ptr, d_bpa_ptr,
                        batch_size, hs, qkvz_dim)?;

                    // C2. Batch GEMM: ba_w × batch_hidden → batch_proj_b (weights loaded ONCE)
                    let ba_w = &graph.weights[*in_proj_ba];
                    let ba_dim = ba_w.rows;
                    self.gemm_bf16_to_f32_batch(
                        ba_w, d_bh_ptr, d_bpb_ptr,
                        batch_size, hs, ba_dim)?;

                    // D. Per-token LA processing (reads from batch_proj, tiny compute)
                    for t in 0..batch_size {
                        // Copy this token's projection outputs to single-token scratch
                        unsafe {
                            cuda_sys::lib().cuMemcpyDtoD_v2(
                                *graph.d_la_qkvz.device_ptr(),
                                d_bpa_ptr + (t * qkvz_dim) as u64 * 4,
                                qkvz_dim * 4);
                            cuda_sys::lib().cuMemcpyDtoD_v2(
                                *graph.d_la_ba.device_ptr(),
                                d_bpb_ptr + (t * ba_dim) as u64 * 4,
                                ba_dim * 4);
                        }

                        // Uninterleave
                        {
                            let group_dim = 2 * dk_ + 2 * hr_ * dv_;
                            let total = nk_ * group_dim;
                            let threads = 256u32;
                            let blocks = ((total as u32) + threads - 1) / threads;
                            unsafe {
                                let f = self.device.get_func(MODULE_NAME, "uninterleave_qkvz")
                                    .ok_or_else(|| "uninterleave_qkvz not found".to_string())?;
                                f.launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (*graph.d_la_conv_out.device_ptr(), *graph.d_la_recur_out.device_ptr(),
                                     *graph.d_la_qkvz.device_ptr(), nk_ as i32, dk_ as i32, hr_ as i32, dv_ as i32),
                                ).map_err(|e| format!("batch uninterleave[{}][{}]: {:?}", layer_idx, t, e))?;
                            }
                        }

                        // Save z
                        {
                            let z_size = nv_ * dv_;
                            unsafe {
                                cuda_sys::lib().cuMemcpyDtoD_v2(
                                    *graph.d_la_gated_out.device_ptr(),
                                    *graph.d_la_recur_out.device_ptr(),
                                    z_size * 4);
                            }
                        }

                        // Conv1d
                        {
                            let threads = 256u32;
                            let blocks = ((cd as u32) + threads - 1) / threads;
                            unsafe {
                                let f = self.device.get_func(MODULE_NAME, "la_conv1d")
                                    .ok_or_else(|| "la_conv1d kernel not found".to_string())?;
                                f.launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (*conv_state_ptr, *graph.d_la_conv_out.device_ptr(),
                                     *graph.d_la_qkvz.device_ptr(), *conv_weight_ptr, cd as i32, kd as i32),
                                ).map_err(|e| format!("batch la_conv1d[{}][{}]: {:?}", layer_idx, t, e))?;
                            }
                        }

                        // Gate/beta
                        let gate_ptr_local = *graph.d_la_conv_out.device_ptr();
                        let beta_ptr_local = unsafe { (*graph.d_la_conv_out.device_ptr() as *const f32).add(nv_) as u64 };
                        {
                            let threads = 256u32;
                            let blocks = ((nv_ as u32) + threads - 1) / threads;
                            unsafe {
                                let f = self.device.get_func(MODULE_NAME, "compute_gate_beta")
                                    .ok_or_else(|| "compute_gate_beta not found".to_string())?;
                                f.launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (gate_ptr_local, beta_ptr_local, *graph.d_la_ba.device_ptr(),
                                     *a_log_ptr, *dt_bias_ptr, nv_ as i32, hr_ as i32),
                                ).map_err(|e| format!("batch gate_beta[{}][{}]: {:?}", layer_idx, t, e))?;
                            }
                        }

                        // Head repeat-interleave
                        let q_ptr_for_recur: u64;
                        let k_ptr_for_recur: u64;
                        if hr_ > 1 {
                            let total_q = (nv_ * dk_) as u32;
                            let threads = 256u32;
                            let blocks = (total_q + threads - 1) / threads;
                            unsafe {
                                let ri_fn = self.device.get_func(MODULE_NAME, "repeat_interleave_heads")
                                    .ok_or_else(|| "repeat_interleave_heads not found".to_string())?;
                                ri_fn.clone().launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (*graph.d_la_recur_out.device_ptr(), *graph.d_la_qkvz.device_ptr(),
                                     nk_ as i32, dk_ as i32, hr_ as i32),
                                ).map_err(|e| format!("batch ri_q[{}][{}]: {:?}", layer_idx, t, e))?;
                                let k_in = (*graph.d_la_qkvz.device_ptr() as *const f32).add(key_dim) as u64;
                                let k_out = (*graph.d_la_recur_out.device_ptr() as *const f32).add(nv_ * dk_) as u64;
                                ri_fn.launch(
                                    LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (k_out, k_in, nk_ as i32, dk_ as i32, hr_ as i32),
                                ).map_err(|e| format!("batch ri_k[{}][{}]: {:?}", layer_idx, t, e))?;
                            }
                            q_ptr_for_recur = *graph.d_la_recur_out.device_ptr();
                            k_ptr_for_recur = unsafe { (*graph.d_la_recur_out.device_ptr() as *const f32).add(nv_ * dk_) as u64 };
                        } else {
                            q_ptr_for_recur = *graph.d_la_qkvz.device_ptr();
                            k_ptr_for_recur = unsafe { (*graph.d_la_qkvz.device_ptr() as *const f32).add(key_dim) as u64 };
                        }

                        // L2 norm
                        {
                            let threads = 256u32;
                            let l2_fn = self.device.get_func(MODULE_NAME, "l2norm_scale_per_head")
                                .ok_or_else(|| "l2norm_scale_per_head not found".to_string())?;
                            unsafe {
                                l2_fn.clone().launch(
                                    LaunchConfig { grid_dim: (nv_ as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (q_ptr_for_recur, *scale, nv_ as i32, dk_ as i32),
                                ).map_err(|e| format!("batch l2_q[{}][{}]: {:?}", layer_idx, t, e))?;
                                l2_fn.launch(
                                    LaunchConfig { grid_dim: (nv_ as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (k_ptr_for_recur, 1.0f32, nv_ as i32, dk_ as i32),
                                ).map_err(|e| format!("batch l2_k[{}][{}]: {:?}", layer_idx, t, e))?;
                            }
                        }

                        // Recurrence
                        let v_ptr = unsafe { (*graph.d_la_qkvz.device_ptr() as *const f32).add(2 * key_dim) as u64 };
                        {
                            let threads = 256u32;
                            let delta_fn = self.device.get_func(MODULE_NAME, "gated_delta_net_step")
                                .ok_or_else(|| "gated_delta_net_step not found".to_string())?;
                            unsafe {
                                delta_fn.launch(
                                    LaunchConfig { grid_dim: (nv_ as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                    (*recur_state_ptr, q_ptr_for_recur, k_ptr_for_recur, v_ptr,
                                     gate_ptr_local, beta_ptr_local, *graph.d_la_ba.device_ptr(),
                                     nv_ as i32, dk_ as i32, dv_ as i32),
                                ).map_err(|e| format!("batch recur[{}][{}]: {:?}", layer_idx, t, e))?;
                            }
                        }

                        // Gated RMSNorm + SiLU → BF16 directly → batch_attn_out[t]
                        {
                            let threads = 256u32;
                            let smem = (dv_ as u32 + 32) * 4;
                            let out_ptr = d_bao_ptr + (t * gated_size * 2) as u64;
                            let f = self.device.get_func(MODULE_NAME, "gated_rmsnorm_silu_bf16")
                                .ok_or_else(|| "gated_rmsnorm_silu_bf16 not found".to_string())?;
                            unsafe {
                                f.launch(
                                    LaunchConfig { grid_dim: (nv_ as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: smem },
                                    (out_ptr, *graph.d_la_ba.device_ptr(),
                                     *graph.d_la_gated_out.device_ptr(), *norm_weight_ptr, eps,
                                     nv_ as i32, dv_ as i32),
                                ).map_err(|e| format!("batch gated_norm_bf16[{}][{}]: {:?}", layer_idx, t, e))?;
                            }
                        }
                    } // end per-token LA loop

                    // E. Batch output projection GEMM (weights loaded ONCE)
                    let out_w = &graph.weights[*out_proj];
                    self.gemm_bf16_batch(
                        out_w, d_bao_ptr, d_bh_ptr,
                        batch_size, gated_size, hs)?;
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
                    let o_size = nh * hd;
                    let is_gated = *gated;
                    let qnp = *q_norm_ptr;
                    let knp = *k_norm_ptr;
                    let sm_sc = *sm_scale;

                    // C. Batch input projection GEMM (weights loaded ONCE)
                    if let Some(fid) = fused_qkv {
                        let fw = &graph.weights[*fid];
                        let fqkv_dim = fw.rows;
                        self.gemm_bf16_to_f32_batch(
                            fw, d_bh_ptr, d_bpa_ptr,
                            batch_size, hs, fqkv_dim)?;

                        let q_size = if is_gated { nh * hd * 2 } else { nh * hd };
                        let k_offset = q_size;
                        let v_offset = k_offset + kv_stride;

                        // D. Per-token GQA processing
                        for t in 0..batch_size {
                            let position = positions[t];
                            let proj_ptr = d_bpa_ptr + (t * fqkv_dim) as u64 * 4;

                            // Copy fused QKV output to single-token scratch, extract K/V
                            unsafe {
                                cuda_sys::lib().cuMemcpyDtoD_v2(
                                    *graph.d_gqa_q.device_ptr(), proj_ptr, q_size * 4);
                                cuda_sys::lib().cuMemcpyDtoD_v2(
                                    *graph.d_gqa_k.device_ptr(),
                                    proj_ptr + (k_offset * 4) as u64,
                                    kv_stride * 4);
                                cuda_sys::lib().cuMemcpyDtoD_v2(
                                    *graph.d_gqa_v.device_ptr(),
                                    proj_ptr + (v_offset * 4) as u64,
                                    kv_stride * 4);
                            }

                            // Split gated Q
                            if is_gated {
                                let total = (nh * hd) as u32;
                                let threads = 256u32;
                                let blocks = (total + threads - 1) / threads;
                                unsafe {
                                    let split_fn = self.device.get_func(MODULE_NAME, "split_gated_q")
                                        .ok_or_else(|| "split_gated_q not found".to_string())?;
                                    split_fn.launch(
                                        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                        (*graph.d_gqa_q.device_ptr(), *graph.d_la_qkvz.device_ptr(),
                                         *graph.d_gqa_q.device_ptr(), nh as i32, hd as i32),
                                    ).map_err(|e| format!("batch split_gated_q[{}][{}]: {:?}", layer_idx, t, e))?;
                                }
                            }

                            // QK norm
                            if qnp != 0 {
                                let threads = 256u32;
                                let norm_fn = self.device.get_func(MODULE_NAME, "per_head_rmsnorm")
                                    .ok_or_else(|| "per_head_rmsnorm not found".to_string())?;
                                unsafe {
                                    norm_fn.clone().launch(
                                        LaunchConfig { grid_dim: (nh as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                        (*graph.d_gqa_q.device_ptr(), qnp, eps, nh as i32, hd as i32, 0i32),
                                    ).map_err(|e| format!("batch qnorm[{}][{}]: {:?}", layer_idx, t, e))?;
                                    norm_fn.launch(
                                        LaunchConfig { grid_dim: (nkv as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                        (*graph.d_gqa_k.device_ptr(), knp, eps, nkv as i32, hd as i32, 0i32),
                                    ).map_err(|e| format!("batch knorm[{}][{}]: {:?}", layer_idx, t, e))?;
                                }
                            }

                            // RoPE
                            if let Some(ref d_cos) = graph.d_rope_cos {
                                let half_dim = graph.rope_half_dim;
                                let total_heads = nh + nkv;
                                let total_work = total_heads * half_dim;
                                let threads = 256u32;
                                let blocks = ((total_work as u32) + threads - 1) / threads;
                                let rope_fn = self.device.get_func(MODULE_NAME, "apply_rope")
                                    .ok_or_else(|| "apply_rope not found".to_string())?;
                                unsafe {
                                    rope_fn.launch(
                                        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                        (*graph.d_gqa_q.device_ptr(), *graph.d_gqa_k.device_ptr(),
                                         *d_cos.device_ptr(), *graph.d_rope_sin.as_ref().unwrap().device_ptr(),
                                         position as i32, nh as i32, nkv as i32, hd as i32, half_dim as i32),
                                    ).map_err(|e| format!("batch rope[{}][{}]: {:?}", layer_idx, t, e))?;
                                }
                            }

                            // KV cache write
                            if layer_idx < graph.kv_k_ptrs.len()
                                && graph.kv_k_ptrs[layer_idx] != 0 {
                                let threads = 256u32;
                                let blocks = ((kv_stride as u32) + threads - 1) / threads;
                                let kv_fn = self.device.get_func(MODULE_NAME, "kv_cache_write")
                                    .ok_or_else(|| "kv_cache_write not found".to_string())?;
                                unsafe {
                                    kv_fn.launch(
                                        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                        (graph.kv_k_ptrs[layer_idx], graph.kv_v_ptrs[layer_idx],
                                         *graph.d_gqa_k.device_ptr(), *graph.d_gqa_v.device_ptr(),
                                         position as i32, kv_stride as i32),
                                    ).map_err(|e| format!("batch kv_write[{}][{}]: {:?}", layer_idx, t, e))?;
                                }
                            }

                            // GQA attention
                            {
                                let seq_len = (position + 1) as u32;
                                let threads = 256u32;
                                let q_smem = (hd as u32) * 4;
                                let use_tiled = graph.d_gqa_tiled_o.is_some()
                                    && seq_len > (graph.gqa_tile_size * graph.gqa_num_q_heads) as u32;
                                if use_tiled {
                                    let tile_size = graph.gqa_tile_size;
                                    let num_tiles = ((seq_len as usize) + tile_size - 1) / tile_size;
                                    let tiled_fn = self.device.get_func(MODULE_NAME, "gqa_attention_tiled")
                                        .ok_or_else(|| "gqa_attention_tiled not found".to_string())?;
                                    let smem = q_smem + (tile_size as u32) * 4 + 128;
                                    unsafe {
                                        tiled_fn.launch(
                                            LaunchConfig {
                                                grid_dim: (nh as u32, num_tiles as u32, 1),
                                                block_dim: (threads, 1, 1),
                                                shared_mem_bytes: smem,
                                            },
                                            (*graph.d_gqa_tiled_o.as_ref().unwrap().device_ptr(),
                                             *graph.d_gqa_tiled_lse.as_ref().unwrap().device_ptr(),
                                             *graph.d_gqa_q.device_ptr(),
                                             graph.kv_k_ptrs[layer_idx],
                                             graph.kv_v_ptrs[layer_idx],
                                             sm_sc, nh as i32, nkv as i32, hd as i32,
                                             seq_len as i32, graph.kv_max_seq as i32,
                                             tile_size as i32),
                                        ).map_err(|e| format!("batch gqa_tiled[{}][{}]: {:?}", layer_idx, t, e))?;
                                    }
                                    let reduce_fn = self.device.get_func(MODULE_NAME, "gqa_attention_reduce")
                                        .ok_or_else(|| "gqa_attention_reduce not found".to_string())?;
                                    unsafe {
                                        reduce_fn.launch(
                                            LaunchConfig { grid_dim: (nh as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                            (*graph.d_gqa_out.device_ptr(),
                                             *graph.d_gqa_tiled_o.as_ref().unwrap().device_ptr(),
                                             *graph.d_gqa_tiled_lse.as_ref().unwrap().device_ptr(),
                                             nh as i32, hd as i32, num_tiles as i32),
                                        ).map_err(|e| format!("batch gqa_reduce[{}][{}]: {:?}", layer_idx, t, e))?;
                                    }
                                } else {
                                    let shared_mem_bytes = q_smem + seq_len * 4 + 128;
                                    let attn_fn = self.device.get_func(MODULE_NAME, "gqa_attention")
                                        .ok_or_else(|| "gqa_attention not found".to_string())?;
                                    unsafe {
                                        attn_fn.launch(
                                            LaunchConfig {
                                                grid_dim: (nh as u32, 1, 1),
                                                block_dim: (threads, 1, 1),
                                                shared_mem_bytes,
                                            },
                                            (*graph.d_gqa_out.device_ptr(), *graph.d_gqa_q.device_ptr(),
                                             graph.kv_k_ptrs[layer_idx], graph.kv_v_ptrs[layer_idx],
                                             sm_sc, nh as i32, nkv as i32, hd as i32,
                                             seq_len as i32, graph.kv_max_seq as i32, 1i32),
                                        ).map_err(|e| format!("batch gqa[{}][{}]: {:?}", layer_idx, t, e))?;
                                    }
                                }
                            }

                            // Gated attention + BF16 conversion (or just BF16 conversion)
                            let out_ptr = d_bao_ptr + (t * o_size * 2) as u64;
                            if is_gated {
                                let total = o_size as u32;
                                let threads = 256u32;
                                let blocks = (total + threads - 1) / threads;
                                unsafe {
                                    k.apply_gated_attn_bf16.clone().launch(
                                        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                        (out_ptr, *graph.d_gqa_out.device_ptr(), *graph.d_la_qkvz.device_ptr(), o_size as i32),
                                    ).map_err(|e| format!("batch gated_attn_bf16[{}][{}]: {:?}", layer_idx, t, e))?;
                                }
                            } else {
                                unsafe {
                                    k.fp32_to_bf16.clone().launch(
                                        LaunchConfig::for_num_elems(o_size as u32),
                                        (out_ptr, *graph.d_gqa_out.device_ptr(), o_size as i32),
                                    ).map_err(|e| format!("batch fp32_to_bf16_o[{}][{}]: {:?}", layer_idx, t, e))?;
                                }
                            }
                        } // end per-token GQA loop

                        // E. Batch output projection GEMM (weights loaded ONCE)
                        let ow = &graph.weights[*o_proj];
                        self.gemm_bf16_batch(
                            ow, d_bao_ptr, d_bh_ptr,
                            batch_size, o_size, hs)?;
                    } else {
                        // Non-fused Q/K/V: fall back to per-token GEMV
                        let qw = &graph.weights[*q_proj];
                        let kw = &graph.weights[*k_proj];
                        let vw = &graph.weights[*v_proj];

                        for t in 0..batch_size {
                            let position = positions[t];
                            let hidden_ptr = d_bh_ptr + (t * hs * 2) as u64;

                            self.gemv_bf16_to_f32(qw, hidden_ptr, *graph.d_gqa_q.device_ptr())?;
                            self.gemv_bf16_to_f32(kw, hidden_ptr, *graph.d_gqa_k.device_ptr())?;
                            self.gemv_bf16_to_f32(vw, hidden_ptr, *graph.d_gqa_v.device_ptr())?;

                            // Split gated Q (same as fused path)
                            if is_gated {
                                let total = (nh * hd) as u32;
                                let threads = 256u32;
                                let blocks = (total + threads - 1) / threads;
                                unsafe {
                                    let split_fn = self.device.get_func(MODULE_NAME, "split_gated_q")
                                        .ok_or_else(|| "split_gated_q not found".to_string())?;
                                    split_fn.launch(
                                        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                        (*graph.d_gqa_q.device_ptr(), *graph.d_la_qkvz.device_ptr(),
                                         *graph.d_gqa_q.device_ptr(), nh as i32, hd as i32),
                                    ).map_err(|e| format!("batch split_gated_q[{}][{}]: {:?}", layer_idx, t, e))?;
                                }
                            }

                            // QK norm
                            if qnp != 0 {
                                let threads = 256u32;
                                let norm_fn = self.device.get_func(MODULE_NAME, "per_head_rmsnorm")
                                    .ok_or_else(|| "per_head_rmsnorm not found".to_string())?;
                                unsafe {
                                    norm_fn.clone().launch(
                                        LaunchConfig { grid_dim: (nh as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                        (*graph.d_gqa_q.device_ptr(), qnp, eps, nh as i32, hd as i32, 0i32),
                                    ).map_err(|e| format!("batch qnorm[{}][{}]: {:?}", layer_idx, t, e))?;
                                    norm_fn.launch(
                                        LaunchConfig { grid_dim: (nkv as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                        (*graph.d_gqa_k.device_ptr(), knp, eps, nkv as i32, hd as i32, 0i32),
                                    ).map_err(|e| format!("batch knorm[{}][{}]: {:?}", layer_idx, t, e))?;
                                }
                            }

                            // RoPE
                            if let Some(ref d_cos) = graph.d_rope_cos {
                                let half_dim = graph.rope_half_dim;
                                let total_heads = nh + nkv;
                                let total_work = total_heads * half_dim;
                                let threads = 256u32;
                                let blocks = ((total_work as u32) + threads - 1) / threads;
                                let rope_fn = self.device.get_func(MODULE_NAME, "apply_rope")
                                    .ok_or_else(|| "apply_rope not found".to_string())?;
                                unsafe {
                                    rope_fn.launch(
                                        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                        (*graph.d_gqa_q.device_ptr(), *graph.d_gqa_k.device_ptr(),
                                         *d_cos.device_ptr(), *graph.d_rope_sin.as_ref().unwrap().device_ptr(),
                                         position as i32, nh as i32, nkv as i32, hd as i32, half_dim as i32),
                                    ).map_err(|e| format!("batch rope[{}][{}]: {:?}", layer_idx, t, e))?;
                                }
                            }

                            // KV cache write
                            if layer_idx < graph.kv_k_ptrs.len()
                                && graph.kv_k_ptrs[layer_idx] != 0 {
                                let threads = 256u32;
                                let blocks = ((kv_stride as u32) + threads - 1) / threads;
                                let kv_fn = self.device.get_func(MODULE_NAME, "kv_cache_write")
                                    .ok_or_else(|| "kv_cache_write not found".to_string())?;
                                unsafe {
                                    kv_fn.launch(
                                        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                        (graph.kv_k_ptrs[layer_idx], graph.kv_v_ptrs[layer_idx],
                                         *graph.d_gqa_k.device_ptr(), *graph.d_gqa_v.device_ptr(),
                                         position as i32, kv_stride as i32),
                                    ).map_err(|e| format!("batch kv_write[{}][{}]: {:?}", layer_idx, t, e))?;
                                }
                            }

                            // GQA attention (same as fused path above)
                            {
                                let seq_len = (position + 1) as u32;
                                let threads = 256u32;
                                let q_smem = (hd as u32) * 4;
                                let use_tiled = graph.d_gqa_tiled_o.is_some()
                                    && seq_len > (graph.gqa_tile_size * graph.gqa_num_q_heads) as u32;
                                if use_tiled {
                                    let tile_size = graph.gqa_tile_size;
                                    let num_tiles = ((seq_len as usize) + tile_size - 1) / tile_size;
                                    let tiled_fn = self.device.get_func(MODULE_NAME, "gqa_attention_tiled")
                                        .ok_or_else(|| "gqa_attention_tiled not found".to_string())?;
                                    let smem = q_smem + (tile_size as u32) * 4 + 128;
                                    unsafe {
                                        tiled_fn.launch(
                                            LaunchConfig {
                                                grid_dim: (nh as u32, num_tiles as u32, 1),
                                                block_dim: (threads, 1, 1),
                                                shared_mem_bytes: smem,
                                            },
                                            (*graph.d_gqa_tiled_o.as_ref().unwrap().device_ptr(),
                                             *graph.d_gqa_tiled_lse.as_ref().unwrap().device_ptr(),
                                             *graph.d_gqa_q.device_ptr(),
                                             graph.kv_k_ptrs[layer_idx],
                                             graph.kv_v_ptrs[layer_idx],
                                             sm_sc, nh as i32, nkv as i32, hd as i32,
                                             seq_len as i32, graph.kv_max_seq as i32,
                                             tile_size as i32),
                                        ).map_err(|e| format!("batch gqa_tiled[{}][{}]: {:?}", layer_idx, t, e))?;
                                    }
                                    let reduce_fn = self.device.get_func(MODULE_NAME, "gqa_attention_reduce")
                                        .ok_or_else(|| "gqa_attention_reduce not found".to_string())?;
                                    unsafe {
                                        reduce_fn.launch(
                                            LaunchConfig { grid_dim: (nh as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                            (*graph.d_gqa_out.device_ptr(),
                                             *graph.d_gqa_tiled_o.as_ref().unwrap().device_ptr(),
                                             *graph.d_gqa_tiled_lse.as_ref().unwrap().device_ptr(),
                                             nh as i32, hd as i32, num_tiles as i32),
                                        ).map_err(|e| format!("batch gqa_reduce[{}][{}]: {:?}", layer_idx, t, e))?;
                                    }
                                } else {
                                    let shared_mem_bytes = q_smem + seq_len * 4 + 128;
                                    let attn_fn = self.device.get_func(MODULE_NAME, "gqa_attention")
                                        .ok_or_else(|| "gqa_attention not found".to_string())?;
                                    unsafe {
                                        attn_fn.launch(
                                            LaunchConfig {
                                                grid_dim: (nh as u32, 1, 1),
                                                block_dim: (threads, 1, 1),
                                                shared_mem_bytes,
                                            },
                                            (*graph.d_gqa_out.device_ptr(), *graph.d_gqa_q.device_ptr(),
                                             graph.kv_k_ptrs[layer_idx], graph.kv_v_ptrs[layer_idx],
                                             sm_sc, nh as i32, nkv as i32, hd as i32,
                                             seq_len as i32, graph.kv_max_seq as i32, 1i32),
                                        ).map_err(|e| format!("batch gqa[{}][{}]: {:?}", layer_idx, t, e))?;
                                    }
                                }
                            }

                            // Gated attention + BF16 conversion (or just BF16 conversion)
                            if is_gated {
                                let total = o_size as u32;
                                let threads = 256u32;
                                let blocks = (total + threads - 1) / threads;
                                unsafe {
                                    k.apply_gated_attn_bf16.clone().launch(
                                        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 },
                                        (*graph.d_scratch.device_ptr(), *graph.d_gqa_out.device_ptr(),
                                         *graph.d_la_qkvz.device_ptr(), o_size as i32),
                                    ).map_err(|e| format!("batch gated_attn_bf16[{}][{}]: {:?}", layer_idx, t, e))?;
                                }
                            } else {
                                unsafe {
                                    k.fp32_to_bf16.clone().launch(
                                        LaunchConfig::for_num_elems(o_size as u32),
                                        (*graph.d_scratch.device_ptr(), *graph.d_gqa_out.device_ptr(), o_size as i32),
                                    ).map_err(|e| format!("batch fp32_to_bf16_o[{}][{}]: {:?}", layer_idx, t, e))?;
                                }
                            }

                            // O projection (per-token GEMV for non-fused path)
                            {
                                let ow = &graph.weights[*o_proj];
                                self.gemv_bf16_internal(ow, *graph.d_scratch.device_ptr(), hidden_ptr)?;
                            }
                        } // end per-token non-fused GQA loop
                    }
                }

                GpuAttnConfig::MLA { .. } => {
                    return Err("MLA not implemented for batched decode".to_string());
                }

                GpuAttnConfig::Mamba2 { .. } => {
                    return Err("Mamba2 not implemented for batched decode".to_string());
                }
            }

            if do_timing {
                self.device.synchronize().map_err(|e| format!("timing: {:?}", e))?;
                tt_attn += t_attn_start.elapsed().as_secs_f64();
            }
            let t_norm2_start = std::time::Instant::now();

            // ── F. Post-attention norm (in-place on batch arrays) ──
            {
                let smem = (hs as u32) * 4;
                let threads = 256u32.min(hs as u32);
                for t in 0..batch_size {
                    let h_ptr = d_bh_ptr + (t * hs * 2) as u64;
                    let r_ptr = d_br_ptr + (t * hs * 2) as u64;
                    unsafe {
                        k.fused_add_rmsnorm.clone().launch(
                            LaunchConfig { grid_dim: (1, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: smem },
                            (h_ptr, r_ptr, graph.layers[layer_idx].post_attn_norm_ptr, eps, hs as i32, 0i32),
                        ).map_err(|e| format!("batch post_norm[{}][{}]: {:?}", layer_idx, t, e))?;
                    }
                }
            }

            if is_la { la_stack_idx += 1; }

            if do_timing {
                self.device.synchronize().map_err(|e| format!("timing: {:?}", e))?;
                tt_norm += t_norm2_start.elapsed().as_secs_f64();
            }
            let t_moe_start = std::time::Instant::now();

            // ── G. MoE or Dense MLP ──
            if has_moe {
                // Batched MoE with fail-fast: route all tokens, check expert divergence,
                // potentially truncate batch, then DMA expert union and compute.
                batch_size = self.moe_forward_batched(graph, layer_idx, batch_size)?;

                // Copy moe_out[t] → hidden[t] for each token
                for t in 0..batch_size {
                    let offset = (t * hs * 2) as u64;
                    unsafe {
                        cuda_sys::lib().cuMemcpyDtoD_v2(
                            d_bh_ptr + offset, d_bmo_ptr + offset, hs * 2);
                    }
                }
            } else if let GpuMlpConfig::Dense { gate_proj, up_proj, down_proj } = &graph.layers[layer_idx].mlp {
                // Dense MLP: process each token separately (TODO: batch GEMM)
                let gw = &graph.weights[*gate_proj];
                let uw = &graph.weights[*up_proj];
                let dw = &graph.weights[*down_proj];
                let intermediate = gw.rows;

                for t in 0..batch_size {
                    let h_ptr = d_bh_ptr + (t * hs * 2) as u64;

                    self.gemv_bf16_internal(gw, h_ptr,
                        *graph.d_expert_gate_up.device_ptr())?;
                    let up_out_ptr = unsafe {
                        (*graph.d_expert_gate_up.device_ptr() as *const u16).add(intermediate) as u64
                    };
                    self.gemv_bf16_internal(uw, h_ptr, up_out_ptr)?;

                    unsafe {
                        k.silu_mul.clone().launch(
                            LaunchConfig::for_num_elems(intermediate as u32),
                            (*graph.d_expert_scratch.device_ptr(), *graph.d_expert_gate_up.device_ptr(), intermediate as i32),
                        ).map_err(|e| format!("batch silu[{}][{}]: {:?}", layer_idx, t, e))?;
                    }

                    self.gemv_bf16_internal(dw, *graph.d_expert_scratch.device_ptr(), h_ptr)?;
                }
            }

            if do_timing {
                self.device.synchronize().map_err(|e| format!("timing: {:?}", e))?;
                tt_moe += t_moe_start.elapsed().as_secs_f64();
            }
        } // end layer loop

        let t_lm_start = std::time::Instant::now();

        // ── 3. Final norm (in-place) + LM head (batch GEMM) ──
        {
            let smem = (hs as u32) * 4;
            let threads = 256u32.min(hs as u32);
            for t in 0..batch_size {
                let h_ptr = d_bh_ptr + (t * hs * 2) as u64;
                let r_ptr = d_br_ptr + (t * hs * 2) as u64;
                unsafe {
                    k.fused_add_rmsnorm.clone().launch(
                        LaunchConfig { grid_dim: (1, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: smem },
                        (h_ptr, r_ptr, graph.final_norm_ptr, eps, hs as i32, 0i32),
                    ).map_err(|e| format!("batch final_norm[{}]: {:?}", t, e))?;
                }
            }
        }

        // Batch LM head GEMM: all tokens at once (weights loaded ONCE)
        {
            let lm_w = &graph.weights[graph.lm_head_wid];
            let logits_ptr = *graph.d_batch_logits.as_ref().unwrap().device_ptr();
            self.gemm_bf16_to_f32_batch(
                lm_w, d_bh_ptr, logits_ptr,
                batch_size, hs, graph.vocab_size)?;
        }

        // ── 4. Sync + D2H all batch logits ──
        self.device.synchronize().map_err(|e| format!("batch sync: {:?}", e))?;
        {
            let total_logits = batch_size * graph.vocab_size;
            unsafe {
                let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                    graph.h_batch_logits.as_mut_ptr() as *mut std::ffi::c_void,
                    *graph.d_batch_logits.as_ref().unwrap().device_ptr(),
                    total_logits * 4);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("batch D2H logits: {:?}", err));
                }
            }
        }

        if do_timing {
            self.device.synchronize().map_err(|e| format!("timing: {:?}", e))?;
            tt_lmhead += t_lm_start.elapsed().as_secs_f64();
            eprintln!("  BATCH-TIMING batch={}/{}: norm={:.1}ms attn={:.1}ms moe={:.1}ms lmhead={:.1}ms total={:.1}ms",
                batch_size, orig_batch_size,
                tt_norm * 1000.0, tt_attn * 1000.0, tt_moe * 1000.0, tt_lmhead * 1000.0,
                (tt_norm + tt_attn + tt_moe + tt_lmhead) * 1000.0);
        }

        Ok(batch_size)
    }

    /// Batched MoE forward: route all batch tokens through one MoE layer.
    /// Takes expert union, DMAs each unique expert once, computes all tokens.
    ///
    /// Returns the (potentially reduced) batch size after fail-fast divergence check.
    /// If draft tokens' expert routing diverges from token[0]'s routing
    /// (Jaccard similarity below threshold), the batch is truncated to exclude
    /// the divergent tokens and all subsequent ones.
    fn moe_forward_batched(
        &self,
        graph: &mut GpuDecodeGraph,
        layer_idx: usize,
        batch_size: usize,
    ) -> Result<usize, String> {
        use std::collections::HashMap;

        let device = &self.device;
        let copy_stream = self.copy_stream.0;

        let moe = graph.moe_layers.get(layer_idx)
            .and_then(|m| m.as_ref())
            .ok_or_else(|| format!("MoE layer {} not registered", layer_idx))?;

        let hs = graph.hidden_size;
        let intermediate = graph.moe_intermediate_size;
        let gs = graph.group_size;
        let topk = moe.topk;
        let ne = moe.num_experts;
        let sf = moe.scoring_func;
        let rsf = moe.routed_scaling_factor;
        let gate_wid = moe.gate_wid;
        let gate_bias_ptr = moe.gate_bias_ptr;
        let e_score_corr_ptr = moe.e_score_corr_ptr;
        let inv_wp = if graph.expert_bits == 8 {
            *graph.d_inv_weight_perm_int8.device_ptr()
        } else {
            *graph.d_inv_weight_perm.device_ptr()
        };
        let inv_sp = *graph.d_inv_scale_perm.device_ptr();
        let is_int8 = graph.expert_bits == 8;

        let w13_n = 2 * intermediate;
        let w13_k_tiles = hs / 16;
        let w13_max_ksplits = w13_k_tiles / 16;
        let w13_ksplits = if w13_max_ksplits > 1 {
            let n_tiles = (w13_n + 15) / 16;
            let target = graph.num_sms * 4;
            let desired = (target + n_tiles - 1) / n_tiles;
            desired.clamp(1, w13_max_ksplits.min(8))
        } else { 1 };
        let use_v2_w13 = w13_ksplits > 1;
        let partial_ptr = *graph.d_v2_partial.device_ptr();

        let k = graph.kernels.as_ref()
            .ok_or_else(|| "Kernels not cached".to_string())?;

        let d_bh_ptr = *graph.d_batch_hidden.as_ref()
            .ok_or("batch buffers not allocated")?.device_ptr();
        let d_bmo_ptr = *graph.d_batch_moe_out.as_ref()
            .ok_or("batch buffers not allocated")?.device_ptr();

        // Use pre-allocated events
        let pre_ev = &graph.pre_events;
        let ev_dma: [cuda_sys::CUevent; 2];
        let ev_compute: [cuda_sys::CUevent; 2];
        if let Some(ref pe) = pre_ev {
            ev_dma = [pe[0].0, pe[1].0];
            ev_compute = [pe[2].0, pe[3].0];
        } else {
            unsafe {
                let flags = cuda_sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32;
                let mut events = [std::ptr::null_mut(); 4];
                for e in events.iter_mut() { cuda_sys::lib().cuEventCreate(e, flags); }
                ev_dma = [events[0], events[1]];
                ev_compute = [events[2], events[3]];
            }
        }

        let default_stream: cuda_sys::CUstream = std::ptr::null_mut();

        // ── Step 1: Batched gate + TopK routing (single sync) ──
        // All gate GEMVs + topk kernels launched without sync, then single sync + D2H.
        let d_gate_ptr = *graph.d_batch_gate_logits.as_ref()
            .ok_or("batch gate logits not allocated")?.device_ptr();
        let d_btk_ids_ptr = *graph.d_batch_topk_ids.as_ref()
            .ok_or("batch topk ids not allocated")?.device_ptr();
        let d_btk_wts_ptr = *graph.d_batch_topk_wts.as_ref()
            .ok_or("batch topk wts not allocated")?.device_ptr();

        for t in 0..batch_size {
            let hidden_ptr = d_bh_ptr + (t * hs * 2) as u64;
            let logits_ptr = d_gate_ptr + (t * ne * 4) as u64;
            let tk_ids_ptr = d_btk_ids_ptr + (t * topk * 4) as u64;
            let tk_wts_ptr = d_btk_wts_ptr + (t * topk * 4) as u64;

            // Gate GEMV → per-token gate logits buffer
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
                        hidden_ptr as *const std::ffi::c_void, cublas_sys::cudaDataType::CUDA_R_16BF, hs as i32,
                        &beta as *const f32 as *const std::ffi::c_void,
                        logits_ptr as *mut std::ffi::c_void,
                        cublas_sys::cudaDataType::CUDA_R_32F, w.rows as i32,
                        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    ).map_err(|e| format!("batch gate GEMV[{}][{}]: {:?}", layer_idx, t, e))?;
                }
            }

            // TopK → per-token topk buffer on GPU
            {
                let smem = (ne as u32) * 4;
                let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: smem };
                if sf == 1 {
                    let bias_ptr = if gate_bias_ptr != 0 { gate_bias_ptr } else { 0u64 };
                    let corr_ptr = if e_score_corr_ptr != 0 { e_score_corr_ptr } else { 0u64 };
                    unsafe {
                        k.sigmoid_topk.clone().launch(cfg, (
                            logits_ptr, bias_ptr, corr_ptr,
                            tk_ids_ptr, tk_wts_ptr,
                            ne as i32, topk as i32,
                        )).map_err(|e| format!("batch sigmoid_topk[{}][{}]: {:?}", layer_idx, t, e))?;
                    }
                } else {
                    unsafe {
                        k.softmax_topk.clone().launch(cfg, (
                            logits_ptr,
                            tk_ids_ptr, tk_wts_ptr,
                            ne as i32, topk as i32,
                        )).map_err(|e| format!("batch softmax_topk[{}][{}]: {:?}", layer_idx, t, e))?;
                    }
                }
            }
        }

        // Single sync + D2H for all tokens' routing results
        device.synchronize().map_err(|e| format!("batch route sync: {:?}", e))?;
        unsafe {
            cuda_sys::lib().cuMemcpyDtoH_v2(
                graph.h_batch_topk_ids.as_mut_ptr() as *mut std::ffi::c_void,
                d_btk_ids_ptr, batch_size * topk * 4);
            cuda_sys::lib().cuMemcpyDtoH_v2(
                graph.h_batch_topk_weights.as_mut_ptr() as *mut std::ffi::c_void,
                d_btk_wts_ptr, batch_size * topk * 4);
        }

        // ── Step 1.5: Fail-fast expert divergence check (Jaccard similarity) ──
        // Compare each draft token's expert routing against token[0]'s routing.
        // If a draft token diverges, truncate the batch at that point.
        let active_batch = if batch_size > 1 && self.spec_jaccard_threshold > 0.0 {
            // Build token[0]'s expert set
            let mut token0_experts = [false; 256]; // bitset (max 256 experts)
            let mut token0_count = 0usize;
            for j in 0..topk {
                let eid = graph.h_batch_topk_ids[j];
                if eid >= 0 && (eid as usize) < 256 {
                    token0_experts[eid as usize] = true;
                    token0_count += 1;
                }
            }

            let mut new_batch = batch_size;
            for t in 1..batch_size {
                let mut intersection = 0usize;
                let mut t_count = 0usize;
                for j in 0..topk {
                    let eid = graph.h_batch_topk_ids[t * topk + j];
                    if eid >= 0 && (eid as usize) < 256 {
                        t_count += 1;
                        if token0_experts[eid as usize] {
                            intersection += 1;
                        }
                    }
                }
                let union = token0_count + t_count - intersection;
                let jaccard = if union > 0 { intersection as f32 / union as f32 } else { 0.0 };

                if jaccard < self.spec_jaccard_threshold {
                    new_batch = t; // Keep tokens 0..t-1, drop t and all after
                    log::debug!("fail-fast: layer {} token {} Jaccard {:.3} < {:.3}, batch {} → {}",
                        layer_idx, t, jaccard, self.spec_jaccard_threshold, batch_size, new_batch);
                    break;
                }
            }
            new_batch
        } else {
            batch_size
        };

        // ── Step 2: Compute expert union across active tokens ──
        // expert_tokens: expert_id → Vec<(token_idx, weight)>
        let mut expert_tokens: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for t in 0..active_batch {
            for j in 0..topk {
                let eid = graph.h_batch_topk_ids[t * topk + j];
                if eid < 0 { continue; }
                let weight = graph.h_batch_topk_weights[t * topk + j];
                expert_tokens.entry(eid as usize).or_default().push((t, weight));
            }

        }

        // ── Step 3: Zero batch moe_out accumulators ──
        for t in 0..active_batch {
            let out_ptr = d_bmo_ptr + (t * hs * 2) as u64;
            unsafe {
                k.zero_bf16.clone().launch(
                    LaunchConfig::for_num_elems(hs as u32),
                    (out_ptr, hs as i32),
                ).map_err(|e| format!("batch zero_moe[{}][{}]: {:?}", layer_idx, t, e))?;
            }
        }

        // ── Step 4: Expert loop — DMA each unique expert once, compute all tokens ──
        let use_double_buf = graph.expert_buf_total_size > 0;
        let buf_base = [
            *graph.d_expert_buf[0].device_ptr(),
            *graph.d_expert_buf[1].device_ptr(),
        ];
        let w13p_off = graph.expert_buf_w13p_offset;
        let w13s_off = graph.expert_buf_w13s_offset;
        let w2p_off = graph.expert_buf_w2p_offset;
        let w2s_off = graph.expert_buf_w2s_offset;

        let buf_w13_packed = *graph.d_expert_buf_a0.device_ptr();
        let buf_w13_scales = *graph.d_expert_buf_b0.device_ptr();
        let buf_w2_packed = *graph.d_expert_buf_a1.device_ptr();
        let buf_w2_scales = *graph.d_expert_buf_b1.device_ptr();

        let mut dma_expert_count = 0u32;

        for (&eid, token_list) in &expert_tokens {
            let expert = &moe.experts[eid];

            // Check HCS first (fast flat lookup)
            let hcs_ptrs = if let Some(ref hcs) = graph.hcs {
                hcs.get_fast(layer_idx, eid)
            } else { None };

            let (w13p, w13s, w2p, w2s) = if let Some(ptrs) = hcs_ptrs {
                // HCS hit — no DMA needed
                ptrs
            } else if use_double_buf {
                // DMA to ping-pong buffer
                let slot = (dma_expert_count % 2) as usize;
                if dma_expert_count >= 2 {
                    unsafe { cuda_sys::lib().cuStreamWaitEvent(copy_stream, ev_compute[slot], 0); }
                }

                unsafe {
                    let base = buf_base[slot];
                    if expert.contiguous_ptr != 0 {
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            base, expert.contiguous_ptr as *const std::ffi::c_void,
                            expert.contiguous_bytes, copy_stream);
                    } else {
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            base + w13p_off as u64, expert.w13_packed_ptr as *const std::ffi::c_void,
                            expert.w13_packed_bytes, copy_stream);
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            base + w13s_off as u64, expert.w13_scales_ptr as *const std::ffi::c_void,
                            expert.w13_scales_bytes, copy_stream);
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            base + w2p_off as u64, expert.w2_packed_ptr as *const std::ffi::c_void,
                            expert.w2_packed_bytes, copy_stream);
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            base + w2s_off as u64, expert.w2_scales_ptr as *const std::ffi::c_void,
                            expert.w2_scales_bytes, copy_stream);
                    }
                    cuda_sys::lib().cuEventRecord(ev_dma[slot], copy_stream);
                    cuda_sys::lib().cuStreamWaitEvent(default_stream, ev_dma[slot], 0);
                }

                let base = buf_base[slot];
                dma_expert_count += 1;

                (base + w13p_off as u64, base + w13s_off as u64,
                 base + w2p_off as u64, base + w2s_off as u64)
            } else {
                // Legacy single-buffer path
                unsafe {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_w13_packed, expert.w13_packed_ptr as *const std::ffi::c_void,
                        expert.w13_packed_bytes, copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_w13_scales, expert.w13_scales_ptr as *const std::ffi::c_void,
                        expert.w13_scales_bytes, copy_stream);
                    cuda_sys::lib().cuEventRecord(ev_dma[0], copy_stream);
                    cuda_sys::lib().cuStreamWaitEvent(default_stream, ev_dma[0], 0);
                }
                (buf_w13_packed, buf_w13_scales, buf_w2_packed, buf_w2_scales)
            };

            // Compute all tokens that route to this expert
            for &(t, weight) in token_list {
                let hidden_ptr = d_bh_ptr + (t * hs * 2) as u64;
                let accum_ptr = d_bmo_ptr + (t * hs * 2) as u64;

                // w13 GEMV
                if use_v2_w13 {
                    self.launch_marlin_gemv_v2(
                        w13p, w13s, hidden_ptr, partial_ptr, inv_wp, inv_sp,
                        hs, w13_n, gs, w13_ksplits, k, is_int8).map_err(|e| format!("{}", e))?;
                    self.launch_reduce_ksplits_bf16(
                        *graph.d_expert_gate_up.device_ptr(), partial_ptr,
                        w13_n, w13_ksplits, k).map_err(|e| format!("{}", e))?;
                } else {
                    self.launch_marlin_gemv_raw(
                        w13p, w13s, hidden_ptr,
                        *graph.d_expert_gate_up.device_ptr(),
                        inv_wp, inv_sp, hs, w13_n, gs, is_int8).map_err(|e| format!("{}", e))?;
                }

                // Fused silu + w2 + weighted accumulate
                self.launch_fused_silu_accum(
                    w2p, w2s,
                    *graph.d_expert_gate_up.device_ptr(),
                    accum_ptr, inv_wp, inv_sp,
                    intermediate, hs, gs,
                    weight, 0u64, k, is_int8).map_err(|e| format!("{}", e))?;
            }

            // Signal compute done for ping-pong
            if use_double_buf && hcs_ptrs.is_none() {
                let slot = ((dma_expert_count - 1) % 2) as usize;
                unsafe {
                    cuda_sys::lib().cuEventRecord(ev_compute[slot], default_stream);
                }
            }

            // Legacy path: DMA w2 after w13 compute
            if !use_double_buf && hcs_ptrs.is_none() {
                unsafe {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_w2_packed, expert.w2_packed_ptr as *const std::ffi::c_void,
                        expert.w2_packed_bytes, copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_w2_scales, expert.w2_scales_ptr as *const std::ffi::c_void,
                        expert.w2_scales_bytes, copy_stream);
                    cuda_sys::lib().cuEventRecord(ev_dma[1], copy_stream);
                    cuda_sys::lib().cuStreamWaitEvent(default_stream, ev_dma[1], 0);
                }
            }
        }

        // ── Step 5: Shared expert for each token ──
        if moe.shared.is_some() {
            let se_vram = graph.shared_expert_vram.get(layer_idx).and_then(|e| e.as_ref());

            let (w13p, w13s, w2p, w2s) = if let Some(entry) = se_vram {
                (entry.w13_packed_ptr(), entry.w13_scales_ptr(),
                 entry.w2_packed_ptr(), entry.w2_scales_ptr())
            } else {
                let shared = moe.shared.as_ref().unwrap();
                unsafe {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_w13_packed, shared.w13_packed_ptr as *const std::ffi::c_void,
                        shared.w13_packed_bytes, copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        buf_w13_scales, shared.w13_scales_ptr as *const std::ffi::c_void,
                        shared.w13_scales_bytes, copy_stream);
                    cuda_sys::lib().cuEventRecord(ev_dma[0], copy_stream);
                    cuda_sys::lib().cuStreamWaitEvent(default_stream, ev_dma[0], 0);
                }
                (buf_w13_packed, buf_w13_scales, buf_w2_packed, buf_w2_scales)
            };

            for t in 0..active_batch {
                let hidden_ptr = d_bh_ptr + (t * hs * 2) as u64;
                let accum_ptr = d_bmo_ptr + (t * hs * 2) as u64;

                // w13 GEMV
                self.launch_marlin_gemv_raw(
                    w13p, w13s, hidden_ptr,
                    *graph.d_expert_gate_up.device_ptr(),
                    inv_wp, inv_sp, hs, 2 * intermediate, gs, is_int8).map_err(|e| format!("{}", e))?;

                // w2 DMA (if not VRAM resident, first token only)
                if se_vram.is_none() && t == 0 {
                    let shared = moe.shared.as_ref().unwrap();
                    unsafe {
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            buf_w2_packed, shared.w2_packed_ptr as *const std::ffi::c_void,
                            shared.w2_packed_bytes, copy_stream);
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            buf_w2_scales, shared.w2_scales_ptr as *const std::ffi::c_void,
                            shared.w2_scales_bytes, copy_stream);
                        cuda_sys::lib().cuEventRecord(ev_dma[1], copy_stream);
                        cuda_sys::lib().cuStreamWaitEvent(default_stream, ev_dma[1], 0);
                    }
                }

                // Shared gate
                let gate_weight_ptr = if let Some(sg_wid) = moe.shared_gate_wid {
                    let sg_w = &graph.weights[sg_wid];
                    self.gemv_bf16_to_f32(sg_w, hidden_ptr, *graph.d_scratch.device_ptr())
                        .map_err(|e| format!("batch shared gate: {}", e))?;
                    *graph.d_scratch.device_ptr()
                } else { 0u64 };
                let shared_weight = if gate_weight_ptr != 0 { 0.0f32 } else { 1.0f32 };

                self.launch_fused_silu_accum(
                    w2p, w2s,
                    *graph.d_expert_gate_up.device_ptr(),
                    accum_ptr, inv_wp, inv_sp,
                    intermediate, hs, gs,
                    shared_weight, gate_weight_ptr, k, is_int8).map_err(|e| format!("{}", e))?;
            }
        }

        // ── Step 6: Scale by routed_scaling_factor ──
        if rsf != 1.0 {
            for t in 0..active_batch {
                let out_ptr = d_bmo_ptr + (t * hs * 2) as u64;
                unsafe {
                    k.scale_bf16.clone().launch(
                        LaunchConfig::for_num_elems(hs as u32),
                        (out_ptr, out_ptr, rsf, hs as i32),
                    ).map_err(|e| format!("batch scale[{}][{}]: {:?}", layer_idx, t, e))?;
                }
            }
        }

        Ok(active_batch)
    }

    /// BF16 GEMV: output_bf16[N] = weight[N,K] @ input_bf16[K]
    /// Supports BF16, Marlin INT8, Marlin INT4, and simple INT4 weights.
    fn gemv_bf16_internal(&self, w: &GpuWeight, input_ptr: u64, output_ptr: u64) -> Result<(), String> {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        if w.is_marlin_int8() {
            return self.launch_marlin_gemv_int8_bf16(w, input_ptr, output_ptr);
        }
        if w.is_marlin_int4() {
            // Prefer simple INT4 for decode (single kernel, no tile permutation overhead)
            if w.has_simple_int4() {
                return self.launch_simple_int4_gemv_bf16(w, input_ptr, output_ptr);
            }
            return self.launch_marlin_gemv_int4_bf16(w, input_ptr, output_ptr);
        }
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

    /// GEMV with FP32 output: output_f32[N] = weight[N,K] @ input_bf16[K]
    /// Supports BF16, Marlin INT8, Marlin INT4, and simple INT4 weights.
    fn gemv_bf16_to_f32(&self, w: &GpuWeight, input_ptr: u64, output_ptr: u64) -> Result<(), String> {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        if w.is_marlin_int8() {
            return self.launch_marlin_gemv_int8_f32(w, input_ptr, output_ptr);
        }
        if w.is_marlin_int4() {
            // Prefer simple INT4 for decode (single kernel, no tile permutation overhead)
            if w.has_simple_int4() {
                return self.launch_simple_int4_gemv_f32(w, input_ptr, output_ptr);
            }
            return self.launch_marlin_gemv_int4_f32(w, input_ptr, output_ptr);
        }
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

    /// Batched GEMM: output_f32[M, N] = weight_bf16[M, K]^T @ input_bf16[K, N]
    /// Weight is [K, M] in memory (column-major), transposed via OP_T to act as [M, K].
    /// Input is [K, N] column-major (N hidden vectors of K elements each, stride = ldb).
    /// Output is [M, N] column-major (N output vectors of M elements each, stride = ldc).
    fn gemm_bf16_to_f32_batch(
        &self, w: &GpuWeight, input_ptr: u64, output_ptr: u64,
        n: usize, ldb: usize, ldc: usize,
    ) -> Result<(), String> {
        if n == 1 {
            return self.gemv_bf16_to_f32(w, input_ptr, output_ptr);
        }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            cublas_result::gemm_ex(
                *self.blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                w.rows as i32, n as i32, w.cols as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                w.ptr as *const std::ffi::c_void, w.cublas_data_type(), w.cols as i32,
                input_ptr as *const std::ffi::c_void, w.cublas_data_type(), ldb as i32,
                &beta as *const f32 as *const std::ffi::c_void,
                output_ptr as *mut std::ffi::c_void,
                cublas_sys::cudaDataType::CUDA_R_32F, ldc as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            ).map_err(|e| format!("cuBLAS gemm_bf16_to_f32_batch: {:?}", e))?;
        }
        Ok(())
    }

    /// Batched GEMM: output_bf16[M, N] = weight_bf16[M, K]^T @ input_bf16[K, N]
    /// Same as gemm_bf16_to_f32_batch but output is BF16. Used for output projections.
    fn gemm_bf16_batch(
        &self, w: &GpuWeight, input_ptr: u64, output_ptr: u64,
        n: usize, ldb: usize, ldc: usize,
    ) -> Result<(), String> {
        if n == 1 {
            return self.gemv_bf16_internal(w, input_ptr, output_ptr);
        }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            cublas_result::gemm_ex(
                *self.blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                w.rows as i32, n as i32, w.cols as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                w.ptr as *const std::ffi::c_void, w.cublas_data_type(), w.cols as i32,
                input_ptr as *const std::ffi::c_void, w.cublas_data_type(), ldb as i32,
                &beta as *const f32 as *const std::ffi::c_void,
                output_ptr as *mut std::ffi::c_void,
                w.cublas_data_type(), ldc as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            ).map_err(|e| format!("cuBLAS gemm_bf16_batch: {:?}", e))?;
        }
        Ok(())
    }

    /// Evict soft-tier HCS experts to free VRAM before prefill.
    /// Uses calibration data to determine how many soft slots to evict
    /// based on the estimated prompt length.
    /// Returns (evicted_count, freed_mb).
    pub fn hcs_evict_for_prefill(&mut self, estimated_tokens: usize) -> (usize, f64) {
        self.last_soft_evict_experts = 0;
        self.last_soft_evict_freed_mb = 0.0;
        self.last_soft_reload_queued = 0;
        self.last_soft_reload_alloc_mb = 0.0;
        self.last_soft_reload_activated = 0;
        let cal = self.vram_calibration;
        let graph = match self.graph.as_mut() {
            Some(g) => g,
            None => return (0, 0.0),
        };
        let hcs = match graph.hcs.as_mut() {
            Some(h) => h,
            None => return (0, 0.0),
        };

        // Always sync the reload stream before eviction to prevent races between
        // async DMA (reload stream) and prefill (default stream). Without this,
        // cuMemFree on soft_buf can free memory that has in-flight DMA, and PyTorch
        // may then reallocate the same address — causing ILLEGAL_ADDRESS when the
        // stale DMA writes to PyTorch's tensor.
        if let Some(ref stream) = hcs.soft_reload_stream {
            unsafe {
                let err = cuda_sys::lib().cuStreamSynchronize(stream.0);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    log::warn!("HCS soft evict: reload stream sync failed: {:?}", err);
                }
            }
        }

        // Cancel any pending async reload — free the reload chunks,
        // then fall through to normal eviction (current prompt may need more VRAM).
        if hcs.soft_reload_pending {
            // The pending entries were counted in soft_num_cached at queue time
            // but never activated into cache_fast/num_cached. Subtract them now.
            let unactivated = hcs.soft_reload_entries.len();
            if unactivated > 0 {
                hcs.soft_num_cached = hcs.soft_num_cached.saturating_sub(unactivated);
            }
            hcs.soft_reload_pending = false;
            hcs.soft_reload_entries.clear();
            if !hcs.soft_chunks.is_empty() {
                let extra_chunks = hcs.soft_chunks.len().saturating_sub(hcs.soft_chunks_loaded);
                if extra_chunks > 0 {
                    let freed_bytes = extra_chunks * hcs.soft_slots_per_chunk * hcs.soft_slot_size;
                    hcs.soft_chunks.truncate(hcs.soft_chunks_loaded);
                    hcs.vram_bytes = hcs.vram_bytes.saturating_sub(freed_bytes);
                    if stderr_debug_enabled() {
                        eprintln!("  \x1b[33mHCS soft: cancelled pending async reload ({} unactivated experts, {:.1} MB freed)\x1b[0m",
                            unactivated, freed_bytes as f64 / (1024.0 * 1024.0));
                    }
                }
            }
            // Fall through — the current prompt may need further eviction
        }

        if hcs.soft_chunks.is_empty() {
            return (0, 0.0);
        }
        // If soft chunks are allocated but empty (0 cached experts), still free them
        if hcs.soft_num_cached == 0 {
            let freed_bytes = hcs.soft_num_slots * hcs.soft_slot_size;
            if freed_bytes > 0 {
                hcs.soft_chunks.clear();
                hcs.soft_chunks_loaded = 0;
                hcs.soft_loaded = false;
                hcs.vram_bytes = hcs.vram_bytes.saturating_sub(freed_bytes);
                log::info!("HCS soft evict: freed empty soft chunks ({:.1} MB)",
                    freed_bytes as f64 / (1024.0 * 1024.0));
                return (0, freed_bytes as f64 / (1024.0 * 1024.0));
            }
            return (0, 0.0);
        }

        let capped_tokens = estimated_tokens.max(128).min(50000);

        // Measure actual free VRAM.
        let free_bytes = unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let err = cuda_sys::lib().cuMemGetInfo_v2(
                &mut free as *mut usize,
                &mut total as *mut usize);
            if err == cuda_sys::CUresult::CUDA_SUCCESS { free } else { 0 }
        };

        // Primary model: no-HCS prefill calibration tells us how much free VRAM
        // must remain while HCS is resident so the measured prefill transient still
        // leaves the configured safety margin.
        //
        // Fallback: if calibration is unavailable, use the older scratch estimate.
        let needed_bytes = if let Some(cal) = cal {
            cal.required_prefill_idle_free_mb(capped_tokens) as usize * 1024 * 1024
        } else if let Some((fixed_bytes, per_token_bytes)) = self.prefill_scratch_info {
            let scratch = fixed_bytes + per_token_bytes * capped_tokens;
            let safety_mb = hcs.safety_margin_mb.max(crate::gpu_prefill::PREFILL_SAFETY_MARGIN_MB);
            scratch + safety_mb * 1024 * 1024
        } else {
            usize::MAX
        };

        if free_bytes >= needed_bytes {
            if stderr_debug_enabled() {
                eprintln!(
                    "  \x1b[32mHCS soft: skip eviction — {:.0} MB free >= {:.0} MB prefill floor\x1b[0m",
                    free_bytes as f64 / (1024.0 * 1024.0),
                    needed_bytes as f64 / (1024.0 * 1024.0)
                );
            }
            log::info!(
                "HCS soft: skip eviction — {:.0} MB free >= {:.0} MB prefill floor for ~{} tokens",
                free_bytes as f64 / (1024.0 * 1024.0),
                needed_bytes as f64 / (1024.0 * 1024.0),
                capped_tokens,
            );
            return (0, 0.0);
        }

        // Proportional eviction: drop chunks from the tail (coldest first)
        // until we have enough free VRAM for scratch + safety.
        let t0 = std::time::Instant::now();
        let deficit = needed_bytes - free_bytes;
        let spc = hcs.soft_slots_per_chunk;
        let nep = hcs.num_experts_per_layer;

        let mut chunks_to_drop = 0usize;
        let mut freed_bytes = 0usize;
        let mut evicted = 0usize;

        // Walk chunks from the tail (coldest experts)
        let total_chunks = hcs.soft_chunks_loaded;
        for drop_idx in (0..total_chunks).rev() {
            if freed_bytes >= deficit {
                break;
            }
            // Compute chunk size from slot layout
            let slots_this = if drop_idx == hcs.soft_total_chunks - 1 {
                hcs.soft_num_slots - drop_idx * spc
            } else {
                spc
            };
            let chunk_bytes = slots_this * hcs.soft_slot_size;

            // Remove cache entries for experts in this chunk.
            // Use cache_fast_clear to also update GPU-side d_expert_ptrs table
            // (reverts to mapped fallback or zeros, preventing stale pointers).
            let slot_start = drop_idx * spc;
            let slot_end = std::cmp::min(slot_start + spc, hcs.soft_num_slots);
            for slot in slot_start..slot_end {
                if let Some((layer_idx, expert_idx)) = hcs.soft_slot_to_expert[slot] {
                    hcs.cache_fast_clear(layer_idx, expert_idx);
                    hcs.cache.remove(&(layer_idx, expert_idx));
                    evicted += 1;
                }
            }

            freed_bytes += chunk_bytes;
            chunks_to_drop += 1;
        }

        // Actually drop the chunks (from the end)
        let new_loaded = total_chunks - chunks_to_drop;
        hcs.soft_chunks.truncate(new_loaded);
        hcs.soft_chunks_loaded = new_loaded;
        hcs.num_cached -= evicted;
        hcs.soft_num_cached -= evicted;
        hcs.vram_bytes = hcs.vram_bytes.saturating_sub(freed_bytes);

        // Mark soft tier as not fully loaded so reload restores the dropped chunks
        if chunks_to_drop > 0 {
            hcs.soft_loaded = false;
        }

        // NOTE: per-layer CUDA graphs remain valid across eviction/reload.
        // Expert kernels dereference through d_batch_upload (fixed address),
        // which CPU populates with fresh pointers before each graph replay.
        // Eviction changes where experts live, but not the graph structure.

        let freed_mb = freed_bytes as f64 / (1024.0 * 1024.0);
        self.last_soft_evict_experts = evicted;
        self.last_soft_evict_freed_mb = freed_mb;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if stderr_debug_enabled() {
            eprintln!("  \x1b[33mHCS soft: evicted {} experts ({:.1} MB, {} of {} chunks dropped) in {:.1}ms for prefill (~{} tokens)\x1b[0m",
                evicted, freed_mb, chunks_to_drop, total_chunks, elapsed_ms, estimated_tokens);
        }
        log::info!("HCS soft: evicted {} experts ({:.1} MB, {} of {} chunks dropped) in {:.1}ms for prefill (~{} tokens)",
            evicted, freed_mb, chunks_to_drop, total_chunks, elapsed_ms, estimated_tokens);

        (evicted, freed_mb)
    }

    /// Reload soft-tier HCS experts after prefill completes.
    /// Uses pre-packed host chunk buffers for batch DMA: one cuMemcpyHtoD per chunk
    /// instead of 4 calls per expert (e.g. 22 calls instead of 55,000).
    /// Returns (loaded_count, reload_ms).
    pub fn hcs_reload_after_prefill(&mut self, actual_tokens: usize) -> (usize, f64) {
        let cal = self.vram_calibration;
        let graph = match self.graph.as_mut() {
            Some(g) => g,
            None => return (0, 0.0),
        };
        let hcs = match graph.hcs.as_mut() {
            Some(h) => h,
            None => return (0, 0.0),
        };

        if hcs.soft_ranking.is_empty() {
            return (0, 0.0);
        }

        let slot_size = hcs.soft_slot_size;
        if slot_size == 0 || hcs.soft_max_mb == 0 {
            return (0, 0.0);
        }

        let spc = hcs.soft_slots_per_chunk;
        let decode_budget_mb = cal.map(|cal| cal.decode_hcs_budget_mb(actual_tokens) as usize)
            .unwrap_or(hcs.hard_budget_mb + hcs.soft_max_mb);
        let idle_floor_mb = cal.map(|cal| cal.required_idle_free_mb(actual_tokens) as usize)
            .unwrap_or(hcs.safety_margin_mb);
        let target_soft_mb = decode_budget_mb
            .saturating_sub(hcs.hard_budget_mb)
            .min(hcs.soft_max_mb);
        let target_soft_bytes = target_soft_mb.saturating_mul(1024 * 1024);
        let target_soft_slots = (target_soft_bytes / slot_size).min(hcs.soft_num_slots);
        let target_chunks = if target_soft_slots == 0 {
            0
        } else {
            ((target_soft_slots + spc - 1) / spc).min(hcs.soft_total_chunks)
        };

        if hcs.soft_chunks_loaded > target_chunks {
            let (evicted, freed_bytes) = hcs.trim_soft_chunks_to(target_chunks);
            if evicted > 0 || freed_bytes > 0 {
                log::info!(
                    "HCS soft reload: trimmed to {} chunks for {} tokens (budget={} MB, evicted {} experts, freed {:.1} MB)",
                    target_chunks,
                    actual_tokens,
                    target_soft_mb,
                    evicted,
                    freed_bytes as f64 / (1024.0 * 1024.0),
                );
            }
        }

        let already_loaded = hcs.soft_chunks_loaded;
        if already_loaded >= target_chunks {
            hcs.soft_loaded = hcs.soft_chunks_loaded == hcs.soft_total_chunks;
            return (0, 0.0);
        }

        let t0 = std::time::Instant::now();
        let mut loaded = 0usize;
        let mut alloc_bytes = 0usize;

        // Reallocate and batch-DMA only the missing chunks
        for c in already_loaded..target_chunks {
            let slots_this_chunk = if c == target_chunks - 1 {
                hcs.soft_num_slots - c * spc
            } else {
                spc
            };
            let chunk_bytes = slots_this_chunk * slot_size;
            if let Ok((actual_free, _)) = cudarc::driver::result::mem_get_info() {
                let actual_free_mb = actual_free / (1024 * 1024);
                let chunk_mb = (chunk_bytes + (1024 * 1024) - 1) / (1024 * 1024);
                if actual_free_mb <= idle_floor_mb
                    || actual_free_mb.saturating_sub(chunk_mb) < idle_floor_mb
                {
                    log::warn!(
                        "HCS soft reload: stopping at chunk {} — free={} MB, chunk={} MB, idle floor={} MB",
                        c, actual_free_mb, chunk_mb, idle_floor_mb,
                    );
                    break;
                }
            }

            let chunk_buf = match self.device.alloc_zeros::<u8>(chunk_bytes) {
                Ok(buf) => buf,
                Err(e) => {
                    log::warn!("HCS soft reload: chunk {} alloc failed ({:.1} MB): {:?}",
                        c, chunk_bytes as f64 / (1024.0 * 1024.0), e);
                    break;
                }
            };
            if let Ok((actual_free, _)) = cudarc::driver::result::mem_get_info() {
                let actual_free_mb = actual_free / (1024 * 1024);
                if actual_free_mb < idle_floor_mb {
                    log::warn!(
                        "HCS soft reload: dropping chunk {} after alloc — free={} MB below idle floor={} MB",
                        c, actual_free_mb, idle_floor_mb,
                    );
                    drop(chunk_buf);
                    break;
                }
            }
            let chunk_base = *chunk_buf.device_ptr();

            // Batch DMA: one call per chunk from pre-packed host buffer
            if c < hcs.soft_host_chunks.len() {
                unsafe {
                    let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                        chunk_base,
                        hcs.soft_host_chunks[c].as_ptr() as *const std::ffi::c_void,
                        chunk_bytes,
                    );
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        log::warn!("HCS soft reload: chunk {} batch DMA failed: {:?}", c, err);
                        break;
                    }
                }
            }

            // Rebuild cache entries from GPU pointers
            let slot_start = c * spc;
            let slot_end = slot_start + slots_this_chunk;
            for slot in slot_start..slot_end {
                if slot >= hcs.soft_ranking.len() {
                    break;
                }
                let (layer_idx, expert_idx) = hcs.soft_ranking[slot];
                let moe = match graph.moe_layers.get(layer_idx).and_then(|m| m.as_ref()) {
                    Some(m) => m,
                    None => continue,
                };
                if expert_idx >= moe.experts.len() {
                    continue;
                }

                let expert = &moe.experts[expert_idx];
                let offset_in_chunk = slot - slot_start;
                let dst = chunk_base + (offset_in_chunk as u64 * slot_size as u64);

                let w13p_off = 0u64;
                let w13s_off = expert.w13_packed_bytes as u64;
                let w2p_off = w13s_off + expert.w13_scales_bytes as u64;
                let w2s_off = w2p_off + expert.w2_packed_bytes as u64;

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
                    pool_slot: None,
                };
                hcs.cache_fast_set(layer_idx, expert_idx, &entry);
                hcs.cache.insert((layer_idx, expert_idx), entry);
                hcs.soft_slot_to_expert[slot] = Some((layer_idx, expert_idx));
                loaded += 1;
            }

            alloc_bytes += chunk_bytes;
            hcs.soft_chunks.push(chunk_buf);
            hcs.soft_chunks_loaded += 1;
        }

        hcs.soft_num_cached += loaded;
        hcs.soft_loaded = hcs.soft_chunks_loaded == hcs.soft_total_chunks;
        hcs.num_cached += loaded;
        hcs.vram_bytes += alloc_bytes;

        let alloc_mb = alloc_bytes as f64 / (1024.0 * 1024.0);
        self.last_soft_reload_queued = loaded;
        self.last_soft_reload_alloc_mb = alloc_mb;
        self.last_soft_reload_activated = loaded;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let chunks_reloaded = hcs.soft_chunks_loaded - already_loaded;
        eprintln!("  \x1b[32mHCS soft: reloaded {} experts ({:.1} MB, {}/{} chunks, batch DMA) in {:.1}ms\x1b[0m",
            loaded, alloc_mb, chunks_reloaded, target_chunks, elapsed_ms);
        log::info!("HCS soft: reloaded {} experts ({:.1} MB, {}/{} chunks, batch DMA) in {:.1}ms",
            loaded, alloc_mb, chunks_reloaded, target_chunks, elapsed_ms);

        hcs.log_hcs_post_reload();

        (loaded, elapsed_ms)
    }

    /// Async version of hcs_reload_after_prefill.
    /// Queues DMA only for dropped chunks on a dedicated CUDA stream and returns immediately.
    /// Call hcs_check_soft_reload_complete() each decode step to activate
    /// the soft tier once all DMA finishes.
    /// Returns (num_experts_queued, alloc_mb).
    pub fn hcs_reload_after_prefill_async(&mut self, actual_tokens: usize) -> (usize, f64) {
        let cal = self.vram_calibration;
        let graph = match self.graph.as_mut() {
            Some(g) => g,
            None => return (0, 0.0),
        };
        let hcs = match graph.hcs.as_mut() {
            Some(h) => h,
            None => return (0, 0.0),
        };

        if hcs.soft_ranking.is_empty() || hcs.soft_reload_pending {
            return (0, 0.0);
        }

        let slot_size = hcs.soft_slot_size;
        if slot_size == 0 || hcs.soft_max_mb == 0 {
            return (0, 0.0);
        }

        let spc = hcs.soft_slots_per_chunk;
        let decode_budget_mb = cal.map(|cal| cal.decode_hcs_budget_mb(actual_tokens) as usize)
            .unwrap_or(hcs.hard_budget_mb + hcs.soft_max_mb);
        let idle_floor_mb = cal.map(|cal| cal.required_idle_free_mb(actual_tokens) as usize)
            .unwrap_or(hcs.safety_margin_mb);
        let target_soft_mb = decode_budget_mb
            .saturating_sub(hcs.hard_budget_mb)
            .min(hcs.soft_max_mb);
        let target_soft_bytes = target_soft_mb.saturating_mul(1024 * 1024);
        let target_soft_slots = (target_soft_bytes / slot_size).min(hcs.soft_num_slots);
        let target_chunks = if target_soft_slots == 0 {
            0
        } else {
            ((target_soft_slots + spc - 1) / spc).min(hcs.soft_total_chunks)
        };

        if hcs.soft_chunks_loaded > target_chunks {
            let (evicted, freed_bytes) = hcs.trim_soft_chunks_to(target_chunks);
            if evicted > 0 || freed_bytes > 0 {
                log::info!(
                    "HCS soft async reload: trimmed to {} chunks for {} tokens (budget={} MB, evicted {} experts, freed {:.1} MB)",
                    target_chunks,
                    actual_tokens,
                    target_soft_mb,
                    evicted,
                    freed_bytes as f64 / (1024.0 * 1024.0),
                );
            }
        }

        let already_loaded = hcs.soft_chunks_loaded;
        if already_loaded >= target_chunks {
            hcs.soft_loaded = hcs.soft_chunks_loaded == hcs.soft_total_chunks;
            return (0, 0.0);
        }

        let t0 = std::time::Instant::now();

        // Create or reuse the reload stream and event
        if hcs.soft_reload_stream.is_none() {
            let mut s: cuda_sys::CUstream = std::ptr::null_mut();
            unsafe {
                let err = cuda_sys::lib().cuStreamCreate(
                    &mut s,
                    cuda_sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                );
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    log::warn!("HCS soft async reload: cuStreamCreate failed: {:?}", err);
                    return (0, 0.0);
                }
            }
            hcs.soft_reload_stream = Some(CudaStream(s));
        }
        if hcs.soft_reload_event.is_none() {
            let mut e: cuda_sys::CUevent = std::ptr::null_mut();
            unsafe {
                let err = cuda_sys::lib().cuEventCreate(
                    &mut e,
                    cuda_sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
                );
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    log::warn!("HCS soft async reload: cuEventCreate failed: {:?}", err);
                    return (0, 0.0);
                }
            }
            hcs.soft_reload_event = Some(CudaEvent(e));
        }
        let reload_stream = hcs.soft_reload_stream.as_ref().unwrap().0;
        let reload_event = hcs.soft_reload_event.as_ref().unwrap().0;

        // Allocate and queue batch DMA for missing chunks (already_loaded..target_chunks)
        // One cuMemcpyHtoDAsync per chunk from pre-packed host buffer instead of
        // 4 calls per expert (e.g. 22 calls instead of 55,000).
        let mut queued = 0usize;
        let mut alloc_bytes = 0usize;
        let mut pending_entries: Vec<(usize, usize, HcsCacheEntry)> = Vec::new();

        for c in already_loaded..target_chunks {
            let slots_this_chunk = if c == target_chunks - 1 {
                hcs.soft_num_slots - c * spc
            } else {
                spc
            };
            let chunk_bytes = slots_this_chunk * slot_size;
            if let Ok((actual_free, _)) = cudarc::driver::result::mem_get_info() {
                let actual_free_mb = actual_free / (1024 * 1024);
                let chunk_mb = (chunk_bytes + (1024 * 1024) - 1) / (1024 * 1024);
                if actual_free_mb <= idle_floor_mb
                    || actual_free_mb.saturating_sub(chunk_mb) < idle_floor_mb
                {
                    log::warn!(
                        "HCS soft async reload: stopping at chunk {} — free={} MB, chunk={} MB, idle floor={} MB",
                        c, actual_free_mb, chunk_mb, idle_floor_mb,
                    );
                    break;
                }
            }

            let chunk_buf = match unsafe { self.device.alloc::<u8>(chunk_bytes) } {
                Ok(buf) => buf,
                Err(e) => {
                    log::warn!("HCS soft async reload: chunk {} alloc failed ({:.1} MB): {:?}",
                        c, chunk_bytes as f64 / (1024.0 * 1024.0), e);
                    break;
                }
            };
            if let Ok((actual_free, _)) = cudarc::driver::result::mem_get_info() {
                let actual_free_mb = actual_free / (1024 * 1024);
                if actual_free_mb < idle_floor_mb {
                    log::warn!(
                        "HCS soft async reload: dropping chunk {} after alloc — free={} MB below idle floor={} MB",
                        c, actual_free_mb, idle_floor_mb,
                    );
                    drop(chunk_buf);
                    break;
                }
            }
            let chunk_base = *chunk_buf.device_ptr();

            // Batch async DMA: one call per chunk from pre-packed host buffer
            if c < hcs.soft_host_chunks.len() {
                unsafe {
                    let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        chunk_base,
                        hcs.soft_host_chunks[c].as_ptr() as *const std::ffi::c_void,
                        chunk_bytes,
                        reload_stream,
                    );
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        log::warn!("HCS soft async reload: chunk {} batch DMA failed: {:?}", c, err);
                        break;
                    }
                }
            }

            // Build pending cache entries (activated after DMA completes)
            let slot_start = c * spc;
            let slot_end = slot_start + slots_this_chunk;
            for slot in slot_start..slot_end {
                if slot >= hcs.soft_ranking.len() {
                    break;
                }
                let (layer_idx, expert_idx) = hcs.soft_ranking[slot];
                let moe = match graph.moe_layers.get(layer_idx).and_then(|m| m.as_ref()) {
                    Some(m) => m,
                    None => continue,
                };
                if expert_idx >= moe.experts.len() {
                    continue;
                }

                let expert = &moe.experts[expert_idx];
                let offset_in_chunk = slot - slot_start;
                let dst = chunk_base + (offset_in_chunk as u64 * slot_size as u64);

                let w13p_off = 0u64;
                let w13s_off = expert.w13_packed_bytes as u64;
                let w2p_off = w13s_off + expert.w13_scales_bytes as u64;
                let w2s_off = w2p_off + expert.w2_packed_bytes as u64;

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
                    pool_slot: None,
                };
                pending_entries.push((layer_idx, expert_idx, entry));
                hcs.soft_slot_to_expert[slot] = Some((layer_idx, expert_idx));
                queued += 1;
            }

            alloc_bytes += chunk_bytes;
            hcs.soft_chunks.push(chunk_buf);
        }

        // Record event after all DMA
        unsafe {
            let err = cuda_sys::lib().cuEventRecord(reload_event, reload_stream);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                log::warn!("HCS soft async reload: cuEventRecord failed: {:?}", err);
            }
        }

        // Store pending state (but don't activate cache entries yet)
        hcs.soft_num_cached += queued;
        hcs.soft_loaded = false;
        hcs.soft_reload_pending = true;
        hcs.soft_reload_entries = pending_entries;
        hcs.vram_bytes += alloc_bytes;

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let alloc_mb = alloc_bytes as f64 / (1024.0 * 1024.0);
        self.last_soft_reload_queued = queued;
        self.last_soft_reload_alloc_mb = alloc_mb;
        self.last_soft_reload_activated = 0;
        let chunks_queued = hcs.soft_chunks.len() - already_loaded;
        if stderr_debug_enabled() {
            eprintln!("  \x1b[36mHCS soft: async reload queued {} experts ({:.1} MB, {}/{} chunks, batch DMA) in {:.1}ms\x1b[0m",
                queued, alloc_mb, chunks_queued, target_chunks, elapsed_ms);
        }
        log::info!("HCS soft: async reload queued {} experts ({:.1} MB, {}/{} chunks, batch DMA) in {:.1}ms",
            queued, alloc_mb, chunks_queued, target_chunks, elapsed_ms);

        (queued, alloc_mb)
    }

    /// Check if async soft-tier reload DMA has completed.
    /// If so, activate all cache entries and mark soft tier as loaded.
    /// Returns true if the soft tier just became available this call.
    pub fn hcs_check_soft_reload_complete(&mut self) -> bool {
        let graph = match self.graph.as_mut() {
            Some(g) => g,
            None => return false,
        };
        let hcs = match graph.hcs.as_mut() {
            Some(h) => h,
            None => return false,
        };

        if !hcs.soft_reload_pending {
            return false;
        }

        let event = match hcs.soft_reload_event {
            Some(ref e) => e.0,
            None => return false,
        };

        // Non-blocking check: is the DMA complete?
        let result = unsafe { cuda_sys::lib().cuEventQuery(event) };
        if result != cuda_sys::CUresult::CUDA_SUCCESS {
            // CUDA_ERROR_NOT_READY means DMA still in flight
            return false;
        }

        // DMA complete — activate all cache entries
        let entries = std::mem::take(&mut hcs.soft_reload_entries);
        let activated = entries.len();
        for (layer_idx, expert_idx, entry) in entries {
            hcs.cache_fast_set(layer_idx, expert_idx, &entry);
            hcs.cache.insert((layer_idx, expert_idx), entry);
        }
        hcs.soft_chunks_loaded = hcs.soft_chunks.len();
        hcs.soft_loaded = true;
        hcs.soft_reload_pending = false;
        hcs.num_cached += activated;
        self.last_soft_reload_activated = activated;

        if stderr_debug_enabled() {
            eprintln!("  \x1b[32mHCS soft: async reload complete — {} experts activated ({}/{} chunks)\x1b[0m",
                activated, hcs.soft_chunks_loaded, hcs.soft_total_chunks);
        }
        log::info!("HCS soft: async reload complete — {} experts activated ({}/{} chunks)",
            activated, hcs.soft_chunks_loaded, hcs.soft_total_chunks);

        hcs.log_hcs_post_reload();

        true
    }

    /// Synchronize pending soft-tier reload: wait for all DMA to complete,
    /// activate cache entries, and return the real DMA wall-clock time in ms.
    /// Used by --sync-hcs-reload to get clean decode measurements.
    /// Returns (activated_count, real_dma_ms). Returns (0, 0.0) if no reload pending.
    pub fn hcs_sync_soft_reload(&mut self) -> (usize, f64) {
        let graph = match self.graph.as_mut() {
            Some(g) => g,
            None => return (0, 0.0),
        };
        let hcs = match graph.hcs.as_mut() {
            Some(h) => h,
            None => return (0, 0.0),
        };

        if !hcs.soft_reload_pending {
            return (0, 0.0);
        }

        let stream = match hcs.soft_reload_stream {
            Some(ref s) => s.0,
            None => return (0, 0.0),
        };

        // Synchronize: wait for all DMA on the reload stream to finish
        let t0 = std::time::Instant::now();
        unsafe {
            let err = cuda_sys::lib().cuStreamSynchronize(stream);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                log::warn!("HCS sync reload: cuStreamSynchronize failed: {:?}", err);
            }
        }
        let real_dma_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Activate all cache entries (same as hcs_check_soft_reload_complete)
        let entries = std::mem::take(&mut hcs.soft_reload_entries);
        let activated = entries.len();
        for (layer_idx, expert_idx, entry) in entries {
            hcs.cache_fast_set(layer_idx, expert_idx, &entry);
            hcs.cache.insert((layer_idx, expert_idx), entry);
        }
        hcs.soft_chunks_loaded = hcs.soft_chunks.len();
        hcs.soft_loaded = true;
        hcs.soft_reload_pending = false;
        hcs.num_cached += activated;
        self.last_soft_reload_activated = activated;

        if stderr_debug_enabled() {
            eprintln!("  \x1b[32mHCS soft: sync reload complete — {} experts ({}/{} chunks) in {:.1}ms real DMA\x1b[0m",
                activated, hcs.soft_chunks_loaded, hcs.soft_total_chunks, real_dma_ms);
        }
        log::info!("HCS soft: sync reload complete — {} experts ({}/{} chunks) in {:.1}ms real DMA",
            activated, hcs.soft_chunks_loaded, hcs.soft_total_chunks, real_dma_ms);

        hcs.log_hcs_post_reload();

        (activated, real_dma_ms)
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
        logprobs_top_n: usize,
        mut on_token: F,
    ) -> usize
    where
        F: FnMut(usize, &str, Option<&str>, Option<&[(u32, f32)]>) -> bool,
    {
        use std::time::Instant;

        if stderr_debug_enabled() {
            eprintln!("[krasis] gpu_generate_stream called: first_token={}, start_pos={}, max_tokens={}", first_token, start_position, max_tokens);
        }

        let decode_diag = std::env::var("KRASIS_DECODE_DIAG").map(|v| v == "1").unwrap_or(false);
        let prefill_debug = prefill_debug_enabled();

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

        if prefill_debug {
            let no_graph = std::env::var("KRASIS_NO_GRAPH").map(|v| v != "0").unwrap_or(false);
            if let Some(graph) = self.graph.as_ref() {
                let (mapped_reads_available, hcs_cached, hcs_soft_cached, hcs_soft_loaded) =
                    if let Some(hcs) = graph.hcs.as_ref() {
                        (
                            hcs.mapped_reads_available,
                            hcs.num_cached,
                            hcs.soft_num_cached,
                            hcs.soft_loaded,
                        )
                    } else {
                        (false, 0, 0, false)
                    };
                eprintln!(
                    "[PREFILL-DEBUG] decode request start_pos={} max_tokens={} no_graph={} graphs_valid={} graphs_ever_captured={} gpu_route_sync={} mapped_reads_active={} mapped_reads_available={} hcs_cached={} hcs_soft_cached={} hcs_soft_loaded={} timing={}",
                    start_position,
                    max_tokens,
                    no_graph,
                    graph.per_layer_graphs_valid,
                    graph.graphs_ever_captured,
                    graph.gpu_route_sync,
                    graph.mapped_reads_active,
                    mapped_reads_available,
                    hcs_cached,
                    hcs_soft_cached,
                    hcs_soft_loaded,
                    graph.timing_enabled,
                );
            }
        }

        // Decode diagnostic: check LA recur state and conv state at decode start
        if decode_diag {
            if let Some(ref graph) = self.graph {
                for (li, layer) in graph.layers.iter().enumerate() {
                    if li >= 3 { break; } // Only check first 3 layers
                    if let GpuAttnConfig::LinearAttention { recur_state_ptr, conv_state_ptr, nv, dk, dv, conv_dim, kernel_dim, .. } = &layer.attn {
                        // Download recur state and compute L2 norm of first head
                        let head_size = dk * dv; // 128*128=16384 for QCN
                        let mut state_buf = vec![0.0f32; head_size];
                        unsafe {
                            let _ = cuda_sys::lib().cuMemcpyDtoH_v2(
                                state_buf.as_mut_ptr() as *mut std::ffi::c_void,
                                *recur_state_ptr, head_size * 4);
                        }
                        let head0_norm: f32 = state_buf.iter()
                            .map(|x| x * x).sum::<f32>().sqrt();
                        let any_nonzero = state_buf.iter().any(|&x| x != 0.0);
                        eprintln!("[DECODE-DIAG] L{:02} recur_state: head0_norm={:.6} any_nonzero={} vals[0..4]=[{:.6},{:.6},{:.6},{:.6}]",
                            li, head0_norm, any_nonzero,
                            state_buf[0], state_buf[1], state_buf[2], state_buf[3]);

                        // Check conv state
                        let conv_size = conv_dim * kernel_dim;
                        let mut conv_buf = vec![0.0f32; conv_size.min(256)];
                        let cdl = conv_buf.len();
                        unsafe {
                            let _ = cuda_sys::lib().cuMemcpyDtoH_v2(
                                conv_buf.as_mut_ptr() as *mut std::ffi::c_void,
                                *conv_state_ptr, cdl * 4);
                        }
                        let conv_any_nonzero = conv_buf.iter().any(|&x| x != 0.0);
                        let conv_norm: f32 = conv_buf.iter().map(|x| x * x).sum::<f32>().sqrt();
                        eprintln!("[DECODE-DIAG] L{:02} conv_state: norm={:.6} any_nonzero={}",
                            li, conv_norm, conv_any_nonzero);
                    }
                }
            }
        }

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

        // CUDA graph reuse: after the first request, KV cache and LA state buffer
        // addresses are stable (KV allocated once at model load; LA Rust-side FP32
        // buffers use copy_() in-place). Verify pointers match capture-time snapshot.
        // If stable, reuse graphs (skip ~30-40ms recapture overhead per request).
        // If pointers moved (shouldn't happen), fall back to invalidate + recapture.
        if self.verify_graph_pointers() {
            if stderr_debug_enabled() {
                eprintln!("[krasis] CUDA graphs reused from previous request (ptrs verified stable)");
            }
        } else {
            let was_captured = self.graph.as_ref()
                .map(|g| g.graphs_ever_captured).unwrap_or(false);
            if was_captured && stderr_debug_enabled() {
                eprintln!("[krasis] CUDA graph ptrs changed — invalidating, will recapture");
            }
            self.invalidate_cuda_graph();
        }

        // Reset per-prompt timing accumulators
        {
            let g = self.graph.as_mut().unwrap();
            g.validation_decode_steps = 0;
            g.validation_per_layer_steps = 0;
            g.validation_ungraphed_steps = 0;
            g.dma_bytes_total = 0;
            g.dma_call_count = 0;
            g.dma_cold_experts = 0;
            g.dma_hcs_experts = 0;
            g.validation_decode_start_num_cached = 0;
            g.validation_decode_start_soft_num_cached = 0;
            g.validation_decode_start_hard_num_cached = 0;
            g.validation_decode_start_dupes = 0;
            g.validation_decode_start_soft_dupes = 0;
            g.validation_decode_start_hash.clear();
            g.validation_decode_start_resident.clear();
            g.validation_decode_start_hcs_file.clear();
            g.validation_decode_start_slots.clear();
            g.validation_decode_start_slots_file.clear();
            g.validation_decode_cold_hist.clear();
            g.validation_decode_cold_events.clear();
            g.validation_decode_cold_file.clear();
            g.validation_decode_cold_events_file.clear();
            if g.timing_enabled {
                g.timing_step_count = 0;
                g.t_total = 0.0; g.t_norm = 0.0; g.t_attn = 0.0;
                g.t_route = 0.0; g.t_expert_dma = 0.0; g.t_expert_compute = 0.0;
                g.t_shared = 0.0; g.t_dense_mlp = 0.0; g.t_lm_head = 0.0;
                g.t_moe_route_sync = 0.0; g.t_moe_expert_loop = 0.0;
                g.t_moe_shared = 0.0; g.t_moe_overhead = 0.0;
                g.t_moe_gate_gemv = 0.0; g.t_moe_d2h_topk = 0.0;
                g.t_moe_apfl = 0.0; g.t_moe_padding_setup = 0.0; g.t_moe_d2d_copy = 0.0;
                g.t_moe_accum = 0.0;
                g.t_attn_la = 0.0; g.t_attn_gqa = 0.0;
                g.t_la_proj = 0.0; g.t_la_conv = 0.0;
                g.t_la_recur = 0.0; g.t_la_out = 0.0;
                g.t_gqa_proj = 0.0; g.t_gqa_attn = 0.0; g.t_gqa_out = 0.0;
                g.t_expert_w13 = 0.0; g.t_expert_silu_w2 = 0.0;
                g.t_dma_expert_wait = 0.0; g.t_dma_expert_compute = 0.0;
            }
        }

        if std::env::var("KRASIS_STATE_VALIDATION").map(|v| v != "0").unwrap_or(false)
            || std::env::var("KRASIS_CONFIG_VALIDATION").map(|v| v != "0").unwrap_or(false)
        {
            self.validation_capture_decode_start(start_position);
        }

        let decode_start = Instant::now();
        let mut next_token = first_token;
        let mut generated = 0usize;
        let mut seen_tokens: std::collections::HashSet<usize> = std::collections::HashSet::new();
        seen_tokens.insert(first_token);

        // Streaming detokenizer: buffers incomplete UTF-8 byte sequences
        // (e.g. emojis split across multiple BPE tokens) and emits only
        // when the decoded text is complete.
        let mut detok = crate::server::StreamDetokenizer::new(tokenizer);

        #[cfg(feature = "gpu-debug")]
        let debug_logits = std::env::var("KRASIS_DEBUG_LOGITS").ok()
            .map(|v| v.parse::<usize>().unwrap_or(0)).unwrap_or(0);

        // ── Speculative decode state ──
        let use_speculative = self.draft.is_some();
        let draft_k = self.draft_k;
        let draft_context_window = self.draft_context_window;
        let mut spec_accepted: u64 = 0;
        let mut spec_rejected: u64 = 0;
        let mut spec_rounds: u64 = 0;
        let mut spec_draft_time: f64 = 0.0;
        let mut spec_verify_time: f64 = 0.0;
        let mut spec_save_time: f64 = 0.0;
        let mut spec_failfast_count: u64 = 0;
        let mut spec_tokens_saved: u64 = 0;

        // Reset draft model KV cache for speculative decode
        if use_speculative && start_position > 0 {
            if let Some(ref mut draft) = self.draft {
                draft.reset_kv();
            }
        }

        // Track min VRAM free during decode (3 snapshots: before loop, after loop, and post-decode query)
        let mut min_vram_free_bytes: usize = usize::MAX;

        // Lightweight decode profiling (KRASIS_DECODE_PROFILE=1)
        // Zero overhead when disabled — just a bool check per token.
        let profile_enabled = std::env::var("KRASIS_DECODE_PROFILE").map(|v| v == "1").unwrap_or(false);
        let profile_interval = 25usize;
        let mut profile_last_step = 0usize;
        let mut profile_last_time = Instant::now();
        let mut profile_cold_total = 0u64;
        let mut profile_hcs_total = 0u64;

        // Initialize CUDA graph infrastructure (GPU-side scalar buffers + dummy expert)
        {
            let needs_init = self.graph.as_ref().map(|g| g.d_graph_token_id.is_none()).unwrap_or(false);
            if needs_init {
                match self.init_cuda_graph_buffers() {
                    Ok(()) => {
                        if stderr_debug_enabled() { eprintln!("[krasis] CUDA graph buffers initialized"); }
                    }
                    Err(e) => eprintln!("[krasis] CUDA graph init failed: {}", e),
                }
            }
        }

        let sample_min_vram = |min_vram_free_bytes: &mut usize| {
            let mut free: usize = 0;
            let mut _total: usize = 0;
            unsafe { let _ = cuda_sys::lib().cuMemGetInfo_v2(&mut free, &mut _total); }
            if free < *min_vram_free_bytes {
                *min_vram_free_bytes = free;
            }
        };

        // Snapshot VRAM before decode loop
        sample_min_vram(&mut min_vram_free_bytes);

        let mut step = 0usize;
        while step < max_tokens {
            let pos = start_position + step;

            // Check if async soft-tier reload has completed
            self.hcs_check_soft_reload_complete();

            if use_speculative {
                // ── Phase 2: Batched Speculative Decode ──
                // Draft generates K tokens, then target verifies all in one batched pass.
                // The "real" decode step is folded into the batch as position 0.

                // 1. Generate draft tokens
                let t_draft = Instant::now();
                let (draft_tokens, draft_pos_before) = if let Some(ref mut draft) = self.draft {
                    let dp = draft.kv_pos();
                    if let Err(e) = draft.forward(&self.device, &self.blas, next_token, dp) {
                        log::warn!("speculative: draft forward failed: {}", e);
                        (Vec::new(), dp)
                    } else {
                        let mut draft_pred = 0usize;
                        let mut best_val = f32::NEG_INFINITY;
                        for j in 0..draft.h_logits.len() {
                            if draft.h_logits[j] > best_val {
                                best_val = draft.h_logits[j];
                                draft_pred = j;
                            }
                        }
                        let mut tokens = vec![draft_pred];
                        if draft_k > 1 {
                            match draft.generate_draft(&self.device, &self.blas, draft_pred, dp + 1, draft_k - 1) {
                                Ok(more) => tokens.extend(more),
                                Err(e) => log::warn!("speculative: draft gen failed: {}", e),
                            }
                        }
                        (tokens, dp)
                    }
                } else {
                    (Vec::new(), 0)
                };

                let draft_elapsed = t_draft.elapsed().as_secs_f64();
                spec_draft_time += draft_elapsed;
                spec_rounds += 1;

                if spec_rounds <= 3 && !draft_tokens.is_empty()
                    && std::env::var("KRASIS_SPEC_DEBUG").is_ok()
                {
                    let draft_strs: Vec<String> = draft_tokens.iter()
                        .map(|&t| format!("{}", t)).collect();
                    log::info!("spec round {}: next_token={}, draft=[{}], dp={}",
                        spec_rounds, next_token, draft_strs.join(","), draft_pos_before);
                }

                // 2. If draft failed, fall back to single-token decode
                if draft_tokens.is_empty() {
                    if let Err(e) = self.gpu_decode_step(next_token, pos) {
                        log::error!("gpu_generate_stream: decode_step error: {}", e);
                        break;
                    }
                    let (target_pred, token_logprobs) = {
                        let logits = &mut self.graph.as_mut().unwrap().h_logits;
                        if presence_penalty != 0.0 {
                            for &tok in &seen_tokens {
                                if tok < vocab_size { logits[tok] -= presence_penalty; }
                            }
                        }
                        for &tok in &self.suppress_tokens {
                            if tok < vocab_size { logits[tok] = f32::NEG_INFINITY; }
                        }
                        let think_end_active = self.think_end_token_for_suppress.is_some() && !self.think_end_seen;
                        let think_within_budget = self.think_suppress_budget == 0
                            || self.think_suppress_count < self.think_suppress_budget;
                        if generated < self.min_new_tokens || (think_end_active && think_within_budget) {
                            for &tok in &self.stop_token_ids_for_suppress {
                                if tok < vocab_size { logits[tok] = f32::NEG_INFINITY; }
                            }
                        }
                        // Force-inject </think> when thinking budget is exhausted
                        let target_pred = if think_end_active && !think_within_budget {
                            if let Some(te) = self.think_end_token_for_suppress {
                                eprintln!("[krasis] Thinking budget exhausted ({} tokens) — force-injecting </think>",
                                    self.think_suppress_count);
                                te
                            } else {
                                crate::decode::sample_from_logits_pub(
                                    logits, vocab_size, temperature, top_k, top_p, &mut rng_next)
                            }
                        } else {
                            crate::decode::sample_from_logits_pub(
                                logits, vocab_size, temperature, top_k, top_p, &mut rng_next)
                        };
                        let token_logprobs = if logprobs_top_n > 0 {
                            Some(crate::decode::extract_top_logprobs(logits, vocab_size, logprobs_top_n))
                        } else { None };
                        (target_pred, token_logprobs)
                    };
                    self.notify_token_generated(target_pred);
                    seen_tokens.insert(target_pred);
                    generated += 1;
                    step += 1;
                    next_token = target_pred;
                    let mut text = detok.add(target_pred as u32);
                    let finish_reason = if stop_set.contains(&target_pred) { Some("stop") }
                        else if generated >= max_tokens { Some("length") }
                        else { None };
                    if finish_reason.is_some() { text.push_str(&detok.flush()); }
                    let finished = finish_reason.is_some();
                    let cont = on_token(target_pred, &text, finish_reason, token_logprobs.as_deref());
                    if finished || !cont { break; }
                    continue;
                }

                // 3. Build batch: [next_token, D0, D1, ..., D_{K-1}]
                let batch_size = 1 + draft_tokens.len();
                let mut batch_tokens: Vec<usize> = Vec::with_capacity(batch_size);
                let mut batch_positions: Vec<usize> = Vec::with_capacity(batch_size);
                batch_tokens.push(next_token);
                batch_positions.push(pos);
                for (i, &dt) in draft_tokens.iter().enumerate() {
                    batch_tokens.push(dt);
                    batch_positions.push(pos + 1 + i);
                }

                // 4. Save LA states before batched decode
                let t_save = Instant::now();
                if let Err(e) = self.save_la_states() {
                    log::warn!("speculative: save_la_states failed: {}", e);
                }
                let save_ms = t_save.elapsed().as_secs_f64() * 1000.0;

                // 5. Batched target model decode — all tokens through all layers,
                //    expert union DMA'd once per MoE layer.
                //    Returns valid_positions: may be < batch_size if fail-fast triggered.
                let t_verify = Instant::now();
                let valid_positions = match self.gpu_decode_step_batched(&batch_tokens, &batch_positions) {
                    Ok(vp) => vp,
                    Err(e) => {
                        log::error!("speculative: batched decode error: {}", e);
                        let _ = self.restore_la_states();
                        break;
                    }
                };
                let verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0;
                if valid_positions < batch_size {
                    spec_failfast_count += 1;
                    spec_tokens_saved += (batch_size - valid_positions) as u64;
                }
                spec_verify_time += verify_ms;
                spec_save_time += save_ms;

                // 6. Extract target's prediction from position 0 (always valid)
                {
                    let graph = self.graph.as_mut().unwrap();
                    if presence_penalty != 0.0 {
                        for &tok in &seen_tokens {
                            if tok < vocab_size {
                                graph.h_batch_logits[tok] -= presence_penalty;
                            }
                        }
                    }
                }
                {
                    let batch_logits = &mut self.graph.as_mut().unwrap().h_batch_logits;
                    for &tok in &self.suppress_tokens {
                        if tok < vocab_size { batch_logits[tok] = f32::NEG_INFINITY; }
                    }
                }
                let target_pred = crate::decode::sample_from_logits_pub(
                    &mut self.graph.as_mut().unwrap().h_batch_logits[..vocab_size],
                    vocab_size, temperature, top_k, top_p, &mut rng_next);

                // 7. Emit target_pred (always produced — this is the "real" decode output)
                seen_tokens.insert(target_pred);
                generated += 1;
                step += 1;
                next_token = target_pred;

                let mut stopped = false;
                {
                    let mut text = detok.add(target_pred as u32);
                    let finish_reason = if stop_set.contains(&target_pred) { Some("stop") }
                        else if generated >= max_tokens { Some("length") }
                        else { None };
                    if finish_reason.is_some() { text.push_str(&detok.flush()); }
                    let finished = finish_reason.is_some();
                    let spec_logprobs = if logprobs_top_n > 0 {
                        let bl = &self.graph.as_ref().unwrap().h_batch_logits;
                        Some(crate::decode::extract_top_logprobs(&bl[..vocab_size], vocab_size, logprobs_top_n))
                    } else { None };
                    let cont = on_token(target_pred, &text, finish_reason, spec_logprobs.as_deref());
                    if finished || !cont {
                        let _ = self.restore_la_states();
                        if let Some(ref mut draft) = self.draft {
                            draft.rollback_kv(draft_pos_before + 1);
                        }
                        break;
                    }
                }

                // 8. Check acceptance: does target_pred match D0?
                //    valid_positions limits how many draft tokens we can verify.
                //    Position 0 logits verify draft_tokens[0].
                //    Position i logits (i < valid_positions) verify draft_tokens[i].
                let max_verifiable = draft_tokens.len().min(valid_positions);
                let mut accepted_in_round = 0usize;
                if target_pred == draft_tokens[0] {
                    accepted_in_round = 1;

                    // Check subsequent draft tokens against target's predictions
                    for i in 1..max_verifiable {
                        if step >= max_tokens { break; }

                        let logit_offset = i * vocab_size;
                        let batch_logits = &self.graph.as_ref().unwrap().h_batch_logits;
                        let mut target_argmax = 0usize;
                        let mut best_val = f32::NEG_INFINITY;
                        for j in 0..vocab_size {
                            if batch_logits[logit_offset + j] > best_val {
                                best_val = batch_logits[logit_offset + j];
                                target_argmax = j;
                            }
                        }

                        if target_argmax == draft_tokens[i] {
                            accepted_in_round += 1;
                            seen_tokens.insert(draft_tokens[i]);
                            generated += 1;
                            step += 1;
                            next_token = draft_tokens[i];

                            let mut text = detok.add(draft_tokens[i] as u32);
                            let finish_reason = if stop_set.contains(&draft_tokens[i]) { Some("stop") }
                                else if generated >= max_tokens { Some("length") }
                                else { None };
                            if finish_reason.is_some() { text.push_str(&detok.flush()); }
                            let finished = finish_reason.is_some();
                            let draft_logprobs = if logprobs_top_n > 0 {
                                let bl = &self.graph.as_ref().unwrap().h_batch_logits;
                                let offset = i * vocab_size;
                                Some(crate::decode::extract_top_logprobs(&bl[offset..offset + vocab_size], vocab_size, logprobs_top_n))
                            } else { None };
                            let cont = on_token(draft_tokens[i], &text, finish_reason, draft_logprobs.as_deref());
                            if finished || !cont { stopped = true; break; }
                        } else {
                            // Reject: use target's prediction at this position
                            next_token = target_argmax;
                            seen_tokens.insert(next_token);
                            generated += 1;
                            step += 1;

                            let mut text = detok.add(next_token as u32);
                            let finish_reason = if stop_set.contains(&next_token) { Some("stop") }
                                else if generated >= max_tokens { Some("length") }
                                else { None };
                            if finish_reason.is_some() { text.push_str(&detok.flush()); }
                            let finished = finish_reason.is_some();
                            let reject_logprobs = if logprobs_top_n > 0 {
                                let bl = &self.graph.as_ref().unwrap().h_batch_logits;
                                let offset = i * vocab_size;
                                Some(crate::decode::extract_top_logprobs(&bl[offset..offset + vocab_size], vocab_size, logprobs_top_n))
                            } else { None };
                            let cont = on_token(next_token, &text, finish_reason, reject_logprobs.as_deref());
                            if finished || !cont { stopped = true; }
                            break;
                        }
                    }

                    // If all verifiable draft tokens accepted, get bonus token
                    // (only if we had logits for the last position)
                    if !stopped && accepted_in_round == max_verifiable
                        && max_verifiable == draft_tokens.len()
                        && valid_positions > draft_tokens.len()
                        && step < max_tokens
                    {
                        let last_offset = draft_tokens.len() * vocab_size;
                        let batch_logits = &self.graph.as_ref().unwrap().h_batch_logits;
                        let mut last_argmax = 0usize;
                        let mut best_val = f32::NEG_INFINITY;
                        for j in 0..vocab_size {
                            if batch_logits[last_offset + j] > best_val {
                                best_val = batch_logits[last_offset + j];
                                last_argmax = j;
                            }
                        }
                        next_token = last_argmax;
                        seen_tokens.insert(next_token);
                        generated += 1;
                        step += 1;

                        let mut text = detok.add(next_token as u32);
                        let finish_reason = if stop_set.contains(&next_token) { Some("stop") }
                            else if generated >= max_tokens { Some("length") }
                            else { None };
                        if finish_reason.is_some() { text.push_str(&detok.flush()); }
                        let finished = finish_reason.is_some();
                        let bonus_logprobs = if logprobs_top_n > 0 {
                            let bl = &self.graph.as_ref().unwrap().h_batch_logits;
                            let offset = draft_tokens.len() * vocab_size;
                            Some(crate::decode::extract_top_logprobs(&bl[offset..offset + vocab_size], vocab_size, logprobs_top_n))
                        } else { None };
                        let cont = on_token(next_token, &text, finish_reason, bonus_logprobs.as_deref());
                        if finished || !cont { stopped = true; }
                    }
                }

                spec_accepted += accepted_in_round as u64;
                spec_rejected += (draft_tokens.len() - accepted_in_round) as u64;

                // 9. Restore LA states and replay only accepted tokens
                let t_restore = Instant::now();
                let num_accepted_batch = 1 + accepted_in_round; // next_token + accepted drafts
                if num_accepted_batch < batch_size {
                    if let Err(e) = self.restore_la_states() {
                        log::warn!("speculative: restore_la_states failed: {}", e);
                    }
                    if num_accepted_batch > 0 {
                        if let Err(e) = self.replay_la_states(
                            num_accepted_batch, &batch_positions[..num_accepted_batch])
                        {
                            log::warn!("speculative: replay_la_states failed: {}", e);
                        }
                    }
                }
                let restore_ms = t_restore.elapsed().as_secs_f64() * 1000.0;

                // 10. Rollback draft KV cache to match accepted tokens
                if let Some(ref mut draft) = self.draft {
                    draft.rollback_kv(draft_pos_before + 1 + accepted_in_round);
                }

                // Timing breakdown for first 5 rounds
                if spec_rounds <= 5 && std::env::var("KRASIS_SPEC_DEBUG").is_ok() {
                    eprintln!("  spec round {}: draft={:.1}ms verify={:.1}ms save={:.1}ms restore={:.1}ms accepted={}/{} valid={}/{} total={:.1}ms target_pred={} draft[0]={}",
                        spec_rounds, draft_elapsed * 1000.0, verify_ms, save_ms, restore_ms,
                        accepted_in_round, draft_tokens.len(),
                        valid_positions, batch_size,
                        (draft_elapsed * 1000.0) + verify_ms + save_ms + restore_ms,
                        target_pred, draft_tokens[0]);
                    // Debug: show top logit values for position 0
                    {
                        let bl = &self.graph.as_ref().unwrap().h_batch_logits;
                        let mut top5: Vec<(usize, f32)> = (0..vocab_size).map(|i| (i, bl[i])).collect();
                        top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        eprintln!("  pos0 top5: {:?}", &top5[..5]);
                    }
                }

                if stopped { break; }

            } else {
                // ── Normal (non-speculative) decode path ──
                // Per-layer CUDA graph replay. Graph failures are hard errors (no ungraphed fallback).
                // Timing always uses the production path (graph replay) — never a separate code path.
                let timing_enabled = self.graph.as_ref()
                    .map(|g| g.timing_enabled).unwrap_or(false);
                let use_per_layer = self.graph.as_ref()
                    .map(|g| g.per_layer_graphs_valid).unwrap_or(false);

                if use_per_layer {
                    if let Some(ref mut g) = self.graph {
                        g.validation_per_layer_steps += 1;
                    }
                    // Per-layer graph replay: handles sync/routing/DMA between graphs internally
                    let t_step = if timing_enabled { std::time::Instant::now() } else { std::time::Instant::now() };
                    if let Err(e) = self.replay_per_layer_graphs(next_token, pos, true) {
                        log::error!("gpu_generate_stream: graph replay failed (no ungraphed fallback): {}", e);
                        break;
                    } else if timing_enabled {
                        // Accumulate total step time for the timing report.
                        // Per-component breakdowns are not available in graph mode
                        // (kernels are baked into opaque CUDA graphs), but total time
                        // and cold/hot expert counts are accurate production numbers.
                        let graph = self.graph.as_mut().unwrap();
                        graph.t_total += t_step.elapsed().as_secs_f64();
                        graph.timing_step_count += 1;
                    }
                    sample_min_vram(&mut min_vram_free_bytes);
                    // logits already D2H'd inside replay_per_layer_graphs
                } else {
                    if let Some(ref mut g) = self.graph {
                        g.validation_ungraphed_steps += 1;
                    }
                    // First token (or after invalidation): run ungraphed, then capture per-layer graphs
                    if let Err(e) = self.gpu_decode_step(next_token, pos) {
                        log::error!("gpu_generate_stream: decode_step error: {}", e);
                        break;
                    }
                    sample_min_vram(&mut min_vram_free_bytes);

                    // Try to capture per-layer CUDA graphs after first token
                    let _has_graph_bufs = self.graph.as_ref()
                        .map(|g| g.d_graph_token_id.is_some()).unwrap_or(false);
                    if stderr_debug_enabled() {
                        eprintln!("[krasis] step={} has_graph_bufs={}", step, _has_graph_bufs);
                    }
                    let no_graph = std::env::var("KRASIS_NO_GRAPH").map(|v| v != "0").unwrap_or(false);
                    if step == 0 && _has_graph_bufs && !no_graph
                    {
                        match self.capture_per_layer_graphs(None, true, true, 0) {
                            Ok(()) => {
                                if stderr_debug_enabled() { eprintln!("[krasis] Per-layer CUDA graphs captured after token 0"); }
                                sample_min_vram(&mut min_vram_free_bytes);
                            }
                            Err(e) => {
                                log::error!("Per-layer graph capture failed (no ungraphed fallback): {}", e);
                                break;
                            }
                        }
                    }
                }

                let logits = &mut self.graph.as_mut().unwrap().h_logits;

                // Decode diagnostic: top-5 logits for first 3 steps
                if decode_diag && step < 3 {
                    let mut indexed: Vec<(usize, f32)> = logits.iter().copied()
                        .enumerate().take(vocab_size).collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let top5: Vec<String> = indexed[..5.min(indexed.len())].iter()
                        .map(|(idx, val)| {
                            let tok_str = tokenizer.decode(&[*idx as u32], true).unwrap_or_default();
                            format!("{}({:.3})\"{}\"", idx, val, tok_str.replace('\n', "\\n"))
                        }).collect();
                    eprintln!("[DECODE-DIAG] step={} pos={} input_tok={} top5=[{}]",
                        step, pos, next_token, top5.join(", "));
                }

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
                for &tok in &self.suppress_tokens {
                    if tok < vocab_size { logits[tok] = f32::NEG_INFINITY; }
                }
                let think_end_active = self.think_end_token_for_suppress.is_some() && !self.think_end_seen;
                let think_within_budget = self.think_suppress_budget == 0
                    || self.think_suppress_count < self.think_suppress_budget;
                let suppressing = generated < self.min_new_tokens || (think_end_active && think_within_budget);
                if suppressing {
                    for &tok in &self.stop_token_ids_for_suppress {
                        if tok < vocab_size { logits[tok] = f32::NEG_INFINITY; }
                    }
                }

                // Force-inject </think> when thinking budget is exhausted.
                // Just lifting suppression isn't enough — degenerate models never
                // produce EOS naturally once stuck in a thinking loop.
                let target_pred = if think_end_active && !think_within_budget {
                    if let Some(te) = self.think_end_token_for_suppress {
                        eprintln!("[krasis] Thinking budget exhausted ({} tokens) — force-injecting </think>",
                            self.think_suppress_count);
                        te
                    } else {
                        crate::decode::sample_from_logits_pub(
                            logits, vocab_size, temperature, top_k, top_p, &mut rng_next)
                    }
                } else {
                    crate::decode::sample_from_logits_pub(
                        logits, vocab_size, temperature, top_k, top_p, &mut rng_next)
                };
                self.notify_token_generated(target_pred);

                seen_tokens.insert(target_pred);
                generated += 1;
                step += 1;
                if let Some(ref mut g) = self.graph {
                    g.validation_decode_steps += 1;
                }
                next_token = target_pred;

                // Lightweight profiling checkpoint
                if profile_enabled && step > 0 && step % profile_interval == 0 {
                    let now = Instant::now();
                    let window_secs = (now - profile_last_time).as_secs_f64();
                    let window_tokens = (step - profile_last_step) as f64;
                    let window_tps = window_tokens / window_secs;
                    let window_ms = window_secs * 1000.0 / window_tokens;
                    let g = self.graph.as_ref().unwrap();
                    let cold = g.dma_cold_experts - profile_cold_total;
                    let hcs = g.dma_hcs_experts - profile_hcs_total;
                    let cold_per_tok = cold as f64 / window_tokens;
                    let hcs_per_tok = hcs as f64 / window_tokens;
                    eprintln!("  [profile] tok {}-{}: {:.1} tok/s ({:.1}ms/tok)  cold={:.0}/tok hcs={:.0}/tok",
                        profile_last_step + 1, step, window_tps, window_ms, cold_per_tok, hcs_per_tok);
                    profile_last_step = step;
                    profile_last_time = now;
                    profile_cold_total = g.dma_cold_experts;
                    profile_hcs_total = g.dma_hcs_experts;
                }

                let mut text = detok.add(target_pred as u32);
                let finish_reason = if stop_set.contains(&target_pred) {
                    Some("stop")
                } else if generated >= max_tokens {
                    Some("length")
                } else {
                    None
                };
                if finish_reason.is_some() {
                    text.push_str(&detok.flush());
                }
                let finished = finish_reason.is_some();
                let top_logprobs = if logprobs_top_n > 0 {
                    let logits = &self.graph.as_ref().unwrap().h_logits;
                    Some(crate::decode::extract_top_logprobs(logits, vocab_size, logprobs_top_n))
                } else { None };
                let cont = on_token(target_pred, &text, finish_reason, top_logprobs.as_deref());
                if finished || !cont { break; }
            }
        }

        let elapsed = decode_start.elapsed().as_secs_f64();
        if generated > 0 {
            let tps = generated as f64 / elapsed;
            // Query current VRAM free to show safety margin headroom
            let mut vram_free: usize = 0;
            let mut vram_total: usize = 0;
            unsafe {
                let _ = cuda_sys::lib().cuMemGetInfo_v2(&mut vram_free, &mut vram_total);
            }
            // Post-decode snapshot also contributes to min tracking
            if vram_free < min_vram_free_bytes {
                min_vram_free_bytes = vram_free;
            }
            let free_mb = vram_free / (1024 * 1024);
            let total_mb = vram_total / (1024 * 1024);
            let used_mb = total_mb - free_mb;
            let min_free_mb = if min_vram_free_bytes < usize::MAX {
                min_vram_free_bytes / (1024 * 1024)
            } else {
                free_mb
            };
            self.last_min_free_vram_mb = min_free_mb;
            eprintln!("  \x1b[32mdecode: {} tokens in {:.2}s ({:.1} tok/s)  VRAM: {} MB free now, {} MB min free during decode\x1b[0m",
                generated, elapsed, tps, free_mb, min_free_mb);

        }

        // Print speculative decode stats
        if use_speculative && spec_rounds > 0 {
            let total_drafted = spec_accepted + spec_rejected;
            let acceptance_rate = if total_drafted > 0 {
                spec_accepted as f64 / total_drafted as f64
            } else { 0.0 };
            let avg_draft_ms = spec_draft_time / spec_rounds as f64 * 1000.0;
            let avg_accepted = spec_accepted as f64 / spec_rounds as f64;
            eprintln!("  \x1b[36m┌─────────────────────────────────────────────────┐\x1b[0m");
            eprintln!("  \x1b[36m│  SPECULATIVE DECODE STATS                       │\x1b[0m");
            eprintln!("  \x1b[36m├─────────────────────────────────────────────────┤\x1b[0m");
            eprintln!("  \x1b[36m│  Rounds:       {:>4}                             │\x1b[0m", spec_rounds);
            eprintln!("  \x1b[36m│  Accepted:     {:>4} / {:>4} ({:.1}%){}│\x1b[0m",
                spec_accepted, total_drafted, acceptance_rate * 100.0,
                " ".repeat(20usize.saturating_sub(format!("{:.1}", acceptance_rate * 100.0).len())));
            eprintln!("  \x1b[36m│  Avg accepted: {:.1}/round                       │\x1b[0m", avg_accepted);
            let avg_verify_ms = spec_verify_time / spec_rounds as f64;
            let avg_save_ms = spec_save_time / spec_rounds as f64;
            eprintln!("  \x1b[36m│  Draft time:   {:.2} ms/round                  │\x1b[0m", avg_draft_ms);
            eprintln!("  \x1b[36m│  Verify time:  {:.2} ms/round                  │\x1b[0m", avg_verify_ms);
            eprintln!("  \x1b[36m│  Save/restore: {:.2} ms/round                  │\x1b[0m", avg_save_ms);
            if spec_failfast_count > 0 {
                eprintln!("  \x1b[36m│  Fail-fast:    {:>4} bailouts ({} tokens saved) │\x1b[0m",
                    spec_failfast_count, spec_tokens_saved);
            }
            eprintln!("  \x1b[36m└─────────────────────────────────────────────────┘\x1b[0m");
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

                let graph_mode = avg_attn < 0.001 && avg_moe < 0.001 && avg_lm < 0.001;
                eprintln!("  \x1b[36m┌─────────────────────────────────────────────────┐\x1b[0m");
                if graph_mode {
                    eprintln!("  \x1b[36m│\x1b[0m  GPU DECODE TIMING ({} tokens, graph mode)     \x1b[36m│\x1b[0m", graph.timing_step_count);
                } else {
                    eprintln!("  \x1b[36m│\x1b[0m  GPU DECODE TIMING ({} tokens avg)             \x1b[36m│\x1b[0m", graph.timing_step_count);
                }
                eprintln!("  \x1b[36m├─────────────────────────────────────────────────┤\x1b[0m");
                eprintln!("  \x1b[36m│\x1b[0m  Total:       {:7.2} ms/tok  ({:5.1} tok/s)    \x1b[36m│\x1b[0m", avg_total, 1000.0 / avg_total);
                if graph_mode {
                    eprintln!("  \x1b[36m│\x1b[0m  (per-component breakdown not available        \x1b[36m│\x1b[0m");
                    eprintln!("  \x1b[36m│\x1b[0m   in CUDA graph mode -- kernels are opaque)    \x1b[36m│\x1b[0m");
                }
                let avg_attn_la = graph.t_attn_la / n * 1000.0;
                let avg_attn_gqa = graph.t_attn_gqa / n * 1000.0;
                if !graph_mode {
                eprintln!("  \x1b[36m│\x1b[0m  Attention:   {:7.2} ms  ({:4.1}%)              \x1b[36m│\x1b[0m", avg_attn, avg_attn / avg_total * 100.0);
                eprintln!("  \x1b[36m│\x1b[0m    LA (36):   {:7.2} ms                        \x1b[36m│\x1b[0m", avg_attn_la);
                let avg_la_proj = graph.t_la_proj / n * 1000.0;
                let avg_la_conv = graph.t_la_conv / n * 1000.0;
                let avg_la_recur = graph.t_la_recur / n * 1000.0;
                let avg_la_out = graph.t_la_out / n * 1000.0;
                let avg_la_other = avg_attn_la - avg_la_proj - avg_la_conv - avg_la_recur - avg_la_out;
                eprintln!("  \x1b[36m│\x1b[0m      Proj:    {:7.2} ms  ({:4.1}%)             \x1b[36m│\x1b[0m", avg_la_proj, if avg_attn_la > 0.001 { avg_la_proj / avg_attn_la * 100.0 } else { 0.0 });
                eprintln!("  \x1b[36m│\x1b[0m      Conv:    {:7.2} ms  ({:4.1}%)             \x1b[36m│\x1b[0m", avg_la_conv, if avg_attn_la > 0.001 { avg_la_conv / avg_attn_la * 100.0 } else { 0.0 });
                eprintln!("  \x1b[36m│\x1b[0m      Recur:   {:7.2} ms  ({:4.1}%)             \x1b[36m│\x1b[0m", avg_la_recur, if avg_attn_la > 0.001 { avg_la_recur / avg_attn_la * 100.0 } else { 0.0 });
                eprintln!("  \x1b[36m│\x1b[0m      Out:     {:7.2} ms  ({:4.1}%)             \x1b[36m│\x1b[0m", avg_la_out, if avg_attn_la > 0.001 { avg_la_out / avg_attn_la * 100.0 } else { 0.0 });
                if avg_la_other.abs() > 0.01 {
                    eprintln!("  \x1b[36m│\x1b[0m      Other:   {:7.2} ms                        \x1b[36m│\x1b[0m", avg_la_other);
                }
                eprintln!("  \x1b[36m│\x1b[0m    GQA (12):  {:7.2} ms                        \x1b[36m│\x1b[0m", avg_attn_gqa);
                let avg_gqa_proj = graph.t_gqa_proj / n * 1000.0;
                let avg_gqa_attn = graph.t_gqa_attn / n * 1000.0;
                let avg_gqa_out = graph.t_gqa_out / n * 1000.0;
                let avg_gqa_other = avg_attn_gqa - avg_gqa_proj - avg_gqa_attn - avg_gqa_out;
                eprintln!("  \x1b[36m│\x1b[0m      Proj:    {:7.2} ms  ({:4.1}%)             \x1b[36m│\x1b[0m", avg_gqa_proj, if avg_attn_gqa > 0.001 { avg_gqa_proj / avg_attn_gqa * 100.0 } else { 0.0 });
                eprintln!("  \x1b[36m│\x1b[0m      Attn:    {:7.2} ms  ({:4.1}%)             \x1b[36m│\x1b[0m", avg_gqa_attn, if avg_attn_gqa > 0.001 { avg_gqa_attn / avg_attn_gqa * 100.0 } else { 0.0 });
                eprintln!("  \x1b[36m│\x1b[0m      Out:     {:7.2} ms  ({:4.1}%)             \x1b[36m│\x1b[0m", avg_gqa_out, if avg_attn_gqa > 0.001 { avg_gqa_out / avg_attn_gqa * 100.0 } else { 0.0 });
                if avg_gqa_other.abs() > 0.01 {
                    eprintln!("  \x1b[36m│\x1b[0m      Other:   {:7.2} ms                        \x1b[36m│\x1b[0m", avg_gqa_other);
                }
                eprintln!("  \x1b[36m│\x1b[0m  MoE:         {:7.2} ms  ({:4.1}%)              \x1b[36m│\x1b[0m", avg_moe, avg_moe / avg_total * 100.0);
                eprintln!("  \x1b[36m│\x1b[0m  Norms+Emb:   {:7.2} ms  ({:4.1}%)              \x1b[36m│\x1b[0m", avg_norm, avg_norm / avg_total * 100.0);
                eprintln!("  \x1b[36m│\x1b[0m  Dense MLP:   {:7.2} ms  ({:4.1}%)              \x1b[36m│\x1b[0m", avg_dense, avg_dense / avg_total * 100.0);
                eprintln!("  \x1b[36m│\x1b[0m  LM Head:     {:7.2} ms  ({:4.1}%)              \x1b[36m│\x1b[0m", avg_lm, avg_lm / avg_total * 100.0);
                let other_ms = avg_total - avg_attn - avg_moe - avg_norm - avg_dense - avg_lm;
                eprintln!("  \x1b[36m│\x1b[0m  Other:       {:7.2} ms  ({:4.1}%)              \x1b[36m│\x1b[0m",
                    other_ms, other_ms / avg_total * 100.0);
                eprintln!("  \x1b[36m├─────────────────────────────────────────────────┤\x1b[0m");
                let avg_route_sync = graph.t_moe_route_sync / n * 1000.0;
                let avg_expert_loop = graph.t_moe_expert_loop / n * 1000.0;
                let avg_shared = graph.t_moe_shared / n * 1000.0;
                let avg_moe_other = avg_moe - avg_route_sync - avg_expert_loop - avg_shared;
                eprintln!("  \x1b[36m│\x1b[0m  MoE breakdown (of {:.2} ms):                    \x1b[36m│\x1b[0m", avg_moe);
                eprintln!("  \x1b[36m│\x1b[0m    Route sync:  {:7.2} ms  ({:4.1}%)             \x1b[36m│\x1b[0m", avg_route_sync, avg_route_sync / avg_moe * 100.0);
                eprintln!("  \x1b[36m│\x1b[0m    Expert loop: {:7.2} ms  ({:4.1}%)             \x1b[36m│\x1b[0m", avg_expert_loop, avg_expert_loop / avg_moe * 100.0);
                let avg_exp_w13 = graph.t_expert_w13 / n * 1000.0;
                let avg_exp_silu = graph.t_expert_silu_w2 / n * 1000.0;
                let avg_exp_other = avg_expert_loop - avg_exp_w13 - avg_exp_silu;
                eprintln!("  \x1b[36m│\x1b[0m      w13 GEMV: {:7.2} ms  ({:4.1}%)             \x1b[36m│\x1b[0m", avg_exp_w13, if avg_expert_loop > 0.001 { avg_exp_w13 / avg_expert_loop * 100.0 } else { 0.0 });
                eprintln!("  \x1b[36m│\x1b[0m      silu+w2:  {:7.2} ms  ({:4.1}%)             \x1b[36m│\x1b[0m", avg_exp_silu, if avg_expert_loop > 0.001 { avg_exp_silu / avg_expert_loop * 100.0 } else { 0.0 });
                let avg_dma_wait = graph.t_dma_expert_wait / n * 1000.0;
                let avg_dma_compute = graph.t_dma_expert_compute / n * 1000.0;
                if avg_dma_wait > 0.01 || avg_dma_compute > 0.01 {
                    eprintln!("  \x1b[36m│\x1b[0m      DMA wait: {:7.2} ms  (cold expert DMA)    \x1b[36m│\x1b[0m", avg_dma_wait);
                    eprintln!("  \x1b[36m│\x1b[0m      DMA comp: {:7.2} ms  (cold expert GEMV)   \x1b[36m│\x1b[0m", avg_dma_compute);
                }
                if avg_exp_other.abs() > 0.01 {
                    let avg_exp_other_adj = avg_expert_loop - avg_exp_w13 - avg_exp_silu - avg_dma_wait - avg_dma_compute;
                    eprintln!("  \x1b[36m│\x1b[0m      Other:    {:7.2} ms                        \x1b[36m│\x1b[0m", avg_exp_other_adj);
                }
                eprintln!("  \x1b[36m│\x1b[0m    Shared exp:  {:7.2} ms  ({:4.1}%)             \x1b[36m│\x1b[0m", avg_shared, avg_shared / avg_moe * 100.0);
                eprintln!("  \x1b[36m│\x1b[0m    MoE other:   {:7.2} ms  ({:4.1}%)             \x1b[36m│\x1b[0m", avg_moe_other, avg_moe_other / avg_moe * 100.0);
                // Fine-grained MoE "other" breakdown
                let avg_gate = graph.t_moe_gate_gemv / n * 1000.0;
                let avg_d2h = graph.t_moe_d2h_topk / n * 1000.0;
                let avg_apfl = graph.t_moe_apfl / n * 1000.0;
                let avg_padding_setup = graph.t_moe_padding_setup / n * 1000.0;
                let avg_d2d = graph.t_moe_d2d_copy / n * 1000.0;
                let avg_rest = avg_moe_other - avg_gate - avg_d2h - avg_apfl - avg_padding_setup - avg_d2d;
                eprintln!("  \x1b[36m│\x1b[0m  MoE other detail:                               \x1b[36m│\x1b[0m");
                eprintln!("  \x1b[36m│\x1b[0m    Gate GEMV:   {:7.2} ms                        \x1b[36m│\x1b[0m", avg_gate);
                eprintln!("  \x1b[36m│\x1b[0m    D2H topk:   {:7.2} ms                        \x1b[36m│\x1b[0m", avg_d2h);
                eprintln!("  \x1b[36m│\x1b[0m    APFL+setup: {:7.2} ms                        \x1b[36m│\x1b[0m", avg_apfl);
                eprintln!("  \x1b[36m│\x1b[0m    Replay pad:  {:7.2} ms                        \x1b[36m│\x1b[0m", avg_padding_setup);
                eprintln!("  \x1b[36m│\x1b[0m    D2D copy:   {:7.2} ms                        \x1b[36m│\x1b[0m", avg_d2d);
                eprintln!("  \x1b[36m│\x1b[0m    Remainder:  {:7.2} ms                        \x1b[36m│\x1b[0m", avg_rest);
                } // end !graph_mode per-component section
                eprintln!("  \x1b[36m└─────────────────────────────────────────────────┘\x1b[0m");

                // Measured PCIe DMA stats
                let (hcs_cached, hcs_hits, hcs_misses) = if let Some(ref hcs) = graph.hcs {
                    (hcs.num_cached, hcs.total_hits, hcs.total_misses)
                } else { (0, 0, 0) };
                let avg_dma_bytes = graph.dma_bytes_total as f64 / n;
                let avg_dma_calls = graph.dma_call_count as f64 / n;
                let avg_cold = graph.dma_cold_experts as f64 / n;
                let avg_hcs = graph.dma_hcs_experts as f64 / n;
                let dma_mb = avg_dma_bytes / (1024.0 * 1024.0);
                // In graph mode, per-component expert_loop time isn't available,
                // so estimate PCIe BW from total time and cold fraction instead
                let avg_expert_loop_for_bw = if graph_mode {
                    // Estimate: MoE is ~70% of total time for QCN-like models
                    avg_total * 0.7
                } else {
                    graph.t_moe_expert_loop / n * 1000.0
                };
                let cold_frac = avg_cold / (avg_cold + avg_hcs).max(1.0);
                let est_dma_time_ms = avg_expert_loop_for_bw * cold_frac;
                let est_pcie_bw = if est_dma_time_ms > 0.001 {
                    dma_mb / (est_dma_time_ms / 1000.0) / 1024.0
                } else { 0.0 };
                eprintln!("  \x1b[36m├─────────────────────────────────────────────────┤\x1b[0m");
                eprintln!("  \x1b[36m│\x1b[0m  PCIe DMA (non-serialized):                     \x1b[36m│\x1b[0m");
                eprintln!("  \x1b[36m│\x1b[0m    Cold experts/tok: {:.1} ({:.0} DMA calls)      \x1b[36m│\x1b[0m", avg_cold, avg_dma_calls);
                eprintln!("  \x1b[36m│\x1b[0m    HCS experts/tok:  {:.1} ({} cached)            \x1b[36m│\x1b[0m", avg_hcs, hcs_cached);
                eprintln!("  \x1b[36m│\x1b[0m    DMA bytes/tok:    {:.2} MB                     \x1b[36m│\x1b[0m", dma_mb);
                eprintln!("  \x1b[36m│\x1b[0m    HCS hit/miss:     {}/{}                        \x1b[36m│\x1b[0m", hcs_hits, hcs_misses);
                let bytes_per_call = if avg_dma_calls > 0.0 { avg_dma_bytes / avg_dma_calls } else { 0.0 };
                eprintln!("  \x1b[36m│\x1b[0m    Avg DMA call size: {:.1} KB                   \x1b[36m│\x1b[0m", bytes_per_call / 1024.0);
                eprintln!("  \x1b[36m│\x1b[0m    Est PCIe BW:      {:.1} GB/s (cold fraction)  \x1b[36m│\x1b[0m", est_pcie_bw);
                eprintln!("  \x1b[36m└─────────────────────────────────────────────────┘\x1b[0m");

                // Also emit to structured log (no ANSI escapes)
                log::info!(
                    "DECODE TIMING SUMMARY ({} tokens, {}): \
                     Total: {:.2} ms/tok ({:.1} tok/s) | \
                     PCIe: {:.1} cold exp/tok, {:.1} HCS exp/tok, {:.2} MB/tok, {:.0} DMA calls/tok | \
                     HCS cached: {}, hit: {}, miss: {}",
                    graph.timing_step_count,
                    if graph_mode { "graph mode" } else { "ungraphed" },
                    avg_total, 1000.0 / avg_total,
                    avg_cold, avg_hcs, dma_mb, avg_dma_calls,
                    hcs_cached, hcs_hits, hcs_misses
                );
            }
        }

        generated
    }
}

// ── Marlin perm table computation + GPU decode internals ──────────────

impl GpuDecodeStore {
    fn emit_batch_decode_timing_summary(&self) {
        let graph = match self.graph.as_ref() {
            Some(g) if g.timing_enabled && g.timing_step_count > 0 => g,
            _ => return,
        };

        let n = graph.timing_step_count as f64;
        let avg_total = graph.t_total / n * 1000.0;
        let avg_attn = graph.t_attn / n * 1000.0;
        let avg_moe = graph.t_route / n * 1000.0;
        let avg_norm = graph.t_norm / n * 1000.0;
        let avg_dense = graph.t_dense_mlp / n * 1000.0;
        let avg_lm = graph.t_lm_head / n * 1000.0;
        let graph_mode = avg_attn < 0.001 && avg_moe < 0.001 && avg_lm < 0.001;

        eprintln!("  \x1b[36m┌─────────────────────────────────────────────────┐\x1b[0m");
        if graph_mode {
            eprintln!("  \x1b[36m│\x1b[0m  GPU DECODE TIMING ({} tokens, graph mode)     \x1b[36m│\x1b[0m", graph.timing_step_count);
        } else {
            eprintln!("  \x1b[36m│\x1b[0m  GPU DECODE TIMING ({} tokens avg)             \x1b[36m│\x1b[0m", graph.timing_step_count);
        }
        eprintln!("  \x1b[36m├─────────────────────────────────────────────────┤\x1b[0m");
        eprintln!("  \x1b[36m│\x1b[0m  Total:       {:7.2} ms/tok  ({:5.1} tok/s)    \x1b[36m│\x1b[0m", avg_total, 1000.0 / avg_total);
        if graph_mode {
            eprintln!("  \x1b[36m│\x1b[0m  Graph replay hides per-component kernels.      \x1b[36m│\x1b[0m");
        } else {
            let avg_attn_la = graph.t_attn_la / n * 1000.0;
            let avg_attn_gqa = graph.t_attn_gqa / n * 1000.0;
            let avg_route_sync = graph.t_moe_route_sync / n * 1000.0;
            let avg_expert_loop = graph.t_moe_expert_loop / n * 1000.0;
            let avg_shared = graph.t_moe_shared / n * 1000.0;
            let avg_gate = graph.t_moe_gate_gemv / n * 1000.0;
            let avg_d2h = graph.t_moe_d2h_topk / n * 1000.0;
            let avg_apfl = graph.t_moe_apfl / n * 1000.0;
            let avg_padding = graph.t_moe_padding_setup / n * 1000.0;
            let avg_d2d = graph.t_moe_d2d_copy / n * 1000.0;
            let avg_exp_w13 = graph.t_expert_w13 / n * 1000.0;
            let avg_exp_silu = graph.t_expert_silu_w2 / n * 1000.0;
            eprintln!("  \x1b[36m│\x1b[0m  Attention:   {:7.2} ms  (LA {:6.2} | GQA {:6.2}) \x1b[36m│\x1b[0m", avg_attn, avg_attn_la, avg_attn_gqa);
            eprintln!("  \x1b[36m│\x1b[0m  MoE:         {:7.2} ms  (sync {:5.2} loop {:5.2}) \x1b[36m│\x1b[0m", avg_moe, avg_route_sync, avg_expert_loop);
            eprintln!("  \x1b[36m│\x1b[0m    w13:       {:7.2} ms  | silu+w2 {:7.2} ms   \x1b[36m│\x1b[0m", avg_exp_w13, avg_exp_silu);
            eprintln!("  \x1b[36m│\x1b[0m    shared:    {:7.2} ms  | gate    {:7.2} ms   \x1b[36m│\x1b[0m", avg_shared, avg_gate);
            eprintln!("  \x1b[36m│\x1b[0m    D2H topk:  {:7.2} ms  | APFL    {:7.2} ms   \x1b[36m│\x1b[0m", avg_d2h, avg_apfl);
            eprintln!("  \x1b[36m│\x1b[0m    Replay pad:{:7.2} ms  | D2D     {:7.2} ms   \x1b[36m│\x1b[0m", avg_padding, avg_d2d);
            eprintln!("  \x1b[36m│\x1b[0m  Norms+Emb:   {:7.2} ms  | Dense   {:7.2} ms   \x1b[36m│\x1b[0m", avg_norm, avg_dense);
            eprintln!("  \x1b[36m│\x1b[0m  LM Head:     {:7.2} ms                         \x1b[36m│\x1b[0m", avg_lm);
        }

        let (hcs_cached, hcs_hits, hcs_misses) = if let Some(ref hcs) = graph.hcs {
            (hcs.num_cached, hcs.total_hits, hcs.total_misses)
        } else {
            (0, 0, 0)
        };
        let avg_dma_bytes = graph.dma_bytes_total as f64 / n;
        let avg_dma_calls = graph.dma_call_count as f64 / n;
        let avg_cold = graph.dma_cold_experts as f64 / n;
        let avg_hcs = graph.dma_hcs_experts as f64 / n;
        eprintln!("  \x1b[36m├─────────────────────────────────────────────────┤\x1b[0m");
        eprintln!("  \x1b[36m│\x1b[0m  PCIe: cold {:5.1}/tok | HCS {:5.1}/tok | {:5.2} MB/tok \x1b[36m│\x1b[0m",
            avg_cold, avg_hcs, avg_dma_bytes / (1024.0 * 1024.0));
        eprintln!("  \x1b[36m│\x1b[0m  DMA calls: {:5.1}/tok | HCS cache {} | hit/miss {}/{} \x1b[36m│\x1b[0m",
            avg_dma_calls, hcs_cached, hcs_hits, hcs_misses);
        eprintln!("  \x1b[36m└─────────────────────────────────────────────────┘\x1b[0m");

        log::info!(
            "BATCH DECODE TIMING SUMMARY ({} tokens, {}): total {:.2} ms/tok ({:.1} tok/s), attn {:.2} ms, moe {:.2} ms, replay_pad {:.2} ms, cold {:.1}/tok, hcs {:.1}/tok",
            graph.timing_step_count,
            if graph_mode { "graph mode" } else { "ungraphed" },
            avg_total,
            1000.0 / avg_total,
            avg_attn,
            avg_moe,
            graph.t_moe_padding_setup / n * 1000.0,
            avg_cold,
            avg_hcs,
        );
    }

    /// Compute inverse Marlin INT4 + INT8 weight perm and scale perm tables,
    /// upload all to GPU device memory.
    fn upload_marlin_perm_tables(
        device: &Arc<CudaDevice>,
    ) -> PyResult<(cudarc::driver::CudaSlice<i32>, cudarc::driver::CudaSlice<i32>, cudarc::driver::CudaSlice<i32>)> {
        use crate::weights::marlin::{generate_weight_perm_int4, generate_weight_perm_int8, generate_scale_perms};

        // INT4 forward tables
        let fwd_weight_int4 = generate_weight_perm_int4();
        let (fwd_scale, _) = generate_scale_perms();

        // INT4 inverse: inv[fwd[i]] = i
        let mut inv_weight_int4 = [0i32; 1024];
        for (i, &src) in fwd_weight_int4.iter().enumerate() {
            inv_weight_int4[src] = i as i32;
        }
        let mut inv_scale = [0i32; 64];
        for (i, &src) in fwd_scale.iter().enumerate() {
            inv_scale[src] = i as i32;
        }

        // INT8 forward tables (different interleave pattern)
        let fwd_weight_int8 = generate_weight_perm_int8();
        let mut inv_weight_int8 = [0i32; 1024];
        for (i, &src) in fwd_weight_int8.iter().enumerate() {
            inv_weight_int8[src] = i as i32;
        }

        // Upload to GPU
        let d_inv_weight_int4 = device.htod_copy(inv_weight_int4.to_vec())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to upload inv_weight_perm_int4: {:?}", e)))?;
        let d_inv_scale = device.htod_copy(inv_scale.to_vec())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to upload inv_scale_perm: {:?}", e)))?;
        let d_inv_weight_int8 = device.htod_copy(inv_weight_int8.to_vec())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to upload inv_weight_perm_int8: {:?}", e)))?;

        log::info!("GpuDecodeStore: uploaded Marlin inverse perm tables (INT4=1024, INT8=1024, scale=64)");
        Ok((d_inv_weight_int4, d_inv_scale, d_inv_weight_int8))
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
                    contiguous_ptr: 0,
                    contiguous_bytes: 0,
                    mapped_w13_packed_dptr: 0,
                    mapped_w13_scales_dptr: 0,
                    mapped_w2_packed_dptr: 0,
                    mapped_w2_scales_dptr: 0,
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
                    contiguous_ptr: 0,
                    contiguous_bytes: 0,
                    mapped_w13_packed_dptr: 0,
                    mapped_w13_scales_dptr: 0,
                    mapped_w2_packed_dptr: 0,
                    mapped_w2_scales_dptr: 0,
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
            activation_type: 0,     // silu_gated (default for existing models)
            gated_experts: true,    // existing models have gate_proj+up_proj
            latent_down_wid: None,  // no latent projections for standard MoE
            latent_up_wid: None,
            moe_input_size: 0,     // 0 = use hidden_size (standard MoE)
        });

        // Pin shared expert in VRAM if present (Certainty Rule: always accessed, zero DMA at runtime)
        while graph.shared_expert_vram.len() <= layer_idx {
            graph.shared_expert_vram.push(None);
        }
        if let Some(ref se) = graph.moe_layers[layer_idx].as_ref().unwrap().shared {
            // Infer shared expert quantization bits from packed weight size.
            // BF16: packed_bytes = k * n * 2, INT8: packed_bytes = k * n, INT4: packed_bytes = k * n / 2
            let se_n_w13 = graph.moe_intermediate_size * 2; // gated: gate+up
            let se_k_w13 = graph.hidden_size;
            let expected_bf16 = se_k_w13 * se_n_w13 * 2;
            let expected_int8 = se_k_w13 * se_n_w13;
            if stderr_debug_enabled() {
                eprintln!("[SHARED_EXPERT_BITS] layer={} packed={} expected_bf16={} expected_int8={} moe_inter={} hidden={}",
                    layer_idx, se.w13_packed_bytes, expected_bf16, expected_int8, graph.moe_intermediate_size, graph.hidden_size);
            }
            if se.w13_packed_bytes >= expected_bf16 {
                graph.shared_expert_bits = 16;
                if stderr_debug_enabled() {
                    eprintln!("[SHARED_EXPERT_BITS] => BF16 (bits=16)");
                }
            } else if se.w13_packed_bytes >= expected_int8 {
                graph.shared_expert_bits = 8;
                if stderr_debug_enabled() {
                    eprintln!("[SHARED_EXPERT_BITS] => INT8 (bits=8)");
                }
            } else {
                graph.shared_expert_bits = 4;
                if stderr_debug_enabled() {
                    eprintln!("[SHARED_EXPERT_BITS] => INT4 (bits=4)");
                }
            }

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
        let is_int8 = graph.expert_bits == 8;
        let inv_wp = if is_int8 {
            *graph.d_inv_weight_perm_int8.device_ptr()
        } else {
            *graph.d_inv_weight_perm.device_ptr()
        };
        self.launch_marlin_gemv_raw(
            packed_ptr, scales_ptr, input_ptr, output_ptr,
            inv_wp,
            *graph.d_inv_scale_perm.device_ptr(),
            k, n, group_size,
            is_int8,
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
        is_int8: bool,
    ) -> PyResult<()> {
        if !self.kernels_loaded {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Decode kernels not loaded"));
        }

        let kernel_name = if is_int8 { "marlin_gemv_int8" } else { "marlin_gemv_int4" };
        let f = self.device.get_func(MODULE_NAME, kernel_name)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                format!("{} kernel not found", kernel_name)))?;

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
        is_int8: bool,
    ) -> PyResult<()> {
        let n_tiles = (n + 15) / 16;
        let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_tiles as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem_bytes,
        };

        let kernel = if is_int8 {
            kernels.fused_silu_accum_int8.clone()
        } else {
            kernels.fused_silu_accum.clone()
        };

        unsafe {
            kernel.launch(cfg, (
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
        is_int8: bool,
    ) -> PyResult<()> {
        let n_tiles = (n + 15) / 16;
        // Shared mem: input BF16 [K*2] + inv_wperm [1024*4] + inv_sperm [64*4] + reduce [16*16*4]
        let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4 + 16 * 16 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_tiles as u32, k_splits as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem_bytes,
        };

        let kernel = if is_int8 {
            kernels.marlin_gemv_int8_v2.clone()
        } else {
            kernels.marlin_gemv_int4_v2.clone()
        };

        unsafe {
            kernel.launch(cfg, (
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
                format!("marlin_gemv_v2 launch: {:?}", e)))?;
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

    /// Launch Marlin INT8 GEMV with BF16 output (for attention projections).
    /// Uses v2 with K-splitting for small-N projections (k_proj, v_proj).
    fn launch_marlin_gemv_int8_bf16(
        &self, w: &GpuWeight, input_ptr: u64, output_ptr: u64,
    ) -> Result<(), String> {
        let n = w.rows;
        let k = w.cols;
        let k_splits = self.compute_k_splits(n, k);

        if k_splits <= 1 {
            // v1: sufficient SM occupancy from N tiles alone
            let n_tiles = (n + 15) / 16;
            let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4) as u32;
            let kernel = self.device.get_func(MODULE_NAME, "marlin_gemv_int8")
                .ok_or("marlin_gemv_int8 kernel not found")?;
            unsafe {
                kernel.launch(
                    cudarc::driver::LaunchConfig {
                        grid_dim: (n_tiles as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: smem_bytes,
                    },
                    (w.ptr, w.scales_ptr, input_ptr, output_ptr,
                     self.perm_inv_weight_int8, self.perm_inv_scale,
                     k as i32, n as i32, w.group_size as i32),
                ).map_err(|e| format!("marlin_gemv_int8 launch: {:?}", e))?;
            }
            return Ok(());
        }

        // v2 with K-splitting for small-N: better SM utilization
        let n_tiles = (n + 15) / 16;
        let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4 + 16 * 16 * 4) as u32;
        let partial_ptr = self.v2_partial_ptr;

        let v2_kernel = self.device.get_func(MODULE_NAME, "marlin_gemv_int8_v2")
            .ok_or("marlin_gemv_int8_v2 kernel not found")?;
        unsafe {
            v2_kernel.launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (n_tiles as u32, k_splits as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem_bytes,
                },
                (w.ptr, w.scales_ptr, input_ptr, partial_ptr,
                 self.perm_inv_weight_int8, self.perm_inv_scale,
                 k as i32, n as i32, w.group_size as i32, k_splits as i32),
            ).map_err(|e| format!("marlin_gemv_int8_v2 bf16 launch: {:?}", e))?;
        }

        let reduce_kernel = self.device.get_func(MODULE_NAME, "reduce_ksplits_bf16")
            .ok_or("reduce_ksplits_bf16 kernel not found")?;
        unsafe {
            reduce_kernel.launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (((n + 255) / 256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (output_ptr, partial_ptr, n as i32, k_splits as i32),
            ).map_err(|e| format!("reduce_ksplits_bf16 int8 launch: {:?}", e))?;
        }
        Ok(())
    }

    /// Launch Marlin INT8 GEMV with FP32 output (for attention score paths).
    /// Uses fused v2 with inline atomic reduction for small-N projections.
    fn launch_marlin_gemv_int8_f32(
        &self, w: &GpuWeight, input_ptr: u64, output_ptr: u64,
    ) -> Result<(), String> {
        let n = w.rows;
        let k = w.cols;
        let k_splits = self.compute_k_splits(n, k);

        if k_splits <= 1 {
            let n_tiles = (n + 15) / 16;
            let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4) as u32;
            let kernel = self.device.get_func(MODULE_NAME, "marlin_gemv_int8_f32")
                .ok_or("marlin_gemv_int8_f32 kernel not found")?;
            unsafe {
                kernel.launch(
                    cudarc::driver::LaunchConfig {
                        grid_dim: (n_tiles as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: smem_bytes,
                    },
                    (w.ptr, w.scales_ptr, input_ptr, output_ptr,
                     self.perm_inv_weight_int8, self.perm_inv_scale,
                     k as i32, n as i32, w.group_size as i32),
                ).map_err(|e| format!("marlin_gemv_int8_f32 launch: {:?}", e))?;
            }
            return Ok(());
        }

        // Fused v2: zero output then atomicAdd directly
        let n_tiles = (n + 15) / 16;
        let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4 + 16 * 16 * 4) as u32;

        // Must use async memset so it's captured inside CUDA graphs.
        // cuMemsetD32_v2 is synchronous (runs on stream 0, not captured).
        unsafe {
            let stream = *self.device.cu_stream();
            let err = cuda_sys::lib().cuMemsetD32Async(output_ptr, 0, n, stream);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuMemsetD32Async for int8 fused v2: {:?}", err));
            }
        }

        let fused_kernel = self.device.get_func(MODULE_NAME, "marlin_gemv_int8_v2_fused_f32")
            .ok_or("marlin_gemv_int8_v2_fused_f32 kernel not found")?;
        unsafe {
            fused_kernel.launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (n_tiles as u32, k_splits as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem_bytes,
                },
                (w.ptr, w.scales_ptr, input_ptr, output_ptr,
                 self.perm_inv_weight_int8, self.perm_inv_scale,
                 k as i32, n as i32, w.group_size as i32, k_splits as i32),
            ).map_err(|e| format!("marlin_gemv_int8_v2_fused_f32 launch: {:?}", e))?;
        }
        Ok(())
    }

    /// Compute k_splits for Marlin GEMV v2 given N and K dimensions.
    fn compute_k_splits(&self, n: usize, k: usize) -> usize {
        let k_tiles = k / 16;
        let max_ksplits = k_tiles / 16;
        if max_ksplits <= 1 { return 1; }
        let n_tiles = (n + 15) / 16;
        let target = self.v2_num_sms * 4;
        let desired = (target + n_tiles - 1) / n_tiles;
        desired.clamp(1, max_ksplits.min(8))
    }

    /// Launch Marlin INT4 GEMV v2 with BF16 output (for attention projections).
    /// Uses K-splitting for better SM utilization, writes partials to v2_partial_ptr,
    /// then reduces to BF16 output.
    fn launch_marlin_gemv_int4_bf16(
        &self, w: &GpuWeight, input_ptr: u64, output_ptr: u64,
    ) -> Result<(), String> {
        let n = w.rows;
        let k = w.cols;
        let k_splits = self.compute_k_splits(n, k);

        if k_splits <= 1 {
            // Fall back to v1 for tiny matrices where splitting doesn't help
            let n_tiles = (n + 15) / 16;
            let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4) as u32;
            let kernel = self.device.get_func(MODULE_NAME, "marlin_gemv_int4")
                .ok_or("marlin_gemv_int4 kernel not found")?;
            unsafe {
                kernel.launch(
                    cudarc::driver::LaunchConfig {
                        grid_dim: (n_tiles as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: smem_bytes,
                    },
                    (w.ptr, w.scales_ptr, input_ptr, output_ptr,
                     self.perm_inv_weight_int4, self.perm_inv_scale,
                     k as i32, n as i32, w.group_size as i32),
                ).map_err(|e| format!("marlin_gemv_int4 launch: {:?}", e))?;
            }
            return Ok(());
        }

        let n_tiles = (n + 15) / 16;
        // v2 shared mem: input BF16 [K*2] + inv_wperm [1024*4] + inv_sperm [64*4] + reduce [16*16*4]
        let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4 + 16 * 16 * 4) as u32;
        let partial_ptr = self.v2_partial_ptr;

        // v2 GEMV: writes FP32 partials [k_splits, N]
        let v2_kernel = self.device.get_func(MODULE_NAME, "marlin_gemv_int4_v2")
            .ok_or("marlin_gemv_int4_v2 kernel not found")?;
        unsafe {
            v2_kernel.launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (n_tiles as u32, k_splits as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem_bytes,
                },
                (w.ptr, w.scales_ptr, input_ptr, partial_ptr,
                 self.perm_inv_weight_int4, self.perm_inv_scale,
                 k as i32, n as i32, w.group_size as i32, k_splits as i32),
            ).map_err(|e| format!("marlin_gemv_int4_v2 attn launch: {:?}", e))?;
        }

        // Reduce partials to BF16 output
        let reduce_kernel = self.device.get_func(MODULE_NAME, "reduce_ksplits_bf16")
            .ok_or("reduce_ksplits_bf16 kernel not found")?;
        unsafe {
            reduce_kernel.launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (((n + 255) / 256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (output_ptr, partial_ptr, n as i32, k_splits as i32),
            ).map_err(|e| format!("reduce_ksplits_bf16 attn launch: {:?}", e))?;
        }
        Ok(())
    }

    /// Launch Marlin INT4 GEMV with FP32 output (for attention score paths).
    /// Uses fused v2 kernel with inline atomic reduction (no separate reduce kernel).
    fn launch_marlin_gemv_int4_f32(
        &self, w: &GpuWeight, input_ptr: u64, output_ptr: u64,
    ) -> Result<(), String> {
        let n = w.rows;
        let k = w.cols;
        let k_splits = self.compute_k_splits(n, k);

        if k_splits <= 1 {
            let n_tiles = (n + 15) / 16;
            let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4) as u32;
            let kernel = self.device.get_func(MODULE_NAME, "marlin_gemv_int4_f32")
                .ok_or("marlin_gemv_int4_f32 kernel not found")?;
            unsafe {
                kernel.launch(
                    cudarc::driver::LaunchConfig {
                        grid_dim: (n_tiles as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: smem_bytes,
                    },
                    (w.ptr, w.scales_ptr, input_ptr, output_ptr,
                     self.perm_inv_weight_int4, self.perm_inv_scale,
                     k as i32, n as i32, w.group_size as i32),
                ).map_err(|e| format!("marlin_gemv_int4_f32 launch: {:?}", e))?;
            }
            return Ok(());
        }

        // Fused v2: zero output then atomicAdd directly (no separate reduce kernel)
        let n_tiles = (n + 15) / 16;
        let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4 + 16 * 16 * 4) as u32;

        // Must use async memset so it's captured inside CUDA graphs.
        // cuMemsetD32_v2 is synchronous (runs on stream 0, not captured).
        unsafe {
            let stream = *self.device.cu_stream();
            let err = cuda_sys::lib().cuMemsetD32Async(output_ptr, 0, n, stream);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuMemsetD32Async for fused v2: {:?}", err));
            }
        }

        let fused_kernel = self.device.get_func(MODULE_NAME, "marlin_gemv_int4_v2_fused_f32")
            .ok_or("marlin_gemv_int4_v2_fused_f32 kernel not found")?;
        unsafe {
            fused_kernel.launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (n_tiles as u32, k_splits as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem_bytes,
                },
                (w.ptr, w.scales_ptr, input_ptr, output_ptr,
                 self.perm_inv_weight_int4, self.perm_inv_scale,
                 k as i32, n as i32, w.group_size as i32, k_splits as i32),
            ).map_err(|e| format!("marlin_gemv_int4_v2_fused_f32 launch: {:?}", e))?;
        }
        Ok(())
    }

    /// Launch simple INT4 GEMV with BF16 output (for fast decode attention projections).
    /// Single kernel, no tile permutation overhead. Uses [rows, cols/2] u8 packed + FP32 scales.
    fn launch_simple_int4_gemv_bf16(
        &self, w: &GpuWeight, input_ptr: u64, output_ptr: u64,
    ) -> Result<(), String> {
        let rows = w.rows;
        let cols = w.cols;
        let gs = w.group_size;
        let blocks = ((rows + 7) / 8) as u32; // 8 warps per block, 1 row per warp
        let smem = (cols as u32) * 2; // BF16 input in shared memory

        let f = self.device.get_func(MODULE_NAME, "simple_int4_gemv_bf16")
            .ok_or("simple_int4_gemv_bf16 kernel not found")?;
        unsafe {
            f.launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem,
                },
                (w.simple_packed_ptr, w.simple_scales_f32_ptr, input_ptr, output_ptr,
                 rows as i32, cols as i32, gs as i32),
            ).map_err(|e| format!("simple_int4_gemv_bf16: {:?}", e))?;
        }
        Ok(())
    }

    /// Launch simple INT4 GEMV with FP32 output (for attention score paths).
    fn launch_simple_int4_gemv_f32(
        &self, w: &GpuWeight, input_ptr: u64, output_ptr: u64,
    ) -> Result<(), String> {
        let rows = w.rows;
        let cols = w.cols;
        let gs = w.group_size;
        let blocks = ((rows + 7) / 8) as u32;
        let smem = (cols as u32) * 2;

        let f = self.device.get_func(MODULE_NAME, "simple_int4_gemv_f32")
            .ok_or("simple_int4_gemv_f32 kernel not found")?;
        unsafe {
            f.launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem,
                },
                (w.simple_packed_ptr, w.simple_scales_f32_ptr, input_ptr, output_ptr,
                 rows as i32, cols as i32, gs as i32),
            ).map_err(|e| format!("simple_int4_gemv_f32: {:?}", e))?;
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
        is_int8: bool,
    ) -> PyResult<()> {
        let n_tiles = (n + 15) / 16;
        let smem_bytes = (k * 2 + 1024 * 4 + 64 * 4 + 16 * 16 * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_tiles as u32, k_splits as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem_bytes,
        };

        let kernel = if is_int8 {
            kernels.fused_silu_accum_v2_int8.clone()
        } else {
            kernels.fused_silu_accum_v2.clone()
        };

        unsafe {
            kernel.launch(cfg, (
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
        let intermediate = graph.moe_intermediate_size;
        let gs = graph.group_size;
        let w13_n = 2 * intermediate;
        let is_int8 = graph.expert_bits == 8;
        let inv_wp = if is_int8 {
            *graph.d_inv_weight_perm_int8.device_ptr()
        } else {
            *graph.d_inv_weight_perm.device_ptr()
        };
        let inv_sp = *graph.d_inv_scale_perm.device_ptr();
        let copy_stream = self.copy_stream.0;
        let default_stream: cuda_sys::CUstream = std::ptr::null_mut();

        // Check HCS first (fast flat lookup)
        let hcs_ptrs = if let Some(ref hcs) = graph.hcs {
            hcs.get_fast(layer_idx, expert_id).map(|(w13p, w13s, _w2p, _w2s)| (w13p, w13s))
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
            hs, w13_n, gs, is_int8,
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
        let intermediate = graph.moe_intermediate_size;
        let gs = graph.group_size;
        let topk = moe.topk;
        let ne = moe.num_experts;
        let sf = moe.scoring_func;
        let rsf = moe.routed_scaling_factor;
        let gate_wid = moe.gate_wid;
        let gate_bias_ptr = moe.gate_bias_ptr;
        let e_score_corr_ptr = moe.e_score_corr_ptr;
        let is_int8 = graph.expert_bits == 8;
        let act_type = moe.activation_type;
        let gated = moe.gated_experts;
        let inv_wp = if is_int8 {
            *graph.d_inv_weight_perm_int8.device_ptr()
        } else {
            *graph.d_inv_weight_perm.device_ptr()
        };
        let inv_sp = *graph.d_inv_scale_perm.device_ptr();

        // LatentMoE: expert input/output dimension may differ from hidden_size
        let expert_hs = if moe.moe_input_size > 0 { moe.moe_input_size } else { hs };
        // Expert w13 input pointer: d_scratch (latent) for LatentMoE, d_hidden for standard
        let expert_input_ptr = if graph.moe_input_override_ptr != 0 {
            graph.moe_input_override_ptr
        } else {
            *graph.d_hidden.device_ptr()
        };

        // v2 K-split config for w13 GEMV (only use v2 if k_splits > 1)
        // For ungated experts: w13_n = intermediate (up_proj only)
        // For gated experts: w13_n = 2 * intermediate (gate_proj + up_proj)
        let w13_n = if gated { 2 * intermediate } else { intermediate };
        let w13_k_tiles = expert_hs / 16;
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
        let t_gate_start = Instant::now();
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
        // Use pinned mapped memory if available — GPU writes directly to host-visible
        // memory, eliminating 2 cuMemcpyDtoH_v2 calls per layer (96/token → 0).
        let use_pinned = graph.pinned_topk_ids.is_some() && graph.pinned_topk_weights.is_some();
        let topk_ids_dptr = if use_pinned {
            graph.pinned_topk_ids.as_ref().unwrap().device_ptr
        } else {
            *graph.d_topk_indices.device_ptr()
        };
        let topk_wts_dptr = if use_pinned {
            graph.pinned_topk_weights.as_ref().unwrap().device_ptr
        } else {
            *graph.d_topk_weights.device_ptr()
        };
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
                        topk_ids_dptr,
                        topk_wts_dptr,
                        ne as i32,
                        topk as i32,
                    )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                        format!("sigmoid_topk: {:?}", e)))?;
                }
            } else {
                unsafe {
                    k.softmax_topk.clone().launch(cfg, (
                        logits_ptr,
                        topk_ids_dptr,
                        topk_wts_dptr,
                        ne as i32,
                        topk as i32,
                    )).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                        format!("softmax_topk: {:?}", e)))?;
                }
            }
        }

        // ── Step 4: Sync default stream only (not copy/prefetch streams) ──
        if timing { graph.t_moe_gate_gemv += (Instant::now() - t_gate_start).as_secs_f64(); }
        let t_route_start = Instant::now();
        unsafe {
            let err = cuda_sys::lib().cuStreamSynchronize(std::ptr::null_mut());
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("route stream sync: {:?}", err)));
            }
        }
        let t_after_route_sync = Instant::now();
        if timing { graph.t_moe_route_sync += (t_after_route_sync - t_route_start).as_secs_f64(); }

        #[cfg(feature = "gpu-debug")]
        let t_route = t_start.elapsed().as_secs_f64() * 1000.0;

        // Read topk results: either zero-copy from pinned memory or D2H copy
        let t_d2h_start = Instant::now();
        if use_pinned {
            // Zero-copy: GPU already wrote to host-visible pinned memory.
            // After sync, values are visible on host. Just copy from pinned → h_topk arrays.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    graph.pinned_topk_ids.as_ref().unwrap().host_ptr as *const i32,
                    graph.h_topk_ids.as_mut_ptr(),
                    topk,
                );
                std::ptr::copy_nonoverlapping(
                    graph.pinned_topk_weights.as_ref().unwrap().host_ptr as *const f32,
                    graph.h_topk_weights.as_mut_ptr(),
                    topk,
                );
            }
        } else {
            // Fallback: explicit D2H copy
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
        }
        if timing { graph.t_moe_d2h_topk += (Instant::now() - t_d2h_start).as_secs_f64(); }

        // ── Step 4.5a: Process PENDING spec results from previous layer ──
        //
        // The previous layer's moe_forward queued spec GEMV+topk on spec_stream.
        // That work has been executing in parallel with everything since then
        // (residual add, attention, norm). Now sync spec_stream, D2H the topk
        // IDs, and queue prefetch DMAs so Phase 1 can find them.
        let apfl_enabled = graph.apfl.as_ref().map_or(false, |a| a.enabled);
        let t_apfl_start = Instant::now();

        if apfl_enabled {
            let apfl = graph.apfl.as_mut().unwrap();
            if apfl.pending_prefetch {
                let pending_layer = apfl.pending_next_layer;
                let pending_count = apfl.pending_spec_count;
                apfl.pending_prefetch = false;

                // Sync spec_stream — should be long done (spec work is ~0.02ms,
                // and we've had the entire previous layer's expert loop + attention
                // + norm since it was queued).
                unsafe {
                    let err = cuda_sys::lib().cuStreamSynchronize(self.spec_stream.0);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            format!("APFL spec_stream sync: {:?}", err)));
                    }
                }

                // D2H: speculative topk indices (spec_stream is done, data is valid)
                if apfl.h_spec_topk_ids.len() < pending_count {
                    apfl.h_spec_topk_ids.resize(pending_count, 0);
                }
                unsafe {
                    let err = cuda_sys::lib().cuMemcpyDtoH_v2(
                        apfl.h_spec_topk_ids.as_mut_ptr() as *mut std::ffi::c_void,
                        *graph.d_topk_indices.device_ptr(),
                        pending_count * 4);
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            format!("APFL D2H spec_topk: {:?}", err)));
                    }
                }

                // Queue prefetch DMAs for the predicted experts
                if pending_layer < graph.moe_layers.len() {
                    if let Some(ref next_moe) = graph.moe_layers[pending_layer] {
                        let next_experts = &next_moe.experts;
                        for s in 0..pending_count {
                            let pred_eid = apfl.h_spec_topk_ids[s];
                            if pred_eid < 0 || pred_eid as usize >= next_experts.len() { continue; }
                            let pred_eid = pred_eid as usize;

                            if apfl.find_slot(pending_layer, pred_eid).is_some() { continue; }

                            if let Some(ref hcs) = graph.hcs {
                                if hcs.get_fast(pending_layer, pred_eid).is_some() { continue; }
                            }

                            let slot_idx = apfl.find_evict_slot(layer_idx);
                            let slot = &mut apfl.slots[slot_idx];
                            let pred_expert = &next_experts[pred_eid];
                            let slot_base = *slot.d_buf.device_ptr();

                            unsafe {
                                if pred_expert.contiguous_ptr != 0 {
                                    let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                        slot_base, pred_expert.contiguous_ptr as *const std::ffi::c_void,
                                        pred_expert.contiguous_bytes, prefetch_stream);
                                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                                        log::warn!("APFL DMA contiguous[{}] failed: {:?}", pred_eid, err);
                                        continue;
                                    }
                                } else {
                                    let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                        slot_base + slot.w13_packed_offset as u64,
                                        pred_expert.w13_packed_ptr as *const std::ffi::c_void,
                                        pred_expert.w13_packed_bytes, prefetch_stream);
                                    if err != cuda_sys::CUresult::CUDA_SUCCESS { continue; }
                                    let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                        slot_base + slot.w13_scales_offset as u64,
                                        pred_expert.w13_scales_ptr as *const std::ffi::c_void,
                                        pred_expert.w13_scales_bytes, prefetch_stream);
                                    if err != cuda_sys::CUresult::CUDA_SUCCESS { continue; }
                                    let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                        slot_base + slot.w2_packed_offset as u64,
                                        pred_expert.w2_packed_ptr as *const std::ffi::c_void,
                                        pred_expert.w2_packed_bytes, prefetch_stream);
                                    if err != cuda_sys::CUresult::CUDA_SUCCESS { continue; }
                                    let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                        slot_base + slot.w2_scales_offset as u64,
                                        pred_expert.w2_scales_ptr as *const std::ffi::c_void,
                                        pred_expert.w2_scales_bytes, prefetch_stream);
                                    if err != cuda_sys::CUresult::CUDA_SUCCESS { continue; }
                                }
                                cuda_sys::lib().cuEventRecord(slot.dma_event.0, prefetch_stream);
                            }

                            slot.layer_idx = pending_layer as i32;
                            slot.expert_idx = pred_eid as i32;
                            slot.dma_queued = true;
                            slot.w13_packed_size = pred_expert.w13_packed_bytes;
                            slot.w13_scales_size = pred_expert.w13_scales_bytes;
                            slot.w2_packed_size = pred_expert.w2_packed_bytes;
                            slot.w2_scales_size = pred_expert.w2_scales_bytes;
                        }
                    }
                }
            }
        }

        // ── Step 4.5b: Queue speculative routing for NEXT layer on spec_stream ──
        //
        // Spec GEMV + topk run on the dedicated spec_stream with its own cuBLAS
        // handle. This is non-blocking: the GPU executes spec work on spec_stream
        // in parallel with the expert loop on the default stream. Results are
        // consumed at the start of the NEXT layer's moe_forward call.
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

                let prefetch_count = graph.apfl.as_ref().unwrap().prefetch_count;
                let spec_topk = prefetch_count.min(next_topk * 2).min(graph.max_experts_per_tok);

                if spec_topk > 0 {
                    // Spec gate GEMV on spec_stream (via dedicated cuBLAS handle).
                    let output_ptr = unsafe {
                        (*graph.d_fp32_scratch.device_ptr() as *const f32).add(hs) as u64
                    };
                    {
                        let w = &graph.weights[next_gate_wid];
                        let alpha: f32 = 1.0;
                        let beta: f32 = 0.0;
                        unsafe {
                            cublas_result::gemm_ex(
                                self.spec_blas_handle.0,
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

                    // Spec topk on spec_stream via raw cuLaunchKernel.
                    let spec_logits_ptr = output_ptr;
                    let smem = (next_ne as u32) * 4;
                    let spec_stream_raw = self.spec_stream.0;
                    unsafe {
                        if next_sf == 1 {
                            let mut p0 = spec_logits_ptr;
                            let mut p1 = next_gate_bias;
                            let mut p2 = next_e_score_corr;
                            let mut p3 = *graph.d_topk_indices.device_ptr();
                            let mut p4 = *graph.d_topk_weights.device_ptr();
                            let mut p5 = next_ne as i32;
                            let mut p6 = spec_topk as i32;
                            let mut params: [*mut std::ffi::c_void; 7] = [
                                &mut p0 as *mut _ as *mut std::ffi::c_void,
                                &mut p1 as *mut _ as *mut std::ffi::c_void,
                                &mut p2 as *mut _ as *mut std::ffi::c_void,
                                &mut p3 as *mut _ as *mut std::ffi::c_void,
                                &mut p4 as *mut _ as *mut std::ffi::c_void,
                                &mut p5 as *mut _ as *mut std::ffi::c_void,
                                &mut p6 as *mut _ as *mut std::ffi::c_void,
                            ];
                            let err = cuda_sys::lib().cuLaunchKernel(
                                self.raw_sigmoid_topk.0,
                                1, 1, 1, 1, 1, 1, smem,
                                spec_stream_raw,
                                params.as_mut_ptr(),
                                std::ptr::null_mut(),
                            );
                            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                    format!("APFL spec sigmoid_topk launch: {:?}", err)));
                            }
                        } else {
                            let mut p0 = spec_logits_ptr;
                            let mut p1 = *graph.d_topk_indices.device_ptr();
                            let mut p2 = *graph.d_topk_weights.device_ptr();
                            let mut p3 = next_ne as i32;
                            let mut p4 = spec_topk as i32;
                            let mut params: [*mut std::ffi::c_void; 5] = [
                                &mut p0 as *mut _ as *mut std::ffi::c_void,
                                &mut p1 as *mut _ as *mut std::ffi::c_void,
                                &mut p2 as *mut _ as *mut std::ffi::c_void,
                                &mut p3 as *mut _ as *mut std::ffi::c_void,
                                &mut p4 as *mut _ as *mut std::ffi::c_void,
                            ];
                            let err = cuda_sys::lib().cuLaunchKernel(
                                self.raw_softmax_topk.0,
                                1, 1, 1, 1, 1, 1, smem,
                                spec_stream_raw,
                                params.as_mut_ptr(),
                                std::ptr::null_mut(),
                            );
                            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                    format!("APFL spec softmax_topk launch: {:?}", err)));
                            }
                        }
                    }

                    // Mark pending — results consumed at start of next layer's moe_forward
                    let apfl = graph.apfl.as_mut().unwrap();
                    apfl.pending_prefetch = true;
                    apfl.pending_next_layer = next_layer;
                    apfl.pending_spec_count = spec_topk;
                }
            }
        }

        // ── Step 5: Zero d_moe_out accumulator ──
        {
            unsafe {
                // For LatentMoE: only zero expert_hs elements (latent_size, not hidden_size)
                k.zero_bf16.clone().launch(
                    LaunchConfig::for_num_elems(expert_hs as u32),
                    (*graph.d_moe_out.device_ptr(), expert_hs as i32),
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

        if timing { graph.t_moe_apfl += (Instant::now() - t_apfl_start).as_secs_f64(); }
        let t_expert_loop_start = Instant::now();

        // ── Phase 1: Classify experts as HCS-hit or DMA-needed ──
        let mut hcs_batch_count = 0usize;
        let mut dma_experts: [(usize, f32); 10] = [(0, 0.0); 10]; // (expert_idx_in_topk, weight)
        let mut dma_count = 0usize;

        for i in 0..topk {
            let eid = graph.h_topk_ids[i];
            if eid < 0 { continue; }
            let eid = eid as usize;
            let weight = graph.h_topk_weights[i];

            // Record activation for heatmap collection (no-op when not collecting)
            if let Some(ref mut hcs) = graph.hcs {
                hcs.record_activation(layer_idx, eid);
            }

            // Check HCS cache (fast flat array lookup)
            let hcs_ptrs = if let Some(ref hcs) = graph.hcs {
                hcs.get_fast(layer_idx, eid)
            } else {
                None
            };

            if let Some((w13p, w13s, w2p, w2s)) = hcs_ptrs {
                // Gather HCS expert pointers for batched launch
                if hcs_batch_count < graph.max_experts_per_tok {
                    graph.h_batch_w13_packed_ptrs[hcs_batch_count] = w13p;
                    graph.h_batch_w13_scales_ptrs[hcs_batch_count] = w13s;
                    graph.h_batch_w2_packed_ptrs[hcs_batch_count] = w2p;
                    graph.h_batch_w2_scales_ptrs[hcs_batch_count] = w2s;
                    graph.h_batch_weights[hcs_batch_count] = weight;
                    hcs_batch_count += 1;
                    hcs_hits += 1;
                    graph.dma_hcs_experts += 1;
                }
            } else {
                // Check APFL slots before marking as cold (DMA-needed).
                // APFL prefetches for this layer were queued by the previous layer's processing.
                // The device.synchronize() above guarantees those DMAs are complete.
                let apfl_ptrs = if let Some(ref apfl) = graph.apfl {
                    if apfl.enabled {
                        apfl.find_slot(layer_idx, eid).and_then(|slot_idx| {
                            let slot = &apfl.slots[slot_idx];
                            if slot.dma_queued {
                                // Verify DMA is complete (should always be true after device sync)
                                let result = unsafe {
                                    cuda_sys::lib().cuEventQuery(slot.dma_event.0)
                                };
                                if result == cuda_sys::CUresult::CUDA_SUCCESS {
                                    Some((slot.w13_packed_ptr(), slot.w13_scales_ptr(),
                                          slot.w2_packed_ptr(), slot.w2_scales_ptr()))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some((w13p, w13s, w2p, w2s)) = apfl_ptrs {
                    // APFL hit: expert is already in VRAM from speculative prefetch
                    if hcs_batch_count < graph.max_experts_per_tok {
                        graph.h_batch_w13_packed_ptrs[hcs_batch_count] = w13p;
                        graph.h_batch_w13_scales_ptrs[hcs_batch_count] = w13s;
                        graph.h_batch_w2_packed_ptrs[hcs_batch_count] = w2p;
                        graph.h_batch_w2_scales_ptrs[hcs_batch_count] = w2s;
                        graph.h_batch_weights[hcs_batch_count] = weight;
                        hcs_batch_count += 1;
                        apfl_hits += 1;
                    }
                } else {
                    // Cold expert: needs DMA from CPU RAM
                    if dma_count < 10 {
                        let token_idx = graph.validation_decode_steps as usize + 1;
                        validation_record_cold_load(
                            &mut graph.validation_decode_cold_hist,
                            &mut graph.validation_decode_cold_events,
                            layer_idx,
                            eid,
                            token_idx,
                        );
                        dma_experts[dma_count] = (i, weight);
                        dma_count += 1;
                    }
                }
            }
        }

        // ── Pre-queue first 2 cold expert DMAs for DMA/compute overlap ──
        // Start DMA on copy_stream before Phase 2 so the PCIe copy engine works
        // while GPU SMs run HCS batch compute.  This hides up to 2 cold expert
        // transfers (~134 us) behind the HCS+shared compute window (~96 us).
        let pre_dma_count = if use_double_buf && dma_count > 0 { dma_count.min(2) } else { 0 };
        for di in 0..pre_dma_count {
            let (i, _weight) = dma_experts[di];
            let eid = graph.h_topk_ids[i] as usize;
            let expert = &moe.experts[eid];
            let slot = di;
            let base = buf_base[slot];
            unsafe {
                if expert.contiguous_ptr != 0 {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        base, expert.contiguous_ptr as *const std::ffi::c_void,
                        expert.contiguous_bytes, copy_stream);
                } else {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        base + w13p_off as u64, expert.w13_packed_ptr as *const std::ffi::c_void,
                        expert.w13_packed_bytes, copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        base + w13s_off as u64, expert.w13_scales_ptr as *const std::ffi::c_void,
                        expert.w13_scales_bytes, copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        base + w2p_off as u64, expert.w2_packed_ptr as *const std::ffi::c_void,
                        expert.w2_packed_bytes, copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        base + w2s_off as u64, expert.w2_scales_ptr as *const std::ffi::c_void,
                        expert.w2_scales_bytes, copy_stream);
                }
                cuda_sys::lib().cuEventRecord(ev_dma[slot], copy_stream);
            }
        }

        // ── Phase 2: Batched HCS expert compute (4 launches instead of 3*N) ──
        // Runs on default_stream while pre-queued cold DMAs proceed on copy_stream.
        if hcs_batch_count > 0 && use_v2_w13 && hcs_batch_count >= 2 {
            let t_w13 = Instant::now();

            // Pack all pointer arrays + weights into contiguous host buffer, then single H2D
            let max_ept = graph.max_experts_per_tok;
            let ptr_stride = max_ept * 8; // bytes per pointer array section
            unsafe {
                let h = graph.h_batch_upload.as_mut_ptr();
                // Layout: [w13_packed_ptrs | w13_scales_ptrs | w2_packed_ptrs | w2_scales_ptrs | weights]
                std::ptr::copy_nonoverlapping(
                    graph.h_batch_w13_packed_ptrs.as_ptr() as *const u8, h, hcs_batch_count * 8);
                std::ptr::copy_nonoverlapping(
                    graph.h_batch_w13_scales_ptrs.as_ptr() as *const u8, h.add(ptr_stride), hcs_batch_count * 8);
                std::ptr::copy_nonoverlapping(
                    graph.h_batch_w2_packed_ptrs.as_ptr() as *const u8, h.add(ptr_stride * 2), hcs_batch_count * 8);
                std::ptr::copy_nonoverlapping(
                    graph.h_batch_w2_scales_ptrs.as_ptr() as *const u8, h.add(ptr_stride * 3), hcs_batch_count * 8);
                std::ptr::copy_nonoverlapping(
                    graph.h_batch_weights.as_ptr() as *const u8, h.add(ptr_stride * 4), hcs_batch_count * 4);

                // Single H2D copy
                let upload_bytes = ptr_stride * 4 + max_ept * 4;
                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    *graph.d_batch_upload.device_ptr(),
                    h as *const std::ffi::c_void,
                    upload_bytes, std::ptr::null_mut());
            }

            // Device pointers into the contiguous upload buffer
            let d_upload_base = *graph.d_batch_upload.device_ptr();
            let d_w13p = d_upload_base;
            let d_w13s = d_upload_base + ptr_stride as u64;
            let d_w2p = d_upload_base + (ptr_stride * 2) as u64;
            let d_w2s = d_upload_base + (ptr_stride * 3) as u64;
            let d_wts = d_upload_base + (ptr_stride * 4) as u64;

            // Batched w13 GEMV v2: all HCS experts in one launch
            let w13_n_tiles = (w13_n + 15) / 16;
            let w13_smem = (expert_hs * 2 + 1024 * 4 + 64 * 4 + 16 * 16 * 4) as u32;
            let w13_kernel = if is_int8 {
                k.marlin_gemv_int8_v2_batched.clone()
            } else {
                k.marlin_gemv_int4_v2_batched.clone()
            };
            unsafe {
                w13_kernel.launch(
                    LaunchConfig {
                        grid_dim: (w13_n_tiles as u32, w13_ksplits as u32, hcs_batch_count as u32),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: w13_smem,
                    },
                    (
                        d_w13p,
                        d_w13s,
                        expert_input_ptr,
                        *graph.d_batch_partials.device_ptr(),
                        inv_wp, inv_sp,
                        expert_hs as i32, w13_n as i32, gs as i32, w13_ksplits as i32,
                    ),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("batched w13 v2: {:?}", e)))?;
            }

            // Batched reduce: all experts' k-split partials → gate_up
            unsafe {
                k.reduce_ksplits_bf16_batched.clone().launch(
                    LaunchConfig {
                        grid_dim: (((w13_n + 255) / 256) as u32, 1, hcs_batch_count as u32),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        *graph.d_batch_gate_ups.device_ptr(),
                        *graph.d_batch_partials.device_ptr(),
                        w13_n as i32, w13_ksplits as i32,
                    ),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("batched reduce: {:?}", e)))?;
            }

            if timing {
                unsafe { cuda_sys::lib().cuStreamSynchronize(std::ptr::null_mut()); }
                graph.t_expert_w13 += (Instant::now() - t_w13).as_secs_f64();
            }
            let t_silu_w2 = Instant::now();

            // Batched activation+w2 GEMV: all experts in one launch
            // For relu2 (act_type=1): use relu2_w2 kernel (ungated, relu^2 activation)
            // For silu (act_type=0): use fused_silu_w2 kernel (gated, SiLU*gate activation)
            let w2_n_tiles = (expert_hs + 15) / 16;
            let w2_smem = (intermediate * 2 + 1024 * 4 + 64 * 4) as u32;
            let w2_kernel = if act_type == 1 {
                // relu2: ungated experts
                if is_int8 { k.relu2_w2_int8_batched.clone() }
                else { k.relu2_w2_batched.clone() }
            } else {
                // silu: gated experts
                if is_int8 { k.fused_silu_w2_int8_batched.clone() }
                else { k.fused_silu_w2_batched.clone() }
            };
            unsafe {
                w2_kernel.launch(
                    LaunchConfig {
                        grid_dim: (w2_n_tiles as u32, 1, hcs_batch_count as u32),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: w2_smem,
                    },
                    (
                        d_w2p,
                        d_w2s,
                        *graph.d_batch_gate_ups.device_ptr(),
                        *graph.d_batch_expert_outs.device_ptr(),
                        inv_wp, inv_sp,
                        intermediate as i32, expert_hs as i32, gs as i32,
                    ),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("batched w2: {:?}", e)))?;
            }

            // Multi-expert weighted add: sum all expert outputs into moe_out
            unsafe {
                k.multi_expert_weighted_add_bf16.clone().launch(
                    LaunchConfig {
                        grid_dim: (((expert_hs + 255) / 256) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        *graph.d_moe_out.device_ptr(),
                        *graph.d_batch_expert_outs.device_ptr(),
                        d_wts,
                        expert_hs as i32, hcs_batch_count as i32,
                    ),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("multi_expert_weighted_add: {:?}", e)))?;
            }

            if timing {
                unsafe { cuda_sys::lib().cuStreamSynchronize(std::ptr::null_mut()); }
                graph.t_expert_silu_w2 += (Instant::now() - t_silu_w2).as_secs_f64();
            }
        } else if hcs_batch_count > 0 {
            // Fallback for 1 HCS expert or non-v2: sequential launch (original code)
            for bi in 0..hcs_batch_count {
                let w13p = graph.h_batch_w13_packed_ptrs[bi];
                let w13s = graph.h_batch_w13_scales_ptrs[bi];
                let w2p = graph.h_batch_w2_packed_ptrs[bi];
                let w2s = graph.h_batch_w2_scales_ptrs[bi];
                let weight = graph.h_batch_weights[bi];

                if use_v2_w13 {
                    self.launch_marlin_gemv_v2(
                        w13p, w13s,
                        expert_input_ptr,
                        partial_ptr, inv_wp, inv_sp,
                        expert_hs, w13_n, gs, w13_ksplits, k, is_int8,
                    )?;
                    self.launch_reduce_ksplits_bf16(
                        *graph.d_expert_gate_up.device_ptr(),
                        partial_ptr,
                        w13_n, w13_ksplits, k,
                    )?;
                } else {
                    self.launch_marlin_gemv_raw(
                        w13p, w13s,
                        expert_input_ptr,
                        *graph.d_expert_gate_up.device_ptr(),
                        inv_wp, inv_sp,
                        expert_hs, w13_n, gs, is_int8,
                    )?;
                }
                self.launch_fused_silu_accum(
                    w2p, w2s,
                    *graph.d_expert_gate_up.device_ptr(),
                    *graph.d_moe_out.device_ptr(),
                    inv_wp, inv_sp,
                    intermediate, expert_hs, gs,
                    weight, 0u64, k, is_int8,
                )?;
            }
        }

        // ── Phase 3: Cold expert compute (DMA/compute overlap for pre-queued experts) ──
        // Experts 0..pre_dma_count already have DMAs in flight from before Phase 2.
        // Their ev_dma events may already be signaled, giving near-zero wait.
        for di in 0..dma_count {
            let (i, _weight) = dma_experts[di];
            let eid = graph.h_topk_ids[i] as usize;
            let weight = graph.h_topk_weights[i];
            let expert = &moe.experts[eid];

            if use_double_buf {
                apfl_misses += 1;
                graph.dma_cold_experts += 1;
                if timing {
                    let dma_bytes = expert.w13_packed_bytes + expert.w13_scales_bytes
                                  + expert.w2_packed_bytes + expert.w2_scales_bytes;
                    graph.dma_bytes_total += dma_bytes as u64;
                    graph.dma_call_count += if expert.contiguous_ptr != 0 { 1 } else { 4 };
                }

                let slot = (dma_expert_count % 2) as usize;

                if di < pre_dma_count {
                    // DMA already queued before Phase 2 — skip DMA, just wait + compute
                } else {
                    // Wait for this buffer's previous compute to finish (free the buffer)
                    if dma_expert_count >= 2 {
                        unsafe {
                            cuda_sys::lib().cuStreamWaitEvent(copy_stream, ev_compute[slot], 0);
                        }
                    }

                    unsafe {
                        let base = buf_base[slot];
                        if expert.contiguous_ptr != 0 {
                            let err = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                base, expert.contiguous_ptr as *const std::ffi::c_void,
                                expert.contiguous_bytes, copy_stream);
                            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                    format!("DMA contiguous[{}]: {:?}", eid, err)));
                            }
                        } else {
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
                        }
                        cuda_sys::lib().cuEventRecord(ev_dma[slot], copy_stream);
                    }
                }

                // Wait for THIS expert's DMA to complete before computing
                // GPU-side only: default_stream waits for copy_stream's DMA event.
                // No host sync here -- that would serialize DMA and compute, destroying overlap.
                let t_dma_wait = Instant::now();
                unsafe {
                    cuda_sys::lib().cuStreamWaitEvent(default_stream, ev_dma[slot], 0);
                }
                if timing {
                    // Sync to measure actual DMA wait (serializes for measurement only)
                    unsafe { cuda_sys::lib().cuStreamSynchronize(default_stream); }
                    graph.t_dma_expert_wait += (Instant::now() - t_dma_wait).as_secs_f64();
                }

                // Compute from buf[slot]
                let t_dma_compute = Instant::now();
                let base = buf_base[slot];
                // w13 GEMV: expert_input -> gate_up (v2 K-split if beneficial)
                if use_v2_w13 {
                    self.launch_marlin_gemv_v2(
                        base + w13p_off as u64, base + w13s_off as u64,
                        expert_input_ptr,
                        partial_ptr, inv_wp, inv_sp,
                        expert_hs, w13_n, gs, w13_ksplits, k, is_int8,
                    )?;
                    self.launch_reduce_ksplits_bf16(
                        *graph.d_expert_gate_up.device_ptr(),
                        partial_ptr,
                        w13_n, w13_ksplits, k,
                    )?;
                } else {
                    self.launch_marlin_gemv_raw(
                        base + w13p_off as u64, base + w13s_off as u64,
                        expert_input_ptr,
                        *graph.d_expert_gate_up.device_ptr(),
                        inv_wp, inv_sp,
                        expert_hs, w13_n, gs, is_int8,
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

                // Fused: activation + w2 GEMV + weighted_add (3 launches -> 1)
                self.launch_fused_silu_accum(
                    base + w2p_off as u64, base + w2s_off as u64,
                    *graph.d_expert_gate_up.device_ptr(),
                    *graph.d_moe_out.device_ptr(),
                    inv_wp, inv_sp,
                    intermediate, expert_hs, gs,
                    weight, 0u64,
                    k, is_int8,
                )?;
                if timing {
                    unsafe { cuda_sys::lib().cuStreamSynchronize(default_stream); }
                    graph.t_dma_expert_compute += (Instant::now() - t_dma_compute).as_secs_f64();
                }

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
                graph.dma_cold_experts += 1;
                if timing {
                    let dma_bytes = expert.w13_packed_bytes + expert.w13_scales_bytes
                                  + expert.w2_packed_bytes + expert.w2_scales_bytes;
                    graph.dma_bytes_total += dma_bytes as u64;
                    graph.dma_call_count += 4;
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

                // w13 GEMV: expert_input -> gate_up (v2 K-split if beneficial)
                if use_v2_w13 {
                    self.launch_marlin_gemv_v2(
                        buf_w13_packed, buf_w13_scales,
                        expert_input_ptr,
                        partial_ptr, inv_wp, inv_sp,
                        expert_hs, w13_n, gs, w13_ksplits, k, is_int8,
                    )?;
                    self.launch_reduce_ksplits_bf16(
                        *graph.d_expert_gate_up.device_ptr(),
                        partial_ptr,
                        w13_n, w13_ksplits, k,
                    )?;
                } else {
                    self.launch_marlin_gemv_raw(
                        buf_w13_packed, buf_w13_scales,
                        expert_input_ptr,
                        *graph.d_expert_gate_up.device_ptr(),
                        inv_wp, inv_sp,
                        expert_hs, w13_n, gs, is_int8,
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

                // Fused: activation + w2 GEMV + weighted_add (3 launches -> 1)
                self.launch_fused_silu_accum(
                    buf_w2_packed, buf_w2_scales,
                    *graph.d_expert_gate_up.device_ptr(),
                    *graph.d_moe_out.device_ptr(),
                    inv_wp, inv_sp,
                    intermediate, expert_hs, gs,
                    weight, 0u64,
                    k, is_int8,
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
                hs, 2 * intermediate, gs, is_int8,
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
                k, is_int8,
            )?;
        }

        // ── Step 8: Scale by routed_scaling_factor ──
        if rsf != 1.0 {
            unsafe {
                k.scale_bf16.clone().launch(
                    LaunchConfig::for_num_elems(expert_hs as u32),
                    (
                        *graph.d_moe_out.device_ptr(),
                        *graph.d_moe_out.device_ptr(),
                        rsf,
                        expert_hs as i32,
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
                store.gpu_num_bits, intermediate_size,
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

        for moe_idx in 0..n_moe_layers {
            // Map MoE layer index to absolute layer index
            let abs_layer_idx = config.moe_abs_layer(moe_idx);

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
                graph.weights.push(GpuWeight::new(
                    *d_gate.device_ptr(), n_experts, hidden_size, 0, // BF16
                ));
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
                    // BF16 diagnostic: verify CPU data at the pointer
                    if expert.num_bits == 16 && stderr_debug_enabled() {
                        let cpu_data = unsafe { std::slice::from_raw_parts(w13p_ptr as *const u16, 8) };
                        let vals: Vec<f32> = cpu_data.iter().map(|&v| half::bf16::from_bits(v).to_f32()).collect();
                        eprintln!("[BF16-DMA] Expert[0][0] CPU w13p first8={:.4?} (ptr=0x{:x}, bytes={})",
                            vals, w13p_ptr, w13p_bytes);
                    }
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

            // e_score_corr_ptr: 0 — not stored in KrasisEngine (only used by sigmoid routing models).
            // shared_gate_wid: None — wired by Python via set_moe_shared_gate_wid() after this call.
            self.register_moe_layer_data(
                abs_layer_idx, expert_ptrs, shared_ptrs,
                n_experts, topk, scoring_func, norm_topk_prob,
                config.routed_scaling_factor, gate_wid, gate_bias_ptr as usize,
                0, None,
            )?;

            // Set contiguous_ptr on ExpertDataPtrs for experts with contiguous backing.
            // This enables single-call DMA (1 cuMemcpyHtoDAsync instead of 4 per expert).
            if let Some(ref mut graph) = self.graph {
                if let Some(ref mut moe_layer) = graph.moe_layers[abs_layer_idx] {
                    for (eidx, expert) in gpu_experts.iter().enumerate() {
                        if let Some(ref backing) = expert.contiguous_backing {
                            moe_layer.experts[eidx].contiguous_ptr = backing.as_ptr() as usize;
                            moe_layer.experts[eidx].contiguous_bytes = backing.len();
                        }
                    }
                }
            }
        }

        // Step 3: size expert DMA buffers (need to hold largest packed + scales)
        // We use buf_a for packed data, buf_b for scales data.
        // The largest packed buffer is max_expert_bytes.
        // Add 20% headroom for alignment.
        let buf_size = ((max_expert_bytes as f64) * 1.2) as usize;
        self.resize_expert_buffers(buf_size.max(1024))?;

        // Step 4: Pin expert weight memory for async DMA (page-lock for full PCIe bandwidth)
        // Without pinning, CUDA must bounce through a staging buffer, halving effective bandwidth.
        // NOTE: We use DEFAULT flag (0), NOT DEVICEMAP (0x02). Benchmarking showed DEVICEMAP
        // causes a 30% DMA regression even when not using mapped reads.
        //
        // With per-layer backing, we pin 4 buffers per layer instead of N per-expert buffers.
        // This reduces cuMemHostRegister calls from ~24K (QCN: 512*48) to ~192 (4*48).
        let t_pin = std::time::Instant::now();
        let mut pinned_regions = 0usize;
        let mut pinned_bytes = 0usize;
        let mut pin_failures = 0usize;

        // Pin per-layer backing buffers (routed experts)
        for moe_idx in 0..n_moe_layers {
            if moe_idx < store.layer_backings_gpu.len() {
                // Per-layer backing: pin 4 contiguous buffers per layer
                let backing = &store.layer_backings_gpu[moe_idx];
                let regions: [(&[u8], &str); 4] = [
                    (&backing.w13_packed, "w13p"),
                    (&backing.w13_scales, "w13s"),
                    (&backing.w2_packed, "w2p"),
                    (&backing.w2_scales, "w2s"),
                ];
                for (buf, label) in &regions {
                    if buf.is_empty() { continue; }
                    let err = unsafe {
                        cuda_sys::lib().cuMemHostRegister_v2(
                            buf.as_ptr() as *mut std::ffi::c_void,
                            buf.len(),
                            0, // CU_MEMHOSTREGISTER_DEFAULT
                        )
                    };
                    if err == cuda_sys::CUresult::CUDA_SUCCESS {
                        pinned_regions += 1;
                        pinned_bytes += buf.len();
                    } else {
                        pin_failures += 1;
                        if pin_failures == 1 {
                            log::warn!("First pin failure at moe_idx={} {}: {:?} (size={})",
                                moe_idx, label, err, buf.len());
                        }
                    }
                }
            } else {
                // Fallback for experts without per-layer backing (e.g. GGUF path):
                // pin each expert's contiguous backing or individual components
                let gpu_experts = &store.experts_gpu[moe_idx];
                for expert in gpu_experts.iter() {
                    if let Some(ref backing) = expert.contiguous_backing {
                        let err = unsafe {
                            cuda_sys::lib().cuMemHostRegister_v2(
                                backing.as_ptr() as *mut std::ffi::c_void,
                                backing.len(),
                                0,
                            )
                        };
                        if err == cuda_sys::CUresult::CUDA_SUCCESS {
                            pinned_regions += 1;
                            pinned_bytes += backing.len();
                        } else {
                            pin_failures += 1;
                        }
                    } else if !expert.borrowed {
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
                    // borrowed experts: skip — their data is already pinned via layer backing
                }
            }
            // Pin shared expert buffers (still per-expert, one per layer)
            if moe_idx < store.shared_experts_gpu.len() {
                let se = &store.shared_experts_gpu[moe_idx];
                if let Some(ref backing) = se.contiguous_backing {
                    let err = unsafe {
                        cuda_sys::lib().cuMemHostRegister_v2(
                            backing.as_ptr() as *mut std::ffi::c_void,
                            backing.len(),
                            0,
                        )
                    };
                    if err == cuda_sys::CUresult::CUDA_SUCCESS {
                        pinned_regions += 1;
                        pinned_bytes += backing.len();
                    } else {
                        pin_failures += 1;
                    }
                } else {
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
        }

        let pin_elapsed = t_pin.elapsed().as_secs_f64();
        log::info!(
            "Expert memory pinning: {} regions ({:.1} GB) pinned in {:.1}s, {} failures",
            pinned_regions, pinned_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            pin_elapsed, pin_failures,
        );

        // Initialize APFL (speculative prefetch) if requested via env var.
        // KRASIS_APFL_PREFETCH=N (default 0 = disabled). N = number of experts
        // to speculatively prefetch per layer. Spec routing runs on dedicated
        // spec_stream with its own cuBLAS handle, so it overlaps with the expert loop.
        let apfl_prefetch = std::env::var("KRASIS_APFL_PREFETCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        if apfl_prefetch > 0 {
            // Slots = 2x prefetch_count * layers per decode pass. For prefetch_count=1
            // with 48 MoE layers, 16 slots is plenty (we only need 1 per layer max,
            // and slots are evicted after consumption).
            let num_slots = (apfl_prefetch * 4).max(16);
            self.init_apfl(num_slots, apfl_prefetch)?;
        } else {
            log::info!("APFL: disabled (set KRASIS_APFL_PREFETCH=1..4 to enable)");
        }

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
            store.gpu_num_bits, intermediate_size,
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
            graph.weights.push(GpuWeight::new(
                *d_gate.device_ptr(), n_experts, hidden_size, 1, // FP32
            ));
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
        let num_moe_layers = config.num_moe_layers();
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
        prefetch_count: usize,
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
            store.gpu_num_bits, intermediate_size,
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
                graph.weights.push(GpuWeight::new(
                    *d_gate.device_ptr(), n_experts, hidden_size, 1,
                ));
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
        self.init_apfl(num_slots, prefetch_count)?;

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
                "Testing: {} MoE layers, hidden={}, APFL slots={}, prefetch_count={}",
                num_moe_layers, hs, apfl.slots.len(), apfl.prefetch_count,
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
        let num_layers = graph.moe_layers.len();
        let num_experts_per_layer = graph.moe_layers.iter()
            .filter_map(|m| m.as_ref())
            .map(|m| m.num_experts)
            .max()
            .unwrap_or(0);

        let mut hcs = HcsState::new();
        hcs.expert_vram_bytes = expert_vram_bytes;
        hcs.num_experts_per_layer = num_experts_per_layer;
        hcs.init_cache_fast(num_layers);
        hcs.init_gpu_expert_ptrs(&self.device, num_layers, num_experts_per_layer);

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

    /// Initialize pool-based HCS.
    fn hcs_pool_init_internal(
        &mut self,
        ranking: Vec<(usize, usize)>,
        budget_mb: usize,
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

        // Determine budget (0 = empty pool, used by tiered init for no-hard-pool case)
        let budget_bytes = budget_mb * 1024 * 1024;
        let num_slots = budget_bytes / slot_size;
        let pool_alloc_bytes = num_slots * slot_size;

        // Determine max experts per layer for bitset indexing
        let num_experts_per_layer = graph.moe_layers.iter()
            .filter_map(|m| m.as_ref())
            .map(|m| m.num_experts)
            .max()
            .unwrap_or(0);
        let num_layers = graph.moe_layers.len();
        // Total unique experts
        let total_experts: usize = graph.moe_layers.iter()
            .filter_map(|m| m.as_ref())
            .map(|m| m.num_experts)
            .sum();

        // Allocate the pool (skip if 0 budget — empty hard pool for tiered init)
        let (pool_buf_opt, pool_base) = if num_slots > 0 {
            log::info!("HCS pool: allocating {:.1} MB ({} slots x {:.1} KB/slot)",
                pool_alloc_bytes as f64 / (1024.0 * 1024.0),
                num_slots, slot_size as f64 / 1024.0);
            let buf = self.device.alloc_zeros::<u8>(pool_alloc_bytes)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("HCS pool alloc ({} MB): {:?}", pool_alloc_bytes / (1024 * 1024), e)))?;
            let base = *buf.device_ptr();
            (Some(buf), base)
        } else {
            log::info!("HCS pool: empty hard pool (0 budget), soft tier only");
            (None, 0u64)
        };

        let mut slot_to_expert: Vec<Option<(usize, usize)>> = vec![None; num_slots];

        // Two-pass allocation: select experts by ranking, then sort by (layer, expert)
        // so same-layer experts are physically contiguous in the pool for better L2 cache.
        let mut to_load: Vec<(usize, usize)> = Vec::new();
        for &(layer_idx, expert_idx) in &ranking {
            if to_load.len() >= num_slots {
                break;
            }
            let moe = match graph.moe_layers.get(layer_idx).and_then(|m| m.as_ref()) {
                Some(m) => m,
                None => continue,
            };
            if expert_idx >= moe.experts.len() {
                continue;
            }
            to_load.push((layer_idx, expert_idx));
        }
        // Sort by layer then expert so same-layer experts get adjacent slots
        to_load.sort_unstable();

        // Remaining slots after loading become the free list
        let mut next_slot = 0usize;

        // Fill slots from sorted list via H2D DMA
        let t0 = std::time::Instant::now();
        let mut loaded = 0usize;
        let mut cache = std::collections::HashMap::new();

        for &(layer_idx, expert_idx) in &to_load {
            let moe = &graph.moe_layers[layer_idx].as_ref().unwrap();
            let slot = next_slot;
            next_slot += 1;
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
                    slot_to_expert[slot] = None;
                    continue;
                }
            }

            // BF16 diagnostic: verify GPU data matches CPU for first expert
            if loaded == 0 && expert.w13_packed_bytes > 16 {
                let mut gpu_check = vec![0u16; 8];
                unsafe {
                    cuda_sys::lib().cuMemcpyDtoH_v2(
                        gpu_check.as_mut_ptr() as *mut _, dst + w13p_off, 16,
                    );
                }
                let gpu_vals: Vec<f32> = gpu_check.iter().map(|&v| half::bf16::from_bits(v).to_f32()).collect();
                let cpu_data = unsafe { std::slice::from_raw_parts(expert.w13_packed_ptr as *const u16, 8) };
                let cpu_vals: Vec<f32> = cpu_data.iter().map(|&v| half::bf16::from_bits(v).to_f32()).collect();
                if stderr_debug_enabled() {
                    eprintln!("[HCS-VERIFY] L{}E{} GPU first8={:.4?}", layer_idx, expert_idx, gpu_vals);
                    eprintln!("[HCS-VERIFY] L{}E{} CPU first8={:.4?}", layer_idx, expert_idx, cpu_vals);
                }
                let match_ok = gpu_vals == cpu_vals;
                if stderr_debug_enabled() {
                    eprintln!("[HCS-VERIFY] Match: {}", match_ok);
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
        hcs.num_experts_per_layer = num_experts_per_layer;
        hcs.init_cache_fast(num_layers);
        // Initialize GPU-side expert pointer table for CUDA graph support
        hcs.init_gpu_expert_ptrs(&self.device, num_layers, num_experts_per_layer);
        // Populate flat cache from loaded entries
        for (&(layer_idx, expert_idx), entry) in &cache {
            hcs.cache_fast_set(layer_idx, expert_idx, entry);
        }
        hcs.cache = cache;
        hcs.expert_vram_bytes = slot_size;
        hcs.vram_bytes = pool_alloc_bytes;
        hcs.num_cached = loaded;
        hcs.pool_buf = pool_buf_opt;
        hcs.pool_slot_size = slot_size;
        hcs.pool_num_slots = num_slots;
        // Build free slot stack from unassigned slots (above next_slot + any failed slots)
        let mut free_slots: Vec<usize> = (next_slot..num_slots).rev().collect();
        // Also reclaim any failed-DMA slots below next_slot
        for s in (0..next_slot).rev() {
            if slot_to_expert[s].is_none() {
                free_slots.push(s);
            }
        }
        hcs.pool_free_slots = free_slots;
        hcs.pool_slot_to_expert = slot_to_expert;

        // GPU-side route sync: disabled. Benchmarking showed zero speed gain (+0.4%,
        // within noise) because the classify kernel runs at end of graph segment — by the
        // time it writes to mapped memory, cuStreamSynchronize would have returned anyway.
        // The kernel also has persistent CUDA_ERROR_ILLEGAL_ADDRESS issues on first token.
        // Keeping d_expert_ptrs for HCS bookkeeping but not running the classify kernel.
        let gpu_rs = false;
        graph.hcs = Some(hcs);
        graph.gpu_route_sync = gpu_rs;

        // ── Populate mapped fallback + d_expert_ptrs for cold experts ──
        // If experts have mapped device pointers, fill d_expert_ptrs for ALL uncached experts
        // so the GPU can read cold expert data directly via PCIe without CPU intervention.
        {
            let hcs = graph.hcs.as_mut().unwrap();
            let total_flat = num_layers * num_experts_per_layer;
            let mut fallback = vec![[0u64; 4]; total_flat];
            let mut fully_mapped = 0usize;

            for (layer_idx, moe_opt) in graph.moe_layers.iter().enumerate() {
                if let Some(ref moe) = moe_opt {
                    for (eidx, expert) in moe.experts.iter().enumerate() {
                        if expert.mapped_w13_packed_dptr != 0 && expert.mapped_w13_scales_dptr != 0 &&
                           expert.mapped_w2_packed_dptr != 0 && expert.mapped_w2_scales_dptr != 0 {
                            let flat_idx = layer_idx * num_experts_per_layer + eidx;
                            if flat_idx < total_flat {
                                fallback[flat_idx] = [
                                    expert.mapped_w13_packed_dptr,
                                    expert.mapped_w13_scales_dptr,
                                    expert.mapped_w2_packed_dptr,
                                    expert.mapped_w2_scales_dptr,
                                ];
                                fully_mapped += 1;
                            }
                        }
                    }
                }
            }

            hcs.mapped_fallback = fallback;
            hcs.mapped_reads_available = fully_mapped == total_experts;

            if hcs.mapped_reads_available {
                // Upload mapped pointers for all cold (uncached) experts to d_expert_ptrs
                if let Some(ref d_ptrs_buf) = hcs.d_expert_ptrs {
                    let mut cold_uploaded = 0usize;
                    for layer_idx in 0..num_layers {
                        for eidx in 0..num_experts_per_layer {
                            // If this expert is NOT in the HCS cache, set its mapped pointer
                            let is_cached = {
                                let flat = layer_idx * num_experts_per_layer + eidx;
                                flat < hcs.cache_fast.len() && hcs.cache_fast[flat][0] != 0
                            };
                            if !is_cached {
                                let flat = layer_idx * num_experts_per_layer + eidx;
                                let ptrs = hcs.mapped_fallback[flat];
                                if ptrs[0] != 0 {
                                    let gpu_idx = (layer_idx * hcs.d_expert_ptrs_ne + eidx) * 4;
                                    unsafe {
                                        let dst = *d_ptrs_buf.device_ptr() + (gpu_idx * 8) as u64;
                                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                            dst,
                                            ptrs.as_ptr() as *const std::ffi::c_void,
                                            32,
                                            std::ptr::null_mut(),
                                        );
                                    }
                                    cold_uploaded += 1;
                                }
                            }
                        }
                    }
                    log::info!(
                        "Mapped reads: uploaded {} cold expert mapped pointers to d_expert_ptrs",
                        cold_uploaded,
                    );
                }

                // Allocate mapped activation buffer: [num_moe_layers * topk] i32
                let topk_val = graph.moe_layers.iter()
                    .filter_map(|m| m.as_ref())
                    .map(|m| m.topk)
                    .max()
                    .unwrap_or(10);
                let num_moe = graph.moe_layers.iter().filter(|m| m.is_some()).count();
                let act_size = num_moe * topk_val * 4; // i32 per topk per MoE layer
                graph.mapped_activations = PinnedMapped::new(act_size).ok();

                // Mapped reads: disabled by default. Benchmarking showed 30% regression due to
                // SM stalls during PCIe reads (Marlin GEMV access pattern generates many small
                // non-coalesced reads, effective bandwidth << 25 GB/s). The 2ms sync savings
                // from eliminating CPU round-trips is dwarfed by ~8.7ms of SM stall time.
                // Enable with KRASIS_MAPPED_READS=1 for testing.
                let mapped_reads_enabled = std::env::var("KRASIS_MAPPED_READS").map(|v| v == "1").unwrap_or(false);
                graph.mapped_reads_active = mapped_reads_enabled;

                log::info!(
                    "Mapped reads {}: {}/{} experts mapped, GPU {}read cold experts via PCIe",
                    if mapped_reads_enabled { "ACTIVE" } else { "AVAILABLE (disabled, use KRASIS_MAPPED_READS=1)" },
                    fully_mapped, total_experts,
                    if mapped_reads_enabled { "will " } else { "can " },
                );
            } else {
                graph.mapped_reads_active = false;
                log::info!(
                    "Mapped reads NOT available: {}/{} experts mapped (need all)",
                    fully_mapped, total_experts,
                );
            }
        }

        let mapped_reads = graph.mapped_reads_active;
        let msg = format!(
            "HCS pool: {}/{} experts loaded in {:.2}s ({:.1}% coverage), {:.1} MB VRAM, \
             {} free slots, gpu_route_sync={}, mapped_reads={}",
            loaded, total_experts, load_elapsed, pct,
            pool_alloc_bytes as f64 / (1024.0 * 1024.0),
            num_slots - loaded, gpu_rs, mapped_reads,
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
        hcs.cache_fast_set(layer_idx, expert_idx, &entry);
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

        // Build sorted list from flat heatmap
        let nep = hcs.num_experts_per_layer;
        let has_data = nep > 0 && hcs.heatmap_flat.iter().any(|&v| v > 0);
        if !has_data && hcs.heatmap.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Heatmap is empty. Call hcs_start_collecting and run some tokens first."));
        }

        // Sort by activation count descending
        let mut sorted: Vec<((usize, usize), u64)> = if has_data {
            // Use flat heatmap
            hcs.heatmap_flat.iter().enumerate()
                .filter(|(_, &v)| v > 0)
                .map(|(idx, &v)| ((idx / nep, idx % nep), v))
                .collect()
        } else {
            // Fallback to HashMap
            hcs.heatmap.iter().map(|(&k, &v)| (k, v)).collect()
        };
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
            store.gpu_num_bits, intermediate_size,
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
                graph.weights.push(GpuWeight::new(
                    *d_gate.device_ptr(), n_experts, hidden_size, 1,
                ));
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
            store.gpu_num_bits, intermediate_size,
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
                graph.weights.push(GpuWeight::new(
                    *d_gate.device_ptr(), n_experts, hidden_size, 1,
                ));
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
            store.gpu_num_bits, intermediate_size,
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
                graph.weights.push(GpuWeight::new(
                    *d_gate.device_ptr(), n_experts, hidden_size, 1,
                ));
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
            let is_int8 = graph.expert_bits == 8;
            let inv_wp = if is_int8 {
                *graph.d_inv_weight_perm_int8.device_ptr()
            } else {
                *graph.d_inv_weight_perm.device_ptr()
            };
            let inv_sp = *graph.d_inv_scale_perm.device_ptr();
            let hs = graph.hidden_size;
            let intermediate = graph.moe_intermediate_size;
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
                    inv_wp, inv_sp, hs, 2 * intermediate, gs, is_int8,
                )?;
                self.launch_fused_silu_accum(
                    w2p, w2s,
                    *graph.d_expert_gate_up.device_ptr(),
                    *graph.d_moe_out.device_ptr(),
                    inv_wp, inv_sp,
                    intermediate, hs, gs, 0.1f32, 0u64, k, is_int8,
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
                    inv_wp, inv_sp, hs, 2 * intermediate, gs, is_int8,
                )?;
                self.launch_fused_silu_accum(
                    w2p, w2s,
                    *graph.d_expert_gate_up.device_ptr(),
                    *graph.d_moe_out.device_ptr(),
                    inv_wp, inv_sp,
                    intermediate, hs, gs, 0.1f32, 0u64, k, is_int8,
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
                    is_int8,
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
                    is_int8,
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
                    is_int8,
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
                    is_int8,
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
                        is_int8,
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
                        is_int8,
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
            let is_int8 = graph.expert_bits == 8;
            let inv_wp = if is_int8 {
                *graph.d_inv_weight_perm_int8.device_ptr()
            } else {
                *graph.d_inv_weight_perm.device_ptr()
            };
            let inv_sp = *graph.d_inv_scale_perm.device_ptr();
            let hs = graph.hidden_size;
            let intermediate = graph.moe_intermediate_size;
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
                            is_int8,
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
                            is_int8,
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
                            is_int8,
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
                            is_int8,
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
            if !self.spec_stream.0.is_null() {
                let _ = cuda_sys::lib().cuStreamDestroy_v2(self.spec_stream.0);
            }
            if !self.spec_blas_handle.0.is_null() {
                let _ = cublas_result::destroy_handle(self.spec_blas_handle.0);
            }
        }
    }
}
