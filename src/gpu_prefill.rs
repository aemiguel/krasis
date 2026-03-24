//! Rust GPU Prefill — replaces Python prefill entirely.
//!
//! This module handles the full prefill pipeline:
//!   1. Embedding lookup
//!   2. Layer-by-layer forward pass (streaming weights from CPU)
//!   3. Attention (GQA) via CUDA kernels
//!   4. MoE expert dispatch via Marlin GEMM
//!   5. Mamba2 SSM layers (conv1d + SSD scan)
//!   6. LM head projection + first token sampling
//!
//! Zero Python, zero GIL, zero PyTorch allocator.
//! All memory managed via cudarc (CUDA driver API).
//! Kernels loaded from PTX (simple ops) or dlopen'd from vendored libkrasis_marlin.so (Marlin).

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr};
use cudarc::driver::sys as cuda_sys;

/// PTX source for prefill kernels (compiled by build.rs).
#[cfg(has_prefill_kernels)]
const PREFILL_KERNELS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/prefill_kernels.ptx"));

// ════════════════════════════════════════════════════════════════════════
//  Raw CUfunction extraction (same pattern as gpu_decode.rs)
// ════════════════════════════════════════════════════════════════════════

/// Newtype for raw CUfunction handle.
#[derive(Clone, Copy)]
struct RawCuFunc(cuda_sys::CUfunction);

/// Extract raw CUfunction from cudarc's CudaFunction.
fn extract_cu_func(func: &CudaFunction) -> RawCuFunc {
    unsafe {
        let struct_ptr = func as *const _ as *const u8;
        let word0: cuda_sys::CUfunction = std::ptr::read(struct_ptr as *const _);
        let mut dummy = 0i32;
        let w0_valid = cuda_sys::lib().cuFuncGetAttribute(
            &mut dummy,
            cuda_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS,
            word0,
        ) == cuda_sys::CUresult::CUDA_SUCCESS;
        RawCuFunc(if w0_valid { word0 } else {
            std::ptr::read(struct_ptr.add(8) as *const _)
        })
    }
}

/// Launch a CUDA kernel with a parameter buffer.
unsafe fn launch(
    func: RawCuFunc,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    smem: u32,
    stream: cuda_sys::CUstream,
    params: &mut [*mut std::ffi::c_void],
) -> Result<(), String> {
    let err = cuda_sys::lib().cuLaunchKernel(
        func.0,
        grid.0, grid.1, grid.2,
        block.0, block.1, block.2,
        smem,
        stream,
        params.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    if err != cuda_sys::CUresult::CUDA_SUCCESS {
        Err(format!("Kernel launch failed: {:?} (grid={:?}, block={:?}, smem={}, nparams={}, func_ptr={:?})",
            err, grid, block, smem, params.len(), func.0))
    } else {
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════════
//  Kernel handles
// ════════════════════════════════════════════════════════════════════════

/// Handle to loaded prefill kernel functions (raw CUfunction handles).
pub struct PrefillKernels {
    rmsnorm: RawCuFunc,
    fused_add_rmsnorm: RawCuFunc,
    embedding: RawCuFunc,
    rope: RawCuFunc,
    silu_mul: RawCuFunc,
    relu2: RawCuFunc,
    sigmoid_mul: RawCuFunc,      // gated GQA: out = attn * sigmoid(gate)
    sigmoid_topk: RawCuFunc,
    softmax_topk: RawCuFunc,
    moe_sum_reduce: RawCuFunc,
    gqa_prefill: RawCuFunc,
    kv_cache_append: RawCuFunc,
    causal_conv1d: RawCuFunc,
    mamba2_ssd: RawCuFunc,
    mamba2_extract: RawCuFunc,
    moe_gather: RawCuFunc,
    moe_scatter_add: RawCuFunc,
    moe_zero_accum: RawCuFunc,
    moe_add_shared: RawCuFunc,
    moe_add_shared_gated: RawCuFunc,
    moe_accum_to_bf16: RawCuFunc,
    fp32_to_bf16_batch: RawCuFunc,

    // Linear attention kernels
    la_uninterleave_qkvz: RawCuFunc,
    la_uninterleave_ba: RawCuFunc,
    la_depthwise_conv1d_silu: RawCuFunc,
    la_l2norm_per_head: RawCuFunc,
    la_compute_gate_beta: RawCuFunc,
    la_repeat_interleave: RawCuFunc,
    la_gated_rmsnorm: RawCuFunc,
    la_cumsum: RawCuFunc,
    la_build_attn_matrix: RawCuFunc,
    la_triangular_solve: RawCuFunc,
    la_compute_v_new: RawCuFunc,
    la_chunk_output: RawCuFunc,
    la_state_update: RawCuFunc,
    la_scale_by_exp_g: RawCuFunc,
    la_apply_beta: RawCuFunc,
    la_bf16_to_fp32: RawCuFunc,
    la_fp32_to_bf16: RawCuFunc,
    la_transpose_f32: RawCuFunc,

    // New performance kernels
    transpose_3d_021: RawCuFunc,     // [A,B,C] -> [B,A,C] FP32
    transpose_3d_021_bf16: RawCuFunc, // [A,B,C] -> [B,A,C] BF16
    flash_attn_tiled: RawCuFunc,     // tiled FA with online softmax
    gated_q_split: RawCuFunc,        // split interleaved [M,H,2D] -> Q + gate
    la_split_conv_output: RawCuFunc, // split [M,conv_dim] -> q + k + v
    concat_3_bf16: RawCuFunc,       // concat q+k+v BF16 -> [M,conv_dim]
    la_compute_v_new_strided: RawCuFunc,  // strided v_new (zero-copy chunk)
    la_chunk_output_strided: RawCuFunc,   // strided chunk output (zero-copy)
    la_state_update_strided: RawCuFunc,   // strided state update (zero-copy)

    // GPU-only MoE routing kernels (eliminates CPU round-trip)
    moe_count_experts: RawCuFunc,
    moe_prefix_sum: RawCuFunc,
    moe_build_maps: RawCuFunc,
    // Fused MoE support kernels (for MarlinDefault integration)
    moe_padded_prefix_sum: RawCuFunc,
    moe_scatter_sorted: RawCuFunc,
    moe_gather_sorted: RawCuFunc,
    moe_scatter_fused: RawCuFunc,

    // Marlin GEMM functions (loaded via dlopen from vendored libkrasis_marlin.so)
    marlin_mm: Option<MarlinMmFn>,
    // Fused MoE: vendored MarlinDefault (dlopen from libkrasis_marlin.so)
    pub fused_moe_fn: Option<FusedMoeFn>,
}

/// Function pointer for marlin::marlin_mm<nv_bfloat16>.
type MarlinMmFn = unsafe extern "C" fn(
    *const std::ffi::c_void, *const std::ffi::c_void,
    *mut std::ffi::c_void, *mut std::ffi::c_void,
    *const std::ffi::c_void, *const std::ffi::c_void,
    *const std::ffi::c_void, *const std::ffi::c_void,
    *const std::ffi::c_void, *mut std::ffi::c_void,
    i32, i32, i32, i32,
    *mut std::ffi::c_void,
    *const ScalarType,
    bool, bool, bool, i32, i32, i32, u64,
    i32, i32, i32, bool, bool, bool,
);

/// Function pointer for vendored krasis_marlin_moe_mm_bf16.
/// Host function that handles template dispatch internally.
/// 38 parameters matching the extern "C" entry point.
type FusedMoeFn = unsafe extern "C" fn(
    /*A*/            *const std::ffi::c_void,
    /*B*/            *const std::ffi::c_void,
    /*C*/            *mut std::ffi::c_void,
    /*C_tmp*/        *mut std::ffi::c_void,
    /*b_bias*/       *const std::ffi::c_void,
    /*s (scales)*/   *const std::ffi::c_void,
    /*s2 (scale2)*/  *const std::ffi::c_void,
    /*zp (zeros)*/   *const std::ffi::c_void,
    /*g_idx*/        *const std::ffi::c_void,
    /*perm*/         *const std::ffi::c_void,
    /*a_tmp*/        *const std::ffi::c_void,
    /*sorted_ids*/   *const std::ffi::c_void,
    /*expert_ids*/   *const std::ffi::c_void,
    /*num_post*/     *const std::ffi::c_void,
    /*topk_wts*/     *const std::ffi::c_void,
    /*moe_blk_size*/ i32,
    /*top_k*/        i32,
    /*mul_topk_wts*/ bool,
    /*is_ep*/        bool,
    /*size_m*/       i32,
    /*size_n*/       i32,
    /*size_k*/       i32,
    /*workspace*/    *mut std::ffi::c_void,
    /*q_type_ptr*/   *const std::ffi::c_void,
    /*has_bias*/     bool,
    /*has_act_order*/ bool,
    /*is_k_full*/    bool,
    /*has_zp*/       bool,
    /*num_groups*/   i32,
    /*group_size*/   i32,
    /*dev*/          i32,
    /*stream_ptr*/   *mut std::ffi::c_void,
    /*thread_k*/     i32,
    /*thread_n*/     i32,
    /*sms*/          i32,
    /*use_atomic*/   bool,
    /*fp32_reduce*/  bool,
    /*is_zp_float*/  bool,
);

/// Matches sglang::ScalarType memory layout for FFI.
#[repr(C)]
pub struct ScalarType {
    pub exponent: u8,
    pub mantissa: u8,
    pub is_signed: u8,
    pub _pad0: u8,
    pub bias: i32,
    pub finite_values_only: u8,
    pub nan_repr: u8,
    pub _pad1: [u8; 2],
}

impl ScalarType {
    pub const U4B8: ScalarType = ScalarType {
        exponent: 0, mantissa: 4, is_signed: 0, _pad0: 0,
        bias: 8, finite_values_only: 0, nan_repr: 1, _pad1: [0; 2], // nan_repr=1 = NAN_IEEE_754
    };
    pub const U8B128: ScalarType = ScalarType {
        exponent: 0, mantissa: 8, is_signed: 0, _pad0: 0,
        bias: 128, finite_values_only: 0, nan_repr: 1, _pad1: [0; 2], // nan_repr=1 = NAN_IEEE_754
    };
}

// ════════════════════════════════════════════════════════════════════════
//  Model config, weights, scratch
// ════════════════════════════════════════════════════════════════════════

pub struct PrefillModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,      // max(moe_inter, dense_inter) for general scratch
    pub moe_intermediate_size: usize,  // actual MoE expert intermediate for MoE-specific buffers
    pub num_hidden_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub expert_bits: u8,
    pub shared_expert_bits: u8,
    pub group_size: usize,
    pub sms: usize,
    pub device_ordinal: usize,
    pub layer_types: Vec<u8>,       // 0=GQA, 1=mamba2, 2=moe_only, 3=linear_attn
    pub first_k_dense: usize,
    pub scoring_func: u8,           // 0=softmax, 1=sigmoid
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub moe_gated: bool,
    pub moe_activation: u8,         // 0=silu, 1=relu2
    pub mamba_d_inner: usize,
    pub mamba_d_state: usize,
    pub mamba_num_heads: usize,
    pub mamba_head_dim: usize,
    pub mamba_conv_dim: usize,
    pub mamba_conv_kernel: usize,
    pub mamba_n_groups: usize,
    pub tie_word_embeddings: bool,
    // Linear attention (Gated DeltaNet) config
    pub la_num_k_heads: usize,
    pub la_num_v_heads: usize,
    pub la_k_head_dim: usize,
    pub la_v_head_dim: usize,
    pub la_head_ratio: usize,
    pub la_conv_kernel_dim: usize,
    pub la_conv_dim: usize,      // key_dim * 2 + value_dim
    pub la_scale: f32,
    pub la_chunk_size: usize,
    pub rope_half_dim: usize,       // actual RoPE half dim (may differ from head_dim/2 for partial rotary)
    pub prefill_chunk_size: usize,  // max tokens per chunk (0 = use max_tokens)
    pub layer_group_size: usize,    // how many MoE layers per group for expert DMA (0 = no grouping)
}

pub struct MarlinWeight {
    pub packed: u64,
    pub scales: u64,
    pub n: usize,
    pub k: usize,
    pub num_groups: usize,
    pub group_size: usize,
    pub num_bits: u8,
}

/// BF16 weight for cuBLAS GEMM (used when attention_quant="bf16").
pub struct Bf16Weight {
    pub ptr: u64,    // BF16 data on GPU [n, k] column-major (transposed)
    pub n: usize,    // output dim
    pub k: usize,    // input dim
}

pub struct PrefillLayerWeights {
    pub input_norm: u64,
    pub post_attn_norm: u64,    // 0 = None
    pub q_proj: Option<MarlinWeight>,
    pub k_proj: Option<MarlinWeight>,
    pub v_proj: Option<MarlinWeight>,
    pub o_proj: Option<MarlinWeight>,
    pub q_proj_bf16: Option<Bf16Weight>,
    pub k_proj_bf16: Option<Bf16Weight>,
    pub v_proj_bf16: Option<Bf16Weight>,
    pub o_proj_bf16: Option<Bf16Weight>,
    pub gqa_gated: bool,        // QCN: q_proj outputs [query, gate], apply sigmoid(gate) before o_proj
    pub mamba2_in_proj: Option<MarlinWeight>,
    pub mamba2_out_proj: Option<MarlinWeight>,
    pub mamba2_conv_weight: u64,    // 0 = None
    pub mamba2_conv_bias: u64,
    #[allow(non_snake_case)]
    pub mamba2_A: u64,
    #[allow(non_snake_case)]
    pub mamba2_D: u64,
    pub mamba2_dt_bias: u64,
    pub mamba2_norm: u64,
    pub moe_gate_ptr: u64,          // BF16 gate weight on GPU [hidden, n_experts]
    pub moe_gate_rows: usize,       // n_experts
    pub moe_gate_cols: usize,       // hidden_size
    pub moe_gate_bias_ptr: u64,     // 0 = no bias
    pub moe_e_score_corr_ptr: u64,  // 0 = none
    pub moe_num_experts: usize,
    pub moe_topk: usize,
    pub moe_scoring_func: u8,       // 0=softmax, 1=sigmoid
    pub moe_norm_topk_prob: bool,
    pub moe_routed_scaling_factor: f32,
    pub moe_gated: bool,
    pub moe_activation: u8,         // 0=silu, 1=relu2
    pub shared_w1: Option<MarlinWeight>,
    pub shared_w2: Option<MarlinWeight>,
    /// Shared expert gate weight ptr (BF16, [hidden, 1]). 0 = no gate (weight=1.0).
    pub shared_gate_ptr: u64,
    pub shared_gate_rows: usize,
    pub shared_gate_cols: usize,
    pub layer_type: u8,
    /// MoE layer index in the MoeLayerData array (for expert data lookup).
    pub moe_layer_idx: Option<usize>,
    // Linear attention weights (Marlin or BF16)
    pub la_in_proj_qkvz: Option<MarlinWeight>,
    pub la_in_proj_ba: Option<MarlinWeight>,
    pub la_out_proj: Option<MarlinWeight>,
    pub la_in_proj_qkvz_bf16: Option<Bf16Weight>,
    pub la_in_proj_ba_bf16: Option<Bf16Weight>,
    pub la_out_proj_bf16: Option<Bf16Weight>,
    pub la_conv_weight_ptr: u64,     // [conv_dim, kernel_dim] FP32
    pub la_a_log_ptr: u64,           // [nv] FP32
    pub la_dt_bias_ptr: u64,         // [nv] FP32
    pub la_norm_weight_ptr: u64,     // [dv] BF16
    pub la_conv_state_ptr: u64,      // [conv_dim, kernel_dim] FP32 (decode state)
    pub la_recur_state_ptr: u64,     // [nv, dk, dv] FP32 (decode state)
}

/// Expert weight pointers for DMA to GPU (mirrors ExpertDataPtr).
pub struct ExpertWeightPtrs {
    pub w13_packed_ptr: usize,
    pub w13_packed_bytes: usize,
    pub w13_scales_ptr: usize,
    pub w13_scales_bytes: usize,
    pub w2_packed_ptr: usize,
    pub w2_packed_bytes: usize,
    pub w2_scales_ptr: usize,
    pub w2_scales_bytes: usize,
    pub contiguous_ptr: usize,
    pub contiguous_bytes: usize,
}

/// Per-MoE-layer expert data for prefill DMA.
pub struct PrefillMoeLayerData {
    pub experts: Vec<ExpertWeightPtrs>,
    pub shared: Option<ExpertWeightPtrs>,
}

pub struct PrefillScratch {
    pub d_hidden: CudaSlice<u16>,
    pub d_residual: CudaSlice<u16>,
    pub d_scratch1: CudaSlice<u16>,
    pub d_scratch2: CudaSlice<u16>,
    pub d_fp32_scratch: CudaSlice<f32>,
    pub d_workspace: CudaSlice<i32>,
    pub d_topk_weights: CudaSlice<f32>,
    pub d_topk_ids: CudaSlice<i32>,
    pub d_token_ids: CudaSlice<i32>,
    pub d_positions: CudaSlice<i32>,
    pub d_attn_out: CudaSlice<u16>,
    pub d_q: CudaSlice<u16>,
    pub d_k: CudaSlice<u16>,
    pub d_v: CudaSlice<u16>,
    pub d_logits: CudaSlice<f32>,
    pub d_mamba2_conv_state: Option<CudaSlice<f32>>,
    pub d_mamba2_ssm_state: Option<CudaSlice<f32>>,
    pub d_mamba2_in_proj: Option<CudaSlice<u16>>,
    pub d_mamba2_ssd_out: Option<CudaSlice<u16>>,
    // MoE scratch buffers
    pub d_gate_out: CudaSlice<f32>,         // [max_tokens, max_experts] FP32
    pub d_moe_accum: CudaSlice<f32>,        // [max_tokens, hidden] FP32 (for atomic scatter)
    pub d_moe_gathered: CudaSlice<u16>,     // [max_tokens * topk, hidden] BF16
    pub d_moe_expert_out: CudaSlice<u16>,   // [max_tokens * topk, hidden] BF16
    pub d_moe_gate_up: CudaSlice<u16>,      // [max_tokens * topk, 2*inter] BF16
    pub d_moe_inter: CudaSlice<u16>,        // [max_tokens * topk, inter] BF16
    pub d_gather_src_map: CudaSlice<i32>,   // [max_tokens * topk]
    pub d_gather_weight_map: CudaSlice<f32>,// [max_tokens * topk]
    // GPU-only routing scratch
    pub d_expert_counts: CudaSlice<i32>,    // [max_experts]
    pub d_expert_offsets: CudaSlice<i32>,   // [max_experts + 1]
    pub d_write_offsets: CudaSlice<i32>,    // [max_experts]
    // Expert weight DMA double-buffers (A and B for ping-pong DMA/compute overlap)
    pub d_expert_w13_packed_a: CudaSlice<u8>,
    pub d_expert_w13_scales_a: CudaSlice<u8>,
    pub d_expert_w2_packed_a: CudaSlice<u8>,
    pub d_expert_w2_scales_a: CudaSlice<u8>,
    pub d_expert_w13_packed_b: CudaSlice<u8>,
    pub d_expert_w13_scales_b: CudaSlice<u8>,
    pub d_expert_w2_packed_b: CudaSlice<u8>,
    pub d_expert_w2_scales_b: CudaSlice<u8>,
    // Linear attention scratch buffers
    pub d_la_q: Option<CudaSlice<f32>>,       // [max_tokens, nv, dk] FP32
    pub d_la_k: Option<CudaSlice<f32>>,       // [max_tokens, nv, dk] FP32
    pub d_la_v: Option<CudaSlice<f32>>,       // [max_tokens, nv, dv] FP32
    pub d_la_z: Option<CudaSlice<u16>>,       // [max_tokens, nv, dv] BF16 (gate)
    pub d_la_b: Option<CudaSlice<u16>>,       // [max_tokens, nv] BF16
    pub d_la_a: Option<CudaSlice<u16>>,       // [max_tokens, nv] BF16
    pub d_la_beta: Option<CudaSlice<f32>>,    // [max_tokens, nv] FP32
    pub d_la_gate: Option<CudaSlice<f32>>,    // [max_tokens, nv] FP32
    pub d_la_conv_out: Option<CudaSlice<f32>>, // [conv_dim, max_tokens] FP32
    pub d_la_v_beta: Option<CudaSlice<f32>>,  // [nv, max_tokens, dv] FP32
    pub d_la_k_beta: Option<CudaSlice<f32>>,  // [nv, max_tokens, dk] FP32
    pub d_la_v_new: Option<CudaSlice<f32>>,   // [nv, chunk_size, dv] FP32 (per chunk)
    pub d_la_g_cum: Option<CudaSlice<f32>>,   // [nv, num_chunks, chunk_size] FP32
    pub d_la_attn: Option<CudaSlice<f32>>,    // [nv, num_chunks, CS, CS] FP32
    pub d_la_state: Option<CudaSlice<f32>>,   // [nv, dk, dv] FP32 (recurrent state)
    pub d_la_chunk_out: Option<CudaSlice<f32>>, // [nv, chunk_size, dv] FP32
    pub d_la_q_contig: Option<CudaSlice<f32>>, // [nv, chunk_size, dk] FP32 (separate from chunk_out to avoid aliasing)
    pub d_la_proj_buf: Option<CudaSlice<u16>>, // projection buffer [max_tokens, proj_dim] BF16
    pub max_tokens: usize,
}

pub struct PrefillResult {
    pub first_token: u32,
    pub prompt_len: usize,
    pub prefill_time_ms: f64,
}

/// Per-position top-k logprob data from prefill logit extraction.
pub struct PrefillLogitPosition {
    pub position: usize,
    pub top_k: Vec<(usize, f32)>, // (token_id, logprob)
}

// ════════════════════════════════════════════════════════════════════════
//  Prefill Engine
// ════════════════════════════════════════════════════════════════════════

pub struct PrefillEngine {
    pub kernels: PrefillKernels,
    pub config: PrefillModelConfig,
    pub scratch: PrefillScratch,
    pub layer_weights: Vec<PrefillLayerWeights>,
    pub moe_layers: Vec<Option<PrefillMoeLayerData>>,
    pub embedding_ptr: u64,
    pub final_norm_ptr: u64,
    pub lm_head: Option<MarlinWeight>,
    pub lm_head_bf16_ptr: u64,   // BF16 weight pointer (when lm_head is None)
    pub lm_head_bf16_rows: usize, // vocab_size
    pub lm_head_bf16_cols: usize, // hidden_size
    pub rope_cos_ptr: u64,
    pub rope_sin_ptr: u64,
    pub kv_k_ptrs: Vec<u64>,       // per-layer K cache pointers (FP8)
    pub kv_v_ptrs: Vec<u64>,       // per-layer V cache pointers (FP8)
    pub kv_max_seq: usize,
    pub stream: cuda_sys::CUstream,
    pub copy_stream: cuda_sys::CUstream,
    pub cublas_handle: cudarc::cublas::sys::cublasHandle_t,
    pub h_logits: Vec<f32>,
    // Host staging buffers for routing results
    pub h_topk_ids: Vec<i32>,
    pub h_topk_weights: Vec<f32>,
    pub h_gather_src_map: Vec<i32>,
    pub h_gather_weight_map: Vec<f32>,
    // HCS snapshot: flat [num_layers * num_experts] -> [w13_packed, w13_scales, w2_packed, w2_scales]
    // Zero = not cached. Updated before each prefill from decode store's HCS state.
    pub hcs_cache_fast: Vec<[u64; 4]>,
    pub hcs_num_experts_per_layer: usize,
    // CUDA events for double-buffer expert DMA synchronization
    pub dma_event: cuda_sys::CUevent,
    pub compute_event: cuda_sys::CUevent,
    // BF16 attention weight streaming: double-buffer for q/k/v/o proj weights
    // These are only used when attention weights are BF16 (not AWQ/INT4).
    // For AWQ attention, weights are small and permanently GPU-resident.
    pub attn_weight_bufs: Option<AttnWeightStreamBufs>,
    // FlashAttention library function pointer (dlopen'd from flash_attn .so)
    // When available, replaces the tiled attention kernel for better performance.
    pub flash_attn_fn: Option<FlashAttnFn>,
    // Dedicated stream for shared expert (gap 4: always async regardless of cold DMA)
    pub shared_stream: cuda_sys::CUstream,
    pub shared_event: cuda_sys::CUevent,
    // Separate Marlin workspace for shared_stream (avoids d_fp32_scratch conflict)
    pub d_shared_fp32_scratch: CudaSlice<f32>,
    pub d_shared_workspace: CudaSlice<i32>,
    // Contiguous expert weight buffers for fused MoE (gap 1)
    // Two sets (A/B) for double-buffered layer-level preload (gap 2)
    pub d_fused_expert_w1_a: Option<CudaSlice<u8>>,  // [E, w1_packed_bytes_per_expert]
    pub d_fused_expert_w1s_a: Option<CudaSlice<u8>>,  // [E, w1_scales_bytes_per_expert]
    pub d_fused_expert_w2_a: Option<CudaSlice<u8>>,  // [E, w2_packed_bytes_per_expert]
    pub d_fused_expert_w2s_a: Option<CudaSlice<u8>>,  // [E, w2_scales_bytes_per_expert]
    pub d_fused_expert_w1_b: Option<CudaSlice<u8>>,
    pub d_fused_expert_w1s_b: Option<CudaSlice<u8>>,
    pub d_fused_expert_w2_b: Option<CudaSlice<u8>>,
    pub d_fused_expert_w2s_b: Option<CudaSlice<u8>>,
    pub fused_expert_buf_cur: usize,  // 0 = A, 1 = B
    // Per-expert byte sizes (computed from config)
    pub w1_packed_per_expert: usize,
    pub w1_scales_per_expert: usize,
    pub w2_packed_per_expert: usize,
    pub w2_scales_per_expert: usize,
    // Sorted MoE buffers for fused kernel
    pub d_sorted_token_ids: Option<CudaSlice<i32>>,   // [max_sorted]
    pub d_fused_expert_ids: Option<CudaSlice<i32>>,    // [max_blocks]
    pub d_num_tokens_post: Option<CudaSlice<i32>>,     // [1]
    pub d_fused_input: Option<CudaSlice<u16>>,         // [max_sorted, hidden] gathered BF16
    pub d_fused_inter_cache: Option<CudaSlice<u16>>,   // [max_sorted, 2*inter] intermediate BF16
    pub d_fused_inter2: Option<CudaSlice<u16>>,        // [max_sorted, inter] after activation BF16
    pub d_fused_output: Option<CudaSlice<u16>>,        // [max_sorted, hidden] output BF16
    pub d_fused_c_tmp: Option<CudaSlice<f32>>,         // FP32 reduce workspace
    pub max_sorted: usize,
    // Preloaded layer index tracking for group-level double-buffer (gap 2)
    pub preloaded_moe_layer: Option<usize>,
    // ScalarType for Marlin GEMM dispatch (matches weight quantization format)
    pub q_type: ScalarType,
}

/// Double-buffer BF16 attention weight streaming buffers.
/// Each buffer holds one layer's Q/K/V/O projection weights in BF16.
pub struct AttnWeightStreamBufs {
    pub buf_a: CudaSlice<u8>,   // Buffer A: holds one layer's attention weights
    pub buf_b: CudaSlice<u8>,   // Buffer B: double-buffer counterpart
    pub buf_size: usize,        // Size of each buffer in bytes
}

/// FlashAttention varlen forward function signature.
/// Matches flash_attn_varlen_func from the FlashAttention C API.
type FlashAttnFn = unsafe extern "C" fn(
    /*q*/ *const std::ffi::c_void,
    /*k*/ *const std::ffi::c_void,
    /*v*/ *const std::ffi::c_void,
    /*out*/ *mut std::ffi::c_void,
    /*cu_seqlens_q*/ *const i32,
    /*cu_seqlens_k*/ *const i32,
    /*max_seqlen_q*/ i32,
    /*max_seqlen_k*/ i32,
    /*softmax_scale*/ f32,
    /*is_causal*/ bool,
    /*num_heads*/ i32,
    /*num_heads_k*/ i32,
    /*head_dim*/ i32,
    /*batch_size*/ i32,
    /*stream*/ u64,
) -> i32;

// Safety: PrefillEngine contains raw CUDA pointers (CUstream, CUevent, CUfunction, cuBLAS handle).
// These are only accessed from the server's request handler which processes one request at a time
// (single-request guarantee). The raw pointers themselves are thread-safe CUDA handles.
unsafe impl Send for PrefillEngine {}
unsafe impl Sync for PrefillEngine {}

impl PrefillEngine {
    /// Update HCS snapshot from the decode store's current HCS state.
    /// Must be called before each prefill so we know which experts are GPU-resident.
    pub fn update_hcs_snapshot(&mut self, cache_fast: &[[u64; 4]], num_experts_per_layer: usize) {
        self.hcs_cache_fast = cache_fast.to_vec();
        self.hcs_num_experts_per_layer = num_experts_per_layer;
    }

    /// Check if an expert is GPU-resident in HCS.
    /// Returns Some((w13_packed, w13_scales, w2_packed, w2_scales)) if resident.
    fn hcs_lookup(&self, moe_layer_idx: usize, expert_idx: usize) -> Option<(u64, u64, u64, u64)> {
        if self.hcs_num_experts_per_layer == 0 { return None; }
        let idx = moe_layer_idx * self.hcs_num_experts_per_layer + expert_idx;
        if idx < self.hcs_cache_fast.len() {
            let ptrs = self.hcs_cache_fast[idx];
            if ptrs[0] != 0 {
                Some((ptrs[0], ptrs[1], ptrs[2], ptrs[3]))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Run the full prefill pipeline with token chunking.
    /// Long prompts are split into chunks to bound intermediate GPU memory usage.
    /// Ensure the CUDA primary context for our device is active on the calling thread.
    /// This is needed when run_prefill is called from a thread different from the one
    /// that created the engine (e.g. benchmark thread vs main server thread).
    fn bind_cuda_context(&self) -> Result<(), String> {
        unsafe {
            let mut ctx: cuda_sys::CUcontext = std::ptr::null_mut();
            let rc = cuda_sys::lib().cuDevicePrimaryCtxRetain(
                &mut ctx,
                self.config.device_ordinal as i32,
            );
            if rc != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuDevicePrimaryCtxRetain failed: {:?}", rc));
            }
            let rc = cuda_sys::lib().cuCtxSetCurrent(ctx);
            if rc != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuCtxSetCurrent failed: {:?}", rc));
            }
        }
        Ok(())
    }

    /// Download BF16 data from GPU and compute L2 norm of a single row (position).
    /// Used for diagnostic comparison against BF16 reference norms.
    fn diag_l2_norm(&self, ptr: u64, pos: usize, stride: usize, dim: usize) -> f32 {
        let offset_bytes = pos * stride * 2; // BF16 = 2 bytes
        let mut buf = vec![0u16; dim];
        unsafe {
            cuda_sys::lib().cuStreamSynchronize(self.stream);
            cuda_sys::lib().cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                ptr + offset_bytes as u64,
                (dim * 2) as usize,
            );
        }
        let mut sum = 0.0f64;
        for &v in &buf {
            let bits = v as u32;
            let f = f32::from_bits(bits << 16) as f64;
            sum += f * f;
        }
        (sum.sqrt()) as f32
    }

    /// Download FP32 data from GPU and compute L2 norm of a single row.
    fn diag_l2_norm_f32(&self, ptr: u64, pos: usize, stride: usize, dim: usize) -> f32 {
        let offset_bytes = pos * stride * 4; // FP32 = 4 bytes
        let mut buf = vec![0.0f32; dim];
        unsafe {
            cuda_sys::lib().cuStreamSynchronize(self.stream);
            cuda_sys::lib().cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                ptr + offset_bytes as u64,
                (dim * 4) as usize,
            );
        }
        let mut sum = 0.0f64;
        for &v in &buf {
            sum += (v as f64) * (v as f64);
        }
        (sum.sqrt()) as f32
    }

    /// Download FP32 values from GPU and return them.
    fn diag_download_f32(&self, ptr: u64, count: usize) -> Vec<f32> {
        let mut buf = vec![0.0f32; count];
        unsafe {
            cuda_sys::lib().cuStreamSynchronize(self.stream);
            cuda_sys::lib().cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                ptr,
                (count * 4) as usize,
            );
        }
        buf
    }

    /// Print L2 norms at multiple positions for a given buffer, in a format
    /// that can be compared against the BF16 sublayer reference data.
    fn diag_print_norms(&self, label: &str, ptr: u64, positions: &[usize], h: usize) {
        unsafe { cuda_sys::lib().cuStreamSynchronize(self.stream); }
        let mut parts = Vec::new();
        for &pos in positions {
            let norm = self.diag_l2_norm(ptr, pos, h, h);
            parts.push(format!("{}:{:.6}", pos, norm));
        }
        eprintln!("[DIAG] {} {}", label, parts.join(" "));
    }

    /// Parse reference positions from KRASIS_PREFILL_DIAG_POSITIONS env var,
    /// or use default positions matching the sublayer reference capture.
    fn diag_positions(m: usize) -> Vec<usize> {
        if let Ok(val) = std::env::var("KRASIS_PREFILL_DIAG_POSITIONS") {
            val.split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .filter(|&p| p < m)
                .collect()
        } else {
            // Default: match reference capture positions [0,1,2,3,4, last-4..last]
            let mut pos: Vec<usize> = (0..std::cmp::min(5, m)).collect();
            if m > 5 {
                for p in m.saturating_sub(5)..m {
                    if !pos.contains(&p) {
                        pos.push(p);
                    }
                }
            }
            pos
        }
    }

    /// Get the diagnostic layer limit from KRASIS_PREFILL_DIAG_LAYERS env var.
    /// Default: 1 (only layer 0). Set to e.g. 5 for layers 0-4, or 999 for all.
    fn diag_layer_limit() -> usize {
        std::env::var("KRASIS_PREFILL_DIAG_LAYERS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(1)
    }

    pub fn run_prefill(
        &mut self,
        token_ids: &[u32],
        temperature: f32,
        suppress_tokens: &[u32],
    ) -> Result<PrefillResult, String> {
        // Bind CUDA context to calling thread (needed for cross-thread use)
        self.bind_cuda_context()?;

        let t0 = Instant::now();
        let total_m = token_ids.len();
        let h = self.config.hidden_size;
        let num_hidden_layers = self.config.num_hidden_layers;
        if total_m == 0 {
            return Err("Empty token sequence".to_string());
        }

        // Diagnostic mode: compare per-layer norms against BF16 reference
        // Set KRASIS_PREFILL_DIAG=1 to enable.
        // KRASIS_PREFILL_DIAG_LAYERS=N to check N layers (default 1 = layer 0 only)
        // KRASIS_PREFILL_DIAG_POSITIONS=0,1,2,3,4,28,29,30,31,32 to set positions
        let diag = std::env::var("KRASIS_PREFILL_DIAG").is_ok();
        let diag_positions = if diag { Self::diag_positions(total_m) } else { vec![] };
        let diag_layer_limit = if diag { Self::diag_layer_limit() } else { 0 };

        // Determine chunk size: use config value, fall back to max_tokens
        let chunk_size = if self.config.prefill_chunk_size > 0 {
            self.config.prefill_chunk_size
        } else {
            self.scratch.max_tokens
        };

        // Process token chunks
        let num_chunks = (total_m + chunk_size - 1) / chunk_size;

        // Zero LA recurrent + conv state for fresh prefill (each prefill processes full prompt)
        if let Some(ref la_state_buf) = self.scratch.d_la_state {
            let nv = self.config.la_num_v_heads;
            let dk = self.config.la_k_head_dim;
            let dv = self.config.la_v_head_dim;
            let state_elems = nv * dk * dv;
            unsafe {
                cuda_sys::lib().cuMemsetD32Async(
                    *la_state_buf.device_ptr(), 0, state_elems, self.stream);
            }
        }
        for layer_idx in 0..num_hidden_layers {
            let lw = &self.layer_weights[layer_idx];
            if lw.la_conv_state_ptr != 0 {
                let conv_dim = self.config.la_conv_dim;
                let kernel_dim = self.config.la_conv_kernel_dim;
                unsafe {
                    cuda_sys::lib().cuMemsetD32Async(
                        lw.la_conv_state_ptr, 0, conv_dim * kernel_dim, self.stream);
                }
            }
            if lw.la_recur_state_ptr != 0 {
                let nv = self.config.la_num_v_heads;
                let dk = self.config.la_k_head_dim;
                let dv = self.config.la_v_head_dim;
                unsafe {
                    cuda_sys::lib().cuMemsetD32Async(
                        lw.la_recur_state_ptr, 0, nv * dk * dv, self.stream);
                }
            }
        }

        for chunk_idx in 0..num_chunks {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = std::cmp::min(chunk_start + chunk_size, total_m);
            let m = chunk_end - chunk_start;
            let chunk_tokens = &token_ids[chunk_start..chunk_end];

            if m > self.scratch.max_tokens {
                return Err(format!("Chunk {} tokens > scratch {}", m, self.scratch.max_tokens));
            }

            // 1. Upload token IDs and positions for this chunk
            self.upload_tokens_with_offset(chunk_tokens, chunk_start)
                .map_err(|e| format!("upload_tokens chunk {}: {}", chunk_idx, e))?;

            // 2. Embedding lookup
            self.launch_embedding(m)
                .map_err(|e| format!("embedding chunk {} m={}: {}", chunk_idx, m, e))?;

            if diag && chunk_idx == 0 {
                eprintln!("[DIAG] === Prefill diagnostic: m={} positions={:?} layers=0..{} ===",
                    m, diag_positions, diag_layer_limit);
                self.diag_print_norms("embedding",
                    *self.scratch.d_hidden.device_ptr(), &diag_positions, h);
            }

            // 3. Layer-by-layer forward pass
            let mut has_residual = false;
            for layer_idx in 0..num_hidden_layers {
                let layer_type = self.layer_weights[layer_idx].layer_type;

                // DIAG: check norm weight values for layer 0
                if diag && chunk_idx == 0 && layer_idx == 0 {
                    let norm_ptr = self.layer_weights[0].input_norm;
                    let mut norm_buf = vec![0u16; std::cmp::min(10, h)];
                    unsafe {
                        cuda_sys::lib().cuStreamSynchronize(self.stream);
                        cuda_sys::lib().cuMemcpyDtoH_v2(
                            norm_buf.as_mut_ptr() as *mut std::ffi::c_void,
                            norm_ptr,
                            (norm_buf.len() * 2) as usize,
                        );
                    }
                    let vals: Vec<f32> = norm_buf.iter().map(|&v| f32::from_bits((v as u32) << 16)).collect();
                    eprintln!("[DIAG] layer0 input_norm_weight ptr={:#x} first_10_vals={:?}", norm_ptr, vals);
                    let norm_l2 = self.diag_l2_norm(norm_ptr, 0, h, h);
                    eprintln!("[DIAG] layer0 input_norm_weight L2={:.6}", norm_l2);
                }

                // Pre-attention RMSNorm
                if !has_residual {
                    self.memcpy_d2d(
                        *self.scratch.d_residual.device_ptr(),
                        *self.scratch.d_hidden.device_ptr(),
                        (m * h * 2) as u64,
                    ).map_err(|e| format!("copy residual layer {}: {}", layer_idx, e))?;
                    self.launch_rmsnorm(
                        *self.scratch.d_hidden.device_ptr(),
                        *self.scratch.d_residual.device_ptr(),
                        self.layer_weights[layer_idx].input_norm,
                        m, h,
                    ).map_err(|e| format!("rmsnorm layer {}: {}", layer_idx, e))?;
                    has_residual = true;
                } else {
                    self.launch_fused_add_rmsnorm(
                        *self.scratch.d_residual.device_ptr(),
                        *self.scratch.d_hidden.device_ptr(),
                        self.layer_weights[layer_idx].input_norm,
                        m, h,
                    ).map_err(|e| format!("fused_add_rmsnorm layer {}: {}", layer_idx, e))?;
                }

                // Mixer
                match layer_type {
                    0 => self.forward_gqa_chunked(layer_idx, m, chunk_start)
                        .map_err(|e| format!("gqa layer {}: {}", layer_idx, e))?,
                    1 => self.forward_mamba2(layer_idx, m)
                        .map_err(|e| format!("mamba2 layer {}: {}", layer_idx, e))?,
                    2 => {
                        self.memcpy_d2d(
                            *self.scratch.d_attn_out.device_ptr(),
                            *self.scratch.d_hidden.device_ptr(),
                            (m * h * 2) as u64,
                        )?;
                    }
                    3 => self.forward_linear_attention(layer_idx, m)
                        .map_err(|e| format!("linear_attn layer {}: {}", layer_idx, e))?,
                    _ => {
                        self.memcpy_d2d(
                            *self.scratch.d_attn_out.device_ptr(),
                            *self.scratch.d_hidden.device_ptr(),
                            (m * h * 2) as u64,
                        )?;
                    }
                }

                if diag && chunk_idx == 0 && layer_idx < diag_layer_limit {
                    let lt_name = match layer_type { 0 => "gqa", 1 => "mamba2", 3 => "la", _ => "?" };
                    self.diag_print_norms(
                        &format!("layer{:02}_{}_mixer", layer_idx, lt_name),
                        *self.scratch.d_attn_out.device_ptr(), &diag_positions, h);
                }

                // Post-attention RMSNorm
                let post_norm = self.layer_weights[layer_idx].post_attn_norm;
                if post_norm != 0 {
                    self.launch_fused_add_rmsnorm(
                        *self.scratch.d_residual.device_ptr(),
                        *self.scratch.d_attn_out.device_ptr(),
                        post_norm,
                        m, h,
                    )?;
                    self.memcpy_d2d(
                        *self.scratch.d_hidden.device_ptr(),
                        *self.scratch.d_attn_out.device_ptr(),
                        (m * h * 2) as u64,
                    )?;
                } else {
                    self.memcpy_d2d(
                        *self.scratch.d_hidden.device_ptr(),
                        *self.scratch.d_attn_out.device_ptr(),
                        (m * h * 2) as u64,
                    )?;
                }

                // MLP (dense or MoE)
                if self.layer_weights[layer_idx].moe_gate_ptr != 0 {
                    if diag && layer_idx < diag_layer_limit {
                        eprintln!("[DIAG] layer{:02} -> forward_moe (gate_ptr=0x{:x})",
                            layer_idx, self.layer_weights[layer_idx].moe_gate_ptr);
                    }
                    self.forward_moe(layer_idx, m)
                        .map_err(|e| format!("moe layer {}: {}", layer_idx, e))?;

                    // Gap 2: start preloading next MoE layer's experts into the other buffer
                    if self.kernels.fused_moe_fn.is_some() && self.d_fused_expert_w1_a.is_some() {
                        self.preload_next_moe_layer(layer_idx, num_hidden_layers)?;
                    }
                } else if self.layer_weights[layer_idx].shared_w1.is_some() {
                    if diag && layer_idx < diag_layer_limit {
                        eprintln!("[DIAG] layer{:02} -> forward_dense_mlp", layer_idx);
                    }
                    self.forward_dense_mlp(layer_idx, m)?;
                } else if diag && layer_idx < diag_layer_limit {
                    eprintln!("[DIAG] layer{:02} -> NO MLP (no gate, no shared_w1)", layer_idx);
                }

                if diag && chunk_idx == 0 && layer_idx < diag_layer_limit {
                    self.diag_print_norms(
                        &format!("layer{:02}_mlp", layer_idx),
                        *self.scratch.d_hidden.device_ptr(), &diag_positions, h);
                }

            }
        }

        // Use last chunk's m for final processing
        let m = total_m - (num_chunks - 1) * chunk_size.min(total_m);

        // 4. Final RMSNorm
        self.launch_fused_add_rmsnorm(
            *self.scratch.d_residual.device_ptr(),
            *self.scratch.d_hidden.device_ptr(),
            self.final_norm_ptr,
            m, h,
        )?;

        if diag {
            let final_norm = self.diag_l2_norm(*self.scratch.d_hidden.device_ptr(), m - 1, h, h);
            eprintln!("[DIAG] final_norm output: last_pos l2={:.6}", final_norm);
        }

        // 5. LM head + sampling
        let first_token = self.lm_head_and_sample(m, temperature, suppress_tokens)?;

        // 6. Sync
        self.stream_sync()?;

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::info!("Rust prefill: {} tok in {:.1}ms ({:.0} tok/s)", total_m, ms, total_m as f64 / (ms / 1000.0));

        Ok(PrefillResult { first_token, prompt_len: total_m, prefill_time_ms: ms })
    }

    /// Run prefill and extract top-k logprobs at sampled positions.
    /// Used by the /v1/internal/prefill_logits test endpoint.
    pub fn run_prefill_logits(
        &mut self,
        token_ids: &[u32],
        top_k: usize,
        sample_every: usize,
    ) -> Result<Vec<PrefillLogitPosition>, String> {
        self.bind_cuda_context()?;

        let t0 = Instant::now();
        let total_m = token_ids.len();
        let h = self.config.hidden_size;
        let v = self.config.vocab_size;
        let num_hidden_layers = self.config.num_hidden_layers;

        if total_m == 0 {
            return Err("Empty token sequence".to_string());
        }

        // No chunking for logits -- we need all positions in one pass to extract from hidden states
        let m = total_m;
        if m > self.scratch.max_tokens {
            return Err(format!("Prompt {} tokens > scratch {}", m, self.scratch.max_tokens));
        }

        // 1. Upload token IDs and positions
        self.upload_tokens_with_offset(token_ids, 0)?;

        // 2. Embedding lookup
        self.launch_embedding(m)?;

        // 3. Layer-by-layer forward pass
        let mut has_residual = false;
        for layer_idx in 0..num_hidden_layers {
            let layer_type = self.layer_weights[layer_idx].layer_type;

            if !has_residual {
                self.memcpy_d2d(
                    *self.scratch.d_residual.device_ptr(),
                    *self.scratch.d_hidden.device_ptr(),
                    (m * h * 2) as u64,
                )?;
                self.launch_rmsnorm(
                    *self.scratch.d_hidden.device_ptr(),
                    *self.scratch.d_residual.device_ptr(),
                    self.layer_weights[layer_idx].input_norm,
                    m, h,
                )?;
                has_residual = true;
            } else {
                self.launch_fused_add_rmsnorm(
                    *self.scratch.d_residual.device_ptr(),
                    *self.scratch.d_hidden.device_ptr(),
                    self.layer_weights[layer_idx].input_norm,
                    m, h,
                )?;
            }

            match layer_type {
                0 => self.forward_gqa_chunked(layer_idx, m, 0)?,
                1 => self.forward_mamba2(layer_idx, m)?,
                2 => {
                    self.memcpy_d2d(
                        *self.scratch.d_attn_out.device_ptr(),
                        *self.scratch.d_hidden.device_ptr(),
                        (m * h * 2) as u64,
                    )?;
                }
                3 => self.forward_linear_attention(layer_idx, m)?,
                _ => {
                    self.memcpy_d2d(
                        *self.scratch.d_attn_out.device_ptr(),
                        *self.scratch.d_hidden.device_ptr(),
                        (m * h * 2) as u64,
                    )?;
                }
            }

            let post_norm = self.layer_weights[layer_idx].post_attn_norm;
            if post_norm != 0 {
                self.launch_fused_add_rmsnorm(
                    *self.scratch.d_residual.device_ptr(),
                    *self.scratch.d_attn_out.device_ptr(),
                    post_norm,
                    m, h,
                )?;
                self.memcpy_d2d(
                    *self.scratch.d_hidden.device_ptr(),
                    *self.scratch.d_attn_out.device_ptr(),
                    (m * h * 2) as u64,
                )?;
            } else {
                self.memcpy_d2d(
                    *self.scratch.d_hidden.device_ptr(),
                    *self.scratch.d_attn_out.device_ptr(),
                    (m * h * 2) as u64,
                )?;
            }

            if self.layer_weights[layer_idx].moe_gate_ptr != 0 {
                self.forward_moe(layer_idx, m)?;
                if self.kernels.fused_moe_fn.is_some() && self.d_fused_expert_w1_a.is_some() {
                    self.preload_next_moe_layer(layer_idx, num_hidden_layers)?;
                }
            } else if self.layer_weights[layer_idx].shared_w1.is_some() {
                self.forward_dense_mlp(layer_idx, m)?;
            }
        }

        // 4. Final RMSNorm
        self.launch_fused_add_rmsnorm(
            *self.scratch.d_residual.device_ptr(),
            *self.scratch.d_hidden.device_ptr(),
            self.final_norm_ptr,
            m, h,
        )?;

        // 5. Extract logits at sampled positions via LM head
        let mut results = Vec::new();
        self.h_logits.resize(v, 0.0);

        // Determine which positions to sample
        let positions: Vec<usize> = (0..m).filter(|&p| p % sample_every == 0 || p == m - 1).collect();

        for &pos in &positions {
            // Run LM head on this position's hidden state
            let pos_hidden = *self.scratch.d_hidden.device_ptr() + (pos * h * 2) as u64;

            if let Some(ref lm) = self.lm_head {
                self.marlin_gemm(pos_hidden, lm, *self.scratch.d_logits.device_ptr(), 1)?;
            } else {
                return Err("No LM head".to_string());
            }

            // Download logits
            unsafe {
                let err = cuda_sys::lib().cuMemcpyDtoHAsync_v2(
                    self.h_logits.as_mut_ptr() as *mut _,
                    *self.scratch.d_logits.device_ptr(),
                    (v * 4) as usize,
                    self.stream,
                );
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("Download logits: {:?}", err));
                }
            }
            self.stream_sync()?;

            // Extract top-k logprobs
            let top = crate::decode::extract_top_logprobs(&self.h_logits, v, top_k);
            results.push(PrefillLogitPosition {
                position: pos,
                top_k: top.iter().map(|&(tid, lp)| (tid as usize, lp)).collect(),
            });
        }

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::info!("Prefill logits: {} positions from {} tokens in {:.1}ms", positions.len(), total_m, ms);

        Ok(results)
    }

    // ── Upload ──

    fn upload_tokens(&self, token_ids: &[u32]) -> Result<(), String> {
        self.upload_tokens_with_offset(token_ids, 0)
    }

    fn upload_tokens_with_offset(&self, token_ids: &[u32], offset: usize) -> Result<(), String> {
        let m = token_ids.len();
        let ids: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        let positions: Vec<i32> = (offset..offset + m).map(|p| p as i32).collect();

        // Use synchronous copies (cuMemcpyHtoD_v2) because the source Vec is unpinned
        // stack memory. Async copies require page-locked host memory, otherwise the
        // driver falls back to sync or behavior is undefined.
        unsafe {
            let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                *self.scratch.d_token_ids.device_ptr(),
                ids.as_ptr() as *const _,
                (m * 4) as usize,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("Upload token_ids: {:?}", err));
            }
            let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                *self.scratch.d_positions.device_ptr(),
                positions.as_ptr() as *const _,
                (m * 4) as usize,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("Upload positions: {:?}", err));
            }
        }
        Ok(())
    }

    // ── Kernel launchers ──

    fn launch_embedding(&self, m: usize) -> Result<(), String> {
        let d = self.config.hidden_size;
        let t = std::cmp::max(32, ((std::cmp::min(1024, d) + 31) / 32) * 32) as u32;
        let mut p0 = *self.scratch.d_hidden.device_ptr();
        let mut p1 = self.embedding_ptr;
        let mut p2 = *self.scratch.d_token_ids.device_ptr();
        let mut p3 = d as i32;
        unsafe {
            launch(self.kernels.embedding,
                (m as u32, 1, 1), (t, 1, 1), 0, self.stream,
                &mut [
                    &mut p0 as *mut _ as *mut std::ffi::c_void,
                    &mut p1 as *mut _ as *mut std::ffi::c_void,
                    &mut p2 as *mut _ as *mut std::ffi::c_void,
                    &mut p3 as *mut _ as *mut std::ffi::c_void,
                ],
            )?;
        }
        Ok(())
    }

    fn launch_rmsnorm(&self, out: u64, x: u64, w: u64, m: usize, d: usize) -> Result<(), String> {
        let t = std::cmp::max(32, ((std::cmp::min(1024, d) + 31) / 32) * 32) as u32;
        let mut p0 = out; let mut p1 = x; let mut p2 = w;
        let mut p3 = d as i32; let mut p4 = self.config.rms_norm_eps;
        unsafe {
            launch(self.kernels.rmsnorm,
                (m as u32, 1, 1), (t, 1, 1), t * 4, self.stream,
                &mut [
                    &mut p0 as *mut _ as *mut std::ffi::c_void,
                    &mut p1 as *mut _ as *mut std::ffi::c_void,
                    &mut p2 as *mut _ as *mut std::ffi::c_void,
                    &mut p3 as *mut _ as *mut std::ffi::c_void,
                    &mut p4 as *mut _ as *mut std::ffi::c_void,
                ],
            )?;
        }
        Ok(())
    }

    fn launch_fused_add_rmsnorm(
        &self, residual: u64, x: u64, w: u64, m: usize, d: usize,
    ) -> Result<(), String> {
        let t = std::cmp::max(32, ((std::cmp::min(1024, d) + 31) / 32) * 32) as u32;
        let mut p0 = residual; let mut p1 = x; let mut p2 = x;
        let mut p3 = w; let mut p4 = d as i32; let mut p5 = self.config.rms_norm_eps;
        unsafe {
            launch(self.kernels.fused_add_rmsnorm,
                (m as u32, 1, 1), (t, 1, 1), t * 4, self.stream,
                &mut [
                    &mut p0 as *mut _ as *mut std::ffi::c_void,
                    &mut p1 as *mut _ as *mut std::ffi::c_void,
                    &mut p2 as *mut _ as *mut std::ffi::c_void,
                    &mut p3 as *mut _ as *mut std::ffi::c_void,
                    &mut p4 as *mut _ as *mut std::ffi::c_void,
                    &mut p5 as *mut _ as *mut std::ffi::c_void,
                ],
            )?;
        }
        Ok(())
    }

    fn memcpy_d2d(&self, dst: u64, src: u64, bytes: u64) -> Result<(), String> {
        unsafe {
            let err = cuda_sys::lib().cuMemcpyDtoDAsync_v2(dst, src, bytes as usize, self.stream);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("memcpy_d2d: {:?}", err));
            }
        }
        Ok(())
    }

    /// Transpose 3D FP32 array: [A, B, C] -> [B, A, C] in a single kernel launch.
    /// Replaces per-element memcpy storms.
    fn launch_transpose_3d_f32(&self, out: u64, inp: u64, a: usize, b: usize, c: usize) -> Result<(), String> {
        if a == 0 || b == 0 || c == 0 { return Ok(()); }
        let threads = std::cmp::max(32, ((std::cmp::min(256, c) + 31) / 32) * 32) as u32;
        let mut p0 = out;
        let mut p1 = inp;
        let mut p2 = a as i32;
        let mut p3 = b as i32;
        let mut p4 = c as i32;
        unsafe {
            launch(self.kernels.transpose_3d_021,
                (b as u32, a as u32, 1), (threads, 1, 1), 0, self.stream,
                &mut [
                    &mut p0 as *mut _ as *mut std::ffi::c_void,
                    &mut p1 as *mut _ as *mut std::ffi::c_void,
                    &mut p2 as *mut _ as *mut std::ffi::c_void,
                    &mut p3 as *mut _ as *mut std::ffi::c_void,
                    &mut p4 as *mut _ as *mut std::ffi::c_void,
                ],
            )?;
        }
        Ok(())
    }

    fn launch_transpose_f32(&self, out: u64, inp: u64, rows: usize, cols: usize) -> Result<(), String> {
        let t = std::cmp::max(32, ((std::cmp::min(1024, rows) + 31) / 32) * 32) as u32;
        let mut p0 = out;
        let mut p1 = inp;
        let mut p2 = rows as i32;
        let mut p3 = cols as i32;
        unsafe {
            launch(self.kernels.la_transpose_f32,
                (cols as u32, 1, 1), (t, 1, 1), 0, self.stream,
                &mut [
                    &mut p0 as *mut _ as *mut std::ffi::c_void,
                    &mut p1 as *mut _ as *mut std::ffi::c_void,
                    &mut p2 as *mut _ as *mut std::ffi::c_void,
                    &mut p3 as *mut _ as *mut std::ffi::c_void,
                ],
            )?;
        }
        Ok(())
    }

    fn stream_sync(&self) -> Result<(), String> {
        unsafe {
            let err = cuda_sys::lib().cuStreamSynchronize(self.stream);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("stream sync: {:?}", err));
            }
        }
        Ok(())
    }

    // ── Marlin GEMM ──

    fn marlin_gemm(&self, a: u64, w: &MarlinWeight, c: u64, m: usize) -> Result<(), String> {
        let f = self.kernels.marlin_mm.ok_or("Marlin GEMM not loaded")?;
        let st = if w.num_bits == 4 { &ScalarType::U4B8 } else { &ScalarType::U8B128 };

        // Zero the workspace lock array before each call. Marlin uses it for
        // inter-block barrier synchronization (barrier_acquire spins until
        // locks[off] == expected). Stale non-zero values from prior calls cause
        // deadlocks or incorrect sync, producing zeros or garbage.
        let ws_ptr = *self.scratch.d_workspace.device_ptr();
        let ws_len = self.config.sms * 4;
        unsafe {
            cuda_sys::lib().cuMemsetD32Async(ws_ptr, 0, ws_len, self.stream);
        }

        // Zero the C_tmp (FP32 scratch) region used by this GEMM call.
        // When use_fp32_reduce=true and use_atomic_add=true, the Marlin kernel
        // atomicAdds partial results to C_tmp. Stale values from prior GEMM calls
        // accumulate, inflating output norms. This is critical for sequential
        // expert GEMM dispatch where hundreds of calls share the same C_tmp buffer.
        let ctmp_ptr = *self.scratch.d_fp32_scratch.device_ptr();
        let ctmp_len = m * w.n;  // FP32 elements needed for this call
        unsafe {
            cuda_sys::lib().cuMemsetD32Async(ctmp_ptr, 0, ctmp_len, self.stream);
        }

        unsafe {
            f(
                a as *const _, w.packed as *const _,
                c as *mut _, *self.scratch.d_fp32_scratch.device_ptr() as *mut _,
                w.scales as *const _, std::ptr::null(), std::ptr::null(),
                std::ptr::null(), std::ptr::null(), std::ptr::null_mut(),
                m as i32, w.n as i32, w.k as i32, w.k as i32,
                ws_ptr as *mut _,
                st, false, true, false,
                w.num_groups as i32, w.group_size as i32,
                self.config.device_ordinal as i32,
                self.stream as u64,
                -1, -1, self.config.sms as i32,
                false, true, false,
            );
        }
        Ok(())
    }

    /// cuBLAS BF16 GEMM: C = A @ W^T, all BF16, computed in FP32.
    /// A: [m, k] BF16 row-major, W: [n, k] BF16 row-major, C: [m, n] BF16 row-major
    fn cublas_bf16_gemm(&self, a: u64, w: &Bf16Weight, c: u64, m: usize) -> Result<(), String> {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            use cudarc::cublas::sys as cublas_sys;
            use cudarc::cublas::result as cublas_result;

            cublas_result::set_stream(
                self.cublas_handle,
                self.stream as cublas_sys::cudaStream_t,
            ).map_err(|e| format!("cublas set_stream: {:?}", e))?;

            cublas_result::gemm_ex(
                self.cublas_handle,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                w.n as i32, m as i32, w.k as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                w.ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType::CUDA_R_16BF, w.k as i32,
                a as *const std::ffi::c_void,
                cublas_sys::cudaDataType::CUDA_R_16BF, w.k as i32,
                &beta as *const f32 as *const std::ffi::c_void,
                c as *mut std::ffi::c_void,
                cublas_sys::cudaDataType::CUDA_R_16BF, w.n as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            ).map_err(|e| format!("cublas bf16 GEMM: {:?}", e))?;
        }
        Ok(())
    }

    /// Dispatch GEMM: Marlin (INT4/INT8) or cuBLAS BF16 fallback.
    fn la_gemm(&self, a: u64, marlin: &Option<MarlinWeight>, bf16: &Option<Bf16Weight>, c: u64, m: usize) -> Result<(), String> {
        if let Some(w) = marlin {
            self.marlin_gemm(a, w, c, m)
        } else if let Some(w) = bf16 {
            self.cublas_bf16_gemm(a, w, c, m)
        } else {
            Err("LA GEMM: no Marlin or BF16 weight available".to_string())
        }
    }

    // ── GQA Attention ──

    fn forward_gqa(
        &self, layer_idx: usize, m: usize,
    ) -> Result<(), String> {
        self.forward_gqa_chunked(layer_idx, m, 0)
    }

    fn forward_gqa_chunked(
        &self, layer_idx: usize, m: usize, start_pos: usize,
    ) -> Result<(), String> {
        let cfg = &self.config;
        let lw = &self.layer_weights[layer_idx];

        let hidden = *self.scratch.d_hidden.device_ptr();
        let q = *self.scratch.d_q.device_ptr();
        let k = *self.scratch.d_k.device_ptr();
        let v = *self.scratch.d_v.device_ptr();
        let attn_out = *self.scratch.d_attn_out.device_ptr();

        // Q/K/V projections (Marlin or cuBLAS BF16)
        let gate_ptr: u64;
        if lw.gqa_gated {
            // Gated attention: q_proj outputs [M, num_q_heads, head_dim * 2]
            // GEMM into scratch1, then split into Q + gate via single kernel
            let q_raw = *self.scratch.d_scratch1.device_ptr();
            self.la_gemm(hidden, &lw.q_proj, &lw.q_proj_bf16, q_raw, m)?;

            let gate_buf = *self.scratch.d_scratch2.device_ptr();
            let threads = std::cmp::max(32, ((std::cmp::min(256, cfg.head_dim) + 31) / 32) * 32) as u32;
            let mut gs0 = q; let mut gs1 = gate_buf; let mut gs2 = q_raw;
            let mut gs3 = cfg.num_q_heads as i32; let mut gs4 = cfg.head_dim as i32;
            unsafe {
                launch(self.kernels.gated_q_split,
                    (m as u32, cfg.num_q_heads as u32, 1), (threads, 1, 1), 0, self.stream,
                    &mut [
                        &mut gs0 as *mut _ as *mut std::ffi::c_void,
                        &mut gs1 as *mut _ as *mut std::ffi::c_void,
                        &mut gs2 as *mut _ as *mut std::ffi::c_void,
                        &mut gs3 as *mut _ as *mut std::ffi::c_void,
                        &mut gs4 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
            gate_ptr = gate_buf;
        } else {
            self.la_gemm(hidden, &lw.q_proj, &lw.q_proj_bf16, q, m)?;
            gate_ptr = 0;
        }
        self.la_gemm(hidden, &lw.k_proj, &lw.k_proj_bf16, k, m)?;
        self.la_gemm(hidden, &lw.v_proj, &lw.v_proj_bf16, v, m)?;

        // Look up per-layer KV cache pointers (needed for both attention and cache append)
        let layer_k_ptr = if layer_idx < self.kv_k_ptrs.len() { self.kv_k_ptrs[layer_idx] } else { 0 };
        let layer_v_ptr = if layer_idx < self.kv_v_ptrs.len() { self.kv_v_ptrs[layer_idx] } else { 0 };

        // RoPE (uses rope_half_dim for partial rotary support)
        if self.rope_cos_ptr != 0 {
            let half = if cfg.rope_half_dim > 0 { cfg.rope_half_dim } else { cfg.head_dim / 2 };
            let rt = std::cmp::max(32, ((std::cmp::min(512, half) + 31) / 32) * 32) as u32;
            let mut r0 = q; let mut r1 = k;
            let mut r2 = *self.scratch.d_positions.device_ptr();
            let mut r3 = self.rope_cos_ptr; let mut r4 = self.rope_sin_ptr;
            let mut r5 = cfg.num_q_heads as i32; let mut r6 = cfg.num_kv_heads as i32;
            let mut r7 = cfg.head_dim as i32; let mut r8 = half as i32;
            unsafe {
                launch(self.kernels.rope,
                    (m as u32, 1, 1), (rt, 1, 1), 0, self.stream,
                    &mut [
                        &mut r0 as *mut _ as *mut std::ffi::c_void,
                        &mut r1 as *mut _ as *mut std::ffi::c_void,
                        &mut r2 as *mut _ as *mut std::ffi::c_void,
                        &mut r3 as *mut _ as *mut std::ffi::c_void,
                        &mut r4 as *mut _ as *mut std::ffi::c_void,
                        &mut r5 as *mut _ as *mut std::ffi::c_void,
                        &mut r6 as *mut _ as *mut std::ffi::c_void,
                        &mut r7 as *mut _ as *mut std::ffi::c_void,
                        &mut r8 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // WMMA Flash Attention with cross-chunk support:
        // - Positions [0, start_pos): read from FP8 KV cache
        // - Positions [start_pos, start_pos+m): read from current BF16 K/V
        // Kernel uses 1 warp (32 threads), BR=16 queries per block, BC=64 KV tile
        let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
        let fa_br = 16u32;  // queries per block (1 wmma M=16 tile)
        let fa_bc = 64usize;  // KV tile size
        let grid_x = ((m as u32) + fa_br - 1) / fa_br;
        // smem: s_q[BR*hd*2] + s_k[BC*hd*2] + s_v[BC*hd*2] + s_scores[BR*BC*4] + s_p[BR*BC*2] + s_o_tmp[16*16*4]
        let hd = cfg.head_dim;
        let smem = (fa_br as usize * hd * 2  // s_q
            + fa_bc * hd * 2                  // s_k
            + fa_bc * hd * 2                  // s_v
            + fa_br as usize * fa_bc * 4      // s_scores
            + fa_br as usize * fa_bc * 2      // s_p
            + 16 * 16 * 4                     // s_o_tmp
        ) as u32;
        let kv_stride = (cfg.num_kv_heads * cfg.head_dim) as i32;
        // For first chunk (start_pos==0), pass null cache pointers
        let k_cache_ptr: u64 = if start_pos > 0 { layer_k_ptr } else { 0 };
        let v_cache_ptr: u64 = if start_pos > 0 { layer_v_ptr } else { 0 };
        let mut a0 = attn_out; let mut a1 = q;
        let mut a2 = k_cache_ptr; let mut a3 = v_cache_ptr;
        let mut a4 = k; let mut a5 = v;
        let mut a6 = m as i32; let mut a7 = cfg.num_q_heads as i32;
        let mut a8 = cfg.num_kv_heads as i32; let mut a9 = cfg.head_dim as i32;
        let mut a10 = scale; let mut a11 = start_pos as i32; let mut a12 = kv_stride;
        unsafe {
            launch(self.kernels.flash_attn_tiled,
                (grid_x, cfg.num_q_heads as u32, 1), (32, 1, 1), smem, self.stream,
                &mut [
                    &mut a0 as *mut _ as *mut std::ffi::c_void,
                    &mut a1 as *mut _ as *mut std::ffi::c_void,
                    &mut a2 as *mut _ as *mut std::ffi::c_void,
                    &mut a3 as *mut _ as *mut std::ffi::c_void,
                    &mut a4 as *mut _ as *mut std::ffi::c_void,
                    &mut a5 as *mut _ as *mut std::ffi::c_void,
                    &mut a6 as *mut _ as *mut std::ffi::c_void,
                    &mut a7 as *mut _ as *mut std::ffi::c_void,
                    &mut a8 as *mut _ as *mut std::ffi::c_void,
                    &mut a9 as *mut _ as *mut std::ffi::c_void,
                    &mut a10 as *mut _ as *mut std::ffi::c_void,
                    &mut a11 as *mut _ as *mut std::ffi::c_void,
                    &mut a12 as *mut _ as *mut std::ffi::c_void,
                ],
            )?;
        }

        // KV cache append: BF16 K,V -> FP8 E4M3 into separate per-layer caches
        if layer_k_ptr != 0 && layer_v_ptr != 0 {
            let kv_stride = (cfg.num_kv_heads * cfg.head_dim) as i32;
            let kt = std::cmp::max(32, ((std::cmp::min(256, kv_stride as usize) + 31) / 32) * 32) as u32;
            let mut k0 = layer_k_ptr; let mut k1 = layer_v_ptr;
            let mut k2 = k; let mut k3 = v;
            let mut k4 = m as i32; let mut k5 = kv_stride;
            let mut k6 = self.kv_max_seq as i32; let mut k7 = start_pos as i32;
            unsafe {
                launch(self.kernels.kv_cache_append,
                    (m as u32, 1, 1), (kt, 1, 1), 0, self.stream,
                    &mut [
                        &mut k0 as *mut _ as *mut std::ffi::c_void,
                        &mut k1 as *mut _ as *mut std::ffi::c_void,
                        &mut k2 as *mut _ as *mut std::ffi::c_void,
                        &mut k3 as *mut _ as *mut std::ffi::c_void,
                        &mut k4 as *mut _ as *mut std::ffi::c_void,
                        &mut k5 as *mut _ as *mut std::ffi::c_void,
                        &mut k6 as *mut _ as *mut std::ffi::c_void,
                        &mut k7 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Gated attention: apply sigmoid(gate) to attention output
        if lw.gqa_gated && gate_ptr != 0 {
            let q_dim = (cfg.num_q_heads * cfg.head_dim) as i32;
            let gt = std::cmp::max(32, ((std::cmp::min(1024, q_dim as usize) + 31) / 32) * 32) as u32;
            let mut g0 = attn_out; let mut g1 = attn_out; let mut g2 = gate_ptr;
            let mut g3 = q_dim;
            unsafe {
                launch(self.kernels.sigmoid_mul,
                    (m as u32, 1, 1), (gt, 1, 1), 0, self.stream,
                    &mut [
                        &mut g0 as *mut _ as *mut std::ffi::c_void,
                        &mut g1 as *mut _ as *mut std::ffi::c_void,
                        &mut g2 as *mut _ as *mut std::ffi::c_void,
                        &mut g3 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // O projection (use scratch1 as temp to avoid input/output aliasing)
        let o_temp = *self.scratch.d_scratch1.device_ptr();
        self.la_gemm(attn_out, &lw.o_proj, &lw.o_proj_bf16, o_temp, m)?;
        self.memcpy_d2d(attn_out, o_temp, (m * cfg.hidden_size * 2) as u64)?;

        Ok(())
    }

    // ── Mamba2 ──

    fn forward_mamba2(&self, layer_idx: usize, m: usize) -> Result<(), String> {
        let cfg = &self.config;
        let lw = &self.layer_weights[layer_idx];

        let in_proj = lw.mamba2_in_proj.as_ref().ok_or("missing in_proj")?;
        let out_proj = lw.mamba2_out_proj.as_ref().ok_or("missing out_proj")?;

        let hidden = *self.scratch.d_hidden.device_ptr();
        let attn_out = *self.scratch.d_attn_out.device_ptr();
        let in_buf = *self.scratch.d_mamba2_in_proj.as_ref().ok_or("no mamba2 buf")?.device_ptr();
        let ssd_out = *self.scratch.d_mamba2_ssd_out.as_ref().ok_or("no ssd buf")?.device_ptr();
        let ssm_state = *self.scratch.d_mamba2_ssm_state.as_ref().ok_or("no ssm state")?.device_ptr();

        let d_inner = cfg.mamba_d_inner;
        let n_heads = cfg.mamba_num_heads;
        let head_dim = cfg.mamba_head_dim;
        let d_state = cfg.mamba_d_state;
        let n_groups = cfg.mamba_n_groups;
        let bc_size = n_groups * d_state;
        let in_proj_dim = 2 * d_inner + 2 * bc_size + n_heads;

        // 1. in_proj GEMM
        self.marlin_gemm(hidden, in_proj, in_buf, m)?;

        // 2. Zero SSM state
        unsafe {
            let err = cuda_sys::lib().cuMemsetD8Async(
                ssm_state, 0, (n_heads * head_dim * d_state * 4) as usize, self.stream,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("zero ssm state: {:?}", err));
            }
        }

        // 3. Extract x, B, C, dt from in_proj output via single kernel launch
        //    in_proj layout: [z(d_inner) | x(d_inner) | B(bc) | C(bc) | dt(n_heads)]
        let x_out = *self.scratch.d_scratch1.device_ptr();
        let b_out = *self.scratch.d_scratch2.device_ptr();
        let c_out = b_out + (m * bc_size * 2) as u64;
        let dt_out = c_out + (m * bc_size * 2) as u64;

        {
            let ext_t = std::cmp::max(32, ((std::cmp::min(1024, d_inner) + 31) / 32) * 32) as u32;
            let mut e0 = x_out; let mut e1 = b_out; let mut e2 = c_out; let mut e3 = dt_out;
            let mut e4 = in_buf;
            let mut e5 = d_inner as i32; let mut e6 = bc_size as i32;
            let mut e7 = n_heads as i32; let mut e8 = in_proj_dim as i32;
            unsafe {
                launch(self.kernels.mamba2_extract,
                    (m as u32, 1, 1), (ext_t, 1, 1), 0, self.stream,
                    &mut [
                        &mut e0 as *mut _ as *mut std::ffi::c_void,
                        &mut e1 as *mut _ as *mut std::ffi::c_void,
                        &mut e2 as *mut _ as *mut std::ffi::c_void,
                        &mut e3 as *mut _ as *mut std::ffi::c_void,
                        &mut e4 as *mut _ as *mut std::ffi::c_void,
                        &mut e5 as *mut _ as *mut std::ffi::c_void,
                        &mut e6 as *mut _ as *mut std::ffi::c_void,
                        &mut e7 as *mut _ as *mut std::ffi::c_void,
                        &mut e8 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // 4. SSD kernel
        let threads = std::cmp::max(32, ((std::cmp::min(256, head_dim) + 31) / 32) * 32);
        let blocks_d = (head_dim + threads - 1) / threads;

        let a_ptr = lw.mamba2_A;
        let d_ptr = lw.mamba2_D;
        let dt_bias = lw.mamba2_dt_bias;

        let mut s0 = ssd_out; let mut s1_v = x_out; let mut s2_v = dt_out;
        let mut s3 = a_ptr; let mut s4 = b_out; let mut s5 = c_out; let mut s6 = d_ptr;
        let mut s7 = ssm_state; let mut s8 = dt_bias;
        let mut s9 = m as i32; let mut s10 = n_heads as i32; let mut s11 = head_dim as i32;
        let mut s12 = d_state as i32; let mut s13 = n_groups as i32; let mut s14 = 1i32;
        unsafe {
            launch(self.kernels.mamba2_ssd,
                (n_heads as u32, blocks_d as u32, 1), (threads as u32, 1, 1), 0, self.stream,
                &mut [
                    &mut s0 as *mut _ as *mut std::ffi::c_void,
                    &mut s1_v as *mut _ as *mut std::ffi::c_void,
                    &mut s2_v as *mut _ as *mut std::ffi::c_void,
                    &mut s3 as *mut _ as *mut std::ffi::c_void,
                    &mut s4 as *mut _ as *mut std::ffi::c_void,
                    &mut s5 as *mut _ as *mut std::ffi::c_void,
                    &mut s6 as *mut _ as *mut std::ffi::c_void,
                    &mut s7 as *mut _ as *mut std::ffi::c_void,
                    &mut s8 as *mut _ as *mut std::ffi::c_void,
                    &mut s9 as *mut _ as *mut std::ffi::c_void,
                    &mut s10 as *mut _ as *mut std::ffi::c_void,
                    &mut s11 as *mut _ as *mut std::ffi::c_void,
                    &mut s12 as *mut _ as *mut std::ffi::c_void,
                    &mut s13 as *mut _ as *mut std::ffi::c_void,
                    &mut s14 as *mut _ as *mut std::ffi::c_void,
                ],
            )?;
        }

        // 5. out_proj GEMM
        self.marlin_gemm(ssd_out, out_proj, attn_out, m)?;

        Ok(())
    }

    // ── Linear Attention (Gated DeltaNet) ──

    fn forward_linear_attention(&self, layer_idx: usize, m: usize) -> Result<(), String> {
        let cfg = &self.config;
        let lw = &self.layer_weights[layer_idx];
        let nk = cfg.la_num_k_heads;
        let nv = cfg.la_num_v_heads;
        let dk = cfg.la_k_head_dim;
        let dv = cfg.la_v_head_dim;
        let hr = cfg.la_head_ratio;
        let kernel_dim = cfg.la_conv_kernel_dim;
        let conv_dim = cfg.la_conv_dim;
        let chunk_size = cfg.la_chunk_size;
        let scale = cfg.la_scale;

        // LA projections: prefer Marlin INT4/INT8, fall back to cuBLAS BF16
        if lw.la_in_proj_qkvz.is_none() && lw.la_in_proj_qkvz_bf16.is_none() {
            return Err("missing la_in_proj_qkvz (no Marlin or BF16)".to_string());
        }
        if lw.la_in_proj_ba.is_none() && lw.la_in_proj_ba_bf16.is_none() {
            return Err("missing la_in_proj_ba (no Marlin or BF16)".to_string());
        }
        if lw.la_out_proj.is_none() && lw.la_out_proj_bf16.is_none() {
            return Err("missing la_out_proj (no Marlin or BF16)".to_string());
        }
        if layer_idx == 0 && std::env::var("KRASIS_PREFILL_DIAG").is_ok() {
            eprintln!("[DIAG L00] LA config: nk={} nv={} dk={} dv={} hr={} chunk_size={} scale={:.4} kernel_dim={} conv_dim={}",
                nk, nv, dk, dv, hr, chunk_size, scale, kernel_dim, conv_dim);
        }

        let hidden = *self.scratch.d_hidden.device_ptr();
        let attn_out = *self.scratch.d_attn_out.device_ptr();
        let proj_buf = *self.scratch.d_la_proj_buf.as_ref().ok_or("no la_proj_buf")?.device_ptr();

        // q/k in [M, nv, dk] FP32, v in [M, nv, dv] FP32 after repeat-interleave
        let la_q = *self.scratch.d_la_q.as_ref().ok_or("no la_q")?.device_ptr();
        let la_k = *self.scratch.d_la_k.as_ref().ok_or("no la_k")?.device_ptr();
        let la_v = *self.scratch.d_la_v.as_ref().ok_or("no la_v")?.device_ptr();
        let la_z = *self.scratch.d_la_z.as_ref().ok_or("no la_z")?.device_ptr();
        let la_b = *self.scratch.d_la_b.as_ref().ok_or("no la_b")?.device_ptr();
        let la_a = *self.scratch.d_la_a.as_ref().ok_or("no la_a")?.device_ptr();
        let la_beta = *self.scratch.d_la_beta.as_ref().ok_or("no la_beta")?.device_ptr();
        let la_gate = *self.scratch.d_la_gate.as_ref().ok_or("no la_gate")?.device_ptr();
        let la_conv_out = *self.scratch.d_la_conv_out.as_ref().ok_or("no la_conv_out")?.device_ptr();
        let la_v_beta = *self.scratch.d_la_v_beta.as_ref().ok_or("no la_v_beta")?.device_ptr();
        let la_k_beta = *self.scratch.d_la_k_beta.as_ref().ok_or("no la_k_beta")?.device_ptr();
        let la_v_new = *self.scratch.d_la_v_new.as_ref().ok_or("no la_v_new")?.device_ptr();
        let la_g_cum = *self.scratch.d_la_g_cum.as_ref().ok_or("no la_g_cum")?.device_ptr();
        let la_attn = *self.scratch.d_la_attn.as_ref().ok_or("no la_attn")?.device_ptr();
        let la_state = *self.scratch.d_la_state.as_ref().ok_or("no la_state")?.device_ptr();
        let la_chunk_out = *self.scratch.d_la_chunk_out.as_ref().ok_or("no la_chunk_out")?.device_ptr();

        let key_dim = nk * dk;
        let value_dim = nv * dv;
        let group_dim = 2 * dk + 2 * hr * dv;
        let qkvz_dim = nk * group_dim;
        let ba_dim = nk * 2 * hr;

        // 1. in_proj_qkvz GEMM: [M, hidden] -> [M, qkvz_dim] BF16
        self.la_gemm(hidden, &lw.la_in_proj_qkvz, &lw.la_in_proj_qkvz_bf16, proj_buf, m)?;
        let la_diag = std::env::var("KRASIS_PREFILL_DIAG").is_ok()
            && layer_idx < Self::diag_layer_limit();
        if la_diag {
            let gemm_norm = self.diag_l2_norm(proj_buf, 0, qkvz_dim, qkvz_dim);
            eprintln!("[DIAG L{:02}] qkvz_gemm pos0 norm={:.6} (dim={})", layer_idx, gemm_norm, qkvz_dim);
        }

        // 2. Uninterleave QKVZ: proj_buf -> q_bf16, k_bf16, v_bf16(as bf16), z
        //    We'll use scratch1 for q_bf16 [M, nk*dk], scratch2 for k_bf16 [M, nk*dk]
        //    v_bf16 goes to la_z buffer (reuse), z stays in la_z
        let q_bf16 = *self.scratch.d_scratch1.device_ptr(); // reuse
        let k_bf16 = *self.scratch.d_scratch2.device_ptr(); // reuse
        // v_bf16 stored temporarily overlapping la_v (then converted to FP32)
        // z stored in la_z (BF16)
        {
            let total = nk * group_dim;
            let t = std::cmp::max(32, ((std::cmp::min(1024, total) + 31) / 32) * 32) as u32;
            let mut p0 = q_bf16;       // q out [M, key_dim] bf16
            let mut p1 = k_bf16;       // k out [M, key_dim] bf16
            // v out goes to attn_out buffer (large enough: M * value_dim), then converted
            let v_bf16 = attn_out;
            let mut p2 = v_bf16;       // v out [M, value_dim] bf16
            let mut p3 = la_z;         // z out [M, value_dim] bf16
            let mut p4 = proj_buf;
            let mut p5 = nk as i32;
            let mut p6 = dk as i32;
            let mut p7 = hr as i32;
            let mut p8 = dv as i32;
            unsafe {
                launch(self.kernels.la_uninterleave_qkvz,
                    (m as u32, 1, 1), (t, 1, 1), 0, self.stream,
                    &mut [
                        &mut p0 as *mut _ as *mut std::ffi::c_void,
                        &mut p1 as *mut _ as *mut std::ffi::c_void,
                        &mut p2 as *mut _ as *mut std::ffi::c_void,
                        &mut p3 as *mut _ as *mut std::ffi::c_void,
                        &mut p4 as *mut _ as *mut std::ffi::c_void,
                        &mut p5 as *mut _ as *mut std::ffi::c_void,
                        &mut p6 as *mut _ as *mut std::ffi::c_void,
                        &mut p7 as *mut _ as *mut std::ffi::c_void,
                        &mut p8 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }

            // DIAG: norms after uninterleave
            if la_diag {
                self.stream_sync()?;
                let q_norm = self.diag_l2_norm(q_bf16, 0, key_dim, key_dim);
                let k_norm = self.diag_l2_norm(k_bf16, 0, key_dim, key_dim);
                let v_norm = self.diag_l2_norm(v_bf16, 0, value_dim, value_dim);
                let z_norm = self.diag_l2_norm(la_z, 0, value_dim, value_dim);
                eprintln!("[DIAG L{:02}] after_uninterleave pos0: q={:.6} k={:.6} v={:.6} z={:.6}",
                    layer_idx, q_norm, k_norm, v_norm, z_norm);
            }

            // 3. in_proj_ba GEMM: [M, hidden] -> [M, ba_dim] BF16
            self.la_gemm(hidden, &lw.la_in_proj_ba, &lw.la_in_proj_ba_bf16, proj_buf, m)?;

            // 4. Uninterleave BA -> b [M, nv] BF16, a [M, nv] BF16
            {
                let ba_total = nk * 2 * hr;
                let bt = std::cmp::max(32, ((std::cmp::min(1024, ba_total) + 31) / 32) * 32) as u32;
                let mut b0 = la_b;
                let mut b1 = la_a;
                let mut b2 = proj_buf;
                let mut b3 = nk as i32;
                let mut b4 = hr as i32;
                unsafe {
                    launch(self.kernels.la_uninterleave_ba,
                        (m as u32, 1, 1), (bt, 1, 1), 0, self.stream,
                        &mut [
                            &mut b0 as *mut _ as *mut std::ffi::c_void,
                            &mut b1 as *mut _ as *mut std::ffi::c_void,
                            &mut b2 as *mut _ as *mut std::ffi::c_void,
                            &mut b3 as *mut _ as *mut std::ffi::c_void,
                            &mut b4 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            // 5. Depthwise Conv1d + SiLU
            // Input for conv: concatenate q_flat, k_flat, v_flat -> [M, conv_dim] BF16
            // The conv kernel expects [conv_dim, M] layout and BF16 input row-major [M, conv_dim]
            // We already have q_bf16 [M, key_dim], k_bf16 [M, key_dim], v_bf16 [M, value_dim]
            // We need to concatenate them into a [M, conv_dim] BF16 buffer.
            // Use proj_buf as the concatenated buffer (it's large enough: qkvz_dim > conv_dim)
            //
            // Actually conv_dim = 2*key_dim + value_dim, and proj_buf was [M, qkvz_dim].
            // qkvz_dim = nk * (2*dk + 2*hr*dv) vs conv_dim = nk*dk*2 + nv*dv
            // For QCN: qkvz = 16 * (256 + 256) = 8192, conv_dim = 2*2048 + 4096 = 8192. Equal.
            // So proj_buf is exactly right size.

            // Concat q, k, v into proj_buf [M, conv_dim] via single kernel launch
            {
                let threads = std::cmp::max(32, ((std::cmp::min(256, conv_dim) + 31) / 32) * 32) as u32;
                let mut cc0 = proj_buf;
                let mut cc1 = q_bf16;
                let mut cc2 = k_bf16;
                let mut cc3 = v_bf16;
                let mut cc4 = key_dim as i32;
                let mut cc5 = key_dim as i32;
                let mut cc6 = value_dim as i32;
                unsafe {
                    launch(self.kernels.concat_3_bf16,
                        (m as u32, 1, 1), (threads, 1, 1), 0, self.stream,
                        &mut [
                            &mut cc0 as *mut _ as *mut std::ffi::c_void,
                            &mut cc1 as *mut _ as *mut std::ffi::c_void,
                            &mut cc2 as *mut _ as *mut std::ffi::c_void,
                            &mut cc3 as *mut _ as *mut std::ffi::c_void,
                            &mut cc4 as *mut _ as *mut std::ffi::c_void,
                            &mut cc5 as *mut _ as *mut std::ffi::c_void,
                            &mut cc6 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            // Launch conv kernel: conv_state + [M, conv_dim] BF16 -> [conv_dim, M] FP32
            let conv_state = lw.la_conv_state_ptr;
            let conv_weight = lw.la_conv_weight_ptr;
            {
                let ct = std::cmp::max(32, ((std::cmp::min(1024, m) + 31) / 32) * 32) as u32;
                let mut c0 = la_conv_out;    // output [conv_dim, M] FP32
                let mut c1 = conv_state;     // conv_state [conv_dim, kernel_dim] FP32
                let mut c2 = proj_buf;       // input [M, conv_dim] BF16
                let mut c3 = conv_weight;    // weight [conv_dim, kernel_dim] FP32
                let mut c4 = m as i32;
                let mut c5 = conv_dim as i32;
                let mut c6 = kernel_dim as i32;
                unsafe {
                    launch(self.kernels.la_depthwise_conv1d_silu,
                        (conv_dim as u32, 1, 1), (ct, 1, 1), 0, self.stream,
                        &mut [
                            &mut c0 as *mut _ as *mut std::ffi::c_void,
                            &mut c1 as *mut _ as *mut std::ffi::c_void,
                            &mut c2 as *mut _ as *mut std::ffi::c_void,
                            &mut c3 as *mut _ as *mut std::ffi::c_void,
                            &mut c4 as *mut _ as *mut std::ffi::c_void,
                            &mut c5 as *mut _ as *mut std::ffi::c_void,
                            &mut c6 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            // 6. Split conv_out [conv_dim, M] FP32 to q, k, v heads
            // Conv output: [conv_dim, M] FP32. Need [M, conv_dim] then split to q/k/v.
            // Step 1: Transpose [conv_dim, M] -> [M, conv_dim] via 2D transpose kernel
            let fp32_scratch = *self.scratch.d_fp32_scratch.device_ptr();
            self.launch_transpose_f32(fp32_scratch, la_conv_out, conv_dim, m)?;

            // Step 2: Split [M, conv_dim] -> q[M,key_dim] + k[M,key_dim] + v[M,value_dim]
            // via single kernel launch (replaces M per-token memcpy loops)
            {
                let threads = std::cmp::max(32, ((std::cmp::min(256, conv_dim) + 31) / 32) * 32) as u32;
                let mut sp0 = la_q; let mut sp1 = la_k; let mut sp2 = la_v;
                let mut sp3 = fp32_scratch;
                let mut sp4 = key_dim as i32; let mut sp5 = value_dim as i32;
                unsafe {
                    launch(self.kernels.la_split_conv_output,
                        (m as u32, 1, 1), (threads, 1, 1), 0, self.stream,
                        &mut [
                            &mut sp0 as *mut _ as *mut std::ffi::c_void,
                            &mut sp1 as *mut _ as *mut std::ffi::c_void,
                            &mut sp2 as *mut _ as *mut std::ffi::c_void,
                            &mut sp3 as *mut _ as *mut std::ffi::c_void,
                            &mut sp4 as *mut _ as *mut std::ffi::c_void,
                            &mut sp5 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            // DIAG: norms after conv split (before L2 norm, FP32)
            if la_diag {
                self.stream_sync()?;
                let q_f32 = self.diag_l2_norm_f32(la_q, 0, key_dim, key_dim);
                let k_f32 = self.diag_l2_norm_f32(la_k, 0, key_dim, key_dim);
                let v_f32 = self.diag_l2_norm_f32(la_v, 0, value_dim, value_dim);
                eprintln!("[DIAG L{:02}] after_conv_split pos0: q={:.6} k={:.6} v={:.6} (FP32, pre-l2norm)",
                    layer_idx, q_f32, k_f32, v_f32);
            }

            // 7. Compute gate and beta
            {
                let gt = std::cmp::max(32, ((std::cmp::min(1024, nv) + 31) / 32) * 32) as u32;
                let mut g0 = la_beta;
                let mut g1 = la_gate;
                let mut g2 = la_b;
                let mut g3 = la_a;
                let mut g4 = lw.la_a_log_ptr;
                let mut g5 = lw.la_dt_bias_ptr;
                let mut g6 = nv as i32;
                unsafe {
                    launch(self.kernels.la_compute_gate_beta,
                        (m as u32, 1, 1), (gt, 1, 1), 0, self.stream,
                        &mut [
                            &mut g0 as *mut _ as *mut std::ffi::c_void,
                            &mut g1 as *mut _ as *mut std::ffi::c_void,
                            &mut g2 as *mut _ as *mut std::ffi::c_void,
                            &mut g3 as *mut _ as *mut std::ffi::c_void,
                            &mut g4 as *mut _ as *mut std::ffi::c_void,
                            &mut g5 as *mut _ as *mut std::ffi::c_void,
                            &mut g6 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            // 8. Repeat-interleave k heads to match v heads (if hr > 1)
            if hr > 1 {
                // q: [M, nk, dk] -> [M, nv, dk]
                // k: [M, nk, dk] -> [M, nv, dk]
                // Need temporary buffers for expanded q and k.
                // Use v_beta and k_beta as temp (they'll be overwritten later).
                {
                    let dt = std::cmp::max(32, ((std::cmp::min(256, dk) + 31) / 32) * 32) as u32;
                    // Expand q
                    let mut r0 = la_v_beta;  // temp for expanded q [M, nv, dk]
                    let mut r1 = la_q;       // src [M, nk, dk]
                    let mut r2 = nk as i32;
                    let mut r3 = dk as i32;
                    let mut r4 = hr as i32;
                    unsafe {
                        launch(self.kernels.la_repeat_interleave,
                            (m as u32, nv as u32, 1), (dt, 1, 1), 0, self.stream,
                            &mut [
                                &mut r0 as *mut _ as *mut std::ffi::c_void,
                                &mut r1 as *mut _ as *mut std::ffi::c_void,
                                &mut r2 as *mut _ as *mut std::ffi::c_void,
                                &mut r3 as *mut _ as *mut std::ffi::c_void,
                                &mut r4 as *mut _ as *mut std::ffi::c_void,
                            ],
                        )?;
                    }
                    // Copy expanded q back
                    self.memcpy_d2d(la_q, la_v_beta, (m * nv * dk * 4) as u64)?;

                    // Expand k
                    r0 = la_v_beta;
                    r1 = la_k;
                    unsafe {
                        launch(self.kernels.la_repeat_interleave,
                            (m as u32, nv as u32, 1), (dt, 1, 1), 0, self.stream,
                            &mut [
                                &mut r0 as *mut _ as *mut std::ffi::c_void,
                                &mut r1 as *mut _ as *mut std::ffi::c_void,
                                &mut r2 as *mut _ as *mut std::ffi::c_void,
                                &mut r3 as *mut _ as *mut std::ffi::c_void,
                                &mut r4 as *mut _ as *mut std::ffi::c_void,
                            ],
                        )?;
                    }
                    self.memcpy_d2d(la_k, la_v_beta, (m * nv * dk * 4) as u64)?;
                }
            }

            // 9. L2 normalize q (with scale) and k (scale=1)
            {
                let lt = std::cmp::max(32, ((std::cmp::min(256, dk) + 31) / 32) * 32) as u32;
                let smem = lt * 4;
                let mut l0 = la_q;
                let mut l1 = scale;
                let mut l2 = nv as i32;
                let mut l3 = dk as i32;
                unsafe {
                    launch(self.kernels.la_l2norm_per_head,
                        (m as u32, nv as u32, 1), (lt, 1, 1), smem, self.stream,
                        &mut [
                            &mut l0 as *mut _ as *mut std::ffi::c_void,
                            &mut l1 as *mut _ as *mut std::ffi::c_void,
                            &mut l2 as *mut _ as *mut std::ffi::c_void,
                            &mut l3 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
                l0 = la_k;
                l1 = 1.0f32;
                unsafe {
                    launch(self.kernels.la_l2norm_per_head,
                        (m as u32, nv as u32, 1), (lt, 1, 1), smem, self.stream,
                        &mut [
                            &mut l0 as *mut _ as *mut std::ffi::c_void,
                            &mut l1 as *mut _ as *mut std::ffi::c_void,
                            &mut l2 as *mut _ as *mut std::ffi::c_void,
                            &mut l3 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            // DIAG: norms after L2 norm (scale applied to q, k unit-normed per head)
            if la_diag {
                self.stream_sync()?;
                let q_normed = self.diag_l2_norm_f32(la_q, 0, nv * dk, nv * dk);
                let k_normed = self.diag_l2_norm_f32(la_k, 0, nv * dk, nv * dk);
                eprintln!("[DIAG L{:02}] after_l2norm pos0: q={:.6} k={:.6} (q scaled by {:.6}, k by 1.0)",
                    layer_idx, q_normed, k_normed, scale);
                // Expected: each q head has norm = scale = 0.0884, so total q_norm = sqrt(nv) * scale
                // Expected: each k head has norm = 1.0, so total k_norm = sqrt(nv) * 1.0
                eprintln!("[DIAG L{:02}] expected: q={:.6} k={:.6}",
                    layer_idx, (nv as f32).sqrt() * scale, (nv as f32).sqrt());
            }

            // 10. Pad to multiple of chunk_size
            let pad_size = (chunk_size - m % chunk_size) % chunk_size;
            let total_len = m + pad_size;
            let num_chunks = total_len / chunk_size;

            if pad_size > 0 {
                // Zero-pad the tail of q, k, v, beta, gate
                unsafe {
                    let _ = cuda_sys::lib().cuMemsetD8Async(
                        la_q + (m * nv * dk * 4) as u64, 0, (pad_size * nv * dk * 4) as usize, self.stream);
                    let _ = cuda_sys::lib().cuMemsetD8Async(
                        la_k + (m * nv * dk * 4) as u64, 0, (pad_size * nv * dk * 4) as usize, self.stream);
                    let _ = cuda_sys::lib().cuMemsetD8Async(
                        la_v + (m * nv * dv * 4) as u64, 0, (pad_size * nv * dv * 4) as usize, self.stream);
                    let _ = cuda_sys::lib().cuMemsetD8Async(
                        la_beta + (m * nv * 4) as u64, 0, (pad_size * nv * 4) as usize, self.stream);
                    let _ = cuda_sys::lib().cuMemsetD8Async(
                        la_gate + (m * nv * 4) as u64, 0, (pad_size * nv * 4) as usize, self.stream);
                }
            }

            // 11. Compute beta-scaled values: v_beta = v * beta, k_beta = k * beta
            {
                let bt = std::cmp::max(32, ((std::cmp::min(256, std::cmp::max(dk, dv)) + 31) / 32) * 32) as u32;
                let mut bp0 = la_v_beta;
                let mut bp1 = la_k_beta;
                let mut bp2 = la_v;
                let mut bp3 = la_k;
                let mut bp4 = la_beta;
                let mut bp5 = nv as i32;
                let mut bp6 = dk as i32;
                let mut bp7 = dv as i32;
                unsafe {
                    launch(self.kernels.la_apply_beta,
                        (total_len as u32, nv as u32, 1), (bt, 1, 1), 0, self.stream,
                        &mut [
                            &mut bp0 as *mut _ as *mut std::ffi::c_void,
                            &mut bp1 as *mut _ as *mut std::ffi::c_void,
                            &mut bp2 as *mut _ as *mut std::ffi::c_void,
                            &mut bp3 as *mut _ as *mut std::ffi::c_void,
                            &mut bp4 as *mut _ as *mut std::ffi::c_void,
                            &mut bp5 as *mut _ as *mut std::ffi::c_void,
                            &mut bp6 as *mut _ as *mut std::ffi::c_void,
                            &mut bp7 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            // Transpose all arrays from [total_len, nv, dim] to [nv, total_len, dim]
            // via single kernel launch each (replaces millions of memcpy calls)

            // q: [total_len, nv, dk] -> [nv, total_len, dk]
            self.launch_transpose_3d_f32(fp32_scratch, la_q, total_len, nv, dk)?;
            self.memcpy_d2d(la_q, fp32_scratch, (nv * total_len * dk * 4) as u64)?;

            // k: [total_len, nv, dk] -> [nv, total_len, dk]
            self.launch_transpose_3d_f32(fp32_scratch, la_k, total_len, nv, dk)?;
            self.memcpy_d2d(la_k, fp32_scratch, (nv * total_len * dk * 4) as u64)?;

            // v: [total_len, nv, dv] -> [nv, total_len, dv]
            self.launch_transpose_3d_f32(fp32_scratch, la_v, total_len, nv, dv)?;
            self.memcpy_d2d(la_v, fp32_scratch, (nv * total_len * dv * 4) as u64)?;

            // gate: [total_len, nv] -> [nv, total_len] (2D, use existing transpose)
            self.launch_transpose_f32(fp32_scratch, la_gate, total_len, nv)?;
            self.memcpy_d2d(la_gate, fp32_scratch, (nv * total_len * 4) as u64)?;

            // v_beta: [total_len, nv, dv] -> [nv, total_len, dv]
            self.launch_transpose_3d_f32(fp32_scratch, la_v_beta, total_len, nv, dv)?;
            self.memcpy_d2d(la_v_beta, fp32_scratch, (nv * total_len * dv * 4) as u64)?;

            // k_beta: [total_len, nv, dk] -> [nv, total_len, dk]
            self.launch_transpose_3d_f32(fp32_scratch, la_k_beta, total_len, nv, dk)?;
            self.memcpy_d2d(la_k_beta, fp32_scratch, (nv * total_len * dk * 4) as u64)?;

            // 12. Compute cumulative sum of g within chunks
            // la_gate is now [nv, total_len] = [nv, num_chunks * CS]
            // = [nv, num_chunks, CS] when reshaped
            {
                let mut cs0 = la_g_cum;
                let mut cs1 = la_gate;
                let mut cs2 = num_chunks as i32;
                let mut cs3 = chunk_size as i32;
                unsafe {
                    launch(self.kernels.la_cumsum,
                        (nv as u32, num_chunks as u32, 1), (1, 1, 1), 0, self.stream,
                        &mut [
                            &mut cs0 as *mut _ as *mut std::ffi::c_void,
                            &mut cs1 as *mut _ as *mut std::ffi::c_void,
                            &mut cs2 as *mut _ as *mut std::ffi::c_void,
                            &mut cs3 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            // 13. Build attn matrix (nilpotent A): attn[i,j] = -(k_beta @ k^T) * decay
            // k_beta: [nv, num_chunks, CS, dk], k: [nv, num_chunks, CS, dk]
            // attn: [nv, num_chunks, CS, CS]
            {
                let bt = chunk_size as u32;
                let mut am0 = la_attn;
                let mut am1 = la_k_beta;
                let mut am2 = la_k;
                let mut am3 = la_g_cum;
                let mut am4 = num_chunks as i32;
                let mut am5 = chunk_size as i32;
                let mut am6 = dk as i32;
                unsafe {
                    launch(self.kernels.la_build_attn_matrix,
                        (nv as u32, num_chunks as u32, 1), (bt, 1, 1), 0, self.stream,
                        &mut [
                            &mut am0 as *mut _ as *mut std::ffi::c_void,
                            &mut am1 as *mut _ as *mut std::ffi::c_void,
                            &mut am2 as *mut _ as *mut std::ffi::c_void,
                            &mut am3 as *mut _ as *mut std::ffi::c_void,
                            &mut am4 as *mut _ as *mut std::ffi::c_void,
                            &mut am5 as *mut _ as *mut std::ffi::c_void,
                            &mut am6 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            // 14. Triangular solve: (I - A) * value_corrected = v_beta
            // Copy v_beta to la_v_new_full (will be overwritten with solution)
            // Actually la_v_beta is [nv, total_len, dv] and we solve per-chunk.
            // The triangular solve kernel operates on [nv, num_chunks, CS, dv] = [nv, total_len, dv]
            // It modifies v_beta in-place to become value_corrected.
            {
                let st = std::cmp::max(32, ((std::cmp::min(256, dv) + 31) / 32) * 32) as u32;
                let mut ts0 = la_v_beta; // in/out (b -> x)
                let mut ts1 = la_attn;   // A matrix
                let mut ts2 = num_chunks as i32;
                let mut ts3 = chunk_size as i32;
                let mut ts4 = dv as i32;
                unsafe {
                    launch(self.kernels.la_triangular_solve,
                        (nv as u32, num_chunks as u32, 1), (st, 1, 1), 0, self.stream,
                        &mut [
                            &mut ts0 as *mut _ as *mut std::ffi::c_void,
                            &mut ts1 as *mut _ as *mut std::ffi::c_void,
                            &mut ts2 as *mut _ as *mut std::ffi::c_void,
                            &mut ts3 as *mut _ as *mut std::ffi::c_void,
                            &mut ts4 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            // 15. Second triangular solve: (I - A) * k_cumdecay = k_beta * exp(g_cum)
            // First, scale k_beta by exp(g_cum)
            {
                let st = std::cmp::max(32, ((std::cmp::min(256, dk) + 31) / 32) * 32) as u32;
                // Use fp32_scratch as temp for k_beta * exp(g_cum)
                let mut se0 = fp32_scratch; // output
                let mut se1 = la_k_beta;    // input
                let mut se2 = la_g_cum;
                let mut se3 = nv as i32;
                let mut se4 = total_len as i32;
                let mut se5 = dk as i32;
                unsafe {
                    launch(self.kernels.la_scale_by_exp_g,
                        (nv as u32, total_len as u32, 1), (st, 1, 1), 0, self.stream,
                        &mut [
                            &mut se0 as *mut _ as *mut std::ffi::c_void,
                            &mut se1 as *mut _ as *mut std::ffi::c_void,
                            &mut se2 as *mut _ as *mut std::ffi::c_void,
                            &mut se3 as *mut _ as *mut std::ffi::c_void,
                            &mut se4 as *mut _ as *mut std::ffi::c_void,
                            &mut se5 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
                // Copy to k_beta (will be solved in place)
                self.memcpy_d2d(la_k_beta, fp32_scratch, (nv * total_len * dk * 4) as u64)?;

                // Triangular solve on k_beta
                let mut ts0 = la_k_beta;
                let mut ts1 = la_attn;
                let mut ts2 = num_chunks as i32;
                let mut ts3 = chunk_size as i32;
                let mut ts4 = dk as i32;
                unsafe {
                    launch(self.kernels.la_triangular_solve,
                        (nv as u32, num_chunks as u32, 1), (st, 1, 1), 0, self.stream,
                        &mut [
                            &mut ts0 as *mut _ as *mut std::ffi::c_void,
                            &mut ts1 as *mut _ as *mut std::ffi::c_void,
                            &mut ts2 as *mut _ as *mut std::ffi::c_void,
                            &mut ts3 as *mut _ as *mut std::ffi::c_void,
                            &mut ts4 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            // 16. Initialize recurrent state (zero)
            unsafe {
                let _ = cuda_sys::lib().cuMemsetD8Async(
                    la_state, 0, (nv * dk * dv * 4) as usize, self.stream);
            }

            // 17. Chunk recurrence loop (zero-copy strided kernels)
            //
            // Strided kernels read directly from [nv, total_len, dim] arrays
            // and write output directly to the [nv, total_len, dv] output buffer.
            // Zero per-head memcpy calls. Three kernel launches per chunk.
            //
            // la_v_beta: value_corrected [nv, total_len, dv] (strided)
            // la_k_beta: k_cumdecay [nv, total_len, dk] (strided)
            // la_q:  [nv, total_len, dk] (strided)
            // la_k:  [nv, total_len, dk] (strided)
            // la_g_cum: [nv, total_len] = [nv, num_chunks*CS] (strided)
            // la_v_new: [nv, CS, dv] (contiguous per-chunk temp)
            // la_state: [nv, dk, dv] (contiguous, carries forward)
            // output_buf = fp32_scratch: [nv, total_len, dv] (strided output)

            let cs = chunk_size;
            let output_buf = fp32_scratch;
            let vt = std::cmp::max(32, ((std::cmp::min(256, dv) + 31) / 32) * 32) as u32;

            for c in 0..num_chunks {
                // 1. v_new = v_corr(strided) - k_cumd(strided) @ state
                {
                    let mut vn0 = la_v_new;    // output [nv, CS, dv] contiguous
                    let mut vn1 = la_v_beta;   // v_corr [nv, total_len, dv] strided
                    let mut vn2 = la_k_beta;   // k_cumd [nv, total_len, dk] strided
                    let mut vn3 = la_state;    // state [nv, dk, dv]
                    let mut vn4 = cs as i32;
                    let mut vn5 = dk as i32;
                    let mut vn6 = dv as i32;
                    let mut vn7 = total_len as i32;
                    let mut vn8 = c as i32;
                    unsafe {
                        launch(self.kernels.la_compute_v_new_strided,
                            (nv as u32, 1, 1), (vt, 1, 1), 0, self.stream,
                            &mut [
                                &mut vn0 as *mut _ as *mut std::ffi::c_void,
                                &mut vn1 as *mut _ as *mut std::ffi::c_void,
                                &mut vn2 as *mut _ as *mut std::ffi::c_void,
                                &mut vn3 as *mut _ as *mut std::ffi::c_void,
                                &mut vn4 as *mut _ as *mut std::ffi::c_void,
                                &mut vn5 as *mut _ as *mut std::ffi::c_void,
                                &mut vn6 as *mut _ as *mut std::ffi::c_void,
                                &mut vn7 as *mut _ as *mut std::ffi::c_void,
                                &mut vn8 as *mut _ as *mut std::ffi::c_void,
                            ],
                        )?;
                    }
                }

                // 2. output(strided) = (q*exp(g))@state + tril(q@k^T*decay)@v_new
                {
                    let mut co0 = output_buf;  // [nv, total_len, dv] strided output
                    let mut co1 = la_q;        // [nv, total_len, dk] strided
                    let mut co2 = la_k;        // [nv, total_len, dk] strided
                    let mut co3 = la_v_new;    // [nv, CS, dv] contiguous
                    let mut co4 = la_g_cum;    // [nv, total_len] strided
                    let mut co5 = la_state;    // [nv, dk, dv]
                    let mut co6 = cs as i32;
                    let mut co7 = dk as i32;
                    let mut co8 = dv as i32;
                    let mut co9 = total_len as i32;
                    let mut co10 = c as i32;
                    unsafe {
                        launch(self.kernels.la_chunk_output_strided,
                            (nv as u32, 1, 1), (vt, 1, 1), 0, self.stream,
                            &mut [
                                &mut co0 as *mut _ as *mut std::ffi::c_void,
                                &mut co1 as *mut _ as *mut std::ffi::c_void,
                                &mut co2 as *mut _ as *mut std::ffi::c_void,
                                &mut co3 as *mut _ as *mut std::ffi::c_void,
                                &mut co4 as *mut _ as *mut std::ffi::c_void,
                                &mut co5 as *mut _ as *mut std::ffi::c_void,
                                &mut co6 as *mut _ as *mut std::ffi::c_void,
                                &mut co7 as *mut _ as *mut std::ffi::c_void,
                                &mut co8 as *mut _ as *mut std::ffi::c_void,
                                &mut co9 as *mut _ as *mut std::ffi::c_void,
                                &mut co10 as *mut _ as *mut std::ffi::c_void,
                            ],
                        )?;
                    }
                }

                // 3. state update: reads strided k, g_cum; reads contiguous v_new
                {
                    let mut su0 = la_state;    // [nv, dk, dv] in/out
                    let mut su1 = la_k;        // [nv, total_len, dk] strided
                    let mut su2 = la_v_new;    // [nv, CS, dv] contiguous
                    let mut su3 = la_g_cum;    // [nv, total_len] strided
                    let mut su4 = cs as i32;
                    let mut su5 = dk as i32;
                    let mut su6 = dv as i32;
                    let mut su7 = total_len as i32;
                    let mut su8 = c as i32;
                    unsafe {
                        launch(self.kernels.la_state_update_strided,
                            (nv as u32, 1, 1), (vt, 1, 1), 0, self.stream,
                            &mut [
                                &mut su0 as *mut _ as *mut std::ffi::c_void,
                                &mut su1 as *mut _ as *mut std::ffi::c_void,
                                &mut su2 as *mut _ as *mut std::ffi::c_void,
                                &mut su3 as *mut _ as *mut std::ffi::c_void,
                                &mut su4 as *mut _ as *mut std::ffi::c_void,
                                &mut su5 as *mut _ as *mut std::ffi::c_void,
                                &mut su6 as *mut _ as *mut std::ffi::c_void,
                                &mut su7 as *mut _ as *mut std::ffi::c_void,
                                &mut su8 as *mut _ as *mut std::ffi::c_void,
                            ],
                        )?;
                    }
                }
            }

            // 18. Copy recurrent state to decode state buffer
            if lw.la_recur_state_ptr != 0 {
                self.memcpy_d2d(lw.la_recur_state_ptr, la_state, (nv * dk * dv * 4) as u64)?;
            }

            // 19. Transpose output [nv, total_len, dv] -> [total_len, nv, dv]
            // Use transpose_3d kernel (output_buf has [nv, total_len, dv], we want [total_len, nv, dv])
            // Note: we use total_len not m because the output buffer was written with total_len
            // (which includes padding). We only use the first m positions after transpose.
            self.launch_transpose_3d_f32(la_v, output_buf, nv, total_len, dv)?;
            // la_v now has [total_len, nv, dv] FP32 — first m positions are valid

            if la_diag {
                // Recurrence output norm at pos 0 (before gated RMSNorm)
                let recur_norm = self.diag_l2_norm_f32(la_v, 0, nv * dv, nv * dv);
                // Z gate norm at pos 0 (BF16)
                let z_norm = self.diag_l2_norm(la_z, 0, value_dim, value_dim);
                eprintln!("[DIAG L{:02}] recurrence pos0 norm={:.6} (before gated_rmsnorm)", layer_idx, recur_norm);
                eprintln!("[DIAG L{:02}] z_gate pos0 norm={:.6}", layer_idx, z_norm);
            }

            // 20. Gated RMSNorm: out = rmsnorm(x) * weight * silu(z)
            {
                let nt = std::cmp::max(32, ((std::cmp::min(256, dv) + 31) / 32) * 32) as u32;
                let smem = nt * 4;
                let mut n0 = attn_out; // BF16 output [M, nv*dv]
                let mut n1 = la_v;     // FP32 input [M, nv, dv]
                let mut n2 = la_z;     // BF16 gate [M, nv, dv]
                let mut n3 = lw.la_norm_weight_ptr; // BF16 weight [dv]
                let mut n4 = nv as i32;
                let mut n5 = dv as i32;
                let mut n6 = cfg.rms_norm_eps;
                unsafe {
                    launch(self.kernels.la_gated_rmsnorm,
                        (m as u32, nv as u32, 1), (nt, 1, 1), smem, self.stream,
                        &mut [
                            &mut n0 as *mut _ as *mut std::ffi::c_void,
                            &mut n1 as *mut _ as *mut std::ffi::c_void,
                            &mut n2 as *mut _ as *mut std::ffi::c_void,
                            &mut n3 as *mut _ as *mut std::ffi::c_void,
                            &mut n4 as *mut _ as *mut std::ffi::c_void,
                            &mut n5 as *mut _ as *mut std::ffi::c_void,
                            &mut n6 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            // 21. Output projection: [M, nv*dv] BF16 -> [M, hidden] BF16
            // Use scratch1 as temp to avoid Marlin GEMM input/output aliasing.
            if la_diag {
                let pre_oproj_norm = self.diag_l2_norm(attn_out, 0, value_dim, value_dim);
                eprintln!("[DIAG L{:02}] pre_out_proj pos0 norm={:.6} (after gated_rmsnorm, dim={})", layer_idx, pre_oproj_norm, value_dim);
            }
            let o_temp = *self.scratch.d_scratch1.device_ptr();
            self.la_gemm(attn_out, &lw.la_out_proj, &lw.la_out_proj_bf16, o_temp, m)?;
            self.memcpy_d2d(attn_out, o_temp, (m * cfg.hidden_size * 2) as u64)?;
            if la_diag {
                let post_oproj_norm = self.diag_l2_norm(attn_out, 0, cfg.hidden_size, cfg.hidden_size);
                eprintln!("[DIAG L{:02}] post_out_proj pos0 norm={:.6} (mixer output, dim={})", layer_idx, post_oproj_norm, cfg.hidden_size);
            }

        }

        Ok(())
    }

    // ── Dense MLP ──

    fn forward_dense_mlp(&self, layer_idx: usize, m: usize) -> Result<(), String> {
        let lw = &self.layer_weights[layer_idx];
        let w1 = lw.shared_w1.as_ref().ok_or("missing w1")?;
        let w2 = lw.shared_w2.as_ref().ok_or("missing w2")?;

        let hidden = *self.scratch.d_hidden.device_ptr();
        let s1 = *self.scratch.d_scratch1.device_ptr();
        let s2 = *self.scratch.d_scratch2.device_ptr();

        // gate_up = hidden @ w1 -> [m, 2*inter]
        self.marlin_gemm(hidden, w1, s1, m)?;

        // SiLU + mul
        let inter = w1.n / 2;
        let t = std::cmp::max(32, ((std::cmp::min(1024, inter) + 31) / 32) * 32);
        let mut sm0 = s2; let mut sm1 = s1; let mut sm2 = inter as i32;
        unsafe {
            launch(self.kernels.silu_mul,
                (m as u32, 1, 1), (t as u32, 1, 1), 0, self.stream,
                &mut [
                    &mut sm0 as *mut _ as *mut std::ffi::c_void,
                    &mut sm1 as *mut _ as *mut std::ffi::c_void,
                    &mut sm2 as *mut _ as *mut std::ffi::c_void,
                ],
            )?;
        }

        // down = inter @ w2 -> [m, hidden]
        self.marlin_gemm(s2, w2, hidden, m)?;

        Ok(())
    }

    // ── MoE Forward ──

    fn forward_moe(&mut self, layer_idx: usize, m: usize) -> Result<(), String> {
        // TODO: Fused MoE path disabled for debugging - sequential path uses proven Marlin GEMM
        // if self.kernels.fused_moe_fn.is_some() && self.d_fused_input.is_some() {
        //     return self.forward_moe_fused(layer_idx, m);
        // }
        // Per-expert sequential dispatch
        self.forward_moe_sequential(layer_idx, m)
    }

    /// Fused MoE forward using MarlinDefault kernel.
    /// One kernel launch handles ALL active experts per GEMM (w1, w2).
    fn forward_moe_fused(&mut self, layer_idx: usize, m: usize) -> Result<(), String> {
        let diag_moe = std::env::var("KRASIS_PREFILL_DIAG").is_ok();
        let lw = &self.layer_weights[layer_idx];
        let h = self.config.hidden_size;
        let n_experts = lw.moe_num_experts;
        let topk = lw.moe_topk;
        let inter = self.config.moe_intermediate_size;
        let scoring_func = lw.moe_scoring_func;
        let gated = lw.moe_gated;
        let activation = lw.moe_activation;
        let scale_factor = lw.moe_routed_scaling_factor;
        let gs = self.config.group_size;
        let bits = self.config.expert_bits;

        let hidden = *self.scratch.d_hidden.device_ptr();
        let gate_out = *self.scratch.d_gate_out.device_ptr();
        let topk_ids_ptr = *self.scratch.d_topk_ids.device_ptr();
        let topk_weights_ptr = *self.scratch.d_topk_weights.device_ptr();

        // 1. Gate GEMM (same as sequential)
        let gate_ptr = lw.moe_gate_ptr;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            use cudarc::cublas::sys as cublas_sys;
            use cudarc::cublas::result as cublas_result;
            cublas_result::set_stream(
                self.cublas_handle,
                self.stream as cublas_sys::cudaStream_t,
            ).map_err(|e| format!("cublas set_stream: {:?}", e))?;
            cublas_result::gemm_ex(
                self.cublas_handle,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n_experts as i32, m as i32, h as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                gate_ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType::CUDA_R_16BF, h as i32,
                hidden as *const std::ffi::c_void,
                cublas_sys::cudaDataType::CUDA_R_16BF, h as i32,
                &beta as *const f32 as *const std::ffi::c_void,
                gate_out as *mut std::ffi::c_void,
                cublas_sys::cudaDataType::CUDA_R_32F, n_experts as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            ).map_err(|e| format!("cublas gate GEMM: {:?}", e))?;
        }

        // 2. Top-K routing
        {
            let t = std::cmp::max(32, ((std::cmp::min(1024, n_experts) + 31) / 32) * 32) as u32;
            let mut p0 = topk_weights_ptr;
            let mut p1 = topk_ids_ptr;
            let mut p2 = gate_out;
            let mut p3 = n_experts as i32;
            let mut p4 = topk as i32;
            let kernel = if scoring_func == 1 { self.kernels.sigmoid_topk } else { self.kernels.softmax_topk };
            let smem = (n_experts * 4 + topk * 8 + if scoring_func == 0 { 32 * 4 } else { 0 }) as u32;
            unsafe {
                launch(kernel,
                    (m as u32, 1, 1), (t, 1, 1), smem, self.stream,
                    &mut [
                        &mut p0 as *mut _ as *mut std::ffi::c_void,
                        &mut p1 as *mut _ as *mut std::ffi::c_void,
                        &mut p2 as *mut _ as *mut std::ffi::c_void,
                        &mut p3 as *mut _ as *mut std::ffi::c_void,
                        &mut p4 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // 3. moe_align_block_size: padded prefix sum + build sorted token/expert maps
        let block_size = 64i32; // MarlinDefault max thread_m_blocks=4 -> max block_size=64
        let expert_counts_ptr = *self.scratch.d_expert_counts.device_ptr();
        let expert_offsets_ptr = *self.scratch.d_expert_offsets.device_ptr();
        let write_offsets_ptr = *self.scratch.d_write_offsets.device_ptr();

        // Zero counts and write offsets
        unsafe {
            cuda_sys::lib().cuMemsetD8Async(expert_counts_ptr, 0, (n_experts * 4) as usize, self.stream);
            cuda_sys::lib().cuMemsetD8Async(write_offsets_ptr, 0, (n_experts * 4) as usize, self.stream);
        }

        // Count tokens per expert
        {
            let mut p0 = expert_counts_ptr;
            let mut p1 = topk_ids_ptr;
            let mut p2 = m as i32;
            let mut p3 = topk as i32;
            let mut p4 = n_experts as i32;
            unsafe {
                launch(self.kernels.moe_count_experts,
                    (m as u32, 1, 1), (1, 1, 1), 0, self.stream,
                    &mut [
                        &mut p0 as *mut _ as *mut std::ffi::c_void,
                        &mut p1 as *mut _ as *mut std::ffi::c_void,
                        &mut p2 as *mut _ as *mut std::ffi::c_void,
                        &mut p3 as *mut _ as *mut std::ffi::c_void,
                        &mut p4 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Padded prefix sum (block_size aligned)
        let sorted_ids_ptr = self.d_sorted_token_ids.as_ref()
            .ok_or("sorted_token_ids not allocated")?.device_ptr();
        let fused_expert_ids_ptr = self.d_fused_expert_ids.as_ref()
            .ok_or("fused_expert_ids not allocated")?.device_ptr();
        let num_tokens_post_ptr = self.d_num_tokens_post.as_ref()
            .ok_or("num_tokens_post not allocated")?.device_ptr();
        {
            let ps_threads = std::cmp::max(32, ((std::cmp::min(1024, n_experts) + 31) / 32) * 32) as u32;
            let mut p0 = expert_offsets_ptr;
            let mut p1 = *num_tokens_post_ptr;
            let mut p2 = expert_counts_ptr;
            let mut p3 = n_experts as i32;
            let mut p4 = block_size;
            unsafe {
                launch(self.kernels.moe_padded_prefix_sum,
                    (1, 1, 1), (ps_threads, 1, 1), (n_experts * 4) as u32, self.stream,
                    &mut [
                        &mut p0 as *mut _ as *mut std::ffi::c_void,
                        &mut p1 as *mut _ as *mut std::ffi::c_void,
                        &mut p2 as *mut _ as *mut std::ffi::c_void,
                        &mut p3 as *mut _ as *mut std::ffi::c_void,
                        &mut p4 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Build sorted token/expert maps
        {
            let mut p0 = *sorted_ids_ptr;
            let mut p1 = *fused_expert_ids_ptr;
            let mut p2 = write_offsets_ptr;
            let mut p3 = topk_ids_ptr;
            let mut p4 = expert_offsets_ptr;
            let mut p5 = expert_counts_ptr;
            let mut p6 = m as i32;
            let mut p7 = topk as i32;
            let mut p8 = n_experts as i32;
            let mut p9 = block_size;
            unsafe {
                launch(self.kernels.moe_scatter_sorted,
                    (m as u32, 1, 1), (1, 1, 1), 0, self.stream,
                    &mut [
                        &mut p0 as *mut _ as *mut std::ffi::c_void,
                        &mut p1 as *mut _ as *mut std::ffi::c_void,
                        &mut p2 as *mut _ as *mut std::ffi::c_void,
                        &mut p3 as *mut _ as *mut std::ffi::c_void,
                        &mut p4 as *mut _ as *mut std::ffi::c_void,
                        &mut p5 as *mut _ as *mut std::ffi::c_void,
                        &mut p6 as *mut _ as *mut std::ffi::c_void,
                        &mut p7 as *mut _ as *mut std::ffi::c_void,
                        &mut p8 as *mut _ as *mut std::ffi::c_void,
                        &mut p9 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // DIAG: dump routing info for layer 0
        if diag_moe && layer_idx == 0 {
            self.stream_sync()?;
            // topk_weights for token 0: [topk] floats
            let mut h_tw = vec![0.0f32; topk];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_tw.as_mut_ptr() as *mut _,
                    topk_weights_ptr, (topk * 4) as usize);
            }
            let mut h_ti = vec![0i32; topk];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_ti.as_mut_ptr() as *mut _,
                    topk_ids_ptr, (topk * 4) as usize);
            }
            eprintln!("[DIAG MoE L0] topk_ids[tok0]={:?}", h_ti);
            eprintln!("[DIAG MoE L0] topk_weights[tok0]={:?}", h_tw);
            eprintln!("[DIAG MoE L0] scale_factor={}, scoring_func={}, n_experts={}, topk={}",
                scale_factor, scoring_func, n_experts, topk);
        }

        // 4. Load expert weights for this layer into contiguous GPU buffer
        //    Uses the current buffer (A or B) of the double-buffer pair
        let moe_layer_idx = lw.moe_layer_idx;
        let cur = self.fused_expert_buf_cur;
        let (w1_buf, w1s_buf, w2_buf, w2s_buf) = if cur == 0 {
            (self.d_fused_expert_w1_a.as_ref(), self.d_fused_expert_w1s_a.as_ref(),
             self.d_fused_expert_w2_a.as_ref(), self.d_fused_expert_w2s_a.as_ref())
        } else {
            (self.d_fused_expert_w1_b.as_ref(), self.d_fused_expert_w1s_b.as_ref(),
             self.d_fused_expert_w2_b.as_ref(), self.d_fused_expert_w2s_b.as_ref())
        };
        let w1_base = *w1_buf.ok_or("fused w1 buf")?.device_ptr();
        let w1s_base = *w1s_buf.ok_or("fused w1s buf")?.device_ptr();
        let w2_base = *w2_buf.ok_or("fused w2 buf")?.device_ptr();
        let w2s_base = *w2s_buf.ok_or("fused w2s buf")?.device_ptr();

        // Skip preload if this layer was already preloaded by group pipelining
        if self.preloaded_moe_layer != Some(layer_idx) {
            if let Some(moe_idx) = moe_layer_idx {
                if let Some(Some(moe_data)) = self.moe_layers.get(moe_idx) {
                    // DMA all experts for this layer into contiguous buffer
                    for eid in 0..moe_data.experts.len() {
                        let e = &moe_data.experts[eid];
                        let w1_offset = eid * self.w1_packed_per_expert;
                        let w1s_offset = eid * self.w1_scales_per_expert;
                        let w2_offset = eid * self.w2_packed_per_expert;
                        let w2s_offset = eid * self.w2_scales_per_expert;

                        // Check HCS first
                        if let Some((hcs_w13p, hcs_w13s, hcs_w2p, hcs_w2s)) = self.hcs_lookup(moe_idx, eid) {
                            // GPU-to-GPU copy from HCS location
                            unsafe {
                                cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                                    w1_base + w1_offset as u64, hcs_w13p,
                                    self.w1_packed_per_expert, self.copy_stream);
                                cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                                    w1s_base + w1s_offset as u64, hcs_w13s,
                                    self.w1_scales_per_expert, self.copy_stream);
                                cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                                    w2_base + w2_offset as u64, hcs_w2p,
                                    self.w2_packed_per_expert, self.copy_stream);
                                cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                                    w2s_base + w2s_offset as u64, hcs_w2s,
                                    self.w2_scales_per_expert, self.copy_stream);
                            }
                        } else {
                            // CPU-to-GPU DMA
                            unsafe {
                                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                    w1_base + w1_offset as u64,
                                    e.w13_packed_ptr as *const _,
                                    e.w13_packed_bytes, self.copy_stream);
                                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                    w1s_base + w1s_offset as u64,
                                    e.w13_scales_ptr as *const _,
                                    e.w13_scales_bytes, self.copy_stream);
                                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                    w2_base + w2_offset as u64,
                                    e.w2_packed_ptr as *const _,
                                    e.w2_packed_bytes, self.copy_stream);
                                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                    w2s_base + w2s_offset as u64,
                                    e.w2_scales_ptr as *const _,
                                    e.w2_scales_bytes, self.copy_stream);
                            }
                        }
                    }
                    // Record DMA event so compute stream waits for it
                    unsafe {
                        cuda_sys::lib().cuEventRecord(self.dma_event, self.copy_stream);
                    }
                }
            }
        }

        // Wait for DMA to complete
        unsafe {
            cuda_sys::lib().cuStreamWaitEvent(self.stream, self.dma_event, 0);
        }

        // 5. Launch shared expert on dedicated shared_stream (gap 4: always async)
        let has_shared = lw.shared_w1.is_some() && lw.shared_w2.is_some();
        if has_shared {
            unsafe {
                cuda_sys::lib().cuEventRecord(self.compute_event, self.stream);
                cuda_sys::lib().cuStreamWaitEvent(self.shared_stream, self.compute_event, 0);
            }
            self.launch_shared_expert_on_shared_stream(layer_idx, m)?;
        }

        // 6. Download num_tokens_post_padded to know total_sorted for fused kernel
        let mut h_num_post = [0i32; 1];
        self.stream_sync()?;
        unsafe {
            cuda_sys::lib().cuMemcpyDtoH_v2(
                h_num_post.as_mut_ptr() as *mut _,
                *num_tokens_post_ptr,
                4,
            );
        }
        let total_sorted = h_num_post[0] as usize;

        // 7. Gather input tokens to sorted order
        let fused_input_ptr = *self.d_fused_input.as_ref().ok_or("fused_input")?.device_ptr();
        if total_sorted > 0 {
            let gt = std::cmp::max(32, ((std::cmp::min(1024, h) + 31) / 32) * 32) as u32;
            let mut g0 = fused_input_ptr;
            let mut g1 = hidden;
            let mut g2 = *sorted_ids_ptr;
            let mut g3 = h as i32;
            let mut g4 = m as i32;
            unsafe {
                launch(self.kernels.moe_gather_sorted,
                    (total_sorted as u32, 1, 1), (gt, 1, 1), 0, self.stream,
                    &mut [
                        &mut g0 as *mut _ as *mut std::ffi::c_void,
                        &mut g1 as *mut _ as *mut std::ffi::c_void,
                        &mut g2 as *mut _ as *mut std::ffi::c_void,
                        &mut g3 as *mut _ as *mut std::ffi::c_void,
                        &mut g4 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // DIAG: check gather output and total_sorted
        if diag_moe && layer_idx == 0 {
            eprintln!("[DIAG MoE L0] total_sorted={}", total_sorted);
            if total_sorted > 0 {
                self.stream_sync()?;
                // Check hidden (MLP input) at token 0
                let mut h_hid = vec![0u16; 8];
                unsafe {
                    cuda_sys::lib().cuMemcpyDtoH_v2(
                        h_hid.as_mut_ptr() as *mut _,
                        hidden, 16);
                }
                let hid_f32: Vec<f32> = h_hid.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
                eprintln!("[DIAG MoE L0] hidden(mlp_input)[0..8] = {:?}", hid_f32);

                // Find first non-padding sorted position
                let mut h_sids = vec![0i32; std::cmp::min(16, total_sorted)];
                unsafe {
                    cuda_sys::lib().cuMemcpyDtoH_v2(
                        h_sids.as_mut_ptr() as *mut _,
                        *sorted_ids_ptr, (h_sids.len() * 4) as usize);
                }
                eprintln!("[DIAG MoE L0] sorted_ids[0..{}] = {:?}", h_sids.len(), h_sids);

                // Check gathered input at first REAL token (non-padding)
                let first_real = h_sids.iter().position(|&id| id < m as i32).unwrap_or(0);
                let mut h_gi = vec![0u16; 8];
                unsafe {
                    cuda_sys::lib().cuMemcpyDtoH_v2(
                        h_gi.as_mut_ptr() as *mut _,
                        fused_input_ptr + (first_real * h * 2) as u64, 16);
                }
                let gi_f32: Vec<f32> = h_gi.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
                eprintln!("[DIAG MoE L0] fused_input[sorted_pos={}(tok={})][0..8] = {:?}",
                    first_real, h_sids[first_real], gi_f32);

                // Check expert_ids
                let n_blocks = (total_sorted + 63) / 64; // block_size=64
                let mut h_eids = vec![0i32; std::cmp::min(16, n_blocks)];
                unsafe {
                    cuda_sys::lib().cuMemcpyDtoH_v2(
                        h_eids.as_mut_ptr() as *mut _,
                        *fused_expert_ids_ptr, (h_eids.len() * 4) as usize);
                }
                eprintln!("[DIAG MoE L0] expert_ids[0..{}] = {:?}", h_eids.len(), h_eids);
                eprintln!("[DIAG MoE L0] n_blocks={}, w1_packed_per_expert={}, w1_scales_per_expert={}",
                    n_blocks, self.w1_packed_per_expert, self.w1_scales_per_expert);

                // Check expert_counts for some active experts
                let mut h_counts = vec![0i32; n_experts];
                unsafe {
                    cuda_sys::lib().cuMemcpyDtoH_v2(
                        h_counts.as_mut_ptr() as *mut _,
                        expert_counts_ptr, (n_experts * 4) as usize);
                }
                let nonzero: Vec<(usize,i32)> = h_counts.iter().enumerate()
                    .filter(|(_, &c)| c > 0).take(10).map(|(i,&c)| (i,c)).collect();
                let total_count: i32 = h_counts.iter().sum();
                eprintln!("[DIAG MoE L0] expert_counts total={}, first 10 nonzero={:?}", total_count, nonzero);

                // Check expert_offsets at active expert indices
                let mut h_offs_all = vec![0i32; n_experts + 1];
                unsafe {
                    cuda_sys::lib().cuMemcpyDtoH_v2(
                        h_offs_all.as_mut_ptr() as *mut _,
                        expert_offsets_ptr, ((n_experts + 1) * 4) as usize);
                }
                // Show offsets for experts with tokens
                let nonzero_offs: Vec<(usize,i32,i32)> = nonzero.iter().take(5)
                    .map(|&(e,c)| (e, h_offs_all[e], c)).collect();
                eprintln!("[DIAG MoE L0] expert_offsets (eid,offset,count) = {:?}", nonzero_offs);
                eprintln!("[DIAG MoE L0] expert_offsets[E]={} (should = total_sorted={})", h_offs_all[n_experts], total_sorted);
            }
        }

        // 8. Call MarlinDefault for w1 (gate_up projection)
        let fused_fn = self.kernels.fused_moe_fn.ok_or("fused MoE fn not loaded")?;
        let w1_n = if gated { 2 * inter } else { inter };
        let num_groups_w1 = (h / gs) as i32;
        let fused_inter_ptr = *self.d_fused_inter_cache.as_ref().ok_or("fused_inter_cache")?.device_ptr();
        let fused_c_tmp_ptr = *self.d_fused_c_tmp.as_ref().ok_or("fused_c_tmp")?.device_ptr();

        // Vendored Marlin MoE uses CUDA runtime default stream.
        // Explicit sync before/after to ensure correct ordering.
        self.stream_sync()?;

        let gs = self.config.group_size as i32;
        let q_type_ptr = &self.q_type as *const ScalarType as *const std::ffi::c_void;

        // Zero the workspace lock array before fused w1 GEMM (runs on CUDA default stream).
        let ws_ptr = *self.scratch.d_workspace.device_ptr();
        let ws_len = self.config.sms * 4;
        unsafe {
            cuda_sys::lib().cuMemsetD32Async(ws_ptr, 0, ws_len, std::ptr::null_mut());
        }

        unsafe {
            fused_fn(
                fused_input_ptr as *const _,        // A: [total_sorted, K=hidden]
                w1_base as *const _,                 // B: [E, K/16, N, pack]
                fused_inter_ptr as *mut _,           // C: [total_sorted, N=w1_n]
                fused_c_tmp_ptr as *mut _,           // C_tmp
                std::ptr::null(),                    // b_bias (none)
                w1s_base as *const _,                // scales: [E, num_groups, N]
                std::ptr::null(),                    // s2 (none)
                std::ptr::null(),                    // zp (none)
                std::ptr::null(),                    // g_idx (none)
                std::ptr::null(),                    // perm (none)
                std::ptr::null(),                    // a_tmp (none)
                *sorted_ids_ptr as *const _,         // sorted_ids
                *fused_expert_ids_ptr as *const _,   // expert_ids
                *num_tokens_post_ptr as *const _,    // num_tokens_post_padded
                topk_weights_ptr as *const _,        // topk_weights
                block_size,                          // moe_block_size
                topk as i32,                         // top_k
                false,                               // mul_topk_weights (false for w1)
                false,                               // is_ep
                total_sorted as i32,                 // size_m
                w1_n as i32,                         // size_n
                h as i32,                            // size_k
                *self.scratch.d_workspace.device_ptr() as *mut _, // workspace
                q_type_ptr,                          // q_type_ptr
                false,                               // has_bias
                false,                               // has_act_order
                true,                                // is_k_full
                false,                               // has_zp
                num_groups_w1,                       // num_groups
                gs,                                  // group_size
                0,                                   // dev
                std::ptr::null_mut(),                // stream_ptr (null = default)
                -1,                                  // thread_k (auto)
                -1,                                  // thread_n (auto)
                self.config.sms as i32,              // sms
                false,                               // use_atomic
                true,                                // fp32_reduce
                false,                               // is_zp_float
            );
        }

        // DIAG: check w1 output at both padding and real token positions
        if diag_moe && layer_idx == 0 {
            unsafe { cuda_sys::lib().cuCtxSynchronize(); }
            // Check C_tmp (FP32 intermediate) - this is what the kernel writes to with fp32_reduce
            let mut h_ctmp = vec![0.0f32; 8];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_ctmp.as_mut_ptr() as *mut _,
                    fused_c_tmp_ptr, 32);
            }
            eprintln!("[DIAG MoE L0] C_tmp[0..8] (FP32) = {:?}", h_ctmp);
            // Check C_tmp at a real token position (position 4 * w1_n = 4 * 1024)
            let mut h_ctmp_r = vec![0.0f32; 8];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_ctmp_r.as_mut_ptr() as *mut _,
                    fused_c_tmp_ptr + (4 * w1_n * 4) as u64, 32);
            }
            eprintln!("[DIAG MoE L0] C_tmp[pos4, 0..8] (FP32) = {:?}", h_ctmp_r);

            // Check position 0 (padding)
            let mut h_w1 = vec![0u16; 8];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_w1.as_mut_ptr() as *mut _,
                    fused_inter_ptr, 16);
            }
            let w1_f32: Vec<f32> = h_w1.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
            eprintln!("[DIAG MoE L0] fused_inter[sorted_pos=0](w1 out) = {:?}", w1_f32);
            // Check first real token position (sorted_pos=4 from earlier)
            let real_pos = 4usize; // first non-padding position
            let mut h_w1r = vec![0u16; 8];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_w1r.as_mut_ptr() as *mut _,
                    fused_inter_ptr + (real_pos * w1_n * 2) as u64, 16);
            }
            let w1r_f32: Vec<f32> = h_w1r.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
            eprintln!("[DIAG MoE L0] fused_inter[sorted_pos=4](w1 out) = {:?}", w1r_f32);
            // Also check expert buffer (are weights loaded?)
            let mut h_ew = vec![0u32; 4];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_ew.as_mut_ptr() as *mut _,
                    w1_base, 16);
            }
            eprintln!("[DIAG MoE L0] w1_base (expert packed)[0..4] = {:?}", h_ew);
        }

        // 9. Activation (silu_mul or relu2) on fused_inter -> fused_inter2
        let fused_inter2_ptr = *self.d_fused_inter2.as_ref().ok_or("fused_inter2")?.device_ptr();
        if gated {
            let act_t = std::cmp::max(32, ((std::cmp::min(1024, inter) + 31) / 32) * 32) as u32;
            let mut ac0 = fused_inter2_ptr;
            let mut ac1 = fused_inter_ptr;
            let mut ac2 = inter as i32;
            let kernel = if activation == 1 { self.kernels.relu2 } else { self.kernels.silu_mul };
            unsafe {
                launch(kernel,
                    (total_sorted as u32, 1, 1), (act_t, 1, 1), 0, std::ptr::null_mut(), // default stream
                    &mut [
                        &mut ac0 as *mut _ as *mut std::ffi::c_void,
                        &mut ac1 as *mut _ as *mut std::ffi::c_void,
                        &mut ac2 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        } else {
            // Ungated: just apply activation in-place
            let act_t = std::cmp::max(32, ((std::cmp::min(1024, inter) + 31) / 32) * 32) as u32;
            let mut ac0 = fused_inter_ptr; // in-place
            let mut ac1 = fused_inter_ptr;
            let mut ac2 = inter as i32;
            unsafe {
                launch(self.kernels.relu2,
                    (total_sorted as u32, 1, 1), (act_t, 1, 1), 0, std::ptr::null_mut(),
                    &mut [
                        &mut ac0 as *mut _ as *mut std::ffi::c_void,
                        &mut ac1 as *mut _ as *mut std::ffi::c_void,
                        &mut ac2 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // 10. MarlinDefault for w2 (down projection), with topk_weights multiplication
        let fused_output_ptr = *self.d_fused_output.as_ref().ok_or("fused_output")?.device_ptr();
        let num_groups_w2 = (inter / self.config.group_size) as i32;
        let w2_input = if gated { fused_inter2_ptr } else { fused_inter_ptr };

        // Apply scale_factor to topk_weights before w2 if needed
        // MarlinDefault multiplies output by topk_weights when mul_topk_weights=true
        // Zero workspace before fused w2 GEMM
        unsafe {
            cuda_sys::lib().cuMemsetD32Async(ws_ptr, 0, ws_len, std::ptr::null_mut());
        }
        unsafe {
            fused_fn(
                w2_input as *const _,                // A: [total_sorted, K=inter]
                w2_base as *const _,                  // B: [E, K/16, N, pack]
                fused_output_ptr as *mut _,           // C: [total_sorted, N=hidden]
                fused_c_tmp_ptr as *mut _,            // C_tmp
                std::ptr::null(),                     // b_bias (none)
                w2s_base as *const _,                 // scales
                std::ptr::null(),                     // s2 (none)
                std::ptr::null(),                     // zp (none)
                std::ptr::null(),                     // g_idx (none)
                std::ptr::null(),                     // perm (none)
                std::ptr::null(),                     // a_tmp (none)
                *sorted_ids_ptr as *const _,          // sorted_ids
                *fused_expert_ids_ptr as *const _,    // expert_ids
                *num_tokens_post_ptr as *const _,     // num_tokens_post_padded
                topk_weights_ptr as *const _,         // topk_weights
                block_size,                           // moe_block_size
                topk as i32,                          // top_k
                true,                                 // mul_topk_weights (true for w2)
                false,                                // is_ep
                total_sorted as i32,                  // size_m
                h as i32,                             // size_n
                inter as i32,                         // size_k
                *self.scratch.d_workspace.device_ptr() as *mut _, // workspace
                q_type_ptr,                           // q_type_ptr
                false,                                // has_bias
                false,                                // has_act_order
                true,                                 // is_k_full
                false,                                // has_zp
                num_groups_w2,                        // num_groups
                gs,                                   // group_size
                0,                                    // dev
                std::ptr::null_mut(),                 // stream_ptr (null = default)
                -1,                                   // thread_k (auto)
                -1,                                   // thread_n (auto)
                self.config.sms as i32,               // sms
                false,                                // use_atomic
                true,                                 // fp32_reduce
                false,                                // is_zp_float
            );
        }

        // Sync default stream -> self.stream (MarlinDefault uses CUDA default stream)
        // Record event on default stream (null), then wait on our compute stream.
        // This is targeted — does NOT block shared_stream.
        unsafe {
            cuda_sys::lib().cuEventRecord(self.compute_event, std::ptr::null_mut());
            cuda_sys::lib().cuStreamWaitEvent(self.stream, self.compute_event, 0);
        }

        // 11. Zero accumulator then scatter-add fused output back to [M, hidden]
        let moe_accum = *self.scratch.d_moe_accum.device_ptr();
        {
            let zt = std::cmp::max(32, ((std::cmp::min(1024, h) + 31) / 32) * 32) as u32;
            let mut z0 = moe_accum;
            let mut z1 = m as i32;
            let mut z2 = h as i32;
            unsafe {
                launch(self.kernels.moe_zero_accum,
                    (m as u32, 1, 1), (zt, 1, 1), 0, self.stream,
                    &mut [
                        &mut z0 as *mut _ as *mut std::ffi::c_void,
                        &mut z1 as *mut _ as *mut std::ffi::c_void,
                        &mut z2 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }
        if total_sorted > 0 {
            let st = std::cmp::max(32, ((std::cmp::min(1024, h) + 31) / 32) * 32) as u32;
            let mut s0 = moe_accum;
            let mut s1 = fused_output_ptr;
            let mut s2 = *sorted_ids_ptr;
            let mut s3 = h as i32;
            let mut s4 = m as i32;
            let mut s5 = scale_factor;
            unsafe {
                launch(self.kernels.moe_scatter_fused,
                    (total_sorted as u32, 1, 1), (st, 1, 1), 0, self.stream,
                    &mut [
                        &mut s0 as *mut _ as *mut std::ffi::c_void,
                        &mut s1 as *mut _ as *mut std::ffi::c_void,
                        &mut s2 as *mut _ as *mut std::ffi::c_void,
                        &mut s3 as *mut _ as *mut std::ffi::c_void,
                        &mut s4 as *mut _ as *mut std::ffi::c_void,
                        &mut s5 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // DIAG: check accum after scatter (before shared expert)
        if diag_moe && layer_idx == 0 {
            self.stream_sync()?;
            // Read full FP32 accum for ALL tokens
            let mut h_accum_all = vec![0.0f32; m * h];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_accum_all.as_mut_ptr() as *mut _,
                    moe_accum, (m * h * 4) as usize);
            }
            // Per-token norms
            for tok in [0usize, 1, 16, 20] {
                if tok < m {
                    let row = &h_accum_all[tok*h..(tok+1)*h];
                    let norm: f32 = row.iter().map(|v| v*v).sum::<f32>().sqrt();
                    let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    eprintln!("[DIAG MoE L0] accum[tok={}] norm={:.6}, max={:.6}", tok, norm, maxv);
                }
            }
            // Full accum stats
            let full_norm: f32 = h_accum_all.iter().map(|v| v*v).sum::<f32>().sqrt();
            let full_max = h_accum_all.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[DIAG MoE L0] accum full: norm={:.6}, max={:.6}", full_norm, full_max);

            // Also check one slot of fused_output to see if routing weights were applied
            let mut h_fout = vec![0u16; std::cmp::min(8, h)];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_fout.as_mut_ptr() as *mut _,
                    fused_output_ptr, (h_fout.len() * 2) as usize);
            }
            let f_f32: Vec<f32> = h_fout.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
            eprintln!("[DIAG MoE L0] fused_output[0][0..8] = {:?}", f_f32);
        }

        // 12. Wait for shared expert and add to accumulator
        if has_shared {
            unsafe {
                cuda_sys::lib().cuEventRecord(self.shared_event, self.shared_stream);
                cuda_sys::lib().cuStreamWaitEvent(self.stream, self.shared_event, 0);
            }
            let s1_buf = *self.scratch.d_scratch1.device_ptr();
            let ast = std::cmp::max(32, ((std::cmp::min(1024, h) + 31) / 32) * 32) as u32;
            let mut as0 = moe_accum; let mut as1 = s1_buf;
            let mut as2 = m as i32; let mut as3 = h as i32;
            unsafe {
                launch(self.kernels.moe_add_shared,
                    (m as u32, 1, 1), (ast, 1, 1), 0, self.stream,
                    &mut [
                        &mut as0 as *mut _ as *mut std::ffi::c_void,
                        &mut as1 as *mut _ as *mut std::ffi::c_void,
                        &mut as2 as *mut _ as *mut std::ffi::c_void,
                        &mut as3 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // 13. Convert FP32 accum -> BF16 hidden
        {
            let ct = std::cmp::max(32, ((std::cmp::min(1024, h) + 31) / 32) * 32) as u32;
            let mut c0 = hidden; let mut c1 = moe_accum;
            let mut c2 = m as i32; let mut c3 = h as i32;
            unsafe {
                launch(self.kernels.moe_accum_to_bf16,
                    (m as u32, 1, 1), (ct, 1, 1), 0, self.stream,
                    &mut [
                        &mut c0 as *mut _ as *mut std::ffi::c_void,
                        &mut c1 as *mut _ as *mut std::ffi::c_void,
                        &mut c2 as *mut _ as *mut std::ffi::c_void,
                        &mut c3 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        Ok(())
    }

    fn forward_moe_sequential(&mut self, layer_idx: usize, m: usize) -> Result<(), String> {
        let lw = &self.layer_weights[layer_idx];
        let h = self.config.hidden_size;
        let n_experts = lw.moe_num_experts;
        let topk = lw.moe_topk;
        let inter = self.config.moe_intermediate_size;
        let scoring_func = lw.moe_scoring_func;
        let gated = lw.moe_gated;
        let activation = lw.moe_activation;
        let _norm_topk = lw.moe_norm_topk_prob;
        let scale_factor = lw.moe_routed_scaling_factor;

        let hidden = *self.scratch.d_hidden.device_ptr();
        let gate_out = *self.scratch.d_gate_out.device_ptr();
        let moe_accum = *self.scratch.d_moe_accum.device_ptr();

        // 1. Gate GEMM: [m, hidden] @ [hidden, n_experts] -> [m, n_experts] FP32
        let gate_ptr = lw.moe_gate_ptr;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            use cudarc::cublas::sys as cublas_sys;
            use cudarc::cublas::result as cublas_result;

            cublas_result::set_stream(
                self.cublas_handle,
                self.stream as cublas_sys::cudaStream_t,
            ).map_err(|e| format!("cublas set_stream: {:?}", e))?;

            cublas_result::gemm_ex(
                self.cublas_handle,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n_experts as i32, m as i32, h as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                gate_ptr as *const std::ffi::c_void,
                cublas_sys::cudaDataType::CUDA_R_16BF, h as i32,
                hidden as *const std::ffi::c_void,
                cublas_sys::cudaDataType::CUDA_R_16BF, h as i32,
                &beta as *const f32 as *const std::ffi::c_void,
                gate_out as *mut std::ffi::c_void,
                cublas_sys::cudaDataType::CUDA_R_32F, n_experts as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            ).map_err(|e| format!("cublas gate GEMM: {:?}", e))?;
        }

        // 2. Top-K routing (GPU-only)
        let topk_ids_ptr = *self.scratch.d_topk_ids.device_ptr();
        let topk_weights_ptr = *self.scratch.d_topk_weights.device_ptr();
        {
            let t = std::cmp::max(32, ((std::cmp::min(1024, n_experts) + 31) / 32) * 32) as u32;
            let mut p0 = topk_weights_ptr;
            let mut p1 = topk_ids_ptr;
            let mut p2 = gate_out;
            let mut p3 = n_experts as i32;
            let mut p4 = topk as i32;
            let kernel = if scoring_func == 1 { self.kernels.sigmoid_topk } else { self.kernels.softmax_topk };
            let smem = (n_experts * 4 + topk * 8 + if scoring_func == 0 { 32 * 4 } else { 0 }) as u32;
            unsafe {
                launch(kernel,
                    (m as u32, 1, 1), (t, 1, 1), smem, self.stream,
                    &mut [
                        &mut p0 as *mut _ as *mut std::ffi::c_void,
                        &mut p1 as *mut _ as *mut std::ffi::c_void,
                        &mut p2 as *mut _ as *mut std::ffi::c_void,
                        &mut p3 as *mut _ as *mut std::ffi::c_void,
                        &mut p4 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // 3. GPU-only routing: count experts, prefix sum, build maps
        //    This replaces the CPU round-trip (stream_sync + D2H + CPU binning + H2D)
        let expert_counts_ptr = *self.scratch.d_expert_counts.device_ptr();
        let expert_offsets_ptr = *self.scratch.d_expert_offsets.device_ptr();
        let write_offsets_ptr = *self.scratch.d_write_offsets.device_ptr();
        let gather_src_ptr = *self.scratch.d_gather_src_map.device_ptr();
        let gather_wt_ptr = *self.scratch.d_gather_weight_map.device_ptr();

        // Zero expert counts and write offsets
        unsafe {
            cuda_sys::lib().cuMemsetD8Async(expert_counts_ptr, 0, (n_experts * 4) as usize, self.stream);
            cuda_sys::lib().cuMemsetD8Async(write_offsets_ptr, 0, (n_experts * 4) as usize, self.stream);
        }

        // Phase 1: count tokens per expert
        {
            let mut p0 = expert_counts_ptr;
            let mut p1 = topk_ids_ptr;
            let mut p2 = m as i32;
            let mut p3 = topk as i32;
            let mut p4 = n_experts as i32;
            unsafe {
                launch(self.kernels.moe_count_experts,
                    (m as u32, 1, 1), (1, 1, 1), 0, self.stream,
                    &mut [
                        &mut p0 as *mut _ as *mut std::ffi::c_void,
                        &mut p1 as *mut _ as *mut std::ffi::c_void,
                        &mut p2 as *mut _ as *mut std::ffi::c_void,
                        &mut p3 as *mut _ as *mut std::ffi::c_void,
                        &mut p4 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Phase 2: prefix sum on expert counts
        {
            let ps_threads = std::cmp::max(32, ((std::cmp::min(1024, n_experts) + 31) / 32) * 32) as u32;
            let mut p0 = expert_offsets_ptr;
            let mut p1 = expert_counts_ptr;
            let mut p2 = n_experts as i32;
            unsafe {
                launch(self.kernels.moe_prefix_sum,
                    (1, 1, 1), (ps_threads, 1, 1), (n_experts * 4) as u32, self.stream,
                    &mut [
                        &mut p0 as *mut _ as *mut std::ffi::c_void,
                        &mut p1 as *mut _ as *mut std::ffi::c_void,
                        &mut p2 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Phase 3: build gather/scatter maps (applies scale_factor to weights)
        {
            let mut p0 = gather_src_ptr;
            let mut p1 = gather_wt_ptr;
            let mut p2 = write_offsets_ptr;
            let mut p3 = topk_ids_ptr;
            let mut p4 = topk_weights_ptr;
            let mut p5 = expert_offsets_ptr;
            let mut p6 = m as i32;
            let mut p7 = topk as i32;
            let mut p8 = n_experts as i32;
            let mut p9 = scale_factor;
            unsafe {
                launch(self.kernels.moe_build_maps,
                    (m as u32, 1, 1), (1, 1, 1), 0, self.stream,
                    &mut [
                        &mut p0 as *mut _ as *mut std::ffi::c_void,
                        &mut p1 as *mut _ as *mut std::ffi::c_void,
                        &mut p2 as *mut _ as *mut std::ffi::c_void,
                        &mut p3 as *mut _ as *mut std::ffi::c_void,
                        &mut p4 as *mut _ as *mut std::ffi::c_void,
                        &mut p5 as *mut _ as *mut std::ffi::c_void,
                        &mut p6 as *mut _ as *mut std::ffi::c_void,
                        &mut p7 as *mut _ as *mut std::ffi::c_void,
                        &mut p8 as *mut _ as *mut std::ffi::c_void,
                        &mut p9 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // 4. Download expert offsets to CPU (small: E+1 ints, no need to avoid this sync)
        //    We need offsets to know how many tokens each expert got for per-expert GEMM dispatch.
        let total_active = m * topk; // upper bound
        self.h_topk_ids.resize(n_experts + 1, 0);
        self.stream_sync()?;
        unsafe {
            cuda_sys::lib().cuMemcpyDtoH_v2(
                self.h_topk_ids.as_mut_ptr() as *mut _,
                expert_offsets_ptr,
                ((n_experts + 1) * 4) as usize,
            );
        }
        let expert_offsets: Vec<usize> = self.h_topk_ids[..n_experts + 1].iter()
            .map(|&x| x as usize).collect();
        let total_active = expert_offsets[n_experts];

        // Diagnostic: dump routing weights, expert ids, and active counts for layer 0
        let diag = std::env::var("KRASIS_PREFILL_DIAG").is_ok();
        let diag_layer_limit = Self::diag_layer_limit();
        if diag && layer_idx < diag_layer_limit {
            // Download top-k weights and ids for token 0
            let mut h_weights = vec![0.0f32; m * topk];
            let mut h_ids = vec![0i32; m * topk];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_weights.as_mut_ptr() as *mut _, topk_weights_ptr,
                    (m * topk * 4) as usize,
                );
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_ids.as_mut_ptr() as *mut _, topk_ids_ptr,
                    (m * topk * 4) as usize,
                );
            }
            // Token 0 routing
            let w0: Vec<f32> = h_weights[..topk].to_vec();
            let ids0: Vec<i32> = h_ids[..topk].to_vec();
            let wsum: f32 = w0.iter().sum();
            eprintln!("[DIAG] layer{:02}_moe scoring_func={} norm_topk={} scale_factor={:.4} topk={} n_experts={}",
                layer_idx, scoring_func, _norm_topk, scale_factor, topk, n_experts);
            eprintln!("[DIAG] layer{:02}_moe tok0_experts={:?} tok0_weights={:.4?} tok0_wsum={:.6}",
                layer_idx, ids0, w0, wsum);
            eprintln!("[DIAG] layer{:02}_moe total_active={} active_experts={}",
                layer_idx, total_active, expert_offsets.iter().enumerate()
                    .filter(|(i, _)| *i < n_experts && expert_offsets[*i + 1] > expert_offsets[*i])
                    .count());
        }

        // 5. Gather tokens: hidden [M, hidden] -> gathered [total_active, hidden]
        let gathered = *self.scratch.d_moe_gathered.device_ptr();
        if total_active > 0 {
            let gt = std::cmp::max(32, ((std::cmp::min(1024, h) + 31) / 32) * 32) as u32;
            let mut g0 = gathered;
            let mut g1 = hidden;
            let mut g2 = gather_src_ptr;
            let mut g3 = total_active as i32;
            let mut g4 = h as i32;
            unsafe {
                launch(self.kernels.moe_gather,
                    (total_active as u32, 1, 1), (gt, 1, 1), 0, self.stream,
                    &mut [
                        &mut g0 as *mut _ as *mut std::ffi::c_void,
                        &mut g1 as *mut _ as *mut std::ffi::c_void,
                        &mut g2 as *mut _ as *mut std::ffi::c_void,
                        &mut g3 as *mut _ as *mut std::ffi::c_void,
                        &mut g4 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // 6. Zero MoE accumulator
        {
            let zt = std::cmp::max(32, ((std::cmp::min(1024, h) + 31) / 32) * 32) as u32;
            let mut z0 = moe_accum;
            let mut z1 = m as i32;
            let mut z2 = h as i32;
            unsafe {
                launch(self.kernels.moe_zero_accum,
                    (m as u32, 1, 1), (zt, 1, 1), 0, self.stream,
                    &mut [
                        &mut z0 as *mut _ as *mut std::ffi::c_void,
                        &mut z1 as *mut _ as *mut std::ffi::c_void,
                        &mut z2 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Diagnostic: check gathered input norms before expert GEMMs
        if diag && layer_idx < diag_layer_limit && total_active > 0 {
            let diag_positions = Self::diag_positions(m);
            let mut parts = Vec::new();
            for &pos in &diag_positions {
                let n = self.diag_l2_norm(gathered, pos, h, h);
                parts.push(format!("{}:{:.4}", pos, n));
            }
            eprintln!("[DIAG] layer{:02}_moe gathered_input_norms (first {} rows) {}",
                layer_idx, std::cmp::min(total_active, diag_positions.len()), parts.join(" "));
            // Also show d_hidden input norms (what the MoE receives after post-attn-norm)
            let mut h_parts = Vec::new();
            for &pos in &diag_positions {
                let n = self.diag_l2_norm(hidden, pos, h, h);
                h_parts.push(format!("{}:{:.4}", pos, n));
            }
            eprintln!("[DIAG] layer{:02}_moe d_hidden_input_norms {}", layer_idx, h_parts.join(" "));
        }

        // 7-8. Shared expert + routed experts
        let has_shared = lw.shared_w1.is_some() && lw.shared_w2.is_some();

        // Per-expert Marlin GEMM with HCS-aware dispatch and double-buffered DMA
        let moe_layer_idx = lw.moe_layer_idx;
        let expert_out = *self.scratch.d_moe_expert_out.device_ptr();
        let gate_up_buf = *self.scratch.d_moe_gate_up.device_ptr();
        let inter_buf = *self.scratch.d_moe_inter.device_ptr();

        let bufs: [[(u64, u64, u64, u64); 1]; 2] = [
            [(*self.scratch.d_expert_w13_packed_a.device_ptr(),
              *self.scratch.d_expert_w13_scales_a.device_ptr(),
              *self.scratch.d_expert_w2_packed_a.device_ptr(),
              *self.scratch.d_expert_w2_scales_a.device_ptr())],
            [(*self.scratch.d_expert_w13_packed_b.device_ptr(),
              *self.scratch.d_expert_w13_scales_b.device_ptr(),
              *self.scratch.d_expert_w2_packed_b.device_ptr(),
              *self.scratch.d_expert_w2_scales_b.device_ptr())],
        ];

        let w13_n = if gated { 2 * inter } else { inter };
        let bits = self.config.expert_bits;
        let gs = self.config.group_size;
        let num_groups_w13 = h / gs;
        let num_groups_w2 = inter / gs;

        // Collect active experts, check HCS residency
        struct ExpertWork {
            eid: usize,
            count: usize,
            offset: usize,
            w13_packed: u64,
            w13_scales: u64,
            w2_packed: u64,
            w2_scales: u64,
            is_hcs: bool,
        }
        let mut work: Vec<ExpertWork> = Vec::new();

        for eid in 0..n_experts {
            let count = if eid + 1 <= n_experts {
                expert_offsets[eid + 1] - expert_offsets[eid]
            } else { 0 };
            if count == 0 { continue; }
            let eo = expert_offsets[eid];

            if let Some(moe_idx) = moe_layer_idx {
                if let Some((w13p, w13s, w2p, w2s)) = self.hcs_lookup(moe_idx, eid) {
                    work.push(ExpertWork {
                        eid, count, offset: eo,
                        w13_packed: w13p, w13_scales: w13s,
                        w2_packed: w2p, w2_scales: w2s,
                        is_hcs: true,
                    });
                    continue;
                }
            }

            work.push(ExpertWork {
                eid, count, offset: eo,
                w13_packed: 0, w13_scales: 0,
                w2_packed: 0, w2_scales: 0,
                is_hcs: false,
            });
        }

        // Launch shared expert async if all experts are HCS-resident (no DMA conflict)
        // Force sync when diagnostics are active so we can add intermediate norm checks
        let all_hcs = work.iter().all(|w| w.is_hcs);
        let shared_async = has_shared && all_hcs && !(diag && layer_idx < diag_layer_limit);
        if shared_async {
            unsafe {
                cuda_sys::lib().cuEventRecord(self.compute_event, self.stream);
                cuda_sys::lib().cuStreamWaitEvent(self.copy_stream, self.compute_event, 0);
            }
            self.launch_shared_expert_async(layer_idx, m)?;
        }

        // Process HCS experts first (no DMA)
        for w in &work {
            if !w.is_hcs { continue; }
            let gathered_slice = gathered + (w.offset * h * 2) as u64;
            self.run_expert_gemm(
                gathered_slice, expert_out, gate_up_buf, inter_buf,
                w.w13_packed, w.w13_scales, w.w2_packed, w.w2_scales,
                w.count, w.offset, w13_n, h, inter, gs, bits,
                num_groups_w13, num_groups_w2, gated, activation,
            )?;
        }

        // Cold experts: double-buffered DMA pipeline
        // TEMP: skip cold experts to test HCS-only path
        let cold_work: Vec<&ExpertWork> = if std::env::var("KRASIS_SKIP_COLD").is_ok() {
            Vec::new()
        } else {
            work.iter().filter(|w| !w.is_hcs).collect()
        };

        if !cold_work.is_empty() {
            if let Some(moe_idx) = moe_layer_idx {
                if let Some(Some(moe_data)) = self.moe_layers.get(moe_idx) {
                    let mut cur_buf = 0usize;

                    let first_eid = cold_work[0].eid;
                    if first_eid < moe_data.experts.len() {
                        self.dma_expert_to_buf(&moe_data.experts[first_eid], &bufs[0][0])?;
                    }

                    for ci in 0..cold_work.len() {
                        let cw = cold_work[ci];
                        let (w13p, w13s, w2p, w2s) = bufs[cur_buf][0];

                        unsafe {
                            let err = cuda_sys::lib().cuStreamWaitEvent(
                                self.stream, self.dma_event, 0);
                            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                                return Err(format!("Stream wait DMA event: {:?}", err));
                            }
                        }

                        let next_buf = 1 - cur_buf;
                        if ci + 1 < cold_work.len() {
                            let next_eid = cold_work[ci + 1].eid;
                            if next_eid < moe_data.experts.len() {
                                unsafe {
                                    cuda_sys::lib().cuEventRecord(
                                        self.compute_event, self.stream);
                                    cuda_sys::lib().cuStreamWaitEvent(
                                        self.copy_stream, self.compute_event, 0);
                                }
                                self.dma_expert_to_buf(&moe_data.experts[next_eid], &bufs[next_buf][0])?;
                            }
                        }

                        let gathered_slice = gathered + (cw.offset * h * 2) as u64;
                        self.run_expert_gemm(
                            gathered_slice, expert_out, gate_up_buf, inter_buf,
                            w13p, w13s, w2p, w2s,
                            cw.count, cw.offset, w13_n, h, inter, gs, bits,
                            num_groups_w13, num_groups_w2, gated, activation,
                        )?;

                        cur_buf = next_buf;
                    }
                }
            }
        }

        // Diagnostic: dump per-row expert_out norms before scatter to find outliers
        if diag && layer_idx < diag_layer_limit && total_active > 0 {
            self.stream_sync()?;
            // Download expert_out [total_active, h] BF16 and gather_src_map [total_active] i32
            let mut h_expert_out = vec![0u16; total_active * h];
            let mut h_src_map = vec![0i32; total_active];
            let mut h_wt_map = vec![0.0f32; total_active];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_expert_out.as_mut_ptr() as *mut _, expert_out,
                    (total_active * h * 2) as usize);
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_src_map.as_mut_ptr() as *mut _, gather_src_ptr,
                    (total_active * 4) as usize);
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_wt_map.as_mut_ptr() as *mut _, gather_wt_ptr,
                    (total_active * 4) as usize);
            }
            // Compute per-row norms and map row -> expert
            let mut token_max_norm: std::collections::HashMap<i32, (f32, usize, usize)> = std::collections::HashMap::new();
            let mut outlier_rows = Vec::new();
            for row in 0..total_active {
                let mut sumsq = 0.0f64;
                for j in 0..h {
                    let bits = h_expert_out[row * h + j];
                    let val = f32::from_bits((bits as u32) << 16) as f64;
                    sumsq += val * val;
                }
                let norm = (sumsq as f32).sqrt();
                let tok = h_src_map[row];
                let wt = h_wt_map[row];
                // Map row to expert via expert_offsets
                let eid = expert_offsets.windows(2).enumerate()
                    .find(|(_, w)| row >= w[0] && row < w[1])
                    .map(|(i, _)| i).unwrap_or(9999);
                let entry = token_max_norm.entry(tok).or_insert((0.0, row, eid));
                if norm > entry.0 { *entry = (norm, row, eid); }
                if norm > 5.0 {
                    outlier_rows.push((row, tok, norm, wt, eid));
                }
            }
            if !outlier_rows.is_empty() {
                eprintln!("[DIAG] layer{:02}_moe OUTLIER expert_out rows (norm>5): {} of {} rows",
                    layer_idx, outlier_rows.len(), total_active);
                for &(row, tok, norm, wt, eid) in outlier_rows.iter().take(30) {
                    let is_hcs = work.iter().find(|w| w.eid == eid).map(|w| w.is_hcs).unwrap_or(false);
                    eprintln!("[DIAG]   row={} tok={} expert={} norm={:.4} wt={:.6} hcs={}",
                        row, tok, eid, norm, wt, is_hcs);
                }
            }
            // Also check gathered input norm for outlier rows
            let mut h_gathered = vec![0u16; total_active * h];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_gathered.as_mut_ptr() as *mut _, gathered,
                    (total_active * h * 2) as usize);
            }
            if !outlier_rows.is_empty() {
                eprintln!("[DIAG] layer{:02}_moe outlier input vs output:", layer_idx);
                for &(row, tok, out_norm, _wt, eid) in outlier_rows.iter().take(10) {
                    let mut in_sumsq = 0.0f64;
                    for j in 0..h {
                        let bits = h_gathered[row * h + j];
                        let val = f32::from_bits((bits as u32) << 16) as f64;
                        in_sumsq += val * val;
                    }
                    let in_norm = (in_sumsq as f32).sqrt();
                    eprintln!("[DIAG]   row={} tok={} expert={} in_norm={:.4} out_norm={:.4} ratio={:.2}x",
                        row, tok, eid, in_norm, out_norm, out_norm / in_norm);
                }
            }
            // Show per-token max norms for diagnostic positions
            let diag_positions = Self::diag_positions(m);
            let mut parts = Vec::new();
            for &pos in &diag_positions {
                if let Some(&(max_norm, _max_row, _max_eid)) = token_max_norm.get(&(pos as i32)) {
                    parts.push(format!("{}:{:.4}", pos, max_norm));
                } else {
                    parts.push(format!("{}:N/A", pos));
                }
            }
            eprintln!("[DIAG] layer{:02}_moe per_tok_max_expert_norm {}", layer_idx, parts.join(" "));
        }

        // 9. Scatter + accumulate: expert_out -> moe_accum with weights
        if total_active > 0 {
            let st = std::cmp::max(32, ((std::cmp::min(1024, h) + 31) / 32) * 32) as u32;
            let mut s0 = moe_accum;
            let mut s1 = expert_out;
            let mut s2 = gather_src_ptr;
            let mut s3 = gather_wt_ptr;
            let mut s4 = total_active as i32;
            let mut s5 = h as i32;
            unsafe {
                launch(self.kernels.moe_scatter_add,
                    (total_active as u32, 1, 1), (st, 1, 1), 0, self.stream,
                    &mut [
                        &mut s0 as *mut _ as *mut std::ffi::c_void,
                        &mut s1 as *mut _ as *mut std::ffi::c_void,
                        &mut s2 as *mut _ as *mut std::ffi::c_void,
                        &mut s3 as *mut _ as *mut std::ffi::c_void,
                        &mut s4 as *mut _ as *mut std::ffi::c_void,
                        &mut s5 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Diagnostic: routed accum norms at all diagnostic positions
        if diag && layer_idx < diag_layer_limit {
            let diag_positions = Self::diag_positions(m);
            let mut routed_parts = Vec::new();
            for &pos in &diag_positions {
                let n = self.diag_l2_norm_f32(moe_accum, pos, h, h);
                routed_parts.push(format!("{}:{:.6}", pos, n));
            }
            eprintln!("[DIAG] layer{:02}_moe routed_accum {}", layer_idx, routed_parts.join(" "));
        }

        // 10. Shared expert: add to accumulator
        if has_shared {
            if shared_async {
                // Wait for async shared expert to finish on shared_stream
                unsafe {
                    cuda_sys::lib().cuEventRecord(self.shared_event, self.shared_stream);
                    cuda_sys::lib().cuStreamWaitEvent(self.stream, self.shared_event, 0);
                }
            } else {
                // Run shared expert synchronously on main stream
                let sw1 = lw.shared_w1.as_ref().ok_or("missing shared w1")?;
                let sw2 = lw.shared_w2.as_ref().ok_or("missing shared w2")?;
                let s1_buf = *self.scratch.d_scratch1.device_ptr();
                let s2_buf = *self.scratch.d_scratch2.device_ptr();
                let shared_inter = sw1.n / (if gated { 2 } else { 1 });

                if diag && layer_idx < diag_layer_limit {
                    eprintln!("[DIAG] layer{:02}_moe shared_expert sw1: n={} k={} bits={} gs={} groups={}",
                        layer_idx, sw1.n, sw1.k, sw1.num_bits, sw1.group_size, sw1.num_groups);
                    eprintln!("[DIAG] layer{:02}_moe shared_expert sw2: n={} k={} bits={} gs={} groups={}",
                        layer_idx, sw2.n, sw2.k, sw2.num_bits, sw2.group_size, sw2.num_groups);
                    eprintln!("[DIAG] layer{:02}_moe shared_inter={} gated={} activation={}",
                        layer_idx, shared_inter, gated, activation);
                }

                self.marlin_gemm(hidden, sw1, s1_buf, m)?;

                if gated {
                    if diag && layer_idx < diag_layer_limit {
                        // s1_buf has w1 output: [m, w13_n] BF16
                        let w1_norm = self.diag_l2_norm(s1_buf, 0, sw1.n, sw1.n);
                        eprintln!("[DIAG] layer{:02}_moe shared_w1_out_norm pos0={:.6} (dim={})", layer_idx, w1_norm, sw1.n);
                    }

                    let act_t = std::cmp::max(32, ((std::cmp::min(1024, shared_inter) + 31) / 32) * 32);
                    let mut ac0 = s2_buf; let mut ac1 = s1_buf; let mut ac2 = shared_inter as i32;
                    let kernel = if activation == 1 { self.kernels.relu2 } else { self.kernels.silu_mul };
                    unsafe {
                        launch(kernel,
                            (m as u32, 1, 1), (act_t as u32, 1, 1), 0, self.stream,
                            &mut [
                                &mut ac0 as *mut _ as *mut std::ffi::c_void,
                                &mut ac1 as *mut _ as *mut std::ffi::c_void,
                                &mut ac2 as *mut _ as *mut std::ffi::c_void,
                            ],
                        )?;
                    }

                    if diag && layer_idx < diag_layer_limit {
                        // s2_buf has activation output: [m, shared_inter] BF16
                        let act_norm = self.diag_l2_norm(s2_buf, 0, shared_inter, shared_inter);
                        eprintln!("[DIAG] layer{:02}_moe shared_act_out_norm pos0={:.6} (dim={})", layer_idx, act_norm, shared_inter);
                    }

                    self.marlin_gemm(s2_buf, sw2, s1_buf, m)?;
                } else {
                    let act_t = std::cmp::max(32, ((std::cmp::min(1024, shared_inter) + 31) / 32) * 32);
                    let mut ac0 = s1_buf; let mut ac1 = s1_buf; let mut ac2 = shared_inter as i32;
                    unsafe {
                        launch(self.kernels.relu2,
                            (m as u32, 1, 1), (act_t as u32, 1, 1), 0, self.stream,
                            &mut [
                                &mut ac0 as *mut _ as *mut std::ffi::c_void,
                                &mut ac1 as *mut _ as *mut std::ffi::c_void,
                                &mut ac2 as *mut _ as *mut std::ffi::c_void,
                            ],
                        )?;
                    }
                    self.marlin_gemm(s1_buf, sw2, s2_buf, m)?;
                    self.memcpy_d2d(s1_buf, s2_buf, (m * h * 2) as u64)?;
                }
            }

            // Add shared expert result (in scratch1) to accumulator, with optional sigmoid gate
            let s1_buf = *self.scratch.d_scratch1.device_ptr();
            let sg_ptr = lw.shared_gate_ptr;
            if sg_ptr != 0 {
                // Compute shared gate: hidden [m, h] @ gate [h, 1] -> gate_out [m, 1] FP32
                let gate_out = *self.scratch.d_gate_out.device_ptr(); // reuse gate_out buffer (has room for m * n_experts, m*1 is fine)
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;
                unsafe {
                    use cudarc::cublas::sys as cublas_sys;
                    use cudarc::cublas::result as cublas_result;
                    cublas_result::set_stream(
                        self.cublas_handle,
                        self.stream as cublas_sys::cudaStream_t,
                    ).map_err(|e| format!("cublas set_stream: {:?}", e))?;
                    cublas_result::gemm_ex(
                        self.cublas_handle,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                        cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                        lw.shared_gate_rows as i32, m as i32, lw.shared_gate_cols as i32,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        sg_ptr as *const std::ffi::c_void,
                        cublas_sys::cudaDataType::CUDA_R_16BF, lw.shared_gate_cols as i32,
                        hidden as *const std::ffi::c_void,
                        cublas_sys::cudaDataType::CUDA_R_16BF, h as i32,
                        &beta as *const f32 as *const std::ffi::c_void,
                        gate_out as *mut std::ffi::c_void,
                        cublas_sys::cudaDataType::CUDA_R_32F, lw.shared_gate_rows as i32,
                        cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    ).map_err(|e| format!("shared gate GEMM: {:?}", e))?;
                }
                // Add shared expert with sigmoid gating
                let ast = std::cmp::max(32, ((std::cmp::min(1024, h) + 31) / 32) * 32) as u32;
                let mut as0 = moe_accum; let mut as1 = s1_buf;
                let mut as2 = gate_out; let mut as3 = m as i32; let mut as4 = h as i32;
                unsafe {
                    launch(self.kernels.moe_add_shared_gated,
                        (m as u32, 1, 1), (ast, 1, 1), 0, self.stream,
                        &mut [
                            &mut as0 as *mut _ as *mut std::ffi::c_void,
                            &mut as1 as *mut _ as *mut std::ffi::c_void,
                            &mut as2 as *mut _ as *mut std::ffi::c_void,
                            &mut as3 as *mut _ as *mut std::ffi::c_void,
                            &mut as4 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            } else {
                // No gate, add directly
                let ast = std::cmp::max(32, ((std::cmp::min(1024, h) + 31) / 32) * 32) as u32;
                let mut as0 = moe_accum; let mut as1 = s1_buf;
                let mut as2 = m as i32; let mut as3 = h as i32;
                unsafe {
                    launch(self.kernels.moe_add_shared,
                        (m as u32, 1, 1), (ast, 1, 1), 0, self.stream,
                        &mut [
                            &mut as0 as *mut _ as *mut std::ffi::c_void,
                            &mut as1 as *mut _ as *mut std::ffi::c_void,
                            &mut as2 as *mut _ as *mut std::ffi::c_void,
                            &mut as3 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }
        }

        // Diagnostic: total accum norms + shared expert norms + gate values at all positions
        if diag && layer_idx < diag_layer_limit {
            let diag_positions = Self::diag_positions(m);
            // Total accum (routed + shared)
            let mut total_parts = Vec::new();
            for &pos in &diag_positions {
                let n = self.diag_l2_norm_f32(moe_accum, pos, h, h);
                total_parts.push(format!("{}:{:.6}", pos, n));
            }
            eprintln!("[DIAG] layer{:02}_moe total_accum {}", layer_idx, total_parts.join(" "));
            // Shared expert output norms
            let s1_ptr = *self.scratch.d_scratch1.device_ptr();
            let mut shared_parts = Vec::new();
            for &pos in &diag_positions {
                let n = self.diag_l2_norm(s1_ptr, pos, h, h);
                shared_parts.push(format!("{}:{:.6}", pos, n));
            }
            eprintln!("[DIAG] layer{:02}_moe shared_out {}", layer_idx, shared_parts.join(" "));
            // Gate values (if gated)
            let sg_ptr = lw.shared_gate_ptr;
            if sg_ptr != 0 {
                let gate_out_ptr = *self.scratch.d_gate_out.device_ptr();
                let gate_vals = self.diag_download_f32(gate_out_ptr, m);
                let mut gate_parts = Vec::new();
                for &pos in &diag_positions {
                    if pos < gate_vals.len() {
                        let raw = gate_vals[pos];
                        let sig = 1.0 / (1.0 + (-raw).exp());
                        gate_parts.push(format!("{}:raw={:.4},sig={:.4}", pos, raw, sig));
                    }
                }
                eprintln!("[DIAG] layer{:02}_moe gate_values {}", layer_idx, gate_parts.join(" "));
            }
        }

        // 11. Convert FP32 accum -> BF16 hidden
        {
            let ct = std::cmp::max(32, ((std::cmp::min(1024, h) + 31) / 32) * 32) as u32;
            let mut c0 = hidden; let mut c1 = moe_accum;
            let mut c2 = m as i32; let mut c3 = h as i32;
            unsafe {
                launch(self.kernels.moe_accum_to_bf16,
                    (m as u32, 1, 1), (ct, 1, 1), 0, self.stream,
                    &mut [
                        &mut c0 as *mut _ as *mut std::ffi::c_void,
                        &mut c1 as *mut _ as *mut std::ffi::c_void,
                        &mut c2 as *mut _ as *mut std::ffi::c_void,
                        &mut c3 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        Ok(())
    }

    /// Launch shared expert computation on copy_stream (async overlap with routed experts).
    /// Result is written to d_scratch1.
    fn launch_shared_expert_async(&self, layer_idx: usize, m: usize) -> Result<(), String> {
        // Use shared_stream + d_shared_fp32_scratch + d_shared_workspace
        // to avoid data races with the main stream's routed expert dispatch
        // which uses d_fp32_scratch + d_workspace + copy_stream.
        let lw = &self.layer_weights[layer_idx];
        let h = self.config.hidden_size;
        let gated = lw.moe_gated;
        let activation = lw.moe_activation;
        let hidden = *self.scratch.d_hidden.device_ptr();
        let s1_buf = *self.scratch.d_scratch1.device_ptr();
        let s2_buf = *self.scratch.d_scratch2.device_ptr();

        let sw1 = lw.shared_w1.as_ref().ok_or("missing shared w1")?;
        let sw2 = lw.shared_w2.as_ref().ok_or("missing shared w2")?;
        let shared_inter = sw1.n / (if gated { 2 } else { 1 });

        let st = if sw1.num_bits == 4 { &ScalarType::U4B8 } else { &ScalarType::U8B128 };
        let f = self.kernels.marlin_mm.ok_or("Marlin GEMM not loaded")?;

        let shared_scratch = *self.d_shared_fp32_scratch.device_ptr();
        let shared_ws = *self.d_shared_workspace.device_ptr();
        let shared_ws_len = self.config.sms * 4;

        // Zero workspace + C_tmp before w1 GEMM
        unsafe {
            cuda_sys::lib().cuMemsetD32Async(shared_ws, 0, shared_ws_len, self.shared_stream);
            cuda_sys::lib().cuMemsetD32Async(shared_scratch, 0, m * sw1.n, self.shared_stream);
        }

        // w1 GEMM on shared_stream
        unsafe {
            f(
                hidden as *const _, sw1.packed as *const _,
                s1_buf as *mut _, shared_scratch as *mut _,
                sw1.scales as *const _, std::ptr::null(), std::ptr::null(),
                std::ptr::null(), std::ptr::null(), std::ptr::null_mut(),
                m as i32, sw1.n as i32, sw1.k as i32, sw1.k as i32,
                shared_ws as *mut _,
                st, false, true, false,
                sw1.num_groups as i32, sw1.group_size as i32,
                self.config.device_ordinal as i32,
                self.shared_stream as u64,
                -1, -1, self.config.sms as i32,
                false, true, false,
            );
        }

        // Activation on shared_stream
        if gated {
            let act_t = std::cmp::max(32, ((std::cmp::min(1024, shared_inter) + 31) / 32) * 32);
            let mut ac0 = s2_buf; let mut ac1 = s1_buf; let mut ac2 = shared_inter as i32;
            let kernel = if activation == 1 { self.kernels.relu2 } else { self.kernels.silu_mul };
            unsafe {
                launch(kernel,
                    (m as u32, 1, 1), (act_t as u32, 1, 1), 0, self.shared_stream,
                    &mut [
                        &mut ac0 as *mut _ as *mut std::ffi::c_void,
                        &mut ac1 as *mut _ as *mut std::ffi::c_void,
                        &mut ac2 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
            // Zero workspace + C_tmp before w2 GEMM
            unsafe {
                cuda_sys::lib().cuMemsetD32Async(shared_ws, 0, shared_ws_len, self.shared_stream);
                cuda_sys::lib().cuMemsetD32Async(shared_scratch, 0, m * sw2.n, self.shared_stream);
            }
            // w2 GEMM on shared_stream
            unsafe {
                f(
                    s2_buf as *const _, sw2.packed as *const _,
                    s1_buf as *mut _, shared_scratch as *mut _,
                    sw2.scales as *const _, std::ptr::null(), std::ptr::null(),
                    std::ptr::null(), std::ptr::null(), std::ptr::null_mut(),
                    m as i32, sw2.n as i32, sw2.k as i32, sw2.k as i32,
                    shared_ws as *mut _,
                    st, false, true, false,
                    sw2.num_groups as i32, sw2.group_size as i32,
                    self.config.device_ordinal as i32,
                    self.shared_stream as u64,
                    -1, -1, self.config.sms as i32,
                    false, true, false,
                );
            }
        } else {
            let act_t = std::cmp::max(32, ((std::cmp::min(1024, shared_inter) + 31) / 32) * 32);
            let mut ac0 = s1_buf; let mut ac1 = s1_buf; let mut ac2 = shared_inter as i32;
            unsafe {
                launch(self.kernels.relu2,
                    (m as u32, 1, 1), (act_t as u32, 1, 1), 0, self.shared_stream,
                    &mut [
                        &mut ac0 as *mut _ as *mut std::ffi::c_void,
                        &mut ac1 as *mut _ as *mut std::ffi::c_void,
                        &mut ac2 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
            // Zero workspace + C_tmp before w2 GEMM
            unsafe {
                cuda_sys::lib().cuMemsetD32Async(shared_ws, 0, shared_ws_len, self.shared_stream);
                cuda_sys::lib().cuMemsetD32Async(shared_scratch, 0, m * sw2.n, self.shared_stream);
            }
            unsafe {
                f(
                    s1_buf as *const _, sw2.packed as *const _,
                    s2_buf as *mut _, shared_scratch as *mut _,
                    sw2.scales as *const _, std::ptr::null(), std::ptr::null(),
                    std::ptr::null(), std::ptr::null(), std::ptr::null_mut(),
                    m as i32, sw2.n as i32, sw2.k as i32, sw2.k as i32,
                    shared_ws as *mut _,
                    st, false, true, false,
                    sw2.num_groups as i32, sw2.group_size as i32,
                    self.config.device_ordinal as i32,
                    self.shared_stream as u64,
                    -1, -1, self.config.sms as i32,
                    false, true, false,
                );
            }
            // Copy result from s2 to s1 on shared_stream
            unsafe {
                cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                    s1_buf, s2_buf, (m * h * 2) as usize, self.shared_stream);
            }
        }

        Ok(())
    }

    /// Preload next MoE layer's expert weights into the other buffer (gap 2).
    /// Called after forward_moe completes for the current layer.
    fn preload_next_moe_layer(&mut self, current_layer: usize, num_layers: usize) -> Result<(), String> {
        // Find the next MoE layer
        let mut next_moe_layer = None;
        for i in (current_layer + 1)..num_layers {
            if self.layer_weights[i].moe_gate_ptr != 0 {
                next_moe_layer = Some(i);
                break;
            }
        }

        let next_layer = match next_moe_layer {
            Some(l) => l,
            None => return Ok(()), // no more MoE layers
        };

        let next_moe_idx = match self.layer_weights[next_layer].moe_layer_idx {
            Some(idx) => idx,
            None => return Ok(()),
        };

        // Swap to the other buffer
        let next_buf = 1 - self.fused_expert_buf_cur;
        let (w1_buf, w1s_buf, w2_buf, w2s_buf) = if next_buf == 0 {
            (self.d_fused_expert_w1_a.as_ref(), self.d_fused_expert_w1s_a.as_ref(),
             self.d_fused_expert_w2_a.as_ref(), self.d_fused_expert_w2s_a.as_ref())
        } else {
            (self.d_fused_expert_w1_b.as_ref(), self.d_fused_expert_w1s_b.as_ref(),
             self.d_fused_expert_w2_b.as_ref(), self.d_fused_expert_w2s_b.as_ref())
        };

        let w1_base = match w1_buf { Some(b) => *b.device_ptr(), None => return Ok(()) };
        let w1s_base = match w1s_buf { Some(b) => *b.device_ptr(), None => return Ok(()) };
        let w2_base = match w2_buf { Some(b) => *b.device_ptr(), None => return Ok(()) };
        let w2s_base = match w2s_buf { Some(b) => *b.device_ptr(), None => return Ok(()) };

        // Record compute event so DMA waits for current compute to stop using the old buffer
        unsafe {
            cuda_sys::lib().cuEventRecord(self.compute_event, self.stream);
            cuda_sys::lib().cuStreamWaitEvent(self.copy_stream, self.compute_event, 0);
        }

        // DMA all experts for next layer into the other buffer
        if let Some(Some(moe_data)) = self.moe_layers.get(next_moe_idx) {
            for eid in 0..moe_data.experts.len() {
                let e = &moe_data.experts[eid];
                let w1_offset = eid * self.w1_packed_per_expert;
                let w1s_offset = eid * self.w1_scales_per_expert;
                let w2_offset = eid * self.w2_packed_per_expert;
                let w2s_offset = eid * self.w2_scales_per_expert;

                if let Some((hcs_w13p, hcs_w13s, hcs_w2p, hcs_w2s)) = self.hcs_lookup(next_moe_idx, eid) {
                    unsafe {
                        cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                            w1_base + w1_offset as u64, hcs_w13p,
                            self.w1_packed_per_expert, self.copy_stream);
                        cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                            w1s_base + w1s_offset as u64, hcs_w13s,
                            self.w1_scales_per_expert, self.copy_stream);
                        cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                            w2_base + w2_offset as u64, hcs_w2p,
                            self.w2_packed_per_expert, self.copy_stream);
                        cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                            w2s_base + w2s_offset as u64, hcs_w2s,
                            self.w2_scales_per_expert, self.copy_stream);
                    }
                } else {
                    unsafe {
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            w1_base + w1_offset as u64,
                            e.w13_packed_ptr as *const _,
                            e.w13_packed_bytes, self.copy_stream);
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            w1s_base + w1s_offset as u64,
                            e.w13_scales_ptr as *const _,
                            e.w13_scales_bytes, self.copy_stream);
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            w2_base + w2_offset as u64,
                            e.w2_packed_ptr as *const _,
                            e.w2_packed_bytes, self.copy_stream);
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            w2s_base + w2s_offset as u64,
                            e.w2_scales_ptr as *const _,
                            e.w2_scales_bytes, self.copy_stream);
                    }
                }
            }
            // Record DMA event so next layer's forward_moe_fused can wait on it
            unsafe {
                cuda_sys::lib().cuEventRecord(self.dma_event, self.copy_stream);
            }
            self.preloaded_moe_layer = Some(next_layer);
            self.fused_expert_buf_cur = next_buf;
        }

        Ok(())
    }

    /// Launch shared expert on the dedicated shared_stream (gap 4).
    /// Uses separate workspace/scratch to avoid data races with main stream.
    fn launch_shared_expert_on_shared_stream(&self, layer_idx: usize, m: usize) -> Result<(), String> {
        let lw = &self.layer_weights[layer_idx];
        let h = self.config.hidden_size;
        let gated = lw.moe_gated;
        let activation = lw.moe_activation;
        let hidden = *self.scratch.d_hidden.device_ptr();
        let s1_buf = *self.scratch.d_scratch1.device_ptr();
        let s2_buf = *self.scratch.d_scratch2.device_ptr();
        let shared_ws_len = self.config.sms * 4;

        let sw1 = lw.shared_w1.as_ref().ok_or("missing shared w1")?;
        let sw2 = lw.shared_w2.as_ref().ok_or("missing shared w2")?;
        let shared_inter = sw1.n / (if gated { 2 } else { 1 });

        let st = if sw1.num_bits == 4 { &ScalarType::U4B8 } else { &ScalarType::U8B128 };
        let f = self.kernels.marlin_mm.ok_or("Marlin GEMM not loaded")?;

        // Use shared_stream + shared workspace to avoid conflicts
        let shared_scratch = *self.d_shared_fp32_scratch.device_ptr();
        let shared_ws = *self.d_shared_workspace.device_ptr();

        // w1 GEMM on shared_stream — zero workspace locks + C_tmp first
        unsafe {
            cuda_sys::lib().cuMemsetD32Async(shared_ws, 0, shared_ws_len, self.shared_stream);
            cuda_sys::lib().cuMemsetD32Async(shared_scratch, 0, m * sw1.n, self.shared_stream);
            f(
                hidden as *const _, sw1.packed as *const _,
                s1_buf as *mut _, shared_scratch as *mut _,
                sw1.scales as *const _, std::ptr::null(), std::ptr::null(),
                std::ptr::null(), std::ptr::null(), std::ptr::null_mut(),
                m as i32, sw1.n as i32, sw1.k as i32, sw1.k as i32,
                shared_ws as *mut _,
                st, false, true, false,
                sw1.num_groups as i32, sw1.group_size as i32,
                self.config.device_ordinal as i32,
                self.shared_stream as u64,
                -1, -1, self.config.sms as i32,
                false, true, false,
            );
        }

        // Activation
        if gated {
            let act_t = std::cmp::max(32, ((std::cmp::min(1024, shared_inter) + 31) / 32) * 32);
            let mut ac0 = s2_buf; let mut ac1 = s1_buf; let mut ac2 = shared_inter as i32;
            let kernel = if activation == 1 { self.kernels.relu2 } else { self.kernels.silu_mul };
            unsafe {
                launch(kernel,
                    (m as u32, 1, 1), (act_t as u32, 1, 1), 0, self.shared_stream,
                    &mut [
                        &mut ac0 as *mut _ as *mut std::ffi::c_void,
                        &mut ac1 as *mut _ as *mut std::ffi::c_void,
                        &mut ac2 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
            // w2 GEMM on shared_stream — zero workspace locks + C_tmp first
            unsafe {
                cuda_sys::lib().cuMemsetD32Async(shared_ws, 0, shared_ws_len, self.shared_stream);
                cuda_sys::lib().cuMemsetD32Async(shared_scratch, 0, m * sw2.n, self.shared_stream);
                f(
                    s2_buf as *const _, sw2.packed as *const _,
                    s1_buf as *mut _, shared_scratch as *mut _,
                    sw2.scales as *const _, std::ptr::null(), std::ptr::null(),
                    std::ptr::null(), std::ptr::null(), std::ptr::null_mut(),
                    m as i32, sw2.n as i32, sw2.k as i32, sw2.k as i32,
                    shared_ws as *mut _,
                    st, false, true, false,
                    sw2.num_groups as i32, sw2.group_size as i32,
                    self.config.device_ordinal as i32,
                    self.shared_stream as u64,
                    -1, -1, self.config.sms as i32,
                    false, true, false,
                );
            }
        } else {
            let act_t = std::cmp::max(32, ((std::cmp::min(1024, shared_inter) + 31) / 32) * 32);
            let mut ac0 = s1_buf; let mut ac1 = s1_buf; let mut ac2 = shared_inter as i32;
            unsafe {
                launch(self.kernels.relu2,
                    (m as u32, 1, 1), (act_t as u32, 1, 1), 0, self.shared_stream,
                    &mut [
                        &mut ac0 as *mut _ as *mut std::ffi::c_void,
                        &mut ac1 as *mut _ as *mut std::ffi::c_void,
                        &mut ac2 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
            // w2 GEMM on shared_stream — zero workspace locks + C_tmp first
            unsafe {
                cuda_sys::lib().cuMemsetD32Async(shared_ws, 0, shared_ws_len, self.shared_stream);
                cuda_sys::lib().cuMemsetD32Async(shared_scratch, 0, m * sw2.n, self.shared_stream);
                f(
                    s1_buf as *const _, sw2.packed as *const _,
                    s2_buf as *mut _, shared_scratch as *mut _,
                    sw2.scales as *const _, std::ptr::null(), std::ptr::null(),
                    std::ptr::null(), std::ptr::null(), std::ptr::null_mut(),
                    m as i32, sw2.n as i32, sw2.k as i32, sw2.k as i32,
                    shared_ws as *mut _,
                    st, false, true, false,
                    sw2.num_groups as i32, sw2.group_size as i32,
                    self.config.device_ordinal as i32,
                    self.shared_stream as u64,
                    -1, -1, self.config.sms as i32,
                    false, true, false,
                );
            }
            unsafe {
                cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                    s1_buf, s2_buf, (m * h * 2) as usize, self.shared_stream);
            }
        }

        Ok(())
    }

    // ── Expert DMA + compute helpers ──

    /// DMA expert weights from CPU RAM into a GPU buffer set.
    /// Records dma_event when complete so compute stream can wait on it.
    fn dma_expert_to_buf(
        &self, e: &ExpertWeightPtrs, buf: &(u64, u64, u64, u64),
    ) -> Result<(), String> {
        let (w13p_dst, w13s_dst, w2p_dst, w2s_dst) = *buf;
        unsafe {
            // Always use 4 separate DMAs. The contiguous single-DMA path is broken
            // for prefill because the destination buffers (w13p, w13s, w2p, w2s) are
            // separate GPU allocations that are NOT contiguous in memory. The single
            // DMA would write all data starting at w13p_dst, leaving the other
            // destination buffers with stale data.
            {
                let _ = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    w13p_dst, e.w13_packed_ptr as *const _, e.w13_packed_bytes, self.copy_stream);
                let _ = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    w13s_dst, e.w13_scales_ptr as *const _, e.w13_scales_bytes, self.copy_stream);
                let _ = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    w2p_dst, e.w2_packed_ptr as *const _, e.w2_packed_bytes, self.copy_stream);
                let _ = cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    w2s_dst, e.w2_scales_ptr as *const _, e.w2_scales_bytes, self.copy_stream);
            }
            // Record event so compute stream can wait
            let err = cuda_sys::lib().cuEventRecord(self.dma_event, self.copy_stream);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("Record DMA event: {:?}", err));
            }
        }
        Ok(())
    }

    /// Run w13 GEMM + activation + w2 GEMM for a single expert.
    #[allow(clippy::too_many_arguments)]
    fn run_expert_gemm(
        &self,
        gathered_slice: u64, expert_out: u64, gate_up_buf: u64, inter_buf: u64,
        w13_packed: u64, w13_scales: u64, w2_packed: u64, w2_scales: u64,
        count: usize, offset: usize,
        w13_n: usize, h: usize, inter: usize, gs: usize, bits: u8,
        num_groups_w13: usize, num_groups_w2: usize,
        gated: bool, activation: u8,
    ) -> Result<(), String> {
        // w1 GEMM: [count, hidden] @ w13 -> [count, w13_n]
        let w13 = MarlinWeight {
            packed: w13_packed, scales: w13_scales,
            n: w13_n, k: h,
            num_groups: num_groups_w13,
            group_size: gs, num_bits: bits,
        };
        self.marlin_gemm(gathered_slice, &w13, gate_up_buf + (offset * w13_n * 2) as u64, count)?;

        if gated {
            let act_t = std::cmp::max(32, ((std::cmp::min(1024, inter) + 31) / 32) * 32);
            let inter_slice = inter_buf + (offset * inter * 2) as u64;
            let gate_up_slice = gate_up_buf + (offset * w13_n * 2) as u64;
            let mut ac0 = inter_slice;
            let mut ac1 = gate_up_slice;
            let mut ac2 = inter as i32;
            let kernel = if activation == 1 { self.kernels.relu2 } else { self.kernels.silu_mul };
            unsafe {
                launch(kernel,
                    (count as u32, 1, 1), (act_t as u32, 1, 1), 0, self.stream,
                    &mut [
                        &mut ac0 as *mut _ as *mut std::ffi::c_void,
                        &mut ac1 as *mut _ as *mut std::ffi::c_void,
                        &mut ac2 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }

            // w2 GEMM: [count, inter] @ w2 -> [count, hidden]
            let w2 = MarlinWeight {
                packed: w2_packed, scales: w2_scales,
                n: h, k: inter,
                num_groups: num_groups_w2,
                group_size: gs, num_bits: bits,
            };
            self.marlin_gemm(inter_slice, &w2, expert_out + (offset * h * 2) as u64, count)?;
        } else {
            // Ungated: relu2 directly on w1 output, then w2
            let act_t = std::cmp::max(32, ((std::cmp::min(1024, inter) + 31) / 32) * 32);
            let w1_out = gate_up_buf + (offset * w13_n * 2) as u64;
            let mut ac0 = w1_out;
            let mut ac1 = w1_out;
            let mut ac2 = inter as i32;
            unsafe {
                launch(self.kernels.relu2,
                    (count as u32, 1, 1), (act_t as u32, 1, 1), 0, self.stream,
                    &mut [
                        &mut ac0 as *mut _ as *mut std::ffi::c_void,
                        &mut ac1 as *mut _ as *mut std::ffi::c_void,
                        &mut ac2 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
            let w2 = MarlinWeight {
                packed: w2_packed, scales: w2_scales,
                n: h, k: inter,
                num_groups: num_groups_w2,
                group_size: gs, num_bits: bits,
            };
            self.marlin_gemm(w1_out, &w2, expert_out + (offset * h * 2) as u64, count)?;
        }
        Ok(())
    }

    // ── LM head + sampling ──

    fn lm_head_and_sample(
        &mut self, m: usize, temperature: f32, suppress_tokens: &[u32],
    ) -> Result<u32, String> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let v = cfg.vocab_size;

        let last_tok = *self.scratch.d_hidden.device_ptr() + ((m - 1) * h * 2) as u64;

        if let Some(ref lm) = self.lm_head {
            self.marlin_gemm(last_tok, lm, *self.scratch.d_logits.device_ptr(), 1)?;
        } else if self.lm_head_bf16_ptr != 0 {
            // BF16 LM head via cuBLAS GEMM
            use cudarc::cublas::sys as cublas_sys;
            use cudarc::cublas::result as cublas_result;
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            unsafe {
                cublas_result::set_stream(
                    self.cublas_handle,
                    self.stream as cublas_sys::cudaStream_t,
                ).map_err(|e| format!("cuBLAS set stream: {:?}", e))?;
                cublas_result::gemm_ex(
                    self.cublas_handle,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    self.lm_head_bf16_rows as i32, 1, self.lm_head_bf16_cols as i32,
                    &alpha as *const f32 as *const std::ffi::c_void,
                    self.lm_head_bf16_ptr as *const std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_16BF, self.lm_head_bf16_cols as i32,
                    last_tok as *const std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_16BF, h as i32,
                    &beta as *const f32 as *const std::ffi::c_void,
                    *self.scratch.d_logits.device_ptr() as *mut std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_32F, self.lm_head_bf16_rows as i32,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                ).map_err(|e| format!("LM head BF16 GEMM: {:?}", e))?;
            }
        } else {
            return Err("No LM head (neither Marlin nor BF16)".to_string());
        }

        // Download logits
        self.h_logits.resize(v, 0.0);
        unsafe {
            let err = cuda_sys::lib().cuMemcpyDtoHAsync_v2(
                self.h_logits.as_mut_ptr() as *mut _,
                *self.scratch.d_logits.device_ptr(),
                (v * 4) as usize,
                self.stream,
            );
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("Download logits: {:?}", err));
            }
        }
        self.stream_sync()?;

        // Suppress tokens
        for &t in suppress_tokens {
            if (t as usize) < v { self.h_logits[t as usize] = f32::NEG_INFINITY; }
        }

        // Sample
        if temperature < 1e-6 {
            // Greedy
            let (best, _) = self.h_logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            Ok(best as u32)
        } else {
            // Temperature-scaled softmax sampling
            let max_l = self.h_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let inv_t = 1.0 / temperature;
            let mut sum = 0.0f64;
            for l in &mut self.h_logits {
                *l = ((*l - max_l) * inv_t).exp();
                sum += *l as f64;
            }
            let inv_sum = 1.0 / sum;
            for l in &mut self.h_logits { *l = (*l as f64 * inv_sum) as f32; }

            let r: f64 = {
                let mut s = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default().as_nanos() as u64;
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                (s as f64) / (u64::MAX as f64)
            };
            let mut cum = 0.0f64;
            for (i, &p) in self.h_logits.iter().enumerate() {
                cum += p as f64;
                if cum >= r { return Ok(i as u32); }
            }
            Ok((v - 1) as u32)
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
//  Kernel loading
// ════════════════════════════════════════════════════════════════════════

impl PrefillKernels {
    #[cfg(has_prefill_kernels)]
    pub fn load(device: Arc<CudaDevice>) -> Result<Self, String> {
        use cudarc::nvrtc::Ptx;
        device.load_ptx(
            Ptx::from_src(PREFILL_KERNELS_PTX),
            "prefill_kernels",
            &[
                "rmsnorm_batched_kernel",
                "fused_add_rmsnorm_batched_kernel",
                "embedding_batched_kernel",
                "rope_batched_kernel",
                "silu_mul_batched_kernel",
                "relu2_batched_kernel",
                "sigmoid_mul_kernel",
                "bf16_to_fp32_kernel",
                "fp32_to_bf16_kernel",
                "sigmoid_topk_kernel",
                "softmax_topk_kernel",
                "moe_sum_reduce_kernel",
                "gqa_prefill_kernel",
                "kv_cache_append_kernel",
                "causal_conv1d_fwd_kernel",
                "mamba2_ssd_sequential_kernel",
                "mamba2_extract_kernel",
                "moe_gather_kernel",
                "moe_scatter_add_kernel",
                "moe_zero_accum_kernel",
                "moe_add_shared_kernel",
                "moe_add_shared_gated_kernel",
                "fp32_to_bf16_batch_kernel",
                // Linear attention kernels
                "la_uninterleave_qkvz_kernel",
                "la_uninterleave_ba_kernel",
                "la_depthwise_conv1d_silu_kernel",
                "la_l2norm_per_head_kernel",
                "la_compute_gate_beta_kernel",
                "la_repeat_interleave_kernel",
                "la_gated_rmsnorm_kernel",
                "la_cumsum_kernel",
                "la_build_attn_matrix_kernel",
                "la_triangular_solve_kernel",
                "la_compute_v_new_kernel",
                "la_chunk_output_kernel",
                "la_state_update_kernel",
                "la_scale_by_exp_g_kernel",
                "la_apply_beta_kernel",
                "la_bf16_to_fp32_kernel",
                "la_fp32_to_bf16_kernel",
                "la_transpose_f32_kernel",
                "transpose_3d_021_kernel",
                "transpose_3d_021_bf16_kernel",
                "flash_attn_tiled_kernel",
                "gated_q_split_kernel",
                "la_split_conv_output_kernel",
                "concat_3_bf16_kernel",
                "la_compute_v_new_strided_kernel",
                "la_chunk_output_strided_kernel",
                "la_state_update_strided_kernel",
                "moe_count_experts_kernel",
                "moe_prefix_sum_kernel",
                "moe_build_maps_kernel",
                "moe_padded_prefix_sum_kernel",
                "moe_scatter_sorted_kernel",
                "moe_gather_sorted_kernel",
                "moe_scatter_fused_kernel",
                "moe_accum_to_bf16_kernel",
                "kv_cache_append_fp8_kernel",
            ],
        ).map_err(|e| format!("Load prefill PTX: {e}"))?;

        let get = |name: &str| -> Result<RawCuFunc, String> {
            let f = device.get_func("prefill_kernels", name)
                .ok_or_else(|| format!("Kernel {name} not found"))?;
            Ok(extract_cu_func(&f))
        };

        let marlin_mm = load_marlin_mm();
        if marlin_mm.is_some() {
            log::info!("Prefill: loaded Marlin GEMM from vendored libkrasis_marlin.so");
        } else {
            log::warn!("Prefill: Marlin GEMM not available");
        }

        // Opt-in to extended shared memory for flash attention kernel.
        // Models with head_dim > ~176 need > 48KB smem per block.
        // Query the device's max smem and set the kernel attribute.
        let flash_attn_func = get("flash_attn_tiled_kernel")?;
        unsafe {
            let mut max_smem: i32 = 0;
            cuda_sys::lib().cuDeviceGetAttribute(
                &mut max_smem,
                cuda_sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                0,
            );
            if max_smem > 49152 {
                let rc = cuda_sys::lib().cuFuncSetAttribute(
                    flash_attn_func.0,
                    cuda_sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    max_smem,
                );
                if rc == cuda_sys::CUresult::CUDA_SUCCESS {
                    log::info!("Prefill flash attention: opt-in shared memory = {} KB", max_smem / 1024);
                } else {
                    log::warn!("Prefill flash attention: failed to set extended smem ({} bytes): {:?}", max_smem, rc);
                }
            }
        }

        Ok(PrefillKernels {
            rmsnorm: get("rmsnorm_batched_kernel")?,
            fused_add_rmsnorm: get("fused_add_rmsnorm_batched_kernel")?,
            embedding: get("embedding_batched_kernel")?,
            rope: get("rope_batched_kernel")?,
            silu_mul: get("silu_mul_batched_kernel")?,
            relu2: get("relu2_batched_kernel")?,
            sigmoid_mul: get("sigmoid_mul_kernel")?,
            sigmoid_topk: get("sigmoid_topk_kernel")?,
            softmax_topk: get("softmax_topk_kernel")?,
            moe_sum_reduce: get("moe_sum_reduce_kernel")?,
            gqa_prefill: get("gqa_prefill_kernel")?,
            kv_cache_append: get("kv_cache_append_kernel")?,
            causal_conv1d: get("causal_conv1d_fwd_kernel")?,
            mamba2_ssd: get("mamba2_ssd_sequential_kernel")?,
            mamba2_extract: get("mamba2_extract_kernel")?,
            moe_gather: get("moe_gather_kernel")?,
            moe_scatter_add: get("moe_scatter_add_kernel")?,
            moe_zero_accum: get("moe_zero_accum_kernel")?,
            moe_add_shared: get("moe_add_shared_kernel")?,
            moe_add_shared_gated: get("moe_add_shared_gated_kernel")?,
            moe_accum_to_bf16: get("moe_accum_to_bf16_kernel")?,
            fp32_to_bf16_batch: get("fp32_to_bf16_batch_kernel")?,
            la_uninterleave_qkvz: get("la_uninterleave_qkvz_kernel")?,
            la_uninterleave_ba: get("la_uninterleave_ba_kernel")?,
            la_depthwise_conv1d_silu: get("la_depthwise_conv1d_silu_kernel")?,
            la_l2norm_per_head: get("la_l2norm_per_head_kernel")?,
            la_compute_gate_beta: get("la_compute_gate_beta_kernel")?,
            la_repeat_interleave: get("la_repeat_interleave_kernel")?,
            la_gated_rmsnorm: get("la_gated_rmsnorm_kernel")?,
            la_cumsum: get("la_cumsum_kernel")?,
            la_build_attn_matrix: get("la_build_attn_matrix_kernel")?,
            la_triangular_solve: get("la_triangular_solve_kernel")?,
            la_compute_v_new: get("la_compute_v_new_kernel")?,
            la_chunk_output: get("la_chunk_output_kernel")?,
            la_state_update: get("la_state_update_kernel")?,
            la_scale_by_exp_g: get("la_scale_by_exp_g_kernel")?,
            la_apply_beta: get("la_apply_beta_kernel")?,
            la_bf16_to_fp32: get("la_bf16_to_fp32_kernel")?,
            la_fp32_to_bf16: get("la_fp32_to_bf16_kernel")?,
            la_transpose_f32: get("la_transpose_f32_kernel")?,
            transpose_3d_021: get("transpose_3d_021_kernel")?,
            transpose_3d_021_bf16: get("transpose_3d_021_bf16_kernel")?,
            flash_attn_tiled: flash_attn_func,
            gated_q_split: get("gated_q_split_kernel")?,
            la_split_conv_output: get("la_split_conv_output_kernel")?,
            concat_3_bf16: get("concat_3_bf16_kernel")?,
            la_compute_v_new_strided: get("la_compute_v_new_strided_kernel")?,
            la_chunk_output_strided: get("la_chunk_output_strided_kernel")?,
            la_state_update_strided: get("la_state_update_strided_kernel")?,
            moe_count_experts: get("moe_count_experts_kernel")?,
            moe_prefix_sum: get("moe_prefix_sum_kernel")?,
            moe_build_maps: get("moe_build_maps_kernel")?,
            moe_padded_prefix_sum: get("moe_padded_prefix_sum_kernel")?,
            moe_scatter_sorted: get("moe_scatter_sorted_kernel")?,
            moe_gather_sorted: get("moe_gather_sorted_kernel")?,
            moe_scatter_fused: get("moe_scatter_fused_kernel")?,
            marlin_mm,
            fused_moe_fn: load_fused_moe(),
        })
    }

    #[cfg(not(has_prefill_kernels))]
    pub fn load(_device: Arc<CudaDevice>) -> Result<Self, String> {
        Err("Prefill kernels not compiled (nvcc not found)".to_string())
    }
}

// ════════════════════════════════════════════════════════════════════════
//  Scratch allocation
// ════════════════════════════════════════════════════════════════════════

pub fn allocate_scratch(
    device: &Arc<CudaDevice>,
    config: &PrefillModelConfig,
    max_tokens: usize,
) -> Result<PrefillScratch, String> {
    let h = config.hidden_size;
    let inter = config.intermediate_size;           // max of dense + moe for general scratch
    let moe_inter = config.moe_intermediate_size;   // actual MoE expert intermediate
    let topk = config.num_experts_per_tok.max(1);

    let alloc_u16 = |n: usize, name: &str| -> Result<CudaSlice<u16>, String> {
        device.alloc_zeros::<u16>(n).map_err(|e| format!("alloc {name}: {e}"))
    };
    let alloc_f32 = |n: usize, name: &str| -> Result<CudaSlice<f32>, String> {
        device.alloc_zeros::<f32>(n).map_err(|e| format!("alloc {name}: {e}"))
    };
    let alloc_i32 = |n: usize, name: &str| -> Result<CudaSlice<i32>, String> {
        device.alloc_zeros::<i32>(n).map_err(|e| format!("alloc {name}: {e}"))
    };

    let max_inter = std::cmp::max(
        max_tokens * inter * 2,
        max_tokens * config.num_q_heads * config.head_dim * 2, // *2 for gated GQA (q + gate)
    );

    let has_mamba2 = config.layer_types.iter().any(|&t| t == 1);
    let has_la = config.layer_types.iter().any(|&t| t == 3);
    // LA pads to chunk_size multiples, so total_len can be up to max_tokens + chunk_size - 1
    let la_max_len = if has_la { max_tokens + config.la_chunk_size } else { max_tokens };

    Ok(PrefillScratch {
        d_hidden: alloc_u16(max_tokens * h, "hidden")?,
        d_residual: alloc_u16(max_tokens * h, "residual")?,
        d_scratch1: alloc_u16(max_inter, "scratch1")?,
        d_scratch2: alloc_u16(max_inter, "scratch2")?,
        d_fp32_scratch: {
            // fp32_scratch must be large enough for:
            // 1. Marlin C_tmp (FP32 reduce): max_tokens * max_gemm_n
            //    Largest GEMM output dimension across all layers:
            //    - LA in_proj_qkvz: nk * group_dim (e.g. 16 * 768 = 12288 for Q3.5)
            //    - GQA q_proj: num_q_heads * head_dim * (2 if gated)
            //    - LA out_proj: hidden_size
            //    - Dense MLP gate_up: inter * 2
            // 2. Conv1d transpose: max_tokens * conv_dim (LA layers)
            // 3. LA chunk loop: nv * max_tokens * dv (output buf)
            //    + nv * chunk_size * (2*dk + dv) + nv * chunk_size (temp buffers)
            let mut max_gemm_n = std::cmp::max(h, inter * 2);
            // GQA q_proj (gated = 2x)
            let gqa_q_n = config.num_q_heads * config.head_dim * 2;
            if gqa_q_n > max_gemm_n { max_gemm_n = gqa_q_n; }
            if has_la {
                let nk = config.la_num_k_heads;
                let dk = config.la_k_head_dim;
                let dv = config.la_v_head_dim;
                let hr = config.la_head_ratio.max(1);
                let group_dim = 2 * dk + 2 * hr * dv;
                let qkvz_n = nk * group_dim;
                if qkvz_n > max_gemm_n { max_gemm_n = qkvz_n; }
            }
            let marlin_ctmp = max_tokens * max_gemm_n;
            let la_size = if has_la {
                let nv = config.la_num_v_heads;
                let dk = config.la_k_head_dim;
                let dv = config.la_v_head_dim;
                let cs = config.la_chunk_size;
                let conv_dim = config.la_conv_dim;
                let output_buf = nv * la_max_len * dv;
                let chunk_temps = nv * cs * (2 * dk + dv) + nv * cs;
                let conv_transpose = la_max_len * conv_dim;
                std::cmp::max(output_buf + chunk_temps, conv_transpose)
            } else { 0 };
            alloc_f32(std::cmp::max(marlin_ctmp, la_size), "fp32_scratch")?
        },
        d_workspace: alloc_i32(config.sms * 4, "workspace")?,
        d_topk_weights: alloc_f32(max_tokens * topk, "topk_weights")?,
        d_topk_ids: alloc_i32(max_tokens * topk, "topk_ids")?,
        d_token_ids: alloc_i32(max_tokens, "token_ids")?,
        d_positions: alloc_i32(max_tokens, "positions")?,
        d_attn_out: alloc_u16(max_tokens * h, "attn_out")?,
        d_q: alloc_u16(max_tokens * config.num_q_heads * config.head_dim, "q")?,
        d_k: alloc_u16(max_tokens * config.num_kv_heads * config.head_dim, "k")?,
        d_v: alloc_u16(max_tokens * config.num_kv_heads * config.head_dim, "v")?,
        d_logits: alloc_f32(config.vocab_size, "logits")?,
        d_mamba2_conv_state: if has_mamba2 {
            Some(alloc_f32(
                config.mamba_conv_dim * config.mamba_conv_kernel.max(2) - 1,
                "mamba2_conv",
            )?)
        } else { None },
        d_mamba2_ssm_state: if has_mamba2 {
            Some(alloc_f32(
                config.mamba_num_heads * config.mamba_head_dim * config.mamba_d_state,
                "mamba2_ssm",
            )?)
        } else { None },
        d_mamba2_in_proj: if has_mamba2 {
            let dim = 2 * config.mamba_d_inner
                + 2 * config.mamba_n_groups * config.mamba_d_state
                + config.mamba_num_heads;
            Some(alloc_u16(max_tokens * dim, "mamba2_in_proj")?)
        } else { None },
        d_mamba2_ssd_out: if has_mamba2 {
            Some(alloc_u16(max_tokens * config.mamba_d_inner, "mamba2_ssd_out")?)
        } else { None },
        // MoE buffers
        d_gate_out: alloc_f32(max_tokens * config.n_routed_experts.max(1), "gate_out")?,
        d_moe_accum: alloc_f32(max_tokens * h, "moe_accum")?,
        d_moe_gathered: alloc_u16(max_tokens * topk * h, "moe_gathered")?,
        d_moe_expert_out: alloc_u16(max_tokens * topk * h, "moe_expert_out")?,
        d_moe_gate_up: alloc_u16(max_tokens * topk * moe_inter * 2, "moe_gate_up")?,
        d_moe_inter: alloc_u16(max_tokens * topk * moe_inter, "moe_inter")?,
        d_gather_src_map: alloc_i32(max_tokens * topk, "gather_src_map")?,
        d_gather_weight_map: alloc_f32(max_tokens * topk, "gather_weight_map")?,
        // GPU-only routing scratch
        d_expert_counts: alloc_i32(config.n_routed_experts.max(1), "expert_counts")?,
        d_expert_offsets: alloc_i32(config.n_routed_experts.max(1) + 1, "expert_offsets")?,
        d_write_offsets: alloc_i32(config.n_routed_experts.max(1), "write_offsets")?,
        // Expert DMA double-buffers (A and B, each sized for one expert)
        d_expert_w13_packed_a: {
            let w13_size = if config.n_routed_experts > 0 {
                let w13_n = if config.moe_gated { 2 * moe_inter } else { moe_inter };
                (h / 16) * w13_n * (config.expert_bits as usize / 2)
            } else { 1 };
            device.alloc_zeros::<u8>(w13_size).map_err(|e| format!("alloc expert_w13_packed_a: {e}"))?
        },
        d_expert_w13_scales_a: {
            let scale_size = if config.n_routed_experts > 0 {
                let w13_n = if config.moe_gated { 2 * moe_inter } else { moe_inter };
                (h / config.group_size) * w13_n * 2
            } else { 1 };
            device.alloc_zeros::<u8>(scale_size).map_err(|e| format!("alloc expert_w13_scales_a: {e}"))?
        },
        d_expert_w2_packed_a: {
            let w2_size = if config.n_routed_experts > 0 {
                (moe_inter / 16) * h * (config.expert_bits as usize / 2)
            } else { 1 };
            device.alloc_zeros::<u8>(w2_size).map_err(|e| format!("alloc expert_w2_packed_a: {e}"))?
        },
        d_expert_w2_scales_a: {
            let scale_size = if config.n_routed_experts > 0 {
                (moe_inter / config.group_size) * h * 2
            } else { 1 };
            device.alloc_zeros::<u8>(scale_size).map_err(|e| format!("alloc expert_w2_scales_a: {e}"))?
        },
        d_expert_w13_packed_b: {
            let w13_size = if config.n_routed_experts > 0 {
                let w13_n = if config.moe_gated { 2 * moe_inter } else { moe_inter };
                (h / 16) * w13_n * (config.expert_bits as usize / 2)
            } else { 1 };
            device.alloc_zeros::<u8>(w13_size).map_err(|e| format!("alloc expert_w13_packed_b: {e}"))?
        },
        d_expert_w13_scales_b: {
            let scale_size = if config.n_routed_experts > 0 {
                let w13_n = if config.moe_gated { 2 * moe_inter } else { moe_inter };
                (h / config.group_size) * w13_n * 2
            } else { 1 };
            device.alloc_zeros::<u8>(scale_size).map_err(|e| format!("alloc expert_w13_scales_b: {e}"))?
        },
        d_expert_w2_packed_b: {
            let w2_size = if config.n_routed_experts > 0 {
                (moe_inter / 16) * h * (config.expert_bits as usize / 2)
            } else { 1 };
            device.alloc_zeros::<u8>(w2_size).map_err(|e| format!("alloc expert_w2_packed_b: {e}"))?
        },
        d_expert_w2_scales_b: {
            let scale_size = if config.n_routed_experts > 0 {
                (moe_inter / config.group_size) * h * 2
            } else { 1 };
            device.alloc_zeros::<u8>(scale_size).map_err(|e| format!("alloc expert_w2_scales_b: {e}"))?
        },
        // Linear attention scratch buffers
        // LA pads to chunk_size multiples, so total_len can be up to max_tokens + chunk_size - 1.
        // Allocate with la_max_len to avoid buffer overflow on padded data.
        d_la_q: if has_la {
            Some(alloc_f32(la_max_len * config.la_num_v_heads * config.la_k_head_dim, "la_q")?)
        } else { None },
        d_la_k: if has_la {
            Some(alloc_f32(la_max_len * config.la_num_v_heads * config.la_k_head_dim, "la_k")?)
        } else { None },
        d_la_v: if has_la {
            Some(alloc_f32(la_max_len * config.la_num_v_heads * config.la_v_head_dim, "la_v")?)
        } else { None },
        d_la_z: if has_la {
            Some(alloc_u16(la_max_len * config.la_num_v_heads * config.la_v_head_dim, "la_z")?)
        } else { None },
        d_la_b: if has_la {
            Some(alloc_u16(la_max_len * config.la_num_v_heads, "la_b")?)
        } else { None },
        d_la_a: if has_la {
            Some(alloc_u16(la_max_len * config.la_num_v_heads, "la_a")?)
        } else { None },
        d_la_beta: if has_la {
            Some(alloc_f32(la_max_len * config.la_num_v_heads, "la_beta")?)
        } else { None },
        d_la_gate: if has_la {
            Some(alloc_f32(la_max_len * config.la_num_v_heads, "la_gate")?)
        } else { None },
        d_la_conv_out: if has_la {
            Some(alloc_f32(config.la_conv_dim * la_max_len, "la_conv_out")?)
        } else { None },
        d_la_v_beta: if has_la {
            Some(alloc_f32(config.la_num_v_heads * la_max_len * config.la_v_head_dim, "la_v_beta")?)
        } else { None },
        d_la_k_beta: if has_la {
            Some(alloc_f32(config.la_num_v_heads * la_max_len * config.la_k_head_dim, "la_k_beta")?)
        } else { None },
        d_la_v_new: if has_la {
            Some(alloc_f32(config.la_num_v_heads * config.la_chunk_size * config.la_v_head_dim, "la_v_new")?)
        } else { None },
        d_la_g_cum: if has_la {
            let num_chunks = (la_max_len + config.la_chunk_size - 1) / config.la_chunk_size;
            Some(alloc_f32(config.la_num_v_heads * num_chunks * config.la_chunk_size, "la_g_cum")?)
        } else { None },
        d_la_attn: if has_la {
            let num_chunks = (la_max_len + config.la_chunk_size - 1) / config.la_chunk_size;
            Some(alloc_f32(config.la_num_v_heads * num_chunks * config.la_chunk_size * config.la_chunk_size, "la_attn")?)
        } else { None },
        d_la_state: if has_la {
            Some(alloc_f32(config.la_num_v_heads * config.la_k_head_dim * config.la_v_head_dim, "la_state")?)
        } else { None },
        d_la_chunk_out: if has_la {
            Some(alloc_f32(config.la_num_v_heads * config.la_chunk_size * config.la_v_head_dim, "la_chunk_out")?)
        } else { None },
        d_la_q_contig: if has_la {
            Some(alloc_f32(config.la_num_v_heads * config.la_chunk_size * config.la_k_head_dim, "la_q_contig")?)
        } else { None },
        d_la_proj_buf: if has_la {
            // Projection buffer: max of qkvz_dim and ba_dim
            let group_dim = 2 * config.la_k_head_dim + 2 * config.la_head_ratio * config.la_v_head_dim;
            let qkvz_dim = config.la_num_k_heads * group_dim;
            Some(alloc_u16(max_tokens * qkvz_dim, "la_proj_buf")?)
        } else { None },
        max_tokens,
    })
}

// ════════════════════════════════════════════════════════════════════════
//  Vendored Marlin GEMM dlopen
// ════════════════════════════════════════════════════════════════════════

fn load_marlin_mm() -> Option<MarlinMmFn> {
    let path = find_marlin_so()?;
    unsafe {
        let lib = libc::dlopen(
            std::ffi::CString::new(path.as_str()).ok()?.as_ptr(),
            libc::RTLD_NOW | libc::RTLD_LOCAL,
        );
        if lib.is_null() {
            let err = libc::dlerror();
            if !err.is_null() {
                log::warn!("dlopen failed: {}", std::ffi::CStr::from_ptr(err).to_string_lossy());
            }
            return None;
        }

        let sym = libc::dlsym(lib, b"krasis_marlin_mm_bf16\0".as_ptr() as *const _);
        if !sym.is_null() {
            log::info!("Loaded vendored krasis_marlin_mm_bf16 from {}", path);
            return Some(std::mem::transmute(sym));
        }
        log::warn!("krasis_marlin_mm_bf16 not found in {}", path);
    }
    None
}

fn load_fused_moe() -> Option<FusedMoeFn> {
    let path = find_marlin_so()?;
    unsafe {
        let lib = libc::dlopen(
            std::ffi::CString::new(path.as_str()).ok()?.as_ptr(),
            libc::RTLD_NOW | libc::RTLD_LOCAL,
        );
        if lib.is_null() { return None; }

        let sym = libc::dlsym(lib, b"krasis_marlin_moe_mm_bf16\0".as_ptr() as *const _);
        if !sym.is_null() {
            log::info!("Loaded vendored krasis_marlin_moe_mm_bf16 from {}", path);
            return Some(std::mem::transmute(sym));
        }
        log::warn!("krasis_marlin_moe_mm_bf16 not found in {}", path);
    }
    None
}

/// Try to load FlashAttention C API via dlopen (public for use from gpu_decode.rs).
pub fn load_flash_attn() -> Option<FlashAttnFn> {
    let f = load_flash_attn_fn();
    if f.is_some() {
        log::info!("Prefill: FlashAttention C API loaded");
    } else {
        log::info!("Prefill: FlashAttention not found, using tiled kernel");
    }
    f
}

/// Try to load FlashAttention C API via dlopen.
/// Looks for flash_attn_varlen_fwd in the flash_attn package's .so files.
fn load_flash_attn_fn() -> Option<FlashAttnFn> {
    let home = std::env::var("HOME").ok()?;

    // Look for flash_attn C++ .so in conda envs
    for env in &["krasis", "ktransformers"] {
        for pv in &["3.11", "3.12", "3.10", "3.13"] {
            let base = format!(
                "{}/miniconda3/envs/{}/lib/python{}/site-packages/flash_attn",
                home, env, pv
            );
            if let Ok(entries) = std::fs::read_dir(&base) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                        if name.contains("flash_attn_2_cuda") && name.ends_with(".so") {
                            unsafe {
                                let lib = libc::dlopen(
                                    std::ffi::CString::new(p.to_string_lossy().as_bytes()).ok()?.as_ptr(),
                                    libc::RTLD_NOW | libc::RTLD_LOCAL,
                                );
                                if !lib.is_null() {
                                    // Try common symbol names for the varlen forward function
                                    for sym_name in &[
                                        b"mha_varlen_fwd\0" as &[u8],
                                        b"flash_attn_varlen_fwd\0" as &[u8],
                                    ] {
                                        let sym = libc::dlsym(lib, sym_name.as_ptr() as *const _);
                                        if !sym.is_null() {
                                            log::info!("Loaded FlashAttention from {:?}", p);
                                            return Some(std::mem::transmute(sym));
                                        }
                                    }
                                    libc::dlclose(lib);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

// ════════════════════════════════════════════════════════════════════════
//  Kernel Unit Tests
// ════════════════════════════════════════════════════════════════════════
//
// These tests are compiled ONLY by `cargo test` (#[cfg(test)]).
// They are completely absent from the release binary.
// Test data lives in tests/kernel_test_data/ (raw .bin + meta.json).
// See feature-unit-test.md for architecture and rationale.
//
#[cfg(test)]
#[cfg(has_prefill_kernels)]
mod kernel_tests {
    use super::*;
    use std::path::PathBuf;

    /// Metadata for a test case (parsed from meta.json).
    #[derive(serde::Deserialize)]
    struct TestMeta {
        atol: f64,
        rtol: f64,
        #[allow(dead_code)]
        note: String,
        inputs: std::collections::HashMap<String, TensorMeta>,
        outputs: std::collections::HashMap<String, TensorMeta>,
    }

    #[derive(serde::Deserialize)]
    struct TensorMeta {
        file: String,
        shape: Vec<usize>,
        dtype: String,
    }

    /// Loaded test data: raw bytes + metadata.
    struct TestData {
        meta: TestMeta,
        dir: PathBuf,
    }

    impl TestData {
        fn load(name: &str) -> Self {
            let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests")
                .join("kernel_test_data")
                .join(name);
            assert!(dir.exists(), "Test data not found: {}. Run: python tests/generate_kernel_test_data.py", dir.display());
            let meta_path = dir.join("meta.json");
            let meta_str = std::fs::read_to_string(&meta_path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {}", meta_path.display(), e));
            let meta: TestMeta = serde_json::from_str(&meta_str)
                .unwrap_or_else(|e| panic!("Failed to parse {}: {}", meta_path.display(), e));
            TestData { meta, dir }
        }

        fn input_bytes(&self, name: &str) -> Vec<u8> {
            let tm = &self.meta.inputs[name];
            std::fs::read(self.dir.join(&tm.file)).unwrap()
        }

        fn expected_bytes(&self, name: &str) -> Vec<u8> {
            let tm = &self.meta.outputs[name];
            std::fs::read(self.dir.join(&tm.file)).unwrap()
        }

        fn input_shape(&self, name: &str) -> &[usize] {
            &self.meta.inputs[name].shape
        }

        fn output_shape(&self, name: &str) -> &[usize] {
            &self.meta.outputs[name].shape
        }

        fn numel(shape: &[usize]) -> usize {
            shape.iter().product::<usize>().max(1)
        }
    }

    /// GPU test context: device + loaded PTX module.
    struct GpuTestCtx {
        dev: Arc<CudaDevice>,
    }

    impl GpuTestCtx {
        fn new() -> Self {
            let dev = CudaDevice::new(0).expect("Failed to create CUDA device for tests");
            dev.load_ptx(
                cudarc::nvrtc::Ptx::from_src(PREFILL_KERNELS_PTX),
                "prefill_kernels",
                &[
                    "rmsnorm_batched_kernel",
                    "fused_add_rmsnorm_batched_kernel",
                    "rope_batched_kernel",
                    "silu_mul_batched_kernel",
                    "relu2_batched_kernel",
                    "sigmoid_mul_kernel",
                    "sigmoid_topk_kernel",
                    "softmax_topk_kernel",
                    "transpose_3d_021_kernel",
                    "transpose_3d_021_bf16_kernel",
                    "concat_3_bf16_kernel",
                    "gated_q_split_kernel",
                    "kv_cache_append_fp8_kernel",
                    "flash_attn_tiled_kernel",
                    "la_l2norm_per_head_kernel",
                    "la_compute_gate_beta_kernel",
                    "la_gated_rmsnorm_kernel",
                    "la_depthwise_conv1d_silu_kernel",
                    "la_uninterleave_qkvz_kernel",
                    "moe_count_experts_kernel",
                    "moe_prefix_sum_kernel",
                    "moe_build_maps_kernel",
                    "moe_scatter_add_kernel",
                    "moe_zero_accum_kernel",
                    "moe_accum_to_bf16_kernel",
                    "moe_sum_reduce_kernel",
                    "la_split_conv_output_kernel",
                    "la_compute_v_new_strided_kernel",
                    "la_chunk_output_strided_kernel",
                    "la_state_update_strided_kernel",
                    "mamba2_extract_kernel",
                    "embedding_batched_kernel",
                    "la_triangular_solve_kernel",
                    "la_compute_v_new_kernel",
                    "la_chunk_output_kernel",
                    "la_state_update_kernel",
                    "moe_gather_sorted_kernel",
                    "moe_scatter_fused_kernel",
                ],
            ).expect("Failed to load prefill kernels PTX");
            GpuTestCtx { dev }
        }

        /// Upload BF16 bytes (u16) to GPU, returns device pointer.
        fn upload_bf16(&self, bytes: &[u8]) -> CudaSlice<u16> {
            let data: Vec<u16> = bytes.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            self.dev.htod_copy(data).unwrap()
        }

        /// Upload FP32 bytes to GPU.
        fn upload_f32(&self, bytes: &[u8]) -> CudaSlice<f32> {
            let data: Vec<f32> = bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            self.dev.htod_copy(data).unwrap()
        }

        /// Upload I32 bytes to GPU.
        fn upload_i32(&self, bytes: &[u8]) -> CudaSlice<i32> {
            let data: Vec<i32> = bytes.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            self.dev.htod_copy(data).unwrap()
        }

        /// Allocate zeroed BF16 buffer.
        fn alloc_bf16(&self, n: usize) -> CudaSlice<u16> {
            self.dev.alloc_zeros::<u16>(n).unwrap()
        }

        /// Allocate zeroed FP32 buffer.
        fn alloc_f32(&self, n: usize) -> CudaSlice<f32> {
            self.dev.alloc_zeros::<f32>(n).unwrap()
        }

        /// Allocate zeroed I32 buffer.
        fn alloc_i32(&self, n: usize) -> CudaSlice<i32> {
            self.dev.alloc_zeros::<i32>(n).unwrap()
        }

        /// Allocate zeroed U8 buffer (for FP8).
        fn alloc_u8(&self, n: usize) -> CudaSlice<u8> {
            self.dev.alloc_zeros::<u8>(n).unwrap()
        }

        /// Upload raw bytes as U8 to GPU.
        fn upload_u8(&self, bytes: &[u8]) -> CudaSlice<u8> {
            self.dev.htod_copy(bytes.to_vec()).unwrap()
        }

        /// Download BF16 from GPU to bytes.
        fn download_bf16(&self, buf: &CudaSlice<u16>) -> Vec<u8> {
            let data = self.dev.dtoh_sync_copy(buf).unwrap();
            data.iter().flat_map(|v| v.to_le_bytes()).collect()
        }

        /// Download FP32 from GPU to bytes.
        fn download_f32(&self, buf: &CudaSlice<f32>) -> Vec<u8> {
            let data = self.dev.dtoh_sync_copy(buf).unwrap();
            data.iter().flat_map(|v| v.to_le_bytes()).collect()
        }

        /// Download I32 from GPU to bytes.
        fn download_i32(&self, buf: &CudaSlice<i32>) -> Vec<u8> {
            let data = self.dev.dtoh_sync_copy(buf).unwrap();
            data.iter().flat_map(|v| v.to_le_bytes()).collect()
        }

        /// Download U8 from GPU to bytes.
        fn download_u8(&self, buf: &CudaSlice<u8>) -> Vec<u8> {
            self.dev.dtoh_sync_copy(buf).unwrap()
        }

        /// Get raw kernel function from loaded PTX.
        fn get_kernel(&self, name: &str) -> RawCuFunc {
            let func = self.dev.get_func("prefill_kernels", name)
                .unwrap_or_else(|| panic!("Kernel not found in PTX: {}", name));
            extract_cu_func(&func)
        }

        fn stream(&self) -> cuda_sys::CUstream {
            // Use default stream (null) for tests
            std::ptr::null_mut()
        }
    }

    /// Compare BF16 bytes with tolerance.
    fn assert_close_bf16(actual: &[u8], expected: &[u8], atol: f64, rtol: f64, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{}: size mismatch {} vs {}", label, actual.len(), expected.len());
        let n = actual.len() / 2;
        let mut max_abs_err = 0.0f64;
        let mut max_rel_err = 0.0f64;
        let mut fail_count = 0usize;

        for i in 0..n {
            let a_bits = u16::from_le_bytes([actual[i*2], actual[i*2+1]]);
            let e_bits = u16::from_le_bytes([expected[i*2], expected[i*2+1]]);
            let a = half::bf16::from_bits(a_bits).to_f64();
            let e = half::bf16::from_bits(e_bits).to_f64();

            let abs_err = (a - e).abs();
            let rel_err = if e.abs() > 1e-6 { abs_err / e.abs() } else { 0.0 };
            max_abs_err = max_abs_err.max(abs_err);
            max_rel_err = max_rel_err.max(rel_err);

            if abs_err > atol && rel_err > rtol {
                fail_count += 1;
                if fail_count <= 5 {
                    eprintln!("  {} mismatch at [{}]: actual={:.6} expected={:.6} abs_err={:.6} rel_err={:.6}",
                              label, i, a, e, abs_err, rel_err);
                }
            }
        }

        if fail_count > 0 {
            panic!("{}: {} / {} elements exceeded tolerance (atol={}, rtol={}). max_abs={:.6}, max_rel={:.6}",
                   label, fail_count, n, atol, rtol, max_abs_err, max_rel_err);
        }
        eprintln!("  {} PASS ({} elements, max_abs_err={:.6}, max_rel_err={:.6})",
                  label, n, max_abs_err, max_rel_err);
    }

    /// Compare FP32 bytes with tolerance.
    fn assert_close_f32(actual: &[u8], expected: &[u8], atol: f64, rtol: f64, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{}: size mismatch", label);
        let n = actual.len() / 4;
        let mut max_abs_err = 0.0f64;
        let mut fail_count = 0usize;

        for i in 0..n {
            let a = f32::from_le_bytes([actual[i*4], actual[i*4+1], actual[i*4+2], actual[i*4+3]]) as f64;
            let e = f32::from_le_bytes([expected[i*4], expected[i*4+1], expected[i*4+2], expected[i*4+3]]) as f64;

            let abs_err = (a - e).abs();
            let rel_err = if e.abs() > 1e-8 { abs_err / e.abs() } else { 0.0 };
            max_abs_err = max_abs_err.max(abs_err);

            if abs_err > atol && rel_err > rtol {
                fail_count += 1;
                if fail_count <= 5 {
                    eprintln!("  {} mismatch at [{}]: actual={:.8} expected={:.8} abs_err={:.8}",
                              label, i, a, e, abs_err);
                }
            }
        }

        if fail_count > 0 {
            panic!("{}: {} / {} elements exceeded tolerance (atol={}, rtol={})",
                   label, fail_count, n, atol, rtol);
        }
        eprintln!("  {} PASS ({} elements, max_abs_err={:.8})", label, n, max_abs_err);
    }

    /// Compare I32 bytes exactly.
    fn assert_exact_i32(actual: &[u8], expected: &[u8], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{}: size mismatch", label);
        let n = actual.len() / 4;
        let mut fail_count = 0usize;

        for i in 0..n {
            let a = i32::from_le_bytes([actual[i*4], actual[i*4+1], actual[i*4+2], actual[i*4+3]]);
            let e = i32::from_le_bytes([expected[i*4], expected[i*4+1], expected[i*4+2], expected[i*4+3]]);
            if a != e {
                fail_count += 1;
                if fail_count <= 5 {
                    eprintln!("  {} mismatch at [{}]: actual={} expected={}", label, i, a, e);
                }
            }
        }

        if fail_count > 0 {
            panic!("{}: {} / {} elements differ", label, fail_count, n);
        }
        eprintln!("  {} PASS ({} elements, exact match)", label, n);
    }

    // ── Test functions ─────────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_m128_h2048() {
        let data = TestData::load("rmsnorm_m128_h2048");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("rmsnorm_batched_kernel");

        let m = data.input_shape("x")[0];
        let d = data.input_shape("x")[1];
        let n = TestData::numel(data.input_shape("x"));

        let d_x = ctx.upload_bf16(&data.input_bytes("x"));
        let d_w = ctx.upload_bf16(&data.input_bytes("weight"));
        let d_out = ctx.alloc_bf16(n);

        let mut eps: f32 = 1e-6;
        let mut m_i32 = m as i32;
        let mut d_i32 = d as i32;

        unsafe {
            let threads = std::cmp::min(1024, d) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let smem = threads * 4; // sizeof(float)
            let mut out_ptr = *d_out.device_ptr() as u64;
            let mut x_ptr = *d_x.device_ptr() as u64;
            let mut w_ptr = *d_w.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut x_ptr as *mut _ as *mut _,
                &mut w_ptr as *mut _ as *mut _,
                &mut d_i32 as *mut _ as *mut _,
                &mut eps as *mut f32 as *mut _,
            ];
            launch(kernel, (m as u32, 1, 1), (threads, 1, 1), smem, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_bf16(&d_out);
        let expected = data.expected_bytes("out");
        assert_close_bf16(&actual, &expected, data.meta.atol, data.meta.rtol, "rmsnorm_m128_h2048");
    }

    #[test]
    fn test_rmsnorm_m1_h2048() {
        let data = TestData::load("rmsnorm_m1_h2048");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("rmsnorm_batched_kernel");

        let m = 1usize;
        let d = data.input_shape("x")[1];
        let n = TestData::numel(data.input_shape("x"));

        let d_x = ctx.upload_bf16(&data.input_bytes("x"));
        let d_w = ctx.upload_bf16(&data.input_bytes("weight"));
        let d_out = ctx.alloc_bf16(n);

        let mut eps: f32 = 1e-6;
        let mut d_i32 = d as i32;

        unsafe {
            let threads = std::cmp::min(1024, d) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let smem = threads * 4;
            let mut out_ptr = *d_out.device_ptr() as u64;
            let mut x_ptr = *d_x.device_ptr() as u64;
            let mut w_ptr = *d_w.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut x_ptr as *mut _ as *mut _,
                &mut w_ptr as *mut _ as *mut _,
                &mut d_i32 as *mut _ as *mut _,
                &mut eps as *mut f32 as *mut _,
            ];
            launch(kernel, (m as u32, 1, 1), (threads, 1, 1), smem, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_bf16(&d_out);
        let expected = data.expected_bytes("out");
        assert_close_bf16(&actual, &expected, data.meta.atol, data.meta.rtol, "rmsnorm_m1_h2048");
    }

    #[test]
    fn test_silu_mul_m128_n2048() {
        let data = TestData::load("silu_mul_m128_n2048");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("silu_mul_batched_kernel");

        let m = data.input_shape("gate_up")[0];
        let n = data.input_shape("gate_up")[1] / 2; // gate_up is [M, 2*N]
        let out_n = TestData::numel(data.output_shape("out"));

        let d_gu = ctx.upload_bf16(&data.input_bytes("gate_up"));
        let d_out = ctx.alloc_bf16(out_n);

        let mut n_i32 = n as i32;

        unsafe {
            let threads = std::cmp::min(1024, n) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let mut out_ptr = *d_out.device_ptr() as u64;
            let mut gu_ptr = *d_gu.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut gu_ptr as *mut _ as *mut _,
                &mut n_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (m as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_bf16(&d_out);
        let expected = data.expected_bytes("out");
        assert_close_bf16(&actual, &expected, data.meta.atol, data.meta.rtol, "silu_mul_m128_n2048");
    }

    #[test]
    fn test_relu2_m128_n2048() {
        let data = TestData::load("relu2_m128_n2048");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("relu2_batched_kernel");

        let m = data.input_shape("x")[0];
        let n = data.input_shape("x")[1];
        let out_n = TestData::numel(data.output_shape("out"));

        let d_x = ctx.upload_bf16(&data.input_bytes("x"));
        let d_out = ctx.alloc_bf16(out_n);

        let mut n_i32 = n as i32;

        unsafe {
            let threads = std::cmp::min(1024, n) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let mut out_ptr = *d_out.device_ptr() as u64;
            let mut x_ptr = *d_x.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut x_ptr as *mut _ as *mut _,
                &mut n_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (m as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_bf16(&d_out);
        let expected = data.expected_bytes("out");
        assert_close_bf16(&actual, &expected, data.meta.atol, data.meta.rtol, "relu2_m128_n2048");
    }

    #[test]
    fn test_sigmoid_topk_m128_e512_k10() {
        let data = TestData::load("sigmoid_topk_m128_e512_k10");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("sigmoid_topk_kernel");

        let m = 128usize;
        let e = 512usize;
        let topk = 10usize;

        let d_gate = ctx.upload_f32(&data.input_bytes("gate"));
        let d_tw = ctx.alloc_f32(m * topk);
        let d_ti = ctx.alloc_i32(m * topk);

        let mut e_i32 = e as i32;
        let mut topk_i32 = topk as i32;

        unsafe {
            let threads = std::cmp::min(256, e) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let smem = (e * 4 + topk * 4 + topk * 4) as u32;
            let mut tw_ptr = *d_tw.device_ptr() as u64;
            let mut ti_ptr = *d_ti.device_ptr() as u64;
            let mut gate_ptr = *d_gate.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut tw_ptr as *mut _ as *mut _,
                &mut ti_ptr as *mut _ as *mut _,
                &mut gate_ptr as *mut _ as *mut _,
                &mut e_i32 as *mut _ as *mut _,
                &mut topk_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (m as u32, 1, 1), (threads, 1, 1), smem, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        // Check IDs match exactly
        let actual_ids = ctx.download_i32(&d_ti);
        let expected_ids = data.expected_bytes("topk_ids");
        assert_exact_i32(&actual_ids, &expected_ids, "sigmoid_topk_ids");

        // Check weights within tolerance
        let actual_wts = ctx.download_f32(&d_tw);
        let expected_wts = data.expected_bytes("topk_weights");
        assert_close_f32(&actual_wts, &expected_wts, data.meta.atol, data.meta.rtol, "sigmoid_topk_weights");
    }

    #[test]
    fn test_softmax_topk_m128_e512_k10() {
        let data = TestData::load("softmax_topk_m128_e512_k10");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("softmax_topk_kernel");

        let m = 128usize;
        let e = 512usize;
        let topk = 10usize;

        let d_gate = ctx.upload_f32(&data.input_bytes("gate"));
        let d_tw = ctx.alloc_f32(m * topk);
        let d_ti = ctx.alloc_i32(m * topk);

        let mut e_i32 = e as i32;
        let mut topk_i32 = topk as i32;

        unsafe {
            let threads = std::cmp::min(256, e) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let smem = (e * 4 + topk * 4 + topk * 4) as u32;
            let mut tw_ptr = *d_tw.device_ptr() as u64;
            let mut ti_ptr = *d_ti.device_ptr() as u64;
            let mut gate_ptr = *d_gate.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut tw_ptr as *mut _ as *mut _,
                &mut ti_ptr as *mut _ as *mut _,
                &mut gate_ptr as *mut _ as *mut _,
                &mut e_i32 as *mut _ as *mut _,
                &mut topk_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (m as u32, 1, 1), (threads, 1, 1), smem, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual_ids = ctx.download_i32(&d_ti);
        let expected_ids = data.expected_bytes("topk_ids");
        assert_exact_i32(&actual_ids, &expected_ids, "softmax_topk_ids");

        let actual_wts = ctx.download_f32(&d_tw);
        let expected_wts = data.expected_bytes("topk_weights");
        assert_close_f32(&actual_wts, &expected_wts, data.meta.atol, data.meta.rtol, "softmax_topk_weights");
    }

    #[test]
    fn test_transpose_3d_f32() {
        let data = TestData::load("transpose_3d_f32_32_500_128");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("transpose_3d_021_kernel");

        let a = data.input_shape("x")[0]; // 32
        let b = data.input_shape("x")[1]; // 500
        let c = data.input_shape("x")[2]; // 128
        let n = a * b * c;

        let d_x = ctx.upload_f32(&data.input_bytes("x"));
        let d_out = ctx.alloc_f32(n);

        let mut a_i32 = a as i32;
        let mut b_i32 = b as i32;
        let mut c_i32 = c as i32;

        unsafe {
            // Kernel uses grid=(B, A), each block copies C elements
            let threads = std::cmp::min(256, c as u32);
            let mut out_ptr = *d_out.device_ptr() as u64;
            let mut x_ptr = *d_x.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut x_ptr as *mut _ as *mut _,
                &mut a_i32 as *mut _ as *mut _,
                &mut b_i32 as *mut _ as *mut _,
                &mut c_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (b as u32, a as u32, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_f32(&d_out);
        let expected = data.expected_bytes("out");
        // Transpose is exact — data movement only
        assert_eq!(actual, expected, "transpose_3d_f32: byte-level mismatch");
        eprintln!("  transpose_3d_f32 PASS ({} elements, exact match)", n);
    }

    #[test]
    fn test_transpose_3d_bf16() {
        let data = TestData::load("transpose_3d_bf16_32_500_128");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("transpose_3d_021_bf16_kernel");

        let a = data.input_shape("x")[0];
        let b = data.input_shape("x")[1];
        let c = data.input_shape("x")[2];
        let n = a * b * c;

        let d_x = ctx.upload_bf16(&data.input_bytes("x"));
        let d_out = ctx.alloc_bf16(n);

        let mut a_i32 = a as i32;
        let mut b_i32 = b as i32;
        let mut c_i32 = c as i32;

        unsafe {
            // Kernel uses grid=(B, A), each block copies C elements
            let threads = std::cmp::min(256, c as u32);
            let mut out_ptr = *d_out.device_ptr() as u64;
            let mut x_ptr = *d_x.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut x_ptr as *mut _ as *mut _,
                &mut a_i32 as *mut _ as *mut _,
                &mut b_i32 as *mut _ as *mut _,
                &mut c_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (b as u32, a as u32, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_bf16(&d_out);
        let expected = data.expected_bytes("out");
        assert_eq!(actual, expected, "transpose_3d_bf16: byte-level mismatch");
        eprintln!("  transpose_3d_bf16 PASS ({} elements, exact match)", n);
    }

    #[test]
    fn test_sigmoid_mul_m256_n2048() {
        let data = TestData::load("sigmoid_mul_m256_n2048");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("sigmoid_mul_kernel");

        let m = data.input_shape("attn")[0];
        let n = data.input_shape("attn")[1];
        let out_n = m * n;

        let d_attn = ctx.upload_bf16(&data.input_bytes("attn"));
        let d_gate = ctx.upload_bf16(&data.input_bytes("gate"));
        let d_out = ctx.alloc_bf16(out_n);

        let mut n_i32 = n as i32;

        unsafe {
            let threads = std::cmp::min(1024, n) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let mut out_ptr = *d_out.device_ptr() as u64;
            let mut attn_ptr = *d_attn.device_ptr() as u64;
            let mut gate_ptr = *d_gate.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut attn_ptr as *mut _ as *mut _,
                &mut gate_ptr as *mut _ as *mut _,
                &mut n_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (m as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_bf16(&d_out);
        let expected = data.expected_bytes("out");
        assert_close_bf16(&actual, &expected, data.meta.atol, data.meta.rtol, "sigmoid_mul_m256_n2048");
    }

    #[test]
    fn test_rope_m256_partial25() {
        let data = TestData::load("rope_m256_h128_partial25");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("rope_batched_kernel");

        let m = data.input_shape("q")[0];        // 256
        let num_q = data.input_shape("q")[1];     // 16
        let num_kv = data.input_shape("k")[1];    // 4
        let hd = data.input_shape("q")[2];        // 128
        let half_dim = data.input_shape("cos_cache")[1]; // from partial rotary

        let d_q = ctx.upload_bf16(&data.input_bytes("q"));
        let d_k = ctx.upload_bf16(&data.input_bytes("k"));
        let d_pos = ctx.upload_i32(&data.input_bytes("positions"));
        let d_cos = ctx.upload_f32(&data.input_bytes("cos_cache"));
        let d_sin = ctx.upload_f32(&data.input_bytes("sin_cache"));

        let mut num_q_i32 = num_q as i32;
        let mut num_kv_i32 = num_kv as i32;
        let mut hd_i32 = hd as i32;
        let mut half_dim_i32 = half_dim as i32;

        unsafe {
            let threads = std::cmp::min(512, half_dim) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let threads = if threads == 0 { 32 } else { threads };
            let mut q_ptr = *d_q.device_ptr() as u64;
            let mut k_ptr = *d_k.device_ptr() as u64;
            let mut pos_ptr = *d_pos.device_ptr() as u64;
            let mut cos_ptr = *d_cos.device_ptr() as u64;
            let mut sin_ptr = *d_sin.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut q_ptr as *mut _ as *mut _,
                &mut k_ptr as *mut _ as *mut _,
                &mut pos_ptr as *mut _ as *mut _,
                &mut cos_ptr as *mut _ as *mut _,
                &mut sin_ptr as *mut _ as *mut _,
                &mut num_q_i32 as *mut _ as *mut _,
                &mut num_kv_i32 as *mut _ as *mut _,
                &mut hd_i32 as *mut _ as *mut _,
                &mut half_dim_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (m as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual_q = ctx.download_bf16(&d_q);
        let expected_q = data.expected_bytes("q_out");
        assert_close_bf16(&actual_q, &expected_q, data.meta.atol, data.meta.rtol, "rope_q_partial25");

        let actual_k = ctx.download_bf16(&d_k);
        let expected_k = data.expected_bytes("k_out");
        assert_close_bf16(&actual_k, &expected_k, data.meta.atol, data.meta.rtol, "rope_k_partial25");
    }

    #[test]
    fn test_fused_add_rmsnorm_m128_h2048() {
        let data = TestData::load("fused_add_rmsnorm_m128_h2048");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("fused_add_rmsnorm_batched_kernel");

        let m = data.input_shape("residual")[0];
        let d = data.input_shape("residual")[1];
        let n = m * d;

        let d_res = ctx.upload_bf16(&data.input_bytes("residual"));
        let d_x = ctx.upload_bf16(&data.input_bytes("x"));
        let d_w = ctx.upload_bf16(&data.input_bytes("weight"));
        let d_out = ctx.alloc_bf16(n);

        let mut eps: f32 = 1e-6;
        let mut d_i32 = d as i32;

        unsafe {
            let threads = std::cmp::min(1024, d) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let smem = threads * 4;
            let mut res_ptr = *d_res.device_ptr() as u64;
            let mut out_ptr = *d_out.device_ptr() as u64;
            let mut x_ptr = *d_x.device_ptr() as u64;
            let mut w_ptr = *d_w.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut res_ptr as *mut _ as *mut _,
                &mut out_ptr as *mut _ as *mut _,
                &mut x_ptr as *mut _ as *mut _,
                &mut w_ptr as *mut _ as *mut _,
                &mut d_i32 as *mut _ as *mut _,
                &mut eps as *mut f32 as *mut _,
            ];
            launch(kernel, (m as u32, 1, 1), (threads, 1, 1), smem, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        // Check updated residual
        let actual_res = ctx.download_bf16(&d_res);
        let expected_res = data.expected_bytes("residual_out");
        assert_close_bf16(&actual_res, &expected_res, data.meta.atol, data.meta.rtol, "fused_add_rmsnorm_residual");

        // Check normed output
        let actual_out = ctx.download_bf16(&d_out);
        let expected_out = data.expected_bytes("out");
        assert_close_bf16(&actual_out, &expected_out, data.meta.atol, data.meta.rtol, "fused_add_rmsnorm_out");
    }

    // ── Tier 2: GQA Attention ───────────────────────────────────────────

    #[test]
    fn test_gqa_attention_m64() {
        let data = TestData::load("gqa_attention_m64_h16_kv4_d128");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("flash_attn_tiled_kernel");

        let m = 64usize;
        let num_q = 16usize;
        let num_kv = 4usize;
        let hd = 128usize;
        let mut scale: f32 = 1.0 / (hd as f32).sqrt();
        let kv_stride = (num_kv * hd) as i32;

        let d_q = ctx.upload_bf16(&data.input_bytes("q"));
        let d_k = ctx.upload_bf16(&data.input_bytes("k"));
        let d_v = ctx.upload_bf16(&data.input_bytes("v"));
        let d_out = ctx.alloc_bf16(m * num_q * hd);

        let mut m_i32 = m as i32;
        let mut nq_i32 = num_q as i32;
        let mut nkv_i32 = num_kv as i32;
        let mut hd_i32 = hd as i32;
        let mut start_pos_i32: i32 = 0;
        let mut kv_stride_i32 = kv_stride;

        unsafe {
            // FA_BR=16, 1 warp (32 threads)
            let br = 16u32;
            let bc = 64u32;
            let blocks_m = ((m as u32) + br - 1) / br;
            // smem: s_q(BR*hd*2) + s_k(BC*hd*2) + s_v(BC*hd*2) + s_scores(BR*BC*4) + s_p(BR*BC*2) + s_o_tmp(BR*16*4)
            let smem = (br * hd as u32 * 2 + bc * hd as u32 * 2 + bc * hd as u32 * 2
                       + br * bc * 4 + br * bc * 2 + br * 16 * 4) as u32;

            let mut out_ptr = *d_out.device_ptr() as u64;
            let mut q_ptr = *d_q.device_ptr() as u64;
            let mut k_cache_ptr: u64 = 0;  // null = no FP8 cache
            let mut v_cache_ptr: u64 = 0;
            let mut k_cur_ptr = *d_k.device_ptr() as u64;
            let mut v_cur_ptr = *d_v.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut q_ptr as *mut _ as *mut _,
                &mut k_cache_ptr as *mut _ as *mut _,
                &mut v_cache_ptr as *mut _ as *mut _,
                &mut k_cur_ptr as *mut _ as *mut _,
                &mut v_cur_ptr as *mut _ as *mut _,
                &mut m_i32 as *mut _ as *mut _,
                &mut nq_i32 as *mut _ as *mut _,
                &mut nkv_i32 as *mut _ as *mut _,
                &mut hd_i32 as *mut _ as *mut _,
                &mut scale as *mut _ as *mut _,
                &mut start_pos_i32 as *mut _ as *mut _,
                &mut kv_stride_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (blocks_m, num_q as u32, 1), (32, 1, 1), smem, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_bf16(&d_out);
        let expected = data.expected_bytes("out");
        assert_close_bf16(&actual, &expected, data.meta.atol, data.meta.rtol, "gqa_attention_m64");
    }

    #[test]
    fn test_gqa_cross_chunk_attention() {
        let data = TestData::load("gqa_cross_chunk_m32_cached32_h16_kv4_d128");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("flash_attn_tiled_kernel");

        let new_len = 32usize;
        let cached_len = 32usize;
        let num_q = 16usize;
        let num_kv = 4usize;
        let hd = 128usize;
        let mut scale: f32 = 1.0 / (hd as f32).sqrt();
        let kv_stride = (num_kv * hd) as i32;

        let d_q = ctx.upload_bf16(&data.input_bytes("q_new"));
        let d_k_cur = ctx.upload_bf16(&data.input_bytes("k_new"));
        let d_v_cur = ctx.upload_bf16(&data.input_bytes("v_new"));
        // FP8 cache -- need to put in a [max_seq, kv_stride] layout
        let max_seq = cached_len + new_len;
        let d_k_cache = ctx.alloc_u8(max_seq * num_kv * hd);
        let d_v_cache = ctx.alloc_u8(max_seq * num_kv * hd);
        // Upload FP8 data to positions [0..cached_len]
        let k_fp8_data = data.input_bytes("k_cache_fp8");
        let v_fp8_data = data.input_bytes("v_cache_fp8");
        // Copy to the start of the cache
        unsafe {
            cuda_sys::lib().cuMemcpyHtoD_v2(
                *d_k_cache.device_ptr(), k_fp8_data.as_ptr() as *const _, k_fp8_data.len());
            cuda_sys::lib().cuMemcpyHtoD_v2(
                *d_v_cache.device_ptr(), v_fp8_data.as_ptr() as *const _, v_fp8_data.len());
        }

        let d_out = ctx.alloc_bf16(new_len * num_q * hd);

        let mut m_i32 = new_len as i32;
        let mut nq_i32 = num_q as i32;
        let mut nkv_i32 = num_kv as i32;
        let mut hd_i32 = hd as i32;
        let mut start_pos_i32 = cached_len as i32;
        let mut kv_stride_i32 = kv_stride;

        unsafe {
            let br = 16u32;
            let bc = 64u32;
            let blocks_m = ((new_len as u32) + br - 1) / br;
            let smem = (br * hd as u32 * 2 + bc * hd as u32 * 2 + bc * hd as u32 * 2
                       + br * bc * 4 + br * bc * 2 + br * 16 * 4) as u32;

            let mut out_ptr = *d_out.device_ptr() as u64;
            let mut q_ptr = *d_q.device_ptr() as u64;
            let mut k_cache_ptr = *d_k_cache.device_ptr() as u64;
            let mut v_cache_ptr = *d_v_cache.device_ptr() as u64;
            let mut k_cur_ptr = *d_k_cur.device_ptr() as u64;
            let mut v_cur_ptr = *d_v_cur.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut q_ptr as *mut _ as *mut _,
                &mut k_cache_ptr as *mut _ as *mut _,
                &mut v_cache_ptr as *mut _ as *mut _,
                &mut k_cur_ptr as *mut _ as *mut _,
                &mut v_cur_ptr as *mut _ as *mut _,
                &mut m_i32 as *mut _ as *mut _,
                &mut nq_i32 as *mut _ as *mut _,
                &mut nkv_i32 as *mut _ as *mut _,
                &mut hd_i32 as *mut _ as *mut _,
                &mut scale as *mut _ as *mut _,
                &mut start_pos_i32 as *mut _ as *mut _,
                &mut kv_stride_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (blocks_m, num_q as u32, 1), (32, 1, 1), smem, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_bf16(&d_out);
        let expected = data.expected_bytes("out_new");
        assert_close_bf16(&actual, &expected, data.meta.atol, data.meta.rtol, "gqa_cross_chunk");
    }

    // ── Tier 2: FP8 KV Cache Round-trip ──────────────────────────────

    #[test]
    fn test_fp8_kv_roundtrip() {
        let data = TestData::load("fp8_kv_roundtrip_m256_d512");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("kv_cache_append_fp8_kernel");

        let m = data.input_shape("k")[0];
        let d = data.input_shape("k")[1];

        let d_k = ctx.upload_bf16(&data.input_bytes("k"));
        let d_v = ctx.upload_bf16(&data.input_bytes("v"));

        // Allocate FP8 cache buffers (max_seq=m for simplicity)
        let max_seq = m;
        let d_k_cache = ctx.alloc_u8(max_seq * d);
        let d_v_cache = ctx.alloc_u8(max_seq * d);

        let mut m_i32 = m as i32;
        let mut d_i32 = d as i32;
        let mut max_seq_i32 = max_seq as i32;
        let mut start_pos_i32: i32 = 0;

        // Write BF16 K/V to FP8 cache
        unsafe {
            let threads = std::cmp::min(256, d) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let mut kc_ptr = *d_k_cache.device_ptr() as u64;
            let mut vc_ptr = *d_v_cache.device_ptr() as u64;
            let mut k_ptr = *d_k.device_ptr() as u64;
            let mut v_ptr = *d_v.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut kc_ptr as *mut _ as *mut _,
                &mut vc_ptr as *mut _ as *mut _,
                &mut k_ptr as *mut _ as *mut _,
                &mut v_ptr as *mut _ as *mut _,
                &mut m_i32 as *mut _ as *mut _,
                &mut d_i32 as *mut _ as *mut _,
                &mut max_seq_i32 as *mut _ as *mut _,
                &mut start_pos_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (m as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        // Read FP8 back, dequantize to BF16 manually, and compare
        let k_fp8_raw = ctx.download_u8(&d_k_cache);
        let v_fp8_raw = ctx.download_u8(&d_v_cache);

        // Dequantize FP8 E4M3 to BF16 by reading the expected roundtrip values
        let expected_k = data.expected_bytes("k_roundtrip");
        let expected_v = data.expected_bytes("v_roundtrip");

        // Convert FP8 back to BF16 values for comparison
        // FP8 E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
        let actual_k: Vec<u8> = k_fp8_raw.iter().flat_map(|&fp8| {
            let val = fp8_e4m3_to_f32(fp8);
            let bf16 = half::bf16::from_f32(val);
            bf16.to_bits().to_le_bytes().to_vec()
        }).collect();
        let actual_v: Vec<u8> = v_fp8_raw.iter().flat_map(|&fp8| {
            let val = fp8_e4m3_to_f32(fp8);
            let bf16 = half::bf16::from_f32(val);
            bf16.to_bits().to_le_bytes().to_vec()
        }).collect();

        assert_close_bf16(&actual_k, &expected_k, data.meta.atol, data.meta.rtol, "fp8_kv_roundtrip_k");
        assert_close_bf16(&actual_v, &expected_v, data.meta.atol, data.meta.rtol, "fp8_kv_roundtrip_v");
    }

    /// Dequantize FP8 E4M3 byte to f32.
    fn fp8_e4m3_to_f32(bits: u8) -> f32 {
        let sign = (bits >> 7) & 1;
        let exp = (bits >> 3) & 0xF;
        let mant = bits & 0x7;

        if exp == 0 && mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }

        let (exponent, mantissa) = if exp == 0 {
            // Subnormal: E4M3 bias=7, subnormal exponent = 1-7 = -6
            (-6i32, mant as f32 / 8.0)
        } else {
            // Normal
            (exp as i32 - 7, 1.0 + mant as f32 / 8.0)
        };

        let val = mantissa * 2.0f32.powi(exponent);
        if sign == 1 { -val } else { val }
    }

    // ── Tier 2: LA L2 Norm ───────────────────────────────────────────

    #[test]
    fn test_la_l2norm() {
        let data = TestData::load("la_l2norm_m128_h32_d128");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("la_l2norm_per_head_kernel");

        let m = 128usize;
        let num_heads = 32usize;
        let dim = 128usize;
        let mut scale: f32 = (dim as f32).sqrt();

        let d_x = ctx.upload_f32(&data.input_bytes("x"));

        let mut num_heads_i32 = num_heads as i32;
        let mut dim_i32 = dim as i32;

        unsafe {
            let threads = std::cmp::min(256, dim) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let smem = threads * 4;
            let mut x_ptr = *d_x.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut x_ptr as *mut _ as *mut _,
                &mut scale as *mut f32 as *mut _,
                &mut num_heads_i32 as *mut _ as *mut _,
                &mut dim_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (m as u32, num_heads as u32, 1), (threads, 1, 1), smem, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_f32(&d_x);
        let expected = data.expected_bytes("out");
        assert_close_f32(&actual, &expected, data.meta.atol as f64, data.meta.rtol as f64, "la_l2norm");
    }

    // ── Tier 2: LA Gate/Beta ─────────────────────────────────────────

    #[test]
    fn test_la_gate_beta() {
        let data = TestData::load("la_gate_beta_m128_nv32");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("la_compute_gate_beta_kernel");

        let m = 128usize;
        let nv = 32usize;

        let d_b = ctx.upload_bf16(&data.input_bytes("b"));
        let d_a = ctx.upload_bf16(&data.input_bytes("a"));
        let d_alog = ctx.upload_f32(&data.input_bytes("A_log"));
        let d_dtbias = ctx.upload_f32(&data.input_bytes("dt_bias"));
        let d_beta = ctx.alloc_f32(m * nv);
        let d_gate = ctx.alloc_f32(m * nv);

        let mut nv_i32 = nv as i32;

        unsafe {
            let threads = std::cmp::min(1024, nv) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let mut beta_ptr = *d_beta.device_ptr() as u64;
            let mut gate_ptr = *d_gate.device_ptr() as u64;
            let mut b_ptr = *d_b.device_ptr() as u64;
            let mut a_ptr = *d_a.device_ptr() as u64;
            let mut alog_ptr = *d_alog.device_ptr() as u64;
            let mut dtbias_ptr = *d_dtbias.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut beta_ptr as *mut _ as *mut _,
                &mut gate_ptr as *mut _ as *mut _,
                &mut b_ptr as *mut _ as *mut _,
                &mut a_ptr as *mut _ as *mut _,
                &mut alog_ptr as *mut _ as *mut _,
                &mut dtbias_ptr as *mut _ as *mut _,
                &mut nv_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (m as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual_beta = ctx.download_f32(&d_beta);
        let expected_beta = data.expected_bytes("beta");
        assert_close_f32(&actual_beta, &expected_beta, data.meta.atol as f64, data.meta.rtol as f64, "la_gate_beta_beta");

        let actual_gate = ctx.download_f32(&d_gate);
        let expected_gate = data.expected_bytes("gate");
        assert_close_f32(&actual_gate, &expected_gate, data.meta.atol as f64, data.meta.rtol as f64, "la_gate_beta_gate");
    }

    // ── Tier 2: LA Gated RMSNorm ─────────────────────────────────────

    #[test]
    fn test_la_gated_rmsnorm() {
        let data = TestData::load("la_gated_rmsnorm_m64_nv32_dv128");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("la_gated_rmsnorm_kernel");

        let m = 64usize;
        let nv = 32usize;
        let dv = 128usize;
        let eps: f32 = 1e-6;

        let d_x = ctx.upload_f32(&data.input_bytes("x"));
        let d_gate = ctx.upload_bf16(&data.input_bytes("gate"));
        let d_weight = ctx.upload_bf16(&data.input_bytes("weight"));
        let d_out = ctx.alloc_bf16(m * nv * dv);

        let mut nv_i32 = nv as i32;
        let mut dv_i32 = dv as i32;
        let mut eps_f32 = eps;

        unsafe {
            let threads = std::cmp::min(256, dv) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let smem = threads * 4;
            let mut out_ptr = *d_out.device_ptr() as u64;
            let mut x_ptr = *d_x.device_ptr() as u64;
            let mut gate_ptr = *d_gate.device_ptr() as u64;
            let mut w_ptr = *d_weight.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut x_ptr as *mut _ as *mut _,
                &mut gate_ptr as *mut _ as *mut _,
                &mut w_ptr as *mut _ as *mut _,
                &mut nv_i32 as *mut _ as *mut _,
                &mut dv_i32 as *mut _ as *mut _,
                &mut eps_f32 as *mut _ as *mut _,
            ];
            launch(kernel, (m as u32, nv as u32, 1), (threads, 1, 1), smem, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_bf16(&d_out);
        let expected = data.expected_bytes("out");
        assert_close_bf16(&actual, &expected, data.meta.atol, data.meta.rtol, "la_gated_rmsnorm");
    }

    // ── Tier 2: LA Conv1d + SiLU ─────────────────────────────────────

    #[test]
    fn test_la_conv1d() {
        let data = TestData::load("la_conv1d_m64_cd384_k4");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("la_depthwise_conv1d_silu_kernel");

        let m = 64usize;
        let conv_dim = 384usize;
        let kernel_dim = 4usize;

        let d_input = ctx.upload_bf16(&data.input_bytes("x"));
        let d_weight = ctx.upload_f32(&data.input_bytes("weight"));
        let d_state = ctx.upload_f32(&data.input_bytes("conv_state"));
        let d_output = ctx.alloc_f32(conv_dim * m);

        let mut m_i32 = m as i32;
        let mut cd_i32 = conv_dim as i32;
        let mut kd_i32 = kernel_dim as i32;

        unsafe {
            let threads = std::cmp::min(256, m) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let mut out_ptr = *d_output.device_ptr() as u64;
            let mut state_ptr = *d_state.device_ptr() as u64;
            let mut input_ptr = *d_input.device_ptr() as u64;
            let mut weight_ptr = *d_weight.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut state_ptr as *mut _ as *mut _,
                &mut input_ptr as *mut _ as *mut _,
                &mut weight_ptr as *mut _ as *mut _,
                &mut m_i32 as *mut _ as *mut _,
                &mut cd_i32 as *mut _ as *mut _,
                &mut kd_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (conv_dim as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual_out = ctx.download_f32(&d_output);
        let expected_out = data.expected_bytes("out");
        assert_close_f32(&actual_out, &expected_out, data.meta.atol as f64, data.meta.rtol as f64, "la_conv1d_output");

        let actual_state = ctx.download_f32(&d_state);
        let expected_state = data.expected_bytes("state_out");
        assert_close_f32(&actual_state, &expected_state, data.meta.atol as f64, data.meta.rtol as f64, "la_conv1d_state");
    }

    // ── Tier 2: LA Uninterleave QKVZ ─────────────────────────────────

    #[test]
    fn test_la_uninterleave_qkvz() {
        let data = TestData::load("la_uninterleave_qkvz_m32_nk8_dk128_hr4_dv128");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("la_uninterleave_qkvz_kernel");

        let m = 32usize;
        let nk = 8usize;
        let dk = 128usize;
        let hr = 4usize;
        let dv = 128usize;
        let nv = nk * hr;

        let d_qkvz = ctx.upload_bf16(&data.input_bytes("qkvz"));
        let d_q = ctx.alloc_bf16(m * nk * dk);
        let d_k = ctx.alloc_bf16(m * nk * dk);
        let d_v = ctx.alloc_bf16(m * nv * dv);
        let d_z = ctx.alloc_bf16(m * nv * dv);

        let mut nk_i32 = nk as i32;
        let mut dk_i32 = dk as i32;
        let mut hr_i32 = hr as i32;
        let mut dv_i32 = dv as i32;

        let group_dim = 2 * dk + 2 * hr * dv;
        let total_dim = nk * group_dim;

        unsafe {
            let threads = std::cmp::min(1024, total_dim) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let mut q_ptr = *d_q.device_ptr() as u64;
            let mut k_ptr = *d_k.device_ptr() as u64;
            let mut v_ptr = *d_v.device_ptr() as u64;
            let mut z_ptr = *d_z.device_ptr() as u64;
            let mut qkvz_ptr = *d_qkvz.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut q_ptr as *mut _ as *mut _,
                &mut k_ptr as *mut _ as *mut _,
                &mut v_ptr as *mut _ as *mut _,
                &mut z_ptr as *mut _ as *mut _,
                &mut qkvz_ptr as *mut _ as *mut _,
                &mut nk_i32 as *mut _ as *mut _,
                &mut dk_i32 as *mut _ as *mut _,
                &mut hr_i32 as *mut _ as *mut _,
                &mut dv_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (m as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual_q = ctx.download_bf16(&d_q);
        let expected_q = data.expected_bytes("q");
        assert_eq!(actual_q, expected_q, "la_uninterleave q: byte-level mismatch");
        eprintln!("  la_uninterleave q PASS (exact)");

        let actual_k = ctx.download_bf16(&d_k);
        let expected_k = data.expected_bytes("k");
        assert_eq!(actual_k, expected_k, "la_uninterleave k: byte-level mismatch");
        eprintln!("  la_uninterleave k PASS (exact)");

        let actual_v = ctx.download_bf16(&d_v);
        let expected_v = data.expected_bytes("v");
        assert_eq!(actual_v, expected_v, "la_uninterleave v: byte-level mismatch");
        eprintln!("  la_uninterleave v PASS (exact)");

        let actual_z = ctx.download_bf16(&d_z);
        let expected_z = data.expected_bytes("z");
        assert_eq!(actual_z, expected_z, "la_uninterleave z: byte-level mismatch");
        eprintln!("  la_uninterleave z PASS (exact)");
    }

    // ── Tier 2: Pipeline kernels ─────────────────────────────────────────

    #[test]
    fn test_embedding_m64() {
        let data = TestData::load("embedding_m64_d2048_v1000");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("embedding_batched_kernel");

        let m = 64usize;
        let d = 2048usize;
        let vocab = 1000usize;

        let d_table = ctx.upload_bf16(&data.input_bytes("table"));
        let d_ids = ctx.upload_i32(&data.input_bytes("token_ids"));
        let d_out = ctx.alloc_bf16(m * d);

        let mut d_i32 = d as i32;

        unsafe {
            let threads = std::cmp::min(1024, d) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let mut out_ptr = *d_out.device_ptr() as u64;
            let mut table_ptr = *d_table.device_ptr() as u64;
            let mut ids_ptr = *d_ids.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut table_ptr as *mut _ as *mut _,
                &mut ids_ptr as *mut _ as *mut _,
                &mut d_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (m as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_bf16(&d_out);
        let expected = data.expected_bytes("out");
        assert_eq!(actual, expected, "embedding: byte-level mismatch");
        eprintln!("  embedding PASS ({} tokens, exact match)", m);
    }

    #[test]
    fn test_gated_q_split_m256() {
        let data = TestData::load("gated_q_split_m256_h16_d128");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("gated_q_split_kernel");

        let m = 256usize;
        let h = 16usize;
        let d = 128usize;

        let d_qg = ctx.upload_bf16(&data.input_bytes("qg"));
        let d_q = ctx.alloc_bf16(m * h * d);
        let d_gate = ctx.alloc_bf16(m * h * d);

        let mut h_i32 = h as i32;
        let mut d_i32 = d as i32;

        unsafe {
            let threads = std::cmp::min(256, d) as u32;
            let mut q_ptr = *d_q.device_ptr() as u64;
            let mut g_ptr = *d_gate.device_ptr() as u64;
            let mut qg_ptr = *d_qg.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut q_ptr as *mut _ as *mut _,
                &mut g_ptr as *mut _ as *mut _,
                &mut qg_ptr as *mut _ as *mut _,
                &mut h_i32 as *mut _ as *mut _,
                &mut d_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (m as u32, h as u32, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual_q = ctx.download_bf16(&d_q);
        let expected_q = data.expected_bytes("q");
        assert_eq!(actual_q, expected_q, "gated_q_split q: byte-level mismatch");
        eprintln!("  gated_q_split q PASS (exact)");

        let actual_g = ctx.download_bf16(&d_gate);
        let expected_g = data.expected_bytes("gate");
        assert_eq!(actual_g, expected_g, "gated_q_split gate: byte-level mismatch");
        eprintln!("  gated_q_split gate PASS (exact)");
    }

    #[test]
    fn test_concat_3_bf16_m500() {
        let data = TestData::load("concat_3_bf16_m500");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("concat_3_bf16_kernel");

        let m = 500usize;
        let d1 = 128usize;
        let d2 = 128usize;
        let d3 = 128usize;
        let total = d1 + d2 + d3;

        let d_a = ctx.upload_bf16(&data.input_bytes("a"));
        let d_b = ctx.upload_bf16(&data.input_bytes("b"));
        let d_c = ctx.upload_bf16(&data.input_bytes("c"));
        let d_out = ctx.alloc_bf16(m * total);

        let mut d1_i32 = d1 as i32;
        let mut d2_i32 = d2 as i32;
        let mut d3_i32 = d3 as i32;

        unsafe {
            let threads = std::cmp::min(256, total) as u32;
            let mut out_ptr = *d_out.device_ptr() as u64;
            let mut a_ptr = *d_a.device_ptr() as u64;
            let mut b_ptr = *d_b.device_ptr() as u64;
            let mut c_ptr = *d_c.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut a_ptr as *mut _ as *mut _,
                &mut b_ptr as *mut _ as *mut _,
                &mut c_ptr as *mut _ as *mut _,
                &mut d1_i32 as *mut _ as *mut _,
                &mut d2_i32 as *mut _ as *mut _,
                &mut d3_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (m as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_bf16(&d_out);
        let expected = data.expected_bytes("out");
        assert_eq!(actual, expected, "concat_3_bf16: byte-level mismatch");
        eprintln!("  concat_3_bf16 PASS ({} tokens, exact match)", m);
    }

    // ── Tier 2: LA complex operations ─────────────────────────────────────

    #[test]
    fn test_la_triangular_solve() {
        let data = TestData::load("la_tri_solve_nv4_nc2_cs8_d16");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("la_triangular_solve_kernel");

        let nv = 4usize;
        let num_chunks = 2usize;
        let cs = 8usize;
        let dim = 16usize;

        // x starts as b, gets modified in-place
        let d_x = ctx.upload_f32(&data.input_bytes("b"));
        let d_a = ctx.upload_f32(&data.input_bytes("A"));

        let mut num_chunks_i32 = num_chunks as i32;
        let mut cs_i32 = cs as i32;
        let mut dim_i32 = dim as i32;

        unsafe {
            let threads = std::cmp::min(256, dim) as u32;
            let threads = ((threads + 31) / 32) * 32;
            let mut x_ptr = *d_x.device_ptr() as u64;
            let mut a_ptr = *d_a.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut x_ptr as *mut _ as *mut _,
                &mut a_ptr as *mut _ as *mut _,
                &mut num_chunks_i32 as *mut _ as *mut _,
                &mut cs_i32 as *mut _ as *mut _,
                &mut dim_i32 as *mut _ as *mut _,
            ];
            // Grid: (nv, num_chunks, 1)
            launch(kernel, (nv as u32, num_chunks as u32, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_f32(&d_x);
        let expected = data.expected_bytes("x_out");
        assert_close_f32(&actual, &expected, data.meta.atol, data.meta.rtol, "la_triangular_solve");
    }

    #[test]
    fn test_la_chunk_recurrence() {
        // Tests all 3 chunk kernels: compute_v_new + chunk_output + state_update
        let data = TestData::load("la_chunk_recurrence_nv4_cs8_dk16_dv16");
        let ctx = GpuTestCtx::new();

        let nv = 4usize;
        let cs = 8usize;
        let dk = 16usize;
        let dv = 16usize;

        let d_q = ctx.upload_f32(&data.input_bytes("q"));
        let d_k = ctx.upload_f32(&data.input_bytes("k"));
        let d_v_corr = ctx.upload_f32(&data.input_bytes("v_corr"));
        let d_k_cumd = ctx.upload_f32(&data.input_bytes("k_cumd"));
        let d_g_cum = ctx.upload_f32(&data.input_bytes("g_cum"));
        let d_state = ctx.upload_f32(&data.input_bytes("state"));

        let d_v_new = ctx.alloc_f32(nv * cs * dv);
        let d_output = ctx.alloc_f32(nv * cs * dv);

        let mut nv_i32 = nv as i32;
        let mut cs_i32 = cs as i32;
        let mut dk_i32 = dk as i32;
        let mut dv_i32 = dv as i32;

        // Step 1: compute_v_new
        let k_vn = ctx.get_kernel("la_compute_v_new_kernel");
        unsafe {
            let threads = std::cmp::min(256, dv) as u32;
            let mut vn_ptr = *d_v_new.device_ptr() as u64;
            let mut vc_ptr = *d_v_corr.device_ptr() as u64;
            let mut kc_ptr = *d_k_cumd.device_ptr() as u64;
            let mut st_ptr = *d_state.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut vn_ptr as *mut _ as *mut _,
                &mut vc_ptr as *mut _ as *mut _,
                &mut kc_ptr as *mut _ as *mut _,
                &mut st_ptr as *mut _ as *mut _,
                &mut nv_i32 as *mut _ as *mut _,
                &mut cs_i32 as *mut _ as *mut _,
                &mut dk_i32 as *mut _ as *mut _,
                &mut dv_i32 as *mut _ as *mut _,
            ];
            launch(k_vn, (nv as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual_vn = ctx.download_f32(&d_v_new);
        let expected_vn = data.expected_bytes("v_new");
        assert_close_f32(&actual_vn, &expected_vn, data.meta.atol, data.meta.rtol, "la_compute_v_new");

        // Step 2: chunk_output
        let k_out = ctx.get_kernel("la_chunk_output_kernel");
        unsafe {
            let threads = std::cmp::min(256, dv) as u32;
            let mut out_ptr = *d_output.device_ptr() as u64;
            let mut q_ptr = *d_q.device_ptr() as u64;
            let mut k_ptr = *d_k.device_ptr() as u64;
            let mut vn_ptr = *d_v_new.device_ptr() as u64;
            let mut gc_ptr = *d_g_cum.device_ptr() as u64;
            let mut st_ptr = *d_state.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut q_ptr as *mut _ as *mut _,
                &mut k_ptr as *mut _ as *mut _,
                &mut vn_ptr as *mut _ as *mut _,
                &mut gc_ptr as *mut _ as *mut _,
                &mut st_ptr as *mut _ as *mut _,
                &mut nv_i32 as *mut _ as *mut _,
                &mut cs_i32 as *mut _ as *mut _,
                &mut dk_i32 as *mut _ as *mut _,
                &mut dv_i32 as *mut _ as *mut _,
            ];
            launch(k_out, (nv as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual_out = ctx.download_f32(&d_output);
        let expected_out = data.expected_bytes("output");
        assert_close_f32(&actual_out, &expected_out, data.meta.atol, data.meta.rtol, "la_chunk_output");

        // Step 3: state_update (modifies state in place)
        let k_su = ctx.get_kernel("la_state_update_kernel");
        unsafe {
            let threads = std::cmp::min(256, dv) as u32;
            let mut st_ptr = *d_state.device_ptr() as u64;
            let mut k_ptr = *d_k.device_ptr() as u64;
            let mut vn_ptr = *d_v_new.device_ptr() as u64;
            let mut gc_ptr = *d_g_cum.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut st_ptr as *mut _ as *mut _,
                &mut k_ptr as *mut _ as *mut _,
                &mut vn_ptr as *mut _ as *mut _,
                &mut gc_ptr as *mut _ as *mut _,
                &mut nv_i32 as *mut _ as *mut _,
                &mut cs_i32 as *mut _ as *mut _,
                &mut dk_i32 as *mut _ as *mut _,
                &mut dv_i32 as *mut _ as *mut _,
            ];
            launch(k_su, (nv as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual_state = ctx.download_f32(&d_state);
        let expected_state = data.expected_bytes("state_out");
        assert_close_f32(&actual_state, &expected_state, data.meta.atol, data.meta.rtol, "la_state_update");
    }

    #[test]
    fn test_moe_gather_sorted() {
        let data = TestData::load("moe_gather_scatter_m32_h64_k4");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("moe_gather_sorted_kernel");

        let m = 32usize;
        let hidden = 64usize;
        let sorted_ids_bytes = data.input_bytes("sorted_ids");
        let total_padded = sorted_ids_bytes.len() / 4;

        let d_src = ctx.upload_bf16(&data.input_bytes("src"));
        let d_sorted_ids = ctx.upload_i32(&sorted_ids_bytes);
        let d_out = ctx.alloc_bf16(total_padded * hidden);

        let mut dim_i32 = hidden as i32;
        let mut m_i32 = m as i32;

        unsafe {
            let threads = std::cmp::min(256, hidden) as u32;
            let mut out_ptr = *d_out.device_ptr() as u64;
            let mut src_ptr = *d_src.device_ptr() as u64;
            let mut ids_ptr = *d_sorted_ids.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut src_ptr as *mut _ as *mut _,
                &mut ids_ptr as *mut _ as *mut _,
                &mut dim_i32 as *mut _ as *mut _,
                &mut m_i32 as *mut _ as *mut _,
            ];
            launch(kernel, (total_padded as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_bf16(&d_out);
        let expected = data.expected_bytes("gather_out");
        assert_eq!(actual, expected, "moe_gather_sorted: byte-level mismatch");
        eprintln!("  moe_gather_sorted PASS ({} positions, exact match)", total_padded);
    }

    #[test]
    fn test_moe_scatter_fused() {
        let data = TestData::load("moe_gather_scatter_m32_h64_k4");
        let ctx = GpuTestCtx::new();
        let kernel = ctx.get_kernel("moe_scatter_fused_kernel");

        let m = 32usize;
        let hidden = 64usize;
        let sorted_ids_bytes = data.input_bytes("sorted_ids");
        let total_padded = sorted_ids_bytes.len() / 4;

        let d_expert_out = ctx.upload_bf16(&data.input_bytes("expert_out"));
        let d_sorted_ids = ctx.upload_i32(&sorted_ids_bytes);
        let d_accum = ctx.alloc_f32(m * hidden);

        let mut hidden_i32 = hidden as i32;
        let mut m_i32 = m as i32;
        let mut scale: f32 = 1.0;

        unsafe {
            let threads = std::cmp::min(256, hidden) as u32;
            let mut accum_ptr = *d_accum.device_ptr() as u64;
            let mut src_ptr = *d_expert_out.device_ptr() as u64;
            let mut ids_ptr = *d_sorted_ids.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut accum_ptr as *mut _ as *mut _,
                &mut src_ptr as *mut _ as *mut _,
                &mut ids_ptr as *mut _ as *mut _,
                &mut hidden_i32 as *mut _ as *mut _,
                &mut m_i32 as *mut _ as *mut _,
                &mut scale as *mut _ as *mut _,
            ];
            launch(kernel, (total_padded as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let actual = ctx.download_f32(&d_accum);
        let expected = data.expected_bytes("accum");
        assert_close_f32(&actual, &expected, data.meta.atol, data.meta.rtol, "moe_scatter_fused");
    }

    /// Synchronize the CUDA device (wait for all kernel launches to complete).
    fn cuda_sync() {
        unsafe {
            cuda_sys::lib().cuCtxSynchronize();
        }
    }
}

/// Find the vendored libkrasis_marlin.so built by build.rs.
/// Searches: 1) next to the Rust extension .so, 2) Cargo OUT_DIR, 3) common build paths.
fn find_marlin_so() -> Option<String> {
    // 1. Look in KRASIS_MARLIN_SO env var (set by build.rs or wrapper script)
    if let Ok(p) = std::env::var("KRASIS_MARLIN_SO") {
        if std::path::Path::new(&p).exists() { return Some(p); }
    }

    // 2. Look next to the current executable / shared library
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("libkrasis_marlin.so");
            if p.exists() { return Some(p.to_string_lossy().to_string()); }
        }
    }

    // 3. Search common Cargo build output directories
    let home = std::env::var("HOME").unwrap_or_default();
    for profile in &["release", "debug"] {
        // Walk the build directory to find the OUT_DIR
        let build_dir = format!("{}/Documents/Claude/krasisx/target/{}/build", home, profile);
        if let Ok(entries) = std::fs::read_dir(&build_dir) {
            for e in entries.flatten() {
                let p = e.path().join("out/libkrasis_marlin.so");
                if p.exists() { return Some(p.to_string_lossy().to_string()); }
            }
        }
    }

    // 4. Look in the maturin target directory
    for profile in &["release", "debug"] {
        let build_dir = format!("{}/Documents/Claude/krasisx/target/{}/build", home, profile);
        if let Ok(entries) = std::fs::read_dir(&build_dir) {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with("krasis-") {
                    let p = e.path().join("out/libkrasis_marlin.so");
                    if p.exists() { return Some(p.to_string_lossy().to_string()); }
                }
            }
        }
    }

    log::warn!("libkrasis_marlin.so not found — Marlin GEMM will be unavailable");
    None
}
