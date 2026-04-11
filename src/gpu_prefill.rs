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

use std::cell::Cell;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr};
use cudarc::driver::sys as cuda_sys;
use pyo3::prelude::*;

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

const MARLIN_PIPE_STAGES: usize = 4;
const MARLIN_MIN_THREAD_N: usize = 64;
const MARLIN_MIN_THREAD_K: usize = 64;
const MARLIN_MAX_BLOCKS_PER_SM: usize = 4;
const MARLIN_MAX_LOCK_SLOTS_PER_SM: usize = 4;

fn marlin_moe_scales_cache_size(
    thread_n: usize,
    thread_k: usize,
    prob_n: usize,
    prob_k: usize,
    num_bits: usize,
    group_size: isize,
    has_act_order: bool,
    is_k_full: bool,
) -> usize {
    let cache_scales_chunk = has_act_order && !is_k_full;
    let tb_n = thread_n;
    let tb_k = thread_k;
    let tb_groups = if group_size == -1 {
        1
    } else if group_size == 0 {
        tb_k.div_ceil(32)
    } else {
        tb_k.div_ceil(group_size as usize)
    };

    if cache_scales_chunk {
        let load_groups = (tb_groups * MARLIN_PIPE_STAGES * 2).max(32);
        load_groups * tb_n * 2
    } else {
        let _ = (prob_n, prob_k, num_bits);
        tb_groups * tb_n * 2 * MARLIN_PIPE_STAGES
    }
}

fn marlin_moe_kernel_cache_size(
    thread_m_blocks: usize,
    thread_n: usize,
    thread_k: usize,
    prob_n: usize,
    prob_k: usize,
    num_bits: usize,
    group_size: isize,
    has_act_order: bool,
    is_k_full: bool,
    has_zp: bool,
    is_zp_float: bool,
) -> usize {
    let pack_factor = 32 / num_bits;
    let tb_k = thread_k;
    let tb_n = thread_n;
    let tb_m = thread_m_blocks * 16;
    // MoE metadata ahead of sh_new matches the CUDA template layout:
    // - sh_block_sorted_ids: moe_block_size / 4 int4s
    // - sh_rd_block_sorted_ids: moe_block_size / 4 int4s
    // - sh_block_topk_weights plus pad: moe_block_size / 2 + moe_block_size int4s
    // Total = 2 * moe_block_size int4s = tb_m * 32 bytes.
    let sh_block_meta_size = tb_m * 32;
    let sh_a_size = MARLIN_PIPE_STAGES * (tb_m * tb_k) * 2;
    let sh_b_size = MARLIN_PIPE_STAGES * (tb_k * tb_n / pack_factor) * 4;
    let sh_red_size = tb_m * (tb_n + 8) * 2;
    let sh_bias_size = tb_n * 2;
    let mut tmp_size = sh_red_size.max(sh_b_size) + sh_bias_size;
    tmp_size = tmp_size.max(sh_b_size.max(sh_red_size));
    let sh_s_size = marlin_moe_scales_cache_size(
        thread_n,
        thread_k,
        prob_n,
        prob_k,
        num_bits,
        group_size,
        has_act_order,
        is_k_full,
    );
    let sh_g_idx_size = if has_act_order && !is_k_full {
        MARLIN_PIPE_STAGES * tb_k / 4
    } else {
        0
    };
    let sh_zp_size = if has_zp {
        if is_zp_float {
            sh_s_size
        } else if num_bits == 4 {
            sh_s_size / 4
        } else if num_bits == 8 {
            sh_s_size / 2
        } else {
            0
        }
    } else {
        0
    };
    tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size + sh_block_meta_size
}

fn fused_moe_fp32_reduce_floats(
    moe_block_size: usize,
    prob_n: usize,
    prob_k: usize,
    num_bits: usize,
    group_size: usize,
    sms: usize,
    max_shared_mem: usize,
) -> usize {
    let thread_m_blocks = moe_block_size.div_ceil(16);
    let candidate_cfgs: &[(usize, usize)] = if thread_m_blocks > 1 {
        &[(64, 256), (64, 128)]
    } else {
        &[(128, 128), (64, 128)]
    };
    let group_size = group_size as isize;
    let mut max_floats = 0usize;

    for &(thread_k, thread_n) in candidate_cfgs {
        if prob_k % thread_k != 0 || prob_n % thread_n != 0 {
            continue;
        }
        if thread_n < MARLIN_MIN_THREAD_N || thread_k < MARLIN_MIN_THREAD_K {
            continue;
        }
        let cache_size = marlin_moe_kernel_cache_size(
            thread_m_blocks,
            thread_n,
            thread_k,
            prob_n,
            prob_k,
            num_bits,
            group_size,
            false,
            true,
            false,
            false,
        );
        if cache_size + 512 > max_shared_mem {
            continue;
        }
        // The fused MoE kernel indexes C_tmp by locks_off rather than by launch
        // block id. locks_off can consume any slot in the fixed workspace lock
        // arena (sms * 4), so the FP32 reduction scratch must cover the full
        // lock-slot fanout even when the chosen launch occupancy is 1 block/SM.
        let floats = sms * MARLIN_MAX_LOCK_SLOTS_PER_SM * moe_block_size * thread_n;
        max_floats = max_floats.max(floats);
    }

    if max_floats == 0 {
        sms * moe_block_size * 256
    } else {
        max_floats
    }
}

fn max_shared_mem_for_device(device_ordinal: usize) -> usize {
    let mut max_shared_mem: i32 = 0;
    unsafe {
        cuda_sys::lib().cuDeviceGetAttribute(
            &mut max_shared_mem,
            cuda_sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            device_ordinal as i32,
        );
    }
    max_shared_mem.max(0) as usize
}

pub fn fused_moe_ctmp_floats_for_config(config: &PrefillModelConfig) -> (usize, usize) {
    if config.n_routed_experts == 0 {
        return (0, 0);
    }
    let moe_block_size = 64;
    let hidden_size = config.hidden_size;
    let moe_intermediate_size = config.moe_intermediate_size;
    let expert_bits = config.expert_bits as usize;
    let group_size = config.group_size.max(1);
    let max_shared_mem = max_shared_mem_for_device(config.device_ordinal);
    let w1_n = if config.moe_gated { 2 * moe_intermediate_size } else { moe_intermediate_size };
    let w1_ctmp_floats = fused_moe_fp32_reduce_floats(
        moe_block_size,
        w1_n,
        hidden_size,
        expert_bits,
        group_size,
        config.sms,
        max_shared_mem,
    );
    let w2_ctmp_floats = fused_moe_fp32_reduce_floats(
        moe_block_size,
        hidden_size,
        moe_intermediate_size,
        expert_bits,
        group_size,
        config.sms,
        max_shared_mem,
    );
    (w1_ctmp_floats, w2_ctmp_floats)
}

fn installed_package_sidecar(name: &str) -> Option<String> {
    use pyo3::types::PyModule;

    Python::with_gil(|py| {
        let pkg = PyModule::import(py, "krasis").ok()?;
        let pkg_file: String = pkg.getattr("__file__").ok()?.extract().ok()?;
        let pkg_dir = PathBuf::from(pkg_file).parent()?.to_path_buf();
        for candidate in [
            pkg_dir.join(name),
            pkg_dir.parent()?.join("krasis.libs").join(name),
        ] {
            if candidate.exists() {
                return Some(candidate.to_string_lossy().to_string());
            }
        }
        None
    })
}

// ════════════════════════════════════════════════════════════════════════
//  GpuBuf: raw synchronous GPU allocation (bypasses cudarc pool)
// ════════════════════════════════════════════════════════════════════════
//
// cudarc's CudaSlice uses cuMemAllocAsync/cuMemFreeAsync through the CUDA
// stream-ordered memory pool. When scratch buffers are freed and reallocated
// between prefill cycles, the pool's async free/alloc interacts badly with
// our CU_STREAM_NON_BLOCKING compute streams, causing GPU kernel hangs.
//
// GpuBuf uses cuMemAlloc_v2/cuMemFree_v2 (synchronous, no pool) so
// allocation and deallocation are immediate and deterministic.

pub struct GpuBuf<T: Copy> {
    ptr: u64,       // CUdeviceptr
    len: usize,     // element count (not bytes)
    device_ordinal: i32,
    _phantom: std::marker::PhantomData<T>,
}

unsafe impl<T: Copy + Send> Send for GpuBuf<T> {}
unsafe impl<T: Copy + Sync> Sync for GpuBuf<T> {}

impl<T: Copy> GpuBuf<T> {
    pub fn alloc_zeroed(count: usize) -> Result<Self, String> {
        let bytes = count * std::mem::size_of::<T>();
        if bytes == 0 {
            return Ok(Self { ptr: 0, len: 0, device_ordinal: 0, _phantom: std::marker::PhantomData });
        }
        // Get current device ordinal for drop
        let mut dev: i32 = 0;
        unsafe { cuda_sys::lib().cuCtxGetDevice(&mut dev); }
        let mut ptr: u64 = 0;
        let err = unsafe { cuda_sys::lib().cuMemAlloc_v2(&mut ptr, bytes) };
        if err != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuMemAlloc_v2({} bytes) failed: {:?}", bytes, err));
        }
        let err = unsafe { cuda_sys::lib().cuMemsetD8_v2(ptr, 0, bytes) };
        if err != cuda_sys::CUresult::CUDA_SUCCESS {
            unsafe { cuda_sys::lib().cuMemFree_v2(ptr); }
            return Err(format!("cuMemsetD8({} bytes) failed: {:?}", bytes, err));
        }
        Ok(Self { ptr, len: count, device_ordinal: dev, _phantom: std::marker::PhantomData })
    }

    /// Compatible with CudaSlice::device_ptr() — returns &u64 so *buf.device_ptr() works.
    pub fn device_ptr(&self) -> &u64 {
        &self.ptr
    }
}

impl<T: Copy> Drop for GpuBuf<T> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe {
                // Retain the primary context for this device and make it current,
                // so cuMemFree_v2 works regardless of which thread we're on.
                let mut ctx: cuda_sys::CUcontext = std::ptr::null_mut();
                cuda_sys::lib().cuDevicePrimaryCtxRetain(&mut ctx, self.device_ordinal);
                cuda_sys::lib().cuCtxSetCurrent(ctx);
                cuda_sys::lib().cuMemFree_v2(self.ptr);
                cuda_sys::lib().cuDevicePrimaryCtxRelease_v2(self.device_ordinal);
            }
            self.ptr = 0;
        }
    }
}

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
    kv_dequant_concat: RawCuFunc,
    kv_cache_append_polar4: RawCuFunc,
    kv_dequant_concat_polar4: RawCuFunc,
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
    // Optimized BF16 LA pipeline (fused kernels for FLA path)
    la_fused_conv1d_silu_bf16: RawCuFunc,  // fused conv+silu: 3 BF16 in -> 3 BF16 out
    la_update_conv_state: RawCuFunc,       // update conv state after fused conv
    la_compute_gate_beta_bf16: RawCuFunc,  // gate/beta with BF16 output
    la_fused_repeat_l2norm_bf16: RawCuFunc, // fused repeat_interleave + l2norm BF16
    la_gated_rmsnorm_bf16in: RawCuFunc,    // gated RMSNorm with BF16 input
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
    moe_finalize_sorted: RawCuFunc,
    moe_gather_sorted: RawCuFunc,
    moe_replicate_hidden: RawCuFunc,
    moe_scatter_fused: RawCuFunc,
    moe_scatter_weighted: RawCuFunc,

    // Marlin GEMM functions (loaded via dlopen from vendored libkrasis_marlin.so)
    marlin_mm: Option<MarlinMmFn>,
    // Fused MoE: vendored MarlinDefault (dlopen from libkrasis_marlin.so)
    pub fused_moe_fn: Option<FusedMoeFn>,
    // FlashAttention-2: vendored (dlopen from libkrasis_flash_attn.so)
    pub flash_attn_fwd: Option<FlashAttnFwdFn>,
    // FlashAttention-2 FP8 KV: BF16 Q with FP8 E4M3 K/V (for cross-chunk FP8 KV cache)
    pub flash_attn_fwd_fp8kv: Option<FlashAttnFwdFn>,
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
/// 40 parameters matching the extern "C" entry point.
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
    /*B_expert_ptrs*/ *const std::ffi::c_void,
    /*S_expert_ptrs*/ *const std::ffi::c_void,
);

/// Function pointer for vendored krasis_flash_attn_fwd_bf16.
/// Calls FlashAttention-2 forward pass from libkrasis_flash_attn.so.
type FlashAttnFwdFn = unsafe extern "C" fn(
    /*q_ptr*/              *const std::ffi::c_void,
    /*k_ptr*/              *const std::ffi::c_void,
    /*v_ptr*/              *const std::ffi::c_void,
    /*out_ptr*/            *mut std::ffi::c_void,
    /*softmax_lse_ptr*/    *mut std::ffi::c_void,
    /*cu_seqlens_q_ptr*/   *const std::ffi::c_void,
    /*cu_seqlens_k_ptr*/   *const std::ffi::c_void,
    /*batch_size*/         i32,
    /*seqlen_q*/           i32,
    /*seqlen_k*/           i32,
    /*num_heads*/          i32,
    /*num_heads_k*/        i32,
    /*head_dim*/           i32,
    /*total_q*/            i32,
    /*total_k*/            i32,
    /*softmax_scale*/      f32,
    /*is_causal*/          i32,
    /*unpadded_lse*/       i32,
    /*stream_ptr*/         *mut std::ffi::c_void,
) -> i32;

/// FLA (Flash Linear Attention) kernel function pointers.
/// Loaded from vendored libkrasis_fla.so (pre-compiled Triton cubins).
/// These replace the custom chunk recurrence kernels for LA layers.
pub struct FlaKernels {
    // 1. chunk_local_cumsum_scalar_kernel(s, o, T, grid_x, grid_y, stream)
    pub cumsum: FlaScalarKernelFn,
    // 2. chunk_scaled_dot_kkt_fwd_kernel(k, g, beta, A, T, grid_x, grid_y, stream)
    pub kkt: FlaKktKernelFn,
    // 3. merge_16x16_to_64x64_inverse_kernel(A, Ai, T, grid_x, grid_y, stream)
    pub solve_tril: FlaSolveTrilFn,
    // 4. recompute_w_u_fwd_kernel(k, v, beta, w, u, A, g, T, grid_x, grid_y, stream)
    pub wy_repr: FlaWyReprFn,
    // 5. chunk_gated_delta_rule_fwd_kernel_h_blockdim64(k, v, w, v_new, g, h, h0, ht, T, grid_x, grid_y, stream)
    pub state_recurrence: FlaStateRecurrenceFn,
    // 6. chunk_fwd_kernel_o(q, k, v, h, g, o, scale, T, grid_x, grid_y, stream)
    pub output: FlaOutputFn,
}

// FLA kernel function pointer types (matching extern "C" signatures in fla_vendor.cu).
// The Triton cross-compiler keeps cu_seqlens/chunk_indices as runtime args even when
// IS_VARLEN=False — we pass 0 (null) for them.  Similarly scale, g_gamma, gk are
// runtime args that we supply with the correct values.
type FlaInitFn = unsafe extern "C" fn() -> i32;
type FlaScalarKernelFn = unsafe extern "C" fn(
    /*s*/ u64, /*o*/ u64, /*scale*/ f32,
    /*cu_seqlens*/ u64, /*chunk_indices*/ u64, /*T*/ i32,
    /*grid_x*/ u32, /*grid_y*/ u32, /*grid_z*/ u32, /*stream*/ *mut std::ffi::c_void,
) -> u32;
type FlaKktKernelFn = unsafe extern "C" fn(
    /*k*/ u64, /*g*/ u64, /*beta*/ u64, /*A*/ u64,
    /*cu_seqlens*/ u64, /*chunk_indices*/ u64, /*T*/ i32,
    /*grid_x*/ u32, /*grid_y*/ u32, /*grid_z*/ u32, /*stream*/ *mut std::ffi::c_void,
) -> u32;
type FlaSolveTrilFn = unsafe extern "C" fn(
    /*A*/ u64, /*Ai*/ u64,
    /*cu_seqlens*/ u64, /*chunk_indices*/ u64, /*T*/ i32,
    /*grid_x*/ u32, /*grid_y*/ u32, /*grid_z*/ u32, /*stream*/ *mut std::ffi::c_void,
) -> u32;
type FlaWyReprFn = unsafe extern "C" fn(
    /*k*/ u64, /*v*/ u64, /*beta*/ u64, /*w*/ u64, /*u*/ u64, /*A*/ u64, /*g*/ u64,
    /*cu_seqlens*/ u64, /*chunk_indices*/ u64, /*T*/ i32,
    /*grid_x*/ u32, /*grid_y*/ u32, /*grid_z*/ u32, /*stream*/ *mut std::ffi::c_void,
) -> u32;
type FlaStateRecurrenceFn = unsafe extern "C" fn(
    /*k*/ u64, /*v*/ u64, /*w*/ u64, /*v_new*/ u64, /*g*/ u64, /*gk*/ u64,
    /*h*/ u64, /*h0*/ u64, /*ht*/ u64,
    /*cu_seqlens*/ u64, /*chunk_offsets*/ u64, /*T*/ i32,
    /*grid_x*/ u32, /*grid_y*/ u32, /*grid_z*/ u32, /*stream*/ *mut std::ffi::c_void,
) -> u32;
type FlaOutputFn = unsafe extern "C" fn(
    /*q*/ u64, /*k*/ u64, /*v*/ u64, /*h*/ u64, /*g*/ u64, /*g_gamma*/ u64,
    /*o*/ u64, /*cu_seqlens*/ u64, /*chunk_indices*/ u64, /*scale*/ f32, /*T*/ i32,
    /*grid_x*/ u32, /*grid_y*/ u32, /*grid_z*/ u32, /*stream*/ *mut std::ffi::c_void,
) -> u32;

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
    pub fused_moe_w1_ctmp_floats: usize,
    pub fused_moe_w2_ctmp_floats: usize,
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
    pub shared_w1_bf16: Option<Bf16Weight>,
    pub shared_w2_bf16: Option<Bf16Weight>,
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
    // QK norm (Qwen3 models): per-head RMSNorm on Q and K after projection, before RoPE
    pub q_norm_ptr: u64,             // BF16 [head_dim], 0 = no QK norm
    pub k_norm_ptr: u64,             // BF16 [head_dim], 0 = no QK norm
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
    /// Per-layer contiguous backing for bulk DMA (pinned host memory).
    /// When set (ptr != 0), enables 4 DMA calls per layer instead of 4*E.
    /// Points into the WeightStore's LayerExpertBacking which is pinned via cuMemHostRegister.
    pub bulk_w13p: (usize, usize),  // (host_ptr, total_bytes) for ALL experts' w13_packed
    pub bulk_w13s: (usize, usize),  // w13_scales
    pub bulk_w2p: (usize, usize),   // w2_packed
    pub bulk_w2s: (usize, usize),   // w2_scales
}

pub struct PrefillScratch {
    pub d_hidden: GpuBuf<u16>,
    pub d_residual: GpuBuf<u16>,
    pub d_scratch1: GpuBuf<u16>,
    pub d_scratch2: GpuBuf<u16>,
    pub d_fp32_scratch: GpuBuf<f32>,
    pub d_workspace: GpuBuf<i32>,
    pub d_topk_weights: GpuBuf<f32>,
    pub d_topk_ids: GpuBuf<i32>,
    pub d_token_ids: GpuBuf<i32>,
    pub d_positions: GpuBuf<i32>,
    pub d_attn_out: GpuBuf<u16>,
    pub d_q: GpuBuf<u16>,
    pub d_k: GpuBuf<u16>,
    pub d_v: GpuBuf<u16>,
    pub d_logits: GpuBuf<f32>,
    pub d_mamba2_conv_state: Option<GpuBuf<f32>>,
    pub d_mamba2_ssm_state: Option<GpuBuf<f32>>,
    pub d_mamba2_in_proj: Option<GpuBuf<u16>>,
    pub d_mamba2_ssd_out: Option<GpuBuf<u16>>,
    // MoE scratch buffers
    pub d_gate_out: GpuBuf<f32>,         // [max_tokens, max_experts] FP32
    // Shared MoE buffers: sized for fused_sorted_count (= max_tokens * topk + n_routed * block_size).
    // Used by BOTH sequential and fused MoE paths (never simultaneously).
    // Sequential uses [max_tokens * topk] entries; fused uses [fused_sorted_count].
    pub d_moe_accum: GpuBuf<f32>,        // [fused_sorted_count, max(w1_n, h)] FP32 -- fused C_tmp or [m,h] scatter
    pub d_moe_gathered: GpuBuf<u16>,     // [fused_sorted_count, hidden] BF16 -- fused_input or gathered
    pub d_moe_expert_out: GpuBuf<u16>,   // [fused_sorted_count, hidden] BF16 -- fused_output or expert_out
    pub d_moe_gate_up: GpuBuf<u16>,      // [fused_sorted_count, w1_n] BF16 -- fused_inter_cache or gate_up
    pub d_moe_inter: GpuBuf<u16>,        // [fused_sorted_count, inter] BF16 -- fused_inter2 or inter
    pub d_gather_src_map: GpuBuf<i32>,   // [fused_sorted_count] -- sorted_token_ids or gather_src_map
    pub d_gather_weight_map: GpuBuf<f32>,// [max_tokens * topk]
    // Fused MoE sorted dispatch buffers (in scratch so both paths share VRAM)
    pub d_fused_expert_ids: GpuBuf<i32>, // [fused_blocks]
    pub d_num_tokens_post: GpuBuf<i32>,  // [1]
    pub fused_sorted_count: usize,
    pub fused_blocks: usize,
    // GPU-only routing scratch
    pub d_expert_counts: GpuBuf<i32>,    // [max_experts]
    pub d_expert_offsets: GpuBuf<i32>,   // [max_experts + 1]
    pub d_write_offsets: GpuBuf<i32>,    // [max_experts]
    // Expert weight DMA double-buffers (A and B for ping-pong DMA/compute overlap)
    pub d_expert_w13_packed_a: GpuBuf<u8>,
    pub d_expert_w13_scales_a: GpuBuf<u8>,
    pub d_expert_w2_packed_a: GpuBuf<u8>,
    pub d_expert_w2_scales_a: GpuBuf<u8>,
    pub d_expert_w13_packed_b: GpuBuf<u8>,
    pub d_expert_w13_scales_b: GpuBuf<u8>,
    pub d_expert_w2_packed_b: GpuBuf<u8>,
    pub d_expert_w2_scales_b: GpuBuf<u8>,
    // Linear attention scratch buffers
    pub d_la_q: Option<GpuBuf<f32>>,       // [max_tokens, nv, dk] FP32
    pub d_la_k: Option<GpuBuf<f32>>,       // [max_tokens, nv, dk] FP32
    pub d_la_v: Option<GpuBuf<f32>>,       // [max_tokens, nv, dv] FP32
    pub d_la_z: Option<GpuBuf<u16>>,       // [max_tokens, nv, dv] BF16 (gate)
    pub d_la_b: Option<GpuBuf<u16>>,       // [max_tokens, nv] BF16
    pub d_la_a: Option<GpuBuf<u16>>,       // [max_tokens, nv] BF16
    pub d_la_beta: Option<GpuBuf<f32>>,    // [max_tokens, nv] FP32
    pub d_la_gate: Option<GpuBuf<f32>>,    // [max_tokens, nv] FP32
    pub d_la_conv_out: Option<GpuBuf<f32>>, // [conv_dim, max_tokens] FP32
    pub d_la_v_beta: Option<GpuBuf<f32>>,  // [nv, max_tokens, dv] FP32
    pub d_la_k_beta: Option<GpuBuf<f32>>,  // [nv, max_tokens, dk] FP32
    pub d_la_v_new: Option<GpuBuf<f32>>,   // [nv, chunk_size, dv] FP32 (per chunk)
    pub d_la_g_cum: Option<GpuBuf<f32>>,   // [nv, num_chunks, chunk_size] FP32
    pub d_la_attn: Option<GpuBuf<f32>>,    // [nv, num_chunks, CS, CS] FP32
    pub d_la_state: Option<GpuBuf<f32>>,   // [nv, dk, dv] FP32 (recurrent state)
    // FlashAttention-2 scratch
    pub d_fa2_lse: Option<GpuBuf<f32>>,    // [num_heads, max_tokens] FP32 softmax LSE
    pub d_la_chunk_out: Option<GpuBuf<f32>>, // [nv, chunk_size, dv] FP32
    pub d_la_q_contig: Option<GpuBuf<f32>>, // [nv, chunk_size, dk] FP32 (separate from chunk_out to avoid aliasing)
    pub d_la_proj_buf: Option<GpuBuf<u16>>, // projection buffer [max_tokens, proj_dim] BF16
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
    pub device: Arc<CudaDevice>,
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
    /// KV cache format: 0=bf16, 1=fp8, 2=polar4
    pub kv_format: u32,
    /// Polar4 KV cache pointers (only used when kv_format==2)
    pub kv_k_radius_ptrs: Vec<u64>,
    pub kv_v_radius_ptrs: Vec<u64>,
    pub kv_k_angles_ptrs: Vec<u64>,
    pub kv_v_angles_ptrs: Vec<u64>,
    pub kv_num_blocks: usize,
    pub stream: cuda_sys::CUstream,
    pub copy_stream: cuda_sys::CUstream,
    pub cublas_handle: cudarc::cublas::sys::cublasHandle_t,
    pub shared_cublas_handle: cudarc::cublas::sys::cublasHandle_t,
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
    // Dedicated stream for shared expert (gap 4: always async regardless of cold DMA)
    pub shared_stream: cuda_sys::CUstream,
    pub shared_event: cuda_sys::CUevent,
    // Separate Marlin workspace for shared_stream (avoids d_fp32_scratch conflict)
    // These are allocated dynamically in prepare_for_prefill and freed in release_scratch
    // to maximize VRAM available for HCS during decode.
    pub d_shared_fp32_scratch: Option<CudaSlice<f32>>,
    pub d_shared_workspace: Option<CudaSlice<i32>>,
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
    // Fused MoE token buffers now live in PrefillScratch (shared with sequential).
    // Only the fused_sorted_count is tracked here for reference.
    pub max_sorted: usize,
    // Preloaded layer index tracking for group-level double-buffer (gap 2)
    pub preloaded_moe_layer: Option<usize>,
    // Expert pinning pool for prefill MoE optimization
    // The pool holds hottest experts (from gate pre-scan) in GPU VRAM.
    // Per-layer DMA copies pinned experts via fast D2D instead of slow PCIe H2D.
    pub pinning_pool_ptr: u64,                           // Raw GPU pointer for pinning pool (0 = not allocated)
    pub pinning_pool_bytes: usize,                       // Total allocated bytes
    pub pinned_expert_offsets: Vec<Vec<Option<usize>>>,  // [moe_layer_idx][expert_id] -> byte offset in pool (None = not pinned)
    pub pinning_pool_expert_bytes: usize,                // total bytes per expert in pool (w1p + w1s + w2p + w2s)
    pub pinning_active: bool,
    // Pre-scan routing data: per-layer list of predicted active expert IDs per chunk
    pub prescan_active_experts: Vec<Vec<Vec<usize>>>,    // [moe_layer_idx][chunk_idx] -> Vec<expert_id>
    // Expert pointer table for zero-copy MoE (avoids contiguous fused buffer)
    // GPU-side arrays: each entry is a raw GPU pointer to that expert's weights/scales.
    // For HCS-resident experts, points directly into HCS VRAM (zero copy).
    // For cold experts, points into cold_staging buffer after H2D transfer.
    pub d_expert_w1_ptrs: Option<CudaSlice<u64>>,       // [n_experts] GPU ptrs to packed w1
    pub d_expert_w1s_ptrs: Option<CudaSlice<u64>>,      // [n_experts] GPU ptrs to w1 scales
    pub d_expert_w2_ptrs: Option<CudaSlice<u64>>,       // [n_experts] GPU ptrs to packed w2
    pub d_expert_w2s_ptrs: Option<CudaSlice<u64>>,      // [n_experts] GPU ptrs to w2 scales
    // Host-side staging arrays for building pointer tables
    pub h_expert_w1_ptrs: Vec<u64>,
    pub h_expert_w1s_ptrs: Vec<u64>,
    pub h_expert_w2_ptrs: Vec<u64>,
    pub h_expert_w2s_ptrs: Vec<u64>,
    // Cold expert staging buffer: small GPU buffer for H2D of non-HCS experts
    pub d_cold_staging: Option<CudaSlice<u8>>,
    pub cold_expert_bytes: usize,   // bytes per cold expert slot (w1p + w1s + w2p + w2s)
    pub max_cold_experts: usize,    // max cold experts the staging buffer can hold
    // ScalarType for Marlin GEMM dispatch (matches weight quantization format)
    pub q_type: ScalarType,
    // BF16 copies of QK norm weights (converted from FP32 at engine creation)
    // Kept alive here so GPU memory isn't freed.
    pub qk_norm_bf16_bufs: Vec<CudaSlice<u16>>,
    // GQA sub-component timing accumulators (Cell for interior mutability in &self methods)
    pub gqa_timing_enabled: Cell<bool>,
    pub t_gqa_proj: Cell<f64>,       // Q/K/V projection GEMMs
    pub t_gqa_norm: Cell<f64>,       // QK LayerNorm
    pub t_gqa_rope: Cell<f64>,       // RoPE
    pub t_gqa_kv_prep: Cell<f64>,    // KV cache store (FP8 path) or dequant+concat (BF16 path)
    pub t_gqa_fa2: Cell<f64>,        // FlashAttention-2 kernel call
    pub t_gqa_gate: Cell<f64>,       // sigmoid gate (gated attention)
    pub t_gqa_oproj: Cell<f64>,      // O projection GEMM
    pub gqa_fa2_calls: Cell<u64>,    // number of FA2 calls (for averaging)
    pub gqa_fp8_calls: Cell<u64>,    // how many used FP8 path
    pub gqa_bf16_calls: Cell<u64>,   // how many used BF16 dequant path
    // LA sub-component timing accumulators
    pub t_la_proj: Cell<f64>,        // in_proj_qkvz + in_proj_ba GEMMs
    pub t_la_uninterleave: Cell<f64>, // uninterleave qkvz + ba
    pub t_la_conv: Cell<f64>,        // concat + conv1d + split
    pub t_la_prep: Cell<f64>,        // gate/beta + repeat_interleave + l2norm
    pub t_la_convert: Cell<f64>,     // FP32->BF16 conversions for FLA
    pub t_la_fla: Cell<f64>,         // FLA kernel calls (6 steps)
    pub t_la_postfla: Cell<f64>,     // BF16->FP32 conversion + state copy
    pub t_la_norm: Cell<f64>,        // gated RMSNorm
    pub t_la_oproj: Cell<f64>,       // output projection GEMM
    // MoE sub-component timing accumulators
    pub t_moe_gate: Cell<f64>,       // gate GEMM + top-k routing + alignment
    pub t_moe_dma: Cell<f64>,        // expert weight DMA + sync
    pub t_moe_w1: Cell<f64>,         // fused w1 GEMM + activation
    pub t_moe_w2: Cell<f64>,         // fused w2 GEMM
    pub t_moe_scatter: Cell<f64>,    // scatter-add + accum_to_bf16
    pub t_moe_shared: Cell<f64>,     // shared expert (wait + add)
    // FLA (Flash Linear Attention) vendored kernels — replaces custom LA chunk recurrence
    pub fla: Option<FlaKernels>,
    // FLA intermediate buffers (allocated once at engine init if FLA available)
    pub d_fla_g_cumsum: Option<CudaSlice<f32>>,  // [B, T, H] FP32 cumsum of gate
    pub d_fla_a: Option<CudaSlice<f32>>,          // [B, T, H, BT] FP32 attention matrix
    pub d_fla_ai: Option<CudaSlice<u16>>,         // [B, T, H, BT] BF16 inverse matrix
    pub d_fla_w: Option<CudaSlice<u16>>,          // [B, T, H, K] BF16 WY representation w
    pub d_fla_u: Option<CudaSlice<u16>>,          // [B, T, H, V] BF16 WY representation u
    pub d_fla_h: Option<CudaSlice<u16>>,          // [B, NT, H, K, V] BF16 per-chunk states
    pub d_fla_final_state: Option<CudaSlice<f32>>,  // [B, H, K, V] FP32 final state
    pub d_fla_v_new: Option<CudaSlice<u16>>,      // [B, T, H, V] BF16 corrected values
    pub d_fla_o: Option<CudaSlice<u16>>,          // [B, T, H, V] BF16 output
    /// Runtime-configured VRAM reserve for dynamic prefill scratch sizing.
    pub safety_margin_mb: usize,
    /// Temporary pointer back to the decode store for chunk-boundary HCS guardrails.
    /// Set only for the lifetime of an active prefill request.
    pub prefill_hcs_store_addr: usize,
}

/// Double-buffer BF16 attention weight streaming buffers.
/// Each buffer holds one layer's Q/K/V/O projection weights in BF16.
pub struct AttnWeightStreamBufs {
    pub buf_a: CudaSlice<u8>,   // Buffer A: holds one layer's attention weights
    pub buf_b: CudaSlice<u8>,   // Buffer B: double-buffer counterpart
    pub buf_size: usize,        // Size of each buffer in bytes
}


// Safety: PrefillEngine contains raw CUDA pointers (CUstream, CUevent, CUfunction, cuBLAS handle).
// These are only accessed from the server's request handler which processes one request at a time
// (single-request guarantee). The raw pointers themselves are thread-safe CUDA handles.
unsafe impl Send for PrefillEngine {}
unsafe impl Sync for PrefillEngine {}

impl Drop for PrefillEngine {
    fn drop(&mut self) {
        unsafe {
            if !self.shared_event.is_null() {
                let _ = cuda_sys::lib().cuEventDestroy_v2(self.shared_event);
            }
            if !self.compute_event.is_null() {
                let _ = cuda_sys::lib().cuEventDestroy_v2(self.compute_event);
            }
            if !self.dma_event.is_null() {
                let _ = cuda_sys::lib().cuEventDestroy_v2(self.dma_event);
            }
            if !self.shared_stream.is_null() {
                let _ = cuda_sys::lib().cuStreamDestroy_v2(self.shared_stream);
            }
            if !self.copy_stream.is_null() {
                let _ = cuda_sys::lib().cuStreamDestroy_v2(self.copy_stream);
            }
            if !self.stream.is_null() {
                let _ = cuda_sys::lib().cuStreamDestroy_v2(self.stream);
            }
            if !self.shared_cublas_handle.is_null() {
                let _ = cudarc::cublas::result::destroy_handle(self.shared_cublas_handle);
            }
            if !self.cublas_handle.is_null() {
                let _ = cudarc::cublas::result::destroy_handle(self.cublas_handle);
            }
        }
    }
}

impl PrefillEngine {
    pub fn set_safety_margin_mb(&mut self, margin_mb: usize) {
        self.safety_margin_mb = margin_mb.max(PREFILL_SAFETY_MARGIN_MB);
    }

    /// Update HCS snapshot from the decode store's current HCS state.
    /// Must be called before each prefill so we know which experts are GPU-resident.
    pub fn update_hcs_snapshot(&mut self, cache_fast: &[[u64; 4]], num_experts_per_layer: usize) {
        self.hcs_cache_fast = cache_fast.to_vec();
        self.hcs_num_experts_per_layer = num_experts_per_layer;
    }

    fn hcs_cached_expert_count(&self) -> usize {
        self.hcs_cache_fast.iter()
            .filter(|ptrs| ptrs[0] != 0 || ptrs[1] != 0 || ptrs[2] != 0 || ptrs[3] != 0)
            .count()
    }

    pub fn set_prefill_hcs_guard_store_addr(&mut self, store_addr: usize) {
        self.prefill_hcs_store_addr = store_addr;
    }

    pub fn clear_prefill_hcs_guard_store_addr(&mut self) {
        self.prefill_hcs_store_addr = 0;
    }

    /// Check if an expert is GPU-resident in HCS.
    /// Returns Some((w13_packed, w13_scales, w2_packed, w2_scales)) if resident.
    /// During prefill, surviving HCS experts (not evicted for scratch) are used
    /// opportunistically — no separate prefill cache needed.
    fn expert_lookup(&self, moe_layer_idx: usize, expert_idx: usize) -> Option<(u64, u64, u64, u64)> {
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

    fn run_prefill_chunk_hcs_guard(
        &mut self,
        chunk_tokens: usize,
        chunk_idx: usize,
        total_chunks: usize,
    ) -> Result<(), String> {
        if self.prefill_hcs_store_addr == 0 {
            return Ok(());
        }

        let (evicted, freed_mb, cache_fast_snapshot, num_experts_per_layer) = unsafe {
            let store = &mut *(self.prefill_hcs_store_addr as *mut crate::gpu_decode::GpuDecodeStore);
            let (evicted, freed_mb) = store.hcs_evict_for_prefill(chunk_tokens);
            let (cache_fast, ne) = store.export_hcs_snapshot();
            (evicted, freed_mb, cache_fast.to_vec(), ne)
        };
        self.update_hcs_snapshot(&cache_fast_snapshot, num_experts_per_layer);

        if evicted > 0 && stderr_debug_enabled() {
            eprintln!(
                "[PREFILL] Chunk guard {}/{}: evicted {} HCS experts ({:.1} MB) for {}-token chunk",
                chunk_idx + 1,
                total_chunks,
                evicted,
                freed_mb,
                chunk_tokens,
            );
        }

        Ok(())
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

    /// Download BF16 values from GPU as f32.
    fn diag_download_bf16(&self, ptr: u64, count: usize) -> Vec<f32> {
        let mut buf = vec![0u16; count];
        unsafe {
            cuda_sys::lib().cuStreamSynchronize(self.stream);
            cuda_sys::lib().cuMemcpyDtoH_v2(
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                ptr,
                (count * 2) as usize,
            );
        }
        buf.iter().map(|&bits| {
            let f32_bits = (bits as u32) << 16;
            f32::from_bits(f32_bits)
        }).collect()
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
    /// Default: 9999 (all layers). Set to e.g. 5 for layers 0-4.
    fn diag_layer_limit() -> usize {
        std::env::var("KRASIS_PREFILL_DIAG_LAYERS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(9999)
    }

    /// Dynamically allocate scratch buffers sized for this prompt.
    /// Called before each prefill. GpuBuf uses raw cuMemAlloc_v2 (synchronous,
    /// no pool) so alloc/free is immediate — no cudarc pool interaction.
    ///
    /// Flow: sync GPU → trim cudarc pool (release HCS soft VRAM) → measure free
    /// VRAM → compute chunk_size → drop old scratch → allocate new scratch.
    pub fn prepare_for_prefill(&mut self, prompt_tokens: usize) -> Result<(), String> {
        self.bind_cuda_context()?;

        // 1. Synchronize all GPU work (ensures pending cuMemFreeAsync from HCS
        //    soft eviction have completed on cudarc's internal stream).
        unsafe { cuda_sys::lib().cuCtxSynchronize(); }

        // 2. Trim cudarc's default memory pool to release HCS soft VRAM back to OS.
        //    This only releases UNUSED (freed) pool memory — HCS hard pool is still
        //    allocated and untouched. Safe because we just synchronized all streams.
        let mut free_before: usize = 0;
        let mut total_before: usize = 0;
        unsafe { cuda_sys::lib().cuMemGetInfo_v2(&mut free_before, &mut total_before); }
        unsafe {
            let mut pool: cuda_sys::CUmemoryPool = std::ptr::null_mut();
            let mut dev: i32 = 0;
            cuda_sys::lib().cuCtxGetDevice(&mut dev);
            let err = cuda_sys::lib().cuDeviceGetDefaultMemPool(&mut pool, dev);
            if err == cuda_sys::CUresult::CUDA_SUCCESS && !pool.is_null() {
                let trim_err = cuda_sys::lib().cuMemPoolTrimTo(pool, 0);
                let mut free_after: usize = 0;
                let mut total_after: usize = 0;
                cuda_sys::lib().cuMemGetInfo_v2(&mut free_after, &mut total_after);
                if stderr_debug_enabled() {
                    eprintln!("[SCRATCH] Pool trim: {:?}, freed {} MB",
                        trim_err, (free_after - free_before) / (1024 * 1024));
                }
            }
        }

        // 3. Drop old scratch to free its VRAM (GpuBuf::drop = cuMemFree_v2, immediate).
        //    Also free prefill-only GPU buffers from previous request.
        let old_tokens = self.scratch.max_tokens;
        self.scratch = allocate_scratch(&self.device, &self.config, 0)
            .map_err(|e| format!("alloc empty scratch: {e}"))?;
        self.d_cold_staging = None;
        self.d_shared_fp32_scratch = None;
        self.d_shared_workspace = None;
        self.d_fla_g_cumsum = None;
        self.d_fla_a = None;
        self.d_fla_ai = None;
        self.d_fla_w = None;
        self.d_fla_u = None;
        self.d_fla_h = None;
        self.d_fla_final_state = None;
        self.d_fla_v_new = None;
        self.d_fla_o = None;
        // Old scratch + prefill buffers now dropped, VRAM is freed.

        // 4. Measure free VRAM and compute optimal chunk_size for this prompt.
        let (fixed_bytes, per_token_bytes) = compute_scratch_vram(&self.config);
        let mut free_bytes: usize = 0;
        let mut total_bytes: usize = 0;
        unsafe { cuda_sys::lib().cuMemGetInfo_v2(&mut free_bytes, &mut total_bytes); }
        // Safety margin covers CUDA context overhead, allocator fragmentation, and
        // FLA Triton kernel temporaries. 600 MB is sufficient based on runtime measurements.
        let safety_margin_mb = self.safety_margin_mb.max(PREFILL_SAFETY_MARGIN_MB);
        let safety_bytes: usize = safety_margin_mb * 1024 * 1024;

        let usable = free_bytes.saturating_sub(safety_bytes).saturating_sub(fixed_bytes);
        let max_by_vram = if per_token_bytes > 0 { usable / per_token_bytes } else { 50000 };

        // Size for this prompt: cap at prompt_tokens (no point allocating more).
        // Minimum 128: fused MoE Marlin kernels use block_size_m=64, need enough
        // tokens for stable sorted dispatch. 128 adds ~57 MB scratch overhead.
        let target = prompt_tokens.min(50000);
        let mut scratch_tokens = max_by_vram.min(target).max(128);
        let debug_prefill = prefill_debug_enabled();
        let mut measured_cap_debug: Option<usize> = None;

        // If decode store calibration is available for this live request, further cap
        // scratch/chunk size using the measured no-HCS prefill model. This handles
        // model/GPU-specific VRAM consumers that the static scratch estimator misses.
        if self.prefill_hcs_store_addr != 0 {
            let free_mb = (free_bytes / (1024 * 1024)) as u64;
            let measured_cap = unsafe {
                let store = &*(self.prefill_hcs_store_addr as *const crate::gpu_decode::GpuDecodeStore);
                store.max_safe_prefill_chunk_tokens(target, free_mb)
            };
            if let Some(measured_cap) = measured_cap {
                measured_cap_debug = Some(measured_cap);
                scratch_tokens = scratch_tokens.min(measured_cap.max(128));
            }
        }
        if debug_prefill {
            eprintln!(
                "[PREFILL-DEBUG] prepare prompt_tokens={} free_mb={} total_mb={} safety_mb={} fixed_mb={:.1} per_tok_kb={:.1} usable_mb={:.1} target={} max_by_vram={} measured_cap={} initial_scratch={}",
                prompt_tokens,
                free_bytes / (1024 * 1024),
                total_bytes / (1024 * 1024),
                safety_margin_mb,
                fixed_bytes as f64 / (1024.0 * 1024.0),
                per_token_bytes as f64 / 1024.0,
                usable as f64 / (1024.0 * 1024.0),
                target,
                max_by_vram,
                measured_cap_debug.map(|v| v.to_string()).unwrap_or_else(|| "none".to_string()),
                scratch_tokens,
            );
        }

        // 5. Allocate prefill-only GPU buffers and scratch. If live post-allocation
        //    free VRAM still lands below the configured floor, shrink the chunk and
        //    retry instead of trusting the linear model right at the boundary.
        let mut attempt = 0usize;
        let mut post_alloc_free_bytes = free_bytes;
        loop {
            attempt += 1;

            let n_routed = self.config.n_routed_experts;
            if n_routed > 0 {
                let total_cold = n_routed * self.cold_expert_bytes;
                let buf = self.device.alloc_zeros::<u8>(total_cold)
                    .map_err(|e| format!("alloc cold_staging: {e}"))?;
                if stderr_debug_enabled() && attempt == 1 {
                    eprintln!("[PREFILL] Cold staging: {} MB ({} experts)", total_cold / (1024 * 1024), n_routed);
                }
                self.d_cold_staging = Some(buf);
                self.max_cold_experts = n_routed;
            }

            if self.fla.is_some() && self.config.layer_types.iter().any(|&t| t == 3) {
                let nv = self.config.la_num_v_heads;
                let dk = self.config.la_k_head_dim;
                let dv = self.config.la_v_head_dim;
                let fla_bt = 64usize;
                let fla_total = ((scratch_tokens + fla_bt - 1) / fla_bt) * fla_bt;
                let fla_nt = fla_total / fla_bt;
                self.d_fla_g_cumsum = Some(self.device.alloc_zeros::<f32>(fla_total * nv)
                    .map_err(|e| format!("alloc fla_g_cumsum: {e}"))?);
                self.d_fla_a = Some(self.device.alloc_zeros::<f32>(fla_total * nv * fla_bt)
                    .map_err(|e| format!("alloc fla_a: {e}"))?);
                self.d_fla_ai = Some(self.device.alloc_zeros::<u16>(fla_total * nv * fla_bt)
                    .map_err(|e| format!("alloc fla_ai: {e}"))?);
                self.d_fla_w = Some(self.device.alloc_zeros::<u16>(fla_total * nv * dk)
                    .map_err(|e| format!("alloc fla_w: {e}"))?);
                self.d_fla_u = Some(self.device.alloc_zeros::<u16>(fla_total * nv * dv)
                    .map_err(|e| format!("alloc fla_u: {e}"))?);
                self.d_fla_h = Some(self.device.alloc_zeros::<u16>(fla_nt * nv * dk * dv)
                    .map_err(|e| format!("alloc fla_h: {e}"))?);
                self.d_fla_final_state = Some(self.device.alloc_zeros::<f32>(nv * dk * dv)
                    .map_err(|e| format!("alloc fla_final_state: {e}"))?);
                self.d_fla_v_new = Some(self.device.alloc_zeros::<u16>(fla_total * nv * dv)
                    .map_err(|e| format!("alloc fla_v_new: {e}"))?);
                self.d_fla_o = Some(self.device.alloc_zeros::<u16>(fla_total * nv * dv)
                    .map_err(|e| format!("alloc fla_o: {e}"))?);
            }

            // Shared expert Marlin workspace: sized for actual chunk, not max possible.
            let shared_scratch_n = std::cmp::max(self.config.hidden_size, self.config.moe_intermediate_size);
            let shared_scratch_elems = scratch_tokens * shared_scratch_n;
            if shared_scratch_elems > 0 {
                self.d_shared_fp32_scratch = Some(self.device.alloc_zeros::<f32>(shared_scratch_elems)
                    .map_err(|e| format!("alloc shared_fp32_scratch: {e}"))?);
                self.d_shared_workspace = Some(self.device.alloc_zeros::<i32>(self.config.sms * MARLIN_MAX_LOCK_SLOTS_PER_SM)
                    .map_err(|e| format!("alloc shared_workspace: {e}"))?);
            }

            self.scratch = allocate_scratch(&self.device, &self.config, scratch_tokens)?;

            unsafe { cuda_sys::lib().cuCtxSynchronize(); }
            unsafe { cuda_sys::lib().cuMemGetInfo_v2(&mut post_alloc_free_bytes, &mut total_bytes); }
            if debug_prefill {
                eprintln!(
                    "[PREFILL-DEBUG] alloc attempt={} scratch_tokens={} post_alloc_free_mb={} safety_mb={} cold_staging_mb={} shared_fp32_mb={:.1}",
                    attempt,
                    scratch_tokens,
                    post_alloc_free_bytes / (1024 * 1024),
                    safety_margin_mb,
                    (n_routed * self.cold_expert_bytes) / (1024 * 1024),
                    (shared_scratch_elems * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0),
                );
            }
            if post_alloc_free_bytes >= safety_bytes || scratch_tokens <= 128 {
                self.config.prefill_chunk_size = scratch_tokens;
                break;
            }

            let deficit_bytes = safety_bytes - post_alloc_free_bytes;
            let deficit_tokens = if per_token_bytes > 0 {
                (deficit_bytes + per_token_bytes - 1) / per_token_bytes
            } else {
                0
            };
            let shrink_tokens = deficit_tokens
                .saturating_mul(2)
                .max((scratch_tokens / 8).max(64));
            let next_tokens = scratch_tokens.saturating_sub(shrink_tokens).max(128);

            if stderr_debug_enabled() {
                eprintln!(
                    "[PREFILL] Retry chunk size {} -> {} after post-alloc free {} MB fell below {} MB floor (attempt {})",
                    scratch_tokens,
                    next_tokens,
                    post_alloc_free_bytes / (1024 * 1024),
                    safety_margin_mb,
                    attempt,
                );
            }

            self.scratch = allocate_scratch(&self.device, &self.config, 0)
                .map_err(|e| format!("alloc empty scratch: {e}"))?;
            self.d_cold_staging = None;
            self.d_shared_fp32_scratch = None;
            self.d_shared_workspace = None;
            self.d_fla_g_cumsum = None;
            self.d_fla_a = None;
            self.d_fla_ai = None;
            self.d_fla_w = None;
            self.d_fla_u = None;
            self.d_fla_h = None;
            self.d_fla_final_state = None;
            self.d_fla_v_new = None;
            self.d_fla_o = None;
            unsafe { cuda_sys::lib().cuCtxSynchronize(); }
            unsafe {
                let mut pool: cuda_sys::CUmemoryPool = std::ptr::null_mut();
                let mut dev: i32 = 0;
                cuda_sys::lib().cuCtxGetDevice(&mut dev);
                let err = cuda_sys::lib().cuDeviceGetDefaultMemPool(&mut pool, dev);
                if err == cuda_sys::CUresult::CUDA_SUCCESS && !pool.is_null() {
                    let _ = cuda_sys::lib().cuMemPoolTrimTo(pool, 0);
                }
            }

            if next_tokens >= scratch_tokens {
                self.config.prefill_chunk_size = scratch_tokens;
                break;
            }
            scratch_tokens = next_tokens;
        }

        let scratch_mb = (fixed_bytes + per_token_bytes * scratch_tokens) as f64 / (1024.0 * 1024.0);
        if debug_prefill {
            eprintln!(
                "[PREFILL-DEBUG] ready scratch_tokens={} scratch_mb={:.1} prompt_tokens={} post_alloc_free_mb={} safety_mb={}",
                scratch_tokens,
                scratch_mb,
                prompt_tokens,
                post_alloc_free_bytes / (1024 * 1024),
                safety_margin_mb,
            );
        }
        if scratch_tokens != old_tokens {
            if stderr_debug_enabled() {
                eprintln!("[PREFILL] Dynamic scratch: {} tokens ({:.0} MB) for {}-token prompt ({} MB safety, {:.0} MB free)",
                    scratch_tokens, scratch_mb, prompt_tokens, safety_margin_mb,
                    free_bytes as f64 / (1024.0 * 1024.0));
            }
        }
        if stderr_debug_enabled() && post_alloc_free_bytes < safety_bytes {
            eprintln!(
                "[PREFILL] Post-alloc free still below floor at min chunk: {} MB free vs {} MB target",
                post_alloc_free_bytes / (1024 * 1024),
                safety_margin_mb,
            );
        }

        Ok(())
    }

    /// Release scratch VRAM and prefill-only buffers so HCS can reclaim VRAM for decode.
    /// GpuBuf::drop calls cuMemFree_v2 (synchronous, immediate release).
    pub fn release_scratch(&mut self) -> Result<(), String> {
        self.bind_cuda_context()?;
        // Synchronize to ensure all prefill GPU work is done before freeing buffers.
        unsafe { cuda_sys::lib().cuCtxSynchronize(); }
        // Replace scratch with zero-capacity (drops old, frees VRAM immediately).
        self.scratch = allocate_scratch(&self.device, &self.config, 0)
            .map_err(|e| format!("alloc empty scratch: {e}"))?;
        self.config.prefill_chunk_size = 0;
        // Free prefill-only GPU buffers: cold staging + shared expert scratch.
        // These are only needed during prefill and waste ~1.2 GB of HCS space if kept.
        self.d_cold_staging = None;
        self.d_shared_fp32_scratch = None;
        self.d_shared_workspace = None;
        self.d_fla_g_cumsum = None;
        self.d_fla_a = None;
        self.d_fla_ai = None;
        self.d_fla_w = None;
        self.d_fla_u = None;
        self.d_fla_h = None;
        self.d_fla_final_state = None;
        self.d_fla_v_new = None;
        self.d_fla_o = None;
        // Trim cudarc's memory pool so freed memory returns to the OS immediately.
        // Without this, cudarc holds the freed cold_staging/shared_scratch in its pool,
        // and HCS reload can't use that VRAM.
        unsafe {
            let mut pool: cuda_sys::CUmemoryPool = std::ptr::null_mut();
            let mut dev: i32 = 0;
            cuda_sys::lib().cuCtxGetDevice(&mut dev);
            let err = cuda_sys::lib().cuDeviceGetDefaultMemPool(&mut pool, dev);
            if err == cuda_sys::CUresult::CUDA_SUCCESS && !pool.is_null() {
                cuda_sys::lib().cuMemPoolTrimTo(pool, 0);
            }
        }
        Ok(())
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
        // KRASIS_PREFILL_DIAG_LAYERS=N to check N layers (default all)
        // KRASIS_PREFILL_DIAG_POSITIONS=0,1,2,3,4,28,29,30,31,32 to set positions
        // KRASIS_PREFILL_DIAG_MOE_DETAIL=1 to enable heavy per-row MoE diagnostics
        let diag = std::env::var("KRASIS_PREFILL_DIAG").is_ok();
        let diag_positions = if diag { Self::diag_positions(total_m) } else { vec![] };
        let diag_layer_limit = if diag { Self::diag_layer_limit() } else { 0 };
        let diag_moe_detail = std::env::var("KRASIS_PREFILL_DIAG_MOE_DETAIL").is_ok();
        let debug_prefill = prefill_debug_enabled();

        // Dynamic chunk sizing: use largest clean divisor that fits in scratch buffers.
        // This minimises the number of chunk passes (and therefore MoE DMA repetitions).
        let max_chunk = if self.config.prefill_chunk_size > 0 {
            self.config.prefill_chunk_size.min(self.scratch.max_tokens)
        } else {
            self.scratch.max_tokens
        };
        let (chunk_size, num_chunks) = if total_m <= max_chunk {
            // Single chunk — best case, zero redundant DMA
            (total_m, 1)
        } else {
            // Clean divisor: N equal-sized chunks, no runt
            let n = (total_m + max_chunk - 1) / max_chunk;
            let cs = (total_m + n - 1) / n;
            (cs, n)
        };
        if num_chunks > 1 {
            if stderr_debug_enabled() {
                eprintln!("[PREFILL] Dynamic chunking: {} tokens -> {} chunks of {} (max_chunk={})",
                    total_m, num_chunks, chunk_size, max_chunk);
            }
        }
        if debug_prefill {
            let has_linear_attn = self.config.la_num_k_heads > 0 || self.config.la_num_v_heads > 0
                || self.config.layer_types.iter().any(|&t| t == 3);
            let has_mamba = self.config.layer_types.iter().any(|&t| t == 1);
            eprintln!(
                "[PREFILL-DEBUG] run total_tokens={} scratch_max_tokens={} max_chunk={} chunk_size={} num_chunks={} marlin={} fused_moe={} flash_attn={} flash_attn_fp8kv={} linear_attn={} mamba={} gqa_heads={} kv_heads={}",
                total_m,
                self.scratch.max_tokens,
                max_chunk,
                chunk_size,
                num_chunks,
                self.kernels.marlin_mm.is_some(),
                self.kernels.fused_moe_fn.is_some(),
                self.kernels.flash_attn_fwd.is_some(),
                self.kernels.flash_attn_fwd_fp8kv.is_some(),
                has_linear_attn,
                has_mamba,
                self.config.num_q_heads,
                self.config.num_kv_heads,
            );
        }

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

        let use_fused = self.kernels.fused_moe_fn.is_some()
            && (self.d_fused_expert_w1_a.is_some() || self.d_expert_w1_ptrs.is_some())
            && std::env::var("KRASIS_SEQUENTIAL_MOE").is_err();

        // ═══════════════════════════════════════════════════════════
        //  Gate pre-scan + expert pinning (before main chunk loop)
        // ═══════════════════════════════════════════════════════════
        // Run gate GEMMs on embedded tokens to predict expert routing,
        // then pin the hottest experts in a VRAM pool for fast D2D access.
        // This eliminates most PCIe DMA for repeatedly-used experts.
        let pinning_enabled = use_fused
            && self.config.n_routed_experts > 0
            && std::env::var("KRASIS_NO_PINNING").is_err();
        if debug_prefill {
            let hcs_cached_experts = self.hcs_cached_expert_count();
            eprintln!(
                "[PREFILL-DEBUG] moe setup fused={} pointer_table={} pinning_enabled={} pinning_active={} pinning_pool_mb={:.1} prescan_layers={} hcs_cached_experts={} hcs_stride={}",
                use_fused,
                self.d_expert_w1_ptrs.is_some(),
                pinning_enabled,
                self.pinning_active,
                self.pinning_pool_bytes as f64 / (1024.0 * 1024.0),
                self.prescan_active_experts.len(),
                hcs_cached_experts,
                self.hcs_num_experts_per_layer,
            );
        }

        if pinning_enabled {
            // Embed the full prompt (or chunk-by-chunk for pre-scan)
            // We use chunk_size chunks to stay within scratch buffer limits
            let prescan_m = std::cmp::min(total_m, self.scratch.max_tokens);
            let prescan_tokens = &token_ids[..prescan_m];

            // Upload tokens and embed for pre-scan
            self.upload_tokens_with_offset(prescan_tokens, 0)
                .map_err(|e| format!("prescan upload: {}", e))?;
            self.launch_embedding(prescan_m)
                .map_err(|e| format!("prescan embedding: {}", e))?;

            // Run gate pre-scan on embedded tokens (routing info used by pointer table DMA)
            match self.gate_prescan(prescan_m, chunk_size, num_chunks) {
                Ok(prescan_counts) => {
                    let pinned_per_layer = self.allocate_pinning_pool(&prescan_counts)
                        .map_err(|e| format!("pinning pool allocation: {}", e))?;
                    if debug_prefill {
                        let active_layers = prescan_counts.len();
                        let total_active: usize = prescan_counts.iter()
                            .map(|layer| layer.iter().filter(|&&cnt| cnt > 0).count())
                            .sum();
                        let avg_active = if active_layers > 0 {
                            total_active as f64 / active_layers as f64
                        } else {
                            0.0
                        };
                        eprintln!(
                            "[PREFILL-DEBUG] prescan tokens={} active_layers={} avg_active_experts_per_layer={:.1} pinned_per_layer={} pinning_active={} pinning_pool_mb={:.1}",
                            prescan_m,
                            active_layers,
                            avg_active,
                            pinned_per_layer,
                            self.pinning_active,
                            self.pinning_pool_bytes as f64 / (1024.0 * 1024.0),
                        );
                    }
                }
                Err(e) => {
                    if stderr_debug_enabled() {
                        eprintln!("[PREFILL] Gate pre-scan failed (non-fatal): {}", e);
                    }
                }
            }
        }

        // Preload first MoE layer's experts into fused buffer A.
        // Skip in pointer table mode (forward_moe_fused builds tables per-layer).
        // With pinning active, use selective DMA (pinned D2D + cold H2D).
        // Without pinning, use bulk DMA (4 large contiguous transfers).
        if use_fused && self.d_expert_w1_ptrs.is_none() {
            for i in 0..num_hidden_layers {
                if self.layer_weights[i].moe_gate_ptr != 0 {
                    if let Some(moe_idx) = self.layer_weights[i].moe_layer_idx {
                        if let Some(Some(moe_data)) = self.moe_layers.get(moe_idx) {
                            let w1_base = *self.d_fused_expert_w1_a.as_ref().unwrap().device_ptr();
                            let w1s_base = *self.d_fused_expert_w1s_a.as_ref().unwrap().device_ptr();
                            let w2_base = *self.d_fused_expert_w2_a.as_ref().unwrap().device_ptr();
                            let w2s_base = *self.d_fused_expert_w2s_a.as_ref().unwrap().device_ptr();

                            // Find which MoE layer index this corresponds to for pinning lookup
                            let mi = (0..self.config.num_hidden_layers)
                                .filter(|&j| self.layer_weights[j].moe_gate_ptr != 0)
                                .position(|j| j == i);

                            if let Some(mi) = mi {
                                // Use pre-scan data or HCS for selective DMA
                                let active = if !self.prescan_active_experts.is_empty()
                                    && mi < self.prescan_active_experts.len()
                                    && !self.prescan_active_experts[mi].is_empty()
                                {
                                    self.prescan_active_experts[mi][0].clone()
                                } else {
                                    // No pre-scan data: transfer all experts
                                    (0..self.config.n_routed_experts).collect()
                                };
                                if !self.prescan_active_experts.is_empty() || self.hcs_num_experts_per_layer > 0 {
                                    let (h, p, c) = self.selective_dma_layer(
                                        moe_data, i, mi,
                                        w1_base, w1s_base, w2_base, w2s_base,
                                        &active)?;
                                    if stderr_debug_enabled() || debug_prefill {
                                        eprintln!(
                                            "[PREFILL-DEBUG] first_moe_layer={} active_experts={} selective_dma hcs={} pinned={} cold={} pinning_active={} pointer_table={}",
                                            i,
                                            active.len(),
                                            h,
                                            p,
                                            c,
                                            self.pinning_active,
                                            self.d_expert_w1_ptrs.is_some(),
                                        );
                                    }
                                } else {
                                    self.bulk_dma_layer(moe_data, w1_base, w1s_base, w2_base, w2s_base)?;
                                    let mb = (moe_data.bulk_w13p.1 + moe_data.bulk_w13s.1
                                        + moe_data.bulk_w2p.1 + moe_data.bulk_w2s.1) as f64 / 1e6;
                                    if stderr_debug_enabled() {
                                        eprintln!("[PREFILL] Preloading first MoE layer {} ({:.1} MB) on copy_stream",
                                            i, mb);
                                    }
                                }
                            } else {
                                self.bulk_dma_layer(moe_data, w1_base, w1s_base, w2_base, w2s_base)?;
                            }

                            unsafe {
                                cuda_sys::lib().cuEventRecord(self.dma_event, self.copy_stream);
                            }
                            self.preloaded_moe_layer = Some(i);
                            self.fused_expert_buf_cur = 0;
                        }
                    }
                    break;
                }
            }
        }

        // Per-component timing (KRASIS_PREFILL_TIMING=1)
        let timing = std::env::var("KRASIS_PREFILL_TIMING").is_ok();
        // Enable GQA sub-timing and reset accumulators
        self.gqa_timing_enabled.set(timing);
        if timing {
            self.t_gqa_proj.set(0.0);
            self.t_gqa_norm.set(0.0);
            self.t_gqa_rope.set(0.0);
            self.t_gqa_kv_prep.set(0.0);
            self.t_gqa_fa2.set(0.0);
            self.t_gqa_gate.set(0.0);
            self.t_gqa_oproj.set(0.0);
            self.gqa_fa2_calls.set(0);
            self.gqa_fp8_calls.set(0);
            self.gqa_bf16_calls.set(0);
            self.t_la_proj.set(0.0);
            self.t_la_uninterleave.set(0.0);
            self.t_la_conv.set(0.0);
            self.t_la_prep.set(0.0);
            self.t_la_convert.set(0.0);
            self.t_la_fla.set(0.0);
            self.t_la_postfla.set(0.0);
            self.t_la_norm.set(0.0);
            self.t_la_oproj.set(0.0);
            self.t_moe_gate.set(0.0);
            self.t_moe_dma.set(0.0);
            self.t_moe_w1.set(0.0);
            self.t_moe_w2.set(0.0);
            self.t_moe_scatter.set(0.0);
            self.t_moe_shared.set(0.0);
        }
        let mut t_norm_ms = 0.0f64;
        let mut t_attn_ms = 0.0f64;
        let mut t_gqa_ms = 0.0f64;
        let mut t_la_ms = 0.0f64;
        let mut t_moe_ms = 0.0f64;
        let mut t_embed_ms = 0.0f64;
        let mut t_other_ms = 0.0f64;

        for chunk_idx in 0..num_chunks {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = std::cmp::min(chunk_start + chunk_size, total_m);
            let m = chunk_end - chunk_start;
            let chunk_tokens = &token_ids[chunk_start..chunk_end];
            let chunk_t0 = if debug_prefill { Some(Instant::now()) } else { None };

            if m > self.scratch.max_tokens {
                return Err(format!("Chunk {} tokens > scratch {}", m, self.scratch.max_tokens));
            }

            self.run_prefill_chunk_hcs_guard(m, chunk_idx, num_chunks)?;

            // 1. Upload token IDs and positions for this chunk
            let tc0 = Instant::now();
            self.upload_tokens_with_offset(chunk_tokens, chunk_start)
                .map_err(|e| format!("upload_tokens chunk {}: {}", chunk_idx, e))?;

            // 2. Embedding lookup
            self.launch_embedding(m)
                .map_err(|e| format!("embedding chunk {} m={}: {}", chunk_idx, m, e))?;
            if timing {
                self.stream_sync()?;
                t_embed_ms += tc0.elapsed().as_secs_f64() * 1000.0;
            }

            if diag && chunk_idx == 0 {
                eprintln!("[DIAG] === Prefill diagnostic: m={} positions={:?} layers=0..{} ===",
                    m, diag_positions, diag_layer_limit);
                // Log input token IDs for verification against reference
                let id_str: Vec<String> = chunk_tokens.iter().map(|t| t.to_string()).collect();
                eprintln!("[DIAG] input_token_ids [{}]: {}", chunk_tokens.len(), id_str.join(", "));
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
                let tn0 = Instant::now();
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
                if timing { self.stream_sync()?; t_norm_ms += tn0.elapsed().as_secs_f64() * 1000.0; }


                // Mixer
                let ta0 = Instant::now();
                match layer_type {
                    0 => self.forward_gqa_chunked(layer_idx, m, chunk_start, true)
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
                if timing {
                    self.stream_sync()?;
                    let attn_elapsed = ta0.elapsed().as_secs_f64() * 1000.0;
                    t_attn_ms += attn_elapsed;
                    match layer_type { 0 => t_gqa_ms += attn_elapsed, 3 => t_la_ms += attn_elapsed, _ => {} }
                }

                if diag && chunk_idx == 0 && layer_idx < diag_layer_limit {
                    let lt_name = match layer_type { 0 => "gqa", 1 => "mamba2", 3 => "la", _ => "?" };
                    self.diag_print_norms(
                        &format!("layer{:02}_{}_mixer", layer_idx, lt_name),
                        *self.scratch.d_attn_out.device_ptr(), &diag_positions, h);
                }

                // Post-attention RMSNorm
                let tn1 = Instant::now();
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
                if timing { self.stream_sync()?; t_norm_ms += tn1.elapsed().as_secs_f64() * 1000.0; }

                // DIAG: d_hidden before MLP (after post-attn-norm = MoE input)
                if diag && chunk_idx == 0 && layer_idx < diag_layer_limit {
                    self.diag_print_norms(
                        &format!("layer{:02}_pre_mlp", layer_idx),
                        *self.scratch.d_hidden.device_ptr(), &diag_positions, h);
                }

                // MLP (dense or MoE)
                let tm0 = Instant::now();
                if self.layer_weights[layer_idx].moe_gate_ptr != 0 {
                    if diag && layer_idx < diag_layer_limit {
                        eprintln!("[DIAG] layer{:02} -> forward_moe (gate_ptr=0x{:x})",
                            layer_idx, self.layer_weights[layer_idx].moe_gate_ptr);
                    }
                    self.forward_moe(layer_idx, m)
                        .map_err(|e| format!("moe layer {}: {}", layer_idx, e))?;
                    // Double-buffer preload is now inside forward_moe_fused itself,
                    // so DMA(N+1) overlaps with compute(N) on the GPU.
                } else if self.layer_weights[layer_idx].shared_w1.is_some() {
                    if diag && layer_idx < diag_layer_limit {
                        eprintln!("[DIAG] layer{:02} -> forward_dense_mlp", layer_idx);
                    }
                    self.forward_dense_mlp(layer_idx, m)?;
                } else if diag && layer_idx < diag_layer_limit {
                    eprintln!("[DIAG] layer{:02} -> NO MLP (no gate, no shared_w1)", layer_idx);
                }
                if timing { self.stream_sync()?; t_moe_ms += tm0.elapsed().as_secs_f64() * 1000.0; }

                if diag && chunk_idx == 0 && layer_idx < diag_layer_limit {
                    self.diag_print_norms(
                        &format!("layer{:02}_mlp", layer_idx),
                        *self.scratch.d_hidden.device_ptr(), &diag_positions, h);
                }

                // DIAG: per-layer residual norm at last position
                if diag && chunk_idx == 0 {
                    let last_pos_norm = self.diag_l2_norm(
                        *self.scratch.d_residual.device_ptr(), m - 1, h, h);
                    let lt_name = match layer_type {
                        0 => "GQA", 1 => "Mamba2", 2 => "Pass", 3 => "LA", _ => "?",
                    };
                    let has_moe = self.layer_weights[layer_idx].moe_gate_ptr != 0;
                    eprintln!("[DIAG] L{:02} {} moe={} residual_lastpos={:.4}",
                        layer_idx, lt_name, has_moe, last_pos_norm);
                }

            }
            if let Some(chunk_t0) = chunk_t0 {
                let chunk_ms = chunk_t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "[PREFILL-DEBUG] chunk idx={} start={} end={} tokens={} ms={:.1} tok_per_s={:.1}",
                    chunk_idx,
                    chunk_start,
                    chunk_end,
                    m,
                    chunk_ms,
                    m as f64 / (chunk_ms / 1000.0),
                );
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

        // Free pinning pool — VRAM is released for decode HCS allocation
        if self.pinning_active {
            if self.pinning_pool_ptr != 0 {
                unsafe { cuda_sys::lib().cuMemFree_v2(self.pinning_pool_ptr); }
                self.pinning_pool_ptr = 0;
                self.pinning_pool_bytes = 0;
            }
            self.pinned_expert_offsets.clear();
            self.pinning_active = false;
            self.prescan_active_experts.clear();
        }

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        log::info!("Rust prefill: {} tok in {:.1}ms ({:.0} tok/s)", total_m, ms, total_m as f64 / (ms / 1000.0));

        if timing {
            let total = t_embed_ms + t_norm_ms + t_attn_ms + t_moe_ms + t_other_ms;
            let unaccounted = ms - total;
            eprintln!("[PREFILL-TIMING] {} tokens, {:.1}ms total ({:.0} tok/s)", total_m, ms, total_m as f64 / (ms / 1000.0));
            eprintln!("[PREFILL-TIMING]   embed:  {:>8.1}ms ({:>5.1}%)", t_embed_ms, t_embed_ms / ms * 100.0);
            eprintln!("[PREFILL-TIMING]   norm:   {:>8.1}ms ({:>5.1}%)", t_norm_ms, t_norm_ms / ms * 100.0);
            eprintln!("[PREFILL-TIMING]   attn:   {:>8.1}ms ({:>5.1}%)  [gqa: {:.1}ms, la: {:.1}ms]", t_attn_ms, t_attn_ms / ms * 100.0, t_gqa_ms, t_la_ms);
            // GQA sub-component breakdown
            let gp = self.t_gqa_proj.get();
            let gn = self.t_gqa_norm.get();
            let gr = self.t_gqa_rope.get();
            let gk = self.t_gqa_kv_prep.get();
            let gf = self.t_gqa_fa2.get();
            let gg = self.t_gqa_gate.get();
            let go = self.t_gqa_oproj.get();
            let ga = gp + gn + gr + gk + gf + gg + go;
            let fc = self.gqa_fa2_calls.get();
            let fp8c = self.gqa_fp8_calls.get();
            let bf16c = self.gqa_bf16_calls.get();
            if ga > 0.0 {
                eprintln!("[PREFILL-TIMING]     gqa breakdown ({} FA2 calls: {} bf16, {} fp8):",
                    fc, bf16c, fp8c);
                eprintln!("[PREFILL-TIMING]       proj:    {:>8.1}ms ({:>5.1}% of gqa)", gp, if t_gqa_ms > 0.0 { gp / t_gqa_ms * 100.0 } else { 0.0 });
                eprintln!("[PREFILL-TIMING]       qknorm:  {:>8.1}ms ({:>5.1}% of gqa)", gn, if t_gqa_ms > 0.0 { gn / t_gqa_ms * 100.0 } else { 0.0 });
                eprintln!("[PREFILL-TIMING]       rope:    {:>8.1}ms ({:>5.1}% of gqa)", gr, if t_gqa_ms > 0.0 { gr / t_gqa_ms * 100.0 } else { 0.0 });
                eprintln!("[PREFILL-TIMING]       kv_prep: {:>8.1}ms ({:>5.1}% of gqa)", gk, if t_gqa_ms > 0.0 { gk / t_gqa_ms * 100.0 } else { 0.0 });
                eprintln!("[PREFILL-TIMING]       fa2:     {:>8.1}ms ({:>5.1}% of gqa)  [{:.1}ms/call avg]",
                    gf, if t_gqa_ms > 0.0 { gf / t_gqa_ms * 100.0 } else { 0.0 },
                    if fc > 0 { gf / fc as f64 } else { 0.0 });
                eprintln!("[PREFILL-TIMING]       gate:    {:>8.1}ms ({:>5.1}% of gqa)", gg, if t_gqa_ms > 0.0 { gg / t_gqa_ms * 100.0 } else { 0.0 });
                eprintln!("[PREFILL-TIMING]       o_proj:  {:>8.1}ms ({:>5.1}% of gqa)", go, if t_gqa_ms > 0.0 { go / t_gqa_ms * 100.0 } else { 0.0 });
            }
            // LA sub-component breakdown
            let lp = self.t_la_proj.get();
            let lu = self.t_la_uninterleave.get();
            let lc = self.t_la_conv.get();
            let lr = self.t_la_prep.get();
            let lv = self.t_la_convert.get();
            let lf = self.t_la_fla.get();
            let lo = self.t_la_postfla.get();
            let ln = self.t_la_norm.get();
            let lq = self.t_la_oproj.get();
            let la = lp + lu + lc + lr + lv + lf + lo + ln + lq;
            if la > 0.0 {
                eprintln!("[PREFILL-TIMING]     la breakdown (36 layers):");
                eprintln!("[PREFILL-TIMING]       proj:    {:>8.1}ms ({:>5.1}% of la)", lp, lp / t_la_ms * 100.0);
                eprintln!("[PREFILL-TIMING]       uninter: {:>8.1}ms ({:>5.1}% of la)", lu, lu / t_la_ms * 100.0);
                eprintln!("[PREFILL-TIMING]       conv:    {:>8.1}ms ({:>5.1}% of la)", lc, lc / t_la_ms * 100.0);
                eprintln!("[PREFILL-TIMING]       prep:    {:>8.1}ms ({:>5.1}% of la)", lr, lr / t_la_ms * 100.0);
                eprintln!("[PREFILL-TIMING]       convert: {:>8.1}ms ({:>5.1}% of la)", lv, lv / t_la_ms * 100.0);
                eprintln!("[PREFILL-TIMING]       fla:     {:>8.1}ms ({:>5.1}% of la)", lf, lf / t_la_ms * 100.0);
                eprintln!("[PREFILL-TIMING]       postfla: {:>8.1}ms ({:>5.1}% of la)", lo, lo / t_la_ms * 100.0);
                eprintln!("[PREFILL-TIMING]       norm:    {:>8.1}ms ({:>5.1}% of la)", ln, ln / t_la_ms * 100.0);
                eprintln!("[PREFILL-TIMING]       o_proj:  {:>8.1}ms ({:>5.1}% of la)", lq, lq / t_la_ms * 100.0);
            }
            eprintln!("[PREFILL-TIMING]   moe:    {:>8.1}ms ({:>5.1}%)", t_moe_ms, t_moe_ms / ms * 100.0);
            // MoE sub-component breakdown
            let mg = self.t_moe_gate.get();
            let md = self.t_moe_dma.get();
            let mw1 = self.t_moe_w1.get();
            let mw2 = self.t_moe_w2.get();
            let msc = self.t_moe_scatter.get();
            let msh = self.t_moe_shared.get();
            let ma = mg + md + mw1 + mw2 + msc + msh;
            if ma > 0.0 {
                eprintln!("[PREFILL-TIMING]     moe breakdown (48 layers):");
                eprintln!("[PREFILL-TIMING]       gate:    {:>8.1}ms ({:>5.1}% of moe)", mg, if t_moe_ms > 0.0 { mg / t_moe_ms * 100.0 } else { 0.0 });
                eprintln!("[PREFILL-TIMING]       dma:     {:>8.1}ms ({:>5.1}% of moe)", md, if t_moe_ms > 0.0 { md / t_moe_ms * 100.0 } else { 0.0 });
                eprintln!("[PREFILL-TIMING]       w1+act:  {:>8.1}ms ({:>5.1}% of moe)", mw1, if t_moe_ms > 0.0 { mw1 / t_moe_ms * 100.0 } else { 0.0 });
                eprintln!("[PREFILL-TIMING]       w2:      {:>8.1}ms ({:>5.1}% of moe)", mw2, if t_moe_ms > 0.0 { mw2 / t_moe_ms * 100.0 } else { 0.0 });
                eprintln!("[PREFILL-TIMING]       scatter: {:>8.1}ms ({:>5.1}% of moe)", msc, if t_moe_ms > 0.0 { msc / t_moe_ms * 100.0 } else { 0.0 });
                eprintln!("[PREFILL-TIMING]       shared:  {:>8.1}ms ({:>5.1}% of moe)", msh, if t_moe_ms > 0.0 { msh / t_moe_ms * 100.0 } else { 0.0 });
            }
            eprintln!("[PREFILL-TIMING]   other:  {:>8.1}ms ({:>5.1}%)", unaccounted, unaccounted / ms * 100.0);
        }

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

        // Zero LA recurrent + conv state for fresh prefill
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

        // 1. Upload token IDs and positions
        self.upload_tokens_with_offset(token_ids, 0)?;

        // 2. Embedding lookup
        self.launch_embedding(m)?;

        // Preload first MoE layer (same as run_prefill)
        // Skip in pointer table mode (forward_moe_fused builds tables per-layer)
        let use_fused = self.kernels.fused_moe_fn.is_some()
            && (self.d_fused_expert_w1_a.is_some() || self.d_expert_w1_ptrs.is_some())
            && std::env::var("KRASIS_SEQUENTIAL_MOE").is_err();
        if use_fused && self.d_expert_w1_ptrs.is_none() {
            for i in 0..num_hidden_layers {
                if self.layer_weights[i].moe_gate_ptr != 0 {
                    if let Some(moe_idx) = self.layer_weights[i].moe_layer_idx {
                        if let Some(Some(moe_data)) = self.moe_layers.get(moe_idx) {
                            let w1_base = *self.d_fused_expert_w1_a.as_ref().unwrap().device_ptr();
                            let w1s_base = *self.d_fused_expert_w1s_a.as_ref().unwrap().device_ptr();
                            let w2_base = *self.d_fused_expert_w2_a.as_ref().unwrap().device_ptr();
                            let w2s_base = *self.d_fused_expert_w2s_a.as_ref().unwrap().device_ptr();
                            self.bulk_dma_layer(moe_data, w1_base, w1s_base, w2_base, w2s_base)?;
                            unsafe {
                                cuda_sys::lib().cuEventRecord(self.dma_event, self.copy_stream);
                            }
                            self.preloaded_moe_layer = Some(i);
                            self.fused_expert_buf_cur = 0;
                        }
                    }
                    break;
                }
            }
        }

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
                0 => self.forward_gqa_chunked(layer_idx, m, 0, false)?,
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
                if self.kernels.fused_moe_fn.is_some()
                    && self.d_fused_expert_w1_a.is_some()
                    && self.d_expert_w1_ptrs.is_none()  // skip preload in pointer table mode
                {
                    self.preload_next_moe_layer(layer_idx, num_hidden_layers)?;
                }
            } else if self.layer_weights[layer_idx].shared_w1.is_some() {
                self.forward_dense_mlp(layer_idx, m)?;
            }

            // DIAG: per-layer hidden state norm at last position
            if std::env::var("KRASIS_PREFILL_DIAG").is_ok() {
                let last_pos_norm = self.diag_l2_norm(
                    *self.scratch.d_residual.device_ptr(), m - 1, h, h);
                let lt_name = match layer_type {
                    0 => "GQA",
                    1 => "Mamba2",
                    2 => "Pass",
                    3 => "LA",
                    _ => "?",
                };
                let has_moe = self.layer_weights[layer_idx].moe_gate_ptr != 0;
                let has_mlp = self.layer_weights[layer_idx].shared_w1.is_some();
                eprintln!("[DIAG] L{:02} {} moe={} mlp={} residual_lastpos_norm={:.4}",
                    layer_idx, lt_name, has_moe, has_mlp, last_pos_norm);
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
                        pos_hidden as *const std::ffi::c_void,
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
        let ws_len = self.config.sms * MARLIN_MAX_LOCK_SLOTS_PER_SM;
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

    // ── Custom Tiled Attention (fallback when FA2 not available or cross-chunk) ──

    fn launch_custom_tiled_attn(
        &self, attn_out: u64, q: u64, k: u64, v: u64,
        layer_k_ptr: u64, layer_v_ptr: u64,
        m: usize, start_pos: usize,
        cfg: &PrefillModelConfig,
    ) -> Result<(), String> {
        let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
        let fa_br = 16u32;
        let fa_bc = 64usize;
        let grid_x = ((m as u32) + fa_br - 1) / fa_br;
        let hd = cfg.head_dim;
        let smem = (fa_br as usize * hd * 2
            + fa_bc * hd * 2
            + fa_bc * hd * 2
            + fa_br as usize * fa_bc * 4
            + fa_br as usize * fa_bc * 2
            + 16 * 16 * 4
        ) as u32;
        let kv_stride = (cfg.num_kv_heads * cfg.head_dim) as i32;
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
        Ok(())
    }

    // ── GQA Attention ──

    fn forward_gqa(
        &self, layer_idx: usize, m: usize,
    ) -> Result<(), String> {
        self.forward_gqa_chunked(layer_idx, m, 0, true)
    }

    fn forward_gqa_chunked(
        &self, layer_idx: usize, m: usize, start_pos: usize, capture_kv_cache: bool,
    ) -> Result<(), String> {
        let cfg = &self.config;
        let lw = &self.layer_weights[layer_idx];
        let gt = self.gqa_timing_enabled.get();

        let hidden = *self.scratch.d_hidden.device_ptr();
        let q = *self.scratch.d_q.device_ptr();
        let k = *self.scratch.d_k.device_ptr();
        let v = *self.scratch.d_v.device_ptr();
        let attn_out = *self.scratch.d_attn_out.device_ptr();

        // Q/K/V projections (Marlin or cuBLAS BF16)
        let gt0 = Instant::now();
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

        if gt { self.stream_sync()?; self.t_gqa_proj.set(self.t_gqa_proj.get() + gt0.elapsed().as_secs_f64() * 1000.0); }

        // QK LayerNorm: per-head RMSNorm on Q and K after projection, before RoPE
        // Q is [m, num_q_heads, head_dim], K is [m, num_kv_heads, head_dim]
        // Treat as [m*num_heads, head_dim] rows — existing rmsnorm kernel handles this.
        let gt1 = Instant::now();
        if lw.q_norm_ptr != 0 {
            self.launch_rmsnorm(q, q, lw.q_norm_ptr, m * cfg.num_q_heads, cfg.head_dim)?;
        }
        if lw.k_norm_ptr != 0 {
            self.launch_rmsnorm(k, k, lw.k_norm_ptr, m * cfg.num_kv_heads, cfg.head_dim)?;
        }

        if gt { self.stream_sync()?; self.t_gqa_norm.set(self.t_gqa_norm.get() + gt1.elapsed().as_secs_f64() * 1000.0); }

        // Look up per-layer KV cache pointers (needed for both attention and cache append)
        let layer_k_ptr = if layer_idx < self.kv_k_ptrs.len() { self.kv_k_ptrs[layer_idx] } else { 0 };
        let layer_v_ptr = if layer_idx < self.kv_v_ptrs.len() { self.kv_v_ptrs[layer_idx] } else { 0 };

        // RoPE (uses rope_half_dim for partial rotary support)
        let gt2 = Instant::now();
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

        if gt { self.stream_sync()?; self.t_gqa_rope.set(self.t_gqa_rope.get() + gt2.elapsed().as_secs_f64() * 1000.0); }

        // Attention: use vendored FlashAttention-2 when available (start_pos==0),
        // otherwise fall back to custom tiled kernel for cross-chunk KV cache support.
        let mut kv_cache_already_stored = false; // Set true by FP8 cross-chunk path
        if let Some(fa2_fwd) = self.kernels.flash_attn_fwd {
            if start_pos == 0 {
                // FA2 varlen forward: Q/K/V are [total_q, heads, head_dim] contiguous BF16.
                // For single-sequence prefill: batch=1, cu_seqlens_q=[0,m], cu_seqlens_k=[0,m].
                let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
                let lse_ptr = self.scratch.d_fa2_lse.as_ref()
                    .map(|s| *s.device_ptr() as *mut std::ffi::c_void)
                    .unwrap_or(std::ptr::null_mut());

                // Build cu_seqlens on host and upload (2 ints: [0, m])
                let cu_data: [i32; 2] = [0, m as i32];
                let cu_ptr = *self.scratch.d_workspace.device_ptr();
                unsafe {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        cu_ptr, cu_data.as_ptr() as *const _, 8, self.stream,
                    );
                }

                if gt { self.stream_sync()?; }
                let gt_fa2 = Instant::now();
                let ret = unsafe {
                    fa2_fwd(
                        q as *const _,
                        k as *const _,
                        v as *const _,
                        attn_out as *mut _,
                        lse_ptr,
                        cu_ptr as *const _,  // cu_seqlens_q
                        cu_ptr as *const _,  // cu_seqlens_k (same: self-attention)
                        1,                   // batch_size
                        m as i32,            // seqlen_q (max)
                        m as i32,            // seqlen_k (max)
                        cfg.num_q_heads as i32,
                        cfg.num_kv_heads as i32,
                        cfg.head_dim as i32,
                        m as i32,            // total_q
                        m as i32,            // total_k
                        scale,
                        1,                   // is_causal
                        1,                   // unpadded_lse
                        self.stream as *mut _,
                    )
                };
                if ret != 0 {
                    return Err(format!("FlashAttention-2 forward failed with code {}", ret));
                }
                if gt {
                    self.stream_sync()?;
                    self.t_gqa_fa2.set(self.t_gqa_fa2.get() + gt_fa2.elapsed().as_secs_f64() * 1000.0);
                    self.gqa_fa2_calls.set(self.gqa_fa2_calls.get() + 1);
                    self.gqa_bf16_calls.set(self.gqa_bf16_calls.get() + 1);
                }
            } else if self.kv_format == 2
                && layer_idx < self.kv_k_radius_ptrs.len()
                && self.kv_k_radius_ptrs[layer_idx] != 0
            {
                // Cross-chunk attention: reconstruct cached Polar4 K/V to BF16,
                // concat current BF16 chunk, then run standard BF16 FA2.
                let kv_stride = cfg.num_kv_heads * cfg.head_dim;
                let total_kv = start_pos + m;
                let kv_buf_bytes = (total_kv * kv_stride * 2) as u64; // BF16
                let scratch_bytes =
                    (self.scratch.d_fp32_scratch.len * std::mem::size_of::<f32>()) as u64;
                let required_bytes = kv_buf_bytes * 2;
                if required_bytes > scratch_bytes {
                    return Err(format!(
                        "Polar4 cross-chunk KV staging needs {} bytes, fp32 scratch has {}",
                        required_bytes, scratch_bytes
                    ));
                }
                let fp32_scratch = *self.scratch.d_fp32_scratch.device_ptr();
                let full_k_bf16 = fp32_scratch;
                let full_v_bf16 = fp32_scratch + kv_buf_bytes;

                let gt_dequant = Instant::now();
                let grid = (total_kv as u32, 1, 1);
                let num_blocks = (kv_stride / 16) as u32;
                unsafe {
                    let mut a0 = full_k_bf16;
                    let mut a1 = self.kv_k_radius_ptrs[layer_idx];
                    let mut a2 = self.kv_k_angles_ptrs[layer_idx];
                    let mut a3 = k;
                    let mut a4 = start_pos as i32;
                    let mut a5 = m as i32;
                    let mut a6 = kv_stride as i32;
                    launch(self.kernels.kv_dequant_concat_polar4,
                        grid, (num_blocks, 1, 1), 0, self.stream,
                        &mut [&mut a0 as *mut _ as *mut std::ffi::c_void,
                              &mut a1 as *mut _ as *mut std::ffi::c_void,
                              &mut a2 as *mut _ as *mut std::ffi::c_void,
                              &mut a3 as *mut _ as *mut std::ffi::c_void,
                              &mut a4 as *mut _ as *mut std::ffi::c_void,
                              &mut a5 as *mut _ as *mut std::ffi::c_void,
                              &mut a6 as *mut _ as *mut std::ffi::c_void])?;
                    let mut b0 = full_v_bf16;
                    let mut b1 = self.kv_v_radius_ptrs[layer_idx];
                    let mut b2 = self.kv_v_angles_ptrs[layer_idx];
                    let mut b3 = v;
                    launch(self.kernels.kv_dequant_concat_polar4,
                        grid, (num_blocks, 1, 1), 0, self.stream,
                        &mut [&mut b0 as *mut _ as *mut std::ffi::c_void,
                              &mut b1 as *mut _ as *mut std::ffi::c_void,
                              &mut b2 as *mut _ as *mut std::ffi::c_void,
                              &mut b3 as *mut _ as *mut std::ffi::c_void,
                              &mut a4 as *mut _ as *mut std::ffi::c_void,
                              &mut a5 as *mut _ as *mut std::ffi::c_void,
                              &mut a6 as *mut _ as *mut std::ffi::c_void])?;
                }

                if gt {
                    self.stream_sync()?;
                    self.t_gqa_kv_prep.set(self.t_gqa_kv_prep.get() + gt_dequant.elapsed().as_secs_f64() * 1000.0);
                }

                let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
                let lse_ptr = self.scratch.d_fa2_lse.as_ref()
                    .map(|s| *s.device_ptr() as *mut std::ffi::c_void)
                    .unwrap_or(std::ptr::null_mut());
                let cu_q_data: [i32; 2] = [0, m as i32];
                let cu_k_data: [i32; 2] = [0, total_kv as i32];
                let cu_ptr = *self.scratch.d_workspace.device_ptr();
                let cu_k_ptr = cu_ptr + 8;
                unsafe {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        cu_ptr, cu_q_data.as_ptr() as *const _, 8, self.stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        cu_k_ptr, cu_k_data.as_ptr() as *const _, 8, self.stream);
                }

                let gt_fa2_bf16 = Instant::now();
                let ret = unsafe {
                    fa2_fwd(
                        q as *const _,
                        full_k_bf16 as *const _,
                        full_v_bf16 as *const _,
                        attn_out as *mut _,
                        lse_ptr,
                        cu_ptr as *const _,
                        cu_k_ptr as *const _,
                        1,
                        m as i32,
                        total_kv as i32,
                        cfg.num_q_heads as i32,
                        cfg.num_kv_heads as i32,
                        cfg.head_dim as i32,
                        m as i32,
                        total_kv as i32,
                        scale,
                        1,
                        1,
                        self.stream as *mut _,
                    )
                };
                if ret != 0 {
                    return Err(format!("FlashAttention-2 Polar4 cross-chunk forward failed: {}", ret));
                }
                if gt {
                    self.stream_sync()?;
                    self.t_gqa_fa2.set(self.t_gqa_fa2.get() + gt_fa2_bf16.elapsed().as_secs_f64() * 1000.0);
                    self.gqa_fa2_calls.set(self.gqa_fa2_calls.get() + 1);
                    self.gqa_bf16_calls.set(self.gqa_bf16_calls.get() + 1);
                }
            } else if layer_k_ptr != 0 && layer_v_ptr != 0 {
                // Cross-chunk attention: Q attends to all K/V (cached FP8 + current BF16)
                let kv_stride = cfg.num_kv_heads * cfg.head_dim;
                let total_kv = start_pos + m;

                // FP8 FA2: cp.async staging (FP8 -> staging -> BF16 smem).
                // Only used for head_dim <= 128 where kBlockN=64 staging fits in smem.
                // For head_dim > 128, staging forces kBlockN=32 which is slower than
                // the dequant+BF16 path (separate dequant kernel + standard BF16 FA2).
                let use_fp8_fa2 = self.kernels.flash_attn_fwd_fp8kv.is_some()
                    && cfg.head_dim <= 128;
                if let (true, Some(fa2_fp8)) = (use_fp8_fa2, self.kernels.flash_attn_fwd_fp8kv) {
                    // FP8 FA2 path: store current K/V to FP8 cache FIRST, then
                    // FA2 reads directly from cache (all FP8). No temp buffer needed.
                    let gt_kv_store = Instant::now();
                    let kv_stride_i32 = kv_stride as i32;
                    let kt = std::cmp::max(32, ((std::cmp::min(256, kv_stride) + 31) / 32) * 32) as u32;
                    let mut k0 = layer_k_ptr; let mut k1 = layer_v_ptr;
                    let mut k2 = k; let mut k3 = v;
                    let mut k4 = m as i32; let mut k5 = kv_stride_i32;
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
                    if gt {
                        self.stream_sync()?;
                        self.t_gqa_kv_prep.set(self.t_gqa_kv_prep.get() + gt_kv_store.elapsed().as_secs_f64() * 1000.0);
                    }

                    // FA2 FP8: Q (BF16) attends to K/V (FP8 in cache [0..total_kv])
                    let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
                    let lse_ptr = self.scratch.d_fa2_lse.as_ref()
                        .map(|s| *s.device_ptr() as *mut std::ffi::c_void)
                        .unwrap_or(std::ptr::null_mut());
                    let cu_q_data: [i32; 2] = [0, m as i32];
                    let cu_k_data: [i32; 2] = [0, total_kv as i32];
                    let cu_ptr = *self.scratch.d_workspace.device_ptr();
                    let cu_k_ptr = cu_ptr + 8;
                    unsafe {
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            cu_ptr, cu_q_data.as_ptr() as *const _, 8, self.stream);
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            cu_k_ptr, cu_k_data.as_ptr() as *const _, 8, self.stream);
                    }

                    let gt_fa2 = Instant::now();
                    // FP8 KV cache layout: [max_seq, num_kv_heads * head_dim], stride = kv_stride
                    let ret = unsafe {
                        fa2_fp8(
                            q as *const _,
                            layer_k_ptr as *const _,  // FP8 KV cache directly
                            layer_v_ptr as *const _,  // FP8 KV cache directly
                            attn_out as *mut _,
                            lse_ptr,
                            cu_ptr as *const _,    // cu_seqlens_q = [0, m]
                            cu_k_ptr as *const _,  // cu_seqlens_k = [0, total_kv]
                            1,                     // batch_size
                            m as i32,              // seqlen_q (max)
                            total_kv as i32,       // seqlen_k (max)
                            cfg.num_q_heads as i32,
                            cfg.num_kv_heads as i32,
                            cfg.head_dim as i32,
                            m as i32,              // total_q
                            total_kv as i32,       // total_k
                            scale,
                            1,                     // is_causal=1: FA2 auto-adjusts mask for seqlen_k > seqlen_q
                            1,                     // unpadded_lse
                            self.stream as *mut _,
                        )
                    };
                    if ret != 0 {
                        return Err(format!("FlashAttention-2 FP8 KV cross-chunk failed: {}", ret));
                    }
                    if gt {
                        self.stream_sync()?;
                        self.t_gqa_fa2.set(self.t_gqa_fa2.get() + gt_fa2.elapsed().as_secs_f64() * 1000.0);
                        self.gqa_fa2_calls.set(self.gqa_fa2_calls.get() + 1);
                        self.gqa_fp8_calls.set(self.gqa_fp8_calls.get() + 1);
                    }
                    kv_cache_already_stored = true;
                } else {
                    // Fallback: dequant FP8 cache + concat current BF16 K/V, then FA2 BF16
                    let kv_buf_bytes = (total_kv * kv_stride * 2) as u64; // BF16
                    let scratch_bytes =
                        (self.scratch.d_fp32_scratch.len * std::mem::size_of::<f32>()) as u64;
                    let required_bytes = kv_buf_bytes * 2;
                    if required_bytes > scratch_bytes {
                        return Err(format!(
                            "FP8 cross-chunk KV staging needs {} bytes, fp32 scratch has {}",
                            required_bytes, scratch_bytes
                        ));
                    }
                    let fp32_scratch = *self.scratch.d_fp32_scratch.device_ptr();
                    let full_k_bf16 = fp32_scratch;
                    let full_v_bf16 = fp32_scratch + kv_buf_bytes;

                    let gt_dequant = Instant::now();
                    let grid = (total_kv as u32, 1, 1);
                    let kv_threads = std::cmp::max(32, ((std::cmp::min(512, kv_stride) + 31) / 32) * 32) as u32;
                    unsafe {
                        let mut a0 = full_k_bf16; let mut a1 = layer_k_ptr; let mut a2 = k;
                        let mut a3 = start_pos as i32; let mut a4 = m as i32;
                        let mut a5 = kv_stride as i32;
                        launch(self.kernels.kv_dequant_concat,
                            grid, (kv_threads, 1, 1), 0, self.stream,
                            &mut [&mut a0 as *mut _ as *mut std::ffi::c_void,
                                  &mut a1 as *mut _ as *mut std::ffi::c_void,
                                  &mut a2 as *mut _ as *mut std::ffi::c_void,
                                  &mut a3 as *mut _ as *mut std::ffi::c_void,
                                  &mut a4 as *mut _ as *mut std::ffi::c_void,
                                  &mut a5 as *mut _ as *mut std::ffi::c_void])?;
                        let mut a0 = full_v_bf16; let mut a1 = layer_v_ptr; let mut a2 = v;
                        launch(self.kernels.kv_dequant_concat,
                            grid, (kv_threads, 1, 1), 0, self.stream,
                            &mut [&mut a0 as *mut _ as *mut std::ffi::c_void,
                                  &mut a1 as *mut _ as *mut std::ffi::c_void,
                                  &mut a2 as *mut _ as *mut std::ffi::c_void,
                                  &mut a3 as *mut _ as *mut std::ffi::c_void,
                                  &mut a4 as *mut _ as *mut std::ffi::c_void,
                                  &mut a5 as *mut _ as *mut std::ffi::c_void])?;
                    }

                    if gt {
                        self.stream_sync()?;
                        self.t_gqa_kv_prep.set(self.t_gqa_kv_prep.get() + gt_dequant.elapsed().as_secs_f64() * 1000.0);
                    }

                    let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
                    let lse_ptr = self.scratch.d_fa2_lse.as_ref()
                        .map(|s| *s.device_ptr() as *mut std::ffi::c_void)
                        .unwrap_or(std::ptr::null_mut());
                    let cu_q_data: [i32; 2] = [0, m as i32];
                    let cu_k_data: [i32; 2] = [0, total_kv as i32];
                    let cu_ptr = *self.scratch.d_workspace.device_ptr();
                    let cu_k_ptr = cu_ptr + 8;
                    unsafe {
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            cu_ptr, cu_q_data.as_ptr() as *const _, 8, self.stream);
                        cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                            cu_k_ptr, cu_k_data.as_ptr() as *const _, 8, self.stream);
                    }

                    let gt_fa2_bf16 = Instant::now();
                    let ret = unsafe {
                        fa2_fwd(
                            q as *const _,
                            full_k_bf16 as *const _,
                            full_v_bf16 as *const _,
                            attn_out as *mut _,
                            lse_ptr,
                            cu_ptr as *const _,
                            cu_k_ptr as *const _,
                            1,
                            m as i32,
                            total_kv as i32,
                            cfg.num_q_heads as i32,
                            cfg.num_kv_heads as i32,
                            cfg.head_dim as i32,
                            m as i32,
                            total_kv as i32,
                            scale,
                            1,                     // is_causal
                            1,
                            self.stream as *mut _,
                        )
                    };
                    if ret != 0 {
                        return Err(format!("FlashAttention-2 cross-chunk forward failed: {}", ret));
                    }
                    if gt {
                        self.stream_sync()?;
                        self.t_gqa_fa2.set(self.t_gqa_fa2.get() + gt_fa2_bf16.elapsed().as_secs_f64() * 1000.0);
                        self.gqa_fa2_calls.set(self.gqa_fa2_calls.get() + 1);
                        self.gqa_bf16_calls.set(self.gqa_bf16_calls.get() + 1);
                    }
                }
            } else {
                // Cross-chunk but no KV cache pointers: fallback to custom tiled kernel
                self.launch_custom_tiled_attn(
                    attn_out, q, k, v, layer_k_ptr, layer_v_ptr,
                    m, start_pos, cfg,
                )?;
            }
        } else {
            // No FA2 available: use custom tiled kernel
            self.launch_custom_tiled_attn(
                attn_out, q, k, v, layer_k_ptr, layer_v_ptr,
                m, start_pos, cfg,
            )?;
        }

        // KV cache append: BF16 K,V -> cache format into separate per-layer caches
        // Skip if cross-chunk path already stored K/V before attention.
        if capture_kv_cache && self.kv_format == 2 && layer_idx < self.kv_k_radius_ptrs.len()
            && self.kv_k_radius_ptrs[layer_idx] != 0
        {
            // Polar4: BF16 K,V -> 4-bit rotated polar format
            let kv_stride = (cfg.num_kv_heads * cfg.head_dim) as i32;
            let num_blocks = (kv_stride / 16) as u32;
            let mut p0 = self.kv_k_radius_ptrs[layer_idx];
            let mut p1 = self.kv_v_radius_ptrs[layer_idx];
            let mut p2 = self.kv_k_angles_ptrs[layer_idx];
            let mut p3 = self.kv_v_angles_ptrs[layer_idx];
            let mut p4 = k; let mut p5 = v;
            let mut p6 = m as i32; let mut p7 = kv_stride;
            let mut p8 = self.kv_max_seq as i32; let mut p9 = start_pos as i32;
            unsafe {
                launch(self.kernels.kv_cache_append_polar4,
                    (m as u32, 1, 1), (num_blocks, 1, 1), 0, self.stream,
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
        } else if capture_kv_cache && layer_k_ptr != 0 && layer_v_ptr != 0 && !kv_cache_already_stored {
            // FP8 E4M3: standard path
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
        let gt_gate = Instant::now();
        if lw.gqa_gated && gate_ptr != 0 {
            let q_dim = (cfg.num_q_heads * cfg.head_dim) as i32;
            let gate_threads = std::cmp::max(32, ((std::cmp::min(1024, q_dim as usize) + 31) / 32) * 32) as u32;
            let mut g0 = attn_out; let mut g1 = attn_out; let mut g2 = gate_ptr;
            let mut g3 = q_dim;
            unsafe {
                launch(self.kernels.sigmoid_mul,
                    (m as u32, 1, 1), (gate_threads, 1, 1), 0, self.stream,
                    &mut [
                        &mut g0 as *mut _ as *mut std::ffi::c_void,
                        &mut g1 as *mut _ as *mut std::ffi::c_void,
                        &mut g2 as *mut _ as *mut std::ffi::c_void,
                        &mut g3 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }
        if gt { self.stream_sync()?; self.t_gqa_gate.set(self.t_gqa_gate.get() + gt_gate.elapsed().as_secs_f64() * 1000.0); }

        // O projection (use scratch1 as temp to avoid input/output aliasing)
        let gt_oproj = Instant::now();
        let o_temp = *self.scratch.d_scratch1.device_ptr();
        self.la_gemm(attn_out, &lw.o_proj, &lw.o_proj_bf16, o_temp, m)?;
        self.memcpy_d2d(attn_out, o_temp, (m * cfg.hidden_size * 2) as u64)?;
        if gt { self.stream_sync()?; self.t_gqa_oproj.set(self.t_gqa_oproj.get() + gt_oproj.elapsed().as_secs_f64() * 1000.0); }

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
        // mut: reassigned after repeat_interleave to avoid redundant memcpy_d2d
        let mut la_q = *self.scratch.d_la_q.as_ref().ok_or("no la_q")?.device_ptr();
        let mut la_k = *self.scratch.d_la_k.as_ref().ok_or("no la_k")?.device_ptr();
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

        let lt = self.gqa_timing_enabled.get(); // reuse same env flag for LA timing
        let lt0 = Instant::now();
        let la_diag = std::env::var("KRASIS_PREFILL_DIAG").is_ok()
            && layer_idx < Self::diag_layer_limit();
        // 1. in_proj_qkvz GEMM: [M, hidden] -> [M, qkvz_dim] BF16
        self.la_gemm(hidden, &lw.la_in_proj_qkvz, &lw.la_in_proj_qkvz_bf16, proj_buf, m)?;
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
            // v out needs [M, value_dim] BF16. attn_out is only [M, hidden_size] which overflows
            // when value_dim > hidden_size (e.g. QCN: 4096 vs 2048). Use la_k_beta (FP32 buf,
            // ~411 MB) as temp -- it's not used until FLA step 3 (ai_fla), well after conv reads v.
            let v_bf16 = *self.scratch.d_la_k_beta.as_ref().expect("d_la_k_beta required for LA").device_ptr();
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
            if lt { self.stream_sync()?; self.t_la_proj.set(self.t_la_proj.get() + lt0.elapsed().as_secs_f64() * 1000.0); }

            // RMSNorm output goes to a temp buffer sized [M, nv*dv] since value_dim
            // may exceed hidden_size (QCN: 4096 vs 2048). FLA path uses la_v (free after FLA),
            // old FP32 path uses la_v_beta (free after gate/beta ops).
            let mut rmsnorm_out;

            // ── BF16 optimized pipeline (FLA path) ──
            // Fused kernels: conv outputs BF16 directly, gate_beta outputs BF16,
            // fused repeat+l2norm, no FP32/BF16 conversion overhead.
            if self.fla.is_some() {
                rmsnorm_out = la_v;  // [M, nv*dv] FP32 = 2x needed for BF16
                let fp32_scratch = *self.scratch.d_fp32_scratch.device_ptr();
                let lt1 = Instant::now();
                let conv_state = lw.la_conv_state_ptr;
                let conv_weight = lw.la_conv_weight_ptr;

                // Fused conv1d+SiLU: 3 BF16 inputs -> 3 BF16 outputs (single kernel)
                // Replaces: concat + conv + transpose + split (4 kernels, ~458ms -> ~100ms)
                // Conv output goes to la_q/la_k/la_v FP32 buffers used at BF16 (half capacity)
                let conv_q_bf16 = la_q;  // reuse FP32 buffer for BF16 (half the bytes)
                let conv_k_bf16 = la_k;
                let conv_v_bf16 = la_v;
                {
                    let ct = std::cmp::min(1024, ((conv_dim + 31) / 32) * 32) as u32;
                    let mut c0 = conv_q_bf16;
                    let mut c1 = conv_k_bf16;
                    let mut c2 = conv_v_bf16;
                    let mut c3 = conv_state;
                    let mut c4 = q_bf16;     // input from uninterleave
                    let mut c5 = k_bf16;
                    let mut c6 = v_bf16;
                    let mut c7 = conv_weight;
                    let mut c8 = m as i32;
                    let mut c9 = key_dim as i32;
                    let mut c10 = value_dim as i32;
                    let mut c11 = kernel_dim as i32;
                    unsafe {
                        launch(self.kernels.la_fused_conv1d_silu_bf16,
                            (m as u32, 1, 1), (ct, 1, 1), 0, self.stream,
                            &mut [
                                &mut c0 as *mut _ as *mut std::ffi::c_void,
                                &mut c1 as *mut _ as *mut std::ffi::c_void,
                                &mut c2 as *mut _ as *mut std::ffi::c_void,
                                &mut c3 as *mut _ as *mut std::ffi::c_void,
                                &mut c4 as *mut _ as *mut std::ffi::c_void,
                                &mut c5 as *mut _ as *mut std::ffi::c_void,
                                &mut c6 as *mut _ as *mut std::ffi::c_void,
                                &mut c7 as *mut _ as *mut std::ffi::c_void,
                                &mut c8 as *mut _ as *mut std::ffi::c_void,
                                &mut c9 as *mut _ as *mut std::ffi::c_void,
                                &mut c10 as *mut _ as *mut std::ffi::c_void,
                                &mut c11 as *mut _ as *mut std::ffi::c_void,
                            ],
                        )?;
                    }
                }

                // Update conv state (tiny kernel, negligible cost)
                {
                    let ut = std::cmp::min(32, ((kernel_dim + 31) / 32) * 32) as u32;
                    let mut u0 = conv_state;
                    let mut u1 = q_bf16;  // original BF16 input (not conv output)
                    let mut u2 = k_bf16;
                    let mut u3 = v_bf16;
                    let mut u4 = m as i32;
                    let mut u5 = key_dim as i32;
                    let mut u6 = value_dim as i32;
                    let mut u7 = kernel_dim as i32;
                    unsafe {
                        launch(self.kernels.la_update_conv_state,
                            (conv_dim as u32, 1, 1), (ut, 1, 1), 0, self.stream,
                            &mut [
                                &mut u0 as *mut _ as *mut std::ffi::c_void,
                                &mut u1 as *mut _ as *mut std::ffi::c_void,
                                &mut u2 as *mut _ as *mut std::ffi::c_void,
                                &mut u3 as *mut _ as *mut std::ffi::c_void,
                                &mut u4 as *mut _ as *mut std::ffi::c_void,
                                &mut u5 as *mut _ as *mut std::ffi::c_void,
                                &mut u6 as *mut _ as *mut std::ffi::c_void,
                                &mut u7 as *mut _ as *mut std::ffi::c_void,
                            ],
                        )?;
                    }
                }

                if lt { self.stream_sync()?; self.t_la_conv.set(self.t_la_conv.get() + lt1.elapsed().as_secs_f64() * 1000.0); }
                let lt2 = Instant::now();

                // Gate/beta computation directly to BF16 (eliminates 2 FP32->BF16 conversions)
                let beta_bf16_buf = la_beta;  // reuse FP32 buffer at BF16
                let gate_bf16_buf = la_gate;
                {
                    let gt = std::cmp::max(32, ((std::cmp::min(1024, nv) + 31) / 32) * 32) as u32;
                    let mut g0 = beta_bf16_buf;
                    let mut g1 = gate_bf16_buf;
                    let mut g2 = la_b;
                    let mut g3 = la_a;
                    let mut g4 = lw.la_a_log_ptr;
                    let mut g5 = lw.la_dt_bias_ptr;
                    let mut g6 = nv as i32;
                    unsafe {
                        launch(self.kernels.la_compute_gate_beta_bf16,
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

                // Fused repeat-interleave + L2 norm (BF16 in, BF16 out)
                // Reads [M, nk, dk] BF16, writes [M, nv, dk] BF16 normalized
                // Eliminates: 2 repeat_interleave + 2 l2norm + intermediate buffers
                let fla_q_bf16 = *self.scratch.d_scratch1.device_ptr();
                let fla_k_bf16 = *self.scratch.d_scratch2.device_ptr();
                {
                    let nt = std::cmp::max(32, ((std::cmp::min(256, dk) + 31) / 32) * 32) as u32;
                    let smem = nt * 4;
                    // q: conv_q_bf16 [M, nk, dk] -> fla_q_bf16 [M, nv, dk], scaled
                    let mut r0 = fla_q_bf16;
                    let mut r1 = conv_q_bf16;
                    let mut r2 = nk as i32;
                    let mut r3 = dk as i32;
                    let mut r4 = hr as i32;
                    let mut r5 = scale;
                    unsafe {
                        launch(self.kernels.la_fused_repeat_l2norm_bf16,
                            (m as u32, nv as u32, 1), (nt, 1, 1), smem, self.stream,
                            &mut [
                                &mut r0 as *mut _ as *mut std::ffi::c_void,
                                &mut r1 as *mut _ as *mut std::ffi::c_void,
                                &mut r2 as *mut _ as *mut std::ffi::c_void,
                                &mut r3 as *mut _ as *mut std::ffi::c_void,
                                &mut r4 as *mut _ as *mut std::ffi::c_void,
                                &mut r5 as *mut _ as *mut std::ffi::c_void,
                            ],
                        )?;
                    }
                    // k: conv_k_bf16 [M, nk, dk] -> fla_k_bf16 [M, nv, dk], scale=1.0
                    r0 = fla_k_bf16;
                    r1 = conv_k_bf16;
                    r5 = 1.0f32;
                    unsafe {
                        launch(self.kernels.la_fused_repeat_l2norm_bf16,
                            (m as u32, nv as u32, 1), (nt, 1, 1), smem, self.stream,
                            &mut [
                                &mut r0 as *mut _ as *mut std::ffi::c_void,
                                &mut r1 as *mut _ as *mut std::ffi::c_void,
                                &mut r2 as *mut _ as *mut std::ffi::c_void,
                                &mut r3 as *mut _ as *mut std::ffi::c_void,
                                &mut r4 as *mut _ as *mut std::ffi::c_void,
                                &mut r5 as *mut _ as *mut std::ffi::c_void,
                            ],
                        )?;
                    }
                }

                if lt { self.stream_sync()?; self.t_la_prep.set(self.t_la_prep.get() + lt2.elapsed().as_secs_f64() * 1000.0); }

                // ── FLA (Flash Linear Attention) ──
                // All inputs are already BF16 — no conversion needed!
                let fla_bt: usize = 64;
                let fla_pad = (fla_bt - m % fla_bt) % fla_bt;
                let fla_total = m + fla_pad;
                let fla_nt = fla_total / fla_bt;

                // v_bf16 for FLA: conv_v_bf16 [M, value_dim] BF16
                // FLA expects v at [fla_total, nv, dv], which is same layout
                let fla_v_bf16 = conv_v_bf16;
                // beta and gate for FLA
                let fla_beta_bf16 = beta_bf16_buf;
                let fla_gate_bf16 = gate_bf16_buf;

                // Zero-pad beyond M (BF16 = 2 bytes per element)
                if fla_pad > 0 {
                    unsafe {
                        let _ = cuda_sys::lib().cuMemsetD8Async(
                            fla_q_bf16 + (m * nv * dk * 2) as u64, 0, (fla_pad * nv * dk * 2) as usize, self.stream);
                        let _ = cuda_sys::lib().cuMemsetD8Async(
                            fla_k_bf16 + (m * nv * dk * 2) as u64, 0, (fla_pad * nv * dk * 2) as usize, self.stream);
                        let _ = cuda_sys::lib().cuMemsetD8Async(
                            fla_v_bf16 + (m * nv * dv * 2) as u64, 0, (fla_pad * nv * dv * 2) as usize, self.stream);
                        let _ = cuda_sys::lib().cuMemsetD8Async(
                            fla_beta_bf16 + (m * nv * 2) as u64, 0, (fla_pad * nv * 2) as usize, self.stream);
                        let _ = cuda_sys::lib().cuMemsetD8Async(
                            fla_gate_bf16 + (m * nv * 2) as u64, 0, (fla_pad * nv * 2) as usize, self.stream);
                    }
                }

                // No conversion timing — there are no conversions!
                self.t_la_convert.set(0.0); // explicit zero for reporting
                let lt4 = Instant::now();

                // Buffer assignments for FLA intermediates
                // All FLA intermediate dtypes match what compile_kernels.py compiled:
                //   BF16: q, k, v, beta, gate, w, u, h, h0, v_new, o, Ai
                //   FP32: g_cumsum, A, ht
                let a_fla = *self.d_fla_a.as_ref().ok_or("no fla_a")?.device_ptr();
                let ai_fla = *self.d_fla_ai.as_ref().ok_or("no fla_ai")?.device_ptr();
                let w_fla = *self.d_fla_w.as_ref().ok_or("no fla_w")?.device_ptr();
                let u_fla = *self.d_fla_u.as_ref().ok_or("no fla_u")?.device_ptr();
                let h_fla = *self.d_fla_h.as_ref().ok_or("no fla_h")?.device_ptr();
                // h0 is only read by step 5, so reuse the output buffer as a zeroed
                // BF16 initial state and let step 6 overwrite it later.
                let h0_fla = *self.d_fla_o.as_ref().ok_or("no fla_o")?.device_ptr();
                let ht_fla = *self.d_fla_final_state.as_ref().ok_or("no fla_final_state")?.device_ptr();
                let v_new_fla = *self.d_fla_v_new.as_ref().ok_or("no fla_v_new")?.device_ptr();
                let o_fla = *self.d_fla_o.as_ref().ok_or("no fla_o")?.device_ptr();

                let fla = self.fla.as_ref().unwrap();
                let t_arg = fla_total as i32;

                // DEBUG: sync before FLA to catch conv/gate/repeat kernel errors

                // Zero h0 (BF16 initial state) — separate from ht (FP32 output)
                unsafe {
                    let _ = cuda_sys::lib().cuMemsetD8Async(
                        h0_fla, 0, (nv * dk * dv * 2) as usize, self.stream);
                }

                let stream_ptr = self.stream as *mut std::ffi::c_void;

                let g_cum_fla = *self.d_fla_g_cumsum.as_ref().ok_or("no fla_g_cumsum")?.device_ptr();

                // Step 1: cumsum — gate(BF16) → g_cumsum(FP32)
                // Grid: (NT, B*H) = (fla_nt, nv)
                let rc = unsafe {
                    (fla.cumsum)(fla_gate_bf16, g_cum_fla, 1.0f32,
                        0, 0, t_arg,
                        fla_nt as u32, nv as u32, 1, stream_ptr)
                };
                if rc != 0 { return Err(format!("FLA cumsum failed: {}", rc)); }

                // Step 2: kkt — k, g_cumsum, beta → A
                // Grid: (NT, B*H) = (fla_nt, nv)
                let rc = unsafe {
                    (fla.kkt)(fla_k_bf16, g_cum_fla, fla_beta_bf16, a_fla,
                        0, 0, t_arg,
                        fla_nt as u32, nv as u32, 1, stream_ptr)
                };
                if rc != 0 { return Err(format!("FLA kkt failed: {}", rc)); }

                // Step 3: solve_tril — A → Ai
                // Grid: (NT, B*H) = (fla_nt, nv)
                let rc = unsafe {
                    (fla.solve_tril)(a_fla, ai_fla,
                        0, 0, t_arg,
                        fla_nt as u32, nv as u32, 1, stream_ptr)
                };
                if rc != 0 { return Err(format!("FLA solve_tril failed: {}", rc)); }

                // Step 4: recompute_w_u — k, v, beta, Ai, g → w, u
                // Grid: (NT, B*H) = (fla_nt, nv)
                // Note: after step 3, A (la_conv_out) is dead, so u can reuse that space
                let rc = unsafe {
                    (fla.wy_repr)(fla_k_bf16, fla_v_bf16, fla_beta_bf16, w_fla, u_fla, ai_fla, g_cum_fla,
                        0, 0, t_arg,
                        fla_nt as u32, nv as u32, 1, stream_ptr)
                };
                if rc != 0 { return Err(format!("FLA wy_repr failed: {}", rc)); }

                // Step 5: state_recurrence — k, u, w, g, h0 → h, v_new, ht
                // Grid: (cdiv(V, BV), B*H) = (cdiv(dv, 32), nv) — BV=32 baked into cubin
                // "blockdim64" in the kernel name refers to BT=64 (chunk time dim), NOT BV.
                // C wrapper params: k, v(=u), w, v_new, g, h, h0, ht, T
                // h0 is BF16 (zeroed), ht is FP32 — separate buffers to avoid dtype/overlap issues
                let sr_grid_x = ((dv + 31) / 32) as u32;
                let rc = unsafe {
                    (fla.state_recurrence)(
                        fla_k_bf16,     // k (BF16)
                        u_fla,      // v (kernel param) = Python's u (BF16)
                        w_fla,      // w (kernel param) = Python's w (BF16)
                        v_new_fla,  // v_new output (BF16)
                        g_cum_fla,   // g (cumsum gate, FP32)
                        0,          // gk (unused — USE_GK=False constexpr)
                        h_fla,      // h output (BF16, per-chunk states)
                        h0_fla,     // h0 (BF16, zeroed initial state)
                        ht_fla,     // ht (FP32, final state output)
                        0, 0,       // cu_seqlens, chunk_offsets (IS_VARLEN=False)
                        t_arg,
                        sr_grid_x, nv as u32, 1, stream_ptr)
                };
                if rc != 0 { return Err(format!("FLA state_recurrence failed: {}", rc)); }

                // Step 6: output — q, k, v_new, h, g → o
                // Grid: (cdiv(V, BV), NT, B*H) = (cdiv(dv, BV), fla_nt, nv) — 3D grid
                // BV for chunk_fwd_kernel_o was autotuned. With 8 warps: BV=128, BK=128.
                let o_bv = 128usize; // from autotune: Config(BK=128, BV=128, num_warps=8)
                let o_grid_x = ((dv + o_bv - 1) / o_bv) as u32;
                // q was already pre-scaled by `scale` in step 9 (L2 norm).
                // chunk_fwd_kernel_o multiplies ALL output by its `scale` arg (line 122:
                //   b_o = b_o * scale + dot(A, v) * scale).
                // Passing scale=1.0 avoids double-scaling.
                let rc = unsafe {
                    (fla.output)(
                        fla_q_bf16,     // q (already includes scale from fused repeat+l2norm)
                        fla_k_bf16,     // k
                        v_new_fla,  // v (kernel param) = v_new from step 5
                        h_fla,      // h (per-chunk states from step 5)
                        g_cum_fla,   // g (cumsum gate, FP32)
                        0,          // g_gamma (unused — USE_G_GAMMA=False constexpr)
                        o_fla,      // o (output, BF16)
                        0, 0,       // cu_seqlens, chunk_indices (IS_VARLEN=False)
                        1.0f32,     // scale=1.0 since q is already pre-scaled
                        t_arg,
                        o_grid_x, fla_nt as u32, nv as u32, stream_ptr)
                };
                if rc != 0 { return Err(format!("FLA output failed: {}", rc)); }
                if std::env::var("KRASIS_FLA_DEBUG").is_ok() { self.stream_sync().map_err(|e| format!("FLA output sync: {e}"))?; eprintln!("[FLA-DBG] step6 output OK"); }
                if lt { self.stream_sync()?; self.t_la_fla.set(self.t_la_fla.get() + lt4.elapsed().as_secs_f64() * 1000.0); }
                let lt5 = Instant::now();

                // Copy final state to decode state buffer (FP32, unchanged)
                if lw.la_recur_state_ptr != 0 {
                    self.memcpy_d2d(lw.la_recur_state_ptr, ht_fla, (nv * dk * dv * 4) as u64)?;
                }

                if lt { self.stream_sync()?; self.t_la_postfla.set(self.t_la_postfla.get() + lt5.elapsed().as_secs_f64() * 1000.0); }

                // Gated RMSNorm after FLA. Default stays on the direct BF16-input kernel,
                // but a debug switch can force BF16->FP32 conversion and reuse the
                // fallback FP32 RMSNorm kernel to isolate fast-path norm issues.
                let lt6 = Instant::now();
                if std::env::var("KRASIS_LA_FLA_FP32_NORM").is_ok() {
                    let fp32_norm_in = fp32_scratch;
                    {
                        let nt = std::cmp::max(32, ((std::cmp::min(1024, value_dim) + 31) / 32) * 32) as u32;
                        let mut c0 = fp32_norm_in;
                        let mut c1 = o_fla;
                        let mut c2 = value_dim as i32;
                        unsafe {
                            launch(self.kernels.la_bf16_to_fp32,
                                (m as u32, 1, 1), (nt, 1, 1), 0, self.stream,
                                &mut [
                                    &mut c0 as *mut _ as *mut std::ffi::c_void,
                                    &mut c1 as *mut _ as *mut std::ffi::c_void,
                                    &mut c2 as *mut _ as *mut std::ffi::c_void,
                                ],
                            )?;
                        }
                    }
                    let nt = std::cmp::max(32, ((std::cmp::min(256, dv) + 31) / 32) * 32) as u32;
                    let smem = nt * 4;
                    let mut n0 = rmsnorm_out;
                    let mut n1 = fp32_norm_in;
                    let mut n2 = la_z;
                    let mut n3 = lw.la_norm_weight_ptr;
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
                } else {
                    {
                        let nt = std::cmp::max(32, ((std::cmp::min(256, dv) + 31) / 32) * 32) as u32;
                        let smem = nt * 4;
                        let mut n0 = rmsnorm_out;  // BF16 output [M, nv*dv] into la_v
                        let mut n1 = o_fla;      // BF16 input (FLA output, in la_v_beta)
                        let mut n2 = la_z;       // BF16 gate [M, nv, dv]
                        let mut n3 = lw.la_norm_weight_ptr; // FP32 weight [dv]
                        let mut n4 = nv as i32;
                        let mut n5 = dv as i32;
                        let mut n6 = cfg.rms_norm_eps;
                        unsafe {
                            launch(self.kernels.la_gated_rmsnorm_bf16in,
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
                }
                if lt { self.stream_sync()?; self.t_la_norm.set(self.t_la_norm.get() + lt6.elapsed().as_secs_f64() * 1000.0); }

            } else {
            // ── Old FP32 pipeline (non-FLA fallback) ──
            rmsnorm_out = la_v_beta;  // default; overwritten by gated_rmsnorm if fla.is_none()
            let lt1 = Instant::now();
            let conv_state = lw.la_conv_state_ptr;
            let conv_weight = lw.la_conv_weight_ptr;

            // 5. Concat + Conv1d + SiLU + Split (old 4-kernel path)
            {
                let threads = std::cmp::max(32, ((std::cmp::min(256, conv_dim) + 31) / 32) * 32) as u32;
                let mut cc0 = proj_buf; let mut cc1 = q_bf16; let mut cc2 = k_bf16;
                let mut cc3 = v_bf16; let mut cc4 = key_dim as i32;
                let mut cc5 = key_dim as i32; let mut cc6 = value_dim as i32;
                unsafe {
                    launch(self.kernels.concat_3_bf16,
                        (m as u32, 1, 1), (threads, 1, 1), 0, self.stream,
                        &mut [&mut cc0 as *mut _ as *mut std::ffi::c_void,
                              &mut cc1 as *mut _ as *mut std::ffi::c_void,
                              &mut cc2 as *mut _ as *mut std::ffi::c_void,
                              &mut cc3 as *mut _ as *mut std::ffi::c_void,
                              &mut cc4 as *mut _ as *mut std::ffi::c_void,
                              &mut cc5 as *mut _ as *mut std::ffi::c_void,
                              &mut cc6 as *mut _ as *mut std::ffi::c_void])?;
                }
            }
            {
                let ct = std::cmp::max(32, ((std::cmp::min(1024, m) + 31) / 32) * 32) as u32;
                let mut c0 = la_conv_out; let mut c1 = conv_state; let mut c2 = proj_buf;
                let mut c3 = conv_weight; let mut c4 = m as i32;
                let mut c5 = conv_dim as i32; let mut c6 = kernel_dim as i32;
                unsafe {
                    launch(self.kernels.la_depthwise_conv1d_silu,
                        (conv_dim as u32, 1, 1), (ct, 1, 1), 0, self.stream,
                        &mut [&mut c0 as *mut _ as *mut std::ffi::c_void,
                              &mut c1 as *mut _ as *mut std::ffi::c_void,
                              &mut c2 as *mut _ as *mut std::ffi::c_void,
                              &mut c3 as *mut _ as *mut std::ffi::c_void,
                              &mut c4 as *mut _ as *mut std::ffi::c_void,
                              &mut c5 as *mut _ as *mut std::ffi::c_void,
                              &mut c6 as *mut _ as *mut std::ffi::c_void])?;
                }
            }
            let fp32_scratch = *self.scratch.d_fp32_scratch.device_ptr();
            self.launch_transpose_f32(fp32_scratch, la_conv_out, conv_dim, m)?;
            {
                let threads = std::cmp::max(32, ((std::cmp::min(256, conv_dim) + 31) / 32) * 32) as u32;
                let mut sp0 = la_q; let mut sp1 = la_k; let mut sp2 = la_v;
                let mut sp3 = fp32_scratch; let mut sp4 = key_dim as i32; let mut sp5 = value_dim as i32;
                unsafe {
                    launch(self.kernels.la_split_conv_output,
                        (m as u32, 1, 1), (threads, 1, 1), 0, self.stream,
                        &mut [&mut sp0 as *mut _ as *mut std::ffi::c_void,
                              &mut sp1 as *mut _ as *mut std::ffi::c_void,
                              &mut sp2 as *mut _ as *mut std::ffi::c_void,
                              &mut sp3 as *mut _ as *mut std::ffi::c_void,
                              &mut sp4 as *mut _ as *mut std::ffi::c_void,
                              &mut sp5 as *mut _ as *mut std::ffi::c_void])?;
                }
            }
            if lt { self.stream_sync()?; self.t_la_conv.set(self.t_la_conv.get() + lt1.elapsed().as_secs_f64() * 1000.0); }
            let lt2 = Instant::now();

            // 7. Gate/beta + repeat_interleave + l2norm (FP32 path)
            {
                let gt = std::cmp::max(32, ((std::cmp::min(1024, nv) + 31) / 32) * 32) as u32;
                let mut g0 = la_beta; let mut g1 = la_gate; let mut g2 = la_b;
                let mut g3 = la_a; let mut g4 = lw.la_a_log_ptr;
                let mut g5 = lw.la_dt_bias_ptr; let mut g6 = nv as i32;
                unsafe {
                    launch(self.kernels.la_compute_gate_beta,
                        (m as u32, 1, 1), (gt, 1, 1), 0, self.stream,
                        &mut [&mut g0 as *mut _ as *mut std::ffi::c_void,
                              &mut g1 as *mut _ as *mut std::ffi::c_void,
                              &mut g2 as *mut _ as *mut std::ffi::c_void,
                              &mut g3 as *mut _ as *mut std::ffi::c_void,
                              &mut g4 as *mut _ as *mut std::ffi::c_void,
                              &mut g5 as *mut _ as *mut std::ffi::c_void,
                              &mut g6 as *mut _ as *mut std::ffi::c_void])?;
                }
            }
            if hr > 1 {
                let dt = std::cmp::max(32, ((std::cmp::min(256, dk) + 31) / 32) * 32) as u32;
                let mut r0 = la_v_beta; let mut r1 = la_q; let mut r2 = nk as i32;
                let mut r3 = dk as i32; let mut r4 = hr as i32;
                unsafe {
                    launch(self.kernels.la_repeat_interleave,
                        (m as u32, nv as u32, 1), (dt, 1, 1), 0, self.stream,
                        &mut [&mut r0 as *mut _ as *mut std::ffi::c_void,
                              &mut r1 as *mut _ as *mut std::ffi::c_void,
                              &mut r2 as *mut _ as *mut std::ffi::c_void,
                              &mut r3 as *mut _ as *mut std::ffi::c_void,
                              &mut r4 as *mut _ as *mut std::ffi::c_void])?;
                }
                r0 = la_k_beta; r1 = la_k;
                unsafe {
                    launch(self.kernels.la_repeat_interleave,
                        (m as u32, nv as u32, 1), (dt, 1, 1), 0, self.stream,
                        &mut [&mut r0 as *mut _ as *mut std::ffi::c_void,
                              &mut r1 as *mut _ as *mut std::ffi::c_void,
                              &mut r2 as *mut _ as *mut std::ffi::c_void,
                              &mut r3 as *mut _ as *mut std::ffi::c_void,
                              &mut r4 as *mut _ as *mut std::ffi::c_void])?;
                }
                la_q = la_v_beta;
                la_k = la_k_beta;
            }
            {
                let nt = std::cmp::max(32, ((std::cmp::min(256, dk) + 31) / 32) * 32) as u32;
                let smem = nt * 4;
                let mut l0 = la_q; let mut l1 = scale; let mut l2 = nv as i32; let mut l3 = dk as i32;
                unsafe {
                    launch(self.kernels.la_l2norm_per_head,
                        (m as u32, nv as u32, 1), (nt, 1, 1), smem, self.stream,
                        &mut [&mut l0 as *mut _ as *mut std::ffi::c_void,
                              &mut l1 as *mut _ as *mut std::ffi::c_void,
                              &mut l2 as *mut _ as *mut std::ffi::c_void,
                              &mut l3 as *mut _ as *mut std::ffi::c_void])?;
                }
                l0 = la_k; l1 = 1.0f32;
                unsafe {
                    launch(self.kernels.la_l2norm_per_head,
                        (m as u32, nv as u32, 1), (nt, 1, 1), smem, self.stream,
                        &mut [&mut l0 as *mut _ as *mut std::ffi::c_void,
                              &mut l1 as *mut _ as *mut std::ffi::c_void,
                              &mut l2 as *mut _ as *mut std::ffi::c_void,
                              &mut l3 as *mut _ as *mut std::ffi::c_void])?;
                }
            }
            if lt { self.stream_sync()?; self.t_la_prep.set(self.t_la_prep.get() + lt2.elapsed().as_secs_f64() * 1000.0); }

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

            // 11. Preserve raw q/k in canonical [nv, total_len, dim] layout before
            // la_apply_beta reuses the token-major staging for scaled outputs.
            self.launch_transpose_3d_f32(fp32_scratch, la_q, total_len, nv, dk)?;
            self.memcpy_d2d(la_q, fp32_scratch, (nv * total_len * dk * 4) as u64)?;

            self.launch_transpose_3d_f32(fp32_scratch, la_k, total_len, nv, dk)?;
            self.memcpy_d2d(la_k, fp32_scratch, (nv * total_len * dk * 4) as u64)?;

            // 12. Compute beta-scaled values:
            // v_beta = v * beta while v is still token-major [total_len, nv, dv]
            // k_beta = k * beta directly into canonical [nv, total_len, dk]
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
                let mut bp8 = total_len as i32;
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
                            &mut bp8 as *mut _ as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            // Transpose all arrays from [total_len, nv, dim] to [nv, total_len, dim]
            // via single kernel launch each (replaces millions of memcpy calls)

            // v: [total_len, nv, dv] -> [nv, total_len, dv]
            self.launch_transpose_3d_f32(fp32_scratch, la_v, total_len, nv, dv)?;
            self.memcpy_d2d(la_v, fp32_scratch, (nv * total_len * dv * 4) as u64)?;

            // gate: [total_len, nv] -> [nv, total_len] (2D, use existing transpose)
            self.launch_transpose_f32(fp32_scratch, la_gate, total_len, nv)?;
            self.memcpy_d2d(la_gate, fp32_scratch, (nv * total_len * 4) as u64)?;

            // v_beta: [total_len, nv, dv] -> [nv, total_len, dv]
            self.launch_transpose_3d_f32(fp32_scratch, la_v_beta, total_len, nv, dv)?;
            self.memcpy_d2d(la_v_beta, fp32_scratch, (nv * total_len * dv * 4) as u64)?;

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

            // DIAG: after transposition + cumsum, before attn matrix
            if la_diag {
                self.stream_sync()?;
                // Layout is now [nv, total_len, dim]. Head 0, pos 0 is at offset 0.
                // v_beta [nv, total_len, dv]: head 0 pos 0 = offset 0
                let vb_h0 = self.diag_l2_norm_f32(la_v_beta, 0, dv, dv);
                // k_beta [nv, total_len, dk]: head 0 pos 0 = offset 0
                let kb_h0 = self.diag_l2_norm_f32(la_k_beta, 0, dk, dk);
                // beta [M, nv] was not transposed, but gate [nv, total_len] was
                // g_cum [nv, total_len]: head 0, first 4 values
                let gcum_vals = self.diag_download_f32(la_g_cum, 4);
                eprintln!("[DIAG L{:02}] pre_attn: v_beta_h0p0={:.6} k_beta_h0p0={:.6} g_cum[0..4]=[{:.4},{:.4},{:.4},{:.4}]",
                    layer_idx, vb_h0, kb_h0, gcum_vals[0], gcum_vals[1], gcum_vals[2], gcum_vals[3]);
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

            // DIAG: after both triangular solves
            if la_diag {
                self.stream_sync()?;
                // v_corr (la_v_beta) [nv, total_len, dv]: head 0 pos 0
                let vc_h0 = self.diag_l2_norm_f32(la_v_beta, 0, dv, dv);
                // k_cumdecay (la_k_beta) [nv, total_len, dk]: head 0 pos 0
                let kc_h0 = self.diag_l2_norm_f32(la_k_beta, 0, dk, dk);
                eprintln!("[DIAG L{:02}] post_trisolve: v_corr_h0p0={:.6} k_cumd_h0p0={:.6}",
                    layer_idx, vc_h0, kc_h0);
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

                // DIAG: after v_new for chunk 0
                if la_diag && c == 0 {
                    self.stream_sync()?;
                    // v_new [nv, CS, dv]: head 0 pos 0 = offset 0
                    let vn_h0 = self.diag_l2_norm_f32(la_v_new, 0, dv, dv);
                    // Also check a few raw values from head 0, pos 0
                    let vn_raw = self.diag_download_f32(la_v_new, 4);
                    eprintln!("[DIAG L{:02}] chunk0_v_new: h0p0_norm={:.6} raw[0..4]=[{:.6},{:.6},{:.6},{:.6}]",
                        layer_idx, vn_h0, vn_raw[0], vn_raw[1], vn_raw[2], vn_raw[3]);
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

                // DIAG: after chunk output for chunk 0
                if la_diag && c == 0 {
                    self.stream_sync()?;
                    // output [nv, total_len, dv]: head 0 pos 0 = offset 0
                    let out_h0 = self.diag_l2_norm_f32(output_buf, 0, dv, dv);
                    // Check a few raw values
                    let out_raw = self.diag_download_f32(output_buf, 4);
                    eprintln!("[DIAG L{:02}] chunk0_output: h0p0_norm={:.6} raw[0..4]=[{:.6},{:.6},{:.6},{:.6}]",
                        layer_idx, out_h0, out_raw[0], out_raw[1], out_raw[2], out_raw[3]);
                    // Total output norm at pos 0 across ALL heads
                    // output_buf layout: [nv, total_len, dv]. pos 0 of head h = h * total_len * dv
                    let mut total_sq = 0.0f64;
                    for h in 0..nv {
                        let offset = (h * total_len * dv) * 4;
                        let head_vals = self.diag_download_f32(output_buf + offset as u64, dv);
                        for &v in &head_vals {
                            total_sq += (v as f64) * (v as f64);
                        }
                    }
                    eprintln!("[DIAG L{:02}] chunk0_output_total_pos0: norm={:.6} (across all {} heads)",
                        layer_idx, (total_sq.sqrt()) as f32, nv);
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
            } // end else (custom chunk recurrence path)

            // Gated RMSNorm for non-FLA fallback path (FP32 input from chunk recurrence)
            // FLA path has its own BF16 gated_rmsnorm above
            if self.fla.is_none() {
                // Output is [M, nv*dv] BF16. When value_dim > hidden_size,
                // d_attn_out is too small. Use la_v_beta ([nv, M, dv] FP32) as temp.
                // la_v (input to this kernel) can't be reused here.
                rmsnorm_out = la_v_beta;
                let lt6 = Instant::now();
                {
                    let nt = std::cmp::max(32, ((std::cmp::min(256, dv) + 31) / 32) * 32) as u32;
                    let smem = nt * 4;
                    let mut n0 = rmsnorm_out;
                    let mut n1 = la_v;
                    let mut n2 = la_z;
                    let mut n3 = lw.la_norm_weight_ptr;
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
                if lt { self.stream_sync()?; self.t_la_norm.set(self.t_la_norm.get() + lt6.elapsed().as_secs_f64() * 1000.0); }
            }
            let lt7 = Instant::now();
            // 21. Output projection: [M, nv*dv] BF16 -> [M, hidden] BF16
            // Input is rmsnorm_out (la_v or la_v_beta, sized [M, nv*dv]).
            // Use scratch1 as temp to avoid Marlin GEMM input/output aliasing.
            if la_diag {
                self.stream_sync()?;
                let pre_oproj_norm = self.diag_l2_norm(rmsnorm_out, 0, value_dim, value_dim);
                eprintln!("[DIAG L{:02}] pre_out_proj pos0 norm={:.6} (after gated_rmsnorm, dim={})", layer_idx, pre_oproj_norm, value_dim);

                // Element-level: actual kernel output [M, nv*dv] BF16, head 0 pos 0
                let out_vals = self.diag_download_bf16(rmsnorm_out, 4);
                eprintln!("[DIAG L{:02}] grmsnorm out[h0p0][0:4]=[{:.8},{:.8},{:.8},{:.8}]",
                    layer_idx, out_vals[0], out_vals[1], out_vals[2], out_vals[3]);
            }
            let o_temp = *self.scratch.d_scratch1.device_ptr();
            self.la_gemm(rmsnorm_out, &lw.la_out_proj, &lw.la_out_proj_bf16, o_temp, m)?;
            self.memcpy_d2d(attn_out, o_temp, (m * cfg.hidden_size * 2) as u64)?;
            if la_diag {
                let post_oproj_norm = self.diag_l2_norm(attn_out, 0, cfg.hidden_size, cfg.hidden_size);
                eprintln!("[DIAG L{:02}] post_out_proj pos0 norm={:.6} (mixer output, dim={})", layer_idx, post_oproj_norm, cfg.hidden_size);
            }

            if lt { self.stream_sync()?; self.t_la_oproj.set(self.t_la_oproj.get() + lt7.elapsed().as_secs_f64() * 1000.0); }
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
        // Use fused MoE path when available (default for prefill).
        // Supports two modes: contiguous buffer (d_fused_expert_w1_a) or pointer table
        // (d_expert_w1_ptrs). Pointer table mode computes B_expert_off = expert_ptr - B;
        // with B=0 (no contiguous buffer), the offset equals the absolute address, which
        // cancels correctly in B_ptr[i] + B_expert_off accesses.
        // Set KRASIS_SEQUENTIAL_MOE=1 to force sequential path for debugging.
        let use_fused = self.kernels.fused_moe_fn.is_some()
            && (self.d_fused_expert_w1_a.is_some() || self.d_expert_w1_ptrs.is_some())
            && std::env::var("KRASIS_SEQUENTIAL_MOE").is_err();
        if use_fused {
            if layer_idx == 0 {
                if stderr_debug_enabled() {
                    eprintln!("[PREFILL] Using FUSED MoE path (bulk layer DMA)");
                }
            }
            return self.forward_moe_fused(layer_idx, m);
        }
        if layer_idx == 0 {
            if stderr_debug_enabled() {
                eprintln!("[PREFILL] Using SEQUENTIAL MoE path");
            }
        }
        // Per-expert sequential dispatch
        self.forward_moe_sequential(layer_idx, m)
    }

    /// Fused MoE forward using MarlinDefault kernel.
    /// One kernel launch handles ALL active experts per GEMM (w1, w2).
    fn forward_moe_fused(&mut self, layer_idx: usize, m: usize) -> Result<(), String> {
        let diag_moe = std::env::var("KRASIS_PREFILL_DIAG").is_ok();
        let debug_prefill = prefill_debug_enabled();
        let mt = self.gqa_timing_enabled.get(); // MoE timing flag (reuses same env var)
        let mt0 = if mt { self.stream_sync()?; Some(Instant::now()) } else { None };
        // Extract all needed fields from layer_weights up front so the borrow
        // on self.layer_weights is dropped before we call &mut self methods.
        let h = self.config.hidden_size;
        let n_experts = self.layer_weights[layer_idx].moe_num_experts;
        let topk = self.layer_weights[layer_idx].moe_topk;
        let inter = self.config.moe_intermediate_size;
        let scoring_func = self.layer_weights[layer_idx].moe_scoring_func;
        let gated = self.layer_weights[layer_idx].moe_gated;
        let activation = self.layer_weights[layer_idx].moe_activation;
        let scale_factor = self.layer_weights[layer_idx].moe_routed_scaling_factor;
        let gs = self.config.group_size;
        let bits = self.config.expert_bits;
        let moe_layer_idx = self.layer_weights[layer_idx].moe_layer_idx;
        let has_shared = (self.layer_weights[layer_idx].shared_w1.is_some()
            && self.layer_weights[layer_idx].shared_w2.is_some())
            || (self.layer_weights[layer_idx].shared_w1_bf16.is_some()
                && self.layer_weights[layer_idx].shared_w2_bf16.is_some());

        let hidden = *self.scratch.d_hidden.device_ptr();
        let gate_out = *self.scratch.d_gate_out.device_ptr();
        let topk_ids_ptr = *self.scratch.d_topk_ids.device_ptr();
        let topk_weights_ptr = *self.scratch.d_topk_weights.device_ptr();

        // 1. Gate GEMM (same as sequential)
        let gate_ptr = self.layer_weights[layer_idx].moe_gate_ptr;
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
        // Dereference device_ptr() immediately to avoid holding borrows on self
        let sorted_ids_val = *self.scratch.d_gather_src_map.device_ptr();
        let fused_expert_ids_val = *self.scratch.d_fused_expert_ids.device_ptr();
        let num_tokens_post_val = *self.scratch.d_num_tokens_post.device_ptr();
        {
            let ps_threads = std::cmp::max(32, ((std::cmp::min(1024, n_experts) + 31) / 32) * 32) as u32;
            let mut p0 = expert_offsets_ptr;
            let mut p1 = num_tokens_post_val;
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
            let mut p0 = sorted_ids_val;
            let mut p1 = write_offsets_ptr;
            let mut p2 = topk_ids_ptr;
            let mut p3 = expert_offsets_ptr;
            let mut p4 = m as i32;
            let mut p5 = topk as i32;
            let mut p6 = n_experts as i32;
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
                    ],
                )?;
            }
        }
        {
            let mut p0 = sorted_ids_val;
            let mut p1 = fused_expert_ids_val;
            let mut p2 = expert_offsets_ptr;
            let mut p3 = expert_counts_ptr;
            let mut p4 = m as i32;
            let mut p5 = topk as i32;
            let mut p6 = n_experts as i32;
            let mut p7 = block_size;
            unsafe {
                launch(self.kernels.moe_finalize_sorted,
                    (n_experts as u32, 1, 1), (1, 1, 1), 0, self.stream,
                    &mut [
                        &mut p0 as *mut _ as *mut std::ffi::c_void,
                        &mut p1 as *mut _ as *mut std::ffi::c_void,
                        &mut p2 as *mut _ as *mut std::ffi::c_void,
                        &mut p3 as *mut _ as *mut std::ffi::c_void,
                        &mut p4 as *mut _ as *mut std::ffi::c_void,
                        &mut p5 as *mut _ as *mut std::ffi::c_void,
                        &mut p6 as *mut _ as *mut std::ffi::c_void,
                        &mut p7 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Sync to get routing results on CPU for selective expert DMA
        self.stream_sync()?;
        if let Some(t) = mt0 {
            self.t_moe_gate.set(self.t_moe_gate.get() + t.elapsed().as_secs_f64() * 1000.0);
        }
        let mt1 = if mt { Some(Instant::now()) } else { None };

        // Download expert counts to determine which experts are active
        let mut h_expert_counts = vec![0i32; n_experts];
        unsafe {
            cuda_sys::lib().cuMemcpyDtoH_v2(
                h_expert_counts.as_mut_ptr() as *mut _,
                expert_counts_ptr, (n_experts * 4) as usize);
        }

        // Download num_tokens_post_padded for fused kernel
        let mut h_num_post = [0i32; 1];
        unsafe {
            cuda_sys::lib().cuMemcpyDtoH_v2(
                h_num_post.as_mut_ptr() as *mut _,
                num_tokens_post_val, 4);
        }
        let total_sorted = h_num_post[0] as usize;
        let active_experts = h_expert_counts.iter().filter(|&&c| c > 0).count();
        if diag_moe && layer_idx == 0 {
            eprintln!("[DIAG MoE L0] selective DMA: {}/{} experts active, total_sorted={}",
                active_experts, n_experts, total_sorted);
        }

        // DIAG: dump routing info for layer 0
        if diag_moe && layer_idx == 0 {
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

        // 4. Build expert pointer tables (zero-copy for HCS, H2D staging for cold).
        //    Instead of copying ALL experts into a contiguous fused buffer, we build
        //    a per-expert pointer table. HCS-resident experts are referenced in-place
        //    (zero copy), cold experts are H2D'd to a small staging buffer.
        let use_ptr_table = self.d_expert_w1_ptrs.is_some() && self.d_cold_staging.is_some();
        let cold_staging_base = self.d_cold_staging.as_ref().map_or(0, |s| *s.device_ptr());

        // Also keep fused buffer pointers for the B parameter (needed as base reference
        // for the kernel's B_ptr arithmetic even in pointer table mode)
        let cur = self.fused_expert_buf_cur;
        let (w1_buf, w1s_buf, w2_buf, w2s_buf) = if cur == 0 {
            (self.d_fused_expert_w1_a.as_ref(), self.d_fused_expert_w1s_a.as_ref(),
             self.d_fused_expert_w2_a.as_ref(), self.d_fused_expert_w2s_a.as_ref())
        } else {
            (self.d_fused_expert_w1_b.as_ref(), self.d_fused_expert_w1s_b.as_ref(),
             self.d_fused_expert_w2_b.as_ref(), self.d_fused_expert_w2s_b.as_ref())
        };
        let w1_base = w1_buf.map_or(0, |b| *b.device_ptr());
        let w1s_base = w1s_buf.map_or(0, |b| *b.device_ptr());
        let w2_base = w2_buf.map_or(0, |b| *b.device_ptr());
        let w2s_base = w2s_buf.map_or(0, |b| *b.device_ptr());

        // GPU pointers for the pointer table buffers.
        // IMPORTANT: Allocate fresh each layer to avoid corruption from cudarc pool
        // overlapping with HCS raw cuMemAlloc_v2. Only 4 * 4KB = 16KB, negligible.
        let ptrs_bytes = n_experts * 8;
        let (w1_ptrs_gpu, w1s_ptrs_gpu, w2_ptrs_gpu, w2s_ptrs_gpu) = unsafe {
            let mut p1: u64 = 0; let mut p2: u64 = 0; let mut p3: u64 = 0; let mut p4: u64 = 0;
            cuda_sys::lib().cuMemAlloc_v2(&mut p1, ptrs_bytes);
            cuda_sys::lib().cuMemAlloc_v2(&mut p2, ptrs_bytes);
            cuda_sys::lib().cuMemAlloc_v2(&mut p3, ptrs_bytes);
            cuda_sys::lib().cuMemAlloc_v2(&mut p4, ptrs_bytes);
            (p1, p2, p3, p4)
        };

        if use_ptr_table {
            // Build pointer tables for active experts
            let active: Vec<usize> = h_expert_counts.iter().enumerate()
                .filter_map(|(eid, &cnt)| if cnt > 0 { Some(eid) } else { None })
                .collect();

            let mi = moe_layer_idx.unwrap_or(0);
            let mut hcs_count = 0usize;
            let mut pinned_count = 0usize;
            let mut cold_count = 0usize;
            let mut cold_slot = 0usize; // next available slot in cold staging

            // Clear pointer tables (inactive experts get 0 = won't be accessed by kernel)
            for i in 0..n_experts {
                self.h_expert_w1_ptrs[i] = 0;
                self.h_expert_w1s_ptrs[i] = 0;
                self.h_expert_w2_ptrs[i] = 0;
                self.h_expert_w2s_ptrs[i] = 0;
            }

            let moe_data = if let Some(moe_idx) = moe_layer_idx {
                self.moe_layers.get(moe_idx).and_then(|o| o.as_ref())
            } else { None };

            for &eid in &active {
                // Try prefill cache + HCS first (zero copy — point directly into VRAM)
                if let Some((hw1p, hw1s, hw2p, hw2s)) = self.expert_lookup(mi, eid) {
                    self.h_expert_w1_ptrs[eid] = hw1p;
                    self.h_expert_w1s_ptrs[eid] = hw1s;
                    self.h_expert_w2_ptrs[eid] = hw2p;
                    self.h_expert_w2s_ptrs[eid] = hw2s;
                    hcs_count += 1;
                    continue;
                }

                let pin_offset = if self.pinning_active
                    && mi < self.pinned_expert_offsets.len()
                    && eid < self.pinned_expert_offsets[mi].len()
                {
                    self.pinned_expert_offsets[mi][eid]
                } else {
                    None
                };

                if let Some(pool_off) = pin_offset {
                    let src = self.pinning_pool_ptr + pool_off as u64;
                    let mut off = 0u64;
                    self.h_expert_w1_ptrs[eid] = src + off;
                    off += self.w1_packed_per_expert as u64;
                    self.h_expert_w1s_ptrs[eid] = src + off;
                    off += self.w1_scales_per_expert as u64;
                    self.h_expert_w2_ptrs[eid] = src + off;
                    off += self.w2_packed_per_expert as u64;
                    self.h_expert_w2s_ptrs[eid] = src + off;
                    pinned_count += 1;
                    continue;
                }

                if let Some(md) = moe_data {
                    // Cold expert: H2D to staging slot.
                    // If cold staging is full, fall back to the contiguous fused buffer.
                    if eid < md.experts.len() && cold_slot >= self.max_cold_experts {
                        // Overflow: DMA to contiguous fused buffer slot (if available)
                        let (w1_buf, w1s_buf, w2_buf, w2s_buf) = if self.fused_expert_buf_cur == 0 {
                            (self.d_fused_expert_w1_a.as_ref(), self.d_fused_expert_w1s_a.as_ref(),
                             self.d_fused_expert_w2_a.as_ref(), self.d_fused_expert_w2s_a.as_ref())
                        } else {
                            (self.d_fused_expert_w1_b.as_ref(), self.d_fused_expert_w1s_b.as_ref(),
                             self.d_fused_expert_w2_b.as_ref(), self.d_fused_expert_w2s_b.as_ref())
                        };
                        if let (Some(w1b), Some(w1sb), Some(w2b), Some(w2sb)) = (w1_buf, w1s_buf, w2_buf, w2s_buf) {
                            let e = &md.experts[eid];
                            let w1_off = eid * self.w1_packed_per_expert;
                            let w1s_off = eid * self.w1_scales_per_expert;
                            let w2_off = eid * self.w2_packed_per_expert;
                            let w2s_off = eid * self.w2_scales_per_expert;
                            unsafe {
                                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                    *w1b.device_ptr() + w1_off as u64, e.w13_packed_ptr as *const _,
                                    e.w13_packed_bytes, self.copy_stream);
                                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                    *w1sb.device_ptr() + w1s_off as u64, e.w13_scales_ptr as *const _,
                                    e.w13_scales_bytes, self.copy_stream);
                                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                    *w2b.device_ptr() + w2_off as u64, e.w2_packed_ptr as *const _,
                                    e.w2_packed_bytes, self.copy_stream);
                                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                    *w2sb.device_ptr() + w2s_off as u64, e.w2_scales_ptr as *const _,
                                    e.w2_scales_bytes, self.copy_stream);
                            }
                            self.h_expert_w1_ptrs[eid] = *w1b.device_ptr() + w1_off as u64;
                            self.h_expert_w1s_ptrs[eid] = *w1sb.device_ptr() + w1s_off as u64;
                            self.h_expert_w2_ptrs[eid] = *w2b.device_ptr() + w2_off as u64;
                            self.h_expert_w2s_ptrs[eid] = *w2sb.device_ptr() + w2s_off as u64;
                            cold_count += 1;
                        }
                    } else if eid < md.experts.len() && cold_slot < self.max_cold_experts {
                        let e = &md.experts[eid];
                        let slot_base = cold_staging_base + (cold_slot * self.cold_expert_bytes) as u64;
                        let mut off = 0u64;
                        // w1 packed
                        unsafe {
                            cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                slot_base + off, e.w13_packed_ptr as *const _,
                                e.w13_packed_bytes, self.copy_stream);
                        }
                        self.h_expert_w1_ptrs[eid] = slot_base + off;
                        off += self.w1_packed_per_expert as u64;
                        // w1 scales
                        unsafe {
                            cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                slot_base + off, e.w13_scales_ptr as *const _,
                                e.w13_scales_bytes, self.copy_stream);
                        }
                        self.h_expert_w1s_ptrs[eid] = slot_base + off;
                        off += self.w1_scales_per_expert as u64;
                        // w2 packed
                        unsafe {
                            cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                slot_base + off, e.w2_packed_ptr as *const _,
                                e.w2_packed_bytes, self.copy_stream);
                        }
                        self.h_expert_w2_ptrs[eid] = slot_base + off;
                        off += self.w2_packed_per_expert as u64;
                        // w2 scales
                        unsafe {
                            cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                slot_base + off, e.w2_scales_ptr as *const _,
                                e.w2_scales_bytes, self.copy_stream);
                        }
                        self.h_expert_w2s_ptrs[eid] = slot_base + off;
                        cold_slot += 1;
                        cold_count += 1;
                    }
                }
            }

            // Upload pointer tables to GPU using SYNCHRONOUS copy.
            // cudarc pool allocations can be corrupted by HCS raw cuMemAlloc_v2 allocations,
            // so we must verify the data reaches the correct GPU address.
            // Expert H2D staging was already on copy_stream; sync it first.
            unsafe {
                cuda_sys::lib().cuStreamSynchronize(self.copy_stream);
                // Now use synchronous cuMemcpyHtoD on the default stream
                cuda_sys::lib().cuMemcpyHtoD_v2(
                    w1_ptrs_gpu, self.h_expert_w1_ptrs.as_ptr() as *const _,
                    n_experts * 8);
                cuda_sys::lib().cuMemcpyHtoD_v2(
                    w1s_ptrs_gpu, self.h_expert_w1s_ptrs.as_ptr() as *const _,
                    n_experts * 8);
                cuda_sys::lib().cuMemcpyHtoD_v2(
                    w2_ptrs_gpu, self.h_expert_w2_ptrs.as_ptr() as *const _,
                    n_experts * 8);
                cuda_sys::lib().cuMemcpyHtoD_v2(
                    w2s_ptrs_gpu, self.h_expert_w2s_ptrs.as_ptr() as *const _,
                    n_experts * 8);
                cuda_sys::lib().cuEventRecord(self.dma_event, self.copy_stream);
            }

            if layer_idx == 0 {
                // Dump host-side pointer table to verify correctness
                let first_active = active.first().copied().unwrap_or(0);
                if stderr_debug_enabled() || debug_prefill {
                    eprintln!("[PTR-TABLE] layer0: {} hcs + {} pinned + {} cold, active={}/{} pinning_active={}",
                        hcs_count, pinned_count, cold_count, active.len(), n_experts, self.pinning_active);
                    eprintln!("[PTR-TABLE] host ptrs[0..4] = [{:#x}, {:#x}, {:#x}, {:#x}]",
                        self.h_expert_w1_ptrs[0], self.h_expert_w1_ptrs[1],
                        self.h_expert_w1_ptrs[2], self.h_expert_w1_ptrs[3]);
                    eprintln!("[PTR-TABLE] host ptrs first_active[{}] = {:#x}",
                        first_active, self.h_expert_w1_ptrs[first_active]);
                    eprintln!("[PTR-TABLE] gpu buf addr={:#x}, n_experts={}, upload_bytes={}",
                        w1_ptrs_gpu, n_experts, n_experts * 8);
                }
            }
        } else {
            // Legacy path: contiguous fused buffer DMA
            if self.preloaded_moe_layer != Some(layer_idx) {
                if let Some(moe_idx) = moe_layer_idx {
                    if let Some(Some(moe_data)) = self.moe_layers.get(moe_idx) {
                        if !self.prescan_active_experts.is_empty() || self.hcs_num_experts_per_layer > 0 {
                            let active: Vec<usize> = h_expert_counts.iter().enumerate()
                                .filter_map(|(eid, &cnt)| if cnt > 0 { Some(eid) } else { None })
                                .collect();
                            let mi = (0..self.config.num_hidden_layers)
                                .filter(|&j| self.layer_weights[j].moe_gate_ptr != 0)
                                .position(|j| j == layer_idx)
                                .unwrap_or(0);
                            let (h, p, c) = self.selective_dma_layer(
                                moe_data, layer_idx, mi,
                                w1_base, w1s_base, w2_base, w2s_base,
                                &active)?;
                            let _ = (h, p, c);
                        } else {
                            self.bulk_dma_layer(moe_data, w1_base, w1s_base, w2_base, w2s_base)?;
                        }
                        unsafe {
                            cuda_sys::lib().cuEventRecord(self.dma_event, self.copy_stream);
                        }
                    }
                }
            }
        }

        // Wait for DMA/pointer table upload to complete (stream dependency)
        unsafe {
            cuda_sys::lib().cuStreamWaitEvent(self.stream, self.dma_event, 0);
        }
        if let Some(t) = mt1 {
            // Sync to measure actual DMA wait time (timing mode only)
            self.stream_sync()?;
            self.t_moe_dma.set(self.t_moe_dma.get() + t.elapsed().as_secs_f64() * 1000.0);
        }
        let mt2 = if mt { Some(Instant::now()) } else { None };

        // Double-buffer overlap: only needed in legacy (non-pointer-table) mode.
        // In pointer table mode, each layer builds its tables from HCS + cold staging
        // dynamically, so no preloading is needed.
        if !use_ptr_table {
            let num_layers = self.config.num_hidden_layers;
            if self.d_fused_expert_w1_b.is_some() {
                self.preload_next_moe_layer(layer_idx, num_layers)?;
            }
        }

        // Verify DMA: check first 16 bytes of fused buffer are non-zero
        if diag_moe && layer_idx == 0 {
            self.stream_sync()?;
            let mut h_fused = vec![0u32; 4];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_fused.as_mut_ptr() as *mut _, w1_base, 16);
            }
            let is_zero = h_fused.iter().all(|&v| v == 0);
            eprintln!("[FUSED-DMA] layer0 w1_base[0..4] = {:?} (zero={})", h_fused, is_zero);
        }

        // 5. Launch shared expert on dedicated shared_stream (async overlap with fused MoE)
        if has_shared {
            unsafe {
                cuda_sys::lib().cuEventRecord(self.compute_event, self.stream);
                cuda_sys::lib().cuStreamWaitEvent(self.shared_stream, self.compute_event, 0);
            }
            self.launch_shared_expert_on_shared_stream(layer_idx, m)?;
        }

        // 6. Replicate hidden states: [m, h] -> [m*topk, h]
        // Required for top_k=1 trick: avoids C_tmp collision in fp32_reduce.
        // Each (token, slot) pair needs its own entry so C_tmp positions are unique.
        let fused_input_ptr = *self.scratch.d_moe_gathered.device_ptr();
        let m_topk = m * topk;
        {
            let rt = std::cmp::max(32, ((std::cmp::min(1024, h) + 31) / 32) * 32) as u32;
            let mut r0 = fused_input_ptr;
            let mut r1 = hidden;
            let mut r2 = h as i32;
            let mut r3 = m as i32;
            let mut r4 = topk as i32;
            unsafe {
                launch(self.kernels.moe_replicate_hidden,
                    (m_topk as u32, 1, 1), (rt, 1, 1), 0, self.stream,
                    &mut [
                        &mut r0 as *mut _ as *mut std::ffi::c_void,
                        &mut r1 as *mut _ as *mut std::ffi::c_void,
                        &mut r2 as *mut _ as *mut std::ffi::c_void,
                        &mut r3 as *mut _ as *mut std::ffi::c_void,
                        &mut r4 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // 7. Call MarlinDefault for w1 (gate_up projection)
        // Both w1 and w2 use top_k=1 trick: sorted_id is used as direct index.
        // For w1: A = fused_input (replicated), kernel reads A[sorted_id] directly.
        let fused_fn = self.kernels.fused_moe_fn.ok_or("fused MoE fn not loaded")?;
        let w1_n = if gated { 2 * inter } else { inter };
        let num_groups_w1 = (h / gs) as i32;
        let fused_inter_ptr = *self.scratch.d_moe_gate_up.device_ptr();
        let fused_c_tmp_ptr = *self.scratch.d_moe_accum.device_ptr();

        // Stream dependency: compute stream already waits on DMA event (above).
        // Routing data is on self.stream (same as compute). No ctx sync needed.
        let stream_ptr = self.stream as *mut std::ffi::c_void;

        let gs = self.config.group_size as i32;
        let q_type_ptr = &self.q_type as *const ScalarType as *const std::ffi::c_void;
        let w1_ctmp_floats = self.config.fused_moe_w1_ctmp_floats;
        let w2_ctmp_floats = self.config.fused_moe_w2_ctmp_floats;

        // Zero the workspace lock array and C_tmp before fused w1 GEMM.
        let ws_ptr = *self.scratch.d_workspace.device_ptr();
        let ws_len = self.config.sms * MARLIN_MAX_LOCK_SLOTS_PER_SM;
        unsafe {
            cuda_sys::lib().cuMemsetD32Async(ws_ptr, 0, ws_len, self.stream);
            // Zero C_tmp (FP32 scratch) — Marlin uses this for fp32_reduce accumulation
            cuda_sys::lib().cuMemsetD32Async(fused_c_tmp_ptr, 0, w1_ctmp_floats, self.stream);
        }

        // Pointer-table mode supplies absolute per-expert addresses for weights/scales.
        // Use a null base so the device-side ptr-table path derives addresses from the
        // active expert pointers rather than a mixed-source cold-staging fallback.
        let w1_b_param = if use_ptr_table { 0 } else { w1_base };
        let w1s_b_param = if use_ptr_table { 0 } else { w1s_base };
        let w2_b_param = if use_ptr_table { 0 } else { w2_base };
        let w2s_b_param = if use_ptr_table { 0 } else { w2s_base };

        // w1: A = fused_input (replicated [m*topk, h]), top_k=1 so kernel reads A[sorted_id] directly
        //     C written at sorted_id positions (token*topk+slot), range [0, m*topk)
        //     C_tmp also at sorted_id positions (unique, no collision in fp32_reduce)
        unsafe {
            fused_fn(
                fused_input_ptr as *const _,         // A: [m*topk, K=hidden] replicated
                w1_b_param as *const _,              // B: base ref (ptr table overrides per-expert)
                fused_inter_ptr as *mut _,           // C: written at sorted_id positions [0..m*topk)
                fused_c_tmp_ptr as *mut _,           // C_tmp
                std::ptr::null(),                    // b_bias (none)
                w1s_b_param as *const _,             // scales: base ref (ptr table overrides)
                std::ptr::null(),                    // s2 (none)
                std::ptr::null(),                    // zp (none)
                std::ptr::null(),                    // g_idx (none)
                std::ptr::null(),                    // perm (none)
                std::ptr::null(),                    // a_tmp (none)
                sorted_ids_val as *const _,         // sorted_ids (vLLM format: token*topk+slot)
                fused_expert_ids_val as *const _,   // expert_ids
                num_tokens_post_val as *const _,    // num_tokens_post_padded
                topk_weights_ptr as *const _,        // topk_weights
                block_size,                          // moe_block_size
                1i32,                                // top_k=1: sorted_id/1 = direct index into A
                false,                               // mul_topk_weights (false for w1)
                false,                               // is_ep
                m_topk as i32,                       // size_m = m*topk (padding threshold = m*topk*1)
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
                stream_ptr,                          // stream_ptr (our compute stream)
                -1,                                  // thread_k (auto)
                -1,                                  // thread_n (auto)
                self.config.sms as i32,              // sms
                false,                               // use_atomic
                true,                                // fp32_reduce
                false,                               // is_zp_float
                if use_ptr_table { w1_ptrs_gpu as *const _ } else { std::ptr::null() },   // B_expert_ptrs
                if use_ptr_table { w1s_ptrs_gpu as *const _ } else { std::ptr::null() },  // S_expert_ptrs
            );
        }
        // DIAG: Compare MoE Marlin vs Regular Marlin for one expert
        // (Skip in pointer table mode -- fused buffer offsets not valid)
        if diag_moe && layer_idx == 0 && !use_ptr_table {
            self.stream_sync()?;
            // Download sorted_ids and expert_ids to find first expert block
            let mut h_sorted = vec![0i32; std::cmp::min(64, total_sorted)];
            let mut h_expert_ids = vec![0i32; std::cmp::min(1, total_sorted / 64 + 1)];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_sorted.as_mut_ptr() as *mut _, sorted_ids_val, (h_sorted.len() * 4) as usize);
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_expert_ids.as_mut_ptr() as *mut _, fused_expert_ids_val, (h_expert_ids.len() * 4) as usize);
            }
            let first_expert = h_expert_ids[0] as usize;
            let first_sid = h_sorted[0]; // sorted_id for first token of first expert
            let first_token = first_sid / topk as i32;
            eprintln!("[DIAG FUSED L0] first expert block: expert={}, sorted_id={}, token={}",
                first_expert, first_sid, first_token);

            // Read fused w1 output at sorted_id position
            let w1_n_check = if gated { 2 * inter } else { inter };
            let fused_w1_offset = (first_sid as usize) * w1_n_check * 2;
            let mut h_fused_w1 = vec![0u16; std::cmp::min(8, w1_n_check)];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_fused_w1.as_mut_ptr() as *mut _,
                    fused_inter_ptr + fused_w1_offset as u64, (h_fused_w1.len() * 2) as usize);
            }
            let fused_vals: Vec<f32> = h_fused_w1.iter()
                .map(|&v| half::bf16::from_bits(v).to_f32()).collect();

            // Now run the REGULAR Marlin GEMM for the same expert and token for comparison
            // Use the fused buffer's expert weights (same as what MoE kernel used)
            let ref_w13 = MarlinWeight {
                packed: w1_base + (first_expert * self.w1_packed_per_expert) as u64,
                scales: w1s_base + (first_expert * self.w1_scales_per_expert) as u64,
                n: w1_n, k: h,
                num_groups: (h / self.config.group_size),
                group_size: self.config.group_size, num_bits: bits,
            };
            // A = hidden state for first_token
            let a_ptr = hidden + (first_token as usize * h * 2) as u64;
            // C = write to a temp area (reuse some scratch)
            let ref_c = *self.scratch.d_scratch2.device_ptr();
            self.marlin_gemm(a_ptr, &ref_w13, ref_c, 1)?;
            self.stream_sync()?;
            let mut h_ref_w1 = vec![0u16; std::cmp::min(8, w1_n_check)];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_ref_w1.as_mut_ptr() as *mut _, ref_c, (h_ref_w1.len() * 2) as usize);
            }
            let ref_vals: Vec<f32> = h_ref_w1.iter()
                .map(|&v| half::bf16::from_bits(v).to_f32()).collect();

            // Also compute full-row L2 for both
            let mut h_fused_full = vec![0u16; w1_n_check];
            let mut h_ref_full = vec![0u16; w1_n_check];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_fused_full.as_mut_ptr() as *mut _,
                    fused_inter_ptr + fused_w1_offset as u64, (w1_n_check * 2) as usize);
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_ref_full.as_mut_ptr() as *mut _, ref_c, (w1_n_check * 2) as usize);
            }
            let fused_l2: f32 = h_fused_full.iter().map(|&v| {
                let f = half::bf16::from_bits(v).to_f32(); f * f
            }).sum::<f32>().sqrt();
            let ref_l2: f32 = h_ref_full.iter().map(|&v| {
                let f = half::bf16::from_bits(v).to_f32(); f * f
            }).sum::<f32>().sqrt();

            eprintln!("[DIAG FUSED L0] w1_moe[expert{}][0..8]  = {:?}, L2={:.4}",
                first_expert, fused_vals, fused_l2);
            eprintln!("[DIAG FUSED L0] w1_reg[expert{}][0..8]  = {:?}, L2={:.4}",
                first_expert, ref_vals, ref_l2);
            let match_count = fused_vals.iter().zip(ref_vals.iter())
                .filter(|&(a, b)| (*a - *b).abs() < 0.01).count();
            eprintln!("[DIAG FUSED L0] w1 match: {}/{} values within 0.01", match_count, fused_vals.len());
        }

        // 9. Activation (silu_mul or relu2) on fused_inter -> fused_inter2
        // w1 writes C at sorted_id positions [0..m*topk), so activation grid = m_topk
        let fused_inter2_ptr = *self.scratch.d_moe_inter.device_ptr();
        if gated {
            let act_t = std::cmp::max(32, ((std::cmp::min(1024, inter) + 31) / 32) * 32) as u32;
            let mut ac0 = fused_inter2_ptr;
            let mut ac1 = fused_inter_ptr;
            let mut ac2 = inter as i32;
            let kernel = if activation == 1 { self.kernels.relu2 } else { self.kernels.silu_mul };
            unsafe {
                launch(kernel,
                    (m_topk as u32, 1, 1), (act_t, 1, 1), 0, self.stream,
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
                    (m_topk as u32, 1, 1), (act_t as u32, 1, 1), 0, self.stream,
                    &mut [
                        &mut ac0 as *mut _ as *mut std::ffi::c_void,
                        &mut ac1 as *mut _ as *mut std::ffi::c_void,
                        &mut ac2 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        if let Some(t) = mt2 {
            self.stream_sync()?;
            self.t_moe_w1.set(self.t_moe_w1.get() + t.elapsed().as_secs_f64() * 1000.0);
        }
        let mt3 = if mt { Some(Instant::now()) } else { None };

        // 10. MarlinDefault for w2 (down projection)
        // w2 trick: top_k=1, size_m=m*topk so kernel reads A[sorted_id/1] = A[sorted_id] directly
        // This lets each (token,expert) pair access its own intermediate result
        let fused_output_ptr = *self.scratch.d_moe_expert_out.device_ptr();
        let num_groups_w2 = (inter / self.config.group_size) as i32;
        let w2_input = if gated { fused_inter2_ptr } else { fused_inter_ptr };

        // Zero workspace and C_tmp before fused w2 GEMM
        unsafe {
            cuda_sys::lib().cuMemsetD32Async(ws_ptr, 0, ws_len, self.stream);
            cuda_sys::lib().cuMemsetD32Async(fused_c_tmp_ptr, 0, w2_ctmp_floats, self.stream);
        }
        unsafe {
            fused_fn(
                w2_input as *const _,                // A: [m*topk, K=inter] indexed by sorted_id
                w2_b_param as *const _,               // B: base ref (ptr table overrides per-expert)
                fused_output_ptr as *mut _,           // C: written at sorted_id positions [0..m*topk)
                fused_c_tmp_ptr as *mut _,            // C_tmp
                std::ptr::null(),                     // b_bias (none)
                w2s_b_param as *const _,              // scales: base ref (ptr table overrides)
                std::ptr::null(),                     // s2 (none)
                std::ptr::null(),                     // zp (none)
                std::ptr::null(),                     // g_idx (none)
                std::ptr::null(),                     // perm (none)
                std::ptr::null(),                     // a_tmp (none)
                sorted_ids_val as *const _,          // sorted_ids (vLLM format)
                fused_expert_ids_val as *const _,    // expert_ids
                num_tokens_post_val as *const _,     // num_tokens_post_padded
                topk_weights_ptr as *const _,         // topk_weights
                block_size,                           // moe_block_size
                1i32,                                 // top_k=1: sorted_id/1 = direct index into A
                false,                                // mul_topk_weights=false (scatter handles it)
                false,                                // is_ep
                m_topk as i32,                        // size_m = m*topk (padding threshold = m*topk*1)
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
                stream_ptr,                           // stream_ptr
                -1,                                   // thread_k (auto)
                -1,                                   // thread_n (auto)
                self.config.sms as i32,               // sms
                false,                                // use_atomic
                true,                                 // fp32_reduce
                false,                                // is_zp_float
                if use_ptr_table { w2_ptrs_gpu as *const _ } else { std::ptr::null() },   // B_expert_ptrs
                if use_ptr_table { w2s_ptrs_gpu as *const _ } else { std::ptr::null() },  // S_expert_ptrs
            );
        }
        if let Some(t) = mt3 {
            self.stream_sync()?;
            self.t_moe_w2.set(self.t_moe_w2.get() + t.elapsed().as_secs_f64() * 1000.0);
        }
        let mt4 = if mt { Some(Instant::now()) } else { None };

        // DIAG: compare fused w2 output vs ORIGINAL weights (HCS/cold) for token 0's experts
        // (Skip in pointer table mode)
        if diag_moe && layer_idx == 0 && !use_ptr_table {
            self.stream_sync()?;
            let mut h_ti = vec![0i32; topk];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_ti.as_mut_ptr() as *mut _, topk_ids_ptr, (topk * 4) as usize);
            }
            let ref_c1 = *self.scratch.d_scratch2.device_ptr();
            let ref_act_out = ref_c1 + (w1_n * 2) as u64;
            let ref_c2 = ref_act_out + (inter * 2) as u64;
            let act_t = std::cmp::max(32, ((std::cmp::min(1024, inter) + 31) / 32) * 32) as u32;

            for slot in 0..topk {
                let eid = h_ti[slot] as usize;
                // Get ORIGINAL weight locations (HCS or cold), NOT from fused buffer
                let (orig_w1p, orig_w1s, orig_w2p, orig_w2s) =
                    if let Some(moe_idx) = moe_layer_idx {
                        if let Some((hw1p, hw1s, hw2p, hw2s)) = self.expert_lookup(moe_idx, eid) {
                            (hw1p, hw1s, hw2p, hw2s)
                        } else {
                            // Cold expert: skip (would need H2D copy)
                            let fused_slot_ptr = fused_output_ptr + (slot * h * 2) as u64;
                            let fused_l2: f32 = {
                                let mut buf = vec![0u16; h];
                                unsafe { cuda_sys::lib().cuMemcpyDtoH_v2(
                                    buf.as_mut_ptr() as *mut _, fused_slot_ptr, (h * 2) as usize); }
                                buf.iter().map(|&v| { let f = half::bf16::from_bits(v).to_f32(); f*f }).sum::<f32>().sqrt()
                            };
                            eprintln!("[DIAG FUSED L0] slot{} expert={}: fused_L2={:.4} (COLD, skip orig comparison)", slot, eid, fused_l2);
                            continue;
                        }
                    } else { continue; };

                // Run pipeline with ORIGINAL weights
                let orig_w13 = MarlinWeight {
                    packed: orig_w1p, scales: orig_w1s,
                    n: w1_n, k: h,
                    num_groups: (h / self.config.group_size),
                    group_size: self.config.group_size, num_bits: bits,
                };
                self.marlin_gemm(hidden, &orig_w13, ref_c1, 1)?;
                if gated {
                    let mut ac0 = ref_act_out; let mut ac1 = ref_c1; let mut ac2 = inter as i32;
                    let kernel = if activation == 1 { self.kernels.relu2 } else { self.kernels.silu_mul };
                    unsafe {
                        launch(kernel, (1, 1, 1), (act_t, 1, 1), 0, self.stream,
                            &mut [&mut ac0 as *mut _ as *mut std::ffi::c_void,
                                  &mut ac1 as *mut _ as *mut std::ffi::c_void,
                                  &mut ac2 as *mut _ as *mut std::ffi::c_void])?;
                    }
                }
                let orig_w2 = MarlinWeight {
                    packed: orig_w2p, scales: orig_w2s,
                    n: h, k: inter,
                    num_groups: (inter / self.config.group_size),
                    group_size: self.config.group_size, num_bits: bits,
                };
                let w2_in = if gated { ref_act_out } else { ref_c1 };
                self.marlin_gemm(w2_in, &orig_w2, ref_c2, 1)?;
                self.stream_sync()?;

                let fused_slot_ptr = fused_output_ptr + (slot * h * 2) as u64;
                let mut h_fused = vec![0u16; h];
                let mut h_orig = vec![0u16; h];
                unsafe {
                    cuda_sys::lib().cuMemcpyDtoH_v2(h_fused.as_mut_ptr() as *mut _, fused_slot_ptr, (h * 2) as usize);
                    cuda_sys::lib().cuMemcpyDtoH_v2(h_orig.as_mut_ptr() as *mut _, ref_c2, (h * 2) as usize);
                }
                let fused_l2: f32 = h_fused.iter().map(|&v| { let f = half::bf16::from_bits(v).to_f32(); f*f }).sum::<f32>().sqrt();
                let orig_l2: f32 = h_orig.iter().map(|&v| { let f = half::bf16::from_bits(v).to_f32(); f*f }).sum::<f32>().sqrt();
                let diff_l2: f32 = h_fused.iter().zip(h_orig.iter()).map(|(&a, &b)| {
                    let fa = half::bf16::from_bits(a).to_f32();
                    let fb = half::bf16::from_bits(b).to_f32();
                    (fa - fb) * (fa - fb)
                }).sum::<f32>().sqrt();
                let status = if diff_l2 < 0.01 { "MATCH" } else if diff_l2 < orig_l2 * 0.1 { "CLOSE" } else { "DIFFER" };
                eprintln!("[DIAG FUSED L0] slot{} expert={}: fused_L2={:.4} orig_L2={:.4} diff_L2={:.4} ({})",
                    slot, eid, fused_l2, orig_l2, diff_l2, status);
            }
        }

        // DIAG: manually compute accum for token 0 from fused_output and compare with kernel
        if diag_moe && layer_idx == 0 {
            self.stream_sync()?;
            // Read fused_output for token 0's 10 slots (positions 0..topk-1)
            let mut manual_accum = vec![0.0f32; h];
            let mut h_tw = vec![0.0f32; topk];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_tw.as_mut_ptr() as *mut _, topk_weights_ptr, (topk * 4) as usize);
            }
            for k in 0..topk {
                let mut h_row = vec![0u16; h];
                unsafe {
                    cuda_sys::lib().cuMemcpyDtoH_v2(
                        h_row.as_mut_ptr() as *mut _,
                        fused_output_ptr + (k * h * 2) as u64, (h * 2) as usize);
                }
                let w = h_tw[k] * scale_factor;
                let row_l2: f32 = h_row.iter().map(|&v| {
                    let f = half::bf16::from_bits(v).to_f32(); f * f
                }).sum::<f32>().sqrt();
                eprintln!("[DIAG FUSED L0] fused_output[slot{}] L2={:.4}, topk_w={:.4}",
                    k, row_l2, h_tw[k]);
                for i in 0..h {
                    manual_accum[i] += half::bf16::from_bits(h_row[i]).to_f32() * w;
                }
            }
            let manual_l2: f32 = manual_accum.iter().map(|v| v*v).sum::<f32>().sqrt();
            eprintln!("[DIAG FUSED L0] manual_accum[tok0][0..8] = {:?}, L2={:.4}",
                &manual_accum[..8], manual_l2);
        }

        // 11. Zero accumulator then scatter-add with topk_weights * scale_factor
        // moe_scatter_weighted iterates m*topk entries (no padding waste)
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
        if m_topk > 0 {
            let st = std::cmp::max(32, ((std::cmp::min(1024, h) + 31) / 32) * 32) as u32;
            let mut s0 = moe_accum;
            let mut s1 = fused_output_ptr;
            let mut s2 = topk_weights_ptr;
            let mut s3 = h as i32;
            let mut s4 = m as i32;
            let mut s5 = topk as i32;
            let mut s6 = scale_factor;
            unsafe {
                launch(self.kernels.moe_scatter_weighted,
                    (m_topk as u32, 1, 1), (st, 1, 1), 0, self.stream,
                    &mut [
                        &mut s0 as *mut _ as *mut std::ffi::c_void,
                        &mut s1 as *mut _ as *mut std::ffi::c_void,
                        &mut s2 as *mut _ as *mut std::ffi::c_void,
                        &mut s3 as *mut _ as *mut std::ffi::c_void,
                        &mut s4 as *mut _ as *mut std::ffi::c_void,
                        &mut s5 as *mut _ as *mut std::ffi::c_void,
                        &mut s6 as *mut _ as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // DIAG: check accum after scatter (before shared expert)
        if diag_moe && layer_idx == 0 {
            self.stream_sync()?;
            let mut h_acc = vec![0.0f32; std::cmp::min(8, h)];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_acc.as_mut_ptr() as *mut _, moe_accum, (h_acc.len() * 4) as usize);
            }
            // Also check L2 norm of full accum for token 0
            let mut h_acc_full = vec![0.0f32; h];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_acc_full.as_mut_ptr() as *mut _, moe_accum, (h * 4) as usize);
            }
            let norm: f32 = h_acc_full.iter().map(|v| v*v).sum::<f32>().sqrt();
            eprintln!("[DIAG FUSED L0] accum[tok0][0..8] = {:?}, L2={:.4}", h_acc, norm);
        }

        if let Some(t) = mt4 {
            self.stream_sync()?;
            self.t_moe_scatter.set(self.t_moe_scatter.get() + t.elapsed().as_secs_f64() * 1000.0);
        }
        let mt5 = if mt { Some(Instant::now()) } else { None };

        // 12. Wait for shared expert on shared_stream, then add to accumulator with sigmoid gate
        if has_shared {
            // Wait for shared expert to complete on shared_stream
            unsafe {
                cuda_sys::lib().cuEventRecord(self.shared_event, self.shared_stream);
                cuda_sys::lib().cuStreamWaitEvent(self.stream, self.shared_event, 0);
            }
            let s1_buf = *self.scratch.d_scratch1.device_ptr();
            let lw_ref = &self.layer_weights[layer_idx];

            // Add shared expert to accumulator (with optional sigmoid gate)
            let sg_ptr = lw_ref.shared_gate_ptr;
            if sg_ptr != 0 {
                // Compute shared gate: hidden [m, h] @ gate [h, 1] -> gate_out [m, 1] FP32
                let gate_out = *self.scratch.d_gate_out.device_ptr();
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
                        lw_ref.shared_gate_rows as i32, m as i32, lw_ref.shared_gate_cols as i32,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        sg_ptr as *const std::ffi::c_void,
                        cublas_sys::cudaDataType::CUDA_R_16BF, lw_ref.shared_gate_cols as i32,
                        hidden as *const std::ffi::c_void,
                        cublas_sys::cudaDataType::CUDA_R_16BF, h as i32,
                        &beta as *const f32 as *const std::ffi::c_void,
                        gate_out as *mut std::ffi::c_void,
                        cublas_sys::cudaDataType::CUDA_R_32F, lw_ref.shared_gate_rows as i32,
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

        // DIAG: accum after shared expert add
        if diag_moe && layer_idx == 0 {
            self.stream_sync()?;
            let mut h_acc = vec![0.0f32; h];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_acc.as_mut_ptr() as *mut _, moe_accum, (h * 4) as usize);
            }
            let norm: f32 = h_acc.iter().map(|v| v*v).sum::<f32>().sqrt();
            eprintln!("[DIAG FUSED L0] accum_after_shared[tok0] L2={:.4} first4={:?}", norm, &h_acc[..4]);
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

        if let Some(t) = mt5 {
            self.stream_sync()?;
            self.t_moe_shared.set(self.t_moe_shared.get() + t.elapsed().as_secs_f64() * 1000.0);
        }

        // DIAG: d_hidden after bf16 conversion
        if diag_moe && layer_idx == 0 {
            self.stream_sync()?;
            let mut h_hid = vec![0u16; h];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_hid.as_mut_ptr() as *mut _, hidden, (h * 2) as usize);
            }
            let norm: f32 = h_hid.iter().map(|&v| {
                let f = half::bf16::from_bits(v).to_f32(); f*f
            }).sum::<f32>().sqrt();
            eprintln!("[DIAG FUSED L0] hidden_after_bf16[tok0] L2={:.4} first4={:?}",
                norm, h_hid[..4].iter().map(|&v| half::bf16::from_bits(v).to_f32()).collect::<Vec<f32>>());
        }

        // NOTE: Pin-as-you-go caching is deferred to rolling-scan pipeline (multi-chunk only).
        // Single-chunk prompts have no cross-chunk reuse, so pinning wastes VRAM and fragments
        // the allocator. The fused_pin_queue data is available when rolling-scan is implemented.

        // Free per-layer pointer table GPU buffers (allocated with raw cuMemAlloc_v2)
        if use_ptr_table {
            unsafe {
                cuda_sys::lib().cuMemFree_v2(w1_ptrs_gpu);
                cuda_sys::lib().cuMemFree_v2(w1s_ptrs_gpu);
                cuda_sys::lib().cuMemFree_v2(w2_ptrs_gpu);
                cuda_sys::lib().cuMemFree_v2(w2s_ptrs_gpu);
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
        let diag_moe_detail = std::env::var("KRASIS_PREFILL_DIAG_MOE_DETAIL").is_ok();
        if diag && diag_moe_detail && layer_idx < diag_layer_limit {
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
        if diag && diag_moe_detail && layer_idx < diag_layer_limit && total_active > 0 {
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
        let has_shared = (lw.shared_w1.is_some() && lw.shared_w2.is_some())
            || (lw.shared_w1_bf16.is_some() && lw.shared_w2_bf16.is_some());

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
        let num_groups_w13 = if gs > 0 { h / gs } else { 0 };
        let num_groups_w2 = if gs > 0 { inter / gs } else { 0 };

        // Collect active experts, check cache residency (prefill cache -> HCS -> cold)
        struct ExpertWork {
            eid: usize,
            count: usize,
            offset: usize,
            w13_packed: u64,
            w13_scales: u64,
            w2_packed: u64,
            w2_scales: u64,
            is_cached: bool,  // true = in prefill cache or HCS (zero-copy, no DMA needed)
        }
        let mut work: Vec<ExpertWork> = Vec::new();

        for eid in 0..n_experts {
            let count = if eid + 1 <= n_experts {
                expert_offsets[eid + 1] - expert_offsets[eid]
            } else { 0 };
            if count == 0 { continue; }
            let eo = expert_offsets[eid];

            if let Some(moe_idx) = moe_layer_idx {
                if let Some((w13p, w13s, w2p, w2s)) = self.expert_lookup(moe_idx, eid) {
                    work.push(ExpertWork {
                        eid, count, offset: eo,
                        w13_packed: w13p, w13_scales: w13s,
                        w2_packed: w2p, w2_scales: w2s,
                        is_cached: true,
                    });
                    continue;
                }
            }

            work.push(ExpertWork {
                eid, count, offset: eo,
                w13_packed: 0, w13_scales: 0,
                w2_packed: 0, w2_scales: 0,
                is_cached: false,
            });
        }

        // Launch shared expert async if all experts are cached (no DMA conflict)
        // Force sync when diagnostics are active so we can add intermediate norm checks
        let all_cached = work.iter().all(|w| w.is_cached);
        let shared_async = has_shared && all_cached && !(diag && layer_idx < diag_layer_limit);
        if shared_async {
            unsafe {
                cuda_sys::lib().cuEventRecord(self.compute_event, self.stream);
                // shared_stream must wait for main stream's d_hidden to be ready
                cuda_sys::lib().cuStreamWaitEvent(self.shared_stream, self.compute_event, 0);
            }
            self.launch_shared_expert_async(layer_idx, m)?;
        }

        // Process cached experts first (no DMA — from prefill cache or HCS)
        for w in &work {
            if !w.is_cached { continue; }
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
            work.iter().filter(|w| !w.is_cached).collect()
        };
        if !cold_work.is_empty() {
            if let Some(moe_idx) = moe_layer_idx {
                if let Some(Some(moe_data)) = self.moe_layers.get(moe_idx) {
                    let mut cur_buf = 0usize;

                    // Diagnostic: log first cold expert DMA parameters + buffer sizes
                    if diag && diag_moe_detail && layer_idx < diag_layer_limit {
                        let fe = &moe_data.experts[cold_work[0].eid];
                        let buf_w13p_bytes = h * w13_n * bits as usize / 8;
                        let buf_w13s_bytes = (h / gs) * w13_n * 2;
                        let buf_w2p_bytes = inter * h * bits as usize / 8;
                        let buf_w2s_bytes = (inter / gs) * h * 2;
                        eprintln!("[DIAG] layer{:02}_moe COLD DMA eid={}: w13p_src={:#x} w13p_bytes={} (buf={}) w13s_src={:#x} w13s_bytes={} (buf={}) w2p_src={:#x} w2p_bytes={} (buf={}) w2s_src={:#x} w2s_bytes={} (buf={})",
                            layer_idx, cold_work[0].eid,
                            fe.w13_packed_ptr, fe.w13_packed_bytes, buf_w13p_bytes,
                            fe.w13_scales_ptr, fe.w13_scales_bytes, buf_w13s_bytes,
                            fe.w2_packed_ptr, fe.w2_packed_bytes, buf_w2p_bytes,
                            fe.w2_scales_ptr, fe.w2_scales_bytes, buf_w2s_bytes);
                        // Size validation
                        if fe.w13_packed_bytes > buf_w13p_bytes {
                            eprintln!("[DIAG] ERROR: w13_packed_bytes {} > buffer {}", fe.w13_packed_bytes, buf_w13p_bytes);
                        }
                        if fe.w13_scales_bytes > buf_w13s_bytes {
                            eprintln!("[DIAG] ERROR: w13_scales_bytes {} > buffer {}", fe.w13_scales_bytes, buf_w13s_bytes);
                        }
                        if fe.w2_packed_bytes > buf_w2p_bytes {
                            eprintln!("[DIAG] ERROR: w2_packed_bytes {} > buffer {}", fe.w2_packed_bytes, buf_w2p_bytes);
                        }
                        if fe.w2_scales_bytes > buf_w2s_bytes {
                            eprintln!("[DIAG] ERROR: w2_scales_bytes {} > buffer {}", fe.w2_scales_bytes, buf_w2s_bytes);
                        }
                    }

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

        // NOTE: Pin-as-you-go caching deferred to rolling-scan pipeline (multi-chunk only).
        // pin_queue data is available when rolling-scan is implemented.

        // Diagnostic: dump per-row expert_out norms before scatter to find outliers
        if diag && diag_moe_detail && layer_idx < diag_layer_limit && total_active > 0 {
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
                    let is_cached = work.iter().find(|w| w.eid == eid).map(|w| w.is_cached).unwrap_or(false);
                    eprintln!("[DIAG]   row={} tok={} expert={} norm={:.4} wt={:.6} cached={}",
                        row, tok, eid, norm, wt, is_cached);
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

        // DIAG: sequential path accum before shared expert (compare with fused)
        if diag && layer_idx == 0 {
            self.stream_sync()?;
            let mut h_acc = vec![0.0f32; std::cmp::min(8, h)];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_acc.as_mut_ptr() as *mut _, moe_accum, (h_acc.len() * 4) as usize);
            }
            let mut h_acc_full = vec![0.0f32; h];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_acc_full.as_mut_ptr() as *mut _, moe_accum, (h * 4) as usize);
            }
            let norm: f32 = h_acc_full.iter().map(|v| v*v).sum::<f32>().sqrt();
            eprintln!("[DIAG SEQ L0] accum[tok0][0..8] = {:?}, L2={:.4}", h_acc, norm);
        }

        // 10. Shared expert: add to accumulator
        if has_shared {
            if shared_async {
                // Wait for async shared expert to finish on shared_stream
                unsafe {
                    cuda_sys::lib().cuEventRecord(self.shared_event, self.shared_stream);
                    cuda_sys::lib().cuStreamWaitEvent(self.stream, self.shared_event, 0);
                }
            } else if let (Some(sw1_bf16), Some(sw2_bf16)) = (&lw.shared_w1_bf16, &lw.shared_w2_bf16) {
                // BF16 shared expert synchronously on main stream via cuBLAS
                let s1_buf = *self.scratch.d_scratch1.device_ptr();
                let s2_buf = *self.scratch.d_scratch2.device_ptr();
                let shared_inter = sw1_bf16.n / (if gated { 2 } else { 1 });

                // w1 GEMM
                self.cublas_bf16_gemm(hidden, sw1_bf16, s1_buf, m)?;

                if gated {
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
                    // w2 GEMM
                    self.cublas_bf16_gemm(s2_buf, sw2_bf16, s1_buf, m)?;
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
                    self.cublas_bf16_gemm(s1_buf, sw2_bf16, s2_buf, m)?;
                    self.memcpy_d2d(s1_buf, s2_buf, (m * h * 2) as u64)?;
                }
            } else {
                // Run shared expert synchronously on main stream (Marlin)
                let sw1 = lw.shared_w1.as_ref().ok_or("missing shared w1")?;
                let sw2 = lw.shared_w2.as_ref().ok_or("missing shared w2")?;
                let s1_buf = *self.scratch.d_scratch1.device_ptr();
                let s2_buf = *self.scratch.d_scratch2.device_ptr();
                let shared_inter = sw1.n / (if gated { 2 } else { 1 });

                if diag && diag_moe_detail && layer_idx < diag_layer_limit {
                    eprintln!("[DIAG] layer{:02}_moe shared_expert sw1: n={} k={} bits={} gs={} groups={}",
                        layer_idx, sw1.n, sw1.k, sw1.num_bits, sw1.group_size, sw1.num_groups);
                    eprintln!("[DIAG] layer{:02}_moe shared_expert sw2: n={} k={} bits={} gs={} groups={}",
                        layer_idx, sw2.n, sw2.k, sw2.num_bits, sw2.group_size, sw2.num_groups);
                    eprintln!("[DIAG] layer{:02}_moe shared_inter={} gated={} activation={}",
                        layer_idx, shared_inter, gated, activation);
                }

                self.marlin_gemm(hidden, sw1, s1_buf, m)?;

                if gated {
                    if diag && diag_moe_detail && layer_idx < diag_layer_limit {
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

                    if diag && diag_moe_detail && layer_idx < diag_layer_limit {
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

        // BF16 validation mode: use cuBLAS on shared_stream
        if let (Some(sw1_bf16), Some(sw2_bf16)) = (&lw.shared_w1_bf16, &lw.shared_w2_bf16) {
            return self.launch_shared_expert_async_bf16(layer_idx, m, sw1_bf16, sw2_bf16);
        }

        let sw1 = lw.shared_w1.as_ref().ok_or("missing shared w1")?;
        let sw2 = lw.shared_w2.as_ref().ok_or("missing shared w2")?;
        let shared_inter = sw1.n / (if gated { 2 } else { 1 });

        let st = if sw1.num_bits == 4 { &ScalarType::U4B8 } else { &ScalarType::U8B128 };
        let f = self.kernels.marlin_mm.ok_or("Marlin GEMM not loaded")?;

        let shared_scratch = *self.d_shared_fp32_scratch.as_ref()
            .ok_or("d_shared_fp32_scratch not allocated (call prepare_for_prefill first)")?.device_ptr();
        let shared_ws = *self.d_shared_workspace.as_ref()
            .ok_or("d_shared_workspace not allocated (call prepare_for_prefill first)")?.device_ptr();
        let shared_ws_len = self.config.sms * MARLIN_MAX_LOCK_SLOTS_PER_SM;

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

    /// BF16 shared expert via cuBLAS on shared_stream.
    fn launch_shared_expert_async_bf16(
        &self, layer_idx: usize, m: usize,
        sw1: &Bf16Weight, sw2: &Bf16Weight,
    ) -> Result<(), String> {
        let lw = &self.layer_weights[layer_idx];
        let h = self.config.hidden_size;
        let gated = lw.moe_gated;
        let activation = lw.moe_activation;
        let hidden = *self.scratch.d_hidden.device_ptr();
        let s1_buf = *self.scratch.d_scratch1.device_ptr();
        let s2_buf = *self.scratch.d_scratch2.device_ptr();
        let shared_inter = sw1.n / (if gated { 2 } else { 1 });

        // w1 GEMM on shared_stream: [m, h] @ [w13_n, h]^T -> [m, w13_n]
        {
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            unsafe {
                use cudarc::cublas::sys as cublas_sys;
                use cudarc::cublas::result as cublas_result;
                cublas_result::gemm_ex(
                    self.shared_cublas_handle,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    sw1.n as i32, m as i32, sw1.k as i32,
                    &alpha as *const f32 as *const std::ffi::c_void,
                    sw1.ptr as *const std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_16BF, sw1.k as i32,
                    hidden as *const std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_16BF, sw1.k as i32,
                    &beta as *const f32 as *const std::ffi::c_void,
                    s1_buf as *mut std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_16BF, sw1.n as i32,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                ).map_err(|e| format!("shared w1 BF16 GEMM: {:?}", e))?;
            }
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
            // w2 GEMM on shared_stream
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            unsafe {
                use cudarc::cublas::sys as cublas_sys;
                use cudarc::cublas::result as cublas_result;
                cublas_result::gemm_ex(
                    self.shared_cublas_handle,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    sw2.n as i32, m as i32, sw2.k as i32,
                    &alpha as *const f32 as *const std::ffi::c_void,
                    sw2.ptr as *const std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_16BF, sw2.k as i32,
                    s2_buf as *const std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_16BF, sw2.k as i32,
                    &beta as *const f32 as *const std::ffi::c_void,
                    s1_buf as *mut std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_16BF, sw2.n as i32,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                ).map_err(|e| format!("shared w2 BF16 GEMM: {:?}", e))?;
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
            // w2 GEMM
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            unsafe {
                use cudarc::cublas::sys as cublas_sys;
                use cudarc::cublas::result as cublas_result;
                cublas_result::gemm_ex(
                    self.shared_cublas_handle,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    sw2.n as i32, m as i32, sw2.k as i32,
                    &alpha as *const f32 as *const std::ffi::c_void,
                    sw2.ptr as *const std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_16BF, sw2.k as i32,
                    s1_buf as *const std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_16BF, sw2.k as i32,
                    &beta as *const f32 as *const std::ffi::c_void,
                    s2_buf as *mut std::ffi::c_void,
                    cublas_sys::cudaDataType::CUDA_R_16BF, sw2.n as i32,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                ).map_err(|e| format!("shared w2 BF16 GEMM: {:?}", e))?;
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
    /// Bulk DMA all expert weights for a MoE layer to GPU fused buffer.
    /// Uses 4 contiguous H2D transfers (pinned host -> GPU) from per-layer backing.
    /// Falls back to per-expert DMA if contiguous backing not available.
    fn bulk_dma_layer(
        &self,
        moe_data: &PrefillMoeLayerData,
        w1_base: u64, w1s_base: u64, w2_base: u64, w2s_base: u64,
    ) -> Result<(), String> {
        if moe_data.bulk_w13p.0 != 0 {
            // Fast path: 4 bulk DMA calls from contiguous pinned host memory
            unsafe {
                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    w1_base, moe_data.bulk_w13p.0 as *const _,
                    moe_data.bulk_w13p.1, self.copy_stream);
                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    w1s_base, moe_data.bulk_w13s.0 as *const _,
                    moe_data.bulk_w13s.1, self.copy_stream);
                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    w2_base, moe_data.bulk_w2p.0 as *const _,
                    moe_data.bulk_w2p.1, self.copy_stream);
                cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                    w2s_base, moe_data.bulk_w2s.0 as *const _,
                    moe_data.bulk_w2s.1, self.copy_stream);
            }
        } else {
            // Fallback: per-expert DMA (4 calls per expert)
            for (eid, e) in moe_data.experts.iter().enumerate() {
                let w1_off = (eid * self.w1_packed_per_expert) as u64;
                let w1s_off = (eid * self.w1_scales_per_expert) as u64;
                let w2_off = (eid * self.w2_packed_per_expert) as u64;
                let w2s_off = (eid * self.w2_scales_per_expert) as u64;
                unsafe {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        w1_base + w1_off, e.w13_packed_ptr as *const _,
                        e.w13_packed_bytes, self.copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        w1s_base + w1s_off, e.w13_scales_ptr as *const _,
                        e.w13_scales_bytes, self.copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        w2_base + w2_off, e.w2_packed_ptr as *const _,
                        e.w2_packed_bytes, self.copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        w2s_base + w2s_off, e.w2_scales_ptr as *const _,
                        e.w2_scales_bytes, self.copy_stream);
                }
            }
        }
        Ok(())
    }

    /// Called after forward_moe completes for the current layer.
    /// Starts async DMA of next MoE layer's experts into the other buffer
    /// while compute continues on the current buffer.
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

        // No need to wait for compute: double-buffer ensures we write to the
        // OTHER buffer while compute uses the current one.  The copy_stream is
        // sequential, so DMA(N+1) naturally waits for DMA(N) to finish.
        // Two full DMA cycles elapse before a buffer is reused, which exceeds
        // the compute time per layer (26ms DMA >> 4.5ms compute).

        // DMA experts for next layer into the other buffer.
        // With pinning: selective DMA using pre-scan routing (pinned D2D + cold H2D).
        // Without pinning: bulk DMA (4 large contiguous transfers).
        if let Some(Some(moe_data)) = self.moe_layers.get(next_moe_idx) {
            let has_prescan = !self.prescan_active_experts.is_empty();
            let has_hcs = self.hcs_num_experts_per_layer > 0;

            if has_prescan || has_hcs {
                // Find the MoE layer index for the next layer
                let mi = (0..self.config.num_hidden_layers)
                    .filter(|&j| self.layer_weights[j].moe_gate_ptr != 0)
                    .position(|j| j == next_layer);

                if let Some(mi) = mi {
                    // Use pre-scan data: union of all chunks' active experts for this layer
                    let mut active_set = vec![false; self.config.n_routed_experts];
                    if has_prescan && mi < self.prescan_active_experts.len() {
                        for chunk_experts in &self.prescan_active_experts[mi] {
                            for &eid in chunk_experts {
                                if eid < active_set.len() { active_set[eid] = true; }
                            }
                        }
                    }
                    // Also include all pinned experts (they need D2D copy)
                    if self.pinning_active && mi < self.pinned_expert_offsets.len() {
                        for (eid, off) in self.pinned_expert_offsets[mi].iter().enumerate() {
                            if off.is_some() { active_set[eid] = true; }
                        }
                    }
                    // If no prescan data, fall back to all experts
                    let active: Vec<usize> = if active_set.iter().any(|&a| a) {
                        active_set.iter().enumerate()
                            .filter_map(|(eid, &a)| if a { Some(eid) } else { None })
                            .collect()
                    } else {
                        (0..self.config.n_routed_experts).collect()
                    };
                    let _ = self.selective_dma_layer(
                        moe_data, next_layer, mi,
                        w1_base, w1s_base, w2_base, w2s_base,
                        &active);
                } else {
                    self.bulk_dma_layer(moe_data, w1_base, w1s_base, w2_base, w2s_base)?;
                }
            } else {
                self.bulk_dma_layer(moe_data, w1_base, w1s_base, w2_base, w2s_base)?;
            }
            unsafe {
                cuda_sys::lib().cuEventRecord(self.dma_event, self.copy_stream);
            }
            self.preloaded_moe_layer = Some(next_layer);
            self.fused_expert_buf_cur = next_buf;
        }

        Ok(())
    }

    /// Gate pre-scan: run gate GEMMs + top-K for all MoE layers on embedded tokens.
    /// Returns per-MoE-layer per-expert activation count, and populates
    /// self.prescan_active_experts with per-chunk active expert lists.
    /// The input is the embedding output (d_hidden), which is an approximation
    /// of the true per-layer hidden state. Expert activation follows a power law,
    /// so even approximate routing identifies the hottest experts correctly.
    fn gate_prescan(&mut self, total_m: usize, chunk_size: usize, num_chunks: usize)
        -> Result<Vec<Vec<u32>>, String>
    {
        let h = self.config.hidden_size;
        let n_experts = self.config.n_routed_experts;
        if n_experts == 0 { return Ok(Vec::new()); }
        let topk = self.config.num_experts_per_tok;
        let num_layers = self.config.num_hidden_layers;

        // Identify MoE layers (those with gate weights)
        let moe_layer_indices: Vec<usize> = (0..num_layers)
            .filter(|&i| self.layer_weights[i].moe_gate_ptr != 0)
            .collect();
        let num_moe_layers = moe_layer_indices.len();
        if num_moe_layers == 0 { return Ok(Vec::new()); }

        // Activation counts: [moe_layer][expert] -> total token count
        let mut counts: Vec<Vec<u32>> = vec![vec![0u32; n_experts]; num_moe_layers];
        // Per-chunk active experts: [moe_layer][chunk] -> Vec<expert_id>
        let mut per_chunk: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); num_chunks]; num_moe_layers];

        // Use d_hidden as input (contains embedded tokens)
        let hidden = *self.scratch.d_hidden.device_ptr();
        let gate_out = *self.scratch.d_gate_out.device_ptr();
        let topk_ids_ptr = *self.scratch.d_topk_ids.device_ptr();
        let topk_weights_ptr = *self.scratch.d_topk_weights.device_ptr();
        let scoring_func = self.layer_weights[moe_layer_indices[0]].moe_scoring_func;

        let t_prescan = Instant::now();

        for (mi, &layer_idx) in moe_layer_indices.iter().enumerate() {
            let gate_ptr = self.layer_weights[layer_idx].moe_gate_ptr;

            // Gate GEMM: [total_m, hidden] @ [hidden, n_experts] -> [total_m, n_experts] FP32
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            unsafe {
                use cudarc::cublas::sys as cublas_sys;
                use cudarc::cublas::result as cublas_result;
                cublas_result::set_stream(
                    self.cublas_handle,
                    self.stream as cublas_sys::cudaStream_t,
                ).map_err(|e| format!("prescan cublas set_stream: {:?}", e))?;
                cublas_result::gemm_ex(
                    self.cublas_handle,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    n_experts as i32, total_m as i32, h as i32,
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
                ).map_err(|e| format!("prescan gate GEMM layer {}: {:?}", layer_idx, e))?;
            }

            // Top-K routing
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
                        (total_m as u32, 1, 1), (t, 1, 1), smem, self.stream,
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

            // Download topk_ids to CPU
            self.stream_sync()?;
            let total_topk = total_m * topk;
            if self.h_topk_ids.len() < total_topk {
                self.h_topk_ids.resize(total_topk, 0);
            }
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    self.h_topk_ids.as_mut_ptr() as *mut _,
                    topk_ids_ptr, (total_topk * 4) as usize);
            }

            // Count activations per expert, and per-chunk active sets
            let mut layer_counts = vec![0u32; n_experts];
            for chunk_idx in 0..num_chunks {
                let cstart = chunk_idx * chunk_size;
                let cend = std::cmp::min(cstart + chunk_size, total_m);
                let mut chunk_active = vec![false; n_experts];
                for tok in cstart..cend {
                    for k in 0..topk {
                        let eid = self.h_topk_ids[tok * topk + k] as usize;
                        if eid < n_experts {
                            layer_counts[eid] += 1;
                            chunk_active[eid] = true;
                        }
                    }
                }
                let active_list: Vec<usize> = (0..n_experts).filter(|&e| chunk_active[e]).collect();
                per_chunk[mi][chunk_idx] = active_list;
            }
            counts[mi] = layer_counts;
        }

        self.prescan_active_experts = per_chunk;
        let ms = t_prescan.elapsed().as_secs_f64() * 1000.0;
        let total_active: usize = counts.iter().map(|c| c.iter().filter(|&&v| v > 0).count()).sum();
        if stderr_debug_enabled() || prefill_debug_enabled() {
            eprintln!("[PREFILL] Gate pre-scan: {} MoE layers in {:.1}ms, avg {:.0} active experts/layer",
                num_moe_layers, ms, total_active as f64 / num_moe_layers as f64);
        }

        Ok(counts)
    }

    /// Allocate expert pinning pool and fill with hottest experts from pre-scan data.
    /// Returns the number of experts pinned per layer.
    fn allocate_pinning_pool(&mut self, prescan_counts: &[Vec<u32>]) -> Result<usize, String> {
        if prescan_counts.is_empty() { return Ok(0); }
        let n_experts = self.config.n_routed_experts;
        let num_moe_layers = prescan_counts.len();

        // Compute per-expert byte sizes (same as fused buffer layout)
        let expert_bytes = self.w1_packed_per_expert + self.w1_scales_per_expert
            + self.w2_packed_per_expert + self.w2_scales_per_expert;
        if expert_bytes == 0 { return Ok(0); }
        self.pinning_pool_expert_bytes = expert_bytes;

        // Measure free VRAM at runtime
        let (free, _total) = unsafe {
            let mut f = 0usize;
            let mut t = 0usize;
            cuda_sys::lib().cuMemGetInfo_v2(&mut f as *mut _, &mut t as *mut _);
            (f, t)
        };

        // Reserve safety margin (proportional to GPU size, min 512 MB)
        let safety = std::cmp::max(512 * 1024 * 1024, free / 8);
        let pool_budget = if free > safety { free - safety } else { 0 };

        // Total pinnable experts across all layers
        let max_total_experts = pool_budget / expert_bytes;
        let experts_per_layer = std::cmp::min(max_total_experts / num_moe_layers, n_experts);

        if experts_per_layer == 0 {
            if stderr_debug_enabled() {
                eprintln!("[PREFILL] Pinning pool: insufficient VRAM ({:.0} MB free, {:.1} MB safety), skipping",
                    free as f64 / 1e6, safety as f64 / 1e6);
            }
            return Ok(0);
        }

        let active_counts: Vec<usize> = prescan_counts.iter()
            .map(|layer| layer.iter().filter(|&&cnt| cnt > 0).count())
            .collect();
        let pin_counts: Vec<usize> = active_counts.iter()
            .map(|&active| std::cmp::min(active, experts_per_layer))
            .collect();
        let total_pinned: usize = pin_counts.iter().sum();
        if total_pinned == 0 {
            if stderr_debug_enabled() || prefill_debug_enabled() {
                eprintln!("[PREFILL] Pinning pool: no active experts in prescan, skipping");
            }
            return Ok(0);
        }
        let pool_bytes = total_pinned * expert_bytes;

        if stderr_debug_enabled() {
            let avg_pinned = total_pinned as f64 / num_moe_layers as f64;
            eprintln!("[PREFILL] Pinning pool: {:.0} MB free, cap {}/{} experts/layer, pinning {} total ({:.1} avg/layer, {:.0} MB)",
                free as f64 / 1e6, experts_per_layer, n_experts, total_pinned, avg_pinned, pool_bytes as f64 / 1e6);
        }

        // Allocate the pool using raw CUDA driver API
        let pool_base: u64 = unsafe {
            let mut dptr: u64 = 0;
            let err = cuda_sys::lib().cuMemAlloc_v2(&mut dptr, pool_bytes);
            if err != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuMemAlloc pinning pool {} bytes: {:?}", pool_bytes, err));
            }
            cuda_sys::lib().cuMemsetD8Async(dptr, 0, pool_bytes, self.stream);
            dptr
        };

        // For each MoE layer, sort experts by activation count and pin the top N
        let mut offsets: Vec<Vec<Option<usize>>> = vec![vec![None; n_experts]; num_moe_layers];

        // Identify MoE layers
        let moe_layer_indices: Vec<usize> = (0..self.config.num_hidden_layers)
            .filter(|&i| self.layer_weights[i].moe_gate_ptr != 0)
            .collect();

        let t_pin = Instant::now();

        let mut layer_base_offset = 0usize;
        for (mi, &layer_idx) in moe_layer_indices.iter().enumerate() {
            if mi >= num_moe_layers { break; }

            // Sort experts by activation count (descending)
            let mut ranked: Vec<(usize, u32)> = prescan_counts[mi].iter()
                .enumerate()
                .map(|(eid, &cnt)| (eid, cnt))
                .collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1));

            let moe_idx = match self.layer_weights[layer_idx].moe_layer_idx {
                Some(idx) => idx,
                None => continue,
            };

            // Pin top experts
            for rank in 0..pin_counts[mi] {
                let (eid, _cnt) = ranked[rank];
                let pool_offset = layer_base_offset + rank * expert_bytes;
                offsets[mi][eid] = Some(pool_offset);

                // DMA expert weights from CPU to pinning pool
                if let Some(Some(moe_data)) = self.moe_layers.get(moe_idx) {
                    if eid < moe_data.experts.len() {
                        let e = &moe_data.experts[eid];
                        let dst = pool_base + pool_offset as u64;
                        let mut off = 0u64;
                        unsafe {
                            cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                dst + off, e.w13_packed_ptr as *const _, e.w13_packed_bytes, self.copy_stream);
                            off += self.w1_packed_per_expert as u64;
                            cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                dst + off, e.w13_scales_ptr as *const _, e.w13_scales_bytes, self.copy_stream);
                            off += self.w1_scales_per_expert as u64;
                            cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                dst + off, e.w2_packed_ptr as *const _, e.w2_packed_bytes, self.copy_stream);
                            off += self.w2_packed_per_expert as u64;
                            cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                                dst + off, e.w2_scales_ptr as *const _, e.w2_scales_bytes, self.copy_stream);
                        }
                    }
                }
            }
            layer_base_offset += pin_counts[mi] * expert_bytes;
        }

        // Wait for all DMA to complete
        unsafe {
            cuda_sys::lib().cuStreamSynchronize(self.copy_stream);
        }

        let pin_ms = t_pin.elapsed().as_secs_f64() * 1000.0;
        if stderr_debug_enabled() || prefill_debug_enabled() {
            let avg_pinned = total_pinned as f64 / num_moe_layers as f64;
            eprintln!("[PREFILL] Pinned {} total experts across {} layers ({:.1} avg/layer) in {:.0}ms ({:.0} MB @ {:.1} GB/s)",
                total_pinned, num_moe_layers, avg_pinned, pin_ms,
                pool_bytes as f64 / 1e6,
                pool_bytes as f64 / 1e9 / (pin_ms / 1000.0));
        }

        self.pinning_pool_ptr = pool_base;
        self.pinning_pool_bytes = pool_bytes;
        self.pinned_expert_offsets = offsets;
        self.pinning_active = true;

        Ok(pin_counts.iter().copied().max().unwrap_or(0))
    }

    /// Selective DMA: copy only specific experts to fused buffer.
    /// Three-tier lookup: HCS D2D > pinning pool D2D > cold H2D from CPU.
    /// model_layer_idx is the full model layer index (for HCS cache_fast lookup).
    /// moe_layer_idx is the MoE-specific sequential index (for pinning pool lookup).
    fn selective_dma_layer(
        &self,
        moe_data: &PrefillMoeLayerData,
        model_layer_idx: usize,  // model layer index for HCS lookup
        moe_layer_idx: usize,    // index into pinned_expert_offsets
        w1_base: u64, w1s_base: u64, w2_base: u64, w2s_base: u64,
        active_experts: &[usize],  // list of expert IDs to transfer
    ) -> Result<(usize, usize, usize), String>  // (hcs_count, pinned_count, cold_count)
    {
        let mut hcs = 0usize;
        let mut pinned = 0usize;
        let mut cold = 0usize;
        let pool_base = self.pinning_pool_ptr;

        for &eid in active_experts {
            if eid >= moe_data.experts.len() { continue; }

            let w1_off = (eid * self.w1_packed_per_expert) as u64;
            let w1s_off = (eid * self.w1_scales_per_expert) as u64;
            let w2_off = (eid * self.w2_packed_per_expert) as u64;
            let w2s_off = (eid * self.w2_scales_per_expert) as u64;

            // Tier 1: Check prefill cache + HCS (GPU-resident experts)
            if let Some((hw1p, hw1s, hw2p, hw2s)) = self.expert_lookup(model_layer_idx, eid) {
                // Fast D2D from HCS cache to fused buffer
                unsafe {
                    cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                        w1_base + w1_off, hw1p,
                        self.w1_packed_per_expert, self.copy_stream);
                    cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                        w1s_base + w1s_off, hw1s,
                        self.w1_scales_per_expert, self.copy_stream);
                    cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                        w2_base + w2_off, hw2p,
                        self.w2_packed_per_expert, self.copy_stream);
                    cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                        w2s_base + w2s_off, hw2s,
                        self.w2_scales_per_expert, self.copy_stream);
                }
                hcs += 1;
                continue;
            }

            // Tier 2: Check pinning pool (pre-scan hot experts)
            let pin_offset = if self.pinning_active
                && moe_layer_idx < self.pinned_expert_offsets.len()
                && eid < self.pinned_expert_offsets[moe_layer_idx].len()
            {
                self.pinned_expert_offsets[moe_layer_idx][eid]
            } else {
                None
            };

            if let Some(pool_off) = pin_offset {
                // Fast D2D from pinning pool to fused buffer
                let src = pool_base + pool_off as u64;
                let mut src_off = 0u64;
                unsafe {
                    cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                        w1_base + w1_off, src + src_off,
                        self.w1_packed_per_expert, self.copy_stream);
                    src_off += self.w1_packed_per_expert as u64;
                    cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                        w1s_base + w1s_off, src + src_off,
                        self.w1_scales_per_expert, self.copy_stream);
                    src_off += self.w1_scales_per_expert as u64;
                    cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                        w2_base + w2_off, src + src_off,
                        self.w2_packed_per_expert, self.copy_stream);
                    src_off += self.w2_packed_per_expert as u64;
                    cuda_sys::lib().cuMemcpyDtoDAsync_v2(
                        w2s_base + w2s_off, src + src_off,
                        self.w2_scales_per_expert, self.copy_stream);
                }
                pinned += 1;
            } else {
                // Tier 3: Cold - H2D from CPU via copy_stream
                let e = &moe_data.experts[eid];
                unsafe {
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        w1_base + w1_off, e.w13_packed_ptr as *const _,
                        e.w13_packed_bytes, self.copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        w1s_base + w1s_off, e.w13_scales_ptr as *const _,
                        e.w13_scales_bytes, self.copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        w2_base + w2_off, e.w2_packed_ptr as *const _,
                        e.w2_packed_bytes, self.copy_stream);
                    cuda_sys::lib().cuMemcpyHtoDAsync_v2(
                        w2s_base + w2s_off, e.w2_scales_ptr as *const _,
                        e.w2_scales_bytes, self.copy_stream);
                }
                cold += 1;
            }
        }

        Ok((hcs, pinned, cold))
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
        let shared_ws_len = self.config.sms * MARLIN_MAX_LOCK_SLOTS_PER_SM;

        // BF16 validation mode: reuse the async BF16 method
        if let (Some(sw1_bf16), Some(sw2_bf16)) = (&lw.shared_w1_bf16, &lw.shared_w2_bf16) {
            return self.launch_shared_expert_async_bf16(layer_idx, m, sw1_bf16, sw2_bf16);
        }

        let sw1 = lw.shared_w1.as_ref().ok_or("missing shared w1")?;
        let sw2 = lw.shared_w2.as_ref().ok_or("missing shared w2")?;
        let shared_inter = sw1.n / (if gated { 2 } else { 1 });

        let st = if sw1.num_bits == 4 { &ScalarType::U4B8 } else { &ScalarType::U8B128 };
        let f = self.kernels.marlin_mm.ok_or("Marlin GEMM not loaded")?;

        // Use shared_stream + shared workspace to avoid conflicts
        let shared_scratch = *self.d_shared_fp32_scratch.as_ref()
            .ok_or("d_shared_fp32_scratch not allocated")?.device_ptr();
        let shared_ws = *self.d_shared_workspace.as_ref()
            .ok_or("d_shared_workspace not allocated")?.device_ptr();

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
    /// Uses synchronous cuMemcpyHtoD because the expert weight host memory is
    /// not pinned (comes from Python numpy/torch allocations). Async H2D requires
    /// page-locked host memory and fails with CUDA_ERROR_INVALID_VALUE otherwise.
    fn dma_expert_to_buf(
        &self, e: &ExpertWeightPtrs, buf: &(u64, u64, u64, u64),
    ) -> Result<(), String> {
        let (w13p_dst, w13s_dst, w2p_dst, w2s_dst) = *buf;
        unsafe {
            // Synchronous H2D: blocks CPU but works with any host memory.
            // The double-buffer pipeline still functions (DMA fills one buffer
            // while compute uses the other) but without CPU-side overlap.
            {
                let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                    w13p_dst, e.w13_packed_ptr as *const _, e.w13_packed_bytes);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("DMA w13_packed (src={:#x} dst={:#x} bytes={}): {:?}",
                        e.w13_packed_ptr, w13p_dst, e.w13_packed_bytes, err));
                }
                let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                    w13s_dst, e.w13_scales_ptr as *const _, e.w13_scales_bytes);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("DMA w13_scales (src={:#x} dst={:#x} bytes={}): {:?}",
                        e.w13_scales_ptr, w13s_dst, e.w13_scales_bytes, err));
                }
                let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                    w2p_dst, e.w2_packed_ptr as *const _, e.w2_packed_bytes);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("DMA w2_packed (src={:#x} dst={:#x} bytes={}): {:?}",
                        e.w2_packed_ptr, w2p_dst, e.w2_packed_bytes, err));
                }
                let err = cuda_sys::lib().cuMemcpyHtoD_v2(
                    w2s_dst, e.w2_scales_ptr as *const _, e.w2_scales_bytes);
                if err != cuda_sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("DMA w2_scales (src={:#x} dst={:#x} bytes={}): {:?}",
                        e.w2_scales_ptr, w2s_dst, e.w2_scales_bytes, err));
                }
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
        if bits == 16 {
            // BF16 validation mode: use cuBLAS GEMM instead of Marlin
            return self.run_expert_gemm_bf16(
                gathered_slice, expert_out, gate_up_buf, inter_buf,
                w13_packed, w2_packed,
                count, offset,
                w13_n, h, inter,
                gated, activation,
            );
        }

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

    /// BF16 expert GEMM via cuBLAS (validation mode).
    /// w13_data and w2_data are raw BF16 weights on GPU (row-major [rows, cols]).
    fn run_expert_gemm_bf16(
        &self,
        gathered_slice: u64, expert_out: u64, gate_up_buf: u64, inter_buf: u64,
        w13_data: u64, w2_data: u64,
        count: usize, offset: usize,
        w13_n: usize, h: usize, inter: usize,
        gated: bool, activation: u8,
    ) -> Result<(), String> {
        // Diagnostic: check first expert's weight data on GPU (layer 0 only)
        static BF16_DIAG_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !BF16_DIAG_DONE.load(std::sync::atomic::Ordering::Relaxed) {
            BF16_DIAG_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
            self.stream_sync()?;
            // Read first 64 BF16 values from w13 weight on GPU
            let n_vals = 64.min(w13_n * h);
            let mut h_w13 = vec![0u16; n_vals];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_w13.as_mut_ptr() as *mut _, w13_data, n_vals * 2,
                );
            }
            let vals: Vec<f32> = h_w13.iter().map(|&v| half::bf16::from_bits(v).to_f32()).collect();
            let norm: f32 = vals.iter().map(|v| v*v).sum::<f32>().sqrt();
            let non_zero = vals.iter().filter(|&&v| v != 0.0).count();
            eprintln!("[BF16-DIAG] First expert w13 GPU data: norm={:.6}, non_zero={}/{}, first8={:.4?}",
                norm, non_zero, n_vals, &vals[..8.min(vals.len())]);
            // Also check input data (gathered_slice)
            let n_in = 8.min(h);
            let mut h_in = vec![0u16; n_in];
            unsafe {
                cuda_sys::lib().cuMemcpyDtoH_v2(
                    h_in.as_mut_ptr() as *mut _, gathered_slice, n_in * 2,
                );
            }
            let in_vals: Vec<f32> = h_in.iter().map(|&v| half::bf16::from_bits(v).to_f32()).collect();
            eprintln!("[BF16-DIAG] First expert gathered input first8={:.4?}", in_vals);
        }

        // w1 GEMM: [count, h] @ [w13_n, h]^T -> [count, w13_n]
        // BF16 weight is row-major [w13_n, h], need to transpose
        let w13 = Bf16Weight { ptr: w13_data, n: w13_n, k: h };
        self.cublas_bf16_gemm(gathered_slice, &w13, gate_up_buf + (offset * w13_n * 2) as u64, count)?;

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

            // w2 GEMM: [count, inter] @ [h, inter]^T -> [count, h]
            let w2 = Bf16Weight { ptr: w2_data, n: h, k: inter };
            self.cublas_bf16_gemm(inter_slice, &w2, expert_out + (offset * h * 2) as u64, count)?;
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
            let w2 = Bf16Weight { ptr: w2_data, n: h, k: inter };
            self.cublas_bf16_gemm(w1_out, &w2, expert_out + (offset * h * 2) as u64, count)?;
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
                "moe_finalize_sorted_kernel",
                "moe_gather_sorted_kernel",
                "moe_replicate_hidden_kernel",
                "moe_scatter_fused_kernel",
                "moe_scatter_weighted_kernel",
                "moe_accum_to_bf16_kernel",
                "kv_cache_append_fp8_kernel",
                "kv_cache_dequant_concat_kernel",
                "kv_cache_append_polar4_kernel",
                "kv_cache_dequant_concat_polar4_kernel",
                // Optimized LA kernels (BF16 pipeline)
                "la_fused_conv1d_silu_bf16_kernel",
                "la_update_conv_state_kernel",
                "la_compute_gate_beta_bf16_kernel",
                "la_fused_repeat_l2norm_bf16_kernel",
                "la_gated_rmsnorm_bf16in_kernel",
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
            kv_dequant_concat: get("kv_cache_dequant_concat_kernel")?,
            kv_cache_append_polar4: get("kv_cache_append_polar4_kernel")?,
            kv_dequant_concat_polar4: get("kv_cache_dequant_concat_polar4_kernel")?,
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
            la_fused_conv1d_silu_bf16: get("la_fused_conv1d_silu_bf16_kernel")?,
            la_update_conv_state: get("la_update_conv_state_kernel")?,
            la_compute_gate_beta_bf16: get("la_compute_gate_beta_bf16_kernel")?,
            la_fused_repeat_l2norm_bf16: get("la_fused_repeat_l2norm_bf16_kernel")?,
            la_gated_rmsnorm_bf16in: get("la_gated_rmsnorm_bf16in_kernel")?,
            la_compute_v_new_strided: get("la_compute_v_new_strided_kernel")?,
            la_chunk_output_strided: get("la_chunk_output_strided_kernel")?,
            la_state_update_strided: get("la_state_update_strided_kernel")?,
            moe_count_experts: get("moe_count_experts_kernel")?,
            moe_prefix_sum: get("moe_prefix_sum_kernel")?,
            moe_build_maps: get("moe_build_maps_kernel")?,
            moe_padded_prefix_sum: get("moe_padded_prefix_sum_kernel")?,
            moe_scatter_sorted: get("moe_scatter_sorted_kernel")?,
            moe_finalize_sorted: get("moe_finalize_sorted_kernel")?,
            moe_gather_sorted: get("moe_gather_sorted_kernel")?,
            moe_replicate_hidden: get("moe_replicate_hidden_kernel")?,
            moe_scatter_fused: get("moe_scatter_fused_kernel")?,
            moe_scatter_weighted: get("moe_scatter_weighted_kernel")?,
            marlin_mm,
            fused_moe_fn: load_fused_moe(),
            flash_attn_fwd: load_flash_attn(),
            flash_attn_fwd_fp8kv: load_flash_attn_fp8kv(),
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

/// Safety margin (MB) reserved during scratch allocation to cover CUDA context overhead,
/// allocator fragmentation, cuBLAS workspace, and FLA Triton kernel temporaries.
/// Used by both `prepare_for_prefill` and `hcs_evict_for_prefill` to ensure consistency.
pub const PREFILL_SAFETY_MARGIN_MB: usize = 600;

/// Compute total VRAM bytes needed for prefill buffers as a function of max_tokens.
/// Returns (fixed_bytes, per_token_bytes) including both scratch AND fused MoE buffers.
/// This is used to dynamically compute the largest chunk size that fits in available VRAM.
pub fn compute_scratch_vram(config: &PrefillModelConfig) -> (usize, usize) {
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let moe_inter = config.moe_intermediate_size;
    let topk = config.num_experts_per_tok.max(1);
    let has_la = config.layer_types.iter().any(|&t| t == 3);
    let has_mamba2 = config.layer_types.iter().any(|&t| t == 1);

    // Per-token costs (in bytes):
    let mut per_token: usize = 0;

    // d_hidden: max_tokens * h * 2 (BF16)
    per_token += h * 2;
    // d_residual: max_tokens * h * 2
    per_token += h * 2;
    // d_scratch1: max(max_tokens * inter * 2, max_tokens * num_q_heads * head_dim * 2) * 2 bytes
    let max_inter_per_tok = std::cmp::max(
        inter * 2,
        config.num_q_heads * config.head_dim * 2,
    );
    per_token += max_inter_per_tok * 2; // BF16
    // d_scratch2: same size
    per_token += max_inter_per_tok * 2;
    // d_fp32_scratch: max(marlin_ctmp, la_size) -- both scale with max_tokens
    {
        let mut max_gemm_n = std::cmp::max(h, inter * 2);
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
        let marlin_per_tok = max_gemm_n; // FP32 = 4 bytes each
        let la_per_tok = if has_la {
            let nv = config.la_num_v_heads;
            let dv = config.la_v_head_dim;
            let conv_dim = config.la_conv_dim;
            // output_buf = nv * max_tokens * dv, conv_transpose = max_tokens * conv_dim
            // (chunk_temps are fixed-size based on la_chunk_size, not max_tokens)
            std::cmp::max(nv * dv, conv_dim)
        } else { 0 };
        per_token += std::cmp::max(marlin_per_tok, la_per_tok) * 4; // FP32
    }
    // d_workspace: fixed (sms * MARLIN_MAX_LOCK_SLOTS_PER_SM * 4 bytes) -- ignore per-token
    // d_topk_weights: max_tokens * topk * 4
    per_token += topk * 4;
    // d_topk_ids: max_tokens * topk * 4
    per_token += topk * 4;
    // d_token_ids: max_tokens * 4
    per_token += 4;
    // d_positions: max_tokens * 4
    per_token += 4;
    // d_attn_out: max_tokens * attn_dim * 2, where attn_dim = max(h, q_dim, value_dim)
    // GQA writes [M, num_q_heads * head_dim], LA gated_rmsnorm no longer uses this buffer.
    let attn_dim = h
        .max(config.num_q_heads * config.head_dim)
        .max(config.la_num_v_heads * config.la_v_head_dim);
    per_token += attn_dim * 2;
    // d_q: max_tokens * num_q_heads * head_dim * 2
    per_token += config.num_q_heads * config.head_dim * 2;
    // d_k: max_tokens * num_kv_heads * head_dim * 2
    per_token += config.num_kv_heads * config.head_dim * 2;
    // d_v: max_tokens * num_kv_heads * head_dim * 2
    per_token += config.num_kv_heads * config.head_dim * 2;
    // MoE buffers: sized for fused_sorted_count = max_tokens * topk + n_routed * block_size.
    // Both sequential and fused paths share these buffers (never run simultaneously).
    let w1_n = if config.moe_gated { 2 * moe_inter } else { moe_inter };
    let block_size_moe: usize = 64; // MarlinDefault block_size_m
    // d_gate_out: max_tokens * n_routed_experts * 4
    per_token += config.n_routed_experts.max(1) * 4;
    // d_moe_accum: fused_sorted_count * max(w1_n, h) * 4 (serves as Marlin C_tmp for fused,
    // or [m, h] FP32 scatter accumulator for sequential -- fused size dominates)
    per_token += topk * std::cmp::max(w1_n, h) * 4;
    // d_moe_gathered: fused_sorted_count * h * 2 (fused_input or gathered)
    per_token += topk * h * 2;
    // d_moe_expert_out: fused_sorted_count * h * 2 (fused_output or expert_out)
    per_token += topk * h * 2;
    // d_moe_gate_up: fused_sorted_count * w1_n * 2 (fused_inter_cache or gate_up)
    per_token += topk * w1_n * 2;
    // d_moe_inter: fused_sorted_count * moe_inter * 2 (fused_inter2 or inter)
    per_token += topk * moe_inter * 2;
    // d_gather_src_map: fused_sorted_count * 4 (sorted_token_ids or gather_src_map)
    per_token += topk * 4;
    // d_gather_weight_map: max_tokens * topk * 4 (sequential only, small)
    per_token += topk * 4;
    // d_shared_fp32_scratch: max_tokens * max(h, moe_inter) * 4 (FP32)
    // Shared expert Marlin workspace, allocated dynamically in prepare_for_prefill.
    if config.n_routed_experts > 0 {
        per_token += std::cmp::max(h, moe_inter) * 4;
    }
    // LA buffers (scale with la_max_len = max_tokens + la_chunk_size, approx max_tokens)
    if has_la {
        let nv = config.la_num_v_heads;
        let dk = config.la_k_head_dim;
        let dv = config.la_v_head_dim;
        // d_la_q, d_la_k: max_tokens * nv * dk * 4 (FP32) each
        per_token += nv * dk * 4 * 2;
        // d_la_v: max_tokens * nv * dv * 4
        per_token += nv * dv * 4;
        // d_la_z: max_tokens * nv * dv * 2 (BF16)
        per_token += nv * dv * 2;
        // d_la_b, d_la_a: max_tokens * nv * 2 each
        per_token += nv * 2 * 2;
        // d_la_beta, d_la_gate: max_tokens * nv * 4 each
        per_token += nv * 4 * 2;
        // d_la_conv_out: conv_dim * max_tokens * 4
        per_token += config.la_conv_dim * 4;
        // d_la_v_beta, d_la_v_new: nv * max_tokens * dv * 4 each
        per_token += nv * dv * 4 * 2;
        // d_la_g_cum: nv * max_tokens * 4 (approx -- actual is nv * num_chunks * chunk_size)
        per_token += nv * 4;
        // d_la_proj_buf: max_tokens * qkvz_dim * 2 (BF16)
        let hr = config.la_head_ratio.max(1);
        let group_dim_proj = 2 * dk + 2 * hr * dv;
        per_token += config.la_num_k_heads * group_dim_proj * 2;
        // d_la_chunk_out, d_la_q_contig: nv * chunk_size * dk * 4 -- fixed, not per-token
        // Dedicated FLA buffers allocated dynamically in prepare_for_prefill.
        // Most scale linearly with padded T; h scales as T/64 chunks.
        per_token += nv * 4;              // d_fla_g_cumsum
        per_token += nv * 64 * 4;         // d_fla_a
        per_token += nv * 64 * 2;         // d_fla_ai
        per_token += nv * dk * 2;         // d_fla_w
        per_token += nv * dv * 2 * 3;     // d_fla_u + d_fla_v_new + d_fla_o
        per_token += nv * dk * dv * 2 / 64; // d_fla_h averaged over 64-token chunks
    }
    // d_fa2_lse: num_q_heads * max_tokens * 4 (FP32)
    if config.num_q_heads > 0 {
        per_token += config.num_q_heads * 4;
    }
    // Mamba2 buffers: mostly fixed-size, ignore for per-token estimate

    // Fixed costs (independent of max_tokens):
    let mut fixed: usize = 0;
    // d_workspace: sms * MARLIN_MAX_LOCK_SLOTS_PER_SM * 4
    fixed += config.sms * MARLIN_MAX_LOCK_SLOTS_PER_SM * 4;
    // d_shared_workspace: sms * MARLIN_MAX_LOCK_SLOTS_PER_SM * 4 (shared expert stream, allocated in prepare_for_prefill)
    if config.n_routed_experts > 0 {
        fixed += config.sms * MARLIN_MAX_LOCK_SLOTS_PER_SM * 4;
    }
    // d_logits: vocab_size * 4
    fixed += config.vocab_size * 4;
    if has_la {
        fixed += config.la_num_v_heads * config.la_k_head_dim * config.la_v_head_dim * 4; // d_fla_final_state
    }
    // d_expert_counts, offsets, write_offsets: n_routed * 4 each
    fixed += config.n_routed_experts.max(1) * 4 * 3;
    // Expert DMA double-buffers (A+B for w13, w13s, w2, w2s -- 8 buffers total, each one expert)
    if config.n_routed_experts > 0 {
        let n_routed = config.n_routed_experts;
        let gs = config.group_size.max(1);
        let bits = config.expert_bits as usize;
        let w13_n = if config.moe_gated { 2 * moe_inter } else { moe_inter };
        let max_moe_n = std::cmp::max(w13_n, h);
        let expert_bytes = h * w13_n * bits / 8  // w13 packed
            + (h / gs) * w13_n * 2               // w13 scales
            + moe_inter * h * bits / 8           // w2 packed
            + (moe_inter / gs) * h * 2;          // w2 scales
        fixed += expert_bytes * 2; // A + B staging buffers (one expert each)

        // Fused MoE: block padding on shared sorted buffers.
        // fused_sorted_count = max_tokens * topk + n_routed * block_size.
        // The per-token part is accounted for above. The n_routed * block_size padding is fixed.
        let per_sorted_entry = std::cmp::max(w1_n, h) * 4  // d_moe_accum (FP32)
            + h * 2 + h * 2 + w1_n * 2 + moe_inter * 2    // gathered + expert_out + gate_up + inter (BF16)
            + 4;                                             // d_gather_src_map (i32)
        fixed += n_routed * block_size_moe * per_sorted_entry;

        let reduce_floor = fused_moe_fp32_reduce_floats(
            block_size_moe,
            max_moe_n,
            h,
            bits,
            gs,
            config.sms,
            usize::MAX,
        );
        let base_fixed_moe_accum = n_routed * block_size_moe * max_moe_n;
        if reduce_floor > base_fixed_moe_accum {
            fixed += (reduce_floor - base_fixed_moe_accum) * 4;
        }

        // Fused expert_ids buffer: fused_blocks * 4 (approx n_routed + padding)
        // and num_tokens_post: 4 bytes. Small, just add a rough estimate.
        fixed += (n_routed + 1024) * 4 + 4;

        // Cold staging buffer: 1 full set of all expert weights (worst case: no HCS hits).
        // In pointer-table mode, cold staging is the primary expert DMA target.
        // Fused expert weight buffers (A/B sets) are allocated opportunistically AFTER
        // cold staging only if spare VRAM exists — they are NOT budgeted here.
        fixed += n_routed * expert_bytes;

        // Pointer table buffers: 4 arrays of n_routed * u64
        fixed += 4 * n_routed * 8;
    }
    if has_la {
        let nv = config.la_num_v_heads;
        let dk = config.la_k_head_dim;
        let dv = config.la_v_head_dim;
        let cs = config.la_chunk_size;
        // LA chunk-sized buffers (fixed)
        fixed += nv * cs * dk * 4; // d_la_chunk_out
        fixed += nv * cs * dk * 4; // d_la_q_contig
    }
    if has_mamba2 {
        fixed += config.mamba_conv_dim * config.mamba_conv_kernel.max(2) * 4;
        fixed += config.mamba_num_heads * config.mamba_head_dim * config.mamba_d_state * 4;
        let dim = 2 * config.mamba_d_inner
            + 2 * config.mamba_n_groups * config.mamba_d_state
            + config.mamba_num_heads;
        fixed += dim * 2; // per token but just 1 token for mamba init
        fixed += config.mamba_d_inner * 2;
    }

    (fixed, per_token)
}

pub fn allocate_scratch(
    _device: &Arc<CudaDevice>,
    config: &PrefillModelConfig,
    max_tokens: usize,
) -> Result<PrefillScratch, String> {
    let h = config.hidden_size;
    let inter = config.intermediate_size;           // max of dense + moe for general scratch
    let moe_inter = config.moe_intermediate_size;   // actual MoE expert intermediate
    let topk = config.num_experts_per_tok.max(1);

    // Fused sorted count: max_tokens * topk + n_routed * block_size (block padding for Marlin).
    // When max_tokens=0 (init/release), use fsc=1 to avoid allocating ~608 MB of block-padding
    // buffers that sit idle during decode. These buffers eat HCS space and cause double-counting
    // in the VRAM budget. compute_scratch_vram correctly reports the full cost including padding
    // for budget purposes; we just avoid physically allocating it when nothing is running.
    let block_size_moe: usize = 64;
    let n_routed = config.n_routed_experts;
    let fsc = if max_tokens > 0 {
        max_tokens * topk + n_routed * block_size_moe
    } else {
        1 // minimal placeholder — no MoE execution at 0 tokens
    };
    // GpuBuf uses raw cuMemAlloc_v2 (synchronous, no pool) so alloc/free
    // is immediate and deterministic — no interaction with cudarc's pool.
    let alloc_u16 = |n: usize, name: &str| -> Result<GpuBuf<u16>, String> {
        GpuBuf::<u16>::alloc_zeroed(n).map_err(|e| format!("alloc {name}: {e}"))
    };
    let alloc_f32 = |n: usize, name: &str| -> Result<GpuBuf<f32>, String> {
        GpuBuf::<f32>::alloc_zeroed(n).map_err(|e| format!("alloc {name}: {e}"))
    };
    let alloc_i32 = |n: usize, name: &str| -> Result<GpuBuf<i32>, String> {
        GpuBuf::<i32>::alloc_zeroed(n).map_err(|e| format!("alloc {name}: {e}"))
    };
    let alloc_u8 = |n: usize, name: &str| -> Result<GpuBuf<u8>, String> {
        GpuBuf::<u8>::alloc_zeroed(n).map_err(|e| format!("alloc {name}: {e}"))
    };

    let has_mamba2 = config.layer_types.iter().any(|&t| t == 1);
    let has_la = config.layer_types.iter().any(|&t| t == 3);
    // LA pads to chunk_size multiples, so total_len can be up to max_tokens + chunk_size - 1
    let la_max_len = if has_la { max_tokens + config.la_chunk_size } else { max_tokens };

    let mut max_inter = std::cmp::max(
        max_tokens * inter * 2,
        max_tokens * config.num_q_heads * config.head_dim * 2, // *2 for gated GQA (q + gate)
    );
    // FLA path reuses scratch1/scratch2 as [fla_total, nv, dk] BF16 (after repeat-interleave).
    // For models where nv * dk > num_q_heads * head_dim * 2 (e.g. 122B: 8192 vs 6144),
    // the FLA buffers are larger than the GQA scratch.  Account for FLA padding too.
    if has_la {
        let fla_req = la_max_len * config.la_num_v_heads * config.la_k_head_dim;
        if fla_req > max_inter {
            max_inter = fla_req;
        }
    }

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
        d_workspace: alloc_i32(config.sms * MARLIN_MAX_LOCK_SLOTS_PER_SM, "workspace")?,
        d_topk_weights: alloc_f32(max_tokens * topk, "topk_weights")?,
        d_topk_ids: alloc_i32(max_tokens * topk, "topk_ids")?,
        d_token_ids: alloc_i32(max_tokens, "token_ids")?,
        d_positions: alloc_i32(max_tokens, "positions")?,
        d_attn_out: {
            // Must fit max(hidden_size, num_q_heads*head_dim, value_dim) per token
            let attn_dim = h
                .max(config.num_q_heads * config.head_dim)
                .max(config.la_num_v_heads * config.la_v_head_dim);
            alloc_u16(max_tokens * attn_dim, "attn_out")?
        },
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
        // MoE buffers: sized for fsc (fused_sorted_count) computed above.
        // When max_tokens=0 (init/release), fsc=1 to avoid wasting ~608 MB on block padding.
        d_gate_out: alloc_f32(max_tokens * config.n_routed_experts.max(1), "gate_out")?,
        d_moe_accum: {
            let w1_n = if config.moe_gated { 2 * moe_inter } else { moe_inter };
            let max_moe_n = std::cmp::max(w1_n, h);
            let reduce_floor = config
                .fused_moe_w1_ctmp_floats
                .max(config.fused_moe_w2_ctmp_floats);
            alloc_f32((fsc * max_moe_n).max(reduce_floor), "moe_accum")?
        },
        d_moe_gathered: alloc_u16(fsc * h, "moe_gathered")?,
        d_moe_expert_out: alloc_u16(fsc * h, "moe_expert_out")?,
        d_moe_gate_up: {
            let w1_n = if config.moe_gated { 2 * moe_inter } else { moe_inter };
            alloc_u16(fsc * w1_n, "moe_gate_up")?
        },
        d_moe_inter: alloc_u16(fsc * moe_inter, "moe_inter")?,
        d_gather_src_map: alloc_i32(fsc, "gather_src_map")?,
        d_gather_weight_map: alloc_f32(max_tokens * topk, "gather_weight_map")?,
        d_fused_expert_ids: {
            let fb = fsc / block_size_moe + n_routed;
            alloc_i32(fb.max(1), "fused_expert_ids")?
        },
        d_num_tokens_post: alloc_i32(1, "num_tokens_post")?,
        fused_sorted_count: fsc,
        fused_blocks: fsc / block_size_moe + n_routed,
        // GPU-only routing scratch
        d_expert_counts: alloc_i32(config.n_routed_experts.max(1), "expert_counts")?,
        d_expert_offsets: alloc_i32(config.n_routed_experts.max(1) + 1, "expert_offsets")?,
        d_write_offsets: alloc_i32(config.n_routed_experts.max(1), "write_offsets")?,
        // Expert DMA double-buffers (A and B, each sized for one expert)
        d_expert_w13_packed_a: {
            let w13_size = if config.n_routed_experts > 0 {
                let w13_n = if config.moe_gated { 2 * moe_inter } else { moe_inter };
                h * w13_n * config.expert_bits as usize / 8
            } else { 1 };
            alloc_u8(w13_size, "expert_w13_packed_a")?
        },
        d_expert_w13_scales_a: {
            let scale_size = if config.n_routed_experts > 0 && config.group_size > 0 {
                let w13_n = if config.moe_gated { 2 * moe_inter } else { moe_inter };
                (h / config.group_size) * w13_n * 2
            } else { 1 };
            alloc_u8(scale_size, "expert_w13_scales_a")?
        },
        d_expert_w2_packed_a: {
            let w2_size = if config.n_routed_experts > 0 {
                moe_inter * h * config.expert_bits as usize / 8
            } else { 1 };
            alloc_u8(w2_size, "expert_w2_packed_a")?
        },
        d_expert_w2_scales_a: {
            let scale_size = if config.n_routed_experts > 0 && config.group_size > 0 {
                (moe_inter / config.group_size) * h * 2
            } else { 1 };
            alloc_u8(scale_size, "expert_w2_scales_a")?
        },
        d_expert_w13_packed_b: {
            let w13_size = if config.n_routed_experts > 0 {
                let w13_n = if config.moe_gated { 2 * moe_inter } else { moe_inter };
                h * w13_n * config.expert_bits as usize / 8
            } else { 1 };
            alloc_u8(w13_size, "expert_w13_packed_b")?
        },
        d_expert_w13_scales_b: {
            let scale_size = if config.n_routed_experts > 0 && config.group_size > 0 {
                let w13_n = if config.moe_gated { 2 * moe_inter } else { moe_inter };
                (h / config.group_size) * w13_n * 2
            } else { 1 };
            alloc_u8(scale_size, "expert_w13_scales_b")?
        },
        d_expert_w2_packed_b: {
            let w2_size = if config.n_routed_experts > 0 {
                moe_inter * h * config.expert_bits as usize / 8
            } else { 1 };
            alloc_u8(w2_size, "expert_w2_packed_b")?
        },
        d_expert_w2_scales_b: {
            let scale_size = if config.n_routed_experts > 0 && config.group_size > 0 {
                (moe_inter / config.group_size) * h * 2
            } else { 1 };
            alloc_u8(scale_size, "expert_w2_scales_b")?
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
        d_fa2_lse: if config.num_q_heads > 0 {
            // FA2 softmax_lse: [num_heads, max_tokens] in unpadded format
            Some(alloc_f32(config.num_q_heads * max_tokens, "fa2_lse")?)
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
    let path = match find_marlin_so() {
        Some(p) => p,
        None => {
            log::debug!("Fused MoE unavailable: libkrasis_marlin.so not found");
            return None;
        }
    };
    unsafe {
        let lib = libc::dlopen(
            std::ffi::CString::new(path.as_str()).ok()?.as_ptr(),
            libc::RTLD_NOW | libc::RTLD_LOCAL,
        );
        if lib.is_null() {
            log::warn!("Fused MoE dlopen({}) failed", path);
            return None;
        }

        let sym = libc::dlsym(lib, b"krasis_marlin_moe_mm_bf16\0".as_ptr() as *const _);
        if !sym.is_null() {
            log::info!("Loaded fused MoE from {}", path);
            return Some(std::mem::transmute(sym));
        }
        log::warn!("Fused MoE symbol not found in {}", path);
    }
    None
}

/// Load vendored FlashAttention-2 from libkrasis_flash_attn.so.
fn load_flash_attn() -> Option<FlashAttnFwdFn> {
    let path = match find_vendor_so("libkrasis_flash_attn.so") {
        Some(p) => p,
        None => {
            log::warn!("libkrasis_flash_attn.so not found — FlashAttention disabled, using custom kernels");
            return None;
        }
    };
    unsafe {
        let lib = libc::dlopen(
            std::ffi::CString::new(path.as_str()).ok()?.as_ptr(),
            libc::RTLD_NOW | libc::RTLD_LOCAL,
        );
        if lib.is_null() {
            let err = libc::dlerror();
            if !err.is_null() {
                log::warn!("dlopen({}) failed: {}", path,
                    std::ffi::CStr::from_ptr(err).to_string_lossy());
            } else {
                log::warn!("dlopen({}) failed", path);
            }
            return None;
        }

        let sym = libc::dlsym(lib, b"krasis_flash_attn_fwd_bf16\0".as_ptr() as *const _);
        if !sym.is_null() {
            log::info!("Loaded FlashAttention-2 from {}", path);
            return Some(std::mem::transmute(sym));
        }
        log::warn!("krasis_flash_attn_fwd_bf16 symbol not found in {}", path);
    }
    None
}

/// Load the FP8 KV variant of FlashAttention-2 (BF16 Q, FP8 E4M3 K/V).
/// Same .so as regular FA2, different symbol.
fn load_flash_attn_fp8kv() -> Option<FlashAttnFwdFn> {
    let path = find_vendor_so("libkrasis_flash_attn.so")?;
    unsafe {
        let lib = libc::dlopen(
            std::ffi::CString::new(path.as_str()).ok()?.as_ptr(),
            libc::RTLD_NOW | libc::RTLD_LOCAL,
        );
        if lib.is_null() { return None; }

        let sym = libc::dlsym(lib, b"krasis_flash_attn_fwd_bf16q_fp8kv\0".as_ptr() as *const _);
        if !sym.is_null() {
            log::info!("Loaded FlashAttention-2 FP8 KV from {}", path);
            return Some(std::mem::transmute(sym));
        }
        log::warn!("krasis_flash_attn_fwd_bf16q_fp8kv symbol not found in {}", path);
    }
    None
}

/// Find a vendored .so file built by build.rs.
/// Searches: 1) env var, 2) next to executable, 3) Cargo build output.
fn find_vendor_so(name: &str) -> Option<String> {
    // 1. Look in env var (uppercase, dots/hyphens → underscore)
    let env_name = name.replace('.', "_").replace('-', "_").to_uppercase();
    let env_key = format!("KRASIS_{}", env_name);
    if let Ok(p) = std::env::var(&env_key) {
        if std::path::Path::new(&p).exists() { return Some(p); }
    }

    if let Some(p) = installed_package_sidecar(name) {
        return Some(p);
    }

    // 2. Look next to the current executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join(name);
            if p.exists() { return Some(p.to_string_lossy().to_string()); }
        }
    }

    // 3. Search Cargo build output directories
    let home = std::env::var("HOME").unwrap_or_default();
    for repo in &["krasis", "krasisx"] {
        for profile in &["release", "debug"] {
            let build_dir = format!("{}/Documents/Claude/{}/target/{}/build", home, repo, profile);
            if let Ok(entries) = std::fs::read_dir(&build_dir) {
                for e in entries.flatten() {
                    let p = e.path().join(format!("out/{}", name));
                    if p.exists() { return Some(p.to_string_lossy().to_string()); }
                }
            }
        }
    }

    None
}

/// Load vendored FLA (Flash Linear Attention) kernels from libkrasis_fla.so.
/// For LA models this is required unless the user explicitly opted out with
/// KRASIS_NO_FLA, because silently falling back to the older custom LA path
/// hides packaging/runtime regressions and degrades performance.
pub fn load_fla(require_for_la_model: bool, nv: usize) -> Result<Option<FlaKernels>, String> {
    let fail_or_warn = |message: String| -> Result<Option<FlaKernels>, String> {
        if require_for_la_model {
            Err(format!(
                "{message}. Refusing to fall back to custom LA kernels for a linear-attention model. \
Set KRASIS_NO_FLA=1 only if you explicitly want the slower custom LA path."
            ))
        } else {
            log::warn!("{message} — FLA disabled");
            Ok(None)
        }
    };

    if std::env::var("KRASIS_NO_FLA").is_ok() {
        log::info!("KRASIS_NO_FLA set — FLA disabled, using custom LA kernels");
        return Ok(None);
    }

    // Detect GPU compute capability and load the best-matching FLA .so.
    // We ship pre-compiled cubins for multiple GPU architectures (sm_80, sm_89,
    // sm_90, sm_120).  Selection priority:
    //   1. Exact arch match (e.g. sm_120 on a 5090)
    //   2. Same-generation base arch (e.g. sm_80 on an sm_86 RTX 3090)
    //   3. Highest available .so if the GPU is newer than anything we compiled
    //      for (forward compat — cubins from the latest gen we ship are the
    //      best chance of working on a future GPU)
    // Pre-Ampere (sm_75 and below) is not supported — Marlin INT4 kernels
    // require Ampere or newer.
    let path = {
        // Query GPU compute capability via CUDA driver API
        let mut cc_major: i32 = 0;
        let mut cc_minor: i32 = 0;
        unsafe {
            cuda_sys::lib().cuDeviceGetAttribute(
                &mut cc_major,
                cuda_sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                0,
            );
            cuda_sys::lib().cuDeviceGetAttribute(
                &mut cc_minor,
                cuda_sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                0,
            );
        }
        let arch = cc_major * 10 + cc_minor;

        if arch > 0 && arch < 80 {
            return fail_or_warn(format!(
                "GPU sm_{} (pre-Ampere) is not supported. \
                 Krasis requires an Ampere (sm_80) or newer GPU.",
                arch
            ));
        }

        let mut found: Option<String> = None;

        if arch > 0 {
            // 1. Exact arch match (e.g. libkrasis_fla_sm120.so)
            let arch_name = format!("libkrasis_fla_sm{}.so", arch);
            if let Some(p) = find_vendor_so(&arch_name) {
                log::info!("Found arch-specific FLA library: {} (GPU sm_{})", arch_name, arch);
                found = Some(p);
            }

            // 2. Same-generation base arch (sm_X0 cubins run on sm_XY via
            //    CUDA binary compatibility within the same major version)
            if found.is_none() {
                let base_arch = cc_major * 10;
                if base_arch != arch {
                    let fallback_name = format!("libkrasis_fla_sm{}.so", base_arch);
                    if let Some(p) = find_vendor_so(&fallback_name) {
                        log::info!(
                            "GPU sm_{} — using compatible FLA library: {} (base arch sm_{})",
                            arch, fallback_name, base_arch
                        );
                        found = Some(p);
                    }
                }
            }

            // 3. Forward compatibility: if the GPU is newer than anything we
            //    compiled for, try the highest available arch .so.  This is
            //    best-effort — it may work (especially within the same CUDA
            //    generation) or fail at kernel launch with a clear error.
            if found.is_none() {
                let mut best: Option<(u32, String)> = None;
                for candidate_arch in [120, 90, 89, 80] {
                    let name = format!("libkrasis_fla_sm{}.so", candidate_arch);
                    if let Some(p) = find_vendor_so(&name) {
                        match best {
                            None => best = Some((candidate_arch, p)),
                            Some((prev, _)) if candidate_arch > prev => {
                                best = Some((candidate_arch, p));
                            }
                            _ => {}
                        }
                        break; // list is descending, first hit is highest
                    }
                }
                if let Some((best_arch, p)) = best {
                    log::warn!(
                        "GPU sm_{} is newer than any pre-compiled FLA library. \
                         Using sm_{} as best-effort forward compatibility. \
                         If FLA kernels fail, rebuild with ./dev build on this GPU.",
                        arch, best_arch
                    );
                    found = Some(p);
                }
            }
        }

        match found {
            Some(p) => p,
            None => {
                let msg = if arch > 0 {
                    format!(
                        "No FLA library found for GPU sm_{}. \
                         Rebuild with ./dev build or reinstall from a release wheel.",
                        arch
                    )
                } else {
                    "FLA library not found (could not detect GPU compute capability)".to_string()
                };
                return fail_or_warn(msg);
            }
        }
    };
    let path_cstr = match std::ffi::CString::new(path.as_str()) {
        Ok(v) => v,
        Err(_) => return fail_or_warn(format!("invalid FLA library path: {}", path)),
    };
    unsafe {
        let lib = libc::dlopen(
            path_cstr.as_ptr(),
            libc::RTLD_NOW | libc::RTLD_LOCAL,
        );
        if lib.is_null() {
            let err = libc::dlerror();
            if !err.is_null() {
                let detail = std::ffi::CStr::from_ptr(err).to_string_lossy();
                return fail_or_warn(format!("dlopen({}) failed: {}", path, detail));
            } else {
                return fail_or_warn(format!("dlopen({}) failed", path));
            }
        }

        // Initialize FLA modules (loads cubins into CUDA context)
        let init_sym = libc::dlsym(lib, b"krasis_fla_init\0".as_ptr() as *const _);
        if init_sym.is_null() {
            let _ = libc::dlclose(lib);
            return fail_or_warn(format!("krasis_fla_init not found in {}", path));
        }
        let init_fn: FlaInitFn = std::mem::transmute(init_sym);
        let rc = init_fn();
        if rc != 0 {
            let _ = libc::dlclose(lib);
            return fail_or_warn(format!("krasis_fla_init() failed with rc={}", rc));
        }

        // Load all 6 kernel function pointers.
        // FLA kernels have H (num_v_heads) baked in as a Triton constexpr.
        // Try H-specific symbols first (e.g. _h64), fall back to generic.
        let h_suffix = format!("_h{}", nv);
        eprintln!("[FLA] Loading kernels for H={} from {}", nv, path);
        let load_sym_for = |base_name: &str| -> Result<*mut std::ffi::c_void, String> {
            // Try H-specific first (using CString for safe null termination)
            let h_name_str = format!("{}{}", base_name, h_suffix);
            let h_cstr = std::ffi::CString::new(h_name_str.clone())
                .map_err(|_| format!("invalid symbol name: {}", h_name_str))?;
            let sym = libc::dlsym(lib, h_cstr.as_ptr());
            if !sym.is_null() {
                eprintln!("[FLA] loaded {} (H={})", base_name, nv);
                return Ok(sym);
            }
            // Fall back to generic (for H-independent kernels)
            let generic_cstr = std::ffi::CString::new(base_name.to_string())
                .map_err(|_| format!("invalid symbol name: {}", base_name))?;
            let sym = libc::dlsym(lib, generic_cstr.as_ptr());
            if !sym.is_null() {
                eprintln!("[FLA] loaded {} (generic)", base_name);
                return Ok(sym);
            }
            Err(format!("{} (H={}) not found in {}", base_name, nv, path))
        };
        macro_rules! load_fla_sym {
            ($name:expr) => {{
                match load_sym_for($name) {
                    Ok(sym) => std::mem::transmute(sym),
                    Err(msg) => {
                        let _ = libc::dlclose(lib);
                        return fail_or_warn(msg);
                    }
                }
            }};
        }

        let kernels = FlaKernels {
            cumsum: load_fla_sym!("krasis_fla_chunk_local_cumsum_scalar_kernel"),
            kkt: load_fla_sym!("krasis_fla_chunk_scaled_dot_kkt_fwd_kernel"),
            solve_tril: load_fla_sym!("krasis_fla_merge_16x16_to_64x64_inverse_kernel"),
            wy_repr: load_fla_sym!("krasis_fla_recompute_w_u_fwd_kernel"),
            state_recurrence: load_fla_sym!("krasis_fla_chunk_gated_delta_rule_fwd_kernel_h_blockdim64"),
            output: load_fla_sym!("krasis_fla_chunk_fwd_kernel_o"),
        };

        log::info!("Loaded FLA (Flash Linear Attention) kernels from {}", path);
        Ok(Some(kernels))
    }
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
                    "kv_cache_append_polar4_kernel",
                    "kv_cache_dequant_concat_polar4_kernel",
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
                "moe_replicate_hidden_kernel",
                    "moe_scatter_fused_kernel",
                    "moe_scatter_weighted_kernel",
                ],
            ).expect("Failed to load prefill kernels PTX");
            #[cfg(has_decode_kernels)]
            dev.load_ptx(
                cudarc::nvrtc::Ptx::from_src(include_str!(concat!(env!("OUT_DIR"), "/decode_kernels_sm80.ptx"))),
                "decode_kernels",
                &[
                    "kv_cache_write_polar4",
                    "gqa_attention_polar4",
                ],
            ).expect("Failed to load decode kernels PTX");
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

        #[cfg(has_decode_kernels)]
        fn get_decode_kernel(&self, name: &str) -> RawCuFunc {
            let func = self.dev.get_func("decode_kernels", name)
                .unwrap_or_else(|| panic!("Decode kernel not found in PTX: {}", name));
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

    #[test]
    fn test_polar4_kv_roundtrip_smoke() {
        let ctx = GpuTestCtx::new();
        let append = ctx.get_kernel("kv_cache_append_polar4_kernel");
        let dequant = ctx.get_kernel("kv_cache_dequant_concat_polar4_kernel");

        let m = 32usize;
        let d = 512usize;
        let max_seq = m;
        let num_blocks = d / 16;

        let make_input = |phase: f32| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(m * d * 2);
            for idx in 0..(m * d) {
                let x = idx as f32;
                let val = ((x * 0.03125 + phase).sin() * 0.75)
                    + ((x * 0.0078125 + phase * 1.7).cos() * 0.25);
                let bf = half::bf16::from_f32(val);
                bytes.extend_from_slice(&bf.to_bits().to_le_bytes());
            }
            bytes
        };

        let k_bytes = make_input(0.3);
        let v_bytes = make_input(1.1);
        let d_k = ctx.upload_bf16(&k_bytes);
        let d_v = ctx.upload_bf16(&v_bytes);

        let d_k_radius = ctx.alloc_bf16(max_seq * num_blocks);
        let d_v_radius = ctx.alloc_bf16(max_seq * num_blocks);
        let d_k_angles = ctx.alloc_u8(max_seq * num_blocks * 8);
        let d_v_angles = ctx.alloc_u8(max_seq * num_blocks * 8);
        let d_k_out = ctx.alloc_bf16(m * d);
        let d_v_out = ctx.alloc_bf16(m * d);

        let mut m_i32 = m as i32;
        let mut d_i32 = d as i32;
        let mut max_seq_i32 = max_seq as i32;
        let mut start_pos_i32: i32 = 0;

        unsafe {
            let threads = num_blocks as u32;
            let mut kr_ptr = *d_k_radius.device_ptr() as u64;
            let mut vr_ptr = *d_v_radius.device_ptr() as u64;
            let mut ka_ptr = *d_k_angles.device_ptr() as u64;
            let mut va_ptr = *d_v_angles.device_ptr() as u64;
            let mut k_ptr = *d_k.device_ptr() as u64;
            let mut v_ptr = *d_v.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut kr_ptr as *mut _ as *mut _,
                &mut vr_ptr as *mut _ as *mut _,
                &mut ka_ptr as *mut _ as *mut _,
                &mut va_ptr as *mut _ as *mut _,
                &mut k_ptr as *mut _ as *mut _,
                &mut v_ptr as *mut _ as *mut _,
                &mut m_i32 as *mut _ as *mut _,
                &mut d_i32 as *mut _ as *mut _,
                &mut max_seq_i32 as *mut _ as *mut _,
                &mut start_pos_i32 as *mut _ as *mut _,
            ];
            launch(append, (m as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let mut cache_len_i32 = m as i32;
        let mut cur_m_i32: i32 = 0;
        unsafe {
            let threads = num_blocks as u32;
            let mut out_ptr = *d_k_out.device_ptr() as u64;
            let mut radius_ptr = *d_k_radius.device_ptr() as u64;
            let mut angles_ptr = *d_k_angles.device_ptr() as u64;
            let mut new_ptr = *d_k.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut out_ptr as *mut _ as *mut _,
                &mut radius_ptr as *mut _ as *mut _,
                &mut angles_ptr as *mut _ as *mut _,
                &mut new_ptr as *mut _ as *mut _,
                &mut cache_len_i32 as *mut _ as *mut _,
                &mut cur_m_i32 as *mut _ as *mut _,
                &mut d_i32 as *mut _ as *mut _,
            ];
            launch(dequant, (m as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();

            let mut out_v_ptr = *d_v_out.device_ptr() as u64;
            let mut radius_v_ptr = *d_v_radius.device_ptr() as u64;
            let mut angles_v_ptr = *d_v_angles.device_ptr() as u64;
            let mut new_v_ptr = *d_v.device_ptr() as u64;
            let mut params_v: Vec<*mut std::ffi::c_void> = vec![
                &mut out_v_ptr as *mut _ as *mut _,
                &mut radius_v_ptr as *mut _ as *mut _,
                &mut angles_v_ptr as *mut _ as *mut _,
                &mut new_v_ptr as *mut _ as *mut _,
                &mut cache_len_i32 as *mut _ as *mut _,
                &mut cur_m_i32 as *mut _ as *mut _,
                &mut d_i32 as *mut _ as *mut _,
            ];
            launch(dequant, (m as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params_v).unwrap();
        }
        self::cuda_sync();

        let actual_k = ctx.download_bf16(&d_k_out);
        let actual_v = ctx.download_bf16(&d_v_out);

        let max_abs_err = |actual: &[u8], expected: &[u8]| -> f32 {
            actual.chunks_exact(2)
                .zip(expected.chunks_exact(2))
                .map(|(a, e)| {
                    let av = half::bf16::from_bits(u16::from_le_bytes([a[0], a[1]])).to_f32();
                    let ev = half::bf16::from_bits(u16::from_le_bytes([e[0], e[1]])).to_f32();
                    (av - ev).abs()
                })
                .fold(0.0f32, f32::max)
        };

        let avg_abs_err = |actual: &[u8], expected: &[u8]| -> f32 {
            let mut sum = 0.0f32;
            let mut n = 0usize;
            for (a, e) in actual.chunks_exact(2).zip(expected.chunks_exact(2)) {
                let av = half::bf16::from_bits(u16::from_le_bytes([a[0], a[1]])).to_f32();
                let ev = half::bf16::from_bits(u16::from_le_bytes([e[0], e[1]])).to_f32();
                sum += (av - ev).abs();
                n += 1;
            }
            sum / n as f32
        };

        let k_max = max_abs_err(&actual_k, &k_bytes);
        let v_max = max_abs_err(&actual_v, &v_bytes);
        let k_avg = avg_abs_err(&actual_k, &k_bytes);
        let v_avg = avg_abs_err(&actual_v, &v_bytes);
        eprintln!(
            "  polar4_kv_roundtrip_smoke stats: k(max={:.6}, avg={:.6}) v(max={:.6}, avg={:.6})",
            k_max, k_avg, v_max, v_avg
        );

        assert!(k_max < 0.40 && v_max < 0.40, "polar4 roundtrip too lossy: k_max={k_max:.6} v_max={v_max:.6}");
        assert!(k_avg < 0.10 && v_avg < 0.10, "polar4 roundtrip avg too lossy: k_avg={k_avg:.6} v_avg={v_avg:.6}");
    }

    #[test]
    fn test_polar4_kv_append_bounds_smoke() {
        let ctx = GpuTestCtx::new();
        let append = ctx.get_kernel("kv_cache_append_polar4_kernel");

        let m = 32usize;
        let d = 512usize;
        let max_seq = m;
        let num_blocks = d / 16;
        let guard_bf16 = 64usize;
        let guard_u8 = 128usize;

        let make_input = |phase: f32| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(m * d * 2);
            for idx in 0..(m * d) {
                let x = idx as f32;
                let val = ((x * 0.03125 + phase).sin() * 0.75)
                    + ((x * 0.0078125 + phase * 1.7).cos() * 0.25);
                let bf = half::bf16::from_f32(val);
                bytes.extend_from_slice(&bf.to_bits().to_le_bytes());
            }
            bytes
        };

        let k_bytes = make_input(0.3);
        let v_bytes = make_input(1.1);
        let d_k = ctx.upload_bf16(&k_bytes);
        let d_v = ctx.upload_bf16(&v_bytes);

        let radius_elems = max_seq * num_blocks;
        let angle_elems = max_seq * num_blocks * 8;
        let radius_guard_val: u16 = 0x7E5A;
        let angle_guard_val: u8 = 0xA5;

        let radius_init: Vec<u16> = vec![radius_guard_val; radius_elems + 2 * guard_bf16];
        let angles_init: Vec<u8> = vec![angle_guard_val; angle_elems + 2 * guard_u8];
        let d_k_radius = ctx.dev.htod_copy(radius_init.clone()).unwrap();
        let d_v_radius = ctx.dev.htod_copy(radius_init.clone()).unwrap();
        let d_k_angles = ctx.dev.htod_copy(angles_init.clone()).unwrap();
        let d_v_angles = ctx.dev.htod_copy(angles_init.clone()).unwrap();

        let mut m_i32 = m as i32;
        let mut d_i32 = d as i32;
        let mut max_seq_i32 = max_seq as i32;
        let mut start_pos_i32: i32 = 0;

        unsafe {
            let threads = num_blocks as u32;
            let mut kr_ptr = (*d_k_radius.device_ptr() as u64) + (guard_bf16 as u64 * 2);
            let mut vr_ptr = (*d_v_radius.device_ptr() as u64) + (guard_bf16 as u64 * 2);
            let mut ka_ptr = (*d_k_angles.device_ptr() as u64) + guard_u8 as u64;
            let mut va_ptr = (*d_v_angles.device_ptr() as u64) + guard_u8 as u64;
            let mut k_ptr = *d_k.device_ptr() as u64;
            let mut v_ptr = *d_v.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut kr_ptr as *mut _ as *mut _,
                &mut vr_ptr as *mut _ as *mut _,
                &mut ka_ptr as *mut _ as *mut _,
                &mut va_ptr as *mut _ as *mut _,
                &mut k_ptr as *mut _ as *mut _,
                &mut v_ptr as *mut _ as *mut _,
                &mut m_i32 as *mut _ as *mut _,
                &mut d_i32 as *mut _ as *mut _,
                &mut max_seq_i32 as *mut _ as *mut _,
                &mut start_pos_i32 as *mut _ as *mut _,
            ];
            launch(append, (m as u32, 1, 1), (threads, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let k_radius_all = ctx.dev.dtoh_sync_copy(&d_k_radius).unwrap();
        let v_radius_all = ctx.dev.dtoh_sync_copy(&d_v_radius).unwrap();
        let k_angles_all = ctx.download_u8(&d_k_angles);
        let v_angles_all = ctx.download_u8(&d_v_angles);

        let check_u16_guards = |name: &str, data: &[u16]| {
            assert!(data[..guard_bf16].iter().all(|&x| x == radius_guard_val),
                "{name}: prefix guard overwritten");
            assert!(data[guard_bf16 + radius_elems..].iter().all(|&x| x == radius_guard_val),
                "{name}: suffix guard overwritten");
        };
        let check_u8_guards = |name: &str, data: &[u8]| {
            assert!(data[..guard_u8].iter().all(|&x| x == angle_guard_val),
                "{name}: prefix guard overwritten");
            assert!(data[guard_u8 + angle_elems..].iter().all(|&x| x == angle_guard_val),
                "{name}: suffix guard overwritten");
        };

        check_u16_guards("k_radius", &k_radius_all);
        check_u16_guards("v_radius", &v_radius_all);
        check_u8_guards("k_angles", &k_angles_all);
        check_u8_guards("v_angles", &v_angles_all);
    }

    #[test]
    fn test_polar4_prefill_append_matches_decode_read_path() {
        let ctx = GpuTestCtx::new();
        let append = ctx.get_kernel("kv_cache_append_polar4_kernel");
        let write_decode = ctx.get_decode_kernel("kv_cache_write_polar4");
        let attn_decode = ctx.get_decode_kernel("gqa_attention_polar4");

        let nh = 32usize;
        let nkv = 8usize;
        let hd = 128usize;
        let seq_len = 12usize;
        let kv_stride = nkv * hd;
        let num_blocks = kv_stride / 16;
        let max_seq = seq_len;
        let sm_sc = 1.0f32 / (hd as f32).sqrt();

        let make_bf16 = |phase: f32, len: usize| -> Vec<u8> {
            let mut bytes = Vec::with_capacity(len * 2);
            for idx in 0..len {
                let x = idx as f32;
                let val = ((x * 0.017 + phase).sin() * 0.70)
                    + ((x * 0.005 + phase * 1.9).cos() * 0.20);
                let bf = half::bf16::from_f32(val);
                bytes.extend_from_slice(&bf.to_bits().to_le_bytes());
            }
            bytes
        };
        let bf16_bytes_to_f32 = |bytes: &[u8]| -> Vec<u8> {
            let mut out = Vec::with_capacity(bytes.len() * 2);
            for chunk in bytes.chunks_exact(2) {
                let bf = half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]));
                out.extend_from_slice(&bf.to_f32().to_le_bytes());
            }
            out
        };

        let k_bf16 = make_bf16(0.25, seq_len * kv_stride);
        let v_bf16 = make_bf16(1.05, seq_len * kv_stride);
        let q_bf16 = make_bf16(0.65, nh * hd);
        let k_f32 = bf16_bytes_to_f32(&k_bf16);
        let v_f32 = bf16_bytes_to_f32(&v_bf16);
        let q_f32 = bf16_bytes_to_f32(&q_bf16);

        let d_k_bf16 = ctx.upload_bf16(&k_bf16);
        let d_v_bf16 = ctx.upload_bf16(&v_bf16);
        let d_k_f32 = ctx.upload_f32(&k_f32);
        let d_v_f32 = ctx.upload_f32(&v_f32);
        let d_q_f32 = ctx.upload_f32(&q_f32);

        let d_prefill_k_radius = ctx.alloc_bf16(max_seq * num_blocks);
        let d_prefill_v_radius = ctx.alloc_bf16(max_seq * num_blocks);
        let d_prefill_k_angles = ctx.alloc_u8(max_seq * num_blocks * 8);
        let d_prefill_v_angles = ctx.alloc_u8(max_seq * num_blocks * 8);
        let d_decode_k_radius = ctx.alloc_bf16(max_seq * num_blocks);
        let d_decode_v_radius = ctx.alloc_bf16(max_seq * num_blocks);
        let d_decode_k_angles = ctx.alloc_u8(max_seq * num_blocks * 8);
        let d_decode_v_angles = ctx.alloc_u8(max_seq * num_blocks * 8);
        let d_out_prefill = ctx.alloc_f32(nh * hd);
        let d_out_decode = ctx.alloc_f32(nh * hd);

        let mut m_i32 = seq_len as i32;
        let mut kv_stride_i32 = kv_stride as i32;
        let mut max_seq_i32 = max_seq as i32;
        let mut start_pos_i32 = 0i32;
        unsafe {
            let mut kr_ptr = *d_prefill_k_radius.device_ptr() as u64;
            let mut vr_ptr = *d_prefill_v_radius.device_ptr() as u64;
            let mut ka_ptr = *d_prefill_k_angles.device_ptr() as u64;
            let mut va_ptr = *d_prefill_v_angles.device_ptr() as u64;
            let mut k_ptr = *d_k_bf16.device_ptr() as u64;
            let mut v_ptr = *d_v_bf16.device_ptr() as u64;
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &mut kr_ptr as *mut _ as *mut _,
                &mut vr_ptr as *mut _ as *mut _,
                &mut ka_ptr as *mut _ as *mut _,
                &mut va_ptr as *mut _ as *mut _,
                &mut k_ptr as *mut _ as *mut _,
                &mut v_ptr as *mut _ as *mut _,
                &mut m_i32 as *mut _ as *mut _,
                &mut kv_stride_i32 as *mut _ as *mut _,
                &mut max_seq_i32 as *mut _ as *mut _,
                &mut start_pos_i32 as *mut _ as *mut _,
            ];
            launch(append, (seq_len as u32, 1, 1), (num_blocks as u32, 1, 1), 0, ctx.stream(), &mut params).unwrap();
        }
        self::cuda_sync();

        let base_k_f32 = *d_k_f32.device_ptr() as u64;
        let base_v_f32 = *d_v_f32.device_ptr() as u64;
        let token_bytes_f32 = (kv_stride * std::mem::size_of::<f32>()) as u64;
        for pos in 0..seq_len {
            let mut kr_ptr = *d_decode_k_radius.device_ptr() as u64;
            let mut vr_ptr = *d_decode_v_radius.device_ptr() as u64;
            let mut ka_ptr = *d_decode_k_angles.device_ptr() as u64;
            let mut va_ptr = *d_decode_v_angles.device_ptr() as u64;
            let mut k_ptr = base_k_f32 + pos as u64 * token_bytes_f32;
            let mut v_ptr = base_v_f32 + pos as u64 * token_bytes_f32;
            let mut pos_i32 = pos as i32;
            unsafe {
                let mut params: Vec<*mut std::ffi::c_void> = vec![
                    &mut kr_ptr as *mut _ as *mut _,
                    &mut vr_ptr as *mut _ as *mut _,
                    &mut ka_ptr as *mut _ as *mut _,
                    &mut va_ptr as *mut _ as *mut _,
                    &mut k_ptr as *mut _ as *mut _,
                    &mut v_ptr as *mut _ as *mut _,
                    &mut pos_i32 as *mut _ as *mut _,
                    &mut kv_stride_i32 as *mut _ as *mut _,
                ];
                launch(
                    write_decode,
                    (((num_blocks as u32) + 255) / 256, 1, 1),
                    (256, 1, 1),
                    0,
                    ctx.stream(),
                    &mut params,
                ).unwrap();
            }
        }
        self::cuda_sync();

        let run_attn = |out: &CudaSlice<f32>, k_radius: &CudaSlice<u16>, v_radius: &CudaSlice<u16>,
                        k_angles: &CudaSlice<u8>, v_angles: &CudaSlice<u8>| {
            let threads = 256u32;
            let num_warps = threads / 32;
            let shared_mem_bytes = ((hd as u32) * (num_warps + 1) + 2 * num_warps) * 4 + 128;
            let mut out_ptr = *out.device_ptr() as u64;
            let mut q_ptr = *d_q_f32.device_ptr() as u64;
            let mut kr_ptr = *k_radius.device_ptr() as u64;
            let mut vr_ptr = *v_radius.device_ptr() as u64;
            let mut ka_ptr = *k_angles.device_ptr() as u64;
            let mut va_ptr = *v_angles.device_ptr() as u64;
            let mut sm_sc_val = sm_sc;
            let mut nh_i32 = nh as i32;
            let mut nkv_i32 = nkv as i32;
            let mut hd_i32 = hd as i32;
            let mut seq_len_i32 = seq_len as i32;
            let mut max_seq_attn_i32 = max_seq as i32;
            unsafe {
                let mut params: Vec<*mut std::ffi::c_void> = vec![
                    &mut out_ptr as *mut _ as *mut _,
                    &mut q_ptr as *mut _ as *mut _,
                    &mut kr_ptr as *mut _ as *mut _,
                    &mut vr_ptr as *mut _ as *mut _,
                    &mut ka_ptr as *mut _ as *mut _,
                    &mut va_ptr as *mut _ as *mut _,
                    &mut sm_sc_val as *mut _ as *mut _,
                    &mut nh_i32 as *mut _ as *mut _,
                    &mut nkv_i32 as *mut _ as *mut _,
                    &mut hd_i32 as *mut _ as *mut _,
                    &mut seq_len_i32 as *mut _ as *mut _,
                    &mut max_seq_attn_i32 as *mut _ as *mut _,
                ];
                launch(
                    attn_decode,
                    (nh as u32, 1, 1),
                    (threads, 1, 1),
                    shared_mem_bytes,
                    ctx.stream(),
                    &mut params,
                ).unwrap();
            }
        };

        run_attn(&d_out_prefill, &d_prefill_k_radius, &d_prefill_v_radius, &d_prefill_k_angles, &d_prefill_v_angles);
        run_attn(&d_out_decode, &d_decode_k_radius, &d_decode_v_radius, &d_decode_k_angles, &d_decode_v_angles);
        self::cuda_sync();

        let out_prefill = ctx.dev.dtoh_sync_copy(&d_out_prefill).unwrap();
        let out_decode = ctx.dev.dtoh_sync_copy(&d_out_decode).unwrap();

        let mut max_abs = 0.0f32;
        let mut avg_abs = 0.0f32;
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (a, b) in out_prefill.iter().zip(out_decode.iter()) {
            let diff = (a - b).abs();
            max_abs = max_abs.max(diff);
            avg_abs += diff;
            dot += (*a as f64) * (*b as f64);
            norm_a += (*a as f64) * (*a as f64);
            norm_b += (*b as f64) * (*b as f64);
        }
        avg_abs /= out_prefill.len() as f32;
        let cosine = dot / ((norm_a.sqrt() * norm_b.sqrt()).max(1e-12));
        eprintln!(
            "  polar4_prefill_vs_decode stats: cosine={:.6} max_abs={:.6} avg_abs={:.6}",
            cosine, max_abs, avg_abs
        );

        assert!(cosine > 0.995, "prefill append vs decode write cosine too low: {cosine:.6}");
        assert!(max_abs < 0.12, "prefill append vs decode write max abs too high: {max_abs:.6}");
        assert!(avg_abs < 0.01, "prefill append vs decode write avg abs too high: {avg_abs:.6}");
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

    if let Some(p) = installed_package_sidecar("libkrasis_marlin.so") {
        return Some(p);
    }

    // 2. Look next to the current executable / shared library
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("libkrasis_marlin.so");
            if p.exists() { return Some(p.to_string_lossy().to_string()); }
        }
    }

    // 3. Search Cargo build output directories for krasis (and krasisx for compat)
    let home = std::env::var("HOME").unwrap_or_default();
    for repo in &["krasis", "krasisx"] {
        for profile in &["release", "debug"] {
            let build_dir = format!("{}/Documents/Claude/{}/target/{}/build", home, repo, profile);
            if let Ok(entries) = std::fs::read_dir(&build_dir) {
                for e in entries.flatten() {
                    let p = e.path().join("out/libkrasis_marlin.so");
                    if p.exists() { return Some(p.to_string_lossy().to_string()); }
                }
            }
        }
    }

    log::warn!("libkrasis_marlin.so not found — Marlin GEMM will be unavailable");
    None
}
