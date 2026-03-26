// Krasis FlashAttention-2 vendor entry points.
// Provides extern "C" functions callable from Rust via dlopen.
// Same pattern as marlin_vendor.cu / marlin_moe_vendor.cu.
//
// Copyright (c) 2023, Tri Dao. (original FlashAttention-2 code)
// Vendored and modified by Krasis for standalone compilation.

#ifndef KRASIS_FA_VENDOR
#define KRASIS_FA_VENDOR
#endif

#include "namespace_config.h"
#include "hardware_info.h"
#include "flash.h"
#include "static_switch.h"

#include <cutlass/numeric_types.h>
#include <cmath>
#include <cstring>

// These are instantiated in the separate flash_fwd_hdim*_bf16_*.cu files.
// We declare them here so the linker resolves them.
namespace FLASH_NAMESPACE {
    template<typename T, int Headdim, bool Is_causal>
    void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
}

// Helper: fill Flash_fwd_params from raw pointers.
// This is the standalone equivalent of set_params_fprop in flash_api.cpp,
// but takes raw GPU pointers instead of torch::Tensor objects.
static void set_params_from_raw(
    flash::Flash_fwd_params &params,
    // Pointers (all device pointers)
    void *q_ptr, void *k_ptr, void *v_ptr, void *out_ptr,
    void *softmax_lse_ptr,   // [num_heads, total_q] or [batch, num_heads, seqlen_q]
    void *cu_seqlens_q_ptr,  // [batch+1], int32  (NULL for fixed-length)
    void *cu_seqlens_k_ptr,  // [batch+1], int32  (NULL for fixed-length)
    // Dimensions
    int batch_size,
    int seqlen_q,       // max seqlen_q (or actual if fixed-length)
    int seqlen_k,       // max seqlen_k (or actual if fixed-length)
    int num_heads,
    int num_heads_k,
    int head_dim,
    int total_q,        // sum of all q sequence lengths (for varlen)
    int total_k,        // sum of all k sequence lengths (for varlen)
    // Config
    float softmax_scale,
    bool is_causal,
    bool is_bf16,
    bool unpadded_lse   // true for varlen: LSE in [num_heads, total_q] format
) {
    memset(&params, 0, sizeof(params));

    params.is_bf16 = is_bf16;

    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = out_ptr;
    params.softmax_lse_ptr = softmax_lse_ptr;

    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = head_dim;
    params.total_q = total_q;

    // Rounded dimensions (for kernel indexing)
    params.seqlen_q_rounded = ((seqlen_q + 127) / 128) * 128;
    params.seqlen_k_rounded = ((seqlen_k + 127) / 128) * 128;
    params.d_rounded = ((head_dim + 31) / 32) * 32;

    // Softmax scaling
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * float(M_LOG2E);
    params.softcap = 0.0f;

    // No dropout for inference
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;

    // Causal mask
    params.is_causal = is_causal;
    params.window_size_left = -1;  // no window
    params.window_size_right = is_causal ? 0 : -1;

    // Cumulative sequence lengths for variable-length batching
    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_ptr);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_ptr);
    params.is_seqlens_k_cumulative = true;

    params.unpadded_lse = unpadded_lse;

    // Strides for varlen (packed format): [total, heads, head_dim]
    if (cu_seqlens_q_ptr != nullptr) {
        // Varlen: tensors are [total_q, num_heads, head_dim]
        params.q_row_stride = num_heads * head_dim;
        params.k_row_stride = num_heads_k * head_dim;
        params.v_row_stride = num_heads_k * head_dim;
        params.o_row_stride = num_heads * head_dim;
        params.q_head_stride = head_dim;
        params.k_head_stride = head_dim;
        params.v_head_stride = head_dim;
        params.o_head_stride = head_dim;
        // Batch strides not used in varlen mode
        params.q_batch_stride = 0;
        params.k_batch_stride = 0;
        params.v_batch_stride = 0;
        params.o_batch_stride = 0;
    } else {
        // Fixed-length: tensors are [batch, seqlen, num_heads, head_dim]
        params.q_row_stride = num_heads * head_dim;
        params.k_row_stride = num_heads_k * head_dim;
        params.v_row_stride = num_heads_k * head_dim;
        params.o_row_stride = num_heads * head_dim;
        params.q_head_stride = head_dim;
        params.k_head_stride = head_dim;
        params.v_head_stride = head_dim;
        params.o_head_stride = head_dim;
        params.q_batch_stride = seqlen_q * num_heads * head_dim;
        params.k_batch_stride = seqlen_k * num_heads_k * head_dim;
        params.v_batch_stride = seqlen_k * num_heads_k * head_dim;
        params.o_batch_stride = seqlen_q * num_heads * head_dim;
    }

    // Not used: rotary, KV append, paged KV, ALiBi
    params.rotary_dim = 0;
    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.is_rotary_interleaved = false;
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.seqlenq_ngroups_swapped = false;
    params.p_ptr = nullptr;
    params.oaccum_ptr = nullptr;
    params.softmax_lseaccum_ptr = nullptr;
    params.blockmask = nullptr;
    params.cache_batch_idx = nullptr;
    params.block_table = nullptr;
    params.leftpad_k = nullptr;
    params.seqused_k = nullptr;
    params.alibi_slopes_ptr = nullptr;
    params.rng_state = nullptr;
    params.num_splits = 0;
    params.seqlen_knew = 0;
}

// ============================================================================
// Extern "C" entry points for Rust FFI (dlopen + dlsym)
// ============================================================================

// Forward pass: BF16, all head dimensions, causal and non-causal.
// The head_dim selects which template instantiation to dispatch to.
// Returns 0 on success, non-zero on error.
extern "C" int krasis_flash_attn_fwd_bf16(
    // Device pointers
    void *q_ptr, void *k_ptr, void *v_ptr, void *out_ptr,
    void *softmax_lse_ptr,
    void *cu_seqlens_q_ptr, void *cu_seqlens_k_ptr,
    // Dimensions
    int batch_size, int seqlen_q, int seqlen_k,
    int num_heads, int num_heads_k, int head_dim,
    int total_q, int total_k,
    // Config
    float softmax_scale, int is_causal,
    int unpadded_lse,
    // CUDA stream
    void *stream_ptr
) {
    flash::Flash_fwd_params params;
    set_params_from_raw(
        params,
        q_ptr, k_ptr, v_ptr, out_ptr,
        softmax_lse_ptr,
        cu_seqlens_q_ptr, cu_seqlens_k_ptr,
        batch_size, seqlen_q, seqlen_k,
        num_heads, num_heads_k, head_dim,
        total_q, total_k,
        softmax_scale, is_causal != 0, true, unpadded_lse != 0
    );

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    // Dispatch by head_dim and causal flag.
    // Only BF16 (cutlass::bfloat16_t) since that's all we use.
    using T = cutlass::bfloat16_t;

    if (is_causal) {
        switch (head_dim) {
            case 64:  flash::run_mha_fwd_<T, 64, true>(params, stream); break;
            case 96:  flash::run_mha_fwd_<T, 96, true>(params, stream); break;
            case 128: flash::run_mha_fwd_<T, 128, true>(params, stream); break;
            case 192: flash::run_mha_fwd_<T, 192, true>(params, stream); break;
            case 256: flash::run_mha_fwd_<T, 256, true>(params, stream); break;
            default:
                fprintf(stderr, "krasis_flash_attn: unsupported head_dim=%d\n", head_dim);
                return -1;
        }
    } else {
        switch (head_dim) {
            case 64:  flash::run_mha_fwd_<T, 64, false>(params, stream); break;
            case 96:  flash::run_mha_fwd_<T, 96, false>(params, stream); break;
            case 128: flash::run_mha_fwd_<T, 128, false>(params, stream); break;
            case 192: flash::run_mha_fwd_<T, 192, false>(params, stream); break;
            case 256: flash::run_mha_fwd_<T, 256, false>(params, stream); break;
            default:
                fprintf(stderr, "krasis_flash_attn: unsupported head_dim=%d\n", head_dim);
                return -1;
        }
    }

    return 0;
}
