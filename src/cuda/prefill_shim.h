/*
 * Krasis Prefill Shim — C API for GPU prefill kernels.
 *
 * This header defines the interface between Rust (caller) and CUDA kernels.
 * All functions take raw GPU pointers, dimensions, and a CUDA stream.
 * No PyTorch, no Python, no GIL.
 *
 * Memory management is Rust's responsibility (cuMemAlloc/cuMemFree via cudarc).
 * The shim only launches kernels and returns.
 */
#ifndef KRASIS_PREFILL_SHIM_H
#define KRASIS_PREFILL_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Types ─────────────────────────────────────────────────────────────── */

/* Opaque CUDA stream handle (same as cudaStream_t / CUstream) */
typedef void* krasis_stream_t;

/* Scalar quantization type for Marlin kernels.
 * Matches sglang::ScalarType layout exactly so we can pass to the .so. */
typedef struct {
    uint8_t  exponent;
    uint8_t  mantissa;
    uint8_t  is_signed;        /* bool */
    uint8_t  _pad0;
    int32_t  bias;
    uint8_t  finite_values_only; /* bool */
    uint8_t  nan_repr;         /* 0=none, 1=ieee754, 2=extd_range */
    uint8_t  _pad1[2];
} krasis_scalar_type_t;

/* Pre-defined scalar types matching sgl_kernel constants */
#define KRASIS_QTYPE_U4B8    { 0, 4, 0, 0,  8, 0, 0, {0,0} }  /* INT4 unsigned, bias=8 */
#define KRASIS_QTYPE_U8B128  { 0, 8, 0, 0, 128, 0, 0, {0,0} } /* INT8 unsigned, bias=128 */

/* ── Prefill utility kernels (compiled from PTX, no external deps) ───── */

/* Batched RMSNorm: out[i] = norm(x[i]) * weight, for i in 0..num_tokens
 * x: [num_tokens, hidden_size] bf16
 * weight: [hidden_size] bf16
 * out: [num_tokens, hidden_size] bf16
 */
void krasis_rmsnorm_batched(
    void*       out,        /* bf16, [M, D] */
    const void* x,          /* bf16, [M, D] */
    const void* weight,     /* bf16, [D] */
    int         M,          /* num tokens */
    int         D,          /* hidden size */
    float       eps,
    krasis_stream_t stream);

/* Fused add + RMSNorm: residual += x; out = norm(residual) * weight
 * Both residual and out are updated in-place. */
void krasis_fused_add_rmsnorm_batched(
    void*       residual,   /* bf16, [M, D], updated in-place */
    void*       out,        /* bf16, [M, D] */
    const void* x,          /* bf16, [M, D] */
    const void* weight,     /* bf16, [D] */
    int         M,
    int         D,
    float       eps,
    krasis_stream_t stream);

/* Embedding lookup: out[i] = table[token_ids[i]]
 * token_ids: [M] int64 or int32
 * table: [vocab_size, hidden_size] bf16
 * out: [M, hidden_size] bf16
 */
void krasis_embedding_batched(
    void*       out,        /* bf16, [M, D] */
    const void* table,      /* bf16, [V, D] */
    const void* token_ids,  /* int32, [M] */
    int         M,
    int         D,
    krasis_stream_t stream);

/* RoPE (Rotary Position Embedding) applied to Q and K in-place.
 * q: [M, num_heads, head_dim] bf16
 * k: [M, num_kv_heads, head_dim] bf16
 * positions: [M] int32
 * cos_cache, sin_cache: [max_pos, head_dim/2] bf16 (precomputed)
 */
void krasis_rope_batched(
    void*       q,          /* bf16, in-place */
    void*       k,          /* bf16, in-place */
    const void* positions,  /* int32, [M] */
    const void* cos_cache,  /* bf16, [max_pos, head_dim/2] */
    const void* sin_cache,  /* bf16, [max_pos, head_dim/2] */
    int         M,
    int         num_q_heads,
    int         num_kv_heads,
    int         head_dim,
    krasis_stream_t stream);

/* SiLU activation with gating: out = silu(gate) * up
 * gate_up: [M, 2*N] bf16 (gate in first N cols, up in last N cols)
 * out: [M, N] bf16
 */
void krasis_silu_mul_batched(
    void*       out,        /* bf16, [M, N] */
    const void* gate_up,    /* bf16, [M, 2*N] */
    int         M,
    int         N,
    krasis_stream_t stream);

/* relu² activation: out = relu(x)²
 * x: [M, N] bf16
 * out: [M, N] bf16
 */
void krasis_relu2_batched(
    void*       out,        /* bf16, [M, N] */
    const void* x,          /* bf16, [M, N] */
    int         M,
    int         N,
    krasis_stream_t stream);

/* BF16 → FP32 conversion */
void krasis_bf16_to_fp32(
    void*       out_fp32,   /* float, [count] */
    const void* in_bf16,    /* bf16, [count] */
    int         count,
    krasis_stream_t stream);

/* FP32 → BF16 conversion */
void krasis_fp32_to_bf16(
    void*       out_bf16,   /* bf16, [count] */
    const void* in_fp32,    /* float, [count] */
    int         count,
    krasis_stream_t stream);

/* ── MoE routing ───────────────────────────────────────────────────────── */

/* Sigmoid routing with top-k selection.
 * gate_logits: [M, num_experts] bf16 (from gate projection)
 * topk_weights: [M, topk] float (output)
 * topk_ids: [M, topk] int32 (output)
 */
void krasis_sigmoid_topk(
    void*       topk_weights,   /* float, [M, topk] */
    void*       topk_ids,       /* int32, [M, topk] */
    const void* gate_logits,    /* bf16, [M, E] */
    int         M,
    int         num_experts,
    int         topk,
    krasis_stream_t stream);

/* Softmax routing with top-k selection (same signature as sigmoid variant). */
void krasis_softmax_topk(
    void*       topk_weights,
    void*       topk_ids,
    const void* gate_logits,
    int         M,
    int         num_experts,
    int         topk,
    krasis_stream_t stream);

/* MoE sum-reduce: output = sum_k(expert_outputs[k] * topk_weights[k]) * scale
 * expert_outputs: [M * topk, K] bf16
 * output: [M, K] bf16
 * topk: number of active experts per token
 * scale: routed scaling factor
 */
void krasis_moe_sum_reduce(
    void*       output,         /* bf16, [M, K] */
    const void* expert_outputs, /* bf16, [M*topk, K] */
    const void* topk_weights,   /* float, [M, topk] */
    int         M,
    int         K,
    int         topk,
    float       scale,
    krasis_stream_t stream);

/* MoE align block size: sort tokens to experts with block-size padding.
 * Returns sorted_token_ids, expert_ids, num_tokens_post_padded.
 * These are required by the Marlin MoE GEMM kernel.
 */
void krasis_moe_align_block_size(
    void*       sorted_token_ids,     /* int32, [M*topk + padding] */
    void*       expert_ids,           /* int32, [num_blocks] */
    void*       num_tokens_post_pad,  /* int32, [1] */
    const void* topk_ids,             /* int32, [M, topk] */
    int         M,
    int         num_experts,
    int         topk,
    int         block_size,
    krasis_stream_t stream);

/* ── Marlin GEMM (batched, for attention projections) ────────────────── */

/* Marlin INT4/INT8 batched GEMM: C = A @ dequant(B)
 * A: [M, K] bf16 (input activations)
 * B: [K/pack, N*pack/4] int4/int8 packed (Marlin format)
 * C: [M, N] bf16 (output)
 * scales: [num_groups, N] bf16
 */
void krasis_marlin_gemm(
    void*       C,          /* bf16, [M, N] */
    void*       C_tmp,      /* float, [M, N] (scratch for fp32 accumulation) */
    const void* A,          /* bf16, [M, K] */
    const void* B,          /* packed, [K/pack, N*pack/4] */
    const void* scales,     /* bf16, [num_groups, N] */
    const void* workspace,  /* int32, [sms] (Marlin workspace) */
    int         M,
    int         N,
    int         K,
    int         lda,        /* leading dimension of A (usually K) */
    int         num_groups,
    int         group_size,
    int         num_bits,   /* 4 or 8 */
    int         sms,        /* number of streaming multiprocessors */
    int         dev,        /* CUDA device index */
    krasis_stream_t stream);

/* ── Marlin MoE GEMM (batched experts) ──────────────────────────────── */

/* MoE Marlin GEMM: dispatches tokens to experts, runs Marlin per-expert.
 *
 * hidden: [M, K] bf16
 * w1: [E, K/pack, w1_N*pack/4] packed (gate+up or just up)
 * w2: [E, N/pack, K*pack/4] packed (down)
 * w1_scale, w2_scale: [E, num_groups, dim] bf16
 * sorted_token_ids, expert_ids, num_tokens_post_padded: from moe_align_block_size
 * topk_weights: [M, topk] float
 * output: [M, K] bf16
 * is_gated: if true, w1 output is 2*N (gate+up with silu), else N (up with relu2)
 */
void krasis_marlin_moe_gemm(
    void*       output,             /* bf16, [M, K] */
    const void* hidden,             /* bf16, [M, K] */
    const void* w1,                 /* packed */
    const void* w2,                 /* packed */
    const void* w1_scale,           /* bf16 */
    const void* w2_scale,           /* bf16 */
    const void* sorted_token_ids,   /* int32 */
    const void* expert_ids,         /* int32 */
    const void* num_tokens_post_padded, /* int32 */
    const void* topk_weights,       /* float, [M, topk] */
    int         M,
    int         K,                  /* hidden_size */
    int         N,                  /* intermediate_size */
    int         E,                  /* num_experts */
    int         topk,
    int         num_bits,           /* 4 or 8 */
    int         group_size,
    int         block_size_m,
    int         is_gated,           /* 1 = silu gating, 0 = relu2 */
    float       routed_scale,
    int         sms,
    krasis_stream_t stream);

/* ── Attention prefill ─────────────────────────────────────────────────── */

/* GQA paged attention for prefill.
 * q: [M, num_q_heads, head_dim] bf16
 * k, v: [M, num_kv_heads, head_dim] bf16
 * kv_cache: [num_pages, 2, page_size, num_kv_heads, head_dim] bf16/fp8
 * page_table: [max_pages] int32
 * out: [M, num_q_heads, head_dim] bf16
 */
void krasis_gqa_prefill(
    void*       out,            /* bf16, [M, num_q_heads, head_dim] */
    const void* q,              /* bf16, [M, num_q_heads, head_dim] */
    const void* k,              /* bf16, [M, num_kv_heads, head_dim] */
    const void* v,              /* bf16, [M, num_kv_heads, head_dim] */
    void*       kv_cache,       /* bf16/fp8, paged */
    const void* page_table,     /* int32, page indices */
    int         M,              /* sequence length */
    int         num_q_heads,
    int         num_kv_heads,
    int         head_dim,
    int         page_size,
    int         num_existing_tokens, /* tokens already in KV cache (for append) */
    float       softmax_scale,
    int         kv_dtype,       /* 0=bf16, 1=fp8_e4m3 */
    krasis_stream_t stream);

/* Write K,V to paged KV cache (append new tokens).
 * k, v: [M, num_kv_heads, head_dim] bf16
 * kv_cache: [num_pages, 2, page_size, num_kv_heads, head_dim]
 * page_table: page indices
 */
void krasis_kv_cache_append(
    void*       kv_cache,
    const void* k,
    const void* v,
    const void* page_table,
    int         M,
    int         num_kv_heads,
    int         head_dim,
    int         page_size,
    int         start_pos,
    int         kv_dtype,
    krasis_stream_t stream);

/* ── Mamba2 prefill ────────────────────────────────────────────────────── */

/* Mamba2 causal conv1d for prefill (processes full sequence).
 * x: [B, D, L] bf16 (batch, channels, length)
 * weight: [D, width] bf16
 * bias: [D] bf16 (or NULL)
 * out: [B, D, L] bf16
 * conv_state: [B, D, width-1] bf16 (updated with final state)
 */
void krasis_causal_conv1d_fwd(
    void*       out,            /* bf16, [B, D, L] */
    const void* x,              /* bf16, [B, D, L] */
    const void* weight,         /* bf16, [D, width] */
    const void* bias,           /* bf16, [D] or NULL */
    void*       conv_state,     /* bf16, [B, D, width-1], updated */
    int         B,              /* batch size */
    int         D,              /* channels */
    int         L,              /* sequence length */
    int         width,          /* conv kernel width */
    int         silu_activation, /* apply silu after conv */
    krasis_stream_t stream);

/* Mamba2 SSD (Structured State-Space Duality) chunked parallel scan.
 * Processes the full sequence using the chunked SSD algorithm:
 *   - Chunk size C (typically 128)
 *   - Parallel scan within chunks
 *   - Sequential state passing between chunks
 *
 * x: [B, L, n_heads, head_dim] bf16 (post-conv, post-discretize)
 * dt: [B, L, n_heads] bf16 (discretization steps)
 * A: [n_heads] float (state matrix diagonal, log space)
 * B_mat: [B, L, n_groups, state_size] bf16
 * C_mat: [B, L, n_groups, state_size] bf16
 * D_vec: [n_heads] float (skip connection)
 * ssm_state: [B, n_heads, head_dim, state_size] bf16 (updated with final state)
 * out: [B, L, n_heads, head_dim] bf16
 */
void krasis_mamba2_ssd_fwd(
    void*       out,            /* bf16, [B, L, n_heads, head_dim] */
    const void* x,              /* bf16, [B, L, n_heads, head_dim] */
    const void* dt,             /* bf16, [B, L, n_heads] */
    const void* A,              /* float, [n_heads] */
    const void* B_mat,          /* bf16, [B, L, n_groups, state_size] */
    const void* C_mat,          /* bf16, [B, L, n_groups, state_size] */
    const void* D_vec,          /* float, [n_heads] or NULL */
    void*       ssm_state,      /* bf16, [B, n_heads, head_dim, state_size], updated */
    int         B_batch,
    int         L,
    int         n_heads,
    int         head_dim,
    int         state_size,
    int         n_groups,
    int         chunk_size,     /* typically 128 */
    const void* dt_bias,        /* float, [n_heads] or NULL */
    float       dt_softplus,    /* 1.0 to apply softplus to dt, 0.0 to skip */
    krasis_stream_t stream);

/* ── Memory management helpers ─────────────────────────────────────────── */

/* Zero a GPU buffer */
void krasis_zero_buffer(void* ptr, int64_t bytes, krasis_stream_t stream);

/* Copy GPU → GPU on a specific stream */
void krasis_memcpy_d2d(void* dst, const void* src, int64_t bytes, krasis_stream_t stream);

/* Copy CPU → GPU (async, requires pinned host memory) */
void krasis_memcpy_h2d(void* dst, const void* src, int64_t bytes, krasis_stream_t stream);

/* Copy GPU → CPU (async, requires pinned host memory) */
void krasis_memcpy_d2h(void* dst, const void* src, int64_t bytes, krasis_stream_t stream);

/* Stream synchronization */
void krasis_stream_sync(krasis_stream_t stream);

/* ── Initialization ────────────────────────────────────────────────────── */

/* Initialize the prefill shim. Must be called once at startup.
 * Loads external kernel libraries (sgl_kernel, etc.) if available.
 * Returns 0 on success, non-zero on error.
 */
int krasis_prefill_init(int device);

/* Cleanup. */
void krasis_prefill_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* KRASIS_PREFILL_SHIM_H */
