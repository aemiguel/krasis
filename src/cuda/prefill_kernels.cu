/*
 * Krasis Prefill Kernels — GPU prefill without Python/PyTorch.
 *
 * Simple element-wise and reduction kernels compiled to PTX via nvcc.
 * Complex kernels (Marlin GEMM, attention, Mamba2 SSD) are in separate files.
 *
 * All functions follow the C API defined in prefill_shim.h.
 * BF16 = unsigned short (cuda_bf16.h nv_bfloat16).
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>

/* ── Helpers ───────────────────────────────────────────────────────────── */

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ __nv_fp8_e4m3 f32_to_fp8e4m3(float x) {
    return __nv_fp8_e4m3(x);
}

__device__ __forceinline__ __nv_fp8_e4m3 bf16_to_fp8e4m3(__nv_bfloat16 x) {
    return f32_to_fp8e4m3(bf16_to_float(x));
}

/* ── Batched RMSNorm ──────────────────────────────────────────────────── */

/* One block per token. Shared memory reduction for variance. */
extern "C" __global__ void rmsnorm_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    int D,
    float eps)
{
    int token = blockIdx.x;
    const __nv_bfloat16* x_row = x + (int64_t)token * D;
    __nv_bfloat16* o_row = out + (int64_t)token * D;

    extern __shared__ float smem[];

    /* Compute sum of squares */
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]);
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    /* Tree reduction */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / (float)D + eps);

    /* Normalize and scale */
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]) * rms_inv;
        o_row[i] = float_to_bf16(v * bf16_to_float(weight[i]));
    }
}

extern "C" __global__ void rmsnorm_batched_fp32w_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ weight,
    int D,
    float eps)
{
    int token = blockIdx.x;
    const __nv_bfloat16* x_row = x + (int64_t)token * D;
    __nv_bfloat16* o_row = out + (int64_t)token * D;

    extern __shared__ float smem[];

    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]);
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / (float)D + eps);

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]) * rms_inv;
        o_row[i] = float_to_bf16(v * weight[i]);
    }
}

extern "C" void krasis_rmsnorm_batched(
    void* out, const void* x, const void* weight,
    int M, int D, float eps, void* stream)
{
    if (M == 0) return;
    int threads = min(1024, D);
    /* Round up to next warp */
    threads = ((threads + 31) / 32) * 32;
    int smem = threads * sizeof(float);
    rmsnorm_batched_kernel<<<M, threads, smem, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)weight, D, eps);
}

/* ── Fused Add + RMSNorm ──────────────────────────────────────────────── */

extern "C" __global__ void fused_add_rmsnorm_batched_kernel(
    __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    int D,
    float eps)
{
    int token = blockIdx.x;
    __nv_bfloat16* res_row = residual + (int64_t)token * D;
    __nv_bfloat16* o_row = out + (int64_t)token * D;
    const __nv_bfloat16* x_row = x + (int64_t)token * D;

    extern __shared__ float smem[];

    /* First pass: add and compute sum of squares */
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float r = bf16_to_float(res_row[i]) + bf16_to_float(x_row[i]);
        res_row[i] = float_to_bf16(r);
        local_ss += r * r;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / (float)D + eps);

    /* Second pass: normalize */
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(res_row[i]) * rms_inv;
        o_row[i] = float_to_bf16(v * bf16_to_float(weight[i]));
    }
}

extern "C" void krasis_fused_add_rmsnorm_batched(
    void* residual, void* out, const void* x, const void* weight,
    int M, int D, float eps, void* stream)
{
    if (M == 0) return;
    int threads = min(1024, D);
    threads = ((threads + 31) / 32) * 32;
    int smem = threads * sizeof(float);
    fused_add_rmsnorm_batched_kernel<<<M, threads, smem, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)residual, (__nv_bfloat16*)out,
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)weight, D, eps);
}

/* ── Embedding Lookup ──────────────────────────────────────────────────── */

extern "C" __global__ void embedding_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ table,
    const int* __restrict__ token_ids,
    int D)
{
    int token = blockIdx.x;
    int tid = token_ids[token];
    const __nv_bfloat16* src = table + (int64_t)tid * D;
    __nv_bfloat16* dst = out + (int64_t)token * D;

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = src[i];
    }
}

extern "C" void krasis_embedding_batched(
    void* out, const void* table, const void* token_ids,
    int M, int D, void* stream)
{
    if (M == 0) return;
    int threads = min(1024, D);
    threads = ((threads + 31) / 32) * 32;
    embedding_batched_kernel<<<M, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)table,
        (const int*)token_ids, D);
}

/* ── HQQ4 Dequant ─────────────────────────────────────────────────────── */

extern "C" __global__ void hqq4_dequant_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int rows,
    int cols,
    int group_size)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = rows * cols;
    if (idx >= total) return;

    int row = idx / cols;
    int col = idx - row * cols;
    int groups = (cols + group_size - 1) / group_size;
    int packed_cols = (groups * group_size) / 2;
    int group = col / group_size;
    int packed_idx = row * packed_cols + (col >> 1);
    unsigned char byte = packed[packed_idx];
    int q = (col & 1) ? (int)(byte >> 4) : (int)(byte & 0x0F);
    float scale = scales[row * groups + group];
    float zero = zeros[row * groups + group];
    out[idx] = float_to_bf16((float(q) - zero) * scale);
}

extern "C" void krasis_hqq4_dequant_bf16(
    void* out,
    const void* packed,
    const void* scales,
    const void* zeros,
    int rows,
    int cols,
    int group_size,
    void* stream)
{
    int total = rows * cols;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    hqq4_dequant_bf16_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out,
        (const unsigned char*)packed,
        (const float*)scales,
        (const float*)zeros,
        rows,
        cols,
        group_size);
}

__device__ __forceinline__ int hqq_group_idx(int col, int group_size) {
    return group_size == 128 ? (col >> 7) : (col / group_size);
}

template<int NBITS>
__device__ __forceinline__ float hqq_load_weight(
    const unsigned char* __restrict__ row,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int col,
    int group_size)
{
    int q;
    if constexpr (NBITS == 4) {
        unsigned char packed = row[col >> 1];
        q = (col & 1) ? (int)(packed >> 4) : (int)(packed & 0x0F);
    } else if constexpr (NBITS == 6) {
        int group4 = col >> 2;
        int offset = col & 3;
        const unsigned char* tri = row + group4 * 3;
        unsigned int bits = ((unsigned int)tri[0]) | (((unsigned int)tri[1]) << 8) | (((unsigned int)tri[2]) << 16);
        q = (bits >> (offset * 6)) & 0x3F;
    } else {
        q = (int)row[col];
    }
    int group = hqq_group_idx(col, group_size);
    return ((float)q - zeros[group]) * scales[group];
}

template<int NBITS>
__device__ void hqq_quantized_prefill_gemm_bf16_device(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int M,
    int rows,
    int cols,
    int group_size,
    int packed_row_stride_bytes,
    int scales_row_stride_bytes,
    int zeros_row_stride_bytes)
{
    __shared__ float partial[256];
    int tid = threadIdx.x;
    int lane = tid & 3;
    int out_lane = tid >> 2;
    int local_m = out_lane >> 3;
    int local_row = out_lane & 7;
    int token = blockIdx.y * 8 + local_m;
    int row = blockIdx.x * 8 + local_row;
    if (token >= M || row >= rows) return;

    const __nv_bfloat16* x_row = input + (long long)token * cols;
    const unsigned char* w_row = packed + (long long)row * packed_row_stride_bytes;
    const float* s_row = (const float*)((const char*)scales + (long long)row * scales_row_stride_bytes);
    const float* z_row = (const float*)((const char*)zeros + (long long)row * zeros_row_stride_bytes);

    float acc = 0.0f;
    for (int col = lane; col < cols; col += 4) {
        float w = hqq_load_weight<NBITS>(w_row, s_row, z_row, col, group_size);
        float x = bf16_to_float(x_row[col]);
        acc += w * x;
    }

    partial[tid] = acc;
    __syncthreads();
    if (lane == 0) {
        float sum = partial[tid] + partial[tid + 1] + partial[tid + 2] + partial[tid + 3];
        out[(long long)token * rows + row] = float_to_bf16(sum);
    }
}

extern "C" __global__ void hqq4_prefill_gemm_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int M,
    int rows,
    int cols,
    int group_size,
    int packed_row_stride_bytes,
    int scales_row_stride_bytes,
    int zeros_row_stride_bytes)
{
    hqq_quantized_prefill_gemm_bf16_device<4>(
        out, input, packed, scales, zeros, M, rows, cols, group_size,
        packed_row_stride_bytes, scales_row_stride_bytes, zeros_row_stride_bytes);
}

extern "C" __global__ void hqq8_prefill_gemm_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int M,
    int rows,
    int cols,
    int group_size,
    int packed_row_stride_bytes,
    int scales_row_stride_bytes,
    int zeros_row_stride_bytes)
{
    hqq_quantized_prefill_gemm_bf16_device<8>(
        out, input, packed, scales, zeros, M, rows, cols, group_size,
        packed_row_stride_bytes, scales_row_stride_bytes, zeros_row_stride_bytes);
}

extern "C" __global__ void hqq6_prefill_gemm_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    int M,
    int rows,
    int cols,
    int group_size,
    int packed_row_stride_bytes,
    int scales_row_stride_bytes,
    int zeros_row_stride_bytes)
{
    hqq_quantized_prefill_gemm_bf16_device<6>(
        out, input, packed, scales, zeros, M, rows, cols, group_size,
        packed_row_stride_bytes, scales_row_stride_bytes, zeros_row_stride_bytes);
}

extern "C" __global__ void hqq_prefill_group_sums_bf16_kernel(
    float* __restrict__ group_sums,
    const __nv_bfloat16* __restrict__ input,
    int M,
    int cols,
    int group_size,
    int groups)
{
    __shared__ float smem[256];
    int token = blockIdx.x;
    int group = blockIdx.y;
    int tid = threadIdx.x;
    if (token >= M || group >= groups) return;

    int start = group * group_size;
    int end = min(start + group_size, cols);
    float acc = 0.0f;
    const __nv_bfloat16* x_row = input + (long long)token * cols;
    for (int col = start + tid; col < end; col += blockDim.x) {
        acc += bf16_to_float(x_row[col]);
    }
    smem[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        group_sums[(long long)token * groups + group] = smem[0];
    }
}

extern "C" __global__ void hqq8_marlin_zero_correct_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ delta_out,
    const float* __restrict__ group_sums,
    const float* __restrict__ zero_correction,
    int M,
    int rows,
    int groups)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * rows;
    if (idx >= total) return;

    int row = (int)(idx % rows);
    int token = (int)(idx / rows);
    float acc = 0.0f;
    const float* sums = group_sums + (long long)token * groups;
    const float* corr = zero_correction + (long long)row * groups;
    for (int g = 0; g < groups; ++g) {
        acc += sums[g] * corr[g];
    }
    __nv_bfloat16 base = out[idx];
    __nv_bfloat16 delta = delta_out[idx];
    out[idx] = float_to_bf16(bf16_to_float(base) + bf16_to_float(delta) + acc);
}

extern "C" __global__ void hqq8_marlin_intercept_correct_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ group_sums,
    const float* __restrict__ intercept_correction,
    int M,
    int rows,
    int groups)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * rows;
    if (idx >= total) return;

    int row = (int)(idx % rows);
    int token = (int)(idx / rows);
    float acc = 0.0f;
    const float* sums = group_sums + (long long)token * groups;
    const float* corr = intercept_correction + (long long)row * groups;
    for (int g = 0; g < groups; ++g) {
        acc += sums[g] * corr[g];
    }
    out[idx] = float_to_bf16(bf16_to_float(out[idx]) + acc);
}

extern "C" __global__ void hqq_prefill_int8_exception_delta_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const signed char* __restrict__ exception_qint8,
    const float* __restrict__ exception_scales,
    const int* __restrict__ output_rows,
    const int* __restrict__ start_cols,
    const int* __restrict__ widths,
    const float* __restrict__ hqq_base_f32,
    const unsigned char* __restrict__ hqq_packed_w,
    const float* __restrict__ hqq_scales,
    const float* __restrict__ hqq_zeros,
    int M,
    int rows,
    int row_group_count,
    int cols,
    int group_size,
    int max_width,
    int packed_row_stride_bytes,
    int scales_row_stride_bytes,
    int zeros_row_stride_bytes,
    int nbits)
{
    extern __shared__ float smem[];
    int entry = blockIdx.x;
    int token = blockIdx.y;
    int tid = threadIdx.x;
    if (entry >= row_group_count || token >= M) return;

    int row = output_rows[entry];
    int start_col = start_cols[entry];
    int width = widths[entry];
    const __nv_bfloat16* x_row = input + (long long)token * cols;
    float exc_scale = exception_scales[entry];

    float acc = 0.0f;
    for (int local = tid; local < width; local += blockDim.x) {
        int col = start_col + local;
        if (col >= cols) continue;
        float hqq_w;
        if (hqq_base_f32 != nullptr) {
            hqq_w = hqq_base_f32[entry * max_width + local];
        } else {
            const unsigned char* w_row = hqq_packed_w + (long long)row * packed_row_stride_bytes;
            const float* s_row = (const float*)((const char*)hqq_scales + (long long)row * scales_row_stride_bytes);
            const float* z_row = (const float*)((const char*)hqq_zeros + (long long)row * zeros_row_stride_bytes);
            int q;
            if (nbits == 4) {
                unsigned char packed_byte = w_row[col >> 1];
                q = (col & 1) ? (int)(packed_byte >> 4) : (int)(packed_byte & 0x0F);
            } else {
                q = (int)w_row[col];
            }
            int group = hqq_group_idx(col, group_size);
            hqq_w = ((float)q - z_row[group]) * s_row[group];
        }
        float int8_w = (float)exception_qint8[entry * max_width + local] * exc_scale;
        float x = bf16_to_float(x_row[col]);
        acc += (int8_w - hqq_w) * x;
    }

    smem[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        __nv_bfloat16 base = out[(long long)token * rows + row];
        out[(long long)token * rows + row] = float_to_bf16(bf16_to_float(base) + smem[0]);
    }
}

extern "C" __global__ void hqq_apply_sidecar_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const signed char* __restrict__ correction_qint8,
    const __nv_bfloat16* __restrict__ correction_bf16,
    const float* __restrict__ scales,
    const int* __restrict__ output_rows,
    const int* __restrict__ start_cols,
    const int* __restrict__ widths,
    int row_group_count,
    int cols,
    int max_width,
    int mode)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = row_group_count * max_width;
    if (idx >= total) return;

    int entry = idx / max_width;
    int local = idx - entry * max_width;
    int width = widths[entry];
    if (local >= width) return;

    int row = output_rows[entry];
    int col = start_cols[entry] + local;
    float value = 0.0f;
    if (mode == 1) {
        value = (float)correction_qint8[idx] * scales[entry];
    } else if (mode == 2) {
        value = bf16_to_float(correction_bf16[idx]);
    } else if (mode == 3) {
        value = (float)correction_qint8[idx] * scales[entry];
    } else {
        return;
    }
    int out_idx = row * cols + col;
    if (mode == 3) {
        out[out_idx] = float_to_bf16(value);
    } else {
        float base = bf16_to_float(out[out_idx]);
        out[out_idx] = float_to_bf16(base + value);
    }
}

/* ── RoPE (Rotary Position Embedding) ─────────────────────────────────── */

/* Apply RoPE to Q and K tensors in-place.
 * Layout: [M, num_heads, head_dim] bf16
 * cos/sin cache: [max_pos, head_dim/2] bf16
 */
extern "C" __global__ void rope_batched_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const int* __restrict__ positions,
    const float* __restrict__ cos_cache,   /* FP32 [max_seq, half_dim] */
    const float* __restrict__ sin_cache,   /* FP32 [max_seq, half_dim] */
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int half_dim)
{
    int token = blockIdx.x;
    int pos = positions[token];

    const float* cos_row = cos_cache + (int64_t)pos * half_dim;
    const float* sin_row = sin_cache + (int64_t)pos * half_dim;

    /* Apply to Q heads */
    int q_stride = num_q_heads * head_dim;
    __nv_bfloat16* q_row = q + (int64_t)token * q_stride;
    for (int h = 0; h < num_q_heads; h++) {
        __nv_bfloat16* qh = q_row + h * head_dim;
        for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
            float q0 = bf16_to_float(qh[i]);
            float q1 = bf16_to_float(qh[i + half_dim]);
            float c = cos_row[i];
            float s = sin_row[i];
            qh[i] = float_to_bf16(q0 * c - q1 * s);
            qh[i + half_dim] = float_to_bf16(q1 * c + q0 * s);
        }
    }

    /* Apply to K heads */
    int k_stride = num_kv_heads * head_dim;
    __nv_bfloat16* k_row = k + (int64_t)token * k_stride;
    for (int h = 0; h < num_kv_heads; h++) {
        __nv_bfloat16* kh = k_row + h * head_dim;
        for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
            float k0 = bf16_to_float(kh[i]);
            float k1 = bf16_to_float(kh[i + half_dim]);
            float c = cos_row[i];
            float s = sin_row[i];
            kh[i] = float_to_bf16(k0 * c - k1 * s);
            kh[i + half_dim] = float_to_bf16(k1 * c + k0 * s);
        }
    }
}

extern "C" void krasis_rope_batched(
    void* q, void* k, const void* positions,
    const void* cos_cache, const void* sin_cache,
    int M, int num_q_heads, int num_kv_heads, int head_dim,
    void* stream)
{
    if (M == 0) return;
    int half_dim = head_dim / 2;
    int threads = min(512, half_dim);
    threads = ((threads + 31) / 32) * 32;
    if (threads == 0) threads = 32;
    rope_batched_kernel<<<M, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)q, (__nv_bfloat16*)k,
        (const int*)positions,
        (const float*)cos_cache,
        (const float*)sin_cache,
        num_q_heads, num_kv_heads, head_dim, half_dim);
}

/* ── SiLU + Mul ────────────────────────────────────────────────────────── */

/* gate_up: [M, 2*N], out: [M, N]
 * out[i,j] = silu(gate_up[i,j]) * gate_up[i, N+j]
 */
extern "C" __global__ void silu_mul_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gate_up,
    int N)
{
    int token = blockIdx.x;
    int two_N = 2 * N;
    const __nv_bfloat16* gu = gate_up + (int64_t)token * two_N;
    __nv_bfloat16* o = out + (int64_t)token * N;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float gate = bf16_to_float(gu[i]);
        float up = bf16_to_float(gu[N + i]);
        float silu_gate = gate / (1.0f + __expf(-gate));
        o[i] = float_to_bf16(silu_gate * up);
    }
}

extern "C" void krasis_silu_mul_batched(
    void* out, const void* gate_up,
    int M, int N, void* stream)
{
    if (M == 0) return;
    int threads = min(1024, N);
    threads = ((threads + 31) / 32) * 32;
    silu_mul_batched_kernel<<<M, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)gate_up, N);
}

/* ── ReLU² ─────────────────────────────────────────────────────────────── */

/* out[i,j] = max(0, x[i,j])² */
extern "C" __global__ void relu2_batched_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    int N)
{
    int token = blockIdx.x;
    const __nv_bfloat16* x_row = x + (int64_t)token * N;
    __nv_bfloat16* o_row = out + (int64_t)token * N;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]);
        v = fmaxf(v, 0.0f);
        o_row[i] = float_to_bf16(v * v);
    }
}

extern "C" void krasis_relu2_batched(
    void* out, const void* x,
    int M, int N, void* stream)
{
    if (M == 0) return;
    int threads = min(1024, N);
    threads = ((threads + 31) / 32) * 32;
    relu2_batched_kernel<<<M, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)x, N);
}

/* ── Sigmoid-gated multiply (for gated GQA attention) ───────────────── */

/* out[i] = attn[i] * sigmoid(gate[i])
 * Used by QCN: q_proj outputs [query, gate], gate applied to attention output.
 * attn: [M, N] bf16 — attention output
 * gate: [M, N] bf16 — gate values (raw logits, sigmoid applied here)
 * out:  [M, N] bf16 — gated output (can be same buffer as attn)
 */
extern "C" __global__ void sigmoid_mul_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ attn,
    const __nv_bfloat16* __restrict__ gate,
    int N)
{
    int token = blockIdx.x;
    const __nv_bfloat16* a_row = attn + (int64_t)token * N;
    const __nv_bfloat16* g_row = gate + (int64_t)token * N;
    __nv_bfloat16* o_row = out + (int64_t)token * N;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float a = bf16_to_float(a_row[i]);
        float g = bf16_to_float(g_row[i]);
        float sig = 1.0f / (1.0f + __expf(-g));
        o_row[i] = float_to_bf16(a * sig);
    }
}

/* ── BF16 ↔ FP32 conversion ──────────────────────────────────────────── */

extern "C" __global__ void bf16_to_fp32_kernel(
    float* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = bf16_to_float(in[idx]);
    }
}

extern "C" void krasis_bf16_to_fp32(
    void* out_fp32, const void* in_bf16, int count, void* stream)
{
    if (count == 0) return;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    bf16_to_fp32_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)out_fp32, (const __nv_bfloat16*)in_bf16, count);
}

extern "C" __global__ void fp32_to_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ in,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = float_to_bf16(in[idx]);
    }
}

extern "C" void krasis_fp32_to_bf16(
    void* out_bf16, const void* in_fp32, int count, void* stream)
{
    if (count == 0) return;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    fp32_to_bf16_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out_bf16, (const float*)in_fp32, count);
}

/* ── Sigmoid Top-K routing ─────────────────────────────────────────────── */

/* One block per token. Computes sigmoid(gate_logits), then selects top-k.
 * Gate input is FP32 (from cuBLAS GEMM output). */
extern "C" __global__ void sigmoid_topk_kernel(
    float* __restrict__ topk_weights,       /* [M, topk] */
    int* __restrict__ topk_ids,             /* [M, topk] */
    const float* __restrict__ gate,         /* [M, E] FP32 */
    int E,
    int topk)
{
    int token = blockIdx.x;
    const float* g = gate + (int64_t)token * E;
    float* tw = topk_weights + (int64_t)token * topk;
    int* ti = topk_ids + (int64_t)token * topk;

    /* Initialize top-k with -inf */
    extern __shared__ char smem_raw[];
    float* scores = (float*)smem_raw;    /* [E] */
    float* top_vals = scores + E;        /* [topk] */
    int* top_idxs = (int*)(top_vals + topk); /* [topk] */

    /* Compute sigmoid scores */
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] = 1.0f / (1.0f + __expf(-g[i]));
    }
    __syncthreads();

    /* Single-threaded top-k selection (E is typically small, e.g. 128) */
    if (threadIdx.x == 0) {
        for (int k = 0; k < topk; k++) {
            top_vals[k] = -1e30f;
            top_idxs[k] = -1;
        }
        for (int i = 0; i < E; i++) {
            float s = scores[i];
            /* Find insertion point in sorted top-k */
            if (s > top_vals[topk - 1]) {
                int pos = topk - 1;
                while (pos > 0 && s > top_vals[pos - 1]) {
                    top_vals[pos] = top_vals[pos - 1];
                    top_idxs[pos] = top_idxs[pos - 1];
                    pos--;
                }
                top_vals[pos] = s;
                top_idxs[pos] = i;
            }
        }
        for (int k = 0; k < topk; k++) {
            tw[k] = top_vals[k];
            ti[k] = top_idxs[k];
        }
    }
}

extern "C" void krasis_sigmoid_topk(
    void* topk_weights, void* topk_ids, const void* gate_logits,
    int M, int num_experts, int topk, void* stream)
{
    if (M == 0) return;
    int threads = min(256, num_experts);
    threads = ((threads + 31) / 32) * 32;
    if (threads == 0) threads = 32;
    int smem = num_experts * sizeof(float) + topk * (sizeof(float) + sizeof(int));
    sigmoid_topk_kernel<<<M, threads, smem, (cudaStream_t)stream>>>(
        (float*)topk_weights, (int*)topk_ids,
        (const float*)gate_logits,
        num_experts, topk);
}

/* ── Softmax Top-K routing ─────────────────────────────────────────────── */

/* Gate input is FP32 (from cuBLAS GEMM output).
 * Uses warp-shuffle reduction instead of atomicMax to handle negative floats correctly. */
extern "C" __global__ void softmax_topk_kernel(
    float* __restrict__ topk_weights,
    int* __restrict__ topk_ids,
    const float* __restrict__ gate,
    int E,
    int topk)
{
    int token = blockIdx.x;
    const float* g = gate + (int64_t)token * E;
    float* tw = topk_weights + (int64_t)token * topk;
    int* ti = topk_ids + (int64_t)token * topk;

    extern __shared__ char smem_raw[];
    float* scores = (float*)smem_raw;
    float* top_vals = scores + E;
    int* top_idxs = (int*)(top_vals + topk);

    /* Compute exp(logit - max), then normalize only selected top-k values.
     * The full softmax denominator cancels during top-k renormalization, and
     * avoiding it removes atomicAdd reduction-order sensitivity. */
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] = g[i];
        max_val = fmaxf(max_val, scores[i]);
    }
    /* Reduce max across threads using warp shuffle + shared memory
     * (atomicMax on float-as-int fails for negative values) */
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }
    __shared__ float s_warp_max[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) s_warp_max[warp_id] = max_val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -1e30f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) m = fmaxf(m, s_warp_max[w]);
        s_warp_max[0] = m;
    }
    __syncthreads();
    max_val = s_warp_max[0];

    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] = __expf(scores[i] - max_val);
    }
    __syncthreads();

    /* Top-k selection (single thread, E small) */
    if (threadIdx.x == 0) {
        for (int k = 0; k < topk; k++) {
            top_vals[k] = -1e30f;
            top_idxs[k] = -1;
        }
        for (int i = 0; i < E; i++) {
            float s = scores[i];
            if (s > top_vals[topk - 1]) {
                int pos = topk - 1;
                while (pos > 0 && s > top_vals[pos - 1]) {
                    top_vals[pos] = top_vals[pos - 1];
                    top_idxs[pos] = top_idxs[pos - 1];
                    pos--;
                }
                top_vals[pos] = s;
                top_idxs[pos] = i;
            }
        }
        /* Normalize top-k weights to sum to 1.0 */
        float wsum = 0.0f;
        for (int k = 0; k < topk; k++) wsum += top_vals[k];
        float inv_wsum = (wsum > 0.0f) ? 1.0f / wsum : 0.0f;
        for (int k = 0; k < topk; k++) {
            tw[k] = top_vals[k] * inv_wsum;
            ti[k] = top_idxs[k];
        }
    }
}

extern "C" void krasis_softmax_topk(
    void* topk_weights, void* topk_ids, const void* gate_logits,
    int M, int num_experts, int topk, void* stream)
{
    if (M == 0) return;
    int threads = min(256, num_experts);
    threads = ((threads + 31) / 32) * 32;
    if (threads == 0) threads = 32;
    /* Extra smem for warp-max reduction: 32 floats */
    int smem = num_experts * sizeof(float) + topk * (sizeof(float) + sizeof(int)) + 32 * sizeof(float);
    softmax_topk_kernel<<<M, threads, smem, (cudaStream_t)stream>>>(
        (float*)topk_weights, (int*)topk_ids,
        (const float*)gate_logits,
        num_experts, topk);
}

/* Diagnostic-only softmax top-k sum probe. This reproduces the denominator and
 * selected-weight path for observation, but writes only to a separate debug
 * buffer and is never used by normal routing. Output row layout:
 * [max, sum, inv_sum, topk_prob_sum, inv_topk_sum,
 *  topk ids as f32, topk softmax probabilities, topk renormalized weights].
 */
extern "C" __global__ void softmax_topk_sum_probe_kernel(
    float* __restrict__ probe_out,
    const float* __restrict__ gate,
    int E,
    int topk,
    int fields)
{
    int token = blockIdx.x;
    const float* g = gate + (int64_t)token * E;
    float* out = probe_out + (int64_t)token * fields;

    extern __shared__ char smem_raw[];
    float* scores = (float*)smem_raw;
    float* top_vals = scores + E;
    int* top_idxs = (int*)(top_vals + topk);

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] = g[i];
        max_val = fmaxf(max_val, scores[i]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }
    __shared__ float s_warp_max[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) s_warp_max[warp_id] = max_val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -1e30f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) m = fmaxf(m, s_warp_max[w]);
        s_warp_max[0] = m;
    }
    __syncthreads();
    max_val = s_warp_max[0];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] = __expf(scores[i] - max_val);
        sum += scores[i];
    }
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum);
    __syncthreads();
    float inv_sum = 1.0f / s_sum;

    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < topk; k++) {
            top_vals[k] = -1e30f;
            top_idxs[k] = -1;
        }
        for (int i = 0; i < E; i++) {
            float s = scores[i];
            if (s > top_vals[topk - 1]) {
                int pos = topk - 1;
                while (pos > 0 && s > top_vals[pos - 1]) {
                    top_vals[pos] = top_vals[pos - 1];
                    top_idxs[pos] = top_idxs[pos - 1];
                    pos--;
                }
                top_vals[pos] = s;
                top_idxs[pos] = i;
            }
        }
        float wsum = 0.0f;
        for (int k = 0; k < topk; k++) wsum += top_vals[k];
        float inv_wsum = (wsum > 0.0f) ? 1.0f / wsum : 0.0f;

        if (fields >= 5 + 3 * topk) {
            out[0] = max_val;
            out[1] = s_sum;
            out[2] = inv_sum;
            out[3] = wsum;
            out[4] = inv_wsum;
            for (int k = 0; k < topk; k++) {
                out[5 + k] = (float)top_idxs[k];
                out[5 + topk + k] = top_vals[k];
                out[5 + 2 * topk + k] = top_vals[k] * inv_wsum;
            }
        }
    }
}

/* Diagnostic-only selected-logit normalization probe. Unlike
 * softmax_topk_sum_probe_kernel, this never uses the full softmax denominator:
 * it selects top-k by exp(logit - max) and normalizes only the selected exp
 * values. It writes only to a separate debug buffer.
 *
 * Output row layout:
 * [max, selected_exp_sum, inv_selected_exp_sum,
 *  topk ids as f32, selected exp values, selected-only weights].
 */
extern "C" __global__ void softmax_topk_selected_probe_kernel(
    float* __restrict__ probe_out,
    const float* __restrict__ gate,
    int E,
    int topk,
    int fields)
{
    int token = blockIdx.x;
    const float* g = gate + (int64_t)token * E;
    float* out = probe_out + (int64_t)token * fields;

    extern __shared__ char smem_raw[];
    float* scores = (float*)smem_raw;
    float* top_vals = scores + E;
    int* top_idxs = (int*)(top_vals + topk);

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] = g[i];
        max_val = fmaxf(max_val, scores[i]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }
    __shared__ float s_warp_max[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) s_warp_max[warp_id] = max_val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -1e30f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) m = fmaxf(m, s_warp_max[w]);
        s_warp_max[0] = m;
    }
    __syncthreads();
    max_val = s_warp_max[0];

    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] = __expf(scores[i] - max_val);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < topk; k++) {
            top_vals[k] = -1e30f;
            top_idxs[k] = -1;
        }
        for (int i = 0; i < E; i++) {
            float s = scores[i];
            if (s > top_vals[topk - 1]) {
                int pos = topk - 1;
                while (pos > 0 && s > top_vals[pos - 1]) {
                    top_vals[pos] = top_vals[pos - 1];
                    top_idxs[pos] = top_idxs[pos - 1];
                    pos--;
                }
                top_vals[pos] = s;
                top_idxs[pos] = i;
            }
        }
        float wsum = 0.0f;
        for (int k = 0; k < topk; k++) wsum += top_vals[k];
        float inv_wsum = (wsum > 0.0f) ? 1.0f / wsum : 0.0f;

        if (fields >= 3 + 3 * topk) {
            out[0] = max_val;
            out[1] = wsum;
            out[2] = inv_wsum;
            for (int k = 0; k < topk; k++) {
                out[3 + k] = (float)top_idxs[k];
                out[3 + topk + k] = top_vals[k];
                out[3 + 2 * topk + k] = top_vals[k] * inv_wsum;
            }
        }
    }
}

/* ── MoE Sum Reduce ────────────────────────────────────────────────────── */

/* Reduce expert outputs weighted by topk_weights.
 * expert_outputs: [M*topk, K] viewed as [M, topk, K]
 * output: [M, K]
 */
extern "C" __global__ void moe_sum_reduce_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ expert_outputs,
    const float* __restrict__ topk_weights,
    int K,
    int topk,
    float scale)
{
    int token = blockIdx.x;
    __nv_bfloat16* o = output + (int64_t)token * K;
    const float* tw = topk_weights + (int64_t)token * topk;

    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < topk; k++) {
            const __nv_bfloat16* e = expert_outputs + ((int64_t)token * topk + k) * K;
            acc += bf16_to_float(e[i]) * tw[k];
        }
        o[i] = float_to_bf16(acc * scale);
    }
}

extern "C" void krasis_moe_sum_reduce(
    void* output, const void* expert_outputs, const void* topk_weights,
    int M, int K, int topk, float scale, void* stream)
{
    if (M == 0) return;
    int threads = min(1024, K);
    threads = ((threads + 31) / 32) * 32;
    moe_sum_reduce_kernel<<<M, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)output, (const __nv_bfloat16*)expert_outputs,
        (const float*)topk_weights, K, topk, scale);
}

/* ── MoE Align Block Size ──────────────────────────────────────────────── */

/* Sort tokens by expert assignment with block-size padding.
 * This is a CPU-side operation typically, but we implement on GPU for
 * the Rust prefill path (avoids D2H copy of topk_ids).
 *
 * Each block handles one expert. Outputs:
 *   sorted_token_ids: padded list of token indices sorted by expert
 *   expert_ids: which expert each block of tokens belongs to
 *   num_tokens_post_padded: total padded token count
 */
extern "C" __global__ void moe_align_block_size_kernel(
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ num_tokens_post_padded,
    int* __restrict__ expert_counts,     /* [E] scratch */
    int* __restrict__ expert_offsets,    /* [E+1] scratch */
    const int* __restrict__ topk_ids,   /* [M, topk] */
    int M,
    int topk,
    int block_size)
{
    /* Phase 1: count tokens per expert (single block, single pass) */
    int E = gridDim.x;  /* one block per expert initially, but use 1 block for counting */
    /* Actually this needs a 2-phase approach. Use a simple sequential kernel. */
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int total = M * topk;
        /* Clear counts */
        for (int e = 0; e < E; e++) expert_counts[e] = 0;
        /* Count */
        for (int i = 0; i < total; i++) {
            int eid = topk_ids[i];
            if (eid >= 0 && eid < E) expert_counts[eid]++;
        }
        /* Compute offsets with padding */
        int offset = 0;
        for (int e = 0; e < E; e++) {
            expert_offsets[e] = offset;
            int padded = ((expert_counts[e] + block_size - 1) / block_size) * block_size;
            offset += padded;
            /* Fill expert_ids for each block */
            int num_blocks_for_expert = padded / block_size;
            for (int b = 0; b < num_blocks_for_expert; b++) {
                expert_ids[expert_offsets[e] / block_size + b] = e;
            }
        }
        expert_offsets[E] = offset;
        *num_tokens_post_padded = offset;

        /* Scatter token indices */
        /* First, fill sorted_token_ids with padding value (M*topk = invalid) */
        for (int i = 0; i < offset; i++) sorted_token_ids[i] = total;
        /* Reset counts as write cursors */
        for (int e = 0; e < E; e++) expert_counts[e] = 0;
        /* Scatter */
        for (int i = 0; i < total; i++) {
            int eid = topk_ids[i];
            if (eid >= 0 && eid < E) {
                int pos = expert_offsets[eid] + expert_counts[eid];
                sorted_token_ids[pos] = i;
                expert_counts[eid]++;
            }
        }
    }
}

extern "C" void krasis_moe_align_block_size(
    void* sorted_token_ids, void* expert_ids, void* num_tokens_post_padded,
    const void* topk_ids,
    int M, int num_experts, int topk, int block_size,
    void* stream)
{
    /* We need scratch space for expert_counts [E] and expert_offsets [E+1].
     * For simplicity, run on 1 block 1 thread (it's fast for small E).
     * TODO: optimize for large E with parallel counting. */
    /* The scratch pointers are after the main outputs.
     * Caller must allocate extra: (E + E + 1) * sizeof(int) after sorted_token_ids. */
    /* Actually, we'll use a different approach: caller provides scratch. */
    /* For now, simple single-thread kernel. E <= 512 typically. */

    /* Use global memory for scratch: allocate after sorted_token_ids.
     * Actually, let's just use a simple approach: the caller ensures the buffer
     * is large enough. We'll put scratch at sorted_token_ids + M*topk + padding. */

    /* Simple approach: use CUDA managed memory for scratch. Actually, just
     * allocate on the device. Let Rust handle this. For now, minimal kernel. */
    /* TODO: This needs scratch memory. For initial impl, use shared memory. */

    if (M == 0) return;
    /* E * 2 + 1 ints for scratch, E <= 1024, fits in shared memory */
    int smem = (num_experts * 2 + 1) * sizeof(int);
    /* Override: put scratch in shared memory instead of separate buffers */
    /* Actually the kernel above uses separate pointers. Let me simplify. */

    /* For initial implementation, run a single-thread sequential kernel.
     * This is fine because MoE routing is not the bottleneck (GEMM is). */
    /* Provide scratch via dynamic shared memory */

    /* Rewrite: use shared memory scratch */
    /* TODO: rewrite with shared memory. For now, this is a placeholder
     * that will be replaced by linking against sgl_kernel's moe_align_block_size. */
    (void)sorted_token_ids;
    (void)expert_ids;
    (void)num_tokens_post_padded;
    (void)topk_ids;
    (void)M;
    (void)num_experts;
    (void)topk;
    (void)block_size;
    (void)stream;
}

/* ── Memory helpers ────────────────────────────────────────────────────── */

extern "C" void krasis_zero_buffer(void* ptr, int64_t bytes, void* stream) {
    cudaMemsetAsync(ptr, 0, bytes, (cudaStream_t)stream);
}

extern "C" void krasis_memcpy_d2d(void* dst, const void* src, int64_t bytes, void* stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
}

extern "C" void krasis_memcpy_h2d(void* dst, const void* src, int64_t bytes, void* stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);
}

extern "C" void krasis_memcpy_d2h(void* dst, const void* src, int64_t bytes, void* stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
}

extern "C" void krasis_stream_sync(void* stream) {
    cudaStreamSynchronize((cudaStream_t)stream);
}

/* ── GQA Prefill Attention ─────────────────────────────────────────────── */

/* Basic tiled GQA attention for prefill. Not FlashAttention-optimized yet,
 * but correct and usable. One block per (query_head, tile) pair.
 *
 * q: [M, num_q_heads, head_dim] bf16
 * k: [M, num_kv_heads, head_dim] bf16 (before KV cache append)
 * v: [M, num_kv_heads, head_dim] bf16
 * out: [M, num_q_heads, head_dim] bf16
 *
 * Causal mask: position i can attend to positions 0..i+start_pos
 */
extern "C" __global__ void gqa_prefill_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float softmax_scale,
    int start_pos)  /* number of previous tokens in KV cache */
{
    /* blockIdx.x = query position, blockIdx.y = query head */
    int qi = blockIdx.x;  /* query token index */
    int qh = blockIdx.y;  /* query head index */
    int kv_h = qh / (num_q_heads / num_kv_heads);  /* corresponding KV head */

    /* Query vector for this position and head */
    const __nv_bfloat16* q_vec = q + ((int64_t)qi * num_q_heads + qh) * head_dim;

    /* Output vector */
    __nv_bfloat16* o_vec = out + ((int64_t)qi * num_q_heads + qh) * head_dim;

    /* Causal attention: attend to positions 0..qi (within this prefill batch) */
    int num_attend = qi + 1;  /* causal mask: can attend to 0..qi inclusive */

    /* Online softmax: maintain running max and sum */
    float max_score = -1e30f;
    float sum_exp = 0.0f;

    /* Accumulate output in fp32 */
    extern __shared__ float smem[];
    float* acc = smem;  /* [head_dim] */

    /* Initialize accumulator */
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        acc[d] = 0.0f;
    }
    __syncthreads();

    /* Iterate over KV positions */
    for (int ki = 0; ki < num_attend; ki++) {
        const __nv_bfloat16* k_vec = k + ((int64_t)ki * num_kv_heads + kv_h) * head_dim;
        const __nv_bfloat16* v_vec = v + ((int64_t)ki * num_kv_heads + kv_h) * head_dim;

        /* Compute Q·K dot product */
        float dot = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += bf16_to_float(q_vec[d]) * bf16_to_float(k_vec[d]);
        }
        /* Warp reduce the dot product */
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xffffffff, dot, offset);
        }
        /* Cross-warp reduce via shared memory */
        __shared__ float s_dots[32];  /* max 32 warps */
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        if (lane_id == 0) s_dots[warp_id] = dot;
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = 0.0f;
            int num_warps = (blockDim.x + 31) / 32;
            for (int w = 0; w < num_warps; w++) total += s_dots[w];
            s_dots[0] = total * softmax_scale;
        }
        __syncthreads();
        float score = s_dots[0];

        /* Online softmax update */
        float old_max = max_score;
        if (score > max_score) max_score = score;
        float rescale = __expf(old_max - max_score);
        float new_exp = __expf(score - max_score);
        sum_exp = sum_exp * rescale + new_exp;

        /* Update accumulator: rescale old values and add new contribution */
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            acc[d] = acc[d] * rescale + new_exp * bf16_to_float(v_vec[d]);
        }
        __syncthreads();
    }

    /* Write output = acc / sum_exp */
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        o_vec[d] = float_to_bf16(acc[d] * inv_sum);
    }
}

extern "C" void krasis_gqa_prefill(
    void* out, const void* q, const void* k, const void* v,
    void* kv_cache, const void* page_table,
    int M, int num_q_heads, int num_kv_heads, int head_dim,
    int page_size, int num_existing_tokens, float softmax_scale,
    int kv_dtype, void* stream)
{
    if (M == 0) return;
    /* This initial version does NOT use paged KV cache — it operates directly
     * on the Q, K, V tensors from the current prefill batch.
     * KV cache append is done separately via krasis_kv_cache_append. */
    (void)kv_cache;
    (void)page_table;
    (void)page_size;
    (void)kv_dtype;

    dim3 grid(M, num_q_heads);
    int threads = min(256, head_dim);
    threads = ((threads + 31) / 32) * 32;
    int smem = head_dim * sizeof(float);
    gqa_prefill_kernel<<<grid, threads, smem, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)q,
        (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
        M, num_q_heads, num_kv_heads, head_dim,
        softmax_scale, num_existing_tokens);
}

/* ── KV Cache Append ───────────────────────────────────────────────────── */

/* Append M tokens of K and V to FP8 E4M3 KV caches.
 * k_cache, v_cache: separate [max_seq, kv_stride] FP8 E4M3 buffers
 * k, v: [M, kv_stride] BF16 input from GEMM projections
 * kv_stride = num_kv_heads * head_dim
 * Converts BF16 -> FP8 E4M3 on write, matching decode's cache format.
 */
extern "C" __global__ void kv_cache_append_fp8_kernel(
    __nv_fp8_e4m3* __restrict__ k_cache,   /* [max_seq, kv_stride] */
    __nv_fp8_e4m3* __restrict__ v_cache,   /* [max_seq, kv_stride] */
    const __nv_bfloat16* __restrict__ k,   /* [M, kv_stride] */
    const __nv_bfloat16* __restrict__ v,   /* [M, kv_stride] */
    int M,
    int kv_stride,   /* num_kv_heads * head_dim */
    int max_seq,
    int start_pos)
{
    int ti = blockIdx.x;  /* token index 0..M-1 */
    int pos = start_pos + ti;
    if (pos >= max_seq) return;

    int64_t src_off = (int64_t)ti * kv_stride;
    int64_t dst_off = (int64_t)pos * kv_stride;

    for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
        k_cache[dst_off + d] = bf16_to_fp8e4m3(k[src_off + d]);
        v_cache[dst_off + d] = bf16_to_fp8e4m3(v[src_off + d]);
    }
}

/* PTX entry point — called from Rust via cuLaunchKernel.
 * Same signature as kv_cache_append_fp8_kernel, launched with grid=(M,1,1). */
extern "C" __global__ void kv_cache_append_kernel(
    __nv_fp8_e4m3* __restrict__ k_cache,
    __nv_fp8_e4m3* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos)
{
    int ti = blockIdx.x;
    int pos = start_pos + ti;
    if (pos >= max_seq) return;

    int64_t src_off = (int64_t)ti * kv_stride;
    int64_t dst_off = (int64_t)pos * kv_stride;

    for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
        k_cache[dst_off + d] = bf16_to_fp8e4m3(k[src_off + d]);
        v_cache[dst_off + d] = bf16_to_fp8e4m3(v[src_off + d]);
    }
}

extern "C" __global__ void kv_cache_append_bf16_kernel(
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos)
{
    int ti = blockIdx.x;
    int pos = start_pos + ti;
    if (pos >= max_seq) return;

    int64_t src_off = (int64_t)ti * kv_stride;
    int64_t dst_off = (int64_t)pos * kv_stride;

    for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
        k_cache[dst_off + d] = k[src_off + d];
        v_cache[dst_off + d] = v[src_off + d];
    }
}

/* ── FP8 KV Cache Dequant + Concat for Cross-Chunk FA2 ─────────────────
 * Dequantizes FP8 E4M3 KV cache [0..cache_len] to BF16, then copies
 * current chunk BF16 K/V [0..m] into [cache_len..cache_len+m].
 * Result: contiguous BF16 [cache_len+m, kv_stride] buffer for FA2.
 * Grid: (cache_len + m, 1, 1), Block: (threads, 1, 1)
 */
extern "C" __global__ void kv_cache_dequant_concat_kernel(
    __nv_bfloat16* __restrict__ out,            /* [cache_len+m, kv_stride] BF16 output */
    const __nv_fp8_e4m3* __restrict__ kv_cache, /* [max_seq, kv_stride] FP8 cache */
    const __nv_bfloat16* __restrict__ kv_new,   /* [m, kv_stride] BF16 current chunk */
    int cache_len,                               /* number of cached tokens */
    int m,                                       /* current chunk size */
    int kv_stride)                               /* num_kv_heads * head_dim */
{
    int ti = blockIdx.x;
    if (ti < cache_len) {
        /* Dequant FP8 -> BF16 from cache */
        int64_t off = (int64_t)ti * kv_stride;
        for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
            float val = float(kv_cache[off + d]);
            out[off + d] = __float2bfloat16(val);
        }
    } else {
        /* Copy BF16 from current chunk */
        int ci = ti - cache_len;
        if (ci < m) {
            int64_t src_off = (int64_t)ci * kv_stride;
            int64_t dst_off = (int64_t)ti * kv_stride;
            for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
                out[dst_off + d] = kv_new[src_off + d];
            }
        }
    }
}

extern "C" __global__ void kv_cache_concat_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ kv_cache,
    const __nv_bfloat16* __restrict__ kv_new,
    int cache_len,
    int m,
    int kv_stride)
{
    int ti = blockIdx.x;
    if (ti < cache_len) {
        int64_t off = (int64_t)ti * kv_stride;
        for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
            out[off + d] = kv_cache[off + d];
        }
    } else {
        int ci = ti - cache_len;
        if (ci < m) {
            int64_t src_off = (int64_t)ci * kv_stride;
            int64_t dst_off = (int64_t)ti * kv_stride;
            for (int d = threadIdx.x; d < kv_stride; d += blockDim.x) {
                out[dst_off + d] = kv_new[src_off + d];
            }
        }
    }
}

/* ── Mamba2 Strided Extraction ──────────────────────────────────────────
 * Extract x, B, C, dt from in_proj output [M, proj_dim] BF16 into separate
 * contiguous buffers. Replaces M per-token memcpy loops.
 *
 * in_proj layout per row: [z(d_inner) | x(d_inner) | B(bc) | C(bc) | dt(n_heads)]
 * where bc = n_groups * d_state, proj_dim = 2*d_inner + 2*bc + n_heads
 *
 * Grid: (M, 1, 1), Block: (threads, 1, 1)
 */
extern "C" __global__ void mamba2_extract_kernel(
    __nv_bfloat16* __restrict__ x_out,    /* [M, d_inner] */
    __nv_bfloat16* __restrict__ b_out,    /* [M, bc] */
    __nv_bfloat16* __restrict__ c_out,    /* [M, bc] */
    __nv_bfloat16* __restrict__ dt_out,   /* [M, n_heads] */
    const __nv_bfloat16* __restrict__ inp, /* [M, proj_dim] */
    int d_inner, int bc, int n_heads, int proj_dim)
{
    int t = blockIdx.x;
    const __nv_bfloat16* row = inp + (int64_t)t * proj_dim;

    /* x: offset d_inner, length d_inner */
    for (int d = threadIdx.x; d < d_inner; d += blockDim.x) {
        x_out[(int64_t)t * d_inner + d] = row[d_inner + d];
    }
    /* B: offset 2*d_inner, length bc */
    for (int d = threadIdx.x; d < bc; d += blockDim.x) {
        b_out[(int64_t)t * bc + d] = row[2 * d_inner + d];
    }
    /* C: offset 2*d_inner+bc, length bc */
    for (int d = threadIdx.x; d < bc; d += blockDim.x) {
        c_out[(int64_t)t * bc + d] = row[2 * d_inner + bc + d];
    }
    /* dt: offset 2*d_inner+2*bc, length n_heads */
    for (int d = threadIdx.x; d < n_heads; d += blockDim.x) {
        dt_out[(int64_t)t * n_heads + d] = row[2 * d_inner + 2 * bc + d];
    }
}

/* ── Mamba2 Causal Conv1d (prefill) ────────────────────────────────────── */

/* Simple causal conv1d for Mamba2 prefill.
 * x: [1, D, L] bf16 (batch=1 for inference)
 * weight: [D, width] bf16
 * bias: [D] bf16 or NULL
 * out: [1, D, L] bf16
 * conv_state: [1, D, width-1] bf16 (updated with final state)
 *
 * Each thread block handles one channel dimension.
 */
extern "C" __global__ void causal_conv1d_fwd_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ conv_state,
    int D,
    int L,
    int width,
    int silu_act)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    /* Load weight for this channel */
    float w[8];  /* width <= 8 for Mamba2 (typically 4) */
    for (int j = 0; j < width; j++) {
        w[j] = bf16_to_float(weight[d * width + j]);
    }
    float b = (bias != NULL) ? bf16_to_float(bias[d]) : 0.0f;

    /* Process sequence positions */
    const __nv_bfloat16* x_d = x + (int64_t)d * L;  /* [L] for this channel */
    __nv_bfloat16* out_d = out + (int64_t)d * L;

    for (int t = 0; t < L; t++) {
        float acc = b;
        for (int j = 0; j < width; j++) {
            int src_t = t - (width - 1) + j;
            float xv = 0.0f;
            if (src_t >= 0) {
                xv = bf16_to_float(x_d[src_t]);
            }
            /* else: zero padding (causal, no initial state for prefill) */
            acc += xv * w[j];
        }
        if (silu_act) {
            acc = acc / (1.0f + __expf(-acc));
        }
        out_d[t] = float_to_bf16(acc);
    }

    /* Save final conv state: last (width-1) values of x for this channel */
    __nv_bfloat16* cs = conv_state + (int64_t)d * (width - 1);
    for (int j = 0; j < width - 1; j++) {
        int src_t = L - (width - 1) + j;
        cs[j] = (src_t >= 0) ? x_d[src_t] : float_to_bf16(0.0f);
    }
}

extern "C" void krasis_causal_conv1d_fwd(
    void* out, const void* x, const void* weight, const void* bias,
    void* conv_state,
    int B, int D, int L, int width, int silu_activation,
    void* stream)
{
    if (D == 0 || L == 0) return;
    /* B is always 1 for inference. Process all channels in parallel. */
    int threads = 256;
    int blocks = (D + threads - 1) / threads;
    causal_conv1d_fwd_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)weight,
        (const __nv_bfloat16*)bias,
        (__nv_bfloat16*)conv_state,
        D, L, width, silu_activation);
}

/* ── Mamba2 SSD Forward ────────────────────────────────────────────────── */

/* Mamba2 SSD (Structured State-Space Duality) chunked scan.
 *
 * This implements the chunked SSD algorithm for prefill:
 * 1. Discretize: dt → A_bar = exp(A * softplus(dt + bias)), B_bar = B * softplus(dt + bias)
 * 2. Within each chunk of size C:
 *    - Compute chunk-level SSM: parallel O(C²) attention-like computation
 * 3. Between chunks: sequential state passing
 *
 * For initial correctness, we implement a simple sequential scan
 * (equivalent to running decode N times). Chunk-parallel version comes later.
 */
extern "C" __global__ void mamba2_ssd_sequential_kernel(
    __nv_bfloat16* __restrict__ out,        /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ x,    /* [L, n_heads, head_dim] */
    const __nv_bfloat16* __restrict__ dt_in,/* [L, n_heads] */
    const float* __restrict__ A,            /* [n_heads] */
    const __nv_bfloat16* __restrict__ B_mat,/* [L, n_groups, state_size] */
    const __nv_bfloat16* __restrict__ C_mat,/* [L, n_groups, state_size] */
    const float* __restrict__ D_vec,        /* [n_heads] or NULL */
    float* __restrict__ ssm_state,          /* [n_heads, head_dim, state_size] */
    const float* __restrict__ dt_bias,      /* [n_heads] or NULL */
    int L,
    int n_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int use_softplus)
{
    /* One block per (head, head_dim_idx) pair */
    int head = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;

    int group = head / (n_heads / n_groups);
    float A_val = A[head];
    float D_val = (D_vec != NULL) ? D_vec[head] : 0.0f;

    /* SSM state for this (head, d): [state_size] floats */
    float* h = ssm_state + ((int64_t)head * head_dim + d) * state_size;

    /* Sequential scan over time steps */
    for (int t = 0; t < L; t++) {
        /* Get dt, apply bias and softplus */
        float dt = bf16_to_float(dt_in[t * n_heads + head]);
        if (dt_bias != NULL) dt += dt_bias[head];
        if (use_softplus) dt = logf(1.0f + __expf(dt));

        /* Discretize: A_bar = exp(A * dt) */
        float A_bar = __expf(A_val * dt);

        /* Get x for this timestep */
        float x_val = bf16_to_float(x[(t * n_heads + head) * head_dim + d]);

        /* Compute output: y = sum_s(C[s] * h[s]) + D * x */
        float y = D_val * x_val;

        for (int s = 0; s < state_size; s++) {
            float B_val = bf16_to_float(B_mat[(t * n_groups + group) * state_size + s]);
            float C_val = bf16_to_float(C_mat[(t * n_groups + group) * state_size + s]);

            /* State update: h[s] = A_bar * h[s] + B_bar * x */
            float B_bar = B_val * dt;
            h[s] = A_bar * h[s] + B_bar * x_val;

            /* Output contribution */
            y += C_val * h[s];
        }

        out[(t * n_heads + head) * head_dim + d] = float_to_bf16(y);
    }
}

extern "C" void krasis_mamba2_ssd_fwd(
    void* out, const void* x, const void* dt,
    const void* A, const void* B_mat, const void* C_mat,
    const void* D_vec, void* ssm_state,
    int B_batch, int L, int n_heads, int head_dim, int state_size,
    int n_groups, int chunk_size,
    const void* dt_bias, float dt_softplus,
    void* stream)
{
    if (L == 0) return;
    (void)B_batch;  /* always 1 for inference */
    (void)chunk_size;  /* TODO: use chunked algorithm */

    /* SSM state must be in float32 for numerical stability */
    /* Caller provides float32 ssm_state: [n_heads, head_dim, state_size] */

    int threads = min(256, head_dim);
    threads = ((threads + 31) / 32) * 32;
    if (threads == 0) threads = 32;
    int blocks_d = (head_dim + threads - 1) / threads;
    dim3 grid(n_heads, blocks_d);

    mamba2_ssd_sequential_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)dt, (const float*)A,
        (const __nv_bfloat16*)B_mat, (const __nv_bfloat16*)C_mat,
        (const float*)D_vec, (float*)ssm_state,
        (const float*)dt_bias,
        L, n_heads, head_dim, state_size, n_groups,
        dt_softplus > 0.5f ? 1 : 0);
}

/* ── MoE Gather: collect tokens by expert ID ─────────────────────────── */

/*
 * Given topk_ids [M, topk] and hidden [M, D] (bf16), gather tokens into
 * per-expert contiguous batches.
 *
 * expert_offsets [E+1]: exclusive prefix sum of tokens per expert (host-computed).
 * expert_token_map [M*topk]: maps each (token,k) slot to its position in the
 *   gathered buffer. Computed on host: for each (t,k) where topk_ids[t,k]==e,
 *   expert_token_map[t*topk+k] = expert_offsets[e] + count_e++.
 *
 * gathered [total_active, D] (bf16): output.
 * gather_src_map [total_active]: maps each row in gathered to source token index.
 *
 * Grid: (total_active, 1, 1), Block: (min(1024, D_padded32), 1, 1)
 */
extern "C" __global__ void moe_gather_kernel(
    __nv_bfloat16* __restrict__ gathered,
    const __nv_bfloat16* __restrict__ hidden,
    const int* __restrict__ gather_src_map,
    int total_active,
    int D)
{
    int row = blockIdx.x;
    if (row >= total_active) return;
    int src_token = gather_src_map[row];
    const __nv_bfloat16* src = hidden + (int64_t)src_token * D;
    __nv_bfloat16* dst = gathered + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = src[i];
    }
}

/* ── MoE Scatter + Accumulate (FP32 accumulator) ────────────────────── */

/*
 * Scatter expert outputs back and accumulate with routing weights.
 *
 * expert_out [total_active, D] (bf16): expert GEMM outputs.
 * accum [M, D] (fp32): accumulator (zero-initialized before MoE).
 *   NOTE: accum is FP32 to allow safe atomicAdd when multiple experts
 *   map to the same token. Caller converts to BF16 after scatter completes.
 * gather_src_map [total_active]: source token index for each gathered row.
 * gather_weight_map [total_active]: routing weight (fp32) for each gathered row.
 *
 * Grid: (total_active, 1, 1), Block: (min(1024, D_padded32), 1, 1)
 */
extern "C" __global__ void moe_scatter_add_kernel(
    float* __restrict__ accum,
    const __nv_bfloat16* __restrict__ expert_out,
    const int* __restrict__ gather_src_map,
    const float* __restrict__ gather_weight_map,
    int total_active,
    int D)
{
    int row = blockIdx.x;
    if (row >= total_active) return;
    int dst_token = gather_src_map[row];
    float w = gather_weight_map[row];
    const __nv_bfloat16* src = expert_out + (int64_t)row * D;
    float* dst = accum + (int64_t)dst_token * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = bf16_to_float(src[i]) * w;
        atomicAdd(&dst[i], val);
    }
}

/* ── MoE Zero Accumulator ────────────────────────────────────────────── */

/* Zero an FP32 buffer. Grid: (M, 1, 1), Block: (min(1024, D_padded32), 1, 1) */
extern "C" __global__ void moe_zero_accum_kernel(
    float* __restrict__ buf,
    int M, int D)
{
    int row = blockIdx.x;
    if (row >= M) return;
    float* r = buf + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        r[i] = 0.0f;
    }
}

/* ── MoE Add Shared Expert ───────────────────────────────────────────── */

/* Add shared expert output (BF16) to FP32 MoE accumulator.
 * Grid: (M, 1, 1), Block: (min(1024, D_padded32), 1, 1) */
extern "C" __global__ void moe_add_shared_kernel(
    float* __restrict__ accum,
    const __nv_bfloat16* __restrict__ shared_out,
    int M, int D)
{
    int row = blockIdx.x;
    if (row >= M) return;
    float* a = accum + (int64_t)row * D;
    const __nv_bfloat16* s = shared_out + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        a[i] += bf16_to_float(s[i]);
    }
}

/* ── MoE Add Shared (gated) ──────────────────────────────────────────── */

/* Same as moe_add_shared, but applies sigmoid gating: accum += sigmoid(gate[row]) * shared_out[row]
 * gate_values is [M] FP32 (output of hidden @ gate_weight GEMM). */
extern "C" __global__ void moe_add_shared_gated_kernel(
    float* __restrict__ accum,
    const __nv_bfloat16* __restrict__ shared_out,
    const float* __restrict__ gate_values,
    int M, int D)
{
    int row = blockIdx.x;
    if (row >= M) return;
    float gate = 1.0f / (1.0f + __expf(-gate_values[row]));
    float* a = accum + (int64_t)row * D;
    const __nv_bfloat16* s = shared_out + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        a[i] += gate * bf16_to_float(s[i]);
    }
}

/* ── MoE FP32 Accum -> BF16 output ──────────────────────────────────── */

/* Convert FP32 accumulator to BF16 output.
 * Grid: (M, 1, 1), Block: (min(1024, D_padded32), 1, 1) */
extern "C" __global__ void moe_accum_to_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ accum,
    int M, int D)
{
    int row = blockIdx.x;
    if (row >= M) return;
    __nv_bfloat16* o = out + (int64_t)row * D;
    const float* a = accum + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        o[i] = float_to_bf16(a[i]);
    }
}

/* ── FP32 -> BF16 batch convert ──────────────────────────────────────── */

/* Convert FP32 buffer to BF16 (for cuBLAS output conversion).
 * Grid: (M, 1, 1), Block: (min(1024, N_padded32), 1, 1) */
extern "C" __global__ void fp32_to_bf16_batch_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ in,
    int M, int N)
{
    int row = blockIdx.x;
    if (row >= M) return;
    const float* src = in + (int64_t)row * N;
    __nv_bfloat16* dst = out + (int64_t)row * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        dst[i] = float_to_bf16(src[i]);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  LINEAR ATTENTION (Gated DeltaNet) PREFILL KERNELS
 *
 *  These kernels implement the batched (multi-token) linear attention
 *  prefill path used by QCN (36/48 layers).
 *
 *  Algorithm: Gated Delta Rule with chunked parallel formulation.
 *  Reference: linear_attention.py _forward_chunked()
 * ══════════════════════════════════════════════════════════════════════════ */

/* ── Uninterleave QKVZ (batched) ─────────────────────────────────────── */
/*
 * in_proj_qkvz output is interleaved per key-head group:
 *   [M, nk * group_dim]  where group_dim = 2*dk + 2*hr*dv
 * Within each key-head group:
 *   [dk (q), dk (k), hr*dv (v), hr*dv (z)]
 *
 * Output layout:
 *   q_out: [M, nk, dk]   (BF16)
 *   k_out: [M, nk, dk]   (BF16)
 *   v_out: [M, nv, dv]   (BF16)
 *   z_out: [M, nv, dv]   (BF16)
 *
 * Grid: (M, 1, 1), Block: (min(1024, nk*group_dim_padded), 1, 1)
 */
extern "C" __global__ void la_uninterleave_qkvz_kernel(
    __nv_bfloat16* __restrict__ q_out,    /* [M, nk*dk] */
    __nv_bfloat16* __restrict__ k_out,    /* [M, nk*dk] */
    __nv_bfloat16* __restrict__ v_out,    /* [M, nv*dv] */
    __nv_bfloat16* __restrict__ z_out,    /* [M, nv*dv] */
    const __nv_bfloat16* __restrict__ qkvz, /* [M, nk*group_dim] */
    int nk, int dk, int hr, int dv)
{
    int token = blockIdx.x;
    int group_dim = 2 * dk + 2 * hr * dv;
    int total = nk * group_dim;
    int key_dim = nk * dk;
    int nv = nk * hr;

    const __nv_bfloat16* src = qkvz + (int64_t)token * total;
    __nv_bfloat16* q_dst = q_out + (int64_t)token * key_dim;
    __nv_bfloat16* k_dst = k_out + (int64_t)token * key_dim;
    __nv_bfloat16* v_dst = v_out + (int64_t)token * (nv * dv);
    __nv_bfloat16* z_dst = z_out + (int64_t)token * (nv * dv);

    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int head_group = i / group_dim;
        int offset = i % group_dim;
        __nv_bfloat16 val = src[i];

        if (offset < dk) {
            /* q: first dk elements */
            q_dst[head_group * dk + offset] = val;
        } else if (offset < 2 * dk) {
            /* k: next dk elements */
            k_dst[head_group * dk + (offset - dk)] = val;
        } else if (offset < 2 * dk + hr * dv) {
            /* v: next hr*dv elements -> reshape to [nv, dv] */
            int v_offset = offset - 2 * dk;
            int v_sub_head = v_offset / dv;
            int v_elem = v_offset % dv;
            int v_head = head_group * hr + v_sub_head;
            v_dst[v_head * dv + v_elem] = val;
        } else {
            /* z: last hr*dv elements -> reshape to [nv, dv] */
            int z_offset = offset - 2 * dk - hr * dv;
            int z_sub_head = z_offset / dv;
            int z_elem = z_offset % dv;
            int z_head = head_group * hr + z_sub_head;
            z_dst[z_head * dv + z_elem] = val;
        }
    }
}

/* ── Uninterleave BA (batched) ───────────────────────────────────────── */
/*
 * in_proj_ba output: [M, nk * 2*hr] interleaved per key-head group:
 *   [hr (b), hr (a)] per key head
 *
 * Output:
 *   b_out: [M, nv] (BF16)
 *   a_out: [M, nv] (BF16)
 *
 * Grid: (M, 1, 1), Block: (min(1024, nk*2*hr_padded), 1, 1)
 */
extern "C" __global__ void la_uninterleave_ba_kernel(
    __nv_bfloat16* __restrict__ b_out,    /* [M, nv] */
    __nv_bfloat16* __restrict__ a_out,    /* [M, nv] */
    const __nv_bfloat16* __restrict__ ba, /* [M, nk * 2*hr] */
    int nk, int hr)
{
    int token = blockIdx.x;
    int ba_group = 2 * hr;
    int total = nk * ba_group;
    int nv = nk * hr;

    const __nv_bfloat16* src = ba + (int64_t)token * total;
    __nv_bfloat16* b_dst = b_out + (int64_t)token * nv;
    __nv_bfloat16* a_dst = a_out + (int64_t)token * nv;

    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int head_group = i / ba_group;
        int offset = i % ba_group;
        __nv_bfloat16 val = src[i];

        if (offset < hr) {
            b_dst[head_group * hr + offset] = val;
        } else {
            a_dst[head_group * hr + (offset - hr)] = val;
        }
    }
}

/* ── Depthwise Conv1d + SiLU (batched over full sequence) ────────────── */
/*
 * Causal depthwise conv1d with SiLU activation over a full sequence.
 * Input: [conv_dim, M] (concatenated q_flat, k_flat, v_flat per token)
 * Conv state: [conv_dim, kernel_dim] (left-padded context)
 * Weight: [conv_dim, kernel_dim] (per-channel conv weights, no bias)
 * Output: [conv_dim, M] after conv + SiLU
 *
 * Also updates conv_state to last kernel_dim columns of input.
 *
 * Grid: (conv_dim, 1, 1), Block: (min(1024, M_padded), 1, 1)
 */
extern "C" __global__ void la_depthwise_conv1d_silu_kernel(
    float* __restrict__ output,          /* [conv_dim, M] FP32 */
    float* __restrict__ conv_state,      /* [conv_dim, kernel_dim] FP32, updated */
    const __nv_bfloat16* __restrict__ input, /* [M, conv_dim] BF16, row-major */
    const float* __restrict__ weight,    /* [conv_dim, kernel_dim] FP32 */
    int M, int conv_dim, int kernel_dim)
{
    int ch = blockIdx.x;
    if (ch >= conv_dim) return;

    float* st = conv_state + (int64_t)ch * kernel_dim;
    const float* wt = weight + (int64_t)ch * kernel_dim;
    float* out = output + (int64_t)ch * M;

    /* Process each token position */
    for (int t = threadIdx.x; t < M; t += blockDim.x) {
        float acc = 0.0f;
        for (int w = 0; w < kernel_dim; w++) {
            /* Position in the padded sequence: state has kernel_dim columns,
               then input has M columns. We want position (t + w) in this
               concatenated view, reading from right to left for the filter. */
            int src_pos = t + w - (kernel_dim - 1);
            float val;
            if (src_pos < 0) {
                /* Read from conv state (left padding) */
                val = st[kernel_dim + src_pos];  /* src_pos is negative */
            } else {
                /* Read from input: input is [M, conv_dim] row-major,
                   so element at position src_pos, channel ch is input[src_pos * conv_dim + ch] */
                val = bf16_to_float(input[src_pos * conv_dim + ch]);
            }
            acc += val * wt[w];
        }
        /* SiLU activation: x * sigmoid(x) */
        float sig = 1.0f / (1.0f + __expf(-acc));
        out[t] = acc * sig;
    }

    /* Update conv state with last kernel_dim tokens.
       Wait for all threads to finish reading before overwriting state. */
    __syncthreads();
    for (int w = threadIdx.x; w < kernel_dim; w += blockDim.x) {
        int src_pos = M - kernel_dim + w;
        if (src_pos < 0) {
            /* Still from old state */
            st[w] = st[kernel_dim + src_pos];
        } else {
            st[w] = bf16_to_float(input[src_pos * conv_dim + ch]);
        }
    }
}

/* ── L2 Norm per head (batched) ──────────────────────────────────────── */
/*
 * L2-normalize each head vector and optionally scale.
 * x: [M, num_heads, dim] FP32, in-place
 * Scale is applied after normalization: out = x / ||x|| * scale
 *
 * Grid: (M, num_heads, 1), Block: (min(256, dim_padded32), 1, 1)
 */
extern "C" __global__ void la_l2norm_per_head_kernel(
    float* __restrict__ x,     /* [M, num_heads, dim] in-place */
    float scale,
    int num_heads, int dim)
{
    int token = blockIdx.x;
    int head = blockIdx.y;
    float* vec = x + ((int64_t)token * num_heads + head) * dim;

    extern __shared__ float smem[];

    /* Compute sum of squares */
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = vec[i];
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float inv_norm = rsqrtf(smem[0] + 1e-6f);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        vec[i] = vec[i] * inv_norm * scale;
    }
}

/* ── Compute gate and beta (batched) ─────────────────────────────────── */
/*
 * beta = sigmoid(b)
 * gate = -exp(A_log) * softplus(a + dt_bias)
 *
 * b_in: [M, nv] BF16 (from uninterleave)
 * a_in: [M, nv] BF16 (from uninterleave)
 * A_log: [nv] FP32 (model parameter)
 * dt_bias: [nv] FP32 (model parameter)
 *
 * beta_out: [M, nv] FP32
 * gate_out: [M, nv] FP32
 *
 * Grid: (M, 1, 1), Block: (min(1024, nv_padded32), 1, 1)
 */
extern "C" __global__ void la_compute_gate_beta_kernel(
    float* __restrict__ beta_out,   /* [M, nv] FP32 */
    float* __restrict__ gate_out,   /* [M, nv] FP32 */
    const __nv_bfloat16* __restrict__ b_in,  /* [M, nv] BF16 */
    const __nv_bfloat16* __restrict__ a_in,  /* [M, nv] BF16 */
    const float* __restrict__ A_log,         /* [nv] FP32 */
    const float* __restrict__ dt_bias,       /* [nv] FP32 */
    int nv)
{
    int token = blockIdx.x;
    const __nv_bfloat16* b_row = b_in + (int64_t)token * nv;
    const __nv_bfloat16* a_row = a_in + (int64_t)token * nv;
    float* beta_row = beta_out + (int64_t)token * nv;
    float* gate_row = gate_out + (int64_t)token * nv;

    for (int i = threadIdx.x; i < nv; i += blockDim.x) {
        float b = bf16_to_float(b_row[i]);
        float a = bf16_to_float(a_row[i]);

        /* beta = sigmoid(b) */
        beta_row[i] = 1.0f / (1.0f + __expf(-b));

        /* gate = -exp(A_log) * softplus(a + dt_bias) */
        float a_val = __expf(A_log[i]);
        float x_sp = a + dt_bias[i];
        float sp = (x_sp > 20.0f) ? x_sp : logf(1.0f + __expf(x_sp));
        gate_row[i] = -a_val * sp;
    }
}

/* ── Repeat-interleave heads (batched) ───────────────────────────────── */
/*
 * Expand nk key heads to nv value heads (each key head repeated hr times).
 * input: [M, nk, dim] FP32
 * output: [M, nv, dim] FP32 where nv = nk * hr
 *
 * Grid: (M, nv, 1), Block: (min(256, dim_padded32), 1, 1)
 */
extern "C" __global__ void la_repeat_interleave_kernel(
    float* __restrict__ output,      /* [M, nv, dim] */
    const float* __restrict__ input, /* [M, nk, dim] */
    int nk, int dim, int hr)
{
    int token = blockIdx.x;
    int v_head = blockIdx.y;
    int k_head = v_head / hr;

    const float* src = input + ((int64_t)token * nk + k_head) * dim;
    float* dst = output + ((int64_t)token * (nk * hr) + v_head) * dim;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

/* ── BF16 to FP32 batched (for conv1d input transpose) ──────────────── */
/*
 * Convert [M, D] BF16 to [M, D] FP32
 * Grid: (M, 1, 1), Block: (min(1024, D_padded), 1, 1)
 */
extern "C" __global__ void la_bf16_to_fp32_kernel(
    float* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int D)
{
    int row = blockIdx.x;
    const __nv_bfloat16* src = in + (int64_t)row * D;
    float* dst = out + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = bf16_to_float(src[i]);
    }
}

/* ── FP32 to BF16 batched ───────────────────────────────────────────── */
extern "C" __global__ void la_fp32_to_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ in,
    int D)
{
    int row = blockIdx.x;
    const float* src = in + (int64_t)row * D;
    __nv_bfloat16* dst = out + (int64_t)row * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = float_to_bf16(src[i]);
    }
}

/* ── Gated RMSNorm (batched) ─────────────────────────────────────────── */
/*
 * Gated RMSNorm: out = rmsnorm(x) * weight * silu(gate)
 *
 * x: [M, nv, dv] FP32 (attention output)
 * gate: [M, nv, dv] BF16 (z from uninterleave)
 * weight: [dv] BF16 (per-head norm weight, shared across heads)
 * out: [M, nv*dv] BF16
 *
 * Grid: (M, nv, 1), Block: (min(256, dv_padded32), 1, 1)
 */
extern "C" __global__ void la_gated_rmsnorm_kernel(
    __nv_bfloat16* __restrict__ out,      /* [M, nv*dv] BF16 */
    const float* __restrict__ x,          /* [M, nv, dv] FP32 */
    const __nv_bfloat16* __restrict__ gate, /* [M, nv, dv] BF16 */
    const float* __restrict__ weight,     /* [dv] FP32 */
    int nv, int dv, float eps)
{
    int token = blockIdx.x;
    int head = blockIdx.y;
    const float* x_head = x + ((int64_t)token * nv + head) * dv;
    const __nv_bfloat16* g_head = gate + ((int64_t)token * nv + head) * dv;
    __nv_bfloat16* o_head = out + ((int64_t)token * nv + head) * dv;

    extern __shared__ float smem[];

    /* Compute variance */
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        float v = x_head[i];
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / (float)dv + eps);

    /* Normalize, scale by weight, multiply by silu(gate) */
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        float normed = x_head[i] * rms_inv * weight[i];
        float g = bf16_to_float(g_head[i]);
        float silu_g = g / (1.0f + __expf(-g));
        o_head[i] = float_to_bf16(normed * silu_g);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Optimized BF16 LA pipeline — fused kernels for FLA path
 * ═══════════════════════════════════════════════════════════════════════
 *
 * These replace the old 4-kernel conv pipeline (concat + conv + transpose + split)
 * and the FP32 intermediate stages with an all-BF16 path.
 * Used only when FLA is available. Non-FLA fallback uses the old FP32 path.
 */

/* ── Fused Conv1d+SiLU (BF16 in, BF16 out) ──────────────────────────── */
/*
 * Replaces: concat_3_bf16 + la_depthwise_conv1d_silu + transpose + la_split_conv_output
 *
 * Reads q,k,v as separate BF16 inputs (no concat step).
 * Applies depthwise conv1d + SiLU per channel.
 * Writes q,k,v as separate BF16 outputs (no transpose/split step).
 *
 * Each thread processes channels in a coalesced pattern (adjacent threads
 * read adjacent channels within same token = adjacent memory addresses).
 *
 * Grid: (M, 1, 1), Block: (min(1024, conv_dim_pad32), 1, 1)
 */
extern "C" __global__ void la_fused_conv1d_silu_bf16_kernel(
    __nv_bfloat16* __restrict__ q_out,       /* [M, key_dim] BF16 */
    __nv_bfloat16* __restrict__ k_out,       /* [M, key_dim] BF16 */
    __nv_bfloat16* __restrict__ v_out,       /* [M, value_dim] BF16 */
    const float* __restrict__ conv_state,    /* [conv_dim, kernel_dim] FP32 */
    const __nv_bfloat16* __restrict__ q_in,  /* [M, key_dim] BF16 */
    const __nv_bfloat16* __restrict__ k_in,  /* [M, key_dim] BF16 */
    const __nv_bfloat16* __restrict__ v_in,  /* [M, value_dim] BF16 */
    const float* __restrict__ weight,        /* [conv_dim, kernel_dim] FP32 */
    int M, int key_dim, int value_dim, int kernel_dim)
{
    int token = blockIdx.x;
    int conv_dim = 2 * key_dim + value_dim;

    for (int ch = threadIdx.x; ch < conv_dim; ch += blockDim.x) {
        const float* wt = weight + (int64_t)ch * kernel_dim;

        float acc = 0.0f;
        for (int w = 0; w < kernel_dim; w++) {
            int src_pos = token + w - (kernel_dim - 1);
            float val;
            if (src_pos < 0) {
                /* Read from conv state (left padding) */
                val = conv_state[(int64_t)ch * kernel_dim + (kernel_dim + src_pos)];
            } else {
                /* Read from appropriate BF16 input buffer */
                if (ch < key_dim) {
                    val = bf16_to_float(q_in[(int64_t)src_pos * key_dim + ch]);
                } else if (ch < 2 * key_dim) {
                    val = bf16_to_float(k_in[(int64_t)src_pos * key_dim + (ch - key_dim)]);
                } else {
                    val = bf16_to_float(v_in[(int64_t)src_pos * value_dim + (ch - 2 * key_dim)]);
                }
            }
            acc += val * wt[w];
        }

        /* SiLU activation */
        float sig = 1.0f / (1.0f + __expf(-acc));
        float result = acc * sig;

        /* Write to appropriate BF16 output buffer */
        if (ch < key_dim) {
            q_out[(int64_t)token * key_dim + ch] = float_to_bf16(result);
        } else if (ch < 2 * key_dim) {
            k_out[(int64_t)token * key_dim + (ch - key_dim)] = float_to_bf16(result);
        } else {
            v_out[(int64_t)token * value_dim + (ch - 2 * key_dim)] = float_to_bf16(result);
        }
    }
}

extern "C" void krasis_la_fused_conv1d_silu_bf16(
    void* q_out, void* k_out, void* v_out,
    const void* conv_state,
    const void* q_in, const void* k_in, const void* v_in,
    const void* weight,
    int M, int key_dim, int value_dim, int kernel_dim, void* stream)
{
    if (M == 0) return;
    int conv_dim = 2 * key_dim + value_dim;
    int threads = min(1024, ((conv_dim + 31) / 32) * 32);
    la_fused_conv1d_silu_bf16_kernel<<<M, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)q_out, (__nv_bfloat16*)k_out, (__nv_bfloat16*)v_out,
        (const float*)conv_state,
        (const __nv_bfloat16*)q_in, (const __nv_bfloat16*)k_in, (const __nv_bfloat16*)v_in,
        (const float*)weight, M, key_dim, value_dim, kernel_dim);
}

/* ── Update conv state after fused conv ──────────────────────────────── */
/*
 * Copies the last kernel_dim token positions into conv_state.
 * Tiny kernel (conv_dim * kernel_dim = ~32K values).
 *
 * Grid: (conv_dim, 1, 1), Block: (kernel_dim, 1, 1)
 */
extern "C" __global__ void la_update_conv_state_kernel(
    float* __restrict__ conv_state,          /* [conv_dim, kernel_dim] FP32 */
    const __nv_bfloat16* __restrict__ q_in,  /* [M, key_dim] BF16 */
    const __nv_bfloat16* __restrict__ k_in,  /* [M, key_dim] BF16 */
    const __nv_bfloat16* __restrict__ v_in,  /* [M, value_dim] BF16 */
    int M, int key_dim, int value_dim, int kernel_dim)
{
    int ch = blockIdx.x;
    int conv_dim = 2 * key_dim + value_dim;
    if (ch >= conv_dim) return;

    float* st = conv_state + (int64_t)ch * kernel_dim;

    for (int w = threadIdx.x; w < kernel_dim; w += blockDim.x) {
        int src_pos = M - kernel_dim + w;
        if (src_pos < 0) {
            /* Still from old state */
            st[w] = st[kernel_dim + src_pos];
        } else {
            if (ch < key_dim) {
                st[w] = bf16_to_float(q_in[(int64_t)src_pos * key_dim + ch]);
            } else if (ch < 2 * key_dim) {
                st[w] = bf16_to_float(k_in[(int64_t)src_pos * key_dim + (ch - key_dim)]);
            } else {
                st[w] = bf16_to_float(v_in[(int64_t)src_pos * value_dim + (ch - 2 * key_dim)]);
            }
        }
    }
}

extern "C" void krasis_la_update_conv_state(
    void* conv_state,
    const void* q_in, const void* k_in, const void* v_in,
    int M, int key_dim, int value_dim, int kernel_dim, void* stream)
{
    if (M == 0) return;
    int conv_dim = 2 * key_dim + value_dim;
    int threads = min(32, ((kernel_dim + 31) / 32) * 32);
    la_update_conv_state_kernel<<<conv_dim, threads, 0, (cudaStream_t)stream>>>(
        (float*)conv_state,
        (const __nv_bfloat16*)q_in, (const __nv_bfloat16*)k_in, (const __nv_bfloat16*)v_in,
        M, key_dim, value_dim, kernel_dim);
}

/* ── Gate/Beta computation with BF16 output ──────────────────────────── */
/*
 * Same as la_compute_gate_beta but outputs BF16 for direct FLA consumption.
 * Eliminates 2 FP32->BF16 conversion kernels.
 */
extern "C" __global__ void la_compute_gate_beta_bf16_kernel(
    __nv_bfloat16* __restrict__ beta_out,   /* [M, nv] BF16 */
    __nv_bfloat16* __restrict__ gate_out,   /* [M, nv] BF16 */
    const __nv_bfloat16* __restrict__ b_in, /* [M, nv] BF16 */
    const __nv_bfloat16* __restrict__ a_in, /* [M, nv] BF16 */
    const float* __restrict__ A_log,        /* [nv] FP32 */
    const float* __restrict__ dt_bias,      /* [nv] FP32 */
    int nv)
{
    int token = blockIdx.x;
    const __nv_bfloat16* b_row = b_in + (int64_t)token * nv;
    const __nv_bfloat16* a_row = a_in + (int64_t)token * nv;
    __nv_bfloat16* beta_row = beta_out + (int64_t)token * nv;
    __nv_bfloat16* gate_row = gate_out + (int64_t)token * nv;

    for (int i = threadIdx.x; i < nv; i += blockDim.x) {
        float b = bf16_to_float(b_row[i]);
        float a = bf16_to_float(a_row[i]);
        /* beta = sigmoid(b) */
        beta_row[i] = float_to_bf16(1.0f / (1.0f + __expf(-b)));
        /* gate = -exp(A_log) * softplus(a + dt_bias)
         * Match the FP32 path's stable softplus to avoid BF16 fast-path overflow.
         */
        float x_sp = a + dt_bias[i];
        float sp = (x_sp > 20.0f) ? x_sp : logf(1.0f + __expf(x_sp));
        gate_row[i] = float_to_bf16(-__expf(A_log[i]) * sp);
    }
}

/* ── Fused Repeat-Interleave + L2 Norm (BF16) ───────────────────────── */
/*
 * Combines repeat_interleave (nk -> nv heads) with L2 normalization.
 * One pass: read from [M, nk, dk], normalize in FP32, write to [M, nv, dk] BF16.
 * Eliminates separate repeat_interleave + l2norm + their intermediate buffers.
 *
 * Grid: (M, nv, 1), Block: (min(256, dk_pad32), 1, 1)
 */
extern "C" __global__ void la_fused_repeat_l2norm_bf16_kernel(
    __nv_bfloat16* __restrict__ output,     /* [M, nv, dk] BF16 */
    const __nv_bfloat16* __restrict__ input, /* [M, nk, dk] BF16 */
    int nk, int dk, int hr, float scale)
{
    int token = blockIdx.x;
    int v_head = blockIdx.y;
    int k_head = v_head / hr;

    const __nv_bfloat16* src = input + ((int64_t)token * nk + k_head) * dk;
    __nv_bfloat16* dst = output + ((int64_t)token * (nk * hr) + v_head) * dk;

    extern __shared__ float smem[];

    /* Load values and compute sum of squares */
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        float v = bf16_to_float(src[i]);
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    /* Tree reduction */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float inv_norm = rsqrtf(smem[0] + 1e-6f);

    /* Normalize, scale, and write BF16 */
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        float v = bf16_to_float(src[i]);
        dst[i] = float_to_bf16(v * inv_norm * scale);
    }
}

/* ── Gated RMSNorm with BF16 input ──────────────────────────────────── */
/*
 * Same as la_gated_rmsnorm but reads x as BF16 (FLA output) instead of FP32.
 * Eliminates the BF16->FP32 conversion after FLA.
 *
 * x: [M, nv, dv] BF16 (FLA output)
 * gate: [M, nv, dv] BF16 (z from uninterleave)
 * weight: [dv] FP32
 * out: [M, nv*dv] BF16
 */
extern "C" __global__ void la_gated_rmsnorm_bf16in_kernel(
    __nv_bfloat16* __restrict__ out,          /* [M, nv*dv] BF16 */
    const __nv_bfloat16* __restrict__ x,      /* [M, nv, dv] BF16 */
    const __nv_bfloat16* __restrict__ gate,   /* [M, nv, dv] BF16 */
    const float* __restrict__ weight,         /* [dv] FP32 */
    int nv, int dv, float eps)
{
    int token = blockIdx.x;
    int head = blockIdx.y;
    const __nv_bfloat16* x_head = x + ((int64_t)token * nv + head) * dv;
    const __nv_bfloat16* g_head = gate + ((int64_t)token * nv + head) * dv;
    __nv_bfloat16* o_head = out + ((int64_t)token * nv + head) * dv;

    extern __shared__ float smem[];

    /* Compute variance in FP32 */
    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        float v = bf16_to_float(x_head[i]);
        local_ss += v * v;
    }
    smem[threadIdx.x] = local_ss;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(smem[0] / (float)dv + eps);

    /* Normalize, scale by weight, multiply by silu(gate) */
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        float normed = bf16_to_float(x_head[i]) * rms_inv * weight[i];
        float g = bf16_to_float(g_head[i]);
        float silu_g = g / (1.0f + __expf(-g));
        o_head[i] = float_to_bf16(normed * silu_g);
    }
}

/* ── Chunked delta rule: cumsum of g ─────────────────────────────────── */
/*
 * Compute cumulative sum within each chunk.
 * g: [nv, num_chunks, chunk_size] FP32
 * g_cum: [nv, num_chunks, chunk_size] FP32 (output)
 *
 * Grid: (nv, num_chunks, 1), Block: (1, 1, 1)
 * (Sequential within chunk since chunk_size=64 is small)
 */
extern "C" __global__ void la_cumsum_kernel(
    float* __restrict__ g_cum,
    const float* __restrict__ g,
    int num_chunks, int chunk_size)
{
    int head = blockIdx.x;
    int chunk = blockIdx.y;
    int offset = (head * num_chunks + chunk) * chunk_size;

    float sum = 0.0f;
    for (int i = 0; i < chunk_size; i++) {
        sum += g[offset + i];
        g_cum[offset + i] = sum;
    }
}

/* ── Build decay mask and intra-chunk attention ──────────────────────── */
/*
 * For each chunk, compute:
 *   decay_mask[i,j] = exp(g_cum[i] - g_cum[j]) for i >= j, else 0
 *   attn[i,j] = -(k_beta @ k^T)[i,j] * decay_mask[i,j] for i > j, else 0
 *
 * k_beta, k: [nv, num_chunks, chunk_size, dk] FP32
 * g_cum: [nv, num_chunks, chunk_size] FP32
 *
 * attn: [nv, num_chunks, chunk_size, chunk_size] FP32 (strictly lower tri)
 *
 * Grid: (nv, num_chunks, 1), Block: (chunk_size, 1, 1)
 * Each thread handles one row of the chunk_size x chunk_size attn matrix.
 */
extern "C" __global__ void la_build_attn_matrix_kernel(
    float* __restrict__ attn,      /* [nv, num_chunks, CS, CS] */
    const float* __restrict__ k_beta, /* [nv, num_chunks, CS, dk] */
    const float* __restrict__ k,      /* [nv, num_chunks, CS, dk] */
    const float* __restrict__ g_cum,  /* [nv, num_chunks, CS] */
    int num_chunks, int chunk_size, int dk)
{
    int head = blockIdx.x;
    int chunk = blockIdx.y;
    int row = threadIdx.x;
    if (row >= chunk_size) return;

    int cs = chunk_size;
    int base_g = (head * num_chunks + chunk) * cs;
    int base_kbeta = ((head * num_chunks + chunk) * cs + row) * dk;
    int base_attn = (head * num_chunks + chunk) * cs * cs + row * cs;

    float g_i = g_cum[base_g + row];

    /* Compute attn[row, col] for col < row (strictly lower triangular) */
    for (int col = 0; col < cs; col++) {
        if (col >= row) {
            attn[base_attn + col] = 0.0f;
        } else {
            /* k_beta[row] @ k[col] -- dot product */
            float dot = 0.0f;
            int base_k_col = ((head * num_chunks + chunk) * cs + col) * dk;
            for (int d = 0; d < dk; d++) {
                dot += k_beta[base_kbeta + d] * k[base_k_col + d];
            }
            float g_j = g_cum[base_g + col];
            float decay = __expf(g_i - g_j);
            attn[base_attn + col] = -dot * decay;
        }
    }
}

/* ── Triangular solve: (I - A)x = b ─────────────────────────────────── */
/*
 * Forward substitution for unitriangular lower system.
 * A: [nv, num_chunks, CS, CS] (strictly lower triangular)
 * b: [nv, num_chunks, CS, dim] (RHS, overwritten with solution)
 *
 * Solves: (I - A)x = b => x = b + A*x (forward sub since A is strictly lower)
 *
 * Grid: (nv, num_chunks, 1), Block: (min(256, dim), 1, 1)
 * Sequential over rows within each chunk (CS=64 is small).
 */
extern "C" __global__ void la_triangular_solve_kernel(
    float* __restrict__ x,      /* [nv, num_chunks, CS, dim] in/out (starts as b) */
    const float* __restrict__ A, /* [nv, num_chunks, CS, CS] strictly lower tri */
    int num_chunks, int chunk_size, int dim)
{
    int head = blockIdx.x;
    int chunk = blockIdx.y;
    int cs = chunk_size;

    /* base offsets */
    int64_t x_base = ((int64_t)(head * num_chunks + chunk)) * cs * dim;
    int64_t a_base = ((int64_t)(head * num_chunks + chunk)) * cs * cs;

    /* Forward substitution: for row i, x[i] += sum_j(A[i,j] * x[j]) for j < i */
    for (int i = 1; i < cs; i++) {
        /* Each thread handles a subset of the dim dimension */
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < i; j++) {
                sum += A[a_base + i * cs + j] * x[x_base + j * dim + d];
            }
            x[x_base + i * dim + d] += sum;
        }
        __syncthreads();  /* Ensure row i is complete before row i+1 reads it */
    }
}

/* ── Chunk recurrence step ───────────────────────────────────────────── */
/*
 * Single chunk step of the recurrent delta rule.
 *
 * For chunk i:
 *   attn_intra = (q @ k^T) * decay_mask, masked upper 0
 *   v_prime = k_cumd @ state
 *   v_new = v_corrected - v_prime
 *   attn_inter = (q * exp(g)) @ state
 *   output = attn_inter + attn_intra @ v_new
 *   g_last_exp = exp(g_last)
 *   k_decay = exp(g_last - g) per row
 *   state = state * g_last_exp + (k * k_decay)^T @ v_new
 *
 * This is called sequentially for each chunk; state carries forward.
 *
 * q, k: [nv, CS, dk]
 * v_corrected: [nv, CS, dv] (from triangular solve)
 * k_cumd: [nv, CS, dk] (from triangular solve)
 * g_cum: [nv, CS] (cumulative gate values)
 * decay_mask: [nv, CS, CS] (precomputed from build_attn_matrix step)
 * state: [nv, dk, dv] (in/out)
 * output: [nv, CS, dv]
 *
 * Grid: (nv, 1, 1), Block: (min(256, dk), 1, 1)
 *
 * NOTE: This kernel is sequential over CS positions per head.
 * For correctness, it processes one head per block.
 */
extern "C" __global__ void la_chunk_recurrence_kernel(
    float* __restrict__ output,        /* [nv, CS, dv] */
    float* __restrict__ state,         /* [nv, dk, dv] in/out */
    const float* __restrict__ q,       /* [nv, CS, dk] */
    const float* __restrict__ k,       /* [nv, CS, dk] */
    const float* __restrict__ v_corr,  /* [nv, CS, dv] (value_corrected) */
    const float* __restrict__ k_cumd,  /* [nv, CS, dk] */
    const float* __restrict__ g_cum,   /* [nv, CS] */
    const float* __restrict__ attn,    /* [nv, CS, CS] (decay_mask for intra) */
    int chunk_size, int dk, int dv)
{
    int head = blockIdx.x;
    int cs = chunk_size;

    float* st = state + (int64_t)head * dk * dv;
    const float* q_h = q + (int64_t)head * cs * dk;
    const float* k_h = k + (int64_t)head * cs * dk;
    const float* vc_h = v_corr + (int64_t)head * cs * dv;
    const float* kc_h = k_cumd + (int64_t)head * cs * dk;
    const float* g_h = g_cum + (int64_t)head * cs;
    const float* attn_h = attn + (int64_t)head * cs * cs;
    float* out_h = output + (int64_t)head * cs * dv;

    /* We need shared memory for v_new[CS * dv] - too large for 64*128=8192 floats.
       Instead, compute row by row. */

    /* Process each position in the chunk */
    for (int t = 0; t < cs; t++) {
        /* 1. v_prime[dv] = k_cumd[t] @ state  (dk x dk,dv -> dv) */
        /* 2. v_new[dv] = v_corrected[t] - v_prime */
        /* 3. attn_inter[dv] = (q[t] * exp(g[t])) @ state  (dk x dk,dv -> dv) */

        float g_t = g_h[t];

        /* For each output dimension d in dv: */
        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            /* Compute v_prime[d] = sum_j k_cumd[t,j] * state[j,d] */
            float v_prime_d = 0.0f;
            for (int j = 0; j < dk; j++) {
                v_prime_d += kc_h[t * dk + j] * st[j * dv + d];
            }
            float v_new_d = vc_h[t * dv + d] - v_prime_d;

            /* Compute attn_inter[d] = sum_j (q[t,j] * exp(g_t)) * state[j,d] */
            float attn_inter_d = 0.0f;
            float exp_g = __expf(g_t);
            for (int j = 0; j < dk; j++) {
                attn_inter_d += q_h[t * dk + j] * exp_g * st[j * dv + d];
            }

            /* Compute attn_intra @ v_new for position t:
               sum over s<t of attn[t,s] * v_new[s,d]
               But we only have v_new for position t, not for earlier positions.
               We need to compute v_new for ALL positions first... */

            /* Actually, this needs to be restructured. The chunk step function
               in Python computes:
                 v_prime = k_cumd @ state  (all CS positions at once)
                 v_new = v_corrected - v_prime  (all CS positions)
                 attn_intra = (q @ k^T) * decay_mask  (CS x CS)
                 attn_intra masked upper = 0
                 output = (q * exp(g)) @ state + attn_intra @ v_new
                 state update uses g_last and k_decay

               The intra-chunk attention is a GEMM (CS x CS) @ (CS x dv).
               The inter-chunk attention is a GEMM (CS x dk) @ (dk x dv).
               We need to compute these as matrix operations, not row-by-row.

               For correctness in a single kernel, we need to pre-compute
               v_new for all positions, then do the matrix products. */

            /* Store partial result for now - v_new[t, d] */
            out_h[t * dv + d] = v_new_d;
        }
        __syncthreads();
    }

    /* Now out_h contains v_new[CS, dv].
       Compute the full output:
       attn_inter[CS, dv] = (q * exp(g)) @ state   -- (CS,dk) @ (dk,dv)
       attn_intra[CS, CS] = (q @ k^T) * decay_mask -- but we already have decay_mask as 'attn'
       Wait, the 'attn' parameter is the original build_attn_matrix output, which was the
       nilpotent correction matrix. The intra-chunk attention for the chunk step is different.
       The chunk step uses:
         attn_intra = (q_i @ k_i^T) * decay_mask_i  (not the nilpotent A matrix)

       This is getting complex. Let me restructure this as two separate kernels:
       1. Compute v_new = v_corrected - k_cumd @ state  (GEMM + subtract)
       2. Compute attn_intra = (q @ k^T) * decay, output = q*exp(g)@state + attn_intra@v_new
       3. State update

       For now, let's use cuBLAS for the GEMMs from Rust and only use CUDA kernels
       for element-wise ops. This matches the Python approach better.
    */

    /* Simpler approach: just compute v_new and output element-by-element.
       This is O(CS * dk * dv) per head which is 64 * 128 * 128 = ~1M FLOPs.
       With 32 heads, that's 32M FLOPs - tiny for a GPU. */

    /* Recompute properly: */
    /* First, build v_new[CS, dv] (already done above, stored in out_h) */
    /* Build intra attention: q @ k^T * decay (CS x CS) */

    /* We'll use shared memory for the CS x CS attention matrix */
    extern __shared__ float shared[];
    /* shared[0..CS*CS-1] = intra attention matrix */
    float* s_attn = shared;

    /* Build q@k^T * decay_mask (recompute decay from g_cum) */
    if (threadIdx.x == 0) {
        for (int i = 0; i < cs; i++) {
            for (int j = 0; j < cs; j++) {
                if (j >= i) {
                    s_attn[i * cs + j] = 0.0f;
                } else {
                    float dot = 0.0f;
                    for (int dd = 0; dd < dk; dd++) {
                        dot += q_h[i * dk + dd] * k_h[j * dk + dd];
                    }
                    float decay = __expf(g_h[i] - g_h[j]);
                    s_attn[i * cs + j] = dot * decay;
                }
            }
        }
    }
    __syncthreads();

    /* Compute inter + intra attention output */
    for (int t = 0; t < cs; t++) {
        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            /* attn_inter = (q[t] * exp(g[t])) @ state */
            float inter = 0.0f;
            float exp_g = __expf(g_h[t]);
            for (int j = 0; j < dk; j++) {
                inter += q_h[t * dk + j] * exp_g * st[j * dv + d];
            }

            /* attn_intra @ v_new = sum_s attn[t,s] * v_new[s,d] */
            float intra = 0.0f;
            for (int s = 0; s < t; s++) {
                intra += s_attn[t * cs + s] * out_h[s * dv + d];
            }

            out_h[t * dv + d] = inter + intra;
        }
        __syncthreads();
    }

    /* State update: state = state * exp(g_last) + (k * k_decay)^T @ v_new
       where v_new was stored in out_h before we overwrote it...
       We need to save v_new first. */

    /* This kernel is getting too complex. Let's split into multiple kernels
       and orchestrate from Rust. See la_chunk_* kernels below. */
}

/* ── Simpler chunked kernels (orchestrated from Rust) ─────────────────── */

/* Compute v_new = v_corrected - k_cumd @ state for all positions in one chunk.
 * k_cumd: [CS, dk], state: [dk, dv], v_corrected: [CS, dv]
 * v_new: [CS, dv] output
 * One block per head.
 * Grid: (nv, 1, 1), Block: (min(256, dv), 1, 1)
 */
extern "C" __global__ void la_compute_v_new_kernel(
    float* __restrict__ v_new,         /* [nv, CS, dv] */
    const float* __restrict__ v_corr,  /* [nv, CS, dv] */
    const float* __restrict__ k_cumd,  /* [nv, CS, dk] */
    const float* __restrict__ state,   /* [nv, dk, dv] */
    int nv, int chunk_size, int dk, int dv)
{
    int head = blockIdx.x;
    if (head >= nv) return;
    int cs = chunk_size;

    const float* vc = v_corr + (int64_t)head * cs * dv;
    const float* kc = k_cumd + (int64_t)head * cs * dk;
    const float* st = state + (int64_t)head * dk * dv;
    float* vn = v_new + (int64_t)head * cs * dv;

    for (int t = 0; t < cs; t++) {
        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            float v_prime = 0.0f;
            for (int j = 0; j < dk; j++) {
                v_prime += kc[t * dk + j] * st[j * dv + d];
            }
            vn[t * dv + d] = vc[t * dv + d] - v_prime;
        }
    }
}

/* Compute output for one chunk:
 *   attn_inter = (q * exp(g)) @ state
 *   attn_intra = tril((q @ k^T) * decay, -1) @ v_new
 *   output = attn_inter + attn_intra
 *
 * Grid: (nv, 1, 1), Block: (min(256, dv), 1, 1)
 */
extern "C" __global__ void la_chunk_output_kernel(
    float* __restrict__ output,       /* [nv, CS, dv] */
    const float* __restrict__ q,      /* [nv, CS, dk] */
    const float* __restrict__ k,      /* [nv, CS, dk] */
    const float* __restrict__ v_new,  /* [nv, CS, dv] */
    const float* __restrict__ g_cum,  /* [nv, CS] */
    const float* __restrict__ state,  /* [nv, dk, dv] */
    int nv, int chunk_size, int dk, int dv)
{
    int head = blockIdx.x;
    if (head >= nv) return;
    int cs = chunk_size;

    const float* q_h = q + (int64_t)head * cs * dk;
    const float* k_h = k + (int64_t)head * cs * dk;
    const float* vn_h = v_new + (int64_t)head * cs * dv;
    const float* g_h = g_cum + (int64_t)head * cs;
    const float* st = state + (int64_t)head * dk * dv;
    float* out = output + (int64_t)head * cs * dv;

    for (int t = 0; t < cs; t++) {
        float exp_g = __expf(g_h[t]);

        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            /* Inter-chunk: (q[t] * exp(g[t])) @ state[:, d] */
            float inter = 0.0f;
            for (int j = 0; j < dk; j++) {
                inter += q_h[t * dk + j] * st[j * dv + d];
            }
            inter *= exp_g;

            /* Intra-chunk: sum_{s<=t} [(q[t] @ k[s]) * decay(t,s)] * v_new[s, d] */
            float intra = 0.0f;
            for (int s = 0; s <= t; s++) {
                float qk_dot = 0.0f;
                for (int j = 0; j < dk; j++) {
                    qk_dot += q_h[t * dk + j] * k_h[s * dk + j];
                }
                float decay = __expf(g_h[t] - g_h[s]);
                intra += qk_dot * decay * vn_h[s * dv + d];
            }

            out[t * dv + d] = inter + intra;
        }
    }
}

/* Update recurrent state after one chunk:
 *   state = state * exp(g_last) + (k * k_decay)^T @ v_new
 *   where k_decay[t] = exp(g_last - g_cum[t])
 *
 * Grid: (nv, 1, 1), Block: (min(256, dv), 1, 1)
 */
extern "C" __global__ void la_state_update_kernel(
    float* __restrict__ state,        /* [nv, dk, dv] in/out */
    const float* __restrict__ k,      /* [nv, CS, dk] */
    const float* __restrict__ v_new,  /* [nv, CS, dv] */
    const float* __restrict__ g_cum,  /* [nv, CS] */
    int nv, int chunk_size, int dk, int dv)
{
    int head = blockIdx.x;
    if (head >= nv) return;
    int cs = chunk_size;

    float* st = state + (int64_t)head * dk * dv;
    const float* k_h = k + (int64_t)head * cs * dk;
    const float* vn_h = v_new + (int64_t)head * cs * dv;
    const float* g_h = g_cum + (int64_t)head * cs;

    float g_last = g_h[cs - 1];
    float g_last_exp = __expf(g_last);

    /* state[j, d] = state[j, d] * g_last_exp + sum_t k[t, j] * k_decay[t] * v_new[t, d] */
    for (int j = 0; j < dk; j++) {
        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            float s = st[j * dv + d] * g_last_exp;
            for (int t = 0; t < cs; t++) {
                float k_decay = __expf(g_last - g_h[t]);
                s += k_h[t * dk + j] * k_decay * vn_h[t * dv + d];
            }
            st[j * dv + d] = s;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Concat 3 BF16 arrays row-wise: [M,a_dim]+[M,b_dim]+[M,c_dim] -> [M,total]
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Replaces per-token memcpy loops for conv1d input preparation.
 * Grid: (M, 1, 1), Block: (min(256, a_dim+b_dim+c_dim), 1, 1)
 */
extern "C" __global__ void concat_3_bf16_kernel(
    __nv_bfloat16* __restrict__ out,       /* [M, a_dim+b_dim+c_dim] */
    const __nv_bfloat16* __restrict__ a,   /* [M, a_dim] */
    const __nv_bfloat16* __restrict__ b,   /* [M, b_dim] */
    const __nv_bfloat16* __restrict__ c,   /* [M, c_dim] */
    int a_dim, int b_dim, int c_dim)
{
    int token = blockIdx.x;
    int total = a_dim + b_dim + c_dim;
    __nv_bfloat16* dst = out + (int64_t)token * total;
    const __nv_bfloat16* a_src = a + (int64_t)token * a_dim;
    const __nv_bfloat16* b_src = b + (int64_t)token * b_dim;
    const __nv_bfloat16* c_src = c + (int64_t)token * c_dim;

    for (int d = threadIdx.x; d < total; d += blockDim.x) {
        if (d < a_dim) {
            dst[d] = a_src[d];
        } else if (d < a_dim + b_dim) {
            dst[d] = b_src[d - a_dim];
        } else {
            dst[d] = c_src[d - a_dim - b_dim];
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * Strided chunk kernels — read directly from [nv, total_len, dim] arrays
 * ═══════════════════════════════════════════════════════════════════════
 *
 * These replace the per-head memcpy storms in the chunk recurrence loop.
 * Instead of copying chunk data to contiguous buffers, these kernels
 * compute offsets from total_len stride and chunk_idx directly.
 */

/* Strided v_new computation:
 *   v_new[nv, CS, dv] = v_corr[strided] - k_cumd[strided] @ state[nv, dk, dv]
 *
 * v_corr: [nv, total_len, dv] FP32 (strided)
 * k_cumd: [nv, total_len, dk] FP32 (strided)
 * state:  [nv, dk, dv] FP32 (contiguous)
 * v_new:  [nv, CS, dv] FP32 (contiguous output)
 *
 * Grid: (nv, 1, 1), Block: (min(256, dv), 1, 1)
 */
extern "C" __global__ void la_compute_v_new_strided_kernel(
    float* __restrict__ v_new,         /* [nv, CS, dv] contiguous output */
    const float* __restrict__ v_corr,  /* [nv, total_len, dv] strided */
    const float* __restrict__ k_cumd,  /* [nv, total_len, dk] strided */
    const float* __restrict__ state,   /* [nv, dk, dv] contiguous */
    int chunk_size, int dk, int dv,
    int total_len, int chunk_idx)
{
    int head = blockIdx.x;
    int cs = chunk_size;

    /* Strided offsets: head * total_len * dim + chunk_idx * CS * dim */
    const float* vc = v_corr + ((int64_t)head * total_len + chunk_idx * cs) * dv;
    const float* kc = k_cumd + ((int64_t)head * total_len + chunk_idx * cs) * dk;
    const float* st = state + (int64_t)head * dk * dv;
    float* vn = v_new + (int64_t)head * cs * dv;

    for (int t = 0; t < cs; t++) {
        /* Strided access: vc[t * dv] reads from total_len-strided array
         * but within a chunk, positions are contiguous (t * dv stride is fine
         * because the chunk is a contiguous slice of the total_len dimension). */
        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            float v_prime = 0.0f;
            for (int j = 0; j < dk; j++) {
                v_prime += kc[t * dk + j] * st[j * dv + d];
            }
            vn[t * dv + d] = vc[t * dv + d] - v_prime;
        }
    }
}

/* Strided chunk output:
 *   output[strided] = (q * exp(g)) @ state + tril(q @ k^T * decay) @ v_new
 *
 * Reads q, k, g_cum from strided [nv, total_len, dim] arrays.
 * Reads v_new from contiguous [nv, CS, dv] buffer.
 * Writes output directly to strided [nv, total_len, dv] buffer.
 *
 * Grid: (nv, 1, 1), Block: (min(256, dv), 1, 1)
 */
extern "C" __global__ void la_chunk_output_strided_kernel(
    float* __restrict__ output,        /* [nv, total_len, dv] strided */
    const float* __restrict__ q,       /* [nv, total_len, dk] strided */
    const float* __restrict__ k,       /* [nv, total_len, dk] strided */
    const float* __restrict__ v_new,   /* [nv, CS, dv] contiguous */
    const float* __restrict__ g_cum,   /* [nv, total_len] strided */
    const float* __restrict__ state,   /* [nv, dk, dv] contiguous */
    int chunk_size, int dk, int dv,
    int total_len, int chunk_idx)
{
    int head = blockIdx.x;
    int cs = chunk_size;

    const float* q_h = q + ((int64_t)head * total_len + chunk_idx * cs) * dk;
    const float* k_h = k + ((int64_t)head * total_len + chunk_idx * cs) * dk;
    const float* vn_h = v_new + (int64_t)head * cs * dv;
    const float* g_h = g_cum + (int64_t)head * total_len + chunk_idx * cs;
    const float* st = state + (int64_t)head * dk * dv;
    float* out = output + ((int64_t)head * total_len + chunk_idx * cs) * dv;

    for (int t = 0; t < cs; t++) {
        float exp_g = __expf(g_h[t]);

        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            /* Inter-chunk: (q[t] * exp(g[t])) @ state[:, d] */
            float inter = 0.0f;
            for (int j = 0; j < dk; j++) {
                inter += q_h[t * dk + j] * st[j * dv + d];
            }
            inter *= exp_g;

            /* Intra-chunk: sum_{s<=t} [(q[t] @ k[s]) * decay(t,s)] * v_new[s, d] */
            float intra = 0.0f;
            for (int s = 0; s <= t; s++) {
                float qk_dot = 0.0f;
                for (int j = 0; j < dk; j++) {
                    qk_dot += q_h[t * dk + j] * k_h[s * dk + j];
                }
                float decay = __expf(g_h[t] - g_h[s]);
                intra += qk_dot * decay * vn_h[s * dv + d];
            }

            out[t * dv + d] = inter + intra;
        }
    }
}

/* Strided state update:
 *   state = state * exp(g_last) + (k * k_decay)^T @ v_new
 *
 * Reads k, g_cum from strided [nv, total_len, dim] arrays.
 * Reads v_new from contiguous [nv, CS, dv] buffer.
 * Updates state [nv, dk, dv] in-place.
 *
 * Grid: (nv, 1, 1), Block: (min(256, dv), 1, 1)
 */
extern "C" __global__ void la_state_update_strided_kernel(
    float* __restrict__ state,        /* [nv, dk, dv] in/out */
    const float* __restrict__ k,      /* [nv, total_len, dk] strided */
    const float* __restrict__ v_new,  /* [nv, CS, dv] contiguous */
    const float* __restrict__ g_cum,  /* [nv, total_len] strided */
    int chunk_size, int dk, int dv,
    int total_len, int chunk_idx)
{
    int head = blockIdx.x;
    int cs = chunk_size;

    float* st = state + (int64_t)head * dk * dv;
    const float* k_h = k + ((int64_t)head * total_len + chunk_idx * cs) * dk;
    const float* vn_h = v_new + (int64_t)head * cs * dv;
    const float* g_h = g_cum + (int64_t)head * total_len + chunk_idx * cs;

    float g_last = g_h[cs - 1];
    float g_last_exp = __expf(g_last);

    for (int j = 0; j < dk; j++) {
        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            float s = st[j * dv + d] * g_last_exp;
            for (int t = 0; t < cs; t++) {
                float k_decay = __expf(g_last - g_h[t]);
                s += k_h[t * dk + j] * k_decay * vn_h[t * dv + d];
            }
            st[j * dv + d] = s;
        }
    }
}


/* ── Multiply k_beta by exp(g_cum) ───────────────────────────────────── */
/*
 * Prepares k_beta_g = k_beta * exp(g_cum) for the second triangular solve.
 * k_beta: [nv, total_len, dk] FP32
 * g_cum: [nv, total_len] FP32 (per-chunk cumulative)
 * k_beta_g: [nv, total_len, dk] FP32 output
 *
 * Grid: (nv, total_len, 1), Block: (min(256, dk), 1, 1)
 */
extern "C" __global__ void la_scale_by_exp_g_kernel(
    float* __restrict__ k_beta_g,
    const float* __restrict__ k_beta,
    const float* __restrict__ g_cum,
    int nv, int total_len, int dk)
{
    int head = blockIdx.x;
    int pos = blockIdx.y;
    if (head >= nv || pos >= total_len) return;

    float eg = __expf(g_cum[head * total_len + pos]);
    const float* src = k_beta + ((int64_t)head * total_len + pos) * dk;
    float* dst = k_beta_g + ((int64_t)head * total_len + pos) * dk;

    for (int d = threadIdx.x; d < dk; d += blockDim.x) {
        dst[d] = src[d] * eg;
    }
}

/* ── FP32 2D Transpose ───────────────────────────────────────────────── */
/*
 * Transpose a [rows, cols] FP32 matrix to [cols, rows].
 * Grid: (cols, 1, 1), Block: (min(1024, rows_padded), 1, 1)
 */
extern "C" __global__ void la_transpose_f32_kernel(
    float* __restrict__ out,        /* [cols, rows] */
    const float* __restrict__ in,   /* [rows, cols] */
    int rows, int cols)
{
    int col = blockIdx.x;
    if (col >= cols) return;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        out[col * rows + row] = in[row * cols + col];
    }
}

/* ── Prepare beta-scaled k and v ─────────────────────────────────────── */
/*
 * v_beta = v * beta,  k_beta = k * beta
 * v: [M, nv, dv] FP32, k: [M, nv, dk] FP32, beta: [M, nv] FP32
 * Grid: (M, nv, 1), Block: (min(256, max(dk,dv)), 1, 1)
 */
extern "C" __global__ void la_apply_beta_kernel(
    float* __restrict__ v_beta,  /* [M, nv, dv] */
    float* __restrict__ k_beta,  /* [nv, M, dk] */
    const float* __restrict__ v, /* [M, nv, dv] */
    const float* __restrict__ k, /* [nv, M, dk] */
    const float* __restrict__ beta, /* [M, nv] */
    int nv, int dk, int dv, int total_len)
{
    int token = blockIdx.x;
    int head = blockIdx.y;
    float b = beta[token * nv + head];

    /* v_beta */
    const float* v_src = v + ((int64_t)token * nv + head) * dv;
    float* v_dst = v_beta + ((int64_t)token * nv + head) * dv;
    for (int d = threadIdx.x; d < dv; d += blockDim.x) {
        v_dst[d] = v_src[d] * b;
    }

    /* k_beta */
    const float* k_src = k + ((int64_t)head * total_len + token) * dk;
    float* k_dst = k_beta + ((int64_t)head * total_len + token) * dk;
    for (int d = threadIdx.x; d < dk; d += blockDim.x) {
        k_dst[d] = k_src[d] * b;
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * 3D Transpose: [A, B, C] -> [B, A, C]
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Single-launch replacement for per-element memcpy storms.
 * Grid: (B, A, 1), Block: (min(256, C), 1, 1)
 * Each thread block copies one (b, a) row of C elements.
 */
extern "C" __global__ void transpose_3d_021_kernel(
    float* __restrict__ out,    /* [B, A, C] */
    const float* __restrict__ in, /* [A, B, C] */
    int A, int B, int C)
{
    int b = blockIdx.x;
    int a = blockIdx.y;
    if (a >= A || b >= B) return;

    const float* src = in  + ((int64_t)a * B + b) * C;
    float*       dst = out + ((int64_t)b * A + a) * C;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        dst[c] = src[c];
    }
}

/* BF16 version */
extern "C" __global__ void transpose_3d_021_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int A, int B, int C)
{
    int b = blockIdx.x;
    int a = blockIdx.y;
    if (a >= A || b >= B) return;

    const __nv_bfloat16* src = in  + ((int64_t)a * B + b) * C;
    __nv_bfloat16*       dst = out + ((int64_t)b * A + a) * C;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        dst[c] = src[c];
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * Gated Q Split: [M, H, 2*D] -> Q[M, H*D] + gate[M, H*D]
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Used by QCN's gated GQA attention where q_proj outputs interleaved [query, gate].
 * Grid: (M, H, 1), Block: (min(256, D), 1, 1)
 */
extern "C" __global__ void gated_q_split_kernel(
    __nv_bfloat16* __restrict__ q_out,    /* [M, H*D] */
    __nv_bfloat16* __restrict__ gate_out, /* [M, H*D] */
    const __nv_bfloat16* __restrict__ qg, /* [M, H, 2*D] */
    int H, int D)
{
    int token = blockIdx.x;
    int head  = blockIdx.y;

    const __nv_bfloat16* src = qg + ((int64_t)token * H + head) * (2 * D);
    __nv_bfloat16* q_dst = q_out  + ((int64_t)token * H + head) * D;
    __nv_bfloat16* g_dst = gate_out + ((int64_t)token * H + head) * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        q_dst[d] = src[d];
        g_dst[d] = src[D + d];
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * LA Conv Output Split: [M, conv_dim] -> q[M, key_dim] + k[M, key_dim] + v[M, value_dim]
 * ═══════════════════════════════════════════════════════════════════════
 *
 * conv_dim = 2 * key_dim + value_dim
 * Grid: (M, 1, 1), Block: (min(256, conv_dim), 1, 1)
 */
extern "C" __global__ void la_split_conv_output_kernel(
    float* __restrict__ q_out,     /* [M, key_dim] */
    float* __restrict__ k_out,     /* [M, key_dim] */
    float* __restrict__ v_out,     /* [M, value_dim] */
    const float* __restrict__ inp, /* [M, conv_dim] */
    int key_dim, int value_dim)
{
    int token = blockIdx.x;
    int conv_dim = 2 * key_dim + value_dim;

    const float* src = inp + (int64_t)token * conv_dim;
    float* q = q_out + (int64_t)token * key_dim;
    float* k = k_out + (int64_t)token * key_dim;
    float* v = v_out + (int64_t)token * value_dim;

    for (int d = threadIdx.x; d < conv_dim; d += blockDim.x) {
        float val = src[d];
        if (d < key_dim) {
            q[d] = val;
        } else if (d < 2 * key_dim) {
            k[d - key_dim] = val;
        } else {
            v[d - 2 * key_dim] = val;
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * Flash Attention with Tensor Core MMA — BF16/FP8 KV cache, Causal, GQA
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Uses WMMA (Warp Matrix Multiply-Accumulate) for Q*K^T and P*V products.
 * Supports cross-chunk attention: reads from FP8 KV cache for
 * positions [0, start_pos) and from BF16 GEMM output for the current
 * chunk [start_pos, start_pos + M).
 *
 * Design:
 *   BR = 16 queries per block (one wmma M=16 tile)
 *   BC = 64 KV positions per tile
 *   1 warp (32 threads) per block
 *   16x better K/V tile amortization vs old BR=4 kernel
 *
 * Q*K^T: [16, d] * [d, 64] via wmma m16n16k16, d/16 * 4 = 32 MMA calls
 * P*V:   [16, 64] * [64, d] via wmma m16n16k16, 4 * d/16 = 32 MMA calls
 * Online softmax between phases, accumulator O in registers.
 *
 * Shared memory (head_dim=128):
 *   s_q:     [16, 128] bf16 = 4 KB  — query tile, loaded once
 *   s_k:     [64, 128] bf16 = 16 KB — K tile (reused for zero-pad)
 *   s_v:     [64, 128] bf16 = 16 KB — V tile
 *   s_scores [16, 64]  f32  = 4 KB  — score tile for softmax
 *   s_p:     [16, 64]  bf16 = 2 KB  — P (softmax output) for P*V
 *   s_o_tmp: [16, 16]  f32  = 1 KB  — partial O from each P*V fragment
 *   Total: ~43 KB (fits in default 48 KB limit)
 *
 * Grid: (ceil(M/16), num_q_heads, 1)
 * Block: (32, 1, 1) = 1 warp
 */
#include <mma.h>

#define FA_BC 64    /* KV tile size */
#define FA_BR 16    /* queries per block = wmma tile M */
#define FA_TILE 16  /* wmma tile dimension */
#define FA_DPT 8    /* max dims per thread = ceil(256/32) */

__device__ __forceinline__ float fp8e4m3_to_float(__nv_fp8_e4m3 x) {
    return float(x);
}

extern "C" __global__ void flash_attn_tiled_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,       /* [M, num_q_heads, head_dim] */
    const __nv_fp8_e4m3* __restrict__ k_cache, /* [max_seq, kv_stride] FP8 or null */
    const __nv_fp8_e4m3* __restrict__ v_cache, /* [max_seq, kv_stride] FP8 or null */
    const __nv_bfloat16* __restrict__ k_cur,   /* [M, kv_stride] BF16 current chunk */
    const __nv_bfloat16* __restrict__ v_cur,   /* [M, kv_stride] BF16 current chunk */
    int M, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int start_pos, int kv_stride)
{
    int q_base = blockIdx.x * FA_BR;
    int qh = blockIdx.y;
    int kv_h = qh / (num_q_heads / num_kv_heads);
    int lane = threadIdx.x;  /* 0..31 */

    /* ── Shared memory layout ── */
    extern __shared__ char smem_fa[];
    __nv_bfloat16* s_q = (__nv_bfloat16*)smem_fa;                    /* [BR, hd] */
    __nv_bfloat16* s_k = s_q + FA_BR * head_dim;                     /* [BC, hd] */
    __nv_bfloat16* s_v = s_k + FA_BC * head_dim;                     /* [BC, hd] */
    float* s_scores = (float*)(s_v + FA_BC * head_dim);               /* [BR, BC] */
    __nv_bfloat16* s_p = (__nv_bfloat16*)(s_scores + FA_BR * FA_BC);  /* [BR, BC] */
    float* s_o_tmp = (float*)(s_p + FA_BR * FA_BC);                    /* [TILE, TILE] */

    /* ── Load Q tile to shared memory (contiguous row-major) ── */
    int q_stride = num_q_heads * head_dim;
    for (int idx = lane; idx < FA_BR * head_dim; idx += 32) {
        int r = idx / head_dim;
        int d = idx % head_dim;
        int qi = q_base + r;
        s_q[r * head_dim + d] = (qi < M)
            ? q[(int64_t)qi * q_stride + qh * head_dim + d]
            : float_to_bf16(0.0f);
    }
    __syncthreads();

    /* ── Per-thread state ── */
    int dpt = head_dim / 32;  /* dims per thread (head_dim multiple of 32) */
    float row_max_arr[FA_BR], row_sum_arr[FA_BR];
    float O_reg[FA_BR * FA_DPT];  /* output accumulator: 16 rows * dpt dims */
    for (int r = 0; r < FA_BR; r++) {
        row_max_arr[r] = -1e30f;
        row_sum_arr[r] = 0.0f;
    }
    for (int i = 0; i < FA_BR * dpt; i++) O_reg[i] = 0.0f;

    /* Block-level causal bound */
    int block_last_qi = min(q_base + FA_BR - 1, M - 1);
    int block_max_kv = (q_base < M) ? (start_pos + block_last_qi + 1) : 0;

    /* ══ Main loop over KV tiles ══ */
    for (int kv_start = 0; kv_start < block_max_kv; kv_start += FA_BC) {
        int tile_end = min(kv_start + FA_BC, block_max_kv);
        int tile_size = tile_end - kv_start;

        /* ── Load K and V tile (cooperative, 32 threads) ── */
        for (int idx = lane; idx < FA_BC * head_dim; idx += 32) {
            int ki = idx / head_dim;
            int d = idx % head_dim;
            __nv_bfloat16 kval, vval;
            if (ki < tile_size) {
                int abs_pos = kv_start + ki;
                if (abs_pos < start_pos && k_cache != nullptr) {
                    int64_t off = (int64_t)abs_pos * kv_stride + kv_h * head_dim + d;
                    kval = float_to_bf16(fp8e4m3_to_float(k_cache[off]));
                    vval = float_to_bf16(fp8e4m3_to_float(v_cache[off]));
                } else {
                    int cp = abs_pos - start_pos;
                    if (cp >= 0 && cp < M) {
                        int64_t off = ((int64_t)cp * num_kv_heads + kv_h) * head_dim + d;
                        kval = k_cur[off];
                        vval = v_cur[off];
                    } else {
                        kval = float_to_bf16(0.0f);
                        vval = float_to_bf16(0.0f);
                    }
                }
            } else {
                kval = float_to_bf16(0.0f);
                vval = float_to_bf16(0.0f);
            }
            s_k[ki * head_dim + d] = kval;
            s_v[ki * head_dim + d] = vval;
        }
        __syncwarp();

        /* ── Phase 1: Q * K^T via WMMA → s_scores [BR, BC] ──
         * 4 output column tiles (BC/16), 8 k-dim iterations (hd/16 for hd=128)
         * = 32 wmma mma_sync calls total */
        for (int nj = 0; nj < FA_BC / FA_TILE; nj++) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> s_frag;
            nvcuda::wmma::fill_fragment(s_frag, 0.0f);

            for (int kk = 0; kk < head_dim / FA_TILE; kk++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::row_major> q_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::col_major> k_frag;

                /* Q[0:16, kk*16:(kk+1)*16] from s_q, stride=head_dim */
                nvcuda::wmma::load_matrix_sync(q_frag,
                    &s_q[kk * FA_TILE], head_dim);
                /* K[nj*16:(nj+1)*16, kk*16:(kk+1)*16] from s_k, col_major transposes */
                nvcuda::wmma::load_matrix_sync(k_frag,
                    &s_k[nj * FA_TILE * head_dim + kk * FA_TILE], head_dim);

                nvcuda::wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
            }

            /* Store score tile [16, 16] to s_scores at column offset nj*16 */
            nvcuda::wmma::store_matrix_sync(
                &s_scores[nj * FA_TILE], s_frag, FA_BC,
                nvcuda::wmma::mem_row_major);
        }
        __syncwarp();

        /* ── Phase 2: Online softmax on s_scores [BR, BC] ──
         * Each of 32 threads processes a subset of each row's columns.
         * Warp-level reductions for max and sum. */
        for (int r = 0; r < FA_BR; r++) {
            int qi = q_base + r;
            if (qi >= M) continue;
            int abs_qi_r = start_pos + qi;
            float* row = &s_scores[r * FA_BC];

            /* Scale scores + causal mask + local max */
            float local_max = -1e30f;
            for (int c = lane; c < FA_BC; c += 32) {
                int abs_kv = kv_start + c;
                if (c < tile_size && abs_kv <= abs_qi_r) {
                    row[c] *= softmax_scale;
                    local_max = fmaxf(local_max, row[c]);
                } else {
                    row[c] = -1e30f;
                }
            }
            for (int off = 16; off > 0; off >>= 1)
                local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));

            /* Update online softmax state */
            float old_max = row_max_arr[r];
            float new_max = fmaxf(old_max, local_max);
            float rescale = __expf(old_max - new_max);
            row_max_arr[r] = new_max;

            /* Compute P = exp(score - max), local sum */
            float local_sum = 0.0f;
            for (int c = lane; c < FA_BC; c += 32) {
                float p = __expf(row[c] - new_max);
                row[c] = p;
                local_sum += p;
            }
            for (int off = 16; off > 0; off >>= 1)
                local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);

            row_sum_arr[r] = row_sum_arr[r] * rescale + local_sum;

            /* Rescale existing O accumulator for this row */
            for (int j = 0; j < dpt; j++)
                O_reg[r * dpt + j] *= rescale;

            /* Convert P to BF16 for WMMA P*V */
            for (int c = lane; c < FA_BC; c += 32)
                s_p[r * FA_BC + c] = float_to_bf16(row[c]);
        }
        __syncwarp();

        /* ── Phase 3: O += P * V via WMMA ──
         * Process head_dim/16 output column tiles.
         * For each: 4 k-dim iterations (BC/16), then store to s_o_tmp
         * and accumulate to O_reg. */
        for (int nj = 0; nj < head_dim / FA_TILE; nj++) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> o_partial;
            nvcuda::wmma::fill_fragment(o_partial, 0.0f);

            for (int kk = 0; kk < FA_BC / FA_TILE; kk++) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::row_major> p_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                    __nv_bfloat16, nvcuda::wmma::row_major> v_frag;

                /* P[0:16, kk*16:(kk+1)*16], stride=BC */
                nvcuda::wmma::load_matrix_sync(p_frag,
                    &s_p[kk * FA_TILE], FA_BC);
                /* V[kk*16:(kk+1)*16, nj*16:(nj+1)*16], stride=head_dim */
                nvcuda::wmma::load_matrix_sync(v_frag,
                    &s_v[kk * FA_TILE * head_dim + nj * FA_TILE], head_dim);

                nvcuda::wmma::mma_sync(o_partial, p_frag, v_frag, o_partial);
            }

            /* Store [16, 16] partial output, accumulate to O_reg */
            nvcuda::wmma::store_matrix_sync(s_o_tmp, o_partial,
                FA_TILE, nvcuda::wmma::mem_row_major);
            __syncwarp();

            /* Map fragment columns to thread's O_reg:
             * Thread t owns dims: t, t+32, t+64, ...
             * Fragment nj covers columns [nj*16, nj*16+16).
             * Active threads: (nj*16)%32 .. (nj*16)%32+15 */
            int first_t = (nj * FA_TILE) % 32;
            int my_col = lane - first_t;
            if (my_col >= 0 && my_col < FA_TILE) {
                int d_global = nj * FA_TILE + my_col;
                int j = d_global / 32;
                for (int r = 0; r < FA_BR; r++)
                    O_reg[r * dpt + j] += s_o_tmp[r * FA_TILE + my_col];
            }
            __syncwarp();
        }
        __syncwarp();  /* before next tile load */
    }

    /* ── Write output ── */
    for (int r = 0; r < FA_BR; r++) {
        int qi = q_base + r;
        if (qi >= M) continue;
        float inv_sum = (row_sum_arr[r] > 0.0f) ? (1.0f / row_sum_arr[r]) : 0.0f;
        __nv_bfloat16* o_row = out + ((int64_t)qi * num_q_heads + qh) * head_dim;
        for (int j = 0; j < dpt; j++) {
            int d = lane + j * 32;
            if (d < head_dim)
                o_row[d] = float_to_bf16(O_reg[r * dpt + j] * inv_sum);
        }
    }
}


/* ══════════════════════════════════════════════════════════════════════════
 *  GPU-only MoE routing — replaces CPU round-trip for token binning
 *
 *  Two-phase approach:
 *  Phase 1: moe_count_experts_kernel
 *    Count how many tokens are assigned to each expert. Atomic increments.
 *    Grid: (M, 1, 1), each thread handles one token's topk assignments.
 *  Phase 2: moe_build_maps_kernel
 *    Build gather_src_map, gather_weight_map, and per-expert offsets.
 *    Grid: (M, 1, 1), each thread writes its topk entries.
 *
 *  This eliminates two stream syncs and CPU binning from forward_moe.
 * ══════════════════════════════════════════════════════════════════════════ */

/* Phase 1: count tokens per expert using atomicAdd */
extern "C" __global__ void moe_count_experts_kernel(
    int* __restrict__ expert_counts,  /* [E] output: count per expert */
    const int* __restrict__ topk_ids, /* [M, topk] */
    int M, int topk, int E
) {
    int t = blockIdx.x;
    if (t >= M) return;
    for (int k = 0; k < topk; k++) {
        int eid = topk_ids[t * topk + k];
        if (eid >= 0 && eid < E) {
            atomicAdd(&expert_counts[eid], 1);
        }
    }
}

/* Phase 2: prefix sum on expert_counts to get offsets, then build maps.
 * This runs on a single block since E is typically small (512).
 * Performs prefix sum in shared memory, then each token writes its entries. */
extern "C" __global__ void moe_prefix_sum_kernel(
    int* __restrict__ expert_offsets,  /* [E+1] output: exclusive prefix sum */
    const int* __restrict__ expert_counts, /* [E] */
    int E
) {
    /* Single-block prefix sum over E experts.
     * Use char smem and reinterpret to avoid type conflict with global extern __shared__ float smem[]. */
    extern __shared__ char smem_raw_ps[];
    int* smem_i = reinterpret_cast<int*>(smem_raw_ps);
    int tid = threadIdx.x;

    /* Load counts into shared memory */
    for (int i = tid; i < E; i += blockDim.x) {
        smem_i[i] = expert_counts[i];
    }
    __syncthreads();

    /* Sequential prefix sum (E is small, typically 512) */
    if (tid == 0) {
        expert_offsets[0] = 0;
        for (int i = 0; i < E; i++) {
            expert_offsets[i + 1] = expert_offsets[i] + smem_i[i];
        }
    }
}

/* Phase 3: build gather maps using atomic offsets.
 * scale_factor is applied to weights (e.g. routed_scaling_factor for MoE). */
extern "C" __global__ void moe_build_maps_kernel(
    int* __restrict__ gather_src_map,     /* [total_active] output */
    float* __restrict__ gather_weight_map, /* [total_active] output */
    int* __restrict__ write_offsets,       /* [E] atomically incremented write positions */
    const int* __restrict__ topk_ids,      /* [M, topk] */
    const float* __restrict__ topk_weights,/* [M, topk] */
    const int* __restrict__ expert_offsets, /* [E+1] base offsets */
    int M, int topk, int E,
    float scale_factor
) {
    int t = blockIdx.x;
    if (t >= M) return;
    for (int k = 0; k < topk; k++) {
        int eid = topk_ids[t * topk + k];
        if (eid >= 0 && eid < E) {
            int base = expert_offsets[eid];
            int slot = atomicAdd(&write_offsets[eid], 1);
            int pos = base + slot;
            gather_src_map[pos] = t;
            gather_weight_map[pos] = topk_weights[t * topk + k] * scale_factor;
        }
    }
}


/* ══════════════════════════════════════════════════════════════════════════
 *  Fused MoE support kernels — moe_align_block_size and gather/scatter
 *
 *  These kernels produce the sorted_token_ids, expert_ids, and
 *  num_tokens_post_padded arrays required by the fused MarlinDefault
 *  kernel from sgl_kernel. The fused kernel processes ALL experts in
 *  one launch — our Rust code assembles contiguous expert weight buffers
 *  and calls MarlinDefault directly via dlopen.
 * ══════════════════════════════════════════════════════════════════════════ */

/*
 * moe_align_block_size: given topk_ids [M, topk], produce:
 *   sorted_token_ids [total_padded]  — which token maps to each sorted position
 *   expert_ids       [num_blocks]    — which expert each block processes
 *   num_tokens_post  [1]             — total_padded
 *
 * Three phases (launched separately from Rust):
 *   Phase 1: count tokens per expert (reuses moe_count_experts_kernel)
 *   Phase 2: prefix sum with block_size padding (this kernel)
 *   Phase 3: scatter tokens into sorted positions + fill expert_ids (this kernel)
 */

/* Phase 2: Prefix sum with block_size padding.
 * Input:  expert_counts [E]
 * Output: expert_offsets [E+1] (padded prefix sum), num_tokens_post [1]
 * Grid: (1,1,1), Block: (threads,1,1) with threads >= E
 */
extern "C" __global__ void moe_padded_prefix_sum_kernel(
    int* __restrict__ expert_offsets,
    int* __restrict__ num_tokens_post,
    const int* __restrict__ expert_counts,
    int E, int block_size
) {
    extern __shared__ int ps_smem[];
    int tid = threadIdx.x;
    int val = (tid < E) ? expert_counts[tid] : 0;
    // Pad to block_size
    int padded = ((val + block_size - 1) / block_size) * block_size;
    ps_smem[tid] = padded;
    __syncthreads();

    // Simple serial prefix sum (E is small, <= 1024)
    if (tid == 0) {
        int running = 0;
        for (int i = 0; i < E; i++) {
            expert_offsets[i] = running;
            running += ps_smem[i];
        }
        expert_offsets[E] = running;
        num_tokens_post[0] = running;
    }
}

/* Phase 3: Scatter tokens into sorted positions.
 * Grid: (M, 1, 1), Block: (1, 1, 1)
 * Each thread handles one token, scatters its topk slots.
 */
extern "C" __global__ void moe_scatter_sorted_kernel(
    int* __restrict__ sorted_token_ids,
    int* __restrict__ write_offsets,       // [E] atomically incremented
    const int* __restrict__ topk_ids,      // [M, topk]
    const int* __restrict__ expert_offsets, // [E+1]
    int M, int topk, int E
) {
    int t = blockIdx.x;
    if (t >= M) return;
    for (int k = 0; k < topk; k++) {
        int eid = topk_ids[t * topk + k];
        if (eid >= 0 && eid < E) {
            int base = expert_offsets[eid];
            int slot = atomicAdd(&write_offsets[eid], 1);
            sorted_token_ids[base + slot] = t * topk + k;  // vLLM format: token*topk+slot
        }
    }
}

/* Phase 4: finalize expert_ids and padding after scatter completes.
 * Grid: (E, 1, 1), Block: (1, 1, 1)
 * Each block handles one expert after the scatter writes are visible on-stream.
 */
extern "C" __global__ void moe_finalize_sorted_kernel(
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids_out,
    const int* __restrict__ expert_offsets, // [E+1]
    const int* __restrict__ expert_counts,  // [E]
    int M, int topk, int E, int block_size
) {
    int e = blockIdx.x;
    if (e >= E) return;
    int base = expert_offsets[e];
    int count = expert_counts[e];
    int padded = expert_offsets[e + 1] - base;
    int num_blocks = padded / block_size;
    // Fill padding slots with M*topk (Marlin kernel checks >= prob_m * top_k)
    for (int p = count; p < padded; p++) {
        sorted_token_ids[base + p] = M * topk; // padding sentinel
    }
    // Fill expert_ids for each block
    int block_start = base / block_size;
    for (int b = 0; b < num_blocks; b++) {
        expert_ids_out[block_start + b] = e;
    }
}

/* Replicate hidden states for fused MoE: out[i] = src[i / topk] for i in [0, M*topk).
 * Creates [M*topk, dim] buffer where each token's hidden state appears topk times.
 * Required for the top_k=1 trick: avoids C_tmp collision in fp32_reduce.
 * Grid: (M * topk, 1, 1), Block: (threads, 1, 1)
 */
extern "C" __global__ void moe_replicate_hidden_kernel(
    __nv_bfloat16* __restrict__ out,       // [M*topk, dim]
    const __nv_bfloat16* __restrict__ src, // [M, dim]
    int dim, int M, int topk
) {
    int idx = blockIdx.x;  // 0 to M*topk-1
    int token = idx / topk;
    if (token >= M) return;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out[(int64_t)idx * dim + i] = src[(int64_t)token * dim + i];
    }
}

/* Gather tokens by sorted order for fused MoE input.
 * sorted_token_ids maps positions in the sorted sequence to original token indices.
 * Grid: (total_padded, 1, 1), Block: (threads, 1, 1)
 */
extern "C" __global__ void moe_gather_sorted_kernel(
    __nv_bfloat16* __restrict__ out,       // [total_padded, dim]
    const __nv_bfloat16* __restrict__ src, // [M, dim]
    const int* __restrict__ sorted_ids,    // [total_padded]
    int dim, int M
) {
    int pos = blockIdx.x;
    int tid = sorted_ids[pos];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (tid < M) {
            out[pos * dim + i] = src[tid * dim + i];
        } else {
            out[pos * dim + i] = __float2bfloat16(0.0f); // padding
        }
    }
}

/* Scatter fused MoE output back to per-token FP32 accumulator.
 * The fused kernel outputs [total_sorted, hidden] with topk_weights already applied
 * (mul_topk_weights=True). sorted_token_ids[pos] gives the original token index.
 * Multiple sorted positions map to the same token (one per topk expert).
 * We scatter-add using atomicAdd on FP32.
 *
 * Grid: (total_sorted, 1, 1), Block: (threads, 1, 1)
 */
extern "C" __global__ void moe_scatter_fused_kernel(
    float* __restrict__ accum,              // [M, hidden] FP32 accumulator (pre-zeroed)
    const __nv_bfloat16* __restrict__ src,  // [total_sorted, hidden] fused output
    const int* __restrict__ sorted_ids,     // [total_sorted] maps sorted pos -> token*topk+slot
    int hidden, int M, float scale_factor,
    int topk                                // topk for vLLM-format sorted_ids (divide to get token)
) {
    int pos = blockIdx.x;
    int sid = sorted_ids[pos];
    int tid = topk > 1 ? sid / topk : sid;
    if (tid >= M) return; // padding slot
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float val = bf16_to_float(src[pos * hidden + i]) * scale_factor;
        atomicAdd(&accum[tid * hidden + i], val);
    }
}


/* Scatter fused MoE w2 output back to per-token FP32 accumulator with topk weights.
 * The fused Marlin w2 kernel writes compact rows directly at sorted_id = token*topk+slot,
 * so the routed contribution can be accumulated by iterating the compact [M*topk, hidden]
 * buffer directly. The padded sorted_ids metadata is still produced for the fused GEMM, but
 * it is not needed for the final scatter once rows are compact.
 *
 * Grid: (M*topk, 1, 1), Block: (threads, 1, 1)
 */
extern "C" __global__ void moe_scatter_weighted_kernel(
    float* __restrict__ accum,              // [M, hidden] FP32 accumulator (pre-zeroed)
    const __nv_bfloat16* __restrict__ src,  // [M*topk, hidden] fused w2 output (indexed by sorted_id)
    const int* __restrict__ sorted_ids,     // [total_sorted] maps sorted position -> token*topk+slot
    const float* __restrict__ topk_weights, // [M * topk] routing weights
    int hidden, int M, int topk, int total_sorted, float scale_factor
) {
    int row = blockIdx.x;  // compact sorted_id row = token*topk+slot
    int m_topk = M * topk;
    if (row >= m_topk) return;
    int token = row / topk;
    float w = topk_weights[row] * scale_factor;
    if (w == 0.0f) return;  // skip zero-weight entries
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float val = bf16_to_float(src[row * hidden + i]) * w;
        atomicAdd(&accum[token * hidden + i], val);
    }
}


/* ── Init / Cleanup (no-ops for the PTX kernel path) ──────────────────── */

extern "C" int krasis_prefill_init(int device) {
    cudaError_t err = cudaSetDevice(device);
    return (err == cudaSuccess) ? 0 : -1;
}

extern "C" void krasis_prefill_cleanup(void) {
    /* Nothing to clean up for PTX kernels */
}

// ── 4-bit PolarQuant KV Cache (Prefill) ──────────────────────────────────

// 16-level codebook for quantized angles (normalized components).
__device__ __constant__ float polar4_codebook_p[16] = {
    -0.6892f, -0.5241f, -0.4115f, -0.3206f, -0.2412f, -0.1685f, -0.0997f, -0.0330f,
     0.0330f,  0.0997f,  0.1685f,  0.2412f,  0.3206f,  0.4115f,  0.5241f,  0.6892f
};

// Fixed sign flip for SRR (16 elements)
__device__ __constant__ float polar4_signs_p[16] = {
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

// Fast Hadamard Transform for 16 elements (in-place)
__device__ inline void fht16_p(float* x) {
    float a, b;
    #pragma unroll
    for (int i = 0; i < 8; i++) { a = x[i]; b = x[i+8]; x[i] = a + b; x[i+8] = a - b; }
    #pragma unroll
    for (int i = 0; i < 4; i++) { a = x[i]; b = x[i+4]; x[i] = a + b; x[i+4] = a - b; }
    #pragma unroll
    for (int i = 8; i < 12; i++) { a = x[i]; b = x[i+4]; x[i] = a + b; x[i+4] = a - b; }
    #pragma unroll
    for (int i = 0; i < 16; i += 4) {
        a = x[i]; b = x[i+2]; x[i] = a + b; x[i+2] = a - b;
        a = x[i+1]; b = x[i+3]; x[i+1] = a + b; x[i+3] = a - b;
    }
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
        a = x[i]; b = x[i+1]; x[i] = a + b; x[i+1] = a - b;
    }
}

__device__ inline int quantize_polar4_p(float val) {
    int best_idx = 0;
    float min_diff = fabsf(val - polar4_codebook_p[0]);
    #pragma unroll
    for (int i = 1; i < 16; i++) {
        float diff = fabsf(val - polar4_codebook_p[i]);
        if (diff < min_diff) {
            min_diff = diff;
            best_idx = i;
        }
    }
    return best_idx;
}

__device__ inline float tq4_codebook_value_p(int idx, int head_dim) {
    // vLLM turboquant_4bit_nc Lloyd-Max centroids for N(0, 1), scaled to
    // N(0, 1/head_dim). These are algorithm constants, not model calibration.
    static const float lloyd4_unit[16] = {
        -2.7309222221f, -2.0684471130f, -1.6178817749f, -1.2562575340f,
        -0.9424482584f, -0.6568799019f, -0.3881377876f, -0.1284276545f,
         0.1284276545f,  0.3881377876f,  0.6568799019f,  0.9424482584f,
         1.2562575340f,  1.6178817749f,  2.0684471130f,  2.7309222221f,
    };
    return lloyd4_unit[idx] * rsqrtf((float)head_dim);
}

__device__ inline int quantize_tq4_p(float val, int head_dim) {
    int best_idx = 0;
    float best_diff = fabsf(val - tq4_codebook_value_p(0, head_dim));
    #pragma unroll
    for (int i = 1; i < 16; i++) {
        float diff = fabsf(val - tq4_codebook_value_p(i, head_dim));
        if (diff < best_diff) {
            best_diff = diff;
            best_idx = i;
        }
    }
    return best_idx;
}

__device__ inline void fht_shared_serial_p(float* x, int n) {
    for (int step = 1; step < n; step <<= 1) {
        int jump = step << 1;
        for (int base = 0; base < n; base += jump) {
            for (int j = 0; j < step; j++) {
                float a = x[base + j];
                float b = x[base + j + step];
                x[base + j] = a + b;
                x[base + j + step] = a - b;
            }
        }
    }
}

extern "C" __global__ void kv_cache_append_polar4_kernel(
    unsigned short* __restrict__ k_radius_cache,
    unsigned short* __restrict__ v_radius_cache,
    unsigned char* __restrict__ k_angles_cache,
    unsigned char* __restrict__ v_angles_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos,
    int norm_correction
) {
    int ti = blockIdx.x; // token index in prefill batch
    if (ti >= M) return;

    int num_blocks = kv_stride / 16;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;

        float k_local[16], v_local[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            k_local[i] = __bfloat162float(k[src_offset + i]) * polar4_signs_p[i];
            v_local[i] = __bfloat162float(v[src_offset + i]) * polar4_signs_p[i];
        }

        fht16_p(k_local);
        fht16_p(v_local);

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            k_local[i] *= 0.25f;
            v_local[i] *= 0.25f;
        }

        float k_r = 0.0f, v_r = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            k_r += k_local[i] * k_local[i];
            v_r += v_local[i] * v_local[i];
        }
        k_r = sqrtf(k_r + 1e-12f);
        v_r = sqrtf(v_r + 1e-12f);

        float inv_k_r = 1.0f / k_r;
        float inv_v_r = 1.0f / v_r;
        unsigned char* k_ang = k_angles_cache + (dst_pos * num_blocks + block_idx) * 8;
        unsigned char* v_ang = v_angles_cache + (dst_pos * num_blocks + block_idx) * 8;

        float k_qnorm2 = 0.0f;
        float v_qnorm2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int k0 = quantize_polar4_p(k_local[i*2] * inv_k_r);
            int k1 = quantize_polar4_p(k_local[i*2+1] * inv_k_r);
            k_ang[i] = (unsigned char)((k1 << 4) | k0);
            float k0v = polar4_codebook_p[k0];
            float k1v = polar4_codebook_p[k1];
            k_qnorm2 += k0v * k0v + k1v * k1v;
            int v0 = quantize_polar4_p(v_local[i*2] * inv_v_r);
            int v1 = quantize_polar4_p(v_local[i*2+1] * inv_v_r);
            v_ang[i] = (unsigned char)((v1 << 4) | v0);
            float v0v = polar4_codebook_p[v0];
            float v1v = polar4_codebook_p[v1];
            v_qnorm2 += v0v * v0v + v1v * v1v;
        }

        if (norm_correction & 1) {
            k_r = k_r / sqrtf(k_qnorm2 + 1e-12f);
        }
        if (norm_correction & 2) {
            v_r = v_r / sqrtf(v_qnorm2 + 1e-12f);
        }

        __nv_bfloat16 k_rb = __float2bfloat16(k_r);
        __nv_bfloat16 v_rb = __float2bfloat16(v_r);
        k_radius_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_rb);
        v_radius_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_rb);
    }
}

extern "C" __global__ void kv_cache_append_k8v4_kernel(
    __nv_fp8_e4m3* __restrict__ k_cache,
    unsigned short* __restrict__ v_radius_cache,
    unsigned char* __restrict__ v_angles_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos,
    int norm_correction
) {
    int ti = blockIdx.x;
    if (ti >= M) return;

    int num_blocks = kv_stride / 16;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;
        int dst_offset = dst_pos * kv_stride + block_idx * 16;
        float v_local[16];

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            k_cache[dst_offset + i] = bf16_to_fp8e4m3(k[src_offset + i]);
            v_local[i] = __bfloat162float(v[src_offset + i]) * polar4_signs_p[i];
        }

        fht16_p(v_local);
        #pragma unroll
        for (int i = 0; i < 16; i++) v_local[i] *= 0.25f;

        float v_r = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) v_r += v_local[i] * v_local[i];
        v_r = sqrtf(v_r + 1e-12f);

        float inv_v_r = 1.0f / v_r;
        unsigned char* v_ang = v_angles_cache + (dst_pos * num_blocks + block_idx) * 8;
        float v_qnorm2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int v0 = quantize_polar4_p(v_local[i*2] * inv_v_r);
            int v1 = quantize_polar4_p(v_local[i*2+1] * inv_v_r);
            v_ang[i] = (unsigned char)((v1 << 4) | v0);
            float v0v = polar4_codebook_p[v0];
            float v1v = polar4_codebook_p[v1];
            v_qnorm2 += v0v * v0v + v1v * v1v;
        }
        if (norm_correction & 2) {
            v_r = v_r / sqrtf(v_qnorm2 + 1e-12f);
        }

        __nv_bfloat16 v_rb = __float2bfloat16(v_r);
        v_radius_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_rb);
    }
}

__device__ inline void pack_k4_16_p(unsigned char* dst, const unsigned char* codes) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        dst[i] = (unsigned char)((codes[i * 2 + 1] << 4) | (codes[i * 2] & 0x0f));
    }
}

__device__ inline int unpack_k4_p(const unsigned char* src, int idx) {
    unsigned char packed = src[idx >> 1];
    return (idx & 1) ? (int)(packed >> 4) : (int)(packed & 0x0f);
}

__device__ inline float quantize_k4_one_pass_ls_p(const float* src, unsigned char* codes) {
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        max_abs = fmaxf(max_abs, fabsf(src[i]));
    }

    float k_scale = fmaxf(max_abs * (1.0f / 7.0f), 1e-8f);
    float inv_k_scale = 1.0f / k_scale;
    float ls_num = 0.0f;
    float ls_den = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float scaled = src[i] * inv_k_scale;
        int q = (int)(scaled >= 0.0f ? floorf(scaled + 0.5f) : -floorf(-scaled + 0.5f));
        q = max(-7, min(7, q));
        codes[i] = (unsigned char)(q + 8);
        float qf = (float)q;
        ls_num += src[i] * qf;
        ls_den += qf * qf;
    }
    if (ls_den > 1e-12f) {
        k_scale = fmaxf(ls_num / ls_den, 1e-8f);
    }
    return k_scale;
}

__device__ inline void pack_k6_16_p(unsigned char* dst, const unsigned char* codes) {
    #pragma unroll
    for (int i = 0; i < 12; i++) dst[i] = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int bit = i * 6;
        int byte = bit >> 3;
        int shift = bit & 7;
        unsigned int val = ((unsigned int)codes[i]) & 0x3fu;
        dst[byte] |= (unsigned char)(val << shift);
        if (shift > 2) {
            dst[byte + 1] |= (unsigned char)(val >> (8 - shift));
        }
    }
}

__device__ inline int unpack_k6_p(const unsigned char* src, int idx) {
    int bit = idx * 6;
    int byte = bit >> 3;
    int shift = bit & 7;
    unsigned int val = ((unsigned int)src[byte]) >> shift;
    if (shift > 2) {
        val |= ((unsigned int)src[byte + 1]) << (8 - shift);
    }
    return (int)(val & 0x3fu);
}

__device__ inline float quantize_k6_one_pass_ls_p(const float* src, unsigned char* codes) {
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        max_abs = fmaxf(max_abs, fabsf(src[i]));
    }

    float k_scale = fmaxf(max_abs * (1.0f / 31.0f), 1e-8f);
    float inv_k_scale = 1.0f / k_scale;
    float ls_num = 0.0f;
    float ls_den = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float scaled = src[i] * inv_k_scale;
        int q = (int)(scaled >= 0.0f ? floorf(scaled + 0.5f) : -floorf(-scaled + 0.5f));
        q = max(-31, min(31, q));
        codes[i] = (unsigned char)(q + 32);
        float qf = (float)q;
        ls_num += src[i] * qf;
        ls_den += qf * qf;
    }
    if (ls_den > 1e-12f) {
        k_scale = fmaxf(ls_num / ls_den, 1e-8f);
    }
    return k_scale;
}

__device__ inline void pack_k7_16_p(unsigned char* dst, const unsigned char* codes) {
    #pragma unroll
    for (int i = 0; i < 14; i++) dst[i] = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int bit = i * 7;
        int byte = bit >> 3;
        int shift = bit & 7;
        unsigned int val = ((unsigned int)codes[i]) & 0x7fu;
        dst[byte] |= (unsigned char)(val << shift);
        if (shift > 1) {
            dst[byte + 1] |= (unsigned char)(val >> (8 - shift));
        }
    }
}

__device__ inline int unpack_k7_p(const unsigned char* src, int idx) {
    int bit = idx * 7;
    int byte = bit >> 3;
    int shift = bit & 7;
    unsigned int val = ((unsigned int)src[byte]) >> shift;
    if (shift > 1) {
        val |= ((unsigned int)src[byte + 1]) << (8 - shift);
    }
    return (int)(val & 0x7fu);
}

__device__ inline float quantize_k7_one_pass_ls_p(const float* src, unsigned char* codes) {
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        max_abs = fmaxf(max_abs, fabsf(src[i]));
    }

    float k_scale = fmaxf(max_abs * (1.0f / 63.0f), 1e-8f);
    float inv_k_scale = 1.0f / k_scale;
    float ls_num = 0.0f;
    float ls_den = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float scaled = src[i] * inv_k_scale;
        int q = (int)(scaled >= 0.0f ? floorf(scaled + 0.5f) : -floorf(-scaled + 0.5f));
        q = max(-63, min(63, q));
        codes[i] = (unsigned char)(q + 64);
        float qf = (float)q;
        ls_num += src[i] * qf;
        ls_den += qf * qf;
    }
    if (ls_den > 1e-12f) {
        k_scale = fmaxf(ls_num / ls_den, 1e-8f);
    }
    return k_scale;
}

__device__ inline float quantize_k8_one_pass_ls_p(const float* src, unsigned char* codes) {
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        max_abs = fmaxf(max_abs, fabsf(src[i]));
    }

    float k_scale = fmaxf(max_abs * (1.0f / 127.0f), 1e-8f);
    float inv_k_scale = 1.0f / k_scale;
    float ls_num = 0.0f;
    float ls_den = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float scaled = src[i] * inv_k_scale;
        int q = (int)(scaled >= 0.0f ? floorf(scaled + 0.5f) : -floorf(-scaled + 0.5f));
        q = max(-127, min(127, q));
        codes[i] = (unsigned char)(q + 128);
        float qf = (float)q;
        ls_num += src[i] * qf;
        ls_den += qf * qf;
    }
    if (ls_den > 1e-12f) {
        k_scale = fmaxf(ls_num / ls_den, 1e-8f);
    }
    return k_scale;
}

extern "C" __global__ void kv_cache_append_k4v4_kernel(
    unsigned short* __restrict__ k_scale_cache,
    unsigned char* __restrict__ k_idx_cache,
    unsigned short* __restrict__ v_radius_cache,
    unsigned char* __restrict__ v_angles_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos,
    int norm_correction
) {
    int ti = blockIdx.x;
    if (ti >= M) return;

    int num_blocks = kv_stride / 16;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;
        float k_local[16];
        float v_local[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float kval = __bfloat162float(k[src_offset + i]);
            k_local[i] = kval;
            v_local[i] = __bfloat162float(v[src_offset + i]) * polar4_signs_p[i];
        }

        unsigned char codes[16];
        float k_scale = quantize_k4_one_pass_ls_p(k_local, codes);
        unsigned char* k_pack = k_idx_cache + (dst_pos * num_blocks + block_idx) * 8;
        pack_k4_16_p(k_pack, codes);
        __nv_bfloat16 k_sb = __float2bfloat16(k_scale);
        k_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_sb);

        fht16_p(v_local);
        #pragma unroll
        for (int i = 0; i < 16; i++) v_local[i] *= 0.25f;

        float v_r = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) v_r += v_local[i] * v_local[i];
        v_r = sqrtf(v_r + 1e-12f);
        float inv_v_r = 1.0f / v_r;
        unsigned char* v_ang = v_angles_cache + (dst_pos * num_blocks + block_idx) * 8;
        float v_qnorm2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int v0 = quantize_polar4_p(v_local[i*2] * inv_v_r);
            int v1 = quantize_polar4_p(v_local[i*2+1] * inv_v_r);
            v_ang[i] = (unsigned char)((v1 << 4) | v0);
            float v0v = polar4_codebook_p[v0];
            float v1v = polar4_codebook_p[v1];
            v_qnorm2 += v0v * v0v + v1v * v1v;
        }
        if (norm_correction & 2) {
            v_r = v_r / sqrtf(v_qnorm2 + 1e-12f);
        }
        __nv_bfloat16 v_rb = __float2bfloat16(v_r);
        v_radius_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_rb);
    }
}

extern "C" __global__ void kv_cache_append_k6v4_kernel(
    unsigned short* __restrict__ k_scale_cache,
    unsigned char* __restrict__ k_idx_cache,
    unsigned short* __restrict__ v_radius_cache,
    unsigned char* __restrict__ v_angles_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos,
    int norm_correction
) {
    int ti = blockIdx.x;
    if (ti >= M) return;

    int num_blocks = kv_stride / 16;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;
        float k_local[16];
        float v_local[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float kval = __bfloat162float(k[src_offset + i]);
            k_local[i] = kval;
            v_local[i] = __bfloat162float(v[src_offset + i]) * polar4_signs_p[i];
        }

        unsigned char codes[16];
        float k_scale = quantize_k6_one_pass_ls_p(k_local, codes);
        unsigned char* k_pack = k_idx_cache + (dst_pos * num_blocks + block_idx) * 12;
        pack_k6_16_p(k_pack, codes);
        __nv_bfloat16 k_sb = __float2bfloat16(k_scale);
        k_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_sb);

        fht16_p(v_local);
        #pragma unroll
        for (int i = 0; i < 16; i++) v_local[i] *= 0.25f;

        float v_r = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) v_r += v_local[i] * v_local[i];
        v_r = sqrtf(v_r + 1e-12f);
        float inv_v_r = 1.0f / v_r;
        unsigned char* v_ang = v_angles_cache + (dst_pos * num_blocks + block_idx) * 8;
        float v_qnorm2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int v0 = quantize_polar4_p(v_local[i*2] * inv_v_r);
            int v1 = quantize_polar4_p(v_local[i*2+1] * inv_v_r);
            v_ang[i] = (unsigned char)((v1 << 4) | v0);
            float v0v = polar4_codebook_p[v0];
            float v1v = polar4_codebook_p[v1];
            v_qnorm2 += v0v * v0v + v1v * v1v;
        }
        if (norm_correction & 2) {
            v_r = v_r / sqrtf(v_qnorm2 + 1e-12f);
        }
        __nv_bfloat16 v_rb = __float2bfloat16(v_r);
        v_radius_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_rb);
    }
}

extern "C" __global__ void kv_cache_append_k7v4_kernel(
    unsigned short* __restrict__ k_scale_cache,
    unsigned char* __restrict__ k_idx_cache,
    unsigned short* __restrict__ v_radius_cache,
    unsigned char* __restrict__ v_angles_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos,
    int norm_correction
) {
    int ti = blockIdx.x;
    if (ti >= M) return;

    int num_blocks = kv_stride / 16;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;
        float k_local[16];
        float v_local[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            float kval = __bfloat162float(k[src_offset + i]);
            k_local[i] = kval;
            v_local[i] = __bfloat162float(v[src_offset + i]) * polar4_signs_p[i];
        }

        unsigned char codes[16];
        float k_scale = quantize_k7_one_pass_ls_p(k_local, codes);
        unsigned char* k_pack = k_idx_cache + (dst_pos * num_blocks + block_idx) * 14;
        pack_k7_16_p(k_pack, codes);
        __nv_bfloat16 k_sb = __float2bfloat16(k_scale);
        k_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_sb);

        fht16_p(v_local);
        #pragma unroll
        for (int i = 0; i < 16; i++) v_local[i] *= 0.25f;

        float v_r = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) v_r += v_local[i] * v_local[i];
        v_r = sqrtf(v_r + 1e-12f);
        float inv_v_r = 1.0f / v_r;
        unsigned char* v_ang = v_angles_cache + (dst_pos * num_blocks + block_idx) * 8;
        float v_qnorm2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int v0 = quantize_polar4_p(v_local[i*2] * inv_v_r);
            int v1 = quantize_polar4_p(v_local[i*2+1] * inv_v_r);
            v_ang[i] = (unsigned char)((v1 << 4) | v0);
            float v0v = polar4_codebook_p[v0];
            float v1v = polar4_codebook_p[v1];
            v_qnorm2 += v0v * v0v + v1v * v1v;
        }
        if (norm_correction & 2) {
            v_r = v_r / sqrtf(v_qnorm2 + 1e-12f);
        }
        __nv_bfloat16 v_rb = __float2bfloat16(v_r);
        v_radius_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_rb);
    }
}

extern "C" __global__ void kv_cache_append_tq4_kernel(
    unsigned short* __restrict__ k_norm_cache,
    unsigned char* __restrict__ k_idx_cache,
    unsigned short* __restrict__ v_meta_cache,
    unsigned char* __restrict__ v_idx_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ signs,
    int M,
    int num_kv_heads,
    int head_dim,
    int max_seq,
    int start_pos
) {
    int ti = blockIdx.x;
    int kv_head = blockIdx.y;
    if (ti >= M || kv_head >= num_kv_heads || threadIdx.x != 0) return;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    extern __shared__ float scratch[];
    int kv_stride = num_kv_heads * head_dim;
    int packed_hd = (head_dim + 1) >> 1;
    int token_head = dst_pos * num_kv_heads + kv_head;
    int src_off = ti * kv_stride + kv_head * head_dim;

    float k_norm2 = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float val = __bfloat162float(k[src_off + d]);
        scratch[d] = val * signs[d];
        k_norm2 += val * val;
    }
    float k_norm = sqrtf(k_norm2 + 1e-12f);
    fht_shared_serial_p(scratch, head_dim);
    float inv_sqrt_hd = rsqrtf((float)head_dim);
    float inv_norm = 1.0f / k_norm;
    float q_norm2 = 0.0f;
    unsigned char* k_pack = k_idx_cache + token_head * packed_hd;
    for (int d = 0; d < head_dim; d += 2) {
        int q0 = quantize_tq4_p(scratch[d] * inv_sqrt_hd * inv_norm, head_dim);
        int q1 = 0;
        q_norm2 += tq4_codebook_value_p(q0, head_dim) * tq4_codebook_value_p(q0, head_dim);
        if (d + 1 < head_dim) {
            q1 = quantize_tq4_p(scratch[d + 1] * inv_sqrt_hd * inv_norm, head_dim);
            q_norm2 += tq4_codebook_value_p(q1, head_dim) * tq4_codebook_value_p(q1, head_dim);
        }
        k_pack[d >> 1] = (unsigned char)((q1 << 4) | q0);
    }
    float corrected_norm = k_norm / sqrtf(q_norm2 + 1e-12f);
    __half k_nb = __float2half(corrected_norm);
    k_norm_cache[token_head] = *reinterpret_cast<unsigned short*>(&k_nb);

    float v_min = 1e30f;
    float v_max = -1e30f;
    for (int d = 0; d < head_dim; d++) {
        float val = __bfloat162float(v[src_off + d]);
        v_min = fminf(v_min, val);
        v_max = fmaxf(v_max, val);
    }
    float scale = (v_max - v_min) * (1.0f / 15.0f);
    if (scale < 1e-8f) scale = 1e-8f;
    float inv_scale = 1.0f / scale;
    unsigned char* v_pack = v_idx_cache + token_head * packed_hd;
    for (int d = 0; d < head_dim; d += 2) {
        int q0 = (int)floorf((__bfloat162float(v[src_off + d]) - v_min) * inv_scale + 0.5f);
        q0 = max(0, min(15, q0));
        int q1 = 0;
        if (d + 1 < head_dim) {
            q1 = (int)floorf((__bfloat162float(v[src_off + d + 1]) - v_min) * inv_scale + 0.5f);
            q1 = max(0, min(15, q1));
        }
        v_pack[d >> 1] = (unsigned char)((q1 << 4) | q0);
    }
    __half scale_b = __float2half(scale);
    __half zero_b = __float2half(v_min);
    v_meta_cache[token_head * 2] = *reinterpret_cast<unsigned short*>(&scale_b);
    v_meta_cache[token_head * 2 + 1] = *reinterpret_cast<unsigned short*>(&zero_b);
}

/* ── Polar4 KV Cache Dequant + Concat for Cross-Chunk FA2 ───────────────
 * Reconstructs Polar4 cache [0..cache_len] back to BF16 K/V in the original
 * domain, then copies current-chunk BF16 K/V [0..m] into [cache_len..cache_len+m].
 *
 * Grid: (cache_len + m, 1, 1), Block: (threads, 1, 1)
 * Threads grid-stride over 16-value blocks within the KV stride.
 */
extern "C" __global__ void kv_cache_dequant_concat_polar4_kernel(
    __nv_bfloat16* __restrict__ out,                /* [cache_len+m, kv_stride] BF16 output */
    const unsigned short* __restrict__ radius_cache,/* [max_seq, num_blocks] BF16 radii */
    const unsigned char* __restrict__ angles_cache, /* [max_seq, num_blocks * 8] packed 4-bit */
    const __nv_bfloat16* __restrict__ kv_new,       /* [m, kv_stride] BF16 current chunk */
    int cache_len,                                  /* number of cached tokens */
    int m,                                          /* current chunk size */
    int kv_stride)                                  /* num_kv_heads * head_dim */
{
    int ti = blockIdx.x;
    int num_blocks = kv_stride / 16;
    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int base = block_idx * 16;

        if (ti < cache_len) {
            float local[16];
            float r = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(&radius_cache[ti * num_blocks + block_idx])
            );
            const unsigned char* ang = angles_cache + (ti * num_blocks + block_idx) * 8;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                unsigned char p = ang[i];
                local[i * 2] = r * polar4_codebook_p[p & 0xF];
                local[i * 2 + 1] = r * polar4_codebook_p[p >> 4];
            }

            // Inverse SRR: x = S * H(y) / 4
            fht16_p(local);
            int64_t dst_off = (int64_t)ti * kv_stride + base;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                out[dst_off + i] = __float2bfloat16(local[i] * 0.25f * polar4_signs_p[i]);
            }
        } else {
            int ci = ti - cache_len;
            if (ci < m) {
                int64_t src_off = (int64_t)ci * kv_stride + base;
                int64_t dst_off = (int64_t)ti * kv_stride + base;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    out[dst_off + i] = kv_new[src_off + i];
                }
            }
        }
    }
}

extern "C" __global__ void kv_cache_append_k6v6_kernel(
    unsigned short* __restrict__ k_scale_cache,
    unsigned char* __restrict__ k_idx_cache,
    unsigned short* __restrict__ v_scale_cache,
    unsigned char* __restrict__ v_idx_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos,
    int norm_correction
) {
    (void)norm_correction;
    int ti = blockIdx.x;
    if (ti >= M) return;

    int num_blocks = kv_stride / 16;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;
        float k_local[16];
        float v_local[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            k_local[i] = __bfloat162float(k[src_offset + i]);
            v_local[i] = __bfloat162float(v[src_offset + i]);
        }

        unsigned char codes[16];
        float k_scale = quantize_k6_one_pass_ls_p(k_local, codes);
        unsigned char* k_pack = k_idx_cache + (dst_pos * num_blocks + block_idx) * 12;
        pack_k6_16_p(k_pack, codes);
        __nv_bfloat16 k_sb = __float2bfloat16(k_scale);
        k_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_sb);

        float v_scale = quantize_k6_one_pass_ls_p(v_local, codes);
        unsigned char* v_pack = v_idx_cache + (dst_pos * num_blocks + block_idx) * 12;
        pack_k6_16_p(v_pack, codes);
        __nv_bfloat16 v_sb = __float2bfloat16(v_scale);
        v_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_sb);
    }
}

extern "C" __global__ void kv_cache_append_k8v6_kernel(
    unsigned short* __restrict__ k_scale_cache,
    unsigned char* __restrict__ k_idx_cache,
    unsigned short* __restrict__ v_scale_cache,
    unsigned char* __restrict__ v_idx_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int M,
    int kv_stride,
    int max_seq,
    int start_pos,
    int norm_correction
) {
    (void)norm_correction;
    int ti = blockIdx.x;
    if (ti >= M) return;

    int num_blocks = kv_stride / 16;
    int dst_pos = start_pos + ti;
    if (dst_pos >= max_seq) return;

    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int src_offset = ti * kv_stride + block_idx * 16;
        float k_local[16];
        float v_local[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            k_local[i] = __bfloat162float(k[src_offset + i]);
            v_local[i] = __bfloat162float(v[src_offset + i]);
        }

        unsigned char codes[16];
        float k_scale = quantize_k8_one_pass_ls_p(k_local, codes);
        unsigned char* k_pack = k_idx_cache + (dst_pos * num_blocks + block_idx) * 16;
        #pragma unroll
        for (int i = 0; i < 16; i++) k_pack[i] = codes[i];
        __nv_bfloat16 k_sb = __float2bfloat16(k_scale);
        k_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&k_sb);

        float v_scale = quantize_k6_one_pass_ls_p(v_local, codes);
        unsigned char* v_pack = v_idx_cache + (dst_pos * num_blocks + block_idx) * 12;
        pack_k6_16_p(v_pack, codes);
        __nv_bfloat16 v_sb = __float2bfloat16(v_scale);
        v_scale_cache[dst_pos * num_blocks + block_idx] = *reinterpret_cast<unsigned short*>(&v_sb);
    }
}

extern "C" __global__ void kv_cache_dequant_concat_k4_kernel(
    __nv_bfloat16* __restrict__ out,
    const unsigned short* __restrict__ k_scale_cache,
    const unsigned char* __restrict__ k_idx_cache,
    const __nv_bfloat16* __restrict__ k_new,
    int cache_len,
    int m,
    int kv_stride)
{
    int ti = blockIdx.x;
    int num_blocks = kv_stride / 16;
    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int base = block_idx * 16;
        int64_t dst_off = (int64_t)ti * kv_stride + base;
        if (ti < cache_len) {
            float scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                &k_scale_cache[ti * num_blocks + block_idx]));
            const unsigned char* k_pack = k_idx_cache + (ti * num_blocks + block_idx) * 8;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float val = scale * (float)(unpack_k4_p(k_pack, i) - 8);
                out[dst_off + i] = __float2bfloat16(val);
            }
        } else {
            int ci = ti - cache_len;
            if (ci < m) {
                int64_t src_off = (int64_t)ci * kv_stride + base;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    out[dst_off + i] = k_new[src_off + i];
                }
            }
        }
    }
}

extern "C" __global__ void kv_cache_dequant_concat_k6_kernel(
    __nv_bfloat16* __restrict__ out,
    const unsigned short* __restrict__ k_scale_cache,
    const unsigned char* __restrict__ k_idx_cache,
    const __nv_bfloat16* __restrict__ k_new,
    int cache_len,
    int m,
    int kv_stride)
{
    int ti = blockIdx.x;
    int num_blocks = kv_stride / 16;
    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int base = block_idx * 16;
        int64_t dst_off = (int64_t)ti * kv_stride + base;
        if (ti < cache_len) {
            float scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                &k_scale_cache[ti * num_blocks + block_idx]));
            const unsigned char* k_pack = k_idx_cache + (ti * num_blocks + block_idx) * 12;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float val = scale * (float)(unpack_k6_p(k_pack, i) - 32);
                out[dst_off + i] = __float2bfloat16(val);
            }
        } else {
            int ci = ti - cache_len;
            if (ci < m) {
                int64_t src_off = (int64_t)ci * kv_stride + base;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    out[dst_off + i] = k_new[src_off + i];
                }
            }
        }
    }
}

extern "C" __global__ void kv_cache_dequant_concat_k7_kernel(
    __nv_bfloat16* __restrict__ out,
    const unsigned short* __restrict__ k_scale_cache,
    const unsigned char* __restrict__ k_idx_cache,
    const __nv_bfloat16* __restrict__ k_new,
    int cache_len,
    int m,
    int kv_stride)
{
    int ti = blockIdx.x;
    int num_blocks = kv_stride / 16;
    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int base = block_idx * 16;
        int64_t dst_off = (int64_t)ti * kv_stride + base;
        if (ti < cache_len) {
            float scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                &k_scale_cache[ti * num_blocks + block_idx]));
            const unsigned char* k_pack = k_idx_cache + (ti * num_blocks + block_idx) * 14;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float val = scale * (float)(unpack_k7_p(k_pack, i) - 64);
                out[dst_off + i] = __float2bfloat16(val);
            }
        } else {
            int ci = ti - cache_len;
            if (ci < m) {
                int64_t src_off = (int64_t)ci * kv_stride + base;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    out[dst_off + i] = k_new[src_off + i];
                }
            }
        }
    }
}

extern "C" __global__ void kv_cache_dequant_concat_k8_kernel(
    __nv_bfloat16* __restrict__ out,
    const unsigned short* __restrict__ k_scale_cache,
    const unsigned char* __restrict__ k_idx_cache,
    const __nv_bfloat16* __restrict__ k_new,
    int cache_len,
    int m,
    int kv_stride)
{
    int ti = blockIdx.x;
    int num_blocks = kv_stride / 16;
    for (int block_idx = threadIdx.x; block_idx < num_blocks; block_idx += blockDim.x) {
        int base = block_idx * 16;
        int64_t dst_off = (int64_t)ti * kv_stride + base;
        if (ti < cache_len) {
            float scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
                &k_scale_cache[ti * num_blocks + block_idx]));
            const unsigned char* k_pack = k_idx_cache + (ti * num_blocks + block_idx) * 16;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float val = scale * (float)((int)k_pack[i] - 128);
                out[dst_off + i] = __float2bfloat16(val);
            }
        } else {
            int ci = ti - cache_len;
            if (ci < m) {
                int64_t src_off = (int64_t)ci * kv_stride + base;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    out[dst_off + i] = k_new[src_off + i];
                }
            }
        }
    }
}

extern "C" __global__ void kv_cache_dequant_concat_tq4_k_kernel(
    __nv_bfloat16* __restrict__ out,
    const unsigned short* __restrict__ k_norm_cache,
    const unsigned char* __restrict__ k_idx_cache,
    const float* __restrict__ signs,
    const __nv_bfloat16* __restrict__ k_new,
    int cache_len,
    int m,
    int num_kv_heads,
    int head_dim)
{
    int ti = blockIdx.x;
    int kv_head = blockIdx.y;
    if (threadIdx.x != 0) return;
    int packed_hd = (head_dim + 1) >> 1;
    int kv_stride = num_kv_heads * head_dim;
    int64_t dst_base = (int64_t)ti * kv_stride + kv_head * head_dim;
    if (ti < cache_len) {
        extern __shared__ float scratch[];
        int token_head = ti * num_kv_heads + kv_head;
        float kn = __half2float(*reinterpret_cast<const __half*>(
            &k_norm_cache[token_head]));
        for (int d = 0; d < head_dim; d++) {
            unsigned char p = k_idx_cache[token_head * packed_hd + (d >> 1)];
            int idx = ((d & 1) == 0) ? (p & 0xF) : (p >> 4);
            scratch[d] = kn * tq4_codebook_value_p(idx, head_dim);
        }
        fht_shared_serial_p(scratch, head_dim);
        float inv_sqrt_hd = rsqrtf((float)head_dim);
        for (int d = 0; d < head_dim; d++) {
            out[dst_base + d] = __float2bfloat16(scratch[d] * inv_sqrt_hd * signs[d]);
        }
    } else {
        int ci = ti - cache_len;
        if (ci < m) {
            int64_t src_base = (int64_t)ci * kv_stride + kv_head * head_dim;
            for (int d = 0; d < head_dim; d++) out[dst_base + d] = k_new[src_base + d];
        }
    }
}

extern "C" __global__ void kv_cache_dequant_concat_tq4_v_kernel(
    __nv_bfloat16* __restrict__ out,
    const unsigned short* __restrict__ v_meta_cache,
    const unsigned char* __restrict__ v_idx_cache,
    const __nv_bfloat16* __restrict__ v_new,
    int cache_len,
    int m,
    int num_kv_heads,
    int head_dim)
{
    int ti = blockIdx.x;
    int kv_head = blockIdx.y;
    int packed_hd = (head_dim + 1) >> 1;
    int kv_stride = num_kv_heads * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        int64_t dst_off = (int64_t)ti * kv_stride + kv_head * head_dim + d;
        if (ti < cache_len) {
            int token_head = ti * num_kv_heads + kv_head;
            float scale = __half2float(*reinterpret_cast<const __half*>(
                &v_meta_cache[token_head * 2]));
            float zero = __half2float(*reinterpret_cast<const __half*>(
                &v_meta_cache[token_head * 2 + 1]));
            unsigned char p = v_idx_cache[token_head * packed_hd + (d >> 1)];
            int idx = ((d & 1) == 0) ? (p & 0xF) : (p >> 4);
            out[dst_off] = __float2bfloat16(zero + scale * (float)idx);
        } else {
            int ci = ti - cache_len;
            if (ci < m) {
                int64_t src_off = (int64_t)ci * kv_stride + kv_head * head_dim + d;
                out[dst_off] = v_new[src_off];
            }
        }
    }
}
