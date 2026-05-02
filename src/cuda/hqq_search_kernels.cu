#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

static __device__ __forceinline__ float clamp_f32(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

static __device__ __forceinline__ uint8_t quant_value(float value, float scale, float zero, float qmax) {
    float q = nearbyintf(value / scale + zero);
    q = clamp_f32(q, 0.0f, qmax);
    return (uint8_t)q;
}

static __device__ float block_sum(float value) {
    __shared__ float scratch[256];
    int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    float result = scratch[0];
    __syncthreads();
    return result;
}

static __device__ float block_min(float value) {
    __shared__ float scratch[256];
    int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fminf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    float result = scratch[0];
    __syncthreads();
    return result;
}

static __device__ float block_max(float value) {
    __shared__ float scratch[256];
    int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    float result = scratch[0];
    __syncthreads();
    return result;
}

static __device__ float solve_candidate(
    const float* input,
    int row_offset,
    int start_col,
    int chunk_cols,
    float zero,
    float scale_seed,
    float qmax,
    int iters,
    uint8_t* thread_q_out
) {
    int tid = threadIdx.x;
    float value = 0.0f;
    bool active = tid < chunk_cols;
    if (active) {
        value = input[row_offset + start_col + tid];
    }

    float scale = fmaxf(scale_seed, 1.0e-8f);
    for (int iter = 0; iter < iters; ++iter) {
        uint8_t q = active ? quant_value(value, scale, zero, qmax) : 0;
        float centered = active ? ((float)q - zero) : 0.0f;
        float denom = block_sum(centered * centered);
        float numer = block_sum(value * centered);
        if (denom > 1.0e-12f) {
            scale = fmaxf(numer / denom, 1.0e-8f);
        } else {
            scale = fmaxf(scale, 1.0e-8f);
        }
    }

    uint8_t q = active ? quant_value(value, scale, zero, qmax) : 0;
    float centered = active ? ((float)q - zero) : 0.0f;
    float deq = centered * scale;
    float diff = active ? (deq - value) : 0.0f;
    float rmse = block_sum(diff * diff) / (float)chunk_cols;
    if (active) {
        *thread_q_out = q;
    }
    return rmse;
}

extern "C" __global__ void hqq_search_global_kernel(
    const float* input,
    int rows,
    int cols,
    int group_size,
    int groups,
    int padded_cols,
    int nbits,
    int global_steps,
    int iters,
    uint8_t* quant,
    float* scales,
    float* zeros,
    float* best_rmse
) {
    int row_group = (int)blockIdx.x;
    int row = row_group / groups;
    int group = row_group - row * groups;
    if (row >= rows || group >= groups) {
        return;
    }
    int tid = threadIdx.x;
    int start_col = group * group_size;
    int chunk_cols = min(group_size, cols - start_col);
    int row_offset = row * cols;
    bool active = tid < chunk_cols;
    float value = active ? input[row_offset + start_col + tid] : 0.0f;
    float qmax = (float)((1 << nbits) - 1);

    float minv = block_min(active ? value : FLT_MAX);
    float maxv = block_max(active ? value : -FLT_MAX);
    float amax = block_max(active ? fabsf(value) : 0.0f);
    float range_scale = fmaxf((maxv - minv) / qmax, 1.0e-8f);
    float abs_scale = fmaxf((2.0f * amax) / qmax, 1.0e-8f);
    float init_zero = clamp_f32(-minv / range_scale, 0.0f, qmax);

    uint8_t best_q = active ? quant_value(value, range_scale, init_zero, qmax) : 0;
    float init_centered = active ? ((float)best_q - init_zero) : 0.0f;
    float init_diff = active ? (init_centered * range_scale - value) : 0.0f;
    float best_error = block_sum(init_diff * init_diff) / (float)chunk_cols;
    float best_scale = range_scale;
    float best_zero = init_zero;

    for (int phase = 0; phase < 2; ++phase) {
        float seed = phase == 0 ? range_scale : abs_scale;
        for (int step = 0; step < global_steps; ++step) {
            float zero = global_steps <= 1
                ? 0.0f
                : qmax * ((float)step / (float)(global_steps - 1));
            uint8_t candidate_q = 0;
            float rmse = solve_candidate(
                input,
                row_offset,
                start_col,
                chunk_cols,
                zero,
                seed,
                qmax,
                iters,
                &candidate_q
            );
            if (rmse < best_error) {
                best_error = rmse;
                best_scale = fmaxf(seed, 1.0e-8f);
                // Re-run cheaply enough to recover the solved scale. The candidate
                // helper returns rmse only to keep register pressure low in the loop.
                float scale = fmaxf(seed, 1.0e-8f);
                for (int iter = 0; iter < iters; ++iter) {
                    uint8_t q = active ? quant_value(value, scale, zero, qmax) : 0;
                    float centered = active ? ((float)q - zero) : 0.0f;
                    float denom = block_sum(centered * centered);
                    float numer = block_sum(value * centered);
                    if (denom > 1.0e-12f) {
                        scale = fmaxf(numer / denom, 1.0e-8f);
                    }
                }
                best_scale = scale;
                best_zero = zero;
                if (active) {
                    best_q = quant_value(value, best_scale, best_zero, qmax);
                }
            }
        }
    }

    if (active) {
        quant[row * padded_cols + start_col + tid] = best_q;
    }
    if (tid == 0) {
        int meta_idx = row * groups + group;
        scales[meta_idx] = best_scale;
        zeros[meta_idx] = best_zero;
        best_rmse[meta_idx] = best_error;
    }
}

extern "C" __global__ void hqq_search_zero_bounds_kernel(
    const float* zeros,
    int rows,
    int groups,
    float qmax,
    float* local_min,
    float* local_max
) {
    int group = (int)blockIdx.x;
    int tid = threadIdx.x;
    float zmin = FLT_MAX;
    float zmax = -FLT_MAX;
    for (int row = tid; row < rows; row += blockDim.x) {
        float z = zeros[row * groups + group];
        zmin = fminf(zmin, z);
        zmax = fmaxf(zmax, z);
    }
    zmin = block_min(zmin);
    zmax = block_max(zmax);
    if (tid == 0) {
        local_min[group] = fmaxf(zmin - 0.5f, 0.0f);
        local_max[group] = fminf(zmax + 0.5f, qmax);
    }
}

extern "C" __global__ void hqq_search_local_kernel(
    const float* input,
    int rows,
    int cols,
    int group_size,
    int groups,
    int padded_cols,
    int nbits,
    int local_steps,
    int iters,
    const float* local_min,
    const float* local_max,
    uint8_t* quant,
    float* scales,
    float* zeros,
    float* best_rmse
) {
    int row_group = (int)blockIdx.x;
    int row = row_group / groups;
    int group = row_group - row * groups;
    if (row >= rows || group >= groups) {
        return;
    }
    int tid = threadIdx.x;
    int start_col = group * group_size;
    int chunk_cols = min(group_size, cols - start_col);
    int row_offset = row * cols;
    bool active = tid < chunk_cols;
    float value = active ? input[row_offset + start_col + tid] : 0.0f;
    float qmax = (float)((1 << nbits) - 1);
    int meta_idx = row * groups + group;
    float best_scale = scales[meta_idx];
    float best_zero = zeros[meta_idx];
    float best_error = best_rmse[meta_idx];
    uint8_t best_q = active ? quant[row * padded_cols + start_col + tid] : 0;

    float zlo = local_min[group];
    float zhi = local_max[group];
    float seed = best_scale;
    for (int step = 0; step < local_steps; ++step) {
        float zero = local_steps <= 1
            ? zlo
            : zlo + (zhi - zlo) * ((float)step / (float)(local_steps - 1));
        uint8_t candidate_q = 0;
        float rmse = solve_candidate(
            input,
            row_offset,
            start_col,
            chunk_cols,
            zero,
            seed,
            qmax,
            iters,
            &candidate_q
        );
        if (rmse < best_error) {
            best_error = rmse;
            float scale = fmaxf(seed, 1.0e-8f);
            for (int iter = 0; iter < iters; ++iter) {
                uint8_t q = active ? quant_value(value, scale, zero, qmax) : 0;
                float centered = active ? ((float)q - zero) : 0.0f;
                float denom = block_sum(centered * centered);
                float numer = block_sum(value * centered);
                if (denom > 1.0e-12f) {
                    scale = fmaxf(numer / denom, 1.0e-8f);
                }
            }
            best_scale = scale;
            best_zero = zero;
            if (active) {
                best_q = quant_value(value, best_scale, best_zero, qmax);
            }
        }
    }

    if (active) {
        quant[row * padded_cols + start_col + tid] = best_q;
    }
    if (tid == 0) {
        scales[meta_idx] = best_scale;
        zeros[meta_idx] = best_zero;
        best_rmse[meta_idx] = best_error;
    }
}
