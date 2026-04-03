#include <stdint.h>

struct KernelDebugErrorRecord {
    uint32_t status;
    uint32_t site;
    uint32_t block_x;
    uint32_t thread_x;
    int32_t index;
    int32_t bound;
    int32_t aux0;
    int32_t aux1;
};

static __device__ __forceinline__ void krasis_record_kernel_debug_failure(
    KernelDebugErrorRecord* error,
    uint32_t site,
    int32_t index,
    int32_t bound,
    int32_t aux0,
    int32_t aux1)
{
    if (error == nullptr) {
        return;
    }
    if (atomicCAS(&error->status, 0u, 1u) == 0u) {
        error->site = site;
        error->block_x = blockIdx.x;
        error->thread_x = threadIdx.x;
        error->index = index;
        error->bound = bound;
        error->aux0 = aux0;
        error->aux1 = aux1;
    }
}

extern "C" __global__ void krasis_guard_check_kernel(
    const unsigned char* prefix,
    const unsigned char* suffix,
    int prefix_len,
    int suffix_len,
    uint64_t alloc_id,
    uint64_t generation,
    uint64_t payload_bytes,
    uint64_t tag_hash,
    int live,
    uint32_t* result_flags,
    uint32_t* prefix_mismatch_idx,
    uint32_t* suffix_mismatch_idx)
{
    const unsigned char alloc_pattern = live ? 0xA7 : 0xDD;
    const unsigned char prefix_pattern = live ? 0xC3 : 0xDD;
    const unsigned char suffix_pattern = live ? 0x3C : 0xDD;

    auto expected_guard_byte = [&](int idx, int len, bool is_prefix) -> unsigned char {
        if (len <= 0) {
            return 0;
        }
        unsigned char expected = alloc_pattern;
        if (live) {
            expected = is_prefix ? prefix_pattern : suffix_pattern;
        }
        if (idx < 0 || idx >= len) {
            return expected;
        }

        uint64_t metadata_value = 0;
        bool metadata_hit = true;
        if (idx < 8) {
            metadata_value = alloc_id;
        } else if (idx < 16) {
            metadata_value = generation;
        } else if (idx < 24) {
            metadata_value = payload_bytes;
        } else if (idx < 32) {
            metadata_value = tag_hash;
        } else if (len >= 16 && idx >= len - 16 && idx < len - 8) {
            metadata_value = alloc_id ^ tag_hash;
            idx -= (len - 16);
        } else if (len >= 8 && idx >= len - 8) {
            metadata_value = generation ^ payload_bytes;
            idx -= (len - 8);
        } else {
            metadata_hit = false;
        }

        if (!metadata_hit) {
            return expected;
        }
        return static_cast<unsigned char>((metadata_value >> ((idx & 7) * 8)) & 0xFFu);
    };

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (int i = static_cast<int>(tid); i < prefix_len; i += static_cast<int>(stride)) {
        unsigned char expected = expected_guard_byte(i, prefix_len, true);
        if (prefix[i] != expected) {
            atomicOr(result_flags, 1u);
            atomicMin(prefix_mismatch_idx, static_cast<uint32_t>(i));
        }
    }

    for (int i = static_cast<int>(tid); i < suffix_len; i += static_cast<int>(stride)) {
        unsigned char expected = expected_guard_byte(i, suffix_len, false);
        if (suffix[i] != expected) {
            atomicOr(result_flags, 2u);
            atomicMin(suffix_mismatch_idx, static_cast<uint32_t>(i));
        }
    }
}

extern "C" __global__ void krasis_sigmoid_topk_debug_kernel(
    float* __restrict__ topk_weights,
    int* __restrict__ topk_ids,
    const float* __restrict__ gate,
    int E,
    int topk,
    int smem_bytes,
    KernelDebugErrorRecord* __restrict__ error)
{
    if (E <= 0) {
        if (threadIdx.x == 0) {
            krasis_record_kernel_debug_failure(error, 1001u, E, 0, topk, smem_bytes);
        }
        return;
    }
    if (topk <= 0 || topk > E) {
        if (threadIdx.x == 0) {
            krasis_record_kernel_debug_failure(error, 1002u, topk, E, smem_bytes, 0);
        }
        return;
    }

    int required_smem = E * (int)sizeof(float) + topk * ((int)sizeof(float) + (int)sizeof(int));
    if (required_smem > smem_bytes) {
        if (threadIdx.x == 0) {
            krasis_record_kernel_debug_failure(error, 1003u, required_smem, smem_bytes, E, topk);
        }
        return;
    }

    int token = blockIdx.x;
    const float* g = gate + (int64_t)token * E;
    float* tw = topk_weights + (int64_t)token * topk;
    int* ti = topk_ids + (int64_t)token * topk;

    extern __shared__ char smem_raw[];
    float* scores = (float*)smem_raw;
    float* top_vals = scores + E;
    int* top_idxs = (int*)(top_vals + topk);

    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        scores[i] = 1.0f / (1.0f + expf(-g[i]));
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
        for (int k = 0; k < topk; k++) {
            tw[k] = top_vals[k];
            ti[k] = top_idxs[k];
        }
    }
}

extern "C" __global__ void krasis_moe_scatter_sorted_debug_kernel(
    int* __restrict__ sorted_token_ids,
    int* __restrict__ write_offsets,
    const int* __restrict__ topk_ids,
    const int* __restrict__ expert_offsets,
    int M,
    int topk,
    int E,
    int sorted_capacity,
    int write_offsets_capacity,
    KernelDebugErrorRecord* __restrict__ error)
{
    int t = blockIdx.x;
    if (t >= M) {
        return;
    }

    for (int k = 0; k < topk; k++) {
        int eid = topk_ids[t * topk + k];
        if (eid < 0) {
            continue;
        }
        if (eid >= E || eid >= write_offsets_capacity) {
            krasis_record_kernel_debug_failure(error, 2001u, eid, E, t, k);
            return;
        }

        int base = expert_offsets[eid];
        if (base < 0 || base >= sorted_capacity) {
            krasis_record_kernel_debug_failure(error, 2002u, base, sorted_capacity, eid, t);
            return;
        }

        int slot = atomicAdd(&write_offsets[eid], 1);
        if (slot < 0) {
            krasis_record_kernel_debug_failure(error, 2003u, slot, 0, eid, t);
            return;
        }

        int dst = base + slot;
        if (dst < 0 || dst >= sorted_capacity) {
            krasis_record_kernel_debug_failure(error, 2004u, dst, sorted_capacity, eid, slot);
            return;
        }

        sorted_token_ids[dst] = t * topk + k;
    }
}
