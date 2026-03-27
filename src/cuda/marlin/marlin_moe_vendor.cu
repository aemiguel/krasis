// Vendored MoE fused Marlin GEMM kernel for Krasis
// Original source: sgl-project/sglang (Apache 2.0)
//
// This file compiles the fused MoE Marlin kernel (MarlinDefault).
// Compiled separately from the regular Marlin kernel to avoid C++ ADL
// (Argument Dependent Lookup) conflicts between identically-named function
// templates in ::marlin and device::marlin_moe namespaces.

#include "marlin_vendor_common.h"

// Include MoE kernel declaration (defines MARLIN_KERNEL_PARAMS for MoE - 24 params)
#include "moe/kernel.h"

// Include MoE kernel template implementation
// This opens namespace device::marlin_moe and defines mma, ldsm, scale, etc.
#include "moe/marlin_template.h"

// ── MoE dispatch logic (extracted from moe_wna16_marlin.cuh) ────────────────
// We inline the dispatch functions here instead of including moe_wna16_marlin.cuh
// because that file has #include <sgl_kernel/tensor.h> for the TVM FFI wrapper.

namespace device {
namespace marlin_moe {

// The MoE template already established `using namespace device::marlin;` in
// moe/marlin_template.h. Since device::marlin doesn't exist in this TU (only
// the regular marlin_vendor.cu defines it), we bring in ::marlin directly.
// This is safe because there's no regular marlin_template.h in this TU,
// so no ADL conflict with helper functions.
using namespace ::marlin;

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS) {};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
// sm < 80 not supported
#else

// permute_cols_kernel for act_order MoE
template <int moe_block_size>
__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr,
    int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    const int32_t* __restrict__ sorted_token_ids_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_past_padded_ptr,
    int size_m, int size_k, int top_k) {
    int num_tokens_past_padded = num_tokens_past_padded_ptr[0];
    int num_moe_blocks = div_ceil(num_tokens_past_padded, moe_block_size);
    int32_t block_sorted_ids[moe_block_size];
    int block_num_valid_tokens = 0;
    int64_t old_expert_id = 0;
    int64_t expert_id = 0;
    int row_stride = size_k * sizeof(half) / 16;

    auto read_moe_block_data = [&](int block_id) {
        block_num_valid_tokens = moe_block_size;
        int4* tmp_block_sorted_ids = reinterpret_cast<int4*>(block_sorted_ids);
        for (int i = 0; i < moe_block_size / 4; i++) {
            tmp_block_sorted_ids[i] = ((int4*)sorted_token_ids_ptr)[block_id * moe_block_size / 4 + i];
        }
        for (int i = 0; i < moe_block_size; i++) {
            if (block_sorted_ids[i] >= size_m * top_k) {
                block_num_valid_tokens = i;
                break;
            }
        }
    };

    auto permute_row = [&](int row) {
        int iters = size_k / default_threads;
        int rest = size_k % default_threads;
        int in_offset = (row / top_k) * row_stride;
        int out_offset = row * row_stride;
        half const* a_row_half = reinterpret_cast<half const*>(a_int4_ptr + in_offset);
        half* out_half = reinterpret_cast<half*>(out_int4_ptr + out_offset);
        int base_k = 0;
        for (int i = 0; i < iters; i++) {
            auto cur_k = base_k + threadIdx.x;
            int src_pos = perm_int_ptr[cur_k];
            out_half[cur_k] = a_row_half[src_pos];
            base_k += default_threads;
        }
        if (rest) {
            if (threadIdx.x < rest) {
                auto cur_k = base_k + threadIdx.x;
                int src_pos = perm_int_ptr[cur_k];
                out_half[cur_k] = a_row_half[src_pos];
            }
        }
    };

    for (int index = blockIdx.x; index < num_moe_blocks; index += gridDim.x) {
        old_expert_id = expert_id;
        int tmp_expert_id = expert_ids_ptr[index];
        if (tmp_expert_id == -1) continue;
        expert_id = tmp_expert_id;
        perm_int_ptr += (expert_id - old_expert_id) * size_k;
        read_moe_block_data(index);
        for (int i = 0; i < block_num_valid_tokens; i++)
            permute_row(block_sorted_ids[i]);
    }
}

typedef struct {
    int thread_k;
    int thread_n;
    int num_threads;
} moe_thread_config_t;

static moe_thread_config_t moe_small_batch_thread_configs[] = {
    {128, 128, 256},
    {64, 128, 128}
};

static moe_thread_config_t moe_large_batch_thread_configs[] = {
    {64, 256, 256},
    {64, 128, 128}
};

typedef struct {
    int blocks_per_sm;
    moe_thread_config_t tb_cfg;
} moe_exec_config_t;

static int moe_get_scales_cache_size(
    moe_thread_config_t const& th_config, int prob_m, int prob_n, int prob_k,
    int num_bits, int group_size, bool has_act_order, bool is_k_full) {
    bool cache_scales_chunk = has_act_order && !is_k_full;
    int tb_n = th_config.thread_n;
    int tb_k = th_config.thread_k;
    int tb_groups;
    if (group_size == -1) tb_groups = 1;
    else if (group_size == 0) tb_groups = div_ceil(tb_k, 32);
    else tb_groups = div_ceil(tb_k, group_size);

    if (cache_scales_chunk) {
        int load_groups = tb_groups * pipe_stages * 2;
        load_groups = max(load_groups, 32);
        return load_groups * tb_n * 2;
    } else {
        return tb_groups * tb_n * 2 * pipe_stages;
    }
}

static int moe_get_kernel_cache_size(
    moe_thread_config_t const& th_config, bool m_block_size_8, int thread_m_blocks,
    int prob_m, int prob_n, int prob_k, int num_bits, int group_size,
    bool has_act_order, bool is_k_full, int has_zp, int is_zp_float) {
    int pack_factor = 32 / num_bits;
    int tb_k = th_config.thread_k;
    int tb_n = th_config.thread_n;
    int tb_m = thread_m_blocks * 16;
    int sh_block_meta_size = tb_m * 4;
    int sh_a_size = pipe_stages * (tb_m * tb_k) * 2;
    int sh_b_size = pipe_stages * (tb_k * tb_n / pack_factor) * 4;
    int sh_red_size = tb_m * (tb_n + 8) * 2;
    int sh_bias_size = tb_n * 2;
    int tmp_size = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
    tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);
    int sh_s_size = moe_get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits, group_size, has_act_order, is_k_full);
    int sh_g_idx_size = has_act_order && !is_k_full ? pipe_stages * tb_k / 4 : 0;
    int sh_zp_size = 0;
    if (has_zp) {
        if (is_zp_float) sh_zp_size = sh_s_size;
        else if (num_bits == 4) sh_zp_size = sh_s_size / 4;
        else if (num_bits == 8) sh_zp_size = sh_s_size / 2;
    }
    return tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size + sh_block_meta_size;
}

static bool moe_is_valid_config(
    moe_thread_config_t const& th_config, bool m_block_size_8, int thread_m_blocks,
    int prob_m, int prob_n, int prob_k, int num_bits, int group_size,
    bool has_act_order, bool is_k_full, int has_zp, int is_zp_float, int max_shared_mem) {
    if (th_config.thread_k == -1) return false;
    if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) return false;
    if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) return false;
    if (th_config.num_threads < 128) return false;
    int cache_size = moe_get_kernel_cache_size(th_config, m_block_size_8, thread_m_blocks,
        prob_m, prob_n, prob_k, num_bits, group_size, has_act_order, is_k_full, has_zp, is_zp_float);
    return cache_size + 512 <= max_shared_mem;
}

// MoE _GET_IF uses s_type_id in addition to w_type_id
#define _MOE_GET_IF(W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8, GROUP_BLOCKS, NUM_THREADS, IS_ZP_FLOAT) \
    else if (q_type == W_TYPE && thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS && \
             thread_k_blocks == THREAD_K_BLOCKS && m_block_size_8 == M_BLOCK_SIZE_8 && group_blocks == GROUP_BLOCKS && \
             num_threads == NUM_THREADS && is_zp_float == IS_ZP_FLOAT) { \
        constexpr auto S_TYPE = (std::is_same<scalar_t, half>::value ? host::kFloat16 : host::kBFloat16); \
        kernel = ::device::marlin_moe::Marlin<scalar_t, W_TYPE.id(), S_TYPE.id(), NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, \
                                  THREAD_K_BLOCKS, M_BLOCK_SIZE_8, pipe_stages, GROUP_BLOCKS, IS_ZP_FLOAT>; \
    }

#define MOE_COMMON_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS) \
    _MOE_GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, -1, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 2, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 4, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 8, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define MOE_COMMON_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS) \
    _MOE_GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false) \
    _MOE_GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define MOE_COMMON_GET_IF(W_TYPE) \
    MOE_COMMON_GET_IF_M1(W_TYPE, 8, 8, 256) \
    MOE_COMMON_GET_IF_M1(W_TYPE, 8, 4, 128) \
    MOE_COMMON_GET_IF_M234(W_TYPE, 16, 4, 256) \
    MOE_COMMON_GET_IF_M234(W_TYPE, 8, 4, 128)

template <typename scalar_t>
MarlinFuncPtr moe_get_marlin_kernel(
    const sglang::ScalarType q_type, int thread_m_blocks, int thread_n_blocks,
    int thread_k_blocks, bool m_block_size_8, bool has_act_order, bool has_zp,
    int group_blocks, int num_threads, bool is_zp_float) {
    auto kernel = MarlinDefault;
    if (false) {}

    MOE_COMMON_GET_IF(sglang::kU4B8)
    MOE_COMMON_GET_IF(sglang::kU8B128)

    return kernel;
}

template <typename scalar_t>
moe_exec_config_t moe_determine_exec_config(
    const sglang::ScalarType& q_type, int prob_m, int prob_n, int prob_k,
    int thread_m_blocks, bool m_block_size_8, int num_bits, int group_size,
    bool has_act_order, bool is_k_full, bool has_zp, bool is_zp_float,
    int max_shared_mem) {
    moe_exec_config_t exec_cfg = {1, moe_thread_config_t{-1, -1, -1}};
    moe_thread_config_t* thread_configs = thread_m_blocks > 1
        ? moe_large_batch_thread_configs : moe_small_batch_thread_configs;
    int thread_configs_size = thread_m_blocks > 1
        ? sizeof(moe_large_batch_thread_configs) / sizeof(moe_thread_config_t)
        : sizeof(moe_small_batch_thread_configs) / sizeof(moe_thread_config_t);

    int count = 0;
    constexpr int device_max_reg_size = 255 * 1024;
    for (int i = 0; i < thread_configs_size; i++) {
        moe_thread_config_t th_config = thread_configs[i];
        if (!moe_is_valid_config(th_config, m_block_size_8, thread_m_blocks,
                prob_m, prob_n, prob_k, num_bits, group_size,
                has_act_order, is_k_full, has_zp, is_zp_float, max_shared_mem))
            continue;

        int cache_size = moe_get_kernel_cache_size(th_config, m_block_size_8, thread_m_blocks,
            prob_m, prob_n, prob_k, num_bits, group_size, has_act_order, is_k_full, has_zp, is_zp_float);

        int group_blocks = 0;
        if (!has_act_order) {
            group_blocks = group_size == -1 ? -1 : (group_size / 16);
        }

        auto kernel = moe_get_marlin_kernel<scalar_t>(
            q_type, thread_m_blocks, th_config.thread_n / 16, th_config.thread_k / 16,
            m_block_size_8, has_act_order, has_zp, group_blocks,
            th_config.num_threads, is_zp_float);

        if (kernel == MarlinDefault) continue;

        if (thread_m_blocks > 1) {
            exec_cfg = {1, th_config};
            break;
        } else {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, kernel);
            int reg_size = max(attr.numRegs, 1) * th_config.num_threads * 4;
            int allow_count = min(device_max_reg_size / reg_size, max_shared_mem / (cache_size + 1024));
            allow_count = max(min(allow_count, 4), 1);
            if (allow_count > count) {
                count = allow_count;
                exec_cfg = {count, th_config};
            }
        }
    }
    return exec_cfg;
}

template <typename scalar_t>
void moe_marlin_mm_impl(
    const void* A, const void* B, void* C, void* C_tmp,
    void* b_bias, void* s, void* s2, void* zp, void* g_idx,
    void* perm, void* a_tmp, void* sorted_token_ids, void* expert_ids,
    void* num_tokens_past_padded, void* topk_weights,
    int moe_block_size, int top_k, bool mul_topk_weights, bool is_ep,
    int prob_m, int prob_n, int prob_k,
    void* workspace, const sglang::ScalarType& q_type,
    bool has_bias, bool has_act_order, bool is_k_full, bool has_zp,
    int num_groups, int group_size, int dev, cudaStream_t stream,
    int thread_k, int thread_n, int sms,
    bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float,
    const void* B_expert_ptrs, const void* S_expert_ptrs) {

    int thread_m_blocks = div_ceil(moe_block_size, 16);
    bool m_block_size_8 = moe_block_size == 8;

    int group_blocks = 0;
    if (has_act_order) {
        if (is_k_full) {
            group_blocks = group_size / 16;
        }
    } else {
        if (group_size == -1) group_blocks = -1;
        else group_blocks = group_size / 16;
    }

    int num_bits = q_type.size_bits();
    const int4* A_ptr = (const int4*)A;
    const int4* B_ptr = (const int4*)B;
    int4* C_ptr = (int4*)C;
    int4* C_tmp_ptr = (int4*)C_tmp;
    const int4* bias_ptr = (const int4*)b_bias;
    const int4* s_ptr = (const int4*)s;
    const uint16_t* s2_ptr = (const uint16_t*)s2;
    const int4* zp_ptr = (const int4*)zp;
    const int* g_idx_ptr = (const int*)g_idx;
    const int* perm_ptr = (const int*)perm;
    int4* a_tmp_ptr = (int4*)a_tmp;
    const int32_t* sorted_token_ids_ptr = (const int32_t*)sorted_token_ids;
    const int32_t* expert_ids_ptr = (const int32_t*)expert_ids;
    const int32_t* num_tokens_past_padded_ptr = (const int32_t*)num_tokens_past_padded;
    const float* topk_weights_ptr = (const float*)topk_weights;
    int* locks = (int*)workspace;

    if (has_act_order) {
        auto perm_kernel = permute_cols_kernel<8>;
        if (moe_block_size == 8) {}
        else if (moe_block_size == 16) perm_kernel = permute_cols_kernel<16>;
        else if (moe_block_size == 32) perm_kernel = permute_cols_kernel<32>;
        else if (moe_block_size == 48) perm_kernel = permute_cols_kernel<48>;
        else if (moe_block_size == 64) perm_kernel = permute_cols_kernel<64>;

        perm_kernel<<<sms, default_threads, 0, stream>>>(
            A_ptr, perm_ptr, a_tmp_ptr, sorted_token_ids_ptr, expert_ids_ptr,
            num_tokens_past_padded_ptr, prob_m, prob_k, top_k);
        A_ptr = a_tmp_ptr;
        prob_m = prob_m * top_k;
        top_k = 1;
        if (is_k_full) has_act_order = false;
    }

    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (max_shared_mem <= 0) return;

    moe_exec_config_t exec_cfg;
    moe_thread_config_t thread_tfg;
    if (thread_k != -1 && thread_n != -1) {
        thread_tfg = moe_thread_config_t{thread_k, thread_n, default_threads};
        exec_cfg = moe_exec_config_t{1, thread_tfg};
    } else {
        exec_cfg = moe_determine_exec_config<scalar_t>(
            q_type, prob_m, prob_n, prob_k, thread_m_blocks, m_block_size_8,
            num_bits, group_size, has_act_order, is_k_full, has_zp, is_zp_float,
            max_shared_mem);
        thread_tfg = exec_cfg.tb_cfg;
    }

    int num_threads = thread_tfg.num_threads;
    thread_k = thread_tfg.thread_k;
    thread_n = thread_tfg.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1)
        max_shared_mem = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    auto kernel = moe_get_marlin_kernel<scalar_t>(
        q_type, thread_m_blocks, thread_n_blocks, thread_k_blocks,
        m_block_size_8, has_act_order, has_zp, group_blocks,
        num_threads, is_zp_float);

    if (kernel == MarlinDefault) {
        fprintf(stderr, "[DIAG Marlin MoE] KERNEL NOT FOUND! thread_m_blocks=%d thread_n_blocks=%d thread_k_blocks=%d "
                "m_block_size_8=%d num_bits=%d group_blocks=%d num_threads=%d prob_m=%d prob_n=%d prob_k=%d\n",
                thread_m_blocks, thread_n_blocks, thread_k_blocks,
                (int)m_block_size_8, num_bits, group_blocks, num_threads, prob_m, prob_n, prob_k);
        return;
    }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);

    kernel<<<blocks, num_threads, max_shared_mem, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, s_ptr, s2_ptr, zp_ptr, g_idx_ptr,
        sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
        topk_weights_ptr, top_k, mul_topk_weights, is_ep, num_groups, prob_m,
        prob_n, prob_k, locks, has_bias, use_atomic_add, use_fp32_reduce, max_shared_mem,
        (const int64_t*)B_expert_ptrs, (const int64_t*)S_expert_ptrs);
}

#endif  // __CUDA_ARCH__ >= 800

}  // namespace marlin_moe
}  // namespace device

// ── Extern "C" entry point ──────────────────────────────────────────────────

extern "C" void krasis_marlin_moe_mm_bf16(
    const void* A, const void* B, void* C, void* C_tmp,
    void* b_bias, void* s, void* s2, void* zp, void* g_idx,
    void* perm, void* a_tmp, void* sorted_token_ids, void* expert_ids,
    void* num_tokens_past_padded, void* topk_weights,
    int moe_block_size, int top_k, bool mul_topk_weights, bool is_ep,
    int prob_m, int prob_n, int prob_k,
    void* workspace,
    const void* q_type_ptr,
    bool has_bias, bool has_act_order, bool is_k_full, bool has_zp,
    int num_groups, int group_size, int dev, void* stream_ptr,
    int thread_k, int thread_n, int sms,
    bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float,
    const void* B_expert_ptrs, const void* S_expert_ptrs) {

    const sglang::ScalarType& q_type = *reinterpret_cast<const sglang::ScalarType*>(q_type_ptr);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    device::marlin_moe::moe_marlin_mm_impl<nv_bfloat16>(
        A, B, C, C_tmp, b_bias, s, s2, zp, g_idx, perm, a_tmp,
        sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights,
        moe_block_size, top_k, mul_topk_weights, is_ep,
        prob_m, prob_n, prob_k, workspace, q_type,
        has_bias, has_act_order, is_k_full, has_zp,
        num_groups, group_size, dev, stream,
        thread_k, thread_n, sms,
        use_atomic_add, use_fp32_reduce, is_zp_float,
        B_expert_ptrs, S_expert_ptrs);
}
