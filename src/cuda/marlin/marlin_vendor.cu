// Vendored regular Marlin GEMM kernel for Krasis
// Original source: sgl-project/sglang (Apache 2.0)
//
// This file compiles the regular Marlin INT4/INT8 GEMM kernel.
// Compiled separately from the MoE kernel to avoid C++ ADL conflicts
// between identically-named function templates in different namespaces.

#include "marlin_vendor_common.h"

// Include regular kernel declaration (defines MARLIN_KERNEL_PARAMS for regular Marlin)
#include "kernel.h"

// Include the actual kernel template implementation
#include "marlin_template.h"

// ── Dispatch logic (adapted from gptq_marlin.cuh) ──────────────────────────

namespace device {
namespace marlin {

using namespace ::marlin;

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS) {};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
// sm < 80 not supported
#else

typedef struct {
    int thread_k;
    int thread_n;
    int num_threads;
} thread_config_t;

static thread_config_t small_batch_thread_configs[] = {
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}
};

static thread_config_t large_batch_thread_configs[] = {
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}
};

typedef struct {
    int blocks_per_sm;
    thread_config_t tb_cfg;
} exec_config_t;

static int get_scales_cache_size(
    thread_config_t const& th_config, int prob_m, int prob_n, int prob_k,
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
        load_groups = std::max(load_groups, 32);
        return load_groups * tb_n * 2;
    } else {
        return tb_groups * tb_n * 2 * pipe_stages;
    }
}

static int get_kernel_cache_size(
    thread_config_t const& th_config, int thread_m_blocks,
    int prob_m, int prob_n, int prob_k, int num_bits, int group_size,
    bool has_act_order, bool is_k_full, int has_zp, int is_zp_float,
    bool has_scale2) {
    int pack_factor = 32 / num_bits;
    int tb_k = th_config.thread_k;
    int tb_n = th_config.thread_n;
    int tb_m = thread_m_blocks * 16;
    int sh_a_size = pipe_stages * (tb_m * tb_k) * 2;
    int sh_b_size = pipe_stages * (tb_k * tb_n / pack_factor) * 4;
    int sh_red_size = tb_m * (tb_n + 8);
    int sh_s_size = get_scales_cache_size(
        th_config, prob_m, prob_n, prob_k, num_bits, group_size,
        has_act_order, is_k_full);
    int sh_g_idx_size = (has_act_order && !is_k_full) ? pipe_stages * tb_k / 4 : 0;
    int sh_zp_size = 0;
    if (has_zp) {
        if (is_zp_float) sh_zp_size = sh_s_size;
        else if (num_bits == 4) sh_zp_size = sh_s_size / 4;
        else if (num_bits == 8) sh_zp_size = sh_s_size / 2;
    }
    int sh_s2_size = has_scale2 ? sh_s_size : 0;
    return std::max(sh_b_size, sh_red_size) + sh_a_size + sh_s_size + sh_s2_size + sh_zp_size + sh_g_idx_size;
}

static bool is_valid_config(
    thread_config_t const& th_config, int thread_m_blocks,
    int prob_m, int prob_n, int prob_k, int num_bits, int group_size,
    bool has_act_order, bool is_k_full, int has_zp, int is_zp_float,
    bool has_scale2, int max_shared_mem) {
    if (th_config.thread_k == -1) return false;
    if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) return false;
    if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) return false;
    if (th_config.num_threads < 128) return false;
    int cache_size = get_kernel_cache_size(
        th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits,
        group_size, has_act_order, is_k_full, has_zp, is_zp_float,
        has_scale2);
    return cache_size <= max_shared_mem;
}

// Template dispatch -- only instantiate what we need:
// BF16 compute, U4B8/U8B128 no-zp, and U8 with HQQ float zero points.
#define _GET_IF(W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8, GROUP_BLOCKS, NUM_THREADS, IS_ZP_FLOAT) \
    else if (q_type == W_TYPE && thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS && \
             thread_k_blocks == THREAD_K_BLOCKS && m_block_size_8 == M_BLOCK_SIZE_8 && group_blocks == GROUP_BLOCKS && \
             num_threads == NUM_THREADS && is_zp_float == IS_ZP_FLOAT) { \
        kernel = ::marlin::Marlin<scalar_t, W_TYPE.id(), NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, \
                                  THREAD_K_BLOCKS, M_BLOCK_SIZE_8, pipe_stages, GROUP_BLOCKS, IS_ZP_FLOAT>; \
    }

// Common configs for U4B8 and U8B128 (no act order, no zp)
#define COMMON_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, -1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 2, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 4, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 8, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define COMMON_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS) \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define COMMON_GET_IF(W_TYPE) \
    COMMON_GET_IF_M1(W_TYPE, 8, 8, 256) \
    COMMON_GET_IF_M1(W_TYPE, 8, 4, 128) \
    COMMON_GET_IF_M1(W_TYPE, 4, 8, 128) \
    COMMON_GET_IF_M234(W_TYPE, 16, 4, 256) \
    COMMON_GET_IF_M234(W_TYPE, 8, 4, 128) \
    COMMON_GET_IF_M234(W_TYPE, 4, 8, 128)

#define COMMON_GET_IF_FLOAT_ZP_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, -1, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 2, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 4, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 8, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, true)

#define COMMON_GET_IF_FLOAT_ZP_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS) \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, true)

#define COMMON_GET_IF_FLOAT_ZP(W_TYPE) \
    COMMON_GET_IF_FLOAT_ZP_M1(W_TYPE, 8, 8, 256) \
    COMMON_GET_IF_FLOAT_ZP_M1(W_TYPE, 8, 4, 128) \
    COMMON_GET_IF_FLOAT_ZP_M1(W_TYPE, 4, 8, 128) \
    COMMON_GET_IF_FLOAT_ZP_M234(W_TYPE, 16, 4, 256) \
    COMMON_GET_IF_FLOAT_ZP_M234(W_TYPE, 8, 4, 128) \
    COMMON_GET_IF_FLOAT_ZP_M234(W_TYPE, 4, 8, 128)

template <typename scalar_t>
MarlinFuncPtr get_marlin_kernel(
    const sglang::ScalarType q_type, int thread_m_blocks, int thread_n_blocks,
    int thread_k_blocks, bool m_block_size_8, bool has_act_order, bool has_zp,
    int group_blocks, int num_threads, bool is_zp_float) {

    auto kernel = MarlinDefault;
    if (false) {}

    // U4/U8 are used for native fused HQQ4/HQQ8 prefill with BF16 zero points.
    COMMON_GET_IF(sglang::kU4B8)
    COMMON_GET_IF(sglang::kU8B128)
    COMMON_GET_IF_FLOAT_ZP(sglang::kU4)
    COMMON_GET_IF_FLOAT_ZP(sglang::kU8)

    return kernel;
}

template <typename scalar_t>
exec_config_t determine_exec_config(
    const sglang::ScalarType& q_type, int prob_m, int prob_n, int prob_k,
    int thread_m_blocks, bool m_block_size_8, int num_bits, int group_size,
    bool has_act_order, bool is_k_full, bool has_zp, bool is_zp_float,
    bool has_scale2, int max_shared_mem, int sms) {

    exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
    thread_config_t* thread_configs = thread_m_blocks > 1
        ? large_batch_thread_configs : small_batch_thread_configs;
    int thread_configs_size = thread_m_blocks > 1
        ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
        : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

    for (int i = 0; i < thread_configs_size; i++) {
        thread_config_t th_config = thread_configs[i];
        if (!is_valid_config(th_config, thread_m_blocks, prob_m, prob_n, prob_k,
                            num_bits, group_size, has_act_order, is_k_full,
                            has_zp, is_zp_float, has_scale2, max_shared_mem))
            continue;

        int group_blocks = 0;
        if (!has_act_order) {
            group_blocks = group_size == -1 ? -1 : group_size / 16;
        }

        auto kernel = get_marlin_kernel<scalar_t>(
            q_type, thread_m_blocks, th_config.thread_n / 16,
            th_config.thread_k / 16, m_block_size_8, has_act_order, has_zp,
            group_blocks, th_config.num_threads, is_zp_float);

        if (kernel == MarlinDefault) continue;
        return {1, th_config};
    }
    return exec_cfg;
}

template <typename scalar_t>
void marlin_mm_impl(
    const void* A, const void* B, void* C, void* C_tmp,
    void* s, void* s2, void* zp, void* g_idx, void* perm, void* a_tmp,
    int prob_m, int prob_n, int prob_k, int lda,
    void* workspace, const sglang::ScalarType& q_type,
    bool has_act_order, bool is_k_full, bool has_zp,
    int num_groups, int group_size, int dev, cudaStream_t stream,
    int thread_k_init, int thread_n_init, int sms,
    bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {

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
    const int4* s_ptr = (const int4*)s;
    const uint16_t* s2_ptr = (const uint16_t*)s2;
    const int4* zp_ptr = (const int4*)zp;
    const int* g_idx_ptr = (const int*)g_idx;
    int* locks = (int*)workspace;

    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (max_shared_mem <= 0) return;

    int max_par_local = 16;
    bool has_scale2 = s2_ptr != nullptr && has_zp && is_zp_float;
    if (prob_n <= 4096) max_par_local = 16 * 8;
    int max_shared_mem_new = max_shared_mem;
    int rest_m = prob_m;
    int max_thread_m_blocks = 4;

    while (rest_m) {
        int par_count = rest_m / (max_thread_m_blocks * 16);
        if (par_count > max_par_local) par_count = max_par_local;
        int prob_m_split = par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

        int thread_k = thread_k_init;
        int thread_n = thread_n_init;
        int thread_m_blocks = std::min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
        int m_block_size_8 = prob_m_split <= 8;

        exec_config_t exec_cfg;
        thread_config_t thread_tfg;
        if (thread_k != -1 && thread_n != -1) {
            thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
            exec_cfg = exec_config_t{1, thread_tfg};
        } else {
            exec_cfg = determine_exec_config<scalar_t>(
                q_type, prob_m_split, prob_n, prob_k, thread_m_blocks,
                m_block_size_8, num_bits, group_size, has_act_order, is_k_full,
                has_zp, is_zp_float, has_scale2, max_shared_mem, sms);
            thread_tfg = exec_cfg.tb_cfg;
            if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
                max_thread_m_blocks--;
                continue;
            }
        }

        int num_threads = thread_tfg.num_threads;
        thread_k = thread_tfg.thread_k;
        thread_n = thread_tfg.thread_n;
        int blocks = sms * exec_cfg.blocks_per_sm;
        if (exec_cfg.blocks_per_sm > 1) {
            max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;
        }

        int thread_k_blocks = thread_k / 16;
        int thread_n_blocks = thread_n / 16;

        auto kernel = get_marlin_kernel<scalar_t>(
            q_type, thread_m_blocks, thread_n_blocks, thread_k_blocks,
            m_block_size_8, has_act_order, has_zp, group_blocks,
            num_threads, is_zp_float);

        if (kernel == MarlinDefault) return;  // unsupported config

        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem_new);

        bool part_use_atomic_add = use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

        kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(
            A_ptr, B_ptr, C_ptr, C_tmp_ptr, s_ptr, s2_ptr, zp_ptr, g_idx_ptr,
            num_groups, prob_m_split, prob_n, prob_k, lda, locks,
            part_use_atomic_add, use_fp32_reduce, max_shared_mem_new);

        A_ptr += prob_m_split * (lda / 8);
        C_ptr += prob_m_split * (prob_n / 8);
        rest_m -= prob_m_split;
    }
}

#endif  // __CUDA_ARCH__ >= 800

}  // namespace marlin
}  // namespace device

// ── Extern "C" entry point ──────────────────────────────────────────────────

extern "C" void krasis_marlin_mm_bf16(
    const void* A, const void* B, void* C, void* C_tmp,
    void* s, void* s2, void* zp, void* g_idx, void* perm, void* a_tmp,
    int prob_m, int prob_n, int prob_k, int lda,
    void* workspace,
    const void* q_type_ptr,
    bool has_act_order, bool is_k_full, bool has_zp,
    int num_groups, int group_size, int dev, void* stream_ptr,
    int thread_k, int thread_n, int sms,
    bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {

    const sglang::ScalarType& q_type = *reinterpret_cast<const sglang::ScalarType*>(q_type_ptr);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    device::marlin::marlin_mm_impl<nv_bfloat16>(
        A, B, C, C_tmp, s, s2, zp, g_idx, perm, a_tmp,
        prob_m, prob_n, prob_k, lda, workspace, q_type,
        has_act_order, is_k_full, has_zp,
        num_groups, group_size, dev, stream,
        thread_k, thread_n, sms,
        use_atomic_add, use_fp32_reduce, is_zp_float);
}

// Version info
extern "C" const char* krasis_marlin_version() {
    return "vendored-from-sglang-0.4.0";
}
