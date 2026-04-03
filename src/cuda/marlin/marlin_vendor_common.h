// Common infrastructure for vendored Marlin GEMM kernels.
// Included by both marlin_vendor.cu (regular Marlin) and marlin_moe_vendor.cu (MoE Marlin).
// Provides: TORCH_CHECK stub, sglang::ScalarType, host namespace, ::marlin namespace
// with constants/types/async functions, and marlin_dtypes.cuh (ScalarType<T> template).
#pragma once

// Guard to skip torch-dependent includes in vendored headers
#define KRASIS_MARLIN_VENDOR

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>

// ── Stub out torch/sglang dependencies ──────────────────────────────────────

// Replace TORCH_CHECK with a no-op for constexpr contexts.
#define TORCH_CHECK(...) ((void)0)

// Include the vendored scalar_type.hpp (now uses our TORCH_CHECK stub)
#include "scalar_type.hpp"

// The MoE kernel dispatch uses `host::` as a namespace for ScalarType and
// utilities. We define `host` as an alias to `sglang`.
namespace sglang {

template <typename... Args>
inline void RuntimeCheck(bool cond, Args... args) {
    if (!cond) {
        fprintf(stderr, "Marlin RuntimeCheck failed\n");
    }
}

template <typename... Args>
inline void Panic(Args... args) {
    fprintf(stderr, "Marlin Panic\n");
}

inline void RuntimeDeviceCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Marlin CUDA error: %s\n", cudaGetErrorString(err));
    }
}

}  // namespace sglang

namespace host = sglang;

// ── ::marlin namespace with infrastructure ──────────────────────────────────

#ifndef _marlin_cuh_override
#define _marlin_cuh_override
#define MARLIN_CUH_INCLUDED
#endif

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME {

static constexpr int default_threads = 256;
static constexpr int pipe_stages = 4;
static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;
static constexpr int max_thread_n = 256;
static constexpr int tile_size = 16;
static constexpr int max_par = 16;
static constexpr int repack_stages = 8;
static constexpr int repack_threads = 256;
static constexpr int tile_k_size = tile_size;
static constexpr int tile_n_size = tile_k_size * 4;

template <typename T, int n>
struct Vec {
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;

constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#else

__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int)pred), "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

#endif

}  // namespace MARLIN_NAMESPACE_NAME

// Include type definitions (ScalarType<half>, ScalarType<nv_bfloat16>)
#include "marlin_dtypes.cuh"
