/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "namespace_config.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__forceinline__ __device__ uint32_t relu2(const uint32_t x);

template<>
__forceinline__ __device__ uint32_t relu2<cutlass::half_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
#else
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela;\n" \
        "\t set.gtu.u32.f16x2 sela, %1, %2;\n" \
        "\t and.b32 %0, sela, %1;\n" 
        "}\n" : "=r"(res) : "r"(x), "r"(zero));
#endif
    return res;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<>
__forceinline__ __device__ uint32_t relu2<cutlass::bfloat16_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
    asm volatile("max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
    return res;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

template<typename T>
__forceinline__ __device__ uint32_t convert_relu2(const float2 x);

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::half_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::bfloat16_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator> 
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_dropout(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void relu_(Tensor<Engine, Layout> &tensor) {
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
    using value_t = typename Engine::value_type;
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_uint32 = recast<uint32_t>(tensor);
    #pragma unroll
    for (int i = 0; i < size(tensor_uint32); ++i) {
        tensor_uint32(i) = relu2<value_t>(tensor_uint32(i));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// On SM80 and above, we can fuse fp32 -> fp16/bf16 conversion and relu into 1 instruction
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type_relu(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    static_assert(std::is_same_v<To_type, cutlass::half_t> || std::is_same_v<To_type, cutlass::bfloat16_t>);
    static_assert(std::is_same_v<float, From_type>);
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_float2 = recast<float2>(tensor);
    Tensor out_uint32 = make_tensor<uint32_t>(tensor_float2.layout());
    #pragma unroll
    for (int i = 0; i < size(out_uint32); ++i) {
        out_uint32(i) = convert_relu2<To_type>(tensor_float2(i));
    }
    Tensor out = make_tensor(make_rmem_ptr<To_type>(out_uint32.data()), tensor.layout());
#else
    Tensor out = FLASH_NAMESPACE::convert_type<To_type>(tensor);
    FLASH_NAMESPACE::relu_(out);
#endif
    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
    // TD [2023-04-13]: Strange that the code below can cause race condition.
    // I think it's because the copies are under an if statement.
    // if (Is_even_K) {
    //     #pragma unroll
    //     for (int m = 0; m < size<1>(S); ++m) {
    //         if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
    //             copy(tiled_copy, S(_, m, _), D(_, m, _));
    //         } else if (Clear_OOB_MN) {
    //             clear(D(_, m, _));
    //         }
    //     }
    // } else {  // It's slightly faster in this case if iterate over K first
    //     #pragma unroll
    //     for (int k = 0; k < size<2>(S); ++k) {
    //         if (predicate_K(k)) {
    //             #pragma unroll
    //             for (int m = 0; m < size<1>(S); ++m) {
    //                 if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
    //                     copy(tiled_copy, S(_, m, k), D(_, m, k));
    //                 } else if (Clear_OOB_MN) {
    //                     clear(D(_, m, k));
    //                 }
    //             }
    //         } else if (Clear_OOB_K) {  // There's no case where !Clear_OOB_K && Clear_OOB_MN
    //             if (Clear_OOB_MN || Is_even_MN) {
    //                 clear(D(_, _, k));
    //             } else {
    //                 #pragma unroll
    //                 for (int m = 0; m < size<1>(S); ++m) {
    //                     if (!(Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN)) {
    //                         clear(D(_, m, k));
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S,
                                      Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                      Tensor<Engine3, Layout3> const &predicate_K,
                                      const int max_MN=0, const int min_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, max_MN = %d, min_MN = %d\n", blockIdx.y, max_MN, min_MN); }
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("Inner loop, blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_softcap(Tensor<Engine, Layout> &tensor, const float softcap){
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = cutlass::fast_tanh(tensor(i) * softcap);
    }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void calculate_dtanh(Tensor<Engine0, Layout0> &src_tensor, Tensor<Engine1, Layout1> &dst_tensor, const float softcap){
    #pragma unroll
    for (int i = 0; i < size(src_tensor); ++i) {
        dst_tensor(i) = (1.f - (src_tensor(i) * src_tensor(i))) * softcap;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

// FP8 KV cache: cp.async-based pipelined loading.
//
// Two-phase approach that restores async pipelining for FP8:
//   Phase 1: cp.async copies raw FP8 bytes from global memory to a linear staging
//            buffer in shared memory. This is ASYNC — threads continue executing.
//   Phase 2: After cp.async completes, all threads cooperatively convert
//            FP8 staging → BF16 in the swizzled KV shared memory buffer.
//            This is pure smem→smem, ~0.2us for 8K elements.
//
// The old copy_fp8_to_bf16_smem did synchronous global loads, which blocked
// all threads on memory latency with zero compute overlap. This new approach
// lets the cp.async overlap with gemm computation (same as the BF16 path).

// Phase 1: Issue cp.async to copy FP8 bytes from global memory to staging area.
// The staging area has a simple linear layout (row-major, no swizzle).
// Call cp_async_fence() after this, then cp_async_wait<0>() + __syncthreads()
// before calling expand_fp8_staging_to_bf16_smem.
template <int kBlockN, int kHeadDim, int kNThreads>
__forceinline__ __device__ void cp_async_fp8_to_staging(
    void* __restrict__ smem_staging,          // Linear FP8 staging buffer in smem
    const void* __restrict__ gmem_fp8_base,   // FP8 E4M3 global memory base for this head
    int row_offset,                           // Starting row in global memory
    int row_stride,                           // Stride between rows in global (elements = bytes for FP8)
    int valid_rows                            // Number of valid rows to load
) {
    constexpr int BYTES_PER_COPY = 16;  // 128-bit cp.async
    constexpr int COPIES_PER_ROW = kHeadDim / BYTES_PER_COPY;
    static_assert(kHeadDim % BYTES_PER_COPY == 0, "kHeadDim must be multiple of 16");
    constexpr int TOTAL_COPIES = kBlockN * COPIES_PER_ROW;

    const char* src_base = reinterpret_cast<const char*>(gmem_fp8_base);
    char* dst_base = reinterpret_cast<char*>(smem_staging);

    #pragma unroll
    for (int copy_idx = threadIdx.x; copy_idx < TOTAL_COPIES; copy_idx += kNThreads) {
        int row = copy_idx / COPIES_PER_ROW;
        int col_chunk = copy_idx % COPIES_PER_ROW;

        if (row < valid_rows) {
            int gmem_byte_offset = (row_offset + row) * row_stride + col_chunk * BYTES_PER_COPY;
            int smem_byte_offset = row * kHeadDim + col_chunk * BYTES_PER_COPY;

            uint32_t smem_addr = __cvta_generic_to_shared(dst_base + smem_byte_offset);

            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], %2;\n"
                :: "r"(smem_addr), "l"(src_base + gmem_byte_offset), "n"(BYTES_PER_COPY)
            );
        } else {
            // Zero-fill OOB rows in staging (will become zeros after expand)
            int smem_byte_offset = row * kHeadDim + col_chunk * BYTES_PER_COPY;
            uint4* dst = reinterpret_cast<uint4*>(dst_base + smem_byte_offset);
            *dst = make_uint4(0, 0, 0, 0);
        }
    }
}

// Phase 2: Cooperatively convert FP8 staging → BF16 in swizzled shared memory.
// Must be called after cp_async_wait<0>() + __syncthreads() to ensure staging is ready.
// Follow with __syncthreads() before reading the BF16 smem in a gemm.
template <typename SmemLayout, int kBlockN, int kHeadDim, int kNThreads, bool Clear_OOB = true>
__forceinline__ __device__ void expand_fp8_staging_to_bf16_smem(
    cutlass::bfloat16_t* __restrict__ smem_bf16,  // Swizzled BF16 destination
    const void* __restrict__ smem_staging,         // Linear FP8 staging source
    SmemLayout smem_layout,
    int valid_rows,
    int valid_cols
) {
    auto dst = cute::make_tensor(cute::make_smem_ptr(smem_bf16), smem_layout);
    const __nv_fp8_e4m3* staging = reinterpret_cast<const __nv_fp8_e4m3*>(smem_staging);

    // Vectorized: process 16 FP8 elements per iteration
    constexpr int ELEMS_PER_VEC = 16;
    static_assert(kHeadDim % ELEMS_PER_VEC == 0);
    constexpr int total_vecs = kBlockN * (kHeadDim / ELEMS_PER_VEC);

    #pragma unroll
    for (int vidx = threadIdx.x; vidx < total_vecs; vidx += kNThreads) {
        int row = vidx / (kHeadDim / ELEMS_PER_VEC);
        int col_base = (vidx % (kHeadDim / ELEMS_PER_VEC)) * ELEMS_PER_VEC;

        // Vectorized read from staging (linear layout, 16 bytes aligned)
        const uint4* src_vec = reinterpret_cast<const uint4*>(
            &staging[row * kHeadDim + col_base]);
        uint4 fp8_vec = *src_vec;
        const unsigned char* fp8_bytes = reinterpret_cast<const unsigned char*>(&fp8_vec);

        if (row < valid_rows && col_base + ELEMS_PER_VEC <= valid_cols) {
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                __nv_fp8_e4m3 fp8_val = *reinterpret_cast<const __nv_fp8_e4m3*>(&fp8_bytes[k]);
                float fval = float(fp8_val);
                dst(row, col_base + k) = cutlass::bfloat16_t(__float2bfloat16(fval));
            }
        } else if (Clear_OOB) {
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                int col = col_base + k;
                if (row < valid_rows && col < valid_cols) {
                    __nv_fp8_e4m3 fp8_val = *reinterpret_cast<const __nv_fp8_e4m3*>(&fp8_bytes[k]);
                    float fval = float(fp8_val);
                    dst(row, col) = cutlass::bfloat16_t(__float2bfloat16(fval));
                } else {
                    dst(row, col) = cutlass::bfloat16_t(0.0f);
                }
            }
        }
    }
}

// Legacy synchronous FP8 copy (kept for reference, no longer used in hot path)
template <typename SmemLayout, int kBlockN, int kHeadDim, int kNThreads, bool Clear_OOB = true>
__forceinline__ __device__ void copy_fp8_to_bf16_smem(
    const void* __restrict__ gmem_fp8_base,
    cutlass::bfloat16_t* __restrict__ smem_bf16,
    SmemLayout smem_layout,
    int row_offset,
    int row_stride,
    int valid_rows,
    int valid_cols
) {
    auto smem_tensor = cute::make_tensor(cute::make_smem_ptr(smem_bf16), smem_layout);
    const auto* src = reinterpret_cast<const __nv_fp8_e4m3*>(gmem_fp8_base);

    constexpr int ELEMS_PER_VEC = 16;
    static_assert(kHeadDim % ELEMS_PER_VEC == 0, "kHeadDim must be multiple of 16");
    constexpr int total_vecs = kBlockN * (kHeadDim / ELEMS_PER_VEC);

    #pragma unroll
    for (int vidx = threadIdx.x; vidx < total_vecs; vidx += kNThreads) {
        int row = vidx / (kHeadDim / ELEMS_PER_VEC);
        int col_base = (vidx % (kHeadDim / ELEMS_PER_VEC)) * ELEMS_PER_VEC;

        if (row < valid_rows && col_base + ELEMS_PER_VEC <= valid_cols) {
            int gmem_byte_offset = (row_offset + row) * row_stride + col_base;
            uint4 fp8_vec = *reinterpret_cast<const uint4*>(&src[gmem_byte_offset]);
            const unsigned char* fp8_bytes = reinterpret_cast<const unsigned char*>(&fp8_vec);
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                __nv_fp8_e4m3 fp8_val = *reinterpret_cast<const __nv_fp8_e4m3*>(&fp8_bytes[k]);
                float fval = float(fp8_val);
                smem_tensor(row, col_base + k) = cutlass::bfloat16_t(__float2bfloat16(fval));
            }
        } else if (Clear_OOB) {
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                int col = col_base + k;
                if (row < valid_rows && col < valid_cols) {
                    int gmem_idx = (row_offset + row) * row_stride + col;
                    float fval = float(src[gmem_idx]);
                    smem_tensor(row, col) = cutlass::bfloat16_t(__float2bfloat16(fval));
                } else {
                    smem_tensor(row, col) = cutlass::bfloat16_t(0.0f);
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace FLASH_NAMESPACE
