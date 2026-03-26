// Krasis: Philox RNG unpack stub.
// Original includes ATen/cuda/detail/UnpackRaw.cuh for dropout RNG.
// We disable dropout entirely (FLASHATTENTION_DISABLE_DROPOUT) so this is
// never called at runtime, but the kernel template still references it
// unconditionally. We provide a dummy implementation that returns zeros.

#pragma once

#ifdef KRASIS_FA_VENDOR

#include <cstdint>
#include <tuple>

namespace at { namespace cuda { namespace philox {

// Stub: returns (seed=0, offset=0). Never called because FLASHATTENTION_DISABLE_DROPOUT
// forces Is_dropout=false, so the dropout path is dead code. But nvcc still
// needs to parse it.
template<typename T>
__host__ __device__ inline std::tuple<uint64_t, uint64_t> unpack(const T& /*philox_args*/) {
    return std::make_tuple(0ULL, 0ULL);
}

}}} // namespace at::cuda::philox

#else
#include <ATen/cuda/detail/UnpackRaw.cuh>
#endif
