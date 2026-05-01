// Krasis FlashAttention-2 vendor shim
// Replaces torch/ATen dependencies for standalone compilation.
// Same pattern as marlin_vendor_common.h.
//
// When KRASIS_FA_VENDOR is defined, this header provides stubs for:
//   - at::PhiloxCudaState (unused -- we don't use dropout in inference)
//   - C10_CUDA_CHECK / C10_CUDA_KERNEL_LAUNCH_CHECK
//   - ATen/cuda/CUDAGeneratorImpl.h

#pragma once

#ifdef KRASIS_FA_VENDOR

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ---- Stub for at::PhiloxCudaState ----
// FA2 stores this in Flash_fwd_params for dropout RNG.
// We never use dropout (inference only), so this is never read.
// We just need a struct of the right size so the params layout matches.
namespace at {
struct PhiloxCudaState {
    uint64_t seed_;
    uint64_t offset_;
    // Additional fields to match upstream size (may have captured/increment fields)
    uint64_t captured_offset_;
    bool captured_;
    PhiloxCudaState() : seed_(0), offset_(0), captured_offset_(0), captured_(false) {}
};
} // namespace at

// ---- Stub for C10_CUDA_CHECK ----
// Replaces c10/cuda/CUDAException.h
#define C10_CUDA_CHECK(call)                                                   \
  do {                                                                         \
    cudaError_t status_ = (call);                                              \
    if (status_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(status_));                                    \
    }                                                                          \
  } while (0)

// ---- Stub for C10_CUDA_KERNEL_LAUNCH_CHECK ----
#define C10_CUDA_KERNEL_LAUNCH_CHECK()                                         \
  do {                                                                         \
    cudaError_t err = cudaPeekAtLastError();                                   \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA kernel launch error (%s:%d): %s\n",               \
              __FILE__, __LINE__, cudaGetErrorString(err));                    \
    }                                                                          \
  } while (0)

#endif // KRASIS_FA_VENDOR
