# Vendored FlashAttention Kernels

## Origin

### FlashAttention-2 (fa2/)
- Author: Tri Dao (Stanford / Together AI)
- Repo: https://github.com/Dao-AILab/flash-attention
- Tag: v2.8.3
- License: BSD 3-Clause (see LICENSE in this directory)
- Vendored date: 2026-03-26

### CUTLASS Headers (cutlass/)
- Source: https://github.com/NVIDIA/cutlass
- Commit: dc4817921edda44a549197ff3a9dcf5df0636e7b (v4.0)
- License: BSD 3-Clause (NVIDIA)
- This is the exact CUTLASS version FA2 v2.8.3 uses as a submodule.
- Header-only library; only instantiated templates are compiled.

## Why vendored

Eliminates flash-attn as a pip runtime dependency. The kernels are compiled
into libkrasis_flash_attn.so during our build (via build.rs) and loaded at
runtime via dlopen from Rust.

Supply chain security: we control exactly what code runs. No external pip
package that could be silently modified (cf. LiteLLM supply chain attack).

## What we changed

Modifications to make FA2 compile standalone without torch/ATen, plus
FP8 KV cache support:

- Added KRASIS_FA_VENDOR preprocessor guard in flash.h (skips ATen include)
- Added KRASIS_FA_VENDOR guard in flash_fwd_launch_template.h (skips c10 include)
- Created flash_attn_vendor.h with stubs for:
  - at::PhiloxCudaState (zero-initialized, unused in inference -- no dropout)
  - C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK macros
- Created flash_attn_vendor.cu with extern "C" entry points:
  - krasis_flash_attn_fwd_bf16 (BF16 Q/K/V)
  - krasis_flash_attn_fwd_bf16q_fp8kv (BF16 Q, FP8 E4M3 K/V)
- Added FP8 KV load path to flash_fwd_kernel.h (Is_KV_FP8 template parameter)
  - Loads FP8 from global memory, converts to BF16 in registers, stores to
    shared memory. Eliminates external dequant step for cross-chunk attention.
  - Halves KV global memory bandwidth (1 byte/elem vs 2 bytes/elem).

All upstream kernel logic (softmax, masking, MMA, tiling) is UNMODIFIED.
The FP8 path only changes the global-to-shared-memory load for K/V.

## FA3 (removed)

FA3 (CUTLASS DSL kernels for sm_90/sm_100) was previously vendored but
has been removed. It does not support sm_120 (RTX 5090) or sm_86 (RTX 3090).
FA2 covers all our target GPUs via sm_80 binary compatibility.

## Files

Upstream (with conditional include patches only):
  flash.h                           Core types and params struct
  flash_fwd_kernel.h                Forward pass kernel template (+ FP8 KV path)
  flash_fwd_launch_template.h       Template dispatch and tile selection
  kernel_traits.h                   Tile sizes, shared memory layout
  softmax.h                         Online softmax
  mask.h                            Causal/window/local masking
  utils.h                           CUDA utilities (+ FP8-to-BF16 copy)
  block_info.h                      Block indexing
  hardware_info.h                   GPU capability detection
  static_switch.h                   Compile-time dispatch macros
  namespace_config.h                Namespace management
  alibi.h, dropout.h, rotary.h     Not used by us but required by templates
  philox.cuh, philox_unpack.cuh    Dropout RNG (not used, required by templates)

  Template instantiations (BF16 Q, BF16 K/V forward only):
  flash_fwd_hdim{64,96,128,192,256}_bf16_{causal_,}sm80.cu

  Template instantiations (BF16 Q, FP8 K/V forward only):
  flash_fwd_hdim128_bf16q_fp8kv_causal_sm80.cu
  flash_fwd_hdim128_bf16q_fp8kv_sm80.cu

Ours:
  flash_attn_vendor.h               Standalone stubs replacing torch/ATen
  flash_attn_vendor.cu              extern "C" entry points for Rust FFI

Reference (not compiled):
  flash_api_reference.cpp           Original torch-based API for reference

## Updating from upstream

1. Pull updated files from Dao-AILab/flash-attention (specific tag)
2. Re-apply KRASIS_FA_VENDOR guards in flash.h and flash_fwd_launch_template.h
3. Re-apply Is_KV_FP8 modifications to flash_fwd_kernel.h
4. Verify flash_attn_vendor.h stubs still match at::PhiloxCudaState layout
5. Run ./dev build
6. Run reference test to validate correctness
7. Run benchmark to verify no regression
