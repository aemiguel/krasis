# Vendored Marlin Kernels

## Origin

These files are vendored from the sglang project's Marlin GEMM kernels.

Original author: Elias Frantar (IST-DASLab)
Original repo: https://github.com/IST-DASLab/marlin
Integrated by: Neural Magic (modifications for GPTQ/AWQ support)
Vendored from: https://github.com/sgl-project/sglang
  - Regular Marlin: sgl-kernel/csrc/gemm/marlin/
  - Fused MoE Marlin: python/sglang/jit_kernel/csrc/gemm/marlin_moe_wna16/
License: Apache 2.0 (see LICENSE in this directory)
Vendored date: 2026-03-23

## Why vendored

Eliminates sgl_kernel as a pip runtime dependency. The kernels are compiled
into libkrasis_marlin.so during our build (via build.rs) and loaded at runtime
via dlopen from Rust and ctypes from Python (AWQ calibration only).

This gives us zero pip dependencies in the execution path. Python is only
needed for startup (config parsing, weight loading, cache build).

## What we changed

Minimal modifications to make the files compile standalone without torch/ATen:

- Added KRASIS_MARLIN_VENDOR preprocessor guard to skip torch includes
- Stubbed TORCH_CHECK with a no-op macro
- Fixed include paths for standalone directory structure (../marlin/ references)
- Added #pragma once to dequant.h (missing in upstream)
- Added extern "C" entry points (marlin_vendor.cu, marlin_moe_vendor.cu)
- Created marlin_vendor_common.h with shared stubs replacing torch/ATen

All upstream kernel logic (marlin_template.h, dequant.h, dispatch in
gptq_marlin.cuh and moe_wna16_marlin.cuh) is unmodified. The actual GEMM
computation is identical to sgl_kernel.

## Files

Upstream (with include path patches only):
  marlin_template.h         Regular Marlin kernel template (~1850 lines)
  gptq_marlin.cuh           Regular Marlin dispatch logic
  dequant.h                 INT4/INT8 dequantization helpers
  marlin_dtypes.cuh         BF16/FP16 type definitions
  marlin.cuh                Constants and helpers
  kernel.h                  MARLIN_KERNEL_PARAMS macro (regular)
  scalar_type.hpp           ScalarType class
  moe/marlin_template.h     Fused MoE kernel template (~2200 lines)
  moe/moe_wna16_marlin.cuh  Fused MoE dispatch logic
  moe/kernel.h              MARLIN_KERNEL_PARAMS macro (MoE)

Ours:
  marlin_vendor.cu          Regular Marlin extern "C" entry point
  marlin_moe_vendor.cu      Fused MoE extern "C" entry point
  marlin_vendor_common.h    Shared stubs replacing torch/ATen dependencies
  LICENSE                   Apache 2.0 license text
  VENDORED.md               This file

## Updating from upstream

1. Pull updated files from sgl-project/sglang (paths listed above)
2. Re-apply the KRASIS_MARLIN_VENDOR include guards (grep for it in current files)
3. Re-apply #pragma once to dequant.h if missing
4. Verify build: ./dev build
5. Run kernel tests: ./dev test-kernels
6. Run model benchmark: ./dev benchmark qcn

## Contributing back

If we fix a bug in the vendored kernels, we should contribute the fix back
to upstream (sgl-project/sglang) via pull request. The upstream repo benefits
from bug fixes and we reduce future merge friction when pulling updates.

Fixes to our wrapper files (marlin_vendor.cu, marlin_vendor_common.h) are
Krasis-specific and don't need to go upstream.
