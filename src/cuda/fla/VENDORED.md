# Vendored: Flash Linear Attention (FLA) v0.4.2

## Origin
- Repository: https://github.com/fla-org/flash-linear-attention
- Authors: Songlin Yang, Yu Zhang
- License: MIT
- Version: 0.4.2

## What's Vendored
Forward-only subset of the Gated DeltaNet chunked attention algorithm.
This is used by QCN's 36 linear attention layers during prefill.

### Files (from fla-core package):
- ops/gated_delta_rule/chunk.py — Top-level forward orchestration
- ops/gated_delta_rule/wy_fast.py — WY representation kernel
- ops/common/chunk_delta_h.py — State recurrence kernel
- ops/common/chunk_o.py — Output computation kernel
- ops/common/chunk_scaled_dot_kkt.py — Attention matrix kernel
- ops/utils/ — cumsum, solve_tril, index helpers
- ops/backends/ — Backend dispatch system
- modules/l2norm.py — L2 normalization kernel
- utils.py — Shared utilities

### Files NOT vendored (backward-only, unused for inference):
- All *_bwd_* functions and kernels
- fused_recurrent.py (not used by QCN)
- naive.py (reference implementation)
- ops/cp/ (context parallelism, stubs provided)

## Modifications
- ops/__init__.py: Stripped to only import gated_delta_rule
- ops/gated_delta_rule/__init__.py: Stripped to only import chunk forward
- ops/utils/__init__.py: Stripped to only needed functions
- modules/__init__.py: Stripped to only l2norm
- ops/cp/__init__.py: Stub with FLACPContext placeholder
- ops/cp/chunk_delta_h.py: Stub (raises NotImplementedError)

## Build Process
1. compile_kernels.py imports this code, runs forward pass to trigger Triton JIT
2. Extracts compiled cubins from Triton cache
3. Generates fla_vendor.cu with embedded cubins and extern "C" wrappers
4. nvcc compiles fla_vendor.cu into libkrasis_fla.so
5. Rust loads via dlopen at runtime (no Python/Triton dependency)

## How to Update
1. Check new version at https://github.com/fla-org/flash-linear-attention/releases
2. Copy updated files from the forward-only subset listed above
3. Re-apply modifications to __init__.py files
4. Run compile_kernels.py to regenerate cubins and C wrapper
5. Run reference test to validate correctness
6. Run benchmark to verify no regression
