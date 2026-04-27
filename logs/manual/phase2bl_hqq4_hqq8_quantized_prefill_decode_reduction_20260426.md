# Phase 2BL - Quantized HQQ4/HQQ8 Prefill And Decode

## Decision

Implemented the first real quantized HQQ4/HQQ8 attention execution family:
prefill and decode now run from quantized HQQ descriptors instead of requiring
persistent full BF16 attention weight materialization for requests.

This is not a final optimized Marlin-class kernel pass. It is the architectural
cutover: HQQ4 and HQQ8 now have shared cache/runtime contracts, direct quantized
prefill kernels, direct quantized decode kernels, and separate HQQ8 cache
storage.

## Implementation

- Added `hqq8` as a supported attention quantization mode.
- Added separate HQQ8 cache paths so HQQ8 artifacts do not overwrite HQQ4.
- Added HQQ8 artifact writing and validation:
  - HQQ4 layout: `row_major_axis1_grouped_uint4_packed`
  - HQQ8 layout: `row_major_axis1_grouped_uint8`
- Added HQQ8 decode runtime layout:
  - `row_aligned_grouped_uint8_decode_v1`
- Added HQQ8 decode GEMV kernels for f32 and BF16 outputs.
- Added quantized HQQ4/HQQ8 prefill GEMM kernels.
- Removed server request-time calls that rematerialized HQQ attention weights to
  persistent BF16 matrices.
- Kept INT8 exception support as explicit manifest-driven deltas on top of the
  quantized path.

## Validation

Commands run:

- `./dev build`
- `./dev python -m py_compile python/krasis/attention_backend.py python/krasis/config.py python/krasis/model.py tests/test_hqq_self_calibrate.py`
- `./dev hqq-self-calibrate-test`
- `./dev witness-compare tests/qcn-hqq4.conf --profile llama_witness_stage3_qcn_expanded`
- `./dev witness-compare logs/manual/phase2bl_qcn_hqq8_quantized_20260426.conf --profile llama_witness_stage3_qcn_expanded`

QCN expanded llama-witness smoke:

| Run | First-token | Selected sum | Top-overlap |
| --- | ---: | ---: | ---: |
| HQQ4 quantized prefill/decode | 8/8 | 0.1167466204 | 74/80 |
| HQQ8 quantized prefill/decode | 8/8 | 0.1087577255 | 72/80 |

Cache/runtime residency from startup logs:

| Run | HQQ cache/runtime | HCS soft experts |
| --- | ---: | ---: |
| HQQ4 | 947 MB | 14904/24576 |
| HQQ8 | 1789 MB | 14337/24576 |

The HQQ4 HCS count is materially higher than the previous HQQ4 benchmark runs
that had about `12960/24576`, matching the expectation that removing persistent
BF16 prefill materialization recovers VRAM for decode expert residency.

## Boundaries

- No default model config was changed.
- No `testconfigs/` file was modified.
- No standard benchmark was run in Phase 2BL, so no throughput claim is made
  here.
- The direct quantized prefill kernels are structured for the right execution
  model but still need performance tuning before they should be treated as final.
- HQQ8 is implemented and smoke-tested on QCN; Q35 validation remains open.

## Next

Run standard `./dev benchmark` comparisons for QCN HQQ4 and HQQ8 after reviewing
the new kernels for obvious launch/tiling issues. Archive benchmark logs in
`benchmarks/` when that run is performed.
