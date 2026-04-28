# Phase 2BU HQQ8 native fused Marlin prefill reduction - 2026-04-27

## Goal

Test a native fused HQQ8 Marlin prefill path after the residual two-GEMM
prototype reached good speed but still had a quality gap.

## Implementation

Mode:

`KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin`

This mode stages HQQ8 attention weights as Marlin U8 packed weights with BF16
scales and BF16 float zero-points, then dispatches one Marlin GEMM with
`has_zp=true` and `is_zp_float=true`.

It avoids:

- the residual second Marlin GEMM
- the external grouped zero-correction kernel
- intermediate BF16 output composition

Important limitation: this path still uses Marlin BF16 metadata for scales and
zero-points. It is not a full FP32-scale/FP32-zero HQQ accumulator.

## Accuracy gate

Command:

`KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin ./dev witness-compare logs/manual/phase2bm_qcn_accuracy_hqq8_20260426.conf --profile phase2bn_qcn_64tok`

Run dir:

`logs/reference-test_20260427_225530`

Comparison on the QCN 64-token witness profile:

| Variant | First-token | Prefill argmax | Avg exact prefix | Worst | Full matches | Decode containment | Selected logprob delta sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| residual Marlin HQQ8 | 14/14 | 14/14 | 15.79 | 1 | 4/14 | 285/653 | 0.354103 |
| symmetric Marlin HQQ8 | 14/14 | 14/14 | 11.71 | 1 | 4/14 | 230/653 | 0.163842 |
| native fused Marlin HQQ8 | 14/14 | 14/14 | 14.29 | 1 | 5/14 | 262/653 | 0.081758 |

Interpretation:

Native fused is clearly better than simple symmetric conversion and has the
best selected first-token logprob delta sum of the three fast experiments. It
does not fully recover the residual Marlin path's recurrent decode quality.

## Speed benchmark

Command:

`KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin ./dev benchmark tests/qcn-polar4-hqq8.conf`

Timing instrumentation was off.

Run dir:

`logs/dev-benchmark_20260427_230455`

Archived benchmark log:

`benchmarks/20260427_230455_qcn_polar4_hqq8_native_fused_speed.log`

| Variant | Prefill internal | Decode internal | HCS | Min free VRAM |
| --- | ---: | ---: | ---: | ---: |
| AWQ/Polar4 baseline | 7,295.6 tok/s | 91.77 tok/s | 16848/24576 (68.6%) | 688 MB |
| scalar HQQ8/Polar4 | 486.8 tok/s | 75.16 tok/s | 14256/24576 (58.0%) | 734 MB |
| residual Marlin HQQ8/Polar4 | 5,011.6 tok/s | 79.80 tok/s | 14256/24576 (58.0%) | 734 MB |
| native fused Marlin HQQ8/Polar4 | 7,132.9 tok/s | 79.68 tok/s | 14256/24576 (58.0%) | 734 MB |

Native fused HQQ8 prefill is:

- `14.7x` faster than scalar HQQ8 prefill
- `42.3%` faster than the residual Marlin prototype
- `97.8%` of the AWQ/Polar4 prefill baseline

## Conclusion

The architecture works for speed. A one-GEMM native Marlin HQQ8 path reaches
near-AWQ prefill throughput on the QCN Polar4 speed-test surface.

It is not production-ready because recurrent witness quality is still below the
residual Marlin HQQ8 prototype. The remaining gap is likely from BF16 Marlin
scale/zero metadata rather than from external correction-pass structure. The
next accuracy target is a fused path that preserves more FP32 HQQ scale/zero
behavior inside the accumulator or epilogue without returning to multiple output
stores.
