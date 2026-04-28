# Phase 2BY - HQQ8 Two-Scale + Intercept

Date: 2026-04-28

Mode:

```bash
KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-twoscale-intercept
```

## Change

Added a combined HQQ8 prefill experiment that uses:

- Marlin U8 float-zero-point GEMM
- BF16 base scale plane
- BF16 residual scale plane
- FP32 intercept correction plane

The target math is:

```text
q * (scale_hi_bf16 + scale_lo_bf16) - zero_f32 * scale_f32
```

This tests the missing combination from the previous variants:

- v2 restored intercept but missed the scale slope term
- two-scale restored slope but still used BF16 zero/intercept
- this mode restores both while avoiding the residual path's second full Marlin GEMM

## Accuracy

Command:

```bash
KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-twoscale-intercept ./dev witness-compare logs/manual/phase2bm_qcn_accuracy_hqq8_20260426.conf --profile phase2bn_qcn_64tok
```

Run dir:

```text
logs/reference-test_20260428_062206
```

| Variant | First-token | Avg exact prefix | Decode containment | Selected-logprob delta |
| --- | ---: | ---: | ---: | ---: |
| residual Marlin HQQ8 | 14/14 | 15.79 | 285/653 | 0.354102521143 |
| native fused v1 | 14/14 | 14.29 | 262/653 | 0.081758249973 |
| native fused two-scale | 14/14 | 13.57 | 294/653 | 0.121019028340 |
| two-scale + intercept | 14/14 | 18.14 | 326/653 | 0.082125920769 |

Result: accuracy improved over both residual Marlin and native fused v1 on the main recurrent gate.

## Speed

Command:

```bash
KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-twoscale-intercept ./dev benchmark tests/qcn-polar4-hqq8.conf
```

Run dir:

```text
logs/dev-benchmark_20260428_063003
```

Timing instrumentation: off.

| Variant | Prefill | Decode | HCS | Min free VRAM |
| --- | ---: | ---: | ---: | ---: |
| residual Marlin HQQ8 | 5,011.6 tok/s | 79.80 tok/s | 14256/24576 | 734 MB |
| native fused v1 | 7,132.9 tok/s | 79.68 tok/s | 14256/24576 | 734 MB |
| two-scale + intercept | 5,340.1 tok/s | 83.45 tok/s | 14256/24576 | 702 MB |

Result: speed is faster than residual Marlin and much faster than scalar HQQ8, but below native fused v1 because this mode adds group sums plus an intercept correction kernel.

## Decision

Two-scale + intercept is the current quality-favored HQQ8 fast-prefill candidate. It is not the max-speed design, but it closes the observed accuracy gap while keeping most of the Marlin speedup.
