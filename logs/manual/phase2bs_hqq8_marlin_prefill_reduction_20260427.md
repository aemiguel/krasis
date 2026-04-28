# Phase 2BS - HQQ8 Marlin Prefill Prototype

Date: 2026-04-27

## Scope

Build the first production-speed HQQ8 prefill path. The old HQQ8 prefill path
used a scalar/tiny-tile correctness kernel and measured only `486.8 tok/s` on
the QCN Polar4 speed-test surface. This pass replaces HQQ8 prefill GEMM with a
Marlin U8B128 path.

## Implementation

- Repacked HQQ8 prefill weights into Marlin U8B128 layout.
- Ran Marlin for the `(q - 128) * scale_bf16` term.
- Added a second Marlin pass with residual BF16 scales approximating
  `scale_fp32 - scale_bf16`.
- Added grouped zero correction:
  `sum(input_group) * (128 - zero) * scale_fp32`.
- Reused existing large non-overlapping scratch buffers for the residual output;
  no new VRAM allocation or hardcoded budget was added.
- HQQ4/HQQ4SC remain on the old prefill path for now.

## Correctness Gate

Command:

```bash
./dev witness-compare logs/manual/phase2bm_qcn_accuracy_hqq8_20260426.conf --profile phase2bn_qcn_64tok
```

Result: `logs/reference-test_20260427_214956/reference_test_summary.json`

| Variant | First-token | Avg exact prefix | Worst | Full matches | Decode top-k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous fixed HQQ8 | 14/14 | 16.43 | 2 | 4/14 | 59.1% |
| HQQ8 Marlin prefill prototype | 14/14 | 15.79 | 1 | 4/14 | 57.2% |

The prototype is close enough to run speed measurements, but it is not yet a
clean quality replacement for the previous fixed HQQ8 path because selected
first-token logprob deltas are worse.

## Speed Result

Command:

```bash
./dev benchmark tests/qcn-polar4-hqq8.conf
```

Result: `logs/dev-benchmark_20260427_215803/benchmark_report.log`

| Variant | Prefill internal | Decode internal | HCS | Min free VRAM |
| --- | ---: | ---: | ---: | ---: |
| AWQ/Polar4 baseline | 7,295.6 tok/s | 91.77 tok/s | 16848/24576 | 688 MB |
| HQQ8 scalar prefill | 486.8 tok/s | 75.16 tok/s | 14256/24576 | 734 MB |
| HQQ8 Marlin prefill prototype | 5,011.6 tok/s | 79.80 tok/s | 14256/24576 | 734 MB |

The Marlin prototype improves HQQ8 Polar4 prefill by `10.3x` over the scalar HQQ
path and reaches about `68.7%` of the AWQ/Polar4 baseline prefill speed.

## Current Read

The architectural direction is correct: HQQ8 needs a packed Marlin-class prefill
path, not tuning of the scalar HQQ kernel. The first prototype removes most of
the prefill gap while preserving broadly acceptable QCN correctness.

Remaining work:

- close the residual quality gap versus previous fixed HQQ8, especially
  selected first-token logprob delta and the prompt with exact-prefix `1`
- decide whether the residual-scale second Marlin pass is the right permanent
  math path or whether native FP32 scale handling inside the Marlin kernel is
  worth building
- build HQQ4SC fast prefill/sidecar fusion only after HQQ8 is settled
