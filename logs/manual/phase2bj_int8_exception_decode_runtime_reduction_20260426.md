# Phase 2BJ - INT8 Exception Decode Runtime

## Decision

`int8_exception_decode_path_implemented_end_to_end_top4_quality_small_improvement_decode_cost_visible`

## Implementation

- Added decode-side INT8 exception execution in Rust/CUDA.
- Existing HQQ decode GEMV still runs first.
- For selected manifest blocks, decode applies a CUDA delta equal to `INT8_exception_dequant - HQQ_dequant` over the selected column block, which gives replacement semantics for that block.
- The runtime remains manifest-driven and opt-in through `hqq_sidecar_manifest`.
- No default enablement, decode fallback, Python hot-path code, residual sidecar promotion, or qkvz behavior change was added.

## Manifests

| Model | Tensor | Groups | Row entries | Bytes |
| --- | --- | ---: | ---: | ---: |
| QCN | layer-0 `in_proj_qkvz` | 4 | 49,152 | 7,274,496 |
| Q35 | layer-0 `in_proj_qkvz` | 4 | 49,152 | 7,274,496 |

## Witness

Baselines are the same-day HQQ-only witness runs from Phase 2BH. INT8 rows are the Phase 2BI top-4 manifests after decode execution was implemented.

| Run | First-token | Selected sum | Delta vs HQQ-only | Top-overlap |
| --- | ---: | ---: | ---: | ---: |
| QCN HQQ-only | 8/8 | 0.1236746500 | 0 | 73/80 |
| QCN INT8 top-4 | 8/8 | 0.1168007312 | -0.0068739188 | 73/80 |
| Q35 HQQ-only | 10/10 | 0.8936375677 | 0 | 85/100 |
| Q35 INT8 top-4 | 10/10 | 0.8868880518 | -0.0067495159 | 84/100 |

Largest remaining INT8 top-4 case deltas:

| Run | Case | Selected-logprob delta |
| --- | ---: | ---: |
| QCN INT8 top-4 | 6 | 0.0469750293 |
| QCN INT8 top-4 | 4 | 0.0292705967 |
| QCN INT8 top-4 | 8 | 0.0285629623 |
| Q35 INT8 top-4 | 2 | 0.4912393825 |
| Q35 INT8 top-4 | 5 | 0.2061549459 |
| Q35 INT8 top-4 | 10 | 0.0690234286 |

## Standard Benchmarks

Timing instrumentation was off. Decode numbers below are the internal engine readings, not HTTP round-trip.

| Run | Prefill tok/s | Decode tok/s | Min free VRAM |
| --- | ---: | ---: | ---: |
| QCN HQQ-only | 7,912.5 | 73.97 | 686 MB |
| QCN INT8 top-4 | 6,984.6 | 71.30 | 692 MB |
| Q35 HQQ-only | 9,360.5 | 116.24 | 5452 MB |
| Q35 INT8 top-4 | 9,523.2 | 111.88 | 5428 MB |

## Reduction

- The feature now runs end-to-end across prefill and decode for explicit INT8 exception manifests.
- Top-4 INT8 exceptions slightly improved selected-logprob sum on both QCN and Q35 versus the HQQ-only baselines used here.
- Q35 top-overlap regressed by one slot, and case 2 remains a large outlier.
- Decode throughput cost is visible: about `-3.6%` on QCN and `-3.8%` on Q35 for these top-4 manifests.
- The implementation is usable for full HQQ-only versus HQQ+INT8-exception comparisons, but current manifests are still diagnostic and not promotion/default candidates.
