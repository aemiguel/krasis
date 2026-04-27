# Phase 2BH - Minimal Opt-In INT8 Exception Runtime

## Decision

`int8_exception_prefill_runtime_loads_and_executes_q35_g14_improves_selected_logprob_qcn_g12_regresses_slightly`

## Scope

- Implemented a manifest-driven, opt-in prefill-only INT8 exception execution path.
- Runtime mode is explicit via `CFG_HQQ_SIDECAR_MANIFEST`; no default/config promotion was made.
- Decode, sigmoid, qkvz expansion behavior, residual sidecar promotion, combined manifests, and fallback behavior were not changed.
- Hot replacement path is Rust/CUDA: the prefill sidecar kernel uses mode `int8_exception` to replace selected HQQ-dequantized BF16 blocks with INT8-dequantized source blocks before the existing matmul.

## Runtime Contract

- Manifest mode: `int8_exception`.
- Tested tensor: layer `0`, `in_proj_qkvz`.
- Tested groups: QCN group `12`, Q35 group `14`.
- Per-manifest payload: `12,288` row groups, `1,572,864` INT8 weight bytes, `245,760` metadata bytes, `1,818,624` total bytes.
- Strict validation checks source contract, source tensor hash, HQQ artifact hash, sidecar artifact hash, tensor/layer bounds, group bounds, layout, dtype, and model/profile identity.

## Intrinsic Projection Signal

| Model | Group | Heldout HQQ RMS | Heldout INT8 RMS | Reduction |
| --- | ---: | ---: | ---: | ---: |
| QCN | `12` | `0.0570843741` | `0.0034785673` | `93.9063%` |
| Q35 | `14` | `0.0362454876` | `0.0022250710` | `93.8611%` |

## Witness External Validation

| Run | First-token | Selected sum | Delta vs HQQ-only | Top-overlap | Duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| QCN HQQ-only | `8/8` | `0.1236746500` | `0` | `73/80` | `129.8s` |
| QCN INT8 exception | `8/8` | `0.1249526500` | `+0.0012780000` | `73/80` | `126.5s` |
| Q35 HQQ-only | `10/10` | `0.8936375677` | `0` | `85/100` | `137.4s` |
| Q35 INT8 exception | `10/10` | `0.8429255677` | `-0.0507120000` | `84/100` | `136.5s` |

Interpretation:
- QCN group `12` is functionally valid but not accepted for quality; selected-logprob regressed by `+0.0012780000`.
- Q35 group `14` is the first INT8 exception runtime probe that improves selected-logprob versus adjacent HQQ-only, but it loses one top-overlap slot, so it is still diagnostic-only and not promotion-ready.

## Benchmark Impact

Standard `./dev benchmark` runs were executed without timing instrumentation.

| Model | Run | Prefill internal | Decode internal | Network round trip | Min free VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| QCN | HQQ-only | `7,912.5 tok/s` | `73.97 tok/s` | `150.85 tok/s` | `686 MB` |
| QCN | INT8 exception | `7,832.5 tok/s` | `71.95 tok/s` | `146.84 tok/s` | `682 MB` |
| Q35 | HQQ-only | `9,360.5 tok/s` | `116.24 tok/s` | `266.64 tok/s` | `5,452 MB` |
| Q35 | INT8 exception | `9,332.0 tok/s` | `116.06 tok/s` | `235.24 tok/s` | `5,452 MB` |

Notes:
- QCN INT8 exception prefill best was `1.0%` lower and min free VRAM was `4 MB` lower.
- Q35 INT8 exception prefill best was `0.3%` lower and min free VRAM was unchanged.
- Network round-trip numbers are client/HTTP numbers and are not used as internal engine speed.

## Per-Case Witness Deltas

Largest QCN regressions:
- case `4`: `+0.0038590000`, overlap delta `0`
- case `1`: `+0.0024150000`, overlap delta `0`
- case `5`: `+0.0009660000`, overlap delta `0`

Largest Q35 improvements:
- case `10`: `-0.0439220000`, overlap delta `0`
- case `2`: `-0.0383210000`, overlap delta `+1`
- case `7`: `-0.0199060000`, overlap delta `0`

Largest Q35 regressions:
- case `5`: `+0.0468650000`, overlap delta `0`
- case `6`: `+0.0091330000`, overlap delta `-1`
- case `3`: `+0.0015160000`, overlap delta `-1`

## Artifacts

- QCN INT8 manifest write: `logs/manual/phase2bh_qcn_int8_exception_write_20260426.json`
- Q35 INT8 manifest write: `logs/manual/phase2bh_q35_int8_exception_write_20260426.json`
- QCN witness HQQ-only: `logs/reference-test_20260426_192008/reference_test_summary.json`
- QCN witness INT8: `logs/reference-test_20260426_192240/reference_test_summary.json`
- Q35 witness HQQ-only: `logs/reference-test_20260426_192514/reference_test_summary.json`
- Q35 witness INT8: `logs/reference-test_20260426_192825/reference_test_summary.json`
- QCN benchmark HQQ-only: `logs/dev-benchmark_20260426_193218/benchmark_report.log`
- QCN benchmark INT8: `logs/dev-benchmark_20260426_193537/benchmark_report.log`
- Q35 benchmark HQQ-only: `logs/dev-benchmark_20260426_193851/benchmark_report.log`
- Q35 benchmark INT8: `logs/dev-benchmark_20260426_194201/benchmark_report.log`

## Remaining Work

- Do not promote current manifests.
- Extend candidate selection beyond one group only after a rule accounts for Q35 case-5 regression and QCN group-12 non-improvement.
- Consider narrower/lower-cost INT8 exception buckets or additional groups only with adjacent HQQ-only baselines and standard benchmark runs.
