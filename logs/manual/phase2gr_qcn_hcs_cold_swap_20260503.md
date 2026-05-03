# Phase 2GR - QCN HCS Cold Swap Speed Test

Date: 2026-05-03

## Goal

Run Qwen3-Coder-Next with opt-in approximate HCS cold swaps enabled and compare
against the recent exact prompt-HCS QCN speed rows.

## Command

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_HCS_COLD_SWAP=1 ./dev speed-test
```

Run dir:
`logs/dev-benchmark_20260503_202952`

Archived benchmark log:
`benchmarks/20260503_phase2gr_qcn_hcs_cold_swap_speedtest.log`

Timing instrumentation was disabled.

## Result

| Metric | Result |
| --- | ---: |
| Prefill best | `8488.2 tok/s` |
| Decode internal best | `84.45 tok/s` |
| Round trip best | `192.63 tok/s` |
| HCS | `15147/24576` |
| Min free VRAM | `706 MB` |

Official internal decode rows:

| Row | Decode | Swaps | Swaps/tok | Cold after swaps |
| --- | ---: | ---: | ---: | ---: |
| 50 tok | `84.45 tok/s` | `155` | `3.16` | `10.47/tok` |
| 100 tok | `84.03 tok/s` | `204` | `2.06` | `7.00/tok` |
| 250 tok | `76.92 tok/s` | `1245` | `5.00` | `21.43/tok` |

Weighted internal rows:

| Metric | Result |
| --- | ---: |
| Decode tokens | `397` |
| Swaps | `1604` |
| Swaps/token | `4.04` |
| Weighted cold after swaps | `16.48/tok` |

## Comparison

Recent exact QCN rows:

| Run | Decode internal |
| --- | ---: |
| Phase 2GO 85% default | `89.37 tok/s` |
| Phase 2GP 90% retain | `90.78 tok/s` |
| Phase 2GR HCS cold swaps | `84.45 tok/s` |

QCN HCS cold swaps are not a speed win. They also affect model behavior by
changing selected expert IDs, so there is no reason to pursue this as a QCN
default from this result.

## Errors

No `CUDA_ERROR`, `RuntimeError`, `Traceback`, `panic`, `ILLEGAL`,
`INVALID_CONTEXT`, or timeout markers were found in the checked logs.
