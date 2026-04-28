# Phase 2CS QCN k8v4 HQQ8 Benchmarks - 2026-04-28

Standard benchmark runs for QCN HQQ8 attention with k8v4 KV cache, comparing
INT8 and INT4 expert caches. Timing instrumentation was disabled.

Commands:

```bash
KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-twoscale-intercept ./dev benchmark tests/qcn-k8v4-hqq8-int8-benchmark.conf
KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-twoscale-intercept ./dev benchmark tests/qcn-k8v4-hqq8-int4-benchmark.conf
```

## Summary

| Run | Prefill best | Decode best | Round-trip best | HCS | Min free VRAM | Log |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| HQQ8 + k8v4 + INT8 experts | `4,238.1 tok/s` | `35.00 tok/s` | `57.13 tok/s` | `7175/24576 (29.2%)` | `752 MB` | `benchmarks/20260428_194459_qcn_k8v4_hqq8_int8_benchmark.log` |
| HQQ8 + k8v4 + INT4 experts | `5,245.5 tok/s` | `76.74 tok/s` | `150.26 tok/s` | `14256/24576 (58.0%)` | `708 MB` | `benchmarks/20260428_195237_qcn_k8v4_hqq8_int4_benchmark.log` |

## Prefill Detail

| Run | 1K | 5K | 10K | 20K | 35K | 50K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| INT8 experts | `392.3` | `1,330.2` | `2,310.1` | `3,285.6` | `4,238.1` | `3,861.5` |
| INT4 experts | `363.1` | `1,966.3` | `3,337.6` | `4,286.9` | `5,245.5` | `4,932.1` |

## Decode Detail

| Run | 50 tok | 100 tok | 250 tok |
| --- | ---: | ---: | ---: |
| INT8 experts | `29.96 tok/s` | `35.00 tok/s` | `28.60 tok/s` |
| INT4 experts | `74.62 tok/s` | `76.74 tok/s` | `67.43 tok/s` |

## Notes

- Decode numbers above are the benchmark's internal engine numbers, not HTTP
  round-trip numbers.
- INT4 experts are much faster here because they allow roughly double the HCS
  soft coverage (`58.0%` versus `29.2%`) and lower expert cache bandwidth.
- The INT4 benchmark emitted one VRAM monitor warning during timed prefill:
  `564 MB` free, below the configured `600 MB` safety margin. The final decode
  benchmark summary reported `708 MB` min free during decode.
- These are speed-only benchmarks. The matching Phase 2CR accuracy run for
  k8v4 used INT8 experts and reached `450/653` exact tokens; INT4 accuracy for
  k8v4 has not been measured in this phase.
