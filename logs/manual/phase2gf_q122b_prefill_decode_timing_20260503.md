# Phase 2GF - Q122B Prefill/Decode Timing Attribution

Date: 2026-05-03

## Purpose

Re-run detailed timing on the current fastest validated 122B surface after:

- Phase 2GC stable HQQ CUDA graph/VMM work
- Phase 2GE materialized HQQ prefill

This is an instrumentation run. Do not compare its tok/s directly against
timing-free speed benchmarks except as a rough sanity check.

## Command

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 \
KRASIS_PREFILL_TIMING=1 \
KRASIS_PREFILL_DEBUG=1 \
  ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf --timing
```

Run directory:

- `logs/dev-benchmark_20260503_091214`
- wrapper log:
  `logs/manual/phase2gf_q122b_hqq6_k4v4_prefill_decode_timing_20260503.log`

## Summary

Timing-enabled benchmark headline:

| Metric | Result |
| --- | ---: |
| Prefill internal | `2989.5 tok/s` |
| Decode internal | `23.62 tok/s` |
| Round trip | `42.32 tok/s` |
| HCS | `3780/12288 (30.8%)` |
| Min free VRAM | `662 MB` |

All official decode rows were in `graph mode`. The graph-recapture issue is
not the current decode blocker on this run.

## Prefill Attribution

The 50K timing block is the best long-prompt attribution sample:

| Component | Time | Share |
| --- | ---: | ---: |
| Total | `18723.0 ms` | `100%` |
| MoE | `12661.9 ms` | `67.6%` |
| Attention | `5720.4 ms` | `30.6%` |
| Other | about `340 ms` | about `1.8%` |

MoE is now the dominant prefill cost:

| MoE subcomponent | Time |
| --- | ---: |
| DMA total | `9774.2 ms` |
| DMA wait | `9244.9 ms` |
| cold H2D | `512.9 ms` |
| w1 + activation | `1733.1 ms` |
| w2 | `752.1 ms` |
| scatter | `235.5 ms` |

Attention is split roughly evenly between GQA and LA:

| Attention subcomponent | Time |
| --- | ---: |
| GQA total | `2776.4 ms` |
| GQA FA2 | `2268.1 ms` |
| GQA projection | `308.8 ms` |
| LA total | `2944.0 ms` |
| LA projection | `1132.2 ms` |
| LA FLA | `990.9 ms` |
| LA O projection | `436.8 ms` |

Interpretation:

- The old HQQ Marlin float-zp projection bottleneck is no longer primary.
- Prefill is now mostly MoE DMA/wait and attention execution.
- KV append/cross-stage work is small in this run (`~7.5 ms` GQA KV prep on
  the 50K block).
- The remaining LA projection cost is still non-trivial, but it is no longer
  the main prefill limiter.

## Decode Attribution

Official internal decode rows:

| Decode row | Total | Internal speed | Cold experts/tok | HCS experts/tok |
| --- | ---: | ---: | ---: | ---: |
| 50 tokens | `41.92 ms/tok` | `23.9 tok/s` | `124.6` | `259.4` |
| 100 tokens | `42.86 ms/tok` | `23.3 tok/s` | `128.7` | `255.3` |
| 250 tokens | `44.84 ms/tok` | `22.3 tok/s` | `140.3` | `243.7` |

Representative graph timing:

| Decode row | Sync wait | GPU compute | Cold DMA | Upload | Launch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 tokens | `37.90 ms/tok` | `1.58` | `1.21` | `0.12` | `1.07` |
| 100 tokens | `39.62 ms/tok` | `1.42` | `1.23` | `0.12` | `0.42` |
| 250 tokens | `41.39 ms/tok` | `1.53` | `1.34` | `0.12` | `0.41` |

Graph replay CUDA event totals are stable:

| Decode row | Segment total | LA route | GQA route | Final |
| --- | ---: | ---: | ---: | ---: |
| 50 tokens | `17.38 ms/tok` | `12.20` | `3.87` | `1.10` |
| 100 tokens | `17.18 ms/tok` | `11.91` | `3.96` | `1.10` |
| 250 tokens | `17.15 ms/tok` | `11.71` | `4.13` | `1.10` |

Interpretation:

- Decode is graph-captured and graph replay is stable.
- Current decode is not dominated by explicit launch overhead.
- The official rows still cold-load about `125-140` experts/token.
- HCS hit quality is much better than the broken pre-graph-fix run, but cold
  traffic remains material.
- The `Sync wait` bucket dominates the wall-clock table. This needs a more
  precise split before optimizing it; it is likely waiting on graph/stream
  completion and serialized work, not necessarily pure CPU idle.

## Current Optimization Targets

Prefill:

1. Reduce MoE DMA/wait during prefill. Current long-prompt prefill spends
   `~9.2s` of a 50K block in MoE DMA wait.
2. Reduce attention execution cost, especially GQA FA2 on long chunks and LA
   projection/FLA. This is secondary to MoE DMA on the measured 50K block.

Decode:

1. Improve HCS hit quality / reduce cold experts per token. The official rows
   still cold-load `~125-140` experts/token.
2. Split the dominant `Sync wait` bucket into graph replay wait, cold-load
   dependency wait, and any CPU-side synchronization. The current timing label
   is too broad to safely optimize by name alone.
3. Keep graph capture stable; no evidence in this run that graph recapture is
   the decode bottleneck.

## Process State

The timing run completed successfully. No runtime code changes were made for
Phase 2GF.
