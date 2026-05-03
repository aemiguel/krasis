# Phase 2GG - Q122B Decode Sync-Wait Split

Date: 2026-05-03

## Question

After Phase 2GF, decode timing still had a broad `Sync wait` bucket around
`38-41 ms/tok`. The user asked to split that bucket and asked whether LA time
is largely FLA.

This run instruments graph-mode decode so each inter-graph sync wait is
attributed to the graph segment that just completed. The final stream sync is
reported separately and attributed to the final segment in the non-mapped graph
path.

## Run

Command:

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 \
KRASIS_PREFILL_TIMING=1 KRASIS_PREFILL_DEBUG=1 \
  ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf --timing
```

Run dir:

- `logs/dev-benchmark_20260503_093825`

Wrapper log:

- `logs/manual/phase2gg_q122b_decode_sync_split_timing_20260503.log`

This is a timing/instrumentation run. The timing-free speed baseline remains
Phase 2GE.

## Headline

Timing-enabled benchmark report:

- prefill best: `2977.9 tok/s`
- decode internal best: `23.95 tok/s`
- round trip best: `42.99 tok/s`
- HCS: `3780/12288 (30.8%)`
- min free VRAM: `662 MB`

## Decode Rows

Official decode rows, all graph mode:

| Row | Total | Cold experts/tok | HCS experts/tok | DMA bytes/tok |
| --- | ---: | ---: | ---: | ---: |
| 50 tokens | `41.57 ms/tok` | `122.9` | `261.1` | `570.23 MB` |
| 100 tokens | `43.07 ms/tok` | `128.5` | `255.5` | `596.30 MB` |
| 250 tokens | `46.17 ms/tok` | `146.5` | `237.5` | `679.73 MB` |

## Graph Event Time

CUDA event timing inside graph replay:

| Row | Segment total | LA route | GQA route | Final |
| --- | ---: | ---: | ---: | ---: |
| 50 tokens | `17.05 ms/tok` | `11.90` | `3.84` | `1.10` |
| 100 tokens | `17.17 ms/tok` | `11.91` | `3.95` | `1.10` |
| 250 tokens | `17.21 ms/tok` | `11.76` | `4.15` | `1.10` |

The LA route is the linear-attention decode route. For Qwen3.5 this is the
same family of work we refer to as FLA in prefill, plus adjacent captured
route/expert work in that graph segment.

## Sync-Wait Attribution

New CPU sync-wait attribution:

| Row | Sync total | Inter | Final | LA route wait | GQA route wait | Final wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 tokens | `39.19 ms/tok` | `37.78` | `1.41` | `28.14` | `9.32` | `1.41` |
| 100 tokens | `41.08 ms/tok` | `39.77` | `1.32` | `29.33` | `10.17` | `1.32` |
| 250 tokens | `44.04 ms/tok` | `42.41` | `1.63` | `31.30` | `10.89` | `1.63` |

Per-segment wait is similar for LA and GQA, around `0.8-0.9 ms/seg`, but there
are three times as many LA route segments as GQA route segments in this model
(`36` LA layers vs `12` GQA layers). That makes LA the dominant total wait
bucket.

## Interpretation

Yes: decode LA time is largely the LA/FLA-family route. The split now shows two
separate but aligned signals:

- CUDA event graph replay time is dominated by LA route: about
  `11.8-11.9 ms/tok` versus `3.8-4.2 ms/tok` for GQA.
- CPU sync wait is also mostly attributable to LA route: about
  `28-31 ms/tok` versus `9-11 ms/tok` for GQA.

This does not prove that a single FLA kernel is the only cost. The label is the
captured LA route graph segment, so it includes linear-attention work plus the
segment's adjacent expert/route dependencies. The next useful timing split is
inside LA route itself: linear attention kernel time, LA projection/state work,
expert compute, and waits induced by cold expert availability.

## Next Targets

1. Add an LA-route internal split for decode, analogous to prefill's
   projection/FLA breakdown.
2. Keep cold expert metrics beside that split, because cold experts rise from
   `122.9/tok` to `146.5/tok` across the official rows and likely contribute
   to segment wait.
3. Do not optimize the generic `Sync wait` label directly anymore. The current
   evidence says the wait follows LA and GQA graph segments, especially LA.
