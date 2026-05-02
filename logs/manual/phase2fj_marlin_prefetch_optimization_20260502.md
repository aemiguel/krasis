# Phase 2FJ: Marlin Non-Stacked Prefetch Optimization

Date: 2026-05-02

## Goal

Continue optimizing the QCN INT4 Marlin cache routed-layer loop after Phase 2FI
reduced full cold build time from `544s` to `176s`.

The remaining Phase 2FI profile had a gap:

| Bucket | Time |
| --- | ---: |
| Routed layer loop | `169s` |
| Routed load/quantize | `23.8s` |
| Routed Marlin repack | `21.3s` |
| Routed cache write | `46.8s` |
| Accounted routed time | `91.9s` |
| Unaccounted routed time | `~77s` |

## Instrumentation Run

Command:

```bash
./dev run tests/qcn-bf16.conf --build-cache
```

Log:

```text
logs/manual/phase2fj_qcn_marlin_int4_prefetch_timing_20260502.log
```

Result:

| Bucket | Time |
| --- | ---: |
| Full cache build | `173s` |
| Routed layer loop | `164s` |
| Initial prefetch | `1.1s` |
| Routed load/quantize | `23.8s` |
| Routed next-layer prefetch | `75.7s` |
| Routed Marlin repack | `16.5s` |
| Routed cache write | `46.0s` |
| Routed misc | `1.3s` |
| Post-build cache load | `30s` |

Read: the missing routed-loop time was synchronous non-stacked per-expert
`MADV_WILLNEED` prefetch.

## Change

Removed non-stacked per-expert prefetch for the Marlin cache builder.

Kept stacked/MXFP4 bulk-tensor prefetch unchanged because those paths issue a
small number of layer-level prefetch calls and were not implicated by this
measurement.

The change preserves:

- one-layer-at-a-time safetensor residency
- cache format
- quantization math
- deterministic expert ordering

## After

Same command:

```bash
./dev run tests/qcn-bf16.conf --build-cache
```

Log:

```text
logs/manual/phase2fj_qcn_marlin_int4_no_prefetch_20260502.log
```

Result:

| Bucket | Before Phase 2FI | Phase 2FI | Phase 2FJ |
| --- | ---: | ---: | ---: |
| Full cache build | `544s` | `176s` | `116s` |
| Routed layer loop | `537s` | `169s` | `108s` |
| Routed load/quantize | `389.4s` | `23.8s` | `36.5s` |
| Routed next-layer prefetch | unmeasured | `75.7s` | `0.0s` |
| Routed Marlin repack | `20.5s` | `21.3s` | `22.6s` |
| Routed cache write | `44.6s` | `46.8s` | `48.1s` |
| Routed misc | unmeasured | `1.3s` | `0.7s` |
| Post-build cache load | `30s` | `31s` | `30s` |

Demand paging increased routed load/quantize by `12.7s`, but removing
synchronous prefetch saved `75.7s`, for a net cache-build reduction of about
`60s` versus Phase 2FI.

## Validation

- `./dev build` passed.
- No-prefetch cold build completed with `BUILD CACHE COMPLETE`.
- No-prefetch cache loaded successfully after build: `39.9 GB in 30s`.
- No-prefetch cache was byte-identical to:
  - the prefetch-enabled cache built immediately before the change
  - the preserved cache from before the instrumentation run
- Temporary 38GB backup caches created during measurement were removed after
  byte-identical validation.

## Remaining Bottleneck

After Phase 2FJ, the largest measured remaining routed bucket is cache write:
`48.1s` of the `116s` full build.

Further write-side optimization is possible, but it should be measured as a
storage/syscall/memory-copy problem before changing the cache writer.
