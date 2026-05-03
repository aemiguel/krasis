# Phase 2GC - Stable HQQ CUDA Graph Addresses

Date: 2026-05-03

## Goal

CUDA graphs should be captured once and reused across requests. HCS reloads and
HQQ prefill/decode stage swaps must update graph-visible indirection or stable
addresses, not force recapture. The fix must not add BF16 attention fallback or
increase steady physical HQQ VRAM residency.

## Implementation

- HQQ runtime packed/scales/zeros buffers now use CUDA VMM slots.
- Each HQQ runtime component reserves a stable virtual address sized to the
  larger of prefill/decode, but maps only the active stage's physical bytes.
- HQQ stage swaps unmap/release old physical backing and remap the target stage
  at the same graph-visible virtual address.
- Graph-visible HQQ descriptor pointers are verified to remain stable after
  swaps; pointer changes still fail closed by invalidating graphs.
- The `prefill_logits` success path no longer invalidates graphs after restoring
  decode runtime. Error paths still invalidate/fail closed.
- HCS soft expert buffers now use raw synchronous `cuMemAlloc_v2/cuMemFree_v2`
  through `AlignedGpuBuffer`, because VMM physical remap on 122B could not rely
  on cudarc async-pool frees becoming driver-visible in time.

No BF16 attention projection tensors were restored, and no second HQQ resident
copy was added.

## Validation

All accuracy runs were completed before speed testing.

| Surface | Result | Exact | Containment | Full | First token |
| --- | --- | ---: | ---: | ---: | ---: |
| QCN HQQ8/BF16-KV VMM | PASS | `484/653` | `523/653` | `9/14` | `14/14` |
| QCN HQQ8+k6v6 VMM | PASS | `477/653` | `518/653` | `8/14` | `14/14` |
| QCN HQQ8/BF16-KV raw-HCS trace | PASS | `450/653` | `504/653` | `7/14` | `14/14` |
| QCN HQQ8+k6v6 raw-HCS | PASS | `432/653` | `482/653` | `7/14` | `14/14` |
| Q122B HQQ6+k4v4 raw-HCS | PASS | `283/361` | `304/361` | `8/14` | `14/14` |
| QCN BF16-attn/BF16-KV raw-HCS | PASS | `535/653` | `548/653` | `11/14` | `14/14` |

The Q122B surface improved versus Phase 2GB (`258/361` exact, `284/361`
containment). The non-HQQ QCN BF16-attention control also passed above the
latest comparable recent run (`485/653` exact, `558/653` containment).

## Graph Reuse

Trace run:

```bash
KRASIS_TRACE=1 KRASIS_TRACE_COMPONENTS=graph \
  ./dev witness-compare tests/qcn-bf16kv-hqq8-accuracy.conf \
  --profile phase2bn_qcn_64tok --startup-timeout 1200
```

Run dir: `logs/reference-test_20260503_025237`

Graph trace summary:

- `capture_complete`: `1`
- `graph_pointer_check result=reuse`: `15`
- invalidations / HQQ decode pointer changes: `0`

This confirms the reference path no longer recaptures graphs after every
request on the traced HQQ surface.

## Notes

The raw-HCS change was required after Q122B diagnostics showed HCS eviction
reduced the soft pool from 151 chunks to one chunk, but driver-visible free VRAM
did not rise before HQQ VMM `cuMemCreate`. Using raw `cuMemFree_v2` makes HCS
eviction return physical VRAM deterministically for VMM remap.

## Speed

Timing-free benchmark:

```bash
./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf
```

Run dir: `logs/dev-benchmark_20260503_030956`

| Metric | Phase 2GB | Phase 2GC |
| --- | ---: | ---: |
| Prefill internal | `2071.5 tok/s` | `2029.1 tok/s` |
| Decode internal | `22.54 tok/s` | `24.49 tok/s` |
| Round trip | `41.47 tok/s` | `44.35 tok/s` |
| HCS | `3483/12288 (28.3%)` | `3780/12288 (30.8%)` |
| Min free VRAM | `640 MB` | `664 MB` |

Archived benchmark log:
`benchmarks/20260503_phase2gc_q122b_k4v4_hqq6_stable_graph_benchmark.log`

Decode and round-trip improved without increasing HQQ residency. Prefill is
flat to slightly lower on this run.
