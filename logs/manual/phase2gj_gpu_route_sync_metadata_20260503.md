# Phase 2GJ - Metadata-Only GPU Route Sync

Date: 2026-05-03

## Goal

Implement a repaired GPU route-sync path for decode without enabling mapped
cold-weight reads. The intended design is metadata-only:

- exact top-k routing is unchanged
- the GPU classifies selected experts against the HCS resident pointer table
- resident expert pointers are written into the graph-visible batch upload table
- cold expert IDs/slots are returned as compact metadata
- cold experts still use CPU-initiated DMA into VRAM before expert compute
- direct GPU reads of cold CPU weights over PCIe remain disabled

## Implementation

Changed `src/gpu_decode.rs`:

- Added explicit `KRASIS_GPU_ROUTE_SYNC=1` opt-in.
- Made `KRASIS_MAPPED_READS=1` fail closed. Mapped cold-weight reads are not a
  supported route-sync mode.
- Required route-sync graph capture to include `expert_classify_prepare`.
- Invalidated pre-HCS CUDA graphs when route sync is enabled after HCS load.
- Added bounded ready-flag polling via `KRASIS_GPU_ROUTE_SYNC_TIMEOUT_MS`
  (default 10s).
- Added metadata invariants for cold expert counts, expert IDs, and batch slots.
- Kept `d_expert_ptrs` VRAM-only; cold experts are staged by CPU DMA.
- Synchronized HCS pointer-table updates before route-sync decode can consume
  them, including soft eviction/reload boundaries.
- Bound the CUDA device on server threads before route-sync pointer-table
  updates/synchronization.

## Bugs Fixed

1. Pre-HCS graphs were reused after enabling route sync. Those graphs did not
   contain `expert_classify_prepare`, so replay timed out at layer 0. Fixed by
   invalidating and recapturing graph segments when route sync becomes active.

2. HCS pointer-table updates were asynchronous relative to route-sync graph
   replay. Decode could see stale pointers to freed/reloaded soft chunks and
   fault with `CUDA_ERROR_ILLEGAL_ADDRESS`. Fixed by synchronizing route-sync
   pointer-table updates at soft chunk free/reload boundaries.

3. Pointer-table synchronization on server threads could hit
   `CUDA_ERROR_INVALID_CONTEXT`. Fixed by binding the store CUDA device before
   route-sync pointer-table updates and sync.

## Accuracy

Q122B HQQ6+k4v4 seq32 witness with materialized HQQ prefill and
`KRASIS_GPU_ROUTE_SYNC=1`:

- run dir: `logs/reference-test_20260503_142849`
- first token: `14/14`
- prefill containment: `14/14`
- generated exact: `283/361`
- generated containment: `304/361`
- full exact: `8/14`
- result: PASS

## Timing Diagnostic

Timing-enabled benchmark:

- run dir: `logs/dev-benchmark_20260503_143610`
- command: `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_GPU_ROUTE_SYNC=1 ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf --timing`
- HCS load: `gpu_route_sync=true`, `mapped_reads=false`, initial soft
  `4050/12288 (33.0%)`
- benchmark summary HCS after prefill/decode transitions: `3780/12288 (30.8%)`
- internal decode best: `24.21 tok/s`
- no timeout, context, or CUDA illegal-address errors after the final fix

Compared with the prior timing-enabled Phase 2GG run, decode was only a small
diagnostic improvement (`23.95 -> 24.21 tok/s`). This was not strong enough to
call a win because timing runs perturb the path.

## Timing-Free Benchmark

Timing-free route-sync benchmark:

- run dir: `logs/dev-benchmark_20260503_144612`
- log: `benchmarks/20260503_phase2gj_q122b_k4v4_hqq6_gpu_route_sync_benchmark.log`
- command: `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_GPU_ROUTE_SYNC=1 ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf`

Results:

- prefill best: `4094.5 tok/s`
- internal decode best: `23.87 tok/s`
- round trip best: `42.69 tok/s`
- HCS: `3780/12288 (30.8%)`
- min free VRAM: `1846 MB`

Current best Q122B baseline, Phase 2GH pointer-prefetch with route sync off:

- prefill best: `4689.8 tok/s`
- internal decode best: `24.80 tok/s`
- round trip best: `42.45 tok/s`
- HCS: `3780/12288 (30.8%)`
- min free VRAM: `662 MB`

## Conclusion

Metadata-only GPU route sync is implemented and stable as an opt-in path, but
it is not a speed win in its current form. It should remain disabled by default.

The experiment also shows why this does not yet reduce graph segmentation:
CPU still must stage cold experts for hybrid layers, so the per-layer boundary
remains. The next route-sync work would need to use the metadata to skip CPU
work for provably resident-only layers or otherwise reduce graph boundaries;
metadata-only classification by itself is not enough.
