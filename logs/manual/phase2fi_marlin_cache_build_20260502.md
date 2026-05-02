# Phase 2FI: Marlin Cache Build Speed

Date: 2026-05-02

## Goal

Measure and speed up the routed expert Marlin cache build without changing
quantization math, cache format, or runtime behavior.

## Baseline

Command:

```bash
./dev run tests/qcn-bf16.conf --build-cache
```

Cache:

```text
~/.krasis/cache/Qwen3-Coder-Next/experts_marlin_int4_g128_calamax.bin
```

Result:

| Phase | Time |
| --- | ---: |
| Routed layer loop | 537s |
| Full cache build | 544s |
| Post-build cache load | 30s |
| Routed load/quantize | 389.4s |
| Routed Marlin repack | 20.5s |
| Routed cache write | 44.6s |
| Shared load/quantize | 0.7s |
| Shared Marlin repack | 0.4s |
| Shared cache write | 0.1s |

Read: the main bottleneck was non-stacked routed expert load/quantize, not
Marlin tile repack.

## Change

Parallelized the non-stacked safetensors routed expert load/quantize loop across
experts within each layer using Rayon.

The change preserves:

- one-layer-at-a-time BF16 safetensor residency
- deterministic expert ordering from indexed parallel collection
- existing quantization math
- existing Marlin cache format

## After

Same command:

```bash
./dev run tests/qcn-bf16.conf --build-cache
```

Result:

| Phase | Before | After |
| --- | ---: | ---: |
| Routed layer loop | 537s | 169s |
| Full cache build | 544s | 176s |
| Post-build cache load | 30s | 31s |
| Routed load/quantize | 389.4s | 23.8s |
| Routed Marlin repack | 20.5s | 21.3s |
| Routed cache write | 44.6s | 46.8s |
| Shared load/quantize | 0.7s | 0.7s |
| Shared Marlin repack | 0.4s | 0.4s |
| Shared cache write | 0.1s | 0.1s |

## Validation

- `./dev build` passed.
- Both cold builds completed with `BUILD CACHE COMPLETE`.
- Optimized cache loaded successfully after build: `39.9 GB in 31s`.
- Optimized cache was byte-identical to:
  - the sequential baseline cache built immediately before the change
  - the original pre-run cache moved aside before measurement
- Temporary 38GB backup caches created during measurement were removed after
  byte-identical validation.

## Remaining Bottlenecks

The cache builder is now mostly:

- cache writes: ~47s
- Marlin repack: ~21s
- fixed startup/load overhead around the build

Further speedups are possible, but the first-order sequential quantization
bottleneck is removed.
