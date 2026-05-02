# Phase 2FN: 122B/235B Slow Benchmark Diagnosis

Date: 2026-05-02

## Question

The fresh 122B and 235B HQQ6/k4v4 benchmark attempts were much slower than the README v0.1.63 release rows.

README baseline:

| Model | README prefill | README decode |
| --- | ---: | ---: |
| Qwen3.5-122B-A10B | 2897 tok/s | 27.7 tok/s |
| Qwen3-235B-A22B | 2124 tok/s | 9.3 tok/s |

Fresh Phase 2FM HQQ6/k4v4 result:

| Model | Surface | Result |
| --- | --- | --- |
| 122B | INT4 experts, HQQ6 attention, k4v4 KV, 2 GB KV | 541.9 tok/s prefill, 17.66 tok/s decode |
| 235B | INT4 experts, HQQ6 attention, k4v4 KV, 1 GB KV | loaded/calibrated, but did not complete first 17,324-token benchmark warmup in a practical window |

## Key Diagnosis

The slow result is not evidence that current Krasis is globally slower than the previous release. It is primarily the selected benchmark surface:

1. k4v4 KV is very slow during long prefill because it quantizes and packs prompt K/V into compact cache format during append.
2. HQQ6 attention consumes substantial extra VRAM for prefill/runtime staging, reducing HCS expert coverage and hurting decode.
3. The README 122B row matches a BF16-attention / fp8-KV style surface, not the HQQ6/k4v4 surface.

## 122B Control Evidence

Recent Phase 2EV control, same current code, using `tests/q122b-4-4-a16.conf`:

- BF16 attention
- `fp8_e4m3` KV
- 1 GB KV cache
- INT4 experts

Measured:

- startup long calibration: 39,920 prompt tokens in 16.46s = 2425.3 tok/s
- benchmark best prefill: 2765.6 tok/s
- benchmark best decode: 23.20 tok/s

That is close to the README 122B row and proves the current runtime can still produce the old-class 122B speed on the old surface.

## k4v4 Isolation Evidence

Ran BF16 attention with k4v4 KV using `tests/q122b-k4v4-a16-accuracy.conf`:

- BF16 attention
- k4v4 KV
- 1 GB KV cache
- INT4 experts

The run reached long calibration and then stayed GPU-bound for more than four minutes on the same 39,920-token prefill that BF16/fp8 completed in 16.46s. It was stopped as an isolation run, not a completed benchmark.

This isolates the long-prefill collapse to the k4v4 prefill KV path, independent of HQQ6 attention.

## HQQ6/k4v4 Timing Evidence

Diagnostic run:

```bash
KRASIS_STARTUP_DIAG=1 KRASIS_PREFILL_TIMING=1 \
  ./dev run tests/q122b-k4v4-hqq6-int4-benchmark.conf
```

This was an attribution run, not a speed benchmark.

Short prompts:

| Prompt | Total | Main bottleneck |
| --- | ---: | --- |
| 300 tokens | 618.6 ms, 485 tok/s | MoE DMA wait: 303.2 ms |
| 500 tokens | 1954.8 ms, 256 tok/s | MoE DMA wait: 1336.1 ms |

Long prompt:

| Prompt | Total | Main bottleneck |
| --- | ---: | --- |
| 39,920 tokens | 195,866.6 ms, 204 tok/s | GQA KV append: 172,239.1 ms |

Long-prompt breakdown:

- total prefill: 195.9s
- attention: 177.1s
- GQA KV prep: 172.2s
- GQA KV append: 172.2s over 96 calls
- HQQ projection internals: 3.7s total
- MoE: 18.6s
- MoE DMA wait: 15.2s

So HQQ projection math is not the primary long-prefill bottleneck. The k4v4 append path is.

## Code-Level Difference

FP8 append is a simple per-token store/conversion:

- `src/cuda/prefill_kernels.cu`: `kv_cache_append_kernel`
- BF16 K/V -> FP8 K/V

k4v4 append does much more work for every prompt token:

- `src/cuda/prefill_kernels.cu`: `kv_cache_append_k4v4_kernel`
- quantizes K in 16-value blocks
- writes per-block K scales and packed 4-bit indices
- transforms V with Polar4/FHT work
- writes per-block V radius and packed angles

This is useful for compact decode KV, but expensive when performed during full prompt ingestion.

## VRAM/HCS Effect

122B BF16/fp8 control:

- startup free VRAM: 17,176 MB
- reclaimable HCS budget: 16,490 MB
- HCS coverage: 3510/12288 experts, 28.6%

122B HQQ6/k4v4:

- startup free VRAM: 11,138 MB
- HQQ runtime staging:
  - prefill materialization: 4904.93 MB
  - decode cache: 3696.47 MB
- reclaimable HCS budget: 10,384 MB
- HCS coverage: 2187/12288 experts, 17.8%

Q235 HQQ6/k4v4 is worse:

- HQQ6 runtime staging:
  - prefill materialization: 10,539.75 MB
  - decode cache: 7,943.00 MB
- 1 GB k4v4 retry reclaimable HCS budget: 2180 MB
- HCS coverage: 234/12032 experts, 1.9%

That explains the decode slowdown: HQQ6 reduces available HCS expert residency enough that decode falls back to far more cold/soft expert movement.

## Conclusion

The poor 122B/235B numbers are caused by the requested HQQ6/k4v4 surface, not by a general regression in the old BF16/fp8 surface.

For prefill:

- k4v4 prompt-time KV append is the main long-prefill bottleneck.
- It is doing real compact-KV quantization work during prompt ingestion.

For decode:

- HQQ6 attention consumes enough VRAM to reduce HCS coverage.
- On Q235 this is severe enough that only 1.9% of experts fit in the HCS soft pool.

## Next Work

1. Do not update README speed claims from HQQ6/k4v4.
2. Treat k4v4 prefill append as the first optimization target if k4v4 must be a speed surface.
3. Test k6v6/BF16-attention and BF16-KV controls before choosing the next public benchmark surface.
4. For Q235, avoid HQQ6 attention as the speed benchmark surface unless its decode-store VRAM cost is reduced.
5. If compact KV is required, investigate a faster prefill strategy:
   - optimize/fuse k4v4 append kernel
   - avoid storing compact KV during pure prefill benchmark when decode is not needed
   - append BF16 transiently and compact asynchronously or after prefill
   - reduce per-call overhead and improve chunk-level append scheduling
