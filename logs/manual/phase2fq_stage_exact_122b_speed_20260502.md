# Phase 2FQ: Stage-Exact HQQ/KV 122B Speed Test

Date: 2026-05-02

## Goal

Build the stage-exact residency design for HQQ and k4/k6 KV, then start speed
testing on 122B before accuracy and 235B work.

## Code Changes

- HQQ runtime slots now default to stage compaction:
  - HQQ6 122B prefill format: `4904.93 MiB`
  - HQQ6 122B decode format: `3696.47 MiB`
  - active GPU device allocation after compaction: `3696.47 MiB`
- k4v4/k6v6 prefill uses temporary FP8 KV sized to the current prompt and
  bulk-exports to compact decode KV before decode.
- FA2 BF16-Q/FP8-KV sidecar build was fixed by restoring the missing FP8-KV
  template instantiation `.cu` files in `build.rs`.
- The invalid post-HCS measured chunk cap was removed from
  `PrefillEngine::prepare_for_prefill`. That cap reused startup min-free data
  that includes adaptive expert pinning and forced post-HCS 25K prefill to
  `128`-token chunks.

## Diagnostic Evidence

Post-FA2 bounded diagnostic:

- stage-exact KV active: `decode_format=9`, prefill `kv_format=1`
- prefill KV append: `fp8_append`
- GQA path: `fa2_bf16_fixed`
- 2,000-token bounded long calibration: `574.2 tok/s`

Post-HCS benchmark diagnostic before the chunk-cap fix:

- first 25K prefill warmup:
  - `measured_cap=128`
  - `chunk_size=128`
  - `num_chunks=196`
- this explained the minutes-scale warmup despite FA2 and FP8 prefill KV being
  active.

After removing the invalid cap, the timing-free standard benchmark completed.

## 122B Benchmark Result

Command:

```bash
./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf
```

Log:

- `logs/manual/phase2fq_q122b_stage_exact_chunkfix_benchmark_20260502.log`
- `benchmarks/20260502_phase2fq_q122b_k4v4_hqq6_stage_exact_chunkfix_benchmark.log`

Config:

- model: `Qwen3.5-122B-A10B`
- GPU experts: INT4 Marlin
- CPU experts: INT4
- attention: HQQ6
- KV: k4v4
- shared/dense/lm-head: INT8
- GPU: one RTX 5090 32 GB selected

Results:

| Metric | Result |
| --- | ---: |
| Best prefill | `1377.3 tok/s` |
| Best internal decode | `16.41 tok/s` |
| Best network round trip | `29.46 tok/s` |
| HCS coverage after benchmark | `1755/12288 (14.3%)` |
| Decode min free VRAM | `716 MB` |
| HCS loaded at startup | `2214/12288 (18.0%)` |
| HQQ6 active device residency | `3696.47 MiB` |

Timed prefill rows:

| Prompt | Speed | Time |
| ---: | ---: | ---: |
| 1,000 | `205.6 tok/s` | `4864.8 ms` |
| 5,000 | `1179.5 tok/s` | `4239.0 ms` |
| 10,000 | `1377.3 tok/s` | `7260.4 ms` |
| 20,000 | `1148.7 tok/s` | `17411.4 ms` |
| 35,000 | `1187.0 tok/s` | `29485.0 ms` |
| 50,000 | `686.1 tok/s` | `72872.6 ms` |

Decode rows:

| Target | Speed |
| ---: | ---: |
| 50 | `16.41 tok/s` |
| 100 | `16.37 tok/s` |
| 250 | `15.84 tok/s` |

## Interpretation

Stage-exact HQQ/KV fixed the catastrophic k4v4 prefill behavior and recovered
122B HQQ6/k4v4 from the previous broken benchmark row:

- Phase 2FM HQQ6/k4v4 prefill: `541.9 tok/s`
- Phase 2FQ stage-exact/chunkfix HQQ6/k4v4 prefill: `1377.3 tok/s`

This is a large improvement, but it is still below the old BF16-attention /
FP8-KV control surface (`2765.6 tok/s` prefill, `23.20 tok/s` decode).

Remaining issues:

- VRAM headroom is tight on the 2 GB k4v4 config. Warmup emitted monitor lows
  at `530 MB` and `512 MB`, below the configured `600 MB` safety margin.
- HCS coverage after benchmark is only `14.3%`, which limits decode speed.
- The 50K prefill row is much slower than 5K-35K, suggesting remaining
  long-context chunking or HCS/pinning interaction worth a follow-up diagnostic.

## Next Work

Before 235B speed work:

1. Decide whether 122B should use 2 GB k4v4 KV or a smaller KV budget for more
   safety headroom and HCS coverage.
2. Run accuracy against llama-witness for the stage-exact path.
3. Consider a targeted timing diagnostic for the 50K prefill row only, with
   instrumentation enabled, to explain the drop from `~1.18-1.38K tok/s` to
   `686 tok/s`.
