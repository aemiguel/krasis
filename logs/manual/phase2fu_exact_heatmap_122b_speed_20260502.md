# Phase 2FU: 122B Exact-Heatmap Speed Retest

Date: 2026-05-02

## Goal

Retest Qwen3.5-122B-A10B after changing HCS heatmap generation so benchmark
startup uses the same decode parameters as benchmark runtime:

- `temperature=0.0`
- `top_k=50`
- `top_p=0.95`
- `enable_thinking=false`
- held-out `heatmap_prompts.txt`, not benchmark `decode_prompt_*`

This run tests whether the prior decode regression was caused primarily by the
heatmap using sampled decode (`temperature=0.6`) while the benchmark used
greedy decode.

## Command

```bash
./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf
```

Timing/debug instrumentation was disabled.

Run log:

- `logs/manual/phase2fu_q122b_exact_heatmap_benchmark_20260502.log`
- `logs/dev-benchmark_20260502_195820/benchmark_report.log`
- archived standard log:
  `benchmarks/20260502_phase2fu_q122b_k4v4_hqq6_exact_heatmap_benchmark.log`

## Heatmap Validation

The run did not use a supplied heatmap:

```text
heatmap_path = None
```

Startup rebuilt `auto_heatmap.json` under benchmark params:

```text
Building heatmap from 6 held-out prompts (256 decode tokens each,
temperature=0.000 top_k=50 top_p=0.950 enable_thinking=False mode=benchmark)
Heatmap saved to ~/.krasis/cache/Qwen3.5-122B-A10B/auto_heatmap.json
```

The saved heatmap includes `_metadata` with:

- `format = krasis_hcs_heatmap`
- `format_version = 2`
- Krasis version `0.1.66`
- runtime code hash
  `0528354c58243a35b85134cdb373d6f5e3675c38be346eca8512bafc8664f1df`
- prompt source `python/krasis/prompts/heatmap_prompts.txt`
- prompt set hash
  `efd41f7bff999bf99d51ecd1c1fbf48d138c0327a38f534561945a9e87fad6fc`
- runtime params including `attention_quant=hqq6`, `kv_dtype=k4v4`,
  `kv_cache_mb=2000`, `gpu_expert_bits=4`, `cpu_expert_bits=4`,
  `layer_group_size=2`, `selected_gpus=0`

## Result

| Metric | Phase 2FR | Phase 2FU |
| --- | ---: | ---: |
| Heatmap decode params | sampled old path | exact benchmark greedy |
| Startup HCS | `3888/12288 (31.6%)` | `3888/12288 (31.6%)` |
| Final HCS | `3456/12288 (28.1%)` | `3483/12288 (28.3%)` |
| Best prefill | `2070.2 tok/s` | `2030.4 tok/s` |
| Internal decode | `9.03 tok/s` | `8.85 tok/s` |
| Round trip | `14.62 tok/s` | `14.36 tok/s` |
| Decode min free | `672 MB` | `640 MB` |

Timed prefill emitted a lower transient VRAM warning than Phase 2FR:

```text
VRAM MONITOR: new low on cuda:0 -- 218 MB free
```

The benchmark still completed.

## Interpretation

Exact benchmark-parameter heatmap generation did not improve 122B decode speed.
It also did not materially change HCS pool size or final HCS coverage.

This falsifies the narrow hypothesis that the decode regression was mainly due
to heatmap sampler mismatch (`temperature=0.6` vs greedy). The heatmap metadata
and fail-closed validation are still correct engineering changes, but the
remaining 122B decode issue is deeper:

- held-out heatmap prompts still do not predict benchmark decode routes well, or
- HCS reload/eviction changes the effective resident set after each benchmark
  prefill, or
- graph recapture and pointer invalidation remain a significant per-request
  cost, or
- the HQQ6 decode route changes expert access patterns enough that the current
  heatmap score model is not predictive.

Prefill remains governed by the previously measured HQQ6 prefill
projection/dequant overhead plus tight scratch headroom. This run does not
change that diagnosis.

## Next Work

- Run state-validation on the Phase 2FU exact heatmap to compare resident HCS
  experts vs actual cold loads. The prior validation used a reused heatmap
  before Phase 2FT, so this should be repeated once under the exact metadata
  path.
- Add a benchmark-specific held-out heatmap prompt profile that is not identical
  to benchmark prompts but is closer in distribution/length.
- Investigate graph pointer invalidation after HCS reload independently of
  heatmap ranking.
- Reduce transient prefill VRAM pressure: the `218 MB` low suggests the 2 GB
  k4v4 config is too close to OOM for long prefill rows.
