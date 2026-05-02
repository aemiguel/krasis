# Phase 2FV: 122B Decode Slowdown Diagnosis

Date: 2026-05-02

## Scope

Investigate why Qwen3.5-122B HQQ6/k4v4 decode remains slow after:

- HQQ exclusive residency fix
- k4v4 stage-exact prefill/decode KV
- invalid measured prefill chunk cap removal
- exact benchmark-parameter heatmap generation and metadata validation

Primary comparison:

| Run | Internal decode | HCS |
| --- | ---: | ---: |
| BF16/fp8 control | `23.20 tok/s` | `3510/12288 (28.6%)` |
| HQQ6/k4v4 exact heatmap | `8.85 tok/s` | `3483/12288 (28.3%)` |
| Phase 2FV validation | `8.84 tok/s` | `3564/12288 (29.0%)` |

## Diagnostic Run

Command:

```bash
KRASIS_CONFIG_VALIDATION=1 KRASIS_STATE_VALIDATION=1 \
KRASIS_TRACE=1 KRASIS_TRACE_COMPONENTS=graph,hcs \
./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf \
  --heatmap-path ~/.krasis/cache/Qwen3.5-122B-A10B/auto_heatmap.json
```

Artifacts:

- `logs/manual/phase2fv_q122b_decode_validation_20260502.log`
- `logs/dev-benchmark_20260502_202758/benchmark_report.log`
- `logs/state-validation/prompt61_1777750434855_*`
- `logs/state-validation/prompt66_1777750445452_*`
- `logs/state-validation/prompt61_1777750461869_*`

The explicit heatmap path validated successfully against the current runtime metadata.

## Heatmap Integrity

The latest exact heatmap is not stale and not a first-token-only collection:

- metadata format: `krasis_hcs_heatmap` v2
- runtime: HQQ6/k4v4, `kv_cache_mb=2000`, benchmark greedy decode
- params: `temperature=0.0`, `top_k=50`, `top_p=0.95`, `enable_thinking=false`
- collected layers: `48`
- per-layer count: `12288`
- total routed events: `589824`
- expected events: `48 layers * 6 prompts * 256 decode tokens * topk 8 = 589824`

So the heatmap is complete for its held-out prompt set. The problem is not heatmap truncation or graph-capture-only collection.

## HCS Hit Quality

Current exact-heatmap HQQ6/k4v4 decode has enough HCS capacity, but the resident experts are poor matches for the benchmark decode routes.

| Decode row | Resident HCS | Cold expert events | HCS events | HCS hit % | Cold/tok | HCS/tok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 tokens | `3510` | `14179` | `4637` | `24.6%` | `289.4` | `94.6` |
| 100 tokens | `3537` | `26654` | `11362` | `29.9%` | `269.2` | `114.8` |
| 250 tokens | `3564` | `69458` | `26158` | `27.4%` | `278.9` | `105.1` |

BF16/fp8 control with similar resident count:

| Decode row | Cold/tok | HCS/tok | HCS hit % |
| --- | ---: | ---: | ---: |
| 50 tokens | `134.0` | `250.0` | `65.1%` |
| 100 tokens | `148.9` | `235.1` | `61.2%` |
| 250 tokens | `164.3` | `219.7` | `57.2%` |

The current run cold-loads roughly 2x as many experts per token as the old control.

The cold experts are not near misses in the global heatmap ranking. For the official decode rows, cold expert median heatmap ranks were:

- 50-token row: median rank `8262`
- 100-token row: median rank `8422.5`
- 250-token row: median rank `8193.5`

Examples of frequently cold-loaded experts:

- `(3, 91)` loaded `246` times in the 250-token row, heatmap rank `3609`
- `(13, 98)` loaded `246` times, heatmap rank `6182`
- `(15, 185)` loaded `245` times, heatmap rank `9128`

This means HCS is doing what the heatmap ranking says, but the ranking is not predictive enough for the benchmark decode prompt distribution on this HQQ6/k4v4 surface.

## Graph Recapture

A second issue is repeated graph recapture:

- `graph_pointer_check result=invalidate`: `13`
- `per_layer_capture status=ok`: `14`
- `graph_pointer_check result=reuse`: `0`

Every post-prefill decode request starts by recapturing all per-layer graphs.

The trace did not print pointer-change details because `invalidate_cuda_graph()` drains the graph list, and `verify_graph_pointers()` returns false immediately when `per_layer_graphs` is empty. The most likely source is the HQQ stage switch around prefill/decode:

- HQQ compaction is enabled by default unless `KRASIS_HQQ_STAGE_COMPACT=0`.
- Decode residency is compact (`3696.47 MiB`), while prefill staging is larger (`4904.93 MiB`).
- Switching HQQ slots between decode and prefill can change descriptor pointers and `prepare_runtime_for_decode_rust()` invalidates graphs if HQQ decode pointers change.

This graph issue is real, but the HCS hit data is the larger measured decode problem. Even with graph replay active after the first token, current HQQ6/k4v4 still has only `~25-30%` HCS hit coverage versus `~57-65%` in the control.

## Diagnosis

Decode remains slow primarily because HCS is caching the wrong experts for the actual decode routes on this surface.

The exact-parameter heatmap is valid and complete, but six held-out heatmap prompts do not generalize well enough to the benchmark decode prompts for HQQ6/k4v4. The soft reload then restores the top global heatmap experts after every prefill, while the actual decode needs many experts ranked in the lower half of the heatmap.

Secondary issue: graph recapture happens on every request after prefill/HCS reload. This adds overhead and should be fixed, but it does not explain the poor HCS hit mix by itself.

## Next Fix Targets

1. Add a request-adaptive HCS ranking source based on the current request's prefill routing/pinning data.
   - Do not use benchmark prompts in heatmap.
   - Use actual runtime request routing to prioritize reload after prefill.
   - Merge request-local hot experts with the global heatmap ranking.

2. Improve held-out heatmap coverage.
   - More prompts and/or more decode tokens.
   - Prompt set should remain held-out from benchmark prompts.
   - Metadata must include prompt hash and params as implemented in Phase 2FT.

3. Fix graph reuse diagnostics and then graph reuse.
   - Preserve enough captured pointer snapshot state to report the exact invalidation reason.
   - Avoid invalidating when only HCS expert residency changes.
   - For HQQ stage compaction, verify whether slot reallocation is forcing pointer changes; if so, use stable decode execution pointers or a graph-safe stage ownership policy.

4. Retest speed only after the HCS hit mix improves.
   - Target is to move current `~105 HCS/tok` back toward the BF16/fp8 control's `~220-250 HCS/tok`.
