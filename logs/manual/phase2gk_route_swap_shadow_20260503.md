# Phase 2GK - Route Swap Shadow Measurement

Date: 2026-05-03

## Goal

Measure the proposed approximate routing optimization before changing model
behavior:

- preserve the top 75-80% selected experts
- for lower-rank selected experts, if the selected expert is cold, estimate
  replacing it with a same-layer HCS-resident expert with similar router score
- keep execution exact during measurement

## Implementation

Added `KRASIS_ROUTE_SWAP_SHADOW=1` in `src/gpu_decode.rs`.

The diagnostic:

- does not change selected expert IDs, weights, HCS lookup, cold DMA, pointer
  uploads, or expert compute
- reconstructs router scores from full router logits during decode only when
  enabled
- protects the top configured rank fraction, default `75%`
- evaluates lower-rank cold selected experts against HCS-resident alternatives
  not already selected
- default match tolerance: `max(0.005 absolute, 10% relative)`

Additional knobs:

- `KRASIS_ROUTE_SWAP_PROTECT_RANK_PCT`
- `KRASIS_ROUTE_SWAP_ABS_TOL`
- `KRASIS_ROUTE_SWAP_REL_TOL_PCT`

## Run

Command:

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 \
KRASIS_ROUTE_SWAP_SHADOW=1 \
./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf --timing
```

Run directory:

- `logs/dev-benchmark_20260503_155951`
- wrapper log: `logs/manual/phase2gk_q122b_route_swap_shadow_timing_20260503.log`

This was a timing/instrumentation run, not a timing-free speed benchmark.

## Results

Official internal decode rows:

| Row | Exact cold/tok | Estimated saved/tok | Saved % of cold | Matched low-rank cold | Affected weight/tok | Avg score delta | Avg rel delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 tok | `122.2` | `19.6` | `16.0%` | `959/1892` (`50.7%`) | `1.575816` | `0.004083` | `5.05%` |
| 100 tok | `128.7` | `20.6` | `16.0%` | `2035/4233` (`48.1%`) | `1.673907` | `0.004258` | `5.27%` |
| 250 tok | `145.2` | `21.1` | `14.5%` | `5243/11328` (`46.3%`) | `1.709310` | `0.004323` | `5.35%` |

Weighted over the three official internal rows:

- exact cold experts/token: `138.24`
- estimated avoidable cold experts/token: `20.75`
- estimated cold reduction: `15.0%`
- affected matched router weight mass: `1.684/token` across all layers

Benchmark timing with instrumentation enabled:

- prefill best: `4975.7 tok/s`
- internal decode best: `23.85 tok/s`
- HCS: `3780/12288 (30.8%)`
- min decode free: `662 MB`

These speed numbers are diagnostic only because the shadow path copies router
logits and runs extra CPU analysis.

## Interpretation

The upper bound is meaningful: the policy finds roughly `20-21` cold expert
selections per token that could be replaced by HCS-resident experts with close
router scores. That is about `15%` of current Q122B cold expert traffic in the
official rows.

This is not accuracy-neutral. The shadow diagnostic only proves that similar
router-score HCS alternatives exist. It does not prove the substitute experts
produce equivalent activations.

## Recommendation

The measurement is strong enough to justify an opt-in approximate swap
prototype, but not default behavior.

Next implementation should:

- remain disabled by default
- preserve the top protected ranks exactly
- apply only to lower-rank cold experts with a matched resident alternative
- emit exact swap counts and affected mass
- run Q122B witness first, then QCN/35B regression if Q122B passes
- compare timing-free decode speed only after correctness gates pass
