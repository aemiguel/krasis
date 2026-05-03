# Phase 2GN Prompt-HCS High-Retention Shadow

Date: 2026-05-03

Goal: extend the Phase 2GM blended prompt-HCS shadow sweep with high-retention
points `75/85/90%`, because the first blend run showed the useful region starts
around `70-80%` heatmap retention.

## Contract

- Exact execution only.
- No routing changes.
- No HCS residency changes.
- No cold DMA changes.
- No weight/output changes.
- Same hard tier and same loaded soft capacity.
- Each shadow variant keeps the top N% of the current heatmap-ranked soft tier,
  then fills the lower tail from prompt-prefill expert counts.

## Code Change

- Extended `PromptHcsShadowStats::RETAIN_PCTS` from
  `10/20/30/40/50/60/70/80` to
  `10/20/30/40/50/60/70/75/80/85/90`.

## Run

Command:

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_PROMPT_HCS_SHADOW=1 ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf --timing
```

Artifacts:

- Run dir: `logs/dev-benchmark_20260503_173124`
- Wrapper log: `logs/manual/phase2gn_q122b_prompt_hcs_highretain_shadow_timing_20260503.log`

The run completed cleanly with no CUDA/runtime errors. Timing numbers from this
run are diagnostic only because `--timing` and shadow instrumentation were
enabled.

## Official Internal Rows

Rows used: generated `49/99/249` decode rows from the internal benchmark
section, not warmup and not network round-trip rows.

| Generated | Actual cold/tok | 75% net hits/tok | 80% net hits/tok | 85% net hits/tok | 90% net hits/tok |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 49 | 124.0 | +8.1 | +9.8 | +11.0 | +11.6 |
| 99 | 127.8 | +2.6 | +3.9 | +3.7 | +5.7 |
| 249 | 143.5 | +7.5 | +9.2 | +9.9 | +8.9 |

Weighted by generated tokens:

| Retain heatmap | Shadow cold/tok | Net hits/tok | Cold reduction |
| ---: | ---: | ---: | ---: |
| 10% | 148.00 | -10.82 | -7.9% |
| 20% | 146.59 | -9.41 | -6.9% |
| 30% | 144.59 | -7.41 | -5.4% |
| 40% | 141.35 | -4.17 | -3.0% |
| 50% | 139.39 | -2.21 | -1.6% |
| 60% | 135.85 | +1.33 | +1.0% |
| 70% | 132.73 | +4.45 | +3.2% |
| 75% | 130.85 | +6.33 | +4.6% |
| 80% | 129.23 | +7.95 | +5.8% |
| 85% | 128.69 | +8.49 | +6.2% |
| 90% | 128.74 | +8.44 | +6.1% |

Weighted actual cold/tok: `137.18`.

## Interpretation

The high-retention points confirm the earlier direction: prompt counts are
useful as lower-tail replacement, not as a replacement for the global heatmap.

`85%` was the best measured point in this run, narrowly ahead of `90%` and
`80%`. The plateau is tight enough that a behavior-changing HCS reload
experiment should start with `85%`, while keeping `80%` and `90%` as immediate
A/B controls.

This is still a hit-rate shadow result, not a speed or quality result. The next
step is an opt-in real HCS reload mode that applies the blended list after
prefill, then runs witness accuracy and a timing-free benchmark.
