# Phase 2GM: Prompt-HCS Blend Shadow

Date: 2026-05-03

## Goal

Measure whether prompt-conditioned expert usage can improve HCS soft-tier
ranking when used only to replace the lower end of the existing global heatmap.

This is an exact-execution diagnostic:

- No routing changes.
- No HCS residency changes.
- No cold DMA changes.
- No expert weights or outputs changed.
- Timing numbers are instrumentation-perturbed and not speed claims.

## Implementation

`KRASIS_PROMPT_HCS_SHADOW=1` now builds eight hypothetical same-capacity soft
tiers:

- Preserve the current hard tier.
- Preserve the current loaded soft capacity.
- Keep the top `10/20/30/40/50/60/70/80%` of the existing global heatmap soft
  ranking.
- Fill the remaining soft slots from prompt-count-ranked experts collected
  during exact prefill.
- Backfill from heatmap only if prompt candidates are exhausted.

Decode then compares exact current HCS hits/colds against each hypothetical
resident set.

## Run

Command:

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_PROMPT_HCS_SHADOW=1 ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf --timing
```

Run directory:

- `logs/dev-benchmark_20260503_171516`

Wrapper log:

- `logs/manual/phase2gm_q122b_prompt_hcs_blend_shadow_timing_20260503.log`

No CUDA/runtime errors were found.

## Official Internal Rows

| Retain heatmap | 49 tok shadow cold/tok | 99 tok shadow cold/tok | 249 tok shadow cold/tok | Weighted shadow cold/tok | Net hits/tok | Cold change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actual | `119.3` | `126.2` | `143.1` | `135.95` | `0.00` | `0.0%` |
| `10%` | `125.4` | `147.1` | `146.9` | `144.30` | `-8.35` | `-6.1%` |
| `20%` | `124.9` | `144.5` | `144.9` | `142.33` | `-6.38` | `-4.7%` |
| `30%` | `122.9` | `143.4` | `143.7` | `141.06` | `-5.11` | `-3.8%` |
| `40%` | `121.4` | `139.4` | `142.1` | `138.87` | `-2.92` | `-2.2%` |
| `50%` | `118.6` | `135.6` | `141.2` | `137.01` | `-1.07` | `-0.8%` |
| `60%` | `114.6` | `130.7` | `137.2` | `132.79` | `+3.16` | `+2.3%` |
| `70%` | `113.0` | `127.3` | `135.8` | `130.87` | `+5.08` | `+3.7%` |
| `80%` | `109.0` | `123.8` | `133.3` | `127.93` | `+8.02` | `+5.9%` |

## Interpretation

Prompt-only ranking regressed in Phase 2GL. This blended sweep shows why: the
prompt signal can help, but replacing too much of the global heatmap drops many
experts that the heatmap was already catching.

On this Q122B run:

- `10-50%` heatmap retention regressed hit rate.
- `60%` crossed positive.
- `70-80%` looked useful.
- `80%` was best among this sweep: about `+8.0` hits/tok and `5.9%` fewer cold
  experts/token.

If we implement a behavior-changing prompt-conditioned HCS reload, start with a
conservative policy that preserves at least `70-80%` of heatmap ranking and uses
prompt counts only for the lower tail.
