# Phase 2GL - Prompt-Conditioned HCS Shadow

Date: 2026-05-03

## Goal

Measure whether exact prefill expert usage can build a better request-specific
HCS soft-tier reload list for decode, without changing routing, residency, DMA,
expert weights, or generated output.

This is the exact, quality-preserving alternative to reviving old APFL
speculative H2D prefetch. The old APFL failure mode was copy-engine contention;
this diagnostic only asks whether the HCS list itself would improve hit rate.

## Implementation

Added opt-in diagnostic flag:

```bash
KRASIS_PROMPT_HCS_SHADOW=1
```

When enabled:

- prefill accumulates actual `moe_count_experts` outputs per MoE layer across
  all prompt chunks.
- after exact prefill and HCS soft reload, decode builds a shadow resident set:
  - current hard tier is preserved.
  - current loaded soft capacity is reused.
  - soft candidates are ranked first by prompt counts.
  - remaining slots are backfilled by the existing heatmap ranking.
- decode records actual HCS hit/cold versus shadow hit/cold for every selected
  expert.

Execution remains exact. This diagnostic only counts hypothetical hits.

## Command

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 \
KRASIS_PROMPT_HCS_SHADOW=1 \
./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf --timing
```

Run dir:

```text
logs/dev-benchmark_20260503_165624
```

Wrapper log:

```text
logs/manual/phase2gl_q122b_prompt_hcs_shadow_timing_20260503.log
```

## Result

Official internal decode rows:

| Row | Actual cold/tok | Shadow cold/tok | Extra hits/tok | Lost hits/tok | Net hits/tok | Cold change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 tok | 120.8 | 127.4 | 57.9 | 64.5 | -6.6 | -5.5% |
| 100 tok | 128.2 | 145.7 | 55.7 | 73.2 | -17.5 | -13.6% |
| 250 tok | 148.5 | 144.5 | 67.9 | 63.9 | +4.0 | +2.7% |

Weighted across the official internal rows:

```text
tokens:          397
actual cold/tok: 140.02
shadow cold/tok: 142.69
net hits/tok:    -2.67
cold change:     -1.9%
```

## Interpretation

Prompt-only HCS ranking is not a hit-rate win on these benchmark prompts.
It finds many prompt-conditioned hits, but loses roughly as many or more experts
that the existing heatmap ranking already catches during decode.

This argues against replacing the global heatmap soft ranking with prompt-only
ranking. Better candidates:

- blend prompt counts with heatmap ranking instead of replacing it.
- use prompt counts only as a tie-breaker or boost.
- weight by decode-like evidence if we add exact short decode capture.
- measure per-layer where prompt counts help versus hurt before changing HCS.

No speed benchmark should be run from this diagnostic alone. The measured
same-capacity prompt-only list is not a cold-miss improvement.

## Validation

- `./dev build` passed.
- Q122B HQQ6+k4v4 timing diagnostic completed.
- No `CUDA_ERROR`, `RuntimeError`, or `Traceback` markers found in the run logs.
