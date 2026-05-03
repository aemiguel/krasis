# Phase 2GO: Prompt-Conditioned HCS Reload Default

Date: 2026-05-03

## Change

Implemented prompt-conditioned HCS soft reload as the default decode residency
policy.

Behavior:
- preserve the hard tier exactly.
- keep the top `85%` of the global heatmap soft ranking.
- fill the lower soft tail from exact prefill expert counts for the current
  request.
- routing, selected experts, weights, DMA correctness, and generated-token
  logic remain exact.

Controls:
- default: enabled.
- disable: `KRASIS_PROMPT_HCS_RELOAD=0`.
- retain override: `KRASIS_PROMPT_HCS_RETAIN_PCT=<percent>`.
- verbose reload markers: `KRASIS_PROMPT_HCS_LOG=1`.

The measured `85%` default came from Phase 2GN shadow results, where
`85%` retained heatmap gave the best weighted Q122B hit-rate estimate among
`75/80/85/90%`.

## Accuracy

Q122B HQQ6+k4v4 seq32 witness:

- command: `KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 ./dev witness-compare tests/q122b-k4v4-hqq6-int4-benchmark.conf --profile llama_witness_q122b_seq32 --startup-timeout 1800`
- run dir: `logs/reference-test_20260503_182056`
- result: PASS
- first token: `14/14`
- prefill top-k: `14/14`
- exact generated prefix: `280/361`
- containment: `303/361`
- full exact: `7/14`

## Activation Check

A timing-free Q122B benchmark was run with explicit `[PROMPT-HCS]` markers
before the markers were gated behind `KRASIS_PROMPT_HCS_LOG=1`.

Observed markers:
- prefill collection enabled for benchmark prompts.
- prompt counts installed before HCS reload.
- reload plan used `retain_pct=85`.
- effective heatmap slots were `3456`.
- prompt tail repacked roughly `588-594` soft slots for short benchmark
  requests.

That run confirmed the implementation path was active. The final production
speed run below used marker logging off.

## Speed

Timing instrumentation was disabled.

### Q122B Production Default

Command:

`KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf`

Run dir: `logs/dev-benchmark_20260503_185520`

| Metric | Result |
| --- | ---: |
| Prefill best | `4880.4 tok/s` |
| Internal decode best | `25.29 tok/s` |
| Round trip best | `44.95 tok/s` |
| HCS | `3780/12288 (30.8%)` |
| Min free VRAM | `662 MB` |

Phase 2GH baseline:

| Metric | Phase 2GH | Phase 2GO |
| --- | ---: | ---: |
| Prefill best | `4689.8 tok/s` | `4880.4 tok/s` |
| Internal decode best | `24.80 tok/s` | `25.29 tok/s` |
| Min free VRAM | `662 MB` | `662 MB` |

### Q122B Reload-Off Control

Command:

`KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_PROMPT_HCS_RELOAD=0 ./dev benchmark tests/q122b-k4v4-hqq6-int4-benchmark.conf`

Run dir: `logs/dev-benchmark_20260503_184329`

| Metric | Result |
| --- | ---: |
| Prefill best | `4060.3 tok/s` |
| Internal decode best | `24.30 tok/s` |
| Round trip best | `43.63 tok/s` |
| HCS | `3780/12288 (30.8%)` |
| Min free VRAM | `662 MB` |

### QCN Speed-Test

Command:

`KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 ./dev speed-test`

Run dir: `logs/dev-benchmark_20260503_190404`

| Metric | Result |
| --- | ---: |
| Prefill best | `8231.6 tok/s` |
| Internal decode best | `89.37 tok/s` |
| Round trip best | `145.21 tok/s` |
| HCS | `15147/24576 (61.6%)` |
| Min free VRAM | `706 MB` |

Compared with Phase 2GI QCN speed-test:
- prefill: `8352.2 -> 8231.6 tok/s`
- internal decode: `81.02 -> 89.37 tok/s`
- HCS and min free VRAM unchanged.

## Conclusion

The `85%` prompt-conditioned HCS reload is safe on the Q122B witness gate and
improves the timing-free Q122B production-default benchmark without additional
VRAM pressure. It also does not regress the standard QCN speed-test decode path.

Keep it enabled by default, with `KRASIS_PROMPT_HCS_RELOAD=0` available as an
explicit opt-out and `KRASIS_PROMPT_HCS_RETAIN_PCT` available for further A/B
testing.
