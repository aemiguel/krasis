# Phase 2GP: 35B Control and QCN Prompt-HCS Retain Sweep

Date: 2026-05-03

Purpose:
- Check that prompt-conditioned HCS reload basically does not change 35B,
  because 35B already has full HCS expert coverage.
- Run QCN timing-free `./dev speed-test` at prompt-HCS retain percentages
  `75/80/85/90`.

All runs were timing-free speed runs. No `--timing` instrumentation was enabled.

## Commands

35B:

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 ./dev benchmark tests/q35b-4-4-hqq6-benchmark.conf
```

QCN:

```bash
KRASIS_HQQ_PREFILL_MATERIALIZE_BF16=1 KRASIS_PROMPT_HCS_RETAIN_PCT=<75|80|85|90> ./dev speed-test
```

## 35B Result

Run dir: `logs/dev-benchmark_20260503_191846`

| Model | Prefill | Decode internal | Round trip | HCS | Min free |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-35B HQQ6 | `12127.0 tok/s` | `113.70 tok/s` | `230.83 tok/s` | `10240/10240` | `8990 MB` |

Prior Phase 2GI 35B row: prefill `11074.0 tok/s`, decode `113.32 tok/s`,
round trip `230.62 tok/s`, HCS `10240/10240`, min free `8990 MB`.

Interpretation: decode is effectively unchanged, as expected for a model with
full HCS expert residency.

## QCN Retain Sweep

All QCN runs used the standard fixed `./dev speed-test` surface:
QCN HQQ8 attention, k4v4 KV, INT4 experts, timing disabled.

| Retain pct | Run dir | Prefill | Decode internal | Round trip | HCS | Min free |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `75%` | `logs/dev-benchmark_20260503_192309` | `7875.8 tok/s` | `87.81 tok/s` | `143.15 tok/s` | `15147/24576` | `706 MB` |
| `80%` | `logs/dev-benchmark_20260503_192834` | `7910.7 tok/s` | `88.07 tok/s` | `142.08 tok/s` | `15147/24576` | `706 MB` |
| `85%` | `logs/dev-benchmark_20260503_193334` | `8178.1 tok/s` | `87.50 tok/s` | `153.63 tok/s` | `15147/24576` | `706 MB` |
| `90%` | `logs/dev-benchmark_20260503_193839` | `7993.1 tok/s` | `90.78 tok/s` | `136.69 tok/s` | `15147/24576` | `706 MB` |

Reference rows:
- Phase 2GI QCN pre-prompt-HCS row: prefill `8352.2 tok/s`, decode
  `81.02 tok/s`, round trip `165.05 tok/s`.
- Phase 2GO QCN default 85% row: prefill `8231.6 tok/s`, decode
  `89.37 tok/s`, round trip `145.21 tok/s`.

Interpretation:
- All four retain settings are materially above the Phase 2GI pre-prompt-HCS
  QCN decode baseline.
- In this pass, `90%` produced the best internal decode speed:
  `90.78 tok/s`.
- `85%` had the best round-trip number, but internal decode is the engine-side
  speed metric we care about.
- The `85%` repeat was lower than the earlier Phase 2GO 85% run
  (`87.50` vs `89.37`), so there is normal run-to-run variance. A final default
  change from `85%` to `90%` should be based on another paired A/B if we want to
  be strict.

## Errors

No `CUDA_ERROR`, `RuntimeError`, `Traceback`, `panic`, `ILLEGAL`,
`INVALID_CONTEXT`, or timeout markers were found in the 35B/QCN wrapper or
server logs checked after the runs.
