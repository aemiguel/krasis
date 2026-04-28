# Phase 2CO QCN Graph Replay Cold Expert Fix - 2026-04-28

## Change

Fixed graph replay routed-expert materialization in `src/gpu_decode.rs`.

Before this patch, graph replay could only materialize two cold experts unless APFL slots were available. Any remaining cold top-k experts were silently skipped and replaced by zero-weight dummy expert slots. QCN uses `topk=10`, so under ~32% HCS coverage this removed real routed expert contributions from graph replay.

The patch adds dedicated graph replay cold expert buffers sized to `max_experts_per_tok`, DMAs every cold top-k expert into a distinct resident buffer before the captured batched expert graph launches, and hard-fails if graph replay cannot materialize exactly topk routed experts.

Follow-up parity fixes:

- graph replay increments `decode_step_counter` for diagnostics
- graph replay MoE now reads per-layer gated/expert dimensions instead of hardcoding gated hidden-sized experts
- latent MoE graph replay now errors visibly instead of running with hidden-sized assumptions
- final `./dev build` passed after the follow-up parity fixes

## Validation

Focused graph case:

- Command: `./dev witness-compare tests/qcn-bf16kv-a16-accuracy.conf --profile phase2cm_qcn_case1_fullrun_64tok --startup-timeout 1200`
- Run: `logs/reference-test_20260428_173837`
- Result: `PASS`, exact-prefix `34/34`

Post-follow-up focused graph case:

- Command: `./dev witness-compare tests/qcn-bf16kv-a16-accuracy.conf --profile phase2cm_qcn_case1_fullrun_64tok --startup-timeout 1200`
- Run: `logs/reference-test_20260428_175636`
- Result: `PASS`, exact-prefix `34/34`

Full graph-enabled 14-case witness:

- Command: `./dev witness-compare tests/qcn-bf16kv-a16-accuracy.conf --profile phase2bn_qcn_64tok --startup-timeout 1200`
- Run: `logs/reference-test_20260428_174322`
- Result: `PASS`

| Run | Avg exact prefix | Total exact | Full matches | Containment |
| --- | ---: | ---: | ---: | ---: |
| Old graph BF16 attention + BF16 KV | 18.79 | 263/653 | 5/14 | 338/653 |
| Fixed graph BF16 attention + BF16 KV | 34.64 | 485/653 | 10/14 | 558/653 |
| No-graph BF16 attention + BF16 KV | 37.43 | 524/653 | 9/14 | 566/653 |
| llama Q8_0 default | 37.50 | 525/653 | 11/14 | n/a |

## Interpretation

The missing-cold-expert bug was the main graph replay accuracy reducer. Graph replay now recovers most of the gap and fixes the focused divergence case.

A smaller graph/no-graph gap remains (`485/653` vs `524/653`). The remaining difference is likely graph replay MoE math/order parity rather than missing experts. The graph path still computes the routed expert batch in one captured batched graph, while normal decode computes HCS experts batched and cold experts sequentially with two DMA buffers.
