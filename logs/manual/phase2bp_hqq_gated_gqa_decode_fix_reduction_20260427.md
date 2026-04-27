# Phase 2BP HQQ gated GQA decode fix reduction - 2026-04-27

## Scope

Diagnose the suspicious QCN HQQ8 64-token divergence result from Phase 2BN.

The failing signal was:

- HQQ8 first-token quality was strong.
- HQQ8 full-prefix prefill matched witness top-1 for forced reference prefixes.
- HQQ8 token-time generation averaged only `7.43` exact-prefix tokens.
- BF16 with the same INT8 expert / Polar4 KV control averaged `17.14`.

## Diagnosis

The bug was in gated GQA decode for HQQ fused-QKV tensors.

For gated GQA:

1. raw Q+gate must be split from `[head, q, gate]` form before QK norm/RoPE.
2. `split_gated_q` intentionally reads from `d_gqa_out` to avoid aliasing.
3. BF16 fused-QKV copied raw Q+gate from `d_gqa_q` to `d_gqa_out` before split.
4. HQQ fused-QKV wrote raw Q+gate into `d_gqa_q`, but skipped that copy.

So HQQ gated GQA split stale scratch contents instead of the HQQ fused-QKV projection output.

Trace symptom:

- HQQ trace showed `gqa_gated_q_pre_split` with the expected raw projection.
- `gqa_q_post_split` then matched stale scratch-like values, not the raw projection.

## Fix

Updated `src/gpu_decode.rs` gated GQA split setup:

- copy `d_gqa_q -> d_gqa_out` whenever the raw Q+gate input is not already in `d_gqa_out`
- preserve the HQQ split-Q path unchanged, because it already writes raw Q+gate directly to `d_gqa_out`

## Validation

Build:

- `./dev build` passed.

QCN controlled 64-token matrix, same Phase 2BM settings:

- Polar4 KV
- GPU/CPU routed experts INT8
- shared/dense/lm-head INT8
- same `phase2bn_qcn_64tok` llama-witness artifact

| Variant | Selected sum | Top-overlap | Avg exact prefix | Worst | Full matches |
| --- | ---: | ---: | ---: | ---: | ---: |
| HQQ8 before | `0.129170737779` | `136/140` | `7.43` | `1` | `3/14` |
| HQQ8 fixed | `0.065985003411` | `134/140` | `16.43` | `2` | `4/14` |
| BF16 control | `0.132351834630` | `135/140` | `17.14` | `4` | `5/14` |
| AWQ control | `0.372141474328` | `123/140` | `14.57` | `4` | `4/14` |

HQQ4 safety run:

| Variant | Selected sum | Top-overlap | Avg exact prefix | Worst | Full matches |
| --- | ---: | ---: | ---: | ---: | ---: |
| HQQ4 before | `0.340154573046` | `129/140` | `7.43` | `0` | `3/14` |
| HQQ4 fixed | `0.383182601676` | `128/140` | `11.79` | `0` | `4/14` |

## Conclusion

The original HQQ8 result was a real implementation bug, not normal quantization drift.

HQQ8 is now close to BF16 on 64-token exact-prefix stability and better than BF16 on first-token selected-logprob sum in this controlled QCN matrix. Remaining differences are plausible quantization/runtime drift rather than the severe stale-buffer failure seen before.

HQQ4 also benefits in exact-prefix stability, though its selected-logprob sum worsens slightly in this run.

## Artifacts

- HQQ8 fixed run: `logs/reference-test_20260427_084817/reference_test_summary.json`
- HQQ4 fixed run: `logs/reference-test_20260427_085538/reference_test_summary.json`
- HQQ8 no-graph discriminator: `logs/reference-test_20260427_084018/reference_test_summary.json`
- HQQ8 trace log: `logs/manual/phase2bp_hqq8_nograph_trace_server_20260427.log`
- BF16 trace log: `logs/manual/phase2bp_bf16_trace_server_20260427.log`
