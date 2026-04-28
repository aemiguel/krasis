# Phase 2CP QCN HQQ8 Graph-Fix BF16/Polar4 Rerun - 2026-04-28

Both runs use `KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-twoscale-intercept`, graph replay enabled, INT8 experts, and `phase2bn_qcn_64tok`.

| Run | Avg exact | Total exact | Full matches | Containment | First token | Prefill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Old HQQ8 + Polar4 graph | `18.14` | `254/653` | `5/14` | `326/653` | `14/14` | `14/14` |
| Old HQQ8 + BF16 KV graph | `18.21` | `255/653` | `5/14` | `317/653` | `14/14` | `14/14` |
| Fixed graph HQQ8 + Polar4 | `30.5` | `427/653` | `8/14` | `459/653` | `14/14` | `14/14` |
| Fixed graph HQQ8 + BF16 KV | `36.5` | `511/653` | `10/14` | `554/653` | `14/14` | `14/14` |
| Fixed graph BF16 attention + BF16 KV | `34.64` | `485/653` | `10/14` | `558/653` | `14/14` | `14/14` |
| No-graph BF16 attention + BF16 KV | `37.43` | `524/653` | `9/14` | `566/653` | `14/14` | `14/14` |
| llama Q8_0 default | `37.5` | `525/653` | `11/14` | `n/a` | `14/14` | `n/a` |

## Commands

- BF16 KV: `KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-twoscale-intercept ./dev witness-compare tests/qcn-bf16kv-hqq8-accuracy.conf --profile phase2bn_qcn_64tok --startup-timeout 1200`
- Polar4 KV: `KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-twoscale-intercept ./dev witness-compare logs/manual/phase2bm_qcn_accuracy_hqq8_20260426.conf --profile phase2bn_qcn_64tok --startup-timeout 1200`

## Artifacts

- BF16 KV summary: `logs/reference-test_20260428_180715/reference_test_summary.json`
- Polar4 summary: `logs/reference-test_20260428_181329/reference_test_summary.json`
- Reduction JSON: `logs/manual/phase2cp_qcn_hqq8_graphfix_bf16_polar4_reduction_20260428.json`

## Notes

- The graph replay cold-expert fix substantially improves HQQ8 too.
- BF16 KV is stronger than Polar4 on this HQQ8 surface: `511/653` vs `427/653` exact tokens.
- Fixed graph HQQ8 + BF16 KV (`36.5`) is close to llama Q8 exact-prefix length (`37.5`) and the no-graph BF16-attention control (`37.43`).
- Fixed graph HQQ8 + Polar4 still has prompt-specific early rank swaps, most visibly `Who trained you?` and the longer whale follow-up.
