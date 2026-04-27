# Phase 2BP HQQ4SC Post-Fix 64-Token Matrix

Date: 2026-04-27

## Purpose

Rerun HQQ4SC after the Phase 2BP gated-GQA fused-QKV decode fix. The old Phase
2BN HQQ4SC divergence numbers were collected before the shared HQQ decode bug was
fixed and are now stale.

## Command

```bash
./dev witness-compare logs/manual/phase2bm_qcn_accuracy_hqq4sc_20260426.conf --profile phase2bn_qcn_64tok
```

## Controlled Settings

- Model: Qwen3-Coder-Next
- Reference: `phase2bn_qcn_64tok`
- KV dtype: Polar4
- GPU/CPU routed experts: INT8
- Shared expert, dense MLP, LM head: INT8
- Attention: HQQ4
- Sidecar: explicit top-4 INT8 exception manifest
- Speed/timing claim: none; this was an accuracy/divergence run only

## Result

| Variant | First-token | Selected sum | Top-10 overlap | Avg exact prefix | Worst prefix | Full matches | Avg decode top-k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AWQ | 14/14 | 0.372141474328 | 123/140 | 14.57 | 4 | 4/14 | 55.0% |
| BF16 | 14/14 | 0.132351834630 | 135/140 | 17.14 | 4 | 5/14 | 61.4% |
| HQQ4 fixed | 13/14 | 0.383182601676 | 128/140 | 11.79 | 0 | 4/14 | 51.5% |
| HQQ4SC fixed | 13/14 | 0.380880880659 | 129/140 | 14.29 | 0 | 4/14 | 53.3% |
| HQQ8 fixed | 14/14 | 0.065985003411 | 134/140 | 16.43 | 2 | 4/14 | 59.1% |

## HQQ4SC Delta

Compared to stale Phase 2BN HQQ4SC:

- Avg exact prefix improved from 7.57 to 14.29.
- Selected sum improved from 0.405901880659 to 0.380880880659.
- Top-10 overlap improved from 128/140 to 129/140.
- Full matches improved from 3/14 to 4/14.
- First-token remained 13/14 with prompt 12 still missing at token 0.

Compared to fixed HQQ4:

- Avg exact prefix improved from 11.79 to 14.29.
- Selected sum improved slightly from 0.383182601676 to 0.380880880659.
- Top-10 overlap improved from 128/140 to 129/140.
- Avg decode top-k improved from 51.5% to 53.3%.
- First-token, worst prefix, and full-match count remained the same.

## Interpretation

HQQ4SC benefits from the shared HQQ gated-GQA decode fix and is now meaningfully
better than fixed plain HQQ4 on exact-prefix divergence. It is still not clearly
better than HQQ8 or BF16 in this controlled 64-token matrix. HQQ8 remains the
best HQQ option by selected-logprob and is close to BF16 on exact-prefix average.

## Artifacts

- Run summary: `logs/reference-test_20260427_093541/reference_test_summary.json`
- Run log: `logs/manual/phase2bp_hqq4sc_fixed_matrix_20260427.log`
