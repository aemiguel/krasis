# Phase 2FZ - HQQ Decode Regression Cross-Model Check

Date: 2026-05-02

## Question

Q122B HQQ6+k4v4 and HQQ8+k6v6 collapse after the first generated tokens. This does not match prior QCN expectations, so check whether the failure is model-specific, compact-KV-specific, or a current-code HQQ decode regression.

## Current-Code Controls

Ran current-code QCN controls against the same `phase2bn_qcn_64tok` llama-witness artifact used by the historical QCN rows.

```bash
./dev witness-compare tests/qcn-k6v6-hqq8-accuracy.conf \
  --profile phase2bn_qcn_64tok --startup-timeout 1200

./dev witness-compare tests/qcn-bf16kv-hqq8-accuracy.conf \
  --profile phase2bn_qcn_64tok --startup-timeout 1200
```

Run dirs:

- `logs/reference-test_20260502_220413` - QCN HQQ8+k6v6
- `logs/reference-test_20260502_221254` - QCN HQQ8+BF16-KV

## Results

| Surface | Exact | Containment | Full exact | First token | Avg run |
| --- | ---: | ---: | ---: | ---: | ---: |
| QCN old HQQ8+k6v6 | 562/653 | 572/653 | 10/14 | 14/14 | 40.1 |
| QCN old HQQ8+BF16-KV | 568/653 | 576/653 | 11/14 | 14/14 | 40.6 |
| QCN current HQQ8+k6v6 | 28/653 | 28/653 | 0/14 | 14/14 | 2.0 |
| QCN current HQQ8+BF16-KV | 28/653 | 28/653 | 0/14 | 14/14 | 2.0 |
| Q122 current HQQ6+k4v4 | 28/361 | 28/361 | 0/14 | 14/14 | 2.0 |
| Q122 current HQQ8+k6v6 | 24/361 | 26/361 | 0/14 | 12/14 | 1.7 |

The current QCN failures have the same signature as Q122: prefill/first-token alignment remains strong, then generated decode collapses to about a two-token exact run.

## Interpretation

This is not a Q122B-specific model bug.

This is also not primarily compact KV. Current QCN HQQ8 with BF16 KV collapses just as badly as QCN HQQ8+k6v6.

The strongest current hypothesis is a recent HQQ decode regression. The timing matches the recent exclusive-HQQ residency work:

- HQQ modes now skip the normal BF16 attention registration path.
- BF16 attention projection tensors are released after HQQ descriptors are staged.
- The old passing QCN HQQ rows may have been using or depending on resident BF16 attention state even when HQQ descriptors existed.
- Current runs force true HQQ decode, and true HQQ decode diverges after the first generated tokens.

This does not prove the BF16 release itself is the bug; it proves the current HQQ decode path is not witness-aligned without the old resident BF16 path masking it.

## Next Debug Target

Stop speed work on HQQ modes until decode correctness is restored.

Use QCN first because it is smaller and has a known old-good HQQ8/BF16-KV witness baseline. The next diagnostic should compare current true-HQQ decode against the old BF16-backed path around generated token 2:

- HQQ linear-attention decode projections (`in_proj_qkvz`, `in_proj_ba`, `out_proj`)
- recurrent LA state update inputs/outputs
- whether any HQQ path still expects normal BF16 attention descriptors to be populated
- graph/non-graph decode consistency for HQQ layers

Do not add a silent BF16 fallback. If a BF16-backed diagnostic switch is needed, it should be explicit and diagnostic-only.
