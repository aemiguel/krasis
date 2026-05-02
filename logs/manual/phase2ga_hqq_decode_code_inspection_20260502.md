# Phase 2GA - HQQ Decode Code Inspection

Date: 2026-05-02

## Question

After Q122B HQQ6+k4v4 and Q122B HQQ8+k6v6 both failed seq32, QCN was rerun as a current-code control. QCN HQQ8 now fails even with BF16 KV, while older QCN HQQ8 runs were good. Inspect the recent HQQ changes and identify a plausible bad code path.

## Relevant Accuracy Data

| Surface | Exact | Containment | Notes |
| --- | ---: | ---: | --- |
| old QCN HQQ8+k6v6 | `562/653` | `572/653` | historical good control |
| old QCN HQQ8/BF16-KV | `421/653` | `491/653` | historical good control |
| current QCN HQQ8+k6v6 | `28/653` | `28/653` | current failure |
| current QCN HQQ8/BF16-KV | `28/653` | `28/653` | failure without compact KV |
| current QCN HQQ8/BF16-KV, `KRASIS_HQQ_STAGE_COMPACT=0` | `27/653` | `27/653` | compaction falsification |

The compaction-off run used:

```bash
KRASIS_HQQ_STAGE_COMPACT=0 ./dev witness-compare tests/qcn-bf16kv-hqq8-accuracy.conf \
  --profile phase2bn_qcn_64tok --startup-timeout 1200
```

Run dir: `logs/reference-test_20260502_223906`

## What Changed

Old passing QCN HQQ8 logs show:

- initial HQQ runtime staging
- normal attention registration later
- second HQQ runtime staging after shared decode setup
- `HQQ attention execution descriptors restored after shared decode setup`
- no BF16 attention projection residency release

Current failing QCN HQQ8 logs show:

- one HQQ runtime staging pass
- `HQQ BF16 attention projection residency released`
- normal attention registration skipped under HQQ
- `HQQ attention execution descriptors retained after shared decode setup`

## Code Finding

`register_la_layer()` and `register_gqa_layer()` fill normal `GpuAttnConfig` fields and explicitly clear HQQ descriptors. The old path therefore had to restore HQQ descriptors after normal registration.

The current path skips those normal registration calls. When `register_hqq_attention_layer_common()` runs without an existing normal `GpuAttnConfig`, it creates fallback attention configs with zero projection IDs:

- GQA: `q_proj = 0`, `k_proj = 0`, `v_proj = 0`, `o_proj = 0`, `fused_qkv = None`
- Linear attention: `in_proj_qkvz = 0`, `in_proj_ba = 0`, `out_proj = 0`

Several generic decode/setup paths still inspect projection IDs from `GpuAttnConfig`, including batch/projection buffer sizing and swap/graph/debug paths. With HQQ-only fallback configs, those paths can silently read `graph.weights[0]` or lose normal-attention metadata instead of failing closed.

That is a plausible catastrophic case. Old runs could pass because the normal BF16 attention IDs and tensors were still registered/resident before HQQ descriptors were restored, masking any hidden dependency on `GpuAttnConfig`.

## Current Interpretation

The primary suspect is not compact KV and not HQQ slot compaction. It is the HQQ-exclusive registration path:

1. Skips normal attention metadata registration.
2. Releases BF16 attention projections.
3. Leaves fallback zero projection IDs in `GpuAttnConfig`.
4. Relies on every runtime path to use `hqq_exec` perfectly.

The witness shape also fits a decode-state bug: first token and prefill are aligned, then generation collapses after about token 2.

## Next Debug Target

Use QCN HQQ8/BF16-KV as the smallest repro. The next code step should be either:

- restore normal attention metadata registration without allowing silent BF16 fallback, then restore HQQ descriptors after shared KV setup; or
- remove the need for normal projection IDs in HQQ mode by deriving all generic metadata from HQQ descriptors and making any normal-ID access fail closed.

Do not add a silent BF16 fallback. If needed, use an explicit diagnostic switch only to prove whether old resident-BF16 metadata masked the true-HQQ decode bug.
