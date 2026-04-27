# Phase 2BQ Q35 64-token controlled attention matrix

Date: 2026-04-27

Purpose: rerun the controlled 64-token llama-witness attention comparison on
Qwen3.5-35B-A3B after the HQQ gated-GQA decode fix and after narrowing the
interactive launcher path to HQQ8/HQQ4SC.

Reference artifact:
`/home/main/Documents/Claude/krasis-internal/reference-outputs/output/Qwen3.5-35B-A3B/phase2bq_q35_64tok.json`

Reference profile:
`phase2bq_q35_64tok`

Common non-attention settings:

- `CFG_KV_DTYPE=polar4`
- `CFG_GPU_EXPERT_BITS=8`
- `CFG_CPU_EXPERT_BITS=8`
- `CFG_SHARED_EXPERT_BITS=int8`
- `CFG_DENSE_MLP_BITS=int8`
- `CFG_LM_HEAD_BITS=int8`
- thinking disabled
- greedy reference-test comparison

## Results

| Variant | First-token | Selected sum | Top-10 overlap | Avg exact prefix | Worst prefix | Full matches | Avg decode top-k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AWQ control | `10/10` | `0.522607635548` | `81/100` | `8.90` | `2` | `2/10` | `64.0%` |
| HQQ4 | `10/10` | `0.540509303189` | `88/100` | `8.60` | `1` | `3/10` | `50.2%` |
| HQQ4SC | `10/10` | `0.544169848590` | `90/100` | `12.80` | `1` | `3/10` | `65.2%` |
| HQQ8 | `10/10` | `0.138064427202` | `97/100` | `10.40` | `4` | `4/10` | `71.2%` |
| BF16 control | `10/10` | `0.108668261465` | `93/100` | `15.90` | `4` | `5/10` | `80.3%` |

## Interpretation

- HQQ8 is the strongest quantized mode for first-token probability fidelity on
  Q35: selected-logprob sum is far lower than HQQ4/HQQ4SC/AWQ, and top-10
  overlap is highest at `97/100`.
- HQQ4SC behaves as hoped as a low-memory option: it improves average exact
  prefix over plain HQQ4 from `8.60` to `12.80`, beats AWQ on this exact-prefix
  metric, and substantially improves decode top-k containment.
- HQQ8 does not win average exact-prefix length on this Q35 profile. HQQ4SC is
  higher (`12.80` vs `10.40`), while BF16 remains the control winner at
  `15.90`.
- The Q35 result is therefore more nuanced than QCN: HQQ8 remains the best
  clean/default attention path by logprob/top-k fidelity, but HQQ4SC is a real
  useful alternate mode and should not be dismissed.
- This was not a speed benchmark. Durations include normal witness setup,
  calibration, and serving overhead.

## Run directories

- AWQ: `logs/reference-test_20260427_112904`
- HQQ4: `logs/reference-test_20260427_113313`
- HQQ4SC: `logs/reference-test_20260427_113712`
- HQQ8: `logs/reference-test_20260427_114111`
- BF16: `logs/reference-test_20260427_114822`

## HCS / attention residency notes

- AWQ HCS: `7585/10240` experts, `74.1%`
- HQQ4 HCS: `6724/10240` experts, `65.7%`; HQQ staged attention `789.61 MB`
- HQQ4SC HCS: `6724/10240` experts, `65.7%`; HQQ staged attention `789.61 MB`
- HQQ8 HCS: `6519/10240` experts, `63.7%`; HQQ staged attention `1491.48 MB`
- BF16 HCS: `7011/10240` experts, `68.5%`; resident attention `2090 MB`

The HCS differences are expected from the attention memory footprint. They may
contribute to sequence-level divergence, but HQQ8 still has much better
first-token/top-k fidelity than HQQ4/HQQ4SC/AWQ under the same control settings.
