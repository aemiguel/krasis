# Phase 2BN: QCN 64-token attention divergence matrix

Date: 2026-04-27 UTC

## Goal

Measure real multi-token divergence for QCN attention modes against llama-witness, using the same controlled non-attention settings as Phase 2BM:

- KV cache: Polar4
- GPU routed experts: INT8
- CPU routed experts: INT8
- Shared/dense/lm-head weights: INT8
- Generation: deterministic greedy, thinking disabled
- Only intended variable: attention backend/mode

Variants:

- AWQ
- HQQ4
- HQQ4SC: HQQ4 plus explicit top-4 INT8 exception sidecar
- HQQ8: sidecar-free HQQ8

## Witness capture

The prior Phase 2BM artifact was first-token-only. For Phase 2BN, `krasis-llama-witness/tools/witness-validate/witness-validate.cpp` was extended to capture full greedy token sequences:

- New artifact: `/home/main/Documents/Claude/krasis-internal/reference-outputs/output/Qwen3-Coder-Next/phase2bn_qcn_64tok.json`
- Profile: `phase2bn_qcn_64tok`
- Format: `krasis_llama_witness_sequence`
- Format version: `7`
- Schema version: `2`
- Max new tokens: `64`
- Turns: `14`
- Reference token counts: `[12, 34, 64, 9, 11, 11, 64, 64, 64, 64, 64, 64, 64, 64]`

Some turns end before 64 tokens because llama-witness reached EOS/EOG.

## Commands

Capture:

```bash
KRASIS_WITNESS_MAX_NEW_TOKENS=64 ./dev witness-capture qcn --profile phase2bn_qcn_64tok --output /home/main/Documents/Claude/krasis-internal/reference-outputs/output/Qwen3-Coder-Next/phase2bn_qcn_64tok.json
```

Matrix:

```bash
./dev witness-compare logs/manual/phase2bm_qcn_accuracy_awq_20260426.conf --profile phase2bn_qcn_64tok
./dev witness-compare logs/manual/phase2bm_qcn_accuracy_hqq4_20260426.conf --profile phase2bn_qcn_64tok
./dev witness-compare logs/manual/phase2bm_qcn_accuracy_hqq4sc_20260426.conf --profile phase2bn_qcn_64tok
./dev witness-compare logs/manual/phase2bm_qcn_accuracy_hqq8_20260426.conf --profile phase2bn_qcn_64tok
```

## Results

Lower selected-logprob delta is better. Higher exact-prefix and decode top-k containment are better.

| Variant | First-token | Selected sum | Top-10 overlap | Avg exact prefix | Worst prefix | Full matches | Avg decode top-k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AWQ | 14/14 | 0.372141474328 | 123/140 | 14.57 | 4 | 4/14 | 55.0% |
| HQQ4 | 13/14 | 0.340154573046 | 129/140 | 7.43 | 0 | 3/14 | 42.5% |
| HQQ4SC | 13/14 | 0.405901880659 | 128/140 | 7.57 | 0 | 3/14 | 44.0% |
| HQQ8 | 14/14 | 0.129170737779 | 136/140 | 7.43 | 1 | 3/14 | 45.0% |

Exact-prefix runs by turn:

| Variant | Prefix runs |
| --- | --- |
| AWQ | `[12, 14, 32, 9, 11, 11, 33, 17, 5, 4, 10, 8, 31, 7]` |
| HQQ4 | `[12, 4, 5, 5, 11, 11, 18, 1, 15, 6, 2, 0, 7, 7]` |
| HQQ4SC | `[12, 4, 5, 5, 11, 11, 18, 2, 18, 4, 2, 0, 7, 7]` |
| HQQ8 | `[12, 7, 5, 8, 11, 11, 18, 1, 5, 4, 2, 8, 5, 7]` |

Run directories:

- AWQ: `/home/main/Documents/Claude/krasis-hqq/logs/reference-test_20260427_021702`
- HQQ4: `/home/main/Documents/Claude/krasis-hqq/logs/reference-test_20260427_022304`
- HQQ4SC: `/home/main/Documents/Claude/krasis-hqq/logs/reference-test_20260427_022705`
- HQQ8: `/home/main/Documents/Claude/krasis-hqq/logs/reference-test_20260427_023105`

## Interpretation

Phase 2BM's first-token-only matrix made HQQ8 look clearly best, and HQQ8 still wins the first-token distribution metrics here:

- Best selected-logprob delta sum: `0.129170737779`
- Best top-10 overlap: `136/140`
- Best first-token match count tied with AWQ: `14/14`

The 64-token divergence view changes the practical conclusion:

- AWQ has the strongest exact generation stability in this controlled matrix: average exact prefix `14.57`, worst prefix `4`, and `4/14` full sequence matches.
- HQQ8 is much better than HQQ4/HQQ4SC at first-token fidelity, but its multi-token exact-prefix average is the same as plain HQQ4 in this run: `7.43`.
- HQQ4SC gives only a small divergence improvement over HQQ4 (`7.57` vs `7.43` average prefix, `44.0%` vs `42.5%` decode top-k), and is worse than HQQ4 on selected-logprob delta in this 14-turn/64-token matrix.

For deterministic generation, exact-prefix length is the more useful signal than top-10 first-token overlap. On this run, AWQ is the current divergence winner; HQQ8 is the current first-token/logprob winner; HQQ4SC is not clearly better than HQQ4 once multi-token divergence is measured.

## Notes

No speed benchmark was run. This was an accuracy/divergence-only pass.

