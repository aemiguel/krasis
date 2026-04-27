# Phase 2BM - QCN Attention Accuracy Matrix

## Decision

Run the relative attention accuracy comparison as a fixed QCN matrix with all
non-attention settings held constant:

- model: Qwen3-Coder-Next
- witness artifact: `llama_witness_stage3_qcn_expanded`
- prompts: 8 existing QCN expanded llama-witness cases
- KV cache: `polar4`
- routed experts: GPU INT8 and CPU INT8
- shared expert / dense MLP / LM head: INT8
- thinking: disabled
- no speed benchmark and no timing instrumentation

The only intentional differences were:

| Variant | Attention setting |
| --- | --- |
| AWQ | `CFG_ATTENTION_QUANT=awq` |
| HQQ4 | `CFG_ATTENTION_QUANT=hqq4` |
| HQQ4SC | HQQ4 plus explicit top-4 INT8 exception sidecar |
| HQQ8 | `CFG_ATTENTION_QUANT=hqq8`, no sidecar |

## Result Summary

Lower selected-logprob sum is better.

| Variant | First-token | Selected sum | Selected avg | Top-k overlap | Exact top-id order | Mean common logprob delta | Worst common logprob delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AWQ | 8/8 | 0.089002684241 | 0.011125335530 | 75/80 | 0/8 | 0.587293348503 | 1.975271108850 |
| HQQ4 | 8/8 | 0.097168526459 | 0.012146065807 | 74/80 | 2/8 | 0.424763662518 | 2.456171084193 |
| HQQ4SC | 8/8 | 0.061853578719 | 0.007731697340 | 71/80 | 1/8 | 0.382802682029 | 2.035900066476 |
| HQQ8 | 8/8 | 0.034590591524 | 0.004323823941 | 76/80 | 1/8 | 0.197510577807 | 0.810552084193 |

Ranking by selected-logprob sum:

1. HQQ8
2. HQQ4SC
3. AWQ
4. HQQ4

HQQ8 is the clear winner in this first-token matrix: best selected-logprob,
best top-k overlap, best mean common-logprob delta, and best worst common-logprob
delta.

HQQ4SC improves selected-logprob over plain HQQ4 and AWQ, but loses top-k
overlap versus both. The top-4 self-correction manifest is useful but still not
as clean as HQQ8.

## Per-Prompt Selected-Logprob Delta

Lower is better.

| Prompt | AWQ | HQQ4 | HQQ4SC | HQQ8 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.000795976533 | 0.001295976533 | 0.002197976533 | 0.000095023467 |
| 2 | 0.000145259195 | 0.000111740805 | 0.000165740805 | 0.000202259195 |
| 3 | 0.000016264305 | 0.000007735695 | 0.000005735695 | 0.000005264305 |
| 4 | 0.016577596703 | 0.017248596703 | 0.018040596703 | 0.002745596703 |
| 5 | 0.010436555391 | 0.000704444609 | 0.002427555391 | 0.006010555391 |
| 6 | 0.032572029262 | 0.044971029262 | 0.001865970738 | 0.012975970738 |
| 7 | 0.001699040564 | 0.000519040564 | 0.001012040564 | 0.000088959436 |
| 8 | 0.026759962289 | 0.032309962289 | 0.036137962289 | 0.012466962289 |

## Per-Prompt Top-K Overlap

| Prompt | AWQ | HQQ4 | HQQ4SC | HQQ8 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10/10 | 9/10 | 8/10 | 10/10 |
| 2 | 9/10 | 9/10 | 9/10 | 9/10 |
| 3 | 10/10 | 10/10 | 10/10 | 10/10 |
| 4 | 9/10 | 10/10 | 10/10 | 10/10 |
| 5 | 8/10 | 8/10 | 8/10 | 9/10 |
| 6 | 10/10 | 9/10 | 9/10 | 10/10 |
| 7 | 10/10 | 9/10 | 9/10 | 9/10 |
| 8 | 9/10 | 10/10 | 8/10 | 9/10 |

## Divergence Limitation

The existing canonical QCN llama-witness artifacts are first-token artifacts.
Every prompt has `ref_tokens_count=1`, `our_tokens_count=1`, and
`token_table_len=1`.

Therefore:

- average match run is `1.0` for every variant
- worst match run is `1` for every variant
- divergence position is `null` for every prompt
- exact sequence match is `8/8` for every variant, but only for a one-token
  sequence

Attempted follow-up:

- command intent: capture a 6-case, 32-token llama-witness artifact with
  `KRASIS_WITNESS_MAX_NEW_TOKENS=32`
- built command used: `./dev witness-capture qcn --reference ... --profile phase2bm_qcn_accuracy_32tok --max-cases 6`
- result: the tool still wrote first-token schema cases containing
  `greedy_first_token` and top-k logits only, not generated token sequences
- action: stopped the redundant 32-token matrix and deleted the misleading
  generated reference artifact so it cannot be mistaken for a multi-token
  witness reference

Conclusion: Phase 2BM gives a clean first-token/top-k accuracy matrix. It does
not yet answer multi-token average/worst divergence. To get that, llama-witness
capture needs a real multi-token artifact mode.

## Artifacts

- AWQ run: `logs/reference-test_20260426_235149/reference_test_summary.json`
- HQQ4 run: `logs/reference-test_20260426_235716/reference_test_summary.json`
- HQQ4SC run: `logs/reference-test_20260427_000053/reference_test_summary.json`
- HQQ8 run: `logs/reference-test_20260427_000435/reference_test_summary.json`
- attempted multi-token witness capture log:
  `logs/witness-capture_20260427_001044/witness_capture.log`

