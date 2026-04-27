# Phase 2BO: HQQ8 Decode Suspicion Triage

Date: 2026-04-27

## Question

Phase 2BN showed QCN HQQ8 averaging only `7.43` exact-prefix tokens over the
64-token llama-witness matrix despite using INT8 experts. This looked too weak
for a clean HQQ8 attention mode.

## Diagnostic 1: BF16 attention control

Ran a BF16 attention control with the same non-attention settings as Phase 2BM:

- Qwen3-Coder-Next
- Polar4 KV
- GPU/CPU routed experts INT8
- shared/dense/lm-head INT8
- thinking off
- profile `phase2bn_qcn_64tok`

Config:

- `logs/manual/phase2bo_qcn_accuracy_bf16_control_20260427.conf`

Run:

- `logs/reference-test_20260427_075501`

Result:

| Variant | First-token | Selected sum | Top-10 overlap | Avg exact prefix | Worst prefix | Full matches | Weighted decode top-k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AWQ | `14/14` | `0.372141474328` | `123/140` | `14.57` | `4` | `4/14` | `40.7%` |
| BF16 | `14/14` | `0.132351834630` | `135/140` | `17.14` | `4` | `5/14` | `49.9%` |
| HQQ8 | `14/14` | `0.129170737779` | `136/140` | `7.43` | `1` | `3/14` | `27.9%` |

HCS residency:

- AWQ: `8528/24576`
- BF16: `7831/24576`
- HQQ8: `7216/24576`

BF16 has lower HCS residency than AWQ but still gets the best exact-prefix
result in this run. That weakens the HCS-residency explanation for HQQ8's poor
prefix stability.

## Diagnostic 2: HQQ8 teacher-forced full-prefix prefill

Started HQQ8 with test endpoints and probed `/v1/internal/prefill_logits` using
reference-token prefixes for prompts where runtime HQQ8 decode diverged early.
This recomputes the whole prefix through HQQ8 quantized prefill rather than
using token-time decode state.

Server config:

- `logs/manual/phase2bm_qcn_accuracy_hqq8_20260426.conf`

Probe prompts:

- prompt index 7: runtime HQQ8 diverged at token `1`
- prompt index 10: runtime HQQ8 diverged at token `2`
- prompt index 1: runtime HQQ8 diverged at token `7`

Teacher-forced HQQ8 prefill result:

| Prompt index | Runtime HQQ8 exact prefix | Teacher-forced prefill top-1 over first 16 tokens |
| ---: | ---: | ---: |
| 7 | `1` | `16/16` |
| 10 | `2` | `16/16` |
| 1 | `7` | `16/16` |

The HQQ8 artifact/prefill path can reproduce the witness continuation when the
full prefix is recomputed. Runtime decode diverges much earlier on the same
prompts.

## Conclusion

Treat this as an active HQQ decode/recurrent-state correctness bug until proven
otherwise.

Current evidence points away from:

- HQQ8 artifact quality: cache mean absolute weight error is about
  `0.000119`
- HQQ8 quantized prefill: teacher-forced full-prefix prefill matched top-1
  across checked early positions
- HCS residency as the sole explanation: BF16 has lower HCS than AWQ but better
  exact-prefix stability

Current likely failure region:

- HQQ8 token-time decode projections
- linear-attention recurrent/conv state update under HQQ decode
- accumulated decode-state drift from HQQ projection error

Next debugging step:

- add or use a forced-decode diagnostic path that feeds reference tokens through
  the token-time decode path and records logits/state by step, then compare
  HQQ8 decode against BF16 decode and HQQ8 full-prefix prefill.
