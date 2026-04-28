# Phase 2CD-CH QCN Accuracy Attribution

Date: 2026-04-28

Reference surface: `phase2bn_qcn_64tok`, 14 QCN prompts, thinking disabled.

## Runtime Controls

| Run | First-token | Prefill | Avg exact prefix | Total exact | Full matches | Containment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HQQ8 two-scale-intercept + Polar4 KV | `14/14` | `14/14` | `18.14` | `254/653` | `5/14` | `326/653` |
| HQQ8 two-scale-intercept + BF16 KV | `14/14` | `14/14` | `18.21` | `255/653` | `5/14` | `317/653` |
| BF16 attention + BF16 KV | `14/14` | `14/14` | `18.79` | `263/653` | `5/14` | `338/653` |
| BF16 attention + BF16 KV + unfused LA decode | `14/14` | `14/14` | `17.86` | `250/653` | `5/14` | `295/653` |
| BF16 attention + BF16 KV + BF16 LM head | `14/14` | `14/14` | `13.93` | `195/653` | `4/14` | `314/653` |
| BF16 attention + BF16 KV + BF16 shared expert | `14/14` | `14/14` | `18.86` | `264/653` | `4/14` | `357/653` |
| llama Q8_0 default | `14/14` | n/a | `37.50` | `525/653` | `11/14` | n/a |

## Teacher-Forced Probe

Derived profile: `phase2cd_qcn_teacher_forced_prefixes`

Input for each case was the original prompt plus the BF16 witness prefix up to the prior Krasis BF16-attention/BF16-KV divergence point. Reference was the next BF16 token only.

Result: first-token `13/14`, prefill argmax `12/14`, decode top-k containment `14/14`.

Interpretation: when supplied with the correct BF16 history, Krasis usually scores the next BF16 token correctly. The low free-running exact-prefix result is therefore drift from repeated small deviations, not a simple failure to score the correct next token from the correct context.

## Quantization Error Probe

Probe file: `logs/manual/phase2ch_qcn_int8_quant_error_probe_20260428.json`

Representative tensor reconstruction error compared a coarse Krasis-style symmetric INT8 probe against a llama-style Q8_0 block32 approximation. Later Phase 2CI code inspection clarified that the production Rust expert cache is already grouped, with default group size `128`, not a single scale for a whole row.

| Tensor group | Coarse Krasis-style INT8 MSE vs block32 Q8_0 |
| --- | ---: |
| Routed expert `gate/up` | usually `2.4x-3.7x` higher |
| Routed expert `down` | usually `1.9x-2.6x` higher |
| Shared expert | usually `2.6x-4.7x` higher |
| LM head sample | `3.32x` higher |
| Overall sampled average | `2.68x` higher |

Conclusion: the controls rule out Polar4 KV, HQQ attention, fused LA decode, LM-head staging, and shared-expert precision as the main explanation. The largest remaining difference versus llama Q8 is still likely in the main routed MoE expert path, but Phase 2CI showed the first simple fix, `g32` expert scales, did not improve aggregate exact-prefix accuracy.
