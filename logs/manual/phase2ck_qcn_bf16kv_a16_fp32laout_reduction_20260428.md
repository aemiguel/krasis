# Phase 2CK - QCN FP32 Linear-Attention Out-Projection Control

Date: 2026-04-28

## Question

Phase 2CJ found that expert group size and expert scale precision did not explain the remaining gap to llama Q8. The next candidate was the QCN linear/recurrent attention precision boundary: llama keeps gated-delta output/state as F32 through `ssm_out`, while Krasis narrowed the LA recurrent output to BF16 before `out_proj`.

This run tested whether keeping that LA output as FP32 through `out_proj` improves free-running exact-prefix accuracy.

## Implementation

- Added disabled debug mode: `KRASIS_LA_FP32_OUT_PROJ=1`
- Added CUDA kernel `la_fused_post_proj_f32`
- Added CUDA kernel `gemv_bf16_weight_f32_input_bf16`
- Added config: `tests/qcn-bf16kv-a16-fp32laout-accuracy.conf`

The first run failed before witness rows because cuBLAS rejected the BF16-weight/F32-input/BF16-output GEMV shape. The valid rerun used the custom debug GEMV instead of falling back to BF16 input.

## Result

| Run | Avg exact prefix | Total exact | Full matches | Containment |
| --- | ---: | ---: | ---: | ---: |
| BF16 attention + BF16 KV baseline | `18.79` | `263/653` | `5/14` | `338/653` |
| FP32 LA out-proj debug mode | `12.21` | `171/653` | `4/14` | `293/647` |
| llama Q8_0 default | `37.50` | `525/653` | `11/14` | n/a |

The debug mode still passed first-token and prefill checks:

- first-token: `14/14`
- prefill argmax: `14/14`
- reference-test result: `PASS`

## Per-Prompt Exact Prefix

| Prompt | Baseline | FP32 LA out-proj |
| --- | ---: | ---: |
| Hi | `12` | `12` |
| What's your name? | `4` | `4` |
| Who trained you? | `17` | `17` |
| 2+2 | `9` | `9` |
| 17 * 23 | `11` | `11` |
| 100 / 4 | `11` | `11` |
| Largest animal | `1` | `22` |
| Largest body of water | `27` | `17` |
| Binary search explanation | `29` | `9` |
| Towel story | `6` | `6` |
| Blue whale | `46` | `10` |
| Whales general | `11` | `11` |
| Whales geography | `15` | `7` |
| Quicksort | `64` | `25` |

## Conclusion

Preserving FP32 through the LA output projection is not the missing accuracy fix. It improves one weak prompt but substantially worsens several longer prompts, dropping aggregate exact-prefix accuracy from `18.79` to `12.21`.

No speed benchmark was run because accuracy regressed.

The remaining gap should be investigated as Qwen3Next recurrent-attention math/order/state behavior rather than the simple BF16 narrowing before LA `out_proj`.
