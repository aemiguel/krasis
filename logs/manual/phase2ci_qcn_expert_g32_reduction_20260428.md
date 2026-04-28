# Phase 2CI QCN Expert INT8 g32 Reduction

Date: 2026-04-28

Reference surface: `phase2bn_qcn_64tok`, 14 QCN prompts, thinking disabled.

## Implementation

- Added `CFG_EXPERT_GROUP_SIZE` / `--expert-group-size` with supported values `32`, `64`, `128`.
- Routed the group size through config validation, RAM/VRAM estimates, expert cache selection, Rust engine load, and decode store configuration.
- Fixed an existing silent fallback where the Marlin expert loader could reuse a cache with a different group size.

## Invalid First Attempt

Run: `logs/reference-test_20260428_095734`

Result: first-token `0/14`, prefill top-10 `0/14`.

Cause: the loader accepted existing `experts_marlin_int8_g128.bin` while the runtime was configured for `g32`, corrupting the expert scale layout. This result is invalid and should not be used for quality comparisons.

## Valid g32 Run

Run: `logs/reference-test_20260428_100614`

Cache built: `/home/main/.krasis/cache/Qwen3-Coder-Next/experts_marlin_int8_g32.bin` (`82,301,681,728` bytes).

| Run | First-token | Prefill | Avg exact prefix | Total exact | Full matches | Containment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HQQ8 + Polar4 KV + g128 experts | `14/14` | `14/14` | `18.14` | `254/653` | `5/14` | `326/653` |
| BF16 attention + BF16 KV + g128 experts | `14/14` | `14/14` | `18.79` | `263/653` | `5/14` | `338/653` |
| BF16 attention + BF16 KV + BF16 shared expert | `14/14` | `14/14` | `18.86` | `264/653` | `4/14` | `357/653` |
| BF16 attention + BF16 KV + g32 experts | `14/14` | `14/14` | `18.29` | `256/653` | `5/14` | `326/653` |

## Per-Prompt Delta

Compared to BF16 attention + BF16 KV + g128 experts:

| Prompt | Exact prefix | Containment |
| --- | ---: | ---: |
| Hi | `12 -> 12` | `12 -> 12` |
| What's your name? | `4 -> 14` | `7 -> 15` |
| Who trained you? | `17 -> 17` | `21 -> 21` |
| What is 2+2? | `9 -> 9` | `9 -> 9` |
| Now multiply that by 10 | `11 -> 11` | `11 -> 11` |
| And divide the result by 5 | `11 -> 11` | `11 -> 11` |
| Largest animal | `1 -> 22` | `9 -> 28` |
| Largest body of water | `27 -> 17` | `32 -> 29` |
| Binary chop | `29 -> 29` | `59 -> 33` |
| Towel reasoning | `6 -> 6` | `12 -> 15` |
| Blue whale facts | `46 -> 26` | `50 -> 34` |
| Whales general | `11 -> 11` | `19 -> 29` |
| Whales geography | `15 -> 7` | `22 -> 15` |
| Quicksort Rust | `64 -> 64` | `64 -> 64` |

## Decision

Do not run speed. The exact-cache fix is correct and g32 is now a valid selectable expert format, but g32 does not improve the aggregate accuracy gate. Smaller scale groups alone are not the missing llama Q8 behavior.
