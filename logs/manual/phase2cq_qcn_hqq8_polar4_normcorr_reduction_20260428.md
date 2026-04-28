# Phase 2CQ QCN HQQ8 Polar4 Norm-Correction Controls - 2026-04-28

## Summary

| Run | Avg exact | Total exact | Full matches | Containment |
| --- | ---: | ---: | ---: | ---: |
| HQQ8 + BF16 KV | `36.50` | `511/653` | `10/14` | `554/653` |
| Polar4 baseline | `30.50` | `427/653` | `8/14` | `459/653` |
| Polar4 norm-correct K+V | `30.50` | `427/653` | `7/14` | `496/653` |
| Polar4 norm-correct K only | `29.36` | `411/653` | `7/14` | `494/653` |
| Polar4 norm-correct V only | `24.36` | `341/653` | `6/14` | `405/653` |

## Per Prompt Exact Prefix

| # | Prompt | BF16 | Base | K+V NC | K NC | V NC |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Hi | `12` | `12` | `12` | `12` | `12` |
| 2 | What's your name? | `34` | `34` | `34` | `34` | `34` |
| 3 | Who trained you? | `17` | `2` | `17` | `2` | `24` |
| 4 | What is 2+2? | `9` | `9` | `9` | `9` | `9` |
| 5 | Now multiply that by 10 | `11` | `11` | `11` | `11` | `11` |
| 6 | And divide the result by 5 | `11` | `11` | `11` | `11` | `11` |
| 7 | What is the largest animal in the world? | `64` | `64` | `56` | `55` | `22` |
| 8 | What is the largest body of water in the world? | `40` | `55` | `17` | `17` | `17` |
| 9 | Describe the binary chop algorithm in depth | `64` | `47` | `50` | `50` | `44` |
| 10 | If it takes 4 hours for 4 towels to dry on a clothesline in the sun, how long do | `24` | `13` | `24` | `24` | `24` |
| 11 | Tell me facts about the blue whale | `64` | `64` | `64` | `64` | `64` |
| 12 | Tell me more about whales in general | `64` | `8` | `64` | `64` | `11` |
| 13 | Where do whales live geographically? | `33` | `33` | `33` | `33` | `33` |
| 14 | Write me a quicksort implementation in Rust | `64` | `64` | `25` | `25` | `25` |

## Conclusion

- Norm correction is not an aggregate exact-prefix win on this surface.
- K+V norm correction preserves the baseline exact total (`427/653`) but improves containment (`459/653 -> 496/653`) and changes which prompts fail.
- K-only and V-only are worse than baseline on exact prefix (`411/653` and `341/653` respectively).
- The next useful Polar4 control is a separate K/V quantizer or mixed K precision path, not norm correction alone.
