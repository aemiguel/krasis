# Phase 2CR QCN HQQ8 k8v4 Reduction - 2026-04-28

## Summary

| Run | Avg exact | Total exact | Full matches | First token | Containment |
| --- | ---: | ---: | ---: | ---: | ---: |
| HQQ8 + BF16 KV | `36.50` | `511/653` | `10/14` | `14/14` | `554/653` |
| HQQ8 + Polar4 | `30.50` | `427/653` | `8/14` | `14/14` | `459/653` |
| Polar4 norm-correct K+V | `30.50` | `427/653` | `7/14` | `14/14` | `496/653` |
| HQQ8 + k8v4 | `32.14` | `450/653` | `8/14` | `14/14` | `504/653` |

## k8v4 Prompt Deltas

| Case | Prompt | BF16 | Polar4 | k8v4 | k8v4 - Polar4 | Gap to BF16 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Hi | `12` | `12` | `12` | `+0` | `0` |
| 2 | What's your name? | `34` | `34` | `34` | `+0` | `0` |
| 3 | Who trained you? | `17` | `2` | `15` | `+13` | `2` |
| 4 | What is 2+2? | `9` | `9` | `9` | `+0` | `0` |
| 5 | Now multiply that by 10 | `11` | `11` | `11` | `+0` | `0` |
| 6 | And divide the result by 5 | `11` | `11` | `11` | `+0` | `0` |
| 7 | What is the largest animal in the world? | `64` | `64` | `55` | `-9` | `9` |
| 8 | What is the largest body of water in the world? | `40` | `55` | `64` | `+9` | `-24` |
| 9 | Describe the binary chop algorithm in depth | `64` | `47` | `29` | `-18` | `35` |
| 10 | If it takes 4 hours for 4 towels to dry on a clothesline in  | `24` | `13` | `24` | `+11` | `0` |
| 11 | Tell me facts about the blue whale | `64` | `64` | `64` | `+0` | `0` |
| 12 | Tell me more about whales in general | `64` | `8` | `64` | `+56` | `0` |
| 13 | Where do whales live geographically? | `33` | `33` | `33` | `+0` | `0` |
| 14 | Write me a quicksort implementation in Rust | `64` | `64` | `25` | `-39` | `39` |

## Conclusion

k8v4 is a valid mixed KV path and improves aggregate exact-prefix over current Polar4, but it does not recover BF16 KV accuracy. It narrows the total exact gap from Polar4 `427/653` to k8v4 `450/653`, while BF16 KV remains `511/653` on the same HQQ8/INT8-expert graph-enabled surface.

Because k8v4 remains materially below BF16 KV and has prompt-specific regressions versus Polar4, it is not yet a clear accuracy win to promote as the next default.
