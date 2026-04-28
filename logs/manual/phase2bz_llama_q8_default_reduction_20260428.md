# Phase 2BZ llama Q8 default witness reduction

- BF16 artifact: `/home/main/Documents/Claude/krasis-internal/reference-outputs/output/Qwen3-Coder-Next/phase2bn_qcn_64tok.json`
- Q8 artifact: `/home/main/Documents/Claude/krasis-internal/reference-outputs/output/Qwen3-Coder-Next/phase2bz_qcn_llama_q8_64tok.json`
- Cases: `14`
- Average exact prefix: `37.50`
- Total exact prefix: `525/653`
- First-token matches: `14/14`
- Full matches: `11/14`
- Worst exact prefix: `6`

| case | prompt tokens | BF16 tokens | Q8 tokens | exact prefix | first token | full |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 9 | 12 | 12 | 12 | yes | yes |
| 1 | 35 | 34 | 34 | 34 | yes | yes |
| 2 | 82 | 64 | 64 | 64 | yes | yes |
| 3 | 15 | 9 | 9 | 9 | yes | yes |
| 4 | 40 | 11 | 11 | 11 | yes | yes |
| 5 | 67 | 11 | 11 | 11 | yes | yes |
| 6 | 17 | 64 | 64 | 18 | yes | no |
| 7 | 19 | 64 | 64 | 40 | yes | no |
| 8 | 15 | 64 | 64 | 64 | yes | yes |
| 9 | 46 | 64 | 64 | 6 | yes | no |
| 10 | 15 | 64 | 64 | 64 | yes | yes |
| 11 | 232 | 64 | 64 | 64 | yes | yes |
| 12 | 449 | 64 | 64 | 64 | yes | yes |
| 13 | 16 | 64 | 64 | 64 | yes | yes |
