# Phase 2CC - QCN BF16 KV + BF16 Attention Accuracy Control

Date: 2026-04-28

Command:

```bash
./dev witness-compare tests/qcn-bf16kv-a16-accuracy.conf --profile phase2bn_qcn_64tok
```

Purpose: test whether restoring BF16 attention weights, after fixing BF16 KV,
substantially closes the gap to llama Q8 default.

Run directory: `logs/reference-test_20260428_080508`

## Result

| Runtime | First-token | Prefill | Avg exact prefix | Total exact | Full matches | Containment | Logprob delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HQQ8 two-scale-intercept + Polar4 KV | 14/14 | 14/14 | 18.14 | 254/653 | 5/14 | 326/653 | 0.082125920769 |
| HQQ8 two-scale-intercept + BF16 KV | 14/14 | 14/14 | 18.21 | 255/653 | 5/14 | 317/653 | 0.190790702423 |
| BF16 attention + BF16 KV | 14/14 | 14/14 | 18.79 | 263/653 | 5/14 | 338/653 | 0.067247679384 |
| llama Q8_0 default | 14/14 | n/a | 37.50 | 525/653 | 11/14 | n/a | n/a |

## Interpretation

BF16 attention plus BF16 KV improves Krasis only modestly over HQQ8
two-scale-intercept. The gap to llama Q8 default remains large. On this surface,
the main reduction is therefore unlikely to be explained by Polar4 KV or HQQ8
attention projection quantization alone.

The remaining likely contributors are elsewhere in the runtime comparison:
linear/recurrent attention implementation differences, recurrent state/update
precision or ordering, expert/decode path differences, LM-head or sampling/logit
path differences, or witness/runtime architectural differences.
