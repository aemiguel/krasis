# Phase 2CB - QCN HQQ8 BF16 KV Accuracy Control

Date: 2026-04-28

Command:

```bash
KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-twoscale-intercept ./dev witness-compare tests/qcn-bf16kv-hqq8-accuracy.conf --profile phase2bn_qcn_64tok
```

Purpose: isolate KV cache precision by comparing the current best HQQ8
two-scale-intercept attention path with BF16 KV instead of Polar4 KV.

The first BF16-KV run (`phase2ca`) was invalid: decode used BF16 kernels, but
prefill still appended prompt K/V through the FP8 cache append kernel. After
adding BF16 prefill append/concat kernels, the fixed run is
`logs/reference-test_20260428_075446`.

## Result

| Runtime | First-token | Prefill | Avg exact prefix | Total exact | Full matches | Containment | Logprob delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HQQ8 two-scale-intercept + Polar4 KV | 14/14 | 14/14 | 18.14 | 254/653 | 5/14 | 326/653 | 0.082125920769 |
| HQQ8 two-scale-intercept + BF16 KV | 14/14 | 14/14 | 18.21 | 255/653 | 5/14 | 317/653 | 0.190790702423 |

## Interpretation

BF16 KV does not materially improve the HQQ8 two-scale-intercept result on the
14-case QCN 64-token witness surface. The exact-prefix delta is +1 token total,
while containment is lower. This points away from Polar4 KV as the main source
of the gap to llama Q8 default (`37.50` avg exact prefix).
