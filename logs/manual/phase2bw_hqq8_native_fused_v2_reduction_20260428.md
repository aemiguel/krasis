# Phase 2BW - HQQ8 Native Fused v2 Intercept Correction

Command:

```bash
KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-v2 ./dev witness-compare logs/manual/phase2bm_qcn_accuracy_hqq8_20260426.conf --profile phase2bn_qcn_64tok
```

Run directory:

`logs/reference-test_20260428_000742`

Implementation tested:

- one native fused Marlin U8 + BF16 float-zero-point GEMM
- appended FP32 intercept-correction plane
- BF16 input group sums plus correction:
  `sum(input_group) * (zero_bf16 * scale_bf16 - zero_f32 * scale_f32)`
- no second Marlin GEMM

Accuracy comparison:

| Variant | First-token | Prefill argmax | Avg exact prefix | Decode containment | Selected-logprob delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual Marlin HQQ8 | 14/14 | 14/14 | 15.79 | 285/653 | 0.354102521143 |
| native fused v1 | 14/14 | 14/14 | 14.29 | 262/653 | 0.081758249973 |
| native fused v2 | 14/14 | 14/14 | 11.21 | 265/650 | 0.117291019720 |

Conclusion:

Native fused v2 did not improve accuracy. The first-token and prefill checks still pass, and containment is roughly comparable to v1, but exact-prefix quality regressed materially. No speed benchmark was run because the accuracy gate did not improve over native fused v1.

Current decision:

- keep `native-fused-marlin-v2` only as an explicit experiment
- keep `native-fused-marlin` as the better native-fused speed experiment
- do not promote v2
