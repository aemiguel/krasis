# Phase 2BX - HQQ8 Native Fused Two-Scale

Command:

```bash
KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin-twoscale ./dev witness-compare logs/manual/phase2bm_qcn_accuracy_hqq8_20260426.conf --profile phase2bn_qcn_64tok
```

Run directory:

```text
logs/reference-test_20260428_004043
```

## Result

Two-scale mode passed first-token and prefill checks, but did not improve the
main recurrent exact-prefix gate.

| Variant | Avg exact prefix | Decode containment | Selected-logprob delta |
| --- | ---: | ---: | ---: |
| residual Marlin HQQ8 | 15.79 | 285/653 | 0.354102521143 |
| native fused v1 | 14.29 | 262/653 | 0.081758249973 |
| native fused v2 | 11.21 | 265/650 | 0.117291019720 |
| native fused two-scale | 13.57 | 294/653 | 0.121019028340 |

Two-scale improves decode containment over both residual and native v1, but
exact-prefix stability is worse than native v1 and residual. No speed benchmark
was run.

## Interpretation

The two-scale path fixes the v2 slope omission by applying a BF16 residual scale
plane inside the same Marlin kernel accumulator:

```text
(q - zero_bf16) * (scale_hi_bf16 + scale_lo_bf16)
```

This is closer than v2, but it still uses BF16 zero points. The remaining
accuracy gap is likely not just slope precision; exact recurrent trajectory
still depends on the zero term and/or on numerical ordering inside the fused
kernel.

Decision: keep this mode as an explicit experiment only. The quality-favored
default remains residual Marlin HQQ8.
