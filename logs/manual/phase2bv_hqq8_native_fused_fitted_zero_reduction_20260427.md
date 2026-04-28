# Phase 2BV - HQQ8 Native Fused Fitted-Zero Test

Date: 2026-04-27

## Question

Would fitting the native fused BF16 zero-point metadata against the BF16-rounded
scale improve HQQ8 native fused accuracy without adding runtime correction
passes?

## Change Tested

Temporarily changed `marlin_repack_hqq8_native_zp_prefill` so the staged zero
metadata used:

```text
zero_bf16 ~= zero_f32 * scale_f32 / scale_bf16_effective
```

The intended runtime effect was to keep one Marlin U8 + float-zero-point GEMM
while preserving the original HQQ intercept `zero_f32 * scale_f32` more closely.

## Validation

Build:

```text
./dev build
```

Witness gate:

```text
KRASIS_HQQ8_PREFILL_MODE=native-fused-marlin ./dev witness-compare logs/manual/phase2bm_qcn_accuracy_hqq8_20260426.conf --profile phase2bn_qcn_64tok
```

Run directory:

```text
logs/reference-test_20260427_232502
```

## Results

| Variant | First-token | Prefill argmax | Avg exact prefix | Decode containment | Selected-logprob delta sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual Marlin HQQ8 | 14/14 | 14/14 | 15.79 | 285/653 | 0.354102521143 |
| native fused HQQ8 before fit | 14/14 | 14/14 | 14.29 | 262/653 | 0.081758249973 |
| native fused HQQ8 fitted-zero | 14/14 | 14/14 | 9.57 | 186/653 | 0.049697407994 |

The fitted-zero variant improved the first-token selected-logprob delta but
substantially regressed recurrent decode quality.

## Decision

Do not keep the fitted-zero behavior. No speed benchmark was run because the
accuracy gate did not improve. The fitted-zero code was reverted and the local
extension was rebuilt with `./dev build` to restore the previous native fused
behavior.
