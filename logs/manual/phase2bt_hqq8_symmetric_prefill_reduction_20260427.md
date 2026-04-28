# Phase 2BT - QCN HQQ8 Symmetric Marlin Prefill Accuracy

Date: 2026-04-27

## Goal

Test the fastest simple HQQ8 prefill strategy:

- dequantize HQQ8 attention weights on the host
- requantize them to native symmetric INT8 Marlin format
- run prefill as one standard Marlin INT8 GEMM
- avoid residual-scale Marlin pass and HQQ zero-correction kernel

Experiment switch:

```bash
KRASIS_HQQ8_PREFILL_MODE=symmetric-marlin
```

## Command

```bash
KRASIS_HQQ8_PREFILL_MODE=symmetric-marlin ./dev witness-compare logs/manual/phase2bm_qcn_accuracy_hqq8_20260426.conf --profile phase2bn_qcn_64tok
```

Run directory:

```text
logs/reference-test_20260427_223238
```

## Result

The symmetric Marlin path passed the coarse gate but did not preserve the
previous HQQ8 exact-prefix quality.

| Variant | First-token | Prefill argmax | Avg exact prefix | Worst | Full matches | Decode top-k | Selected logprob abs-sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| residual Marlin HQQ8 | 14/14 | 14/14 | 15.79 | 1 | 4/14 | 285/653 (43.6%) | 0.354103 |
| symmetric Marlin HQQ8 | 14/14 | 14/14 | 11.71 | 1 | 4/14 | 230/653 (35.2%) | 0.163842 |

Per-prompt exact-prefix runs:

```text
residual:  [12, 14, 15, 9, 11, 11, 1, 34, 29, 6, 46, 11, 15, 7]
symmetric: [12, 4, 24, 9, 11, 11, 1, 17, 29, 6, 10, 8, 15, 7]
```

HCS was the same for both residual and symmetric runs in this INT8 witness
config:

```text
7216/24576 experts (29.4%)
```

## Interpretation

Accuracy did not hold strongly enough to make the symmetric conversion the
production HQQ8 default.

The good news:

- first-token stayed 14/14
- prefill argmax stayed 14/14
- selected first-token logprob abs-sum improved versus the residual path

The problem:

- recurrent exact-prefix average dropped from 15.79 to 11.71
- decode top-k containment dropped from 43.6% to 35.2%
- several prompts diverged much earlier despite first-token agreement

Conclusion: HQQ8-to-native-symmetric-Marlin is a useful speed-ceiling idea, but
this simple host requantization is not quality-equivalent to HQQ8. The next
production target remains a native fused HQQ8 Marlin kernel that preserves HQQ
scale/zero math inside the tiled accumulator, or a calibrated symmetric
conversion/self-correction profile if we want to pursue the symmetric route.
