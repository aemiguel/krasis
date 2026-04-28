# Phase 2BR - QCN Polar4 HQQ4SC/HQQ8 Speed Variants

Date: 2026-04-27

## Goal

Run speed-test-equivalent QCN benchmarks with the AWQ attention choice replaced by
the two new interactive attention options:

- HQQ4SC: `CFG_ATTENTION_QUANT=hqq4` plus the top-4 INT8 exception sidecar.
- HQQ8: `CFG_ATTENTION_QUANT=hqq8`.

These runs keep the standard QCN speed-test surface otherwise unchanged:
single RTX 5090, INT4 GPU/CPU experts, Polar4 KV, INT8 shared/dense/lm-head,
layer group size 2, timing instrumentation off.

## Commands

```bash
./dev benchmark tests/qcn-polar4-hqq4sc.conf
./dev benchmark tests/qcn-polar4-hqq8.conf
```

The wrapper used tmux and `./dev kill` between runs:

```bash
bash logs/manual/phase2br_qcn_hqq_speed_matrix_20260427.sh
```

The first invocation rebuilt the local extension before the benchmark because
source files were newer than the compiled module. That build occurred before
benchmark timing and is not part of the speed numbers below.

## Results

| Variant | KV | Attention | Prefill internal | Decode internal | HCS | Min free VRAM | Full log |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| AWQ speed-test baseline, 2026-04-26 | Polar4 | AWQ | 7,295.6 tok/s | 91.77 tok/s | 16848/24576 (68.6%) | 688 MB | `benchmarks/20260426_211755_qcn_polar4_awq_speed_regression_check.log` |
| HQQ4SC speed-test variant | Polar4 | HQQ4 + top-4 sidecar | 466.2 tok/s | 79.17 tok/s | 14823/24576 (60.3%) | 706 MB | `benchmarks/20260427_201735_qcn_polar4_hqq4sc_speed_variant.log` |
| HQQ8 speed-test variant | Polar4 | HQQ8 | 486.8 tok/s | 75.16 tok/s | 14256/24576 (58.0%) | 734 MB | `benchmarks/20260427_202957_qcn_polar4_hqq8_speed_variant.log` |

Per-run prefill and decode details:

| Variant | 1K | 5K | 10K | 20K | 35K | 50K | 50 decode | 100 decode | 250 decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HQQ4SC | 274.9 | 414.0 | 441.1 | 457.1 | 466.2 | 464.0 | 74.44 | 79.17 | 70.53 |
| HQQ8 | 187.5 | 373.5 | 432.9 | 469.0 | 486.8 | 482.9 | 69.29 | 75.16 | 61.19 |

## Interpretation

The decode result is usable but slower than AWQ:

- HQQ4SC decode is `79.17 tok/s`, about `86%` of the AWQ baseline.
- HQQ8 decode is `75.16 tok/s`, about `82%` of the AWQ baseline.

The prefill result is not acceptable for replacing AWQ on the current Polar4
speed-test surface:

- HQQ4SC prefill is `466.2 tok/s`, about `6.4%` of the AWQ baseline.
- HQQ8 prefill is `486.8 tok/s`, about `6.7%` of the AWQ baseline.

This is not the same as the earlier fast HQQ4 benchmarks, which used FP8 KV.
The Phase 2BK QCN HQQ4 FP8-KV baseline reached `7,242.6 tok/s` prefill and
`70.79 tok/s` decode. The poor result here is specifically the HQQ attention
path on the AWQ speed-test-style Polar4 KV configuration.

Both HQQ4SC and HQQ8 entered low-VRAM territory during timed prefill:

- HQQ4SC: VRAM monitor lows at `576 MB` and `532 MB` free.
- HQQ8: VRAM monitor low at `516 MB` free.

## Follow-up

Before making HQQ8 the speed-test/default production attention path, diagnose
why HQQ + Polar4 prefill is around `15x` slower than AWQ + Polar4 and HQQ +
FP8-KV. The immediate discriminator should be an HQQ8 speed run with FP8 KV on
the same current code, followed by instrumentation of the HQQ + Polar4 prefill
path if the FP8-KV run remains fast.
