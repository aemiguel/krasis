# Phase 2CJ - QCN Quantization vs llama Q8 Comparison

Date: 2026-04-28 09:52 UTC

## Question

After the valid `g32` expert run failed to improve the aggregate witness gate, compare Krasis expert quantization against llama Q8 behavior and identify what still looks like the accuracy reducer.

## Existing Witness Context

| Run | Avg exact prefix | Total exact | Full matches | Containment |
| --- | ---: | ---: | ---: | ---: |
| HQQ8 + Polar4 KV + g128 | 18.14 | 254/653 | 5/14 | 326/653 |
| BF16 attention + BF16 KV + g128 | 18.79 | 263/653 | 5/14 | 338/653 |
| BF16 attention + BF16 KV + g32 | 18.29 | 256/653 | 5/14 | 326/653 |
| llama Q8_0 default | 37.50 | 525/653 | 11/14 | n/a |

## Corrected Expert Quantization Finding

The earlier working theory that routed experts were "row-wise INT8" was wrong. Production expert INT8 is grouped, with default group size `128`. Phase 2CI correctly exposed `CFG_EXPERT_GROUP_SIZE` and fixed a real wrong-group cache fallback, but the valid `g32` run did not improve aggregate accuracy.

Offline checks now show smaller expert groups reduce static reconstruction error, but that does not translate to witness accuracy:

| Variant | Avg sampled expert tensor MSE | Ratio vs g128 BF16 scale |
| --- | ---: | ---: |
| Krasis g128, BF16 scales | 1.29598e-08 | 1.000 |
| Krasis g128, FP16 scales | 1.29155e-08 | 0.997 |
| Krasis g32, BF16 scales | 8.92507e-09 | 0.689 |
| Krasis g32, FP16 scales | 8.78932e-09 | 0.678 |

BF16 scale storage is not the main issue: it is within about 1-2% of FP16/FP32 scale storage in these probes.

## Expert Output Simulation

A standalone routed-expert simulation on sampled QCN experts compared BF16 expert output to Krasis-style grouped INT8 and a llama Q8_0-style block32 approximation.

| Variant | Relative RMSE | Cosine |
| --- | ---: | ---: |
| Krasis g128 BF16 scales | 0.01092 | 0.9999405 |
| Krasis g32 BF16 scales | 0.009005 | 0.9999596 |
| Krasis g32 FP16 scales | 0.008964 | 0.9999599 |
| llama Q8_0-style block32 weight+activation | 0.01418 | 0.9999001 |

This does not prove the production expert runtime is perfect, but it does rule out the simple claim that Krasis expert quantization is inherently much worse than llama Q8_0. In this isolated expert calculation, Krasis-style INT8 is closer to BF16 than the Q8_0-style approximation.

The llama Q8_0 quantize log confirms llama did quantize routed expert tensors such as `ffn_down_exps`, `ffn_gate_exps`, and `ffn_up_exps` from BF16 to Q8_0. llama's better witness score is therefore not because routed experts stayed BF16.

## Runtime Difference That Still Matters

The strongest source-level difference is QCN linear/recurrent attention precision through the gated-delta path.

llama Qwen3Next:

- `ggml_gated_delta_net` requires Q/K/V/g/beta/state tensors to be `GGML_TYPE_F32`.
- It returns a `GGML_TYPE_F32` tensor.
- Qwen3Next then applies gated norm and feeds that `final_output` directly into `ssm_out`.

Krasis:

- `attention_quant=bf16` does cover QCN linear-attention projection weights; they register through the BF16 path, not Marlin.
- LA decode uses F32 projection outputs, F32 conv/recurrent state, and F32 gate/beta.
- The fused LA recurrent/norm kernel writes BF16 into `d_scratch`.
- `out_proj` then consumes that BF16 `d_scratch`.
- The Python reference path also explicitly casts the gated recurrent output to BF16 before `out_proj`.

That means the prior BF16-attention/BF16-KV control did not make the whole recurrent attention path llama-like. It still kept Krasis' BF16 boundary between gated-delta output and the LA output projection. This is a plausible recurrent drift source because QCN has 36 linear-attention layers and exact-prefix loss is accumulated over free-running decode.

## Current Conclusion

The valid comparison no longer supports "fix expert group size" as the next main fix. The remaining gap is more likely in QCN linear/recurrent attention runtime precision/order, especially the BF16 boundary before LA `out_proj`, or in another hidden-state precision boundary common to all Krasis controls.

## Next Measurement

Add an explicit debug/accuracy mode for LA decode that keeps the gated-delta/gated-norm output in FP32 through the LA output projection, matching llama's graph more closely. Then rerun the same `phase2bn_qcn_64tok` witness on the BF16-attention/BF16-KV config.

Do not run speed until the witness gate improves.
