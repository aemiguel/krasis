# Phase 2CN - QCN graph replay/state verification

Date: 2026-04-28

## Question

Verify the remaining QCN accuracy suspects: recurrent state after prefill versus decoding the same prefix, convolution state handoff, gated-delta update order, beta/gate/norm/epsilon behavior, and the first layer/step where state drift starts.

## Source-level checks

Gated-delta order matches llama's non-KDA Qwen3Next path: decay state, compute current `S*k`, compute `(v - S*k) * beta`, update state, then read with `q`.

Beta/gate math is algebraically equivalent:

- llama uses `beta = sigmoid(b)`.
- llama GGUF conversion stores `ssm_a = -exp(A_log)`, then graph computes `softplus(a + dt_bias) * ssm_a`; the CUDA gated-delta kernel exponentiates that gate internally.
- Krasis computes `exp(-exp(A_log) * softplus(a + dt_bias))` before the recurrent update.

One real epsilon mismatch remains:

- llama uses the model RMS norm epsilon for Q/K L2 normalization.
- Krasis fused LA decode currently uses `1e-12` for Q/K L2 normalization; Python fallback uses `1e-6`.
- This should be tested after graph replay parity is fixed, but it is not the leading explanation for the large exact-prefix gap.

## Trace instrumentation

Added trace-gated LA state summaries in `src/gpu_decode.rs`:

- `KRASIS_TRACE_COMPONENTS=la_state`
- `KRASIS_TRACE_LA_STATE_FULL=1`
- state events at `decode_start`, `step_start`, and `step_end`
- summaries include full-state hash, L2, mean, min, max, and sample values for recurrent and convolution state

Build gate: `./dev build` passed.

## Focused graph/no-graph comparison

Prompt: `What's your name?`

Runs:

- graph replay enabled: `logs/reference-test_20260428_162225`
- graph replay disabled with `KRASIS_NO_GRAPH=1`: `logs/reference-test_20260428_162648`
- teacher-forced BF16 prefix: `logs/reference-test_20260428_161447`

| Run | Exact prefix |
| --- | ---: |
| Graph replay enabled | `4/34` |
| No graph replay | `34/34` |
| Teacher-forced BF16 prefix | `1/1` |

Graph and no-graph share the same generated token history through decode steps 0-3:

- step 0: token `40`
- step 1: token `2776`
- step 2: token `1207`
- step 3: token `16948`

At step 4, graph replay selects token `0`; no-graph selects token `1959`, matching the BF16 trajectory.

First meaningful state drift:

- after replay step 1
- layer 1
- convolution state
- graph L2 `197.490157`
- no-graph L2 `199.148852`

By step 4 start at position 39:

- layer 1 conv state graph L2 `199.329723`
- layer 1 conv state no-graph L2 `203.883364`
- token selection diverges at the same step

Teacher-forced versus no-graph at the same BF16 prefix is not bit-identical by hash, but state statistics are close and both paths choose the correct continuation. Example at layer 0:

- teacher conv L2 `417.771311`, no-graph conv L2 `417.761082`
- teacher recurrent L2 `20.897376`, no-graph recurrent L2 `20.896675`

Interpretation: broad prefill/decode handoff is close enough to produce the correct continuation in the no-graph path. The sharp accuracy loss appears when graph replay takes over.

## Full 14-case no-graph control

Command:

```bash
KRASIS_NO_GRAPH=1 ./dev witness-compare tests/qcn-bf16kv-a16-accuracy.conf --profile phase2bn_qcn_64tok --startup-timeout 1200
```

Run dir:

`logs/reference-test_20260428_163327`

| Run | Avg exact prefix | Total exact | Full matches | Containment |
| --- | ---: | ---: | ---: | ---: |
| BF16 attention + BF16 KV, graph replay enabled | `18.79` | `263/653` | `5/14` | `338/653` |
| BF16 attention + BF16 KV, `KRASIS_NO_GRAPH=1` | `37.43` | `524/653` | `9/14` | `566/653` |
| llama Q8_0 default | `37.50` | `525/653` | `11/14` | n/a |

Per-prompt exact-prefix deltas:

| Prompt | Graph | No graph |
| --- | ---: | ---: |
| Hi | `12/12` | `12/12` |
| What's your name? | `4/34` | `34/34` |
| Who trained you? | `17/64` | `15/64` |
| What is 2+2? | `9/9` | `9/9` |
| Now multiply that by 10 | `11/11` | `11/11` |
| And divide the result by 5 | `11/11` | `11/11` |
| Largest animal | `1/64` | `55/64` |
| Largest body of water | `27/64` | `40/64` |
| Binary chop | `29/64` | `64/64` |
| Towels | `6/64` | `24/64` |
| Blue whale facts | `46/64` | `57/64` |
| Whales in general | `11/64` | `64/64` |
| Whale geography | `15/64` | `64/64` |
| Quicksort Rust | `64/64` | `64/64` |

## Conclusion

The large gap to llama Q8 is mostly explained by CUDA graph replay parity, not BF16 attention/KV, expert group size, LM head precision, shared expert precision, or the simple LA FP32 output-projection boundary.

The no-graph Krasis run reaches `37.43` average exact prefix, essentially equal to llama Q8_0 default at `37.50`. The focused state trace shows graph replay diverging from the normal decode path after replay step 1 around layer 1 before the observed token divergence.

Next fix target:

- compare `gpu_decode_step` versus `replay_per_layer_graphs` / segmented graph replay for QCN linear-attention layers
- make graph replay numerically/path equivalent to the normal no-graph decode path
- after parity is fixed, re-test the Q/K L2 epsilon mismatch as a smaller remaining correctness item
