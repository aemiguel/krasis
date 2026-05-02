# Phase 2FW - Q122B HQQ6+k4v4 Exact-Surface Accuracy

Date: 2026-05-02

## Question

The 122B HQQ6/k4v4 benchmark surface had poor decode speed and poor HCS hit
quality. The open question was whether the runtime itself was producing bad
decode output. If so, bad HCS route prediction is a symptom rather than the
root problem.

## Command

```bash
./dev witness-compare tests/q122b-k4v4-hqq6-int4-benchmark.conf \
  --profile llama_witness_q122b_seq32 --startup-timeout 1800
```

Run artifacts:

- Wrapper log: `logs/manual/phase2fw_q122b_hqq6_k4v4_witness_seq32_20260502.log`
- Run dir: `logs/reference-test_20260502_205305`
- Summary: `logs/reference-test_20260502_205305/reference_test_summary.json`
- Reference: `llama_witness_q122b_seq32.json`

## Result

The harness final label was `REFERENCE TEST: PASS`, but that pass label is too
weak for seq32 decode correctness. The detailed metrics show a real decode
failure:

| Metric | Result |
| --- | ---: |
| First-token match | `14/14` |
| Prefill argmax match | `14/14` |
| Prefill top-10 containment | `14/14` |
| Exact generated prefix | `28/361` |
| Full exact cases | `0/14` |
| Decode top-k containment | `28/361` (`7.8%`) |
| Average match run | `2.0` tokens |

Every prompt matched the first token, then diverged after a two-token generated
prefix. Most rows had only about `6%` decode containment after that.

## Controls

Existing 122B seq32 controls on the same llama-witness artifact:

| Surface | Exact | Full | Containment | First token |
| --- | ---: | ---: | ---: | ---: |
| BF16 attention + BF16 KV | `264/361` | `8/14` | `287/361` (`79.5%`) | `13/14` |
| HQQ6 attention + BF16 KV | `283/361` | `8/14` | `305/361` (`84.5%`) | `14/14` |
| HQQ6 attention + k4v4 KV | `28/361` | `0/14` | `28/361` (`7.8%`) | `14/14` |

This rules out HQQ6 attention alone as the explanation. HQQ6 with BF16 KV is
substantially witness-aligned. The failure appears when k4v4 KV is combined
with the HQQ6 benchmark surface.

## Interpretation

The current 122B HQQ6/k4v4 runtime is not decode-correct beyond the initial
token/prefix. That is sufficient to explain why HCS route prediction looks bad:
the decode stream is already off the witness distribution, so expert routes
after the first tokens are not meaningful for speed diagnosis.

Speed/HCS work should pause until this correctness issue is isolated.

## Next Checks

1. Run `tests/q122b-k4v4-a16-accuracy.conf` against
   `llama_witness_q122b_seq32` to determine whether k4v4 alone fails long
   decode when attention is BF16.
2. If BF16-attention+k4v4 passes, trace the HQQ6+k4v4 decode state after token
   1, especially KV descriptors, stage-exact KV export/import, and HQQ decode
   pointer refresh.
3. Tighten the reference-test pass criteria so a run with `7.8%` decode
   containment cannot be reported as PASS.
