# Phase 2FY - Q122B HQQ8+k6v6 Accuracy Probe

Date: 2026-05-02

## Question

After Q122B HQQ6+k4v4 failed seq32 witness alignment, test whether a higher-quality combined surface, HQQ8 attention plus k6v6 KV, fares the same.

## Config

Created `tests/q122b-k6v6-hqq8-accuracy.conf`:

- Model: Qwen3.5-122B-A10B
- Experts: INT4
- Attention: HQQ8
- KV: k6v6
- KV cache: 1000 MB
- GPUs: 0
- Thinking: enabled, matching the existing Q122B accuracy configs

## Command

```bash
./dev witness-compare tests/q122b-k6v6-hqq8-accuracy.conf \
  --profile llama_witness_q122b_seq32 --startup-timeout 1800
```

Run dir: `logs/reference-test_20260502_215046`
Log: `logs/manual/phase2fy_q122b_hqq8_k6v6_witness_seq32_20260502.log`

## Result

The harness reports `PASS`, but the detailed seq32 metrics are bad.

| Surface | Exact | Containment | Full exact | First token | Avg run |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16/BF16 | 264/361 (73.1%) | 287/361 (79.5%) | 8/14 | 13/14 | 18.9 |
| HQQ6/BF16-KV | 283/361 (78.4%) | 305/361 (84.5%) | 8/14 | 14/14 | 20.2 |
| BF16-attn/k4v4 | 247/361 (68.4%) | 291/361 (80.6%) | 8/14 | 14/14 | 17.6 |
| HQQ6/k4v4 | 28/361 (7.8%) | 28/361 (7.8%) | 0/14 | 14/14 | 2.0 |
| HQQ8/k6v6 | 24/361 (6.6%) | 26/361 (7.2%) | 0/14 | 12/14 | 1.7 |

The per-prompt pattern is the same class of failure as HQQ6+k4v4: prefill/first-token is often close, then decode diverges almost immediately. Most prompts have exact run of 2 tokens and roughly 6% decode top-k containment.

## Interpretation

HQQ8+k6v6 does not fix the combined-surface failure. On Q122B seq32 it is slightly worse than HQQ6+k4v4.

This weakens the earlier narrow hypothesis that the issue is specifically HQQ6 6-bit packing. It now looks more like a broader HQQ attention plus compact quantized KV decode interaction on Q122B, or a recent shared/stage-exact descriptor path issue that appears only when both sides are quantized.

Important caveat: existing Q122B HQQ8-alone and k6v6-alone checks were found only on the short `expanded` artifact, not the seq32 artifact. The next clean isolation should run these two seq32 controls:

```bash
./dev witness-compare tests/q122b-bf16kv-hqq8-accuracy.conf \
  --profile llama_witness_q122b_seq32 --startup-timeout 1800

./dev witness-compare tests/q122b-k6v6-a16-accuracy.conf \
  --profile llama_witness_q122b_seq32 --startup-timeout 1800
```

If both pass on seq32, the failure is definitively the HQQ+compact-KV combination. If either fails, the problem is in that individual path on Q122B.
