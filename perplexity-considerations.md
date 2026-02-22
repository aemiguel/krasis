# Perplexity Considerations for Krasis

Measuring perplexity (PPL) is the standard way to validate that a custom runtime like Krasis, with its aggressive quantization and specialized kernels, preserves the model's intelligence and predictive accuracy.

## 1. Why Perplexity Matters for Krasis

Krasis uses a unique **dual-format cache** system:
- **Marlin (INT4/INT8)** on GPU for fast prefill.
- **Sequential/AVX2 optimized (INT4/INT8)** on CPU for decode.

Because these two paths use different quantization methods and execution kernels, we must ensure that neither path introduces significant numerical drift or "garbage" outputs that wouldn't be caught by simple token-matching tests. Perplexity provides a continuous metric for this.

## 2. Implementation Challenges in the Current Runtime

The current Krasis implementation is optimized for **throughput and latency**, which introduces two main hurdles for PPL measurement:

### A. The "Logit Slicing" Optimization
In `python/krasis/model.py`, the `forward` method explicitly slices the hidden state when processing sequences ($M > 1$):
```python
if M > 1:
    hidden = hidden[-1:, :]
logits = _linear(hidden, self.lm_head_data)
```
This is a standard optimization for autoregressive generation where only the next token's probability is needed. To calculate PPL for a sequence of $N$ tokens, we need the logits for $N-1$ positions (all except the very first input).

### B. Dual-Path Execution
- **Prefill Path:** When $M \ge 	ext{threshold}$, Krasis uses `GpuPrefillManager` and Marlin kernels.
- **Decode Path:** When $M=1$ (or below threshold), Krasis uses the CPU expert engine (or HCS).

An accurate PPL measurement suite should be able to toggle between these paths to verify that the Marlin prefill produces the same quality as the CPU decode.

## 3. Proposed Methodology

### Step 1: Modify `forward()` for Full Logit Access
We should add a `return_all_logits` flag to the `forward` pass:
```python
def forward(self, ..., return_all_logits=False):
    # ... layer loop ...
    if not return_all_logits and M > 1:
        hidden = hidden[-1:, :]
    logits = _linear(hidden, self.lm_head_data)
    return logits.float()
```

### Step 2: Cross-Entropy Calculation
Perplexity is defined as $e^{	ext{loss}}$, where loss is the average negative log-likelihood:
1.  Shift logits and labels so that `logits[i]` predicts `labels[i+1]`.
2.  Use `torch.nn.functional.cross_entropy`.
3.  **Critical:** Always perform the final softmax and loss calculation in **FP32** to avoid overflows/underflows common in quantized outputs.

### Step 3: Sliding Window Evaluation
For long datasets (like WikiText-2), use a sliding window (e.g., 2048 or 4096 tokens).
- Avoid "stride=1" as it is extremely slow.
- Use a stride equal to half the window size for a good balance between speed and accuracy (handling the "early token" problem where the model has little context).

## 4. Specific Krasis Considerations

### Verifying Marlin Prefill Accuracy
Since Marlin is used for the bulk of prompt processing in Opencode/IDE use cases, we need to ensure that the prefill hidden states are accurate.
- **Test:** Compare PPL of a 2048-token block processed as a single "prefill" chunk vs. processed token-by-token (decode). They should be nearly identical.

### Impact of `routed_scaling_factor`
Krasis calculates scaling factors dynamically for MoE. If PPL is significantly higher (worse) than the base model (e.g., in `llama.cpp` or `transformers`), the first place to look is the MoE routing and scaling logic in `gpu_prefill.py` and `moe.rs`.

### KV Cache Quantization (FP8)
Krasis supports FP8 KV caches. PPL measurement is the only way to determine if the `kv_dtype=torch.float8_e4m3fn` setting is causing degradation in long-context reasoning.

## 5. Recommended Test Suite

To demonstrate Krasis is a "valid runtime," we should provide a `measure_ppl.py` script that:
1.  Loads a model (e.g., DeepSeek-V2-Lite or Qwen3-Coder-Next).
2.  Downloads **WikiText-2-raw-v1** (test split).
3.  Runs evaluation on the first 10-20MB of text.
4.  Reports:
    - **Total PPL**
    - **Bits-per-character (BPC)**
    - **Execution path used** (Prefill vs. Decode)

## 6. Target Benchmarks
We should compare Krasis PPL against:
- **FP16 Baseline:** The "gold standard" for the model.
- **GGUF (Q4_K_M / Q8_0):** To show we are competitive with industry-standard quantization.
- **KTransformers:** To demonstrate our custom engine is at least as accurate as the previous solution.
