#!/usr/bin/env python3
"""Pure CPU decode timing breakdown for Qwen3-Coder-Next.

Runs the model with pure_cpu decode (expert_divisor=None equivalent)
and KRASIS_DECODE_TIMING=1 to get per-component timing stats.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python tests/test_pure_cpu_timing.py
"""

import logging, sys, time, os

os.environ["KRASIS_DECODE_TIMING"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from krasis.model import KrasisModel

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3-Coder-Next")


def main():
    print("=== Pure CPU Decode Timing: Qwen3-Coder-Next ===")
    print("Loading model (PP=2, 2 GPUs, pure_cpu decode)...")

    model = KrasisModel(
        model_path=os.path.abspath(MODEL),
        pp_partition=[24, 24],
        devices=["cuda:0", "cuda:1"],
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=False,  # No GPU prefill — pure CPU for all MoE
    )
    model.load()
    t_load = time.perf_counter()

    # Warmup: small prompt + 2 tokens
    print("Warmup...")
    warmup = model.tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}])
    model.generate(warmup, max_new_tokens=2, temperature=0.0, top_k=1)

    # Actual test: moderate prompt + 16 decode tokens with timing
    print("\n" + "=" * 70)
    print("DECODE TIMING (16 tokens, per-layer breakdown)")
    print("=" * 70)

    messages = [{"role": "user", "content": "Explain the difference between TCP and UDP in 2 sentences."}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt)} tokens, generating 16 decode tokens\n")

    t0 = time.perf_counter()
    out = model.generate(prompt, max_new_tokens=16, temperature=0.0, top_k=1)
    t1 = time.perf_counter()

    text = model.tokenizer.decode(out[len(prompt):])
    print(f"\nOutput: {repr(text)}")
    print(f"Total generation: {t1 - t0:.2f}s ({16 / (t1 - t0):.2f} tok/s)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Check the per-layer LAYER-TIMING and CPU-EXPERT logs above for breakdown.")
    print("Key components per MoE layer:")
    print("  - norm1: input RMSNorm (GPU)")
    print("  - attn: attention computation (GPU)")
    print("  - norm2: post-attention RMSNorm (GPU)")
    print("  - mlp: MoE expert dispatch (contains GPU->CPU->GPU cycle)")
    print()
    print("Key components in CPU-EXPERT per layer:")
    print("  - gpu->cpu: hidden state DMA to CPU")
    print("  - bytes: torch->numpy conversion for PyO3")
    print("  - rust: Rust MoE kernel (INT4 AVX2)")
    print("  - cpu->gpu: output DMA back to GPU")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
