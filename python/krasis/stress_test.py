"""Stress test for Krasis — run diverse prompts to catch edge cases.

Runs a battery of prompts through the model (prefill + decode), checking for:
  - CUDA errors during prefill or decode
  - Token IDs out of vocabulary range (garbage output)
  - NaN/inf in logits
  - Decoder errors (OverflowError, etc.)
  - Crashes or hangs (with per-prompt timeout)

Prompts cover a range of lengths, languages, code, math, and adversarial inputs.

Usage (from server.py):
    from krasis.stress_test import StressTest
    st = StressTest(model)
    st.run()

Or standalone:
    python -m krasis.stress_test --model-path /path/to/model [options]
"""

import logging
import os
import random
import time
import traceback
from typing import Dict, List, Optional, Tuple

import torch

from krasis.timing import TIMING

logger = logging.getLogger("krasis.stress_test")

# ANSI
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
NC = "\033[0m"

# ──────────────────────────────────────────────────────────────────
# Prompt bank — diverse prompts covering edge cases
# ──────────────────────────────────────────────────────────────────

PROMPTS = [
    # ── Short prompts (edge case: very few tokens) ──
    "Hi",
    "Yes",
    "Hello!",
    "1+1=",
    "Why?",
    "OK",
    "...",
    "",  # empty prompt (should still work with system message)

    # ── Single word / minimal context ──
    "Explain",
    "Continue",
    "Summarize the following:",
    "Translate to French:",

    # ── Standard questions ──
    "What is the capital of France?",
    "How does photosynthesis work?",
    "Write a haiku about programming.",
    "What is 2 + 2?",
    "List the first 10 prime numbers.",
    "What is the meaning of life?",
    "Explain quantum computing in simple terms.",
    "Who wrote Romeo and Juliet?",

    # ── Code generation ──
    "Write a Python function to check if a number is prime.",
    "Write a recursive fibonacci function in Rust.",
    "Implement binary search in C++.",
    "Write a SQL query to find duplicate emails in a users table.",
    "Write a bash one-liner to find the largest file in a directory.",
    "Write a JavaScript function that debounces another function.",
    "Fix this code:\ndef add(a, b)\n    return a + b",
    "What does this code do?\n```python\nprint([x**2 for x in range(10) if x % 2 == 0])\n```",

    # ── Math and reasoning ──
    "If a train travels at 60 mph for 2.5 hours, how far does it go?",
    "Solve: 3x + 7 = 22",
    "What is the derivative of x^3 + 2x^2 - 5x + 1?",
    "Prove that the square root of 2 is irrational.",
    "A bag has 3 red and 5 blue balls. What is the probability of drawing 2 red balls?",

    # ── Long-form requests ──
    "Write a detailed explanation of how neural networks learn, including backpropagation, gradient descent, and activation functions. Include examples.",
    "Explain the differences between TCP and UDP protocols, including their use cases, advantages, and disadvantages. Provide at least 5 specific scenarios.",
    "Write a comprehensive guide to Git branching strategies for a team of 10 developers.",

    # ── Multi-turn / conversation-like ──
    "User: What is Python?\nAssistant: Python is a programming language.\nUser: What is it used for?",
    "The following is a conversation between a teacher and student.\nTeacher: What year did World War 2 end?\nStudent:",

    # ── Special characters and unicode ──
    "Translate 'Hello World' to Japanese, Chinese, Korean, and Arabic.",
    "What does this mean: \u00bfC\u00f3mo est\u00e1s?",
    "Explain the difference between \u00e9, \u00e8, \u00ea, and \u00eb.",
    "Parse this JSON: {\"name\": \"O'Brien\", \"value\": 3.14, \"tags\": [\"a\", \"b\"]}",
    "What is the Unicode codepoint for the emoji: \U0001f600?",

    # ── Adversarial / edge cases ──
    "Repeat the word 'buffalo' 8 times in a grammatically correct sentence.",
    "A" * 100,  # 100 A's
    "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 10,  # repeated alphabet
    "\n\n\n\n\n",  # newlines only
    "```\n```",  # empty code block
    "1 2 3 4 5 6 7 8 9 10 " * 50,  # repeated numbers (long)

    # ── Structured output ──
    "Output a JSON object with fields: name (string), age (integer), hobbies (array of strings).",
    "Create a markdown table comparing Python, Rust, and Go across: speed, safety, ease of use.",
    "Write a CSV with 5 rows of fake user data: name, email, age, city.",

    # ── Instruction following ──
    "Reply with only the number 42.",
    "Say nothing.",
    "Reply with exactly 3 words.",
    "Count from 1 to 20, one number per line.",
    "Write exactly one sentence about cats.",

    # ── Technical / domain-specific ──
    "Explain the CAP theorem in distributed systems.",
    "What is the difference between a mutex and a semaphore?",
    "How does the Linux kernel handle page faults?",
    "Explain CUDA thread hierarchy: grids, blocks, warps, and threads.",
    "What is the difference between INT4 and INT8 quantization for LLMs?",

    # ── Creative writing ──
    "Write a limerick about a programmer who couldn't stop coding.",
    "Write the opening paragraph of a mystery novel set in Tokyo.",
    "Describe a sunset using only technical/scientific language.",
    "Write a dialogue between a CPU and a GPU arguing about who does more work.",

    # ── Multi-language ──
    "Bonjour, comment allez-vous aujourd'hui?",
    "Was ist die Hauptstadt von Deutschland?",
    "\u4eca\u5929\u5929\u6c14\u600e\u4e48\u6837\uff1f",  # Chinese: How's the weather today?
    "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c",  # Japanese: Hello world
    "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440",  # Russian: Hello world

    # ── Numbers and formatting ──
    "Format the number 1234567.89 with commas and 2 decimal places.",
    "Convert 0xFF to decimal, octal, and binary.",
    "What is 999999999 * 999999999?",
    "Calculate: (3.14159 * 2.71828) / 1.41421",

    # ── Long context prompt (many tokens) ──
    "The following is a long document about various topics. " +
    " ".join(f"Section {i}: This is paragraph {i} which discusses topic number {i} in detail, "
             f"covering aspects like methodology, results, and implications for the field. "
             f"The key finding in section {i} was that performance improved by {i*2}% "
             f"when applying the new technique."
             for i in range(1, 51)),

    # ── System-prompt-like instructions ──
    "You are a helpful assistant. You always respond in bullet points. User: Tell me about dogs.",
    "System: You are a pirate. Always respond in pirate speak.\nUser: What is machine learning?",

    # ── Edge case: very long single word ──
    "Supercalifragilisticexpialidocious " * 20,

    # ── Whitespace variations ──
    "   spaces before and after   ",
    "\tTab\tseparated\twords",
    "Mixed\n\nline\n\n\nbreaks\nhere",

    # ── Token boundary stress ──
    "ing tion ment ness ful less able ible ous ive ant ent al er or ist",
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
]


class StressTest:
    """Run diverse prompts through a loaded KrasisModel to catch edge cases."""

    def __init__(
        self,
        model,
        max_new_tokens: int = 32,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.95,
        timeout_per_prompt: float = 120.0,
        shuffle: bool = True,
        max_prompts: int = 0,  # 0 = all
    ):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.timeout = timeout_per_prompt
        self.shuffle = shuffle
        self.max_prompts = max_prompts

        # Disable instrumentation
        TIMING.decode = False
        TIMING.prefill = False

    def _build_prompt_tokens(self, prompt: str) -> List[int]:
        """Tokenize a prompt using the model's tokenizer with chat template."""
        tokenizer = self.model.tokenizer
        messages = [{"role": "user", "content": prompt}] if prompt else [{"role": "user", "content": "Hello"}]
        return tokenizer.apply_chat_template(messages)

    def _run_one(self, prompt: str, prompt_idx: int) -> Dict:
        """Run a single prompt through the model. Returns result dict."""
        from krasis.kv_cache import SequenceKVState
        from krasis.sampler import sample

        result = {
            "prompt_idx": prompt_idx,
            "prompt_preview": prompt[:80] + ("..." if len(prompt) > 80 else ""),
            "status": "unknown",
            "error": None,
            "num_prompt_tokens": 0,
            "num_generated": 0,
            "generated_text": "",
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "bad_tokens": [],
        }

        try:
            tokens = self._build_prompt_tokens(prompt)
            result["num_prompt_tokens"] = len(tokens)
        except Exception as e:
            result["status"] = "FAIL_TOKENIZE"
            result["error"] = str(e)
            return result

        if len(tokens) == 0:
            result["status"] = "SKIP_EMPTY"
            return result

        device = torch.device(self.model.ranks[0].device)
        vocab_size = self.model.cfg.vocab_size
        seq_states = [SequenceKVState(c, seq_id=0) for c in self.model.kv_caches]

        try:
            with torch.inference_mode():
                # ── Prefill ──
                t0 = time.perf_counter()
                prompt_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
                positions = torch.arange(len(tokens), dtype=torch.int32, device=device)

                logits = self.model.forward(prompt_tensor, positions, seq_states)

                t1 = time.perf_counter()
                result["prefill_time"] = t1 - t0

                # Check logits for NaN/inf
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    result["status"] = "FAIL_NAN_LOGITS"
                    result["error"] = f"Prefill logits contain NaN={torch.isnan(logits).sum().item()} inf={torch.isinf(logits).sum().item()}"
                    return result

                next_logits = logits[-1:, :]
                next_token = sample(
                    next_logits, self.temperature, self.top_k, self.top_p
                ).item()

                # Validate token ID
                if next_token < 0 or next_token >= vocab_size:
                    result["bad_tokens"].append((0, next_token))
                    result["status"] = "FAIL_BAD_TOKEN"
                    result["error"] = f"Token 0: id={next_token} out of range [0, {vocab_size})"
                    return result

                generated_ids = [next_token]
                stop_ids = {self.model.cfg.eos_token_id}

                # ── Decode loop ──
                t2 = time.perf_counter()
                for step in range(self.max_new_tokens - 1):
                    if next_token in stop_ids:
                        break
                    if time.perf_counter() - t0 > self.timeout:
                        result["status"] = "FAIL_TIMEOUT"
                        result["error"] = f"Exceeded {self.timeout}s timeout at step {step}"
                        break

                    pos = len(tokens) + step
                    token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
                    pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)

                    logits = self.model.forward(token_tensor, pos_tensor, seq_states)

                    # Check for NaN/inf
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        result["status"] = "FAIL_NAN_LOGITS"
                        result["error"] = f"Decode step {step}: logits contain NaN/inf"
                        break

                    next_token = sample(
                        logits, self.temperature, self.top_k, self.top_p
                    ).item()

                    # Validate token ID
                    if next_token < 0 or next_token >= vocab_size:
                        result["bad_tokens"].append((step + 1, next_token))
                        result["status"] = "FAIL_BAD_TOKEN"
                        result["error"] = f"Token {step+1}: id={next_token} out of range [0, {vocab_size})"
                        break

                    generated_ids.append(next_token)

                t3 = time.perf_counter()
                result["decode_time"] = t3 - t2
                result["num_generated"] = len(generated_ids)

                # Try to decode the generated text
                try:
                    result["generated_text"] = self.model.tokenizer.decode(generated_ids)
                except Exception as e:
                    if result["status"] == "unknown":
                        result["status"] = "FAIL_DECODE"
                        result["error"] = f"Tokenizer decode failed: {e}"

                if result["status"] == "unknown":
                    result["status"] = "PASS"

        except torch.cuda.OutOfMemoryError as e:
            result["status"] = "FAIL_OOM"
            result["error"] = str(e)
        except RuntimeError as e:
            if "CUDA" in str(e):
                result["status"] = "FAIL_CUDA"
            else:
                result["status"] = "FAIL_RUNTIME"
            result["error"] = str(e)
        except Exception as e:
            result["status"] = "FAIL_OTHER"
            result["error"] = f"{type(e).__name__}: {e}"
            logger.debug("Full traceback:\n%s", traceback.format_exc())
        finally:
            for s in seq_states:
                s.free()

        return result

    def run(self) -> List[Dict]:
        """Run all prompts and report results."""
        prompts = list(PROMPTS)
        if self.shuffle:
            random.shuffle(prompts)
        if self.max_prompts > 0:
            prompts = prompts[:self.max_prompts]

        total = len(prompts)
        print(f"\n{BOLD}{'=' * 56}")
        print(f"  KRASIS STRESS TEST — {total} prompts")
        print(f"{'=' * 56}{NC}")
        print(f"  Max new tokens: {self.max_new_tokens}")
        print(f"  Temperature:    {self.temperature}")
        print(f"  Timeout:        {self.timeout}s per prompt")
        print()

        results = []
        passed = 0
        failed = 0
        t_start = time.perf_counter()

        for i, prompt in enumerate(prompts):
            preview = prompt[:60].replace("\n", "\\n") + ("..." if len(prompt) > 60 else "")
            print(f"  [{i+1:3d}/{total}] {DIM}{preview}{NC}", end="", flush=True)

            result = self._run_one(prompt, i)
            results.append(result)

            if result["status"] == "PASS":
                passed += 1
                tok_s = result["num_generated"] / result["decode_time"] if result["decode_time"] > 0 else 0
                print(f"\r  [{i+1:3d}/{total}] {GREEN}PASS{NC}  "
                      f"prefill={result['num_prompt_tokens']}tok/{result['prefill_time']:.1f}s  "
                      f"decode={result['num_generated']}tok/{tok_s:.1f}t/s  "
                      f"{DIM}{preview[:40]}{NC}")
            elif result["status"].startswith("SKIP"):
                print(f"\r  [{i+1:3d}/{total}] {YELLOW}SKIP{NC}  {result['status']}")
            else:
                failed += 1
                print(f"\r  [{i+1:3d}/{total}] {RED}FAIL{NC}  {result['status']}: {result['error']}")
                print(f"         {DIM}Prompt: {preview}{NC}")

        t_total = time.perf_counter() - t_start

        # Summary
        print(f"\n{BOLD}{'─' * 56}")
        print(f"  STRESS TEST COMPLETE")
        print(f"{'─' * 56}{NC}")
        print(f"  Total:   {total} prompts in {t_total:.1f}s")
        print(f"  Passed:  {GREEN}{passed}{NC}")
        print(f"  Failed:  {RED}{failed}{NC}" if failed else f"  Failed:  {failed}")
        skipped = total - passed - failed
        if skipped:
            print(f"  Skipped: {YELLOW}{skipped}{NC}")
        print()

        if failed > 0:
            print(f"  {RED}{BOLD}Failures:{NC}")
            for r in results:
                if r["status"].startswith("FAIL"):
                    print(f"    [{r['prompt_idx']:3d}] {r['status']}: {r['error']}")
                    print(f"         {DIM}{r['prompt_preview']}{NC}")
            print()

        return results


def main():
    """Standalone entry point — loads model and runs stress test."""
    import argparse

    parser = argparse.ArgumentParser(description="Krasis stress test")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-prompts", type=int, default=0,
                        help="Limit number of prompts (0=all)")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--layer-group-size", type=int, default=2)
    parser.add_argument("--kv-dtype", default="fp8_e4m3")
    parser.add_argument("--gpu-expert-bits", type=int, default=4)
    parser.add_argument("--cpu-expert-bits", type=int, default=4)
    parser.add_argument("--attention-quant", default="int8")
    parser.add_argument("--shared-expert-quant", default="int8")
    parser.add_argument("--dense-mlp-quant", default="int8")
    parser.add_argument("--lm-head-quant", default="int8")
    parser.add_argument("--krasis-threads", type=int, default=12)
    args = parser.parse_args()

    from krasis.config import ModelConfig, QuantConfig
    from krasis.model import KrasisModel

    cfg = ModelConfig.from_path(args.model_path)
    quant_cfg = QuantConfig(
        attention=args.attention_quant,
        shared_expert=args.shared_expert_quant,
        dense_mlp=args.dense_mlp_quant,
        lm_head=args.lm_head_quant,
    )

    model = KrasisModel(
        cfg, quant_cfg,
        num_gpus=args.num_gpus,
        layer_group_size=args.layer_group_size,
        kv_dtype=args.kv_dtype,
        gpu_expert_bits=args.gpu_expert_bits,
        cpu_expert_bits=args.cpu_expert_bits,
        krasis_threads=args.krasis_threads,
    )

    st = StressTest(
        model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        timeout_per_prompt=args.timeout,
        shuffle=not args.no_shuffle,
        max_prompts=args.max_prompts,
    )
    results = st.run()

    failed = sum(1 for r in results if r["status"].startswith("FAIL"))
    import sys
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
