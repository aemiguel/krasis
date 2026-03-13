"""Quick test: run stress test prompts and print generated text to evaluate quality.

Focuses on reproducing the issue where large prompts produce irrelevant output.
Tests: short prompt (baseline), medium prompt, the 60K word prompt.
"""

import os
import sys
import time
import torch

os.environ.setdefault("KRASIS_LOG", "warn")

from krasis.config import QuantConfig, ModelConfig
from krasis.model import KrasisModel
from krasis.kv_cache import SequenceKVState
from krasis.sampler import sample
from krasis.timing import TIMING

TIMING.decode = False
TIMING.prefill = False


def generate(model, prompt, max_new_tokens=128, temperature=0.6, top_k=50, top_p=0.95):
    """Generate text from a prompt and return (text, stats_dict)."""
    tokenizer = model.tokenizer
    messages = [{"role": "user", "content": prompt}]
    tokens = tokenizer.apply_chat_template(messages)

    device = torch.device(model.ranks[0].device)
    vocab_size = model.cfg.vocab_size
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    stats = {
        "num_prompt_tokens": len(tokens),
        "prefill_time": 0,
        "decode_time": 0,
        "num_generated": 0,
        "error": None,
    }

    try:
        with torch.inference_mode():
            # Prefill
            t0 = time.perf_counter()
            prompt_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            positions = torch.arange(len(tokens), dtype=torch.int32, device=device)
            logits = model.forward(prompt_tensor, positions, seq_states)
            t1 = time.perf_counter()
            stats["prefill_time"] = t1 - t0

            # Check for NaN
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                stats["error"] = f"Prefill logits NaN={torch.isnan(logits).sum().item()} inf={torch.isinf(logits).sum().item()}"
                return "", stats

            next_logits = logits[-1:, :]
            next_token = sample(next_logits, temperature, top_k, top_p).item()
            generated_ids = [next_token]
            stop_ids = {model.cfg.eos_token_id}

            # Decode
            t2 = time.perf_counter()
            for step in range(max_new_tokens - 1):
                if next_token in stop_ids:
                    break
                pos = len(tokens) + step
                token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
                pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)
                logits = model.forward(token_tensor, pos_tensor, seq_states)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    stats["error"] = f"Decode step {step}: NaN/inf in logits"
                    break
                next_token = sample(logits, temperature, top_k, top_p).item()
                if next_token < 0 or next_token >= vocab_size:
                    stats["error"] = f"Token {step}: id={next_token} out of range"
                    break
                generated_ids.append(next_token)

            t3 = time.perf_counter()
            stats["decode_time"] = t3 - t2
            stats["num_generated"] = len(generated_ids)
            text = tokenizer.decode(generated_ids)
            return text, stats

    except Exception as e:
        stats["error"] = f"{type(e).__name__}: {e}"
        import traceback
        traceback.print_exc()
        return "", stats
    finally:
        for s in seq_states:
            s.free()


def run_tests(model_path, num_gpus, layer_group_size):
    """Load model and run test prompts."""
    quant_cfg = QuantConfig(
        attention="awq",
        shared_expert="int8",
        dense_mlp="int8",
        lm_head="int8",
    )

    print(f"\n{'='*60}")
    print(f"  LONG PROMPT TEST — {model_path}")
    print(f"  GPUs: {num_gpus}, LGS: {layer_group_size}")
    print(f"{'='*60}\n")

    model = KrasisModel(
        model_path=model_path,
        num_gpus=num_gpus,
        layer_group_size=layer_group_size,
        kv_dtype=torch.float8_e4m3fn,
        quant_cfg=quant_cfg,
        krasis_threads=12,
    )
    model.load()

    # ── Load Gutenberg prompts ──
    prompts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmarks", "prompts",
    )
    prompt_files = sorted(f for f in os.listdir(prompts_dir) if f.endswith(".txt"))
    def load_prompt(idx, max_chars=None):
        path = os.path.join(prompts_dir, prompt_files[idx % len(prompt_files)])
        with open(path) as f:
            text = f.read()
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        return text

    # ── Test prompts ──
    test_cases = []

    # 1. Short baseline
    test_cases.append(("SHORT BASELINE", "What is the capital of France?", 64))

    # 2. Medium prompt (~500 words / ~2K chars from Moby Dick)
    medium = load_prompt(0, max_chars=2000) + "\n\nSummarize the above passage."
    test_cases.append(("MEDIUM (~500 words)", medium, 128))

    # 3. Long prompt (~5K words / ~20K chars from War and Peace)
    long_5k = load_prompt(1, max_chars=20000) + "\n\nWhat are the main themes in the text above?"
    test_cases.append(("LONG (~5K words)", long_5k, 128))

    # 4. Very long prompt (~20K words / ~80K chars from Les Miserables)
    long_20k = load_prompt(2, max_chars=80000) + "\n\nSummarize the key events described above."
    test_cases.append(("VERY LONG (~20K words)", long_20k, 128))

    # 5. Full book (~60K words from Count of Monte Cristo)
    long_full = load_prompt(3) + "\n\nWhat is the overall narrative arc of the text above?"
    test_cases.append(("FULL BOOK (~60K words)", long_full, 128))

    # Warmup
    print("Warming up...", flush=True)
    text, stats = generate(model, "Hello, how are you?", max_new_tokens=16)
    print(f"  Warmup done: {stats['num_prompt_tokens']} tok prefill, generated: {text[:60]}\n")

    # Run tests
    for label, prompt, max_tokens in test_cases:
        print(f"\n{'─'*60}")
        print(f"  TEST: {label}")
        print(f"  Prompt preview: {prompt[:100]}...")
        print(f"  Prompt length: ~{len(prompt.split())} words")
        print(f"{'─'*60}")
        sys.stdout.flush()

        text, stats = generate(model, prompt, max_new_tokens=max_tokens)

        print(f"\n  Prompt tokens: {stats['num_prompt_tokens']}")
        print(f"  Prefill time:  {stats['prefill_time']:.1f}s ({stats['num_prompt_tokens']/stats['prefill_time']:.0f} tok/s)" if stats['prefill_time'] > 0 else "  Prefill time: N/A")
        print(f"  Generated:     {stats['num_generated']} tokens in {stats['decode_time']:.1f}s ({stats['num_generated']/stats['decode_time']:.1f} tok/s)" if stats['decode_time'] > 0 else f"  Generated: {stats['num_generated']} tokens")
        if stats["error"]:
            print(f"  \033[31mERROR: {stats['error']}\033[0m")
        print(f"\n  \033[1mGenerated text:\033[0m")
        print(f"  ┌{'─'*56}┐")
        for line in text.split('\n'):
            # wrap long lines
            while len(line) > 54:
                print(f"  │ {line[:54]} │")
                line = line[54:]
            print(f"  │ {line:<54} │")
        print(f"  └{'─'*56}┘")

        # Relevance check: does the output mention any key terms from the prompt?
        prompt_lower = prompt.lower()
        text_lower = text.lower()
        relevant_terms = []
        for term in ["summary", "finding", "section", "chapter", "experiment", "latency",
                      "consensus", "distributed", "algorithm", "capital", "france", "paris"]:
            if term in prompt_lower and term in text_lower:
                relevant_terms.append(term)

        if relevant_terms:
            print(f"  \033[32mRelevance: mentions {', '.join(relevant_terms)} from prompt\033[0m")
        elif len(text.strip()) < 5:
            print(f"  \033[33mRelevance: very short/empty output\033[0m")
        else:
            print(f"  \033[31mRelevance: NO key terms from prompt found in output — possible quality issue\033[0m")

        sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"  ALL TESTS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--layer-group-size", type=int, default=2)
    args = parser.parse_args()
    run_tests(args.model_path, args.num_gpus, args.layer_group_size)
