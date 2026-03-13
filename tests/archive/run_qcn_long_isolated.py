"""Test QCN at 25K+ tokens in isolation (no prior tests) with varied content."""

import os
import sys
import time
import torch

os.environ.setdefault("KRASIS_LOG", "warn")

from krasis.config import QuantConfig
from krasis.model import KrasisModel
from krasis.kv_cache import SequenceKVState
from krasis.sampler import sample
from krasis.timing import TIMING

TIMING.decode = False
TIMING.prefill = False


def generate(model, prompt, max_new_tokens=64):
    tokenizer = model.tokenizer
    messages = [{"role": "user", "content": prompt}]
    tokens = tokenizer.apply_chat_template(messages)

    device = torch.device(model.ranks[0].device)
    vocab_size = model.cfg.vocab_size
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    try:
        with torch.inference_mode():
            prompt_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            positions = torch.arange(len(tokens), dtype=torch.int32, device=device)
            t0 = time.perf_counter()
            logits = model.forward(prompt_tensor, positions, seq_states)
            t1 = time.perf_counter()

            nan_count = torch.isnan(logits).sum().item()
            inf_count = torch.isinf(logits).sum().item()

            print(f"  Prompt: {len(tokens)} tok, Prefill: {t1-t0:.1f}s ({len(tokens)/(t1-t0):.0f} t/s)")
            print(f"  Logits: NaN={nan_count}, inf={inf_count}, max={logits.float().max().item():.2f}")

            if nan_count > 0 or inf_count > 0:
                print(f"  ERROR: Bad logits!")
                return

            next_token = sample(logits[-1:, :], 0.6, 50, 0.95).item()
            generated_ids = [next_token]
            stop_ids = {model.cfg.eos_token_id}

            for step in range(max_new_tokens - 1):
                if next_token in stop_ids:
                    break
                pos = len(tokens) + step
                token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
                pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)
                logits = model.forward(token_tensor, pos_tensor, seq_states)
                if torch.isnan(logits).any():
                    print(f"  NaN at decode step {step}")
                    break
                next_token = sample(logits, 0.6, 50, 0.95).item()
                if next_token < 0 or next_token >= vocab_size:
                    break
                generated_ids.append(next_token)

            text = tokenizer.decode(generated_ids)
            print(f"  Generated {len(generated_ids)} tokens: {text[:200]}")

    finally:
        for s in seq_states:
            s.free()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--layer-group-size", type=int, default=2)
    args = parser.parse_args()

    quant_cfg = QuantConfig(attention="awq", shared_expert="int8", dense_mlp="int8", lm_head="int8")

    model = KrasisModel(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        layer_group_size=args.layer_group_size,
        kv_dtype=torch.float8_e4m3fn,
        quant_cfg=quant_cfg,
        krasis_threads=12,
    )
    model.load()

    # Warmup only
    print("Warmup...")
    generate(model, "Hello!", max_new_tokens=8)

    # Test 1: 10K tokens with VARIED content (not repetitive)
    import random
    random.seed(42)
    topics = [
        "machine learning", "database systems", "network security", "compiler design",
        "operating systems", "distributed computing", "computer graphics", "natural language processing",
        "robotics", "quantum computing", "bioinformatics", "game theory",
        "signal processing", "cryptography", "human-computer interaction",
    ]
    methods = ["regression analysis", "Monte Carlo simulation", "gradient descent", "dynamic programming",
               "branch and bound", "simulated annealing", "genetic algorithms", "Bayesian inference"]

    sections = []
    for i in range(200):
        topic = topics[i % len(topics)]
        method = methods[i % len(methods)]
        acc = 70 + random.randint(0, 29)
        lat = random.randint(5, 500)
        sections.append(
            f"Section {i+1} ({topic}): Using {method}, we achieved {acc}% accuracy "
            f"with {lat}ms latency. The key finding was that {topic} benefits significantly "
            f"from {method} when the dataset exceeds {random.randint(100, 10000)} samples. "
            f"Statistical significance was confirmed with p={random.uniform(0.001, 0.05):.4f}."
        )

    varied_10k = "Summarize the key findings from this research survey:\n\n" + "\n\n".join(sections)
    print(f"\n{'='*60}")
    print(f"  TEST 1: VARIED 10K — isolated, first real test")
    print(f"{'='*60}")
    generate(model, varied_10k, max_new_tokens=128)

    # Test 2: 25K tokens with varied content
    sections_25k = []
    for i in range(500):
        topic = topics[i % len(topics)]
        method = methods[i % len(methods)]
        acc = 70 + random.randint(0, 29)
        lat = random.randint(5, 500)
        sections_25k.append(
            f"Section {i+1} ({topic}): Using {method}, we achieved {acc}% accuracy "
            f"with {lat}ms latency. The key finding was that {topic} benefits significantly "
            f"from {method} when the dataset exceeds {random.randint(100, 10000)} samples. "
            f"Statistical significance was confirmed with p={random.uniform(0.001, 0.05):.4f}."
        )

    varied_25k = "Summarize the key findings from this comprehensive research survey:\n\n" + "\n\n".join(sections_25k)
    print(f"\n{'='*60}")
    print(f"  TEST 2: VARIED 25K — isolated")
    print(f"{'='*60}")
    generate(model, varied_25k, max_new_tokens=128)

    # Test 3: Run the 10K test AGAIN (multiple long prompts)
    print(f"\n{'='*60}")
    print(f"  TEST 3: VARIED 10K again — check for state leakage")
    print(f"{'='*60}")
    generate(model, varied_10k, max_new_tokens=128)

    print("\nDone.")
