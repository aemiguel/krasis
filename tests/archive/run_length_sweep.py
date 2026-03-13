"""Sweep prompt lengths to find where output quality degrades."""

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


def generate(model, prompt, max_new_tokens=64, temperature=0.6, top_k=50, top_p=0.95):
    tokenizer = model.tokenizer
    messages = [{"role": "user", "content": prompt}]
    tokens = tokenizer.apply_chat_template(messages)

    device = torch.device(model.ranks[0].device)
    vocab_size = model.cfg.vocab_size
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

    try:
        with torch.inference_mode():
            t0 = time.perf_counter()
            prompt_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            positions = torch.arange(len(tokens), dtype=torch.int32, device=device)
            logits = model.forward(prompt_tensor, positions, seq_states)
            t1 = time.perf_counter()

            has_nan = torch.isnan(logits).any().item()
            has_inf = torch.isinf(logits).any().item()

            if has_nan or has_inf:
                return "", len(tokens), t1 - t0, 0, f"NaN={torch.isnan(logits).sum().item()} inf={torch.isinf(logits).sum().item()}"

            next_logits = logits[-1:, :]
            next_token = sample(next_logits, temperature, top_k, top_p).item()
            generated_ids = [next_token]
            stop_ids = {model.cfg.eos_token_id}

            t2 = time.perf_counter()
            for step in range(max_new_tokens - 1):
                if next_token in stop_ids:
                    break
                pos = len(tokens) + step
                token_tensor = torch.tensor([next_token], dtype=torch.long, device=device)
                pos_tensor = torch.tensor([pos], dtype=torch.int32, device=device)
                logits = model.forward(token_tensor, pos_tensor, seq_states)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    return tokenizer.decode(generated_ids), len(tokens), t1 - t0, len(generated_ids), f"NaN at decode step {step}"
                next_token = sample(logits, temperature, top_k, top_p).item()
                if next_token < 0 or next_token >= vocab_size:
                    break
                generated_ids.append(next_token)
            t3 = time.perf_counter()

            return tokenizer.decode(generated_ids), len(tokens), t1 - t0, len(generated_ids), None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return "", len(tokens) if 'tokens' in dir() else 0, 0, 0, str(e)
    finally:
        for s in seq_states:
            s.free()


def make_prompt(target_words):
    """Load a Gutenberg prompt truncated to approximately target_words words."""
    from prompt_utils import load_prompt_text_truncated
    # ~5 chars per word is a rough estimate for literary text
    return load_prompt_text_truncated(max_chars=target_words * 5)


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

    # Warmup
    print("Warming up...", flush=True)
    generate(model, "Hello!", max_new_tokens=8)
    generate(model, "What is 2+2?", max_new_tokens=8)
    print("Warmup done.\n")

    # Sweep: 100, 200, 500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000 words
    word_counts = [100, 200, 500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000, 30000]

    print(f"{'Words':>8} {'Tokens':>8} {'Prefill':>10} {'GenTok':>6} {'Error':>20}  Output (first 100 chars)")
    print("-" * 120)

    for target_words in word_counts:
        prompt = make_prompt(target_words)
        actual_words = len(prompt.split())
        text, n_tokens, prefill_t, n_gen, error = generate(model, prompt, max_new_tokens=64)

        preview = text.replace('\n', '\\n')[:100]
        err_str = error[:20] if error else ""
        pp_speed = n_tokens / prefill_t if prefill_t > 0 else 0

        status = "OK" if not error else "ERR"
        if not error and n_gen > 5:
            # Quick relevance check
            text_lower = text.lower()
            if any(w in text_lower for w in ["section", "experiment", "result", "accuracy", "method", "document", "summarize", "summary", "finding"]):
                status = "RELEVANT"
            else:
                status = "IRRELEVANT?"

        print(f"{actual_words:>8} {n_tokens:>8} {pp_speed:>8.0f}t/s {n_gen:>6} {status:>12}  {preview}")
        sys.stdout.flush()

    print("\nDone.")
