#!/usr/bin/env python3
"""Thorough V2-Lite testing: multiple prompts, repetition detection, HF reference comparison.

Tests:
1. Generation quality: diverse prompts, check for coherent output
2. Repetition detection: generate longer sequences, detect looping
3. HF reference: compare prefill logits vs HuggingFace model (gold standard)
"""

import logging, sys, time, json, os, gc
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("v2lite-test")

import torch
import numpy as np

MODEL_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"
REFERENCE_DIR = "/tmp/v2lite_reference"


# ── Test prompts covering diverse capabilities ──
TEST_CASES = [
    {
        "name": "simple_math",
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "max_tokens": 16,
        "expect_contains": ["4"],
    },
    {
        "name": "counting",
        "messages": [{"role": "user", "content": "Count from 1 to 5, separated by commas."}],
        "max_tokens": 32,
        "expect_contains": ["1", "2", "3", "4", "5"],
    },
    {
        "name": "factual",
        "messages": [{"role": "user", "content": "What is the capital of France? One word answer."}],
        "max_tokens": 16,
        "expect_contains": ["Paris"],
    },
    {
        "name": "reasoning",
        "messages": [{"role": "user", "content": "If a train travels 60 miles in 1 hour, how far does it travel in 3 hours? Just give the number of miles."}],
        "max_tokens": 32,
        "expect_contains": ["180"],
    },
    {
        "name": "longer_generation",
        "messages": [{"role": "user", "content": "Explain why the sky is blue in 2-3 sentences."}],
        "max_tokens": 128,
        "expect_contains": [],  # Just check coherence and no repetition
    },
    {
        "name": "code",
        "messages": [{"role": "user", "content": "Write a Python function that adds two numbers. Just the function, no explanation."}],
        "max_tokens": 64,
        "expect_contains": ["def", "return"],
    },
]


def detect_repetition(text: str, min_pattern_len: int = 5, min_repeats: int = 3) -> bool:
    """Detect if text contains repeating patterns (sign of degeneration).

    Returns True if a pattern of length >= min_pattern_len repeats >= min_repeats times.
    """
    words = text.split()
    if len(words) < min_pattern_len * min_repeats:
        return False

    for pattern_len in range(min_pattern_len, len(words) // min_repeats + 1):
        for start in range(len(words) - pattern_len * min_repeats + 1):
            pattern = words[start:start + pattern_len]
            count = 1
            pos = start + pattern_len
            while pos + pattern_len <= len(words):
                if words[pos:pos + pattern_len] == pattern:
                    count += 1
                    pos += pattern_len
                else:
                    break
            if count >= min_repeats:
                return True
    return False


def test_generation_quality(model):
    """Test 1: Run all test cases, check output quality."""
    logger.info("=" * 60)
    logger.info("TEST 1: Generation Quality")
    logger.info("=" * 60)

    passed = 0
    failed = 0
    results = {}

    for tc in TEST_CASES:
        name = tc["name"]
        logger.info("\n--- %s ---", name)

        prompt = model.tokenizer.apply_chat_template(tc["messages"])
        logger.info("Prompt: %d tokens", len(prompt))

        t0 = time.perf_counter()
        out = model.generate(prompt, max_new_tokens=tc["max_tokens"], temperature=0.0, top_k=1)
        elapsed = time.perf_counter() - t0

        text = model.tokenizer.decode(out)
        logger.info("Output (%d tokens, %.2fs): %s", len(out), elapsed, repr(text))

        # Check expected content
        ok = True
        for expected in tc["expect_contains"]:
            if expected.lower() not in text.lower():
                logger.warning("  MISSING expected: %s", expected)
                ok = False

        # Check for repetition
        has_repeat = detect_repetition(text)
        if has_repeat:
            logger.warning("  REPETITION DETECTED in output!")
            ok = False

        if ok:
            logger.info("  PASS")
            passed += 1
        else:
            logger.warning("  FAIL")
            failed += 1

        results[name] = {"text": text, "tokens": len(out), "time": elapsed, "pass": ok}

    logger.info("\n=== Generation Quality: %d/%d passed ===", passed, passed + failed)
    return results


def test_long_generation(model):
    """Test 2: Generate 256 tokens, check for degeneration."""
    logger.info("=" * 60)
    logger.info("TEST 2: Long Generation (256 tokens)")
    logger.info("=" * 60)

    messages = [{"role": "user", "content": "Write a short story about a robot learning to paint."}]
    prompt = model.tokenizer.apply_chat_template(messages)
    logger.info("Prompt: %d tokens", len(prompt))

    t0 = time.perf_counter()
    out = model.generate(prompt, max_new_tokens=256, temperature=0.0, top_k=1)
    elapsed = time.perf_counter() - t0

    text = model.tokenizer.decode(out)
    logger.info("Output (%d tokens, %.2fs):", len(out), elapsed)
    # Print in chunks for readability
    for i in range(0, len(text), 120):
        logger.info("  %s", text[i:i+120])

    # Check for repetition
    has_repeat = detect_repetition(text)
    if has_repeat:
        logger.warning("FAIL: Repetition detected in long generation!")
    else:
        logger.info("PASS: No repetition in long generation")

    # Check unique token ratio (low ratio = degenerate)
    unique_tokens = len(set(out))
    ratio = unique_tokens / len(out) if out else 0
    logger.info("Unique token ratio: %d/%d = %.2f", unique_tokens, len(out), ratio)
    if ratio < 0.3:
        logger.warning("FAIL: Very low unique token ratio (likely looping)")
    else:
        logger.info("PASS: Healthy token diversity")

    return {"text": text, "tokens": len(out), "unique_ratio": ratio, "has_repeat": has_repeat}


def generate_hf_reference():
    """Generate reference logits using HuggingFace transformers."""
    logger.info("=" * 60)
    logger.info("Generating HuggingFace reference logits...")
    logger.info("=" * 60)

    os.makedirs(REFERENCE_DIR, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading HF tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)

    logger.info("Loading HF model (BF16, auto device map, native transformers)...")
    t0 = time.perf_counter()
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    hf_model.eval()
    logger.info("HF model loaded in %.1fs", time.perf_counter() - t0)

    # Generate reference for each test case
    for tc in TEST_CASES[:4]:  # First 4 test cases (quick ones)
        name = tc["name"]
        logger.info("--- HF reference: %s ---", name)

        # Use chat template
        messages = tc["messages"]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(hf_model.device)

        logger.info("Prompt tokens: %s", input_ids.shape)

        # Get prefill logits (use_cache=False to avoid DynamicCache compat issue)
        with torch.no_grad():
            outputs = hf_model(input_ids, use_cache=False)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        # Save last-token logits (for next-token prediction)
        last_logits = logits[0, -1, :].float().cpu()
        torch.save(last_logits, os.path.join(REFERENCE_DIR, f"{name}_logits.pt"))

        # Save top-k predictions
        topk_vals, topk_ids = last_logits.topk(10)
        tok_strs = [repr(tokenizer.decode([tid.item()])) for tid in topk_ids]
        logger.info("HF top10: %s", list(zip(tok_strs, [f"{v:.1f}" for v in topk_vals.tolist()])))

        # Also save token IDs for comparison
        torch.save(input_ids.cpu(), os.path.join(REFERENCE_DIR, f"{name}_input_ids.pt"))

        # Generate greedy output for comparison
        with torch.no_grad():
            gen_ids = hf_model.generate(
                input_ids,
                max_new_tokens=tc["max_tokens"],
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        gen_text = tokenizer.decode(gen_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        logger.info("HF output: %s", repr(gen_text))

        # Save generated text
        with open(os.path.join(REFERENCE_DIR, f"{name}_output.txt"), "w") as f:
            f.write(gen_text)

    # Clean up HF model to free VRAM
    del hf_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("HF model unloaded, VRAM freed")


def test_hf_comparison(model):
    """Test 3: Compare Krasis prefill logits against HuggingFace reference."""
    logger.info("=" * 60)
    logger.info("TEST 3: HF Reference Comparison")
    logger.info("=" * 60)

    if not os.path.exists(REFERENCE_DIR):
        logger.warning("No HF reference found at %s, skipping comparison", REFERENCE_DIR)
        return None

    from krasis.kv_cache import SequenceKVState

    device = torch.device(model.ranks[0].device)
    results = {}

    for tc in TEST_CASES[:4]:
        name = tc["name"]
        logits_path = os.path.join(REFERENCE_DIR, f"{name}_logits.pt")
        ids_path = os.path.join(REFERENCE_DIR, f"{name}_input_ids.pt")
        output_path = os.path.join(REFERENCE_DIR, f"{name}_output.txt")

        if not os.path.exists(logits_path):
            logger.info("Skipping %s (no reference)", name)
            continue

        logger.info("\n--- %s ---", name)

        # Load HF reference
        hf_logits = torch.load(logits_path, weights_only=True)  # [vocab_size] float32
        hf_input_ids = torch.load(ids_path, weights_only=True)  # [1, seq_len]
        with open(output_path) as f:
            hf_output = f.read()

        # Get Krasis token IDs for same prompt
        prompt = model.tokenizer.apply_chat_template(tc["messages"])
        logger.info("Krasis prompt tokens: %s", prompt[:20])
        logger.info("HF prompt tokens:     %s", hf_input_ids[0, :20].tolist())

        # Check if tokenization matches
        hf_ids = hf_input_ids[0].tolist()
        if prompt != hf_ids:
            logger.warning("Token mismatch! Krasis=%d tokens, HF=%d tokens", len(prompt), len(hf_ids))
            # Show first differences
            for i in range(min(len(prompt), len(hf_ids))):
                if prompt[i] != hf_ids[i]:
                    logger.warning("  First diff at pos %d: krasis=%d, hf=%d", i, prompt[i], hf_ids[i])
                    break
            # Use HF token IDs for fair comparison
            prompt = hf_ids
            logger.info("Using HF token IDs for fair logits comparison")

        # Run Krasis prefill
        seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]
        try:
            prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device)
            positions = torch.arange(len(prompt), dtype=torch.int32, device=device)
            krasis_logits = model.forward(prompt_tensor, positions, seq_states)
            # Last token logits
            kr_last = krasis_logits[-1].float().cpu()  # [vocab_size]
        finally:
            for s in seq_states:
                s.free()

        # Compare logits
        cos_sim = torch.nn.functional.cosine_similarity(
            kr_last.flatten(), hf_logits.flatten(), dim=0
        ).item()
        max_diff = (kr_last - hf_logits).abs().max().item()
        mean_diff = (kr_last - hf_logits).abs().mean().item()

        # Compare top-k predictions
        kr_topk_vals, kr_topk_ids = kr_last.topk(10)
        hf_topk_vals, hf_topk_ids = hf_logits.topk(10)

        kr_top10 = set(kr_topk_ids.tolist())
        hf_top10 = set(hf_topk_ids.tolist())
        top10_overlap = len(kr_top10 & hf_top10)

        # Check if greedy next token matches
        greedy_match = kr_topk_ids[0].item() == hf_topk_ids[0].item()

        kr_tok_strs = [repr(model.tokenizer.decode([tid.item()])) for tid in kr_topk_ids]
        hf_tok_strs = [repr(model.tokenizer.decode([tid.item()])) for tid in hf_topk_ids]

        logger.info("Krasis top5: %s", list(zip(kr_tok_strs[:5], [f"{v:.1f}" for v in kr_topk_vals[:5].tolist()])))
        logger.info("HF     top5: %s", list(zip(hf_tok_strs[:5], [f"{v:.1f}" for v in hf_topk_vals[:5].tolist()])))
        logger.info("Cosine similarity:  %.6f", cos_sim)
        logger.info("Max abs diff:       %.4f", max_diff)
        logger.info("Mean abs diff:      %.4f", mean_diff)
        logger.info("Top-10 overlap:     %d/10", top10_overlap)
        logger.info("Greedy match:       %s", greedy_match)

        if cos_sim > 0.999:
            logger.info("PASS: Logits match HF reference (cos=%.6f)", cos_sim)
        elif cos_sim > 0.99:
            logger.info("CLOSE: Minor differences (cos=%.6f)", cos_sim)
        elif cos_sim > 0.95:
            logger.warning("WARN: Noticeable differences (cos=%.6f)", cos_sim)
        else:
            logger.warning("FAIL: Logits diverge from HF (cos=%.6f)", cos_sim)

        # Compare generated text
        krasis_out = model.generate(
            prompt, max_new_tokens=tc["max_tokens"], temperature=0.0, top_k=1
        )
        krasis_text = model.tokenizer.decode(krasis_out)
        logger.info("Krasis output: %s", repr(krasis_text))
        logger.info("HF output:     %s", repr(hf_output))
        text_match = krasis_text.strip() == hf_output.strip()
        logger.info("Text match:    %s", text_match)

        results[name] = {
            "cos_sim": cos_sim,
            "max_diff": max_diff,
            "greedy_match": greedy_match,
            "top10_overlap": top10_overlap,
            "text_match": text_match,
        }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-only", action="store_true", help="Only generate HF reference (no Krasis)")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HF reference generation")
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation quality tests")
    args = parser.parse_args()

    # Step 1: Generate HF reference (if not already done)
    if args.hf_only:
        generate_hf_reference()
        return

    if not args.skip_hf and not os.path.exists(REFERENCE_DIR):
        generate_hf_reference()

    # Step 2: Load Krasis model
    from krasis.config import QuantConfig
    from krasis.model import KrasisModel

    logger.info("\n" + "=" * 60)
    logger.info("Loading Krasis V2-Lite PP=1...")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    qcfg = QuantConfig(attention="bf16", shared_expert="bf16", dense_mlp="bf16", lm_head="bf16")
    model = KrasisModel(
        model_path=MODEL_PATH,
        num_gpus=1,
        gpu_prefill=False,
        krasis_threads=16,
        quant_cfg=qcfg,
        kv_dtype=torch.bfloat16,
    )
    model.load()
    logger.info("Krasis model loaded in %.1fs", time.perf_counter() - t0)

    # Step 3: Run tests
    if not args.skip_gen:
        gen_results = test_generation_quality(model)
        long_results = test_long_generation(model)
    else:
        gen_results = None
        long_results = None

    hf_results = test_hf_comparison(model)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if gen_results:
        passed = sum(1 for v in gen_results.values() if v["pass"])
        total = len(gen_results)
        logger.info("Generation quality: %d/%d passed", passed, total)

    if long_results:
        logger.info("Long generation: repeat=%s, unique_ratio=%.2f",
                     long_results["has_repeat"], long_results["unique_ratio"])

    if hf_results:
        for name, r in hf_results.items():
            logger.info("HF comparison [%s]: cos=%.6f greedy_match=%s text_match=%s",
                         name, r["cos_sim"], r["greedy_match"], r["text_match"])

    logger.info("=" * 60)
    logger.info("V2-Lite thorough test complete")


if __name__ == "__main__":
    main()
