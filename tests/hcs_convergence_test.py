#!/usr/bin/env python3
"""HCS convergence test: send 100 diverse prompts and track decode speed over time.

Run against a running server:
    ./dev run qcn   # in one terminal
    python3 tests/hcs_convergence_test.py  # in another

Prints per-prompt decode_tok_s from krasis_timing to see if HCS rebalancing
improves (or degrades) decode speed over many prompts.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

PORT = int(os.environ.get("PORT", 8080))
SERVER = f"http://localhost:{PORT}"
MODEL = os.environ.get("MODEL", "Qwen3-Coder-Next")
NUM_PROMPTS = 100
MAX_TOKENS = 80  # short responses to keep it fast
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "prompts")

INSTRUCTIONS = [
    "Summarize the following passage in 3 sentences:",
    "What are the main themes in this passage? Answer briefly:",
    "Describe the characters mentioned in this passage:",
    "What is the setting and mood of this passage?",
    "Analyze the writing style of this passage briefly:",
    "What happens in this passage? Give a brief plot summary:",
    "What literary devices are used in this passage?",
    "Explain the significance of this passage in the context of the full work:",
    "What emotions does this passage convey? Explain:",
    "Rewrite the key events of this passage as bullet points:",
]


def load_prompts():
    """Load all Gutenberg texts and create 100 diverse prompts."""
    texts = []
    for f in sorted(os.listdir(PROMPTS_DIR)):
        if f.endswith(".txt"):
            with open(os.path.join(PROMPTS_DIR, f)) as fh:
                texts.append(fh.read())

    prompts = []
    for i in range(NUM_PROMPTS):
        book_idx = i % len(texts)
        text = texts[book_idx]
        # Use different starting positions within each book
        chunk_size = 1500  # ~375 tokens
        start = (i * 3001) % max(1, len(text) - chunk_size)
        excerpt = text[start:start + chunk_size]
        instruction = INSTRUCTIONS[i % len(INSTRUCTIONS)]
        prompts.append(f"{instruction}\n\n{excerpt}")

    return prompts


def send_prompt(prompt):
    """Send a prompt and return engine decode_tok_s from krasis_timing."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "stream": True,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    engine_timing = None
    token_count = 0

    with urllib.request.urlopen(req, timeout=120) as resp:
        buffer = b""
        for chunk in iter(lambda: resp.read(1), b""):
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    try:
                        obj = json.loads(line[6:])
                        if "krasis_timing" in obj:
                            engine_timing = obj["krasis_timing"]
                            continue
                        delta = obj["choices"][0].get("delta", {})
                        if delta.get("content", ""):
                            token_count += 1
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

    return engine_timing, token_count


def main():
    # Check server
    try:
        req = urllib.request.Request(f"{SERVER}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                print(f"Server not healthy at {SERVER}")
                sys.exit(1)
    except Exception:
        print(f"Server not running at {SERVER}")
        sys.exit(1)

    prompts = load_prompts()
    print(f"Running {NUM_PROMPTS} diverse prompts against {SERVER}")
    print(f"Max tokens per response: {MAX_TOKENS}")
    print()
    print(f"{'#':<5} {'Decode tok/s':<14} {'Tokens':<8} {'Decode ms':<12} {'Book':<6} {'Instruction'}")
    print("-" * 80)

    books = ["moby", "war", "les_mis", "monte", "quixote", "karamaz"]
    results = []

    for i, prompt in enumerate(prompts):
        try:
            timing, tokens = send_prompt(prompt)
            book = books[i % len(books)]
            instr = INSTRUCTIONS[i % len(INSTRUCTIONS)][:40]

            if timing:
                tok_s = timing.get("decode_tok_s", 0)
                decode_ms = timing.get("decode_time_ms", 0)
                results.append({
                    "prompt_num": i + 1,
                    "decode_tok_s": tok_s,
                    "tokens": tokens,
                    "decode_ms": decode_ms,
                })
                print(f"{i+1:<5} {tok_s:<14.2f} {tokens:<8} {decode_ms:<12.1f} {book:<6} {instr}")
            else:
                print(f"{i+1:<5} {'NO TIMING':<14} {tokens:<8} {'?':<12} {book:<6} {instr}")
        except Exception as e:
            print(f"{i+1:<5} ERROR: {e}")

    # Summary by groups of 10
    if results:
        print()
        print("=== Summary by groups of 10 ===")
        print(f"{'Group':<12} {'Avg tok/s':<12} {'Min':<10} {'Max':<10} {'Count'}")
        print("-" * 55)

        for g in range(0, NUM_PROMPTS, 10):
            group = [r for r in results if g < r["prompt_num"] <= g + 10]
            if group:
                speeds = [r["decode_tok_s"] for r in group]
                label = f"{g+1}-{g+10}"
                print(f"{label:<12} {sum(speeds)/len(speeds):<12.2f} {min(speeds):<10.2f} {max(speeds):<10.2f} {len(group)}")

        all_speeds = [r["decode_tok_s"] for r in results]
        first_10 = [r["decode_tok_s"] for r in results[:10]]
        last_10 = [r["decode_tok_s"] for r in results[-10:]]
        print()
        print(f"Overall avg: {sum(all_speeds)/len(all_speeds):.2f} tok/s")
        print(f"First 10 avg: {sum(first_10)/len(first_10):.2f} tok/s")
        print(f"Last 10 avg: {sum(last_10)/len(last_10):.2f} tok/s")
        delta = sum(last_10)/len(last_10) - sum(first_10)/len(first_10)
        print(f"Delta (last10 - first10): {delta:+.2f} tok/s")


if __name__ == "__main__":
    main()
