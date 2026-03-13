#!/usr/bin/env python3
"""Benchmark decode speed at multiple context lengths.

Uses streaming API to measure TTFT (time-to-first-token) and ITL
(inter-token-latency) separately.

Usage:
    python3 bench_decode.py [--output results.json]
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

PORT = int(os.environ.get("PORT", 8080))
SERVER = f"http://localhost:{PORT}"
MODEL = os.environ.get("MODEL", "Qwen3-235B-A22B")

# Test configs: (label, approximate input chars, max_output_tokens)
TESTS = [
    ("short",     200,   80),
    ("medium",    800,   80),
    ("long",     2000,   80),
    ("1k",       4000,   80),
]


def _load_gutenberg_text() -> str:
    """Load the first available Gutenberg prompt from benchmarks/prompts/."""
    prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    prompt_files = sorted(f for f in os.listdir(prompts_dir) if f.endswith(".txt"))
    if not prompt_files:
        raise FileNotFoundError(f"No prompt files in {prompts_dir}")
    with open(os.path.join(prompts_dir, prompt_files[0])) as f:
        return f.read()


_GUTENBERG_TEXT = None


def check_server():
    try:
        req = urllib.request.Request(f"{SERVER}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def make_prompt(target_chars):
    global _GUTENBERG_TEXT
    if _GUTENBERG_TEXT is None:
        _GUTENBERG_TEXT = _load_gutenberg_text()
    excerpt = _GUTENBERG_TEXT[:target_chars]
    return f"Please summarize the following text in exactly 3 sentences:\n\n{excerpt}\n\nSummary:"


def run_streaming_test(prompt, max_tokens, temperature=0.7):
    """Run a streaming test and measure TTFT + ITL separately."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.perf_counter()
    t_first_token = None
    token_times = []
    token_count = 0
    engine_timing = None  # krasis_timing from SSE stream

    with urllib.request.urlopen(req, timeout=3600) as resp:
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
                        # Check for engine timing chunk (no choices with content)
                        if "krasis_timing" in obj:
                            engine_timing = obj["krasis_timing"]
                            continue
                        delta = obj["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            now = time.perf_counter()
                            if t_first_token is None:
                                t_first_token = now
                            else:
                                token_times.append(now)
                            token_count += 1
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

    t_end = time.perf_counter()

    ttft = (t_first_token - t_start) if t_first_token else 0
    total = t_end - t_start

    # Compute average ITL from second token onwards
    if len(token_times) >= 2:
        decode_duration = token_times[-1] - t_first_token
        decode_tokens = len(token_times)  # excludes first token
        avg_itl = decode_duration / decode_tokens
        decode_tok_s = decode_tokens / decode_duration
    elif t_first_token:
        decode_duration = t_end - t_first_token
        decode_tokens = max(token_count - 1, 1)
        avg_itl = decode_duration / decode_tokens
        decode_tok_s = decode_tokens / decode_duration
    else:
        avg_itl = 0
        decode_tok_s = 0

    result = {
        "tokens": token_count,
        "ttft_s": round(ttft, 2),
        "decode_tok_s": round(decode_tok_s, 2),
        "avg_itl_ms": round(avg_itl * 1000, 1),
        "total_s": round(total, 2),
    }

    # Add engine-reported timing if available
    if engine_timing:
        result["engine"] = {
            "decode_tok_s": round(engine_timing.get("decode_tok_s", 0), 2),
            "decode_ms": round(engine_timing.get("decode_time_ms", 0), 1),
            "overhead_ms": round(engine_timing.get("overhead_ms", 0), 1),
        }
        oh = engine_timing.get("overhead", {})
        if oh:
            result["engine"]["overhead_detail"] = {
                "parse_ms": round(oh.get("parse_ms", 0), 1),
                "evict_ms": round(oh.get("evict_ms", 0), 1),
                "prefill_ms": round(oh.get("prefill_ms", 0), 0),
                "reload_ms": round(oh.get("reload_ms", 0), 0),
            }

    return result


def main():
    if not check_server():
        print(f"ERROR: Server not running at {SERVER}")
        sys.exit(1)

    print(f"Benchmarking {MODEL} at {SERVER}")
    print()
    print(f"{'Test':<10} {'Engine tok/s':<14} {'HTTP tok/s':<12} {'Overhead(ms)':<14} {'Prefill(ms)':<13} {'Reload(ms)':<12} {'Tokens':<8} {'Total(s)'}")
    print("-" * 100)

    results = []

    # Warmup (two requests to ensure all caches are warm)
    print("Warming up...", end="", flush=True)
    run_streaming_test(make_prompt(50), 20)
    run_streaming_test(make_prompt(100), 20)
    print(" done\n")

    for label, target_tokens, max_output in TESTS:
        prompt = make_prompt(target_tokens)
        try:
            r = run_streaming_test(prompt, max_output)
            r["label"] = label
            r["target_input"] = target_tokens
            results.append(r)

            eng = r.get("engine", {})
            eng_tok_s = eng.get("decode_tok_s", 0)
            overhead = eng.get("overhead_ms", 0)
            detail = eng.get("overhead_detail", {})
            prefill = detail.get("prefill_ms", 0)
            reload = detail.get("reload_ms", 0)

            print(
                f"{label:<10} {eng_tok_s:<14} {r['decode_tok_s']:<12} "
                f"{overhead:<14.0f} {prefill:<13.0f} {reload:<12.0f} "
                f"{r['tokens']:<8} {r['total_s']}"
            )
        except Exception as e:
            print(f"{label:<10} ERROR: {e}")

    print()

    if results:
        eng_rates = [r["engine"]["decode_tok_s"] for r in results if r.get("engine", {}).get("decode_tok_s", 0) > 0]
        http_rates = [r["decode_tok_s"] for r in results if r["decode_tok_s"] > 0]
        if eng_rates:
            print(f"Engine decode:  min={min(eng_rates):.1f}, max={max(eng_rates):.1f}, avg={sum(eng_rates)/len(eng_rates):.1f} tok/s")
        if http_rates:
            print(f"HTTP decode:    min={min(http_rates):.1f}, max={max(http_rates):.1f}, avg={sum(http_rates)/len(http_rates):.1f} tok/s")
        overheads = [r["engine"]["overhead_ms"] for r in results if r.get("engine", {}).get("overhead_ms", 0) > 0]
        if overheads:
            print(f"Overhead:       min={min(overheads):.0f}, max={max(overheads):.0f}, avg={sum(overheads)/len(overheads):.0f} ms")

    output_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--output" else None
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
