#!/usr/bin/env python3
"""Compare benchmark (generate_batch) speeds vs network (Rust HTTP server) speeds.

Loads the same prompts the benchmark uses, sends them over HTTP to the running
server, and measures prefill tok/s and decode tok/s. Then compares with the
benchmark numbers.

Usage:
    # First: start the server with --benchmark flag to get benchmark numbers:
    #   python -m krasis.server --config testconfigs/qcn-4-4.conf --benchmark
    #
    # Then in another terminal, run this:
    python tests/bench_network_vs_benchmark.py [--host localhost] [--port 8012] [--decode-tokens 64]
"""

import argparse
import json
import os
import sys
import time
import requests


def load_prompt_file(filename: str) -> str:
    """Load a prompt file from the krasis prompts directory."""
    prompts_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "python", "krasis", "prompts",
    )
    path = os.path.join(prompts_dir, filename)
    with open(path) as f:
        return f.read().strip()


def send_chat_request(host, port, messages, max_tokens, stream=True, temperature=0.6):
    """Send a chat completion request and measure timing.

    Returns (prefill_tok_s, decode_tok_s, ttft, total_time, num_generated, text)
    """
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": 50,
        "top_p": 0.95,
        "stream": stream,
    }

    if stream:
        return _send_streaming(url, payload)
    else:
        return _send_blocking(url, payload)


def _send_streaming(url, payload):
    """Send streaming request, measure TTFT and decode speed."""
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    num_tokens = 0

    resp = requests.post(url, json=payload, stream=True, timeout=600)
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break

        chunk = json.loads(data)
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content", "")

        if content and ttft is None:
            ttft = time.perf_counter() - t0

        if content:
            chunks.append(content)
            num_tokens += 1

    total_time = time.perf_counter() - t0
    text = "".join(chunks)

    if ttft is None:
        ttft = total_time

    # decode time = total - ttft
    decode_time = total_time - ttft
    decode_tps = (num_tokens - 1) / decode_time if decode_time > 0 and num_tokens > 1 else 0

    return {
        "ttft": ttft,
        "total_time": total_time,
        "decode_time": decode_time,
        "num_tokens": num_tokens,
        "decode_tps": decode_tps,
        "text": text,
    }


def _send_blocking(url, payload):
    """Send blocking request, measure total time."""
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=600)
    total_time = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()

    text = data["choices"][0]["message"]["content"]
    # Can't separate prefill/decode from blocking — just report total
    return {
        "ttft": total_time,  # can't measure separately
        "total_time": total_time,
        "decode_time": total_time,
        "num_tokens": 0,  # unknown from blocking
        "decode_tps": 0,
        "text": text,
    }


def main():
    parser = argparse.ArgumentParser(description="Network vs Benchmark speed comparison")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--decode-tokens", type=int, default=64,
                        help="Number of tokens to generate for decode test")
    parser.add_argument("--prefill-tokens", type=int, default=20000,
                        help="Approximate prompt size for prefill test (chars, ~5 chars/tok)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Network vs Benchmark Speed Comparison")
    print("=" * 60)

    # Check server is up
    try:
        r = requests.get(f"http://{args.host}:{args.port}/health", timeout=5)
        info = r.json()
        print(f"Server: OK (max_context={info.get('max_context_tokens', '?')})")
    except Exception as e:
        print(f"Server not reachable at {args.host}:{args.port}: {e}")
        sys.exit(1)

    # ── Warmup ──
    print("\n-- Warmup (1 short request) --")
    warmup_msg = [{"role": "user", "content": "Say hello in exactly 5 words."}]
    result = send_chat_request(args.host, args.port, warmup_msg, max_tokens=20)
    print(f"  Warmup done: {result['num_tokens']} tokens in {result['total_time']:.1f}s")
    print(f"  Output: {result['text'][:100]}")

    # ── Decode test (short prompt, measure decode tok/s) ──
    print("\n-- Decode Test (3 runs, different prompts) --")
    decode_results = []
    for i in range(1, 4):
        try:
            content = load_prompt_file(f"decode_prompt_{i}")
        except FileNotFoundError:
            content = load_prompt_file("decode_prompt")

        messages = [{"role": "user", "content": content}]
        result = send_chat_request(
            args.host, args.port, messages,
            max_tokens=args.decode_tokens,
        )
        decode_results.append(result)
        print(f"  Run {i}: {result['decode_tps']:.2f} tok/s, "
              f"TTFT={result['ttft']:.2f}s, "
              f"{result['num_tokens']} tokens in {result['decode_time']:.1f}s")
        print(f"    Output: {result['text'][:80]}...")

    avg_decode = sum(r["decode_tps"] for r in decode_results) / len(decode_results)
    print(f"  Average decode: {avg_decode:.2f} tok/s")

    # ── Prefill test (long prompt, measure TTFT -> prefill tok/s) ──
    print("\n-- Prefill Test (3 runs at different lengths) --")
    target_lengths = [20000, 35000, 50000]  # same as benchmark
    prefill_results = []

    for i, target_chars in enumerate(target_lengths):
        file_idx = (i % 6) + 1
        try:
            content = load_prompt_file(f"prefill_prompt_{file_idx}")
        except FileNotFoundError:
            content = load_prompt_file("prefill_prompt_1")

        # Truncate to approximate token count (rough: ~4 chars/token)
        char_limit = target_chars * 5
        if len(content) > char_limit:
            content = content[:char_limit]

        messages = [{"role": "user", "content": content}]

        # Use streaming with max_tokens=1 to isolate prefill
        result = send_chat_request(
            args.host, args.port, messages,
            max_tokens=1,
        )

        # Estimate token count from chars (rough)
        est_tokens = len(content) // 4
        prefill_tps = est_tokens / result["ttft"] if result["ttft"] > 0 else 0

        prefill_results.append({
            "ttft": result["ttft"],
            "est_tokens": est_tokens,
            "prefill_tps": prefill_tps,
        })
        print(f"  Run {i+1}: ~{est_tokens:,} tokens, TTFT={result['ttft']:.2f}s, "
              f"~{prefill_tps:.0f} tok/s")

    best_prefill = max(prefill_results, key=lambda r: r["prefill_tps"])

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Network decode:  {avg_decode:.2f} tok/s (avg of 3 runs)")
    print(f"  Network prefill: ~{best_prefill['prefill_tps']:.0f} tok/s (best run, TTFT={best_prefill['ttft']:.2f}s)")
    print()
    print("  Compare with benchmark output above to see if there's a gap.")
    print("  The server log also prints per-request prefill/decode tok/s")
    print("  which are measured server-side (no network latency).")
    print("=" * 60)


if __name__ == "__main__":
    main()
