#!/usr/bin/env python3
"""Reproduce multi-turn chat failure.

Simulates exactly what krasis chat does: send "hi", capture raw response
including thinking tags, store in history, send "hi" again. Reports full
details of what's sent and received at each step.

Usage:
    ./dev run qwen35 &   # start server first
    python3 tests/repro_multiturn.py [--port 8012] [--turns 5]
"""

import argparse
import http.client
import json
import sys
import time


def stream_chat_raw(host, port, messages, temperature=0.6, max_tokens=16384):
    """Send streaming chat request, return (raw_text, timing, all_chunks)."""
    body = json.dumps({
        "model": "krasis",
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.95,
    }).encode("utf-8")

    conn = http.client.HTTPConnection(host, port, timeout=600)
    try:
        conn.request("POST", "/v1/chat/completions", body, {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        })
        resp = conn.getresponse()
        if resp.status != 200:
            error_body = resp.read().decode(errors="replace")
            return None, None, [f"HTTP {resp.status}: {error_body[:500]}"]

        raw_text = ""
        timing = None
        chunks = []

        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                chunks.append("[DONE]")
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                chunks.append(f"PARSE_ERROR: {payload[:100]}")
                continue

            if "krasis_timing" in obj:
                timing = obj["krasis_timing"]
                chunks.append(f"TIMING: think={timing.get('thinking_tokens',0)} answer={timing.get('answer_tokens',0)}")
                continue

            choices = obj.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            finish = choices[0].get("finish_reason")
            if content:
                raw_text += content
                chunks.append(f"CONTENT: {repr(content)}")
            if finish:
                chunks.append(f"FINISH: {finish}")

        return raw_text, timing, chunks
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Reproduce multi-turn chat failure")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--message", default="hi")
    args = parser.parse_args()

    messages = []

    for turn in range(1, args.turns + 1):
        print(f"\n{'='*60}")
        print(f"TURN {turn}")
        print(f"{'='*60}")

        messages.append({"role": "user", "content": args.message})

        print(f"\nSENDING {len(messages)} messages:")
        for i, m in enumerate(messages):
            content_preview = m["content"][:100]
            if len(m["content"]) > 100:
                content_preview += "..."
            print(f"  [{i}] {m['role']}: {repr(content_preview)}")

        raw_text, timing, chunks = stream_chat_raw(
            args.host, args.port, messages
        )

        if raw_text is None:
            print(f"\nERROR: {chunks}")
            messages.pop()  # remove failed user message
            break

        print(f"\nSSE CHUNKS ({len(chunks)}):")
        for c in chunks:
            print(f"  {c}")

        print(f"\nRAW TEXT ({len(raw_text)} chars): {repr(raw_text[:200])}")

        if timing:
            think_tok = timing.get("thinking_tokens", 0)
            answer_tok = timing.get("answer_tokens", 0)
            total = timing.get("total_generated", 0)
            pp_tok = timing.get("prompt_tokens", 0)
            print(f"\nTIMING: pp={pp_tok}t, think={think_tok}t, answer={answer_tok}t, total={total}t")

            if answer_tok == 0 and think_tok > 0:
                print(f"\n*** FAILURE: {think_tok} thinking tokens but 0 answer tokens! ***")
                print(f"*** Storing broken response to reproduce cascading failure ***")
        else:
            print("\nTIMING: none received")

        # Store in history exactly as chat.py does (raw_text including <think> tags)
        messages.append({"role": "assistant", "content": raw_text})

        print(f"\nSTORED assistant content: {repr(raw_text[:200])}")

    print(f"\n{'='*60}")
    print("FINAL MESSAGE HISTORY:")
    print(f"{'='*60}")
    print(json.dumps(messages, indent=2))


if __name__ == "__main__":
    main()
