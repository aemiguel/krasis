#!/usr/bin/env python3
"""Compare multi-turn behavior between Krasis and llama.cpp.
Run against any OpenAI-compatible server on specified port."""
import sys
import json
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
URL = f"http://localhost:{PORT}/v1/chat/completions"

CONVERSATIONS = [
    {
        "name": "Paris (5 turns)",
        "prompts": [
            "What is the capital of France?",
            "What is the population of that city?",
            "Name three famous landmarks there",
            "Which of those landmarks is the oldest?",
            "Tell me more about its history",
        ],
    },
    {
        "name": "Math (6 turns)",
        "prompts": [
            "What is 15 multiplied by 7?",
            "Add 50 to that result",
            "Now divide everything by 5",
            "Is the result a prime number?",
            "What are the prime numbers between 1 and 50?",
            "How many of those primes are also Fibonacci numbers?",
        ],
    },
    {
        "name": "Japanese (5 turns)",
        "prompts": [
            "Hi, who are you?",
            "What languages can you speak?",
            "Say hello in Japanese",
            "Now say goodbye in Japanese",
            "Translate 'the weather is nice today' into Japanese",
        ],
    },
]


def send_request(messages, max_tokens=512):
    data = json.dumps({
        "model": "test",
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            content = result["choices"][0]["message"]["content"]
            return content
    except Exception as e:
        return f"ERROR: {e}"


def extract_answer(content):
    """Strip thinking block and return visible answer."""
    if "</think>" in content:
        return content.split("</think>", 1)[1].strip()
    elif "<think>" in content:
        return ""  # Unclosed thinking = no answer
    return content.strip()


def main():
    print(f"Testing multi-turn conversations against {URL}")
    print("=" * 60)

    failures = 0
    total = 0

    for conv in CONVERSATIONS:
        print(f"\n{'='*60}")
        print(f"  Conversation: {conv['name']}")
        print(f"{'='*60}")

        messages = []
        for i, prompt in enumerate(conv["prompts"], 1):
            total += 1
            messages.append({"role": "user", "content": prompt})

            print(f"\n  Turn {i}/{len(conv['prompts'])}: {prompt}")
            content = send_request(messages, max_tokens=512)

            if content.startswith("ERROR"):
                print(f"  >>> {content}")
                failures += 1
                messages.pop()
                continue

            answer = extract_answer(content)
            # Show first 150 chars of answer
            preview = answer[:150].replace("\n", " ")
            if len(answer) > 150:
                preview += "..."

            think_len = 0
            if "</think>" in content:
                think_len = len(content.split("</think>")[0])

            status = "OK" if len(answer) > 5 else "DEGENERATE"
            if status == "DEGENERATE":
                failures += 1

            print(f"  >>> [{status}] think={think_len}chars answer={len(answer)}chars")
            print(f"  >>> {preview}")

            messages.append({"role": "assistant", "content": content})

    print(f"\n{'='*60}")
    print(f"  Results: {total - failures}/{total} passed, {failures} failures")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
