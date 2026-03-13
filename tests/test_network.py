#!/usr/bin/env python3
"""Network validation tests for Krasis server (tests 4 & 5 from testing-plan.md).

Sends prompts to a running Krasis server over HTTP and validates responses.
Tests the full server stack: HTTP, tokenization, scheduling, prefill, decode,
streaming SSE, and response assembly.

Usage:
    python tests/test_network.py                          # auto-discover
    python tests/test_network.py --port 8080              # specific port
    python tests/test_network.py --port 8080 --large      # include large-prompt tests
    python tests/test_network.py --port 8080 --quick      # only known-answer tests
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════
# ANSI formatting
# ═══════════════════════════════════════════════════════════════════
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
NC = "\033[0m"


def _pass(msg: str) -> str:
    return f"{GREEN}PASS{NC} {msg}"


def _fail(msg: str) -> str:
    return f"{RED}FAIL{NC} {msg}"


def _warn(msg: str) -> str:
    return f"{YELLOW}WARN{NC} {msg}"


# ═══════════════════════════════════════════════════════════════════
# HTTP helpers
# ═══════════════════════════════════════════════════════════════════

def send_chat_request(
    base_url: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 128,
    temperature: float = 0.1,
    stream: bool = False,
    timeout: int = 120,
) -> Tuple[int, Any]:
    """Send a chat completion request. Returns (status_code, response_body)."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "test",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        status = resp.status
        if stream:
            # Return raw SSE chunks
            body = resp.read().decode("utf-8")
            return status, body
        else:
            body = json.loads(resp.read().decode("utf-8"))
            return status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        return 0, str(e.reason)
    except Exception as e:
        return 0, str(e)


def parse_sse_content(sse_body: str) -> Tuple[str, Optional[str]]:
    """Parse SSE response body into (full_text, finish_reason)."""
    text_parts = []
    finish_reason = None
    for line in sse_body.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    text_parts.append(content)
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr
        except json.JSONDecodeError:
            pass
    return "".join(text_parts), finish_reason


def is_garbage(text: str) -> bool:
    """Heuristic check for garbage output."""
    if not text or not text.strip():
        return True
    # Repeated single character (e.g., "aaaaaaaaaa")
    stripped = text.strip()
    if len(stripped) > 10 and len(set(stripped)) <= 2:
        return True
    # All whitespace
    if not stripped:
        return True
    # Contains raw unicode replacement chars
    if stripped.count("\ufffd") > len(stripped) // 4:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════
# Test definitions
# ═══════════════════════════════════════════════════════════════════

class TestResult:
    def __init__(self, name: str, passed: bool, detail: str = "",
                 response_text: str = "", elapsed: float = 0.0):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.response_text = response_text
        self.elapsed = elapsed


def check_health(base_url: str) -> bool:
    """Check if server is healthy."""
    try:
        resp = urllib.request.urlopen(f"{base_url}/health", timeout=5)
        data = json.loads(resp.read())
        return data.get("status") == "ok"
    except Exception:
        return False


def test_known_answer(
    base_url: str, name: str, prompt: str, expected_substr: str,
    stream: bool = False, max_tokens: int = 64, timeout: int = 60,
) -> TestResult:
    """Test with a prompt that has a known expected substring in the answer."""
    messages = [{"role": "user", "content": prompt}]
    t0 = time.time()
    status, body = send_chat_request(
        base_url, messages, max_tokens=max_tokens,
        temperature=0.1, stream=stream, timeout=timeout,
    )
    elapsed = time.time() - t0

    if status == 0:
        return TestResult(name, False, f"Connection error: {body}", elapsed=elapsed)
    if status != 200:
        return TestResult(name, False, f"HTTP {status}", elapsed=elapsed)

    if stream:
        text, finish = parse_sse_content(body)
    else:
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        finish = body.get("choices", [{}])[0].get("finish_reason")

    if is_garbage(text):
        return TestResult(name, False, f"Garbage output: {text[:100]!r}",
                          response_text=text, elapsed=elapsed)

    if expected_substr.lower() not in text.lower():
        return TestResult(
            name, False,
            f"Expected '{expected_substr}' not found in: {text[:200]!r}",
            response_text=text, elapsed=elapsed,
        )

    return TestResult(name, True, f"OK ({elapsed:.1f}s)",
                      response_text=text, elapsed=elapsed)


def test_coherent_response(
    base_url: str, name: str, messages: List[Dict[str, str]],
    stream: bool = False, max_tokens: int = 128, timeout: int = 120,
    min_words: int = 3,
) -> TestResult:
    """Test that a prompt produces a coherent (non-garbage) response."""
    t0 = time.time()
    status, body = send_chat_request(
        base_url, messages, max_tokens=max_tokens,
        temperature=0.3, stream=stream, timeout=timeout,
    )
    elapsed = time.time() - t0

    if status == 0:
        return TestResult(name, False, f"Connection error: {body}", elapsed=elapsed)
    if status != 200:
        return TestResult(name, False, f"HTTP {status}", elapsed=elapsed)

    if stream:
        text, finish = parse_sse_content(body)
    else:
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        finish = body.get("choices", [{}])[0].get("finish_reason")

    if is_garbage(text):
        return TestResult(name, False, f"Garbage output: {text[:100]!r}",
                          response_text=text, elapsed=elapsed)

    word_count = len(text.split())
    if word_count < min_words:
        return TestResult(name, False,
                          f"Too short ({word_count} words): {text[:100]!r}",
                          response_text=text, elapsed=elapsed)

    if finish not in ("stop", "length"):
        return TestResult(name, False,
                          f"Bad finish_reason: {finish!r}",
                          response_text=text, elapsed=elapsed)

    return TestResult(name, True, f"OK ({elapsed:.1f}s, {word_count} words)",
                      response_text=text, elapsed=elapsed)


# ═══════════════════════════════════════════════════════════════════
# Test 4: Multi-prompt validation
# ═══════════════════════════════════════════════════════════════════

def run_multi_prompt_tests(base_url: str) -> List[TestResult]:
    """Test 4: diverse prompts over the network with validation."""
    results = []

    print(f"\n{BOLD}{CYAN}Test 4: Network Multi-Prompt Validation{NC}")
    print("=" * 60)

    # 1. Factual Q&A with known answer
    r = test_known_answer(base_url, "factual_qa",
                          "What is the capital of France? Answer in one word.",
                          "Paris")
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    # 2. Math with verifiable answer
    r = test_known_answer(base_url, "math_simple",
                          "What is 7 * 8? Answer with just the number.",
                          "56")
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    # 3. Code generation (512 tokens: thinking models need room for CoT + answer)
    r = test_known_answer(base_url, "code_gen",
                          "Write a Python function called fibonacci that returns the nth fibonacci number. Only output the code.",
                          "def", max_tokens=512)
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    # 4. Multi-turn conversation (512 tokens: thinking models need room for CoT + answer)
    r = test_coherent_response(base_url, "multi_turn", [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What is my name?"},
    ], max_tokens=512)
    results.append(r)
    if r.passed and "alice" not in r.response_text.lower():
        r.passed = False
        r.detail = f"Didn't recall name: {r.response_text[:100]!r}"
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    # 5. Short prompt
    r = test_coherent_response(base_url, "short_prompt", [
        {"role": "user", "content": "Hello"},
    ], max_tokens=32, min_words=1)
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    # 6. Streaming mode - factual
    r = test_known_answer(base_url, "streaming_factual",
                          "What is the capital of Japan? Answer in one word.",
                          "Tokyo", stream=True)
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    # 7. Streaming mode - code (512 tokens: thinking models need room for CoT + answer)
    r = test_known_answer(base_url, "streaming_code",
                          "Write a Python hello world one-liner.",
                          "print", stream=True, max_tokens=512)
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    # 8. JSON output
    r = test_known_answer(base_url, "json_output",
                          'Output valid JSON with fields "name" and "age". Example: {"name": "Bob", "age": 30}. Only output the JSON.',
                          '"name"', max_tokens=64)
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    # 9. Non-streaming blocking mode (512 tokens: thinking models need room for CoT + answer)
    r = test_known_answer(base_url, "blocking_math",
                          "What is 15 + 27? Answer with just the number.",
                          "42", stream=False, max_tokens=512)
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    return results


# ═══════════════════════════════════════════════════════════════════
# Test 5: Large prompt validation
# ═══════════════════════════════════════════════════════════════════

def _load_gutenberg_prompt(target_chars: int, prompt_index: int = 0) -> str:
    """Load a Gutenberg prompt from benchmarks/prompts/ and truncate to target size.

    Uses real literary text (Moby Dick, War and Peace, etc.) instead of
    generated garbage. These are consistent, unique, and representative of
    real-world content — essential for meaningful test results.
    """
    prompts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmarks", "prompts",
    )
    prompt_files = sorted(f for f in os.listdir(prompts_dir) if f.endswith(".txt"))
    if not prompt_files:
        raise FileNotFoundError(f"No prompt files found in {prompts_dir}")

    path = os.path.join(prompts_dir, prompt_files[prompt_index % len(prompt_files)])
    with open(path) as f:
        text = f.read()

    if len(text) > target_chars:
        text = text[:target_chars]
    return text


def run_large_prompt_tests(base_url: str) -> List[TestResult]:
    """Test 5: large prompts over the network.

    Uses real Gutenberg literary texts (Moby Dick, War and Peace, etc.)
    from benchmarks/prompts/. These are consistent across runs, unique,
    and representative of real-world content.
    """
    results = []

    print(f"\n{BOLD}{CYAN}Test 5: Network Large-Prompt Validation{NC}")
    print("=" * 60)

    # Each test uses a different book for content diversity.
    # ~3.5 chars per token is the rough ratio for English literary text.
    test_configs = [
        ("large_2k",   7_000,   0, "Summarize the above text in one paragraph.",             128, 120),
        ("large_10k",  35_000,  1, "What are the main themes in the text above?",            128, 300),
        ("large_25k",  87_500,  2, "Summarize the key events described in the text above.",  128, 600),
        ("large_100k", 350_000, 3, "What is the overall narrative arc of the text above?",   128, 1200),
    ]

    for name, target_chars, prompt_idx, question, max_tokens, timeout in test_configs:
        approx_tokens = target_chars // 4
        print(f"  Loading ~{approx_tokens // 1000}K token prompt...", end="", flush=True)
        text = _load_gutenberg_prompt(target_chars, prompt_index=prompt_idx)
        prompt = f"{text}\n\n{question}"
        print(f" done ({len(prompt)} chars)")

        r = test_coherent_response(base_url, name, [
            {"role": "user", "content": prompt},
        ], max_tokens=max_tokens, timeout=timeout, stream=True, min_words=3)
        results.append(r)
        print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    return results


# ═══════════════════════════════════════════════════════════════════
# Test 6: Multi-turn sequential conversation
# ═══════════════════════════════════════════════════════════════════

def run_multi_turn_tests(base_url: str) -> List[TestResult]:
    """Test 6: multi-turn conversation with separate sequential HTTP requests.

    Simulates a real chat where each request includes growing conversation
    history. Each request goes through a full prefill -> KV cache copy -> decode
    cycle, exercising _copy_kv_cache and _copy_recurrent_state_from_gpu
    repeatedly. This catches inference_mode bugs on inplace ops on KV tensors.
    """
    results = []

    print(f"\n{BOLD}{CYAN}Test 6: Multi-Turn Sequential Conversation{NC}")
    print("=" * 60)

    # ── Turn 1: Establish facts ──
    messages = [
        {"role": "user", "content": (
            "I need you to remember three facts for our conversation: "
            "1) My dog's name is Biscuit. "
            "2) I live in Wellington, New Zealand. "
            "3) My favorite number is 42. "
            "Acknowledge each fact."
        )},
    ]
    print(f"  Turn 1: establishing facts...", end="", flush=True)
    t0 = time.time()
    status, body = send_chat_request(
        base_url, messages, max_tokens=128, temperature=0.1,
        stream=False, timeout=120,
    )
    elapsed = time.time() - t0

    if status != 200:
        r = TestResult("multi_turn_t1", False, f"HTTP {status}", elapsed=elapsed)
        results.append(r)
        print(f" {_fail(r.detail)}")
        return results  # can't continue

    t1_text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    r = TestResult("multi_turn_t1", not is_garbage(t1_text),
                   f"OK ({elapsed:.1f}s)" if not is_garbage(t1_text) else f"Garbage: {t1_text[:100]!r}",
                   response_text=t1_text, elapsed=elapsed)
    results.append(r)
    print(f" {_pass(r.detail) if r.passed else _fail(r.detail)}")
    if not r.passed:
        return results

    # ── Turn 2: Ask about one fact (tests KV cache copy from turn 1 prefill) ──
    messages.append({"role": "assistant", "content": t1_text})
    messages.append({"role": "user", "content": "What is my dog's name?"})

    print(f"  Turn 2: recalling dog's name...", end="", flush=True)
    t0 = time.time()
    status, body = send_chat_request(
        base_url, messages, max_tokens=256, temperature=0.1,
        stream=False, timeout=120,
    )
    elapsed = time.time() - t0

    if status != 200:
        r = TestResult("multi_turn_t2_dog", False, f"HTTP {status}", elapsed=elapsed)
        results.append(r)
        print(f" {_fail(r.detail)}")
        return results

    t2_text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    t2_pass = not is_garbage(t2_text) and "biscuit" in t2_text.lower()
    detail = f"OK ({elapsed:.1f}s)" if t2_pass else f"Expected 'Biscuit': {t2_text[:100]!r}"
    r = TestResult("multi_turn_t2_dog", t2_pass, detail,
                   response_text=t2_text, elapsed=elapsed)
    results.append(r)
    print(f" {_pass(r.detail) if r.passed else _fail(r.detail)}")

    # ── Turn 3: Ask about another fact (tests repeated KV copy cycles) ──
    messages.append({"role": "assistant", "content": t2_text})
    messages.append({"role": "user", "content": "Where do I live?"})

    print(f"  Turn 3: recalling city...", end="", flush=True)
    t0 = time.time()
    status, body = send_chat_request(
        base_url, messages, max_tokens=256, temperature=0.1,
        stream=False, timeout=120,
    )
    elapsed = time.time() - t0

    if status != 200:
        r = TestResult("multi_turn_t3_city", False, f"HTTP {status}", elapsed=elapsed)
        results.append(r)
        print(f" {_fail(r.detail)}")
        return results

    t3_text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    t3_pass = not is_garbage(t3_text) and "wellington" in t3_text.lower()
    detail = f"OK ({elapsed:.1f}s)" if t3_pass else f"Expected 'Wellington': {t3_text[:100]!r}"
    r = TestResult("multi_turn_t3_city", t3_pass, detail,
                   response_text=t3_text, elapsed=elapsed)
    results.append(r)
    print(f" {_pass(r.detail) if r.passed else _fail(r.detail)}")

    # ── Turn 4: Ask about the third fact via streaming (tests stream + KV copy) ──
    messages.append({"role": "assistant", "content": t3_text})
    messages.append({"role": "user", "content": "What is my favorite number?"})

    print(f"  Turn 4: recalling number (streaming)...", end="", flush=True)
    t0 = time.time()
    status, body = send_chat_request(
        base_url, messages, max_tokens=256, temperature=0.1,
        stream=True, timeout=120,
    )
    elapsed = time.time() - t0

    if status != 200:
        r = TestResult("multi_turn_t4_number", False, f"HTTP {status}", elapsed=elapsed)
        results.append(r)
        print(f" {_fail(r.detail)}")
        return results

    t4_text, _ = parse_sse_content(body)
    t4_pass = not is_garbage(t4_text) and "42" in t4_text
    detail = f"OK ({elapsed:.1f}s)" if t4_pass else f"Expected '42': {t4_text[:100]!r}"
    r = TestResult("multi_turn_t4_number", t4_pass, detail,
                   response_text=t4_text, elapsed=elapsed)
    results.append(r)
    print(f" {_pass(r.detail) if r.passed else _fail(r.detail)}")

    # ── Turn 5: All three facts at once (verifies full context retention) ──
    messages.append({"role": "assistant", "content": t4_text})
    messages.append({"role": "user", "content": (
        "Repeat all three facts I told you at the start: "
        "my dog's name, where I live, and my favorite number."
    )})

    print(f"  Turn 5: recalling all three facts...", end="", flush=True)
    t0 = time.time()
    status, body = send_chat_request(
        base_url, messages, max_tokens=128, temperature=0.1,
        stream=False, timeout=120,
    )
    elapsed = time.time() - t0

    if status != 200:
        r = TestResult("multi_turn_t5_all", False, f"HTTP {status}", elapsed=elapsed)
        results.append(r)
        print(f" {_fail(r.detail)}")
        return results

    t5_text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    has_biscuit = "biscuit" in t5_text.lower()
    has_wellington = "wellington" in t5_text.lower()
    has_42 = "42" in t5_text
    t5_pass = not is_garbage(t5_text) and has_biscuit and has_wellington and has_42
    missing = []
    if not has_biscuit:
        missing.append("Biscuit")
    if not has_wellington:
        missing.append("Wellington")
    if not has_42:
        missing.append("42")
    detail = f"OK ({elapsed:.1f}s)" if t5_pass else f"Missing: {', '.join(missing)} in: {t5_text[:200]!r}"
    r = TestResult("multi_turn_t5_all", t5_pass, detail,
                   response_text=t5_text, elapsed=elapsed)
    results.append(r)
    print(f" {_pass(r.detail) if r.passed else _fail(r.detail)}")

    return results


# ═══════════════════════════════════════════════════════════════════
# Server discovery
# ═══════════════════════════════════════════════════════════════════

def discover_server() -> Optional[str]:
    """Find a running Krasis server from registry files."""
    import os
    from pathlib import Path

    registry_dir = Path.home() / ".krasis" / "servers"
    if not registry_dir.exists():
        return None

    for f in registry_dir.glob("*.json"):
        try:
            entry = json.loads(f.read_text())
            pid = entry.get("pid")
            # Check if process is still alive
            if pid:
                try:
                    os.kill(pid, 0)
                except OSError:
                    f.unlink(missing_ok=True)
                    continue
            port = entry.get("port", 8080)
            host = entry.get("host", "localhost")
            if host == "0.0.0.0":
                host = "localhost"
            url = f"http://{host}:{port}"
            if check_health(url):
                return url
        except Exception:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Krasis network validation tests")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--url", default=None, help="Full server URL")
    parser.add_argument("--large", action="store_true",
                        help="Include large-prompt tests (10K/25K tokens)")
    parser.add_argument("--quick", action="store_true",
                        help="Only run known-answer tests (fastest)")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-request timeout in seconds")
    args = parser.parse_args()

    # Determine server URL
    if args.url:
        base_url = args.url.rstrip("/")
    elif args.port:
        base_url = f"http://{args.host}:{args.port}"
    else:
        print("Discovering Krasis server...", end="", flush=True)
        base_url = discover_server()
        if base_url is None:
            print(f" {RED}not found{NC}")
            print("No running Krasis server found. Start one or use --port/--url.")
            sys.exit(1)
        print(f" found: {base_url}")

    # Health check
    print(f"Checking server health at {base_url}...", end="", flush=True)
    if not check_health(base_url):
        print(f" {RED}FAILED{NC}")
        print("Server not responding. Is it running and loaded?")
        sys.exit(1)
    print(f" {GREEN}OK{NC}")

    all_results = []
    t_start = time.time()

    # Warmup: first request to a cold server is slow (CUDA graph compilation, etc.)
    print(f"Sending warmup request...", end="", flush=True)
    t_w = time.time()
    status, body = send_chat_request(
        base_url,
        [{"role": "user", "content": "Say hi."}],
        max_tokens=8, temperature=0.1, stream=True, timeout=300,
    )
    warmup_ok = status == 200
    warmup_elapsed = time.time() - t_w
    print(f" {'done' if warmup_ok else 'FAILED'} ({warmup_elapsed:.1f}s)")
    if warmup_ok:
        warmup_text, _ = parse_sse_content(body)
        print(f"  {DIM}>>> {warmup_text[:200]}{NC}")
    if not warmup_ok:
        print(f"  Warmup failed: status={status}, body={body}")
        print("  Server may be stuck. Aborting tests.")
        sys.exit(1)

    # Test 4: Multi-prompt
    results_4 = run_multi_prompt_tests(base_url)
    all_results.extend(results_4)

    # Test 5: Large prompts (optional)
    if args.large and not args.quick:
        results_5 = run_large_prompt_tests(base_url)
        all_results.extend(results_5)

    # Test 6: Multi-turn sequential conversation (always runs unless --quick)
    if not args.quick:
        results_6 = run_multi_turn_tests(base_url)
        all_results.extend(results_6)

    # Summary
    total_time = time.time() - t_start
    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)
    total = len(all_results)

    print(f"\n{'=' * 60}")
    print(f"{BOLD}SUMMARY{NC}: {passed}/{total} passed, {failed} failed, {total_time:.1f}s total")

    # Show ALL response texts for manual review
    print(f"\n{BOLD}Response texts (review for coherence):{NC}")
    for r in all_results:
        status = f"{GREEN}PASS{NC}" if r.passed else f"{RED}FAIL{NC}"
        print(f"\n  [{status}] {r.name}: {r.detail}")
        if r.response_text:
            # Show full text (truncated at 500 chars for readability)
            preview = r.response_text[:500].replace("\n", "\\n")
            print(f"  {DIM}>>> {preview}{NC}")
        else:
            print(f"  {DIM}>>> (no response text captured){NC}")

    if failed > 0:
        print(f"\n{RED}Failed tests:{NC}")
        for r in all_results:
            if not r.passed:
                print(f"  {r.name}: {r.detail}")

    print()
    if failed == 0:
        print(f"{GREEN}{BOLD}ALL TESTS PASSED{NC}")
    else:
        print(f"{RED}{BOLD}{failed} TEST(S) FAILED{NC}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
