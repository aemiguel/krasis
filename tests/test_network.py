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

    # 3. Code generation
    r = test_known_answer(base_url, "code_gen",
                          "Write a Python function called fibonacci that returns the nth fibonacci number. Only output the code.",
                          "def", max_tokens=256)
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    # 4. Multi-turn conversation
    r = test_coherent_response(base_url, "multi_turn", [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What is my name?"},
    ], max_tokens=32)
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

    # 7. Streaming mode - code
    r = test_known_answer(base_url, "streaming_code",
                          "Write a Python hello world one-liner.",
                          "print", stream=True, max_tokens=64)
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    # 8. JSON output
    r = test_known_answer(base_url, "json_output",
                          'Output valid JSON with fields "name" and "age". Example: {"name": "Bob", "age": 30}. Only output the JSON.',
                          '"name"', max_tokens=64)
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    # 9. Non-streaming blocking mode
    r = test_known_answer(base_url, "blocking_math",
                          "What is 15 + 27? Answer with just the number.",
                          "42", stream=False)
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    return results


# ═══════════════════════════════════════════════════════════════════
# Test 5: Large prompt validation
# ═══════════════════════════════════════════════════════════════════

def _build_large_text(target_tokens: int) -> str:
    """Build a large text block targeting approximately target_tokens tokens.

    Uses repetitive but varied text to hit the target.
    Rough estimate: 1 token ~ 3.5 chars for English text.
    """
    sections = [
        "Section {n}: The principles of distributed systems involve consensus algorithms "
        "such as Paxos and Raft, which ensure that multiple nodes in a network can agree "
        "on a single value even in the presence of failures. These algorithms are critical "
        "for building reliable database systems, distributed file systems, and cloud "
        "infrastructure. The CAP theorem states that it is impossible for a distributed "
        "system to simultaneously provide all three of consistency, availability, and "
        "partition tolerance. In practice, systems must make trade-offs between these "
        "properties based on their specific requirements and use cases. ",

        "Section {n}: Modern compiler optimization techniques include dead code elimination, "
        "constant folding, loop unrolling, register allocation, and instruction scheduling. "
        "These optimizations work at different levels of the compilation pipeline, from "
        "high-level IR transformations to low-level machine code generation. Profile-guided "
        "optimization (PGO) uses runtime execution data to make better optimization decisions. "
        "Link-time optimization (LTO) enables optimizations across compilation unit boundaries. ",

        "Section {n}: Neural network architectures have evolved from simple perceptrons to "
        "complex transformer models with billions of parameters. The attention mechanism, "
        "introduced in the transformer architecture, allows models to focus on relevant parts "
        "of the input when producing each output element. Key innovations include multi-head "
        "attention, positional encoding, layer normalization, and residual connections. "
        "Modern training techniques include mixed-precision training, gradient accumulation, "
        "distributed data parallelism, and model parallelism across multiple devices. ",

        "Section {n}: Operating system memory management involves virtual memory, page tables, "
        "TLBs, and various page replacement algorithms including LRU, FIFO, and clock algorithms. "
        "Modern operating systems use demand paging to load only needed pages into physical memory, "
        "with the page fault handler managing the loading of pages from disk or swap space. "
        "Memory-mapped files allow processes to access file contents as if they were in memory, "
        "while copy-on-write semantics reduce memory usage for forked processes. ",

        "Section {n}: Cryptographic systems rely on mathematical hardness assumptions such as "
        "the difficulty of factoring large integers (RSA) or computing discrete logarithms "
        "(Diffie-Hellman, elliptic curves). Symmetric encryption algorithms like AES provide "
        "fast bulk encryption, while asymmetric algorithms enable key exchange and digital "
        "signatures without pre-shared secrets. Hash functions like SHA-256 provide collision "
        "resistance for data integrity verification and proof-of-work systems. ",
    ]

    target_chars = int(target_tokens * 3.5)
    text_parts = []
    total_chars = 0
    n = 1
    while total_chars < target_chars:
        for section_template in sections:
            part = section_template.format(n=n)
            text_parts.append(part)
            total_chars += len(part)
            n += 1
            if total_chars >= target_chars:
                break

    return "".join(text_parts)


def run_large_prompt_tests(base_url: str) -> List[TestResult]:
    """Test 5: large prompts over the network."""
    results = []

    print(f"\n{BOLD}{CYAN}Test 5: Network Large-Prompt Validation{NC}")
    print("=" * 60)

    # Test 5a: ~2K tokens (warmup / sanity)
    print(f"  Building ~2K token prompt...", end="", flush=True)
    text_2k = _build_large_text(2000)
    prompt_2k = f"{text_2k}\n\nSummarize the above text in exactly 2 sentences."
    print(f" done ({len(prompt_2k)} chars)")

    r = test_coherent_response(base_url, "large_2k", [
        {"role": "user", "content": prompt_2k},
    ], max_tokens=128, timeout=120, stream=True, min_words=5)
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    # Test 5b: ~10K tokens
    print(f"  Building ~10K token prompt...", end="", flush=True)
    text_10k = _build_large_text(10000)
    prompt_10k = f"{text_10k}\n\nWhat topic was discussed in Section 15?"
    print(f" done ({len(prompt_10k)} chars)")

    r = test_coherent_response(base_url, "large_10k", [
        {"role": "user", "content": prompt_10k},
    ], max_tokens=128, timeout=300, stream=True, min_words=5)
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    # Test 5c: ~25K tokens
    print(f"  Building ~25K token prompt...", end="", flush=True)
    text_25k = _build_large_text(25000)
    prompt_25k = f"{text_25k}\n\nWhat was the main theme of Section 40? Answer briefly."
    print(f" done ({len(prompt_25k)} chars)")

    r = test_coherent_response(base_url, "large_25k", [
        {"role": "user", "content": prompt_25k},
    ], max_tokens=128, timeout=600, stream=True, min_words=3)
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

    # Test 5d: ~100K tokens
    print(f"  Building ~100K token prompt...", end="", flush=True)
    text_100k = _build_large_text(100000)
    prompt_100k = f"{text_100k}\n\nWhat was the main theme of Section 100? Answer briefly."
    print(f" done ({len(prompt_100k)} chars)")

    r = test_coherent_response(base_url, "large_100k", [
        {"role": "user", "content": prompt_100k},
    ], max_tokens=128, timeout=1200, stream=True, min_words=3)
    results.append(r)
    print(f"  {_pass(r.detail) if r.passed else _fail(r.detail)} [{r.name}]")

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
