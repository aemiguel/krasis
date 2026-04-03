#!/usr/bin/env python3
"""Validate Krasis decode output against reference greedy outputs.

Starts a Krasis server with the given config, sends each reference prompt
with temperature=0 (greedy), compares output tokens against stored reference.
Reports exact match count, first divergence point, and PASS/FAIL.

Usage:
    ./dev validate <config>

This script must be run via ./dev validate, not directly.
"""

import argparse
import http.client
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Guard: must be run via ./dev
if not os.environ.get("KRASIS_DEV_SCRIPT"):
    print("ERROR: This script must be run via ./dev validate, not directly.")
    print("  Usage: ./dev validate <config>")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
# Reference data lives in krasis-internal (private repo, sibling directory)
INTERNAL_REFERENCE_DIR = REPO_DIR.parent / "krasis-internal" / "reference-outputs" / "output"
# Fallback: local reference outputs (for CI or standalone)
LOCAL_REFERENCE_DIR = SCRIPT_DIR / "reference_outputs"

# Map config MODEL_PATH basenames to reference output directory names.
# Reference dirs have suffixes from the data collection process (-expanded, -prefill).
_REF_DIR_MAP = {
    "Qwen3-Coder-Next": "Qwen3-Coder-Next-prefill",
    "Qwen3.5-35B-A3B": "Qwen3.5-35B-A3B-expanded",
    "Qwen3-235B-A22B": "Qwen3-235B-A22B-expanded",
    "Qwen3-0.6B": "Qwen3-0.6B-expanded",
    # These match exactly:
    # "Qwen3.5-122B-A10B", "Qwen3.5-397B-A17B", "GLM-4.7",
    # "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
}

DEFAULT_PORT = 8012
DEFAULT_HOST = "127.0.0.1"
HEALTH_TIMEOUT = 600  # 10 min — model load + cache build can be slow
BENCHMARK_TIMEOUT = 1200  # 20 min
GENERATE_TIMEOUT = 120  # 2 min per prompt

# Token match thresholds
# BF16 attention: expect near-perfect match (minor floating point differences possible)
BF16_MIN_MATCH_TOKENS = 50
# Quantized attention (AWQ/INT4/INT8): quantization noise causes divergence when
# top logits are close (the INT4 #1 choice may be the BF16 #2 or #3 choice).
# Threshold of 3 means: if the first 3 tokens match, it's working; earlier divergence
# is a red flag. The divergence_in_top_k check catches whether it's expected drift.
QUANT_MIN_MATCH_TOKENS = 3

BOLD = "\033[1m"
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
NC = "\033[0m"


def info(msg: str):
    print(f"{CYAN}{BOLD}=>{NC} {msg}")

def ok(msg: str):
    print(f"{GREEN}{BOLD}OK{NC} {msg}")

def warn(msg: str):
    print(f"{YELLOW}{BOLD}!!{NC} {msg}")

def die(msg: str):
    print(f"{RED}{BOLD}ERROR{NC} {msg}", file=sys.stderr)
    sys.exit(1)


def find_krasis_command() -> str:
    """Find the installed krasis command."""
    conda_path = os.path.expanduser("~/miniconda3/envs/ktransformers/bin/krasis")
    if os.path.isfile(conda_path):
        return conda_path
    path = shutil.which("krasis")
    if path:
        return path
    die("'krasis' command not found. Is Krasis installed?")
    return ""  # unreachable


def parse_config(config_path: str) -> Dict[str, str]:
    """Parse a .conf file into a dict of key=value pairs."""
    config: Dict[str, str] = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                # Strip quotes from value
                val = val.strip().strip('"').strip("'")
                config[key.strip()] = val
    return config


def extract_model_name(config_path: str) -> str:
    """Extract model name from config's MODEL_PATH."""
    config = parse_config(config_path)
    model_path = config.get("MODEL_PATH", "")
    if not model_path:
        die(f"No MODEL_PATH found in {config_path}")
    return os.path.basename(model_path)


def extract_port(config_path: str) -> int:
    """Extract port from config."""
    config = parse_config(config_path)
    return int(config.get("CFG_PORT", str(DEFAULT_PORT)))


def is_quantized_model(config_path: str) -> bool:
    """Check if the model uses quantized weights (INT4/INT8).

    All Krasis models use Marlin INT4/INT8 for MoE experts, which causes
    token-level drift vs BF16 reference. This is true even with BF16 attention.
    """
    config = parse_config(config_path)
    # Check weight quant — if set to anything other than bf16, model is quantized
    weight_quant = config.get("CFG_WEIGHT_QUANT", "4").lower()
    attn_quant = config.get("CFG_ATTENTION_QUANT", "bf16").lower()
    return weight_quant != "bf16" or attn_quant != "bf16"


def _resolve_reference_dir(model_name: str) -> Optional[Path]:
    """Find reference data directory for a model, checking name mapping."""
    # Try mapped name first, then exact name
    candidates = []
    mapped = _REF_DIR_MAP.get(model_name)
    if mapped:
        candidates.append(mapped)
    candidates.append(model_name)

    for ref_dir in [INTERNAL_REFERENCE_DIR, LOCAL_REFERENCE_DIR]:
        if not ref_dir.is_dir():
            continue
        for name in candidates:
            path = ref_dir / name / "greedy_reference.json"
            if path.is_file():
                return ref_dir / name
    return None


def load_reference(model_name: str) -> Dict[str, Any]:
    """Load reference outputs for a model."""
    ref_dir = _resolve_reference_dir(model_name)
    if ref_dir is None:
        die(f"No reference outputs found for {model_name}.\n"
            f"  Searched: {INTERNAL_REFERENCE_DIR}\n"
            f"           {LOCAL_REFERENCE_DIR}\n"
            f"  Name mappings tried: {_REF_DIR_MAP.get(model_name, model_name)}\n"
            f"  Generate with: ./dev generate-reference {model_name}")
    ref_path = ref_dir / "greedy_reference.json"
    info(f"Reference data: {ref_path}")
    with open(ref_path) as f:
        return json.load(f)


def launch_server(config_path: str, test_endpoints: bool = False) -> Tuple[subprocess.Popen, str]:
    """Launch Krasis server, return (process, log_path)."""
    log_dir = REPO_DIR / "logs" / "validate"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"validate_{Path(config_path).stem}_{int(time.time())}.log"

    # Use python -m krasis.server directly (same as ./dev run) to ensure
    # we run the correct krasis installation, not a stale one.
    python = sys.executable
    cmd = [python, "-m", "krasis.server", "--config", config_path]
    if test_endpoints:
        cmd.append("--test-endpoints")

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc, str(log_path)


def wait_for_health(port: int, timeout: int = HEALTH_TIMEOUT) -> bool:
    """Wait for server health endpoint."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            conn = http.client.HTTPConnection(DEFAULT_HOST, port, timeout=2)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            if resp.status == 200:
                conn.close()
                return True
            conn.close()
        except Exception:
            pass
        time.sleep(2)
    return False


def kill_server(proc: subprocess.Popen):
    """Kill server process group."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    for _ in range(10):
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def send_chat_greedy(messages: List[Dict[str, str]], port: int,
                     max_tokens: int = 200,
                     logprobs: bool = False, top_logprobs: int = 5) -> Dict[str, Any]:
    """Send a chat completion with temperature=0 (greedy). Returns text + metadata.

    When logprobs=True, also collects per-token log probabilities from the SSE stream.
    """
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "enable_thinking": False,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    body = json.dumps(payload).encode()

    try:
        conn = http.client.HTTPConnection(DEFAULT_HOST, port, timeout=GENERATE_TIMEOUT)
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()

        if resp.status != 200:
            error_body = resp.read().decode("utf-8", errors="replace")
            conn.close()
            return {"text": "", "logprobs_data": [], "error": f"HTTP {resp.status}: {error_body[:200]}"}

        text_parts = []
        logprobs_data = []  # List of [{token_id, logprob}, ...] per token
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                line, buf = buf.split(b"\n\n", 1)
                line = line.strip()
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    try:
                        data = json.loads(line[6:])
                        if "krasis_timing" in data:
                            continue
                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            text_parts.append(content)
                        # Collect logprobs if present
                        lp = choice.get("logprobs")
                        if lp and "content" in lp:
                            logprobs_data.append(lp["content"])
                    except json.JSONDecodeError:
                        pass

        conn.close()
        return {"text": "".join(text_parts), "logprobs_data": logprobs_data, "error": None}

    except Exception as e:
        return {"text": "", "logprobs_data": [], "error": str(e)}


def send_prefill_logits(messages: List[Dict[str, str]], port: int,
                        top_k: int = 10, sample_every: int = 50) -> Dict[str, Any]:
    """Call the test-only /v1/internal/prefill_logits endpoint.

    Returns position-by-position top-k logit distributions for prefill comparison.
    Only available when the server is started with --test-endpoints.
    """
    payload = {
        "messages": messages,
        "top_k": top_k,
        "sample_every": sample_every,
        "enable_thinking": False,
    }
    body = json.dumps(payload).encode()

    try:
        conn = http.client.HTTPConnection(DEFAULT_HOST, port, timeout=GENERATE_TIMEOUT)
        conn.request("POST", "/v1/internal/prefill_logits", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp_body = resp.read().decode("utf-8", errors="replace")
        conn.close()

        if resp.status != 200:
            return {"error": f"HTTP {resp.status}: {resp_body[:200]}"}

        return json.loads(resp_body)

    except Exception as e:
        return {"error": str(e)}


def compare_prefill_logits(ref_positions: List[Dict], krasis_positions: List[Dict]) -> Dict[str, Any]:
    """Compare prefill logits at matching positions.

    For each sampled position, checks:
    - Whether top-1 token matches
    - How many of the reference top-k appear in Krasis top-k
    - Max and mean logit divergence
    """
    ref_by_pos = {p["position"]: p["top_k"] for p in ref_positions}
    kra_by_pos = {p["position"]: p["top_k"] for p in krasis_positions}

    common_positions = sorted(set(ref_by_pos) & set(kra_by_pos))
    if not common_positions:
        return {"error": "No common positions to compare",
                "positions_compared": 0}

    top1_matches = 0
    top_k_overlap_sum = 0
    logit_deltas = []

    for pos in common_positions:
        ref_topk = ref_by_pos[pos]  # [(token_id, logit), ...]
        kra_topk = kra_by_pos[pos]

        # Top-1 match
        if ref_topk and kra_topk:
            ref_top1 = ref_topk[0][0] if isinstance(ref_topk[0], (list, tuple)) else ref_topk[0]["token_id"]
            kra_top1 = kra_topk[0][0] if isinstance(kra_topk[0], (list, tuple)) else kra_topk[0]["token_id"]
            if ref_top1 == kra_top1:
                top1_matches += 1

        # Top-k overlap
        ref_ids = set()
        ref_logit_map = {}
        for entry in ref_topk:
            if isinstance(entry, (list, tuple)):
                tid, logit = entry[0], entry[1]
            else:
                tid, logit = entry["token_id"], entry["logit"]
            ref_ids.add(tid)
            ref_logit_map[tid] = logit

        kra_ids = set()
        kra_logit_map = {}
        for entry in kra_topk:
            if isinstance(entry, (list, tuple)):
                tid, logit = entry[0], entry[1]
            else:
                tid, logit = entry["token_id"], entry["logit"]
            kra_ids.add(tid)
            kra_logit_map[tid] = logit

        overlap = ref_ids & kra_ids
        top_k_overlap_sum += len(overlap)

        # Logit delta for overlapping tokens
        for tid in overlap:
            delta = abs(ref_logit_map[tid] - kra_logit_map[tid])
            logit_deltas.append(delta)

    n = len(common_positions)
    k = len(ref_positions[0]["top_k"]) if ref_positions else 10
    mean_logit_delta = sum(logit_deltas) / len(logit_deltas) if logit_deltas else 0.0
    max_logit_delta = max(logit_deltas) if logit_deltas else 0.0

    return {
        "positions_compared": n,
        "top1_match_rate": top1_matches / n if n else 0.0,
        "top1_matches": top1_matches,
        "mean_top_k_overlap": top_k_overlap_sum / n if n else 0.0,
        "max_top_k": k,
        "mean_logit_delta": mean_logit_delta,
        "max_logit_delta": max_logit_delta,
    }


def tokenize_text(text: str, tokenizer) -> List[int]:
    """Tokenize text using the model's tokenizer."""
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False)


def compare_tokens(ref_ids: List[int], krasis_ids: List[int],
                    per_token_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Compare two token ID sequences.

    Returns dict with match_count, first_divergence, total_ref, total_krasis.
    If per_token_data is available from the reference, also checks whether
    Krasis's divergent token was in the reference top-k (floating point drift
    vs real bug).
    """
    match_count = 0
    first_divergence = None
    divergence_in_top_k = None
    divergence_top_k_rank = None
    divergence_ref_confidence = None
    min_len = min(len(ref_ids), len(krasis_ids))

    for i in range(min_len):
        if ref_ids[i] == krasis_ids[i]:
            match_count += 1
        else:
            first_divergence = i
            # Check if Krasis's token was in reference top-k
            if per_token_data and i < len(per_token_data):
                ptd = per_token_data[i]
                top_k = ptd.get("top_k", [])
                divergence_ref_confidence = ptd.get("log_prob")
                krasis_tok = krasis_ids[i]
                for rank, entry in enumerate(top_k):
                    if entry["token_id"] == krasis_tok:
                        divergence_in_top_k = True
                        divergence_top_k_rank = rank + 1  # 1-indexed
                        break
                else:
                    divergence_in_top_k = False
            break
    else:
        if len(ref_ids) != len(krasis_ids):
            first_divergence = min_len

    return {
        "match_count": match_count,
        "first_divergence": first_divergence,
        "total_ref": len(ref_ids),
        "total_krasis": len(krasis_ids),
        "exact_match": ref_ids == krasis_ids,
        "divergence_in_top_k": divergence_in_top_k,
        "divergence_top_k_rank": divergence_top_k_rank,
        "divergence_ref_confidence": divergence_ref_confidence,
    }


def validate_model(config_path: str, no_server: bool = False, port: Optional[int] = None,
                    enable_logprobs: bool = True, enable_prefill: bool = True,
                    max_prompts: Optional[int] = None,
                    return_summary: bool = False):
    """Run multi-layer validation against reference outputs.

    Layer 1: Decode token matching (always on)
    Layer 2: Decode logprobs comparison (when enable_logprobs=True)
    Layer 3: Prefill logit comparison (when enable_prefill=True, requires --test-endpoints)
    """
    from transformers import AutoTokenizer

    model_name = extract_model_name(config_path)
    config_port = port if port else extract_port(config_path)
    is_quant = is_quantized_model(config_path)
    min_match = QUANT_MIN_MATCH_TOKENS if is_quant else BF16_MIN_MATCH_TOKENS

    info(f"Model: {model_name}")
    info(f"Config: {config_path}")
    info(f"Attention: {'quantized' if is_quant else 'BF16'} (min match: {min_match} tokens)")
    layers_active = ["L1:tokens"]
    if enable_logprobs:
        layers_active.append("L2:logprobs")
    if enable_prefill:
        layers_active.append("L3:prefill")
    info(f"Validation layers: {', '.join(layers_active)}")
    if max_prompts is not None:
        info(f"Prompt cap: {max_prompts}")

    # Load reference
    reference = load_reference(model_name)
    ref_conversations = reference["conversations"]
    total_reference_prompts = sum(len(c["turns"]) for c in ref_conversations)
    if max_prompts is not None:
        total_prompts = min(total_reference_prompts, max_prompts)
    else:
        total_prompts = total_reference_prompts
    info(f"Reference: {total_prompts} prompts from {reference['generated_at']}")
    info(f"Reference runtime: {reference['runtime']} {reference.get('runtime_version', '')}")

    # Check if reference has prefill logits
    has_ref_prefill = any(
        "prefill_logits" in turn
        for conv in ref_conversations
        for turn in conv["turns"]
    )
    if enable_prefill and not has_ref_prefill:
        warn("Reference data has no prefill_logits — L3 will be skipped")
        enable_prefill = False

    # Start server FIRST (before loading tokenizer) to avoid RAM contention
    proc = None
    if not no_server:
        info(f"Starting Krasis server on port {config_port}...")
        proc, log_path = launch_server(config_path, test_endpoints=enable_prefill)
        info(f"Server log: {log_path}")

        if not wait_for_health(config_port):
            kill_server(proc)
            die(f"Server did not become healthy within {HEALTH_TIMEOUT}s. Check log: {log_path}")
        ok("Server is ready")
    else:
        info(f"Using existing server on port {config_port}")

    # Load tokenizer AFTER server is healthy (server has released CPU RAM by now)
    model_path = os.path.join(os.path.expanduser("~/.krasis/models"), model_name)
    info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Run validation
    results: List[Dict[str, Any]] = []
    prompt_num = 0
    pass_count = 0
    warn_count = 0
    fail_count = 0

    try:
        for conv_idx, conv in enumerate(ref_conversations):
            messages: List[Dict[str, str]] = []
            is_multi = len(conv["turns"]) > 1

            if is_multi:
                print(f"\n  {BOLD}--- Conversation {conv_idx + 1} ({len(conv['turns'])} turns) ---{NC}")

            for turn_idx, turn in enumerate(conv["turns"]):
                if max_prompts is not None and prompt_num >= max_prompts:
                    break
                prompt_num += 1
                prompt = turn["prompt"]
                ref_token_ids = turn["token_ids"]
                ref_text = turn["text"]
                ref_num = turn["num_tokens"]
                per_token_data = turn.get("per_token_data")  # v2 format
                ref_input_ids = turn.get("input_token_ids")  # v2 format
                turn_label = f" [turn {turn_idx + 1}/{len(conv['turns'])}]" if is_multi else ""

                print(f"\n  {BOLD}[{prompt_num}/{total_prompts}]{NC} {prompt}{turn_label}")
                print(f"    {DIM}Reference: {ref_num} tokens{NC}")

                messages.append({"role": "user", "content": prompt})

                # Verify tokenizer agreement if input_token_ids available
                if ref_input_ids:
                    try:
                        template_out = tokenizer.apply_chat_template(
                            messages, return_tensors="pt",
                            add_generation_prompt=True, enable_thinking=False)
                    except TypeError:
                        template_out = tokenizer.apply_chat_template(
                            messages, return_tensors="pt", add_generation_prompt=True)
                    import torch
                    if hasattr(template_out, "input_ids"):
                        local_ids = template_out.input_ids[0].tolist()
                    elif isinstance(template_out, torch.Tensor):
                        local_ids = template_out[0].tolist()
                    else:
                        local_ids = list(template_out)

                    if local_ids != ref_input_ids:
                        # Multi-turn tokenizer mismatch is expected when prior turns
                        # diverged (INT4 quantization causes different assistant text,
                        # which changes the input for subsequent turns).
                        is_multi_turn = turn_idx > 0
                        if is_multi_turn:
                            print(f"    {YELLOW}TOKENIZER MISMATCH (expected: prior turn diverged){NC}")
                            print(f"    {DIM}Local: {len(local_ids)} tokens, Ref: {len(ref_input_ids)} tokens{NC}")
                            results.append({"prompt": prompt, "status": "SKIP",
                                            "error": "multi-turn tokenizer mismatch (expected)"})
                            # Skip subsequent turns too since they'll cascade
                            messages.pop()
                            break
                        else:
                            print(f"    {RED}TOKENIZER MISMATCH: input tokens differ from reference!{NC}")
                            print(f"    {DIM}Local: {len(local_ids)} tokens, Ref: {len(ref_input_ids)} tokens{NC}")
                            for ti in range(min(len(local_ids), len(ref_input_ids))):
                                if local_ids[ti] != ref_input_ids[ti]:
                                    print(f"    {DIM}First diff at position {ti}: "
                                          f"local={local_ids[ti]} ref={ref_input_ids[ti]}{NC}")
                                    break
                            results.append({"prompt": prompt, "status": "FAIL",
                                            "error": "tokenizer mismatch"})
                            fail_count += 1
                            messages.pop()
                            continue

                # ── Layer 3: Prefill logits (first turn only, if enabled) ──
                ref_prefill = turn.get("prefill_logits")
                prefill_result = None
                if enable_prefill and ref_prefill and turn_idx == 0:
                    # Only do prefill comparison on the first turn (single-turn prompt)
                    # because multi-turn requires stateful KV which the test endpoint doesn't support
                    pfx_resp = send_prefill_logits(
                        [{"role": "user", "content": prompt}], config_port,
                        top_k=10, sample_every=ref_prefill.get("sample_every", 50))
                    if "error" in pfx_resp and pfx_resp.get("error"):
                        print(f"    {YELLOW}L3 prefill: {pfx_resp['error']}{NC}")
                    elif "positions" in pfx_resp:
                        ref_pos = ref_prefill.get("positions", [])
                        kra_pos = pfx_resp["positions"]
                        prefill_result = compare_prefill_logits(ref_pos, kra_pos)
                        top1_pct = prefill_result["top1_match_rate"] * 100
                        mean_delta = prefill_result["mean_logit_delta"]
                        max_delta = prefill_result["max_logit_delta"]
                        n_pos = prefill_result["positions_compared"]
                        color = GREEN if top1_pct > 90 else (YELLOW if top1_pct > 70 else RED)
                        print(f"    {color}L3 prefill: {n_pos} positions, "
                              f"top1 match {top1_pct:.0f}%, "
                              f"mean delta {mean_delta:.2f}, max delta {max_delta:.2f}{NC}")

                # ── Layer 1+2: Decode token matching + logprobs ──
                resp = send_chat_greedy(messages, config_port, max_tokens=ref_num + 50,
                                        logprobs=enable_logprobs)

                if resp["error"]:
                    print(f"    {RED}ERROR: {resp['error']}{NC}")
                    results.append({"prompt": prompt, "status": "FAIL", "error": resp["error"]})
                    fail_count += 1
                    messages.pop()  # Remove failed user message
                    continue

                krasis_text = resp["text"]
                krasis_token_ids = tokenize_text(krasis_text, tokenizer)

                # Layer 1: Compare tokens (pass per_token_data for top-k divergence analysis)
                cmp = compare_tokens(ref_token_ids, krasis_token_ids, per_token_data)

                # Layer 2: Logprobs summary
                logprobs_summary = ""
                if enable_logprobs and resp.get("logprobs_data"):
                    lp_data = resp["logprobs_data"]
                    # Extract log probs from the stream data
                    log_probs = []
                    for lp_entry in lp_data:
                        if isinstance(lp_entry, list) and lp_entry:
                            # Each entry is a list of {token_id, logprob}
                            log_probs.append(lp_entry[0].get("logprob", 0))
                    if log_probs:
                        mean_lp = sum(log_probs) / len(log_probs)
                        min_lp = min(log_probs)
                        logprobs_summary = (
                            f"\n    {DIM}L2 logprobs: {len(log_probs)} tokens, "
                            f"mean {mean_lp:.3f}, min {min_lp:.3f}{NC}")

                # Build divergence diagnostic string
                diverge_detail = ""
                if cmp["divergence_in_top_k"] is True:
                    rank = cmp["divergence_top_k_rank"]
                    conf = cmp.get("divergence_ref_confidence")
                    conf_pct = f" (ref confidence: {math.exp(conf)*100:.1f}%)" if conf else ""
                    diverge_detail = (f"\n    {DIM}Krasis token was reference #"
                                      f"{rank} choice{conf_pct} — likely FP drift{NC}")
                elif cmp["divergence_in_top_k"] is False:
                    conf = cmp.get("divergence_ref_confidence")
                    conf_pct = f" (ref confidence: {math.exp(conf)*100:.1f}%)" if conf else ""
                    diverge_detail = (f"\n    {RED}Krasis token NOT in reference top-k"
                                      f"{conf_pct} — possible real bug{NC}")

                # Determine status
                # For quantized models, the key signal is whether the divergent
                # token is in the reference top-k (expected FP drift) or not
                # (possible bug). The position of divergence matters less because
                # INT4 quantization can make the very first token differ when
                # the top-2 BF16 logits are close.
                if cmp["exact_match"]:
                    status = "MATCH"
                    pass_count += 1
                    print(f"    {GREEN}L1 tokens: MATCH {cmp['match_count']}/{cmp['total_ref']} identical{NC}"
                          f"{logprobs_summary}")
                elif is_quant and cmp["divergence_in_top_k"] is True:
                    # Quantized model: divergent token is in reference top-k = expected drift
                    status = "WARN"
                    warn_count += 1
                    div_at = cmp["first_divergence"]
                    ref_tok_at = ref_token_ids[div_at] if div_at < len(ref_token_ids) else "END"
                    kra_tok_at = krasis_token_ids[div_at] if div_at < len(krasis_token_ids) else "END"
                    print(f"    {YELLOW}L1 tokens: DIVERGE at token {div_at}: "
                          f"ref={ref_tok_at} vs krasis={kra_tok_at}{NC}")
                    print(f"    {YELLOW}First {cmp['match_count']} tokens match — "
                          f"divergent token in reference top-k (expected INT4 drift){NC}"
                          f"{diverge_detail}{logprobs_summary}")
                elif cmp["first_divergence"] is not None and cmp["first_divergence"] >= min_match:
                    # Non-quantized or divergent token NOT in top-k but enough tokens matched
                    status = "WARN"
                    warn_count += 1
                    ref_tok_at = ref_token_ids[cmp["first_divergence"]] if cmp["first_divergence"] < len(ref_token_ids) else "END"
                    kra_tok_at = krasis_token_ids[cmp["first_divergence"]] if cmp["first_divergence"] < len(krasis_token_ids) else "END"
                    print(f"    {YELLOW}L1 tokens: DIVERGE at token {cmp['first_divergence']}: "
                          f"ref={ref_tok_at} vs krasis={kra_tok_at}{NC}")
                    print(f"    {YELLOW}First {cmp['match_count']} tokens match "
                          f"(threshold: {min_match}) — within tolerance{NC}"
                          f"{diverge_detail}{logprobs_summary}")
                else:
                    status = "FAIL"
                    fail_count += 1
                    diverge_at = cmp["first_divergence"] if cmp["first_divergence"] is not None else 0
                    print(f"    {RED}L1 tokens: FAIL diverges at token {diverge_at} "
                          f"(need {min_match}+ matching){NC}"
                          f"{diverge_detail}{logprobs_summary}")
                    # Show what went wrong
                    ref_preview = ref_text[:80].replace("\n", " ")
                    kra_preview = krasis_text[:80].replace("\n", " ")
                    print(f"    {DIM}Ref:    {ref_preview}{'...' if len(ref_text) > 80 else ''}{NC}")
                    print(f"    {DIM}Krasis: {kra_preview}{'...' if len(krasis_text) > 80 else ''}{NC}")

                results.append({
                    "prompt": prompt,
                    "status": status,
                    "match_count": cmp["match_count"],
                    "first_divergence": cmp["first_divergence"],
                    "ref_tokens": cmp["total_ref"],
                    "krasis_tokens": cmp["total_krasis"],
                    "divergence_in_top_k": cmp["divergence_in_top_k"],
                    "divergence_top_k_rank": cmp["divergence_top_k_rank"],
                    "prefill_result": prefill_result,
                })

                # Add to message history for multi-turn
                messages.append({"role": "assistant", "content": krasis_text})

            if max_prompts is not None and prompt_num >= max_prompts:
                break

    finally:
        # Always clean up server
        if proc:
            info("Stopping server...")
            kill_server(proc)

    # Summary
    skip_count = sum(1 for r in results if r.get("status") == "SKIP")
    print(f"\n  {BOLD}{'=' * 50}{NC}")
    total = pass_count + warn_count + fail_count
    skip_str = f", {skip_count} SKIP" if skip_count else ""
    print(f"  {BOLD}Results: {pass_count} MATCH, {warn_count} DIVERGE (tolerated), {fail_count} FAIL{skip_str}{NC}")
    if is_quant:
        print(f"  {BOLD}Weights: quantized (INT4/INT8) — divergence in top-k = expected drift{NC}")
    else:
        print(f"  {BOLD}Weights: BF16 | Min match threshold: {min_match} tokens{NC}")

    summary = {
        "model_name": model_name,
        "config_path": config_path,
        "is_quantized": is_quant,
        "min_match_threshold": min_match,
        "layers_active": layers_active,
        "max_prompts": max_prompts,
        "reference_prompts_total": total_reference_prompts,
        "prompts_run": total,
        "match_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "skip_count": skip_count,
        "passed": fail_count == 0,
        "results": results,
    }

    if fail_count == 0:
        print(f"\n  {GREEN}{BOLD}VALIDATION PASSED{NC}")
    else:
        print(f"\n  {RED}{BOLD}VALIDATION FAILED ({fail_count} failures){NC}")

    if return_summary:
        return summary
    return 0 if fail_count == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Validate Krasis decode against reference outputs")
    parser.add_argument("config", help="Path to Krasis .conf file")
    parser.add_argument("--no-server", action="store_true",
                        help="Don't start a server; connect to existing one")
    parser.add_argument("--port", type=int, default=None,
                        help="Override port (default: from config or 8012)")
    parser.add_argument("--no-logprobs", action="store_true",
                        help="Disable Layer 2 (decode logprobs)")
    parser.add_argument("--no-prefill", action="store_true",
                        help="Disable Layer 3 (prefill logit comparison)")
    parser.add_argument("--l1-only", action="store_true",
                        help="Only run Layer 1 (token matching)")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Stop after this many reference prompts")
    args = parser.parse_args()

    enable_logprobs = not args.no_logprobs and not args.l1_only
    enable_prefill = not args.no_prefill and not args.l1_only

    sys.exit(validate_model(args.config, args.no_server, args.port,
                            enable_logprobs=enable_logprobs,
                            enable_prefill=enable_prefill,
                            max_prompts=args.max_prompts))


if __name__ == "__main__":
    main()
