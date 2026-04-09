#!/usr/bin/env python3
"""Reference test runner for Krasis.

Compares engine output against BF16 HuggingFace Transformers reference data.
Feeds raw input_token_ids directly to the prefill engine (no tokenization,
no chat template re-application) and compares greedy decode output.

Produces a self-contained HTML report with per-prompt diagnostics.

Modes:
  Single config:  ./dev reference-test <config>
  Comparison:     ./dev reference-test --compare <model>

This script must be run via ./dev reference-test, not directly.
"""

import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from krasis.run_paths import get_run_dir


# ── Config Matrix for Comparison Mode ─────────────────────────────
# Maps model short name -> list of (label, relative_config_path) tuples.
# Paths are relative to the krasisx directory (script_dir).
# Missing configs are skipped with a warning.

COMPARE_CONFIGS = {
    "qcn": [
        ("INT8/BF16", "tests/qcn-8-8-a16.conf"),
        ("INT8/AWQ",  "tests/qcn-8-8-awq.conf"),
        ("INT4/BF16", "tests/qcn-bf16.conf"),
        ("INT4/AWQ",  "tests/qcn-awq.conf"),
    ],
    "q35b": [
        ("INT8/BF16", "tests/qwen35-8-8-a16.conf"),
        ("INT4/BF16", "tests/q35b-4-4-a16.conf"),
    ],
    "q122b": [
        ("INT4/BF16", "tests/q122b-4-4-a16.conf"),
        ("INT4/AWQ",  "tests/q122b-4-4-awq.conf"),
    ],
}

CONFIG_COLORS = {
    "INT8/BF16": "#58a6ff",
    "INT8/AWQ":  "#3fb950",
    "INT4/BF16": "#d29922",
    "INT4/AWQ":  "#d2a8ff",
}


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ── Config Parsing ──────────────────────────────────────────────────

def parse_config(conf_path: str) -> Dict[str, str]:
    """Parse a Krasis .conf file into a dict."""
    cfg = {}
    with open(conf_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # Handle both MODEL_PATH and CFG_MODEL_PATH
                if key == "MODEL_PATH":
                    key = "CFG_MODEL_PATH"
                cfg[key] = val
    return cfg


def resolve_model_name(cfg: Dict[str, str]) -> str:
    """Extract model name from config's model path."""
    model_path = cfg.get("CFG_MODEL_PATH", "")
    model_path = os.path.expanduser(model_path)
    return os.path.basename(model_path.rstrip("/"))


def find_reference_data(model_name: str, script_dir: str) -> Optional[str]:
    """Find reference data JSON for a model."""
    # krasis-internal sits beside krasisx
    internal_dir = os.path.join(os.path.dirname(script_dir), "krasis-internal")
    ref_base = os.path.join(internal_dir, "reference-outputs", "output")

    # Search order
    suffixes = ["-prefill", "-expanded", ""]
    for suffix in suffixes:
        candidate = os.path.join(ref_base, f"{model_name}{suffix}", "greedy_reference.json")
        if os.path.isfile(candidate):
            return candidate

    return None


def list_available_references(script_dir: str) -> List[str]:
    """List all available reference data directories."""
    internal_dir = os.path.join(os.path.dirname(script_dir), "krasis-internal")
    ref_base = os.path.join(internal_dir, "reference-outputs", "output")
    if not os.path.isdir(ref_base):
        return []
    return sorted(os.listdir(ref_base))


# ── Server Management ───────────────────────────────────────────────

def start_server(conf_path: str, script_dir: str) -> Tuple[subprocess.Popen, int]:
    """Start Krasis server with --test-endpoints and return (proc, port)."""
    cfg = parse_config(conf_path)
    port = int(cfg.get("CFG_PORT", "8012"))

    python = os.path.join(script_dir, "..", "miniconda3", "envs", "ktransformers", "bin", "python")
    if not os.path.isfile(python):
        python = "/home/main/miniconda3/envs/ktransformers/bin/python"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Reference testing cares about correctness, not startup VRAM profiling.
    # Cap the long startup probe so test-endpoint runs do not stall in the
    # server's heavyweight calibration path before the first request.
    env.setdefault("KRASIS_STARTUP_CALIBRATION_LONG_TOKENS_CAP", "4000")

    # Redirect server output to a file instead of PIPE to avoid pipe-buffer
    # deadlock: the server writes diagnostic output to stdout/stderr, and if
    # the pipe buffer fills up (64KB) and nobody drains it, the server blocks.
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    server_log_path = os.path.join(log_dir, "reference_test_server.log")
    server_log_file = open(server_log_path, "w")

    proc = subprocess.Popen(
        [python, "-m", "krasis.server", "--config", conf_path, "--test-endpoints"],
        stdout=server_log_file,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=script_dir,
    )
    proc._server_log_file = server_log_file  # keep reference to close later
    return proc, port


def wait_for_server(port: int, timeout: int = 600) -> bool:
    """Wait for server to respond on /v1/models."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/v1/models")
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
    """Kill the server process."""
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
    # Close server log file if we opened one
    if hasattr(proc, '_server_log_file'):
        try:
            proc._server_log_file.close()
        except Exception:
            pass


def cleanup_between_configs():
    """Kill any lingering krasis processes and wait for GPU memory release."""
    subprocess.run(["pkill", "-9", "-f", "python.*krasis\\."], capture_output=True)
    time.sleep(3)


# ── API Calls ───────────────────────────────────────────────────────

def call_reference_test(port: int, input_token_ids: List[int], max_tokens: int,
                        stop_token_ids: List[int], top_logprobs: int = 10,
                        timeout: int = 120) -> Dict:
    """Call the /v1/internal/reference_test endpoint."""
    payload = json.dumps({
        "input_token_ids": input_token_ids,
        "max_tokens": max_tokens,
        "top_logprobs": top_logprobs,
        "stop_token_ids": stop_token_ids,
    })

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("POST", "/v1/internal/reference_test", body=payload,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read().decode()
    conn.close()

    if resp.status != 200:
        raise RuntimeError(f"reference_test endpoint returned {resp.status}: {body}")

    return json.loads(body)


def call_prefill_logits(port: int, input_token_ids: List[int], top_k: int = 10,
                        sample_every: int = 1, timeout: int = 60) -> Dict:
    """Call the /v1/internal/prefill_logits endpoint with raw token IDs."""
    payload = json.dumps({
        "input_token_ids": input_token_ids,
        "top_k": top_k,
        "sample_every": sample_every,
    })
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("POST", "/v1/internal/prefill_logits", body=payload,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read().decode()
    conn.close()
    if resp.status != 200:
        return {"error": body, "positions": []}
    return json.loads(body)


def compute_prefill_metrics(ref_turn: Dict, prefill_result: Dict) -> Dict:
    """Compare our prefill logits against BF16 reference at sampled positions."""
    ref_logits = ref_turn.get("prefill_logits", {})
    ref_positions = ref_logits.get("positions", [])
    our_positions = {p["position"]: p for p in prefill_result.get("positions", [])}

    argmax_match = 0
    top10_hits = 0
    total = 0

    for rp in ref_positions:
        pos = rp["position"]
        if pos not in our_positions:
            continue
        ref_top_id = rp["top_k"][0]["token_id"]
        our_tk = our_positions[pos]["top_k"]
        our_top_id = our_tk[0]["token_id"] if our_tk else None
        our_ids = {t["token_id"] for t in our_tk}
        if ref_top_id == our_top_id:
            argmax_match += 1
        if ref_top_id in our_ids:
            top10_hits += 1
        total += 1

    return {
        "prefill_argmax_match": argmax_match,
        "prefill_top10_hits": top10_hits,
        "prefill_total": total,
        "prefill_argmax_rate": argmax_match / total if total > 0 else 0.0,
        "prefill_containment": top10_hits / total if total > 0 else 0.0,
    }


# ── Metrics ─────────────────────────────────────────────────────────

def compute_metrics(ref_turn: Dict, engine_result: Dict) -> Dict:
    """Compute all comparison metrics for one turn."""
    ref_tokens = ref_turn["token_ids"]
    our_tokens = engine_result["token_ids"]
    ref_per_token = ref_turn.get("per_token_data", [])
    our_per_token = engine_result.get("per_token_data", [])

    # 1. First-token match
    first_match = len(our_tokens) > 0 and len(ref_tokens) > 0 and our_tokens[0] == ref_tokens[0]

    # 2. Token match run length
    match_run = 0
    min_len = min(len(ref_tokens), len(our_tokens))
    for i in range(min_len):
        if ref_tokens[i] == our_tokens[i]:
            match_run += 1
        else:
            break

    # 3. Top-k containment: is the reference token in our engine's top-k?
    containment_hits = 0
    containment_total = 0
    for i in range(min(len(our_per_token), len(ref_tokens))):
        our_tk = our_per_token[i].get("top_k", [])
        our_top_ids = {t["token_id"] for t in our_tk}
        if ref_tokens[i] in our_top_ids:
            containment_hits += 1
        containment_total += 1

    containment_rate = containment_hits / containment_total if containment_total > 0 else 0.0

    # 4. Divergence info
    divergence_pos = match_run if match_run < min_len else None
    divergence_info = None
    if divergence_pos is not None and divergence_pos < min_len:
        ref_tok = ref_tokens[divergence_pos]
        our_tok = our_tokens[divergence_pos]
        # Find ref token rank in our top-k
        ref_rank = None
        ref_in_topk = False
        if divergence_pos < len(our_per_token):
            our_tk = our_per_token[divergence_pos].get("top_k", [])
            for rank, t in enumerate(our_tk):
                if t["token_id"] == ref_tok:
                    ref_rank = rank + 1
                    ref_in_topk = True
                    break
        # Get log probs
        ref_lp = ref_per_token[divergence_pos]["log_prob"] if divergence_pos < len(ref_per_token) else None
        our_lp = our_per_token[divergence_pos].get("log_prob", None) if divergence_pos < len(our_per_token) else None

        divergence_info = {
            "position": divergence_pos,
            "ref_token": ref_tok,
            "our_token": our_tok,
            "ref_in_our_topk": ref_in_topk,
            "ref_rank_in_our_topk": ref_rank,
            "ref_logprob": ref_lp,
            "our_logprob": our_lp,
        }

    # Build token comparison table
    show_tokens = max(30, match_run + 5) if divergence_pos is not None else 30
    show_tokens = min(show_tokens, min_len)
    token_table = []
    for i in range(show_tokens):
        ref_tok = ref_tokens[i] if i < len(ref_tokens) else None
        our_tok = our_tokens[i] if i < len(our_tokens) else None
        match = ref_tok == our_tok if ref_tok is not None and our_tok is not None else False

        # Check if ref token is in our top-k
        in_topk = False
        ref_rank = None
        if i < len(our_per_token):
            for rank, t in enumerate(our_per_token[i].get("top_k", [])):
                if t["token_id"] == ref_tok:
                    in_topk = True
                    ref_rank = rank + 1
                    break

        ref_lp = ref_per_token[i]["log_prob"] if i < len(ref_per_token) else None
        our_lp = our_per_token[i].get("log_prob", None) if i < len(our_per_token) else None

        token_table.append({
            "pos": i,
            "ref_token": ref_tok,
            "our_token": our_tok,
            "match": match,
            "in_topk": in_topk,
            "ref_rank": ref_rank,
            "ref_logprob": ref_lp,
            "our_logprob": our_lp,
        })

    return {
        "first_match": first_match,
        "match_run": match_run,
        "ref_tokens_count": len(ref_tokens),
        "our_tokens_count": len(our_tokens),
        "containment_rate": containment_rate,
        "containment_hits": containment_hits,
        "containment_total": containment_total,
        "divergence_pos": divergence_pos,
        "divergence_info": divergence_info,
        "token_table": token_table,
        "ref_text": ref_turn.get("text", "")[:500],
        "our_text": engine_result.get("text", "")[:500],
    }


def judge_prompt(metrics: Dict, prefill_metrics: Dict) -> str:
    """Return PASS, WARN, or FAIL for a single prompt.

    Primary metric: prefill top-10 containment (same input, only quant differs).
    Secondary: first-token match and decode match run.
    """
    pc = prefill_metrics.get("prefill_containment", 0.0)

    # FAIL: prefill logits badly off — likely a code bug
    if pc < 0.60:
        return "FAIL"

    # WARN: prefill is reasonable but not great
    if pc < 0.80:
        return "WARN"

    # PASS: prefill logits match well (quantization noise only)
    return "PASS"


def judge_overall(prompt_verdicts: List[str]) -> str:
    """Return overall PASS, WARN, or FAIL."""
    if any(v == "FAIL" for v in prompt_verdicts):
        return "FAIL"
    warn_count = sum(1 for v in prompt_verdicts if v == "WARN")
    if warn_count >= 3:
        return "WARN"
    return "PASS"


# ── Run Single Config ──────────────────────────────────────────────

def run_config_test(conf_path: str, ref_data: Dict, script_dir: str,
                    label: Optional[str] = None, verbose: bool = False) -> Optional[Dict]:
    """Run reference test for one config. Returns results dict or None on failure."""
    cfg = parse_config(conf_path)
    model_name = resolve_model_name(cfg)
    port = int(cfg.get("CFG_PORT", "8012"))

    expert_bits = cfg.get("CFG_GPU_EXPERT_BITS", "4")
    attn_quant = cfg.get("CFG_ATTENTION_QUANT", "bf16")
    if label is None:
        label = f"INT{expert_bits}/{attn_quant.upper()}"

    print(f"\n{'='*60}")
    print(f"Config: {label} ({os.path.basename(conf_path)})")
    print(f"{'='*60}")

    t_start = time.time()
    proc = None

    try:
        print("Starting server...")
        proc, port = start_server(conf_path, script_dir)
        print(f"Waiting for server on port {port}...")
        if not wait_for_server(port, timeout=1200):
            print("ERROR: Server did not start within timeout", file=sys.stderr)
            return None
        print("Server ready.")

        # Run all prompts
        eos_token_ids = ref_data.get("eos_token_ids", [])
        max_new_tokens = ref_data.get("max_new_tokens", 200)
        prompt_results = []

        for conv_idx, conv in enumerate(ref_data["conversations"]):
            turns = conv["turns"]
            for turn_idx, turn in enumerate(turns):
                prompt_text = turn.get("prompt", "")
                input_token_ids = turn["input_token_ids"]
                n_input = len(input_token_ids)

                turn_label = f"Conv {conv_idx+1}"
                if len(turns) > 1:
                    turn_label += f" Turn {turn_idx+1}"
                print(f"  {turn_label}: {n_input} input tokens, prompt: {prompt_text[:60]}...")

                try:
                    result = call_reference_test(
                        port=port,
                        input_token_ids=input_token_ids,
                        max_tokens=max_new_tokens,
                        stop_token_ids=eos_token_ids,
                        top_logprobs=10,
                    )
                except Exception as e:
                    print(f"    ERROR: {e}", file=sys.stderr)
                    prompt_results.append({
                        "conv_idx": conv_idx,
                        "turn": turn_idx + 1,
                        "multi_turn": len(turns) > 1,
                        "prompt": prompt_text,
                        "verdict": "FAIL",
                        "metrics": {
                            "first_match": False, "match_run": 0,
                            "ref_tokens_count": len(turn["token_ids"]),
                            "our_tokens_count": 0, "containment_rate": 0.0,
                            "containment_hits": 0, "containment_total": 0,
                            "divergence_pos": 0, "divergence_info": None,
                            "token_table": [], "ref_text": turn.get("text", "")[:500],
                            "our_text": f"ERROR: {e}",
                        },
                        "prefill_metrics": {
                            "prefill_argmax_rate": 0.0, "prefill_containment": 0.0,
                            "prefill_argmax_match": 0, "prefill_top10_hits": 0, "prefill_total": 0,
                        },
                        "timing": None,
                    })
                    continue

                metrics = compute_metrics(turn, result)

                # Prefill logits comparison (same input, no divergence)
                prefill_metrics = {
                    "prefill_argmax_rate": 0.0, "prefill_containment": 0.0,
                    "prefill_argmax_match": 0, "prefill_top10_hits": 0, "prefill_total": 0,
                }
                has_ref_logits = bool(turn.get("prefill_logits", {}).get("positions"))
                if has_ref_logits:
                    try:
                        pl_result = call_prefill_logits(port, input_token_ids, top_k=10, sample_every=1)
                        prefill_metrics = compute_prefill_metrics(turn, pl_result)
                    except Exception as e:
                        print(f"    (prefill_logits failed: {e})")

                verdict = judge_prompt(metrics, prefill_metrics)
                timing = result.get("timing")

                prompt_results.append({
                    "conv_idx": conv_idx,
                    "turn": turn_idx + 1,
                    "multi_turn": len(turns) > 1,
                    "prompt": prompt_text,
                    "verdict": verdict,
                    "metrics": metrics,
                    "prefill_metrics": prefill_metrics,
                    "timing": timing,
                })

                # Print summary
                fm = "MATCH" if metrics["first_match"] else "MISS"
                pc = prefill_metrics.get("prefill_containment", 0)
                pa = prefill_metrics.get("prefill_argmax_rate", 0)
                print(f"    {verdict}: first={fm}, run={metrics['match_run']}/{metrics['ref_tokens_count']}, "
                      f"prefill={100*pa:.0f}%/{100*pc:.0f}%, "
                      f"decode-top-k={100*metrics['containment_rate']:.0f}%")

    finally:
        if proc:
            print("Stopping server...")
            kill_server(proc)

    duration = time.time() - t_start

    # Compute aggregates
    prompt_verdicts = [r["verdict"] for r in prompt_results]
    overall = judge_overall(prompt_verdicts)

    total_prompts = len(prompt_results)
    total_pf_argmax = sum(r.get("prefill_metrics", {}).get("prefill_argmax_match", 0) for r in prompt_results)
    total_pf_top10 = sum(r.get("prefill_metrics", {}).get("prefill_top10_hits", 0) for r in prompt_results)
    total_pf_pos = sum(r.get("prefill_metrics", {}).get("prefill_total", 0) for r in prompt_results)

    aggregates = {
        "overall": overall,
        "n_pass": sum(1 for v in prompt_verdicts if v == "PASS"),
        "n_warn": sum(1 for v in prompt_verdicts if v == "WARN"),
        "n_fail": sum(1 for v in prompt_verdicts if v == "FAIL"),
        "total_prompts": total_prompts,
        "first_match_count": sum(1 for r in prompt_results if r["metrics"]["first_match"]),
        "first_match_rate": sum(1 for r in prompt_results if r["metrics"]["first_match"]) / total_prompts if total_prompts else 0,
        "avg_match_run": sum(r["metrics"]["match_run"] for r in prompt_results) / total_prompts if total_prompts else 0,
        "avg_containment": sum(r["metrics"]["containment_rate"] for r in prompt_results) / total_prompts if total_prompts else 0,
        "prefill_argmax_rate": total_pf_argmax / total_pf_pos if total_pf_pos > 0 else 0,
        "prefill_containment": total_pf_top10 / total_pf_pos if total_pf_pos > 0 else 0,
        "prefill_argmax_count": total_pf_argmax,
        "prefill_top10_count": total_pf_top10,
        "prefill_total_positions": total_pf_pos,
    }

    return {
        "label": label,
        "conf_path": conf_path,
        "cfg": cfg,
        "model_name": model_name,
        "prompt_results": prompt_results,
        "duration": duration,
        "aggregates": aggregates,
    }


# ── Single-Config HTML Report ──────────────────────────────────────

# CSS shared between single and comparative reports
REPORT_CSS = '''
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px 40px;
    line-height: 1.6;
}
h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
h2 { color: #79c0ff; margin-top: 32px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
h3 { color: #d2a8ff; }
.meta { color: #8b949e; margin-bottom: 20px; }
.meta span { margin-right: 24px; }

/* Summary dashboard */
.dashboard { width: 100%; border-collapse: collapse; margin: 16px 0 32px; }
.dashboard th {
    background: #161b22; color: #8b949e; text-align: left;
    padding: 8px 12px; border-bottom: 2px solid #30363d; font-size: 13px;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.dashboard td { padding: 8px 12px; border-bottom: 1px solid #21262d; }
.dashboard tr:hover td { background: #161b22; }

/* Status badges */
.badge { padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; display: inline-block; }
.badge-pass { background: #1a3a2a; color: #3fb950; }
.badge-fail { background: #3a1a1a; color: #f85149; }
.badge-warn { background: #3a2a1a; color: #d29922; }

/* Collapsible cards */
details { margin: 8px 0; }
summary {
    cursor: pointer; padding: 8px 12px; background: #161b22;
    border: 1px solid #30363d; border-radius: 6px; color: #c9d1d9;
    font-weight: 600; user-select: none;
}
summary:hover { background: #1c2128; }
details[open] > summary { border-radius: 6px 6px 0 0; border-bottom: none; }
.detail-body {
    border: 1px solid #30363d; border-top: none; border-radius: 0 0 6px 6px;
    padding: 16px; background: #0d1117;
}

/* Token table */
table.tokens { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 12px; font-family: monospace; }
table.tokens th {
    background: #161b22; color: #8b949e; text-align: left;
    padding: 4px 8px; border-bottom: 1px solid #30363d; font-size: 11px;
}
table.tokens td { padding: 4px 8px; border-bottom: 1px solid #21262d; }
table.tokens .match { color: #3fb950; }
table.tokens .mismatch { color: #f85149; }
table.tokens .diverged { background: #2a1f0a; }

/* Metrics row */
.metrics-row { display: flex; gap: 24px; margin: 8px 0 16px; flex-wrap: wrap; }
.metric-item { font-size: 13px; }
.metric-label { color: #8b949e; }
.metric-value { color: #c9d1d9; font-weight: 600; }
.metric-good { color: #3fb950; }
.metric-bad { color: #f85149; }
.metric-ok { color: #d29922; }

/* Text comparison */
.text-compare { margin: 12px 0; }
.text-block { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 12px 16px; margin: 6px 0; font-size: 13px; white-space: pre-wrap;
    max-height: 200px; overflow-y: auto; }
.text-label { color: #8b949e; font-size: 12px; margin-bottom: 4px; }
.text-match { color: #3fb950; }
.text-diverge { color: #d29922; background: #2a1f0a; padding: 0 2px; }
.text-after { color: #8b949e; }

/* Divergence analysis */
.divergence { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 12px 16px; margin: 12px 0; font-size: 13px; }
.divergence .label { color: #8b949e; display: inline-block; width: 180px; }

pre { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 12px 16px; overflow-x: auto; font-size: 13px; color: #e6edf3; }
'''


def _render_prompt_card(pr: Dict, badge_class: Dict) -> str:
    """Render a single prompt result card (shared between single and comparative reports)."""
    h = []
    m = pr["metrics"]
    verdict = pr["verdict"]
    prompt_text = pr["prompt"][:80]
    turn_label = f" (Turn {pr['turn']})" if pr.get("turn", 1) > 1 or pr.get("multi_turn", False) else ""
    conv_label = f"Conv {pr['conv_idx']+1}" if "conv_idx" in pr else ""

    # Expanded by default for FAIL/WARN
    open_attr = " open" if verdict != "PASS" else ""

    h.append(f'<details{open_attr}>')
    h.append(f'<summary><span class="badge {badge_class[verdict]}">{verdict}</span> '
             f'{_html_escape(conv_label)}{_html_escape(turn_label)}: '
             f'{_html_escape(prompt_text)}...</summary>')
    h.append('<div class="detail-body">')

    # Metrics row
    h.append('<div class="metrics-row">')
    pm = pr.get("prefill_metrics", {})
    pc = pm.get("prefill_containment", 0)
    pa = pm.get("prefill_argmax_rate", 0)
    pc_class = "metric-good" if pc >= 0.80 else ("metric-ok" if pc >= 0.60 else "metric-bad")
    pa_class = "metric-good" if pa >= 0.50 else ("metric-ok" if pa >= 0.30 else "metric-bad")
    h.append(f'<div class="metric-item"><span class="metric-label">Prefill argmax: </span>'
             f'<span class="metric-value {pa_class}">{100*pa:.0f}%</span></div>')
    h.append(f'<div class="metric-item"><span class="metric-label">Prefill top-10: </span>'
             f'<span class="metric-value {pc_class}">{100*pc:.0f}%</span></div>')

    fm_class = "metric-good" if m["first_match"] else "metric-bad"
    fm_label = "MATCH" if m["first_match"] else "MISMATCH"
    h.append(f'<div class="metric-item"><span class="metric-label">First token: </span>'
             f'<span class="metric-value {fm_class}">{fm_label}</span></div>')

    mr_class = "metric-good" if m["match_run"] >= 15 else ("metric-ok" if m["match_run"] >= 5 else "metric-bad")
    h.append(f'<div class="metric-item"><span class="metric-label">Match run: </span>'
             f'<span class="metric-value {mr_class}">{m["match_run"]}/{m["ref_tokens_count"]}</span></div>')

    ct_class = "metric-good" if m["containment_rate"] >= 0.90 else ("metric-ok" if m["containment_rate"] >= 0.75 else "metric-bad")
    h.append(f'<div class="metric-item"><span class="metric-label">Decode top-k: </span>'
             f'<span class="metric-value {ct_class}">{100*m["containment_rate"]:.1f}%</span></div>')

    if pr.get("timing"):
        t = pr["timing"]
        h.append(f'<div class="metric-item"><span class="metric-label">Time: </span>'
                 f'<span class="metric-value">{t.get("total_ms", 0)/1000:.1f}s</span></div>')
    h.append('</div>')

    # Token comparison table
    h.append('<table class="tokens"><thead><tr>')
    h.append('<th>Pos</th><th>Ref Token</th><th>Our Token</th><th>Match</th><th>In Top-K</th><th>Rank</th><th>Ref LP</th><th>Our LP</th>')
    h.append('</tr></thead><tbody>')

    for row in m["token_table"]:
        row_class = ""
        if row["pos"] == m.get("divergence_pos") and m.get("divergence_pos") is not None:
            row_class = ' class="diverged"'
        match_sym = '<span class="match">&#10003;</span>' if row["match"] else '<span class="mismatch">&#10007;</span>'
        topk_sym = f'<span class="match">#{row["ref_rank"]}</span>' if row["in_topk"] else '<span class="mismatch">-</span>'

        ref_lp = f'{row["ref_logprob"]:.3f}' if row["ref_logprob"] is not None else "-"
        our_lp = f'{row["our_logprob"]:.3f}' if row["our_logprob"] is not None else "-"

        h.append(f'<tr{row_class}>')
        h.append(f'<td>{row["pos"]}</td>')
        h.append(f'<td>{row["ref_token"]}</td>')
        h.append(f'<td>{row["our_token"]}</td>')
        h.append(f'<td>{match_sym}</td>')
        h.append(f'<td>{topk_sym}</td>')
        h.append(f'<td>{row["ref_rank"] or "-"}</td>')
        h.append(f'<td>{ref_lp}</td>')
        h.append(f'<td>{our_lp}</td>')
        h.append('</tr>')

    h.append('</tbody></table>')

    # Divergence analysis
    if m["divergence_info"]:
        di = m["divergence_info"]
        h.append('<div class="divergence">')
        h.append(f'<div><span class="label">Divergence position:</span> {di["position"]}</div>')
        if di["ref_logprob"] is not None:
            h.append(f'<div><span class="label">Reference token:</span> {di["ref_token"]} (logprob: {di["ref_logprob"]:.3f})</div>')
        if di["our_logprob"] is not None:
            h.append(f'<div><span class="label">Our token:</span> {di["our_token"]} (logprob: {di["our_logprob"]:.3f})</div>')
        in_topk = f'Yes (rank #{di["ref_rank_in_our_topk"]})' if di["ref_in_our_topk"] else 'No'
        h.append(f'<div><span class="label">Ref token in our top-k:</span> {in_topk}</div>')
        if di["ref_logprob"] is not None and di["our_logprob"] is not None:
            gap = abs(di["our_logprob"] - di["ref_logprob"])
            h.append(f'<div><span class="label">Logprob gap:</span> {gap:.3f}</div>')
        h.append('</div>')

    # Text comparison
    h.append('<details><summary>Full text output</summary>')
    h.append('<div class="detail-body">')
    h.append('<div class="text-compare">')
    h.append('<div class="text-label">Reference (BF16):</div>')
    h.append(f'<div class="text-block">{_html_escape(m["ref_text"])}</div>')
    h.append('<div class="text-label">Our engine:</div>')
    h.append(f'<div class="text-block">{_html_escape(m["our_text"])}</div>')
    h.append('</div>')
    h.append('</div></details>')

    h.append('</div></details>')
    return "\n".join(h)


def generate_html_report(
    model_name: str,
    conf_path: str,
    ref_path: str,
    ref_data: Dict,
    prompt_results: List[Dict],
    overall_verdict: str,
    gpu_info: str,
    total_duration: float,
) -> str:
    """Generate the full HTML report for a single config."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    h = []
    badge_class = {"PASS": "badge-pass", "WARN": "badge-warn", "FAIL": "badge-fail"}

    h.append('<!DOCTYPE html>')
    h.append('<html lang="en"><head><meta charset="UTF-8">')
    h.append(f'<title>Reference Test — {_html_escape(model_name)}</title>')
    h.append(f'<style>{REPORT_CSS}</style></head><body>')

    # ── Header ──
    h.append(f'<h1>Krasis Reference Test — {_html_escape(model_name)}</h1>')
    h.append('<div class="meta">')
    h.append(f'<span>Date: {now}</span>')
    h.append(f'<span>GPU: {_html_escape(gpu_info)}</span>')
    h.append(f'<span>Config: {_html_escape(os.path.basename(conf_path))}</span>')
    h.append(f'<span>Duration: {total_duration:.1f}s</span>')
    h.append('</div>')
    h.append('<div class="meta">')
    h.append(f'<span>Reference: {_html_escape(ref_data.get("model", "?"))} (format v{ref_data.get("format_version", "?")})</span>')
    h.append(f'<span>Conversations: {len(ref_data.get("conversations", []))}</span>')
    h.append(f'<span>Max tokens: {ref_data.get("max_new_tokens", "?")}</span>')
    h.append('</div>')

    # ── Summary Dashboard ──
    h.append('<h2>Summary</h2>')

    prompt_verdicts = [r["verdict"] for r in prompt_results]
    n_pass = sum(1 for v in prompt_verdicts if v == "PASS")
    n_warn = sum(1 for v in prompt_verdicts if v == "WARN")
    n_fail = sum(1 for v in prompt_verdicts if v == "FAIL")
    total_prompts = len(prompt_results)

    # Aggregate metrics
    first_match_count = sum(1 for r in prompt_results if r["metrics"]["first_match"])
    avg_match_run = sum(r["metrics"]["match_run"] for r in prompt_results) / total_prompts if total_prompts else 0
    avg_containment = sum(r["metrics"]["containment_rate"] for r in prompt_results) / total_prompts if total_prompts else 0
    # Prefill aggregates
    total_pf_argmax = sum(r.get("prefill_metrics", {}).get("prefill_argmax_match", 0) for r in prompt_results)
    total_pf_top10 = sum(r.get("prefill_metrics", {}).get("prefill_top10_hits", 0) for r in prompt_results)
    total_pf_pos = sum(r.get("prefill_metrics", {}).get("prefill_total", 0) for r in prompt_results)
    avg_pf_arg = total_pf_argmax / total_pf_pos if total_pf_pos > 0 else 0
    avg_pf_cont = total_pf_top10 / total_pf_pos if total_pf_pos > 0 else 0

    h.append('<table class="dashboard"><thead><tr>')
    h.append('<th>Metric</th><th>Value</th><th>Status</th>')
    h.append('</tr></thead><tbody>')

    h.append(f'<tr><td><b>Overall</b></td><td>—</td><td><span class="badge {badge_class[overall_verdict]}">{overall_verdict}</span></td></tr>')

    # Primary metric: prefill containment
    h.append(f'<tr><td><b>Prefill argmax match</b></td><td>{total_pf_argmax}/{total_pf_pos} ({100*avg_pf_arg:.0f}%)</td>')
    pa_status = "PASS" if avg_pf_arg >= 0.50 else ("WARN" if avg_pf_arg >= 0.30 else "FAIL")
    h.append(f'<td><span class="badge {badge_class[pa_status]}">{pa_status}</span></td></tr>')

    h.append(f'<tr><td><b>Prefill top-10 containment</b></td><td>{total_pf_top10}/{total_pf_pos} ({100*avg_pf_cont:.0f}%)</td>')
    pc_status = "PASS" if avg_pf_cont >= 0.80 else ("WARN" if avg_pf_cont >= 0.60 else "FAIL")
    h.append(f'<td><span class="badge {badge_class[pc_status]}">{pc_status}</span></td></tr>')

    h.append(f'<tr><td>First-token match</td><td>{first_match_count}/{total_prompts} ({100*first_match_count/total_prompts:.0f}%)</td>')
    fm_status = "PASS" if first_match_count >= total_prompts - 2 else ("WARN" if first_match_count >= total_prompts // 2 else "FAIL")
    h.append(f'<td><span class="badge {badge_class[fm_status]}">{fm_status}</span></td></tr>')

    h.append(f'<tr><td>Avg match run length</td><td>{avg_match_run:.1f} tokens</td>')
    mr_status = "PASS" if avg_match_run >= 15 else ("WARN" if avg_match_run >= 5 else "FAIL")
    h.append(f'<td><span class="badge {badge_class[mr_status]}">{mr_status}</span></td></tr>')

    h.append(f'<tr><td>Avg decode top-k containment</td><td>{100*avg_containment:.1f}%</td>')
    ct_status = "PASS" if avg_containment >= 0.90 else ("WARN" if avg_containment >= 0.75 else "FAIL")
    h.append(f'<td><span class="badge {badge_class[ct_status]}">{ct_status}</span></td></tr>')

    h.append(f'<tr><td>Prompts PASS</td><td>{n_pass}</td><td></td></tr>')
    h.append(f'<tr><td>Prompts WARN</td><td>{n_warn}</td><td></td></tr>')
    h.append(f'<tr><td>Prompts FAIL</td><td>{n_fail}</td><td></td></tr>')

    h.append('</tbody></table>')

    # ── Per-Prompt Cards ──
    h.append('<h2>Per-Prompt Results</h2>')

    for pr in prompt_results:
        h.append(_render_prompt_card(pr, badge_class))

    # ── Footer ──
    h.append(f'<div class="meta" style="margin-top:32px;border-top:1px solid #30363d;padding-top:12px">')
    h.append(f'<span>Generated by Krasis reference-test</span>')
    h.append(f'<span>Total time: {total_duration:.1f}s</span>')
    h.append('</div>')
    h.append('</body></html>')

    return "\n".join(h)


# ── Comparison SVG Charts ──────────────────────────────────────────

def generate_comparison_svg(all_results: List[Dict]) -> str:
    """Generate SVG grouped bar chart comparing key metrics across configs."""
    metrics_defs = [
        ("Prefill Argmax", "prefill_argmax_rate"),
        ("Prefill Top-10", "prefill_containment"),
        ("First-Token", "first_match_rate"),
        ("Decode Top-K", "avg_containment"),
    ]

    n_configs = len(all_results)
    n_metrics = len(metrics_defs)

    width = 780
    height = 340
    ml, mr, mt, mb = 55, 20, 30, 70
    chart_w = width - ml - mr
    chart_h = height - mt - mb

    group_w = chart_w / n_metrics
    bar_w = min(40, (group_w - 24) / max(n_configs, 1))
    bar_gap = 3

    config_labels = [r["label"] for r in all_results]
    colors = [CONFIG_COLORS.get(label, "#8b949e") for label in config_labels]

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
               f' style="background:#161b22;border-radius:8px;margin:16px 0">')

    # Y gridlines and labels
    for pct in [0, 25, 50, 75, 100]:
        y = mt + chart_h * (1 - pct / 100)
        svg.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" '
                   f'stroke="#30363d" stroke-dasharray="4"/>')
        svg.append(f'<text x="{ml-6}" y="{y+4:.1f}" text-anchor="end" '
                   f'fill="#8b949e" font-size="11" font-family="monospace">{pct}%</text>')

    # Bars
    for gi, (metric_label, metric_key) in enumerate(metrics_defs):
        total_bar_w = n_configs * (bar_w + bar_gap) - bar_gap
        group_x = ml + gi * group_w + (group_w - total_bar_w) / 2

        for ci, result in enumerate(all_results):
            value = result["aggregates"].get(metric_key, 0)
            bar_h = max(chart_h * value, 0)
            bx = group_x + ci * (bar_w + bar_gap)
            by = mt + chart_h - bar_h

            svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" '
                       f'height="{max(bar_h, 0.5):.1f}" fill="{colors[ci]}" rx="2"/>')

            # Value label
            if value > 0.04:
                svg.append(f'<text x="{bx + bar_w/2:.1f}" y="{by - 4:.1f}" '
                           f'text-anchor="middle" fill="#c9d1d9" font-size="10" '
                           f'font-family="monospace">{100*value:.0f}%</text>')

        # X label
        label_x = ml + gi * group_w + group_w / 2
        svg.append(f'<text x="{label_x:.1f}" y="{mt + chart_h + 18:.1f}" '
                   f'text-anchor="middle" fill="#c9d1d9" font-size="11" '
                   f'font-family="monospace">{metric_label}</text>')

    # Legend
    legend_y = height - 16
    total_legend_w = sum(len(label) * 7 + 30 for label in config_labels)
    legend_x = ml + (chart_w - total_legend_w) / 2
    for ci, label in enumerate(config_labels):
        svg.append(f'<rect x="{legend_x:.1f}" y="{legend_y - 9}" width="12" height="12" '
                   f'fill="{colors[ci]}" rx="2"/>')
        svg.append(f'<text x="{legend_x + 16:.1f}" y="{legend_y:.1f}" '
                   f'fill="#c9d1d9" font-size="11" font-family="monospace">{label}</text>')
        legend_x += len(label) * 7 + 36

    svg.append('</svg>')
    return "\n".join(svg)


def generate_divergence_svg(all_results: List[Dict]) -> str:
    """Generate SVG scatter chart showing where each config first diverges per prompt."""
    n_configs = len(all_results)
    if n_configs == 0:
        return ""

    n_prompts = len(all_results[0]["prompt_results"])
    if n_prompts == 0:
        return ""

    width = 780
    height = 300
    ml, mr, mt, mb = 55, 20, 30, 60
    chart_w = width - ml - mr
    chart_h = height - mt - mb

    config_labels = [r["label"] for r in all_results]
    colors = [CONFIG_COLORS.get(label, "#8b949e") for label in config_labels]

    # Find max tokens for Y scale
    max_tokens = max(
        r["metrics"]["ref_tokens_count"]
        for res in all_results for r in res["prompt_results"]
    )
    if max_tokens == 0:
        max_tokens = 200

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
               f' style="background:#161b22;border-radius:8px;margin:16px 0">')

    # Title
    svg.append(f'<text x="{width/2}" y="18" text-anchor="middle" fill="#79c0ff" '
               f'font-size="13" font-family="monospace">Token Match Run Length by Prompt</text>')

    # Y gridlines
    for tok in range(0, max_tokens + 1, 50):
        y = mt + chart_h * (1 - tok / max_tokens)
        svg.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" '
                   f'stroke="#30363d" stroke-dasharray="4"/>')
        svg.append(f'<text x="{ml-6}" y="{y+4:.1f}" text-anchor="end" '
                   f'fill="#8b949e" font-size="10" font-family="monospace">{tok}</text>')

    # Points and lines for each config
    prompt_spacing = chart_w / max(n_prompts - 1, 1) if n_prompts > 1 else chart_w
    for ci, result in enumerate(all_results):
        points = []
        for pi, pr in enumerate(result["prompt_results"]):
            match_run = pr["metrics"]["match_run"]
            px = ml + pi * prompt_spacing
            py = mt + chart_h * (1 - match_run / max_tokens)
            points.append((px, py))

            # Draw dot
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{colors[ci]}" '
                       f'stroke="#0d1117" stroke-width="1"/>')

        # Connect dots with line
        if len(points) > 1:
            path_d = f'M {points[0][0]:.1f} {points[0][1]:.1f}'
            for px, py in points[1:]:
                path_d += f' L {px:.1f} {py:.1f}'
            svg.append(f'<path d="{path_d}" stroke="{colors[ci]}" stroke-width="1.5" '
                       f'fill="none" opacity="0.6"/>')

    # X axis labels (prompt numbers)
    for pi in range(n_prompts):
        px = ml + pi * prompt_spacing
        svg.append(f'<text x="{px:.1f}" y="{mt + chart_h + 16:.1f}" '
                   f'text-anchor="middle" fill="#8b949e" font-size="10" '
                   f'font-family="monospace">{pi+1}</text>')
    svg.append(f'<text x="{ml + chart_w/2:.1f}" y="{mt + chart_h + 32:.1f}" '
               f'text-anchor="middle" fill="#8b949e" font-size="11" '
               f'font-family="monospace">Prompt</text>')

    # Legend
    legend_y = height - 10
    total_legend_w = sum(len(label) * 7 + 30 for label in config_labels)
    legend_x = ml + (chart_w - total_legend_w) / 2
    for ci, label in enumerate(config_labels):
        svg.append(f'<rect x="{legend_x:.1f}" y="{legend_y - 9}" width="12" height="12" '
                   f'fill="{colors[ci]}" rx="2"/>')
        svg.append(f'<text x="{legend_x + 16:.1f}" y="{legend_y:.1f}" '
                   f'fill="#c9d1d9" font-size="11" font-family="monospace">{label}</text>')
        legend_x += len(label) * 7 + 36

    svg.append('</svg>')
    return "\n".join(svg)


# ── Comparative HTML Report ────────────────────────────────────────

def generate_comparative_html_report(
    model_name: str,
    ref_data: Dict,
    all_results: List[Dict],
    gpu_info: str,
    total_duration: float,
) -> str:
    """Generate HTML report comparing multiple configs side by side."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    badge_class = {"PASS": "badge-pass", "WARN": "badge-warn", "FAIL": "badge-fail"}
    h = []

    h.append('<!DOCTYPE html>')
    h.append('<html lang="en"><head><meta charset="UTF-8">')
    h.append(f'<title>Reference Test Comparison — {_html_escape(model_name)}</title>')
    h.append(f'<style>{REPORT_CSS}')
    # Additional styles for comparison
    h.append('''
.compare-table { width: 100%; border-collapse: collapse; margin: 16px 0 32px; }
.compare-table th {
    background: #161b22; color: #8b949e; text-align: center;
    padding: 8px 12px; border-bottom: 2px solid #30363d; font-size: 13px;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.compare-table th:first-child { text-align: left; }
.compare-table td { padding: 8px 12px; border-bottom: 1px solid #21262d; text-align: center; }
.compare-table td:first-child { text-align: left; color: #c9d1d9; }
.compare-table tr:hover td { background: #161b22; }
.compare-table .best { font-weight: 700; }
.prompt-compare { margin: 12px 0; }
.prompt-compare-header {
    display: grid; gap: 8px; margin: 4px 0;
    font-size: 12px; font-family: monospace;
}
''')
    h.append('</style></head><body>')

    # ── Header ──
    h.append(f'<h1>Krasis Reference Test Comparison — {_html_escape(model_name)}</h1>')
    h.append('<div class="meta">')
    h.append(f'<span>Date: {now}</span>')
    h.append(f'<span>GPU: {_html_escape(gpu_info)}</span>')
    h.append(f'<span>Configs: {len(all_results)}</span>')
    h.append(f'<span>Duration: {total_duration:.1f}s</span>')
    h.append('</div>')
    h.append('<div class="meta">')
    h.append(f'<span>Reference: {_html_escape(ref_data.get("model", "?"))} (format v{ref_data.get("format_version", "?")})</span>')
    h.append(f'<span>Conversations: {len(ref_data.get("conversations", []))}</span>')
    h.append(f'<span>Max tokens: {ref_data.get("max_new_tokens", "?")}</span>')
    h.append('</div>')

    # ── Summary Comparison Table ──
    h.append('<h2>Summary Comparison</h2>')

    config_labels = [r["label"] for r in all_results]
    config_colors = [CONFIG_COLORS.get(label, "#8b949e") for label in config_labels]

    h.append('<table class="compare-table"><thead><tr>')
    h.append('<th>Metric</th>')
    for i, label in enumerate(config_labels):
        h.append(f'<th style="border-top: 3px solid {config_colors[i]}">{_html_escape(label)}</th>')
    h.append('</tr></thead><tbody>')

    # Row: Overall verdict
    h.append('<tr><td><b>Overall</b></td>')
    for r in all_results:
        v = r["aggregates"]["overall"]
        h.append(f'<td><span class="badge {badge_class[v]}">{v}</span></td>')
    h.append('</tr>')

    # Helper to find best value and mark it
    def _metric_row(label: str, values: List[float], fmt: str = "{:.0f}%", scale: float = 100,
                    raw_strs: Optional[List[str]] = None):
        h.append(f'<tr><td>{label}</td>')
        if not values:
            h.append('</tr>')
            return
        best = max(values)
        for i, v in enumerate(values):
            cls = ' class="best"' if v == best and values.count(best) < len(values) else ''
            display = raw_strs[i] if raw_strs else fmt.format(v * scale)
            h.append(f'<td{cls}>{display}</td>')
        h.append('</tr>')

    # Prefill argmax
    vals = [r["aggregates"]["prefill_argmax_rate"] for r in all_results]
    raw = [f'{r["aggregates"]["prefill_argmax_count"]}/{r["aggregates"]["prefill_total_positions"]} '
           f'({100*r["aggregates"]["prefill_argmax_rate"]:.0f}%)' for r in all_results]
    _metric_row("<b>Prefill argmax match</b>", vals, raw_strs=raw)

    # Prefill top-10
    vals = [r["aggregates"]["prefill_containment"] for r in all_results]
    raw = [f'{r["aggregates"]["prefill_top10_count"]}/{r["aggregates"]["prefill_total_positions"]} '
           f'({100*r["aggregates"]["prefill_containment"]:.0f}%)' for r in all_results]
    _metric_row("<b>Prefill top-10 containment</b>", vals, raw_strs=raw)

    # First-token match
    vals = [r["aggregates"]["first_match_rate"] for r in all_results]
    raw = [f'{r["aggregates"]["first_match_count"]}/{r["aggregates"]["total_prompts"]} '
           f'({100*r["aggregates"]["first_match_rate"]:.0f}%)' for r in all_results]
    _metric_row("First-token match", vals, raw_strs=raw)

    # Match run
    vals = [r["aggregates"]["avg_match_run"] for r in all_results]
    raw = [f'{v:.1f} tokens' for v in vals]
    _metric_row("Avg match run", vals, raw_strs=raw)

    # Decode top-k
    vals = [r["aggregates"]["avg_containment"] for r in all_results]
    raw = [f'{100*v:.1f}%' for v in vals]
    _metric_row("Avg decode top-k containment", vals, raw_strs=raw)

    # Prompts
    h.append('<tr><td>Prompts PASS / WARN / FAIL</td>')
    for r in all_results:
        a = r["aggregates"]
        h.append(f'<td>{a["n_pass"]} / {a["n_warn"]} / {a["n_fail"]}</td>')
    h.append('</tr>')

    # Duration
    h.append('<tr><td>Duration</td>')
    for r in all_results:
        h.append(f'<td>{r["duration"]:.0f}s</td>')
    h.append('</tr>')

    h.append('</tbody></table>')

    # ── SVG Charts ──
    h.append('<h2>Metric Comparison</h2>')
    h.append(generate_comparison_svg(all_results))

    h.append('<h2>Divergence Analysis</h2>')
    h.append(generate_divergence_svg(all_results))

    # ── Per-Prompt Comparison Grid ──
    h.append('<h2>Per-Prompt Results</h2>')

    # Build prompt list from first config
    n_prompts = len(all_results[0]["prompt_results"])

    # Quick overview table
    h.append('<table class="compare-table"><thead><tr>')
    h.append('<th>Prompt</th>')
    for label in config_labels:
        h.append(f'<th>{_html_escape(label)}</th>')
    h.append('</tr></thead><tbody>')

    for pi in range(n_prompts):
        pr0 = all_results[0]["prompt_results"][pi]
        conv_label = f"C{pr0['conv_idx']+1}"
        if pr0.get("multi_turn"):
            conv_label += f"T{pr0['turn']}"
        prompt_short = pr0["prompt"][:40]

        h.append(f'<tr><td>{conv_label}: {_html_escape(prompt_short)}...</td>')
        for result in all_results:
            pr = result["prompt_results"][pi]
            v = pr["verdict"]
            m = pr["metrics"]
            pm = pr.get("prefill_metrics", {})
            pa = pm.get("prefill_argmax_rate", 0)
            pc = pm.get("prefill_containment", 0)
            h.append(f'<td><span class="badge {badge_class[v]}">{v}</span><br>'
                     f'<span style="font-size:11px;color:#8b949e">'
                     f'pf:{100*pa:.0f}%/{100*pc:.0f}% run:{m["match_run"]}</span></td>')
        h.append('</tr>')

    h.append('</tbody></table>')

    # ── Per-Config Detailed Cards ──
    h.append('<h2>Detailed Per-Config Results</h2>')

    for result in all_results:
        label = result["label"]
        color = CONFIG_COLORS.get(label, "#8b949e")
        a = result["aggregates"]
        v = a["overall"]

        h.append(f'<details>')
        h.append(f'<summary style="border-left:4px solid {color}">'
                 f'<span class="badge {badge_class[v]}">{v}</span> '
                 f'{_html_escape(label)} — '
                 f'{_html_escape(os.path.basename(result["conf_path"]))} '
                 f'({a["n_pass"]}P/{a["n_warn"]}W/{a["n_fail"]}F, '
                 f'prefill {100*a["prefill_argmax_rate"]:.0f}%/{100*a["prefill_containment"]:.0f}%)'
                 f'</summary>')
        h.append('<div class="detail-body">')

        for pr in result["prompt_results"]:
            h.append(_render_prompt_card(pr, badge_class))

        h.append('</div></details>')

    # ── Footer ──
    h.append('<div class="meta" style="margin-top:32px;border-top:1px solid #30363d;padding-top:12px">')
    h.append(f'<span>Generated by Krasis reference-test --compare</span>')
    h.append(f'<span>Total time: {total_duration:.1f}s</span>')
    h.append('</div>')
    h.append('</body></html>')

    return "\n".join(h)


# ── Main ────────────────────────────────────────────────────────────

def main():
    if not os.environ.get("KRASIS_DEV_SCRIPT"):
        print("ERROR: This script must be run via ./dev reference-test, not directly.", file=sys.stderr)
        print("Usage: ./dev reference-test <config>", file=sys.stderr)
        print("       ./dev reference-test --compare <model>", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Krasis reference test")
    parser.add_argument("config", nargs="?", help="Config file path (single-config mode)")
    parser.add_argument("--compare", metavar="MODEL",
                        help="Run comparative test with all configs for MODEL (e.g. qcn, q35b, q122b)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    parser.add_argument("--no-server", action="store_true",
                        help="Don't start server, connect to already-running one")
    parser.add_argument("--port", type=int, default=0,
                        help="Port override (for --no-server)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Get GPU info
    try:
        gpu_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True
        ).strip().split("\n")[0]
    except Exception:
        gpu_out = "Unknown GPU"

    if args.compare:
        # ── Comparison Mode ──
        model_key = args.compare.lower()
        if model_key not in COMPARE_CONFIGS:
            available = ", ".join(sorted(COMPARE_CONFIGS.keys()))
            print(f"ERROR: Unknown model '{args.compare}' for comparison.", file=sys.stderr)
            print(f"Available models: {available}", file=sys.stderr)
            sys.exit(1)

        config_list = COMPARE_CONFIGS[model_key]

        # Validate configs exist
        valid_configs = []
        for label, rel_path in config_list:
            full_path = os.path.join(script_dir, rel_path)
            if os.path.isfile(full_path):
                valid_configs.append((label, full_path))
            else:
                print(f"WARNING: Config not found, skipping: {rel_path}", file=sys.stderr)

        if not valid_configs:
            print("ERROR: No valid configs found for comparison.", file=sys.stderr)
            sys.exit(1)

        # Find reference data from the first config's model
        first_cfg = parse_config(valid_configs[0][1])
        model_name = resolve_model_name(first_cfg)
        ref_path = find_reference_data(model_name, script_dir)
        if ref_path is None:
            available = list_available_references(script_dir)
            print(f"ERROR: No reference data found for model '{model_name}'", file=sys.stderr)
            print(f"Available reference data: {', '.join(available)}", file=sys.stderr)
            sys.exit(1)

        print(f"Comparative reference test: {model_name}")
        print(f"Reference data: {ref_path}")
        print(f"Configs to test: {', '.join(label for label, _ in valid_configs)}")

        with open(ref_path) as f:
            ref_data = json.load(f)

        if ref_data.get("format_version", 0) < 3:
            print(f"ERROR: Reference data format_version must be >= 3, got {ref_data.get('format_version')}", file=sys.stderr)
            sys.exit(1)

        t_total_start = time.time()
        all_results = []

        for ci, (label, conf_path) in enumerate(valid_configs):
            if ci > 0:
                print("\nCleaning up between configs...")
                cleanup_between_configs()

            result = run_config_test(conf_path, ref_data, script_dir,
                                     label=label, verbose=args.verbose)
            if result is not None:
                all_results.append(result)
            else:
                print(f"WARNING: Config {label} failed, skipping.", file=sys.stderr)

        total_duration = time.time() - t_total_start

        if not all_results:
            print("ERROR: All configs failed.", file=sys.stderr)
            sys.exit(1)

        # Generate comparative HTML report
        html = generate_comparative_html_report(
            model_name=model_name,
            ref_data=ref_data,
            all_results=all_results,
            gpu_info=gpu_out,
            total_duration=total_duration,
        )

        # Save report
        report_path = os.path.join(get_run_dir("reference-test"), "reference_compare.html")
        with open(report_path, "w") as f:
            f.write(html)

        # Print summary
        print(f"\n{'='*60}")
        print(f"COMPARATIVE REFERENCE TEST")
        print(f"{'='*60}")
        print(f"  Model: {model_name}")
        print(f"  Configs tested: {len(all_results)}/{len(valid_configs)}")
        print()

        # Summary table
        header = f"  {'Metric':<28}"
        for r in all_results:
            header += f" {r['label']:>12}"
        print(header)
        print(f"  {'-'*28}" + "".join(f" {'-'*12}" for _ in all_results))

        print(f"  {'Overall':<28}" + "".join(f" {r['aggregates']['overall']:>12}" for r in all_results))
        print(f"  {'Prefill argmax':<28}" +
              "".join(f" {100*r['aggregates']['prefill_argmax_rate']:>11.0f}%" for r in all_results))
        print(f"  {'Prefill top-10':<28}" +
              "".join(f" {100*r['aggregates']['prefill_containment']:>11.0f}%" for r in all_results))
        print(f"  {'First-token match':<28}" +
              "".join(f" {r['aggregates']['first_match_count']:>4}/{r['aggregates']['total_prompts']:<7}" for r in all_results))
        print(f"  {'Avg match run':<28}" +
              "".join(f" {r['aggregates']['avg_match_run']:>10.1f}t" for r in all_results))
        print(f"  {'Decode top-k':<28}" +
              "".join(f" {100*r['aggregates']['avg_containment']:>11.1f}%" for r in all_results))
        print(f"  {'Duration':<28}" +
              "".join(f" {r['duration']:>11.0f}s" for r in all_results))
        print()
        print(f"  Total duration: {total_duration:.0f}s")
        print(f"  Report: {report_path}")
        print()

        # Exit code based on worst config
        worst = "PASS"
        for r in all_results:
            v = r["aggregates"]["overall"]
            if v == "FAIL":
                worst = "FAIL"
                break
            if v == "WARN":
                worst = "WARN"

        if worst == "FAIL":
            sys.exit(1)
        elif worst == "WARN":
            sys.exit(2)
        else:
            sys.exit(0)

    else:
        # ── Single Config Mode (original behavior) ──
        if not args.config:
            parser.error("config is required in single-config mode (or use --compare)")

        # Parse config
        cfg = parse_config(args.config)
        model_name = resolve_model_name(cfg)
        port = int(cfg.get("CFG_PORT", "8012"))
        if args.port:
            port = args.port

        print(f"Reference test: {model_name}")
        print(f"Config: {args.config}")

        # Find reference data
        ref_path = find_reference_data(model_name, script_dir)
        if ref_path is None:
            available = list_available_references(script_dir)
            print(f"ERROR: No reference data found for model '{model_name}'", file=sys.stderr)
            print(f"Available reference data: {', '.join(available)}", file=sys.stderr)
            sys.exit(1)

        print(f"Reference data: {ref_path}")

        with open(ref_path) as f:
            ref_data = json.load(f)

        if ref_data.get("format_version", 0) < 3:
            print(f"ERROR: Reference data format_version must be >= 3, got {ref_data.get('format_version')}", file=sys.stderr)
            sys.exit(1)

        t_total_start = time.time()

        if args.no_server:
            # Run without starting server
            result = run_config_test.__wrapped__(args.config, ref_data, script_dir) if hasattr(run_config_test, '__wrapped__') else None
            # Actually, for --no-server, run the prompts directly against the port
            print(f"Using existing server on port {port}")

            eos_token_ids = ref_data.get("eos_token_ids", [])
            max_new_tokens = ref_data.get("max_new_tokens", 200)
            prompt_results = []

            for conv_idx, conv in enumerate(ref_data["conversations"]):
                turns = conv["turns"]
                for turn_idx, turn in enumerate(turns):
                    prompt_text = turn.get("prompt", "")
                    input_token_ids = turn["input_token_ids"]
                    n_input = len(input_token_ids)

                    label = f"Conv {conv_idx+1}"
                    if len(turns) > 1:
                        label += f" Turn {turn_idx+1}"
                    print(f"  {label}: {n_input} input tokens, prompt: {prompt_text[:60]}...")

                    try:
                        result = call_reference_test(
                            port=port,
                            input_token_ids=input_token_ids,
                            max_tokens=max_new_tokens,
                            stop_token_ids=eos_token_ids,
                            top_logprobs=10,
                        )
                    except Exception as e:
                        print(f"    ERROR: {e}", file=sys.stderr)
                        prompt_results.append({
                            "conv_idx": conv_idx, "turn": turn_idx + 1,
                            "multi_turn": len(turns) > 1, "prompt": prompt_text,
                            "verdict": "FAIL",
                            "metrics": {
                                "first_match": False, "match_run": 0,
                                "ref_tokens_count": len(turn["token_ids"]),
                                "our_tokens_count": 0, "containment_rate": 0.0,
                                "containment_hits": 0, "containment_total": 0,
                                "divergence_pos": 0, "divergence_info": None,
                                "token_table": [], "ref_text": turn.get("text", "")[:500],
                                "our_text": f"ERROR: {e}",
                            },
                            "prefill_metrics": {
                                "prefill_argmax_rate": 0.0, "prefill_containment": 0.0,
                                "prefill_argmax_match": 0, "prefill_top10_hits": 0, "prefill_total": 0,
                            },
                            "timing": None,
                        })
                        continue

                    metrics = compute_metrics(turn, result)
                    prefill_metrics = {
                        "prefill_argmax_rate": 0.0, "prefill_containment": 0.0,
                        "prefill_argmax_match": 0, "prefill_top10_hits": 0, "prefill_total": 0,
                    }
                    has_ref_logits = bool(turn.get("prefill_logits", {}).get("positions"))
                    if has_ref_logits:
                        try:
                            pl_result = call_prefill_logits(port, input_token_ids, top_k=10, sample_every=1)
                            prefill_metrics = compute_prefill_metrics(turn, pl_result)
                        except Exception as e:
                            print(f"    (prefill_logits failed: {e})")

                    verdict = judge_prompt(metrics, prefill_metrics)
                    timing = result.get("timing")

                    prompt_results.append({
                        "conv_idx": conv_idx, "turn": turn_idx + 1,
                        "multi_turn": len(turns) > 1, "prompt": prompt_text,
                        "verdict": verdict, "metrics": metrics,
                        "prefill_metrics": prefill_metrics, "timing": timing,
                    })

                    fm = "MATCH" if metrics["first_match"] else "MISS"
                    pc = prefill_metrics.get("prefill_containment", 0)
                    pa = prefill_metrics.get("prefill_argmax_rate", 0)
                    print(f"    {verdict}: first={fm}, run={metrics['match_run']}/{metrics['ref_tokens_count']}, "
                          f"prefill={100*pa:.0f}%/{100*pc:.0f}%, "
                          f"decode-top-k={100*metrics['containment_rate']:.0f}%")
        else:
            # Start server and run
            result = run_config_test(args.config, ref_data, script_dir, verbose=args.verbose)
            if result is None:
                print("ERROR: Test failed to run.", file=sys.stderr)
                sys.exit(1)
            prompt_results = result["prompt_results"]

        total_duration = time.time() - t_total_start

        # Overall verdict
        prompt_verdicts = [r["verdict"] for r in prompt_results]
        overall = judge_overall(prompt_verdicts)

        # Generate HTML report
        html = generate_html_report(
            model_name=model_name,
            conf_path=args.config,
            ref_path=ref_path,
            ref_data=ref_data,
            prompt_results=prompt_results,
            overall_verdict=overall,
            gpu_info=gpu_out,
            total_duration=total_duration,
        )

        # Save report
        report_path = os.path.join(get_run_dir("reference-test"), "reference_test.html")

        with open(report_path, "w") as f:
            f.write(html)

        # Print summary
        print()
        print(f"{'='*60}")
        print(f"REFERENCE TEST: {overall}")
        print(f"{'='*60}")
        n_pass = sum(1 for v in prompt_verdicts if v == "PASS")
        n_warn = sum(1 for v in prompt_verdicts if v == "WARN")
        n_fail = sum(1 for v in prompt_verdicts if v == "FAIL")
        first_match_count = sum(1 for r in prompt_results if r["metrics"]["first_match"])
        avg_run = sum(r["metrics"]["match_run"] for r in prompt_results) / len(prompt_results) if prompt_results else 0
        avg_cont = sum(r["metrics"]["containment_rate"] for r in prompt_results) / len(prompt_results) if prompt_results else 0

        # Prefill-level aggregates
        total_pf_argmax = sum(r.get("prefill_metrics", {}).get("prefill_argmax_match", 0) for r in prompt_results)
        total_pf_top10 = sum(r.get("prefill_metrics", {}).get("prefill_top10_hits", 0) for r in prompt_results)
        total_pf_pos = sum(r.get("prefill_metrics", {}).get("prefill_total", 0) for r in prompt_results)
        avg_pf_arg = total_pf_argmax / total_pf_pos if total_pf_pos > 0 else 0
        avg_pf_cont = total_pf_top10 / total_pf_pos if total_pf_pos > 0 else 0

        print(f"  Prompts: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL (of {len(prompt_results)})")
        print(f"  Prefill argmax match: {total_pf_argmax}/{total_pf_pos} ({100*avg_pf_arg:.0f}%)")
        print(f"  Prefill top-10 containment: {total_pf_top10}/{total_pf_pos} ({100*avg_pf_cont:.0f}%)")
        print(f"  First-token match: {first_match_count}/{len(prompt_results)}")
        print(f"  Avg match run: {avg_run:.1f} tokens")
        print(f"  Avg decode top-k: {100*avg_cont:.1f}%")
        print(f"  Duration: {total_duration:.1f}s")
        print(f"  Report: {report_path}")
        print()

        if overall == "FAIL":
            sys.exit(1)
        elif overall == "WARN":
            sys.exit(2)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
