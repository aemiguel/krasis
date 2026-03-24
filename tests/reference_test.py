#!/usr/bin/env python3
"""Reference test runner for Krasis.

Compares engine output against BF16 HuggingFace Transformers reference data.
Feeds raw input_token_ids directly to the prefill engine (no tokenization,
no chat template re-application) and compares greedy decode output.

Produces a self-contained HTML report with per-prompt diagnostics.

Usage:
    ./dev reference-test <config>

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

    proc = subprocess.Popen(
        [python, "-m", "krasis.server", "--config", conf_path, "--test-endpoints"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=script_dir,
    )
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


def judge_prompt(metrics: Dict) -> str:
    """Return PASS, WARN, or FAIL for a single prompt."""
    # FAIL conditions
    if not metrics["first_match"]:
        return "FAIL"
    if metrics["match_run"] < 5:
        return "FAIL"
    if metrics["containment_rate"] < 0.75:
        return "FAIL"

    # WARN conditions
    if metrics["match_run"] < 15:
        return "WARN"
    if metrics["containment_rate"] < 0.90:
        return "WARN"

    return "PASS"


def judge_overall(prompt_verdicts: List[str]) -> str:
    """Return overall PASS, WARN, or FAIL."""
    if any(v == "FAIL" for v in prompt_verdicts):
        return "FAIL"
    warn_count = sum(1 for v in prompt_verdicts if v == "WARN")
    if warn_count >= 3:
        return "WARN"
    return "PASS"


# ── HTML Report ─────────────────────────────────────────────────────

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
    """Generate the full HTML report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    h = []

    h.append('<!DOCTYPE html>')
    h.append('<html lang="en"><head><meta charset="UTF-8">')
    h.append(f'<title>Reference Test — {_html_escape(model_name)}</title>')
    h.append('<style>')
    h.append('''
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
''')
    h.append('</style></head><body>')

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

    badge_class = {"PASS": "badge-pass", "WARN": "badge-warn", "FAIL": "badge-fail"}

    h.append('<table class="dashboard"><thead><tr>')
    h.append('<th>Metric</th><th>Value</th><th>Status</th>')
    h.append('</tr></thead><tbody>')

    h.append(f'<tr><td><b>Overall</b></td><td>—</td><td><span class="badge {badge_class[overall_verdict]}">{overall_verdict}</span></td></tr>')
    h.append(f'<tr><td>First-token match</td><td>{first_match_count}/{total_prompts} ({100*first_match_count/total_prompts:.0f}%)</td>')
    fm_status = "PASS" if first_match_count >= total_prompts - 2 else ("WARN" if first_match_count >= total_prompts // 2 else "FAIL")
    h.append(f'<td><span class="badge {badge_class[fm_status]}">{fm_status}</span></td></tr>')

    h.append(f'<tr><td>Avg match run length</td><td>{avg_match_run:.1f} tokens</td>')
    mr_status = "PASS" if avg_match_run >= 15 else ("WARN" if avg_match_run >= 5 else "FAIL")
    h.append(f'<td><span class="badge {badge_class[mr_status]}">{mr_status}</span></td></tr>')

    h.append(f'<tr><td>Avg top-k containment</td><td>{100*avg_containment:.1f}%</td>')
    ct_status = "PASS" if avg_containment >= 0.90 else ("WARN" if avg_containment >= 0.75 else "FAIL")
    h.append(f'<td><span class="badge {badge_class[ct_status]}">{ct_status}</span></td></tr>')

    h.append(f'<tr><td>Prompts PASS</td><td>{n_pass}</td><td></td></tr>')
    h.append(f'<tr><td>Prompts WARN</td><td>{n_warn}</td><td></td></tr>')
    h.append(f'<tr><td>Prompts FAIL</td><td>{n_fail}</td><td></td></tr>')

    h.append('</tbody></table>')

    # ── Per-Prompt Cards ──
    h.append('<h2>Per-Prompt Results</h2>')

    for pr in prompt_results:
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
        fm_class = "metric-good" if m["first_match"] else "metric-bad"
        fm_label = "MATCH" if m["first_match"] else "MISMATCH"
        h.append(f'<div class="metric-item"><span class="metric-label">First token: </span>'
                 f'<span class="metric-value {fm_class}">{fm_label}</span></div>')

        mr_class = "metric-good" if m["match_run"] >= 15 else ("metric-ok" if m["match_run"] >= 5 else "metric-bad")
        h.append(f'<div class="metric-item"><span class="metric-label">Match run: </span>'
                 f'<span class="metric-value {mr_class}">{m["match_run"]}/{m["ref_tokens_count"]}</span></div>')

        ct_class = "metric-good" if m["containment_rate"] >= 0.90 else ("metric-ok" if m["containment_rate"] >= 0.75 else "metric-bad")
        h.append(f'<div class="metric-item"><span class="metric-label">Top-k containment: </span>'
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
            h.append(f'<div><span class="label">Reference token:</span> {di["ref_token"]} (logprob: {di["ref_logprob"]:.3f})</div>' if di["ref_logprob"] is not None else '')
            h.append(f'<div><span class="label">Our token:</span> {di["our_token"]} (logprob: {di["our_logprob"]:.3f})</div>' if di["our_logprob"] is not None else '')
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

    # ── Footer ──
    h.append(f'<div class="meta" style="margin-top:32px;border-top:1px solid #30363d;padding-top:12px">')
    h.append(f'<span>Generated by Krasis reference-test</span>')
    h.append(f'<span>Total time: {total_duration:.1f}s</span>')
    h.append('</div>')
    h.append('</body></html>')

    return "\n".join(h)


# ── Main ────────────────────────────────────────────────────────────

def main():
    if not os.environ.get("KRASIS_DEV_SCRIPT"):
        print("ERROR: This script must be run via ./dev reference-test, not directly.", file=sys.stderr)
        print("Usage: ./dev reference-test <config>", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Krasis reference test")
    parser.add_argument("config", help="Config file path")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    parser.add_argument("--no-server", action="store_true",
                        help="Don't start server, connect to already-running one")
    parser.add_argument("--port", type=int, default=0,
                        help="Port override (for --no-server)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

    # Get GPU info
    try:
        gpu_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True
        ).strip().split("\n")[0]
    except Exception:
        gpu_out = "Unknown GPU"

    # Start server
    proc = None
    t_total_start = time.time()

    if not args.no_server:
        print("Starting server...")
        proc, port = start_server(args.config, script_dir)
        print(f"Waiting for server on port {port}...")
        if not wait_for_server(port, timeout=600):
            print("ERROR: Server did not start within timeout", file=sys.stderr)
            if proc:
                kill_server(proc)
            sys.exit(1)
        print("Server ready.")
    else:
        print(f"Using existing server on port {port}")

    try:
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
                        "conv_idx": conv_idx,
                        "turn": turn_idx + 1,
                        "multi_turn": len(turns) > 1,
                        "prompt": prompt_text,
                        "verdict": "FAIL",
                        "metrics": {
                            "first_match": False,
                            "match_run": 0,
                            "ref_tokens_count": len(turn["token_ids"]),
                            "our_tokens_count": 0,
                            "containment_rate": 0.0,
                            "containment_hits": 0,
                            "containment_total": 0,
                            "divergence_pos": 0,
                            "divergence_info": None,
                            "token_table": [],
                            "ref_text": turn.get("text", "")[:500],
                            "our_text": f"ERROR: {e}",
                        },
                        "timing": None,
                    })
                    continue

                metrics = compute_metrics(turn, result)
                verdict = judge_prompt(metrics)
                timing = result.get("timing")

                prompt_results.append({
                    "conv_idx": conv_idx,
                    "turn": turn_idx + 1,
                    "multi_turn": len(turns) > 1,
                    "prompt": prompt_text,
                    "verdict": verdict,
                    "metrics": metrics,
                    "timing": timing,
                })

                # Print summary
                fm = "MATCH" if metrics["first_match"] else "MISS"
                print(f"    {verdict}: first={fm}, run={metrics['match_run']}/{metrics['ref_tokens_count']}, "
                      f"top-k={100*metrics['containment_rate']:.0f}%")

    finally:
        if proc:
            print("Stopping server...")
            kill_server(proc)

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
    log_dir = os.path.join(script_dir, "logs", "reference-tests")
    os.makedirs(log_dir, exist_ok=True)

    quant_info = f"int{cfg.get('CFG_GPU_EXPERT_BITS', '4')}_{cfg.get('CFG_ATTENTION_QUANT', 'a16')}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"krasis_reference_test_{model_name}_{quant_info}_{timestamp}.html"
    report_path = os.path.join(log_dir, report_name)

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

    print(f"  Prompts: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL (of {len(prompt_results)})")
    print(f"  First-token match: {first_match_count}/{len(prompt_results)}")
    print(f"  Avg match run: {avg_run:.1f} tokens")
    print(f"  Avg top-k containment: {100*avg_cont:.1f}%")
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
