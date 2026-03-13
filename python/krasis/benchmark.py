"""Standardized benchmark for Krasis model + hardware combinations.

Reports three numbers:
  1. Prefill (internal)    — Rust Instant timing, tok/s
  2. Decode (internal)     — Rust Instant timing, tok/s
  3. Round trip (network)  — HTTP client wall clock, tok/s (full prefill + decode)

Usage (from server.py):
    from krasis.benchmark import KrasisBenchmark
    bench = KrasisBenchmark(model, rust_server=rust_server, host="127.0.0.1", port=8012)
    bench.run()
"""

import http.client
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch

logger = logging.getLogger("krasis.benchmark")

# ANSI formatting
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
NC = "\033[0m"


def _section(label: str) -> str:
    """Format a highlighted section header."""
    return f"\n{BOLD}{CYAN}▸ {label}{NC}"


class KrasisBenchmark:
    """Standardized benchmark suite — Prefill (internal), Decode (internal), Round trip (network)."""

    PREFILL_LENGTHS = [1000, 5000, 10000, 20000, 35000, 50000]
    DECODE_LENGTHS = [50, 100, 250]
    WARMUP_MAX_CHARS = 125000  # ~25K tokens

    def __init__(self, model, rust_server=None, host: str = "127.0.0.1",
                 port: int = 8012, timing: bool = False):
        self.model = model  # for system/model info collection + tokenizer
        self.rust_server = rust_server  # for engine benchmarks
        self.host = host
        self.port = port
        # Respect env var KRASIS_DECODE_TIMING=1 even if timing=False was passed
        if os.environ.get("KRASIS_DECODE_TIMING", "") == "1":
            timing = True
        self.timing = timing

        self.decode_tokens = 64  # legacy, kept for compatibility
        self.n_runs = len(self.DECODE_LENGTHS)  # 3 runs = 3 decode lengths

        if not timing:
            os.environ.pop("KRASIS_DECODE_TIMING", None)

    # ──────────────────────────────────────────────────────────
    # System info collection
    # ──────────────────────────────────────────────────────────

    def _collect_system_info(self) -> Dict:
        """Collect CPU, RAM, GPU info."""
        info = {
            "cpu_model": "unknown",
            "cpu_cores": 0,
            "ram_total_gb": 0,
            "ram_process_gb": 0.0,
            "gpus": [],
        }

        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        except (OSError, ValueError):
            pass

        try:
            result = subprocess.run(
                ["lscpu"], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                cores = 0
                sockets = 1
                for line in result.stdout.split("\n"):
                    if line.startswith("Core(s) per socket:"):
                        cores = int(line.split(":")[-1].strip())
                    elif line.startswith("Socket(s):"):
                        sockets = int(line.split(":")[-1].strip())
                info["cpu_cores"] = cores * sockets
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["ram_total_gb"] = kb // (1024 * 1024)
                        break
        except (OSError, ValueError):
            pass

        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        info["ram_process_gb"] = round(kb / (1024 * 1024), 1)
                        break
        except (OSError, ValueError):
            pass

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        info["gpus"].append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "vram_mb": int(parts[2]),
                        })
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        return info

    def _collect_vram_usage(self) -> List[Dict]:
        """Collect per-GPU VRAM usage after model load."""
        usage = []
        for i in range(torch.cuda.device_count()):
            usage.append({
                "index": i,
                "allocated_mb": round(torch.cuda.memory_allocated(i) / (1024 * 1024)),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated(i) / (1024 * 1024)),
            })
        return usage

    def _collect_model_info(self) -> Dict:
        """Collect model config and strategy info."""
        cfg = self.model.cfg
        qcfg = self.model.quant_cfg

        model_type = "unknown"
        config_path = os.path.join(cfg.model_path, "config.json")
        try:
            with open(config_path) as f:
                raw = json.load(f)
            model_type = raw.get("text_config", raw).get("model_type", "unknown")
        except (OSError, json.JSONDecodeError):
            pass

        lgs = self.model.layer_group_size
        if lgs == 0:
            expert_mode = "persistent"
        else:
            expert_mode = f"layer_grouped({lgs})"
        first_mgr = next(iter(self.model.gpu_prefill_managers.values()), None)
        if first_mgr and getattr(first_mgr, '_hcs_initialized', False):
            expert_mode = f"hcs + {expert_mode}"

        if getattr(self.model, '_stream_attn_enabled', False):
            expert_mode += " + stream_attn"

        gpu_threshold = getattr(self.model, 'gpu_prefill_threshold', 300)
        if gpu_threshold <= 1:
            decode_mode = f"gpu_decode ({expert_mode})"
        else:
            decode_mode = f"pure_cpu decode, {expert_mode} prefill"

        kv_dtype_str = "FP8 E4M3" if self.model.kv_dtype == torch.float8_e4m3fn else "BF16"
        num_gpus_used = getattr(self.model, '_num_gpus', len(self.model.pp_partition))

        return {
            "model_name": os.path.basename(cfg.model_path),
            "model_type": model_type,
            "num_layers": cfg.num_hidden_layers,
            "n_routed_experts": cfg.n_routed_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "n_shared_experts": cfg.n_shared_experts,
            "is_hybrid": cfg.is_hybrid,
            "num_full_attention_layers": cfg.num_full_attention_layers if cfg.is_hybrid else cfg.num_hidden_layers,
            "pp_partition": self.model.pp_partition,
            "num_gpus": num_gpus_used,
            "gpu_expert_bits": qcfg.gpu_expert_bits,
            "cpu_expert_bits": qcfg.cpu_expert_bits,
            "attention_quant": qcfg.attention,
            "shared_expert_quant": qcfg.shared_expert,
            "dense_mlp_quant": qcfg.dense_mlp,
            "lm_head_quant": qcfg.lm_head,
            "kv_dtype": kv_dtype_str,
            "layer_group_size": lgs,
            "expert_mode": expert_mode,
            "gpu_prefill_threshold": gpu_threshold,
            "decode_mode": decode_mode,
        }

    # ──────────────────────────────────────────────────────────
    # Prompt building
    # ──────────────────────────────────────────────────────────

    def _load_prompt_file(self, filename: str) -> str:
        """Load a prompt file bundled with the package."""
        prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
        prompt_path = os.path.join(prompts_dir, filename)
        if os.path.isfile(prompt_path):
            with open(prompt_path) as f:
                return f.read().strip()
        raise FileNotFoundError(
            f"Benchmark prompt file '{filename}' not found at {prompt_path}"
        )

    def _kv_cache_max_tokens(self) -> int:
        """Return the maximum number of tokens the KV cache can hold."""
        for cache in self.model.kv_caches:
            if cache is not None:
                return cache.max_pages * cache.page_size
        return 100000

    def _truncate_content_to_tokens(self, content: str, max_tokens: int) -> str:
        """Truncate content so that wrapping in chat template produces <= max_tokens.

        Uses binary search on content length since character-to-token ratio
        varies across different text regions and models.
        """
        messages = [{"role": "user", "content": content}]
        tokens = self.model.tokenizer.apply_chat_template(messages, enable_thinking=False)
        if len(tokens) <= max_tokens:
            return content

        # Binary search for the right content length
        lo, hi = 0, len(content)
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            trial = content[:mid]
            msgs = [{"role": "user", "content": trial}]
            n_tok = len(self.model.tokenizer.apply_chat_template(msgs, enable_thinking=False))
            if n_tok <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return content[:best]

    def _discover_prefill_files(self) -> List[str]:
        """Discover available prefill_prompt_N files."""
        files = []
        for i in range(1, 100):
            try:
                self._load_prompt_file(f"prefill_prompt_{i}")
                files.append(f"prefill_prompt_{i}")
            except FileNotFoundError:
                break
        return files

    def _make_prefill_prompts(self, n: int, max_tokens_override: int = 0) -> tuple:
        """Build n different prefill prompts from separate numbered files."""
        kv_max = self._kv_cache_max_tokens()
        # Reserve headroom for tokenization variance between truncation and re-tokenization
        kv_limit = kv_max - 100
        cap = max_tokens_override if max_tokens_override > 0 else kv_limit
        cap = min(cap, kv_limit)

        files = self._discover_prefill_files()
        if not files:
            raise FileNotFoundError(
                "No prefill prompt files found. Expected prefill_prompt_1, prefill_prompt_2, etc."
            )

        prompts = []
        content_texts = []
        for i in range(n):
            content = self._load_prompt_file(files[i % len(files)])
            content = self._truncate_content_to_tokens(content, cap)
            messages = [{"role": "user", "content": content}]
            tokens = self.model.tokenizer.apply_chat_template(messages, enable_thinking=False)
            prompts.append(tokens)
            content_texts.append(content)
        return prompts, content_texts

    def _make_prefill_prompts_at_lengths(
        self, lengths: List[int], file_offset: int = 0,
    ) -> tuple:
        """Build one prefill prompt per target length from different files."""
        kv_max = self._kv_cache_max_tokens()
        kv_limit = kv_max - 100
        files = self._discover_prefill_files()

        if not files:
            raise FileNotFoundError(
                "No prefill prompt files found. Expected prefill_prompt_1, prefill_prompt_2, etc."
            )

        prompts = []
        content_texts = []
        for i, target in enumerate(lengths):
            content = self._load_prompt_file(files[(i + file_offset) % len(files)])
            cap = min(target, kv_limit)
            content = self._truncate_content_to_tokens(content, cap)
            messages = [{"role": "user", "content": content}]
            tokens = self.model.tokenizer.apply_chat_template(messages, enable_thinking=False)
            prompts.append(tokens)
            content_texts.append(content)
        return prompts, content_texts

    def _make_decode_prompts(self, n: int) -> tuple:
        """Build n different decode prompts from separate numbered files."""
        files = []
        for i in range(1, 100):
            try:
                self._load_prompt_file(f"decode_prompt_{i}")
                files.append(f"decode_prompt_{i}")
            except FileNotFoundError:
                break

        if not files:
            files = ["decode_prompt"]

        prompts = []
        content_texts = []
        for i in range(n):
            content = self._load_prompt_file(files[i % len(files)])
            messages = [{"role": "user", "content": content}]
            prompts.append(self.model.tokenizer.apply_chat_template(messages, enable_thinking=False))
            content_texts.append(content)
        return prompts, content_texts

    # ──────────────────────────────────────────────────────────
    # Engine request (direct Rust, no HTTP/SSE)
    # ──────────────────────────────────────────────────────────

    def _engine_request(self, content_text: str, max_new_tokens: int,
                        temperature: float = 0.6) -> Dict:
        """Single request through the Rust engine (no HTTP/SSE).

        Calls RustServer.benchmark_request() which runs the full
        evict -> prefill -> reload -> decode pipeline with Rust
        Instant timing. Returns dict with engine-internal timings.
        """
        messages = [{"role": "user", "content": content_text}]
        messages_json = json.dumps(messages)
        result_json = self.rust_server.benchmark_request(
            messages_json, max_new_tokens, temperature, False)
        return json.loads(result_json)

    # ──────────────────────────────────────────────────────────
    # Round trip (network) measurement
    # ──────────────────────────────────────────────────────────

    def _http_round_trip(self, content_text: str, max_tokens: int) -> Dict:
        """Full HTTP round trip — prefill + decode via streaming SSE.

        Measures wall clock from HTTP request sent to last token received.
        Reports effective decode tok/s as seen by the network client.
        """
        req_body = {
            "messages": [{"role": "user", "content": content_text}],
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "stream": True,
            "enable_thinking": False,
        }
        body = json.dumps(req_body).encode()

        conn = http.client.HTTPConnection(self.host, self.port, timeout=600)
        t_start = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", body=body,
                      headers={"Content-Type": "application/json"})
        resp = conn.getresponse()

        t_first = None
        token_count = 0
        server_finish_reason = None
        server_decode_tokens = 0
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
                        # Server timing chunk has authoritative token count
                        if "krasis_timing" in data:
                            kt = data["krasis_timing"]
                            server_decode_tokens = kt.get("total_generated", 0)
                            continue
                        choice = data.get("choices", [{}])[0]
                        fr = choice.get("finish_reason")
                        if fr is not None:
                            server_finish_reason = fr
                        delta = choice.get("delta", {})
                        if delta.get("content"):
                            now = time.perf_counter()
                            if t_first is None:
                                t_first = now
                            token_count += 1
                    except json.JSONDecodeError:
                        pass

        conn.close()

        t_end = time.perf_counter()
        total_s = t_end - t_start

        # Use server's authoritative token count if available (content chunk
        # counting undercounts when the stream detokenizer buffers multi-byte
        # sequences, producing empty text for some tokens).
        actual_tokens = server_decode_tokens if server_decode_tokens > 0 else token_count

        # Decode tok/s: tokens after first / time from first to last
        if t_first and actual_tokens > 1:
            decode_time = t_end - t_first
            decode_tokens = actual_tokens - 1
            decode_tok_s = decode_tokens / decode_time
        else:
            decode_tok_s = 0

        # EOS: use server's finish_reason (authoritative) instead of counting
        failed = server_finish_reason == "stop" and actual_tokens < max_tokens

        return {
            "total_s": round(total_s, 2),
            "decode_tok_s": round(decode_tok_s, 2),
            "tokens": actual_tokens,
            "target_tokens": max_tokens,
            "failed": failed,
        }

    def _wait_for_server(self) -> bool:
        """Wait for HTTP server to be ready (up to 10s)."""
        import urllib.request
        for _ in range(100):
            time.sleep(0.1)
            try:
                req = urllib.request.Request(f"http://{self.host}:{self.port}/health")
                with urllib.request.urlopen(req, timeout=1) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
        return False

    # ──────────────────────────────────────────────────────────
    # Benchmark phases
    # ──────────────────────────────────────────────────────────

    def _warmup_engine(self):
        """Warmup via engine path — compile kernels + warm caches."""
        print("  Warmup: 2 short engine requests...")
        self._engine_request("Hello, how are you?", max_new_tokens=8)
        self._engine_request("What is 2+2?", max_new_tokens=8)
        print("  Warmup complete.")

    def _benchmark_prefill_engine(self, prompt_tokens: List[List[int]],
                                  prompt_texts: List[str]) -> Dict:
        """Engine prefill benchmark — Rust Instant timing, no HTTP."""
        runs = []
        for tokens, text in zip(prompt_tokens, prompt_texts):
            n_tokens = len(tokens)
            r = self._engine_request(text, max_new_tokens=1)

            tok_s = r["prefill_tok_s"]
            ms = r["prefill_ms"]
            srv_toks = r["prompt_tokens"]
            if abs(srv_toks - n_tokens) > n_tokens * 0.05:
                print(f"  {YELLOW}WARNING: token mismatch — expected {n_tokens:,} vs engine {srv_toks:,}{NC}")

            runs.append({
                "tok_s": round(tok_s, 1),
                "ms": round(ms, 1),
                "num_tokens": srv_toks or n_tokens,
            })

        best_run = max(runs, key=lambda r: r["tok_s"])
        return {
            "best_tok_s": best_run["tok_s"],
            "best_ms": best_run["ms"],
            "best_num_tokens": best_run["num_tokens"],
            "runs": runs,
        }

    def _benchmark_decode_engine(self, prompt_texts: List[str],
                                 decode_lengths: List[int]) -> Dict:
        """Engine decode benchmark — Rust Instant timing, no HTTP.

        Runs one prompt per decode length. If the model hits EOS before
        generating enough tokens, that run is reported as -1 (failure).
        The headline speed is the max across all successful runs.

        Also tracks min free VRAM across all runs and HCS coverage.
        """
        runs = []
        min_free_vram_mb = None  # track minimum across all runs
        hcs_loaded = 0
        hcs_total = 0
        hcs_pct = 0.0

        for text, target_tokens in zip(prompt_texts, decode_lengths):
            r = self._engine_request(text, max_new_tokens=target_tokens)

            generated = r["decode_tokens"]  # includes first_token from prefill
            tok_s = r["decode_tok_s"]
            decode_ms = r["decode_ms"]

            # Track min free VRAM (minimum across all runs)
            run_min_free = r.get("min_free_vram_mb", 0)
            if min_free_vram_mb is None or run_min_free < min_free_vram_mb:
                min_free_vram_mb = run_min_free

            # HCS stats (same across runs, just take the latest)
            hcs_loaded = r.get("hcs_loaded", 0)
            hcs_total = r.get("hcs_total", 0)
            hcs_pct = r.get("hcs_pct", 0.0)

            # EOS hit early — not enough output, discard this run
            if generated < target_tokens:
                runs.append({
                    "tok_s": -1,
                    "ms_per_tok": -1,
                    "target_tokens": target_tokens,
                    "actual_tokens": generated,
                    "failed": True,
                })
            else:
                ms_per_tok = (1000.0 / tok_s) if tok_s > 0 else 0
                runs.append({
                    "tok_s": round(tok_s, 2),
                    "ms_per_tok": round(ms_per_tok, 1),
                    "target_tokens": target_tokens,
                    "actual_tokens": generated,
                    "failed": False,
                })

        successful = [r for r in runs if not r["failed"]]
        if successful:
            best_run = max(successful, key=lambda r: r["tok_s"])
            best_tok_s = best_run["tok_s"]
            best_ms = best_run["ms_per_tok"]
        else:
            best_tok_s = -1
            best_ms = -1

        return {
            "decode_lengths": decode_lengths,
            "best_tok_s": best_tok_s,
            "best_ms_per_tok": best_ms,
            "runs": runs,
            "min_free_vram_mb": min_free_vram_mb or 0,
            "hcs_loaded": hcs_loaded,
            "hcs_total": hcs_total,
            "hcs_pct": round(hcs_pct, 1),
        }

    def _benchmark_round_trip(self, decode_texts: List[str],
                              decode_lengths: List[int]) -> Dict:
        """Round trip (network) benchmark — full HTTP prefill + decode.

        Uses the same decode prompts/lengths as the engine benchmark.
        Reports effective decode tok/s as seen by the network client.
        """
        runs = []
        for text, target_tokens in zip(decode_texts, decode_lengths):
            r = self._http_round_trip(text, target_tokens)
            runs.append(r)

        successful = [r for r in runs if not r["failed"]]
        if successful:
            best_tok_s = max(r["decode_tok_s"] for r in successful)
        else:
            best_tok_s = -1

        return {
            "decode_lengths": decode_lengths,
            "best_tok_s": best_tok_s,
            "runs": runs,
        }

    # ──────────────────────────────────────────────────────────
    # Output formatting
    # ──────────────────────────────────────────────────────────

    def _format_results(self, sys_info, model_info, vram_usage,
                        prefill_result, decode_result,
                        round_trip_result) -> str:
        """Format results as a human-readable block."""
        lines = []
        sep = "=" * 64
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines.append(sep)
        lines.append(f"Krasis Benchmark — {now}")
        lines.append(sep)

        # Model
        arch_parts = [f"{model_info['model_type']}"]
        arch_parts.append(f"{model_info['num_layers']} layers")
        if model_info['n_routed_experts'] > 0:
            arch_parts.append(f"{model_info['n_routed_experts']} experts")
            arch_parts.append(f"top-{model_info['num_experts_per_tok']}")
        if model_info['is_hybrid']:
            n_full = model_info['num_full_attention_layers']
            n_lin = model_info['num_layers'] - n_full
            arch_parts.append(f"{n_full} GQA + {n_lin} linear")

        lines.append(f"Model:            {model_info['model_name']}")
        lines.append(f"Architecture:     {', '.join(arch_parts)}")
        lines.append(f"PP Partition:     {model_info['pp_partition']} ({model_info['num_gpus']} GPUs)")

        # Hardware
        lines.append("")
        lines.append("Hardware:")
        lines.append(f"  CPU:            {sys_info['cpu_model']} ({sys_info['cpu_cores']} cores)")
        lines.append(f"  RAM:            {sys_info['ram_total_gb']} GB total, {sys_info['ram_process_gb']} GB used by process")
        for gpu in sys_info["gpus"]:
            alloc = "N/A"
            for vu in vram_usage:
                if vu["index"] == gpu["index"]:
                    alloc = f"{vu['allocated_mb']} MB allocated"
                    break
            lines.append(f"  GPU {gpu['index']}:          {gpu['name']} ({gpu['vram_mb']} MB), {alloc}")

        # Quantization
        lines.append("")
        lines.append("Quantization:")
        lines.append(f"  GPU experts:    INT{model_info['gpu_expert_bits']} (Marlin)")
        lines.append(f"  CPU experts:    INT{model_info['cpu_expert_bits']}")
        lines.append(f"  Attention:      {model_info['attention_quant'].upper()}")
        lines.append(f"  Shared expert:  {model_info['shared_expert_quant'].upper()}")
        lines.append(f"  Dense MLP:      {model_info['dense_mlp_quant'].upper()}")
        lines.append(f"  LM head:        {model_info['lm_head_quant'].upper()}")
        lines.append(f"  KV cache:       {model_info['kv_dtype']}")

        # Strategy
        lines.append("")
        lines.append("Strategy:")
        lines.append(f"  Layer group size: {model_info['layer_group_size']} ({model_info['expert_mode']})")
        lines.append(f"  Prefill threshold: {model_info['gpu_prefill_threshold']}")
        lines.append(f"  Mode:           {model_info['decode_mode']}")

        # Prefill (internal)
        n_prefill_runs = len(prefill_result["runs"])
        lines.append("")
        lines.append(f"Prefill (internal) — {n_prefill_runs} runs at different lengths:")
        for i, run in enumerate(prefill_result["runs"]):
            lines.append(f"  Run {i+1}:  {run['tok_s']:,.1f} tok/s ({run['num_tokens']:,} tokens in {run['ms']:.1f}ms)")
        lines.append(f"  Best:   {prefill_result['best_tok_s']:,.1f} tok/s ({prefill_result['best_num_tokens']:,} tokens)")

        # Decode (internal)
        decode_len_str = "/".join(str(l) for l in decode_result.get("decode_lengths", [50, 100, 250]))
        lines.append("")
        lines.append(f"Decode (internal) — {decode_len_str} tokens, 3 separate prompts:")
        for run in decode_result["runs"]:
            if run["failed"]:
                lines.append(f"  {run['target_tokens']:>3} tokens: FAILED (EOS at {run['actual_tokens']} tokens) -> -1 tok/s")
            else:
                lines.append(f"  {run['target_tokens']:>3} tokens: {run['tok_s']:.2f} tok/s ({run['ms_per_tok']:.1f}ms/tok)")
        if decode_result["best_tok_s"] > 0:
            lines.append(f"  Best:    {decode_result['best_tok_s']:.2f} tok/s")
        else:
            lines.append(f"  Best:    FAILED (all runs hit EOS early)")
        lines.append(f"  HCS:     {decode_result.get('hcs_loaded', 0)}/{decode_result.get('hcs_total', 0)} experts ({decode_result.get('hcs_pct', 0):.1f}%)")
        lines.append(f"  Min free VRAM: {decode_result.get('min_free_vram_mb', 0)} MB")

        # Round trip (network)
        if round_trip_result:
            rt_len_str = "/".join(str(l) for l in round_trip_result.get("decode_lengths", [50, 100, 250]))
            lines.append("")
            lines.append(f"Round trip (network) — {rt_len_str} tokens via HTTP:")
            for run in round_trip_result["runs"]:
                if run["failed"]:
                    lines.append(f"  {run['target_tokens']:>3} tokens: FAILED (EOS at {run['tokens']} tokens) -> -1 tok/s")
                else:
                    lines.append(f"  {run['target_tokens']:>3} tokens: {run['decode_tok_s']:.2f} tok/s ({run['total_s']:.2f}s total)")
            if round_trip_result["best_tok_s"] > 0:
                lines.append(f"  Best:    {round_trip_result['best_tok_s']:.2f} tok/s")
            else:
                lines.append(f"  Best:    FAILED (all runs hit EOS early)")

        lines.append(sep)
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────
    # Benchmark archive
    # ──────────────────────────────────────────────────────────

    def _archive_benchmark(self, model_info: Dict, report: str) -> Optional[str]:
        """Write benchmark report to benchmarks/<name>.log for archival."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        benchmarks_dir = os.path.join(repo_root, "benchmarks")

        model_name = model_info["model_name"]
        gguf_path = getattr(self.model, "gguf_path", None)
        if gguf_path:
            gguf_name = os.path.splitext(os.path.basename(gguf_path))[0]
        else:
            gguf_name = "native"
        num_gpus = model_info["num_gpus"]
        gpu_quant = f"int{model_info['gpu_expert_bits']}gpu"
        cpu_quant = f"int{model_info['cpu_expert_bits']}cpu"
        suffix = ""
        if getattr(self.model, '_stream_attn_enabled', False):
            lgs = getattr(self.model, 'layer_group_size', 0)
            suffix = f"_stream_lgs{lgs}"
        filename = f"{model_name}_{gguf_name}_{num_gpus}gpu_{gpu_quant}_{cpu_quant}{suffix}.log"
        rel_path = f"benchmarks/{filename}"

        try:
            os.makedirs(benchmarks_dir, exist_ok=True)
            archive_path = os.path.join(benchmarks_dir, filename)
            with open(archive_path, "w") as f:
                f.write(report + "\n")
            return rel_path
        except OSError as e:
            logger.warning("Could not archive benchmark: %s", e)
            return None

    # ──────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """Run the full benchmark suite. Returns results dict."""
        print(f"\n{BOLD}{'═' * 48}")
        print(f"  KRASIS BENCHMARK")
        print(f"{'═' * 48}{NC}")

        # 1. Collect info
        print(_section("Collecting system info"))
        sys_info = self._collect_system_info()
        model_info = self._collect_model_info()
        vram_usage = self._collect_vram_usage()

        print(f"  Model:    {BOLD}{model_info['model_name']}{NC}")
        print(f"  GPUs:     {len(sys_info['gpus'])}x {sys_info['gpus'][0]['name'] if sys_info['gpus'] else 'none'}")
        print(f"  Strategy: {model_info['decode_mode']}")

        # 2. Build prompts
        n_timed_prefill = len(self.PREFILL_LENGTHS)
        lengths_str = "/".join(f"{l//1000}K" for l in self.PREFILL_LENGTHS)
        print(_section(f"Loading benchmark prompts (warmup + timed at {lengths_str})"))

        warmup_prefill_tokens, warmup_prefill_texts = self._make_prefill_prompts(self.n_runs, max_tokens_override=25000)
        timed_prefill_tokens, timed_prefill_texts = self._make_prefill_prompts_at_lengths(
            self.PREFILL_LENGTHS, file_offset=self.n_runs)
        n_decode_prompts = len(self.DECODE_LENGTHS) * 2  # warmup + timed
        decode_tokens_all, decode_texts_all = self._make_decode_prompts(n_decode_prompts)

        decode_len_str = "/".join(str(l) for l in self.DECODE_LENGTHS)
        print(f"  Warmup prefill:  {self.n_runs} prompts, ~{len(warmup_prefill_tokens[0]):,} tokens each")
        print(f"  Timed prefill:   {n_timed_prefill} prompts at {lengths_str} tokens")
        for i, p in enumerate(timed_prefill_tokens):
            print(f"    [{i+1}] {len(p):,} tokens")
        print(f"  Decode:          {n_decode_prompts} prompts ({decode_len_str} tokens, warmup + timed)")

        warmup_decode_texts = decode_texts_all[:len(self.DECODE_LENGTHS)]
        timed_decode_texts = decode_texts_all[len(self.DECODE_LENGTHS):]

        # 3. Warmup — compile kernels + warm caches (engine path)
        print(_section(f"Warmup ({self.n_runs} prefill + {len(self.DECODE_LENGTHS)} decode via engine)"))
        self._warmup_engine()
        t0 = time.perf_counter()
        print(f"  Prefill warmup ({self.n_runs} runs, ~25K tokens each)...")
        for i, (tokens, text) in enumerate(zip(warmup_prefill_tokens, warmup_prefill_texts)):
            self._engine_request(text, max_new_tokens=1)
            print(f"    [{i+1}/{self.n_runs}] {len(tokens):,} tokens")
        print(f"  Decode warmup ({decode_len_str} tokens, 3 separate prompts)...")
        for text, length in zip(warmup_decode_texts, self.DECODE_LENGTHS):
            self._engine_request(text, max_new_tokens=length)
            print(f"    {length} tokens done")
        warmup_s = time.perf_counter() - t0
        print(f"  Warmup complete ({warmup_s:.1f}s). All kernels compiled.")

        # 4. Prefill (internal)
        print(_section(f"Prefill (internal) — {lengths_str} tokens, {n_timed_prefill} runs"))
        prefill_result = self._benchmark_prefill_engine(timed_prefill_tokens, timed_prefill_texts)
        for i, run in enumerate(prefill_result["runs"]):
            print(f"  Run {i+1}: {run['tok_s']:,.1f} tok/s ({run['num_tokens']:,} tokens in {run['ms']:.1f}ms)")
        print(f"  {BOLD}Best: {prefill_result['best_tok_s']:,.1f} tok/s ({prefill_result['best_num_tokens']:,} tokens){NC}")

        # 5. Decode (internal)
        print(_section(f"Decode (internal) — {decode_len_str} tokens, 3 separate prompts"))
        decode_result = self._benchmark_decode_engine(timed_decode_texts, self.DECODE_LENGTHS)
        for run in decode_result["runs"]:
            if run["failed"]:
                print(f"  {run['target_tokens']:>3} tokens: {YELLOW}FAILED (EOS at {run['actual_tokens']} tokens){NC}")
            else:
                print(f"  {run['target_tokens']:>3} tokens: {run['tok_s']:.2f} tok/s ({run['ms_per_tok']:.1f}ms/tok)")
        if decode_result["best_tok_s"] > 0:
            print(f"  {BOLD}Best: {decode_result['best_tok_s']:.2f} tok/s{NC}")
        else:
            print(f"  {YELLOW}{BOLD}All decode runs failed (EOS too early){NC}")
        print(f"  HCS:  {decode_result['hcs_loaded']}/{decode_result['hcs_total']} experts loaded ({decode_result['hcs_pct']:.1f}%)")
        print(f"  VRAM: {decode_result['min_free_vram_mb']} MB min free during benchmark")

        # 6. Round trip (network) — full HTTP prefill + decode
        round_trip_result = None
        print(_section(f"Round trip (network) — {decode_len_str} tokens via HTTP"))
        if self._wait_for_server():
            # HTTP warmup
            self._http_round_trip("warmup", 8)
            round_trip_result = self._benchmark_round_trip(timed_decode_texts, self.DECODE_LENGTHS)
            for run in round_trip_result["runs"]:
                if run["failed"]:
                    print(f"  {run['target_tokens']:>3} tokens: {YELLOW}FAILED (EOS at {run['tokens']} tokens){NC}")
                else:
                    print(f"  {run['target_tokens']:>3} tokens: {run['decode_tok_s']:.2f} tok/s ({run['total_s']:.2f}s total)")
            if round_trip_result["best_tok_s"] > 0:
                print(f"  {BOLD}Best: {round_trip_result['best_tok_s']:.2f} tok/s{NC}")
            else:
                print(f"  {YELLOW}{BOLD}All round trip runs failed (EOS too early){NC}")
        else:
            print(f"  {YELLOW}Server not ready — skipping round trip benchmark{NC}")

        # 7. Format and write report
        print(_section("Writing results"))
        report = self._format_results(
            sys_info, model_info, vram_usage,
            prefill_result, decode_result, round_trip_result,
        )

        # 8. Archive to benchmarks/ directory
        archive_path = self._archive_benchmark(model_info, report)

        # 9. Final summary
        print(f"\n{BOLD}{'─' * 48}")
        print(f"  BENCHMARK COMPLETE")
        print(f"{'─' * 48}{NC}")
        print(f"  Prefill (internal):    {GREEN}{BOLD}{prefill_result['best_tok_s']:,.1f} tok/s{NC}  (best of {lengths_str})")
        if decode_result["best_tok_s"] > 0:
            print(f"  Decode (internal):     {GREEN}{BOLD}{decode_result['best_tok_s']:.2f} tok/s{NC}  (best of {decode_len_str})")
        else:
            print(f"  Decode (internal):     {YELLOW}{BOLD}FAILED{NC}  (all runs hit EOS early)")
        if round_trip_result and round_trip_result["best_tok_s"] > 0:
            print(f"  Round trip (network):  {GREEN}{BOLD}{round_trip_result['best_tok_s']:.2f} tok/s{NC}  (best of {decode_len_str})")
        elif round_trip_result:
            print(f"  Round trip (network):  {YELLOW}{BOLD}FAILED{NC}  (all runs hit EOS early)")
        print(f"  HCS coverage:          {decode_result.get('hcs_loaded', 0)}/{decode_result.get('hcs_total', 0)} ({decode_result.get('hcs_pct', 0):.1f}%)")
        print(f"  Min free VRAM:         {decode_result.get('min_free_vram_mb', 0)} MB")
        if archive_path:
            print(f"  Log:     {DIM}{archive_path}{NC}")
        print()

        # 10. Return structured results
        return {
            "system": sys_info,
            "model": model_info,
            "vram": vram_usage,
            "prefill": prefill_result,
            "decode": decode_result,
            "round_trip": round_trip_result,
        }
