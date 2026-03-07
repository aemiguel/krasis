"""Standardized benchmark for Krasis model + hardware combinations.

Two measurement tiers:
  1. Engine — calls Rust engine directly (no HTTP/SSE). Rust Instant timing.
     Reports: prefill tok/s, decode tok/s.
  2. Client — sends HTTP requests to the Rust server. Client-side wall clock.
     Reports: TTFT (time to first token).

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
    """Standardized benchmark suite — engine timing + HTTP TTFT."""

    PREFILL_LENGTHS = [1000, 5000, 10000, 20000, 35000, 50000]
    WARMUP_MAX_CHARS = 125000  # ~25K tokens

    def __init__(self, model, rust_server=None, host: str = "127.0.0.1",
                 port: int = 8012, timing: bool = False):
        self.model = model  # for system/model info collection + tokenizer
        self.rust_server = rust_server  # for engine benchmarks
        self.host = host
        self.port = port
        self.timing = timing

        self.decode_tokens = 64
        self.n_runs = int(os.environ.get("KRASIS_BENCH_RUNS", "3"))

        if timing:
            from krasis.timing import TIMING
            TIMING.decode = True
            TIMING.prefill = False
        else:
            from krasis.timing import TIMING
            TIMING.decode = False
            TIMING.prefill = False
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
        kv_limit = self._kv_cache_max_tokens() - 200
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
            messages = [{"role": "user", "content": content}]
            tokens = self.model.tokenizer.apply_chat_template(messages, enable_thinking=False)
            if len(tokens) > cap:
                orig_len = len(tokens)
                tokens = tokens[:cap]
                ratio = cap / orig_len
                content = content[:int(len(content) * ratio)]
            prompts.append(tokens)
            content_texts.append(content)
        return prompts, content_texts

    def _make_prefill_prompts_at_lengths(
        self, lengths: List[int], file_offset: int = 0,
    ) -> tuple:
        """Build one prefill prompt per target length from different files."""
        kv_limit = self._kv_cache_max_tokens() - 200
        files = self._discover_prefill_files()

        if not files:
            raise FileNotFoundError(
                "No prefill prompt files found. Expected prefill_prompt_1, prefill_prompt_2, etc."
            )

        prompts = []
        content_texts = []
        for i, target in enumerate(lengths):
            content = self._load_prompt_file(files[(i + file_offset) % len(files)])
            char_limit = target * 6
            if len(content) > char_limit:
                content = content[:char_limit]
            messages = [{"role": "user", "content": content}]
            tokens = self.model.tokenizer.apply_chat_template(messages, enable_thinking=False)
            cap = min(target, kv_limit)
            if len(tokens) > cap:
                orig_len = len(tokens)
                tokens = tokens[:cap]
                ratio = cap / orig_len
                content = content[:int(len(content) * ratio)]
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
    # HTTP TTFT measurement (client-side wall clock)
    # ──────────────────────────────────────────────────────────

    def _http_ttft(self, content_text: str) -> Dict:
        """Send an HTTP request and measure client-side TTFT only.

        TTFT = wall clock from HTTP request sent to first SSE content token.
        This is what the user actually experiences.
        """
        req_body = {
            "messages": [{"role": "user", "content": content_text}],
            "max_tokens": 1,
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
        prompt_tokens = 0
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
                            prompt_tokens = data["krasis_timing"].get("prompt_tokens", 0)
                            continue
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta and t_first is None:
                            t_first = time.perf_counter()
                    except json.JSONDecodeError:
                        pass

        conn.close()

        ttft = (t_first - t_start) if t_first else 0
        return {"ttft": round(ttft, 3), "prompt_tokens": prompt_tokens}

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

    def _benchmark_decode_engine(self, prompt_texts: List[str]) -> Dict:
        """Engine decode benchmark — Rust Instant timing, no HTTP."""
        runs = []
        for text in prompt_texts:
            r = self._engine_request(text, max_new_tokens=self.decode_tokens)

            tok_s = r["decode_tok_s"]
            decode_ms = r["decode_ms"]
            # decode_tokens includes first_token; decode_tok_s excludes it
            ms_per_tok = (1000.0 / tok_s) if tok_s > 0 else 0
            runs.append({
                "tok_s": round(tok_s, 2),
                "ms_per_tok": round(ms_per_tok, 1),
                "n_gen": r["decode_tokens"],
            })

        avg_tok_s = sum(r["tok_s"] for r in runs) / len(runs) if runs else 0
        avg_ms = sum(r["ms_per_tok"] for r in runs) / len(runs) if runs else 0

        return {
            "num_tokens": self.decode_tokens,
            "avg_tok_s": round(avg_tok_s, 2),
            "avg_ms_per_tok": round(avg_ms, 1),
            "runs": runs,
        }

    def _benchmark_http_ttft(self, prompt_tokens: List[List[int]],
                             prompt_texts: List[str]) -> Dict:
        """HTTP TTFT benchmark — client-side wall clock at different lengths."""
        runs = []
        for tokens, text in zip(prompt_tokens, prompt_texts):
            n_tokens = len(tokens)
            r = self._http_ttft(text)
            runs.append({
                "ttft": r["ttft"],
                "num_tokens": r["prompt_tokens"] or n_tokens,
            })
        return {"runs": runs}

    # ──────────────────────────────────────────────────────────
    # Output formatting
    # ──────────────────────────────────────────────────────────

    def _format_results(self, sys_info, model_info, vram_usage,
                        prefill_result, decode_result,
                        ttft_result) -> str:
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

        # Engine prefill
        n_prefill_runs = len(prefill_result["runs"])
        lines.append("")
        lines.append(f"Engine Prefill ({n_prefill_runs} runs at different lengths):")
        for i, run in enumerate(prefill_result["runs"]):
            lines.append(f"  Run {i+1}:  {run['tok_s']:,.1f} tok/s ({run['num_tokens']:,} tokens in {run['ms']:.1f}ms)")
        lines.append(f"  Best:   {prefill_result['best_tok_s']:,.1f} tok/s ({prefill_result['best_num_tokens']:,} tokens)")

        # Engine decode
        lines.append("")
        lines.append(f"Engine Decode ({self.decode_tokens} tokens, {self.n_runs} runs, different prompts):")
        for i, run in enumerate(decode_result["runs"]):
            lines.append(f"  Run {i+1}:  {run['tok_s']:.2f} tok/s ({run['ms_per_tok']:.1f}ms/tok)")
        lines.append(f"  Average: {decode_result['avg_tok_s']:.2f} tok/s ({decode_result['avg_ms_per_tok']:.1f}ms/tok)")

        # Client TTFT
        if ttft_result:
            lines.append("")
            lines.append(f"Client TTFT ({len(ttft_result['runs'])} runs via HTTP at different lengths):")
            for run in ttft_result["runs"]:
                lines.append(f"  {run['num_tokens']:>6,} tokens:  {run['ttft']:.2f}s")

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
        decode_tokens_all, decode_texts_all = self._make_decode_prompts(self.n_runs * 2)

        print(f"  Warmup prefill:  {self.n_runs} prompts, ~{len(warmup_prefill_tokens[0]):,} tokens each")
        print(f"  Timed prefill:   {n_timed_prefill} prompts at {lengths_str} tokens")
        for i, p in enumerate(timed_prefill_tokens):
            print(f"    [{i+1}] {len(p):,} tokens")
        print(f"  Decode:          {len(decode_tokens_all)} prompts")

        warmup_decode_texts = decode_texts_all[:self.n_runs]
        timed_decode_texts = decode_texts_all[self.n_runs:]

        # 3. Warmup — compile kernels + warm caches (engine path)
        print(_section(f"Warmup ({self.n_runs} prefill + {self.n_runs} decode via engine)"))
        self._warmup_engine()
        t0 = time.perf_counter()
        print(f"  Prefill warmup ({self.n_runs} runs, ~25K tokens each)...")
        for i, (tokens, text) in enumerate(zip(warmup_prefill_tokens, warmup_prefill_texts)):
            self._engine_request(text, max_new_tokens=1)
            print(f"    [{i+1}/{self.n_runs}] {len(tokens):,} tokens")
        print(f"  Decode warmup ({self.n_runs} runs, different prompts)...")
        for i, text in enumerate(warmup_decode_texts):
            self._engine_request(text, max_new_tokens=8)
        warmup_s = time.perf_counter() - t0
        print(f"  Warmup complete ({warmup_s:.1f}s). All kernels compiled.")

        # 4. Engine prefill benchmark
        print(_section(f"Engine prefill benchmark ({lengths_str} tokens, {n_timed_prefill} runs)"))
        prefill_result = self._benchmark_prefill_engine(timed_prefill_tokens, timed_prefill_texts)
        for i, run in enumerate(prefill_result["runs"]):
            print(f"  Run {i+1}: {run['tok_s']:,.1f} tok/s ({run['num_tokens']:,} tokens in {run['ms']:.1f}ms)")
        print(f"  {BOLD}Best: {prefill_result['best_tok_s']:,.1f} tok/s ({prefill_result['best_num_tokens']:,} tokens){NC}")

        # 5. Engine decode benchmark
        print(_section(f"Engine decode benchmark ({self.decode_tokens} tokens, {self.n_runs} runs, different prompts)"))
        decode_result = self._benchmark_decode_engine(timed_decode_texts)
        for i, run in enumerate(decode_result["runs"]):
            print(f"  Run {i+1}: {run['tok_s']:.2f} tok/s ({run['ms_per_tok']:.1f}ms/tok)")
        print(f"  {BOLD}Average: {decode_result['avg_tok_s']:.2f} tok/s ({decode_result['avg_ms_per_tok']:.1f}ms/tok){NC}")

        # 6. Client TTFT benchmark (requires HTTP server)
        ttft_result = None
        print(_section(f"Client TTFT benchmark ({lengths_str} tokens via HTTP)"))
        if self._wait_for_server():
            # HTTP warmup
            self._http_ttft("warmup")
            ttft_result = self._benchmark_http_ttft(timed_prefill_tokens, timed_prefill_texts)
            for run in ttft_result["runs"]:
                print(f"  {run['num_tokens']:>6,} tokens:  {run['ttft']:.2f}s")
        else:
            print(f"  {YELLOW}Server not ready — skipping HTTP TTFT benchmark{NC}")

        # 7. Format and write report
        print(_section("Writing results"))
        report = self._format_results(
            sys_info, model_info, vram_usage,
            prefill_result, decode_result, ttft_result,
        )

        # 8. Archive to benchmarks/ directory
        archive_path = self._archive_benchmark(model_info, report)

        # 9. Final summary
        print(f"\n{BOLD}{'─' * 48}")
        print(f"  BENCHMARK COMPLETE")
        print(f"{'─' * 48}{NC}")
        print(f"  Prefill: {GREEN}{BOLD}{prefill_result['best_tok_s']:,.1f} tok/s{NC}  (engine, best of {lengths_str})")
        print(f"  Decode:  {GREEN}{BOLD}{decode_result['avg_tok_s']:.2f} tok/s{NC}  (engine, {decode_result['avg_ms_per_tok']:.1f}ms/tok)")
        if ttft_result:
            # Show TTFT for ~10K tokens if available, else middle run
            ttft_runs = ttft_result["runs"]
            ttft_show = None
            for r in ttft_runs:
                if 8000 <= r["num_tokens"] <= 12000:
                    ttft_show = r
                    break
            if not ttft_show:
                ttft_show = ttft_runs[len(ttft_runs) // 2]
            print(f"  TTFT:    {GREEN}{BOLD}{ttft_show['ttft']:.2f}s{NC}  (client HTTP, {ttft_show['num_tokens']:,} tokens)")
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
            "ttft": ttft_result,
        }
