"""Standardized benchmark for Krasis model + hardware combinations.

Runs after model load (in server.py), before serving begins.
Results are printed to stdout and appended to benchmark_results.log.

Usage (from server.py):
    from krasis.benchmark import KrasisBenchmark
    bench = KrasisBenchmark(model)
    bench.run()
"""

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch

from krasis.timing import TIMING

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
    """Standardized benchmark suite for a loaded KrasisModel."""

    def __init__(self, model):
        self.model = model

        self.decode_tokens = 64
        self.n_runs = int(os.environ.get("KRASIS_BENCH_RUNS", "3"))

        # Ensure instrumentation is OFF for benchmarking
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

        # CPU model
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        except (OSError, ValueError):
            pass

        # Physical cores
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

        # Total RAM
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["ram_total_gb"] = kb // (1024 * 1024)
                        break
        except (OSError, ValueError):
            pass

        # Process RSS
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        info["ram_process_gb"] = round(kb / (1024 * 1024), 1)
                        break
        except (OSError, ValueError):
            pass

        # GPUs via nvidia-smi
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

        # Read model_type from config.json (not stored on ModelConfig)
        model_type = "unknown"
        config_path = os.path.join(cfg.model_path, "config.json")
        try:
            with open(config_path) as f:
                raw = json.load(f)
            model_type = raw.get("text_config", raw).get("model_type", "unknown")
        except (OSError, json.JSONDecodeError):
            pass

        # Determine strategy description
        lgs = self.model.layer_group_size
        if lgs == 0:
            expert_mode = "persistent"
        else:
            expert_mode = f"layer_grouped({lgs})"
        # Check if HCS is active
        first_mgr = next(iter(self.model.gpu_prefill_managers.values()), None)
        if first_mgr and getattr(first_mgr, '_hcs_initialized', False):
            expert_mode = f"hcs + {expert_mode}"

        # Streaming attention
        if getattr(self.model, '_stream_attn_enabled', False):
            expert_mode += " + stream_attn"

        # Decode mode
        gpu_threshold = getattr(self.model, 'gpu_prefill_threshold', 300)
        if gpu_threshold <= 1:
            decode_mode = f"gpu_decode ({expert_mode})"
        else:
            decode_mode = f"pure_cpu decode, {expert_mode} prefill"

        # KV dtype
        kv_dtype_str = "FP8 E4M3" if self.model.kv_dtype == torch.float8_e4m3fn else "BF16"

        # Actual GPU count: may differ from PP partition length (e.g. HCS uses
        # extra GPUs for pinned experts without splitting layers across them).
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
        """Load a prompt from benchmark_data/<filename> inside the package."""
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(pkg_dir, "benchmark_data", filename)
        if os.path.isfile(prompt_path):
            with open(prompt_path) as f:
                return f.read().strip()
        # Fallback: repo layout (benchmarks/ at repo root)
        repo_dir = os.path.dirname(os.path.dirname(pkg_dir))
        prompt_path = os.path.join(repo_dir, "benchmarks", filename)
        if os.path.isfile(prompt_path):
            with open(prompt_path) as f:
                return f.read().strip()
        raise FileNotFoundError(
            f"Benchmark prompt file '{filename}' not found in package or repo"
        )

    def _make_prompt(self) -> List[int]:
        """Build prefill prompt (~10K tokens) from file."""
        content = self._load_prompt_file("prefill_prompt_20k")
        messages = [{"role": "user", "content": content}]
        tokens = self.model.tokenizer.apply_chat_template(messages)
        return tokens[:10000]

    def _make_short_prompt(self) -> List[int]:
        """Build decode prompt from file."""
        content = self._load_prompt_file("decode_prompt")
        messages = [{"role": "user", "content": content}]
        return self.model.tokenizer.apply_chat_template(messages)

    # ──────────────────────────────────────────────────────────
    # Benchmark phases
    # ──────────────────────────────────────────────────────────

    def _warmup(self):
        """Run 2 short generations to warm up CUDA graphs and caches."""
        print("  Warmup: 2 short generations...")
        short = self._make_short_prompt()
        for i in range(2):
            self.model.generate(short, max_new_tokens=8, temperature=0.6, top_k=50)
        print("  Warmup complete.")

    def _benchmark_prefill(self, tokens: List[int]) -> Dict:
        """Measure prefill speed over n_runs."""
        runs = []
        for i in range(self.n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.model.generate(tokens, max_new_tokens=1, temperature=0.6)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            ttft = getattr(self.model, '_last_ttft', t1 - t0)
            tok_s = len(tokens) / ttft if ttft > 0 else 0
            runs.append({"ttft": round(ttft, 2), "tok_s": round(tok_s, 1)})

        avg_ttft = sum(r["ttft"] for r in runs) / len(runs)
        avg_tok_s = sum(r["tok_s"] for r in runs) / len(runs)
        return {
            "num_tokens": len(tokens),
            "avg_ttft": round(avg_ttft, 2),
            "avg_tok_s": round(avg_tok_s, 1),
            "runs": runs,
        }

    def _benchmark_decode(self, tokens: List[int]) -> Dict:
        """Measure decode speed over n_runs. Captures output from last run."""
        runs = []
        last_generated_ids = []
        for i in range(self.n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            generated = self.model.generate(
                tokens, max_new_tokens=self.decode_tokens,
                temperature=0.6, top_k=50,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            n_gen = len(generated)
            total_s = t1 - t0
            ttft = getattr(self.model, '_last_ttft', 0)
            decode_s = total_s - ttft
            if decode_s > 0 and n_gen > 0:
                tok_s = n_gen / decode_s
                ms_per_tok = (decode_s / n_gen) * 1000
            else:
                tok_s = 0
                ms_per_tok = 0
            runs.append({
                "tok_s": round(tok_s, 2),
                "ms_per_tok": round(ms_per_tok, 1),
                "n_gen": n_gen,
            })
            last_generated_ids = generated

        avg_tok_s = sum(r["tok_s"] for r in runs) / len(runs)
        avg_ms = sum(r["ms_per_tok"] for r in runs) / len(runs)

        # Decode the generated output text for verification
        output_text = ""
        if last_generated_ids:
            try:
                output_text = self.model.tokenizer.decode(last_generated_ids)
            except Exception:
                output_text = f"[{len(last_generated_ids)} tokens, decode failed]"

        return {
            "num_tokens": self.decode_tokens,
            "avg_tok_s": round(avg_tok_s, 2),
            "avg_ms_per_tok": round(avg_ms, 1),
            "runs": runs,
            "output_text": output_text,
        }

    # ──────────────────────────────────────────────────────────
    # Output formatting
    # ──────────────────────────────────────────────────────────

    def _format_results(self, sys_info, model_info, vram_usage,
                        prefill_result, decode_result,
                        prefill_prompt: str = "", decode_prompt: str = "") -> str:
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

        # Prefill results
        lines.append("")
        lines.append(f"Prefill ({prefill_result['num_tokens']} tokens, {self.n_runs} runs):")
        for i, run in enumerate(prefill_result["runs"]):
            lines.append(f"  Run {i+1}:  {run['tok_s']:.1f} tok/s, TTFT={run['ttft']:.2f}s")
        lines.append(f"  Average: {prefill_result['avg_tok_s']:.1f} tok/s, TTFT={prefill_result['avg_ttft']:.2f}s")

        # Decode results
        lines.append("")
        lines.append(f"Decode ({self.decode_tokens} tokens, {self.n_runs} runs):")
        for i, run in enumerate(decode_result["runs"]):
            lines.append(f"  Run {i+1}:  {run['tok_s']:.2f} tok/s ({run['ms_per_tok']:.1f}ms/tok)")
        lines.append(f"  Average: {decode_result['avg_tok_s']:.2f} tok/s ({decode_result['avg_ms_per_tok']:.1f}ms/tok)")

        # Verification: prompts and generated output
        lines.append("")
        lines.append("Verification:")
        if prefill_prompt:
            # Show first 200 chars of prefill prompt
            truncated = prefill_prompt[:200]
            if len(prefill_prompt) > 200:
                truncated += f"... [{len(prefill_prompt)} chars total]"
            lines.append(f"  Prefill prompt: {truncated}")
        if decode_prompt:
            lines.append(f"  Decode prompt:  {decode_prompt}")
        output_text = decode_result.get("output_text", "")
        if output_text:
            last_run = decode_result["runs"][-1] if decode_result["runs"] else {}
            n_gen = last_run.get("n_gen", "?")
            lines.append(f"  Generated output ({n_gen} tokens):")
            for out_line in output_text.split("\n"):
                lines.append(f"    {out_line}")

        lines.append(sep)
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────
    # Benchmark archive
    # ──────────────────────────────────────────────────────────

    def _archive_benchmark(self, model_info: Dict, report: str) -> Optional[str]:
        """Write benchmark report to benchmarks/<name>.log for archival.

        Returns the relative path (benchmarks/<filename>) or None on failure.
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        benchmarks_dir = os.path.join(repo_root, "benchmarks")

        # Build filename: <model>_<gguf>_<N>gpu_int<X>gpu_int<Y>cpu.log
        model_name = model_info["model_name"]
        gguf_path = getattr(self.model, "gguf_path", None)
        if gguf_path:
            gguf_name = os.path.splitext(os.path.basename(gguf_path))[0]
        else:
            gguf_name = "native"
        num_gpus = model_info["num_gpus"]
        gpu_quant = f"int{model_info['gpu_expert_bits']}gpu"
        cpu_quant = f"int{model_info['cpu_expert_bits']}cpu"
        # Add stream_attn and layer_group_size to filename for differentiation
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
        print(_section("Loading benchmark prompts"))
        large_tokens = self._make_prompt()
        short_tokens = self._make_short_prompt()
        print(f"  Prefill: {len(large_tokens):,} tokens")
        print(f"  Decode:  {len(short_tokens):,} tokens")

        # 3. Warmup
        print(_section("Warmup (2 short generations)"))
        self._warmup()

        # 4. Prefill benchmark
        print(_section(f"Running prefill benchmark ({len(large_tokens):,} tokens, {self.n_runs} runs)"))
        prefill_result = self._benchmark_prefill(large_tokens)
        for i, run in enumerate(prefill_result["runs"]):
            print(f"  Run {i+1}: {run['tok_s']:.1f} tok/s, TTFT={run['ttft']:.2f}s")
        print(f"  {BOLD}Average: {prefill_result['avg_tok_s']:.1f} tok/s, TTFT={prefill_result['avg_ttft']:.2f}s{NC}")

        # 5. Decode benchmark
        print(_section(f"Running decode benchmark ({self.decode_tokens} tokens, {self.n_runs} runs)"))
        decode_result = self._benchmark_decode(short_tokens)
        for i, run in enumerate(decode_result["runs"]):
            print(f"  Run {i+1}: {run['tok_s']:.2f} tok/s ({run['ms_per_tok']:.1f}ms/tok)")
        print(f"  {BOLD}Average: {decode_result['avg_tok_s']:.2f} tok/s ({decode_result['avg_ms_per_tok']:.1f}ms/tok){NC}")

        # 6. Decode prompt text for the log
        try:
            prefill_prompt_text = self.model.tokenizer.decode(large_tokens)
        except Exception:
            prefill_prompt_text = f"[{len(large_tokens)} tokens]"
        try:
            decode_prompt_text = self._load_prompt_file("decode_prompt")
        except OSError:
            decode_prompt_text = "[decode prompt]"

        # 7. Format and write report
        print(_section("Writing results"))
        report = self._format_results(
            sys_info, model_info, vram_usage,
            prefill_result, decode_result,
            prefill_prompt=prefill_prompt_text,
            decode_prompt=decode_prompt_text,
        )

        # 8. Archive to benchmarks/ directory
        archive_path = self._archive_benchmark(model_info, report)

        # 10. Final summary
        print(f"\n{BOLD}{'─' * 48}")
        print(f"  BENCHMARK COMPLETE")
        print(f"{'─' * 48}{NC}")
        print(f"  Prefill: {GREEN}{BOLD}{prefill_result['avg_tok_s']:.1f} tok/s{NC}  TTFT={prefill_result['avg_ttft']:.2f}s")
        print(f"  Decode:  {GREEN}{BOLD}{decode_result['avg_tok_s']:.2f} tok/s{NC}  ({decode_result['avg_ms_per_tok']:.1f}ms/tok)")
        if archive_path:
            print(f"  Log:     {DIM}{archive_path}{NC}")
        print()

        # 11. Return structured results
        return {
            "system": sys_info,
            "model": model_info,
            "vram": vram_usage,
            "prefill": prefill_result,
            "decode": decode_result,
        }
