#!/usr/bin/env python3
"""Krasis Meta-Optimiser: auto-discovers best prefill + decode strategy pair.

Phase 1: Test GPU prefill strategies with a large prompt (~10K tokens)
Phase 2: Test decode strategies with a short prompt + 64 generated tokens

All instrumentation is OFF during benchmarking for accurate measurements.
Results saved to ~/.krasis/cache/<model_name>/meta_optimiser.json
"""

import gc
import json
import logging
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.WARNING,  # Quiet during benchmarking
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("meta_optimiser")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from krasis.model import KrasisModel
from krasis.config import cache_dir_for_model
from krasis.timing import TIMING

# ══════════════════════════════════════════════════════════════════
# Strategy definitions
# ══════════════════════════════════════════════════════════════════

# Phase 1: GPU prefill strategies (tested with large prompt)
PREFILL_STRATEGIES = [
    ("persistent",      1,  "All experts in VRAM (zero DMA) — fastest if they fit"),
    ("hcs_prefill",    -3,  "1-layer groups, hot experts skip DMA"),
    ("layer_grouped_2", 2,  "2-layer groups, amortised DMA"),
    ("layer_grouped_4", 4,  "4-layer groups, fewer DMA rounds"),
    ("active_only",    -1,  "DMA only activated experts per layer"),
    ("chunked",         0,  "Legacy per-layer full DMA"),
]

# Phase 2: Decode strategies (tested with short prompt + generation)
DECODE_STRATEGIES = [
    ("hcs_hybrid", -3,  "Hot experts GPU + cold experts CPU in parallel"),
    ("compact",    -1,  "Cross-token LRU caching in per-layer VRAM slots"),
    ("lru",        -2,  "Cross-layer LRU expert caching"),
    ("pure_cpu",  None, "CPU-only decode (no GPU involvement)"),
]

# ══════════════════════════════════════════════════════════════════
# Model configurations
# ══════════════════════════════════════════════════════════════════

MODELS = {
    "coder_next": {
        "model_path": "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next",
        "heatmap_path": "/home/main/Documents/Claude/krasis/tests/coder_next_heatmap.json",
        "pp_partition": [24, 24],
        "devices": ["cuda:0", "cuda:1"],
    },
    "qwen235b": {
        "model_path": "/home/main/Documents/Claude/krasis/models/Qwen3-235B-A22B",
        "heatmap_path": "/home/main/Documents/Claude/krasis/tests/qwen235b_heatmap.json",
        "pp_partition": [47, 47],
        "devices": ["cuda:0", "cuda:1"],
    },
}


def gpu_mem_mb():
    """Current GPU memory usage per device in MB."""
    return {i: torch.cuda.memory_allocated(i) / 1e6
            for i in range(torch.cuda.device_count())}


class MetaOptimiser:
    """Discovers optimal prefill + decode strategy pair for a loaded model."""

    def __init__(
        self,
        model_path: str,
        pp_partition: List[int],
        devices: List[str],
        heatmap_path: Optional[str] = None,
        kv_dtype: torch.dtype = torch.float8_e4m3fn,
        threads: int = 48,
        prefill_tokens: int = 10000,
        decode_tokens: int = 64,
        n_runs: int = 3,
    ):
        self.model_path = model_path
        self.heatmap_path = heatmap_path
        self.prefill_target = prefill_tokens
        self.decode_tokens = decode_tokens
        self.n_runs = n_runs

        # Ensure ALL instrumentation is OFF
        TIMING.decode = False
        TIMING.prefill = False
        os.environ.pop("KRASIS_DECODE_TIMING", None)

        print("=" * 70)
        print("KRASIS META-OPTIMISER")
        print("=" * 70)
        print(f"  Model:  {os.path.basename(model_path)}")
        print(f"  PP:     {pp_partition}")
        print(f"  Devices: {devices}")
        print(f"  Target prefill: ~{prefill_tokens} tokens")
        print(f"  Decode tokens:  {decode_tokens}")
        print(f"  Runs per test:  {n_runs}")
        print()

        # Load model once with chunked (simplest, no special init needed)
        print("Loading model (one-time)...")
        # Temporarily raise log level for loading
        logging.getLogger().setLevel(logging.INFO)
        t0 = time.perf_counter()
        self.model = KrasisModel(
            model_path,
            pp_partition=pp_partition,
            devices=devices,
            kv_dtype=kv_dtype,
            krasis_threads=threads,
            gpu_prefill=True,
            gpu_prefill_threshold=1,
            expert_divisor=0,  # neutral start
        )
        self.model.load()
        t_load = time.perf_counter() - t0
        logging.getLogger().setLevel(logging.WARNING)

        print(f"Model loaded in {t_load:.1f}s")
        mem = gpu_mem_mb()
        for i, mb in mem.items():
            print(f"  GPU{i}: {mb:.0f} MB")
        print()

        # Prepare prompts
        self.large_tokens = self._make_prompt(prefill_tokens)
        self.small_tokens = self._make_short_prompt()
        print(f"  Large prompt: {len(self.large_tokens)} tokens")
        print(f"  Small prompt: {len(self.small_tokens)} tokens")

        # Results storage
        self.results: Dict = {
            "model": os.path.basename(model_path),
            "pp_partition": pp_partition,
            "devices": devices,
            "prefill_prompt_tokens": len(self.large_tokens),
            "decode_prompt_tokens": len(self.small_tokens),
            "decode_gen_tokens": decode_tokens,
            "n_runs": n_runs,
            "prefill": {},
            "decode": {},
            "best_prefill": None,
            "best_decode": None,
        }

    def _make_prompt(self, target_tokens: int) -> List[int]:
        """Generate a chat prompt of approximately target_tokens length."""
        # Build a long content block
        sections = [
            "Explain distributed consensus algorithms including Paxos, Raft, and PBFT. ",
            "Describe database transaction isolation levels and their trade-offs. ",
            "Discuss compiler optimization passes such as dead code elimination and loop unrolling. ",
            "Explain the CAP theorem and its practical implications for system design. ",
            "Describe memory management strategies in operating systems including paging and segmentation. ",
            "Discuss the principles of functional programming and category theory. ",
            "Explain how neural network backpropagation works with gradient descent. ",
            "Describe the architecture of modern CPUs including pipelining and branch prediction. ",
            "Discuss cryptographic primitives including AES, RSA, and elliptic curve cryptography. ",
            "Explain container orchestration with Kubernetes including pods, services, and deployments. ",
        ]
        # Repeat until we overshoot, then truncate
        content = ""
        while True:
            for section in sections:
                content += section
            # Check token count
            messages = [{"role": "user", "content": content}]
            tokens = self.model.tokenizer.apply_chat_template(messages)
            if len(tokens) >= target_tokens:
                return tokens[:target_tokens]

    def _make_short_prompt(self) -> List[int]:
        """Generate a short chat prompt for decode testing."""
        messages = [{"role": "user", "content": "Write a poem about recursion in programming."}]
        return self.model.tokenizer.apply_chat_template(messages)

    def _switch_strategy(self, expert_divisor: Optional[int]):
        """Reinitialize GPU prefill managers with a new strategy.

        Args:
            expert_divisor: New strategy value, or None for pure CPU mode.
        """
        # 1. Clear manager references from all layers
        for layer in self.model.layers:
            if hasattr(layer, 'gpu_prefill_manager'):
                layer.gpu_prefill_manager = None

        # 2. Delete old managers (frees GPU buffers)
        self.model.gpu_prefill_managers.clear()
        gc.collect()
        torch.cuda.empty_cache()

        if expert_divisor is None:
            # Pure CPU mode: no GPU prefill managers
            self.model.gpu_prefill_enabled = False
            return

        # 3. Ensure GPU prefill is enabled
        self.model.gpu_prefill_enabled = True
        self.model.expert_divisor = expert_divisor

        # 4. Recreate managers via the model's init method
        self.model._init_gpu_prefill()

        # 5. Sync back model.expert_divisor in case of OOM fallback
        # (e.g., persistent → layer_grouped(2) when experts don't fit)
        # The manager's internal _expert_divisor may differ after fallback,
        # and the model's forward routing uses model.expert_divisor.
        for manager in self.model.gpu_prefill_managers.values():
            if manager._expert_divisor != expert_divisor:
                actual = manager._expert_divisor
                logger.warning(
                    "Strategy %d fell back to %d (OOM) — syncing model",
                    expert_divisor, actual,
                )
                self.model.expert_divisor = actual
                break

        # 6. For HCS: initialize hot cached static experts
        if expert_divisor == -3 and self.heatmap_path:
            for manager in self.model.gpu_prefill_managers.values():
                manager._init_hot_cached_static(heatmap_path=self.heatmap_path)

    def _benchmark_prefill(self, tokens: List[int]) -> Dict:
        """Measure prefill speed over n_runs. Returns avg TTFT and tok/s."""
        runs = []

        # Warmup run (not timed)
        self.model.generate(self.small_tokens, max_new_tokens=1, temperature=0.6)

        for i in range(self.n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.model.generate(tokens, max_new_tokens=1, temperature=0.6)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            ttft = getattr(self.model, '_last_ttft', t1 - t0)
            tok_s = len(tokens) / ttft if ttft > 0 else 0
            runs.append({"ttft": round(ttft, 3), "tok_s": round(tok_s, 1)})

        avg_ttft = sum(r["ttft"] for r in runs) / len(runs)
        avg_tok_s = sum(r["tok_s"] for r in runs) / len(runs)
        mem = gpu_mem_mb()
        return {
            "avg_ttft": round(avg_ttft, 3),
            "avg_tok_s": round(avg_tok_s, 1),
            "vram_mb": {str(k): round(v, 0) for k, v in mem.items()},
            "runs": runs,
        }

    def _benchmark_decode(self, tokens: List[int]) -> Dict:
        """Measure decode speed over n_runs. Returns avg tok/s."""
        runs = []

        # Warmup run (not timed)
        self.model.generate(tokens, max_new_tokens=5, temperature=0.6, top_k=50)

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
            decode_tok_s = n_gen / decode_s if decode_s > 0 else 0
            runs.append({
                "decode_tok_s": round(decode_tok_s, 2),
                "n_gen": n_gen,
                "total_s": round(total_s, 2),
                "ttft": round(ttft, 3),
                "decode_s": round(decode_s, 2),
            })

        avg_tok_s = sum(r["decode_tok_s"] for r in runs) / len(runs)
        mem = gpu_mem_mb()
        return {
            "avg_tok_s": round(avg_tok_s, 2),
            "vram_mb": {str(k): round(v, 0) for k, v in mem.items()},
            "runs": runs,
        }

    def run(self):
        """Execute the full two-phase optimisation."""

        # ══════════════════════════════════════════════════════════
        # PHASE 1: GPU Prefill Strategy Selection
        # ══════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("PHASE 1: GPU PREFILL STRATEGY SELECTION")
        print(f"Testing with {len(self.large_tokens)} token prompt, {self.n_runs} runs each")
        print("=" * 70)

        for name, divisor, desc in PREFILL_STRATEGIES:
            print(f"\n{'─' * 50}")
            print(f"  Strategy: {name} (expert_divisor={divisor})")
            print(f"  {desc}")
            print(f"{'─' * 50}")
            try:
                t_switch = time.perf_counter()
                self._switch_strategy(divisor)
                t_switch = time.perf_counter() - t_switch
                print(f"  Switch time: {t_switch:.1f}s")

                result = self._benchmark_prefill(self.large_tokens)
                result["switch_time"] = round(t_switch, 1)
                self.results["prefill"][name] = result

                print(f"  Avg: {result['avg_tok_s']:.1f} tok/s, TTFT: {result['avg_ttft']:.2f}s")
                for i, run in enumerate(result["runs"]):
                    print(f"    Run {i+1}: {run['tok_s']:.1f} tok/s, TTFT: {run['ttft']:.2f}s")
                vram = result["vram_mb"]
                print(f"  VRAM: {', '.join(f'GPU{k}={v:.0f}MB' for k, v in vram.items())}")

            except torch.cuda.OutOfMemoryError:
                print(f"  SKIPPED: CUDA OOM")
                self.results["prefill"][name] = {"error": "CUDA OOM"}
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()
                self.results["prefill"][name] = {"error": str(e)}

        # Select best prefill strategy
        valid_prefill = [
            (name, r) for name, r in self.results["prefill"].items()
            if "avg_tok_s" in r
        ]
        if valid_prefill:
            best_name, best_result = max(valid_prefill, key=lambda x: x[1]["avg_tok_s"])
            self.results["best_prefill"] = best_name
            print(f"\n>>> BEST PREFILL: {best_name} ({best_result['avg_tok_s']:.1f} tok/s, "
                  f"TTFT: {best_result['avg_ttft']:.2f}s)")
        else:
            print("\n>>> NO VALID PREFILL STRATEGY FOUND")
            return self.results

        # ══════════════════════════════════════════════════════════
        # PHASE 2: Decode Strategy Selection
        # ══════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("PHASE 2: DECODE STRATEGY SELECTION")
        print(f"Testing with {len(self.small_tokens)} token prompt, "
              f"{self.decode_tokens} decode tokens, {self.n_runs} runs each")
        print("=" * 70)

        for name, divisor, desc in DECODE_STRATEGIES:
            print(f"\n{'─' * 50}")
            print(f"  Strategy: {name} (expert_divisor={divisor})")
            print(f"  {desc}")
            print(f"{'─' * 50}")
            try:
                t_switch = time.perf_counter()
                self._switch_strategy(divisor)
                t_switch = time.perf_counter() - t_switch
                print(f"  Switch time: {t_switch:.1f}s")

                result = self._benchmark_decode(self.small_tokens)
                result["switch_time"] = round(t_switch, 1)
                self.results["decode"][name] = result

                print(f"  Avg: {result['avg_tok_s']:.2f} tok/s")
                for i, run in enumerate(result["runs"]):
                    print(f"    Run {i+1}: {run['decode_tok_s']:.2f} tok/s "
                          f"({run['n_gen']} tok in {run['decode_s']:.1f}s, "
                          f"TTFT={run['ttft']:.2f}s)")
                vram = result["vram_mb"]
                print(f"  VRAM: {', '.join(f'GPU{k}={v:.0f}MB' for k, v in vram.items())}")

            except torch.cuda.OutOfMemoryError:
                print(f"  SKIPPED: CUDA OOM")
                self.results["decode"][name] = {"error": "CUDA OOM"}
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()
                self.results["decode"][name] = {"error": str(e)}

        # Select best decode strategy
        valid_decode = [
            (name, r) for name, r in self.results["decode"].items()
            if "avg_tok_s" in r
        ]
        if valid_decode:
            best_name, best_result = max(valid_decode, key=lambda x: x[1]["avg_tok_s"])
            self.results["best_decode"] = best_name
            print(f"\n>>> BEST DECODE: {best_name} ({best_result['avg_tok_s']:.2f} tok/s)")
        else:
            print("\n>>> NO VALID DECODE STRATEGY FOUND")

        # ══════════════════════════════════════════════════════════
        # Summary + Save
        # ══════════════════════════════════════════════════════════
        self._print_summary()
        self._save_results()
        return self.results

    def _print_summary(self):
        """Print a final summary table of all results."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        print("\nPrefill strategies (ranked by tok/s):")
        print(f"  {'Strategy':<20} {'tok/s':>10} {'TTFT (s)':>10} {'VRAM':>20}")
        print(f"  {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 20}")
        valid = [(n, r) for n, r in self.results["prefill"].items() if "avg_tok_s" in r]
        for name, r in sorted(valid, key=lambda x: -x[1]["avg_tok_s"]):
            vram_str = "+".join(f"{v:.0f}" for v in r["vram_mb"].values())
            marker = " <<<" if name == self.results["best_prefill"] else ""
            print(f"  {name:<20} {r['avg_tok_s']:>10.1f} {r['avg_ttft']:>10.2f} {vram_str:>20}{marker}")
        for name, r in self.results["prefill"].items():
            if "error" in r:
                print(f"  {name:<20} {'FAILED':>10} {r['error']}")

        print("\nDecode strategies (ranked by tok/s):")
        print(f"  {'Strategy':<20} {'tok/s':>10} {'VRAM':>20}")
        print(f"  {'─' * 20} {'─' * 10} {'─' * 20}")
        valid = [(n, r) for n, r in self.results["decode"].items() if "avg_tok_s" in r]
        for name, r in sorted(valid, key=lambda x: -x[1]["avg_tok_s"]):
            vram_str = "+".join(f"{v:.0f}" for v in r["vram_mb"].values())
            marker = " <<<" if name == self.results["best_decode"] else ""
            print(f"  {name:<20} {r['avg_tok_s']:>10.2f} {vram_str:>20}{marker}")
        for name, r in self.results["decode"].items():
            if "error" in r:
                print(f"  {name:<20} {'FAILED':>10} {r['error']}")

        print(f"\nOptimal pair: prefill={self.results['best_prefill']}, "
              f"decode={self.results['best_decode']}")
        print("=" * 70)

    def _save_results(self):
        """Save results to disk."""
        cache_dir = cache_dir_for_model(self.model_path)
        os.makedirs(cache_dir, exist_ok=True)
        save_path = os.path.join(cache_dir, "meta_optimiser.json")
        with open(save_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Krasis Meta-Optimiser")
    parser.add_argument("model", choices=list(MODELS.keys()),
                        help="Model to optimise")
    parser.add_argument("--prefill-tokens", type=int, default=10000,
                        help="Target prefill prompt length (default: 10000)")
    parser.add_argument("--decode-tokens", type=int, default=64,
                        help="Tokens to generate for decode test (default: 64)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per strategy (default: 3)")
    args = parser.parse_args()

    cfg = MODELS[args.model]
    opt = MetaOptimiser(
        model_path=cfg["model_path"],
        pp_partition=cfg["pp_partition"],
        devices=cfg["devices"],
        heatmap_path=cfg["heatmap_path"],
        prefill_tokens=args.prefill_tokens,
        decode_tokens=args.decode_tokens,
        n_runs=args.runs,
    )
    opt.run()


if __name__ == "__main__":
    main()
