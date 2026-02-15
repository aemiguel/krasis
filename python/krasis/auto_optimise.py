"""Auto-optimiser: discovers best prefill + decode strategy on first run.

First launch: builds heatmap, benchmarks strategies, saves results.
Subsequent launches: loads cached results and applies in ~2 seconds.

Results cached in {model_path}/.krasis_cache/auto_optimise.json
Heatmap cached in {model_path}/.krasis_cache/auto_heatmap.json

Usage (from server.py):
    from krasis.auto_optimise import auto_optimise_or_load
    auto_optimise_or_load(model, force=False)
"""

import gc
import hashlib
import json
import logging
import os
import subprocess
import time
import traceback
from typing import Dict, List, Optional, Tuple

import torch

from krasis.timing import TIMING

logger = logging.getLogger("krasis.auto_optimise")

# ══════════════════════════════════════════════════════════════════
# Strategy definitions (same as meta_optimiser.py)
# ══════════════════════════════════════════════════════════════════

PREFILL_STRATEGIES = [
    ("persistent",       1, "All experts in VRAM (zero DMA) — fastest if they fit"),
    ("hcs_prefill",     -3, "1-layer groups, hot experts skip DMA"),
    ("layer_grouped_2",  2, "2-layer groups, amortised DMA"),
    ("layer_grouped_4",  4, "4-layer groups, fewer DMA rounds"),
    ("active_only",     -1, "DMA only activated experts per layer"),
    ("chunked",          0, "Legacy per-layer full DMA"),
]

DECODE_STRATEGIES = [
    ("hcs_hybrid", -3, "Hot experts GPU + cold experts CPU in parallel"),
    ("compact",    -1, "Cross-token LRU caching in per-layer VRAM slots"),
    ("lru",        -2, "Cross-layer LRU expert caching"),
    ("pure_cpu",  None, "CPU-only decode (no GPU involvement)"),
]

CACHE_FILE = "auto_optimise.json"
HEATMAP_FILE = "auto_heatmap.json"


def _gpu_mem_mb() -> Dict[int, float]:
    """Current GPU memory usage per device in MB."""
    return {i: torch.cuda.memory_allocated(i) / 1e6
            for i in range(torch.cuda.device_count())}


def _hardware_fingerprint() -> str:
    """Hash of GPU model/count/VRAM for cache invalidation."""
    parts = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts.append(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        parts.append("no-nvidia-smi")

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _model_fingerprint(model_path: str, pp_partition: List[int],
                       quant_cfg) -> str:
    """Hash of model path, PP partition, and quantization config."""
    parts = [
        os.path.abspath(model_path),
        str(pp_partition),
        str(getattr(quant_cfg, 'gpu_expert_bits', 4)),
        str(getattr(quant_cfg, 'cpu_expert_bits', 4)),
        str(getattr(quant_cfg, 'attention', 'int8')),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class AutoOptimiser:
    """Discovers optimal prefill + decode strategy for a loaded KrasisModel."""

    def __init__(
        self,
        model,  # KrasisModel, already loaded
        heatmap_path: Optional[str] = None,
        prefill_tokens: int = 10000,
        decode_tokens: int = 64,
        n_runs: int = 3,
    ):
        self.model = model
        self.model_path = model.cfg.model_path
        self.cache_dir = os.path.join(self.model_path, ".krasis_cache")
        self.heatmap_path = heatmap_path
        self.prefill_target = prefill_tokens
        self.decode_tokens = decode_tokens
        self.n_runs = n_runs

        # Ensure instrumentation is OFF for benchmarking
        TIMING.decode = False
        TIMING.prefill = False
        os.environ.pop("KRASIS_DECODE_TIMING", None)

        # Build prompts
        self.large_tokens = self._make_prompt(prefill_tokens)
        self.small_tokens = self._make_short_prompt()

        # Results storage
        self.results: Dict = {
            "model": os.path.basename(self.model_path),
            "pp_partition": model.pp_partition,
            "hardware_fingerprint": _hardware_fingerprint(),
            "model_fingerprint": _model_fingerprint(
                self.model_path, model.pp_partition, model.quant_cfg,
            ),
            "prefill_prompt_tokens": len(self.large_tokens),
            "decode_prompt_tokens": len(self.small_tokens),
            "decode_gen_tokens": decode_tokens,
            "n_runs": n_runs,
            "prefill": {},
            "decode": {},
            "best_prefill": None,
            "best_decode": None,
            "best_prefill_divisor": None,
            "best_decode_divisor": None,
        }

    # ──────────────────────────────────────────────────────────
    # Prompt building
    # ──────────────────────────────────────────────────────────

    def _make_prompt(self, target_tokens: int) -> List[int]:
        """Generate a chat prompt of approximately target_tokens length."""
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
        content = ""
        while True:
            for section in sections:
                content += section
            messages = [{"role": "user", "content": content}]
            tokens = self.model.tokenizer.apply_chat_template(messages)
            if len(tokens) >= target_tokens:
                return tokens[:target_tokens]

    def _make_short_prompt(self) -> List[int]:
        """Generate a short chat prompt for decode testing."""
        messages = [{"role": "user", "content": "Write a poem about recursion in programming."}]
        return self.model.tokenizer.apply_chat_template(messages)

    # ──────────────────────────────────────────────────────────
    # Strategy switching (reused from meta_optimiser.py)
    # ──────────────────────────────────────────────────────────

    def _switch_strategy(self, expert_divisor: Optional[int]):
        """Reinitialize GPU prefill managers with a new strategy."""
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
        heatmap = self._resolve_heatmap_path()
        if expert_divisor == -3 and heatmap:
            for manager in self.model.gpu_prefill_managers.values():
                manager._init_hot_cached_static(heatmap_path=heatmap)

    def _resolve_heatmap_path(self) -> Optional[str]:
        """Return heatmap path: explicit > auto-built > None."""
        if self.heatmap_path and os.path.exists(self.heatmap_path):
            return self.heatmap_path
        auto_path = os.path.join(self.cache_dir, HEATMAP_FILE)
        if os.path.exists(auto_path):
            return auto_path
        return None

    # ──────────────────────────────────────────────────────────
    # Phase 0: Auto-build heatmap
    # ──────────────────────────────────────────────────────────

    def _build_heatmap(self) -> str:
        """Run calibration prompt to build expert activation heatmap.

        Uses active_only mode (cheapest that tracks activations).
        Returns path to saved heatmap file.
        """
        heatmap_path = os.path.join(self.cache_dir, HEATMAP_FILE)
        if os.path.exists(heatmap_path):
            logger.info("Heatmap already exists at %s — reusing", heatmap_path)
            return heatmap_path

        logger.info("Phase 0: Building expert heatmap (~%d tokens + %d decode)...",
                     len(self.large_tokens), 128)

        # Switch to active_only (cheapest mode that tracks activations)
        self._switch_strategy(-1)

        # Run calibration: large prefill + some decode tokens
        self.model.generate(
            self.large_tokens,
            max_new_tokens=128,
            temperature=0.6,
            top_k=50,
        )

        # Save heatmap from each manager
        os.makedirs(self.cache_dir, exist_ok=True)
        for manager in self.model.gpu_prefill_managers.values():
            manager.save_heatmap(heatmap_path)
            break  # Only need one (heatmaps are per-layer, managers cover disjoint layers)

        # If multiple managers (PP), merge heatmaps
        if len(self.model.gpu_prefill_managers) > 1:
            merged = {}
            for manager in self.model.gpu_prefill_managers.values():
                for (layer, expert), count in manager._heatmap.items():
                    key = f"{layer},{expert}"
                    merged[key] = merged.get(key, 0) + count
            with open(heatmap_path, "w") as f:
                json.dump(merged, f, indent=2)

        logger.info("Heatmap saved to %s", heatmap_path)
        return heatmap_path

    # ──────────────────────────────────────────────────────────
    # Benchmarking
    # ──────────────────────────────────────────────────────────

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
        mem = _gpu_mem_mb()
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
        mem = _gpu_mem_mb()
        return {
            "avg_tok_s": round(avg_tok_s, 2),
            "vram_mb": {str(k): round(v, 0) for k, v in mem.items()},
            "runs": runs,
        }

    # ──────────────────────────────────────────────────────────
    # Main run loop
    # ──────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """Execute the full optimisation: Phase 0 (heatmap) + Phase 1 (prefill) + Phase 2 (decode)."""
        print("=" * 70)
        print("KRASIS AUTO-OPTIMISER")
        print("=" * 70)
        print(f"  Model:  {os.path.basename(self.model_path)}")
        print(f"  PP:     {self.model.pp_partition}")
        print(f"  Target prefill: ~{self.prefill_target} tokens")
        print(f"  Decode tokens:  {self.decode_tokens}")
        print(f"  Runs per test:  {self.n_runs}")
        print()

        # Phase 0: Build heatmap (needed for HCS strategies)
        self.heatmap_path = self._build_heatmap()

        # Phase 1: GPU Prefill Strategy Selection
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
                result["divisor"] = divisor
                self.results["prefill"][name] = result

                print(f"  Avg: {result['avg_tok_s']:.1f} tok/s, TTFT: {result['avg_ttft']:.2f}s")
                for j, run in enumerate(result["runs"]):
                    print(f"    Run {j+1}: {run['tok_s']:.1f} tok/s, TTFT: {run['ttft']:.2f}s")
                vram = result["vram_mb"]
                print(f"  VRAM: {', '.join(f'GPU{k}={v:.0f}MB' for k, v in vram.items())}")

            except torch.cuda.OutOfMemoryError:
                print("  SKIPPED: CUDA OOM")
                self.results["prefill"][name] = {"error": "CUDA OOM", "divisor": divisor}
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()
                self.results["prefill"][name] = {"error": str(e), "divisor": divisor}

        # Select best prefill strategy
        valid_prefill = [
            (name, r) for name, r in self.results["prefill"].items()
            if "avg_tok_s" in r
        ]
        if valid_prefill:
            best_name, best_result = max(valid_prefill, key=lambda x: x[1]["avg_tok_s"])
            self.results["best_prefill"] = best_name
            self.results["best_prefill_divisor"] = best_result["divisor"]
            print(f"\n>>> BEST PREFILL: {best_name} ({best_result['avg_tok_s']:.1f} tok/s, "
                  f"TTFT: {best_result['avg_ttft']:.2f}s)")
        else:
            print("\n>>> NO VALID PREFILL STRATEGY FOUND")
            self._save_results()
            return self.results

        # Phase 2: Decode Strategy Selection
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
                result["divisor"] = divisor
                self.results["decode"][name] = result

                print(f"  Avg: {result['avg_tok_s']:.2f} tok/s")
                for j, run in enumerate(result["runs"]):
                    print(f"    Run {j+1}: {run['decode_tok_s']:.2f} tok/s "
                          f"({run['n_gen']} tok in {run['decode_s']:.1f}s, "
                          f"TTFT={run['ttft']:.2f}s)")
                vram = result["vram_mb"]
                print(f"  VRAM: {', '.join(f'GPU{k}={v:.0f}MB' for k, v in vram.items())}")

            except torch.cuda.OutOfMemoryError:
                print("  SKIPPED: CUDA OOM")
                self.results["decode"][name] = {"error": "CUDA OOM", "divisor": divisor}
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()
                self.results["decode"][name] = {"error": str(e), "divisor": divisor}

        # Select best decode strategy
        valid_decode = [
            (name, r) for name, r in self.results["decode"].items()
            if "avg_tok_s" in r
        ]
        if valid_decode:
            best_name, best_result = max(valid_decode, key=lambda x: x[1]["avg_tok_s"])
            self.results["best_decode"] = best_name
            self.results["best_decode_divisor"] = best_result["divisor"]
            print(f"\n>>> BEST DECODE: {best_name} ({best_result['avg_tok_s']:.2f} tok/s)")
        else:
            print("\n>>> NO VALID DECODE STRATEGY FOUND")

        self._print_summary()
        self._save_results()
        return self.results

    def _print_summary(self):
        """Print a final summary table."""
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
            if "error" in r and "avg_tok_s" not in r:
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
            if "error" in r and "avg_tok_s" not in r:
                print(f"  {name:<20} {'FAILED':>10} {r['error']}")

        print(f"\nOptimal pair: prefill={self.results['best_prefill']}, "
              f"decode={self.results['best_decode']}")
        print("=" * 70)

    def _save_results(self):
        """Save results to disk."""
        os.makedirs(self.cache_dir, exist_ok=True)
        save_path = os.path.join(self.cache_dir, CACHE_FILE)
        with open(save_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info("Auto-optimise results saved to %s", save_path)


# ══════════════════════════════════════════════════════════════════
# Cache validation + apply
# ══════════════════════════════════════════════════════════════════

def _load_cached(model) -> Optional[Dict]:
    """Load cached auto-optimise results if valid."""
    cache_dir = os.path.join(model.cfg.model_path, ".krasis_cache")
    cache_path = os.path.join(cache_dir, CACHE_FILE)

    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path) as f:
            cached = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read auto-optimise cache — re-running")
        return None

    # Validate fingerprints
    hw_fp = _hardware_fingerprint()
    model_fp = _model_fingerprint(
        model.cfg.model_path, model.pp_partition, model.quant_cfg,
    )

    if cached.get("hardware_fingerprint") != hw_fp:
        logger.info("Hardware changed (was %s, now %s) — re-optimising",
                     cached.get("hardware_fingerprint"), hw_fp)
        return None
    if cached.get("model_fingerprint") != model_fp:
        logger.info("Model config changed — re-optimising")
        return None

    # Check that best strategies were actually found
    if not cached.get("best_prefill") or not cached.get("best_decode"):
        logger.info("Previous optimisation incomplete — re-running")
        return None

    # If HCS was selected, verify heatmap file exists
    if cached.get("best_decode_divisor") == -3:
        heatmap_path = os.path.join(cache_dir, HEATMAP_FILE)
        if not os.path.exists(heatmap_path):
            logger.info("HCS selected but heatmap missing — re-optimising")
            return None

    return cached


def _apply_result(model, result: Dict):
    """Configure a loaded KrasisModel based on auto-optimise results.

    Strategy selection logic:
    - Use decode winner's divisor (decode is sustained user experience)
    - Unless it severely hurts prefill (>3x slower TTFT)
    - For pure_cpu decode: keep GPU prefill enabled but with best prefill divisor
    """
    best_decode = result["best_decode"]
    best_prefill = result["best_prefill"]
    decode_divisor = result.get("best_decode_divisor")
    prefill_divisor = result.get("best_prefill_divisor")

    cache_dir = os.path.join(model.cfg.model_path, ".krasis_cache")
    heatmap_path = os.path.join(cache_dir, HEATMAP_FILE)

    logger.info(
        "Applying auto-optimise: prefill=%s (div=%s), decode=%s (div=%s)",
        best_prefill, prefill_divisor, best_decode, decode_divisor,
    )

    # Clear existing GPU prefill managers
    for layer in model.layers:
        if hasattr(layer, 'gpu_prefill_manager'):
            layer.gpu_prefill_manager = None
    model.gpu_prefill_managers.clear()
    gc.collect()
    torch.cuda.empty_cache()

    if best_decode == "pure_cpu":
        # CPU decode: use GPU for prefill only (large prompts)
        # Keep prefill divisor for GPU prefill, threshold high so M=1 stays CPU
        model.gpu_prefill_enabled = True
        model.expert_divisor = prefill_divisor if prefill_divisor is not None else 0
        model.gpu_prefill_threshold = 300
        model._init_gpu_prefill()
        logger.info(
            "Strategy: pure_cpu decode, GPU prefill with divisor=%d, threshold=%d",
            model.expert_divisor, model.gpu_prefill_threshold,
        )
    elif best_decode == "hcs_hybrid":
        # HCS: hot experts on GPU (static), cold on CPU (parallel)
        model.gpu_prefill_enabled = True
        model.expert_divisor = -3
        model.gpu_prefill_threshold = 1  # GPU handles all, including M=1
        model._init_gpu_prefill()

        # Initialize HCS with heatmap
        for manager in model.gpu_prefill_managers.values():
            manager._init_hot_cached_static(
                heatmap_path=heatmap_path if os.path.exists(heatmap_path) else None,
            )
        logger.info("Strategy: hcs_hybrid, threshold=1, heatmap=%s", heatmap_path)

    elif best_decode == "compact":
        # Compact: cross-token LRU caching in per-layer VRAM slots
        model.gpu_prefill_enabled = True
        model.expert_divisor = -1
        model.gpu_prefill_threshold = 1
        model._init_gpu_prefill()
        logger.info("Strategy: compact (active_only with cross-token caching), threshold=1")

    elif best_decode == "lru":
        # LRU: cross-layer LRU expert caching
        model.gpu_prefill_enabled = True
        model.expert_divisor = -2
        model.gpu_prefill_threshold = 1
        model._init_gpu_prefill()
        logger.info("Strategy: lru, threshold=1")

    else:
        # Fallback: use decode divisor directly
        if decode_divisor is not None:
            model.gpu_prefill_enabled = True
            model.expert_divisor = decode_divisor
            model.gpu_prefill_threshold = 1
            model._init_gpu_prefill()
        else:
            model.gpu_prefill_enabled = False
        logger.info("Strategy: fallback divisor=%s", decode_divisor)


# ══════════════════════════════════════════════════════════════════
# Top-level entry point
# ══════════════════════════════════════════════════════════════════

def auto_optimise_or_load(model, force: bool = False) -> Dict:
    """Auto-optimise or load cached results. Called from server.py.

    Args:
        model: Loaded KrasisModel (expert_divisor=0, neutral start)
        force: If True, re-run even if valid cache exists

    Returns:
        Dict of optimisation results
    """
    if not force:
        cached = _load_cached(model)
        if cached is not None:
            logger.info(
                "Loading cached auto-optimise results: prefill=%s, decode=%s",
                cached["best_prefill"], cached["best_decode"],
            )
            print(f"Auto-optimise: loading cached results "
                  f"(prefill={cached['best_prefill']}, decode={cached['best_decode']})")
            _apply_result(model, cached)
            return cached

    # Run full optimisation
    opt = AutoOptimiser(model)
    result = opt.run()

    # Apply the winning strategy
    if result.get("best_decode"):
        _apply_result(model, result)
    else:
        logger.warning("No valid strategy found — keeping neutral config")

    return result
