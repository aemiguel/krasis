#!/usr/bin/env python3
"""Perplexity measurement for Krasis models.

Measures through the PRODUCTION forward path (GPU Marlin prefill or CPU-only)
to validate that quantization doesn't degrade model quality.

Supported datasets: wikitext-2, wikitext-103

Usage:
    # GPU prefill path (production, default wikitext-2):
    python -m perplexity.measure_ppl --model-path ~/.krasis/DeepSeek-V2-Lite --num-gpus 1

    # WikiText-103 (larger, more stable):
    python -m perplexity.measure_ppl --model-path ~/.krasis/DeepSeek-V2-Lite --dataset wikitext-103

    # CPU-only baseline (no GPU prefill):
    python -m perplexity.measure_ppl --model-path ~/.krasis/DeepSeek-V2-Lite --cpu-only

    # Quick test (first 5000 tokens):
    python -m perplexity.measure_ppl --model-path ~/.krasis/DeepSeek-V2-Lite --max-tokens 5000
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add krasis python path
_script_dir = Path(__file__).resolve().parent
_krasis_root = _script_dir.parent
sys.path.insert(0, str(_krasis_root / "python"))

from krasis.config import QuantConfig
from krasis.model import KrasisModel
from krasis.kv_cache import SequenceKVState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "wikitext-2": {
        "description": "WikiText-2 (standard LLM benchmark)",
        "tokens_approx": "~290K",
        "hf_path": "wikitext",
        "hf_name": "wikitext-2-raw-v1",
        "cache_file": "wikitext-2-raw-v1-test.txt",
    },
    "wikitext-103": {
        "description": "WikiText-103 (same test split as wikitext-2)",
        "tokens_approx": "~290K",
        "hf_path": "wikitext",
        "hf_name": "wikitext-103-raw-v1",
        "cache_file": "wikitext-103-raw-v1-test.txt",
    },
    "c4": {
        "description": "C4 validation (standard LLM benchmark, 364K docs)",
        "tokens_approx": "~200M",
        "hf_path": "allenai/c4",
        "hf_name": "en",
        "hf_split": "validation",
        "streaming": True,
        "cache_file": "c4-en-validation.txt",
    },
}

_datasets_dir = _script_dir / "datasets"


def load_dataset_text(dataset_name: str) -> str:
    """Load a dataset by name. Downloads and caches on first use.

    Args:
        dataset_name: Key from DATASETS registry (e.g. "wikitext-2")

    Returns:
        Full text of the dataset test/validation split.
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {', '.join(DATASETS.keys())}"
        )

    info = DATASETS[dataset_name]
    cache_path = _datasets_dir / info["cache_file"]

    if cache_path.exists():
        logger.info("Loading cached %s from %s", dataset_name, cache_path)
        return cache_path.read_text(encoding="utf-8")

    split = info.get("hf_split", "test")
    logger.info("Downloading %s (%s) %s split...", dataset_name, info["hf_name"], split)
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        print(
            "ERROR: 'datasets' library required. Install with: pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    if info.get("streaming"):
        # Stream large datasets — concatenate documents with double newlines
        ds = hf_load_dataset(info["hf_path"], info["hf_name"], split=split, streaming=True)
        chunks = []
        total_chars = 0
        for i, example in enumerate(ds):
            chunks.append(example["text"])
            total_chars += len(example["text"])
            if (i + 1) % 10000 == 0:
                print(f"\r  Streaming {dataset_name}: {i + 1} docs, {total_chars / 1e6:.1f}M chars", end="", flush=True)
        print(f"\r  Streaming {dataset_name}: {i + 1} docs, {total_chars / 1e6:.1f}M chars — done")
        text = "\n\n".join(chunks)
    else:
        ds = hf_load_dataset(info["hf_path"], info["hf_name"], split=split)
        text = "\n".join(ds["text"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    logger.info("Cached %s to %s (%d chars)", dataset_name, cache_path, len(text))
    return text


def list_datasets() -> list[dict]:
    """Return dataset info for menu display.

    Returns:
        List of dicts with keys: name, description, tokens_approx
    """
    return [
        {
            "name": name,
            "description": info["description"],
            "tokens_approx": info["tokens_approx"],
        }
        for name, info in DATASETS.items()
    ]


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

def evaluate_perplexity(
    model: KrasisModel,
    tokens: list,
    window_size: int,
    stride: int,
    max_tokens: int | None = None,
) -> dict:
    """Sliding-window perplexity evaluation.

    For each window of `window_size` tokens:
    - Run forward with return_all_logits=True to get [W, V] logits
    - Compute cross-entropy on the non-overlapping region (positions >= stride)
      to avoid double-counting, except for the first window which scores all

    Args:
        model: Loaded KrasisModel
        tokens: Full token list
        window_size: Context window per forward pass
        stride: Step size between windows (typically window_size // 2)
        max_tokens: If set, truncate tokens to this length

    Returns:
        Dict with ppl, bpc, mean_loss, num_tokens_scored, windows, elapsed_s
    """
    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    total_tokens = len(tokens)
    if total_tokens < 2:
        raise ValueError(f"Need at least 2 tokens, got {total_tokens}")

    device = torch.device(model.ranks[0].device)
    total_nll = 0.0
    total_scored = 0
    num_windows = 0

    t_start = time.perf_counter()

    # Compute window positions
    starts = list(range(0, total_tokens - 1, stride))
    total_windows = len(starts)

    for win_idx, begin in enumerate(starts):
        end = min(begin + window_size, total_tokens)
        win_len = end - begin

        if win_len < 2:
            break

        # Create fresh KV state for each window (no cross-window contamination)
        seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

        # Reset linear attention states for hybrid models
        if model.cfg.is_hybrid:
            for layer in model.layers:
                if layer.layer_type == "linear_attention":
                    layer.attention.reset_state()

        # Evict soft-tier HCS experts before prefill (same as Rust server)
        gpu_store = getattr(model, '_gpu_decode_store', None)
        evicted = 0
        if gpu_store is not None:
            evicted, _freed = gpu_store.py_hcs_evict_for_prefill(win_len)

        try:
            token_tensor = torch.tensor(tokens[begin:end], dtype=torch.long, device=device)
            positions = torch.arange(win_len, dtype=torch.int32, device=device)

            with torch.inference_mode():
                logits = model.forward(
                    token_tensor, positions, seq_states,
                    return_all_logits=True,
                )  # [W, V]

            # Shift: logits[:-1] predicts tokens[1:]
            shift_logits = logits[:-1, :].float()  # [W-1, V]
            shift_labels = token_tensor[1:]          # [W-1]

            # Cross-entropy per position
            loss_per_pos = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, reduction="none"
            )  # [W-1]

            # Only score the non-overlapping region to avoid double-counting.
            # For the first window (begin == 0), score everything.
            # For subsequent windows, only score positions >= stride.
            if begin == 0:
                score_start = 0
            else:
                # In the shifted label space, position i corresponds to
                # predicting token at original position (begin + i + 1).
                # We want to score original positions >= begin + stride,
                # which means shifted positions where (begin + i + 1) >= (begin + stride)
                # i.e., i >= stride - 1
                score_start = stride - 1

            scored_loss = loss_per_pos[score_start:]
            n_scored = scored_loss.shape[0]

            if n_scored > 0:
                total_nll += scored_loss.sum().item()
                total_scored += n_scored

            num_windows += 1

            # Free logits immediately (can be ~1.2 GB for large vocab)
            del logits, shift_logits, loss_per_pos, scored_loss

        finally:
            # Reload soft-tier HCS experts after prefill
            if gpu_store is not None and evicted > 0:
                gpu_store.py_hcs_reload_after_prefill()
            for s in seq_states:
                s.free()

        # Progress — print every window on its own line so running PPL is visible
        elapsed = time.perf_counter() - t_start
        running_ppl = math.exp(total_nll / total_scored) if total_scored > 0 else float("inf")
        tok_per_s = total_scored / elapsed if elapsed > 0 else 0
        pct = 100.0 * (win_idx + 1) / total_windows
        print(
            f"  [{pct:5.1f}%] Window {win_idx + 1}/{total_windows} | "
            f"scored {total_scored}/{total_tokens} | "
            f"PPL={running_ppl:.4f} | "
            f"{tok_per_s:.0f} tok/s",
            flush=True,
        )

    elapsed_s = time.perf_counter() - t_start
    print()  # newline after progress

    if total_scored == 0:
        raise ValueError("No tokens scored — check window/stride settings")

    mean_loss = total_nll / total_scored
    ppl = math.exp(mean_loss)
    bpc = mean_loss / math.log(2)

    return {
        "perplexity": ppl,
        "bits_per_char": bpc,
        "mean_loss": mean_loss,
        "total_nll": total_nll,
        "num_tokens_scored": total_scored,
        "num_tokens_total": total_tokens,
        "num_windows": num_windows,
        "window_size": window_size,
        "stride": stride,
        "elapsed_s": elapsed_s,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(results: dict, config: dict, output_dir: Path):
    """Save results as JSON and human-readable log."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(config["model_path"]).name
    dataset_name = config.get("dataset", "wikitext-2")
    mode = "cpu" if config.get("cpu_only") else "gpu"
    gpu_bits = config.get("gpu_expert_bits", 4)
    cpu_bits = config.get("cpu_expert_bits", 4)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    basename = f"{model_name}_{dataset_name}_{mode}_g{gpu_bits}_c{cpu_bits}_{timestamp}"

    # JSON
    combined = {"config": config, "results": results}
    json_path = output_dir / f"{basename}.json"
    json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    # Human-readable log
    log_path = output_dir / f"{basename}.log"
    lines = [
        f"Krasis Perplexity Measurement",
        f"{'=' * 50}",
        f"Date:          {datetime.now().isoformat()}",
        f"Model:         {config['model_path']}",
        f"Dataset:       {dataset_name}",
        f"Mode:          {'CPU-only' if config.get('cpu_only') else 'GPU prefill (Marlin)'}",
        f"GPUs:          {config.get('num_gpus', 'auto')}",
        f"GPU bits:      {gpu_bits}",
        f"CPU bits:      {cpu_bits}",
        f"Window size:   {results['window_size']}",
        f"Stride:        {results['stride']}",
        f"",
        f"Results",
        f"{'-' * 50}",
        f"Perplexity:    {results['perplexity']:.4f}",
        f"BPC:           {results['bits_per_char']:.4f}",
        f"Mean loss:     {results['mean_loss']:.6f}",
        f"Tokens scored: {results['num_tokens_scored']} / {results['num_tokens_total']}",
        f"Windows:       {results['num_windows']}",
        f"Elapsed:       {results['elapsed_s']:.1f}s",
        f"Throughput:    {results['num_tokens_scored'] / results['elapsed_s']:.0f} tok/s",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return json_path, log_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_config_file(config_path: str) -> dict:
    """Parse a Krasis .conf file into a dict of settings.

    Uses the same key mapping as server.py so any testconfig .conf file works.
    """
    _CFG_KEY_MAP = {
        "MODEL_PATH": "model_path",
        "CFG_SELECTED_GPUS": "_selected_gpus",
        "CFG_LAYER_GROUP_SIZE": "layer_group_size",
        "CFG_KV_DTYPE": "kv_dtype",
        "CFG_GPU_EXPERT_BITS": "gpu_expert_bits",
        "CFG_CPU_EXPERT_BITS": "cpu_expert_bits",
        "CFG_ATTENTION_QUANT": "attention_quant",
        "CFG_SHARED_EXPERT_QUANT": "shared_expert_quant",
        "CFG_DENSE_MLP_QUANT": "dense_mlp_quant",
        "CFG_LM_HEAD_QUANT": "lm_head_quant",
        "CFG_KRASIS_THREADS": "krasis_threads",
        "CFG_KV_CACHE_MB": "kv_cache_mb",
    }
    defaults = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key in _CFG_KEY_MAP:
                dest = _CFG_KEY_MAP[key]
                if dest is None:
                    continue
                if key == "CFG_SELECTED_GPUS":
                    gpu_list = [x.strip() for x in val.split(",") if x.strip()]
                    if gpu_list:
                        defaults["num_gpus"] = len(gpu_list)
                    continue
            else:
                dest = key.replace("-", "_").lower()
            # Type conversion
            try:
                defaults[dest] = int(val)
            except ValueError:
                try:
                    defaults[dest] = float(val)
                except ValueError:
                    defaults[dest] = val
    if "model_path" in defaults and isinstance(defaults["model_path"], str):
        defaults["model_path"] = os.path.expanduser(defaults["model_path"])
    return defaults


def parse_args():
    ds_names = ", ".join(DATASETS.keys())
    p = argparse.ArgumentParser(
        description="Measure perplexity through Krasis production path",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file (loads defaults, CLI overrides)
    p.add_argument("--config", default=None,
                   help="Path to .conf file (same format as testconfigs). CLI args override.")

    # Model
    p.add_argument("--model-path", default=None, help="Path to HF model directory")
    p.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs (auto-detect if omitted)")

    # Dataset
    p.add_argument("--dataset", default="wikitext-2", choices=list(DATASETS.keys()),
                    help=f"Dataset to evaluate on (default: wikitext-2). Available: {ds_names}")

    # Eval params
    p.add_argument("--window-size", type=int, default=2048, help="Context window per forward pass (default: 2048)")
    p.add_argument("--stride", type=int, default=None, help="Stride between windows (default: window_size // 2)")
    p.add_argument("--max-tokens", type=int, default=None, help="Truncate dataset to N tokens (for quick tests)")

    # Quantization
    p.add_argument("--gpu-expert-bits", type=int, default=None, help="GPU Marlin expert bits (default: 4)")
    p.add_argument("--cpu-expert-bits", type=int, default=None, help="CPU expert bits (default: 4)")
    p.add_argument("--attention-quant", default=None, choices=["bf16", "adaptive_fp8"], help="Attention quantization")
    p.add_argument("--lm-head-quant", default=None, choices=["bf16", "int8"], help="LM head quantization")

    # Paths
    p.add_argument("--cpu-only", action="store_true", help="Disable GPU prefill — measure through CPU-only path")
    p.add_argument("--layer-group-size", type=int, default=None, help="Layers per group for prefill DMA")
    p.add_argument("--kv-cache-mb", type=int, default=None, help="KV cache size in MB")
    p.add_argument("--krasis-threads", type=int, default=None, help="CPU threads for Rust engine")

    args = p.parse_args()

    # Load config file defaults, then apply CLI overrides
    if args.config:
        if not os.path.isfile(args.config):
            print(f"Error: config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        cfg = parse_config_file(args.config)
        # Only override if CLI arg was not explicitly set
        if args.model_path is None:
            args.model_path = cfg.get("model_path")
        if args.num_gpus is None:
            args.num_gpus = cfg.get("num_gpus")
        if args.gpu_expert_bits is None:
            args.gpu_expert_bits = cfg.get("gpu_expert_bits", 4)
        if args.cpu_expert_bits is None:
            args.cpu_expert_bits = cfg.get("cpu_expert_bits", 4)
        if args.attention_quant is None:
            args.attention_quant = cfg.get("attention_quant", "bf16")
        if args.lm_head_quant is None:
            args.lm_head_quant = cfg.get("lm_head_quant", "int8")
        if args.layer_group_size is None:
            args.layer_group_size = cfg.get("layer_group_size", 2)
        if args.kv_cache_mb is None:
            args.kv_cache_mb = cfg.get("kv_cache_mb", 1000)
        if args.krasis_threads is None:
            args.krasis_threads = cfg.get("krasis_threads", 48)

    # Apply defaults for anything still None
    if args.model_path is None:
        print("Error: --model-path or --config required", file=sys.stderr)
        sys.exit(1)
    if args.gpu_expert_bits is None:
        args.gpu_expert_bits = 4
    if args.cpu_expert_bits is None:
        args.cpu_expert_bits = 4
    if args.attention_quant is None:
        args.attention_quant = "bf16"
    if args.lm_head_quant is None:
        args.lm_head_quant = "int8"
    if args.layer_group_size is None:
        args.layer_group_size = 2
    if args.kv_cache_mb is None:
        args.kv_cache_mb = 1000
    if args.krasis_threads is None:
        args.krasis_threads = 48

    return args


def run_perplexity(
    model: KrasisModel,
    dataset_name: str = "wikitext-2",
    window_size: int = 2048,
    stride: int | None = None,
    max_tokens: int | None = None,
    config: dict | None = None,
) -> dict:
    """End-to-end perplexity evaluation: load text, tokenize, evaluate, save.

    This is the reusable entry point called by both standalone CLI and the
    server.py launcher.

    Args:
        model: Loaded KrasisModel
        dataset_name: Key from DATASETS registry
        window_size: Context window per forward pass
        stride: Step size between windows (default: window_size // 2)
        max_tokens: Truncate dataset to N tokens
        config: Optional config dict for save_results (adds model_path etc.)

    Returns:
        Results dict with ppl, bpc, log_path, etc.
    """
    if stride is None:
        stride = window_size // 2

    ds_info = DATASETS[dataset_name]

    print(f"\n  Loading {ds_info['description']}...")
    text = load_dataset_text(dataset_name)
    print(f"  Dataset: {len(text):,} chars")

    # Pre-truncate text before tokenizing to avoid tokenizing entire huge corpus.
    # Use ~5 chars/token as conservative estimate (overshoots slightly, then trim).
    if max_tokens and len(text) > max_tokens * 6:
        char_limit = max_tokens * 6
        print(f"  Pre-truncating text to ~{char_limit / 1e6:.1f}M chars (for {max_tokens:,} token target)")
        text = text[:char_limit]

    # Tokenize (raw text, no chat template)
    print("  Tokenizing...")
    tokens = model.tokenizer.encode(text, add_special_tokens=False)
    print(f"  Tokens: {len(tokens):,}")
    if max_tokens:
        tokens = tokens[:max_tokens]
        print(f"  Truncated to: {len(tokens):,}")
    print()

    # Evaluate
    print("  Evaluating perplexity...")
    results = evaluate_perplexity(
        model=model,
        tokens=tokens,
        window_size=window_size,
        stride=stride,
    )
    results["dataset"] = dataset_name

    # Save
    save_config = dict(config) if config else {}
    save_config.setdefault("model_path", model.cfg.model_path)
    save_config["dataset"] = dataset_name
    save_config["window_size"] = window_size
    save_config["stride"] = stride
    save_config["max_tokens"] = max_tokens

    results_dir = _script_dir / "results"
    json_path, log_path = save_results(results, save_config, results_dir)
    results["json_path"] = str(json_path)
    results["log_path"] = str(log_path)

    # Print summary
    tok_per_s = results["num_tokens_scored"] / results["elapsed_s"] if results["elapsed_s"] > 0 else 0
    print()
    bar = "\u2550" * 56
    print(bar)
    print(f"  PERPLEXITY COMPLETE \u2014 {dataset_name}")
    print(bar)
    print(f"  Perplexity:    {results['perplexity']:.2f}")
    print(f"  BPC:           {results['bits_per_char']:.2f}")
    print(f"  Tokens scored: {results['num_tokens_scored']:,}")
    print(f"  Elapsed:       {results['elapsed_s']:.1f}s ({tok_per_s:.0f} tok/s)")
    print(f"  Log:           {log_path}")
    print(bar)

    return results


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    stride = args.stride if args.stride is not None else args.window_size // 2

    print(f"Krasis Perplexity Measurement")
    print(f"{'=' * 50}")
    if args.config:
        print(f"Config:      {args.config}")
    print(f"Model:       {args.model_path}")
    print(f"Dataset:     {args.dataset}")
    print(f"Mode:        {'CPU-only' if args.cpu_only else 'GPU prefill (Marlin)'}")
    print(f"GPU bits:    {args.gpu_expert_bits}")
    print(f"CPU bits:    {args.cpu_expert_bits}")
    print(f"Window:      {args.window_size}, stride={stride}")
    if args.max_tokens:
        print(f"Max tokens:  {args.max_tokens}")

    # ── Load model ──
    print("\nLoading model...")
    quant_cfg = QuantConfig(
        attention=args.attention_quant,
        shared_expert="int8",
        dense_mlp="int8",
        lm_head=args.lm_head_quant,
        gpu_expert_bits=args.gpu_expert_bits,
        cpu_expert_bits=args.cpu_expert_bits,
    )

    model = KrasisModel(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        layer_group_size=args.layer_group_size,
        kv_dtype=torch.float8_e4m3fn,
        quant_cfg=quant_cfg,
        krasis_threads=args.krasis_threads,
        gpu_prefill=not args.cpu_only,
        kv_cache_mb=args.kv_cache_mb,
    )
    model.load()

    config = {
        "model_path": args.model_path,
        "num_gpus": args.num_gpus,
        "cpu_only": args.cpu_only,
        "gpu_expert_bits": args.gpu_expert_bits,
        "cpu_expert_bits": args.cpu_expert_bits,
        "attention_quant": args.attention_quant,
        "lm_head_quant": args.lm_head_quant,
        "layer_group_size": args.layer_group_size,
        "krasis_threads": args.krasis_threads,
        "kv_cache_mb": args.kv_cache_mb,
    }

    # ── Run perplexity ──
    run_perplexity(
        model=model,
        dataset_name=args.dataset,
        window_size=args.window_size,
        stride=stride,
        max_tokens=args.max_tokens,
        config=config,
    )


if __name__ == "__main__":
    main()
