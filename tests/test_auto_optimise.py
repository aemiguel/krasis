#!/usr/bin/env python3
"""Generic auto-optimiser test for any Krasis-supported model.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python tests/test_auto_optimise.py \
        --model models/Qwen3-Coder-Next --num-gpus 2

    CUDA_VISIBLE_DEVICES=0,1 python tests/test_auto_optimise.py \
        --model models/Qwen3-235B-A22B --num-gpus 2 --force
"""

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from krasis.model import KrasisModel
from krasis.auto_optimise import auto_optimise_or_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="Force re-run even if cached")
    parser.add_argument("--prefill-tokens", type=int, default=10000)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    model_name = os.path.basename(model_path)
    num_gpus = args.num_gpus

    # Read num_layers from config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    num_layers = config["num_hidden_layers"]

    # Compute balanced PP partition
    base = num_layers // num_gpus
    remainder = num_layers % num_gpus
    pp_partition = [base + (1 if i < remainder else 0) for i in range(num_gpus)]

    devices = [f"cuda:{i}" for i in range(num_gpus)]

    print(f"=== Auto-Optimise: {model_name} ===")
    print(f"  Layers: {num_layers}, GPUs: {num_gpus}, PP: {pp_partition}")
    print(f"  Devices: {devices}")
    print(f"  Prefill tokens: {args.prefill_tokens}, Decode tokens: {args.decode_tokens}")
    print(f"  Runs per test: {args.n_runs}")

    t0 = time.perf_counter()
    model = KrasisModel(
        model_path=model_path,
        pp_partition=pp_partition,
        devices=devices,
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        gpu_prefill_threshold=300,
        expert_divisor=0,  # neutral start
    )
    model.load()
    t_load = time.perf_counter() - t0
    print(f"Model loaded in {t_load:.1f}s")

    # Delete cached results if --force
    cache_file = os.path.join(model_path, ".krasis_cache", "auto_optimise.json")
    if args.force and os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Removed cached results: {cache_file}")

    t_opt_start = time.perf_counter()
    result = auto_optimise_or_load(model, force=args.force)
    t_opt = time.perf_counter() - t_opt_start

    print(f"\n{'='*70}")
    print(f"COMPLETE: {model_name} ({num_gpus} GPU(s))")
    print(f"  Optimise time: {t_opt:.1f}s")
    print(f"  Best prefill: {result.get('best_prefill')} "
          f"({result['prefill'].get(result.get('best_prefill', ''), {}).get('avg_tok_s', '?')} tok/s)")
    print(f"  Best decode:  {result.get('best_decode')} "
          f"({result['decode'].get(result.get('best_decode', ''), {}).get('avg_tok_s', '?')} tok/s)")

    # Save results with model name and GPU count
    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    out_path = os.path.join(os.path.dirname(__file__),
                            f"{safe_name}_auto_{num_gpus}gpu.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
