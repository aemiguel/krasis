#!/usr/bin/env python3
"""Test auto-optimiser on DeepSeek-V2-Lite.

Usage:
    # 1 GPU:
    CUDA_VISIBLE_DEVICES=0 python tests/test_auto_optimise_v2lite.py --num-gpus 1
    # 2 GPUs:
    CUDA_VISIBLE_DEVICES=0,1 python tests/test_auto_optimise_v2lite.py --num-gpus 2
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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "DeepSeek-V2-Lite")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Force re-run even if cached")
    args = parser.parse_args()

    num_gpus = args.num_gpus
    num_layers = 27

    # Compute balanced PP partition
    base = num_layers // num_gpus
    remainder = num_layers % num_gpus
    pp_partition = [base + (1 if i < remainder else 0) for i in range(num_gpus)]

    devices = [f"cuda:{i}" for i in range(num_gpus)]

    print(f"=== V2-Lite Auto-Optimise: {num_gpus} GPU(s), PP={pp_partition} ===")
    print(f"Devices: {devices}")

    t0 = time.perf_counter()
    model = KrasisModel(
        model_path=MODEL_PATH,
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

    # Delete any existing auto_optimise.json to force a fresh run
    # (we want to benchmark from scratch for each GPU config)
    cache_file = os.path.join(MODEL_PATH, ".krasis_cache", "auto_optimise.json")
    if args.force and os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Removed cached results: {cache_file}")

    t_opt_start = time.perf_counter()
    result = auto_optimise_or_load(model, force=args.force)
    t_opt = time.perf_counter() - t_opt_start

    print(f"\n{'='*70}")
    print(f"COMPLETE: {num_gpus} GPU(s)")
    print(f"  Optimise time: {t_opt:.1f}s")
    print(f"  Best prefill: {result.get('best_prefill')} "
          f"({result['prefill'].get(result.get('best_prefill', ''), {}).get('avg_tok_s', '?')} tok/s)")
    print(f"  Best decode:  {result.get('best_decode')} "
          f"({result['decode'].get(result.get('best_decode', ''), {}).get('avg_tok_s', '?')} tok/s)")

    # Save results with GPU count in filename
    out_path = os.path.join(os.path.dirname(__file__),
                            f"v2lite_auto_{num_gpus}gpu.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
