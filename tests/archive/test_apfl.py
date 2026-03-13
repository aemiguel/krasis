#!/usr/bin/env python3
"""Test APFL (Adaptive Prefetch Layer) for GPU MoE decode.

All computation runs in Rust/CUDA. Python is only the test driver.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import krasis

QCN_DIR = os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next")

def main():
    store = krasis.GpuDecodeStore(0)

    # Single test: APFL with prefetch=5, 16 slots, 10 tokens
    print("=== APFL prefetch=5, 16 slots ===")
    result = store.test_apfl_e2e_py(
        QCN_DIR,
        num_tokens=10,
        initial_prefetch=5,
        max_prefetch=10,
        num_slots=16,
    )
    print(result)
    print("\nDone.")

if __name__ == "__main__":
    main()
