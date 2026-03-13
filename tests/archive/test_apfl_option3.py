#!/usr/bin/env python3
"""Test APFL Option 3: prefetch=10 (match topk) vs baseline.

All computation runs in Rust/CUDA. Python is only the test driver.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import krasis

QCN_DIR = os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next")
NUM_TOKENS = 15

configs = [
    ("Baseline (no APFL)", 0, 0, 4),
    ("APFL prefetch=5", 5, 10, 16),
    ("APFL prefetch=10 (Option 3)", 10, 10, 20),
]

for name, init_pf, max_pf, slots in configs:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    store = krasis.GpuDecodeStore(0)
    result = store.test_apfl_e2e_py(
        QCN_DIR,
        num_tokens=NUM_TOKENS,
        initial_prefetch=init_pf,
        max_prefetch=max_pf,
        num_slots=slots,
    )
    print(result)
    del store

print("\nDone.")
