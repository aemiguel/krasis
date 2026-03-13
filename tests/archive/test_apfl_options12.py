#!/usr/bin/env python3
"""Test APFL Options 1+2: separate prefetch stream + early speculative routing.

Compares:
  - Baseline (no APFL)
  - Options 1+2 with prefetch=5
  - Options 1+2 with prefetch=10

All computation runs in Rust/CUDA. Python is only the test driver.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import krasis

QCN_DIR = os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next")
NUM_TOKENS = 15

configs = [
    ("Baseline (no APFL)", 0, 0, 4),
    ("Options 1+2: prefetch=5, 16 slots", 5, 10, 16),
    ("Options 1+2: prefetch=10, 20 slots", 10, 10, 20),
    ("Options 1+2: prefetch=10, 32 slots", 10, 10, 32),
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
