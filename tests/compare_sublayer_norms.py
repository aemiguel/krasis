#!/usr/bin/env python3
"""Compare Rust prefill diagnostic norms against BF16 sublayer reference.

Usage:
    KRASIS_PREFILL_DIAG=1 ./dev run qwen35 ... 2>diag.log
    python3 tests/compare_sublayer_norms.py diag.log [model]

Or pipe directly:
    KRASIS_PREFILL_DIAG=1 ./dev run qwen35 ... 2>&1 | python3 tests/compare_sublayer_norms.py - qwen35

Models: qwen35 (default), qcn, q122
"""
import json
import re
import sys
import os

REFERENCE_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'krasis-internal',
    'reference-outputs', 'output'
)

MODEL_MAP = {
    'qwen35': 'Qwen3.5-35B-A3B-sublayers',
    'q35': 'Qwen3.5-35B-A3B-sublayers',
    'qcn': 'Qwen3-Coder-Next-sublayers',
    'q122': 'Qwen3.5-122B-A10B-sublayers',
}

def load_reference(model_key):
    dirname = MODEL_MAP.get(model_key, model_key)
    path = os.path.join(REFERENCE_DIR, dirname, 'greedy_reference.json')
    with open(path) as f:
        data = json.load(f)
    # Get first conversation, first turn
    turn = data['conversations'][0]['turns'][0]
    return turn['layer_norms']

def parse_diag_line(line):
    """Parse a [DIAG] line into (label, {pos: norm})."""
    m = re.match(r'\[DIAG\]\s+(\S+)\s+(.*)', line.strip())
    if not m:
        return None, None
    label = m.group(1)
    norms = {}
    for part in m.group(2).split():
        pm = re.match(r'(\d+):([0-9.eE+-]+)', part)
        if pm:
            norms[int(pm.group(1))] = float(pm.group(2))
    return label, norms

def compare(ref_data, diag_lines):
    """Compare diagnostic norms against reference and print table."""
    # Build reference lookup: label -> {pos: l2_norm}
    ref = {}

    # Embedding
    emb = {}
    for entry in ref_data.get('embedding', []):
        emb[entry['position']] = entry['l2_norm']
    ref['embedding'] = emb

    # Per-layer
    for layer_data in ref_data.get('layers', []):
        layer_idx = layer_data['layer']
        mixer_type = layer_data.get('mixer_type', 'unknown')
        lt = 'la' if 'linear' in mixer_type else 'gqa'

        mixer = {}
        for entry in layer_data.get('mixer', []):
            mixer[entry['position']] = entry['l2_norm']
        ref[f'layer{layer_idx:02d}_{lt}_mixer'] = mixer

        mlp = {}
        for entry in layer_data.get('mlp', []):
            mlp[entry['position']] = entry['l2_norm']
        ref[f'layer{layer_idx:02d}_mlp'] = mlp

    # Parse diag lines
    diag = {}
    for line in diag_lines:
        if '[DIAG]' not in line:
            continue
        label, norms = parse_diag_line(line)
        if label and norms and label != '===':
            diag[label] = norms

    if not diag:
        print("No [DIAG] lines found in input!")
        return False

    # Compare
    all_ok = True
    for label, actual_norms in sorted(diag.items()):
        ref_norms = ref.get(label)
        if ref_norms is None:
            print(f"\n{label}: no reference data")
            continue

        print(f"\n{label}:")
        print(f"  {'pos':>4s}  {'ref':>10s}  {'actual':>10s}  {'err%':>7s}  {'status':>6s}")
        print(f"  {'----':>4s}  {'----------':>10s}  {'----------':>10s}  {'-------':>7s}  {'------':>6s}")

        for pos in sorted(actual_norms.keys()):
            actual = actual_norms[pos]
            if pos in ref_norms:
                expected = ref_norms[pos]
                if expected > 0:
                    err_pct = abs(actual - expected) / expected * 100
                else:
                    err_pct = 0 if actual == 0 else 999

                # Thresholds: <5% OK, 5-20% WARN, >20% FAIL
                if err_pct < 5:
                    status = "OK"
                elif err_pct < 20:
                    status = "WARN"
                    all_ok = False
                else:
                    status = "FAIL"
                    all_ok = False

                print(f"  {pos:4d}  {expected:10.6f}  {actual:10.6f}  {err_pct:6.1f}%  {status:>6s}")
            else:
                print(f"  {pos:4d}  {'n/a':>10s}  {actual:10.6f}")

    return all_ok

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    log_file = sys.argv[1]
    model_key = sys.argv[2] if len(sys.argv) > 2 else 'qwen35'

    # Load reference
    ref_data = load_reference(model_key)

    # Read diag lines
    if log_file == '-':
        lines = sys.stdin.readlines()
    else:
        with open(log_file) as f:
            lines = f.readlines()

    ok = compare(ref_data, lines)
    print()
    if ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — see WARN/FAIL above")
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
