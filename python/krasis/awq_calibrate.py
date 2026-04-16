#!/usr/bin/env python3
"""AWQ (Activation-Aware Weight Quantization) calibration for Krasis.

Implements the AWQ algorithm from Lin et al. (2023):
  1. Run calibration tokens through BF16 model, capture per-layer per-channel
     activation magnitudes at each attention input.
  2. For each layer, grid search alpha in [0,1] to find optimal per-channel
     scaling s[j] = activation_magnitude[j]^alpha that minimizes
     activation-weighted INT4 quantization error across all input projections.
  3. Save per-layer scales in a template.
  4. At load time: scale weight columns by s[j] before INT4 quantization,
     fold 1/s[j] into the preceding RMSNorm weight (zero runtime overhead).

The key insight: ~1% of weight channels carry most information (because they
multiply with high-magnitude activation channels). AWQ gives these channels
finer quantization grid resolution by pre-scaling, achieving near-INT8 quality
at INT4 size.

Usage:
    python -m krasis.awq_calibrate --config testconfigs/qcn-4-4-a16.conf
    ./dev awq-calibrate qcn [--tokens 256000] [--dataset wikitext-2]
"""

import argparse
import hashlib
import json
import logging
import math
import os
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Add krasis python path
_script_dir = Path(__file__).resolve().parent
_krasis_root = _script_dir.parent.parent  # krasis/python/krasis -> krasis/
sys.path.insert(0, str(_krasis_root / "python"))

from krasis.config import QuantConfig, ModelConfig

# ---------------------------------------------------------------------------
# Model hash (cheap fingerprint for template matching)
# ---------------------------------------------------------------------------

def compute_model_hash(model_path: str) -> str:
    """Compute a fast fingerprint of a model for template matching.

    Hashes config.json, attention tensor names/shapes/dtypes, and first/last
    64 bytes of each attention tensor's data. Returns SHA-256[:16].
    """
    h = hashlib.sha256()

    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            h.update(f.read())

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        for fn in sorted(os.listdir(model_path)):
            if fn.endswith(".safetensors"):
                fp = os.path.join(model_path, fn)
                h.update(fn.encode())
                h.update(str(os.path.getsize(fp)).encode())
        return h.hexdigest()[:16]

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    attn_keys = sorted(k for k in weight_map
                       if any(p in k for p in (
                           "q_proj", "k_proj", "v_proj", "o_proj",
                           "in_proj_qkvz", "in_proj_ba", "out_proj",
                           "kv_a_proj", "kv_b_proj",
                       )) and "weight" in k)

    for key in attn_keys:
        h.update(key.encode())
        h.update(weight_map[key].encode())

    from collections import defaultdict
    shard_tensors = defaultdict(list)
    for key in attn_keys:
        shard_tensors[weight_map[key]].append(key)

    for shard_name, tensor_names in sorted(shard_tensors.items()):
        shard_path = os.path.join(model_path, shard_name)
        if not os.path.exists(shard_path):
            continue
        with open(shard_path, "rb") as f:
            header_size_bytes = f.read(8)
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            header_json = f.read(header_size)
            data_start = 8 + header_size
            try:
                header = json.loads(header_json)
            except json.JSONDecodeError:
                continue
            for tname in tensor_names:
                if tname not in header:
                    continue
                info = header[tname]
                offsets = info.get("data_offsets", [0, 0])
                dtype = info.get("dtype", "")
                shape = info.get("shape", [])
                h.update(f"{tname}:{dtype}:{shape}".encode())
                abs_start = data_start + offsets[0]
                abs_end = data_start + offsets[1]
                tensor_size = abs_end - abs_start
                f.seek(abs_start)
                h.update(f.read(min(64, tensor_size)))
                if tensor_size > 64:
                    f.seek(abs_end - 64)
                    h.update(f.read(64))

    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# INT4 quantization simulation (matches Marlin symmetric group quantization)
# ---------------------------------------------------------------------------

def _quantize_dequantize_int4(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Simulate INT4 symmetric group quantization: quantize then dequantize."""
    assert w.ndim == 2
    N, K = w.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"

    w_flat = w.reshape(N, K // group_size, group_size).float()
    scales = w_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / 7.0
    w_q = (w_flat / scales).round().clamp(-8, 7)
    w_deq = (w_q * scales).reshape(N, K)
    return w_deq


# ---------------------------------------------------------------------------
# Phase 1: Per-layer per-channel activation capture
# ---------------------------------------------------------------------------

class ActivationCapture:
    """Context manager that captures per-layer per-channel activation magnitudes.

    Monkey-patches TransformerLayer.forward and forward_attn to compute
    the post-input-RMSNorm hidden state (what attention projections see)
    and accumulate per-channel mean |X[:,j]| statistics.
    """

    def __init__(self):
        self.stats = {}  # layer_idx -> {sum: Tensor[hidden_size], count: int}
        self._orig_forward = None
        self._orig_forward_attn = None

    def __enter__(self):
        from krasis.layer import TransformerLayer

        self._orig_forward = TransformerLayer.forward
        self._orig_forward_attn = TransformerLayer.forward_attn
        capture = self

        def _rms_norm(x, weight, eps):
            """RMSNorm using torch."""
            variance = x.float().pow(2).mean(-1, keepdim=True)
            x = x.float() * torch.rsqrt(variance + eps)
            return (weight.float() * x).to(x.dtype)

        def _capture_activations(layer_self, hidden, residual):
            """Compute post-norm hidden state and accumulate per-channel stats."""
            with torch.no_grad():
                if residual is None:
                    post_norm = _rms_norm(
                        hidden.clone(), layer_self.input_norm_weight,
                        layer_self.cfg.rms_norm_eps)
                else:
                    h = hidden.clone()
                    r = residual.clone()
                    h = h + r
                    post_norm = _rms_norm(
                        h, layer_self.input_norm_weight,
                        layer_self.cfg.rms_norm_eps)

                # Per-channel mean magnitude: [hidden_size]
                channel_mag = post_norm.float().abs().mean(dim=0)
                idx = layer_self.layer_idx

                if idx not in capture.stats:
                    capture.stats[idx] = {
                        'sum': torch.zeros(channel_mag.shape[0], dtype=torch.float64),
                        'count': 0,
                    }
                capture.stats[idx]['sum'] += channel_mag.cpu().double()
                capture.stats[idx]['count'] += 1

        def patched_forward(layer_self, hidden, residual, positions, *args, **kwargs):
            _capture_activations(layer_self, hidden, residual)
            return capture._orig_forward(layer_self, hidden, residual, positions,
                                         *args, **kwargs)

        def patched_forward_attn(layer_self, hidden, residual, positions, *args, **kwargs):
            _capture_activations(layer_self, hidden, residual)
            return capture._orig_forward_attn(layer_self, hidden, residual, positions,
                                              *args, **kwargs)

        TransformerLayer.forward = patched_forward
        TransformerLayer.forward_attn = patched_forward_attn
        return self

    def __exit__(self, *exc):
        from krasis.layer import TransformerLayer
        TransformerLayer.forward = self._orig_forward
        TransformerLayer.forward_attn = self._orig_forward_attn

    def get_per_layer_activations(self) -> Dict[int, torch.Tensor]:
        """Return per-layer mean per-channel activation magnitudes.

        Returns:
            dict: layer_idx -> Tensor[hidden_size] of mean |activation| per channel
        """
        result = {}
        for idx, s in self.stats.items():
            if s['count'] > 0:
                result[idx] = (s['sum'] / s['count']).float()
        return result


def _collect_activations(
    model,
    tokens: List[int],
    window_size: int = 2048,
    stride: int = 1024,
    max_windows: int = 32,
) -> Dict[int, torch.Tensor]:
    """Run calibration tokens through model, capture per-layer activations.

    Args:
        model: Loaded KrasisModel with BF16 attention
        tokens: Calibration token IDs
        window_size: Tokens per calibration window
        stride: Stride between windows
        max_windows: Max windows to process

    Returns:
        dict: layer_idx -> Tensor[hidden_size] of mean per-channel |activation|
    """
    from krasis.kv_cache import SequenceKVState

    device = torch.device(model.ranks[0].device)
    total_tokens = len(tokens)
    starts = list(range(0, total_tokens - 1, stride))[:max_windows]
    total_windows = len(starts)

    # GPU prefill uses layer.forward() / layer.forward_attn() which our
    # monkey-patches intercept, so we keep it enabled for speed.
    print(f"\n  Phase 1: Capturing per-layer per-channel activations "
          f"({total_windows} windows, {window_size} tokens each)...")

    capture = ActivationCapture()
    with capture:
        for win_idx, begin in enumerate(starts):
            end = min(begin + window_size, total_tokens)
            win_len = end - begin
            if win_len < 2:
                break

            seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]

            # Reset LA states
            if model.cfg.is_hybrid:
                for layer in model.layers:
                    if layer.layer_type == "linear_attention":
                        layer.attention.reset_state()

            try:
                token_tensor = torch.tensor(tokens[begin:end], dtype=torch.long, device=device)
                positions = torch.arange(win_len, dtype=torch.int32, device=device)

                with torch.inference_mode():
                    model.forward(token_tensor, positions, seq_states)

            finally:
                for s in seq_states:
                    s.free()

            print(f"\r  Window {win_idx + 1}/{total_windows}", end="", flush=True)

    print()

    activations = capture.get_per_layer_activations()
    print(f"  Captured activations for {len(activations)} layers")
    return activations


# ---------------------------------------------------------------------------
# Phase 2: Per-layer alpha grid search
# ---------------------------------------------------------------------------

def _get_weight_tensor(model, layer_idx: int, tensor_name: str) -> Optional[torch.Tensor]:
    """Get the original BF16 weight tensor for an attention projection."""
    cpu_weights = getattr(model, '_stream_attn_cpu', {})
    layer = model.layers[layer_idx]
    attn = layer.attention
    gqa_weights = getattr(layer, "gqa_weights", None)

    w = None
    if layer_idx in cpu_weights:
        w = cpu_weights[layer_idx].get(tensor_name)
    if w is None:
        if gqa_weights is not None and tensor_name in gqa_weights:
            w = gqa_weights.get(tensor_name)
        elif attn is not None:
            w = getattr(attn, tensor_name, None)

    if w is not None and isinstance(w, torch.Tensor):
        return w.cpu().to(torch.bfloat16) if w.is_cuda else w.to(torch.bfloat16)
    return None


def _is_marlin_compatible(w: torch.Tensor, group_size: int = 128) -> bool:
    """Check if a weight tensor is compatible with Marlin quantization."""
    if w.ndim != 2:
        return False
    N, K = w.shape
    return N % 64 == 0 and K % 16 == 0 and K % group_size == 0


# Which projections take the post-input-norm hidden state as input
# (these can have AWQ scaling folded into the preceding norm)
# Keys must match layer_type values from weight_loader: "linear_attention",
# "full_attention", "sliding_attention"
INPUT_PROJECTIONS = {
    "linear_attention": ["in_proj_qkvz", "in_proj_ba"],
    "full_attention": ["q_proj", "k_proj", "v_proj"],
    "sliding_attention": ["q_proj", "k_proj", "v_proj"],
    "mla": ["q_a_proj", "kv_a_proj"],
    "mla_direct_q": ["q_proj", "kv_a_proj"],
}

# Which projections take attention output as input (no norm folding possible)
OUTPUT_PROJECTIONS = {
    "linear_attention": ["out_proj"],
    "full_attention": ["o_proj"],
    "sliding_attention": ["o_proj"],
    "mla": ["o_proj"],
    "mla_direct_q": ["o_proj"],
}


def _search_alpha_for_layer(
    s_x: torch.Tensor,          # [hidden_size] per-channel activation magnitudes
    weights: List[torch.Tensor], # list of [N_i, K] BF16 weight tensors
    group_size: int = 128,
    alpha_steps: int = 21,
) -> Tuple[float, float, float]:
    """Grid search for optimal alpha that minimizes activation-weighted INT4 error.

    For each candidate alpha:
      s = s_x^alpha (per-channel scale)
      For each weight W:
        W_scaled = W * s (scale columns to protect high-activation channels)
        W_deq = quantize_dequantize(W_scaled) / s (quantize, then unscale)
        error += mean(((W - W_deq) * s_x)^2)  (activation-weighted MSE)

    Returns:
        (best_alpha, best_error, baseline_error)
    """
    s_x_f = s_x.float().clamp(min=1e-6)
    alphas = torch.linspace(0.0, 1.0, alpha_steps)

    best_alpha = 0.0
    best_error = float('inf')
    baseline_error = None

    for alpha in alphas:
        alpha_val = alpha.item()
        s = s_x_f.pow(alpha_val).clamp(min=1e-5)

        total_error = 0.0
        for w in weights:
            w_f = w.float()

            # Scale weight columns: W_scaled[:,j] = W[:,j] * s[j]
            w_scaled = w_f * s.unsqueeze(0)

            # Quantize and dequantize at INT4
            w_deq_scaled = _quantize_dequantize_int4(
                w_scaled.to(torch.bfloat16), group_size).float()

            # Unscale: W_deq[:,j] = W_deq_scaled[:,j] / s[j]
            w_deq = w_deq_scaled / s.unsqueeze(0)

            # Activation-weighted MSE: weight each column's error by its
            # activation magnitude (channels with higher activation matter more)
            err = ((w_f - w_deq) * s_x_f.unsqueeze(0)).pow(2).mean().item()
            total_error += err

        if alpha_val == 0.0:
            baseline_error = total_error

        if total_error < best_error:
            best_error = total_error
            best_alpha = alpha_val

    return best_alpha, best_error, baseline_error or best_error


def _search_all_layers(
    model,
    per_layer_activations: Dict[int, torch.Tensor],
    group_size: int = 128,
    alpha_steps: int = 21,
) -> Dict[int, Dict]:
    """Run alpha search for all layers.

    Returns:
        dict: layer_idx -> {
            'alpha': float,
            'scales': list[float],   # s[j] = s_x[j]^alpha, len=hidden_size
            'best_mse': float,
            'baseline_mse': float,
            'improvement': float,     # (baseline - best) / baseline
            'input_projs': list[str], # names of scaled input projections
            'output_projs': list[str], # names of unscaled output projections
            'bf16_projs': list[str],  # Marlin-incompatible projections
        }
    """
    results = {}
    num_layers = len(model.layers)

    print(f"\n  Phase 2: Searching optimal alpha per layer ({alpha_steps} candidates)...")

    for layer_idx, layer in enumerate(model.layers):
        if layer_idx not in per_layer_activations:
            continue

        s_x = per_layer_activations[layer_idx]
        lt = layer.layer_type
        # MLA layers report as "full_attention" but have different projections
        if lt != "linear_attention" and hasattr(layer.attention, 'kv_a_proj'):
            lt = "mla" if hasattr(layer.attention, 'q_a_proj') else "mla_direct_q"
        input_proj_names = INPUT_PROJECTIONS.get(lt, [])
        output_proj_names = OUTPUT_PROJECTIONS.get(lt, [])

        # Collect Marlin-compatible input projection weights
        input_weights = []
        scaled_proj_names = []
        bf16_proj_names = []

        for name in input_proj_names:
            w = _get_weight_tensor(model, layer_idx, name)
            if w is None:
                bf16_proj_names.append(name)
                continue
            if not _is_marlin_compatible(w, group_size):
                bf16_proj_names.append(name)
                continue
            input_weights.append(w)
            scaled_proj_names.append(name)

        # Check output projections for Marlin compatibility
        valid_output_projs = []
        for name in output_proj_names:
            w = _get_weight_tensor(model, layer_idx, name)
            if w is not None and _is_marlin_compatible(w, group_size):
                valid_output_projs.append(name)
            else:
                bf16_proj_names.append(name)

        if not input_weights:
            # No scalable projections — skip layer
            results[layer_idx] = {
                'alpha': 0.0,
                'scales': [1.0] * s_x.shape[0],
                'best_mse': 0.0,
                'baseline_mse': 0.0,
                'improvement': 0.0,
                'input_projs': [],
                'output_projs': valid_output_projs,
                'bf16_projs': bf16_proj_names,
            }
            continue

        # Grid search alpha
        best_alpha, best_mse, baseline_mse = _search_alpha_for_layer(
            s_x, input_weights, group_size, alpha_steps)

        # Compute final scales
        s_x_f = s_x.float().clamp(min=1e-6)
        final_scales = s_x_f.pow(best_alpha).clamp(min=1e-5)

        improvement = (baseline_mse - best_mse) / max(baseline_mse, 1e-10)

        results[layer_idx] = {
            'alpha': best_alpha,
            'scales': final_scales.tolist(),
            'best_mse': best_mse,
            'baseline_mse': baseline_mse,
            'improvement': improvement,
            'input_projs': scaled_proj_names,
            'output_projs': valid_output_projs,
            'bf16_projs': bf16_proj_names,
        }

        # Clean up weight tensors
        del input_weights

        print(f"\r  [{layer_idx + 1}/{num_layers}] layer {layer_idx} ({lt}): "
              f"alpha={best_alpha:.2f}  "
              f"error reduction={improvement * 100:.1f}%  "
              f"scaled={len(scaled_proj_names)} output={len(valid_output_projs)} "
              f"bf16={len(bf16_proj_names)}",
              end="", flush=True)

    print()
    return results


# ---------------------------------------------------------------------------
# Phase 3: Template building and I/O
# ---------------------------------------------------------------------------

def build_template_v2(
    model_path: str,
    layer_results: Dict[int, Dict],
    num_tokens: int,
    dataset: str,
    group_size: int = 128,
    alpha_steps: int = 21,
) -> Dict[str, Any]:
    """Build a v2 AWQ template with per-layer channel scales."""
    model_hash = compute_model_hash(model_path)

    # Per-layer data (scales stored as lists of rounded floats)
    layers = {}
    total_scaled = 0
    total_output = 0
    total_bf16 = 0

    for layer_idx, lr in sorted(layer_results.items()):
        layers[str(layer_idx)] = {
            "alpha": round(lr['alpha'], 4),
            "scales": [round(s, 6) for s in lr['scales']],
            "improvement": round(lr['improvement'], 4),
            "input_projs": lr['input_projs'],
            "output_projs": lr['output_projs'],
            "bf16_projs": lr['bf16_projs'],
        }
        total_scaled += len(lr['input_projs'])
        total_output += len(lr['output_projs'])
        total_bf16 += len(lr['bf16_projs'])

    # Compute average improvement
    improvements = [lr['improvement'] for lr in layer_results.values()
                    if lr['input_projs']]
    avg_improvement = sum(improvements) / max(len(improvements), 1)

    template = {
        "version": 2,
        "model_hash": model_hash,
        "model_path": os.path.basename(model_path),
        "calibration": {
            "tokens": num_tokens,
            "dataset": dataset,
            "group_size": group_size,
            "alpha_steps": alpha_steps,
            "date": datetime.now().isoformat(),
        },
        "summary": {
            "total_layers": len(layer_results),
            "awq_scaled_tensors": total_scaled,
            "plain_int4_tensors": total_output,
            "bf16_tensors": total_bf16,
            "avg_error_reduction": round(avg_improvement, 4),
        },
        "layers": layers,
    }

    return template


def save_template(template: Dict, output_dir: str) -> str:
    """Save template to the standard directory structure."""
    model_hash = template["model_hash"]
    out_dir = os.path.join(output_dir, model_hash)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "template.json")

    with open(out_path, "w") as f:
        json.dump(template, f, indent=2)

    return out_path


_TEMPLATE_BASE_URL = (
    "https://raw.githubusercontent.com/brontoguana/krasis/main"
    "/templates/attention"
)


def _download_template(model_hash: str) -> Optional[Dict]:
    """Try to download an AWQ template from GitHub.

    Downloads from the main branch raw content URL and caches locally
    in ~/.krasis/templates/<model_hash>/template.json.

    Returns the parsed template dict, or None if download fails.
    """
    import urllib.request
    import urllib.error

    url = f"{_TEMPLATE_BASE_URL}/{model_hash}/template.json"
    cache_dir = os.path.expanduser(f"~/.krasis/templates/{model_hash}")
    cache_path = os.path.join(cache_dir, "template.json")

    logger.info("Downloading AWQ template for model hash %s ...", model_hash)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "krasis"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        logger.warning("Failed to download AWQ template from %s: %s", url, e)
        return None

    try:
        template = json.loads(data)
    except json.JSONDecodeError:
        logger.warning("Downloaded AWQ template is not valid JSON")
        return None

    if template.get("model_hash") != model_hash:
        logger.warning("Downloaded template hash mismatch (expected %s, got %s)",
                       model_hash, template.get("model_hash"))
        return None

    # Cache locally
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(template, f, indent=2)
        logger.info("AWQ template cached at %s", cache_path)
    except OSError as e:
        logger.warning("Failed to cache AWQ template: %s", e)

    return template


def load_template(template_dir: str, model_path: str) -> Optional[Dict]:
    """Load a template for a model, if one exists.

    Searches:
    0. Explicit override from KRASIS_AWQ_TEMPLATE_PATH
    1. Bundled templates in template_dir/<model_hash>/template.json
    2. User cache in ~/.krasis/templates/<model_hash>/template.json
    3. Downloads from GitHub if not found locally
    """
    model_hash = compute_model_hash(model_path)
    override_path = os.environ.get("KRASIS_AWQ_TEMPLATE_PATH", "").strip()

    if override_path:
        override_path = os.path.expanduser(override_path)
        if not os.path.exists(override_path):
            raise RuntimeError(
                f"KRASIS_AWQ_TEMPLATE_PATH points to missing file: {override_path}"
            )
        with open(override_path) as f:
            template = json.load(f)
        if template.get("model_hash") != model_hash:
            raise RuntimeError(
                "KRASIS_AWQ_TEMPLATE_PATH hash mismatch "
                f"(expected {model_hash}, got {template.get('model_hash')})"
            )
        logger.info("Using AWQ template override: %s", override_path)
        return template

    search_paths = [
        os.path.join(template_dir, model_hash, "template.json"),
        os.path.expanduser(f"~/.krasis/templates/{model_hash}/template.json"),
    ]

    for path in search_paths:
        if os.path.exists(path):
            with open(path) as f:
                template = json.load(f)
            if template.get("model_hash") == model_hash:
                return template
            else:
                logger.warning("Template hash mismatch at %s (expected %s, got %s)",
                               path, model_hash, template.get("model_hash"))

    # Not found locally — try downloading from GitHub
    return _download_template(model_hash)


def get_layer_scales(template: Dict, layer_idx: int) -> Optional[torch.Tensor]:
    """Get per-channel AWQ scales for a layer from a v2 template.

    Returns:
        Tensor[hidden_size] of per-channel scales, or None if layer not in template
        or template is v1.
    """
    if template.get("version", 1) < 2:
        return None

    layers = template.get("layers", {})
    layer_data = layers.get(str(layer_idx))
    if layer_data is None:
        return None

    scales = layer_data.get("scales")
    if scales is None:
        return None

    return torch.tensor(scales, dtype=torch.float32)


def get_tensor_decision(template: Dict, layer_idx: int, layer_type: str,
                        tensor_name: str) -> str:
    """Get the quantization decision for a specific tensor.

    For v2 templates: input projections get "int4" (AWQ-scaled at load time),
    output projections get "int4" (plain), incompatible get "bf16".
    Unknown tensors are a template contract error and must fail loudly.

    For v1 templates: returns per-tensor decision from old format.
    """
    if template.get("version", 1) >= 2:
        layers = template.get("layers", {})
        layer_data = layers.get(str(layer_idx))
        if layer_data is None:
            return "bf16"

        if tensor_name in layer_data.get("input_projs", []):
            return "int4"
        elif tensor_name in layer_data.get("output_projs", []):
            return "int4"
        elif tensor_name in layer_data.get("bf16_projs", []):
            return "bf16"
        raise RuntimeError(
            "AWQ v2 template missing tensor decision for "
            f"layer {layer_idx} ({layer_type}) tensor {tensor_name}"
        )
    else:
        # v1 fallback
        key = f"layers.{layer_idx}.{layer_type}.{tensor_name}"
        decisions = template.get("decisions", {})
        entry = decisions.get(key, {})
        return entry.get("decision", "bf16")


def is_awq_scaled_tensor(template: Dict, layer_idx: int, tensor_name: str) -> bool:
    """Check if a tensor should have AWQ per-channel scaling applied."""
    if template.get("version", 1) < 2:
        return False

    layers = template.get("layers", {})
    layer_data = layers.get(str(layer_idx))
    if layer_data is None:
        return False

    return tensor_name in layer_data.get("input_projs", [])


# ---------------------------------------------------------------------------
# Main calibration
# ---------------------------------------------------------------------------

def calibrate(
    model,
    tokens: List[int],
    group_size: int = 128,
    window_size: int = 2048,
    stride: int = 1024,
    max_windows: int = 32,
    alpha_steps: int = 21,
) -> Dict[int, Dict]:
    """Run full AWQ calibration.

    Returns:
        dict: layer_idx -> per-layer results (alpha, scales, etc.)
    """
    # Phase 1: Collect per-layer per-channel activation magnitudes
    per_layer_activations = _collect_activations(
        model, tokens, window_size, stride, max_windows)

    # Phase 2: Search optimal alpha per layer
    layer_results = _search_all_layers(
        model, per_layer_activations, group_size, alpha_steps)

    return layer_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_config_file(config_path: str) -> Dict[str, str]:
    """Parse a KEY=VALUE config file, return as dict."""
    result = {}
    _CFG_KEY_MAP = {
        "MODEL_PATH": "model_path",
        "CFG_LAYER_GROUP_SIZE": "layer_group_size",
        "CFG_KV_DTYPE": "kv_dtype",
        "CFG_GPU_EXPERT_BITS": "gpu_expert_bits",
        "CFG_CPU_EXPERT_BITS": "cpu_expert_bits",
        "CFG_ATTENTION_QUANT": "attention_quant",
        "CFG_SHARED_EXPERT_QUANT": "shared_expert_quant",
        "CFG_DENSE_MLP_QUANT": "dense_mlp_quant",
        "CFG_LM_HEAD_QUANT": "lm_head_quant",
        "CFG_KV_CACHE_MB": "kv_cache_mb",
    }
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
                if dest:
                    result[dest] = val
            else:
                result[key.replace("-", "_").lower()] = val
    if "model_path" in result:
        result["model_path"] = os.path.expanduser(result["model_path"])
    return result


def main():
    parser = argparse.ArgumentParser(
        description="AWQ calibration for Krasis attention quantization")
    parser.add_argument("--config", required=True,
                        help="Path to .conf config file")
    parser.add_argument("--tokens", type=int, default=256000,
                        help="Number of calibration tokens (default: 256000)")
    parser.add_argument("--dataset", default="wikitext-2",
                        choices=["wikitext-2", "wikitext-103", "c4"],
                        help="Calibration dataset (default: wikitext-2)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Quantization group size (default: 128)")
    parser.add_argument("--output-dir", default=None,
                        help="Template output directory (default: krasis/templates/attention)")
    parser.add_argument("--window-size", type=int, default=2048,
                        help="Tokens per calibration window")
    parser.add_argument("--stride", type=int, default=1024,
                        help="Stride between windows")
    parser.add_argument("--max-windows", type=int, default=32,
                        help="Max calibration windows")
    parser.add_argument("--alpha-steps", type=int, default=21,
                        help="Number of alpha candidates to search (default: 21)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = _parse_config_file(args.config)
    model_path = cfg.get("model_path")
    if not model_path:
        print("Error: MODEL_PATH not found in config file", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  AWQ Attention Calibration (v2 — proper per-channel scaling)")
    print(f"{'='*60}")
    print(f"  Model:       {model_path}")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Tokens:      {args.tokens:,}")
    print(f"  Group size:  {args.group_size}")
    print(f"  Alpha steps: {args.alpha_steps}")
    print(f"  Windows:     {args.max_windows} x {args.window_size} tokens")
    print(f"{'='*60}")

    # Load model with BF16 attention — calibration always uses full precision
    print("\n  Loading model with BF16 attention...")
    from krasis.model import KrasisModel

    kv_dtype_str = cfg.get("kv_dtype", "fp8_e4m3")
    kv_dtype_map = {
        "fp8_e4m3": torch.float8_e4m3fn,
        "bf16": torch.bfloat16,
    }
    kv_dtype = kv_dtype_map.get(kv_dtype_str, torch.float8_e4m3fn)

    model = KrasisModel(
        model_path=model_path,
        quant_cfg=QuantConfig(
            attention="bf16",  # Always BF16 for calibration
            lm_head=cfg.get("lm_head_quant", "int8"),
            shared_expert=cfg.get("shared_expert_quant", "int8"),
            dense_mlp=cfg.get("dense_mlp_quant", "int8"),
            gpu_expert_bits=int(cfg.get("gpu_expert_bits", 4)),
            cpu_expert_bits=int(cfg.get("cpu_expert_bits", 4)),
        ),
        kv_dtype=kv_dtype,
        layer_group_size=int(cfg.get("layer_group_size", 2)),
        kv_cache_mb=int(cfg.get("kv_cache_mb", 1000)),
        stream_attention=True,  # Always stream — calibration is one-time, and large models may not fit BF16 attention in VRAM
    )
    model.load(gpu_only=True)

    # Load calibration dataset
    print(f"\n  Loading calibration dataset: {args.dataset}...")
    perplexity_dir = Path(__file__).resolve().parent.parent.parent / "perplexity"
    sys.path.insert(0, str(perplexity_dir))
    from measure_ppl import load_dataset_text

    text = load_dataset_text(args.dataset)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    all_tokens = tokenizer.encode(text)
    cal_tokens = all_tokens[:args.tokens]
    print(f"  Tokenized: {len(all_tokens):,} total, using {len(cal_tokens):,}")

    model_hash = compute_model_hash(model_path)
    print(f"  Model hash: {model_hash}")

    # Run calibration
    print(f"\n  Starting calibration...")
    t_start = time.perf_counter()

    layer_results = calibrate(
        model=model,
        tokens=cal_tokens,
        group_size=args.group_size,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
        alpha_steps=args.alpha_steps,
    )

    elapsed = time.perf_counter() - t_start

    # Build and save template
    template = build_template_v2(
        model_path=model_path,
        layer_results=layer_results,
        num_tokens=len(cal_tokens),
        dataset=args.dataset,
        group_size=args.group_size,
        alpha_steps=args.alpha_steps,
    )

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(Path(__file__).resolve().parent.parent.parent
                         / "templates" / "attention")

    out_path = save_template(template, output_dir)

    # Summary
    summary = template["summary"]
    print(f"\n{'='*60}")
    print(f"  Calibration Complete ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Total layers:         {summary['total_layers']}")
    print(f"  AWQ-scaled tensors:   {summary['awq_scaled_tensors']} (INT4 with per-channel scaling)")
    print(f"  Plain INT4 tensors:   {summary['plain_int4_tensors']} (output projections)")
    print(f"  BF16 tensors:         {summary['bf16_tensors']} (Marlin-incompatible)")
    print(f"  Avg error reduction:  {summary['avg_error_reduction'] * 100:.1f}%")
    print(f"  Template:             {out_path}")
    print(f"  Model hash:           {model_hash}")

    # Per-layer breakdown
    print(f"\n  Per-layer breakdown:")
    print(f"  {'Layer':>6} {'Type':>15} {'Alpha':>6} {'Error Reduction':>16} "
          f"{'Scaled':>7} {'Output':>7} {'BF16':>5}")
    print(f"  {'-'*70}")
    for layer_idx in sorted(layer_results.keys()):
        lr = layer_results[layer_idx]
        layer = model.layers[layer_idx]
        lt = layer.layer_type
        if lt != "linear_attention" and hasattr(layer.attention, 'kv_a_proj'):
            lt = "mla"
        print(f"  {layer_idx:>6} {lt:>15} {lr['alpha']:>6.2f} "
              f"{lr['improvement'] * 100:>15.1f}% "
              f"{len(lr['input_projs']):>7} {len(lr['output_projs']):>7} "
              f"{len(lr['bf16_projs']):>5}")

    print(f"\n  To use: set CFG_ATTENTION_QUANT=awq in config. "
          f"Template auto-loaded by model hash.")
    print()


if __name__ == "__main__":
    main()
