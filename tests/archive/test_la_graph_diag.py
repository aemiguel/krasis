"""Minimal test: does LA graph capture succeed with streaming attention?

Loads Qwen3-Coder-Next with 2 GPUs + stream-attention, generates 5 tokens,
checks stderr for DIAG messages to see if capture happens.

No HCS — pure CPU experts for simplicity.
"""
import os
import sys
import logging

# Ensure no timing interference
os.environ.pop("KRASIS_DECODE_TIMING", None)
os.environ.pop("KRASIS_PREFILL_TIMING", None)
os.environ.pop("KRASIS_OVERLAP_TIMING", None)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import torch
from krasis.model import KrasisModel
from krasis.config import QuantConfig

model_path = os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next")

qcfg = QuantConfig(
    attention="awq", shared_expert="int8", dense_mlp="int8",
    lm_head="int8", gpu_expert_bits=4, cpu_expert_bits=4,
)

print("Loading model with stream_attention=True, 2 GPUs, no HCS...", file=sys.stderr, flush=True)
model = KrasisModel(
    model_path=model_path,
    pp_partition=[48],  # all on GPU0
    num_gpus=2,
    kv_dtype=torch.float8_e4m3fn,
    krasis_threads=48,
    quant_cfg=qcfg,
    layer_group_size=2,
    stream_attention=True,
    gpu_prefill_threshold=1,  # GPU decode (triggers streaming weight loading)
    kv_cache_mb=1000,
)
model.load()

print("\n=== Generating 5 tokens (watch for [DIAG] messages) ===", file=sys.stderr, flush=True)
# Use a longer prompt to ensure layer-grouped prefill path
prompt = ("Explain the theory of general relativity in detail. " * 50)
tokens = model.tokenizer.apply_chat_template([{"role": "user", "content": prompt}])
print(f"Prompt: {len(tokens)} tokens", file=sys.stderr, flush=True)

generated = model.generate(tokens, max_new_tokens=3, temperature=0.6)
text = model.tokenizer.decode(generated)
print(f"\nGenerated: {text}", file=sys.stderr, flush=True)
print("=== Done ===", file=sys.stderr, flush=True)
