#!/usr/bin/env python3
"""Profile decode WITHOUT pinning to isolate pinning overhead."""
import logging, sys, time, os
os.environ["KRASIS_DECODE_TIMING"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger("profile")

import torch
from krasis.model import KrasisModel

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"

model = KrasisModel(
    MODEL,
    pp_partition=[24, 24],
    devices=["cuda:0", "cuda:1"],
    kv_dtype=torch.float8_e4m3fn,
    krasis_threads=48,
    gpu_prefill=True,
    gpu_prefill_threshold=1,
    expert_divisor=-1,
)
model.load()

# NO pinning configured — all experts come from CPU DMA

# Warmup
warmup = model.tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}])
model.generate(warmup, max_new_tokens=4, temperature=0.0, top_k=1)

# Test with 10 tokens
messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = model.tokenizer.apply_chat_template(messages)
print(f"Prompt: {len(prompt)} tokens")

t0 = time.perf_counter()
out = model.generate(prompt, max_new_tokens=10, temperature=0.0, top_k=1)
t1 = time.perf_counter()
text = model.tokenizer.decode(out)
print(f"Output: {repr(text)}")
print(f"Wall-clock: {(t1-t0)*1000:.1f}ms for {len(out)} tokens")
