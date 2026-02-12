#!/usr/bin/env python3
"""Test native GGUF kernel path (no dequant→requant)."""
import logging, time, torch
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

MODEL_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"
GGUF_PATH = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite-GGUF/DeepSeek-V2-Lite.Q4_K_M.gguf"

from krasis.model import KrasisModel
from krasis.config import QuantConfig
from krasis.kv_cache import SequenceKVState
from krasis.sampler import sample

qcfg = QuantConfig(
    attention="bf16", shared_expert="bf16", dense_mlp="bf16", lm_head="bf16",
    gpu_expert_bits=4, cpu_expert_bits=4,
)

model = KrasisModel(
    model_path=MODEL_PATH, num_gpus=1,
    gpu_prefill=True, gpu_prefill_threshold=10,
    krasis_threads=16, quant_cfg=qcfg, kv_dtype=torch.bfloat16,
    expert_divisor=1,
    gguf_path=GGUF_PATH,
)
t0 = time.perf_counter()
model.load()
t_load = time.perf_counter() - t0
print(f"GGUF load: {t_load:.1f}s")

device = torch.device(model.ranks[0].device)

def gen(prompt, num_tokens=10):
    messages = [{"role": "user", "content": prompt}]
    tokens = model.tokenizer.apply_chat_template(messages)
    seq_states = [SequenceKVState(c, seq_id=0) for c in model.kv_caches]
    token_ids = torch.tensor(tokens, dtype=torch.long, device=device)
    positions = torch.arange(len(tokens), dtype=torch.int32, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model.forward(token_ids, positions, seq_states)
    torch.cuda.synchronize()
    pf_t = time.perf_counter() - t0
    next_token = sample(logits[-1:, :], 0.0, 1, 1.0).item()
    generated = [next_token]
    decode_times = []
    for step in range(num_tokens - 1):
        pos = len(tokens) + step
        tt = torch.tensor([next_token], dtype=torch.long, device=device)
        pt = torch.tensor([pos], dtype=torch.int32, device=device)
        torch.cuda.synchronize()
        dt0 = time.perf_counter()
        logits = model.forward(tt, pt, seq_states)
        torch.cuda.synchronize()
        decode_times.append(time.perf_counter() - dt0)
        next_token = sample(logits, 0.0, 1, 1.0).item()
        generated.append(next_token)
        if next_token == model.cfg.eos_token_id:
            break
    for s in seq_states:
        s.free()
    text = model.tokenizer.decode(generated)
    return text, pf_t, decode_times, len(tokens)

# Test 1: 2+2
text, pf, dt, ntok = gen("What is 2+2? Reply with just the number.", 10)
first = text.strip().split()[0] if text.strip() else ""
ok1 = "4" in first
print(f"  2+2 = '{text.strip()[:80]}' -> {'PASS' if ok1 else 'FAIL'} (prefill {ntok} tok in {pf:.3f}s)")

# Test 2: Capital of France
text2, pf2, dt2, ntok2 = gen("What is the capital of France? One word only.", 10)
first2 = text2.strip().split()[0] if text2.strip() else ""
ok2 = "paris" in first2.lower()
print(f"  Capital = '{text2.strip()[:80]}' -> {'PASS' if ok2 else 'FAIL'}")

# Speed test
text3, pf3, dt3, ntok3 = gen("Write a poem about space.", 30)
if dt3:
    avg_ms = sum(t * 1000 for t in dt3) / len(dt3)
    print(f"  Prefill: {ntok3} tokens in {pf3:.3f}s ({ntok3/pf3:.0f} tok/s)")
    print(f"  Decode: avg {avg_ms:.1f}ms = {1000/avg_ms:.2f} tok/s")
    print(f"  Text: {text3[:150]}")

print(f"\nRESULT: {'ALL PASS' if ok1 and ok2 else 'FAIL'}")
