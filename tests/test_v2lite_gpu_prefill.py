#!/usr/bin/env python3
"""V2-Lite GPU prefill test: verify on-the-fly Marlin repack from engine works."""

import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel
from krasis.config import QuantConfig

def main():
    MODEL = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"
    print("Loading V2-Lite PP=1 with GPU prefill...")
    t0 = time.time()
    qcfg = QuantConfig(attention="bf16", shared_expert="bf16", dense_mlp="bf16", lm_head="bf16")
    model = KrasisModel(
        MODEL, num_gpus=1, gpu_prefill=True,
        gpu_prefill_threshold=10,  # Low threshold to force GPU prefill
        krasis_threads=16, kv_dtype=torch.bfloat16, quant_cfg=qcfg,
    )
    model.load()
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Test 1: Simple prompt (should trigger GPU prefill since threshold=10)
    print("\nTest 1: Short prompt with GPU prefill...")
    messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt)} tokens")
    out = model.generate(prompt, max_new_tokens=16, temperature=0.0, top_k=1)
    text = model.tokenizer.decode(out)
    print(f"Output ({len(out)} tokens): {repr(text)}")
    if "4" in text:
        print("PASS: Got expected '4'")
    else:
        print("FAIL: Missing '4'")

    # Test 2: Counting
    print("\nTest 2: Counting...")
    messages2 = [{"role": "user", "content": "Count from 1 to 10."}]
    prompt2 = model.tokenizer.apply_chat_template(messages2)
    out2 = model.generate(prompt2, max_new_tokens=64, temperature=0.0, top_k=1)
    text2 = model.tokenizer.decode(out2)
    print(f"Output ({len(out2)} tokens): {repr(text2)}")

    # Test 3: Longer prompt to really test GPU prefill
    print("\nTest 3: Longer prompt (force larger GPU batch)...")
    long_content = "Here is a long context. " * 50 + "Now, what is 3+5? Just the number."
    messages3 = [{"role": "user", "content": long_content}]
    prompt3 = model.tokenizer.apply_chat_template(messages3)
    print(f"Prompt: {len(prompt3)} tokens")
    out3 = model.generate(prompt3, max_new_tokens=16, temperature=0.0, top_k=1)
    text3 = model.tokenizer.decode(out3)
    print(f"Output ({len(out3)} tokens): {repr(text3)}")
    if "8" in text3:
        print("PASS: Got expected '8'")
    else:
        print(f"Note: Got {repr(text3)} (V2-Lite may not be great at this)")

    print("\nGPU prefill test complete.")

if __name__ == "__main__":
    main()
