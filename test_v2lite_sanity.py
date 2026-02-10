#!/usr/bin/env python3
"""Quick V2-Lite sanity test: verify standalone model still generates correctly."""

import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel

def main():
    MODEL = "/home/main/Documents/Claude/hf-models/DeepSeek-V2-Lite"
    print("Loading V2-Lite PP=1...")
    t0 = time.time()
    model = KrasisModel(MODEL, num_gpus=1, gpu_prefill=False, krasis_threads=16, kv_dtype=torch.bfloat16)
    model.load()
    print(f"Loaded in {time.time()-t0:.1f}s")

    print("\nGenerating (greedy)...")
    messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    prompt = model.tokenizer.apply_chat_template(messages)
    print(f"Prompt: {len(prompt)} tokens")
    out = model.generate(prompt, max_new_tokens=32, temperature=0.0, top_k=1)
    text = model.tokenizer.decode(out)
    print(f"Output ({len(out)} tokens): {repr(text)}")

    # Second test: slightly longer prompt
    messages2 = [{"role": "user", "content": "Count from 1 to 10."}]
    prompt2 = model.tokenizer.apply_chat_template(messages2)
    out2 = model.generate(prompt2, max_new_tokens=64, temperature=0.0, top_k=1)
    text2 = model.tokenizer.decode(out2)
    print(f"Output2 ({len(out2)} tokens): {repr(text2)}")

    print("\nV2-Lite test complete.")

if __name__ == "__main__":
    main()
