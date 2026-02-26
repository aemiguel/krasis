"""Compare Qwen3.5-35B-A3B output: HuggingFace FP32 CPU ground truth.

Uses model.generate() with KV cache for speed.
Greedy decode (temp=0 equivalent via do_sample=False).
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/main/.krasis/models/Qwen3.5-35B-A3B"

prompts = [
    [{"role": "user", "content": "Hi"}],
    [{"role": "user", "content": "Who trained you?"}],
    [{"role": "user", "content": "What is 2+2?"}],
    [{"role": "user", "content": "What is the largest animal on Earth?"}],
    [{"role": "user", "content": "Write me a short poem about the ocean."}],
]

MAX_NEW_TOKENS = 100

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"EOS token id: {tokenizer.eos_token_id} = {repr(tokenizer.eos_token)}")

print("Loading model (FP32 on CPU)... this will take a few minutes for ~134GB")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)
model.eval()
print(f"Model loaded in {time.time()-t0:.0f}s\n", flush=True)

for i, messages in enumerate(prompts):
    prompt_text = messages[0]["content"]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )
    input_ids = encoded["input_ids"]

    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    print(f"=== Prompt {i+1}: \"{prompt_text}\" ===")
    print(f"Input ({input_ids.shape[1]} tokens): {repr(input_text)}", flush=True)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy
            use_cache=True,
        )
    elapsed = time.time() - t0

    new_ids = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_ids, skip_special_tokens=False)

    # Print token by token
    for step, tid in enumerate(new_ids.tolist()):
        tok_text = tokenizer.decode([tid], skip_special_tokens=False)
        print(f"  token {step}: {tid} = {repr(tok_text)}")

    print(f"\nFull response ({len(new_ids)} tokens, {elapsed:.1f}s): {response}")
    print("=" * 60)
    print("", flush=True)
