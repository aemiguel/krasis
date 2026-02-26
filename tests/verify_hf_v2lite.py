"""Compare V2-Lite output: HuggingFace BF16 CPU vs Krasis.

Manual autoregressive decode (no model.generate) to avoid KV cache API issues.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/main/.krasis/models/DeepSeek-V2-Lite"

prompts = [
    [{"role": "user", "content": "Hi"}],
    [{"role": "user", "content": "Who trained you?"}],
    [{"role": "user", "content": "What is 2+2?"}],
    [{"role": "user", "content": "What is the largest animal on Earth?"}],
    [{"role": "user", "content": "Write me a short poem about the ocean."}],
]

MAX_NEW_TOKENS = 60

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading model (BF16 on CPU)... this will take a minute")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
model.eval()
print(f"Model loaded. EOS token id: {tokenizer.eos_token_id}\n")

for i, messages in enumerate(prompts):
    prompt_text = messages[0]["content"]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )

    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    print(f"=== Prompt {i+1}: \"{prompt_text}\" ===")
    print(f"Input ({len(input_ids[0])} tokens): {repr(input_text)}")

    # Manual greedy autoregressive decode (no KV cache)
    generated_ids = input_ids[0].tolist()
    for step in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            inp = torch.tensor([generated_ids], dtype=torch.long)
            outputs = model(inp, use_cache=False)
            logits = outputs.logits[0, -1, :]  # last position
            next_token = torch.argmax(logits).item()

        if next_token == tokenizer.eos_token_id:
            break
        generated_ids.append(next_token)
        # Print token as we go
        tok_text = tokenizer.decode([next_token], skip_special_tokens=False)
        print(f"  token {step}: {next_token} = {repr(tok_text)}")

    new_tokens = generated_ids[len(input_ids[0]):]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"\nFull response ({len(new_tokens)} tokens): {response}")
    print("=" * 60)
    print()
