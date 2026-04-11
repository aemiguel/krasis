#!/usr/bin/env python3
"""Generate reference greedy decode outputs for deterministic model validation.

Loads a model via HuggingFace transformers in BF16, runs the sanity prompts
with greedy decoding (do_sample=False), and stores output token IDs and text
as JSON. This provides the ground truth for validating Krasis decode correctness.

Usage:
    ./dev generate-reference <model-name> [--max-tokens N]

This script must be run via ./dev generate-reference, not directly.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from reference_contract import (
        REFERENCE_PROFILES,
        apply_capture_template,
        build_contract,
        canonical_profile_id,
        capture_settings_for_profile,
        emit_reference_generation_trace,
        profile_filename,
    )
except ModuleNotFoundError:
    from tests.reference_contract import (
        REFERENCE_PROFILES,
        apply_capture_template,
        build_contract,
        canonical_profile_id,
        capture_settings_for_profile,
        emit_reference_generation_trace,
        profile_filename,
    )

# Guard: must be run via ./dev
if not os.environ.get("KRASIS_DEV_SCRIPT"):
    print("ERROR: This script must be run via ./dev generate-reference, not directly.")
    print("  Usage: ./dev generate-reference <model-name>")
    sys.exit(1)

MODELS_DIR = os.path.expanduser("~/.krasis/models")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
REFERENCE_DIR = SCRIPT_DIR / "reference_outputs"
PROMPTS_FILE = REPO_DIR / "benchmarks" / "sanity_test_prompts.txt"

BOLD = "\033[1m"
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
NC = "\033[0m"


def info(msg: str):
    print(f"{CYAN}{BOLD}=>{NC} {msg}")

def ok(msg: str):
    print(f"{GREEN}{BOLD}OK{NC} {msg}")

def warn(msg: str):
    print(f"{YELLOW}{BOLD}!!{NC} {msg}")

def die(msg: str):
    print(f"{RED}{BOLD}ERROR{NC} {msg}", file=sys.stderr)
    sys.exit(1)


def parse_prompt_conversations(lines: List[str]) -> List[List[str]]:
    """Parse prompt lines into conversations.

    Lines starting with '- ' are continuations of the previous conversation.
    All other non-empty lines start a new conversation.
    """
    conversations: List[List[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            prompt_text = stripped[2:].strip()
            if not prompt_text:
                continue
            if conversations:
                conversations[-1].append(prompt_text)
            else:
                conversations.append([prompt_text])
        else:
            conversations.append([stripped])
    return conversations


def generate_reference(model_name: str, max_new_tokens: int = 200, profile: str = "auto"):
    """Generate reference outputs using HuggingFace transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_path):
        die(f"Model not found: {model_path}")

    if not PROMPTS_FILE.is_file():
        die(f"Prompts file not found: {PROMPTS_FILE}")

    # Parse prompts
    with open(PROMPTS_FILE) as f:
        lines = f.readlines()
    conversations = parse_prompt_conversations(lines)
    total_prompts = sum(len(c) for c in conversations)

    info(f"Model: {model_name}")
    info(f"Path: {model_path}")
    info(f"Conversations: {len(conversations)} ({total_prompts} prompts)")
    info(f"Max new tokens: {max_new_tokens}")
    # Use only GPU 0 for loading; offload the rest to CPU.
    # QCN has 512 experts/layer — too large for multi-GPU with transformers.
    # Speed doesn't matter here, correctness does.
    num_gpus = torch.cuda.device_count()
    max_memory = {"cpu": "200GiB"}
    # Use all available GPUs — give each one most of its memory
    for i in range(num_gpus):
        free_mb = torch.cuda.mem_get_info(i)[0] // (1024 * 1024)
        alloc_gb = max(1, (free_mb - 2048)) / 1024  # leave 2GB headroom
        max_memory[i] = f"{alloc_gb:.1f}GiB"
        info(f"GPU {i}: {free_mb}MB free, allocating {alloc_gb:.1f}GiB")
    info(f"Loading model in BF16 ({num_gpus} GPUs + CPU offload)...")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    info(f"Tokenizer loaded ({time.time() - t0:.1f}s)")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )
    model.eval()
    load_time = time.time() - t0
    info(f"Model loaded ({load_time:.1f}s)")

    # Determine EOS tokens
    eos_token_ids = []
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)
    # Qwen3 models use <|im_end|> as end of turn
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
        eos_token_ids.append(im_end_id)
    # Also check for <|endoftext|>
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id:
        eos_token_ids.append(eot_id)

    eos_token_ids = list(set(eos_token_ids))
    info(f"EOS token IDs: {eos_token_ids}")

    profile_id = canonical_profile_id(tokenizer, profile)
    capture_settings = capture_settings_for_profile(profile_id)
    capture_settings["source"] = "local_generate_reference"
    info(f"Reference profile: {profile_id}")

    # Generate reference outputs
    result = {
        "format_version": 4,
        "model": model_name,
        "model_path": model_path,
        "generated_at": datetime.now().isoformat(),
        "runtime": "transformers",
        "max_new_tokens": max_new_tokens,
        "eos_token_ids": eos_token_ids,
        "capture_settings": capture_settings,
        "conversations": [],
    }

    try:
        import transformers
        result["runtime_version"] = transformers.__version__
    except Exception:
        result["runtime_version"] = "unknown"

    prompt_num = 0
    for conv_idx, conversation in enumerate(conversations):
        conv_result: Dict[str, Any] = {"turns": []}
        messages: List[Dict[str, str]] = []
        is_multi = len(conversation) > 1

        if is_multi:
            info(f"Conversation {conv_idx + 1} ({len(conversation)} turns)")

        for turn_idx, prompt in enumerate(conversation):
            prompt_num += 1
            turn_label = f" [turn {turn_idx + 1}/{len(conversation)}]" if is_multi else ""
            print(f"  {GREEN}[{prompt_num}/{total_prompts}]{NC} {prompt}{turn_label}")

            messages.append({"role": "user", "content": prompt})

            # Apply chat template — disable thinking if the model supports it
            # We want the "normal" response for reference, not thinking tokens
            try:
                template_out = apply_capture_template(tokenizer, messages, capture_settings)
            except TypeError:
                die(
                    "Requested reference profile is incompatible with this tokenizer/chat template: "
                    f"profile={profile_id}"
                )

            # apply_chat_template may return a tensor or a BatchEncoding
            if hasattr(template_out, "input_ids"):
                input_ids = template_out.input_ids
            elif isinstance(template_out, torch.Tensor):
                input_ids = template_out
            else:
                input_ids = torch.tensor([template_out], dtype=torch.long)

            input_ids = input_ids.to(model.device)
            input_len = input_ids.shape[1]

            t0 = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding
                    eos_token_id=eos_token_ids if eos_token_ids else None,
                )
            gen_time = time.time() - t0

            # Extract only new tokens
            new_token_ids = output_ids[0][input_len:].tolist()

            # Strip trailing EOS/pad tokens
            while new_token_ids and new_token_ids[-1] in eos_token_ids:
                new_token_ids.pop()

            text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
            tok_s = len(new_token_ids) / gen_time if gen_time > 0 else 0

            preview = text[:120].replace("\n", " ")
            print(f"    {DIM}{len(new_token_ids)} tokens, {gen_time:.1f}s ({tok_s:.1f} tok/s){NC}")
            print(f"    {DIM}{preview}{'...' if len(text) > 120 else ''}{NC}")

            turn_result = {
                "prompt": prompt,
                "input_token_ids": input_ids[0].tolist(),
                "token_ids": new_token_ids,
                "text": text,
                "num_tokens": len(new_token_ids),
            }
            conv_result["turns"].append(turn_result)

            # Add assistant response to history for multi-turn
            messages.append({"role": "assistant", "content": text})

        result["conversations"].append(conv_result)

    result["contract"] = build_contract(
        model_name=model_name,
        model_path=model_path,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        add_generation_prompt=True,
        enable_thinking=capture_settings["enable_thinking"],
        profile_id=capture_settings["profile_id"],
        prompt_source_path=str(PROMPTS_FILE),
        runtime_name="transformers",
        runtime_version=result.get("runtime_version"),
        torch_dtype="bfloat16",
    )

    # Save reference
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REFERENCE_DIR / model_name / profile_filename(profile_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    emit_reference_generation_trace(
        "generate_reference",
        result["contract"],
        str(output_path),
        max_new_tokens=max_new_tokens,
    )

    ok(f"Reference saved: {output_path}")
    ok(
        "Reference metadata: "
        f"profile={result['contract'].get('profile_id')} "
        f"template_hash={result['contract'].get('tokenizer', {}).get('chat_template_hash')} "
        f"prompt_sha256={result['contract'].get('prompt_source', {}).get('sha256')}"
    )
    ok(f"Total: {total_prompts} prompts, {sum(len(c['turns']) for c in result['conversations'])} turns")


def main():
    parser = argparse.ArgumentParser(
        description="Generate BF16 HuggingFace greedy reference outputs with stored contract metadata"
    )
    parser.add_argument("model", help="Model directory name under ~/.krasis/models/")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max new tokens per turn (default: 200)")
    parser.add_argument(
        "--profile",
        default="auto",
        choices=("auto",) + REFERENCE_PROFILES,
        help=(
            "Reference capture profile. auto chooses greedy_chat_thinking_off when "
            "the tokenizer chat template supports enable_thinking, otherwise greedy_chat_default"
        ),
    )
    args = parser.parse_args()

    generate_reference(args.model, args.max_tokens, args.profile)


if __name__ == "__main__":
    main()
