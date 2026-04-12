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
import shlex
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
        build_reference_sanity_report,
        canonical_profile_id,
        collect_capture_stop_ids,
        capture_settings_for_profile,
        emit_reference_generation_trace,
        profile_filename,
    )
except ModuleNotFoundError:
    from tests.reference_contract import (
        REFERENCE_PROFILES,
        apply_capture_template,
        build_contract,
        build_reference_sanity_report,
        canonical_profile_id,
        collect_capture_stop_ids,
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
DEFAULT_DIAGNOSTIC_STEPS = 4
DEFAULT_DIAGNOSTIC_TOPK = 10

MODEL_ALIASES = {
    "qcn": "Qwen3-Coder-Next",
    "qwen3-coder-next": "Qwen3-Coder-Next",
    "qwen35": "Qwen3.5-35B-A3B",
    "q35b": "Qwen3.5-35B-A3B",
    "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B",
    "q122b": "Qwen3.5-122B-A10B",
    "qwen122": "Qwen3.5-122B-A10B",
    "qwen3.5-122b-a10b": "Qwen3.5-122B-A10B",
    "gemma": "gemma-4-26B-A4B-it",
    "gemma26": "gemma-4-26B-A4B-it",
    "gemma-4-26b-a4b-it": "gemma-4-26B-A4B-it",
    "minimax": "MiniMax-M2.5",
    "minimax25": "MiniMax-M2.5",
    "minimax-m2.5": "MiniMax-M2.5",
    "q235": "Qwen3-235B-A22B",
    "qwen235": "Qwen3-235B-A22B",
    "qwen3-235b-a22b": "Qwen3-235B-A22B",
    "q397": "Qwen3.5-397B-A17B",
    "qwen397": "Qwen3.5-397B-A17B",
    "qwen3.5-397b-a17b": "Qwen3.5-397B-A17B",
}

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


def build_invocation_metadata(model_name: str, profile_id: str, max_new_tokens: int) -> Dict[str, Any]:
    run_dir = os.environ.get("KRASIS_RUN_DIR")
    script_argv = [str(Path(arg)) if idx == 0 else arg for idx, arg in enumerate(sys.argv)]
    dev_command = os.environ.get("KRASIS_DEV_CMD")
    if not dev_command:
        dev_command = " ".join(shlex.quote(arg) for arg in (["./dev", "generate-reference"] + sys.argv[1:]))
    metadata: Dict[str, Any] = {
        "captured_at": datetime.now().isoformat(),
        "dev_command": dev_command,
        "script_argv": script_argv,
        "cwd": str(Path.cwd().resolve()),
        "run_type": os.environ.get("KRASIS_RUN_TYPE"),
        "model": model_name,
        "profile_id": profile_id,
        "max_new_tokens": max_new_tokens,
    }
    if run_dir:
        metadata["run_dir"] = str(Path(run_dir).resolve())
    return metadata


def resolve_model_name(raw_model: str) -> str:
    raw = raw_model.strip()
    if not raw:
        die("Model name cannot be empty")

    candidate = Path(raw)
    if candidate.name != raw:
        raw = candidate.name

    if os.path.isdir(os.path.join(MODELS_DIR, raw)):
        return raw

    lowered = raw.lower()
    resolved = MODEL_ALIASES.get(lowered)
    if resolved and os.path.isdir(os.path.join(MODELS_DIR, resolved)):
        return resolved

    if "/" in raw_model:
        basename = Path(raw_model).name
        if os.path.isdir(os.path.join(MODELS_DIR, basename)):
            return basename

    if resolved:
        return resolved
    return raw


def write_run_manifest(
    invocation: Dict[str, Any],
    output_path: Path,
    model_name: str,
    profile_id: str,
    max_new_tokens: int,
    diagnostic_steps: int,
    diagnostic_top_k: int,
    sanity: Optional[Dict[str, Any]] = None,
) -> None:
    run_dir = invocation.get("run_dir")
    if not run_dir:
        return
    manifest_path = Path(run_dir) / "generate_reference_manifest.json"
    manifest = {
        "kind": "generate_reference_manifest",
        "captured_at": invocation.get("captured_at"),
        "dev_command": invocation.get("dev_command"),
        "script_argv": invocation.get("script_argv"),
        "cwd": invocation.get("cwd"),
        "run_dir": run_dir,
        "run_type": invocation.get("run_type"),
        "model": model_name,
        "profile_id": profile_id,
        "max_new_tokens": max_new_tokens,
        "decode_diagnostics": {
            "captured_steps_per_turn": diagnostic_steps,
            "top_k": diagnostic_top_k,
        },
        "reference_output_path": str(output_path.resolve()),
        "sanity": sanity,
        "follow_up_commands": [
            "./dev reference-inventory",
            "./dev validate <config> --max-prompts 0 --no-server --port 1",
            "./dev reference-test <config>",
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    ok(f"Run manifest saved: {manifest_path}")


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


def _decode_token_text(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return ""


def build_teacher_forced_per_token_data(
    model: Any,
    tokenizer: Any,
    prompt_input_ids: Any,
    generated_token_ids: List[int],
    *,
    diagnostic_steps: int,
    diagnostic_top_k: int,
) -> List[Dict[str, Any]]:
    if diagnostic_steps <= 0 or diagnostic_top_k <= 0 or not generated_token_ids:
        return []

    import torch

    steps_to_capture = min(len(generated_token_ids), diagnostic_steps)
    teacher_suffix = generated_token_ids[: max(0, steps_to_capture - 1)]
    teacher_ids = prompt_input_ids[0].tolist() + teacher_suffix
    teacher_tensor = torch.tensor([teacher_ids], dtype=prompt_input_ids.dtype, device=model.device)

    with torch.no_grad():
        logits = model(teacher_tensor).logits[0].float()

    prompt_len = prompt_input_ids.shape[1]
    vocab_size = logits.shape[-1]
    top_k = min(int(diagnostic_top_k), int(vocab_size))
    per_token_data: List[Dict[str, Any]] = []

    for step in range(steps_to_capture):
        logit_pos = prompt_len - 1 + step
        step_logits = logits[logit_pos]
        log_probs = torch.log_softmax(step_logits, dim=-1)
        expected_token_id = int(generated_token_ids[step])
        expected_log_prob = float(log_probs[expected_token_id].item())
        expected_logit = float(step_logits[expected_token_id].item())
        expected_rank = int(torch.count_nonzero(step_logits > step_logits[expected_token_id]).item()) + 1

        top_vals, top_ids = torch.topk(log_probs, k=top_k)
        top_k_entries = []
        for tok_id, log_prob in zip(top_ids.tolist(), top_vals.tolist()):
            top_k_entries.append(
                {
                    "token_id": int(tok_id),
                    "text": _decode_token_text(tokenizer, int(tok_id)),
                    "log_prob": float(log_prob),
                }
            )

        prev_token_id = int(prompt_input_ids[0, -1].item()) if step == 0 else int(generated_token_ids[step - 1])
        per_token_data.append(
            {
                "position": int(logit_pos),
                "previous_token_id": prev_token_id,
                "previous_token_text": _decode_token_text(tokenizer, prev_token_id),
                "token_id": expected_token_id,
                "text": _decode_token_text(tokenizer, expected_token_id),
                "log_prob": expected_log_prob,
                "logit": expected_logit,
                "rank": expected_rank,
                "top_k": top_k_entries,
            }
        )

    return per_token_data


def generate_reference(
    model_name: str,
    max_new_tokens: int = 200,
    profile: str = "auto",
    diagnostic_steps: int = DEFAULT_DIAGNOSTIC_STEPS,
    diagnostic_top_k: int = DEFAULT_DIAGNOSTIC_TOPK,
):
    """Generate reference outputs using HuggingFace transformers."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    model_name = resolve_model_name(model_name)
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
    info(f"Decode diagnostics: first {diagnostic_steps} steps, top-{diagnostic_top_k}")
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
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    info(f"Config loaded ({time.time() - t0:.1f}s)")

    architectures = list(getattr(config, "architectures", []) or [])
    use_conditional_loader = any("ConditionalGeneration" in arch for arch in architectures)
    model_loader = AutoModelForImageTextToText if use_conditional_loader else AutoModelForCausalLM
    info(
        "HF model loader: "
        f"{model_loader.__name__} (architectures={architectures if architectures else ['unknown']})"
    )

    t0 = time.time()
    model = model_loader.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )
    model.eval()
    load_time = time.time() - t0
    info(f"Model loaded ({load_time:.1f}s)")

    eos_token_ids = collect_capture_stop_ids(
        tokenizer,
        model_path=model_path,
        config_json=config.to_dict() if hasattr(config, "to_dict") else None,
    )
    info(f"EOS token IDs: {eos_token_ids}")

    profile_id = canonical_profile_id(tokenizer, profile)
    capture_settings = capture_settings_for_profile(profile_id)
    capture_settings["source"] = "local_generate_reference"
    info(f"Reference profile: {profile_id}")
    invocation = build_invocation_metadata(model_name, profile_id, max_new_tokens)

    # Generate reference outputs
    result = {
        "format_version": 5,
        "model": model_name,
        "model_path": model_path,
        "generated_at": datetime.now().isoformat(),
        "runtime": "transformers",
        "profile_id": profile_id,
        "max_new_tokens": max_new_tokens,
        "eos_token_ids": eos_token_ids,
        "capture_settings": capture_settings,
        "capture_invocation": invocation,
        "decode_diagnostics": {
            "schema_version": 1,
            "coverage": "teacher_forced_first_generated_steps",
            "captured_steps_per_turn": diagnostic_steps,
            "top_k": diagnostic_top_k,
            "log_prob_base": "natural_log",
        },
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
            per_token_data = build_teacher_forced_per_token_data(
                model,
                tokenizer,
                input_ids,
                new_token_ids,
                diagnostic_steps=diagnostic_steps,
                diagnostic_top_k=diagnostic_top_k,
            )
            if per_token_data:
                turn_result["per_token_data"] = per_token_data
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
        extra={"capture_invocation": invocation},
    )
    result["sanity"] = build_reference_sanity_report(
        result,
        expected_conversations=len(conversations),
        expected_turns=total_prompts,
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
    write_run_manifest(
        invocation,
        output_path,
        model_name,
        profile_id,
        max_new_tokens,
        diagnostic_steps,
        diagnostic_top_k,
        result["sanity"],
    )

    ok(f"Reference saved: {output_path}")
    ok(
        "Reference metadata: "
        f"profile={result['contract'].get('profile_id')} "
        f"template_hash={result['contract'].get('tokenizer', {}).get('chat_template_hash')} "
        f"prompt_sha256={result['contract'].get('prompt_source', {}).get('sha256')}"
    )
    ok(f"Total: {total_prompts} prompts, {sum(len(c['turns']) for c in result['conversations'])} turns")
    sanity = result["sanity"]
    if sanity.get("status") == "pass":
        ok("Reference sanity gate: PASS")
    else:
        warn(
            "Reference sanity gate: FAIL "
            + ", ".join(sanity.get("failed_checks", []))
        )
        for check in sanity.get("checks", []):
            if not check.get("ok"):
                warn(f"  {check.get('name')}: {check.get('detail')}")
        for sample in sanity.get("thought_leakage", {}).get("samples", []):
            warn(
                "  thought sample "
                f"conv={sample.get('conversation_index')} turn={sample.get('turn_index')}: "
                f"{sample.get('preview')}"
            )
        die(
            "Reference capture saved but not blessed. Fix the capture issue and recapture before using it as golden data."
        )


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
    parser.add_argument(
        "--diag-steps",
        type=int,
        default=DEFAULT_DIAGNOSTIC_STEPS,
        help=f"Teacher-forced decode diagnostic coverage per turn (default: {DEFAULT_DIAGNOSTIC_STEPS})",
    )
    parser.add_argument(
        "--diag-topk",
        type=int,
        default=DEFAULT_DIAGNOSTIC_TOPK,
        help=f"Top-k entries to store for each diagnostic step (default: {DEFAULT_DIAGNOSTIC_TOPK})",
    )
    args = parser.parse_args()

    if args.diag_steps < 0:
        die("--diag-steps must be >= 0")
    if args.diag_topk < 0:
        die("--diag-topk must be >= 0")

    generate_reference(
        args.model,
        args.max_tokens,
        args.profile,
        diagnostic_steps=args.diag_steps,
        diagnostic_top_k=args.diag_topk,
    )


if __name__ == "__main__":
    main()
