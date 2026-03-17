#!/usr/bin/env python3
"""Test what the Qwen3 chat template renders for multi-turn with thinking.

Doesn't need a running server -- just loads the tokenizer and renders templates.
Shows exactly what prompt the model sees at each turn.

Usage:
    python3 tests/repro_template.py
"""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer


MODEL_PATH = str(Path.home() / ".krasis/models/Qwen3.5-35B-A3B")


def render_and_show(tokenizer, messages, label, enable_thinking=True):
    """Render messages with the chat template and show the result."""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"\nMessages ({len(messages)}):")
    for i, m in enumerate(messages):
        content = m["content"][:80]
        if len(m["content"]) > 80:
            content += "..."
        print(f"  [{i}] {m['role']}: {repr(content)}")

    # Render as text (not token IDs)
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        tokenize=False,
    )
    print(f"\nRendered prompt ({len(rendered)} chars):")
    print("---START---")
    print(rendered)
    print("---END---")

    # Also get token IDs
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    print(f"\nToken count: {len(token_ids)}")

    # Show last ~20 tokens to see exactly what the model starts generating from
    last_n = min(20, len(token_ids))
    last_tokens = token_ids[-last_n:]
    print(f"\nLast {last_n} tokens (what model sees right before generation):")
    for tid in last_tokens:
        text = tokenizer.decode([tid])
        print(f"  {tid:>8d} = {repr(text)}")

    return rendered, token_ids


def main():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Scenario 1: Turn 1 (just "hi")
    render_and_show(tokenizer, [
        {"role": "user", "content": "hi"},
    ], "TURN 1: Fresh conversation")

    # Scenario 2a: Turn 2 where model DID NOT think (trivial thinking)
    # This is what happens: server injects <think>, model immediately generates </think>
    # So raw_text = "<think></think>Hello! How can I help you today?"
    render_and_show(tokenizer, [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think></think>Hello! How can I help you today?"},
        {"role": "user", "content": "hi"},
    ], "TURN 2a: History has <think></think> (trivial thinking)")

    # Scenario 2b: Turn 2 where model DID think
    # raw_text = "<think>Let me greet the user.</think>Hello! How can I help you today?"
    render_and_show(tokenizer, [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>Let me greet the user.</think>Hello! How can I help you today?"},
        {"role": "user", "content": "hi"},
    ], "TURN 2b: History has thinking content")

    # Scenario 2c: Turn 2 where model response has NO thinking tags at all
    # (what if we strip thinking from history?)
    render_and_show(tokenizer, [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": "hi"},
    ], "TURN 2c: History has NO thinking tags (stripped)")

    # Scenario 2d: Turn 2 with reasoning_content field (proper API format)
    render_and_show(tokenizer, [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello! How can I help you today?", "reasoning_content": ""},
        {"role": "user", "content": "hi"},
    ], "TURN 2d: Using reasoning_content field (empty)")

    # Scenario 2e: Turn 2 with thinking=False
    render_and_show(tokenizer, [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": "hi"},
    ], "TURN 2e: enable_thinking=False", enable_thinking=False)

    # Scenario 3: What the model actually produces - simulate broken turn 2
    # where model thinks but produces no answer
    # raw_text would be "<think>some thinking</think>" (no answer after closing tag)
    # or "<think>some thinking" (never closes think tag, hits EOS)
    print(f"\n\n{'#'*70}")
    print("KEY QUESTION: What is different about the turn 2 prompt that makes")
    print("the model fail? Compare the prompts above visually.")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
