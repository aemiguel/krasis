"""Shared prompt loading utilities for test scripts.

All test prompts must use the canonical Gutenberg literary texts from
benchmarks/prompts/. We NEVER generate synthetic text for testing.

See krasis-internal/feature-test-methodology.md for the full rationale.
"""

import os
from typing import List, Optional

_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmarks", "prompts",
)

_cached_texts: dict = {}


def _prompt_files() -> List[str]:
    """Return sorted list of prompt filenames."""
    return sorted(f for f in os.listdir(_PROMPTS_DIR) if f.endswith(".txt"))


def load_prompt_text(index: int = 0) -> str:
    """Load a full Gutenberg prompt by index (0-5). Cached after first load."""
    files = _prompt_files()
    if not files:
        raise FileNotFoundError(f"No prompt files found in {_PROMPTS_DIR}")
    fname = files[index % len(files)]
    if fname not in _cached_texts:
        with open(os.path.join(_PROMPTS_DIR, fname)) as f:
            _cached_texts[fname] = f.read()
    return _cached_texts[fname]


def load_prompt_text_truncated(max_chars: int, index: int = 0) -> str:
    """Load a Gutenberg prompt truncated to max_chars."""
    text = load_prompt_text(index)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def build_prompt_tokens(model, target_tokens: int, index: int = 0) -> list:
    """Build a tokenized prompt of approximately target_tokens length.

    Uses a Gutenberg text, truncated via binary search on the tokenizer
    to hit the target token count precisely.
    """
    text = load_prompt_text(index)
    question = "\n\nSummarize the above text briefly."
    content = text + question

    # Quick check: if full text is under target, just use it
    messages = [{"role": "user", "content": content}]
    tokens = model.tokenizer.apply_chat_template(messages)
    if len(tokens) <= target_tokens:
        return tokens[:target_tokens] if len(tokens) > target_tokens else tokens

    # Binary search for the right content length
    lo, hi = 0, len(text)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = text[:mid] + question
        msgs = [{"role": "user", "content": trial}]
        n_tok = len(model.tokenizer.apply_chat_template(msgs))
        if n_tok <= target_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    content = text[:best] + question
    messages = [{"role": "user", "content": content}]
    return model.tokenizer.apply_chat_template(messages)
