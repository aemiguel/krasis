"""HF tokenizer wrapper with chat template support."""

import logging
from typing import Dict, List, Optional

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class Tokenizer:
    """Wraps HuggingFace tokenizer with chat template formatting."""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        logger.info(
            "Tokenizer loaded: vocab=%d, eos=%d, bos=%d",
            self.tokenizer.vocab_size,
            self.tokenizer.eos_token_id or -1,
            self.tokenizer.bos_token_id or -1,
        )

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.tokenizer.bos_token_id

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """Format messages using the model's chat template and tokenize."""
        result = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
        return self._ensure_int_list(result)

    def _ensure_int_list(self, result) -> List[int]:
        """Force result to List[int] regardless of what transformers returned."""
        if isinstance(result, str):
            return self.tokenizer.encode(result, add_special_tokens=False)
        # transformers v5: apply_chat_template returns BatchEncoding dict
        if isinstance(result, dict):
            result = result["input_ids"]
        # BatchEncoding may also behave like a dict but have .input_ids
        if hasattr(result, "input_ids"):
            result = result.input_ids
        if isinstance(result, list):
            if result and isinstance(result[0], str):
                return self.tokenizer.encode(
                    "".join(result), add_special_tokens=False
                )
            return [int(t) for t in result]
        # numpy array or tensor
        return [int(t) for t in result]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_incremental(self, token_id: int) -> str:
        """Decode a single token. Returns the text piece."""
        return self.tokenizer.decode([token_id], skip_special_tokens=False)
