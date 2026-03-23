"""Token sampling using torch."""

from typing import Optional, Set

import torch


def sample(
    logits: torch.Tensor,
    temperature: float = 0.6,
    top_k: int = 50,
    top_p: float = 0.95,
    presence_penalty: float = 0.0,
    generated_tokens: Optional[Set[int]] = None,
) -> torch.Tensor:
    """Sample next token from logits.

    Args:
        logits: [batch_size, vocab_size] float32/bf16
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k filtering (0 = disabled)
        top_p: Top-p (nucleus) filtering (1.0 = disabled)
        presence_penalty: Penalty subtracted from logits of already-seen tokens
        generated_tokens: Set of token IDs already generated (for presence penalty)

    Returns:
        [batch_size] int32 token IDs
    """
    logits = logits.float()

    # Apply presence penalty before temperature scaling
    if presence_penalty != 0.0 and generated_tokens:
        token_ids = torch.tensor(list(generated_tokens), dtype=torch.long,
                                 device=logits.device)
        logits[:, token_ids] -= presence_penalty

    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=False).to(torch.int32)

    if temperature != 1.0:
        logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, top_k, dim=-1)
        threshold = topk_vals[:, -1].unsqueeze(-1)
        logits = logits.where(logits >= threshold, torch.tensor(float('-inf'), device=logits.device))

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift so that first token above threshold is kept
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    probs = torch.softmax(logits, dim=-1)
    samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return samples.to(torch.int32)
