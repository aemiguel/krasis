"""Token sampling using FlashInfer kernels."""

import torch
from flashinfer.sampling import top_k_top_p_sampling_from_logits


def sample(
    logits: torch.Tensor,
    temperature: float = 0.6,
    top_k: int = 50,
    top_p: float = 0.95,
) -> torch.Tensor:
    """Sample next token from logits.

    Args:
        logits: [batch_size, vocab_size] float32/bf16
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k filtering (0 = disabled)
        top_p: Top-p (nucleus) filtering (1.0 = disabled)

    Returns:
        [batch_size] int32 token IDs
    """
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=False).to(torch.int32)

    logits = logits.float()
    if temperature != 1.0:
        logits = logits / temperature

    # FlashInfer 0.6.1: returns single tensor of sampled token IDs
    samples = top_k_top_p_sampling_from_logits(
        logits, top_k=top_k, top_p=top_p, filter_apply_order="joint",
    )
    return samples.to(torch.int32)
