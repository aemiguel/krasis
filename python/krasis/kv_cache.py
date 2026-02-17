"""Paged KV cache supporting both MLA and GQA attention.

MLA (DeepSeek/Kimi): compresses all KV heads into a single latent vector per token.
  Split (FlashInfer MLA):
    - ckv_cache: [num_layers, num_pages, page_size, kv_lora_rank]
    - kpe_cache: [num_layers, num_pages, page_size, qk_rope_head_dim]
  Combined (TRTLLM MLA):
    - kv_cache: [num_layers, num_pages, page_size, kv_lora_rank + qk_rope_head_dim]

GQA (Qwen3): standard K/V caches with head dimension (NHD layout).
    - k_cache: [num_layers, num_pages, page_size, num_kv_heads, head_dim]
    - v_cache: [num_layers, num_pages, page_size, num_kv_heads, head_dim]
"""

import logging
import math
from typing import List, Optional, Tuple

import torch

from krasis.config import ModelConfig

logger = logging.getLogger(__name__)

PAGE_SIZE = 16  # tokens per page

# TRTLLM kernel constraint: block_num % (128 / page_size) == 0
TRTLLM_BLOCK_CONSTRAINT = 128


class PagedKVCache:
    """Manages paged KV cache for a set of layers on one GPU.

    Allocates a fixed pool of pages at init. Sequences claim pages
    from the free list as they grow; pages are returned on free.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        num_layers: int,
        device: torch.device,
        max_pages: Optional[int] = None,
        kv_dtype: torch.dtype = torch.float8_e4m3fn,
        page_size: int = PAGE_SIZE,
        combined: bool = False,
        max_mb: Optional[int] = None,
    ):
        self.cfg = cfg
        self.num_layers = num_layers
        self.device = device
        self.page_size = page_size
        self.kv_dtype = kv_dtype
        self.combined = combined
        self.attention_type = cfg.attention_type  # "mla" or "gqa"

        # Compute cache dimensions based on attention type
        if cfg.is_mla:
            self.ckv_dim = cfg.kv_lora_rank        # 512
            self.kpe_dim = cfg.qk_rope_head_dim    # 64
            self.kv_cache_dim = self.ckv_dim + self.kpe_dim  # 576
            self.num_kv_heads = None
            self.gqa_head_dim = None
        else:
            # GQA: standard K/V with head dimension
            self.ckv_dim = None
            self.kpe_dim = None
            self.num_kv_heads = cfg.num_key_value_heads
            self.gqa_head_dim = cfg.gqa_head_dim
            self.kv_cache_dim = cfg.num_key_value_heads * cfg.gqa_head_dim * 2  # K + V

        # Size from max_mb (preferred) or max_pages (explicit)
        if max_pages is None:
            if max_mb is None:
                max_mb = 2000  # default 2 GB
            budget_bytes = max_mb * 1024 * 1024
            bytes_per_page = self._bytes_per_page()
            max_pages = max(64, budget_bytes // bytes_per_page)
            logger.info(
                "KV cache: %d MB → %d pages (%.1fK tokens)",
                max_mb, max_pages, max_pages * page_size / 1000,
            )

        self.max_pages = max_pages

        # GQA caches (separate K and V)
        self.k_cache = None
        self.v_cache = None
        # MLA caches
        self.ckv_cache = None
        self.kpe_cache = None
        self.kv_cache = None

        if cfg.is_gqa:
            # GQA: separate K and V caches [layers, pages, page_size, heads, head_dim]
            self.k_cache = torch.zeros(
                num_layers, max_pages, page_size, self.num_kv_heads, self.gqa_head_dim,
                dtype=kv_dtype, device=device,
            )
            self.v_cache = torch.zeros(
                num_layers, max_pages, page_size, self.num_kv_heads, self.gqa_head_dim,
                dtype=kv_dtype, device=device,
            )
            alloc_mb = (self.k_cache.nbytes + self.v_cache.nbytes) / (1024**2)
            layout_str = "gqa-split"
        elif combined:
            # TRTLLM MLA format: single combined cache
            self.kv_cache = torch.zeros(
                num_layers, max_pages, page_size, self.kv_cache_dim,
                dtype=kv_dtype, device=device,
            )
            alloc_mb = self.kv_cache.nbytes / (1024**2)
            layout_str = "mla-combined"
        else:
            # FlashInfer MLA format: split ckv + kpe caches
            self.ckv_cache = torch.zeros(
                num_layers, max_pages, page_size, self.ckv_dim,
                dtype=kv_dtype, device=device,
            )
            self.kpe_cache = torch.zeros(
                num_layers, max_pages, page_size, self.kpe_dim,
                dtype=kv_dtype, device=device,
            )
            alloc_mb = (self.ckv_cache.nbytes + self.kpe_cache.nbytes) / (1024**2)
            layout_str = "mla-split"

        logger.info(
            "KV cache allocated: %d layers × %d pages × %d tokens = %.0f MB (%s, %s)",
            num_layers, max_pages, page_size, alloc_mb,
            self.attention_type, layout_str,
        )

        # Free page tracking
        self._free_pages: List[int] = list(range(max_pages))
        self._free_pages.reverse()  # pop from end

    def _bytes_per_page(self) -> int:
        elem_size = 1 if self.kv_dtype == torch.float8_e4m3fn else 2
        return self.page_size * self.kv_cache_dim * elem_size * self.num_layers

    @property
    def free_page_count(self) -> int:
        return len(self._free_pages)

    def alloc_pages(self, n: int) -> List[int]:
        """Allocate n pages from the free pool."""
        if n > len(self._free_pages):
            raise RuntimeError(
                f"KV cache exhausted: need {n} pages, have {len(self._free_pages)}"
            )
        pages = [self._free_pages.pop() for _ in range(n)]
        return pages

    def free_pages(self, pages: List[int]):
        """Return pages to the free pool."""
        self._free_pages.extend(pages)

    # ── MLA cache access ──

    def get_layer_caches(self, layer_offset: int):
        """Get split cache tensors for MLA (FlashInfer).

        Returns (ckv_cache, kpe_cache) each [max_pages, page_size, dim].
        """
        assert self.attention_type == "mla" and not self.combined
        return self.ckv_cache[layer_offset], self.kpe_cache[layer_offset]

    def get_combined_layer_cache(self, layer_offset: int) -> torch.Tensor:
        """Get combined KV cache for MLA (TRTLLM format)."""
        assert self.attention_type == "mla" and self.combined
        return self.kv_cache[layer_offset].unsqueeze(0)

    # ── GQA cache access ──

    def get_gqa_layer_caches(self, layer_offset: int):
        """Get (k_cache, v_cache) for GQA (FlashInfer standard paged attention).

        Returns (k, v) each [max_pages, page_size, num_kv_heads, head_dim].
        """
        assert self.attention_type == "gqa"
        return self.k_cache[layer_offset], self.v_cache[layer_offset]


class SequenceKVState:
    """KV cache state for a single sequence (request).

    Tracks which pages are allocated and current position.
    Provides FlashInfer-compatible index arrays.
    """

    def __init__(self, cache: PagedKVCache, seq_id: int = 0):
        self.cache = cache
        self.seq_id = seq_id
        self.pages: List[int] = []
        self.seq_len: int = 0  # number of tokens in cache

    def ensure_capacity(self, new_tokens: int):
        """Ensure we have enough pages for new_tokens more tokens."""
        total_needed = self.seq_len + new_tokens
        pages_needed = (total_needed + self.cache.page_size - 1) // self.cache.page_size
        if pages_needed > len(self.pages):
            extra = pages_needed - len(self.pages)
            new_pages = self.cache.alloc_pages(extra)
            self.pages.extend(new_pages)

    def advance(self, num_tokens: int):
        """Record that num_tokens were appended to the cache."""
        self.seq_len += num_tokens

    def free(self):
        """Release all pages back to the pool."""
        if self.pages:
            self.cache.free_pages(self.pages)
            self.pages = []
            self.seq_len = 0

    def kv_indices(self, device: torch.device) -> torch.Tensor:
        """Page indices for FlashInfer: all allocated pages."""
        return torch.tensor(self.pages, dtype=torch.int32, device=device) if self.pages else torch.zeros(0, dtype=torch.int32, device=device)

    def kv_indptr(self, device: torch.device) -> torch.Tensor:
        """Page indptr for FlashInfer (single sequence): [0, num_allocated_pages]."""
        return torch.tensor([0, len(self.pages)], dtype=torch.int32, device=device)

    def kv_len_arr(self, device: torch.device) -> torch.Tensor:
        """Sequence length array: [seq_len]."""
        return torch.tensor([self.seq_len], dtype=torch.int32, device=device)

    def last_page_len(self) -> int:
        """Number of valid tokens in the last page."""
        if self.seq_len == 0:
            return 0
        rem = self.seq_len % self.cache.page_size
        return rem if rem > 0 else self.cache.page_size

    def last_page_len_tensor(self, device: torch.device) -> torch.Tensor:
        """Last page length as tensor (for FlashInfer decode)."""
        return torch.tensor([self.last_page_len()], dtype=torch.int32, device=device)

    def block_tables(self, device: torch.device, pad_to_multiple: int = 8) -> torch.Tensor:
        """Block (page) indices for TRTLLM decode kernel.

        Returns [1, padded_num_blocks] int32.
        """
        num_blocks = len(self.pages)
        constraint = TRTLLM_BLOCK_CONSTRAINT // self.cache.page_size
        padded = math.ceil(num_blocks / constraint) * constraint if num_blocks > 0 else constraint
        table = torch.full((1, padded), -1, dtype=torch.int32, device=device)
        if num_blocks > 0:
            table[0, :num_blocks] = torch.tensor(self.pages, dtype=torch.int32, device=device)
        return table

    def store_kv_combined(
        self,
        layer_offset: int,
        kv_combined: torch.Tensor,
        positions: torch.Tensor,
    ):
        """Store combined KV [M, kv_cache_dim] into the paged cache (TRTLLM MLA)."""
        assert self.cache.combined, "store_kv_combined requires combined cache"
        page_size = self.cache.page_size
        pages_tensor = torch.tensor(self.pages, dtype=torch.long, device=kv_combined.device)

        page_indices = pages_tensor[positions.long() // page_size]
        slots = (positions.long() % page_size)

        self.cache.kv_cache[layer_offset, page_indices, slots] = kv_combined.to(self.cache.kv_dtype)
