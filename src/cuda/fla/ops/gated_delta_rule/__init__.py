# Vendored subset: only chunked forward path for prefill inference
from .chunk import chunk_gated_delta_rule

__all__ = ["chunk_gated_delta_rule"]
