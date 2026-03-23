"""TRTLLM MLA attention backend — DISABLED.

MLA (Multi-head Latent Attention) is not currently supported.
This will be re-implemented with native CUDA kernels when needed.
"""


class TRTLLMMLAAttention:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MLA attention is not currently supported. "
            "DeepSeek models using MLA will be re-implemented with "
            "native CUDA kernels in a future release."
        )
