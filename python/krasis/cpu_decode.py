"""All-CPU decode: runs entire M=1 decode without GPU involvement.

After GPU prefill completes, the CpuDecoder:
1. Copies KV cache from GPU paged format to flat CPU arrays
2. Copies linear attention state (conv + recurrent) from GPU to CPU
3. Copies all attention/norm/embedding/lm_head weights to CPU float32
4. Runs decode steps entirely on CPU: PyTorch CPU ops + Rust MoE engine

This eliminates all GPU-CPU synchronization overhead during decode.
GPU prefill is completely untouched.
"""

import collections
import logging
import math
import os
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from krasis import CpuDecodeStore
from krasis.config import ModelConfig

logger = logging.getLogger(__name__)

_CPU_DECODE_TIMING = os.environ.get("KRASIS_CPU_DECODE_TIMING", "0") == "1"
_TIMING_REPORT_INTERVAL = int(os.environ.get("KRASIS_TIMING_INTERVAL", "20"))
_NO_QUANT = os.environ.get("KRASIS_NO_QUANT", "0") == "1"
_PYTHON_NORMS = os.environ.get("KRASIS_PYTHON_NORMS", "0") == "1"
_PYTHON_DECODE = os.environ.get("KRASIS_PYTHON_DECODE", "0") == "1"


def sample_cpu(logits: torch.Tensor, temperature: float = 0.6,
               top_k: int = 50, top_p: float = 0.95) -> int:
    """Sample next token on CPU. Returns token ID as int."""
    logits = logits.float().squeeze(0)  # [vocab_size]

    if temperature == 0:
        return logits.argmax().item()

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0 and top_k < logits.shape[0]:
        topk_vals, topk_idx = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(0, topk_idx, topk_vals)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=0), dim=0)
        # Remove tokens with cumulative probability above threshold
        remove_mask = cum_probs - F.softmax(sorted_logits, dim=0) >= top_p
        sorted_logits[remove_mask] = float('-inf')
        logits.scatter_(0, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=0)
    return torch.multinomial(probs, 1).item()


class CpuDecoder:
    """All-CPU decode engine.

    Usage:
        decoder = CpuDecoder(model)
        decoder.init_weights()   # Once at model load time
        # ... later, per request:
        decoder.prepare(seq_states)
        for step in range(max_tokens):
            logits = decoder.step(token_id, position)
            token_id = sample_cpu(logits, ...)
    """

    def __init__(self, model):
        """Initialize from a loaded KrasisModel.

        Does NOT copy any data — call init_weights() at load time,
        then prepare() after each prefill.
        """
        self.cfg = model.cfg
        self.engine = model.krasis_engine
        self._model = model

        # Rust decode store for quantized matmuls
        # norm_bias_one=False because weight_loader already pre-corrects norm
        # weights to (1+w) for qwen3_next models. We just need w*x, not (1+w)*x.
        self._store = CpuDecodeStore(
            group_size=128, parallel=True, norm_bias_one=False)
        self._decode_bits = 4  # INT4 for non-MoE weights (3x faster matmul than INT8)

        # Populated by init_weights() — immutable after init
        self._embedding = None        # [vocab_size, hidden] float32
        self._final_norm = None       # [hidden] float32
        self._lm_head_wid = None      # weight ID in store
        self._lm_head_buf = None      # pre-allocated output buffer
        self._layers = []             # per-layer CPU weight dicts
        self._weights_initialized = False

        # GQA KV cache (CPU, flat layout) — pre-allocated once at init, reused
        self._kv_k = {}  # layer_idx -> [max_kv_seq, kv_heads, head_dim] float32
        self._kv_v = {}  # layer_idx -> [max_kv_seq, kv_heads, head_dim] float32
        # MLA KV cache (CPU, flat layout) — pre-allocated once at init, reused
        self._kv_ckv = {}  # layer_idx -> [max_kv_seq, kv_lora_rank] float32
        self._kv_kpe = {}  # layer_idx -> [max_kv_seq, qk_rope_dim] float32

        # Linear attention state (per-layer, CPU float32) — pre-allocated, zeroed per request
        self._la_conv_state = {}    # layer_idx -> [1, conv_dim, kernel_dim]
        self._la_recur_state = {}   # layer_idx -> [1, nv, dk, dv]
        # Templates for resetting state each request (set by init_weights)
        self._la_conv_state_templates = {}   # layer_idx -> (shape, dtype)
        self._la_recur_state_templates = {}  # layer_idx -> (shape, dtype)

        # Max KV sequence length for pre-allocated buffers (set at init)
        self._max_kv_seq = 0

        # RoPE tables (CPU float32) — set by init_weights
        self._rope_cos = None
        self._rope_sin = None
        self._mla_rope_cos = None
        self._mla_rope_sin = None

        # Pre-allocated pinned MoE buffers for Rust engine interface
        hidden = self.cfg.hidden_size
        topk = self.cfg.num_experts_per_tok
        if topk > 0:
            self._moe_act_buf = torch.empty(1, hidden, dtype=torch.bfloat16, pin_memory=True)
            self._moe_ids_buf = torch.empty(1, topk, dtype=torch.int32, pin_memory=True)
            self._moe_wts_buf = torch.empty(1, topk, dtype=torch.float32, pin_memory=True)
            self._moe_out_buf = torch.empty(1, hidden, dtype=torch.bfloat16, pin_memory=True)
        else:
            self._moe_act_buf = None

        self._seq_len = 0
        self._max_seq = 0
        self._prepared = False
        self._step_count = 0
        self._use_rust_decode = False
        self._decode_graph_tensors = []  # prevent GC of tensors passed to Rust

        # Timing instrumentation (enabled via KRASIS_CPU_DECODE_TIMING=1)
        self._timing = _CPU_DECODE_TIMING
        if self._timing:
            self._timings = collections.defaultdict(float)
            self._timing_counts = collections.defaultdict(int)
            self._timing_step_start = 0.0
            self._weight_sizes = {}  # component -> bytes touched per token
            logger.info("CPU decode timing ENABLED (report every %d steps)", _TIMING_REPORT_INTERVAL)

    # ──────────────────────────────────────────────────────
    # Weight copying helpers
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _to_cpu_f32(weight_data):
        """Convert GPU weight (BF16 tensor or INT8 tuple) to CPU float32."""
        if isinstance(weight_data, tuple):
            # INT8: (weight_int8 [N, K], scale [N])
            w, s = weight_data
            return (w.float() * s.unsqueeze(1)).cpu()
        else:
            return weight_data.float().cpu()

    # ──────────────────────────────────────────────────────
    # One-time weight initialization (call at model load)
    # ──────────────────────────────────────────────────────

    def init_weights(self):
        """Copy and quantize all immutable weights from GPU model to CPU.

        Called once at model load time. After this, the GPU is never
        touched during decode. Only KV cache + recurrent state (which
        change per request) are deferred to prepare().
        """
        t0 = time.perf_counter()
        model = self._model
        cfg = self.cfg

        # ── Global weights ──
        self._embedding = model.embedding.float().cpu()
        self._final_norm = model.final_norm.float().cpu().contiguous()
        _lm_head_f32 = self._to_cpu_f32(model.lm_head_data).contiguous()
        if _NO_QUANT:
            self._lm_head_wid = None
            self._lm_head_f32 = _lm_head_f32
            self._lm_head_buf = None
        else:
            self._lm_head_wid = self._store.store_weight_f32(
                _lm_head_f32.data_ptr(), _lm_head_f32.shape[0], _lm_head_f32.shape[1],
                self._decode_bits)
            self._lm_head_buf = torch.empty(_lm_head_f32.shape[0], dtype=torch.float32)
            del _lm_head_f32

        # When streaming attention is enabled, the layer objects' attention
        # attributes point to shared GPU ping-pong buffers (wrong data).
        # Read from model._attn_cpu_weights which has correct per-layer CPU copies.
        use_cpu_weight_store = hasattr(model, '_attn_cpu_weights') and model._attn_cpu_weights

        # ── Per-layer weights ──
        self._layers = []

        for layer_idx in range(cfg.num_hidden_layers):
            layer = model.layers[layer_idx]
            ld = {
                'type': layer.layer_type,
                'is_moe': layer.is_moe,
                'input_norm': layer.input_norm_weight.float().cpu().contiguous(),
                'post_attn_norm': layer.post_attn_norm_weight.float().cpu().contiguous(),
            }

            # ── Attention weights ──
            if use_cpu_weight_store:
                cpu_w = model._attn_cpu_weights[layer_idx]
                self._init_attention_from_cpu_store(layer_idx, layer, ld, cpu_w)
            elif layer.layer_type == "linear_attention":
                self._init_linear_attention(layer_idx, layer, ld)
            elif cfg.is_gqa:
                self._init_gqa(layer_idx, layer, ld)
            elif cfg.is_mla:
                self._init_mla(layer_idx, layer, ld)

            # ── MLP weights ──
            if layer.is_moe:
                if use_cpu_weight_store:
                    self._prepare_moe_from_cpu_store(layer, ld, cpu_w)
                else:
                    self._prepare_moe(layer, ld)
            else:
                if use_cpu_weight_store and 'dense_mlp' in cpu_w:
                    self._prepare_dense_mlp_from_cpu_store(ld, cpu_w)
                elif not layer.is_moe:
                    self._prepare_dense_mlp(layer, ld)

            self._layers.append(ld)

        # ── Quantize attention/MLP weights into Rust store ──
        if _NO_QUANT:
            logger.info("KRASIS_NO_QUANT=1: skipping weight quantization (F.linear fallback)")
        else:
            self._quantize_all_weights()

        # ── Store norm weights in Rust for zero-overhead access ──
        store = self._store
        self._final_norm_id = store.store_norm_weight(
            self._final_norm.data_ptr(), self._final_norm.numel())
        for ld in self._layers:
            for nk in ('input_norm', 'post_attn_norm'):
                nw = ld[nk]
                ld[f'{nk}_id'] = store.store_norm_weight(nw.data_ptr(), nw.numel())

        # ── Pre-compute RoPE tables (max 128K positions) ──
        self._max_rope_seq = 131072
        self._init_rope()

        # ── Pre-allocate KV cache + recurrent state buffers ──
        # Use 32K as default max decode context; actual prompts up to this length
        # are supported without reallocation. Larger prompts trigger reallocation.
        self._preallocate_buffers(max_seq=32768)

        # ── Configure Rust decode graph (single-call decode_step) ──
        self._configure_decode_graph()

        self._weights_initialized = True
        elapsed = time.perf_counter() - t0

        # Memory summary
        total_bytes = 0
        for ld in self._layers:
            for k, v in ld.items():
                if isinstance(v, torch.Tensor):
                    total_bytes += v.nelement() * v.element_size()
                elif isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, torch.Tensor):
                            total_bytes += vv.nelement() * vv.element_size()
        total_bytes += self._embedding.nelement() * 4
        total_bytes += self._final_norm.nelement() * 4
        total_bytes += self._store.total_bytes()  # quantized weights in Rust store

        logger.info("CpuDecoder weights initialized in %.2fs: %.1f GB, "
                     "rust_store=%d weights/%d KB",
                     elapsed, total_bytes / 1e9,
                     self._store.num_weights(), self._store.total_bytes() // 1024)

        # Compute per-component weight sizes for bandwidth analysis
        if self._timing:
            self._compute_weight_sizes()

    # ──────────────────────────────────────────────────────
    # Per-request preparation (call after each GPU prefill)
    # ──────────────────────────────────────────────────────

    def prepare(self, seq_states, max_new_tokens=4096):
        """Prepare per-request state: KV cache copy + recurrent state reset.

        Weights are already on CPU from init_weights(). This only copies
        the request-specific KV cache from GPU and resets recurrent state.
        Does NOT touch GPU weights.

        Args:
            seq_states: List of SequenceKVState (one per GPU split group)
            max_new_tokens: Maximum decode tokens to allocate for
        """
        assert self._weights_initialized, "Call init_weights() first (at model load time)"
        t0 = time.perf_counter()
        model = self._model
        cfg = self.cfg

        # Determine current seq_len from KV state
        self._seq_len = 0
        for ss in seq_states:
            if ss is not None and ss.seq_len > 0:
                self._seq_len = ss.seq_len
                break

        self._max_seq = self._seq_len + max_new_tokens
        logger.info("CpuDecoder.prepare: seq_len=%d, max_seq=%d", self._seq_len, self._max_seq)

        # ── Reallocate if needed (rare — only if request exceeds pre-allocated size) ──
        if self._max_seq > self._max_kv_seq:
            logger.warning("Request needs %d tokens but buffers pre-allocated for %d — reallocating",
                           self._max_seq, self._max_kv_seq)
            self._preallocate_buffers(self._max_seq)

        # ── Zero KV cache and copy from GPU ──
        self._zero_kv_cache()

        self._copy_kv_cache(model, seq_states)

        # ── Copy linear attention state from GPU (not zero!) ──
        self._copy_recurrent_state_from_gpu()

        # ── Update Rust decode state pointers ──
        if self._use_rust_decode:
            self._set_decode_state()

        # Reset step counter and timing
        self._step_count = 0
        if self._timing:
            self._timings = collections.defaultdict(float)
            self._timing_counts = collections.defaultdict(int)

        self._prepared = True
        elapsed = time.perf_counter() - t0
        logger.info("CpuDecoder prepared in %.3fs (KV cache + state reset only)",
                     elapsed)

    def _compute_weight_sizes(self):
        """Compute bytes of weights touched per token for bandwidth analysis."""
        ws = {}
        cfg = self.cfg
        bits = self._decode_bits
        # Quantized weights: size = rows * cols * bits/8 + scales
        # For INT8: rows*cols bytes + rows*(cols/gs)*2 scales
        # For INT4: rows*cols/2 bytes + rows*(cols/gs)*2 scales

        def _qsize(d, key):
            """Bytes for a quantized weight in the Rust store."""
            wid = d.get(f'{key}_wid')
            if wid is not None:
                # Get info from stored buf shape
                buf = d.get(f'{key}_buf')
                if buf is not None:
                    rows = buf.shape[0]
                    # We don't store cols directly, estimate from store
                    return self._store.weight_bytes(wid)
            return 0

        la_proj_bytes = 0
        la_out_proj_bytes = 0
        gqa_proj_bytes = 0
        gqa_o_proj_bytes = 0
        moe_rust_bytes = 0
        se_gate_up_bytes = 0
        se_down_bytes = 0
        lm_head_bytes = 0
        la_state_bytes = 0  # recurrent state + conv state
        la_conv_bytes = 0
        norm_bytes = 0

        n_la = 0
        n_gqa = 0
        n_moe = 0

        for ld in self._layers:
            a = ld.get('attn', {})
            if ld['type'] == 'linear_attention':
                n_la += 1
                la_proj_bytes += _qsize(a, 'in_proj_qkvz') + _qsize(a, 'in_proj_ba')
                la_out_proj_bytes += _qsize(a, 'out_proj')
                # Conv and recurrent state
                for k in ('conv1d_weight', 'norm_weight', 'A_log', 'dt_bias'):
                    t = a.get(k)
                    if isinstance(t, torch.Tensor):
                        la_conv_bytes += t.nelement() * t.element_size()
            elif cfg.is_gqa:
                n_gqa += 1
                for k in ('q_proj', 'k_proj', 'v_proj'):
                    gqa_proj_bytes += _qsize(a, k)
                gqa_o_proj_bytes += _qsize(a, 'o_proj')

            if ld['is_moe'] and 'shared_expert' in ld:
                n_moe += 1
                se = ld['shared_expert']
                se_gate_up_bytes += _qsize(se, 'gate_up_proj')
                se_down_bytes += _qsize(se, 'down_proj')

            # Norms
            for nk in ('input_norm', 'post_attn_norm'):
                t = ld.get(nk)
                if isinstance(t, torch.Tensor):
                    norm_bytes += t.nelement() * t.element_size()

        # Recurrent state per token
        for layer_idx in self._la_recur_state:
            s = self._la_recur_state[layer_idx]
            la_state_bytes += s.nelement() * s.element_size()
        for layer_idx in self._la_conv_state:
            s = self._la_conv_state[layer_idx]
            la_state_bytes += s.nelement() * s.element_size()

        if self._lm_head_wid is not None:
            lm_head_bytes = self._store.weight_bytes(self._lm_head_wid)

        # MoE expert bytes per token from Rust engine
        moe_rust_bytes = self.engine.expert_bytes_per_token() if hasattr(self.engine, 'expert_bytes_per_token') else 0

        # Embedding
        emb_bytes = cfg.hidden_size * 4  # one row lookup

        ws['la_proj'] = la_proj_bytes
        ws['la_out_proj'] = la_out_proj_bytes
        ws['la_state'] = la_state_bytes
        ws['la_conv_misc'] = la_conv_bytes
        ws['gqa_proj'] = gqa_proj_bytes
        ws['gqa_o_proj'] = gqa_o_proj_bytes
        ws['se_gate_up'] = se_gate_up_bytes
        ws['se_down'] = se_down_bytes
        ws['lm_head'] = lm_head_bytes
        ws['norm'] = norm_bytes
        ws['embedding'] = emb_bytes
        ws['moe_rust'] = moe_rust_bytes

        total = sum(ws.values())
        ws['total'] = total

        self._weight_sizes = ws

        logger.info("=== WEIGHT SIZES (bytes touched per token) ===")
        logger.info("  %-20s  %10s  %8s", "component", "bytes", "MB")
        for name in ('la_proj', 'la_out_proj', 'la_state', 'la_conv_misc',
                     'gqa_proj', 'gqa_o_proj', 'se_gate_up', 'se_down',
                     'lm_head', 'norm', 'embedding', 'moe_rust'):
            b = ws.get(name, 0)
            if b > 0:
                logger.info("  %-20s  %10d  %8.2f", name, b, b / 1e6)
        logger.info("  %-20s  %10d  %8.2f", "TOTAL", total, total / 1e6)
        logger.info("  %d LA layers, %d GQA layers, %d MoE layers", n_la, n_gqa, n_moe)

    def _init_linear_attention(self, layer_idx, layer, ld):
        """Copy linear attention weights to CPU (immutable). State templates stored for reset."""
        attn = layer.attention
        # conv1d_weight: squeeze [conv_dim, 1, kernel_dim] -> [conv_dim, kernel_dim] for Rust
        conv_w = attn.conv1d_weight.float().cpu().squeeze(1).contiguous()
        ld['attn'] = {
            'in_proj_qkvz': self._to_cpu_f32(attn.in_proj_qkvz),
            'in_proj_ba': self._to_cpu_f32(attn.in_proj_ba),
            'conv1d_weight': conv_w,
            'out_proj': self._to_cpu_f32(attn.out_proj),
            'A_log': attn.A_log.float().cpu().contiguous(),
            'dt_bias': attn.dt_bias.float().cpu().contiguous(),
            'norm_weight': attn.norm_weight.float().cpu().reshape(-1).contiguous(),
            'num_k_heads': attn.num_k_heads,
            'num_v_heads': attn.num_v_heads,
            'k_head_dim': attn.k_head_dim,
            'v_head_dim': attn.v_head_dim,
            'head_ratio': attn.head_ratio,
            'conv_dim': attn.conv_dim,
            'kernel_dim': attn.kernel_dim,
            'scale': attn.scale,
        }
        # Store state shapes for per-request reset (don't copy actual state yet)
        attn._init_state()
        conv_shape = attn._conv_state.shape   # [1, conv_dim, kernel_dim]
        recur_shape = attn._recurrent_state.shape  # [1, nv, dk, dv]
        self._la_conv_state_templates[layer_idx] = conv_shape
        self._la_recur_state_templates[layer_idx] = recur_shape

    def _init_gqa(self, layer_idx, layer, ld):
        """Copy GQA attention weights to CPU (immutable). KV cache allocated per request."""
        attn = layer.attention
        ld['attn'] = {
            'q_proj': self._to_cpu_f32(attn.q_proj),
            'k_proj': self._to_cpu_f32(attn.k_proj),
            'v_proj': self._to_cpu_f32(attn.v_proj),
            'o_proj': self._to_cpu_f32(attn.o_proj),
            'q_norm': attn.q_norm.float().cpu() if attn.q_norm is not None else None,
            'k_norm': attn.k_norm.float().cpu() if attn.k_norm is not None else None,
            'gated': attn.gated_attention,
            'num_heads': attn.num_heads,
            'num_kv_heads': attn.num_kv_heads,
            'head_dim': attn.head_dim,
            'sm_scale': attn.sm_scale,
            'q_proj_bias': attn.q_proj_bias.float().cpu() if attn.q_proj_bias is not None else None,
            'k_proj_bias': attn.k_proj_bias.float().cpu() if attn.k_proj_bias is not None else None,
            'v_proj_bias': attn.v_proj_bias.float().cpu() if attn.v_proj_bias is not None else None,
            'o_proj_bias': attn.o_proj_bias.float().cpu() if attn.o_proj_bias is not None else None,
        }

    def _init_mla(self, layer_idx, layer, ld):
        """Copy MLA attention weights to CPU (immutable). KV cache allocated per request."""
        attn = layer.attention
        a = {
            'kv_a_proj': self._to_cpu_f32(attn.kv_a_proj),
            'kv_a_norm': attn.kv_a_norm_weight.float().cpu(),
            'o_proj': self._to_cpu_f32(attn.o_proj),
            'w_kc': attn.w_kc.float().cpu(),    # [H, nope, lora_rank]
            'w_vc': attn.w_vc.float().cpu(),    # [H, v_head, lora_rank]
            'sm_scale': attn.sm_scale,
            'num_heads': attn.num_heads,
            'kv_lora_rank': attn.kv_lora_rank,
            'qk_nope_dim': attn.qk_nope_dim,
            'qk_rope_dim': attn.qk_rope_dim,
            'v_head_dim': attn.v_head_dim,
            'head_dim': attn.head_dim,
            'has_q_lora': attn.has_q_lora,
        }
        if attn.has_q_lora:
            a['q_a_proj'] = self._to_cpu_f32(attn.q_a_proj)
            a['q_a_norm'] = attn.q_a_norm_weight.float().cpu()
            a['q_b_proj'] = self._to_cpu_f32(attn.q_b_proj)
        else:
            a['q_proj'] = self._to_cpu_f32(attn.q_proj)

        ld['attn'] = a

    def _prepare_moe(self, layer, ld):
        """Copy MoE routing and shared expert weights."""
        ld['gate_weight'] = layer._gate_weight_f32.cpu()
        if layer._gate_bias_f32 is not None:
            ld['gate_bias'] = layer._gate_bias_f32.cpu()
        if layer._e_score_correction_bias_f32 is not None:
            ld['e_score_corr'] = layer._e_score_correction_bias_f32.cpu()

        # Shared expert
        if layer.shared_expert is not None:
            se = layer.shared_expert
            ld['shared_expert'] = {
                'gate_up_proj': self._to_cpu_f32(se['gate_up_proj']),
                'down_proj': self._to_cpu_f32(se['down_proj']),
            }
            if layer.shared_expert_gate is not None:
                ld['shared_expert']['gate'] = self._to_cpu_f32(layer.shared_expert_gate)

    def _prepare_dense_mlp(self, layer, ld):
        """Copy dense MLP weights."""
        if layer.dense_mlp is not None:
            ld['dense_mlp'] = {
                'gate_proj': self._to_cpu_f32(layer.dense_mlp['gate_proj']),
                'up_proj': self._to_cpu_f32(layer.dense_mlp['up_proj']),
                'down_proj': self._to_cpu_f32(layer.dense_mlp['down_proj']),
            }

    # ──────────────────────────────────────────────────────
    # Prepare from CPU weight store (streaming attention mode)
    # ──────────────────────────────────────────────────────

    def _init_attention_from_cpu_store(self, layer_idx, layer, ld, cpu_w):
        """Copy attention weights from model._attn_cpu_weights (correct per-layer copies).
        Immutable weights only — KV cache and state are handled per request."""
        layer_type = layer.layer_type

        if layer_type == "linear_attention":
            attn_d = cpu_w.get("linear_attention", {})
            conv_w = attn_d['conv1d_weight'].float().cpu()
            if conv_w.dim() == 3:
                conv_w = conv_w.squeeze(1)
            conv_w = conv_w.contiguous()
            ld['attn'] = {
                'in_proj_qkvz': self._to_cpu_f32(attn_d['in_proj_qkvz']),
                'in_proj_ba': self._to_cpu_f32(attn_d['in_proj_ba']),
                'conv1d_weight': conv_w,
                'out_proj': self._to_cpu_f32(attn_d['out_proj']),
                'A_log': attn_d['A_log'].float().cpu().contiguous(),
                'dt_bias': attn_d['dt_bias'].float().cpu().contiguous(),
                'norm_weight': attn_d['norm_weight'].float().cpu().reshape(-1).contiguous(),
                'num_k_heads': layer.attention.num_k_heads,
                'num_v_heads': layer.attention.num_v_heads,
                'k_head_dim': layer.attention.k_head_dim,
                'v_head_dim': layer.attention.v_head_dim,
                'head_ratio': layer.attention.head_ratio,
                'conv_dim': layer.attention.conv_dim,
                'kernel_dim': layer.attention.kernel_dim,
                'scale': layer.attention.scale,
            }
            # Store state shapes for per-request reset
            attn = layer.attention
            attn._init_state()
            self._la_conv_state_templates[layer_idx] = attn._conv_state.shape
            self._la_recur_state_templates[layer_idx] = attn._recurrent_state.shape

        elif self.cfg.is_gqa:
            attn_d = cpu_w.get("attention", {})
            attn = layer.attention
            ld['attn'] = {
                'q_proj': self._to_cpu_f32(attn_d['q_proj']),
                'k_proj': self._to_cpu_f32(attn_d['k_proj']),
                'v_proj': self._to_cpu_f32(attn_d['v_proj']),
                'o_proj': self._to_cpu_f32(attn_d['o_proj']),
                'q_norm': self._to_cpu_f32(attn_d['q_norm']) if 'q_norm' in attn_d else None,
                'k_norm': self._to_cpu_f32(attn_d['k_norm']) if 'k_norm' in attn_d else None,
                'gated': attn.gated_attention,
                'num_heads': attn.num_heads,
                'num_kv_heads': attn.num_kv_heads,
                'head_dim': attn.head_dim,
                'sm_scale': attn.sm_scale,
                'q_proj_bias': self._to_cpu_f32(attn_d['q_proj_bias']) if 'q_proj_bias' in attn_d else None,
                'k_proj_bias': self._to_cpu_f32(attn_d['k_proj_bias']) if 'k_proj_bias' in attn_d else None,
                'v_proj_bias': self._to_cpu_f32(attn_d['v_proj_bias']) if 'v_proj_bias' in attn_d else None,
                'o_proj_bias': self._to_cpu_f32(attn_d['o_proj_bias']) if 'o_proj_bias' in attn_d else None,
            }

        elif self.cfg.is_mla:
            attn_d = cpu_w.get("attention", {})
            attn = layer.attention
            a = {
                'kv_a_proj': self._to_cpu_f32(attn_d['kv_a_proj_with_mqa'] if 'kv_a_proj_with_mqa' in attn_d else attn_d['kv_a_proj']),
                'kv_a_norm': self._to_cpu_f32(attn_d.get('kv_a_layernorm')),
                'o_proj': self._to_cpu_f32(attn_d['o_proj']),
                'w_kc': attn_d['w_kc'].float().cpu(),
                'w_vc': attn_d['w_vc'].float().cpu(),
                'sm_scale': attn.sm_scale,
                'num_heads': attn.num_heads,
                'kv_lora_rank': attn.kv_lora_rank,
                'qk_nope_dim': attn.qk_nope_dim,
                'qk_rope_dim': attn.qk_rope_dim,
                'v_head_dim': attn.v_head_dim,
                'head_dim': attn.head_dim,
                'has_q_lora': attn.has_q_lora,
            }
            if attn.has_q_lora:
                a['q_a_proj'] = self._to_cpu_f32(attn_d['q_a_proj'])
                a['q_a_norm'] = self._to_cpu_f32(attn_d.get('q_a_layernorm'))
                a['q_b_proj'] = self._to_cpu_f32(attn_d['q_b_proj'])
            else:
                a['q_proj'] = self._to_cpu_f32(attn_d['q_proj'])
            ld['attn'] = a

    def _prepare_moe_from_cpu_store(self, layer, ld, cpu_w):
        """Copy MoE routing and shared expert weights from CPU store."""
        # Gate weights are already on GPU (reloaded by _init_stream_attention)
        ld['gate_weight'] = layer._gate_weight_f32.cpu()
        if layer._gate_bias_f32 is not None:
            ld['gate_bias'] = layer._gate_bias_f32.cpu()
        if layer._e_score_correction_bias_f32 is not None:
            ld['e_score_corr'] = layer._e_score_correction_bias_f32.cpu()

        # Shared expert from CPU store
        if 'shared_expert' in cpu_w and cpu_w['shared_expert']:
            se_cpu = cpu_w['shared_expert']
            # CPU store has separate gate_proj/up_proj; fuse them for CPU decode
            gp = self._to_cpu_f32(se_cpu.get('gate_proj'))
            up = self._to_cpu_f32(se_cpu.get('up_proj'))
            if gp is not None and up is not None:
                se = {
                    'gate_up_proj': torch.cat([gp, up], dim=0),
                    'down_proj': self._to_cpu_f32(se_cpu['down_proj']),
                }
            else:
                gate_up = se_cpu.get('gate_up_proj')
                se = {
                    'gate_up_proj': self._to_cpu_f32(gate_up),
                    'down_proj': self._to_cpu_f32(se_cpu['down_proj']),
                }
            if 'shared_expert_gate' in se_cpu:
                se['gate'] = self._to_cpu_f32(se_cpu['shared_expert_gate'])
            ld['shared_expert'] = se

    def _prepare_dense_mlp_from_cpu_store(self, ld, cpu_w):
        """Copy dense MLP weights from CPU store."""
        d = cpu_w['dense_mlp']
        ld['dense_mlp'] = {
            'gate_proj': self._to_cpu_f32(d['gate_proj']),
            'up_proj': self._to_cpu_f32(d['up_proj']),
            'down_proj': self._to_cpu_f32(d['down_proj']),
        }

    # ──────────────────────────────────────────────────────
    # Weight quantization into Rust decode store
    # ──────────────────────────────────────────────────────

    def _quantize_all_weights(self):
        """Quantize all matmul weights into the Rust CpuDecodeStore."""
        bits = self._decode_bits
        gs = 128  # must match store's group_size

        for ld in self._layers:
            a = ld.get('attn', {})

            if ld['type'] == 'linear_attention':
                self._qw(a, 'in_proj_qkvz', bits, gs)
                self._qw(a, 'in_proj_ba', bits, gs)
                self._qw(a, 'out_proj', bits, gs)
            elif self.cfg.is_gqa:
                for key in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                    self._qw(a, key, bits, gs)
            elif self.cfg.is_mla:
                self._qw(a, 'kv_a_proj', bits, gs)
                self._qw(a, 'o_proj', bits, gs)
                if a.get('has_q_lora'):
                    self._qw(a, 'q_a_proj', bits, gs)
                    self._qw(a, 'q_b_proj', bits, gs)
                else:
                    self._qw(a, 'q_proj', bits, gs)

            if 'shared_expert' in ld:
                se = ld['shared_expert']
                self._qw(se, 'gate_up_proj', bits, gs)
                self._qw(se, 'down_proj', bits, gs)
                # Quantize shared_expert_gate if present (ensure 2D)
                if 'gate' in se and isinstance(se['gate'], torch.Tensor):
                    g = se['gate']
                    if g.dim() == 1:
                        se['gate'] = g.unsqueeze(0)
                    self._qw(se, 'gate', bits, gs)

            if 'dense_mlp' in ld:
                d = ld['dense_mlp']
                self._qw(d, 'gate_proj', bits, gs)
                self._qw(d, 'up_proj', bits, gs)
                self._qw(d, 'down_proj', bits, gs)

            # Store MoE routing weights as float32 in Rust
            if 'gate_weight' in ld:
                gw = ld['gate_weight'].contiguous()
                ne, hd = gw.shape
                bias_ptr = None
                bias_len = 0
                esc_ptr = None
                esc_len = 0
                if 'gate_bias' in ld:
                    gb = ld['gate_bias'].contiguous()
                    bias_ptr = gb.data_ptr()
                    bias_len = gb.numel()
                    ld['_gate_bias_tensor'] = gb  # prevent GC
                if 'e_score_corr' in ld:
                    ec = ld['e_score_corr'].contiguous()
                    esc_ptr = ec.data_ptr()
                    esc_len = ec.numel()
                    ld['_e_score_corr_tensor'] = ec  # prevent GC
                rid = self._store.store_route_weight(
                    gw.data_ptr(), ne, hd,
                    bias_ptr, bias_len, esc_ptr, esc_len)
                ld['_route_id'] = rid
                # Free Python copies (data now lives in Rust)
                ld.pop('gate_weight', None)
                ld.pop('gate_bias', None)
                ld.pop('e_score_corr', None)
                ld.pop('_gate_bias_tensor', None)
                ld.pop('_e_score_corr_tensor', None)

        logger.info("Quantized %d weights + %d route weights into Rust store (%d KB INT%d)",
                     self._store.num_weights(), self._store.num_route_weights(),
                     self._store.total_bytes() // 1024, bits)

    def _qw(self, d, key, bits, gs):
        """Quantize one weight matrix: d[key] → d[key+'_wid'] + d[key+'_buf']."""
        w = d.get(key)
        if w is None or not isinstance(w, torch.Tensor) or w.dim() != 2:
            return
        w = w.contiguous()
        rows, cols = w.shape
        if cols % gs != 0:
            logger.debug("Skipping %s: cols %d not divisible by %d", key, cols, gs)
            return
        if bits == 4 and cols % 8 != 0:
            logger.debug("Skipping %s: cols %d not divisible by 8", key, cols)
            return
        wid = self._store.store_weight_f32(w.data_ptr(), rows, cols, bits)
        buf = torch.empty(rows, dtype=torch.float32)
        d[f'{key}_wid'] = wid
        d[f'{key}_buf'] = buf
        d[key] = None  # free f32 weight

    def _qlinear(self, d, key, x):
        """Quantized linear: if quantized weight exists, use Rust; else F.linear."""
        wid = d.get(f'{key}_wid')
        if wid is not None:
            buf = d[f'{key}_buf']
            self._store.matmul(wid, x.data_ptr(), buf.data_ptr())
            return buf.unsqueeze(0)  # [1, N]
        else:
            return F.linear(x, d[key])

    def _qlinear_batch(self, d, keys, x):
        """Batch quantized linear: quantize input once, run multiple matmuls."""
        wids = []
        bufs = []
        fallback_keys = []
        for key in keys:
            wid = d.get(f'{key}_wid')
            if wid is not None:
                wids.append(wid)
                bufs.append(d[f'{key}_buf'])
            else:
                fallback_keys.append(key)

        results = {}
        if wids:
            self._store.matmul_batch(
                wids, x.data_ptr(), [b.data_ptr() for b in bufs])
            idx = 0
            for key in keys:
                if d.get(f'{key}_wid') is not None:
                    results[key] = bufs[idx].unsqueeze(0)
                    idx += 1

        for key in fallback_keys:
            results[key] = F.linear(x, d[key])

        return [results[k] for k in keys]

    # ──────────────────────────────────────────────────────
    # Per-request KV cache allocation + recurrent state reset
    # ──────────────────────────────────────────────────────

    def _zero_kv_cache(self):
        """Zero pre-allocated KV cache buffers for new request."""
        for t in self._kv_k.values():
            t.zero_()
        for t in self._kv_v.values():
            t.zero_()
        for t in self._kv_ckv.values():
            t.zero_()
        for t in self._kv_kpe.values():
            t.zero_()

    def _copy_recurrent_state_from_gpu(self):
        """Copy linear attention recurrent + conv state from GPU after prefill.

        GPU prefill builds up recurrent state in each layer's attention object.
        We must copy this to CPU for decode, not zero it — otherwise 36 of 48
        layers lose all prompt context.
        """
        model = self._model
        copied = 0
        for layer_idx in self._la_recur_state:
            attn = model.layers[layer_idx].attention
            if attn._recurrent_state is not None:
                self._la_recur_state[layer_idx].copy_(
                    attn._recurrent_state.float().cpu())
                copied += 1
            else:
                self._la_recur_state[layer_idx].zero_()

        for layer_idx in self._la_conv_state:
            attn = model.layers[layer_idx].attention
            if attn._conv_state is not None:
                self._la_conv_state[layer_idx].copy_(
                    attn._conv_state.float().cpu())
            else:
                self._la_conv_state[layer_idx].zero_()

        logger.info("Copied recurrent state from GPU: %d/%d layers",
                     copied, len(self._la_recur_state))

    # ──────────────────────────────────────────────────────
    # KV cache copy (GPU paged → CPU flat)
    # ──────────────────────────────────────────────────────

    def _copy_kv_cache(self, model, seq_states):
        """Unpage GPU KV cache into flat CPU float32 arrays."""
        for gpu_idx, (start, end) in enumerate(model._layer_split):
            kv_cache = model.kv_caches[gpu_idx]
            if kv_cache is None:
                continue
            ss = seq_states[gpu_idx]
            if ss is None or ss.seq_len == 0:
                continue

            seq_len = ss.seq_len
            page_size = kv_cache.page_size
            page_indices = ss.pages

            for abs_layer in range(start, end):
                kv_offset = model._kv_layer_offsets.get(abs_layer, -1)
                if kv_offset < 0:
                    continue

                if self.cfg.is_gqa and abs_layer in self._kv_k:
                    # GQA: separate K, V caches
                    k_layer = kv_cache.k_cache[kv_offset]
                    v_layer = kv_cache.v_cache[kv_offset]

                    token_idx = 0
                    for page_idx in page_indices:
                        tokens_in_page = min(page_size, seq_len - token_idx)
                        if tokens_in_page <= 0:
                            break
                        self._kv_k[abs_layer][token_idx:token_idx + tokens_in_page] = \
                            k_layer[page_idx, :tokens_in_page].float().cpu()
                        self._kv_v[abs_layer][token_idx:token_idx + tokens_in_page] = \
                            v_layer[page_idx, :tokens_in_page].float().cpu()
                        token_idx += tokens_in_page

                elif self.cfg.is_mla and abs_layer in self._kv_ckv:
                    # MLA: ckv + kpe caches
                    if kv_cache.combined:
                        kv_layer = kv_cache.kv_cache[kv_offset]
                        lora_rank = self.cfg.kv_lora_rank
                        token_idx = 0
                        for page_idx in page_indices:
                            tokens_in_page = min(page_size, seq_len - token_idx)
                            if tokens_in_page <= 0:
                                break
                            data = kv_layer[page_idx, :tokens_in_page].float().cpu()
                            self._kv_ckv[abs_layer][token_idx:token_idx + tokens_in_page] = data[:, :lora_rank]
                            self._kv_kpe[abs_layer][token_idx:token_idx + tokens_in_page] = data[:, lora_rank:]
                            token_idx += tokens_in_page
                    else:
                        ckv_layer = kv_cache.ckv_cache[kv_offset]
                        kpe_layer = kv_cache.kpe_cache[kv_offset]
                        token_idx = 0
                        for page_idx in page_indices:
                            tokens_in_page = min(page_size, seq_len - token_idx)
                            if tokens_in_page <= 0:
                                break
                            self._kv_ckv[abs_layer][token_idx:token_idx + tokens_in_page] = \
                                ckv_layer[page_idx, :tokens_in_page].float().cpu()
                            self._kv_kpe[abs_layer][token_idx:token_idx + tokens_in_page] = \
                                kpe_layer[page_idx, :tokens_in_page].float().cpu()
                            token_idx += tokens_in_page

        logger.info("Copied KV cache to CPU: %d positions", self._seq_len)

    # ──────────────────────────────────────────────────────
    # RoPE initialization
    # ──────────────────────────────────────────────────────

    def _init_rope(self):
        """Pre-compute RoPE cos/sin tables for CPU decode.
        Uses self._max_rope_seq (set at init_weights time) for table size."""
        cfg = self.cfg
        max_pos = self._max_rope_seq

        if cfg.is_gqa:
            dim = cfg.rotary_dim
            freqs = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(max_pos, dtype=torch.float32)
            freqs = torch.outer(t, freqs)
            self._rope_cos = freqs.cos()
            self._rope_sin = freqs.sin()

        if cfg.is_mla:
            dim = cfg.qk_rope_head_dim
            theta = cfg.rope_theta
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

            # YaRN frequency interpolation
            rope_cfg = cfg.rope_scaling
            if rope_cfg:
                factor = rope_cfg.get("factor", 1.0)
                if factor > 1.0:
                    original_max = rope_cfg.get("original_max_position_embeddings", 4096)
                    beta_fast = rope_cfg.get("beta_fast", 32.0)
                    beta_slow = rope_cfg.get("beta_slow", 1.0)

                    low = max(0, math.floor(dim * math.log(original_max / (beta_fast * 2 * math.pi))
                                            / (2 * math.log(theta))))
                    high = min(dim // 2 - 1,
                               math.ceil(dim * math.log(original_max / (beta_slow * 2 * math.pi))
                                          / (2 * math.log(theta))))

                    freq_extra = freqs.clone()
                    freq_inter = freqs / factor
                    ramp = torch.clamp(
                        (torch.arange(dim // 2).float() - low) / max(high - low, 0.001), 0, 1)
                    inv_mask = 1.0 - ramp
                    freqs = freq_inter * (1 - inv_mask) + freq_extra * inv_mask

            t = torch.arange(max_pos, dtype=torch.float32)
            freqs = torch.outer(t, freqs)
            self._mla_rope_cos = freqs.cos()
            self._mla_rope_sin = freqs.sin()

    # ──────────────────────────────────────────────────────
    # Pre-allocation of KV cache + recurrent state
    # ──────────────────────────────────────────────────────

    def _preallocate_buffers(self, max_seq: int = 32768):
        """Pre-allocate all per-request buffers at init time.

        KV cache, recurrent state, and conv state are allocated once and
        reused across requests. Only zeroing/copying happens per request.
        If a request exceeds max_seq, buffers are reallocated (rare).
        """
        cfg = self.cfg
        model = self._model
        self._max_kv_seq = max_seq
        kv_bytes = 0

        # ── KV cache for GQA/MLA layers ──
        for layer_idx, ld in enumerate(self._layers):
            kv_offset = model._kv_layer_offsets.get(layer_idx, -1)
            if kv_offset < 0:
                continue

            a = ld.get('attn', {})
            layer_type = ld['type']

            if layer_type in ('full_attention', 'sliding_attention'):
                num_kv_heads = a.get('num_kv_heads', 0)
                head_dim = a.get('head_dim', 0)
                if num_kv_heads > 0 and head_dim > 0:
                    self._kv_k[layer_idx] = torch.zeros(
                        max_seq, num_kv_heads, head_dim, dtype=torch.float32)
                    self._kv_v[layer_idx] = torch.zeros(
                        max_seq, num_kv_heads, head_dim, dtype=torch.float32)
                    kv_bytes += 2 * max_seq * num_kv_heads * head_dim * 4

            kv_lora_rank = a.get('kv_lora_rank', 0)
            if kv_lora_rank > 0:
                qk_rope_dim = a.get('qk_rope_dim', 0)
                self._kv_ckv[layer_idx] = torch.zeros(
                    max_seq, kv_lora_rank, dtype=torch.float32)
                self._kv_kpe[layer_idx] = torch.zeros(
                    max_seq, qk_rope_dim, dtype=torch.float32)
                kv_bytes += max_seq * (kv_lora_rank + qk_rope_dim) * 4

        # ── Recurrent + conv state for linear attention layers ──
        state_bytes = 0
        for layer_idx, shape in self._la_conv_state_templates.items():
            self._la_conv_state[layer_idx] = torch.zeros(shape, dtype=torch.float32)
            state_bytes += torch.zeros(shape).nelement() * 4

        for layer_idx, shape in self._la_recur_state_templates.items():
            self._la_recur_state[layer_idx] = torch.zeros(shape, dtype=torch.float32)
            state_bytes += torch.zeros(shape).nelement() * 4

        logger.info("Pre-allocated buffers: KV cache %.1f MB (%d layers, max_seq=%d), "
                     "recurrent state %.1f MB (%d layers)",
                     kv_bytes / 1e6, len(self._kv_k) + len(self._kv_ckv), max_seq,
                     state_bytes / 1e6, len(self._la_recur_state))

    # ──────────────────────────────────────────────────────
    # Rust decode graph configuration
    # ──────────────────────────────────────────────────────

    def _configure_decode_graph(self):
        """Configure the Rust decode graph for single-call decode_step.

        Only supported for models with linear_attention or GQA (not MLA).
        Falls back to Python path for unsupported models or when disabled.
        """
        cfg = self.cfg

        if _PYTHON_DECODE:
            logger.info("KRASIS_PYTHON_DECODE=1: using Python decode path")
            self._use_rust_decode = False
            return

        if cfg.is_mla or _NO_QUANT or self._lm_head_wid is None:
            self._use_rust_decode = False
            return

        store = self._store

        # scoring_func: 0=sigmoid, 1=softmax, 2=swiglu
        topk = cfg.num_experts_per_tok if cfg.num_experts_per_tok > 0 else 0
        if topk > 0:
            if hasattr(cfg, 'swiglu_limit') and cfg.swiglu_limit > 0:
                sf = 2
            elif cfg.scoring_func == "sigmoid":
                sf = 0
            else:
                sf = 1
        else:
            sf = 0
        rsf = getattr(cfg, 'routed_scaling_factor', 1.0)
        ntp = getattr(cfg, 'norm_topk_prob', False)

        store.configure_decode(
            cfg.hidden_size, cfg.num_hidden_layers,
            cfg.rms_norm_eps, self._final_norm_id, self._lm_head_wid,
            cfg.vocab_size, topk, sf, ntp, rsf,
            self._embedding.data_ptr())

        first_k = cfg.first_k_dense_replace

        for layer_idx, ld in enumerate(self._layers):
            a = ld.get('attn', {})

            if ld['type'] == 'linear_attention':
                # Expand norm_weight [dv] → [nv*dv] if needed
                nw = a['norm_weight']
                nv = a['num_v_heads']
                dv = a['v_head_dim']
                if nw.numel() == dv:
                    nw_expanded = nw.repeat(nv).contiguous()
                else:
                    nw_expanded = nw.contiguous()
                self._decode_graph_tensors.append(nw_expanded)

                store.add_decode_la_layer(
                    ld['input_norm_id'], ld['post_attn_norm_id'],
                    a['in_proj_qkvz_wid'], a['in_proj_ba_wid'], a['out_proj_wid'],
                    a['conv1d_weight'].data_ptr(),
                    a['A_log'].data_ptr(), a['dt_bias'].data_ptr(),
                    nw_expanded.data_ptr(),
                    a['num_k_heads'], a['num_v_heads'],
                    a['k_head_dim'], a['v_head_dim'],
                    a['head_ratio'], a['kernel_dim'], a['scale'])

            elif cfg.is_gqa:
                q_norm = a.get('q_norm')
                k_norm = a.get('k_norm')
                q_norm_ptr = q_norm.data_ptr() if q_norm is not None else 0
                q_norm_len = q_norm.numel() if q_norm is not None else 0
                k_norm_ptr = k_norm.data_ptr() if k_norm is not None else 0
                k_norm_len = k_norm.numel() if k_norm is not None else 0
                if q_norm is not None:
                    self._decode_graph_tensors.append(q_norm.contiguous())
                if k_norm is not None:
                    self._decode_graph_tensors.append(k_norm.contiguous())

                store.add_decode_gqa_layer(
                    ld['input_norm_id'], ld['post_attn_norm_id'],
                    a['q_proj_wid'], a['k_proj_wid'],
                    a['v_proj_wid'], a['o_proj_wid'],
                    q_norm_ptr, q_norm_len, k_norm_ptr, k_norm_len,
                    a['gated'], a['num_heads'], a['num_kv_heads'],
                    a['head_dim'], a['sm_scale'])
            else:
                logger.warning("Unsupported attention type for decode graph at layer %d", layer_idx)
                self._use_rust_decode = False
                return

            # MLP config
            moe_layer_idx = layer_idx - first_k if layer_idx >= first_k else None

            if ld['is_moe'] and '_route_id' in ld:
                se = ld.get('shared_expert', {})
                sgu_wid = se.get('gate_up_proj_wid')
                sd_wid = se.get('down_proj_wid')
                sg_wid = se.get('gate_wid')
                store.set_decode_layer_moe(
                    layer_idx, ld['_route_id'],
                    moe_layer_idx if moe_layer_idx is not None else 0,
                    sgu_wid, sd_wid, sg_wid)
            elif 'dense_mlp' in ld:
                d = ld['dense_mlp']
                store.set_decode_layer_dense(
                    layer_idx,
                    d['gate_proj_wid'], d['up_proj_wid'], d['down_proj_wid'])

        # RoPE (for GQA layers)
        if self._rope_cos is not None:
            store.set_decode_rope(
                self._rope_cos.data_ptr(), self._rope_sin.data_ptr(),
                self._rope_cos.shape[-1], self._rope_cos.shape[0])

        # MoE store
        if topk > 0 and self.engine is not None:
            store.set_moe_store(self.engine)

        store.finalize_decode()
        self._use_rust_decode = True
        logger.info("Rust decode graph configured: %d layers, topk=%d",
                     cfg.num_hidden_layers, topk)

    def _set_decode_state(self):
        """Update Rust decode state pointers after prepare()."""
        cfg = self.cfg
        num_layers = cfg.num_hidden_layers
        kv_k_ptrs = []
        kv_v_ptrs = []
        conv_state_ptrs = []
        recur_state_ptrs = []

        for layer_idx in range(num_layers):
            if layer_idx in self._kv_k:
                kv_k_ptrs.append(self._kv_k[layer_idx].data_ptr())
                kv_v_ptrs.append(self._kv_v[layer_idx].data_ptr())
            else:
                kv_k_ptrs.append(0)
                kv_v_ptrs.append(0)

            if layer_idx in self._la_conv_state:
                cs = self._la_conv_state[layer_idx]
                if cs.dim() == 3:
                    cs = cs.squeeze(0)
                conv_state_ptrs.append(cs.data_ptr())
            else:
                conv_state_ptrs.append(0)

            if layer_idx in self._la_recur_state:
                rs = self._la_recur_state[layer_idx]
                if rs.dim() == 4:
                    rs = rs.squeeze(0)
                recur_state_ptrs.append(rs.data_ptr())
            else:
                recur_state_ptrs.append(0)

        self._store.set_decode_state(
            self._seq_len, self._max_kv_seq,
            kv_k_ptrs, kv_v_ptrs,
            conv_state_ptrs, recur_state_ptrs)

    def _step_rust(self, token_id: int, position: int) -> torch.Tensor:
        """Rust single-call decode step."""
        _t = self._timing
        if _t:
            _t0 = time.perf_counter()

        self._store.decode_step(token_id, position, self._lm_head_buf.data_ptr())
        logits = self._lm_head_buf.unsqueeze(0)

        self._step_count += 1

        if _t:
            total_step = time.perf_counter() - _t0
            self._timings['total'] += total_step
            self._timing_counts['steps'] += 1
            if self._timing_counts['steps'] % _TIMING_REPORT_INTERVAL == 0:
                n = self._timing_counts['steps']
                avg = self._timings['total'] / n * 1000
                logger.info("=== CPU DECODE (Rust, %d steps, avg %.1f ms/tok, %.2f tok/s) ===",
                            n, avg, 1000.0 / avg if avg > 0 else 0)

        return logits

    # ──────────────────────────────────────────────────────
    # Core decode step
    # ──────────────────────────────────────────────────────

    def step(self, token_id: int, position: int) -> torch.Tensor:
        """One full decode step on CPU.

        Args:
            token_id: Token ID to decode
            position: Position in the sequence

        Returns:
            logits: [1, vocab_size] float32 CPU tensor
        """
        assert self._prepared, "Call prepare() first"

        # ── Rust single-call decode path ──
        if self._use_rust_decode:
            return self._step_rust(token_id, position)

        # ── Python fallback path ──
        cfg = self.cfg
        _t = self._timing

        if _t:
            self._timing_step_start = time.perf_counter()

        # Embedding lookup
        if _t:
            _t0 = time.perf_counter()
        hidden = self._embedding[token_id].clone()  # [hidden_size] flat f32
        if hidden.dim() == 2:
            hidden = hidden.squeeze(0)
        if _t:
            self._timings['embedding'] += time.perf_counter() - _t0

        # Pre-allocated residual buffer (flat [hidden_size])
        if not hasattr(self, '_residual_buf') or self._residual_buf is None:
            self._residual_buf = torch.zeros(cfg.hidden_size, dtype=torch.float32)
        residual = self._residual_buf
        first_residual = True
        use_rust_norms = not _PYTHON_NORMS
        first_k = cfg.first_k_dense_replace
        eps = cfg.rms_norm_eps
        h_size = cfg.hidden_size
        store = self._store

        _debug = self._step_count < 3  # Log first 3 steps

        for layer_idx in range(cfg.num_hidden_layers):
            if _t:
                _layer_start = time.perf_counter()
            ld = self._layers[layer_idx]
            moe_layer_idx = layer_idx - first_k if layer_idx >= first_k else None

            # ── Pre-attention norm ──
            if _t:
                _t0 = time.perf_counter()
            if use_rust_norms:
                store.fused_add_rmsnorm_id(
                    hidden.data_ptr(), residual.data_ptr(),
                    ld['input_norm_id'], eps, h_size, first_residual)
            else:
                if first_residual:
                    residual.copy_(hidden)
                else:
                    residual.add_(hidden)
                variance = residual.pow(2).mean() + eps
                rms = torch.rsqrt(variance)
                # norm weights are pre-corrected by weight_loader (already 1+w)
                hidden.copy_(residual * rms * ld['input_norm'])
            first_residual = False
            if _t:
                self._timings['norm'] += time.perf_counter() - _t0

            # ── Attention ──
            if _t:
                _t0 = time.perf_counter()
            if ld['type'] == 'linear_attention':
                attn_out = self._linear_attention_step(layer_idx, ld, hidden)
                if _t:
                    self._timings['attn_linear'] += time.perf_counter() - _t0
            elif cfg.is_mla:
                attn_out = self._mla_attention_step(layer_idx, ld, hidden, position)
                if _t:
                    self._timings['attn_mla'] += time.perf_counter() - _t0
            else:
                attn_out = self._gqa_attention_step(layer_idx, ld, hidden, position)
                if _t:
                    self._timings['attn_gqa'] += time.perf_counter() - _t0

            if _debug and layer_idx < 4:
                h_norm = hidden.norm().item()
                a_norm = attn_out.norm().item() if attn_out.dim() >= 1 else 0
                r_norm = residual.norm().item()
                has_nan = torch.isnan(attn_out).any().item()
                logger.info("STEP %d L%d/%s: hidden=%.4f attn=%.4f res=%.4f nan=%s",
                            self._step_count, layer_idx, ld['type'][:3],
                            h_norm, a_norm, r_norm, has_nan)

            # ── Post-attention reshape + norm ──
            if _t:
                _t0 = time.perf_counter()
            # attn_out may be [1, hidden] from attention — flatten
            if attn_out.dim() == 2:
                attn_out = attn_out.squeeze(0)
            hidden = attn_out.contiguous()
            if use_rust_norms:
                store.fused_add_rmsnorm_id(
                    hidden.data_ptr(), residual.data_ptr(),
                    ld['post_attn_norm_id'], eps, h_size, False)
            else:
                residual.add_(hidden)
                variance = residual.pow(2).mean() + eps
                rms = torch.rsqrt(variance)
                hidden.copy_(residual * rms * ld['post_attn_norm'])
            if _t:
                self._timings['norm'] += time.perf_counter() - _t0

            # ── MLP ──
            if _t:
                _t0 = time.perf_counter()
            if ld['is_moe']:
                hidden = self._moe_forward(ld, hidden, moe_layer_idx)
                if hidden.dim() == 2:
                    hidden = hidden.squeeze(0)
                if _t:
                    self._timings['moe'] += time.perf_counter() - _t0
            elif 'dense_mlp' in ld:
                hidden = self._dense_mlp_forward(ld, hidden)
                if hidden.dim() == 2:
                    hidden = hidden.squeeze(0)
                if _t:
                    self._timings['dense_mlp'] += time.perf_counter() - _t0

            if _debug and layer_idx < 4:
                m_norm = hidden.norm().item()
                logger.info("STEP %d L%d MoE-out: %.4f", self._step_count, layer_idx, m_norm)

            if _t:
                self._timings['layer_total'] += time.perf_counter() - _layer_start

        # ── Final norm + LM head ──
        if _t:
            _t0 = time.perf_counter()
        if use_rust_norms:
            store.fused_add_rmsnorm_id(
                hidden.data_ptr(), residual.data_ptr(),
                self._final_norm_id, eps, h_size, False)
        else:
            residual.add_(hidden)
            variance = residual.pow(2).mean() + eps
            rms = torch.rsqrt(variance)
            hidden.copy_(residual * rms * self._final_norm)
        if _t:
            self._timings['norm'] += time.perf_counter() - _t0
            _t0 = time.perf_counter()
        if self._lm_head_wid is not None:
            store.matmul(self._lm_head_wid, hidden.data_ptr(), self._lm_head_buf.data_ptr())
            logits = self._lm_head_buf.unsqueeze(0)  # [1, vocab_size]
        else:
            logits = F.linear(hidden, self._lm_head_f32).unsqueeze(0)  # [1, vocab_size]
        if _t:
            self._timings['lm_head'] += time.perf_counter() - _t0

        if _debug:
            top_vals, top_ids = logits[0].topk(5)
            logger.info("STEP %d logits: top5 ids=%s vals=%s norm=%.4f nan=%s",
                        self._step_count, top_ids.tolist(), top_vals.tolist(),
                        logits.norm().item(), torch.isnan(logits).any().item())

        self._step_count += 1

        # Timing report
        if _t:
            total_step = time.perf_counter() - self._timing_step_start
            self._timings['total'] += total_step
            self._timing_counts['steps'] += 1
            if self._timing_counts['steps'] % _TIMING_REPORT_INTERVAL == 0:
                self._report_timing()

        return logits

    def _report_timing(self):
        """Print accumulated timing breakdown."""
        n = self._timing_counts['steps']
        t = self._timings
        total = t['total']
        avg = total / n * 1000  # ms per step

        # Top-level components
        parts = [
            ('attn_linear', t.get('attn_linear', 0)),
            ('attn_gqa', t.get('attn_gqa', 0)),
            ('attn_mla', t.get('attn_mla', 0)),
            ('moe', t.get('moe', 0)),
            ('dense_mlp', t.get('dense_mlp', 0)),
            ('norm', t.get('norm', 0)),
            ('embedding', t.get('embedding', 0)),
            ('lm_head', t.get('lm_head', 0)),
        ]

        # Sub-breakdowns
        la_parts = [
            ('  la_proj', t.get('la_proj', 0)),
            ('  la_conv', t.get('la_conv', 0)),
            ('  la_recurrent', t.get('la_recurrent', 0)),
            ('  la_gate_norm', t.get('la_gate_norm', 0)),
            ('  la_out_proj', t.get('la_out_proj', 0)),
        ]
        gqa_parts = [
            ('  gqa_proj', t.get('gqa_proj', 0)),
            ('  gqa_rope', t.get('gqa_rope', 0)),
            ('  gqa_kv_write', t.get('gqa_kv_write', 0)),
            ('  gqa_attn_compute', t.get('gqa_attn_compute', 0)),
            ('  gqa_o_proj', t.get('gqa_o_proj', 0)),
        ]
        moe_parts = [
            ('  moe_routing', t.get('moe_routing', 0)),
            ('  moe_rust', t.get('moe_rust', 0)),
            ('  moe_shared', t.get('moe_shared', 0)),
        ]

        logger.info("=== CPU DECODE TIMING (%d steps, avg %.1f ms/tok, %.2f tok/s) ===",
                     n, avg, 1000.0 / avg if avg > 0 else 0)
        for name, val in parts:
            if val > 0:
                pct = val / total * 100
                per_step = val / n * 1000
                logger.info("  %-16s %7.1f ms/tok  %5.1f%%", name, per_step, pct)
                # Show sub-breakdowns under parent
                if name == 'attn_linear':
                    for sn, sv in la_parts:
                        if sv > 0:
                            logger.info("    %-14s %7.1f ms/tok  %5.1f%%",
                                        sn, sv / n * 1000, sv / total * 100)
                elif name == 'attn_gqa':
                    for sn, sv in gqa_parts:
                        if sv > 0:
                            logger.info("    %-14s %7.1f ms/tok  %5.1f%%",
                                        sn, sv / n * 1000, sv / total * 100)
                elif name == 'moe':
                    for sn, sv in moe_parts:
                        if sv > 0:
                            logger.info("    %-14s %7.1f ms/tok  %5.1f%%",
                                        sn, sv / n * 1000, sv / total * 100)
        accounted = sum(v for _, v in parts)
        unaccounted = total - accounted
        if unaccounted > 0.001:
            logger.info("  %-16s %7.1f ms/tok  %5.1f%%", "other",
                        unaccounted / n * 1000, unaccounted / total * 100)

        # Layer loop breakdown: shows Python loop overhead
        layer_total = t.get('layer_total', 0)
        if layer_total > 0:
            loop_overhead = total - layer_total - t.get('embedding', 0) - t.get('lm_head', 0) - t.get('norm', 0) * (1.0 / (self.cfg.num_hidden_layers * 2 + 1))  # approx final norm
            logger.info("  --- layer_total=%.1f ms, step_total=%.1f ms, overhead=%.1f ms (%.1f%%)",
                        layer_total / n * 1000, total / n * 1000,
                        (total - layer_total) / n * 1000 if total > layer_total else 0,
                        (total - layer_total) / total * 100 if total > layer_total else 0)

        # Bandwidth analysis
        ws = getattr(self, '_weight_sizes', {})
        if ws:
            total_bytes = ws.get('total', 0)
            if total_bytes > 0 and avg > 0:
                bw_gbps = (total_bytes / 1e9) / (avg / 1000.0)
                logger.info("  --- Bandwidth: %.2f GB/tok, %.1f GB/s effective (at %.1f ms/tok)",
                            total_bytes / 1e9, bw_gbps, avg)

    # ──────────────────────────────────────────────────────
    # Norm operations
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _fused_add_rmsnorm(hidden, residual, weight, eps):
        """Fused add + RMSNorm on CPU.

        If residual is None: residual = hidden, hidden = rmsnorm(residual)
        Else: residual += hidden, hidden = rmsnorm(residual)
        """
        if residual is None:
            residual = hidden
        else:
            residual = residual + hidden

        # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
        variance = residual.pow(2).mean(dim=-1, keepdim=True)
        hidden = residual * torch.rsqrt(variance + eps) * weight

        return hidden, residual

    @staticmethod
    def _rmsnorm(x, weight, eps):
        """Simple RMSNorm."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * weight

    # ──────────────────────────────────────────────────────
    # GQA Attention (Qwen3-Next GQA layers, or pure GQA models)
    # ──────────────────────────────────────────────────────

    def _gqa_attention_step(self, layer_idx, ld, hidden, position):
        """GQA attention decode step (M=1)."""
        a = ld['attn']
        num_heads = a['num_heads']
        num_kv_heads = a['num_kv_heads']
        head_dim = a['head_dim']
        _t = self._timing

        # Q/K/V projections (Rust batch matmul — quantize input once)
        if _t:
            _t0 = time.perf_counter()
        h_in = hidden if hidden.dim() == 1 else hidden.squeeze(0)
        q_raw, k, v = self._qlinear_batch(a, ['q_proj', 'k_proj', 'v_proj'], h_in)

        # Gated attention: split q into query + gate
        if a['gated']:
            q_raw = q_raw.view(1, num_heads, head_dim * 2)
            q, attn_gate = q_raw.chunk(2, dim=-1)
            attn_gate = attn_gate.reshape(1, num_heads * head_dim)
        else:
            q = q_raw
            attn_gate = None

        # Biases
        if not a['gated'] and a['q_proj_bias'] is not None:
            q = q + a['q_proj_bias']
        if a['k_proj_bias'] is not None:
            k = k + a['k_proj_bias']
        if a['v_proj_bias'] is not None:
            v = v + a['v_proj_bias']

        # Reshape to heads
        if not a['gated']:
            q = q.reshape(1, num_heads, head_dim)
        k = k.reshape(1, num_kv_heads, head_dim)
        v = v.reshape(1, num_kv_heads, head_dim)

        # QKNorm
        if a['q_norm'] is not None:
            q = self._rmsnorm(q, a['q_norm'], self.cfg.rms_norm_eps)
        if a['k_norm'] is not None:
            k = self._rmsnorm(k, a['k_norm'], self.cfg.rms_norm_eps)
        if _t:
            self._timings['gqa_proj'] += time.perf_counter() - _t0

        # RoPE
        if _t:
            _t0 = time.perf_counter()
        q, k = self._apply_rope_gqa(q, k, position)
        if _t:
            self._timings['gqa_rope'] += time.perf_counter() - _t0

        # Append to KV cache
        if _t:
            _t0 = time.perf_counter()
        self._kv_k[layer_idx][position] = k[0]
        self._kv_v[layer_idx][position] = v[0]
        if _t:
            self._timings['gqa_kv_write'] += time.perf_counter() - _t0

        # Attention: Q · K^T → softmax → @ V
        if _t:
            _t0 = time.perf_counter()
        seq_len = position + 1
        k_cache = self._kv_k[layer_idx][:seq_len]  # [S, kv_heads, head_dim]
        v_cache = self._kv_v[layer_idx][:seq_len]

        # Expand KV heads to match Q heads (grouped query)
        num_groups = num_heads // num_kv_heads
        # k_cache: [S, kv_heads, dim] → [S, kv_heads, groups, dim] → [S, num_heads, dim]
        k_expanded = k_cache.unsqueeze(2).expand(-1, -1, num_groups, -1)
        k_expanded = k_expanded.reshape(seq_len, num_heads, head_dim)
        v_expanded = v_cache.unsqueeze(2).expand(-1, -1, num_groups, -1)
        v_expanded = v_expanded.reshape(seq_len, num_heads, head_dim)

        # Scores: [heads, dim] @ [S, heads, dim]^T → [heads, S]
        q_t = q.squeeze(0)  # [heads, dim]
        scores = torch.einsum('hd,shd->hs', q_t, k_expanded) * a['sm_scale']

        # Softmax
        weights = F.softmax(scores, dim=-1)

        # Weighted sum: [heads, S] @ [S, heads, dim] → [heads, dim]
        attn_out = torch.einsum('hs,shd->hd', weights, v_expanded)
        if _t:
            self._timings['gqa_attn_compute'] += time.perf_counter() - _t0

        # Reshape and optional gating
        if _t:
            _t0 = time.perf_counter()
        attn_flat = attn_out.reshape(1, num_heads * head_dim)
        if attn_gate is not None:
            attn_flat = attn_flat * torch.sigmoid(attn_gate)

        # O projection (Rust matmul)
        o_in = attn_flat.squeeze(0) if attn_flat.dim() == 2 else attn_flat
        output = self._qlinear(a, 'o_proj', o_in)
        if a['o_proj_bias'] is not None:
            output = output + a['o_proj_bias']
        if _t:
            self._timings['gqa_o_proj'] += time.perf_counter() - _t0

        return output

    def _apply_rope_gqa(self, q, k, position):
        """Apply RoPE at a single position for GQA."""
        cos = self._rope_cos[position]  # [rotary_dim//2]
        sin = self._rope_sin[position]

        rotary_dim = self.cfg.rotary_dim
        d2 = rotary_dim // 2

        def rotate(x):
            x1, x2 = x[..., :d2], x[..., d2:2 * d2]
            rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
            if 2 * d2 < x.shape[-1]:
                rotated = torch.cat([rotated, x[..., 2 * d2:]], dim=-1)
            return rotated

        return rotate(q), rotate(k)

    # ──────────────────────────────────────────────────────
    # Linear Attention (Gated DeltaNet, Qwen3-Next)
    # ──────────────────────────────────────────────────────

    def _init_la_conv_buffers(self, a):
        """Pre-allocate reusable buffers for linear attention conv (once per model)."""
        nv = a['num_v_heads']
        dk = a['k_head_dim']
        dv = a['v_head_dim']
        a['_q_buf'] = torch.empty(nv * dk, dtype=torch.float32)
        a['_k_buf'] = torch.empty(nv * dk, dtype=torch.float32)
        a['_v_buf'] = torch.empty(nv * dv, dtype=torch.float32)
        a['_z_buf'] = torch.empty(nv * dv, dtype=torch.float32)
        a['_g_buf'] = torch.empty(nv, dtype=torch.float32)
        a['_beta_buf'] = torch.empty(nv, dtype=torch.float32)

    def _linear_attention_step(self, layer_idx, ld, hidden):
        """Gated DeltaNet recurrent decode step (M=1)."""
        a = ld['attn']
        nk = a['num_k_heads']
        nv = a['num_v_heads']
        dk = a['k_head_dim']
        dv = a['v_head_dim']
        hr = a['head_ratio']
        kernel_dim = a['kernel_dim']
        _t = self._timing
        store = self._store

        # ── Projections (Rust batch matmul) ──
        if _t:
            _t0 = time.perf_counter()
        h_in = hidden if hidden.dim() == 1 else hidden.squeeze(0)
        qkvz, ba = self._qlinear_batch(a, ['in_proj_qkvz', 'in_proj_ba'], h_in)
        if _t:
            self._timings['la_proj'] += time.perf_counter() - _t0

        # ── Un-interleave + Conv + Gate params (Rust) ──
        if _t:
            _t0 = time.perf_counter()

        # Lazily allocate conv buffers
        if '_q_buf' not in a:
            self._init_la_conv_buffers(a)

        q_buf = a['_q_buf']
        k_buf = a['_k_buf']
        v_buf = a['_v_buf']
        z_buf = a['_z_buf']
        g_buf = a['_g_buf']
        beta_buf = a['_beta_buf']

        # conv_state is [1, conv_dim, kernel_dim] — squeeze to [conv_dim, kernel_dim] for Rust
        conv_state = self._la_conv_state[layer_idx]
        if conv_state.dim() == 3:
            conv_state = conv_state.squeeze(0)
        conv_state = conv_state.contiguous()

        qkvz_flat = qkvz.squeeze(0).contiguous() if qkvz.dim() == 2 else qkvz.contiguous()
        ba_flat = ba.squeeze(0).contiguous() if ba.dim() == 2 else ba.contiguous()

        store.linear_attention_conv(
            qkvz_flat.data_ptr(), ba_flat.data_ptr(),
            conv_state.data_ptr(), a['conv1d_weight'].data_ptr(),
            a['A_log'].data_ptr(), a['dt_bias'].data_ptr(),
            a['scale'],
            q_buf.data_ptr(), k_buf.data_ptr(),
            v_buf.data_ptr(), z_buf.data_ptr(),
            g_buf.data_ptr(), beta_buf.data_ptr(),
            nk, nv, dk, dv, hr, kernel_dim)

        # Update conv state (modified in-place by Rust)
        self._la_conv_state[layer_idx] = conv_state.unsqueeze(0)

        if _t:
            self._timings['la_conv'] += time.perf_counter() - _t0

        # ── Recurrent state update + output (Rust) ──
        if _t:
            _t0 = time.perf_counter()
        state = self._la_recur_state[layer_idx]  # [1, nv, dk, dv]

        # State is [1, nv, dk, dv] — squeeze to [nv, dk, dv] for Rust
        state_3d = state.squeeze(0).contiguous()

        # Pre-allocate output buffer
        if not hasattr(self, '_la_recur_out') or self._la_recur_out is None:
            self._la_recur_out = torch.empty(nv * dv, dtype=torch.float32)
        recur_out = self._la_recur_out

        store.linear_attention_recurrent(
            state_3d.data_ptr(), q_buf.data_ptr(), k_buf.data_ptr(),
            v_buf.data_ptr(), g_buf.data_ptr(), beta_buf.data_ptr(),
            recur_out.data_ptr(), nv, dk, dv)

        self._la_recur_state[layer_idx] = state_3d.unsqueeze(0)
        if _t:
            self._timings['la_recurrent'] += time.perf_counter() - _t0

        # ── Gated RMSNorm + SiLU (Rust) ──
        if _t:
            _t0 = time.perf_counter()

        if not hasattr(self, '_la_gated_out') or self._la_gated_out is None:
            self._la_gated_out = torch.empty(nv * dv, dtype=torch.float32)
        gated_out = self._la_gated_out

        # norm_weight is [dv] shared across all heads — expand to [nv*dv] once
        nw_key = '_norm_expanded'
        if nw_key not in a:
            nw = a['norm_weight']
            if nw.numel() == dv:
                a[nw_key] = nw.repeat(nv).contiguous()
            else:
                a[nw_key] = nw.contiguous()
        store.gated_rmsnorm_silu(
            recur_out.data_ptr(), z_buf.data_ptr(),
            a[nw_key].data_ptr(),
            gated_out.data_ptr(), self.cfg.rms_norm_eps, nv, dv)

        if self._step_count < 3 and layer_idx < 2:
            logger.info("STEP %d L%d LA debug: recur_out=%.4f/max%.4f gated=%.4f/max%.4f z=%.4f/max%.4f nw=%.4f state=%.4f",
                        self._step_count, layer_idx,
                        recur_out.norm().item(), recur_out.abs().max().item(),
                        gated_out.norm().item(), gated_out.abs().max().item(),
                        z_buf.norm().item(), z_buf.abs().max().item(),
                        a['norm_weight'].norm().item(), state_3d.norm().item())

        if _t:
            self._timings['la_gate_norm'] += time.perf_counter() - _t0

        # ── Output projection (Rust matmul) ──
        if _t:
            _t0 = time.perf_counter()
        result = self._qlinear(a, 'out_proj', gated_out)
        if _t:
            self._timings['la_out_proj'] += time.perf_counter() - _t0
        return result

    # ──────────────────────────────────────────────────────
    # MLA Attention (DeepSeek-V2-Lite)
    # ──────────────────────────────────────────────────────

    def _mla_attention_step(self, layer_idx, ld, hidden, position):
        """MLA attention decode step (M=1)."""
        a = ld['attn']
        H = a['num_heads']
        kv_lora_rank = a['kv_lora_rank']
        qk_nope_dim = a['qk_nope_dim']
        qk_rope_dim = a['qk_rope_dim']
        v_head_dim = a['v_head_dim']
        head_dim = a['head_dim']

        # ── KV compressed projection (Rust matmul) ──
        h_in = hidden if hidden.dim() == 1 else hidden.squeeze(0)
        kv_out = self._qlinear(a, 'kv_a_proj', h_in)
        kv_compressed = kv_out[:, :kv_lora_rank]     # [1, lora_rank]
        k_pe = kv_out[:, kv_lora_rank:]              # [1, rope_dim]

        # KV LayerNorm
        kv_compressed = self._rmsnorm(kv_compressed, a['kv_a_norm'], self.cfg.rms_norm_eps)

        # ── Q projection (Rust matmul) ──
        if a['has_q_lora']:
            q_compressed = self._qlinear(a, 'q_a_proj', h_in)
            q_compressed = self._rmsnorm(q_compressed, a['q_a_norm'], self.cfg.rms_norm_eps)
            qc_in = q_compressed.squeeze(0) if q_compressed.dim() == 2 else q_compressed
            q_full = self._qlinear(a, 'q_b_proj', qc_in)
        else:
            q_full = self._qlinear(a, 'q_proj', h_in)

        q_full = q_full.reshape(1, H, head_dim)
        q_nope = q_full[:, :, :qk_nope_dim]   # [1, H, 128]
        q_pe = q_full[:, :, qk_nope_dim:]      # [1, H, 64]

        # ── RoPE (MLA-style with de-interleave) ──
        k_pe_heads = k_pe.unsqueeze(1)  # [1, 1, 64]
        q_pe, k_pe_heads = self._apply_rope_mla(q_pe, k_pe_heads, position)
        k_pe = k_pe_heads.squeeze(1)  # [1, 64]

        # ── Absorb w_kc into query ──
        q_nope_absorbed = torch.einsum('mhi,hid->mhd', q_nope, a['w_kc'])  # [1, H, lora_rank]

        # ── Append to KV cache ──
        self._kv_ckv[layer_idx][position] = kv_compressed[0]
        self._kv_kpe[layer_idx][position] = k_pe[0]

        # ── Attention ──
        seq_len = position + 1
        ckv_cache = self._kv_ckv[layer_idx][:seq_len]  # [S, lora_rank]
        kpe_cache = self._kv_kpe[layer_idx][:seq_len]   # [S, rope_dim]

        # Combined query: [H, lora_rank + rope_dim]
        q_combined = torch.cat([q_nope_absorbed[0], q_pe[0]], dim=-1)  # [H, lora_rank+rope_dim]

        # Combined KV: [S, lora_rank + rope_dim]
        kv_combined = torch.cat([ckv_cache, kpe_cache], dim=-1)

        # Scores: [H, dim] @ [S, dim]^T → [H, S]
        scores = torch.mm(q_combined, kv_combined.t()) * a['sm_scale']

        # Softmax
        weights = F.softmax(scores, dim=-1)  # [H, S]

        # Output: [H, S] @ [S, lora_rank] → [H, lora_rank]
        output = torch.mm(weights, ckv_cache)

        # Apply w_vc: [H, lora_rank] → [H, v_head_dim]
        attn_final = torch.einsum('hd,hvd->hv', output, a['w_vc'])

        # O projection (Rust matmul)
        attn_flat = attn_final.reshape(H * v_head_dim)
        return self._qlinear(a, 'o_proj', attn_flat)

    def _apply_rope_mla(self, q_pe, k_pe, position):
        """Apply RoPE for MLA attention with de-interleave."""
        q_pe = self._deinterleave(q_pe)
        k_pe = self._deinterleave(k_pe)

        cos = self._mla_rope_cos[position]
        sin = self._mla_rope_sin[position]
        d2 = cos.shape[-1]

        def rotate(x):
            c, s = cos, sin
            while c.dim() < x.dim():
                c = c.unsqueeze(0)
                s = s.unsqueeze(0)
            x1, x2 = x[..., :d2], x[..., d2:]
            return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

        return rotate(q_pe), rotate(k_pe)

    @staticmethod
    def _deinterleave(x):
        """De-interleave [re0, im0, re1, im1, ...] → [re0, re1, ..., im0, im1, ...]."""
        d = x.shape[-1]
        return x.view(*x.shape[:-1], d // 2, 2).transpose(-1, -2).reshape(x.shape)

    # ──────────────────────────────────────────────────────
    # MoE forward (CPU routing + Rust experts + CPU shared expert)
    # ──────────────────────────────────────────────────────

    def _moe_forward(self, ld, hidden, moe_layer_idx):
        """MoE: Rust routing + Rust engine experts + Rust shared expert."""
        cfg = self.cfg
        topk = cfg.num_experts_per_tok
        _t = self._timing

        # ── Routing (Rust) ──
        if _t:
            _t0 = time.perf_counter()
        h_flat = hidden if hidden.dim() == 1 else hidden.squeeze(0)
        h_2d = hidden.unsqueeze(0) if hidden.dim() == 1 else hidden

        route_id = ld.get('_route_id')
        if route_id is not None:
            # Rust routing path
            # Pre-allocate routing output buffers (reuse across layers)
            if not hasattr(self, '_route_ids_buf') or self._route_ids_buf is None:
                self._route_ids_buf = torch.empty(topk, dtype=torch.int32)
                self._route_wts_buf = torch.empty(topk, dtype=torch.float32)

            # scoring_func: 0=sigmoid, 1=softmax, 2=swiglu
            if cfg.swiglu_limit > 0:
                sf = 2
            elif cfg.scoring_func == "sigmoid":
                sf = 0
            else:
                sf = 1

            self._store.moe_route(
                route_id, h_flat.data_ptr(),
                self._route_ids_buf.data_ptr(),
                self._route_wts_buf.data_ptr(),
                topk, sf, cfg.norm_topk_prob)

            topk_ids = self._route_ids_buf.unsqueeze(0)    # [1, topk]
            topk_weights = self._route_wts_buf.unsqueeze(0)  # [1, topk]
        else:
            # Python fallback (only if store_route_weight wasn't called)
            router_logits = torch.matmul(h_2d, ld['gate_weight'].t())
            if 'gate_bias' in ld:
                router_logits = router_logits + ld['gate_bias']

            if cfg.swiglu_limit > 0:
                topk_weights, topk_ids = torch.topk(router_logits, topk, dim=-1)
                topk_weights = F.softmax(topk_weights, dim=-1)
            elif cfg.scoring_func == "sigmoid":
                scores = torch.sigmoid(router_logits)
                if 'e_score_corr' in ld:
                    topk_weights, topk_ids = torch.topk(
                        scores + ld['e_score_corr'], topk, dim=-1)
                    topk_weights = scores.gather(1, topk_ids)
                else:
                    topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)
                if cfg.norm_topk_prob:
                    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            else:
                scores = F.softmax(router_logits, dim=-1)
                if 'e_score_corr' in ld:
                    topk_weights, topk_ids = torch.topk(
                        scores + ld['e_score_corr'], topk, dim=-1)
                    topk_weights = scores.gather(1, topk_ids)
                else:
                    topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)
                if cfg.norm_topk_prob:
                    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if _t:
            self._timings['moe_routing'] += time.perf_counter() - _t0

        # ── Routed experts via Rust engine ──
        if _t:
            _t0 = time.perf_counter()
        self._moe_act_buf.copy_(h_2d.bfloat16())
        self._moe_ids_buf.copy_(topk_ids.to(torch.int32))
        self._moe_wts_buf.copy_(topk_weights.float())

        self.engine.forward_moe_direct(
            moe_layer_idx,
            self._moe_act_buf.data_ptr(),
            self._moe_ids_buf.data_ptr(),
            self._moe_wts_buf.data_ptr(),
            self._moe_out_buf.data_ptr(),
            1, topk,
        )

        moe_out = self._moe_out_buf.float()  # [1, hidden]
        if _t:
            self._timings['moe_rust'] += time.perf_counter() - _t0

        # Routed scaling factor
        if cfg.routed_scaling_factor != 1.0:
            moe_out = moe_out * cfg.routed_scaling_factor

        # ── Shared expert (fused Rust: gate_up → SiLU → down) ──
        if 'shared_expert' in ld:
            if _t:
                _t0 = time.perf_counter()
            se = ld['shared_expert']
            h_flat = hidden if hidden.dim() == 1 else hidden.squeeze(0)
            gu_wid = se.get('gate_up_proj_wid')
            dn_wid = se.get('down_proj_wid')

            if gu_wid is not None and dn_wid is not None:
                # Fused path: single Rust call for gate_up → silu → down
                dn_buf = se['down_proj_buf']
                self._store.fused_shared_expert(
                    gu_wid, dn_wid, h_flat.data_ptr(), dn_buf.data_ptr())
                shared_out = dn_buf.unsqueeze(0)
            else:
                # Fallback: separate calls
                gate_up = self._qlinear(se, 'gate_up_proj', h_flat)
                gate_up_flat = gate_up.squeeze(0) if gate_up.dim() == 2 else gate_up
                intermediate = gate_up_flat.shape[-1] // 2
                se_hidden = torch.empty(intermediate, dtype=torch.float32)
                self._store.silu_mul(
                    gate_up_flat[:intermediate].data_ptr(),
                    gate_up_flat[intermediate:].data_ptr(),
                    se_hidden.data_ptr(), intermediate)
                shared_out = self._qlinear(se, 'down_proj', se_hidden)

            # Sigmoid gate (Qwen3-Next shared_expert_gate)
            if 'gate' in se:
                gate_value = torch.sigmoid(F.linear(h_flat.unsqueeze(0), se['gate']))
                shared_out = gate_value * shared_out

            moe_out = moe_out + shared_out
            if _t:
                self._timings['moe_shared'] += time.perf_counter() - _t0

        return moe_out

    # ──────────────────────────────────────────────────────
    # Dense MLP
    # ──────────────────────────────────────────────────────

    def _dense_mlp_forward(self, ld, hidden):
        """Dense MLP: gate+up → SiLU → down (Rust matmul)."""
        d = ld['dense_mlp']
        h_flat = hidden if hidden.dim() == 1 else hidden.squeeze(0)
        gate, up = self._qlinear_batch(d, ['gate_proj', 'up_proj'], h_flat)
        gate_flat = gate.squeeze(0) if gate.dim() == 2 else gate
        up_flat = up.squeeze(0) if up.dim() == 2 else up
        intermediate = gate_flat.shape[0]
        mlp_hidden = torch.empty(intermediate, dtype=torch.float32)
        self._store.silu_mul(
            gate_flat.data_ptr(), up_flat.data_ptr(),
            mlp_hidden.data_ptr(), intermediate)
        return self._qlinear(d, 'down_proj', mlp_hidden)
