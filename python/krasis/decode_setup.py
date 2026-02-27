"""Decode setup and orchestration.

Handles one-time weight initialization (GPU→CPU copy + quantization into Rust store)
and per-request preparation (KV cache copy, recurrent state reset). All actual decode
compute happens in the Rust engine via generate_batch() or generate_stream().

GPU prefill is completely untouched by this module.
"""

import logging
import os
import time
from typing import List, Optional, Tuple

import torch

from krasis import CpuDecodeStore
from krasis.config import ModelConfig

logger = logging.getLogger(__name__)

class CpuDecoder:
    """Decode setup and orchestration engine.

    Usage:
        decoder = CpuDecoder(model)
        decoder.init_weights()   # Once at model load time
        # ... later, per request:
        decoder.prepare(seq_states)
        decoder._store.generate_batch(...)  # Batch decode (zero Python per token)
        decoder._store.generate_stream(...)  # Streaming decode (Rust HTTP server)
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

        self._layers = []             # per-layer CPU weight dicts
        self._weights_initialized = False

        # GQA KV cache (CPU, flat layout) — pre-allocated once at init, reused
        self._kv_k = {}  # layer_idx -> [max_kv_seq, kv_heads, head_dim] float32
        self._kv_v = {}  # layer_idx -> [max_kv_seq, kv_heads, head_dim] float32

        # MLA KV cache (CPU, flat layout) — compressed KV + rope position embeddings
        self._mla_ckv = {}  # layer_idx -> [max_kv_seq, kv_lora_rank] float32
        self._mla_kpe = {}  # layer_idx -> [max_kv_seq, qk_rope_dim] float32

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
        self._decode_graph_tensors = []  # prevent GC of tensors passed to Rust

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
        self._lm_head_wid = self._store.store_weight_f32(
            _lm_head_f32.data_ptr(), _lm_head_f32.shape[0], _lm_head_f32.shape[1],
            self._decode_bits)
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
            elif cfg.is_mla:
                self._init_mla(layer_idx, layer, ld)
            elif cfg.is_gqa:
                self._init_gqa(layer_idx, layer, ld)

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
        self._set_decode_state()

        self._prepared = True
        elapsed = time.perf_counter() - t0
        logger.info("CpuDecoder prepared in %.3fs (KV cache + state reset only)",
                     elapsed)

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
            'o_proj': self._to_cpu_f32(attn.o_proj),
            'kv_a_norm': attn.kv_a_norm_weight.float().cpu().contiguous(),
            'w_kc': attn.w_kc.to(torch.bfloat16).cpu().contiguous(),
            'w_vc': attn.w_vc.to(torch.bfloat16).cpu().contiguous(),
            'num_heads': attn.num_heads,
            'kv_lora_rank': attn.kv_lora_rank,
            'qk_nope_dim': attn.qk_nope_dim,
            'qk_rope_dim': attn.qk_rope_dim,
            'v_head_dim': attn.v_head_dim,
            'sm_scale': attn.sm_scale,
        }
        if attn.has_q_lora:
            a['q_a_proj'] = self._to_cpu_f32(attn.q_a_proj)
            a['q_b_proj'] = self._to_cpu_f32(attn.q_b_proj)
            a['q_a_norm'] = attn.q_a_norm_weight.float().cpu().contiguous()
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

        elif self.cfg.is_mla:
            attn_d = cpu_w.get("attention", {})
            attn = layer.attention
            a = {
                'kv_a_proj': self._to_cpu_f32(attn_d['kv_a_proj_with_mqa']),
                'o_proj': self._to_cpu_f32(attn_d['o_proj']),
                'kv_a_norm': attn_d['kv_a_layernorm'].float().cpu().contiguous(),
                'w_kc': attn_d['w_kc'].to(torch.bfloat16).cpu().contiguous(),
                'w_vc': attn_d['w_vc'].to(torch.bfloat16).cpu().contiguous(),
                'num_heads': attn.num_heads,
                'kv_lora_rank': attn.kv_lora_rank,
                'qk_nope_dim': attn.qk_nope_dim,
                'qk_rope_dim': attn.qk_rope_dim,
                'v_head_dim': attn.v_head_dim,
                'sm_scale': attn.sm_scale,
            }
            if attn.has_q_lora:
                a['q_a_proj'] = self._to_cpu_f32(attn_d['q_a_proj'])
                a['q_b_proj'] = self._to_cpu_f32(attn_d['q_b_proj'])
                a['q_a_norm'] = attn_d['q_a_layernorm'].float().cpu().contiguous()
            else:
                a['q_proj'] = self._to_cpu_f32(attn_d['q_proj'])
            ld['attn'] = a

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
            elif self.cfg.is_mla:
                self._qw(a, 'kv_a_proj', bits, gs)
                self._qw(a, 'o_proj', bits, gs)
                if 'q_a_proj' in a:
                    self._qw(a, 'q_a_proj', bits, gs)
                    self._qw(a, 'q_b_proj', bits, gs)
                else:
                    self._qw(a, 'q_proj', bits, gs)
            elif self.cfg.is_gqa:
                for key in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                    self._qw(a, key, bits, gs)

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
        """Quantize one weight matrix: d[key] -> d[key+'_wid'] + d[key+'_buf']."""
        w = d.get(key)
        if w is None or not isinstance(w, torch.Tensor) or w.dim() != 2:
            return
        w = w.contiguous()
        rows, cols = w.shape
        # Pad cols to be divisible by group_size (and by 8 for INT4)
        align = gs if bits != 4 else max(gs, 8)
        if cols % align != 0:
            new_cols = ((cols + align - 1) // align) * align
            logger.debug("Padding %s cols %d -> %d for alignment", key, cols, new_cols)
            padded = torch.zeros(rows, new_cols, dtype=w.dtype)
            padded[:, :cols] = w
            w = padded.contiguous()
            cols = new_cols
        wid = self._store.store_weight_f32(w.data_ptr(), rows, cols, bits)
        buf = torch.empty(rows, dtype=torch.float32)
        d[f'{key}_wid'] = wid
        d[f'{key}_buf'] = buf
        d[key] = None  # free f32 weight

    # ──────────────────────────────────────────────────────
    # Per-request KV cache allocation + recurrent state reset
    # ──────────────────────────────────────────────────────

    def _zero_kv_cache(self):
        """Zero pre-allocated KV cache buffers for new request."""
        with torch.inference_mode():
            for t in self._kv_k.values():
                t.zero_()
            for t in self._kv_v.values():
                t.zero_()
            for t in self._mla_ckv.values():
                t.zero_()
            for t in self._mla_kpe.values():
                t.zero_()

    def _copy_recurrent_state_from_gpu(self):
        """Copy linear attention recurrent + conv state from GPU after prefill.

        GPU prefill builds up recurrent state in each layer's attention object.
        We must copy this to CPU for decode, not zero it — otherwise 36 of 48
        layers lose all prompt context.
        """
        model = self._model
        copied = 0
        with torch.inference_mode():
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
    # KV cache copy (GPU paged -> CPU flat)
    # ──────────────────────────────────────────────────────

    def _copy_kv_cache(self, model, seq_states):
        """Unpage GPU KV cache into flat CPU FP8 arrays (zero-conversion copy)."""
        with torch.inference_mode():
            self._copy_kv_cache_inner(model, seq_states)

    def _copy_kv_cache_inner(self, model, seq_states):
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

                if self.cfg.is_mla and abs_layer in self._mla_ckv:
                    # MLA: compressed KV + rope position embeddings
                    # GPU stores FP8 E4M3 -> copy as raw bytes (view as uint8)
                    ckv_layer, kpe_layer = kv_cache.get_layer_caches(kv_offset)

                    token_idx = 0
                    for page_idx in page_indices:
                        tokens_in_page = min(page_size, seq_len - token_idx)
                        if tokens_in_page <= 0:
                            break
                        self._mla_ckv[abs_layer][token_idx:token_idx + tokens_in_page] = \
                            ckv_layer[page_idx, :tokens_in_page].view(torch.uint8).cpu()
                        self._mla_kpe[abs_layer][token_idx:token_idx + tokens_in_page] = \
                            kpe_layer[page_idx, :tokens_in_page].view(torch.uint8).cpu()
                        token_idx += tokens_in_page

                elif self.cfg.is_gqa and abs_layer in self._kv_k:
                    # GQA: separate K, V caches
                    # GPU stores FP8 E4M3 -> copy as raw bytes (view as uint8)
                    k_layer = kv_cache.k_cache[kv_offset]
                    v_layer = kv_cache.v_cache[kv_offset]

                    token_idx = 0
                    for page_idx in page_indices:
                        tokens_in_page = min(page_size, seq_len - token_idx)
                        if tokens_in_page <= 0:
                            break
                        self._kv_k[abs_layer][token_idx:token_idx + tokens_in_page] = \
                            k_layer[page_idx, :tokens_in_page].view(torch.uint8).cpu()
                        self._kv_v[abs_layer][token_idx:token_idx + tokens_in_page] = \
                            v_layer[page_idx, :tokens_in_page].view(torch.uint8).cpu()
                        token_idx += tokens_in_page

        logger.info("Copied KV cache to CPU (FP8): %d positions", self._seq_len)

    # ──────────────────────────────────────────────────────
    # RoPE initialization
    # ──────────────────────────────────────────────────────

    def _init_rope(self):
        """Pre-compute RoPE cos/sin tables for CPU decode."""
        import math
        cfg = self.cfg
        max_pos = self._max_rope_seq

        if cfg.is_mla:
            # MLA YaRN RoPE — same logic as attention.py MLAAttention._get_rope_cos_sin
            dim = cfg.qk_rope_head_dim
            freqs = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2).float() / dim))

            rope_cfg = cfg.rope_scaling
            if rope_cfg:
                factor = rope_cfg.get("factor", 1.0)
                if factor > 1.0:
                    original_max = rope_cfg.get("original_max_position_embeddings", 4096)
                    beta_fast = rope_cfg.get("beta_fast", 32.0)
                    beta_slow = rope_cfg.get("beta_slow", 1.0)

                    low = max(0, math.floor(dim * math.log(original_max / (beta_fast * 2 * math.pi))
                                            / (2 * math.log(cfg.rope_theta))))
                    high = min(dim // 2 - 1, math.ceil(dim * math.log(original_max / (beta_slow * 2 * math.pi))
                                                        / (2 * math.log(cfg.rope_theta))))
                    freq_extra = freqs.clone()
                    freq_inter = freqs / factor
                    ramp = torch.clamp(
                        (torch.arange(dim // 2).float() - low) / max(high - low, 0.001), 0, 1)
                    inv_freq_mask = 1.0 - ramp
                    freqs = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

            t = torch.arange(max_pos, dtype=torch.float32)
            freqs = torch.outer(t, freqs)
            self._mla_rope_cos = freqs.cos().contiguous()
            self._mla_rope_sin = freqs.sin().contiguous()

        if cfg.is_gqa:
            dim = cfg.rotary_dim
            freqs = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(max_pos, dtype=torch.float32)
            freqs = torch.outer(t, freqs)
            self._rope_cos = freqs.cos()
            self._rope_sin = freqs.sin()

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

        # ── KV cache for GQA layers ──
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
                    # FP8 E4M3: 1 byte per element, matching GPU KV cache dtype
                    self._kv_k[layer_idx] = torch.zeros(
                        max_seq, num_kv_heads, head_dim, dtype=torch.uint8)
                    self._kv_v[layer_idx] = torch.zeros(
                        max_seq, num_kv_heads, head_dim, dtype=torch.uint8)
                    kv_bytes += 2 * max_seq * num_kv_heads * head_dim  # 1 byte/elem

            # MLA layers: compressed KV + rope position embeddings (FP8)
            if 'kv_lora_rank' in a:
                klr = a['kv_lora_rank']
                qk_rd = a['qk_rope_dim']
                self._mla_ckv[layer_idx] = torch.zeros(
                    max_seq, klr, dtype=torch.uint8)
                self._mla_kpe[layer_idx] = torch.zeros(
                    max_seq, qk_rd, dtype=torch.uint8)
                kv_bytes += max_seq * (klr + qk_rd)  # 1 byte/elem

        # ── Recurrent + conv state for linear attention layers ──
        state_bytes = 0
        for layer_idx, shape in self._la_conv_state_templates.items():
            self._la_conv_state[layer_idx] = torch.zeros(shape, dtype=torch.float32)
            state_bytes += torch.zeros(shape).nelement() * 4

        for layer_idx, shape in self._la_recur_state_templates.items():
            self._la_recur_state[layer_idx] = torch.zeros(shape, dtype=torch.float32)
            state_bytes += torch.zeros(shape).nelement() * 4

        total_kv_layers = len(self._kv_k) + len(self._mla_ckv)
        logger.info("Pre-allocated buffers: KV cache %.1f MB (%d GQA + %d MLA layers, max_seq=%d), "
                     "recurrent state %.1f MB (%d layers)",
                     kv_bytes / 1e6, len(self._kv_k), len(self._mla_ckv), max_seq,
                     state_bytes / 1e6, len(self._la_recur_state))

    # ──────────────────────────────────────────────────────
    # Rust decode graph configuration
    # ──────────────────────────────────────────────────────

    def _configure_decode_graph(self):
        """Configure the Rust decode graph for single-call decode_step."""
        cfg = self.cfg
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
                # Expand norm_weight [dv] -> [nv*dv] if needed
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

            elif cfg.is_mla:
                # Keep tensors alive for Rust to read via pointers
                w_kc = a['w_kc']
                w_vc = a['w_vc']
                kv_a_norm = a['kv_a_norm']
                self._decode_graph_tensors.extend([w_kc, w_vc, kv_a_norm])

                q_a_norm = a.get('q_a_norm')
                q_a_norm_ptr = q_a_norm.data_ptr() if q_a_norm is not None else 0
                q_a_norm_len = q_a_norm.numel() if q_a_norm is not None else 0
                if q_a_norm is not None:
                    self._decode_graph_tensors.append(q_a_norm)

                # RoPE tables (computed in _init_rope)
                self._decode_graph_tensors.extend([self._mla_rope_cos, self._mla_rope_sin])

                store.add_decode_mla_layer(
                    ld['input_norm_id'], ld['post_attn_norm_id'],
                    a['kv_a_proj_wid'], a['o_proj_wid'],
                    a.get('q_proj_wid'), a.get('q_a_proj_wid'), a.get('q_b_proj_wid'),
                    w_kc.data_ptr(), w_kc.numel(),
                    w_vc.data_ptr(), w_vc.numel(),
                    kv_a_norm.data_ptr(), kv_a_norm.numel(),
                    q_a_norm_ptr, q_a_norm_len,
                    self._mla_rope_cos.data_ptr(), self._mla_rope_sin.data_ptr(),
                    self._mla_rope_cos.numel(), self._max_rope_seq,
                    a['num_heads'], a['kv_lora_rank'],
                    a['qk_nope_dim'], a['qk_rope_dim'],
                    a['v_head_dim'], a['sm_scale'],
                )

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
                raise RuntimeError(f"Unsupported attention type at layer {layer_idx} for Rust decode graph")

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

        # MoE store (experts are already tiled during engine loading)
        if topk > 0 and self.engine is not None:
            store.set_moe_store(self.engine)

        store.finalize_decode()

        # Repack weights to tiled layout for better memory bandwidth.
        # Expert weights must be repacked via engine (before Arc is cloned).
        # Non-expert weights are repacked via store.
        store.repack_to_tiled()

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

        # MLA cache pointers
        mla_ckv_ptrs = None
        mla_kpe_ptrs = None
        if self._mla_ckv:
            mla_ckv_ptrs = []
            mla_kpe_ptrs = []
            for layer_idx in range(num_layers):
                if layer_idx in self._mla_ckv:
                    mla_ckv_ptrs.append(self._mla_ckv[layer_idx].data_ptr())
                    mla_kpe_ptrs.append(self._mla_kpe[layer_idx].data_ptr())
                else:
                    mla_ckv_ptrs.append(0)
                    mla_kpe_ptrs.append(0)

        self._store.set_decode_state(
            self._seq_len, self._max_kv_seq,
            kv_k_ptrs, kv_v_ptrs,
            conv_state_ptrs, recur_state_ptrs,
            mla_ckv_ptrs, mla_kpe_ptrs)

