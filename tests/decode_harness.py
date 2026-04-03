#!/usr/bin/env python3
"""Synthetic GPU decode harness for config-driven compute benchmarking.

Runs through the supported ./dev entrypoint only. Uses config-derived layer
shapes, synthetic weights, and real Rust GPU decode kernels. The focus is
decode compute, so the default path pins all synthetic experts into HCS and
reuses zeroed KV/state buffers between runs.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


def _dev_guard() -> None:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit(
            "Run this through ./dev decode-harness ..., not python tests/decode_harness.py directly."
        )


def _load_conf(path: str) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    if not os.path.isfile(path):
        return cfg
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            cfg[key.strip()] = val.strip().strip('"').strip("'")
    return cfg


def _prescan_selected_gpus(argv: Sequence[str]) -> None:
    conf_path = None
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            conf_path = argv[i + 1]
            break
    if not conf_path:
        return
    cfg = _load_conf(conf_path)
    selected = cfg.get("CFG_SELECTED_GPUS", "").strip()
    if selected:
        gpus = [x.strip() for x in selected.split(",") if x.strip()]
        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
            print(f"Pre-scan: set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")


_dev_guard()
_prescan_selected_gpus(sys.argv[1:])

import numpy as np
import torch

from krasis import GpuDecodeStore
from krasis.config import ModelConfig
from krasis.kv_cache import PagedKVCache


GROUP_SIZE = 128
PAGE_SIZE = 16


@dataclass
class HarnessRunResult:
    elapsed_s: float
    tok_s: float
    min_free_vram_mb: int
    hcs_loaded: int
    hcs_total: int
    hcs_pct: float
    generated: int


class SyntheticDecodeHarness:
    def __init__(
        self,
        cfg: ModelConfig,
        gpu_idx: int,
        prompt_len: int,
        steps: int,
        max_experts: int,
        topk: int,
        expert_bits: int,
        attn_mode: str,
        kv_format: str,
        vocab_size: int,
        use_timing: bool,
    ):
        if cfg.is_mla:
            raise RuntimeError("decode harness currently supports GQA and LA/GQA hybrid models, not MLA.")
        if cfg.is_nemotron_h:
            raise RuntimeError("decode harness currently supports GQA and LA/GQA hybrid models, not Nemotron-H.")

        self.cfg = cfg
        self.gpu_idx = gpu_idx
        self.device = torch.device(f"cuda:{gpu_idx}")
        self.prompt_len = prompt_len
        self.steps = steps
        self.max_experts = max_experts
        self.topk = min(topk, max_experts) if max_experts > 0 else topk
        self.expert_bits = expert_bits
        self.attn_mode = attn_mode
        self.kv_format = kv_format
        self.vocab_size = vocab_size
        self.use_timing = use_timing
        self.resident_experts = max(1, min(self.max_experts, max(self.topk * 2, 16)))

        self.keepalive: List[object] = []
        self.state_tensors: List[torch.Tensor] = []
        self.cache_tensors: List[torch.Tensor] = []
        self.cached_bf16_wids: Dict[Tuple[int, int], int] = {}
        self.cached_attn_wids: Dict[Tuple[str, int, int], int] = {}
        self.cached_expert_ptrs: Dict[Tuple[int, int, int], Tuple[int, int, int, int]] = {}
        self.cached_norms: Dict[int, torch.Tensor] = {}
        self.cached_fp32_vecs: Dict[Tuple[str, int], torch.Tensor] = {}
        self.cached_fp32_mats: Dict[Tuple[str, int, int], torch.Tensor] = {}
        self.cached_gate_wids: Dict[Tuple[int, int], int] = {}

        self.store = GpuDecodeStore(gpu_idx)
        self._setup_store()

    def _rand_bf16(self, shape: Tuple[int, ...], *, device: torch.device) -> torch.Tensor:
        return (torch.randn(shape, device=device, dtype=torch.float32) * 0.02).to(
            torch.bfloat16
        ).contiguous()

    def _rand_fp32(self, shape: Tuple[int, ...], *, device: torch.device) -> torch.Tensor:
        return (torch.randn(shape, device=device, dtype=torch.float32) * 0.02).contiguous()

    def _register_bf16_weight(self, shape: Tuple[int, int]) -> int:
        cached = self.cached_bf16_wids.get(shape)
        if cached is not None:
            return cached
        weight = self._rand_bf16(shape, device=self.device)
        self.keepalive.append(weight)
        wid = self.store.register_weight(weight.data_ptr(), weight.shape[0], weight.shape[1], 0)
        self.cached_bf16_wids[shape] = wid
        return wid

    def _register_attn_weight(self, shape: Tuple[int, int]) -> int:
        cache_key = (self.attn_mode, shape[0], shape[1])
        cached = self.cached_attn_wids.get(cache_key)
        if cached is not None:
            return cached
        weight_cpu = self._rand_bf16(shape, device=torch.device("cpu"))
        if self.attn_mode == "bf16":
            weight = weight_cpu.to(self.device).contiguous()
            self.keepalive.append(weight)
            wid = self.store.register_weight(weight.data_ptr(), weight.shape[0], weight.shape[1], 0)
            self.cached_attn_wids[cache_key] = wid
            return wid

        n, k = weight_cpu.shape
        if self.attn_mode == "int4":
            packed_bytes, scales_bytes, _, _ = self.store.repack_marlin_int4_cpu(
                weight_cpu.data_ptr(), n, k, GROUP_SIZE
            )
            packed_np = np.frombuffer(packed_bytes, dtype=np.uint32)
            scales_np = np.frombuffer(scales_bytes, dtype=np.uint16)

            repacked = torch.from_numpy(packed_np.copy()).to(torch.int32).to(self.device)
            scale_raw = torch.from_numpy(scales_np.copy()).to(torch.int16)
            n_scale_elements = (k // GROUP_SIZE) * n
            scales_slot = torch.empty(n_scale_elements * 2, dtype=torch.int16, device=self.device)
            bf16_on_gpu = scale_raw.view(torch.bfloat16).to(self.device)
            scales_slot[:n_scale_elements].copy_(bf16_on_gpu.view(torch.int16).reshape(-1))

            self.keepalive.extend([repacked, scales_slot])
            wid = self.store.register_marlin_int4_weight(
                repacked.data_ptr(), scales_slot.data_ptr(), n, k, GROUP_SIZE
            )
            self.cached_attn_wids[cache_key] = wid
            return wid

        packed_bytes, scales_bytes, _, _ = self.store.repack_marlin_int8_cpu(
            weight_cpu.data_ptr(), n, k, GROUP_SIZE
        )
        packed_np = np.frombuffer(packed_bytes, dtype=np.uint32)
        scales_np = np.frombuffer(scales_bytes, dtype=np.uint16)
        repacked = torch.from_numpy(packed_np.copy()).to(torch.int32).to(self.device)
        scale_perm = torch.from_numpy(scales_np.copy()).view(torch.int16).view(torch.bfloat16)
        scale_perm = scale_perm.to(self.device).reshape(k // GROUP_SIZE, n).contiguous()
        self.keepalive.extend([repacked, scale_perm])
        wid = self.store.register_marlin_int8_weight(
            repacked.data_ptr(), scale_perm.data_ptr(), n, k, GROUP_SIZE
        )
        self.cached_attn_wids[cache_key] = wid
        return wid

    def _pack_expert_matrix(self, weight_cpu: torch.Tensor) -> Tuple[int, int, int, int]:
        rows, cols = weight_cpu.shape
        cache_key = (self.expert_bits, rows, cols)
        cached = self.cached_expert_ptrs.get(cache_key)
        if cached is not None:
            return cached
        if self.expert_bits == 16:
            pinned = weight_cpu.contiguous().pin_memory()
            self.keepalive.append(pinned)
            result = (pinned.data_ptr(), pinned.numel() * 2, 0, 0)
            self.cached_expert_ptrs[cache_key] = result
            return result

        if self.expert_bits == 4:
            packed_bytes, scales_bytes, _, _ = self.store.repack_marlin_int4_cpu_no_simple(
                weight_cpu.data_ptr(), rows, cols, GROUP_SIZE
            )
        else:
            packed_bytes, scales_bytes, _, _ = self.store.repack_marlin_int8_cpu(
                weight_cpu.data_ptr(), rows, cols, GROUP_SIZE
            )

        packed_host = torch.from_numpy(
            np.frombuffer(packed_bytes, dtype=np.uint8).copy()
        ).pin_memory()
        scales_host = torch.from_numpy(
            np.frombuffer(scales_bytes, dtype=np.uint8).copy()
        ).pin_memory()
        self.keepalive.extend([packed_host, scales_host])
        result = (
            packed_host.data_ptr(),
            packed_host.numel(),
            scales_host.data_ptr(),
            scales_host.numel(),
        )
        self.cached_expert_ptrs[cache_key] = result
        return result

    def _shared_norm(self, size: int) -> torch.Tensor:
        cached = self.cached_norms.get(size)
        if cached is not None:
            return cached
        norm = self._rand_bf16((size,), device=self.device)
        self.keepalive.append(norm)
        self.cached_norms[size] = norm
        return norm

    def _shared_fp32_vec(self, tag: str, size: int) -> torch.Tensor:
        cache_key = (tag, size)
        cached = self.cached_fp32_vecs.get(cache_key)
        if cached is not None:
            return cached
        vec = self._rand_fp32((size,), device=self.device)
        self.keepalive.append(vec)
        self.cached_fp32_vecs[cache_key] = vec
        return vec

    def _shared_fp32_mat(self, tag: str, rows: int, cols: int) -> torch.Tensor:
        cache_key = (tag, rows, cols)
        cached = self.cached_fp32_mats.get(cache_key)
        if cached is not None:
            return cached
        mat = self._rand_fp32((rows, cols), device=self.device)
        self.keepalive.append(mat)
        self.cached_fp32_mats[cache_key] = mat
        return mat

    def _make_dense_mlp(self, layer_idx: int) -> None:
        inter = self.cfg.intermediate_size
        gate_wid = self._register_bf16_weight((inter, self.cfg.hidden_size))
        up_wid = self._register_bf16_weight((inter, self.cfg.hidden_size))
        down_wid = self._register_bf16_weight((self.cfg.hidden_size, inter))
        self.store.register_mlp(
            layer_idx,
            "dense",
            gate_proj_wid=gate_wid,
            up_proj_wid=up_wid,
            down_proj_wid=down_wid,
        )

    def _make_moe(self, layer_idx: int) -> None:
        n_experts = self.max_experts
        topk = min(self.topk, n_experts)
        gate_shape = (n_experts, self.cfg.hidden_size)
        gate_wid = self.cached_gate_wids.get(gate_shape)
        if gate_wid is None:
            gate = torch.zeros(gate_shape, dtype=torch.float32, device=self.device)
            self.keepalive.append(gate)
            gate_wid = self.store.register_weight(gate.data_ptr(), gate.shape[0], gate.shape[1], 1)
            self.cached_gate_wids[gate_shape] = gate_wid

        gate_bias = torch.full((n_experts,), -1000.0, dtype=torch.float32, device=self.device)
        for expert_idx in range(self.resident_experts):
            gate_bias[expert_idx] = float(self.resident_experts - expert_idx)
        self.keepalive.append(gate_bias)

        expert_ptrs = []
        shared_w13 = self._rand_bf16(
            (self.cfg.moe_intermediate_size * 2, self.cfg.hidden_size),
            device=torch.device("cpu"),
        )
        shared_w2 = self._rand_bf16(
            (self.cfg.hidden_size, self.cfg.moe_intermediate_size),
            device=torch.device("cpu"),
        )
        w13p, w13pb, w13s, w13sb = self._pack_expert_matrix(shared_w13)
        w2p, w2pb, w2s, w2sb = self._pack_expert_matrix(shared_w2)
        for _ in range(n_experts):
            expert_ptrs.append((w13p, w13pb, w13s, w13sb, w2p, w2pb, w2s, w2sb))

        shared_ptrs = None
        if self.cfg.effective_shared_expert_intermediate > 0:
            se_inter = self.cfg.effective_shared_expert_intermediate
            se_w13 = self._rand_bf16(
                (se_inter * 2, self.cfg.hidden_size), device=torch.device("cpu")
            )
            se_w2 = self._rand_bf16(
                (self.cfg.hidden_size, se_inter), device=torch.device("cpu")
            )
            w13p, w13pb, w13s, w13sb = self._pack_expert_matrix(se_w13)
            w2p, w2pb, w2s, w2sb = self._pack_expert_matrix(se_w2)
            shared_ptrs = (w13p, w13pb, w13s, w13sb, w2p, w2pb, w2s, w2sb)

        scoring_func = 1 if self.cfg.scoring_func == "softmax" else 0
        self.store.register_moe_layer(
            layer_idx,
            expert_ptrs,
            shared_ptrs,
            n_experts,
            topk,
            scoring_func,
            self.cfg.norm_topk_prob,
            self.cfg.routed_scaling_factor,
            gate_wid,
            gate_bias.data_ptr(),
            0,
            None,
        )

        if shared_ptrs is not None and self.cfg.model_type in ("qwen3_next", "qwen3_5_moe_text"):
            shared_gate = self._rand_bf16((1, self.cfg.hidden_size), device=self.device)
            self.keepalive.append(shared_gate)
            sg_wid = self.store.register_weight(
                shared_gate.data_ptr(), shared_gate.shape[0], shared_gate.shape[1], 0
            )
            self.store.set_moe_shared_gate_wid(layer_idx, sg_wid)

        self.store.register_mlp(layer_idx, "moe")

    def _register_layer(self, layer_idx: int, layer_type: str) -> None:
        inp_norm = self._shared_norm(self.cfg.hidden_size)
        post_norm = self._shared_norm(self.cfg.hidden_size)

        if layer_type == "linear_attention":
            nk = self.cfg.linear_num_key_heads
            nv = self.cfg.linear_num_value_heads
            dk = self.cfg.linear_key_head_dim
            dv = self.cfg.linear_value_head_dim
            hr = max(1, nv // nk)
            qkvz_out = nk * (2 * dk + 2 * hr * dv)
            ba_out = nk * 2 * hr
            out_in = nv * dv
            conv_dim = nk * dk * 2 + nv * dv
            kernel_dim = self.cfg.linear_conv_kernel_dim

            qkvz_wid = self._register_attn_weight((qkvz_out, self.cfg.hidden_size))
            ba_wid = self._register_attn_weight((ba_out, self.cfg.hidden_size))
            out_wid = self._register_attn_weight((self.cfg.hidden_size, out_in))

            conv_weight = self._shared_fp32_mat("la_conv_weight", conv_dim, kernel_dim)
            a_log = self._shared_fp32_vec("la_a_log", nv)
            dt_bias = self._shared_fp32_vec("la_dt_bias", nv)
            norm_weight = self._shared_fp32_vec("la_norm_weight", dv)
            conv_state = torch.zeros(conv_dim, kernel_dim, dtype=torch.float32, device=self.device)
            recur_state = torch.zeros(nv, dk, dv, dtype=torch.float32, device=self.device)
            self.keepalive.extend([conv_state, recur_state])
            self.state_tensors.extend([conv_state, recur_state])

            self.store.register_la_layer(
                layer_idx=layer_idx,
                input_norm_ptr=inp_norm.data_ptr(),
                input_norm_size=inp_norm.numel(),
                post_attn_norm_ptr=post_norm.data_ptr(),
                post_attn_norm_size=post_norm.numel(),
                in_proj_qkvz_wid=qkvz_wid,
                in_proj_ba_wid=ba_wid,
                out_proj_wid=out_wid,
                conv_weight_ptr=conv_weight.data_ptr(),
                a_log_ptr=a_log.data_ptr(),
                dt_bias_ptr=dt_bias.data_ptr(),
                norm_weight_ptr=norm_weight.data_ptr(),
                conv_state_ptr=conv_state.data_ptr(),
                recur_state_ptr=recur_state.data_ptr(),
                nk=nk,
                nv=nv,
                dk=dk,
                dv=dv,
                hr=hr,
                kernel_dim=kernel_dim,
                conv_dim=conv_dim,
                scale=1.0 / math.sqrt(dk),
            )
        elif layer_type == "full_attention":
            head_dim = self.cfg.gqa_head_dim or self.cfg.head_dim
            q_out = self.cfg.num_attention_heads * head_dim * (2 if self.cfg.gated_attention else 1)
            kv_out = self.cfg.num_key_value_heads * head_dim

            q_wid = self._register_attn_weight((q_out, self.cfg.hidden_size))
            k_wid = self._register_attn_weight((kv_out, self.cfg.hidden_size))
            v_wid = self._register_attn_weight((kv_out, self.cfg.hidden_size))
            o_wid = self._register_attn_weight(
                (self.cfg.hidden_size, self.cfg.num_attention_heads * head_dim)
            )

            q_norm = self._shared_fp32_vec("gqa_q_norm", head_dim)
            k_norm = self._shared_fp32_vec("gqa_k_norm", head_dim)

            self.store.register_gqa_layer(
                layer_idx=layer_idx,
                input_norm_ptr=inp_norm.data_ptr(),
                input_norm_size=inp_norm.numel(),
                post_attn_norm_ptr=post_norm.data_ptr(),
                post_attn_norm_size=post_norm.numel(),
                q_proj_wid=q_wid,
                k_proj_wid=k_wid,
                v_proj_wid=v_wid,
                o_proj_wid=o_wid,
                fused_qkv_wid=None,
                num_heads=self.cfg.num_attention_heads,
                num_kv_heads=self.cfg.num_key_value_heads,
                head_dim=head_dim,
                sm_scale=1.0 / math.sqrt(head_dim),
                q_norm_ptr=q_norm.data_ptr(),
                k_norm_ptr=k_norm.data_ptr(),
                gated=self.cfg.gated_attention,
            )
        else:
            raise RuntimeError(f"Unsupported synthetic layer type: {layer_type}")

        if self.cfg.is_moe_layer(layer_idx):
            self._make_moe(layer_idx)
        elif layer_idx < self.cfg.first_k_dense_replace and self.cfg.intermediate_size > 0:
            self._make_dense_mlp(layer_idx)
        else:
            self.store.register_mlp(layer_idx, "none")

    def _setup_kv_cache(self) -> int:
        full_attn_layers = self.cfg.num_full_attention_layers
        total_tokens = max(self.prompt_len + self.steps + 32, PAGE_SIZE)
        max_pages = (total_tokens + PAGE_SIZE - 1) // PAGE_SIZE
        kv_dtype = torch.float8_e4m3fn if self.kv_format == "fp8" else torch.bfloat16
        cache = PagedKVCache(
            self.cfg,
            full_attn_layers,
            self.device,
            max_pages=max_pages,
            kv_dtype=kv_dtype,
            kv_format=self.kv_format,
        )

        if cache.kv_format == 2 and cache.k_radius_cache is not None:
            polar4_ptrs = []
            gqa_idx = 0
            for layer_idx in range(self.cfg.num_hidden_layers):
                layer_type = (
                    self.cfg.layer_types[layer_idx]
                    if self.cfg.layer_types is not None
                    else "full_attention"
                )
                if layer_type in ("full_attention", "sliding_attention"):
                    kr = cache.k_radius_cache[gqa_idx]
                    vr = cache.v_radius_cache[gqa_idx]
                    ka = cache.k_angles_cache[gqa_idx]
                    va = cache.v_angles_cache[gqa_idx]
                    polar4_ptrs.append(
                        (layer_idx, kr.data_ptr(), vr.data_ptr(), ka.data_ptr(), va.data_ptr())
                    )
                    self.cache_tensors.extend([kr, vr, ka, va])
                    gqa_idx += 1
            num_blocks = (cache.num_kv_heads * cache.gqa_head_dim) // 16
            self.store.set_kv_cache_ptrs_polar4(
                polar4_ptrs, cache.max_pages * cache.page_size, num_blocks
            )
        elif cache.k_cache is not None:
            kv_ptrs = []
            gqa_idx = 0
            for layer_idx in range(self.cfg.num_hidden_layers):
                layer_type = (
                    self.cfg.layer_types[layer_idx]
                    if self.cfg.layer_types is not None
                    else "full_attention"
                )
                if layer_type in ("full_attention", "sliding_attention"):
                    k_layer = cache.k_cache[gqa_idx]
                    v_layer = cache.v_cache[gqa_idx]
                    kv_ptrs.append((layer_idx, k_layer.data_ptr(), v_layer.data_ptr()))
                    self.cache_tensors.extend([k_layer, v_layer])
                    gqa_idx += 1
            self.store.set_kv_cache_ptrs(kv_ptrs, cache.max_pages * cache.page_size)
        else:
            raise RuntimeError("Unsupported KV cache setup for synthetic harness.")

        self.keepalive.append(cache)
        return cache.max_pages * cache.page_size

    def _setup_store(self) -> None:
        max_qkv = self.cfg.hidden_size * 3
        if self.cfg.layer_types is not None:
            for layer_type in self.cfg.layer_types:
                if layer_type == "linear_attention":
                    qkvz_out = self.cfg.linear_num_key_heads * (
                        2 * self.cfg.linear_key_head_dim
                        + 2
                        * (self.cfg.linear_num_value_heads // self.cfg.linear_num_key_heads)
                        * self.cfg.linear_value_head_dim
                    )
                    max_qkv = max(max_qkv, qkvz_out)
        max_inter = max(self.cfg.moe_intermediate_size, self.cfg.intermediate_size)

        self.store.configure(
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_hidden_layers,
            vocab_size=self.vocab_size,
            eps=self.cfg.rms_norm_eps,
            max_experts_per_tok=max(1, min(self.topk, self.max_experts)),
            max_intermediate_size=max_inter,
            max_qkv_size=max_qkv,
            group_size=GROUP_SIZE,
            expert_bits=self.expert_bits,
            moe_intermediate_size=self.cfg.moe_intermediate_size,
        )

        embedding = self._rand_bf16((self.vocab_size, self.cfg.hidden_size), device=self.device)
        final_norm = self._rand_bf16((self.cfg.hidden_size,), device=self.device)
        lm_head = self._rand_bf16((self.vocab_size, self.cfg.hidden_size), device=self.device)
        self.keepalive.extend([embedding, final_norm, lm_head])

        self.store.set_embedding(embedding.data_ptr())
        self.store.set_final_norm(final_norm.data_ptr(), final_norm.numel())
        lm_head_wid = self.store.register_weight(
            lm_head.data_ptr(), lm_head.shape[0], lm_head.shape[1], 0
        )
        self.store.set_lm_head(lm_head_wid)
        self.store.set_norm_bias_one(self.cfg.norm_bias_one)

        layer_types = (
            self.cfg.layer_types
            if self.cfg.layer_types is not None
            else ["full_attention"] * self.cfg.num_hidden_layers
        )
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_idx == 0 or (layer_idx + 1) % 8 == 0 or layer_idx + 1 == len(layer_types):
                print(
                    f"Registering synthetic layer {layer_idx + 1}/{len(layer_types)} ({layer_type})",
                    flush=True,
                )
            self._register_layer(layer_idx, layer_type)

        max_seq = self._setup_kv_cache()
        if any(t == "full_attention" for t in layer_types):
            rope_half = self.cfg.rotary_dim // 2
            inv_freq = 1.0 / (
                self.cfg.rope_theta
                ** (torch.arange(0, rope_half * 2, 2, dtype=torch.float32) / (rope_half * 2))
            )
            t = torch.arange(max_seq, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            cos_f32 = freqs.cos().contiguous().to(self.device)
            sin_f32 = freqs.sin().contiguous().to(self.device)
            self.keepalive.extend([cos_f32, sin_f32])
            self.store.set_rope_tables(cos_f32.data_ptr(), sin_f32.data_ptr(), cos_f32.shape[1], max_seq)

        free_mb = int(torch.cuda.mem_get_info(self.device)[0] // (1024 * 1024))
        hcs_budget_mb = max(256, free_mb - 512)
        self.store.init_hcs(hcs_budget_mb, 0)
        moe_layers = [
            idx for idx in range(self.cfg.num_hidden_layers)
            if self.cfg.is_moe_layer(idx)
        ]
        for layer_idx in moe_layers:
            for expert_idx in range(self.resident_experts):
                self.store.hcs_pin_expert(layer_idx, expert_idx)
        _, hcs_loaded, hcs_total, _ = self.store.get_benchmark_stats()
        required_hcs = len(moe_layers) * self.resident_experts
        if hcs_loaded < required_hcs:
            raise RuntimeError(
                f"Synthetic harness requires routed experts to be resident for compute-only decode, "
                f"but only pinned {hcs_loaded}/{required_hcs} required experts "
                f"(total synthetic experts {hcs_total})."
            )
        if self.use_timing:
            self.store.set_timing(True)

    def reset_runtime_state(self) -> None:
        for tensor in self.state_tensors:
            tensor.zero_()
        for tensor in self.cache_tensors:
            tensor.zero_()
        self.store.set_kv_position(self.prompt_len)

    def run_once(self, *, temperature: float) -> HarnessRunResult:
        self.reset_runtime_state()
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        generated = self.store.gpu_generate_batch(
            first_token=1,
            start_position=self.prompt_len,
            max_tokens=self.steps,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            stop_ids=[],
            presence_penalty=0.0,
        )
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - t0
        min_free, hcs_loaded, hcs_total, hcs_pct = self.store.get_benchmark_stats()
        return HarnessRunResult(
            elapsed_s=elapsed,
            tok_s=(len(generated) / elapsed) if elapsed > 0 else 0.0,
            min_free_vram_mb=min_free,
            hcs_loaded=hcs_loaded,
            hcs_total=hcs_total,
            hcs_pct=hcs_pct,
            generated=len(generated),
        )


def _resolve_gpu_idx(conf: Dict[str, str]) -> int:
    selected = conf.get("CFG_SELECTED_GPUS", "").strip()
    if not selected:
        return 0
    return 0


def _default_max_experts(cfg: ModelConfig) -> int:
    if cfg.n_routed_experts <= 0:
        return 0
    return min(cfg.n_routed_experts, max(cfg.num_experts_per_tok * 2, 16))


def _default_attn_mode(conf: Dict[str, str]) -> str:
    quant = conf.get("CFG_ATTENTION_QUANT", "bf16").strip().lower()
    return "int4" if quant == "awq" else "bf16"


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic GPU decode harness")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--prompt-len", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-experts", type=int, default=0,
                        help="Synthetic experts per MoE layer (0 = auto)")
    parser.add_argument("--topk", type=int, default=0,
                        help="Synthetic routed experts per token (0 = model default)")
    parser.add_argument("--expert-bits", type=int, choices=[4, 8, 16], default=None)
    parser.add_argument("--attn-mode", choices=["bf16", "int4", "int8"], default=None)
    parser.add_argument("--kv-format", choices=["fp8", "bf16", "polar4"], default=None)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--no-graph", action="store_true",
                        help="Disable CUDA graphs to expose per-component kernel timings")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    conf = _load_conf(args.config)
    model_path = conf.get("MODEL_PATH")
    if not model_path:
        raise SystemExit(f"MODEL_PATH missing from {args.config}")
    model_path = os.path.expanduser(model_path)
    cfg = ModelConfig.from_model_path(model_path)

    if args.no_graph:
        os.environ["KRASIS_NO_GRAPH"] = "1"
    env_timing = os.environ.get("KRASIS_DECODE_TIMING", "") == "1"
    use_timing = args.timing or env_timing
    if use_timing:
        os.environ["KRASIS_DECODE_TIMING"] = "1"

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    gpu_idx = _resolve_gpu_idx(conf)
    max_experts = args.max_experts if args.max_experts > 0 else _default_max_experts(cfg)
    if cfg.n_routed_experts > 0 and max_experts <= 0:
        raise SystemExit("Synthetic harness needs at least one expert for MoE layers.")
    topk = args.topk if args.topk > 0 else cfg.num_experts_per_tok
    expert_bits = args.expert_bits
    if expert_bits is None:
        expert_bits = int(conf.get("CFG_GPU_EXPERT_BITS", "4"))
    attn_mode = args.attn_mode or _default_attn_mode(conf)
    kv_format = args.kv_format or conf.get("CFG_KV_DTYPE", "polar4").replace("fp8_e4m3", "fp8")

    print("Synthetic GPU decode harness")
    print(f"  model: {os.path.basename(model_path)}")
    print(f"  gpu: cuda:{gpu_idx}")
    print(f"  layers: {cfg.num_hidden_layers}")
    print(f"  layer types: {cfg.layer_types if cfg.layer_types is not None else ['full_attention'] * cfg.num_hidden_layers}")
    print(f"  prompt_len: {args.prompt_len}")
    print(f"  steps: {args.steps}")
    print(f"  runs: {args.runs}")
    print(f"  warmup: {args.warmup}")
    print(f"  attn_mode: {attn_mode}")
    print(f"  expert_bits: {expert_bits}")
    print(f"  kv_format: {kv_format}")
    print(f"  max_experts: {max_experts}")
    print(f"  topk: {topk}")
    print(f"  resident_experts: {min(max_experts, max(topk * 2, 16))}")
    print(f"  synthetic_vocab: {args.vocab_size}")
    print(f"  timing: {'on' if use_timing else 'off'}")
    print(f"  graphs: {'off' if args.no_graph else 'on'}")
    print("", flush=True)

    harness = SyntheticDecodeHarness(
        cfg=cfg,
        gpu_idx=gpu_idx,
        prompt_len=args.prompt_len,
        steps=args.steps,
        max_experts=max_experts,
        topk=topk,
        expert_bits=expert_bits,
        attn_mode=attn_mode,
        kv_format=kv_format,
        vocab_size=args.vocab_size,
        use_timing=use_timing,
    )

    for i in range(args.warmup):
        result = harness.run_once(temperature=args.temperature)
        print(
            f"Warmup {i + 1}/{args.warmup}: {result.generated} tokens in "
            f"{result.elapsed_s:.3f}s = {result.tok_s:.2f} tok/s"
        )

    results = []
    for i in range(args.runs):
        result = harness.run_once(temperature=args.temperature)
        results.append(result)
        print(
            f"Run {i + 1}/{args.runs}: {result.generated} tokens in "
            f"{result.elapsed_s:.3f}s = {result.tok_s:.2f} tok/s  "
            f"(min_free={result.min_free_vram_mb} MB, HCS={result.hcs_loaded}/{result.hcs_total} "
            f"{result.hcs_pct:.1f}%)"
        )

    best = max(results, key=lambda r: r.tok_s)
    avg = sum(r.tok_s for r in results) / len(results)
    print("")
    print("Summary")
    print(f"  Best decode: {best.tok_s:.2f} tok/s")
    print(f"  Avg decode:  {avg:.2f} tok/s")
    print(f"  Best run:    {best.generated} tokens in {best.elapsed_s:.3f}s")
    print(f"  Min free:    {best.min_free_vram_mb} MB")
    print(f"  HCS:         {best.hcs_loaded}/{best.hcs_total} ({best.hcs_pct:.1f}%)")
    print("")
    print("Notes")
    print("  Synthetic harness uses config-derived layer schedule and real Rust decode kernels.")
    print("  Attention INT4 mode uses synthetic Marlin/simple-INT4 registration, not AWQ calibration.")
    print("  The gate still scales with total synthetic experts, but routing is biased into a resident subset so this measures decode compute rather than cold DMA.")


if __name__ == "__main__":
    main()
