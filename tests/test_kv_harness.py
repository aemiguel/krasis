import torch
import numpy as np
import os
import sys
import time

# Add python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from krasis import GpuDecodeStore
from krasis.config import ModelConfig

def test_kv_harness(use_polar4=False):
    if use_polar4 and os.environ.get("KRASIS_ALLOW_DEPRECATED_POLAR4_KV") != "1":
        raise RuntimeError(
            "Polar4 KV is deprecated and disabled. Set "
            "KRASIS_ALLOW_DEPRECATED_POLAR4_KV=1 only for legacy implementation diagnostics."
        )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    
    # Minimal config for testing
    nh = 32
    nkv = 8
    hd = 128
    max_seq = 1024
    num_layers = 1
    
    mode_name = "POLAR4" if use_polar4 else "FP8"
    print(f"Initializing KV Harness ({mode_name}): nh={nh}, nkv={nkv}, hd={hd}, max_seq={max_seq}")
    
    # Initialize store
    store = GpuDecodeStore(0)
    
    # Configure it with enough space for our tests
    store.configure(
        hidden_size=hd * nh, 
        num_layers=num_layers,
        vocab_size=32000,
        eps=1e-5,
        max_experts_per_tok=10,
        max_intermediate_size=4096,
        max_qkv_size=nh * hd * 3,
        group_size=128,
        expert_bits=4,
        moe_intermediate_size=0
    )
    
    # Allocation
    kv_stride = nkv * hd
    num_blocks = kv_stride // 16
    
    if use_polar4:
        # Polar4 uses radius (BF16) and angles (4-bit packed as uint8)
        cache_k_radius = torch.zeros((max_seq, num_blocks), dtype=torch.bfloat16, device=device)
        cache_v_radius = torch.zeros((max_seq, num_blocks), dtype=torch.bfloat16, device=device)
        cache_k_angles = torch.zeros((max_seq, num_blocks, 8), dtype=torch.uint8, device=device)
        cache_v_angles = torch.zeros((max_seq, num_blocks, 8), dtype=torch.uint8, device=device)
        
        k_ptr_1 = cache_k_radius.data_ptr()
        v_ptr_1 = cache_v_radius.data_ptr()
        k_ptr_2 = cache_k_angles.data_ptr()
        v_ptr_2 = cache_v_angles.data_ptr()
    else:
        cache_k = torch.zeros((max_seq, kv_stride), dtype=torch.float8_e4m3fn, device=device)
        cache_v = torch.zeros((max_seq, kv_stride), dtype=torch.float8_e4m3fn, device=device)
        k_ptr_1 = cache_k.data_ptr()
        v_ptr_1 = cache_v.data_ptr()
        k_ptr_2 = 0
        v_ptr_2 = 0
    
    # Reference KV cache (BF16)
    ref_k = torch.zeros((max_seq, nkv, hd), dtype=torch.bfloat16, device=device)
    ref_v = torch.zeros((max_seq, nkv, hd), dtype=torch.bfloat16, device=device)
    
    # Attention output buffer (FP32)
    out_f32 = torch.zeros((nh, hd), dtype=torch.float32, device=device)
    
    # Scale for attention
    sm_sc = 1.0 / (hd ** 0.5)
    
    print("Running sequential write/read tests...")
    
    # Test over 10 steps
    for pos in range(10):
        # 1. Generate new Q, K, V
        q = torch.randn((nh, hd), dtype=torch.bfloat16, device=device)
        k = torch.randn((nkv, hd), dtype=torch.bfloat16, device=device)
        v = torch.randn((nkv, hd), dtype=torch.bfloat16, device=device)
        
        # Store in reference
        ref_k[pos] = k
        ref_v[pos] = v
        
        # Convert to FP32 for kernel inputs
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)
        
        # 2. Run kernel-based KV write + Attention
        store.test_kv_cache(
            q_ptr=q_f32.data_ptr(),
            k_ptr=k_f32.data_ptr(),
            v_ptr=v_f32.data_ptr(),
            out_ptr=out_f32.data_ptr(),
            cache_k_ptr=k_ptr_1,
            cache_v_ptr=v_ptr_1,
            cache_k_angles_ptr=k_ptr_2,
            cache_v_angles_ptr=v_ptr_2,
            nh=nh, nkv=nkv, hd=hd, pos=pos, max_seq=max_seq, kv_stride=kv_stride, sm_sc=sm_sc,
            format=2 if use_polar4 else 1 # 2=POLAR4, 1=FP8
        )
        
        # 3. PyTorch reference attention
        k_expanded = ref_k[:pos+1].repeat_interleave(nh // nkv, dim=1) # [seq_len, nh, hd]
        v_expanded = ref_v[:pos+1].repeat_interleave(nh // nkv, dim=1) # [seq_len, nh, hd]
        
        q_ref = q.unsqueeze(1).to(torch.float32)
        k_ref = k_expanded.transpose(0, 1).to(torch.float32)
        v_ref = v_expanded.transpose(0, 1).to(torch.float32)
        
        scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * sm_sc # [nh, 1, seq_len]
        probs = torch.softmax(scores, dim=-1)
        ref_out = torch.matmul(probs, v_ref).squeeze(1) # [nh, hd]
        
        # 4. Compare
        diff = (out_f32 - ref_out).abs()
        max_diff = diff.max().item()
        avg_diff = diff.mean().item()
        
        print(f"Step {pos}: Max Diff = {max_diff:.6f}, Avg Diff = {avg_diff:.6f}")
        
    print("Harness completed.")

if __name__ == "__main__":
    test_kv_harness(use_polar4=False)
    if os.environ.get("KRASIS_ALLOW_DEPRECATED_POLAR4_KV") == "1":
        print("-" * 40)
        test_kv_harness(use_polar4=True)
