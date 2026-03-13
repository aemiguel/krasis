#!/usr/bin/env python3
"""Generate SVG charts for Q3.5 decode component analysis."""

# Measured timing data from 200-step average (KRASIS_CPU_DECODE_TIMING)
# Qwen3.5-35B-A3B INT4/INT4, 16 threads, 4 NUMA nodes
components = [
    # (name, time_ms, bytes_per_token, description)
    ("la_proj",      16.3,  391_311_360, "LA in_proj (30L)"),
    ("la_conv",       1.9,    1_966_080, "LA conv1d (30L)"),
    ("la_recur",      8.1,   62_914_560, "LA recurrence (30L)"),
    ("la_gate_norm",  0.1,           0, "LA gate+norm (30L)"),
    ("la_out_proj",   7.7,  129_761_280, "LA out_proj (30L)"),
    ("gqa_proj",      4.0,   97_320_960, "GQA QKV proj (10L)"),
    ("gqa_rope",      0.4,           0, "GQA RoPE (10L)"),
    ("gqa_attn",      1.6,      276_480, "GQA attention (10L)"),
    ("gqa_o_proj",    2.2,   43_253_760, "GQA O proj (10L)"),
    ("moe_route",     6.4,   83_886_080, "MoE routing (40L)"),
    ("moe_experts",  13.9,  519_045_120, "MoE experts (40L×8E)"),
    ("moe_shared",    0.0,           0, "Shared expert (GPU)"),
    ("lm_head",       8.3,  262_225_920, "LM head (248K vocab)"),
    ("overhead",      0.1,           0, "Overhead"),
]

total_ms = 71.2
total_bytes = sum(c[2] for c in components)
epyc_peak = 204.8  # GB/s
single_node = 51.2  # GB/s per NUMA node
achievable = 150.0  # typical achievable

print(f"Total bytes per token: {total_bytes:,} ({total_bytes/1e6:.1f} MB)")
print(f"Effective bandwidth: {total_bytes/1e9/(total_ms/1000):.1f} GB/s")
print(f"Theoretical at {achievable:.0f} GB/s: {total_bytes/1e9/achievable*1000:.1f} ms = {1000/(total_bytes/1e9/achievable*1000):.0f} tok/s")
print(f"Theoretical at {epyc_peak:.0f} GB/s: {total_bytes/1e9/epyc_peak*1000:.1f} ms = {1000/(total_bytes/1e9/epyc_peak*1000):.0f} tok/s")
