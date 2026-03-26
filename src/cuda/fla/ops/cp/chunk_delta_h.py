# Stubs for context parallelism (not used in Krasis, single-node inference)

def chunk_gated_delta_rule_bwd_dhu_pre_process(*args, **kwargs):
    raise NotImplementedError("Context parallelism not supported in Krasis")

def chunk_gated_delta_rule_fwd_h_pre_process(*args, **kwargs):
    raise NotImplementedError("Context parallelism not supported in Krasis")

def compress_h0(*args, **kwargs):
    raise NotImplementedError("Context parallelism not supported in Krasis")

def expand_h0(*args, **kwargs):
    raise NotImplementedError("Context parallelism not supported in Krasis")
