// Copyright (c) 2024, Tri Dao.
// Modified by Krasis: BF16 Q with FP8 E4M3 K/V (causal, hdim96)
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_fp8kv_<cutlass::bfloat16_t, 96, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_fp8kv_hdim96<cutlass::bfloat16_t, true>(params, stream);
}

} // namespace FLASH_NAMESPACE
