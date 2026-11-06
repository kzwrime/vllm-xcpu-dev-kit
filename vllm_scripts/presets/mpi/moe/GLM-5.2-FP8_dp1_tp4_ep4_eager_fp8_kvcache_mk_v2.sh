#!/bin/bash

# GLM-5.2-FP8_dp1_tp4_ep4_eager_fp8_kvcache with the fused MLA front half (fused_mla_qnorm_qb_uk, v2 / A1
# mega-kernel) enabled: VLLM_XCPU_FUSED_MLA_QNORM_QB_UK=2. Default is off.

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/GLM-5.2-FP8_dp1_tp4_ep4_eager_fp8_kvcache.sh"
export VLLM_XCPU_FUSED_MLA_QNORM_QB_UK=2
echo "Preset: fused_mla_qnorm_qb_uk=2 (v2 / A1)"
