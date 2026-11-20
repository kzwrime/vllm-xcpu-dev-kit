#!/bin/bash

# GLM-5.2-INT8 (UE8M0 16x16) TP4/EP4 + GLM-5.3 DFlash2 compile preset with XCPU INT8 mega-kernels:
#   - layer 0 (rms_norm input)   -> fused_mla_qnorm_qb_uk_rope_cache_int8 (A3)
#   - layers 1..N shared         -> fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache_int8 (A2)
#   - layers 1..N full (indexer) -> fused_mla_add_norm_qkva_wk_wq_qnorm_qb_uk_rope_cache_int8 (A1)
#   - MoE post-attention gate    -> fused_add_norm_router_gemm (A2, BF16 gate)

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/GLM-5.2-INT8_dp1_tp4_ep4_compile_sparse_attn_fp8_kvcache_dflash2_5.sh"

export VLLM_XCPU_COMPILE_PASS_FUSED_ADD_NORM_ROUTER_GEMM="${VLLM_XCPU_COMPILE_PASS_FUSED_ADD_NORM_ROUTER_GEMM:-1}"
export TORCH_XCPU_FUSED_ADD_NORM_ROUTER_GEMM_IMPL="${TORCH_XCPU_FUSED_ADD_NORM_ROUTER_GEMM_IMPL:-A2}"

export VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE_INT8="${VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE_INT8:-1}"
export VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_INT8="${VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_INT8:-1}"

if [[ "${VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE_INT8}" == "1" || \
      "${VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_INT8}" == "1" ]]; then
    export VLLM_XCPU_FUSED_MLA=0
fi

# Mega-kernel runtime checks omp_get_max_threads() >= 96 at import time.
export OMP_NUM_THREADS=96
unset OMP_THREAD_LIMIT
export VLLM_XCPU_CPU_PARTITION="${VLLM_XCPU_CPU_PARTITION:-1}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-1}"

echo "Preset: INT8 TP4 + DFlash2 MK ON (shared add_norm=A2, full wk_wq=A1, layer0=A3," \
     "router_gemm=A2, cpu_partition=${VLLM_XCPU_CPU_PARTITION}," \
     "OMP_NUM_THREADS=${OMP_NUM_THREADS}, KMP_BLOCKTIME=${KMP_BLOCKTIME})"
