#!/bin/bash

# GLM-5.2-INT8 (UE8M0 16x16) DP2/TP4/EP8 alltoallv_v5 compile preset with XCPU INT8 mega-kernels:
#   - layer 0 (rms_norm input)   -> fused_mla_qnorm_qb_uk_rope_cache_int8 (A3)
#   - layers 1..N shared         -> fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache_int8 (A2)
#   - layers 1..N full (indexer) -> fused_mla_add_norm_qkva_wk_wq_qnorm_qb_uk_rope_cache_int8 (A1)
#   - MoE post-attention gate    -> fused_add_norm_router_gemm (A2, BF16 gate)
#
# Notes on DP>1 + TP>1 + EP alltoallv_v5:
#   - With VLLM_DISABLE_ATTN_REDUCE_SCATTER_FOR_MOE=0 (default cross-layer SP), MoE layers place
#     next-layer input_layernorm before TP all_gather, so MoE->MoE layers gracefully stay unfused for
#     Phase-2 add_norm and fall through to Phase-1 qnorm INT8 (A3), while dense->MoE layers (L1..3)
#     hit Phase-2. Setting VLLM_DISABLE_ATTN_REDUCE_SCATTER_FOR_MOE=1 runs SP all_gather before
#     input_layernorm so Phase-2 absorbs SP all_gather across all MoE layers too.
#   - 8 ranks on one 256-CPU host: VLLM_XCPU_CPU_PARTITION=1 gives 32 CPUs/rank, and OMP_NUM_THREADS=96
#     enables the 96-PE MK probe/launch (3x oversubscription within slice; 1.5x on 2-node 4-rank/node).

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/GLM-5.2-INT8_dp2_tp4_ep8_compile_fp8_kvcache_alltoallv_v5.sh"

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

echo "Preset: INT8 DP2 TP4 EP8 alltoallv_v5 MK ON (shared add_norm=A2, full wk_wq=A1, layer0=A3," \
     "router_gemm=A2, cpu_partition=${VLLM_XCPU_CPU_PARTITION}," \
     "OMP_NUM_THREADS=${OMP_NUM_THREADS}, KMP_BLOCKTIME=${KMP_BLOCKTIME})"
