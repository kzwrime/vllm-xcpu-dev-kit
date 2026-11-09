#!/bin/bash

# GLM-5.2-FP8_dp1_tp4_ep4_compile_sparse_attn_fp8_kvcache_dflash2_5 with the fused layer
# front (fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache, mega-kernel v1 / impl A1) enabled:
# input_layernorm + fused_qkv_a_proj + q/kv norm + q_b_proj + W_UK_T + RoPE + KV-cache
# write as one torch_xcpu op. The plugin's compile-pass env only decides whether the
# graph is fused (0/1); torch_xcpu's IMPL env picks the mega-kernel (A1; _mk_v2.sh = A2).
#
# Threads: the mega-kernel launches 96 OpenMP threads (mk::arch::kNumPe), TP4 ranks get
# OMP_NUM_THREADS=64 each on a 256-thread machine — deliberate oversubscription, kept in
# check by VLLM_XCPU_CPU_PARTITION=1 (per-rank CPU slice -> 1.5x intra-slice) and
# KMP_BLOCKTIME=1 (no 200 ms spin after the parallel region). OMP_THREAD_LIMIT must stay
# unset / >= 96: a short team makes the op fall back to the unfused sequence (logged, not
# a crash).

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/GLM-5.2-FP8_dp1_tp4_ep4_compile_sparse_attn_fp8_kvcache_dflash2_5.sh"

export VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE="${VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE:-1}"
export TORCH_XCPU_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_IMPL="${TORCH_XCPU_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_IMPL:-A1}"
export VLLM_XCPU_CPU_PARTITION="${VLLM_XCPU_CPU_PARTITION:-1}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-1}"
unset OMP_THREAD_LIMIT

echo "Preset: fused layer front ON (add_norm mega-kernel v1, impl A1)" \
     "(cpu_partition=${VLLM_XCPU_CPU_PARTITION}, OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}," \
     "KMP_BLOCKTIME=${KMP_BLOCKTIME})"
