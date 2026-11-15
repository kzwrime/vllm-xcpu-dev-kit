#!/bin/bash

# GLM-5.2-FP8_dp1_tp4_ep4_compile_sparse_attn_fp8_kvcache_dflash2_5 with the fused layer
# front enabled: input_layernorm + fused_qkv_a_proj + q/kv norm + q_b_proj + W_UK_T +
# RoPE + KV-cache write as one torch_xcpu op. 两个互不相干的算子按层各走各的：
#   - shared 层（无 sparse indexer）→ fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache，
#     mega-kernel 由 TORCH_XCPU_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_IMPL 选
#     （A1；_mk_v2.sh = A2）
#   - full 层（带 sparse indexer）→ fused_mla_add_norm_qkva_wk_wq_qnorm_qb_uk_rope_cache
#     （wk/wq 两个投影在同一个 mega-kernel 里），由自己的
#     TORCH_XCPU_FUSED_MLA_ADD_NORM_QKVA_WK_WQ_QNORM_QB_UK_ROPE_CACHE_IMPL 选（默认 A1）
# The plugin's compile-pass env only decides whether the graph is fused (0/1).
#
# Threads: the PE gate is evaluated once at torch_xcpu import as
# omp_get_max_threads() >= mk::arch::kNumPe (= 96), so OMP_NUM_THREADS must be 96 here.
# env.sh 全局 export 的是 64（TP4 在 256 CPU 机器上每 rank 刚好占满）—— 不覆盖它团队就被
# 压到 96 以下，算子静默退回逐算子序列（worker 日志有 "needs 96 PEs" 提示，不是 crash）。
# VLLM_XCPU_CPU_PARTITION=1 把每个 rank 钉在 64 CPU 的切片上，96 线程 = 切片内 1.5 倍超订
# （serve_mp_rpc_all_mpi_template.sh 注释里的设计值）；KMP_BLOCKTIME=1 避免 parallel region
# 后 200 ms 自旋。OMP_THREAD_LIMIT 必须保持 unset / >= 96。

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/GLM-5.2-FP8_dp1_tp4_ep4_compile_sparse_attn_fp8_kvcache_dflash2_5.sh"

export VLLM_XCPU_COMPILE_PASS_FUSED_ADD_NORM_ROUTER_GEMM="${VLLM_XCPU_COMPILE_PASS_FUSED_ADD_NORM_ROUTER_GEMM:-1}"
export TORCH_XCPU_FUSED_ADD_NORM_ROUTER_GEMM_IMPL="${TORCH_XCPU_FUSED_ADD_NORM_ROUTER_GEMM_IMPL:-A2}"

export VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE="${VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE:-1}"
export TORCH_XCPU_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_IMPL="${TORCH_XCPU_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_IMPL:-A2}"

export VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE="${VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE:-1}"
export TORCH_XCPU_FUSED_MLA_ADD_NORM_QKVA_WK_WQ_QNORM_QB_UK_ROPE_CACHE_IMPL="${TORCH_XCPU_FUSED_MLA_ADD_NORM_QKVA_WK_WQ_QNORM_QB_UK_ROPE_CACHE_IMPL:-A1}"

if [[ "${VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE}" == "1" || \
      "${VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE}" == "1" ]]; then
    export VLLM_XCPU_FUSED_MLA=0
fi

# 这几行是用于普通多核 CPU 上模拟验证所需的设置
# 因为 torch_xcpu 的 Mega-Kernel 硬性规定了线程数 96，核数不够时会回退成非融合版本
# 这里必须无条件覆盖 env.sh 达到足够的核数
export OMP_NUM_THREADS=96
unset OMP_THREAD_LIMIT
export VLLM_XCPU_CPU_PARTITION="${VLLM_XCPU_CPU_PARTITION:-1}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-1}"

echo "Preset: fused layer front ON (shared 层 add_norm mk v1 = A1, full 层 wk_wq op = A1)" \
     "(cpu_partition=${VLLM_XCPU_CPU_PARTITION}, OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}," \
     "KMP_BLOCKTIME=${KMP_BLOCKTIME})"
