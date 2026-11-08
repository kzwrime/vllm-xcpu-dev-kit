#!/bin/bash

# 阶段 1(MLA 融合迁移到 compile pass)端到端验证:
#   在 phase0 的 unfused 基线之上打开 VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE=3,
#   post-grad pattern pass 应把每层 MLA 前半(q/kv RMSNorm + q_b_proj + W_UK_T
#   + RoPE + cache write)替换为 fused_mla_qnorm_qb_uk_rope_cache。
# 预期日志(每 rank 一条):
#   "XCPU post-grad pattern pass installed (VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE=3)"
#   "XCPU post-grad pass fused 7 MLA front halves"
# pass 的 uuid 参与编译 cache key,会在共享的隔离 cache 中新增条目,不影响
# phase0 产物。VLLM_XCPU_DUMP_POST_GRAD_GRAPH 输出 pass 前后的 FX 图供离线比对。

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
_DIAG_CACHE_ROOT="${SCRIPT_DIR%/presets/*}/cache"

source "${SCRIPT_DIR%/diag*}/presets/mpi/moe/GLM-5.2-FP8-dummy-7-layer-lite_dp2_tp4_ep8_eager_fp8_kvcache_alltoallv_v5.sh"

export USER_VLLM_EAGER_OR_NOT=""
export VLLM_XCPU_FUSED_MLA=0
export VLLM_XCPU_FUSED_MLA_VPROJ=0
export VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE=3
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-1500}"
export VLLM_XCPU_DUMP_POST_GRAD_GRAPH="${_DIAG_CACHE_ROOT}/post_grad_dump/graph"
mkdir -p "${_DIAG_CACHE_ROOT}/post_grad_dump"

export VLLM_CACHE_ROOT="${_DIAG_CACHE_ROOT}/vllm_cache"
export TORCHINDUCTOR_CACHE_DIR="${_DIAG_CACHE_ROOT}/inductor"
unset _DIAG_CACHE_ROOT

echo "Preset: diag/mk_pass/deepseek_v32_fused_mla_qnorm_qb_uk_rope_cache_dp2tp4ep8 | compile pass A3," \
     "DP2 TP4 EP8, isolated cache=${TORCHINDUCTOR_CACHE_DIR}"
