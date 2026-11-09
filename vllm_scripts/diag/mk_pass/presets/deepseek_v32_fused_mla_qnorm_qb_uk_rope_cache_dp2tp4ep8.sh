#!/bin/bash

# 阶段 1(MLA 融合迁移到 compile pass)端到端验证:
#   在 phase0 的 unfused 基线之上打开融合(compile pass env=1)并选中 qnorm 的
#   mega-kernel v3(IMPL env=A3):plugin env 只管是否融合(0/1),torch_xcpu 的
#   IMPL env 管算子内部用哪个实现。每层 MLA 前半应替换为
#   fused_mla_qnorm_qb_uk_rope_cache("fused 7 MLA front halves")。

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
_DIAG_CACHE_ROOT="${SCRIPT_DIR%/presets/*}/cache"

source "${SCRIPT_DIR%/diag*}/presets/mpi/moe/GLM-5.2-FP8-dummy-7-layer-lite_dp2_tp4_ep8_eager_fp8_kvcache_alltoallv_v5.sh"

export USER_VLLM_EAGER_OR_NOT=""
export VLLM_XCPU_FUSED_MLA=0
export VLLM_XCPU_FUSED_MLA_VPROJ=0
export VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE=1
export TORCH_XCPU_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE_IMPL=A3
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-1500}"
export VLLM_XCPU_DUMP_POST_GRAD_GRAPH="${_DIAG_CACHE_ROOT}/post_grad_dump/graph"
mkdir -p "${_DIAG_CACHE_ROOT}/post_grad_dump"

export VLLM_CACHE_ROOT="${_DIAG_CACHE_ROOT}/vllm_cache"
export TORCHINDUCTOR_CACHE_DIR="${_DIAG_CACHE_ROOT}/inductor"
unset _DIAG_CACHE_ROOT

echo "Preset: diag/mk_pass/deepseek_v32_fused_mla_qnorm_qb_uk_rope_cache_dp2tp4ep8 | compile pass on + qnorm impl A3," \
     "DP2 TP4 EP8, isolated cache=${TORCHINDUCTOR_CACHE_DIR}"
