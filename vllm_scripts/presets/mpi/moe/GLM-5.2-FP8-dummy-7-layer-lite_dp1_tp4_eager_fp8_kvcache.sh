#!/bin/bash

# 7-layer dummy GLM-5.2 at TP4 (16 heads per rank): the fused MLA front half
# (fused_mla_qnorm_qb_uk_rope_cache) only instantiates for 16 heads, so TP4 is
# the smallest layout that exercises it. Random weights, tiny memory footprint;
# use it to validate the operator flow and cross-version output agreement, not
# model quality. Pick the version with VLLM_XCPU_FUSED_MLA_QNORM_QB_UK
# (0 off, unfused=A1, 1=A0/v1, 2=A2/v2, 3=A3/v3).

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export PD_MODE="NOT_MOE"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-900}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"
source "$SCRIPT_DIR/user_env_template.sh"

unset VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16
export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="../models/ZhipuAI/GLM-5.2-FP8-dummy-7-layer-lite"
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=4
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))

export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1

_VLLM_OPTIONAL_ARGS+=" --all2all-backend all_to_all_single"
_VLLM_OPTIONAL_ARGS+=" --load-format dummy"
_VLLM_OPTIONAL_ARGS+=" --kv-cache-dtype fp8"
_VLLM_OPTIONAL_ARGS+=' --kernel-config {"enable_jit_warmup":false}'
_VLLM_OPTIONAL_ARGS+=' --tool-call-parser glm47'
_VLLM_OPTIONAL_ARGS+=' --enable-auto-tool-choice'
_VLLM_OPTIONAL_ARGS+=' --reasoning-parser glm45'
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "Preset: ${preset_dir}/${preset_name} | dummy 7-layer GLM-5.2 FP8, TP4, FP8 KV cache, fused_mla_qnorm_qb_uk=${VLLM_XCPU_FUSED_MLA_QNORM_QB_UK:-0}"
