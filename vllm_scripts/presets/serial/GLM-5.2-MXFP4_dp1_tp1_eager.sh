#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../"

export PD_MODE="NOT_MOE"
source "$SCRIPT_DIR/user_env_template.sh"

export TORCH_XCPU_ENABLE_CHECK=0
# The published AMD checkpoint declares dynamic MXFP4 activations (W4A4).
# XCPU currently consumes the same packed weights with BF16 activations (W4A16).
export VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16=1
# Full 78-layer construction plus 282 local shards exceeds the launcher default.
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-1200}"
export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="amd/GLM-5.2-MXFP4"
export USER_VLLM_MAX_MODEL_LEN=4096
export USER_VLLM_MAX_NUM_BATCHED_TOKENS=32
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=1
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))

_VLLM_OPTIONAL_ARGS+=" --all2all-backend all_to_all_single"
_VLLM_OPTIONAL_ARGS+=' --kernel-config {"enable_jit_warmup":false}'
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "🚀 Preset: ${preset_dir}/${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
