#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../"

export PD_MODE="NOT_MOE"
source "$SCRIPT_DIR/user_env_template.sh"

export TORCH_XCPU_ENABLE_CHECK=0
export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="../models/ZhipuAI/GLM-5.2-MXFP4-W4A16-dummy-7-layer-lite"
export USER_VLLM_MAX_MODEL_LEN=512
export USER_VLLM_MAX_NUM_BATCHED_TOKENS=32
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=1
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))

_VLLM_OPTIONAL_ARGS+=" --all2all-backend all_to_all_single"
_VLLM_OPTIONAL_ARGS+=" --load-format dummy"
_VLLM_OPTIONAL_ARGS+=' --kernel-config {"enable_jit_warmup":false}'
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "Preset: ${preset_dir}/${preset_name} | DP=1 TP=1 Quark MXFP4 W4A16 dummy lite"
