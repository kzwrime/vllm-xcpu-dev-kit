#!/bin/bash
# Two-layer MXFP4 dummy smoke preset; the cropped config is staged beside models.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../"
export PD_MODE="NOT_MOE"
source "$SCRIPT_DIR/user_env_template.sh"
export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="../models/nm-testing/Qwen3-30B-A3B-MXFP4A16-dummy-2-layer"
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=1
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=1
_VLLM_OPTIONAL_ARGS+=" --load-format dummy"
echo "Preset: serial/$(basename "${BASH_SOURCE[0]}" .sh) | DP=1 TP=1 dummy"
