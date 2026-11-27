#!/bin/bash
# Preset: LLM-Research/Llama-3.2-1B-Instruct
# Configuration: DP=1, TP=1, PP=1, enforce-eager mode

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export PD_MODE="NOT_MOE"
_CALLER_USER_VLLM_MODEL="${USER_VLLM_MODEL:-}"

source "$SCRIPT_DIR/user_env_template.sh"

export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
_MS_CACHE_DIR="${MODELSCOPE_CACHE:-${HOME}/.cache/modelscope}"
if [ -n "${LLAMA3_2_1B_MODEL_PATH:-}" ]; then
    export USER_VLLM_MODEL="${LLAMA3_2_1B_MODEL_PATH}"
elif [ -n "${_CALLER_USER_VLLM_MODEL}" ]; then
    export USER_VLLM_MODEL="${_CALLER_USER_VLLM_MODEL}"
elif [ -d "${_MS_CACHE_DIR}/models/LLM-Research--Llama-3.2-1B-Instruct/snapshots/master" ]; then
    export USER_VLLM_MODEL="${_MS_CACHE_DIR}/models/LLM-Research--Llama-3.2-1B-Instruct/snapshots/master"
elif [ -d "${_MS_CACHE_DIR}/hub/models/LLM-Research/Llama-3.2-1B-Instruct" ]; then
    export USER_VLLM_MODEL="${_MS_CACHE_DIR}/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
else
    export USER_VLLM_MODEL="LLM-Research/Llama-3.2-1B-Instruct"
fi
unset _CALLER_USER_VLLM_MODEL _MS_CACHE_DIR
export USER_VLLM_MAX_MODEL_LEN="${USER_VLLM_MAX_MODEL_LEN:-8192}"
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=1
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
unset USER_VLLM_MP_RPC_WORKER_PER_NODE
unset VLLM_DISABLE_TQDM_AND_MONITOR

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "🚀 Preset: ${preset_dir}/${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
