#!/bin/bash
# Preset: Qwen3-30B-A3B-Instruct-2507-FP8
# Configuration: DP=2, PP=2, TP=2, EP=4, enforce-eager mode, MPI alltoallv v5
# MPI Processes: 8 (DP * TP * PP = 2 * 2 * 2; ep_size = DP * TP = 4 per PP stage)

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export PD_MODE="MIXED"
_CALLER_USER_VLLM_MODEL="${USER_VLLM_MODEL:-}"

source "$SCRIPT_DIR/user_env_template.sh"

export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
_MS_CACHE_DIR="${MODELSCOPE_CACHE:-${HOME}/.cache/modelscope}"
if [ -n "${QWEN3_30B_A3B_FP8_MODEL_PATH:-}" ]; then
    export USER_VLLM_MODEL="${QWEN3_30B_A3B_FP8_MODEL_PATH}"
elif [ -n "${_CALLER_USER_VLLM_MODEL}" ]; then
    export USER_VLLM_MODEL="${_CALLER_USER_VLLM_MODEL}"
elif [ -d "${_MS_CACHE_DIR}/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8" ]; then
    export USER_VLLM_MODEL="${_MS_CACHE_DIR}/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
elif [ -d "${_MS_CACHE_DIR}/models/Qwen--Qwen3-30B-A3B-Instruct-2507-FP8/snapshots/master" ]; then
    export USER_VLLM_MODEL="${_MS_CACHE_DIR}/models/Qwen--Qwen3-30B-A3B-Instruct-2507-FP8/snapshots/master"
else
    export USER_VLLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
fi
unset _CALLER_USER_VLLM_MODEL _MS_CACHE_DIR
export USER_VLLM_MAX_MODEL_LEN="${USER_VLLM_MAX_MODEL_LEN:-4096}"
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=2
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
export OMP_NUM_THREADS=32
export VLLM_CPU_KVCACHE_SPACE=8
export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1
_VLLM_OPTIONAL_ARGS+=" --all2all-backend mpi_alltoallv_v5"
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "🚀 Preset: ${preset_dir}/${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}, EP=4"
