#!/bin/bash
# Preset: moonshotai/Kimi-K2-Instruct-dummy-3-layer
# Configuration: DP=2, TP=2, EP=4, PP=1, enforce-eager mode, ACC FP8->BF16 dequant linear + FP8 KV cache
# MPI Processes: 4 (DP * TP * PP = 2 * 2 * 1)

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/..")"

export PD_MODE="MIXED"

source "$SCRIPT_DIR/user_env_template.sh"

export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="${REPO_ROOT}/models/moonshotai/Kimi-K2-Instruct-dummy-3-layer"
export USER_VLLM_MAX_MODEL_LEN="${USER_VLLM_MAX_MODEL_LEN:-4096}"
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
export VLLM_CPU_KVCACHE_SPACE=8
export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1
export TORCH_XCPU_FP8_LINEAR_BACKEND=acc
export TORCH_XCPU_BF16_LINEAR_BACKEND=acc
export TORCH_XCPU_MOE_GROUPED_GEMM_BACKEND=acc
export TORCH_XCPU_ACC_FP8_DEQUANT_BF16_LINEAR=1

_VLLM_OPTIONAL_ARGS+=" --load-format dummy --trust-remote-code --kv-cache-dtype fp8 --all2all-backend mpi_alltoallv_v5"
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "🚀 Preset: ${preset_dir}/${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}, EP=4, ACC_DEQUANT_BF16=${TORCH_XCPU_ACC_FP8_DEQUANT_BF16_LINEAR}"
