#!/bin/bash
# Qwen3-30B-A3B FP8, DP=2 TP=2 EP=4, MPI alltoallv-v2, eager.
SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"
export PD_MODE="MIXED"
source "$SCRIPT_DIR/user_env_template.sh"
export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=2
export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1
_VLLM_OPTIONAL_ARGS+=" --all2all-backend mpi_alltoallv_v2"
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"
echo "Preset: mpi/moe/$(basename "${BASH_SOURCE[0]}" .sh) | DP=2 TP=2 EP=4"
