#!/bin/bash
# Qwen3-30B-A3B MXFP4A16, DP=2 TP=2 EP=4, MPI alltoallv-v2, compile.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../../"
export PD_MODE="MIXED"
source "$SCRIPT_DIR/user_env_template.sh"
export TORCH_XCPU_ENABLE_CHECK=0
export USER_VLLM_MODEL="nm-testing/Qwen3-30B-A3B-MXFP4A16"
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=2
export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
_VLLM_OPTIONAL_ARGS+=" --all2all-backend mpi_alltoallv_v2"
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"
echo "Preset: mpi/moe/$(basename "${BASH_SOURCE[0]}" .sh) | DP=2 TP=2 EP=4 compile"
