#!/bin/bash
# Preset: Qwen3.6-35B-A3B-FP8
# Configuration: DP=2, TP=2, EP=4, PP=1, enforce-eager, alltoallv_v5, MPI EPLB

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export PD_MODE="MIXED"
source "$SCRIPT_DIR/user_env_template.sh"

export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1
export VLLM_XCPU_GDN_DECODE_ONLY_COMPILE=1

_VLLM_OPTIONAL_ARGS+=" --all2all-backend mpi_alltoallv_v5"
_VLLM_OPTIONAL_ARGS+=" --reasoning-parser qwen3 --language-model-only"
_VLLM_OPTIONAL_ARGS+=" --enable-eplb"
_VLLM_OPTIONAL_ARGS+=" --eplb-config.use_async false"
_VLLM_OPTIONAL_ARGS+=" --eplb-config.communicator mpi"
_VLLM_OPTIONAL_ARGS+=" --eplb-config.num_redundant_experts 4"
_VLLM_OPTIONAL_ARGS+=" --eplb-config.window_size 16"
_VLLM_OPTIONAL_ARGS+=" --eplb-config.step_interval 20"
_VLLM_OPTIONAL_ARGS+=" --eplb-config.log_balancedness true"
_VLLM_OPTIONAL_ARGS+=" --eplb-config.log_balancedness_interval 10"
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"
