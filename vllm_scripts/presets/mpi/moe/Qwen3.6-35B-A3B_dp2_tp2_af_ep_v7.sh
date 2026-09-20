#!/bin/bash

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export PD_MODE="MIXED"
source "$SCRIPT_DIR/user_env_template.sh"

export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="${QWEN3_6_35B_A3B_BF16_MODEL_PATH:-Qwen/Qwen3.6-35B-A3B}"
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_XCPU_ENABLE_AF_EP=1
export USER_VLLM_EP_SIZE="${USER_VLLM_EP_SIZE:-2}"
export USER_VLLM_MPI_SIZE=$((USER_VLLM_DATA_PARALLEL_SIZE * USER_VLLM_MPC_SIZE + USER_VLLM_EP_SIZE))
export VLLM_MPI_WORKER_TEMPLATE="$SCRIPT_DIR/serve/serve_afd_mp_rpc_all_mpi_template.sh"

# V7 is the explicit remote-experts backend; colocated EP keeps V6.
_VLLM_OPTIONAL_ARGS+=" --all2all-backend mpi_alltoallv_v7"
_VLLM_OPTIONAL_ARGS+=" --reasoning-parser qwen3 --language-model-only"
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

export VLLM_XCPU_GDN_DECODE_ONLY_COMPILE=1

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
echo "AF-EP preset: ${preset_name} A=4 F=${USER_VLLM_EP_SIZE} ModelRunner=V2 eager BF16"
