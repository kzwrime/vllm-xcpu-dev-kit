#!/bin/bash
# Preset: Qwen3-0.6B
# Configuration: DP=2, PP=2, TP=2, compile mode
# MPI Processes: 8 (DP * TP * PP = 2 * 2 * 2)

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export PD_MODE="NOT_MOE"

source "$SCRIPT_DIR/user_env_template.sh"

export TORCH_XCPU_ENABLE_CHECK=0
export USER_VLLM_EAGER_OR_NOT=""
export USER_VLLM_MODEL="Qwen/Qwen3-0.6B"
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=2
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
export OMP_NUM_THREADS=32
export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "🚀 Preset: ${preset_dir}/${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
