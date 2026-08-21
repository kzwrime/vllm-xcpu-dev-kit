#!/bin/bash
# Preset: Qwen3.6-35B-A3B with DFlash
# Configuration: DP=1, TP=2, PP=1, enforce-eager mode

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"
export PD_MODE="NOT_MOE"
source "$SCRIPT_DIR/user_env_template.sh"

export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="Qwen/Qwen3.6-35B-A3B"
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1
_VLLM_OPTIONAL_ARGS+=" --all2all-backend all_to_all_single"

_VLLM_OPTIONAL_ARGS+=" --reasoning-parser qwen3 --language-model-only"
_VLLM_OPTIONAL_ARGS+=' --speculative-config {"model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":4,"method":"dflash"}'
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "🚀 Preset: ${preset_dir}/${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
