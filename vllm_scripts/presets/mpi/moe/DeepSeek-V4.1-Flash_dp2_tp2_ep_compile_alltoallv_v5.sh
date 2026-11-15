#!/bin/bash

# Real-weight DeepSeek-V4.1-Flash on DP2/TP2 with expert parallelism over
# mpi_alltoallv_v6 and MoE sequence parallelism: 32 local heads / 4 wo_a groups
# per TP rank, engram hash heads sharded over TP, experts sharded over DP x TP.

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export PD_MODE="MIXED"
# v0.31 auto-selects breakable CUDA graphs for DeepSeek-V4.1.
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export USER_VLLM_MAX_NUM_BATCHED_TOKENS="${USER_VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-3600}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
source "$SCRIPT_DIR/user_env_template.sh"
unset VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16

export USER_VLLM_EAGER_OR_NOT=""
export TORCH_XCPU_ENABLE_CHECK=0
export USER_VLLM_MODEL="${HOME}/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master"
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))

export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1

_VLLM_OPTIONAL_ARGS+=" --all2all-backend mpi_alltoallv_v5"
_VLLM_OPTIONAL_ARGS+=" --block-size 128"
_VLLM_OPTIONAL_ARGS+=' --attention-config {"indexer_kv_dtype":"mxfp4"}'
_VLLM_OPTIONAL_ARGS+=' --limit-mm-per-prompt {"image":0}'
_VLLM_OPTIONAL_ARGS+=' --kernel-config {"enable_jit_warmup":false}'
_VLLM_OPTIONAL_ARGS+=' --tokenizer-mode deepseek_v41'
_VLLM_OPTIONAL_ARGS+=' --reasoning-parser deepseek_v41'
_VLLM_OPTIONAL_ARGS+=' --compilation-config {"mode":3}'
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "🚀 Preset: ${preset_dir}/${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
