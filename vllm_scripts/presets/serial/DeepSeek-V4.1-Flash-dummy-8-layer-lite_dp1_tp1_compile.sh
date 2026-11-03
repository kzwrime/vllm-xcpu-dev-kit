#!/bin/bash

# DeepSeek-V4.1-Flash, 8 dummy layers covering every layer kind (SWA, ratio-2
# and ratio-1 kv/index sources and consumers, candidate source, engram) plus
# the 3 DSpark layers. Records: MXFP8 SWA, NVFP4 compressed, MXFP4 indexer.

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export PD_MODE="NOT_MOE"
# v0.31 auto-selects breakable CUDA graphs for DeepSeek-V4.1.
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export USER_VLLM_MAX_NUM_BATCHED_TOKENS="${USER_VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-1800}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
source "$SCRIPT_DIR/user_env_template.sh"
unset USER_VLLM_MP_RPC_WORKER_PER_NODE
unset VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16

export USER_VLLM_EAGER_OR_NOT=""
export TORCH_XCPU_ENABLE_CHECK=0
export USER_VLLM_MODEL="../models/deepseek-ai/DeepSeek-V4.1-Flash-dummy-8-layer-lite"
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=1
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))

_VLLM_OPTIONAL_ARGS+=" --all2all-backend all_to_all_single"
_VLLM_OPTIONAL_ARGS+=" --load-format dummy"
_VLLM_OPTIONAL_ARGS+=" --block-size 128"
_VLLM_OPTIONAL_ARGS+=' --attention-config {"indexer_kv_dtype":"mxfp4"}'
_VLLM_OPTIONAL_ARGS+=' --limit-mm-per-prompt {"image":0}'
_VLLM_OPTIONAL_ARGS+=' --kernel-config {"enable_jit_warmup":false}'
_VLLM_OPTIONAL_ARGS+=' --compilation-config {"mode":3}'
_VLLM_OPTIONAL_ARGS+=' --tokenizer-mode deepseek_v41'
_VLLM_OPTIONAL_ARGS+=' --reasoning-parser deepseek_v41'
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "🚀 Preset: ${preset_dir}/${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
