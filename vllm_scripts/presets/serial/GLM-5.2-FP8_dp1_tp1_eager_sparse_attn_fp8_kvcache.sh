#!/bin/bash

# FP8 checkpoint with sparse MLA, prefix caching and FP8 KV cache.
# Companion to the MXFP4 validation preset; FP8 has not been validated here.

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export PD_MODE="NOT_MOE"
source "$SCRIPT_DIR/user_env_template.sh"

# Do not carry the MXFP4-only activation override into this preset.
unset VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-1800}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export USER_VLLM_MAX_NUM_BATCHED_TOKENS=2048
export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="ZhipuAI/GLM-5.2-FP8"
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=1
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))

_VLLM_OPTIONAL_ARGS=" --max-num-seqs ${USER_VLLM_MAX_NUM_SEQS}"
_VLLM_OPTIONAL_ARGS+=" --use-fp64-gumbel"
_VLLM_OPTIONAL_ARGS+=" --enable-prefix-caching"
_VLLM_OPTIONAL_ARGS+=" --all2all-backend all_to_all_single"
_VLLM_OPTIONAL_ARGS+=" --kv-cache-dtype fp8"
_VLLM_OPTIONAL_ARGS+=' --kernel-config {"enable_jit_warmup":false}'
_VLLM_OPTIONAL_ARGS+=' --tool-call-parser glm47'
_VLLM_OPTIONAL_ARGS+=' --enable-auto-tool-choice'
_VLLM_OPTIONAL_ARGS+=' --reasoning-parser glm45'
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "Preset: ${preset_dir}/${preset_name} | FP8 checkpoint, FP8 KV cache"
