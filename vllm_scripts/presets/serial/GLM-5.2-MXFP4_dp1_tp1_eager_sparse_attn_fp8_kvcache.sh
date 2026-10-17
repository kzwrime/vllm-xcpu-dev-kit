#!/bin/bash

# Real-weight GLM5.2 serial preset with FP8 KV cache.
# Smoke evaluation: serve_test/test_glm52_serial.py against the running server.

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export PD_MODE="NOT_MOE"
# v0.31 auto-selects breakable CUDA graphs for DSA, which disables compile.
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export USER_VLLM_MAX_NUM_BATCHED_TOKENS="${USER_VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-1800}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
source "$SCRIPT_DIR/user_env_template.sh"
unset USER_VLLM_MP_RPC_WORKER_PER_NODE

# The published AMD checkpoint declares dynamic MXFP4 activations (W4A4).
# XCPU currently consumes the same packed weights with BF16 activations (W4A16).
export VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16=1
export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="amd/GLM-5.2-MXFP4"
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=1
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))

_VLLM_OPTIONAL_ARGS+=" --all2all-backend all_to_all_single"
_VLLM_OPTIONAL_ARGS+=" --kv-cache-dtype fp8"
_VLLM_OPTIONAL_ARGS+=' --kernel-config {"enable_jit_warmup":false}'
_VLLM_OPTIONAL_ARGS+=' --tool-call-parser glm47'
_VLLM_OPTIONAL_ARGS+=' --enable-auto-tool-choice'
_VLLM_OPTIONAL_ARGS+=' --reasoning-parser glm45'
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "Preset: ${preset_dir}/${preset_name} | real MXFP4 weights, BF16 activations, FP8 KV cache"
