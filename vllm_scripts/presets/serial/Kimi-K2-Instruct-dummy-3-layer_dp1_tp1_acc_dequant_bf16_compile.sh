#!/bin/bash
# Preset: moonshotai/Kimi-K2-Instruct-dummy-3-layer
# Configuration: DP=1, TP=1, PP=1, Inductor C++ wrapper, ACC FP8->BF16 dequant linear + FP8 KV cache

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/..")"

export PD_MODE="MIXED"
export USER_VLLM_MAX_NUM_BATCHED_TOKENS="${USER_VLLM_MAX_NUM_BATCHED_TOKENS:-32}"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-1800}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"

source "$SCRIPT_DIR/user_env_template.sh"

unset USER_VLLM_EAGER_OR_NOT
export TORCH_XCPU_ENABLE_CHECK=0
export USER_VLLM_MODEL="${REPO_ROOT}/models/moonshotai/Kimi-K2-Instruct-dummy-3-layer"
export USER_VLLM_MAX_MODEL_LEN="${USER_VLLM_MAX_MODEL_LEN:-4096}"
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=1
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
export VLLM_CPU_KVCACHE_SPACE=8
export TORCH_XCPU_FP8_LINEAR_BACKEND=acc
export TORCH_XCPU_BF16_LINEAR_BACKEND=acc
export TORCH_XCPU_MOE_GROUPED_GEMM_BACKEND=acc
export TORCH_XCPU_ACC_FP8_DEQUANT_BF16_LINEAR=1
unset USER_VLLM_MP_RPC_WORKER_PER_NODE
unset VLLM_DISABLE_TQDM_AND_MONITOR
export TORCHINDUCTOR_CPP_WRAPPER=1

_VLLM_OPTIONAL_ARGS+=" --load-format dummy --trust-remote-code --kv-cache-dtype fp8"
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "🚀 Preset: ${preset_dir}/${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}, ACC_DEQUANT_BF16=${TORCH_XCPU_ACC_FP8_DEQUANT_BF16_LINEAR}"
