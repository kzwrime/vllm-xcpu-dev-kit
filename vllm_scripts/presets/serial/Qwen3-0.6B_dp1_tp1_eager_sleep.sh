#!/bin/bash
# Preset: Qwen3-0.6B with vLLM sleep mode (accelerator release / reacquire)
# Configuration: DP=1, TP=1, PP=1, enforce-eager mode
# Test: ./run_vllm_test.sh -e presets/serial/Qwen3-0.6B_dp1_tp1_eager_sleep.sh --sleep-test

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

# 在加载模板前设置独立配置项
export PD_MODE="NOT_MOE"

# 加载基础模板配置
source "$SCRIPT_DIR/user_env_template.sh"

# 覆盖必要配置
export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="Qwen/Qwen3-0.6B"
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=1
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
# Use the local mp executor for this single-rank preset.
unset USER_VLLM_MP_RPC_WORKER_PER_NODE
unset VLLM_DISABLE_TQDM_AND_MONITOR

# /sleep, /wake_up and /is_sleeping live under the dev routers.
export VLLM_SERVER_DEV_MODE=1
_VLLM_OPTIONAL_ARGS+=" --enable-sleep-mode"
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

# 自动获取预设名称和目录
preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
if [ "$preset_dir" = "presets" ]; then
    echo "🚀 Preset: ${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
else
    echo "🚀 Preset: ${preset_dir}/${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
fi
