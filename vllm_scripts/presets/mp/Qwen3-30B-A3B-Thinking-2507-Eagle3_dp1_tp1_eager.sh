#!/bin/bash
# Preset: Qwen3-30B-A3B-Thinking-2507 with NVIDIA Eagle3
# Configuration: DP=1, TP=1, PP=1, enforce-eager mode
# Speculative decoding: Eagle3 draft length = 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../"

# Keep the same non-MPI mp runtime shape as Qwen3-30B-A3B_dp1_tp1_eager.sh.
export PD_MODE="NOT_MOE"

# 加载基础模板配置
source "$SCRIPT_DIR/user_env_template.sh"

# 覆盖必要配置
export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"
export USER_VLLM_MAX_MODEL_LEN=2048
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=1
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
unset VLLM_ALL2ALL_BACKEND_XCPU

# NVIDIA Eagle3 head. HF_HUB_OFFLINE=1 is set by user_env_template.sh, so the
# corresponding cache entries under ~/.cache/huggingface/hub must already exist.
export USER_VLLM_EAGLE3_MODEL="nvidia/Qwen3-30B-A3B-Thinking-2507-Eagle3"
export USER_VLLM_EAGLE3_NUM_SPECULATIVE_TOKENS=3
export USER_VLLM_EAGLE3_DRAFT_TP_SIZE=1

printf -v _EAGLE3_SPECULATIVE_CONFIG \
    '{"model":"%s","draft_tensor_parallel_size":%s,"num_speculative_tokens":%s,"method":"eagle3"}' \
    "${USER_VLLM_EAGLE3_MODEL}" \
    "${USER_VLLM_EAGLE3_DRAFT_TP_SIZE}" \
    "${USER_VLLM_EAGLE3_NUM_SPECULATIVE_TOKENS}"
export VLLM_OPTIONAL_ARGS="${VLLM_OPTIONAL_ARGS} --speculative-config ${_EAGLE3_SPECULATIVE_CONFIG}"
unset _EAGLE3_SPECULATIVE_CONFIG

# 自动获取预设名称和目录
preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
if [ "$preset_dir" = "presets" ]; then
    echo "🚀 Preset: ${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
else
    echo "🚀 Preset: ${preset_dir}/${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
fi
