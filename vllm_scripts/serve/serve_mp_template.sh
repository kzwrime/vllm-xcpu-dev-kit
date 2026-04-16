#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载通用函数
ENV_FILE="$SCRIPT_DIR/../common.sh"
if [ -f "$ENV_FILE" ]; then
    echo "loading env file: $ENV_FILE"
    source "$ENV_FILE"
else
    echo "ERROR ! Could not find $ENV_FILE"
    exit 1
fi

# 解析命令行参数并加载环境配置
parse_args_and_load_env "$SCRIPT_DIR/.." "$@"

echo "--- 📝 vLLM 服务配置参数检查与设置 ---"

echo "--- 必需参数 ---"
check_and_print_env "USER_VLLM_MODEL"
check_and_print_env "USER_VLLM_LOGGING_LEVEL"
check_and_print_env "USER_VLLM_MAX_MODEL_LEN"
check_and_print_env "USER_VLLM_MAX_NUM_BATCHED_TOKENS"
check_and_print_env "USER_VLLM_DATA_PARALLEL_SIZE"
check_and_print_env "USER_VLLM_TP_SIZE"
check_and_print_env "USER_VLLM_PORT"
check_and_print_env "VLLM_LOOPBACK_IP"

echo ""
echo "--- 🚀 正在启动 vLLM 服务... ---"

# --- C. 执行 vLLM 命令 ---

# 启动 vLLM 服务，使用参数化的环境变量
VLLM_LOGGING_LEVEL=${USER_VLLM_LOGGING_LEVEL} vllm serve ${USER_VLLM_MODEL} \
  --max-model-len ${USER_VLLM_MAX_MODEL_LEN} \
  --max-num-batched-tokens ${USER_VLLM_MAX_NUM_BATCHED_TOKENS} \
  -tp=${USER_VLLM_TP_SIZE} \
  -pp=${USER_VLLM_PP_SIZE} \
  --distributed-executor-backend mp \
  --port ${USER_VLLM_PORT} \
  ${USER_VLLM_EAGER_OR_NOT} \
  ${VLLM_OPTIONAL_ARGS} \
  --tensor-parallel-size ${USER_VLLM_TP_SIZE} \
  --data-parallel-size ${USER_VLLM_DATA_PARALLEL_SIZE} 2>&1 | tee logs/vllm_serve_log.txt

# 检查 vLLM 命令的退出状态
if [ $? -ne 0 ]; then
    echo "❌ 错误：vllm serve 命令执行失败。" >&2
    exit 1
fi
