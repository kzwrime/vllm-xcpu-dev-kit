#!/bin/bash

# vLLM 服务测试脚本
#
# 使用说明:
#   方式1: 通过 -e 参数指定预设文件
#     ./serve_test/serve_test_template.sh -e ./presets/serial/Qwen3-30B-A3B_dp1_tp1_eager.sh
#
#   方式2: 通过 PRESET 环境变量
#     PRESET=serial/Qwen3-30B-A3B_dp1_tp1_eager ./serve_test/serve_test_template.sh
#
#   方式3: 使用 user_env.sh
#     ./serve_test/serve_test_template.sh
#
# 功能说明:
#   向 vLLM 服务发送测试请求，验证服务是否正常工作

# 查看可用模型
# curl http://localhost:8000/v1/models

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

curl http://localhost:${USER_VLLM_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer empty" \
  -d '{
    "model": "'"${USER_VLLM_MODEL}"'",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "请用一段话简单介绍一下量子计算。"}
    ],
    "max_tokens": 16,
    "temperature": 0.5
  }'
