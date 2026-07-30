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

# 如果需要使用 profiler 分析，取消注释以发送 start_profile / stop_profile

# curl --silent --show-error --fail-with-body \
#   -X POST "http://localhost:${USER_VLLM_PORT}/start_profile"

curl --silent --show-error --fail-with-body \
  --write-out '\ncurl_time_total_seconds=%{time_total}\n' \
  "http://localhost:${USER_VLLM_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer empty" \
  -d '{
    "model": "'"${USER_VLLM_MODEL}"'",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "请用一段话简单介绍一下量子计算。"}
    ],
    "chat_template_kwargs": {
      "enable_thinking": false
    },
    "max_tokens": 16,
    "temperature": 0.7
  }'

# curl --silent --show-error --fail-with-body \
#   -X POST "http://localhost:${USER_VLLM_PORT}/stop_profile"

# curl --silent --show-error \
#   --write-out '\ncurl_time_total_seconds=%{time_total}\n' \
#   "http://localhost:${USER_VLLM_PORT}/v1/chat/completions" \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer empty" \
#   -d '{
#     "model": "'"${USER_VLLM_MODEL}"'",
#     "messages": [
#       {"role": "system", "content": "You are a helpful assistant."},
#       {"role": "user", "content": "请用一段话简单介绍一下量子计算。"}
#     ],
#     "max_tokens": 16,
#     "temperature": 0.5
#   }'


# curl --silent --show-error \
#   --write-out '\ncurl_time_total_seconds=%{time_total}\n' \
#   "http://localhost:${USER_VLLM_PORT}/v1/chat/completions" \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer empty" \
#   -d '{
#     "model": "'"${USER_VLLM_MODEL}"'",
#     "messages": [
#       {"role": "system", "content": "You are a helpful assistant."},
#       {"role": "user", "content": "监狱决定给关押的100名囚徒一次特赦的机会，条件是囚徒通过一项挑战。所有囚徒被编号为1-100，对应他们编号的100个号码牌被打乱顺序放在了100个抽屉里。每个囚徒需要从所有抽屉里打开至多半数(50个)，并从中找出对应自己编号的号码牌。如果找到了则该名囚徒的任务成功。所有囚徒会依次单独进入挑战室完成任务，并且从第一个囚徒进入挑战室开始，直到所有囚徒结束挑战为止囚徒之间任何形式的交流都是禁止的。当一名囚徒完成任务后，挑战室会被恢复为他进入之前的样子（号码牌当然也放回原来的抽屉里）。在这100名囚徒中，任意一名囚徒的失败都会导致整个挑战失败，只有当所有囚徒全部成功完成任务时，他们才会统一得到特赦的机会。最后，在开始挑战之前，监狱给了所有囚徒一个月时间商量对策。那么，囚徒究竟有多大的几率得到释放？监狱决定给关押的100名囚徒一次特赦的机会，条件是囚徒通过一项挑战。所有囚徒被编号为1-100，对应他们编号的100个号码牌被打乱顺序放在了100个抽屉里。每个囚徒需要从所有抽屉里打开至多半数(50个)，并从中找出对应自己编号的号码牌。如果找到了则该名囚徒的任务成功。所有囚徒会依次单独进入挑战室完成任务，并且从第一个囚徒进入挑战室开始，直到所有囚徒结束挑战为止囚徒之间任何形式的交流都是禁止的。当一名囚徒完成任务后，挑战室会被恢复为他进入之前的样子（号码牌当然也放回原来的抽屉里）。在这100名囚徒中，任意一名囚徒的失败都会导致整个挑战失败，只有当所有囚徒全部成功完成任务时，他们才会统一得到特赦的机会。最后，在开始挑战之前，监狱给了所有囚徒一个月时间商量对策。那么，囚徒究竟有多大的几率得到释放？监狱决定给关押的100名囚徒一次特赦的机会，条件是囚徒通过一项挑战。所有囚徒被编号为1-100，对应他们编号的100个号码牌被打乱顺序放在了100个抽屉里。每个囚徒需要从所有抽屉里打开至多半数(50个)，并从中找出对应自己编号的号码牌。如果找到了则该名囚徒的任务成功。所有囚徒会依次单独进入挑战室完成任务，并且从第一个囚徒进入挑战室开始，直到所有囚徒结束挑战为止囚徒之间任何形式的交流都是禁止的。当一名囚徒完成任务后，挑战室会被恢复为他进入之前的样子（号码牌当然也放回原来的抽屉里）。在这100名囚徒中，任意一名囚徒的失败都会导致整个挑战失败，只有当所有囚徒全部成功完成任务时，他们才会统一得到特赦的机会。最后，在开始挑战之前，监狱给了所有囚徒一个月时间商量对策。那么，囚徒究竟有多大的几率得到释放？"}
#     ],
#     "max_tokens": 50,
#     "temperature": 0.5
#   }'


# curl --silent --show-error \
#   --write-out '\ncurl_time_total_seconds=%{time_total}\n' \
#   "http://localhost:${USER_VLLM_PORT}/v1/chat/completions" \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer empty" \
#   -d '{
#     "model": "'"${USER_VLLM_MODEL}"'",
#     "messages": [
#       {"role": "system", "content": "You are a helpful assistant."},
#       {"role": "user", "content": "欧盟有多少个国家，详细展开论述欧盟现状。"}
#     ],
#     "max_tokens": 5000,
#     "temperature": 0
#   }'

# curl --silent --show-error \
#   --write-out '\ncurl_time_total_seconds=%{time_total}\n' \
#   "http://localhost:${USER_VLLM_PORT}/v1/chat/completions" \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer empty" \
#   -d '{
#     "model": "'"${USER_VLLM_MODEL}"'",
#     "messages": [
#       {"role": "system", "content": "You are a helpful assistant."},
#       {"role": "user", "content": "量子计算是一种基于量子力学原理的全新信息处理模式，它利用量子比特（qubit）代替传统计算机使用的二进制比特，并通过量子叠加态和量子纠缠等现象实现远超经典计算机的>计算潜力。我希望你能用一段话简要介绍量子计算的基本概念，包括它的核心原理、关键技术特征（如叠加态、纠缠、干涉等）、目前的主要实现方式（例如超导量子比特、离子阱、光子量子计算等），以及它在科学研>究和产业应用中的潜在价值。此外，请结合当前量子计算的发展现状，谈谈主要技术挑战（如量子退相干、误差校正、规模化等）和未来可能的突破方向。最后，如果可以的话，请简要说明量子计算与经典计算的区别，>并举例说明在某些领域（如密码学、药物研发、人工智能、金融建模等）中，它可能带来的变革性影响。这样的一段话，既要通俗易懂，又要涵盖技术与应用的要点，以便读者在短时间内建立起对量子计算的整体认知。"}
#     ],
#     "max_tokens": 50,
#     "temperature": 0
#   }'
