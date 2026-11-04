#!/bin/bash

# vLLM 服务测试脚本
#
# 使用说明:
#   方式1: 通过 -e 参数指定预设文件
#     ./serve_test/serve_bench_template.sh -e ./presets/serial/Qwen3-30B-A3B_dp1_tp1_eager.sh
#
#   方式2: 通过 PRESET 环境变量
#     PRESET=serial/Qwen3-30B-A3B_dp1_tp1_eager ./serve_test/serve_bench_template.sh
#
#   方式3: 使用 user_env.sh
#     ./serve_test/serve_bench_template.sh
#
# 功能说明:
#   向 vLLM 服务发送测试请求，验证服务是否正常工作

# 查看可用模型
# curl http://localhost:8000/v1/models

# Keep the bench script root separate from SCRIPT_DIR. Presets loaded below
# historically reuse SCRIPT_DIR for their own paths, which would otherwise
# make dataset paths depend on the selected preset directory.
BENCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载通用函数
ENV_FILE="$BENCH_SCRIPT_DIR/../common.sh"
if [ -f "$ENV_FILE" ]; then
    echo "loading env file: $ENV_FILE"
    source "$ENV_FILE"
else
    echo "ERROR ! Could not find $ENV_FILE"
    exit 1
fi

# 解析命令行参数并加载环境配置
parse_args_and_load_env "$BENCH_SCRIPT_DIR/.." "$@"

# Prefix cache 隔离：
#   bench 默认固定 --seed 0，同一台服务上重复运行（或同一脚本里连续多次 bench）会发出完全相同的
#   prompt，后一次会命中前一次留下的 prefix cache，导致 TTFT/吞吐不稳定。
#   这里给每次脚本运行生成唯一的 cache_salt，通过 --extra-body 注入请求：
#   不同 salt 的请求之间不会共享 KV block，不依赖 VLLM_SERVER_DEV_MODE，也不需要关闭服务端
#   prefix caching。如果需要对比"命中 prefix cache"的场景，设置 BENCH_CACHE_SALT 为固定值，
#   或设置 BENCH_DISABLE_CACHE_SALT=1 关闭隔离。
#   注意：同一次 bench 内部 prompt 彼此重复造成的命中（如 ShareGPT/SpecBench 数据集本身有重复前缀）
#   仍属于数据集特性，不在隔离范围内。
if [ "${BENCH_DISABLE_CACHE_SALT:-0}" = "1" ]; then
    BENCH_EXTRA_BODY_ARGS=()
else
    BENCH_CACHE_SALT="${BENCH_CACHE_SALT:-bench-$(date +%s%N)-$$}"
    echo "bench cache_salt: ${BENCH_CACHE_SALT}"
    BENCH_EXTRA_BODY_ARGS=(--extra-body "{\"cache_salt\": \"${BENCH_CACHE_SALT}\"}")
fi

# Batch Decode 测试
vllm bench serve --port ${USER_VLLM_PORT} \
    --model ${USER_VLLM_MODEL} \
    --backend vllm \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len 16 \
    --random-output-len 8 \
    --random-range-ratio 0.0 \
    --profile \
    "${BENCH_EXTRA_BODY_ARGS[@]}" \
    --num-prompts 1

# ShareGPT 测试：取消注释即可。路径以本脚本所在目录为基准，不受执行目录影响。
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
# 提前准备到 vllm-xcpu-dev-kit/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
# vllm bench serve --port ${USER_VLLM_PORT} \
#     --model ${USER_VLLM_MODEL} \
#     --backend vllm \
#     --endpoint /v1/completions \
#     --dataset-name sharegpt \
#     --dataset-path "${BENCH_SCRIPT_DIR}/../../datasets/ShareGPT_V3_unfiltered_cleaned_split.json" \
#     --num-prompts 100 \
#     --max-concurrency 16 \
#     --request-rate inf \
#     --sharegpt-output-len 256 \
#     --seed 0 \
#     --profile \
#     "${BENCH_EXTRA_BODY_ARGS[@]}" \
#     --disable-shuffle \
#     --save-result
#
# 常用流量和长度参数：
#   --num-prompts 100          总请求数。
#   --max-concurrency 16       最多同时在飞的请求数。
#   --request-rate inf         不限发送速率，所有请求尽快提交；可改成 8 表示平均 8 requests/s。
#   --burstiness 1             有限 request-rate 下使用泊松到达；小于 1 更突发，大于 1 更均匀。
#   --sharegpt-output-len 256  统一设置 max_tokens；删除该项则使用数据集中 assistant 回复的 token 长度。
#   --ignore-eos               强制生成到目标输出长度；不加时模型可能遇到 EOS 提前停止。
#
# ShareGPT 没有 --sharegpt-input-len 参数，输入长度来自真实 prompt。当前 vLLM loader 会过滤：
# prompt 少于 4 tokens、prompt 大于 1024 tokens，或 prompt + output 大于 2048 tokens 的样本。
# 如需严格限定输入长度范围，需预处理 ShareGPT JSON，或改用 random/SPEED-Bench。
#
# 样本顺序：
#   默认会先 shuffle，但默认 --seed 0，所以数据文件、tokenizer、长度参数和 seed 不变时，
#   多次运行会选到同一批样本；它们不是 JSON 原始顺序的前 N 个。
#   加 --disable-shuffle 才会按 JSON 原始顺序选取前 N 个通过长度检查的样本。



# SpecBench 测试：数据文件位于 serve_test/data/spec_bench.jsonl，路径同样与执行目录无关。
# vllm bench serve --port ${USER_VLLM_PORT} \
#     --model ${USER_VLLM_MODEL} \
#     --backend vllm \
#     --endpoint /v1/completions \
#     --dataset-name spec_bench \
#     --dataset-path "${BENCH_SCRIPT_DIR}/data/spec_bench.jsonl" \
#     --num-prompts 1 \
#     --max-concurrency 16 \
#     --request-rate inf \
#     --spec-bench-output-len 256 \
#     --profile \
#     "${BENCH_EXTRA_BODY_ARGS[@]}" \
#     --disable-shuffle \
#     --save-result
#
# SpecBench 常用参数：
#   --num-prompts 100             总请求数；设为 -1 表示使用选定分类中的全部样本。
#   --max-concurrency 16          最多同时在飞的请求数。
#   --request-rate inf            不限发送速率；可改成 8 表示平均 8 requests/s。
#   --burstiness 1                有限 request-rate 下为泊松到达；小于 1 更突发，大于 1 更均匀。
#   --spec-bench-output-len 256   统一设置每个请求的 max_tokens，默认也是 256。
#   --ignore-eos                  强制生成到目标输出长度；不加时可能因 EOS 提前停止。
#   --no-oversample               当 num-prompts 超过可用样本数时不重复采样。
#
# 按领域限制：在命令中加 --spec-bench-category coding。可用分类包括：
# writing, roleplay, reasoning, math, coding, extraction, stem, humanities,
# translation, summarization, qa, math_reasoning, rag。不指定 category 时使用全部领域。
#
# SpecBench 没有 --spec-bench-input-len 参数，输入长度由每条真实 prompt 决定。
# 需要严格控制输入 token 长度时，需预处理 JSONL，或使用 random/SPEED-Bench。
#
# SpecBench 默认会以固定种子 shuffle，因此数据文件和参数不变时选中样本稳定，
# 但不是 JSONL 原始顺序的前 N 个。加 --disable-shuffle 可按文件原始顺序取样。



# 精度简单校验测试，避免精度明显异常
curl --silent --show-error --fail-with-body \
  --write-out '\ncurl_time_total_seconds=%{time_total}\n' \
  "http://localhost:${USER_VLLM_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer empty" \
  -d '{
    "model": "'"${USER_VLLM_MODEL}"'",
    "messages": [
      {"role": "user", "content": "请用一段话简单介绍一下量子计算。"}
    ],
    "chat_template_kwargs": {
      "enable_thinking": false
    },
    "max_tokens": 16,
    "temperature": 0.7
  }'
