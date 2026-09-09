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

vllm bench serve --port ${USER_VLLM_PORT} \
    --model ${USER_VLLM_MODEL} \
    --backend vllm \
    --endpoint /v1/completions \
    --dataset-name spec_bench \
    --dataset-path "${BENCH_SCRIPT_DIR}/data/spec_bench.jsonl" \
    --num-prompts 1 \
    --spec-bench-category coding \
    --max-concurrency 16 \
    --request-rate inf \
    --spec-bench-output-len 256 \
    --disable-shuffle \
    --save-result

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
