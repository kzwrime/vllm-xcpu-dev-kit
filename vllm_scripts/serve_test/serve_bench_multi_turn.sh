#!/bin/bash

# vLLM 多轮对话 & 中间轮次增量 Prefill/Decode 基准测试脚本
#
# 使用说明:
#   方式1: 通过 -e 参数指定预设文件
#     ./serve_test/serve_bench_multi_turn.sh -e ./presets/serial/dflash/Qwen3.5-0.8B_dp1_tp1_compile_dflash_4.sh
#
#   方式2: 通过 PRESET 环境变量
#     PRESET=serial/dflash/Qwen3.5-0.8B_dp1_tp1_compile_dflash_4 ./serve_test/serve_bench_multi_turn.sh
#
#   方式3: 搭配 run_vllm_test.sh 一键启停测试
#     MULTI_TURN_MEASURE_TURNS="3:4" MULTI_TURN_PLOT_TIMELINE=1 \
#       ./run_vllm_test.sh -e presets/serial/dflash/Qwen3.5-0.8B_dp1_tp1_compile_dflash_4.sh --multi-turn-bench
#
# 典型场景示例（长历史上下文 + 增量 4K Prefill + 128 Token Decode + 投机解码接受率测量）:
#   MULTI_TURN_NUM_CONVERSATIONS=4 MULTI_TURN_MAX_CONCURRENCY=4 \
#   MULTI_TURN_NUM_TURNS=5 MULTI_TURN_INITIAL_CONTEXT_TOKENS=16384 \
#   MULTI_TURN_INPUT_TOKENS=4096 MULTI_TURN_OUTPUT_TOKENS=128 \
#   MULTI_TURN_MEASURE_TURNS="3:5" MULTI_TURN_PLOT_TIMELINE=1 \
#     ./serve_test/serve_bench_multi_turn.sh -e ./presets/...
#
# 真实 ShareGPT / 对话集模式示例:
#   MULTI_TURN_DATASET_MODE=sharegpt \
#   MULTI_TURN_DATASET_PATH=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
#   MULTI_TURN_NUM_TURNS=4 MULTI_TURN_MEASURE_TURNS="2:4" \
#     ./serve_test/serve_bench_multi_turn.sh -e ./presets/...
#
# 常用环境变量:
#   MULTI_TURN_DATASET_MODE          corpus (默认，使用本地真实语料按 token 预算组装) | sharegpt
#   MULTI_TURN_DATASET_PATH          sharegpt 模式下的 JSON/JSONL 文件路径
#   MULTI_TURN_CORPUS_PATHS          corpus 模式自定义语料路径（空格分隔）
#   MULTI_TURN_NUM_CONVERSATIONS     并发测试的会话总数，默认 4
#   MULTI_TURN_MAX_CONCURRENCY       最大在飞请求并发数，默认 4
#   MULTI_TURN_NUM_TURNS             每个会话总轮数，默认 4
#   MULTI_TURN_COMMON_PREFIX_TOKENS  所有会话共享的系统前缀 token 数，默认 0
#   MULTI_TURN_INITIAL_CONTEXT_TOKENS 第 1 轮输入 token 数，默认 640（可跨过如 Qwen3.5 576-token 混合块对齐粒度）
#   MULTI_TURN_INPUT_TOKENS          第 2..N 轮每轮新增输入 token 数，默认 256
#   MULTI_TURN_OUTPUT_TOKENS         每轮生成 max_tokens，默认 64
#   MULTI_TURN_MEASURE_TURNS         聚焦测量的中间轮次区间 (1-indexed, 如 "3:4")，默认 "3:4"
#   MULTI_TURN_PHASE_BARRIER         设为 1（默认）时在测量窗口前后做阶段屏障与 /metrics 快照隔离；
#                                    设为 0 时全程无屏障连续并发
#   MULTI_TURN_TEMPERATURE           采样温度，默认 0.0
#   MULTI_TURN_IGNORE_EOS            设为 1 时强制生成满 MULTI_TURN_OUTPUT_TOKENS
#   MULTI_TURN_ENABLE_THINKING       设为 1 时允许 thinking，默认 0（传 enable_thinking=false）
#   MULTI_TURN_PROFILE_WINDOW        设为 1 时仅对 MEASURE_TURNS 窗口触发 /start_profile ~ /stop_profile
#   MULTI_TURN_PLOT_TIMELINE         设为 1 时生成 *.timeline.html 时间线图
#   MULTI_TURN_TIMELINE_ITL_THRESHOLDS Timeline ITL 分档阈值 (ms)，默认 "50,150"
#   MULTI_TURN_TIMELINE_SCOPE        all（默认，绘制全部轮次）| window（仅绘制测量窗口轮次）
#   MULTI_TURN_RESULT_DIR            结果与 timeline HTML 输出目录，默认 <serve_test>/multi_turn_results
#   MULTI_TURN_HOST                  服务地址，默认 127.0.0.1
#   MULTI_TURN_EXTRA_ARGS            透传给 bench_multi_turn.py 的额外参数

set -euo pipefail

BENCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="$BENCH_SCRIPT_DIR/../common.sh"
if [ -f "$ENV_FILE" ]; then
    echo "loading env file: $ENV_FILE"
    set +u
    source "$ENV_FILE"
    parse_args_and_load_env "$BENCH_SCRIPT_DIR/.." "$@"
    set -u
else
    echo "ERROR ! Could not find $ENV_FILE"
    exit 1
fi

MULTI_TURN_HOST="${MULTI_TURN_HOST:-127.0.0.1}"
MULTI_TURN_DATASET_MODE="${MULTI_TURN_DATASET_MODE:-corpus}"
MULTI_TURN_DATASET_PATH="${MULTI_TURN_DATASET_PATH:-}"
MULTI_TURN_NUM_CONVERSATIONS="${MULTI_TURN_NUM_CONVERSATIONS:-4}"
MULTI_TURN_MAX_CONCURRENCY="${MULTI_TURN_MAX_CONCURRENCY:-4}"
MULTI_TURN_NUM_TURNS="${MULTI_TURN_NUM_TURNS:-4}"
MULTI_TURN_COMMON_PREFIX_TOKENS="${MULTI_TURN_COMMON_PREFIX_TOKENS:-0}"
MULTI_TURN_INITIAL_CONTEXT_TOKENS="${MULTI_TURN_INITIAL_CONTEXT_TOKENS:-640}"
MULTI_TURN_INPUT_TOKENS="${MULTI_TURN_INPUT_TOKENS:-256}"
MULTI_TURN_OUTPUT_TOKENS="${MULTI_TURN_OUTPUT_TOKENS:-64}"
MULTI_TURN_MEASURE_TURNS="${MULTI_TURN_MEASURE_TURNS:-3:4}"
MULTI_TURN_PHASE_BARRIER="${MULTI_TURN_PHASE_BARRIER:-1}"
MULTI_TURN_REQUEST_RATE="${MULTI_TURN_REQUEST_RATE:-inf}"
MULTI_TURN_TEMPERATURE="${MULTI_TURN_TEMPERATURE:-0.0}"
MULTI_TURN_IGNORE_EOS="${MULTI_TURN_IGNORE_EOS:-0}"
MULTI_TURN_ENABLE_THINKING="${MULTI_TURN_ENABLE_THINKING:-0}"
MULTI_TURN_PROFILE_WINDOW="${MULTI_TURN_PROFILE_WINDOW:-0}"
MULTI_TURN_PLOT_TIMELINE="${MULTI_TURN_PLOT_TIMELINE:-0}"
MULTI_TURN_TIMELINE_ITL_THRESHOLDS="${MULTI_TURN_TIMELINE_ITL_THRESHOLDS:-50,150}"
MULTI_TURN_TIMELINE_SCOPE="${MULTI_TURN_TIMELINE_SCOPE:-all}"
MULTI_TURN_RESULT_DIR="${MULTI_TURN_RESULT_DIR:-${BENCH_SCRIPT_DIR}/multi_turn_results}"
MULTI_TURN_SEED="${MULTI_TURN_SEED:-0}"
MULTI_TURN_TIMEOUT_SEC="${MULTI_TURN_TIMEOUT_SEC:-600}"

mkdir -p "${MULTI_TURN_RESULT_DIR}"

CMD=(
    python3 "${BENCH_SCRIPT_DIR}/bench_multi_turn.py"
    --host "${MULTI_TURN_HOST}"
    --port "${USER_VLLM_PORT}"
    --model "${USER_VLLM_MODEL}"
    --dataset-mode "${MULTI_TURN_DATASET_MODE}"
    --num-conversations "${MULTI_TURN_NUM_CONVERSATIONS}"
    --num-turns "${MULTI_TURN_NUM_TURNS}"
    --common-prefix-tokens "${MULTI_TURN_COMMON_PREFIX_TOKENS}"
    --initial-context-tokens "${MULTI_TURN_INITIAL_CONTEXT_TOKENS}"
    --turn-input-tokens "${MULTI_TURN_INPUT_TOKENS}"
    --turn-output-tokens "${MULTI_TURN_OUTPUT_TOKENS}"
    --max-concurrency "${MULTI_TURN_MAX_CONCURRENCY}"
    --request-rate "${MULTI_TURN_REQUEST_RATE}"
    --temperature "${MULTI_TURN_TEMPERATURE}"
    --measure-turns "${MULTI_TURN_MEASURE_TURNS}"
    --result-dir "${MULTI_TURN_RESULT_DIR}"
    --seed "${MULTI_TURN_SEED}"
    --timeout-sec "${MULTI_TURN_TIMEOUT_SEC}"
    --timeline-itl-thresholds "${MULTI_TURN_TIMELINE_ITL_THRESHOLDS}"
    --timeline-scope "${MULTI_TURN_TIMELINE_SCOPE}"
)

if [ -n "${MULTI_TURN_DATASET_PATH}" ]; then
    CMD+=(--dataset-path "${MULTI_TURN_DATASET_PATH}")
fi

if [ -n "${MULTI_TURN_CORPUS_PATHS:-}" ]; then
    read -r -a CORPUS_ARR <<< "${MULTI_TURN_CORPUS_PATHS}"
    if [ "${#CORPUS_ARR[@]}" -gt 0 ]; then
        CMD+=(--corpus-paths "${CORPUS_ARR[@]}")
    fi
fi

if [ "${MULTI_TURN_PHASE_BARRIER}" = "1" ]; then
    CMD+=(--phase-barrier)
else
    CMD+=(--no-phase-barrier)
fi

if [ "${MULTI_TURN_IGNORE_EOS}" = "1" ]; then
    CMD+=(--ignore-eos)
fi

if [ "${MULTI_TURN_ENABLE_THINKING}" = "1" ]; then
    CMD+=(--enable-thinking)
fi

if [ "${MULTI_TURN_PROFILE_WINDOW}" = "1" ]; then
    CMD+=(--profile-window)
fi

if [ "${MULTI_TURN_PLOT_TIMELINE}" = "1" ]; then
    CMD+=(--plot-timeline)
fi

if [ "${BENCH_DISABLE_CACHE_SALT:-0}" = "1" ]; then
    CMD+=(--disable-cache-salt)
elif [ -n "${BENCH_CACHE_SALT:-}" ]; then
    CMD+=(--cache-salt "${BENCH_CACHE_SALT}")
fi

if [ -n "${MULTI_TURN_EXTRA_ARGS:-}" ]; then
    read -r -a MT_EXTRA_ARR <<< "${MULTI_TURN_EXTRA_ARGS}"
    CMD+=("${MT_EXTRA_ARR[@]}")
fi

"${CMD[@]}"
