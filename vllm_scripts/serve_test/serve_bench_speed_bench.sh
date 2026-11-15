#!/bin/bash

# vLLM SPEED-Bench 投机解码 / 长短上下文基准测试脚本
#
# 使用说明:
#   方式1: 通过 -e 参数指定预设文件
#     ./serve_test/serve_bench_speed_bench.sh -e ./presets/serial/dflash/Qwen3.5-0.8B_dp1_tp1_compile_dflash_4.sh
#
#   方式2: 通过 PRESET 环境变量
#     PRESET=serial/dflash/Qwen3.5-0.8B_dp1_tp1_compile_dflash_4 ./serve_test/serve_bench_speed_bench.sh
#
#   方式3: 搭配 run_vllm_test.sh 一键启停测试（短上下文冒烟测试示例）
#     SPEED_BENCH_SUBSET=qualitative SPEED_BENCH_NUM_PROMPTS=4 SPEED_BENCH_OUTPUT_LEN=32 \
#       ./run_vllm_test.sh -e presets/serial/dflash/Qwen3.5-0.8B_dp1_tp1_compile_dflash_4.sh --speed-bench
#     说明: 该命令会自动拉起服务端、等待就绪并跑完后销毁进程；客户端先做 1 次单条预热检查，
#     随后从 qualitative.jsonl 顺序取前 4 条（默认不打乱，均为 coding 类；经 Qwen3.5 chat
#     模板渲染后输入长度分别约 61/301/166/39 共 567 tokens），并发发送（concurrency=16 上限下
#     4 条同时发出）、每条最多生成 32 tokens，默认 temperature=0 且附带 cache_salt 防历史前缀串扰，
#     采集 TTFT/TPOT/ITL 与 DFlash 投机解码接受率 / 平均接受长度。
#
# 典型场景示例:
#   # 场景1: 特定任务域或跨领域混合的投机解码接受率评估（搭配 --plot-timeline）
#   SPEED_BENCH_SUBSET=qualitative SPEED_BENCH_CATEGORY=math \
#   SPEED_BENCH_NUM_PROMPTS=16 SPEED_BENCH_OUTPUT_LEN=64 SPEED_BENCH_PLOT_TIMELINE=1 \
#     ./serve_test/serve_bench_speed_bench.sh -e presets/serial/dflash/Qwen3.5-0.8B_dp1_tp1_compile_dflash_4.sh
#   # （若要跨 qualitative 全部 10 类随机抽样，可设置 SPEED_BENCH_DISABLE_SHUFFLE=0）
#
#   # 场景2: 固定输入长度档位 (1K~32K) 与高低熵 (low_entropy / high_entropy) 连续扫描
#   SPEED_BENCH_SUBSETS="throughput_1k throughput_2k throughput_8k" \
#   SPEED_BENCH_CATEGORY=low_entropy SPEED_BENCH_NUM_PROMPTS=8 SPEED_BENCH_OUTPUT_LEN=64 \
#     ./serve_test/serve_bench_speed_bench.sh -e presets/...
#
# 常用环境变量:
#   SPEED_BENCH_DATA_DIR       JSONL 数据目录，默认 <serve_test>/data/speed_bench
#   SPEED_BENCH_ARCHIVE        独立数据集压缩包路径，默认 <serve_test>/data/speed_bench_jsonl.tar.gz
#                              （若数据目录不存在且压缩包存在，将自动解压到对应位置）
#   SPEED_BENCH_SUBSET         单次运行的子集，默认 qualitative
#                              可选: qualitative | throughput_1k | throughput_2k | throughput_8k | throughput_16k | throughput_32k
#   SPEED_BENCH_SUBSETS        多子集连续扫描（空格或逗号分隔），如 "throughput_1k throughput_2k throughput_8k"
#                              若设置则覆盖 SPEED_BENCH_SUBSET
#   SPEED_BENCH_CATEGORY       过滤分类（不设置则使用子集全部可用样本）
#                              - qualitative: coding, humanities, math, multilingual, qa, rag, reasoning, roleplay, stem, writing
#                              - throughput_*: low_entropy, high_entropy (完整构建版还含 mixed)
#   SPEED_BENCH_NUM_PROMPTS    每子集发送请求数，默认 4（设为 -1 使用子集全部样本）
#   SPEED_BENCH_OUTPUT_LEN     每个请求的 max output tokens，默认 64
#   SPEED_BENCH_MAX_CONCURRENCY 最大并发数，默认 16
#   SPEED_BENCH_REQUEST_RATE   请求发送速率，默认 inf
#   SPEED_BENCH_DISABLE_SHUFFLE 设为 1（默认）时传 --disable-shuffle 保证按序可复现抽样；设为 0 则按 seed 随机打乱（跨多类均衡抽样）
#   SPEED_BENCH_TEMPERATURE    采样温度，默认 0（保证投机解码接受率测量可复现；置空则沿用服务端默认）
#   SPEED_BENCH_IGNORE_EOS     设为 1 时传入 --ignore-eos，强制生成满 SPEED_BENCH_OUTPUT_LEN
#   SPEED_BENCH_PROFILE        设为 1 时开启 --profile
#   SPEED_BENCH_PLOT_TIMELINE  设为 1 时开启 --plot-timeline 生成 *.timeline.html
#   SPEED_BENCH_TIMELINE_ITL_THRESHOLDS
#                              Timeline ITL 颜色分档阈值 (ms)，默认 "50,150"
#   SPEED_BENCH_RESULT_DIR     结果与 timeline HTML 输出目录，默认 <serve_test>/bench_results
#   SPEED_BENCH_HOST           服务地址，默认 127.0.0.1
#   SPEED_BENCH_EXTRA_ARGS     透传给 vllm bench serve 的额外参数（如 "--save-detailed --seed 0"）

set -euo pipefail

BENCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载通用函数
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

SPEED_BENCH_HOST="${SPEED_BENCH_HOST:-127.0.0.1}"
SPEED_BENCH_DATA_DIR="${SPEED_BENCH_DATA_DIR:-${BENCH_SCRIPT_DIR}/data/speed_bench}"
SPEED_BENCH_ARCHIVE="${SPEED_BENCH_ARCHIVE:-${BENCH_SCRIPT_DIR}/data/speed_bench_jsonl.tar.gz}"
SPEED_BENCH_SUBSET="${SPEED_BENCH_SUBSET:-qualitative}"
SPEED_BENCH_SUBSETS="${SPEED_BENCH_SUBSETS:-${SPEED_BENCH_SUBSET}}"
SPEED_BENCH_CATEGORY="${SPEED_BENCH_CATEGORY:-}"
SPEED_BENCH_NUM_PROMPTS="${SPEED_BENCH_NUM_PROMPTS:-4}"
SPEED_BENCH_OUTPUT_LEN="${SPEED_BENCH_OUTPUT_LEN:-64}"
SPEED_BENCH_MAX_CONCURRENCY="${SPEED_BENCH_MAX_CONCURRENCY:-16}"
SPEED_BENCH_REQUEST_RATE="${SPEED_BENCH_REQUEST_RATE:-inf}"
SPEED_BENCH_DISABLE_SHUFFLE="${SPEED_BENCH_DISABLE_SHUFFLE:-1}"
SPEED_BENCH_TEMPERATURE="${SPEED_BENCH_TEMPERATURE-0}"
SPEED_BENCH_IGNORE_EOS="${SPEED_BENCH_IGNORE_EOS:-0}"
SPEED_BENCH_PROFILE="${SPEED_BENCH_PROFILE:-0}"
SPEED_BENCH_PLOT_TIMELINE="${SPEED_BENCH_PLOT_TIMELINE:-0}"
SPEED_BENCH_TIMELINE_ITL_THRESHOLDS="${SPEED_BENCH_TIMELINE_ITL_THRESHOLDS:-50,150}"
SPEED_BENCH_RESULT_DIR="${SPEED_BENCH_RESULT_DIR:-${BENCH_SCRIPT_DIR}/bench_results}"

SUBSETS_RAW="${SPEED_BENCH_SUBSETS//,/ }"
read -r -a SUBSET_LIST <<< "${SUBSETS_RAW}"
if [ "${#SUBSET_LIST[@]}" -eq 0 ]; then
    echo "ERROR: No SPEED-Bench subsets specified." >&2
    exit 1
fi

mkdir -p "${SPEED_BENCH_RESULT_DIR}"

# 若本地尚未解压数据目录且存在独立压缩包，则自动解压
if [ ! -d "${SPEED_BENCH_DATA_DIR}" ] && [ -f "${SPEED_BENCH_ARCHIVE}" ]; then
    echo "Extracting standalone SPEED-Bench archive: ${SPEED_BENCH_ARCHIVE}"
    mkdir -p "$(dirname "${SPEED_BENCH_DATA_DIR}")"
    tar -xzf "${SPEED_BENCH_ARCHIVE}" -C "$(dirname "${SPEED_BENCH_DATA_DIR}")"
fi

if [ ! -d "${SPEED_BENCH_DATA_DIR}" ]; then
    echo "ERROR: SPEED-Bench data directory not found: ${SPEED_BENCH_DATA_DIR}" >&2
    echo "请解压 speed_bench_jsonl.tar.gz 到 $(dirname "${SPEED_BENCH_DATA_DIR}")，" >&2
    echo "或执行: python3 ${BENCH_SCRIPT_DIR}/prepare_speed_bench_local.py --download" >&2
    exit 1
fi

for subset in "${SUBSET_LIST[@]}"; do
    subset_jsonl="${SPEED_BENCH_DATA_DIR}/${subset}.jsonl"
    if [ ! -f "${subset_jsonl}" ]; then
        echo "ERROR: Missing subset JSONL: ${subset_jsonl}" >&2
        echo "Available files in ${SPEED_BENCH_DATA_DIR}:" >&2
        ls -la "${SPEED_BENCH_DATA_DIR}" >&2 || true
        exit 1
    fi
done

if [ "${BENCH_DISABLE_CACHE_SALT:-0}" = "1" ]; then
    BASE_CACHE_SALT=""
else
    BASE_CACHE_SALT="${BENCH_CACHE_SALT:-bench-$(date +%s%N)-$$}"
fi

EXTRA_ARGS=()
if [ -n "${SPEED_BENCH_EXTRA_ARGS:-}" ]; then
    read -r -a EXTRA_ARGS <<< "${SPEED_BENCH_EXTRA_ARGS}"
fi

for subset in "${SUBSET_LIST[@]}"; do
    echo "========================================="
    echo "  SPEED-Bench Run"
    echo "  subset:          ${subset}"
    echo "  category:        ${SPEED_BENCH_CATEGORY:-<all>}"
    echo "  num_prompts:     ${SPEED_BENCH_NUM_PROMPTS}"
    echo "  output_len:      ${SPEED_BENCH_OUTPUT_LEN}"
    echo "  max_concurrency: ${SPEED_BENCH_MAX_CONCURRENCY}"
    echo "  request_rate:    ${SPEED_BENCH_REQUEST_RATE}"
    echo "  plot_timeline:   ${SPEED_BENCH_PLOT_TIMELINE}"
    echo "========================================="

    CMD=(
        vllm bench serve
        --host "${SPEED_BENCH_HOST}"
        --port "${USER_VLLM_PORT}"
        --model "${USER_VLLM_MODEL}"
        --backend vllm
        --endpoint /v1/completions
        --dataset-name speed_bench
        --dataset-path "${SPEED_BENCH_DATA_DIR}"
        --speed-bench-dataset-subset "${subset}"
        --num-prompts "${SPEED_BENCH_NUM_PROMPTS}"
        --max-concurrency "${SPEED_BENCH_MAX_CONCURRENCY}"
        --request-rate "${SPEED_BENCH_REQUEST_RATE}"
        --speed-bench-output-len "${SPEED_BENCH_OUTPUT_LEN}"
        --label "speed_bench-${subset}"
        --result-dir "${SPEED_BENCH_RESULT_DIR}"
        --save-result
    )

    if [ "${SPEED_BENCH_DISABLE_SHUFFLE}" = "1" ]; then
        CMD+=(--disable-shuffle)
    fi

    if [ -n "${SPEED_BENCH_CATEGORY}" ]; then
        CMD+=(--speed-bench-category "${SPEED_BENCH_CATEGORY}")
    fi

    if [ -n "${SPEED_BENCH_TEMPERATURE}" ]; then
        CMD+=(--temperature "${SPEED_BENCH_TEMPERATURE}")
    fi

    if [ "${SPEED_BENCH_IGNORE_EOS}" = "1" ]; then
        CMD+=(--ignore-eos)
    fi

    if [ "${SPEED_BENCH_PROFILE}" = "1" ]; then
        CMD+=(--profile)
    fi

    if [ "${SPEED_BENCH_PLOT_TIMELINE}" = "1" ]; then
        CMD+=(
            --plot-timeline
            --timeline-itl-thresholds "${SPEED_BENCH_TIMELINE_ITL_THRESHOLDS}"
        )
    fi

    if [ -n "${BASE_CACHE_SALT}" ]; then
        subset_salt="${BASE_CACHE_SALT}-${subset}"
        echo "bench cache_salt: ${subset_salt}"
        CMD+=(--extra-body "{\"cache_salt\": \"${subset_salt}\"}")
    fi

    if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    "${CMD[@]}"
done
