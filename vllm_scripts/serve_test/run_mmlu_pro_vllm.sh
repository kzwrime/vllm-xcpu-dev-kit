#!/usr/bin/env bash
# 使用已启动的、兼容 OpenAI API 的 vLLM 服务评测 MMLU-Pro。
# 本文件可单独复制到其他机器；只要求目标环境已安装 lm-eval（python -m lm_eval）和 curl。
#
# 用法：
#   # 快速检查：biology 的前 2 题（不能作为正式分数）
#   USER_VLLM_MODEL=Qwen/Qwen3.5-0.8B USER_VLLM_PORT=8000 \
#     bash run_mmlu_pro_vllm.sh smoke --no-thinking
#
#   # 正式全量评测：14 个 MMLU-Pro 学科
#   USER_VLLM_MODEL=Qwen/Qwen3.5-0.8B USER_VLLM_PORT=8000 \
#     bash run_mmlu_pro_vllm.sh full --thinking
#
#   # 只完整评测 math 和 physics 两个学科，缩短运行时间
#   TASKS=mmlu_pro_math,mmlu_pro_physics \
#     bash run_mmlu_pro_vllm.sh full --thinking
#
#   # 波次提交：每批 32 题，整批完成后才提交下一批
#   BATCH_SIZE=32 NUM_CONCURRENT=1 API_TIMEOUT=1800 \
#     bash run_mmlu_pro_vllm.sh full --thinking
#
#   # 模拟器快速估测：math、physics 每个学科各取前 20 题
#   TASKS=mmlu_pro_math,mmlu_pro_physics LIMIT_PER_TASK=20 \
#     bash run_mmlu_pro_vllm.sh full --thinking
#
# 使用 /v1/completions 而不是 /v1/chat/completions：MMLU-Pro 内置原始文本形式的
# 5-shot CoT prompt；completions 不会额外改变其聊天模板格式。

set -euo pipefail

MODE="smoke"
THINKING_MODE="auto"

# --thinking / --no-thinking 用于选择 Qwen 官方对应的采样 profile。
# MMLU-Pro 的题目本身始终要求 “Think step by step”，因此 --thinking 不会删除
# 该 CoT 指令；它也不会修改 vLLM 服务启动配置。若关闭 --reasoning-parser qwen3，
# 模型生成的思考内容会作为普通 completion 正文返回，正适合本任务的答案提取器处理。
for arg in "$@"; do
  case "${arg}" in
    smoke|full)
      MODE="${arg}"
      ;;
    --thinking)
      THINKING_MODE="thinking"
      ;;
    --no-thinking)
      THINKING_MODE="nonthinking"
      ;;
    -h|--help)
      sed -n '1,32p' "$0"
      exit 0
      ;;
    *)
      echo "未知选项: ${arg}" >&2
      echo "用法: $0 [smoke|full] [--thinking|--no-thinking]" >&2
      exit 2
      ;;
  esac
done

# MODEL 必须与 ${VLLM_ROOT}/v1/models 返回的 data[].id 一致。
USER_VLLM_MODEL="${USER_VLLM_MODEL:-Qwen/Qwen3.5-0.8B}"
USER_VLLM_PORT="${USER_VLLM_PORT:-8000}"
VLLM_ROOT="${VLLM_ROOT:-http://localhost:${USER_VLLM_PORT}}"
# 一个 HTTP /v1/completions 请求中包含的题目数。API 后端不支持 batch_size=auto。
BATCH_SIZE="${BATCH_SIZE:-1}"
# 同时在飞的 HTTP 请求数。大于 1 时，某个 batch 结束就会补充新 batch。
NUM_CONCURRENT="${NUM_CONCURRENT:-64}"
# 大 batch 的整批响应可能超过默认 300 秒；按平台速度酌情提高。
API_TIMEOUT="${API_TIMEOUT:-300}"
MAX_RETRIES="${MAX_RETRIES:-3}"

if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ && "${NUM_CONCURRENT}" =~ ^[1-9][0-9]*$ && "${API_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE、NUM_CONCURRENT 和 API_TIMEOUT 必须是正整数。" >&2
  exit 2
fi

# 采样 profile：auto 会按“模型名包含关系”（忽略大小写）选择默认项。
#
#   Qwen3.5 -> 默认 qwen35_text_nonthinking；--thinking 时为 qwen35_text_thinking。
#   Qwen3.6 -> 默认 qwen36_instruct；--thinking 时为 qwen36_thinking。
#   其他模型 -> generic：给出 warning，并采用保守的通用参数。
#
# 若你的 Qwen3.6 服务已明确启用原生 thinking，可改为：
#   SAMPLING_PROFILE=qwen36_thinking bash run_mmlu_pro_vllm.sh full
# 所有可选 profile：qwen35_text_nonthinking、qwen35_text_thinking、
# qwen36_thinking、qwen36_coding、qwen36_instruct、generic。
SAMPLING_PROFILE="${SAMPLING_PROFILE:-auto}"

# 显式提供 GEN_KWARGS 时优先级最高，可完全自行覆盖采样参数。若设为空字符串，
# 则不传 --gen_kwargs，恢复 MMLU-Pro 任务 YAML 的默认贪心解码。
#   GEN_KWARGS='' bash run_mmlu_pro_vllm.sh full
if [[ ! -v GEN_KWARGS ]]; then
  model_name_lower="${USER_VLLM_MODEL,,}"
  if [[ "${SAMPLING_PROFILE}" == "auto" ]]; then
    if [[ "${model_name_lower}" == *"qwen3.5"* ]]; then
      if [[ "${THINKING_MODE}" == "thinking" ]]; then
        SAMPLING_PROFILE="qwen35_text_thinking"
      else
        SAMPLING_PROFILE="qwen35_text_nonthinking"
      fi
    elif [[ "${model_name_lower}" == *"qwen3.6"* ]]; then
      if [[ "${THINKING_MODE}" == "thinking" ]]; then
        SAMPLING_PROFILE="qwen36_thinking"
      else
        SAMPLING_PROFILE="qwen36_instruct"
      fi
    else
      SAMPLING_PROFILE="generic"
      echo "WARNING: 模型名 '${USER_VLLM_MODEL}' 不包含 qwen3.5 或 qwen3.6。" >&2
      echo "WARNING: 将使用通用采样参数；请依据该模型的 model card 设置 GEN_KWARGS。" >&2
    fi
  fi

  case "${SAMPLING_PROFILE}" in
    qwen35_text_nonthinking)
      # Qwen3.5：非 thinking 文本任务。
      GEN_KWARGS='do_sample=True,temperature=1.0,top_p=1.0,top_k=20,min_p=0.0,presence_penalty=2.0,repetition_penalty=1.0'
      ;;
    qwen35_text_thinking)
      # Qwen3.5：thinking 文本任务。
      GEN_KWARGS='do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0'
      ;;
    qwen36_thinking)
      # Qwen3.6：thinking 通用任务。
      GEN_KWARGS='do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0'
      ;;
    qwen36_coding)
      # Qwen3.6：thinking 精确代码任务。
      GEN_KWARGS='do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0'
      ;;
    qwen36_instruct)
      # Qwen3.6：Instruct / 非 thinking 任务。
      GEN_KWARGS='do_sample=True,temperature=0.7,top_p=0.8,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0'
      ;;
    generic)
      # 未识别模型的通用默认值；务必优先采用该模型 model card 的建议。
      GEN_KWARGS='do_sample=True,temperature=1.0,top_p=0.95,top_k=40'
      ;;
    *)
      echo "未知的 SAMPLING_PROFILE: ${SAMPLING_PROFILE}" >&2
      exit 2
      ;;
  esac
fi

# presence_penalty 通常可在 0--2 内调整以减轻无限重复；过高可能导致语言混杂或
# 小幅降低性能。若 samples_*.jsonl 中出现大量 [invalid] 或重复输出，可优先调整它。

case "${MODE}" in
  smoke)
    # 仅两个样本，用于确认服务连通、答案格式和采样参数是否正常。
    # --limit 仅限调试，绝不可将本模式结果作为 benchmark 分数报告。
    TASKS="${TASKS:-mmlu_pro_biology}"
    LIMIT="${LIMIT:-${LIMIT_PER_TASK:-2}}"
    ;;
  full)
    # mmlu_pro 覆盖 14 个学科。TASKS 可限制为一个或多个完整学科任务，例如：
    #   TASKS=mmlu_pro_math
    #   TASKS=mmlu_pro_math,mmlu_pro_physics,mmlu_pro_computer_science
    # 这种方式仍会跑选中学科的全部题目，适合节省时间并保持该学科分数有效。
    # 若模拟器速度有限，LIMIT_PER_TASK=N 会令每个选中学科各跑前 N 题；
    # 该结果只能用于调试/估测，不是可报告的正式 benchmark 分数。
    TASKS="${TASKS:-mmlu_pro}"
    LIMIT="${LIMIT_PER_TASK:-}"
    ;;
  *)
    echo "用法: $0 [smoke|full] [--thinking|--no-thinking]" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-mmlu_pro_vllm_${MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-./results/${RUN_NAME}}"
# 可选：将终端中的实时状态同时写入文件，便于断线后检查。
# 示例：STATUS_LOG=./logs/mmlu_pro.log bash run_mmlu_pro_vllm.sh full --thinking
STATUS_LOG="${STATUS_LOG:-}"

# tokenized_requests=False 不依赖 vLLM 的 tokenizer 扩展；vLLM 负责实际长度检查。
MODEL_ARGS="model=${USER_VLLM_MODEL},base_url=${VLLM_ROOT}/v1/completions,num_concurrent=${NUM_CONCURRENT},timeout=${API_TIMEOUT},max_retries=${MAX_RETRIES},tokenized_requests=False"

if ! curl --fail --silent --show-error --max-time 5 "${VLLM_ROOT}/v1/models" >/dev/null; then
  echo "无法访问 vLLM: ${VLLM_ROOT}/v1/models" >&2
  exit 1
fi

cmd=(
  python -m lm_eval run
  --model local-completions
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --batch_size "${BATCH_SIZE}"
  --output_path "${OUTPUT_PATH}"
  --log_samples
)

if [[ -n "${GEN_KWARGS}" ]]; then
  cmd+=(--gen_kwargs "${GEN_KWARGS}")
fi

if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

# WRITE_OUT=1 时打印前几条完整 prompt，便于调试。
if [[ "${WRITE_OUT:-0}" == "1" ]]; then
  cmd+=(--write_out)
fi

printf '思考模式：%s；采样 profile：%s\n' "${THINKING_MODE}" "${SAMPLING_PROFILE}"
if [[ "${NUM_CONCURRENT}" == "1" ]]; then
  printf '客户端调度：严格波次；每个 HTTP batch=%s 题，整批响应返回后才发送下一批。\n' "${BATCH_SIZE}"
else
  printf '客户端调度：连续补充；HTTP batch=%s 题，同时最多 %s 个 batch 在飞。\n' "${BATCH_SIZE}" "${NUM_CONCURRENT}"
  printf '%s\n' '提示：若要避免完成即补，设置 NUM_CONCURRENT=1；可同时增大 BATCH_SIZE。'
fi
printf '实际执行命令：\n'
printf ' %q' "${cmd[@]}"
printf '\n结果目录：%s\n' "${OUTPUT_PATH}"
printf '%s\n' '运行时会由 lm-eval 依次显示任务名、构建 prompt 进度和 Requesting API 进度条。'
printf '%s\n' '全量 mmlu_pro 按 14 个学科顺序执行；每个学科的进度条完成后会切换到下一个学科。'

# -u 使 stdout/stderr 尽量实时刷新，避免批处理或终端重定向时长时间看不到进度。
if [[ -n "${STATUS_LOG}" ]]; then
  mkdir -p "$(dirname "${STATUS_LOG}")"
  printf '实时状态日志：%s\n' "${STATUS_LOG}"
  PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "${STATUS_LOG}"
else
  PYTHONUNBUFFERED=1 "${cmd[@]}"
fi

# 结果说明：results_*.json 的 exact_match,custom-extract 是主分数；
# samples_*.jsonl 可检查 resps（原始输出）和 filtered_resps（提取的选项）。
