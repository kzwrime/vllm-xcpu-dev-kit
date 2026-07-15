#!/usr/bin/env bash
# Repeat serve_test/test_multl_stream.py and keep per-iteration evidence.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test_multl_stream.py"

ITERATIONS=10
PRESET_INPUT=""
MAX_TOKENS=16
TEMPERATURE=0.7
OUTPUT_ROOT="$REPO_ROOT/logs/repeat_multl_stream"
CASE_NAME=""
PYTHON_BIN="${PYTHON:-python3}"
STOP_ON_FAIL=0
INCLUDE_LONG=0
HEALTH_CHECK=1

usage() {
    cat <<'USAGE'
用法:
  ./repeat_multl_stream_test.sh -n 50 -e presets/serial/xxx.sh
  ./serve_test/repeat_multl_stream_test.sh -n 50 -e presets/serial/xxx.sh

功能:
  重复执行 serve_test/test_multl_stream.py，用于复现概率性错误。
  每一轮都会创建独立目录，保存:
    - test_multl_stream.output.log  本轮 stdout/stderr
    - vllm_task_*.log               本轮并发任务日志
    - metadata.txt                  本轮命令、时间、退出码
  总目录下会额外生成 summary.tsv，方便快速查看每轮结果。

选项:
  -n, --iterations NUM    重复次数，默认 10
  -e PRESET               预设文件，必填
  --max-tokens NUM        传给 test_multl_stream.py 的 --max-tokens，默认 16
  --temperature NUM       传给 test_multl_stream.py 的 --temperature，默认 0.7
  -o, --output-root DIR   输出根目录，默认 logs/repeat_multl_stream
  --case NAME             问题/场景名称，用于输出目录命名
  --python BIN            Python 命令，默认 python3 或环境变量 PYTHON
  --include-long          设置 VLLM_PD_MULTI_INCLUDE_LONG=1，包含长 prompt 用例
  --no-health-check       跳过启动前 /v1/models 连通性检查
  --stop-on-fail          任一轮非 0 退出后停止
  -h, --help              显示帮助

示例:
  ./serve_test/repeat_multl_stream_test.sh -n 100 -e presets/serial/xxx.sh --case empty-stream
  ./serve_test/repeat_multl_stream_test.sh -n 20 -e presets/serial/xxx.sh --max-tokens 64 --temperature 0 -o logs/issue_1234
USAGE
}

die() {
    echo "[错误] $*" >&2
    exit 1
}

is_positive_int() {
    [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

is_nonnegative_number() {
    [[ "${1:-}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]
}

make_abs_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s\n' "$REPO_ROOT/$path"
    fi
}

make_abs_input_path() {
    local path="$1"
    local dir
    local base

    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    elif [ -f "$path" ]; then
        dir="$(cd "$(dirname "$path")" && pwd)"
        base="$(basename "$path")"
        printf '%s/%s\n' "$dir" "$base"
    else
        printf '%s\n' "$REPO_ROOT/$path"
    fi
}

sanitize_label() {
    printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

make_unique_dir() {
    local parent="$1"
    local name="$2"
    local dir="$parent/$name"
    local idx=1

    while [ -e "$dir" ]; do
        dir="$parent/${name}_$idx"
        idx=$((idx + 1))
    done

    mkdir -p "$dir"
    printf '%s\n' "$dir"
}

resolve_vllm_port() {
    bash -c '
set -e
source "$1/env.sh" >/dev/null
source "$2" >/dev/null
printf "%s\n" "${USER_VLLM_PORT:-}"
' bash "$REPO_ROOT" "$PRESET_FILE"
}

check_vllm_health() {
    local port="$1"
    local url="http://127.0.0.1:${port}/v1/models"

    [ -n "$port" ] || die "无法从预设文件解析 USER_VLLM_PORT"

    if ! curl --silent --show-error --fail --max-time 5 "$url" >/dev/null; then
        cat >&2 <<EOF
[错误] vLLM API 不可访问: $url
       请先启动服务并确认 /v1/models 可访问，例如:
       $REPO_ROOT/run_vllm_test.sh -e $PRESET_FILE --no-test
       curl $url
EOF
        exit 1
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--iterations)
            [ $# -ge 2 ] || die "$1 需要一个数字"
            ITERATIONS="$2"
            shift 2
            ;;
        --iterations=*)
            ITERATIONS="${1#*=}"
            shift
            ;;
        -e)
            [ $# -ge 2 ] || die "-e 需要一个预设文件路径"
            PRESET_INPUT="$2"
            shift 2
            ;;
        --max-tokens)
            [ $# -ge 2 ] || die "--max-tokens 需要一个数字"
            MAX_TOKENS="$2"
            shift 2
            ;;
        --max-tokens=*)
            MAX_TOKENS="${1#*=}"
            shift
            ;;
        --temperature)
            [ $# -ge 2 ] || die "--temperature 需要一个数字"
            TEMPERATURE="$2"
            shift 2
            ;;
        --temperature=*)
            TEMPERATURE="${1#*=}"
            shift
            ;;
        -o|--output-root)
            [ $# -ge 2 ] || die "$1 需要一个目录"
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --output-root=*)
            OUTPUT_ROOT="${1#*=}"
            shift
            ;;
        --case)
            [ $# -ge 2 ] || die "--case 需要一个名称"
            CASE_NAME="$2"
            shift 2
            ;;
        --case=*)
            CASE_NAME="${1#*=}"
            shift
            ;;
        --python)
            [ $# -ge 2 ] || die "--python 需要一个命令"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --python=*)
            PYTHON_BIN="${1#*=}"
            shift
            ;;
        --include-long)
            INCLUDE_LONG=1
            shift
            ;;
        --no-health-check)
            HEALTH_CHECK=0
            shift
            ;;
        --stop-on-fail)
            STOP_ON_FAIL=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "未知参数: $1"
            ;;
    esac
done

is_positive_int "$ITERATIONS" || die "--iterations 必须是大于 0 的整数"
is_positive_int "$MAX_TOKENS" || die "--max-tokens 必须是大于 0 的整数"
is_nonnegative_number "$TEMPERATURE" || die "--temperature 必须是大于等于 0 的数字"
[ -f "$TEST_SCRIPT" ] || die "找不到测试脚本: $TEST_SCRIPT"
[ -n "$PRESET_INPUT" ] || die "请通过 -e 指定预设文件"

PRESET_FILE="$(make_abs_input_path "$PRESET_INPUT")"
OUTPUT_ROOT="$(make_abs_path "$OUTPUT_ROOT")"
[ -f "$PRESET_FILE" ] || die "找不到预设文件: $PRESET_FILE"

VLLM_PORT="$(resolve_vllm_port)"
if [ "$HEALTH_CHECK" -eq 1 ]; then
    check_vllm_health "$VLLM_PORT"
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
if [ -n "$CASE_NAME" ]; then
    RUN_LABEL="$(sanitize_label "$CASE_NAME")"
else
    RUN_LABEL="$(basename "$PRESET_FILE" .sh)"
fi
RUN_DIR="$(make_unique_dir "$OUTPUT_ROOT" "${RUN_STAMP}_${RUN_LABEL}")"
SUMMARY_FILE="$RUN_DIR/summary.tsv"

printf 'iteration\tstatus\texit_code\tstart_time\tend_time\toutput_log\titer_dir\n' > "$SUMMARY_FILE"

echo "[信息] 重复次数: $ITERATIONS"
echo "[信息] 预设文件: $PRESET_FILE"
echo "[信息] vLLM API: http://127.0.0.1:${VLLM_PORT}/v1"
echo "[信息] max_tokens: $MAX_TOKENS"
echo "[信息] temperature: $TEMPERATURE"
echo "[信息] 输出目录: $RUN_DIR"
echo "[信息] Python: $PYTHON_BIN"

if [ "$INCLUDE_LONG" -eq 1 ]; then
    echo "[信息] 长 prompt 用例: enabled"
else
    echo "[信息] 长 prompt 用例: disabled"
fi

overall_exit=0

for ((i = 1; i <= ITERATIONS; i++)); do
    iter_name="$(printf 'iter_%04d' "$i")"
    iter_dir="$RUN_DIR/$iter_name"
    output_log="$iter_dir/test_multl_stream.output.log"
    metadata_file="$iter_dir/metadata.txt"
    start_time="$(date '+%Y-%m-%d %H:%M:%S %z')"

    mkdir -p "$iter_dir"

    {
        echo "iteration=$i"
        echo "start_time=$start_time"
        echo "workdir=$iter_dir"
        echo "python=$PYTHON_BIN"
        echo "test_script=$TEST_SCRIPT"
        echo "preset=$PRESET_FILE"
        echo "max_tokens=$MAX_TOKENS"
        echo "temperature=$TEMPERATURE"
        echo "include_long=$INCLUDE_LONG"
        echo "command=$PYTHON_BIN $TEST_SCRIPT -e $PRESET_FILE --max-tokens $MAX_TOKENS --temperature $TEMPERATURE"
    } > "$metadata_file"

    echo ""
    echo "[信息] 开始第 $i/$ITERATIONS 轮: $iter_dir"

    (
        cd "$iter_dir" || exit 1
        if [ "$INCLUDE_LONG" -eq 1 ]; then
            export VLLM_PD_MULTI_INCLUDE_LONG=1
        fi
        "$PYTHON_BIN" "$TEST_SCRIPT" -e "$PRESET_FILE" --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE"
    ) > "$output_log" 2>&1
    exit_code=$?

    end_time="$(date '+%Y-%m-%d %H:%M:%S %z')"
    shopt -s nullglob
    task_logs=("$iter_dir"/vllm_task_*.log)
    shopt -u nullglob

    if [ "$exit_code" -eq 0 ]; then
        scan_files=("$output_log" "${task_logs[@]}")
        if grep -Eq '\[警告\]|\[请求失败\]|\[空输出诊断\]|空正文|空输出' "${scan_files[@]}" 2>/dev/null; then
            status="warning"
            echo "[警告] 第 $i/$ITERATIONS 轮完成但发现告警/空输出线索"
        else
            status="success"
            echo "[信息] 第 $i/$ITERATIONS 轮完成: success"
        fi
    else
        status="failed"
        overall_exit=1
        echo "[警告] 第 $i/$ITERATIONS 轮失败: exit_code=$exit_code"
    fi

    task_log_names=""
    if [ "${#task_logs[@]}" -gt 0 ]; then
        task_log_names="$(printf '%s\n' "${task_logs[@]##*/}" | paste -sd ' ' -)"
    fi

    {
        echo "end_time=$end_time"
        echo "exit_code=$exit_code"
        echo "status=$status"
        echo "output_log=$output_log"
        echo "task_logs=$task_log_names"
    } >> "$metadata_file"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$i" "$status" "$exit_code" "$start_time" "$end_time" "$output_log" "$iter_dir" \
        >> "$SUMMARY_FILE"

    if [ "$exit_code" -ne 0 ] && [ "$STOP_ON_FAIL" -eq 1 ]; then
        echo "[警告] --stop-on-fail enabled，停止后续轮次"
        break
    fi
done

echo ""
echo "[信息] 全部轮次结束"
echo "[信息] 汇总文件: $SUMMARY_FILE"
echo "[信息] 证据目录: $RUN_DIR"

exit "$overall_exit"
