#!/bin/bash
# vLLM startup + test wrapper.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${VLLM_TEST_LOG_DIR:-$SCRIPT_DIR/logs}"
if [[ "$LOG_DIR" != /* ]]; then
    LOG_DIR="$PWD/${LOG_DIR#./}"
fi
BACKUP_ROOT="$LOG_DIR/backups"
SUCCESS_ROOT="$LOG_DIR/success"
FAILED_ROOT="$LOG_DIR/failed"
LAUNCHER_COMMON_SH="$SCRIPT_DIR/launcher_common.sh"

if [ ! -f "$LAUNCHER_COMMON_SH" ]; then
    echo "Could not find $LAUNCHER_COMMON_SH" >&2
    exit 1
fi
source "$LAUNCHER_COMMON_SH"
launcher_common_init

usage() {
    cat <<'USAGE'
用法:
  ./run_vllm_test.sh -e presets/serial/Qwen3-0.6B_dp1_tp1_eager.sh
  ./run_vllm_test.sh --pd -e presets/mpi/moe/Qwen3-30B-A3B_dp2_tp2_ep_eager_alltoallv_v2.sh --multi-test

选项:
  -e <preset_file>   指定预设文件路径
  --no-test          只启动服务，不运行测试
  --multi-test       启动服务后运行 serve_test/test_multl_stream.py
  --multi-test-max-tokens NUM
                     multi test 每个请求的最大输出 token 数，默认 16
  --multi-test-temperature NUM
                     multi test 采样温度，默认 0.7；设为 0 时使用 greedy 解码
  --bench            启动服务后运行 bench
  --coverage         启动服务后运行 coverage bench，并 dump shapes
  --profile          在普通测试或 multi test 前后调用 vLLM profiler
  --test-timeout SECONDS
                     测试 / multi test / bench 最长运行时间（秒），默认不限制
  --pd               以 P/D 分离模式运行
  --launcher MODE    强制指定启动方式: auto | mp | mpi
  --auto-port        如果 USER_VLLM_PORT 被占用，则从该端口开始寻找空闲端口
  -h, --help         显示帮助

环境变量:
  VLLM_TEST_LOG_DIR   本次运行的日志根目录，默认 <vllm_scripts>/logs
  VLLM_TEST_MAX_WAIT   服务启动最大等待时间（秒），默认 300
  RUN_VLLM_TEST_TIMEOUT
                       测试 / multi test / bench 最长运行时间（秒），默认不限制
  RUN_VLLM_TEST_AUTO_PORT
                       设为 1/true/yes/on 时等价于 --auto-port
USAGE
}

TEST_MODE="test"
LAUNCHER="${RUN_VLLM_TEST_LAUNCHER:-auto}"
PRESET_FILE_INPUT=""
ENV_ARGS=()
RUN_START_TS="$(date +%Y%m%d_%H%M%S)"
PRESET_TAG=""
PRESET_NAME=""
TEST_EXIT_CODE=0
MULTI_TEST_MAX_TOKENS=16
MULTI_TEST_TEMPERATURE=0.7
TEST_TIMEOUT="${RUN_VLLM_TEST_TIMEOUT:-}"
DISAGG_PREFILL=0
PROFILE_TEST=0
REQUESTED_NO_TEST=0
REQUESTED_BENCH=0
REQUESTED_COVERAGE=0
TEST_ENV_ARGS=()
ORIG_CONFIG_FILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        -e)
            if [ $# -lt 2 ]; then
                log_error "-e 需要一个预设文件路径"
                usage
                exit 1
            fi
            PRESET_FILE_INPUT="$2"
            ENV_ARGS=("-e" "$2")
            shift 2
            ;;
        --no-test)
            TEST_MODE="none"
            REQUESTED_NO_TEST=1
            shift
            ;;
        --multi-test)
            TEST_MODE="multi"
            shift
            ;;
        --multi-test-max-tokens)
            if [ $# -lt 2 ]; then
                log_error "--multi-test-max-tokens 需要一个数字"
                usage
                exit 1
            fi
            MULTI_TEST_MAX_TOKENS="$2"
            shift 2
            ;;
        --multi-test-max-tokens=*)
            MULTI_TEST_MAX_TOKENS="${1#*=}"
            shift
            ;;
        --multi-test-temperature)
            if [ $# -lt 2 ]; then
                log_error "--multi-test-temperature 需要一个数字"
                usage
                exit 1
            fi
            MULTI_TEST_TEMPERATURE="$2"
            shift 2
            ;;
        --multi-test-temperature=*)
            MULTI_TEST_TEMPERATURE="${1#*=}"
            shift
            ;;
        --bench)
            TEST_MODE="bench"
            REQUESTED_BENCH=1
            shift
            ;;
        --coverage)
            TEST_MODE="coverage"
            REQUESTED_COVERAGE=1
            shift
            ;;
        --profile)
            PROFILE_TEST=1
            shift
            ;;
        --test-timeout)
            if [ $# -lt 2 ]; then
                log_error "--test-timeout 需要一个数字"
                usage
                exit 1
            fi
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        --test-timeout=*)
            TEST_TIMEOUT="${1#*=}"
            shift
            ;;
        --pd|--disagg-prefill)
            DISAGG_PREFILL=1
            shift
            ;;
        --launcher)
            if [ $# -lt 2 ]; then
                log_error "--launcher 需要一个值"
                usage
                exit 1
            fi
            LAUNCHER="$2"
            shift 2
            ;;
        --launcher=*)
            LAUNCHER="${1#*=}"
            shift
            ;;
        --auto-port)
            export RUN_VLLM_TEST_AUTO_PORT=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

launcher_validate_launcher
if ! [[ "$MULTI_TEST_MAX_TOKENS" =~ ^[0-9]+$ ]] || [ "$MULTI_TEST_MAX_TOKENS" -eq 0 ]; then
    log_error "--multi-test-max-tokens 必须是大于 0 的整数"
    exit 1
fi
if ! [[ "$MULTI_TEST_TEMPERATURE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    log_error "--multi-test-temperature 必须是大于等于 0 的数字"
    exit 1
fi
if [ "$PROFILE_TEST" -eq 1 ]; then
    if [ "$REQUESTED_NO_TEST" -eq 1 ] || [ "$REQUESTED_BENCH" -eq 1 ] || [ "$REQUESTED_COVERAGE" -eq 1 ]; then
        log_error "--profile 不能和 --bench/--coverage/--no-test 混用"
        exit 1
    fi

    case "$TEST_MODE" in
        test|multi)
            ;;
        *)
            log_error "--profile 只能和普通测试或 --multi-test 一起使用"
            exit 1
            ;;
    esac
fi
if [ -n "$TEST_TIMEOUT" ]; then
    if ! [[ "$TEST_TIMEOUT" =~ ^[0-9]+$ ]] || [ "$TEST_TIMEOUT" -eq 0 ]; then
        log_error "--test-timeout / RUN_VLLM_TEST_TIMEOUT 必须是大于 0 的整数秒"
        exit 1
    fi
    if ! command -v timeout >/dev/null 2>&1; then
        log_error "启用测试超时需要 timeout 命令"
        exit 1
    fi
fi

launcher_load_config

launcher_auto_configure_modelscope() {
    case "${VLLM_USE_MODELSCOPE:-}" in
        [aA][uU][tT][oO])
            local cache_status
            local hf_found=0
            local modelscope_found=0

            # Use the libraries' own cache resolution and repository lookup.
            # local_files_only prevents this probe from contacting either hub.
            cache_status="$({
                python3 - "${USER_VLLM_MODEL:-}" <<'PY'
import sys

model = sys.argv[1]
found = {"hf": 0, "modelscope": 0}

if model:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=model, local_files_only=True)
        found["hf"] = 1
    except Exception:
        pass

    try:
        from modelscope import snapshot_download

        snapshot_download(model_id=model, local_files_only=True)
        found["modelscope"] = 1
    except Exception:
        pass

print(f"{found['hf']} {found['modelscope']}")
PY
            } 2>/dev/null)"

            read -r hf_found modelscope_found <<< "$cache_status"
            [[ "$hf_found" == 1 ]] || hf_found=0
            [[ "$modelscope_found" == 1 ]] || modelscope_found=0

            if [ "$hf_found" -eq 1 ]; then
                export VLLM_USE_MODELSCOPE=False
            elif [ "$modelscope_found" -eq 1 ]; then
                export VLLM_USE_MODELSCOPE=True
            else
                export VLLM_USE_MODELSCOPE=False
            fi

            log_info "自动检测模型缓存: ${USER_VLLM_MODEL}"
            log_info "  Hugging Face: $([ "$hf_found" -eq 1 ] && echo found || echo not-found)"
            log_info "  ModelScope: $([ "$modelscope_found" -eq 1 ] && echo found || echo not-found)"
            log_info "已设置 VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE}"
            ;;
    esac
}

launcher_auto_configure_modelscope
launcher_check_required_env
mkdir -p "$BACKUP_ROOT" "$SUCCESS_ROOT" "$FAILED_ROOT"
launcher_prepare_runtime "run_vllm_test.log"

TEST_LOG="$LOG_DIR/test.log"
BENCH_LOG="$LOG_DIR/bench.log"
PROFILER_LOG="$LOG_DIR/profiler.log"

backup_old_logs() {
    local files=()
    local file
    local latest_mtime=0
    local backup_stamp
    local backup_dir

    while IFS= read -r -d '' file; do
        files+=("$file")
        local mtime
        mtime=$(stat -c %Y "$file")
        if [ "$mtime" -gt "$latest_mtime" ]; then
            latest_mtime="$mtime"
        fi
    done < <(find "$LOG_DIR" -maxdepth 1 -type f ! -name '.gitignore' -print0)

    [ "${#files[@]}" -eq 0 ] && return

    backup_stamp=$(date -d "@${latest_mtime}" +%Y%m%d_%H%M%S)
    backup_dir=$(make_unique_dir "$BACKUP_ROOT" "${backup_stamp}")
    mkdir -p "$backup_dir"

    for file in "${files[@]}"; do
        mv "$file" "$backup_dir/"
    done

    log_info "已备份旧日志到: $backup_dir"
}

copy_current_logs() {
    local dest_dir="$1"
    local file
    mkdir -p "$dest_dir"

    if [ "$DISAGG_PREFILL" -eq 1 ]; then
        if [ -n "${PD_ROOT:-}" ] && [ -d "$PD_ROOT" ]; then
            mkdir -p "$dest_dir/pd"
            cp -a "$PD_ROOT" "$dest_dir/pd/"
        fi
        for file in "$TEST_LOG" "$BENCH_LOG"; do
            [ -f "$file" ] && cp -p "$file" "$dest_dir/"
        done
        return
    fi

    while IFS= read -r -d '' file; do
        cp -p "$file" "$dest_dir/"
    done < <(find "$LOG_DIR" -maxdepth 1 -type f ! -name '.gitignore' -print0)
}

archive_run_logs() {
    local root="$1"
    local label="$2"
    local archive_dir

    archive_dir=$(make_unique_dir "$root" "${RUN_START_TS}_${PRESET_TAG}")
    copy_current_logs "$archive_dir"
    log_info "${label}日志已归档到: $archive_dir"
}

cleanup() {
    local exit_code=$?
    local archive_root=""
    local archive_label=""

    if [ "$exit_code" -eq 0 ] && [ "$TEST_MODE" != "none" ]; then
        archive_root="$SUCCESS_ROOT"
        archive_label="成功"
    elif [ "$exit_code" -ne 0 ]; then
        archive_root="$FAILED_ROOT"
        archive_label="失败"
    fi

    launcher_cleanup_processes
    if [ -n "$archive_root" ]; then
        archive_run_logs "$archive_root" "$archive_label"
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

extract_model_reply() {
    local content=""
    local raw_output="$1"

    if command -v jq >/dev/null 2>&1; then
        content=$(printf '%s' "$raw_output" | jq -r '.choices[0].message.content // .choices[0].text // .choices[0].message.reasoning // empty' 2>/dev/null || true)
    fi

    if [ -z "$content" ]; then
        content=$(printf '%s' "$raw_output" | grep -oP '"(content|text|reasoning)":\s*"\K[^"]*' | head -1 || true)
    fi

    if [ -n "$content" ]; then
        echo -e "${GREEN}${content}${NC}"
    else
        log_warning "未能从响应中提取 content/text/reasoning 字段"
    fi
}

cat_multi_test_results() {
    local files=()
    local file
    local log_dir="$LOG_DIR"

    if [ "$DISAGG_PREFILL" -eq 1 ] && [ -n "${PD_ROOT:-}" ]; then
        log_dir="$PD_ROOT"
    fi

    shopt -s nullglob
    files=("$log_dir"/vllm_task_*.log)
    shopt -u nullglob

    if [ "${#files[@]}" -eq 0 ]; then
        log_warning "未找到 multi test 结果日志: $log_dir/vllm_task_*.log"
        return
    fi

    echo ""
    log_info "Multi test 结果汇总:"
    for file in "${files[@]}"; do
        echo ""
        echo "===== ${file} ====="
        cat "$file"
        echo ""
    done
}

call_profiler_endpoint() {
    local label="$1"
    local endpoint="$2"
    local url="http://127.0.0.1:${USER_VLLM_PORT}/${endpoint}"
    local rc=0

    log_info "${label} profiler: $url"
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') ${endpoint} =====" >> "$PROFILER_LOG"
    if curl --silent --show-error --fail -X POST "$url" >> "$PROFILER_LOG" 2>&1; then
        echo "" >> "$PROFILER_LOG"
        return 0
    fi

    rc=$?
    echo "" >> "$PROFILER_LOG"
    return "$rc"
}

start_profiler() {
    : > "$PROFILER_LOG"
    if call_profiler_endpoint "启动" "start_profile"; then
        log_success "Profiler 已启动"
        return 0
    fi

    log_error "启动 profiler 失败，详情见: $PROFILER_LOG"
    return 1
}

stop_profiler() {
    if call_profiler_endpoint "停止" "stop_profile"; then
        log_success "Profiler 已停止"
        log_info "Profiler 日志: $PROFILER_LOG"
        return 0
    fi

    log_error "停止 profiler 失败，详情见: $PROFILER_LOG"
    return 1
}

run_with_test_timeout() {
    if [ -n "$TEST_TIMEOUT" ]; then
        timeout --kill-after=30s "${TEST_TIMEOUT}s" "$@"
    else
        "$@"
    fi
}

log_test_exit() {
    local label="$1"
    local exit_code="$2"

    if [ -n "$TEST_TIMEOUT" ] && [ "$exit_code" -eq 124 ]; then
        log_warning "${label} 超过最长运行时间 ${TEST_TIMEOUT} 秒"
    fi
    log_warning "${label} 退出码: $exit_code"
}

run_test() {
    TEST_EXIT_CODE=0
    local stop_profile_rc=0

    if [ "$PROFILE_TEST" -eq 1 ]; then
        if ! start_profiler; then
            return 1
        fi
    fi

    if [ "$TEST_MODE" = "test" ]; then
        log_info "运行测试..."
        [ -n "$TEST_TIMEOUT" ] && log_info "测试最长运行时间: ${TEST_TIMEOUT} 秒"
        local test_output
        if test_output=$(run_with_test_timeout bash "$SCRIPT_DIR/serve_test/serve_test_template.sh" "${TEST_ENV_ARGS[@]}" 2>&1); then
            printf '%s\n' "$test_output" | tee "$TEST_LOG"
            log_success "测试完成"
        else
            TEST_EXIT_CODE=$?
            printf '%s\n' "$test_output" | tee "$TEST_LOG"
            log_test_exit "测试" "$TEST_EXIT_CODE"
        fi

        log_info "模型回答:"
        extract_model_reply "$test_output"
        log_info "测试日志: $TEST_LOG"
    elif [ "$TEST_MODE" = "multi" ]; then
        log_info "运行并发 multi test..."
        [ -n "$TEST_TIMEOUT" ] && log_info "Multi test 最长运行时间: ${TEST_TIMEOUT} 秒"
        local multi_log_dir="$LOG_DIR"
        if [ "$DISAGG_PREFILL" -eq 1 ] && [ -n "${PD_ROOT:-}" ]; then
            multi_log_dir="$PD_ROOT"
        fi
        mkdir -p "$multi_log_dir"
        if (cd "$multi_log_dir" && run_with_test_timeout python "$SCRIPT_DIR/serve_test/test_multl_stream.py" "${TEST_ENV_ARGS[@]}" --max-tokens "$MULTI_TEST_MAX_TOKENS" --temperature "$MULTI_TEST_TEMPERATURE") > "$TEST_LOG" 2>&1; then
            log_success "Multi test 完成"
        else
            TEST_EXIT_CODE=$?
            log_test_exit "Multi test" "$TEST_EXIT_CODE"
        fi
        log_info "测试日志: $TEST_LOG"
        cat_multi_test_results
    elif [ "$TEST_MODE" = "bench" ]; then
        log_info "运行 bench..."
        [ -n "$TEST_TIMEOUT" ] && log_info "Bench 最长运行时间: ${TEST_TIMEOUT} 秒"
        if run_with_test_timeout bash "$SCRIPT_DIR/serve_test/serve_bench_template.sh" "${TEST_ENV_ARGS[@]}" > "$BENCH_LOG" 2>&1; then
            log_success "Bench 完成"
        else
            TEST_EXIT_CODE=$?
            log_test_exit "Bench" "$TEST_EXIT_CODE"
        fi
        log_info "Bench 日志: $BENCH_LOG"
    elif [ "$TEST_MODE" = "coverage" ]; then
        log_info "运行 coverage bench..."
        [ -n "$TEST_TIMEOUT" ] && log_info "Coverage bench 最长运行时间: ${TEST_TIMEOUT} 秒"
        if run_with_test_timeout bash "$SCRIPT_DIR/serve_test/serve_bench_coverage.sh" "${TEST_ENV_ARGS[@]}" > "$BENCH_LOG" 2>&1; then
            log_success "Coverage bench 完成"
        else
            TEST_EXIT_CODE=$?
            log_test_exit "Coverage bench" "$TEST_EXIT_CODE"
        fi
        log_info "Bench 日志: $BENCH_LOG"
    else
        log_info "跳过测试，服务保持运行"
        log_info "手动测试示例: curl http://localhost:${USER_VLLM_PORT}/v1/models"
    fi

    if [ "$PROFILE_TEST" -eq 1 ]; then
        if stop_profiler; then
            :
        else
            stop_profile_rc=$?
        fi
    fi

    if [ "$TEST_EXIT_CODE" -eq 0 ] && [ "$stop_profile_rc" -ne 0 ]; then
        TEST_EXIT_CODE="$stop_profile_rc"
    fi

    return "$TEST_EXIT_CODE"
}

log_info "========================================="
log_info "  VLLM 启动与测试"
log_info "========================================="
log_info "配置来源: $CONFIG_SOURCE"
log_info "Preset 标识: $PRESET_TAG"
log_info "模型: $USER_VLLM_MODEL"
[ -n "${TORCH_XCPU_FP8_MOE_BACKEND:-}" ] && \
    log_info "FP8 MoE compute backend: $TORCH_XCPU_FP8_MOE_BACKEND"
log_info "端口: $USER_VLLM_PORT"
log_info "并行配置: DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
log_info "测试模式: $TEST_MODE"
[ -n "$TEST_TIMEOUT" ] && log_info "测试最长运行时间: ${TEST_TIMEOUT} 秒"
log_info "启动器: $LAUNCHER"
if [ "$PROFILE_TEST" -eq 1 ]; then
    log_info "Profiler: enabled"
fi
if [ "$DISAGG_PREFILL" -eq 1 ]; then
    log_info "P/D 分离模式: enabled"
fi

if [ "$TEST_MODE" = "coverage" ]; then
    export TORCH_XCPU_DUMP_SHAPES=1
    export TORCH_XCPU_DUMP_SHAPES_OUTPUT_DIR="./dump_shape/${PRESET_NAME}"
    log_info "Dump shapes: TORCH_XCPU_DUMP_SHAPES=$TORCH_XCPU_DUMP_SHAPES"
    log_info "Dump shapes 输出目录: $TORCH_XCPU_DUMP_SHAPES_OUTPUT_DIR"
fi

if [ "$DISAGG_PREFILL" -eq 1 ]; then
    log_info "P/D 分离模式跳过旧日志备份"
else
    backup_old_logs
fi
launcher_start_service
launcher_wait_for_service
if [ "$DISAGG_PREFILL" -eq 1 ]; then
    launcher_start_error_monitor "$$"
fi

if [ "$DISAGG_PREFILL" -eq 1 ] && [ "$TEST_MODE" = "multi" ]; then
    export VLLM_PD_MULTI_INCLUDE_LONG=1
fi
if run_test; then
    :
else
    test_rc=$?
    launcher_print_error_details || true
    exit "$test_rc"
fi
if [ "$DISAGG_PREFILL" -eq 1 ] && [ "$TEST_MODE" != "none" ]; then
    if launcher_print_error_details; then
        exit 1
    fi
fi
pd_validate_transfer
if [ "$DISAGG_PREFILL" -eq 1 ] && [ "$TEST_MODE" != "none" ]; then
    launcher_stop_error_monitor
fi

echo ""
log_info "日志文件位置:"
launcher_print_service_locations
if [ "$TEST_MODE" = "test" ] || [ "$TEST_MODE" = "multi" ]; then
    log_info "  Test:  $TEST_LOG"
elif [ "$TEST_MODE" = "bench" ] || [ "$TEST_MODE" = "coverage" ]; then
    log_info "  Bench: $BENCH_LOG"
fi
log_info "  Cleanup: $MPI_CLEANUP_LOG"

if [ "$TEST_MODE" = "none" ]; then
    echo ""
    log_info "服务正在运行，按 Ctrl+C 停止"
    wait
fi
