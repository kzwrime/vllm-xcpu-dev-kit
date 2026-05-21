#!/bin/bash
# vLLM startup + test wrapper.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
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
  --bench            启动服务后运行 bench
  --coverage         启动服务后运行 coverage bench，并 dump shapes
  --pd               以 P/D 分离模式运行
  --launcher MODE    强制指定启动方式: auto | mp | mpi
  -h, --help         显示帮助

环境变量:
  VLLM_TEST_MAX_WAIT   服务启动最大等待时间（秒），默认 300
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
DISAGG_PREFILL=0
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
        --bench)
            TEST_MODE="bench"
            shift
            ;;
        --coverage)
            TEST_MODE="coverage"
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

launcher_load_config
launcher_check_required_env
mkdir -p "$BACKUP_ROOT" "$SUCCESS_ROOT" "$FAILED_ROOT"
launcher_prepare_runtime "run_vllm_test.log"

TEST_LOG="$LOG_DIR/test.log"
BENCH_LOG="$LOG_DIR/bench.log"

backup_old_logs() {
    local patterns=(
        "$LOG_DIR"/run_vllm_test.log
        "$LOG_DIR"/run_vllm_test_*.log
        "$LOG_DIR"/test.log
        "$LOG_DIR"/test_*.log
        "$SCRIPT_DIR"/vllm_task_*.log
        "$LOG_DIR"/bench.log
        "$LOG_DIR"/bench_*.log
        "$LOG_DIR"/mpi_cleanup.log
        "$LOG_DIR"/mpi_cleanup_*.log
        "$LOG_DIR"/mpi_workers.log
        "$LOG_DIR"/mpi_workers_*.log
        "$LOG_DIR"/vllm_head_log.txt
        "$LOG_DIR"/vllm_head_log.txt.old
        "$LOG_DIR"/vllm_serve_log_dp_rank*.txt
        "$LOG_DIR"/vllm_worker_log_rank*.txt
        "$LOG_DIR"/vllm_worker_log_rank*.txt.old
        "$LOG_DIR"/vllm_serve_log.txt
        "$LOG_DIR"/vllm_serve_log.txt.old
    )
    local files=()
    local file
    local latest_mtime=0
    local backup_stamp
    local backup_dir

    shopt -s nullglob
    for pattern in "${patterns[@]}"; do
        for file in $pattern; do
            [ -f "$file" ] || continue
            files+=("$file")
            local mtime
            mtime=$(stat -c %Y "$file")
            if [ "$mtime" -gt "$latest_mtime" ]; then
                latest_mtime="$mtime"
            fi
        done
    done
    shopt -u nullglob

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
    mkdir -p "$dest_dir"

    find "$LOG_DIR" -mindepth 1 \( \
        -path "$BACKUP_ROOT" -o \
        -path "$BACKUP_ROOT/*" -o \
        -path "$SUCCESS_ROOT" -o \
        -path "$SUCCESS_ROOT/*" -o \
        -path "$FAILED_ROOT" -o \
        -path "$FAILED_ROOT/*" -o \
        -name '.gitignore' \
    \) -prune -o -type f -print | while read -r file; do
        cp -p "$file" "$dest_dir/"
    done

    shopt -s nullglob
    for file in "$SCRIPT_DIR"/vllm_task_*.log; do
        cp -p "$file" "$dest_dir/"
    done
    shopt -u nullglob
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

    if [ "$exit_code" -eq 0 ] && [ "$TEST_MODE" != "none" ]; then
        archive_run_logs "$SUCCESS_ROOT" "成功"
    elif [ "$exit_code" -ne 0 ]; then
        archive_run_logs "$FAILED_ROOT" "失败"
    fi

    launcher_cleanup_processes
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

    shopt -s nullglob
    files=("$SCRIPT_DIR"/vllm_task_*.log)
    shopt -u nullglob

    if [ "${#files[@]}" -eq 0 ]; then
        log_warning "未找到 multi test 结果日志: $SCRIPT_DIR/vllm_task_*.log"
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

run_test() {
    TEST_EXIT_CODE=0

    if [ "$TEST_MODE" = "test" ]; then
        log_info "运行测试..."
        local test_output
        if test_output=$(bash "$SCRIPT_DIR/serve_test/serve_test_template.sh" "${TEST_ENV_ARGS[@]}" 2>&1); then
            printf '%s\n' "$test_output" | tee "$TEST_LOG"
            log_success "测试完成"
        else
            TEST_EXIT_CODE=$?
            printf '%s\n' "$test_output" | tee "$TEST_LOG"
            log_warning "测试退出码: $TEST_EXIT_CODE"
        fi

        log_info "模型回答:"
        extract_model_reply "$test_output"
        log_info "测试日志: $TEST_LOG"
    elif [ "$TEST_MODE" = "multi" ]; then
        log_info "运行并发 multi test..."
        if python "$SCRIPT_DIR/serve_test/test_multl_stream.py" "${TEST_ENV_ARGS[@]}" --max-tokens "$MULTI_TEST_MAX_TOKENS" > "$TEST_LOG" 2>&1; then
            log_success "Multi test 完成"
        else
            TEST_EXIT_CODE=$?
            log_warning "Multi test 退出码: $TEST_EXIT_CODE"
        fi
        log_info "测试日志: $TEST_LOG"
        cat_multi_test_results
    elif [ "$TEST_MODE" = "bench" ]; then
        log_info "运行 bench..."
        if bash "$SCRIPT_DIR/serve_test/serve_bench_template.sh" "${TEST_ENV_ARGS[@]}" > "$BENCH_LOG" 2>&1; then
            log_success "Bench 完成"
        else
            TEST_EXIT_CODE=$?
            log_warning "Bench 退出码: $TEST_EXIT_CODE"
        fi
        log_info "Bench 日志: $BENCH_LOG"
    elif [ "$TEST_MODE" = "coverage" ]; then
        log_info "运行 coverage bench..."
        if bash "$SCRIPT_DIR/serve_test/serve_bench_coverage.sh" "${TEST_ENV_ARGS[@]}" > "$BENCH_LOG" 2>&1; then
            log_success "Coverage bench 完成"
        else
            TEST_EXIT_CODE=$?
            log_warning "Coverage bench 退出码: $TEST_EXIT_CODE"
        fi
        log_info "Bench 日志: $BENCH_LOG"
    else
        log_info "跳过测试，服务保持运行"
        log_info "手动测试示例: curl http://localhost:${USER_VLLM_PORT}/v1/models"
    fi

    return "$TEST_EXIT_CODE"
}

log_info "========================================="
log_info "  VLLM 启动与测试"
log_info "========================================="
log_info "配置来源: $CONFIG_SOURCE"
log_info "Preset 标识: $PRESET_TAG"
log_info "模型: $USER_VLLM_MODEL"
log_info "端口: $USER_VLLM_PORT"
log_info "并行配置: DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
log_info "测试模式: $TEST_MODE"
log_info "启动器: $LAUNCHER"
if [ "$DISAGG_PREFILL" -eq 1 ]; then
    log_info "P/D 分离模式: enabled"
fi

if [ "$TEST_MODE" = "coverage" ]; then
    export TORCH_XCPU_DUMP_SHAPES=1
    export TORCH_XCPU_DUMP_SHAPES_OUTPUT_DIR="./dump_shape/${PRESET_NAME}"
    log_info "Dump shapes: TORCH_XCPU_DUMP_SHAPES=$TORCH_XCPU_DUMP_SHAPES"
    log_info "Dump shapes 输出目录: $TORCH_XCPU_DUMP_SHAPES_OUTPUT_DIR"
fi

backup_old_logs
launcher_start_service
launcher_wait_for_service

if [ "$DISAGG_PREFILL" -eq 1 ] && [ "$TEST_MODE" = "multi" ]; then
    export VLLM_PD_MULTI_INCLUDE_LONG=1
fi
run_test
pd_validate_transfer

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
