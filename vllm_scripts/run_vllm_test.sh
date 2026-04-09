#!/bin/bash
# vLLM 启动+测试包装脚本
#
# 用法与 serve/serve_mp_template.sh、serve_test/serve_test_template.sh 保持一致：
#   方式1: 通过 -e 参数指定预设文件
#     ./run_vllm_test.sh -e ./presets/serial/Qwen3-0.6B_dp1_tp1_eager.sh
#
#   方式2: 通过 PRESET 环境变量
#     PRESET=serial/Qwen3-0.6B_dp1_tp1_eager ./run_vllm_test.sh
#
#   方式3: 使用 user_env.sh
#     ./run_vllm_test.sh
#
# 额外选项:
#   --no-test         只启动服务，不运行测试
#   --bench           启动服务后运行 serve_test/serve_bench_template.sh
#   --launcher MODE   强制指定启动方式: auto(默认) | mp | mpi

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
BACKUP_ROOT="$LOG_DIR/backups"
SUCCESS_ROOT="$LOG_DIR/success"
FAILED_ROOT="$LOG_DIR/failed"
COMMON_SH="$SCRIPT_DIR/common.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

usage() {
    sed -n '1,18p' "$0"
    cat <<'USAGE'
  选项:
    -e <preset_file>   指定预设文件路径
    --no-test          只启动服务，不运行测试
    --bench            启动服务后运行 bench
    --launcher MODE    强制指定启动方式: auto | mp | mpi
    -h, --help         显示帮助
USAGE
}

sanitize_name() {
    printf '%s' "$1" | sed -E 's#^\./##; s#^presets/##; s#\.sh$##; s#[^A-Za-z0-9._-]+#_#g; s#_+#_#g; s#^_##; s#_$##'
}

make_unique_dir() {
    local root="$1"
    local base="$2"
    local dir="$root/$base"
    local idx=1

    while [ -e "$dir" ]; do
        dir="$root/${base}_$idx"
        idx=$((idx + 1))
    done

    printf '%s' "$dir"
}

if [ ! -f "$COMMON_SH" ]; then
    log_error "Could not find $COMMON_SH"
    exit 1
fi
source "$COMMON_SH"

TEST_MODE="test"
LAUNCHER="${RUN_VLLM_TEST_LAUNCHER:-auto}"
PRESET_FILE_INPUT=""
ENV_ARGS=()
RUN_START_TS="$(date +%Y%m%d_%H%M%S)"
PRESET_TAG=""
TEST_EXIT_CODE=0

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
        --bench)
            TEST_MODE="bench"
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

case "$LAUNCHER" in
    auto|mp|mpi)
        ;;
    *)
        log_error "无效的 --launcher 值: $LAUNCHER"
        exit 1
        ;;
esac

load_env_file "$SCRIPT_DIR/env.sh"
if [ -n "$PRESET_FILE_INPUT" ]; then
    load_preset_file "$PRESET_FILE_INPUT"
    CONFIG_SOURCE="$PRESET_FILE_INPUT"
    PRESET_TAG="$(sanitize_name "$PRESET_FILE_INPUT")"
elif [ -n "${PRESET:-}" ]; then
    load_user_config "$SCRIPT_DIR"
    CONFIG_SOURCE="PRESET=${PRESET}"
    PRESET_TAG="$(sanitize_name "$PRESET")"
else
    load_user_config "$SCRIPT_DIR"
    CONFIG_SOURCE="user_env.sh / user_env_template.sh"
    PRESET_TAG="user_env"
fi

[ -n "$PRESET_TAG" ] || PRESET_TAG="user_env"

check_and_print_env "USER_VLLM_MODEL"
check_and_print_env "USER_VLLM_PORT"
check_and_print_env "USER_VLLM_DATA_PARALLEL_SIZE"
check_and_print_env "USER_VLLM_TP_SIZE"
check_and_print_env "USER_VLLM_PP_SIZE"

mkdir -p "$LOG_DIR" "$BACKUP_ROOT" "$SUCCESS_ROOT" "$FAILED_ROOT"
cd "$SCRIPT_DIR"

TEST_LOG="$LOG_DIR/test.log"
BENCH_LOG="$LOG_DIR/bench.log"
MPI_CLEANUP_LOG="$LOG_DIR/mpi_cleanup.log"
MPI_WORKERS_LOG="$LOG_DIR/mpi_workers.log"
LAUNCH_LOG="$LOG_DIR/run_vllm_test.log"

MP_SERVE_LOG="$SCRIPT_DIR/vllm_serve_log.txt"
HEAD_SERVE_LOG="$LOG_DIR/vllm_head_log.txt"

USER_VLLM_MPC_SIZE="${USER_VLLM_MPC_SIZE:-$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))}"
MPI_COUNT=$((USER_VLLM_DATA_PARALLEL_SIZE * USER_VLLM_MPC_SIZE))

if [ "$LAUNCHER" = "auto" ]; then
    if [ "${VLLM_CPU_USE_MPI:-0}" = "1" ] || [ "${VLLM_USE_MPI_COORD:-0}" = "1" ] || [ -n "${USER_VLLM_MP_RPC_WORKER_PER_NODE:-}" ]; then
        LAUNCHER="mpi"
    else
        LAUNCHER="mp"
    fi
fi

PIDS=()
record_pid() {
    PIDS+=("$1")
}

backup_old_logs() {
    local patterns=(
        "$LOG_DIR"/run_vllm_test.log
        "$LOG_DIR"/run_vllm_test_*.log
        "$LOG_DIR"/test.log
        "$LOG_DIR"/test_*.log
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
        "$SCRIPT_DIR"/vllm_serve_log.txt
        "$SCRIPT_DIR"/vllm_serve_log.txt.old
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

    if [ "${#files[@]}" -eq 0 ]; then
        return
    fi

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

    if [ -f "$MP_SERVE_LOG" ]; then
        cp -p "$MP_SERVE_LOG" "$dest_dir/"
    fi
}

archive_run_logs() {
    local root="$1"
    local label="$2"
    local archive_dir

    archive_dir=$(make_unique_dir "$root" "${RUN_START_TS}_${PRESET_TAG}")
    copy_current_logs "$archive_dir"
    log_info "${label}日志已归档到: $archive_dir"
}

cleanup_process_group() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    fi
}

cleanup() {
    local exit_code=$?

    if [ "$LAUNCHER" = "mpi" ]; then
        {
            pkill -TERM -f "vllm serve" 2>&1 || true
            pkill -TERM -f "run_mp_rpc_worker" 2>&1 || true
            pkill -TERM -f "serve_mp_rpc_all_mpi_template.sh" 2>&1 || true
            sleep 5
            pkill -9 -f "vllm serve" 2>&1 || true
            pkill -9 -f "run_mp_rpc_worker" 2>&1 || true
            pkill -9 -f "serve_mp_rpc_all_mpi_template.sh" 2>&1 || true
            pkill -9 -f "mpirun.*serve_mp_rpc_all_mpi_template.sh" 2>&1 || true
        } >> "$MPI_CLEANUP_LOG" 2>&1
    fi

    if [ "$exit_code" -eq 0 ] && [ "$TEST_MODE" != "none" ]; then
        archive_run_logs "$SUCCESS_ROOT" "成功"
    elif [ "$exit_code" -ne 0 ]; then
        archive_run_logs "$FAILED_ROOT" "失败"
    fi

    if [ "${#PIDS[@]}" -gt 0 ]; then
        log_info "清理进程..."
        for pid in "${PIDS[@]}"; do
            cleanup_process_group "$pid"
        done
        sleep 3
    fi

    exit "$exit_code"
}
trap cleanup EXIT INT TERM

collect_error_details() {
    local error_details=""
    local files=()

    if [ -f "$LAUNCH_LOG" ]; then
        files+=("$LAUNCH_LOG")
    fi
    if [ -f "$HEAD_SERVE_LOG" ]; then
        files+=("$HEAD_SERVE_LOG")
    fi
    if [ -f "$MPI_WORKERS_LOG" ]; then
        files+=("$MPI_WORKERS_LOG")
    fi
    if [ -f "$MP_SERVE_LOG" ]; then
        files+=("$MP_SERVE_LOG")
    fi

    if [ "${#files[@]}" -gt 0 ]; then
        error_details=$(grep -HinE "(^|[[:space:]])(ERROR|CRITICAL|FATAL)([[:space:]]|$)|Traceback \(most recent call last\)|RuntimeError:|AssertionError:|ValueError:|KeyError:|TypeError:|ImportError:|ModuleNotFoundError:|Segmentation fault" "${files[@]}" 2>/dev/null | head -n 5 || true)
    fi

    printf '%s' "$error_details"
}

wait_for_service() {
    local max_wait="${VLLM_TEST_MAX_WAIT:-300}"
    local wait_time=0
    local check_interval=5
    local ready_log

    if [ "$LAUNCHER" = "mpi" ]; then
        ready_log="$HEAD_SERVE_LOG"
    else
        ready_log="$MP_SERVE_LOG"
    fi

    log_info "等待服务启动..."
    while [ "$wait_time" -lt "$max_wait" ]; do
        if [ -f "$ready_log" ] && grep -q "Application startup complete" "$ready_log"; then
            echo ""
            log_success "服务启动成功"
            sleep 5
            return 0
        fi

        local error_detail
        error_detail="$(collect_error_details)"
        if [ -n "$error_detail" ]; then
            echo ""
            log_error "检测到异常中断"
            echo "$error_detail" | while read -r line; do
                [ -n "$line" ] && log_error " -> $line"
            done
            return 1
        fi

        echo -n "."
        sleep "$check_interval"
        wait_time=$((wait_time + check_interval))
    done

    echo ""
    log_error "等待超时 (${max_wait} 秒)"
    if [ -f "$ready_log" ]; then
        tail -30 "$ready_log"
    fi
    return 1
}

extract_model_reply() {
    local content=""
    local raw_output="$1"

    if command -v jq >/dev/null 2>&1; then
        content=$(printf '%s' "$raw_output" | jq -r '.choices[0].message.content // .choices[0].text // empty' 2>/dev/null || true)
    fi

    if [ -z "$content" ]; then
        content=$(printf '%s' "$raw_output" | grep -oP '"(content|text)":\s*"\K[^"]*' | head -1 || true)
    fi

    if [ -n "$content" ]; then
        echo -e "${GREEN}${content}${NC}"
    else
        log_warning "未能从响应中提取 content/text 字段"
    fi
}

start_mp_launcher() {
    log_info "启动模式: mp"
    setsid bash "$SCRIPT_DIR/serve/serve_mp_template.sh" "${ENV_ARGS[@]}" > "$LAUNCH_LOG" 2>&1 &
    local pid=$!
    record_pid "$pid"
    log_info "服务 PID: $pid"
    log_info "启动日志: $LAUNCH_LOG"
    log_info "服务日志: $MP_SERVE_LOG"
}

start_mpi_launcher() {
    log_info "启动模式: mpi"
    log_info "MPI 进程数: $MPI_COUNT"

    setsid bash "$SCRIPT_DIR/serve/serve_head_only_template.sh" "${ENV_ARGS[@]}" > "$LAUNCH_LOG" 2>&1 &
    local head_pid=$!
    record_pid "$head_pid"
    log_info "Head PID: $head_pid"
    log_info "Head 启动日志: $LAUNCH_LOG"
    log_info "Head 服务日志: $HEAD_SERVE_LOG"

    sleep 2

    setsid mpirun -np "$MPI_COUNT" bash "$SCRIPT_DIR/serve/serve_mp_rpc_all_mpi_template.sh" "${ENV_ARGS[@]}" >> "$MPI_WORKERS_LOG" 2>&1 &
    local mpi_pid=$!
    record_pid "$mpi_pid"
    log_info "MPI PID: $mpi_pid"
    log_info "MPI 日志: $MPI_WORKERS_LOG"
}

run_test() {
    TEST_EXIT_CODE=0

    if [ "$TEST_MODE" = "test" ]; then
        log_info "运行测试..."
        local test_output
        if test_output=$(bash "$SCRIPT_DIR/serve_test/serve_test_template.sh" "${ENV_ARGS[@]}" 2>&1); then
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
    elif [ "$TEST_MODE" = "bench" ]; then
        log_info "运行 bench..."
        if bash "$SCRIPT_DIR/serve_test/serve_bench_template.sh" "${ENV_ARGS[@]}" > "$BENCH_LOG" 2>&1; then
            log_success "Bench 完成"
        else
            TEST_EXIT_CODE=$?
            log_warning "Bench 退出码: $TEST_EXIT_CODE"
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

backup_old_logs

if [ "$LAUNCHER" = "mpi" ]; then
    start_mpi_launcher
else
    start_mp_launcher
fi

wait_for_service
run_test

echo ""
log_info "日志文件位置:"
if [ "$LAUNCHER" = "mpi" ]; then
    log_info "  Head: $HEAD_SERVE_LOG"
    log_info "  MPI:  $MPI_WORKERS_LOG"
else
    log_info "  Serve: $MP_SERVE_LOG"
fi
if [ "$TEST_MODE" = "test" ]; then
    log_info "  Test:  $TEST_LOG"
elif [ "$TEST_MODE" = "bench" ]; then
    log_info "  Bench: $BENCH_LOG"
fi
log_info "  Launch: $LAUNCH_LOG"
log_info "  Cleanup: $MPI_CLEANUP_LOG"

if [ "$TEST_MODE" = "none" ]; then
    echo ""
    log_info "服务正在运行，按 Ctrl+C 停止"
    wait
fi
