#!/bin/bash
# Shared vLLM launch helpers for startup-only and startup+test wrappers.

COMMON_SH="$SCRIPT_DIR/common.sh"
PD_LAUNCHER_SH="$SCRIPT_DIR/pd_launcher.sh"

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

sanitize_name() {
    printf '%s' "$1" | sed -E 's#^\./##; s#^presets/##; s#\.sh$##; s#[^A-Za-z0-9._-]+#_#g; s#_+#_#g; s#^_##; s#_$##'
}

preset_basename() {
    local value="$1"
    value="${value%/}"
    value="${value##*/}"
    value="${value%.sh}"
    sanitize_name "$value"
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

launcher_common_init() {
    if [ ! -f "$COMMON_SH" ]; then
        log_error "Could not find $COMMON_SH"
        exit 1
    fi
    source "$COMMON_SH"

    if [ ! -f "$PD_LAUNCHER_SH" ]; then
        log_error "Could not find $PD_LAUNCHER_SH"
        exit 1
    fi
}

launcher_validate_launcher() {
    case "$LAUNCHER" in
        auto|mp|mpi)
            ;;
        *)
            log_error "无效的 --launcher 值: $LAUNCHER"
            exit 1
            ;;
    esac
}

launcher_load_config() {
    load_env_file "$SCRIPT_DIR/env.sh"
    if [ -n "$PRESET_FILE_INPUT" ]; then
        load_preset_file "$PRESET_FILE_INPUT"
        CONFIG_SOURCE="$PRESET_FILE_INPUT"
        PRESET_TAG="$(sanitize_name "$PRESET_FILE_INPUT")"
        PRESET_NAME="$(preset_basename "$PRESET_FILE_INPUT")"
        if [[ "$PRESET_FILE_INPUT" = /* ]]; then
            ORIG_CONFIG_FILE="$PRESET_FILE_INPUT"
        else
            ORIG_CONFIG_FILE="$(realpath "$PRESET_FILE_INPUT")"
        fi
    elif [ -n "${PRESET:-}" ]; then
        load_user_config "$SCRIPT_DIR"
        CONFIG_SOURCE="PRESET=${PRESET}"
        PRESET_TAG="$(sanitize_name "$PRESET")"
        PRESET_NAME="$(preset_basename "$PRESET")"
        ORIG_CONFIG_FILE="$SCRIPT_DIR/presets/${PRESET}.sh"
    else
        load_user_config "$SCRIPT_DIR"
        CONFIG_SOURCE="user_env.sh / user_env_template.sh"
        PRESET_TAG="user_env"
        PRESET_NAME="user_env"
        if [ -f "$SCRIPT_DIR/user_env.sh" ]; then
            ORIG_CONFIG_FILE="$SCRIPT_DIR/user_env.sh"
        else
            ORIG_CONFIG_FILE="$SCRIPT_DIR/user_env_template.sh"
        fi
    fi

    [ -n "$PRESET_TAG" ] || PRESET_TAG="user_env"
    [ -n "$PRESET_NAME" ] || PRESET_NAME="$PRESET_TAG"
    TEST_ENV_ARGS=("${ENV_ARGS[@]}")
}

launcher_check_required_env() {
    check_and_print_env "USER_VLLM_MODEL"
    check_and_print_env "USER_VLLM_PORT"
    check_and_print_env "USER_VLLM_DATA_PARALLEL_SIZE"
    check_and_print_env "USER_VLLM_TP_SIZE"
    check_and_print_env "USER_VLLM_PP_SIZE"
}

launcher_prepare_runtime() {
    local launch_log_name="$1"

    mkdir -p "$LOG_DIR"
    cd "$SCRIPT_DIR"

    MPI_CLEANUP_LOG="$LOG_DIR/mpi_cleanup.log"
    MPI_WORKERS_LOG="$LOG_DIR/mpi_workers.log"
    LAUNCH_LOG="$LOG_DIR/$launch_log_name"
    MP_SERVE_LOG="$LOG_DIR/vllm_serve_log.txt"
    HEAD_SERVE_LOG="$LOG_DIR/vllm_head_log.txt"

    USER_VLLM_MPC_SIZE="${USER_VLLM_MPC_SIZE:-$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))}"
    MPI_COUNT=$((USER_VLLM_DATA_PARALLEL_SIZE * USER_VLLM_MPC_SIZE))

    if [[ " ${VLLM_OPTIONAL_ARGS:-} " == *" --language-model-only"* ]] && [[ "${VLLM_XCPU_DISABLE_TORCHVISION:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        export PYTHONPATH="$SCRIPT_DIR/python_patches${PYTHONPATH:+:$PYTHONPATH}"
        log_info "已为 language-model-only 启用 torchvision 运行时屏蔽"
    fi

    if [ "$LAUNCHER" = "auto" ]; then
        if [ "${VLLM_CPU_USE_MPI:-0}" = "1" ] || [ "${VLLM_USE_MPI_COORD:-0}" = "1" ] || [ -n "${USER_VLLM_MP_RPC_WORKER_PER_NODE:-}" ]; then
            LAUNCHER="mpi"
        else
            LAUNCHER="mp"
        fi
    fi

    PIDS=()
    LAUNCHER_ERROR_MONITOR_PID=""
    source "$PD_LAUNCHER_SH"
}

record_pid() {
    PIDS+=("$1")
}

launcher_cleanup_process_group() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    fi
}

launcher_cleanup_processes() {
    launcher_stop_error_monitor

    if [ "$LAUNCHER" = "mpi" ] && [ -n "${MPI_CLEANUP_LOG:-}" ]; then
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

    if [ "${#PIDS[@]}" -gt 0 ]; then
        log_info "清理进程..."
        for pid in "${PIDS[@]}"; do
            launcher_cleanup_process_group "$pid"
        done
        sleep 3
    fi
}

launcher_start_error_monitor() {
    local target_pid="${1:-$$}"
    local interval="${VLLM_TEST_ERROR_MONITOR_INTERVAL:-10}"

    if [ -n "${LAUNCHER_ERROR_MONITOR_PID:-}" ] && kill -0 "$LAUNCHER_ERROR_MONITOR_PID" 2>/dev/null; then
        return 0
    fi

    (
        while true; do
            sleep "$interval"
            if [ -n "$(launcher_collect_error_details)" ]; then
                echo ""
                launcher_print_error_details
                log_error "运行期检测到错误，终止测试流程"
                for pid in "${PIDS[@]}"; do
                    launcher_cleanup_process_group "$pid"
                done
                kill -TERM "$target_pid" 2>/dev/null || true
                exit 0
            fi
        done
    ) &
    LAUNCHER_ERROR_MONITOR_PID=$!
    log_info "错误监控已启动: pid=$LAUNCHER_ERROR_MONITOR_PID, interval=${interval}s"
}

launcher_stop_error_monitor() {
    if [ -n "${LAUNCHER_ERROR_MONITOR_PID:-}" ] && kill -0 "$LAUNCHER_ERROR_MONITOR_PID" 2>/dev/null; then
        kill "$LAUNCHER_ERROR_MONITOR_PID" 2>/dev/null || true
        wait "$LAUNCHER_ERROR_MONITOR_PID" 2>/dev/null || true
    fi
    LAUNCHER_ERROR_MONITOR_PID=""
}

launcher_wait_for_http_service() {
    local name="$1"
    local port="$2"
    local log_file="$3"
    local max_wait="${VLLM_TEST_MAX_WAIT:-300}"
    local wait_time=0
    local check_interval=5

    log_info "等待 $name 服务启动..."
    while [ "$wait_time" -lt "$max_wait" ]; do
        if curl --silent --fail "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
            echo ""
            log_success "$name 服务启动成功"
            return 0
        fi

        echo -n "."
        sleep "$check_interval"
        wait_time=$((wait_time + check_interval))
    done

    echo ""
    log_error "$name 等待超时 (${max_wait} 秒)"
    [ -f "$log_file" ] && tail -60 "$log_file"
    return 1
}

launcher_collect_error_details() {
    local error_details=""
    local files=()
    local file

    [ -f "${LAUNCH_LOG:-}" ] && files+=("$LAUNCH_LOG")
    [ -f "${HEAD_SERVE_LOG:-}" ] && files+=("$HEAD_SERVE_LOG")
    [ -f "${MPI_WORKERS_LOG:-}" ] && files+=("$MPI_WORKERS_LOG")
    [ -f "${MP_SERVE_LOG:-}" ] && files+=("$MP_SERVE_LOG")
    if [ -n "${PD_ROOT:-}" ] && [ -d "$PD_ROOT" ]; then
        while IFS= read -r file; do
            files+=("$file")
        done < <(find "$PD_ROOT" -type f \( -name '*.log' -o -name 'vllm_*_log*.txt' \) -print 2>/dev/null)
    fi

    if [ "${#files[@]}" -gt 0 ]; then
        error_details=$(grep -HinE "(^|[^[:alnum:]_])(ERROR|CRITICAL|FATAL)([^[:alnum:]_]|$)|Traceback \(most recent call last\)|RuntimeError:|AssertionError:|ValueError:|KeyError:|TypeError:|ImportError:|ModuleNotFoundError:|Segmentation fault|Fatal Python error|terminate called after throwing|c10::Error|Exception raised from|ServerDisconnectedError|ConnectionRefusedError|Connection reset by peer|Aborted" "${files[@]}" 2>/dev/null | head -n 8 || true)
    fi

    printf '%s' "$error_details"
}

launcher_print_error_details() {
    local error_detail
    error_detail="$(launcher_collect_error_details)"
    [ -n "$error_detail" ] || return 1

    log_error "检测到异常中断"
    echo "$error_detail" | while read -r line; do
        [ -n "$line" ] && log_error " -> $line"
    done
    return 0
}

launcher_wait_for_log_startup() {
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

        if [ -n "$(launcher_collect_error_details)" ]; then
            echo ""
            launcher_print_error_details
            return 1
        fi

        echo -n "."
        sleep "$check_interval"
        wait_time=$((wait_time + check_interval))
    done

    echo ""
    log_error "等待超时 (${max_wait} 秒)"
    log_error "提示: 可通过设置 VLLM_TEST_MAX_WAIT 环境变量调整超时时间（当前: ${max_wait} 秒）"
    [ -f "$ready_log" ] && tail -30 "$ready_log"
    return 1
}

launcher_start_mp() {
    log_info "启动模式: mp"
    setsid bash "$SCRIPT_DIR/serve/serve_mp_template.sh" "${ENV_ARGS[@]}" > "$LAUNCH_LOG" 2>&1 &
    local pid=$!
    record_pid "$pid"
    log_info "服务 PID: $pid"
    log_info "启动日志: $LAUNCH_LOG"
    log_info "服务日志: $MP_SERVE_LOG"
}

launcher_start_mpi() {
    log_info "启动模式: mpi"
    log_info "MPI 进程数: $MPI_COUNT"

    setsid bash "$SCRIPT_DIR/serve/serve_head_only_template.sh" "${ENV_ARGS[@]}" > "$LAUNCH_LOG" 2>&1 &
    local head_pid=$!
    record_pid "$head_pid"
    log_info "Head PID: $head_pid"
    log_info "Head 启动日志: $LAUNCH_LOG"
    log_info "Head 服务日志: $HEAD_SERVE_LOG"

    sleep 2

    local mpi_run_args_string="${VLLM_MPI_RUN_ARGS:---bind-to none --map-by slot}"
    local mpi_run_args=()
    # shellcheck disable=SC2206
    mpi_run_args=($mpi_run_args_string)
    log_info "MPI 额外参数: ${mpi_run_args[*]}"

    setsid mpirun "${mpi_run_args[@]}" -np "$MPI_COUNT" bash "$SCRIPT_DIR/serve/serve_mp_rpc_all_mpi_template.sh" "${ENV_ARGS[@]}" >> "$MPI_WORKERS_LOG" 2>&1 &
    local mpi_pid=$!
    record_pid "$mpi_pid"
    log_info "MPI PID: $mpi_pid"
    log_info "MPI 日志: $MPI_WORKERS_LOG"
}

launcher_start_service() {
    if [ "$DISAGG_PREFILL" -eq 1 ]; then
        pd_start
    elif [ "$LAUNCHER" = "mpi" ]; then
        launcher_start_mpi
    else
        launcher_start_mp
    fi
}

launcher_wait_for_service() {
    if [ "$DISAGG_PREFILL" -eq 1 ]; then
        return 0
    fi
    launcher_wait_for_log_startup
}

launcher_wait_for_api() {
    if [ "$DISAGG_PREFILL" -eq 1 ]; then
        return 0
    fi

    if [ "$LAUNCHER" = "mpi" ]; then
        launcher_wait_for_http_service "vLLM" "$USER_VLLM_PORT" "$HEAD_SERVE_LOG"
    else
        launcher_wait_for_http_service "vLLM" "$USER_VLLM_PORT" "$MP_SERVE_LOG"
    fi
}

launcher_print_service_locations() {
    if [ "$DISAGG_PREFILL" -eq 1 ]; then
        log_info "  OpenAI API: http://127.0.0.1:${PD_PROXY_PORT}"
        log_info "  手动测试 preset: $PD_ROOT/proxy.sh"
        pd_print_log_locations
    else
        log_info "  OpenAI API: http://127.0.0.1:${USER_VLLM_PORT}"
        log_info "  手动测试 preset: $ORIG_CONFIG_FILE"
        if [ "$LAUNCHER" = "mpi" ]; then
            log_info "  Head: $HEAD_SERVE_LOG"
            log_info "  MPI:  $MPI_WORKERS_LOG"
        else
            log_info "  Serve: $MP_SERVE_LOG"
        fi
    fi
    log_info "  Launch: $LAUNCH_LOG"
}
