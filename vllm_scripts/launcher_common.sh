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

launcher_bool_enabled() {
    [[ "${1:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]
}

launcher_port_is_free() {
    local port="$1"
    ! timeout 1 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1
}

launcher_port_range_is_free() {
    local start="$1"
    local count="$2"
    local port

    for ((port = start; port < start + count; port++)); do
        launcher_port_is_free "$port" || return 1
    done
    return 0
}

launcher_reserved_port_ranges=()
launcher_reserved_port_labels=()

launcher_port_range_is_reserved() {
    local start="$1"
    local count="$2"
    local end=$((start + count - 1))
    local range
    local range_start
    local range_count
    local range_end

    for range in "${launcher_reserved_port_ranges[@]}"; do
        range_start="${range%%:*}"
        range_count="${range##*:}"
        range_end=$((range_start + range_count - 1))
        if [ "$start" -le "$range_end" ] && [ "$range_start" -le "$end" ]; then
            return 0
        fi
    done
    return 1
}

launcher_port_range_is_available() {
    local start="$1"
    local count="$2"

    launcher_port_range_is_free "$start" "$count" || return 1
    ! launcher_port_range_is_reserved "$start" "$count"
}

launcher_find_free_port_range() {
    local start="$1"
    local count="$2"

    while ! launcher_port_range_is_available "$start" "$count"; do
        start=$((start + 1))
    done
    printf '%s' "$start"
}

launcher_reserve_port_range() {
    local label="$1"
    local start="$2"
    local count="$3"

    launcher_reserved_port_labels+=("$label")
    launcher_reserved_port_ranges+=("${start}:${count}")
}

launcher_format_port_range() {
    local start="$1"
    local count="$2"

    if [ "$count" -eq 1 ]; then
        printf '%s' "$start"
    else
        printf '%s-%s' "$start" "$((start + count - 1))"
    fi
}

launcher_prepare_port_assignment() {
    local label="$1"
    local value_var="$2"
    local default_value="$3"
    local count="$4"
    local effective_var="$5"
    local requested_port="${!value_var:-$default_value}"
    local selected_port
    local requested_range

    if [ -z "$requested_port" ]; then
        log_error "${label} 端口未设置"
        return 1
    fi
    if ! [[ "$requested_port" =~ ^[0-9]+$ ]] || [ "$requested_port" -le 0 ]; then
        log_error "${label} 端口非法: ${requested_port}"
        return 1
    fi

    requested_range="$(launcher_format_port_range "$requested_port" "$count")"

    if launcher_bool_enabled "${RUN_VLLM_TEST_AUTO_PORT:-0}"; then
        selected_port="$(launcher_find_free_port_range "$requested_port" "$count")"
        if [ "$selected_port" != "$requested_port" ]; then
            log_warning "${label} 端口 ${requested_range} 不可用，自动改用 $(launcher_format_port_range "$selected_port" "$count")"
        fi
        export "$value_var=$selected_port"
        export "$effective_var=$selected_port"
        launcher_reserve_port_range "$label" "$selected_port" "$count"
        return 0
    fi

    if ! launcher_port_range_is_free "$requested_port" "$count"; then
        log_error "${label} 端口 ${requested_range} 已被占用；默认严格模式下不会复用或自动改端口"
        log_error "请清理占用进程，或显式使用 --auto-port / RUN_VLLM_TEST_AUTO_PORT=1"
        return 1
    fi
    if launcher_port_range_is_reserved "$requested_port" "$count"; then
        log_error "${label} 端口 ${requested_range} 与本次运行的其他端口冲突"
        log_error "请调整配置，或显式使用 --auto-port / RUN_VLLM_TEST_AUTO_PORT=1"
        return 1
    fi

    launcher_reserve_port_range "$label" "$requested_port" "$count"
}

launcher_prepare_non_pd_ports() {
    local ready_port_count

    if [ "${DISAGG_PREFILL:-0}" -eq 1 ]; then
        return 0
    fi

    launcher_reserved_port_ranges=()
    launcher_reserved_port_labels=()

    launcher_prepare_port_assignment "OpenAI API" USER_VLLM_PORT "" 1 RUN_VLLM_EFFECTIVE_PORT || return 1

    if [ "$LAUNCHER" != "mpi" ]; then
        return 0
    fi

    launcher_prepare_port_assignment "DP RPC" USER_VLLM_DATA_PARALLEL_RPC_PORT 13345 1 RUN_VLLM_EFFECTIVE_DATA_PARALLEL_RPC_PORT || return 1

    if [ "${VLLM_USE_MPI_COORD:-0}" = "1" ]; then
        launcher_prepare_port_assignment "MPI coord" VLLM_MPI_COORD_PORT 15555 1 RUN_VLLM_EFFECTIVE_MPI_COORD_PORT || return 1
    fi

    ready_port_count=$((MPI_COUNT + 4))
    launcher_prepare_port_assignment "MP RPC ready" VLLM_MP_RPC_READY_PORT_BASE 28888 "$ready_port_count" RUN_VLLM_EFFECTIVE_MP_RPC_READY_PORT_BASE || return 1
}

launcher_models_contains_model() {
    local models_response="$1"
    local model="$2"

    if command -v jq >/dev/null 2>&1; then
        printf '%s' "$models_response" | jq -e --arg model "$model" 'any(.data[]?; .id == $model)' >/dev/null 2>&1
        return
    fi

    if command -v python3 >/dev/null 2>&1; then
        python3 -c '
import json
import sys

models_response = sys.argv[1]
model = sys.argv[2]
data = json.loads(models_response)
sys.exit(0 if any(item.get("id") == model for item in data.get("data", [])) else 1)
' "$models_response" "$model" >/dev/null 2>&1
        return
    fi

    printf '%s' "$models_response" | grep -F "\"id\":\"${model}\"" >/dev/null 2>&1
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
    # Presets historically use the global SCRIPT_DIR variable while they are
    # being sourced to locate user_env_template.sh.  Restore the launcher's
    # own root afterwards; otherwise paths to serve/*.sh point into presets/.
    local launcher_script_dir="$SCRIPT_DIR"

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

    SCRIPT_DIR="$launcher_script_dir"

    [ -n "$PRESET_TAG" ] || PRESET_TAG="user_env"
    [ -n "$PRESET_NAME" ] || PRESET_NAME="$PRESET_TAG"
    # Tests may run from logs/ (multi-test does), so a relative -e path would
    # no longer resolve against the vllm_scripts directory.  Keep ENV_ARGS
    # unchanged for the launch scripts, but pass the canonical preset path to
    # all test entry points.
    if [ -n "$PRESET_FILE_INPUT" ]; then
        TEST_ENV_ARGS=("-e" "$ORIG_CONFIG_FILE")
    else
        TEST_ENV_ARGS=("${ENV_ARGS[@]}")
    fi
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

    launcher_prepare_non_pd_ports || return 1

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
    local models_response

    log_info "等待 $name 服务启动..."
    while [ "$wait_time" -lt "$max_wait" ]; do
        if models_response=$(curl --silent --fail "http://127.0.0.1:${port}/v1/models" 2>/dev/null); then
            if launcher_models_contains_model "$models_response" "$USER_VLLM_MODEL"; then
                echo ""
                log_success "$name API 服务就绪"
                return 0
            fi
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
    log_error "$name 等待超时 (${max_wait} 秒)"
    if [ -n "$models_response" ]; then
        log_error "端口 ${port} 上的 /v1/models 未包含当前模型: ${USER_VLLM_MODEL}"
        printf '%s\n' "$models_response"
    fi
    [ -f "$log_file" ] && tail -60 "$log_file"
    return 1
}

launcher_collect_error_details() {
    local error_details=""
    local files=()
    local file

    if [ "${DISAGG_PREFILL:-0}" -eq 1 ] && [ -n "${PD_ROOT:-}" ] && [ -d "$PD_ROOT" ]; then
        # Every P/D run has a unique root.  Restrict the scan to that root so
        # stale top-level logs from an earlier non-P/D run cannot abort startup.
        while IFS= read -r file; do
            files+=("$file")
        done < <(find "$PD_ROOT" -type f \( -name '*.log' -o -name 'vllm_*_log*.txt' \) -print 2>/dev/null)
    else
        [ -f "${LAUNCH_LOG:-}" ] && files+=("$LAUNCH_LOG")
        [ -f "${HEAD_SERVE_LOG:-}" ] && files+=("$HEAD_SERVE_LOG")
        [ -f "${MPI_WORKERS_LOG:-}" ] && files+=("$MPI_WORKERS_LOG")
        [ -f "${MP_SERVE_LOG:-}" ] && files+=("$MP_SERVE_LOG")
    fi

    if [ "${#files[@]}" -gt 0 ]; then
        # Log levels and exception names are case-sensitive.  A case-insensitive
        # scan misclassifies benign DEBUG messages containing ordinary prose
        # such as "An error happened while trying to locate the file".
        error_details=$(grep -HnE "(^|[^[:alnum:]_])(ERROR|CRITICAL|FATAL)([^[:alnum:]_]|$)|Traceback \(most recent call last\)|RuntimeError:|AssertionError:|ValueError:|KeyError:|TypeError:|ImportError:|ModuleNotFoundError:|Segmentation fault|Fatal Python error|terminate called after throwing|c10::Error|Exception raised from|ServerDisconnectedError|ConnectionRefusedError|Connection reset by peer|Aborted" "${files[@]}" 2>/dev/null | head -n 8 || true)
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
    launcher_wait_for_api
}

launcher_wait_for_api() {
    if [ "$DISAGG_PREFILL" -eq 1 ]; then
        return 0
    fi

    launcher_wait_for_http_service "vLLM" "$USER_VLLM_PORT" "$LAUNCH_LOG"
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
            log_info "  DP RPC: ${USER_VLLM_DATA_PARALLEL_RPC_PORT}"
            if [ "${VLLM_USE_MPI_COORD:-0}" = "1" ]; then
                log_info "  MPI coord: ${VLLM_MPI_COORD_PORT}"
            fi
            log_info "  MP RPC ready: $(launcher_format_port_range "${VLLM_MP_RPC_READY_PORT_BASE:-28888}" "$((MPI_COUNT + 4))")"
            log_info "  Head: $HEAD_SERVE_LOG"
            log_info "  MPI:  $MPI_WORKERS_LOG"
        else
            log_info "  Serve: $MP_SERVE_LOG"
        fi
    fi
    log_info "  Launch: $LAUNCH_LOG"
}
