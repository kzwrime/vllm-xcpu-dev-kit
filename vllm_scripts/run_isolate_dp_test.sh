#!/bin/bash
# Launch and test vLLM with API server isolated from headless DP engines.
#
# Current scope: DP-only isolation, TP=1. MP RPC workers are intentionally
# excluded until the MP RPC path supports cross-node isolation.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_SCRIPT_DIR="$SCRIPT_DIR"
COMMON_SH="$SCRIPT_DIR/common.sh"

if [ ! -f "$COMMON_SH" ]; then
    echo "Could not find $COMMON_SH" >&2
    exit 1
fi
source "$COMMON_SH"

LOCAL_VENV_BIN="$(cd "$SCRIPT_DIR/.." && pwd)/.venv/bin"
if [ -d "$LOCAL_VENV_BIN" ]; then
    export PATH="$LOCAL_VENV_BIN:$PATH"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

usage() {
    cat <<'USAGE'
用法:
  ./run_isolate_dp_test.sh -e presets/isolate_dp/Qwen3-0.6B_dp2_tp1_eager.sh

常用选项:
  -e <preset_file>       指定 preset
  --dp-hosts LIST        DP engine 节点列表，逗号或空格分隔
  --api-ip IP            API/head 节点对外 IP；默认从本机 hostname -I 选择
  --dp-ips LIST          DP engine 节点 IP，逗号或空格分隔；默认逐台 ssh 发现
  --remote-dir DIR       远端 vllm_scripts 路径；默认与本机当前脚本路径相同
  --remote-venv-bin DIR  远端 Python venv bin；默认 <remote-dir>/../.venv/bin
  --ssh-opts STRING      追加 SSH 参数
  --no-test              只启动服务，不运行测试
  --multi-test           运行 serve_test/test_multl_stream.py
  --multi-test-max-tokens NUM
                         multi test 每个请求最大输出 token，默认 16
  --keep-running         测试后不自动清理，按 Ctrl+C 停止
  -h, --help             显示帮助

可配置环境变量:
  ISOLATE_DP_DP_HOSTS    默认 DP engine 节点列表
  ISOLATE_DP_API_IP      默认 API/head IP
  ISOLATE_DP_DP_IPS      默认 DP engine IP 列表
  ISOLATE_DP_REMOTE_DIR  默认远端 vllm_scripts 路径
  ISOLATE_DP_REMOTE_VENV_BIN
                       默认远端 Python venv bin；默认 <remote-dir>/../.venv/bin
  ISOLATE_DP_SSH_OPTS    默认 SSH 参数
  ISOLATE_DP_RUN_ID      指定日志 run id
  VLLM_TEST_MAX_WAIT     服务启动最大等待时间，默认 300 秒

说明:
  当前脚本只支持 TP=1。TP>1 需要 MP RPC worker 跨节点支持后再接入。
USAGE
}

split_list() {
    local value="$1"
    value="${value//,/ }"
    # shellcheck disable=SC2206
    SPLIT_LIST_RESULT=($value)
}

select_ip_from_words() {
    local ip
    for ip in "$@"; do
        case "$ip" in
            127.*|169.254.*|172.17.*|"")
                continue
                ;;
            *)
                printf '%s' "$ip"
                return 0
                ;;
        esac
    done
    return 1
}

local_ip() {
    local output
    output="$(hostname -I 2>/dev/null || true)"
    # shellcheck disable=SC2206
    local words=($output)
    select_ip_from_words "${words[@]}"
}

remote_ip() {
    local host="$1"
    local output
    output="$(ssh "${SSH_ARGS[@]}" "$host" "hostname -I" 2>/dev/null || true)"
    # shellcheck disable=SC2206
    local words=($output)
    select_ip_from_words "${words[@]}"
}

load_config() {
    SCRIPT_DIR="$RUNNER_SCRIPT_DIR"
    load_env_file "$SCRIPT_DIR/env.sh"
    if [ -n "$PRESET_FILE" ]; then
        load_preset_file "$PRESET_FILE"
    else
        load_user_config "$SCRIPT_DIR"
    fi
    apply_runtime_overrides
    SCRIPT_DIR="$RUNNER_SCRIPT_DIR"
}

validate_config_for_isolate_dp() {
    if [ "${USER_VLLM_TP_SIZE}" -ne 1 ] || [ "${USER_VLLM_PP_SIZE}" -ne 1 ]; then
        log_error "当前 isolate DP 脚本只支持 TP=1, PP=1；实际 TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
        log_error "TP>1 需要 MP RPC worker 跨节点支持后再接入。"
        exit 1
    fi
}

run_dp_engine_role() {
    load_config
    validate_config_for_isolate_dp

    : "${ROLE_DP_RANK:?missing --dp-rank}"
    : "${ROLE_API_IP:?missing --api-ip}"
    : "${ROLE_DP0_IP:?missing --dp0-ip}"
    : "${ROLE_LOG_FILE:?missing --log-file}"
    : "${ROLE_PID_FILE:?missing --pid-file}"

    export USER_VLLM_DATA_PARALLEL_ADDRESS="$ROLE_DP0_IP"
    export USER_VLLM_DATA_PARALLEL_RPC_IP="$ROLE_API_IP"

    mkdir -p "$(dirname "$ROLE_LOG_FILE")" "$(dirname "$ROLE_PID_FILE")"

    log_info "DP engine rank ${ROLE_DP_RANK}: rpc_ip=${USER_VLLM_DATA_PARALLEL_RPC_IP}, dp0_ip=${USER_VLLM_DATA_PARALLEL_ADDRESS}"

    setsid env VLLM_LOGGING_LEVEL="${USER_VLLM_LOGGING_LEVEL}" \
        vllm serve "${USER_VLLM_MODEL}" \
        --headless \
        --api-server-count 0 \
        --max-model-len "${USER_VLLM_MAX_MODEL_LEN}" \
        --max-num-batched-tokens "${USER_VLLM_MAX_NUM_BATCHED_TOKENS}" \
        -tp="${USER_VLLM_TP_SIZE}" \
        -pp="${USER_VLLM_PP_SIZE}" \
        --distributed-executor-backend mp \
        --data-parallel-size "${USER_VLLM_DATA_PARALLEL_SIZE}" \
        --data-parallel-size-local 1 \
        --data-parallel-start-rank "${ROLE_DP_RANK}" \
        ${USER_VLLM_EAGER_OR_NOT} \
        ${VLLM_OPTIONAL_ARGS} \
        --data-parallel-address "${USER_VLLM_DATA_PARALLEL_ADDRESS}" \
        --data-parallel-rpc-ip "${USER_VLLM_DATA_PARALLEL_RPC_IP}" \
        --data-parallel-rpc-port "${USER_VLLM_DATA_PARALLEL_RPC_PORT}" \
        > "$ROLE_LOG_FILE" 2>&1 &

    local child_pid=$!
    echo "$child_pid" > "$ROLE_PID_FILE"
    wait "$child_pid"
}

PRESET_FILE=""
TEST_MODE="test"
MULTI_TEST_MAX_TOKENS=16
KEEP_RUNNING=0
ROLE="main"
ROLE_DP_RANK=""
ROLE_API_IP=""
ROLE_DP0_IP=""
ROLE_LOG_FILE=""
ROLE_PID_FILE=""
DP_HOSTS_VALUE="${ISOLATE_DP_DP_HOSTS:-wzk-vllm-xcpu-03-c8,wzk-vllm-xcpu-03-c7}"
DP_IPS_VALUE="${ISOLATE_DP_DP_IPS:-}"
API_IP_VALUE="${ISOLATE_DP_API_IP:-}"
REMOTE_DIR="${ISOLATE_DP_REMOTE_DIR:-$SCRIPT_DIR}"
REMOTE_VENV_BIN_VALUE="${ISOLATE_DP_REMOTE_VENV_BIN:-}"
SSH_OPTS_VALUE="${ISOLATE_DP_SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}"
RUN_ID="${ISOLATE_DP_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

while [ $# -gt 0 ]; do
    case "$1" in
        -e)
            PRESET_FILE="$2"
            shift 2
            ;;
        --dp-hosts)
            DP_HOSTS_VALUE="$2"
            shift 2
            ;;
        --dp-hosts=*)
            DP_HOSTS_VALUE="${1#*=}"
            shift
            ;;
        --api-ip)
            API_IP_VALUE="$2"
            ROLE_API_IP="$2"
            shift 2
            ;;
        --api-ip=*)
            API_IP_VALUE="${1#*=}"
            ROLE_API_IP="${1#*=}"
            shift
            ;;
        --dp-ips)
            DP_IPS_VALUE="$2"
            shift 2
            ;;
        --dp-ips=*)
            DP_IPS_VALUE="${1#*=}"
            shift
            ;;
        --remote-dir)
            REMOTE_DIR="$2"
            shift 2
            ;;
        --remote-dir=*)
            REMOTE_DIR="${1#*=}"
            shift
            ;;
        --remote-venv-bin)
            REMOTE_VENV_BIN_VALUE="$2"
            shift 2
            ;;
        --remote-venv-bin=*)
            REMOTE_VENV_BIN_VALUE="${1#*=}"
            shift
            ;;
        --ssh-opts)
            SSH_OPTS_VALUE="$2"
            shift 2
            ;;
        --ssh-opts=*)
            SSH_OPTS_VALUE="${1#*=}"
            shift
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
            MULTI_TEST_MAX_TOKENS="$2"
            shift 2
            ;;
        --multi-test-max-tokens=*)
            MULTI_TEST_MAX_TOKENS="${1#*=}"
            shift
            ;;
        --keep-running)
            KEEP_RUNNING=1
            shift
            ;;
        --role)
            ROLE="$2"
            shift 2
            ;;
        --dp-rank)
            ROLE_DP_RANK="$2"
            shift 2
            ;;
        --dp0-ip)
            ROLE_DP0_IP="$2"
            shift 2
            ;;
        --log-file)
            ROLE_LOG_FILE="$2"
            shift 2
            ;;
        --pid-file)
            ROLE_PID_FILE="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
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

# shellcheck disable=SC2206
SSH_ARGS=($SSH_OPTS_VALUE)
REMOTE_VENV_BIN="${REMOTE_VENV_BIN_VALUE:-$(dirname "$REMOTE_DIR")/.venv/bin}"

if [ "$ROLE" = "dp-engine" ]; then
    run_dp_engine_role
    exit 0
elif [ "$ROLE" != "main" ]; then
    log_error "未知 role: $ROLE"
    exit 1
fi

load_config
validate_config_for_isolate_dp

split_list "$DP_HOSTS_VALUE"
DP_HOSTS=("${SPLIT_LIST_RESULT[@]}")
if [ "${#DP_HOSTS[@]}" -ne "$USER_VLLM_DATA_PARALLEL_SIZE" ]; then
    log_error "DP 节点数 (${#DP_HOSTS[@]}) 必须等于 USER_VLLM_DATA_PARALLEL_SIZE (${USER_VLLM_DATA_PARALLEL_SIZE})"
    log_error "请设置 --dp-hosts 或 ISOLATE_DP_DP_HOSTS。"
    exit 1
fi

if [ -n "$DP_IPS_VALUE" ]; then
    split_list "$DP_IPS_VALUE"
    DP_IPS=("${SPLIT_LIST_RESULT[@]}")
    if [ "${#DP_IPS[@]}" -ne "${#DP_HOSTS[@]}" ]; then
        log_error "DP IP 数 (${#DP_IPS[@]}) 必须等于 DP 节点数 (${#DP_HOSTS[@]})"
        exit 1
    fi
else
    DP_IPS=()
    for host in "${DP_HOSTS[@]}"; do
        ip="$(remote_ip "$host")"
        if [ -z "$ip" ]; then
            log_error "无法发现远端节点 IP: $host；请使用 --dp-ips 或 ISOLATE_DP_DP_IPS 显式指定。"
            exit 1
        fi
        DP_IPS+=("$ip")
    done
fi

API_IP="${API_IP_VALUE:-$(local_ip)}"
if [ -z "$API_IP" ]; then
    log_error "无法发现本机 API/head IP；请使用 --api-ip 或 ISOLATE_DP_API_IP 显式指定。"
    exit 1
fi

export USER_VLLM_DATA_PARALLEL_ADDRESS="${DP_IPS[0]}"
export USER_VLLM_DATA_PARALLEL_RPC_IP="$API_IP"

LOG_ROOT="$SCRIPT_DIR/logs/isolate_dp/$RUN_ID"
PID_DIR="$LOG_ROOT/pids"
mkdir -p "$LOG_ROOT" "$PID_DIR"

API_LOG="$LOG_ROOT/api_server.log"
API_PID_FILE="$PID_DIR/api_server.pid"
TEST_LOG="$LOG_ROOT/test.log"
TEST_ENV_ARGS=()
if [ -n "$PRESET_FILE" ]; then
    TEST_ENV_ARGS=(-e "$PRESET_FILE")
fi

REMOTE_SSH_PIDS=()
REMOTE_PID_FILES=()
REMOTE_HOSTS=()
API_PID=""
CLEANED_UP=0

cleanup() {
    local exit_code=$?
    if [ "$CLEANED_UP" -eq 1 ]; then
        return
    fi
    CLEANED_UP=1

    if [ "$KEEP_RUNNING" -eq 1 ] && [ "$exit_code" -eq 0 ]; then
        log_info "--keep-running 已启用，保留服务进程。日志: $LOG_ROOT"
        return
    fi

    log_info "清理 isolate DP 进程..."
    if [ -n "${API_PID:-}" ] && kill -0 "$API_PID" 2>/dev/null; then
        kill -- -"$API_PID" 2>/dev/null || kill "$API_PID" 2>/dev/null || true
    fi

    local idx host pid_file remote_pid
    for idx in "${!REMOTE_HOSTS[@]}"; do
        host="${REMOTE_HOSTS[$idx]}"
        pid_file="${REMOTE_PID_FILES[$idx]}"
        ssh "${SSH_ARGS[@]}" "$host" "if [ -f $(printf '%q' "$pid_file") ]; then pid=\$(cat $(printf '%q' "$pid_file")); kill -- -\"\$pid\" 2>/dev/null || kill \"\$pid\" 2>/dev/null || true; fi" >/dev/null 2>&1 || true
    done

    for remote_pid in "${REMOTE_SSH_PIDS[@]}"; do
        kill "$remote_pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

log_info "========================================="
log_info "  Isolated DP 启动与测试"
log_info "========================================="
log_info "Preset: ${PRESET_FILE:-user_env}"
log_info "模型: $USER_VLLM_MODEL"
log_info "并行配置: DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
log_info "API/head IP: $API_IP"
log_info "DP hosts: ${DP_HOSTS[*]}"
log_info "DP IPs: ${DP_IPS[*]}"
log_info "DP RPC port: $USER_VLLM_DATA_PARALLEL_RPC_PORT"
log_info "OpenAI API port: $USER_VLLM_PORT"
log_info "日志目录: $LOG_ROOT"

log_info "启动 API/head server..."
setsid env VLLM_LOGGING_LEVEL="${USER_VLLM_LOGGING_LEVEL}" \
    vllm serve "${USER_VLLM_MODEL}" \
    --api-server-count 1 \
    --max-model-len "${USER_VLLM_MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${USER_VLLM_MAX_NUM_BATCHED_TOKENS}" \
    -tp="${USER_VLLM_TP_SIZE}" \
    -pp="${USER_VLLM_PP_SIZE}" \
    --distributed-executor-backend mp \
    --port "${USER_VLLM_PORT}" \
    ${USER_VLLM_EAGER_OR_NOT} \
    ${VLLM_OPTIONAL_ARGS} \
    --data-parallel-size "${USER_VLLM_DATA_PARALLEL_SIZE}" \
    --data-parallel-size-local 0 \
    --data-parallel-address "${USER_VLLM_DATA_PARALLEL_ADDRESS}" \
    --data-parallel-rpc-ip "${USER_VLLM_DATA_PARALLEL_RPC_IP}" \
    --data-parallel-rpc-port "${USER_VLLM_DATA_PARALLEL_RPC_PORT}" \
    > "$API_LOG" 2>&1 &
API_PID=$!
echo "$API_PID" > "$API_PID_FILE"
log_info "API PID: $API_PID"

sleep 2

for rank in "${!DP_HOSTS[@]}"; do
    host="${DP_HOSTS[$rank]}"
    remote_log="$LOG_ROOT/dp_rank${rank}_${host}.log"
    remote_pid_file="$PID_DIR/dp_rank${rank}_${host}.pid"
    remote_script="$REMOTE_DIR/run_isolate_dp_test.sh"
    remote_preset="$PRESET_FILE"
    if [[ "$remote_preset" != /* ]]; then
        remote_preset="$REMOTE_DIR/$remote_preset"
    fi

    remote_cmd="export PATH=$(printf '%q' "$REMOTE_VENV_BIN"):\$PATH; cd $(printf '%q' "$REMOTE_DIR") && bash $(printf '%q' "$remote_script") --role dp-engine --run-id $(printf '%q' "$RUN_ID") -e $(printf '%q' "$remote_preset") --api-ip $(printf '%q' "$API_IP") --dp0-ip $(printf '%q' "${DP_IPS[0]}") --dp-rank $(printf '%q' "$rank") --log-file $(printf '%q' "$remote_log") --pid-file $(printf '%q' "$remote_pid_file")"

    log_info "启动 DP rank ${rank} on ${host} (${DP_IPS[$rank]})"
    ssh "${SSH_ARGS[@]}" "$host" "$remote_cmd" > "$LOG_ROOT/ssh_dp_rank${rank}_${host}.log" 2>&1 &
    REMOTE_SSH_PIDS+=("$!")
    REMOTE_HOSTS+=("$host")
    REMOTE_PID_FILES+=("$remote_pid_file")
done

wait_for_api() {
    local max_wait="${VLLM_TEST_MAX_WAIT:-300}"
    local wait_time=0
    local interval=5
    local response=""

    log_info "等待 API 服务就绪..."
    while [ "$wait_time" -lt "$max_wait" ]; do
        if response="$(curl --silent --fail "http://127.0.0.1:${USER_VLLM_PORT}/v1/models" 2>/dev/null)"; then
            if printf '%s' "$response" | grep -F "\"id\":\"${USER_VLLM_MODEL}\"" >/dev/null 2>&1 || printf '%s' "$response" | grep -F "\"id\": \"${USER_VLLM_MODEL}\"" >/dev/null 2>&1; then
                echo ""
                log_success "API 服务就绪"
                return 0
            fi
        fi

        if ! kill -0 "$API_PID" 2>/dev/null; then
            echo ""
            log_error "API/head server 已退出，tail 日志:"
            tail -80 "$API_LOG" || true
            return 1
        fi

        echo -n "."
        sleep "$interval"
        wait_time=$((wait_time + interval))
    done

    echo ""
    log_error "等待 API 超时 (${max_wait}s)，tail 日志:"
    tail -80 "$API_LOG" || true
    return 1
}

wait_for_api

if [ "$TEST_MODE" = "test" ]; then
    log_info "运行单请求测试..."
    if bash "$SCRIPT_DIR/serve_test/serve_test_template.sh" "${TEST_ENV_ARGS[@]}" > "$TEST_LOG" 2>&1; then
        log_success "单请求测试通过"
    else
        rc=$?
        log_error "单请求测试失败，日志: $TEST_LOG"
        tail -80 "$TEST_LOG" || true
        exit "$rc"
    fi
elif [ "$TEST_MODE" = "multi" ]; then
    log_info "运行 multi test..."
    if python "$SCRIPT_DIR/serve_test/test_multl_stream.py" "${TEST_ENV_ARGS[@]}" --max-tokens "$MULTI_TEST_MAX_TOKENS" > "$TEST_LOG" 2>&1; then
        log_success "Multi test 通过"
    else
        rc=$?
        log_error "Multi test 失败，日志: $TEST_LOG"
        tail -80 "$TEST_LOG" || true
        exit "$rc"
    fi
else
    log_info "跳过测试"
fi

log_info "日志文件:"
log_info "  API:  $API_LOG"
for rank in "${!DP_HOSTS[@]}"; do
    log_info "  DP${rank}: $LOG_ROOT/dp_rank${rank}_${DP_HOSTS[$rank]}.log"
done
if [ "$TEST_MODE" != "none" ]; then
    log_info "  Test: $TEST_LOG"
fi

if [ "$KEEP_RUNNING" -eq 1 ]; then
    log_info "服务正在运行，按 Ctrl+C 停止"
    wait
fi
