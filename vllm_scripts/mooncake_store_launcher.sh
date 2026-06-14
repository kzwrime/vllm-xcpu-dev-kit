#!/bin/bash
# Helper for Mooncake Store standalone CPU/SSD pool experiments.
#
# This does not change USER_VLLM_PD_KV_BACKEND=mooncake. That backend still
# uses MooncakeConnector for direct P/D transfer. Use this helper when testing
# MooncakeStoreConnector after it is available in the local vLLM tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MOONCAKE_BUILD_DIR="${MOONCAKE_BUILD_DIR:-$REPO_ROOT/deps/Mooncake/build}"
MOONCAKE_MASTER_BIN="${MOONCAKE_MASTER_BIN:-$MOONCAKE_BUILD_DIR/mooncake-store/src/mooncake_master}"
MOONCAKE_CLIENT_BIN="${MOONCAKE_CLIENT_BIN:-$MOONCAKE_BUILD_DIR/mooncake-store/src/mooncake_client}"

MOONCAKE_STORE_ROOT="${MOONCAKE_STORE_ROOT:-$SCRIPT_DIR/logs/mooncake_store}"
MOONCAKE_STORE_CONFIG="${MOONCAKE_STORE_CONFIG:-$MOONCAKE_STORE_ROOT/mooncake_config.json}"
MOONCAKE_STORE_SSD_PATH="${MOONCAKE_STORE_SSD_PATH:-$MOONCAKE_STORE_ROOT/ssd}"

MOONCAKE_MASTER_HOST="${MOONCAKE_MASTER_HOST:-127.0.0.1}"
MOONCAKE_MASTER_BIND="${MOONCAKE_MASTER_BIND:-0.0.0.0}"
MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-50051}"
MOONCAKE_CLIENT_HOST="${MOONCAKE_CLIENT_HOST:-0.0.0.0}"
MOONCAKE_CLIENT_PORT="${MOONCAKE_CLIENT_PORT:-50053}"
MOONCAKE_METADATA_SERVER="${MOONCAKE_METADATA_SERVER:-P2PHANDSHAKE}"
MOONCAKE_STORE_PROTOCOL="${MOONCAKE_STORE_PROTOCOL:-tcp}"
MOONCAKE_STORE_DEVICE_NAME="${MOONCAKE_STORE_DEVICE_NAME:-}"
MOONCAKE_OWNER_GLOBAL_SEGMENT_SIZE="${MOONCAKE_OWNER_GLOBAL_SEGMENT_SIZE:-64 GB}"
MOONCAKE_REQUESTER_LOCAL_BUFFER_SIZE="${MOONCAKE_REQUESTER_LOCAL_BUFFER_SIZE:-4GB}"
MOONCAKE_STORE_ENABLE_OFFLOAD="${MOONCAKE_STORE_ENABLE_OFFLOAD:-false}"
MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES="${MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES:-0}"
MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR="${MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR:-bucket_storage_backend}"

usage() {
    cat <<'USAGE'
用法:
  ./mooncake_store_launcher.sh config
  ./mooncake_store_launcher.sh master
  ./mooncake_store_launcher.sh client
  ./mooncake_store_launcher.sh env

子命令:
  config   生成 vLLM MooncakeStoreConnector standalone-store JSON
  master   前台启动 mooncake_master
  client   前台启动 external mooncake_client owner
  env      打印 vLLM requester 侧需要 export 的环境变量

常用环境变量:
  MOONCAKE_STORE_ENABLE_OFFLOAD=true|false
  MOONCAKE_STORE_ROOT=vllm_scripts/logs/mooncake_store
  MOONCAKE_STORE_SSD_PATH=<absolute-or-relative-ssd-dir>
  MOONCAKE_OWNER_GLOBAL_SEGMENT_SIZE="64 GB"
  MOONCAKE_MASTER_PORT=50051
  MOONCAKE_CLIENT_PORT=50053
  MOONCAKE_STORE_PROTOCOL=tcp|rdma
USAGE
}

bool_flag() {
    case "$1" in
        1|true|TRUE|yes|YES|on|ON) printf 'true' ;;
        *) printf 'false' ;;
    esac
}

ensure_bins() {
    if [ ! -x "$MOONCAKE_MASTER_BIN" ]; then
        echo "Cannot execute mooncake_master: $MOONCAKE_MASTER_BIN" >&2
        exit 1
    fi
    if [ ! -x "$MOONCAKE_CLIENT_BIN" ]; then
        echo "Cannot execute mooncake_client: $MOONCAKE_CLIENT_BIN" >&2
        exit 1
    fi
}

write_config() {
    local enable_offload
    enable_offload="$(bool_flag "$MOONCAKE_STORE_ENABLE_OFFLOAD")"
    mkdir -p "$(dirname "$MOONCAKE_STORE_CONFIG")"
    cat > "$MOONCAKE_STORE_CONFIG" <<EOF
{
  "mode": "standalone-store",
  "metadata_server": "$MOONCAKE_METADATA_SERVER",
  "master_server_address": "$MOONCAKE_MASTER_HOST:$MOONCAKE_MASTER_PORT",
  "global_segment_size": 0,
  "local_buffer_size": "$MOONCAKE_REQUESTER_LOCAL_BUFFER_SIZE",
  "protocol": "$MOONCAKE_STORE_PROTOCOL",
  "device_name": "$MOONCAKE_STORE_DEVICE_NAME",
  "enable_offload": $enable_offload
}
EOF
    echo "$MOONCAKE_STORE_CONFIG"
}

start_master() {
    ensure_bins
    local enable_offload
    enable_offload="$(bool_flag "$MOONCAKE_STORE_ENABLE_OFFLOAD")"
    exec "$MOONCAKE_MASTER_BIN" \
        --rpc_address="$MOONCAKE_MASTER_BIND" \
        --rpc_port="$MOONCAKE_MASTER_PORT" \
        --enable_offload="$enable_offload"
}

start_client() {
    ensure_bins
    local enable_offload
    enable_offload="$(bool_flag "$MOONCAKE_STORE_ENABLE_OFFLOAD")"
    mkdir -p "$MOONCAKE_STORE_SSD_PATH"
    export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="$MOONCAKE_STORE_SSD_PATH"
    export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR
    if [ "$MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES" != "0" ]; then
        export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES
    fi
    exec "$MOONCAKE_CLIENT_BIN" \
        --host="$MOONCAKE_CLIENT_HOST" \
        --port="$MOONCAKE_CLIENT_PORT" \
        --master_server_address="$MOONCAKE_MASTER_HOST:$MOONCAKE_MASTER_PORT" \
        --metadata_server="$MOONCAKE_METADATA_SERVER" \
        --global_segment_size="$MOONCAKE_OWNER_GLOBAL_SEGMENT_SIZE" \
        --protocol="$MOONCAKE_STORE_PROTOCOL" \
        --device_names="$MOONCAKE_STORE_DEVICE_NAME" \
        --enable_offload="$enable_offload"
}

print_env() {
    write_config >/dev/null
    cat <<EOF
export MOONCAKE_CONFIG_PATH="$MOONCAKE_STORE_CONFIG"
export MOONCAKE_PREFERRED_SEGMENT="$MOONCAKE_MASTER_HOST:$MOONCAKE_CLIENT_PORT"
EOF
}

cmd="${1:-help}"
case "$cmd" in
    config)
        write_config
        ;;
    master)
        start_master
        ;;
    client)
        start_client
        ;;
    env)
        print_env
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: $cmd" >&2
        usage >&2
        exit 1
        ;;
esac
