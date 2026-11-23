#!/bin/bash
# P/D disaggregated launcher helpers shared by startup and test wrappers.
#
# Uses vllm-router (--vllm-pd-disaggregation --kv-connector mooncake) to
# coordinate N prefill instances x M decode instances with MooncakeConnector
# (P2P) or MultiConnector(MooncakeConnector + MooncakeStoreConnector).
# The default topology is 1P x 1D.

PD_ROOT=""
PD_PREFILL_DIR=""
PD_DECODE_DIR=""
PD_ROUTER_LOG=""
PD_ROUTER_PORT=""
PD_PROXY_LOG=""
PD_PROXY_PORT=""
PD_PREFILL_URLS=""
PD_DECODE_URLS=""
PD_PREFILL_BOOTSTRAP_PORTS=""
PD_PREFILL_URL_LIST=()
PD_PREFILL_BOOTSTRAP_PORT_LIST=()
PD_DECODE_URL_LIST=()
PD_PREFILL_NAMES=()
PD_DECODE_NAMES=()

pd_port_is_free() {
    local port="$1"
    ! timeout 1 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1
}

pd_find_free_port() {
    local port="$1"
    while ! pd_port_is_free "$port"; do
        port=$((port + 1))
    done
    printf '%s' "$port"
}

pd_port_range_is_free() {
    local start="$1"
    local count="$2"
    local port
    for ((port = start; port < start + count; port++)); do
        pd_port_is_free "$port" || return 1
    done
    return 0
}

pd_find_free_port_range() {
    local start="$1"
    local count="$2"
    while ! pd_port_range_is_free "$start" "$count"; do
        start=$((start + count + 1))
    done
    printf '%s' "$start"
}

pd_join_csv() {
    local IFS=,
    printf '%s' "$*"
}

pd_role_name() {
    local role="$1"
    local index="$2"
    printf '%s%d' "$role" "$index"
}

pd_http_log_for_role() {
    local role_dir="$1"
    if [ "$LAUNCHER" = "mp" ]; then
        printf '%s/logs/vllm_serve_log.txt' "$role_dir"
    else
        printf '%s/logs/vllm_head_log.txt' "$role_dir"
    fi
}

pd_resolve_router_bin() {
    local candidate="${VLLM_ROUTER_BIN:-vllm-router}"
    if command -v "$candidate" >/dev/null 2>&1; then
        command -v "$candidate"
        return 0
    fi
    local py_bin=""
    if command -v python3 >/dev/null 2>&1; then
        py_bin="$(dirname "$(command -v python3)")/vllm-router"
    elif command -v python >/dev/null 2>&1; then
        py_bin="$(dirname "$(command -v python)")/vllm-router"
    fi
    if [ -n "$py_bin" ] && [ -x "$py_bin" ]; then
        printf '%s' "$py_bin"
        return 0
    fi
    if [ -x "$SCRIPT_DIR/../.venv/bin/vllm-router" ]; then
        printf '%s' "$SCRIPT_DIR/../.venv/bin/vllm-router"
        return 0
    fi
    return 1
}

pd_write_role_preset() {
    local role="$1"
    local index="$2"
    local port="$3"
    local rpc_port="$4"
    local coord_port="$5"
    local ready_port_base="$6"
    local role_dir="$7"
    local mooncake_config_path="$8"
    local bootstrap_port="$9"
    local output_file="${10}"
    local pd_kv_backend="${USER_VLLM_PD_KV_BACKEND:-mooncake}"
    local mooncake_protocol="${USER_VLLM_PD_MOONCAKE_PROTOCOL:-tcp}"
    local mooncake_num_workers="${USER_VLLM_PD_MOONCAKE_NUM_WORKERS:-4}"
    local device_json=""
    if [ -n "${USER_VLLM_PD_MOONCAKE_DEVICE_NAME:-}" ]; then
        device_json=",\"device_name\":\"${USER_VLLM_PD_MOONCAKE_DEVICE_NAME}\""
    fi

    # P 和 D 可以使用不同的 preset（例如 P 开启 prefill_context_kv_only）。
    local role_config_file="$ORIG_CONFIG_FILE"
    if [ "$role" = "prefill" ] && [ -n "${USER_VLLM_PD_PREFILL_PRESET:-}" ]; then
        role_config_file="$(realpath "$USER_VLLM_PD_PREFILL_PRESET")"
    elif [ "$role" = "decode" ] && [ -n "${USER_VLLM_PD_DECODE_PRESET:-}" ]; then
        role_config_file="$(realpath "$USER_VLLM_PD_DECODE_PRESET")"
    fi

    cat > "$output_file" <<EOF
#!/bin/bash
source "$role_config_file"

export USER_VLLM_PORT=$port
export USER_VLLM_DATA_PARALLEL_RPC_PORT=$rpc_port
export VLLM_MPI_COORD_PORT=$coord_port
export VLLM_MPI_ENV_EXPORT_FILE="$role_dir/vllm_mpi_env_server.sh"
export VLLM_MP_RPC_READY_PORT_BASE=$ready_port_base
export VLLM_RUN_LOG_DIR="$role_dir/logs"
EOF

    if [ "$pd_kv_backend" = "mooncake" ]; then
        local kv_role="kv_consumer"
        if [ "$role" = "prefill" ]; then
            kv_role="kv_producer"
        fi
        cat >> "$output_file" <<EOF
export MOONCAKE_PROTOCOL="${mooncake_protocol}"
_VLLM_PD_KV_TRANSFER_CONFIG='{"kv_connector":"MooncakeConnector","kv_role":"$kv_role","kv_connector_extra_config":{"mooncake_protocol":"${mooncake_protocol}","num_workers":${mooncake_num_workers}${device_json}}}'
export VLLM_OPTIONAL_ARGS="\${VLLM_OPTIONAL_ARGS} --kv-transfer-config \${_VLLM_PD_KV_TRANSFER_CONFIG} --no-disable-hybrid-kv-cache-manager"
EOF
        if [ "$role" = "prefill" ]; then
            cat >> "$output_file" <<EOF
export VLLM_MOONCAKE_BOOTSTRAP_PORT=$bootstrap_port
EOF
        fi
    elif [ "$pd_kv_backend" = "mooncake_multi" ]; then
        local python_hash_seed="${PYTHONHASHSEED:-0}"
        local store_lookup_async="${USER_VLLM_PD_STORE_LOOKUP_ASYNC:-true}"
        local save_decode_cache="${USER_VLLM_PD_SAVE_DECODE_CACHE:-true}"
        local kv_role="kv_consumer"
        local multi_json=""
        if [ "$role" = "prefill" ]; then
            kv_role="kv_producer"
            multi_json='{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"'"${mooncake_protocol}"'","num_workers":'"${mooncake_num_workers}${device_json}"'}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true,"lookup_async":'"${store_lookup_async}"'}}]}}'
        else
            multi_json='{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"'"${mooncake_protocol}"'","num_workers":'"${mooncake_num_workers}${device_json}"'}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"load_async":true,"save_decode_cache":'"${save_decode_cache}"'}}]}}'
        fi
        cat >> "$output_file" <<EOF
export MOONCAKE_PROTOCOL="${mooncake_protocol}"
export MOONCAKE_CONFIG_PATH="${mooncake_config_path}"
export PYTHONHASHSEED="${python_hash_seed}"
_VLLM_PD_KV_TRANSFER_CONFIG='${multi_json}'
export VLLM_OPTIONAL_ARGS="\${VLLM_OPTIONAL_ARGS} --kv-transfer-config \${_VLLM_PD_KV_TRANSFER_CONFIG} --no-disable-hybrid-kv-cache-manager"
EOF
        if [ -n "${MOONCAKE_PREFERRED_SEGMENT:-}" ]; then
            cat >> "$output_file" <<EOF
export MOONCAKE_PREFERRED_SEGMENT="${MOONCAKE_PREFERRED_SEGMENT}"
EOF
        fi
        if [ "$role" = "prefill" ]; then
            cat >> "$output_file" <<EOF
export VLLM_MOONCAKE_BOOTSTRAP_PORT=$bootstrap_port
EOF
        fi
    else
        log_error "Unsupported USER_VLLM_PD_KV_BACKEND: $pd_kv_backend"
        return 1
    fi
}

pd_write_router_preset() {
    local router_port="$1"
    local output_file="$2"

    cat > "$output_file" <<EOF
#!/bin/bash
source "$ORIG_CONFIG_FILE"
export USER_VLLM_PORT=$router_port
EOF
}

pd_start_process_in_dir() {
    local work_dir="$1"
    local pid_file="$2"
    shift 2

    (
        cd "$work_dir"
        setsid "$@" &
        echo $! > "$pid_file"
    )
    record_pid "$(cat "$pid_file")"
}

pd_start_role_launcher() {
    local role_name="$1"
    local role_dir="$2"
    local preset_file="$3"
    local launch_log="$role_dir/launch.log"

    mkdir -p "$role_dir/logs"

    if [ "$LAUNCHER" = "mpi" ]; then
        log_info "P/D $role_name 启动模式: mpi"
        log_info "P/D $role_name MPI 进程数: $MPI_COUNT"

        pd_start_process_in_dir "$role_dir" "$role_dir/head.pid" \
            bash "$SCRIPT_DIR/serve/serve_head_only_template.sh" -e "$preset_file" > "$launch_log" 2>&1

        sleep 2

        local mpi_run_args_string="${VLLM_MPI_RUN_ARGS:---bind-to none --map-by slot}"
        local mpi_run_args=()
        # shellcheck disable=SC2206
        mpi_run_args=($mpi_run_args_string)
        log_info "P/D $role_name MPI 额外参数: ${mpi_run_args[*]}"

        pd_start_process_in_dir "$role_dir" "$role_dir/mpi.pid" \
            mpirun "${mpi_run_args[@]}" -np "$MPI_COUNT" \
            bash "$SCRIPT_DIR/serve/serve_mp_rpc_all_mpi_template.sh" -e "$preset_file" >> "$role_dir/logs/mpi_workers.log" 2>&1
    else
        log_info "P/D $role_name 启动模式: mp"
        pd_start_process_in_dir "$role_dir" "$role_dir/mp.pid" \
            bash "$SCRIPT_DIR/serve/serve_mp_template.sh" -e "$preset_file" > "$launch_log" 2>&1
    fi

    log_info "P/D $role_name 启动日志: $launch_log"
}

pd_wait_for_http_service() {
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
    [ -f "$log_file" ] && tail -60 "$log_file"
    return 1
}

pd_role_port() {
    local env_name="$1"
    local base="$2"
    local stride="$3"
    local index="$4"
    local explicit="${!env_name:-}"

    if [ "$index" -eq 0 ] && [ -n "$explicit" ]; then
        pd_find_free_port "$explicit"
    else
        pd_find_free_port "$((base + index * stride))"
    fi
}

pd_role_ready_base() {
    local env_name="$1"
    local base="$2"
    local stride="$3"
    local index="$4"
    local explicit="${!env_name:-}"

    if [ "$index" -eq 0 ] && [ -n "$explicit" ]; then
        pd_find_free_port_range "$explicit" "$((MPI_COUNT + 4))"
    else
        pd_find_free_port_range "$((base + index * stride))" "$((MPI_COUNT + 4))"
    fi
}

pd_role_bootstrap_port() {
    local base="$1"
    local index="$2"
    local explicit="${USER_VLLM_PD_PREFILL_BOOTSTRAP_PORT:-}"

    if [ "$index" -eq 0 ] && [ -n "$explicit" ]; then
        pd_find_free_port "$explicit"
    else
        pd_find_free_port "$((base + index * 100))"
    fi
}

pd_create_role() {
    local role="$1"
    local index="$2"
    local http_base="$3"
    local rpc_base="$4"
    local coord_base="$5"
    local ready_base="$6"
    local pd_root="$7"
    local mooncake_config_path="$8"
    local role_name
    local role_dir
    local preset_file
    local port
    local rpc_port
    local coord_port
    local ready_port_base
    local bootstrap_port=""
    local http_log

    role_name="$(pd_role_name "$role" "$index")"
    role_dir="$pd_root/$role_name"
    preset_file="$pd_root/$role_name.sh"
    mkdir -p "$role_dir/logs"

    if [ "$role" = "prefill" ]; then
        port="$(pd_role_port USER_VLLM_PD_PREFILL_PORT "$http_base" 100 "$index")"
        rpc_port="$(pd_role_port USER_VLLM_PD_PREFILL_RPC_PORT "$rpc_base" 100 "$index")"
        coord_port="$(pd_role_port USER_VLLM_PD_PREFILL_COORD_PORT "$coord_base" 100 "$index")"
        ready_port_base="$(pd_role_ready_base USER_VLLM_PD_PREFILL_READY_PORT_BASE "$ready_base" 100 "$index")"
        bootstrap_port="$(pd_role_bootstrap_port 18998 "$index")"
        PD_PREFILL_NAMES+=("$role_name")
    else
        port="$(pd_role_port USER_VLLM_PD_DECODE_PORT "$http_base" 100 "$index")"
        rpc_port="$(pd_role_port USER_VLLM_PD_DECODE_RPC_PORT "$rpc_base" 100 "$index")"
        coord_port="$(pd_role_port USER_VLLM_PD_DECODE_COORD_PORT "$coord_base" 100 "$index")"
        ready_port_base="$(pd_role_ready_base USER_VLLM_PD_DECODE_READY_PORT_BASE "$ready_base" 100 "$index")"
        PD_DECODE_NAMES+=("$role_name")
    fi

    pd_write_role_preset "$role" "$index" "$port" "$rpc_port" "$coord_port" "$ready_port_base" "$role_dir" "$mooncake_config_path" "$bootstrap_port" "$preset_file"

    http_log="$(pd_http_log_for_role "$role_dir")"
    if [ "$role" = "prefill" ]; then
        log_info "P/D $role_name: http=$port, bootstrap=$bootstrap_port, dp_rpc=$rpc_port, coord=$coord_port, ready_base=$ready_port_base"
    else
        log_info "P/D $role_name: http=$port, dp_rpc=$rpc_port, coord=$coord_port, ready_base=$ready_port_base"
    fi
    pd_start_role_launcher "$role_name" "$role_dir" "$preset_file"
    pd_wait_for_http_service "$role_name" "$port" "$http_log"

    if [ "$role" = "prefill" ]; then
        PD_PREFILL_URL_LIST+=("http://127.0.0.1:${port}")
        PD_PREFILL_BOOTSTRAP_PORT_LIST+=("$bootstrap_port")
        PD_PREFILL_URLS="${PD_PREFILL_URLS:+$PD_PREFILL_URLS,}http://127.0.0.1:${port}"
        PD_PREFILL_BOOTSTRAP_PORTS="${PD_PREFILL_BOOTSTRAP_PORTS:+$PD_PREFILL_BOOTSTRAP_PORTS,}$bootstrap_port"
        if [ "$index" -eq 0 ]; then
            PD_PREFILL_DIR="$role_dir"
        fi
    else
        PD_DECODE_URL_LIST+=("http://127.0.0.1:${port}")
        PD_DECODE_URLS="${PD_DECODE_URLS:+$PD_DECODE_URLS,}http://127.0.0.1:${port}"
        if [ "$index" -eq 0 ]; then
            PD_DECODE_DIR="$role_dir"
        fi
    fi
}

pd_start_router() {
    local router_dir="$1"
    local router_port="$2"
    local router_preset="$3"
    local router_bin=""
    local router_policy="${USER_VLLM_PD_ROUTER_POLICY:-cache_aware}"
    local router_prom_port="${USER_VLLM_PD_ROUTER_PROMETHEUS_PORT:-$(pd_find_free_port 29000)}"
    local idx

    if ! router_bin="$(pd_resolve_router_bin)"; then
        log_error "未找到 vllm-router 可执行文件；请确认已在虚拟环境中安装 vllm-router，或设置 VLLM_ROUTER_BIN"
        return 1
    fi

    mkdir -p "$router_dir/logs"
    pd_write_router_preset "$router_port" "$router_preset"
    ln -sf "$(basename "$router_preset")" "$(dirname "$router_preset")/proxy.sh"
    TEST_ENV_ARGS=("-e" "$router_preset")

    PD_ROUTER_LOG="$router_dir/router.log"
    PD_ROUTER_PORT="$router_port"
    PD_PROXY_LOG="$PD_ROUTER_LOG"
    PD_PROXY_PORT="$PD_ROUTER_PORT"

    local router_args=(
        --host "${USER_VLLM_PD_ROUTER_HOST:-127.0.0.1}"
        --port "$router_port"
        --vllm-pd-disaggregation
        --kv-connector mooncake
        --policy "$router_policy"
        --request-timeout-secs "${USER_VLLM_PD_ROUTER_TIMEOUT_SECS:-21600}"
        --worker-startup-timeout-secs "${VLLM_TEST_MAX_WAIT:-600}"
        --health-check-timeout-secs "${USER_VLLM_PD_ROUTER_HEALTH_TIMEOUT_SECS:-30}"
        --prometheus-host "${USER_VLLM_PD_ROUTER_PROMETHEUS_HOST:-127.0.0.1}"
        --prometheus-port "$router_prom_port"
    )
    if [ -n "${USER_VLLM_PD_PREFILL_POLICY:-}" ]; then
        router_args+=(--prefill-policy "$USER_VLLM_PD_PREFILL_POLICY")
    fi
    if [ -n "${USER_VLLM_PD_DECODE_POLICY:-}" ]; then
        router_args+=(--decode-policy "$USER_VLLM_PD_DECODE_POLICY")
    fi
    if [ -n "${USER_VLLM_PD_INTRA_NODE_DP_SIZE:-}" ]; then
        router_args+=(--intra-node-data-parallel-size "$USER_VLLM_PD_INTRA_NODE_DP_SIZE")
    fi
    for idx in "${!PD_PREFILL_URL_LIST[@]}"; do
        router_args+=(--prefill "${PD_PREFILL_URL_LIST[$idx]}" "${PD_PREFILL_BOOTSTRAP_PORT_LIST[$idx]}")
    done
    for idx in "${!PD_DECODE_URL_LIST[@]}"; do
        router_args+=(--decode "${PD_DECODE_URL_LIST[$idx]}")
    done
    if [ -n "${USER_VLLM_PD_ROUTER_EXTRA_ARGS:-}" ]; then
        local extra_router_args=()
        # shellcheck disable=SC2206
        extra_router_args=(${USER_VLLM_PD_ROUTER_EXTRA_ARGS})
        router_args+=("${extra_router_args[@]}")
    fi

    log_info "启动 vllm-router ($router_bin, policy=$router_policy, prom_port=$router_prom_port)..."
    pd_start_process_in_dir "$router_dir" "$router_dir/router.pid" \
        "$router_bin" "${router_args[@]}" > "$PD_ROUTER_LOG" 2>&1
    pd_wait_for_http_service "vllm-router" "$router_port" "$PD_ROUTER_LOG"
}

pd_start() {
    local pd_root="$LOG_DIR/pd/${RUN_START_TS}_${PRESET_TAG}"
    local router_dir="$pd_root/router"
    local prefill_count="${USER_VLLM_PD_PREFILL_COUNT:-1}"
    local decode_count="${USER_VLLM_PD_DECODE_COUNT:-1}"
    local router_port
    local rpc_base="${USER_VLLM_DATA_PARALLEL_RPC_PORT:-13345}"
    local index
    local pd_kv_backend="${USER_VLLM_PD_KV_BACKEND:-mooncake}"
    local mooncake_config_path="${USER_VLLM_PD_MOONCAKE_CONFIG_PATH:-${MOONCAKE_CONFIG_PATH:-}}"
    local decode_http_base
    local decode_rpc_base
    local decode_coord_base
    local decode_ready_base

    if [ "$pd_kv_backend" = "example" ]; then
        log_error "USER_VLLM_PD_KV_BACKEND=example 与 pd_proxy.py 已废弃移除；请改用 vllm-router 支持的 mooncake 或 mooncake_multi"
        return 1
    fi
    if [ "$pd_kv_backend" != "mooncake" ] && [ "$pd_kv_backend" != "mooncake_multi" ]; then
        log_error "USER_VLLM_PD_KV_BACKEND 仅支持 mooncake 或 mooncake_multi，当前为: $pd_kv_backend"
        return 1
    fi
    if [ "$pd_kv_backend" = "mooncake_multi" ]; then
        if [ -z "$mooncake_config_path" ] || [ ! -r "$mooncake_config_path" ]; then
            log_error "mooncake_multi 模式需要设置可读的 MOONCAKE_CONFIG_PATH（或 USER_VLLM_PD_MOONCAKE_CONFIG_PATH）"
            return 1
        fi
        mooncake_config_path="$(realpath "$mooncake_config_path")"
        export MOONCAKE_CONFIG_PATH="$mooncake_config_path"
        export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
    fi
    if ! [[ "$prefill_count" =~ ^[0-9]+$ ]] || [ "$prefill_count" -eq 0 ]; then
        log_error "USER_VLLM_PD_PREFILL_COUNT 必须是大于 0 的整数"
        return 1
    fi
    if ! [[ "$decode_count" =~ ^[0-9]+$ ]] || [ "$decode_count" -eq 0 ]; then
        log_error "USER_VLLM_PD_DECODE_COUNT 必须是大于 0 的整数"
        return 1
    fi

    PD_ROOT="$pd_root"
    MPI_CLEANUP_LOG="$PD_ROOT/mpi_cleanup.log"
    PD_PREFILL_URLS=""
    PD_DECODE_URLS=""
    PD_PREFILL_BOOTSTRAP_PORTS=""
    PD_PREFILL_URL_LIST=()
    PD_PREFILL_BOOTSTRAP_PORT_LIST=()
    PD_DECODE_URL_LIST=()
    PD_PREFILL_NAMES=()
    PD_DECODE_NAMES=()
    router_port="${USER_VLLM_PD_ROUTER_PORT:-${USER_VLLM_PD_PROXY_PORT:-$(pd_find_free_port "$USER_VLLM_PORT")}}"

    decode_http_base=$((USER_VLLM_PORT + (prefill_count + 1) * 100))
    decode_rpc_base=$((rpc_base + (prefill_count + 1) * 1000))
    decode_coord_base=$((16555 + prefill_count * 1000))
    decode_ready_base=$((29888 + prefill_count * 1000))

    mkdir -p "$pd_root" "$router_dir/logs"

    log_info "P/D 工作目录: $pd_root"
    log_info "P/D 拓扑: ${prefill_count}P x ${decode_count}D"
    log_info "P/D KV backend: $pd_kv_backend"
    log_info "P/D Mooncake protocol: ${USER_VLLM_PD_MOONCAKE_PROTOCOL:-tcp}"
    if [ "$pd_kv_backend" = "mooncake_multi" ]; then
        log_info "P/D Mooncake Store config: $mooncake_config_path"
    fi

    for ((index = 0; index < prefill_count; index++)); do
        pd_create_role "prefill" "$index" "$((USER_VLLM_PORT + 100))" "$((rpc_base + 1000))" 16555 29888 "$pd_root" "$mooncake_config_path"
    done
    for ((index = 0; index < decode_count; index++)); do
        pd_create_role "decode" "$index" "$decode_http_base" "$decode_rpc_base" "$decode_coord_base" "$decode_ready_base" "$pd_root" "$mooncake_config_path"
    done

    log_info "P/D vllm-router 端口: $router_port"
    log_info "P/D prefill URLs: $PD_PREFILL_URLS"
    log_info "P/D prefill bootstrap ports: $PD_PREFILL_BOOTSTRAP_PORTS"
    log_info "P/D decode URLs: $PD_DECODE_URLS"
    pd_start_router "$router_dir" "$router_port" "$pd_root/router.sh"
}

pd_validate_transfer() {
    if [ "$DISAGG_PREFILL" -ne 1 ] || [ "$TEST_MODE" = "none" ]; then
        return 0
    fi

    if grep -R "Receiving Mooncake KV\|pulling kv_caches\|MooncakeXferMetadata\|Sending to .* done" \
        "$PD_ROOT"/decode*/logs "$PD_ROOT"/prefill*/logs "$PD_ROOT"/decode*/launch.log "$PD_ROOT"/prefill*/launch.log >/dev/null 2>&1; then
        log_success "P/D Mooncake 侧确认触发 KV transfer"
        return 0
    fi
    log_error "P/D Mooncake 侧未发现 KV transfer 日志"
    log_error "Decode 日志目录: $PD_ROOT/decode*/logs"
    return 1
}

pd_print_log_locations() {
    log_info "  P/D root: $PD_ROOT"
    log_info "  Prefill roles: $(pd_join_csv "${PD_PREFILL_NAMES[@]}")"
    log_info "  Decode roles:  $(pd_join_csv "${PD_DECODE_NAMES[@]}")"
    log_info "  Router: $PD_ROUTER_LOG"
}
