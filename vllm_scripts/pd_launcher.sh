#!/bin/bash
# P/D disaggregated launcher helpers shared by startup and test wrappers.
#
# This file intentionally keeps P/D topology concerns out of wrapper scripts.
# The current proxy path supports N prefill instances x M decode instances by
# round-robin request routing. The default topology is 1P x 1D.

PD_ROOT=""
PD_PREFILL_DIR=""
PD_DECODE_DIR=""
PD_PROXY_LOG=""
PD_PROXY_PORT=""
PD_PREFILL_URLS=""
PD_DECODE_URLS=""
PD_PREFILL_BOOTSTRAP_PORTS=""
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

pd_write_role_preset() {
    local role="$1"
    local index="$2"
    local port="$3"
    local rpc_port="$4"
    local coord_port="$5"
    local ready_port_base="$6"
    local role_dir="$7"
    local storage_dir="$8"
    local bootstrap_port="$9"
    local output_file="${10}"
    local pd_kv_backend="${USER_VLLM_PD_KV_BACKEND:-example}"

    cat > "$output_file" <<EOF
#!/bin/bash
source "$ORIG_CONFIG_FILE"

export USER_VLLM_PORT=$port
export USER_VLLM_DATA_PARALLEL_RPC_PORT=$rpc_port
export VLLM_MPI_COORD_PORT=$coord_port
export VLLM_MPI_ENV_EXPORT_FILE="$role_dir/vllm_mpi_env_server.sh"
export VLLM_MP_RPC_READY_PORT_BASE=$ready_port_base
EOF

    if [ "$pd_kv_backend" = "example" ]; then
        cat >> "$output_file" <<EOF
_VLLM_PD_KV_TRANSFER_CONFIG='{"kv_connector":"ExampleConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"$storage_dir"}}'
export VLLM_OPTIONAL_ARGS="\${VLLM_OPTIONAL_ARGS} --kv-transfer-config \${_VLLM_PD_KV_TRANSFER_CONFIG} --no-disable-hybrid-kv-cache-manager"
EOF
    elif [ "$pd_kv_backend" = "mooncake" ]; then
        local kv_role="kv_consumer"
        [ "$role" = "prefill" ] && kv_role="kv_producer"
        cat >> "$output_file" <<EOF
export MOONCAKE_PROTOCOL="${USER_VLLM_PD_MOONCAKE_PROTOCOL:-tcp}"
_VLLM_PD_KV_TRANSFER_CONFIG='{"kv_connector":"MooncakeConnector","kv_role":"$kv_role","kv_connector_extra_config":{"mooncake_protocol":"${USER_VLLM_PD_MOONCAKE_PROTOCOL:-tcp}","num_workers":${USER_VLLM_PD_MOONCAKE_NUM_WORKERS:-4}}}'
export VLLM_OPTIONAL_ARGS="\${VLLM_OPTIONAL_ARGS} --kv-transfer-config \${_VLLM_PD_KV_TRANSFER_CONFIG} --no-disable-hybrid-kv-cache-manager"
EOF
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

pd_write_proxy_preset() {
    local proxy_port="$1"
    local output_file="$2"

    cat > "$output_file" <<EOF
#!/bin/bash
source "$ORIG_CONFIG_FILE"
export USER_VLLM_PORT=$proxy_port
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
    local storage_dir="$8"
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

    pd_write_role_preset "$role" "$index" "$port" "$rpc_port" "$coord_port" "$ready_port_base" "$role_dir" "$storage_dir" "$bootstrap_port" "$preset_file"

    http_log="$(pd_http_log_for_role "$role_dir")"
    if [ "$role" = "prefill" ] && [ "${USER_VLLM_PD_KV_BACKEND:-example}" = "mooncake" ]; then
        log_info "P/D $role_name: http=$port, bootstrap=$bootstrap_port, dp_rpc=$rpc_port, coord=$coord_port, ready_base=$ready_port_base"
    else
        log_info "P/D $role_name: http=$port, dp_rpc=$rpc_port, coord=$coord_port, ready_base=$ready_port_base"
    fi
    pd_start_role_launcher "$role_name" "$role_dir" "$preset_file"
    pd_wait_for_http_service "$role_name" "$port" "$http_log"

    if [ "$role" = "prefill" ]; then
        PD_PREFILL_URLS="${PD_PREFILL_URLS:+$PD_PREFILL_URLS,}http://127.0.0.1:${port}"
        PD_PREFILL_BOOTSTRAP_PORTS="${PD_PREFILL_BOOTSTRAP_PORTS:+$PD_PREFILL_BOOTSTRAP_PORTS,}$bootstrap_port"
        [ "$index" -eq 0 ] && PD_PREFILL_DIR="$role_dir"
    else
        PD_DECODE_URLS="${PD_DECODE_URLS:+$PD_DECODE_URLS,}http://127.0.0.1:${port}"
        [ "$index" -eq 0 ] && PD_DECODE_DIR="$role_dir"
    fi
}

pd_start_proxy() {
    local proxy_dir="$1"
    local proxy_port="$2"
    local proxy_preset="$3"
    local pd_kv_backend="${USER_VLLM_PD_KV_BACKEND:-example}"

    mkdir -p "$proxy_dir/logs"
    pd_write_proxy_preset "$proxy_port" "$proxy_preset"
    TEST_ENV_ARGS=("-e" "$proxy_preset")

    PD_PROXY_LOG="$proxy_dir/proxy.log"
    PD_PROXY_PORT="$proxy_port"

    log_info "启动 P/D proxy..."
    if [ "$pd_kv_backend" = "mooncake" ]; then
        pd_start_process_in_dir "$proxy_dir" "$proxy_dir/proxy.pid" \
            python "$SCRIPT_DIR/serve_test/pd_proxy.py" \
            --mode mooncake \
            --port "$proxy_port" \
            --prefill-url "$PD_PREFILL_URLS" \
            --prefill-bootstrap-port "$PD_PREFILL_BOOTSTRAP_PORTS" \
            --decode-url "$PD_DECODE_URLS" > "$PD_PROXY_LOG" 2>&1
    else
        pd_start_process_in_dir "$proxy_dir" "$proxy_dir/proxy.pid" \
            python "$SCRIPT_DIR/serve_test/pd_proxy.py" \
            --mode example \
            --port "$proxy_port" \
            --prefill-url "$PD_PREFILL_URLS" \
            --decode-url "$PD_DECODE_URLS" > "$PD_PROXY_LOG" 2>&1
    fi
    pd_wait_for_http_service "proxy" "$proxy_port" "$PD_PROXY_LOG"
}

pd_start() {
    local pd_root="$LOG_DIR/pd/${RUN_START_TS}_${PRESET_TAG}"
    local proxy_dir="$pd_root/proxy"
    local storage_dir="$pd_root/example_connector_storage"
    local prefill_count="${USER_VLLM_PD_PREFILL_COUNT:-1}"
    local decode_count="${USER_VLLM_PD_DECODE_COUNT:-1}"
    local proxy_port
    local rpc_base="${USER_VLLM_DATA_PARALLEL_RPC_PORT:-13345}"
    local index
    local pd_kv_backend="${USER_VLLM_PD_KV_BACKEND:-example}"

    if [ "$pd_kv_backend" != "example" ] && [ "$pd_kv_backend" != "mooncake" ]; then
        log_error "USER_VLLM_PD_KV_BACKEND 仅支持 example 或 mooncake，当前为: $pd_kv_backend"
        return 1
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
    PD_PREFILL_URLS=""
    PD_DECODE_URLS=""
    PD_PREFILL_BOOTSTRAP_PORTS=""
    PD_PREFILL_NAMES=()
    PD_DECODE_NAMES=()
    proxy_port="${USER_VLLM_PD_PROXY_PORT:-$(pd_find_free_port "$USER_VLLM_PORT")}"

    mkdir -p "$pd_root" "$proxy_dir/logs" "$storage_dir"

    log_info "P/D 工作目录: $pd_root"
    log_info "P/D 拓扑: ${prefill_count}P x ${decode_count}D"
    log_info "P/D KV backend: ${USER_VLLM_PD_KV_BACKEND:-example}"
    if [ "${USER_VLLM_PD_KV_BACKEND:-example}" = "example" ]; then
        log_info "P/D KV 共享目录: $storage_dir"
    else
        log_info "P/D Mooncake protocol: ${USER_VLLM_PD_MOONCAKE_PROTOCOL:-tcp}"
    fi

    for ((index = 0; index < prefill_count; index++)); do
        pd_create_role "prefill" "$index" "$((USER_VLLM_PORT + 100))" "$((rpc_base + 1000))" 16555 29888 "$pd_root" "$storage_dir"
    done
    for ((index = 0; index < decode_count; index++)); do
        pd_create_role "decode" "$index" "$((USER_VLLM_PORT + 200))" "$((rpc_base + 2000))" 17555 30888 "$pd_root" "$storage_dir"
    done

    log_info "P/D proxy 端口: $proxy_port"
    log_info "P/D prefill URLs: $PD_PREFILL_URLS"
    [ "${USER_VLLM_PD_KV_BACKEND:-example}" = "mooncake" ] && log_info "P/D prefill bootstrap ports: $PD_PREFILL_BOOTSTRAP_PORTS"
    log_info "P/D decode URLs: $PD_DECODE_URLS"
    pd_start_proxy "$proxy_dir" "$proxy_port" "$pd_root/proxy.sh"
}

pd_validate_transfer() {
    if [ "$DISAGG_PREFILL" -ne 1 ] || [ "$TEST_MODE" = "none" ]; then
        return 0
    fi

    if [ "${USER_VLLM_PD_KV_BACKEND:-example}" = "mooncake" ]; then
        if grep -R "Receiving Mooncake KV\|pulling kv_caches\|MooncakeXferMetadata" "$PD_ROOT"/decode*/logs "$PD_ROOT"/prefill*/logs >/dev/null 2>&1; then
            log_success "P/D Mooncake 侧确认触发 KV transfer"
            return 0
        fi
        log_error "P/D Mooncake 侧未发现 KV transfer 日志"
    elif grep -R "External Cache Hit" "$PD_ROOT"/decode*/logs >/dev/null 2>&1; then
        log_success "P/D decode 侧确认命中外部 KV cache"
        return 0
    else
        log_error "P/D decode 侧未发现 External Cache Hit"
    fi

    log_error "Decode 日志目录: $PD_ROOT/decode*/logs"
    return 1
}

pd_print_log_locations() {
    log_info "  P/D root: $PD_ROOT"
    log_info "  Prefill roles: $(pd_join_csv "${PD_PREFILL_NAMES[@]}")"
    log_info "  Decode roles:  $(pd_join_csv "${PD_DECODE_NAMES[@]}")"
    log_info "  Proxy: $PD_PROXY_LOG"
}
