#!/bin/bash
# Each invocation owns one run directory; copy a classified snapshot after cleanup.

test_prepare_log_directory() {
    mkdir -p "$LOG_ROOT/runs" "$SUCCESS_ROOT" "$FAILED_ROOT" "$LOG_ROOT/stopped" || return "$?"
    LOG_DIR=$(mktemp -d "$LOG_ROOT/runs/${RUN_START_TS}_${PRESET_TAG}.XXXXXX") || return "$?"
    export VLLM_RUN_LOG_DIR="$LOG_DIR"
}

archive_run_logs() {
    local root="$1" label="$2"
    local archive_dir="$root/${LOG_DIR##*/}"
    if [ "${DISAGG_PREFILL:-0}" -eq 1 ] && [ -d "${PD_ROOT:-}" ]; then
        cp -a "$PD_ROOT" "$LOG_DIR/pd" || return "$?"
    fi
    cp -a "$LOG_DIR" "$archive_dir" || return "$?"
    log_info "运行日志保留在: $LOG_DIR"
    log_info "${label}日志已归档到: $archive_dir"
}

cleanup() {
    local exit_code=$? cleanup_code=0 archive_code=0
    local archive_root archive_label
    # Signals during cleanup must not invoke a second archive or mask its result.
    trap - EXIT
    trap '' INT TERM
    if launcher_cleanup_processes; then
        :
    else
        cleanup_code=$?
    fi
    local final_code=$exit_code
    if [ "$final_code" -eq 0 ] && [ "$cleanup_code" -ne 0 ]; then
        final_code=$cleanup_code
    fi
    if [ "$final_code" -ne 0 ]; then
        archive_root="$FAILED_ROOT"
        archive_label="失败"
    elif [ "$TEST_MODE" = none ]; then
        archive_root="$LOG_ROOT/stopped"
        archive_label="停止"
    else
        archive_root="$SUCCESS_ROOT"
        archive_label="成功"
    fi
    printf '{"run_exit_code":%d,"cleanup_exit_code":%d,"exit_code":%d}\n' \
        "$exit_code" "$cleanup_code" "$final_code" > "$LOG_DIR/run_result.json"
    if archive_run_logs "$archive_root" "$archive_label"; then
        :
    else
        archive_code=$?
        log_error "日志归档失败，原始日志保留在: $LOG_DIR"
    fi
    if [ "$final_code" -eq 0 ] && [ "$archive_code" -ne 0 ]; then
        final_code=$archive_code
    fi
    if [ "$archive_code" -ne 0 ]; then
        printf '{"run_exit_code":%d,"cleanup_exit_code":%d,"exit_code":%d,"archive_exit_code":%d}\n' \
            "$exit_code" "$cleanup_code" "$final_code" "$archive_code" \
            > "$LOG_DIR/run_result.json" || true
    fi
    exit "$final_code"
}
