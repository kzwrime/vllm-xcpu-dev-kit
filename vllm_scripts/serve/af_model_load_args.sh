#!/bin/bash
# Match A's optional-argument splitting, forwarding only shared model-load flags.
af_model_load_args() {
    AF_MODEL_LOAD_ARGS=()
    local args=() arg i
    local optional_args="${VLLM_OPTIONAL_ARGS:-}"
    read -r -a args <<< "${optional_args//$'\n'/ }"
    for ((i=0; i<${#args[@]}; i++)); do
        arg="${args[i]}"
        case "$arg" in
            --revision)
                if (( i + 1 >= ${#args[@]} )); then
                    echo '--revision requires a value' >&2
                    return 1
                fi
                AF_MODEL_LOAD_ARGS+=("$arg" "${args[i+1]}")
                i=$((i + 1))
                ;;
            --revision=*|--trust-remote-code|--no-trust-remote-code)
                AF_MODEL_LOAD_ARGS+=("$arg")
                ;;
        esac
    done
}
