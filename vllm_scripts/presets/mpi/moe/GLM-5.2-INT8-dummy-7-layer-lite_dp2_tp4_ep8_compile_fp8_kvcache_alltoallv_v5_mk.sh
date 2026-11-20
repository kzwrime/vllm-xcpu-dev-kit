#!/bin/bash

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export USER_VLLM_MODEL="${USER_VLLM_MODEL:-$SCRIPT_DIR/../models/ZhipuAI/GLM-5.2-INT8-UE8M0-dummy-7-layer-lite}"
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/GLM-5.2-INT8_dp2_tp4_ep8_compile_fp8_kvcache_alltoallv_v5_mk.sh"

_VLLM_OPTIONAL_ARGS+=" --load-format dummy"
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"
