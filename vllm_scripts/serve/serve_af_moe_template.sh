#!/bin/bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/common.sh"
parse_args_and_load_env "$SCRIPT_DIR" "$@"
source "$SCRIPT_DIR/serve/af_model_load_args.sh"
af_model_load_args
echo "AFD placement role=F host=$(hostname) global_rank=${OMPI_COMM_WORLD_RANK:-?} local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-?}"
exec python -m vllm_xcpu_plugin.af_ep.moe \
    --model "$USER_VLLM_MODEL" \
    --max-num-batched-tokens "$USER_VLLM_MAX_NUM_BATCHED_TOKENS" \
    --load-format "${USER_VLLM_LOAD_FORMAT:-auto}" \
    "${AF_MODEL_LOAD_ARGS[@]}"
