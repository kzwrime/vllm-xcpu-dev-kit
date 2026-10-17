#!/bin/bash

# GLM5.2 AFD: A = DP1/TP1 (1 rank), F = EP2 (2 ranks), V8 transport, FP8 checkpoint.
# AFD does not duplicate weights: A holds attention/dense/shared experts, F the routed experts.
# DFlash2 (GLM-5.3 drafter, 5 tokens) drafts on A; F only serves target MoE layers.
# Validated 2026-10-15: docs/dev_log/20261015_GLM5.2-AFD-DFlash2-adaptation.md

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export PD_MODE="MIXED"
# v0.31 auto-selects breakable CUDA graphs for DSA, which disables compile.
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export USER_VLLM_MAX_NUM_BATCHED_TOKENS="${USER_VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-2400}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
source "$SCRIPT_DIR/user_env_template.sh"

# Do not carry the MXFP4-only activation override into this preset.
unset VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16
export TORCH_XCPU_ENABLE_CHECK=0
export USER_VLLM_EAGER_OR_NOT=""
export USER_VLLM_AFD_F_COMPILE=1
export USER_VLLM_MODEL="ZhipuAI/GLM-5.2-FP8"
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=1
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))

export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_XCPU_ENABLE_AF_EP=1
export USER_VLLM_EP_SIZE="${USER_VLLM_EP_SIZE:-2}"
export USER_VLLM_MPI_SIZE=$((USER_VLLM_DATA_PARALLEL_SIZE * USER_VLLM_MPC_SIZE + USER_VLLM_EP_SIZE))
export VLLM_MPI_WORKER_TEMPLATE="$SCRIPT_DIR/serve/serve_afd_mp_rpc_all_mpi_template.sh"
# Single-host V8 uses MPI collectives only; V7 RMA would need the UCX OSC options.
export VLLM_MPI_RUN_ARGS="${VLLM_MPI_RUN_ARGS:---allow-run-as-root --bind-to none --map-by slot}"

_VLLM_OPTIONAL_ARGS+=' --compilation-config {"mode":3}'
# Forwarded to F by serve/af_model_load_args.sh so both sides use V8.
_VLLM_OPTIONAL_ARGS+=" --all2all-backend mpi_alltoallv_v8"
_VLLM_OPTIONAL_ARGS+=" --kv-cache-dtype fp8"
_VLLM_OPTIONAL_ARGS+=' --kernel-config {"enable_jit_warmup":false}'
_VLLM_OPTIONAL_ARGS+=' --tool-call-parser glm47'
_VLLM_OPTIONAL_ARGS+=' --enable-auto-tool-choice'
_VLLM_OPTIONAL_ARGS+=' --reasoning-parser glm45'
# Keep sliding-window attention, but allocate its cache through the full-cache
# manager: mixed MLA/DSA/draft page sizes cannot use hybrid page unification.
_VLLM_OPTIONAL_ARGS+=" --disable-hybrid-kv-cache-manager"
_GLM_DFLASH_MODEL="${USER_VLLM_DFLASH_MODEL:-${HOME}/.cache/modelscope/models/incoai--GLM-5.3-DFlash2/snapshots/master}"
# XCPU grouped draft-cache writes use BF16; the target MLA cache remains FP8.
_VLLM_OPTIONAL_ARGS+=" --speculative-config {\"method\":\"dflash\",\"model\":\"${_GLM_DFLASH_MODEL}\",\"num_speculative_tokens\":5,\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"${USER_VLLM_DFLASH_SAMPLE_METHOD:-greedy}\"}"
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "AF-EP preset: ${preset_dir}/${preset_name} A=1 F=${USER_VLLM_EP_SIZE} | FP8 checkpoint, FP8 KV cache"
unset _GLM_DFLASH_MODEL
