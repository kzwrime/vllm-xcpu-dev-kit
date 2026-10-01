#!/bin/bash
# GLM-5.2 MXFP4 sparse MLA with the GLM-5.3 DFlash2 drafter.
# Configuration: DP=2, TP=2, PP=1, EP enabled, eager mode
# The draft was trained for GLM-5.3; GLM-5.2 verifies every proposed token.

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

# MIXED enables expert parallelism through user_env_template.sh.
export PD_MODE="MIXED"
export USER_VLLM_MAX_NUM_BATCHED_TOKENS="${USER_VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-1800}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
source "$SCRIPT_DIR/user_env_template.sh"

# The published AMD checkpoint declares dynamic MXFP4 activations (W4A4).
# XCPU currently consumes the same packed weights with BF16 activations (W4A16).
export VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16=1
export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="amd/GLM-5.2-MXFP4"
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))

export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1

_VLLM_OPTIONAL_ARGS+=" --all2all-backend mpi_alltoallv_v5"
_VLLM_OPTIONAL_ARGS+=" --kv-cache-dtype fp8"
_VLLM_OPTIONAL_ARGS+=' --kernel-config {"enable_jit_warmup":false}'
_VLLM_OPTIONAL_ARGS+=' --tool-call-parser glm47'
_VLLM_OPTIONAL_ARGS+=' --enable-auto-tool-choice'
_VLLM_OPTIONAL_ARGS+=' --reasoning-parser glm45'
# Keep sliding-window attention, but allocate its cache through the full-cache
# manager: mixed MLA/DSA/draft page sizes cannot use hybrid page unification.
_VLLM_OPTIONAL_ARGS+=" --disable-hybrid-kv-cache-manager"
# The target is cached on Hugging Face, while the draft is on ModelScope.
_GLM_DFLASH_MODEL="${USER_VLLM_DFLASH_MODEL:-${HOME}/.cache/modelscope/models/incoai--GLM-5.3-DFlash2/snapshots/master}"
# XCPU grouped draft-cache writes use BF16; the target MLA cache remains FP8.
_VLLM_OPTIONAL_ARGS+=" --speculative-config {\"method\":\"dflash\",\"model\":\"${_GLM_DFLASH_MODEL}\",\"num_speculative_tokens\":5,\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"${USER_VLLM_DFLASH_SAMPLE_METHOD:-greedy}\"}"
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "Preset: ${preset_dir}/${preset_name} | DFlash2 draft: ${_GLM_DFLASH_MODEL}"
unset _GLM_DFLASH_MODEL
