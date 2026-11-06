#!/bin/bash

# GLM-5.2-FP8 at TP4 (16 heads per rank), torch.compile, sparse MLA with FP8 KV
# cache, DFlash2 draft (5 speculative tokens). TP4 is the layout the fused MLA
# front half (VLLM_XCPU_FUSED_MLA_QNORM_QB_UK, default off) instantiates for.
# Companion to GLM-5.2-FP8_dp1_tp4_ep4_compile_fp8_kvcache.sh plus the DFlash2 lines
# of the dp2_tp2 dflash2_5 presets.

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export PD_MODE="MIXED"
export VLLM_SHARED_EXPERT_DISABLE_TP=0
export USER_VLLM_MAX_NUM_BATCHED_TOKENS="${USER_VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-3600}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
source "$SCRIPT_DIR/user_env_template.sh"

unset VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16
export TORCH_XCPU_ENABLE_CHECK=0
export USER_VLLM_EAGER_OR_NOT=""
export USER_VLLM_MODEL="ZhipuAI/GLM-5.2-FP8"
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=4
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))

export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1

_VLLM_OPTIONAL_ARGS+=" --all2all-backend all_to_all_single"
_VLLM_OPTIONAL_ARGS+=" --kv-cache-dtype fp8"
_VLLM_OPTIONAL_ARGS+=' --kernel-config {"enable_jit_warmup":false}'
_VLLM_OPTIONAL_ARGS+=" --disable-hybrid-kv-cache-manager"
_VLLM_OPTIONAL_ARGS+=' --tool-call-parser glm47'
_VLLM_OPTIONAL_ARGS+=' --enable-auto-tool-choice'
_VLLM_OPTIONAL_ARGS+=' --reasoning-parser glm45'
_GLM_DFLASH_MODEL="${USER_VLLM_DFLASH_MODEL:-${HOME}/.cache/modelscope/models/incoai--GLM-5.3-DFlash2/snapshots/master}"
_VLLM_OPTIONAL_ARGS+=" --speculative-config {\"method\":\"dflash\",\"model\":\"${_GLM_DFLASH_MODEL}\",\"num_speculative_tokens\":5,\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"${USER_VLLM_DFLASH_SAMPLE_METHOD:-greedy}\"}"
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "Preset: ${preset_dir}/${preset_name} | TP4 compile, DFlash2 draft: ${_GLM_DFLASH_MODEL}, fused_mla_qnorm_qb_uk=${VLLM_XCPU_FUSED_MLA_QNORM_QB_UK:-0}"
unset _GLM_DFLASH_MODEL
