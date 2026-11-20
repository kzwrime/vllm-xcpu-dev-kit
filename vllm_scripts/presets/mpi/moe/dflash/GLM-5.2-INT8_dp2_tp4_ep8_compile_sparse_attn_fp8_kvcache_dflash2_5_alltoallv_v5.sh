#!/bin/bash

# GLM-5.2-INT8 (UE8M0 [16, 16]) at DP=2, TP=4, EP=8 over mpi_alltoallv_v5,
# torch.compile, sparse MLA with FP8/INT8 KV cache and GLM-5.3 DFlash2 drafter (5 speculative tokens).

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export PD_MODE="MIXED"
export VLLM_SHARED_EXPERT_DISABLE_TP=0
export USER_VLLM_MAX_NUM_BATCHED_TOKENS="${USER_VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-3600}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
_OVERRIDE_USER_VLLM_MODEL="${USER_VLLM_MODEL:-}"
source "$SCRIPT_DIR/user_env_template.sh"

unset VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16
export TORCH_XCPU_ENABLE_CHECK=0
export USER_VLLM_EAGER_OR_NOT="${USER_VLLM_EAGER_OR_NOT-}"
export USER_VLLM_MODEL="${_OVERRIDE_USER_VLLM_MODEL:-$SCRIPT_DIR/../models/ZhipuAI/GLM-5.2-INT8-UE8M0-16x16}"
unset _OVERRIDE_USER_VLLM_MODEL
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=4
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))

export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1

_CPUS=$(lscpu -p=CPU | grep -cv '^#')
export OMP_NUM_THREADS="${USER_OMP_NUM_THREADS:-$((_CPUS / (USER_VLLM_DATA_PARALLEL_SIZE * USER_VLLM_MPC_SIZE)))}"
unset _CPUS
export VLLM_XCPU_CPU_PARTITION="${VLLM_XCPU_CPU_PARTITION:-1}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-1}"
unset OMP_THREAD_LIMIT

_VLLM_OPTIONAL_ARGS+=" --all2all-backend mpi_alltoallv_v5"
_VLLM_OPTIONAL_ARGS+=" --kv-cache-dtype ${USER_VLLM_KV_CACHE_DTYPE:-fp8}"
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
echo "Preset: ${preset_dir}/${preset_name} | INT8 UE8M0 16x16 DP2 TP4 EP8 alltoallv_v5 compile," \
     "DFlash2 draft: ${_GLM_DFLASH_MODEL}, ${USER_VLLM_KV_CACHE_DTYPE:-fp8} KV cache," \
     "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
unset _GLM_DFLASH_MODEL
