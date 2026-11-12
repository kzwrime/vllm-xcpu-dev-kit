#!/bin/bash

# DeepSeek-V4.1-Flash 8-layer dummy (engram layers 1 and 2): A4 (DP2/TP2, EP over mpi_alltoallv_v6, MoE sequence parallelism, compile) + E2.
# Engram disaggregation: the Engram tables are served by USER_VLLM_ENGRAM_SIZE
# dedicated MPI ranks after the attention ranks (docs/upgrade/13-dsv41-engram-disagg.md).

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export PD_MODE="MIXED"
# v0.31 auto-selects breakable CUDA graphs for DeepSeek-V4.1.
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export USER_VLLM_MAX_NUM_BATCHED_TOKENS="${USER_VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-1800}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
source "$SCRIPT_DIR/user_env_template.sh"
unset VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16

export USER_VLLM_EAGER_OR_NOT=""
export TORCH_XCPU_ENABLE_CHECK=0
export USER_VLLM_MODEL="../models/deepseek-ai/DeepSeek-V4.1-Flash-dummy-8-layer-lite"
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))

export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1
export VLLM_XCPU_ENABLE_ENGRAM_DISAGG=1
# Must divide the 24 hash columns: 1/2/3/4/6/8/12/24.
export USER_VLLM_ENGRAM_SIZE="${USER_VLLM_ENGRAM_SIZE:-2}"
export USER_VLLM_MPI_SIZE=$((USER_VLLM_DATA_PARALLEL_SIZE * USER_VLLM_MPC_SIZE + USER_VLLM_ENGRAM_SIZE))
export VLLM_MPI_WORKER_TEMPLATE="$SCRIPT_DIR/serve/serve_engram_mp_rpc_all_mpi_template.sh"
# The exchange uses MPI collectives only; no UCX OSC (RMA) setup is needed.
export VLLM_MPI_RUN_ARGS="${VLLM_MPI_RUN_ARGS:---allow-run-as-root --bind-to none --map-by slot}"
# Engram ranks only gather rows; keep their OpenMP footprint small.
export USER_VLLM_ENGRAM_OMP_NUM_THREADS="${USER_VLLM_ENGRAM_OMP_NUM_THREADS:-8}"
# Both the attention ranks (--load-format) and the Engram ranks (USER_VLLM_LOAD_FORMAT) fill dummy tables.
export USER_VLLM_LOAD_FORMAT="dummy"

_VLLM_OPTIONAL_ARGS+=" --all2all-backend mpi_alltoallv_v6"
_VLLM_OPTIONAL_ARGS+=" --load-format dummy"
_VLLM_OPTIONAL_ARGS+=" --block-size 128"
_VLLM_OPTIONAL_ARGS+=' --attention-config {"indexer_kv_dtype":"mxfp4"}'
_VLLM_OPTIONAL_ARGS+=' --limit-mm-per-prompt {"image":0}'
_VLLM_OPTIONAL_ARGS+=' --kernel-config {"enable_jit_warmup":false}'
_VLLM_OPTIONAL_ARGS+=' --tokenizer-mode deepseek_v41'
_VLLM_OPTIONAL_ARGS+=' --reasoning-parser deepseek_v41'
_VLLM_OPTIONAL_ARGS+=' --compilation-config {"mode":3}'
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "🚀 Preset: ${preset_dir}/${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}, Engram ranks=${USER_VLLM_ENGRAM_SIZE}"
