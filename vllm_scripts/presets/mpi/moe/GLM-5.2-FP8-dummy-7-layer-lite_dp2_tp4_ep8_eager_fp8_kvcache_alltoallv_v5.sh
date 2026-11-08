#!/bin/bash

# 7-layer dummy GLM-5.2 at DP2 x TP4 with EP8 over mpi_alltoallv_v5: the layout where the
# MoE runs sequence parallel (dp>1 + TP>1 + EP + an all2all backend that supports it), so
# every MoE layer ends with a TP all-gather. The all-gather-absorbing fused layer front
# (VLLM_XCPU_FUSED_MLA_ADD_NORM_QKVA + VLLM_XCPU_FUSED_MLA_FRONT_ABSORB_SP_ALLGATHER) was
# removed with the 2026-11-08 vllm fork revert; this preset is now the plain SP-MoE
# reference layout for the compile-pass migration (docs/upgrade/v0.31, phase 3 will
# re-land the all-gather absorption as a pattern).
# Random weights, tiny memory footprint: use it to check the operator flow and that every
# combination produces the same text, not model quality (the dummy logits are nearly flat,
# so it cannot see rounding differences). USER_VLLM_EAGER_OR_NOT="" runs it compiled.
#
# Threads: 8 ranks on the machine, so each gets a 1/8 slice (OMP_NUM_THREADS 32 on 256
# CPUs) and VLLM_XCPU_CPU_PARTITION=1 pins each rank to its slice; the mega-kernel's one
# launch still starts 96 threads (3x oversubscribed inside the slice, expected).

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export PD_MODE="MIXED"
# v0.31 auto-selects breakable CUDA graphs for DSA, which disables compile; keep it off so
# USER_VLLM_EAGER_OR_NOT="" really compiles (it is a no-op under --enforce-eager).
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-1200}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"
source "$SCRIPT_DIR/user_env_template.sh"

unset VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16
export USER_VLLM_EAGER_OR_NOT="${USER_VLLM_EAGER_OR_NOT---enforce-eager}"
export USER_VLLM_MODEL="../models/ZhipuAI/GLM-5.2-FP8-dummy-7-layer-lite"
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
_VLLM_OPTIONAL_ARGS+=" --load-format dummy"
_VLLM_OPTIONAL_ARGS+=" --kv-cache-dtype fp8"
_VLLM_OPTIONAL_ARGS+=' --kernel-config {"enable_jit_warmup":false}'
_VLLM_OPTIONAL_ARGS+=' --tool-call-parser glm47'
_VLLM_OPTIONAL_ARGS+=' --enable-auto-tool-choice'
_VLLM_OPTIONAL_ARGS+=' --reasoning-parser glm45'
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "Preset: ${preset_dir}/${preset_name} | dummy 7-layer GLM-5.2 FP8, DP2 TP4 EP8 alltoallv_v5," \
     "eager='${USER_VLLM_EAGER_OR_NOT}' OMP_NUM_THREADS=${OMP_NUM_THREADS}"
