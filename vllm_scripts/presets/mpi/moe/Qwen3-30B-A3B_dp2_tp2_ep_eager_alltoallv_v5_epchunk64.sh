#!/bin/bash
# Qwen3-30B-A3B BF16, DP=2 TP=2 EP=4, MPI alltoallv-v5, eager, EP 接收配额 C=64。
#
# 分块条件（开启即生效，按容量判定）：C=64 < MAX_NUM_BATCHED_TOKENS(256)/TP(2)*min(K=8, E=32) = 1024，
# 且 C=64 < 每 rank 上界 128（256/SP2），chunked 路径开启。
# 预期行为（不发送视作 no next chunk）：
#   - Prefill（满批 ~128 token/rank，每 (src,dst) 对 ~115 条记录）-> 每层 2 轮，非最终轮含保活；
#   - Decode / dummy run（每 rank 仅数个 token）-> 单轮，且无记录的目的 sendcount=0，无额外发送。
SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"
export PD_MODE="MIXED"
source "$SCRIPT_DIR/user_env_template.sh"
export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=2
export USER_VLLM_MAX_NUM_BATCHED_TOKENS=256
export VLLM_XCPU_MOE_EP_MAX_RECV_TOKENS_PER_RANK=64
export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1
_VLLM_OPTIONAL_ARGS+=" --all2all-backend mpi_alltoallv_v5"
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"
echo "Preset: mpi/moe/$(basename "${BASH_SOURCE[0]}" .sh) | DP=2 TP=2 EP=4 | EP chunk C=64"
