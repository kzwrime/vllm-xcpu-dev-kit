#!/bin/bash
# Shared source tree, venv, and checkpoint paths must exist on every host.
# A0/A1 run TP2; F0/F1 run EP2, for four MPI ranks total.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VLLM_MPI_HOSTFILE="$(realpath -e "${VLLM_MPI_HOSTFILE:-$SCRIPT_DIR/mpi_tools/afd_tp2_ep2.hostfile}")"
export VLLM_MPI_HOSTFILE

cd "$SCRIPT_DIR"

export USER_VLLM_DATA_PARALLEL_RPC_IP="${USER_VLLM_DATA_PARALLEL_RPC_IP:-$(hostname -I | awk '{print $1}')}"

# Open MPI 4.x pt2pt OSC cannot serve MPI_THREAD_MULTIPLE; use UCX over TCP.
export UCX_TLS="${UCX_TLS:-tcp,self}"
export VLLM_MPI_RUN_ARGS="${VLLM_MPI_RUN_ARGS:---allow-run-as-root --bind-to none --map-by slot --mca osc ucx --mca osc_ucx_tls any --mca osc_ucx_devices any --mca btl self,tcp}"

export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-2000}"
for preset in \
    Qwen3.6-35B-A3B_dp2_tp2_af_ep_v7 \
    Qwen3-30B-A3B_dp1_tp2_af_ep_v7 \
    Qwen3-30B-A3B-FP8_dp1_tp2_af_ep_v7 \
    Qwen3-30B-A3B-MXFP4A16_dp1_tp2_af_ep_v7; do
    ./run_vllm_test.sh -e "presets/mpi/moe/${preset}.sh" \
        --multi-test --multi-test-temperature 0 --test-timeout 300 "$@"
done
