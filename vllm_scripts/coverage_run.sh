#!/bin/bash

set -x

USER_VLLM_MAX_NUM_BATCHED_TOKENS=32 ./run_vllm_test.sh -e presets/serial/Qwen3.5-0.8B_dp1_tp1_eager.sh --coverage
USER_VLLM_MAX_NUM_BATCHED_TOKENS=32 ./run_vllm_test.sh -e presets/mpi/dense/Qwen3.5-0.8B_dp1_tp2_eager.sh --coverage
USER_VLLM_MAX_NUM_BATCHED_TOKENS=32 ./run_vllm_test.sh -e presets/serial/Qwen3.5-4B_dp1_tp1_eager.sh --coverage
USER_VLLM_MAX_NUM_BATCHED_TOKENS=32 ./run_vllm_test.sh -e presets/mpi/dense/Qwen3.5-4B_dp1_tp2_eager.sh --coverage
VLLM_TEST_MAX_WAIT=1000 USER_VLLM_MAX_NUM_BATCHED_TOKENS=32 ./run_vllm_test.sh -e presets/serial/Qwen3.6-35B-A3B_dp1_tp1_eager.sh --coverage
VLLM_TEST_MAX_WAIT=1000 USER_VLLM_MAX_NUM_BATCHED_TOKENS=32 ./run_vllm_test.sh -e presets/mpi/moe/Qwen3.6-35B-A3B_dp1_tp2_eager.sh --coverage
VLLM_TEST_MAX_WAIT=1000 USER_VLLM_MAX_NUM_BATCHED_TOKENS=32 ./run_vllm_test.sh -e presets/mpi/moe/Qwen3.6-35B-A3B_dp1_tp4_eager.sh --coverage

