#!/bin/bash

# 阶段 0 快速复现载体:GLM-5.2-FP8 dummy 7 层,DP1 x TP1 单进程(无 MPI),
# compile 模式,MLA 侧融合全关。用于定位 dynamo ConstraintViolationError 的
# 真实特化点(TORCH_LOGS=+dynamic 可直接注入本 preset)与获取编译产物。
# 单进程迭代一次 ~1-2 分钟,远快于 8 rank MPI 的 dp2tp4ep8 preset。

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
# 关键:serve 模板 source 本 preset 后仍会用 $SCRIPT_DIR 定位 mpi_tools,
# 必须把 SCRIPT_DIR 重置回 vllm_scripts 根目录(与标准 preset 一致)
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export PD_MODE="NOT_MOE"
export USER_VLLM_MAX_NUM_BATCHED_TOKENS="${USER_VLLM_MAX_NUM_BATCHED_TOKENS:-256}"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-600}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-600}"
source "$SCRIPT_DIR/user_env_template.sh"

unset USER_VLLM_EAGER_OR_NOT
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export TORCH_XCPU_ENABLE_CHECK=0
export USER_VLLM_MODEL="$SCRIPT_DIR/../models/ZhipuAI/GLM-5.2-FP8-dummy-7-layer-lite"
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=1
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=1
export TORCHINDUCTOR_CPP_WRAPPER=1
unset VLLM_DISABLE_TQDM_AND_MONITOR

# MLA 侧融合全关:未融合上游序列(ADD_NORM_QKVA 开关已随 2026-11-08 回退移除)
export VLLM_XCPU_FUSED_MLA=0
export VLLM_XCPU_FUSED_MLA_VPROJ=0

_VLLM_OPTIONAL_ARGS+=" --load-format dummy"
_VLLM_OPTIONAL_ARGS+=" --kv-cache-dtype fp8"
# GLM-5.2/DSv4 在 XCPU 上走自有 indexer 路径,不需要 triton JIT warmup
# (与 kit 内其他 GLM-5.2 preset 一致);开 warmup 会触发未适配的
# PackSeq/UnpackSeq kernel 注册,fake-triton adapter 会给出本提示。
_VLLM_OPTIONAL_ARGS+=' --kernel-config {"enable_jit_warmup":false}'
_VLLM_OPTIONAL_ARGS+=' --tool-call-parser glm47'
_VLLM_OPTIONAL_ARGS+=' --enable-auto-tool-choice'
_VLLM_OPTIONAL_ARGS+=' --reasoning-parser glm45'
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

# 独立 compile cache
export VLLM_CACHE_ROOT="$SCRIPT_DIR/cache_tp1/vllm_cache"
export TORCHINDUCTOR_CACHE_DIR="$SCRIPT_DIR/cache_tp1/inductor"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
echo "Preset: presets/serial/${preset_name} | unfused MLA, compile, DP1 TP1," \
     "cache=${TORCHINDUCTOR_CACHE_DIR}"
