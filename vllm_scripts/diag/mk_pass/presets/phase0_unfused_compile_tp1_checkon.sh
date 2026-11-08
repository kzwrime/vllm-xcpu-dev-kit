#!/bin/bash

# 阶段 0 机理验证:与 phase0_unfused_compile_tp1.sh 唯一差异是重新打开
# torch_xcpu 的 shape check(TORCH_XCPU_ENABLE_CHECK=1,即全局默认值;
# 原 TP1 preset 与 kit 内其他 compile preset 一样默认关掉了它)。
#
# 预测(2026-11-08 根因分析):若 DP>1 ConstraintViolationError 的根因是
#   `_check_fused_indexer_k` 的 slots.numel() <= positions.numel()
# 在 dummy-run 静态 256 行 slots 上产生下界 guard s72 >= 256,再与
# topk_indices_buffer[: query.shape[0]] 的上界 s72 <= 256 夹逼成单点
# (range_refined_to_singleton),则 TP1 + check=on 必须同样失败 ——
# 证明问题与 DP/TP 并行度无关,只取决于 check 开关。

_DIAG_DIR="$(realpath "${BASH_SOURCE[0]}")"
_DIAG_DIR="${_DIAG_DIR%/presets/*}"
source "${_DIAG_DIR%/diag*}/presets/serial/GLM-5.2-FP8-dummy-7-layer-lite_phase0_unfused_compile_tp1.sh"

export TORCH_XCPU_ENABLE_CHECK=1
export VLLM_CACHE_ROOT="${_DIAG_DIR}/cache_tp1_checkon/vllm_cache"
export TORCHINDUCTOR_CACHE_DIR="${_DIAG_DIR}/cache_tp1_checkon/inductor"
unset _DIAG_DIR

echo "Preset: diag/mk_pass/phase0_unfused_compile_tp1_checkon | TP1 with torch_xcpu checks ON," \
     "cache=${TORCHINDUCTOR_CACHE_DIR}"
