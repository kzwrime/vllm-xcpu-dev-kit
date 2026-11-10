#!/bin/bash

# As _mk_v1.sh, but the SHARED layers' front window runs the shared op's v2
# mega-kernel slot (torch_xcpu impl A2；2026-11-10 解耦后 v2 恢复为不带 indexer 的段).
# full 层的 sparse indexer 投影走独立的 fused_mla_add_norm_qkva_wk_wq 算子，
# 其 IMPL env 保持 _mk_v1.sh 里的 A1（它只有 A1 一个 mega-kernel 段）。

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/GLM-5.2-FP8_dp1_tp4_ep4_compile_sparse_attn_fp8_kvcache_dflash2_5_mk_v1.sh"

# mk_v1 上面已经 export A1，${VAR:-A2} 只会保留 A1 —— 必须无条件覆盖成 A2。
export TORCH_XCPU_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_IMPL=A2

echo "Preset: shared 层 add_norm mega-kernel v2 (impl A2, 不带 indexer), full 层 wk_wq op = A1"
