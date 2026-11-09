#!/bin/bash

# As _mk_v1.sh, but the v2 mega-kernel (torch_xcpu impl A2): both sub-dies compute every
# token's fused_add_rms_norm and the stage N -> stage A handoff goes through sub-die RMA
# with a per-buffer credit handshake instead of HBM plus a full-core barrier. Which of
# the two wins is a hardware measurement; on x86 this only exercises the simulation.

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/GLM-5.2-FP8_dp1_tp4_ep4_compile_sparse_attn_fp8_kvcache_dflash2_5_mk_v1.sh"

export TORCH_XCPU_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_IMPL="${TORCH_XCPU_FUSED_MLA_ADD_NORM_QKVA_QNORM_QB_UK_ROPE_CACHE_IMPL:-A2}"

echo "Preset: add_norm mega-kernel v2 (impl A2)"
