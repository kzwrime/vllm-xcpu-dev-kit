#!/bin/bash

# As _mk_v1.sh, but the v2 mega-kernel (torch_xcpu impl A2): both sub-dies compute every
# token's fused_add_rms_norm and the stage N -> stage A handoff goes through sub-die RMA
# with a per-buffer credit handshake instead of HBM plus a full-core barrier. Which of
# the two wins is a hardware measurement; on x86 this only exercises the simulation.
#
# 2026-11-08: the python-path switch selecting A0/A2 was removed with the vllm fork
# revert; see the note in _mk_v1.sh. The A1/A2 selection (2026-11-09 renumber; v1 was impl A0) will come back as a build/env
# knob of the torch_xcpu op once the compile-pass fusion re-lands.

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/GLM-5.2-FP8_dp1_tp4_ep4_compile_sparse_attn_fp8_kvcache_dflash2_5_mk_v1.sh"
