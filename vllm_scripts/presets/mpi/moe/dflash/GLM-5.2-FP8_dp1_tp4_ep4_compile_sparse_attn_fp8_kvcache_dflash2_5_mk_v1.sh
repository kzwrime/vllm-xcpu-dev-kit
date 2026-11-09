#!/bin/bash

# GLM-5.2-FP8_dp1_tp4_ep4_compile_sparse_attn_fp8_kvcache_dflash2_5 with the fused layer
# front (fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache, v1 / impl A1 (2026-11-09 重编号, 旧 A0)) enabled:
# input_layernorm + fused_qkv_a_proj + q/kv norm + q_b_proj + W_UK_T + RoPE + KV-cache
# write as one torch_xcpu op. Default is off; the companion _mk_v2.sh picks v2 (impl A2).
#
# Threads: the mega-kernel's PE count is a build-time constant (mk::arch::kNumPe = 96),
# so that one launch always starts 96 OpenMP threads, while TP4 on a 256-thread machine
# gives each rank 64 (OMP_NUM_THREADS). That is deliberate oversubscription. The settings
# below keep it contained:
#   VLLM_XCPU_CPU_PARTITION=1  each rank runs on its own CPUs/SIZE slice, so the 96
#                              threads oversubscribe that slice 1.5x instead of fighting
#                              the other ranks (serve/serve_mp_rpc_all_mpi_template.sh)
#   KMP_BLOCKTIME=1            Intel OpenMP spins 200 ms by default after a parallel
#                              region; with a 96-thread team in a 64-CPU slice those
#                              idle spinners would steal a third of the slice from the
#                              next (64-thread) kernel. Raise it if wake-up latency
#                              dominates instead.
#   OMP_THREAD_LIMIT           must stay unset / >= 96: a smaller team makes the op fall
#                              back to the unfused sequence (it says so in the log).
# The operator never aborts on a short team; it runs the per-operator sequence instead,
# so a mis-set thread budget shows up as "disabled" in the warning, not as a crash.
#
# 2026-11-08: the python-path switch VLLM_XCPU_FUSED_MLA_ADD_NORM_QKVA (and the vllm fork
# plumbing it needed) was removed; this preset now degrades to the unfused sequence until
# the fusion is re-landed as a torch.compile pattern pass (docs/upgrade/v0.31, phase 2).
# The thread settings below stay: they apply to the mega-kernel the pass will emit.

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/GLM-5.2-FP8_dp1_tp4_ep4_compile_sparse_attn_fp8_kvcache_dflash2_5.sh"

export VLLM_XCPU_CPU_PARTITION="${VLLM_XCPU_CPU_PARTITION:-1}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-1}"
unset OMP_THREAD_LIMIT

echo "Preset: fused front OFF (python-path switch removed, pass migration pending)" \
     "(cpu_partition=${VLLM_XCPU_CPU_PARTITION}, OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}," \
     "KMP_BLOCKTIME=${KMP_BLOCKTIME})"
