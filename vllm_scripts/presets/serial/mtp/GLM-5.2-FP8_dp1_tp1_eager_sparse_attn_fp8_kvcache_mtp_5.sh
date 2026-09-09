#!/bin/bash
# GLM-5.2 real-weight sparse MLA: prefix caching plus five MTP draft tokens.
_GLM_MTP_PRESET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_GLM_MTP_PRESET_DIR}/../GLM-5.2-FP8_dp1_tp1_eager_sparse_attn_fp8_kvcache.sh"
export VLLM_OPTIONAL_ARGS="${VLLM_OPTIONAL_ARGS}"' --speculative-config {"method":"mtp","num_speculative_tokens":5}'
unset _GLM_MTP_PRESET_DIR
