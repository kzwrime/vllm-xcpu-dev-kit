#!/bin/bash
# Preset: Qwen3.8-27B-FP8 + DFlash2, P/D 分离（Prefill 角色, TP4 Agent 生产配置）
# Configuration: DP=1, TP=4, PP=1, compile mode, block_size=128/256, chunk=6144
#
# 与同目录 Decode preset 配对使用，见同目录 README.md 与 docs/qwen3.8-27b-dflash2-tp4-128rank-pd-deployment.md。
# 关键要点：
#   1. vllm 0.31 已合入 #59468 与 #54163，Prefill 保持 max_num_batched_tokens=6144（无需 6144+K）；
#      通过 USER_VLLM_PREFILL_MAX_NUM_BATCHED_TOKENS / USER_VLLM_PREFILL_MAX_NUM_SEQS 隔离 Decode 覆写。
#   2. Prefill 启用 prefill_context_kv_only=true（#65217/#67291）：保持 15 个 KV Group 与 D 对齐，
#      仅计算并传输 drafter context KV，释放非 context KV 所需的 draft 权重并跳过首次 propose。
#   3. 默认 --block-size 128（B=896，可通过 USER_VLLM_BLOCK_SIZE=256 切换为 B=1024），按 KV 容量档位
#      16/17/18 GiB 自动映射最优 max_model_len（196608 / 204800 / 215040），并剥离默认开启的 --profiler-config。

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"
_CALLER_USER_VLLM_MODEL="${USER_VLLM_MODEL:-}"
_CALLER_USER_VLLM_MAX_MODEL_LEN="${USER_VLLM_MAX_MODEL_LEN:-}"
_KV_SPACE_GB="${USER_VLLM_KVCACHE_SPACE_GB:-18}"
_BLOCK_SIZE="${USER_VLLM_BLOCK_SIZE:-128}"
if [ -n "${_CALLER_USER_VLLM_MAX_MODEL_LEN}" ]; then
    _DEFAULT_MAX_LEN="${_CALLER_USER_VLLM_MAX_MODEL_LEN}"
elif [ "${_KV_SPACE_GB}" -ge 18 ]; then
    _DEFAULT_MAX_LEN=215040
elif [ "${_KV_SPACE_GB}" -eq 17 ]; then
    _DEFAULT_MAX_LEN=204800
else
    _DEFAULT_MAX_LEN=196608
fi

export PD_MODE="NOT_MOE"
export USER_VLLM_MAX_NUM_BATCHED_TOKENS="${USER_VLLM_PREFILL_MAX_NUM_BATCHED_TOKENS:-6144}"
export USER_VLLM_MAX_NUM_SEQS="${USER_VLLM_PREFILL_MAX_NUM_SEQS:-8}"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-1800}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"

source "$SCRIPT_DIR/user_env_template.sh"

# 必须放在 source user_env_template.sh 之后，避免被模板硬编码默认值覆盖
export USER_VLLM_MODEL="${_CALLER_USER_VLLM_MODEL:-Qwen/Qwen3.8-27B-FP8}"
export USER_VLLM_MAX_MODEL_LEN="${_DEFAULT_MAX_LEN}"
export VLLM_CPU_KVCACHE_SPACE="${_KV_SPACE_GB}"
unset _CALLER_USER_VLLM_MODEL _CALLER_USER_VLLM_MAX_MODEL_LEN _KV_SPACE_GB _DEFAULT_MAX_LEN

export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=4
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1
export TORCH_XCPU_ENABLE_CHECK=0

if [ "${USER_VLLM_ENABLE_PROFILER:-0}" != "1" ]; then
    _VLLM_OPTIONAL_ARGS="$(printf '%s' "${_VLLM_OPTIONAL_ARGS}" | sed -E 's/[[:space:]]--profiler-config[[:space:]]+[^[:space:]]+//g')"
fi

_VLLM_OPTIONAL_ARGS+=" --skip-mm-profiling --reasoning-parser qwen3 --language-model-only --block-size ${_BLOCK_SIZE}"
unset _BLOCK_SIZE
if [ -n "${USER_VLLM_PREFIX_MATCH_UNIT:-}" ]; then
    _VLLM_OPTIONAL_ARGS+=" --prefix-match-unit ${USER_VLLM_PREFIX_MATCH_UNIT}"
fi
if [ -n "${USER_VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-}" ]; then
    _VLLM_OPTIONAL_ARGS+=" --prefix-cache-retention-interval ${USER_VLLM_PREFIX_CACHE_RETENTION_INTERVAL}"
fi
_VLLM_OPTIONAL_ARGS+=' --speculative-config {"model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":5,"method":"dflash","prefill_context_kv_only":true}'
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "🚀 Preset: ${preset_dir}/${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}, MAX_LEN=${USER_VLLM_MAX_MODEL_LEN}, KV=${VLLM_CPU_KVCACHE_SPACE}GiB, BATCHED=${USER_VLLM_MAX_NUM_BATCHED_TOKENS}, SEQS=${USER_VLLM_MAX_NUM_SEQS}"
