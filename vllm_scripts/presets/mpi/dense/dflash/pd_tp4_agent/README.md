# Qwen3.8-27B + DFlash2 TP4 P/D 分离（Agent 代码开发场景）Preset 说明

本目录提供面向多用户 Coding Agent 场景的 **TP=4 Prefill / Decode** 生产对齐 preset；完整量化推导见 [qwen3.8-27b-dflash2-tp4-128rank-pd-deployment.md](../../../../../../docs/qwen3.8-27b-dflash2-tp4-128rank-pd-deployment.md)，面向已部署服务的在线校准脚本见 [calc_pd_deploy_plan.py](../../../../../tools/calc_pd_deploy_plan.py)。

## 1. KV 档位与 Preset 默认映射

| 角色 / 档位 | `max_num_batched_tokens` | `max_num_seqs` | 默认 `max_model_len` | `--block-size` | `speculative-config` |
|---|---|---|---|---|---|
| **Prefill** (`16 / 17 / 18 GiB`) | `6144` | `8` | `196608`<br>`204800`<br>`215040` | `128`（可选 `256`） | `dflash`, `K=5`,<br>`prefill_context_kv_only: true` |
| **Decode** (`16 / 17 / 18 GiB`) | `1024` | `12` / `14` / `16` | `196608`<br>`204800`<br>`215040` | `128`（须与 P 一致） | `dflash`, `K=5` |

> **关键约束**：Prefill 与 Decode **不得共享** `USER_VLLM_MAX_NUM_BATCHED_TOKENS`（P 保持 `6144` 吃满 chunked prefill 算力；D 固定 `1024` 降低激活峰值 `0.79 -> 0.13 GiB/rank` 与编译中间池并消除长 chunk 重算卡顿）；两者通过 `USER_VLLM_PREFILL_*` 与 `USER_VLLM_DECODE_*` 彻底隔离。

## 2. 环境变量覆写清单（每行一项）

```bash
# 核心档位与块大小（Prefill/Decode 必须一致）
export USER_VLLM_KVCACHE_SPACE_GB=18                    # 可选 16 | 17 | 18
export USER_VLLM_BLOCK_SIZE=128                         # 可选 128 (B=896) | 256 (B=1024)
export USER_VLLM_MAX_MODEL_LEN=215040                   # 留空则按 16->196608 / 17->204800 / 18->215040 自动匹配
export USER_VLLM_MODEL="Qwen/Qwen3.8-27B-FP8"           # BF16 权重改设为 "Qwen/Qwen3.8-27B"

# 角色隔离并发与批预算
export USER_VLLM_PREFILL_MAX_NUM_BATCHED_TOKENS=6144    # bs=128 下若需单请求满块对齐可设 6272
export USER_VLLM_PREFILL_MAX_NUM_SEQS=8
export USER_VLLM_DECODE_MAX_NUM_BATCHED_TOKENS=1024     # D 端固定 1024，不建议跟随 P 设为 6144
export USER_VLLM_DECODE_MAX_NUM_SEQS=16                 # 留空则按 16->12 / 17->14 / 18->16 自动匹配

# 可选前缀匹配细粒度控制
export USER_VLLM_PREFIX_MATCH_UNIT=128                  # 须整除 B (128/256 可选；64 为 1024 约数，注意 896%64==0)
export USER_VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0
```

## 3. 已启动服务在线校准与单组验证

针对**已经启动好的**真实 NPU OpenAI 服务接口直接在线测速与推导配比（不绑定特定启动方式）：

```bash
python vllm_scripts/tools/calc_pd_deploy_plan.py \
  --url http://<running-service-host>:<port> \
  --kv-gb 18 \
  --block-size 128 \
  --max-output-tokens 32768 \
  --tpot-slo-ms 25.0 \
  --store-cap-tib 2.0 \
  --store-bw-gibs 10.0
```

也可以分离传入 `--prefill-url` / `--decode-url`，或省略 URL 进入离线公式计算模式（详见 `--help`）。

若需在套件内跑单组 `1P x 1D` 验证（`run_vllm_test.sh --pd` + `vllm-router`）：

```bash
cd vllm_scripts
D=presets/mpi/dense/dflash/pd_tp4_agent
USER_VLLM_KVCACHE_SPACE_GB=18 \
USER_VLLM_BLOCK_SIZE=128 \
USER_VLLM_PD_KV_BACKEND=mooncake \
USER_VLLM_PD_MOONCAKE_PROTOCOL=rdma \
USER_VLLM_PD_PREFILL_PRESET=$D/Qwen3.8-27B-FP8_dp1_tp4_compile_dflash2_5_prefill.sh \
USER_VLLM_PD_DECODE_PRESET=$D/Qwen3.8-27B-FP8_dp1_tp4_compile_dflash2_5_decode.sh \
./run_vllm_test.sh --pd --multi-test -e $D/Qwen3.8-27B-FP8_dp1_tp4_compile_dflash2_5_decode.sh
```
