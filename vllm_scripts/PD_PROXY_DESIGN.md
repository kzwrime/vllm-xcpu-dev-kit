# P/D Router Integration Design (`vllm-router`)

本文说明当前 P/D 分离路由架构的实现与启动原理。本套件已将原有的 Python 本地测试代理 `pd_proxy.py` 及 `ExampleConnector` 共享目录方案正式替换为官方高性能 Rust 路由 **`vllm-router`**（`--vllm-pd-disaggregation --kv-connector mooncake`），支持两类 Mooncake KV backend：

- `mooncake`（默认）：使用 vLLM `MooncakeConnector` 与 Mooncake Transfer Engine（`tcp` / `rdma`）执行 P→D 点对点 KV 传输。
- `mooncake_multi`：使用 `MultiConnector` 组合 `MooncakeConnector`（P→D 直传）与 `MooncakeStoreConnector`（共享分布式 CPU DRAM / SSD KV pool，需配置 `MOONCAKE_CONFIG_PATH`）。

相关文件：

- `vllm_scripts/pd_launcher.sh`：P/D 拓扑与 `vllm-router` 启动器。
- `vllm_scripts/launcher_common.sh`：通用服务管理、错误监控与进程组清理。
- `vllm_scripts/RUNBOOK.md`：运行与调试入口说明。
- `docs/qwen3.8-27b-dflash2-tp4-128rank-pd-deployment.md`：128 Rank NPU `Qwen3.8-27B + DFlash2 TP4` 生产部署推导。

## 总体结构

P/D 分离由独立的 Prefill vLLM 实例池、Decode vLLM 实例池，以及前端状态感知路由 `vllm-router` 组成：

```text
client
  |
  v
vllm-router (--vllm-pd-disaggregation --kv-connector mooncake)
  |-- HTTP /v1/models, /v1/chat/completions, /v1/completions, /v1/responses
  |-- 启动阶段: 轮询 Worker /health、/metrics (发现 DP size)、Mooncake Bootstrap /query (获取 DP rank -> engine_id)
  |
  +--> prefill vLLM instance(s)  [kv_producer + MooncakeBootstrapServer]
  |        |
  |        +--- Mooncake TransferEngine (P->D push, RDMA / TCP) ---> decode vLLM instance(s)
  |        |                                                                  ^
  |        +--- (可选 mooncake_multi) MooncakeStoreConnector <----------------+
  v
decode  vLLM instance(s)         [kv_consumer]
```

工作流语义：

1. 客户端向 `vllm-router` 发送普通 OpenAI API 请求（无需携带任何 P/D 参数）。
2. `vllm-router` 按 `--prefill-policy` / `--decode-policy`（默认均继承 `--policy cache_aware`）选定目标 Prefill 与 Decode worker（含 `@dp_rank`）。
3. `vllm-router` 向选定 Prefill 实例发送 `stream=false`、`max_tokens=1` 的请求，注入 `X-Request-Id`、`X-data-parallel-rank`（DP>1 时）与 `kv_transfer_params`（`do_remote_decode=true, do_remote_prefill=false, transfer_id="xfer-<uuid>"`）。
4. Prefill 完成后，`vllm-router` 向 Decode 实例转发用户原始请求，并注入 `kv_transfer_params`（`do_remote_decode=false, do_remote_prefill=true, remote_bootstrap_addr, remote_engine_id, transfer_id`）。
5. Decode 侧通过 ZMQ side-channel 向 Prefill `MooncakeBootstrapServer` 查询 worker 地址并触发 Mooncake `batch_transfer_sync_write` 拉取/写入 KV；`vllm-router` 将 Decode 输出（流式 SSE 或非流式 JSON，自动合并 usage / cached_tokens / logprobs）返回给客户端。

## Backend 与环境变量选择

默认启用 `mooncake` backend + `vllm-router`：

```bash
./run_vllm_test.sh --pd -e <preset.sh>
```

切换到 `mooncake_multi`（`MooncakeConnector` + `MooncakeStoreConnector`）：

```bash
MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json \
USER_VLLM_PD_KV_BACKEND=mooncake_multi \
USER_VLLM_PD_MOONCAKE_PROTOCOL=rdma \
./run_vllm_test.sh --pd -e <preset.sh>
```

相关可调环境变量：

```text
# KV backend 与 Mooncake 参数
USER_VLLM_PD_KV_BACKEND=mooncake|mooncake_multi   # 默认 mooncake
USER_VLLM_PD_MOONCAKE_PROTOCOL=tcp|rdma           # 默认 tcp（真实 RDMA 环境设 rdma）
USER_VLLM_PD_MOONCAKE_NUM_WORKERS=4               # 默认 4
USER_VLLM_PD_MOONCAKE_DEVICE_NAME=                # 可选 RDMA 网卡白名单，如 "mlx5_0,mlx5_1"
USER_VLLM_PD_PREFILL_BOOTSTRAP_PORT=18998         # 默认从 18998 起，实例间 stride 100
USER_VLLM_PD_MOONCAKE_CONFIG_PATH=                # mooncake_multi 下的配置文件路径（默认读 MOONCAKE_CONFIG_PATH）
USER_VLLM_PD_STORE_LOOKUP_ASYNC=true              # mooncake_multi P 侧异步 lookup
USER_VLLM_PD_SAVE_DECODE_CACHE=true               # mooncake_multi D 侧回写 decode prefix block

# 拓扑与角色预设
USER_VLLM_PD_PREFILL_COUNT=1                      # Prefill 实例数
USER_VLLM_PD_DECODE_COUNT=1                       # Decode 实例数
USER_VLLM_PD_PREFILL_PRESET=<preset.sh>           # 可选：Prefill 专用 preset
USER_VLLM_PD_DECODE_PRESET=<preset.sh>            # 可选：Decode 专用 preset

# vllm-router 参数
VLLM_ROUTER_BIN=vllm-router                       # 可选自定义 router 可执行文件路径
USER_VLLM_PD_ROUTER_HOST=127.0.0.1                # 默认 127.0.0.1
USER_VLLM_PD_ROUTER_PORT=                         # 默认取空闲 USER_VLLM_PORT（兼容 USER_VLLM_PD_PROXY_PORT）
USER_VLLM_PD_ROUTER_POLICY=cache_aware            # 默认 cache_aware；可选 random|round_robin|cache_aware|kv_aware|power_of_two|consistent_hash|sticky_least_loaded
USER_VLLM_PD_PREFILL_POLICY=                      # 可选覆盖 Prefill 路由策略
USER_VLLM_PD_DECODE_POLICY=                       # 可选覆盖 Decode 路由策略
USER_VLLM_PD_INTRA_NODE_DP_SIZE=                  # 可选显式 DP 大小（默认 router 自动从 /metrics 发现）
USER_VLLM_PD_ROUTER_TIMEOUT_SECS=21600            # 请求超时，默认 21600
USER_VLLM_PD_ROUTER_HEALTH_TIMEOUT_SECS=30        # 健康检查超时，默认 30
USER_VLLM_PD_ROUTER_PROMETHEUS_HOST=127.0.0.1     # Prometheus 指标绑定地址
USER_VLLM_PD_ROUTER_PROMETHEUS_PORT=              # 默认自动查找 29000 起空闲端口
USER_VLLM_PD_ROUTER_EXTRA_ARGS=                   # 额外透传给 vllm-router 的 CLI 参数
```

> [!IMPORTANT]
> **DFlash / DSpark 与混合模型对齐要求**：
> - P 和 D 两边的 KV 布局必须一致：模型、`--block-size`、prefix caching / mamba 模式、以及 `speculative-config`。
> - P 不带 drafter 而 D 带 DFlash 会在 Mooncake 对齐 KV 区域时报 `registered group index mismatch`，且 D 的 drafter context KV 无法由 P 传来。
> - 当前 `vllm` 0.31 已合入 `#59468` / `#54163` / `#65217` / `#64109`：Prefill 无需再为 drafter 额外留出 `+K` 预算（设 `max_num_batched_tokens=6144` 即可），而 Decode 可独立收敛至 `1024` 以降低激活峰值与编译内存池占用；同时 Prefill 推荐在 `--speculative-config` 中开启 `"prefill_context_kv_only": true`，既保持 15 个 KV Group 对齐，又跳过无效 `propose` 并释放多余 draft 权重。详见 `docs/qwen3.8-27b-dflash2-tp4-128rank-pd-deployment.md`、`docs/dev_log/20261029_Qwen3.8-DFlash2-PD-prefill-chunk.md` 与 `docs/release_notes/20261118 更新说明.md`。

## 启动流程与目录布局

`pd_launcher.sh` 中的 `pd_start()` 会创建如下运行目录：

```text
logs/pd/<run-id>/
  prefill0.sh
  decode0.sh
  router.sh
  proxy.sh -> router.sh   # 兼容软链
  prefill0/
    logs/
  decode0/
    logs/
  router/
    router.pid
    router.log
```

`pd_write_role_preset()` 为每个实例生成 overlay preset：

- 隔离端口：`USER_VLLM_PORT`、`USER_VLLM_DATA_PARALLEL_RPC_PORT`、`VLLM_MPI_COORD_PORT`、`VLLM_MPI_ENV_EXPORT_FILE`、`VLLM_MP_RPC_READY_PORT_BASE`。
- 注入 `--kv-transfer-config` 与 `--no-disable-hybrid-kv-cache-manager`。
- 为每个 Prefill 实例分配并导出独立 `VLLM_MOONCAKE_BOOTSTRAP_PORT`（默认 `18998 + index * 100`）。

`pd_start_router()` 在所有 Prefill / Decode 实例通过 `/v1/models` 就绪检查后启动 `vllm-router`：

```bash
vllm-router \
  --host 127.0.0.1 \
  --port <router-port> \
  --vllm-pd-disaggregation \
  --kv-connector mooncake \
  --policy cache_aware \
  --prometheus-host 127.0.0.1 \
  --prometheus-port <prom-port> \
  --prefill http://127.0.0.1:<p0-port> <p0-bootstrap-port> \
  --prefill http://127.0.0.1:<p1-port> <p1-bootstrap-port> \
  --decode http://127.0.0.1:<d0-port>
```

> [!NOTE]
> `vllm-router` 的 `--prefill` 与 `--decode` 为可重复参数（不支持逗号拼接 URL）：每个 Prefill 实例传入 `--prefill <URL> <BOOTSTRAP_PORT>` 两个参数，每个 Decode 实例传入 `--decode <URL>`。同时，当某侧实例数小于 2（如默认 `1P x 1D`）时，`power_of_two` 策略会被 `vllm-router` 配置校验拒绝，因此默认策略采用 `cache_aware`。

## Mooncake Transfer 验证

启动器中的 `pd_validate_transfer()` 会在 `prefill*/logs` 与 `decode*/logs` 中检索以下任一标志：

```text
Receiving Mooncake KV
pulling kv_caches
MooncakeXferMetadata
Sending to .* done
```

若启用 `vllm-router` Prometheus 指标，可同时访问 `http://<prom-host>:<prom-port>/metrics` 观测 P/D 请求分布、错误计数与阶段耗时。
