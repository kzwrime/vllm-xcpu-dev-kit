# vLLM Scripts Runbook

本文档说明 `vllm_scripts` 的统一入口、底层等价命令和调试拆解方式。

## 统一入口

对外统一使用：

```bash
cd vllm_scripts
./run_vllm_test.sh -e <preset.sh> [options]
```

常用选项：

```bash
# 常规启动 + 单请求正确性测试
./run_vllm_test.sh -e presets/serial/Qwen3-0.6B_dp1_tp1_eager.sh

# 只启动，不测试；用于手动调试
./run_vllm_test.sh --no-test -e presets/serial/Qwen3-0.6B_dp1_tp1_eager.sh --launcher mp

# 目标端口被占用时，显式允许自动选择后续空闲端口
./run_vllm_test.sh -e presets/serial/Qwen3-0.6B_dp1_tp1_eager.sh --launcher mp --auto-port

# MPI 启动 + 测试
./run_vllm_test.sh -e presets/mpi/moe/Qwen3-30B-A3B_dp2_tp2_ep_eager_alltoallv_v2.sh

# P/D 分离 + 并发测试
./run_vllm_test.sh --pd --multi-test -e presets/mpi/moe/Qwen3-30B-A3B_dp2_tp2_ep_eager_alltoallv_v2.sh
```

`run_vllm_test.sh` 的职责是编排：加载 preset、启动服务、等待就绪、执行测试、归档日志、清理进程。真正的底层启动仍然由 `serve/` 和 `serve_test/` 下的脚本完成。

## 端口与就绪判定

非 P/D 的 `mp`/`mpi` wrapper 默认使用严格端口策略：

```bash
./run_vllm_test.sh -e <preset.sh> --launcher mp
./run_vllm_test.sh -e <preset.sh> --launcher mpi
```

如果本次运行需要使用的端口已被占用，脚本会在启动 vLLM 前失败返回非 0，而不是复用端口上的旧服务。这可以避免请求误打到历史残留服务，并出现类似当前模型不存在的 404：

```json
{"error":{"message":"The model `<model>` does not exist.","type":"NotFoundError","code":404}}
```

非 P/D `mp` 会检查：

```text
USER_VLLM_PORT
```

非 P/D `mpi` 会检查：

```text
USER_VLLM_PORT
USER_VLLM_DATA_PARALLEL_RPC_PORT
VLLM_MPI_COORD_PORT            # 仅 VLLM_USE_MPI_COORD=1 时
VLLM_MP_RPC_READY_PORT_BASE    # 按 MPI_COUNT + 4 检查连续端口段
```

确实希望开发调试时自动避开占用端口，需要显式开启：

```bash
./run_vllm_test.sh -e <preset.sh> --launcher mp --auto-port

# 等价环境变量
RUN_VLLM_TEST_AUTO_PORT=1 ./run_vllm_test.sh -e <preset.sh> --launcher mp
```

`--auto-port` 会从各自配置的端口开始向后查找空闲端口，并避免本次运行内的端口段互相重叠。实际端口会写入运行时 override：

```text
RUN_VLLM_EFFECTIVE_PORT
RUN_VLLM_EFFECTIVE_DATA_PARALLEL_RPC_PORT
RUN_VLLM_EFFECTIVE_MPI_COORD_PORT
RUN_VLLM_EFFECTIVE_MP_RPC_READY_PORT_BASE
```

因此后续 `serve/` 和 `serve_test/` 下的脚本即使重新加载 preset，也会继续使用 wrapper 选择的实际端口。

就绪判定分两层：

1. 先等待启动日志出现服务启动成功信号。
2. 再访问 `GET /v1/models`，并要求返回结果中包含当前 `USER_VLLM_MODEL`。

因此端口上即使已有一个可访问的 OpenAI API 服务，只要模型 id 不匹配，也不会被当成本次启动成功。

请求测试使用 `curl --fail-with-body`。HTTP 4xx/5xx 会导致测试失败返回非 0，同时保留响应体便于排查。

P/D 模式不使用上述非 P/D 端口预检查；它由 `pd_launcher.sh` 为 prefill、
decode、proxy、DP RPC、MPI coord、MP RPC ready 等端口分别寻找空闲端口。

## 非 P/D：mp 模式展开

统一入口：

```bash
./run_vllm_test.sh -e presets/serial/Qwen3-0.6B_dp1_tp1_eager.sh --launcher mp
```

本质上等价于两个终端：

```bash
# 终端 1：启动 vLLM
cd vllm_scripts
./serve/serve_mp_template.sh -e presets/serial/Qwen3-0.6B_dp1_tp1_eager.sh
```

```bash
# 终端 2：测试
cd vllm_scripts
./serve_test/serve_test_template.sh -e presets/serial/Qwen3-0.6B_dp1_tp1_eager.sh
```

如果 mp 模式出问题，优先直接用这两条命令复现，避免先排查 wrapper。

## 非 P/D：mpi 模式展开

统一入口：

```bash
./run_vllm_test.sh -e presets/mpi/moe/Qwen3-30B-A3B_dp2_tp2_ep_eager_alltoallv_v2.sh --launcher mpi
```

本质上等价于三个终端：

```bash
# 终端 1：启动 API head
cd vllm_scripts
./serve/serve_head_only_template.sh -e presets/mpi/moe/Qwen3-30B-A3B_dp2_tp2_ep_eager_alltoallv_v2.sh
```

```bash
# 终端 2：启动 MPI worker
cd vllm_scripts
mpirun --bind-to none --map-by slot -np 4 \
  bash ./serve/serve_mp_rpc_all_mpi_template.sh \
  -e presets/mpi/moe/Qwen3-30B-A3B_dp2_tp2_ep_eager_alltoallv_v2.sh
```

```bash
# 终端 3：测试
cd vllm_scripts
./serve_test/serve_test_template.sh -e presets/mpi/moe/Qwen3-30B-A3B_dp2_tp2_ep_eager_alltoallv_v2.sh
```

其中 `-np 4` 来自：

```bash
USER_VLLM_DATA_PARALLEL_SIZE * USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE
```

对于 `Qwen3-30B-A3B_dp2_tp2_ep_eager_alltoallv_v2.sh`，即 `2 * 2 * 1 = 4`。

## 只启动、不测试

仍然使用统一入口：

```bash
./run_vllm_test.sh --no-test -e <preset.sh> --launcher mp
./run_vllm_test.sh --no-test -e <preset.sh> --launcher mpi
./run_vllm_test.sh --no-test --pd -e <preset.sh>
```

`--no-test` 会在服务就绪后保持前台运行。另开终端手动测试：

```bash
cd vllm_scripts
./serve_test/serve_test_template.sh -e <preset.sh>
```

P/D 模式下，测试应使用启动日志中打印的 proxy preset，例如：

```bash
./serve_test/serve_test_template.sh -e logs/pd/<run-id>/proxy.sh
```

## P/D 分离模式展开

统一入口：

```bash
./run_vllm_test.sh --pd -e presets/mpi/moe/Qwen3-30B-A3B_dp2_tp2_ep_eager_alltoallv_v2.sh --multi-test
```

P/D 分离不是单个 vLLM 进程，而是：

```text
prefill vLLM instance(s)
decode  vLLM instance(s)
local P/D proxy
test client
```

启动器会在 `logs/pd/<run-id>/` 下生成临时 overlay preset：

```text
prefill0.sh
decode0.sh
proxy.sh
prefill0/
decode0/
proxy/
example_connector_storage/
```

每个 prefill/decode overlay preset 都会：

```bash
source <original-preset>
export USER_VLLM_PORT=<role-http-port>
export USER_VLLM_DATA_PARALLEL_RPC_PORT=<role-dp-rpc-port>
export VLLM_MPI_COORD_PORT=<role-mpi-coord-port>
export VLLM_MPI_ENV_EXPORT_FILE=<role-dir>/vllm_mpi_env_server.sh
export VLLM_MP_RPC_READY_PORT_BASE=<role-ready-port-base>
export VLLM_OPTIONAL_ARGS="${VLLM_OPTIONAL_ARGS} --kv-transfer-config <ExampleConnector-config> --no-disable-hybrid-kv-cache-manager"
```

注意：`VLLM_PD_ROLE`、`VLLM_PD_ROLE_INDEX` 这类变量没有实际消费者，不应作为功能依赖。P/D 的关键配置是 `--kv-transfer-config` 中的 `shared_storage_path`。

LMCache 作为 P/D KV backend 的调研和分层落地建议见
`PD_LMCACHE_INVESTIGATION.md`。当前默认 backend 仍是 `ExampleConnector`。
Mooncake 直接 P/D 传输路线见 `PD_MOONCAKE_INVESTIGATION.md`。
Mooncake CPU/SSD 持久化和 standalone store 拓扑见
`PD_MOONCAKE_PERSISTENCE.md`。该文档的“配置出处”部分列出了官方 vLLM
`KVTransferConfig` API、MooncakeStoreConnector guide、上游
`deps/vllm-main` 实现和 Mooncake master/client 源码对应关系。

使用 Mooncake non-CUDA/TCP backend：

```bash
USER_VLLM_PD_KV_BACKEND=mooncake \
USER_VLLM_PD_MOONCAKE_PROTOCOL=tcp \
./run_vllm_test.sh --pd -e <preset.sh>
```

注意：这个入口使用的是 `MooncakeConnector`，只验证 prefill/decode 之间的
点对点 KV transfer。它不启动 Mooncake Store，也不提供纯 CPU 内存池或 SSD
持久化。要让 prefill/decode 以外的节点作为 CPU/SSD KV pool，需要
`MooncakeStoreConnector` + `mooncake_master` + 外部 `mooncake_client`
的 `standalone-store` 拓扑；当前本地 `vllm/` 尚未移植该 connector。

当前已通过 `Qwen3-0.6B` 单请求、`Qwen3-0.6B --multi-test` 并发请求、
以及 `Qwen3-30B-A3B dp2/tp2/ep` MPI/MoE 单请求。`Qwen3.5-0.8B`
hybrid full-attention + linear-attention 仍需要 group-aware Mooncake metadata，
不能只把 HMA 多组 block ids 展平。

### P/D + mpi 的手工展开

以 `1P x 1D` 为例，启动日志会打印实际路径和端口。假设生成：

```text
logs/pd/<run-id>/prefill0.sh
logs/pd/<run-id>/decode0.sh
logs/pd/<run-id>/proxy.sh
```

则可以拆成多个终端：

```bash
# 终端 1：prefill head
cd vllm_scripts
./serve/serve_head_only_template.sh -e logs/pd/<run-id>/prefill0.sh
```

```bash
# 终端 2：prefill MPI worker
cd vllm_scripts
mpirun --bind-to none --map-by slot -np 4 \
  bash ./serve/serve_mp_rpc_all_mpi_template.sh \
  -e logs/pd/<run-id>/prefill0.sh
```

```bash
# 终端 3：decode head
cd vllm_scripts
./serve/serve_head_only_template.sh -e logs/pd/<run-id>/decode0.sh
```

```bash
# 终端 4：decode MPI worker
cd vllm_scripts
mpirun --bind-to none --map-by slot -np 4 \
  bash ./serve/serve_mp_rpc_all_mpi_template.sh \
  -e logs/pd/<run-id>/decode0.sh
```

```bash
# 终端 5：P/D proxy
cd vllm_scripts
python ./serve_test/pd_proxy.py \
  --port <proxy-port> \
  --prefill-url http://127.0.0.1:<prefill-port> \
  --decode-url http://127.0.0.1:<decode-port>
```

```bash
# 终端 6：测试 proxy
cd vllm_scripts
./serve_test/serve_test_template.sh -e logs/pd/<run-id>/proxy.sh
```

## N P x M D 拓扑

当前 P/D wrapper 支持通过环境变量启动多组 prefill/decode：

```bash
USER_VLLM_PD_PREFILL_COUNT=2 USER_VLLM_PD_DECODE_COUNT=1 \
./run_vllm_test.sh --pd --multi-test -e <preset.sh>
```

proxy 接收逗号分隔 URL，并做 round-robin：

```bash
python ./serve_test/pd_proxy.py \
  --port 14800 \
  --prefill-url http://127.0.0.1:14900,http://127.0.0.1:15000 \
  --decode-url http://127.0.0.1:15100
```

## 日志位置

常规 mp：

```text
logs/run_vllm_test.log
logs/vllm_serve_log.txt
logs/test.log
```

常规 mpi：

```text
logs/run_vllm_test.log
logs/vllm_head_log.txt
logs/mpi_workers.log
logs/test.log
```

P/D：

```text
logs/pd/<run-id>/prefill0/launch.log
logs/pd/<run-id>/prefill0/logs/
logs/pd/<run-id>/decode0/launch.log
logs/pd/<run-id>/decode0/logs/
logs/pd/<run-id>/proxy/proxy.log
logs/test.log
```

成功/失败归档：

```text
logs/success/<run-id>/
logs/failed/<run-id>/
```

## 调试建议

1. 先用统一入口复现问题，记录 preset、launcher、端口和日志目录。
2. 如果是非 P/D mpi 问题，按“三终端展开”直接运行 head、worker、test。
3. 如果是 P/D 问题，先用 `--pd --no-test` 生成 overlay preset，再按 P/D 展开命令手工启动。
4. 非 P/D 端口冲突默认会提前失败；需要自动避让时加 `--auto-port`，并以启动日志打印的 OpenAI API 端口为准。
5. 如果测试返回 404 model not found，先确认请求命中的端口是否是本次启动的端口，再看 `/v1/models` 中的模型 id。
6. P/D 会为 HTTP、DP RPC、MPI coord、MP RPC ready 分别分配端口，端口问题优先看 `logs/pd/<run-id>/` 下生成的 overlay preset 和启动日志。
7. decode 侧是否命中外部 KV cache，以 decode 日志中的 `External Cache Hit` 为准。
