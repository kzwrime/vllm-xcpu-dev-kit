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

# MPI 启动 + 测试
./run_vllm_test.sh -e presets/mpi/moe/Qwen3-30B-A3B_dp2_tp2_ep_eager_alltoallv_v2.sh

# P/D 分离 + 并发测试
./run_vllm_test.sh --pd --multi-test -e presets/mpi/moe/Qwen3-30B-A3B_dp2_tp2_ep_eager_alltoallv_v2.sh
```

`run_vllm_test.sh` 的职责是编排：加载 preset、启动服务、等待就绪、执行测试、归档日志、清理进程。真正的底层启动仍然由 `serve/` 和 `serve_test/` 下的脚本完成。

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
4. 端口冲突时优先看启动日志打印的实际端口；P/D 会为 HTTP、DP RPC、MPI coord、MP RPC ready 分别分配端口。
5. decode 侧是否命中外部 KV cache，以 decode 日志中的 `External Cache Hit` 为准。
