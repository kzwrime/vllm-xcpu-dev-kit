# MP RPC + MPI Cross-Node Test

本文档用于验证 vLLM 0.24.0 上移植后的 `mp_rpc` executor。典型目标是：

```text
API/head server: wzk-vllm-xcpu-03-c3
MPI rank 0:      wzk-vllm-xcpu-03-c4  DP0 TP0, 同时启动 DP0 headless engine
MPI rank 1:      wzk-vllm-xcpu-03-c5  DP0 TP1
MPI rank 2:      wzk-vllm-xcpu-03-c6  DP1 TP0, 同时启动 DP1 headless engine
MPI rank 3:      wzk-vllm-xcpu-03-c7  DP1 TP1
```

也就是 `DP=2, TP=2, PP=1`，每个 MP RPC worker 一个节点。

## 前置检查

在 `wzk-vllm-xcpu-03-c3` 上执行：

```bash
cd /shared/vllm-xcpu-dev-kit-0.24/vllm_scripts
for h in wzk-vllm-xcpu-03-c4 wzk-vllm-xcpu-03-c5 wzk-vllm-xcpu-03-c6 wzk-vllm-xcpu-03-c7; do
  ssh "$h" 'hostname; hostname -I; cd /shared/vllm-xcpu-dev-kit-0.24/vllm_scripts && pwd'
done
```

如果 `hostname -I` 返回多个网段，确认第一个 IP 是这些节点之间可互通的业务网段。
当前脚本默认使用：

```bash
export VLLM_LOOPBACK_IP=$(hostname -I | awk '{print $1}')
```

网卡选择不正确时，需要在 preset 或命令行 overlay 中显式覆盖
`VLLM_LOOPBACK_IP`。

## Hostfile

在 `wzk-vllm-xcpu-03-c3` 上生成 hostfile：

```bash
cd /shared/vllm-xcpu-dev-kit-0.24/vllm_scripts
cat > hostfile.dp2_tp2_c4_c7 <<'EOF'
wzk-vllm-xcpu-03-c4 slots=1
wzk-vllm-xcpu-03-c5 slots=1
wzk-vllm-xcpu-03-c6 slots=1
wzk-vllm-xcpu-03-c7 slots=1
EOF
```

如果使用 MPICH hostfile，可去掉 `slots=1`：

```text
wzk-vllm-xcpu-03-c4
wzk-vllm-xcpu-03-c5
wzk-vllm-xcpu-03-c6
wzk-vllm-xcpu-03-c7
```

## 推荐验证入口

在 `wzk-vllm-xcpu-03-c3` 上执行：

```bash
cd /shared/vllm-xcpu-dev-kit-0.24
. .venv/bin/activate
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/shared/vllm-xcpu-dev-kit-0.24/.venv/lib/libiomp5.so"
cd vllm_scripts
VLLM_MPI_RUN_ARGS="--allow-run-as-root --hostfile hostfile.dp2_tp2_c4_c7 --bind-to none --map-by slot -x PATH -x VIRTUAL_ENV -x LD_PRELOAD" \
./run_vllm_test.sh \
  -e presets/mpi/dense/Qwen3-0.6B_dp2_tp2_eager.sh \
  --launcher mpi
```

`run_vllm_test.sh` 会执行：

```text
1. 在当前节点启动 API/head server
2. 在当前节点启动 MPI coordination server
3. 通过 mpirun 在 hostfile 的 c4-c7 上启动 4 个 rank
4. 每个 rank 启动一个 run_mp_rpc_worker
5. rank 0 和 rank 2 额外启动各自 DP rank 的 headless vLLM engine
6. 等待 /v1/models 就绪并运行 serve_test
7. 归档日志到 logs/success 或 logs/failed
```

如果只想启动服务后手工测试：

```bash
VLLM_MPI_RUN_ARGS="--allow-run-as-root --hostfile hostfile.dp2_tp2_c4_c7 --bind-to none --map-by slot -x PATH -x VIRTUAL_ENV -x LD_PRELOAD" \
./run_vllm_test.sh \
  -e presets/mpi/dense/Qwen3-0.6B_dp2_tp2_eager.sh \
  --launcher mpi \
  --no-test
```

另开终端：

```bash
cd /shared/vllm-xcpu-dev-kit-0.24/vllm_scripts
./serve_test/serve_test_template.sh \
  -e presets/mpi/dense/Qwen3-0.6B_dp2_tp2_eager.sh
```

## 关键配置

`presets/mpi/dense/Qwen3-0.6B_dp2_tp2_eager.sh` 已包含：

```bash
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1
```

`VLLM_USE_MPI_COORD=1` 会自动协调并生成：

```text
USER_VLLM_DATA_PARALLEL_ADDRESS  = rank 0 IP，即 DP rank 0 engine 地址
VLLM_DP_MASTER_WORKER_IP         = rank 0 IP，即 DP rank 0 worker0 地址
ExecutorIP                       = 当前 DP 组第一个 rank 的 IP
```

因此新版流程不需要手动修改 `serve_mp_rpc_all_mpi_template.sh` 中的
`ExecutorIP`。

## 手工拆分验证

统一入口等价于下面两个启动终端加一个测试终端。

终端 1，`wzk-vllm-xcpu-03-c3`：

```bash
cd /shared/vllm-xcpu-dev-kit-0.24/vllm_scripts
./serve/serve_head_only_template.sh \
  -e presets/mpi/dense/Qwen3-0.6B_dp2_tp2_eager.sh
```

终端 2，`wzk-vllm-xcpu-03-c3`：

```bash
cd /shared/vllm-xcpu-dev-kit-0.24/vllm_scripts
mpirun --allow-run-as-root --hostfile hostfile.dp2_tp2_c4_c7 --bind-to none --map-by slot \
  -x PATH -x VIRTUAL_ENV -x LD_PRELOAD -np 4 \
  bash ./serve/serve_mp_rpc_all_mpi_template.sh \
  -e presets/mpi/dense/Qwen3-0.6B_dp2_tp2_eager.sh
```

终端 3，`wzk-vllm-xcpu-03-c3`：

```bash
cd /shared/vllm-xcpu-dev-kit-0.24/vllm_scripts
./serve_test/serve_test_template.sh \
  -e presets/mpi/dense/Qwen3-0.6B_dp2_tp2_eager.sh
```

## 验收点

成功后检查：

```bash
grep -H "Application startup complete" logs/vllm_head_log.txt
grep -H "Starting vLLM serve" logs/vllm_serve_log_dp_rank*.txt
grep -H "Starting vLLM mp_rpc_worker" logs/vllm_worker_log_rank*.txt
grep -H "rank=" logs/mpi_workers.log
curl -s http://127.0.0.1:14800/v1/models
```

期望看到：

```text
logs/vllm_head_log.txt                         API/head server 就绪
logs/vllm_serve_log_dp_rank0.txt               DP rank 0 headless engine
logs/vllm_serve_log_dp_rank1.txt               DP rank 1 headless engine
logs/vllm_worker_log_rank0.txt                 c4 / DP0 TP0
logs/vllm_worker_log_rank1.txt                 c5 / DP0 TP1
logs/vllm_worker_log_rank2.txt                 c6 / DP1 TP0
logs/vllm_worker_log_rank3.txt                 c7 / DP1 TP1
```

MPI coordination 日志中应有类似拓扑：

```text
rank 0 -> c4 IP
rank 1 -> c5 IP
rank 2 -> c6 IP
rank 3 -> c7 IP
ExecutorIPs = {0: c4 IP, 2: c6 IP}
```

也可以直接检查生成的临时环境：

```bash
sed -n '1,80p' logs/tmp/vllm_mpi_env_server.sh
sed -n '1,80p' logs/tmp/vllm_mpi_env_rank_0.sh
sed -n '1,80p' logs/tmp/vllm_mpi_env_rank_2.sh
```

## 2026-07-07 实测记录

实测控制节点：

```text
当前会话节点: wzk-vllm-xcpu-03-c2
API/head:     wzk-vllm-xcpu-03-c3 / 172.30.3.13
rank 0:       wzk-vllm-xcpu-03-c4 / 172.30.3.14 / DP0 TP0
rank 1:       wzk-vllm-xcpu-03-c5 / 172.30.3.15 / DP0 TP1
rank 2:       wzk-vllm-xcpu-03-c6 / 172.30.3.16 / DP1 TP0
rank 3:       wzk-vllm-xcpu-03-c7 / 172.30.3.17 / DP1 TP1
```

由于 `wzk-vllm-xcpu-03-c3` 的 `hostname -I` 第一个地址是 `172.17.0.4`，
本次使用 overlay preset 固定 head RPC IP，并让每个 rank 选择 `172.30.*`
地址：

```bash
cat > logs/tmp/Qwen3-0.6B_dp2_tp2_eager_c3_head.sh <<'EOF'
source /shared/vllm-xcpu-dev-kit-0.24/vllm_scripts/presets/mpi/dense/Qwen3-0.6B_dp2_tp2_eager.sh
export USER_VLLM_DATA_PARALLEL_RPC_IP="172.30.3.13"
export VLLM_LOOPBACK_IP=$(hostname -I | tr " " "\n" | grep "^172\.30\." | head -1)
EOF
```

单请求验证通过：

```bash
cd /shared/vllm-xcpu-dev-kit-0.24
. .venv/bin/activate
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/shared/vllm-xcpu-dev-kit-0.24/.venv/lib/libiomp5.so"
cd vllm_scripts
VLLM_MPI_RUN_ARGS="--allow-run-as-root --hostfile logs/tmp/hostfile.dp2_tp2_c4_c7 --bind-to none --map-by slot -x PATH -x VIRTUAL_ENV -x LD_PRELOAD" \
./run_vllm_test.sh -e logs/tmp/Qwen3-0.6B_dp2_tp2_eager_c3_head.sh --launcher mpi
```

成功归档：

```text
logs/success/20260707_012317_logs_tmp_Qwen3-0.6B_dp2_tp2_eager_c3_head/
```

关键证据：

```text
rank=0, ip=172.30.3.14, host=wzk-vllm-xcpu-03-c4
rank=1, ip=172.30.3.15, host=wzk-vllm-xcpu-03-c5
rank=2, ip=172.30.3.16, host=wzk-vllm-xcpu-03-c6
rank=3, ip=172.30.3.17, host=wzk-vllm-xcpu-03-c7
[RANK=0][DP_RANK=0] Starting vLLM serve
[RANK=2][DP_RANK=1] Starting vLLM serve
system_fingerprint="vllm-...-tp2-dp2-..."
```

并发 `--multi-test` 在补齐 `torch_mcpu` 的 `aten::scatter.src_out` 后通过：

```bash
VLLM_MPI_RUN_ARGS="--allow-run-as-root --hostfile logs/tmp/hostfile.dp2_tp2_c4_c7 --bind-to none --map-by slot -x PATH -x VIRTUAL_ENV -x LD_PRELOAD" \
./run_vllm_test.sh -e logs/tmp/Qwen3-0.6B_dp2_tp2_eager_c3_head.sh --launcher mpi --multi-test
```

成功归档：

```text
logs/success/20260707_013912_logs_tmp_Qwen3-0.6B_dp2_tp2_eager_c3_head/
```

`logs/success/20260707_013912_logs_tmp_Qwen3-0.6B_dp2_tp2_eager_c3_head/test.log`
显示 T0-T4 五个并发任务全部完成，归档日志未再出现 `scatter.src_out`、
`EngineDeadError` 或 scheduler `KeyError`。

历史记录：补算子前，并发 `--multi-test` 失败，但不属于 MP RPC 启动失败：

```text
logs/failed/20260707_012620_logs_tmp_Qwen3-0.6B_dp2_tp2_eager_c3_head/
```

直接根因是 `torch_mcpu` 缺少并发路径触发的 ATen 算子：

```text
RuntimeError: Operator 'aten::scatter.src_out' is not implemented for torch_mcpu.
```

同时 engine shutdown 暴露 scheduler `KeyError`：

```text
KeyError: 'chatcmpl-...'
```

当时 `logs/test.log` 显示 5 个并发请求中 3 个完成、2 个失败。

## 覆盖矩阵

新功能验证建议至少覆盖下面几组：

```bash
# Dense, DP2 TP2, eager, 单请求
VLLM_MPI_RUN_ARGS="--allow-run-as-root --hostfile hostfile.dp2_tp2_c4_c7 --bind-to none --map-by slot -x PATH -x VIRTUAL_ENV -x LD_PRELOAD" \
./run_vllm_test.sh -e presets/mpi/dense/Qwen3-0.6B_dp2_tp2_eager.sh --launcher mpi

# Dense, DP2 TP2, compile, 单请求
VLLM_MPI_RUN_ARGS="--allow-run-as-root --hostfile hostfile.dp2_tp2_c4_c7 --bind-to none --map-by slot -x PATH -x VIRTUAL_ENV -x LD_PRELOAD" \
./run_vllm_test.sh -e presets/mpi/dense/Qwen3-0.6B_dp2_tp2_compile.sh --launcher mpi

# Dense, DP2 TP2, eager, 并发请求
VLLM_MPI_RUN_ARGS="--allow-run-as-root --hostfile hostfile.dp2_tp2_c4_c7 --bind-to none --map-by slot -x PATH -x VIRTUAL_ENV -x LD_PRELOAD" \
./run_vllm_test.sh -e presets/mpi/dense/Qwen3-0.6B_dp2_tp2_eager.sh --launcher mpi --multi-test
```

MoE/EP 覆盖可以使用已有 preset，例如：

```bash
VLLM_MPI_RUN_ARGS="--allow-run-as-root --hostfile hostfile.dp2_tp2_c4_c7 --bind-to none --map-by slot -x PATH -x VIRTUAL_ENV -x LD_PRELOAD" \
./run_vllm_test.sh \
  -e presets/mpi/moe/Qwen3-30B-A3B_dp2_tp2_ep_eager_alltoallv_v2.sh \
  --launcher mpi
```

如果只验证 MP RPC 跨节点而不启用 MPI 通信算子，可临时覆盖：

```bash
mkdir -p logs/tmp
cat > logs/tmp/Qwen3-0.6B_dp2_tp2_eager_gloo.sh <<'EOF'
source /shared/vllm-xcpu-dev-kit-0.24/vllm_scripts/presets/mpi/dense/Qwen3-0.6B_dp2_tp2_eager.sh
export VLLM_CPU_USE_MPI=0
EOF

VLLM_MPI_RUN_ARGS="--allow-run-as-root --hostfile hostfile.dp2_tp2_c4_c7 --bind-to none --map-by slot -x PATH -x VIRTUAL_ENV -x LD_PRELOAD" \
./run_vllm_test.sh -e logs/tmp/Qwen3-0.6B_dp2_tp2_eager_gloo.sh --launcher mpi
```

此时 `mpirun` 仍只是进程启动器，worker 间通信走 gloo。

## 常见问题

Open MPI 拒绝 root 运行时，在 `VLLM_MPI_RUN_ARGS` 中追加：

```bash
--allow-run-as-root
```

非交互 SSH 环境没有自动激活 `.venv` 时，需要显式激活 venv，并在
`VLLM_MPI_RUN_ARGS` 中追加：

```bash
-x PATH -x VIRTUAL_ENV -x LD_PRELOAD
```

端口占用时，可使用：

```bash
RUN_VLLM_TEST_AUTO_PORT=1 \
VLLM_MPI_RUN_ARGS="--allow-run-as-root --hostfile hostfile.dp2_tp2_c4_c7 --bind-to none --map-by slot -x PATH -x VIRTUAL_ENV -x LD_PRELOAD" \
./run_vllm_test.sh -e presets/mpi/dense/Qwen3-0.6B_dp2_tp2_eager.sh --launcher mpi
```

如果 head 启动后 worker 连接失败，优先检查：

```bash
grep -H "ERROR\\|Traceback\\|ConnectionRefused\\|Timeout" \
  logs/run_vllm_test.log logs/mpi_workers.log logs/vllm_*_log*.txt
```

若 `logs/tmp/vllm_mpi_env_rank_*.sh` 中的 IP 不在预期网段，说明
`VLLM_LOOPBACK_IP` 选择错误，需要在 preset 里按节点覆盖，或调整
`user_env_template.sh` 的 IP 获取方式。
