# AFD-EP 当前交付入口

当前功能的端到端测试步骤见 [AFD 跨机端到端测试](e2e_test.md)。

跨平台移植请先使用 [分层测试指南](portability_test_guide.md) 和 [结果填写模板](portability_test_result_template.md)，分别报告 MPI 基础、跨作业连接、真实 V7、三种精度模型及退出/清理结果。

[默认初始化：当前决定](mpi_default_20260915.md) · [UCX 退出错误：原因探查](ucx_finalize_20260915.md) · [SERIALIZED：阶段记录](serialized_20260915.md) · [窗口提前创建：后续修订](early_windows_20260915.md) · [验收报告（2026-09-15）](acceptance_20260915.md) · [HTML 阅读版](acceptance_20260915.html) · [开发套件修改记录](devkit_changes.md) · [重建与验证证据](validation_20260915/)

AFD 将 routed experts 放到独立 F ranks：A 保留调度、Attention/GDN、KV、gate/top-k、shared experts 和 dense 层；F 只加载本地 EP 专家。当前是固定 MPMD world，使用 V7 RMA 数据面，不是动态加入/扩缩容服务。

## 支持和配置

当前重建验收覆盖 Qwen3-30B-A3B-Instruct-2507 的 BF16、block-FP8、MXFP4A16，V2 runner、eager、BF16 activation、A2（DP1/TP2）+ F2（EP2）。Qwen3.6 和 dummy GLM preset 保留为已有开发入口，本轮未重跑，不据此扩大支持声明。

- 同时设置 `VLLM_XCPU_ENABLE_AF_EP=1`、`VLLM_CPU_USE_MPI=1`、`VLLM_USE_V2_MODEL_RUNNER=1`；AFD preset 使用 `--all2all-backend mpi_alltoallv_v7`。
- AFD 提供文件名包含 `_eager_` / `_compile_` 的独立 preset。compile 文件启用 A 侧 `DYNAMO_TRACE_ONCE + Inductor` 和 F 侧每层事务 Inductor 编译。CUDAGraph capture 不支持；实现和验证见 [A/F compile 适配](compile.md)，测试方法见 [E2E 文档](e2e_test.md)。
- A/F 权重加载完成后、profile/forward 前共同创建窗口；首次 forward 不再创建窗口。
- A/F 与非 AFD 使用 mpi4py 默认初始化，移除显式线程等级设置及最低等级检查；当前默认请求 MULTIPLE，实测提供等级 3。
- `VLLM_XCPU_AF_FORWARD_ALLREDUCE` 默认关闭；撤销 SERIALIZED 阶段为此选项增加的同步。UCX/PMIx 退出问题保留记录，按当前决定暂不解决。
- launcher 按 `DP × TP × PP` 计算 A rank 数，`USER_VLLM_EP_SIZE` 独立配置 F rank 数；MPI world size 为两者之和，支持 A4/F2 等不对称拓扑。
- 各 rank 必须使用相同模型配置、revision 和 MoE 层序；每步不传 epoch。初始化验证通信容量/shape，不验证完整 checkpoint 内容身份。
- `VLLM_OPTIONAL_ARGS` 中的 `--revision VALUE` / `--revision=VALUE`、`--trust-remote-code` / `--no-trust-remote-code` 透传 F；model、load-format、capacity 已单独传入。
- 固定 world 通过 launcher 终止。当前只验收启动失败，不承诺 STOP/ACK、运行期故障恢复或无损重启。
- 保留现有全局进程名清理策略；共享机器上的非 AFD 清理可能结束 AFD，按本轮意见接受此限制。AFD 自身仍有 run ID 清理及远端 CHECK。

## 重建与运行

从根目录激活 `.venv`。按 `torch_mcpu → torch_xcpu → torch_mpi_ext` 次序 clean/build；具体已执行命令和开关见验收报告。V7 删除 epoch 改变算子签名，不能混用旧安装库。

```bash
source .venv/bin/activate
export QWEN3_30B_A3B_BF16_MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507
export QWEN3_30B_A3B_FP8_MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507-FP8
export QWEN3_30B_A3B_MXFP4_MODEL_PATH=/path/to/Qwen3-30B-A3B-MXFP4A16
export VLLM_MPI_HOSTFILE="$PWD/vllm_scripts/mpi_tools/afd_tp2_ep2.hostfile"
export USER_VLLM_DATA_PARALLEL_RPC_IP=172.30.3.12
export VLLM_TEST_LOG_DIR="$PWD/vllm_scripts/logs/my_afd_run"
./vllm_scripts/e2e/run_afd_crossnode_matrix.sh
```

hostfile、RPC IP 按部署修改；所有节点需访问相同源码、venv、checkpoint 路径并支持免交互 SSH。示例 node02–05 是同一物理宿主的网络 namespace，不能代替物理多机/NPU 验收。Open MPI 4.1.6 + UCX 1.16.0 TCP 的 `MPI_Finalize` 有间歇退出错误，详见 B08。

## 日志与历史文档

每次 wrapper 在 `VLLM_TEST_LOG_DIR/runs/<时间>_<preset>.<随机后缀>/` 创建独立目录，子服务和协调文件继承 `VLLM_RUN_LOG_DIR`。退出时清理并归档一次到 `success/`、`failed/` 或 `stopped/`，`run_result.json` 记录原运行、清理和最终退出码。归档移动失败则保留 runs 原目录并返回非零；SIGKILL 留下的未完成目录不会被下一次启动扫描。

旧设计、阶段记录、旧 manifest 和旧评估已归入 [开发记录存档](../archive/afd/20260914/README.md)。普通 EP 公共文档与 [动态 MPI 实验](../../tests/test_dynamic_mpi/README.md) 仍独立保留，不混入本轮固定 world 验收结论。

## 当前最小测试集

插件 AF-EP 保留 [5 个功能回归用例](../../vllm-xcpu-plugin/tests/af_ep/README.md)（2026-09-16 收敛）。真实 MPI 数值及完整模型 E2E 继续使用各自已有入口；历史验收报告中的测试数量保留当时数值。
