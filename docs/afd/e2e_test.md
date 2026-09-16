# AFD 跨机端到端测试

仅覆盖简单实验

测试规模：
- 4 Ranks
    - 1 Attn Cluster with vLLM (Rank0 & Rank1)
        - DP1
        - TP2
    - 1 MoE Cluster with standalone service (Rank2 & Rank3)
        - EP2

测试模型：
- `Qwen/Qwen3-30B-A3B-Instruct-2507`
- `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`
- `nm-testing/Qwen3-30B-A3B-MXFP4A16`

AFD 将 Attention 与 routed experts 放到不同进程：
- A 侧负责请求调度、Attention 和路由
- F 侧执行专家计算，再把结果返回 A 侧继续生成。

本测试使用真实模型权重，通过 HTTP API 发送请求，检查“加载模型 → 建立 A/F 通信 → 预热 → 并发生成 → 清理进程”的完整链路。

| 示例节点 | MPI global rank | 角色 |
|---|---|---|
| node02 | 0 | A0，Attention TP rank 0 |
| node03 | 1 | A1，Attention TP rank 1 |
| node04 | 2 | F0，专家 EP rank 0 |
| node05 | 3 | F1，专家 EP rank 1 |

下面的矩阵脚本会为每种精度分别启动一次服务、执行测试并清理，再测试下一种精度。通过表示这条配置下端到端功能可用；此处不测吞吐性能，也不进行模型精度评分。

## 1. 准备

各节点已安装对应架构的运行环境，源码、venv、模型使用相同绝对路径，节点间网络互通并支持免交互 SSH。以下命令在仓库根目录执行。

```bash
source .venv/bin/activate
export QWEN3_30B_A3B_BF16_MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507
export QWEN3_30B_A3B_FP8_MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507-FP8
export QWEN3_30B_A3B_MXFP4_MODEL_PATH=/path/to/Qwen3-30B-A3B-MXFP4A16

# 按实际机器修改；前两行为 A，后两行为 F（--map-by slot）。
cat > /tmp/afd.hostfile <<'EOF'
node02 slots=1
node03 slots=1
node04 slots=1
node05 slots=1
EOF
export VLLM_MPI_HOSTFILE=/tmp/afd.hostfile
export USER_VLLM_DATA_PARALLEL_RPC_IP=192.0.2.10  # 改为启动节点的可达 IP
export VLLM_TEST_LOG_DIR="$PWD/vllm_scripts/logs/afd_e2e_$(date +%Y%m%d_%H%M%S)"
```

默认启动参数适用于当前 Open MPI + UCX TCP 环境。其他版本按本机支持情况设置 `VLLM_MPI_RUN_ARGS`、`UCX_TLS`；当前 launcher 使用 Open MPI 参数，其他 MPI 实现需先适配启动器。整体上不依赖特定 MPI 行为。

`USER_VLLM_DATA_PARALLEL_RPC_IP` 是 vLLM 内部 RPC 的可达地址，也是 API Server 的地址。

真实跨机验收应使用不同物理机器，多个容器共享一台宿主不能代替物理跨机测试。

## 2. 运行

依次测试三种精度，包含并发请求：

```bash
./vllm_scripts/e2e/run_afd_crossnode_matrix.sh
```

`run_afd_crossnode_matrix.sh` 的内容如下。

注：
1. 如果用户已经有良好配置的环境变量和 `user_env_template.sh`， 该脚本也可以不使用。直接按照原有流程启动模型即可。
2. 在 x86 上，使用 OpenMPI + 跨节点 + MPI Win + RMA 会受到一些后端约束，因此才做了一些额外的 MPI 参数配置。使用 OpenMPI 5.x 或 MPICH 不需要额外配置后端或需要其他参数。

```bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VLLM_MPI_HOSTFILE="$(realpath -e "${VLLM_MPI_HOSTFILE:-$SCRIPT_DIR/mpi_tools/afd_tp2_ep2.hostfile}")"
export VLLM_MPI_HOSTFILE

cd "$SCRIPT_DIR"

export USER_VLLM_DATA_PARALLEL_RPC_IP="${USER_VLLM_DATA_PARALLEL_RPC_IP:-$(hostname -I | awk '{print $1}')}"

# Open MPI 4.x pt2pt OSC cannot serve MPI_THREAD_MULTIPLE; use UCX over TCP.
export UCX_TLS="${UCX_TLS:-tcp,self}"
export VLLM_MPI_RUN_ARGS="${VLLM_MPI_RUN_ARGS:---allow-run-as-root --bind-to none --map-by slot --mca osc ucx --mca osc_ucx_tls any --mca osc_ucx_devices any --mca btl self,tcp}"

export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-2000}"
for preset in \
    Qwen3-30B-A3B_dp1_tp2_af_ep_v7 \
    Qwen3-30B-A3B-FP8_dp1_tp2_af_ep_v7 \
    Qwen3-30B-A3B-MXFP4A16_dp1_tp2_af_ep_v7; do
    ./run_vllm_test.sh -e "presets/mpi/moe/${preset}.sh" \
        --multi-test --multi-test-temperature 0 --test-timeout 300 "$@"
done
```

实际调用 [`serve_test/test_multl_stream.py`](../../vllm_scripts/serve_test/test_multl_stream.py)，默认并发发送以下 5 条流式请求（未设置 `VLLM_PD_MULTI_INCLUDE_LONG=1` 时）：

| 任务日志 | 问题 |
|---|---|
| `vllm_task_0.log` | 请用一段话简单介绍一下量子计算。 |
| `vllm_task_1.log` | 写一首关于春天的七言绝句。 |
| `vllm_task_2.log` | 欧盟有多少个国家，详细展开论述欧盟现状。 |
| `vllm_task_3.log` | 解释一下相对论的核心思想。 |
| `vllm_task_4.log` | 给出三个提高编程效率的建议。 |

矩阵使用 `temperature=0`、默认 `max_tokens=16`，默认关闭 thinking。16 tokens 用来快速确认生成链路，通常只能得到回答开头。需要观察更完整的回答时可运行：

```bash
./vllm_scripts/e2e/run_afd_crossnode_matrix.sh --multi-test-max-tokens 128
```

默认服务启动等待上限为 2000 秒，engine ready timeout 为 1800 秒；服务就绪后，并发测试阶段的超时为 300 秒。加载慢与请求阶段超时应分开定位。

脚本遇到失败会停止，后续精度记为“未测试”。

单独重测时，稍微修改脚本即可。

BF16 和 MXFP4A16 的 preset 分别为 `Qwen3-30B-A3B_dp1_tp2_af_ep_v7.sh`、`Qwen3-30B-A3B-MXFP4A16_dp1_tp2_af_ep_v7.sh`。

## 3. 关键日志与预期效果

每种精度都有独立日志目录。运行时位于 `$VLLM_TEST_LOG_DIR/runs/`，结束后移入 `success/` 或 `failed/`；以终端最后的“日志已归档到”路径为准。

| 文件 | 查看内容 |
|---|---|
| `mpi_workers.log` | A/F 权重加载、V7 通信初始化、worker 错误 |
| `vllm_head_log.txt` | API/engine 启动、预热及请求处理 |
| `test.log` | 测试参数、T0–T4 状态、请求异常与空输出警告 |
| `vllm_task_0.log` ～ `vllm_task_4.log` | 每条问题的实际文本和流式统计 |
| `mpi_cleanup.log` | 本次 MPI 进程在各节点的清理结果 |
| `run_result.json` | 运行、清理、最终退出码 |

### 3.1 服务启动：确认确实运行 AFD

终端应显示以下关键内容，run ID、路径和打印顺序可能不同：

```text
AFD run: afd_... hostfile=/tmp/afd.hostfile A ranks=0-1 F ranks=2-3
AF-EP MPMD: A ranks=2 F ranks=2
```

`mpi_workers.log` 中应有 F0/F1 的 `weights ready`，随后四个角色均有 `transport ready`。以下为 Qwen3-30B-A3B 的日志格式，省略 PID/时间前缀，权重计数随精度变化：

```text
AF-EP F0 weights ready: loaded=... remote=...
AF-EP F1 weights ready: loaded=... remote=...
AF-EP V7 transport ready: role=ATTN rank=0 hidden_size=2048 topk=8
AF-EP V7 transport ready: role=ATTN rank=1 hidden_size=2048 topk=8
AF-EP V7 transport ready: role=MOE rank=0 hidden_size=2048 topk=8
AF-EP V7 transport ready: role=MOE rank=1 hidden_size=2048 topk=8
```

这里 `rank` 是**角色内编号**，所以 MOE rank 0/1 对应 global rank 2/3。四条 ready 表示 A/F 的 V7 通信窗口已完成初始化；仅看到权重加载完成还不能判定通信可用。完成预热后，终端应出现：

```text
[SUCCESS] vLLM API 服务就绪
[INFO] 运行并发 multi test...
```

### 3.2 请求完成：检查每条实际输出

`test.log` 应显示 `max_tokens: 16`、`temperature: 0.0`、`thinking: disabled`，最终 T0–T4 均为“已完成”，正文字符数大于 0。终端会输出：

```text
[SUCCESS] Multi test 完成
[INFO] Multi test 结果汇总:
```

下面是已有 FP8 测试中 `vllm_task_0.log` 的输出示例；目标机器无需逐字一致：

```text
【Prompt】: 请用一段话简单介绍一下量子计算。
========================================
[content]
量子计算是一种基于量子力学原理的计算方式，利用量子比特（q

========================================
[统计] reasoning_chars=0, content_chars=29, stream_chunks=17, choice_chunks=17, reasoning_chunks=0, content_chunks=16, finish_reasons=['length'], stop_reasons=[]
【生成结束】
```

检查 5 个文件均有相关、可读的 `[content]`，`content_chars > 0`，没有 `[请求失败]`。`finish_reasons=['length']` 表示到达 16 tokens 限制，回答被截断是预期现象；`stop` 表示提前正常结束。字符数和 chunk 数不要求与示例相同。

**不能仅凭“Multi test 完成”判定通过**：当前脚本对“空正文/空输出”只打印警告，仍可能返回 0。出现 `[空输出诊断]`、`content_chars=0` 时应记录并排查，不能记为通过。明显乱码、重复异常或完全无关的输出也应单独报告。

### 3.3 结束：确认退出状态与清理

`run_result.json` 正常应为：

```json
{"run_exit_code":0,"cleanup_exit_code":0,"exit_code":0}
```

`mpi_cleanup.log` 应覆盖所有参与节点，例如：

```text
node02: CLEAN run=afd_...
node03: CLEAN run=afd_...
node04: CLEAN run=afd_...
node05: CLEAN run=afd_...
```

脚本自动清理本次进程；中断时使用 Ctrl+C，等待清理完成。`CLEAN` 检查的是带本次 run ID 的 MPI 进程；同时确认 head/API 已结束、测试端口释放。若日志有 `remaining PIDs` 或远端 SSH 失败，应清理后再重测，不要使用全局 `pkill python/mpirun`。

即使目录名为 `success`、三个退出码为 0，也应检查服务日志是否有 MPI/UCX/PMIx 错误。推理成功但退出报错，应记录为“推理通过，退出失败”。

## 4. 如何判定与定位

**每种精度通过条件**：A2/F2 初始化完成、API ready、5 条请求均有正常正文、无 MPI/算子错误、退出及清理成功。三种精度分别填写，不以 BF16 通过替代量化模型结果。

| 最后看到的现象 | 优先检查 |
|---|---|
| 没有 A/F worker 启动，或 mpirun 参数报错 | hostfile、SSH、远端路径、MPI 版本和启动参数 |
| 模型加载报错，没有 F0/F1 weights ready | 模型路径/格式、设备算子、内存及权重加载错误 |
| 权重就绪，但缺少某个角色的 transport ready | 缺失 rank 的日志、MPI 窗口初始化、OSC/UCX、网络 |
| 四条 transport ready 齐全，但 API 未就绪 | head/worker 中的预热、KV cache、算子或容量错误 |
| API 就绪，但请求卡住或返回错误 | `test.log`、对应 task 文件、worker 的 dispatch/combine 或执行错误 |
| 请求结束但正文为空 | task 文件的空输出诊断、finish reason、模型/chat template |
| 文本正常但退出或清理失败 | MPI/UCX 退出错误、`mpi_cleanup.log`、残留进程和 SSH 错误 |

提交结果时填写下表，并附机器架构、MPI 版本、hostfile、实际启动参数及整份运行日志目录：

| 精度 | 结果 | 停在哪一步 | 首条错误及日志路径 |
|---|---|---|---|
| BF16 | | 加载 / 窗口初始化 / 预热 / 请求 / 退出 / 清理 | |
| FP8 | | | |
| MXFP4A16 | | | |

未执行的精度记录为“未测试”。上述阶段用于定位本次端到端运行停在哪里，不需要另跑底层 MPI 测试才能填写。
