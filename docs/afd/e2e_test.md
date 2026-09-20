# AFD 跨机端到端测试（独立 A/F rank 与多 DP）

本文维护当前 AFD 启动、拓扑配置和验收方法。AFD 使用一个 MPI world，按 global rank 区分 Attention（A）与 MoE（F），A/F rank 数不要求相等。

并行配置：
- 基础测试
    - 4 ranks（A2/F2、DP1-TP2-EP2）
- 不对称测试
    - 6 ranks（A4/F2、DP2-TP2-EP2）
    - A1/F4
    - A4/F1

测试模型：
- `Qwen/Qwen3-30B-A3B-Instruct-2507`
- `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`
- `nm-testing/Qwen3-30B-A3B-MXFP4A16`
- `Qwen/Qwen3.6-35B-A3B`

AFD 将 Attention 与 routed experts 放到不同进程：
- A 侧负责请求调度、Attention 和路由
- F 侧执行专家计算，再把结果返回 A 侧继续生成。

矩阵中的拓扑和精度测试全部使用真实模型权重，通过 HTTP API 检查“加载模型 → 建立 A/F 通信 → 预热 → 并发生成 → 清理进程”的完整链路。

示例基础测试 4 ranks（A2/F2、DP1-TP2-EP2）如下

| 示例节点 | MPI global rank | A2/F2 角色 |
|---|---|---|
| node02 | 0 | A0，Attention TP rank 0 |
| node03 | 1 | A1，Attention TP rank 1 |
| node04 | 2 | F0，专家 EP rank 0 |
| node05 | 3 | F1，专家 EP rank 1 |

下面的矩阵脚本会为每种拓扑或精度分别启动一次服务、执行测试并清理，再测试下一项。通过表示这条配置下端到端功能可用；此处不测吞吐性能，也不进行模型精度评分。

## 0. Rank 拓扑合同

启动参数满足：

TODO: `A = USER_VLLM_DATA_PARALLEL_SIZE × USER_VLLM_MPC_SIZE` 中用简单的符号 `A` 表达太草率了，这是一个 size。

```text
A = USER_VLLM_DATA_PARALLEL_SIZE × USER_VLLM_MPC_SIZE
USER_VLLM_MPC_SIZE = USER_VLLM_TP_SIZE × USER_VLLM_PP_SIZE
F = USER_VLLM_EP_SIZE
MPI world size = A + F
```

global rank 划分关系
- `[0, A)` 为 Attention, `local rank = global rank`
- `[A, A+F)` 为 MoE, `local rank = global_rank - A`

`serve_afd_mp_rpc_all_mpi_template.sh` 在每个 MPI rank 内完成这一判断；替代掉了 `serve_mp_rpc_all_mpi_template.sh`，如有需要请修改（如依据作业管理系统获取 rank）。

| 配置 | A ranks | F ranks | MPI global rank 映射 |
|---|---:|---:|---|
| DP1-TP1-EP4 | 1 | 4 | A:0; F:1–4 |
| DP2-TP2-EP1 | 4 | 1 | A:0–3; F:4 |
| DP2-TP2-EP2 | 4 | 2 | A:0–3; F:4–5 |
| DP1-TP2-EP2 | 2 | 2 | A:0–1; F:2–3 |

`EP` 在本文中特指 F rank 数。F rank 数必须能均匀切分模型的 routed experts。

A 侧 vLLM 的本地 MoE/EP 布局只描述 A ranks，不再用来决定 F 侧的专家分片；远端分片按 `num_experts / USER_VLLM_EP_SIZE` 计算。

## 1. 准备

各节点已安装对应架构的运行环境，源码、venv、模型使用相同绝对路径，节点间网络互通并支持免交互 SSH。以下命令在仓库根目录执行。

```bash
source .venv/bin/activate
export QWEN3_30B_A3B_BF16_MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507
export QWEN3_30B_A3B_FP8_MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507-FP8
export QWEN3_30B_A3B_MXFP4_MODEL_PATH=/path/to/Qwen3-30B-A3B-MXFP4A16
export QWEN3_6_35B_A3B_BF16_MODEL_PATH=/path/to/Qwen3.6-35B-A3B

# 默认 matrix hostfile 提供 6 个 slot，按实际机器修改主机名。
cat > /tmp/afd.hostfile <<'EOF'
node02 slots=1
node03 slots=1
node04 slots=1
node05 slots=1
node06 slots=1
node07 slots=1
EOF
export VLLM_MPI_HOSTFILE=/tmp/afd.hostfile
export USER_VLLM_DATA_PARALLEL_RPC_IP=192.0.2.10  # 改为启动节点的可达 IP
export VLLM_TEST_LOG_DIR="$PWD/vllm_scripts/logs/afd_e2e_$(date +%Y%m%d_%H%M%S)"
```

AFD preset 还必须设置如下配置

注：仓库内现有 `*_af_ep_v7.sh` preset 已包含这些设置，`USER_VLLM_EP_SIZE` 可在启动前覆盖。默认 `mpi_tools/afd_tp2_ep2.hostfile` 已扩展为 6 个 slot；修改其中的主机名后，整个矩阵复用这一个 hostfile。

```bash
export USER_VLLM_EP_SIZE=2
export USER_VLLM_MPI_SIZE=$((USER_VLLM_DATA_PARALLEL_SIZE * USER_VLLM_MPC_SIZE + USER_VLLM_EP_SIZE))
export VLLM_MPI_WORKER_TEMPLATE="$PWD/vllm_scripts/serve/serve_afd_mp_rpc_all_mpi_template.sh"
```

默认启动参数适用于当前 Open MPI + UCX TCP 环境。其他版本按本机支持情况设置 `VLLM_MPI_RUN_ARGS`、`UCX_TLS`；当前 launcher 使用 Open MPI 参数，其他 MPI 实现需先适配启动器。整体上不依赖特定 MPI 行为。

`USER_VLLM_DATA_PARALLEL_RPC_IP` 是 vLLM 内部 RPC 的可达地址，也是 API Server 的地址。

真实跨机验收应使用不同物理机器，多个容器共享一台宿主不能代替物理跨机测试。

## 2. 运行

### 2.1 统一拓扑与精度矩阵

脚本按顺序覆盖多 DP、非对称 A/F 和三种真实权重精度，所有 case 都包含并发请求：

| 顺序 | 拓扑 | preset |
|---:|---|---|
| 1 | A4/F2，DP2-TP2-EP2 | `Qwen3.6-35B-A3B_dp2_tp2_af_ep_v7.sh` |
| 2 | A1/F4，DP1-TP1-EP4 | `Qwen3-30B-A3B-FP8_dp1_tp1_af_ep4_v7.sh` |
| 3 | A4/F1，DP2-TP2-EP1 | `Qwen3.6-35B-A3B_dp2_tp2_af_ep_v7.sh`，覆盖 `EP=1` |
| 4 | A2/F2，DP1-TP2-EP2 | `Qwen3-30B-A3B_dp1_tp2_af_ep_v7.sh` |
| 5 | A2/F2，DP1-TP2-EP2 | `Qwen3-30B-A3B-FP8_dp1_tp2_af_ep_v7.sh` |
| 6 | A2/F2，DP1-TP2-EP2 | `Qwen3-30B-A3B-MXFP4A16_dp1_tp2_af_ep_v7` |

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

# Real-weight topology coverage.
env USER_VLLM_EP_SIZE=2 \
  ./run_vllm_test.sh -e presets/mpi/moe/Qwen3.6-35B-A3B_dp2_tp2_af_ep_v7.sh \
    --multi-test --multi-test-temperature 0 --test-timeout 300 "$@"

env USER_VLLM_EP_SIZE=4 \
  ./run_vllm_test.sh -e presets/mpi/moe/Qwen3-30B-A3B-FP8_dp1_tp1_af_ep4_v7.sh \
    --multi-test --multi-test-temperature 0 --test-timeout 300 "$@"

env USER_VLLM_EP_SIZE=1 \
  ./run_vllm_test.sh -e presets/mpi/moe/Qwen3.6-35B-A3B_dp2_tp2_af_ep_v7.sh \
    --multi-test --multi-test-temperature 0 --test-timeout 300 "$@"

# Real-weight precision coverage on the baseline A2/F2 topology.
env USER_VLLM_EP_SIZE=2 \
  ./run_vllm_test.sh -e presets/mpi/moe/Qwen3-30B-A3B_dp1_tp2_af_ep_v7.sh \
    --multi-test --multi-test-temperature 0 --test-timeout 300 "$@"

env USER_VLLM_EP_SIZE=2 \
  ./run_vllm_test.sh -e presets/mpi/moe/Qwen3-30B-A3B-FP8_dp1_tp2_af_ep_v7.sh \
    --multi-test --multi-test-temperature 0 --test-timeout 300 "$@"

env USER_VLLM_EP_SIZE=2 \
  ./run_vllm_test.sh -e presets/mpi/moe/Qwen3-30B-A3B-MXFP4A16_dp1_tp2_af_ep_v7.sh \
    --multi-test --multi-test-temperature 0 --test-timeout 300 "$@"
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

脚本遇到失败会停止，后续 case 记为“未测试”。

单独重测时，复制脚本中对应的 `env USER_VLLM_EP_SIZE=... ./run_vllm_test.sh ...` 命令即可。

BF16 和 MXFP4A16 的 preset 分别为 `Qwen3-30B-A3B_dp1_tp2_af_ep_v7.sh`、`Qwen3-30B-A3B-MXFP4A16_dp1_tp2_af_ep_v7.sh`。

### 2.2 独立启动与作业管理系统适配

与 [`vllm_scripts/RUNBOOK.md`](../../vllm_scripts/RUNBOOK.md) 中普通 MPI 模式的“三终端”方法相同，AFD 也可以把 API head、MPI ranks 和测试客户端分别启动。这适合定位启动问题，也适合接入不能直接调用 `mpirun` 的 Slurm、PBS、LSF 或厂商作业管理系统。

这里的“独立启动”有一个重要边界：

- API head 和测试客户端不属于 AFD MPI world，可以作为单独进程或单独作业步骤启动。
- 所有 A/F rank 必须由同一个 MPI/PMI 作业步骤建立，并处于同一个 `MPI_COMM_WORLD`。当前 AFD 通信包含 collective、barrier 和 MPI RMA，仅给若干互不相关的进程设置相同的 rank/size 环境变量并不能建立通信。
- 两次独立的 `srun`、`jsrun` 或其他任务启动通常会创建两个 communicator，不能分别启动 A ranks 和 F ranks；除非平台明确支持把它们放进同一个 MPI world，并能保证本文的 global rank 顺序。

#### 2.2.1 手工拆分启动

以下命令是启动关系示例。三个终端应使用同一个绝对 preset 路径、同一套端口和同一个运行环境；实际日志目录与地址按集群修改。

```bash
cd /absolute/path/to/vllm-xcpu-dev-kit/vllm_scripts
PRESET="$PWD/presets/mpi/moe/Qwen3.6-35B-A3B_dp2_tp2_af_ep_v7.sh"
export VLLM_RUN_LOG_DIR="$PWD/logs/afd_manual_dp2_tp2_ep2"
# 同一次运行的 head 与 ranks 填写相同且唯一的值。
export VLLM_AF_RUN_ID="afd_manual_20260920_01"

# 终端/作业步骤 1：API head，同时提供 MPI coordination 服务。
bash ./serve/serve_head_only_template.sh -e "$PRESET"
```

先启动 head；它在等待 A ranks 连接 coordination 服务时保持前台运行。然后在分配到的计算资源中启动一个包含 `A + F` 个 task 的并行作业步骤：

```bash
cd /absolute/path/to/vllm-xcpu-dev-kit/vllm_scripts
PRESET="$PWD/presets/mpi/moe/Qwen3.6-35B-A3B_dp2_tp2_af_ep_v7.sh"
export VLLM_RUN_LOG_DIR="$PWD/logs/afd_manual_dp2_tp2_ep2"
export VLLM_AF_RUN_ID="afd_manual_20260920_01"

# mpirun 环境下，DP2-TP2-EP2 共 6 ranks。
mpirun --bind-to none --map-by slot -np 6 \
  bash ./serve/serve_afd_mp_rpc_all_mpi_template.sh -e "$PRESET"
```

服务就绪后，在第三个终端执行请求测试：

```bash
cd /absolute/path/to/vllm-xcpu-dev-kit/vllm_scripts
PRESET="$PWD/presets/mpi/moe/Qwen3.6-35B-A3B_dp2_tp2_af_ep_v7.sh"
bash ./serve_test/serve_test_template.sh -e "$PRESET"
```

与统一入口不同，这种方式不会自动选择并传播端口、监控所有子进程、归档日志或完成跨节点清理。不要直接使用 `--auto-port` 后再假设三个独立进程会得到相同的运行时 override；应在 preset 或所有作业步骤的公共环境中显式设置固定且不冲突的 API、DP RPC、MPI coordination 和 MP RPC ready 端口。

#### 2.2.2 不允许直接使用 `mpirun` 时

在作业管理系统中，用平台提供的并行任务启动命令替换上例的 `mpirun`。并行启动器应在每个 task 上执行同一个 AFD serve 脚本，由脚本根据 global rank 自动进入 A 或 F 分支，不要为 A/F 分别准备两个互不相关的作业步骤。以下 Slurm 命令只表示接口关系，站点的 MPI plugin、CPU 绑定、节点分布和环境导出参数需要按集群修改：

```bash
# 提交脚本需先从 preset 或提交参数得到 USER_VLLM_MPI_SIZE=A+F。
srun --ntasks "$USER_VLLM_MPI_SIZE" --export=ALL \
  bash /absolute/path/to/vllm_scripts/serve/serve_afd_mp_rpc_all_mpi_template.sh \
  -e /absolute/path/to/vllm_scripts/presets/mpi/moe/<afd-preset>.sh
```

PBS、LSF、`jsrun`、`yhrun` 和厂商启动器使用各自等价的 task-count、MPI/PMI 接入及环境导出选项，不能机械照搬 `srun` 参数。

平台适配至少要满足下表约束：

| 项目 | 要求 |
|---|---|
| MPI world | 一个作业步骤创建恰好 `A + F` 个进程，并让 `mpi4py.MPI.COMM_WORLD` 在所有进程中看到相同 size。 |
| rank 顺序 | global rank `[0, A)` 为 Attention，`[A, A+F)` 为 MoE；DP2-TP2-EP2 必须是 rank 0–3 为 A、4–5 为 F。 |
| rank/size 检测 | 当前脚本识别 Open MPI、PMIx、PMI、Cray MPICH 和 Slurm 的常见变量。平台变量不同则适配 `mpi_tools/mpi_get_rank_size.sh` 或增加轻量 wrapper。只设置这些变量不能代替真实 MPI world。 |
| 环境 | 所有 task 使用相同的 Python/venv、`PATH`、`PYTHONPATH`、`LD_LIBRARY_PATH`、模型路径、preset 及 `USER_VLLM_*`、`VLLM_*`、`TORCH_*`、`UCX_*` 配置。不能保证环境透传时，应由提交脚本显式导出。 |
| 路径与日志 | 使用各节点都可见的绝对源码、preset、模型和日志路径，或在 job prolog 中以相同路径完成 staging；调度器输出最好按 job/task/rank 分文件保存。 |
| 网络 | Attention ranks 能访问 API head 的 coordination/DP RPC 地址，所有 A/F rank 之间满足所用 MPI transport 和 RMA 的网络要求。`USER_VLLM_DATA_PARALLEL_RPC_IP` 不能写成其他节点不可达的 loopback 地址。 |
| 生命周期 | head、rank 作业步骤和测试客户端应属于同一作业生命周期。取消、超时或任一 rank 失败时要终止整组进程，不能留下占用端口或 MPI collective 的孤儿进程。 |

推荐新建站点级提交脚本，例如 `submit_afd_<site>.sh`，在其中完成资源申请、模块/venv 加载、地址与固定端口选择、日志路径、任务启动和退出清理。AFD 的 rank 分流仍保留在 `serve_afd_mp_rpc_all_mpi_template.sh`，不要把站点逻辑或 AFD 判断混入 `launcher_common.sh`。如果只是调度器的 rank 环境变量命名不同，优先适配 `mpi_tools/mpi_get_rank_size.sh`；如果启动参数、节点放置或日志收集差异较大，可以复制 AFD serve 模板形成站点版本，但必须保留以下合同：

1. world size 始终为 `DP × TP × PP + EP`；
2. A/F 使用连续且不重叠的 global rank 区间；
3. 只有 A ranks 连接 vLLM MPI coordination 服务，F ranks 直接进入 MoE 服务；
4. A/F 全部进入同一个 MPI communicator，并共同完成 AFD collective、RMA 初始化和退出；
5. 任一 rank 出错时，作业管理系统能够整体终止并回收所有角色。

如果平台完全不能提供跨节点 MPI/PMI communicator，只能把每个进程作为互不相关的独立作业启动，那么当前 AFD 不能仅靠修改 shell 脚本运行；需要进一步改造 `vllm-xcpu-plugin` 和 `torch_xcpu` 中依赖 `MPI_COMM_WORLD`、barrier 与 RMA window 的 bootstrap/transport 实现。此类改造完成前，不应把多个独立作业伪装成 A/F ranks，否则通常会在初始化 collective 处挂起。

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
AFD run: afd_... hostfile=/tmp/afd.hostfile
MPI 进程数: 4
```

`mpi_workers.log` 必须为每个 global rank 打印唯一角色。A2/F2 示例为：

```text
AFD topology A=0-1 F=2-3
AFD placement role=A host=node02 global_rank=0 role_rank=0
AFD placement role=A host=node03 global_rank=1 role_rank=1
AFD placement role=F host=node04 global_rank=2 role_rank=0
AFD placement role=F host=node05 global_rank=3 role_rank=1
```

DP2-TP2-EP2 则应有 A global rank 0–3 和 F global rank 4–5；若出现缺失、重复或 world-size mismatch，不能继续以通信初始化成功判定。

`mpi_workers.log` 中应有 F0/F1 的 `weights ready`，随后四个角色均有 `transport ready`。以下为 Qwen3-30B-A3B 的日志格式，省略 PID/时间前缀，权重计数随精度变化：

```text
AF-EP F0 weights ready: loaded=... remote=...
AF-EP F1 weights ready: loaded=... remote=...
AF-EP V7 transport ready: role=ATTN rank=0 hidden_size=2048 topk=8
AF-EP V7 transport ready: role=ATTN rank=1 hidden_size=2048 topk=8
AF-EP V7 transport ready: role=MOE rank=0 hidden_size=2048 topk=8
AF-EP V7 transport ready: role=MOE rank=1 hidden_size=2048 topk=8
```

这里 `rank` 是**角色内编号**，所以 A2/F2 中 MOE rank 0/1 对应 global rank 2/3。ready 行数应等于 `A + F`；例如 DP2-TP2-EP2 应有 4 条 ATTN 和 2 条 MOE。所有 ready 表示 A/F 的 V7 通信窗口已完成初始化；仅看到权重加载完成还不能判定通信可用。完成预热后，终端应出现：

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

**每种精度通过条件**：配置的全部 A/F ranks 初始化完成、所有 DP engine ready、API ready、5 条请求均有正常正文、无 MPI/算子错误、退出及清理成功。三种精度分别填写，不以 BF16 通过替代量化模型结果。

| 最后看到的现象 | 优先检查 |
|---|---|
| 没有 A/F worker 启动，或 mpirun 参数报错 | hostfile、SSH、远端路径、MPI 版本和启动参数 |
| 模型加载报错，没有 F0/F1 weights ready | 模型路径/格式、设备算子、内存及权重加载错误 |
| 权重就绪，但缺少某个角色的 transport ready | 缺失 rank 的日志、MPI 窗口初始化、OSC/UCX、网络 |
| 全部 transport ready 齐全，但 API 未就绪 | head/worker 中的预热、KV cache、算子或容量错误 |
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
