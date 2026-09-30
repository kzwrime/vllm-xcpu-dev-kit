# AFD / MPI 跨平台分层测试指南

基线日期：2026-09-16。适用于不同机器、CPU/NPU 架构、MPI 实现与版本的移植验收。目标是回答：**在哪个配置下，哪项能力已通过，第一处失败在哪个调用/阶段**。不能用一个最终 PASS 代替分层结果。

配套：[结果填写模板](portability_test_result_template.md)、[纯 MPI 探针](../../tests/test_dynamic_mpi/afd_mpi_probe.cpp)、[当前交付说明](README.md)。旧的 `af_ep_status_20260914.md` 属于历史阶段，接口和初始化行为以当前源码为准。

## 1. 测试范围与判定方式

当前 AFD 基础测试是一次 `mpirun` 的固定 MPMD world，A2（DP1、TP2）+ F2（EP2），总共 **4 个 rank**。

暂未测试 2xA2 + F2 的情况，理论支持。

后续会实现“分别启动 MPI 作业再连接”，分为两种情况：

1. **一次性连接**：所有成员连接、确定 rank 映射、开始服务。
2. **动态增量**：在一次性连接的基础上，新增一个 Attn 域，要求原来的全局域先停止服务，然后扩容通信域，然后继续服务，就仿佛这是一次更大规模的一次性连接。

以下是能力分支，不是所有项目必须串行通过的一条等级线。固定 AFD 不依赖动态连接测试；动态测试也不依赖 PyTorch。

| ID | 验证项 | 依赖 | 通过能说明什么 |
|---|---|---|---|
| E0 | 环境、版本、路径、ABI、rank 放置 | 无 | 后续结果可复现且配置可信 |
| P1 | 普通 MPI 启动、Allreduce、Alltoallv | E0 | 固定作业双边/collective 基础可用 |
| P2 | 主线程 Win_create / fence / Put / free | P1 | 固定 communicator 的主机内存 RMA 可用 |
| P3 | 工作线程执行 RMA、完整退出 | P2 | 此线程策略下跨线程调用路径可用 |
| D1 | 两个独立作业 connect/accept、intercomm Alltoallv | P1 | 最小跨作业连接及数据交换可用 |
| D2 | 多个独立作业初始 merge 和通信 | D1 | 一次性组网的多作业基础可用 |
| D3 | 晚启动 A4，8→10 rank 重组再通信 | D2 | demo 的增量 collective 协议可用 |
| A1 | 框架导入、插件逻辑回归 | E0 | Python/插件基本接线可用 |
| A2 | 真实 V7 dispatch/combine 数值、复用与退出 | P3、A1 | 固定 world 的实际 AFD 通信算子可用 |
| A3 | BF16 / FP8 / MXFP4A16 模型端到端 | A2 | 指定模型、量化和放置下固定 AFD 可用 |
| X1 | 动态 communicator 接入真实 V7 / AFD | D2 或 D3、A2 | **当前没有完整实现及标准验收入口** |

每个用例分开填写：

- `data`：数值/协议/请求结果是否正确。
- `exit`：所有 rank 和 launcher 是否正常返回，是否出现 MPI/UCX/PMIx 错误。
- `cleanup`：本次测试的本地、远端 rank、API server、launcher、runtime 作业是否全部结束。

状态统一使用 `PASS / FAIL / BLOCKED / NOT_TESTED / NOT_IMPLEMENTED`。若厂商明确不支持某能力，可记 `UNSUPPORTED` 并附版本与依据；单次卡住不能直接判定“不支持”。`KNOWN_FAIL` 应仍计为失败，只增加已知问题链接。只有相关三项都通过，才能给该用例整体 PASS。预期失败的负向用例按“错误被正确检测且资源已清理”判定。

## 2. E0：环境与部署信息

对**每个实际参与节点**采集，不能只记录提交作业的机器：

| 类别 | 必填信息 |
|---|---|
| 机器 | hostname、物理主机身份、容器/namespace、OS/kernel、CPU ISA、NPU 型号/驱动/固件 |
| MPI | `mpirun` / `mpicc` / `mpicxx` 绝对路径和版本、实际加载的 libmpi、构建参数、线程支持 |
| Runtime | PMIx/PMI、PRRTE/ORTE、UCX/libfabric 版本和组件；调度器、容器、SSH 启动方式 |
| 网络 | rank→物理节点→设备映射、数据网/控制网 IP、网卡和实际 transport；防火墙/路由 |
| 框架 | Python、PyTorch、mpi4py、插件、算子库版本和安装路径 |
| 源码 | 六个仓库 HEAD、工作区差异；未提交/未跟踪代码也需保存或记录校验值 |
| 模型 | 路径、revision、配置/权重身份、量化方法、activation dtype、各 rank 一致性 |
| 运行 | 完整命令、hostfile、相关环境变量、线程等级、超时、开始/结束时间 |

示例采集命令（从仓库根目录；不要把包含凭据的完整环境输出到报告）：

```bash
date -u
hostname
uname -a
lscpu
command -v mpirun mpicc mpicxx python
mpirun --version
python -m pip show mpi4py torch
python -c 'import sys; from mpi4py import MPI; print(sys.executable); print(MPI.__file__); print(MPI.Get_library_version()); print("thread_provided=", MPI.Query_thread())'
for repo in . vllm torch_mcpu torch_xcpu torch_mpi_ext vllm-xcpu-plugin; do
    git -C "$repo" rev-parse HEAD
    git -C "$repo" status --short
done
```

上面的 Python 是单进程检查；还需在实际 MPI 放置下检查每个 rank 的解释器、库路径和版本。用 `ldd` / `readelf` 检查已安装扩展实际链接的 MPI，不要仅凭 `mpirun --version` 判断一致。MPI wrapper 的编译参数查询选项因实现而异，例如 Open MPI 的 `mpicxx --showme`。

换架构必须在目标架构构建对应二进制和算子库，不能复制 x86 `.so` 到 ARM/NPU 环境。换 MPI 后也要检查 mpi4py 和所有 MPI 扩展的 ABI；不假设不同实现、不同大版本或不同构建之间可混用。目标设备适配和算子构建按平台流程执行；当前套件参考顺序是 `torch_mcpu → torch_xcpu → torch_mpi_ext`。性能测试通常关闭 mcpu memory protection；排查越界时另开一组保护模式测试并记录开关。

**线程基线**：当前生产代码使用 mpi4py 默认初始化，本环境默认请求 MULTIPLE，SERIAL 模式大概率能正常工作。必须记录目标环境实际请求与 `provided`，包括 `MPI4PY_RC_*`、`OMPI_MPI_THREAD_LEVEL` 等覆盖项。MPI 可能提供低于请求的等级；SERIALIZED 只允许同一时刻一个线程调用 MPI，MULTIPLE 才允许并发调用。[MPI_Init_thread 说明](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Init_thread.3.html)

历史 SERIALIZED 实验成功不能代替当前默认初始化验收。为定位而改线程等级时，单独记录为另一个配置，不能悄悄替换基线。

## 3. 通用执行与日志约定

建立本次独有、所有节点可访问的目录。源码、venv、模型也应在各节点有相同绝对路径；不共享存储时由部署系统同步并校验。以下命令块按 Bash 编写，后续示例共用这些变量：

```bash
export REPO="$PWD"
export OUT_DIR="$REPO/tests/test_dynamic_mpi/logs/portability_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p "$OUT_DIR/bin"
export MPIEXEC="$(command -v mpirun)"
MPI_COMMON=()
MPI_PLACE=()

run_case() {
    local name="$1" limit="$2" rc
    shift 2
    if timeout --kill-after=10s "$limit" "$@" >"$OUT_DIR/$name.log" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    printf '%s\n' "$rc" >"$OUT_DIR/$name.exit"
    printf '%s exit=%s log=%s\n' "$name" "$rc" "$OUT_DIR/$name.log"
    return "$rc"
}
```

`timeout` 是 GNU coreutils 命令，目标没有时使用调度器等价超时。超时只用于终止和定位，不等于清理成功；记录 124/137 等原始退出码。不要通过 `| tee` 丢失真实退出码。失败后先保存日志和清理，再测试下一种配置。

先同机，再两台物理机，再最终四机放置。Open MPI 示例：

```bash
# 换成真实的四台机器；slots 是 rank 容量，不是 TP/EP 标签。
cat > "$OUT_DIR/hosts" <<'EOF'
hostA slots=1
hostB slots=1
hostC slots=1
hostD slots=1
EOF
MPI_COMMON=(--bind-to none --map-by slot)
MPI_PLACE=(--hostfile "$OUT_DIR/hosts")
# 仅在确实以 root 运行且环境允许时，另加 --allow-run-as-root。
```

这是 **Open MPI 参数**，其他实现请替换为其 hostfile、环境导出和进程放置语法。当前 AFD launcher 使用 `-x`、`--hostfile`、`--wdir` 等 Open MPI 选项；换 MPICH/Intel MPI 等实现时可能先需要 launcher 适配。参数解析失败应归类为“启动器适配”，不能归类为 MPI collective 不支持。

Open MPI 的 `OSC` 是单边通信组件框架，`pt2pt` 是其中基于点对点通信实现 RMA 的组件名称；它不等于所有 MPI 点对点操作。普通 Send/Recv/Alltoallv 成功不保证所选 OSC 能创建窗口。当前仓库默认 Open MPI/UCX 参数只是一组已用过的配置，不是所有版本的通用参数。

推荐配置矩阵：先 MPI 默认组件；失败时再用平台确认支持的 TCP 配置隔离高速网络问题；最后测试部署实际使用的 RDMA/NPU 配置。TCP 通过不能证明 RDMA 或设备内存通信通过。每次改变 OSC、PML、UCX、线程等级都必须产生独立日志。

## 4. P1–P3：不依赖框架的 MPI 探针

在目标平台用目标 MPI wrapper 编译，无 PyTorch、ULFM 或 UCX API 依赖：

```bash
mpicxx -std=c++17 -O2 -Wall -Wextra -pthread \
    "$REPO/tests/test_dynamic_mpi/afd_mpi_probe.cpp" \
    -o "$OUT_DIR/bin/afd_mpi_probe"

# 同机：先保持 MPI_PLACE=()；跨机：使用上一节的放置参数。
run_case P1 90s "$MPIEXEC" "${MPI_COMMON[@]}" "${MPI_PLACE[@]}" -n 4 \
    "$OUT_DIR/bin/afd_mpi_probe" collective multiple 3
run_case P2 90s "$MPIEXEC" "${MPI_COMMON[@]}" "${MPI_PLACE[@]}" -n 4 \
    "$OUT_DIR/bin/afd_mpi_probe" rma multiple 3
run_case P3 90s "$MPIEXEC" "${MPI_COMMON[@]}" "${MPI_PLACE[@]}" -n 4 \
    "$OUT_DIR/bin/afd_mpi_probe" threaded-rma multiple 3
```

每个模式都会验证 Allreduce 与非均匀、含零 count 的 Alltoallv；P2/P3 还验证窗口创建、fence/Put、数据可见性和窗口释放。P3 在主线程建窗口、一个工作线程通信、join 后主线程释放，**没有并发 MPI 调用压力**。参数最后的数字是 RMA 窗口创建/交换/释放循环次数，不是 collective 重复次数。

通过标准：4 个 rank 的身份与放置正确、请求线程等级满足、数据验证通过、每个 rank 都有 `after_finalize` 和 `PASS rank=...`，launcher 返回 0，日志无 runtime ERROR，且没有本次残留进程。探针只测试普通主机内存，不验证 NPU buffer、pin/register、非一致性内存、生产 tensor 生命周期或动态 communicator。

出现卡住时按每个 rank 最后一个 `STAGE` 定位，不能只看日志最后一行（多 rank 日志会交错）：

| 最后完成/停留阶段 | 优先排查 |
|---|---|
| 没有 `after_init` | launcher、路径、PMI/PMIx、网络、MPI 库混用 |
| `thread_level_below_requested` | MPI 构建线程支持或环境覆盖；不是数值错误 |
| `before_allreduce` / `before_alltoallv` | rank 缺失、collective 顺序、基础 transport |
| `before_win_create` | OSC/窗口资源/内存注册/线程策略；核对所有 rank 已到达 |
| `before_fence_open` | collective 参与者或窗口状态、RMA 组件 |
| `after_put_before_fence_close` | RMA 进展、远端可达性、transport |
| `rma_data` 校验失败 | RMA 数据或可见性问题；保存源/目标 rank 信息 |
| `before_win_free` / `before_finalize` | 通信未完成、资源生命周期、runtime 退出路径 |

可另跑 `serialized` 做对照，但不能据此宣布 MULTIPLE 通过。若 P2 通过、P3 失败，优先保留最小线程复现，暂不引入模型。MPI_Finalize 需要完成未结束通信，且与已连接进程的退出协作有关。[MPI_Finalize 说明](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Finalize.3.html)

## 5. D1–D3：独立作业连接与真正晚加入

这些用例测试 MPI 基础能力，**不是当前 AFD launcher 的另一种启动命令**。先在同机独立作业上测试，再跨物理机。每次使用新的共享 state 目录，不能复用旧 port 文件。

### D1：最小两个独立作业

```bash
mpicc -O2 "$REPO/tests/test_dynamic_mpi/intercomm_alltoallv.c" \
    -o "$OUT_DIR/bin/intercomm_alltoallv"
mkdir -p "$OUT_DIR/d1_state"
```

各独立终端使用同一个 `OUT_DIR`、同一 MPI 安装和跨作业 runtime 配置。下面 `JOIN_COMMON` 是**本站已确认支持独立作业 connect/accept 的 launcher 参数数组**；`F_PLACE` / `A_PLACE` 是各自 rank 放置数组，需先配置，不能默认普通 mpirun 就具有跨作业发现能力。

```bash
# 终端 F：2 ranks
run_case D1_F 180s "$MPIEXEC" "${JOIN_COMMON[@]}" "${F_PLACE[@]}" -n 2 \
    "$OUT_DIR/bin/intercomm_alltoallv" f "$OUT_DIR/d1_state/port"

# 终端 A：看到 F 日志的 “F published port” 后再执行，3 ranks
run_case D1_A 180s "$MPIEXEC" "${JOIN_COMMON[@]}" "${A_PLACE[@]}" -n 3 \
    "$OUT_DIR/bin/intercomm_alltoallv" a "$OUT_DIR/d1_state/port"
```

两条命令需要**并发**运行，不要顺序粘贴到同一阻塞终端。A 不会等待 port 文件出现，必须等 F 的发布标记；仅判断文件存在可能读到未写完内容。两端分别出现 `(local=2, remote=3)` 与 `(local=3, remote=2)` 的 Alltoallv PASS，两个作业都正常退出才通过。这里故意使用不相等的两侧 rank 数，检查 intercommunicator 的 remote size 语义。

本站历史 Open MPI 4 实验使用共同 `ompi-server` URI；Open MPI 5 实验使用同一 PRRTE DVM 的独立 `prun` 作业。这些是版本相关的运行条件，不能把不同版本参数直接拼在一起。共享 DVM 不等于共享 MPI_COMM_WORLD；应记录每个作业的 job ID/world size。具体实现参考 [动态 MPI README](../../tests/test_dynamic_mpi/README.md)，目标平台应由其 MPI/runtime 文档确认跨作业发现方式。

### D2 / D3：初始 8 ranks，晚加入后 10 ranks

```bash
mpicxx -std=c++17 -O2 \
    "$REPO/tests/test_dynamic_mpi/m1_v2_functional_single_file.cpp" \
    -o "$OUT_DIR/bin/m1_v2_functional"
mkdir -p "$OUT_DIR/d3_state"
```

该程序只依赖标准 MPI，不依赖 ULFM。用五次独立的 launcher 调用启动以下作业，每个作业 2 ranks：

| 次序 | 程序参数 | 启动条件与预期 |
|---|---|---|
| 1 | `F "$OUT_DIR/d3_state"` | F 发布初始端口并等待 |
| 2 | `A1 "$OUT_DIR/d3_state"` | 初始成员；程序等待自己的端口，超时约 60 秒 |
| 3 | `A2 "$OUT_DIR/d3_state"` | 初始成员 |
| 4 | `A3 "$OUT_DIR/d3_state"` | 初始成员；合并为 F2+A6，共 8 ranks |
| 5 | `A4 "$OUT_DIR/d3_state"` | **必须看到 F 的“初始 super communicator 就绪，等待晚启动 A4”后才启动** |

每个独立终端命令形如（`ROLE` 为表中的一个名字；每个终端设置对应 `ROLE_PLACE`）：

```bash
run_case "D3_$ROLE" 300s "$MPIEXEC" "${JOIN_COMMON[@]}" "${ROLE_PLACE[@]}" -n 2 \
    "$OUT_DIR/bin/m1_v2_functional" "$ROLE" "$OUT_DIR/d3_state"
```

记录 A4 的实际启动时间和初始就绪标记时间，不能用“sleep 固定秒数”代替条件。预期初始 F2↔A6 双向 Alltoallv 数据正确，之后 F2↔A8 数据正确，F 打印 `M1-V2-FUNCTIONAL PASS`，五个作业均正常退出且清理完成。程序请求 SERIALIZED，因此通过不自动证明生产默认 MULTIPLE 策略通过。

D2 的数据证据是初始 8 rank 阶段；若要单独验收 D2 的完整退出，可使用 [`flat_grow_alltoallv.c`](../../tests/test_dynamic_mpi/flat_grow_alltoallv.c) 的初始组网用例，命令见动态测试 README。它在所有预定成员加入后才第一次数据通信，名字中的 grow 不代表晚加入服务。D3 失败时，可记录 `D2 data=PASS, exit=NOT_TESTED`，不要把整个 D2 标成完整通过。

`MPI_Comm_accept` 是旧 communicator 上的 collective，不能只让 F leader 调用而让其余旧成员继续不同的 collective。[MPI_Comm_accept 说明](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Comm_accept.3.html)

定位顺序：端口未发布 → state 路径/权限/Open_port；读不到端口 → 共享目录/发现；connect/accept 卡住 → runtime 作业发现/网络/旧成员参与；merge 后卡住 → rank 顺序、全员 collective、modex；初始通过而 A4 失败 → 晚作业 namespace 发布、新 peer 建连、重组顺序；数据通过但退出失败 → disconnect/free/finalize 协作。

不要直接用 `make all` 判断基础动态 MPI 是否支持：目录中的其他故障实验依赖 `mpi-ext.h` / `MPIX_*`（ULFM），可造成与 D1–D3 无关的编译失败。现成 `scripts/run_scenario.sh` 还绑定特定 Open MPI、PRRTE、节点列表和 ULFM 参数；移植时先审查这些条件。无需故障恢复的本轮测试不要求 ULFM。错误处理器 `MPI_ERRORS_RETURN` 也不等于故障后一定能继续运行。

## 6. A1 / A2：框架接线与真实 V7

当前入口使用 mcpu/xCPU；目标 NPU 后端如更换 import、device、算子注册，必须记录移植差异。后端导入或 kernel 不支持属于设备适配问题，不能归为 MPI 失败。

```bash
source "$REPO/.venv/bin/activate"
python -m pytest -q "$REPO/vllm-xcpu-plugin/tests/af_ep"
python -m pytest -q "$REPO/tests/test_afd_hostfile.py" \
    "$REPO/tests/test_afd_cleanup.py" "$REPO/tests/test_afd_log_lifecycle.py" \
    "$REPO/tests/test_afd_mpi_errors.py"

run_case A2_roundtrip 180s "$MPIEXEC" "${MPI_COMMON[@]}" "${MPI_PLACE[@]}" -n 4 \
    "$REPO/.venv/bin/python" \
    "$REPO/torch_xcpu/test/mpi_custom_ops/test_moe_af_v7_roundtrip_mpi.py"
```

插件当前 5 个回归用例使用 mock，不能替代真实 MPI 数值测试。`test_afd_mpi_errors.py` 的错误注入也不证明真实 peer 故障恢复。

真实 V7 测试检查 top-k、路由、dispatch/combine，使用合成专家结果，并非模型 GEMM 精度验证。保留源码数值容差，不要为“通过”随意放宽。需检查 rank/thread 输出、`PASS: real AF V7 dispatch/combine round-trip`、退出码和 runtime 错误；该 PASS 在 Finalize 之前，单独出现不代表正常退出。

追加矩阵（每次独立启动，环境需导出到**全部 rank**）：

| 用例 | 配置 | 预期 |
|---|---|---|
| 拓扑 | `-n 4 / 6 / 8`，对应 A2F2 / A4F2 / A4F4 | 数值正确；调整 hostfile 容量后运行 |
| 初始化负向 | `XCPU_AF_TEST_INIT_MISMATCH=1` | 全员按预期拒绝配置差异，出现专用 PASS，正常退出 |
| 窗口复用 | `XCPU_AF_TEST_SOAK_EPOCHS=1000` | `PASS: 1000-pass AF V7 window-reuse soak`，退出干净 |
| 退出稳定性 | 基础 roundtrip 新进程重复至少 20 次 | 每次 data/exit/cleanup 均通过，报告失败次数 |

环境变量导出语法随 launcher 变化；Open MPI 可使用 `-x NAME`。此处 `SOAK_EPOCHS` 是历史测试变量名，不表示恢复了生产算子的 epoch 参数。资源不足时将 6/8 ranks 标为 NOT_TESTED，不妨碍明确报告 4 ranks 的能力。

## 7. A3：固定 AFD 的三种精度端到端

先同机四 rank，再跨物理机四 rank。最终放置必须从日志确认：global ranks 0/1 为 A（TP2），2/3 为 F（EP2）。hostfile 描述机器和容量，角色来自 rank 切分；节点名不能作为角色协议。当前 launcher A/F 数量相同，算子 A4F2 通过不代表 launcher 已支持不等数量。

`USER_VLLM_DATA_PARALLEL_RPC_IP` 配置 vLLM DP/RPC rendezvous 可达地址，不是 MPI port_name，也不等于 HTTP API 的监听地址；单独记录 API host/port。当前 MPI hostfile 和 API/RPC 地址仍各有用途。

示例（目标平台确认 launcher 兼容后执行）：

```bash
export QWEN3_30B_A3B_BF16_MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507
export QWEN3_30B_A3B_FP8_MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507-FP8
export QWEN3_30B_A3B_MXFP4_MODEL_PATH=/path/to/Qwen3-30B-A3B-MXFP4A16
export VLLM_MPI_HOSTFILE="$OUT_DIR/hosts"
export USER_VLLM_DATA_PARALLEL_RPC_IP=192.0.2.10  # 替换为真实可达地址
export VLLM_TEST_LOG_DIR="$OUT_DIR/e2e"

# 替换成已在 P1–A2 验证通过的本平台参数；以下仅适用于 Open MPI。
export VLLM_MPI_RUN_ARGS='--bind-to none --map-by slot'
"$REPO/vllm_scripts/e2e/run_afd_crossnode_matrix.sh"
```

矩阵脚本默认带有 Open MPI/UCX TCP 参数，且默认 `UCX_TLS=tcp,self`；需要测试其他 transport 时显式设置实际配置。仅取消某一个 MCA 参数不一定恢复 MPI 默认配置。脚本 `set -e` 会在一个模型失败后停止，后续模型应记 NOT_TESTED；修复/隔离后可直接运行各自 preset，不能把剩余模型记作失败或通过。

每个模型至少记录：权重加载 → A/F 全员窗口初始化 → warmup/profile → API ready → 单请求 → 并发请求 → 清理。当前三项 preset 为：

- `Qwen3-30B-A3B_dp1_tp2_af_ep_eager_v7.sh`
- `Qwen3-30B-A3B-FP8_dp1_tp2_af_ep_eager_v7.sh`
- `Qwen3-30B-A3B-MXFP4A16_dp1_tp2_af_ep_eager_v7.sh`

它们位于 `vllm_scripts/presets/mpi/moe/`。compile 测试选择对应的 `_compile_` 文件，矩阵入口使用 `--mode compile`；具体边界与启动参数见 [A/F compile 适配](compile.md)。MXFP4A16 与其他 FP4 格式不同，报告写全量化方法。BF16 activation 不表示所有权重也是 BF16。

验收分别覆盖：

1. **运行正确性**：请求结束、输出非空、无 rank/MPI/kernel 错误、并发无死锁。
2. **模型正确性**：固定 prompt、seed、temperature=0、token 上限，和同平台可信非 AFD 基线比较；明确数值/文本或任务指标标准。跨架构浮点差异不宜默认要求文本逐字一致。仅“返回了一段文本”不能证明量化精度正确。
3. **边界与稳定性**：短/长输入、小/大 batch、不同路由、重复请求；先小配置跑通，再扩展上线容量。容量不足/OOM 与 MPI 失败分开报告。
4. **生命周期**：正常测试结束、启动失败、用户中断后本次 rank 与 API 全部清理；清理后重新启动一次。固定 AFD 当前通过 launcher 终止，不应描述为已有 STOP/ACK 或无损故障恢复。

日志位于 `VLLM_TEST_LOG_DIR/{runs,success,failed,stopped}/...`，每次独立目录。读取 `run_result.json` 的运行、清理及最终状态，并检查子进程日志；目录名 success 或 launcher exit 0 都不足以排除 UCX ERROR。SIGKILL 遗留目录不会被下一次启动自动扫描清理。

## 8. X1：未来动态 AFD 的补测清单

**D1–D3 全通过仍只能说明 demo 可用。** demo 的 intercommunicator Alltoallv 与现有 V7 的 intracommunicator RMA 数据面不同。以下目前记 NOT_IMPLEMENTED / NOT_TESTED，不能填写现有命令假装已覆盖：

| 范围 | 接入真实 AFD 前必须补测 |
|---|---|
| 一次性连接 | 独立作业合并后的 intra-comm 上运行 P2/P3 和真实 V7；F-first demo rank 顺序转换为实际角色映射；不再隐式依赖原 MPI_COMM_WORLD/launcher 环境 |
| 初始化一致性 | 各作业协商模型/revision、量化、维度、容量、role/group ID、协议版本；不一致时全员退出且无孤立等待 |
| 动态增加 A | 旧成员全部到安全点，完成未结束请求/通信，重建 communicator、窗口、buffer/session 映射；新 A warmup 后才加入调度；校验旧请求和新请求均不丢失/重复 |
| 动态增加 F | 除上述事项外，还需专家分片/权重迁移、路由更新与容量验证，不能沿用仅增加 A 的结论 |
| 生命周期 | 加入超时、失败回滚、重复加入、正常退出、最终断连；标识 communicator 代次以排查旧 buffer/旧 rank 被复用 |

一次性连接主要是启动、拓扑和 communicator 注入的改造，计算路径有复用空间；动态增量则涉及服务状态和资源重建，不能只替换 `mpirun`。全员暂停重组的方案与“其他 A 持续服务、任意 A 独立加入”是不同目标，后者需要额外的数据面/调度设计和测试。

## 9. 清理与问题提交

测试前记录本次 run ID、作业 ID、launcher PID、节点列表；失败或超时后先保存各 rank 最后阶段。使用本次作业的取消机制或 AFD run ID 清理，随后在所有参与节点检查遗留 rank、API、监听端口和设备占用。独立作业测试要清理 F/A1–A4 **所有**作业及本次创建的 runtime daemon；复用公共 DVM 时只结束自己的作业。

不要使用全局 `pkill python/mpirun`。本套件历史非 AFD 清理仍可能按全局进程名执行，测试应安排独占节点/隔离环境，避免与其他作业共存。只杀本地 timeout/mpirun PID 不保证远端已结束；只有逐节点检查完成才能填 cleanup=PASS。

提交问题时附[结果模板](portability_test_result_template.md)、原始 stdout/stderr/exit 文件、版本和库加载信息、实际 rank 放置、完整命令、hostfile、最小失败用例、首次失败 API、失败率，以及“只改一个因素”的对照结果。MPI/PMIx/UCX 错误保留原文，避免只截最后一个 traceback。敏感地址、路径可脱敏，但需保留节点之间的对应关系。

## 10. 本文编写时的验证与历史边界

- 配套纯 MPI 探针已在 node02、Open MPI 4.1.6 上以 `mpirun --allow-run-as-root --bind-to none -n 4` 执行 `collective / rma / threaded-rma`，请求及实际提供均为 MULTIPLE（3），RMA 循环 3 次。三项 launcher exit 0，每项 4 个 rank 均在 Finalize 后打印 PASS，未发现 ERROR/FAIL；检查无探针进程残留。原始日志在本机 `/tmp/afd_doc_validation_20260916/`。这只验证探针本机路径，不是目标平台或物理跨机认证。
- 现有 node02–05 记录来自同一物理宿主的不同网络/PID namespace，不能代替真实多机验证。
- 历史 BF16/FP8/MXFP4 验收与后续默认 mpi4py 初始化复测并非完全相同配置；不能拼接成“当前全部组合均已重测”。当前决策和证据见 [默认初始化记录](mpi_default_20260915.md)。
- 本地 Open MPI/UCX 曾出现数据通过但 Finalize 报错，当前保留为已知问题；详见 [退出错误记录](ucx_finalize_20260915.md)。目标环境应重新测量，不能默认继承通过或失败结论。
