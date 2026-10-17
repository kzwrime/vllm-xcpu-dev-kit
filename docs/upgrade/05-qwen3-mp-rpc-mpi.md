# Qwen3-0.6B MP RPC 与 MPI 跨节点验证

验证 API/head、两个 DP engine、四个外部 worker 的启动、推理和退出。拓扑为 DP=2、TP=2、PP=1，每个 worker 独占一个节点；启动流程依据 [MP_RPC_MPI.md](../../vllm_scripts/MP_RPC_MPI.md)。本指南只定义步骤和通过标准，实际结果另记升级记录；不能将历史成功记录或合成 worker 检查当作本次验收。

## 准备

- head 使用 node03，MPI rank 0–3 分别使用 node04–node07；DP0 engine 位于 node04，DP1 engine 位于 node06。替换主机时同步调整 hostfile 和 head RPC IP。
- 五节点使用同一源码、插件、后端、模型缓存与虚拟环境。记录实际导入路径、vLLM/PyTorch/MPI 版本、git HEAD 和未提交补丁摘要；安装包版本字符串不一定反映当前源码 HEAD。
- 按 [AGENTS.md](../../AGENTS.md) 核对后端构建和内存保护设置。测试前确认节点没有冲突作业，相关端口空闲；核对 head 到 worker 的 SSH 信任链，不关闭主机密钥校验。
- 检查 `GroupCoordinator` 调用与插件 communicator 的构造函数是否兼容，包括 `use_all2all`。MPI 与非 MPI communicator 都要检查；参数必须按上游含义传递，不能仅吞掉未知参数。Dense TP 的构造成功也不能证明 MoE/EP 通信正确。
- 检查测试脚本的清理范围。现有 MPI launcher 使用进程名匹配发送信号，可能影响同节点其他作业；共享节点应使用经过审查的本次作业 PID/标记清理方式。额外退出检查仅操作本次作业。

## 执行文档中的请求矩阵

在 head 节点、开发套件根目录执行。以下 IP 需替换为实际可路由地址；所有节点须能访问同一绝对路径。

```bash
export MP_RPC_DEVKIT_ROOT="$PWD"
source .venv/bin/activate
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$MP_RPC_DEVKIT_ROOT/.venv/lib/libiomp5.so"
cd vllm_scripts
mkdir -p logs/tmp

cat > logs/tmp/mp_rpc.hostfile <<'EOF'
node04 slots=1
node05 slots=1
node06 slots=1
node07 slots=1
EOF

cat > logs/tmp/mp_rpc_overlay.sh <<'EOF'
source "$MP_RPC_DEVKIT_ROOT/vllm_scripts/presets/mpi/dense/Qwen3-0.6B_dp2_tp2_${MP_RPC_MODE}.sh"
if [ "$MP_RPC_MODE" = compile ]; then
    unset USER_VLLM_EAGER_OR_NOT
fi
export USER_VLLM_DATA_PARALLEL_RPC_IP=172.30.3.13
export VLLM_LOOPBACK_IP=$(hostname -I | awk '{print $1}')
export USER_VLLM_MAX_NUM_BATCHED_TOKENS=32
export VLLM_CACHE_ROOT="$MP_RPC_CACHE_DIR/vllm"
export TORCHINDUCTOR_CACHE_DIR="$MP_RPC_CACHE_DIR/inductor"
EOF

export VLLM_MPI_HOSTFILE="$PWD/logs/tmp/mp_rpc.hostfile"
export VLLM_TEST_MAX_WAIT=600
export VLLM_ENGINE_READY_TIMEOUT_S=600
export OMPI_MCA_plm_rsh_args='-o StrictHostKeyChecking=yes'
export VLLM_MPI_RUN_ARGS='--allow-run-as-root --bind-to none --map-by slot -x PATH -x VIRTUAL_ENV -x LD_PRELOAD -x VLLM_RUN_LOG_DIR -x VLLM_ENGINE_READY_TIMEOUT_S -x MP_RPC_DEVKIT_ROOT -x MP_RPC_MODE -x MP_RPC_CACHE_DIR -x MP_RPC_RUN_ID'

export MP_RPC_MODE=eager
export MP_RPC_RUN_ID="mp_rpc_single_$(date +%Y%m%d_%H%M%S)"
export MP_RPC_CACHE_DIR="$PWD/logs/tmp/$MP_RPC_RUN_ID"
./run_vllm_test.sh -e logs/tmp/mp_rpc_overlay.sh --launcher mpi --test-timeout 600
single_rc=$?
echo "single_exit_code=$single_rc"
```

核对日志及五节点清理后，使用新的作业标记执行五路流式请求：

```bash
export MP_RPC_RUN_ID="mp_rpc_multi_$(date +%Y%m%d_%H%M%S)"
export MP_RPC_CACHE_DIR="$PWD/logs/tmp/$MP_RPC_RUN_ID"
./run_vllm_test.sh -e logs/tmp/mp_rpc_overlay.sh --launcher mpi \
    --multi-test --multi-test-max-tokens 96 --test-timeout 600
multi_rc=$?
echo "multi_exit_code=$multi_rc"
```

确认清理完成后，使用不存在的新缓存目录执行 compile 普通请求；日志必须确认未启用 eager。模型 runner 切换时也要更换冷缓存。

```bash
export MP_RPC_MODE=compile
export MP_RPC_RUN_ID="mp_rpc_compile_$(date +%Y%m%d_%H%M%S)"
export MP_RPC_CACHE_DIR="$PWD/logs/tmp/$MP_RPC_RUN_ID"
test ! -e "$MP_RPC_CACHE_DIR"
./run_vllm_test.sh -e logs/tmp/mp_rpc_overlay.sh --launcher mpi --test-timeout 600
compile_rc=$?
echo "compile_exit_code=$compile_rc"
```

出现公共启动阻塞时，记录哪些矩阵项未执行，先定位兼容性问题；不要以重复相同失败或切换 eager 代替 compile 验收。额外 MoE/EP 覆盖使用源文档的对应 preset，需要独立核对模型、EP 拓扑和 all-to-all 后端。`VLLM_CPU_USE_MPI=0` 的 gloo 对照只验证另一条通信路径，不代表 MPI 算子通过。

## 推理与新版本信息检查

- MPI coordination 给出四个不同节点的 rank/IP；两个 DP engine 位于各组 rank 0 节点，head 本地 engine 数为 0。
- 四个 worker 均使用当前源码和正确的本机 `local_rank`，每节点一个 worker 时应为 0。两组 rendezvous 使用各自 rank 0 的 TCP 地址，不依赖跨节点共享 `/tmp`；配置中的 worker 本机 IP 不被 executor IP 覆盖。
- 检查 READY、广播和响应队列连接完成。RPC 数据传输应使用 TCP handle；共享队列类名、逻辑 local rank 或 `--distributed-executor-backend mp` 不能单独证明传输方式。
- 核对实际 MCPU/XCPU 后端、模型 runner、eager/compile、异步调度和 TP/DP 配置。记录新增握手协议及 executor 支持范围，不能推断旧 worker 可混用或所有上游新能力都已适配。
- `/health`、`/version`、`/v1/models` 返回符合预期；版本接口与实际源码记录相互核对。开发信息接口 `/server_info` 可能默认未注册，404 不等价于服务故障；不得为采集信息擅自启用全部开发控制接口，可从启动日志确认配置。
- 普通请求 HTTP 成功，正文非空、可读且与问题相关，completion tokens 大于 0。默认 16 token 的 `finish_reason=length` 可以接受，但不是精度评估。
- 五路流式请求均有实际 chunk 和非空正文，结束原因符合 token 预算，无请求丢失、重复输出或 scheduler 异常。结合逐 DP 日志或 `/metrics` 中的请求计数证明两个 engine 都处理过请求。
- compile 必须检查实际图捕获、C++/共享库与 AOTI 产物，区分完整编译、Python callback 和 fallback。冷缓存编译失败时，请求和产物验收均记为未通过。

## 额外退出验证

为每次作业记录五节点的进程树、PID、创建时间和监听端口，并保留信号发送时间。用作业标记核对进程身份后再发送信号，避免 PID 复用误伤。

| 场景 | 操作与通过标准 |
| --- | --- |
| 测试脚本自动结束 | 普通及多流请求完成后记录 wrapper 与 `run_result.json` 的退出码；在五节点独立确认 worker、engine、head、MPI 子进程退出，固定及本次动态端口释放。 |
| 服务正常关闭 | 以 `--no-test` 启动并完成请求后，向本次 head 服务进程发送 SIGTERM；观察 engine shutdown、worker 收到退出命令、资源清理及进程返回。记录哪些角色依赖 launcher 最终回收，不能把强制清理算作 executor 自主退出。 |
| executor 正常关闭 | 独立启动一轮，完成请求后向两个 DP EngineCore 发送 SIGTERM，检查两组 shutdown 广播、四个 worker 的退出以及 MPI 作业返回。分开评价 worker 执行循环结束、MPI finalize 和 head 的退出，不能由其中一个推断其他层。 |
| worker 等待 executor/配置 | 未启动 executor 时，使用短 `VLLM_ENGINE_READY_TIMEOUT_S` 启动真实 RPC worker；应在连接等待超时后非零退出。另起一轮，以受控 TCP 服务接收 worker 的 HELLO、暂不发送配置，再向该 worker 发送 SIGTERM，核对退出时延及残留。此项没有模型、MPI collective 或 ZMQ READY 屏障。 |
| worker 初始化失败 | 确认 FAILURE 到达对应 executor、worker 非零退出、launcher 检测失败；检查其余节点及 head 是否完成回收。自然出现的初始化错误可作为此项证据，但不算推理通过。 |
| READY 后 worker 崩溃 | 服务就绪并完成请求后，仅对指定 worker 发送 SIGKILL；在预先约定的观察窗口内检查 MPI、engine、head 和请求结果，随后只清理本次作业。必须同时检查有超时限制的推理请求和 `/health`，不能仅凭健康端点 200 判断后端仍可用。明确退出是 MPI 作业中止还是 executor 自主检测。 |
| executor 崩溃 | 单独启动一轮，确认请求成功后终止一个 EngineCore；检查对应 worker、另一个 DP 组和 head。无持续存活通道时可能需要外部清理，必须保留这一限制。 |

将请求阶段、主动退出阶段和强制回收阶段分开评价。即使最终无残留，`EngineDeadError`、MPI/PMIx 段错误、超时、强杀或资源警告仍要记录；`cleanup_exit_code=0` 不能覆盖这些异常。进程崩溃测试也不等价于网络分区、collective 卡死或设备故障测试。

默认 MP 回归在独立进程中运行，按 [串行 eager 指南](01-qwen3-serial-eager.md) 验证基础行为；涉及本地 TP worker 创建时还需 TP2 回归。同进程从 RPC executor 切回普通 MP 的传输开关兼容性，应单独检查，不混入文档中的正常 MPI 拓扑结论。

## 结果记录

在本地升级记录中分别登记通过、失败、未执行及仅接口/合成检查的项目，并链接本次 `logs/runs/`、请求日志、逐 rank 日志、退出快照及编译产物。协议启动成功不能替代模型推理；请求成功不能替代退出、MoE/EP 或性能验证。
