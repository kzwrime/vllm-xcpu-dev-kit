# Qwen3-0.6B Isolated DP 推理

验证 API/head 与 DP rank 0 位于不同节点、使用不同 IP 时，DP=2、TP=1、PP=1 的 eager 推理。启动方式依据 [IsolateDP.md](../../vllm_scripts/IsolateDP.md)。本指南是验收步骤，不表示已完成验收；实际结果另记升级记录。

## 准备

- 先完成目标补丁的兼容性审查，确认保留既有 internal、hybrid、external LB 的引擎管理语义，再启动服务。
- 本机与两台 DP 主机使用一致的源码、插件、后端和虚拟环境，均能读取完整模型缓存。按 [AGENTS.md](../../AGENTS.md) 核对内存保护及后端构建；确认各节点实际导入的源码路径与版本。
- 本机可免交互 SSH 到 DP 主机；三个节点之间相关监听地址及端口可达，测试端口空闲。默认 DP 主机为 `wzk-vllm-xcpu-03-c8,wzk-vllm-xcpu-03-c7`，API/head 在当前节点。若使用别名，先确认解析至预期 IP，并保持 SSH 主机密钥校验。
- 默认使用同一共享目录。自定义布局时设置 `ISOLATE_DP_REMOTE_DIR` 和 `ISOLATE_DP_REMOTE_VENV_BIN`，并确保远端能写入本机脚本传入的日志绝对路径。
- 核对 IP 自动发现结果；有多网卡时显式设置 `ISOLATE_DP_API_IP`、`ISOLATE_DP_DP_HOSTS`、`ISOLATE_DP_DP_IPS`。API/head IP 必须与 DP rank 0 IP 不同，才能覆盖地址分离行为。

## 执行

从开发套件根目录运行。普通请求结束并确认三节点清理完成后，再执行多流测试：

```bash
source .venv/bin/activate
cd vllm_scripts

./run_isolate_dp_test.sh \
  -e presets/isolate_dp/Qwen3-0.6B_dp2_tp1_eager.sh
single_rc=$?
echo "single_exit_code=$single_rc"

# 核对普通请求结果、日志及三节点进程清理后再执行。
./run_isolate_dp_test.sh \
  -e presets/isolate_dp/Qwen3-0.6B_dp2_tp1_eager.sh \
  --multi-test --multi-test-max-tokens 96
multi_rc=$?
echo "multi_exit_code=$multi_rc"
```

需要调整预热或启动等待时间时，使用 `USER_VLLM_MAX_NUM_BATCHED_TOKENS`、`VLLM_TEST_MAX_WAIT`，并核对远端实际值；本地环境变量不会自动经 SSH 转发。`--no-test` 仅检查启动和 `/v1/models`，不能代替请求验收。

## 通过标准

- 启动日志确认 API/head 本地引擎数为 0，两台远端分别运行 headless DP rank 0、1，TP/PP 均为 1，使用 MCPU/XCPU 后端和 eager 模式。
- 三个角色均传入 `--data-parallel-address=<DP rank 0 IP>`、`--data-parallel-rpc-ip=<API/head IP>`。DP 通信初始化仍使用 DP master；远端握手和 API 侧 ZMQ 服务使用 head 地址。无错误绑定、握手超时、通信异常或 traceback。
- `/v1/models` 返回目标模型，两个 DP engine 均完成启动握手。结合逐 rank 日志或请求计数确认两者实际参与处理；仅收到一次 HTTP 200 不足以证明两个 rank 均处理过请求。
- 普通请求成功，正文非空、可读且与提示词相关，`usage.completion_tokens > 0`。默认 16 token 允许 `finish_reason=length`，不能将空正文当作通过。
- 五路流式请求均正常结束，有实际 chunk，正文非空、可读且与各自提示词相关。
- 两次脚本退出码均为 0；根据本次 PID 文件在本机和相应远端核查进程及其子进程已退出，监听端口释放。脚本清理会忽略 kill/SSH 错误，退出码不能单独证明清理成功。

分开核对请求阶段和主动关闭阶段的日志。关闭时若出现 `EngineDeadError`、强制终止或资源回收警告，应保留时间顺序和进程/端口检查结果，不能笼统记为全程无异常。正文可读且相关只是服务冒烟标准；发现事实错误需记录，不能报告精度验收通过。

每次运行的服务、SSH、请求日志和 PID 文件在 `logs/isolate_dp/<run-id>/`；多流逐任务日志在运行目录的 `vllm_task_*.log`，下一次运行前归档。此脚本不生成串行测试使用的 `run_result.json`，需单独记录退出码与清理核查结果。

## 配置兼容性验收

地址分离测试之外，还需验证以下配置契约；可先做配置/单元检查，实际跨节点行为另做端到端验证：

- 未传 `--data-parallel-rpc-ip` 时，RPC 地址回退到 `--data-parallel-address`，保留旧命令行为。
- 直接构造 `ParallelConfig` 或使用环境变量解析 master 地址时，未设置的 RPC 地址也应回退到最终 master；改变 RPC 地址不影响计算图/DP 配置哈希，改变 collective 参数仍应影响哈希。
- internal LB 的 `local_engines_only=False`；hybrid 和 external LB 为 `True`，不能仅因两个 IP 相等而改变 LB 模式。
- 多节点 hybrid LB 示例采用 DP=4、每节点本地引擎数 2、起始 rank 分别为 0 和 2；不传新 RPC IP 时，非零起始 rank 能通过启动断言，前端仍仅管理本地引擎。应使用适合该部署模式的模型和启动方式；IsolateDP wrapper 本身不覆盖此场景。

以上检查不替代 MoE、TP/PP 跨节点、编译、弹性扩缩容、精度或性能验证。
