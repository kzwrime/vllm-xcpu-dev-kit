# Qwen3-MoE：MPI v5/v6 eager 与 compile 验证

使用 `Qwen/Qwen3-30B-A3B-Instruct-2507` BF16，DP=2、TP=2、EP=4、PP=1，分别验收 `mpi_alltoallv_v5` 和 `mpi_alltoallv_v6`。本指南只记录复用步骤与标准；实际运行结果放在对应版本目录。

## 准备

- 按 [AGENTS.md](../../AGENTS.md) 核对构建与内存保护设置，记录各仓库 HEAD、未提交修改、模型和环境版本。
- 四个 MPI rank 必须使用相同代码、原生库和模型权重。单节点结果不代表跨节点验收；跨节点 hostfile、RPC 地址与清理检查参考 [MPI 指南](05-qwen3-mp-rpc-mpi.md)。
- 确认测试端口空闲、节点没有冲突作业。保持默认模型长度、prefix caching、chunked prefill 与 KV cache 配置；若修改 token budget，记录实际值。
- 确认自定义 Prepare/Finalize factory 在 CLI/config 创建前注册；不能仅放宽 CLI 参数而退回默认 allgather/reduce-scatter。

## 执行

从开发套件根目录激活 `.venv`，再进入 `vllm_scripts`：

```bash
source .venv/bin/activate
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$VIRTUAL_ENV/lib/libiomp5.so"
cd vllm_scripts
export VLLM_TEST_MAX_WAIT=1000
# root 环境需要 --allow-run-as-root；跨节点另设 VLLM_MPI_HOSTFILE。
export VLLM_MPI_RUN_ARGS='--allow-run-as-root --bind-to none --map-by slot'
./run_vllm_test.sh \
  -e presets/mpi/moe/Qwen3-30B-A3B_dp2_tp2_ep_eager_alltoallv_v5.sh \
  --multi-test --multi-test-temperature 0 --multi-test-max-tokens 32 \
  --test-timeout 600
```

分别执行 v5/v6 的 eager 普通请求和多流请求。compile 对应 preset 为 `Qwen3-30B-A3B_dp2_tp2_ep_compile_alltoallv_v5.sh` 和 `Qwen3-30B-A3B_dp2_tp2_ep_compile_alltoallv_v6.sh`。

compile 必须使用新缓存：备份现有 `vllm_cache` 和 `torch_compile_cache_opt`，或通过临时 overlay 在 source preset **之后**设置 `VLLM_CACHE_ROOT`、`TORCHINDUCTOR_CACHE_DIR`。overlay 同时执行 `unset USER_VLLM_EAGER_OR_NOT`，避免继承 eager 环境。记录实际路径，并检查四个 rank 均未启用 eager。

## 通过标准

- 四个 worker 加载真实 BF16 权重，EP 分区均匀；factory 为对应 V5/V6，专家计算为 XCPU grouped GEMM，`use_ep=True`、`topk_reduce=True`。核对 TP2 下 sequence parallel 的生效情况。
- 普通请求及五路并发流式请求成功，正文非空可读，结束原因符合 token 预算；核对两个 DP engine 都处理过请求，包含不均衡调度时的空闲 rank 同步。
- v5/v6 使用同一 prompt、greedy 参数和权重比较输出。不同路由/归约顺序的浮点差异应单独调查；少量文本一致不替代完整精度评测。
- compile 完成冷缓存图捕获和 AOTI 共享库生成，再完成请求。检查产物中的 dispatch、grouped GEMM、combine 调用，区分直接 shim、proxy callback 与整段 Python 执行。
- `run_result.json` 的运行、清理、总退出码均为 0；独立确认本次 head、engine、worker、MPI 子进程和端口释放。清理信号后的 `EngineDeadError` 与请求阶段错误分开记录。

这里不覆盖 FP8/MXFP4、EPLB、AF-EP、DBO、其他通信版本、真实 NPU 或性能对比；这些配置需要独立验收。
