# Qwen3.5-0.8B eager：serial 与 TP2

验证混合 Attention/GDN 模型的加载、分块 prefill、decode、并发流式输出，以及 prefix cache 开启时的状态管理。沿用默认 32K 模型长度、256 batched tokens 和 6 GiB KV cache；不要通过关闭 prefix cache、缩短模型长度或跳过兼容性检查掩盖升级问题。

## 准备

参考 [AGENTS.md](../../AGENTS.md)，在工作区 `.venv` 中构建并安装当前 torch_mcpu、torch_xcpu、torch_mpi_ext、vLLM 和插件，准备 `Qwen/Qwen3.5-0.8B` 权重。常规验证应关闭 MCPU memory protection；若修改了后端构建选项，重新构建依赖库。

## 执行

从开发套件根目录运行。服务测试逐个执行，待上一次清理结束后再启动下一个。

```bash
source .venv/bin/activate
cd vllm_scripts

# serial，单请求与 5 路并发流式请求
./run_vllm_test.sh -e presets/serial/Qwen3.5-0.8B_dp1_tp1_eager.sh
./run_vllm_test.sh -e presets/serial/Qwen3.5-0.8B_dp1_tp1_eager.sh --multi-test

# MPI TP2，单请求与 5 路并发流式请求
./run_vllm_test.sh -e presets/mpi/dense/Qwen3.5-0.8B_dp1_tp2_eager.sh
./run_vllm_test.sh -e presets/mpi/dense/Qwen3.5-0.8B_dp1_tp2_eager.sh --multi-test
```

底层语义回归从根目录运行。不同测试套件分进程执行，避免 mock 或注册状态相互影响：

```bash
export PYTHONPATH="$PWD/vllm:$PWD/vllm-xcpu-plugin"
.venv/bin/python -m pytest vllm-xcpu-plugin/tests/plugin_tests/test_gdn_patch.py -q
.venv/bin/python -m pytest vllm-xcpu-plugin/tests/layers/test_qwen_gdn_linear_attn.py -q
.venv/bin/python -m pytest vllm-xcpu-plugin/tests/plugin_tests/test_platform_plugins.py -k qwen35_hybrid -q
.venv/bin/python -m pytest vllm-xcpu-plugin/tests/fake_triton/test_v031_audit.py -q
.venv/bin/python -m pytest vllm-xcpu-plugin/tests/fake_triton/test_v2_model_state.py -k 'mamba or postprocess' -q
.venv/bin/python -m pytest torch_xcpu/test/ops/custom_ops/test_causal_conv1d_update_xthd.py -q
.venv/bin/python -m pytest torch_xcpu/test/ops/custom_ops/test_chunk_gated_delta_rule.py -k state_contract -q
.venv/bin/python -m pytest torch_xcpu/test/ops/custom_ops/test_fused_sigmoid_gating_delta_rule.py -k 'accepts_row_strided or accepts_packed' -q
```

实际 prefix hit 验证：使用上面的 serial 或 TP2 preset 加 `--no-test` 启动服务，另一个终端从根目录执行下列命令。两种配置分别执行一次，测试完成后停止服务。

```bash
.venv/bin/python vllm_scripts/serve_test/test_qwen35_prefix_cache.py --base-url http://127.0.0.1:14800
```

该测试比较同一长 prompt 冷/热请求的 token IDs，并比较共享前缀但更换问题的请求与新 cache salt 的冷请求；同时要求 Prometheus prefix hit 计数实际增加。保持测试期间没有其他客户端，以便计数归因。

## 通过标准

- 四次端到端测试及清理均返回 0，`run_result.json` 中相应退出码全部为 0。
- serial 使用本地 MP 单 worker，TP2 使用两个 MPI worker，均加载正确模型和 XCPU GDN 算子。
- prefix cache 保持开启，Mamba cache 为 `align`，Attention/SSM 页大小按当前 TP 和 speculative 配置对齐。
- 单请求 HTTP 200，正文非空、可读且与问题相关；5 路多流均正常结束，各流都有正文。
- 请求处理期间没有 GDN integration 被跳过、未注册 kernel、指纹失配、非法访存或引擎异常退出。其他模型可选集成的兼容性警告不能作为那些功能通过的证据；区分主动清理时的退出日志。
- 单测检查输出和最终状态；metadata 覆盖首段/续算、零 draft、混合 spec/prefill、padding、非连续索引、checkpoint，以及 `none`/`align` 模式。
- 清零与状态复制测试检查带页 padding 的存储、不同 segment 大小、SD/DS conv 布局、accepted-token 偏移及原地复制。输出 `out` 测试检查 alias、未写行和历史状态。
- prefill 对比 CPU recurrence、标准 initial/final state 接口与状态池接口，覆盖 TP-local head 数、BF16/FP32、跨 chunk 的变长输入、混合初始状态及未使用槽位。

日志在启动时打印的 `logs/runs/<运行目录>/`。普通响应见 `test.log`，并发响应见 `vllm_task_*.log`。默认输出上限为 16 token，`finish_reason=length` 属正常截断，仍需人工确认正文。

## 后续功能的独立验收

以上是 eager 升级基线，不能替代 DFlash、DFlash2、P/D Disagg 或 compile 的端到端验收。扩展验证时保留零 draft 的 accepted-token 偏移、mixed batch 中 QKV 与 a/b 的相同重排、TP-local state shape，以及 prefix hit / 状态槽复用后的 conv 和 SSM 状态一致性。

prefix cache 的功能验收需在同一服务中重复长前缀请求，确认实际缓存命中并比较 greedy 输出；仅带 `--enable-prefix-caching` 启动不等于命中路径已经覆盖。P/D 还须比较传输前后的完整状态页和后续输出，不能只检查 raw cache 注册成功。
