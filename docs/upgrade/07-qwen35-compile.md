# Qwen3.5-0.8B compile：serial 与 TP2

先完成 [eager 基线](06-qwen35-eager.md)，再验证实际 Inductor/AOT 编译、Model Runner V2 采样预热及多流请求。编译失败不能通过添加 `--enforce-eager`、关闭采样预热或跳过 Fake Triton 指纹检查规避。

## 准备

参考 [AGENTS.md](../../AGENTS.md) 构建依赖。更换原生算子实现后重新构建安装 torch_xcpu；常规验证沿用关闭 memory protection 的构建。

开始前重命名保留 `vllm_scripts/vllm_cache` 和 `vllm_scripts/torch_compile_cache_opt`。serial 与 TP2 的首次验证分别使用冷缓存；随后可复用同配置缓存检查缓存加载路径。服务逐个执行，等待清理结束。

## 执行

```bash
source .venv/bin/activate
cd vllm_scripts

./run_vllm_test.sh -e presets/mpi/dense/Qwen3.5-0.8B_dp1_tp2_compile.sh --multi-test
./run_vllm_test.sh -e presets/mpi/dense/Qwen3.5-0.8B_dp1_tp2_compile.sh

# 切换配置前备份并腾空编译缓存。
./run_vllm_test.sh -e presets/serial/Qwen3.5-0.8B_dp1_tp1_compile.sh --multi-test
./run_vllm_test.sh -e presets/serial/Qwen3.5-0.8B_dp1_tp1_compile.sh
```

若机器较慢，可设置 `VLLM_TEST_MAX_WAIT=1000` 延长启动等待时间，不改变模型或 batch 配置。实际缓存命中另按 eager 指南启动同配置 compile 服务并运行 `test_qwen35_prefix_cache.py`。

从根目录验证 MRV2 采样接口和缓存 kernel：

```bash
export PYTHONPATH="$PWD/vllm:$PWD/vllm-xcpu-plugin"
.venv/bin/python -m pytest vllm-xcpu-plugin/tests/fake_triton/test_v031_audit.py -q
.venv/bin/python -m pytest vllm-xcpu-plugin/tests/fake_triton/test_runtime.py vllm-xcpu-plugin/tests/fake_triton/test_provenance.py -q
.venv/bin/python -m pytest vllm-xcpu-plugin/tests/fake_triton/test_installer.py -k jit_monitor -q
```

## 通过标准

- 日志确认 compile 启用、MRV2 runner、正确 TP 数量；冷缓存实际生成并保存编译模型，后续请求执行编译路径。
- 两种配置的普通请求和 5 路流式请求都正常完成，正文非空，`run_result.json` 的运行、清理和总退出码均为 0。
- 采样预热覆盖 logit bias、min tokens、bad words、logprobs 等功能，不能只验证默认采样请求。
- 差分测试覆盖 allowed-token 过滤、bias 累加和 min-token 屏蔽的顺序；仅对允许恢复的请求恢复有限 stop logits，且只在上游定义的无候选边界触发。
- bad-words 测试覆盖已提交输出与多个 draft token 的交界；输入位置 0 为最后一个已提交 token，draft 从位置 1 开始。
- OpenMP 采样回归分别验证 1/2/4 线程；覆盖完整词表、空行/单行、重复禁词及多个禁词命中相同 token。源码映射、等价前提和并行边界见 [验证记录](v0.31/07-qwen35-compile-validation.md#采样-kernel-的-triton-等价性与-openmp-并行)。
- logprob 测试使用带行 padding 的 token-ID 表；不能将其 row stride 当作 top-k 数量。
- 非连续布局的支持范围由算子入口和测试明确限定；保留未知 kernel 与未审查源码变更的 fail-closed 行为。

此基线不替代 DFlash/DFlash2、P/D Disagg、MRV1 或完整模型精度评测。推测解码边界单测通过不等于对应功能已完成端到端验收。
