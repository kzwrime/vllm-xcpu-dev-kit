# DeepSeek-V4.1-Flash：串行 eager / compile 与 DSpark 验证

验收 DeepSeek-V4.1-Flash（mHC、SWA + 压缩 KV、FP4 lightning indexer、Engram、MXFP4 MoE）在 TP=DP=PP=1 下的 eager 与 compile（DYNAMO_TRACE_ONCE + AOT，`--compilation-config {"mode":3}`）推理，以及 DSpark 投机解码。本指南只记录复用步骤与标准；实际运行结果放在对应版本目录或 dev_log。

## 准备

- 按 [AGENTS.md](../../AGENTS.md) 核对构建与内存保护设置（性能与真实权重测试使用 `TORCH_MCPU_ENABLE_MEMORY_PROTECTION=OFF`），记录各仓库 HEAD 与未提交修改。
- 真实权重约 476 GiB（含两张约 98 GiB 的 Engram 表），单进程常驻约 500 GiB，启动前确认空闲内存，不与其他大模型作业并行。
- 两类模型：
  - 真实权重：`~/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master`，preset 为 `presets/serial/DeepSeek-V4.1-Flash_dp1_tp1_eager.sh`。
  - 8 层 dummy：`models/deepseek-ai/DeepSeek-V4.1-Flash-dummy-8-layer-lite`，覆盖 SWA、ratio-2/ratio-1 的 kv/index 源与消费层、candidate 源层、Engram 层和 3 层 DSpark；只用于启动、路径与一致性检查，不判断精度。preset 为 `presets/serial/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp1_tp1_eager.sh`。
- 每个 eager preset 都有同名 `_compile` 版本（serial、dspark）。compile 调试前先移走 `vllm_scripts/vllm_cache/torch_compile_cache`；源层段与 replay 段各生成一份 AOT 产物，再次启动直接加载。
- DSpark preset 位于 `presets/serial/dspark/`（`--speculative-config {"method":"dspark","num_speculative_tokens":5}`），草稿层权重随目标 checkpoint 的 `mtp.*` 提供。
- DSpark Draft Markov 采样运行开关（详见 `torch_xcpu/docs/deepseek_v41/dspark_markov_sample.md`）：
  - `VLLM_XCPU_DSPARK_MARKOV_DTYPE=bf16|fp8|int8`（默认 `bf16`）：设为 `fp8` / `int8` 时在加载期按 `[32, 32]` OCP UE8M0 在线量化并预打包 `markov_w2`；
  - `VLLM_XCPU_DSPARK_MARKOV_FUSED=0|1`（默认 `1`）：设 `0` 手动关闭融合并回退到逐算子循环；
  - `TORCH_XCPU_DSPARK_MARKOV_SAMPLE{,_FP8,_INT8}_IMPL=A0|A1`（默认 `A0`，兼容 32/64 线程 mHC；单算子测试或 `>= 96` 线程环境可设 `A1` 启用 96-PE Mega-Kernel）。
- 精度约定：权重与 KV 存储格式对齐 NVIDIA（MXFP8 32x32 dense、MXFP4 experts；SWA 528 B MXFP8、压缩 288 B NVFP4、indexer 68 B MXFP4），计算一律 BF16 激活 + FP32 累加。

## 执行

从开发套件根目录激活 `.venv`，进入 `vllm_scripts`：

```bash
source .venv/bin/activate
cd vllm_scripts
# dummy：启动 + 单请求
./run_vllm_test.sh -e presets/serial/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp1_tp1_eager.sh
# 真实权重：保留服务，另开终端跑 smoke 客户端
./run_vllm_test.sh -e presets/serial/DeepSeek-V4.1-Flash_dp1_tp1_eager.sh --no-test
python serve_test/test_basic_correctness.py --model ~/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master
# DSpark
./run_vllm_test.sh -e presets/serial/dspark/DeepSeek-V4.1-Flash_dp1_tp1_eager_dspark_5.sh --multi-test
# compile：把上面的 preset 换成 *_compile*.sh，同样执行
./run_vllm_test.sh -e presets/serial/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp1_tp1_compile.sh
```

compile 与 eager 的一致性用同一组 greedy 请求比较（温度 0、逐 token logprob），两边应逐 token 相同。`--multi-test` 默认温度 0.7，只能检查能否跑通，不能用来比较输出。

smoke 客户端依次检查算术、常识、超过 `index_topk × compress_ratio` 的大海捞针检索（indexer 真正做 top-k 与 candidate 两级筛选）、长生成重复度，以及并发 batch 与串行 greedy 结果一致。

长上下文基本项沿用 `serve_test/long_context_accuracy.py`（4k 单请求与 0.5k/4k/8k/16k 并发），参数见 [GLM5.2 串行验证](09-glm52-serial.md)。

单算子测试在 torch_xcpu 中，每个算子一个测试文件，参考实现移植自 vLLM 的 kernel 测试（从 torch_xcpu 源码树之外导入已安装构建；mHC 需要 32 或 64 线程）：

```bash
cd torch_xcpu/test
OMP_NUM_THREADS=64 TORCH_XCPU_CUSTOM_OP_TEST_DEVICES=mcpu pytest -q ops/custom_ops/deepseek_v41
# A0 portable 版本交叉验证（例）
TORCH_XCPU_DEEPSEEK_V41_SPARSE_MLA_IMPL=A0 TORCH_XCPU_DEEPSEEK_V41_INDEXER_TOPK_IMPL=A0 \
  OMP_NUM_THREADS=64 TORCH_XCPU_CUSTOM_OP_TEST_DEVICES=mcpu pytest -q \
  ops/custom_ops/deepseek_v41/test_sparse_mla.py ops/custom_ops/deepseek_v41/test_indexer_topk.py
pytest -q ops/gemm_ops/test_fp8_linear.py -k mxfp8
# DSpark Markov 5 步融合算子（默认 A0；加 *_IMPL=A1 且 OMP_NUM_THREADS=96 测 Mega-Kernel）
TORCH_XCPU_CUSTOM_OP_TEST_DEVICES=mcpu pytest -q ops/custom_ops/test_dspark_markov_sample.py
```

## 通过标准

- 服务日志出现：`Decoder SWA bounded replay`、`Using DeepSeek's nvfp4_ds_mla KV cache format`、`Using MXFP4 indexer cache`、`XcpuGroupedGemmExperts: format=MXFP4`、`Using XcpuMxfp8LinearKernel`。
- profiler 的 mcpu 算子序列中，注意力热点为 `torch_xcpu::deepseek_v41_*`、`fp8_scaled_mm_intel_amx_out`、`einsum_fp8_tgd_grd_to_tgr`、`bf16_linear_intel_amx_out`（router gate、compressor）与 `fused_moe_compute`，没有逐 token 的 aten 回退。
- dummy：请求成功返回 16 个 token；compile 与 eager 输出一致。dummy 只有 16 个专家，router gate 的 N=16 不满足 AMX 的 32 对齐，走通用 `mm_out`（真实权重 384 专家走 BF16 linear）。
- 真实权重：smoke 客户端全部 PASS；长上下文 4k 大海捞针答对。
- DSpark：多流测试通过，接受率非零且输出与关闭投机解码时 greedy 一致。
- compile：日志出现两次 `saved AOT compiled function`（首次）或 `Directly load AOT compilation`（缓存命中），`*.main.cpp` / wrapper 中注意力为 `vllm::xcpu_deepseek_v41_prepare_and_attn`、mHC 为 `torch_xcpu::deepseek_v41_mhc_fused_post_pre_delayed`（以及 `deepseek_v4_hc_prenorm_gemm` + `deepseek_v41_mhc_pre_delayed_finalize`）；greedy 输出、smoke 与长上下文结果与 eager 一致。
