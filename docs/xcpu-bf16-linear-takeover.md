# VLLM_USE_XCPU_LINEAR 接管 BF16 linear（v0.31）

## 结论

`VLLM_USE_XCPU_LINEAR=1` 时，BF16@BF16 -> BF16/FP32 的未量化 linear 由
torch_xcpu 的 **BF16 linear 后端**（`torch_xcpu/ops_defs/bf16_linear.py`，
显式初始化 + 权重预打包）接管：

- 权重在 `process_weights_after_loading` 阶段一次性预打包
  （后端由 `TORCH_XCPU_BF16_LINEAR_BACKEND` 选择，默认 AMX > oneDNN > ACC）；
- 预打包完成后**释放原始权重**（换成空 Parameter），除非层声明了延迟消费者；
- bias 按 vLLM forward 语义烘焙为 FP32（`skip_bias_add`、RowParallel
  rank>0 不加）；
- Router gate（`GateLinear(out_dtype=float32)`）以 FP32 输出初始化，保留
  FP32 累加；
- **纯 FP32 / FP16、共享权重（embedding / 绑定 lm_head）、shape 不满足后端
  约束的层不接管**，保持原 `torch_xcpu.ops.linear` 路径。

## 改动位置

| 仓库 | 文件 | 内容 |
| --- | --- | --- |
| vllm | `vllm/model_executor/layers/utils.py` | `dispatch_xcpu_bf16_linear()`（预打包+权重释放+keep-raw stash）、`xcpu_raw_linear_weight()` 访问器；`default_unquantized_gemm()` 优先走 `layer._xcpu_bf16_linear`（mcpu 平台的实际执行路径）；`dispatch_cpu_unquantized_gemm()` 的 XCPU_LINEAR 分支同样先尝试 bf16 后端 |
| vllm | `vllm/model_executor/layers/linear.py` | `UnquantizedLinearMethod.process_weights_after_loading()` 新增 mcpu 分支（`is_cpu()` 为 False 的平台在此触发预打包） |
| vllm | `vllm/model_executor/models/qwen3_dflash.py` | qkv_proj 标记 `_xcpu_keep_raw_weight`；`_build_context_kv_buffers()` 经访问器取原始 KV 行、把 fused KV GEMM 也预打包，随后释放原始权重与 stash |
| vllm | `vllm/model_executor/models/laguna.py` / `laguna_dflash.py` | 同上（laguna 的 fused-KV stack 消费原始权重） |
| vllm-xcpu-plugin | `layers/fused_moe/gate_linear.py` | `XcpuGateLinear.forward()` 优先使用 `_xcpu_bf16_linear`（原 `ops.mm(out_dtype=fp32)` 路径在权重被释放后不可用） |

## 权重生命周期（注意点）

1. **普通 linear（qkv/o/gate_up/down/router 等）**：预打包 → 原权重删除
   （空 Parameter）。packed 布局与原始 [N,K] 同为 N*K*2 字节（oneDNN blocked
   布局可能更大），删除原权重后净内存不增。
2. **DFlash qkv_proj**：`_xcpu_keep_raw_weight=True` 时原始权重暂存于
   `layer._xcpu_raw_weight`；模型构建 fused KV buffer（自身也预打包）后置
   None 释放。加载完成到首次 precompute 之间存在临时双份。
3. **embedding / lm_head**（`UnquantizedEmbeddingMethod`，remove_weight=False
   路径）：不预打包、不删权重 —— 原始权重必须保留给 `F.embedding` 与权重
   绑定（tied lm_head 与 embed_tokens 共享同一 Parameter）。
4. **MLA kv_b_proj**：`_cpu_skip_gemm_dispatch` 已有保护，不预打包。
5. **重复 dispatch**（reload 等路径）：`_xcpu_bf16_linear` 已存在时直接复用，
   不二次打包。

## 为什么挂在 default_unquantized_gemm

mcpu 平台的 `McpuPlatform.is_cpu()` 返回 False，因此
`dispatch_unquantized_gemm()` 选择的是 `default_unquantized_gemm`（而
不是 CPU 平台的 `cpu_unquantized_gemm`/`dispatch_cpu_unquantized_gemm`）。
加载期钩子放在 `UnquantizedLinearMethod.process_weights_after_loading()`
的 `elif envs.VLLM_USE_XCPU_LINEAR:` 分支；执行期在
`default_unquantized_gemm()` 内检查 `layer._xcpu_bf16_linear`。

## 验证

单测（layer 级）：bias 烘焙、`skip_bias_add`、RowParallel rank>0 不加 bias、
FP32 输出（~3e-5 相对误差）、纯 FP32 不接管、DFlash fused-KV stash
释放后逐层 forward 仍正确、torch.compile dynamic 追踪冒烟 —— 全部通过；
plugin 侧 gate_linear/fp8/mxfp4 单测 28 passed。

端到端（单次启动内跑 `test_basic_correctness`：单流 4 项 + 并发 batch 4 项；
thinking 模型预算放宽到 640 tokens）。三个 preset 均在服务日志确认
`CPU unquantized GEMM dispatch: using torch_xcpu BF16 linear backend=intel_amx out_dtype=torch.bfloat16`
（TP2 的两个 rank 各自确认）：

| preset | 结果 | 说明 |
| --- | --- | --- |
| Qwen3-0.6B eager | 7/8 | 单流 4/4 全过；batch 3/4，arithmetic 一路 batch 与顺序请求逐字节一致，另一路推理路径分叉（答案仍 391） |
| Qwen3-0.6B compile | 7/8 | 同上；torch.compile 正常追踪打包后的 custom op |
| Qwen3.5-0.8B dflash TP2 compile | 7/8（batch 4/4） | needle/factual/重复率全过，4 路 batch 与顺序请求**逐字节一致**；arithmetic 模型答 511 系模型能力问题（HF 参考实现同样算错，答 491），非本改动引入 |
| GLM-5.2-MXFP4 dp2_tp2_ep compile dflash2_5 | suite test **ALL PASS**（rc=0） | `--suite-test`：single-0.5k/4k、multi-0.5k/4k 全过；4 个 rank 日志均确认 router 走 `out_dtype=torch.float32`、常规 linear 走 `out_dtype=torch.bfloat16`（intel_amx） |

**batch/arithmetic 严格全串相等的失败是既有现象**：stash 本改动后用原始代码
重跑 Qwen3-0.6B eager 基线，batch/arithmetic 0/2 失败（顺序请求与 batch
请求走不同 greedy 推理路径；两路 batch 输出彼此逐字节一致）。原因是 batch
形状改变 GEMM 归约顺序带来的 1-ulp 级数值差异，在 0.6B thinking 模型的
长推理链上被放大——接管后现象不变、频度不升。
