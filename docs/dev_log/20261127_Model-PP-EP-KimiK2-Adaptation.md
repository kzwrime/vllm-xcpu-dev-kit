---
date: 2026-11-27
type: feature
status: completed
scope: multimodal-asr, llama3, mpi-pp, moe-pp-ep, kimi-k2-acc-dequant
author: agent
related:
  vllm-xcpu-dev-kit:
    branch: dev/0.31
    commit:
      - "aea103b"
      - "c4bbfc1"
      - "15378e3"
      - "62c3837"
      - "c140f5c"
  vllm:
    branch: dev/0.31
    commit:
      - "e4c1fafbd0"
  torch_xcpu:
    branch: dev/0.31
    commit:
      - "8d8ff07"
      - "a8be0e4"
  torch_mpi_ext:
    branch: mpi_dev
    commit:
      - "7dcd9e8"
  vllm-xcpu-plugin:
    branch: dev/0.31
    commit:
      - "482e8f6"
      - "9a92d20"
      - "3f1e80f"
---

# 多模态语音、LLaMA3、正式 MPI PP、MoE DP+TP+PP+EP 及 Kimi K2 FP8 反量化执行路径适配日志

## 1. 背景及分析

围绕真实 NPU Infra 部署与架构验证需求，本次开发在 `vllm-xcpu-dev-kit-0.31` 多仓库协同环境下完成了五项核心能力适配与端到端闭环验证：

1. **多模态语音输入模型评估与最简适配（Qwen3-ASR-0.6B）**：
   - 评估候选语音模型后，选择 `Qwen/Qwen3-ASR-0.6B` 作为最简切入点：音频预处理在宿主侧通过 `soundfile` + 特征提取完成，主干结构复用已适配的 Qwen3 文本解码器与 RoPE/RMSNorm/Attention 路径；主要算子缺口仅为音频编码器 2D 下采样卷积触发的 `aten::convolution_overrideable` 在 `PrivateUse1` (`mcpu`) 上未实现。
2. **LLaMA3 架构模型支持（Llama-3.2-1B-Instruct）**：
   - 检查 `vllm-xcpu-plugin` 后确认 `Llama3RotaryEmbedding` 已注册 `@Llama3RotaryEmbedding.register_oot`，缺失的是针对 `llama3` RoPE 的专门单元测试及可重复验证的串行 Eager/Compile 与分布式 TP2 预设脚本。
3. **Qwen3 Dense `DP+PP+TP` 及正式 MPI Send/Recv 支持（Qwen3-0.6B, DP=2, PP=2, TP=2）**：
   - 原有分布式能力缺少正式的流序感知 MPI P2P 原语与 Pipeline Parallelism 调度打通，需从 `torch_mpi_ext` 底层 `MPI_BYTE` 点对点原语（`send` / `recv_` / `broadcast_`）到 vLLM `DeviceCommunicatorBase` / `GroupCoordinator` / `PPHandler` 再到 `vllm-xcpu-plugin` `CpuMPICommunicator` 构建完整正式 PP 数据面与控制面通道。
4. **MoE 模型 `DP+TP+PP+EP` 混合并行运行（Qwen3-30B-A3B-FP8, DP=2, PP=2, TP=2, EP=4）**：
   - 在正式 MPI PP 打通后，需验证 vLLM 按 PP Stage 切分 `_EP` 通信域（8 MPI rank 划分为 2 个 PP stage，每个 stage 内部以 `DP=2 × TP=2 = EP=4` 运行 `mpi_alltoallv_v5`）在并发多流请求与空闲 DP `execute_dummy_batch()` 协同下的正确性。
5. **Kimi K2 以原生精度反量化 → BF16 Linear → 量化路径运行（Dummy Weights）**：
   - 根据 Kimi K2 架构（`model_type="kimi_k2"`，结构同构于 DeepSeek-V3：`hidden_size=7168`、`64` 注意力头、Dense MLA 含 `kv_lora_rank=512` / `qk_rope_head_dim=64`、`first_k_dense_replace=1`、`384` routed experts + `1` shared expert、`num_experts_per_tok=8`，权重采用 `fp8` `[128, 128]` blockwise scale），确定在保持激活为 BF16（W8A16）的基础上：
     - 于 `torch_xcpu` ACC FP8 后端引入初始化环境变量 `TORCH_XCPU_ACC_FP8_DEQUANT_BF16_LINEAR` 切换全局 C++ 标志，在每次 `AccFp8LinearKernel` / `AccFp8MoeGroupedGemm` 前向执行**分离的临时解量化 → `AccBf16LinearKernel`**，之后结合已有 FP8 KV Cache / `quant_fp8` 路径；
     - 为避免 61 层完整大模型仅 dummy 验证时占用过高主机内存并确保算子形状真实，构建 3 层同构缩微模型 `Kimi-K2-Instruct-dummy-3-layer`（1 层 Dense + 2 层 MoE，隐藏维与专家数全尺寸保留）；
     - 补齐 vLLM + `vllm-xcpu-plugin` 中 Dense MLA 在 `mcpu` 上的后端支持（`XCPU_MLA_DENSE`）、`DeepseekScalingRotaryEmbedding` OOT 注册以及 `torch_xcpu.ops.unified_attention` 对无 `logical_topk` Dense MLA 的分支路由。

---

## 2. 方案及效果评估

### 2.1 分阶段方案设计

| 任务 | 涉及仓库与核心修改 | 说明与设计亮点 |
| --- | --- | --- |
| **1. LLaMA3 模型** | `vllm-xcpu-plugin` (`482e8f6`)<br>`.` (`aea103b`) | 在 `tests/kernels/core/test_pos_encoding.py` 新增 `test_llama3_rotary_embedding`（覆盖 `bf16`/`fp32` 与 `neox`/`gptj`）；新增 `Llama-3.2-1B-Instruct` 的 `dp1_tp1_eager`、`dp1_tp1_compile` 与 `dp1_tp2_eager` 可移植预设。 |
| **2. 多模态语音 Qwen3-ASR-0.6B** | `torch_xcpu` (`8d8ff07`)<br>`.` (`c4bbfc1`) | 在 `torch_xcpu_impl` 新增三层架构算子 `conv2d_impl<T>`（`OperatorId::Convolution` `A0`/`B0`，L2 分块无全局 `im2col` 缓冲，原生支持 1D/2D、分组与空洞卷积及 `mcpu::convolution` 计时/内存保护守卫），非 1D/2D 或转置分支走流序感知 CPU facade fallback（无宿主同步与拷贝）；补充 `test/ops/test_convolution.py`（31 项单测）；`install_deps.sh` 新增 `soundfile`；扩展 `vllm_scripts/serve_test/test_multimodal.py` 支持 `AUDIO_CASES`（`audio_url` 与 `input_audio` 双协议），加入 12s 16-bit PCM 公共领域英语测试音频与 ASR 预设。 |
| **3. 正式 MPI Send/Recv PP** | `torch_mpi_ext` (`7dcd9e8`)<br>`vllm` (`e4c1fafbd0`)<br>`vllm-xcpu-plugin` (`9a92d20`)<br>`.` (`15378e3`) | `torch_mpi_ext` 新增流序感知 `MPI_BYTE` `send`、`recv_`、`broadcast_`（支持全 dtype、`> 1 GiB` 分块与 0-byte 同步，注册 Dynamo/AOTAutograd `_EffectType.ORDERED` 副作用防 DCE，并修复 `batch_isend_irecv` 的 `KernelPointerMemoryGuard` 花括号构造）。`vllm` 在 `DeviceCommunicatorBase` 新增 `supports_custom_tensor_dict` / `supports_custom_pp_broadcast` 委托，在 `GroupCoordinator`（含纯 metadata `MPI_Isend` 句柄回收与 `destroy()` 排空）、`PPHandler._pp_broadcast` 与 MRV1 `GPUModelRunner` 路由委托。`vllm-xcpu-plugin` `CpuMPICommunicator` 使用独立 `mpi_meta_comm`、`mpi_bcast_comm` 与专用 `_p2p_send_stream`/`_p2p_recv_stream` 正式实现 `send` / `recv` / `isend_tensor_dict` / `irecv_tensor_dict` / `broadcast` / `pp_broadcast`。 |
| **4. MoE DP+TP+PP+EP** | `.` (`62c3837`) | 基于 Stage 3 正式 MPI PP 与按 PP stage 隔离的 EP 通信域，新增可移植预设 `Qwen3-30B-A3B-FP8_dp2_pp2_tp2_ep_eager_alltoallv_v5.sh`（8 进程：`DP=2, PP=2, TP=2, EP=4`）。 |
| **5. Kimi K2 反量化 → BF16 → 量化** | `torch_xcpu` (`a8be0e4`)<br>`vllm-xcpu-plugin` (`3f1e80f`)<br>`.` (`c140f5c`) | **torch_xcpu**：`AccFp8LinearKernel` 新增由 `TORCH_XCPU_ACC_FP8_DEQUANT_BF16_LINEAR` 控制的 `std::atomic<bool> g_acc_fp8_dequant_bf16_linear` 开关及 C++/Python accessor，开启后在 `launch_scaled_mm`、`scaled_mm_wrapper` 与 `scaled_mm<HasBias>` 中分配临时 `[N/32, K, 32]` packed BF16 缓冲区执行 `acc_fp8_dequant_packed_bf16`（支持 FP32 与 UE8M0 scale），再调用 `AccBf16LinearKernel::linear_wrapper` / `linear`（同步覆盖 `AccFp8MoeGroupedGemm` 与 `MoeGroupedGemmWithIndex`）；同步修复 `ops.py:unified_attention` 仅在 `logical_topk is not None` 时路由到 `sparse_mla_fp8`，使 Dense MLA `fp8_ds_mla` 命中 `unified_attention_mla_fp8_ds`。<br>**plugin**：新增 `vllm_xcpu_plugin/attention/dense_mla.py`（`XcpuMLADenseBackend` / `Impl` / `Metadata` 含 `num_decode_tokens` 与 PCP/Spec 校验）及 `tests/plugin_tests/test_dense_mla.py`，在 `platform.py` 对 `use_mla and not use_sparse` 返回 `XCPU_MLA_DENSE`，在 `mla.py` 支持 `q_lora_rank` 可选分支（兼容 DeepSeek-V2-Lite 与 V3/Kimi K2）并使用 `_XcpuNoOpMLAPrefillBackend` 绕开 `flash_attn` 检测，在 `layers/rotary_embedding.py` 注册 `XcpuDeepseekScalingRotaryEmbedding`。<br>**模型与预设**：建立精简版 `models/moonshotai/Kimi-K2-Instruct-dummy-3-layer/`（移除未使用的 `modeling_deepseek.py`，补充 MIT `LICENSE`）及 `dp1_tp1`（eager/compile）、`dp2_tp2_ep` 预设。 |

### 2.2 关键验证结果汇总

- **LLaMA3**：
  - `pytest -v tests/kernels/core/test_pos_encoding.py -k llama3`：`12 passed`；`./scripts/check.sh` 通过。
  - 端到端：`Llama-3.2-1B-Instruct_dp1_tp1_eager.sh --multi-test`、`dp1_tp1_compile.sh`、`dp1_tp2_eager.sh` 均 exit 0 通过。
- **Qwen3-ASR-0.6B**：
  - `pytest -v test/ops/test_convolution.py`：`31 passed`（含 `TORCH_MCPU_ENABLE_MEMORY_PROTECTION=ON` 验证）；`ruff check` 与 `clang-format --dry-run --Werror` 通过。
  - 端到端：`Qwen3-ASR-0.6B_dp1_tp1_eager_multimodal.sh --multimodal-test`（`audio_url` 与 `input_audio` 双用例）HTTP 200 通过且关键词断言通过（日志归档 `20261127_053840_serial_Qwen3-ASR-0.6B_dp1_tp1_eager_multimodal.csnTvV`）。
- **Qwen3-0.6B DP2+PP2+TP2 正式 MPI PP**：
  - `torch_mpi_ext`：`mpirun --allow-run-as-root -np 4 pytest -v test/test_p2p_send_recv.py test/test_batch_p2p.py` `10 passed`（含 `@torch.compile` 与 `TORCH_MCPU_ENABLE_MEMORY_PROTECTION=ON` 内存保护验证）。
  - `vllm` pre-commit（ruff + mypy）通过；`vllm-xcpu-plugin` `./scripts/check.sh` 与 `mpirun -np 4 pytest tests/distributed/test_mpi_pp_send_recv.py` 通过。
  - 端到端：Eager `--multi-test --profile` 通过（`20261127_031108_mpi_dense_Qwen3-0.6B_dp2_pp2_tp2_eager.5UFg1x`），Profiler 实测记录 `mcpu::torch_mpi_ext::send` (512) 与 `mcpu::torch_mpi_ext::recv` (512)；修复后清理 `vllm_cache` 回归 Eager `--multi-test` 与 Compile `--multi-test` 全部通过。
- **Qwen3-30B-A3B-FP8 DP2+PP2+TP2+EP4**：
  - 单流与 `--multi-test`（8 MPI rank 5 路并发流式响应）均 exit 0 通过（`20261127_031624_mpi_moe_Qwen3-30B-A3B-FP8_dp2_pp2_tp2_ep_eager_alltoallv_v5.bS5GXn`、`20261127_032039_...rBMZMa`），首条请求输出中文量子计算解释。
- **Kimi K2 Dummy（3-layer, ACC Dequant->BF16 + Dense MLA FP8 KV）**：
  - `torch_xcpu`：`test_fp8_linear.py`（`TORCH_XCPU_FP8_LINEAR_BACKEND=acc TORCH_XCPU_BF16_LINEAR_BACKEND=acc TORCH_XCPU_FP8_LINEAR_MODEL_MAX_CONFIGS=6`）**120 passed**；`test_moe_grouped_gemm.py -k "fp8"`（`TORCH_XCPU_MOE_GROUPED_GEMM_BACKEND=acc TORCH_XCPU_ACC_FP8_DEQUANT_BF16_LINEAR=1`）**292 passed**；`unified_attention(..., kv_cache_dtype="fp8_ds_mla", logical_topk=None)` 实机分发至 `unified_attention_mla_fp8_ds` 通过。
  - `vllm-xcpu-plugin`：`./scripts/check.sh`（145 files）、`upstream_contracts check deepseek_v41`、`verify_upstream_compatibility(('sparse_mla',))`、`test_dense_mla.py`（**5 passed**）与 `test_pos_encoding.py -k "deepseek_scaling or llama3"`（**24 passed**）通过。
  - 串行端到端（`Kimi-K2-Instruct-dummy-3-layer_dp1_tp1_acc_dequant_bf16_eager.sh --profile` 及 `_compile.sh`）：推理请求 `[SUCCESS] 测试完成`（Eager 4.89s，Inductor Compile 0.56s）；Worker 日志确认 `Using XCPU_MLA_DENSE backend on MCPU` 与 `Using torch_xcpu FP8 linear backend=acc (dequant_bf16_linear=True)`；Profiler trace 实测包含 `torch_xcpu::fp8_scaled_mm_acc_fp8_backend_out: 240`、`torch_xcpu::bf16_linear_acc_out: 48`、`torch_xcpu::fused_mla_rope_qproj_cat_cache: 48` 与 `torch_xcpu::unified_attention_mla_fp8_ds: 48`（注：当前 `PRESET=release` 编译下 C++ 内核级 `MCPU_LAUNCH_TIMED_KERNEL` 标签被关闭，由上层算子记录与 `dequant_bf16_linear=True` 开关共同交叉印证；测试后 `stop_profile` HTTP 客户端在 154MB `.json.gz` 落盘时等待超时并已手工回收进程）。
  - 分布式端到端（`Kimi-K2-Instruct-dummy-3-layer_dp2_tp2_ep_acc_dequant_bf16_eager.sh --multi-test`）：4 MPI rank (`DP=2, TP=2, EP=4`, `mpi_alltoallv_v5`) 全流程 exit 0 通过（`20261127_034027_mpi_moe_Kimi-K2-Instruct-dummy-3-layer_dp2_tp2_ep_acc_dequant_bf16_eager.8uwSjs`）。

---

## 3. 影响及约束

1. **上游契约合规性**：未修改 `vllm/vllm/model_executor/layers/mla.py`、`mla_attention.py` 或 `sparse_mla_attention.py`，`sparse_mla` SHA256 与 `deepseek_v41` 上游契约全部保持通过。
2. **反量化开关内存与性能代价**：开启 `TORCH_XCPU_ACC_FP8_DEQUANT_BF16_LINEAR=1` 后，每次线性/MoE 算子调用都会在 `mcpu` 上分配同等 `[N/32, K, 32]` 大小的临时 BF16 缓冲区执行解量化（权重显存瞬时翻倍且引入额外带宽读/写开销），仅用于模拟 W8A16 分离执行行为或精度排查，默认保持关闭（`0`）。
3. **MPI PP 状态约束**：`CpuMPICommunicator` 额外 `Comm.Dup` 了 `mpi_meta_comm`（metadata `irecv`）与 `mpi_bcast_comm`（`pp_broadcast` 点对点传输），避免与业务 collective / EP 通信串包。
4. **Conv2d 算子范围**：`torch_xcpu` `convolution_overrideable` 的 `torch_xcpu_impl::conv2d_impl` 加速分支原生覆盖 BF16/FP32 1D/2D 卷积（支持任意分组与空洞，无全局 `im2col` 峰值内存膨胀），3D/转置或其他精度分支走流序感知 CPU facade fallback。

---

## 4. 后续开发

面向真实 NPU Hardware 与 Infra 团队的后续工作建议如下：

1. **真实 NPU Infra 通信替换（P2P PP）**：
   - 将 `CpuMPICommunicator` 中基于 `torch_mpi_ext` CPU staged `MPI_BYTE` 的 `send` / `recv` / `pp_broadcast` 替换为 NPU 原生集合通信/RDMA 原语（HCCL / RoCE Device-to-Device P2P），保留相同的 `DeviceCommunicatorBase` 委托接口与异步流同步语义。
2. **音频编码器 NPU 原生卷积算子**：
   - 针对 `Qwen3-ASR` 等音频/多模态前端下采样层，在 NPU 算子库中实现原生 Conv1d/Conv2d 指令核（替代 CPU simulator 的 `unfold + addmm_v2` 方案），消除 `im2col` 临时内存扩容与 Host fallback 开销。
3. **Kimi K2 / DeepSeek-V3 NPU 量化与 Dense MLA 算子融合**：
   - 基于本次验证的语义合同，NPU 算子团队应提供**寄存器内反量化融合 W8A16 GEMM/Grouped-GEMM 核**（或硬件原生 W8A8/MXFP8 计算路径），避免在 HBM 上反复分配和读写临时 `[N/32, K, 32]` BF16 权重缓冲；
   - 实现 NPU 原生 **Dense MLA 64-head Prefill/Decode Tile Kernel**（对接 `XCPU_MLA_DENSE` 接口与 `fp8_ds_mla` KV 格式），并结合真实权重开展 Kimi K2 全量 61 层多机 `DP+PP+TP+EP`（及 `--enforce-eager` 对比 `compile`）精度与吞吐验收。
4. **Profiler 后处理超时加固**：
   - 在大规模 MoE（384 experts）启用 `--profile` 时，154MB `.json.gz` 序列化耗时可超过 `shm_broadcast` 默认 60s 心跳窗口；建议 NPU Infra 在压测/Profile 框架中放宽导出期间心跳容忍或关闭非必要的 `with_memory`/`with_stack` 以缩短落盘耗时。

---

## 5. 附录

### 5.1 新增模型与预设清单

- `models/moonshotai/Kimi-K2-Instruct-dummy-3-layer/`（3 层同构缩微模型配置 + tokenizer + adaptation notes）
- `vllm_scripts/presets/serial/Llama-3.2-1B-Instruct_dp1_tp1_eager.sh` / `_compile.sh`
- `vllm_scripts/presets/mpi/dense/Llama-3.2-1B-Instruct_dp1_tp2_eager.sh`
- `vllm_scripts/presets/serial/Qwen3-ASR-0.6B_dp1_tp1_eager_multimodal.sh`
- `vllm_scripts/presets/mpi/dense/Qwen3-0.6B_dp2_pp2_tp2_eager.sh` / `_compile.sh`
- `vllm_scripts/presets/mpi/moe/Qwen3-30B-A3B-FP8_dp2_pp2_tp2_ep_eager_alltoallv_v5.sh`
- `vllm_scripts/presets/serial/Kimi-K2-Instruct-dummy-3-layer_dp1_tp1_acc_dequant_bf16_eager.sh` / `_compile.sh`
- `vllm_scripts/presets/mpi/moe/Kimi-K2-Instruct-dummy-3-layer_dp2_tp2_ep_acc_dequant_bf16_eager.sh`

### 5.2 快速复现命令

```bash
cd /shared/vllm-xcpu-dev-kit-0.31/vllm_scripts
source ../.venv/bin/activate

# 1. Qwen3-ASR-0.6B 多模态语音
./run_vllm_test.sh -e presets/serial/Qwen3-ASR-0.6B_dp1_tp1_eager_multimodal.sh --multimodal-test

# 2. LLaMA3-1B (Eager / Compile / TP2)
./run_vllm_test.sh -e presets/serial/Llama-3.2-1B-Instruct_dp1_tp1_eager.sh --multi-test
./run_vllm_test.sh -e presets/mpi/dense/Llama-3.2-1B-Instruct_dp1_tp2_eager.sh

# 3. Qwen3-0.6B DP2+PP2+TP2 (正式 MPI Send/Recv PP)
./run_vllm_test.sh -e presets/mpi/dense/Qwen3-0.6B_dp2_pp2_tp2_eager.sh --multi-test

# 4. Qwen3-30B-A3B-FP8 DP2+PP2+TP2+EP4
export VLLM_TEST_MAX_WAIT=1200 USER_VLLM_MAX_NUM_BATCHED_TOKENS=64
./run_vllm_test.sh -e presets/mpi/moe/Qwen3-30B-A3B-FP8_dp2_pp2_tp2_ep_eager_alltoallv_v5.sh --multi-test

# 5. Kimi K2 Dummy 3-Layer (ACC Dequant->BF16 + Dense MLA FP8 KV)
./run_vllm_test.sh -e presets/mpi/moe/Kimi-K2-Instruct-dummy-3-layer_dp2_tp2_ep_acc_dequant_bf16_eager.sh --multi-test
```
