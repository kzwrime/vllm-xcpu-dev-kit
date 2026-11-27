# DeepSeek-V4.1-Flash Int8 + UE8M0 (`[16, 16]`) 权重转换与部署指南

本文档说明如何将本地 `DeepSeek-V4.1-Flash` 检查点（原始 `FP8 [32, 32] UE8M0` 线性层 + `MXFP4` 路由专家 + `FP8` Engram 查表）转换为 **Int8 `[-128, 127]` + UE8M0 `[16, 16]` W8A16** 混合精度检查点，并通过 Preset 在 XCPU（Eager / Compile Mode 3 / DP+TP+EP `mpi_alltoallv_v5` / EngramParallel / DSpark）下完成端到端推理部署。

> 对照参考：GLM-5.2 / GLM-5.3 全量 Int8 指南见 [docs/glm52-int8-ue8m0-guide.md](glm52-int8-ue8m0-guide.md)。与 GLM-5.2 不同，DeepSeek-V4.1-Flash 原始路由专家已采用 4-bit `MXFP4`，Engram 查表已采用 `FP8`，因此本方案**精准替换原 `FP8 [32, 32]` 线性投影层为 `Int8 [16, 16] UE8M0`**，同时保持路由专家与 Engram 查表字节级不变。

---

## 1. 量化范围与格式规范

### 1.1 权重分类与处理策略（以 `DeepSeek-V4.1-Flash-1017` 48 分片 / 96,085 张量为例）

| 张量类别 | 数量（权重矩阵） | 源格式 | 目标格式 | 说明 |
|---|---:|---|---|---|
| **Attention 线性投影** (`wq_a`, `wkv`, `wq_b`, `wo_a`, `wo_b`) | 200 (40 层 × 5) | `F8_E4M3` + `F8_E8M0 [32,32]` | **`I8` + `U8 [16,16]` (`.scale`)** | 转为有符号 Int8 `[-128, 127]` + UE8M0 `[16, 16]` |
| **Sparse Indexer `wq_b`** | 8 (layers 2,8,14,20,24,28,32,36) | `F8_E4M3` + `F8_E8M0 [32,32]` | **`I8` + `U8 [16,16]` (`.scale`)** | 转为 Int8 + UE8M0 `[16, 16]` |
| **Engram 投影 `engram.wkv`** | 2 (layers 1, 14) | `F8_E4M3` + `F8_E8M0 [32,32]` | **`I8` + `U8 [16,16]` (`.scale`)** | 转为 Int8 + UE8M0 `[16, 16]` |
| **Shared Experts (`w1`, `w2`, `w3`)** | 120 (40 层 × 3) | `F8_E4M3` + `F8_E8M0 [32,32]` | **`I8` + `U8 [16,16]` (`.scale`)** | 转为 Int8 + UE8M0 `[16, 16]`，EP v5 下自动融合 |
| **DSpark / MTP 线性投影** (`mtp.{0..2}.attn.*`, `shared_experts.*`, `mtp.0.main_proj`) | 25 (15 + 9 + 1) | `F8_E4M3` + `F8_E8M0 [32,32]` | **`I8` + `U8 [16,16]` (`.scale`)** | 转为 Int8 + UE8M0 `[16, 16]`；3 层 `wkv` 堆叠为单次 INT8 GEMM |
| **Routed MoE 专家** (`*.experts.*.w{1,2,3}`) | 47,232 | `I8` (packed E2M1) + `F8_E8M0 [1,32]` | **原样保留（MXFP4）** | 流式原始字节零拷贝保留 |
| **Engram 查表** (`layers.{1,14}.engram.embed`) | 2 (~196 GiB) | `F8_E4M3` + `F8_E8M0 [1,32]` | **原样保留（FP8）** | 流式原始字节零拷贝保留 |
| **其余 BF16 / FP32 张量** (`embed`, `head`, `norm`, `compressor.*`, `indexer.{wk,weights_proj,k_norm}`, `ffn.gate.*`, `hc_*`, `attn_sink`, `markov_head.*`, `confidence_head.*`, `vision.*`, `aligner.*`) | 907 张量 | `BF16` / `F32` | **原样保留** | 流式原始字节零拷贝保留 |

- **合计转换**：`355` 个线性投影矩阵（权重载荷 `7.401 GB`），其余 `95,375` 个张量字节级原样保留（`355 × 2 + 95,375 = 96,085`）。
- **关键命名规范**：沿用 DeepSeek-V4.1 原生 `.scale` 后缀（dtype 为 `uint8` / safetensors `"U8"`），由 vLLM 的 `_make_deepseek_v4_weights_mapper` / `_make_deepseek_v4_vl_weights_mapper` 自动映射为线性层的 `weight_scale_inv`，同时保持 `*.experts.*.w[123].scale -> weight_scale` 与 `engram.embed.scale -> engram.embed_tokens.weight_scale_inv` 完全兼容。

### 1.2 `config.json` 配置格式

转换后的 `config.json` 中 `quantization_config` 为：

```json
"quantization_config": {
  "quant_method": "xcpu_int8",
  "fmt": "int8_ue8m0",
  "activation_scheme": "dynamic",
  "weight_block_size": [16, 16],
  "expert_dtype": "fp4"
}
```

- `quant_method: "xcpu_int8"` 直接路由至 `XcpuInt8Ue8m0Config`，无需修改 vLLM 上游代码，启动时无需手动追加 `--quantization`。
- `expert_dtype: "fp4"` 由 `DeepseekV41Config` 与 `XcpuInt8Ue8m0Config` 共同读取：所有 `LinearBase` 走 `XcpuInt8Ue8m0LinearMethod`，所有 `RoutedExperts` 走 `Mxfp4MoEMethod` -> `XcpuMxfp4MoEMethod`。
- UE8M0 指数 `u ∈ [1, 247]` 表示 $s = 2^{u - 127}$（全零块置 `127` 即 $2^0$），保证全部非零反量化值落在 BF16 规格化区间内并命中 SIMD 无乘法快路径。

---

## 2. 从本地检查点离线转换权重

转换脚本位于：[vllm_scripts/tools/convert_deepseek_v41_int8_ue8m0.py](../vllm_scripts/tools/convert_deepseek_v41_int8_ue8m0.py)（单测：[test_convert_deepseek_v41_int8_ue8m0.py](../vllm_scripts/tools/test_convert_deepseek_v41_int8_ue8m0.py)）。

### 2.1 流式分片重写设计

DeepSeek-V4.1-Flash 的两个 Engram 分片（`model-00047/00048`）单文件高达 `94.56 GiB`，而 40 个 MoE 层分片中约 99% 字节为未改动的 MXFP4 专家权重。转换工具采用**流式 Safetensors 重写器**：

1. 仅解析 JSON 头，通过 `safe_open` 按需读取待转换的 355 个 FP8 线性层并做 `[16, 16]` UE8M0 MSE 量化（3 候选指数 `u_ceil - {0, 1, 2}` 最小二乘搜索）。
2. 所有未改动张量（MXFP4 专家、Engram 查表、BF16/FP32 参数）合并连续字节区间，经 `os.sendfile` / 64 MiB 流式缓冲直接从源文件拷贝到目标文件，**单 worker 峰值内存 < 1 GiB**。
3. 调度器按文件大小降序排布任务，使两个 94.6 GiB Engram 分片最先启动，显著压缩尾延迟；并发使用 `ThreadPoolExecutor` 避免 Linux OpenMP `fork` 死锁。

### 2.2 转换命令与参数

```bash
python3 vllm_scripts/tools/convert_deepseek_v41_int8_ue8m0.py \
  --input /root/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master \
  --output /cache/models/deepseek-ai/DeepSeek-V4.1-Flash-INT8-UE8M0-16x16 \
  --block-size 16 16 \
  --workers 12
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input` | 必填 | 本地源 DeepSeek-V4.1-Flash 检查点目录 |
| `--output` | 必填 | 目标 INT8 UE8M0 输出目录（自动拷贝 tokenizer/`encoding/` 等文件并更新 `config.json` 与 `model.safetensors.index.json`） |
| `--block-size` | `16 16` | 目标分块尺寸 `[BLOCK_N, BLOCK_K]` |
| `--workers` | `12` | 并发处理分片的线程数 |
| `--resume` | 关闭 | 跳过输出目录中已完整校验通过的分片，支持断点续传 |
| `--no-mse` | 关闭 | 关闭 3 候选 UE8M0 MSE 搜索，改用解析上界指数 |

### 2.3 实测转换结果（本机全量 48 分片）

- **耗时**：`283.06 s`（~4.7 分钟完成 475 GiB 全量转换，双 94.6 GiB Engram 分片与其余 46 分片完全重叠）
- **转换统计**：`shards=48`，`converted_linears=355`，`kept_tensors=95375`（`47,232` MXFP4 专家权重+缩放、`2` Engram 查表+缩放、`907` BF16/FP32 张量）
- **UE8M0 范围**：`[110, 123]`（100% 落在快路径 `[1, 247]` 内）
- **全量 355 个线性层综合权重 RelRMSE（vs 原 FP8 `[32,32]`）**：`0.564%`
- **Scale 存储增量**：355 个线性层原 `[32, 32]` scale 占 `7.23 MB`，转为 `[16, 16]` scale 占 `28.91 MB`，**总增量仅 `+21.68 MB`**（占 475 GiB 模型的 `+0.0046%`）

---

## 3. 算子映射与架构集成

| 模型计算模块 | `torch_xcpu` 算子 | 权重与缩放格式 | 插件接入位置 |
|---|---|---|---|
| **Attention 常规投影** (`fused_wqa_wkv`, `wq_b`, `wo_b`) | `Int8Linear` (`int8_scaled_mm_out`, Intel AMX / Portable) | `int8 [N,K]` + `uint8 [N/16,K/16]` | `vllm_xcpu_plugin/layers/int8_ue8m0.py::XcpuInt8Ue8m0LinearMethod` |
| **Attention 分组输出投影 `wo_a`** (`tgd,grd->tgr`) | **`einsum_int8_tgd_grd_to_tgr`** (Intel AMX `A0` / Naive `A0-naive` / NPU `B0`) | `int8 [G,R,D]` + `uint8 [G,R/16,D/16]` | `vllm_xcpu_plugin/layers/int8_ue8m0.py` + `models/deepseek_v41/ops.py::grouped_wo_a` |
| **Sparse Indexer `wq_b` / Engram `wkv` / DSpark `main_proj`** | `Int8Linear` | `int8` + `uint8 [16,16]` | `XcpuInt8Ue8m0LinearMethod` |
| **DSpark Context KV 堆叠投影**（3 层 `wkv` 合并为 `[1536, 5120]`） | `Int8Linear`（单次堆叠 GEMM） | `int8 [1536,5120]` + `uint8 [96,320]` | `XcpuInt8Ue8m0LinearMethod` (`_xcpu_raw_wkv_int8`) + `models/deepseek_v41/model.py::build_context_wkv_linear` |
| **Shared Experts** (`gate_up_proj`, `down_proj`) | `Int8Linear`；EP v5 下**直接融合进 `fused_ep_moe_v5`** | `int8` + `uint8 [16,16]` | `layers/fused_moe/shared_experts.py` + `torch_xcpu/csrc/moe_ep/fused_ep_moe_v5.cpp` |
| **Routed MoE Experts** (`w13`, `w2`) | `IntelAmxMxfp4MoeGroupedGemm` / `PortableMxfp4Lut` / `AccMxfp4` / `fused_ep_moe_v5` | `MXFP4`（原样保留） | `XcpuInt8Ue8m0Config.get_quant_method` (`expert_dtype="fp4"`) -> `XcpuMxfp4MoEMethod` |
| **Engram 查表** (`engram.embed_tokens`) | `deepseek_v41_engram_lookup` / `engram_parallel_decode` | `FP8 E4M3` + `UE8M0 [1,32]`（原样保留） | 原生 Engram / EngramParallel 路径 |
| **未量化层** (`compressor`, `indexer.wk`, `ffn.gate`, `markov_head` 等) | `Bf16Linear` / 原生算子 | `BF16` / `FP32`（原样保留） | 原生路径 |

### 3.1 `portable_lut` / `acc` 后端支持与真机 NPU 适配

全部 355 个 INT8 线性投影（312 个常规 `Int8Linear` + 43 个分组 `wo_a` + EP v5/v8 融合共享专家 + DSpark 堆叠 `wkv`）均已统一受环境变量 `TORCH_XCPU_INT8_LINEAR_BACKEND` 控制，**完整支持 `intel_amx`、`portable_lut` 与 `acc` 三套后端**：

| `TORCH_XCPU_INT8_LINEAR_BACKEND` | 常规 `Int8Linear` & EP v5 共享专家 | 分组 `wo_a` (`einsum_int8_tgd_grd_to_tgr`) | 适用场景 |
|---|---|---|---|
| 未设置（x86 AMX 主机默认） | `IntelInt8LinearAmxKernel` (`intel_amx`) | `A0` (`int8_ue8m0_grouped_scaled_mm_unchecked`) | x86 AVX512-BF16/VNNI + AMX 开发与评测 |
| 未设置（**非 x86 / 真机默认**） | `AccInt8LinearKernel` (`acc`) | `A0-naive`（分块朴素无拷贝路径；真机专用融合算子走 `B0`） | 真机 NPU 适配主入口 |
| `portable_lut` | `PortableInt8LinearLutKernel` (`portable_lut`) | `A0-naive`（`PortableInt8LinearLutKernel` 预打包 + `einsum_naive`） | 跨架构纯便携参考实现、数值对齐验证 |
| `acc` | `AccInt8LinearKernel` (`acc`) | `A0-naive`（`B0` 专供真机融合 `einsum` 实现） | x86 本地复现单矩阵 ACC 路径 / 真机 NPU 替换 |

> **真机 NPU 算子团队替换要点**：
> 1. 只需替换 `torch_xcpu/torch_xcpu_impl/src/int8_linear/acc_int8_linear.cpp` 中的单矩阵契约 `AccInt8LinearKernel`（`packed_weight_storage_bytes`、`acc_int8_prepack_weight`、`scaled_mm`），312 个常规 `Int8Linear`、40 层 EP v5 融合共享专家及 DSpark 3 层堆叠 `wkv` 即可直接命中 NPU INT8 硬件实现。
> 2. 43 个分组投影 `wo_a`（`einsum_int8_tgd_grd_to_tgr`）与 `einsum_fp8_tgd_grd_to_tgr` 完全同构：默认在非 AMX 环境走零拷贝 `A0-naive` 基线，真机专用融合 grouped kernel 统一实现于 `OperatorImplVersion::B0`（`TORCH_XCPU_EINSUM_INT8_TGD_GRD_TO_TGR_IMPL=B0`）。

---

## 4. Preset 与启动方式

### 4.1 已提供的 Preset 清单

| 场景 | Preset 路径 | 说明 |
|---|---|---|
| Dummy 8 层（Serial Eager） | `vllm_scripts/presets/serial/DeepSeek-V4.1-Flash-INT8-dummy-8-layer-lite_dp1_tp1_eager.sh` | 快速功能验证（`DP1 TP1` Eager） |
| Dummy 8 层（Serial Compile） | `vllm_scripts/presets/serial/DeepSeek-V4.1-Flash-INT8-dummy-8-layer-lite_dp1_tp1_compile.sh` | `DP1 TP1` Compile Mode 3 |
| Dummy 8 层（Compile + DSpark 5） | `vllm_scripts/presets/serial/dspark/DeepSeek-V4.1-Flash-INT8-dummy-8-layer-lite_dp1_tp1_compile_dspark_5.sh` | Compile Mode 3 + DSpark（验证 3 层 `wkv` INT8 堆叠） |
| Dummy 8 层（MPI DP2/TP2/EP + EngramParallel） | `vllm_scripts/presets/mpi/moe/DeepSeek-V4.1-Flash-INT8-dummy-8-layer-lite_dp2_tp2_ep_compile_alltoallv_v5_engram_parallel.sh` | Compile + `mpi_alltoallv_v5` + 融合 INT8 共享专家 + EngramParallel |
| **真权重（MPI DP2/TP2/EP v5）** | `vllm_scripts/presets/mpi/moe/DeepSeek-V4.1-Flash-INT8_dp2_tp2_ep_compile_alltoallv_v5.sh` | 40 层全量真权重，TP 分片 Engram |
| **真权重（MPI DP2/TP2/EP v5 + EngramParallel，Compile）** | `vllm_scripts/presets/mpi/moe/DeepSeek-V4.1-Flash-INT8_dp2_tp2_ep_compile_alltoallv_v5_engram_parallel.sh` | 40 层全量真权重，Engram 2D 分片至全部 4 Rank（内存占用减半） |
| **真权重（MPI DP2/TP2/EP v5 + EngramParallel，Eager）** | `vllm_scripts/presets/mpi/moe/DeepSeek-V4.1-Flash-INT8_dp2_tp2_ep_eager_alltoallv_v5_engram_parallel.sh` | 40 层全量真权重 Eager 模式（支持 `USER_VLLM_EXTRA_ARGS`，适合快速联调与 QDA-KL 评测） |
| **真权重（MPI DP2/TP2/EP v5 + DSpark 5）** | `vllm_scripts/presets/mpi/moe/dspark/DeepSeek-V4.1-Flash-INT8_dp2_tp2_ep_compile_alltoallv_v5_dspark_5.sh` | 40 层全量真权重 + DSpark 预测解码 |

Dummy 模型目录：`models/deepseek-ai/DeepSeek-V4.1-Flash-INT8-UE8M0-dummy-8-layer-lite/`。

### 4.2 启动与验证命令

切换量化格式或重编 `torch_xcpu` 后，**务必先清理编译缓存**：

```bash
cd vllm_scripts
rm -rf vllm_cache torch_compile_cache_opt

# 1. Dummy 快速验证（缩短预热时间）
USER_VLLM_MAX_NUM_BATCHED_TOKENS=32 VLLM_TEST_MAX_WAIT=420 \
  ./run_vllm_test.sh -e presets/mpi/moe/DeepSeek-V4.1-Flash-INT8-dummy-8-layer-lite_dp2_tp2_ep_compile_alltoallv_v5_engram_parallel.sh

# 2. 40 层全量真权重验证（DP2/TP2/EP + EngramParallel + Compile）
VLLM_TEST_MAX_WAIT=1200 \
  ./run_vllm_test.sh -e presets/mpi/moe/DeepSeek-V4.1-Flash-INT8_dp2_tp2_ep_compile_alltoallv_v5_engram_parallel.sh
```

在 `mpi_workers.log` 中出现以下关键日志即确认全链路命中：

```text
Using torch_xcpu INT8 UE8M0 linear backend=intel_amx
Using torch_xcpu einsum_int8_tgd_grd_to_tgr (A0) for wo_a
Using XcpuGroupedGemmExperts: format=MXFP4 backend=intel_amx implementation=IntelAmxMxfp4MoeGroupedGemm use_ep=True topk_reduce=True
Using XcpuEPExperts: version=5 transport=mpi_alltoallv_v5 compute=IntelAmxMxfp4MoeGroupedGemm ...
Fused shared experts into XCPU EP v5 for 40 MoE layers
```

开启 DSpark 时还会打印：

```text
DSpark context KV: stacked 3 wkv layers into INT8 linear ((1536, 5120), backend=intel_amx)
```

---

## 5. 真权重定量精度评估（`DeepSeek-V4.1-Flash-1017`）

在真实检查点上抽取前/中/后层及 MTP/DSpark 层共 **29 个代表性投影矩阵**（覆盖全部 10 类线性投影），在 `mcpu` 设备上实测 `Int8 [16, 16] UE8M0`（MSE 搜索）与原始 `FP8 [32, 32] UE8M0` 的权重重构误差及真实 `mcpu` W8A16 GEMM / `einsum_int8_tgd_grd_to_tgr` 输出误差（BF16 激活 `M=16`，对比对象为原 FP8 `mcpu` 算子输出）：

| 模块家族 | 抽样层 | 矩阵形状 `[N, K]` | W_RelRMSE (`[16,16]` MSE) | W_RelRMSE (`[16,16]` no-MSE) | W_RelRMSE (`[32,32]` MSE) | `mcpu` GEMM RelL2 (INT8 vs FP8) | `mcpu` GEMM CosSim | UE8M0 范围 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `attn.wq_a` | L0 / L20 / L39 | `[1280, 5120]` | **0.499% – 0.609%** | 0.587% – 0.655% | 0.684% – 0.723% | 0.549% – 0.662% | 0.999983 – 0.999989 | `[116, 118]` |
| `attn.wkv` | L0 / L20 / L39 | `[512, 5120]` | **0.594% – 0.611%** | 0.613% – 0.649% | 0.626% – 0.698% | 0.642% – 0.647% | 0.999981 | `[116, 119]` |
| `attn.wq_b` | L0 / L20 / L39 | `[32768, 1280]` | **0.548% – 0.587%** | 0.594% – 0.631% | 0.672% – 0.741% | 0.595% – 0.632% | 0.999939 – 0.999946 | `[115, 119]` |
| `attn.wo_a` (`einsum`) | L0 / L20 / L39 | `[8192, 4096]` (`8×1024×4096`) | **0.521% – 0.594%** | 0.557% – 0.638% | 0.609% – 0.776% | 0.567% – 0.639% | 0.999973 – 0.999985 | `[115, 119]` |
| `attn.wo_b` | L0 / L20 / L39 | `[5120, 8192]` | **0.468% – 0.612%** | 0.542% – 0.655% | 0.649% – 0.721% | 0.521% – 0.659% | 0.999978 – 0.999992 | `[116, 119]` |
| `indexer.wq_b` | L2 / L20 / L36 | `[4096, 1280]` | **0.601% – 0.636%** | 0.650% – 0.681% | 0.800% – 0.834% | 0.651% – 0.677% | 0.999977 – 0.999980 | `[115, 122]` |
| `engram.wkv` | L1 / L14 | `[25600, 6144]` | **0.777% – 0.876%** | 0.829% – 0.930% | 1.085% – 1.190% | 0.809% – 0.898% | 0.999985 – 0.999998 | `[110, 120]` |
| `shared_experts.w1` | L0 / L20 | `[2304, 5120]` | **0.601% – 0.677%** | 0.641% – 0.741% | 0.789% – 0.927% | 0.641% – 0.714% | 0.999979 – 0.999984 | `[115, 120]` |
| `shared_experts.w2` | L0 / L20 | `[5120, 2304]` | **0.629% – 0.642%** | 0.694% – 0.720% | 0.897% – 0.910% | 0.677% – 0.691% | 0.999977 – 0.999983 | `[116, 120]` |
| `shared_experts.w3` | L0 / L20 | `[2304, 5120]` | **0.622% – 0.727%** | 0.690% – 0.772% | 0.873% – 0.875% | 0.655% – 0.760% | 0.999975 – 0.999984 | `[116, 120]` |
| `mtp.0.main_proj` | MTP 0 | `[5120, 15360]` | **0.483%** | 0.536% | 0.612% | 0.535% | 0.999989 | `[117, 119]` |
| `mtp.0.attn.wo_a` | MTP 0 | `[8192, 4096]` | **0.540%** | 0.597% | 0.685% | 0.584% | 0.999984 | `[116, 120]` |
| `mtp.2.attn.wq_b` | MTP 2 | `[32768, 1280]` | **0.511%** | 0.564% | 0.665% | 0.561% | 0.999949 | `[117, 121]` |

**核心结论**：
1. `[16, 16]` + MSE 相比 `[32, 32]` 将误差再降低约 **20%–28%**（尤其对 `engram.wkv` 与 `shared_experts` 收益最显著：`engram.wkv` 从 `1.09%–1.19%` 降至 `0.78%–0.88%`，`shared_experts` 从 `0.79%–0.93%` 降至 `0.60%–0.73%`）。
2. 所有 10 类线性投影的 `mcpu` W8A16 GEMM / `einsum_int8_tgd_grd_to_tgr` 输出相对原 FP8 `mcpu` 算子的余弦相似度均达到 **`0.99994 – 0.999998`**。
3. 所有 355 个矩阵的 scale 增量仅 `+21.68 MB`，几乎零显存/内存开销。

---

## 6. 轻量级端到端 KL 散度评测（参考 `humanfia/qda` / `qda-kl`）

参考 [QDA (Closed-Loop LLM Quantization Design with Agents)](https://github.com/humanfia/qda) 的 `qda-kl[vllm]` 评测范式，配套提供复用评测脚本 [vllm_scripts/tools/eval_qda_kl_vllm.py](../vllm_scripts/tools/eval_qda_kl_vllm.py)（单测：[test_eval_qda_kl_vllm.py](../vllm_scripts/tools/test_eval_qda_kl_vllm.py)）：

1. **参考轨迹采样与 Teacher-Forced Top-$K$ 分布提取**：在基准模型 $P$（原 FP8 `[32, 32]`）上对 8 条未见推理/代码/数学/物理/系统/中文/结构化 Prompt 贪婪生成参考轨迹（共 `1,024` 个 trace token），通过 vLLM `/v1/completions` 的 `prompt_logprobs=64`（需服务端加 `--max-logprobs 64`）在单次 prefill 内提取每个位置的 Top-64 概率分布（平均覆盖 `99.988%` 概率质量）。
2. **候选模型分布漂移计算**：在量化模型 $Q$（Int8 `[16, 16]`）上对**完全一致的 token ID 序列**做 teacher-forced `prompt_logprobs=64`，按 $U = S_P \cup S_Q$ 加上显式 `<tail>` 桶构造严格归一分布，计算前向 KL $D_{\mathrm{KL}}(P \parallel Q)$、反向 KL $D_{\mathrm{KL}}(Q \parallel P)$、JSD、Top-1/Top-5 一致率，以及零截断误差的精确 trace-token NLL / PPL 增量。
3. **长生成护栏检查（Guardrail）**：对 2 条复杂数学证明 / 算法实现 Prompt 做 256-token 贪婪长生成，检查 4-gram / 8-gram 重复率以排除死循环与长程漂移。

### 6.1 40 层全量真权重实测结果（FP8 `[32,32]` vs INT8 `[16,16]`，DP2/TP2/EP + EngramParallel）

| 指标（`1,024` 个参考推理 trace token，`top_k=64`） | 数值 | 说明 |
|---|---:|---|
| **平均前向 KL $D_{\mathrm{KL}}(P \parallel Q)$** | **`0.005003 nats`** | 中位数 `0.000134`、P95 `0.024067`、P99 `0.063267`、Max `0.328438` |
| **平均反向 KL $D_{\mathrm{KL}}(Q \parallel P)$ / JSD** | **`0.004946`** / **`0.001226 nats`** | 前反向高度对称，无系统性分布偏移 |
| **Top-1 / Top-5 一致率** | **`98.34%`** / **`100.00%`** | `1024/1024` 全部命中 Top-5；Top-64 支持集重叠度 `94.52%` |
| **Top-64 概率质量覆盖率（$M_P$ / $M_Q$）** | **`99.988%`** / **`99.988%`** | 近似全词表精确 KL |
| **精确 Trace NLL / PPL** | `0.19277 → 0.19328` (`ΔNLL = +0.000509`) | `PPL: 1.2126 → 1.2132`（比值 **`1.00051`**，仅 `+0.051%`），Logprob MAE `0.02602` |
| **全序列 1,771 token（含 Prompt + Trace）** | `KL = 0.017381`, `Top-1 = 97.29%`, `Top-5 = 99.89%` | `PPL: 5.1316 → 5.1680`（比值 `1.00710`） |

**分任务明细（每条 128 trace token）**：

| 轨迹 ID | 类别 | Mean $D_{\mathrm{KL}}(P \parallel Q)$ | P95 KL | Top-1 一致率 | $\Delta\text{NLL}$ |
|---|---|---:|---:|---:|---:|
| `math_combinatorics_en` | 组合数学推理 | `0.003584` | `0.019433` | `99.2%` | `-0.006225` |
| `math_probability_zh` | 中文概率推导 | `0.003054` | `0.014884` | `97.7%` | `+0.003768` |
| `code_dp_en` | Python O(n log n) LIS | `0.004526` | `0.019702` | `96.9%` | `-0.005422` |
| `systems_quant_zh` | 量化体系结构分析 | `0.004547` | `0.027955` | `99.2%` | `+0.004302` |
| `logic_puzzle_en` | 多步逻辑推理 | `0.005111` | `0.026465` | `99.2%` | `+0.007498` |
| `physics_kinematics_en` | 物理定量推导 | `0.003513` | `0.011391` | `97.7%` | `+0.003486` |
| `chinese_reasoning_zh` | 中文分布式 MoE 原理 | `0.011050` | `0.036371` | `97.7%` | `-0.005899` |
| `structured_json_en` | 严格 JSON 格式化 | `0.004638` | `0.026344` | `99.2%` | `+0.002562` |

**256-Token 贪婪长生成护栏检查**：
- `longgen_math_proof`（无理数证明 + 极小多项式）：FP8 与 INT8 给出完全相同的代数推导路径（$\alpha^2 = 5 + 2\sqrt{6} \Rightarrow \alpha^4 - 10\alpha^2 + 1 = 0$ 并论证不可约性），无异常循环（`rep8: 0.0% → 6.0%`，属正常数学表达式复用）。
- `longgen_code_lru`（手写双向链表 + 哈希表 O(1) LRU Cache）：无重复循环（`rep4: 3.6% → 2.0%`, `rep8: 0.0% → 0.0%`），两者均严谨区分了 Python `dict` 均摊 O(1) 与最坏情况复杂度。

### 6.2 复用评测命令

```bash
# 1. 启动基准 FP8 服务（附加 --max-logprobs 64）后采集参考轨迹与 logprobs：
python3 vllm_scripts/tools/eval_qda_kl_vllm.py --mode collect-ref \
  --base-url http://127.0.0.1:14800 \
  --model /root/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master \
  --top-k 64 --trace-max-tokens 128 --longgen-max-tokens 256 \
  --out-dir ./qda_kl_eval

# 2. 切换启动量化 INT8 服务（附加 --max-logprobs 64）后评测候选分布漂移：
python3 vllm_scripts/tools/eval_qda_kl_vllm.py --mode eval-cand \
  --base-url http://127.0.0.1:14800 \
  --model /cache/models/deepseek-ai/DeepSeek-V4.1-Flash-INT8-UE8M0-16x16 \
  --out-dir ./qda_kl_eval
```

---

## 7. 端到端验证记录

| 验证项 | 命令 / 配置 | 结果 |
|---|---|---|
| `torch_xcpu` 算子单测（含三套后端与融合 MLA） | `pytest -v torch_xcpu/test/ops/custom_ops/test_einsum_int8_tgd_grd_to_tgr.py` + `test_int8_ue8m0_16x16.py -k test_int8_linear_backends` + `test_fused_mla_add_norm_* -k int8` | **61/61 + 42/42 + 66/66 PASSED**（开启 `TORCH_MCPU_ENABLE_MEMORY_PROTECTION=ON` 全覆盖 `intel_amx`、`portable_lut`、`acc`、`[16,16]`/`[32,32]`、`uint8`/`float8_e8m0fnu`、奇偶 strided 视图、打包存储校验与边界 UE8M0 缩放） |
| AOTI Wrapper 校验 | `python3 torch_xcpu/scripts/generate_aoti_wrappers.py --check` | **PASSED** |
| 插件单测 + 上游契约 | `pytest -v vllm-xcpu-plugin/tests/layers/test_int8_ue8m0.py tests/plugin_tests/test_dspark_phase1.py tests/upstream_contracts/test_upstream_contracts_deepseek_v41.py` + `upstream_contracts lint` + `upstream_contracts check/provenance deepseek_v41` | **21/21 + 4/4 PASSED**，5 个契约分类全部 lint clean，`deepseek_v41` 契约与溯源全部通过 |
| 离线转换与 QDA-KL 工具单测 | `pytest -v vllm_scripts/tools/test_convert_deepseek_v41_int8_ue8m0.py vllm_scripts/tools/test_eval_qda_kl_vllm.py` | **3/3 + 4/4 PASSED** |
| Dummy 8 层 Eager（三套后端对照） | `TORCH_XCPU_INT8_LINEAR_BACKEND={intel_amx,portable_lut,acc}` + `DeepSeek-V4.1-Flash-INT8-dummy-8-layer-lite_dp1_tp1_eager.sh` | **全部 PASSED 且三后端 16/16 输出 token 比特一致**（日志确认命中 `backend={intel_amx,portable_lut,acc}` 与 `einsum_int8_tgd_grd_to_tgr ({A0,A0-naive})`） |
| Dummy 8 层 Compile + DSpark 5 (`DP1 TP1`) | `DeepSeek-V4.1-Flash-INT8-dummy-8-layer-lite_dp1_tp1_compile_dspark_5.sh` | **PASSED**（与 Eager 16/16 token 比特一致；日志确认 `DSpark context KV: stacked 3 wkv layers into INT8 linear ((1536, 5120), backend=intel_amx)`） |
| Dummy 8 层 MPI Compile (`DP2 TP2 EP v5` + EngramParallel) | `DeepSeek-V4.1-Flash-INT8-dummy-8-layer-lite_dp2_tp2_ep_compile_alltoallv_v5_engram_parallel.sh` | **PASSED**（与 Eager/DSpark 16/16 token 比特一致；日志确认 `Fused shared experts into XCPU EP v5 for 8 MoE layers`） |
| **真权重 40 层全量 MPI Compile (`DP2 TP2 EP v5` + EngramParallel)** | `DeepSeek-V4.1-Flash-INT8_dp2_tp2_ep_compile_alltoallv_v5_engram_parallel.sh`（模型：`/cache/models/deepseek-ai/DeepSeek-V4.1-Flash-INT8-UE8M0-16x16`） | **PASSED**（加载 `239–247s`，推理 `3.17s`；40 层共享专家全部融合进 EP v5；输出连贯中文回答） |
| **真权重 40 层 QDA-KL 端到端评测** | `eval_qda_kl_vllm.py`（FP8 `[32,32]` vs INT8 `[16,16]`，`top_k=64`，`1,024` trace token + 2 × 256 长生成） | **PASSED**（Mean Forward KL = **`0.005003 nats`**，Top-1 = **`98.34%`**，Top-5 = **`100.00%`**，PPL ratio = **`1.00051`**，无死循环） |
