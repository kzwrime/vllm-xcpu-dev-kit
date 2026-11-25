# GLM-5.2 / GLM-5.3 Int8 + UE8M0 (`[16, 16]`) 权重转换与部署指南

本文档面向内网部署与算子联调团队，说明如何从**本地模型目录**将 `GLM-5.2-FP8` 或 `GLM-5.3`（FP8；待 `GLM-5.3-BF16` 导入本地后同样直接适用）转换为 **Int8 `[-128, 127]` + UE8M0 `[16, 16]` W8A16** 格式，并指导在不同算子适配阶段如何通过 Preset 启动推理服务（含 `DP16 / TP4 / EP64` 与 **Int8 权重 + Int8 KV Cache** 组合）。第 6 节汇总了完整的**实测方法、8-bit 编码与 SIMD 反量化权衡、多粒度精度实测、子图消融以及 `TP4 + EP32` 单节点显存增量核算**，供选型与评审参考。

---

## 1. 量化格式规范

| 字段 | 类型 / 形态 | 说明 |
|---|---|---|
| 权重 `<name>` | `torch.int8` (`I8`)，`[N, K]` | 二进制补码对称整数 `[-128, 127]` |
| 块缩放因子 `<name>_scale_inv` | `torch.uint8` (`U8`)，`[N/16, K/16]` | 原始 UE8M0 指数 `u`，表示 $s = 2^{u - 127}$；有效范围 `[1, 247]`（全零块置 `127` 即 $2^0$），确保反量化到 BF16 时处于正常值区间 |
| 未量化张量与 `indexer.wk` | 保持原 dtype | 所有 `float32` 张量（如 `e_score_correction_bias`）、未量化 BF16 张量（`embed_tokens`、`lm_head`、`norm`、`mlp.gate`、`eh_proj`、`indexer.weights_proj` 等）以及稀疏注意力 `self_attn.indexer.wk` 均**原样保留**（FP8 源保留 FP8+FP32 scale，BF16 源保留 BF16），由 vLLM 原生加载逻辑直接处理 |

> **关键不变量**：
> - `GLM-5.2-FP8` 中共 `59,044` 个原 FP8 张量，其中 `22` 个 `self_attn.indexer.wk.weight` 原样保留 FP8，`--scope all` 转换其余 `59,022` 个张量，`--scope moe_w13` 仅转换 routed MoE 第一层 GroupedGemm `w13`（`76 × 256 × 2 = 38,912` 个张量）；另外 `465` 个 BF16 张量与 `76` 个 FP32 张量保持不变。
> - 转换脚本支持本地 FP8 或 BF16 源目录，**不需要访问外网 / ModelScope / HuggingFace**。

---

## 2. 从本地目录转换权重

转换脚本位于：`vllm_scripts/tools/convert_glm_int8_ue8m0.py`。

### 2.1 常用参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input` | 必填 | 本地源模型目录（如 `/data/models/ZhipuAI/GLM-5.2-FP8` 或 `/data/models/ZhipuAI/GLM-5.3`） |
| `--output` | 必填 | 目标输出目录（自动拷贝 tokenizer 等非权重文件并生成新的 `config.json` 与 `model.safetensors.index.json`） |
| `--scope` | `all` | 转换范围：`all`（转换全部量化矩阵）或 `moe_w13`（仅转换 routed MoE 第一个 GroupedGemm `gate_proj`/`up_proj`，其余全部保持原模型精度） |
| `--block-size` | `16 16` | 分块量化尺寸 `[BLOCK_N, BLOCK_K]` |
| `--workers` | `16` | 并行处理 safetensors 分片的进程数（建议 16~32，视 CPU 核数与磁盘 IO 调整） |
| `--resume` | 关闭 | 跳过输出目录中已转换完成的非空分片，支持断点续传 |
| `--no-mse` | 关闭 | 关闭 3 候选 UE8M0 MSE 搜索，改用纯上界指数（速度略快，精度略低；默认开启 MSE 搜索） |

### 2.2 转换命令示例

#### 场景 A：第一阶段实验——仅转换 MoE 第一个 GroupedGemm（`w13`）

```bash
python3 vllm_scripts/tools/convert_glm_int8_ue8m0.py \
  --input /path/to/local/GLM-5.2-FP8 \
  --output /path/to/output/GLM-5.2-FP8-MOE-W13-INT8-UE8M0-16x16 \
  --scope moe_w13 \
  --block-size 16 16 \
  --workers 24 \
  --resume
```

若源模型是本地 `GLM-5.3`（FP8），只需替换 `--input` 和 `--output`；后续 `GLM-5.3-BF16` 导入本地后，**命令完全相同**，脚本会自动识别 BF16 源权重，仅量化 `moe_w13`，其余保留 BF16 / FP32。

#### 场景 B：全量矩阵转换（第二阶段——所有矩阵乘法适配完成后）

```bash
python3 vllm_scripts/tools/convert_glm_int8_ue8m0.py \
  --input /path/to/local/GLM-5.2-FP8 \
  --output /path/to/output/GLM-5.2-INT8-UE8M0-16x16 \
  --scope all \
  --block-size 16 16 \
  --workers 24 \
  --resume
```

> 参考实测规模：`GLM-5.2-FP8`（141 个分片）全量转换产出 `quantized=59022, kept_fp8=22 (indexer.wk), kept_bf16=465, kept_f32=76`，总大小约 `706.3 GiB`。

### 2.3 转换后快速抽检

```bash
python3 - <<'PY'
import json
from pathlib import Path
from safetensors import safe_open

out_dir = Path("/path/to/output/GLM-5.2-INT8-UE8M0-16x16")
cfg = json.loads((out_dir / "config.json").read_text())
print("quantization_config:", {k: v for k, v in cfg["quantization_config"].items() if k != "modules_to_not_convert"})
print("modules_to_not_convert count:", len(cfg["quantization_config"].get("modules_to_not_convert", [])))

shard = sorted(out_dir.glob("model-*.safetensors"))[4]
dtypes = {}
with safe_open(shard, framework="pt", device="cpu") as f:
    for k in f.keys():
        dt = str(f.get_tensor(k).dtype)
        dtypes[dt] = dtypes.get(dt, 0) + 1
print("sample shard:", shard.name, "dtype counts:", dtypes)
PY
```

---

## 3. Preset 启动方式（基准模板与 `DP16 TP4 EP64` 配置）

当前最常用的基准编译模式 Preset 为：
`vllm_scripts/presets/mpi/moe/GLM-5.2-FP8_dp2_tp2_compile_fp8_kvcache_alltoallv_v5.sh`

其核心配置如下：
- 编译模式：`USER_VLLM_EAGER_OR_NOT=""`，`VLLM_USE_BREAKABLE_CUDAGRAPH=0`（DSA 必须关闭 breakable cudagraph 才能走 compile）
- 并行通信：`--all2all-backend mpi_alltoallv_v5`，`VLLM_USE_MPI_COORD=1`，`VLLM_CPU_USE_MPI=1`
- KV Cache：`--kv-cache-dtype ${USER_VLLM_KV_CACHE_DTYPE:-fp8}`
- 检查开关：`TORCH_XCPU_ENABLE_CHECK=0`

### 3.1 新建 Int8 Preset（`DP16 TP4 EP64` 示例）

基于基准 Preset 复制一份新文件，例如：
`vllm_scripts/presets/mpi/moe/GLM-5.2-INT8_dp16_tp4_ep64_compile_fp8_kvcache_alltoallv_v5.sh`

```bash
#!/bin/bash

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

export PD_MODE="MIXED"
# DSA 下关闭 breakable CUDA graph 以启用 torch.compile
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export USER_VLLM_MAX_NUM_BATCHED_TOKENS="${USER_VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-2400}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
source "$SCRIPT_DIR/user_env_template.sh"

unset VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16
export TORCH_XCPU_ENABLE_CHECK=0
export USER_VLLM_EAGER_OR_NOT=""

# 指向本地转换后的模型目录（本地目录会自动关闭 ModelScope 下载）
export USER_VLLM_MODEL="/path/to/output/GLM-5.2-INT8-UE8M0-16x16"

# DP16 TP4 EP64（开启 --enable-expert-parallel 后 EP = DP * TP = 64）
export USER_VLLM_DATA_PARALLEL_SIZE=16
export USER_VLLM_TP_SIZE=4
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))

export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1

_VLLM_OPTIONAL_ARGS+=" --all2all-backend mpi_alltoallv_v5"
_VLLM_OPTIONAL_ARGS+=" --kv-cache-dtype ${USER_VLLM_KV_CACHE_DTYPE:-fp8}"
_VLLM_OPTIONAL_ARGS+=' --kernel-config {"enable_jit_warmup":false}'
_VLLM_OPTIONAL_ARGS+=' --tool-call-parser glm47'
_VLLM_OPTIONAL_ARGS+=' --enable-auto-tool-choice'
_VLLM_OPTIONAL_ARGS+=' --reasoning-parser glm45'
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
echo "Preset: ${preset_dir}/${preset_name} | INT8 UE8M0 16x16 checkpoint, ${USER_VLLM_KV_CACHE_DTYPE:-fp8} KV cache"
```

### 3.2 启动命令与缓存清理

**首次切换量化格式、修改算子或切换 KV Cache 类型前，务必先清理编译缓存**，防止加载旧的编译产物：

```bash
cd vllm_scripts
rm -rf vllm_cache torch_compile_cache_opt

# 单请求端到端验证
./run_vllm_test.sh -e presets/mpi/moe/GLM-5.2-INT8_dp16_tp4_ep64_compile_fp8_kvcache_alltoallv_v5.sh

# 并发多请求验证
./run_vllm_test.sh -e presets/mpi/moe/GLM-5.2-INT8_dp16_tp4_ep64_compile_fp8_kvcache_alltoallv_v5.sh --multi-test
```

> 若先排查功能/精度问题，可临时在 Preset 中设置 `export USER_VLLM_EAGER_OR_NOT="--enforce-eager"` 走 Eager 模式验证，通过后再切回 `""` 走 Compile 模式。

---

## 4. 分阶段算子适配操作指南

在 `mpi_alltoallv_v5`（`fused_ep_moe_v5`）主路径中，不同矩阵乘对应的 `torch_xcpu` 算子分工如下：

| 模型计算模块 | 对应 `torch_xcpu` 算子 | 权重张量 |
|---|---|---|
| **Routed MoE 第一个 GroupedGemm (`w13` + 融合 `SiluAndMul`)** | `MoeGroupedGemmWithIndex`（即 Indexer GroupedGemm） | `mlp.experts.<E>.gate_proj`、`up_proj` |
| **Routed MoE 第二个 GroupedGemm (`w2`)** | `MoeGroupedGemm` | `mlp.experts.<E>.down_proj` |
| **Attention / Dense MLP / Shared Experts** | `Int8Linear`（`int8_scaled_mm_out`） | `self_attn.*`、dense `mlp.*`、`mlp.shared_experts.*` |
| **MLA 吸收权重** | `int8_block_dequant`（按需反量化） | `kv_b_proj`（`self_attn.indexer.wk` 保持原 FP8/BF16，由 vLLM 原生加载到 BF16 `wk_weights_proj`） |

---

### 4.1 第一阶段：只有 `MoeGroupedGemmWithIndex`（Indexer GroupedGemm）适配了 Int8 UE8M0

当 NPU / 底层算子库**仅完成 `MoeGroupedGemmWithIndex` 的 `INT8_UE8M0 [16, 16]` 适配**，而 `MoeGroupedGemm`（`w2`）与常规 `Int8Linear` 尚未适配时，有两种推进方案：

#### 方案 1（推荐，严格控制变量）：仅转换 `moe_w13` + 混合精度调度

1. **生成混合精度权重**：
   使用 `--scope moe_w13` 转换源模型（见 2.2 场景 A）。
   - 仅 `mlp.experts.<E>.gate_proj.weight` 和 `up_proj.weight` 被转为 `int8` + `[16, 16]` `uint8` UE8M0；
   - `down_proj`、`shared_experts`、Attention、Dense MLP 均保持原 `float8_e4m3fn` + FP32 `[128, 128]` scale（或 BF16 源模型的 `bfloat16`）；
   - `config.json` 中保留原 `quant_method: "fp8"`，并新增：
     ```json
     "xcpu_int8_overrides": {
       "fmt": "int8_ue8m0",
       "target_scope": "moe_w13",
       "weight_block_size": [16, 16]
     }
     ```
2. **算子与框架层对接要求（注意当前代码边界）**：
   - 当前主干 `vllm-xcpu-plugin` 与 `fused_ep_moe_v5` 默认按**同一 MoE 层内 `w13` 与 `w2` 同格式**（同为 `FP8` 或同为 `INT8_UE8M0`）调度。
   - 若要让 vLLM 直接加载这种 **`w13=INT8_UE8M0 [16,16]` + `w2/Linear=FP8 [128,128]`** 的混合精度 checkpoint 跑端到端实验，需要同步做两处轻量对接：
     1. **`torch_xcpu` (`fused_ep_moe_v5.cpp` / `fused_moe_compute.cpp`)**：将 GEMM1（`IndexBackend`，`MoeGroupedGemmWithIndex`）和 GEMM2（`Gemm2`，`MoeGroupedGemm`）的权重格式与 `scale_block_n/k` 拆分为 `w1_format` / `w2_format`（允许 `w1=INT8_UE8M0 [16,16]` 与 `w2=FP8 [128,128]` 组合实例化）。
     2. **`vllm-xcpu-plugin` (`layers/fp8_moe.py` / `layers/int8_ue8m0.py`)**：读取 `quantization_config["xcpu_int8_overrides"]`，当 `target_scope == "moe_w13"` 时，仅对 `w13_weight` / `w13_weight_scale_inv` 按 `int8` + `uint8 [16, 16]` 创建参数并调用 `Int8MoeGroupedGemmWithIndex` prepack；`w2` 及所有 Linear 层继续走原有 FP8 路径。
3. **独立算子级验证（可立即执行，无需等混合调度改造）**：
   在 NPU 替换 `MoeGroupedGemmWithIndex` 实现后，可直接运行 `torch_xcpu` 算子单测验证数值精度（覆盖 `epilogue=None` 与 `SiluAndMul`）：
   ```bash
   pytest -v torch_xcpu/test/ops/test_int8_ue8m0_16x16.py -k "moe_grouped_gemm_with_index"
   ```

#### 方案 2（零混合调度改造，适合功能联调）：全量转换 `--scope all` + 非热点算子暂走参考后端

如果当前环境允许未适配的 `MoeGroupedGemm`（`w2`）和 `Int8Linear` 暂时走 `torch_xcpu` 内置的 x86 AMX / Portable 参考实现，而仅将 `MoeGroupedGemmWithIndex` 替换为 NPU 实现：
1. 直接用 `--scope all` 转换全量模型；
2. 无需修改 `vllm-xcpu-plugin` 和 `fused_ep_moe_v5` 的格式调度逻辑；
3. 如需强制指定参考后端用于对比精度，可通过环境变量控制：
   - `TORCH_XCPU_INT8_LINEAR_BACKEND`（`intel_amx` / `portable_lut` / `acc`）
   - `TORCH_XCPU_MOE_GROUPED_GEMM_BACKEND`（`intel_amx` / `portable` / `acc`）

---

### 4.2 第二阶段：全部矩阵乘法形状均已适配 Int8 UE8M0

当 `Int8Linear`、`MoeGroupedGemmWithIndex`（`w13`）、`MoeGroupedGemm`（`w2`）以及 `int8_block_dequant` 均完成 NPU 适配后：

1. **全量转换模型**：使用 `--scope all --block-size 16 16` 生成目标 checkpoint（见 2.2 场景 B）。
2. **直接使用 `xcpu_int8` 全链路**：
   - `config.json` 中已自动写入 `"quant_method": "xcpu_int8"`，**不需要**在 vLLM 启动命令中手动追加 `--quantization` 参数。
   - `vllm-xcpu-plugin` 会自动将：
     - 所有 Attention / Dense MLP / Shared Experts 路由至 `Int8Linear`（在 `mpi_alltoallv_v5` 下共享专家自动融合进 `fused_ep_moe_v5`）；
     - Routed MoE `w13` 路由至 `MoeGroupedGemmWithIndex (INT8_UE8M0)`；
     - Routed MoE `w2` 路由至 `MoeGroupedGemm (INT8_UE8M0)`；
     - MLA `kv_b_proj` 路由至 `int8_block_dequant`（`self_attn.indexer.wk` 保持原 FP8/BF16，由 vLLM 原生路径加载）。
3. **清理缓存并按第 3 节的 Preset 启动**，在 `mpi_workers.log` 中确认以下关键日志即表示全链路命中：
   ```text
   Using torch_xcpu INT8 UE8M0 linear backend=...
   Using XcpuGroupedGemmExperts: format=INT8_UE8M0 ...
   Using XcpuEPExperts: version=5 transport=mpi_alltoallv_v5 compute=...Int8MoeGroupedGemm ...
   Fused shared experts into XCPU EP v5 for ... MoE layers
   ```

---

## 5. 同时使用 Int8 权重与 Int8 KV Cache

`vllm-xcpu-plugin` 的 Sparse MLA 后端（`vllm_xcpu_plugin/attention/sparse_mla.py`）已支持 `int8` / `int8_ds_mla` KV Cache：
- 传入 `--kv-cache-dtype int8` 时，插件会自动规范化为 `int8_ds_mla`（每 token 占 `656` 字节，与 `fp8_ds_mla` 同等显存/内存压缩比）。
- 权重格式（`xcpu_int8`）与 KV Cache 类型（`int8_ds_mla`）相互正交，可直接叠加使用。

### 5.1 启动方式

#### 方式 1：通过环境变量覆盖（无需修改 Preset 文件）

所有 GLM-5.2 MPI Preset 均已预留 `--kv-cache-dtype ${USER_VLLM_KV_CACHE_DTYPE:-fp8}`，直接导出环境变量即可：

```bash
cd vllm_scripts
rm -rf vllm_cache torch_compile_cache_opt

export USER_VLLM_KV_CACHE_DTYPE=int8   # 或显式写为 int8_ds_mla
./run_vllm_test.sh -e presets/mpi/moe/GLM-5.2-INT8_dp16_tp4_ep64_compile_fp8_kvcache_alltoallv_v5.sh
```

#### 方式 2：固化为独立 Preset

将 Preset 中的 KV Cache 默认值由 `fp8` 改为 `int8`（或 `int8_ds_mla`）：

```bash
_VLLM_OPTIONAL_ARGS+=" --kv-cache-dtype ${USER_VLLM_KV_CACHE_DTYPE:-int8}"
```

### 5.2 生效确认与注意事项

1. **必须清理编译缓存**：从 `fp8` KV Cache 切换为 `int8` KV Cache 时，务必执行 `rm -rf vllm_scripts/vllm_cache vllm_scripts/torch_compile_cache_opt`，确保 AOTInductor 重新编译包含 `int8_ds_mla` 的图。
2. **日志核对**：在 `mpi_workers.log` 或 `vllm_head_log.txt` 的 Engine 初始化配置行中，确认同时出现：
   - `quantization=xcpu_int8`（全量 Int8 权重）或原 `quantization=fp8`（第一阶段混合模式）
   - `kv_cache_dtype=int8`（或 `int8_ds_mla`）
3. **验证顺序建议**：建议先用 `USER_VLLM_EAGER_OR_NOT="--enforce-eager"` + `USER_VLLM_KV_CACHE_DTYPE=int8` 跑通单条与并发请求确认输出无乱码，再切回 Compile 模式测吞吐与延迟。

---

## 6. 定量分析与选型依据参考

本节系统汇总将 `GLM-5.2-FP8`（`float8_e4m3fn` + `[128, 128]` FP32 scale）迁移至 **二进制补码 `Int8 [-128, 127]` + `UE8M0 [16, 16]`（W8A16）** 背后的实测方法与定量依据，涵盖 8-bit 编码与 SIMD 反量化权衡、不同块粒度（`[32, 32]` / `[16, 16]` / `MXINT8 [1, 32]`）的实测误差、子图消融以及 `TP4 + EP32` 下的单节点显存增量。

---

### 6.1 实测方法与验证基线

为了在**不下载完整 `1.5 TB` 原始 BF16 权重**的前提下完成高精度定量评估，实验设计了四级验证链路：

1. **按需轻量化抽样（ModelScope HTTP Range）**：
   - 解析 `ZhipuAI/GLM-5.2`（282 个分片，约 `1.50 TB`）的 `model.safetensors.index.json`，对目标分片先用 HTTP `206 Partial Content` Range 请求仅拉取 `8 B` 长度头 + JSON Header，再按 `data_offsets` 精确拉取目标张量字节流。
   - **子图与全深度基线抽样**：拉取 `Layers 0, 6, 38, 74`（覆盖首层 Dense + Full Indexer、浅/中/深 MoE + Full Indexer，每层含完整 Attn、Indexer、Shared Expert 与 `experts.{0, 42, 128, 255}`），共 `110` 个张量、合计仅 **`2.78 GiB`**（占原模型体积 `0.185%`）；块粒度与编码格式横向对比则按相同 Range 协议在内存中按需读取代表性矩阵（`L0` 异常值层 + `L10` MoE 路由专家），零磁盘残留。
2. **本地 13 层全深度扫描**：
   - 扫描本地完整 `GLM-5.2-FP8` 中 `Layers [0, 1, 2, 3, 6, 14, 26, 38, 50, 62, 74, 77, 78]` 共 `156` 个矩阵，统计块内峰值因子（Crest Factor，$\text{CF} = \text{amax} / \text{rms}$）与 `UE8M0` 指数范围。
   - 实测全部 `156` 个矩阵的 `UE8M0` 指数均严格落在 **`[108, 122]`**（即 $2^{-19} \sim 2^{-5}$），距离 SIMD 无分支快路径安全边界 `[1, 246]` 留有超过 100 档余量，不存在 subnormal（`u = 0`）或 BF16 溢出风险。
3. **底层算子逐比特校验与双计算路径对拍**：
   - 直接调用 `libtorch_xcpu_impl.so` 中的 `xcpu_impl::int8_block_dequant_impl<uint8_t>` 在 `mcpu` 设备上校验全部 $256 \times 256 = 65,536$ 种 `(int8, ue8m0)` 组合，确认与 `FP32 乘 + RNE 转 BF16` 黄金参考 **100% 逐比特一致**。
   - 在 `L0` 真实矩阵上对拍两种 W8A16 计算组织方式——**路径 1（SIMD 反量化将 `UE8M0` 直接折入 BF16 指数后沿全 `K` 连续累加）** vs **路径 2（逐块无缩放转 BF16 点积后在累加器端乘 FP32 scale）**：实测 `Dice diff <= 7.0e-10`、`RelL2 <= 0.0037%`，证明预折入 BF16 指数在数值上完全等价且免去了频繁中断累加器的开销。
   - 对 6.2 节讨论的原码对称路线 3 电路（`scale ∈ [1, 247]` 共 `63,232` 组合）及 `E2M5` 32 项查表电路（`scale ∈ [10, 246]` 共 `60,672` 组合）均完成全量逐比特零误差验证。
4. **误差评估口径与离线 MSE 优选（严格区分连续张量误差与离散选词指标）**：
   - **权重误差（越小越好）**：在 `float64` 累加精度下计算相对均方根误差 $\text{W\_RelRMSE} = \|W - \hat{W}\|_2 / \|W\|_2$。
   - **单算子 `mcpu` GEMM 误差（越小越好）**：采用重尾通道分布激活 `X [256, K]`（BF16），直接调用 `torch_xcpu.ops.initialize_fp8_linear`（`(128, 128)`）与 `initialize_int8_linear`（`(32, 32)` / `(16, 16)`）在 `mcpu` 设备上实测硬件 W8A16 BF16 输出相对 FP32 黄金结果的 `RelL2 = ||Y - Y_ref||_2 / ||Y_ref||_2`。
   - **子图连续张量误差 vs 离散选词指标（分开建表、严禁混列）**：
     - 对输出为连续浮点张量的子图——**MLA Attention 输出**（`seq_len = 512`）、**DSA Sparse Indexer 打分矩阵 `Score Logits [T_q, T_kv]`**（`T_kv = 4096`）、**Dense MLP** 与 **MoE SwiGLU FFN**——统一采用**相对 L2 误差 `RelL2`（越小越好）**横向对比；
     - 对 DSA Sparse Indexer 经 `topk(logits, 2048)` 产出的**离散 Token 编号集合**（无法对编号本身取范数），单独在专项表中统计**选词失配率 `Miss@2048 = 1 - |S ∩ S_ref| / 2048`（越小越好）**及对应的**选词召回率 `Recall@2048`（越大越好）**。
   - **离线 MSE 指数搜索（零推理开销）**：相比严格无裁剪上界指数 `u_ceil`，在 `{u_ceil - 2, u_ceil - 1, u_ceil}`（见 `convert_glm_int8_ue8m0.py`）中选择块 L2 重建误差最小的指数（允许极个别尖峰裁剪至 `[-128, 127]`），让块内其余绝大多数元素步长减半，推理期格式与开销完全不变。

---

### 6.2 8-bit 编码解释与 SIMD 反量化权衡

对照 `torch_xcpu/docs/ops/int8_ue8m0_dequant_simd_design.md` 中的五级 SIMD 路线（路线 1：`vcvt.bf16.s8/s16`；路线 2：`vcvt.f16.s16` 桥接；路线 3：`vfsub.bf16` 直接摆位减法；路线 4：`vclz + vshl` 纯整数；路线 5：`128` 项 LUT），下表对比四种典型 8-bit 编码在精度与 SIMD 反量化便利性上的综合权衡：

| 8-bit 编码格式 | 数值语义与<br>有效码点数 | 能否跳过<br>整数归一化 | SIMD 路线 1 / 2<br>(`vcvt.bf16` / `vcvt.f16`) | SIMD 路线 3<br>(`vfsub.bf16` 摆位) | SIMD 路线 4 / 5<br>(`vclz` / LUT) | 生态兼容性与<br>W8A8 硬件演进 |
| --- | --- | :---: | --- | --- | --- | --- |
| **二进制补码 `int8_t`**<br>（**本方案采用**） | `[-128, 127]`<br>256 个唯一值 | 否<br>（需隐式 `1.` 归一化） | **最优（3~4 / 6~7 条）**<br>硬件原生补码输入；<br>Xeon 8592+ 单核 `46.5 GB/s` | **5~6 条**<br>`exp_base \| (raw&0x7F)`<br>减 `exp_base + (raw&0x80)`<br>精确区间 `[1, 246]` | 8~12 条<br>（含 `vabs` 与<br>`-128` 特判） | **最佳**：标准 `torch.int8`<br>无缝兼容未来硬件<br>**W8A8 INT8 MAC** |
| **原码对称 `S1M7`**<br>（Sign-Magnitude） | `[-127, 127]`<br>255 值（含 `±0`）<br>精度微损 `~0.008%` | 否<br>（`mag` 仍未归一化，<br>不可直接摆入 BF16 尾数） | **劣化（+2~3 条指令）**<br>需先剥离符号位与处理 `-0`，<br>破坏硬件 `vcvt` 直通 | **4~5 条（省 1 条）**<br>`a=(sext&0x807F)\|ebase`<br>`b=a&0xFF80`，`vfsub(a,b)`<br>精确区间 `[1, 247]` | 7~9 条<br>（省 `vabs` 与<br>`-128` 补丁） | **差**：非标字节语义，<br>无法直连 VNNI / AMX-INT8<br>或 NPU INT8 矩阵核 |
| **偏移二进制**<br>(`uint8 - 128`) | `[-128, 127]`<br>256 个唯一值 | 否 | 需额外减 `128 * scale`<br>或先 `^ 0x80` 转补码 | **不可直接摆位**<br>8-bit 幅值溢出 7-bit 尾数，<br>必须先 `^ 0x80` 还原补码 | 同补码 + `^ 0x80` | 严格被补码支配 |
| **窄指数浮点 `E2M5`**<br>（全有限值 `1s+2e+5m`） | 正半轴分段线性<br>`0..63` 步长 1<br>`64..126` 步长 2<br>`128..252` 步长 4 | **正规数可跳过**<br>（`exp>0` 直接左移 2 位<br>加偏置；`exp==0` 共 32 项<br>恰好塞满 1 条 ZMM 表） | 不适用 `vcvt`；<br>512b SIMD 下纯整数+<br>单表置换 **6 条指令**<br>（与 `E4M3` 同构） | 混合分支约 7~8 条<br>（窄向量无 32 项单指令<br>查表时反而不如补码） | 依赖 32 项查表<br>（64B 表宽） | **差**：私有非标 dtype，<br>仅适合软件解包 W8A16，<br>不支持标准 W8A8 MAC |

**核心结论**：
- **不采用原码对称表示**：它在最高优先级的硬件路线 1/2 上反向变慢；即便在路线 3 上省下 1 条简单位运算，在访存与 MMA 耗时掩盖下几乎无端到端收益，却永久破坏标准 `int8_t` 生态并断送硬件 W8A8 演进路径。
- **不采用私有 `E2M5`**：虽然它在 `[32, 32]` 下依靠核心区 2× 加密步长获得了更低误差，但非标且在无 32 项单指令查表的窄向量 NPU 上反量化更重；通过将标准补码 `int8_t` 的块粒度细化至 `[16, 16]` 即可在保持 100% 标准 `torch.int8` 的前提下取得接近的精度收益。

---

### 6.3 块粒度与格式精度实测对比（`[128, 128]` vs `[32, 32]` vs `[16, 16]` vs `MXINT8 [1, 32]`）

对于均匀量化 `Int8 + UE8M0`，块内相对量化误差由块内峰值因子 $\text{CF}$ 与有效桶数 $Q_{\text{eff}} \in [64, 127]$ 决定：

$$\text{RelRMSE}_{\text{int8}} \approx \frac{\text{CF}}{\sqrt{12} \cdot Q_{\text{eff}}}$$

官方 `FP8 (E4M3FN)` 受限于 3-bit 显式尾数，无论 `[128, 128]` FP32 scale 还是 `MXFP8 [32, 32] / [1, 32]` UE8M0 scale，权重相对误差均卡死在 **`~2.64%–2.66%`** 地板；而 `Int8 + UE8M0` 借由更高线性尾数精度与小块隔离异常值，显著突破该上限。

下表汇总 **GLM-5.2 真实权重（BF16 原版与本地 FP8 版）**在不同格式与块粒度下的实测误差（格式为 **`Weight RelRMSE / mcpu 算子 W8A16 GEMM RelL2`**，全部越小越好）：

| GLM-5.2 代表矩阵<br>(`shape`) | 峰值因子 `CF`<br>(`p50 / p99`)<br>`[32,32]` $\to$ `[16,16]` $\to$ `[1,32]` | 官方 `FP8`<br>`[128, 128]` FP32<br>(基线) | `INT8 [32, 32]`<br>MSE<br>(从 BF16) | `INT8 [16, 16]`<br>Ceil<br>(从 BF16) | `INT8 [16, 16]`<br>**MSE (推荐)**<br>(从 BF16) | `MXINT8 [1, 32]`<br>MSE<br>(从 BF16) | `E2M5 [32, 32]`<br>MSE (对照)<br>(从 BF16) | 从本地 `FP8` 直接转码<br>`[32,32]` $\to$ **`[16,16]`**<br>(Weight / `mcpu` GEMM) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `L0 kv_a_proj`<br>`[576, 6144]`<br>*(极端异常值)* | `4.51 / 13.56`<br>$\to$ `3.58 / 8.26`<br>$\to$ `2.45 / 4.00` | `2.617%`<br>/ `2.661%` | `2.344%`<br>/ `2.326%` | `1.667%`<br>/ `1.644%` | **`1.623%`**<br>/ **`1.615%`** | `1.040%`<br>/ `1.053%` | `1.319%`<br>/ `1.333%` | `3.500%` $\to$ **`3.068%`**<br>*(GEMM `3.087%`)* |
| `L0 q_a_proj`<br>`[2048, 6144]`<br>*(浅层高异常)* | `6.22 / 11.62`<br>$\to$ `4.72 / 8.25`<br>$\to$ `2.36 / 3.62` | `2.639%`<br>/ `2.650%` | `2.114%`<br>/ `2.132%` | `1.714%`<br>/ `1.754%` | **`1.651%`**<br>/ **`1.681%`** | `0.856%`<br>/ `0.877%` | `1.212%`<br>/ `1.223%` | `3.368%` $\to$ **`3.101%`**<br>*(GEMM `3.113%`)* |
| `L0 q_b_proj`<br>`[16384, 2048]`<br>*(浅层 Q 投影)* | `6.53 / 12.03`<br>$\to$ `4.82 / 8.42`<br>$\to$ `3.06 / 4.88` | `2.631%`<br>/ `2.654%` | `2.203%`<br>/ `2.184%` | `1.757%`<br>/ `1.759%` | **`1.696%`**<br>/ **`1.699%`** | `1.178%`<br>/ `1.177%` | — | `3.418%` $\to$ **`3.119%`**<br>*(GEMM `3.132%`)* |
| `L0 o_proj`<br>`[6144, 16384]`<br>*(输出投影)* | `4.07 / 8.75`<br>$\to$ `3.37 / 6.26`<br>$\to$ `2.33 / 3.36` | `2.646%`<br>/ `2.654%` | `1.421%`<br>/ `1.422%` | `1.233%`<br>/ `1.240%` | **`1.188%`**<br>/ **`1.196%`** | `0.822%`<br>/ `0.838%` | — | `2.995%` $\to$ **`2.892%`**<br>*(GEMM `2.895%`)* |
| `L0 indexer.wk`<br>`[128, 6144]`<br>*(稀疏索引 Key)* | `5.22 / 13.75`<br>$\to$ `3.88 / 8.62`<br>$\to$ `2.51 / 4.35` | `2.644%`<br>/ `2.622%` | `1.935%`<br>/ `1.892%` | `1.535%`<br>/ `1.512%` | **`1.477%`**<br>/ **`1.468%`** | `0.949%`<br>/ `0.945%` | `1.110%`<br>/ `1.110%` | `3.263%` $\to$ **`3.015%`**<br>*(GEMM `2.989%`)* |
| `L0 indexer.wq_b`<br>`[4096, 2048]`<br>*(稀疏索引 Q)* | `6.62 / 13.64`<br>$\to$ `4.76 / 8.47`<br>$\to$ `3.10 / 4.86` | `2.636%`<br>/ `2.676%` | `2.298%`<br>/ `2.381%` | `1.793%`<br>/ `1.855%` | **`1.726%`**<br>/ **`1.782%`** | `1.156%`<br>/ `1.217%` | — | `3.484%` $\to$ **`3.140%`**<br>*(GEMM `3.193%`)* |
| `L10 exp0.gate`<br>`[2048, 6144]`<br>*(MoE 占 97.8%)* | `3.42 / 4.47`<br>$\to$ `3.02 / 4.10`<br>$\to$ `2.33 / 3.35` | `2.646%`<br>/ `2.652%` | `1.025%`<br>/ `1.039%` | `0.960%`<br>/ `0.972%` | **`0.911%`**<br>/ **`0.925%`** | `0.814%`<br>/ `0.830%` | `0.804%`<br>/ `0.807%` | `2.827%` $\to$ **`2.789%`**<br>*(GEMM `2.792%`)* |
| `L10 exp0.down`<br>`[6144, 2048]`<br>*(MoE 占 97.8%)* | `3.41 / 4.42`<br>$\to$ `3.01 / 4.09`<br>$\to$ `2.32 / 3.34` | `2.646%`<br>/ `2.658%` | `1.009%`<br>/ `1.018%` | `0.951%`<br>/ `0.965%` | **`0.905%`**<br>/ **`0.919%`** | `0.814%`<br>/ `0.830%` | — | `2.821%` $\to$ **`2.788%`**<br>*(GEMM `2.794%`)* |

> [!WARNING]
> **FP8 源权重非整除块边界陷阱（已修复）**：
> `self_attn.kv_a_proj_with_mqa.weight` 形状为 `[576, 6144]`，其在 `GLM-5.2-FP8`（块大小 `[128, 128]`）下的 `weight_scale_inv` 形状为 `[5, 48]`（因为 `ceil(576 / 128) = 5`）。反量化此类非整除矩阵时，**必须按源 `weight_block_size = [128, 128]`（或 2 的幂块长推断）展开 Scale**，切勿误写为 `math.ceil(576 / 5) = 116` 反推块高——否则每块错位 12 行并把尾部 `k_pe` 的大 Scale 污染进 `kv_c`，会导致 `kv_a_proj` 权重误差暴涨至 `94.5%`、MLA 子图误差恶化至 `31%–91%`。`vllm_scripts/tools/convert_glm_int8_ue8m0.py` 的 `dequantize_fp8_block_weight` 及本地已转换的 `GLM-5.2-INT8-UE8M0-16x16` 权重均已完成修复与逐层校验。

> **关键发现**：
> 1. **`[16, 16]` 是 W8A16 下的精度/开销最佳平衡点**：相比 `[32, 32]`，它在浅层异常值矩阵上用 **1/8 于 `MXINT8` 的 Scale 体积**拿走了 **55%–60%** 的误差降幅（`2.1%–2.3%` $\to$ `1.6%–1.7%`）；在占全模型 97.8% 权重的 MoE 路由专家上，误差降至 **`0.905%–0.911%`**（且直接从本地 FP8 转码的二次噪声增量仅 `+0.14%`，总误差 `2.79%`）。
> 2. **跨子空间与跨 Head 零混叠**：GLM-5.2 中 `kv_a_proj_with_mqa`（`512 kv_c + 64 k_pe = 576`）、`q_b_proj`（`192 q_nope + 64 q_pe = 256`）和 `kv_b_proj`（`192 k_nope + 256 v = 448`）在原 `[128, 128]` 下均存在 `576 % 128 != 0`、`448 % 128 != 0` 的跨子空间/跨 Head 共享 Scale 污染；`[16, 16]` 与 `[32, 32]` 均可彻底消除该污染，且 `[16, 16]` 进一步兼容 16 列半块切片。
> 3. **延伸对照（W8A8 演进潜力）**：若未来硬件升级至原生 `MXINT8 [1, 32]` W8A8（激活同步在线 `[1, 32]` 量化），实测上述四类代表矩阵的 W8A8 GEMM RelRMSE 仅 **`1.41%–1.54%`**，不仅远优于 `MXFP8` W8A8（`3.79%–3.87%`），甚至仍优于官方 `FP8` W8A16（`2.65%`）。

---

### 6.4 子图消融与分阶段替换优先级

为避免连续张量误差与离散选词准确率混淆，下面将 **所有子图的连续浮点输出相对误差（`RelL2`，全部越小越好）** 与 **DSA Sparse Indexer 离散 Top-2048 选词失配率/召回率** 分表列出，覆盖 `[32, 32]` 与 **`[16, 16]`** 双粒度实测：

#### 1. 四类核心子图连续张量相对误差对比（统一指标：`RelL2`，越小越好）

| 端到端子图与评估层<br>（对比完整 BF16 子图输出） | 全 `FP8`<br>`[128, 128]`<br>(现状基线) | 全 `Int8_MSE`<br>`[32, 32]`<br>(从 BF16) | **全 `Int8_MSE`**<br>**`[16, 16]`**<br>**(从 BF16)** | 全 `FP8 -> Int8`<br>`[16, 16]`<br>(修正 128 块后) | `[16, 16]` 关键部分替换消融<br>(`RelL2`，越小越好) |
| --- | :---: | :---: | :---: | :---: | --- |
| **MLA Attention 输出**<br>`L00 / L06 / L38 / L74`<br>(`seq_len = 512`) | `1.62%` / `3.47%`<br>/ `5.51%` / `6.35%` | `1.04%` / `1.87%`<br>/ `2.29%` / `2.77%` | **`0.90%` / `1.50%`**<br>/ **`2.06%` / `2.47%`** | `1.83%` / `3.77%`<br>/ `5.84%` / `6.75%` | 敏感度排序：`kv_b ≈ o_proj > kv_a > q_b ≈ q_a`<br>`[16, 16]` 相比 FP8 将各层误差压缩 **2.3×–2.7×** |
| **DSA Indexer 打分矩阵**<br>`L00 / L06 / L38 / L74`<br>(`Score Logits [32, 4096]`) | `3.48%` / `3.14%`<br>/ `4.22%` / `6.21%` | `2.17%` / `2.01%`<br>/ `2.13%` / `3.22%` | **`1.64%` / `1.53%`**<br>/ **`1.77%` / `2.70%`** | **`3.51%` / `3.18%`**<br>/ **`4.35%` / `6.34%`**<br>*(保留 `wk=FP8`)* | **`wk=BF16, wq_b=I8[16]`**：`0.62% / 0.63% / 1.03% / 1.31%`<br>若把 `wk` 也从 FP8 二次转码：<br>恶化至 `3.78% / 3.42% / 4.55% / 6.72%` |
| **Dense MLP 输出**<br>`L00` (`SwiGLU`) | `3.22%` | `1.46%` | **`1.34%`** | `3.47%` | `[16, 16]` 较 `[32, 32]` 再降 `0.12%`，<br>较原版 FP8 误差降低 **2.4×** |
| **MoE FFN (Shared+Routed)**<br>`L06 / L38 / L74`<br>(`SwiGLU` 组合输出) | `4.03%` / `4.51%`<br>/ `4.42%` | `1.68%` / `1.71%`<br>/ `1.86%` | **`1.44%` / `1.62%`**<br>/ **`1.63%`** | `4.25%` / `4.76%`<br>/ `4.67%`<br>*(仅换 `w13`: `4.15%–4.65%`)* | **仅换 Routed `w13`**：**`2.97% / 3.13% / 3.07%`**<br>*(连 Shared `w13` 一起换：`2.75% / 2.93% / 2.90%`)*<br>仅换 Routed `w2`：`3.40% / 3.88% / 3.79%` |

#### 2. DSA Sparse Indexer 离散选词集合专项评估（`T_kv = 4096` 中选 `Top-2048`）

DSA Indexer 的最终作用是通过 `torch.topk(logits, 2048)` 为稀疏 MLA 筛选出 `2048` 个历史 KV Token 索引。由于输出为离散集合，下表同时给出与上表同方向的 **选词失配率 `Miss@2048`（漏选比例，越小越好）** 与直观的 **选词召回率 `Recall@2048 = 1 - Miss@2048`（命中比例，越大越好）**：

| Indexer 权重配置 | `L00`<br>`Miss@2048` (`Recall`) | `L06`<br>`Miss@2048` (`Recall`) | `L38`<br>`Miss@2048` (`Recall`) | `L74`<br>`Miss@2048` (`Recall`) | 结论 |
| --- | :---: | :---: | :---: | :---: | --- |
| **全 `FP8 [128, 128]`**（现状基线） | `1.45%` (`98.55%`) | `1.21%` (`98.79%`) | `1.67%` (`98.33%`) | `2.14%` (`97.86%`) | 深层 `L74` 约漏选 `44 / 2048` 个候选词 |
| **全 `Int8_MSE [32, 32]`**（从 BF16） | `0.83%` (`99.17%`) | `0.79%` (`99.21%`) | `0.83%` (`99.17%`) | `1.12%` (`98.88%`) | 漏选率较 FP8 接近减半 |
| **全 `Int8_MSE [16, 16]`**（从 BF16） | **`0.66%` (`99.34%`)** | **`0.60%` (`99.40%`)** | **`0.72%` (`99.28%`)** | **`0.95%` (`99.05%`)** | **全深度漏选率均压入 `<1.0%`（Recall `>99%`）** |
| **`wk=BF16` + `wq_b=Int8 [16, 16]`** | **`0.28%` (`99.72%`)** | **`0.27%` (`99.73%`)** | **`0.45%` (`99.55%`)** | **`0.54%` (`99.46%`)** | `wk` 全网仅 `17 MB`，保留 BF16 召回逼近 `100%` |
| **`wk=FP8(保留)` + `wq_b=FP8->I8[16]`** | **`1.47%` (`98.53%`)** | **`1.23%` (`98.77%`)** | **`1.70%` (`98.30%`)** | **`2.23%` (`97.77%`)** | **本地 FP8 转换生产方案**：基本无损保持原 FP8 水平 |
| 反例：`wk, wq_b` 均从 FP8 二次转码 | `1.55%` (`98.45%`) | `1.32%` (`98.68%`) | `1.79%` (`98.21%`) | `2.33%` (`97.67%`) | MQA 共享 `wk` 二次量化会额外放大全序列排序偏移 |

#### 3. 分模块替换优先级与转码策略汇总

| 模块类别 | 逻辑矩阵形状与数量<br>(78 主干层 + 1 MTP) | 占量化参数比 | 从 BF16 量化<br>(`Int8 [16,16] MSE`) | 从本地 FP8 直接转码<br>(`Int8 [16,16] MSE`) | 阶段策略与依据 |
| --- | --- | :---: | --- | --- | --- |
| **MoE Routed `w13`**<br>(`gate_proj`, `up_proj`) | `[2048, 6144]` × 2<br>× 256 专家 × 76 层 | **65.18%**<br>(`489.6 GB`) | 极优：`~0.91%`<br>（较 FP8 降近 3 倍） | **高可行**：`~2.79%`<br>（增量仅 `+0.14%`） | **第一阶段 P0（`--scope moe_w13`）**<br>SwiGLU 自乘放大使单换 `w13` 比单换 `w2`<br>多降 `0.43%–0.75%` MoE 子图误差 |
| **MoE Routed `w2`**<br>(`down_proj`) | `[6144, 2048]`<br>× 256 专家 × 76 层 | **32.59%**<br>(`244.8 GB`) | 极优：`~0.91%` | **高可行**：`~2.79%`<br>（增量仅 `+0.14%`） | **第二阶段 P0**<br>(`--scope all`) |
| **Dense MLP &<br>Shared Experts** | Dense `12288×6144` (L0..2)<br>Shared `2048×6144` (76 层) | **0.47%**<br>(`3.55 GB`) | 极优：`~0.95%–1.30%`<br>（L0 子图 `1.34%`） | 可行：`~2.80%–2.95%` | **第二阶段 P1**<br>(`--scope all`) |
| **MLA Attention &<br>`indexer.wq_b`** | `q_a/kv_a/q_b/kv_b/o_proj`<br>及 22 层 `indexer.wq_b` | **1.76%**<br>(`13.2 GB`) | 优：`1.0%–1.73%`<br>（MLA 子图 `0.90%–2.47%`） | 可用：修正 `kv_a` 128 块后<br>`~2.89%–3.14%` | **第二阶段 P1**<br>（BF16 源效果最佳；FP8 源务必按 `[128,128]` 解块） |
| **DSA `indexer.wk`** | `[128, 6144]`<br>× 22 Full Indexer 层 | **0.002%**<br>(`17 MB`) | 优：`1.48%`（或留 BF16） | **禁止二次转码**<br>（Score 误差升至 `6.72%`） | **默认原样保留**<br>（由 vLLM 加载期直接升至 BF16） |

---

### 6.5 `TP4 + EP32` 分布式部署下的单卡与单节点显存增量核算

#### 1. Scale 体积倍率规律
由于 `FP8` 与 `INT8` 权重本体均为 **1 Byte/元素**，权重体积变化 **100% 来自 Scale 张量差值**。原版 `GLM-5.2-FP8` 使用 **4 字节 FP32** 存储 `[128, 128]` Scale（即 `4 B / 16,384 = 1 B / 4,096` 元素）；改用 **1 字节 `UE8M0`** 本身带来 **4× 压缩**，因此：
- **`[32, 32]` `UE8M0`**（`1 B / 1,024`）：Scale 字节数为原版的 **$4\times$**（净增 `+0.073%`）；
- **`[16, 16]` `UE8M0`**（`1 B / 256`）：Scale 字节数为原版的 **$16\times$**（净增 `+0.366%`）；
- **`MXINT8 [1, 32]` `UE8M0`**（`1 B / 32`）：Scale 字节数为原版的 **$128\times$**（净增 `+3.101%`）。

#### 2. `TP4 + EP32`（`DP8 × TP4 = 32 Ranks`）切分与复制明细
按默认 `vllm_scripts/user_env_template.sh` 配置（78 层主干 `L0..77`，`VLLM_SHARED_EXPERT_DISABLE_TP=1` / SP-MoE 开启，且 `kv_b_proj` 与 `indexer.wk` 在加载期自动反量化为 BF16 并释放原量化权重及 Scale），单 Rank（1/32）的运行期常驻权重拆解如下：

| 模块 | 并行切分方式<br>(`TP4 + EP32`) | 单 Rank 本地矩阵尺寸与层数 | 单 Rank<br>权重本体 | 单 Rank Scale<br>`FP8 [128,128]` | 单 Rank Scale<br>`INT8 [32,32]` | 单 Rank Scale<br>**`INT8 [16,16]`** | 单 Rank Scale<br>`MXINT8 [1,32]` |
| --- | --- | --- | :---: | :---: | :---: | :---: | :---: |
| **Routed MoE**<br>(`w13` + `w2`) | **EP32 切分**<br>(每 Rank 8 完整专家) | `[4096, 6144]` + `[6144, 2048]`<br>× 8 专家 × 75 层 | `22.649 GB` | `5.53 MB` | `22.12 MB` | **`88.47 MB`** | `707.79 MB` |
| **Shared Experts**<br>(`w13` + `w2`) | **全 Rank 复制**<br>(`DISABLE_TP=1` / SP) | `[4096, 6144]` + `[6144, 2048]`<br>× 75 层 | `2.831 GB` | `0.69 MB` | `2.76 MB` | **`11.06 MB`** | `88.47 MB` |
| **MLA Attention**<br>(`qkva` / `q_b` / `o_proj`) | `qkva` **复制**；<br>`q_b` / `o_proj` **TP4** | `[2624,6144]` + `[4096,2048]`<br>+ `[6144,4096]` × 78 层 | `3.875 GB` | `0.95 MB` | `3.78 MB` | **`15.14 MB`** | `121.09 MB` |
| **DSA Indexer**<br>(`wq_b`) | **全 Rank 复制**<br>(21 Full 层) | `[4096, 2048]` × 21 层 | `0.176 GB` | `0.04 MB` | `0.17 MB` | **`0.69 MB`** | `5.51 MB` |
| **Dense MLP**<br>(`gate_up` + `down`) | **TP4 切分**<br>(L0..2 共 3 层) | `[6144, 6144]` + `[6144, 3072]`<br>× 3 层 | `0.170 GB` | `0.04 MB` | `0.17 MB` | **`0.66 MB`** | `5.31 MB` |
| **固定 BF16/FP32**<br>(含展开 `kv_b`/`wk`) | `embed`/`lm_head`/`kv_b` TP4；<br>Norm/Gate/`wk` 复制 | `embed+lm_head` `0.95 GB`<br>`kv_b` BF16 `0.57 GB` 等 | `1.815 GB` | `0` | `0` | **`0`** | `0` |
| **单 Rank 合计** | — | **78 主干层运行期常驻** | **`31.505 GB`** | **`7.26 MB`** | **`29.01 MB`** | **`116.02 MB`** | **`928.16 MB`** |

#### 3. 单 Rank 与单节点总权重及相比 FP8 净增量汇总（78 主干层）

下表同时给出 **单 Rank（1/32）**、标准 **4-Rank 节点（每节点 1 组 TP4，共 8 节点，对应 `MPC_SIZE = 4`）** 与 **8-Rank 节点（每节点 2 组 TP4，共 4 节点）** 的精确统计（`1 GB = 10^9 B`，括号内为二进制 `GiB / MiB`）：

| 方案<br>（全模型统一粒度） | 单 Rank (1/32)<br>总权重 / Scale | 单 Rank<br>**相比 FP8 净增** | **4-Rank 节点 (1×TP4, 8 节点)**<br>总权重 (`GiB`) / Scale | **4-Rank 节点**<br>**相比 FP8 净增** | **8-Rank 节点 (2×TP4, 4 节点)**<br>**相比 FP8 净增** |
| --- | :---: | :---: | :---: | :---: | :---: |
| **原版 `FP8 [128, 128]`**<br>FP32 Scale（基线） | `31.512 GB` (`29.35 GiB`)<br>Scale `7.26 MB` | 基线 `0` | `126.05 GB` (`117.39 GiB`)<br>Scale `29.04 MB` | 基线 `0`<br>(`+0.000%`) | 基线 `0` |
| **`INT8 + UE8M0`<br>`[32, 32]`** | `31.534 GB` (`29.37 GiB`)<br>Scale `29.01 MB` | `+21.75 MB`<br>(`+20.74 MiB`, `+0.069%`) | `126.14 GB` (`117.47 GiB`)<br>Scale `116.02 MB` | **`+86.99 MB`**<br>(`+82.96 MiB`, `+0.069%`) | **`+173.97 MB`**<br>(`+165.91 MiB`) |
| **`INT8 + UE8M0`<br>`[16, 16]`（推荐）** | `31.621 GB` (`29.45 GiB`)<br>Scale `116.02 MB` | **`+108.76 MB`**<br>(`+103.72 MiB`, `+0.345%`) | `126.48 GB` (`117.80 GiB`)<br>Scale `464.08 MB` | **`+435.05 MB`**<br>(`+414.89 MiB`, **`+0.345%`**) | **`+870.09 MB`**<br>(`+829.78 MiB`) |
| **`MXINT8`<br>`[1, 32]`** | `32.433 GB` (`30.21 GiB`)<br>Scale `928.16 MB` | `+920.90 MB`<br>(`+878.24 MiB`, `+2.922%`) | `129.73 GB` (`120.82 GiB`)<br>Scale `3,712.65 MB` | **`+3.684 GB`**<br>(`+3.431 GiB`, `+2.922%`) | **`+7.367 GB`**<br>(`+6.861 GiB`) |

> **边界情形补充**：
> - **开启 MTP 预测解码（加载第 79 层 `layers.78`）**：4-Rank 节点基线总权重升至 `127.84 GB`，三种方案每节点净增仅微增至 **`+88.15 MB`**（`[32, 32]`）、**`+440.87 MB`**（`[16, 16]`）、**`+3.733 GB`**（`[1, 32]`）。
> - **若关闭共享专家复制（非 SP-MoE，`VLLM_SHARED_EXPERT_DISABLE_TP=0`）**：4-Rank 节点净增略降至 `+81.65 MB`（`[32, 32]`）、`+408.38 MB`（`[16, 16]`）、`+3.458 GB`（`[1, 32]`）。
> - **结论**：在 4-Rank 节点上，全模型统一使用 `[16, 16]` 仅增加 **`435 MB`**（不足 0.44 GB，占节点总权重 `0.345%`，比 `MXINT8 [1, 32]` 的 `3.68 GB` 省 **8.5 倍**），换来全网统一算子契约与显著的浅层精度提升，性价比最高。

---

### 6.6 W8A16 SIMD 与 `[16, 16]` 块粒度工程适配要点

相比 `[1, 32]`，`[16, 16]` 在 W8A16 SIMD 反量化路径上具有明显工程优势：

1. **完整保留二维块特性**：
   - 保留 **16 行行间 Scale 复用**（条带化外循环一次解码广播供 16 行共享），避免 `[1, 32]` 在行主序 `[N, ldb]` 布局下逐行重算 Scale；
   - 保留 **转置不变性**（`[16, 16]^T = [16, 16]`），不会像 `[1, 32]` 在转置后退化为 `[32, 1]`。
2. **256-bit / 512-bit SIMD 均能满宽零损耗运行**：
   - **256-bit SIMD（16×BF16）**：单块单行 16 个 `int8` 恰好对应 **1 条满宽向量**；
   - **512-bit SIMD（32×BF16）**：一次处理相邻两个 `[16, 16]` 块（共 32 元素），在 16 行外循环头用 1 条向量拼接指令构造双半幅广播参数 `param_vec = [dup16(s0) | dup16(s1)]`（摊薄到 $16 \times 32 = 512$ 个权重上仅多 1 条指令），**内层 32 通道核心转换指令（见 `torch_xcpu/docs/ops/int8_ue8m0_dequant_simd_design.md`）保持 100% 不变**。
3. **采用「SIMD 预折入 BF16 指数 + 全 `K` 连续累加」**：
   - 由于 AMX 等矩阵核单条点积指令（如 `_tile_dpbf16ps`）的硬件 `K` 深度为 `32`，无法在 `K = 16` 处中断出栈乘标量；将 `UE8M0` 在 SIMD Unpack 阶段直接折入权重 BF16 指数位后，矩阵核即可无视 `block_k = 16` 沿完整 `K` 维连续累加。
   - `torch_xcpu` 底层已同步放开 `block_k % 16 == 0`（`torch_xcpu/torch_xcpu_impl/src/int8_block_dequant.cpp`）并将 `mk::kernels::kMinScaleBlock` 调整为 `16`（`torch_xcpu/torch_xcpu_impl/include/mk/kernels/gemm.h`），可直接承接 `[16, 16]` 全链路联调。
