# GLM-5.2 / GLM-5.3 Int8 + UE8M0 (`[16, 16]`) 权重转换与部署指南

本文档面向内网部署与算子联调团队，说明如何从**本地模型目录**将 `GLM-5.2-FP8` 或 `GLM-5.3`（FP8；待 `GLM-5.3-BF16` 导入本地后同样直接适用）转换为 **Int8 `[-128, 127]` + UE8M0 `[16, 16]` W8A16** 格式，并指导在不同算子适配阶段如何通过 Preset 启动推理服务（含 `DP16 / TP4 / EP64` 与 **Int8 权重 + Int8 KV Cache** 组合）。

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
