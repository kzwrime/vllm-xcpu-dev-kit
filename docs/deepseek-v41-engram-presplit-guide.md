# DeepSeek-V4.1-Flash Engram 权重预拆分与 `EngramParallel` 加载指南

本文档说明如何使用离线脚本将 `DeepSeek-V4.1-Flash` 的超大 Engram Embedding 表预拆分为可控大小的 safetensors 分片，并在开启 `EngramParallel`（`VLLM_XCPU_ENABLE_ENGRAM_PARALLEL=1`）时按需流式加载，以满足“单次 `mmap` 需整文件 `malloc`”的小内存硬件部署约束。

---

## 1. 背景与适用范围

- **硬件与内存约束**：在目标小内存硬件上，`safetensors.safe_open` 对 `.safetensors` 文件做 `mmap` 时需要直接 `malloc` 整个文件的大小。
- **单张量瓶颈**：`DeepSeek-V4.1-Flash` 的两层 Engram 表 `layers.{1,14}.engram.embed.weight` 单张量即达约 `91.56 GiB`（Layer 1 `[384006168, 256]`、Layer 14 `[384016682, 256]`，`float8_e4m3fn`），对应 `layers.{1,14}.engram.embed.scale` 单张量约 `2.86 GiB`（`float8_e8m0fnu`）。常规按整张量切分无法将文件压到单张量体积以下。
- **拓扑无关预拆分**：离线阶段沿第 0 维（行维）按指定字节上限预拆分，单份输出目录可通用于任意 `W = DP × TP` 并行度配置，无需针对特定并行规模重复切分。
- **运行时支持范围**：
  - 当前运行时仅 **`EngramParallel` 模式**（`VLLM_XCPU_ENABLE_ENGRAM_PARALLEL=1`）支持读取 `.row_slice.*` 预拆分权重。
  - `EngramParallelModelLoader` 同时保持对**未拆分原始 checkpoint** 的向后兼容。
  - 若需使用 `EngramDisagg`（A+E 分离）或单机本地模式，请仍使用未拆分的原始权重目录。

---

## 2. 离线预拆分（`tools/reshard_hf_safetensors.py`）

### 2.1 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `src` | 必填 | 源 Hugging Face / ModelScope 模型目录（自动复制 `config.json`、tokenizer 等非权重文件） |
| `dst` | 必填 | 目标输出目录（清理旧 `.safetensors` 与索引文件后重写） |
| `--target-shard-size` | `1GB` | 普通非 Engram 权重的目标分片大小（支持 `B`、`KB`、`MB`、`GB`、`KiB`、`MiB`、`GiB`） |
| `--engram-shard-size` | `None` | 单个 Engram 行分片的目标最大数据字节数；传入该参数即自动启用 `--split-engram` |
| `--split-engram` | 关闭 | 启用 `*.engram.embed.{weight,scale}` 按行维预拆分；未显式传 `--engram-shard-size` 时默认沿用 `--target-shard-size`，但当 `--target-shard-size < 16 MiB` 时自动回退为 `1GB`（防止将普通权重拆到最小粒度时误把 Engram 切成上亿个单行文件） |

> 脚本同时支持对**已预拆分的输出目录**再次执行 `reshard_hf_safetensors.py`：
> - 传入新的 `--engram-shard-size` 时，会流式跨源分片按新阈值重新切分；
> - 不传 `--split-engram` / `--engram-shard-size` 时，会仅重排普通权重分片并原样保留现有 Engram 行分片（仍各自独占文件）及 `engram_row_slices` 元数据。

### 2.2 常用转换命令示例

#### 场景 A：普通权重与 Engram 表统一限制在指定分片上限（推荐）

```bash
python3 tools/reshard_hf_safetensors.py \
  /path/to/DeepSeek-V4.1-Flash \
  /path/to/DeepSeek-V4.1-Flash-presplit-512MiB \
  --target-shard-size 512MiB \
  --engram-shard-size 512MiB
```

#### 场景 B：普通权重切分到单张量最小粒度，同时控制 Engram 单分片上限

```bash
python3 tools/reshard_hf_safetensors.py \
  /path/to/DeepSeek-V4.1-Flash \
  /path/to/DeepSeek-V4.1-Flash-min-shard-engram-512MiB \
  --target-shard-size 1B \
  --engram-shard-size 512MiB
```

若仅传 `--target-shard-size 1B --split-engram` 而省略 `--engram-shard-size`，脚本会提示并自动对 Engram 表使用 `1GB` 分片上限。

### 2.3 分片产物格式与快速抽检

1. **分片键名与独占文件**：原始表 `<base>`（匹配 `*.layers.<id>.engram.embed.{weight,scale}`）按连续行区间 `[row_start, row_end)` 切分为 `<base>.row_slice.<row_start>_<row_end>`，每个行分片**独占一个 `model-XXXXX-of-YYYYY.safetensors` 文件**，绝不与 `wkv`、`q_weight`、`k_weight` 或专家权重混放。
2. **索引元数据**：`model.safetensors.index.json` 在标准 `weight_map` 之外额外写入 `metadata.engram_row_slices`，记录原始 `shape`、`dtype`、`num_chunks` 与 `rows_per_chunk`。

转换完成后可执行快速抽检：

```bash
python3 - <<'PY'
import json
from pathlib import Path

out_dir = Path("/path/to/DeepSeek-V4.1-Flash-presplit-512MiB")
index = json.loads((out_dir / "model.safetensors.index.json").read_text())
print("total shards:", len(set(index["weight_map"].values())))
print("engram_row_slices:")
for name, info in index.get("metadata", {}).get("engram_row_slices", {}).items():
    print(f"  {name}: {info}")
PY
```

---

## 3. `EngramParallel` 运行时加载与启动配置

### 3.1 启动配置要点

1. 确认开启 MPI 与 `EngramParallel` 模式（与 `VLLM_XCPU_ENABLE_ENGRAM_DISAGG` 互斥）：
   ```bash
   export VLLM_CPU_USE_MPI=1
   export VLLM_XCPU_ENABLE_ENGRAM_PARALLEL=1
   export VLLM_XCPU_ENABLE_ENGRAM_DISAGG=0
   ```
2. 将模型路径（如 `USER_VLLM_MODEL` 或 `--model`）指向预拆分后的目录即可，无需额外 CLI 开关：
   - 当 `VLLM_XCPU_ENABLE_ENGRAM_PARALLEL=1` 时，`vllm-xcpu-plugin` 会自动将 `--load-format auto` / `safetensors` 路由到 `EngramParallelModelLoader`（`xcpu_engram_parallel_auto` / `xcpu_engram_parallel_safetensors`）。
   - `--load-format` 仅支持 `auto`、`safetensors` 或 `dummy`；不支持 `enable_multithread_load` 与 `safetensors_load_strategy != "lazy"`（`None` 或 `"lazy"` 均可）。

### 3.2 运行时内存峰值保障机制

1. **按需映射与即时释放（峰值最多 1 个分片）**：
   - `EngramParallelModelLoader` 仅读取 `model.safetensors.index.json` 解析分片区间，再对与本 rank `[lo, hi)` 行区间有交集的 `.row_slice.*` 分片文件逐个调用 `safe_open`。
   - 每个重叠分片通过 `uint8` 视图精确拷贝目标行后即释放切片并 `del handle` 触发即时 `munmap`，任意时刻最多只有 1 个 Engram 分片文件处于 `mmap` 状态。
2. **空切片与草稿模型零打开**：
   - 当全局并行度使某 rank 在某个 Engram 层上分得 `0:0` 空切片时（例如 `W = 48` 下每个 rank 只持有两层中的一层），`EngramParallelModelLoader` 依据 `metadata.engram_row_slices` 直接构造 `(0, dim)` 空张量，对该层分片**零次 `safe_open`**。
   - 若同时加载 DSpark 等不含 `ParallelEngramEmbedding` 的草稿模型，loader 会直接跳过全部纯 Engram 分片文件（零次 `safe_open`）。
3. **VL 包装层流式加载**：
   - `XcpuDeepseekV41ForCausalLM.load_weights` 覆盖了上游 `DeepseekV41ForCausalLM.load_weights` 中一次性 `sorted(...)` 物化全模型张量的实现，改为将 `language_model.*` 权重以单一连续流直接送入 `AutoWeightsLoader`，丢弃 `StageMissingLayer` 视觉塔张量并仅缓存克隆少量非 `language_model.*` 权重在末尾加载，防止全模型大量分片的 `mmap` 句柄同时滞留内存。
