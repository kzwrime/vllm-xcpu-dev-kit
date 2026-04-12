# vLLM Trace 分析工具 - 快速参考卡片

## 📦 核心工具

| 工具 | 说明 |
|------|------|
| `analyze_vllm_enhanced_cn.py` | 中文版，支持 json.gz 的横向/纵向不均衡度分析 |
| `merge_traces.py` | 合并多个 trace JSON/GZ 文件 |
| `filter_trace_by_keywords.py` | 从已合并 trace 中筛选指定算子关键词，导出新的 JSON/GZ 文件 |

---

## 🔄 Trace 合并 (merge_traces.py)

### 功能特性
- ✅ 自动处理 `.json.gz` 文件（自动解压）
- ✅ 输出 `.json.gz` 压缩格式（节省空间）
- ✅ 支持混合输入（`.json` 和 `.json.gz` 可以混用）
- ✅ 自动检测文件格式
- ✅ 支持通配符（`trace_*.json.gz`）

### 最常用的 5 个命令

#### 1. 合并压缩文件，输出压缩
```bash
python merge_traces.py trace_rank0.json.gz trace_rank1.json.gz \
  -o trace_merged.json.gz
```

#### 2. 使用通配符合并所有文件
```bash
python merge_traces.py trace_*.json.gz -o trace_merged.json.gz
```

#### 3. 合并并自定义进程名称
```bash
python merge_traces.py trace_*.json.gz \
  -n "GPU0" "GPU1" "GPU2" "GPU3" \
  -o trace_merged.json.gz
```

#### 4. 合并压缩文件，输出未压缩
```bash
python merge_traces.py trace_*.json.gz -o trace_merged.json
```

#### 5. 合并 complex 目录下的所有文件
```bash
python merge_traces.py complex/*world-rank*.pt.trace.json.gz \
  -o complex/trace_merged.json.gz
```

### 真实输出示例

```text
================================================================================
Merging Chrome Trace Files
================================================================================

Processing 1774868837715880692-world-rank-0-rank-0.1774868963633912666.pt.trace.json.gz as 'Rank 0'...
Reading 1774868837715880692-world-rank-0-rank-0.1774868963633912666.pt.trace.json.gz...
  → Decompressed 1774868837715880692-world-rank-0-rank-0.1774868963633912666.pt.trace.json.gz
  → Loaded 330,795 events

Processing 1774868837715880692-world-rank-1-rank-1.1774868963555750818.pt.trace.json.gz as 'Rank 1'...
Reading 1774868837715880692-world-rank-1-rank-1.1774868963555750818.pt.trace.json.gz...
  → Decompressed 1774868837715880692-world-rank-1-rank-1.1774868963555750818.pt.trace.json.gz
  → Loaded 330,566 events

Processing 1774868839363981232-world-rank-2-rank-0.1774868963582132514.pt.trace.json.gz as 'Rank 2'...
Reading 1774868839363981232-world-rank-2-rank-0.1774868963582132514.pt.trace.json.gz...
  → Decompressed 1774868839363981232-world-rank-2-rank-0.1774868963582132514.pt.trace.json.gz
  → Loaded 332,123 events

Processing 1774868839363981232-world-rank-3-rank-1.1774868963573484081.pt.trace.json.gz as 'Rank 3'...
Reading 1774868839363981232-world-rank-3-rank-1.1774868963573484081.pt.trace.json.gz...
  → Decompressed 1774868839363981232-world-rank-3-rank-1.1774868963573484081.pt.trace.json.gz...
  → Loaded 331,694 events

================================================================================
Writing to trace_merged.json.gz...
  → Compressed: 98.23 MB

✅ Merge complete!
   Total events: 1,325,178
   Number of processes: 4
   Output: trace_merged.json.gz
================================================================================
```

---

## ✂️ Trace 关键词筛选 (filter_trace_by_keywords.py)

### 功能特性
- ✅ 自动处理 `.json.gz` 文件（自动解压）
- ✅ 支持输出 `.json` 或 `.json.gz`
- ✅ 支持一个或多个关键词（OR 匹配）
- ✅ 默认不区分大小写
- ✅ 保留 `process/thread` 元数据，输出文件可继续用 Chrome trace 或分析脚本打开

### 最常用的 4 个命令

#### 1. 只筛选 alltoallv
```bash
python filter_trace_by_keywords.py trace_merged.json.gz \
  -k alltoallv \
  -o trace_alltoallv.json.gz
```

#### 2. 一次筛选多个通信算子
```bash
python filter_trace_by_keywords.py trace_merged.json.gz \
  -k allreduce alltoallv reduce_scatter \
  -o trace_comm.json.gz
```

#### 3. 输出未压缩 JSON
```bash
python filter_trace_by_keywords.py trace_merged.json.gz \
  -k moe_prepare_alltoallv moe_finalize_mpi_alltoallv \
  -o trace_moe_comm.json
```

#### 4. 区分大小写匹配
```bash
python filter_trace_by_keywords.py trace_merged.json.gz \
  -k AllToAllV \
  -o trace_AllToAllV.json.gz \
  --case-sensitive
```

### 使用场景

当 `trace_merged.json.gz` 很大、Chrome trace 打开太慢，或者你只想单独查看某类通信算子时，先筛出一个小文件再观察或分析：

```bash
python merge_traces.py 8ranks/*world-rank*.pt.trace.json.gz \
  -o 8ranks/trace_merged.json.gz

python filter_trace_by_keywords.py 8ranks/trace_merged.json.gz \
  -k alltoallv groupgemm \
  -o 8ranks/trace_alltoallv.json.gz

python analyze_vllm_enhanced_cn.py 8ranks/trace_alltoallv.json.gz --count 5
```

---

## 📊 Trace 分析 (analyze_vllm_enhanced_cn.py)

### 功能特性
- ✅ 中文输出，专有名词保留英文
- ✅ 自动处理 `.json.gz` 文件
- ✅ 支持关键字过滤（如 allreduce, alltoallv）
- ✅ 支持迭代范围控制（skip, count, from-iter, to-iter）
- ✅ 四维分析：传统、方差、迭代、时序
- ✅ 自动检测交替式负载不均
- ✅ Rank 很多时逐迭代详情自动换行（默认每行最多 4 个 Rank）

### 最常用的 6 个命令

#### 1. 跳过指定数量次调用，分析 decode 阶段
```bash
python analyze_vllm_enhanced_cn.py trace.json --skip 96
```

#### 2. 从 50% 开始，分析 5 次（建议）
```bash
python analyze_vllm_enhanced_cn.py trace.json --skip-percent 50 --count 5
```

#### 3. 分析通信操作（关键字过滤）（建议）
```bash
python analyze_vllm_enhanced_cn.py trace.json \
  -k allreduce moe_prepare_alltoallv moe_finalize_mpi_alltoallv
```

#### 4. 快速预览（只分析前 5 次）（建议）
```bash
python analyze_vllm_enhanced_cn.py trace.json --count 5
```

#### 5. 分析特定范围
```bash
python analyze_vllm_enhanced_cn.py trace.json --from-iter 50 --to-iter 100
```

#### 6. Rank 很多时控制逐迭代详情每行显示数量
```bash
# 默认：每行最多 4 个 Rank
python analyze_vllm_enhanced_cn.py 8ranks/trace_merged.json.gz --count 3 -k allreduce

# 自定义：每行最多 8 个 Rank
python analyze_vllm_enhanced_cn.py 8ranks/trace_merged.json.gz \
  --count 3 -k allreduce --iteration-ranks-per-line 8
```

### 真实输出示例

```text
====================================================================================================
vLLM Trace 增强分析: ./complex/trace_merged_4ranks.json
====================================================================================================

检测到进程数: 4
  World-Rank-0         (PID 0):        784 个事件
  World-Rank-1         (PID 1):        784 个事件
  World-Rank-2         (PID 2):        784 个事件
  World-Rank-3         (PID 3):        784 个事件

关键字过滤: allreduce, moe_prepare_alltoallv, moe_finalize_mpi_alltoallv (区分大小写: 否)

====================================================================================================
Top 3 操作 - 增强分析
====================================================================================================

[1] moe_finalize_mpi_alltoallv
----------------------------------------------------------------------------------------------------
  📊 迭代范围: 跳过前 50% (96 次) 迭代，分析 5 次 (总共 192 次迭代，分析 5 次)

  [A] 传统分析（基于平均时间）:
  ------------------------------------------------------------------------------------------------
    World-Rank-0        : 总计=         231757 µs | 平均=  46351.40 µs | 次数=   5 [范围 97-101]
    World-Rank-1        : 总计=         142231 µs | 平均=  28446.13 µs | 次数=   5 [范围 97-101]
    World-Rank-2        : 总计=         159827 µs | 平均=  31965.35 µs | 次数=   5 [范围 97-101]
    World-Rank-3        : 总计=         258638 µs | 平均=  51727.61 µs | 次数=   5 [范围 97-101]
    → 传统不均衡度:  58.76% (最大/最小 = 1.82x) ⚠️ 严重不均衡

  [B] 方差分析（Rank 内部稳定性）:
  ------------------------------------------------------------------------------------------------
    World-Rank-0        : 均值=  46351.40 µs | 标准差=  17229.80 µs | CV= 37.17% ⚠️ 高方差 [5 次迭代]
    World-Rank-1        : 均值=  28446.13 µs | 标准差=  32406.11 µs | CV=113.92% ⚠️ 高方差 [5 次迭代]
    World-Rank-2        : 均值=  31965.35 µs | 标准差=  33948.96 µs | CV=106.21% ⚠️ 高方差 [5 次迭代]
    World-Rank-3        : 均值=  51727.61 µs | 标准差=  40048.43 µs | CV= 77.42% ⚠️ 高方差 [5 次迭代]
    → ⚠️  警告: 检测到高方差: World-Rank-0, World-Rank-1, World-Rank-2, World-Rank-3

  [C] 逐迭代不均衡度分析（显示 5 次迭代，索引 97-101）:
  ------------------------------------------------------------------------------------------------
    迭代  97: World-Rank-0= 68395µs | World-Rank-1= 73588µs | World-Rank-2=  1512µs | World-Rank-3= 80110µs | 不均衡度=140.60% ⚠️
    迭代  98: World-Rank-0= 58699µs | World-Rank-1= 50395µs | World-Rank-2= 44878µs | World-Rank-3=  1172µs | 不均衡度=148.32% ⚠️
    迭代  99: World-Rank-0= 45046µs | World-Rank-1= 17060µs | World-Rank-2=  1238µs | World-Rank-3= 20834µs | 不均衡度=208.17% ⚠️
    迭代 100: World-Rank-0= 29922µs | World-Rank-1=   568µs | World-Rank-2= 29578µs | World-Rank-3= 59664µs | 不均衡度=197.43% ⚠️
    迭代 101: World-Rank-0= 29695µs | World-Rank-1=   620µs | World-Rank-2= 82622µs | World-Rank-3= 96857µs | 不均衡度=183.49% ⚠️

    → 平均逐迭代不均衡度: 175.60%
    → 最大逐迭代不均衡度: 208.17% (迭代 99)
    → ⚠️  严重: 逐迭代不均衡度 (175.6%) >> 传统不均衡度 (58.8%)
       这表明存在交替式负载不均 - 不同迭代中不同的 Rank 表现较慢！

  [D] 时间同步分析（时间偏差分析）:
  ------------------------------------------------------------------------------------------------
    迭代 97: 时间偏差 = 78849 µs ⚠️
    迭代 98: 时间偏差 = 57528 µs ⚠️
    迭代 99: 时间偏差 = 44450 µs ⚠️
    迭代 100: 时间偏差 = 59090 µs ⚠️
    迭代 101: 时间偏差 = 96231 µs ⚠️

====================================================================================================
```

---

## 🎛️ 迭代控制参数速查

| 参数 | 说明 | 示例 |
|------|------|------|
| `--skip N` | 跳过前 N 次 | `--skip 10` |
| `--skip-percent P` | 跳过前 P% | `--skip-percent 50` |
| `--count N` | 只分析 N 次 | `--count 5` |
| `--from-iter N` | 从第 N 次开始 | `--from-iter 100` |
| `--to-iter N` | 到第 N 次结束 | `--to-iter 200` |
| `--iteration-ranks-per-line N` | [C] 每行最多显示 N 个 Rank | `--iteration-ranks-per-line 8` |

---

## 📊 分析维度说明

| 维度 | 名称 | 说明 |
|------|------|------|
| [A] | 传统分析 | 基于平均时间，检测某些 Rank 始终慢 |
| [B] | 方差分析 | 检测 Rank 内部性能不稳定（高 CV） |
| [C] | 逐迭代分析 | 检测交替式负载不均（不同迭代不同 Rank 慢） |
| [D] | 时间同步分析 | 检测各 Rank 启动时间偏差 |

---

## ⚠️ 阈值速查

### 传统不均衡度
| 阈值 | 含义 |
|------|------|
| < 5% | ✓ 均衡 |
| 5-20% | ⚡ 中度 |
| > 20% | ⚠️ 严重 |

### CV（变异系数）
| 阈值 | 含义 |
|------|------|
| < 15% | ✓ 稳定 |
| 15-30% | ⚡ 波动 |
| > 30% | ⚠️ 高方差 |

### 逐迭代不均衡度
| 对比 | 含义 |
|------|------|
| ≈ 传统 | 一致 |
| >> 传统 (2x) | ⚠️ 交替不均 |

### 时间偏差
| 阈值 | 含义 |
|------|------|
| < 0.1ms | ✓ 同步良好 |
| > 0.1ms | ⚠️ 需关注 |

---

## 💡 组合使用示例

### 跳过前 20%，分析 10 次，只看 allreduce
```bash
python analyze_vllm_enhanced_cn.py trace.json \
  --skip-percent 20 --count 10 -k allreduce
```

### 分析第 50-100 次迭代的 MoE 操作
```bash
python analyze_vllm_enhanced_cn.py trace.json \
  --from-iter 50 --to-iter 100 -k moe
```

### 8 个 Rank：逐迭代详情默认按每行 4 个 Rank 自动换行
```bash
python merge_traces.py 8ranks/*world-rank*.pt.trace.json.gz \
  -o 8ranks/trace_merged.json.gz

python analyze_vllm_enhanced_cn.py 8ranks/trace_merged.json.gz \
  --count 3 -k allreduce
```

### 8 个 Rank：逐迭代详情改成每行 8 个 Rank
```bash
python analyze_vllm_enhanced_cn.py 8ranks/trace_merged.json.gz \
  --count 3 -k allreduce --iteration-ranks-per-line 8
```

### 跳过前 10 次（warmup），详细分析
```bash
python analyze_vllm_enhanced_cn.py trace.json \
  --skip 10 --top 5
```

### 完整工作流：合并 + 分析
```bash
# Step 1: 合并 trace 文件
python merge_traces.py complex/*world-rank*.pt.trace.json.gz \
  -o complex/trace_merged.json.gz

# Step 2: 按关键词筛选出更小的 trace
python filter_trace_by_keywords.py complex/trace_merged.json.gz \
  -k allreduce alltoallv \
  -o complex/trace_comm.json.gz

# Step 3: 分析筛选后的文件
python analyze_vllm_enhanced_cn.py complex/trace_comm.json.gz \
  --skip-percent 50 --count 5 \
  -k allreduce alltoallv
```

---


## 🔧 获取帮助

```bash
# Merge 工具帮助
python merge_traces.py --help

# 关键词筛选工具帮助
python filter_trace_by_keywords.py --help

# 分析工具帮助
python analyze_vllm_enhanced_cn.py --help
```
