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
📦 读取压缩文件: trace_merged.json.gz
   → 解压完成
====================================================================================================
vLLM Trace 增强分析: ./8ranks/trace_merged.json.gz
====================================================================================================

检测到进程数: 8
  World-Rank-0         (PID 0):       1764 个事件
  World-Rank-1         (PID 1):       1764 个事件
  World-Rank-2         (PID 2):       1764 个事件
  World-Rank-3         (PID 3):       1764 个事件
  World-Rank-4         (PID 4):       1764 个事件
  World-Rank-5         (PID 5):       1764 个事件
  World-Rank-6         (PID 6):       1764 个事件
  World-Rank-7         (PID 7):       1764 个事件

关键字过滤: allreduce, alltoallv (区分大小写: 否)

====================================================================================================
Top 11 操作 - 增强分析
====================================================================================================

[1] moe_finalize_mpi_alltoallv
----------------------------------------------------------------------------------------------------
  📊 迭代范围: 跳过前 50% (432 次) 迭代，分析 5 次 (总共 864 次迭代，分析 5 次)

  [A] 传统分析（基于平均时间）:
  ------------------------------------------------------------------------------------------------
    World-Rank-0        : 总计=          15297 µs | 平均=   3059.38 µs | 次数=   5 [范围 433-437]
    World-Rank-1        : 总计=          19163 µs | 平均=   3832.55 µs | 次数=   5 [范围 433-437]
    World-Rank-2        : 总计=          15526 µs | 平均=   3105.23 µs | 次数=   5 [范围 433-437]
    World-Rank-3        : 总计=          20001 µs | 平均=   4000.18 µs | 次数=   5 [范围 433-437]
    World-Rank-4        : 总计=           8075 µs | 平均=   1614.92 µs | 次数=   5 [范围 433-437]
    World-Rank-5        : 总计=          16394 µs | 平均=   3278.72 µs | 次数=   5 [范围 433-437]
    World-Rank-6        : 总计=           9958 µs | 平均=   1991.56 µs | 次数=   5 [范围 433-437]
    World-Rank-7        : 总计=          15565 µs | 平均=   3112.98 µs | 次数=   5 [范围 433-437]
    → 传统不均衡度:  79.52% (最大/最小 = 2.48x) ⚠️ 严重不均衡
    → 最小平均时间:    1614.92 µs (World-Rank-4)
    → 中位平均时间:    3109.11 µs (World-Rank-7)
    → 平均平均时间:    2999.44 µs
    → 最大平均时间:    4000.18 µs (World-Rank-3)

  [B] 方差分析（Rank 内部稳定性）:
  ------------------------------------------------------------------------------------------------
    World-Rank-0        : 均值=   3059.38 µs | 标准差=   3434.06 µs | CV=112.25% ⚠️ 高方差 [5 次迭代]
      最小=     59.24 µs (第 435 次调用, 开始于 753567173333 µs)
      中位=   1497.37 µs (第 434 次调用, 开始于 753567165328 µs)
      平均=   3059.38 µs
      最大=   8696.94 µs (第 433 次调用, 开始于 753567148429 µs)
    ......
    World-Rank-7        : 均值=   3112.98 µs | 标准差=   1894.91 µs | CV= 60.87% ⚠️ 高方差 [5 次迭代]
      最小=   1386.66 µs (第 435 次调用, 开始于 753567171997 µs)
      中位=   2956.09 µs (第 434 次调用, 开始于 753567163898 µs)
      平均=   3112.98 µs
      最大=   5950.70 µs (第 433 次调用, 开始于 753567151207 µs)
    → ⚠️  警告: 检测到高方差: World-Rank-0, World-Rank-1, World-Rank-2, World-Rank-3, World-Rank-4, World-Rank-5, World-Rank-6, World-Rank-7

  [C] 逐迭代不均衡度分析（显示 5 次迭代，索引 433-437）:
  ------------------------------------------------------------------------------------------------
    iter     433: World-Rank-0=  8697µs | World-Rank-1=  8714µs | World-Rank-2=  4704µs | World-Rank-3=  9249µs
                  World-Rank-4=  5944µs | World-Rank-5=  8393µs | World-Rank-6=    51µs | World-Rank-7=  5951µs
                  不均衡度=142.32% ⚠️; 最小 World-Rank-6=51µs；最大 World-Rank-3=9249µs；中位 World-Rank-7=7172µs
    iter     434: World-Rank-0=  1497µs | World-Rank-1=  2394µs | World-Rank-2=  2739µs | World-Rank-3=   644µs
                  World-Rank-4=    80µs | World-Rank-5=  2747µs | World-Rank-6=  3365µs | World-Rank-7=  2956µs
                  不均衡度=160.02% ⚠️; 最小 World-Rank-4=80µs；最大 World-Rank-6=3365µs；中位 World-Rank-1=2566µs
    iter     435: World-Rank-0=    59µs | World-Rank-1=  2564µs | World-Rank-2=  2303µs | World-Rank-3=  3085µs
                  World-Rank-4=  1924µs | World-Rank-5=  1299µs | World-Rank-6=  2471µs | World-Rank-7=  1387µs
                  不均衡度=160.39% ⚠️; 最小 World-Rank-0=59µs；最大 World-Rank-3=3085µs；中位 World-Rank-2=2113µs
    iter     436: World-Rank-0=  3820µs | World-Rank-1=  5181µs | World-Rank-2=  3876µs | World-Rank-3=  4478µs
                  World-Rank-4=    55µs | World-Rank-5=  2501µs | World-Rank-6=  2097µs | World-Rank-7=  3829µs
                  不均衡度=158.72% ⚠️; 最小 World-Rank-4=55µs；最大 World-Rank-1=5181µs；中位 World-Rank-0=3825µs
    iter     437: World-Rank-0=  1224µs | World-Rank-1=   310µs | World-Rank-2=  1904µs | World-Rank-3=  2546µs
                  World-Rank-4=    72µs | World-Rank-5=  1454µs | World-Rank-6=  1974µs | World-Rank-7=  1442µs
                  不均衡度=181.14% ⚠️; 最小 World-Rank-4=72µs；最大 World-Rank-3=2546µs；中位 World-Rank-7=1448µs

    → 最小逐迭代不均衡度: 142.32% (迭代 433)
    → 中位逐迭代不均衡度: 160.02% (迭代 434)
    → 平均逐迭代不均衡度: 160.52%
    → 最大逐迭代不均衡度: 181.14% (迭代 437)
    → ⚠️  严重: 逐迭代不均衡度 (160.5%) >> 传统不均衡度 (79.5%)
       这表明存在交替式负载不均 - 不同迭代中不同的 Rank 表现较慢！

  [D] 时间同步分析（时间偏差分析）:
  ------------------------------------------------------------------------------------------------
    迭代 433: 时间偏差 = 9210 µs ⚠️
    迭代 434: 时间偏差 = 3333 µs ⚠️
    迭代 435: 时间偏差 = 3028 µs ⚠️
    迭代 436: 时间偏差 = 5163 µs ⚠️
    迭代 437: 时间偏差 = 2523 µs ⚠️

====================================================================================================

......

====================================================================================================
总体总结
====================================================================================================

关键发现:
  1. 传统分析显示所有 Rank 的平均负载分布
  2. 方差分析检测性能不稳定的 Rank（高 CV）
  3. 逐迭代分析揭示交替式负载不均模式
  4. 时间分析识别 Rank 之间的时间偏差

优化建议:
  - 高方差 per Rank: 检查 CPU 绑定、频率缩放或后台进程
  - 高逐迭代不均衡度: 调查动态负载分配或缓存效应
  - 大时间偏差: 检查同步原语和网络延迟
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
