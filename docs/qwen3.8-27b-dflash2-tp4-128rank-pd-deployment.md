# 128-Rank NPU `Qwen3.8-27B + DFlash2 (TP4)` P/D 分离部署方案

> **适用目标**：`128 Rank` NPU 集群、`Qwen3.8-27B` + `DFlash2 (K=5)`、`TP=4`、Prefill `Chunk = 6144`、`KV Cache = 16~18 GiB/rank`、`Max Model Len = 180K~256K`、RDMA `MooncakeConnector` + `MooncakeStoreConnector` + `vllm-router`，服务**多用户 Agent 代码开发**场景。

---

## 1. 集群拓扑与单 Rank 显存基线

### 1.1 实例组织：`32 × vLLM-Cluster(TP=4, PP=1, DP=1)` + `vllm-router`

- **节点拆分**：`128 Ranks` 按 `TP=4, PP=1, DP=1` 划分为 **`32` 个独立 TP4 实例**（典型如 16 台 8 卡物理机、每机 2 组互不干扰的 TP4 实例，消除跨机 TP 通信）。
- **独立实例（`DP=1`）对比引擎内 `DP>1`**：`Qwen3.8-27B` 为 Dense + GDN 架构（非大 EP MoE）。若单引擎开 `DP>1`，vLLM 需做全 DP 全局同步步与空 rank dummy run，长 chunk 或单 rank 抖动会拖慢整组；拆为独立 `DP=1` 实例交由 `vllm-router` 调度，可获得独立连续批处理循环、精细前缀路由与单实例隔离容错。

### 1.2 单 Rank HBM 显存构成实测对比（`FP8/Int8` vs `BF16`）

单卡 HBM 占用遵循统一模型：

$$M_{\text{rank}} = M_{\text{target}} + M_{\text{draft}}(\text{role}) + M_{\text{KV}} + M_{\text{runtime}} + M_{\text{peak\_act}}(N_{\text{batch}}) + M_{\text{inductor}}$$

基于本地 `Qwen3.8-27B-FP8`（总盘重 `28.75 GiB` = 纯 LM `27.44 GiB` + 视觉 `0.86 GiB` + MTP `0.44 GiB`）与 `Qwen3.8-27B-DFlash2`（`3.58 GiB`，81 个 BF16 张量），在本套件 `McpuWorker TP=4`（`eager`，`bs=128, MAX_LEN=215040`，`--language-model-only`）下实测得到单 Rank 精确占用（`BF16` 因本地缓存目录为空故按 Safetensors 精确参数量加差 `+5.66 GiB/rank` 推算，`compile` 模式需额外计入 Inductor 中间内存池 $M_{\text{inductor}}$）：

| 精度方案 | Target 权重<br>（`TP=4` 每 Rank） | DFlash2 权重<br>（Prefill `ctx_only` / Decode） | 运行时缓冲 + 激活峰值<br>（P `6144` / D `1024`, `eager`） | KV Cache 池<br>（目标分档） | 单 Rank `eager` 基线总占用<br>（`16 / 17 / 18 GiB` KV） |
|---|---|---|---|---|---|
| **`Qwen3.8-27B-FP8/Int8`（推荐）** | **`7.63 GiB`**<br>（实测纯 Target 加载） | Prefill：**`0.33 GiB`**（`+1.18 GiB` 已释放）<br>Decode：**`1.52 GiB`**（含 XCPU packed linear） | Prefill：`0.36 + 0.79 =` **`1.15 GiB`**<br>Decode：`0.40 + 0.13 =` **`0.53 GiB`** | `16` / `17` / `18 GiB` | Prefill：**`25.11 / 26.11 / 27.11 GiB`**<br>Decode：**`25.68 / 26.68 / 27.68 GiB`** |
| **`Qwen3.8-27B`（BF16 推算）** | **`13.29 GiB`**<br>（`7.63 + 5.66 GiB`） | Prefill：**`0.33 GiB`**<br>Decode：**`1.52 GiB`** | Prefill：**`1.15 GiB`**<br>Decode：**`0.53 GiB`** | `16` / `17` / `18 GiB` | Prefill：**`30.77 / 31.77 / 32.77 GiB`**<br>Decode：**`31.34 / 32.34 / 33.34 GiB`** |

> **实测结论与 Prefill 硬约束**：
> 1. `FP8/Int8` 下 P/D 非 KV 基线占用分别为 **`9.11 GiB`** 与 **`9.68 GiB`**，配合 `16~18 GiB` KV 在 32 GiB HBM 硬件上可留出 `4.3~6.9 GiB/rank` 余量供图编译与系统开销；`BF16` 非 KV 基线已达 **`14.77~15.34 GiB`**，32 GiB HBM 卡无法稳定配置 `16~18 GiB` KV，需 `>= 40~48 GiB` HBM。
> 2. Prefill 端 `--speculative-config` 必须配置 `"prefill_context_kv_only": true`：实测单 Rank 权重从 `9.15 GiB` 降至 `7.96 GiB`（日志明确释放 **`1.18 GiB/rank`** 非 context KV 权重并跳过无效首拍 `propose`），同时保持 P/D 双端 15 个 KV Group 严格一致。

---

## 2. 混合 KV/GDN 状态几何、块粒度与批处理预算

### 2.1 `TP=4` 混合 KV / GDN 物理页布局（15 个 KV Group）

依据仓内模型 `config.json` 与 `vllm` `kv_cache_interface.py` 计算 `TP=4` 单 Rank 精确字节（5 层/组，`LBHNC`）：

| KV Group 类型 | 层数与分组 | `TP=4` 单 Rank 维度与精度 | 单 Token / 单页大小 |
|---|---|---|---|
| **FullAttention** | `16` 层<br>（补 4 padding $\rightarrow$ **`4` 组 × 5 层**） | KV heads: `4 / 4 = 1`<br>`head_dim=256`, BF16 (`2 B`) | `2(K+V) × 1 × 256 × 2`<br>= **`1,024 B/token/层`** |
| **GDN Linear Attn**<br>(`MambaSpec`) | `48` 层<br>（补 2 padding $\rightarrow$ **`10` 组 × 5 层**） | `conv`: `(16×128×2+48×128)/4=2560`,<br>行数 `(4-1)+5 = 8`, BF16 (`40 KiB`)；<br>`ssm`: `(12, 128, 128)`, FP32 (`768 KiB`) | **`808 KiB/页/层`**<br>（固定状态页；`conv` 行数由 `vllm_config.num_speculative_tokens=5` 决定，P/D 严格相等） |
| **DFlash2 SWA**<br>(`DFlashSlidingWindowSpec`) | `5` 层<br>（**`1` 组 × 5 层**，`sw=2048`） | KV heads: `8 / 4 = 2`<br>`head_dim=128`, BF16 (`2 B`) | `2(K+V) × 2 × 128 × 2`<br>= **`1,024 B/token/层`**<br>（准入封顶 `8` 块；运行时稳态尾部 `4` 块，跨界 `5` 块） |

### 2.2 块大小方案对比与混合块对齐切分机制（`--block-size 128` vs `256`）

#### (1) 当前代码下 `Chunk=6144 -> 5376` 的源码机理核验

依据本仓库当前 `vllm` 源码，在 Prefill Chunk 上限为 `6144` 且启用 `--block-size 128` 时，**单请求中间 chunk 向下取整为 `5376` 的行为依然成立**：
- 在 [platforms/interface.py](../vllm/vllm/platforms/interface.py#L1031-L1061) 中，开启 `--enable-prefix-caching` 会激活 `mamba_cache_mode="align"`，要求逻辑块字节满足 `B × 1024 B >= 808 KiB` 且为 `--block-size` 的整数倍；对应日志已实测打印 `Setting attention block size to 896 tokens`。
- 在 [scheduler.py](../vllm/vllm/v1/core/sched/scheduler.py#L485-L493) 的 `Scheduler._mamba_block_aligned_split` 中，当 `mamba_has_prefill_checkpoint_blocks = False` 时，任何未到达 `prefill_end` 的非末尾 chunk 都会执行 `aligned_end = end // block_size * block_size` 截断。
- 尽管 `#59468` 已停止从 `6144` 预算中扣减 `draft_slots=5`、`#54163` 已消除 DFlash2 末尾退块，但 `#60050` 合入的 GDN 64-token 内部 checkpoint 框架当前仅在 `gdn_prefill_backend == "flashinfer"` 下激活，`McpuPlatform` 仍维持 `mamba_has_prefill_checkpoint_blocks = False`。因此对于 `8192` 冷输入：`--block-size 128`（$B=896$）单请求仍按 `5376 / 2688 / 128` 三轮切分，而 `--block-size 256`（$B=1024$）按 `6144 / 2048` 两轮切分。

#### (2) 两档 `--block-size` 物理几何定量对比

| `--block-size` | 对齐逻辑块 $B$ | 单组块大小<br>（5 层/Rank） | GDN 页填充<br>浪费率 | 单请求 `Chunk=6144`<br>首轮有效计算量 |
|---|---|---|---|---|
| **`128`** | **`896` tokens** | **`4.375 MiB`** | **`+10.89%`** | `5376` tokens (`87.5%`，余 `768`) |
| **`256`** | **`1024` tokens** | **`5.000 MiB`** | **`+26.73%`** | **`6144` tokens** (`100.0%`，余 `0`) |

#### (3) 不同块大小方案的工程权衡分析

- **P/D 与存储三端一致性前提**：无论选择 `--block-size 128` 还是 `256`，只要 Prefill、Decode 与 Mooncake Store 三端保持同一参数，P/D KV 布局与 RDMA 传输均严格对齐。
- **`--block-size 128`（$B=896$）方案分析**：
  - **显存与池容积占优**：GDN 页对齐填充率仅为 `+10.89%`，每条活跃请求（`30` 个 GDN 保留块 + `8` 个 DFlash2 准入块）节省 `23.75 MiB/rank`，相同 `16/17/18 GiB` 下物理块数多出 `14.3%`。
  - **前缀颗粒度更细**：非整除长上下文命中尾块边界（`896` vs `1024`）平均重算量降低 `12.5%`。
  - **算力预算与补齐空间**：单请求长 chunk 截断为 `5376` 时，多请求并发队列下剩余 `768` token 预算仍可排入其余请求的尾块或短输入；若硬件支持且允许单批略增，将 Prefill 单独设为 `--max-num-batched-tokens 6272`（`7 × 896`）即可实现 `100%` 单请求满块。
- **`--block-size 256`（$B=1024$）方案分析**：
  - **严格匹配 `Chunk=6144`**：`6144 = 6 × 1024` 零余数对齐，对 `8K~16K` 冷输入可减少 `1` 轮 forward kernel launch 开销，降低调度和启动固定延迟。
  - **代价**：GDN 页填充率升至 `+26.73%`，且 `16~18 GiB` 池中单块粒度增大至 `5.000 MiB`。

### 2.3 Decode 实例批处理预算收敛分析（`--max-num-batched-tokens=1024`）

此前部分对比实验为对齐配置曾将 Decode 同样设为 `6144`（见 [`20261029_Qwen3.8-DFlash2-PD-prefill-chunk.md`](dev_log/20261029_Qwen3.8-DFlash2-PD-prefill-chunk.md)）。但 **`MooncakeConnector` 仅约束 P/D 的模型、`--block-size`、`prefix caching / mamba_cache_mode` 与 `--speculative-config` 一致，不要求 `--max-num-batched-tokens` 相等**。将 Decode 设为 **`1024`**（兼容 $B \in \{896, 1024\}$）具有四项确定性收益：

1. **稳态 Decode 实际 Token 数远小于 `1024`**：Decode 阶段每条请求每步最多消耗 `K + 1 = 6` 个 query tokens；经 Mooncake `N-1` 截断（[`mooncake_connector.py`](../vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py#L942-L972)）首步仅重算 `1` 个尾 token。即使并发达 `24`，单步也仅 `144 << 1024`。在 [`vllm.py`](../vllm/vllm/config/vllm.py#L2487) 中，`1024` 支持的最大活跃序列上限为 `1024 // 6 = 170`，不构成本方案 `12~16` 并发的上限限制。
2. **实测降低激活峰值 `83.5%` 并压缩编译内存池**：在本套件实测日志中（`bs=128`），将批量预算由 `6144` 收敛至 `1024` 后，Torch 激活峰值从 **`0.79 GiB/rank` 降至 `0.13 GiB/rank`**（净省 `0.66 GiB/rank`），内存剖析耗时从 **`22.87 s` 缩短至 `5.15 s`**；同时在 `compile` 模式下，[`worker_v1.py`](../vllm-xcpu-plugin/vllm_xcpu_plugin/worker/worker_v1.py#L395-L422) 以 `max_num_batched_tokens` 执行 `_dummy_run(N)` 并常驻 Inductor 中间内存池，收口至 `1024` 可避免 Decode 为虚高的 `6144` 形状长期锁死额外 HBM。
3. **大幅缩短服务冷启动 compile / warmup 时间**。
4. **控制抢占后本地重算的突发阻塞**：若极少数情况触发 Decode 请求抢占（此时 `do_remote_prefill` 已置 `False`），`1024` 将每次本地恢复步限制在最多 `1` 个逻辑块内，避免 `6144` 单次重算引发多秒级同批 `TPOT` 毛刺。

---

## 3. 变长上下文动态调度机制与 KV-Cache 容量搭配

### 3.1 变长上下文双门控调度、抢占边界与系统影响

静态配置 `--max-num-seqs` 与动态变长上下文并不冲突，vLLM V1 混合调度器通过两道运行时闸门协同工作：

1. **启动期校验与运行时双闸门（[scheduler.py](../vllm/vllm/v1/core/sched/scheduler.py#L758-L905)）**：
   - **启动期校验**：仅验证至少 `1` 条满 `max_model_len` 请求可入池（实测 `18 GiB, bs=128, MAX_LEN=215040` 下，EngineCore 日志明确报告：Prefill 池容量 **`869,446` tokens / `4.04x`**，Decode 池容量 **`874,482` tokens / `4.07x`**）。
   - **Gate 1（请求槽位护栏 `--max-num-seqs`）**：控制同时处于 `RUNNING` 状态的序列数上界 `num_running <= max_num_seqs`，在大量短请求（`4K~16K`）涌入时限制单步 batching 规模以守住 `TPOT SLO`。
   - **Gate 2（真实块按需分配 `KVCacheManager.allocate_slots`）**：引擎**不按 `max_model_len` 预占**，而按请求当前实达长度 $L(t)$ 动态申请物理块（DFlash2 准入封顶 `8` 块，运行时稳定维持尾部 `4` 块）：
     $$\text{Blocks}_{\text{steady}}(L) = 4 \cdot \left\lceil\frac{L + K + 1}{B}\right\rceil + 30\;(\text{GDN 保留块}) + 4\;(\text{DFlash2 稳态尾块})$$
     实例内任意混跑序列集 $\{L_i\}$ 的物理上限为 $\sum_{i \in \text{RUNNING}} \text{Blocks}(L_i) \le N_{\text{usable}}$。以 `18 GiB, bs=128`（`4,212` 可用块）为例，单实例按稳态块数精确可容纳：`56` 条 `8K` 请求（`74` 块/条，由 Gate 1 收敛至 `16` 条）、`16` 条 `48K` 请求（`254` 块/条，按 `85%` 水位稳态预留 `14` 条）、`6` 条 `128K` 请求（`622` 块/条）或 `4` 条 `210K` 满长请求。

2. **水位触顶与抢占恢复路径**：
   - 池余量不足时：先驱逐 `ref_cnt == 0` 的 LRU Prefix Cache 块；若仍不足（多因在飞 RUNNING 长会话同时 decode 增长越过块边界），[`_preempt_request`](../vllm/vllm/v1/core/sched/scheduler.py#L1551-L1595) 按 **LIFO** 踢出最新请求、释放其物理块并清零 `num_computed_tokens`。
   - **P/D 下的抢占代价**：因 Decode 接收完 P$\to$D 传输后已在 [`mooncake_connector.py`](../vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py#L1059) 置 `do_remote_prefill=False`，被抢占请求恢复时只能在 Decode 本地重新计算（若挂载 `MooncakeStore` 且启用 `save_decode_cache: true`，则可通过异步 Store 读取大部分历史前缀，仅本地重算最近未满块尾部）。

3. **对配置及 Infra 系统的架构约束**：
   - **引擎侧上限约束**：必须确保单条满长请求准入块占比 $\le 25\%$（保证最坏出现 `4` 条超长并发不引发抢占），同时通过 `max_num_seqs`（`12 / 14 / 16`）封顶中短上下文并行度。直接开到 `256K` 会使单请求占 `16 GiB` 池的 `32.3%~32.5%`（仅容 `3` 条，必触发抢占悬崖）并使单步 FullAttn 读量升至 `4.0 GiB/rank`。
   - **网关与客户端 Infra 约束**：`vllm-router` 默认 `--balance-abs-threshold=64` 在 `max_num_seqs<=16` 时完全失效，必须降至 `2~6` 以避免长短请求在不同 Decode 实例上堆叠失衡；客户端需设定显式 `auto-compact` 阈值，并生产开启 Decode `save_decode_cache: true`。

### 3.2 请求最大输出长度（`max_tokens` / `max_completion_tokens`）约束与治理

Coding Agent 客户端常携带显式的输出长度上限 $M$（OpenAI 接口字段 `max_completion_tokens` / `max_tokens`，例如 `16384` 或 `32768`）。该参数对 P/D 系统产生三项关键影响：

1. **API 入口总长强校验与 P/D 错配风险（[params.py](../vllm/vllm/renderers/params.py#L209-L241)、[api_utils.py](../vllm/vllm/entrypoints/serve/utils/api_utils.py#L169-L206)）**：
   - OpenAI 请求入队前的 `TokenizeParams` 强制校验 $\text{max\_input\_tokens} = \text{max\_model\_len} - M$（当 $L_{\text{in}} + M > \text{max\_model\_len}$ 时直接拒绝并返回 HTTP 400）。
   - **P/D 分离特有限制**：`vllm-router` 发往 Prefill 节点时会将请求覆写为 `max_tokens=1`，而发往 Decode 节点时透传客户端原 `max_tokens=M`。因此若客户端传入 $L_{\text{in}} = 190\text{K}, M = 32\text{K}$（和为 `222K > 215,040`），**Prefill 节点会按 `max_tokens=1` 通过并算完及传输 190K KV，随后 Decode 节点在校验时拒收报 HTTP 400**，白白浪费整次 Prefill 算力与 KV 传输。
2. **Decode 动态输出扩展预留**：
   - vLLM V1 调度器仅按实际当前步生成进度逐块扩充输出槽位（[scheduler.py](../vllm/vllm/v1/core/sched/scheduler.py#L655-L765)）。假设单条深度思考/生成输出达到 $M_{\text{actual}} = 8192$ tokens，每条请求需额外生长 `4 × ceil(8192 / 896) = 40` 块（`bs=128` 恰好 `175.00 MiB/rank`）或 `32` 块（`bs=256` 恰好 `160.00 MiB/rank`）；若达 `32768` tokens 需 `148` 块（`647.50 MiB/rank`）或 `128` 块（`640.00 MiB/rank`）。
3. **工程治理规范**：
   - **有效上下文与压缩线公式**：若业务允许请求级 `max_tokens = M`，则客户侧可用历史输入上限必须缩紧至：
     $$L_{\text{in,eff}} = \text{max\_model\_len} - M, \qquad \text{建议 } \text{auto-compact} \le \text{max\_model\_len} - M - 8192$$
     以主流 Coding Agent 默认 `M = 16384~32768` 为例：`16 GiB`（`196,608`）留足 `32K` 输出后有效输入上限为 **`163,840` (`160K`)**；`17 GiB`（`204,800`）为 **`172,032` (`168K`)**；`18 GiB`（`215,040`）为 **`182,272` (`178K`)**。
   - **网关动态 Clamping**：建议上层业务网关/Router 依据输入长度动态收敛 `max_tokens = min(client_max_tokens, max_model_len - est_input_tokens - safety_margin)`，根除 Prefill 通过而 Decode 400 拒收的无效计算。
   - **配比口径区分**：计算 P/D 实例比应采用实际统计均值 $L_{\text{out}} \approx 500\text{–}1200$，**切勿将准入上限 `M = 32768` 代入 P/D 稳态平衡公式**。

### 3.3 KV-Cache 容量分档（`16 / 17 / 18 GiB`）参数映射

经精算并结合 `18 GiB` 实测校验，确保三档位在 `bs=128` 与 `256` 下满长单请求均严格低于 `25%` 池红线（稳定支撑 `4` 条满长请求不抢占）：

| KV Cache 档位<br>(`VLLM_CPU_KVCACHE_SPACE`) | 总块数（可用数）<br>`bs=128` / `bs=256` | 推荐 `max_model_len`<br>（预留 `32K` 输出后输入上限） | 满长单请求占池比例<br>`bs=128` / `bs=256` | `48K` 均长单请求稳态<br>（`85%` 水位并发容积） | 推荐并发限制<br>Prefill / Decode `max_num_seqs` |
|---|---|---|---|---|---|
| **`16 GiB / rank`** | `3,744 (3,743)` /<br>`3,276 (3,275)` | **`196,608`** (`192K`)<br>输入上界：`160K` | `24.5%` / `24.7%`<br>（`4` 满长并发） | `1.09 GiB` (`254` 块, `bs=128`) $\rightarrow$ **`12` 条**<br>`1.12 GiB` (`230` 块, `bs=256`) $\rightarrow$ **`12` 条** | Prefill：**`8`**<br>Decode：**`12`** |
| **`17 GiB / rank`** | `3,978 (3,977)` /<br>`3,481 (3,480)` | **`204,800`** (`200K`)<br>输入上界：`168K` | `24.0%` / `24.2%`<br>（`4` 满长并发） | `254` 块 (`bs=128`) $\rightarrow$ **`13` 条**<br>`230` 块 (`bs=256`) $\rightarrow$ **`12` 条** | Prefill：**`8`**<br>Decode：**`14`** |
| **`18 GiB / rank`** | `4,213 (4,212)` /<br>`3,686 (3,685)` | **`215,040`** (`210K`)<br>输入上界：`178K` | `23.8%` / `23.9%`<br>（`4` 满长并发；实测 `4.04x / 4.07x`） | `254` 块 (`bs=128`) $\rightarrow$ **`14` 条**<br>`230` 块 (`bs=256`) $\rightarrow$ **`13` 条** | Prefill：**`8`**<br>Decode：**`16`**<br>（紧保 48K TPOT SLO 可设 `10`） |

各档位明确环境变量设置（[`pd_tp4_agent`](../vllm_scripts/presets/mpi/dense/dflash/pd_tp4_agent/README.md) preset 已内置以下三档自动联动）：

```bash
# === 档位 A：16 GiB/rank ===
export USER_VLLM_KVCACHE_SPACE_GB=16
export USER_VLLM_MAX_MODEL_LEN=196608
export USER_VLLM_PREFILL_MAX_NUM_SEQS=8
export USER_VLLM_DECODE_MAX_NUM_SEQS=12

# === 档位 B：17 GiB/rank ===
export USER_VLLM_KVCACHE_SPACE_GB=17
export USER_VLLM_MAX_MODEL_LEN=204800
export USER_VLLM_PREFILL_MAX_NUM_SEQS=8
export USER_VLLM_DECODE_MAX_NUM_SEQS=14

# === 档位 C：18 GiB/rank（推荐生产起点） ===
export USER_VLLM_KVCACHE_SPACE_GB=18
export USER_VLLM_MAX_MODEL_LEN=215040
export USER_VLLM_PREFILL_MAX_NUM_SEQS=8
export USER_VLLM_DECODE_MAX_NUM_SEQS=16
```

---

## 4. 参数化 `TTFT / TPOT` 模型与 128-Rank P/D 配比推导

以下用 **Qwen3.8-27B TP4 + DFlash2 (`K=5`) 实测参考点** 演示整套参数化代入流程：
- **输入实测基准**：`L_test = 8192`（8K）输入时，TP4 vLLM **`TTFT = 2.4 s`**；中短上下文单请求 Decode 速度约 **`65 tok/s`**（DFlash2 `K=5`）。

### 4.1 参数化 `TTFT` 模型与代入

1. **通用公式**：设输入总长 $L_{\text{in}}$，Prefill 本地前置命中比例 $h_{\text{local}}$，Mooncake Store 远程拉取比例 $h_{\text{store}}$（总缓存率 $h = h_{\text{local}} + h_{\text{store}}$），未命中净计算量 $L_{\text{miss}} = (1 - h) \cdot L_{\text{in}}$，TP4 跨卡总 KV 传输每 token 字节 $b_{\text{kv}} = 80.00\text{ KiB/tok}$（含页对齐填充；有效 KV 载荷为 `64.00 KiB/tok`，4 组 FullAttn 逻辑传输），Store / P$\to$D 有效 RDMA 带宽 $BW_{\text{eff}}$：
   $$r_{p,0} = \frac{L_{\text{test}}}{\text{TTFT}_{\text{cold}}(L_{\text{test}})}, \qquad r_p(L_{\text{in}}) = r_{p,0} \cdot \eta_p(L_{\text{in}})$$
   $$\text{TTFT}(L_{\text{in}}) = T_{\text{route}} + T_{\text{queue},P} + \frac{h_{\text{store}} \cdot L_{\text{in}} \cdot b_{\text{kv}}}{BW_{\text{store}\to P}} + \frac{L_{\text{miss}}}{r_p(L_{\text{in}})} + \frac{L_{\text{miss}} \cdot b_{\text{kv}} + M_{\text{GDN/SW}}}{BW_{P\to D}} + T_{\text{step},D}(b, L_{\text{in}})$$

2. **基准数据代入**（设 Coding Agent 均输入 $L_{\text{in}} = 48\text{K} = 49152$，长文注意力衰减因子 $\eta_p(48\text{K}) = 0.85$，总缓存命中率 $h = 85\%$ 中 $h_{\text{local}}=0.50, h_{\text{store}}=0.35$，$BW_{\text{eff}} = 10\text{ GiB/s}$）：
   - $r_{p,0} = 8192 / 2.4 = \mathbf{3,413.3\text{ tok/s}}$（8K TP4 基线）；$r_p(48\text{K}) = 3413.3 \times 0.85 = \mathbf{2,901.3\text{ tok/s}}$。
   - $L_{\text{miss}} = 0.15 \times 49152 \approx \mathbf{7,373\text{ tokens}}$（由 `Store` 远程命中 `17,203` tokens）。
   - 延迟分解：`Store 拉取 0.131 s` + `增量 Prefill 2.541 s` + `P->D RDMA 0.091 s` + `D 首拍 0.094 s` + `路由 0.005 s` $\implies$ **`TTFT = 2.86 s`**（相比冷算 48K 的 `16.94 s` 提速 `5.9` 倍）。

### 4.2 参数化 `TPOT` 与 Decode 吞吐模型及代入

1. **通用公式**：记单请求实测中短上下文速率 $R_{D,1}$，DFlash2 `K=5` 平均每步接受 token 数 $\tau \in [1, K+1]$（参考仓内 DFlash2 `K=5` 实测区间约 `3.5 ~ 4.8`），批扩展开销系数 $\alpha_b$，KV Cache 每并发每千 token 读取增量 $\beta_{\text{KV}}$（`ms / (seq · 1K tokens)`）：
   $$\text{TPOT}(1, L_{\text{short}}) = \frac{1000}{R_{D,1}}, \qquad T_{\text{step},0} = \tau \cdot \text{TPOT}(1, L_{\text{short}})$$
   $$T_{\text{step}}(b, L) = T_{\text{step},0} \cdot \left(1 + \alpha_b (b - 1)\right) + \beta_{\text{KV}} \cdot b \cdot \frac{L}{1024}$$
   $$\text{TPOT}(b, L) = \frac{T_{\text{step}}(b, L)}{\tau}, \qquad R_D(b, L) = \frac{1000 \cdot b}{\text{TPOT}(b, L)}$$

2. **基准数据代入**（取 $R_{D,1} = 65\text{ tok/s}$，$\tau = 3.9$，$\alpha_b = 0.055$，$\beta_{\text{KV}} = 0.06\text{ ms}$，目标 $\text{TPOT SLO} \le 25\text{ ms}$，$L = 48\text{K}$）：
   - $\text{TPOT}(1, L_{\text{short}}) = 1000 / 65 = \mathbf{15.38\text{ ms/tok}} \implies T_{\text{step},0} = 3.9 \times 15.38 = \mathbf{60.0\text{ ms/step}}$。
   - 在 $b^* = 6$ 时：$T_{\text{step}}(6, 48\text{K}) = 60.0 \times (1 + 0.055 \times 5) + 0.06 \times 6 \times 48 = 76.50 + 17.28 = 93.78\text{ ms} \implies \text{TPOT} = 93.78 / 3.9 = \mathbf{24.05\text{ ms/tok}}$，单 TP4 实例稳态输出 **$R_D = 249.5\text{ tok/s}$**。

### 4.3 `128-Rank`（32 TP4 实例）P/D 稳态配比

由排队稳态平衡（取均输出 $L_{\text{out}} = 500$、突发余量 $\gamma_{\text{burst}} = 1.15$）：

$$\frac{N_P}{N_D} = \frac{R_D(b^*, L)}{r_p(L)} \cdot \frac{L_{\text{miss}}}{L_{\text{out}}} \cdot \gamma_{\text{burst}} \quad \text{且} \quad N_P + N_D = 32$$

代入上述参考点：
$$\frac{N_P}{N_D} = \frac{249.5}{2901.3} \times \frac{7373}{500} \times 1.15 \approx 1.46 \implies N_P \approx 19.0,\; N_D \approx 13.0$$

按整机 8 卡节点（2 × TP4 实例）整除对齐：

| 工况 / 口径假设 | 增量/输出<br>($L_{\text{miss}} / L_{\text{out}}$) | 实例速率<br>($r_p$ / $R_D$ TP4) | 推荐拓扑<br>(`N_P + N_D`) | 集群稳态估算<br>（增量 Prefill / 稳态 Decode） |
|---|---|---|---|---|
| **参考实测基准代入**<br>(`8K=2.4s`, `65 tok/s`, `48K, h=85%`) | `7,373 / 500` | `2,901 / 250 tok/s` | **`18P + 14D`**<br>(`72P + 56D` Ranks) | 增量 Prefill **`52.2K tok/s`**；Decode **`3,493 tok/s`**（`84` 并发工作槽） |
| **Prefill 均速接近 `2K/s`**<br>（高长程比例或 `60K/85%` 突发） | `6,000 / 500` | `2,000 / 260 tok/s` | **`20P + 12D`**<br>(`80P + 48D` Ranks) | 增量 Prefill `40K tok/s`；Decode `~3,120 tok/s` |
| **深度思考占比高**<br>（大篇幅 Thinking + 代码重构） | `6,000 / 1,200` | `2,901 / 250 tok/s` | **`12P + 20D`**<br>(`48P + 80D` Ranks) | 将卡数倾斜给长输出并发窗口 |
| **大量冷启/Subagent 扇出** | `12,000 / 500` | `2,901 / 250 tok/s` | **`22P + 10D`**<br>(`88P + 40D` Ranks) | 压制密集代码读取导致的 P95 TTFT 排队 |

---

## 5. Mooncake Store 对路由与调度策略的影响评估

### 5.1 跨实例迁移开销与容量覆盖指标

配备 RDMA `Mooncake Store` 后，KV 获取由"本地热命中或冷端全量重算"转为"**本地优先 + 高带宽远程读取**"，策略收放尺度取决于两个定量指标：

1. **RDMA 拉取与重算加速比 $\eta_{\text{net}}$**：
   $$\eta_{\text{net}} = \frac{BW_{\text{RDMA,eff}}}{r_p \cdot b_{\text{kv}}}$$
   在 `TP=4` 下，16 层有效 FullAttention KV 密度为 **`64.00 KiB/tok`**（4 组 5 层对齐逻辑传输为 **`80.00 KiB/tok`**）。以 `r_p = 2,901.3 tok/s` 计算，Prefill 算力产 KV 等效速率仅 **`0.22 GiB/s`**；若单实例分得 `10 GiB/s` RDMA 带宽，$\eta_{\text{net}} = \mathbf{45.2\times}$（降级至 `2 GiB/s` 仍有 **`9.0\times`**）。
2. **Store 容量覆盖度 $\kappa_{\text{cap}}$**：
   $$\kappa_{\text{cap}} = \frac{C_{\text{store,eff}}}{N_{\text{session,hot}} \cdot M_{\text{session}}(L_{\text{avg}})}$$
   以 `14D × b*=6 × 3` 活跃复用 = `252` 个热会话、`L_avg = 48K`（`bs=128`，单会话 TP4 逻辑页总重 `254 × 4.375 MiB × 4 = 4.34 GiB`）计算，集群精确热工作集为 **`1,093.9 GiB` (`1.07 TiB`)**（`bs=256` 为 `1,106.7 GiB`）。若全集群主机 DRAM 池提供 `>= 1.5~2.0 TiB`，可覆盖 **`1.4x ~ 1.9x`** 热会话集。

### 5.2 Prefill 与 Decode 非对称路由策略分析

- **Prefill 侧放宽亲和性，转向均衡主导**：
  - 当 $\eta_{\text{net}} \ge 15\times$ 且 $\kappa_{\text{cap}} \ge 1.5\times$ 时，路由未命中本地 Prefill 仅产生毫秒级 RDMA Store Pull 开销，却可彻底消除粘滞排队引发的 **Head-of-Line 阻塞**（长冷请求堵住同实例短增量请求）。
  - 注意：`vllm-router` 默认 `--balance-abs-threshold 64` 在 `max_num_seqs = 8~16` 下永远不会触发负载切分，必须显式下调至 **`2 ~ 6`**。
- **Decode 侧保留中度会话亲和**：
  - Decode HBM（`16~18 GiB`）极其稀缺：若将同一长会话散落派发至不同 Decode 实例，会导致同一份 `48K~180K` 历史 KV 在多台 Decode HBM 中重复驻留；保留同节点亲和则仅经 `MooncakeConnector` 传输增量块。
  - **Decode 开启 `save_decode_cache: true` 后**：在日常维持软亲和的同时，允许高负载或抢占发生时将长会话重调度并通过 Store 恢复已生成的 decode blocks。

| 集群 Store 工况 | $\eta_{\text{net}}$ / $\kappa_{\text{cap}}$ 条件 | `vllm-router` 推荐策略与阈值 | 调度效果 |
|---|---|---|---|
| **高带宽充足 DRAM 池**（推荐） | $\eta_{\text{net}} \ge 15\times$<br>$\kappa_{\text{cap}} \ge 1.5\times$ | `--prefill-policy cache_aware`<br>`--decode-policy cache_aware`<br>`--cache-threshold 0.50`<br>`--balance-abs-threshold 2`<br>`--balance-rel-threshold 1.25` | Prefill 优先均匀打散消峰；Decode 在 2 个运行请求差以内维持亲和，超差即切换 |
| **Store 容量偏紧或 RDMA 拥塞** | $5\times \le \eta_{\text{net}} < 15\times$<br>或 $0.8 \le \kappa_{\text{cap}} < 1.5$ | `--prefill-policy cache_aware`<br>`--decode-policy cache_aware`<br>`--cache-threshold 0.35`<br>`--balance-abs-threshold 4`<br>`--balance-rel-threshold 1.4` | 优先最大化本地 HBM 前缀命中，降低跨节点 RDMA 带宽竞争 |
| **接入 ZMQ KV Events（二期进阶）** | 已开启实例级 KV Events 上报 | `--prefill-policy kv_aware`<br>`--decode-policy kv_aware`<br>`--kv-aware-session-affinity-ttl-secs 1800` | 由真实物理块级全局索引替代客户端近似 Radix Tree 估算 |

---

## 6. 核心配置汇总（逐行规范格式）

### 6.1 全集群公共环境变量（Prefill & Decode 一致）

```bash
export PYTHONHASHSEED=0
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_CONFIG_PATH=/etc/mooncake/mooncake_store_rdma.json
export USER_VLLM_PD_KV_BACKEND=mooncake_multi
export USER_VLLM_PD_MOONCAKE_PROTOCOL=rdma
export USER_VLLM_PD_MOONCAKE_NUM_WORKERS=6
export USER_VLLM_KVCACHE_SPACE_GB=18
export USER_VLLM_BLOCK_SIZE=128
export USER_VLLM_MAX_MODEL_LEN=215040
export USER_VLLM_ENABLE_PROFILER=0
```

### 6.2 Prefill TP4 实例参数（参考 `18P`）

```bash
--model Qwen/Qwen3.8-27B-FP8
--tp 4
--pp 1
--data-parallel-size 1
--max-model-len 215040
--max-num-batched-tokens 6144
--max-num-seqs 8
--block-size 128
--enable-prefix-caching
--no-disable-hybrid-kv-cache-manager
--skip-mm-profiling
--language-model-only
--reasoning-parser qwen3
--speculative-config {"model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":5,"method":"dflash","prefill_context_kv_only":true}
--kv-transfer-config {"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"rdma","num_workers":6}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true,"lookup_async":true}}]}}
```

### 6.3 Decode TP4 实例参数（参考 `14D`）

```bash
--model Qwen/Qwen3.8-27B-FP8
--tp 4
--pp 1
--data-parallel-size 1
--max-model-len 215040
--max-num-batched-tokens 1024
--max-num-seqs 16
--block-size 128
--enable-prefix-caching
--no-disable-hybrid-kv-cache-manager
--skip-mm-profiling
--language-model-only
--reasoning-parser qwen3
--speculative-config {"model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":5,"method":"dflash"}
--kv-transfer-config {"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"rdma","num_workers":6}},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"load_async":true,"save_decode_cache":true}}]}}
```

> 若暂按保守 SLO 封顶（防高发并发拉高 TPOT），仅需修改 Decode 一行：`--max-num-seqs 10`（即环境变量 `USER_VLLM_DECODE_MAX_NUM_SEQS=10`）。

### 6.4 `vllm-router` 启动参数

```bash
vllm-router
  --host 0.0.0.0
  --port 30000
  --vllm-pd-disaggregation
  --kv-connector mooncake
  --prefill-policy cache_aware
  --decode-policy cache_aware
  --cache-threshold 0.50
  --balance-abs-threshold 2
  --balance-rel-threshold 1.25
  --request-timeout-secs 1800
  --health-check-timeout-secs 15
  --max-concurrent-requests 4096
  --prometheus-host 0.0.0.0
  --prometheus-port 29000
  --prefill http://<p0-host>:8100 18998
  --prefill http://<p1-host>:8104 18998
  ...
  --decode http://<d0-host>:8200
  ...
```

---

## 7. 在线服务实测校准公式与探测脚本

真实环境上线后形态各自独立（无论通过本仓库脚本、Kubernetes 或自研守护进程拉起），推荐直接对**已启动好的 OpenAI 服务接口**在线发流探测实测指标，并代入闭式方程导出最终部署配置。

### 7.1 实测采集指标与闭式校准公式

| 实测输入项 | 在线采集方式（针对已运行 HTTP 服务） | 闭式校准与配置映射公式 |
|---|---|---|
| **1. `KV_GB`、`--block-size`、$M$** | 读取节点实机空闲 KV 池与接口允许最大输出 `max_tokens = M` | $B = \left\lceil\frac{808\cdot 1024}{\text{bs}\cdot 1024}\right\rceil \cdot \text{bs}$，$N_{\text{pool}} = \lfloor \text{KV\_GB}\cdot 2^{30} / (5B\cdot 1024) \rfloor$<br>按 $4\lceil(L_{\max}+6)/B\rceil + 38 \le \lfloor(N_{\text{pool}}-1)/4\rfloor$ 锁定 `max_model_len`，并保证 `auto_compact <= max_model_len - M - 8192` |
| **2. $r_{p,0}$ 与衰减 $\eta_p$** | 向已启动 Prefill（或直连路由）发送 UUID 随机前缀流式请求（`max_tokens=1`，`8K` 与 `32K`）测冷 TTFT | $r_{p,0} = L_{\text{short}} / \text{TTFT}(L_{\text{short}})$，$\eta_p = r_p(L_{\text{long}}) / r_{p,0}$ |
| **3. $\text{TPOT}(b)$ 与 $R_D(b)$** | 以并发 `1, 4, 8, 12` 向已启动 Decode 接口打入流式生成请求（如 `4K` 输入生成 `96` tokens） | 取满足 $\text{TPOT}_{P95}(b^*) \le \text{TPOT}_{\text{SLO}}$ 的最大并发 $b^*$ 及其输出 $R_D^*$；`--max-num-seqs` 取 KV 档位保护上限与 `b* + 4` 的工程折衷 |
| **4. 业务 Trace 与 Store 带宽** | 统计真实 Coding Agent 的 $(L_{\text{avg}}, h, L_{\text{out}})$ 及 RDMA 带宽 $(C_{\text{store}}, BW_{\text{RDMA}})$ | $\rho = \frac{R_D^*}{r_p(L_{\text{avg}})}\cdot \frac{(1-h)L_{\text{avg}}}{L_{\text{out}}}\cdot \gamma$，$N_P = \text{round\_even}\!\left(32\cdot\frac{\rho}{1+\rho}\right)$；再按 $(\eta_{\text{net}}, \kappa_{\text{cap}})$ 锁定 `vllm-router` 阈值 |

### 7.2 面向已运行服务的在线探测与计算工具

仓库工具 [calc_pd_deploy_plan.py](../vllm_scripts/tools/calc_pd_deploy_plan.py) 仅依赖 Python 标准库，不调用任何本地启动器，直接对运行中的 OpenAI API 执行在线探测并输出完整配置建议：

```bash
# 模式 A：直接测试已经启动好的服务接口（可传单个 --url，或分别传 --prefill-url / --decode-url）
python vllm_scripts/tools/calc_pd_deploy_plan.py \
  --prefill-url http://<prefill-host>:<port> \
  --decode-url http://<decode-host>:<port> \
  --kv-gb 18 \
  --block-size 128 \
  --max-output-tokens 32768 \
  --tpot-slo-ms 25.0 \
  --store-cap-tib 2.0 \
  --store-bw-gibs 10.0

# 模式 B：仅基于已知标量指标离线推导（支持 --format json 接入自动化运维流水线）
python vllm_scripts/tools/calc_pd_deploy_plan.py \
  --kv-gb 18 \
  --block-size 128 \
  --max-output-tokens 32768 \
  --ttft-8k 2.4 \
  --decode-single-tps 65
```
