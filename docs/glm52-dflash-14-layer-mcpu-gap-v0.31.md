# GLM-5.2 + DFlash2 "每 14 层一次 mcpu stream 间隙"分析

2026-11-07/08，x86 调试平台（DP2TP2EP4 复现）定位真实硬件 DP16-TP4-EP64 现场问题。
分析工具与复现预设已落在 `vllm_scripts/diag/mcpu_trace/`（见文末）。

## 现场现象（真实 NPU 硬件）

GLM5.2-MXFP4，DP16-TP4-EP64，`USER_VLLM_MAX_NUM_BATCHED_TOKENS=6144`，fused EP MoE v5，
`VLLM_XCPU_MOE_MAX_RECV_TOKENS=128`，开启 profiler：

1. mcpu stream 上大约每 14 层内的间隙很小（stream 供数充足的状态），但**每 14 层之间
   出现一次较大间隙**（真实硬件上约 300 µs）；
2. **第一块例外**——只有第一个 14 层块没有这个间隙；
3. 间隙**总是紧跟在 all-gather 之后**；
4. `USER_VLLM_MAX_NUM_BATCHED_TOKENS=512`（MoE chunking 不触发）时间隙**依然存在**，
   即与 MoE 分块/batched tokens 无关；
5. 另一个独立现象：fused ep moe 的 **host 派发时间变长**。

## 结论

**每 14 层一次的间隙 = DFlash2（EAGLE3 类）speculative decode 的 6 个
`aux_hidden_state` 捕获层。**与 MoE、GC、allocator 均无关。

启动日志（11-06 DP2TP2EP4 真权重 compile 运行，4 个 worker 均打印）：

```
Using Eagle3 auxiliary layers from config: (6, 20, 34, 48, 62, 76)
```

draft 模型 `incoai--GLM-5.3-DFlash2`：`num_hidden_layers=6`，`num_target_layers=78`。
6 个 draft 层各取一个目标层 hidden state，均匀铺在 78 层上 → 间距恰为 14。
`deepseek_v2.py`（vllm `model_executor/models/deepseek_v2.py` ~1625）：

```python
if idx in self.aux_hidden_state_layers:
    aux_hidden_state = hidden_states if residual is None else hidden_states + residual
    if hidden_states_are_sequence_parallel:
        aux_hidden_state = tensor_model_parallel_all_gather(aux_hidden_state, 0)
        aux_hidden_state = aux_hidden_state[: positions.shape[0]]
    aux_hidden_states.append(aux_hidden_state)
hidden_states, residual = layer(...)
```

这一次 `hidden_states + residual` 正好位于 MoE 的 SP all-gather 之后、下一层
`fused_add_rms_norm` 之前——与"间隙总在 all-gather 后面"完全一致。

与现场观察逐条对应：

| 观察 | 解释 |
| --- | --- |
| 每 14 层一次 | 6 个 aux 层 `(6,20,34,48,62,76)`，间距 14 |
| 第一块例外 | 第一个 aux 层是第 6 层，首段只有 3 个 MoE 层 |
| 总在 all-gather 后 | aux 捕获的 add+all-gather+slice 夹在 SP all-gather 与下层 norm 之间 |
| 512 tokens 仍有 | 与 MoE chunking、batched tokens 完全无关 |

## 证据链（x86 DP2TP2EP4 复现）

- **间隙位置**：compile 模式真权重 trace 中，all-gather 之后 ≥10 µs 的大间隙精确落在
  MoE 层调用序 `mod 75 ∈ {3,17,31,45,59,73}`，8 个 step **8/8 全中**，其余 71 个层位置
  一次都没有。（78 层 = 3 dense + 75 MoE；第 6/20/34/48/62/76 层对应 MoE 序号
  3/17/31/45/59/73。）
- **阴性对照**：不带 spec decode 的同配置运行（runA，6144 tokens，v5）完全没有
  14 层周期间隙。
- **host 侧签名**：只有这 6 个位置，host 在 all-gather 与下一个 `fused_add_rms_norm`
  之间多出 `aten::result_type` + `aten::empty([T, 6144] bf16)`——是整张 78 层图中
  **唯一的非内存池分配**（其余 1691 个 buffer 全部 `aoti_torch__alloc_from_pool`，
  只有 2 次 pool 创建）。AOTI 生成码注释直接写着
  `// Topologically Sorted Source Nodes: [aux_hidden_state_1], Original ATen: [aten.slice, aten.add]`。
- **间隙不是空闲**：host 领先 device 约一整个 step（实测 lag 925–1923 ms，队列中始终
  有整步任务），因此间隙期间 stream 不可能真的无事可做；而这个 add 的 device kernel
  **没有出现在 mcpu 时间线里**（无 `mcpu::add_4d_lastdim` 事件，`mcpu::aten::add`
  每步仅 3 次）。结论：**间隙是未被 profiler 计时的 stream 工作**，不是 device 空闲。

### 为什么这个 add 未被计时、现场为什么 ~300 µs（推断，待 NPU 侧确认）

torch_xcpu 的 add 快路径要求 128 B 对齐（`torch_xcpu_impl/include/aten/add.h`
`kAlignmentBytes = 128`），不满足则 `detail::call_host` 回落到 torch_mcpu 通用 add
路径。Inductor 内存池切出的 buffer 指针不保证该对齐，因此 aux 的 add 大概率走了
回落路径：一条**不经过 `MCPU_LAUNCH_TIMED_KERNEL` 的 host 实现**，于是 profiler
看不见它；现场弱 host CPU 上执行 ~300 µs 与此吻合。（x86 上 host 强，同一位置的
间隙只有 10–170 µs。）

注意 profiler 本身会放大 host 段（with_stack 约使 Python/host 段翻倍，见
`docs/host-overhead-analysis-v0.31.md`），现场关掉 profiler 时间隙应小于 300 µs，
但 add 的额外开销是真实存在的。

## 修复建议（按收益排序）

1. **现场先确认**：`grep "Using Eagle3 auxiliary layers"` 看是否为 6 层、间隔 14；
   再做一次不带 spec decode 的 A/B（间隙应完全消失）。
2. **消掉重复计算**：aux 捕获的 `hidden_states + residual` 与紧接的下一层
   `fused_add_rms_norm` 内部是**同一个加法**。让 aux 层路径复用该和（例如
   `fused_add_rms_norm` 增加可选输出，把 pre-norm 和写入常驻 buffer；或提供
   `add_out` 变体 + 预分配 buffer），可同时去掉 6 次/step 的额外 add 和 6 次新鲜
   设备内存分配（全图唯一的非池分配）。
3. **让这个 add 可见**：把回落路径纳入 `MCPU_LAUNCH_TIMED_KERNEL`，并让 Inductor
   内存池保证 128 B 对齐使其走 torch_xcpu 快路径——否则 profiler 永远把这段工作
   显示成"间隙"，误导排查（本次现场报告即被它误导）。
4. **减少 aux 层数**：若 draft 质量允许，通过 `eagle_aux_hidden_state_layer_ids`
   减少 aux 层，间隙次数按比例下降。

第 2、3 项分别落在 vllm/vllm-xcpu-plugin 与 torch_xcpu/torch_mcpu。

## 修复实施与验证（2026-11-08，建议 2 + 3 已落地）

### 改动清单

- **建议 2（消除重复 add）** —— `vllm/vllm/model_executor/models/deepseek_v2.py`：
  - `DeepseekV2DecoderLayer.forward` 新增 `capture_input_sum` 形参。aux 层调用时
    返回三元组 `(hidden_states, residual, input_sum)`，其中 `input_sum` 直接取
    `input_layernorm` 的 residual 输出（即 pre-norm 和），不再在 model loop 里
    重算 `hidden_states + residual`；末层（76）同理复用 final norm 的和。
  - 模型循环中 aux 分支的显式 add 整体删除。EAGLE3 语义不变：aux 捕获的就是
    pre-norm 和（与原实现数值一致，均经同一 `fused_add_rms_norm` 计算）。
  - **clobber 消除**：eager 下 `fused_add_rms_norm` 默认走 inplace 实现（和写回
    调用方 residual buffer），同层随后的 `post_attention_layernorm` 会原地覆盖
    该 buffer。为此在 `vllm/vllm/ir/op.py` 新增 `IrOp.functional()` 上下文——
    scope 内 dispatch 跳过所有 inplace 实现并保底 `native`，aux 层的两个 norm
    在该 scope 内调用，捕获的和落在新鲜 buffer。compile 优先级
    （`["torch_xcpu","native"]`）本身 functional-first，scope 为 no-op。
  - 复用已有的 `torch_xcpu.ops.fused_add_rms_norm_out`（plugin 注册的非 inplace
    实现），无新 kernel。
  - 单元验证：`vllm_scripts/diag/mcpu_trace/test_aux_sum_capture.py`（eager/compile
    两种优先级 + 负对照：无 scope 时 inplace 实现确实会 clobber 捕获值）。
- **建议 3a（host 回落可见）** —— `torch_xcpu/torch_xcpu/csrc/aten/aten_fallback.h`：
  `call_host` 用 `at::RecordFunction` 记录
  `xcpu::host_fallback::<op_name>` host scope，profiler 打开后回落不再隐身
  （shape 等细节由回落内部嵌套的 aten:: cpu_op 提供）。
- **建议 3b（池对齐提升）** —— `pytorch/torch/_inductor/utils.py`：
  `ALIGN_BYTES` 64 → 256（覆盖 `kAlignmentBytes=128` 的要求，验证运行在 128 下
  完成，256 为其超集），池视图不再因基址对齐
  静默回落 host 路径。**注意**：改动后需删除 `vllm_scripts/vllm_cache` 重编译。
- 配套：`vllm-xcpu-plugin/vllm_xcpu_plugin/upstream_compatibility.py` 中
  deepseek_v2 的 SHA256 已随本次改动更新（仅 capture 管道，indexer/attention
  未动）。

### 验证结果（GLM-5.2-MXFP4 dp2_tp2_ep dflash2_5 alltoallv_v5，DP2TP2EP4）

- eager 与 compile e2e 均通过，输出语义正常（无精度回归）。
- compile trace（11 步 × 75 MoE 层，rank0）：
  - aux 位（MoE# mod 75 ∈ {3,17,31,45,59,73}）上 ≥100 µs 间隙 **0/11 步**
    （修复前 8/8 步）；设备利用率 **99.75%**（13.66 s 窗口 idle 仅 34.4 ms）。
  - `find_period.py`：周期 ≈14 的事件 **无**；仅剩 75（每步）与 4（draft 内部）。
  - `aten::empty` 含 6144 维度 **0 个**（修复前每次 aux 一个）；aux 位
    `result_type` **0 个**。
  - 剩余 24 个 ≥100 µs 间隙全部在步边界（MoE# 为 75 的倍数），归因为
    model_runner 状态管理（`vllm_apply_write`、host↔device memcpy），与 aux 无关。
- 建议 3a 生效：trace 中可见 `xcpu::host_fallback::aten::{fill_.Scalar, add.Tensor,
  topk.values}`，聚在步边界（MoE#74 等，draft/cat 区域的小算子），不再伪装成间隙。

## 附带发现："fused ep moe host 派发时间变长"

- **`VLLM_XCPU_MOE_MAX_RECV_TOKENS` 对 fused EP v5/v6 完全无效**，只有 v4 的
  mpi_alltoallv 读取它。v5/v6 的分块配额是
  `VLLM_XCPU_MOE_EP_MAX_RECV_TOKENS_PER_RANK`（默认 1024，单位为 post-SP token
  行数，0 = 不分块）。`docs/release_notes/20261019、20261026 更新说明.md` 原文推荐
  的旧变量已被勘误（本次修正）。
- v5 的 rounds = `ceil(post_SP_max_tokens / quota)`，host **无条件提交全部 rounds**
  （`moe_ep_chunk_round` 每层 3.00 次，多余 round 只在 device 上被 skip）。
  x86 eager 实测 host self time：1 round 27.4 µs vs 3 rounds 36.7 µs
  （≈25 µs 固定 + ≈4.7 µs/round）。现场 TP4+SP：6144/4 = 1536，quota 1024 → 2 rounds，
  不是主要矛盾。
- **EP64 内存代价**：max_tokens=1536 / quota=1024 → 每层 `torch.empty` workspace
  **4409 MB**（分配后即释放，但会反复冲击 allocator），另有 dispatch/combine MPI
  窗口各 ~770 MB；quota 还受 MPI int32 displacement 限制（EP64 下 chunk ≲ 2715 行，
  6144 不分块会被直接拒绝）。
- **已排除**：GC 与 allocator 慢路径。精确 profiled 窗口（40.4 s，2625 个 MoE 层）
  内 worker 仅 1 次 gen0 GC（~86 µs 量级），`num_alloc_retries` 全程为 0；启动期的
  GC 密集段全部落在窗口之外。（此前一度怀疑 gen1 GC 每 14 层一次，经窗口对齐后
  否定——`gc_per_layer.py` 就是为这次否定写的。）
- 微基准排除"设备分配造成 stream 间隙"：深队列中途 `torch.empty` 1 MiB–2 GiB，
  间隙中位数 0.11–0.47 µs，无影响（`micro_alloc_gap.py`）。
- 可能的 host 抖动因素：mcpu 默认 stream worker 硬编码 `IdlePolicy::Busy`
  （永久自旋、默认不绑核），每个 rank 常驻吃掉一个 host 核；弱 host 上建议
  `TORCH_MCPU_STREAM_WORKER_CORE` 绑到空闲核。

## 复现与分析资产

预设（`cd vllm_scripts && ./run_vllm_test.sh -e diag/mcpu_trace/presets/<x>.sh`）：

- `runA_glm52_mxfp4_dp2tp2ep4_v5_6144.sh` — 现场形状（6144 tokens，无 spec decode），
  阴性对照；
- `runB_glm52_mxfp4_dp2tp2ep4_v5_512.sh` — 512 tokens 单 round；
- `runD_glm52_dummy78_dp2tp2ep4_v5_512.sh` — 78 层 dummy-lite
  （`models/ZhipuAI/GLM-5.2-MXFP4-W4A16-dummy-78-layer-lite`，mlp 3 dense + 75 sparse，
  indexer 21 full + 57 shared）。

复现"每 14 层间隙"本体用带 DFlash2 的预设，如
`presets/mpi/moe/dflash/GLM-5.2-FP8_dp1_tp4_ep4_compile_sparse_attn_fp8_kvcache_dflash2_5.sh`。

x86 调试机上的历史产物（不在仓库内）：

- `vllm_scripts/vllm_profile.dp2.tp2.ep4.v5.bk/` — 11-06 DP2TP2EP4 compile+DFlash2
  trace，14 层周期证据来源；
- `vllm_scripts/vllm_profile.runA.dp2tp2ep4.v5.tok6144/`、`vllm_profile.runB...tok512/`、
  `vllm_profile.runD.dummy78...` — 阴性对照/单 round/dummy 运行；
- 运行日志 `vllm_scripts/logs/success/20261107_09*`。

分析方法与工具说明：`vllm_scripts/diag/mcpu_trace/README.md`。
