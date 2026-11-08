# GLM-5.2 DP>1 compile ConstraintViolationError 排查(2026-11-08)

`docs/upgrade/v0.31/13-glm52-phase0-unfused-compile-opseq.md` 阶段 0 的 DP 基线
被 dynamo `ConstraintViolationError` 阻塞:

```
You marked L['input_ids'].size()[0] / L['positions'].size()[0] as dynamic
but your code specialized it to be a constant (256)
```

发生在 `compile_or_warm_up_model → _dummy_run`(DYNAMO_TRACE_ONCE 单次 trace,
token 维标记动态、采样值 = max_num_batched_tokens = 256)。

## 已确认事实(实验矩阵)

| # | 配置 | 结果 | 日志 |
| --- | --- | --- | --- |
| 1 | TP1, dummy-7-layer-lite, 融合关, fork 已回退 | ✅ compile+端到端通过(**但该 preset 显式 `TORCH_XCPU_ENABLE_CHECK=0`**) | 20261108_080701 (HwNn58) |
| 2 | DP2×TP4×EP8, 同上基线(无任何本日修改;v5 基座未设 check 变量 → 默认 1) | ❌ 同一违规 | 上会话 broebq3is/bbyoqsow8;本日复验 20261108_083135 (IJfz51) |
| 3 | DP2×TP4×EP8 + variant B 编译期旁路(诊断) | ❌ 同一违规 | 20261108_081641 (KYbHoU) |
| 4 | DP2×TP4×EP8 + 6 处 clamp 切片改 narrow | ❌ 同一违规,栈移至 narrow 行 | 20261108_082331 (ZdDjdP) |
| 5 | DP2×TP2×EP4, 基线 | ❌ 同一违规 | 20261108_083256 (LZZ7QL) |
| 6 | (历史) DP2×TP2×EP compile, 真实 MXFP4+dflash2_5 preset(**第 24 行 `TORCH_XCPU_ENABLE_CHECK=0`**) | ✅ 2026-11-06 | logs/success/20261106_141831 |
| 7 | DP2×TP2×EP compile, 同 #6 preset, fork 已回退 | ✅ 通过(check=0,排除 fork 掩蔽假设) | 20261108_083544 (ajp1pu) |
| 8 | **TP1 + `TORCH_XCPU_ENABLE_CHECK=1`**(diag preset `phase0_unfused_compile_tp1_checkon.sh`) | ❌ **同一违规** → 证伪"DP>1 特有" | 20261108_090651 (0PzEbi) |
| 9 | TP1 + check=1 + plugin slots 截断修复 | ✅ compile+端到端通过(~3 分钟,与 #8 同 preset 同 cache,唯一差异 = 修复) | 20261108_090846 (R8xeXm) |

#6 与 #7 的环境完全一致(user_env_template 11-04 起 `VLLM_DISABLE_ATTN_REDUCE_-
SCATTER_FOR_MOE=1`,即 SP variant B;dflash2_5 preset 融合默认关;vllm 同为
83b108dd3f),唯一差异是 fork 的 deepseek_v2.py 插桩(+108 行,含
`absorbs_moe_all_gather`/`defer_moe_all_gather` 边界改写)是否在场。

## 动力学日志证据(20261108_081956, TORCH_LOGS=+dynamic)

- `create_symbol s72 = 256 for L['input_ids'].size()[0] [2, int_oo]`(创建于
  deepseek_v2.py:1586)与 `s80 = 256 for L['positions'].size()[0]`(:1416)——
  两符号创建时均为动态;
- 全日志无任何显式 specialize 事件;
- 违规在 `produce_guards_verbose` 阶段检出 `var_to_val` 含 s72/s80 ⇒
  特化是求解器**被迫**的(写不出 guard 只能用 hint 值回填等式);
- 报告的 user stack:#1(input_ids)→ attention 前向终于
  `sparse_mla.py` 的 `topk_indices_buffer[:/narrow(0,0,query.shape[0])]`;
  #2(positions)→ `deepseek_v2.py` 的 `full_num_tokens = positions.shape[0]`
  绑定行。

## 根因(已确认,2026-11-08)

**与 DP/TP 并行度无关,判别变量是 `TORCH_XCPU_ENABLE_CHECK`(torch_xcpu
`config._ENABLE_CHECKS`,默认 1)。** kit 里几乎所有 compile preset(TP1 系、
dflash2_5 等)都显式设 0 提速,恰好绕开了毒 guard;v5 基座没设 → 默认开 → 崩。
TP1 的"通过"(#1)纯属 preset 顺手关了 check;#8(TP1 + check=1)同样崩,证毕。

机理(extended-debug 日志 `20261108_085636…VSxLD0/mpi_workers.log`,事件按序):

```
eval Eq(s72, s80)  [guard added]   ← _check_fused_indexer_q: q.size(0)==positions.size(0)
                                     (sparse_indexer.py:533;guardable,无害)
eval s72 >= 256    [guard added]   ← _check_fused_indexer_k:
                                     slots.numel() <= positions.numel()
                                     (sparse_indexer.py:158)。dummy run 的
                                     slot_mappings 是持久 buffer 的静态
                                     [:, :max_num_batched_tokens] 切片
                                     (block_table.py:243)→ slots.numel()=256 常量
                                     → 下界 guard 钉在采样值上
eval s72 <= 256    [guard added]   ← sparse_mla.py:81 topk_indices_buffer[: query.shape[0]]
                                     (另一条上界: deepseek_v2.py:1582 4*s72<=1024)
set_replacement s72 = 256 (range_refined_to_singleton) VR[256,256]
                                     ← 下界==上界 → 求解器把动态维收敛成常量
                                     → var_to_val → 结尾 produce_guards 报
                                       ConstraintViolationError
```

每条 guard 单独合法;致命的是**常量下界(check 产生)+ 常量上界(max-token
尺寸 buffer 的符号切片产生)夹逼成单点**。`fused_indexer_qk_rope_quant_cache`
(QK 融合路径)内部同样调用 `_check_fused_indexer_k`(sparse_indexer.py:803),
故 TP1 也中招。

### 修复(plugin,2026-11-08)

`vllm-xcpu-plugin/vllm_xcpu_plugin/attention/sparse_attn_indexer.py`:
`_indexer_k_cache` / `_indexer_qk_cache` 在取到 slots 后截断
`slots = slots[: k.shape[0]]`:

- compile 侧:slots.numel() 变成动态符号 s72,check 不等式退化为
  `s72 <= s72` 恒真,常量下界消失(s72 保持 [2,256] 动态范围,上界 guard
  单独存在是合法且正确的——运行时本就受 max_num_batched_tokens 约束);
- eager/运行时侧:真实 step 的 slots 同为持久 buffer 的
  `[:, :num_tokens_padded]` 切片(block_table.py:209/243),尾部为 PAD_SLOT_ID
  填充,kernel 按 K 行迭代并跳过负 slot → 截断到 k 行数语义等价;
  slots 本来就短于 k 行时切片为 no-op。

(备选但未采用:把 check 里的 numel 子句从 torch_xcpu 侧删掉——治标且弱化
op 契约;以及继续靠 preset 关 check——掩盖问题,新环境默认即踩。)

### 历史假设清算

1. ~~clamp 切片 `x[:n]` 产生 `min(size, n)`~~ —— narrow 无效(#4),实因上界
   guard 与切片写法无关;
2. ~~SP `min()/max()` 布尔求值~~ —— #5 证伪;且 SEQUENCE_PARALLEL_MOE_BACKENDS
   不含 mpi_alltoallv_v5,variant A/B 均为死代码;
3. ~~fork hook 掩蔽~~ —— #7 通过,证伪;
4. ~~dflash speculator 调整 dummy token 数~~ —— dflash preset 成功的真因是其
   `TORCH_XCPU_ENABLE_CHECK=0`(preset 第 24 行),与 token 数无关。

## 工作区状态(2026-11-08 根因修复后)

- plugin:`sparse_attn_indexer.py` 的 slots 截断修复已提交(`cecaec8`);
  `fake_triton/vllm_kernels.py` 的 PackSeq/UnpackSeq **拒绝式 adapter** +
  测试已提交(`adf0c6c`):不实现语义,报错并提示
  `--kernel-config {"enable_jit_warmup":false}`(见 13 §2);
- stash(均已被证对本案无效):plugin `sparse_mla narrows + hash`、
  vllm `narrow+num_valid fixes`(narrow 语义更精确,可择要重做);
- vllm:仅 envs.py 的 QNORM_QB_UK cache-key 注册,基线干净;
- 待办:dp2tp4ep8(check 默认开)修复后完整验证 + 提取 DP 侧 Phase-0
  算子序列(SP MoE 尾部 all-gather 位置 = Phase 3 匹配窗口)。
