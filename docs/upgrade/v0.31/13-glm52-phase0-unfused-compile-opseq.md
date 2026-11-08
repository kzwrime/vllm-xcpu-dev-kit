# GLM-5.2 Phase-0 未融合 compile 基线与算子序列(vLLM 0.31)

2026-11-08。MLA 融合迁移到 torch.compile pattern pass 的阶段 0 基线:
确认未融合 MLA 前向在 MCPU compile(DYNAMO_TRACE_ONCE + Inductor, cpp_wrapper)下
可完整 trace,并从 AOT 产物提取图内算子序列作为后续 pattern 的匹配目标快照。
迁移动机与整体计划见 `docs/mega-kernel-proposal.md` 与
`vllm_scripts/diag/mk_pass/`。

## 环境

- 模型:`models/ZhipuAI/GLM-5.2-FP8-dummy-7-layer-lite`(`--load-format dummy`,
  7 层,indexer 层 0/1/2/6),FP8 KV cache。
- 仓库基线:vLLM `83b108dd3f4`(**已回退 deepseek_v2.py 的 fork 侧融合插桩**,
  envs.py 仅保留 `VLLM_XCPU_FUSED_MLA_QNORM_QB_UK` 的 cache-key 注册);插件
  `2b97937` + 本次新增的 pack/unpack fake-triton adapter。
- preset:`vllm_scripts/presets/serial/GLM-5.2-FP8-dummy-7-layer-lite_phase0_unfused_compile_tp1.sh`
  (DP1 TP1 单进程,MLA 融合全关,独立 cache `vllm_scripts/cache_tp1/`)。
- 单次运行约 2-4 分钟(compile 产物缓存命中后更快)。

## 过程中发现并修复的两个阻塞点

1. **dynamo ConstraintViolationError("marked dynamic but specialized to 256")**:
   栈指向 fork 版 deepseek_v2.py:1451(`full_num_tokens = positions.shape[0]`
   一带)与 sparse_mla.py:81。回退 vllm fork 的 `xcpu_fused_layer_front` /
   `absorbs_moe_all_gather` / `defer_moe_all_gather` 插桩后,TP1 不再复现,
   **但 DP>1 仍复现**(更正 2026-11-08 早前的乐观结论)。实验矩阵见
   `docs/upgrade/v0.31/13a-glm52-dp2-compile-constraint-debug.md`。
2. **JIT warmup 触发未适配的 `PackSeqTritonKernel`/`UnpackSeqTritonKernel`**:
   `UnknownKernelError: vllm.v1.attention.ops.common.PackSeqTritonKernel.kernel`。
   该两 kernel 由上游 `1713b9866a`("[2/N][warmup][DSv4] Migrate sequence and
   DCP kernels")引入;唯一活路径是 `SparseAttnIndexer.__init__` 在
   `enable_jit_warmup`(上游默认 True)时的 `register_warmup` —— 真实发射点
   (`pack_seq_triton`/`unpack_seq_triton`,非均匀 decode len 路径)在 XCPU 上
   被 OOT indexer 完全替换,**永远不可达**。处理(2026-11-08 按用户决定):
   - fake-triton adapter **不实现语义,直接报错**并给出解法:
     `--kernel-config {"enable_jit_warmup":false}`(GLM-5.2/DSv4 在 XCPU
     走自有适配路径,不需要 triton JIT warmup;不改全局默认是因为未来其他
     架构模型可能需要 warmup)。保留注册项 → 上游改动仍被哈希锁强制审查。
   - kit 内 GLM-5.2 compile preset 本就几乎都设了该 flag;phase-0 TP1 preset
     已补(`GLM-5.2-FP8-dummy-7-layer-lite_phase0_unfused_compile_tp1.sh`)。
     注意该 warmup 只在 compile 路径(`compile_or_warm_up_model` →
     `kernel_warmup`)执行,eager 运行不受影响。
   - 测试:`tests/fake_triton/test_pack_seq.py`(真实 tensor 与 warmup 替身
     两种 dispatch 均须报错且含 flag 提示)。fake_triton 套件在 HEAD 上即有
     13 个存量失败(环境漂移),与本次无关。

## 图内算子序列(TP1,AOT 产物)

产物:`cache_tp1/vllm_cache/torch_compile_cache/torch_aot_compile/a0f74a67…/`
`rank_0_0/model`(wrapper `.py` 在同目录 `inductor_cache/up/`)。
全图调用计数(mcpu 命名空间):

| op | 次数 | 说明 |
| --- | --- | --- |
| fp8_scaled_mm_intel_amx_out | 39 | qkva/q_b/o_proj/router/shared/indexer.wq_b 等 FP8 线性 |
| rms_norm_bf16 | 15 | q_a/kv_a layernorm(每层 2)+ 首层输入 norm + 最终 norm |
| fused_add_rms_norm_out_bf16 | 14 | 残差加 + norm(层界) |
| einsum_mhk_hkn_to_mhn | 14 | W_UK / W_UV(每层各 1) |
| mm_out | 8 | router(BF16)+ indexer.wk_weights_proj 等 |
| sparse_mla_fp8 | 7 | 稀疏注意力核心 |
| rotary_embedding_bf16 | 7 | MLA q RoPE |
| reshape_and_cache_mla_fp8_ds | 7 | KV cache 写 |
| cat | 7 | 注意力输入拼接 |
| fused_indexer_qk_rope_quant_cache_out | 4 | indexer 层 |
| sparse_attn_indexer_from_metadata | 4 | indexer 层 topk |
| grouped_topk / moe_scaled_add_out / fused_moe_compute_ | 4+4+4 | 路由 MoE(4 层) |
| silu_and_mul_bf16 | 7 | shared expert 激活 |
| index_select_dim0_out | 1 | embedding gather |

**indexer 层(0/1/2)前段,按执行顺序** —— Phase 1/2/4 pattern 的匹配窗口:

```
fused_add_rms_norm_out_bf16          # ← Phase 2 起点:input_layernorm+残差
fp8_scaled_mm_intel_amx_out          # ← Phase 2:fused_qkv_a_proj
rms_norm_bf16                        # ← Phase 1 起点:q_a_layernorm
rms_norm_bf16                        #   kv_a_layernorm
fp8_scaled_mm_intel_amx_out          # ← Phase 1:q_b_proj
fp8_scaled_mm_intel_amx_out          # ← Phase 4:indexer.wq_b
mm_out                               # ← Phase 4:indexer.wk_weights_proj
fused_indexer_qk_rope_quant_cache_out# ← Phase 4 终点
sparse_attn_indexer_from_metadata    #   indexer topk
rotary_embedding_bf16                # ← Phase 1 终点区:q RoPE
reshape_and_cache_mla_fp8_ds         #   KV cache 写
einsum_mhk_hkn_to_mhn                # ← Phase 1:W_UK
cat / sparse_mla_fp8 / einsum(W_UV) / fp8_mm(o_proj) …
```

注意:indexer 层中 indexer 的 4 个算子(wq_b、wk_weights、qk_rope_quant_cache、
topk)插在 q_b_proj 与 rotary/W_UK 之间 —— python 路径 fused 层需要手工重排,
这正是 pass 方案的优势:pattern 按 def-use 匹配,不受发射顺序影响。
普通层(3-5)无 indexer 4 算子,q_b_proj 后直接 rotary。

## 图内算子序列(DP2×TP4×EP8,2026-11-08 修复后提取)

产物:`cache/vllm_cache/torch_compile_cache/torch_aot_compile/1bcbcac1…/`
`rank_0_0`(TP4 每 rank 一份图,EP 同图)。DP2 compile **完整通过**
(20261108_104750,slots 截断修复 + check 全开,~25 分钟含全新编译)。
相对 TP1 的图差异:

| op | 次数 | 说明 |
| --- | --- | --- |
| all_reduce_out_wrapper | 11 | TP4 集合通信进图(TP1 无) |
| all_gather_into_tensor_out_v2 | 4 | **SP MoE 尾部 all-gather,每 MoE 层 1 次(Phase 3 锚点)** |
| fused_add_rms_norm_chunk_out_bf16 | 4 | chunk 布局残差+norm(接 all-gather 之后;TP1 的 14 个 fused_add_rms_norm 中 4 个变为 chunk 变体) |
| fused_ep_moe_v5_IntelAmxFp8MoeGroupedGemm | 4 | EP MoE v5 单算子(TP1 的 grouped_topk+moe_compute 序列被吸收;shared expert silu 也并入,silu 7→3) |
| get_masked_input_and_mask / fused_embedding_masked_fill_out | 1+1 | DP 入口 padding 掩码 + embedding 掩码填充 |
| fp8_scaled_mm_intel_amx_out | 31 | TP1 为 39(MoE 线性移入 fused_ep_moe) |

**MoE 层尾部图内顺序(每个 MoE 层,Phase 3 匹配窗口):**

```
…router/router_norm/shared expert…
fused_ep_moe_v5_IntelAmxFp8MoeGroupedGemm   # EP MoE,输出为 chunk 布局
all_gather_into_tensor_out_v2               # ← Phase 3 目标:吸收进下一算子
fused_add_rms_norm_chunk_out_bf16           # 下一层残差+norm,动态 chunk 宽
                                            #   3*ceil(s72/TP)(已是部分融合)
```

即"SP 尾部 all-gather 吸收进下层 input_layernorm"已有现成的两算子形态
(gather + chunk-norm 分离);Phase 3 的 pattern 是把二者合一,或验证现有
chunk 算子已达到吸收目标后跳过。DP 入口的
`get_masked_input_and_mask`/`fused_embedding_masked_fill` 与此窗口无关。

注意:diag preset 的"隔离 cache"曾因基座重置 SCRIPT_DIR 实际写入主 cache
(已修,`_DIAG_CACHE_ROOT` 在 source 前捕获);本节产物取自主 cache
`1bcbcac1…`。

## 待办(后续阶段)

- Phase 1:`fused_mla_qnorm_qb_uk_rope_cache` 的 pattern-matcher pass,替换
  插件 python 路径 `_forward_qnorm_qb_uk`。
- Phase 2:`fused_mla_add_norm_qkva_qnorm_qb_uk_rope_cache` 的 pattern(算子
  在 torch_xcpu 中保留,A0/A1/A2 版本开关转为 op 参数)。
- Phase 3:下一层 input_layernorm 吸收 SP MoE 尾部 all-gather 的跨层 pattern。
- Phase 4:indexer 前段(wq_b + wk_weights_proj + qk_rope_quant_cache)融合,
  新 torch_xcpu 算子按其 CLAUDE.md 的 A0/A1/B0 规范实现。
