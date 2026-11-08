# 融合 kernel 接入情况检查（vllm-xcpu-plugin ↔ torch_xcpu，dev/0.31）

日期：2026-11-01。基于工作树现状（含未提交改动）与 `.venv` 中已安装的 torch_xcpu（构建于 2026-10-31 07:34）。

## 1. 接入通路

torch_xcpu 共注册 208 个 `torch_xcpu::*` 分发入口（64 个 `TORCH_LIBRARY_FRAGMENT`，另有 7 处 aten 覆盖）。plugin / vllm fork 通过五条通路使用它们：

| 通路 | 位置 | 覆盖的融合算子 |
| --- | --- | --- |
| `register_oot` 层替换 | `plugin/custom_ops.py`、`layers/*.py` | `silu_and_mul`、`fused_sigmoid_mul`、`fused_embedding_masked_fill` + `get_masked_input_and_mask`、`layer_norm`、`gemma_rms_norm`、`rms_norm_gated`、`rotary_embedding`/`mrope`、GDN（`chunk_gated_delta_rule_separated(_custom_v2)`、`qwen_gdn_attention_core_state`）、FP8/MXFP4 linear、MoE 全家（`fused_moe_compute`、`moe_prepare_fused` v1–v6、`moe_finalize` v3/v5/v6、`fused_ep_moe` v5/v6、`fused_af` v7/v8、`moe_scaled_add`） |
| `vllm.ir.ops.register_impl`（compile 融合 pass 落地） | `plugin/ir_kernels.py`、`platform.py` fused_priority | `rms_norm`、`fused_add_rms_norm`（inplace / out 两版）；`_C::rms_norm` 另由 `torch_xcpu._C.register_vllm_rms_norm()` 直接注册 PrivateUse1 C++ 内核 |
| `upstream_compatibility` 源码指纹 + monkeypatch | `plugin/upstream_compatibility.py` + 各 `*_patch.py` | `unified_attention`、`reshape_and_cache(_grouped)`、`causal_conv1d_fn/update`、`fused_sigmoid_gating_delta_rule_update`、`fused_recurrent_gated_delta_rule_packed_decode`、`fused_post_conv_prep`、`gumbel_sample`、`vllm_temperature_kernel`、`apply_top_k_top_p`、`grouped_topk`、`topk_softmax`、`prepare_dflash_inputs`、`dflash2_grouped_conv`、`einsum_blpr_blcr_to_blpc`、`sparse_attn_indexer(_from_metadata)` |
| fake_triton 内核拦截 | `plugin/fake_triton/vllm_kernels.py` | 约 60 个 triton kernel 被 Python 替身接管，其中 12 个落到 xcpu 原生算子：`qwen3_vl_bilinear_pos_embed_out`、`per_token_group_quant_fp8_out`、`zero_kv_blocks`、`build_indexer_prefill_chunk_metadata`、`logit_bias`、`mamba_aligned_state_indices`、`vllm_rejection_compute_local_logits_stats`、`vllm_rejection_resample`、`dflash2_selector_walk`、`gather_draft_sampled`、`bad_words`、`topk_log_softmax`；其余仍是纯 torch 复现 |
| vllm fork 直接改造 | `vllm/model_executor/models/utils.py`、`layers/utils.py`、sampler/speculator/gpu_worker | `try_xcpu_fused_ffn`（Qwen2/Qwen3 dense MLP，`VLLM_XCPU_USE_FUSED_FFN` 默认 1）、`fused_add_rms_norm_chunk_out`（SP 切片）、`torch_xcpu.ops.linear`、`ops.copy` |

GLM-5.2 稀疏 MLA 路径（`plugin/attention/`）：`einsum_mhk_hkn_to_mhn`、`fused_mla_rope_qproj_cat_cache`、`SparseMlaFp8`（`sparse_mla_fp8` + `sparse_mla_fp8_vproj`）、`fused_indexer_k_norm_rope_cache` / `fused_indexer_q_rope_quant` / `fused_indexer_qk_rope_quant_cache`、`indexer_k_quant_and_cache`、`get_paged_mqa_logits_metadata_out`；`cp_gather_indexer_k_quant_cache`、`fp8_fp4_(paged_)mqa_logits_out`、`top_k_per_row_*` 由 `ops_defs/sparse_indexer.py` 封装间接到达。

## 2. Mega-kernel（fused_mla_qnorm_qb_uk）接入状态

- 框架（`torch_xcpu_impl/include/mk/`，hal / Gemm / RmsNorm 黑盒）已提交：`d897479`（2026-10-30）。
- **算子注册与 plugin 接线均未提交**，只在工作树里：
  - torch_xcpu 未跟踪：`csrc/fused_mla_qnorm_qb_uk.cpp`、`csrc/fused_mla_rope_cat_cache.cpp`、`ops_defs/mla_qnorm_qb_uk.py`、`torch_xcpu_impl/include/fused_mla_qnorm_qb_uk.h`、`torch_xcpu_impl/src/deepseek_v32/fused_mla_qnorm_qb_uk.cpp`、`test/deepseek_v32/test_fused_mla_qnorm_qb_uk.py`；修改：`ops.py`、`ops_defs/embedding.py`、`operator_version.{h,cpp}`、`mk/hal.h`、`mk/op.h`、v1/v2 段程序与测试、`xthdS/launch.*`。
  - plugin 修改：`attention/mla.py`（+116 行：`FusedMlaQNormQbUk` 调用、`_forward_qnorm_qb_uk`、`forward_with_rope_qnope`）、`layers/fp8_linear.py`（+6 行：`_xcpu_keep_raw_fp8` 保留行主序 FP8 权重）。
- 已安装的 `.venv` torch_xcpu 包含这些新算子；`test_fused_mla_qnorm_qb_uk.py` 10 个用例通过。
- 行为：`VLLM_XCPU_FUSED_MLA_QNORM_QB_UK` 默认开；只在 eager 下走（`torch.compiler.is_compiling()` 时跳过）；仅覆盖 TP4 FP8 GLM-5.2 形状且 T ≤ 16，其余在封装内退回原算子序列。
- 文档不一致：`docs/mega-kernel/README.md` 的"状态"仍写"还没有接入 torch_xcpu 的算子注册与 plugin"，而 `deepseek_v32/op-fused-mla-qnorm-qb-uk.md` §12 已写"已接入"。README 该段未随本次改动更新。
- design-guide §12 第 10 步（端到端对拍 + profiler 对比）无留痕：`vllm_scripts/` 的 GLM-5.2 preset 与 serve_test 均未提及该开关。

## 3. torch_xcpu 已注册但 plugin / vllm 无调用方的融合算子

| 算子 | 现状 | 判断 |
| --- | --- | --- |
| `fused_ffn` | 已接线（默认开），但 impl 层无融合实现：A 族只有 `fused_ffn_aten_fallback`（ATen addmm → silu·mul → mm 串行），B 族占位 | 接口已就位、性能收益为零，等矩阵乘团队填 B 族 |
| `fused_dot_sigmoid_mul_add`、`fused_sigmoid_mul_add` | 仅单测 | 无消费者 |
| `fused_gdn_gating` | vllm GDN 路径走 `_C.fused_gdn_gating_cpu`，plugin GDN patch 用的是 `fused_sigmoid_gating_delta_rule_update` | 无消费者 |
| `fused_mla_rope_cache` | 已被 `fused_mla_rope_qproj_cat_cache` / `fused_mla_rope_cat_cache` 取代 | 遗留 |
| `deepseek_v4_*`（7 个 mHC 算子） | 有单测，无 plugin / vllm 调用 | 等 DeepSeek V4.1 模型接入 |
| `unified_attention_mla_fp8_ds`、`reshape_and_cache_mla(_fp8_ds)` | `ops.py` 有封装，plugin 稀疏 MLA 用的是 `reshape_and_cache` + `unified_attention` + `SparseMlaFp8` | dense MLA 遗留，可清理 |
| `moe_finalize_v2`、`moe_finalize_reduce`、`moe_prepare_fused_v2_cleanup` | v1/v2 时代产物 | 遗留 |
| `sparse_indexer_pack_seq`、`index_select_dim0`、`linear_prepacked` | 仅单测 | 无消费者 |

## 4. 建议

1. 决定 mega-kernel 接线的提交节奏：torch_xcpu 与 plugin 两边改动需成对提交，否则 `FusedMlaQNormQbUk` 导入会失败。
2. 跑一次 GLM-5.2-FP8 TP4 eager 端到端对拍并留 profiler 记录（proposal 里的算子序列应从 `rms_norm ×2 + fp8_scaled_mm + einsum` 变为单个 `fused_mla_qnorm_qb_uk`），并修正 README 状态段。
3. `fused_ffn` 在没有 B 族实现前，x86 上多了一次 workspace 分配与 ATen 回退，可考虑默认关闭或在 preset 中显式说明。
4. 清理第 3 节的遗留算子，减少 NPU 算子团队的移植面。

## 5. Mega-kernel 专项复核（2026-11-01）

### 5.1 端到端验证

`presets/mpi/moe/GLM-5.2-FP8_dp1_tp4_eager_fp8_kvcache.sh --suite-test` 在 10-30/10-31 跑了 4 次完整 suite，均 ALL PASS（exit 0）：

| 运行 | 版本 | single-0.5k decode tps/req | 一致性告警 |
| --- | --- | --- | --- |
| 10-30 15:30 | 关闭（0 次调用） | 2.30 | multi 阶段 3–4 种不同输出（基线即有） |
| 10-30 14:43 | A0 (v1) | 2.18 | 同上 |
| 10-30 16:15 | A1 (v2) | 1.52 | 同上 |
| 10-31 07:35 | A1 (v2) | 2.12 | 同上 |

- 功能正确：无 Traceback（结尾 EngineDeadError 是 shutdown 时序产物），回答语义正确。
- multi-stream 下同 prompt 不同输出的告警在关闭 mega-kernel 时同样存在，非本算子引入。
- **开启后 single-stream greedy 输出文本与关闭时不一致**（A0 与 A1 彼此一致）。单测容差 1 bf16 ulp + 2^-10·max 内，差异来自累加顺序/反量化舍入点，但缺少对真实 AMX q_b_proj 路径的逐位对拍，也没有 profiler 证明算子序列已折叠（`vllm_profile/` 最新产物是 10-28，早于接入）。
- x86 上无性能收益（预期：x86 仅是模拟），默认开启反而有约 5% decode 回退。

### 5.2 plugin 开关

- 有开关：`VLLM_XCPU_FUSED_MLA_QNORM_QB_UK`，但**默认开**，且用裸 `os.getenv` 读取，没进 `vllm_xcpu_plugin/envs.py` 注册表。
- 只在 eager 生效（compile 下跳过）。关闭时 `_xcpu_keep_raw_fp8` 也不设置，不多占内存。
- 建议：迁入 `envs.py`，默认 0；preset 显式打开。

### 5.3 torch_xcpu 侧提交就绪度

通过项：
- 符合 CLAUDE.md 算子规范：impl 在 `torch_xcpu_impl`，`switch(version)` A0/A1/B0，模板实例在 launch 外选定，`MCPU_LAUNCH_TIMED_KERNEL` 只捕获指针/标量结构体，`KernelPointerMemoryGuard` 覆盖全部指针。
- 测试：`test/mk make run`（v2 测试在同一二进制里跑 Pe64 与 Pe96 两种拓扑，OMP 团队 <96 时跳过 Pe96）全部 PASS；`test_fused_mla_qnorm_qb_uk.py` A0、A1 各 10 用例通过。
- 封装内回退使 plugin 只见一个契约。

提交前必须处理：
1. `./format.sh --check` 4 个新文件不合规（csrc 两个、impl 头/源），pre-commit 钩子会拒绝。
2. `ref.rma.md`（硬件手册摘录）放在仓库根目录且未跟踪，不应随提交进入；移到 `docs/ref/` 或排除。
3. `docs/mega-kernel/README.md` "状态"段仍写"还没有接入"。
4. ruff：`mla_qnorm_qb_uk.py` UP035（`typing.Callable`→`collections.abc`），`embedding.py`/测试文件 import 排序。

建议整理：
5. （已解决）PE 拓扑由全局 `MK_SIM_*` 宏改为模板参数 `hal::Topology<NumPe, PesPerSubDie>`，Makefile 不再需要为 mk 编译单元加宏，ODR 风险消除，v2 在库内同时支持 64 / 96 PE。
6. `sync_sub_die` 由 thread_local sense 改为读 barrier 自身 sense 是独立的正确性修复（线程↔PE 映射跨 launch 不固定），应单独成提交并写明。
7. `fused_mla_rope_cat_cache.cpp` 重复了 `fused_mla_rope_qproj_cat_cache.cpp` 的 8 条检查；plugin `forward_with_rope_qnope` 重复了 `forward_with_rope` 融合分支的 prepare/rope/vproj 三步。可各抽一个公共函数。
8. 形状常量（2048/512/64/576/16）在 csrc、impl 头、Python 三处各写一遍，GLM-5.2 专用范围内可接受，但 csrc 可直接用 `kFusedMlaQNormQbUk*`。

建议提交拆分：(a) xthdS `xthd_launch_n` + hal 模拟拓扑宏 + barrier 修复 + v1 改名；(b) torch_xcpu 算子注册、Python 封装、单测、文档；(c) plugin 接线（依赖 b，默认关闭）。

## 6. 整改与复验（2026-11-01）

按第 5 节意见修改后的状态。

### 6.1 行为变更

| 项 | 现在 |
| --- | --- |
| 开关 | `VLLM_XCPU_FUSED_MLA_QNORM_QB_UK`（已进 `vllm_xcpu_plugin/envs.py`）：`0` 默认关 / `1`、`v1` → A0 / `2`、`v2` → A1。plugin 在 import torch_xcpu 之前写入 `TORCH_XCPU_FUSED_MLA_QNORM_QB_UK_IMPL`（该变量已显式设置时以它为准）。关闭时 MLA forward 与融合前一致，也不保留 FP8 权重副本 |
| compile | 支持。算子在 C++ 内按 token 数分派：T ≤ 16 走 mega-kernel，T > 16 在同一个算子里顺序跑 rms_norm×2 → 层自己的 packed FP8 `q_b_proj` → einsum（调用方传入 `w_qb_packed / w_qb_packed_scale / fp8_backend`），调用方与 compile 图都不再按 T 分支 |
| 非 x86 | `USE_ACCELERATOR` 构建不编译 A0/A1（x86 模拟），B0 占位；`hal::Topology` 为模板参数，无拓扑宏；通用 x86-64（无 AVX512/AMX）与 `-DUSE_ACCELERATOR` 下该编译单元均通过语法编译 |
| 去重 | `csrc/fused_mla_rope_checks.h` 被 `fused_mla_rope_qproj_cat_cache` 与 `fused_mla_rope_cat_cache` 共用；plugin 两处 fused 分支合并为 `XcpuMLAAttention._attend_fused`；形状常量集中到 `fused_mla_qnorm_qb_uk.h` |
| 整理 | 4 个新 C++ 文件及改动过的 C++ 文件已 clang-format（`./format.sh --check` 通过）；ruff 干净；README 状态已更新；`ref.rma.md` 移到 `docs/npu-handoff/` |

### 6.2 验证

- `test/deepseek_v32/test_fused_mla_qnorm_qb_uk.py`：A0、A1 各 13 例通过。参考改为**真实 FP8 后端**（`initialize_fp8_linear` + rms_norm + einsum），而不是之前的 CPU float；T = 1/7/16 在舍入容差内，T = 17/33/64 与未融合序列**逐位相等**；另有一例 `torch.compile(dynamic=True, fullgraph=True)` 跨 T = 5/16/40 与 eager 逐位相等。
- 发现：真实 AMX FP8 线性在 **M = 1 与 M ≥ 2 的结果相差约 1 ulp**（同一输入行，T=1 和 T=7 中的该行不同），mega-kernel 与 M ≥ 2 路径一致且与 T 无关。所以单流 decode（T = 1）开启后文本与关闭时不同是舍入路径不同，不是数据流错误。
- `test/mk`：`make run`、`make accel-check` 全部通过。
- GLM-5.2-FP8 TP4 e2e（`--suite-test`，只跑 single-0.5k / single-4k / multi-0.5k，并发 4）：

  | 配置 | 结果 | 融合算子调用 |
  | --- | --- | --- |
  | eager，默认（关闭） | PASS | 0 |
  | eager，v1 | PASS | 12636 |
  | compile，v1 | PASS | 12636（compile 图内） |
  | eager，v2（完整 suite，multi-4k 被 3000 s 超时截断） | single-0.5k/4k、multi-0.5k PASS | 12636 |

  compile+v1 的三个阶段输出与 eager+v1 **逐字相同**，两者与关闭时不同（见上，舍入路径）。x86 上吞吐无差异（噪声内），符合"x86 只是模拟"的预期。
- 发现并修掉的问题：算子 schema 里的 `str` 参数不被 AOTI shim 生成器支持（`aoti_torch_xcpu_ops.txt` 里被 skipped），compile 时报 `aoti_torch_mcpu_fused_mla_qnorm_qb_uk was not declared`；改为 `int fp8_backend`（`Fp8LinearBackend` 整数值）后通过。

### 6.3 遗留

- `test_fused_mla_rope_cache.py` / `test_fused_mla_rope_qproj_cat_cache.py` 里 30 个例子在 `ops.py:309`（Python 侧 "DMA outer strides must be four-byte aligned"）失败，看起来与本次改动无关（该检查未改动，失败例子在到达 C++ 之前就被拒绝；未在 HEAD 上复跑确认）；qproj 的 37 个到达 C++ 的例子通过。
- 尚未提交；建议仍按 a/b/c 三段（hal barrier 修复单独说明原因）。
- 与关闭相比文本不同这件事，若要做逐位验收，需要和 M ≥ 2 的 AMX 路径对拍（单测已覆盖），线上 T = 1 的差异属于 AMX 自身的 M = 1 路径。

## 7. Phase-1：qnorm_qb_uk pattern pass（compile 路径融合，2026-11-08）

### 7.1 接入方式

- MCPU compile 是 DYNAMO_TRACE_ONCE + Inductor，vllm 的 PostGradPassManager 不运行；
  plugin 在 `platform.py` 经 `inductor_compile_config["post_grad_custom_post_pass"]`
  安装 `XcpuPostGradPass`（`torch._inductor` CustomGraphPass）。pass 收到的是
  auto_functionalized 形式的 post-grad FX 图（reinplace 之前）。
- 开关 `VLLM_XCPU_COMPILE_PASS_QNORM_QB_UK`（envs.py，默认 0；取值语义同
  `VLLM_XCPU_FUSED_MLA_QNORM_QB_UK`，如 3/A3）。pass 的 `uuid()` 为全部 pass
  源文件哈希，自动进入 Inductor FX cache key 与 vllm `compute_hash`。
- 匹配（每层，def-use 而非发射顺序）：cache-write af → kv rms → qkv_base
  （fused_qkv_a 线性输出）→ 2 个 rms → q_b FP8 线性（形状 + 前向可达 W_UK
  einsum 双重判定；TP4 下 indexer wq_b 与 q_b_proj 同形 [4096,2048]，靠 einsum
  区分）→ rotary → einsum → cat；静态 shape 门（qkv [T,2624]、W_UK_T
  [16,192,512]、cat [T,16,576]）通过后整体替换为
  `fused_mla_qnorm_qb_uk_rope_cache` af，三个对外结果（q_c_norm / kv_cache /
  out_q）按 def-use 重定向，region 外消费者经 view 链 rebase。
- 实现-版本传递：`_select_fused_mla_qnorm_qb_uk_impl` 在 FUSED_MLA=0（默认关）
  时改取 COMPILE_PASS 环境的版本；否则 torch_xcpu 会被钉到 impl 0（mega-kernel
  v0 语义）。
- plugin logger 现在以 `vllm` logger 为 parent（vllm 只给 `vllm.*` 层级配
  handler，否则 plugin INFO 全部不可见）。

### 7.2 验证

- 离线 harness：`/root/.claude/jobs/9970d539/tmp/offline_pass_test.py`（可移至
  vllm_scripts/diag/mk_pass/）从 AOT cache 字符串或 print_readable dump 重建
  af 形式 FX 图，运行 pass 后用 inductor 同构的 stable_topological_sort +
  FakeTensorUpdater + lint 校验。TP4 全部 8 个 rank 的 live pre-dump
  `replaced=7`；TP1 正确 0（64 头/rank 不支持）。
- 端到端 DP2×TP4×EP8（preset
  `vllm_scripts/diag/mk_pass/presets/phase1_pass_qnorm_qb_uk_dp2tp4ep8.sh`，
  `VLLM_XCPU_DUMP_POST_GRAD_GRAPH` 输出图）：每 rank 日志
  `XCPU post-grad pattern pass installed (…=A3)`、`XCPU post-grad pass fused
  7 MLA front halves`，0 skip；post 图 af 139→104（7 fused + 42 erased），
  redirect getitem [2]/[3]/[6]；服务正常生成，输出与 unfused 基线逐字一致
  （dummy 权重 + 贪心采样）。

### 7.3 已知坑（后续 phase 复用）

- live post-grad 图的 view 连接是 `aten.reshape`（AOT 图里是 `aten.view`）：
  transparent 集合必须含完整 view 族（reshape/t/transpose/permute/expand/
  squeeze.dims），否则 def-use 走查在 reshape 处断链、整层静默 skip。
- auto_functionalized tuple 取结果必须按 schema 推导的 clone 下标
  （`_af_indices`），不能假设 tuple 顺序；`-> ()` 算子也有前导 None 槽。
- post-grad dump 用 print_readable（带注解）输出后可直接离线回放；terse
  `str(graph)` 无 meta，无法回放。

### 7.4 命名重构（验证后同日）

- pass 模块移至
  `vllm_xcpu_plugin/compile_passes/deepseek_v32/fused_mla_qnorm_qb_uk_rope_cache.py`
  （入口函数 `replace_with_fused_mla_qnorm_qb_uk_rope_cache`），env 改全名
  `VLLM_XCPU_COMPILE_PASS_DEEPSEEK_V32_FUSED_MLA_QNORM_QB_UK_ROPE_CACHE`
  （原名 `VLLM_XCPU_COMPILE_PASS_QNORM_QB_UK`）。
- 删除 2b97937 引入的 eager 前半融合路径（`attention/mla.py` 的
  `_use_qnorm_qb_uk`/`_enable_qnorm_qb_uk`/`_forward_qnorm_qb_uk`/
  `_post_load_hooks` 与 env `VLLM_XCPU_FUSED_MLA_QNORM_QB_UK`）：compile pass
  是该算子唯一发射方，eager 保持 unfused 回归基线。impl 版本传递改为只读
  compile-pass env。历史 mk_v1/v2 preset 里对该 env 的引用随之失效（preset
  等同普通 unfused 运行）。
- 诊断 preset 改名
  `deepseek_v32_fused_mla_qnorm_qb_uk_rope_cache_dp2tp4ep8.sh`。
