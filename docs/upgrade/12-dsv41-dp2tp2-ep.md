# DeepSeek-V4.1-Flash：DP2/TP2 EP（mpi_alltoallv_v6）验证

验收 DeepSeek-V4.1-Flash 在 DP=2、TP=2 下的专家并行（EP=4，`mpi_alltoallv_v6`，fused EP MoE）。串行准备与通过标准见 [DeepSeek-V4.1 串行验证](11-dsv41-serial.md)。

## 准备

- TP=2 时每个 rank：32 个本地注意力头、4 个 wo_a 组；indexer（32 头）、compressor、`fused_wqa_wkv` 为 replicated；Engram 表按 hash head 在 TP 内分片，并在节点内 DP 组（日志 `Engram node-local DP group size: 2`）上再分片：MCPU 关闭了 `dp_shared_memory`，两个 DP 副本合计只存一份表，查表时在 DP 组内交换行。
- 专家 384 个分到 4 个 EP rank；DeepSeek-V4.1 的 SwiGLU clamp（`swiglu_limit=10`）由 fused EP v6 的 `fused_moe_compute` 阶段执行，v5 不支持 clamp，会直接报错。
- 真实权重 4 进程合计内存约：专家 1/4 × 4 + dense（TP 切分）× 2 个 DP 副本 + Engram 一份（约 189 GiB，按 TP × DP 分到 4 个 rank），启动前确认空闲内存。
- preset（各有 `_compile_` 版本，把文件名中的 `eager` 换成 `compile`）：
  - dummy：`presets/mpi/moe/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp2_tp2_ep_eager_alltoallv_v6.sh`
  - 真实权重：`presets/mpi/moe/DeepSeek-V4.1-Flash_dp2_tp2_ep_eager_alltoallv_v6.sh`
- DP2/TP2 + EP 自动开启 sequence parallel（`_use_sequence_parallel`），decoder replay 因此关闭；compile 下 SP 的 reduce-scatter pad 与 Engram SP 分片已改为符号形状（见 dev_log）。

## 执行

```bash
source .venv/bin/activate
cd vllm_scripts
./run_vllm_test.sh -e presets/mpi/moe/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp2_tp2_ep_eager_alltoallv_v6.sh
./run_vllm_test.sh -e presets/mpi/moe/DeepSeek-V4.1-Flash_dp2_tp2_ep_eager_alltoallv_v6.sh --no-test
python serve_test/test_basic_correctness.py --model ~/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master
```

## 通过标准

- dummy：DP2/TP2 的 greedy 输出与串行 dummy preset 完全一致（同一 prompt、同样 16 个 token）。
- 各 rank worker 日志出现 `XcpuGroupedGemmExperts ... use_ep=True`，无 Fake Triton `UnknownKernelError`（Engram 的 `_engram_select_rows_kernel` 已有 adapter）。
- 真实权重：smoke 客户端全部 PASS；`long_context_accuracy.py --lengths 0.5k,4k,8k,16k --multi-stream` 全部 HIT。

## 参考结果（2026-10-23）

- 真实权重 4 进程常驻约 568 GiB（专家按 EP4 切分，dense 每个 DP 副本一份，Engram 合计一份）。
- 单请求 4k：TTFT 16.2 s，decode 约 133 ms/token；串行同请求 TTFT 9.9 s（每个 DP 副本只用一半核心）。
- 并发 0.5k/4k/8k/16k：4/4 HIT，全部首 token 在 77 s 内。

## 参考结果：compile（2026-10-24）

- `DeepSeek-V4.1-Flash_dp2_tp2_ep_compile_alltoallv_v6.sh`：smoke 全部 PASS，并发 0.5k/4k/8k/16k × 2 轮 8/8 HIT；常驻约 568 GiB（与 eager 相同）。
- 与 DP2/TP2 eager 的 6 个 greedy 请求逐 token 相同（logprob 差 0）；与串行结果在若干 token 后分叉（TP 归约顺序不同，属预期）。
- 单流探针：decode compile 131–139 ms/token 对 eager 137–140 ms/token；12.6k token 请求 41.3 s 对 45.2 s。
