# GLM5.2：DP2/TP2 MoE 序列并行（SP-MoE）服务验证

验收 GLM5.2 在 DP=2、TP=2、EP=4、`mpi_alltoallv_v6` 下，MoE 序列并行（SP）开/关、eager/compile 的服务行为。本指南只记录复用步骤与标准；实际运行结果放在对应版本目录。单卡验证见 [GLM5.2 串行](09-glm52-serial.md)，Qwen 的 MoE 通信验证见 [Qwen3-MoE](08-qwen3-moe.md)。

## 准备

- 按 [AGENTS.md](../../AGENTS.md) 核对构建与内存保护设置，记录各仓库 HEAD、未提交修改、模型和环境版本。
- 默认使用 7 层 dummy 权重（`--load-format dummy`）：FP8 为 `GLM-5.2-FP8-dummy-7-layer-lite`，MXFP4 为 `GLM-5.2-MXFP4-W4A16-dummy-7-layer-lite`，位于 `models/ZhipuAI/`。dummy 权重只验证并行与调度行为，不判断精度。
- 脚本固定的配置：MRV2、PD_MODE=MIXED、MPI launcher、FP8 KV cache、`max_model_len=4096`、`max_num_batched_tokens=32`、seed=42，并带 `--kernel-config {"enable_jit_warmup":false}`。32 token 的预算保证长 prompt 被切成多块 prefill。
- SP 配置本身只在 EP、TP>1、DP>1 且 all2all 后端声明支持序列并行时生效，两种取值下 worker 日志里都是 `sp_size=2`。`VLLM_DISABLE_ATTN_REDUCE_SCATTER_FOR_MOE` 控制各层是否在 decoder 层之间保留 SP 切分的 hidden states：0（脚本中的 `on`）保留，attention 用 reduce-scatter；1（`off`）不保留，attention 做 all-reduce，MoE 仍在内部切分、计算并 gather。本指南所说的 SP 开/关指这个开关。
- 确认测试端口空闲、没有残留的 `mpirun`；每个组合独立启动、使用独立编译缓存。冷编译不得复用旧产物。

## 执行

`test_glm52_sp_moe.py run` 为每个组合生成预设、启动服务、发送固定请求集并保存 `*-responses.json`；`compare` 比较已保存的运行。

```bash
source .venv/bin/activate
OUT=vllm_scripts/logs/glm52-sp-moe-$(date +%Y%m%d)
T=vllm_scripts/serve_test/test_glm52_sp_moe.py
# 每个组合一次（QUANT=FP8|MXFP4，MODE=eager|compile，SP=on|off），顺序执行
for q in FP8 MXFP4; do for m in eager compile; do for s in on off; do
  python $T run $q $m $s --out-dir $OUT
done; done; done
# 先看每组自洽性，再两两比较
python $T compare $OUT \
  FP8-eager-spon:FP8-eager-spoff FP8-compile-spon:FP8-compile-spoff \
  FP8-eager-spon:FP8-compile-spon FP8-eager-spoff:FP8-compile-spoff \
  MXFP4-eager-spon:MXFP4-eager-spoff MXFP4-compile-spon:MXFP4-compile-spoff \
  MXFP4-eager-spon:MXFP4-compile-spon MXFP4-eager-spoff:MXFP4-compile-spoff
```

请求集（temperature=0、ignore_eos、记录 logprob）：

- `single`：单请求，decode 时一个 DP rank 只有 1 个 token（小于 TP），另一个 rank 运行空 batch。
- `long`：约 200 token 的 prompt，分块 prefill，块大小不被 TP 整除。
- `batch`：一次 completion 调用中 3 条不等长 prompt。
- `c0`–`c4`：5 路长度不同的并发请求，分散到两个 DP rank。
- `single-repeat`：并发后重发 `single`，命中前缀缓存并复用状态。

单次运行失败时，脚本退出码非 0，`*-launcher.log` 保存服务日志（含各 rank 的 worker 日志）。

## 通过标准

- 8 个组合全部 `completed`，所有请求返回 200，无 `EngineDeadError`（清理信号后出现的 `EngineDeadError` 与请求阶段错误分开记录）。
- 所有运行的 worker 日志中 `FusedMoEParallelConfig` 显示 `sp_size=2` 且 `all2all_backend='mpi_alltoallv_v6'`；若是 `sp_size=1`，说明 SP 配置没有生效，结果不能作为 SP 证据。
- `single == single-repeat`（`compare` 打印）。MXFP4 在近似平局下的前缀缓存路径可能不一致，需结合下条判断。
- **FP8 是判定依据**：SP 开 vs 关、eager vs compile 的四组比较都应 `prefix-equal 160/160`，`max |dlogprob| 0`。
- **MXFP4 不以 token 一致性判定**：dummy 权重下 logits 近似均匀，top-1 logprob 约等于 −ln(词表大小)（≈ −11.950），token 差异多是近似平局的 argmax 翻转。要求 `compare` 打印的 top-1 logprob 区间很窄（量级 1e-3），共享前缀内 `max |dlogprob|` 为 0；compile 下 SP 开/关通常仍应逐位一致。数值结论需要真实权重复核。
- compile 在引擎 READY 之前完成编译：没有首个请求触发的懒编译；编译产物里的图是动态 shape（含 symint 输入），不是 token 数固定为 1 的静态图。出现首个 prefill 返回 500 且日志只有 `API call failed` 时，先检查是否有 rank 在空闲时特化出静态 T=1 的图。
- 对照上游检查 `sparse_attn_indexer` 与 `torch_xcpu/csrc/sparse_indexer.cpp` 的升级核对清单（函数头注释）。

## 排错线索

- DP + compile 的预热由插件 worker 的 `compile_or_warm_up_model` 负责：`DYNAMO_TRACE_ONCE` 下用 `max_num_batched_tokens` 先跑一次 dummy 批。该日志行不会出现在 worker 日志里，判断预热生效以编译时间点和动态图为准。
- AOTI shim 会丢掉 C++ `TORCH_CHECK` 的文本，只留 `API call failed at main.cpp line N`；需要原因时在 eager 下复现。
- 不要在 `bash` 里用 `pgrep -f "vllm serve|mpirun"` 等待上一组结束，它会匹配到自身命令行；用 `pgrep -x mpirun`。

## 真实权重基本项

真实权重的 DP2/TP2 配置（如 `presets/mpi/moe/GLM-5.2-FP8_dp2_tp2_{eager,compile}_fp8_kvcache_alltoallv_v6.sh`、`GLM-5.2-MXFP4_dp2_tp2_*`）要把 [GLM5.2 串行](09-glm52-serial.md)里的 **4096 大海捞针与长上下文并发** 作为基本项：用同一 preset 启动后运行 `long_context_accuracy.py`（`--lengths 4k`，以及 `--lengths 0.5k,4k,8k,16k --multi-stream -n 2`，均加 `--disable-thinking`），通过标准相同。DP2 下并发请求会分散到两个 DP rank，SP 下 token 数需要被 TP 切分，这是单卡覆盖不到的组合。

本指南的 dummy 脚本不含该项（`max_model_len=4096` 放不下 4k 用例），真实权重两 rank 需要接近 800 GiB 内存，需在内存充足的机器上执行，运行前确认空闲内存。

这里不覆盖其他真实权重配置、Eagle/MTP、AF-EP、DBO、EPLB、Qwen MoE 的 SP 回归、真实 NPU 或性能对比；这些配置需要独立验收。
