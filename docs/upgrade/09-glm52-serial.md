# GLM5.2：FP8 / MXFP4 串行 eager 与 compile 验证

验收 GLM5.2（稀疏 MLA / DSA）在 TP=DP=PP=1 下的 FP8 与 MXFP4（W4A16）权重，各自 eager 与冷 compile。本指南只记录复用步骤与标准；实际运行结果放在对应版本目录。多 rank 与序列并行见 [GLM5.2 DP2/TP2 SP-MoE](10-glm52-sp-moe-dp2tp2.md)。

## 准备

- 按 [AGENTS.md](../../AGENTS.md) 核对构建与内存保护设置，记录各仓库 HEAD、未提交修改、模型和环境版本。
- 真实权重单进程占用很大（FP8 约 400 GiB 量级），启动前确认空闲内存，并确保没有其他大模型作业。
- 两类模型：
  - 真实权重：用于精度 smoke。preset 为 `presets/serial/GLM-5.2-{FP8,MXFP4}_dp1_tp1_{eager,compile}_sparse_attn_fp8_kvcache.sh`。
  - 7 层 dummy 权重：启动快，只用来比较 eager/compile 的一致性，不判断精度。preset 为 `presets/serial/GLM-5.2-FP8-dummy-7-layer-lite_dp1_tp1_eager.sh` 和 `GLM-5.2-MXFP4-W4A16-dummy-7-layer-lite_dp1_tp1_eager.sh`。
- 记录实际 `max_model_len` 和 `max_num_batched_tokens`。长输入检索用例要求 `max_model_len` 至少 4096。
- compile 必须使用新缓存：备份现有 `vllm_cache`，或在 source preset **之后**覆盖 `VLLM_CACHE_ROOT` 和 `TORCHINDUCTOR_CACHE_DIR`，并 `unset USER_VLLM_EAGER_OR_NOT`。dummy 的 compile 在 eager preset 上叠加 `export USER_VLLM_EAGER_OR_NOT='""'` 和 `--compilation-config {"mode":3}`。

## 执行

从开发套件根目录激活 `.venv`，进入 `vllm_scripts` 启动服务（保留运行）：

```bash
source .venv/bin/activate
cd vllm_scripts
VLLM_TEST_MAX_WAIT=1800 ./run_vllm_test.sh --launcher mp \
  -e presets/serial/GLM-5.2-FP8_dp1_tp1_compile_sparse_attn_fp8_kvcache.sh --no-test
```

另一终端运行 smoke 客户端（端口取 preset 实际值）：

```bash
.venv/bin/python vllm_scripts/serve_test/test_glm52_serial.py \
  --base-url http://127.0.0.1:14800 --output /tmp/glm52-serial.json
```

客户端依次发送算术、常识、超过 `index_topk=2048` 的长输入检索、同一长输入的重复请求，再发一批两条不等长的 completion，保存响应和 logprob。对 FP8/MXFP4 与 eager/compile 的四种组合分别重复，每次独立启动服务。

### 长上下文基本项（4096 大海捞针与并发）

真实权重必跑；用 `vllm_scripts/serve_test/long_context_accuracy.py` 从公开 SQuAD 数据拼出长上下文，把含答案的段落放在约 85% 位置，再要求按固定格式给出“最终答案”。数据已缓存在 `serve_test/long_context_data/`，缺失时先 `--prepare-only` 生成（需要网络）。服务沿用上面启动的那一个（同一 preset），在 `vllm_scripts` 目录下：

```bash
# 1) 4096 大海捞针：单请求
python serve_test/long_context_accuracy.py -e presets/serial/<同一个 preset>.sh \
  --lengths 4k --disable-thinking --results-dir <结果目录>/needle-4k
# 2) 长上下文并发：4 个长度同时发出，并重复 2 轮（第 2 轮与第 1 轮请求相同，可命中前缀缓存）
python serve_test/long_context_accuracy.py -e presets/serial/<同一个 preset>.sh \
  --lengths 0.5k,4k,8k,16k --multi-stream -n 2 --disable-thinking \
  --results-dir <结果目录>/concurrent
```

- 工具会按 GLM tokenizer 重新计数，4k 用例实际约 4.6k token（另加 `--max-tokens` 输出预算），超过 `max_model_len` 时直接报错退出。因此 4k 单请求要求 `max_model_len` 至少 8192，并发项含 16k 用例（重算后约 18k token）要求至少约 20k；preset 默认的 32768 满足两者。上面 smoke 客户端用的 4096 配置不能用来跑它们。
- `--disable-thinking` 是基本项的固定设置：GLM 开启 thinking 时 256 token 的预算会被推理耗尽。需要另测 thinking 时单独加大 `--max-tokens`，不计入基本项。
- 4k、8k、16k 都超过 `index_topk=2048`，覆盖 indexer 真正做 top-k 选择的路径；0.5k 小于 2048，覆盖保留全部 key 的路径。

相关单测（各组计数不能相加当作唯一用例数）：

```bash
OMP_NUM_THREADS=2 TORCHINDUCTOR_CPP_WRAPPER=1 .venv/bin/python -m pytest \
  vllm-xcpu-plugin/tests/layers/test_mla_rope_kvcache_routing.py \
  vllm-xcpu-plugin/tests/layers/test_quark_mxfp4.py \
  vllm-xcpu-plugin/tests/fake_triton/test_runtime.py -q
# 从 torch_xcpu 源码目录之外运行（例如 /tmp），确保导入的是已安装的构建
cd /tmp && OMP_NUM_THREADS=2 $VIRTUAL_ENV/bin/python -m pytest \
  <开发套件根目录>/torch_xcpu/test/deepseek_v32/test_indexer_core.py -q
```

## 通过标准

- 客户端退出码为 0：算术返回 `42`，常识返回 `Paris`，长输入检索返回 `BLUE-7391` 且 `prompt_tokens > 2048`，重复请求同样通过；batch 返回两条 choice。
- 所有生成 token 的 logprob 为有限值；服务由测试主动停止时的 SIGTERM 退出码属预期，验收依据是请求结果。
- 同一量化下，eager 与 compile 的 chat 与 batch 的 token、logprob 逐项一致。长前缀重复请求允许 logprob 有 1e-5 量级差，不宣称缓存命中路径逐位等价；超出需单独调查。
- dummy 权重只作一致性证据：logits 近似均匀，MXFP4 出现近似平局的 token 翻转时不判为数值错误，需用真实权重复核。
- compile 在冷缓存下完成图捕获和 AOTI 产物生成，再完成请求；检查 `.main.cpp` 中存在 fused MLA、fused indexer QK/cache 和 C++ sparse indexer 的直接调用，而不是整段 Python 回退。
- 长上下文基本项：两条命令退出码均为 0（工具在任一用例未命中或失败时返回 1）。`results.csv`/`summary.json` 中每个用例 `success` 为真（按“最终答案”行评分并命中标准答案），`answer_format_valid` 为真；完成行显示 `N/N 个输出包含标准答案，0 个请求失败`；`finish_reasons` 为 `stop`，不是 `length`。并发运行中各长度都在同一轮内完成，无请求超时，服务日志无 worker 错误。
- 同一量化下 eager 与 compile 的长上下文基本项都要通过；任一未命中时，保存 `*_output.txt` 和 `*_meta.json`，先区分格式不合规（`answer_format_errors`）与答案错误，再调查。
- dummy 权重跑这两项只能验证流程：回答必然 MISS（工具退出码 1），此时只检查“请求全部完成、0 个失败、无服务错误”，不作为精度结论。
- 单测通过；任何未纳入的用例要写明原因，不能把子集通过写成整个文件通过。

这里不覆盖 TP/DP/EP、序列并行、投机解码（Eagle/MTP/DFlash）、真实 NPU 或性能基准；这些配置需要独立验收。
