# 13. DeepSeek-V4.1 Engram 拆分（Engram disaggregation）

Engram 表放到独立的 Engram rank 上（`VLLM_XCPU_ENABLE_ENGRAM_DISAGG=1`），注意力 rank 不再持表。
设计与实现见 `docs/dev_log/20261104_DeepSeek-V4.1-Engram-disaggregation.md`。

## 前置条件

- 按 `torch_mcpu → torch_xcpu → torch_mpi_ext` 重新编译（新增 `deepseek_v41_engram_remote_*` / `deepseek_v41_engram_serve` 算子）。
- MPI launcher（`VLLM_CPU_USE_MPI=1`、`VLLM_USE_MPI_COORD=1`），单机 `mpirun` 即可，不需要 hostfile 或 UCX 参数。
- `USER_VLLM_ENGRAM_SIZE` 必须整除 24（1/2/3/4/6/8/12/24）；`USER_VLLM_MPI_SIZE = DP × TP × PP + USER_VLLM_ENGRAM_SIZE`；
  `VLLM_MPI_WORKER_TEMPLATE=serve/serve_engram_mp_rpc_all_mpi_template.sh`（`serve_mpmd_mp_rpc_all_mpi_template.sh` 的包装，
  与 AF-EP 共用，可同时设置 `USER_VLLM_EP_SIZE` 与 `USER_VLLM_ENGRAM_SIZE`）。preset 已经设置好这些变量。

## 算子级与插件级检查（已执行，2026-11-04）

```bash
cd torch_xcpu/test
python -m pytest -q ops/custom_ops/deepseek_v41/test_engram_remote_decode.py          # gather + decode 与本地 lookup 逐位一致
TORCH_XCPU_DEEPSEEK_V41_ENGRAM_REMOTE_DECODE_IMPL=A0 TORCH_XCPU_DEEPSEEK_V41_ENGRAM_SERVE_IMPL=A0 \
  python -m pytest -q ops/custom_ops/deepseek_v41/test_engram_remote_decode.py        # 标量参考版本
TORCH_XCPU_ENABLE_CHECK=1 mpirun --allow-run-as-root --bind-to none -np 4 \
  python mpi_custom_ops/test_engram_remote_roundtrip_mpi.py                           # A2 + E2
XCPU_ENGRAM_TEST_ATTENTION_RANKS=1 ... -np 3                                          # A1 + E2
XCPU_ENGRAM_TEST_ATTENTION_RANKS=2 ... -np 5                                          # A2 + E3

cd vllm-xcpu-plugin
python -m pytest -q tests/engram_disagg tests/upstream_contracts
python -m vllm_xcpu_plugin.upstream_contracts lint deepseek_v41
python -m vllm_xcpu_plugin.upstream_contracts provenance deepseek_v41
```

## 端到端（待执行：用户审阅代码后）

dummy 8 层模型（engram 层 [1, 2]，两表不到 30 MB），全部单机：

```bash
cd vllm_scripts
export VLLM_TEST_MAX_WAIT=1800
# A1 + E2（USER_VLLM_ENGRAM_SIZE 可改为 1 / 3）
./run_vllm_test.sh -e presets/mpi/engram/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp1_tp1_eager_engram2.sh --multi-test --multi-test-temperature 0
# compile（先清理 vllm_cache）
./run_vllm_test.sh -e presets/mpi/engram/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp1_tp1_compile_engram2.sh --multi-test --multi-test-temperature 0
# A2（dp1 tp2，非 SP：每个 TP rank 请求全部 token）+ E2
./run_vllm_test.sh -e presets/mpi/engram/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp1_tp2_eager_engram2.sh --multi-test --multi-test-temperature 0
# 其非拆分 baseline（TP2 的归约次序与 TP1 不同，不能拿 serial TP1 preset 对照）
./run_vllm_test.sh -e presets/mpi/dense/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp1_tp2_eager.sh --multi-test --multi-test-temperature 0
# A4（dp2 tp2 EP，SP 分片、空分片、空闲 DP rank 的 dummy run）+ E2，eager / compile
./run_vllm_test.sh -e presets/mpi/moe/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp2_tp2_ep_eager_alltoallv_v6_engram2.sh --multi-test --multi-test-temperature 0
./run_vllm_test.sh -e presets/mpi/moe/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp2_tp2_ep_compile_alltoallv_v6_engram2.sh --multi-test --multi-test-temperature 0
# DSpark 投机解码（草稿模型无 engram；每次 verify 一次目标 forward = 一次交换）+ E2，eager / compile
./run_vllm_test.sh -e presets/mpi/engram/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp1_tp1_eager_dspark_5_engram2.sh --multi-test --multi-test-temperature 0
./run_vllm_test.sh -e presets/mpi/engram/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp1_tp1_compile_dspark_5_engram2.sh --multi-test --multi-test-temperature 0
```

判定：

- 同一组 greedy 请求的输出与对应的非拆分 preset（`presets/serial/..._dp1_tp1_eager.sh`、`presets/mpi/dense/..._dp1_tp2_eager.sh`、
  `presets/mpi/moe/..._dp2_tp2_ep_eager_alltoallv_v6.sh`、`presets/serial/dspark/..._dp1_tp1_eager_dspark_5.sh`）逐 token 一致
  （lookup 逐位一致，其余路径不变）。2026-11-04 实测：7 个 Engram preset 全部一致，见 dev_log 3.2。
- `mpi_workers.log` 中每个 Engram rank 打印 `Engram rank N tables ready: ...` 与 `Engram rank N serving`；`run_result.json` 三个退出码均为 0、
  清理无残留进程。launcher 结束固定 world 时直接 TERM 所有 rank（与 AFD 相同），所以正常测试流程里不会出现 `Engram rank N stopped`；
  该行只在 A 侧 worker 的 shutdown 先于 launcher 清理运行（发出 STOP）时出现，属 best-effort。
- DP2 下用并发请求制造一个 DP rank 空闲、另一个忙的步，服务不挂。

真实权重（E8，本机 1 TiB 内存可容纳，E rank 各约 23.6 GiB）：

```bash
# A4（dp2 tp2 EP，eager）+ E8
./run_vllm_test.sh -e presets/mpi/moe/DeepSeek-V4.1-Flash_dp2_tp2_ep_eager_alltoallv_v6_engram8.sh --multi-test --multi-test-temperature 0
# A1（compile + DSpark）+ E8
./run_vllm_test.sh -e presets/mpi/engram/DeepSeek-V4.1-Flash_dp1_tp1_compile_dspark_5_engram8.sh --multi-test --multi-test-temperature 0
# 保持服务运行并跑正确性脚本（--no-test 只启动服务；脚本结束后 Ctrl-C 结束 launcher。
# 从脚本里以后台作业启动时 launcher 忽略 SIGINT，要用 kill -TERM <launcher pid>，它 trap TERM 后同样走清理/归档）
./run_vllm_test.sh -e <上述 preset> --no-test &
python serve_test/test_basic_correctness.py --model <USER_VLLM_MODEL 路径> --base-url http://127.0.0.1:${USER_VLLM_PORT:-14800}
```

判定：`serve_test/test_basic_correctness.py` 全部 PASS；greedy 请求与非拆分 EP preset 逐 token 一致——**并发 multi-test 下
batch 组成不同会让 greedy 输出在十几个 token 后分叉（基线之间也如此）**，逐 token 对照请用串行探针：
`python serve_test/test_greedy_logprobs.py dump http://127.0.0.1:14800 <模型路径> engram.json`（基线同样 dump 一份），
`python serve_test/test_greedy_logprobs.py compare engram.json baseline.json` 应输出 `RESULT: IDENTICAL` 且每步 top-5
logprob 差值为 0；
单流探针记录 decode ms/token；profiler 中每步 A 侧 Engram 相关算子只剩 `engram_hash`、`engram_remote_pack`、
`engram_remote_request_comm`（发起两个非阻塞集合，应为微秒级）、`engram_remote_collect_comm`（Waitall）、
`engram_remote_decode`×2、`engram_post_wkv`×2。

2026-11-04 实测（dev_log 3.3）：A4+E8 EP eager 的 `test_basic_correctness.py` ALL PASSED，串行探针与 EP 基线逐步 top-5 logprob
完全相同；A1+E8 compile+DSpark 与其非拆分基线都在 needle 长提示用例上崩溃（compile+DSpark 既有问题，与拆分无关），其余用例 PASS。
decode ms/token 与 profiler 算子序列本轮未记录。

## 失败路径（启动即报错）

- `USER_VLLM_ENGRAM_SIZE=5`（不整除 24）；
- `--engram-config '{"embedding_across_dp": true}'`；
- 同时设置 `VLLM_XCPU_ENABLE_AF_EP=1`；
- 不用 MPI launcher（`VLLM_CPU_USE_MPI=0`）；
- `--load-format` 不是 `auto`/`safetensors`/`dummy`；PP>1；DBO。
