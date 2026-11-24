# 14. DeepSeek-V4.1 EngramParallel（EP 域全量 rank 2D 分片）

将全部 Engram 表按 `(column-group × row-range)` 静态二维分片分散到所有模型 worker rank（`N = DP × TP = EP` 域，环境变量 `VLLM_XCPU_ENABLE_ENGRAM_PARALLEL=1`），消除独立 Engram 服务节点与 MPMD 启动模板。设计与实现见 `docs/dev_log/20261120_DeepSeek-V4.1-EngramParallel.md`。

## 前置条件

- 按 `torch_mcpu → torch_xcpu → torch_mpi_ext` 重新编译（包含 `deepseek_v41_engram_parallel_*` 系列算子及 AOTI wrappers 检查）。
- 使用常规单角色 MPI launcher（`VLLM_CPU_USE_MPI=1`、`VLLM_USE_MPI_COORD=1`、`serve_mp_rpc_all_mpi_template.sh`），不需要额外 `USER_VLLM_ENGRAM_SIZE` 或 `USER_VLLM_MPI_SIZE`。
- `PP = 1`，未开启 PCP / DCP / elastic EP / DBO / ubatching，未开启 `skip_dp_coordination`，未同时设置 `VLLM_XCPU_ENABLE_ENGRAM_DISAGG=1`，未启用 `engram_config.embedding_across_dp`。

## 算子级与插件级检查（已执行，2026-11-24）

```bash
cd torch_xcpu && ./build-all.sh && python scripts/generate_aoti_wrappers.py --check

cd torch_xcpu/test
pytest -s -x -v ops/custom_ops/deepseek_v41/test_engram_parallel.py ops/custom_ops/deepseek_v41/test_engram_remote_decode.py ops/custom_ops/deepseek_v41/test_engram_lookup.py  # 默认 A1: 73 passed
TORCH_XCPU_DEEPSEEK_V41_ENGRAM_PARALLEL_GATHER_IMPL=A0 TORCH_XCPU_DEEPSEEK_V41_ENGRAM_PARALLEL_DECODE_IMPL=A0 \
  pytest -s -x -v ops/custom_ops/deepseek_v41/test_engram_parallel.py  # 标量参考 A0: 46 passed
for np in 1 2 3 4 5; do
  TORCH_XCPU_ENABLE_CHECK=1 mpirun --allow-run-as-root --bind-to none -np $np \
    python mpi_custom_ops/test_engram_parallel_roundtrip_mpi.py
done

cd vllm-xcpu-plugin
pytest -s -x -v tests/engram_parallel tests/engram_disagg tests/upstream_contracts
mpirun --allow-run-as-root --bind-to none -np 4 python tests/mpi/engram_parallel_model_mpi.py
python -m vllm_xcpu_plugin.upstream_contracts lint deepseek_v41
python -m vllm_xcpu_plugin.upstream_contracts check deepseek_v41
python -m vllm_xcpu_plugin.upstream_contracts provenance deepseek_v41
```

实测结果：

- `generate_aoti_wrappers.py --check` 退出码 `0`。
- `torch_xcpu` 单进程默认 A1 套件（`test_engram_parallel.py`、`test_engram_remote_decode.py`、`test_engram_lookup.py`）`73 passed`（涵盖 `world_size` `1, 2, 3, 4, 8, 32, 48, 64`、空 token、`DEAD_ID`、SP padding、OpenMP pack 与重复热点防溢出）；标量参考 A0 `46 passed`。
- MPI 往返测试 `test_engram_parallel_roundtrip_mpi.py`（`TORCH_XCPU_ENABLE_CHECK=1`）在 `np=1, 2, 3, 4, 5` 上全部通过（含动态 `MPI_Ialltoallv` 参差/零发送清零与尾部哨兵校验，以及 `np=2, 4` 的 `DP × TP` + `SP` 验证）。
- `vllm-xcpu-plugin` `tests/engram_parallel` + `tests/engram_disagg` + `tests/upstream_contracts` 全部通过（78 passed，含 4-rank synthetic safetensors MPI 模型测试）；`upstream_contracts` 的 `lint`、`check`、`provenance`、`mypy`、`ruff` 均通过。

## 端到端测试（dummy 已执行，2026-11-24）

dummy 8 层模型（engram 层 `[1, 2]`，`DP=2, TP=2, EP=4`）：

```bash
cd vllm_scripts
export VLLM_TEST_MAX_WAIT=1800
rm -rf vllm_cache

# 非 EngramParallel 基线（DP2 x TP2 EP, eager）
./run_vllm_test.sh -e presets/mpi/moe/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp2_tp2_ep_eager_alltoallv_v6.sh --multi-test --multi-test-temperature 0
# EngramParallel（DP2 x TP2 EP, eager, mpi_alltoallv_v5）
./run_vllm_test.sh -e presets/mpi/moe/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp2_tp2_ep_eager_alltoallv_v5_engram_parallel.sh --multi-test --multi-test-temperature 0
# EngramParallel（DP2 x TP2 EP, compile mode:3, mpi_alltoallv_v5；先清理 vllm_cache）
rm -rf vllm_cache
./run_vllm_test.sh -e presets/mpi/moe/DeepSeek-V4.1-Flash-dummy-8-layer-lite_dp2_tp2_ep_compile_alltoallv_v5_engram_parallel.sh --multi-test --multi-test-temperature 0
```

真实权重（`DP=2, TP=2, EP=4`，`mpi_alltoallv_v5`，通过 `xcpu_engram_parallel_auto` 按 `[lo:hi]` 流式切片读取，本轮未运行）：

```bash
./run_vllm_test.sh -e presets/mpi/moe/DeepSeek-V4.1-Flash_dp2_tp2_ep_eager_alltoallv_v5_engram_parallel.sh --no-test &
python serve_test/test_greedy_logprobs.py dump http://127.0.0.1:${USER_VLLM_PORT:-14800} "${HOME}/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master" engram_parallel.json
python serve_test/test_greedy_logprobs.py compare engram_parallel.json baseline.json
```

判定：

- `run_result.json` 中退出码均为 `0`，服务正常退出、无残留进程。
- `mpi_workers.log` 中 4 个 rank 均输出 `EngramParallel ready: rank=<r>/4 layers=(1, 2) max_cols_per_rank=12 max_tokens=...`。
- 同一组 greedy 请求输出与非拆分 EP 基线逐 token 一致（并发 `--multi-test` 若因 batch 组成差异导致个别长回复分叉，以 `serve_test/test_greedy_logprobs.py compare` 输出 `RESULT: IDENTICAL` 为准）。

2026-11-20 初版基线对比（见 dev_log 第 5 节）：

- dummy 8 层 `dp2_tp2_ep` eager 基线（`20261120_080104_...IA618`）、EngramParallel eager（`20261120_080341_...GAyu2L`）、以及清理 `vllm_cache` 后的 EngramParallel compile `mode:3`（`20261120_080456_...JjCCxg`）三组运行 `run_result.json` 均返回 `{"run_exit_code":0,"cleanup_exit_code":0,"exit_code":0}` 且无残留进程。
- 两组 EngramParallel 运行的 `mpi_workers.log` 均在 4 个 rank 上打印 `EngramParallel ready: rank=0..3/4 layers=(1, 2) max_cols_per_rank=12 max_tokens=1024`。
- 5 条 `--multi-test --multi-test-temperature 0` 任务（`vllm_task_0..4.log`）在 eager EngramParallel、compile `mode:3` EngramParallel 与 eager 基线之间逐字节完全一致。

2026-11-24 动态 `MPI_Ialltoallv` + `mpi_alltoallv_v5` 复测：

- EngramParallel `mpi_alltoallv_v5` eager（`20261124_050117_...eAQwgl`）与清理 `vllm_cache` 后的 EngramParallel `mpi_alltoallv_v5` compile `mode:3`（`20261124_050225_...EkemxC`）`run_result.json` 均返回 `{"run_exit_code":0,"cleanup_exit_code":0,"exit_code":0}`；4 个 rank 均输出 `EngramParallel ready: rank=0..3/4 layers=(1, 2) max_cols_per_rank=12 max_tokens=1024`；5 条 `--multi-test --multi-test-temperature 0` 任务输出（`vllm_task_0..4.log`）在 eager 与 compile `mode:3` 之间逐字节完全一致。

## 失败路径（启动即报错）

- 同时设置 `VLLM_XCPU_ENABLE_ENGRAM_DISAGG=1` 与 `VLLM_XCPU_ENABLE_ENGRAM_PARALLEL=1`。
- 不用 MPI launcher（`VLLM_CPU_USE_MPI=0`）。
- `--engram-config '{"embedding_across_dp": true}'`。
- `PP > 1`、开启 PCP / DCP / elastic EP / DBO / ubatching、`skip_dp_coordination`、`enable_multithread_load=True`、或不支持的 `load_format` / `safetensors_load_strategy`。
