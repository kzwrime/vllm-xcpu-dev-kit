# AFD A/F compile 适配

本文记录 2026-09-30 的 A/F compile 实现、启动方式和验证范围。拓扑、权重配置、跨机 hostfile 与清理方法见 [端到端测试](e2e_test.md)。

## 启动方式

AFD 为每个模型和拓扑提供独立的 eager / compile preset。文件内固定 A/F 模式：

| BF16 A2/F2 示例 | A | F |
|---|---|---|
| `Qwen3-30B-A3B_dp1_tp2_af_ep_eager_v7.sh` | eager | eager |
| `Qwen3-30B-A3B_dp1_tp2_af_ep_compile_v7.sh` | compile | compile |

其余配置同样成对提供 eager / compile 文件，目前共八组、十六份 AFD V7 preset。Qwen 和 GLM dummy 文件以 `*_eager_v7.sh` / `*_compile_v7.sh` 结尾；GLM-5.2 真实权重文件在模式后保留 `fp8_kvcache_alltoallv_v7`，例如 `GLM-5.2-MXFP4_dp2_tp2_af_ep2_compile_fp8_kvcache_alltoallv_v7.sh`。原无模式文件已替换，选择文件即可确定模式。

需要临时验证 A-only 或 F-only 时，在自定义 preset 中先 source 对应官方文件，再设置 `USER_VLLM_AFD_F_COMPILE=0` 或 `1`。官方 preset 自身会固定该值，因此命令行提前赋值不能覆盖文件的模式。

F 命令行的 `--compile` 由启动脚本自动传入，控制链是：

```text
compile preset：USER_VLLM_AFD_F_COMPILE=1
  → serve_afd_mp_rpc_all_mpi_template.sh 的 F 分支追加 --compile
  → af_ep/moe/__main__.py 解析为 compile_model=True
  → ExpertServiceOptions → ExpertServiceV7 → torch.compile
```

eager preset 设置该变量为 `0`，模板不传 `--compile`，F 的 `compile_model` 默认值为 False。使用 `run_vllm_test.sh -e <preset>` 时无需自行添加 F 参数。A 的模式由 `USER_VLLM_EAGER_OR_NOT` 控制：compile 文件清空该变量，eager 文件设置 `--enforce-eager`。

```bash
source .venv/bin/activate
export QWEN3_30B_A3B_BF16_MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507
export VLLM_MPI_HOSTFILE="$PWD/vllm_scripts/mpi_tools/afd_tp2_ep2.hostfile"
export VLLM_TEST_MAX_WAIT=1000
export VLLM_ENGINE_READY_TIMEOUT_S=900
export USER_VLLM_MAX_NUM_BATCHED_TOKENS=32
export TORCH_XCPU_ENABLE_CHECK=1

cd vllm_scripts
./run_vllm_test.sh \
  -e presets/mpi/moe/Qwen3-30B-A3B_dp1_tp2_af_ep_compile_v7.sh \
  --multi-test --multi-test-temperature 0 --test-timeout 300
```

hostfile、模型路径和 RPC IP 按部署环境修改。对现有跨机矩阵启用 compile：

```bash
./vllm_scripts/e2e/run_afd_crossnode_matrix.sh --mode compile
```

修改编译相关实现后，提前重命名原 compile cache，或在自定义 preset **source 官方 preset 之后**设置独立的 `VLLM_CACHE_ROOT` 和 `TORCHINDUCTOR_CACHE_DIR`。`user_env_template.sh` 会设置这些变量，命令行预先赋值会被覆盖。A/F 可以共享一个 Inductor cache，进程各自维护运行期 tensor 和通信状态。

## 原因与修改

AFD 来自 compile 可用的 MoE alltoallv V6；专家计算继续使用已有 grouped GEMM / fused MoE native 算子。本次适配集中在 A/F 拆分后的校验、通信副作用和独立 F runner。

| 原问题 | 影响 | 修复 |
|---|---|---|
| A compatibility 要求 `enforce_eager`，preset 强制 eager | 无法进入普通 compile 路径 | 允许 A compile，继续拒绝 CUDAGraph capture；分别提供 eager / compile preset |
| V7 forward/fake 校验对 CPU metadata 调用 `.tolist()` | tracing 读取 FakeTensor 数据，产生数据依赖异常 | forward/fake 仅检查 shape、dtype、device；rank 数从 counts tensor 长度推导；初始化和 native endpoint 保留内容校验 |
| F combine-send 无 tensor 输出且无输入 mutation | FX/AOT 的死代码消除可以删除发送节点，A 随后等待返回 | 为 BF16、FP32 combine-send 注册 `EffectType.ORDERED` |
| F 独立服务直接执行 Python receive/compute/send | A compile 不会自动编译 F | F CLI 增加 `--compile`，为每层完整通信与计算事务调用 `torch.compile` |
| F endpoint schema 的 `int layer_idx` 强制特化 | 真实多层模型为每层重编译，第 9 层触发默认上限 | F dispatch-recv / combine-send 的 layer index 改为 `SymInt`；AOTI generator 将其映射到现有 `int64_t` ABI |

### A 的编译路径

A 使用 V2 ModelRunner；平台将 vLLM compile 映射到 `DYNAMO_TRACE_ONCE + Inductor`，与现有 V6 路径一致。模型中的远程专家调用包含 V7 dispatch-send 和 combine-recv。A 不加载 routed expert 权重；路由、shared experts 和 Attention 仍在 A。

修改位置：

- `vllm-xcpu-plugin/vllm_xcpu_plugin/af_ep/attn/compatibility.py`
- `torch_xcpu/torch_xcpu/ops_defs/moe_af_v7.py`
- 八组 AFD V7 eager / compile presets（共十六份文件）

### F 的编译边界

F 使用独立的专家服务，没有 vLLM ModelRunner。`ExpertServiceV7` 以每层事务作为编译单元：

```text
eager：加载本地 EP 权重 → 初始化 packed GEMM backend/workspace → 创建 V7 窗口
每个 model pass：
  eager：session 校验及可选 forward-entry allreduce
  按模型 MoE 层序循环：
    compiled：dispatch-recv → fused-moe-compute → combine-send
  eager：设备 synchronize，限制待执行任务数量
```

`torch.compile(backend="inductor", fullgraph=True, dynamic=True)` 编译 `_execute_layer`。`fullgraph=True` 使无法 tracing 的事务明确报错；layer index 是动态标量，不同层权重与 expert map 作为 tensor 输入。固定容量 buffer 中的真实有效行数由 `num_input_rows_valid` tensor 传给 native MoE 算子，不在 Python 中读取 `.item()`。

仅设置 `dynamic=True` 无法解决 schema 中的 `int` 参数强制特化。实际完整模型测试确认了这个问题，因此 F 两个 endpoint 的 `layer_idx` 声明为 `SymInt`。执行时仍传具体整数，C++ kernel 和 AOTI C ABI 保留 `int64_t`；A 的 endpoint 在模型图中按静态层序展开，保留原 schema。

层循环保留在 host，每次提交一个完整事务，沿同一设备 stream 执行。这样每层的 receive/compute/send 由图内 tensor 依赖连接，跨层 buffer 复用和 MPI collective 顺序由连续调用保证。pass 结束同步继续限制无限服务循环的任务积压。传输初始化、MPI barrier/allreduce 和设备同步不参与图 tracing。

F ModelConfig 中的 `enforce_eager=True` 用于独立权重加载配置；实际 F 执行模式由 `ExpertServiceOptions.compile_model` 与服务的 `torch.compile` 入口控制。

Inductor 配置与当前 MCPU 路径一致，关闭 epilogue fusion、pattern matcher 和 combo kernels。通信及 GEMM 仍通过 stream-aware native 算子提交，避免生成直接读写 MCPU 内存的 host 融合 kernel。

修改位置：

- `vllm-xcpu-plugin/vllm_xcpu_plugin/af_ep/moe/{__main__,runner,service_v7}.py`
- `vllm_scripts/serve/serve_afd_mp_rpc_all_mpi_template.sh`
- `torch_xcpu/torch_xcpu/csrc/moe_ep/moe_af_v7_ops.cpp`
- `torch_xcpu/scripts/generate_aoti_wrappers.py`

### 发送副作用

`moe_af_combine_send_v7_{bf16,fp32}` 修改的是 MPI window，而不是输入 tensor；schema 中的 `-> ()` 不能表达通信副作用。注册 `torch.library._register_effectful_op(..., EffectType.ORDERED)` 后，FX 不会删除发送节点，AOT/Inductor 使用 effect token 保留并排序发送。

dispatch-recv、dispatch-send、combine-recv 已通过 schema 声明 tensor mutation。当前 PyTorch effect wrapper 不支持带 alias/mutation schema 的算子，因此只为无 mutation 的 combine-send 注册 ORDERED；F 图内依赖和 host 层循环共同保证完整事务顺序。

这使用当前开发套件 PyTorch 的内部 effect API。升级 PyTorch 时需要重跑无输出节点保留、Inductor 实际执行和完整模型测试；不能以关闭 DCE 代替副作用声明。

## 构建与回归

本次没有修改 native 算子 ABI、V7 协议或 MPI world 拓扑。修改 `torch_xcpu` Python 注册代码后，仍需更新安装包；只修改源目录不会更新非 editable 安装。插件在当前开发环境中为 editable 安装。

```bash
CXX=mpicxx MAX_JOBS=64 uv pip install --python "$PWD/.venv/bin/python" \
  --no-deps --no-build-isolation ./torch_xcpu

.venv/bin/python -m pytest -q \
  vllm-xcpu-plugin/tests/af_ep tests/test_afd_hostfile.py
```

如果重建 backend，按仓库指南以 memory protection OFF 验证常规功能，依次重建 `torch_mcpu`、`torch_xcpu`、`torch_mpi_ext`。本次沿用已安装的 memory protection OFF、异步 kernel launch 配置。

真实 MPI round-trip 测试可分别编译 A/F 端点：

```bash
TORCH_XCPU_ENABLE_CHECK=1 \
TORCHINDUCTOR_CPP_WRAPPER=1 \
TORCHINDUCTOR_CACHE_DIR=/tmp/afd_compile_roundtrip_cache \
XCPU_AF_TEST_COMPILE_ATTENTION=1 \
XCPU_AF_TEST_COMPILE_EXPERT=1 \
UCX_TLS=tcp,self,sm \
timeout --kill-after=10s 240s \
mpirun --allow-run-as-root --bind-to none \
  --mca osc ucx --mca osc_ucx_tls any --mca osc_ucx_devices any \
  --mca btl self,tcp -np 4 \
  .venv/bin/python torch_xcpu/test/mpi_custom_ops/test_moe_af_v7_roundtrip_mpi.py
```

该测试核对真实 V7 传输和归并数值，覆盖 top-k 6/8、不同输入行数、空目标 F rank 和 dummy A。专家值使用独立参考公式构造；完整 native fused MoE 编译链路由模型 E2E 测试覆盖。

## 验证与范围

A-only 阶段已通过：15 个并发请求（A4/F2 小模型、真实 BF16 A compile、真实 BF16 eager 各 5 个）；真实 BF16 的两种执行模式下，5 个 greedy 请求的可见输出文本全部一致。该比较仅覆盖测试请求，不是全模型精度评估。

F compile 阶段的最终实现验证：

| 用例 | 配置与结果 |
|---|---|
| pytest（compile 适配时） | 16 个用例、30 个子测试通过；包含 BF16/FP32 无输出发送节点保留、跨 12 层图复用和当时四种 A/F 模式选择 |
| 真实 MPI round-trip | A2/F2，A/F 端点同时 Inductor compile，top-k 6/8、行数 3/10/1、dummy A 和空目标 F；数值检查通过 |
| 小模型多 DP 服务 | 两层 Qwen3 MoE dummy 权重，A4/F2、DP2-TP2-EP2，A/F 同时 compile；最终 schema 修复后重新验证，5/5 请求完成，run/cleanup/final exit code 均为 0 |
| 真实 Qwen3-30B-A3B-Instruct-2507 BF16 | A2/F2、DP1-TP2-EP2，A/F 同时 compile；5/5 并发 greedy 请求完成，run/cleanup/final exit code 均为 0 |
| 同请求输出比较 | 每个请求最多生成 16 tokens；上述 5 个请求的可见输出与之前的 eager、A-only compile 全部一致 |
| F 图复用 | 完整模型测试开启 `TORCH_LOGS=recompiles,graph_breaks`；F `_execute_layer` 无重编译记录，无 graph break / cache-limit 错误 |

所有功能测试开启 `TORCH_XCPU_ENABLE_CHECK=1`。环境为本机 x86、PyTorch `2.11.0a0+git58a3a97`、MCPU C++ wrapper、UCX TCP/共享内存。详细记录保存在开发机 `/tmp/afd_f_compile_20260930/results.json`，真实模型日志在该目录的 `real_both_symint/success/`，多 DP 小模型日志在 `tiny_both_symint/success/`，MPI 数值日志为 `mpi_symint_compile.log`。A-only 和 eager 对照记录在 `/tmp/afd_a_compile_20260930/results.json`。这些临时目录是本次开发证据，部署验证请使用各自持久日志目录。

preset 拆分后重新通过 17 个 pytest 用例、21 个子测试，验证十二个文件固定 A/F 模式、成对配置一致，以及矩阵默认 eager / 显式 eager / compile 的文件选择和测试参数透传；非法或缺失模式会在启动前拒绝。日志为 `/tmp/afd_preset_split_20260930.log`。本次拆分未重新运行完整模型 E2E，前述记录对应相同的 compile 实现。

加入 GLM-5.2 真实权重 presets 后，回归更新为按文件名中的 `_eager_` / `_compile_` 判定模式，直接核对 A/F 配置、V2 runner 与 MPI world size；不依赖日志文字或模式必须紧邻 `_v7` 的命名约束。通过 17 个用例、25 个子测试，日志为 `/tmp/afd_glm_preset_update_20260930.log`。两份新增 V6 compile preset 也通过继承 eager 环境的检查，四份新增 V7 preset 显式启用 V2 runner。

适配不扩大原支持范围：要求 V2 runner、BF16 activation、固定 A/F world 和相同模型 MoE 层序；CUDAGraph capture、DBO、PP、context parallel、EPLB、LoRA、speculative decoding、实际 multimodal 数据路径和 shared-expert fusion 继续受原 compatibility 限制。已有 FP8/MXFP4A16 配置已提供独立 compile preset，是否通过 compile 需以对应真实权重实测为准。

本轮本机测试不代替物理跨机、目标 NPU 或完整量化矩阵验收。Inductor 对 mutable workspace 的 functionalization 可能引入 clone/copy-back；本次目标是 compile 功能适配，未测吞吐收益或消除这些拷贝。
