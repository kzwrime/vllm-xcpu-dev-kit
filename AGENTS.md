# 概述

这是一个定制版的 vllm 开发套件。包含几个独立的 git 开发仓库

包括 

- vllm (vllm 推理引擎本体)
- torch_mcpu (torch mcpu backend using privateuse1)。编译方式 cd torch_mcpu && ./build.sh
- torch_xcpu (算子库，包括 .claude/.codex skills)。编译方式 cd torch_xcpu && ./build-all.sh
- torch_mpi_ext (通用通信库)
- vllm-xcpu-plugin (遵循 vllm 扩展插件体系开发的插件，用于连接 vllm 和 torch_mcpu/torch_xcpu/torch_mpi_ext，提供平台特定功能)。
- vllm-xcpu-dev-kit (本仓库，包含启动设置、测试程序、开发套件等)

出于开发和调试便利性，这个 vllm 开发套件主要在 x86 CPU 平台开发，但事实上是为其他 NPU 架构服务。主路径上的性能敏感算子应该编写在 torch_xcpu 中，NPU 算子团队会基于其 x86 实现编写和替换。

# 端到端测试方法

举例如下

```
cd vllm_scripts
# 典型可选参数
# 1. 减少 BATCHED_TOKENS 以降低启动预热时间，默认 256，长序列测试应当适当调大
# export USER_VLLM_MAX_NUM_BATCHED_TOKENS=32
# 2. 提高测试启动时间阈值，以进行较为复杂的端到端测试（如 MoE + Compile, 大型模型），默认 300
# export VLLM_TEST_MAX_WAIT=1000
# 3. 测试多流并发
# 附加 --multi-test 启动参数

# 最简测试
./run_vllm_test.sh -e presets/serial/Qwen3-0.6B_dp1_tp1_eager.sh
# Compile 最简测试
./run_vllm_test.sh -e presets/serial/Qwen3-0.6B_dp1_tp1_compile.sh
# Qwen3.5 最简测试
vllm_scripts/presets/serial/Qwen3.5-0.8B_dp1_tp1_eager.sh
# Qwen3.6 量化 + EP 测试
./run_vllm_test.sh -e presets/mpi/moe/Qwen3.6-35B-A3B-FP8_dp2_tp2_ep_eager_alltoallv_v6.sh
# 预测解码
./run_vllm_test.sh -e presets/serial/dflash/Qwen3.5-4B_dp1_tp1_eager_dflash_4.sh --multi-test
```

日志一般在 ./vllm_scripts/logs

# torch_mcpu 内存保护

常规测试、测试性能相关问题时，应当修改 torch_mcpu 的 build.sh，添加 export TORCH_MCPU_ENABLE_MEMORY_PROTECTION="OFF"，然后重新编译。以提高运行速度。然后重编其他库。

```
cd torch_xcpu && ./clean-all.sh && ./build-all.sh
cd torch_mpi_ext && rm build -rf && ./build.sh
```

进行不确定的内存错误等 Debug 时，可以设置该选项为 ON，以确定没有越界访存、Launch 区间外的非法访存等。
