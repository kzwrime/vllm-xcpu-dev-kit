这是一个定制版的 vllm 开发套件。包含几个独立的 git 开发仓库

包括 

- vllm (vllm 推理引擎本体)
- torch_mcpu (torch mcpu backend using privateuse1)
- torch_xcpu (算子库，包括 .claude/.codex skills)
- ./hide/torch_mpi_ext (通用通信库)
- vllm-xcpu-plugin (遵循 vllm 扩展插件体系开发的插件，用于连接 vllm 和 torch_mcpu/torch_xcpu/torch_mpi_ext，提供平台特定功能)。

出于开发和调试便利性，这个 vllm 开发套件主要在 x86 CPU 平台开发，但事实上是为其他 CPU 架构服务。
