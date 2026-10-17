# 升级功能验证指南

每次升级后，按功能执行测试并核对通过标准。指南只保留必要准备、执行命令和通过标准，不绑定版本或日期，不收录运行结果、排错过程和日志归档。

- [Qwen3-0.6B 串行 eager 推理：普通请求与多流请求](01-qwen3-serial-eager.md)
- [Profiler：采集 trace 与算子耗时](02-profiler.md)
- [Qwen3-0.6B 串行编译：MRV2、MRV1 与 AOTI 产物](03-qwen3-serial-compile.md)
- [Qwen3-0.6B Isolated DP：API/head 与 DP master 地址分离](04-qwen3-isolate-dp.md)
- [Qwen3-0.6B MP RPC 与 MPI：跨节点推理、版本兼容与退出](05-qwen3-mp-rpc-mpi.md)
- [Qwen3.5-0.8B eager：serial、TP2 与 GDN 状态语义](06-qwen35-eager.md)
- [Qwen3.5-0.8B compile：serial、TP2 与 MRV2 采样语义](07-qwen35-compile.md)
- [Qwen3-MoE：MPI v5/v6 eager 与 compile](08-qwen3-moe.md)
- [GLM5.2：FP8 / MXFP4 串行 eager 与 compile](09-glm52-serial.md)
- [GLM5.2：DP2/TP2 MoE 序列并行（SP-MoE）服务验证](10-glm52-sp-moe-dp2tp2.md)

环境构建与测试约定见 [AGENTS.md](../../AGENTS.md)。
