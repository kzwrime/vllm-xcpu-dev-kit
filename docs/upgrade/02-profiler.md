# Profiler 采集

使用 [Qwen3 串行 eager 预设](01-qwen3-serial-eager.md) 验证 profiler 启停、trace 导出和算子耗时统计。环境准备同推理测试。

## 执行

从开发套件根目录运行：

```bash
source .venv/bin/activate
cd vllm_scripts
./run_vllm_test.sh -e presets/serial/Qwen3-0.6B_dp1_tp1_eager.sh --bench
```

`--bench` 内置 profiler 启停，无需再加 `--profile`。默认使用 1 条随机请求，输入 16 token、最多输出 8 token；停止采集后另发一条中文请求检查输出。

## 通过标准

- `bench.log` 显示 profiler 启动、停止成功，请求成功数为 1、失败数为 0，输出 token 数大于 0。
- 采集后的中文请求正常返回，正文非空、可读且与提示词相关。
- 服务日志无 profiler 异常，测试及清理退出码均为 0。
- 配置的 `torch_profiler_dir`（默认 `vllm_scripts/vllm_profile`）生成本次 worker 的非空 `*.pt.trace.json.gz` 和 `*profiler_out.txt`。
- trace 解压后是合法 JSON，`traceEvents` 中包含模型执行、CPU 算子和 `mcpu_kernel` 事件；主要算子有正耗时，不能只有 API/调度事件。
- worker 统计表包含模型主要算子的非零设备耗时，内容与 trace 中的执行路径相符。
- vLLM cherry-pick 移植了 [Linear] Use torch_xcpu.ops.linear as default_unquantized_gemm or cpu_unquantized_gemm，且 txt 中能看见

用 Perfetto 打开 worker trace，检查 linear、attention、norm 等算子及连续 decode 步骤。API 进程 trace 用于查看请求处理，不能代替 worker trace 验收。只检查本次运行生成的文件，避免把旧产物误认为本次结果。

该短负载用于验证采集功能；随机文本不作语义验收，开启 profiler 后的耗时不作推理性能基线。
