# Qwen3-0.6B 串行 eager 推理

验证 MCPU/XCPU 后端下，Qwen3-0.6B 在 DP/TP/PP=1、eager 模式的模型加载、prefill、decode 和并发流式输出。

## 准备

在工作区 `.venv` 中安装当前源码的 vLLM、插件和后端扩展，准备好 `Qwen/Qwen3-0.6B` 的完整权重、配置及 tokenizer 缓存。

## 执行

从开发套件根目录运行，普通测试结束并完成清理后再执行多流测试：

```bash
source .venv/bin/activate
cd vllm_scripts

# 普通请求
./run_vllm_test.sh -e presets/serial/Qwen3-0.6B_dp1_tp1_eager.sh

# 5 路并发流式请求
./run_vllm_test.sh -e presets/serial/Qwen3-0.6B_dp1_tp1_eager.sh --multi-test
```

## 通过标准

- 服务加载正确模型并就绪，使用 MCPU/XCPU 后端，日志中无运行异常。
- 普通请求返回 HTTP 200，`choices[0].message.content` 非空、可读且与提示词相关，`usage.completion_tokens > 0`。
- 多流测试的 5 个任务均正常结束，有流式 chunk，正文非空、可读且与各自提示词相关。
- 两次测试及清理退出码均为 0，结束后无相关服务进程残留。

日志位于启动时打印的 `logs/runs/<运行目录>/`：普通响应看 `test.log`，多流输出看 `vllm_task_*.log`；`run_result.json` 中的 `run_exit_code`、`cleanup_exit_code`、`exit_code` 均应为 0。退出码为 0 后仍需检查正文。

默认最多生成 16 token，`finish_reason=length`、句子未结束属正常现象；采样输出不要求逐字一致。本项通过仅表示上述推理功能正常，精度与性能另行验证。
