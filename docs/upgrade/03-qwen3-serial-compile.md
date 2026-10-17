# Qwen3-0.6B 串行编译推理

验证 MCPU/XCPU 下 DP/TP/PP=1 的 attention direct call 编译路径，覆盖 MRV2 和 MRV1。环境构建见 [AGENTS.md](../../AGENTS.md)，先完成 [eager 推理检查](01-qwen3-serial-eager.md)。

## 执行

每个 runner 使用独立冷缓存，顺序运行并等待清理结束。开始前将已有 `vllm_scripts/vllm_cache`、`vllm_scripts/torch_compile_cache_opt` 重命名备份；切换 runner 前也保留前一次产物并腾空这两个路径，避免误用旧产物。

```bash
source .venv/bin/activate
cd vllm_scripts

# 预设默认 batched tokens=32、启动超时=1800 秒、模型执行超时=3600 秒。
VLLM_USE_V2_MODEL_RUNNER=1 ./run_vllm_test.sh \
  -e presets/serial/Qwen3-0.6B_dp1_tp1_compile.sh --launcher mp \
  --multi-test --multi-test-max-tokens 96 --multi-test-temperature 0 \
  --test-timeout 900

# 备份并腾空两处编译缓存后运行 MRV1。
VLLM_USE_V2_MODEL_RUNNER=0 ./run_vllm_test.sh \
  -e presets/serial/Qwen3-0.6B_dp1_tp1_compile.sh --launcher mp \
  --multi-test --multi-test-max-tokens 96 --multi-test-temperature 0 \
  --test-timeout 900
```

## 通过标准

- 日志确认目标 runner 和 Inductor 编译启用，完成实际编译并服务请求，无编译异常或运行时异常。
- 五路流式请求均正常完成，正文非空、可读、与各自提示词相关；不能只看 HTTP 状态码。`run_result.json` 中测试、清理和总退出码均为 0，结束后没有本次服务进程残留。
- 本次冷缓存生成非空 C++ 源文件和可加载的共享库，缓存中存在对应的编译模型。实际请求执行已生成的模型。
- C++ 模型计算部分包含直接 `aoti_torch_mcpu_*` 调用；核对共享库的动态符号与链接依赖，确认后端 shim 可解析。
- 计算路径没有 `custom_op_wrapper`、`PyObject_CallObject` 等 Python 算子回调，也没有调用 `torch.ops` 的 Python 计算 fallback。Python 加载器、调试 FX 图、pybind 模块初始化，以及纯 C++ dispatcher 调用本身不等于 Python fallback，需结合所在文件和调用位置判断。
- 核查 attention、linear、norm 等主路径确实保留在编译产物中；缺少 attention metadata 的预运行走 eager，不应捕获成缺失 attention 的正式模型图。

本项验证当前模型、runner 和后端组合的编译执行，不替代精度评测或其他设备验证。`env_override.py` 中针对特定 PyTorch 版本的 patch，应另外记录是否在当前环境实际启用，不能仅凭端到端通过判定所有版本分支已验证。
