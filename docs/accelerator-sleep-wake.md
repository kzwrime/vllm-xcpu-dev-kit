# 加速器释放 / 重载（vLLM sleep mode 接入）

上层 LLM 路由可以显式要求某个 vLLM 实例释放 NPU（供查验、借用），之后再要求它重新加载。本文描述这套机制在 vLLM、vllm-xcpu-plugin、torch_xcpu 三层的接法，以及 NPU 团队需要实现的接口。

## 1. 对路由方的接口

实例启动参数：`--enable-sleep-mode`，并设置 `VLLM_SERVER_DEV_MODE=1`（端点在 dev router 下）。

| 动作 | HTTP | 说明 |
|---|---|---|
| 释放加速器 | `POST /sleep?level=1&mode=wait` | 暂停调度、清 prefix cache，向所有 TP/DP rank 广播 `worker.sleep`。`mode=wait` 等在途请求结束再释放，`abort` 立即中止在途请求。HTTP 返回即释放完成 |
| 重新加载 | `POST /wake_up` | 广播 `worker.wake_up`，全部成功后恢复调度。返回 `{"status": "awake"}` |
| 查询 | `GET /is_sleeping` | 调度器暂停或 executor 处于 sleep 即为 true |
| 仅静止不释放 | `POST /sleep?level=0` | 只暂停调度 + 同步设备，不调用 torch_xcpu |

语义保证（0.31 代码核对结果）：

- vLLM 不会自动 sleep，也不会自动 wake_up；两者只由上述 HTTP 调用触发。
- sleep 期间到达的请求进入 waiting 队列排队，不会执行、不会唤醒、不会报错，直到 `/wake_up`。
- `/sleep` 和 `/wake_up` 均幂等，重复调用只打 warning。
- `/wake_up` 失败（例如 NPU 仍被他用）时返回 HTTP 500，executor 与调度器仍处于 sleep 状态，可以重试。
- DP>1 时 level 0 不会阻止本 rank 为其他 rank 跑 dummy batch；需要设备静止时用 level≥1。

## 2. 各层实现

- `torch_xcpu_impl/include/device_runtime.h`、`src/device_runtime.cpp`：**NPU 团队实现点**。`xcpu_impl::device_runtime_sleep(level)` / `device_runtime_wake_up()`，x86 模拟器为空实现。
- `torch_xcpu/csrc/runtime_context.cpp`：持有状态与互斥锁，拒绝重复 sleep / 重复 wake_up、未初始化即 sleep；wake_up 抛异常时保持 sleep 状态。Python 侧 `torch_xcpu.device_runtime_sleep(level)`、`device_runtime_wake_up()`、`is_device_runtime_sleeping()`。
- `vllm_xcpu_plugin/sleep_mode.py`：`XcpuSleepModeBackend`，按 vLLM 的 `SleepModeBackend` 抽象注册为 `"xcpu"`。`McpuPlatform.is_sleep_mode_available()` 返回 True，`check_and_update_config` 在开启 sleep mode 时把默认的 `cumem` 换成 `xcpu`。
- `vllm_xcpu_plugin/worker/worker_v1.py`：`McpuWorker.sleep/wake_up` 只做"同步设备 → 调后端 → 同步设备"，不卸载权重、不丢 KV（设备内存全程原位有效）。`release_kv_cache_memory` 不支持。

## 3. NPU 实现契约

- `device_runtime_sleep` 在所有 stream 同步完成后被调用。可以释放计算上下文（PE 团队、stream、DMA、驱动句柄），**不得**释放或搬移设备内存：进程持有的每个设备指针在 wake_up 后必须仍然映射且内容不变。
- `device_runtime_wake_up` 恢复上一次 sleep 释放的全部资源，之后内核可在同样的 stream、同样的指针上继续运行。
- 两者进程级、每 rank 调一次。torch_xcpu 层已做串行化和状态校验，实现无需自己记状态。
- 失败以异常上报（`c10::Error` / `std::runtime_error`）。wake_up 失败时设备必须保持完全释放，以便路由方稍后重试。
- 若驱动把设备内存绑定在 context 生命周期上（释放 context 内存即失效），则本契约无法满足，需要回到 `docs` 中分析过的"卸载权重 + 丢 KV + 重建"路线，请尽早反馈。

## 4. 测试

```
# torch_xcpu 状态机
cd torch_xcpu && python -m pytest test/test_device_runtime_sleep.py -v
# 插件后端
cd vllm-xcpu-plugin && python -m pytest tests/plugin_tests/test_sleep_mode.py -v
# 端到端：greedy 输出 → sleep → wake_up → 同 prompt 输出逐 token 一致，循环两轮
cd vllm_scripts && ./run_vllm_test.sh -e presets/serial/Qwen3-0.6B_dp1_tp1_eager_sleep.sh --sleep-test
```
