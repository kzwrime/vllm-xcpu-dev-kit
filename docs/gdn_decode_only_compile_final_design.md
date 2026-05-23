# Qwen3.5 GDN decode-only compile 最终设计

## 目标范围

本文描述当前定制 vLLM 0.19.0 XCPU 开发套件中，Qwen3.5 GDN
decode-only 路径去除 Python fallback 的最终设计。

本方案是一个窄作用域的 0.19.0 专用实现：

- 目标模型：Qwen3.5 系列中使用 GDN attention 的层；
- 目标阶段：非 speculative 的 decode-only；
- prefill 和 mixed prefill/decode 不进入本优化路径；
- 通过 `VLLM_XCPU_GDN_DECODE_ONLY_COMPILE=1` 显式开启；
- vLLM 版本升级后必须重新审视 metadata、KV cache 生命周期和 compile 入口。

## 核心问题

原始 GDN core 通过 `torch.ops.vllm.gdn_attention_core` 调用。这个 op
在当前编译链路中会形成 Python callback 边界，导致 AOTI 产物不能彻底消除
GDN core 的 Python fallback。

Qwen3.5 的 decode-only 场景本身已经会进入一个较紧凑的 Python 实现：
`_forward_core_decode_non_spec`。最终设计保留 vLLM 的模型结构、metadata
构建和 KV cache 绑定方式，只把这个 decode core 的主体移入一个常规
`torch_xcpu` C++ op，使 compile 产物直接调用 C++。

## 总体执行路径

prefill 或 mixed prefill/decode：

- `vllm/compilation/decorators.py` 在 compiled model 入口检查当前 GDN
  metadata；
- 如果当前 forward 不是 decode-only，则绕过本次 compile；
- GDN 层继续走原始 `torch.ops.vllm.gdn_attention_core` 路径。

decode-only：

- compiled model 允许进入正常 compile 路径；
- `GatedDeltaNetAttention.forward()` 判断当前 GDN metadata 满足 decode-only
  条件；
- GDN core 调用 `torch.ops.torch_xcpu.gdn_decode_non_spec_core`；
- AOTI 产物中出现的是
  `aoti_torch_mcpu_gdn_decode_non_spec_core(...)`，不再出现 Python callback。

## 开关

环境变量：

```bash
VLLM_XCPU_GDN_DECODE_ONLY_COMPILE=1
```

默认关闭。Qwen3.5 compile preset 中显式开启：

```text
vllm_scripts/presets/serial/Qwen3.5-0.8B_dp1_tp1_compile.sh
```

关闭时，vLLM 使用原始 GDN 路径。

## decode-only 判定

只有同时满足以下条件，GDN 层才进入优化路径：

- `VLLM_XCPU_GDN_DECODE_ONLY_COMPILE=1`；
- `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1`；
- `self.enable_packed_recurrent_decode` 为真；
- `attn_metadata_i.spec_sequence_masks is None`；
- `attn_metadata_i.num_prefills == 0`；
- `attn_metadata_i.num_decodes > 0`。

这对应 `_forward_core_decode_non_spec` 的语义范围。

## KV cache 传递方式

vLLM 原本会把 Mamba KV cache 绑定到 layer：

```python
self.kv_cache = [conv_state, ssm_state]
```

终版设计沿用这条原始路径：

- 不修改 `kv_cache` list；
- 不向 `kv_cache` 追加额外 metadata；
- 不在 forward 中传递 shape、dtype 或 layout 描述；
- 不在 C++ 中重新推导底层 KV cache 布局。

在 `bind_kv_cache()` 阶段，如果发现当前 layer 是 GDN attention，且
`kv_cache` 正好是 `[conv_state, ssm_state]`，则额外把这两个 typed tensor
注册到 `torch_xcpu` 的进程内 registry：

```python
handle = next(_XCPU_GDN_DECODE_STATE_HANDLE_COUNTER)
torch.ops.torch_xcpu.register_gdn_decode_state(
    handle, kv_cache[0], kv_cache[1]
)
layer._xcpu_gdn_decode_state_handle_019 = handle
```

这里的 handle 是进程内单调递增 token。当前 vLLM 0.19.0 生命周期中，
Mamba KV cache 在分配和绑定后保持稳定；重新 bind 时会先注册新 handle，
成功后更新 layer 属性，再注销 layer 上保存的旧 handle。属性名中的 `_019`
是有意保留的版本提示。

## torch_xcpu 状态 registry

`torch_xcpu` 保存的是 vLLM 已经构建好的 typed KV cache view：

```cpp
struct GdnDecodeStateEntry {
  at::Tensor conv_state;
  at::Tensor ssm_state;
};
```

注册接口：

```cpp
void register_gdn_decode_state(
    int64_t handle,
    const at::Tensor& conv_state,
    const at::Tensor& ssm_state);
```

生命周期接口：

```cpp
void unregister_gdn_decode_state(int64_t handle);
void clear_gdn_decode_states();
```

vLLM 重新执行 `bind_kv_cache()` 时会对 layer 的旧 handle 调用
`unregister_gdn_decode_state`。`clear_gdn_decode_states` 保留给测试或进程内
显式 teardown 使用。

decode core 通过 handle 取回 state：

```cpp
auto state_entry = get_gdn_decode_state_xcpu(state_handle);
auto conv_state = state_entry.conv_state.transpose(-1, -2);
auto ssm_state = state_entry.ssm_state;
```

这与 Python 原实现一致：

- `conv_state` 来自 `self.kv_cache[0].transpose(-1, -2)`；
- `ssm_state` 来自 `self.kv_cache[1]`。

## GDN decode core op

decode-only 编译路径调用：

```python
torch.ops.torch_xcpu.gdn_decode_non_spec_core(
    mixed_qkv,
    b,
    a,
    core_attn_out,
    self.A_log,
    self.dt_bias,
    conv_weights,
    self.conv1d.bias,
    non_spec_state_indices_tensor,
    num_actual_tokens_tensor,
    self.head_k_dim**-0.5,
    state_handle,
    self.activation in ("silu", "swish"),
    True,
)
```

其中：

- `non_spec_state_indices_tensor` 来自 GDN metadata；
- `num_actual_tokens_tensor` 使用 `non_spec_query_start_loc[-1:]`；
- `state_handle` 来自 `bind_kv_cache()` 时注册的 KV cache state。

C++ op 主体对应 Python `_forward_core_decode_non_spec`：

1. 读取 `num_actual_tokens`。
2. 使用 `TORCH_CHECK` 校验入口 tensor 的 defined/device/rank、`state_indices`
   的 int32 dtype、关键数据 tensor dtype 一致性，以及 registry 中 state 的
   基本 rank/device。
3. 校验 `num_actual_tokens` 非负，且不超过相关 per-token tensor 的第 0 维。
4. 按 Python 语义截取真实 token 前缀：
   `mixed_qkv[:n]`、`a[:n]`、`b[:n]`、`state_indices[:n]`、
   `core_attn_out[:n]`。
5. 调用 `causal_conv1d_update_cpu`，更新 conv state 并生成 conv 输出。
6. 调用 `fused_recurrent_gated_delta_rule_packed_decode_cpu`，更新 SSM state
   并写入 `core_attn_out[:n]`。

`num_actual_tokens` 不做 clamp。它是 vLLM metadata 给出的语义输入；如果
metadata 不满足预期，应通过 `TORCH_CHECK` 显式失败，而不是静默修正。

C++ 入口检查复用 `torch_xcpu` 现有开关：

```bash
TORCH_XCPU_ENABLE_CHECK=0
```

`torch_xcpu` 在首次调用检查函数时读取一次该环境变量并缓存为进程级状态。
默认开启检查；设置为 `0` 后关闭 GDN decode 组合 op 的额外入口检查。

`core_attn_out` 只有前 `num_actual_tokens` 个位置有语义。由于 compile/cudagraph
会保留 padded slot，后续逻辑不得消费 padded 部分的语义；这些位置保持 forward
中初始化的 zero。

## 为什么能消除 Python fallback

decode-only 下，Python forward 只负责：

- 读取当前 GDN metadata；
- 从 layer 上读取已注册的 state handle；
- 准备普通 tensor 参数；
- 调用一次 `torch_xcpu.gdn_decode_non_spec_core`。

会修改 KV cache 的主体逻辑已经在 `torch_xcpu` C++ op 内：

- 更新 `conv_state`；
- 更新 `ssm_state`；
- 写入 `core_attn_out`。

因此 AOTI 可以把这部分降为直接 C++ 调用：

```text
aoti_torch_mcpu_gdn_decode_non_spec_core(...)
```

state 注册发生在 `bind_kv_cache()` 阶段，不属于 compiled forward graph。
compile 产物中不应该出现 `register_gdn_decode_state`。

## 正确性假设

本方案依赖以下假设：

- decode-only metadata 判定准确；
- `non_spec_query_start_loc[-1:]` 是当前 step 的真实 token 数；
- `non_spec_state_indices_tensor` 可以为 compile/cudagraph 保持 padded shape，
  但只有前 `num_actual_tokens` 个 entry 具有语义；
- vLLM 0.19.0 中 Mamba KV cache 在 `bind_kv_cache()` 后生命周期稳定；
- 每个 GDN layer 保存的单调 handle 与当前已注册 KV cache state 对应。
- 正常 worker shutdown 会调用 `clear_gdn_decode_states()` 释放 registry 中对
  tensor 的强引用；异常退出或解释器强制终止时依赖进程生命周期回收。

这些假设如果不成立，应该触发显式错误，而不是继续产生静默错误结果。

## 验证方式

构建 `torch_xcpu` 必须使用：

```bash
cd torch_xcpu
./build-all.sh
```

正确性测试：

```bash
cd vllm_scripts
./run_vllm_test.sh -e presets/serial/Qwen3.5-0.8B_dp1_tp1_compile.sh
```

bench 后正确性检查：

```bash
cd vllm_scripts
./run_vllm_test.sh -e presets/serial/Qwen3.5-0.8B_dp1_tp1_compile.sh --bench
```

compile 产物应满足：

- 不出现 `gdn_attention_core`；
- 不出现 `custom_op_wrapper`；
- 不出现 `python_kernel` 或 `python_call`；
- forward graph 中不出现 `register_gdn_decode_state`；
- GDN decode core 以 `aoti_torch_mcpu_gdn_decode_non_spec_core` 形式出现。

已验证：

- `torch_xcpu/build-all.sh` 通过；
- Qwen3.5 compile 正确性测试通过；
- bench 后请求成功数为 1，失败数为 0；
- bench 后最终回答正常，没有重复标点退化。
