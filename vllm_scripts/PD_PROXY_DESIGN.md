# P/D Proxy Wrapper Design

本文说明当前 P/D 分离 wrapper 的实现原理。它支持两类 KV backend：

- `example`：默认路径，使用 vLLM `ExampleConnector` 和共享存储目录。
- `mooncake`：实验路径，使用 vLLM `MooncakeConnector` 和 Mooncake
  non-CUDA/TCP transfer engine。

相关文件：

- `vllm_scripts/pd_launcher.sh`：P/D 拓扑启动器。
- `vllm_scripts/serve_test/pd_proxy.py`：本地 OpenAI API proxy。
- `vllm_scripts/RUNBOOK.md`：运行和调试入口说明。

## 总体结构

当前实现不是单个 vLLM 服务，也不是完整的生产级 P/D scheduler，而是由多个普通 vLLM OpenAI API 服务和一个本地 proxy 组成：

```text
client
  |
  v
local P/D proxy: http://127.0.0.1:<proxy-port>
  |
  +--> prefill vLLM instance(s)
  |
  +--> decode  vLLM instance(s)
```

ExampleConnector 模式下，prefill/decode 共同使用
`logs/pd/<run-id>/example_connector_storage/`。
Mooncake 模式下没有共享目录，prefill 侧额外启动 Mooncake bootstrap
server，decode 侧通过 bootstrap 查询 producer worker 地址，再触发 TCP KV
transfer。

proxy 对客户端暴露普通 OpenAI API；内部把一次生成请求拆成 prefill 和
decode 两次请求。不同 backend 的请求语义不同：

- `example` 模式：prefill 先生成 1 个 token；proxy 把这个首 token 接到
  prompt/messages 后再请求 decode；最终把 prefill 首 token 和 decode 输出拼接。
- `mooncake` 模式：prefill 请求只负责生产 KV，decode 请求通过
  `kv_transfer_params` 从 prefill 侧拉取 KV；proxy 不把 prefill 首 token 拼回响应。

## Backend 选择

默认 backend 是 `example`：

```bash
./run_vllm_test.sh --pd -e <preset.sh>
```

Mooncake backend 通过环境变量开启：

```bash
USER_VLLM_PD_KV_BACKEND=mooncake \
USER_VLLM_PD_MOONCAKE_PROTOCOL=tcp \
./run_vllm_test.sh --pd -e <preset.sh>
```

相关可调变量：

```text
USER_VLLM_PD_KV_BACKEND=example|mooncake
USER_VLLM_PD_MOONCAKE_PROTOCOL=tcp
USER_VLLM_PD_MOONCAKE_NUM_WORKERS=4
USER_VLLM_PD_PREFILL_BOOTSTRAP_PORT=18998
```

当前已验证：

```bash
USER_VLLM_PD_KV_BACKEND=mooncake USER_VLLM_PD_MOONCAKE_PROTOCOL=tcp \
./run_vllm_test.sh --pd -e presets/serial/Qwen3-0.6B_dp1_tp1_eager.sh --launcher mp
```

```bash
USER_VLLM_PD_KV_BACKEND=mooncake USER_VLLM_PD_MOONCAKE_PROTOCOL=tcp \
./run_vllm_test.sh --pd -e presets/serial/Qwen3-0.6B_dp1_tp1_eager.sh --launcher mp --multi-test
```

```bash
USER_VLLM_PD_KV_BACKEND=mooncake USER_VLLM_PD_MOONCAKE_PROTOCOL=tcp \
./run_vllm_test.sh --pd -e presets/mpi/moe/Qwen3-30B-A3B_dp2_tp2_ep_eager_alltoallv_v2.sh
```

## ExampleConnector 模式

ExampleConnector 模式把一次生成请求拆成两次请求：

1. 先发给 prefill 实例，强制 `max_tokens=1`，得到首 token。
2. 再发给 decode 实例，把首 token 接回上下文，继续生成剩余 token。
3. 最后 proxy 把首 token 和 decode 输出拼成一个完整响应返回客户端。

prefill 和 decode 之间的 KV cache 交换依赖 vLLM 的 `ExampleConnector`，两侧配置同一个共享存储目录。

## 启动流程

`pd_launcher.sh` 中的 `pd_start()` 会创建 P/D 工作目录：

```text
logs/pd/<run-id>/
  prefill0.sh
  decode0.sh
  proxy.sh
  prefill0/
  decode0/
  proxy/
  example_connector_storage/
```

其中：

- `prefill0.sh`、`decode0.sh` 是基于原始 preset 生成的 overlay preset。
- `proxy.sh` 是给测试客户端使用的 proxy preset。
- `example_connector_storage/` 仅在 `example` backend 中作为 prefill 和
  decode 共用的 KV 交换目录。

每个 prefill/decode overlay preset 都会：

```bash
source "$ORIG_CONFIG_FILE"

export USER_VLLM_PORT=<role-http-port>
export USER_VLLM_DATA_PARALLEL_RPC_PORT=<role-dp-rpc-port>
export VLLM_MPI_COORD_PORT=<role-mpi-coord-port>
export VLLM_MPI_ENV_EXPORT_FILE="<role-dir>/vllm_mpi_env_server.sh"
export VLLM_MP_RPC_READY_PORT_BASE=<role-ready-port-base>
```

并追加关键 vLLM 参数：

ExampleConnector：

```bash
_VLLM_PD_KV_TRANSFER_CONFIG='{"kv_connector":"ExampleConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"<storage-dir>"}}'

export VLLM_OPTIONAL_ARGS="${VLLM_OPTIONAL_ARGS} \
  --kv-transfer-config ${_VLLM_PD_KV_TRANSFER_CONFIG} \
  --no-disable-hybrid-kv-cache-manager"
```

这里真正决定 P/D 共享 KV 的配置是 `--kv-transfer-config` 中的 `shared_storage_path`。`VLLM_PD_ROLE`、`VLLM_PD_ROLE_INDEX` 这类变量没有实际消费者，不能作为功能依赖。

Mooncake：

```bash
# prefill role
export VLLM_MOONCAKE_BOOTSTRAP_PORT=<prefill-bootstrap-port>
_VLLM_PD_KV_TRANSFER_CONFIG='{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"tcp","num_workers":4}}'

# decode role
_VLLM_PD_KV_TRANSFER_CONFIG='{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"tcp","num_workers":4}}'

export VLLM_OPTIONAL_ARGS="${VLLM_OPTIONAL_ARGS} \
  --kv-transfer-config ${_VLLM_PD_KV_TRANSFER_CONFIG} \
  --no-disable-hybrid-kv-cache-manager"
```

## 端口和实例

`pd_launcher.sh` 会分别给 prefill、decode、proxy 分配端口：

- prefill HTTP 端口默认从 `USER_VLLM_PORT + 100` 附近寻找。
- decode HTTP 端口默认从 `USER_VLLM_PORT + 200` 附近寻找。
- proxy 默认使用 `USER_VLLM_PORT`，如果被占用则寻找空闲端口。
- DP RPC、MPI coord、MP RPC ready port 也会分别分配，避免 prefill/decode 冲突。
- Mooncake 模式额外为每个 prefill role 分配 bootstrap 端口，默认从
  `18998` 开始，实例间 stride 为 100。

默认拓扑是 `1P x 1D`。可以通过环境变量扩展：

```bash
USER_VLLM_PD_PREFILL_COUNT=2 USER_VLLM_PD_DECODE_COUNT=1 \
./run_vllm_test.sh --pd --multi-test -e <preset.sh>
```

多实例场景下，proxy 接收逗号分隔 URL：

```bash
python ./serve_test/pd_proxy.py \
  --mode example \
  --port 14800 \
  --prefill-url http://127.0.0.1:14900,http://127.0.0.1:15000 \
  --decode-url http://127.0.0.1:15100
```

Mooncake 模式需要传入与 prefill URL 一一对应的 bootstrap 端口：

```bash
python ./serve_test/pd_proxy.py \
  --mode mooncake \
  --port 14800 \
  --prefill-url http://127.0.0.1:14900,http://127.0.0.1:15000 \
  --prefill-bootstrap-port 18998,19098 \
  --decode-url http://127.0.0.1:15100
```

proxy 对 prefill URL 和 decode URL 分别做简单 round-robin。

## Proxy 请求处理

`pd_proxy.py` 只处理以下接口：

```text
GET  /v1/models
POST /v1/completions
POST /v1/chat/completions
```

`GET /v1/models` 会直接转发到第一个 decode 实例。

生成请求由 `_handle_generation()` 处理。如果 `--mode mooncake`，会转入
`_handle_mooncake_generation()`；否则走默认 ExampleConnector 流程。

## ExampleConnector 请求处理

ExampleConnector 核心流程如下：

```text
用户请求:
  prompt/messages, max_tokens=N, stream=<原始值>

proxy -> prefill:
  prompt/messages, max_tokens=1, stream=false

prefill:
  执行 prefill
  生成第 1 个 token
  通过 ExampleConnector 写出 KV cache 到共享存储
  返回 first_text

proxy -> decode:
  原始上下文 + first_text, max_tokens=N-1, stream=<原始值>

decode:
  通过 ExampleConnector 从共享存储读取 KV cache
  如果命中，日志中应出现 External Cache Hit
  继续生成剩余 token

proxy -> 用户:
  first_text + decode_text
```

### Completion 请求

对 `/v1/completions`，proxy 会把 prefill 返回的 `first_text` 拼到原始 `prompt` 后：

```text
decode_prompt = original_prompt + first_text
```

如果 `prompt` 是字符串列表，则对每个字符串元素追加 `first_text`。

### Chat Completion 请求

对 `/v1/chat/completions`，proxy 会在原始 messages 后追加一条 assistant 消息：

```json
{"role": "assistant", "content": "<first_text>"}
```

然后由 decode 实例继续生成。

### 非流式响应

非流式场景下：

1. proxy 等 prefill 返回 JSON。
2. proxy 等 decode 返回 JSON。
3. proxy 把 `first_text` prepend 到 decode 响应的首个 choice。
4. 同时把 usage 中的 `completion_tokens` 和 `total_tokens` 各加 1。

### 流式响应

流式场景下：

1. prefill 仍然是非流式请求。
2. proxy 先自己构造一个 SSE chunk，把 `first_text` 发给客户端。
3. 然后把 decode 实例返回的 SSE 数据原样转发给客户端。

## KV 共享和命中验证

prefill 和 decode 都带同一个 `ExampleConnector` 配置：

```json
{
  "kv_connector": "ExampleConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "shared_storage_path": "logs/pd/<run-id>/example_connector_storage"
  }
}
```

因此 prefill 请求写出的 KV cache 可以被 decode 请求读取。是否真正命中外部 KV cache，不由 proxy 判断，而应查看 decode 侧 vLLM 日志。

启动器中的 `pd_validate_transfer()` 会检查：

```text
External Cache Hit
```

如果 decode 日志里没有该字符串，说明当前测试未确认 KV transfer 命中。

## Mooncake 请求处理

Mooncake 模式不依赖共享目录，也不通过“首 token 拼接”来衔接 P/D。proxy
先懒加载 prefill bootstrap 信息：

```text
GET http://<prefill-host>:<bootstrap-port>/query
```

bootstrap 返回每个 DP rank 的 `engine_id` 和 worker 地址。proxy 将每个
prefill DP rank 展开成一个可调度 entry，并对这些 entry 做 round-robin。

### Prefill 请求

proxy 发给 prefill 的请求会强制：

```json
{
  "stream": false,
  "max_tokens": 1,
  "kv_transfer_params": {
    "do_remote_decode": true,
    "do_remote_prefill": false,
    "transfer_id": "xfer-<uuid>"
  }
}
```

同时附加 headers：

```text
X-Request-Id: <uuid>
X-data-parallel-rank: <prefill-dp-rank>
```

prefill 请求会触发 producer 侧保留本次请求的 KV block，等待 decode 侧
通过 Mooncake 拉取。

### Decode 请求

proxy 发给 decode 的请求保留用户原始 prompt/messages，并注入：

```json
{
  "kv_transfer_params": {
    "do_remote_decode": false,
    "do_remote_prefill": true,
    "remote_bootstrap_addr": "http://<prefill-host>:<bootstrap-port>",
    "remote_engine_id": "<prefill-engine-id>",
    "transfer_id": "xfer-<uuid>"
  }
}
```

decode 侧连接 prefill bootstrap，找到 producer worker，然后通过 Mooncake
TCP transfer engine 拉取 KV cache。

### Mooncake 流式响应

Mooncake 流式响应中，proxy 不构造首 token SSE chunk，而是把 decode
实例返回的 SSE 数据原样转发。prefill 请求仍以后台非流式任务运行；proxy
会在 decode 完成后等待 prefill 任务收尾并记录 upstream 错误。

### Mooncake transfer 验证

启动器中的 `pd_validate_transfer()` 会在 prefill/decode 日志中检查：

```text
Receiving Mooncake KV
pulling kv_caches
MooncakeXferMetadata
```

命中这些日志只说明 Mooncake transfer 被触发；它不等价于所有 hybrid KV
group 语义都正确。

## 当前实现边界

这套 wrapper 适合用于本地验证和测试 P/D 分离链路，但不是完整生产级实现。

主要限制：

- prefill 固定只生成 1 个 token。
- proxy 只支持 `/v1/models`、`/v1/completions`、`/v1/chat/completions`。
- 多 P 多 D 只是 round-robin，没有负载感知、队列协调或会话亲和。
- proxy 不直接理解 KV cache 状态，只依赖 vLLM connector 和 decode 日志验证。
- ExampleConnector 的 Chat 场景通过追加 assistant 消息承接首 token，行为依赖模型 chat template 和 vLLM 对 partial assistant 内容的处理。
- ExampleConnector 的流式响应中，首 token 的 SSE chunk 是 proxy 自己构造的，后续 chunk 来自 decode 实例；Mooncake 流式响应只透传 decode SSE。
- Mooncake 当前验证范围是普通 full attention Qwen3、MPI/MoE EP 场景；
  `Qwen3.5-0.8B` 这类 full attention + linear attention/HMA 多 KV group
  架构尚未支持。当前失败根因是 `LinearAttentionBackend.get_kv_cache_shape()`
  未实现，并且 Mooncake wire metadata 仍是单组 block ids 语义，不能简单把
  HMA 多组 block ids 展平后宣称正确。

## 一句话总结

`pd_launcher.sh` 负责启动 prefill/decode vLLM 实例并生成对应 KV backend
配置；`pd_proxy.py` 负责把一次 OpenAI 生成请求拆成 prefill 和 decode 两段。
ExampleConnector 依赖共享存储和首 token 拼接；Mooncake 依赖 bootstrap 查询和
`kv_transfer_params` 触发 TCP KV transfer。
