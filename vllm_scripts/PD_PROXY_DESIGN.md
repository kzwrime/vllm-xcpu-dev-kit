# P/D Proxy Wrapper Design

本文说明当前基于共享存储的 P/D 分离部署 wrapper 的实现原理。

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

prefill/decode 共同使用:
logs/pd/<run-id>/example_connector_storage/
```

proxy 对客户端暴露普通 OpenAI API；内部把一次生成请求拆成两次请求：

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
- `example_connector_storage/` 是 prefill 和 decode 共用的 KV 交换目录。

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

```bash
_VLLM_PD_KV_TRANSFER_CONFIG='{"kv_connector":"ExampleConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"<storage-dir>"}}'

export VLLM_OPTIONAL_ARGS="${VLLM_OPTIONAL_ARGS} \
  --kv-transfer-config ${_VLLM_PD_KV_TRANSFER_CONFIG} \
  --no-disable-hybrid-kv-cache-manager"
```

这里真正决定 P/D 共享 KV 的配置是 `--kv-transfer-config` 中的 `shared_storage_path`。`VLLM_PD_ROLE`、`VLLM_PD_ROLE_INDEX` 这类变量没有实际消费者，不能作为功能依赖。

## 端口和实例

`pd_launcher.sh` 会分别给 prefill、decode、proxy 分配端口：

- prefill HTTP 端口默认从 `USER_VLLM_PORT + 100` 附近寻找。
- decode HTTP 端口默认从 `USER_VLLM_PORT + 200` 附近寻找。
- proxy 默认使用 `USER_VLLM_PORT`，如果被占用则寻找空闲端口。
- DP RPC、MPI coord、MP RPC ready port 也会分别分配，避免 prefill/decode 冲突。

默认拓扑是 `1P x 1D`。可以通过环境变量扩展：

```bash
USER_VLLM_PD_PREFILL_COUNT=2 USER_VLLM_PD_DECODE_COUNT=1 \
./run_vllm_test.sh --pd --multi-test -e <preset.sh>
```

多实例场景下，proxy 接收逗号分隔 URL：

```bash
python ./serve_test/pd_proxy.py \
  --port 14800 \
  --prefill-url http://127.0.0.1:14900,http://127.0.0.1:15000 \
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

生成请求由 `_handle_generation()` 处理，核心流程如下：

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

## 当前实现边界

这套 wrapper 适合用于本地验证和测试 P/D 分离链路，但不是完整生产级实现。

主要限制：

- prefill 固定只生成 1 个 token。
- proxy 只支持 `/v1/models`、`/v1/completions`、`/v1/chat/completions`。
- 多 P 多 D 只是 round-robin，没有负载感知、队列协调或会话亲和。
- proxy 不直接理解 KV cache 状态，只依赖 vLLM connector 和 decode 日志验证。
- Chat 场景通过追加 assistant 消息承接首 token，行为依赖模型 chat template 和 vLLM 对 partial assistant 内容的处理。
- 流式响应中，首 token 的 SSE chunk 是 proxy 自己构造的，后续 chunk 来自 decode 实例。

## 一句话总结

`pd_launcher.sh` 负责启动多个挂载同一共享存储 KV connector 的 vLLM 实例；`pd_proxy.py` 负责把一次 OpenAI 生成请求拆成 “prefill 生成首 token + decode 续写剩余 token”。共享存储中的 KV cache 是 prefill 和 decode 衔接的关键。
