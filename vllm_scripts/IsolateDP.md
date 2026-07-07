# Isolated DP Test

`run_isolate_dp_test.sh` starts one API/head server locally and starts one
headless DP engine per remote host over SSH.

Current scope is DP isolation only. `TP=1, PP=1` is enforced by this wrapper.
For cross-node TP/PP through MP RPC workers, use `MP_RPC_MPI.md`.

## Default Smoke Test

```bash
./run_isolate_dp_test.sh \
  -e presets/isolate_dp/Qwen3-0.6B_dp2_tp1_eager.sh
```

Default DP hosts are:

```bash
wzk-vllm-xcpu-03-c8,wzk-vllm-xcpu-03-c7
```

The script auto-discovers the first non-loopback/non-docker IP from
`hostname -I` on the API node and each DP host.

## Common Overrides

Use different hostnames:

```bash
ISOLATE_DP_DP_HOSTS="node-a,node-b" \
./run_isolate_dp_test.sh \
  -e presets/isolate_dp/Qwen3-0.6B_dp2_tp1_eager.sh
```

Use explicit IPs when the subnet or interface selection is ambiguous:

```bash
ISOLATE_DP_API_IP=10.0.0.10 \
ISOLATE_DP_DP_HOSTS="node-a,node-b" \
ISOLATE_DP_DP_IPS="10.0.0.11,10.0.0.12" \
./run_isolate_dp_test.sh \
  -e presets/isolate_dp/Qwen3-0.6B_dp2_tp1_eager.sh
```

Use a different shared checkout or venv layout:

```bash
ISOLATE_DP_REMOTE_DIR=/path/to/vllm_scripts \
ISOLATE_DP_REMOTE_VENV_BIN=/path/to/.venv/bin \
./run_isolate_dp_test.sh \
  -e presets/isolate_dp/Qwen3-0.6B_dp2_tp1_eager.sh
```

The same can be passed on the command line:

```bash
./run_isolate_dp_test.sh \
  -e presets/isolate_dp/Qwen3-0.6B_dp2_tp1_eager.sh \
  --remote-dir /path/to/vllm_scripts \
  --remote-venv-bin /path/to/.venv/bin
```

Run only startup and `/v1/models` readiness:

```bash
./run_isolate_dp_test.sh \
  -e presets/isolate_dp/Qwen3-0.6B_dp2_tp1_eager.sh \
  --no-test
```

Keep processes running for manual testing:

```bash
./run_isolate_dp_test.sh \
  -e presets/isolate_dp/Qwen3-0.6B_dp2_tp1_eager.sh \
  --keep-running
```

Logs are written under:

```bash
logs/isolate_dp/<run-id>/
```

## Role Mapping

For `DP=2, TP=1`, the launched processes are:

```text
API/head server: current node
DP rank 0 engine: ISOLATE_DP_DP_HOSTS[0]
DP rank 1 engine: ISOLATE_DP_DP_HOSTS[1]
```

The script passes:

```text
--data-parallel-address = DP rank 0 IP
--data-parallel-rpc-ip  = API/head IP
```
