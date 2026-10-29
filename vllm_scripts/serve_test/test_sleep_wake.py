"""Release and reacquire the accelerator on a running server via /sleep and /wake_up.

Flow: greedy completion -> /sleep -> /is_sleeping -> /wake_up -> the same
greedy completion. The two completions must match token for token, because
device memory (weights, KV cache, runtime buffers) stays in place across the
interval; only the device compute runtime is released. Requires the server to
run with --enable-sleep-mode and VLLM_SERVER_DEV_MODE=1.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request


def load_preset_env(script_dir, preset_file):
    vllm_scripts_dir = os.path.abspath(os.path.join(script_dir, ".."))
    common_sh = os.path.join(vllm_scripts_dir, "common.sh")
    command = f"""
set -e
source {shlex.quote(common_sh)}
load_env_file {shlex.quote(os.path.join(vllm_scripts_dir, "env.sh"))}
"""
    if preset_file:
        command += f"load_preset_file {shlex.quote(preset_file)}\n"
    else:
        command += f"load_user_config {shlex.quote(vllm_scripts_dir)}\n"
    command += "apply_runtime_overrides\nenv\n"
    result = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, env=os.environ
    )
    if result.returncode != 0:
        print("[错误] 加载环境变量失败:\n" + result.stderr)
        sys.exit(1)
    env = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            env[key] = value
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", dest="preset", default=None, help="preset file")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--level", type=int, default=1, help="sleep level (1 or 2)")
    parser.add_argument("--mode", default="wait", choices=("abort", "wait", "keep"))
    parser.add_argument("--rounds", type=int, default=2, help="sleep/wake cycles")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    env = load_preset_env(script_dir, args.preset)
    base = (args.base_url or f"http://127.0.0.1:{env.get('USER_VLLM_PORT', '8000')}")
    base = base.rstrip("/")
    model = args.model or env.get("USER_VLLM_MODEL")

    def call(method, path, payload=None, timeout=600):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode()
                return response.status, (json.loads(body) if body else None)
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def complete(prompt):
        status, result = call(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 32,
                "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        assert status == 200, (status, result)
        choice = result["choices"][0]
        assert choice["token_ids"], result
        return choice["token_ids"], choice["message"]["content"]

    def is_sleeping():
        status, result = call("GET", "/is_sleeping", timeout=30)
        assert status == 200, (status, result)
        return result["is_sleeping"]

    prompt = "请用一句话介绍一下量子计算。"
    assert not is_sleeping(), "server must start awake"
    reference_ids, reference_text = complete(prompt)
    report = {"reference": reference_text, "cycles": []}

    for cycle in range(args.rounds):
        started = time.monotonic()
        status, result = call(
            "POST", f"/sleep?level={args.level}&mode={args.mode}", timeout=600
        )
        assert status == 200, ("sleep failed", status, result)
        sleep_seconds = time.monotonic() - started
        assert is_sleeping(), "server reports awake after /sleep"

        # A second /sleep is idempotent; the executor only logs a warning.
        status, result = call("POST", f"/sleep?level={args.level}", timeout=600)
        assert status == 200, ("repeated sleep failed", status, result)
        assert is_sleeping()

        started = time.monotonic()
        status, result = call("POST", "/wake_up", timeout=600)
        assert status == 200, ("wake_up failed", status, result)
        assert result["status"] == "awake", result
        wake_seconds = time.monotonic() - started
        assert not is_sleeping(), "server reports sleeping after /wake_up"

        # Repeated /wake_up is a no-op as well.
        status, result = call("POST", "/wake_up", timeout=600)
        assert status == 200, ("repeated wake_up failed", status, result)

        token_ids, text = complete(prompt)
        assert token_ids == reference_ids, {
            "cycle": cycle,
            "reference": reference_text,
            "after_wake": text,
        }
        report["cycles"].append(
            {
                "cycle": cycle,
                "sleep_seconds": round(sleep_seconds, 3),
                "wake_seconds": round(wake_seconds, 3),
                "tokens_match": True,
            }
        )

    report["passed"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
