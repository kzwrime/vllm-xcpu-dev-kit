#!/usr/bin/env python3
"""Automated end-to-end test runner for vLLM on xcpu.

Runs the test matrix from the requirements doc:
  3 configs (v2_no_pc, v1_pc, v1_no_pc) × 3 models × 4 test types

Usage:
  python3 run_e2e_tests.py                           # full matrix
  python3 run_e2e_tests.py --models 0.8B --tests simple  # minimal smoke test
  python3 run_e2e_tests.py --configs v1_pc --models 0.8B 4B
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE = SCRIPT_DIR / "user_env_template.sh"

CONFIGS = {
    "v2_no_pc": {"v2": True, "prefix_caching": False},
    "v1_no_pc": {"v2": False, "prefix_caching": False},
    "v1_pc": {"v2": False, "prefix_caching": True},
}

MODELS = {
    "0.8B": "presets/serial/Qwen3.5-0.8B_dp1_tp1_eager.sh",
    "4B": "presets/serial/Qwen3.5-4B_dp1_tp1_eager.sh",
    "35B": "presets/serial/Qwen3.6-35B-A3B_dp1_tp1_eager.sh",
}

TEST_TYPES = ["simple", "stream", "multi_stream", "bench"]

DEFAULT_PORT = 14800
BATCHED_TOKENS = 768
SERVER_TIMEOUT = 600
CLIENT_TIMEOUT = 1800

# Complete user_env_template.sh content, owned by this script.
# The two placeholders {V2_MODEL_RUNNER} and {PREFIX_CACHING_ARG} are
# substituted per-config before each test run.
USER_ENV_TEMPLATE = r'''
export USER_VLLM_MODEL="Qwen/Qwen3-0.6B"
export USER_VLLM_MAX_MODEL_LEN=65536

export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_DATA_PARALLEL_ADDRESS="127.0.0.1"
export USER_VLLM_DATA_PARALLEL_RPC_IP="127.0.0.1"
export VLLM_DP_MASTER_WORKER_IP="127.0.0.1"
export USER_VLLM_DATA_PARALLEL_RPC_PORT=13345
export USER_VLLM_PORT=14800
export VLLM_CPU_KVCACHE_SPACE=8

export VLLM_USE_MPI_COORD=0

export ExecutorIP=127.0.0.1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
export USER_VLLM_MP_RPC_WORKER_PER_NODE=1

export VLLM_USE_CPU_SHM_DIST=0
export VLLM_LOOPBACK_IP=$(hostname -I | awk '{{print $1}}')

export VLLM_USE_XCPU_LINEAR=1
export TORCH_XCPU_ENABLE_CHECK=0
export VLLM_CPU_USE_MPI=0
export TORCHINDUCTOR_CPP_WRAPPER=1
export TORCH_MCPU_INDUCTOR_FALLBACK_BY_DEFAULT=0
export VLLM_DISABLE_TQDM_AND_MONITOR=0
export VLLM_SHARED_EXPERT_DISABLE_TP=1
export VLLM_USE_XCPU_TOPK_SOFTMAX=1
export VLLM_USE_XCPU_TOPK_TOPP_SAMPLER=1
export VLLM_USE_V2_MODEL_RUNNER={V2_MODEL_RUNNER}
export VLLM_XCPU_USE_FUSED_DOT_SIGMOID_MUL_ADD=0
export VLLM_XCPU_FUSE_GDN_IN_PROJ_QKVZBA=0
export VLLM_XCPU_USE_FUSED_FFN=0

export VLLM_ENABLE_SEQUENCE_PARALLEL_MOE=0

export VLLM_ALL2ALL_BACKEND_XCPU="mpi_alltoallv"
export VLLM_MPI_ALLTOALLV_VERSION="v2"

export VLLM_XCPU_ENABLE_DUMMY_RUN_FAST_PATH="${{VLLM_XCPU_ENABLE_DUMMY_RUN_FAST_PATH:-1}}"

export PD_MODE="${{PD_MODE:-MIXED}}"

_VLLM_OPTIONAL_ARGS=" "
_VLLM_OPTIONAL_ARGS+=" --max-num-seqs 16"
_VLLM_OPTIONAL_ARGS+=" {PREFIX_CACHING_ARG}"
_VLLM_OPTIONAL_ARGS+=" --mamba-cache-mode none"

_VLLM_OPTIONAL_ARGS+=' --profiler-config {{"profiler":"torch","torch_profiler_dir":"./vllm_profile","torch_profiler_record_shapes":true,"torch_profiler_with_memory":true,"torch_profiler_with_stack":true,"torch_profiler_with_flops":true,"torch_profiler_use_gzip":true,"torch_profiler_dump_cuda_time_total":true,"torch_profiler_no_trace_file":false}}'

case ${{PD_MODE}} in
    "PREFILL")
        echo "[VLLM-XCPU] PD Mode: PREFILL - Optimizing for TTFT"
        _VLLM_OPTIONAL_ARGS+=" --enable-expert-parallel"
        MAX_BATCHED_TOKENS="${{USER_VLLM_MAX_NUM_BATCHED_TOKENS:-4096}}"
        export USER_VLLM_MAX_NUM_BATCHED_TOKENS=${{MAX_BATCHED_TOKENS}}
        export VLLM_MOE_DP_CHUNK_SIZE=${{MAX_BATCHED_TOKENS}}
        export VLLM_ENABLE_MOE_DP_CHUNK=0
        export VLLM_SHARED_EXPERT_DISABLE_TP=1
        ;;
    "DECODE" | "MIXED")
        echo "[VLLM-XCPU] PD Mode: ${{PD_MODE}} - Optimizing for TPOT"
        _VLLM_OPTIONAL_ARGS+=" --enable-expert-parallel"
        MAX_BATCHED_TOKENS="${{USER_VLLM_MAX_NUM_BATCHED_TOKENS:-256}}"
        export USER_VLLM_MAX_NUM_BATCHED_TOKENS=${{MAX_BATCHED_TOKENS}}
        export VLLM_MOE_DP_CHUNK_SIZE=${{MAX_BATCHED_TOKENS}}
        export VLLM_ENABLE_MOE_DP_CHUNK=0
        export VLLM_SHARED_EXPERT_DISABLE_TP=1
        ;;
    "NOT_MOE")
        export USER_VLLM_MAX_NUM_BATCHED_TOKENS="${{USER_VLLM_MAX_NUM_BATCHED_TOKENS:-256}}"
        export VLLM_SHARED_EXPERT_DISABLE_TP=0
        ;;
    *)
        echo "Error: Invalid PD_MODE '${{PD_MODE}}'."
        exit 1
        ;;
esac

export VLLM_OPTIONAL_ARGS=${{_VLLM_OPTIONAL_ARGS}}

echo "========================================="
echo "  VLLM Configuration Summary"
echo "========================================="
echo "  PD_MODE: ${{PD_MODE}}"
echo "  USER_VLLM_EAGER_OR_NOT: '${{USER_VLLM_EAGER_OR_NOT}}'"
echo "  VLLM_OPTIONAL_ARGS: ${{VLLM_OPTIONAL_ARGS}}"
echo "========================================="

export TORCHINDUCTOR_CACHE_DIR="$PWD/torch_compile_cache_opt"

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_HUB_OFFLINE=1

export AOTI_TORCH_ALWAYS_REUSE=1
export TORCHINDUCTOR_DIRECT_DISPATCH_PREFIXES="torch_xcpu,torch_mpi_ext.all_reduce__wrapper"

_USER_ENV_TEMPLATE_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
_TORCH_XCPU_AOTI_ENV="$(
AOTI_EXTRA_CFLAGS= \
AOTI_EXTRA_LDFLAGS= \
python "${{_USER_ENV_TEMPLATE_DIR}}/torch_xcpu_aoti_env.py"
)" || exit 1
eval "${{_TORCH_XCPU_AOTI_ENV}}"
unset _USER_ENV_TEMPLATE_DIR _TORCH_XCPU_AOTI_ENV

echo "AOTI_EXTRA_CFLAGS: ${{AOTI_EXTRA_CFLAGS}}"
echo "AOTI_EXTRA_LDFLAGS: ${{AOTI_EXTRA_LDFLAGS}}"
'''


def generate_template(cfg: dict) -> str:
    """Generate complete user_env_template.sh for the given config."""
    v2_val = "1" if cfg["v2"] else "0"
    pc_arg = "--enable-prefix-caching" if cfg["prefix_caching"] else "--no-enable-prefix-caching"
    return USER_ENV_TEMPLATE.format(
        V2_MODEL_RUNNER=v2_val,
        PREFIX_CACHING_ARG=pc_arg,
    )


def wait_for_server(port: int, timeout: int = SERVER_TIMEOUT) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def kill_process_group(proc: subprocess.Popen) -> None:
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def run_simple(preset: str, out_dir: Path, env: dict) -> int:
    with open(out_dir / "stdout.log", "w") as fout, \
         open(out_dir / "stderr.log", "w") as ferr:
        proc = subprocess.run(
            ["bash", str(SCRIPT_DIR / "run_vllm_test.sh"), "-e", preset],
            cwd=str(SCRIPT_DIR),
            env=env,
            stdout=fout,
            stderr=ferr,
            timeout=SERVER_TIMEOUT + CLIENT_TIMEOUT,
        )
    return proc.returncode


def run_with_client(
    preset: str,
    client_cmd: list[str],
    out_dir: Path,
    env: dict,
    port: int = DEFAULT_PORT,
) -> int:
    server_out = open(out_dir / "server_stdout.log", "w")
    server_err = open(out_dir / "server_stderr.log", "w")

    server = subprocess.Popen(
        ["bash", str(SCRIPT_DIR / "run_vllm_test.sh"), "-e", preset, "--no-test"],
        cwd=str(SCRIPT_DIR),
        env=env,
        stdout=server_out,
        stderr=server_err,
        start_new_session=True,
    )

    try:
        if not wait_for_server(port):
            print(f"    ERROR: server did not start within {SERVER_TIMEOUT}s")
            return 1

        with open(out_dir / "client_stdout.log", "w") as cout, \
             open(out_dir / "client_stderr.log", "w") as cerr:
            result = subprocess.run(
                client_cmd,
                cwd=str(SCRIPT_DIR),
                env=env,
                stdout=cout,
                stderr=cerr,
                timeout=CLIENT_TIMEOUT,
            )
        return result.returncode
    except subprocess.TimeoutExpired:
        print("    ERROR: client timed out")
        return 1
    finally:
        kill_process_group(server)
        server_out.close()
        server_err.close()


def run_bench(preset: str, out_dir: Path, env: dict) -> int:
    with open(out_dir / "stdout.log", "w") as fout, \
         open(out_dir / "stderr.log", "w") as ferr:
        proc = subprocess.run(
            ["bash", str(SCRIPT_DIR / "run_vllm_test.sh"), "-e", preset, "--bench"],
            cwd=str(SCRIPT_DIR),
            env=env,
            stdout=fout,
            stderr=ferr,
            timeout=SERVER_TIMEOUT + CLIENT_TIMEOUT,
        )
    return proc.returncode


def extract_answer(out_dir: Path) -> str:
    for name in ["stdout.log", "client_stdout.log"]:
        path = out_dir / name
        if not path.exists():
            continue
        text = path.read_text()
        for line in text.splitlines():
            if "模型回答" in line or "reasoning" in line.lower():
                return line.strip()[:200]
        for line in text.splitlines():
            if '"reasoning"' in line and len(line) > 50:
                return line.strip()[:200]
    return "(no answer extracted)"


def run_test_case(
    config_name: str,
    model_name: str,
    test_type: str,
    out_dir: Path,
    env: dict,
) -> tuple[bool, str]:
    preset = MODELS[model_name]
    out_dir.mkdir(parents=True, exist_ok=True)

    if test_type == "simple":
        rc = run_simple(preset, out_dir, env)
    elif test_type == "stream":
        rc = run_with_client(
            preset,
            ["python3", "serve_test/test_stream.py", "-e", preset],
            out_dir,
            env,
        )
    elif test_type == "multi_stream":
        rc = run_with_client(
            preset,
            [
                "python3", "serve_test/test_multl_stream.py",
                "--max-tokens", "6000", "-e", preset,
            ],
            out_dir,
            env,
        )
    elif test_type == "bench":
        rc = run_bench(preset, out_dir, env)
    else:
        return False, f"unknown test type: {test_type}"

    answer = extract_answer(out_dir)
    if rc == 0:
        return True, answer
    return False, f"exit_code={rc}  {answer}"


def main():
    parser = argparse.ArgumentParser(description="Automated e2e test runner")
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS.keys()),
        choices=list(MODELS.keys()), help="Models to test",
    )
    parser.add_argument(
        "--configs", nargs="+", default=list(CONFIGS.keys()),
        choices=list(CONFIGS.keys()), help="Configs to test",
    )
    parser.add_argument(
        "--tests", nargs="+", default=TEST_TYPES,
        choices=TEST_TYPES, help="Test types to run",
    )
    parser.add_argument(
        "--batched-tokens", type=int, default=BATCHED_TOKENS,
        help="USER_VLLM_MAX_NUM_BATCHED_TOKENS",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = SCRIPT_DIR / "e2e_results" / timestamp
    results_root.mkdir(parents=True, exist_ok=True)

    original_template = TEMPLATE.read_text()
    base_env = os.environ.copy()
    base_env["USER_VLLM_MAX_NUM_BATCHED_TOKENS"] = str(args.batched_tokens)

    results: list[tuple[str, str, str, bool, str]] = []
    total = len(args.configs) * len(args.models) * len(args.tests)
    idx = 0

    try:
        for config_name in args.configs:
            TEMPLATE.write_text(generate_template(CONFIGS[config_name]))

            for model_name in args.models:
                for test_type in args.tests:
                    idx += 1
                    tag = f"{config_name}__{model_name}__{test_type}"
                    out_dir = results_root / tag
                    print(f"[{idx}/{total}] {tag} ...")

                    ok, detail = run_test_case(
                        config_name, model_name, test_type, out_dir, base_env,
                    )
                    status = "PASS" if ok else "FAIL"
                    print(f"  -> {status}  {detail[:120]}")
                    results.append((config_name, model_name, test_type, ok, detail))
    finally:
        TEMPLATE.write_text(original_template)
        print(f"\nRestored {TEMPLATE}")

    summary_path = results_root / "summary.txt"
    passed = sum(1 for *_, ok, _ in results if ok)
    failed = sum(1 for *_, ok, _ in results if not ok)
    with open(summary_path, "w") as f:
        f.write(f"E2E Test Summary  {timestamp}\n")
        f.write(f"Total: {len(results)}  Passed: {passed}  Failed: {failed}\n")
        f.write("=" * 80 + "\n\n")
        for config, model, ttype, ok, detail in results:
            status = "PASS" if ok else "FAIL"
            f.write(f"[{status}] {config} / {model} / {ttype}\n")
            f.write(f"       {detail[:200]}\n\n")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Output:  {results_root}")
    print(f"Summary: {summary_path}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
