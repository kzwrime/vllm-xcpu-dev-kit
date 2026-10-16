import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPTS = Path(__file__).resolve().parents[1]
PRESET = "presets/serial/GLM-5.2-MXFP4-W4A16-dummy-7-layer-lite_dp1_tp1_eager.sh"
CASES = {
    "baseline": (1, 1, 0, 1),
    "pcp": (2, 1, 0, 1),
    "tp": (1, 2, 0, 1),
    "tp-pcp": (2, 2, 0, 1),
    "ep-pcp": (2, 1, 1, 1),
    "tp-pcp-mpi0": (2, 2, 0, 0),
    "ep-pcp-mpi0": (2, 1, 1, 0),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run dummy PCP cases through the kit launcher"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    summary = []
    for name in args.cases:
        pcp, tp, ep, mpi = CASES[name]
        folder = root / name
        folder.mkdir()
        settings = {
            "USER_VLLM_PCP_SIZE": str(pcp),
            "RUN_VLLM_EFFECTIVE_TP_SIZE": str(tp),
            "RUN_VLLM_EFFECTIVE_EXPERT_PARALLEL": str(ep),
            "VLLM_CPU_USE_MPI": str(mpi),
            "USER_VLLM_MAX_NUM_BATCHED_TOKENS": "32",
            "USER_VLLM_MAX_NUM_SEQS": "1",
            "VLLM_TEST_MAX_WAIT": "600",
            "VLLM_TEST_LOG_DIR": str(folder / "logs"),
        }
        command = [
            "bash",
            str(SCRIPTS / "run_vllm_test.sh"),
            "--launcher",
            "mpi",
            "-e",
            PRESET,
            "--test-timeout",
            "180",
        ]
        (folder / "config.json").write_text(
            json.dumps(
                {"command": command, "overrides": settings, "python": sys.executable},
                indent=2,
            )
        )
        env = dict(os.environ, **settings)
        env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
        with (folder / "launcher.log").open("w") as log:
            result = subprocess.run(
                command, cwd=SCRIPTS, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        summary.append({"case": name, "returncode": result.returncode})
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary[-1]), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
