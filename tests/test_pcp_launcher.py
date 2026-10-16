import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1] / "vllm_scripts"
PRESET = "presets/serial/GLM-5.2-MXFP4-W4A16-dummy-7-layer-lite_dp1_tp1_eager.sh"


class PCPLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        files = [
            "common.sh",
            "launcher_common.sh",
            "user_env_template.sh",
            "mpi_tools/mpi_get_rank_size.sh",
            PRESET,
            "serve/serve_mp_template.sh",
            "serve/serve_head_only_template.sh",
            "serve/serve_mp_rpc_all_mpi_template.sh",
        ]
        for name in files:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        (self.root / "env.sh").write_text("export USER_VLLM_LOGGING_LEVEL=INFO\n")
        (self.root / "logs").mkdir()
        bindir = self.root / "bin"
        bindir.mkdir()
        for name in ("vllm", "python", "python3", "sleep"):
            stub = bindir / name
            stub.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "name = Path(sys.argv[0]).name\n"
                "if name in ('vllm', 'python3'):\n"
                "    data = {'name': name, 'args': sys.argv[1:], "
                "'mpi': os.environ.get('VLLM_CPU_USE_MPI')}\n"
                "    Path(f'call-{os.getpid()}.json').write_text(json.dumps(data))\n"
            )
            stub.chmod(0o755)
        self.env = {
            "PATH": f"{bindir}:/usr/bin:/bin",
            "HOME": self.temp.name,
            "USER_VLLM_PCP_SIZE": "2",
            "VLLM_CPU_USE_MPI": "1",
            "OMPI_COMM_WORLD_RANK": "0",
            "OMPI_COMM_WORLD_SIZE": "2",
        }

    def run_shell(self, command, env=None):
        return subprocess.run(
            ["bash", "-c", command],
            cwd=self.root,
            env=self.env if env is None else env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_preset_and_launcher_counts(self):
        for pcp in (None, "2"):
            for mpi in (None, "0", "1"):
                with self.subTest(pcp=pcp, mpi=mpi):
                    env = dict(self.env)
                    if mpi is None:
                        env.pop("VLLM_CPU_USE_MPI")
                    else:
                        env["VLLM_CPU_USE_MPI"] = mpi
                    if pcp is None:
                        env.pop("USER_VLLM_PCP_SIZE")
                    result = self.run_shell(
                        'source ./common.sh; SCRIPT_DIR="$PWD"; '
                        "source ./launcher_common.sh; "
                        f'PRESET_FILE_INPUT="{PRESET}"; ENV_ARGS=(); '
                        'launcher_load_config; LOG_DIR="$PWD/logs"; LAUNCHER=mpi; '
                        "launcher_prepare_non_pd_ports() { :; }; "
                        "launcher_prepare_runtime launch.log; "
                        'printf "RESULT %s %s %s %s\\n" "$USER_VLLM_PCP_SIZE" '
                        '"$USER_VLLM_MPC_SIZE" "$MPI_COUNT" "$VLLM_CPU_USE_MPI"',
                        env,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    size = pcp or "1"
                    expected_mpi = mpi or "0"
                    self.assertIn(
                        f"RESULT {size} {size} {size} {expected_mpi}", result.stdout
                    )

    def test_parallel_products_and_invalid_values(self):
        command = (
            "source ./common.sh; USER_VLLM_DATA_PARALLEL_SIZE=3; "
            "USER_VLLM_TP_SIZE=2; USER_VLLM_PP_SIZE=2; USER_VLLM_MPC_SIZE=4; "
            'configure_parallel_sizes || exit 1; echo "WORKERS=$USER_VLLM_MPC_SIZE"'
        )
        result = self.run_shell(command)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("WORKERS=8", result.stdout)
        for name, values in (
            ("USER_VLLM_PCP_SIZE", ("0", "-1", "abc", "1+1")),
            ("VLLM_CPU_USE_MPI", ("2", "yes")),
        ):
            for value in values:
                with self.subTest(name=name, value=value):
                    result = self.run_shell(command, dict(self.env, **{name: value}))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(f"Invalid {name}", result.stderr)

    def test_explicit_mpi_size_is_preserved(self):
        result = self.run_shell(
            'source ./common.sh; SCRIPT_DIR="$PWD"; '
            'source ./launcher_common.sh; '
            f'PRESET_FILE_INPUT="{PRESET}"; ENV_ARGS=(); '
            'launcher_load_config; LOG_DIR="$PWD/logs"; LAUNCHER=mpi; '
            'USER_VLLM_MPI_SIZE=6; '
            'launcher_prepare_non_pd_ports() { :; }; '
            'launcher_prepare_runtime launch.log; '
            'printf "WORKERS=%s\\n" "$MPI_COUNT"'
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("WORKERS=6", result.stdout)

    def test_mpi_launch_has_scoped_cleanup_without_afd(self):
        result = self.run_shell(
            'SCRIPT_DIR="$PWD"; source ./launcher_common.sh; '
            'MPI_COUNT=2; ENV_ARGS=(); LAUNCH_LOG=/dev/null; '
            'MPI_WORKERS_LOG=/dev/null; '
            'setsid() { :; }; sleep() { :; }; record_pid() { :; }; '
            'launcher_start_mpi; wait; '
            'printf "RUN=%s HOST=%s\\n" "$VLLM_AF_RUN_ID" '
            '"${AF_CLEANUP_HOSTS[0]}"'
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertRegex(result.stdout, r"RUN=afd_[0-9_]+ HOST=\S+")

    def test_serve_commands_and_worker_rank(self):
        for mpi in ("0", "1"):
            for template in ("mp", "head_only", "mp_rpc_all_mpi"):
                for rank in ("0", "1") if template == "mp_rpc_all_mpi" else ("0",):
                    with self.subTest(mpi=mpi, template=template, rank=rank):
                        for path in self.root.glob("call-*.json"):
                            path.unlink()
                        result = self.run_shell(
                            f'bash serve/serve_{template}_template.sh -e "{PRESET}"',
                            dict(
                                self.env,
                                VLLM_CPU_USE_MPI=mpi,
                                OMPI_COMM_WORLD_RANK=rank,
                            ),
                        )
                        self.assertEqual(
                            result.returncode, 0, result.stdout + result.stderr
                        )
                        calls = [
                            json.loads(p.read_text())
                            for p in self.root.glob("call-*.json")
                        ]
                        servers = [c for c in calls if c["name"] == "vllm"]
                        self.assertEqual(len(servers), 0 if rank == "1" else 1)
                        for call in servers:
                            args = call["args"]
                            self.assertEqual(
                                args.count("--prefill-context-parallel-size"), 1
                            )
                            self.assertEqual(
                                args[args.index("--prefill-context-parallel-size") + 1],
                                "2",
                            )
                            self.assertEqual(call["mpi"], mpi)
                        if template == "mp_rpc_all_mpi":
                            (worker,) = [c for c in calls if c["name"] == "python3"]
                            self.assertEqual(
                                worker["args"][worker["args"].index("--rank") + 1], rank
                            )

    def test_matrix_overrides_reach_service_templates(self):
        spec = importlib.util.spec_from_file_location(
            "pcp_matrix", ROOT / "e2e/pcp_matrix.py"
        )
        matrix = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(matrix)
        for name, (pcp, tp, ep, mpi) in matrix.CASES.items():
            with self.subTest(case=name):
                for path in self.root.glob("call-*.json"):
                    path.unlink()
                env = dict(
                    self.env,
                    USER_VLLM_PCP_SIZE=str(pcp),
                    RUN_VLLM_EFFECTIVE_TP_SIZE=str(tp),
                    RUN_VLLM_EFFECTIVE_EXPERT_PARALLEL=str(ep),
                    VLLM_CPU_USE_MPI=str(mpi),
                    OMPI_COMM_WORLD_SIZE=str(pcp * tp),
                )
                result = self.run_shell(
                    f'bash serve/serve_mp_rpc_all_mpi_template.sh -e "{PRESET}"', env
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                (server,) = [
                    json.loads(p.read_text())
                    for p in self.root.glob("call-*.json")
                    if json.loads(p.read_text())["name"] == "vllm"
                ]
                args = server["args"]
                self.assertIn(f"-tp={tp}", args)
                self.assertEqual(
                    args[args.index("--prefill-context-parallel-size") + 1], str(pcp)
                )
                self.assertEqual("--enable-expert-parallel" in args, bool(ep))
                if ep:
                    self.assertEqual(
                        args[args.index("--all2all-backend") + 1],
                        "torch_all_to_all_single",
                    )
                self.assertEqual(server["mpi"], str(mpi))

    def test_matrix_calls_launcher_and_stops_on_failure(self):
        spec = importlib.util.spec_from_file_location(
            "pcp_matrix", ROOT / "e2e/pcp_matrix.py"
        )
        matrix = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(matrix)
        output = self.root / "matrix"
        with (
            patch.object(sys, "argv", ["pcp_matrix.py", "--output-dir", str(output)]),
            patch.object(
                matrix.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 7),
            ) as run,
        ):
            with self.assertRaises(SystemExit) as error:
                matrix.main()
        self.assertEqual(error.exception.code, 7)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["bash", str(ROOT / "run_vllm_test.sh")])
        self.assertIn(PRESET, command)
        self.assertEqual(
            json.loads((output / "summary.json").read_text()),
            [{"case": "baseline", "returncode": 7}],
        )

    def test_wrong_mpi_world_size_is_rejected(self):
        result = self.run_shell(
            f'bash serve/serve_mp_rpc_all_mpi_template.sh -e "{PRESET}"',
            dict(self.env, OMPI_COMM_WORLD_SIZE="1"),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match DP * TP * PP * PCP", result.stderr)
        self.assertEqual(list(self.root.glob("call-*.json")), [])


if __name__ == "__main__":
    unittest.main()
