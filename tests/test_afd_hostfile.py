"""Hostfile placement and cleanup must use the same node configuration."""

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = r"""
set -e
SCRIPT_DIR="$TEST_SCRIPT_DIR"
source "$SCRIPT_DIR/launcher_common.sh"
MPI_COUNT=2
VLLM_XCPU_ENABLE_AF_EP=1
VLLM_MPI_WORKER_TEMPLATE="$SCRIPT_DIR/serve/serve_afd_mp_rpc_all_mpi_template.sh"
ENV_ARGS=()
LAUNCH_LOG="$TEST_TMP/head.log"
MPI_WORKERS_LOG="$TEST_TMP/mpi.log"
setsid() { printf '%s\0' "$@" > "$TEST_TMP/command_$1"; }
sleep() { :; }
record_pid() { :; }
launcher_start_mpi
wait
printf '%s\n' "${AF_CLEANUP_HOSTS[@]}" > "$TEST_TMP/cleanup_hosts"
"""


class AfHostfileTest(unittest.TestCase):
    def test_afd_presets_fix_execution_mode_without_changing_rank_layout(self):
        presets = sorted((ROOT / "vllm_scripts/presets/mpi/moe").glob("*af_ep*v7.sh"))
        self.assertTrue(presets)
        for preset in presets:
            mode = re.search(r"_(eager|compile)_", preset.name).group(1)
            opposite = "eager" if mode == "compile" else "compile"
            self.assertTrue(
                preset.with_name(
                    preset.name.replace(f"_{mode}_", f"_{opposite}_")
                ).exists()
            )
            with self.subTest(preset=preset.name):
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        (
                            'source "$1" || exit; '
                            'printf "RESULT:%s:%s:%s:%s:%s\\n" '
                            '"$USER_VLLM_EAGER_OR_NOT" "$USER_VLLM_AFD_F_COMPILE" '
                            '"$USER_VLLM_MPI_SIZE" '
                            '"$((USER_VLLM_DATA_PARALLEL_SIZE * USER_VLLM_MPC_SIZE '
                            '+ USER_VLLM_EP_SIZE))" "$VLLM_USE_V2_MODEL_RUNNER"'
                        ),
                        "bash",
                        str(preset),
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                    env={
                        **os.environ,
                        # File choice must win over inherited execution modes.
                        "USER_VLLM_EAGER_OR_NOT": "--enforce-eager"
                        if mode == "compile"
                        else "",
                        "VLLM_USE_V2_MODEL_RUNNER": "0",
                        "USER_VLLM_AFD_F_COMPILE": "0" if mode == "compile" else "1",
                        "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
                    },
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                line = next(
                    line
                    for line in result.stdout.splitlines()
                    if line.startswith("RESULT:")
                )
                _, flag, compile_f, size, expected_size, runner_v2 = line.split(":")
                self.assertEqual(flag, "" if mode == "compile" else "--enforce-eager")
                self.assertEqual(compile_f, "1" if mode == "compile" else "0")
                self.assertEqual(size, expected_size)
                self.assertEqual(runner_v2, "1")

    def test_afd_matrix_selects_preset_files_and_forwards_test_arguments(self):
        matrix = ROOT / "vllm_scripts/e2e/run_afd_crossnode_matrix.sh"
        for mode_args, mode in (
            ([], "eager"),
            (["--mode", "eager"], "eager"),
            (["--mode", "compile"], "compile"),
        ):
            with (
                self.subTest(mode_args=mode_args),
                tempfile.TemporaryDirectory() as tmp,
            ):
                directory = Path(tmp)
                (directory / "e2e").mkdir()
                copied_matrix = directory / "e2e/matrix.sh"
                copied_matrix.write_text(matrix.read_text())
                hostfile = directory / "hosts"
                hostfile.write_text("node02 slots=6\n")
                calls = directory / "calls"
                stub = directory / "run_vllm_test.sh"
                stub.write_text(
                    '#!/bin/bash\nprintf "%s\t%s\n" '
                    '"$USER_VLLM_EP_SIZE" "$*" >> "$TEST_MATRIX_CALLS"\n'
                )
                stub.chmod(0o755)
                result = subprocess.run(
                    [
                        "bash",
                        str(copied_matrix),
                        *mode_args,
                        "--multi-test-max-tokens",
                        "8",
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                    env={
                        **os.environ,
                        "VLLM_MPI_HOSTFILE": str(hostfile),
                        "TEST_MATRIX_CALLS": str(calls),
                    },
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                lines = calls.read_text().splitlines()
                self.assertEqual(len(lines), 6)
                self.assertEqual(
                    [line.split("\t")[0] for line in lines],
                    ["2", "4", "1", "2", "2", "2"],
                )
                for line in lines:
                    args = line.split("\t")[1].split()
                    self.assertTrue(args[1].endswith(f"_{mode}_v7.sh"))
                    self.assertTrue((ROOT / "vllm_scripts" / args[1]).exists())
                    self.assertEqual(args[-2:], ["--multi-test-max-tokens", "8"])
                    self.assertNotIn("--mode", args)

    def run_launcher(self, directory, hostfile):
        return subprocess.run(
            ["bash", "-c", SCRIPT],
            text=True,
            capture_output=True,
            check=False,
            env={
                **os.environ,
                "TEST_SCRIPT_DIR": str(ROOT / "vllm_scripts"),
                "TEST_TMP": str(directory),
                "VLLM_MPI_HOSTFILE": str(hostfile),
                "VLLM_MPI_RUN_ARGS": "--bind-to none --map-by slot",
            },
        )

    def test_shared_hostfile_and_unique_cleanup_hosts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            hostfile = directory / "hosts with spaces"
            hostfile.write_text(
                "# placement\n\nnode03 slots=1 # A\nnode04 slots=1\nnode03 slots=1"
            )
            result = self.run_launcher(directory, hostfile)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = (directory / "command_mpirun").read_bytes().decode().split("\0")
            self.assertEqual(args.count("--hostfile"), 1)
            self.assertEqual(args[args.index("--hostfile") + 1], str(hostfile))
            self.assertNotIn("--host", args)
            self.assertEqual(args.count("-np"), 1)
            self.assertEqual(args[args.index("-np") + 1], "2")
            self.assertNotIn(":", args)
            self.assertTrue(
                any(
                    arg.endswith("serve_afd_mp_rpc_all_mpi_template.sh") for arg in args
                )
            )
            hosts = (directory / "cleanup_hosts").read_text().splitlines()
            self.assertEqual(hosts.count("node03"), 1)
            self.assertEqual(hosts.count("node04"), 1)

    def test_bad_hostfile_fails_before_starting_head(self):
        for content in (None, "# empty\n", "-bad-host slots=1\n"):
            with (
                self.subTest(content=content),
                tempfile.TemporaryDirectory() as temporary,
            ):
                directory = Path(temporary)
                hostfile = directory / "hosts"
                if content is not None:
                    hostfile.write_text(content)
                result = self.run_launcher(directory, hostfile)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((directory / "command_bash").exists())

    def test_afd_template_selects_moe_by_global_rank_for_unequal_topologies(self):
        template = ROOT / "vllm_scripts/serve/serve_afd_mp_rpc_all_mpi_template.sh"
        for dp, mpc, ep, rank, compile_f in (
            (1, 1, 4, 1, "0"),
            (2, 2, 1, 4, "1"),
            (2, 2, 2, 5, "1"),
        ):
            with (
                self.subTest(dp=dp, mpc=mpc, ep=ep),
                tempfile.TemporaryDirectory() as temporary,
            ):
                directory = Path(temporary)
                preset = directory / "preset.sh"
                preset.write_text(
                    f"""
export USER_VLLM_DATA_PARALLEL_SIZE={dp}
export USER_VLLM_MPC_SIZE={mpc}
export USER_VLLM_EP_SIZE={ep}
export USER_VLLM_MODEL=test-model
export USER_VLLM_MAX_NUM_BATCHED_TOKENS=8
export USER_VLLM_LOAD_FORMAT=dummy
export USER_VLLM_AFD_F_COMPILE={compile_f}
export VLLM_OPTIONAL_ARGS=
"""
                )
                fake_bin = directory / "bin"
                fake_bin.mkdir()
                fake_python = fake_bin / "python"
                fake_python.write_text(
                    '#!/bin/bash\nprintf "%s\\n" "$@" > "$TEST_PYTHON_ARGS"\n'
                )
                fake_python.chmod(0o755)
                args_file = directory / "python_args"
                world_size = dp * mpc + ep
                result = subprocess.run(
                    ["bash", str(template), "-e", str(preset)],
                    text=True,
                    capture_output=True,
                    check=False,
                    env={
                        **os.environ,
                        "PATH": f"{fake_bin}:{os.environ['PATH']}",
                        "TEST_PYTHON_ARGS": str(args_file),
                        "OMPI_COMM_WORLD_RANK": str(rank),
                        "OMPI_COMM_WORLD_SIZE": str(world_size),
                        "OMPI_COMM_WORLD_LOCAL_RANK": "0",
                    },
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("role=F host=", result.stdout)
                self.assertIn(
                    f"global_rank={rank} role_rank={rank - dp * mpc}",
                    result.stdout,
                )
                self.assertEqual(
                    "--compile" in args_file.read_text().splitlines(), compile_f == "1"
                )
                self.assertEqual(
                    args_file.read_text().splitlines()[:3],
                    ["-m", "vllm_xcpu_plugin.af_ep.moe", "--model"],
                )


if __name__ == "__main__":
    unittest.main()
