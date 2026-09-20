"""Hostfile placement and cleanup must use the same node configuration."""

import os
import subprocess
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
        for dp, mpc, ep, rank in ((1, 1, 4, 1), (2, 2, 1, 4), (2, 2, 2, 5)):
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
                    args_file.read_text().splitlines()[:3],
                    ["-m", "vllm_xcpu_plugin.af_ep.moe", "--model"],
                )


if __name__ == "__main__":
    unittest.main()
