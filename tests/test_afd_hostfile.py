"""Hostfile placement and cleanup must use the same node configuration."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = r'''
set -e
SCRIPT_DIR="$TEST_SCRIPT_DIR"
source "$SCRIPT_DIR/launcher_common.sh"
MPI_COUNT=2
VLLM_XCPU_ENABLE_AF_EP=1
ENV_ARGS=()
LAUNCH_LOG="$TEST_TMP/head.log"
MPI_WORKERS_LOG="$TEST_TMP/mpi.log"
setsid() { printf '%s\0' "$@" > "$TEST_TMP/command_$1"; }
sleep() { :; }
record_pid() { :; }
launcher_start_mpi
wait
printf '%s\n' "${AF_CLEANUP_HOSTS[@]}" > "$TEST_TMP/cleanup_hosts"
'''


class AfHostfileTest(unittest.TestCase):
    def run_launcher(self, directory, hostfile):
        return subprocess.run(
            ["bash", "-c", SCRIPT], text=True, capture_output=True,
            env={**os.environ, "TEST_SCRIPT_DIR": str(ROOT / "vllm_scripts"),
                 "TEST_TMP": str(directory), "VLLM_MPI_HOSTFILE": str(hostfile),
                 "VLLM_MPI_RUN_ARGS": "--bind-to none --map-by slot"},
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
            self.assertEqual(args.count("-np"), 2)
            self.assertIn(":", args)
            hosts = (directory / "cleanup_hosts").read_text().splitlines()
            self.assertEqual(hosts.count("node03"), 1)
            self.assertEqual(hosts.count("node04"), 1)

    def test_bad_hostfile_fails_before_starting_head(self):
        for content in (None, "# empty\n", "-bad-host slots=1\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                hostfile = directory / "hosts"
                if content is not None:
                    hostfile.write_text(content)
                result = self.run_launcher(directory, hostfile)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((directory / "command_bash").exists())


if __name__ == "__main__":
    unittest.main()
