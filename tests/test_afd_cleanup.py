"""Process isolation regressions for cross-node AFD launcher cleanup."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest

HELPER = Path(__file__).resolve().parents[1] / "vllm_scripts/mpi_tools/cleanup_af_run.py"


class AfCleanupTest(unittest.TestCase):
    def test_cleanup_preserves_unrelated_run(self):
        tag = f"afd_{time.time_ns()}_{os.getpid()}"
        processes = []
        try:
            for marker in (tag, tag + "_1"):
                processes.append(subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(120)"],
                    env={**os.environ, "VLLM_AF_RUN_ID": marker},
                ))
            subprocess.run([sys.executable, str(HELPER), tag, "TERM"], check=True)
            self.assertEqual(processes[0].wait(timeout=5), -signal.SIGTERM)
            self.assertIsNone(processes[1].poll())
            subprocess.run([sys.executable, str(HELPER), tag, "CHECK"], check=True)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait()

    def test_rejects_empty_or_malformed_run_id(self):
        for tag in ("", "afd_", "other", "afd_123;456"):
            result = subprocess.run(
                [sys.executable, str(HELPER), tag, "KILL"], capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
