from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.vllm_triton_kernel_analyzer.matcher import compare_inventories
from tools.vllm_triton_kernel_analyzer.report import write_comparison
from tools.vllm_triton_kernel_analyzer.scanner import scan_repository


def make_repo(root: Path, files: dict[str, str]) -> Path:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    return root


class AnalyzerTest(unittest.TestCase):
    def test_scan_aliases_nested_and_excludes_numba(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = make_repo(Path(temp) / "repo", {
                "kernels.py": """\
import triton as tt
from numba import jit

@tt.autotune(configs=[], key=[])
@tt.jit(debug=True)
def kernel_a(x):
    return x + 1

@jit(nopython=True)
def cpu_function(x):
    return x

class Container:
    @tt.jit
    def kernel_b(x):
        return x * 2
""",
                "optional.py": """\
try:
    from triton import jit as triton_jit
except ImportError:
    triton_jit = None

@triton_jit
def optional_kernel(x):
    return x
""",
                "reexport.py": """\
from vllm.triton_utils import triton as vt

@vt.jit
def reexported_kernel(x):
    return x
""",
            })
            inventory = scan_repository(repo)
            self.assertEqual([k.qualname for k in inventory.kernels],
                             ["kernel_a", "Container.kernel_b", "optional_kernel", "reexported_kernel"])
            self.assertEqual(inventory.kernels[0].decorator_line, 4)
            self.assertIn("@tt.autotune", inventory.kernels[0].source)
            self.assertFalse(inventory.issues)

    def test_match_move_rename_modify_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            old_repo = make_repo(base / "old", {
                "old/place.py": """\
import triton
@triton.jit
def old_kernel(x, BLOCK: int):
    offsets = range(BLOCK)
    values = x + offsets
    return values

@triton.jit
def changed(x):
    return x + 1

@triton.jit
def gone(x):
    return x - 1
""",
            })
            new_repo = make_repo(base / "new", {
                "new/place.py": """\
import triton
@triton.jit
def renamed_kernel(x, BLOCK: int):
    offsets = range(BLOCK)
    values = x + offsets
    return values
""",
                "old/place.py": """\
import triton
@triton.jit
def changed(x):
    y = x + 2
    return y

@triton.jit
def added(x):
    return x * x
""",
            })
            comparison = compare_inventories(scan_repository(old_repo), scan_repository(new_repo))
            statuses = {match.status for match in comparison.matches}
            self.assertIn("moved_renamed_semantic_exact", statuses)
            self.assertIn("modified", statuses)
            self.assertIn("removed", statuses)
            self.assertIn("added", statuses)
            output = base / "report"
            write_comparison(comparison, output)
            self.assertTrue((output / "report.html").is_file())
            self.assertTrue((output / "report.md").is_file())
            self.assertGreaterEqual(len(list((output / "diffs").glob("*.diff"))), 4)
            self.assertIn("renamed_kernel", (output / "report.html").read_text())

    def test_non_git_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                scan_repository(temp)

    def test_duplicate_qualified_names_match_one_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            old_repo = make_repo(base / "old", {"duplicate.py": """\
import triton
if True:
    @triton.jit
    def kernel(x):
        return x + 1
else:
    @triton.jit
    def kernel(x):
        return x * 2
"""})
            new_repo = make_repo(base / "new", {"duplicate.py": """\
import triton
if True:
    @triton.jit
    def kernel(x):
        return x + 2
else:
    @triton.jit
    def kernel(x):
        return x * 3
"""})
            comparison = compare_inventories(scan_repository(old_repo), scan_repository(new_repo))
            pairs = [item for item in comparison.matches if item.old and item.new]
            self.assertEqual(len(pairs), 2)
            self.assertTrue(all(item.method == "location" for item in pairs))


if __name__ == "__main__":
    unittest.main()
