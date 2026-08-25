from __future__ import annotations

import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import generate_update_notes, package_release_publish


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return proc.stdout.strip()


class ReleaseScriptsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)
        git(self.repo, "init", "-b", "main")
        git(self.repo, "config", "user.name", "Release Test")
        git(self.repo, "config", "user.email", "release-test@example.com")
        self._commit_file("shared.txt", "initial\n", "initial")
        self.initial = git(self.repo, "rev-parse", "HEAD")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _commit_file(self, name: str, content: str, subject: str) -> str:
        (self.repo / name).write_text(content, encoding="utf-8")
        git(self.repo, "add", name)
        git(self.repo, "commit", "-m", subject)
        return git(self.repo, "rev-parse", "HEAD")

    def test_commit_line_includes_file_and_line_stats(self) -> None:
        self._commit_file("feature.txt", "one\ntwo\n", "add feature")
        repository = {
            "name": "another-repository",
            "path": str(self.repo),
            "version": self.initial,
        }

        lines = generate_update_notes.get_commit_lines(repository)

        self.assertEqual(len(lines), 1)
        self.assertIn("add feature", lines[0])
        self.assertTrue(
            lines[0].endswith("（修改 1 个文件，新增 2 行，删除 0 行）")
        )

    def test_divergent_vllm_uses_main_and_skips_patch(self) -> None:
        git(self.repo, "switch", "-c", "old-release")
        old_release = self._commit_file("old.txt", "old\n", "old release")
        git(self.repo, "switch", "main")
        git(self.repo, "switch", "-c", "new-development")
        self._commit_file("new.txt", "new\n", "new development")
        repository = {
            "name": "vllm",
            "path": str(self.repo),
            "version": old_release,
        }

        lines = generate_update_notes.get_commit_lines(
            repository, vllm_main_ref="main"
        )

        self.assertIn("按 main..HEAD 统计", lines[0])
        self.assertTrue(any("new development" in line for line in lines))
        self.assertFalse(any("old release" in line for line in lines))

        output_dir = self.repo / "publish"
        output_dir.mkdir()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            patch = package_release_publish.create_vllm_patch(
                repository, output_dir=output_dir
            )
        self.assertIsNone(patch)
        self.assertIn("skip vllm patch", stderr.getvalue())
        self.assertEqual(list(output_dir.glob("*.patch")), [])

    def test_contiguous_vllm_patch_does_not_require_repository_helper(self) -> None:
        head = self._commit_file("feature.txt", "feature\n", "add feature")
        repository = {
            "name": "vllm",
            "path": str(self.repo),
            "version": self.initial,
        }
        output_dir = self.repo / "publish"
        output_dir.mkdir()

        patch = package_release_publish.create_vllm_patch(
            repository,
            output_dir=output_dir,
            release_version="20990101",
        )

        self.assertIsNotNone(patch)
        assert patch is not None
        self.assertEqual(
            patch.name,
            f"vllm_20990101_{head[:6]}_{head[:6]}.patch",
        )
        self.assertIn("Subject: [PATCH] add feature", patch.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
