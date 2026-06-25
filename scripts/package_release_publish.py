#!/usr/bin/env python3
"""Package release artifacts for repositories listed in the release manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / ".release" / "repository_versions.json"
DEFAULT_OUTPUT_DIR = ROOT / ".release" / "publish"
VLLM_REPOSITORY_NAME = "vllm"


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    stdout: int | None = subprocess.PIPE,
) -> subprocess.CompletedProcess[bytes]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        stdout=stdout,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        command = " ".join(cmd)
        raise RuntimeError(f"{command} failed:\n{stderr}")
    return proc


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema_version") != 1:
        raise ValueError(f"{path} has unsupported schema_version: {data.get('schema_version')}")
    if not isinstance(data.get("repositories"), list):
        raise ValueError(f"{path} does not contain a repositories list")
    return data


def resolve_repo_path(repository: dict[str, Any]) -> Path:
    path = Path(repository["path"])
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def git_text(repo_path: Path, args: list[str]) -> str:
    proc = run(["git", "-C", str(repo_path), *args])
    return proc.stdout.decode("utf-8", errors="replace").strip()


def short_commit(repo_path: Path, commit: str = "HEAD") -> str:
    return git_text(repo_path, ["rev-parse", "--short=6", commit])


def archive_repository(
    repository: dict[str, Any],
    *,
    output_dir: Path,
    release_version: str,
) -> Path:
    name = repository["name"]
    repo_path = resolve_repo_path(repository)
    head_short = short_commit(repo_path)
    output = output_dir / f"{name}_{release_version}_{head_short}.tar.gz"
    tmp_output = output.with_name(f"{output.name}.tmp")
    prefix = f"{name}/"

    if tmp_output.exists():
        tmp_output.unlink()
    proc = subprocess.Popen(
        ["git", "-C", str(repo_path), "archive", "--format=tar", f"--prefix={prefix}", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    with gzip.open(tmp_output, "wb") as f:
        shutil.copyfileobj(proc.stdout, f)
    stderr = proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else ""
    if proc.wait() != 0:
        tmp_output.unlink(missing_ok=True)
        raise RuntimeError(f"git archive {name} failed:\n{stderr}")

    tmp_output.replace(output)
    return output


def create_vllm_patch(
    repository: dict[str, Any],
    *,
    output_dir: Path,
) -> Path:
    repo_path = resolve_repo_path(repository)
    start_commit = repository["version"]
    end_commit = git_text(repo_path, ["rev-parse", "HEAD"])
    start_short = short_commit(repo_path, start_commit)
    end_short = short_commit(repo_path, end_commit)

    generated_name = f"vllm_{dt.date.today().strftime('%Y%m%d')}_{start_short}_{end_short}.patch"
    generated_path = repo_path / generated_name
    output_path = output_dir / generated_name
    if generated_path.exists():
        generated_path.unlink()
    if output_path.exists():
        output_path.unlink()

    run(["bash", "git-format-patch.sh", start_commit, end_commit], cwd=repo_path)
    if not generated_path.exists():
        raise RuntimeError(f"{repo_path / 'git-format-patch.sh'} did not create {generated_path}")
    shutil.move(str(generated_path), str(output_path))
    return output_path


def package_release(manifest_path: Path, output_dir: Path, release_version: str) -> list[Path]:
    manifest = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[Path] = []
    for repository in manifest["repositories"]:
        repo_path = resolve_repo_path(repository)
        name = repository["name"]
        if name == VLLM_REPOSITORY_NAME:
            artifacts.append(create_vllm_patch(repository, output_dir=output_dir))
        elif repo_path != ROOT:
            artifacts.append(
                archive_repository(
                    repository,
                    output_dir=output_dir,
                    release_version=release_version,
                )
            )
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--release-version",
        default=dt.date.today().strftime("%Y%m%d"),
        help="release version used in archive filenames",
    )
    args = parser.parse_args()

    try:
        artifacts = package_release(args.manifest, args.output_dir, args.release_version)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for artifact in artifacts:
        print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
