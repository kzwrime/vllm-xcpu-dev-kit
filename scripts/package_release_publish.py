#!/usr/bin/env python3
"""Package release artifacts for repositories listed in the release manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / ".release" / "repository_versions.json"
DEFAULT_OUTPUT_DIR = ROOT / ".release" / "publish"
PUBLISHED_MANIFEST_NAME = "repository_versions_currently.json"
EXPORT_VERSION_METADATA = ".release/repository_version.json"
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


def current_branch(repo_path: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_path), "symbolic-ref", "--quiet", "--short", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{repo_path} is in detached HEAD state; release packages must be "
            "built from a checked-out branch"
        )
    return proc.stdout.decode("utf-8", errors="replace").strip()


def short_commit(repo_path: Path, commit: str = "HEAD") -> str:
    return git_text(repo_path, ["rev-parse", "--short=6", commit])


def first_commit_after(repo_path: Path, base_commit: str, end_commit: str) -> str | None:
    output = git_text(repo_path, ["rev-list", "--reverse", f"{base_commit}..{end_commit}"])
    if not output:
        return None
    return output.splitlines()[0]


def repository_version_entry(repository: dict[str, Any], version: str) -> dict[str, str]:
    return {
        "name": repository["name"],
        "path": repository["path"],
        "type": repository.get("type", "git-repository"),
        "version": version,
    }


def repository_export_metadata(
    repository: dict[str, Any],
    *,
    release_version: str,
    release_date: str,
    version: str,
) -> str:
    data = {
        "schema_version": 1,
        "release": {
            "version": release_version,
            "date": release_date,
        },
        **repository_version_entry(repository, version),
    }
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def write_release_metadata(repository_dir: Path, metadata: str) -> None:
    metadata_path = repository_dir / EXPORT_VERSION_METADATA
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(metadata, encoding="utf-8")


def write_tar_gz(source_dir: Path, output: Path, arcname: str) -> None:
    tmp_output = output.with_name(f"{output.name}.tmp")
    if tmp_output.exists():
        tmp_output.unlink()
    with tarfile.open(tmp_output, "w:gz") as tar:
        tar.add(source_dir, arcname=arcname)
    tmp_output.replace(output)


def clone_and_archive_repository(
    repository: dict[str, Any],
    *,
    output_dir: Path,
    release_version: str,
    release_date: str,
    head: str,
    branch: str,
) -> Path:
    name = repository["name"]
    repo_path = resolve_repo_path(repository)
    head_short = short_commit(repo_path, head)
    output = output_dir / f"{name}_{release_version}_{head_short}.tar.gz"
    temp_dir = output_dir / f".{name}_{release_version}_{head_short}.clone-tmp"
    clone_dir = temp_dir / name
    metadata = repository_export_metadata(
        repository,
        release_version=release_version,
        release_date=release_date,
        version=head,
    )

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    try:
        temp_dir.mkdir(parents=True)
        run(
            [
                "git",
                "clone",
                "--no-hardlinks",
                "--branch",
                branch,
                str(repo_path),
                str(clone_dir),
            ]
        )
        cloned_head = git_text(clone_dir, ["rev-parse", "HEAD"])
        if cloned_head != head:
            raise RuntimeError(
                f"cloned {name} branch {branch} at {cloned_head}, expected {head}"
            )
        write_release_metadata(clone_dir, metadata)
        write_tar_gz(clone_dir, output, name)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return output


def create_vllm_patch(
    repository: dict[str, Any],
    *,
    output_dir: Path,
) -> Path | None:
    repo_path = resolve_repo_path(repository)
    base_commit = repository["version"]
    end_commit = git_text(repo_path, ["rev-parse", "HEAD"])
    start_commit = first_commit_after(repo_path, base_commit, end_commit)
    if start_commit is None:
        return None
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
    release_date = dt.date.today().isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[Path] = []
    published_manifest: dict[str, Any] = {
        "schema_version": 1,
        "release": {
            "version": release_version,
            "date": release_date,
        },
        "repositories": [],
    }
    for repository in manifest["repositories"]:
        repo_path = resolve_repo_path(repository)
        name = repository["name"]
        head = git_text(repo_path, ["rev-parse", "HEAD"])
        published_manifest["repositories"].append(repository_version_entry(repository, head))
        branch = current_branch(repo_path)
        artifacts.append(
            clone_and_archive_repository(
                repository,
                output_dir=output_dir,
                release_version=release_version,
                release_date=release_date,
                head=head,
                branch=branch,
            )
        )
        if name == VLLM_REPOSITORY_NAME:
            vllm_patch = create_vllm_patch(repository, output_dir=output_dir)
            if vllm_patch is not None:
                artifacts.append(vllm_patch)
    manifest_output = output_dir / PUBLISHED_MANIFEST_NAME
    manifest_output.write_text(
        json.dumps(published_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts.append(manifest_output)
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
