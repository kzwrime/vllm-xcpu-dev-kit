#!/usr/bin/env python3
"""Generate release-note templates and update the repository version manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / ".release" / "repository_versions.json"
DEFAULT_CURRENT_MANIFEST = DEFAULT_MANIFEST.with_name(
    "repository_versions_currently.json"
)
DEFAULT_NOTES_DIR = ROOT / "docs" / "release_notes"


def run_git(repo_path: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git -C {repo_path} {' '.join(args)} failed:\n{proc.stderr.strip()}"
        )
    return proc.stdout.rstrip("\n")


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema_version") != 1:
        raise ValueError(f"{path} has unsupported schema_version: {data.get('schema_version')}")
    if "repositories" not in data or not isinstance(data["repositories"], list):
        raise ValueError(f"{path} does not contain a repositories list")
    return data


def resolve_repo_path(repository: dict[str, Any]) -> Path:
    path = Path(repository["path"])
    if not path.is_absolute():
        path = ROOT / path
    return path


def get_commit_lines(repository: dict[str, Any]) -> list[str]:
    repo_path = resolve_repo_path(repository)
    base = repository["version"]
    fmt = "%h%x09%ad%x09%an%x09%s"
    output = run_git(
        repo_path,
        ["log", "--date=short", f"--pretty=format:{fmt}", f"{base}..HEAD"],
    )
    if not output:
        return []

    lines = []
    for raw in output.splitlines():
        commit, date, author, subject = raw.split("\t", 3)
        lines.append(f"\t- {commit} {date} {author} {subject}")
    return lines


def generate_report_template(manifest: dict[str, Any], release_version: str) -> str:
    from_version = manifest["release"]["version"]
    lines = [
        f"{release_version} 更新说明（{from_version} 至 {release_version}）",
        "",
        "更新重点：",
        "- TODO",
        "",
        "备注：",
        "- TODO",
        "",
        "---",
        "",
    ]

    for repository in manifest["repositories"]:
        lines.append(f"- {repository['name']}")
        commit_lines = get_commit_lines(repository)
        if commit_lines:
            lines.extend(commit_lines)
        else:
            lines.append("\t- 无")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def update_manifest_versions(manifest: dict[str, Any], release_version: str, release_date: str) -> dict[str, Any]:
    updated = {
        "schema_version": 1,
        "release": {
            "version": release_version,
            "date": release_date,
        },
        "repositories": [],
    }

    for repository in manifest["repositories"]:
        repo_path = resolve_repo_path(repository)
        head = run_git(repo_path, ["rev-parse", "HEAD"])
        updated["repositories"].append(
            {
                "name": repository["name"],
                "path": repository["path"],
                "type": repository.get("type", "git-repository"),
                "version": head,
            }
        )

    return updated


def write_text_or_stdout(text: str, output: Path | None) -> None:
    if output is None:
        sys.stdout.write(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], output: Path) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    write_text_or_stdout(text, output)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--release-version",
        default=dt.date.today().strftime("%Y%m%d"),
        help="release version used in note title and output filename",
    )
    parser.add_argument(
        "--release-date",
        default=dt.date.today().isoformat(),
        help="release date written to the repository version manifest",
    )
    parser.add_argument(
        "--generate-template",
        action="store_true",
        help="generate a release-note template from manifest versions to current HEADs",
    )
    parser.add_argument(
        "--update-versions",
        action="store_true",
        help="update the manifest versions to current HEADs",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="release-note output path; defaults to docs/release_notes/<release-version> 更新说明.md",
    )
    parser.add_argument(
        "--current-versions-output",
        type=Path,
        default=DEFAULT_CURRENT_MANIFEST,
        help="current-version manifest output path generated with --generate-template",
    )

    args = parser.parse_args()
    if not args.generate_template and not args.update_versions:
        parser.print_help()
        return 0

    manifest = load_manifest(args.manifest)
    current_versions: dict[str, Any] | None = None

    if args.generate_template:
        output = args.output or DEFAULT_NOTES_DIR / f"{args.release_version} 更新说明.md"
        notes = generate_report_template(manifest, args.release_version)
        write_text_or_stdout(notes, output)
        print(f"generated {output}")
        current_versions = update_manifest_versions(
            manifest, args.release_version, args.release_date
        )
        write_json(current_versions, args.current_versions_output)
        print(f"generated {args.current_versions_output}")

    if args.update_versions:
        data = current_versions or update_manifest_versions(
            manifest, args.release_version, args.release_date
        )
        write_json(data, args.manifest)
        print(f"updated {args.manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
