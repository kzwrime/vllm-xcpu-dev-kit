from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .matcher import compare_inventories
from .report import write_comparison, write_inventory
from .scanner import scan_repository


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-triton-kernel-analyzer",
        description="Statically inventory and compare @triton.jit kernels in Git repositories.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser("scan", help="write one repository's kernel inventory")
    scan.add_argument("repository", help="path to a vLLM Git working tree")
    scan.add_argument("-o", "--output", required=True, help="output directory")
    scan.add_argument("--exclude", action="append", default=[], metavar="DIR",
                      help="additional directory basename to skip (repeatable)")

    compare = subparsers.add_parser("compare", help="compare two repository working trees")
    compare.add_argument("old_repository")
    compare.add_argument("new_repository")
    compare.add_argument("-o", "--output", required=True, help="output directory")
    compare.add_argument("--exclude", action="append", default=[], metavar="DIR",
                         help="additional directory basename to skip (repeatable)")
    compare.add_argument("--fuzzy-threshold", type=float, default=0.58, metavar="SCORE",
                         help="minimum 0..1 score for move+rename fuzzy matching (default: 0.58)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        excludes = set(args.exclude)
        if args.command == "scan":
            inventory = scan_repository(args.repository, excludes)
            write_inventory(inventory, args.output)
            print(f"scanned {inventory.scanned_python_files} Python files; "
                  f"found {len(inventory.kernels)} kernels; output: {Path(args.output).resolve()}")
            return 0
        if not 0.0 <= args.fuzzy_threshold <= 1.0:
            raise ValueError("--fuzzy-threshold must be between 0 and 1")
        old = scan_repository(args.old_repository, excludes)
        new = scan_repository(args.new_repository, excludes)
        comparison = compare_inventories(old, new, args.fuzzy_threshold)
        write_comparison(comparison, args.output)
        changed = sum(match.changed for match in comparison.matches)
        print(f"old={len(old.kernels)}, new={len(new.kernels)}, changed={changed}; "
              f"report: {Path(args.output).resolve() / 'report.html'}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

