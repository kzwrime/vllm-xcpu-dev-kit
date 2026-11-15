#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Download nvidia/SPEED-Bench parquet splits (Method 2), filter masked rows,
convert subsets to vLLM SpeedBench JSONL format, and optionally archive them.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import shutil
import tarfile
from typing import Any

import pandas as pd

DEFAULT_SUBSETS = (
    "qualitative",
    "throughput_1k",
    "throughput_2k",
    "throughput_8k",
    "throughput_16k",
    "throughput_32k",
)
MASK_MARKER = "FULL BENCHMARK DATA SHOULD BE FETCHED"


def _normalize_optional(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _normalize_turns(raw_turns: Any) -> list[str]:
    if raw_turns is None:
        return []
    if hasattr(raw_turns, "tolist"):
        raw_turns = raw_turns.tolist()
    if not isinstance(raw_turns, (list, tuple)):
        return []
    turns: list[str] = []
    for turn in raw_turns:
        if turn is None:
            continue
        text = str(turn)
        if not text.strip():
            return []
        turns.append(text)
    return turns


def download_speed_bench_raw(
    raw_dir: Path,
    subsets: list[str],
    repo_id: str = "nvidia/SPEED-Bench",
) -> None:
    """Download selected SPEED-Bench parquet subsets from HuggingFace."""
    # huggingface_hub evaluates HF_HUB_OFFLINE at import time, so clear offline
    # variables before importing snapshot_download.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)

    from huggingface_hub import snapshot_download

    allow_patterns = ["README.md"]
    for subset in subsets:
        allow_patterns.append(f"{subset}/*")

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} subsets {subsets} into {raw_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(raw_dir),
        allow_patterns=allow_patterns,
    )
    print(f"Download completed: {raw_dir}")


def convert_subset(
    raw_dir: Path,
    output_dir: Path,
    subset: str,
) -> dict[str, Any]:
    """Convert one parquet subset directory into <output_dir>/<subset>.jsonl."""
    parquet_pattern = str(raw_dir / subset / "test-*.parquet")
    parquet_files = sorted(glob.glob(parquet_pattern))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found for subset '{subset}' matching {parquet_pattern}"
        )

    frames = [pd.read_parquet(path) for path in parquet_files]
    df = pd.concat(frames, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{subset}.jsonl"

    total_rows = len(df)
    kept_rows = 0
    masked_rows = 0
    empty_rows = 0
    category_counts: dict[str, int] = {}

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in df.to_dict(orient="records"):
            turns = _normalize_turns(row.get("turns"))
            if not turns:
                empty_rows += 1
                continue

            if any(MASK_MARKER in turn for turn in turns):
                masked_rows += 1
                continue

            category = _normalize_optional(row.get("category")) or "unknown"
            category_counts[category] = category_counts.get(category, 0) + 1

            raw_multiturn = row.get("multiturn")
            multiturn = (
                bool(raw_multiturn)
                if raw_multiturn is not None
                else len(turns) > 1
            )

            record = {
                "question_id": _normalize_optional(row.get("question_id")) or "",
                "category": category,
                "sub_category": _normalize_optional(row.get("sub_category")),
                "source": _normalize_optional(row.get("source")),
                "difficulty": _normalize_optional(row.get("difficulty")),
                "multiturn": multiturn,
                "messages": [{"role": "user", "content": turn} for turn in turns],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept_rows += 1

    size_bytes = jsonl_path.stat().st_size
    stats = {
        "subset": subset,
        "jsonl_path": str(jsonl_path),
        "size_bytes": size_bytes,
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "masked_rows": masked_rows,
        "empty_rows": empty_rows,
        "category_counts": dict(sorted(category_counts.items())),
    }
    print(
        f"[{subset}] total={total_rows}, kept={kept_rows}, "
        f"masked={masked_rows}, empty={empty_rows}, "
        f"size={size_bytes / (1024 * 1024):.2f} MiB -> {jsonl_path}"
    )
    return stats


def create_archive(output_dir: Path, archive_path: Path) -> int:
    """Create a standalone .tar.gz archive containing output_dir."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_archive = archive_path.with_name(archive_path.name + ".tmp")
    with tarfile.open(tmp_archive, "w:gz") as tar:
        tar.add(str(output_dir), arcname=output_dir.name)
    tmp_archive.replace(archive_path)
    size_bytes = archive_path.stat().st_size
    print(
        f"Created standalone archive: {archive_path} "
        f"({size_bytes / (1024 * 1024):.2f} MiB)"
    )
    return size_bytes


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Prepare local SPEED-Bench JSONL files from HuggingFace parquet "
            "splits (Method 2: filter masked placeholder samples)."
        )
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download nvidia/SPEED-Bench parquet files from HuggingFace.",
    )
    parser.add_argument(
        "--repo-id",
        default="nvidia/SPEED-Bench",
        help="HuggingFace dataset repository id (default: nvidia/SPEED-Bench).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=script_dir / "data" / "speed_bench_raw",
        help="Directory holding raw parquet subsets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "data" / "speed_bench",
        help="Output directory for <subset>.jsonl files.",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default=",".join(DEFAULT_SUBSETS),
        help="Comma-separated subset list.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Optional output .tar.gz path bundling output_dir.",
    )
    parser.add_argument(
        "--clean-raw",
        action="store_true",
        help="Remove --raw-dir after successful conversion and archiving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    if not subsets:
        raise ValueError("No subsets specified.")

    if args.download:
        download_speed_bench_raw(args.raw_dir, subsets, repo_id=args.repo_id)

    subset_stats = [
        convert_subset(args.raw_dir, args.output_dir, subset)
        for subset in subsets
    ]

    empty_subsets = [s["subset"] for s in subset_stats if s["kept_rows"] == 0]
    manifest = {
        "repo_id": args.repo_id,
        "method": "parquet_unmasked_filter",
        "subsets": subset_stats,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote manifest: {manifest_path}")

    if args.archive is not None:
        create_archive(args.output_dir, args.archive)

    if args.clean_raw and args.raw_dir.exists():
        shutil.rmtree(args.raw_dir)
        print(f"Removed raw directory: {args.raw_dir}")

    if empty_subsets:
        raise RuntimeError(
            f"Empty subsets after filtering masked rows: {empty_subsets}"
        )


if __name__ == "__main__":
    main()
