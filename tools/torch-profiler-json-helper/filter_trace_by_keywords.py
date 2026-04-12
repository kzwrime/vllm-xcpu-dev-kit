#!/usr/bin/env python3
"""
Filter a Chrome trace JSON/JSON.GZ file by one or more operator keywords.

The filtered output keeps metadata events so the result remains easy to open in
Chrome trace viewers and can still be consumed by the existing analysis script.
"""

import argparse
import copy
import gzip
import json
import sys
from pathlib import Path


HEADER_WIDTH = 80
METADATA_PHASES = {"M"}
ALWAYS_KEEP_EVENT_NAMES = {
    "process_name",
    "process_labels",
    "process_sort_index",
    "thread_name",
    "thread_sort_index",
}


def print_banner(title):
    print("=" * HEADER_WIDTH)
    print(title)
    print("=" * HEADER_WIDTH)
    print()


def read_trace_file(path_str):
    path = Path(path_str)
    print(f"Reading {path}...")

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
        print(f"  -> Decompressed {path.name}")
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported trace format in {path}: top-level JSON must be an object")

    events = data.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"Unsupported trace format in {path}: missing traceEvents list")

    print(f"  -> Loaded {len(events):,} events")
    return data


def write_trace_file(path_str, data):
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing to {path}...")
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(data, handle, separators=(",", ":"))
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  -> Compressed: {size_mb:.2f} MB")
    else:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  -> Size: {size_mb:.2f} MB")


def event_matches_keywords(event, keywords, case_sensitive):
    phase = event.get("ph")
    name = event.get("name")

    if phase in METADATA_PHASES:
        return True

    if name in ALWAYS_KEEP_EVENT_NAMES:
        return True

    if not isinstance(name, str):
        return False

    search_name = name if case_sensitive else name.lower()
    for keyword in keywords:
        search_keyword = keyword if case_sensitive else keyword.lower()
        if search_keyword in search_name:
            return True

    return False


def filter_trace_by_keywords(input_path, output_path, keywords, case_sensitive=False):
    print_banner("Filtering Chrome Trace By Keywords")

    trace_data = read_trace_file(input_path)
    source_events = trace_data["traceEvents"]

    print(
        "Keywords: "
        + ", ".join(keywords)
        + f" (case sensitive: {'yes' if case_sensitive else 'no'})"
    )
    print("Keeping metadata events and trace events whose name matches any keyword.")
    print()

    filtered_events = []
    matched_event_count = 0
    metadata_event_count = 0

    for event in source_events:
        if event_matches_keywords(event, keywords, case_sensitive):
            filtered_events.append(event)

            if event.get("ph") in METADATA_PHASES or event.get("name") in ALWAYS_KEEP_EVENT_NAMES:
                metadata_event_count += 1
            else:
                matched_event_count += 1

    result = {key: copy.deepcopy(value) for key, value in trace_data.items() if key != "traceEvents"}
    result["traceEvents"] = filtered_events
    result["filterHelper"] = {
        "tool": "filter_trace_by_keywords.py",
        "input": str(Path(input_path)),
        "keywords": list(keywords),
        "case_sensitive": case_sensitive,
        "original_event_count": len(source_events),
        "filtered_event_count": len(filtered_events),
        "matched_trace_event_count": matched_event_count,
        "kept_metadata_event_count": metadata_event_count,
    }

    print("=" * HEADER_WIDTH)
    write_trace_file(output_path, result)
    print()
    print("Filter complete!")
    print(f"   Original events: {len(source_events):,}")
    print(f"   Matched trace events: {matched_event_count:,}")
    print(f"   Kept metadata events: {metadata_event_count:,}")
    print(f"   Output events: {len(filtered_events):,}")
    print(f"   Output: {output_path}")
    print("=" * HEADER_WIDTH)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter a trace JSON/JSON.GZ file by one or more operator keywords.",
        epilog="""
Examples:
  python filter_trace_by_keywords.py trace_merged.json.gz -k alltoallv -o trace_alltoallv.json.gz
  python filter_trace_by_keywords.py trace_merged.json -k allreduce alltoallv -o trace_comm.json
  python filter_trace_by_keywords.py trace_merged.json.gz -k AllToAllV -o trace.json.gz --case-sensitive
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input trace file (.json or .json.gz)")
    parser.add_argument(
        "-k",
        "--keywords",
        nargs="+",
        required=True,
        help="One or more keywords used to match trace event names",
    )
    parser.add_argument("-o", "--output", required=True, help="Output trace file (.json or .json.gz)")
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Enable case-sensitive keyword matching",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        filter_trace_by_keywords(
            input_path=args.input,
            output_path=args.output,
            keywords=args.keywords,
            case_sensitive=args.case_sensitive,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
