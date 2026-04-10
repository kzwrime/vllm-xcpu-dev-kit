#!/usr/bin/env python3
"""
Merge multiple Chrome trace JSON/JSON.GZ files into a single trace.

The merged output is designed to work smoothly with analyze_vllm_enhanced_cn.py:
each input trace is remapped to a stable PID and receives a readable process_name.
"""

import argparse
import copy
import gzip
import json
import re
import sys
from glob import glob
from pathlib import Path


HEADER_WIDTH = 80


def print_banner(title):
    print("=" * HEADER_WIDTH)
    print(title)
    print("=" * HEADER_WIDTH)
    print()


def expand_input_patterns(patterns):
    expanded = []
    seen = set()

    for pattern in patterns:
        matches = sorted(glob(pattern))
        candidates = matches if matches else [pattern]
        for candidate in candidates:
            if candidate not in seen:
                expanded.append(candidate)
                seen.add(candidate)

    return expanded


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

    if path.suffix == ".gz":
        print(f"Writing to {path}...")
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(data, handle, separators=(",", ":"))
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  -> Compressed: {size_mb:.2f} MB")
    else:
        print(f"Writing to {path}...")
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  -> Size: {size_mb:.2f} MB")


def infer_rank_from_path(path_str):
    path = Path(path_str)
    match = re.search(r"world-rank-(\d+)", path.name)
    if match:
        return int(match.group(1))

    match = re.search(r"rank[_-]?(\d+)", path.stem)
    if match:
        return int(match.group(1))

    return None


def infer_process_name(path_str, trace_data, fallback_index):
    distributed_info = trace_data.get("distributedInfo")
    if isinstance(distributed_info, dict) and isinstance(distributed_info.get("rank"), int):
        return f"World-Rank-{distributed_info['rank']}"

    rank = infer_rank_from_path(path_str)
    if rank is not None:
        return f"Rank {rank}"

    return f"Rank {fallback_index}"


def build_process_meta(pid, name):
    base_ts = 0
    return [
        {"name": "process_name", "ph": "M", "ts": base_ts, "pid": pid, "tid": 0, "args": {"name": name}},
        {"name": "process_labels", "ph": "M", "ts": base_ts, "pid": pid, "tid": 0, "args": {"labels": "CPU"}},
        {"name": "process_sort_index", "ph": "M", "ts": base_ts, "pid": pid, "tid": 0, "args": {"sort_index": pid}},
    ]


def remap_event(event, new_pid):
    remapped = copy.deepcopy(event)
    remapped["pid"] = new_pid

    if remapped.get("ph") == "M":
        if remapped.get("name") == "process_name":
            return None
        if remapped.get("name") == "process_sort_index":
            remapped.setdefault("args", {})["sort_index"] = new_pid

    return remapped


def extract_rank_from_trace(input_path, trace_data):
    """Extract rank from trace data or filename."""
    distributed_info = trace_data.get("distributedInfo")
    if isinstance(distributed_info, dict) and isinstance(distributed_info.get("rank"), int):
        return distributed_info['rank']

    rank = infer_rank_from_path(input_path)
    if rank is not None:
        return rank

    return None


def merge_traces(input_paths, output_path, process_names=None):
    print_banner("Merging Chrome Trace Files")

    if process_names and len(process_names) != len(input_paths):
        raise ValueError(
            f"Number of process names ({len(process_names)}) does not match input files ({len(input_paths)})"
        )

    # First pass: read all traces and extract ranks for sorting
    print("Reading trace files to determine rank order...")
    trace_info_list = []
    for input_path in input_paths:
        trace_data = read_trace_file(input_path)
        rank = extract_rank_from_trace(input_path, trace_data)
        trace_info_list.append({
            'path': input_path,
            'data': trace_data,
            'rank': rank if rank is not None else 999999,  # Put files without rank last
        })

    # Sort by rank
    trace_info_list.sort(key=lambda x: x['rank'])
    print(f"Sorted {len(trace_info_list)} traces by rank")

    merged_events = []
    source_summaries = []
    merged_header = None
    total_events = 0

    for index, trace_info in enumerate(trace_info_list):
        input_path = trace_info['path']
        trace_data = trace_info['data']
        rank = trace_info['rank']

        preview_name = process_names[index] if process_names else None
        if preview_name:
            print(f"Processing {Path(input_path).name} as '{preview_name}'...")

        assigned_name = preview_name if preview_name else infer_process_name(input_path, trace_data, index)

        if not preview_name:
            rank_display = rank if rank != 999999 else "N/A"
            print(f"Processing {Path(input_path).name} (rank={rank_display}) as '{assigned_name}'...")
        source_events = trace_data["traceEvents"]
        source_summaries.append(
            {
                "path": str(Path(input_path)),
                "process_name": assigned_name,
                "source_rank": trace_data.get("distributedInfo", {}).get("rank"),
                "event_count": len(source_events),
            }
        )

        if merged_header is None:
            merged_header = {
                key: copy.deepcopy(value)
                for key, value in trace_data.items()
                if key != "traceEvents"
            }
            merged_header["traceEvents"] = []
        else:
            if "distributedInfo" in merged_header:
                merged_header.pop("distributedInfo", None)

        merged_events.extend(build_process_meta(index, assigned_name))

        for event in source_events:
            remapped = remap_event(event, index)
            if remapped is not None:
                merged_events.append(remapped)

        total_events += len(source_events)
        print()

    if merged_header is None:
        raise ValueError("No input traces were loaded")

    merged_header["traceEvents"] = merged_events
    merged_header["mergedFrom"] = source_summaries
    merged_header["mergeHelper"] = {
        "tool": "merge_traces.py",
        "process_count": len(input_paths),
        "original_event_count": total_events,
    }

    print()
    print("=" * HEADER_WIDTH)
    write_trace_file(output_path, merged_header)
    print()
    print("Merge complete!")
    print(f"   Total events: {total_events:,}")
    print(f"   Number of processes: {len(input_paths)}")
    print(f"   Output: {output_path}")
    print("=" * HEADER_WIDTH)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple Chrome trace JSON/JSON.GZ files into a single output file.",
        epilog="""
Examples:
  python merge_traces.py trace_rank0.json.gz trace_rank1.json.gz -o trace_merged.json.gz
  python merge_traces.py "trace_*.json.gz" -o trace_merged.json.gz
  python merge_traces.py trace_*.json.gz -n GPU0 GPU1 GPU2 GPU3 -o trace_merged.json.gz
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="Input trace files or glob patterns")
    parser.add_argument("-o", "--output", required=True, help="Output trace file (.json or .json.gz)")
    parser.add_argument(
        "-n",
        "--names",
        nargs="+",
        help="Optional process names to use in the merged trace, one per input file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths = expand_input_patterns(args.inputs)

    if not input_paths:
        print("No input files matched.", file=sys.stderr)
        sys.exit(1)

    try:
        merge_traces(input_paths, args.output, args.names)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
