#!/usr/bin/env python3
"""
Export PyTorch profiler Chrome trace cpu_op events as one row per operator call.

The script reads a profiler trace in .json or .json.gz format, filters events
whose category is cpu_op, and writes a flat CSV table that is easy to inspect
with spreadsheets, pandas, awk, or database import tools.
"""

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path


DEFAULT_COLUMNS = [
    "row",
    "name",
    "cat",
    "ph",
    "pid",
    "tid",
    "ts_us",
    "dur_us",
    "end_us",
    "external_id",
    "record_function_id",
    "ev_idx",
    "input_types",
    "input_dims",
    "input_strides",
    "concrete_inputs",
]


def read_trace(path_str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Input trace file not found: {path}")

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Unsupported trace format: top-level JSON must be an object")

    events = data.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("Unsupported trace format: missing traceEvents list")

    return data


def json_cell(value):
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_row(row_index, event, include_args_json=False):
    args = event.get("args") or {}
    ts = event.get("ts", "")
    dur = event.get("dur", "")
    end = ""
    if isinstance(ts, (int, float)) and isinstance(dur, (int, float)):
        end = ts + dur

    row = {
        "row": row_index,
        "name": event.get("name", ""),
        "cat": event.get("cat", ""),
        "ph": event.get("ph", ""),
        "pid": event.get("pid", ""),
        "tid": event.get("tid", ""),
        "ts_us": ts,
        "dur_us": dur,
        "end_us": end,
        "external_id": args.get("External id", ""),
        "record_function_id": args.get("Record function id", ""),
        "ev_idx": args.get("Ev Idx", ""),
        "input_types": json_cell(args.get("Input type")),
        "input_dims": json_cell(args.get("Input Dims")),
        "input_strides": json_cell(args.get("Input Strides")),
        "concrete_inputs": json_cell(args.get("Concrete Inputs")),
    }

    if include_args_json:
        row["args_json"] = json_cell(args)

    return row


def export_cpu_ops(input_path, output_path, include_args_json=False):
    trace = read_trace(input_path)
    events = trace["traceEvents"]

    columns = list(DEFAULT_COLUMNS)
    if include_args_json:
        columns.append("args_json")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_events = 0
    cpu_op_events = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()

        for event in events:
            total_events += 1
            if event.get("cat") != "cpu_op":
                continue

            cpu_op_events += 1
            writer.writerow(build_row(cpu_op_events, event, include_args_json))

    return total_events, cpu_op_events, output


def default_output_path(input_path):
    path = Path(input_path)
    name = path.name
    for suffix in (".pt.trace.json.gz", ".trace.json.gz", ".json.gz", ".json"):
        if name.endswith(suffix):
            return str(path.with_name(name[: -len(suffix)] + ".cpu_ops.csv"))
    return str(path.with_name(name + ".cpu_ops.csv"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export all cpu_op events from a PyTorch profiler trace to CSV.",
        epilog="""
Examples:
  python export_cpu_ops.py profile.pt.trace.json.gz
  python export_cpu_ops.py profile.pt.trace.json.gz -o cpu_ops.csv
  python export_cpu_ops.py profile.pt.trace.json.gz -o cpu_ops.csv --include-args-json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input PyTorch profiler trace (.json or .json.gz)")
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV path. Defaults to INPUT with .cpu_ops.csv suffix.",
    )
    parser.add_argument(
        "--include-args-json",
        action="store_true",
        help="Also write the full event args object as an args_json column.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or default_output_path(args.input)

    try:
        total_events, cpu_op_events, output_path = export_cpu_ops(
            input_path=args.input,
            output_path=output,
            include_args_json=args.include_args_json,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Input events: {total_events:,}")
    print(f"cpu_op rows: {cpu_op_events:,}")
    print(f"Output CSV: {output_path}")


if __name__ == "__main__":
    main()
