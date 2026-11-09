#!/usr/bin/env python3
"""Summarize ep_moe_collective_probe lines from vLLM rank logs.

The probe's XCPU_EP_PROBE_LOG mode prints, per rank, every v5 dispatch count
exchange (``alltoall sendcounts=[...]``: bytes to each EP peer; 0 = silent,
header bytes = keepalive, larger = payload) and every alltoallv's nonzero
send/recv counts. This script classifies the dispatch lines:

  dense      all peers nonzero (round 0 of a multi-token batch, or a decode
             step whose tokens happen to hit every peer)
  sparse     some peers 0 but not all (final round of chunking, or a decode
             step that skips peers) -- the "not sending" of the new design
  silent     all peers 0 (empty rank / dummy run)
  keepalive  some peer sent exactly the header bytes (a non-final chunk round
             whose records for that peer are exhausted) -- proves multi-round

Usage: analyze_ep_probe.py RUN_DIR [--header-bytes 256]
"""

import argparse
import re
from collections import Counter
from pathlib import Path

ALLTOALL = re.compile(r"\[ep-probe rank=(\d+) line=\d+\] alltoall sendcounts=\[([\d,]*)\]")
ALLTOALLV = re.compile(r"\[ep-probe rank=(\d+) line=\d+\] alltoallv nz_send=(\d+)/(\d+) nz_recv=(\d+)/(\d+)")


def classify(counts, header_bytes):
    if not any(counts):
        return "silent"
    if any(c == header_bytes for c in counts):
        return "keepalive"
    if any(c == 0 for c in counts):
        return "sparse"
    return "dense"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--header-bytes", type=int, default=256,
                        help="header bytes of one v5 message (E=32 -> 256)")
    args = parser.parse_args()

    logs = sorted(args.run_dir.glob("vllm_worker_log_rank*.txt"))
    if not logs:
        logs = sorted(args.run_dir.glob("*.log"))
    for log in logs:
        rounds = Counter()
        nz_send = Counter()
        samples = {}
        burst_pairs = 0  # consecutive dispatch rounds of one layer: multi-round calls
        prev = None
        with log.open() as handle:
            for line in handle:
                match = ALLTOALL.search(line)
                if match:
                    counts = [int(c) for c in match.group(2).split(",") if c]
                    kind = classify(counts, args.header_bytes)
                    rounds[kind] += 1
                    samples.setdefault(kind, counts)
                    # A dense round followed by a sparser one on the same rank is
                    # the round 0 -> final round signature of one chunked layer.
                    if prev is not None and prev[0] == match.group(1):
                        if all(prev[1]) and not all(counts):
                            burst_pairs += 1
                    prev = (match.group(1), counts)
                    continue
                match = ALLTOALLV.search(line)
                if match:
                    nz_send[int(match.group(2))] += 1
        total = sum(rounds.values())
        print(f"== {log.name}: {total} dispatch rounds")
        for kind in ("dense", "keepalive", "sparse", "silent"):
            if rounds[kind]:
                print(f"   {kind:9s} {rounds[kind]:6d}  e.g. {samples[kind]}")
        print(f"   multi-round signatures (dense -> sparser next round): {burst_pairs}")
        print(f"   alltoallv nz_send histogram: {dict(sorted(nz_send.items()))}")


if __name__ == "__main__":
    main()
