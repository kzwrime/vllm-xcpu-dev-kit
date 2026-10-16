"""Compare outputs and physical cache ownership from shared_kv.py."""

import argparse
import json
import math
from pathlib import Path


def compare(private_path, shared_path):
    private = json.loads(private_path.read_text())
    shared = json.loads(shared_path.read_text())
    assert len(private) == len(shared) > 0
    pools = {}
    private_bytes = shared_bytes = 0
    for before, after in zip(private, shared, strict=True):
        assert before["rank"] == after["rank"]
        assert before["output"] == after["output"], (
            f"rank {before['rank']}: output mismatch"
        )
        for batch in after["output"]:
            for request in batch:
                assert request["tokens"]
                assert all(
                    math.isfinite(v)
                    for row in request["logprobs"]
                    for v in row.values()
                )
        a, b = before["cache"], after["cache"]
        assert not a["shared"] and b["shared"]
        assert a["runner"] == b["runner"] == "McpuModelRunnerV2"
        assert all(
            a[k] == b[k] for k in ("tp", "dp", "ep", "pcp", "pcp_rank", "blocks")
        )
        pool = (b["host"], b["device"], b["inode"])
        pools.setdefault(pool, []).append(after)
        private_bytes += a["bytes"]
        shared_bytes += b["bytes"] - b["shared_bytes"]
    for ranks in pools.values():
        reports = [r["cache"] for r in ranks]
        assert len(ranks) == reports[0]["tp"]
        assert len({r["dp"] for r in ranks}) == 1
        assert len({r["pcp_rank"] for r in reports}) == 1
        assert sum(r["writer"] for r in reports) == 1
        assert all(r["cache_sha256"] == reports[0]["cache_sha256"] for r in reports)
        shared_bytes += reports[0]["shared_bytes"]
    if shared[0]["cache"]["tp"] > 1:
        assert shared_bytes < private_bytes, "sharing did not reduce physical storage"
    return {
        "private_bytes": private_bytes,
        "shared_bytes": shared_bytes,
        "pools": len(pools),
        "ranks": len(private),
        "exact_output_match": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private", type=Path)
    parser.add_argument("shared", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.private, args.shared), indent=2))
