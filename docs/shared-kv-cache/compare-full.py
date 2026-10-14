"""Compare complete-model real-weight outputs and cache ownership reports."""

import argparse
import json
import math
from pathlib import Path


def compare(directory, weights):
    pools = set()
    private_bytes = shared_bytes = 0
    for dp_rank in range(2):
        stem = f"{weights}.json.dp{dp_rank}"
        private_path = directory / f"private-{stem}"
        shared_path = directory / f"shared-{stem}"
        private_output = json.loads(private_path.read_text())
        shared_output = json.loads(shared_path.read_text())
        assert private_output == shared_output, f"DP{dp_rank}: generation mismatch"
        for batch in private_output:
            for request in batch:
                assert request["tokens"]
                assert all(
                    math.isfinite(value)
                    for row in request["logprobs"]
                    for value in row.values()
                )
        private = json.loads(Path(str(private_path) + ".cache.json").read_text())
        shared = json.loads(Path(str(shared_path) + ".cache.json").read_text())
        assert len(private) == len(shared) == 2
        assert all(not r["shared"] for r in private)
        assert all(r["shared"] for r in shared)
        assert all(
            (r["dp_size"], r["tp_size"], r["ep_size"]) == (2, 2, 4)
            for r in private + shared
        )
        assert len({r["bytes"] for r in private + shared}) == 1
        assert len({r["num_blocks"] for r in private + shared}) == 1
        assert shared[0]["inode"] == shared[1]["inode"]
        assert shared[0]["cache_sha256"] == shared[1]["cache_sha256"]
        assert sum(r["writer"] for r in shared) == 1
        pools.add(shared[0]["inode"])
        private_bytes += sum(r["bytes"] for r in private)
        shared_bytes += shared[0]["bytes"]
    assert len(pools) == 2
    assert private_bytes == 2 * shared_bytes
    return dict(
        weights=weights,
        dp=2,
        tp=2,
        ep=4,
        physical_hosts=1,
        exact_output_match=True,
        finite_logprobs=True,
        private_cache_bytes=private_bytes,
        shared_cache_bytes=shared_bytes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--weights", choices=["mxfp4", "fp8"], required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.directory, args.weights), indent=2))
