#!/usr/bin/env python3
"""Reproduce async raw-pointer reads from reused vLLM UVA backing buffers.

This targets the MCPU implementation of vLLM BlockTables.gather_block_tables:
the kernel receives a raw pointer to UvaBackedTensor.gpu. If Python rotates the
UVA pool back to the same host backing buffer before the queued kernel reads it,
the kernel observes the later host contents rather than the launch-time values.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Keep this script useful after the runtime-side mitigation is enabled by
# default. Explicit user settings still win.
os.environ.setdefault("VLLM_XCPU_SYNC_MCPU_UVA_COPY", "0")

import torch
import torch_mcpu  # noqa: F401 - registers the mcpu backend
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor


DEVICE = torch.device("mcpu")


def run_num_blocks_case(
    *,
    max_concurrency: int,
    rotations: int,
    sleep_ms: int,
    polluted_n: int,
) -> list[list[int]]:
    torch.mcpu.synchronize()

    src = torch.tensor(
        [[11, 12, 13, 14], [21, 22, 23, 24]],
        dtype=torch.int32,
        device=DEVICE,
    )
    dst = torch.full((1, 4), -1, dtype=torch.int32, device=DEVICE)
    idx_mapping = torch.tensor([0], dtype=torch.int32, device=DEVICE)

    num_blocks = UvaBackedTensor(
        (1, 2), dtype=torch.int32, max_concurrency=max_concurrency
    )
    num_blocks.np[:] = 0
    num_blocks.np[0, 0] = 2
    num_blocks.copy_to_uva()
    submit_ptr = num_blocks.gpu.data_ptr()

    stream = torch.Stream(device=DEVICE)
    blocker = torch.empty(1, dtype=torch.int64, device=DEVICE)
    with torch.mcpu.stream(stream):
        torch.ops.mcpu.stream_sleep_fill_(blocker, 1, sleep_ms)
        torch.ops.mcpu.vllm_gather_block_tables(
            idx_mapping,
            [src],
            num_blocks.gpu,
            1,
            [dst],
        )

    rotated_ptrs: list[int] = []
    for i in range(rotations):
        num_blocks.np[0, 0] = polluted_n
        num_blocks.copy_to_uva()
        rotated_ptrs.append(num_blocks.gpu.data_ptr())

    stream.synchronize()
    result = dst.cpu().tolist()
    print(
        f"num_blocks: max_concurrency={max_concurrency} rotations={rotations} "
        f"polluted_n={polluted_n} submit_ptr={submit_ptr} "
        f"rotated_ptrs={rotated_ptrs} result={result}"
    )
    return result


def trigger_idx_mapping_oob(sleep_ms: int) -> None:
    torch.mcpu.synchronize()

    src = torch.tensor(
        [[11, 12, 13, 14], [21, 22, 23, 24]],
        dtype=torch.int32,
        device=DEVICE,
    )
    dst = torch.full((1, 4), -1, dtype=torch.int32, device=DEVICE)
    num_blocks = torch.tensor([[2, 2]], dtype=torch.int32, device=DEVICE)

    idx_mapping = UvaBackedTensor((1,), dtype=torch.int32, max_concurrency=1)
    idx_mapping.np[0] = 0
    idx_mapping.copy_to_uva()

    stream = torch.Stream(device=DEVICE)
    blocker = torch.empty(1, dtype=torch.int64, device=DEVICE)
    with torch.mcpu.stream(stream):
        torch.ops.mcpu.stream_sleep_fill_(blocker, 1, sleep_ms)
        torch.ops.mcpu.vllm_gather_block_tables(
            idx_mapping.gpu,
            [src],
            num_blocks,
            1,
            [dst],
        )

    idx_mapping.np[0] = 99
    idx_mapping.copy_to_uva()
    print(
        "About to synchronize. The worker thread is expected to abort with: "
        "idx_mapping contains out-of-range request index"
    )
    stream.synchronize()


def run_idx_mapping_oob_subprocess(sleep_ms: int) -> None:
    cmd = [
        sys.executable,
        __file__,
        "--sleep-ms",
        str(sleep_ms),
        "--trigger-idx-oob",
    ]
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print("idx_mapping_oob: child exit code", completed.returncode)
    interesting = [
        line
        for line in completed.stdout.splitlines()
        if "idx_mapping contains out-of-range request index" in line
        or "About to synchronize" in line
        or "Aborted" in line
        or "terminate called" in line
    ]
    for line in interesting[-12:]:
        print("idx_mapping_oob:", line)
    if completed.returncode == 0:
        raise SystemExit("idx_mapping_oob did not fail")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleep-ms", type=int, default=300)
    parser.add_argument(
        "--trigger-idx-oob",
        action="store_true",
        help=(
            "intentionally abort the process by overwriting a submitted "
            "idx_mapping UVA backing with an out-of-range value"
        ),
    )
    args = parser.parse_args()

    if args.trigger_idx_oob:
        trigger_idx_mapping_oob(args.sleep_ms)
        return

    expected = [[11, 12, 0, 0]]
    control = run_num_blocks_case(
        max_concurrency=2,
        rotations=0,
        sleep_ms=args.sleep_ms,
        polluted_n=0,
    )
    if control != expected:
        raise SystemExit(f"control failed: expected {expected}, got {control}")

    forced_zero = run_num_blocks_case(
        max_concurrency=1,
        rotations=1,
        sleep_ms=args.sleep_ms,
        polluted_n=0,
    )
    default_zero = run_num_blocks_case(
        max_concurrency=2,
        rotations=2,
        sleep_ms=args.sleep_ms,
        polluted_n=0,
    )
    default_long = run_num_blocks_case(
        max_concurrency=2,
        rotations=2,
        sleep_ms=args.sleep_ms,
        polluted_n=4,
    )

    if forced_zero == expected or default_zero == expected or default_long == expected:
        raise SystemExit(
            "race did not reproduce; increase --sleep-ms or reduce system load"
        )

    print(f"expected without backing reuse: {expected}")
    run_idx_mapping_oob_subprocess(args.sleep_ms)


if __name__ == "__main__":
    main()
