#!/usr/bin/env python3
"""Copy a Hugging Face model directory and reshard safetensors weights."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_ENGRAM_TABLE_RE = re.compile(
    r"^(?P<base>(?:.+\.)?layers\.\d+\.engram\.embed\.(?:weight|scale))$"
)
_ENGRAM_SLICE_RE = re.compile(
    r"^(?P<base>(?:.+\.)?layers\.\d+\.engram\.embed\.(?:weight|scale))"
    r"\.row_slice\.(?P<start>\d+)_(?P<end>\d+)$"
)
_MIN_AUTO_ENGRAM_SHARD_BYTES = 16 * 1024 * 1024
_DEFAULT_ENGRAM_SHARD_SIZE = "1GB"


@dataclass
class ReshardResult:
    shard_sizes: list[tuple[str, int]]
    largest_tensor: tuple[str, int]
    total_size: int
    engram_row_slices: dict[str, dict] = field(default_factory=dict)


def parse_size(size: str) -> int:
    try:
        from transformers.utils.hub import convert_file_size_to_int

        return convert_file_size_to_int(size)
    except (ImportError, ValueError):
        units = {
            "KIB": 2**10,
            "MIB": 2**20,
            "GIB": 2**30,
            "KB": 10**3,
            "MB": 10**6,
            "GB": 10**9,
            "B": 1,
        }
        normalized = size.strip().upper()
        for suffix, multiplier in units.items():
            if normalized.endswith(suffix):
                return int(float(normalized[: -len(suffix)].strip()) * multiplier)
        return int(normalized)


def tensor_nbytes(tensor) -> int:
    return tensor.numel() * tensor.element_size()


def copy_non_weight_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if not path.is_file():
            continue
        if path.name.endswith(".safetensors") or path.name == "model.safetensors.index.json":
            continue
        shutil.copy2(path, dst / path.name)


def load_weight_order(src: Path) -> list[tuple[str, list[str]]]:
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        shards = sorted(src.glob("*.safetensors"))
        return [(shard.name, []) for shard in shards]

    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f, object_pairs_hook=OrderedDict)

    by_file: OrderedDict[str, list[str]] = OrderedDict()
    for weight_name, shard_name in index["weight_map"].items():
        by_file.setdefault(shard_name, []).append(weight_name)
    return list(by_file.items())


def load_source_engram_metadata(src: Path) -> dict[str, dict]:
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    meta = index.get("metadata", {})
    raw = meta.get("engram_row_slices", {})
    return dict(raw) if isinstance(raw, dict) else {}


def flush_shard(
    tensors: OrderedDict,
    out_dir: Path,
    weight_map: dict[str, str],
    shard_sizes: list[tuple[str, int]],
    shard_index: int,
    total_shards_hint: int = 99999,
) -> None:
    if not tensors:
        return
    name = f"model-{shard_index:05d}-of-{total_shards_hint:05d}.safetensors"
    save_file(dict(tensors), out_dir / name)
    shard_size = (out_dir / name).stat().st_size
    for key in tensors:
        weight_map[key] = name
    shard_sizes.append((name, shard_size))
    tensors.clear()


def normalize_shard_names(out_dir: Path, weight_map: dict[str, str], shard_count: int) -> None:
    rename_map = {}
    for old in sorted({name for name in weight_map.values()}):
        shard_number = int(old.split("-")[1])
        new = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
        rename_map[old] = new
        if old != new:
            (out_dir / old).rename(out_dir / new)
    for key, old in list(weight_map.items()):
        weight_map[key] = rename_map[old]


def format_gib(size: float) -> str:
    return f"{size / (1024**3):.2f} GiB"


def _scan_source_engram_slices(
    src: Path, weight_order: list[tuple[str, list[str]]]
) -> dict[str, list[tuple[int, int, str, str]]]:
    """Collect pre-split ``.row_slice.<start>_<end>`` entries grouped by base name."""
    by_base: dict[str, list[tuple[int, int, str, str]]] = {}
    for shard_name, ordered_keys in weight_order:
        if ordered_keys:
            keys = ordered_keys
        else:
            with safe_open(str(src / shard_name), framework="pt", device="cpu") as f:
                keys = list(f.keys())
        for key in keys:
            match = _ENGRAM_SLICE_RE.match(key)
            if match is None:
                continue
            base = match.group("base")
            start = int(match.group("start"))
            end = int(match.group("end"))
            if start >= end:
                raise ValueError(f"invalid Engram row slice [{start}, {end}) in {key!r}")
            by_base.setdefault(base, []).append((start, end, shard_name, key))

    for base, chunks in by_base.items():
        chunks.sort(key=lambda item: item[0])
        if chunks[0][0] != 0:
            raise ValueError(f"Engram row slices for {base!r} start at {chunks[0][0]}, expected 0")
        for (prev_s, prev_e, _, _), (curr_s, curr_e, _, _) in pairwise(chunks):
            if prev_e != curr_s:
                raise ValueError(
                    f"Engram row slices for {base!r} are not contiguous: "
                    f"[{prev_s}, {prev_e}) followed by [{curr_s}, {curr_e})"
                )
    return by_base


def _split_unsplit_engram_tensor(
    key: str,
    ts,
    dst: Path,
    engram_shard_size: int,
    weight_map: dict[str, str],
    shard_sizes: list[tuple[str, int]],
    shard_index: int,
    largest_tensor: tuple[str, int],
) -> tuple[int, tuple[str, int], int, dict]:
    shape = list(ts.get_shape())
    if len(shape) != 2 or shape[0] <= 0:
        raise ValueError(f"Engram table {key!r} must be a non-empty 2D tensor, got {shape}")

    first_row = ts[0:1]
    row_nbytes = tensor_nbytes(first_row)
    dtype_str = str(first_row.dtype).removeprefix("torch.")
    del first_row

    total_rows = shape[0]
    rows_per_chunk = max(1, engram_shard_size // row_nbytes)
    added_bytes = 0
    num_chunks = 0
    for row_start in range(0, total_rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, total_rows)
        chunk_key = f"{key}.row_slice.{row_start}_{row_end}"
        chunk_tensor = ts[row_start:row_end].clone()
        size = tensor_nbytes(chunk_tensor)
        if size > largest_tensor[1]:
            largest_tensor = (chunk_key, size)
        added_bytes += size
        flush_shard(
            OrderedDict([(chunk_key, chunk_tensor)]),
            dst,
            weight_map,
            shard_sizes,
            shard_index,
        )
        shard_index += 1
        num_chunks += 1
        del chunk_tensor

    meta = {
        "shape": shape,
        "dtype": dtype_str,
        "num_chunks": num_chunks,
        "rows_per_chunk": rows_per_chunk,
    }
    return shard_index, largest_tensor, added_bytes, meta


def _rechunk_engram_slices(
    base: str,
    src_chunks: list[tuple[int, int, str, str]],
    src: Path,
    dst: Path,
    engram_shard_size: int,
    weight_map: dict[str, str],
    shard_sizes: list[tuple[str, int]],
    shard_index: int,
    largest_tensor: tuple[str, int],
) -> tuple[int, tuple[str, int], int, dict]:
    first_start, first_end, first_shard, first_key = src_chunks[0]
    with safe_open(str(src / first_shard), framework="pt", device="cpu") as sf:
        ts = sf.get_slice(first_key)
        c_shape = list(ts.get_shape())
        if len(c_shape) != 2 or c_shape[0] != first_end - first_start:
            raise ValueError(f"unexpected shape {c_shape} for {first_key!r}")
        first_row = ts[0:1]
        row_nbytes = tensor_nbytes(first_row)
        orig_dtype = first_row.dtype
        dtype_str = str(orig_dtype).removeprefix("torch.")
        tail_shape = c_shape[1:]
        del first_row, ts
    del sf

    total_rows = src_chunks[-1][1]
    rows_per_chunk = max(1, engram_shard_size // row_nbytes)
    added_bytes = 0
    num_chunks = 0

    for out_start in range(0, total_rows, rows_per_chunk):
        out_end = min(out_start + rows_per_chunk, total_rows)
        out_raw = torch.empty((out_end - out_start, *tail_shape), dtype=torch.uint8)
        for c_start, c_end, c_shard, c_key in src_chunks:
            if c_end <= out_start or c_start >= out_end:
                continue
            ov_lo = max(out_start, c_start)
            ov_hi = min(out_end, c_end)
            with safe_open(str(src / c_shard), framework="pt", device="cpu") as sf:
                ts = sf.get_slice(c_key)
                sub = ts[ov_lo - c_start : ov_hi - c_start]
                out_raw[ov_lo - out_start : ov_hi - out_start].copy_(sub.view(torch.uint8))
                del sub, ts
            del sf

        chunk_tensor = out_raw.view(orig_dtype)
        del out_raw
        chunk_key = f"{base}.row_slice.{out_start}_{out_end}"
        size = tensor_nbytes(chunk_tensor)
        if size > largest_tensor[1]:
            largest_tensor = (chunk_key, size)
        added_bytes += size
        flush_shard(
            OrderedDict([(chunk_key, chunk_tensor)]),
            dst,
            weight_map,
            shard_sizes,
            shard_index,
        )
        shard_index += 1
        num_chunks += 1
        del chunk_tensor

    meta = {
        "shape": [total_rows, *tail_shape],
        "dtype": dtype_str,
        "num_chunks": num_chunks,
        "rows_per_chunk": rows_per_chunk,
    }
    return shard_index, largest_tensor, added_bytes, meta


def reshard(
    src: Path,
    dst: Path,
    target_shard_size: int,
    engram_shard_size: int | None = None,
) -> ReshardResult:
    copy_non_weight_files(src, dst)

    for stale in dst.glob("*.safetensors"):
        stale.unlink()
    stale_index = dst / "model.safetensors.index.json"
    if stale_index.exists():
        stale_index.unlink()

    current_tensors: OrderedDict[str, object] = OrderedDict()
    current_size = 0
    total_size = 0
    shard_index = 1
    shard_sizes: list[tuple[str, int]] = []
    weight_map: OrderedDict[str, str] = OrderedDict()
    largest_tensor = ("", 0)
    engram_row_slices: dict[str, dict] = {}

    weight_order = load_weight_order(src)
    src_engram_meta = load_source_engram_metadata(src)
    src_split_slices = (
        _scan_source_engram_slices(src, weight_order)
        if engram_shard_size is not None
        else {}
    )
    rechunked_bases: set[str] = set()

    for shard_name, ordered_keys in weight_order:
        shard_path = src / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            keys = ordered_keys or list(f.keys())
            for key in keys:
                slice_match = _ENGRAM_SLICE_RE.match(key)
                if slice_match is not None:
                    base = slice_match.group("base")
                    if engram_shard_size is not None:
                        if base in rechunked_bases:
                            continue
                        rechunked_bases.add(base)
                        if current_tensors:
                            flush_shard(
                                current_tensors, dst, weight_map, shard_sizes, shard_index
                            )
                            shard_index += 1
                            current_size = 0
                        shard_index, largest_tensor, added_bytes, meta = (
                            _rechunk_engram_slices(
                                base,
                                src_split_slices[base],
                                src,
                                dst,
                                engram_shard_size,
                                weight_map,
                                shard_sizes,
                                shard_index,
                                largest_tensor,
                            )
                        )
                        total_size += added_bytes
                        engram_row_slices[base] = meta
                        continue

                    # Preserve pre-split Engram slices in exclusive shard files.
                    if current_tensors:
                        flush_shard(
                            current_tensors, dst, weight_map, shard_sizes, shard_index
                        )
                        shard_index += 1
                        current_size = 0
                    tensor = f.get_tensor(key).clone()
                    size = tensor_nbytes(tensor)
                    if size > largest_tensor[1]:
                        largest_tensor = (key, size)
                    total_size += size
                    flush_shard(
                        OrderedDict([(key, tensor)]),
                        dst,
                        weight_map,
                        shard_sizes,
                        shard_index,
                    )
                    shard_index += 1
                    del tensor
                    if base in src_engram_meta and base not in engram_row_slices:
                        engram_row_slices[base] = dict(src_engram_meta[base])
                    continue

                if engram_shard_size is not None and _ENGRAM_TABLE_RE.match(key):
                    if current_tensors:
                        flush_shard(
                            current_tensors, dst, weight_map, shard_sizes, shard_index
                        )
                        shard_index += 1
                        current_size = 0
                    ts = f.get_slice(key)
                    shard_index, largest_tensor, added_bytes, meta = (
                        _split_unsplit_engram_tensor(
                            key,
                            ts,
                            dst,
                            engram_shard_size,
                            weight_map,
                            shard_sizes,
                            shard_index,
                            largest_tensor,
                        )
                    )
                    del ts
                    total_size += added_bytes
                    engram_row_slices[key] = meta
                    continue

                tensor = f.get_tensor(key).clone()
                size = tensor_nbytes(tensor)
                if size > largest_tensor[1]:
                    largest_tensor = (key, size)

                if current_tensors and current_size + size > target_shard_size:
                    flush_shard(current_tensors, dst, weight_map, shard_sizes, shard_index)
                    shard_index += 1
                    current_size = 0

                current_tensors[key] = tensor
                current_size += size
                total_size += size

                if current_size >= target_shard_size:
                    flush_shard(current_tensors, dst, weight_map, shard_sizes, shard_index)
                    shard_index += 1
                    current_size = 0

    flush_shard(current_tensors, dst, weight_map, shard_sizes, shard_index)
    normalize_shard_names(dst, weight_map, len(shard_sizes))

    metadata: dict[str, object] = {"total_size": total_size}
    if engram_row_slices:
        metadata["engram_row_slices"] = engram_row_slices
    index = {
        "metadata": metadata,
        "weight_map": weight_map,
    }
    with (dst / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
        f.write("\n")

    final_shard_sizes = [
        (name.replace("-of-99999", f"-of-{len(shard_sizes):05d}"), size)
        for name, size in shard_sizes
    ]
    return ReshardResult(final_shard_sizes, largest_tensor, total_size, engram_row_slices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Source Hugging Face model directory")
    parser.add_argument("dst", type=Path, help="Destination model directory")
    parser.add_argument(
        "--target-shard-size",
        dest="target_shard_size",
        default="1GB",
        help="Desired approximate shard size, e.g. 1GB or 1024MiB",
    )
    parser.add_argument(
        "--split-engram",
        action="store_true",
        help="Pre-split Engram embedding tables (*.engram.embed.{weight,scale}) along row dim 0 into exclusive shard files",
    )
    parser.add_argument(
        "--engram-shard-size",
        dest="engram_shard_size",
        default=None,
        help="Target max bytes per Engram row-slice shard (e.g. 1GB or 512MiB); implies --split-engram",
    )
    parser.add_argument("--max-shard-size", dest="target_shard_size", help=argparse.SUPPRESS)
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"source directory does not exist: {src}")

    target_shard_size = parse_size(args.target_shard_size)
    engram_shard_size: int | None = None
    if args.engram_shard_size is not None:
        engram_shard_size = parse_size(args.engram_shard_size)
        if engram_shard_size <= 0:
            raise SystemExit(f"--engram-shard-size must be positive, got {engram_shard_size}")
    elif args.split_engram:
        if target_shard_size >= _MIN_AUTO_ENGRAM_SHARD_BYTES:
            engram_shard_size = target_shard_size
        else:
            engram_shard_size = parse_size(_DEFAULT_ENGRAM_SHARD_SIZE)
            print(
                f"note: --target-shard-size ({target_shard_size} B) < 16 MiB; "
                f"using default Engram shard size {_DEFAULT_ENGRAM_SHARD_SIZE} instead."
            )

    result = reshard(src, dst, target_shard_size, engram_shard_size=engram_shard_size)
    print(f"wrote {len(result.shard_sizes)} safetensors shards to {dst}")
    for name, size in result.shard_sizes:
        print(f"{name}\t{format_gib(size)}")

    sizes = [size for _, size in result.shard_sizes]
    print()
    print("shard size summary:")
    print(f"target\t{format_gib(target_shard_size)}")
    print(f"min\t{format_gib(min(sizes))}")
    print(f"median\t{format_gib(statistics.median(sizes))}")
    print(f"max\t{format_gib(max(sizes))}")
    print(f"largest tensor\t{format_gib(result.largest_tensor[1])}\t{result.largest_tensor[0]}")
    if result.engram_row_slices:
        total_chunks = sum(info["num_chunks"] for info in result.engram_row_slices.values())
        print(
            f"engram split\t{len(result.engram_row_slices)} tables -> {total_chunks} exclusive row-slice shards"
        )
    else:
        print()
        print("note: shard size is approximate; the smallest achievable upper bound is limited by the largest single tensor,")
        print("because safetensors/Hugging Face indexes do not split one tensor across multiple shard files.")


if __name__ == "__main__":
    main()
