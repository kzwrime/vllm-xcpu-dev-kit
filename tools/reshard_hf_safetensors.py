#!/usr/bin/env python3
"""Copy a Hugging Face model directory and reshard safetensors weights."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from safetensors import safe_open
from safetensors.torch import save_file


@dataclass
class ReshardResult:
    shard_sizes: list[tuple[str, int]]
    largest_tensor: tuple[str, int]
    total_size: int


def parse_size(size: str) -> int:
    try:
        from transformers.utils.hub import convert_file_size_to_int

        return convert_file_size_to_int(size)
    except Exception:
        units = {"KB": 10**3, "MB": 10**6, "GB": 10**9, "KIB": 2**10, "MIB": 2**20, "GIB": 2**30}
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


def reshard(src: Path, dst: Path, target_shard_size: int) -> ReshardResult:
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

    for shard_name, ordered_keys in load_weight_order(src):
        shard_path = src / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            keys = ordered_keys or list(f.keys())
            for key in keys:
                tensor = f.get_tensor(key)
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

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with (dst / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
        f.write("\n")

    final_shard_sizes = [(name.replace("-of-99999", f"-of-{len(shard_sizes):05d}"), size) for name, size in shard_sizes]
    return ReshardResult(final_shard_sizes, largest_tensor, total_size)


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
    parser.add_argument("--max-shard-size", dest="target_shard_size", help=argparse.SUPPRESS)
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"source directory does not exist: {src}")

    target_shard_size = parse_size(args.target_shard_size)
    result = reshard(src, dst, target_shard_size)
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
    print()
    print("note: shard size is approximate; the smallest achievable upper bound is limited by the largest single tensor,")
    print("because safetensors/Hugging Face indexes do not split one tensor across multiple shard files.")


if __name__ == "__main__":
    main()
