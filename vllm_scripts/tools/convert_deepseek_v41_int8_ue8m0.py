#!/usr/bin/env python3
"""Convert DeepSeek-V4.1-Flash checkpoints (FP8 [32,32] UE8M0 + MXFP4 experts)
to XCPU Int8 [-128, 127] + UE8M0 [16, 16] (W8A16) for linear projections.

Conversion scope:
  - Converted to Int8 ("I8") + UE8M0 uint8 ("U8") [16, 16] `.scale`:
      All 2D FP8 (F8_E4M3) linear projection weights across backbone layers and
      MTP/DSpark layers:
        * attn.{wq_a, wkv, wq_b, wo_a, wo_b}
        * attn.indexer.wq_b
        * engram.wkv
        * ffn.shared_experts.{w1, w2, w3}
        * mtp.0.main_proj
      (355 projection matrices in DeepSeek-V4.1-Flash-1017.)
  - Preserved bit-for-bit verbatim (streaming raw byte copy):
      * Routed MoE MXFP4 experts (`*.experts.*.w{1,2,3}.{weight,scale}`)
      * Engram FP8 lookup tables (`*engram.embed.{weight,scale}`)
      * All BF16 / FP32 tensors (`embed`, `head`, `norm`, `compressor.*`,
        `indexer.{wk,weights_proj,k_norm}`, `ffn.gate.*`, `hc_*`, `attn_sink`,
        `markov_head.*`, `confidence_head.*`, `vision.*`, `aligner.*`, etc.)

Because Engram shards (`model-00047/00048`) are ~94.6 GiB each and routed-expert
shards are ~99% unchanged MXFP4 payloads, this tool uses a streaming safetensors
rewriter: only the small FP8 linear tensors are loaded into Python/PyTorch, while
all unchanged tensors are copied as coalesced raw byte spans directly between
file descriptors. Peak RAM stays well under 1 GiB per worker even on 95 GiB
shards.

Usage:
  python convert_deepseek_v41_int8_ue8m0.py \
      --input /root/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master \
      --output /cache/models/deepseek-ai/DeepSeek-V4.1-Flash-INT8-UE8M0-16x16 \
      --block-size 16 16 --workers 12
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

UE8M0_MIN_FAST = 1
UE8M0_MAX_FAST = 247
UE8M0_ONE = 127

_COPY_CHUNK_BYTES = 64 * 1024 * 1024  # 64 MiB streaming copy buffer

_ENGRAM_EMBED_WEIGHT_SUFFIXES = (
    ".engram.embed.weight",
    ".engram.embed_tokens.weight",
)
_ENGRAM_EMBED_WEIGHT_EXACT = frozenset(
    {
        "engram.embed.weight",
        "engram.embed_tokens.weight",
    }
)


def _norm_block_size(block_size: list[int] | tuple[int, int]) -> tuple[int, int]:
    if len(block_size) != 2:
        raise ValueError(f"--block-size expects 2 ints, got {block_size}")
    bn, bk = int(block_size[0]), int(block_size[1])
    if bn <= 0 or bk <= 0:
        raise ValueError(f"block_size dimensions must be positive, got {(bn, bk)}")
    return bn, bk


def is_engram_embed_weight(name: str) -> bool:
    return name in _ENGRAM_EMBED_WEIGHT_EXACT or name.endswith(
        _ENGRAM_EMBED_WEIGHT_SUFFIXES
    )


def is_target_fp8_linear(name: str, meta: dict[str, Any]) -> bool:
    """Return True iff tensor is a 2D FP8 linear weight to convert to INT8."""
    if not name.endswith(".weight"):
        return False
    if meta.get("dtype") != "F8_E4M3":
        return False
    shape = meta.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        return False
    return not is_engram_embed_weight(name)


def infer_scale_name(
    weight_name: str, available_keys: set[str] | dict[str, Any]
) -> str:
    """Resolve the paired scale tensor name for a DeepSeek-V4.1 linear weight."""
    stem = weight_name[: -len(".weight")]
    candidates = (f"{stem}.scale", f"{weight_name}_scale_inv")
    for cand in candidates:
        if cand in available_keys:
            return cand
    raise ValueError(
        f"Missing paired scale tensor for FP8 linear weight {weight_name!r}; "
        f"checked {candidates}"
    )


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    """Read (data_base_offset, header_dict) without mapping the payload."""
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        (header_len,) = struct.unpack("<Q", raw_len)
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"Truncated safetensors header: {path}")
    header = json.loads(header_bytes.decode("utf-8"))
    return 8 + header_len, header


def dequantize_fp8_ue8m0_block_weight(
    weight_fp8: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize 2D E4M3 weight [N, K] with UE8M0 or FP32 block scale to FP32."""
    if weight_fp8.ndim != 2:
        raise ValueError(f"Expected 2D FP8 weight, got {tuple(weight_fp8.shape)}")
    if scale.ndim != 2:
        raise ValueError(f"Expected 2D block scale, got {tuple(scale.shape)}")
    n, k = weight_fp8.shape
    sn, sk = scale.shape
    if n % sn != 0 or k % sk != 0:
        raise ValueError(
            f"FP8 weight shape {(n, k)} not divisible by scale shape {(sn, sk)}"
        )
    block_n = n // sn
    block_k = k // sk
    w_f32 = weight_fp8.to(torch.float32)
    if scale.dtype in (torch.float8_e8m0fnu, torch.uint8):
        u = scale.view(torch.uint8).to(torch.int32)
        if bool((u == 255).any()):
            raise ValueError("Encountered UE8M0 NaN scale byte (255)")
        s_f32 = torch.where(
            u == 0,
            torch.full_like(u, 2.0**-127, dtype=torch.float32),
            torch.ldexp(torch.ones_like(u, dtype=torch.float32), u - 127),
        )
    else:
        s_f32 = scale.to(torch.float32)
    return w_f32 * s_f32.repeat_interleave(block_n, dim=0).repeat_interleave(
        block_k, dim=1
    )


def quantize_block_int8_ue8m0(
    weight_f32: torch.Tensor,
    block_n: int = 16,
    block_k: int = 16,
    use_mse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D FP32 weight [N, K] into Int8 [-128, 127] + UE8M0 [N/bn, K/bk].

    Each UE8M0 byte u encodes scale = 2^(u - 127) clamped to [1, 247] so all
    non-zero int8 * scale products remain normal BF16 values on the SIMD fast
    path without overflow/subnormal rounding. Zero blocks store u = 127 (2^0).
    """
    if weight_f32.ndim != 2:
        raise ValueError(f"expected 2D tensor, got shape {tuple(weight_f32.shape)}")
    n, k = weight_f32.shape
    if n % block_n != 0 or k % block_k != 0:
        raise ValueError(
            f"weight shape {(n, k)} not divisible by block size {(block_n, block_k)}"
        )
    if weight_f32.dtype != torch.float32:
        weight_f32 = weight_f32.float()

    nb = n // block_n
    kb = k // block_k
    blocks = (
        weight_f32.contiguous()
        .view(nb, block_n, kb, block_k)
        .permute(0, 2, 1, 3)
        .reshape(nb, kb, block_n * block_k)
    )

    w_max = blocks.amax(dim=-1)
    w_min = blocks.amin(dim=-1)
    s_req = torch.maximum(w_max / 127.0, -w_min / 128.0)
    is_zero = s_req <= 0.0

    u_ceil = torch.ceil(torch.log2(s_req.clamp(min=2.0**-126))).to(torch.int32) + 127
    u_ceil = u_ceil.clamp(UE8M0_MIN_FAST, UE8M0_MAX_FAST)
    ones = torch.ones_like(u_ceil, dtype=torch.float32)

    if not use_mse:
        best_u = torch.where(is_zero, torch.full_like(u_ceil, UE8M0_ONE), u_ceil)
        inv_s = torch.ldexp(ones, 127 - best_u).unsqueeze(-1)
        best_q = torch.clamp(torch.round(blocks * inv_s), -128.0, 127.0).to(torch.int8)
    else:
        best_err = torch.full(
            (nb, kb), float("inf"), dtype=torch.float32, device=weight_f32.device
        )
        best_u = u_ceil.clone()
        best_q = torch.zeros_like(blocks, dtype=torch.int8)

        for shift in (0, 1, 2):
            u_cand = (u_ceil - shift).clamp(min=UE8M0_MIN_FAST)
            inv_s = torch.ldexp(ones, 127 - u_cand).unsqueeze(-1)
            s_cand = torch.ldexp(ones, u_cand - 127).unsqueeze(-1)
            q_cand = torch.clamp(torch.round(blocks * inv_s), -128.0, 127.0)
            err = (q_cand * s_cand - blocks).square().sum(dim=-1)
            better = err < best_err
            if better.any():
                best_err = torch.where(better, err, best_err)
                best_u = torch.where(better, u_cand, best_u)
                best_q = torch.where(
                    better.unsqueeze(-1), q_cand.to(torch.int8), best_q
                )

        if is_zero.any():
            best_u = torch.where(is_zero, torch.full_like(best_u, UE8M0_ONE), best_u)
            best_q = torch.where(
                is_zero.unsqueeze(-1), torch.zeros_like(best_q), best_q
            )

    weight_int8 = (
        best_q.view(nb, kb, block_n, block_k)
        .permute(0, 2, 1, 3)
        .reshape(n, k)
        .contiguous()
    )
    scale_u8 = best_u.to(torch.uint8).contiguous()
    return weight_int8, scale_u8


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def _copy_span(
    in_fd: int, out_f: Any, src_offset: int, length: int, buf: bytearray
) -> None:
    """Copy `length` bytes from `in_fd` at `src_offset` to `out_f`."""
    remaining = length
    offset = src_offset
    if hasattr(os, "sendfile"):
        out_f.flush()
        out_fd = out_f.fileno()
        while remaining > 0:
            sent = os.sendfile(out_fd, in_fd, offset, min(remaining, 1 << 30))
            if sent == 0:
                raise EOFError(
                    f"Unexpected EOF in sendfile at offset={offset}, remaining={remaining}"
                )
            offset += sent
            remaining -= sent
        return

    os.lseek(in_fd, offset, os.SEEK_SET)
    while remaining > 0:
        n = min(remaining, len(buf))
        chunk = os.read(in_fd, n)
        if not chunk:
            raise EOFError(
                f"Unexpected EOF reading source shard, remaining={remaining}"
            )
        out_f.write(chunk)
        remaining -= len(chunk)


@dataclass
class ShardStats:
    shard_name: str
    converted_linears: int
    kept_tensors: int
    total_data_bytes: int
    min_ue8m0: int
    max_ue8m0: int
    sq_err_sum: float
    sq_ref_sum: float
    moved_scales: dict[str, str]
    skipped_resume: bool
    elapsed_sec: float


def _is_already_converted_shard(
    dst_path: Path,
    expected_targets: list[tuple[str, str, list[int]]],
    expected_skipped: set[str],
    src_header: dict[str, Any],
    block_n: int,
    block_k: int,
) -> tuple[bool, int]:
    """Return (valid, total_data_bytes) if dst_path is already validly converted."""
    if not dst_path.exists():
        return False, 0
    try:
        data_base, dst_header = read_safetensors_header(dst_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, struct.error):
        return False, 0

    max_end = 0
    for k, v in dst_header.items():
        if k == "__metadata__":
            continue
        end = v["data_offsets"][1]
        max_end = max(max_end, end)
    if dst_path.stat().st_size != data_base + max_end:
        return False, 0

    target_w = {w: (s, shape) for w, s, shape in expected_targets}
    for w_name, (s_name, shape) in target_w.items():
        w_meta = dst_header.get(w_name)
        s_meta = dst_header.get(s_name)
        if w_meta is None or s_meta is None:
            return False, 0
        if w_meta["dtype"] != "I8" or w_meta["shape"] != shape:
            return False, 0
        if s_meta["dtype"] != "U8" or s_meta["shape"] != [
            shape[0] // block_n,
            shape[1] // block_k,
        ]:
            return False, 0
    for k in expected_skipped:
        if k in dst_header:
            return False, 0
    # Verify every non-target, non-skipped source tensor is still present.
    consumed = (
        set(target_w.keys()) | {s for s, _ in target_w.values()} | expected_skipped
    )
    for k, v in src_header.items():
        if k == "__metadata__" or k in consumed:
            continue
        dm = dst_header.get(k)
        if dm is None or dm["dtype"] != v["dtype"] or dm["shape"] != v["shape"]:
            return False, 0
    return True, max_end


def convert_shard_streaming(
    src_path: Path,
    dst_path: Path,
    input_dir: Path,
    weight_map: dict[str, str],
    external_scales_to_skip: set[str],
    block_n: int,
    block_k: int,
    use_mse: bool,
    resume: bool,
    threads_per_worker: int,
) -> ShardStats:
    """Convert one safetensors shard using streaming raw byte copy for unchanged tensors."""
    t0 = time.monotonic()

    src_data_base, src_header = read_safetensors_header(src_path)
    src_file_size = src_path.stat().st_size

    # Identify tensors in source offset order for sequential disk reads.
    ordered_keys = sorted(
        (k for k in src_header if k != "__metadata__"),
        key=lambda k: src_header[k]["data_offsets"][0],
    )
    shard_key_set = set(ordered_keys)
    all_key_set = set(weight_map.keys()) if weight_map else shard_key_set

    target_pairs: list[tuple[str, str, list[int]]] = []
    consumed_local_scales: set[str] = set()
    moved_scales: dict[str, str] = {}

    for name in ordered_keys:
        meta = src_header[name]
        if is_target_fp8_linear(name, meta):
            scale_name = infer_scale_name(name, all_key_set)
            target_pairs.append((name, scale_name, list(meta["shape"])))
            if scale_name in shard_key_set:
                consumed_local_scales.add(scale_name)
            else:
                moved_scales[scale_name] = src_path.name

    if resume:
        ok, existing_data_bytes = _is_already_converted_shard(
            dst_path,
            target_pairs,
            external_scales_to_skip,
            src_header,
            block_n,
            block_k,
        )
        if ok:
            kept = (
                len(ordered_keys)
                - len(target_pairs)
                - len(consumed_local_scales)
                - len(external_scales_to_skip)
            )
            return ShardStats(
                shard_name=src_path.name,
                converted_linears=len(target_pairs),
                kept_tensors=kept,
                total_data_bytes=existing_data_bytes,
                min_ue8m0=UE8M0_ONE,
                max_ue8m0=UE8M0_ONE,
                sq_err_sum=0.0,
                sq_ref_sum=0.0,
                moved_scales=moved_scales,
                skipped_resume=True,
                elapsed_sec=time.monotonic() - t0,
            )

    tmp_path = dst_path.with_name(dst_path.name + ".tmp")

    # Fast path: shard has nothing to convert and nothing to skip.
    if not target_pairs and not external_scales_to_skip:
        buf = bytearray(_COPY_CHUNK_BYTES)
        with open(src_path, "rb") as in_f, open(tmp_path, "wb") as out_f:
            _copy_span(in_f.fileno(), out_f, 0, src_file_size, buf)
        os.replace(tmp_path, dst_path)
        total_data = src_file_size - src_data_base
        return ShardStats(
            shard_name=src_path.name,
            converted_linears=0,
            kept_tensors=len(ordered_keys),
            total_data_bytes=total_data,
            min_ue8m0=255,
            max_ue8m0=0,
            sq_err_sum=0.0,
            sq_ref_sum=0.0,
            moved_scales={},
            skipped_resume=False,
            elapsed_sec=time.monotonic() - t0,
        )

    # Convert ONLY the FP8 linear tensors via safe_open (leaves huge MXFP4 & Engram
    # tensors untouched in source file).
    converted_entries: dict[str, tuple[str, bytes, list[int], bytes, list[int]]] = {}
    min_u = 255
    max_u = 0
    sq_err_sum = 0.0
    sq_ref_sum = 0.0

    with safe_open(str(src_path), framework="pt", device="cpu") as sf:
        for w_name, s_name, shape in target_pairs:
            w_fp8 = sf.get_tensor(w_name)
            if s_name in shard_key_set:
                s_src = sf.get_tensor(s_name)
            else:
                other_shard = input_dir / weight_map[s_name]
                with safe_open(str(other_shard), framework="pt", device="cpu") as osf:
                    s_src = osf.get_tensor(s_name)

            w_f32 = dequantize_fp8_ue8m0_block_weight(w_fp8, s_src)
            q_i8, s_u8 = quantize_block_int8_ue8m0(
                w_f32, block_n=block_n, block_k=block_k, use_mse=use_mse
            )

            # Accumulate exact weight reconstruction error against FP8 reference.
            s_val = torch.ldexp(
                torch.ones_like(s_u8, dtype=torch.float32), s_u8.to(torch.int32) - 127
            )
            w_rec = q_i8.to(torch.float32) * s_val.repeat_interleave(
                block_n, dim=0
            ).repeat_interleave(block_k, dim=1)
            sq_err_sum += float((w_rec - w_f32).square().sum().item())
            sq_ref_sum += float(w_f32.square().sum().item())
            min_u = min(min_u, int(s_u8.min().item()))
            max_u = max(max_u, int(s_u8.max().item()))

            w_bytes = _tensor_to_bytes(q_i8)
            s_bytes = _tensor_to_bytes(s_u8)
            converted_entries[w_name] = (
                s_name,
                w_bytes,
                list(q_i8.shape),
                s_bytes,
                list(s_u8.shape),
            )
            del w_fp8, s_src, w_f32, q_i8, s_u8, s_val, w_rec

    # Build output header + streaming write plan (coalescing contiguous raw copies).
    out_header: dict[str, Any] = {}
    if "__metadata__" in src_header:
        out_header["__metadata__"] = dict(src_header["__metadata__"])
    else:
        out_header["__metadata__"] = {"format": "pt"}

    write_ops: list[tuple[str, Any]] = []
    out_offset = 0
    kept_tensors = 0

    def _add_copy_op(src_abs_start: int, length: int) -> None:
        if length == 0:
            return
        if write_ops and write_ops[-1][0] == "copy":
            prev_start, prev_len = write_ops[-1][1]
            if prev_start + prev_len == src_abs_start:
                write_ops[-1] = ("copy", (prev_start, prev_len + length))
                return
        write_ops.append(("copy", (src_abs_start, length)))

    for name in ordered_keys:
        if name in consumed_local_scales or name in external_scales_to_skip:
            continue
        if name in converted_entries:
            s_name, w_bytes, w_shape, s_bytes, s_shape = converted_entries[name]
            out_header[name] = {
                "dtype": "I8",
                "shape": w_shape,
                "data_offsets": [out_offset, out_offset + len(w_bytes)],
            }
            out_offset += len(w_bytes)
            write_ops.append(("bytes", w_bytes))

            out_header[s_name] = {
                "dtype": "U8",
                "shape": s_shape,
                "data_offsets": [out_offset, out_offset + len(s_bytes)],
            }
            out_offset += len(s_bytes)
            write_ops.append(("bytes", s_bytes))
        else:
            meta = src_header[name]
            src_begin, src_end = meta["data_offsets"]
            length = src_end - src_begin
            out_header[name] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "data_offsets": [out_offset, out_offset + length],
            }
            out_offset += length
            _add_copy_op(src_data_base + src_begin, length)
            kept_tensors += 1

    header_bytes = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    expected_total_size = 8 + len(header_bytes) + out_offset

    buf = bytearray(_COPY_CHUNK_BYTES)
    with open(src_path, "rb") as in_f, open(tmp_path, "wb") as out_f:
        in_fd = in_f.fileno()
        out_f.write(struct.pack("<Q", len(header_bytes)))
        out_f.write(header_bytes)
        for op_kind, payload in write_ops:
            if op_kind == "bytes":
                out_f.write(payload)
            else:
                src_abs_start, length = payload
                _copy_span(in_fd, out_f, src_abs_start, length, buf)

    actual_size = tmp_path.stat().st_size
    if actual_size != expected_total_size:
        tmp_path.unlink(missing_ok=True)
        raise OSError(
            f"Size mismatch writing {dst_path}: expected {expected_total_size}, "
            f"got {actual_size}"
        )
    os.replace(tmp_path, dst_path)

    return ShardStats(
        shard_name=src_path.name,
        converted_linears=len(target_pairs),
        kept_tensors=kept_tensors,
        total_data_bytes=out_offset,
        min_ue8m0=min_u,
        max_ue8m0=max_u,
        sq_err_sum=sq_err_sum,
        sq_ref_sum=sq_ref_sum,
        moved_scales=moved_scales,
        skipped_resume=False,
        elapsed_sec=time.monotonic() - t0,
    )


def build_updated_config(
    src_config: dict[str, Any], block_n: int, block_k: int
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(src_config))
    src_qc = cfg.get("quantization_config") or {}
    expert_dtype = src_qc.get("expert_dtype", "fp4")
    if expert_dtype != "fp4":
        raise ValueError(
            f"Expected DeepSeek-V4.1 expert_dtype='fp4', got {expert_dtype!r}"
        )
    cfg["quantization_config"] = {
        "quant_method": "xcpu_int8",
        "fmt": "int8_ue8m0",
        "activation_scheme": "dynamic",
        "weight_block_size": [block_n, block_k],
        "expert_dtype": "fp4",
    }
    return cfg


def _copy_ancillary_files(input_dir: Path, output_dir: Path) -> None:
    skip_names = {"config.json", "model.safetensors.index.json"}
    for entry in sorted(input_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.name in skip_names or entry.suffix == ".safetensors":
            continue
        dst = output_dir / entry.name
        if entry.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(entry, dst)
        elif entry.is_file():
            shutil.copy2(entry, dst)


def _worker_entry(task: tuple[Any, ...]) -> ShardStats:
    return convert_shard_streaming(*task)


def convert_checkpoint(
    input_dir: Path,
    output_dir: Path,
    *,
    block_size: tuple[int, int] = (16, 16),
    use_mse: bool = True,
    workers: int = 8,
    resume: bool = False,
) -> dict[str, Any]:
    block_n, block_k = _norm_block_size(block_size)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = input_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {input_dir}")
    src_config = json.loads(config_path.read_text(encoding="utf-8"))

    index_path = input_dir / "model.safetensors.index.json"
    if index_path.is_file():
        src_index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: dict[str, str] = dict(src_index["weight_map"])
        shard_names = sorted(set(weight_map.values()))
    else:
        weight_map = {}
        shard_names = sorted(p.name for p in input_dir.glob("*.safetensors"))
        if not shard_names:
            raise FileNotFoundError(f"No *.safetensors found in {input_dir}")

    # Pre-scan headers (tiny, milliseconds) to detect any cross-shard scales.
    all_keys = set(weight_map.keys())
    external_scales_by_shard: dict[str, set[str]] = {s: set() for s in shard_names}
    if weight_map:
        for shard_name in shard_names:
            _, hdr = read_safetensors_header(input_dir / shard_name)
            local_keys = set(hdr.keys())
            for k, meta in hdr.items():
                if k == "__metadata__":
                    continue
                if is_target_fp8_linear(k, meta):
                    s_name = infer_scale_name(k, all_keys)
                    if s_name not in local_keys:
                        external_scales_by_shard[weight_map[s_name]].add(s_name)

    # Sort largest shards first (e.g., 94.6 GiB Engram shards) to minimize tail time.
    shard_names_by_size = sorted(
        shard_names,
        key=lambda s: (input_dir / s).stat().st_size,
        reverse=True,
    )

    num_workers = max(1, min(int(workers), len(shard_names_by_size)))
    cpu_count = os.cpu_count() or 1
    threads_per_worker = max(1, min(16, cpu_count // num_workers))
    orig_num_threads = torch.get_num_threads()

    t_start = time.monotonic()
    tasks = [
        (
            input_dir / shard_name,
            output_dir / shard_name,
            input_dir,
            weight_map,
            external_scales_by_shard[shard_name],
            block_n,
            block_k,
            use_mse,
            resume,
            threads_per_worker,
        )
        for shard_name in shard_names_by_size
    ]

    results: list[ShardStats] = []
    try:
        if num_workers > 1:
            torch.set_num_threads(threads_per_worker)
        if num_workers == 1:
            for task in tasks:
                st = _worker_entry(task)
                results.append(st)
                print(
                    f"[{len(results):2d}/{len(tasks)}] {st.shard_name}: "
                    f"converted={st.converted_linears}, kept={st.kept_tensors}, "
                    f"size={st.total_data_bytes / 1e9:.2f}GB, "
                    f"elapsed={st.elapsed_sec:.1f}s"
                    + (" (resumed)" if st.skipped_resume else ""),
                    flush=True,
                )
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
                futs = [pool.submit(_worker_entry, task) for task in tasks]
                for fut in concurrent.futures.as_completed(futs):
                    st = fut.result()
                    results.append(st)
                    print(
                        f"[{len(results):2d}/{len(tasks)}] {st.shard_name}: "
                        f"converted={st.converted_linears}, kept={st.kept_tensors}, "
                        f"size={st.total_data_bytes / 1e9:.2f}GB, "
                        f"elapsed={st.elapsed_sec:.1f}s"
                        + (" (resumed)" if st.skipped_resume else ""),
                        flush=True,
                    )
    finally:
        if num_workers > 1:
            torch.set_num_threads(orig_num_threads)

    total_converted = sum(r.converted_linears for r in results)
    total_kept = sum(r.kept_tensors for r in results)
    total_data_bytes = sum(r.total_data_bytes for r in results)
    min_u = min((r.min_ue8m0 for r in results if r.converted_linears > 0), default=0)
    max_u = max((r.max_ue8m0 for r in results if r.converted_linears > 0), default=0)
    sq_err = sum(r.sq_err_sum for r in results)
    sq_ref = sum(r.sq_ref_sum for r in results)
    overall_rel_rmse = math.sqrt(sq_err / sq_ref) if sq_ref > 0.0 else 0.0

    # Copy tokenizer/encoding/metadata files and write updated config & index.
    _copy_ancillary_files(input_dir, output_dir)

    updated_config = build_updated_config(src_config, block_n, block_k)
    (output_dir / "config.json").write_text(
        json.dumps(updated_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if index_path.is_file():
        updated_weight_map = dict(weight_map)
        for r in results:
            for s_name, new_shard in r.moved_scales.items():
                updated_weight_map[s_name] = new_shard
        out_meta = dict(src_index.get("metadata") or {})
        out_meta["total_size"] = total_data_bytes
        out_index = {
            **{
                k: v
                for k, v in src_index.items()
                if k not in ("metadata", "weight_map")
            },
            "metadata": out_meta,
            "weight_map": updated_weight_map,
        }
        (output_dir / "model.safetensors.index.json").write_text(
            json.dumps(out_index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    elapsed = time.monotonic() - t_start
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "block_size": [block_n, block_k],
        "use_mse": use_mse,
        "shards": len(results),
        "converted_linears": total_converted,
        "kept_tensors": total_kept,
        "total_data_bytes": total_data_bytes,
        "ue8m0_range": [min_u, max_u],
        "overall_weight_rel_rmse_vs_fp8": overall_rel_rmse,
        "elapsed_sec": round(elapsed, 2),
    }
    print("\nConversion summary:", json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DeepSeek-V4.1-Flash FP8 [32,32] linear projections to "
            "XCPU Int8 [-128,127] + UE8M0 [16,16] while keeping MXFP4 routed "
            "experts and FP8 Engram tables verbatim."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input DeepSeek-V4.1-Flash checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output INT8 UE8M0 checkpoint directory.",
    )
    parser.add_argument(
        "--block-size",
        nargs=2,
        type=int,
        default=[16, 16],
        metavar=("BLOCK_N", "BLOCK_K"),
        help="Target INT8 UE8M0 scale block size (default: 16 16).",
    )
    parser.add_argument(
        "--no-mse",
        action="store_true",
        help="Disable MSE shift search (use analytical ceil-log2 UE8M0 scale).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Parallel shard worker threads (default: 12).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip shards already validly converted in --output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    convert_checkpoint(
        input_dir=args.input,
        output_dir=args.output,
        block_size=(args.block_size[0], args.block_size[1]),
        use_mse=not args.no_mse,
        workers=args.workers,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
