#!/usr/bin/env python3
"""Convert GLM-5.2 / GLM-5.3 checkpoints (FP8 or BF16/FP32) to Int8 + UE8M0.

Target format ("xcpu_int8"):
  - Quantized weight: <name> as torch.int8 ("I8", [-128, 127]), shape [N, K]
  - Block scale:      <name>_scale_inv as torch.uint8 ("U8", UE8M0 byte in [1, 247]
                      except 127 for all-zero blocks), shape [N // block_n, K // block_k]
  - Unquantized tensors (including all float32 tensors such as
    e_score_correction_bias, norms, routers, embeddings, lm_head, eh_proj, and
    indexer.weights_proj) as well as sparse-attention indexer.wk weights are
    preserved verbatim with their exact original dtypes (FP8+FP32 scale or BF16).

Usage examples:
  # From local GLM-5.2-FP8 cache:
  python convert_glm_int8_ue8m0.py \
      --input /root/.cache/modelscope/hub/models/ZhipuAI/GLM-5.2-FP8 \
      --output /cache/models/ZhipuAI/GLM-5.2-INT8-UE8M0-16x16 \
      --block-size 16 16 --workers 24

  # Directly from ModelScope repo (GLM-5.3 or GLM-5.3-BF16):
  python convert_glm_int8_ue8m0.py \
      --input ZhipuAI/GLM-5.3-BF16 \
      --output /cache/models/ZhipuAI/GLM-5.3-INT8-UE8M0-16x16 \
      --block-size 16 16 --workers 16 --delete-downloaded-shards
  # Convert ONLY routed MoE first GroupedGemm (w13: gate_proj & up_proj),
  # preserving all other tensors in their original checkpoint precision:
  python convert_glm_int8_ue8m0.py \
      --input /root/.cache/modelscope/hub/models/ZhipuAI/GLM-5.2-FP8 \
      --output /cache/models/ZhipuAI/GLM-5.2-FP8-MOE-W13-INT8-UE8M0-16x16 \
      --scope moe_w13 --block-size 16 16 --workers 24
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


UE8M0_MIN_FAST = 1
UE8M0_MAX_FAST = 247
UE8M0_ONE = 127

VALID_SCOPES = ("all", "moe_w13")
_MOE_W13_WEIGHT_RE = re.compile(
    r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|gate_up_proj)\.weight$"
)

# Non-weight / unquantized module suffixes in GLM-5.2 / GLM-5.3 checkpoints.
# Note: ZhipuAI's FP8 config.json spells indexer.weights_proj as
# "self_attn.indexers_proj"; both spellings are handled explicitly below.
_KEEP_EXACT_MODULES = frozenset(
    {
        "lm_head",
        "model.embed_tokens",
        "model.norm",
    }
)
_KEEP_MODULE_SUFFIXES = (
    ".embed_tokens",
    ".shared_head.head",
    ".shared_head.norm",
    ".norm",
    ".enorm",
    ".hnorm",
    ".k_norm",
    ".q_norm",
    ".mlp.gate",
    ".eh_proj",
    ".self_attn.indexer.weights_proj",
    ".self_attn.indexers_proj",
    ".visual",
)


def _norm_block_size(block_size: list[int] | tuple[int, int]) -> tuple[int, int]:
    if len(block_size) != 2:
        raise ValueError(f"--block-size expects 2 ints, got {block_size}")
    bn, bk = int(block_size[0]), int(block_size[1])
    if bn <= 0 or bk <= 0:
        raise ValueError(f"block_size dimensions must be positive, got {(bn, bk)}")
    return bn, bk


def _norm_scope(scope: str) -> str:
    if scope not in VALID_SCOPES:
        raise ValueError(f"--scope must be one of {VALID_SCOPES}, got {scope!r}")
    return scope


def is_indexer_wk_weight(param_name: str) -> bool:
    """Return True for sparse attention indexer.wk weights, kept verbatim in FP8/BF16."""
    return param_name.endswith(".self_attn.indexer.wk.weight")


def is_moe_w13_weight(param_name: str) -> bool:
    """Return True if param_name belongs to routed MoE first GroupedGemm (w13)."""
    return bool(_MOE_W13_WEIGHT_RE.search(param_name))


def should_quantize_target(param_name: str, scope: str) -> bool:
    if is_indexer_wk_weight(param_name):
        return False
    if scope == "all":
        return True
    if scope == "moe_w13":
        return is_moe_w13_weight(param_name)
    raise ValueError(f"Unsupported scope: {scope!r}")


def quantize_block_int8_ue8m0(
    weight_f32: torch.Tensor,
    block_n: int = 16,
    block_k: int = 16,
    use_mse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D FP32 weight [N, K] into Int8 [-128, 127] + UE8M0 [N/bn, K/bk].

    Each UE8M0 byte u encodes scale = 2^(u - 127) clamped to [1, 247] so that all
    non-zero int8 * scale products remain normal BF16 values and use the SIMD fast
    path without overflow/subnormal rounding.
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
    # Shape: [nb, kb, block_n * block_k]
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

    # Clamp exponent before log2 to avoid -inf on zero blocks.
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
                best_q = torch.where(better.unsqueeze(-1), q_cand.to(torch.int8), best_q)

        if is_zero.any():
            best_u = torch.where(is_zero, torch.full_like(best_u, UE8M0_ONE), best_u)
            best_q = torch.where(is_zero.unsqueeze(-1), torch.zeros_like(best_q), best_q)

    weight_int8 = (
        best_q.view(nb, kb, block_n, block_k)
        .permute(0, 2, 1, 3)
        .reshape(n, k)
        .contiguous()
    )
    scale_u8 = best_u.to(torch.uint8).contiguous()
    return weight_int8, scale_u8


def _infer_power_of_two_block_dim(dim: int, s_dim: int) -> int:
    if s_dim <= 0 or dim <= 0:
        raise ValueError(f"Invalid dimensions dim={dim}, s_dim={s_dim}")
    if dim % s_dim == 0:
        return dim // s_dim
    # s_dim > 1 and dim % s_dim != 0: valid block size b satisfies ceil(dim / b) == s_dim,
    # i.e. ceil(dim / s_dim) <= b <= (dim - 1) // (s_dim - 1).
    lo = math.ceil(dim / s_dim)
    hi = (dim - 1) // (s_dim - 1)
    pow2_candidates = [
        1 << exp for exp in range(1, 16) if lo <= (1 << exp) <= hi
    ]
    if len(pow2_candidates) == 1:
        return pow2_candidates[0]
    raise ValueError(
        f"Cannot unambiguously infer FP8 block size from weight dim={dim} and "
        f"scale dim={s_dim} (valid range [{lo}, {hi}]); pass src_block_size explicitly"
    )


def dequantize_fp8_block_weight(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    src_block_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Dequantize a 2D FP8 weight and block-wise weight_scale_inv to FP32."""
    n, k = weight_fp8.shape
    sn, sk = scale_inv.shape
    w_f32 = weight_fp8.float()
    if scale_inv.dtype in (torch.uint8, torch.float8_e8m0fnu):
        s_f32 = scale_inv.view(torch.float8_e8m0fnu).float()
    else:
        s_f32 = scale_inv.float()

    if src_block_size is not None:
        bn, bk = int(src_block_size[0]), int(src_block_size[1])
        if math.ceil(n / bn) != sn or math.ceil(k / bk) != sk:
            raise ValueError(
                f"FP8 weight shape {(n, k)} and scale shape {(sn, sk)} do not match "
                f"src_block_size={(bn, bk)}"
            )
    else:
        bn = _infer_power_of_two_block_dim(n, sn)
        bk = _infer_power_of_two_block_dim(k, sk)

    if n % bn == 0 and k % bk == 0:
        return (w_f32.view(sn, bn, sk, bk) * s_f32.view(sn, 1, sk, 1)).reshape(n, k)
    s_full = s_f32.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)[:n, :k]
    return w_f32 * s_full


def _module_prefix(param_name: str) -> str:
    if param_name.endswith(".weight"):
        return param_name[: -len(".weight")]
    return param_name


def _matches_ignored_list(module_name: str, ignored: set[str]) -> bool:
    if module_name in ignored:
        return True
    # Normalize ZhipuAI's indexer.weights_proj typo if ignored only lists indexers_proj
    if module_name.endswith(".self_attn.indexer.weights_proj"):
        alt = module_name.replace(
            ".self_attn.indexer.weights_proj", ".self_attn.indexers_proj"
        )
        if alt in ignored:
            return True
    for entry in ignored:
        if entry and (module_name.startswith(f"{entry}.") or module_name.endswith(f".{entry}")):
            return True
    return False


def should_keep_unquantized_non_fp8(
    name: str,
    tensor: torch.Tensor,
    ignored_modules: set[str],
) -> bool:
    """Decide whether a non-FP8 tensor must stay unquantized verbatim."""
    # Invariant: all FP32 tensors (e.g. e_score_correction_bias) must never change dtype.
    if tensor.dtype == torch.float32:
        return True
    if tensor.dtype not in (torch.bfloat16, torch.float16):
        return True
    if tensor.ndim != 2 or not name.endswith(".weight"):
        return True

    mod = _module_prefix(name)
    if mod in _KEEP_EXACT_MODULES:
        return True
    if "layernorm" in mod:
        return True
    if any(mod.endswith(suffix) for suffix in _KEEP_MODULE_SUFFIXES):
        return True
    if "visual" in mod.split("."):
        return True
    if _matches_ignored_list(mod, ignored_modules):
        return True
    return False


@dataclass
class ShardStats:
    shard_name: str
    num_quantized: int
    num_kept_fp8: int
    num_kept_bf16: int
    num_kept_f32: int
    num_kept_other: int
    total_bytes: int
    weight_map_entries: dict[str, str]
    kept_modules: list[str]
    elapsed_sec: float


def convert_single_shard(
    src_shard_path: str,
    dst_shard_path: str,
    shard_name: str,
    block_n: int,
    block_k: int,
    use_mse: bool,
    ignored_modules: set[str],
    scope: str = "all",
    num_intra_threads: int = 1,
    src_block_size: tuple[int, int] | None = None,
) -> ShardStats:
    t0 = time.perf_counter()
    torch.set_num_threads(max(1, num_intra_threads))

    tensors = load_file(src_shard_path, device="cpu")
    out_tensors: dict[str, torch.Tensor] = {}
    weight_map_entries: dict[str, str] = {}
    kept_modules: list[str] = []

    num_quantized = 0
    num_kept_fp8 = 0
    num_kept_bf16 = 0
    num_kept_f32 = 0
    num_kept_other = 0

    for name in sorted(tensors.keys()):
        if name.endswith("_scale_inv") and name[: -len("_scale_inv")] in tensors:
            base = name[: -len("_scale_inv")]
            if tensors[base].dtype == torch.float8_e4m3fn:
                continue

        tensor = tensors[name]
        if tensor.dtype == torch.float8_e4m3fn:
            scale_name = f"{name}_scale_inv"
            if scale_name not in tensors:
                raise KeyError(
                    f"FP8 tensor {name!r} in {shard_name} missing {scale_name!r} in same shard"
                )
            if should_quantize_target(name, scope):
                w_f32 = dequantize_fp8_block_weight(
                    tensor, tensors[scale_name], src_block_size=src_block_size
                )
                q_w, q_s = quantize_block_int8_ue8m0(
                    w_f32, block_n=block_n, block_k=block_k, use_mse=use_mse
                )
                out_tensors[name] = q_w
                out_tensors[scale_name] = q_s
                weight_map_entries[name] = shard_name
                weight_map_entries[scale_name] = shard_name
                num_quantized += 1
            else:
                out_tensors[name] = tensor
                out_tensors[scale_name] = tensors[scale_name]
                weight_map_entries[name] = shard_name
                weight_map_entries[scale_name] = shard_name
                num_kept_fp8 += 1
        elif tensor.dtype == torch.float32:
            # Never touch FP32 weights/biases unless it was an FP8 scale handled above.
            out_tensors[name] = tensor
            weight_map_entries[name] = shard_name
            kept_modules.append(name if not name.endswith(".weight") else _module_prefix(name))
            num_kept_f32 += 1
        elif should_keep_unquantized_non_fp8(
            name, tensor, ignored_modules
        ) or not should_quantize_target(name, scope):
            out_tensors[name] = tensor
            weight_map_entries[name] = shard_name
            kept_modules.append(name if not name.endswith(".weight") else _module_prefix(name))
            if tensor.dtype == torch.bfloat16:
                num_kept_bf16 += 1
            else:
                num_kept_other += 1
        else:
            # Unquantized 2D BF16/FP16 linear or expert weight -> Int8 + UE8M0.
            q_w, q_s = quantize_block_int8_ue8m0(
                tensor.float(), block_n=block_n, block_k=block_k, use_mse=use_mse
            )
            scale_name = f"{name}_scale_inv"
            out_tensors[name] = q_w
            out_tensors[scale_name] = q_s
            weight_map_entries[name] = shard_name
            weight_map_entries[scale_name] = shard_name
            num_quantized += 1

    tmp_path = f"{dst_shard_path}.tmp.{os.getpid()}"
    save_file(out_tensors, tmp_path)
    os.replace(tmp_path, dst_shard_path)

    total_bytes = sum(t.numel() * t.element_size() for t in out_tensors.values())
    return ShardStats(
        shard_name=shard_name,
        num_quantized=num_quantized,
        num_kept_fp8=num_kept_fp8,
        num_kept_bf16=num_kept_bf16,
        num_kept_f32=num_kept_f32,
        num_kept_other=num_kept_other,
        total_bytes=total_bytes,
        weight_map_entries=weight_map_entries,
        kept_modules=kept_modules,
        elapsed_sec=time.perf_counter() - t0,
    )


def _worker_convert_shard(task: tuple[Any, ...]) -> ShardStats:
    return convert_single_shard(*task)


def _derive_modules_to_not_convert(
    src_cfg: dict[str, Any],
    kept_modules: set[str],
    all_param_names: list[str] | None = None,
) -> list[str]:
    existing_qcfg = src_cfg.get("quantization_config") or {}
    existing = list(existing_qcfg.get("modules_to_not_convert") or [])
    result: set[str] = set(existing) | set(kept_modules)

    # Fix ZhipuAI's typo: whenever self_attn.indexers_proj is present, also list
    # self_attn.indexer.weights_proj so vLLM's is_layer_skipped matches it directly.
    for item in list(result):
        if item.endswith(".self_attn.indexers_proj"):
            result.add(
                item.replace(
                    ".self_attn.indexers_proj", ".self_attn.indexer.weights_proj"
                )
            )
    if all_param_names:
        for pname in all_param_names:
            if pname.endswith(".self_attn.indexer.weights_proj.weight"):
                result.add(pname[: -len(".weight")])
    return sorted(result)


def build_output_config(
    src_cfg: dict[str, Any],
    block_n: int,
    block_k: int,
    kept_modules: set[str],
    all_param_names: list[str] | None = None,
    scope: str = "all",
) -> dict[str, Any]:
    out_cfg = json.loads(json.dumps(src_cfg))
    modules_to_not_convert = _derive_modules_to_not_convert(
        src_cfg, kept_modules, all_param_names=all_param_names
    )
    src_qcfg = src_cfg.get("quantization_config") or {}
    if scope != "all" and src_qcfg:
        # Preserve original checkpoint quantization config (e.g. FP8) for all
        # non-target tensors, and record the Int8 UE8M0 override for target_scope.
        merged_qcfg = json.loads(json.dumps(src_qcfg))
        merged_qcfg["modules_to_not_convert"] = modules_to_not_convert
        merged_qcfg["xcpu_int8_overrides"] = {
            "fmt": "int8_ue8m0",
            "target_scope": scope,
            "weight_block_size": [block_n, block_k],
        }
        out_cfg["quantization_config"] = merged_qcfg
        return out_cfg

    qcfg: dict[str, Any] = {
        "activation_scheme": "dynamic",
        "fmt": "int8_ue8m0",
        "modules_to_not_convert": modules_to_not_convert,
        "quant_method": "xcpu_int8",
        "weight_block_size": [block_n, block_k],
    }
    if scope != "all":
        qcfg["target_scope"] = scope
    out_cfg["quantization_config"] = qcfg
    return out_cfg


def _copy_ancillary_files(src_dir: Path, dst_dir: Path) -> None:
    skip_exact = {"config.json", "model.safetensors.index.json"}
    for item in src_dir.iterdir():
        if item.name in skip_exact:
            continue
        if item.name.endswith(".safetensors") or item.name in (".mdl", ".msc", ".mv"):
            continue
        dst_item = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, dst_item)


def _is_modelscope_repo_id(spec: str) -> bool:
    if os.path.exists(spec):
        return False
    return "/" in spec and not spec.startswith((".", "/"))


def _resolve_input_source(
    input_spec: str,
    dst_dir: Path,
    revision: str | None,
    download_cache_dir: str | None,
    delete_downloaded_shards: bool,
) -> tuple[Path, bool, str | None]:
    """Return (src_dir, is_modelscope_incremental, repo_id)."""
    if os.path.isdir(input_spec):
        return Path(input_spec), False, None
    if not _is_modelscope_repo_id(input_spec):
        raise FileNotFoundError(f"Input path does not exist: {input_spec}")

    from modelscope.hub.snapshot_download import snapshot_download

    cache_root = download_cache_dir or str(dst_dir / ".ms_cache")
    if delete_downloaded_shards:
        os.makedirs(cache_root, exist_ok=True)
        # First fetch only metadata and non-safetensors files.
        meta_dir = snapshot_download(
            input_spec,
            revision=revision,
            cache_dir=cache_root,
            ignore_patterns=["*.safetensors"],
        )
        return Path(meta_dir), True, input_spec

    full_dir = snapshot_download(
        input_spec,
        revision=revision,
        cache_dir=download_cache_dir,
    )
    return Path(full_dir), False, input_spec


def _collect_resumed_shard_stats(
    dst_shard: Path, shard_name: str
) -> tuple[dict[str, str], int, set[str], int, int, int, int]:
    t_dict = load_file(str(dst_shard), device="cpu")
    weight_map_entries: dict[str, str] = {}
    kept_modules: set[str] = set()
    total_bytes = 0
    n_q = n_fp8 = n_bf16 = n_f32 = 0
    for k, t in t_dict.items():
        weight_map_entries[k] = shard_name
        total_bytes += t.numel() * t.element_size()
        if k.endswith("_scale_inv"):
            continue
        if t.dtype == torch.int8:
            n_q += 1
        elif t.dtype == torch.float8_e4m3fn:
            n_fp8 += 1
        else:
            kept_modules.add(k if not k.endswith(".weight") else _module_prefix(k))
            if t.dtype == torch.float32:
                n_f32 += 1
            elif t.dtype == torch.bfloat16:
                n_bf16 += 1
    return weight_map_entries, total_bytes, kept_modules, n_q, n_fp8, n_bf16, n_f32


def _convert_incremental_modelscope(
    repo_id: str,
    meta_dir: Path,
    dst_dir: Path,
    shard_names: list[str],
    block_n: int,
    block_k: int,
    use_mse: bool,
    ignored_modules: set[str],
    scope: str,
    revision: str | None,
    download_cache_dir: str,
    delete_downloaded_shards: bool,
    resume: bool,
    src_block_size: tuple[int, int] | None = None,
) -> tuple[dict[str, str], int, set[str], int, int, int, int]:
    from modelscope.hub.file_download import model_file_download

    weight_map: dict[str, str] = {}
    total_bytes = 0
    kept_modules: set[str] = set()
    total_q = total_fp8 = total_bf16 = total_f32 = 0
    intra_threads = max(1, (os.cpu_count() or 16) // 2)

    for idx, shard_name in enumerate(shard_names, start=1):
        dst_shard = dst_dir / shard_name
        if resume and dst_shard.exists() and dst_shard.stat().st_size > 0:
            wm, tb, km, nq, nfp8, nb, nf = _collect_resumed_shard_stats(
                dst_shard, shard_name
            )
            weight_map.update(wm)
            total_bytes += tb
            kept_modules.update(km)
            total_q += nq
            total_fp8 += nfp8
            total_bf16 += nb
            total_f32 += nf
            print(f"[{idx}/{len(shard_names)}] Resumed {shard_name}", flush=True)
            continue

        downloaded = model_file_download(
            repo_id,
            file_path=shard_name,
            revision=revision,
            cache_dir=download_cache_dir,
        )
        stats = convert_single_shard(
            downloaded,
            str(dst_shard),
            shard_name,
            block_n,
            block_k,
            use_mse,
            ignored_modules,
            scope=scope,
            num_intra_threads=intra_threads,
            src_block_size=src_block_size,
        )
        weight_map.update(stats.weight_map_entries)
        total_bytes += stats.total_bytes
        kept_modules.update(stats.kept_modules)
        total_q += stats.num_quantized
        total_fp8 += stats.num_kept_fp8
        total_bf16 += stats.num_kept_bf16
        total_f32 += stats.num_kept_f32
        print(
            f"[{idx}/{len(shard_names)}] {shard_name}: q={stats.num_quantized} "
            f"fp8={stats.num_kept_fp8} bf16={stats.num_kept_bf16} "
            f"f32={stats.num_kept_f32} ({stats.elapsed_sec:.1f}s)",
            flush=True,
        )
        if delete_downloaded_shards and os.path.exists(downloaded):
            os.remove(downloaded)

    return weight_map, total_bytes, kept_modules, total_q, total_fp8, total_bf16, total_f32


def convert_checkpoint(
    input_spec: str,
    output_dir: str,
    *,
    block_size: tuple[int, int] = (16, 16),
    scope: str = "all",
    workers: int = 8,
    use_mse: bool = True,
    resume: bool = False,
    delete_downloaded_shards: bool = False,
    revision: str | None = None,
    download_cache_dir: str | None = None,
) -> Path:
    block_n, block_k = _norm_block_size(block_size)
    scope = _norm_scope(scope)
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_dir, incremental_ms, repo_id = _resolve_input_source(
        input_spec,
        dst_dir,
        revision=revision,
        download_cache_dir=download_cache_dir,
        delete_downloaded_shards=delete_downloaded_shards,
    )

    src_cfg_path = src_dir / "config.json"
    if not src_cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in {src_dir}")
    src_cfg = json.loads(src_cfg_path.read_text(encoding="utf-8"))

    src_qcfg = src_cfg.get("quantization_config") or {}
    raw_src_bs = src_qcfg.get("weight_block_size")
    src_block_size: tuple[int, int] | None = (
        (int(raw_src_bs[0]), int(raw_src_bs[1]))
        if isinstance(raw_src_bs, (list, tuple)) and len(raw_src_bs) == 2
        else None
    )
    ignored_modules = set(src_qcfg.get("modules_to_not_convert") or [])
    if src_qcfg.get("ignored_layers"):
        ignored_modules.update(src_qcfg["ignored_layers"])

    _copy_ancillary_files(src_dir, dst_dir)

    index_path = src_dir / "model.safetensors.index.json"
    single_safetensors = src_dir / "model.safetensors"

    weight_map: dict[str, str] = {}
    kept_modules: set[str] = set()
    total_bytes = 0
    total_quantized = 0
    total_kept_fp8 = 0
    total_kept_bf16 = 0
    total_kept_f32 = 0
    all_param_names: list[str] | None = None

    if index_path.exists():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        src_weight_map: dict[str, str] = index_data["weight_map"]
        all_param_names = [
            k for k in src_weight_map if not k.endswith("_scale_inv")
        ]
        shard_names = sorted(set(src_weight_map.values()))
        actual_shards_exist = all((src_dir / s).exists() for s in shard_names)

        if not incremental_ms and not actual_shards_exist:
            # Metadata-only dummy checkpoint (e.g., GLM-5.2-FP8-dummy-7-layer-lite):
            # Rewrite config.json and copy/update weight_map adding *_scale_inv if missing.
            dummy_probe_2d = torch.empty((0, 0), dtype=torch.bfloat16)
            for k in sorted(src_weight_map.keys()):
                shard = src_weight_map[k]
                weight_map[k] = shard
                if k.endswith("_scale_inv"):
                    continue
                if f"{k}_scale_inv" in src_weight_map:
                    continue
                if not should_keep_unquantized_non_fp8(
                    k, dummy_probe_2d, ignored_modules
                ) and should_quantize_target(k, scope):
                    weight_map[f"{k}_scale_inv"] = shard
                else:
                    kept_modules.add(
                        k if not k.endswith(".weight") else _module_prefix(k)
                    )
            out_index = {
                "metadata": dict(index_data.get("metadata") or {}),
                "weight_map": {k: weight_map[k] for k in sorted(weight_map.keys())},
            }
            (dst_dir / "model.safetensors.index.json").write_text(
                json.dumps(out_index, indent=2) + "\n", encoding="utf-8"
            )
            out_cfg = build_output_config(
                src_cfg,
                block_n,
                block_k,
                kept_modules,
                all_param_names=all_param_names,
                scope=scope,
            )
            (dst_dir / "config.json").write_text(
                json.dumps(out_cfg, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"Converted metadata-only dummy checkpoint -> {dst_dir} "
                f"({len(weight_map)} index entries, scope={scope})"
            )
            return dst_dir

        if incremental_ms:
            assert repo_id is not None
            cache_root = download_cache_dir or str(dst_dir / ".ms_cache")
            (
                weight_map,
                total_bytes,
                kept_modules,
                total_quantized,
                total_kept_fp8,
                total_kept_bf16,
                total_kept_f32,
            ) = _convert_incremental_modelscope(
                repo_id=repo_id,
                meta_dir=src_dir,
                dst_dir=dst_dir,
                shard_names=shard_names,
                block_n=block_n,
                block_k=block_k,
                use_mse=use_mse,
                ignored_modules=ignored_modules,
                scope=scope,
                revision=revision,
                download_cache_dir=cache_root,
                delete_downloaded_shards=delete_downloaded_shards,
                resume=resume,
                src_block_size=src_block_size,
            )
        else:
            tasks = []
            resumed = 0
            cpu_cnt = os.cpu_count() or 16
            intra_threads = max(1, cpu_cnt // max(1, workers * 2))
            for shard_name in shard_names:
                dst_shard = dst_dir / shard_name
                if resume and dst_shard.exists() and dst_shard.stat().st_size > 0:
                    wm, tb, km, nq, nfp8, nb, nf = _collect_resumed_shard_stats(
                        dst_shard, shard_name
                    )
                    weight_map.update(wm)
                    total_bytes += tb
                    kept_modules.update(km)
                    total_quantized += nq
                    total_kept_fp8 += nfp8
                    total_kept_bf16 += nb
                    total_kept_f32 += nf
                    resumed += 1
                    continue
                tasks.append(
                    (
                        str(src_dir / shard_name),
                        str(dst_shard),
                        shard_name,
                        block_n,
                        block_k,
                        use_mse,
                        ignored_modules,
                        scope,
                        intra_threads,
                        src_block_size,
                    )
                )

            if resumed:
                print(f"Resumed {resumed}/{len(shard_names)} existing shards.")
            if workers <= 1:
                for idx, t in enumerate(tasks, start=1):
                    st = _worker_convert_shard(t)
                    weight_map.update(st.weight_map_entries)
                    kept_modules.update(st.kept_modules)
                    total_bytes += st.total_bytes
                    total_quantized += st.num_quantized
                    total_kept_fp8 += st.num_kept_fp8
                    total_kept_bf16 += st.num_kept_bf16
                    total_kept_f32 += st.num_kept_f32
                    print(
                        f"[{idx}/{len(tasks)}] {st.shard_name}: q={st.num_quantized} "
                        f"fp8={st.num_kept_fp8} bf16={st.num_kept_bf16} "
                        f"f32={st.num_kept_f32} ({st.elapsed_sec:.1f}s)",
                        flush=True,
                    )
            elif tasks:
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                    for idx, st in enumerate(ex.map(_worker_convert_shard, tasks), start=1):
                        weight_map.update(st.weight_map_entries)
                        kept_modules.update(st.kept_modules)
                        total_bytes += st.total_bytes
                        total_quantized += st.num_quantized
                        total_kept_fp8 += st.num_kept_fp8
                        total_kept_bf16 += st.num_kept_bf16
                        total_kept_f32 += st.num_kept_f32
                        print(
                            f"[{idx}/{len(tasks)}] {st.shard_name}: q={st.num_quantized} "
                            f"fp8={st.num_kept_fp8} bf16={st.num_kept_bf16} "
                            f"f32={st.num_kept_f32} ({st.elapsed_sec:.1f}s)",
                            flush=True,
                        )

        sorted_weight_map = {k: weight_map[k] for k in sorted(weight_map.keys())}
        out_index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": sorted_weight_map,
        }
        (dst_dir / "model.safetensors.index.json").write_text(
            json.dumps(out_index, indent=2) + "\n", encoding="utf-8"
        )
    elif single_safetensors.exists():
        st = convert_single_shard(
            str(single_safetensors),
            str(dst_dir / "model.safetensors"),
            "model.safetensors",
            block_n,
            block_k,
            use_mse,
            ignored_modules,
            scope=scope,
            num_intra_threads=max(1, (os.cpu_count() or 16) // 2),
            src_block_size=src_block_size,
        )
        kept_modules.update(st.kept_modules)
        total_bytes = st.total_bytes
        total_quantized = st.num_quantized
        total_kept_fp8 = st.num_kept_fp8
        total_kept_bf16 = st.num_kept_bf16
        total_kept_f32 = st.num_kept_f32
        all_param_names = [
            k for k in st.weight_map_entries if not k.endswith("_scale_inv")
        ]
    else:
        raise FileNotFoundError(
            f"Neither model.safetensors.index.json nor model.safetensors found in {src_dir}"
        )

    out_cfg = build_output_config(
        src_cfg,
        block_n,
        block_k,
        kept_modules,
        all_param_names=all_param_names,
        scope=scope,
    )
    (dst_dir / "config.json").write_text(
        json.dumps(out_cfg, indent=2) + "\n", encoding="utf-8"
    )
    ms_tmp = dst_dir / ".ms_cache"
    if delete_downloaded_shards and ms_tmp.exists():
        shutil.rmtree(ms_tmp, ignore_errors=True)

    print(
        f"Done converting {input_spec} -> {dst_dir} (scope={scope}): "
        f"quantized={total_quantized}, kept_fp8={total_kept_fp8}, "
        f"kept_bf16={total_kept_bf16}, kept_f32={total_kept_f32}, "
        f"total_size={total_bytes / (1024**3):.2f} GiB",
        flush=True,
    )
    return dst_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert GLM-5.2/GLM-5.3 FP8 or BF16 checkpoints to Int8 + UE8M0"
    )
    parser.add_argument("--input", required=True, help="Local dir or ModelScope repo ID")
    parser.add_argument("--output", required=True, help="Destination checkpoint dir")
    parser.add_argument(
        "--block-size",
        type=int,
        nargs=2,
        default=[16, 16],
        metavar=("BLOCK_N", "BLOCK_K"),
        help="Block quantization shape (default: 16 16)",
    )
    parser.add_argument(
        "--scope",
        choices=VALID_SCOPES,
        default="all",
        help=(
            "Which weights to convert to Int8 + UE8M0: "
            "'all' (default, all quantized linear/MoE weights) or "
            "'moe_w13' (only routed MoE first GroupedGemm gate_proj/up_proj; "
            "all other tensors stay in their original checkpoint precision)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel shard worker processes (default: 16)",
    )
    parser.add_argument(
        "--no-mse",
        action="store_true",
        help="Use pure ceiling UE8M0 exponent without 3-candidate MSE search",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already converted non-empty destination shards",
    )
    parser.add_argument(
        "--delete-downloaded-shards",
        action="store_true",
        help="For remote ModelScope repo IDs, delete source shard after converting it",
    )
    parser.add_argument(
        "--modelscope-revision",
        default=None,
        help="Optional ModelScope revision tag",
    )
    parser.add_argument(
        "--download-cache-dir",
        default=None,
        help="Optional ModelScope download cache directory",
    )
    args = parser.parse_args(argv)

    convert_checkpoint(
        args.input,
        args.output,
        block_size=(args.block_size[0], args.block_size[1]),
        scope=args.scope,
        workers=args.workers,
        use_mse=not args.no_mse,
        resume=args.resume,
        delete_downloaded_shards=args.delete_downloaded_shards,
        revision=args.modelscope_revision,
        download_cache_dir=args.download_cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
