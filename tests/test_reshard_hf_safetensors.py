# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/reshard_hf_safetensors.py including Engram pre-splitting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import tools.reshard_hf_safetensors as reshard_mod


def _make_sample_checkpoint(src: Path) -> dict[str, torch.Tensor]:
    src.mkdir(parents=True, exist_ok=True)
    # 100 rows x 256 cols (25,600 B total, 256 B/row)
    w1 = (
        (torch.arange(100 * 256, dtype=torch.int32) % 240)
        .to(torch.uint8)
        .reshape(100, 256)
        .view(torch.float8_e4m3fn)
    )
    # 100 rows x 8 cols (800 B total, 8 B/row), including 0 and 255 boundary bytes.
    s1 = (
        (torch.arange(100 * 8, dtype=torch.int32) % 256)
        .to(torch.uint8)
        .reshape(100, 8)
        .view(torch.float8_e8m0fnu)
    )
    s1[0, 0] = torch.tensor(0, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    s1[99, 7] = torch.tensor(255, dtype=torch.uint8).view(torch.float8_e8m0fnu)

    tensors = {
        "layers.1.engram.embed.weight": w1,
        "layers.1.engram.embed.scale": s1,
        "layers.1.engram.wkv.weight": torch.ones(16, 16, dtype=torch.bfloat16),
        "layers.1.engram.q_weight": torch.full((4, 16), 2.0, dtype=torch.bfloat16),
        "layers.0.mlp.experts.0.w1.weight": torch.arange(32, dtype=torch.bfloat16).reshape(4, 8),
    }

    shard1_keys = [
        "layers.1.engram.embed.weight",
        "layers.1.engram.embed.scale",
        "layers.1.engram.wkv.weight",
    ]
    shard2_keys = [
        "layers.1.engram.q_weight",
        "layers.0.mlp.experts.0.w1.weight",
    ]
    save_file({k: tensors[k] for k in shard1_keys}, str(src / "model-00001-of-00002.safetensors"))
    save_file({k: tensors[k] for k in shard2_keys}, str(src / "model-00002-of-00002.safetensors"))
    weight_map = {k: "model-00001-of-00002.safetensors" for k in shard1_keys}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard2_keys})
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2)
    )
    (src / "config.json").write_text('{"architectures": ["DeepseekV41ForCausalLM"]}')
    (src / "tokenizer.json").write_text("{}")
    return tensors


def _reconstruct_all(ckpt_dir: Path) -> dict[str, torch.Tensor]:
    index = json.loads((ckpt_dir / "model.safetensors.index.json").read_text())
    out: dict[str, torch.Tensor] = {}
    chunks: dict[str, list[tuple[int, int, torch.Tensor]]] = {}
    for key, fname in index["weight_map"].items():
        with safe_open(str(ckpt_dir / fname), framework="pt", device="cpu") as f:
            t = f.get_tensor(key).clone()
        m = reshard_mod._ENGRAM_SLICE_RE.match(key)
        if m is not None:
            chunks.setdefault(m.group("base"), []).append(
                (int(m.group("start")), int(m.group("end")), t)
            )
        else:
            out[key] = t
    for base, parts in chunks.items():
        parts.sort(key=lambda item: item[0])
        orig_dtype = parts[0][2].dtype
        out[base] = torch.cat([p[2].view(torch.uint8) for p in parts], dim=0).view(orig_dtype)
    return out


def test_default_reshard_preserves_unsplit_behavior(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    orig = _make_sample_checkpoint(src)

    result = reshard_mod.reshard(src, dst, target_shard_size=4096, engram_shard_size=None)
    assert result.engram_row_slices == {}

    index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert "engram_row_slices" not in index["metadata"]
    assert set(index["weight_map"].keys()) == set(orig.keys())

    reconstructed = _reconstruct_all(dst)
    for key, expected in orig.items():
        assert torch.equal(reconstructed[key].view(torch.uint8), expected.view(torch.uint8))


def test_split_engram_exclusive_shards_and_bit_exact_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    orig = _make_sample_checkpoint(src)

    # target_shard_size=64 puts regular tensors into tiny shards;
    # engram_shard_size=4096 splits weight (256 B/row -> 16 rows/chunk -> 7 chunks)
    # and scale (8 B/row -> 100 rows fits in 1 chunk of 800 B).
    # Test with engram_shard_size=300 so scale also splits (37 rows/chunk -> 3 chunks)!
    result = reshard_mod.reshard(src, dst, target_shard_size=64, engram_shard_size=300)
    assert len(result.engram_row_slices) == 2

    assert (dst / "config.json").is_file()
    assert (dst / "tokenizer.json").is_file()

    index = json.loads((dst / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for k, fname in index["weight_map"].items():
        by_shard.setdefault(fname, []).append(k)

    # Every Engram row_slice tensor must be in an exclusive shard file of <= 300 data bytes.
    engram_slice_keys = [k for k in index["weight_map"] if ".row_slice." in k]
    assert len(engram_slice_keys) > 2
    for key in engram_slice_keys:
        shard_name = index["weight_map"][key]
        assert by_shard[shard_name] == [key]
        with safe_open(str(dst / shard_name), framework="pt", device="cpu") as f:
            t = f.get_tensor(key)
            assert reshard_mod.tensor_nbytes(t) <= 300

    meta_slices = index["metadata"]["engram_row_slices"]
    assert meta_slices["layers.1.engram.embed.weight"]["shape"] == [100, 256]
    assert meta_slices["layers.1.engram.embed.weight"]["dtype"] == "float8_e4m3fn"
    assert meta_slices["layers.1.engram.embed.weight"]["rows_per_chunk"] == 1
    assert meta_slices["layers.1.engram.embed.scale"]["shape"] == [100, 8]
    assert meta_slices["layers.1.engram.embed.scale"]["dtype"] == "float8_e8m0fnu"

    reconstructed = _reconstruct_all(dst)
    for key, expected in orig.items():
        assert reconstructed[key].dtype == expected.dtype
        assert torch.equal(reconstructed[key].view(torch.uint8), expected.view(torch.uint8))


def test_cli_split_engram_small_target_uses_1gb_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    orig = _make_sample_checkpoint(src)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reshard_hf_safetensors.py",
            str(src),
            str(dst),
            "--target-shard-size",
            "1B",
            "--split-engram",
        ],
    )
    reshard_mod.main()
    captured = capsys.readouterr()
    assert "using default Engram shard size 1GB" in captured.out

    index = json.loads((dst / "model.safetensors.index.json").read_text())
    # With 1GB Engram shard size, the small 100-row synthetic tables stay 1 chunk each.
    assert index["metadata"]["engram_row_slices"]["layers.1.engram.embed.weight"]["num_chunks"] == 1
    assert index["metadata"]["engram_row_slices"]["layers.1.engram.embed.scale"]["num_chunks"] == 1
    reconstructed = _reconstruct_all(dst)
    for key, expected in orig.items():
        assert torch.equal(reconstructed[key].view(torch.uint8), expected.view(torch.uint8))


def test_reshard_presplit_source_is_idempotent_and_rechunkable(tmp_path: Path) -> None:
    src = tmp_path / "src"
    stage1 = tmp_path / "stage1"
    stage2_smaller = tmp_path / "stage2_smaller"
    stage2_larger = tmp_path / "stage2_larger"
    stage2_preserve = tmp_path / "stage2_preserve"
    orig = _make_sample_checkpoint(src)

    reshard_mod.reshard(src, stage1, target_shard_size=1024, engram_shard_size=4096)
    # Rechunk smaller
    reshard_mod.reshard(stage1, stage2_smaller, target_shard_size=1024, engram_shard_size=1024)
    # Rechunk larger
    reshard_mod.reshard(stage2_smaller, stage2_larger, target_shard_size=1024, engram_shard_size=8192)
    # Reshard regular tensors without re-splitting Engram
    reshard_mod.reshard(stage1, stage2_preserve, target_shard_size=512, engram_shard_size=None)

    for out_dir in (stage1, stage2_smaller, stage2_larger, stage2_preserve):
        index = json.loads((out_dir / "model.safetensors.index.json").read_text())
        assert "layers.1.engram.embed.weight" in index["metadata"]["engram_row_slices"]
        reconstructed = _reconstruct_all(out_dir)
        for key, expected in orig.items():
            assert torch.equal(reconstructed[key].view(torch.uint8), expected.view(torch.uint8))
