"""Unit tests for convert_deepseek_v41_int8_ue8m0.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import convert_deepseek_v41_int8_ue8m0 as conv
import convert_glm_int8_ue8m0 as glm_conv


def _make_fp8_ue8m0_tensor(
    shape: tuple[int, int], block_size: int = 32, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, k = shape
    raw = torch.randn(n, k, generator=g, dtype=torch.float32) * 0.05
    nb, kb = n // block_size, k // block_size
    blocks = (
        raw.view(nb, block_size, kb, block_size).permute(0, 2, 1, 3).reshape(nb, kb, -1)
    )
    amax = blocks.abs().amax(dim=-1).clamp(min=2.0**-120)
    u = (torch.ceil(torch.log2(amax / 448.0)).to(torch.int32) + 127).clamp(1, 247)
    s_f32 = torch.ldexp(torch.ones_like(u, dtype=torch.float32), u - 127)
    s_exp = s_f32.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    w_fp8 = (raw / s_exp).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    s_ue8m0 = u.to(torch.uint8).view(torch.float8_e8m0fnu)
    ref_f32 = w_fp8.to(torch.float32) * s_exp
    return w_fp8, s_ue8m0, ref_f32


def test_quantize_matches_glm_converter() -> None:
    g = torch.Generator(device="cpu").manual_seed(42)
    w = torch.randn(64, 96, generator=g, dtype=torch.float32) * 0.1
    q1, s1 = conv.quantize_block_int8_ue8m0(w, 16, 16, use_mse=True)
    q2, s2 = glm_conv.quantize_block_int8_ue8m0(w, 16, 16, use_mse=True)
    assert torch.equal(q1, q2)
    assert torch.equal(s1, s2)


def test_streaming_conversion_preserves_mxfp4_and_engram(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()

    # Create 3 shards:
    # shard 1: BF16/F32 only (unchanged fast-copy path)
    embed_w = torch.randn(32, 64, dtype=torch.bfloat16)
    norm_w = torch.ones(64, dtype=torch.float32)
    shard1 = {"embed.weight": embed_w, "norm.weight": norm_w}

    # shard 2: FP8 [32,32] linears + MXFP4 routed experts + BF16 router/compressor
    wq_a_w, wq_a_s, wq_a_ref = _make_fp8_ue8m0_tensor((64, 64), 32, seed=1)
    wo_a_w, wo_a_s, wo_a_ref = _make_fp8_ue8m0_tensor((128, 64), 32, seed=2)
    sh_w1_w, sh_w1_s, sh_w1_ref = _make_fp8_ue8m0_tensor((64, 64), 32, seed=3)
    mxfp4_w = torch.randint(-128, 128, (64, 32), dtype=torch.int8)
    mxfp4_s = torch.randint(115, 130, (64, 2), dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    gate_w = torch.randn(4, 64, dtype=torch.bfloat16)
    indexer_wk = torch.randn(32, 64, dtype=torch.bfloat16)
    shard2 = {
        "layers.0.attn.wq_a.weight": wq_a_w,
        "layers.0.attn.wq_a.scale": wq_a_s,
        "layers.0.attn.wo_a.weight": wo_a_w,
        "layers.0.attn.wo_a.scale": wo_a_s,
        "layers.0.ffn.shared_experts.w1.weight": sh_w1_w,
        "layers.0.ffn.shared_experts.w1.scale": sh_w1_s,
        "layers.0.ffn.experts.0.w1.weight": mxfp4_w,
        "layers.0.ffn.experts.0.w1.scale": mxfp4_s,
        "layers.0.ffn.gate.weight": gate_w,
        "layers.0.attn.indexer.wk.weight": indexer_wk,
    }

    # shard 3: Engram FP8 lookup table (KEEP VERBATIM) + engram.wkv (CONVERT!)
    engram_embed_w = (torch.randn(100, 32) * 0.2).to(torch.float8_e4m3fn)
    engram_embed_s = torch.full((100, 1), 125, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    engram_wkv_w, engram_wkv_s, engram_wkv_ref = _make_fp8_ue8m0_tensor(
        (64, 64), 32, seed=4
    )
    shard3 = {
        "layers.1.engram.embed.weight": engram_embed_w,
        "layers.1.engram.embed.scale": engram_embed_s,
        "layers.1.engram.wkv.weight": engram_wkv_w,
        "layers.1.engram.wkv.scale": engram_wkv_s,
    }

    s1_name = "model-00001-of-00003.safetensors"
    s2_name = "model-00002-of-00003.safetensors"
    s3_name = "model-00003-of-00003.safetensors"
    save_file(shard1, str(src_dir / s1_name))
    save_file(shard2, str(src_dir / s2_name))
    save_file(shard3, str(src_dir / s3_name))

    weight_map = {}
    for k in shard1:
        weight_map[k] = s1_name
    for k in shard2:
        weight_map[k] = s2_name
    for k in shard3:
        weight_map[k] = s3_name
    engram_Slices_meta = {
        "layers.1.engram.embed.weight": {
            "shape": [100, 32],
            "dtype": "float8_e4m3fn",
            "num_chunks": 1,
            "rows_per_chunk": 100,
        }
    }
    (src_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": 0,
                    "engram_row_slices": engram_Slices_meta,
                },
                "weight_map": weight_map,
            }
        ),
        encoding="utf-8",
    )
    (src_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV41ForCausalLM"],
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                    "weight_block_size": [32, 32],
                    "scale_fmt": "ue8m0",
                    "expert_dtype": "fp4",
                },
            }
        ),
        encoding="utf-8",
    )
    (src_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    orig_threads = torch.get_num_threads()
    summary = conv.convert_checkpoint(
        src_dir, dst_dir, block_size=(16, 16), use_mse=True, workers=2
    )
    assert torch.get_num_threads() == orig_threads
    assert summary["converted_linears"] == 4
    assert summary["overall_weight_rel_rmse_vs_fp8"] < 0.02

    out_idx = json.loads(
        (dst_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert out_idx["metadata"]["engram_row_slices"] == engram_Slices_meta
    assert out_idx["metadata"]["total_size"] > 0

    # Verify config.json
    out_cfg = json.loads((dst_dir / "config.json").read_text(encoding="utf-8"))
    assert out_cfg["quantization_config"] == {
        "quant_method": "xcpu_int8",
        "fmt": "int8_ue8m0",
        "activation_scheme": "dynamic",
        "weight_block_size": [16, 16],
        "expert_dtype": "fp4",
    }
    assert (dst_dir / "tokenizer.json").is_file()

    # Verify shard 1 verbatim
    with safe_open(str(dst_dir / s1_name), framework="pt", device="cpu") as sf:
        assert torch.equal(sf.get_tensor("embed.weight"), embed_w)
        assert torch.equal(sf.get_tensor("norm.weight"), norm_w)

    # Verify shard 2: converted linears are I8 + U8 [N/16, K/16]; MXFP4 & BF16 verbatim
    with safe_open(str(dst_dir / s2_name), framework="pt", device="cpu") as sf:
        for name, ref in (
            ("layers.0.attn.wq_a", wq_a_ref),
            ("layers.0.attn.wo_a", wo_a_ref),
            ("layers.0.ffn.shared_experts.w1", sh_w1_ref),
        ):
            qw = sf.get_tensor(f"{name}.weight")
            sc = sf.get_tensor(f"{name}.scale")
            assert qw.dtype == torch.int8
            assert sc.dtype == torch.uint8
            assert qw.shape == ref.shape
            assert sc.shape == (ref.shape[0] // 16, ref.shape[1] // 16)
            assert int(sc.min()) >= 1 and int(sc.max()) <= 247
            rec = qw.float() * torch.ldexp(
                torch.ones_like(sc, dtype=torch.float32), sc.int() - 127
            ).repeat_interleave(16, 0).repeat_interleave(16, 1)
            rel_rmse = (rec - ref).norm() / ref.norm()
            assert rel_rmse < 0.02

        assert torch.equal(sf.get_tensor("layers.0.ffn.experts.0.w1.weight"), mxfp4_w)
        assert torch.equal(
            sf.get_tensor("layers.0.ffn.experts.0.w1.scale").view(torch.uint8),
            mxfp4_s.view(torch.uint8),
        )
        assert (
            sf.get_tensor("layers.0.ffn.experts.0.w1.scale").dtype
            == torch.float8_e8m0fnu
        )
        assert torch.equal(sf.get_tensor("layers.0.ffn.gate.weight"), gate_w)
        assert torch.equal(sf.get_tensor("layers.0.attn.indexer.wk.weight"), indexer_wk)

    # Verify shard 3: Engram embed table untouched (F8_E4M3 + F8_E8M0), engram.wkv converted
    with safe_open(str(dst_dir / s3_name), framework="pt", device="cpu") as sf:
        emb_w = sf.get_tensor("layers.1.engram.embed.weight")
        emb_s = sf.get_tensor("layers.1.engram.embed.scale")
        assert emb_w.dtype == torch.float8_e4m3fn
        assert emb_s.dtype == torch.float8_e8m0fnu
        assert torch.equal(emb_w.view(torch.uint8), engram_embed_w.view(torch.uint8))
        assert torch.equal(emb_s.view(torch.uint8), engram_embed_s.view(torch.uint8))

        wkv_q = sf.get_tensor("layers.1.engram.wkv.weight")
        wkv_s = sf.get_tensor("layers.1.engram.wkv.scale")
        assert wkv_q.dtype == torch.int8
        assert wkv_s.dtype == torch.uint8
        assert wkv_s.shape == (4, 4)
        wkv_rec = wkv_q.float() * torch.ldexp(
            torch.ones_like(wkv_s, dtype=torch.float32), wkv_s.int() - 127
        ).repeat_interleave(16, 0).repeat_interleave(16, 1)
        assert (wkv_rec - engram_wkv_ref).norm() / engram_wkv_ref.norm() < 0.02

    # Verify --resume skips all 3 shards cleanly
    summary_resume = conv.convert_checkpoint(
        src_dir, dst_dir, block_size=(16, 16), use_mse=True, workers=1, resume=True
    )
    assert summary_resume["converted_linears"] == 4


def test_real_checkpoint_classification_if_present() -> None:
    real_dir = Path(
        "/root/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/master"
    )
    if not real_dir.is_dir():
        pytest.skip("Real DeepSeek-V4.1-Flash checkpoint not present")
    idx = json.loads((real_dir / "model.safetensors.index.json").read_text())
    all_keys = set(idx["weight_map"].keys())
    shard_names = sorted(set(idx["weight_map"].values()))
    n_targets = 0
    n_engram_embed = 0
    for s in shard_names:
        _, hdr = conv.read_safetensors_header(real_dir / s)
        for k, meta in hdr.items():
            if k == "__metadata__":
                continue
            if conv.is_target_fp8_linear(k, meta):
                n_targets += 1
                sc = conv.infer_scale_name(k, all_keys)
                assert sc in hdr
                assert meta["shape"][0] % 16 == 0 and meta["shape"][1] % 16 == 0
            elif conv.is_engram_embed_weight(k):
                n_engram_embed += 1
    assert n_targets == 355
    assert n_engram_embed == 2
