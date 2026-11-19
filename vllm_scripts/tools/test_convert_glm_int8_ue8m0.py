#!/usr/bin/env python3
"""Unit tests for convert_glm_int8_ue8m0.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_glm_int8_ue8m0 as conv


def _ue8m0_to_float(u: torch.Tensor) -> torch.Tensor:
    return torch.ldexp(torch.ones_like(u, dtype=torch.float32), u.to(torch.int32) - 127)


def _dequant_int8_ref(w_i8: torch.Tensor, s_u8: torch.Tensor, bn: int, bk: int) -> torch.Tensor:
    s_f32 = _ue8m0_to_float(s_u8).repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)
    return w_i8.float() * s_f32


def test_quantize_block_int8_ue8m0_roundtrip_and_zero_blocks():
    torch.manual_seed(42)
    w = torch.randn(32, 48, dtype=torch.float32) * 0.05
    # Inject a completely zero [16, 16] block and an extreme outlier block
    w[:16, 16:32] = 0.0
    w[16:32, 32:48] = -256.0

    q_mse, s_mse = conv.quantize_block_int8_ue8m0(w, 16, 16, use_mse=True)
    q_ceil, s_ceil = conv.quantize_block_int8_ue8m0(w, 16, 16, use_mse=False)

    assert q_mse.dtype == torch.int8
    assert s_mse.dtype == torch.uint8
    assert q_mse.shape == (32, 48)
    assert s_mse.shape == (2, 3)

    # Zero block encodes unit scale (127) and all-zero int8 values
    assert int(s_mse[0, 1].item()) == 127
    assert torch.equal(q_mse[:16, 16:32], torch.zeros(16, 16, dtype=torch.int8))

    # All scales remain within the SIMD fast path [1, 247]
    assert int(s_mse.min().item()) >= 1
    assert int(s_mse.max().item()) <= 247

    # Exact edge: -256 = -128 * 2^1 -> u = 128, q = -128 (zero error!)
    dq_mse = _dequant_int8_ref(q_mse, s_mse, 16, 16)
    dq_ceil = _dequant_int8_ref(q_ceil, s_ceil, 16, 16)
    assert torch.allclose(dq_mse[16:32, 32:48], w[16:32, 32:48], atol=0.0, rtol=0.0)

    # MSE search <= ceiling-only error
    err_mse = (dq_mse - w).square().sum().item()
    err_ceil = (dq_ceil - w).square().sum().item()
    assert err_mse <= err_ceil + 1e-7
    rel_err = torch.linalg.norm(dq_mse - w) / torch.linalg.norm(w)
    assert rel_err.item() < 0.015


def _quantize_to_fp8_pair(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = w.shape
    sn = (n + 127) // 128
    sk = (k + 127) // 128
    padded = torch.zeros((sn * 128, sk * 128), dtype=torch.float32)
    padded[:n, :k] = w.float()
    blocks = padded.view(sn, 128, sk, 128).permute(0, 2, 1, 3)
    amax = blocks.abs().amax(dim=(-1, -2)).clamp(min=1e-12)
    scale_inv = (amax / 448.0).to(torch.float32)
    q = (blocks / scale_inv.view(sn, sk, 1, 1)).permute(0, 2, 1, 3).reshape(sn * 128, sk * 128)
    w_fp8 = q[:n, :k].to(torch.float8_e4m3fn)
    return w_fp8, scale_inv


def test_convert_fp8_and_bf16_equivalence(tmp_path: Path):
    torch.manual_seed(0)
    src_fp8 = tmp_path / "src_fp8"
    src_bf16 = tmp_path / "src_bf16"
    dst_fp8 = tmp_path / "dst_fp8"
    dst_bf16 = tmp_path / "dst_bf16"
    src_fp8.mkdir()
    src_bf16.mkdir()

    # Representative subset of GLM-5.2/GLM-5.3 tensors across all types:
    raw_weights = {
        "lm_head.weight": torch.randn(64, 64, dtype=torch.bfloat16),
        "model.embed_tokens.weight": torch.randn(64, 64, dtype=torch.bfloat16),
        "model.norm.weight": torch.randn(64, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.randn(64, dtype=torch.bfloat16),
        "model.layers.0.self_attn.indexer.k_norm.weight": torch.randn(128, dtype=torch.bfloat16),
        "model.layers.0.self_attn.indexer.k_norm.bias": torch.randn(128, dtype=torch.bfloat16),
        "model.layers.0.self_attn.indexer.weights_proj.weight": torch.randn(32, 64, dtype=torch.bfloat16),
        "model.layers.3.mlp.gate.weight": torch.randn(16, 64, dtype=torch.bfloat16),
        "model.layers.3.mlp.gate.e_score_correction_bias": torch.randn(16, dtype=torch.float32) * 0.01,
        "model.layers.78.eh_proj.weight": torch.randn(64, 128, dtype=torch.bfloat16),
        # Quantized weights:
        "model.layers.0.self_attn.q_a_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": torch.randn(64, 128, dtype=torch.bfloat16),
        "model.layers.0.self_attn.indexer.wk.weight": torch.randn(128, 256, dtype=torch.bfloat16),
        "model.layers.3.mlp.experts.0.gate_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "model.layers.3.mlp.shared_experts.down_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
    }
    quant_keys = {
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.indexer.wk.weight",
        "model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.layers.3.mlp.shared_experts.down_proj.weight",
    }

    fp8_tensors: dict[str, torch.Tensor] = {}
    for k, v in raw_weights.items():
        if k in quant_keys:
            w_fp8, s_inv = _quantize_to_fp8_pair(v)
            fp8_tensors[k] = w_fp8
            fp8_tensors[f"{k}_scale_inv"] = s_inv
        else:
            fp8_tensors[k] = v.clone()

    save_file(fp8_tensors, str(src_fp8 / "model-00001-of-00001.safetensors"))
    save_file(raw_weights, str(src_bf16 / "model-00001-of-00001.safetensors"))

    idx_fp8 = {
        "metadata": {"total_size": 123},
        "weight_map": {k: "model-00001-of-00001.safetensors" for k in fp8_tensors},
    }
    idx_bf16 = {
        "metadata": {"total_size": 123},
        "weight_map": {k: "model-00001-of-00001.safetensors" for k in raw_weights},
    }
    (src_fp8 / "model.safetensors.index.json").write_text(json.dumps(idx_fp8))
    (src_bf16 / "model.safetensors.index.json").write_text(json.dumps(idx_bf16))

    # Reproduce ZhipuAI's typo in FP8 config.json ("self_attn.indexers_proj")
    base_cfg = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "dtype": "bfloat16",
        "hidden_size": 64,
    }
    fp8_cfg = dict(base_cfg)
    fp8_cfg["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [
            "model.layers.0.self_attn.indexers_proj",
            "model.layers.3.mlp.gate",
            "model.layers.3.mlp.gate.e_score_correction_bias",
        ],
    }
    (src_fp8 / "config.json").write_text(json.dumps(fp8_cfg))
    (src_bf16 / "config.json").write_text(json.dumps(base_cfg))
    for d in (src_fp8, src_bf16):
        (d / "tokenizer_config.json").write_text('{"tokenizer_class": "PreTrainedTokenizerFast"}')

    conv.convert_checkpoint(str(src_fp8), str(dst_fp8), block_size=(16, 16), workers=1)
    conv.convert_checkpoint(str(src_bf16), str(dst_bf16), block_size=(16, 16), workers=1)

    out_fp8 = load_file(str(dst_fp8 / "model-00001-of-00001.safetensors"))
    out_bf16 = load_file(str(dst_bf16 / "model-00001-of-00001.safetensors"))

    indexer_wk_key = "model.layers.0.self_attn.indexer.wk.weight"
    converted_keys = quant_keys - {indexer_wk_key}

    # indexer.wk stays verbatim in its source precision (FP8+FP32 scale or BF16)
    assert out_fp8[indexer_wk_key].dtype == torch.float8_e4m3fn
    assert out_fp8[f"{indexer_wk_key}_scale_inv"].dtype == torch.float32
    assert torch.equal(
        out_fp8[indexer_wk_key].view(torch.uint8),
        fp8_tensors[indexer_wk_key].view(torch.uint8),
    )
    assert torch.equal(
        out_fp8[f"{indexer_wk_key}_scale_inv"],
        fp8_tensors[f"{indexer_wk_key}_scale_inv"],
    )
    assert out_bf16[indexer_wk_key].dtype == torch.bfloat16
    assert f"{indexer_wk_key}_scale_inv" not in out_bf16
    assert torch.equal(out_bf16[indexer_wk_key], raw_weights[indexer_wk_key])

    # All non-indexer.wk keys match 100% in key set, dtype, and shape
    assert set(out_fp8.keys()) - {f"{indexer_wk_key}_scale_inv"} == set(out_bf16.keys())
    for k in out_bf16:
        if k == indexer_wk_key:
            continue
        assert out_fp8[k].dtype == out_bf16[k].dtype, f"dtype mismatch for {k}"
        assert out_fp8[k].shape == out_bf16[k].shape, f"shape mismatch for {k}"

    # Verify quantized vs kept dtypes & exact FP32 / BF16 preservation
    for k, orig in raw_weights.items():
        if k == indexer_wk_key:
            continue
        if k in converted_keys:
            assert out_fp8[k].dtype == torch.int8
            assert out_fp8[f"{k}_scale_inv"].dtype == torch.uint8
            n, m = orig.shape
            assert out_fp8[f"{k}_scale_inv"].shape == (n // 16, m // 16)
        else:
            assert f"{k}_scale_inv" not in out_fp8
            assert out_fp8[k].dtype == orig.dtype
            assert torch.equal(out_fp8[k], orig)
            assert out_bf16[k].dtype == orig.dtype
            assert torch.equal(out_bf16[k], orig)

    # Specifically double check FP32 bias stayed FP32 and weights_proj stayed BF16
    bias_key = "model.layers.3.mlp.gate.e_score_correction_bias"
    assert out_fp8[bias_key].dtype == torch.float32
    assert out_bf16[bias_key].dtype == torch.float32

    cfg_fp8 = json.loads((dst_fp8 / "config.json").read_text())
    cfg_bf16 = json.loads((dst_bf16 / "config.json").read_text())
    for cfg in (cfg_fp8, cfg_bf16):
        qcfg = cfg["quantization_config"]
        assert qcfg["quant_method"] == "xcpu_int8"
        assert qcfg["fmt"] == "int8_ue8m0"
        assert qcfg["weight_block_size"] == [16, 16]
        assert "model.layers.0.self_attn.indexer.weights_proj" in qcfg["modules_to_not_convert"]
        assert "model.layers.3.mlp.gate" in qcfg["modules_to_not_convert"]
        assert (dst_fp8 / "tokenizer_config.json").exists()
        assert (dst_bf16 / "tokenizer_config.json").exists()


def test_glm52_real_index_classification():
    real_idx_path = Path(
        "/root/.cache/modelscope/hub/models/ZhipuAI/GLM-5.2-FP8/model.safetensors.index.json"
    )
    if not real_idx_path.exists():
        pytest.skip("GLM-5.2-FP8 index not present locally")
    idx = json.loads(real_idx_path.read_text())
    wm = idx["weight_map"]
    scale_bases = {k[: -len("_scale_inv")] for k in wm if k.endswith("_scale_inv")}
    all_params = [k for k in wm if not k.endswith("_scale_inv")]
    fp8_kept = set(all_params) - scale_bases

    probe_2d = torch.empty((0, 0), dtype=torch.bfloat16)
    pred_kept = {
        k
        for k in all_params
        if conv.should_keep_unquantized_non_fp8(k, probe_2d, set())
    }
    assert fp8_kept == pred_kept
    assert len(scale_bases) == 59044
    assert len(fp8_kept) == 541

    indexer_wks = {k for k in all_params if conv.is_indexer_wk_weight(k)}
    assert len(indexer_wks) == 22
    assert indexer_wks.issubset(scale_bases)
    assert all(not conv.should_quantize_target(k, "all") for k in indexer_wks)
    assert all(not conv.should_quantize_target(k, "moe_w13") for k in indexer_wks)
    assert sum(conv.should_quantize_target(k, "all") for k in scale_bases) == 59044 - 22

    moe_w13 = {k for k in all_params if conv.is_moe_w13_weight(k)}
    assert len(moe_w13) == 76 * 256 * 2
    assert moe_w13.issubset(scale_bases)
    assert all(
        (".mlp.experts." in k) and (k.endswith(".gate_proj.weight") or k.endswith(".up_proj.weight"))
        for k in moe_w13
    )
    assert not any(k.endswith(".down_proj.weight") for k in moe_w13)
    assert not any(".shared_experts." in k for k in moe_w13)


def test_convert_scope_moe_w13_preserves_other_precision(tmp_path: Path):
    torch.manual_seed(7)
    src_fp8 = tmp_path / "src_fp8"
    src_bf16 = tmp_path / "src_bf16"
    dst_fp8 = tmp_path / "dst_fp8_w13"
    dst_bf16 = tmp_path / "dst_bf16_w13"
    src_fp8.mkdir()
    src_bf16.mkdir()

    raw_weights = {
        "model.embed_tokens.weight": torch.randn(64, 64, dtype=torch.bfloat16),
        "model.layers.3.mlp.gate.weight": torch.randn(16, 64, dtype=torch.bfloat16),
        "model.layers.3.mlp.gate.e_score_correction_bias": torch.randn(16, dtype=torch.float32) * 0.01,
        "model.layers.0.self_attn.q_a_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "model.layers.3.mlp.shared_experts.gate_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "model.layers.3.mlp.shared_experts.down_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        # Routed MoE experts: w13 (gate_proj, up_proj) vs w2 (down_proj)
        "model.layers.3.mlp.experts.0.gate_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "model.layers.3.mlp.experts.0.up_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "model.layers.3.mlp.experts.0.down_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
    }
    fp8_quant_keys = {
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.3.mlp.shared_experts.gate_proj.weight",
        "model.layers.3.mlp.shared_experts.down_proj.weight",
        "model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.layers.3.mlp.experts.0.up_proj.weight",
        "model.layers.3.mlp.experts.0.down_proj.weight",
    }
    w13_keys = {
        "model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.layers.3.mlp.experts.0.up_proj.weight",
    }

    fp8_tensors: dict[str, torch.Tensor] = {}
    for k, v in raw_weights.items():
        if k in fp8_quant_keys:
            w_fp8, s_inv = _quantize_to_fp8_pair(v)
            fp8_tensors[k] = w_fp8
            fp8_tensors[f"{k}_scale_inv"] = s_inv
        else:
            fp8_tensors[k] = v.clone()

    save_file(fp8_tensors, str(src_fp8 / "model-00001-of-00001.safetensors"))
    save_file(raw_weights, str(src_bf16 / "model-00001-of-00001.safetensors"))

    (src_fp8 / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 123},
                "weight_map": {k: "model-00001-of-00001.safetensors" for k in fp8_tensors},
            }
        )
    )
    (src_bf16 / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 123},
                "weight_map": {k: "model-00001-of-00001.safetensors" for k in raw_weights},
            }
        )
    )

    base_cfg = {"architectures": ["GlmMoeDsaForCausalLM"], "dtype": "bfloat16", "hidden_size": 64}
    fp8_cfg = dict(base_cfg)
    fp8_cfg["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["model.layers.3.mlp.gate"],
    }
    (src_fp8 / "config.json").write_text(json.dumps(fp8_cfg))
    (src_bf16 / "config.json").write_text(json.dumps(base_cfg))

    conv.convert_checkpoint(
        str(src_fp8), str(dst_fp8), block_size=(16, 16), scope="moe_w13", workers=1
    )
    conv.convert_checkpoint(
        str(src_bf16), str(dst_bf16), block_size=(16, 16), scope="moe_w13", workers=1
    )

    out_fp8 = load_file(str(dst_fp8 / "model-00001-of-00001.safetensors"))
    out_bf16 = load_file(str(dst_bf16 / "model-00001-of-00001.safetensors"))

    # 1. FP8 source: ONLY w13 becomes int8 + uint8 [16, 16]; other FP8 tensors keep exact FP8 + FP32 scale.
    for k in raw_weights:
        if k in w13_keys:
            assert out_fp8[k].dtype == torch.int8
            assert out_fp8[f"{k}_scale_inv"].dtype == torch.uint8
            assert out_fp8[f"{k}_scale_inv"].shape == (8, 8)
        elif k in fp8_quant_keys:
            assert out_fp8[k].dtype == torch.float8_e4m3fn
            assert out_fp8[f"{k}_scale_inv"].dtype == torch.float32
            assert torch.equal(out_fp8[k].view(torch.uint8), fp8_tensors[k].view(torch.uint8))
            assert torch.equal(out_fp8[f"{k}_scale_inv"], fp8_tensors[f"{k}_scale_inv"])
        else:
            assert f"{k}_scale_inv" not in out_fp8
            assert out_fp8[k].dtype == raw_weights[k].dtype
            assert torch.equal(out_fp8[k], raw_weights[k])

    # 2. BF16 source: ONLY w13 becomes int8 + uint8 [16, 16]; everything else stays exact BF16/FP32.
    for k, orig in raw_weights.items():
        if k in w13_keys:
            assert out_bf16[k].dtype == torch.int8
            assert out_bf16[f"{k}_scale_inv"].dtype == torch.uint8
            assert out_bf16[f"{k}_scale_inv"].shape == (8, 8)
        else:
            assert f"{k}_scale_inv" not in out_bf16
            assert out_bf16[k].dtype == orig.dtype
            assert torch.equal(out_bf16[k], orig)

    # 3. Verify config.json metadata for both cases
    qcfg_fp8 = json.loads((dst_fp8 / "config.json").read_text())["quantization_config"]
    assert qcfg_fp8["quant_method"] == "fp8"
    assert qcfg_fp8["weight_block_size"] == [128, 128]
    assert qcfg_fp8["xcpu_int8_overrides"] == {
        "fmt": "int8_ue8m0",
        "target_scope": "moe_w13",
        "weight_block_size": [16, 16],
    }
    assert "model.layers.3.mlp.experts.0.down_proj" not in qcfg_fp8["modules_to_not_convert"]

    qcfg_bf16 = json.loads((dst_bf16 / "config.json").read_text())["quantization_config"]
    assert qcfg_bf16["quant_method"] == "xcpu_int8"
    assert qcfg_bf16["target_scope"] == "moe_w13"
    assert qcfg_bf16["weight_block_size"] == [16, 16]
    assert "model.layers.3.mlp.experts.0.down_proj" in qcfg_bf16["modules_to_not_convert"]
    assert "model.layers.0.self_attn.q_a_proj" in qcfg_bf16["modules_to_not_convert"]

    # 4. Resume on existing moe_w13 output succeeds and preserves config cleanly
    conv.convert_checkpoint(
        str(src_fp8), str(dst_fp8), block_size=(16, 16), scope="moe_w13", workers=1, resume=True
    )
    qcfg_fp8_resumed = json.loads((dst_fp8 / "config.json").read_text())["quantization_config"]
    assert qcfg_fp8_resumed == qcfg_fp8
