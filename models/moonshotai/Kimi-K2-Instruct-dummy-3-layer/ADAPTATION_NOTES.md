# Kimi K2 Isomorphic 3-Layer Dummy Model Adaptation Notes

## 1. Source & Structure
- **Upstream checkpoint**: `moonshotai/Kimi-K2-Instruct` (ModelScope metadata only; no safetensors weights downloaded; unused `modeling_deepseek.py` removed since vLLM uses native `DeepseekV3ForCausalLM`, and MIT `LICENSE` retained).
- **Local weight verification**: No full `DeepSeek-V3` or `Kimi-K2` `.safetensors` shards exist on this host (and a full 61-layer BF16/FP8 `Kimi-K2` model exceeds ~1 TB RAM).
- **Isomorphism**: `config.json` keeps exact `Kimi-K2-Instruct` architecture and quantization geometry while reducing `num_hidden_layers` from `61` to `3` (`first_k_dense_replace: 1` -> 1 dense layer + 2 MoE layers):
  - `model_type`: `"kimi_k2"` (`DeepseekV3ForCausalLM` / `DeepseekV3Config`)
  - `hidden_size = 7168`, `intermediate_size = 18432`, `moe_intermediate_size = 2048`
  - `num_attention_heads = 64`, `num_key_value_heads = 64`, `q_lora_rank = 1536`, `kv_lora_rank = 512`
  - `qk_nope_head_dim = 128`, `qk_rope_head_dim = 64`, `v_head_dim = 128` (Dense MLA, `is_sparse = False`, YaRN RoPE)
  - `n_routed_experts = 384`, `num_experts_per_tok = 8`, `n_shared_experts = 1`
  - `quantization_config`: `{"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128], "activation_scheme": "dynamic"}`

## 2. Running Mode (Dequant -> BF16 Linear -> Quant/Cache)
Run with `--load-format dummy --trust-remote-code --kv-cache-dtype fp8` and:
- `TORCH_XCPU_FP8_LINEAR_BACKEND=acc`
- `TORCH_XCPU_BF16_LINEAR_BACKEND=acc`
- `TORCH_XCPU_MOE_GROUPED_GEMM_BACKEND=acc`
- `TORCH_XCPU_ACC_FP8_DEQUANT_BF16_LINEAR=1`

When `TORCH_XCPU_ACC_FP8_DEQUANT_BF16_LINEAR=1` is set before ACC initialization, `AccFp8LinearKernel` explicitly decomposes every FP8 block-scaled linear / ACC MoE grouped GEMM into:
1. Temporary packed FP8 -> ACC packed BF16 dequantization (`mcpu::acc_fp8_dequant_packed_bf16`)
2. Regular BF16 matrix multiplication (`mcpu::bf16_linear_acc` / `mcpu::bf16_linear_acc_bias`)
Activations remain BF16 (W8A16), and MLA KV cache writes use FP8 (`mcpu::fused_mla_rope_qproj_cat_cache` / `mcpu::unified_attention_mla_fp8_ds`).
