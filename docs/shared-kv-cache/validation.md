# Shared KV cache validation

Use the [full-model check](full-validation.md) for FP8 and MXFP4 checkpoints.
The scripts compare generation and physical cache ownership; implementation and
supported configurations are documented in the
[plugin](https://github.com/kzwrime/vllm-xcpu-plugin/blob/feat/shared-kv-cache/docs/shared_kv_cache.md).

For a faster dummy-weight check, use the same commands without `--model`.
Place reduced seven-layer model metadata in
`models/ZhipuAI/GLM-5.2-{FP8,MXFP4-W4A16}-dummy-7-layer-lite` first.
This uses an 8 MiB cache budget and longer token prompts for chunked prefill,
decode, and repeated prefixes. It does not validate pretrained-model quality.
