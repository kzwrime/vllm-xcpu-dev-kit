# GLM-5.2 MXFP4 W4A16 dummy lite

This directory is a metadata-only dummy model for XCPU integration tests.

- Architecture metadata and tokenizer assets are derived from the existing local
  GLM-5.2 7-layer lite model.
- The quantization config uses Quark MXFP4 packed weights with
  `input_tensors: null`, which means BF16 activations (W4A16).
- The model must be launched with `--load-format dummy`.
- The copied safetensors index is not a valid real-weight checkpoint for this
  configuration and is ignored by the dummy loader.
- Successful inference only validates model assembly and execution routing; it
  does not validate model accuracy.

See `docs/glm52_mxfp4_w4a16_adaptation_log.md` for design decisions and test
results.
