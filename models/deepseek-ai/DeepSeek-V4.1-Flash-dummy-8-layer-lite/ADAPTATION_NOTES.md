# DeepSeek-V4.1-Flash dummy 8-layer lite

Metadata-only dummy model for XCPU integration tests, derived from
`deepseek-ai/DeepSeek-V4.1-Flash-1017` (tokenizer, `configuration.json` and the
official `encoding/` prompt-format reference with its tests are copied unchanged
from the official release; vLLM does not read `encoding/`). Launch with
`--load-format dummy`; successful inference validates model assembly and
execution routing, not accuracy.

Changes to `config.json` (`text_config`) against the real model:

| Key | Real | Dummy | Why |
| --- | --- | --- | --- |
| `num_hidden_layers` | 40 | 8 | one layer per kind (below) |
| `compress_ratios` | 2x0, 18x2, 20x1, 3x0 | `[0,0,2,2,1,1,1,1,0,0,0]` | 8 layers + 3 DSpark |
| `kv_source_layer_ids` | `[2,8,14,20]` | `[2,4]` | ratio-2 and ratio-1 KV sources |
| `index_source_layer_ids` | `[2,8,...,36]` | `[2,4,6]` | sources and consumers of each ratio |
| `candidate_source_layer_id` / `candidate_topk_blocks` | 20 / 2048 | 4 / 128 | candidate source + later masked layers |
| `engram_layer_ids` | `[1,14]` | `[1,2]` | engram on an SWA and a compressed layer |
| `engram_vocab_size` / `engram_num_embeddings` | 16M / ~384M | 4000 / `[97968,102814]` | small tables; sizes recomputed from the prime layout |
| `n_routed_experts` | 384 | 32 | top-6 kept; N % 32 for the packed FP32-out gate |
| `dspark_target_layer_ids` / `dspark_n_routed_experts` | `[37,38,39]` / 128 | `[5,6,7]` / 32 | DSpark top-3 kept; N % 32 as above |

Layer kinds: 0 SWA; 1 SWA + engram; 2 ratio-2 kv/index source (+ engram);
3 ratio-2 consumer; 4 ratio-1 kv/index/candidate source; 5 ratio-1 index
source reading candidates; 6 ratio-1 consumer; 7 SWA (decoder replay tail).

Dummy loading leaves integer tensors uninitialized: the XCPU MXFP8 linear and
MXFP4 MoE methods set every UE8M0 scale byte to 127 (2^0), and the engram
table rows to 1.0 at scale 127.
