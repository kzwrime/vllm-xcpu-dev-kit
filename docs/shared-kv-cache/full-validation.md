# GLM5.2 shared KV cache check

Requires [torch_mcpu #5](https://github.com/kzwrime/torch_mcpu/pull/5) and
[plugin #31](https://github.com/kzwrime/vllm-xcpu-plugin/pull/31).
Run from the kit root using its Python environment and a local full FP8 or MXFP4
checkpoint. The runner uses one x86 host, DP2/TP2/EP4, eager V1, synchronous
scheduling, FP8 KV, and a 128 MiB logical cache budget per rank.

```bash
set -euo pipefail
model=/path/to/GLM-5.2-MXFP4
weights=MXFP4-W4A16  # For FP8: set weights=FP8 and label=fp8.
label=mxfp4
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
unset TORCH_MCPU_SHARED_BASE
mkdir -p logs
run_dir=$(mktemp -d "$PWD/logs/shared-kv.XXXXXX")
export TORCH_MCPU_SHARED_SIZE=268435456
export TORCH_MCPU_SHARED_SIM_FILE="/dev/shm/$(basename "$run_dir")"
for mode in private shared; do
  export VLLM_XCPU_SHARED_KV_CACHE=0
  if [ "$mode" = shared ]; then export VLLM_XCPU_SHARED_KV_CACHE=1; fi
  timeout -k 30 14400 python docs/shared-kv-cache/model-smoke.py \
    --model "$model" --weights "$weights" --dp-size 2 --expert-parallel \
    --timeout 14000 --output "$run_dir/$mode-$label.json" \
    > "$run_dir/$mode-$label.log" 2>&1
done
python docs/shared-kv-cache/compare-full.py "$run_dir" --weights "$label"
```

The comparison checks exact tokens/text/logprobs, finite logprobs, equal capacity,
shared inode/content within TP pairs, one writer per pair, DP isolation, and 50%
less unique cache backing. Remove arena files only after all workers exit.

Both real-weight formats passed this check. Requests generate six tokens twice;
this does not cover prefix cache hits, long contexts, multiple hosts, performance,
or real hardware. FP8 needed about 743 GiB resident memory; MXFP4 about 439 GiB.
Generated outputs and logs stay local.
