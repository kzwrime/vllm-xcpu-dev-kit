"""GLM shared/private generation check, with reduced dummy or full real weights."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def cache_report(worker):
    import hashlib

    import torch
    from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
    from vllm_xcpu_plugin import shared_kv_cache

    runner = worker.model_runner
    torch.mcpu.synchronize()
    storages = {}
    for tensor in runner.kv_caches:
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
    shared = shared_kv_cache.runtime()
    report = {
        "shared": shared is not None,
        "bytes": sum(storages.values()),
        "num_blocks": runner.kv_cache_config.num_blocks,
        "tp_size": get_tp_group().world_size,
        "dp_size": get_dp_group().world_size,
        "ep_size": get_ep_group().world_size,
    }
    if shared is not None:
        path = os.environ["TORCH_MCPU_SHARED_SIM_FILE"]
        stat = os.stat(path)
        report.update(
            writer=shared.writer,
            file=path,
            inode=stat.st_ino,
            cache_sha256=[
                hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()
                for t in shared.backing
            ],
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", choices=["FP8", "MXFP4-W4A16"], default="FP8")
    parser.add_argument("--dp-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--expert-parallel", action="store_true")
    parser.add_argument("--model", help="Full real checkpoint; no shape overrides")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--mpi-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.dp_size == 2:
        command = [
            "mpirun",
            "--bind-to",
            "none",
            "-n",
            "4",
            sys.executable,
            __file__,
            "--mpi-worker",
            "--output",
            args.output,
            "--weights",
            args.weights,
        ]
        if args.expert_parallel:
            command.append("--expert-parallel")
        if args.model:
            command.extend(["--model", args.model])
        subprocess.run(command, check=True, timeout=args.timeout)
        if os.getenv("VLLM_XCPU_SHARED_KV_CACHE") == "1":
            reports = [
                json.loads(Path(args.output + f".dp{rank}.cache.json").read_text())
                for rank in range(2)
            ]
            assert reports[0][0]["inode"] != reports[1][0]["inode"]
        print("PASS: DP2 generation and cache checks", flush=True)
        return
    distributed_options = {}
    dp_rank = 0
    if args.mpi_worker:
        from mpi4py import MPI
        from vllm.utils.network_utils import get_open_port

        rank = MPI.COMM_WORLD.Get_rank()
        assert MPI.COMM_WORLD.Get_size() == 4
        port = MPI.COMM_WORLD.bcast(get_open_port() if rank == 0 else None, root=0)
        dp_rank = rank // 2
        os.environ.update(
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE="4",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            VLLM_ENABLE_V1_MULTIPROCESSING="0",
            VLLM_CPU_USE_MPI="1",
        )
        distributed_options = dict(
            data_parallel_size=2, distributed_executor_backend="external_launcher"
        )
        args.output += f".dp{dp_rank}"
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("VLLM_CPU_USE_MPI", "0")
    os.environ.setdefault("VLLM_USE_XCPU_LINEAR", "1")
    os.environ.setdefault("VLLM_USE_XCPU_TOPK_TOPP_SAMPLER", "1")
    os.environ.setdefault("VLLM_USE_XCPU_TOPK_SOFTMAX", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Trusted local test callback used by collective_rpc below.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ.setdefault("VLLM_XCPU_ENABLE_DUMMY_RUN_FAST_PATH", "0")
    os.environ.setdefault("VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16", "1")
    from vllm import LLM, SamplingParams

    if os.getenv("VLLM_XCPU_SHARED_KV_CACHE", "0") != "1":
        os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")

    shape_overrides = (
        {}
        if args.model
        else dict(
            hidden_size=256,
            num_attention_heads=16,
            q_lora_rank=128,
            intermediate_size=256,
            moe_intermediate_size=256,
            n_routed_experts=8 if args.mpi_worker else 4,
            num_experts_per_tok=8 if args.mpi_worker else 2,
            vocab_size=512,
        )
    )
    llm = LLM(
        model=args.model
        or (
            str(
                Path(__file__).resolve().parents[2]
                / f"models/ZhipuAI/GLM-5.2-{args.weights}-dummy-7-layer-lite"
            )
        ),
        load_format="safetensors" if args.model else "dummy",
        skip_tokenizer_init=not args.model,
        tensor_parallel_size=2,
        enable_expert_parallel=args.expert_parallel,
        all2all_backend="mpi_alltoallv_v6" if args.mpi_worker else "all_to_all_single",
        **distributed_options,
        enforce_eager=True,
        async_scheduling=False,
        seed=42,
        max_model_len=256,
        max_num_batched_tokens=32,
        max_num_seqs=2,
        kv_cache_memory_bytes=(128 if args.model else 8) * 1024 * 1024,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        hf_overrides=shape_overrides,
        kernel_config={"enable_jit_warmup": False},
    )
    shift = 100 * dp_rank
    prompts = [
        {"prompt_token_ids": list(range(10 + shift, 90 + shift))},
        {"prompt_token_ids": list(range(20 + shift, 60 + shift))},
    ]
    if args.model:
        text = (
            "Explain in one short sentence why the sky is blue."
            if dp_rank == 0
            else "用一句话解释为什么水会结冰。"
        )
        tokenizer = llm.get_tokenizer()
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        assert isinstance(tokens, list) and all(isinstance(t, int) for t in tokens)
        prompts = [{"prompt_token_ids": tokens}]
    print(f"READY: dp={dp_rank} real_weights={bool(args.model)}", flush=True)
    sampling = SamplingParams(
        temperature=0,
        max_tokens=6,
        ignore_eos=True,
        logprobs=3,
        detokenize=bool(args.model),
    )
    results = []
    for iteration in range(2):
        batch = llm.generate(prompts, sampling, use_tqdm=False)
        print(f"GENERATED: dp={dp_rank} iteration={iteration + 1}", flush=True)
        results.append(
            [
                {
                    "tokens": r.outputs[0].token_ids,
                    "text": r.outputs[0].text,
                    "logprobs": [
                        {str(k): v.logprob for k, v in row.items()}
                        for row in r.outputs[0].logprobs
                    ],
                }
                for r in batch
            ]
        )
    reports = llm.collective_rpc(cache_report)
    if args.mpi_worker:
        all_reports = MPI.COMM_WORLD.allgather(reports[0])
        reports = all_reports[dp_rank * 2 : dp_rank * 2 + 2]
        if os.getenv("VLLM_XCPU_SHARED_KV_CACHE") == "1":
            assert all_reports[0]["inode"] != all_reports[2]["inode"]
        if rank % 2:
            return
    Path(args.output).write_text(json.dumps(results, indent=2))
    assert len(reports) == 2
    assert reports[0]["bytes"] == reports[1]["bytes"]
    if args.expert_parallel:
        assert all(r["ep_size"] == (4 if args.mpi_worker else 2) for r in reports)
    if os.getenv("VLLM_XCPU_SHARED_KV_CACHE") == "1":
        assert all(r["shared"] for r in reports)
        assert reports[0]["inode"] == reports[1]["inode"]
        assert reports[0]["cache_sha256"] == reports[1]["cache_sha256"]
        assert sum(r["writer"] for r in reports) == 1
    else:
        assert not any(r["shared"] for r in reports)
    Path(args.output + ".cache.json").write_text(json.dumps(reports, indent=2))
    print(
        "PASS: generation and cache checks"
        if args.model
        else "PASS: chunked prefill, decode, repeated prefixes",
        flush=True,
    )


if __name__ == "__main__":
    main()
