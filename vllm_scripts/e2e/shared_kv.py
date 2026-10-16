"""Compare GLM shared/private caches under V2, TP/DP/PCP and DFlash2."""

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path


def cache_report(worker):
    import torch
    from vllm.distributed import get_dp_group, get_ep_group, get_pcp_group, get_tp_group
    from vllm_xcpu_plugin import shared_kv_cache

    torch.mcpu.synchronize()
    runner = worker.model_runner
    shared = shared_kv_cache.runtime()
    stores = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
        for t in runner.kv_caches
    }
    report = {
        "shared": shared is not None,
        "host": socket.gethostname(),
        "bytes": sum(stores.values()),
        "tp": get_tp_group().world_size,
        "dp": get_dp_group().world_size,
        "ep": get_ep_group().world_size,
        "pcp": get_pcp_group().world_size,
        "pcp_rank": get_pcp_group().rank_in_group,
        "runner": type(runner).__name__,
        "blocks": runner.kv_cache_config.num_blocks,
    }
    if shared:
        import hashlib

        stat = os.stat(os.environ["TORCH_MCPU_SHARED_SIM_FILE"])
        report.update(
            inode=stat.st_ino,
            device=stat.st_dev,
            writer=shared.writer,
            shared_bytes=sum(t.numel() for t in shared.backing),
            cache_sha256=[
                hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()
                for t in shared.backing
            ],
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=1, choices=(1, 2, 4))
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pcp", type=int, default=1)
    parser.add_argument("--ep", action="store_true")
    parser.add_argument("--draft")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--prompt-len", type=int, default=80)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--batch-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--hostfile", help="MPI hostfile; place each TP group on one node"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mpi_args = (
            ["--hostfile", args.hostfile, "--map-by", "slot"] if args.hostfile else []
        )
        subprocess.run(
            [
                "mpirun",
                *mpi_args,
                "--bind-to",
                "none",
                "-n",
                str(args.tp * args.dp * args.pcp),
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--worker",
            ],
            check=True,
            timeout=args.timeout,
        )
        return
    from mpi4py import MPI
    from vllm.utils.network_utils import get_open_port

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    port = comm.bcast(get_open_port() if rank == 0 else None, root=0)
    address = comm.bcast(
        os.getenv("MASTER_ADDR", socket.gethostbyname(socket.gethostname()))
        if rank == 0
        else None,
        root=0,
    )
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", str(rank)),
        WORLD_SIZE=str(comm.Get_size()),
        MASTER_ADDR=address,
        MASTER_PORT=str(port),
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_CPU_USE_MPI="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_ALLOW_INSECURE_SERIALIZATION="1",
    )
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")
    os.environ.setdefault("VLLM_USE_XCPU_LINEAR", "1")
    os.environ.setdefault("VLLM_USE_XCPU_TOPK_TOPP_SAMPLER", "1")
    os.environ.setdefault("VLLM_USE_XCPU_TOPK_SOFTMAX", "1")
    os.environ.setdefault("VLLM_XCPU_QUARK_MXFP4_FORCE_W4A16", "1")
    os.environ.setdefault("VLLM_XCPU_ENABLE_DUMMY_RUN_FAST_PATH", "1")
    from vllm import LLM, SamplingParams

    options = {}
    if args.draft:
        options["speculative_config"] = {
            "method": "dflash",
            "model": args.draft,
            "num_speculative_tokens": 5,
            "kv_cache_dtype": "auto",
            "draft_sample_method": "greedy",
        }
        options["disable_hybrid_kv_cache_manager"] = True
    if args.dummy:
        options.update(
            load_format="dummy",
            skip_tokenizer_init=True,
            hf_overrides={
                "hidden_size": 256,
                "num_attention_heads": max(16, 8 * args.tp),
                "q_lora_rank": 128,
                "intermediate_size": max(256, 128 * args.tp),
                "moe_intermediate_size": max(256, 128 * args.tp),
                "n_routed_experts": 16,
                "num_experts_per_tok": 8,
                "vocab_size": 512,
            },
        )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        data_parallel_size=args.dp,
        prefill_context_parallel_size=args.pcp,
        distributed_executor_backend="external_launcher",
        enable_expert_parallel=args.ep,
        all2all_backend="mpi_alltoallv_v5",
        enforce_eager=not args.compile,
        max_model_len=args.max_len,
        max_num_batched_tokens=args.batch_tokens,
        max_num_seqs=4,
        kv_cache_memory_bytes=16 * 1024**2 if args.dummy else 256 * 1024**2,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        use_fp64_gumbel=True,
        seed=42,
        kernel_config={"enable_jit_warmup": False},
        **options,
    )
    dp_rank = rank // (args.tp * args.pcp)
    if args.dummy:
        prompts = [
            {"prompt_token_ids": [(i + 10 + dp_rank * 100) % 500 for i in range(n)]}
            for n in (args.prompt_len, args.prompt_len // 2)
        ]
    else:
        tokenizer = llm.get_tokenizer()
        prompts = [
            {
                "prompt_token_ids": tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": f"Explain topic {dp_rank}: why is the sky blue?",
                        }
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
            }
        ]
    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.tokens,
        ignore_eos=True,
        logprobs=3,
        detokenize=not args.dummy,
    )
    results = []
    for _ in range(3):
        batch = llm.generate(prompts, sampling, use_tqdm=False)
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
    all_results = comm.gather(
        {"rank": rank, "dp": dp_rank, "output": results, "cache": reports[0]}, root=0
    )
    if rank == 0:
        args.output.write_text(json.dumps(all_results, indent=2))
        print(f"PASS: generated {args.output}", flush=True)


if __name__ == "__main__":
    main()
