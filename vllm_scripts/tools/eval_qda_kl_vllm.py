#!/usr/bin/env python3
"""Lightweight end-to-end KL-divergence & generation guardrail evaluator for
quantized LLMs served via vLLM, inspired by QDA (humanfia/qda, qda-kl[vllm]).

Methodology:
  1. Reference collection (`--mode collect-ref` against baseline FP8/BF16 server P):
     - Generates unseen reference reasoning/coding/math/multilingual traces.
     - Extracts teacher-forced top-K next-token logprobs (`prompt_logprobs=K`)
       on the exact token IDs of every trace in a single prefill pass per trace.
     - Runs long-generation guardrail prompts (greedy, e.g. 256 tokens) to
       record baseline outputs & repetition stats.
  2. Candidate evaluation (`--mode eval-cand` against quantized server Q):
     - Scores the exact same reference token ID sequences with `prompt_logprobs=K`.
     - Computes exact union + tail-bucket Forward KL D_KL(P || Q), Reverse KL
       D_KL(Q || P), JSD(P, Q), Top-1/Top-5 agreement, exact trace-token NLL /
       PPL delta, logprob MAE, and P50/P95/P99/Max KL quantiles.
     - Runs the same long-generation guardrail prompts and reports prefix match,
       n-gram repetition ratio (loop detector), and side-by-side completions.

Usage:
  # 1. Against reference vLLM server (started with --max-logprobs 64):
  python3 eval_qda_kl_vllm.py --mode collect-ref \
      --base-url http://127.0.0.1:14800 --model <ref_model> \
      --top-k 64 --out-dir ./qda_eval_run

  # 2. Against quantized candidate vLLM server (started with --max-logprobs 64):
  python3 eval_qda_kl_vllm.py --mode eval-cand \
      --base-url http://127.0.0.1:14800 --model <cand_model> \
      --out-dir ./qda_eval_run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Unseen evaluation prompts spanning math reasoning, algorithms/coding,
# systems architecture, logic deduction, physics, and Chinese reasoning.
DEFAULT_TRACE_PROMPTS: list[dict[str, str]] = [
    {
        "id": "math_combinatorics_en",
        "category": "math",
        "prompt": (
            "Solve step by step: How many positive integers less than 1000 are "
            "divisible by 3 or 5, but not by 15? Show your inclusion-exclusion "
            "derivation clearly and give the final integer."
        ),
    },
    {
        "id": "math_probability_zh",
        "category": "math_zh",
        "prompt": (
            "请分步推导：一个袋子里有 4 个红球和 6 个蓝球，不放回连续随机抽取 3 个球，"
            "恰好抽到 2 个红球和 1 个蓝球的概率是多少？请给出最简分数和小数结果。"
        ),
    },
    {
        "id": "code_dp_en",
        "category": "code",
        "prompt": (
            "Write a clean Python function `longest_increasing_subseq_len(nums)` "
            "running in O(n log n) time using binary search, briefly prove why "
            "the tails invariant holds, and trace it on [10, 9, 2, 5, 3, 7, 101, 18]."
        ),
    },
    {
        "id": "systems_quant_zh",
        "category": "systems",
        "prompt": (
            "请简明分析大语言模型推理中 W8A16 分块量化（如 Int8 + UE8M0 16x16 块缩放）"
            "相比 FP8 32x32 块量化在动态范围、尾数精度、离群值敏感度与反量化开销方面的权衡。"
        ),
    },
    {
        "id": "logic_puzzle_en",
        "category": "logic",
        "prompt": (
            "Four colleagues (Alice, Bob, Carol, Dave) each work on exactly one "
            "different language: Rust, Python, Go, or C++. Clues: (1) Alice does "
            "not use Python or C++. (2) Bob uses either Rust or C++. (3) Carol "
            "does not use Go or Rust. (4) Dave uses Go. Deduce each person's "
            "language step by step."
        ),
    },
    {
        "id": "physics_kinematics_en",
        "category": "science",
        "prompt": (
            "A 2.0 kg block slides down a 30-degree incline of length 5.0 m from "
            "rest with kinetic friction coefficient mu_k = 0.20 (take g = 9.8 m/s^2). "
            "Calculate the net acceleration, final speed at the bottom, and work "
            "done by friction."
        ),
    },
    {
        "id": "chinese_reasoning_zh",
        "category": "reasoning_zh",
        "prompt": (
            "请用严谨的逻辑解释为什么在分布式 MoE 模型推理中，将 Shared Expert 融合进 "
            "EP All-to-All 通信路径可以掩盖通信延迟，并指出其生效的前提条件。"
        ),
    },
    {
        "id": "structured_json_en",
        "category": "structured",
        "prompt": (
            "Convert the following facts into a single valid JSON object with keys "
            "`model`, `layers`, `experts` (object with `routed`, `active`, `shared`), "
            "and `quantization` (list of strings): DeepSeek-V4.1-Flash has 40 layers, "
            "384 routed experts with 6 active per token and 1 shared expert, using "
            "Int8 UE8M0 linear weights, MXFP4 routed experts, and FP8 Engram tables."
        ),
    },
]

DEFAULT_LONGGEN_PROMPTS: list[dict[str, str]] = [
    {
        "id": "longgen_math_proof",
        "category": "long_math",
        "prompt": (
            "Prove rigorously that sqrt(2) + sqrt(3) is irrational, and then "
            "find its minimal polynomial over Q. Verify all roots of the minimal "
            "polynomial."
        ),
    },
    {
        "id": "longgen_code_lru",
        "category": "long_code",
        "prompt": (
            "请用 Python 实现一个时间复杂度严格为 O(1) 的线程不安全 LRU Cache（不使用 "
            "OrderedDict，手写双向链表 + 哈希表），支持 get(key) 和 put(key, value)，"
            "并写出 3 个覆盖边界情况（容量为 1、更新已存在键、驱逐最久未使用键）的测试用例。"
        ),
    },
]


def _http_post_json(
    url: str, payload: dict[str, Any], timeout_s: float = 300.0
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:500]}") from exc


def _extract_token_dist(
    pos_entry: dict[str, Any] | None,
) -> dict[int, float]:
    """Parse one vLLM prompt_logprobs position dict into {token_id: logprob}."""
    if not pos_entry:
        return {}
    out: dict[int, float] = {}
    for k, v in pos_entry.items():
        tid = int(k)
        lp = float(v["logprob"]) if isinstance(v, dict) else float(v)
        if math.isfinite(lp):
            out[tid] = lp
    return out


def compute_token_divergence(
    p_logprobs: dict[int, float],
    q_logprobs: dict[int, float],
    actual_token_id: int | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute QDA-style distribution drift metrics from top-K logprobs.

    Constructs normalized distributions on U U {<tail>} where U = supp(P) U supp(Q),
    distributing residual tail mass conservatively so both forward KL D_KL(P||Q)
    and reverse KL D_KL(Q||P) are mathematically non-negative and tight when
    top-K covers >99% of probability mass.
    """
    if not p_logprobs or not q_logprobs:
        raise ValueError("Empty logprob distribution at scored position")

    # Convert logprobs -> probabilities; clamp sum <= 1.0 against tiny fp drift.
    p_raw = {k: math.exp(lp) for k, lp in p_logprobs.items()}
    q_raw = {k: math.exp(lp) for k, lp in q_logprobs.items()}
    p_cov = min(1.0, sum(p_raw.values()))
    q_cov = min(1.0, sum(q_raw.values()))
    p_tail_rem = max(0.0, 1.0 - p_cov)
    q_tail_rem = max(0.0, 1.0 - q_cov)

    p_min_prob = min(p_raw.values())
    q_min_prob = min(q_raw.values())

    union_keys = set(p_raw.keys()) | set(q_raw.keys())
    miss_in_q = [k for k in p_raw if k not in q_raw]
    miss_in_p = [k for k in q_raw if k not in p_raw]

    # Allocate residual mass to keys present in only one top-K list, bounded by
    # both the minimum observed top-K probability and half of remaining tail mass.
    p_floor = (
        max(eps, min(p_min_prob, 0.5 * p_tail_rem / len(miss_in_p)))
        if miss_in_p
        else eps
    )
    q_floor = (
        max(eps, min(q_min_prob, 0.5 * q_tail_rem / len(miss_in_q)))
        if miss_in_q
        else eps
    )

    p_vec: list[float] = []
    q_vec: list[float] = []
    p_alloc_miss = 0.0
    q_alloc_miss = 0.0
    for k in union_keys:
        if k in p_raw:
            pv = p_raw[k]
        else:
            pv = p_floor
            p_alloc_miss += pv
        if k in q_raw:
            qv = q_raw[k]
        else:
            qv = q_floor
            q_alloc_miss += qv
        p_vec.append(pv)
        q_vec.append(qv)

    # Residual tail bucket captures probability outside U for both distributions.
    p_tail_bucket = max(eps, p_tail_rem - p_alloc_miss)
    q_tail_bucket = max(eps, q_tail_rem - q_alloc_miss)
    p_vec.append(p_tail_bucket)
    q_vec.append(q_tail_bucket)

    # Renormalize strictly to 1.0 so Gibbs' inequality holds exactly.
    p_sum = sum(p_vec)
    q_sum = sum(q_vec)
    p_norm = [v / p_sum for v in p_vec]
    q_norm = [v / q_sum for v in q_vec]

    fwd_kl = 0.0
    rev_kl = 0.0
    jsd = 0.0
    for pv, qv in zip(p_norm, q_norm):
        mv = 0.5 * (pv + qv)
        if pv > 0.0:
            fwd_kl += pv * math.log(pv / qv)
            jsd += 0.5 * pv * math.log(pv / mv)
        if qv > 0.0:
            rev_kl += qv * math.log(qv / pv)
            jsd += 0.5 * qv * math.log(qv / mv)

    p_ranked = sorted(p_logprobs.items(), key=lambda kv: kv[1], reverse=True)
    q_ranked = sorted(q_logprobs.items(), key=lambda kv: kv[1], reverse=True)
    p_top1 = p_ranked[0][0]
    q_top1 = q_ranked[0][0]
    q_top5 = {tid for tid, _ in q_ranked[:5]}
    overlap = len(set(p_logprobs.keys()) & set(q_logprobs.keys())) / max(
        1, len(p_logprobs)
    )

    res = {
        "fwd_kl": max(0.0, fwd_kl),
        "rev_kl": max(0.0, rev_kl),
        "jsd": max(0.0, jsd),
        "p_cov": p_cov,
        "q_cov": q_cov,
        "topk_overlap": overlap,
        "top1_agree": 1.0 if p_top1 == q_top1 else 0.0,
        "top5_agree": 1.0 if p_top1 in q_top5 else 0.0,
    }
    if actual_token_id is not None:
        lp_p = p_logprobs.get(actual_token_id)
        lp_q = q_logprobs.get(actual_token_id)
        if lp_p is not None and lp_q is not None:
            res["nll_p"] = -lp_p
            res["nll_q"] = -lp_q
            res["abs_lp_err"] = abs(lp_q - lp_p)
    return res


def ngram_repetition_ratio(token_ids: list[int], n: int = 4) -> float:
    """Fraction of repeated n-grams in a token sequence (0.0 = no loops)."""
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def common_prefix_len(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def _generate_chat_trace(
    base_url: str,
    model: str,
    user_prompt: str,
    max_tokens: int,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """Generate a deterministic trace via /v1/chat/completions with exact token IDs."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "return_token_ids": True,
    }
    resp = _http_post_json(
        f"{base_url}/v1/chat/completions", payload, timeout_s=timeout_s
    )
    choice = resp["choices"][0]
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    combined_text = f"<think>{reasoning}</think>\n{content}" if reasoning else content
    prompt_ids = resp.get("prompt_token_ids")
    token_ids = choice.get("token_ids")
    if not prompt_ids or not token_ids:
        raise RuntimeError(
            "Server did not return prompt_token_ids / token_ids for "
            "return_token_ids=True"
        )
    return {
        "text": combined_text,
        "reasoning": reasoning,
        "content": content,
        "prompt_token_ids": [int(x) for x in prompt_ids],
        "token_ids": [int(x) for x in token_ids],
        "finish_reason": choice.get("finish_reason"),
        "usage": resp.get("usage", {}),
    }


def _generate_completion_from_ids(
    base_url: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "return_token_ids": True,
    }
    resp = _http_post_json(f"{base_url}/v1/completions", payload, timeout_s=timeout_s)
    choice = resp["choices"][0]
    token_ids = choice.get("token_ids")
    if not token_ids:
        raise RuntimeError("Server did not return token_ids from /v1/completions")
    return {
        "text": choice.get("text", ""),
        "token_ids": [int(x) for x in token_ids],
        "finish_reason": choice.get("finish_reason"),
        "usage": resp.get("usage", {}),
    }


def _score_prompt_logprobs(
    base_url: str,
    model: str,
    full_token_ids: list[int],
    top_k: int,
    timeout_s: float = 300.0,
) -> list[dict[str, float]]:
    payload = {
        "model": model,
        "prompt": full_token_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": top_k,
        "echo": False,
    }
    resp = _http_post_json(f"{base_url}/v1/completions", payload, timeout_s=timeout_s)
    choice = resp["choices"][0]
    raw_plp = choice.get("prompt_logprobs")
    if raw_plp is None:
        raise RuntimeError(
            "Server did not return prompt_logprobs. Make sure vLLM is launched "
            f"with --max-logprobs >= {top_k}."
        )
    if len(raw_plp) != len(full_token_ids):
        raise RuntimeError(
            f"prompt_logprobs length {len(raw_plp)} != token count {len(full_token_ids)}"
        )
    # Compact serialization: [{str(token_id): logprob}, ...] for positions 0..L-1
    compact: list[dict[str, float]] = []
    for entry in raw_plp:
        dist = _extract_token_dist(entry)
        compact.append({str(k): v for k, v in dist.items()})
    return compact


def collect_reference(
    base_url: str,
    model: str,
    out_dir: Path,
    *,
    top_k: int = 64,
    trace_max_tokens: int = 128,
    longgen_max_tokens: int = 256,
    concurrency: int = 4,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    print(
        f"[collect-ref] Generating {len(DEFAULT_TRACE_PROMPTS)} reference traces "
        f"(max_tokens={trace_max_tokens}, concurrency={concurrency})...",
        flush=True,
    )
    traces: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {
            pool.submit(
                _generate_chat_trace,
                base_url,
                model,
                item["prompt"],
                trace_max_tokens,
            ): item
            for item in DEFAULT_TRACE_PROMPTS
        }
        for fut in concurrent.futures.as_completed(futs):
            item = futs[fut]
            gen = fut.result()
            prompt_ids = gen["prompt_token_ids"]
            comp_ids = gen["token_ids"]
            full_ids = prompt_ids + comp_ids
            traces.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "prompt": item["prompt"],
                    "prompt_token_ids": prompt_ids,
                    "completion_token_ids": comp_ids,
                    "full_token_ids": full_ids,
                    "completion_text": gen["text"],
                    "finish_reason": gen["finish_reason"],
                }
            )
    traces.sort(key=lambda x: [p["id"] for p in DEFAULT_TRACE_PROMPTS].index(x["id"]))

    print(
        f"[collect-ref] Scoring teacher-forced prompt_logprobs (top_k={top_k}) "
        f"on {sum(len(t['completion_token_ids']) for t in traces)} trace tokens...",
        flush=True,
    )
    ref_logprobs: dict[str, list[dict[str, float]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {
            pool.submit(
                _score_prompt_logprobs, base_url, model, t["full_token_ids"], top_k
            ): t["id"]
            for t in traces
        }
        for fut in concurrent.futures.as_completed(futs):
            tid = futs[fut]
            ref_logprobs[tid] = fut.result()

    print(
        f"[collect-ref] Running {len(DEFAULT_LONGGEN_PROMPTS)} long-generation "
        f"guardrail checks (max_tokens={longgen_max_tokens})...",
        flush=True,
    )
    longgen_ref: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(concurrency, len(DEFAULT_LONGGEN_PROMPTS))
    ) as pool:
        futs = {
            pool.submit(
                _generate_chat_trace,
                base_url,
                model,
                item["prompt"],
                longgen_max_tokens,
            ): item
            for item in DEFAULT_LONGGEN_PROMPTS
        }
        for fut in concurrent.futures.as_completed(futs):
            item = futs[fut]
            gen = fut.result()
            comp_ids = gen["token_ids"]
            longgen_ref.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "prompt": item["prompt"],
                    "prompt_token_ids": gen["prompt_token_ids"],
                    "completion_token_ids": comp_ids,
                    "completion_text": gen["text"],
                    "finish_reason": gen["finish_reason"],
                    "rep4": ngram_repetition_ratio(comp_ids, 4),
                    "rep8": ngram_repetition_ratio(comp_ids, 8),
                }
            )
    longgen_ref.sort(
        key=lambda x: [p["id"] for p in DEFAULT_LONGGEN_PROMPTS].index(x["id"])
    )

    payload = {
        "ref_model": model,
        "top_k": top_k,
        "trace_max_tokens": trace_max_tokens,
        "longgen_max_tokens": longgen_max_tokens,
        "traces": traces,
        "ref_logprobs": ref_logprobs,
        "longgen_ref": longgen_ref,
        "elapsed_sec": round(time.monotonic() - t0, 2),
    }
    (out_dir / "reference.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[collect-ref] Saved reference bundle ({len(traces)} traces, "
        f"{sum(len(t['completion_token_ids']) for t in traces)} trace tokens) "
        f"in {payload['elapsed_sec']}s -> {out_dir / 'reference.json'}",
        flush=True,
    )
    return payload


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(
        len(sorted_vals) - 1,
        max(0, math.ceil(pct / 100.0 * len(sorted_vals)) - 1),
    )
    return sorted_vals[idx]


def _aggregate_token_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    n = len(records)
    if n == 0:
        return {"count": 0}
    fwd_kls = sorted(r["fwd_kl"] for r in records)
    rev_kls = [r["rev_kl"] for r in records]
    jsds = [r["jsd"] for r in records]
    nll_p_vals = [r["nll_p"] for r in records if "nll_p" in r]
    nll_q_vals = [r["nll_q"] for r in records if "nll_q" in r]
    abs_errs = [r["abs_lp_err"] for r in records if "abs_lp_err" in r]

    mean_nll_p = statistics.fmean(nll_p_vals) if nll_p_vals else 0.0
    mean_nll_q = statistics.fmean(nll_q_vals) if nll_q_vals else 0.0
    ppl_p = math.exp(min(20.0, mean_nll_p)) if nll_p_vals else 0.0
    ppl_q = math.exp(min(20.0, mean_nll_q)) if nll_q_vals else 0.0

    return {
        "count": n,
        "mean_fwd_kl": statistics.fmean(fwd_kls),
        "median_fwd_kl": _percentile(fwd_kls, 50.0),
        "p95_fwd_kl": _percentile(fwd_kls, 95.0),
        "p99_fwd_kl": _percentile(fwd_kls, 99.0),
        "max_fwd_kl": fwd_kls[-1],
        "mean_rev_kl": statistics.fmean(rev_kls),
        "mean_jsd": statistics.fmean(jsds),
        "top1_agree": statistics.fmean(r["top1_agree"] for r in records),
        "top5_agree": statistics.fmean(r["top5_agree"] for r in records),
        "mean_p_cov": statistics.fmean(r["p_cov"] for r in records),
        "mean_q_cov": statistics.fmean(r["q_cov"] for r in records),
        "mean_topk_overlap": statistics.fmean(r["topk_overlap"] for r in records),
        "nll_p": mean_nll_p,
        "nll_q": mean_nll_q,
        "delta_nll": mean_nll_q - mean_nll_p,
        "ppl_p": ppl_p,
        "ppl_q": ppl_q,
        "ppl_ratio": (ppl_q / ppl_p) if ppl_p > 0 else 0.0,
        "logprob_mae": statistics.fmean(abs_errs) if abs_errs else 0.0,
    }


def evaluate_candidate(
    base_url: str,
    model: str,
    out_dir: Path,
    *,
    concurrency: int = 4,
) -> dict[str, Any]:
    ref_path = out_dir / "reference.json"
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    top_k = int(ref["top_k"])
    traces = ref["traces"]
    ref_logprobs = ref["ref_logprobs"]
    longgen_ref = ref["longgen_ref"]
    longgen_max_tokens = int(ref["longgen_max_tokens"])

    t0 = time.monotonic()
    print(
        f"[eval-cand] Scoring candidate prompt_logprobs (top_k={top_k}) "
        f"across {len(traces)} traces...",
        flush=True,
    )
    cand_logprobs: dict[str, list[dict[str, float]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {
            pool.submit(
                _score_prompt_logprobs, base_url, model, t["full_token_ids"], top_k
            ): t["id"]
            for t in traces
        }
        for fut in concurrent.futures.as_completed(futs):
            tid = futs[fut]
            cand_logprobs[tid] = fut.result()

    all_trace_records: list[dict[str, float]] = []
    all_seq_records: list[dict[str, float]] = []
    per_trace_summary: list[dict[str, Any]] = []

    for t in traces:
        tid = t["id"]
        full_ids = t["full_token_ids"]
        prompt_len = len(t["prompt_token_ids"])
        p_list = ref_logprobs[tid]
        q_list = cand_logprobs[tid]

        trace_recs: list[dict[str, float]] = []
        seq_recs: list[dict[str, float]] = []
        for pos in range(1, len(full_ids)):
            p_dist = {int(k): float(v) for k, v in p_list[pos].items()}
            q_dist = {int(k): float(v) for k, v in q_list[pos].items()}
            if not p_dist or not q_dist:
                continue
            m = compute_token_divergence(p_dist, q_dist, actual_token_id=full_ids[pos])
            seq_recs.append(m)
            if pos >= prompt_len:
                trace_recs.append(m)

        all_trace_records.extend(trace_recs)
        all_seq_records.extend(seq_recs)
        agg = _aggregate_token_metrics(trace_recs)
        per_trace_summary.append(
            {
                "id": tid,
                "category": t["category"],
                "prompt_tokens": prompt_len,
                "trace_tokens": len(trace_recs),
                **agg,
            }
        )

    print(
        f"[eval-cand] Running {len(longgen_ref)} long-generation guardrail checks...",
        flush=True,
    )
    longgen_comparisons: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(concurrency, len(longgen_ref))
    ) as pool:
        futs = {
            pool.submit(
                _generate_completion_from_ids,
                base_url,
                model,
                item["prompt_token_ids"],
                longgen_max_tokens,
            ): item
            for item in longgen_ref
        }
        for fut in concurrent.futures.as_completed(futs):
            item = futs[fut]
            gen = fut.result()
            cand_ids = gen["token_ids"]
            ref_ids = item["completion_token_ids"]
            pref = common_prefix_len(ref_ids, cand_ids)
            exact_pos = sum(1 for a, b in zip(ref_ids, cand_ids) if a == b) / max(
                1, min(len(ref_ids), len(cand_ids))
            )
            longgen_comparisons.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "ref_tokens": len(ref_ids),
                    "cand_tokens": len(cand_ids),
                    "common_prefix_tokens": pref,
                    "pos_match_rate": exact_pos,
                    "ref_finish_reason": item["finish_reason"],
                    "cand_finish_reason": gen["finish_reason"],
                    "ref_rep4": item["rep4"],
                    "cand_rep4": ngram_repetition_ratio(cand_ids, 4),
                    "ref_rep8": item["rep8"],
                    "cand_rep8": ngram_repetition_ratio(cand_ids, 8),
                    "ref_text_preview": item["completion_text"][:500],
                    "cand_text_preview": gen["text"][:500],
                }
            )
    longgen_comparisons.sort(
        key=lambda x: [p["id"] for p in longgen_ref].index(x["id"])
    )

    overall_trace = _aggregate_token_metrics(all_trace_records)
    overall_all = _aggregate_token_metrics(all_seq_records)

    report = {
        "ref_model": ref["ref_model"],
        "cand_model": model,
        "top_k": top_k,
        "overall_trace_tokens": overall_trace,
        "overall_all_tokens": overall_all,
        "per_trace": per_trace_summary,
        "longgen_guardrail": longgen_comparisons,
        "elapsed_sec": round(time.monotonic() - t0, 2),
    }
    report_path = out_dir / "qda_kl_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\n================ QDA-style KL Divergence Report ================")
    print(f"Reference (P): {ref['ref_model']}")
    print(f"Candidate (Q): {model}")
    print(
        f"Trace tokens scored: {overall_trace['count']} (top_k={top_k}, "
        f"mean P mass coverage={overall_trace['mean_p_cov'] * 100:.3f}%)"
    )
    print(
        f"  Forward KL D_KL(P||Q): mean={overall_trace['mean_fwd_kl']:.6f} nats, "
        f"median={overall_trace['median_fwd_kl']:.6f}, "
        f"p95={overall_trace['p95_fwd_kl']:.6f}, "
        f"p99={overall_trace['p99_fwd_kl']:.6f}, "
        f"max={overall_trace['max_fwd_kl']:.6f}"
    )
    print(
        f"  Reverse KL D_KL(Q||P): mean={overall_trace['mean_rev_kl']:.6f} nats | "
        f"JSD: mean={overall_trace['mean_jsd']:.6f} nats"
    )
    print(
        f"  Agreement: Top-1={overall_trace['top1_agree'] * 100:.2f}%, "
        f"Top-5={overall_trace['top5_agree'] * 100:.2f}%, "
        f"Top-{top_k} overlap={overall_trace['mean_topk_overlap'] * 100:.2f}%"
    )
    print(
        f"  Exact Trace NLL/PPL: NLL_P={overall_trace['nll_p']:.5f} -> "
        f"NLL_Q={overall_trace['nll_q']:.5f} (delta={overall_trace['delta_nll']:+.6f}), "
        f"PPL_P={overall_trace['ppl_p']:.4f} -> PPL_Q={overall_trace['ppl_q']:.4f} "
        f"(ratio={overall_trace['ppl_ratio']:.5f}), "
        f"Logprob MAE={overall_trace['logprob_mae']:.5f}"
    )
    print("  Per-trace breakdown:")
    for row in per_trace_summary:
        print(
            f"    {row['id']:24s} tokens={row['trace_tokens']:3d} "
            f"KL={row['mean_fwd_kl']:.6f} p95={row['p95_fwd_kl']:.6f} "
            f"top1={row['top1_agree'] * 100:5.1f}% dNLL={row['delta_nll']:+.6f}"
        )
    print("  Long-generation guardrail (256 tokens greedy):")
    for lg in longgen_comparisons:
        print(
            f"    {lg['id']:20s} prefix_match={lg['common_prefix_tokens']:3d}/{lg['ref_tokens']:3d} "
            f"rep4={lg['ref_rep4'] * 100:.1f}%->{lg['cand_rep4'] * 100:.1f}% "
            f"rep8={lg['ref_rep8'] * 100:.1f}%->{lg['cand_rep8'] * 100:.1f}% "
            f"finish={lg['ref_finish_reason']}->{lg['cand_finish_reason']}"
        )
    print(f"Saved report -> {report_path}", flush=True)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QDA-style lightweight end-to-end KL divergence evaluator for vLLM."
    )
    parser.add_argument("--mode", required=True, choices=["collect-ref", "eval-cand"])
    parser.add_argument("--base-url", default="http://127.0.0.1:14800")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--trace-max-tokens", type=int, default=128)
    parser.add_argument("--longgen-max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "collect-ref":
        collect_reference(
            args.base_url,
            args.model,
            args.out_dir,
            top_k=args.top_k,
            trace_max_tokens=args.trace_max_tokens,
            longgen_max_tokens=args.longgen_max_tokens,
            concurrency=args.concurrency,
        )
    else:
        evaluate_candidate(
            args.base_url,
            args.model,
            args.out_dir,
            concurrency=args.concurrency,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
