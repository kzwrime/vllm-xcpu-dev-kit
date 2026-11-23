#!/usr/bin/env python3
"""128-Rank (multi-TP4) Qwen3.8-27B + DFlash2 P/D deployment & live calibration tool.

Supports two workflows decoupled from any cluster launcher:
1. Live HTTP Probing (--url, or --prefill-url / --decode-url) against ALREADY-RUNNING
   OpenAI-compatible vLLM / vllm-router endpoints to measure real cold Prefill TTFT,
   long-context decay, and Decode TPOT(b) across concurrency levels.
2. Offline Parameterized Calculation (when no URL is passed) from scalar measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

# Qwen3.8-27B + DFlash2 hardware-invariant KV/GDN geometry constants (TP=4).
FULL_ATTN_LAYERS = 16
GDN_LAYERS = 48
DFLASH_LAYERS = 5
LAYERS_PER_GROUP = 5
FULL_ATTN_GROUPS = math.ceil(FULL_ATTN_LAYERS / LAYERS_PER_GROUP)  # 4
GDN_GROUPS = math.ceil(GDN_LAYERS / LAYERS_PER_GROUP)  # 10
DFLASH_GROUPS = math.ceil(DFLASH_LAYERS / LAYERS_PER_GROUP)  # 1

# Per-rank per-token / per-page bytes under TP=4:
FA_BYTES_PER_TOK_LAYER_TP4 = 1024
GDN_PAGE_BYTES_LAYER_TP4 = 808 * 1024
DFLASH_WINDOW = 2048
GROUP_BYTES_PER_TOK_RANK_TP4 = LAYERS_PER_GROUP * FA_BYTES_PER_TOK_LAYER_TP4  # 5120 B


@dataclass
class PrefillProbePoint:
    target_len: int
    avg_prompt_tokens: float
    mean_ttft_s: float
    tps: float


@dataclass
class DecodeProbePoint:
    concurrency: int
    completed: int
    mean_tpot_ms: float
    p95_tpot_ms: float
    instance_tps: float


@dataclass
class ProbeSummary:
    prefill_url: str | None = None
    decode_url: str | None = None
    model_id: str | None = None
    server_max_model_len: int | None = None
    prefill_points: list[PrefillProbePoint] = field(default_factory=list)
    decode_points: list[DecodeProbePoint] = field(default_factory=list)


@dataclass
class DeployPlan:
    # Calibration Mode & Measured Summary
    probe_mode: str
    probe_summary: ProbeSummary | None

    # Cluster & KV/GDN Geometry
    total_ranks: int
    tp_size: int
    total_instances: int
    kv_gb_per_rank: float
    block_size_arg: int
    aligned_block_size: int
    group_block_mib: float
    gdn_padding_pct: float
    pool_blocks_per_rank: int
    usable_blocks_per_rank: int
    prefill_max_num_batched_tokens: int
    prefill_effective_chunk: int
    prefill_chunk_utilization_pct: float
    decode_max_num_batched_tokens: int

    # Capacity, Output Window & Concurrency Limits
    recommended_max_model_len: int
    recommended_max_model_len_blocks: int
    max_output_tokens_reserved: int
    max_prompt_tokens_with_output_reserved: int
    recommended_auto_compact_tokens: int
    max_output_kv_blocks: int
    max_output_kv_gib: float
    req_full_len_blocks: int
    req_full_len_pool_pct: float
    max_concurrent_full_len_reqs: int
    req_avg_len_blocks: int
    req_avg_len_gib: float
    avg_len_pool_max_reqs: int
    kv_tier_decode_max_num_seqs: int
    prefill_max_num_seqs: int
    decode_max_num_seqs: int

    # Parameterized / Probed Performance
    spec_k: int
    mean_accept_len: float
    prefill_tps_short: float
    prefill_tps_avg_ctx: float
    prefill_decay_factor: float
    decode_single_tps: float
    decode_step_ms_base: float
    optimal_decode_batch: int
    decode_tpot_at_opt_ms: float
    decode_tps_per_instance: float
    miss_tokens_per_turn: float
    est_ttft_ms: float

    # Cluster P/D Split
    num_prefill_instances: int
    num_decode_instances: int
    num_prefill_ranks: int
    num_decode_ranks: int
    cluster_prefill_miss_tps: float
    cluster_decode_tps: float

    # Mooncake Store & Router Policy
    kv_valid_bytes_per_tok_all_ranks: int
    kv_padded_bytes_per_tok_all_ranks: int
    mooncake_store_speedup: float
    working_set_gib: float
    mooncake_store_coverage_ratio: float
    router_prefill_policy: str
    router_decode_policy: str
    router_balance_abs_threshold: int
    router_balance_rel_threshold: float
    router_cache_threshold: float
    router_kv_aware_affinity_ttl_secs: int


def _http_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def fetch_model_info(base_url: str, api_key: str | None, timeout: float) -> tuple[str, int | None]:
    url = base_url.rstrip("/") + "/v1/models"
    req = urllib.request.Request(url, headers=_http_headers(api_key), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"GET {url} returned no models: {data}")
    first = models[0]
    return first["id"], first.get("max_model_len")


def _build_nonce_prompt(approx_tokens: int) -> str:
    # Each " x" / repeating code statement is roughly ~1 token; exact prompt_tokens is read from usage.
    nonce = f"# session-{uuid.uuid4().hex}\n"
    unit = "int fn(int a, int b) { return a + b; }\n"  # ~12-14 tokens
    repeats = max(1, (approx_tokens - 16) // 13)
    return nonce + (unit * repeats) + "Summarize the last integer offset strictly in numerals."


def _stream_chat_request(
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    api_key: str | None,
    timeout: float,
    ignore_eos: bool = False,
) -> tuple[float, float, int, int]:
    """Send streaming OpenAI chat request; returns (ttft_s, gen_s, prompt_tokens, comp_tokens)."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=_http_headers(api_key), method="POST")

    t_start = time.perf_counter()
    t_first: float | None = None
    prompt_tokens = 0
    comp_tokens = 0
    content_chunks = 0

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            payload_str = line[5:].strip()
            if payload_str == "[DONE]":
                break
            chunk = json.loads(payload_str)
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                comp_tokens = int(usage.get("completion_tokens") or comp_tokens)
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    content_chunks += 1

    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    ttft_s = max(1e-4, t_first - t_start)
    gen_s = max(1e-4, t_end - t_first)
    if comp_tokens <= 0:
        comp_tokens = max(1, content_chunks)
    if prompt_tokens <= 0:
        prompt_tokens = max(1, len(prompt) // 4)
    return ttft_s, gen_s, prompt_tokens, comp_tokens


def run_live_probes(args: argparse.Namespace) -> ProbeSummary:
    prefill_url = args.prefill_url or args.url
    decode_url = args.decode_url or args.url
    summary = ProbeSummary(prefill_url=prefill_url, decode_url=decode_url)

    ref_url = prefill_url or decode_url
    assert ref_url is not None
    discovered_model, srv_max_len = fetch_model_info(ref_url, args.api_key, args.probe_timeout)
    model_id = args.model or discovered_model
    summary.model_id = model_id
    summary.server_max_model_len = srv_max_len

    # Warmup 1 minimal request so JIT/connection pool overhead doesn't skew TTFT
    _stream_chat_request(ref_url, model_id, _build_nonce_prompt(64), 4, args.api_key, args.probe_timeout)

    if prefill_url:
        lens = [int(x.strip()) for x in args.probe_prefill_lens.split(",") if x.strip()]
        for target_len in lens:
            if srv_max_len and target_len + 16 > srv_max_len:
                continue
            ttfts: list[float] = []
            ptoks: list[int] = []
            for _ in range(args.probe_prefill_runs):
                ttft_s, _, ptok, _ = _stream_chat_request(
                    prefill_url,
                    model_id,
                    _build_nonce_prompt(target_len),
                    1,
                    args.api_key,
                    args.probe_timeout,
                )
                ttfts.append(ttft_s)
                ptoks.append(ptok)
            mean_ttft = statistics.mean(ttfts)
            mean_ptok = statistics.mean(ptoks)
            summary.prefill_points.append(
                PrefillProbePoint(
                    target_len=target_len,
                    avg_prompt_tokens=round(mean_ptok, 1),
                    mean_ttft_s=round(mean_ttft, 4),
                    tps=round(mean_ptok / mean_ttft, 1),
                )
            )

    if decode_url:
        concs = [int(x.strip()) for x in args.probe_concurrencies.split(",") if x.strip()]
        d_prompt_len = args.probe_decode_prompt_len
        d_max_tok = args.probe_decode_max_tokens
        for c in concs:
            prompts = [_build_nonce_prompt(d_prompt_len) for _ in range(c)]
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=c) as pool:
                futs = [
                    pool.submit(
                        _stream_chat_request,
                        decode_url,
                        model_id,
                        pmt,
                        d_max_tok,
                        args.api_key,
                        args.probe_timeout,
                        True,
                    )
                    for pmt in prompts
                ]
                results = [f.result() for f in futs]
            wall_total_s = time.perf_counter() - t0
            tpots = []
            total_gen_tokens = 0
            mean_ttft = statistics.mean(r[0] for r in results)
            for _, gen_s, _, comp_tok in results:
                dec_steps_toks = max(1, comp_tok - 1)
                tpots.append((gen_s * 1000.0) / dec_steps_toks)
                total_gen_tokens += comp_tok
            wall_decode_s = max(1e-3, wall_total_s - mean_ttft)
            tpots_sorted = sorted(tpots)
            p95_idx = min(len(tpots_sorted) - 1, math.ceil(0.95 * len(tpots_sorted)) - 1)
            summary.decode_points.append(
                DecodeProbePoint(
                    concurrency=c,
                    completed=len(results),
                    mean_tpot_ms=round(statistics.mean(tpots), 2),
                    p95_tpot_ms=round(tpots_sorted[p95_idx], 2),
                    instance_tps=round(total_gen_tokens / wall_decode_s, 1),
                )
            )

    return summary


def align_block_size(block_size: int) -> int:
    blocks_min = math.ceil(GDN_PAGE_BYTES_LAYER_TP4 / (block_size * FA_BYTES_PER_TOK_LAYER_TP4))
    return blocks_min * block_size


def full_req_blocks(seq_len: int, block_b: int, spec_k: int) -> int:
    fa_blocks_per_group = math.ceil((seq_len + spec_k + 1) / block_b)
    return FULL_ATTN_GROUPS * fa_blocks_per_group + GDN_GROUPS * 3 + 8


def active_req_blocks_runtime(seq_len: int, block_b: int, spec_k: int) -> int:
    fa_blocks_per_group = math.ceil((seq_len + spec_k + 1) / block_b)
    # DFlashSlidingWindowManager steady physical tail = cdiv(sliding_window + spec_k + 1, B) + 1 (=4 for sw=2048)
    draft_tail = math.ceil((DFLASH_WINDOW + spec_k + 1) / block_b) + 1
    return FULL_ATTN_GROUPS * fa_blocks_per_group + GDN_GROUPS * 3 + draft_tail


def choose_max_model_len(usable_blocks: int, block_b: int, spec_k: int) -> int:
    max_per_req_4 = usable_blocks // 4
    for candidate in (262144, 229376, 215040, 204800, 196608, 184320):
        if full_req_blocks(candidate, block_b, spec_k) <= max_per_req_4:
            return candidate
    max_fa_n = (max_per_req_4 - (GDN_GROUPS * 3 + 8)) // FULL_ATTN_GROUPS
    step = math.lcm(block_b, 2048)
    raw_max_len = (max_fa_n * block_b - (spec_k + 1)) // step * step
    return max(184320, min(262144, raw_max_len))


def compute_plan(args: argparse.Namespace, probe: ProbeSummary | None = None) -> DeployPlan:
    if args.total_ranks % args.tp != 0:
        raise ValueError(f"total-ranks ({args.total_ranks}) must be divisible by tp ({args.tp})")
    if args.tp != 4:
        raise ValueError(f"This model geometry calculator targets TP=4 (got tp={args.tp})")

    total_instances = args.total_ranks // args.tp
    aligned_b = align_block_size(args.block_size)
    group_block_bytes = aligned_b * GROUP_BYTES_PER_TOK_RANK_TP4
    group_block_mib = group_block_bytes / (1024 * 1024)
    gdn_pad_pct = (aligned_b * FA_BYTES_PER_TOK_LAYER_TP4 - GDN_PAGE_BYTES_LAYER_TP4) / GDN_PAGE_BYTES_LAYER_TP4 * 100.0

    pool_blocks = int((args.kv_gb * 1024**3) // group_block_bytes)
    usable_blocks = pool_blocks - 1

    eff_chunk = (args.prefill_chunk // aligned_b) * aligned_b
    chunk_util_pct = eff_chunk / args.prefill_chunk * 100.0
    decode_batched = max(aligned_b, 1024)

    max_model_len = choose_max_model_len(usable_blocks, aligned_b, args.spec_k)
    max_len_blks = math.ceil((max_model_len + args.spec_k + 1) / aligned_b)
    max_prompt_reserved = max(0, max_model_len - args.max_output_tokens)
    auto_compact = max(0, max_prompt_reserved - 8192)
    max_out_kv_blocks = FULL_ATTN_GROUPS * math.ceil(args.max_output_tokens / aligned_b)
    max_out_kv_gib = (max_out_kv_blocks * group_block_bytes) / (1024**3)

    req_full_blks = full_req_blocks(max_model_len, aligned_b, args.spec_k)
    req_full_pct = req_full_blks / usable_blocks * 100.0
    max_full_conc = usable_blocks // req_full_blks

    req_avg_blks = active_req_blocks_runtime(args.avg_prompt, aligned_b, args.spec_k)
    req_avg_gib = (req_avg_blks * group_block_bytes) / (1024**3)
    avg_len_max_reqs = max(1, int(usable_blocks * 0.85) // req_avg_blks)
    kv_tier_decode_seqs = 16 if args.kv_gb >= 18 else (14 if args.kv_gb >= 17 else 12)

    # Resolve Prefill rates (prefer live probe when available)
    decay_factor = args.prefill_decay
    if probe and probe.prefill_points:
        p_short = probe.prefill_points[0]
        prefill_tps_short = p_short.tps
        if len(probe.prefill_points) >= 2:
            p_long = probe.prefill_points[-1]
            decay_factor = max(0.3, min(1.1, p_long.tps / max(1e-6, p_short.tps)))
        prefill_tps_avg = prefill_tps_short * decay_factor
    else:
        prefill_tps_short = 8192.0 / args.ttft_8k
        prefill_tps_avg = prefill_tps_short * decay_factor

    # Resolve Decode TPOT / concurrency (prefer live probe points when available)
    if probe and probe.decode_points:
        c1 = next((pt for pt in probe.decode_points if pt.concurrency == 1), probe.decode_points[0])
        decode_single_tps = 1000.0 / c1.mean_tpot_ms if c1.concurrency == 1 else args.decode_single_tps
        step_ms_base = (1000.0 / decode_single_tps) * args.accept_len
        passing = [pt for pt in probe.decode_points if pt.p95_tpot_ms <= args.tpot_slo_ms]
        chosen = max(passing, key=lambda pt: pt.instance_tps) if passing else probe.decode_points[0]
        best_b = min(chosen.concurrency, avg_len_max_reqs)
        opt_tpot = chosen.mean_tpot_ms
        best_rd = chosen.instance_tps
        probe_mode = "live_http"
    else:
        probe_mode = "offline_formula"
        decode_single_tps = args.decode_single_tps
        single_tpot_ms = 1000.0 / decode_single_tps
        step_ms_base = single_tpot_ms * args.accept_len
        avg_k = args.avg_prompt / 1024.0

        def step_ms(b: int) -> float:
            return step_ms_base * (1.0 + args.step_batch_slope * (b - 1)) + args.step_kv_ms_per_ktok_req * b * avg_k

        def tpot_ms(b: int) -> float:
            return step_ms(b) / args.accept_len

        best_b = 1
        best_rd = 0.0
        for b in range(1, min(avg_len_max_reqs, 32) + 1):
            cur_tpot = tpot_ms(b)
            cur_rd = 1000.0 * b / cur_tpot
            if cur_tpot <= args.tpot_slo_ms and cur_rd > best_rd:
                best_b = b
                best_rd = cur_rd
        if best_rd == 0.0:
            best_b = 1
            best_rd = 1000.0 / tpot_ms(1)
        opt_tpot = tpot_ms(best_b)

    decode_max_seqs = min(avg_len_max_reqs, max(best_b + 4, math.ceil(best_b * 1.4)))
    decode_max_seqs = min(decode_max_seqs, decode_batched // (args.spec_k + 1))
    prefill_max_seqs = 8 if args.kv_gb >= 16 else 6

    miss_tokens = (1.0 - args.hit_rate) * args.avg_prompt
    store_hit_ratio = max(0.0, args.hit_rate - 0.50)
    kv_valid_bytes_all = FULL_ATTN_LAYERS * FA_BYTES_PER_TOK_LAYER_TP4 * args.tp  # 65,536 B (64 KiB)
    kv_padded_bytes_all = FULL_ATTN_GROUPS * GROUP_BYTES_PER_TOK_RANK_TP4 * args.tp  # 81,920 B (80 KiB)
    store_pull_ms = (store_hit_ratio * args.avg_prompt * kv_padded_bytes_all) / (args.store_bw_gibs * 1024**3) * 1000.0
    compute_ms = (miss_tokens / prefill_tps_avg) * 1000.0
    gdn_state_xfer_bytes = GDN_GROUPS * 2 * group_block_bytes * args.tp
    pd_xfer_ms = (miss_tokens * kv_padded_bytes_all + gdn_state_xfer_bytes) / (args.store_bw_gibs * 1024**3) * 1000.0
    est_ttft_ms = 5.0 + store_pull_ms + compute_ms + pd_xfer_ms + opt_tpot * args.accept_len

    ratio_pd = (best_rd / prefill_tps_avg) * (miss_tokens / args.avg_output) * args.burst_factor
    raw_np = total_instances * ratio_pd / (1.0 + ratio_pd)
    num_p = int(round(raw_np / 2.0) * 2)
    num_p = max(2, min(total_instances - 2, num_p))
    num_d = total_instances - num_p

    compute_kv_gen_gibs = (prefill_tps_avg * kv_padded_bytes_all) / (1024**3)
    store_speedup = args.store_bw_gibs / compute_kv_gen_gibs
    active_sessions = num_d * best_b * args.sessions_per_decode_slot
    working_set_gib = (active_sessions * req_avg_blks * group_block_bytes * args.tp) / (1024**3)
    coverage_ratio = (args.store_cap_tib * 1024.0) / max(1e-6, working_set_gib)

    if store_speedup >= 15.0 and coverage_ratio >= 1.5:
        abs_thresh, rel_thresh, cache_thresh, kv_ttl = 2, 1.25, 0.50, 1800
    elif store_speedup >= 5.0 and coverage_ratio >= 0.8:
        abs_thresh, rel_thresh, cache_thresh, kv_ttl = 4, 1.40, 0.35, 1800
    else:
        abs_thresh, rel_thresh, cache_thresh, kv_ttl = 6, 1.50, 0.25, 3600

    return DeployPlan(
        probe_mode=probe_mode,
        probe_summary=probe,
        total_ranks=args.total_ranks,
        tp_size=args.tp,
        total_instances=total_instances,
        kv_gb_per_rank=args.kv_gb,
        block_size_arg=args.block_size,
        aligned_block_size=aligned_b,
        group_block_mib=round(group_block_mib, 4),
        gdn_padding_pct=round(gdn_pad_pct, 2),
        pool_blocks_per_rank=pool_blocks,
        usable_blocks_per_rank=usable_blocks,
        prefill_max_num_batched_tokens=args.prefill_chunk,
        prefill_effective_chunk=eff_chunk,
        prefill_chunk_utilization_pct=round(chunk_util_pct, 1),
        decode_max_num_batched_tokens=decode_batched,
        recommended_max_model_len=max_model_len,
        recommended_max_model_len_blocks=max_len_blks,
        max_output_tokens_reserved=args.max_output_tokens,
        max_prompt_tokens_with_output_reserved=max_prompt_reserved,
        recommended_auto_compact_tokens=auto_compact,
        max_output_kv_blocks=max_out_kv_blocks,
        max_output_kv_gib=round(max_out_kv_gib, 3),
        req_full_len_blocks=req_full_blks,
        req_full_len_pool_pct=round(req_full_pct, 1),
        max_concurrent_full_len_reqs=max_full_conc,
        req_avg_len_blocks=req_avg_blks,
        req_avg_len_gib=round(req_avg_gib, 3),
        avg_len_pool_max_reqs=avg_len_max_reqs,
        kv_tier_decode_max_num_seqs=kv_tier_decode_seqs,
        prefill_max_num_seqs=prefill_max_seqs,
        decode_max_num_seqs=decode_max_seqs,
        spec_k=args.spec_k,
        mean_accept_len=args.accept_len,
        prefill_tps_short=round(prefill_tps_short, 1),
        prefill_tps_avg_ctx=round(prefill_tps_avg, 1),
        prefill_decay_factor=round(decay_factor, 3),
        decode_single_tps=round(decode_single_tps, 1),
        decode_step_ms_base=round(step_ms_base, 2),
        optimal_decode_batch=best_b,
        decode_tpot_at_opt_ms=round(opt_tpot, 2),
        decode_tps_per_instance=round(best_rd, 1),
        miss_tokens_per_turn=round(miss_tokens, 0),
        est_ttft_ms=round(est_ttft_ms, 1),
        num_prefill_instances=num_p,
        num_decode_instances=num_d,
        num_prefill_ranks=num_p * args.tp,
        num_decode_ranks=num_d * args.tp,
        cluster_prefill_miss_tps=round(num_p * prefill_tps_avg, 1),
        cluster_decode_tps=round(num_d * best_rd, 1),
        kv_valid_bytes_per_tok_all_ranks=kv_valid_bytes_all,
        kv_padded_bytes_per_tok_all_ranks=kv_padded_bytes_all,
        mooncake_store_speedup=round(store_speedup, 1),
        working_set_gib=round(working_set_gib, 1),
        mooncake_store_coverage_ratio=round(coverage_ratio, 2),
        router_prefill_policy="cache_aware",
        router_decode_policy="cache_aware",
        router_balance_abs_threshold=abs_thresh,
        router_balance_rel_threshold=rel_thresh,
        router_cache_threshold=cache_thresh,
        router_kv_aware_affinity_ttl_secs=kv_ttl,
    )


def format_text(p: DeployPlan) -> str:
    lines = [
        f"=== Qwen3.8-27B + DFlash2 (TP4) 部署校准结果 [{p.probe_mode}] ===",
    ]
    if p.probe_summary:
        ps = p.probe_summary
        lines.extend([
            "",
            f"[0] 在线接口实测摘要 (model={ps.model_id}, server_max_len={ps.server_max_model_len})",
        ])
        for pt in ps.prefill_points:
            lines.append(
                f"  Prefill target={pt.target_len:6d} (actual={pt.avg_prompt_tokens:7.1f} tok) -> "
                f"TTFT={pt.mean_ttft_s:.4f} s ({pt.tps:.1f} tok/s)"
            )
        for pt in ps.decode_points:
            lines.append(
                f"  Decode  conc={pt.concurrency:2d} (done={pt.completed}) -> "
                f"TPOT mean={pt.mean_tpot_ms:.2f} ms, p95={pt.p95_tpot_ms:.2f} ms, inst_tps={pt.instance_tps:.1f} tok/s"
            )

    lines.extend([
        "",
        "[1] KV/GDN 块几何与池容量",
        f"  --block-size={p.block_size_arg} -> 对齐逻辑块 B={p.aligned_block_size} tokens",
        f"  单组块大小={p.group_block_mib} MiB/rank | GDN 填充率=+{p.gdn_padding_pct}%",
        f"  KV 池 ({p.kv_gb_per_rank:.0f} GiB/rank): 总块数={p.pool_blocks_per_rank}, 可用={p.usable_blocks_per_rank}",
        f"  Prefill 有效单步块={p.prefill_effective_chunk}/{p.prefill_max_num_batched_tokens} ({p.prefill_chunk_utilization_pct}%)",
        "",
        "[2] 最优上下文、最大输出预留与并发上限",
        f"  推荐 --max-model-len     = {p.recommended_max_model_len} ({p.recommended_max_model_len // 1024}K, {p.recommended_max_model_len_blocks} 块)",
        f"  预留最大输出 M            = {p.max_output_tokens_reserved} tok (需额外 {p.max_output_kv_blocks} 块 / {p.max_output_kv_gib} GiB/rank)",
        f"  单轮有效输入上限          = {p.max_prompt_tokens_with_output_reserved} tok | 建议客户端 auto-compact <= {p.recommended_auto_compact_tokens} tok",
        f"  满长单请求占用            = {p.req_full_len_blocks} 块 ({p.req_full_len_pool_pct}% 池, 满长并发上限 {p.max_concurrent_full_len_reqs} 条)",
        f"  均长单请求稳态占用        = {p.req_avg_len_blocks} 块 ({p.req_avg_len_gib} GiB/rank, 85%池容积 {p.avg_len_pool_max_reqs} 条)",
        f"  Prefill --max-num-seqs    = {p.prefill_max_num_seqs}",
        f"  Decode  --max-num-seqs    = {p.kv_tier_decode_max_num_seqs} (KV 档位通用上限) / {p.decode_max_num_seqs} (SLO 紧保护上限, b*={p.optimal_decode_batch})",
        f"  Decode batched_tokens     = {p.decode_max_num_batched_tokens}",
        "",
        "[3] 参数化 / 实测 TTFT 与 TPOT 及 128-Rank P/D 配比",
        f"  Prefill 速率              = {p.prefill_tps_short} tok/s (基准) -> {p.prefill_tps_avg_ctx} tok/s (衰减系数 {p.prefill_decay_factor})",
        f"  单请求 Decode             = {p.decode_single_tps} tok/s (DFlash2 K={p.spec_k}, tau={p.mean_accept_len}, T_step={p.decode_step_ms_base} ms)",
        f"  Decode SLO 点 (b*={p.optimal_decode_batch:2d})     = TPOT {p.decode_tpot_at_opt_ms} ms/tok | 单实例输出 R_D={p.decode_tps_per_instance} tok/s",
        f"  单轮增量估算              = {p.miss_tokens_per_turn:.0f} miss tokens | 估算 TTFT={p.est_ttft_ms} ms",
        f"  推荐集群切分              = {p.num_prefill_instances}P + {p.num_decode_instances}D ({p.num_prefill_ranks}P + {p.num_decode_ranks}D Ranks)",
        f"  集群总吞吐估算            = Prefill 增量 {p.cluster_prefill_miss_tps} tok/s | Decode 输出 {p.cluster_decode_tps} tok/s",
        "",
        "[4] Mooncake Store 与 vllm-router 策略",
        f"  KV 传输密度 (TP=4)        = {p.kv_valid_bytes_per_tok_all_ranks // 1024} KiB/tok 有效 ({p.kv_padded_bytes_per_tok_all_ranks // 1024} KiB/tok 对齐逻辑组)",
        f"  RDMA 拉取 vs 重算加速比   = {p.mooncake_store_speedup}x | 热工作集={p.working_set_gib} GiB (Store 覆盖率 kappa={p.mooncake_store_coverage_ratio}x)",
        "",
        "[5] 通用引擎与 Router 推荐参数（独立于具体启动框架）",
        f"  # Prefill vLLM args: --max-model-len {p.recommended_max_model_len} --max-num-batched-tokens {p.prefill_max_num_batched_tokens} --max-num-seqs {p.prefill_max_num_seqs} --block-size {p.block_size_arg}",
        f"  # Decode  vLLM args: --max-model-len {p.recommended_max_model_len} --max-num-batched-tokens {p.decode_max_num_batched_tokens} --max-num-seqs {p.kv_tier_decode_max_num_seqs} --block-size {p.block_size_arg}",
        (
            f"  # vllm-router args:  --prefill-policy {p.router_prefill_policy} --decode-policy {p.router_decode_policy}"
            f" --cache-threshold {p.router_cache_threshold} --balance-abs-threshold {p.router_balance_abs_threshold}"
            f" --balance-rel-threshold {p.router_balance_rel_threshold}"
        ),
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe running vLLM/router endpoints and calibrate 128-Rank Qwen3.8-27B + DFlash2 TP4 P/D parameters."
    )
    # Live probing options (for already-running endpoints)
    parser.add_argument("--url", type=str, default=None, help="Running OpenAI API base URL (used for both prefill/decode probes unless overridden)")
    parser.add_argument("--prefill-url", type=str, default=None, help="Running Prefill (or standalone) base URL")
    parser.add_argument("--decode-url", type=str, default=None, help="Running Decode (or router) base URL")
    parser.add_argument("--model", type=str, default=None, help="Model ID override (auto-detected from /v1/models by default)")
    parser.add_argument("--api-key", type=str, default=None, help="Optional Bearer token")
    parser.add_argument("--probe-prefill-lens", type=str, default="8192,32768", help="Comma-separated prompt token targets for cold TTFT probing")
    parser.add_argument("--probe-prefill-runs", type=int, default=2, help="Iterations per prefill probe length")
    parser.add_argument("--probe-concurrencies", type=str, default="1,4,8,12", help="Comma-separated concurrencies for Decode TPOT probing")
    parser.add_argument("--probe-decode-prompt-len", type=int, default=4096, help="Prompt tokens used during Decode concurrency probe")
    parser.add_argument("--probe-decode-max-tokens", type=int, default=96, help="Output tokens generated per Decode probe request")
    parser.add_argument("--probe-timeout", type=float, default=600.0, help="HTTP timeout per probe request in seconds")

    # Cluster, KV geometry, and offline fallback metrics
    parser.add_argument("--total-ranks", type=int, default=128)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--kv-gb", type=float, default=18.0, help="KV cache pool per rank in GiB (e.g. 16, 17, 18)")
    parser.add_argument("--block-size", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--prefill-chunk", type=int, default=6144)
    parser.add_argument("--max-output-tokens", type=int, default=32768, help="Request max_tokens / max_completion_tokens ceiling to reserve")
    parser.add_argument("--ttft-8k", type=float, default=2.4, help="Offline 8K (8192) cold TTFT in seconds (when --prefill-url not given)")
    parser.add_argument("--prefill-decay", type=float, default=0.85, help="Offline long-context Prefill decay factor eta_p")
    parser.add_argument("--decode-single-tps", type=float, default=65.0, help="Offline single-req decode tok/s")
    parser.add_argument("--spec-k", type=int, default=5, help="DFlash2 num_speculative_tokens")
    parser.add_argument("--accept-len", type=float, default=3.9, help="Mean accepted tokens per DFlash2 step tau in [1, K+1]")
    parser.add_argument("--tpot-slo-ms", type=float, default=25.0, help="Target Decode TPOT SLO ceiling in ms")
    parser.add_argument("--step-batch-slope", type=float, default=0.055)
    parser.add_argument("--step-kv-ms-per-ktok-req", type=float, default=0.06)
    parser.add_argument("--avg-prompt", type=int, default=49152)
    parser.add_argument("--hit-rate", type=float, default=0.85)
    parser.add_argument("--avg-output", type=int, default=500)
    parser.add_argument("--burst-factor", type=float, default=1.15)
    parser.add_argument("--sessions-per-decode-slot", type=float, default=3.0)
    parser.add_argument("--store-cap-tib", type=float, default=2.0)
    parser.add_argument("--store-bw-gibs", type=float, default=10.0)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    probe = run_live_probes(args) if (args.url or args.prefill_url or args.decode_url) else None
    plan = compute_plan(args, probe=probe)
    if args.format == "json":
        print(json.dumps(asdict(plan), indent=2, ensure_ascii=False))
    else:
        print(format_text(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
