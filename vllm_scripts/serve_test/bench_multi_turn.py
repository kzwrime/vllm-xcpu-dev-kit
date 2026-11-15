#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Multi-turn serving benchmark with intermediate-window incremental measurement.

Simulates multi-turn conversations with real coherent corpora (or ShareGPT/chat
logs) so speculative decoding achieves realistic acceptance rates and multi-turn
history hits KV prefix cache. Supports isolating any turn range [START..END]
for incremental Prefill TTFT, Decode TPOT/ITL, server Prefix Cache hit ratio,
Speculative Decoding acceptance rate/length, optional profiling, and timeline HTML.
"""

from __future__ import annotations

import os

# Bench client only uses tokenizer/HTTP; avoid torch_mcpu autoload circular
# import inside transformers when invoked outside user_env_template.sh.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import argparse
import asyncio
import contextlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import random
import time
from typing import Any

import aiohttp
import numpy as np
from transformers import AutoTokenizer

from vllm.benchmarks.lib.endpoint_request_func import StreamedResponseHandler


DEFAULT_PERCENTILES = (50.0, 90.0, 99.0)


@dataclass
class ConversationPlan:
    conv_id: int
    system_prompt: str | None
    user_turns: list[str]
    turn_input_token_targets: list[int]
    cache_salt: str | None = None


@dataclass
class TurnResult:
    conv_id: int
    turn_idx: int
    phase: str
    in_measure_window: bool
    history_prompt_tokens: int
    incremental_input_tokens: int
    prompt_tokens: int
    output_tokens: int
    start_time: float
    ttft: float
    latency: float
    tpot: float | None
    itl: list[float] = field(default_factory=list)
    success: bool = True
    error: str = ""


@dataclass
class ServerMetricsSnapshot:
    timestamp: float
    prefix_cache_queries: float | None
    prefix_cache_hits: float | None
    spec_num_drafts: int | None
    spec_num_draft_tokens: int | None
    spec_num_accepted_tokens: int | None
    spec_accepted_per_pos: dict[int, int] = field(default_factory=dict)


class RequestRateLimiter:
    """Async pace limiter for finite request_rate submissions."""

    def __init__(self, request_rate: float) -> None:
        self.interval = (
            1.0 / request_rate
            if request_rate > 0 and request_rate != float("inf")
            else 0.0
        )
        self._lock = asyncio.Lock()
        self._next_ts = 0.0

    async def acquire(self) -> None:
        if self.interval <= 0.0:
            return
        async with self._lock:
            now = time.perf_counter()
            if self._next_ts <= now:
                self._next_ts = now + self.interval
                return
            wait_s = self._next_ts - now
            self._next_ts += self.interval
        if wait_s > 0:
            await asyncio.sleep(wait_s)


def parse_turn_window(spec: str | None, num_turns: int) -> tuple[int, int]:
    """Parse 1-indexed inclusive turn range 'START:END' or 'TURN'."""
    if not spec:
        return 1, num_turns
    raw = spec.strip()
    if ":" in raw:
        parts = raw.split(":", 1)
        start = int(parts[0].strip()) if parts[0].strip() else 1
        end = int(parts[1].strip()) if parts[1].strip() else num_turns
    elif "-" in raw:
        parts = raw.split("-", 1)
        start = int(parts[0].strip()) if parts[0].strip() else 1
        end = int(parts[1].strip()) if parts[1].strip() else num_turns
    else:
        start = int(raw)
        end = start
    if start < 1 or end < start or end > num_turns:
        raise ValueError(
            f"Invalid --measure-turns '{spec}' for num_turns={num_turns}. "
            f"Expected 1 <= start <= end <= {num_turns}."
        )
    return start, end


def load_tokenizer(model_or_path: str, trust_remote_code: bool) -> Any:
    try:
        return AutoTokenizer.from_pretrained(
            model_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            model_or_path,
            trust_remote_code=trust_remote_code,
        )


def _extract_texts_from_corpus_file(path: Path) -> list[str]:
    texts: list[str] = []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    if isinstance(row.get("turns"), list):
                        texts.extend(str(t) for t in row["turns"] if str(t).strip())
                    elif isinstance(row.get("messages"), list):
                        texts.extend(
                            str(m.get("content", ""))
                            for m in row["messages"]
                            if isinstance(m, dict) and str(m.get("content", "")).strip()
                        )
                    elif isinstance(row.get("prompt"), str) and row["prompt"].strip():
                        texts.append(row["prompt"])
        return texts

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            # SQuAD format
            for article in data["data"]:
                for para in article.get("paragraphs", []):
                    ctx = str(para.get("context", "")).strip()
                    qs = [
                        str(qa.get("question", "")).strip()
                        for qa in para.get("qas", [])
                        if str(qa.get("question", "")).strip()
                    ]
                    if ctx:
                        block = ctx + ("\nQuestion: " + qs[0] if qs else "")
                        texts.append(block)
            return texts
        if isinstance(data, list):
            for row in data:
                if isinstance(row, str) and row.strip():
                    texts.append(row.strip())
                elif isinstance(row, dict):
                    for key in ("text", "prompt", "content"):
                        if isinstance(row.get(key), str) and row[key].strip():
                            texts.append(row[key].strip())
            return texts

    raw = path.read_text(encoding="utf-8", errors="replace")
    for para in raw.split("\n\n"):
        para = para.strip()
        if len(para) >= 40:
            texts.append(para)
    return texts


def load_corpus_passages(corpus_paths: list[Path], seed: int) -> list[str]:
    passages: list[str] = []
    for path in corpus_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Corpus file not found: {path}")
        passages.extend(_extract_texts_from_corpus_file(path))
    if not passages:
        raise ValueError(f"No text passages loaded from {corpus_paths}")
    rng = random.Random(seed)
    rng.shuffle(passages)
    return passages


def build_token_budget_text(
    tokenizer: Any,
    passages: list[str],
    cursor: list[int],
    target_tokens: int,
    lead_in: str,
) -> tuple[str, int]:
    """Assemble coherent natural text matching target_tokens cleanly."""
    if target_tokens <= 0:
        return "", 0

    pieces: list[str] = [lead_in] if lead_in else []
    token_ids: list[int] = (
        tokenizer.encode(lead_in, add_special_tokens=False) if lead_in else []
    )
    n_passages = len(passages)

    while len(token_ids) < target_tokens:
        passage = passages[cursor[0] % n_passages]
        cursor[0] += 1
        chunk = f"{passage}\n\n"
        chunk_ids = tokenizer.encode(chunk, add_special_tokens=False)
        pieces.append(chunk)
        token_ids.extend(chunk_ids)

    sliced_ids = token_ids[:target_tokens]
    text = tokenizer.decode(sliced_ids, skip_special_tokens=True).strip()
    if not text:
        text = "Summarize the key details concisely."
    actual_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return text, actual_tokens


def build_corpus_conversations(
    tokenizer: Any,
    corpus_paths: list[Path],
    num_conversations: int,
    num_turns: int,
    common_prefix_tokens: int,
    initial_context_tokens: int,
    turn_input_tokens: int,
    seed: int,
    run_salt: str | None,
) -> list[ConversationPlan]:
    passages = load_corpus_passages(corpus_paths, seed=seed)
    cursor = [0]

    shared_system: str | None = None
    if common_prefix_tokens > 0:
        shared_system, _ = build_token_budget_text(
            tokenizer,
            passages,
            cursor,
            common_prefix_tokens,
            lead_in="Shared reference background for all turns:\n\n",
        )

    plans: list[ConversationPlan] = []
    for conv_idx in range(num_conversations):
        user_turns: list[str] = []
        turn_targets: list[int] = []
        for turn_idx in range(1, num_turns + 1):
            if turn_idx == 1:
                budget = initial_context_tokens
                lead = (
                    f"[Conversation {conv_idx + 1} Turn 1] Read the context below "
                    "and summarize the main facts and technical points clearly:\n\n"
                )
            else:
                budget = turn_input_tokens
                lead = (
                    f"[Turn {turn_idx}] Building on the previous discussion, analyze "
                    "this additional passage and compare key points:\n\n"
                )
            prompt_text, actual_len = build_token_budget_text(
                tokenizer, passages, cursor, budget, lead_in=lead
            )
            user_turns.append(prompt_text)
            turn_targets.append(actual_len)

        conv_salt = None
        if run_salt is not None:
            # Share salt across conversations if common_prefix_tokens > 0 so shared
            # system KV blocks hit across conversations while staying isolated from other runs.
            conv_salt = (
                run_salt
                if common_prefix_tokens > 0
                else f"{run_salt}-c{conv_idx}"
            )
        plans.append(
            ConversationPlan(
                conv_id=conv_idx,
                system_prompt=shared_system,
                user_turns=user_turns,
                turn_input_token_targets=turn_targets,
                cache_salt=conv_salt,
            )
        )
    return plans


def _extract_user_turns_from_record(record: Any) -> tuple[str | None, list[str]]:
    system_prompt: str | None = None
    user_turns: list[str] = []
    if not isinstance(record, dict):
        return system_prompt, user_turns

    if isinstance(record.get("conversations"), list):
        for msg in record["conversations"]:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("from", msg.get("role", ""))).lower()
            val = str(msg.get("value", msg.get("content", ""))).strip()
            if not val:
                continue
            if role == "system" and system_prompt is None:
                system_prompt = val
            elif role in ("human", "user"):
                user_turns.append(val)
    elif isinstance(record.get("messages"), list):
        for msg in record["messages"]:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower()
            val = str(msg.get("content", "")).strip()
            if not val:
                continue
            if role == "system" and system_prompt is None:
                system_prompt = val
            elif role == "user":
                user_turns.append(val)
    elif isinstance(record.get("turns"), list):
        user_turns = [str(t).strip() for t in record["turns"] if str(t).strip()]
    elif isinstance(record.get("rounds"), list):
        for r in record["rounds"]:
            if not isinstance(r, dict):
                continue
            qs = r.get("questions") or []
            if qs and isinstance(qs[0], dict):
                q = str(qs[0].get("question", "")).strip()
                if q:
                    user_turns.append(q)

    return system_prompt, user_turns


def build_sharegpt_conversations(
    tokenizer: Any,
    dataset_path: Path,
    num_conversations: int,
    num_turns: int,
    min_turns_required: int,
    seed: int,
    run_salt: str | None,
) -> list[ConversationPlan]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    records: list[Any]
    if dataset_path.suffix.lower() == ".jsonl":
        records = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        loaded = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"Expected JSON list in {dataset_path}")
        records = loaded

    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    plans: list[ConversationPlan] = []
    for idx in indices:
        system_prompt, user_turns = _extract_user_turns_from_record(records[idx])
        if len(user_turns) < min_turns_required:
            continue
        selected = user_turns[:num_turns]
        token_lens = [
            len(tokenizer.encode(text, add_special_tokens=False)) for text in selected
        ]
        conv_idx = len(plans)
        conv_salt = None if run_salt is None else f"{run_salt}-c{conv_idx}"
        plans.append(
            ConversationPlan(
                conv_id=conv_idx,
                system_prompt=system_prompt,
                user_turns=selected,
                turn_input_token_targets=token_lens,
                cache_salt=conv_salt,
            )
        )
        if len(plans) >= num_conversations:
            break

    if not plans:
        raise ValueError(
            f"No conversations in {dataset_path} had >= {min_turns_required} user turns."
        )
    return plans


async def fetch_model_id(base_url: str, session: aiohttp.ClientSession) -> str:
    async with session.get(f"{base_url}/v1/models") as resp:
        resp.raise_for_status()
        payload = await resp.json()
    models = payload.get("data") or []
    if not models:
        raise RuntimeError(f"No models returned from {base_url}/v1/models")
    return str(models[0]["id"])


async def post_profile_cmd(
    base_url: str, session: aiohttp.ClientSession, action: str
) -> None:
    url = f"{base_url}/{action}"
    try:
        async with session.post(url) as resp:
            if resp.status == 200:
                print(f"[profiler] {action} -> OK")
            else:
                body = await resp.text()
                print(f"[profiler] {action} -> HTTP {resp.status}: {body.strip()}")
    except Exception as exc:
        print(f"[profiler] {action} warning: {exc}")


async def fetch_server_metrics(
    base_url: str, session: aiohttp.ClientSession
) -> ServerMetricsSnapshot:
    now = time.perf_counter()
    prefix_queries: float | None = None
    prefix_hits: float | None = None
    num_drafts: int | None = None
    num_draft_tokens: int | None = None
    num_accepted_tokens: int | None = None
    accepted_per_pos: dict[int, int] = {}

    try:
        async with session.get(f"{base_url}/metrics") as resp:
            if resp.status != 200:
                return ServerMetricsSnapshot(
                    timestamp=now,
                    prefix_cache_queries=None,
                    prefix_cache_hits=None,
                    spec_num_drafts=None,
                    spec_num_draft_tokens=None,
                    spec_num_accepted_tokens=None,
                )
            text = await resp.text()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return ServerMetricsSnapshot(
            timestamp=now,
            prefix_cache_queries=None,
            prefix_cache_hits=None,
            spec_num_drafts=None,
            spec_num_draft_tokens=None,
            spec_num_accepted_tokens=None,
        )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        metric_name = parts[0].split("{", 1)[0]
        try:
            val = float(parts[-1])
        except ValueError:
            continue

        if metric_name in {"vllm:prefix_cache_queries", "vllm:prefix_cache_queries_total"}:
            prefix_queries = (prefix_queries or 0.0) + val
        elif metric_name in {"vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total"}:
            prefix_hits = (prefix_hits or 0.0) + val
        elif metric_name in {"vllm:spec_decode_num_drafts", "vllm:spec_decode_num_drafts_total"}:
            num_drafts = (num_drafts or 0) + int(val)
        elif metric_name in {
            "vllm:spec_decode_num_draft_tokens",
            "vllm:spec_decode_num_draft_tokens_total",
        }:
            num_draft_tokens = (num_draft_tokens or 0) + int(val)
        elif metric_name in {
            "vllm:spec_decode_num_accepted_tokens",
            "vllm:spec_decode_num_accepted_tokens_total",
        }:
            num_accepted_tokens = (num_accepted_tokens or 0) + int(val)
        elif metric_name in {
            "vllm:spec_decode_num_accepted_tokens_per_pos",
            "vllm:spec_decode_num_accepted_tokens_per_pos_total",
        }:
            marker = 'position="'
            if marker in line:
                start = line.index(marker) + len(marker)
                end = line.index('"', start)
                pos = int(line[start:end])
                accepted_per_pos[pos] = accepted_per_pos.get(pos, 0) + int(val)

    return ServerMetricsSnapshot(
        timestamp=now,
        prefix_cache_queries=prefix_queries,
        prefix_cache_hits=prefix_hits,
        spec_num_drafts=num_drafts,
        spec_num_draft_tokens=num_draft_tokens,
        spec_num_accepted_tokens=num_accepted_tokens,
        spec_accepted_per_pos=accepted_per_pos,
    )


async def wait_metrics_advance(
    base_url: str,
    session: aiohttp.ClientSession,
    baseline: ServerMetricsSnapshot,
    phase_results: list[TurnResult],
    min_settle_s: float = 2.2,
    max_wait_s: float = 5.0,
) -> ServerMetricsSnapshot:
    """Wait for async engine Prometheus stats to flush after a phase."""
    ok_results = [r for r in phase_results if r.success]
    if not ok_results or baseline.prefix_cache_queries is None:
        return await fetch_server_metrics(base_url, session)

    await asyncio.sleep(min_settle_s)
    expected_min_queries = max(
        1.0, float(sum(max(0, r.prompt_tokens - 32) for r in ok_results))
    )

    deadline = time.monotonic() + max_wait_s
    snap = await fetch_server_metrics(base_url, session)
    while time.monotonic() < deadline:
        delta_q = (
            (snap.prefix_cache_queries - baseline.prefix_cache_queries)
            if snap.prefix_cache_queries is not None
            else None
        )
        if delta_q is not None and delta_q >= expected_min_queries:
            await asyncio.sleep(0.25)
            snap2 = await fetch_server_metrics(base_url, session)
            if (
                snap2.prefix_cache_queries == snap.prefix_cache_queries
                and snap2.spec_num_drafts == snap.spec_num_drafts
            ):
                return snap2
            snap = snap2
            continue
        await asyncio.sleep(0.25)
        snap = await fetch_server_metrics(base_url, session)
    return snap


def compute_server_delta(
    before: ServerMetricsSnapshot, after: ServerMetricsSnapshot
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if (
        before.prefix_cache_queries is not None
        and after.prefix_cache_queries is not None
        and before.prefix_cache_hits is not None
        and after.prefix_cache_hits is not None
    ):
        q = after.prefix_cache_queries - before.prefix_cache_queries
        h = after.prefix_cache_hits - before.prefix_cache_hits
        out["prefix_cache"] = {
            "queries_tokens": q,
            "hits_tokens": h,
            "hit_rate_pct": (h / q * 100.0) if q > 0 else 0.0,
        }

    if (
        before.spec_num_drafts is not None
        and after.spec_num_drafts is not None
        and before.spec_num_draft_tokens is not None
        and after.spec_num_draft_tokens is not None
        and before.spec_num_accepted_tokens is not None
        and after.spec_num_accepted_tokens is not None
    ):
        drafts = after.spec_num_drafts - before.spec_num_drafts
        draft_tokens = after.spec_num_draft_tokens - before.spec_num_draft_tokens
        accepted = after.spec_num_accepted_tokens - before.spec_num_accepted_tokens
        per_pos_rates: list[float] = []
        if drafts > 0:
            all_pos = sorted(
                set(before.spec_accepted_per_pos.keys())
                | set(after.spec_accepted_per_pos.keys())
            )
            for pos in all_pos:
                d_pos = after.spec_accepted_per_pos.get(
                    pos, 0
                ) - before.spec_accepted_per_pos.get(pos, 0)
                per_pos_rates.append(d_pos / drafts)
        out["spec_decode"] = {
            "num_drafts": drafts,
            "draft_tokens": draft_tokens,
            "accepted_tokens": accepted,
            "acceptance_rate_pct": (
                accepted / draft_tokens * 100.0 if draft_tokens > 0 else 0.0
            ),
            "mean_acceptance_length": (
                1.0 + accepted / drafts if drafts > 0 else 0.0
            ),
            "per_position_acceptance_rates": per_pos_rates,
        }
    return out


async def send_chat_turn(
    session: aiohttp.ClientSession,
    chat_url: str,
    model_id: str,
    tokenizer: Any,
    messages: list[dict[str, str]],
    conv_id: int,
    turn_idx: int,
    phase: str,
    in_measure_window: bool,
    incremental_tokens_est: int,
    max_tokens: int,
    temperature: float,
    ignore_eos: bool,
    enable_thinking: bool,
    cache_salt: str | None,
    timeout_sec: float,
) -> tuple[TurnResult, str]:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt

    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    itl: list[float] = []
    ttft = 0.0
    latency = 0.0
    prompt_tokens = 0
    output_tokens = 0
    chunk_token_events = 0
    first_token_seen = False

    st = time.perf_counter()
    last_token_ts = st
    success = False
    error_msg = ""

    try:
        async with session.post(chat_url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                error_msg = f"HTTP {resp.status}: {await resp.text()}"
            else:
                handler = StreamedResponseHandler()
                async for chunk_bytes in resp.content.iter_any():
                    if not chunk_bytes:
                        continue
                    for message in handler.add_chunk(chunk_bytes):
                        for raw_line in message.split("\n"):
                            if not raw_line or raw_line.startswith(":"):
                                continue
                            if not raw_line.startswith("data:"):
                                continue
                            data_str = raw_line.removeprefix("data:").strip()
                            if not data_str or data_str == "[DONE]":
                                continue
                            ts = time.perf_counter()
                            data = json.loads(data_str)
                            if choices := data.get("choices"):
                                delta = choices[0].get("delta") or {}
                                c_text = delta.get("content") or ""
                                r_text = (
                                    delta.get("reasoning_content")
                                    or delta.get("reasoning")
                                    or ""
                                )
                                piece = c_text or r_text
                                if piece:
                                    if not first_token_seen:
                                        first_token_seen = True
                                        ttft = ts - st
                                    else:
                                        itl.append(ts - last_token_ts)
                                    last_token_ts = ts
                                    chunk_token_events += 1
                                    if c_text:
                                        content_parts.append(c_text)
                                    if r_text:
                                        reasoning_parts.append(r_text)
                            if usage := data.get("usage"):
                                if usage.get("prompt_tokens") is not None:
                                    prompt_tokens = int(usage["prompt_tokens"])
                                if usage.get("completion_tokens") is not None:
                                    output_tokens = int(usage["completion_tokens"])

                latency = last_token_ts - st
                if first_token_seen:
                    success = True
                else:
                    error_msg = "Stream ended without receiving any token chunks."
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"

    assistant_reply = "".join(content_parts)
    assistant_history_text = assistant_reply or "".join(reasoning_parts)

    if output_tokens <= 0:
        if assistant_history_text and tokenizer is not None:
            output_tokens = len(
                tokenizer.encode(assistant_history_text, add_special_tokens=False)
            )
        else:
            output_tokens = chunk_token_events

    if prompt_tokens <= 0 and tokenizer is not None:
        prompt_tokens = sum(
            len(tokenizer.encode(m["content"], add_special_tokens=False))
            for m in messages
        )

    tpot: float | None = None
    if success and output_tokens > 1:
        tpot = max(0.0, latency - ttft) / (output_tokens - 1)

    history_tokens = max(0, prompt_tokens - incremental_tokens_est)

    result = TurnResult(
        conv_id=conv_id,
        turn_idx=turn_idx,
        phase=phase,
        in_measure_window=in_measure_window,
        history_prompt_tokens=history_tokens,
        incremental_input_tokens=incremental_tokens_est,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        start_time=st,
        ttft=ttft,
        latency=latency,
        tpot=tpot,
        itl=itl,
        success=success,
        error=error_msg,
    )
    return result, assistant_history_text


async def run_conversation_range(
    plan: ConversationPlan,
    messages: list[dict[str, str]],
    turn_start: int,
    turn_end: int,
    phase: str,
    measure_start: int,
    measure_end: int,
    semaphore: asyncio.Semaphore,
    rate_limiter: RequestRateLimiter,
    session: aiohttp.ClientSession,
    chat_url: str,
    model_id: str,
    tokenizer: Any,
    max_tokens: int,
    temperature: float,
    ignore_eos: bool,
    enable_thinking: bool,
    timeout_sec: float,
) -> list[TurnResult]:
    results: list[TurnResult] = []
    max_available = len(plan.user_turns)
    actual_end = min(turn_end, max_available)

    for turn_idx in range(turn_start, actual_end + 1):
        user_msg = plan.user_turns[turn_idx - 1]
        inc_tokens = plan.turn_input_token_targets[turn_idx - 1]
        if turn_idx == 1 and plan.system_prompt and tokenizer is not None:
            inc_tokens += len(
                tokenizer.encode(plan.system_prompt, add_special_tokens=False)
            )
        messages.append({"role": "user", "content": user_msg})

        in_window = measure_start <= turn_idx <= measure_end
        effective_phase = "window" if in_window else phase

        await rate_limiter.acquire()
        async with semaphore:
            turn_res, assistant_text = await send_chat_turn(
                session=session,
                chat_url=chat_url,
                model_id=model_id,
                tokenizer=tokenizer,
                messages=messages,
                conv_id=plan.conv_id,
                turn_idx=turn_idx,
                phase=effective_phase,
                in_measure_window=in_window,
                incremental_tokens_est=inc_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                ignore_eos=ignore_eos,
                enable_thinking=enable_thinking,
                cache_salt=plan.cache_salt,
                timeout_sec=timeout_sec,
            )
        results.append(turn_res)
        if not turn_res.success:
            break
        messages.append(
            {"role": "assistant", "content": assistant_text or "(acknowledged)"}
        )

    return results


def summarize_results(
    results: list[TurnResult], wall_duration_s: float
) -> dict[str, Any]:
    completed = [r for r in results if r.success]
    failed = len(results) - len(completed)
    ttfts_ms = [r.ttft * 1000.0 for r in completed]
    tpots_ms = [r.tpot * 1000.0 for r in completed if r.tpot is not None]
    itls_ms = [v * 1000.0 for r in completed for v in r.itl]
    e2e_ms = [r.latency * 1000.0 for r in completed]

    total_prompt_tokens = sum(r.prompt_tokens for r in completed)
    total_inc_tokens = sum(r.incremental_input_tokens for r in completed)
    total_output_tokens = sum(r.output_tokens for r in completed)
    dur = max(wall_duration_s, 1e-6)

    def _stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0}
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
        }

    return {
        "completed": len(completed),
        "failed": failed,
        "duration_s": wall_duration_s,
        "request_throughput_rps": len(completed) / dur,
        "output_throughput_tok_s": total_output_tokens / dur,
        "incremental_input_throughput_tok_s": total_inc_tokens / dur,
        "total_prompt_tokens": total_prompt_tokens,
        "total_incremental_input_tokens": total_inc_tokens,
        "total_output_tokens": total_output_tokens,
        "mean_history_prompt_tokens": (
            float(np.mean([r.history_prompt_tokens for r in completed]))
            if completed
            else 0.0
        ),
        "mean_incremental_input_tokens": (
            float(np.mean([r.incremental_input_tokens for r in completed]))
            if completed
            else 0.0
        ),
        "mean_prompt_tokens": (
            float(np.mean([r.prompt_tokens for r in completed]))
            if completed
            else 0.0
        ),
        "mean_output_tokens": (
            float(np.mean([r.output_tokens for r in completed]))
            if completed
            else 0.0
        ),
        "ttft_ms": _stats(ttfts_ms),
        "tpot_ms": _stats(tpots_ms),
        "itl_ms": _stats(itls_ms),
        "e2el_ms": _stats(e2e_ms),
    }


def print_per_turn_table(results: list[TurnResult]) -> list[dict[str, Any]]:
    turns = sorted({r.turn_idx for r in results})
    per_turn_rows: list[dict[str, Any]] = []

    print("\n" + "=" * 108)
    print(" Multi-Turn Progression Summary (per turn) ".center(108, "="))
    header = (
        f"{'Turn':<6} {'Win':<4} {'OK/Tot':<7} {'HistTok':>8} {'NewTok':>8} "
        f"{'Prompt':>8} {'OutTok':>7} {'TTFT_ms':>9} {'TTFT_P99':>9} "
        f"{'TPOT_ms':>9} {'ITL_ms':>9}"
    )
    print(header)
    print("-" * 108)

    for turn_idx in turns:
        t_res = [r for r in results if r.turn_idx == turn_idx]
        if not t_res:
            continue
        ok = [r for r in t_res if r.success]
        # Span wall time of turn_idx for informational throughput
        if ok:
            t_min = min(r.start_time for r in ok)
            t_max = max(r.start_time + r.latency for r in ok)
            summary = summarize_results(t_res, max(t_max - t_min, 1e-6))
        else:
            summary = summarize_results(t_res, 1e-6)
        in_win = any(r.in_measure_window for r in t_res)
        row = {"turn": turn_idx, "in_measure_window": in_win, **summary}
        per_turn_rows.append(row)

        print(
            f"T{turn_idx:<5d} {'*' if in_win else '-':<4} "
            f"{summary['completed']}/{len(t_res):<5d} "
            f"{summary['mean_history_prompt_tokens']:8.1f} "
            f"{summary['mean_incremental_input_tokens']:8.1f} "
            f"{summary['mean_prompt_tokens']:8.1f} "
            f"{summary['mean_output_tokens']:7.1f} "
            f"{summary['ttft_ms']['mean']:9.2f} "
            f"{summary['ttft_ms']['p99']:9.2f} "
            f"{summary['tpot_ms']['mean']:9.2f} "
            f"{summary['itl_ms']['mean']:9.2f}"
        )
    print("=" * 108)
    return per_turn_rows


def print_window_summary(
    measure_start: int,
    measure_end: int,
    metrics_scope: str,
    window_summary: dict[str, Any],
    server_delta: dict[str, Any],
) -> None:
    print(
        "\n"
        + f" Measured Turn Window [T{measure_start}..T{measure_end}] Summary ".center(
            70, "="
        )
    )
    print(f"{'Metrics isolation scope:':<42} {metrics_scope}")
    print(
        f"{'Requests completed / failed:':<42} "
        f"{window_summary['completed']} / {window_summary['failed']}"
    )
    print(f"{'Window duration (s):':<42} {window_summary['duration_s']:.3f}")
    print(
        f"{'Mean history / incremental tokens:':<42} "
        f"{window_summary['mean_history_prompt_tokens']:.1f} / "
        f"{window_summary['mean_incremental_input_tokens']:.1f}"
    )
    print(
        f"{'Request throughput (req/s):':<42} "
        f"{window_summary['request_throughput_rps']:.2f}"
    )
    print(
        f"{'Output token throughput (tok/s):':<42} "
        f"{window_summary['output_throughput_tok_s']:.2f}"
    )
    print(
        f"{'Incremental input throughput (tok/s):':<42} "
        f"{window_summary['incremental_input_throughput_tok_s']:.2f}"
    )
    print(
        f"{'TTFT mean / median / p99 (ms):':<42} "
        f"{window_summary['ttft_ms']['mean']:.2f} / "
        f"{window_summary['ttft_ms']['median']:.2f} / "
        f"{window_summary['ttft_ms']['p99']:.2f}"
    )
    print(
        f"{'TPOT mean / median / p99 (ms):':<42} "
        f"{window_summary['tpot_ms']['mean']:.2f} / "
        f"{window_summary['tpot_ms']['median']:.2f} / "
        f"{window_summary['tpot_ms']['p99']:.2f}"
    )
    print(
        f"{'ITL mean / median / p99 (ms):':<42} "
        f"{window_summary['itl_ms']['mean']:.2f} / "
        f"{window_summary['itl_ms']['median']:.2f} / "
        f"{window_summary['itl_ms']['p99']:.2f}"
    )

    if pc := server_delta.get("prefix_cache"):
        print("-" * 70)
        print(f"{'Prefix cache queries (tokens):':<42} {pc['queries_tokens']:.0f}")
        print(f"{'Prefix cache hits (tokens):':<42} {pc['hits_tokens']:.0f}")
        print(f"{'Prefix cache hit rate (%):':<42} {pc['hit_rate_pct']:.2f}%")

    if sd := server_delta.get("spec_decode"):
        print("-" * 70)
        print(f"{'Spec decode num drafts:':<42} {sd['num_drafts']}")
        print(f"{'Spec decode draft tokens:':<42} {sd['draft_tokens']}")
        print(f"{'Spec decode accepted tokens:':<42} {sd['accepted_tokens']}")
        print(
            f"{'Spec decode acceptance rate (%):':<42} "
            f"{sd['acceptance_rate_pct']:.2f}%"
        )
        print(
            f"{'Spec decode mean acceptance length:':<42} "
            f"{sd['mean_acceptance_length']:.2f}"
        )
        if sd["per_position_acceptance_rates"]:
            rates_str = ", ".join(
                f"pos{i}={r * 100:.1f}%"
                for i, r in enumerate(sd["per_position_acceptance_rates"])
            )
            print(f"{'Spec decode per-position acceptance:':<42} {rates_str}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Multi-turn vLLM serving benchmark with intermediate turn-window "
            "incremental measurement, prefix-cache & spec-decode deltas, and timeline plot."
        )
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=14800)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument(
        "--dataset-mode",
        choices=("corpus", "sharegpt"),
        default="corpus",
        help="corpus builds coherent token-budgeted turns; sharegpt reads real multi-turn logs.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Conversation dataset path (for --dataset-mode sharegpt).",
    )
    parser.add_argument(
        "--corpus-paths",
        type=Path,
        nargs="+",
        default=[
            script_dir / "long_context_data" / "squad_dev_v1.1.json",
            script_dir / "data" / "spec_bench.jsonl",
        ],
        help="Local text corpus files for --dataset-mode corpus.",
    )
    parser.add_argument("--num-conversations", type=int, default=4)
    parser.add_argument("--num-turns", type=int, default=4)
    parser.add_argument("--common-prefix-tokens", type=int, default=0)
    parser.add_argument("--initial-context-tokens", type=int, default=640)
    parser.add_argument("--turn-input-tokens", type=int, default=256)
    parser.add_argument("--turn-output-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-cache-salt", action="store_true")
    parser.add_argument("--cache-salt", type=str, default=None)
    parser.add_argument("--timeout-sec", type=float, default=600.0)

    parser.add_argument(
        "--measure-turns",
        type=str,
        default=None,
        help="1-indexed inclusive turn window START:END (e.g. '3:4'). Default: full run.",
    )
    parser.add_argument(
        "--phase-barrier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Synchronize before/after the measured turn window so server /metrics "
            "(prefix cache & spec decode) and profiler capture only that window."
        ),
    )
    parser.add_argument(
        "--profile-window",
        action="store_true",
        help="Call /start_profile before and /stop_profile after the measured window.",
    )

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=script_dir / "multi_turn_results",
    )
    parser.add_argument("--result-filename", type=str, default=None)
    parser.add_argument("--plot-timeline", action="store_true")
    parser.add_argument("--timeline-itl-thresholds", type=str, default="50,150")
    parser.add_argument(
        "--timeline-scope",
        choices=("all", "window"),
        default="all",
        help="Whether the timeline HTML plots all turns or only the measured window.",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    base_url = (args.base_url or f"http://{args.host}:{args.port}").rstrip("/")
    chat_url = f"{base_url}/v1/chat/completions"
    measure_start, measure_end = parse_turn_window(args.measure_turns, args.num_turns)

    connector = aiohttp.TCPConnector(limit=max(64, args.max_concurrency * 2))
    async with aiohttp.ClientSession(connector=connector) as session:
        model_id = args.model or await fetch_model_id(base_url, session)
        tokenizer_id = args.tokenizer or model_id
        tokenizer = load_tokenizer(tokenizer_id, args.trust_remote_code)

        if args.disable_cache_salt:
            run_salt = None
        else:
            run_salt = args.cache_salt or f"mt-{int(time.time() * 1e6)}"

        if args.dataset_mode == "corpus":
            plans = build_corpus_conversations(
                tokenizer=tokenizer,
                corpus_paths=args.corpus_paths,
                num_conversations=args.num_conversations,
                num_turns=args.num_turns,
                common_prefix_tokens=args.common_prefix_tokens,
                initial_context_tokens=args.initial_context_tokens,
                turn_input_tokens=args.turn_input_tokens,
                seed=args.seed,
                run_salt=run_salt,
            )
        else:
            if args.dataset_path is None:
                raise ValueError("--dataset-path is required with --dataset-mode sharegpt")
            plans = build_sharegpt_conversations(
                tokenizer=tokenizer,
                dataset_path=args.dataset_path,
                num_conversations=args.num_conversations,
                num_turns=args.num_turns,
                min_turns_required=measure_end,
                seed=args.seed,
                run_salt=run_salt,
            )

        print(
            f"[multi_turn] model={model_id}, conversations={len(plans)}, "
            f"turns={args.num_turns}, measure_window=T{measure_start}..T{measure_end}, "
            f"concurrency={args.max_concurrency}, phase_barrier={args.phase_barrier}, "
            f"cache_salt={run_salt or '<disabled>'}"
        )

        # Maintain mutable message history per conversation across phases.
        histories: dict[int, list[dict[str, str]]] = {}
        for plan in plans:
            msgs: list[dict[str, str]] = []
            if plan.system_prompt:
                msgs.append({"role": "system", "content": plan.system_prompt})
            histories[plan.conv_id] = msgs

        semaphore = asyncio.Semaphore(args.max_concurrency)
        rate_limiter = RequestRateLimiter(args.request_rate)
        all_results: list[TurnResult] = []

        if args.phase_barrier:
            metrics_scope = f"window_isolated (T{measure_start}..T{measure_end})"
            # Phase 1: Warmup / history build turns [1 .. measure_start - 1]
            if measure_start > 1:
                baseline_before_warmup = await fetch_server_metrics(base_url, session)
                print(
                    f"[phase 1] Running warmup turns 1..{measure_start - 1} "
                    f"across {len(plans)} conversations..."
                )
                warmup_batches = await asyncio.gather(
                    *(
                        run_conversation_range(
                            plan=p,
                            messages=histories[p.conv_id],
                            turn_start=1,
                            turn_end=measure_start - 1,
                            phase="warmup",
                            measure_start=measure_start,
                            measure_end=measure_end,
                            semaphore=semaphore,
                            rate_limiter=rate_limiter,
                            session=session,
                            chat_url=chat_url,
                            model_id=model_id,
                            tokenizer=tokenizer,
                            max_tokens=args.turn_output_tokens,
                            temperature=args.temperature,
                            ignore_eos=args.ignore_eos,
                            enable_thinking=args.enable_thinking,
                            timeout_sec=args.timeout_sec,
                        )
                        for p in plans
                    )
                )
                warmup_results = [r for batch in warmup_batches for r in batch]
                all_results.extend(warmup_results)
                before_window_snap = await wait_metrics_advance(
                    base_url,
                    session,
                    baseline_before_warmup,
                    warmup_results,
                )
            else:
                before_window_snap = await fetch_server_metrics(base_url, session)

            # Phase 2: Measured intermediate turn window [measure_start .. measure_end]
            print(
                f"[phase 2] Measuring turns {measure_start}..{measure_end} "
                f"across {len(plans)} conversations..."
            )
            if args.profile_window:
                await post_profile_cmd(base_url, session, "start_profile")

            win_start_ts = time.perf_counter()
            win_batches = await asyncio.gather(
                *(
                    run_conversation_range(
                        plan=p,
                        messages=histories[p.conv_id],
                        turn_start=measure_start,
                        turn_end=measure_end,
                        phase="window",
                        measure_start=measure_start,
                        measure_end=measure_end,
                        semaphore=semaphore,
                        rate_limiter=rate_limiter,
                        session=session,
                        chat_url=chat_url,
                        model_id=model_id,
                        tokenizer=tokenizer,
                        max_tokens=args.turn_output_tokens,
                        temperature=args.temperature,
                        ignore_eos=args.ignore_eos,
                        enable_thinking=args.enable_thinking,
                        timeout_sec=args.timeout_sec,
                    )
                    for p in plans
                )
            )
            window_wall_s = time.perf_counter() - win_start_ts

            if args.profile_window:
                await post_profile_cmd(base_url, session, "stop_profile")

            window_results = [r for batch in win_batches for r in batch]
            all_results.extend(window_results)
            after_window_snap = await wait_metrics_advance(
                base_url,
                session,
                before_window_snap,
                window_results,
            )

            # Phase 3: Tail turns [measure_end + 1 .. num_turns]
            if measure_end < args.num_turns:
                print(
                    f"[phase 3] Running remaining turns {measure_end + 1}..{args.num_turns}..."
                )
                tail_batches = await asyncio.gather(
                    *(
                        run_conversation_range(
                            plan=p,
                            messages=histories[p.conv_id],
                            turn_start=measure_end + 1,
                            turn_end=args.num_turns,
                            phase="tail",
                            measure_start=measure_start,
                            measure_end=measure_end,
                            semaphore=semaphore,
                            rate_limiter=rate_limiter,
                            session=session,
                            chat_url=chat_url,
                            model_id=model_id,
                            tokenizer=tokenizer,
                            max_tokens=args.turn_output_tokens,
                            temperature=args.temperature,
                            ignore_eos=args.ignore_eos,
                            enable_thinking=args.enable_thinking,
                            timeout_sec=args.timeout_sec,
                        )
                        for p in plans
                    )
                )
                all_results.extend(r for batch in tail_batches for r in batch)
        else:
            metrics_scope = "full_run_unbarriered"
            before_window_snap = await fetch_server_metrics(base_url, session)
            if args.profile_window:
                await post_profile_cmd(base_url, session, "start_profile")
            full_batches = await asyncio.gather(
                *(
                    run_conversation_range(
                        plan=p,
                        messages=histories[p.conv_id],
                        turn_start=1,
                        turn_end=args.num_turns,
                        phase="unbarriered",
                        measure_start=measure_start,
                        measure_end=measure_end,
                        semaphore=semaphore,
                        rate_limiter=rate_limiter,
                        session=session,
                        chat_url=chat_url,
                        model_id=model_id,
                        tokenizer=tokenizer,
                        max_tokens=args.turn_output_tokens,
                        temperature=args.temperature,
                        ignore_eos=args.ignore_eos,
                        enable_thinking=args.enable_thinking,
                        timeout_sec=args.timeout_sec,
                    )
                    for p in plans
                )
            )
            if args.profile_window:
                await post_profile_cmd(base_url, session, "stop_profile")
            all_results = [r for batch in full_batches for r in batch]
            window_results = [r for r in all_results if r.in_measure_window]
            ok_win = [r for r in window_results if r.success]
            if ok_win:
                window_wall_s = max(r.start_time + r.latency for r in ok_win) - min(
                    r.start_time for r in ok_win
                )
            else:
                window_wall_s = 0.0
            after_window_snap = await wait_metrics_advance(
                base_url,
                session,
                before_window_snap,
                all_results,
            )

    failed_errors = [r for r in all_results if not r.success]
    if failed_errors:
        for r in failed_errors[:5]:
            print(
                f"[ERROR] conv={r.conv_id} turn={r.turn_idx} failed: {r.error}"
            )

    per_turn_summary = print_per_turn_table(all_results)
    window_summary = summarize_results(window_results, window_wall_s)
    server_delta = compute_server_delta(before_window_snap, after_window_snap)
    print_window_summary(
        measure_start,
        measure_end,
        metrics_scope,
        window_summary,
        server_delta,
    )

    # Persist JSON report
    args.result_dir.mkdir(parents=True, exist_ok=True)
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_model_id = model_id.split("/")[-1]
    out_filename = (
        args.result_filename
        or f"multi_turn-{base_model_id}-T{measure_start}_{measure_end}-{current_dt}.json"
    )
    result_path = args.result_dir / out_filename

    ok_all = [r for r in all_results if r.success]
    full_wall_s = (
        max(r.start_time + r.latency for r in ok_all)
        - min(r.start_time for r in ok_all)
        if ok_all
        else 0.0
    )

    payload = {
        "date": current_dt,
        "model_id": model_id,
        "dataset_mode": args.dataset_mode,
        "num_conversations": len(plans),
        "num_turns": args.num_turns,
        "max_concurrency": args.max_concurrency,
        "request_rate": (
            args.request_rate if args.request_rate < float("inf") else "inf"
        ),
        "common_prefix_tokens": args.common_prefix_tokens,
        "initial_context_tokens": args.initial_context_tokens,
        "turn_input_tokens": args.turn_input_tokens,
        "turn_output_tokens": args.turn_output_tokens,
        "measure_turns": [measure_start, measure_end],
        "phase_barrier": args.phase_barrier,
        "metrics_scope": metrics_scope,
        "window_summary": window_summary,
        "overall_summary": summarize_results(all_results, full_wall_s),
        "server_delta": server_delta,
        "per_turn_summary": per_turn_summary,
        "requests": [asdict(r) for r in all_results],
    }
    result_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved multi-turn result JSON to: {result_path}")

    if args.plot_timeline:
        from vllm.benchmarks.plot import generate_timeline_plot

        plot_source = (
            window_results if args.timeline_scope == "window" else all_results
        )
        plot_items = sorted(
            [r for r in plot_source if r.success],
            key=lambda r: (r.start_time, r.conv_id, r.turn_idx),
        )
        if plot_items:
            per_request_data = [
                {
                    "start_time": r.start_time,
                    "ttft": r.ttft,
                    "itl": r.itl,
                    "latency": r.latency,
                    "prompt_len": r.prompt_tokens,
                    "output_tokens": r.output_tokens,
                }
                for r in plot_items
            ]
            thresholds_sec = [
                float(x) / 1000.0
                for x in args.timeline_itl_thresholds.split(",")
                if x.strip()
            ]
            timeline_path = result_path.with_suffix(".timeline.html")
            generate_timeline_plot(
                per_request_data,
                timeline_path,
                itl_thresholds=thresholds_sec or None,
            )
        else:
            print("[warning] No successful requests to plot timeline.")

    if window_summary["completed"] == 0 or window_summary["failed"] > 0:
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
