#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CMT-Eval multi-turn prefix-cache differential test.

Each record starts with one question.  Later rounds submit independent
branches concurrently; only branch ``id: 0`` and its answer become the chat
history for the next round.  Every hot request is compared with a cold,
cache-isolated reference request.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request
import uuid


QUERY_METRIC = "vllm:prefix_cache_queries"
HIT_METRIC = "vllm:prefix_cache_hits"
HTTP_TIMEOUT_S = 600.0
METRICS_TIMEOUT_S = 10.0


class TestFailure(RuntimeError):
    pass


def repository_root() -> Path:
    script = Path(__file__).resolve()
    for directory in script.parents:
        if (directory / "vllm_scripts" / "common.sh").is_file():
            return directory
    raise TestFailure(f"无法从脚本路径定位仓库根目录: {script}")


def default_dataset_path() -> Path:
    return repository_root() / "dateset" / "cmt_eval_multiturn_6.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


class RunLogger:
    """Persist aggregate and per-request machine-readable diagnostics."""

    def __init__(self, directory: Path, metadata: dict[str, Any]) -> None:
        self.directory = directory
        self.conversations_dir = directory / "conversations"
        self.events_path = directory / "events.jsonl"
        self.report_path = directory / "report.json"
        self.conversations_dir.mkdir(parents=True)
        self.report: dict[str, Any] = {
            "status": "RUNNING",
            "started_at": utc_now(),
            "metadata": metadata,
            "rounds": [],
        }
        write_json(self.report_path, self.report)
        self.event("run_start", metadata)

    def conversation_dir(self, conversation_id: str) -> Path:
        """Return a stable, filesystem-safe directory for one dataset record."""
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", conversation_id).strip("_")
        return self.conversations_dir / (slug or "unnamed")

    def write_conversation_reports(self) -> None:
        """Split the aggregate report into one complete JSON file per record."""
        conversation_ids = {
            item["conversation_id"]
            for round_result in self.report["rounds"]
            for item in round_result["results"]
        }
        for conversation_id in conversation_ids:
            rounds = []
            for round_result in self.report["rounds"]:
                results = [
                    item
                    for item in round_result["results"]
                    if item["conversation_id"] == conversation_id
                ]
                if not results:
                    continue
                rounds.append(
                    {
                        "round": round_result["round"],
                        "requests": len(results),
                        "cache_query_tokens": round_result["cache_query_tokens"],
                        "cache_hit_tokens": round_result["cache_hit_tokens"],
                        "cache_hit_ratio": round_result["cache_hit_ratio"],
                        "status": "FAIL"
                        if any(item["status"] == "FAIL" for item in results)
                        else "PASS",
                        "results": results,
                    }
                )
            report = {
                "status": "FAIL"
                if any(round_result["status"] == "FAIL" for round_result in rounds)
                else "PASS",
                "conversation_id": conversation_id,
                "metadata": self.report["metadata"],
                "rounds": rounds,
            }
            directory = self.conversation_dir(conversation_id)
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "report.json", report)

    def write_question_reports(self, round_result: dict[str, Any]) -> None:
        """Write one independently searchable file for every round/question."""
        for item in round_result["results"]:
            conversation_id = item["conversation_id"]
            directory = self.conversation_dir(conversation_id)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (
                "cmt_eval_prefix_cache_"
                f"round_{round_result['round']:04d}_question_{item['question_id']:04d}.json"
            )
            write_json(
                path,
                {
                    "metadata": self.report["metadata"],
                    "conversation_id": conversation_id,
                    "round": round_result["round"],
                    "cache_query_tokens": round_result["cache_query_tokens"],
                    "cache_hit_tokens": round_result["cache_hit_tokens"],
                    "cache_hit_ratio": round_result["cache_hit_ratio"],
                    "result": item,
                },
            )

    def response_stream_path(
        self, conversation_id: str, round_id: int, question_id: int, kind: str
    ) -> Path:
        directory = self.conversation_dir(conversation_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / (
            "cmt_eval_prefix_cache_"
            f"round_{round_id:04d}_question_{question_id:04d}_{kind}.log"
        )

    def event(self, name: str, data: Any) -> None:
        entry = {"timestamp": utc_now(), "event": name, "data": data}
        append_text(self.events_path, json.dumps(entry, ensure_ascii=False) + "\n")
        self.report["last_event"] = entry
        write_json(self.report_path, self.report)

    def complete_round(self, result: dict[str, Any]) -> None:
        self.report["rounds"].append(result)
        self.write_conversation_reports()
        self.write_question_reports(result)
        self.event(
            "round_complete",
            {key: result[key] for key in ("round", "requests", "status", "failures")},
        )

    def finish(self, status: str, detail: str = "") -> None:
        self.report["status"] = status
        self.report["finished_at"] = utc_now()
        if detail:
            self.report["detail"] = detail
        self.event(
            "run_complete" if status == "PASS" else "run_failed", {"detail": detail}
        )


def source_env_with_preset(script_dir: Path, preset_file: str | None) -> dict[str, str]:
    repo_root = script_dir.parent
    commands = [
        "set -e",
        f"source {shlex.quote(str(repo_root / 'common.sh'))}",
        f"load_env_file {shlex.quote(str(repo_root / 'env.sh'))} >/dev/null",
    ]
    if preset_file:
        commands.append(
            f"load_preset_file {shlex.quote(str(Path(preset_file).resolve()))} >/dev/null"
        )
    else:
        commands.append(f"load_user_config {shlex.quote(str(repo_root))} >/dev/null")
    commands.append("env")
    result = subprocess.run(
        ["bash", "-c", "\n".join(commands)],
        capture_output=True,
        text=True,
        env=os.environ,
        check=False,
    )
    if result.returncode:
        raise TestFailure(f"加载服务环境失败:\n{result.stderr}")
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def request(
    url: str, *, method: str = "GET", payload: dict[str, Any] | None = None
) -> bytes:
    data = json.dumps(payload, ensure_ascii=False).encode() if payload else None
    headers = {"Authorization": "Bearer EMPTY"}
    if data:
        headers["Content-Type"] = "application/json"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, data=data, headers=headers, method=method),
            timeout=HTTP_TIMEOUT_S,
        ) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise TestFailure(f"{method} {url}: HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TestFailure(f"{method} {url}: {exc}") from exc


def metric_value(text: str, name: str) -> float | None:
    values = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].split("{", 1)[0] in {name, name + "_total"}:
            try:
                values.append(float(fields[1]))
            except ValueError:
                pass
    return sum(values) if values else None


def metrics(base_url: str) -> tuple[float, float]:
    text = request(base_url + "/metrics").decode("utf-8", errors="replace")
    query, hit = metric_value(text, QUERY_METRIC), metric_value(text, HIT_METRIC)
    if query is None or hit is None:
        raise TestFailure(f"/metrics 缺少 {QUERY_METRIC}/{HIT_METRIC}")
    return query, hit


def wait_metrics(
    base_url: str, before: tuple[float, float], expected: int
) -> tuple[float, float]:
    deadline = time.monotonic() + METRICS_TIMEOUT_S
    current = before
    while time.monotonic() < deadline:
        current = metrics(base_url)
        if current[0] - before[0] >= expected:
            break
        time.sleep(0.05)
    return current[0] - before[0], current[1] - before[1]


def load_dataset(path: Path) -> list[dict[str, Any]]:
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TestFailure(f"无法读取数据集 {path}: {exc}") from exc
    if not isinstance(records, list) or not records:
        raise TestFailure("数据集必须是非空 JSON 数组")
    for record in records:
        rounds = record.get("rounds") if isinstance(record, dict) else None
        if not isinstance(rounds, list) or not rounds:
            raise TestFailure("每条数据必须包含非空 rounds")
        for round_id, round_data in enumerate(rounds):
            questions = (
                round_data.get("questions") if isinstance(round_data, dict) else None
            )
            expected_count = 1 if round_id == 0 else 2
            if (
                round_data.get("round") != round_id
                or not isinstance(questions, list)
                or [question.get("id") for question in questions]
                != list(range(expected_count))
            ):
                raise TestFailure(
                    f"{record.get('id', '?')} round {round_id} 的分叉格式无效"
                )
    return records


def stream_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    salt: str,
    max_tokens: int | None,
    enable_thinking: bool,
    stream_path: Path,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": 0,
        "return_token_ids": True,
        "cache_salt": salt,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "stream": True,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    started = time.perf_counter()
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    prompt_ids: list[int] | None = None
    output_ids: list[int] = []
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    finish_reason: Any = None
    stop_reason: Any = None
    usage: dict[str, Any] = {}
    stream_chunks = 0
    choice_chunks = 0
    reasoning_chunks = 0
    content_chunks = 0
    try:
        request_obj = urllib.request.Request(
            base_url + "/v1/chat/completions",
            data=data,
            headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
            method="POST",
        )
        with stream_path.open("w", encoding="utf-8") as stream_file, urllib.request.urlopen(
            request_obj, timeout=HTTP_TIMEOUT_S
        ) as response:
            stream_file.write(json.dumps(payload, ensure_ascii=False, indent=2))
            stream_file.write("\n\n")
            wrote_reasoning_header = False
            wrote_content_header = False
            for raw_line in response:
                stream_chunks += 1
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                event_text = line.removeprefix("data:").strip()
                if event_text == "[DONE]":
                    continue
                event = json.loads(event_text)
                if "error" in event:
                    raise TestFailure(f"流式响应错误: {event['error']}")
                if event.get("prompt_token_ids") is not None:
                    prompt_ids = [int(token) for token in event["prompt_token_ids"]]
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice_chunks += 1
                choice = choices[0]
                delta = choice.get("delta") or {}
                reasoning = str(
                    delta.get("reasoning_content") or delta.get("reasoning") or ""
                )
                content = str(delta.get("content") or "")
                if reasoning:
                    reasoning_parts.append(reasoning)
                    reasoning_chunks += 1
                    if not wrote_reasoning_header:
                        stream_file.write("[reasoning_content]\n")
                        wrote_reasoning_header = True
                    stream_file.write(reasoning)
                    stream_file.flush()
                if content:
                    content_parts.append(content)
                    content_chunks += 1
                    if not wrote_content_header:
                        stream_file.write(
                            "\n\n[content]\n" if wrote_reasoning_header else "[content]\n"
                        )
                        wrote_content_header = True
                    stream_file.write(content)
                    stream_file.flush()
                if choice.get("token_ids") is not None:
                    output_ids.extend(int(token) for token in choice["token_ids"])
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                if choice.get("stop_reason") is not None:
                    stop_reason = choice["stop_reason"]
            stream_file.write("\n\n[summary]\n")
            stream_file.write(
                json.dumps(
                    {
                        "stream_chunks": stream_chunks,
                        "choice_chunks": choice_chunks,
                        "reasoning_chunks": reasoning_chunks,
                        "content_chunks": content_chunks,
                        "reasoning_chars": sum(len(item) for item in reasoning_parts),
                        "content_chars": sum(len(item) for item in content_parts),
                        "finish_reason": finish_reason,
                        "stop_reason": stop_reason,
                        "prompt_tokens": usage.get(
                            "prompt_tokens", len(prompt_ids) if prompt_ids else None
                        ),
                        "completion_tokens": usage.get(
                            "completion_tokens", len(output_ids)
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            stream_file.write("\n")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as exc:
        raise TestFailure(f"流式 Chat 响应异常: {exc}") from exc
    if prompt_ids is None or not output_ids:
        raise TestFailure(
            f"流式响应缺少 token IDs；原始 SSE: {stream_path}"
        )
    answer_content = "".join(content_parts)
    reasoning_content = "".join(reasoning_parts)
    return {
        "text": answer_content,
        "answer_content": answer_content,
        "reasoning_content": reasoning_content,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "prompt_ids": prompt_ids,
        "output_ids": output_ids,
        "prompt_tokens": int(usage.get("prompt_tokens", len(prompt_ids))),
        "output_tokens": int(usage.get("completion_tokens", len(output_ids))),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "request_payload": payload,
        "stream_path": str(stream_path),
    }


def send_batch(
    base_url: str,
    model: str,
    tasks: list[dict[str, Any]],
    kind: str,
    run_id: str,
    max_tokens: int | None,
    concurrency: int | None,
    enable_thinking: bool,
    logger: RunLogger,
    round_id: int,
) -> list[dict[str, Any]]:
    def send(index: int, task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        salt = f"{run_id}:cold:{index}" if kind == "cold" else f"{run_id}:hot"
        stream_path = logger.response_stream_path(
            task["state"]["record"]["id"], round_id, task["question"]["id"], kind
        )
        label = (
            f"{kind} round={round_id} conversation={task['state']['record']['id']} "
            f"q={task['question']['id']}"
        )
        print(f"[流] {label} 正在流式输出到 {stream_path}", flush=True)
        result = stream_chat(
            base_url,
            model,
            task[kind + "_messages"],
            salt,
            max_tokens,
            enable_thinking,
            stream_path,
        )
        print(f"[流] {label} 完成流式输出到 {stream_path}", flush=True)
        return index, result

    results = [None] * len(tasks)
    with ThreadPoolExecutor(
        max_workers=min(concurrency, len(tasks)) if concurrency else len(tasks)
    ) as executor:
        futures = [
            executor.submit(send, index, task) for index, task in enumerate(tasks)
        ]
        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result
    return results


def digest(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


def answer_summaries(
    tasks: list[dict[str, Any]], answers: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "conversation_id": task["state"]["record"]["id"],
            "question_id": task["question"]["id"],
            "prompt_tokens": answer["prompt_tokens"],
            "output_tokens": answer["output_tokens"],
            "output_sha256": digest(answer["output_ids"]),
            "latency_ms": answer["latency_ms"],
            "text": answer["text"],
            "answer_content": answer["answer_content"],
            "reasoning_content": answer["reasoning_content"],
            "finish_reason": answer["finish_reason"],
            "stop_reason": answer["stop_reason"],
            "request_payload": answer["request_payload"],
            "stream_path": answer["stream_path"],
        }
        for task, answer in zip(tasks, answers)
    ]


def run_test(
    base_url: str,
    model: str,
    records: list[dict[str, Any]],
    max_tokens: int | None,
    concurrency: int | None,
    max_rounds: int | None,
    enable_thinking: bool,
    logger: RunLogger,
) -> dict[str, Any]:
    run_id = uuid.uuid4().hex
    states = [
        {"record": record, "cold_history": [], "hot_history": []} for record in records
    ]
    round_results = []
    round_count = max(len(record["rounds"]) for record in records)
    if max_rounds is not None:
        round_count = min(round_count, max_rounds)
    for round_id in range(round_count):
        tasks = []
        for state in states:
            rounds = state["record"]["rounds"]
            if round_id >= len(rounds):
                continue
            for question in rounds[round_id]["questions"]:
                tasks.append(
                    {
                        "state": state,
                        "question": question,
                        "cold_messages": state["cold_history"]
                        + [{"role": "user", "content": question["question"]}],
                        "hot_messages": state["hot_history"]
                        + [{"role": "user", "content": question["question"]}],
                    }
                )
        logger.event(
            "round_start",
            {
                "round": round_id,
                "requests": [
                    {
                        "conversation_id": task["state"]["record"]["id"],
                        "question_id": task["question"]["id"],
                        "question": task["question"]["question"],
                    }
                    for task in tasks
                ],
            },
        )
        cold = send_batch(
            base_url,
            model,
            tasks,
            "cold",
            run_id,
            max_tokens,
            concurrency,
            enable_thinking,
            logger,
            round_id,
        )
        logger.event(
            "cold_complete",
            {"round": round_id, "responses": answer_summaries(tasks, cold)},
        )
        before = metrics(base_url)
        hot = send_batch(
            base_url,
            model,
            tasks,
            "hot",
            run_id,
            max_tokens,
            concurrency,
            enable_thinking,
            logger,
            round_id,
        )
        query_tokens, hit_tokens = wait_metrics(
            base_url, before, sum(item["prompt_tokens"] for item in hot)
        )
        logger.event(
            "hot_complete",
            {
                "round": round_id,
                "cache_query_tokens": query_tokens,
                "cache_hit_tokens": hit_tokens,
                "responses": answer_summaries(tasks, hot),
            },
        )
        failures, task_results = [], []
        main_answers: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        for task, cold_answer, hot_answer in zip(tasks, cold, hot):
            prompt_match = cold_answer["prompt_ids"] == hot_answer["prompt_ids"]
            output_match = cold_answer["output_ids"] == hot_answer["output_ids"]
            status = "PASS" if prompt_match and output_match else "FAIL"
            if status == "FAIL":
                failures.append(
                    f"{task['state']['record']['id']} q{task['question']['id']}"
                )
            task_results.append(
                {
                    "conversation_id": task["state"]["record"]["id"],
                    "question_id": task["question"]["id"],
                    "question": task["question"]["question"],
                    "status": status,
                    "prompt_match": prompt_match,
                    "output_match": output_match,
                    "cold": {
                        "text": cold_answer["text"],
                        "answer_content": cold_answer["answer_content"],
                        "reasoning_content": cold_answer["reasoning_content"],
                        "finish_reason": cold_answer["finish_reason"],
                        "stop_reason": cold_answer["stop_reason"],
                        "prompt_tokens": cold_answer["prompt_tokens"],
                        "output_tokens": cold_answer["output_tokens"],
                        "output_sha256": digest(cold_answer["output_ids"]),
                        "latency_ms": cold_answer["latency_ms"],
                        "stream_path": cold_answer["stream_path"],
                        "request": {
                            "context_messages": task["cold_messages"][:-1],
                            "prompt": task["cold_messages"][-1],
                            "request_options": {
                                key: value
                                for key, value in cold_answer["request_payload"].items()
                                if key != "messages"
                            },
                        },
                    },
                    "hot": {
                        "text": hot_answer["text"],
                        "answer_content": hot_answer["answer_content"],
                        "reasoning_content": hot_answer["reasoning_content"],
                        "finish_reason": hot_answer["finish_reason"],
                        "stop_reason": hot_answer["stop_reason"],
                        "prompt_tokens": hot_answer["prompt_tokens"],
                        "output_tokens": hot_answer["output_tokens"],
                        "output_sha256": digest(hot_answer["output_ids"]),
                        "latency_ms": hot_answer["latency_ms"],
                        "stream_path": hot_answer["stream_path"],
                        "request": {
                            "context_messages": task["hot_messages"][:-1],
                            "prompt": task["hot_messages"][-1],
                            "request_options": {
                                key: value
                                for key, value in hot_answer["request_payload"].items()
                                if key != "messages"
                            },
                        },
                    },
                }
            )
            if task["question"]["id"] == 0:
                main_answers[id(task["state"])] = (cold_answer, hot_answer)
        if round_id and hit_tokens <= 0:
            failures.append(
                "prefix-cache 未命中；确认服务以 --enable-prefix-caching 启动"
            )
        for state in states:
            answers = main_answers.get(id(state))
            if answers:
                cold_answer, hot_answer = answers
                question = state["record"]["rounds"][round_id]["questions"][0][
                    "question"
                ]
                state["cold_history"] += [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": cold_answer["text"]},
                ]
                state["hot_history"] += [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": hot_answer["text"]},
                ]
        result = {
            "round": round_id,
            "requests": len(tasks),
            "cache_query_tokens": query_tokens,
            "cache_hit_tokens": hit_tokens,
            "cache_hit_ratio": hit_tokens / query_tokens if query_tokens else 0.0,
            "status": "FAIL" if failures else "PASS",
            "failures": failures,
            "results": task_results,
        }
        round_results.append(result)
        logger.complete_round(result)
        for item in task_results:
            for kind in ("hot", "cold"):
                answer = item[kind]
                print(
                    f"round={round_id} {kind} "
                    f"conversation={item['conversation_id']} q={item['question_id']} "
                    f"answer_content={answer['answer_content']!r} "
                    f"reasoning_content={answer['reasoning_content']!r} "
                    f"finish_reason={answer['finish_reason']!r} "
                    f"stop_reason={answer['stop_reason']!r}"
                )
        print(
            f"round={round_id} requests={len(tasks)} hit={hit_tokens / query_tokens if query_tokens else 0.0:.1%} status={round_results[-1]['status']}"
        )
    return {"run_id": run_id, "rounds": round_results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CMT-Eval 多轮分叉 prefix-cache 冷热差分测试"
    )
    parser.add_argument("-e", metavar="PRESET", help="服务预设文件")
    parser.add_argument("--host", default="127.0.0.1", help="vLLM 服务地址")
    parser.add_argument("--port", help="服务端口；与 --model 一起使用时无需 -e")
    parser.add_argument("--model", help="模型名；与 --port 一起使用时无需 -e")
    parser.add_argument(
        "--dataset", type=Path, default=default_dataset_path(), help="多轮分叉数据集"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="每次回答的最大输出 token 数；默认不设上限",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="冷/热批次的最大并发请求数；默认每轮所有请求并发发送",
    )
    parser.add_argument(
        "--record-ids",
        help="仅运行指定题组 ID，多个 ID 用英文逗号分隔",
    )
    parser.add_argument(
        "--record-indices",
        help="仅运行指定题组的 0 起始索引，多个索引用英文逗号分隔",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        help="每个题组最多运行的轮数（从 round 0 开始）",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="开启模型 thinking（默认）",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="关闭模型 thinking",
    )
    parser.set_defaults(enable_thinking=True)
    args = parser.parse_args()
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens 必须大于 0")
    if args.concurrency is not None and args.concurrency < 1:
        parser.error("--concurrency 必须大于 0")
    if args.max_rounds is not None and args.max_rounds < 1:
        parser.error("--max-rounds 必须大于 0")
    if args.record_ids and args.record_indices:
        parser.error("--record-ids 和 --record-indices 不能同时使用")
    if args.record_ids:
        args.record_ids = [value.strip() for value in args.record_ids.split(",") if value.strip()]
        if not args.record_ids:
            parser.error("--record-ids 至少需要一个非空 ID")
    if args.record_indices:
        try:
            args.record_indices = [int(value.strip()) for value in args.record_indices.split(",")]
        except ValueError:
            parser.error("--record-indices 必须是用英文逗号分隔的整数")
        if not args.record_indices or any(value < 0 for value in args.record_indices):
            parser.error("--record-indices 必须包含非负整数")
    args.dataset = args.dataset.expanduser()
    if not args.dataset.is_file():
        parser.error(f"数据集不存在: {args.dataset}")
    return args


def main() -> int:
    args = parse_args()
    run_dir = (
        Path("./logs")
        / "cmt_eval_prefix_cache"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
    )
    logger = RunLogger(
        run_dir,
        {
            "host": args.host,
            "dataset": str(args.dataset),
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "record_ids": args.record_ids,
            "record_indices": args.record_indices,
            "max_rounds": args.max_rounds,
            "enable_thinking": args.enable_thinking,
        },
    )
    print(f"[日志] {run_dir}")
    print("[流] 每个 cold/hot 请求开始和完成时都会打印各自的 .log 输出路径")
    try:
        env = (
            source_env_with_preset(Path(__file__).resolve().parent, args.e)
            if not (args.port and args.model)
            else {}
        )
        port, model = (
            args.port or env.get("USER_VLLM_PORT"),
            args.model or env.get("USER_VLLM_MODEL"),
        )
        if not port or not model:
            raise TestFailure("请提供 -e，或同时提供 --port 和 --model")
        base_url = f"http://{args.host}:{port}"
        logger.event("server_config", {"base_url": base_url, "model": model})
        request(base_url + "/v1/models")
        records = load_dataset(args.dataset)
        if args.record_ids:
            records_by_id = {record["id"]: record for record in records}
            unknown_ids = [record_id for record_id in args.record_ids if record_id not in records_by_id]
            if unknown_ids:
                raise TestFailure(f"未知题组 ID: {', '.join(unknown_ids)}")
            records = [records_by_id[record_id] for record_id in args.record_ids]
        elif args.record_indices:
            invalid_indices = [index for index in args.record_indices if index >= len(records)]
            if invalid_indices:
                raise TestFailure(f"题组索引越界（共有 {len(records)} 个）: {invalid_indices}")
            records = [records[index] for index in args.record_indices]
        logger.event(
            "dataset_loaded",
            {"records": len(records), "record_ids": [record["id"] for record in records]},
        )
        print(
            f"[CMT-Eval prefix-cache] model={model} records={len(records)} dataset={args.dataset}"
        )
        result = run_test(
            base_url,
            model,
            records,
            args.max_tokens,
            args.concurrency,
            args.max_rounds,
            args.enable_thinking,
            logger,
        )
        failed = sum(round_["status"] == "FAIL" for round_ in result["rounds"])
        logger.finish("FAIL" if failed else "PASS", f"failed_rounds={failed}")
        print(f"report={logger.report_path} failed_rounds={failed}")
        return 1 if failed else 0
    except TestFailure as exc:
        logger.finish("ERROR", f"{type(exc).__name__}: {exc}")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        logger.finish("ERROR", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())
