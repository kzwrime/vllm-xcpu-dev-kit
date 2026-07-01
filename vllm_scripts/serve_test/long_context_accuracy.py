# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 长上下文准确性测试脚本
#
# 数据准备:
#   python serve_test/long_context_accuracy.py --prepare-only
#
# 运行测试:
#   python serve_test/long_context_accuracy.py -e ./presets/serial/xxx.sh
#   PRESET=serial/xxx python serve_test/long_context_accuracy.py
#
# 说明:
#   本脚本下载公开 SQuAD v1.1 数据集，并用真实 Wikipedia QA 段落拼接
#   0.5k/4k/8k/16k/32k/64k 固定长上下文用例。每个用例包含标准答案，
#   方便人工检查模型输出的逻辑性和准确性。

import argparse
import json
import os
import re
import shlex
import string
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pathlib import Path
from typing import Any


SQUAD_DEV_URL = (
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
)
DEFAULT_LENGTHS = [512, 4096, 8192, 16384, 32768, 65536]
SYSTEM_PROMPT = (
    "You are a precise long-context QA evaluator. Answer only from the supplied "
    "context. If the context is insufficient, say so explicitly."
)
USER_HEADER = """请阅读下面的长上下文材料，并只依据材料回答最后的问题。

要求:
1. 先用 2-4 句话说明你定位答案的依据。
2. 最后一行使用格式: 最终答案: <答案>
3. 最终答案尽量保留原文中的实体名、数字或短语。

长上下文开始:
"""
USER_FOOTER = """

长上下文结束。

问题: {question}
"""


@dataclass
class TokenCounter:
    tokenizer: Any | None = None
    name: str = "char_approx"

    @classmethod
    def create(cls, tokenizer_name: str | None, trust_remote_code: bool) -> "TokenCounter":
        if not tokenizer_name:
            return cls()
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=trust_remote_code,
                local_files_only=os.path.exists(tokenizer_name),
            )
            return cls(tokenizer=tokenizer, name=tokenizer_name)
        except Exception as exc:
            print(
                f"[警告] 加载 tokenizer 失败，使用字符近似计数: {exc}",
                file=sys.stderr,
            )
            return cls()

    def count(self, text: str) -> int:
        if self.tokenizer is None:
            return max(1, (len(text) + 3) // 4)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def trim(self, text: str, token_budget: int) -> str:
        if token_budget <= 0:
            return ""
        if self.tokenizer is None:
            return text[: token_budget * 4]
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= token_budget:
            return text
        return self.tokenizer.decode(token_ids[:token_budget])


def source_env_with_preset(script_dir: str, preset_file: str | None = None) -> dict[str, str]:
    vllm_scripts_dir = os.path.abspath(os.path.join(script_dir, ".."))
    common_sh = os.path.join(vllm_scripts_dir, "common.sh")

    bash_command = f"""
set -e
source {shlex.quote(common_sh)}
load_env_file {shlex.quote(os.path.join(vllm_scripts_dir, "env.sh"))}
"""

    if preset_file:
        bash_command += f"""
load_preset_file {shlex.quote(preset_file)}
"""
    else:
        bash_command += f"""
load_user_config {shlex.quote(vllm_scripts_dir)}
"""

    bash_command += """
env
"""

    result = subprocess.run(
        ["bash", "-c", bash_command],
        capture_output=True,
        text=True,
        env=os.environ,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "加载环境变量失败")

    env_vars: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env_vars[key] = value
    return env_vars


def parse_length(value: str) -> int:
    text = value.strip().lower()
    multiplier = 1
    if text.endswith("k"):
        multiplier = 1024
        text = text[:-1]
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"非法长度: {value}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"非法长度: {value}")
    tokens = parsed * multiplier
    if tokens != tokens.to_integral_value():
        raise argparse.ArgumentTypeError(f"长度必须能换算为整数 token: {value}")
    return int(tokens)


def parse_lengths(value: str) -> list[int]:
    return [parse_length(item) for item in value.split(",") if item.strip()]


def format_case_id(target_tokens: int) -> str:
    if target_tokens % 1024 == 0:
        return f"{target_tokens // 1024}k"
    token_k = Decimal(target_tokens) / Decimal(1024)
    text = format(token_k.normalize(), "f").rstrip("0").rstrip(".")
    return f"{text}k"


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    print(f"[数据] 下载: {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        tmp_path.write_bytes(response.read())
    tmp_path.replace(output_path)
    print(f"[数据] 已保存: {output_path}")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_squad_records(raw_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for article in raw.get("data", []):
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            context = normalize_space(paragraph.get("context", ""))
            if len(context) < 240:
                continue
            for qa in paragraph.get("qas", []):
                answers = sorted(
                    {
                        normalize_space(answer.get("text", ""))
                        for answer in qa.get("answers", [])
                        if normalize_space(answer.get("text", ""))
                    }
                )
                question = normalize_space(qa.get("question", ""))
                if not question or not answers:
                    continue
                if not any(answer in context for answer in answers):
                    continue
                records.append(
                    {
                        "id": qa.get("id", ""),
                        "title": title,
                        "context": context,
                        "question": question,
                        "answers": answers,
                    }
                )
    if not records:
        raise RuntimeError(f"未能从 {raw_path} 解析出可用 QA 记录")
    return records


def unique_filler_contexts(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    fillers: list[dict[str, str]] = []
    for record in records:
        context = record["context"]
        if context in seen:
            continue
        seen.add(context)
        fillers.append({"title": record["title"], "context": context})
    return fillers


def format_doc(index: int, title: str, context: str, marker: str = "DOC") -> str:
    return f"\n\n[{marker} {index:05d}] Title: {title}\n{context}\n"


def fill_to_budget(
    fillers: list[dict[str, str]],
    start_index: int,
    token_budget: int,
    counter: TokenCounter,
) -> tuple[str, int]:
    pieces: list[str] = []
    used_tokens = 0
    filler_count = len(fillers)
    cursor = start_index

    while used_tokens < token_budget and filler_count:
        filler = fillers[cursor % filler_count]
        doc = format_doc(cursor, filler["title"], filler["context"])
        doc_tokens = counter.count(doc)
        remaining = token_budget - used_tokens
        if doc_tokens > remaining:
            pieces.append(counter.trim(doc, remaining))
            used_tokens += remaining
            break
        pieces.append(doc)
        used_tokens += doc_tokens
        cursor += 1

    return "".join(pieces), cursor


def choose_record(records: list[dict[str, Any]], case_index: int) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if 3 <= len(record["answers"][0]) <= 80 and len(record["context"]) >= 320
    ]
    if not candidates:
        candidates = records
    return candidates[(case_index * 997 + 211) % len(candidates)]


def build_case(
    records: list[dict[str, Any]],
    fillers: list[dict[str, str]],
    target_tokens: int,
    case_index: int,
    counter: TokenCounter,
    answer_depth: float,
) -> dict[str, Any]:
    record = choose_record(records, case_index)
    footer = USER_FOOTER.format(question=record["question"])
    fixed_tokens = counter.count(SYSTEM_PROMPT) + counter.count(USER_HEADER + footer)
    context_budget = max(512, target_tokens - fixed_tokens - 32)

    answer_doc = format_doc(
        case_index,
        record["title"],
        record["context"],
        marker="ANSWER_DOC",
    )
    answer_doc_tokens = counter.count(answer_doc)
    filler_budget = max(0, context_budget - answer_doc_tokens)
    before_budget = int(filler_budget * answer_depth)
    after_budget = max(0, filler_budget - before_budget)

    before_text, cursor = fill_to_budget(
        fillers,
        start_index=case_index * 137,
        token_budget=before_budget,
        counter=counter,
    )
    after_text, _ = fill_to_budget(
        fillers,
        start_index=cursor + 17,
        token_budget=after_budget,
        counter=counter,
    )
    context = before_text + answer_doc + after_text
    user_prompt = USER_HEADER + context + footer
    actual_tokens = counter.count(SYSTEM_PROMPT) + counter.count(user_prompt)
    answer_offset_tokens = counter.count(SYSTEM_PROMPT) + counter.count(
        USER_HEADER + before_text
    )

    return {
        "case_id": format_case_id(target_tokens),
        "target_input_tokens": target_tokens,
        "estimated_input_tokens": actual_tokens,
        "answer_depth": round(answer_offset_tokens / max(actual_tokens, 1), 4),
        "dataset": "SQuAD v1.1 dev",
        "source_title": record["title"],
        "source_id": record["id"],
        "question": record["question"],
        "answers": record["answers"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }


def prepare_cases(
    data_dir: Path,
    lengths: list[int],
    tokenizer_name: str | None,
    trust_remote_code: bool,
    force: bool,
    answer_depth: float,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "squad_dev_v1.1.json"
    cases_path = data_dir / "long_context_squad_cases.jsonl"

    download_file(SQUAD_DEV_URL, raw_path)
    if cases_path.exists() and not force:
        wanted = {format_case_id(length) for length in lengths}
        existing: set[str] = set()
        with cases_path.open("r", encoding="utf-8") as in_file:
            for line in in_file:
                if line.strip():
                    existing.add(json.loads(line)["case_id"])
        missing = sorted(wanted - existing)
        if not missing:
            return cases_path
        print(
            "[数据] 已有用例文件缺少长度 "
            f"{', '.join(missing)}，重新生成: {cases_path}"
        )

    counter = TokenCounter.create(tokenizer_name, trust_remote_code)
    records = load_squad_records(raw_path)
    fillers = unique_filler_contexts(records)

    print(f"[数据] QA 记录数: {len(records)}")
    print(f"[数据] 填充段落数: {len(fillers)}")
    print(f"[数据] token 计数器: {counter.name}")

    with cases_path.open("w", encoding="utf-8") as out:
        for case_index, target_tokens in enumerate(lengths):
            case = build_case(
                records,
                fillers,
                target_tokens,
                case_index,
                counter,
                answer_depth,
            )
            out.write(json.dumps(case, ensure_ascii=False) + "\n")
            print(
                "[数据] 生成用例 "
                f"{case['case_id']}: estimated_input_tokens="
                f"{case['estimated_input_tokens']}, answer_depth="
                f"{case['answer_depth']}, expected={case['answers'][:3]}"
            )

    print(f"[数据] 用例已保存: {cases_path}")
    return cases_path


def load_cases(cases_path: Path, lengths: list[int]) -> list[dict[str, Any]]:
    wanted = {format_case_id(length) for length in lengths}
    cases: list[dict[str, Any]] = []
    with cases_path.open("r", encoding="utf-8") as in_file:
        for line in in_file:
            if not line.strip():
                continue
            case = json.loads(line)
            if case["case_id"] in wanted:
                cases.append(case)
    missing = sorted(wanted - {case["case_id"] for case in cases})
    if missing:
        raise RuntimeError(f"测试数据缺少长度: {', '.join(missing)}")
    return cases


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def answer_hit(output: str, answers: list[str]) -> bool:
    normalized_output = normalize_answer(output)
    return any(normalize_answer(answer) in normalized_output for answer in answers)


def get_stream_value(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    value = getattr(obj, name, None)
    if value is not None:
        return value
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(name)
    return None


def run_case(
    client: Any,
    model_name: str,
    case: dict[str, Any],
    output_dir: Path,
    max_tokens: int,
    temperature: float,
    echo_stream: bool,
) -> dict[str, Any]:
    case_id = case["case_id"]
    output_path = output_dir / f"{case_id}_output.txt"
    meta_path = output_dir / f"{case_id}_meta.json"
    print(
        f"[测试] {case_id}: estimated_input_tokens="
        f"{case['estimated_input_tokens']}, answer_depth={case['answer_depth']}",
        flush=True,
    )
    print(f"[测试] {case_id}: question={case['question']}", flush=True)
    print(f"[测试] {case_id}: expected={case['answers']}", flush=True)
    print(f"[测试] {case_id}: output_file={output_path}", flush=True)
    print(f"[测试] {case_id}: meta_file={meta_path}", flush=True)

    start = time.time()
    output_text = ""
    reasoning_text = ""
    error = None
    interrupted = False
    chunk_count = 0
    content_chunk_count = 0
    reasoning_chunk_count = 0
    first_chunk_seconds = None
    first_text_seconds = None
    finish_reasons: list[str] = []
    wrote_reasoning_header = False
    wrote_content_header = False
    try:
        if echo_stream:
            print(f"[输出] {case_id}: begin", flush=True)
        else:
            print(f"[输出] {case_id}: 流式写入文件，实时回显关闭", flush=True)
        response = client.chat.completions.create(
            model=model_name,
            messages=case["messages"],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        with output_path.open("w", encoding="utf-8") as out:
            for chunk in response:
                chunk_count += 1
                if first_chunk_seconds is None:
                    first_chunk_seconds = time.time() - start
                choices = get_stream_value(chunk, "choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = get_stream_value(choice, "finish_reason")
                if finish_reason:
                    finish_reasons.append(str(finish_reason))
                delta = get_stream_value(choice, "delta")
                content = get_stream_value(delta, "content") or ""
                reasoning = (
                    get_stream_value(delta, "reasoning_content")
                    or get_stream_value(delta, "reasoning")
                    or ""
                )
                if (content or reasoning) and first_text_seconds is None:
                    first_text_seconds = time.time() - start
                if reasoning:
                    reasoning_chunk_count += 1
                    reasoning_text += reasoning
                    if not wrote_reasoning_header:
                        out.write("[reasoning_content]\n")
                        if echo_stream:
                            print("[reasoning_content]", flush=True)
                        wrote_reasoning_header = True
                    if echo_stream:
                        print(reasoning, end="", flush=True)
                    out.write(reasoning)
                    out.flush()
                if content:
                    content_chunk_count += 1
                    output_text += content
                    if wrote_reasoning_header and not wrote_content_header:
                        out.write("\n\n[content]\n")
                        if echo_stream:
                            print("\n\n[content]", flush=True)
                        wrote_content_header = True
                    if echo_stream:
                        print(content, end="", flush=True)
                    out.write(content)
                    out.flush()
        if echo_stream:
            print(f"\n[输出] {case_id}: end", flush=True)
        else:
            print(f"[输出] {case_id}: 写入完成", flush=True)
    except Exception as exc:
        error = str(exc)
        if output_path.exists() or output_text:
            with output_path.open("a", encoding="utf-8") as out:
                out.write(f"\n\n[请求失败]\n{error}\n")
        else:
            output_path.write_text(f"[请求失败]\n{error}\n", encoding="utf-8")
    except KeyboardInterrupt:
        interrupted = True
        error = "KeyboardInterrupt"
        if output_path.exists() or output_text:
            with output_path.open("a", encoding="utf-8") as out:
                out.write("\n\n[用户中断]\n")
        else:
            output_path.write_text("[用户中断]\n", encoding="utf-8")
        print(f"\n[中断] {case_id}: 已保留当前输出文件: {output_path}", flush=True)

    elapsed = time.time() - start
    content_hit = False if error else answer_hit(output_text, case["answers"])
    reasoning_hit = False if error else answer_hit(reasoning_text, case["answers"])
    hit = content_hit
    metadata = {
        "case_id": case_id,
        "model": model_name,
        "estimated_input_tokens": case["estimated_input_tokens"],
        "target_input_tokens": case["target_input_tokens"],
        "answer_depth": case["answer_depth"],
        "question": case["question"],
        "answers": case["answers"],
        "source_title": case["source_title"],
        "source_id": case["source_id"],
        "elapsed_seconds": round(elapsed, 3),
        "first_chunk_seconds": (
            round(first_chunk_seconds, 3)
            if first_chunk_seconds is not None
            else None
        ),
        "first_text_seconds": (
            round(first_text_seconds, 3)
            if first_text_seconds is not None
            else None
        ),
        "contains_expected_answer": hit,
        "contains_expected_answer_in_content": content_hit,
        "contains_expected_answer_in_reasoning": reasoning_hit,
        "error": error,
        "interrupted": interrupted,
        "streaming_request": True,
        "stream_echo": echo_stream,
        "stream_chunk_count": chunk_count,
        "content_chunk_count": content_chunk_count,
        "reasoning_chunk_count": reasoning_chunk_count,
        "content_chars": len(output_text),
        "reasoning_chars": len(reasoning_text),
        "finish_reasons": finish_reasons,
        "output_file": str(output_path),
    }
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    status = (
        "INTERRUPTED"
        if interrupted
        else ("ERROR" if error else ("HIT" if hit else "MISS"))
    )
    print(
        f"[测试] {case_id}: {status}, elapsed={elapsed:.2f}s, "
        f"output_file={output_path}",
        flush=True,
    )
    return metadata


def parse_args() -> argparse.Namespace:
    default_lengths = ",".join(format_case_id(length) for length in DEFAULT_LENGTHS)
    parser = argparse.ArgumentParser(
        description="使用真实 QA 数据测试 vLLM 长上下文准确性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 只下载并生成固定长上下文测试数据
  python serve_test/long_context_accuracy.py --prepare-only

  # 先用 0.5k 做基础验证，再测试 4k 到 64k
  python serve_test/long_context_accuracy.py -e ./presets/serial/xxx.sh

  # 只测试 16k/32k，并把答案文档放在上下文 90% 附近
  python serve_test/long_context_accuracy.py --lengths 16k,32k --answer-depth 0.9

  # 请求始终使用流式；下面选项只开启终端实时回显
  python serve_test/long_context_accuracy.py --stream
        """,
    )
    parser.add_argument("-e", metavar="预设文件", help="指定预设文件路径")
    parser.add_argument(
        "--lengths",
        default=default_lengths,
        help=f"逗号分隔的输入长度，例如 {default_lengths}",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="数据缓存目录，默认 serve_test/long_context_data",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="结果输出目录，默认 serve_test/long_context_results/<timestamp>",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只下载并生成本地测试数据，不请求 vLLM 服务",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="重新生成 long_context_squad_cases.jsonl",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="用于估算 token 的 tokenizer 路径或名称，默认使用 USER_VLLM_MODEL",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="加载 tokenizer 时允许 trust_remote_code=True",
    )
    parser.add_argument(
        "--answer-depth",
        type=float,
        default=0.85,
        help="答案文档在上下文中的大致位置，0.0=开头，1.0=结尾，默认 0.85",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="每个请求最大输出 token 数，默认 32768",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度，默认 0",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="OpenAI 客户端超时时间，默认 1800 秒",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="开启终端实时回显；请求始终流式写入文件，默认不回显",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lengths = parse_lengths(args.lengths)
    if not 0.0 <= args.answer_depth <= 1.0:
        raise SystemExit("[错误] --answer-depth 必须在 0.0 到 1.0 之间")
    if args.max_tokens <= 0:
        raise SystemExit("[错误] --max-tokens 必须大于 0")

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir / "long_context_data"

    env_vars: dict[str, str] = {}
    if not args.prepare_only or args.tokenizer is None:
        try:
            env_vars = source_env_with_preset(str(script_dir), preset_file=args.e)
        except Exception as exc:
            if not args.prepare_only:
                raise SystemExit(f"[错误] 加载环境变量失败: {exc}") from exc
            print(f"[警告] 加载环境变量失败，将使用字符近似计数: {exc}")

    model_name = env_vars.get("USER_VLLM_MODEL", "")
    tokenizer_name = args.tokenizer or model_name or None
    cases_path = prepare_cases(
        data_dir=data_dir,
        lengths=lengths,
        tokenizer_name=tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        force=args.force_prepare,
        answer_depth=args.answer_depth,
    )

    if args.prepare_only:
        print("[完成] 数据准备完成，未请求 vLLM 服务。")
        return

    port = env_vars.get("USER_VLLM_PORT", "8000")
    if not model_name:
        raise SystemExit("[错误] USER_VLLM_MODEL 未设置")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.results_dir)
        if args.results_dir
        else script_dir / "long_context_results" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(cases_path, lengths)
    try:
        from openai import OpenAI
    except Exception as exc:
        raise SystemExit(f"[错误] 无法导入 openai 包: {exc}") from exc

    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
        timeout=args.timeout,
    )

    print("[测试] 开始长上下文准确性测试")
    print(f"[测试] 模型: {model_name}")
    print(f"[测试] 端口: {port}")
    print(f"[测试] 用例文件: {cases_path}")
    print(f"[测试] 结果目录: {output_dir}")
    echo_stream = args.stream
    print("[测试] 流式请求: 开启，流式写入文件")
    print(f"[测试] 实时回显: {'开启' if echo_stream else '关闭'}")

    results = []
    for case in cases:
        result = run_case(
            client=client,
            model_name=model_name,
            case=case,
            output_dir=output_dir,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            echo_stream=echo_stream,
        )
        results.append(result)
        if result.get("interrupted"):
            print("[中断] 收到 Ctrl-C，停止后续用例并写入 summary。")
            break

    summary_path = output_dir / "summary.json"
    summary = {
        "model": model_name,
        "port": port,
        "cases_path": str(cases_path),
        "results": results,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    hit_count = sum(1 for result in results if result["contains_expected_answer"])
    error_count = sum(1 for result in results if result["error"])
    print(
        f"[完成] {hit_count}/{len(results)} 个输出包含标准答案，"
        f"{error_count} 个请求失败。summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
