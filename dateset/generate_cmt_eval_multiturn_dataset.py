#!/usr/bin/env python3
"""Build the fixed CMT-Eval multi-turn branch dataset."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


SELECTIONS = (
    ("standard_data.json", (1, 3)),
    ("long-text_data.json", (2, 3)),
    ("hard_data.json", (1, 13)),
)


def dataset_root() -> Path:
    return Path(__file__).resolve().parent


def build(source_dir: Path) -> list[dict[str, Any]]:
    records = []
    for filename, indexes in SELECTIONS:
        source_path = source_dir / filename
        try:
            source = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"无法读取 {source_path}: {exc}") from exc
        for source_index in indexes:
            try:
                item = source[source_index]
                source_turns = item["会话内容"]
            except (IndexError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"样本格式异常: {source_path}[{source_index}]"
                ) from exc
            if len(source_turns) % 2 == 0:
                raise RuntimeError(f"问题数必须为奇数: {source_path}[{source_index}]")
            flat_questions = []
            for index, turn in enumerate(source_turns):
                question = turn.get("用户query")
                if not isinstance(question, str) or not question.strip():
                    raise RuntimeError(
                        f"问题为空: {source_path}[{source_index}] question {index}"
                    )
                flat_questions.append(
                    {
                        "question": question,
                        "dataset_reference_answer": turn.get("预设回复"),
                        "source_question_index": index,
                        "source_turn": turn.get("轮次", index + 1),
                        "source_speech_act": turn.get("言语行为"),
                    }
                )
            rounds = []
            cursor = 0
            round_id = 0
            while cursor < len(flat_questions):
                count = 1 if round_id == 0 else 2
                questions = [
                    question | {"id": branch_id}
                    for branch_id, question in enumerate(
                        flat_questions[cursor : cursor + count]
                    )
                ]
                rounds.append(
                    {
                        "round": round_id,
                        "history_main_path": [
                            {"round": prior_round, "id": 0}
                            for prior_round in range(round_id)
                        ],
                        "questions": questions,
                    }
                )
                cursor += count
                round_id += 1
            source_id = item.get("origin_id", item.get("ID", source_index))
            records.append(
                {
                    "id": f"{Path(filename).stem}:{source_id}:{source_index}",
                    "source": {
                        "file": filename,
                        "index": source_index,
                        "id": source_id,
                    },
                    "evaluation_capability": item.get("评测能力"),
                    "user_role": item.get("用户角色"),
                    "history_policy": "previous_rounds_main_path_only",
                    "rounds": rounds,
                }
            )
    return records


def main() -> int:
    root = dataset_root()
    cmt_eval = root / "CMT-Eval"
    source_dir = cmt_eval / "data" / "dialogue_data"
    if not source_dir.is_dir():
        print(f"[缺失] 未找到 CMT-Eval 数据目录: {source_dir}", file=sys.stderr)
        print("[下一步] 将执行以下命令获取数据源：", file=sys.stderr)
        print("git clone https://github.com/hejaida/CMT-Eval.git", file=sys.stderr)
        return 2
    try:
        records = build(source_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    output = root / "cmt_eval_multiturn_6.json"
    output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[数据集] 已写入 {output}（{len(records)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
