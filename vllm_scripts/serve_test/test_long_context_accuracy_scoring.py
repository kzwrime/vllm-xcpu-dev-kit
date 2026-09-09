"""Regressions for the long-context QA evaluator, without a running server."""
from long_context_accuracy import (
    TokenCounter,
    answer_hit,
    build_case,
    parse_lengths,
    validate_answer_format,
)


VALID_KUBIAK_OUTPUT = """依据:
材料直接说明了教练人选。相关原文短语可以直接回答问题。
最终答案: Gary Kubiak"""


def test_scores_final_answer_instead_of_rejected_evidence():
    assert not answer_hit("材料提到 Kubiak。\n最终答案: John Fox", ["Kubiak"])
    assert not answer_hit("Kubiak appears in the context.\n**最终答案**: John Fox", ["Kubiak"])
    assert answer_hit("最终答案: Gary Kubiak", ["Gary Kubiak", "Kubiak"])


def test_answer_boundaries_and_empty_normalization():
    assert not answer_hit("最终答案: 1999", ["199"])
    assert not answer_hit("最终答案: northeast", ["east"])
    assert not answer_hit("最终答案: anything", ["", "the", "."])
    assert answer_hit("Final answer: The Franklin Institute.", ["Franklin Institute"])


def test_only_requires_final_answer_format():
    assert validate_answer_format(VALID_KUBIAK_OUTPUT) == (True, [])
    assert validate_answer_format("人工可读的一句依据。\n最终答案: Gary Kubiak") == (True, [])
    assert validate_answer_format("- 依据一\n- 依据二\n最终答案: Gary Kubiak") == (True, [])
    invalid_outputs = [
        "材料直接说明了答案。",
        "依据:\n第一句。\n最终答案:",
        "依据:\n第一句。第二句。\n最终答案: Gary Kubiak\n附加内容",
        "最终答案: Kubiak\n最终答案: Gary Kubiak",
    ]
    for output in invalid_outputs:
        valid, errors = validate_answer_format(output)
        assert not valid
        assert errors


def test_accepts_arbitrary_unique_lengths():
    assert parse_lengths("2047,2k,8.5k") == [2047, 2048, 8704]


def test_counts_chat_template_batch_encoding():
    class BatchEncodingLike:
        def get(self, key):
            return [11, 12, 13] if key == "input_ids" else None

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return BatchEncodingLike()

    counter = TokenCounter(tokenizer=Tokenizer(), name="test")
    assert counter.count_messages([{"role": "user", "content": "hello"}]) == 3


def test_half_k_does_not_force_512_context_tokens():
    record = dict(id="qa", title="Example", context="Facts about Kubiak. " * 20,
                  question="Who?", answers=["Kubiak"])
    case = build_case([record], [record], 512, 0, TokenCounter(), 0.85)
    assert case["estimated_input_tokens"] <= 512
    assert case["messages"][1]["content"].count(record["context"]) == 1


def test_truncated_and_unfinished_streams_are_not_success(tmp_path):
    from types import SimpleNamespace
    from long_context_accuracy import run_case

    record = dict(id="qa", title="Example", context="Facts about Kubiak. " * 20,
                  question="Who?", answers=["Kubiak"])
    case = build_case([record], [record], 512, 0, TokenCounter(), 0.85)
    for finish in ("stop", "length", None):
        def create(**kwargs):
            assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
            assert kwargs["stream_options"]["include_usage"] is True
            return [{"choices": [{"delta": {"content": VALID_KUBIAK_OUTPUT},
                                   "finish_reason": finish}]}]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = run_case(client, "test", case, tmp_path, 256, 0.0, False, False)
        assert result["contains_expected_answer"] == (finish == "stop")
