"""Unit tests for eval_qda_kl_vllm.py KL divergence and guardrail math."""

from __future__ import annotations

import math
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import eval_qda_kl_vllm as qda


def test_identical_distributions_have_zero_kl() -> None:
    p = {10: math.log(0.7), 20: math.log(0.25), 30: math.log(0.04)}
    m = qda.compute_token_divergence(p, p, actual_token_id=10)
    assert m["fwd_kl"] < 1e-12
    assert m["rev_kl"] < 1e-12
    assert m["jsd"] < 1e-12
    assert m["top1_agree"] == 1.0
    assert m["top5_agree"] == 1.0
    assert abs(m["nll_p"] - m["nll_q"]) < 1e-12
    assert m["abs_lp_err"] < 1e-12


def test_known_kl_and_tail_handling() -> None:
    p = {1: math.log(0.8), 2: math.log(0.19)}
    q = {1: math.log(0.6), 2: math.log(0.39)}
    # Tail = 0.01 for both -> exact KL = 0.8*ln(0.8/0.6) + 0.19*ln(0.19/0.39)
    expected = 0.8 * math.log(0.8 / 0.6) + 0.19 * math.log(0.19 / 0.39)
    m = qda.compute_token_divergence(p, q, actual_token_id=1)
    assert abs(m["fwd_kl"] - expected) < 1e-6
    assert m["rev_kl"] > 0.0
    assert m["jsd"] > 0.0
    assert m["top1_agree"] == 1.0


def test_disjoint_topk_stays_finite_and_non_negative() -> None:
    p = {1: math.log(0.6), 2: math.log(0.35)}
    q = {2: math.log(0.5), 3: math.log(0.45)}
    m = qda.compute_token_divergence(p, q, actual_token_id=2)
    assert math.isfinite(m["fwd_kl"]) and m["fwd_kl"] > 0.0
    assert math.isfinite(m["rev_kl"]) and m["rev_kl"] > 0.0
    assert m["top1_agree"] == 0.0
    assert m["top5_agree"] == 0.0


def test_ngram_repetition_and_prefix() -> None:
    assert qda.ngram_repetition_ratio([1, 2, 3, 4, 5, 6, 7, 8], 4) == 0.0
    assert qda.ngram_repetition_ratio([1, 2, 3, 4] * 4, 4) > 0.6
    assert qda.common_prefix_len([1, 2, 3, 4], [1, 2, 9, 4]) == 2
