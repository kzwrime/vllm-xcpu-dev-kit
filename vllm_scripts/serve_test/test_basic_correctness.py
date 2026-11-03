#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Basic correctness checks against a running OpenAI-compatible server.

Model-agnostic greedy checks (thinking disabled where the chat template
supports it):

* arithmetic / factual answers (short context);
* needle retrieval in a long context. For models with sparse / compressed
  attention (e.g. DeepSeek-V4.1) the context holds more compressed positions
  than the indexer top-k, so the indexer and its candidate blocks decide what
  gets attended to; for dense models it is a plain long-context recall check;
* a long generation checked for degenerate repetition;
* a concurrent batch whose greedy answers must match the sequential ones
  (request isolation under batching).

Usage: test_basic_correctness.py --model NAME [--base-url URL] [--needle-sentences N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HTTP_TIMEOUT_S = 1800.0


def chat(base_url: str, model: str, prompt: str, max_tokens: int) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
        message = json.load(response)["choices"][0]["message"]
    return ((message.get("reasoning") or "") + (message.get("content") or "")).strip()


def needle_prompt(num_sentences: int, seed: int) -> tuple[str, str]:
    rng = random.Random(seed)
    code = str(rng.randint(10000, 99999))
    filler = [
        f"Record {i}: the {rng.choice(['red', 'green', 'blue', 'amber'])} "
        f"{rng.choice(['valve', 'sensor', 'relay', 'pump'])} in bay "
        f"{rng.randint(1, 90)} was inspected and found nominal."
        for i in range(num_sentences)
    ]
    filler.insert(num_sentences // 3, f"Important: the vault access code is {code}.")
    prompt = (
        " ".join(filler)
        + "\n\nWhat is the vault access code mentioned above? Reply with the code only."
    )
    return prompt, code


def repeated_ngram_ratio(text: str, n: int = 8) -> float:
    words = text.split()
    grams = [tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:14800")
    parser.add_argument("--model", required=True)
    parser.add_argument("--needle-sentences", type=int, default=400)
    args = parser.parse_args()

    cases = [
        ("arithmetic", "What is 17 * 23? Reply with the number only.", "391", 16),
        ("factual", "What is the capital of France? Reply with one word.", "paris", 16),
    ]
    needle, code = needle_prompt(args.needle_sentences, seed=7)
    cases.append(("needle", needle, code, 24))

    failures = []
    answers = {}
    for name, prompt, expected, max_tokens in cases:
        answer = chat(args.base_url, args.model, prompt, max_tokens)
        answers[name] = answer
        ok = expected in answer.lower()
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {answer!r}")
        if not ok:
            failures.append(name)

    essay = chat(
        args.base_url,
        args.model,
        "Write a short paragraph about why the sky is blue.",
        160,
    )
    ratio = repeated_ngram_ratio(essay)
    ok = ratio < 0.3
    print(f"[{'PASS' if ok else 'FAIL'}] repetition: ratio={ratio:.2f} {essay[:200]!r}")
    if not ok:
        failures.append("repetition")

    batch = [(name, prompt, max_tokens) for name, prompt, _, max_tokens in cases[:2]] * 2
    with ThreadPoolExecutor(len(batch)) as pool:
        results = list(
            pool.map(lambda c: chat(args.base_url, args.model, c[1], c[2]), batch)
        )
    for (name, _, _), result in zip(batch, results):
        ok = result == answers[name]
        print(f"[{'PASS' if ok else 'FAIL'}] batch/{name}: {result!r}")
        if not ok:
            failures.append(f"batch/{name}")

    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
