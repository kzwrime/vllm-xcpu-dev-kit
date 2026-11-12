"""Sequential greedy probe for token-level comparison between two servers.

Sends the multi-stream prompts one at a time (temperature 0, 16 tokens,
thinking off) and records every step's token and top-5 logprobs. One request
at a time avoids the batch-composition differences of the concurrent
multi-test, which change greedy outputs between otherwise identical servers.

    python serve_test/test_greedy_logprobs.py dump http://127.0.0.1:14800 <model> a.json
    python serve_test/test_greedy_logprobs.py compare a.json b.json   # exit 1 on mismatch
"""

import json
import sys
import urllib.request

PROMPTS = [
    "请用一段话简单介绍一下量子计算。",
    "写一首关于春天的七言绝句。",
    "欧盟有多少个国家，详细展开论述欧盟现状。",
    "解释一下相对论的核心思想。",
    "给出三个提高编程效率的建议。",
]


def dump(base: str, model: str, out: str) -> int:
    results = []
    for prompt in PROMPTS:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 16,
            "logprobs": True,
            "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            choice = json.load(response)["choices"][0]
        steps = [
            {
                "token": item["token"],
                "logprob": item["logprob"],
                "top": [(t["token"], t["logprob"]) for t in item["top_logprobs"]],
            }
            for item in choice["logprobs"]["content"]
        ]
        text = choice["message"]["content"]
        results.append({"prompt": prompt, "text": text, "steps": steps})
        print(prompt, "->", repr(text), flush=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)
    return 0


def compare(path_a: str, path_b: str) -> int:
    with open(path_a, encoding="utf-8") as handle:
        a = json.load(handle)
    with open(path_b, encoding="utf-8") as handle:
        b = json.load(handle)
    mismatch = False
    for ra, rb in zip(a, b):
        same = ra["text"] == rb["text"]
        mismatch |= not same
        print(f"{'SAME' if same else 'DIFF'} {ra['prompt']}")
        print(f"   A: {ra['text']!r}\n   B: {rb['text']!r}")
        max_delta = 0.0
        for i, (sa, sb) in enumerate(zip(ra["steps"], rb["steps"])):
            if sa["token"] != sb["token"]:
                print(
                    f"   first divergence at step {i}: A top={sa['top'][:3]} B top={sb['top'][:3]}"
                )
                break
            max_delta = max(max_delta, abs(sa["logprob"] - sb["logprob"]))
            top_b = dict(sb["top"])
            for token, logprob in sa["top"]:
                if token in top_b:
                    max_delta = max(max_delta, abs(logprob - top_b[token]))
        else:
            print(f"   max |delta logprob| over matched steps / top-5: {max_delta:.4f}")
    print("RESULT:", "MISMATCH" if mismatch else "IDENTICAL")
    return 1 if mismatch else 0


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == "dump":
        return dump(sys.argv[2], sys.argv[3], sys.argv[4])
    if len(sys.argv) == 4 and sys.argv[1] == "compare":
        return compare(sys.argv[2], sys.argv[3])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
