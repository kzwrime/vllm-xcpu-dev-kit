#!/usr/bin/env python3
"""One-launch GLM5.2 end-to-end suite: single/multi stream x 0.5K/4K, short outputs.

Needs a running server (run_vllm_test.sh --suite-test starts it once and calls this):

  python serve_test/test_glm52_suite.py -e presets/mpi/moe/GLM-5.2-FP8_dp1_tp4_eager_fp8_kvcache.sh

Phases (each request is a streamed greedy chat completion capped at --max-tokens,
with a unique cache_salt so no phase is served from the prefix cache):

  warmup        tiny prompt, not reported
  single-0.5k   one request at a time, --single-repeat times
  single-4k
  multi-0.5k    --concurrency simultaneous requests
  multi-4k
  mixed         --concurrency requests alternating 0.5K / 4K (optional, --mixed)

Per request: TTFT, decode tokens/s, completion tokens, finish reason. A phase passes when
every request finishes without error and produces output. Greedy outputs of requests that
share a prompt are also compared (single vs. multi, and within a multi phase); a mismatch
is reported as a warning because batch shape may legitimately change numerics, and fails
the run only with --strict-consistency.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from long_context_accuracy import load_cases, source_env_with_preset

HERE = Path(__file__).resolve().parent
CASES_FILE = HERE / "long_context_data" / "long_context_squad_cases.jsonl"
LENGTHS = {"0.5k": 512, "4k": 4096}


def stream_chat(base_url, model, messages, max_tokens, temperature, timeout, salt):
    """Run one streamed chat completion; returns a metrics dict (never raises)."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": salt,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    out = {"text": "", "ttft": None, "finish": None, "usage": None, "error": None}
    start = time.monotonic()
    last = start
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    out["usage"] = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    piece = (delta.get("content") or "") + (
                        delta.get("reasoning") or delta.get("reasoning_content") or ""
                    )
                    if piece:
                        last = time.monotonic()
                        if out["ttft"] is None:
                            out["ttft"] = last - start
                        out["text"] += piece
                    if choice.get("finish_reason"):
                        out["finish"] = choice["finish_reason"]
    except (urllib.error.URLError, OSError, ValueError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["total"] = time.monotonic() - start
    out["decode_time"] = last - start - (out["ttft"] or 0.0)
    usage = out["usage"] or {}
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["completion_tokens"] = usage.get("completion_tokens")
    n = out["completion_tokens"] or 0
    out["decode_tps"] = (n - 1) / out["decode_time"] if n > 1 and out["decode_time"] > 0 else None
    return out


def summarize(name, results, wall):
    ok = [r for r in results if not r["error"] and r["text"].strip()]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    tps = [r["decode_tps"] for r in ok if r["decode_tps"]]
    comp = sum(r["completion_tokens"] or 0 for r in results)
    prompt = sum(r["prompt_tokens"] or 0 for r in results)
    return {
        "phase": name,
        "requests": len(results),
        "ok": len(ok),
        "pass": len(ok) == len(results),
        "prompt_tokens": prompt,
        "completion_tokens": comp,
        "wall_s": round(wall, 3),
        "ttft_mean_s": round(statistics.mean(ttfts), 3) if ttfts else None,
        "ttft_max_s": round(max(ttfts), 3) if ttfts else None,
        "decode_tps_per_req": round(statistics.mean(tps), 3) if tps else None,
        "decode_tps_total": round(sum(tps), 3) if tps else None,
        "errors": sorted({r["error"] for r in results if r["error"]}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", metavar="PRESET", help="preset file (port / model come from it)")
    parser.add_argument("--base-url", help="override http://127.0.0.1:<USER_VLLM_PORT>")
    parser.add_argument("--model", help="override USER_VLLM_MODEL")
    parser.add_argument("--max-tokens", type=int, default=32, help="output cap per request")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=8, help="requests per multi phase")
    parser.add_argument("--single-repeat", type=int, default=2, help="requests per single phase")
    parser.add_argument("--mixed", action="store_true", help="also run a mixed 0.5K/4K phase")
    parser.add_argument("--only", help="comma-separated phase names to run")
    parser.add_argument("--timeout", type=float, default=7200, help="per-request timeout (s)")
    parser.add_argument("--strict-consistency", action="store_true")
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    args = parser.parse_args()

    env = source_env_with_preset(str(HERE), preset_file=args.e)
    model = args.model or env.get("USER_VLLM_MODEL")
    base_url = args.base_url or f"http://127.0.0.1:{env.get('USER_VLLM_PORT', '8000')}"
    if not model:
        raise SystemExit("[error] USER_VLLM_MODEL not set; pass -e or --model")
    cases = {c["case_id"]: c for c in load_cases(CASES_FILE, list(LENGTHS.values()))}

    def run(case_id, n):
        messages = cases[case_id]["messages"]
        salts = [f"suite-{uuid.uuid4().hex}" for _ in range(n)]
        t0 = time.monotonic()
        with ThreadPoolExecutor(n) as pool:
            futures = [
                pool.submit(
                    stream_chat, base_url, model, messages, args.max_tokens,
                    args.temperature, args.timeout, salt,
                )
                for salt in salts
            ]
            return [f.result() for f in futures], time.monotonic() - t0

    def run_serial(case_id, n):
        results, t0 = [], time.monotonic()
        for _ in range(n):
            results += run(case_id, 1)[0]
        return results, time.monotonic() - t0

    def run_mixed(n):
        ids = ["0.5k", "4k"]
        t0 = time.monotonic()
        with ThreadPoolExecutor(n) as pool:
            futures = [
                pool.submit(
                    stream_chat, base_url, model, cases[ids[i % 2]]["messages"],
                    args.max_tokens, args.temperature, args.timeout, f"suite-{uuid.uuid4().hex}",
                )
                for i in range(n)
            ]
            return [f.result() for f in futures], time.monotonic() - t0

    phases = [
        ("single-0.5k", lambda: run_serial("0.5k", args.single_repeat), "0.5k"),
        ("single-4k", lambda: run_serial("4k", args.single_repeat), "4k"),
        ("multi-0.5k", lambda: run("0.5k", args.concurrency), "0.5k"),
        ("multi-4k", lambda: run("4k", args.concurrency), "4k"),
    ]
    if args.mixed:
        phases.append(("mixed", lambda: run_mixed(args.concurrency), None))
    if args.only:
        wanted = set(args.only.split(","))
        phases = [p for p in phases if p[0] in wanted]

    print(f"[suite] server={base_url} model={model} max_tokens={args.max_tokens} "
          f"concurrency={args.concurrency}", flush=True)
    warm = stream_chat(base_url, model, [{"role": "user", "content": "Say hi."}],
                       4, args.temperature, args.timeout, f"suite-{uuid.uuid4().hex}")
    if warm["error"]:
        raise SystemExit(f"[suite] server not usable: {warm['error']}")
    print(f"[suite] warmup ok ({warm['total']:.2f}s)", flush=True)

    report = {"model": model, "max_tokens": args.max_tokens, "phases": [], "outputs": {}}
    reference = {}  # case_id -> single-stream output text
    consistency = []
    for name, fn, case_id in phases:
        results, wall = fn()
        s = summarize(name, results, wall)
        report["phases"].append(s)
        report["outputs"][name] = [r["text"] for r in results]
        if case_id is not None:
            texts = {r["text"] for r in results if not r["error"]}
            if len(texts) > 1:
                consistency.append(f"{name}: {len(texts)} distinct outputs across identical greedy requests")
            if name.startswith("single"):
                reference.setdefault(case_id, results[0]["text"])
            elif case_id in reference and texts - {reference[case_id]}:
                consistency.append(f"{name}: differs from single-stream output")
        print(f"[suite] {name:<12} {'PASS' if s['pass'] else 'FAIL'}  "
              f"reqs={s['ok']}/{s['requests']} prompt_tok={s['prompt_tokens']} "
              f"ttft_mean={s['ttft_mean_s']}s ttft_max={s['ttft_max_s']}s "
              f"tps/req={s['decode_tps_per_req']} tps_total={s['decode_tps_total']} "
              f"wall={s['wall_s']}s {s['errors'] or ''}", flush=True)
        print(f"[suite]   sample: {results[0]['text'].strip()[:120]!r}", flush=True)

    report["consistency_warnings"] = consistency
    for line in consistency:
        print(f"[suite] WARN {line}", flush=True)
    passed = all(p["pass"] for p in report["phases"]) and not (
        args.strict_consistency and consistency
    )
    report["pass"] = passed
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"[suite] report: {args.output}", flush=True)
    print(f"[suite] {'ALL PASS' if passed else 'FAILED'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
