#!/usr/bin/env python3
"""End-to-end GLM-5.3-Flash benchmark for the DLC plugin (streaming TTFT + decode rate).

Snapshots the vLLM Prometheus counters before/after so the client-side numbers can be
checked against the engine's own view of prompt/generation tokens.

Usage:
  python -u bench_e2e_glm53.py                      # default cases
  python -u bench_e2e_glm53.py --host 127.0.0.1:8000 --tag kernel
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time

import requests

SENTENCE = (
    "Summarize the following text in one sentence:\n"
    + "The quick brown fox jumps over the lazy dog. " * 40
)


def load_prompt(path):
    try:
        with open(path) as handle:
            body = json.load(handle)
        return body["messages"][0]["content"]
    except Exception:  # noqa: BLE001
        return SENTENCE


def one_request(host, prompt, max_tokens, results, idx, timeout=1800):
    payload = {
        "model": "GLM-5.3-Flash-Mix",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    usage = None
    try:
        with requests.post(f"http://{host}/v1/chat/completions", json=payload, stream=True,
                           timeout=timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                body = line[6:]
                if body == b"[DONE]":
                    break
                event = json.loads(body)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
    except Exception as error:  # noqa: BLE001
        results[idx] = {"error": f"{type(error).__name__}: {error}"}
        return
    total = time.perf_counter() - t0
    results[idx] = {
        "ttft": ttft, "total": total, "chunks": chunks,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
    }


def run_case(host, name, prompt, max_tokens, concurrency):
    results = [None] * concurrency
    started = time.perf_counter()
    threads = [threading.Thread(target=one_request, args=(host, prompt, max_tokens, results, i))
               for i in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started

    ok = [r for r in results if r and "error" not in r]
    failed = [r for r in results if r and "error" in r]
    out = {"case": name, "concurrency": concurrency, "wall_s": wall,
           "ok": len(ok), "failed": len(failed)}
    if failed:
        out["errors"] = [r["error"] for r in failed][:2]
    if not ok:
        return out
    out["ttft_ms"] = statistics.median(r["ttft"] * 1000 for r in ok if r["ttft"])
    out["e2e_ms"] = statistics.median(r["total"] * 1000 for r in ok)
    prompt_tokens = [r["prompt_tokens"] for r in ok if r["prompt_tokens"]]
    out["prompt_tokens"] = statistics.median(prompt_tokens) if prompt_tokens else None
    completion = sum(r["completion_tokens"] or r["chunks"] for r in ok)
    out["completion_tokens"] = completion
    out["decode_tok_s"] = completion / wall
    prefill = sum(r["prompt_tokens"] or 0 for r in ok)
    out["prefill_tok_s"] = prefill / wall
    out["out_tok_s_per_req"] = statistics.median(
        (r["completion_tokens"] or r["chunks"]) / (r["total"] - (r["ttft"] or 0)) for r in ok)
    return out


def metrics(host):
    text = requests.get(f"http://{host}/metrics", timeout=30).text
    want = ("vllm:prompt_tokens_total{", "vllm:generation_tokens_total{",
            "vllm:request_success_total{engine=\"0\",finished_reason=\"length\"")
    got = {}
    for line in text.splitlines():
        for key in want:
            if line.startswith(key):
                got[line.rsplit(" ", 1)[0]] = float(line.rsplit(" ", 1)[1])
    return got


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1:8000")
    parser.add_argument("--tag", default="")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--prompt-file", default="/tmp/request_body_from_test_2048_3.json")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    host = args.host
    long_prompt = load_prompt(args.prompt_file)
    print(f"host={host} tag={args.tag} long_prompt_chars={len(long_prompt)}", flush=True)

    if not args.skip_warmup:
        print("\n== warmup ==", flush=True)
        print(json.dumps(run_case(host, "warmup", SENTENCE, 8, 1), ensure_ascii=False), flush=True)

    before = metrics(host)
    cases = [
        ("prefill-2k-1req-32out", long_prompt, 32, 1),
        ("prefill-2k-3req-64out", long_prompt, 64, 3),
        ("short-128out", "Hello, please count from 1 to 20 slowly.", 128, 1),
    ]
    report = []
    for name, prompt, max_tokens, concurrency in cases:
        runs = [run_case(host, name, prompt, max_tokens, concurrency) for _ in range(args.repeat)]
        result = runs[0]
        if len(runs) > 1:
            for field in ("wall_s", "ttft_ms", "e2e_ms", "decode_tok_s", "prefill_tok_s",
                          "out_tok_s_per_req"):
                values = [r[field] for r in runs if r.get(field) is not None]
                if values:
                    result[field] = statistics.median(values)
            result["all_wall_s"] = [round(r["wall_s"], 2) for r in runs]
        report.append(result)
        print("\n" + json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    after = metrics(host)

    print("\n== engine counters (delta over the whole run) ==", flush=True)
    for key in sorted(set(before) | set(after)):
        delta = after.get(key, 0.0) - before.get(key, 0.0)
        if delta:
            print(f"  {key}: {delta:+.1f}", flush=True)
    print("\n== summary ==", flush=True)
    print(f"{'case':24} {'conc':>4} {'wall_s':>8} {'ttft_ms':>9} {'e2e_ms':>9} "
          f"{'prompt_tok':>10} {'out_tok':>8} {'out_tok/s':>10} {'prefill_tok/s':>14}")
    for r in report:
        print(f"{r['case']:24} {r['concurrency']:>4} {r['wall_s']:>8.2f} "
              f"{r.get('ttft_ms', float('nan')):>9.1f} {r.get('e2e_ms', float('nan')):>9.1f} "
              f"{r.get('prompt_tokens', 0):>10.0f} {r.get('completion_tokens', 0):>8} "
              f"{r.get('decode_tok_s', 0):>10.2f} {r.get('prefill_tok_s', 0):>14.2f}", flush=True)


if __name__ == "__main__":
    main()
