#!/usr/bin/env python3
"""Compare GLM-5.3-Flash graph/eager serving latency and throughput.

The script is intentionally standalone so both service modes use exactly the
same requests and measurement code.  Every run records streaming inter-token
intervals, so aggregate wall time is not mistaken for decode latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time

import requests

MODEL = "GLM-5.3-Flash-Mix"


def make_prompt(target_words: int) -> str:
    words = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
    ]
    body = " ".join(words[i % len(words)] for i in range(target_words))
    return (
        "Read the following sequence and then answer with a numbered list. "
        "Do not repeat the sequence.\n\n" + body
    )


def one_request(host: str, prompt: str, max_tokens: int, result: dict, idx: int) -> None:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    ttft = None
    token_times: list[float] = []
    usage = None
    try:
        with requests.post(
            f"http://{host}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=1800,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - started
                    token_times.append(now)
    except Exception as error:  # noqa: BLE001
        result[idx] = {"error": f"{type(error).__name__}: {error}"}
        return

    finished = time.perf_counter()
    gaps_ms = [
        (token_times[i] - token_times[i - 1]) * 1000.0
        for i in range(1, len(token_times))
    ]
    completion = (usage or {}).get("completion_tokens") or len(token_times)
    result[idx] = {
        "ttft_ms": None if ttft is None else ttft * 1000.0,
        "total_ms": (finished - started) * 1000.0,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": completion,
        "stream_chunks": len(token_times),
        "gap_p50_ms": statistics.median(gaps_ms) if gaps_ms else None,
        "gap_p90_ms": (
            sorted(gaps_ms)[min(len(gaps_ms) - 1, int(0.9 * len(gaps_ms)))]
            if gaps_ms
            else None
        ),
    }


def run_case(
    host: str,
    name: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    repeat: int,
) -> dict:
    repeats = []
    for _ in range(repeat):
        results: list[dict | None] = [None] * concurrency
        started = time.perf_counter()
        threads = [
            threading.Thread(
                target=one_request,
                args=(host, prompt, max_tokens, results, i),
            )
            for i in range(concurrency)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall_s = time.perf_counter() - started

        ok = [item for item in results if item and "error" not in item]
        failed = [item for item in results if item and "error" in item]
        case = {
            "case": name,
            "concurrency": concurrency,
            "max_tokens": max_tokens,
            "wall_s": wall_s,
            "ok": len(ok),
            "failed": len(failed),
        }
        if failed:
            case["errors"] = [item["error"] for item in failed[:3]]
        if ok:
            case.update(
                {
                    "ttft_ms": statistics.median(item["ttft_ms"] for item in ok),
                    "e2e_ms": statistics.median(item["total_ms"] for item in ok),
                    "gap_p50_ms": statistics.median(
                        item["gap_p50_ms"] for item in ok if item["gap_p50_ms"] is not None
                    ),
                    "gap_p90_ms": statistics.median(
                        item["gap_p90_ms"] for item in ok if item["gap_p90_ms"] is not None
                    ),
                    "prompt_tokens": statistics.median(
                        item["prompt_tokens"] for item in ok if item["prompt_tokens"]
                    ),
                    "completion_tokens": sum(item["completion_tokens"] for item in ok),
                    "request_tok_s": statistics.median(
                        item["completion_tokens"]
                        / ((item["total_ms"] - item["ttft_ms"]) / 1000.0)
                        for item in ok
                    ),
                    "aggregate_tok_s": sum(
                        item["completion_tokens"] for item in ok
                    )
                    / wall_s,
                }
            )
        repeats.append(case)

    if len(repeats) == 1:
        return repeats[0]

    out = dict(repeats[0])
    for field in (
        "wall_s",
        "ttft_ms",
        "e2e_ms",
        "gap_p50_ms",
        "gap_p90_ms",
        "request_tok_s",
        "aggregate_tok_s",
    ):
        values = [item[field] for item in repeats if item.get(field) is not None]
        if values:
            out[field] = statistics.median(values)
    out["runs"] = [
        {
            "wall_s": round(item["wall_s"], 3),
            "ttft_ms": round(item["ttft_ms"], 1) if item.get("ttft_ms") else None,
            "gap_p50_ms": round(item["gap_p50_ms"], 2)
            if item.get("gap_p50_ms")
            else None,
            "aggregate_tok_s": round(item["aggregate_tok_s"], 2)
            if item.get("aggregate_tok_s")
            else None,
        }
        for item in repeats
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1:18005")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    prompts = {
        "short": make_prompt(16),
        "medium": make_prompt(700),
        "long": make_prompt(1000),
    }
    cases = [
        ("short-decode", "short", 96, 1),
        ("short-decode", "short", 96, 8),
        ("short-decode", "short", 64, 32),
        ("short-decode", "short", 64, 64),
        ("medium-decode", "medium", 64, 1),
        ("medium-decode", "medium", 64, 8),
        ("long-prefill", "long", 8, 1),
    ]

    # Warm the graph buckets and batching paths before timing.
    print(json.dumps(run_case(args.host, "warmup", prompts["short"], 8, 4, 1)), flush=True)
    report = {"tag": args.tag, "host": args.host, "results": []}
    for index, (name, prompt_name, max_tokens, concurrency) in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {name} concurrency={concurrency}", flush=True)
        result = run_case(
            args.host,
            name,
            prompts[prompt_name],
            max_tokens,
            concurrency,
            args.repeat,
        )
        report["results"].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        with open(args.output, "w") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
