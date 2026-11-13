#!/usr/bin/env python3
"""Validate MTP decode output at several concurrent batch sizes."""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

MODEL = "GLM-5.3-Flash-Mix"
PROMPT = [100, 200, 300, 400, 500, 600, 700, 800]
CYCLE = [200, 300, 400, 500, 600, 700]

_thread_local = threading.local()


def session() -> requests.Session:
    value = getattr(_thread_local, "session", None)
    if value is None:
        value = requests.Session()
        _thread_local.session = value
    return value


def expected_tokens(max_tokens: int) -> list[int]:
    return [CYCLE[i % len(CYCLE)] for i in range(max_tokens)]


def request_once(
    port: int,
    max_tokens: int,
    timeout: float,
    barrier: threading.Barrier,
) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
        "return_token_ids": True,
    }
    barrier.wait()
    started = time.perf_counter()
    try:
        response = session().post(
            f"http://127.0.0.1:{port}/v1/completions",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        token_ids = body["choices"][0].get("token_ids")
        if token_ids is None:
            raise RuntimeError("response has no token_ids")
        return {
            "ok": True,
            "token_ids": token_ids,
            "elapsed_s": time.perf_counter() - started,
            "usage": body.get("usage"),
        }
    except Exception as exc:  # noqa: BLE001 - report client/server failures per request
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.perf_counter() - started,
        }


def run_round(
    port: int,
    concurrency: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    expected = expected_tokens(max_tokens)
    barrier = threading.Barrier(concurrency)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(request_once, port, max_tokens, timeout, barrier)
            for _ in range(concurrency)
        ]
        results = [future.result() for future in futures]
    wall_s = time.perf_counter() - started

    failures: list[dict[str, Any]] = []
    latencies: list[float] = []
    for index, result in enumerate(results):
        latencies.append(float(result["elapsed_s"]))
        if not result.get("ok"):
            failures.append({"index": index, "error": result.get("error")})
            continue
        tokens = result["token_ids"]
        if tokens != expected:
            first = next(
                (
                    i
                    for i, (actual, wanted) in enumerate(zip(tokens, expected))
                    if actual != wanted
                ),
                None,
            )
            failures.append(
                {
                    "index": index,
                    "first_divergence": first,
                    "tokens": tokens[:16],
                }
            )

    return {
        "concurrency": concurrency,
        "requests": concurrency,
        "passed": concurrency - len(failures),
        "failed": len(failures),
        "failures": failures,
        "wall_s": round(wall_s, 3),
        "output_tokens_per_s": round(concurrency * max_tokens / wall_s, 3),
        "latency_s": {
            "min": round(min(latencies), 3),
            "max": round(max(latencies), 3),
            "mean": round(sum(latencies) / len(latencies), 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18005)
    parser.add_argument("--concurrencies", default="1,8,16,32,64")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    concurrencies = [int(item) for item in args.concurrencies.split(",") if item]
    report: dict[str, Any] = {
        "port": args.port,
        "rounds": args.rounds,
        "max_tokens": args.max_tokens,
        "cases": [],
    }
    all_ok = True
    for concurrency in concurrencies:
        for round_index in range(1, args.rounds + 1):
            result = run_round(args.port, concurrency, args.max_tokens, args.timeout)
            result["round"] = round_index
            report["cases"].append(result)
            all_ok = all_ok and result["failed"] == 0
            print(
                f"c={concurrency} round={round_index} "
                f"pass={result['passed']}/{result['requests']} "
                f"wall={result['wall_s']:.3f}s "
                f"out_tok/s={result['output_tokens_per_s']:.3f} "
                f"latency_mean={result['latency_s']['mean']:.3f}s",
                flush=True,
            )
            if result["failures"]:
                print(
                    json.dumps(result["failures"][:3], ensure_ascii=False), flush=True
                )

    report["all_ok"] = all_ok
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"all_ok={all_ok}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
