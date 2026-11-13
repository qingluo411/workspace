#!/usr/bin/env python3
"""Validate concurrent MTP requests with a unique marker per request."""

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
_thread_local = threading.local()


def session() -> requests.Session:
    value = getattr(_thread_local, "session", None)
    if value is None:
        value = requests.Session()
        _thread_local.session = value
    return value


def request_once(
    port: int,
    index: int,
    max_tokens: int,
    timeout: float,
    barrier: threading.Barrier,
) -> dict[str, Any]:
    marker = f"REQ-{index}"
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"Reply with exactly the marker {marker} and nothing else.",
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    barrier.wait()
    started = time.perf_counter()
    try:
        response = session().post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"].get("content") or ""
        return {
            "index": index,
            "marker": marker,
            "ok": True,
            "marker_found": marker in content,
            "content_prefix": " ".join(content.split())[:160],
            "elapsed_s": round(time.perf_counter() - started, 3),
            "usage": body.get("usage"),
        }
    except Exception as exc:  # noqa: BLE001 - report per-request failures
        return {
            "index": index,
            "marker": marker,
            "ok": False,
            "marker_found": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.perf_counter() - started, 3),
        }


def run_round(
    port: int, concurrency: int, max_tokens: int, timeout: float
) -> dict[str, Any]:
    barrier = threading.Barrier(concurrency)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(request_once, port, index, max_tokens, timeout, barrier)
            for index in range(concurrency)
        ]
        results = [future.result() for future in futures]
    wall_s = time.perf_counter() - started
    passed = sum(
        1 for result in results if result.get("ok") and result.get("marker_found")
    )
    failures = [
        result
        for result in results
        if not result.get("ok") or not result.get("marker_found")
    ]
    return {
        "concurrency": concurrency,
        "requests": concurrency,
        "passed": passed,
        "failed": len(failures),
        "wall_s": round(wall_s, 3),
        "output_tokens_per_s": round(concurrency * max_tokens / wall_s, 3),
        "failures": failures[:8],
        "records": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18005)
    parser.add_argument("--concurrencies", default="1,8,16,32,64")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    all_ok = True
    for concurrency in [int(item) for item in args.concurrencies.split(",") if item]:
        for round_index in range(1, args.rounds + 1):
            result = run_round(args.port, concurrency, args.max_tokens, args.timeout)
            result["round"] = round_index
            cases.append(result)
            all_ok = all_ok and result["failed"] == 0
            print(
                f"c={concurrency} round={round_index} "
                f"pass={result['passed']}/{result['requests']} "
                f"wall={result['wall_s']:.3f}s "
                f"out_tok/s={result['output_tokens_per_s']:.3f}",
                flush=True,
            )
            if result["failures"]:
                print(
                    json.dumps(result["failures"][:3], ensure_ascii=False), flush=True
                )

    report = {"all_ok": all_ok, "cases": cases}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"all_ok={all_ok}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
