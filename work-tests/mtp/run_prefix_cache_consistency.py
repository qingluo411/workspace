#!/usr/bin/env python3
"""Exercise prefix-cache cold/hot consistency against a running server.

The server should be started with prefix caching enabled and
``--enable-prompt-tokens-details`` so responses expose
``usage.prompt_tokens_details.cached_tokens``.

For correctness triage, start it with the deterministic MoE build and
``USE_CPU_FOR_COLLECT_OP=True`` so communication and MoE reductions are fixed.
"""

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


def make_prompt(length: int, seed: int) -> list[int]:
    """Build a deterministic prompt with a case-specific token prefix."""
    return [1000 + (seed * 7919 + i * 37) % 100000 for i in range(length)]


def request_once(
    port: int,
    prompt: list[int],
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
        "return_token_ids": True,
    }
    started = time.perf_counter()
    try:
        response = requests.post(
            f"http://127.0.0.1:{port}/v1/completions",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return {
            "ok": True,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
            "token_ids": body["choices"][0]["token_ids"],
        }
    except Exception as exc:  # noqa: BLE001 - report request failures
        return {
            "ok": False,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "token_ids": [],
            "cached_tokens": None,
        }


def _all_ok(records: list[dict[str, Any]]) -> bool:
    return all(record.get("ok") for record in records)


def _same_tokens(records: list[dict[str, Any]]) -> bool:
    if not records or not _all_ok(records):
        return False
    expected = records[0]["token_ids"]
    return all(record["token_ids"] == expected for record in records[1:])


def _hit_after_cold(records: list[dict[str, Any]]) -> bool:
    if len(records) < 2:
        return False
    return any((record.get("cached_tokens") or 0) > 0 for record in records[1:])


def _same_hot_cached(records: list[dict[str, Any]]) -> bool:
    if len(records) < 2:
        return False
    hot = [record.get("cached_tokens") or 0 for record in records[1:]]
    return hot[0] > 0 and all(value == hot[0] for value in hot)


def run_exact_repeat(
    port: int,
    length: int,
    seed: int,
    runs: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    prompt = make_prompt(length, seed)
    records = [
        request_once(port, prompt, max_tokens, timeout) for _ in range(runs)
    ]
    same = _same_tokens(records)
    hit = _hit_after_cold(records)
    same_cached = _same_hot_cached(records)
    ok = _all_ok(records) and same and hit and same_cached
    return {
        "name": f"exact_repeat_L{length}",
        "kind": "exact_repeat",
        "prompt_length": length,
        "seed": seed,
        "records": records,
        "checks": {
            "no_errors": _all_ok(records),
            "same_tokens": same,
            "hit_after_cold": hit,
            "same_hot_cached": same_cached,
        },
        "passed": ok,
    }


def run_shared_prefix(
    port: int,
    prefix_length: int,
    suffix_length: int,
    seed: int,
    runs: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    prefix = make_prompt(prefix_length, seed)
    suffix_a = make_prompt(suffix_length, seed + 100)
    suffix_b = make_prompt(suffix_length, seed + 200)
    prompt_a = prefix + suffix_a
    prompt_b = prefix + suffix_b

    records = [request_once(port, prompt_a, max_tokens, timeout)]
    records.extend(
        request_once(port, prompt_b, max_tokens, timeout) for _ in range(runs)
    )
    b_records = records[1:]
    same_b = _same_tokens(b_records)
    hit_b = any((record.get("cached_tokens") or 0) > 0 for record in b_records)
    same_cached_b = _same_hot_cached(records)
    ok = _all_ok(records) and same_b and hit_b and same_cached_b
    return {
        "name": f"shared_prefix_L{prefix_length}_S{suffix_length}",
        "kind": "shared_prefix",
        "prompt_length": prefix_length + suffix_length,
        "prefix_length": prefix_length,
        "suffix_length": suffix_length,
        "seed": seed,
        "records": records,
        "checks": {
            "no_errors": _all_ok(records),
            "shared_prefix_hit": hit_b,
            "same_suffix_tokens": same_b,
            "same_hot_cached": same_cached_b,
        },
        "passed": ok,
    }


def run_concurrent_hot(
    port: int,
    length: int,
    seed: int,
    concurrency: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    prompt = make_prompt(length, seed)
    warm = request_once(port, prompt, max_tokens, timeout)
    if not warm.get("ok"):
        return {
            "name": f"concurrent_hot_L{length}",
            "kind": "concurrent_hot",
            "prompt_length": length,
            "seed": seed,
            "records": [warm],
            "checks": {
                "no_errors": False,
                "same_tokens": False,
                "concurrent_hit": False,
            },
            "passed": False,
        }

    barrier = threading.Barrier(concurrency)

    def worker() -> dict[str, Any]:
        barrier.wait()
        return request_once(port, prompt, max_tokens, timeout)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = list(pool.map(lambda _: worker(), range(concurrency)))
    same = _same_tokens(records)
    hit = any((record.get("cached_tokens") or 0) > 0 for record in records)
    ok = _all_ok(records) and same and hit
    return {
        "name": f"concurrent_hot_L{length}_C{concurrency}",
        "kind": "concurrent_hot",
        "prompt_length": length,
        "seed": seed,
        "warm": warm,
        "records": records,
        "checks": {
            "no_errors": _all_ok(records),
            "same_tokens": same,
            "concurrent_hit": hit,
        },
        "passed": ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18005)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--lengths", default="1024,3584,4096")
    parser.add_argument("--shared-prefix-lengths", default="1024,3584")
    parser.add_argument("--suffix-length", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    for index, length in enumerate(
        int(item) for item in args.lengths.split(",") if item
    ):
        case = run_exact_repeat(
            args.port,
            length,
            args.seed_base + index * 10000,
            args.runs,
            args.max_tokens,
            args.timeout,
        )
        cases.append(case)
        print(
            f"{case['name']} passed={case['passed']} "
            f"cached={[r.get('cached_tokens') for r in case['records']]} "
            f"checks={case['checks']}",
            flush=True,
        )

    for index, length in enumerate(
        int(item) for item in args.shared_prefix_lengths.split(",") if item
    ):
        case = run_shared_prefix(
            args.port,
            length,
            args.suffix_length,
            args.seed_base + 20000 + index * 10000,
            args.runs,
            args.max_tokens,
            args.timeout,
        )
        cases.append(case)
        print(
            f"{case['name']} passed={case['passed']} "
            f"cached={[r.get('cached_tokens') for r in case['records']]} "
            f"checks={case['checks']}",
            flush=True,
        )

    if args.concurrency > 1:
        case = run_concurrent_hot(
            args.port,
            1024,
            args.seed_base + 30000,
            args.concurrency,
            args.max_tokens,
            args.timeout,
        )
        cases.append(case)
        print(
            f"{case['name']} passed={case['passed']} "
            f"cached={[r.get('cached_tokens') for r in case['records']]} "
            f"checks={case['checks']}",
            flush=True,
        )

    all_ok = all(case["passed"] for case in cases)
    report = {
        "model": MODEL,
        "cases": cases,
        "all_ok": all_ok,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"all_ok={all_ok}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
