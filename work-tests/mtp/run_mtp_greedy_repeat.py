#!/usr/bin/env python3
"""Repeat the fixed GLM-5.3 MTP greedy smoke request."""

from __future__ import annotations

import argparse
import json
import sys

import requests

PROMPT = [100, 200, 300, 400, 500, 600, 700, 800]
BASELINE_CYCLE = [200, 300, 400, 500, 600, 700]


def expected_tokens(max_tokens: int) -> list[int]:
    return [BASELINE_CYCLE[i % len(BASELINE_CYCLE)] for i in range(max_tokens)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18005)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--expected-json")
    args = parser.parse_args()

    if args.expected_json:
        with open(args.expected_json, encoding="utf-8") as handle:
            expected = json.load(handle)["token_ids"]
    else:
        expected = expected_tokens(args.max_tokens)

    payload = {
        "model": "GLM-5.3-Flash-Mix",
        "prompt": PROMPT,
        "max_tokens": args.max_tokens,
        "min_tokens": args.max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
        "return_token_ids": True,
    }
    url = f"http://127.0.0.1:{args.port}/v1/completions"
    passed = 0
    for run in range(1, args.runs + 1):
        response = requests.post(url, json=payload, timeout=args.timeout)
        response.raise_for_status()
        token_ids = response.json()["choices"][0]["token_ids"]
        ok = token_ids == expected
        passed += int(ok)
        print(
            f"{run:02d} {'PASS' if ok else 'FAIL'} "
            f"first_divergence={next((i for i, (a, b) in enumerate(zip(token_ids, expected)) if a != b), None)} "
            f"tokens={token_ids[:12]}"
        )
    print(f"passed={passed}/{args.runs}")
    return 0 if passed == args.runs else 1


if __name__ == "__main__":
    sys.exit(main())
