#!/usr/bin/env python3
"""Run a deterministic fixed-length greedy prompt and compare token IDs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

MODEL = "GLM-5.3-Flash-Mix"


def make_prompt(length: int) -> list[int]:
    return [100 + (index % 1000) for index in range(length)]


def request_once(port: int, prompt: list[int], max_tokens: int, timeout: float) -> dict:
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
    response = requests.post(
        f"http://127.0.0.1:{port}/v1/completions",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    return {
        "token_ids": body["choices"][0]["token_ids"],
        "elapsed_s": round(time.perf_counter() - started, 3),
        "usage": body.get("usage"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18005)
    parser.add_argument("--mode", choices=("reference", "compare"), required=True)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    prompt = make_prompt(args.prompt_length)
    expected = None
    if args.mode == "compare":
        if args.reference is None:
            parser.error("--reference is required in compare mode")
        reference = json.loads(args.reference.read_text())
        expected = reference.get("token_ids")
        if expected is None and reference.get("runs"):
            expected = reference["runs"][0]["token_ids"]
        if expected is None:
            parser.error(f"reference has no token_ids: {args.reference}")

    runs = []
    passed = 0
    for index in range(1, args.runs + 1):
        result = request_once(args.port, prompt, args.max_tokens, args.timeout)
        token_ids = result["token_ids"]
        if expected is not None:
            ok = token_ids == expected
            passed += int(ok)
            first_divergence = next(
                (
                    pos
                    for pos, (actual, wanted) in enumerate(zip(token_ids, expected))
                    if actual != wanted
                ),
                None,
            )
            print(
                f"{index:02d} {'PASS' if ok else 'FAIL'} "
                f"first_divergence={first_divergence} "
                f"elapsed={result['elapsed_s']:.3f}s tokens={token_ids[:12]}",
                flush=True,
            )
        else:
            print(
                f"reference elapsed={result['elapsed_s']:.3f}s "
                f"tokens={token_ids}",
                flush=True,
            )
        runs.append(
            {
                "run": index,
                "token_ids": token_ids,
                "elapsed_s": result["elapsed_s"],
                "usage": result["usage"],
            }
        )

    report = {
        "mode": args.mode,
        "prompt_length": args.prompt_length,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "runs": runs,
        "passed": passed if expected is not None else None,
        "total": args.runs if expected is not None else None,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if expected is not None:
        print(f"passed={passed}/{args.runs}")
        return 0 if passed == args.runs else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
