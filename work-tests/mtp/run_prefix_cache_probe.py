#!/usr/bin/env python3
"""Probe repeated-prompt prefix-cache hit timing and greedy stability."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

MODEL = "GLM-5.3-Flash-Mix"


def make_prompt(length: int) -> list[int]:
    return [100 + (index % 1000) for index in range(length)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18005)
    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    prompt = make_prompt(args.prompt_length)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "min_tokens": args.max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
        "return_token_ids": True,
    }
    url = f"http://127.0.0.1:{args.port}/v1/completions"
    runs = []
    for index in range(1, args.runs + 1):
        started = time.perf_counter()
        response = requests.post(url, json=payload, timeout=args.timeout)
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        token_ids = body["choices"][0]["token_ids"]
        elapsed = time.perf_counter() - started
        runs.append(
            {
                "run": index,
                "elapsed_s": round(elapsed, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": details.get("cached_tokens"),
                "token_ids": token_ids,
            }
        )
        print(
            f"{index:02d} elapsed={elapsed:.3f}s "
            f"cached={details.get('cached_tokens')} "
            f"tokens={token_ids[:10]}",
            flush=True,
        )
    same = all(run["token_ids"] == runs[0]["token_ids"] for run in runs[1:])
    report = {
        "prompt_length": args.prompt_length,
        "max_tokens": args.max_tokens,
        "runs": runs,
        "all_same_tokens": same,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"all_same_tokens={same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
