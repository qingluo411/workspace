#!/usr/bin/env python3
"""Create a long-context reference or compare an MTP server against it."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import requests

MODEL = "GLM-5.3-Flash-Mix"
DEFAULT_LENGTHS = [4096, 16384, 32768, 65536]
PROMPT_SEED = 20261113
TOKEN_MIN = 1000
TOKEN_MAX = 50000


def make_prompts(lengths: list[int]) -> dict[int, list[int]]:
    rng = random.Random(PROMPT_SEED)
    base = [rng.randrange(TOKEN_MIN, TOKEN_MAX) for _ in range(max(lengths))]
    return {length: base[:length] for length in lengths}


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
    response = requests.post(
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
        "token_ids": token_ids,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "usage": body.get("usage"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18005)
    parser.add_argument("--mode", choices=("reference", "compare"), required=True)
    parser.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS)))
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    lengths = [int(item) for item in args.lengths.split(",") if item]
    prompts = make_prompts(lengths)
    reference: dict[str, Any] | None = None
    if args.mode == "compare":
        if args.reference is None:
            parser.error("--reference is required in compare mode")
        reference = json.loads(args.reference.read_text())

    report: dict[str, Any] = {
        "mode": args.mode,
        "port": args.port,
        "seed": PROMPT_SEED,
        "max_tokens": args.max_tokens,
        "lengths": {},
    }
    all_ok = True
    for length in lengths:
        print(f"length={length} start", flush=True)
        result = request_once(args.port, prompts[length], args.max_tokens, args.timeout)
        entry: dict[str, Any] = {
            "elapsed_s": result["elapsed_s"],
            "prompt_tokens": (result.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (result.get("usage") or {}).get("completion_tokens"),
            "token_ids": result["token_ids"],
        }
        if reference is not None:
            wanted = reference["lengths"][str(length)]["token_ids"]
            entry["matches_reference"] = result["token_ids"] == wanted
            entry["first_divergence"] = next(
                (
                    i
                    for i, (actual, expected) in enumerate(
                        zip(result["token_ids"], wanted)
                    )
                    if actual != expected
                ),
                None,
            )
            all_ok = all_ok and entry["matches_reference"]
        report["lengths"][str(length)] = entry
        print(
            f"length={length} elapsed={result['elapsed_s']:.3f}s "
            f"tokens={result['token_ids'][:8]} "
            f"match={entry.get('matches_reference', 'ref')}",
            flush=True,
        )

    report["all_ok"] = all_ok
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"all_ok={all_ok}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
