#!/usr/bin/env python3
"""Repeat one greedy prompt and save exact generated token IDs per run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

MODEL = "GLM-5.3-Flash-Mix"


def request_once(
    port: int,
    prompt: str | list[int],
    max_tokens: int,
    min_tokens: int,
    timeout: float,
    request_id: str | None = None,
) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
        "return_token_ids": True,
    }
    if request_id is not None:
        payload["request_id"] = request_id
    started = time.perf_counter()
    response = requests.post(
        f"http://127.0.0.1:{port}/v1/completions",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    return {
        "elapsed_s": round(time.perf_counter() - started, 3),
        "token_ids": choice["token_ids"],
        "text": choice["text"],
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
    }


def first_divergence(left: list[int], right: list[int]) -> int | None:
    return next(
        (
            index
            for index, (left_token, right_token) in enumerate(
                zip(left, right)
            )
            if left_token != right_token
        ),
        None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18005)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt-file", type=Path)
    prompt_group.add_argument("--prompt-token-ids", type=Path)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--min-tokens", type=int)
    parser.add_argument("--request-id-prefix", default="repeat-token-dump")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8")
        prompt_metadata = {
            "prompt_file": str(args.prompt_file),
            "prompt_num_chars": len(prompt),
        }
    else:
        prompt = json.loads(args.prompt_token_ids.read_text(encoding="utf-8"))
        if not isinstance(prompt, list) or not all(
            isinstance(token, int) for token in prompt
        ):
            parser.error("--prompt-token-ids must contain a JSON list of integers")
        prompt_metadata = {
            "prompt_token_ids_file": str(args.prompt_token_ids),
            "prompt_num_tokens": len(prompt),
        }
    min_tokens = (
        args.max_tokens if args.min_tokens is None else args.min_tokens
    )

    warmup = None
    if args.warmup_tokens > 0:
        warmup = request_once(
            args.port,
            prompt,
            args.warmup_tokens,
            args.warmup_tokens,
            args.timeout,
            f"{args.request_id_prefix}-warmup",
        )
        print(
            "warmup "
            f"elapsed={warmup['elapsed_s']:.3f}s "
            f"tokens={warmup['token_ids']}",
            flush=True,
        )

    runs = []
    for index in range(1, args.runs + 1):
        result = request_once(
            args.port,
            prompt,
            args.max_tokens,
            min_tokens,
            args.timeout,
            f"{args.request_id_prefix}-run{index}",
        )
        result["run"] = index
        runs.append(result)
        usage = result.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        print(
            f"run={index} elapsed={result['elapsed_s']:.3f}s "
            f"cached={details.get('cached_tokens')} "
            f"generated={len(result['token_ids'])} "
            f"first={result['token_ids'][:12]}",
            flush=True,
        )

    baseline = runs[0]["token_ids"]
    comparisons = []
    for run in runs[1:]:
        divergence = first_divergence(baseline, run["token_ids"])
        comparisons.append(
            {
                "run": run["run"],
                "first_divergence": divergence,
                "same": divergence is None
                and len(baseline) == len(run["token_ids"]),
            }
        )
        print(
            f"run={run['run']} first_divergence={divergence}",
            flush=True,
        )

    report = {
        **prompt_metadata,
        "warmup": warmup,
        "runs": runs,
        "comparisons": comparisons,
        "all_same": all(item["same"] for item in comparisons),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"all_same={report['all_same']} output={args.output}")
    return 0 if report["all_same"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
