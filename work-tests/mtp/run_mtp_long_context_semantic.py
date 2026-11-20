#!/usr/bin/env python3
"""Run a long-context needle-in-a-haystack check against a GLM server."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

MODEL = "GLM-5.3-Flash-Mix"
LINE = "The quick brown fox jumps over the lazy dog. GLM long-context marker line.\\n"


def count_tokens(host: str, text: str) -> int:
    response = requests.post(
        f"http://{host}/tokenize",
        json={"model": MODEL, "prompt": text, "add_special_tokens": True},
        timeout=120,
    )
    response.raise_for_status()
    return int(response.json()["count"])


def build_prompt(lines: int, marker: str) -> str:
    return (
        f"The secret marker is {marker}. Read the following text and reply with "
        f"exactly the marker and nothing else.\\n" + LINE * lines
    )


def request_once(
    host: str, text: str, max_tokens: int, timeout: float
) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    started = time.perf_counter()
    response = requests.post(
        f"http://{host}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"].get("content") or ""
    return {
        "content": content,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "usage": body.get("usage"),
        "finish_reason": body["choices"][0].get("finish_reason"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1:18005")
    parser.add_argument("--lines", default="1000,2000,4000")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = [int(item) for item in args.lines.split(",") if item]
    report: dict[str, Any] = {"host": args.host, "cases": {}}
    all_ok = True
    for lines in cases:
        marker = f"NEEDLE-{lines}"
        text = build_prompt(lines, marker)
        prompt_tokens = count_tokens(args.host, text)
        print(f"lines={lines} prompt_tokens={prompt_tokens} start", flush=True)
        result = request_once(args.host, text, args.max_tokens, args.timeout)
        compact = " ".join(result["content"].split())
        ok = marker in compact
        all_ok = all_ok and ok
        entry = {
            "lines": lines,
            "marker": marker,
            "prompt_tokens": prompt_tokens,
            "elapsed_s": result["elapsed_s"],
            "usage": result["usage"],
            "finish_reason": result["finish_reason"],
            "content_prefix": compact[:240],
            "marker_found": ok,
        }
        report["cases"][str(lines)] = entry
        print(
            f"lines={lines} elapsed={entry['elapsed_s']:.3f}s "
            f"prompt_tokens={prompt_tokens} marker_found={ok} "
            f"content={compact[:120]!r}",
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
