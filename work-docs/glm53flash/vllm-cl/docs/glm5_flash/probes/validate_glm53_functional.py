#!/usr/bin/env python3
"""GLM-5.3-Flash text-serving correctness matrix for the DLC backend.

The script targets the OpenAI-compatible server started by
``start_glm5_flash_vllm_cl.sh``.  It covers single/concurrent requests,
decode, chunked prefill, a 32K-token context, and the streaming
prefill-to-decode boundary.  Results are printed and optionally written to a
JSON file for the release archive.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests


MODEL = "GLM-5.3-Flash-Mix"
LINE = "The quick brown fox jumps over the lazy dog. GLM validation marker line.\n"


def long_prompt(lines: int, marker: str) -> str:
    return (
        f"Validation marker: {marker}. Read the following repeated text and then "
        "reply with exactly one short sentence stating the marker.\n"
        + LINE * lines
    )


def count_tokens(host: str, text: str) -> int:
    response = requests.post(
        f"http://{host}/tokenize",
        json={"model": MODEL, "prompt": text, "add_special_tokens": True},
        timeout=120,
    )
    response.raise_for_status()
    return int(response.json()["count"])


def request_once(
    host: str,
    text: str,
    max_tokens: int,
    *,
    stream: bool,
    timeout: float = 1800,
) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    started = time.perf_counter()
    first_token_s: float | None = None
    content_parts: list[str] = []
    usage: dict[str, Any] | None = None
    with requests.post(
        f"http://{host}/v1/chat/completions",
        json=payload,
        stream=stream,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        if not stream:
            body = response.json()
            content = body["choices"][0]["message"].get("content") or ""
            usage = body.get("usage")
            return {
                "ok": True,
                "elapsed_s": time.perf_counter() - started,
                "ttft_s": None,
                "content": content,
                "usage": usage,
                "finish_reason": body["choices"][0].get("finish_reason"),
            }

        for line in response.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            body = line[6:]
            if body == b"[DONE]":
                break
            event = json.loads(body)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {}).get("content")
            if delta:
                if first_token_s is None:
                    first_token_s = time.perf_counter() - started
                content_parts.append(delta)

    return {
        "ok": True,
        "elapsed_s": time.perf_counter() - started,
        "ttft_s": first_token_s,
        "content": "".join(content_parts),
        "usage": usage,
    }


def validate_result(result: dict[str, Any], expected_prompt_tokens: int) -> None:
    if not result.get("ok"):
        raise AssertionError(f"request failed: {result}")
    content = result.get("content", "")
    usage = result.get("usage") or {}
    completion = usage.get("completion_tokens")
    prompt = usage.get("prompt_tokens")
    if not content.strip():
        raise AssertionError("empty completion content")
    if content.strip() and set(content.strip()) <= set("!！ "):
        raise AssertionError(f"degenerate completion: {content!r}")
    if not completion or completion <= 0:
        raise AssertionError(f"invalid completion usage: {usage}")
    # /tokenize counts the raw prompt; the chat endpoint additionally applies
    # the model's chat template and special tokens.
    if not prompt or abs(prompt - expected_prompt_tokens) > 128:
        raise AssertionError(
            f"prompt token mismatch: usage={prompt}, expected~={expected_prompt_tokens}"
        )


def compact(result: dict[str, Any]) -> dict[str, Any]:
    content = " ".join(result.get("content", "").split())
    return {
        "ok": result.get("ok"),
        "ttft_s": round(result["ttft_s"], 3) if result.get("ttft_s") else None,
        "elapsed_s": round(result["elapsed_s"], 3),
        "prompt_tokens": (result.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (result.get("usage") or {}).get("completion_tokens"),
        "finish_reason": result.get("finish_reason"),
        "content_prefix": content[:180],
        "content_chars": len(content),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1:18005")
    parser.add_argument("--output")
    args = parser.parse_args()
    host = args.host

    prompts = {
        "short": "请用两句话说明 2+3 等于多少，并给出理由。",
        "decode": "请从 1 开始依次写到 40。",
        "chunked": long_prompt(1000, "CHUNKED-16K"),
        "long": long_prompt(2000, "LONG-32K"),
    }
    expected = {name: count_tokens(host, text) for name, text in prompts.items()}
    report: dict[str, Any] = {
        "host": host,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_tokens": expected,
        "cases": {},
    }
    print(json.dumps({"prompt_tokens": expected}, ensure_ascii=False), flush=True)

    short = request_once(host, prompts["short"], 48, stream=False)
    validate_result(short, expected["short"])
    report["cases"]["single_nonstream"] = compact(short)
    print("single_nonstream", json.dumps(report["cases"]["single_nonstream"], ensure_ascii=False), flush=True)

    def concurrent_call(index: int) -> dict[str, Any]:
        prompt = f"并发请求 {index}：请用一句话回答 7*8 是多少。"
        result = request_once(host, prompt, 48, stream=False)
        validate_result(result, count_tokens(host, prompt))
        result["index"] = index
        return result

    concurrent_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        concurrent_results = list(pool.map(concurrent_call, range(4)))
    concurrent_wall = time.perf_counter() - concurrent_started
    report["cases"]["four_concurrent"] = {
        "wall_s": round(concurrent_wall, 3),
        "requests": [compact(item) for item in concurrent_results],
    }
    print(
        "four_concurrent",
        json.dumps(report["cases"]["four_concurrent"], ensure_ascii=False),
        flush=True,
    )

    decode = request_once(host, prompts["decode"], 128, stream=True)
    validate_result(decode, expected["decode"])
    decode["output_tokens_per_s"] = (
        (decode["usage"]["completion_tokens"] - 1) / (decode["elapsed_s"] - decode["ttft_s"])
        if decode.get("ttft_s") and decode["elapsed_s"] > decode["ttft_s"]
        else None
    )
    report["cases"]["decode_stream_128"] = compact(decode)
    report["cases"]["decode_stream_128"]["output_tokens_per_s"] = (
        round(decode["output_tokens_per_s"], 4)
        if decode["output_tokens_per_s"] is not None
        else None
    )
    print("decode_stream_128", json.dumps(report["cases"]["decode_stream_128"], ensure_ascii=False), flush=True)

    chunked = request_once(host, prompts["chunked"], 32, stream=True)
    validate_result(chunked, expected["chunked"])
    report["cases"]["chunked_prefill_16k"] = compact(chunked)
    print("chunked_prefill_16k", json.dumps(report["cases"]["chunked_prefill_16k"], ensure_ascii=False), flush=True)

    long = request_once(host, prompts["long"], 24, stream=True)
    validate_result(long, expected["long"])
    report["cases"]["long_context_32k"] = compact(long)
    print("long_context_32k", json.dumps(report["cases"]["long_context_32k"], ensure_ascii=False), flush=True)

    # The streaming requests above are the explicit prefill->decode boundary
    # check: TTFT must be finite, completion tokens must be non-zero, and the
    # final content must not collapse to punctuation-only output.
    ttfts = [
        result["ttft_s"]
        for result in (decode, chunked, long)
        if result.get("ttft_s") is not None
    ]
    report["boundary_summary"] = {
        "streaming_cases": 3,
        "nonempty_content": all((result.get("content") or "").strip() for result in (decode, chunked, long)),
        "median_ttft_s": round(statistics.median(ttfts), 3),
        "all_have_completion_usage": all(
            (result.get("usage") or {}).get("completion_tokens", 0) > 0
            for result in (decode, chunked, long)
        ),
    }
    print("boundary_summary", json.dumps(report["boundary_summary"], ensure_ascii=False), flush=True)

    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
