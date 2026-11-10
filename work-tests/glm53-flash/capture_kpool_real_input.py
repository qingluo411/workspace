#!/usr/bin/env python3
"""Send one exact random-token request so the server can capture KPool inputs."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18005")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", default="/data/models/GLM-5.3-Flash-Mix")
    parser.add_argument("--input-len", type=int, default=16384)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--timeout-sec", type=float, default=7200)
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "kpool-real-input-16k",
        help="Directory scanned after the request; also receives the prompt token IDs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    models = request_json(f"{base_url}/v1/models", None, 30)
    model = args.model or models["data"][0]["id"]

    config_path = Path(args.model_path) / "config.json"
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    vocab_size_value = text_config.get("vocab_size")
    if vocab_size_value is None:
        vocab_size_value = config.get("vocab_size")
    if vocab_size_value is None:
        raise RuntimeError(f"vocab_size is missing from {config_path}")
    vocab_size = int(vocab_size_value)
    generator = torch.Generator().manual_seed(args.seed)
    prompt = torch.randint(
        0, vocab_size, (args.input_len,), generator=generator, dtype=torch.int32
    )

    args.capture_dir.mkdir(parents=True, exist_ok=True)
    request_metadata = {
        "model": model,
        "model_path": args.model_path,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "seed": args.seed,
        "vocab_size": vocab_size,
        "base_url": base_url,
    }
    torch.save(prompt, args.capture_dir / "request_tokens.pt")
    (args.capture_dir / "request.json").write_text(
        json.dumps(request_metadata, indent=2) + "\n"
    )

    payload = {
        "model": model,
        "prompt": prompt.tolist(),
        "max_tokens": args.output_len,
        "min_tokens": args.output_len,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": args.seed,
        "ignore_eos": True,
        "stream": False,
        "add_special_tokens": False,
    }
    print(
        f"request: model={model} input={args.input_len} output={args.output_len} "
        f"seed={args.seed}",
        flush=True,
    )
    started = time.perf_counter()
    response = request_json(
        f"{base_url}/v1/completions", payload, args.timeout_sec
    )
    elapsed = time.perf_counter() - started

    usage = response.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", -1))
    completion_tokens = int(usage.get("completion_tokens", -1))
    result = {
        **request_metadata,
        "elapsed_sec": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": response["choices"][0].get("finish_reason"),
    }
    (args.capture_dir / "request_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)

    captures = sorted(args.capture_dir.glob("capture_*/metadata.json"))
    print(f"captures={len(captures)} under {args.capture_dir}", flush=True)
    for path in captures:
        metadata = json.loads(path.read_text())
        print(
            f"  {path.parent.name}: T={metadata['query_tokens']} "
            f"context={metadata['max_context']} shape={metadata['query_shape']}",
            flush=True,
        )

    if prompt_tokens != args.input_len:
        raise RuntimeError(
            f"prompt token count is {prompt_tokens}, expected {args.input_len}"
        )
    if completion_tokens != args.output_len:
        raise RuntimeError(
            f"completion token count is {completion_tokens}, "
            f"expected {args.output_len}"
        )
    return 0 if captures else 2


if __name__ == "__main__":
    raise SystemExit(main())
