#!/usr/bin/env python3
"""Run one token-exact long-context KV-cache probe and compare two runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


FILLER = (
    "The maintenance log records coolant pressure, clock stability, memory "
    "training status, and link health for the accelerator cluster. Operators "
    "compare each reading with the previous shift and investigate any sudden "
    "change before scheduling production traffic. Routine checks are repeated "
    "so that the long prompt exercises many physical cache pages without "
    "depending on sampling variability. "
)


def make_prompt(
    tokenizer: Any, token_count: int, needle: str, question: str
) -> dict[str, Any]:
    prefix_ids = tokenizer.encode(
        "Read the following engineering log carefully.\n", add_special_tokens=False
    )
    needle_ids = tokenizer.encode(f"\nIMPORTANT RECORD: {needle}\n", add_special_tokens=False)
    suffix_ids = tokenizer.encode(f"\nQuestion: {question}\nAnswer:", add_special_tokens=False)
    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    remaining = token_count - len(prefix_ids) - len(needle_ids) - len(suffix_ids)
    if remaining < 0:
        raise ValueError("token_count is too small for the prefix and suffix")

    repeated = (filler_ids * (remaining // len(filler_ids) + 1))[:remaining]
    split = len(repeated) // 2
    prompt_ids = prefix_ids + repeated[:split] + needle_ids + repeated[split:] + suffix_ids
    if len(prompt_ids) != token_count:
        raise AssertionError((len(prompt_ids), token_count))
    return {
        "prompt_ids": prompt_ids,
        "needle": needle,
        "question": question,
    }


def request_completion(
    base_url: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "stream": False,
        "echo": True,
        "prompt_logprobs": 0,
        "return_token_ids": True,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def summarize_logprobs(
    prompt_ids: list[int], entries: list[Any]
) -> tuple[list[dict[str, Any]], float]:
    if len(entries) != len(prompt_ids):
        raise RuntimeError(
            f"expected {len(prompt_ids)} prompt-logprob entries, got {len(entries)}"
        )
    rows = []
    for position, (expected_token, entry) in enumerate(zip(prompt_ids, entries, strict=True)):
        if entry is None or not entry:
            continue
        token_id, details = next(iter(entry.items()))
        if int(token_id) != expected_token:
            raise RuntimeError(
                f"prompt-logprob token mismatch at {position}: "
                f"{token_id} != {expected_token}"
            )
        rows.append(
            {
                "position": position,
                "token_id": int(token_id),
                "logprob": float(details["logprob"]),
                "rank": int(details["rank"]),
            }
        )
    if not rows:
        raise RuntimeError("server returned no prompt logprobs")
    mean_nll = -math.fsum(row["logprob"] for row in rows) / len(rows)
    return rows, mean_nll


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_ids = reference["completion_token_ids"] or []
    candidate_ids = candidate["completion_token_ids"] or []
    common = min(len(reference_ids), len(candidate_ids))
    first_difference = next(
        (index for index in range(common) if reference_ids[index] != candidate_ids[index]),
        common,
    )

    reference_tokens = reference["prompt_logprob_tokens"]
    candidate_tokens = candidate["prompt_logprob_tokens"]
    if len(reference_tokens) != len(candidate_tokens):
        raise RuntimeError("scored prompt positions differ between runs")
    absolute_deltas = []
    signed_deltas = []
    for expected, actual in zip(reference_tokens, candidate_tokens, strict=True):
        if expected["position"] != actual["position"] or expected["token_id"] != actual["token_id"]:
            raise RuntimeError("prompt token alignment differs between runs")
        delta = actual["logprob"] - expected["logprob"]
        signed_deltas.append(delta)
        absolute_deltas.append(abs(delta))

    return {
        "reference_label": reference["label"],
        "candidate_label": candidate["label"],
        "token_count": reference["token_count"],
        "response_text_equal": reference["response_text"] == candidate["response_text"],
        "completion_token_ids_equal": reference_ids == candidate_ids,
        "first_completion_token_difference": first_difference,
        "reference_response": reference["response_text"],
        "candidate_response": candidate["response_text"],
        "reference_mean_nll": reference["mean_prompt_nll"],
        "candidate_mean_nll": candidate["mean_prompt_nll"],
        "mean_nll_delta": candidate["mean_prompt_nll"] - reference["mean_prompt_nll"],
        "mean_abs_logprob_delta": math.fsum(absolute_deltas) / len(absolute_deltas),
        "mean_logprob_delta": math.fsum(signed_deltas) / len(signed_deltas),
        "max_abs_logprob_delta": max(absolute_deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18005")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", default="/data/models/Qwen3-32B-Mix")
    parser.add_argument("--label", required=True)
    parser.add_argument("--token-count", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt = make_prompt(
        tokenizer,
        args.token_count,
        needle="The secret access code is ORANGE-7319.",
        question="What is the secret access code? Answer with only the code.",
    )
    print(f"built prompt with {len(prompt['prompt_ids'])} tokens")
    response = request_completion(
        args.base_url,
        args.model,
        prompt["prompt_ids"],
        args.max_tokens,
        args.timeout,
    )
    choice = response["choices"][0]
    prompt_logprob_tokens, mean_prompt_nll = summarize_logprobs(
        prompt["prompt_ids"], choice["prompt_logprobs"]
    )
    result = {
        "label": args.label,
        "model": args.model,
        "token_count": args.token_count,
        "needle": prompt["needle"],
        "question": prompt["question"],
        "prompt_ids_sha256": hashlib.sha256(
            json.dumps(prompt["prompt_ids"]).encode()
        ).hexdigest(),
        "response_text": choice["text"],
        "completion_token_ids": choice.get("token_ids"),
        "finish_reason": choice.get("finish_reason"),
        "elapsed_seconds": response["elapsed_seconds"],
        "num_scored_prompt_tokens": len(prompt_logprob_tokens),
        "mean_prompt_nll": mean_prompt_nll,
        "mean_prompt_perplexity": math.exp(mean_prompt_nll),
        "prompt_logprob_tokens": prompt_logprob_tokens,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        f"elapsed={result['elapsed_seconds']:.2f}s nll={mean_prompt_nll:.6f} "
        f"response={result['response_text']!r}"
    )
    print(f"wrote {args.output}")

    if args.reference is not None:
        reference = json.loads(args.reference.read_text())
        comparison = compare(reference, result)
        comparison_output = args.comparison_output or args.output.with_name(
            args.output.stem + "_comparison.json"
        )
        comparison_output.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        print(f"wrote {comparison_output}")


if __name__ == "__main__":
    main()
