#!/usr/bin/env python3
"""Collect and compare deterministic Qwen3 KV-cache accuracy probes."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any


GENERATION_PROMPTS = [
    "The capital of France is",
    "If a train travels 60 miles in 1.5 hours, its average speed in miles per hour is",
    "Compute 17 * 23. The answer is",
    "请用一句话解释什么是 KV cache：",
    "Complete the Python function using a loop:\n\ndef total(numbers):\n    result = 0\n",
    "A store has 48 apples and sells one quarter of them. It has",
    "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius. At 50 degrees Celsius, water is",
    "Answer with only the final number: 9 + 8 * 7 =",
]

LOGPROB_DOCUMENTS = [
    (
        "KV cache stores the key and value tensors produced by previous tokens so "
        "that autoregressive inference does not recompute the full attention context "
        "for every decode step. Quantizing these tensors reduces memory traffic and "
        "increases the number of resident tokens, but the approximation must remain "
        "accurate enough for the model to preserve its predictions."
    ),
    (
        "A distributed tensor-parallel service partitions attention heads across "
        "accelerators. Each rank owns a slice of the KV heads and therefore a slice "
        "of the cache. Correct block bookkeeping is essential: every request maps its "
        "logical context positions to physical pages, and those pages must be reused "
        "only after every rank has finished reading them."
    ),
    (
        "在长上下文推理中，预填充和解码都会访问历史键值。缓存布局既要满足硬件的"
        "对齐要求，也要保持每页的容量可精确计算。数值量化、页内偏移、跨页写入和"
        "空槽映射必须一起验证，不能只检查平均误差。"
    ),
]


def request_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def generate(base_url: str, model: str, prompt: str, timeout: float) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 32,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "stream": False,
        "return_token_ids": True,
    }
    started = time.perf_counter()
    result = request_json(f"{base_url}/v1/completions", payload, timeout)
    choice = result["choices"][0]
    return {
        "prompt": prompt,
        "text": choice["text"],
        "token_ids": choice.get("token_ids"),
        "finish_reason": choice.get("finish_reason"),
        "elapsed_seconds": time.perf_counter() - started,
    }


def prompt_logprobs(
    base_url: str, model: str, document: str, timeout: float
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": document,
        "max_tokens": 1,
        "temperature": 0.0,
        "echo": True,
        "prompt_logprobs": 0,
        "stream": False,
    }
    result = request_json(f"{base_url}/v1/completions", payload, timeout)
    choice = result["choices"][0]
    entries = choice.get("prompt_logprobs")
    if entries is None:
        raise RuntimeError("server did not return prompt_logprobs")

    token_logprobs: list[dict[str, Any]] = []
    for position, entry in enumerate(entries):
        if entry is None or not entry:
            continue
        token_id, details = next(iter(entry.items()))
        token_logprobs.append(
            {
                "position": position,
                "token_id": int(token_id),
                "logprob": float(details["logprob"]),
                "decoded_token": details.get("decoded_token"),
            }
        )
    mean_nll = -math.fsum(item["logprob"] for item in token_logprobs) / len(
        token_logprobs
    )
    return {
        "document": document,
        "num_scored_tokens": len(token_logprobs),
        "mean_nll": mean_nll,
        "mean_perplexity": math.exp(mean_nll),
        "tokens": token_logprobs,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    generations = []
    for prompt in GENERATION_PROMPTS:
        generation = generate(args.base_url, args.model, prompt, args.timeout)
        generations.append(generation)
        print(
            f"generated {len(generation['text'])} chars in "
            f"{generation['elapsed_seconds']:.3f}s: {prompt[:48]!r}"
        )

    likelihoods = []
    for document in LOGPROB_DOCUMENTS:
        likelihood = prompt_logprobs(args.base_url, args.model, document, args.timeout)
        likelihoods.append(likelihood)
        print(
            f"scored {likelihood['num_scored_tokens']} tokens: "
            f"nll={likelihood['mean_nll']:.6f}"
        )

    return {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "generations": generations,
        "prompt_likelihoods": likelihoods,
    }


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    generation_rows = []
    for expected, actual in zip(
        reference["generations"], candidate["generations"], strict=True
    ):
        expected_ids = expected.get("token_ids")
        actual_ids = actual.get("token_ids")
        if expected_ids is not None and actual_ids is not None:
            common = min(len(expected_ids), len(actual_ids))
            first_difference = next(
                (
                    index
                    for index in range(common)
                    if expected_ids[index] != actual_ids[index]
                ),
                common,
            )
            token_ids_equal = expected_ids == actual_ids
        else:
            first_difference = None
            token_ids_equal = expected["text"] == actual["text"]
        generation_rows.append(
            {
                "prompt": expected["prompt"],
                "reference_text": expected["text"],
                "candidate_text": actual["text"],
                "text_equal": expected["text"] == actual["text"],
                "token_ids_equal": token_ids_equal,
                "first_token_difference": first_difference,
            }
        )

    likelihood_rows = []
    for expected, actual in zip(
        reference["prompt_likelihoods"],
        candidate["prompt_likelihoods"],
        strict=True,
    ):
        expected_tokens = expected["tokens"]
        actual_tokens = actual["tokens"]
        if len(expected_tokens) != len(actual_tokens):
            raise RuntimeError("prompt token counts differ between runs")
        position_deltas = []
        absolute_deltas = []
        for expected_token, actual_token in zip(
            expected_tokens, actual_tokens, strict=True
        ):
            if expected_token["token_id"] != actual_token["token_id"]:
                raise RuntimeError("prompt tokenization differs between runs")
            delta = actual_token["logprob"] - expected_token["logprob"]
            position_deltas.append(delta)
            absolute_deltas.append(abs(delta))
        likelihood_rows.append(
            {
                "num_scored_tokens": len(position_deltas),
                "reference_mean_nll": expected["mean_nll"],
                "candidate_mean_nll": actual["mean_nll"],
                "mean_nll_delta": actual["mean_nll"] - expected["mean_nll"],
                "mean_abs_logprob_delta": math.fsum(absolute_deltas)
                / len(absolute_deltas),
                "max_abs_logprob_delta": max(absolute_deltas),
                "mean_logprob_delta": math.fsum(position_deltas)
                / len(position_deltas),
            }
        )

    return {
        "reference_label": reference["label"],
        "candidate_label": candidate["label"],
        "generation_rows": generation_rows,
        "prompt_likelihood_rows": likelihood_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18005")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    args = parser.parse_args()

    result = collect(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
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
        equal_count = sum(row["token_ids_equal"] for row in comparison["generation_rows"])
        print(
            f"exact token-id matches: {equal_count}/"
            f"{len(comparison['generation_rows'])}"
        )
        for row in comparison["prompt_likelihood_rows"]:
            print(
                f"NLL delta={row['mean_nll_delta']:+.6f}, "
                f"mean |logprob delta|={row['mean_abs_logprob_delta']:.6f}, "
                f"max={row['max_abs_logprob_delta']:.6f}"
            )
        print(f"wrote {comparison_output}")


if __name__ == "__main__":
    main()
