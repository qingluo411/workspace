#!/usr/bin/env python3
"""Compare two hash-mode model dumps and report the first divergent tensor."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _matches(record: dict, request_substring: str | None) -> bool:
    if request_substring is None:
        return True
    return any(
        request_substring in request_id
        for request_id in record.get("request_ids", [])
    )


def load_dump(root: Path, request_substring: str | None):
    tensor_records = {}
    input_records = {}
    relative_by_request = Counter()
    iter_dirs = sorted(
        root.glob("iter_*"),
        key=lambda path: int(path.name.removeprefix("iter_")),
    )
    for iter_dir in iter_dirs:
        meta_path = iter_dir / "meta.json"
        if not meta_path.is_file():
            continue
        meta = _load_json(meta_path)
        request_ids = tuple(sorted(meta.get("request_ids", [])))
        matching_ids = tuple(
            request_id
            for request_id in request_ids
            if request_substring is None or request_substring in request_id
        )
        if request_substring is not None and not matching_ids:
            continue
        request_key = matching_ids or request_ids or ("__no_request__",)
        relative_iteration = relative_by_request[request_key]
        relative_by_request[request_key] += 1

        hash_path = iter_dir / "hashes.jsonl"
        if hash_path.is_file():
            for line in hash_path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                record = json.loads(line)
                if record.get("request_ids") and not _matches(
                    record, request_substring
                ):
                    continue
                key = (
                    relative_iteration,
                    record["call"],
                    record["module"],
                    record["path"],
                )
                tensor_records[key] = record

        input_path = iter_dir / "input_hashes.json"
        if input_path.is_file():
            for record in _load_json(input_path):
                if record.get("request_ids") and not _matches(
                    record, request_substring
                ):
                    continue
                key = (relative_iteration, record["name"])
                input_records[key] = record
    return tensor_records, input_records


def compare_records(left: dict, right: dict, label: str):
    mismatches = []
    for key in sorted(set(left) | set(right)):
        left_record = left.get(key)
        right_record = right.get(key)
        if left_record is None or right_record is None:
            mismatches.append(
                {
                    "kind": label,
                    "key": key,
                    "left": left_record,
                    "right": right_record,
                    "reason": "missing",
                }
            )
            continue
        fields = ("hash", "dtype", "shape", "numel")
        if any(left_record.get(field) != right_record.get(field) for field in fields):
            mismatches.append(
                {
                    "kind": label,
                    "key": key,
                    "left": left_record,
                    "right": right_record,
                    "reason": "value_mismatch",
                }
            )
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-request-substring")
    parser.add_argument("--right-request-substring")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    left_tensors, left_inputs = load_dump(
        args.left, args.left_request_substring
    )
    right_tensors, right_inputs = load_dump(
        args.right, args.right_request_substring
    )
    mismatches = compare_records(left_tensors, right_tensors, "module_output")
    mismatches.extend(compare_records(left_inputs, right_inputs, "model_input"))
    mismatches.sort(
        key=lambda item: (
            item["key"][0],
            item["kind"] != "model_input",
            item["key"],
        )
    )

    summary = {
        "left": str(args.left),
        "right": str(args.right),
        "left_tensor_records": len(left_tensors),
        "right_tensor_records": len(right_tensors),
        "left_input_records": len(left_inputs),
        "right_input_records": len(right_inputs),
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "first_mismatch_iterations": Counter(
            mismatch["key"][0] for mismatch in mismatches
        ).most_common(20),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {**summary, "mismatches": mismatches[:1000]},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
