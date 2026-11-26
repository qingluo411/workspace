#!/usr/bin/env python3
"""Compare exact runtime hidden-state and logits hashes between two requests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


RUNTIME_MODULES = ("hidden_states", "sample_hidden_states", "logits")


def load_runtime_hashes(root: Path, request_substring: str):
    records = {}
    relative_by_request = Counter()
    iteration_dirs = sorted(
        root.glob("iter_*"),
        key=lambda path: int(path.name.removeprefix("iter_")),
    )
    for iteration_dir in iteration_dirs:
        meta_path = iteration_dir / "meta.json"
        hashes_path = iteration_dir / "hashes.jsonl"
        if not meta_path.is_file() or not hashes_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        request_ids = tuple(sorted(meta.get("request_ids", [])))
        matching_ids = tuple(
            request_id
            for request_id in request_ids
            if request_substring in request_id
        )
        if not matching_ids:
            continue
        request_key = matching_ids
        relative_iteration = relative_by_request[request_key]
        relative_by_request[request_key] += 1

        for line in hashes_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            record = json.loads(line)
            if record.get("module") not in RUNTIME_MODULES:
                continue
            record_ids = record.get("request_ids") or []
            if not any(request_substring in item for item in record_ids):
                continue
            key = (
                relative_iteration,
                record["module"],
                record.get("path", ""),
            )
            records[key] = record
    return records, relative_by_request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--left-request-substring", required=True)
    parser.add_argument("--right-request-substring", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    left, left_requests = load_runtime_hashes(
        args.dump_dir, args.left_request_substring
    )
    right, right_requests = load_runtime_hashes(
        args.dump_dir, args.right_request_substring
    )

    mismatches = []
    for key in sorted(set(left) | set(right)):
        left_record = left.get(key)
        right_record = right.get(key)
        if left_record is None or right_record is None:
            mismatches.append(
                {
                    "key": key,
                    "reason": "missing",
                    "left": left_record,
                    "right": right_record,
                }
            )
            continue
        fields = ("hash", "dtype", "shape", "numel")
        if any(left_record.get(field) != right_record.get(field) for field in fields):
            mismatches.append(
                {
                    "key": key,
                    "reason": "value_mismatch",
                    "left": left_record,
                    "right": right_record,
                }
            )

    summary = {
        "dump_dir": str(args.dump_dir),
        "left_request_substring": args.left_request_substring,
        "right_request_substring": args.right_request_substring,
        "left_request_counts": {
            str(key): value for key, value in left_requests.items()
        },
        "right_request_counts": {
            str(key): value for key, value in right_requests.items()
        },
        "left_records": len(left),
        "right_records": len(right),
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "all_runtime_logits_same": not mismatches,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {**summary, "mismatches": mismatches[:100]},
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
