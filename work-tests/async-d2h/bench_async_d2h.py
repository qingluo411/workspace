#!/usr/bin/env python3
"""Measure D2H submit latency and completion latency for one path group."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

torch: Any = None


def load_torch() -> None:
    global torch
    import torch as torch_module  # noqa: PLC0415

    torch = torch_module


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((pct / 100.0) * (len(ordered) - 1))
    return ordered[max(0, min(index, len(ordered) - 1))]


def make_expected(size: int, dtype: torch.dtype) -> Any:
    base = torch.arange(size, dtype=torch.int64).remainder(251)
    if dtype == torch.bool:
        return base.remainder(2).to(torch.bool).reshape(size, 1)
    return base.to(dtype).reshape(size, 1)


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("async", "fallback", "o2"), default=None)
    parser.add_argument(
        "--dtype",
        default="int32",
        choices=(
            "int8",
            "uint8",
            "int16",
            "int32",
            "int64",
            "float16",
            "bfloat16",
            "float32",
        ),
    )
    parser.add_argument(
        "--sizes",
        default="1,16,64,127,128,129,256,257,1024,16384,16385,65536",
        help="Comma-separated logical element counts; tensors are [N,1]",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    load_torch()
    if not torch.dlc.is_available():
        raise SystemExit("torch.dlc is not available")

    mode = args.mode
    if mode is None:
        if os.environ.get("DLC_DISABLE_DEVICE_D2H_LAYOUT") == "1":
            mode = "fallback"
        elif os.environ.get("DLC_SYN_COPY_ASYNC", "O3") in ("O0", "O1", "O2"):
            mode = "o2"
        else:
            mode = "async"

    dtype = getattr(torch, args.dtype)
    sizes = [int(item) for item in args.sizes.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []

    for size in sizes:
        expected = make_expected(size, dtype)
        source = expected.to("dlc")
        torch.dlc.synchronize()

        for _ in range(max(0, args.warmup)):
            warm = source.to("cpu", non_blocking=True)
            event = torch.dlc.Event()
            event.record()
            event.synchronize()
            assert torch.equal(warm, expected), f"warmup mismatch at size={size}"

        for repeat in range(args.repeats):
            started = time.perf_counter_ns()
            result = source.to("cpu", non_blocking=True)
            submitted = time.perf_counter_ns()

            event = torch.dlc.Event()
            event.record()
            pending_at_return = not event.query()
            wait_started = time.perf_counter_ns()
            event.synchronize()
            finished = time.perf_counter_ns()

            data_ok = torch.equal(result, expected)
            rows.append(
                {
                    "mode": mode,
                    "dtype": args.dtype,
                    "size": size,
                    "repeat": repeat,
                    "submit_ms": (submitted - started) / 1_000_000,
                    "event_wait_ms": (finished - wait_started) / 1_000_000,
                    "total_ms": (finished - started) / 1_000_000,
                    "pending_at_return": pending_at_return,
                    "data_ok": bool(data_ok),
                    "storage_bytes": result.untyped_storage().nbytes(),
                    "logical_bytes": result.numel() * result.element_size(),
                    "pinned": bool(result.is_pinned()),
                }
            )
            assert data_ok, f"data mismatch at size={size}, repeat={repeat}"

        del source, expected

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["size"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for size, items in groups.items():
        summary.append(
            {
                "size": int(size),
                "dtype": args.dtype,
                "mode": mode,
                "repeats": len(items),
                "submit_ms": summarize([float(x["submit_ms"]) for x in items]),
                "event_wait_ms": summarize([float(x["event_wait_ms"]) for x in items]),
                "total_ms": summarize([float(x["total_ms"]) for x in items]),
                "pending_at_return": sum(
                    int(bool(x["pending_at_return"])) for x in items
                ),
                "data_ok": all(bool(x["data_ok"]) for x in items),
            }
        )

    payload = {
        "mode": mode,
        "dtype": args.dtype,
        "sizes": sizes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "env": {
            "DLC_SYN_COPY_ASYNC": os.environ.get("DLC_SYN_COPY_ASYNC"),
            "DLC_DISABLE_DEVICE_D2H_LAYOUT": os.environ.get(
                "DLC_DISABLE_DEVICE_D2H_LAYOUT"
            ),
            "DLC_SYN_USE_SIM": os.environ.get("DLC_SYN_USE_SIM"),
        },
        "summary": summary,
        "raw": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"WROTE {args.output}", flush=True)
    print(f"WROTE {csv_path}", flush=True)
    for item in summary:
        submit = item["submit_ms"]
        wait = item["event_wait_ms"]
        print(
            f"{mode} {args.dtype} N={item['size']}: "
            f"submit_median={submit['median']:.4f} ms, "
            f"wait_median={wait['median']:.4f} ms, "
            f"pending={item['pending_at_return']}/{item['repeats']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
