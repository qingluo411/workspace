#!/usr/bin/env python3
"""Supplemental async-D2H contract and lifetime checks.

The PyTorch test already covers the main copy matrix. This script adds the
checks called out by work-docs/async-d2h/test.md that are awkward to infer
from pass/fail alone:

* controlled backlog: submit vs completion is observable;
* device allocation reuse after source release;
* pinned host allocation reuse after output release;
* non-default stream dependency;
* fallback output contracts (A) and O2 output contracts (C).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

torch: Any = None
USING_SIMULATOR = os.environ.get("DLC_SYN_USE_SIM", "0").lower() in ("1", "true")


def load_torch() -> None:
    global torch
    import torch as torch_module  # noqa: PLC0415

    torch = torch_module


def wait_current_stream() -> None:
    event = torch.dlc.Event()
    event.record()
    event.synchronize()


def _result_contract(result: Any) -> dict[str, Any]:
    logical_bytes = result.numel() * result.element_size()
    return {
        "shape": list(result.shape),
        "dtype": str(result.dtype),
        "logical_bytes": logical_bytes,
        "storage_bytes": result.untyped_storage().nbytes(),
        "pinned": bool(result.is_pinned()),
    }


def check_basic(mode: str) -> dict[str, Any]:
    expected = torch.arange(257, dtype=torch.int32).reshape(257, 1)
    source = expected.to("dlc")
    torch.dlc.synchronize()
    result = source.to("cpu", non_blocking=True)
    event = torch.dlc.Event()
    event.record()
    event.synchronize()

    assert torch.equal(result, expected), "basic sampled-token copy mismatch"
    contract = _result_contract(result)
    if mode in ("async", "fallback"):
        assert contract["pinned"], "O3 output should be pinned"
        padded = (contract["logical_bytes"] + 511) // 512 * 512
        assert contract["storage_bytes"] == padded, "O3 output should be padded"
    else:
        assert not contract["pinned"], "O2 output should not be auto-pinned"
        assert contract["storage_bytes"] == contract["logical_bytes"], (
            "O2 output should have exact logical storage"
        )
    return contract


def check_large_async_contract(mode: str, elements: int) -> dict[str, Any]:
    if USING_SIMULATOR:
        return {"skipped": "simulator executes synchronously"}
    if mode == "o2":
        return {"skipped": "O2 is a strategy comparison, not async D2H"}

    expected = torch.arange(elements, dtype=torch.int32)
    source = expected.to("dlc")
    torch.dlc.synchronize()

    started = time.perf_counter_ns()
    result = source.to("cpu", non_blocking=True)
    submitted = time.perf_counter_ns()
    event = torch.dlc.Event()
    event.record()
    pending_after_submit = not event.query()
    event.synchronize()
    finished = time.perf_counter_ns()

    assert torch.equal(result, expected), "large async copy mismatch"
    if mode == "async":
        assert pending_after_submit, "new path did not return before completion"

    return {
        "expected_async": mode == "async",
        "elements": elements,
        "submit_ms": (submitted - started) / 1_000_000,
        "event_wait_ms": (finished - submitted) / 1_000_000,
        "pending_after_submit": pending_after_submit,
        "contract": _result_contract(result),
    }


def check_fallback_cases() -> dict[str, Any]:
    base = torch.arange(1025, dtype=torch.int32)
    cases = {
        "slice": (base, lambda x: x[7:800:3]),
        "transpose": (
            torch.arange(17 * 19, dtype=torch.int64).reshape(17, 19),
            lambda x: x.t(),
        ),
    }
    observed: dict[str, Any] = {}
    for name, (cpu_base, view) in cases.items():
        expected = view(cpu_base)
        source = view(cpu_base.to("dlc"))
        result = source.to("cpu", non_blocking=True)
        event = torch.dlc.Event()
        event.record()
        event.synchronize()
        assert torch.equal(result, expected), f"{name}: data mismatch"
        observed[name] = _result_contract(result)
    return observed


def check_device_lifetime(rounds: int) -> dict[str, Any]:
    expected = torch.arange(513, dtype=torch.int64).reshape(513, 1)
    for index in range(rounds):
        source = expected.to("dlc")
        torch.dlc.synchronize()
        result = source.to("cpu", non_blocking=True)
        event = torch.dlc.Event()
        event.record()
        del source
        pressure = torch.empty_like(expected, device="dlc")
        pressure.fill_(index + 1)
        event.synchronize()
        assert torch.equal(result, expected), f"device lifetime round {index} failed"
        del pressure, result
        gc.collect()
    return {"rounds": rounds, "status": "ok"}


def check_host_lifetime(rounds: int) -> dict[str, Any]:
    expected = torch.arange(257, dtype=torch.int32).reshape(257, 1)
    source = expected.to("dlc")
    torch.dlc.synchronize()
    reuse_hits = 0
    pending_hits = 0

    for index in range(rounds):
        result = source.to("cpu", non_blocking=True)
        event = torch.dlc.Event()
        event.record()
        old_ptr = result.data_ptr()
        storage_bytes = result.untyped_storage().nbytes()
        pending_hits += int(not event.query())
        del result
        gc.collect()

        reuse = torch.full(
            (storage_bytes,), 0xA5, dtype=torch.uint8, pin_memory=True
        )
        reuse_hits += int(reuse.data_ptr() == old_ptr)
        event.synchronize()
        assert bool((reuse == 0xA5).all()), (
            f"host lifetime round {index} was overwritten by pending DMA"
        )
        del reuse
        gc.collect()

    return {
        "rounds": rounds,
        "pending_hits": pending_hits,
        "pointer_reuse_hits": reuse_hits,
        "status": "ok",
    }


def check_stream_dependency() -> dict[str, Any]:
    expected = torch.arange(4097, dtype=torch.int32)
    source = expected.to("dlc")
    torch.dlc.synchronize()
    source.add_(17)

    main_stream = torch.dlc.current_stream()
    copy_stream = torch.dlc.Stream()
    done = torch.dlc.Event()
    with torch.dlc.stream(copy_stream):
        copy_stream.wait_stream(main_stream)
        result = source.to("cpu", non_blocking=True)
        done.record()
    done.synchronize()

    expected.add_(17)
    assert torch.equal(result, expected), "producer -> copy stream dependency failed"
    return {"elements": expected.numel(), "status": "ok"}


def run_check(
    name: str,
    fn: Callable[[], Any],
    results: list[dict[str, Any]],
) -> None:
    print(f"START {name}", flush=True)
    started = time.perf_counter()
    try:
        details = fn()
    except Exception as exc:  # noqa: BLE001 - collect every failure
        results.append(
            {
                "name": name,
                "status": "FAIL",
                "elapsed_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
        return
    results.append(
        {
            "name": name,
            "status": "PASS",
            "elapsed_s": time.perf_counter() - started,
            "details": details,
        }
    )
    print(f"PASS {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("async", "fallback", "o2"),
        default=None,
        help="Expected path; defaults from DLC_SYN_COPY_ASYNC/DLC_DISABLE_DEVICE_D2H_LAYOUT",
    )
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--large-elements", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--skip-large", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
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

    results: list[dict[str, Any]] = []
    run_check("basic_contract", lambda: check_basic(mode), results)
    run_check("fallback_views", check_fallback_cases, results)
    run_check("device_lifetime", lambda: check_device_lifetime(args.rounds), results)
    if mode != "o2":
        run_check(
            "host_lifetime", lambda: check_host_lifetime(args.rounds), results
        )
        run_check("stream_dependency", check_stream_dependency, results)
    if not args.skip_large:
        run_check(
            "controlled_backlog",
            lambda: check_large_async_contract(mode, args.large_elements),
            results,
        )

    failed = [item for item in results if item["status"] == "FAIL"]
    payload = {
        "mode": mode,
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "env": {
            "DLC_SYN_COPY_ASYNC": os.environ.get("DLC_SYN_COPY_ASYNC"),
            "DLC_DISABLE_DEVICE_D2H_LAYOUT": os.environ.get(
                "DLC_DISABLE_DEVICE_D2H_LAYOUT"
            ),
            "DLC_SYN_USE_SIM": os.environ.get("DLC_SYN_USE_SIM"),
        },
        "checks": results,
        "passed": len(results) - len(failed),
        "failed": len(failed),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    print(text, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
