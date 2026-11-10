#!/usr/bin/env python3
"""Run pytorch/test/dlc_async_d2h_copy.py with a fallback-mode wrapper.

Group A intentionally sets DLC_DISABLE_DEVICE_D2H_LAYOUT=1 while keeping O3.
In that group the padded/pinned allocation checks still apply, but the
"event must still be pending immediately after .to()" check must be skipped:
the copy is expected to be synchronous by construction.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("dlc_async_d2h_copy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()

    if not args.test_file.is_file():
        raise SystemExit(f"test file not found: {args.test_file}")

    os.environ.setdefault("DLC_SYN_COPY_ASYNC", "O3")
    module = load_module(args.test_file)

    if args.fallback:
        def skip_deferred_check(name: str, expected: object) -> None:
            print(
                "SKIP deferred D2H "
                f"{name}: fallback mode intentionally completes synchronously",
                flush=True,
            )

        module.check_copy_is_deferred_to_event = skip_deferred_check
        print(
            "INFO fallback wrapper: skip only the event-pending assertion; "
            "all data/layout/lifetime checks still run",
            flush=True,
        )

    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
