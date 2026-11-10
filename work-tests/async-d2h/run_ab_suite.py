#!/usr/bin/env python3
"""Run the async-D2H A/B/C suite described by work-docs/async-d2h/test.md."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = WORKSPACE / "work-tests" / "async-d2h" / "results"
GROUP_MODES = {"A": "fallback", "B": "async", "C": "o2"}


def group_env(group: str) -> dict[str, str]:
    env = os.environ.copy()
    if group == "A":
        env["DLC_SYN_COPY_ASYNC"] = "O3"
        env["DLC_DISABLE_DEVICE_D2H_LAYOUT"] = "1"
    elif group == "B":
        env["DLC_SYN_COPY_ASYNC"] = "O3"
        env.pop("DLC_DISABLE_DEVICE_D2H_LAYOUT", None)
    elif group == "C":
        env["DLC_SYN_COPY_ASYNC"] = "O2"
        env.pop("DLC_DISABLE_DEVICE_D2H_LAYOUT", None)
    else:
        raise ValueError(f"unknown group: {group}")
    return env


def marker_counts(log_path: Path) -> dict[str, int]:
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return counts
    for line in lines:
        for marker in counts:
            if line.startswith(marker + " "):
                counts[marker] += 1
                break
    return counts


def run_command(
    label: str,
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    timeout: int,
    dry_run: bool,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] {' '.join(cmd)}", flush=True)
    if dry_run:
        return {
            "label": label,
            "status": "DRY_RUN",
            "command": cmd,
            "log": str(log_path),
        }

    started = time.monotonic()
    error: str | None = None
    returncode = -1
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(
                cmd,
                cwd=WORKSPACE,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
            returncode = proc.returncode
    except subprocess.TimeoutExpired:
        error = f"timed out after {timeout}s"
    except OSError as exc:
        error = f"failed to launch: {exc}"

    record = {
        "label": label,
        "status": "PASS" if returncode == 0 and error is None else "FAIL",
        "returncode": returncode,
        "elapsed_s": time.monotonic() - started,
        "command": cmd,
        "log": str(log_path),
        "markers": marker_counts(log_path),
    }
    if error is not None:
        record["error"] = error
    print(
        f"[{label}] {record['status']} rc={returncode} "
        f"elapsed={record['elapsed_s']:.2f}s",
        flush=True,
    )
    return record


def script_path(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--groups", default="A,B", help="Comma-separated A,B,C")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-supplemental", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.result_root.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "capture": workspace / "work-tests" / "async-d2h" / "capture_env.py",
        "existing": workspace / "pytorch" / "test" / "dlc_async_d2h_copy.py",
        "wrapper": workspace / "work-tests" / "async-d2h" / "run_existing_async_test.py",
        "supplemental": workspace / "work-tests" / "async-d2h" / "supplemental_async_d2h.py",
        "bench": workspace / "work-tests" / "async-d2h" / "bench_async_d2h.py",
    }

    groups = [item.strip().upper() for item in args.groups.split(",") if item.strip()]
    invalid = [group for group in groups if group not in GROUP_MODES]
    if invalid:
        raise SystemExit(f"unknown groups: {invalid}")

    suite: dict[str, Any] = {
        "run_name": run_name,
        "started_at": datetime.now().astimezone().isoformat(),
        "workspace": str(workspace),
        "python": args.python,
        "groups": {},
    }

    for group in groups:
        mode = GROUP_MODES[group]
        group_dir = run_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        env = group_env(group)
        records: list[dict[str, Any]] = []

        capture_cmd = [
            args.python,
            script_path(paths["capture"]),
            "--output",
            str(group_dir / "env.json"),
        ]
        records.append(
            run_command(
                f"{group}/env",
                capture_cmd,
                env,
                group_dir / "env.log",
                args.timeout,
                args.dry_run,
            )
        )

        if not args.skip_existing:
            if group == "A":
                cmd = [
                    args.python,
                    script_path(paths["wrapper"]),
                    "--test-file",
                    script_path(paths["existing"]),
                    "--fallback",
                ]
            else:
                cmd = [args.python, script_path(paths["existing"])]
            records.append(
                run_command(
                    f"{group}/existing",
                    cmd,
                    env,
                    group_dir / "existing.log",
                    args.timeout,
                    args.dry_run,
                )
            )

        if not args.skip_supplemental:
            cmd = [
                args.python,
                script_path(paths["supplemental"]),
                "--mode",
                mode,
                "--rounds",
                "32",
                "--output",
                str(group_dir / "supplemental.json"),
            ]
            records.append(
                run_command(
                    f"{group}/supplemental",
                    cmd,
                    env,
                    group_dir / "supplemental.log",
                    args.timeout,
                    args.dry_run,
                )
            )

        if not args.skip_bench:
            cmd = [
                args.python,
                script_path(paths["bench"]),
                "--mode",
                mode,
                "--repeats",
                str(args.repeats),
                "--warmup",
                str(args.warmup),
                "--output",
                str(group_dir / "micro.json"),
            ]
            records.append(
                run_command(
                    f"{group}/micro",
                    cmd,
                    env,
                    group_dir / "micro.log",
                    args.timeout,
                    args.dry_run,
                )
            )

        suite["groups"][group] = {
            "mode": mode,
            "env": {
                "DLC_SYN_COPY_ASYNC": env.get("DLC_SYN_COPY_ASYNC"),
                "DLC_DISABLE_DEVICE_D2H_LAYOUT": env.get(
                    "DLC_DISABLE_DEVICE_D2H_LAYOUT"
                ),
            },
            "records": records,
        }

        failed = [record for record in records if record["status"] == "FAIL"]
        if failed and args.fail_fast:
            print(f"stopping after {group} failure", flush=True)
            break

    suite["finished_at"] = datetime.now().astimezone().isoformat()
    suite["status"] = (
        "PASS"
        if all(
            record["status"] != "FAIL"
            for group in suite["groups"].values()
            for record in group["records"]
        )
        else "FAIL"
    )
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(suite, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"RESULTS {summary_path}", flush=True)
    print(f"STATUS {suite['status']}", flush=True)
    return 0 if suite["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
