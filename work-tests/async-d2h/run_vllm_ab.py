#!/usr/bin/env python3
"""Client-side vLLM A/B benchmark for the async-D2H change.

The script deliberately does not hard-code a model.  Run it once per group,
either against an already-running server, or pass --server-cmd so it starts
and stops a fresh server with the group environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_VLLM = "/root/workspace/.venv/bin/vllm"
DEFAULT_BASE_URL = "http://127.0.0.1:18005"
DEFAULT_RESULT_ROOT = Path("/root/workspace/work-tests/async-d2h/results")
DEFAULT_VLLM_CWD = Path("/root/workspace/vllm")


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


def wait_health(
    base_url: str, timeout_s: int, proc: subprocess.Popen[Any] | None = None
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        for path in ("/health", "/v1/models"):
            url = base_url.rstrip("/") + path
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if 200 <= response.status < 300:
                        return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"{url}: {exc}"
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def post_endpoint(base_url: str, path: str) -> None:
    url = base_url.rstrip("/") + path
    request = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=600) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"{url} returned HTTP {response.status}")


def stop_server(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait(timeout=10)


def build_bench_cmd(
    args: argparse.Namespace,
    group: str,
    concurrency: int,
    repeat: int,
    result_dir: Path,
) -> list[str]:
    label = f"{group}_c{concurrency}_r{repeat}"
    result_file = result_dir / f"{label}.json"
    cmd = [
        args.vllm_bin,
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--base-url",
        args.base_url,
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer or args.model,
        "--trust-remote-code",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.input_len),
        "--random-output-len",
        str(args.output_len),
        "--num-prompts",
        str(args.num_prompts),
        "--max-concurrency",
        str(concurrency),
        "--ignore-eos",
        "--temperature",
        "0",
        "--seed",
        str(args.seed),
        "--metric-percentiles",
        "50,95,99",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--disable-tqdm",
        "--label",
        label,
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_file.name,
    ]
    if repeat == 0 and args.warmup_runs > 0:
        cmd.extend(["--num-warmups", str(args.warmup_runs)])
    cmd.extend(args.extra_bench_arg)
    return cmd


def run_bench(
    args: argparse.Namespace,
    group: str,
    env: dict[str, str],
    log_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for concurrency in args.concurrencies:
        case_dir = log_dir / f"c{concurrency}"
        case_dir.mkdir(parents=True, exist_ok=True)
        for repeat in range(args.repeats):
            wait_health(args.base_url, args.health_timeout_sec)
            cmd = build_bench_cmd(args, group, concurrency, repeat, case_dir)
            log_path = case_dir / f"repeat_{repeat:02d}.log"
            started = time.monotonic()
            result = subprocess.run(
                cmd,
                cwd=args.vllm_cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_sec,
                check=False,
            )
            log_path.write_text(result.stdout or "", encoding="utf-8")
            record = {
                "concurrency": concurrency,
                "repeat": repeat,
                "returncode": result.returncode,
                "elapsed_s": time.monotonic() - started,
                "command": cmd,
                "log": str(log_path),
                "result_json": str(
                    case_dir / f"{group}_c{concurrency}_r{repeat}.json"
                ),
            }
            records.append(record)
            print(
                f"{group} c={concurrency} run={repeat + 1}/{args.repeats} "
                f"rc={result.returncode}",
                flush=True,
            )
            if result.returncode != 0 and args.fail_fast:
                return records
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("A", "B", "C"), default="B")
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--vllm-bin", default=os.environ.get("VLLM_BIN", DEFAULT_VLLM))
    parser.add_argument(
        "--vllm-cwd",
        type=Path,
        default=Path(os.environ.get("VLLM_CWD", DEFAULT_VLLM_CWD)),
        help="Working directory for vllm serve/bench; avoids the workspace vllm namespace package",
    )
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--health-timeout-sec", type=int, default=900)
    parser.add_argument("--extra-bench-arg", action="append", default=[])
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="POST /start_profile and /stop_profile; server must be configured with --profiler-config",
    )
    parser.add_argument(
        "--server-cmd",
        default=None,
        help="Optional shell command that starts one server for this group",
    )
    parser.add_argument("--server-log", type=Path, default=None)
    parser.add_argument("--server-ready-timeout-sec", type=int, default=900)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.result_root.resolve() / run_name / f"vllm_{args.group}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = group_env(args.group)
    server_proc: subprocess.Popen[Any] | None = None
    server_handle = None
    profile_started = False

    try:
        if args.server_cmd:
            server_log = args.server_log or (run_dir / "server.log")
            server_log.parent.mkdir(parents=True, exist_ok=True)
            print(f"starting server: {args.server_cmd}", flush=True)
            server_handle = server_log.open("w", encoding="utf-8")
            server_proc = subprocess.Popen(
                args.server_cmd,
                shell=True,
                cwd=args.vllm_cwd,
                env=env,
                stdout=server_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            wait_health(args.base_url, args.server_ready_timeout_sec, server_proc)

        if args.profile:
            post_endpoint(args.base_url, "/start_profile")
            profile_started = True
        bench_records = run_bench(args, args.group, env, run_dir)
        payload = {
            "started_at": datetime.now().astimezone().isoformat(),
            "group": args.group,
            "base_url": args.base_url,
            "model": args.model,
            "env": {
                "DLC_SYN_COPY_ASYNC": env.get("DLC_SYN_COPY_ASYNC"),
                "DLC_DISABLE_DEVICE_D2H_LAYOUT": env.get(
                    "DLC_DISABLE_DEVICE_D2H_LAYOUT"
                ),
            },
            "server_cmd": args.server_cmd,
            "vllm_cwd": str(args.vllm_cwd),
            "profile_requested": args.profile,
            "bench": bench_records,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"RESULTS {run_dir / 'summary.json'}", flush=True)
        return 0 if all(item["returncode"] == 0 for item in bench_records) else 1
    finally:
        if profile_started:
            try:
                post_endpoint(args.base_url, "/stop_profile")
            except Exception as exc:  # noqa: BLE001 - server may already be gone
                print(f"warning: /stop_profile failed: {exc}", flush=True)
        if server_proc is not None:
            stop_server(server_proc)
        if server_handle is not None:
            server_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
