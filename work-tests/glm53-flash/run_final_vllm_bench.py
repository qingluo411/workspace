#!/usr/bin/env python3
"""Run the final GLM-5.3-Flash serving benchmark and render Markdown.

The script is deliberately client-side only: it never starts, restarts, or
modifies the served model.  Start the service first, then run this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_VLLM = "/root/workspace/.venv/bin/vllm"
DEFAULT_MODEL_PATH = "/data/models/GLM-5.3-Flash-Mix"
DEFAULT_MODEL_NAME = "GLM-5.3-Flash-Mix"
DEFAULT_RESULT_ROOT = Path("/root/workspace/work-tests/glm53-flash/results")
DEFAULT_DOC_DIR = Path("/root/workspace/work-docs/glm53flash")
DEFAULT_SEED = 20261102
CHAT_TEMPLATE_RESERVE = 64


@dataclass(frozen=True)
class BenchCase:
    key: str
    title: str
    input_len: int
    output_len: int
    num_prompts: int
    concurrency: int
    group: str = "core"
    repeats: int | None = None

    def file_stem(self, repeat: int) -> str:
        return f"{self.key}.run{repeat:02d}"


CORE_CASES = [
    BenchCase("short_decode_c1", "短上下文 decode / 单请求", 49, 96, 1, 1),
    BenchCase("short_decode_c8", "短上下文 decode / 8 并发", 49, 96, 8, 8),
    BenchCase("short_decode_c32", "短上下文 decode / 32 并发", 49, 64, 32, 32),
    BenchCase("short_decode_c64", "短上下文 decode / 64 并发", 49, 64, 64, 64),
    BenchCase("medium_decode_c1", "中等 prompt decode / 单请求", 860, 64, 1, 1),
    BenchCase("medium_decode_c8", "中等 prompt decode / 8 并发", 860, 64, 8, 8),
    BenchCase("ttft_1217_c1", "1K prompt TTFT / 单请求", 1217, 8, 1, 1),
    BenchCase("prefill_4k_c1", "4K prompt / 单请求", 3610, 8, 1, 1),
    BenchCase("guide_baseline_c8", "指南随机基线 / 8 并发", 512, 64, 16, 8),
]

EXTENDED_CASES = [
    BenchCase("ttft_8k_c1", "8K prompt / 单请求", 7800, 8, 1, 1, "extended"),
    BenchCase("ttft_16k_c1", "16K prompt / 单请求", 15800, 8, 1, 1, "extended"),
    BenchCase("ttft_32k_c1", "32K prompt / 单请求", 32000, 8, 1, 1, "extended"),
]


@dataclass
class RunRecord:
    repeat: int
    returncode: int
    result_path: str
    log_path: str
    result: dict[str, Any] | None
    error: str | None = None


@dataclass
class CaseOutcome:
    case: BenchCase
    repeats: int
    skipped: str | None = None
    runs: list[RunRecord] = field(default_factory=list)
    aggregate: dict[str, Any] | None = None

    @property
    def valid_runs(self) -> list[RunRecord]:
        return [r for r in self.runs if r.error is None and r.result is not None]

    @property
    def failed_runs(self) -> list[RunRecord]:
        return [r for r in self.runs if r.error is not None]

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIPPED"
        if not self.runs:
            return "NOT_RUN"
        if not self.valid_runs:
            return "FAILED"
        if self.failed_runs or len(self.valid_runs) != self.repeats:
            return "PARTIAL"
        return "PASS"


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final vLLM serving benchmarks and write Markdown tables."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BASE_URL"),
        help="OpenAI API base URL; defaults to http://127.0.0.1:$PORT",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", "18005"))
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL"),
        help="Served model name; auto-detected from /v1/models when omitted",
    )
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("TOKENIZER", DEFAULT_MODEL_PATH),
        help="Tokenizer path/name passed to vllm bench serve",
    )
    parser.add_argument(
        "--vllm-bin", default=os.environ.get("VLLM_BIN", DEFAULT_VLLM)
    )
    parser.add_argument(
        "--matrix",
        choices=("core", "extended"),
        default="core",
        help="core covers through 4K prompt; extended adds 8K/16K/32K TTFT",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case keys; overrides --matrix",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured runs per case (default: 3; median is reported)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Uncounted vllm bench warmups before the first measured run",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=1800,
        help="Timeout for one vllm bench serve invocation",
    )
    parser.add_argument(
        "--health-timeout-sec",
        type=int,
        default=60,
        help="Maximum wait for /health before or during the run",
    )
    parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=1.0,
        help="Pause between benchmark invocations",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Root directory for timestamped result runs",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Result run directory name; defaults to a timestamp",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=None,
        help="Markdown output path; defaults under work-docs/glm53flash",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random dataset seed"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override server max_model_len for table and case guards",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Override server max_num_seqs for table and concurrency guards",
    )
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=None, help="Server config"
    )
    parser.add_argument(
        "--gpu-memory-utilization", default=None, help="Server config"
    )
    parser.add_argument("--tp-size", type=int, default=None, help="Server config")
    parser.add_argument("--block-size", type=int, default=None, help="Server config")
    parser.add_argument("--graph-mode", default=None, help="Server graph mode")
    parser.add_argument(
        "--graph-cache", default=None, help="DLC_SYN_GRAPH_CACHE value"
    )
    parser.add_argument(
        "--prompt-reserve-tokens",
        type=int,
        default=CHAT_TEMPLATE_RESERVE,
        help="Reserved tokens for chat-template overhead in max-length guard",
    )
    parser.add_argument(
        "--require-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort if the detected server is running eager",
    )
    parser.add_argument(
        "--require-graph-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort if the detected server has DLC_SYN_GRAPH_CACHE=0",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop at the first failed run"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without contacting the service",
    )
    parser.add_argument(
        "--check-service",
        action="store_true",
        help="With --dry-run, also perform /health and /v1/models checks",
    )
    parser.add_argument(
        "--extra-vllm-arg",
        action="append",
        default=[],
        help="Append one extra argument to every vllm bench serve command",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be >= 0")
    if args.timeout_sec < 1:
        parser.error("--timeout-sec must be >= 1")
    if not args.base_url:
        args.base_url = f"http://127.0.0.1:{args.port}"
    args.base_url = args.base_url.rstrip("/")
    parsed_base_url = urlparse(args.base_url)
    if parsed_base_url.port is not None:
        args.port = parsed_base_url.port
    if args.doc is None:
        args.doc = DEFAULT_DOC_DIR / f"GLM-5.3-Flash_最终性能结果_{datetime.now():%Y-%m-%d}.md"
    return args


def http_get(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def wait_for_health(base_url: str, timeout_sec: int) -> None:
    deadline = time.monotonic() + timeout_sec
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            http_get(f"{base_url}/health", timeout=5)
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"service is not healthy at {base_url}: {last_error}")


def read_models(base_url: str) -> tuple[str, dict[str, Any]]:
    payload = json.loads(http_get(f"{base_url}/v1/models", timeout=10))
    data = payload.get("data") or []
    if not data:
        raise RuntimeError("/v1/models returned no models")
    return str(data[0].get("id", "")), data[0]


def choose_model(
    base_url: str, requested: str | None
) -> tuple[str, dict[str, Any]]:
    first_id, first_card = read_models(base_url)
    if requested and requested != first_id:
        raise RuntimeError(
            f"requested model {requested!r} is not served; /v1/models reports {first_id!r}"
        )
    return requested or first_id, first_card


def _read_proc_bytes(pid: str, name: str) -> bytes | None:
    try:
        return Path(f"/proc/{pid}/{name}").read_bytes()
    except (OSError, PermissionError):
        return None


def _proc_argv(pid: str) -> list[str]:
    raw = _read_proc_bytes(pid, "cmdline")
    if not raw:
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def _proc_env(pid: str) -> dict[str, str]:
    raw = _read_proc_bytes(pid, "environ")
    if not raw:
        return {}
    env: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return env


def _flag_value(argv: list[str], name: str) -> str | None:
    for index, token in enumerate(argv):
        if token == name and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_compilation_mode(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, dict):
        mode = parsed.get("cudagraph_mode") or parsed.get("mode")
        return str(mode) if mode is not None else None
    return None


def detect_server_process(port: int, model_path: str | None) -> dict[str, Any] | None:
    candidates: list[tuple[int, dict[str, Any]]] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        argv = _proc_argv(entry.name)
        if "serve" not in argv or not any(
            token.endswith("/vllm") or token == "vllm" or "vllm.entrypoints" in token
            for token in argv
        ):
            continue
        serve_index = argv.index("serve")
        model_arg = argv[serve_index + 1] if serve_index + 1 < len(argv) else ""
        process_port = _parse_int(_flag_value(argv, "--port"))
        score = 0
        if process_port == port:
            score += 100
        if model_path and model_arg and Path(model_arg) == Path(model_path):
            score += 20
        candidates.append(
            (score, {"pid": int(entry.name), "argv": argv, "model_arg": model_arg})
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] == 0:
        return None
    return candidates[0][1]


def collect_process_metadata(proc: dict[str, Any] | None) -> dict[str, Any]:
    if not proc:
        return {}
    argv = proc["argv"]
    env = _proc_env(str(proc["pid"]))
    compilation = _flag_value(argv, "--compilation-config")
    graph_mode = _parse_compilation_mode(compilation)
    eager = "--enforce-eager" in argv
    if eager:
        graph_enabled: bool | None = False
    elif graph_mode:
        graph_enabled = graph_mode.upper() not in {"NONE", "NO", "DISABLED"}
    else:
        graph_enabled = None
    profiler_args = any(
        token == "--profiler-config" or token.startswith("--profiler-config.")
        for token in argv
    )
    verbose = env.get("DLC_SYN_VERBOSE", "0").lower() not in {"", "0", "false", "off"}
    debug = env.get("DLC_SYN_DEBUG", "0").lower() not in {"", "0", "false", "off"}
    return {
        "pid": proc["pid"],
        "graph_enabled": graph_enabled,
        "graph_mode": graph_mode,
        "graph_cache": env.get("DLC_SYN_GRAPH_CACHE"),
        "max_model_len": _parse_int(_flag_value(argv, "--max-model-len")),
        "max_num_seqs": _parse_int(_flag_value(argv, "--max-num-seqs")),
        "max_num_batched_tokens": _parse_int(
            _flag_value(argv, "--max-num-batched-tokens")
        ),
        "gpu_memory_utilization": _flag_value(argv, "--gpu-memory-utilization"),
        "tp_size": _parse_int(_flag_value(argv, "--tensor-parallel-size")),
        "block_size": _parse_int(_flag_value(argv, "--block-size")),
        "enforce_eager": eager,
        "profiler_enabled": profiler_args,
        "kernel_log_enabled": verbose or debug,
        "vllm_plugins": env.get("VLLM_PLUGINS"),
        "col_major_matmul": env.get("VLLM_USE_DLC_COL_MAJOR_MATMUL"),
        "alloc_conf": env.get("PYTORCH_DLC_ALLOC_CONF"),
    }


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def build_service_metadata(
    args: argparse.Namespace,
    model_id: str,
    model_card: dict[str, Any],
    proc_meta: dict[str, Any],
) -> dict[str, Any]:
    model_path = model_card.get("root")
    if not model_path or not isinstance(model_path, str):
        model_path = None
    max_model_len = _first_not_none(
        args.max_model_len, proc_meta.get("max_model_len"), model_card.get("max_model_len")
    )
    graph_enabled = proc_meta.get("graph_enabled")
    if args.graph_mode:
        graph_enabled = args.graph_mode.upper() not in {"NONE", "NO", "DISABLED"}
    graph_cache = _first_not_none(args.graph_cache, proc_meta.get("graph_cache"))
    return {
        "base_url": args.base_url,
        "model_id": model_id,
        "model_path": model_path or DEFAULT_MODEL_PATH,
        "tokenizer": args.tokenizer,
        "max_model_len": max_model_len,
        "max_num_seqs": _first_not_none(args.max_num_seqs, proc_meta.get("max_num_seqs")),
        "max_num_batched_tokens": _first_not_none(
            args.max_num_batched_tokens, proc_meta.get("max_num_batched_tokens")
        ),
        "gpu_memory_utilization": _first_not_none(
            args.gpu_memory_utilization, proc_meta.get("gpu_memory_utilization")
        ),
        "tp_size": _first_not_none(args.tp_size, proc_meta.get("tp_size")),
        "block_size": _first_not_none(args.block_size, proc_meta.get("block_size")),
        "graph_enabled": graph_enabled,
        "graph_mode": _first_not_none(args.graph_mode, proc_meta.get("graph_mode")),
        "graph_cache": graph_cache,
        "server_pid": proc_meta.get("pid"),
        "profiler_enabled": proc_meta.get("profiler_enabled"),
        "kernel_log_enabled": proc_meta.get("kernel_log_enabled"),
        "vllm_plugins": proc_meta.get("vllm_plugins"),
        "col_major_matmul": proc_meta.get("col_major_matmul"),
        "alloc_conf": proc_meta.get("alloc_conf"),
    }


def command_output(argv: list[str], cwd: str | None = None) -> str | None:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def build_fingerprint() -> dict[str, Any]:
    kernel_repo = Path("/root/workspace/DLC_Custom_Kernel")
    kernel_lib = Path(
        os.environ.get(
            "CUSTOM_KERNEL_LIB",
            "/usr/local/chipltech/synapse/lib/libcustom_dlc_perf_lib.so",
        )
    )
    revision = command_output(["git", "rev-parse", "--short", "HEAD"], str(kernel_repo))
    status = command_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], str(kernel_repo)
    )
    stat_info: dict[str, Any] = {}
    if kernel_lib.exists():
        stat = kernel_lib.stat()
        stat_info = {
            "path": str(kernel_lib),
            "size": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        }
    return {
        "kernel_repo_revision": revision,
        "kernel_repo_dirty": bool(status),
        "installed_kernel_lib": stat_info,
    }


def resolve_cases(args: argparse.Namespace) -> list[BenchCase]:
    all_cases = CORE_CASES + (EXTENDED_CASES if args.matrix == "extended" else [])
    if args.cases:
        wanted = [item.strip() for item in args.cases.split(",") if item.strip()]
        by_key = {case.key: case for case in all_cases}
        missing = [key for key in wanted if key not in by_key]
        if missing:
            raise RuntimeError(f"unknown case key(s): {', '.join(missing)}")
        return [by_key[key] for key in wanted]
    return list(all_cases)


def skip_reason(
    case: BenchCase,
    max_model_len: int | None,
    max_num_seqs: int | None,
    reserve: int,
) -> str | None:
    if max_model_len is not None:
        required = case.input_len + case.output_len + reserve
        if required > max_model_len:
            return (
                f"prompt+output+chat reserve ({required}) exceeds "
                f"max_model_len ({max_model_len})"
            )
    if max_num_seqs is not None and case.concurrency > max_num_seqs:
        return (
            f"concurrency ({case.concurrency}) exceeds max_num_seqs "
            f"({max_num_seqs})"
        )
    return None


def build_command(
    args: argparse.Namespace,
    case: BenchCase,
    repeat: int,
    result_file: Path,
    warmup_runs: int,
) -> list[str]:
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
        getattr(args, "model_id", args.model or DEFAULT_MODEL_NAME),
        "--tokenizer",
        args.tokenizer,
        "--trust-remote-code",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(case.input_len),
        "--random-output-len",
        str(case.output_len),
        "--num-prompts",
        str(case.num_prompts),
        "--max-concurrency",
        str(case.concurrency),
        "--ignore-eos",
        "--temperature",
        "0",
        "--seed",
        str(args.seed),
        "--metric-percentiles",
        "50,90,99",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--disable-tqdm",
        "--label",
        case.key,
        "--save-result",
        "--result-dir",
        str(result_file.parent),
        "--result-filename",
        result_file.name,
    ]
    if repeat == 0 and warmup_runs > 0:
        cmd.extend(["--num-warmups", str(warmup_runs)])
    cmd.extend(args.extra_vllm_arg)
    return cmd


def parse_console_metrics(text: str) -> dict[str, Any] | None:
    patterns: dict[str, tuple[str, type]] = {
        "completed": (r"Successful requests:\s+(\d+)", int),
        "failed": (r"Failed requests:\s+(\d+)", int),
        "duration": (r"Benchmark duration \(s\):\s+([0-9.]+)", float),
        "total_input_tokens": (r"Total input tokens:\s+(\d+)", int),
        "total_output_tokens": (r"Total generated tokens:\s+(\d+)", int),
        "request_throughput": (r"Request throughput \(req/s\):\s+([0-9.]+)", float),
        "output_throughput": (
            r"Output token throughput \(tok/s\):\s+([0-9.]+)",
            float,
        ),
        "max_output_tokens_per_s": (
            r"Peak output token throughput \(tok/s\):\s+([0-9.]+)",
            float,
        ),
        "max_concurrent_requests": (r"Peak concurrent requests:\s+(\d+)", int),
        "mean_ttft_ms": (r"Mean TTFT \(ms\):\s+([0-9.]+)", float),
        "median_ttft_ms": (r"Median TTFT \(ms\):\s+([0-9.]+)", float),
        "p50_ttft_ms": (r"P50 TTFT \(ms\):\s+([0-9.]+)", float),
        "p90_ttft_ms": (r"P90 TTFT \(ms\):\s+([0-9.]+)", float),
        "p99_ttft_ms": (r"P99 TTFT \(ms\):\s+([0-9.]+)", float),
        "mean_tpot_ms": (r"Mean TPOT \(ms\):\s+([0-9.]+)", float),
        "median_tpot_ms": (r"Median TPOT \(ms\):\s+([0-9.]+)", float),
        "p50_tpot_ms": (r"P50 TPOT \(ms\):\s+([0-9.]+)", float),
        "mean_itl_ms": (r"Mean ITL \(ms\):\s+([0-9.]+)", float),
        "median_itl_ms": (r"Median ITL \(ms\):\s+([0-9.]+)", float),
        "p50_itl_ms": (r"P50 ITL \(ms\):\s+([0-9.]+)", float),
        "mean_e2el_ms": (r"Mean E2EL \(ms\):\s+([0-9.]+)", float),
        "median_e2el_ms": (r"Median E2EL \(ms\):\s+([0-9.]+)", float),
        "p50_e2el_ms": (r"P50 E2EL \(ms\):\s+([0-9.]+)", float),
        "p90_e2el_ms": (r"P90 E2EL \(ms\):\s+([0-9.]+)", float),
        "p99_e2el_ms": (r"P99 E2EL \(ms\):\s+([0-9.]+)", float),
    }
    data: dict[str, Any] = {}
    for key, (pattern, converter) in patterns.items():
        match = re.search(pattern, text)
        if match:
            data[key] = converter(match.group(1))
    return data if data else None


def load_run_result(
    result_file: Path, log_file: Path, expected: int
) -> tuple[dict[str, Any] | None, str | None]:
    data: dict[str, Any] | None = None
    if result_file.exists():
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"cannot parse result JSON: {exc}"
    else:
        text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
        fallback = parse_console_metrics(text)
        if fallback:
            data = fallback
        else:
            return None, "result JSON and console metrics are both missing"

    completed = int(data.get("completed", 0) or 0)
    failed = int(data.get("failed", 0) or 0)
    if failed:
        return data, f"vLLM reported {failed} failed request(s)"
    if completed != expected:
        return data, f"expected {expected} completed requests, got {completed}"
    if not data.get("duration"):
        return data, "result has no benchmark duration"
    return data, None


def run_case(
    args: argparse.Namespace,
    outcome: CaseOutcome,
    run_dir: Path,
) -> None:
    case = outcome.case
    repeats = outcome.repeats
    case_dir = run_dir / case.key
    case_dir.mkdir(parents=True, exist_ok=True)
    for repeat in range(repeats):
        # Check the service before every measured run.  This avoids attributing
        # a dead service to several unrelated benchmark cases.
        wait_for_health(args.base_url, args.health_timeout_sec)
        result_file = case_dir / f"{case.file_stem(repeat)}.json"
        log_file = case_dir / f"{case.file_stem(repeat)}.log"
        # Never let a stale result from an interrupted earlier rerun make a
        # failed invocation look successful.
        result_file.unlink(missing_ok=True)
        cmd = build_command(args, case, repeat, result_file, args.warmup_runs)
        log(
            f"run {case.key} repeat {repeat + 1}/{repeats}: "
            f"{case.input_len}+{case.output_len} tokens, "
            f"n={case.num_prompts}, c={case.concurrency}"
        )
        started = time.monotonic()
        returncode = -1
        run_error: str | None = None
        try:
            with log_file.open("w", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=args.timeout_sec,
                    check=False,
                    env=os.environ.copy(),
                )
                returncode = completed.returncode
        except subprocess.TimeoutExpired:
            run_error = f"benchmark timed out after {args.timeout_sec}s"
        except OSError as exc:
            run_error = f"failed to launch vllm: {exc}"
        elapsed = time.monotonic() - started

        result, parse_error = load_run_result(result_file, log_file, case.num_prompts)
        if returncode != 0 and run_error is None:
            run_error = f"vllm bench serve exited with code {returncode}"
        if parse_error and run_error is None:
            run_error = parse_error
        record = RunRecord(
            repeat=repeat,
            returncode=returncode,
            result_path=str(result_file),
            log_path=str(log_file),
            result=result,
            error=run_error,
        )
        outcome.runs.append(record)
        if run_error:
            eprint(f"[{case.key}] repeat {repeat + 1} failed: {run_error}")
            if args.fail_fast:
                raise RuntimeError(run_error)
        else:
            assert result is not None
            log(
                f"[{case.key}] repeat {repeat + 1} done in {elapsed:.1f}s: "
                f"output={result.get('output_throughput', 0):.2f} tok/s, "
                f"TTFT p50={result.get('p50_ttft_ms', 0):.2f} ms, "
                f"TPOT p50={result.get('p50_tpot_ms', 0):.2f} ms"
            )
        if repeat + 1 < repeats:
            time.sleep(args.cooldown_sec)


def _numeric_values(runs: list[RunRecord], key: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        if run.result is None:
            continue
        value = run.result.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _median(runs: list[RunRecord], key: str) -> float | None:
    values = _numeric_values(runs, key)
    return median(values) if values else None


def aggregate_case(outcome: CaseOutcome) -> dict[str, Any] | None:
    runs = outcome.valid_runs
    if not runs:
        return None
    data: dict[str, Any] = {
        "valid_runs": len(runs),
        "planned_runs": outcome.repeats,
        "completed": int(_median(runs, "completed") or 0),
        "failed": int(sum(float(r.result.get("failed", 0)) for r in runs if r.result)),
    }
    for key in (
        "duration",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "max_output_tokens_per_s",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p50_ttft_ms",
        "p90_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p50_tpot_ms",
        "p90_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "p50_itl_ms",
        "p90_itl_ms",
        "p99_itl_ms",
        "mean_e2el_ms",
        "median_e2el_ms",
        "p50_e2el_ms",
        "p90_e2el_ms",
        "p99_e2el_ms",
        "max_concurrent_requests",
    ):
        data[key] = _median(runs, key)
    input_tokens = _numeric_values(runs, "total_input_tokens")
    output_tokens = _numeric_values(runs, "total_output_tokens")
    if input_tokens and data["completed"]:
        data["avg_input_tokens"] = median(input_tokens) / data["completed"]
    if output_tokens and data["completed"]:
        data["avg_output_tokens"] = median(output_tokens) / data["completed"]
    return data


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "—"
        return f"{float(value):,.{digits}f}"
    return str(value).replace("|", "\\|")


def fmt_triple(runs: list[RunRecord], key: str, digits: int = 2) -> str:
    values = _numeric_values(runs, key)
    if not values:
        return "—"
    return f"{fmt(min(values), digits)} / {fmt(median(values), digits)} / {fmt(max(values), digits)}"


def status_label(status: str) -> str:
    return {
        "PASS": "通过",
        "PARTIAL": "部分失败",
        "FAILED": "失败",
        "SKIPPED": "跳过",
        "NOT_RUN": "未运行",
    }.get(status, status)


def md_bool(value: Any) -> str:
    if value is True:
        return "开启"
    if value is False:
        return "关闭"
    return "未采集"


def build_markdown(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    outcomes: list[CaseOutcome],
    state: str,
    started_at: str,
    finished_at: str | None,
    run_dir: Path,
) -> str:
    service = metadata["service"]
    fingerprint = metadata["fingerprint"]
    lines: list[str] = []
    lines.append("# GLM-5.3-Flash 最终性能结果")
    lines.append("")
    lines.append(f"> 测试时间：{started_at}（UTC）  ")
    lines.append(
        f"> 状态：{'测试完成' if state == 'complete' else '测试未完成（脚本中断或仍在运行）'}  "
    )
    lines.append(
        "> 数据口径：`vllm bench serve` 客户端实测；当前发布路径；"
        "**不包含 KPool 优化后的理想值，也未替换生产路径中的该算子**。"
    )
    lines.append("")

    summaries: list[str] = []
    for key, label in (
        ("short_decode_c1", "短上下文单请求"),
        ("short_decode_c8", "短上下文 8 并发"),
        ("short_decode_c32", "短上下文 32 并发"),
        ("short_decode_c64", "短上下文 64 并发"),
        ("prefill_4k_c1", "4K prompt 单请求"),
    ):
        outcome = next((item for item in outcomes if item.case.key == key), None)
        if outcome and outcome.aggregate:
            data = outcome.aggregate
            if key == "prefill_4k_c1":
                summaries.append(
                    f"- {label}：TTFT P50 `{fmt(data.get('p50_ttft_ms'))} ms`，"
                    f"E2EL 中位数 `{fmt(data.get('median_e2el_ms'))} ms`。"
                )
            else:
                summaries.append(
                    f"- {label}：TTFT P50 `{fmt(data.get('p50_ttft_ms'))} ms`，"
                    f"TPOT P50 `{fmt(data.get('p50_tpot_ms'))} ms`，"
                    f"输出吞吐 `{fmt(data.get('output_throughput'))} tok/s`。"
                )
    passed = sum(1 for item in outcomes if item.status == "PASS")
    failed = sum(1 for item in outcomes if item.status in {"FAILED", "PARTIAL"})
    skipped = sum(1 for item in outcomes if item.status == "SKIPPED")
    lines.append("## 1. 结论摘要")
    lines.append("")
    lines.append(
        f"- 计划场景 `{len(outcomes)}` 个：通过 `{passed}`、失败/部分失败 `{failed}`、"
        f"跳过 `{skipped}`。"
    )
    lines.extend(summaries or ["- 尚无有效场景结果。"])
    lines.append("- 所有时延和吞吐均为客户端侧统计，不含服务端 profiler 插桩。")
    lines.append("")

    lines.append("## 2. 服务与测试口径")
    lines.append("")
    lines.append("### 2.1 服务配置")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| API | `{service['base_url']}` |")
    lines.append(f"| Model | `{service['model_id']}` |")
    lines.append(f"| 模型路径 | `{service['model_path']}` |")
    lines.append(f"| Tokenizer | `{service['tokenizer']}` |")
    lines.append(f"| `max_model_len` | `{fmt(service['max_model_len'], 0)}` |")
    lines.append(
        f"| `max_num_batched_tokens` | `{fmt(service['max_num_batched_tokens'], 0)}` |"
    )
    lines.append(f"| `max_num_seqs` | `{fmt(service['max_num_seqs'], 0)}` |")
    lines.append(f"| TP | `{fmt(service['tp_size'], 0)}` |")
    lines.append(f"| `block_size` | `{fmt(service['block_size'], 0)}` |")
    lines.append(
        f"| `gpu_memory_utilization` | `{fmt(service['gpu_memory_utilization'])}` |"
    )
    lines.append(f"| Graph | `{md_bool(service['graph_enabled'])}` |")
    lines.append(f"| Graph mode | `{fmt(service['graph_mode'])}` |")
    lines.append(f"| `DLC_SYN_GRAPH_CACHE` | `{fmt(service['graph_cache'])}` |")
    lines.append("")
    lines.append("### 2.2 压测方法")
    lines.append("")
    lines.append("- 客户端：`vllm bench serve --backend openai-chat`。")
    lines.append(
        "- 数据集：`random`，固定 `--seed`；`--temperature 0`、`--ignore-eos`，"
        "使输出长度可重复。"
    )
    lines.append(
        f"- 每个场景首轮先做 `{args.warmup_runs}` 个不计时 warmup，随后正式重复 "
        f"`{args.repeats}` 次；表中数值为正式运行的跨 run 中位数。"
    )
    lines.append(
        "- 时延指标来自客户端请求时间戳：TTFT 为首 token 时延，TPOT 排除首 token，"
        "ITL 为 token 间隔，E2EL 为请求端到端时延。"
    )
    lines.append("- 未启用服务端 profiler / kernel log；未在测试期间运行其他压测。")
    lines.append("")
    lines.append("### 2.3 构建标记")
    lines.append("")
    lines.append(f"- Kernel 仓库 revision：`{fmt(fingerprint.get('kernel_repo_revision'))}`；"
                 f"工作区是否 dirty：`{md_bool(fingerprint.get('kernel_repo_dirty'))}`。")
    kernel_lib = fingerprint.get("installed_kernel_lib") or {}
    lines.append(
        f"- 已安装算子库：`{fmt(kernel_lib.get('path'))}`，大小 `{fmt(kernel_lib.get('size'), 0)}` B，"
        f"mtime `{fmt(kernel_lib.get('mtime'))}`。"
    )
    lines.append("")

    lines.append("## 3. 吞吐结果")
    lines.append("")
    lines.append(
        "| 场景 | 名义输入/输出 token | 请求数 | 并发 | 有效/计划 run | "
        "实际平均输入/输出 token | 输出 tok/s | 请求/s | 完成/失败 | 峰值并发 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for outcome in outcomes:
        case = outcome.case
        if outcome.skipped:
            lines.append(
                f"| {case.title} (`{case.key}`) | {case.input_len}/{case.output_len} | "
                f"{case.num_prompts} | {case.concurrency} | 0/{outcome.repeats} | "
                f"— | — | — | 跳过：{outcome.skipped} | — |"
            )
            continue
        data = outcome.aggregate or {}
        actual = "—"
        if data.get("avg_input_tokens") is not None and data.get("avg_output_tokens") is not None:
            actual = f"{fmt(data['avg_input_tokens'], 1)}/{fmt(data['avg_output_tokens'], 1)}"
        lines.append(
            f"| {case.title} (`{case.key}`) | {case.input_len}/{case.output_len} | "
            f"{case.num_prompts} | {case.concurrency} | {len(outcome.valid_runs)}/{outcome.repeats} | "
            f"{actual} | {fmt(data.get('output_throughput'))} | "
            f"{fmt(data.get('request_throughput'))} | "
            f"{fmt(data.get('completed'), 0)}/{fmt(data.get('failed'), 0)} | "
            f"{fmt(data.get('max_concurrent_requests'), 0)} |"
        )
    lines.append("")

    lines.append("## 4. 时延结果")
    lines.append("")
    lines.append(
        "| 场景 | TTFT mean | TTFT P50 | TTFT P90 | TPOT mean | TPOT P50 | "
        "ITL P50 | E2EL mean | E2EL P50 | E2EL P90 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for outcome in outcomes:
        case = outcome.case
        if outcome.skipped or not outcome.aggregate:
            lines.append(
                f"| {case.title} (`{case.key}`) | — | — | — | — | — | — | — | — | — |"
            )
            continue
        data = outcome.aggregate
        lines.append(
            f"| {case.title} (`{case.key}`) | {fmt(data.get('mean_ttft_ms'))} | "
            f"{fmt(data.get('p50_ttft_ms'))} | {fmt(data.get('p90_ttft_ms'))} | "
            f"{fmt(data.get('mean_tpot_ms'))} | {fmt(data.get('p50_tpot_ms'))} | "
            f"{fmt(data.get('p50_itl_ms'))} | {fmt(data.get('mean_e2el_ms'))} | "
            f"{fmt(data.get('p50_e2el_ms'))} | {fmt(data.get('p90_e2el_ms'))} |"
        )
    lines.append("")
    lines.append("> 时延单位均为 ms。TTFT/P90 仅反映对应请求形状的观测值；"
                 "不同输入和输出长度不应直接混作同一 SLA。")
    lines.append("")

    lines.append("## 5. 重复性")
    lines.append("")
    lines.append(
        "| 场景 | 有效 run | 输出 tok/s (min / median / max) | "
        "TTFT P50 ms (min / median / max) | TPOT P50 ms (min / median / max) |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for outcome in outcomes:
        if outcome.skipped:
            lines.append(
                f"| {outcome.case.title} (`{outcome.case.key}`) | — | — | — | — |"
            )
            continue
        runs = outcome.valid_runs
        lines.append(
            f"| {outcome.case.title} (`{outcome.case.key}`) | {len(runs)} | "
            f"{fmt_triple(runs, 'output_throughput')} | "
            f"{fmt_triple(runs, 'p50_ttft_ms')} | "
            f"{fmt_triple(runs, 'p50_tpot_ms')} |"
        )
    lines.append("")

    lines.append("## 6. 状态与限制")
    lines.append("")
    lines.append(f"- 完成时间：{finished_at or '尚未完成'}（UTC）。")
    if state != "complete":
        lines.append("- 本文档是部分结果；脚本退出后可用相同参数重跑生成完整结果。")
    graph_status = md_bool(service.get("graph_enabled"))
    lines.append(
        f"- 本次表格对应 Graph `{graph_status}`、`DLC_SYN_GRAPH_CACHE={fmt(service.get('graph_cache'))}` 的服务路径。"
    )
    lines.append(
        "- 4K 及更长 prompt 的 E2EL 主要受 prefill 支配；decode 吞吐应以对应输出长度的结果解读。"
    )
    lines.append(
        "- 64 并发可能落入非整除 graph bucket；该成本已包含在实测值中，不另行剔除。"
    )
    lines.append(
        "- **本表没有采用 KPool 优化后的性能模型或估算数据**。如需评估优化收益，应另列对照表。"
    )
    failed_details: list[str] = []
    for outcome in outcomes:
        for run in outcome.failed_runs:
            failed_details.append(
                f"- `{outcome.case.key}` run {run.repeat + 1}: {run.error} "
                f"(log: `{run.log_path}`)"
            )
    lines.append("")
    lines.append("### 失败项")
    lines.append("")
    lines.extend(failed_details or ["- 无。"])
    lines.append("")
    lines.append("## 7. 原始数据")
    lines.append("")
    lines.append(f"- 结果目录：`{run_dir}`")
    lines.append(f"- 汇总 JSON：`{run_dir / 'summary.json'}`")
    lines.append(f"- 每个场景的 JSON 和日志位于 `{run_dir}/<case_key>/`。")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_outputs(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    outcomes: list[CaseOutcome],
    state: str,
    started_at: str,
    finished_at: str | None,
    run_dir: Path,
) -> None:
    summary = {
        "state": state,
        "started_at": started_at,
        "finished_at": finished_at,
        "metadata": metadata,
        "cases": [
            {
                "key": outcome.case.key,
                "title": outcome.case.title,
                "input_len": outcome.case.input_len,
                "output_len": outcome.case.output_len,
                "num_prompts": outcome.case.num_prompts,
                "concurrency": outcome.case.concurrency,
                "repeats": outcome.repeats,
                "status": outcome.status,
                "skipped": outcome.skipped,
                "aggregate": outcome.aggregate,
                "runs": [
                    {
                        "repeat": run.repeat,
                        "returncode": run.returncode,
                        "result_path": run.result_path,
                        "log_path": run.log_path,
                        "error": run.error,
                    }
                    for run in outcome.runs
                ],
            }
            for outcome in outcomes
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = build_markdown(
        args, metadata, outcomes, state, started_at, finished_at, run_dir
    )
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    temp_doc = args.doc.with_suffix(args.doc.suffix + ".tmp")
    temp_doc.write_text(markdown, encoding="utf-8")
    os.replace(temp_doc, args.doc)


def main() -> int:
    args = parse_args()
    if args.dry_run:
        cases = resolve_cases(args)
        args.model_id = args.model or DEFAULT_MODEL_NAME
        print(f"base_url={args.base_url}")
        print(f"model={args.model_id}")
        print(f"tokenizer={args.tokenizer}")
        print(f"repeats={args.repeats}, warmup_runs={args.warmup_runs}")
        print()
        for case in cases:
            reason = skip_reason(
                case, args.max_model_len, args.max_num_seqs, args.prompt_reserve_tokens
            )
            label = f"SKIP ({reason})" if reason else "RUN"
            print(f"{label:4s} {case.key}: {case.input_len}+{case.output_len}, "
                  f"n={case.num_prompts}, c={case.concurrency}")
            command = build_command(
                args, case, 0, Path("CASE.json"), args.warmup_runs
            )
            print("     " + shlex.join(command))
        if args.check_service:
            wait_for_health(args.base_url, args.health_timeout_sec)
            model_id, card = choose_model(args.base_url, args.model)
            print(f"service OK: model={model_id}, max_model_len={card.get('max_model_len')}")
        return 0

    vllm_bin = (
        args.vllm_bin
        if Path(args.vllm_bin).exists()
        else shutil.which(args.vllm_bin)
    )
    if not vllm_bin:
        eprint(f"vllm executable not found: {args.vllm_bin}")
        return 2
    args.vllm_bin = vllm_bin
    if (
        (args.tokenizer.startswith("/") or args.tokenizer.startswith("."))
        and not Path(args.tokenizer).exists()
    ):
        eprint(f"tokenizer/model path not found: {args.tokenizer}")
        return 2

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    log(f"checking service health at {args.base_url}")
    wait_for_health(args.base_url, args.health_timeout_sec)
    model_id, model_card = choose_model(args.base_url, args.model)
    model_path = model_card.get("root")
    process = None
    if "127.0.0.1" in args.base_url or "localhost" in args.base_url:
        process = detect_server_process(args.port, model_path)
    proc_meta = collect_process_metadata(process)
    args.model_id = model_id
    service = build_service_metadata(args, model_id, model_card, proc_meta)
    args.model = model_id

    if service.get("graph_enabled") is False and args.require_graph:
        eprint("detected server is running eager; this matrix must use graph")
        return 2
    if service.get("graph_enabled") is None and args.require_graph:
        eprint("warning: graph state could not be detected from the server process")
    if str(service.get("graph_cache")) == "0" and args.require_graph_cache:
        eprint("detected DLC_SYN_GRAPH_CACHE=0; final matrix requires cache=1")
        return 2
    if service.get("profiler_enabled"):
        eprint("warning: server command contains profiler arguments")
    if service.get("kernel_log_enabled"):
        eprint("warning: server environment enables DLC kernel logging")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.result_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "service": service,
        "fingerprint": build_fingerprint(),
        "benchmark": {
            "matrix": args.matrix,
            "repeats": args.repeats,
            "warmup_runs": args.warmup_runs,
            "seed": args.seed,
            "percentiles": [50, 90, 99],
            "kpool_optimization_modeled": False,
        },
    }

    cases = resolve_cases(args)
    outcomes: list[CaseOutcome] = []
    for case in cases:
        reason = skip_reason(
            case, service.get("max_model_len"), service.get("max_num_seqs"),
            args.prompt_reserve_tokens,
        )
        outcome = CaseOutcome(
            case=case,
            repeats=case.repeats or args.repeats,
            skipped=reason,
        )
        outcomes.append(outcome)
        if reason:
            log(f"skip {case.key}: {reason}")

    write_outputs(args, metadata, outcomes, "running", started_at, None, run_dir)
    log(f"results: {run_dir}")
    log(f"markdown: {args.doc}")
    for outcome in outcomes:
        if outcome.skipped:
            continue
        try:
            run_case(args, outcome, run_dir)
        except KeyboardInterrupt:
            outcome.aggregate = aggregate_case(outcome)
            write_outputs(args, metadata, outcomes, "partial", started_at, None, run_dir)
            eprint("interrupted")
            return 130
        except RuntimeError as exc:
            eprint(f"stopping: {exc}")
            outcome.aggregate = aggregate_case(outcome)
            write_outputs(args, metadata, outcomes, "partial", started_at, None, run_dir)
            return 1
        outcome.aggregate = aggregate_case(outcome)
        write_outputs(args, metadata, outcomes, "running", started_at, None, run_dir)

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    write_outputs(args, metadata, outcomes, "complete", started_at, finished_at, run_dir)
    log(f"complete: {args.doc}")
    for outcome in outcomes:
        log(
            f"{outcome.case.key}: {outcome.status} "
            f"({len(outcome.valid_runs)}/{outcome.repeats} valid)"
        )
    return 0 if all(item.status in {"PASS", "SKIPPED"} for item in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
