#!/usr/bin/env python3
"""Capture the environment used by one async-D2H test group."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata, util
from pathlib import Path
from typing import Any

ENV_KEYS = (
    "DLC_SYN_COPY_ASYNC",
    "DLC_DISABLE_DEVICE_D2H_LAYOUT",
    "DLC_SYN_USE_SIM",
    "DLC_SYN_BLOCKING",
    "DLC_SYN_GRAPH_CACHE",
    "DLC_TPU_VERSION",
    "DLCKERNELDRIVER_PATH",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "PYTORCH_DLC_ALLOC_CONF",
    "CUSTOM_KERNEL_PATH",
    "VLLM_PLUGINS",
)


def _run_text(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"<error: {exc}>"
    text = proc.stdout.strip()
    return text if text else None


def _read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _package_info(name: str, module_name: str) -> dict[str, Any]:
    info: dict[str, Any] = {"distribution": name, "module": module_name}
    try:
        info["version"] = metadata.version(name)
    except metadata.PackageNotFoundError:
        info["version"] = None
    try:
        spec = util.find_spec(module_name)
        info["file"] = None if spec is None else spec.origin
    except (ImportError, ValueError) as exc:
        info["file"] = f"<error: {type(exc).__name__}: {exc}>"
    return info


def _file_info(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return None
    return {
        "path": str(target),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def collect() -> dict[str, Any]:
    result: dict[str, Any] = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "cwd": os.getcwd(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "env": {key: os.environ.get(key) for key in ENV_KEYS},
        "sysfs": {
            "cltech0_version": _read_text(
                "/sys/devices/virtual/chipltech/cltech0/version"
            ),
        },
        "packages": {
            "vllm": _package_info("vllm", "vllm"),
            "vllm_cl": _package_info("vllm-cl", "vllm_cl"),
        },
        "files": {
            "custom_kernel": _file_info(
                "/usr/local/chipltech/synapse/lib/libcustom_dlc_perf_lib.so"
            ),
            "libsynapse_core": _file_info(
                "/usr/local/chipltech/synapse/lib/libsynapse_core.so"
            ),
            "libhl_thunk": _file_info(
                "/usr/local/chipltech/synapse/lib/libhl-thunk.so"
            ),
        },
        "kernel_module": _run_text(["lsmod"]),
    }

    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - the build may still be in progress
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["torch"] = {
        "version": torch.__version__,
        "git_version": getattr(torch.version, "git_version", None),
        "file": getattr(torch, "__file__", None),
    }
    result["dlc"] = {
        "available": bool(torch.dlc.is_available()),
    }
    for key, getter in (
        ("device_count", getattr(torch.dlc, "device_count", None)),
        ("current_device", getattr(torch.dlc, "current_device", None)),
    ):
        if getter is None:
            continue
        try:
            result["dlc"][key] = getter()
        except Exception as exc:  # noqa: BLE001
            result["dlc"][key] = f"<error: {type(exc).__name__}: {exc}>"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = collect()
    text = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
