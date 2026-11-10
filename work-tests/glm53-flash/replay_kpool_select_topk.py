#!/usr/bin/env python3
"""Replay captured real-model inputs through custom_vllm_kpool_select_topk."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

DEFAULT_CAPTURE_ROOT = Path(__file__).resolve().parent / "kpool-real-input-16k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT
    )
    parser.add_argument(
        "--bundle",
        action="append",
        default=None,
        help="Capture directory; repeat to select several. Defaults to every capture.",
    )
    parser.add_argument("--device", default="dlc:0")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output (default: <capture-root>/replay_results.json)",
    )
    return parser.parse_args()


def discover_bundles(root: Path, selected: list[str] | None) -> list[Path]:
    if selected:
        bundles = [Path(item) for item in selected]
    else:
        bundles = sorted(root.glob("capture_*"))
    valid = [path for path in bundles if (path / "metadata.json").is_file()]
    if not valid:
        raise SystemExit(f"no completed captures under {root}")
    return valid


def load_bundle(path: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    metadata = json.loads((path / "metadata.json").read_text())
    tensors = torch.load(path / "tensors.pt", map_location="cpu", weights_only=True)
    moved = {name: tensor.to(device) for name, tensor in tensors.items()}
    return metadata, moved


def call_operator(
    metadata: dict[str, Any], tensors: dict[str, torch.Tensor]
) -> torch.Tensor:
    out = torch.empty_like(tensors["output"], device=tensors["query"].device)
    torch.dlc_vllm_kpool_select_topk(
        tensors["query"],
        tensors["head_weights"],
        tensors["pool_cache"],
        tensors["block_table"],
        tensors["query_start_loc"],
        tensors["seq_lens"],
        int(metadata["topk_tokens"]),
        int(metadata["pool_size"]),
        float(metadata["scale"]),
        float(metadata["head_weight_scale"]),
        out=out,
    )
    return out


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("warmup must be >= 0 and iterations must be > 0")
    if not torch.dlc.is_available():
        raise SystemExit("torch.dlc is not available")

    device = torch.device(args.device)
    torch.dlc.set_device(device)
    bundles = discover_bundles(args.capture_root, args.bundle)
    rows: list[dict[str, Any]] = []

    for bundle in bundles:
        metadata, tensors = load_bundle(bundle, device)
        print(
            f"\n{bundle.name}: T={metadata['query_tokens']} "
            f"context={metadata['max_context']} "
            f"query={tuple(metadata['query_shape'])} "
            f"pool={tuple(metadata['pool_cache_shape'])}",
            flush=True,
        )

        for _ in range(args.warmup):
            call_operator(metadata, tensors)
        torch.dlc.synchronize()

        verify_ok = True
        mismatch = 0
        if not args.no_verify and "output" in tensors:
            got = call_operator(metadata, tensors).cpu()
            expected = tensors["output"].cpu()
            verify_ok = torch.equal(got, expected)
            mismatch = int((got != expected).sum().item())
            del got, expected
            torch.dlc.synchronize()

        wall_ms: list[float] = []
        event_ms: list[float] = []
        for _ in range(args.iterations):
            start = torch.dlc.Event(enable_timing=True)
            end = torch.dlc.Event(enable_timing=True)
            started = time.perf_counter()
            start.record()
            call_operator(metadata, tensors)
            end.record()
            end.synchronize()
            wall_ms.append((time.perf_counter() - started) * 1000.0)
            event_ms.append(start.elapsed_time(end))

        row = {
            "bundle": bundle.name,
            "query_tokens": metadata["query_tokens"],
            "max_context": metadata["max_context"],
            "query_shape": metadata["query_shape"],
            "pool_cache_shape": metadata["pool_cache_shape"],
            "verify_exact": verify_ok,
            "mismatch_elements": mismatch,
            "iterations": args.iterations,
            "wall_ms": wall_ms,
            "wall_median_ms": statistics.median(wall_ms),
            "device_event_ms": event_ms,
            "device_event_median_ms": statistics.median(event_ms),
        }
        rows.append(row)
        print(
            f"  exact={verify_ok} mismatch={mismatch} "
            f"device_median={row['device_event_median_ms']:.3f} ms "
            f"wall_median={row['wall_median_ms']:.3f} ms",
            flush=True,
        )

    output = args.output or (args.capture_root / "replay_results.json")
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {output}", flush=True)
    return 0 if all(row["verify_exact"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
