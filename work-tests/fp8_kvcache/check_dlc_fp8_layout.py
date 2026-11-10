#!/usr/bin/env python3
"""Measure DLC address strides relevant to a paged FP8 KV cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _stride0(tensor: torch.Tensor, axis: int = 0) -> int:
    if tensor.shape[axis] < 2:
        raise ValueError("need at least two elements along the measured axis")
    first = tensor.select(axis, 0).data_ptr()
    second = tensor.select(axis, 1).data_ptr()
    return second - first


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="dlc:0")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    blocks, block_size, width = 4, 256, 512
    slots = blocks * block_size
    cases = []

    def add(name: str, shape: tuple[int, ...], dtype: torch.dtype,
            axis: int, expected: int) -> None:
        tensor = torch.empty(shape, dtype=dtype, device=device)
        measured = _stride0(tensor, axis)
        if measured != expected:
            raise AssertionError(f"{name}: stride {measured} != {expected}")
        cases.append({
            "name": name,
            "shape": list(shape),
            "dtype": str(dtype),
            "axis": axis,
            "address_stride_bytes": measured,
            "expected_bytes": expected,
            "status": "PASS",
        })

    add("bf16_value_row", (blocks, 1, block_size, width), torch.bfloat16,
        0, block_size * width * 2)
    add("fp8_value_row", (blocks, 1, block_size, width), torch.float8_e4m3fn,
        0, block_size * width)
    add("fp8_legacy_uint8_scale_516_row_3d", (blocks, block_size, 516), torch.float8_e4m3fn,
        0, block_size * 1024)
    add("fp8_fp32_scale_528_row_3d", (blocks, block_size, 528), torch.float8_e4m3fn,
        0, block_size * 1024)
    add("scale_slot4_2d", (slots, 4), torch.float32, 0, 512)
    add("scale_flat", (slots * 4,), torch.float32, 0, 4)
    add("scale_group_major", (4, slots), torch.float32, 1, 4)

    torch.dlc.synchronize()
    report = {
        "status": "PASS",
        "device": str(device),
        "torch": torch.__version__,
        "geometry": {"blocks": blocks, "block_size": block_size,
                     "width": width, "slots": slots},
        "cases": cases,
        "conclusions": {
            "fp8_value_bytes_per_token": width,
            "bf16_value_bytes_per_token": width * 2,
            "fp8_value_bytes_per_block": block_size * width,
            "bf16_value_bytes_per_block": block_size * width * 2,
            "fp32_scale_bytes_per_token": 4 * 4,
            "fp8_legacy_uint8_scale_516_row_3d_bytes_per_token": 1024,
            "fp8_fp32_scale_528_row_3d_bytes_per_token": 1024,
            "scale_slot4_bytes_per_token": 512,
            "recommended_flat_bytes_per_token": width + 4 * 4,
        },
    }
    print("PASS 7 DLC FP8 layout stride checks")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
