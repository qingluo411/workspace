#!/usr/bin/env python3
"""Device checks for native FP8 E4M3FN initialization on DLC."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


FP8 = torch.float8_e4m3fn
U8 = torch.uint8


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(U8).cpu()


def _expect_bytes(name: str, actual: torch.Tensor, expected: int) -> dict:
    flat = actual.reshape(-1)
    if not torch.all(flat == expected):
        unique = torch.unique(flat).tolist()
        raise AssertionError(f"{name}: expected 0x{expected:02x}, got {unique[:16]}")
    return {"name": name, "status": "PASS", "bytes": int(flat.numel()),
            "value": f"0x{expected:02x}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="dlc:0")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if not torch.dlc.is_available():
        raise RuntimeError("DLC device is not available")
    device = torch.device(args.device)
    checks: list[dict] = []

    zeros = torch.zeros((4, 256, 512), dtype=FP8, device=device)
    checks.append(_expect_bytes("zeros_cache_shape", _bits(zeros), 0x00))

    single = torch.empty((1,), dtype=FP8, device=device)
    single.fill_(float("nan"))
    checks.append(_expect_bytes("single_nan_fill", _bits(single), 0x7f))
    single.fill_(1.0)
    checks.append(_expect_bytes("single_one_fill", _bits(single), 0x38))

    rank0 = torch.empty((), dtype=FP8, device=device)
    rank0.fill_(-2.0)
    checks.append(_expect_bytes("rank0_fill", _bits(rank0), 0xc0))

    expected_values = [
        ("plus_zero", 0.0, 0x00),
        ("minus_zero", -0.0, 0x80),
        ("one", 1.0, 0x38),
        ("max_finite", 448.0, 0x7e),
        ("nan", math.nan, 0x7f),
    ]
    for name, value, expected in expected_values:
        tensor = torch.full((2, 513), value, dtype=FP8, device=device)
        checks.append(_expect_bytes(f"full_{name}", _bits(tensor), expected))

    for name, value in (("finite_overflow", 449.0), ("infinity", math.inf)):
        try:
            torch.full((2,), value, dtype=FP8, device=device)
        except RuntimeError as error:
            if "without overflow" not in str(error):
                raise
            checks.append({"name": f"reject_{name}_like_cpu", "status": "PASS"})
        else:
            raise AssertionError(f"{name} must be rejected before the DLC fill kernel")

    ones = torch.ones((1, 513), dtype=FP8, device=device)
    checks.append(_expect_bytes("ones", _bits(ones), 0x38))

    transposed_backing = torch.zeros((4, 3, 513), dtype=FP8, device=device)
    transposed_backing.transpose(0, 1).fill_(1.0)
    checks.append(_expect_bytes("transposed_view_fill", _bits(transposed_backing), 0x38))

    zeroed = torch.empty((2, 513), dtype=FP8, device=device)
    zeroed.fill_(448.0)
    zeroed.zero_()
    checks.append(_expect_bytes("zero_inplace", _bits(zeroed), 0x00))

    pattern = torch.arange(13, dtype=torch.uint8).reshape(1, 13)
    backing = pattern.view(FP8).to(device)
    backing[0, 3:9].fill_(1.0)
    actual = _bits(backing)[0].tolist()
    expected = pattern[0].tolist()
    for index in range(3, 9):
        expected[index] = 0x38
    if actual != expected:
        raise AssertionError(f"sliced_fill_preserves_neighbors: {actual} != {expected}")
    checks.append({"name": "sliced_fill_preserves_neighbors", "status": "PASS",
                   "bytes": len(actual), "actual": actual, "expected": expected})

    backing[0, 5].fill_(float("nan"))
    actual = _bits(backing)[0].tolist()
    expected[5] = 0x7f
    if actual != expected:
        raise AssertionError(f"single_byte_write_preserves_neighbors: {actual} != {expected}")
    checks.append({"name": "single_byte_write_preserves_neighbors", "status": "PASS",
                   "bytes": len(actual), "actual": actual, "expected": expected})

    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        deterministic = torch.empty((2, 513), dtype=FP8, device=device)
        checks.append(_expect_bytes("deterministic_empty_nan", _bits(deterministic), 0x7f))
        deterministic.zero_()
        checks.append(_expect_bytes("deterministic_empty_then_zero", _bits(deterministic), 0x00))
        deterministic_zeros = torch.zeros((2, 513), dtype=FP8, device=device)
        checks.append(_expect_bytes("deterministic_zeros", _bits(deterministic_zeros), 0x00))
    finally:
        torch.use_deterministic_algorithms(old_deterministic)

    torch.dlc.synchronize()
    report = {
        "status": "PASS",
        "device": str(device),
        "torch": torch.__version__,
        "checks": checks,
        "check_count": len(checks),
    }
    print(f"PASS {len(checks)} FP8 fill/zero checks on {device}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
