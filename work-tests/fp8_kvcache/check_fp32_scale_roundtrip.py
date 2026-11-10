#!/usr/bin/env python3
"""Validate the DLC G128 E4M3 <-> BF16 operators with flat FP32 scales.

The device path uses the formal APIs:
  torch.dlc_quant_e4m3(input, output, scales, 128, scale_round=...)
  torch.dlc_dequant_e4m3(input, scales, 128)
  torch.dlc_dequant_e4m3(input, scales, 128, out=output)

The CPU reference is independent of the DLC kernels. FP8 bytes, FP32 scale
bits, BF16 output bits and the functional/out APIs are checked separately.
This is a storage/ABI test for FP8 KV-cache bring-up, not an end-to-end model
accuracy or vLLM integration result.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch


GROUP_SIZE = 128
FP8 = torch.float8_e4m3fn
PATTERNS = ("normal", "outliers", "group_scales", "tiny", "signed_zero")
SHAPES = ((1, 128), (17, 512), (128, 576), (129, 576), (257, 512))


def make_input(shape, pattern, seed):
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(shape, generator=generator)
    if pattern == "outliers":
        value *= 0.01
        value[:, ::GROUP_SIZE] = 16.0
        value[1::2, ::GROUP_SIZE] = -16.0
    elif pattern == "group_scales":
        groups = torch.arange(shape[1]) // GROUP_SIZE
        value *= torch.exp2(((groups % 5) * 6 - 12).float())
    elif pattern == "tiny":
        powers = -126 + torch.arange(shape[1]) % 107
        value = torch.exp2(powers.float()).expand(shape).clone()
        value[:, 1::2].neg_()
    elif pattern == "signed_zero":
        value.zero_()
        value[:, 1::2] = -0.0
    return value.to(torch.bfloat16)


def quant_reference(src, scale_round):
    rows, width = src.shape
    groups = math.ceil(width / GROUP_SIZE)
    values = torch.nn.functional.pad(src.float(), (0, groups * GROUP_SIZE - width))
    grouped = values.reshape(rows, groups, GROUP_SIZE)
    magnitudes = grouped.abs().masked_fill(torch.isinf(grouped), 0.0)
    # The 1/448 constant and the product are each rounded to FP32 by the kernel.
    scales = magnitudes.amax(dim=-1).clamp(min=1e-4) * torch.tensor(
        1.0 / 448.0, dtype=torch.float32
    )
    if scale_round:
        scales = torch.exp2(torch.ceil(torch.log2(scales.double()))).float()
    normalized = (grouped * torch.reciprocal(scales).unsqueeze(-1)).clamp(-448.0, 448.0)
    normalized = torch.where(torch.isnan(normalized), float("nan"), normalized)
    bits = normalized.reshape(rows, groups * GROUP_SIZE)[:, :width].to(FP8).view(torch.uint8)
    return bits, scales.reshape(-1)


def dequant_reference(bits, scales):
    rows, width = bits.shape
    groups = math.ceil(width / GROUP_SIZE)
    column_scales = scales.reshape(rows, groups).repeat_interleave(GROUP_SIZE, dim=1)[
        :, :width
    ]
    return (bits.view(FP8).float() * column_scales).to(torch.bfloat16)


def exact_equal(name, actual, expected):
    if actual.dtype == torch.bfloat16 or actual.dtype == torch.float32:
        actual = actual.view(torch.int16 if actual.dtype == torch.bfloat16 else torch.int32)
        expected = expected.view(
            torch.int16 if expected.dtype == torch.bfloat16 else torch.int32
        )
    mismatch = actual != expected
    count = mismatch.sum().item()
    if count:
        index = tuple(mismatch.nonzero()[0].tolist())
        raise AssertionError(
            f"{name}: {count}/{actual.numel()} mismatches; first {index}: "
            f"got {actual[index].item()}, expected {expected[index].item()}"
        )


def loss_metrics(src, restored):
    x, y = src.double().flatten(), restored.double().flatten()
    error = y - x
    xnorm, ynorm = x.norm().item(), y.norm().item()
    cosine = ((x / xnorm) @ (y / ynorm)).item() if xnorm and ynorm else float(xnorm == ynorm)
    rms = x.square().mean().sqrt().item()
    rmse = error.square().mean().sqrt().item()
    return {
        "max_abs": error.abs().max().item(),
        "rmse": rmse,
        "normalized_rmse": rmse / rms if rms else 0.0,
        "cosine": cosine,
        "new_zero_fraction": ((x != 0) & (y == 0)).double().mean().item(),
    }


def device_roundtrip(src, ref_bits, ref_scales, scale_round, device):
    rows, width = src.shape
    groups = math.ceil(width / GROUP_SIZE)
    src_device = src.to(device)
    bits = torch.empty(src.shape, dtype=FP8, device=device)
    scales = torch.empty((rows * groups,), dtype=torch.float32, device=device)
    torch.dlc_quant_e4m3(
        src_device, bits, scales, GROUP_SIZE, scale_round=scale_round
    )

    functional = torch.dlc_dequant_e4m3(bits, scales, GROUP_SIZE)
    output = torch.empty(src.shape, dtype=torch.bfloat16, device=device)
    returned = torch.dlc_dequant_e4m3(bits, scales, GROUP_SIZE, out=output)
    if returned is not output:
        raise AssertionError("dequant .out must return the supplied output tensor")
    torch.dlc.synchronize()

    got_bits = bits.cpu().view(torch.uint8)
    got_scales = scales.cpu()
    got_functional = functional.cpu()
    got_out = output.cpu()
    exact_equal("E4M3FN bytes", got_bits, ref_bits)
    exact_equal("FP32 scale bits", got_scales, ref_scales)
    expected = dequant_reference(got_bits, got_scales)
    exact_equal("functional BF16 bits", got_functional, expected)
    exact_equal("out BF16 bits", got_out, expected)
    exact_equal("functional/out identity", got_out, got_functional)
    return got_out


def standalone_dequant_checks(device):
    results = []

    def run(name, check):
        case = {"name": name}
        try:
            check()
            case["status"] = "PASS"
            print(f"PASS API/{name}")
        except Exception as error:
            case.update(status="FAIL", error=str(error))
            print(f"FAIL API/{name}: {error}", file=sys.stderr)
        results.append(case)

    def all_codes():
        bits = torch.arange(256, dtype=torch.uint8).repeat(4, 1)
        scales = torch.tensor(
            [0.3, 1.0, 2.0**-22, 2.0**120], dtype=torch.float32
        ).repeat_interleave(2)
        expected = dequant_reference(bits, scales)
        actual = torch.dlc_dequant_e4m3(
            bits.view(FP8).to(device), scales.to(device), GROUP_SIZE
        ).cpu()
        nan = torch.isnan(expected)
        exact_equal("NaN mask", torch.isnan(actual), nan)
        finite = ~nan
        exact_equal("finite BF16 bits", actual[finite], expected[finite])

    def empty(shape):
        bits = torch.empty(shape, dtype=FP8, device=device)
        scales = torch.empty(
            (shape[0] * math.ceil(shape[1] / GROUP_SIZE),),
            dtype=torch.float32,
            device=device,
        )
        actual = torch.dlc_dequant_e4m3(bits, scales, GROUP_SIZE)
        if actual.shape != bits.shape or actual.dtype != torch.bfloat16:
            raise AssertionError(f"unexpected empty output {actual.shape}/{actual.dtype}")

    run("all_256_codes", all_codes)
    for shape in ((0, 576), (2, 0)):
        run(f"empty_{shape[0]}x{shape[1]}", lambda shape=shape: empty(shape))

    bits = torch.empty((2, 256), dtype=FP8, device=device)
    scales = torch.empty((2 * 2,), dtype=torch.float32, device=device)
    output = torch.empty((2, 256), dtype=torch.bfloat16, device=device)
    invalid = (
        ("group32", (bits, scales, 32), {}, "only supports group_size of 128"),
        ("input_uint8", (bits.view(torch.uint8), scales, GROUP_SIZE), {}, "float8_e4m3fn"),
        ("scale_uint8", (bits, scales.to(torch.uint8), GROUP_SIZE), {}, "float32"),
        ("output_fp32", (bits, scales, GROUP_SIZE), {"out": output.float()}, "bfloat16"),
        ("short_scale", (bits, scales[:3], GROUP_SIZE), {}, "M * ceil(K/128)"),
    )

    def must_reject(args, kwargs, message):
        try:
            torch.dlc_dequant_e4m3(*args, **kwargs)
        except RuntimeError as error:
            if message not in str(error):
                raise AssertionError(f"expected '{message}' in: {error}") from error
        else:
            raise AssertionError("invalid arguments were accepted")

    for name, args, kwargs, message in invalid:
        run(
            name,
            lambda args=args, kwargs=kwargs, message=message: must_reject(
                args, kwargs, message
            ),
        )
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="dlc:0")
    parser.add_argument("--shape", action="append", type=lambda s: tuple(map(int, s.split(","))))
    parser.add_argument("--patterns", nargs="+", choices=PATTERNS, default=PATTERNS)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if torch.device(args.device).type != "dlc":
        parser.error("--device must select DLC")
    missing = [name for name in ("dlc_quant_e4m3", "dlc_dequant_e4m3") if not hasattr(torch, name)]
    if missing:
        parser.error(f"missing torch API: {', '.join(missing)}")

    samples = [
        (f"{rows}x{width}/{pattern}/{scale_round}",
         make_input((rows, width), pattern, args.seed + index),
         scale_round)
        for index, (rows, width) in enumerate(args.shape or SHAPES)
        for pattern in args.patterns
        for scale_round in (True, False)
    ]
    report = {
        "device": args.device,
        "group_size": GROUP_SIZE,
        "format": "OCP E4M3FN, one byte, RNE, finite saturation at +/-448",
        "scale_format": "flat FP32 [T*ceil(D/128)]",
        "cases": [],
        "api_checks": [],
    }
    failures = 0
    print(
        f"device={args.device}; group_size=128; {len(samples)} cases; "
        "FP8 bytes/scales/BF16 bits are checked independently"
    )
    for name, src, scale_round in samples:
        case = {"name": name, "shape": list(src.shape), "scale_round": scale_round}
        try:
            ref_bits, ref_scales = quant_reference(src, scale_round)
            restored = device_roundtrip(src, ref_bits, ref_scales, scale_round, args.device)
            metrics = loss_metrics(src, restored)
            per_token = src.shape[1] + 4 * math.ceil(src.shape[1] / GROUP_SIZE)
            case.update(status="PASS", metrics=metrics, fp8_bytes_per_token=per_token,
                        bf16_bytes_per_token=2 * src.shape[1])
            print(
                f"PASS {name:30s} max_abs={metrics['max_abs']:.6g} "
                f"rmse={metrics['rmse']:.6g} cos={metrics['cosine']:.8f} "
                f"bytes={per_token}/{2 * src.shape[1]}"
            )
        except Exception as error:
            failures += 1
            case.update(status="FAIL", error=str(error))
            print(f"FAIL {name}: {error}", file=sys.stderr)
        report["cases"].append(case)

    report["api_checks"] = standalone_dequant_checks(args.device)
    failures += sum(case["status"] == "FAIL" for case in report["api_checks"])
    report["failures"] = failures
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    total = len(samples) + len(report["api_checks"])
    print(f"{total - failures}/{total} passed")
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
