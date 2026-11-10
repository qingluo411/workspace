#!/usr/bin/env python3
"""Check the existing group128 BF16 -> E4M3FN -> BF16 path.

Examples (from /root/workspace):
  uv run --no-sync --project . .venv/bin/python \
      work-tests/fp8_kvcache/check_fp8_g128_roundtrip.py --cpu-only
  uv run --no-sync --project . .venv/bin/python \
      work-tests/fp8_kvcache/check_fp8_g128_roundtrip.py --device dlc:0
  # A torch.save(cpu_bf16_tensor, path) sample, with shape [T, D]:
  .../check_fp8_g128_roundtrip.py --input sample.pt --json result.json

The temporary PyTorch source registration has been archived and reverted.
Device mode requires those dedicated APIs in the loaded build; rebuilding the
restored PyTorch sources alone does not add them. CPU-only mode stays usable.

This tests implementation correctness and reports quantization loss; it does
not establish model accuracy or an acceptable KV-cache quality threshold.
The scale rule deliberately matches today's group128 kernel, including its
1e-10 floor. It is not a declaration of full MXFP8 specification compatibility.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


GROUP_SIZE = 128
FP8 = torch.float8_e4m3fn
PATTERNS = ("signed_zero", "constant", "normal", "tiny", "group_scales", "outliers")
SHAPES = [(t, d) for t in (1, 17, 129, 257) for d in (64, 128, 129, 512, 576)]


def decode_e4m3fn(bits):
    """Independent CPU decoder, including signed zeros and the two NaN codes."""
    raw = bits.to(torch.int32)
    magnitude = raw & 127
    exponent = magnitude >> 3
    mantissa = magnitude & 7
    normal = (1.0 + mantissa.float() / 8.0) * torch.exp2(exponent.float() - 7.0)
    subnormal = mantissa.float() * (2.0 ** -9)
    value = torch.where(exponent == 0, subnormal, normal)
    value = torch.where((raw & 128) != 0, -value, value)
    return torch.where(magnitude == 127, float("nan"), value)


def dequant_reference(bits, codes):
    groups = torch.arange(bits.shape[1]) // GROUP_SIZE
    scales = torch.exp2(codes.float() - 127.0)
    return (decode_e4m3fn(bits) * scales[:, groups]).to(torch.bfloat16)


def quant_reference(src):
    """CPU FP32 group scale calculation + PyTorch CPU E4M3FN RNE encoder."""
    tokens, hidden = src.shape
    num_groups = math.ceil(hidden / GROUP_SIZE)
    grouped = F.pad(src.float(), (0, num_groups * GROUP_SIZE - hidden))
    grouped = grouped.reshape(tokens, num_groups, GROUP_SIZE)
    max_abs = grouped.abs().amax(dim=-1)
    exponent = torch.ceil(torch.log2((max_abs / 448.0).clamp_min(1e-10)))
    scales = torch.exp2(exponent)
    codes = (exponent + 127.0).clamp(0, 255).to(torch.uint8)
    normalized = grouped * scales.reciprocal().unsqueeze(-1)
    bits = normalized.reshape(tokens, -1)[:, :hidden].to(FP8).view(torch.uint8)
    return bits, codes


def check_reference():
    bits = torch.arange(256, dtype=torch.uint8)
    decoded = decode_e4m3fn(bits)
    native = bits.view(FP8).float()
    torch.testing.assert_close(decoded, native, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(native)
    assert torch.equal(torch.signbit(decoded[finite]), torch.signbit(native[finite]))
    for value, expected_bits, expected_scale in (
        (0.0, 0x00, 94), (-0.0, 0x80, 94),
        (1.0, 0x78, 119), (-1.0, 0xF8, 119), (448.0, 0x7E, 127),
    ):
        src = torch.full((1, 128), value, dtype=torch.bfloat16)
        q, s = quant_reference(src)
        assert bool((q == expected_bits).all()), (value, q)
        assert s.item() == expected_scale, (value, s)
        assert torch.equal(dequant_reference(q, s).view(torch.int16), src.view(torch.int16))


def make_input(shape, pattern, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=generator)
    if pattern == "signed_zero":
        x.zero_()
        x[:, 1::2] = -0.0
    elif pattern == "constant":
        x.fill_(1.0)
        x[1::2].neg_()
    elif pattern == "tiny":
        # Finite BF16 normal values; today's scale floor makes many become FP8 zero.
        powers = -126 + torch.arange(shape[1]) % 107
        x = torch.exp2(powers.float()).expand(shape).clone()
        x[:, 1::2].neg_()
    elif pattern == "group_scales":
        groups = torch.arange(shape[1]) // GROUP_SIZE
        x *= torch.exp2(((groups % 5) * 6 - 12).float())
    elif pattern == "outliers":
        x *= 0.01
        # One outlier in every group, with alternating signs across tokens.
        x[:, ::GROUP_SIZE] = 16.0
        x[1::2, ::GROUP_SIZE] = -16.0
    return x.to(torch.bfloat16)


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
        "zero_fraction": (y == 0).double().mean().item(),
        "new_zero_fraction": ((x != 0) & (y == 0)).double().mean().item(),
    }


def exact_check(name, got, expected):
    mismatch = got != expected
    count = mismatch.sum().item()
    if count:
        index = tuple(mismatch.nonzero()[0].tolist())
        raise AssertionError(
            f"{name}: {count}/{got.numel()} mismatches; first {index}: "
            f"got {got[index].item()}, expected {expected[index].item()}"
        )


def device_roundtrip(src, ref_bits, ref_codes, device):
    tokens, hidden = src.shape
    # Native DLC float8 fill is not needed: initialize on CPU, then copy the
    # same dtype. Byte views here only expose storage; they do not dequantize.
    out = torch.zeros(src.shape, dtype=torch.uint8).view(FP8).to(device)
    codes = torch.empty((tokens, math.ceil(hidden / GROUP_SIZE)), dtype=torch.uint8, device=device)
    restored = torch.empty(src.shape, dtype=torch.bfloat16, device=device)
    torch.dlc_quant_e4m3(src.to(device), out, codes, GROUP_SIZE)
    if torch.dlc_dequant_e4m3(out, codes, restored, GROUP_SIZE) is not None:
        raise AssertionError("dequant must write the supplied output and return None")
    # Copy bytes and BF16 without a generic device float8 -> BF16 conversion.
    got_bits, got_codes, got_restored = out.view(torch.uint8).cpu(), codes.cpu(), restored.cpu()
    exact_check("UE8M0 scales", got_codes, ref_codes)
    exact_check("E4M3FN bytes", got_bits, ref_bits)
    expected = dequant_reference(got_bits, got_codes)
    exact_check("restored BF16 bits", got_restored.view(torch.int16), expected.view(torch.int16))
    return got_restored


def api_checks(device):
    """Small device-only dequant/API checks; invalid arguments must not launch."""
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

    def dequant(bits, codes, output, group_size=GROUP_SIZE):
        if torch.dlc_dequant_e4m3(bits, codes, output, group_size) is not None:
            raise AssertionError("dequant must return None")

    def all_codes():
        bits = torch.arange(256, dtype=torch.uint8).repeat(3, 1)
        codes = torch.tensor([[126, 126], [127, 127], [128, 128]], dtype=torch.uint8)
        expected = dequant_reference(bits, codes)
        output = torch.empty(bits.shape, dtype=torch.bfloat16, device=device)
        dequant(bits.view(FP8).to(device), codes.to(device), output)
        got = output.cpu()
        exact_check("NaN mask", torch.isnan(got), torch.isnan(expected))
        finite = torch.isfinite(expected)
        exact_check("finite BF16 bits", got.view(torch.int16)[finite], expected.view(torch.int16)[finite])

    def empty(shape):
        bits = torch.empty(shape, dtype=torch.uint8).view(FP8).to(device)
        codes = torch.empty((shape[0], math.ceil(shape[1] / GROUP_SIZE)), dtype=torch.uint8, device=device)
        dequant(bits, codes, torch.empty(shape, dtype=torch.bfloat16, device=device))

    run("all_256_codes_scales_126_127_128", all_codes)
    for shape in ((0, 576), (2, 0)):
        run(f"empty_{shape[0]}x{shape[1]}", lambda shape=shape: empty(shape))

    bits = torch.zeros((2, 256), dtype=torch.uint8).view(FP8).to(device)
    codes = torch.empty((2, 2), dtype=torch.uint8, device=device)
    output = torch.empty((2, 256), dtype=torch.bfloat16, device=device)

    def must_reject(args, expected_message):
        try:
            torch.dlc_dequant_e4m3(*args)
        except RuntimeError as error:
            if expected_message not in str(error):
                raise AssertionError(f"expected '{expected_message}' in: {error}") from error
        else:
            raise AssertionError("invalid arguments were accepted")

    # Metadata-only invalid views, plus wrong dtypes/shapes. Every case keeps
    # the other arguments valid so the intended wrapper check is exercised.
    offset_bits = torch.zeros((3, 256), dtype=torch.uint8).view(FP8).to(device)[1:]
    alias_base = torch.zeros((1, 512), dtype=torch.uint8).to(device)
    invalid = [
        ("group32", (bits, codes, output, 32), "group_size=128"),
        ("input_uint8", (bits.view(torch.uint8), codes, output), "float8_e4m3fn input"),
        ("scale_fp32", (bits, torch.empty((2, 2), dtype=torch.float32, device=device), output), "uint8 scale codes"),
        ("output_fp32", (bits, codes, torch.empty((2, 256), dtype=torch.float32, device=device)), "bfloat16 output"),
        ("rank1", (torch.zeros(512, dtype=torch.uint8).view(FP8).to(device), codes, output), "expects 2D"),
        ("output_shape", (bits, codes, torch.empty((2, 128), dtype=torch.bfloat16, device=device)), "shapes must match"),
        ("scale_shape", (bits, torch.empty((2, 1), dtype=torch.uint8, device=device), output), "scale shape"),
        ("noncontiguous", (bits, codes.t(), output), "contiguous tensors"),
        ("storage_offset", (offset_bits, codes, output), "storage_offset=0"),
        ("output_alias", (alias_base[:, :256].view(FP8), codes[:1], alias_base.view(torch.bfloat16)), "must not share storage"),
    ]
    for name, args, message in invalid:
        run(name, lambda args=args, message=message: must_reject(args, message))
    return results


def parse_shape(text):
    try:
        shape = tuple(int(n) for n in text.split(","))
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError
        return shape
    except ValueError:
        raise argparse.ArgumentTypeError("shape must be two positive integers: T,D") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu-only", action="store_true", help="validate/reference and measure CPU roundtrips; never call DLC")
    parser.add_argument("--device", default="dlc:0", help="DLC device for both dedicated operators (default: dlc:0)")
    parser.add_argument("--shape", action="append", type=parse_shape, help="repeat to override the default 20 [T,D] shapes")
    parser.add_argument("--patterns", nargs="+", choices=PATTERNS, default=PATTERNS)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--input", type=Path, help="torch.save CPU BF16 tensor [T,D]; run only this sample")
    parser.add_argument("--json", type=Path, help="write a machine-readable report, including failures")
    args = parser.parse_args()
    torch.set_num_threads(1)
    check_reference()

    if not args.cpu_only:
        missing = [name for name in ("dlc_quant_e4m3", "dlc_dequant_e4m3") if not hasattr(torch, name)]
        if missing:
            parser.error(
                f"Missing torch API: {', '.join(missing)}. This archived device test requires "
                "the temporary registration or a future formal implementation in the loaded build; "
                "rebuilding the restored PyTorch sources alone does not add these APIs. "
                "Use --cpu-only for the independent CPU reference."
            )
        if torch.device(args.device).type != "dlc":
            parser.error("--device must select DLC; use --cpu-only for the independent CPU reference")

    if args.input:
        src = torch.load(args.input, map_location="cpu", weights_only=True)
        if not isinstance(src, torch.Tensor) or src.dtype != torch.bfloat16 or src.ndim != 2 or min(src.shape) < 1:
            parser.error("--input must contain one nonempty CPU BF16 tensor with shape [T,D]")
        samples = [(str(args.input), src.contiguous())]
    else:
        samples = [
            (f"{t}x{d}/{pattern}", make_input((t, d), pattern, args.seed + index))
            for index, (t, d) in enumerate(args.shape or SHAPES)
            for pattern in args.patterns
        ]

    mode = "cpu_reference_only" if args.cpu_only else args.device
    print(f"mode={mode}; group_size=128; {len(samples)} cases; quantization loss is informational")
    report = {"mode": mode, "group_size": GROUP_SIZE, "cases": [], "api_checks": []}
    failures = 0
    for name, src in samples:
        case = {"name": name, "shape": list(src.shape)}
        try:
            if not bool(torch.isfinite(src).all()):
                raise ValueError("roundtrip input must be finite; NaN/Inf belong in standalone dequant tests")
            bits, codes = quant_reference(src)
            restored = (dequant_reference(bits, codes) if args.cpu_only else
                        device_roundtrip(src, bits, codes, args.device))
            metrics = loss_metrics(src, restored)
            case.update(status="PASS", metrics=metrics)
            print(f"PASS {name:24s} max_abs={metrics['max_abs']:.6g} rmse={metrics['rmse']:.6g} "
                  f"cos={metrics['cosine']:.8f} zero={metrics['zero_fraction']:.2%} "
                  f"new_zero={metrics['new_zero_fraction']:.2%}")
        except Exception as error:
            failures += 1
            case.update(status="FAIL", error=str(error))
            print(f"FAIL {name}: {error}", file=sys.stderr)
        report["cases"].append(case)
    if not args.cpu_only:
        try:
            report["api_checks"] = api_checks(args.device)
        except Exception as error:
            print(f"FAIL API/setup: {error}", file=sys.stderr)
            report["api_checks"] = [{"name": "setup", "status": "FAIL", "error": str(error)}]
        failures += sum(case["status"] == "FAIL" for case in report["api_checks"])
    report["failures"] = failures
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    total = len(samples) + len(report["api_checks"])
    print(f"{total - failures}/{total} passed; " +
          ("CPU reference only; DLC execution has not been tested." if args.cpu_only else
           "FP8 bytes, scale codes and restored BF16 bits checked against CPU reference."))
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
