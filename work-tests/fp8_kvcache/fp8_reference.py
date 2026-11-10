#!/usr/bin/env python3
"""CPU golden rules for the proposed FP8 operator requests, not device kernels.

Proposed packed group128 and MXFP8 E4M3 block32 share the OCP MX v1.0 section
6.3 recommended finite scale recipe, E8M0 code255-as-NaN decoding, and the
documented project policy for nonfinite inputs. Group128 is not a standard MX
format: only the block32 path is MXFP8. Physical packing is implementation-
defined in OCP MX and does not select the numerical behavior.

The separately named legacy_g128_* rules retain today's 1e-10 scale floor,
ceil(log2(amax/448)) recipe, and code255-as-Inf behavior for compatibility
comparisons. They must not serve as the golden for the new packed operators.

Run this file with --self-test. All tensors and operations stay on the CPU;
passing these checks does not establish DLC implementation or hardware support.
"""

import argparse
import math

import torch
import torch.nn.functional as F


FP8 = torch.float8_e4m3fn


def _tensor(value, dtype, rank, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu" or value.dtype != dtype:
        raise ValueError(f"{name} must be CPU {dtype}")
    if rank is not None and value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")


def _dimensions(T, D, group_size):
    if not isinstance(T, int) or not isinstance(D, int) or T < 0 or D < 0:
        raise ValueError("T and D must be nonnegative integers")
    if group_size not in (32, 128):
        raise ValueError("group_size must be 32 or 128")
    G = (D + group_size - 1) // group_size
    Q = (G + 3) // 4
    return G, Q, ((T + 127) // 128) * Q


def pack_scales(codes):
    """Pack CPU uint8[T,G] to logical uint8[num_tiles,512], with zero padding.

    tile = (t // 128) * ceil(G / 4) + g // 4
    k = (t % 128) * 4 + g % 4

    These are logical CPU bytes. Same-dtype CPU-to-DLC copy applies the normal
    native uint8 byte weave; do not apply physical_byte_offset() to this result.
    Empty T or G produces [0,512]. No existing padded storage is reused.
    """
    _tensor(codes, torch.uint8, 2, "codes")
    T, G = codes.shape
    Q = (G + 3) // 4
    tiles = torch.zeros((((T + 127) // 128) * Q, 512), dtype=torch.uint8)
    if T and G:
        t = torch.arange(T)[:, None]
        g = torch.arange(G)[None, :]
        tiles[(t // 128) * Q + g // 4, (t % 128) * 4 + g % 4] = codes
    return tiles


def unpack_scales(scale_tiles, T, D, group_size):
    """Recover logical CPU uint8[T,ceil(D/group_size)] from logical tiles.

    Padding bytes are ignored. Supports both group128 and group32; the layout
    stores no shape metadata, so callers must supply T, D, and group_size.
    """
    _tensor(scale_tiles, torch.uint8, 2, "scale_tiles")
    G, Q, count = _dimensions(T, D, group_size)
    if tuple(scale_tiles.shape) != (count, 512):
        raise ValueError(f"scale_tiles must have shape [{count},512]")
    if not T or not G:
        return torch.empty((T, G), dtype=torch.uint8)
    t = torch.arange(T)[:, None]
    g = torch.arange(G)[None, :]
    return scale_tiles[(t // 128) * Q + g // 4, (t % 128) * 4 + g % 4]


def physical_byte_offset(tile, k):
    """DLC physical byte offset for logical byte k of a native uint8 tile row.

    This explains the existing 4-plane/128-lane weave after CPU-to-DLC copy:
    offset = tile*512 + 4*(k%128) + k//128. It is not NVIDIA's 128x4 swizzle,
    and is not an extra permutation to apply to pack_scales() on the CPU.
    """
    if not isinstance(tile, int) or tile < 0 or not isinstance(k, int) or not 0 <= k < 512:
        raise ValueError("tile must be nonnegative and k must be in [0,512)")
    return tile * 512 + 4 * (k % 128) + k // 128


def decode_e4m3fn(bits):
    """Independently decode CPU uint8 of any shape to FP32, preserving +/-0."""
    _tensor(bits, torch.uint8, None, "bits")
    raw = bits.to(torch.int32)
    magnitude = raw & 127
    exponent = magnitude >> 3
    mantissa = magnitude & 7
    normal = (1.0 + mantissa.float() / 8.0) * torch.exp2(exponent.float() - 7.0)
    subnormal = mantissa.float() * (2.0 ** -9)
    value = torch.where(exponent == 0, subnormal, normal)
    value = torch.where((raw & 128) != 0, -value, value)
    return torch.where(magnitude == 127, float("nan"), value)


def _grouped(src, group_size):
    _tensor(src, torch.bfloat16, 2, "src")
    T, D = src.shape
    G = (D + group_size - 1) // group_size
    grouped = F.pad(src.float(), (0, G * group_size - D)).reshape(T, G, group_size)
    return T, D, G, grouped


def _dequant_inputs(bits, codes, group_size):
    _tensor(bits, torch.uint8, 2, "bits")
    _tensor(codes, torch.uint8, 2, "codes")
    T, D = bits.shape
    G = (D + group_size - 1) // group_size
    if tuple(codes.shape) != (T, G):
        raise ValueError(f"codes must have shape [{T},{G}]")
    return T, D, torch.arange(D) // group_size


def legacy_g128_quant(src):
    """Existing finite BF16 group128 golden: returns uint8 bits[T,D], codes[T,G]."""
    T, D, G, grouped = _grouped(src, 128)
    if not bool(torch.isfinite(src).all()):
        raise ValueError("legacy g128 quant golden accepts only finite input")
    if not T or not D:
        return torch.empty((T, D), dtype=torch.uint8), torch.empty((T, G), dtype=torch.uint8)
    amax = grouped.abs().amax(dim=-1)
    e = torch.ceil(torch.log2((amax / 448.0).clamp_min(1e-10)))
    codes = (e + 127.0).clamp(0, 255).to(torch.uint8)
    scaled = grouped * torch.exp2(e).reciprocal().unsqueeze(-1)
    bits = scaled.reshape(T, G * 128)[:, :D].to(FP8).view(torch.uint8)
    return bits.contiguous(), codes


def legacy_g128_dequant(bits, codes):
    """Legacy FP8 group128 -> BF16; scale255 means exp2(128)=Inf, not MX NaN.

    In particular, a zero element times code255 produces NaN, while a nonzero
    finite element produces signed Inf. Keep this behavior out of MX decoding.
    """
    _, _, groups = _dequant_inputs(bits, codes, 128)
    scales = torch.exp2(codes.float() - 127.0)
    return (decode_e4m3fn(bits) * scales[:, groups]).to(torch.bfloat16)


def _ocp_quant(src, group_size, saturate):
    """Shared proposed OCP numerical policy; block size is a caller choice.

    For finite amax>0: e=clamp(floor(log2(amax))-8,-127,127), code=e+127.
    NaN/Inf are excluded from amax (project policy); amax=0 selects scale=1.
    Element conversion is RNE, SAT by default, OVF if saturate=False. NaNs
    canonicalize to 0x7f. Inf becomes signed448 in SAT and NaN in OVF. Signed
    zeros/subnormals are preserved by the arithmetic; there is no 1e-10 floor.
    Quantization does not emit scale255; dequantization must accept it.
    """
    if not isinstance(saturate, bool):
        raise TypeError("saturate must be bool")
    T, D, G, grouped = _grouped(src, group_size)
    if not T or not D:
        return torch.empty((T, D), dtype=torch.uint8), torch.empty((T, G), dtype=torch.uint8)
    finite_abs = torch.where(torch.isfinite(grouped), grouped.abs(), 0.0)
    amax = finite_abs.amax(dim=-1)
    # frexp avoids a log2 approximation choosing the wrong side of a power of 2.
    _, exponent = torch.frexp(amax)
    e = torch.where(amax > 0, exponent - 1 - 8, 0).clamp(-127, 127)
    codes = (e + 127).to(torch.uint8)
    normalized = torch.ldexp(grouped, -e.unsqueeze(-1))
    if saturate:
        normalized = normalized.clamp(-448.0, 448.0)
    q = normalized.to(FP8).view(torch.uint8)
    q = torch.where(torch.isnan(normalized), 0x7F, q)
    # OVF may produce either sign of NaN; this project fixes the payload/sign.
    q = torch.where((q & 127) == 127, 0x7F, q)
    return q.reshape(T, G * group_size)[:, :D].contiguous(), codes


def _ocp_dequant(bits, codes, group_size):
    """Shared proposed FP8 -> BF16 policy with full E8M0 decoding.

    Code255 makes the entire block NaN, regardless of element encodings. Codes
    0..254 mean 2**(code-127). Multiply in FP32, then round to BF16 with RNE;
    preserve subnormals and signed zero. Product overflow is defined here as
    signed Inf, including combinations outside the Float32 range (for which
    OCP MX leaves behavior implementation-defined). NaN payload is unspecified.
    """
    _, _, groups = _dequant_inputs(bits, codes, group_size)
    scale_nan = codes == 255
    safe_codes = torch.where(scale_nan, 127, codes)
    scales = torch.exp2(safe_codes.float() - 127.0)
    values = decode_e4m3fn(bits) * scales[:, groups]
    values = torch.where(scale_nan[:, groups], float("nan"), values)
    return values.to(torch.bfloat16)


def g128_quant(src, saturate=True):
    """Proposed packed-group128 golden using _ocp_quant(), not standard MX.

    Returns logical uint8 bits[T,D] and codes[T,ceil(D/128)]; pack_scales()
    separately defines storage. This intentionally differs from legacy_g128_*.
    """
    return _ocp_quant(src, 128, saturate)


def g128_dequant(bits, codes):
    """Proposed group128 golden: E8M0 code255 makes all 128 elements NaN."""
    return _ocp_dequant(bits, codes, 128)


def mx_quant(src, saturate=True):
    """Proposed MXFP8 E4M3 block32 golden using _ocp_quant().

    Returns logical uint8 bits[T,D] and codes[T,ceil(D/32)]; pack_scales()
    separately defines the implementation-specific physical scale layout.
    """
    return _ocp_quant(src, 32, saturate)


def mx_dequant(bits, codes):
    """Proposed MXFP8 block32 golden: E8M0 code255 makes the block NaN."""
    return _ocp_dequant(bits, codes, 32)


def _same_values(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    mask = ~torch.isnan(expected)
    storage = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(actual.view(storage)[mask], expected.view(storage)[mask])


def _self_test():
    # Independent bit-field decoder vs native CPU decoder, including zero signs.
    all_bits = torch.arange(256, dtype=torch.uint8)
    _same_values(decode_e4m3fn(all_bits), all_bits.view(FP8).float())

    # Fixed old-format golden values, independent of the formulas under test.
    for value, byte, scale in ((0.0, 0, 94), (-0.0, 128, 94),
                               (1.0, 0x78, 119), (-1.0, 0xF8, 119), (448.0, 0x7E, 127)):
        src = torch.full((1, 128), value, dtype=torch.bfloat16)
        bits, codes = legacy_g128_quant(src)
        assert bool((bits == byte).all()) and codes.item() == scale
        _same_values(legacy_g128_dequant(bits, codes), src)

    # Both new operators share numerical policy. Fixed values below are
    # handwritten golden bytes/codes, not computed with the shared helpers.
    for B, quant, dequant in ((128, g128_quant, g128_dequant),
                              (32, mx_quant, mx_dequant)):
        for value, sat_byte, ovf_byte in ((480.0, 0x7E, 0x7F),
                                         (464.0, 0x7E, 0x7E), (466.0, 0x7E, 0x7F)):
            src = torch.full((1, B), value, dtype=torch.bfloat16)
            sat, codes = quant(src)
            ovf, other_codes = quant(src, saturate=False)
            assert codes.item() == 127 and torch.equal(codes, other_codes)
            assert bool((sat == sat_byte).all()) and bool((ovf == ovf_byte).all())
            assert bool((dequant(sat, codes) == 448.0).all())

        # A 256 anchor fixes scale=1; test exact halfway values and zero signs.
        src = torch.zeros((1, B), dtype=torch.bfloat16)
        src[0, :6] = torch.tensor([0.0, -0.0, 2.0**-10, -(2.0**-10),
                                    3 * 2.0**-10, -3 * 2.0**-10], dtype=torch.bfloat16)
        src[0, B - 1] = 256.0
        bits, codes = quant(src)
        assert codes.item() == 127 and bits[0, :6].tolist() == [0, 128, 0, 128, 2, 130]
        for negative in (False, True):
            zeros = torch.full((2, B + 1), -0.0 if negative else 0.0, dtype=torch.bfloat16)
            q, s = quant(zeros)
            assert bool((s == 127).all()) and bool((q == (128 if negative else 0)).all())
            _same_values(dequant(q, s), zeros)

        tiny = torch.full((1, B), 2.0**-133, dtype=torch.bfloat16)
        q, s = quant(tiny)
        assert s.item() == 0 and bool((q == 0x08).all())
        _same_values(dequant(q, s), tiny)
        largest = torch.full((1, B), torch.finfo(torch.bfloat16).max, dtype=torch.bfloat16)
        q, s = quant(largest)
        assert s.item() == 246 and bool((q == 0x7E).all())
        assert bool((dequant(q, s).view(torch.int16) == 0x7F60).all())

        mixed = torch.zeros((1, B), dtype=torch.bfloat16)
        mixed[0, :6] = torch.tensor([1.0, -2.0, float("nan"), float("inf"),
                                      -float("inf"), -0.0], dtype=torch.bfloat16)
        q, s = quant(mixed)
        assert s.item() == 120 and q[0, :6].tolist() == [0x70, 0xF8, 0x7F, 0x7E, 0xFE, 0x80]
        q, s = quant(mixed, saturate=False)
        assert s.item() == 120 and q[0, :6].tolist() == [0x70, 0xF8, 0x7F, 0x7F, 0x7F, 0x80]
        for value, sat_byte in ((float("inf"), 0x7E), (-float("inf"), 0xFE),
                                (float("nan"), 0x7F)):
            mixed.fill_(value)
            q, s = quant(mixed)
            assert s.item() == 127 and bool((q == sat_byte).all())
            q, s = quant(mixed, saturate=False)
            assert s.item() == 127 and bool((q == 0x7F).all())

    # Numerical migration is intentional, not a layout-only transformation:
    # legacy selects scale=2 for 480, whereas OCP selects scale=1 then SAT.
    src = torch.full((1, 128), 480.0, dtype=torch.bfloat16)
    old_q, old_s = legacy_g128_quant(src)
    new_q, new_s = g128_quant(src)
    assert old_s.item() == 128 and bool((old_q == 0x77).all())
    assert new_s.item() == 127 and bool((new_q == 0x7E).all())
    _same_values(legacy_g128_dequant(old_q, old_s), src)
    assert bool((g128_dequant(new_q, new_s) == 448.0).all())
    tiny = torch.full((1, 128), 2.0**-133, dtype=torch.bfloat16)
    old_q, old_s = legacy_g128_quant(tiny)
    new_q, new_s = g128_quant(tiny)
    assert old_s.item() == 94 and not bool(old_q.any())
    assert new_s.item() == 0 and bool((new_q == 0x08).all())
    _same_values(g128_dequant(new_q, new_s), tiny)

    # Every scale x every byte, using native element decode and FP64 ldexp as
    # an independent reference before final BF16 rounding. This includes scale
    # 0/254/255, both zero signs, element NaNs, underflow and output overflow.
    bits = all_bits.repeat(256, 1)
    expected = torch.stack([
        all_bits.view(FP8).double() * math.ldexp(1.0, code - 127)
        if code != 255 else torch.full((256,), float("nan"), dtype=torch.float64)
        for code in range(256)
    ]).to(torch.bfloat16)
    for B, dequant in ((128, g128_dequant), (32, mx_dequant)):
        codes = torch.arange(256, dtype=torch.uint8)[:, None].expand(256, 256 // B).clone()
        _same_values(dequant(bits, codes), expected)
    legacy_codes = torch.arange(256, dtype=torch.uint8)[:, None].expand(256, 2).clone()
    legacy = legacy_g128_dequant(bits, legacy_codes)
    assert torch.isnan(legacy[255, 0]) and torch.isposinf(legacy[255, 1])
    assert torch.isneginf(legacy[255, 129])
    assert bool(torch.isnan(g128_dequant(bits, legacy_codes)[255]).all())

    # Layout boundary cases: detect collisions independently, verify zero
    # padding, and check pack/unpack for both group sizes and partial groups.
    for group_size in (32, 128):
        for T in (0, 1, 17, 127, 128, 129, 257):
            for D in (0, 1, 31, 32, 64, 128, 129, 512, 576, 1025):
                G, Q, count = _dimensions(T, D, group_size)
                src_codes = ((torch.arange(T * G).reshape(T, G) % 255) + 1).to(torch.uint8)
                tiles = pack_scales(src_codes)
                assert tuple(tiles.shape) == (count, 512)
                assert torch.equal(unpack_scales(tiles, T, D, group_size), src_codes)
                addresses = [((t // 128) * Q + g // 4) * 512 + (t % 128) * 4 + g % 4
                             for t in range(T) for g in range(G)]
                assert len(set(addresses)) == T * G
                used = torch.zeros(count * 512, dtype=torch.bool)
                used[addresses] = True
                assert not bool(tiles.flatten()[~used].any())
    # Fixed tile coordinates (128-token boundary and fifth scale column).
    codes = torch.arange(129 * 5, dtype=torch.int64).reshape(129, 5).remainder(256).to(torch.uint8)
    tiles = pack_scales(codes)
    assert tiles[0, 511] == codes[127, 3]
    assert tiles[1, 0] == codes[0, 4] and tiles[2, 0] == codes[128, 0]
    assert tiles[3, 0] == codes[128, 4]
    assert physical_byte_offset(0, 1) == 4 and physical_byte_offset(0, 128) == 1
    assert sorted(physical_byte_offset(1, k) for k in range(512)) == list(range(512, 1024))

    for T, D in ((0, 0), (0, 576), (2, 0)):
        src = torch.empty((T, D), dtype=torch.bfloat16)
        for quant, dequant in ((legacy_g128_quant, legacy_g128_dequant),
                              (g128_quant, g128_dequant), (mx_quant, mx_dequant)):
            q, s = quant(src)
            assert tuple(q.shape) == (T, D)
            _same_values(dequant(q, s), src)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run CPU golden checks only")
    args = parser.parse_args()
    if not args.self_test:
        parser.print_help()
        return
    _self_test()
    print("PASS: CPU FP8 golden/layout self-tests; no DLC kernel or hardware was tested.")


if __name__ == "__main__":
    main()
