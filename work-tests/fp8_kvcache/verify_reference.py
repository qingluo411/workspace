"""Independent CPU checks for the FP8 KV design; does not launch DLC work."""
import json
import math
import struct
from pathlib import Path


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def decode(code):
    sign, exponent, mantissa = code >> 7, (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        return math.nan
    value = mantissa * 2.0**-9 if exponent == 0 else (1 + mantissa / 8) * 2.0**(exponent - 7)
    return math.copysign(value, -1 if sign else 1)


POSITIVE = [decode(i) for i in range(127)]


def encode(y):
    if math.isnan(y):
        return 0x7f
    sign = 128 if math.copysign(1.0, y) < 0 else 0
    value = min(abs(y), 448.0)
    # Enumeration is deliberately independent from the proposed bit algorithm.
    code = min(range(127), key=lambda i: (abs(POSITIVE[i] - value), i & 1))
    return sign | code


def bf16_bits(value):
    if math.isnan(value):
        return 0x7fc0
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    return ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffffffff) >> 16


def run():
    finite = [i for i in range(256) if i not in (127, 255)]
    assert all(encode(decode(code)) == code for code in finite)
    vectors = [(0., 0), (-0., 128), (1.0625, 0x38), (1.1875, 0x3a),
               (2**-10, 0), (-2**-10, 128), (2**-9, 1), (448., 126),
               (512., 126), (math.inf, 126), (-math.inf, 254), (math.nan, 127)]
    assert all(encode(value) == code for value, code in vectors)
    assert bf16_bits(1.00390625) == 0x3f80
    assert bf16_bits(1.01171875) == 0x3f82
    inputs, scales = [0.125, -0.375, 0.0625, 0.5], [0.125, 0.25, 0.125, 0.25]
    codes = [encode(f32(x / s)) for x, s in zip(inputs, scales)]
    word = sum(c << (8 * i) for i, c in enumerate(codes))
    assert word == 0x4030bc38
    restored = [bf16_bits(f32(decode(c) * s)) for c, s in zip(codes, scales)]
    assert restored == [0x3e00, 0xbec0, 0x3d80, 0x3f00]
    assert restored[0] | (restored[1] << 16) == 0xbec03e00
    assert restored[2] | (restored[3] << 16) == 0x3f003d80
    initial = 0xaabbccdd
    even = (initial & 0xffff0000) | 0xbc38
    assert even == 0xaabbbc38
    final = (even & 0x0000ffff) | (0x4030 << 16)
    assert final == word
    odd_first = (initial & 0x0000ffff) | (0x4030 << 16)
    assert ((odd_first & 0xffff0000) | 0xbc38) == final
    # A normal-range counterexample: a correctly rounded reciprocal still differs.
    x, scale = 1.0078125, f32(1.4021739959716797)
    quotient = f32(x / scale)
    product = f32(x * f32(1.0 / scale))
    assert encode(quotient) == 0x33 and encode(product) == 0x34
    assert 1024 * 8 * 65536 == 512 * 1024**2
    # Large offsets: preserve exact integer arithmetic beyond signed int32.
    tile_id = ((65535 * 8 + 7) * 128 + 127)
    offset = tile_id * 512 + 4 * 127 + 3
    assert offset == 65536 * 8 * 65536 - 1 and offset > 2**31
    result = {
        'status': 'passed', 'execution': 'CPU only, independent scalar enumeration',
        'finite_codes_roundtrip': 254, 'special_and_midpoint_vectors': len(vectors),
        'bf16_midpoints': 2, 'physical_word': hex(word),
        'restored_bf16_bits': [hex(v) for v in restored],
        'rmw_order_invariance': True, 'capacity_example_bytes': 1024 * 8 * 65536,
        'large_byte_offset': offset,
        'division_counterexample': {'bf16_input': x, 'fp32_scale': scale,
            'rn32_division': quotient, 'rn32_reciprocal_multiply': product,
            'division_fp8_code': hex(encode(quotient)),
            'reciprocal_fp8_code': hex(encode(product))},
        'limits': ['No DLC kernel execution', 'No hardware performance or model quality claim',
                   'No proof of DLC intrinsic RNE, FTZ, DMA, or stream behavior'],
    }
    return result


if __name__ == '__main__':
    output = run()
    target = Path(__file__).with_name('reference_checks.json')
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output, ensure_ascii=False, indent=2))
