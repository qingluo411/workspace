from typing import Tuple

import torch
from torch import Tensor

from pytorch_test.core.base import OpType, TestBase
from pytorch_test.core.decorators import exec_test_decorator as test_exec
from pytorch_test.core.synshape import SynShape


DEFAULT_GROUP_SIZE = 128
E4M3_DTYPE = torch.float8_e4m3fn
E5M2_DTYPE = torch.float8_e5m2

E4M3_CODES = [
    0x00, 0x01, 0x07, 0x08, 0x10, 0x20, 0x38, 0x40,
    0x48, 0x50, 0x58, 0x60, 0x70, 0x7B, 0x7E, 0x80, 0x81,
    0x87, 0x88, 0xB8, 0xF8, 0xFB, 0xFE,
]
E5M2_CODES = [
    0x00, 0x01, 0x03, 0x04, 0x08, 0x10, 0x20, 0x3C,
    0x40, 0x48, 0x50, 0x60, 0x70, 0x78, 0x7B, 0x81,
    0x83, 0x84, 0xBC, 0xF0, 0xF8, 0xFB,
]
SCALE_CODES = [127, 126, 128, 120, 130, 124, 132]


def _to_int8(raw: int) -> int:
    return raw - 256 if raw >= 128 else raw


def _make_fp8_bits(fmt: str, num_tokens: int, hidden_size: int) -> list[list[int]]:
    codes = E5M2_CODES if fmt == "e5m2" else E4M3_CODES
    data = []
    for token in range(num_tokens):
        row = []
        for hidden in range(hidden_size):
            row.append(_to_int8(codes[(hidden + token * 13) % len(codes)]))
        data.append(row)
    return data


def _make_scale_codes(num_tokens: int, num_groups: int) -> list[list[int]]:
    data = []
    for token in range(num_tokens):
        row = []
        for group in range(num_groups):
            row.append(SCALE_CODES[(group + token * 3) % len(SCALE_CODES)])
        data.append(row)
    return data


def _cpu_ref(bits: Tensor, scales: Tensor, fp8_dtype: torch.dtype, group_size: int) -> Tensor:
    num_tokens, hidden_size = bits.shape
    num_groups = (hidden_size + group_size - 1) // group_size
    src = bits.view(fp8_dtype).to(torch.float32)
    scale = torch.exp2(scales.to(torch.float32) - 127.0)
    out = torch.empty((num_tokens, hidden_size), dtype=torch.float32)

    for group in range(num_groups):
        begin = group * group_size
        end = min(begin + group_size, hidden_size)
        out[:, begin:end] = src[:, begin:end] * scale[:, group : group + 1]

    return out.to(torch.bfloat16)


class TestDequantFp8G128Bf16(TestBase):
    tolerance = OpType.COPY

    def static_draw(self):
        return [
            ("e4m3", 1, 1),
            ("e5m2", 1, 1),
            ("e4m3", 1, 127),
            ("e5m2", 1, 128),
            ("e4m3", 2, 129),
            ("e4m3", 17, 64),
            ("e4m3", 129, 512),
            ("e4m3", 257, 576),
            ("e5m2", 17, 255),
            ("e4m3", 129, 1025),
            ("e5m2", 257, 640),
        ]

    def dyn_draw(self, ctx) -> Tuple[str, int, int]:
        fmt = ctx.draw_choice(["e4m3", "e5m2"])
        num_tokens = ctx.draw_int(1, 192)
        hidden_size = ctx.draw_int(1, 2048)
        return fmt, num_tokens, hidden_size

    def predict_variant(self, draw_params) -> str:
        fmt, _, _ = draw_params
        if fmt == "e5m2":
            return "custom_dequant_e5m2_g128_bf16"
        return "custom_dequant_e4m3_g128_bf16"

    def build_input(self, ctx, draw_params):
        fmt, num_tokens, hidden_size = draw_params
        group_size = DEFAULT_GROUP_SIZE
        num_groups = (hidden_size + group_size - 1) // group_size

        bits = ctx.make_tensor_from_data(
            _make_fp8_bits(fmt, num_tokens, hidden_size),
            dtype=torch.int8,
        )
        scales = ctx.make_tensor_from_data(
            _make_scale_codes(num_tokens, num_groups),
            dtype=torch.uint8,
        )
        out = ctx.zeros(SynShape(hidden_size, num_tokens), dtype=torch.bfloat16)
        return bits, scales, out, group_size

    @test_exec(for_variant=["custom_dequant_e4m3_g128_bf16"])
    def exec_e4m3(self, bits: Tensor, scales: Tensor, out: Tensor, group_size: int = 128):
        if bits.device.type == "cpu":
            return _cpu_ref(bits, scales, E4M3_DTYPE, group_size)
        torch.dlc_dequant_e4m3(bits.view(E4M3_DTYPE), scales, out, group_size)
        return out

    @test_exec(for_variant=["custom_dequant_e5m2_g128_bf16"])
    def exec_e5m2(self, bits: Tensor, scales: Tensor, out: Tensor, group_size: int = 128):
        if bits.device.type == "cpu":
            return _cpu_ref(bits, scales, E5M2_DTYPE, group_size)
        torch.dlc_dequant_e5m2(bits.view(E5M2_DTYPE), scales, out, group_size)
        return out
