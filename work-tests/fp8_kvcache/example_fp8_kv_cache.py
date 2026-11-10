#!/usr/bin/env python3
"""Run a real-device example of the separated FP8 KV-cache layout."""

from __future__ import annotations

import argparse
import gc

import torch
from torch.dlc.fp8_kv_cache import FP8KVCacheLayout, create_fp8_kv_cache


def measure_allocation(build) -> int:
    gc.collect()
    torch.dlc.empty_cache()
    before = torch.dlc.memory_allocated()
    value = build()
    allocated = torch.dlc.memory_allocated() - before
    del value
    gc.collect()
    torch.dlc.empty_cache()
    return allocated


def print_capacity(layout: FP8KVCacheLayout, num_layers: int) -> None:
    slots = layout.num_slots
    bf16_bytes = slots * layout.bf16_bytes_per_token
    fp8_bytes = slots * layout.bytes_per_token
    print(f"blocks={layout.num_blocks} block_size={layout.block_size} slots={slots}")
    print(f"values shape={layout.value_shape} scales shape={layout.scale_shape}")
    print(
        f"one layer: bf16={bf16_bytes / 2**20:.3f} MiB, "
        f"fp8+fp32_scale={fp8_bytes / 2**20:.3f} MiB, "
        f"saving={layout.storage_saving:.4%}"
    )
    print(
        f"{num_layers} layers: bf16={num_layers * bf16_bytes / 2**20:.3f} MiB, "
        f"fp8+fp32_scale={num_layers * fp8_bytes / 2**20:.3f} MiB, "
        f"saved={num_layers * (bf16_bytes - fp8_bytes) / 2**20:.3f} MiB"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="dlc:0")
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--head-size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=11)
    args = parser.parse_args()

    device = torch.device(args.device)
    layout = FP8KVCacheLayout(args.blocks, args.block_size, args.head_size)
    print_capacity(layout, args.layers)

    blocks = layout.num_blocks
    block_size = layout.block_size
    head_size = layout.head_size
    slots_per_block = block_size
    groups = layout.num_groups

    bf16_bytes = measure_allocation(
        lambda: torch.empty(
            (blocks, block_size, head_size), dtype=torch.bfloat16, device=device
        )
    )
    fp8_bytes = measure_allocation(
        lambda: create_fp8_kv_cache(blocks, block_size, head_size, device=device)
    )
    bad_scale_bytes = measure_allocation(
        lambda: (
            torch.empty(
                (blocks, block_size, head_size),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            torch.empty((layout.num_slots, groups), dtype=torch.float32, device=device),
        )
    )
    inline_528_bytes = measure_allocation(
        lambda: torch.empty(
            (blocks, block_size, head_size + groups * 4),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
    )

    print(f"actual BF16 allocation:                 {bf16_bytes} bytes")
    print(f"actual separated FP8 allocation:        {fp8_bytes} bytes")
    print(f"actual FP8 + [slots,4] scales:          {bad_scale_bytes} bytes")
    print(f"actual FP8 inline [B,256,528]:          {inline_528_bytes} bytes")
    if not fp8_bytes < bf16_bytes:
        raise AssertionError("separated FP8 layout did not save device memory")
    if bad_scale_bytes < bf16_bytes:
        raise AssertionError("unexpected [slots,4] scale measurement")
    if inline_528_bytes < bf16_bytes:
        raise AssertionError("unexpected inline 528-byte row measurement")

    cache = create_fp8_kv_cache(blocks, block_size, head_size, device=device)
    hidden = (torch.randn(8, head_size) * 3).to(torch.bfloat16).to(device)
    slot_mapping = torch.tensor(
        [
            0,
            1,
            slots_per_block - 1,
            slots_per_block,
            slots_per_block + 1,
            2 * slots_per_block - 1,
            2 * slots_per_block,
            -1,
        ],
        dtype=torch.int64,
        device=device,
    )
    cache.quantize_and_scatter(hidden, slot_mapping)
    values, scales = cache.gather_quantized(slot_mapping)
    output = cache.gather_dequantize(slot_mapping)
    torch.dlc.synchronize()

    print(f"gathered values={tuple(values.shape)} {values.dtype}")
    print(f"gathered scales={tuple(scales.shape)} {scales.dtype}")
    print(f"dequantized output={tuple(output.shape)} {output.dtype}")
    print(f"negative slot row is zero: {not bool(output[-1].cpu().any())}")

    cache.copy_blocks([0], [1])
    cache.zero_blocks([1])
    print("block copy and block zero completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
