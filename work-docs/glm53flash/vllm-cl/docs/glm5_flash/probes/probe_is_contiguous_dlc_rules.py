#!/usr/bin/env python3
"""实测 DLC launcher「自动连续化」判定到底看什么。

代码依据（pytorch 仓）：
  * 触发点 `aten/src/ATen/dlc/DLCKernelLaunch.h:1046 _set_dlc_tensor_maybe_copy`：
    只有 format == dlc_dlc_contiguous 时才检查/拷贝；不满足就 `to_contiguous_dlc()` 实拷一份。
    `KernelDesc::input(t)`/`output(t)` 是 contiguous；`input_format(t)` 默认 **dlc_strided**（不检查）。
  * 判定 `aten/src/ATen/dlc/DLCTensorImpl.h:56 at::detail::is_contiguous_dlc`。
  * alignment `c10/dlc/kernel_arg_types_copy.h:64 dlc_storage_alignment`：int8→512、int16→256、
    int32/int64→128（= 512B / itemsize，单位是**元素**，不是字节）。

本脚本用 `torch.add`（走 `k.input()`，contiguous 语义）对比：
  A. 独占 buffer（numel == storage 宽度、offset 0）
  B. 打包 buffer 里的段视图（storage 宽度是整个打包 buffer）
     B1 段长是 alignment 的整数倍、段起点按 alignment 对齐
     B2 段长不是 alignment 的整数倍（模拟 query_start_loc = max_num_reqs+1 = 257）
若判定为「不连续」，B 相对 A 会多出一次设备拷贝（~25 µs 量级）。

Usage: python -u probe_is_contiguous_dlc_rules.py [dlc:6]
"""
from __future__ import annotations

import statistics
import sys
import time

import torch


def bench(fn, iters=300, warmup=30):
    for _ in range(warmup):
        fn()
    torch.dlc.synchronize()
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.dlc.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6 / iters)
    return statistics.median(samples)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    dev_name = argv[0] if argv else "dlc:6"
    torch.dlc.set_device(int(dev_name.split(":")[-1]))
    print(f"device={dev_name}", flush=True)

    for dtype, align in ((torch.int32, 128), (torch.int64, 128), (torch.bool, 512)):
        print(f"\n== dtype={str(dtype).replace('torch.','')} alignment={align} 元素 ==", flush=True)
        print(f"  {'case':34} {'numel':>6} {'ptr%512':>8} {'us':>9} {'相对独占':>9}", flush=True)
        for numel in (align, align * 2, align * 2 + 1):
            base_len = align * 4
            # A: 独占 buffer
            solo = torch.ones(numel, dtype=dtype, device=dev_name)
            solo2 = torch.ones(numel, dtype=dtype, device=dev_name)
            # B: 打包 buffer 内的段视图（起点按 alignment 元素对齐）
            pack = torch.zeros(base_len + numel + align, dtype=dtype, device=dev_name)
            seg = pack.narrow(0, align, numel)
            other = torch.ones(numel, dtype=dtype, device=dev_name)
            t_solo = bench(lambda: torch.add(solo, solo2))
            t_seg = bench(lambda: torch.add(seg, other))
            tag = "长度%alignment==0" if numel % align == 0 else "长度%alignment!=0"
            print(f"  A 独占(n={numel},{tag})".ljust(36)
                  + f" {numel:6} {solo.data_ptr() % 512:8} {t_solo:9.2f}", flush=True)
            print(f"  B 打包段(offset={align})".ljust(36)
                  + f" {numel:6} {seg.data_ptr() % 512:8} {t_seg:9.2f} {t_seg / t_solo:9.2f}x", flush=True)

    # 真实 metadata 形状：note 里 6 个字段逐个过一遍
    print("\n== 真实 metadata 字段（独占 vs 打包段，int32/64 + bool）==", flush=True)
    fields = [
        ("query_start_loc", torch.int32, 257),
        ("num_scheduled_tokens", torch.int32, 256),
        ("discard_request_mask", torch.bool, 256),
        ("input_ids", torch.int32, 8192),
        ("req_indices", torch.int64, 8192),
        ("query_pos", torch.int64, 8192),
    ]
    print(f"  {'field':22} {'dtype':6} {'numel':>6} {'独占us':>8} {'打包段us':>10} {'差':>8}", flush=True)
    for name, dtype, numel in fields:
        solo = torch.ones(numel, dtype=dtype, device=dev_name)
        solo2 = torch.ones(numel, dtype=dtype, device=dev_name)
        pack = torch.zeros(numel + 1024, dtype=dtype, device=dev_name)
        seg = pack.narrow(0, 1024, numel)
        t_solo = bench(lambda a=solo, b=solo2: torch.add(a, b), iters=200)
        t_seg = bench(lambda a=seg, b=solo2: torch.add(a, b), iters=200)
        print(f"  {name:22} {str(dtype).replace('torch.',''):6} {numel:6} {t_solo:8.2f} {t_seg:10.2f} "
              f"{t_seg - t_solo:+8.2f}", flush=True)

    # 读侧是「按实际 batch 切片」（`buf.gpu[:num_reqs+1]`）—— 切片后 sizes.back() 不再是
    # alignment 整数倍，也未必等于 dlc_dim0_，按判定链应当同样触发拷贝。验证之。
    print("\n== 切片视图（模型真实读法 buf[:n]）是否触发自动连续化 ==", flush=True)
    print(f"  {'case':40} {'独占us':>9} {'切片us':>9} {'差':>8}", flush=True)
    for numel, live in ((257, 4), (257, 129), (8192, 2048), (8192, 4097)):
        buf = torch.ones(numel, dtype=torch.int32, device=dev_name)
        solo = torch.ones(live, dtype=torch.int32, device=dev_name)
        other = torch.ones(live, dtype=torch.int32, device=dev_name)
        full_view = buf[:live]
        t_full = bench(lambda a=solo, b=other: torch.add(a, b), iters=200)
        t_slice = bench(lambda a=full_view, b=other: torch.add(a, b), iters=200)
        print((f"  buf[{numel}] 取 [:{live}]").ljust(42)
              + f" {t_full:9.2f} {t_slice:9.2f} {t_slice - t_full:+8.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
