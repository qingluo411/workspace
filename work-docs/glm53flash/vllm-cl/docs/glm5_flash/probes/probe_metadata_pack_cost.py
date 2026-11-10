#!/usr/bin/env python3
"""C1 打包上传的前置测算：打包成一段大 buffer 后，被切片出来的 metadata 张量
是不是「连续但 storage_offset≠0」，以及「需要单独连续化」时那份拷贝值多少钱。

回答两个问题：
  1. 打包后的段视图：`is_contiguous()` 是否仍为 True？`.contiguous()` 会不会真的拷？
     → 段视图是连续的，`.contiguous()` 是 no-op（返回自身、不产生拷贝）；
       真正可能发生拷贝的地方是消费侧主动 `clone()`/launcher 自动连续化。
  2. 若真的需要物化，代价多少？和它能省掉的 1 次 H2D（≈44 µs）相比值不值？

尺寸取 `dlc_model_runner` 的真实 buffer（`_make_buffer`）：
  input_ids          int32  max_num_tokens
  query_start_loc    int32  max_num_reqs+1
  num_scheduled_tokens int32 max_num_reqs
  req_indices        int64  max_num_tokens
  query_pos          int64  max_num_tokens
  discard_request_mask bool  max_num_reqs

Usage:
  python -u probe_metadata_pack_cost.py [dlc:5] [--max-tokens 8192] [--max-reqs 256]
  DLC_SYN_LOG_DIR=/tmp/packlog DLC_SYN_VERBOSE=1 DLC_SYN_DEBUG=1 DLC_SYN_PROF_CYCLE=1 \
      python -u probe_metadata_pack_cost.py dlc:5 --kernel-log-check
"""
from __future__ import annotations

import statistics
import sys
import time

import torch

ALIGN = 512


def bench(fn, iters=200, warmup=20):
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


def align_up(offset, alignment=ALIGN):
    return (offset + alignment - 1) // alignment * alignment


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    dev_name = argv[0] if argv else "dlc:5"
    max_tokens = 8192
    max_reqs = 256
    for i, a in enumerate(sys.argv):
        if a == "--max-tokens":
            max_tokens = int(sys.argv[i + 1])
        if a == "--max-reqs":
            max_reqs = int(sys.argv[i + 1])
    torch.dlc.set_device(int(dev_name.split(":")[-1]))
    print(f"device={dev_name} max_tokens={max_tokens} max_reqs={max_reqs} align={ALIGN}B", flush=True)

    fields = [
        ("query_start_loc", torch.int32, max_reqs + 1),
        ("num_scheduled_tokens", torch.int32, max_reqs),
        ("discard_request_mask", torch.bool, max_reqs),
        ("input_ids", torch.int32, max_tokens),
        ("req_indices", torch.int64, max_tokens),
        ("query_pos", torch.int64, max_tokens),
    ]

    # 每个 dtype 一条打包 buffer（不同 dtype 不能塞进同一条：DLC storage 的
    # dlc_layout_ 按 dtype 固定，跨 dtype 视图会让描述符不匹配）。
    def esize(dtype):
        return torch.empty(0, dtype=dtype).element_size()

    packed = {}
    for dtype in (torch.int32, torch.int64, torch.bool):
        cursor = 0
        for _name, fdt, numel in fields:
            if fdt is dtype:
                cursor = align_up(cursor) + numel * esize(fdt)
        packed[dtype] = torch.zeros(align_up(cursor) // esize(dtype), dtype=dtype, device=dev_name)

    # 字节游标（对齐按字节算，narrow 的下标按元素算）。
    byte_cursor = {torch.int32: 0, torch.int64: 0, torch.bool: 0}
    views = {}
    print("\n== 1. 段视图属性（打包 buffer 内，512B 对齐切段）==", flush=True)
    for name, dtype, numel in fields:
        start = align_up(byte_cursor[dtype])
        seg = packed[dtype].narrow(0, start // esize(dtype), numel)
        byte_cursor[dtype] = start + numel * esize(dtype)
        views[name] = seg
        byte_off = seg.storage_offset() * seg.element_size()
        print(f"  {name:22} {str(dtype).replace('torch.',''):6} numel={numel:6} "
              f"contig={seg.is_contiguous()} offset={seg.storage_offset():6} "
              f"byte_off={byte_off:7} off%512={byte_off % ALIGN:4} "
              f"ptr%512={seg.data_ptr() % ALIGN:4} nbytes={seg.numel() * seg.element_size()}", flush=True)

    print("\n== 2. .contiguous() 是否是 no-op（不产生拷贝）==", flush=True)
    for name in ("query_start_loc", "input_ids", "req_indices"):
        seg = views[name]
        same = seg.contiguous().data_ptr() == seg.data_ptr()
        print(f"  {name:22} contiguous() 返回自身={same}", flush=True)

    print("\n== 3. 若消费侧要物化：clone() 的单次代价 ==", flush=True)
    print(f"  {'field':22} {'nbytes':>8} {'clone_us':>9} {'h2d_us':>8} {'clone/h2d':>10}", flush=True)
    for name, dtype, numel in fields:
        seg = views[name]
        clone_us = bench(lambda s=seg: s.clone())
        src = torch.empty(numel, dtype=dtype, pin_memory=True)
        dst = torch.empty(numel, dtype=dtype, device=dev_name)
        h2d_us = bench(lambda d=dst, s=src: d.copy_(s, non_blocking=True))
        print(f"  {name:22} {seg.numel() * seg.element_size():8} {clone_us:9.2f} {h2d_us:8.2f} "
              f"{clone_us / h2d_us:10.2f}", flush=True)

    print("\n== 4. 消费侧实测：offset==0 张量 vs 打包段视图 ==", flush=True)
    # `dlc_kda_state_gather` 的 indices 是我们已知能吃「大 buffer 的连续切片」的算子
    # （模型的 block_table[:, 0] 就是这种），用它对比 launcher 是否会额外物化。
    h, dv, dk = 4, 8, 8
    pages = 64
    state = torch.zeros((pages, h, dv, dk), dtype=torch.float32, device=dev_name)
    mask_src = torch.ones(max_reqs, dtype=torch.bool, pin_memory=True)
    mask = torch.ones(max_reqs, dtype=torch.bool, device=dev_name)
    fresh = torch.arange(max_reqs, dtype=torch.int32).remainder(pages - 1).contiguous().to(dev_name)
    seg_idx = views["query_start_loc"][:max_reqs].clone()  # 先给出与 fresh 同形的段视图
    seg_idx.copy_(fresh)
    t_fresh = bench(lambda: torch.dlc_kda_state_gather(state, fresh, mask))
    t_packed = bench(lambda: torch.dlc_kda_state_gather(state, seg_idx, mask))
    print(f"  gather(indices=offset-0) {t_fresh:7.3f} us | gather(indices=打包段视图) {t_packed:7.3f} us "
          f"| 差 {t_packed - t_fresh:+.3f} us", flush=True)

    # 再对一个通用的 elementwise op 做同样对比：launcher 若因 storage 布局自动连续化，
    # 段视图这一侧会明显更慢。
    x_fresh = torch.ones(max_tokens, dtype=torch.float32, device=dev_name)
    seg_f32 = torch.empty(max_tokens, dtype=torch.float32, device=dev_name)
    base = torch.empty(max_tokens + 1024, dtype=torch.float32, device=dev_name)
    seg_view = base.narrow(0, 1024, max_tokens).fill_(1.0)
    t_a = bench(lambda: torch.add(x_fresh, 1.0))
    t_b = bench(lambda: torch.add(seg_view, 1.0))
    print(f"  add(offset-0 {x_fresh.numel()}) {t_a:7.3f} us | add(段视图 offset=1024) {t_b:7.3f} us "
          f"| 差 {t_b - t_a:+.3f} us", flush=True)
    print(f"  （对照组：clone 一次 f32[{max_tokens}] = "
          f"{bench(lambda: seg_f32.clone()):.2f} us）", flush=True)
    del base, seg_f32

    if "--kernel-log-check" in sys.argv:
        print("\n== 5. kernel log 计数（是否出现额外 copy kernel）==", flush=True)
        for tag, idx in (("offset-0", fresh), ("packed-view", seg_idx)):
            torch.dlc.synchronize()
            for _ in range(50):
                torch.dlc_kda_state_gather(state, idx, mask)
            torch.dlc.synchronize()
            print(f"  {tag} 50 次 gather 已下发；请用 syn log 里 custom_kda_state_gather 前后的"
                  f" launch 名称比对是否多出 copy/clone kernel", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
