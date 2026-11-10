"""#08 读出侧 `dlc_kpool_pooled_read` 的语义核对与真机对拍脚本。

对应需求单 [`../08_dlc_kpool_append_update.md`](../08_dlc_kpool_append_update.md) 的读出侧；
参考实现（批量版）在 [`../refs/kpool_reference.py`](../refs/kpool_reference.py)。

用法：
  A) 纯 CPU 语义核对（默认，不占 DLC 设备）：
       python probe_kpool_pooled_read.py
     包含：批量版 vs native `NativeKPoolPagedCache.read_pools`（单请求）逐元素对拍、
     负 id → 0、越界报错、dtype/空输入，以及与 native「负数直接报错」的差异确认。
  B) 真机计时/对拍（需要干净卡；读出是纯 gather，不会卡死设备）：
       python probe_kpool_pooled_read.py dlc:6 [--batch 64] [--pools 512] [--layers 11]
     会打印 native 逐请求循环 vs 批量参考的每步耗时；若 `torch.dlc_kpool_pooled_read`
     已交付，则一并做数值对拍。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "refs"))
from kpool_reference import (  # noqa: E402
    compare_with_native_class,
    kpool_pooled_read_ref,
    selftest,
)

DIM = 128
PPB = 64          # block_size=256 / pool_size=4


def cpu_part():
    ok = selftest()
    print("=== 与仓库 native class 对拍 ===", flush=True)
    r = compare_with_native_class()
    return ok and (r is not False)


def device_part(dev, batch, pools, layers, num_blocks=8192, block_cols=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    pool_cache = torch.randn(num_blocks, PPB, DIM, generator=g).to(torch.bfloat16).to(dev)
    block_table = (torch.arange(batch * block_cols, dtype=torch.int32).reshape(batch, block_cols)
                   % num_blocks).to(dev)
    q = batch                      # decode：每请求 1 个 query token
    pool_ids = (torch.randint(0, block_cols * PPB, (q, pools), generator=g)).to(torch.int32).to(dev)
    pool_ids[0, :4] = -1           # 无效槽
    req_ids = torch.arange(q, dtype=torch.int32, device=dev)

    def native_loop():
        """插件现状：逐请求调用 read_pools（每请求一次高级索引 + 索引算术）。"""
        outs = []
        for r in range(q):
            ids = pool_ids[r].long()
            ids = ids[ids >= 0]                     # native 对负数直接 raise，插件需先过滤
            blk = torch.div(ids, PPB, rounding_mode="floor")
            off = ids.remainder(PPB)
            outs.append(pool_cache[block_table[r, blk].long(), off])
        return outs

    ref = kpool_pooled_read_ref(pool_ids.cpu(), block_table.cpu(), pool_cache.cpu(), req_ids.cpu())

    def timeit(fn, iters=20, warmup=5):
        for _ in range(warmup):
            fn()
        torch.dlc.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.dlc.synchronize()
        return (time.time() - t0) / iters * 1e3

    t_native = timeit(native_loop)
    t_batch = timeit(lambda: kpool_pooled_read_ref(pool_ids, block_table, pool_cache, req_ids))
    print(f"=== {dev} 读出计时（batch={batch}, K={pools} 池, D={DIM}, 单层）===", flush=True)
    print(f"    native 逐请求循环（插件现状）: {t_native:.3f} ms/层  → {t_native * layers:.3f} ms/step（{layers} 层）",
          flush=True)
    print(f"    批量一次高级索引（参考实现）：  {t_batch:.3f} ms/层  → {t_batch * layers:.3f} ms/step",
          flush=True)

    if hasattr(torch, "dlc_kpool_pooled_read"):
        out = torch.dlc_kpool_pooled_read(pool_ids, block_table, pool_cache, req_ids)
        torch.dlc.synchronize()
        got = out.cpu()
        err = (got.float() - ref.float()).abs().max().item()
        zero_ok = bool((got[0, :4] == 0).all())
        print(f"    融合算子对拍：out_err={err:.3e}  负 id 位置为 0={zero_ok}", flush=True)
        print(f"    融合算子耗时：{timeit(lambda: torch.dlc_kpool_pooled_read(pool_ids, block_table, pool_cache, req_ids)):.3f} ms/层",
              flush=True)
    else:
        print("    `torch.dlc_kpool_pooled_read` 尚未交付 → 只报告现状；交付后用本脚本对拍。", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="dlc_kpool_pooled_read 语义核对/计时")
    p.add_argument("device", nargs="?", default=None, help="如 dlc:6；不给则只跑 CPU 语义核对")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--pools", type=int, default=512, help="每个 query 读出的池数（topk_tokens/4）")
    p.add_argument("--layers", type=int, default=11)
    a = p.parse_args(argv)
    ok = cpu_part()
    if a.device:
        device_part(a.device, a.batch, a.pools, a.layers)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
