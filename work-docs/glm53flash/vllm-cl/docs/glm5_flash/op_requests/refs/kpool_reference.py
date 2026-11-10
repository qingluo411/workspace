"""KPool 参考实现：批量池读出（`dlc_kpool_pooled_read`，需求单 08 的读出侧）。

布局与语义
    pool_cache  [num_blocks, pools_per_block, D]  bf16   块内第 p 个池
    block_table [num_reqs, blocks]                int32  逻辑块 -> 物理块
    pool_ids    [Q, K]                            int32  每个 query 的 K 个逻辑池 id
    req_ids     [Q]                               int32  每个 query 属于哪个请求
    --> out     [Q, K, D]                         与 pool_cache 同 dtype

    slot(q, k) = block_table[req_ids[q], pool_ids[q,k] // pools_per_block] * pools_per_block
                 + pool_ids[q,k] % pools_per_block
    out[q, k]  = pool_cache.flat[slot(q, k)]        # pool_ids < 0 → 该位置输出全 0

与 native golden（`vllm_cl/patch/models/kpool_indexer.py:359` `read_pools`）的差异（**请在 issue 里
写死，避免算子组按 native 实现**）：
  1. native 是**单请求、1D**（`read_pools(request_id, pool_ids)` -> `[K, D]`），本参考是**批量 [Q,K]**；
  2. native 对**任何负 id 直接 raise IndexError**，本参考按「无效槽 → 输出 0」处理 —— 与 08 号需求单
     的「`pool_ids == -1` 返回 0」以及其它 DLC 算子的 padding 约定一致；插件在调 native 前必须自己过滤。

用法（纯 CPU，不需要 DLC 设备）：
    python kpool_reference.py --selftest
"""
from __future__ import annotations

import argparse

import torch


def kpool_pooled_read_ref(pool_ids, block_table, pool_cache, req_ids, *, fill=0.0):
    """批量池读出参考实现；语义见模块 docstring。

    pool_ids [Q,K] int32、req_ids [Q] int32、block_table [num_reqs, blocks] int32、
    pool_cache [num_blocks, pools_per_block, D]。
    返回 [Q, K, D]：`pool_ids < 0` 的位置为 `fill`（默认 0）。
    """
    ids = pool_ids.long()
    req = req_ids.long()
    if ids.dim() != 2:
        raise ValueError(f"pool_ids 必须是 [Q,K]，实际 {tuple(pool_ids.shape)}")
    if req.dim() != 1 or req.numel() != ids.shape[0]:
        raise ValueError(f"req_ids 必须是 [Q]，实际 {tuple(req_ids.shape)} / Q={ids.shape[0]}")
    if block_table.dim() != 2:
        raise ValueError(f"block_table 必须是 [num_reqs, blocks]，实际 {tuple(block_table.shape)}")
    num_reqs, blocks = block_table.shape
    ppb = int(pool_cache.shape[1])
    q, k = ids.shape
    if q and (int(req.min()) < 0 or int(req.max()) >= num_reqs):
        raise IndexError(f"req_ids 越界：合法范围 [0,{num_reqs})")
    out = torch.zeros((q, k, pool_cache.shape[-1]), dtype=pool_cache.dtype,
                      device=pool_cache.device)
    if q == 0 or k == 0:
        return out
    if blocks == 0:
        raise IndexError("block_table 没有列，无法寻址任何池")

    valid = ids >= 0
    if bool(valid.any()) and int(ids[valid].max()) >= blocks * ppb:
        raise IndexError(f"pool id 越界：合法范围 [0,{blocks * ppb})")

    table = block_table[req]                                   # [Q, blocks]
    safe = ids.clamp(min=0)
    physical = torch.gather(table, 1, safe // ppb)             # [Q, K]
    if bool(valid.any()):
        bad = valid & ((physical < 0) | (physical >= pool_cache.shape[0]))
        if bool(bad.any()):
            raise IndexError("block_table 指向的物理块越界（含 -1 无效块）")

    physical = torch.where(valid, physical, torch.zeros_like(physical))
    out = pool_cache[physical, safe % ppb].to(pool_cache.dtype)   # [Q,K,D]
    if bool(valid.any()) and not bool(valid.all()):
        out = out.masked_fill(~valid[..., None], fill)
    return out


def _native_read_pools_inline(pool_cache, block_table, ppb, request_id, pool_ids):
    """`NativeKPoolPagedCache.read_pools`（`kpool_indexer.py:359`）的等价内联版，用于对拍。"""
    ids = torch.as_tensor(pool_ids, dtype=torch.long, device=pool_cache.device)
    if ids.numel() and (ids.min() < 0 or ids.max() >= block_table.shape[1] * ppb):
        raise IndexError("pool id outside block table")
    blocks = torch.div(ids, ppb, rounding_mode="floor")
    offsets = ids.remainder(ppb)
    physical = block_table[request_id, blocks].long()
    if physical.numel() and ((physical < 0).any() or (physical >= pool_cache.shape[0]).any()):
        raise IndexError("physical block outside pool cache")
    return pool_cache[physical, offsets]


def _make_case(num_reqs=3, blocks=4, ppb=8, dim=4, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    pool_cache = torch.randn(num_reqs * 2, ppb, dim, generator=g).to(dtype)
    table = (torch.arange(num_reqs * blocks, dtype=torch.int32) % (num_reqs * 2)).reshape(
        num_reqs, blocks)
    return pool_cache, table, ppb


def selftest(verbose=True):
    """批量版 vs 单请求 golden / 负 id / 越界 / dtype / 空输入。"""
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'OK ' if cond else 'FAIL'}] {name}{('：' + extra) if extra else ''}", flush=True)

    print("=== kpool_pooled_read 参考实现自检 ===", flush=True)
    pool_cache, table, ppb = _make_case()
    q_ids = torch.tensor([[0, 1, 7], [8, 0, 15], [3, 3, 3], [16, 17, 23]], dtype=torch.int32)
    req_ids = torch.tensor([0, 1, 2, 0], dtype=torch.int32)
    got = kpool_pooled_read_ref(q_ids, table, pool_cache, req_ids)
    want = torch.stack([
        _native_read_pools_inline(pool_cache, table, ppb, int(req_ids[q]), q_ids[q])
        for q in range(q_ids.shape[0])
    ])
    check("批量 == 单请求 golden（含跨请求混排、跨块）", torch.equal(got, want),
          f"max_err={(got - want).abs().max().item():.1e}")
    check("形状 [Q,K,D]", tuple(got.shape) == (4, 3, pool_cache.shape[-1]), str(tuple(got.shape)))

    bad = q_ids.clone()
    bad[0, 1] = -1
    bad[2, :] = -1
    got_bad = kpool_pooled_read_ref(bad, table, pool_cache, req_ids)
    check("负 id → 全 0，其余位置不受影响",
          bool((got_bad[0, 1] == 0).all()) and bool((got_bad[2] == 0).all())
          and torch.equal(got_bad[0, 0], want[0, 0]) and torch.equal(got_bad[3], want[3]))
    got_fill = kpool_pooled_read_ref(bad, table, pool_cache, req_ids, fill=-3.0)
    check("fill 参数生效", bool((got_fill[2] == -3.0).all()))

    for name, fn in (
        ("pool id 越界报错", lambda: kpool_pooled_read_ref(
            torch.tensor([[table.shape[1] * ppb]], dtype=torch.int32), table, pool_cache,
            torch.tensor([0], dtype=torch.int32))),
        ("req_ids 越界报错", lambda: kpool_pooled_read_ref(
            torch.tensor([[0]], dtype=torch.int32), table, pool_cache,
            torch.tensor([table.shape[0]], dtype=torch.int32))),
        ("物理块 -1 报错", lambda: kpool_pooled_read_ref(
            torch.tensor([[0]], dtype=torch.int32), torch.full_like(table, -1), pool_cache,
            torch.tensor([0], dtype=torch.int32))),
    ):
        try:
            fn()
            check(name, False, "没有抛异常")
        except IndexError:
            check(name, True)

    empty = kpool_pooled_read_ref(torch.zeros(0, 3, dtype=torch.int32), table, pool_cache,
                                  torch.zeros(0, dtype=torch.int32))
    check("Q == 0", tuple(empty.shape) == (0, 3, pool_cache.shape[-1]))

    for dtype in (torch.float32, torch.bfloat16):
        pc, tb, p = _make_case(dtype=dtype, seed=1)
        o = kpool_pooled_read_ref(q_ids, tb, pc, req_ids)
        check(f"dtype {dtype} 保持", o.dtype == dtype, str(o.dtype))

    print(f"=== {'通过' if ok else '失败'} ===", flush=True)
    return ok


def compare_with_native_class(verbose=True):
    """与仓库里真正的 `NativeKPoolPagedCache.read_pools` 对拍（会 import vllm_cl，较慢）。"""
    try:
        from vllm_cl.patch.models.kpool_indexer import NativeKPoolPagedCache
    except Exception as exc:                                    # noqa: BLE001
        print(f"  跳过：无法 import NativeKPoolPagedCache（{type(exc).__name__}）", flush=True)
        return None
    pool_cache, table, ppb = _make_case(dtype=torch.bfloat16, seed=2)
    tail, lengths = (torch.zeros(table.shape[0], 2, 4, pool_cache.shape[-1], dtype=torch.bfloat16),
                     torch.zeros(table.shape[0], dtype=torch.long))
    state = NativeKPoolPagedCache(pool_cache, tail, lengths, table, pool_size=4, block_size=32)
    q_ids = torch.tensor([[0, 1, 7], [8, 0, 15], [3, 3, 3], [16, 17, 23]], dtype=torch.int32)
    req_ids = torch.tensor([0, 1, 2, 0], dtype=torch.int32)
    got = kpool_pooled_read_ref(q_ids, table, pool_cache, req_ids)
    want = torch.stack([state.read_pools(int(req_ids[q]), q_ids[q]) for q in range(q_ids.shape[0])])
    same = torch.equal(got, want)
    if verbose:
        print(f"  [{'OK ' if same else 'FAIL'}] 与 NativeKPoolPagedCache.read_pools 逐元素一致", flush=True)
        try:
            state.read_pools(0, torch.tensor([-1]))
            print("  ! 注：native 居然接受了 -1（与文档预期不符）", flush=True)
        except IndexError:
            print("  [OK ] native 对 -1 报 IndexError（本参考按「跳过→0」处理，算子按后者实现）", flush=True)
    return same


def main(argv=None):
    p = argparse.ArgumentParser(description="KPool pooled read 参考实现（批量）")
    p.add_argument("--selftest", action="store_true", help="跑自检（默认动作，纯 CPU）")
    p.add_argument("--with-native", action="store_true",
                   help="额外与仓库 NativeKPoolPagedCache.read_pools 对拍（import vllm_cl，较慢）")
    a = p.parse_args(argv)
    ok = selftest()
    if a.with_native:
        print("=== 与 native class 对拍 ===", flush=True)
        r = compare_with_native_class()
        ok = ok and (r is not False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
