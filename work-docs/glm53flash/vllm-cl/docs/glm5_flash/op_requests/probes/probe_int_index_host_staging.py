"""P0-3 子项：`_convert_req_index_to_global_index_native` 的 host staging 来源与替代写法。

背景（2026-09-29 worker trace 归因）：该函数在 prefill 里被调 22 次、产生 55 次
`aten::copy_`，合计 1.225 s（prefill 的 10%）。逐 primitive 定位后（dlc:0，[20,2048]
int32，每个 primitive 单独一个进程）：

    op                                    ms/call    staged DMA/call
    torch.div(x, 64, rounding_mode=floor)   0.505        3     ← 每次调用 3 次 host 往返
    x // 64  (floor_divide)                 0.075        0
    block_table[i64, i64]  (2-D 高级索引)    2.971        0     ← 单次 2.9 ms，函数主要成本
    torch.gather(rows, 1, i64)              0.139        0     ← 同语义，快 21 倍
    bflat.index_select(0, flat)             4.380        0     ← 摊平后 1-D gather 反而更慢
    x.long()  (int32->int64)                0.168        1     ← gather 的 index 必须是 int64

本探针把「旧写法（A：div(floor) + 2-D 高级索引）」与「新写法（D：`//` + 逐行 gather）」
在真机上对比，两者都必须与暴力 CPU reference 逐位一致：

    A 4.04 ms/call、11 次 staged DMA     →     D 0.50 ms/call、1 次 staged DMA

注意：同一个进程里 DLC profiler 事件是累计的（`dlc_get_profile_info` 返回本次
`dlc_enable_profiler()` 以来的全部事件），所以第二次窗口的计数要减掉第一次的。

判据（退出码）：
    0 = D 与 CPU reference 一致，且比 A 快 2x 以上
    1 = 否则（需要继续找替代写法）

用法：
    python -u probe_int_index_host_staging.py [dlc:N] [T] [TOPK] [R] [MB] [BLOCK]
"""
import collections
import gzip
import json
import os
import sys
import tempfile
import time

import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 20
TOPK = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
R = int(sys.argv[4]) if len(sys.argv) > 4 else 8
MB = int(sys.argv[5]) if len(sys.argv) > 5 else 512
BLOCK = int(sys.argv[6]) if len(sys.argv) > 6 else 64
ITERS = 10

torch.dlc.set_device(int(dev.split(":")[-1]))
from torch.profiler import ProfilerActivity, profile  # noqa: E402

STAGED = ("aten::copy_after_async", "driver.memcpy", "synMemCopySync")


def old_impl(req_id, block_table, token_indices, block_size):
    """当前实现（A）：`torch.div(..., floor)` + 2-D 高级索引。"""
    max_blocks = block_table.shape[1]
    num_reqs = block_table.shape[0]
    if max_blocks == 0 or num_reqs == 0:
        return torch.full_like(token_indices, -1)
    valid = token_indices >= 0
    valid &= req_id[:, None] >= 0
    valid &= req_id[:, None] < num_reqs
    block_id = torch.div(token_indices.clamp_min(0), block_size, rounding_mode="floor")
    valid &= block_id < max_blocks
    safe_req = req_id.clamp(0, max(num_reqs - 1, 0)).long()
    safe_block = block_id.clamp(0, max(max_blocks - 1, 0)).long()
    block_number = block_table[safe_req[:, None], safe_block]
    valid &= block_number >= 0
    slots = block_number * block_size + token_indices.clamp_min(0) % block_size
    return torch.where(valid, slots, torch.full_like(slots, -1))


def new_impl(req_id, block_table, token_indices, block_size):
    """新实现（D）：`//` + 每个 token 取一整行 block table 后沿 dim1 gather。"""
    max_blocks = block_table.shape[1]
    num_reqs = block_table.shape[0]
    if max_blocks == 0 or num_reqs == 0:
        return torch.full_like(token_indices, -1)
    positions = token_indices.clamp_min(0)
    block_id = positions // block_size
    inblock_off = positions % block_size
    valid = token_indices >= 0
    valid &= req_id[:, None] >= 0
    valid &= req_id[:, None] < num_reqs
    valid &= block_id < max_blocks
    safe_req = req_id.clamp(0, max(num_reqs - 1, 0))
    rows = block_table[safe_req]  # [T, max_blocks]，1-D 索引，无 staging
    safe_block = block_id.clamp(0, max(max_blocks - 1, 0)).long()
    block_number = torch.gather(rows, 1, safe_block)
    valid &= block_number >= 0
    slots = block_number * block_size + inblock_off
    return torch.where(valid, slots, torch.full_like(slots, -1))


def reference(req_id, block_table, token_indices, block_size):
    """CPU reference，逐行等价改写（不依赖设备 kernel）。"""
    max_blocks = block_table.shape[1]
    num_reqs = block_table.shape[0]
    out = torch.full_like(token_indices, -1)
    if max_blocks == 0 or num_reqs == 0:
        return out
    for t in range(token_indices.shape[0]):
        req = int(req_id[t])
        if not (0 <= req < num_reqs):
            continue
        for k in range(token_indices.shape[1]):
            pos = int(token_indices[t, k])
            if pos < 0:
                continue
            block = pos // block_size
            if block >= max_blocks:
                continue
            number = int(block_table[req, block])
            if number < 0:
                continue
            out[t, k] = number * block_size + pos % block_size
    return out


def bench(fn, iters=ITERS):
    fn()
    torch.dlc.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.dlc.synchronize()
    return (time.perf_counter() - t0) / iters


def staged_events(fn, iters=3):
    """Number of staged DMA events recorded while running ``fn`` ``iters`` times.

    DLC profile events accumulate inside one process, so callers must subtract
    the counts of an earlier window instead of reading them absolutely.
    """
    fn()
    torch.dlc.synchronize()
    fd = tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False)
    fd.close()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.DLC]) as prof:
        for _ in range(iters):
            fn()
        torch.dlc.synchronize()
    prof.export_chrome_trace(fd.name)
    events = json.load(gzip.open(fd.name))["traceEvents"]
    os.remove(fd.name)
    counter = collections.Counter(e["name"] for e in events if e.get("ph") == "X")
    return sum(counter[name] for name in STAGED)


torch.manual_seed(0)
req_id = torch.randint(0, R, (T,), dtype=torch.int32, device=dev)
block_table = torch.randint(0, 1168, (R, MB), dtype=torch.int32, device=dev)
token_indices = torch.randint(0, 60 * MB, (T, TOPK), dtype=torch.int32, device=dev)
keep = torch.rand(T, TOPK, device=dev) < 0.05  # 真实 topk 里绝大多数是 -1
token_indices = torch.where(keep, token_indices, torch.full_like(token_indices, -1))
req_id[0] = -1  # PAD 行
token_indices[1, 0] = -1

print(f"device={dev}  T={T} TOPK={TOPK} R={R} MB={MB} BLOCK={BLOCK}")
print(f"  dtypes: req_id={req_id.dtype} block_table={block_table.dtype} "
      f"token_indices={token_indices.dtype}")

ref = reference(req_id.cpu(), block_table.cpu(), token_indices.cpu(), BLOCK)
outs = {"A old div+2d-index": old_impl(req_id, block_table, token_indices, BLOCK),
        "D new //+gather": new_impl(req_id, block_table, token_indices, BLOCK)}
# 设备侧融合算子是给索引全有效的 576-wide 路径用的，不对 req_id<0 的行做 mask
# （本函数是 512-wide NoPE 的兜底，会 mask），所以只比较 req_id>=0 的行。
valid_rows = req_id.cpu() >= 0
if hasattr(torch, "convert_req_index_to_global_index"):
    outs["fused dlc op"] = torch.convert_req_index_to_global_index(
        req_id.clamp_min(0), block_table, token_indices, BLOCK)

t_old = bench(lambda: old_impl(req_id, block_table, token_indices, BLOCK))
t_new = bench(lambda: new_impl(req_id, block_table, token_indices, BLOCK))
# Fresh window for A, then a second window whose accumulated counter covers A
# plus D; the difference is D's own contribution.
s_old = staged_events(lambda: old_impl(req_id, block_table, token_indices, BLOCK))
s_new = staged_events(lambda: new_impl(req_id, block_table, token_indices, BLOCK))
s_old, s_new = s_old / 3, (s_new - s_old) / 3

for name, out in outs.items():
    got, want = out.cpu(), ref
    if name == "fused dlc op":
        got, want = got[valid_rows], want[valid_rows]
    ok = bool(torch.equal(got, want))
    print(f"  {name:22s} matches CPU reference: {ok}")
    if not ok:
        print(f"      first mismatches: {(got != want).nonzero()[:5].tolist()}")
print(f"\n  A old  {t_old*1e3:8.3f} ms/call   staged={s_old:5.1f}/call")
print(f"  D new  {t_new*1e3:8.3f} ms/call   staged={s_new:5.1f}/call")
print(f"  speedup D/A = {t_old/t_new:.2f}x")

correct = bool(torch.equal(outs["D new //+gather"].cpu(), ref))
sys.exit(0 if correct and t_old / t_new >= 2 else 1)
