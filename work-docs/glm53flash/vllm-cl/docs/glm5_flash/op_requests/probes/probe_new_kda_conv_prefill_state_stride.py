"""#01 第二个 P0 的最小复现：`dlc_kda_causal_conv_prefill` 在非连续 state 视图下的行为。

背景
----
真实运行时 conv_state 不是独立连续缓冲，而是 Mamba 分页池里的一个视图：
  MambaBase.bind_kv_cache -> pages[:, off:off+nbytes].view(dtype).view(-1, *shape)
GLM5 的 KDA 层把 conv_state 与 recurrent_state 打包在同一个 MambaSpec 页里，于是
conv 段的块间行距 = page_size/itemsize，**大于** L*D；即真实的 conv_state 视图
（[S_slot,state_len,D]）是**非连续**的（stride(0) != state_len*D）。

修复前的缺陷：host wrapper 有一个 `direct_state` 快路径，把「满足 stride(1)==1、
stride 偶数、storage_offset 偶数」的 state 直接零拷贝交给 kernel，而 kernel 地址写死成
`slot*state_len*dim`（不读 stride）——于是读错行、并把行间的 recurrent 段写坏。

本脚本用「数据完全相同、只有 state 行距不同」的四组对照复现：
  A) 新算子 + 紧凑行距 L*D            —— 基线
  B) 新算子 + 池子行距 L*D+2048       —— 修复前失败（写坏行间段）；修复后应通过
  C) 现有 torch._causal_conv1d + 池子行距 —— 同一布局下现有算子的表现（对照）
  D) 现有 torch._causal_conv1d + 紧凑行距 —— C 的基线

B 组里那 2048 个「行间」元素在脚本里填了哨兵值 3.0；若跑完哨兵被改，就证明算子写坏了
真实池子里的 recurrent 段，而不只是数值错。

2026-09-24 修复后的行为：wrapper 已删除 `direct_state` 快路径，改用 `kernel.output(conv_state)`
（DLC 侧 `dlc_dlc_contiguous`）：非连续（B 组）会由 torch 自动做「连续副本 + copy_ 回写」，
所以 B 组结果正确、行间段不被改写，代价是单 slot conv 段（18 KB 量级）的一次额外拷贝。

用法：python probe_new_kda_conv_prefill_state_stride.py [dlc:N]    （安全，不会卡设备）
"""
import sys
import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
torch.manual_seed(0)
W, HIST, D = 4, 3, 3072
L = HIST                      # conv_state 宽度（num_spec=0 时 = kernel-1）
GAP = 2048                    # 模拟「行间的 recurrent/SSM 段 + 页填充」
SENTINEL = 3.0
lens = [4, 4]
idxs = [1, 2]


def ref_rows(rows, x_dn, w2d):
    """rows: list(L) of [D]；x_dn: [D,n] -> (out [D,n] bf16, new rows list(L))"""
    rows = [r.clone() for r in rows]
    outs = []
    for t in range(x_dn.shape[1]):
        win = rows[-3:]
        y = sum(w2d[:, k] * win[k] for k in range(3)) + x_dn[:, t] * w2d[:, 3]
        outs.append(torch.nn.functional.silu(y).to(torch.bfloat16))
        rows.append(x_dn[:, t])
        rows = rows[-L:]
    return torch.stack(outs, 1), rows


def build(strided):
    """返回 (pool, state, rows, page)。state 的新 ABI 逻辑形状 [S,L,D]（SD）。"""
    slots, page = 3, (L * D + GAP if strided else L * D)
    pool = torch.full((slots, page), SENTINEL, dtype=torch.bfloat16, device=dev)
    for s in range(slots):
        pool[s, : L * D] = torch.randn(L * D, dtype=torch.bfloat16, device=dev) * 0.4
    if strided:
        state = pool.as_strided((slots, L, D), (page, D, 1), 0)   # [S,L,D] SD，块间行距 page
        rows = {s: [pool[s, t * D:(t + 1) * D].clone() for t in range(L)] for s in idxs}
    else:
        state = pool.view(slots, L, D)                            # [S,L,D] SD，连续
        rows = {s: [state[s][t].clone() for t in range(L)] for s in idxs}
    return pool, state, rows, page


def check(tag, state, pool, page, rows, out, ref_out, ref_new):
    oerr = (out.float() - ref_out.float()).abs().max().item()
    serr = max(max((state[s][t].float() - b.float()).abs().max().item()
                   for t, b in enumerate(ref_new[s]))
               for s in idxs)
    sent = 0
    if page != L * D:                       # 行间段是否被改写
        for s in range(pool.shape[0]):
            sent += int((pool[s, L * D:] != SENTINEL).sum().item())
    print(f"[{tag}] out_err={oerr:.3e}  state_err={serr:.3e}  "
          f"行间段被改写元素数={sent}  (stride0={state.stride(0)}, L*D={L * D})", flush=True)
    return oerr, serr, sent


def main():
    N = sum(lens)
    x = (torch.randn(N, D, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    w2d = (torch.randn(D, W, dtype=torch.float32, device=dev) * 0.3).contiguous()
    w4d = w2d.t().contiguous()                       # 新 ABI: [4, D] fp32
    qsl = torch.tensor([0, lens[0], N], dtype=torch.int32, device=dev)
    sidx = torch.tensor(idxs, dtype=torch.int32, device=dev)
    hinit = torch.tensor([True, True], dtype=torch.bool, device=dev).to(torch.int32)
    print(f"dev={dev} D={D} L={L} GAP={GAP} 用例={lens} idx={idxs}", flush=True)

    for tag, strided in (("A 新算子 + 紧凑行距 L*D   ", False),
                         ("B 新算子 + 池子行距 L*D+GAP", True)):
        pool, state, rows, page = build(strided)
        ref_out = torch.empty(N, D, dtype=torch.bfloat16, device=dev)
        ref_new = {}
        for b, s in enumerate(idxs):
            st, en = int(qsl[b]), int(qsl[b + 1])
            o, nr = ref_rows(rows[s], x[st:en].transpose(0, 1).contiguous(), w2d)
            ref_out[st:en] = o.transpose(0, 1)
            ref_new[s] = nr
        out = torch.dlc_kda_causal_conv_prefill(x, w4d, state, qsl, sidx, hinit,
                                                activation="silu", null_state_id=0)
        torch.dlc.synchronize()
        check(tag, state, pool, page, rows, out, ref_out, ref_new)

    # C/D：现有算子在同样两种布局下的表现（它内部会先做 contigeous 拷贝，期望都正确）
    for tag, strided in (("D 现有 _causal_conv1d + 紧凑行距", False),
                         ("C 现有 _causal_conv1d + 池子行距", True)):
        pool, sd_state, rows, page = build(strided)     # sd_state 供校验读值
        ds_state = sd_state.transpose(1, 2)             # 旧算子要 DS（[S,D,L]）
        ref_out = torch.empty(N, D, dtype=torch.bfloat16, device=dev)
        ref_new = {}
        for b, s in enumerate(idxs):
            st, en = int(qsl[b]), int(qsl[b + 1])
            o, nr = ref_rows(rows[s], x[st:en].transpose(0, 1).contiguous(), w2d)
            ref_out[st:en] = o.transpose(0, 1)
            ref_new[s] = nr
        try:
            out_cf = torch._causal_conv1d(
                x.transpose(0, 1), w2d.to(torch.bfloat16),
                torch.zeros(D, dtype=torch.bfloat16, device=dev), ds_state, qsl, sidx, hinit)
            out = out_cf.transpose(0, 1).contiguous()
            torch.dlc.synchronize()
            check(tag, sd_state, pool, page, rows, out, ref_out, ref_new)
        except Exception as e:                                   # noqa: BLE001
            print(f"[{tag}] 调用失败: {type(e).__name__}: {str(e)[:160]}", flush=True)


main()
