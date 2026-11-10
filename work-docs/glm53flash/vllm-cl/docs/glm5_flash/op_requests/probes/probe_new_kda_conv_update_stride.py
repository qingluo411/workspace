"""#02 `dlc_kda_causal_conv_update` 在分页池子布局（真实 GLM5 形态）下的正确性 + 开销对照。

背景：真实 conv_state 不是独立连续缓冲，而是 Mamba 分页池里的一个视图
（`MambaBase.bind_kv_cache`：`pages[:, off:off+nbytes].view(dtype).view(-1, *shape)`）。
GLM5 的 KDA 层把 conv_state 与 recurrent_state 打包在同一个 MambaSpec 页里，
所以 conv 段的块间行距 = page_size/itemsize，**大于** conv 段本身的 L*D —— 即真实的
conv_state 视图是**非连续**的（`state.stride(0) != L*D`）。

本脚本用同一份数据、两种 state 视图各跑一次：

  A) 紧凑 [S,L,D]（stride(0)=L*D，连续）；
  B) 池子 as_strided -> [S,L,D]（stride(0)=L*D+GAP），行间 GAP 段预填哨兵 3.0。

对 out / 就地 state / 行间哨兵各做断言，并与 fp32 逐窗递推 reference 对比，同时计时。

2026-09-24 算子组修复后的行为（本脚本的实测结论）：
  - kernel 自身仍按 slot*state_len*D 紧凑寻址、不读 stride；
  - 但 pytorch wrapper 用 `dlc_dlc_contiguous` 提交 conv_state：非连续（B 组）会被
    torch 侧自动做一次「连续副本 + 结束后 copy_ 回写」，所以 B 组**结果正确**，
    代价是每次调用多一次「单个 slot 的 conv 段」拷贝（18 KB 量级，不是整池）。
  - A 组是连续张量，零拷贝。

用法：
    python -u probe_new_kda_conv_update_stride.py [dlc:N]     # 默认 dlc:0
安全：不会卡设备。
"""
import sys
import time
import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
torch.manual_seed(0)
W, L, D = 4, 3, 3072
GAP, SENT = 2048, 3.0
B = 2
idxs = [1, 2]


def ref(base_LD, x_TD, w2d):
    """逐窗递推：h:[L,D] 旧->新；返回 (out[T,D] bf16, 末态 snap[L,D] bf16)。"""
    h = base_LD.float().clone()
    outs = []
    for t in range(x_TD.shape[0]):
        y = (h[0] * w2d[:, 0] + h[1] * w2d[:, 1] + h[2] * w2d[:, 2]
             + x_TD[t].float() * w2d[:, 3])
        outs.append(torch.nn.functional.silu(y).to(torch.bfloat16))
        h = torch.cat([h[1:], x_TD[t].float()[None]], 0)
    return torch.stack(outs), h.to(torch.bfloat16)


x = (torch.randn(B, D, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
w2d = (torch.randn(D, W, dtype=torch.float32, device=dev) * 0.3).contiguous()
w4d = w2d.t().contiguous()                       # 新 ABI: [4, D] fp32
qsl = torch.arange(B + 1, dtype=torch.int32, device=dev)
sidx = torch.tensor(idxs, dtype=torch.int32, device=dev)

print(f"dev={dev} D={D} L={L} GAP={GAP}  (state 视图不同、数据相同)", flush=True)
for tag, strided in (("A 紧凑行距 L*D   ", False), ("B 池子行距 L*D+GAP", True)):
    slots, page = 3, (L * D + GAP if strided else L * D)
    pool = torch.full((slots, page), SENT, dtype=torch.bfloat16, device=dev)
    for s in range(slots):
        pool[s, : L * D] = torch.randn(L * D, dtype=torch.bfloat16, device=dev) * 0.4
    if strided:
        # [S,L,D] SD 视图，块间行距 = page = L*D+GAP（非连续）
        state = pool.as_strided((slots, L, D), (page, D, 1), 0)
    else:
        state = pool.view(slots, L, D)
    rows = {s: pool[s, : L * D].view(L, D).clone() for s in idxs}

    ref_out = torch.empty(B, D, dtype=torch.bfloat16, device=dev)
    ref_snap = {}
    for b, s in enumerate(idxs):
        o, snap = ref(rows[s], x[b:b + 1], w2d)
        ref_out[b] = o[0]
        ref_snap[s] = snap

    out = torch.dlc_kda_causal_conv_update(
        x, w4d, state, sidx, query_start_loc=qsl, max_query_len=1,
        activation="silu", null_state_id=0)
    torch.dlc.synchronize()

    oerr = (out.float() - ref_out.float()).abs().max().item()
    serr = max((pool[s, : L * D].view(L, D).float() - ref_snap[s].float()).abs().max().item()
               for s in idxs)
    sent = 0
    if strided:
        for s in range(slots):
            sent = max(sent, int((pool[s, L * D:] != SENT).sum().item()))

    # 计时用第二次调用（会再次改写 state，放在校验之后）
    t = time.time()
    torch.dlc_kda_causal_conv_update(
        x, w4d, state, sidx, query_start_loc=qsl, max_query_len=1,
        activation="silu", null_state_id=0)
    torch.dlc.synchronize()
    dt = time.time() - t
    print(f"[{tag}] out_err={oerr:.3e} state_err={serr:.3e} "
          f"行间段被改写元素={sent} state_contiguous={state.is_contiguous()} t={dt * 1e3:.2f}ms",
          flush=True)
