"""#02 槽位规模扫描：新 update 算子 vs 已有 _causal_conv1d_update（2026-09-22）。

真实几何：num_blocks=1509、每块 conv 状态 3x3072 bf16（18KB，打包在共享页里）。
本脚本扫描 state 槽位数，测「可用性（是否卡死）+ 正确性 + 耗时」。

2026-09-24 ABI：conv_state=[S_slot,state_len,D](SD，in-place)，weight=[4,D] fp32，只返回 out。
"""
import sys
import time
import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
torch.manual_seed(0)
W, HIST = 4, 3
D = 3072


def ref_one(base_slot, xs, w2d, accepted):
    """base_slot:[L,D] 旧->新；xs:[T,D] -> ([T,D] bf16, snap[L,D])"""
    L = base_slot.shape[0]
    h = base_slot.float().clone()
    outs = []
    for t in range(xs.shape[0]):
        y = torch.nn.functional.silu(
            h[L - 3] * w2d[:, 0] + h[L - 2] * w2d[:, 1] + h[L - 1] * w2d[:, 2]
            + xs[t].float() * w2d[:, 3])
        outs.append(y.to(torch.bfloat16))
        h = torch.cat([h[1:], xs[t].float()[None]], 0)
        if t == accepted - 1:
            snap = h.clone()
    return (torch.stack(outs), snap if outs else base_slot.clone())


def run(slots, B=8, T=1, accepted=None):
    xs = (torch.randn(B * T, D, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    w2d = (torch.randn(D, W, dtype=torch.float32, device=dev) * 0.3).contiguous()
    w4d = w2d.t().contiguous()                       # 新 ABI: [4, D] fp32
    base = torch.randn(slots, HIST, D, dtype=torch.bfloat16, device=dev) * 0.5
    # 注意：slots <= B 时 idx 会重复，服务端不会这样做（同一 slot 只能被一个请求占用），
    # 这几行只用于看「时间随槽位规模的趋势」，正确性看 slots >= 1024 的行
    idx = (torch.arange(B, dtype=torch.int32, device=dev) % (slots - 1)) + 1
    qsl = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=dev)
    acc = None if accepted is None else torch.full((B,), accepted, dtype=torch.int32, device=dev)

    ref_out = torch.empty(B * T, D, dtype=torch.bfloat16, device=dev)
    ref_buf = base.clone()
    for b in range(B):
        o, snap = ref_one(base[idx[b]], xs[qsl[b]:qsl[b + 1]], w2d,
                          (qsl[b + 1] - qsl[b]) if acc is None else int(acc[b]))
        ref_out[qsl[b]:qsl[b + 1]] = o
        ref_buf[idx[b]] = snap

    # 先各跑一次做 warm-up（排除首次 kernel 加载/JIT 的一次性开销），再计时
    torch.dlc_kda_causal_conv_update(
        xs, w4d, base.clone(), idx, query_start_loc=qsl,
        num_accepted_tokens=acc, max_query_len=T, activation="silu", null_state_id=0)
    torch._causal_conv1d_update(
        xs, w2d.to(torch.bfloat16), torch.zeros(D, dtype=torch.bfloat16, device=dev),
        base.clone().transpose(-1, -2), idx, acc, qsl if T > 1 else None, T if T > 1 else -1,
        None, None, True, True)
    torch.dlc.synchronize()

    buf = base.clone()
    t = time.time()
    out = torch.dlc_kda_causal_conv_update(
        xs, w4d, buf, idx, query_start_loc=qsl,
        num_accepted_tokens=acc, max_query_len=T, activation="silu", null_state_id=0)
    torch.dlc.synchronize()
    dt_new = time.time() - t
    oerr = (out.float() - ref_out.float()).abs().max().item()
    serr = (buf.float() - ref_buf.float()).abs().max().item()

    b_old = base.clone()
    t = time.time()
    o_old = torch._causal_conv1d_update(
        xs, w2d.to(torch.bfloat16), torch.zeros(D, dtype=torch.bfloat16, device=dev),
        b_old.transpose(-1, -2), idx, acc, qsl if T > 1 else None, T if T > 1 else -1,
        None, None, True, True)
    torch.dlc.synchronize()
    dt_old = time.time() - t
    oerr_old = (o_old.float() - ref_out.float()).abs().max().item()
    serr_old = (b_old.float() - ref_buf.float()).abs().max().item()
    print(f"slots={slots:5d} B={B} T={T} acc={accepted} | "
          f"new: out_err={oerr:.3e} state_err={serr:.3e} t={dt_new*1e3:8.2f}ms | "
          f"old: out_err={oerr_old:.3e} state_err={serr_old:.3e} t={dt_old*1e3:6.2f}ms",
          flush=True)


print(f"dev={dev}  D={D} L={HIST}", flush=True)
for slots in [4, 256, 1024, 1509]:
    run(slots)
print("--- spec 形态（T=4, accepted=3）---", flush=True)
for slots in [4, 1024, 1509]:
    run(slots, B=8, T=4, accepted=3)
print("--- 与 GLM5 真实页数同量级的大槽位 ---", flush=True)
for slots in [2048, 3072]:
    run(slots)
