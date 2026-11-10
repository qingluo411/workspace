"""#02 state 宽度（num_spec）与 spec 语义复核（2026-09-22）。

GLM5 的 conv state 宽度 L = kernel-1+num_spec。真机 num_blocks=1509、L=3+num_spec。
参考实现：state 是「整窗」滑动的 L 行缓冲，滑窗读最后 3 行，新 token 追加到末行；
         num_accepted_tokens 之后提交快照。
本脚本：L=3/4/6 下对比 新算子 dlc_kda_causal_conv_update 与 已有 _causal_conv1d_update。

2026-09-24 ABI：conv_state=[S_slot,state_len,D](SD，in-place)，weight=[4,D] fp32，只返回 out。
"""
import sys
import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "dlc:4"
torch.manual_seed(0)
W = 4
D = 3072


def ref_one(slot_state, xs, w2d, accepted):
    """slot_state:[L,D] 旧->新；xs:[T,D] -> ([T,D], snap[L,D])"""
    h = slot_state.float().clone()
    L = h.shape[0]
    snap = slot_state.float().clone()
    outs = []
    for t in range(xs.shape[0]):
        y = torch.nn.functional.silu(
            h[L - 3] * w2d[:, 0] + h[L - 2] * w2d[:, 1] + h[L - 1] * w2d[:, 2]
            + xs[t].float() * w2d[:, 3])
        outs.append(y.to(torch.bfloat16))
        h = torch.cat([h[1:], xs[t].float()[None]], 0)
        if t == accepted - 1:
            snap = h.clone()
    return (torch.stack(outs) if outs else xs.new_empty((0, D)), snap.to(slot_state.dtype))


def case(name, L, lens, acc=None, slots=4, bias=0.0):
    B, N = len(lens), sum(lens)
    idxs = list(range(1, B + 1))
    xs = (torch.randn(N, D, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    w2d = (torch.randn(D, W, dtype=torch.float32, device=dev) * 0.3).contiguous()
    w4d = w2d.t().contiguous()                       # 新 ABI: [4, D] fp32
    base = torch.randn(slots, L, D, dtype=torch.bfloat16, device=dev) * 0.5
    qsl = torch.tensor([0] + torch.cumsum(torch.tensor(lens), 0).tolist(),
                       dtype=torch.int32, device=dev)
    sidx = torch.tensor(idxs, dtype=torch.int32, device=dev)
    acc_t = None if acc is None else torch.tensor(acc, dtype=torch.int32, device=dev)

    ref_out = torch.empty(N, D, dtype=torch.bfloat16, device=dev)
    ref_buf = base.clone()
    for b in range(B):
        a = (lens[b] if acc is None else acc[b])
        o, snap = ref_one(base[idxs[b]], xs[qsl[b]:qsl[b + 1]], w2d, a)
        ref_out[qsl[b]:qsl[b + 1]] = o
        ref_buf[idxs[b]] = snap

    buf = base.clone()
    out = torch.dlc_kda_causal_conv_update(
        xs, w4d, buf, sidx, query_start_loc=qsl,
        num_accepted_tokens=acc_t, max_query_len=max(lens),
        activation="silu", null_state_id=0)
    torch.dlc.synchronize()
    oerr = (out.float() - ref_out.float()).abs().max().item()
    serr = (buf.float() - ref_buf.float()).abs().max().item()

    b_old = base.clone()
    o_old = torch._causal_conv1d_update(
        xs, w2d.to(torch.bfloat16), torch.full((D,), bias, dtype=torch.bfloat16, device=dev),
        b_old.transpose(-1, -2), sidx, acc_t, qsl, L, None, None, True, True)
    torch.dlc.synchronize()
    oerr_old = (o_old.float() - ref_out.float()).abs().max().item()
    serr_old = (b_old.float() - ref_buf.float()).abs().max().item()

    print(f"[{name}] L={L} lens={lens} acc={acc}", flush=True)
    print(f"    新算子: out_err={oerr:.3e} state_err={serr:.3e}", flush=True)
    print(f"    已有  : out_err={oerr_old:.3e} state_err={serr_old:.3e}", flush=True)


print(f"dev={dev} D={D}", flush=True)
print("=== L=3（num_spec=0）===", flush=True)
case("L3-T1", 3, [1, 1, 1])
case("L3-T2", 3, [2, 2], acc=[1, 2])
print("=== L=4（num_spec=1）===", flush=True)
case("L4-T2", 4, [2, 2], acc=[1, 2])
case("L4-T2-acc0", 4, [2, 2], acc=[0, 1])
print("=== L=6（num_spec=3）===", flush=True)
case("L6-T4", 6, [4, 4], acc=[2, 4])
case("L6-T4-acc0", 6, [4, 4], acc=[0, 3])
