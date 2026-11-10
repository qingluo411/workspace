"""#04 新算子 dlc_kda_recurrent_decode 实测（2026-09-22）。

GLM5 几何：H=8(TP8), Dk=Dv=128, state=[S_slot,H,128,128] fp32。
reference：patch_kda._run_kda 的逐 token fp32 递推（q/k L2Norm eps=1e-6、safe gate、beta sigmoid）。
kernel 约定：slot 0 = null（输出 0、state 不写）；num_accepted_tokens 后提交 state 快照。
"""
import sys
import time
import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
torch.manual_seed(0)
H, DK, DV = 8, 128, 128
LB, SCALE = -5.0, 128 ** -0.5


def ref_run(states, idxs, q, k, v, raw_g, beta, a_log, g_bias, lens, acc=None):
    N = q.shape[1]
    out = torch.empty(1, N, H, DV, dtype=torch.bfloat16, device=dev)
    snap = states.clone()
    qsl = [0]
    for l in lens:
        qsl.append(qsl[-1] + l)
    for b in range(len(lens)):
        s, e = qsl[b], qsl[b + 1]
        if e <= s:
            continue
        slot = int(idxs[b])
        if slot == 0:
            out[0, s:e] = 0
            continue
        hv = states[slot].float().clone()
        accepted = (e - s) if acc is None else int(acc[b])
        for t in range(s, e):
            qt, kt, vt = q[0, t].float(), k[0, t].float(), v[0, t].float()
            qt = qt * torch.rsqrt(qt.square().sum(-1, keepdim=True) + 1e-6)
            kt = kt * torch.rsqrt(kt.square().sum(-1, keepdim=True) + 1e-6)
            gate = LB * torch.sigmoid(torch.exp(a_log)[:, None] * (raw_g[0, t].float() + g_bias))
            hv = hv * torch.exp(gate).unsqueeze(1)
            bt = torch.sigmoid(beta[0, t].float())
            delta = (vt - torch.bmm(hv, kt.unsqueeze(-1)).squeeze(-1)) * bt[:, None]
            hv = hv + delta.unsqueeze(-1) * kt.unsqueeze(1)
            out[0, t] = torch.bmm(hv, (qt * SCALE).unsqueeze(-1)).squeeze(-1).to(torch.bfloat16)
            if t - s + 1 == accepted:
                snap[slot] = hv.clone()
    return out, snap


def case(name, lens, slots=4, idxs=None, acc=None, hinit=True):
    B, N = len(lens), sum(lens)
    idxs = idxs if idxs is not None else list(range(1, B + 1))
    q = (torch.randn(1, N, H, DK, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    k = (torch.randn(1, N, H, DK, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    v = (torch.randn(1, N, H, DV, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    raw_g = (torch.randn(1, N, H, DK, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    beta = (torch.randn(1, N, H, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    a_log = (torch.randn(H, dtype=torch.float32, device=dev) * 0.5).contiguous()
    g_bias = (torch.randn(H, DK, dtype=torch.float32, device=dev) * 0.2).contiguous()
    states = torch.randn(slots, H, DK, DV, dtype=torch.float32, device=dev) * 0.05
    sidx = torch.tensor(idxs, dtype=torch.int32, device=dev)
    qsl = torch.tensor([0] + torch.cumsum(torch.tensor(lens), 0).tolist(),
                       dtype=torch.int32, device=dev)
    acc_t = None if acc is None else torch.tensor(acc, dtype=torch.int32, device=dev)

    ref_out, ref_states = ref_run(states, idxs, q, k, v, raw_g, beta, a_log, g_bias, lens, acc)

    st = states.clone()
    out = torch.empty(1, N, H, DV, dtype=torch.bfloat16, device=dev)
    t = time.time()
    torch.dlc_kda_recurrent_decode(q, k, v, raw_g, beta, a_log, g_bias, st, sidx, qsl,
                                   acc_t, lower_bound=LB, scale=SCALE, out=out)
    torch.dlc.synchronize()
    dt = time.time() - t
    oerr = (out.float() - ref_out.float()).abs().max().item()
    serr = (st.float() - ref_states.float()).abs().max().item()
    print(f"[{name}] lens={lens} idx={idxs} acc={acc} slots={slots}", flush=True)
    print(f"    out_err={oerr:.3e}  state_err={serr:.3e}  t={dt*1e3:.2f}ms", flush=True)
    if 0 in idxs:
        r = idxs.index(0)
        s0, e0 = int(qsl[r]), int(qsl[r + 1])
        print(f"    null slot: out_all_zero={bool((out[0, s0:e0] == 0).all())} "
              f"state_unchanged={bool(torch.equal(st[0], states[0]))}", flush=True)
    return out, st


print(f"dev={dev} H={H} Dk={DK} Dv={DV}", flush=True)
print("=== 1) plain decode B=4 单 token，含 null slot ===", flush=True)
case("decode-b4", [1, 1, 1, 1], idxs=[1, 2, 3, 0])
print("=== 2) plain decode B=8 单 token ===", flush=True)
case("decode-b8", [1] * 8, slots=10, idxs=[1, 2, 3, 4, 5, 6, 7, 8])
print("=== 3) spec: T=2, accepted=[1,2] ===", flush=True)
case("spec-T2", [2, 2], idxs=[1, 2], acc=[1, 2])
print("=== 4) spec: T=4, accepted=[0,3] ===", flush=True)
case("spec-T4", [4, 4], idxs=[1, 2], acc=[0, 3])
print("=== 5) 变长混排: [3,1,2] ===", flush=True)
case("varlen", [3, 1, 2], idxs=[1, 2, 3])
print("=== 6) 异常路径 ===", flush=True)
try:
    case("bad-lb", [1], idxs=[1])
except Exception as e:
    print("   ->", type(e).__name__, str(e)[:150], flush=True)
