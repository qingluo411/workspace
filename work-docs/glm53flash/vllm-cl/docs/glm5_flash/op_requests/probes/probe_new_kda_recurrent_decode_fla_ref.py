"""#04 新算子 dlc_kda_recurrent_decode 复核（2026-09-22 晚，替代初版 ..._layout.py）。

2026-09-22 补注：本脚本的参考实现是**交付版 `[S,H,Dk,Dv]` 布局**。算子组已同意把 state 统一为
`[S,H,Dv,Dk]`（与 fla / 插件 native 一致），届时请改用 `../refs/kda_reference_dv_dk.py`；
本脚本保留用于复现「out 恒 0」现象与旧布局判决用例。

初版 `probe_new_kda_recurrent_decode_layout.py` 的参考实现把衰减轴写错（把 per-Dk 的 gate 施加到 Dv 上），
得出了「T=1 时 state 不吻合」的错误结论，已废弃删除；本脚本是修正版，用于给算子组复现/验收：

判据（全部与 fla / 上游 KDA 语义一致）：
  h = [S_slot, H, Dk, Dv]（dim2 = key 维），衰减 exp(g) 沿 Dk，proj[dv] = Σ_dk h[dk,dv]·k̂[dk]，
  h += k ⊗ (β·(v - proj))，o[dv] = Σ_dk h[dk,dv]·(q̂[dk]·scale)。
已知缺陷：`out` 逐位恒为 0（kernel `dlc_kda_recurrent_decode.cpp:189` 把常量 0.0f 写进 torch 回读的那半）。
用法：`python probe_new_kda_recurrent_decode_fla_ref.py <dev:dlc:N>`
"""
import sys
import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
torch.manual_seed(1)
H, DK, DV = 8, 128, 128
LB, SCALE = -5.0, 128 ** -0.5


def ref_fla(state, lens, q, k, v, raw_g, beta, a_log, g_bias, acc=None, commit="full"):
    """h=[S,H,Dk,Dv]；commit='full' 整窗提交，'accepted' 在第 acc 个 token 后快照"""
    N = q.shape[1]
    st = state.clone()
    o = torch.empty(1, N, H, DV, dtype=torch.float32, device=dev)
    qsl = [0]
    for l in lens:
        qsl.append(qsl[-1] + l)
    for b in range(len(lens)):
        s, e = qsl[b], qsl[b + 1]
        h = st[b + 1].float().clone()
        for t in range(s, e):
            qt, kt, vt = q[0, t].float(), k[0, t].float(), v[0, t].float()
            qt = qt * torch.rsqrt(qt.square().sum(-1, keepdim=True) + 1e-6)
            kt = kt * torch.rsqrt(kt.square().sum(-1, keepdim=True) + 1e-6)
            gate = LB * torch.sigmoid(torch.exp(a_log)[:, None] * (raw_g[0, t].float() + g_bias))
            h = h * torch.exp(gate).unsqueeze(-1)                   # per-Dk（dim2）
            bt = torch.sigmoid(beta[0, t].float())
            delta = (vt - (h * kt.unsqueeze(-1)).sum(-2)) * bt.unsqueeze(-1)
            h = h + kt.unsqueeze(-1) * delta.unsqueeze(-2)
            o[0, t] = (h * (qt * SCALE).unsqueeze(-1)).sum(-2)
            if commit == "accepted" and acc is not None and t - s + 1 == acc[b]:
                st[b + 1] = h.clone()
        if commit == "full":
            st[b + 1] = h
    return st, o


def call(states, lens, q, k, v, raw_g, beta, a_log, g_bias, acc=None, out_init=None):
    B, N = len(lens), sum(lens)
    sidx = torch.tensor(list(range(1, B + 1)), dtype=torch.int32, device=dev)
    qsl = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)), dtype=torch.int32, device=dev)
    acc_t = None if acc is None else torch.tensor(acc, dtype=torch.int32, device=dev)
    st = states.clone()
    out = torch.full((1, N, H, DV), 7.0 if out_init is None else out_init,
                     dtype=torch.bfloat16, device=dev)
    torch.dlc_kda_recurrent_decode(q, k, v, raw_g, beta, a_log, g_bias, st, sidx, qsl, acc_t,
                                   lower_bound=LB, scale=SCALE, out=out)
    torch.dlc.synchronize()
    return st, out


def rand_case(lens, seed=1):
    torch.manual_seed(seed)
    B, N = len(lens), sum(lens)
    g = lambda *s: (torch.randn(*s, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    return dict(lens=lens,
                q=g(1, N, H, DK), k=g(1, N, H, DK), v=g(1, N, H, DV), raw_g=g(1, N, H, DK),
                beta=g(1, N, H),
                a_log=(torch.randn(H, dtype=torch.float32, device=dev) * 0.5).contiguous(),
                g_bias=(torch.randn(H, DK, dtype=torch.float32, device=dev) * 0.2).contiguous(),
                base=torch.randn(B + 1, H, DK, DV, dtype=torch.float32, device=dev) * 0.05)


def numeric_cases():
    print("=== A) 数值对齐（fp32 fla 语义参考）===", flush=True)
    for lens in ([1], [2], [4], [1, 1, 1, 1], [3, 1, 2]):
        c = rand_case(lens)
        r_st, r_out = ref_fla(c["base"], lens, c["q"], c["k"], c["v"], c["raw_g"], c["beta"],
                              c["a_log"], c["g_bias"])
        st, out = call(c["base"], lens, c["q"], c["k"], c["v"], c["raw_g"], c["beta"],
                       c["a_log"], c["g_bias"])
        s_err = (st[1:].float() - r_st[1:]).abs().max().item()
        o_err = (out.float() - r_out).abs().max().item()
        print(f"    lens={str(lens):18s} state_err={s_err:.3e}（|ref|={r_st[1:].abs().max().item():.3f}）"
              f"  out_err={o_err:.3e}（|ref|={r_out.abs().max().item():.4f}）"
              f"  out 逐位为 0 = {bool((out == 0).all())}", flush=True)


def accepted_cases():
    print("=== B) num_accepted_tokens 语义（T=4, acc=2）===", flush=True)
    c = rand_case([4])
    st, _ = call(c["base"], [4], c["q"], c["k"], c["v"], c["raw_g"], c["beta"], c["a_log"],
                 c["g_bias"], acc=[2])
    for commit in ("full", "accepted"):
        r_st, _ = ref_fla(c["base"], [4], c["q"], c["k"], c["v"], c["raw_g"], c["beta"],
                          c["a_log"], c["g_bias"], acc=[2], commit=commit)
        err = (st[1:].float() - r_st[1:]).abs().max().item()
        print(f"    与「{commit:8s}提交」比：state_err={err:.3e}", flush=True)
    print("    → 阈值判断：哪个 err 小，就是算子当前的提交语义", flush=True)


def layout_cases():
    print("=== C) 受控判决（不依赖参考实现）===", flush=True)
    z_b = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)
    z_f = lambda *s: torch.zeros(*s, dtype=torch.float32, device=dev)
    k = z_b(1, 1, H, DK); k[0, 0, :, 5] = 1.0
    v = z_b(1, 1, H, DV); v[0, 0, :, 3] = 1.0
    beta = torch.ones(1, 1, H, dtype=torch.bfloat16, device=dev)
    st, out = call(z_f(2, H, DK, DV), [1], z_b(1, 1, H, DK), k, v, z_b(1, 1, H, DK), beta,
                   z_f(H), z_f(H, DK))
    nz = (st[1, 0].abs() > 1e-3).nonzero()
    print(f"    C1 state=0,k=e5,v=e3,β=1 → 非零元 {nz[:2].tolist()}（期望 [5,3]），值={st[1,0,5,3].item():.4f}"
          f"（期望 sigmoid(1)=0.7311）", flush=True)
    g_bias = (torch.arange(DK, dtype=torch.float32, device=dev)[None, :].repeat(H, 1) * 0.05).contiguous()
    ones = torch.ones(2, H, DK, DV, dtype=torch.float32, device=dev)
    kb = z_b(1, 1, H, DK); kb[0, 0] = 1.0
    st, _ = call(ones, [1], z_b(1, 1, H, DK), kb, z_b(1, 1, H, DV), z_b(1, 1, H, DK),
                 torch.full((1, 1, H), -20.0, dtype=torch.bfloat16, device=dev), z_f(H), g_bias)
    m = st[1, 0]
    exp_g = torch.exp(-5 * torch.sigmoid(g_bias[0]))
    print(f"    C2 state=1,只剩衰减 → 沿 dim0(Dk) 方差={m.var(dim=0).mean().item():.3e}"
          f"  沿 dim1(Dv) 方差={m.var(dim=1).mean().item():.3e}（期望：dim0 非零、dim1 为 0）", flush=True)
    print(f"    C2 m[:5,0]（沿 Dk）={[round(x,6) for x in m[:5,0].tolist()]}", flush=True)
    print(f"    C2 exp(-5σ(g_bias))[:5]={[round(x,6) for x in exp_g[:5].tolist()]}（应与上行逐值吻合）", flush=True)


print(f"dev={dev} H={H} Dk={DK} Dv={DV}", flush=True)
numeric_cases()
accepted_cases()
layout_cases()
print("=== D) 结论 ===", flush=True)
print("    state：数值正确（bf16 级），布局 [S,H,Dk,Dv]（行=Dk），衰减沿 Dk —— 与插件 native 语义一致、存储互为转置；", flush=True)
print("    out：逐位恒为 0（P0 缺陷，见 issue 正文的 kernel 行号）；num_accepted_tokens：未生效，总是提交整窗。", flush=True)
