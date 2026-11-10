"""GLM-5.3-Flash KDA 参考实现（state 布局 [.., Dv, Dk]）。

对应两个算子（算子组 2026-09-22 决定把 state 布局统一到 [.., Dv, Dk]，与 fla /
插件 native 一致；本条参考实现即按新布局给出）：

  * #03 `torch.dlc_kda_chunk_prefill`  —— `chunk_prefill_ref()`
  * #04 `torch.dlc_kda_recurrent_decode` —— `recurrent_decode_ref()`

两者共用同一套语义，逐 token 递推（等价于 fla 的 `fused_recurrent_kda`）：

    state : [S, H, Dv, Dk] fp32，S = slot 数；**行 = value 维 dv，列 = key 维 dk**
    gate  : g[t,h,dk] = lower_bound * sigmoid(exp(a_log[h]) * (raw_g[t,h,dk] + g_bias[h,dk]))
    递推  : S       = S * exp(g_t)                       # 沿 Dk 列乘（每列一个标量）
            proj[dv]= sum_dk S[dv,dk] * k_hat[t,dk]
            S[dv,dk]+= k_hat[t,dk] * beta_t * (v[t,dv] - proj[dv])
            o[t,dv] = sum_dk S[dv,dk] * q_hat[t,dk]        # 注意是**更新后**的 S
    q_hat = l2norm(q, eps=1e-6) * scale，k_hat = l2norm(k, eps=1e-6)，beta = sigmoid(beta_logits)

与旧版参考（`probe_new_kda_recurrent_decode_fla_ref.py`，`[S,H,Dk,Dv]`、行=Dk）的关系：
同一条公式，只是把每个 (slot, head) 的 128x128 矩阵转置存放 —— 即
`state_dv_dk[h] == state_dk_dv[h].T`。

用法：见文件末尾 `main()`；不带参数时跑 `--selftest`（纯 CPU，不碰 DLC 设备）：
    python kda_reference_dv_dk.py --selftest
"""
from __future__ import annotations

import argparse
import torch

# --------------------------------------------------------------------------
# 语义原语
# --------------------------------------------------------------------------


def kda_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """在最后一维（Dk）上做 l2 归一化；与 kernel 的 rsqrt(sum + 1e-6) 一致。"""
    x = x.float()
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps)


def kda_gate(raw_g: torch.Tensor, a_log: torch.Tensor, g_bias: torch.Tensor,
             lower_bound: float = -5.0) -> torch.Tensor:
    """per-channel gate。

    raw_g [1,N,H,Dk]（或 [N,H,Dk]）bf16、a_log [H] fp32、g_bias [H,Dk] fp32
    -> gate [N,H,Dk] fp32。

    注：两个算子的 `g_bias` 公开签名不一致 —— #03 是 `[Dk,H]`（wrapper 内部
    `permute({1,0})`），#04 是 `[H,Dk]`。本函数统一收 `[H,Dk]`，调用 #03 时请传
    `g_bias.t().contiguous()`。
    """
    rg = raw_g.float()
    if rg.ndim == 4:
        rg = rg[0]
    heads, dk = rg.shape[-2], rg.shape[-1]
    bias = g_bias.float().reshape(heads, dk)
    rate = torch.exp(a_log.float().reshape(heads))[:, None]
    return lower_bound * torch.sigmoid(rate[None] * (rg + bias[None]))


def kda_beta(beta_logits: torch.Tensor) -> torch.Tensor:
    """beta_logits [1,N,H]（或 [N,H]）-> beta = sigmoid(...)，[N,H] fp32。"""
    b = beta_logits.float()
    if b.ndim == 3:
        b = b[0]
    return torch.sigmoid(b)


def _check_state(state: torch.Tensor, heads: int, dv: int, dk: int) -> None:
    if state.dim() != 4 or tuple(state.shape[1:]) != (heads, dv, dk):
        raise ValueError(
            f"state 必须是 [S,H,Dv,Dk] 布局（行=value、列=key），"
            f"期望 [*,{heads},{dv},{dk}]，实际 {tuple(state.shape)}；"
            f"若是 [..,Dk,Dv] 的旧布局请先 .transpose(-1,-2).contiguous()")
    if state.dtype != torch.float32:
        raise ValueError(f"state 必须是 fp32，实际 {state.dtype}")


def _matrices(qh, kh, G):
    """片内衰减矩阵（供 chunk 版使用）。

    qh/kh: [C,H,Dk]；G: [C,H,Dk] 片内累积 gate（含当前 token，<=0 递减）。
    返回 (Aout [C,C,H], Mkk [C,C,H])，均已按 j<=t / j<t 掩码：
      Aout[t,j,h] = sum_dk qh[t]*kh[j]*exp(G_t-G_j)   (j<=t)
      Mkk [t,j,h] = sum_dk kh[t]*kh[j]*exp(G_t-G_j)   (j<t)
    """
    C = qh.shape[0]
    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=qh.device))
    strict = lower & ~torch.eye(C, dtype=torch.bool, device=qh.device)
    d = G[:, None] - G[None, :]                      # [C,C,H,Dk]，t>=j 时 <=0
    # 上三角（j>t）先归零再 exp，否则 exp(正值) 会溢出；掩码后面还会再乘一次
    d = d.masked_fill(~lower[:, :, None, None], 0.0)
    decay = torch.exp(d)
    Aout = (decay * (qh[:, None] * kh[None, :])).sum(-1) * lower[:, :, None]
    Mkk = (decay * (kh[:, None] * kh[None, :])).sum(-1) * strict[:, :, None]
    return Aout, Mkk


def _forward_subst_unit_lower(L, rhs):
    """解 (I + L) x = rhs：L [H,C,C] 严格下三角、rhs [C,H,Dv] -> x [C,H,Dv]。

    对应 kernel 的 `solve_tril`：δ_t = β_t(v_t - p0_t) - Σ_{j<t} β_t·M[t,j]·δ_j。
    """
    steps = rhs.shape[0]
    x = torch.empty_like(rhs)
    for t in range(steps):
        acc = rhs[t].clone()
        if t:
            acc = acc - torch.einsum("hj,jhd->hd", L[:, t, :t], x[:t])
        x[t] = acc
    return x


def _chunk_step(S, qh, kh, vv, gg, bb, out_slice):
    """一个 chunk 的 WY 形式（与 kernel 的 A / solve_tril / O_inter / K_to_end 对应）。

    S [H,Dv,Dk] 片首状态（in: 读、out: 返回值），其余 [C,H,*]，返回片末状态。
    """
    G = torch.cumsum(gg, dim=0)                      # [C,H,Dk] 片内累积
    expG = torch.exp(G)
    Aout, Mkk = _matrices(qh, kh, G)

    # L[t,j,h] = beta_t * Mkk[t,j,h]（j<t），delta 解 (I + L) delta = rhs
    # （单位下三角前代，对应 kernel 的 solve_tril；不依赖 LAPACK）
    Lm = (bb[:, None] * Mkk).permute(2, 0, 1)        # [H,C,C]

    # 片首状态贡献（proj 用于 delta，另一路直接进输出）
    p0v = torch.einsum("hvd,thd->thv", S, expG * kh)
    p0o = torch.einsum("hvd,thd->thv", S, expG * qh)
    rhs = bb[:, :, None] * (vv - p0v)                # [C,H,Dv]
    delta = _forward_subst_unit_lower(Lm, rhs)       # [C,H,Dv]
    out_slice.copy_(p0o + torch.einsum("tjh,jhd->thd", Aout, delta))

    Kend = kh * torch.exp(G[-1][None] - G)           # [C,H,Dk] 片内 key 到片末的衰减
    S_new = S * torch.exp(G[-1])[:, None, :] + torch.einsum("jhd,jhk->hdk", delta, Kend)
    return S_new

# --------------------------------------------------------------------------
# #04 recurrent decode
# --------------------------------------------------------------------------


def recurrent_decode_ref(q, k, v, raw_g, beta_logits, a_log, g_bias, state,
                         state_indices, query_start_loc, num_accepted_tokens=None,
                         lower_bound: float = -5.0, scale: float | None = None,
                         use_qk_l2norm: bool = True, skip_slot_ids=(0,), out=None):
    """`torch.dlc_kda_recurrent_decode` 的参考实现（state 布局 [S,H,Dv,Dk]）。

    q/k/raw_g [1,N,H,Dk] bf16、v [1,N,H,Dv] bf16、beta_logits [1,N,H] bf16、
    a_log [H] fp32、g_bias [H,Dk] fp32、state [S,H,Dv,Dk] fp32（**原地更新**）、
    state_indices [B] 或 [B,next_n]（只用第 0 列）、query_start_loc [B+1] int32。

    `num_accepted_tokens` [B]（可 None）：与 kernel 现行语义一致 —— 输出仍按整窗
    展开计算，但写回的 state 是「第 accepted 个 token 之后」的快照：
    `accepted == 0` 写回**进入本步之前**的 state（回滚，kernel 里显式
    `copy_state(state, snapshot)`），`accepted >= 窗口长度` 写回窗口末状态。
    plain decode 不传该参数。

    `skip_slot_ids=(0,)` 复刻当前 kernel 行为：slot==0 → out 全 0、state 不动
    （PAD_SLOT_ID）。要纯语义（不跳过任何槽）请传 `()`。

    返回 (out [1,N,H,Dv] fp32, state)。
    """
    heads, dk = int(q.shape[2]), int(q.shape[3])
    dv, n = int(v.shape[3]), int(q.shape[1])
    if scale is None:
        scale = dk ** -0.5
    _check_state(state, heads, dv, dk)
    qh = (kda_l2norm(q) * scale if use_qk_l2norm else q.float() * scale)[0]  # -> [N,H,Dk]
    kh = (kda_l2norm(k) if use_qk_l2norm else k.float())[0]
    vv_all = v.float()[0]                                     # [N,H,Dv]
    gate = kda_gate(raw_g, a_log, g_bias, lower_bound)         # [N,H,Dk]
    betas = kda_beta(beta_logits)                             # [N,H]
    qsl = [int(x) for x in query_start_loc.reshape(-1).tolist()]
    idx = (state_indices.reshape(state_indices.shape[0], -1)
           if state_indices.dim() == 2 else state_indices.reshape(-1, 1))
    out_f = torch.zeros(1, n, heads, dv, dtype=torch.float32, device=q.device)
    for b in range(len(qsl) - 1):
        start, end = qsl[b], qsl[b + 1]
        if end <= start:
            continue
        slot = int(idx[b, 0])
        if slot in skip_slot_ids:
            continue
        window = end - start
        accepted = int(num_accepted_tokens[b]) if num_accepted_tokens is not None else window
        S = state[slot].float()                                   # [H,Dv,Dk]
        S_init = S.clone()                                        # accepted == 0 的回滚快照
        committed = None
        for t in range(start, end):
            S = S * torch.exp(gate[t])[:, None, :]                # 沿 Dk 列衰减
            proj = torch.einsum("hvd,hd->hv", S, kh[t])
            delta = (vv_all[t] - proj) * betas[t][:, None]
            S = S + kh[t][:, None, :] * delta[:, :, None]
            out_f[0, t] = torch.einsum("hvd,hd->hv", S, qh[t])
            if committed is None and 0 < accepted < window and t - start + 1 == accepted:
                committed = S.clone()
        if accepted == 0:
            # 一个 draft token 都没被接受：state 回滚到进入本步之前。
            committed = S_init
        state[slot].copy_((committed if committed is not None else S).to(state.dtype))
    if out is not None:
        out.copy_(out_f.to(out.dtype))
        return out, state
    return out_f, state


# --------------------------------------------------------------------------
# #03 chunk prefill
# --------------------------------------------------------------------------


def chunk_prefill_ref(q, k, v, raw_g, beta_logits, a_log, g_bias, initial_state,
                      query_start_loc, lower_bound: float = -5.0,
                      query_scale: float | None = None, chunk_size: int = 64,
                      use_qk_l2norm: bool = True, out_dtype=None):
    """`torch.dlc_kda_chunk_prefill` 的参考实现（state 布局 [B,H,Dv,Dk]）。

    输入与 recurrent_decode_ref 同构，只是 state 是「每序列一个片首状态」
    [B,H,Dv,Dk]；`query_start_loc` [B+1]。返回 (out [1,N,H,Dv], final_state
    [B,H,Dv,Dk])，均为 fp32（与算子返回的 bf16 比对时请 `.float()`）。

    `chunk_size` 与 kernel 的入参同义；算法对任意 chunk_size（含不能整除的长尾）
    都必须与逐 token 递推一致 —— `--selftest` 就在做这件事。

    注：两个算子 `g_bias` 签名不一致（#03 是 [Dk,H]，wrapper 内部 permute 成 [H,Dk]），
    这里统一收 [H,Dk]，调用 #03 时传 `g_bias.t().contiguous()`。
    """
    heads, dk = int(q.shape[2]), int(q.shape[3])
    dv, n = int(v.shape[3]), int(q.shape[1])
    if query_scale is None:
        query_scale = dk ** -0.5
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正")
    _check_state(initial_state, heads, dv, dk)
    qh = (kda_l2norm(q) * query_scale if use_qk_l2norm else q.float() * query_scale)[0]
    kh = (kda_l2norm(k) if use_qk_l2norm else k.float())[0]
    vv_all = v.float()[0]
    gate = kda_gate(raw_g, a_log, g_bias, lower_bound)
    betas = kda_beta(beta_logits)
    qsl = [int(x) for x in query_start_loc.reshape(-1).tolist()]
    batch = min(initial_state.shape[0], len(qsl) - 1)
    out = torch.zeros(1, n, heads, dv, dtype=torch.float32, device=q.device)
    final_state = initial_state.float().clone()
    for b in range(batch):
        start, end = qsl[b], qsl[b + 1]
        S = final_state[b].clone()
        for cs in range(start, end, chunk_size):
            ce = min(cs + chunk_size, end)
            S = _chunk_step(S, qh[cs:ce], kh[cs:ce], vv_all[cs:ce],
                            gate[cs:ce], betas[cs:ce], out[0, cs:ce])
        final_state[b] = S
    if out_dtype is not None:
        return out.to(out_dtype), final_state
    return out, final_state

# --------------------------------------------------------------------------
# 自检 / 对拍
# --------------------------------------------------------------------------


def _rand_inputs(lens, heads=2, dk=16, dv=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = sum(lens)
    r = lambda *s: torch.randn(*s, generator=g)
    return dict(
        lens=list(lens),
        q=r(1, n, heads, dk), k=r(1, n, heads, dk), v=r(1, n, heads, dv),
        raw_g=r(1, n, heads, dk) * 0.5, beta_logits=r(1, n, heads),
        a_log=r(heads) * 0.5, g_bias=r(heads, dk) * 0.2,
        state=r(len(lens), heads, dv, dk) * 0.05,
        query_start_loc=torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()),
                                     dtype=torch.int32),
    )


def selftest(chunk_sizes=(1, 2, 8, 64), heads=2, dk=16, dv=16, tol=5e-5):
    """chunk prefill == 逐 token 递推（同一个 state 布局下必须成立）。"""
    print(f"=== selftest: chunk_prefill_ref vs recurrent_decode_ref "
          f"(H={heads}, Dk={dk}, Dv={dv}) ===", flush=True)
    cases = [[1], [2], [4], [7], [1, 1, 1, 1], [5, 2, 1, 3], [16]]
    bad = 0
    for lens in cases:
        for cs in chunk_sizes:
            d = _rand_inputs(lens, heads, dk, dv, seed=1000 * len(lens) + cs)
            idx = torch.arange(len(lens), dtype=torch.int32)
            out_seq, st_seq = recurrent_decode_ref(
                d["q"], d["k"], d["v"], d["raw_g"], d["beta_logits"], d["a_log"], d["g_bias"],
                d["state"].clone(), idx, d["query_start_loc"], skip_slot_ids=())
            out_ch, st_ch = chunk_prefill_ref(
                d["q"], d["k"], d["v"], d["raw_g"], d["beta_logits"], d["a_log"], d["g_bias"],
                d["state"].clone(), d["query_start_loc"], chunk_size=cs)
            e_o = (out_ch - out_seq).abs().max().item()
            e_s = (st_ch.float() - st_seq.float()).abs().max().item()
            flag = "OK " if e_o < tol and e_s < tol else "FAIL"
            bad += flag == "FAIL"
            print(f"  [{flag}] lens={str(lens):12s} chunk={cs:3d}  out_err={e_o:.2e}  state_err={e_s:.2e}",
                  flush=True)
    print(f"=== selftest {'通过' if bad == 0 else f'失败（{bad} 例）'}："
          f"chunk 版与逐 token 递推等价（fp32 容差 {tol:g}）===", flush=True)
    sem_ok = test_decode_semantics(heads, dk, dv)
    return bad == 0 and sem_ok


def test_decode_semantics(heads=2, dk=16, dv=16, seed=7):
    """decode 的两个行为约定：#04 的 PAD slot（slot 0 跳过）与 num_accepted_tokens 快照。"""
    print("=== selftest: recurrent_decode 语义（PAD slot / num_accepted_tokens）===", flush=True)
    d = _rand_inputs([4], heads, dk, dv, seed=seed)
    pool = torch.randn(2, heads, dv, dk) * 0.05
    pool[0] = 3.0                                   # slot 0 是 PAD，初值故意非零
    out, st = recurrent_decode_ref(
        d["q"], d["k"], d["v"], d["raw_g"], d["beta_logits"], d["a_log"], d["g_bias"],
        pool.clone(), torch.tensor([0], dtype=torch.int32), d["query_start_loc"])
    pad_ok = bool((out == 0).all()) and bool((st[0] == 3.0).all())
    print(f"  [{'OK ' if pad_ok else 'FAIL'}] slot=0(PAD) -> out 全 0={bool((out == 0).all())}、"
          f"state 未改动={bool((st[0] == 3.0).all())}", flush=True)

    idx = torch.tensor([1], dtype=torch.int32)
    qsl4 = d["query_start_loc"]
    _, st_full = recurrent_decode_ref(
        d["q"], d["k"], d["v"], d["raw_g"], d["beta_logits"], d["a_log"], d["g_bias"],
        pool.clone(), idx, qsl4, skip_slot_ids=())
    _, st_acc = recurrent_decode_ref(
        d["q"], d["k"], d["v"], d["raw_g"], d["beta_logits"], d["a_log"], d["g_bias"],
        pool.clone(), idx, qsl4, num_accepted_tokens=torch.tensor([2]), skip_slot_ids=())
    qsl2 = torch.tensor([0, 2], dtype=torch.int32)
    _, st2 = recurrent_decode_ref(
        d["q"][:, :2].contiguous(), d["k"][:, :2].contiguous(), d["v"][:, :2].contiguous(),
        d["raw_g"][:, :2].contiguous(), d["beta_logits"][:, :2].contiguous(),
        d["a_log"], d["g_bias"], pool.clone(), idx, qsl2, skip_slot_ids=())
    e2 = (st_acc[1] - st2[1]).abs().max().item()
    e4 = (st_acc[1] - st_full[1]).abs().max().item()
    ok = e2 < 1e-6 and e4 > 1e-3
    print(f"  [{'OK ' if ok else 'FAIL'}] acc=2/窗口=4 -> 与「2 个 token 后的快照」差 {e2:.2e}、"
          f"与「整窗」差 {e4:.2e}（应分别接近 0 / 明显非 0）", flush=True)
    return pad_ok and ok


def _to(obj, device, dtype=None):
    t = obj.to(device)
    return t.to(dtype) if dtype is not None else t


def compare_on_device(dev="dlc:0", chunk_size=64, force_chunk=False,
                      heads=8, dk=128, dv=128, lens=(4,), seed=0):
    """把真机算子与参考实现对拍（需要 DLC 设备；#03 未修复前会卡死设备，默认跳过）。"""
    LB, SCALE = -5.0, dk ** -0.5
    d = _rand_inputs(lens, heads, dk, dv, seed=seed)
    dev_t = {name: _to(t, dev) for name, t in d.items() if name not in ("lens",)}
    bf16 = {name: dev_t[name].to(torch.bfloat16) for name in
            ("q", "k", "v", "raw_g", "beta_logits")}
    print(f"=== 对拍 dlc_kda_recurrent_decode @ {dev}（lens={lens}）===", flush=True)
    st = dev_t["state"].clone()
    out = torch.full((1, sum(lens), heads, dv), 7.0, dtype=torch.bfloat16, device=dev)
    torch.dlc_kda_recurrent_decode(
        bf16["q"], bf16["k"], bf16["v"], bf16["raw_g"], bf16["beta_logits"],
        dev_t["a_log"].float(), dev_t["g_bias"].float(), st,
        torch.arange(len(lens), dtype=torch.int32, device=dev), dev_t["query_start_loc"],
        None, lower_bound=LB, scale=SCALE, out=out)
    torch.dlc.synchronize()
    r_out, r_st = recurrent_decode_ref(
        bf16["q"].cpu(), bf16["k"].cpu(), bf16["v"].cpu(), bf16["raw_g"].cpu(),
        bf16["beta_logits"].cpu(), dev_t["a_log"].cpu(), dev_t["g_bias"].cpu(),
        dev_t["state"].cpu().clone(), torch.arange(len(lens), dtype=torch.int32),
        dev_t["query_start_loc"].cpu(), skip_slot_ids=())
    print(f"  out_init=7.0 被覆写={not bool((out == 7.0).any())}  "
          f"out_err={(out.cpu().float() - r_out).abs().max().item():.3e}  "
          f"state_err={(st.cpu().float() - r_st).abs().max().item():.3e}", flush=True)

    if not force_chunk:
        print("=== 跳过 dlc_kda_chunk_prefill 对拍（未修复前必抛异常并卡死设备；"
              "确认要用 `--force-chunk`）===", flush=True)
        return
    print(f"=== 对拍 dlc_kda_chunk_prefill @ {dev}（chunk_size={chunk_size}）===", flush=True)
    st0 = dev_t["state"].clone()
    out_c, fin_c = torch.dlc_kda_chunk_prefill(
        bf16["q"], bf16["k"], bf16["v"], bf16["raw_g"], bf16["beta_logits"],
        dev_t["a_log"].float(), dev_t["g_bias"].float().t().contiguous(),
        st0, dev_t["query_start_loc"], float(LB), float(SCALE), int(chunk_size))
    torch.dlc.synchronize()
    r_out, r_fin = chunk_prefill_ref(
        bf16["q"].cpu(), bf16["k"].cpu(), bf16["v"].cpu(), bf16["raw_g"].cpu(),
        bf16["beta_logits"].cpu(), dev_t["a_log"].cpu(), dev_t["g_bias"].cpu(),
        dev_t["state"].cpu().clone(), dev_t["query_start_loc"].cpu(), chunk_size=chunk_size)
    print(f"  out_err={(out_c.cpu().float() - r_out).abs().max().item():.3e}  "
          f"final_state_err={(fin_c.cpu().float() - r_fin).abs().max().item():.3e}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="GLM5 KDA 参考实现（state 布局 [..,Dv,Dk]）")
    p.add_argument("--selftest", action="store_true", help="跑 CPU 自检（默认动作）")
    p.add_argument("--compare-device", default=None, help="与真机算子对拍，如 dlc:6")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--force-chunk", action="store_true",
                   help="#03 未修复前不要加：会卡死设备")
    p.add_argument("--lens", default="4", help="逗号分隔的序列长度，如 4 或 3,1,2")
    p.add_argument("--heads", type=int, default=2, help="自检用 head 数")
    p.add_argument("--dk", type=int, default=16, help="自检用 Dk")
    p.add_argument("--dv", type=int, default=16, help="自检用 Dv")
    a = p.parse_args(argv)
    lens = tuple(int(x) for x in a.lens.split(","))
    if a.compare_device:
        compare_on_device(a.compare_device, a.chunk_size, a.force_chunk,
                          heads=a.heads, dk=a.dk, dv=a.dv, lens=lens)
    else:
        return 0 if selftest(heads=a.heads, dk=a.dk, dv=a.dv) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
