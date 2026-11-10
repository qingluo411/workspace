"""#02 新算子 dlc_kda_causal_conv_update 实测。

对比：A) fp32 reference（向量化） B) 新算子 C) 已有 torch._causal_conv1d_update
kernel 约定（kda_causal_conv_update.c）：
  y[t]=SiLU(s[-3]*w0+s[-2]*w1+s[-1]*w2+x[t]*w3)；state 行序 旧->新，新值追加到末行；
  只在处理完 num_accepted_tokens 个 token 后提交快照；slot 0 = null（输出 0、state 不动）；
  slot<0 或越界 -> 只写 0（host 用 at::zeros_like 预置 out）、state 不动。

2026-09-24 算子组修复后的 ABI（本脚本已适配）：
  - conv_state 逻辑形状 [S_slot, state_len, D]（SD，D 连续），kernel 直接 in-place 写回；
  - weight 是已转置的 [4, D] fp32；
  - 返回值只有 out（不再返回 committed_state）。
"""
import time
import torch

import sys

dev = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
torch.manual_seed(0)
W, HIST = 4, 3


def ref_update(base, slot, xs, w2d, accepted):
    """base:[S,L,D]; xs:[T,D] -> ([T,D] bf16, snap[L,D] bf16)

    2026-09-22 晚修正：初版只滚动最后 HIST 行、把前 L-HIST 行原样留在快照里，L>3 时与算子
    （以及 `probe_new_kda_conv_update_window.py` 的参考）不一致，误报 L=6 失败。正确语义是
    **整窗滚动 L 行**，滑窗读最后 3 行，第 accepted 个 token 后整窗快照。
    """
    L = base.shape[1]
    h = base[slot].float().clone()
    outs, snap = [], base[slot].clone()
    for t in range(xs.shape[0]):
        y = torch.nn.functional.silu(
            h[L - 3] * w2d[:, 0] + h[L - 2] * w2d[:, 1] + h[L - 1] * w2d[:, 2]
            + xs[t].float() * w2d[:, 3])
        outs.append(y.to(torch.bfloat16))
        h = torch.cat([h[1:], xs[t].float()[None]], 0)
        if t == accepted - 1:
            snap = h.to(torch.bfloat16)
    return torch.stack(outs) if outs else xs.new_empty((0, xs.shape[1])), snap


def case(name, D, lens, slots=4, idxs=None, acc=None, state_len=HIST, qkv3d=False):
    B, N = len(lens), sum(lens)
    idxs = idxs if idxs is not None else list(range(1, B + 1))
    xs = (torch.randn(N, D, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    w2d = (torch.randn(D, W, dtype=torch.float32, device=dev) * 0.3).contiguous()
    w4d = w2d.t().contiguous()                       # 新 ABI: [4, D] fp32
    if qkv3d:
        xs = xs.view(B, lens[0], D)
    base = torch.randn(slots, state_len, D, dtype=torch.bfloat16, device=dev) * 0.5
    qsl = torch.tensor([0] + torch.cumsum(torch.tensor(lens), 0).tolist(),
                       dtype=torch.int32, device=dev)
    sidx = torch.tensor(idxs, dtype=torch.int32, device=dev)
    acc_t = None if acc is None else torch.tensor(acc, dtype=torch.int32, device=dev)

    ref_out = torch.empty(N, D, dtype=torch.bfloat16, device=dev)
    ref_buf = base.clone()
    for b in range(B):
        s, e = int(qsl[b]), int(qsl[b + 1])
        a = (e - s) if acc is None else acc[b]
        if e <= s:
            continue
        seg = xs[b] if qkv3d else xs[s:e]      # 3D [B,T,D]：按请求取 [T,D]
        o, snap = ref_update(base, idxs[b], seg, w2d, a)
        ref_out[s:e] = o
        if idxs[b] > 0:
            ref_buf[idxs[b]] = snap

    buf = base.clone()
    t = time.time()
    out = torch.dlc_kda_causal_conv_update(
        xs, w4d, buf, sidx,
        query_start_loc=qsl, num_accepted_tokens=acc_t,
        max_query_len=max(lens), activation="silu", null_state_id=0)
    torch.dlc.synchronize()
    dt = time.time() - t
    if qkv3d:
        out = out.reshape(N, D)                # 3D 返回 [B,T,D] -> 与参考同一 [N,D] 行序
    rows = torch.cat([torch.arange(int(qsl[i]), int(qsl[i + 1])) for i in range(B)
                      if idxs[i] > 0]) if any(i > 0 for i in idxs) else torch.zeros(0, dtype=torch.long)
    oerr = (out[rows].float() - ref_out[rows].float()).abs().max().item() if rows.numel() else 0.0
    serr = (buf.float() - ref_buf.float()).abs().max().item()
    print(f"[{name}] D={D} lens={lens} idx={idxs} acc={acc} L={state_len}", flush=True)
    print(f"    out_err={oerr:.3e} state_err={serr:.3e} t={dt:.4f}s",
          flush=True)
    if 0 in idxs:
        r = idxs.index(0)
        seg = out[int(qsl[r]):int(qsl[r + 1])]
        print(f"    null slot: out_all_zero={bool((seg == 0).all())} "
              f"state_unchanged={bool(torch.equal(buf[0], base[0]))}", flush=True)
    if -1 in idxs:
        r = idxs.index(-1)
        seg = out[int(qsl[r]):int(qsl[r + 1])]
        print(f"    idx=-1: out_all_zero={bool((seg == 0).all())} "
              f"out_maxabs={seg.abs().max().item():.3e} "
              f"state_unchanged={bool(torch.equal(buf[-1], base[-1]))}", flush=True)
    return out, buf


print("=== 1) plain decode: B=4 单 token, null slot 0 ===", flush=True)
case("decode-b4", 3072, [1, 1, 1, 1], idxs=[1, 2, 0, 3])
print("=== 2) spec decode: T=4, accepted=[0,2,4,1] ===", flush=True)
case("spec-T4", 3072, [4, 4, 4, 4], idxs=[1, 2, 3, 1], acc=[0, 2, 4, 1])
print("=== 3) spec: T=6, 部分接受 ===", flush=True)
case("spec-T6", 3072, [6, 6], idxs=[1, 2], acc=[3, 0])
print("=== 4) 3D qkv [B,T,D] (T=2) ===", flush=True)
case("3d-b2", 3072, [2, 2], idxs=[1, 2], acc=[1, 2], qkv3d=True)
print("=== 5) idx=-1（pad slot）===", flush=True)
case("neg-idx", 3072, [1, 1], idxs=[-1, 2])
print("=== 6) D=320（非 256 倍数）===", flush=True)
try:
    case("D320", 320, [1], idxs=[1])
except Exception as e:
    print(f"[D320] -> {type(e).__name__}: {str(e)[:180]}", flush=True)
print("=== 7) state_len=6 ===", flush=True)
case("statelen6", 3072, [3], slots=2, idxs=[1], acc=[2], state_len=6)
print("=== 8) 与已有 torch._causal_conv1d_update 对比（plain decode B=4）===", flush=True)
D, Bn = 3072, 4
torch.manual_seed(7)
x = (torch.randn(Bn, D, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
w = (torch.randn(D, W, dtype=torch.float32, device=dev) * 0.3)
bias = torch.zeros(D, dtype=torch.bfloat16, device=dev)
base0 = torch.randn(5, HIST, D, dtype=torch.bfloat16, device=dev) * 0.5
idx = torch.tensor([1, 2, 0, 3], dtype=torch.int32, device=dev)

b_old = base0.clone()
o_old = torch._causal_conv1d_update(
    x, w.to(torch.bfloat16), bias, b_old.transpose(-1, -2), idx,
    None, None, -1, None, None, True, True)
torch.dlc.synchronize()

b_new = base0.clone()
o_new = torch.dlc_kda_causal_conv_update(
    x, w.t().contiguous(), b_new, idx,
    query_start_loc=torch.arange(Bn + 1, dtype=torch.int32, device=dev),
    max_query_len=1, activation="silu", null_state_id=0)
torch.dlc.synchronize()

real = torch.tensor([0, 1, 3], device=dev)
# b_old / b_new 都是 SD([S,L,D])（旧算子写入的是 b_old.transpose(-1,-2) 视图，同一块存储）
b_new_sd = b_new
b_old_sd = b_old
print(f"    real row: old vs new out_err={(o_old.float()-o_new.float())[real].abs().max().item():.3e}"
      f"  state_err={(b_old_sd.float()-b_new_sd.float())[[1,2,3]].abs().max().item():.3e}", flush=True)
for nm, o, b in (("old", o_old, b_old_sd), ("new", o_new, b_new_sd)):
    print(f"    {nm}: null-row0 out_all_zero={bool((o[2] == 0).all())}"
          f"  state[0]_unchanged={bool(torch.equal(b[0], base0[0]))}", flush=True)
