"""#01 新算子 dlc_kda_causal_conv_prefill 实测（2026-09-22）。

输入形状按 GLM5 模型：D=3072=(64*128*3)/TP8, kernel=4, state=[S_slot,3,D]（SD 布局，D 连续）。
reference：向量化的 fp32 depthwise causal conv + silu（等价于 patch_kda._native_conv_sequence）。
kernel 约定（kda_causal_conv_prefill.c）：state 行序 旧->新；取「最后 3 行」为 history；
slot 0 = null（输出全 0、state 不写）；处理完把「最新 state_len 个原始投影」写回 state。

2026-09-24 算子组修复后的 ABI（本脚本已适配）：
  - conv_state 逻辑形状 [S_slot, state_len, D]（SD，D 连续）；
  - weight 是已转置的 [4, D] fp32；
  - has_initial_state 是 int32（不再接受 bool）。

用法：`python probe_new_kda_conv_prefill.py <dev> [case_ids]`，例如
`python ... dlc:0 2` 只跑第 2 例（N=4096，修复前会卡死设备）。
便于「一例一张干净卡」地逐例验证（每个用例内部都已 `dlc.synchronize()`）。
"""
import sys
import time
import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None


def want(tag):
    # 默认**不跑**第 2 例（N=4096，修复前会卡死设备，要验证修复请显式写 case id）
    # 和第 9 例（异常路径：kernel 断言会 poison 设备上下文，进程随后 abort，必须单独占用一张卡）。
    if ONLY is not None:
        return tag in ONLY
    return tag not in ("2", "9")

torch.manual_seed(0)
W, HIST = 4, 3


def ref_vec(x_dl, w_dl, hist_dl):
    """x_dl:[D,N] bf16, w_dl:[D,4] fp32, hist_dl:[D,3] bf16 -> ([D,N], [D,3])

    注意：全部在 **CPU** 上、按 CH 个 token 分块计算。原因：DLC 设备端对超大张量
    （N>8192 的 [D,3+N] fp32）做逐元素 mul/silu 会给出错误结果（实测 max_err≈2.0），
    会把「kernel 正确」误判成「kernel 错」。CPU 分块参考与逐 token 循环参考逐位一致。
    """
    N = x_dl.shape[1]
    CH = 512
    outs = []
    for s in range(0, N, CH):
        e = min(N, s + CH)
        hist = hist_dl if s == 0 else x_dl[:, s - HIST : s]
        n = e - s
        xs = torch.cat([hist.float(), x_dl[:, s:e].float()], 1)   # [D, 3+n]
        y = sum(w_dl[:, k : k + 1] * xs[:, k : k + n] for k in range(W))
        outs.append(torch.nn.functional.silu(y).to(x_dl.dtype))
    out = torch.cat(outs, 1) if outs else x_dl.new_empty((x_dl.shape[0], 0))
    if N >= HIST:
        newh = x_dl[:, -HIST:].to(x_dl.dtype)
    else:
        newh = torch.cat([hist_dl[:, -(HIST - N):], x_dl], 1)
    return out, newh


def case(name, D, lens, slots=4, idxs=None, hinit_flags=None, state_len=HIST, seed=0,
         timeit=True, ref_ok=True):
    torch.manual_seed(seed)
    B, N = len(lens), sum(lens)
    idxs = idxs if idxs is not None else list(range(1, B + 1))
    hinit_flags = hinit_flags if hinit_flags is not None else [True] * B
    qkv = (torch.randn(N, D, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    w2d = (torch.randn(D, W, dtype=torch.float32, device=dev) * 0.3).contiguous()
    w4d = w2d.t().contiguous()                       # 新 ABI: [4, D] fp32
    base = torch.randn(slots, state_len, D, dtype=torch.bfloat16, device=dev) * 0.5
    qsl = torch.tensor([0] + torch.cumsum(torch.tensor(lens), 0).tolist(),
                       dtype=torch.int32, device=dev)
    sidx = torch.tensor(idxs, dtype=torch.int32, device=dev)
    hinit = torch.tensor(hinit_flags, dtype=torch.bool, device=dev).to(torch.int32)

    # ---- reference：全部在 CPU 上算（见 ref_vec 注释）----
    qkv_c = qkv.cpu()                                  # [N,D] bf16
    w2d_c = w2d.cpu()                                  # [D,4] fp32
    ref_out = torch.empty(N, D, dtype=torch.bfloat16)
    ref_buf = base.cpu().clone()                       # [S,L,D] bf16 (CPU)
    for b in range(B if ref_ok else 0):
        s, e = int(qsl[b]), int(qsl[b + 1])
        if e <= s:
            continue
        slot = idxs[b]
        if slot == 0:
            ref_out[s:e] = 0
            continue
        hist = (ref_buf[slot].transpose(0, 1)[:, -HIST:].contiguous() if hinit_flags[b]
                else torch.zeros(D, HIST, dtype=torch.bfloat16))
        o, h = ref_vec(qkv_c[s:e].transpose(0, 1).contiguous(), w2d_c, hist)
        ref_out[s:e] = o.transpose(0, 1)
        if e - s < state_len:
            ref_buf[slot] = torch.cat(
                [hist[:, -(state_len - (e - s)):], qkv_c[s:e].transpose(0, 1)], 1).transpose(0, 1)
        else:
            ref_buf[slot] = qkv_c[e - state_len:e]
    torch.dlc.synchronize()

    buf = base.clone()                                 # [S,state_len,D] SD（连续）
    t = time.time()
    out = torch.dlc_kda_causal_conv_prefill(
        qkv, w4d, buf, qsl, sidx, hinit, activation="silu", null_state_id=0)
    torch.dlc.synchronize()
    dt = time.time() - t
    oerr = (out.cpu().float() - ref_out.float()).abs().max().item() if ref_ok else float("nan")
    serr = (buf.cpu().float() - ref_buf.float()).abs().max().item() if ref_ok else float("nan")
    print(f"[{name}] D={D} lens={lens} idx={idxs} hinit={hinit_flags} L={state_len} "
          f"slots={slots}", flush=True)
    print(f"    out_err={oerr:.3e}  state_err={serr:.3e}  t={dt:.3f}s", flush=True)
    if 0 in idxs:
        rows = [i for i, x in enumerate(idxs) if x == 0]
        seg = torch.cat([out[int(qsl[r]):int(qsl[r + 1])] for r in rows])
        print(f"    null slot: out_all_zero={bool((seg == 0).all())} "
              f"state_unchanged={bool(torch.equal(buf[0], base[0]))}", flush=True)
    return out, buf


print(f"dev={dev} only={sorted(ONLY) if ONLY else 'all'}", flush=True)

if want("1"):
    print("=== 1) GLM5 真实几何：变长多序列 ===", flush=True)
    case("glm5-3seq", 3072, [16, 5, 33])
if want("2"):
    print("=== 2) 真实 prefill 长度 N=4096（修复前会卡死设备，本卡需可复位）===", flush=True)
    print("    （N=3072/4096 修复前会挂；跑完请用 probe_device_health.py 检查该卡）", flush=True)
    case("b1-4096", 3072, [4096], slots=2, idxs=[1])
if want("3"):
    print("=== 3) B=8 变长 ===", flush=True)
    case("b8-var", 3072, [128, 1, 512, 7, 256, 33, 1024, 3], slots=9)
if want("4"):
    print("=== 4) 短序列 (<state_len) ===", flush=True)
    case("short-seq", 3072, [1, 2], idxs=[1, 2], hinit_flags=[True, False])
if want("5"):
    print("=== 5) has_initial_state=False ===", flush=True)
    case("no-init", 3072, [9, 9], idxs=[1, 2], hinit_flags=[False, True])
if want("6"):
    print("=== 6) null slot 0 ===", flush=True)
    case("null-slot", 3072, [7, 7], idxs=[0, 2])
if want("7"):
    print("=== 7) 非 256 倍数 D（kernel 内部 ALIGN256 处理）===", flush=True)
    case("pad-D320", 320, [6, 6], idxs=[1, 2])
if want("8"):
    print("=== 8) state_len=6（num_spec>0 的宽度）===", flush=True)
    case("statelen6", 3072, [10], slots=2, idxs=[1], state_len=6)
if want("10"):
    print("=== 10) 真实 dump 几何：N=4096, state_len=6（算子组单测里的用例）===", flush=True)
    case("real-4096-l6", 3072, [4096], slots=2, idxs=[1], state_len=6)
if want("11"):
    print("=== 11) 更长 prefill：N=8192 ===", flush=True)
    case("b1-8192", 3072, [8192], slots=2, idxs=[1])
if want("12"):
    print("=== 12) 大 batch prefill：B=16, N=8192 ===", flush=True)
    case("b16-8192", 3072, [512] * 16, slots=17)
if want("13"):
    print("=== 13) 超长单序列：N=16384 ===", flush=True)
    case("b1-16384", 3072, [16384], slots=2, idxs=[1])
if want("14"):
    print("=== 14) 超长单序列：N=32768 ===", flush=True)
    case("b1-32768", 3072, [32768], slots=2, idxs=[1])
if want("9"):
    print("=== 9) 异常路径（**会把设备置为错误态并 abort，单独占一张卡跑**）===", flush=True)
    for nm, kw in [("dup-slot", dict(idxs=[2, 2])), ("neg-slot", dict(idxs=[-1, 2])),
                   ("oob-slot", dict(idxs=[9, 2]))]:
        try:
            case(nm, 3072, [4, 4], ref_ok=False, **kw)
        except Exception as e:
            print(f"[{nm}] -> {type(e).__name__}: {str(e)[:150]}", flush=True)
