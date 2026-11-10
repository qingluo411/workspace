"""#03 新算子 dlc_kda_chunk_prefill 复现脚本（2026-09-22，2026-09-22 晚改为两段式）。

GLM5 几何：H=8(TP8), Dk=Dv=128；q/k/v/raw_g [1,T,H,128] bf16，beta_logits [1,T,H] bf16，
a_log [H] fp32，g_bias [D,H] fp32（wrapper 内 permute 成 [H,D]），initial_state fp32，
（state 布局 2026-09-22 已定稿为 [B,H,Dv,Dk]；本脚本用的 [B,H,Dk,Dv] 是交付版旧布局，仅用于复现异常）
query_start_loc [B+1] int32，lower_bound=-5.0，query_scale=128**-0.5，chunk_size=64。

两段式用法（**默认不碰设备**）：
  A) 静态复现（无需 DLC 设备，证明 host wrapper 必抛异常）：
       python probe_new_kda_chunk_prefill.py --static
  B) 真机复现（**会把设备卡死，必须在可复位的卡上跑**）：
       python probe_new_kda_chunk_prefill.py dlc:5 --wedge-device

背景：host wrapper 在 `kernel.launch()` 之后才对 5 维张量做 4 阶 `permute`，所以调用方
**即使捕获了异常，设备也已经被 launch 出去的 kernel 弄坏**（实测必须复位才能再用）。
"""
import sys
import time
import torch

WEDGE = "--wedge-device" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
dev = args[0] if args else "dlc:0"
torch.manual_seed(0)
H, D = 8, 128
LB, SCALE = -5.0, 128 ** -0.5


def static_repro():
    print("=== A) 静态复现（不开 DLC 设备）===", flush=True)
    print("FusedOps.cpp:780-845 的真实调用序列（简化）：", flush=True)
    v = torch.zeros(1, 7, H, D)                       # [1,N,H,D]
    v_dlc = v.permute({3, 2, 1, 0}).unsqueeze(-1) if False else v.permute(3, 2, 1, 0).unsqueeze(-1)
    print(f"    v               : {tuple(v.shape)}", flush=True)
    print(f"    v_dlc           : {tuple(v_dlc.shape)}  ← .permute({{3,2,1,0}}).unsqueeze(-1) 后已是 5 维", flush=True)
    out_dlc = torch.empty_like(v_dlc)
    print(f"    out_dlc         : {tuple(out_dlc.shape)}  ← at::empty_like(v_dlc)，同样 5 维", flush=True)
    try:
        out_dlc.permute(3, 2, 1, 0)
    except Exception as e:
        print(f"    out_dlc.permute({{3,2,1,0}}) -> {type(e).__name__}: {str(e)[:200]}", flush=True)
    print("    → 与真机报错同一条（host wrapper 返回处 5 维张量用 4 阶 permute，必现、与输入数据无关）", flush=True)


def make(lens, chunk):
    """a["beta_logits"] / a["initial_state"] 是调用方张量的副本，用于确认是否被就地改写"""
    B, N = len(lens), sum(lens)
    q = (torch.randn(1, N, H, D, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    k = (torch.randn(1, N, H, D, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    v = (torch.randn(1, N, H, D, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    raw_g = (torch.randn(1, N, H, D, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    beta = (torch.randn(1, N, H, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    a_log = (torch.randn(H, dtype=torch.float32, device=dev) * 0.5).contiguous()
    g_bias = (torch.randn(D, H, dtype=torch.float32, device=dev) * 0.2).contiguous()
    st0 = torch.randn(B, H, D, D, dtype=torch.float32, device=dev) * 0.05
    qsl = torch.tensor([0] + torch.cumsum(torch.tensor(lens), 0).tolist(),
                       dtype=torch.int32, device=dev)
    return dict(q=q, k=k, v=v, raw_g=raw_g, beta_logits=beta, a_log=a_log,
                g_bias=g_bias, initial_state=st0, query_start_loc=qsl), st0.clone(), beta.clone()


def case(name, lens, chunk=64):
    a, st0, beta0 = make(lens, chunk)
    t = time.time()
    try:
        out, fin = torch.dlc_kda_chunk_prefill(
            a["q"], a["k"], a["v"], a["raw_g"], a["beta_logits"], a["a_log"], a["g_bias"],
            a["initial_state"], a["query_start_loc"], float(LB), float(SCALE), int(chunk))
        torch.dlc.synchronize()
        print(f"[{name}] 成功 out={tuple(out.shape)} fin={tuple(fin.shape)} t={(time.time()-t)*1e3:.2f}ms",
              flush=True)
    except Exception as e:
        print(f"[{name}] lens={lens} chunk={chunk} -> {type(e).__name__}: {str(e)[:170]}", flush=True)
        print("    异常抛出时 kernel.launch() 已发生 → 该设备已被卡死：**不要再读回任何 DLC 张量",
              flush=True)
        print("    （任何 .item()/synchronize() 都会永久挂住）**，请直接换卡；本脚本到这里即退出。",
              flush=True)
        print("    `beta_logits` 被就地改写见 kernel `dlc_kda_chunk_prefill.cpp:416-418`"
              "（sigmoid_bf16 后 Vmem2HBM 写回调用方地址），已由代码审阅确认。", flush=True)
        sys.exit(3)


print(f"dev={dev} wedge_mode={WEDGE} H={H} D={D}", flush=True)
static_repro()
if not WEDGE:
    print("=== B) 真机复现已跳过（默认安全模式）===", flush=True)
    print("    要复现「必抛异常 + 卡死设备 + beta_logits 被就地改写」，请在**可复位**的卡上执行：", flush=True)
    print("    python probe_new_kda_chunk_prefill.py dlc:5 --wedge-device", flush=True)
    sys.exit(0)

print(f"=== B) 真机复现（会卡死设备）dev={dev} ===", flush=True)
case("b1-64", [64])
case("b1-200", [200])
case("b3-var", [64, 130, 7])
case("b1-200-c128", [200], chunk=128)
case("b1-4", [4])
