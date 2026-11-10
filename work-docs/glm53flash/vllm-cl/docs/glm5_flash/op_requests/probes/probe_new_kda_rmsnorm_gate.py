"""#06 新算子 dlc_kda_rmsnorm_sigmoid_gate 实测（2026-09-22）。

GLM5：o_norm = FusedRMSNormGated(head_dim, activation="sigmoid")，
      x=core_attn_out [1,T,H,128] bf16, g=g2 同形, weight=[128], eps=1e-5。
语义候选：sigmoid 版 y=x*rsqrt(mean(x^2)+eps)*w*sigmoid(g)；swish 版 ...*g*sigmoid(g)。
"""
import sys
import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
torch.manual_seed(0)
H, D = 8, 128


def ref(x, g, w, eps, mode):
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    y = y * w.float()
    gf = g.float()
    y = y * (gf.sigmoid() if mode == "sigmoid" else gf * gf.sigmoid())
    return y.to(x.dtype)


def case(name, T, dims, wdtype=torch.bfloat16, eps=1e-5):
    shape = (1, T, H, D) if dims == 4 else (T, H, D)
    x = (torch.randn(*shape, dtype=torch.bfloat16, device=dev) * 0.8).contiguous()
    g = (torch.randn(*shape, dtype=torch.bfloat16, device=dev) * 2.0).contiguous()
    w = (torch.randn(D, dtype=wdtype, device=dev) * 0.2 + 1.0).contiguous()
    out = torch.dlc_kda_rmsnorm_sigmoid_gate(x, g, w, eps)
    torch.dlc.synchronize()
    for mode in ("sigmoid", "swish"):
        r = ref(x, g, w, eps, mode)
        print(f"[{name}] mode={mode:8s} err={(out.float()-r.float()).abs().max().item():.3e}"
              f"  |ref|max={r.abs().max().item():.3f}", flush=True)
    print(f"[{name}] out |max|={out.abs().max().item():.3f} out_all_zero={bool((out == 0).all())}",
          flush=True)
    # fp32 reference 版（不做 L2/L1 偏差）用于对照量级
    return out


print(f"dev={dev}", flush=True)
print("=== 1) 4D [1,T,H,128] T=64, weight bf16 ===", flush=True)
case("4d-T64", 64, 4)
print("=== 2) 3D [T,H,128] T=16 ===", flush=True)
case("3d-T16", 16, 3)
print("=== 3) weight fp32 ===", flush=True)
case("4d-T8-wf32", 8, 4, wdtype=torch.float32)
print("=== 4) T=1 边界 ===", flush=True)
case("T1", 1, 4)
# T=0 只在算子侧验证（空张量 reduction 会触发 DLC 设备卡死，与算子无关）
xe = torch.randn(1, 0, H, D, dtype=torch.bfloat16, device=dev)
ge = torch.randn(1, 0, H, D, dtype=torch.bfloat16, device=dev)
oe = torch.dlc_kda_rmsnorm_sigmoid_gate(xe, ge, torch.ones(D, dtype=torch.bfloat16, device=dev), 1e-5)
torch.dlc.synchronize()
print(f"[T0] ok 返回空张量 shape={tuple(oe.shape)} numel={oe.numel()}", flush=True)
print("=== 5) 异常：最后一维非 128 / 形状不匹配 ===", flush=True)
for nm, args in [("D=64", (torch.randn(1, 4, H, 64, dtype=torch.bfloat16, device=dev),
                           torch.randn(1, 4, H, 64, dtype=torch.bfloat16, device=dev),
                           torch.randn(64, dtype=torch.bfloat16, device=dev), 1e-5)),
                 ("x!=g", (torch.randn(1, 4, H, D, dtype=torch.bfloat16, device=dev),
                           torch.randn(1, 8, H, D, dtype=torch.bfloat16, device=dev),
                           torch.randn(D, dtype=torch.bfloat16, device=dev), 1e-5))]:
    try:
        torch.dlc_kda_rmsnorm_sigmoid_gate(*args)
        torch.dlc.synchronize()
        print(f"[{nm}] 未报错", flush=True)
    except Exception as e:
        print(f"[{nm}] -> {type(e).__name__}: {str(e)[:140]}", flush=True)
