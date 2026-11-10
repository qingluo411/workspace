import torch
dev='dlc'
torch.manual_seed(0)
T, S = 128, 8192
x  = torch.randn(T, S, dtype=torch.float32, device=dev)*0.5
w  = torch.randn(S, dtype=torch.bfloat16, device=dev)*0.3
def ref(xf, w, eps):
    return xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+eps)*w.float()
try:
    o = torch.dlc_fused_rms_norm_f32_bf16(x, w, eps=1e-5)
    torch.dlc.synchronize()
    print("[#17] out dtype:", o.dtype, "shape", tuple(o.shape), "err:", (o-ref(x,w,1e-5)).abs().max().item())
except Exception as e:
    print("[#17] FAILED", repr(e)[:300])
