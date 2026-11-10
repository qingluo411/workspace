import torch
dev='dlc'
torch.manual_seed(0)
DIM, W, T, NP = 3072, 4, 16, 4

def native_conv(x, w, bias, hist):
    h=hist.clone(); ys=[]
    for t in range(x.shape[1]):
        win=torch.cat([h, x[:,t:t+1]],1)
        y=(win.float()*w.float()).sum(-1)
        if bias is not None: y=y+bias.float()
        ys.append(torch.nn.functional.silu(y).to(x.dtype))
        h=torch.cat([h[:,1:], x[:,t:t+1]],1)
    return torch.stack(ys,1), h

w = torch.randn(DIM, W, dtype=torch.bfloat16, device=dev)*0.3
bias = torch.randn(DIM, dtype=torch.bfloat16, device=dev)*0.1
hist0 = torch.randn(W-1, DIM, dtype=torch.bfloat16, device=dev)*0.5

# ---- prefill ----
x = torch.randn(DIM, T, dtype=torch.bfloat16, device=dev)
buf = torch.zeros(NP, W-1, DIM, dtype=torch.bfloat16, device=dev); buf[0].copy_(hist0)
state = buf.permute(0,2,1)
out = torch._causal_conv1d(x, w, bias, state,
        torch.tensor([0,T], dtype=torch.int32, device=dev),
        torch.tensor([0], dtype=torch.int32, device=dev),
        torch.tensor([True], dtype=torch.bool, device=dev), None,None,None,None,0,True,True)
torch.dlc.synchronize()
ref, refh = native_conv(x, w, bias, hist0.t().contiguous())
print(f"[prefill] out_err={(out.float()-ref.float()).abs().max().item():.3e}  state_err={(buf[0].float()-refh.t().float()).abs().max().item():.3e}")

# ---- update (decode, 1 token/req, B=2) ----
buf2 = torch.zeros(NP, W-1, DIM, dtype=torch.bfloat16, device=dev)
buf2[0].copy_(hist0); buf2[1].copy_(hist0*0.5)
st2 = buf2.permute(0,2,1)
xu = torch.randn(2, DIM, dtype=torch.bfloat16, device=dev)
o = torch._causal_conv1d_update(xu, w, bias, st2,
        torch.tensor([0,1], dtype=torch.int32, device=dev), None, None, -1, None, None, True, True)
torch.dlc.synchronize()
e0=(o[0].float()-native_conv(xu[0].unsqueeze(1),w,bias,hist0.t().contiguous())[0][:,0].float()).abs().max().item()
e1=(o[1].float()-native_conv(xu[1].unsqueeze(1),w,bias,(hist0*0.5).t().contiguous())[0][:,0].float()).abs().max().item()
print(f"[update ] out_err0={e0:.3e} out_err1={e1:.3e}")
