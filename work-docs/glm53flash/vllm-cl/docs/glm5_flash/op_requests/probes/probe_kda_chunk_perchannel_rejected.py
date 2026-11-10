import torch
dev='dlc'; torch.manual_seed(0)
H=K=V=8; T=32
q = torch.randn(1,T,H,K, dtype=torch.bfloat16, device=dev)*0.5
k = torch.randn(1,T,H,K, dtype=torch.bfloat16, device=dev)*0.5
v = torch.randn(1,T,H,V, dtype=torch.bfloat16, device=dev)*0.5
st0 = torch.randn(1,H,K,V, dtype=torch.bfloat16, device=dev)*0.1
cu = torch.tensor([0,T], dtype=torch.int32, device=dev)
beta2d = torch.rand(T,H, dtype=torch.bfloat16, device=dev)*0.5+0.2
g2d    = -torch.rand(T,H, dtype=torch.bfloat16, device=dev)*0.3
g3d_flat = -torch.rand(T,H,K, dtype=torch.bfloat16, device=dev)*0.3   # flattened per-channel

def ref_recurrence(q,k,v,g,beta,st,scale):
    T,Hh,Kk=q.shape[1],q.shape[2],q.shape[3]; Vv=v.shape[-1]
    h = st[0].float().clone()
    out = torch.zeros(T,Hh,Vv, dtype=torch.float32, device=q.device)
    for t in range(T):
        qt = q[0,t].float()*scale; kt = k[0,t].float(); vt = v[0,t].float()
        if g.dim()==2: h = h*torch.exp(g[t]).unsqueeze(-1)
        else:          h = h*torch.exp(g[t]).unsqueeze(-1)
        delta = (vt - torch.einsum('hkv,hk->hv', h, kt))*beta[t].float().unsqueeze(-1)
        h = h + torch.einsum('hv,hk->hkv', delta, kt)
        out[t] = torch.einsum('hkv,hk->hv', h, qt)
    return out.to(torch.bfloat16), h

for name, g in [("2D per-head (T,H)", g2d), ("3D per-channel (T,H,K)", g3d_flat)]:
    try:
        o, fs = torch.chunk_gated_delta_rule(q,k,v,g,beta2d,scale=K**-0.5,initial_state=st0,
                                             output_final_state=True, cu_seqlens=cu,
                                             head_first=False, use_qk_l2norm_in_kernel=False)
        torch.dlc.synchronize()
        ro, rh = ref_recurrence(q,k,v,g,beta2d,st0,K**-0.5)
        print(f"[chunk {name}] OK out_err={(o.float()-ro.float()).abs().max().item():.3e} "
              f"state_err={(fs.float()-rh.float()).abs().max().item():.3e}")
    except Exception as e:
        print(f"[chunk {name}] FAILED:", repr(e)[:200])
