import torch
dev='dlc'; torch.manual_seed(5)
H=2;K=4;V=4;T=2
q = torch.randn(1,T,H,K, dtype=torch.bfloat16, device=dev)*0.5
k = torch.randn(1,T,H,K, dtype=torch.bfloat16, device=dev)*0.5
v = torch.randn(1,T,H,V, dtype=torch.bfloat16, device=dev)*0.5
st0 = torch.randn(1,H,K,V, dtype=torch.bfloat16, device=dev)*0.1
cu = torch.tensor([0,T], dtype=torch.int32, device=dev)
beta = torch.rand(T,H, dtype=torch.bfloat16, device=dev)*0.5+0.2
a = -torch.rand(T,H, dtype=torch.bfloat16, device=dev)*0.3   # per-token increments

def run(g_in):
    o, fs = torch.chunk_gated_delta_rule(q,k,v,g_in,beta,scale=K**-0.5,initial_state=st0,
                                         output_final_state=True, cu_seqlens=cu,
                                         head_first=False, use_qk_l2norm_in_kernel=False)
    torch.dlc.synchronize(); return o.cpu().float(), fs.cpu().float()

def ref_per_token(ginc):   # decay_t = exp(ginc_t)
    h = st0.cpu().float()[0].clone()
    o = torch.zeros(T,H,V)
    kc,vc,qc,bc = k.cpu().float(), v.cpu().float(), q.cpu().float(), beta.cpu().float()
    for t in range(T):
        h = h*torch.exp(ginc[t].cpu().float()).view(H,1,1)
        d = (vc[0,t]-torch.einsum('hkv,hk->hv',h,kc[0,t]))*bc[t].unsqueeze(-1)
        h = h + torch.einsum('hv,hk->hkv',d,kc[0,t])
        o[t] = torch.einsum('hkv,hk->hv',h,qc[0,t]*K**-0.5)
    return o, h

cum = a.cumsum(0)
for name, g_in in [("per-token g (natural)", a), ("cumsum g", cum)]:
    o, fs = run(g_in)
    for rname, gin in [("per-token ref", a), ("cumsum ref", cum)]:
        ro, rh = ref_per_token(gin)
        print(f"input={name:22s} vs {rname:14s} out_err={(o[0]-ro).abs().max().item():.3e} state_err={(fs[0]-rh).abs().max().item():.3e}")
