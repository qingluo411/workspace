import torch
dev='dlc'; torch.manual_seed(7)
H=K=V=8; T=6
q = torch.randn(1,T,H,K, dtype=torch.bfloat16, device=dev)*0.5
k = torch.randn(1,T,H,K, dtype=torch.bfloat16, device=dev)*0.5
v = torch.randn(1,T,H,V, dtype=torch.bfloat16, device=dev)*0.5
st  = torch.randn(2,H,K,V, dtype=torch.bfloat16, device=dev)*0.1
cu  = torch.tensor([0,T], dtype=torch.int32, device=dev)
idx = torch.tensor([[0]], dtype=torch.int32, device=dev)
acc = torch.tensor([1], dtype=torch.int32, device=dev)
beta= torch.rand(T,H, dtype=torch.bfloat16, device=dev)*0.5+0.2
g2  = -torch.rand(T,H, dtype=torch.float32, device=dev)*0.3             # per-head scalar
g3  = -torch.rand(T,H,K, dtype=torch.float32, device=dev)*0.3           # per-channel

def ref(q,k,v,g,beta,st):
    h = st[0].cpu().float().clone(); kc=k.cpu().float(); vc=v.cpu().float(); qc=q.cpu().float(); bc=beta.cpu().float()
    gC = g.cpu().float()
    o = torch.zeros(T,H,V)
    for t in range(T):
        if gC.dim()==2: h = h*torch.exp(gC[t]).view(H,1,1)
        else:           h = h*torch.exp(gC[t])          # [H,K] -> broadcast over V
        d = (vc[0,t]-torch.einsum('hkv,hk->hv',h,kc[0,t]))*bc[t].unsqueeze(-1)
        h = h + torch.einsum('hv,hk->hkv',d,kc[0,t])
        o[t] = torch.einsum('hkv,hk->hv',h,qc[0,t]*K**-0.5)
    return o, h

for name, g in [("per-head (T,H)", g2), ("per-channel (T,H,K)", g3)]:
    try:
        stc = st.clone()
        o, fs = torch.fused_recurrent_gated_delta_rule(q,k,v,g,beta,scale=K**-0.5,
                    initial_state=stc, inplace_final_state=True, cu_seqlens=cu,
                    ssm_state_indices=idx, num_accepted_tokens=acc,
                    use_qk_l2norm_in_kernel=False)
        torch.dlc.synchronize()
        ro, rh = ref(q,k,v,g,beta,st)
        print(f"[recurrent {name}] OK out_err={(o.cpu()[0]-ro).abs().max().item():.3e} "
              f"state_err={(fs.cpu()[0]-rh).abs().max().item():.3e}")
    except Exception as e:
        print(f"[recurrent {name}] FAILED:", repr(e)[:180])
