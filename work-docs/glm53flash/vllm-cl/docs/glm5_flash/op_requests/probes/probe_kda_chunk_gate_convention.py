import torch
dev='dlc'; torch.manual_seed(3)
H=K=V=2; T=1
def mk():
    return (torch.randn(1,T,H,K, dtype=torch.bfloat16, device=dev)*0.5,
            torch.randn(1,T,H,K, dtype=torch.bfloat16, device=dev)*0.5,
            torch.randn(1,T,H,V, dtype=torch.bfloat16, device=dev)*0.5,
            torch.randn(1,H,K,V, dtype=torch.bfloat16, device=dev)*0.1)
q,k,v,st0 = mk()
cu = torch.tensor([0,T], dtype=torch.int32, device=dev)
beta = torch.rand(T,H, dtype=torch.bfloat16, device=dev)*0.5+0.2
g    = -torch.rand(T,H, dtype=torch.bfloat16, device=dev)*0.3
o, fs = torch.chunk_gated_delta_rule(q,k,v,g,beta,scale=K**-0.5,initial_state=st0,
                                     output_final_state=True, cu_seqlens=cu,
                                     head_first=False, use_qk_l2norm_in_kernel=False)
torch.dlc.synchronize()
h0=st0[0].float()
for name, decay in [("exp(g)", torch.exp(g[0].float())), ("exp2(g)", torch.exp2(g[0].float()))]:
    h = h0*decay.unsqueeze(-1)
    delta = (v[0,0].float() - torch.einsum('hkv,hk->hv', h, k[0,0].float()))*beta[0].float().unsqueeze(-1)
    h2 = h + torch.einsum('hv,hk->hkv', delta, k[0,0].float())
    ref_o = torch.einsum('hkv,hk->hv', h2, q[0,0].float()*K**-0.5)
    print(f"{name:9s} out_err={(o[0,0].float()-ref_o).abs().max().item():.3e} state_err={(fs[0].float()-h2).abs().max().item():.3e}")
