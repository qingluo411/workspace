import torch
dev='dlc'; torch.manual_seed(0)
M,H,D,N = 4,3,8,16
q = torch.randn(M,H,D, dtype=torch.bfloat16, device=dev)
k = torch.randn(N,D, dtype=torch.bfloat16, device=dev)
w = torch.randn(M,H, dtype=torch.bfloat16, device=dev)
ks = torch.zeros(M, dtype=torch.int32, device=dev)
ke = torch.full((M,), N, dtype=torch.int32, device=dev)
out = torch.dlc_multi_query_attention(q,k,w,ks,ke)
torch.dlc.synchronize()
# KPool semantics: per-head dot -> ReLU -> weighted sum
scale = D**-0.5
kpool = (torch.relu(torch.einsum('mhd,nd->mhn', q.float(), k.float())*scale) * (w.float()/(H**0.5)).unsqueeze(-1)).sum(1)
# existing MQA semantics: head-weighted query -> dot
mqa = (q.float()*w.float().unsqueeze(-1)).sum(1) @ k.float().t()
print("out shape", tuple(out.shape))
print("err vs KPool-semantics :", (out.float()-kpool).abs().max().item())
print("err vs head-weight-first:", (out.float()-mqa).abs().max().item())
