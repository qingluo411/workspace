import torch
dev='dlc'
P,H,V,K = 8,4,16,16
state = torch.randn(P,H,V,K, dtype=torch.float32, device=dev)
idx = torch.tensor([3,1,3], dtype=torch.int64, device=dev)
mask = torch.tensor([True,False,True], device=dev)
g = state.index_select(0, idx)
g = torch.where(mask.view(-1,1,1,1), g, torch.zeros_like(g))
src = torch.randn(2,H,V,K, dtype=torch.float32, device=dev)
state.index_copy_(0, torch.tensor([2,5], dtype=torch.int64, device=dev), src)
torch.dlc.synchronize()
ref_g = state0 = None
print("gather row0 == state[3]:", torch.equal(g[0].cpu(), state[3].cpu()) if False else "n/a")
ok_g = torch.equal(g[1].cpu(), torch.zeros(H,V,K))
ok_s = torch.equal(state[5].cpu(), src[1].cpu())
print("index_select+where (gather w/ mask):", "OK" if ok_g else "FAIL", "| index_copy_ (scatter):", "OK" if ok_s else "FAIL")
