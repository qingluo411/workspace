import sys, torch
dev='dlc'
torch.manual_seed(0)
latent = int(sys.argv[1]); rope = int(sys.argv[2]); K = int(sys.argv[3])
nslots, Q, H = 256, 2, 4
D = latent + rope
scale = latent ** -0.5
cache = torch.randn(nslots, D, dtype=torch.bfloat16, device=dev)
q = torch.randn(Q, H, D, dtype=torch.bfloat16, device=dev)
topk = torch.randint(-1, nslots, (Q, K), dtype=torch.int32, device=dev)
out, lse = torch.dlc_sparse_attention_kv_cache(q, cache, topk, scale, latent)
torch.dlc.synchronize()
valid = topk >= 0
idx = topk.clamp(min=0).long()
g = cache.float()[idx]
kvc = g[..., :latent]; kpe = g[..., latent:]
s = (torch.einsum('bnh,bkh->bnk', q.float()[..., :latent], kvc)
     + torch.einsum('bnr,bkr->bnk', q.float()[..., latent:], kpe)) * scale
s = s.masked_fill(~valid.unsqueeze(1), float('-inf'))
p = torch.nan_to_num(torch.softmax(s, -1), nan=0.0)
ref = torch.einsum('bnk,bkl->bnl', p, kvc).to(q.dtype)
print(f"OK max_abs_err={(out.float()-ref.float()).abs().max().item():.4e}")
