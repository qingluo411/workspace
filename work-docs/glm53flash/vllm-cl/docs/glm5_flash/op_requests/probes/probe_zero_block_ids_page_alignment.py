import torch
dev='dlc'; torch.manual_seed(0)
nb, page_el, n_segs = 8, 128, 2
kv = torch.randint(1,100,(n_segs, nb, page_el), dtype=torch.bfloat16, device=dev)
page_el_i32 = page_el*2//4
stride = nb*page_el_i32*4
base = kv.data_ptr()
addrs = torch.tensor([base + s*stride for s in range(n_segs)], dtype=torch.int64, device=dev)
bid = torch.tensor([1,3], dtype=torch.int32, device=dev)
before = kv.clone()
torch.dlc_vllm_zero_block_ids_(addrs, bid, 2, n_segs, page_el_i32, nb)
torch.dlc.synchronize()
nz = (kv != 0).any(dim=2)
print("per-seg non-zero block ids (expect all but 1,3):")
for s in range(n_segs):
    print("  seg",s,[int(b) for b in torch.nonzero(nz[s]).flatten().tolist()])
# any partial zeroing inside a block?
chg = (kv != before)
print("elements changed total:", int(chg.sum()), "of", kv.numel())
idx = torch.nonzero(chg)
if idx.numel():
    print("first changed coords:", idx[:5].tolist(), " last:", idx[-3:].tolist())
