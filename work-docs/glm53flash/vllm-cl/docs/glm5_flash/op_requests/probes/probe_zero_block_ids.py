import torch
dev='dlc'
torch.manual_seed(0)

def run(desc, nb, page_el, n_segs, ids, dtype):
    kv = torch.randint(1, 100, (n_segs, nb, page_el), dtype=dtype, device=dev)
    esz = kv.element_size()
    # page_size_el is in int32 units for the kernel
    if dtype != torch.int32:
        page_el_i32 = page_el*esz//4
    else:
        page_el_i32 = page_el
    stride_bytes = nb*page_el_i32*4
    base = kv.data_ptr()
    addrs = torch.tensor([base + s*stride_bytes for s in range(n_segs)], dtype=torch.int64, device=dev)
    bid = torch.tensor(ids, dtype=torch.int32, device=dev)
    ref = kv.clone()
    for b in ids: ref[:, b] = 0
    torch.dlc_vllm_zero_block_ids_(addrs, bid, len(ids), n_segs, page_el_i32, nb)
    torch.dlc.synchronize()
    ok = bool(torch.equal(kv, ref))
    print(f"[{desc}] nb={nb} page_el(int32)={page_el_i32} dtype={dtype} ids={ids} -> exact_match={ok}")
    if not ok:
        d = (kv != ref)
        nbad = int(d.any(dim=2).sum())
        print("   blocks differing from ref:", int(d.flatten(1).any(1).sum().item()), "of", n_segs)

run("official-config int32", 8, 128, 2, [0,2,5], torch.int32)
run("official-config int32 2", 16, 256, 3, [1,15], torch.int32)
run("bf16 kv-cache pages", 8, 128, 2, [1,3], torch.bfloat16)
