"""#18：用 GLM-5.3-Flash 真实页几何验证 dlc_vllm_zero_block_ids_ 的 512B 对齐前提。

覆盖 5 类页（MLA full / KPool pool / KPool tail / KDA conv / KDA recurrent），
每类的页字节数都取自真机 spec（TP8、block_size=256、index_kpool=4）：
  262144 / 32768 / 4096 / 18432 / 524288 字节，全部是 512B 的整数倍。
"""
import torch

dev = "dlc"
torch.manual_seed(0)

# (描述, 每页字节数, dtype)
PAGE_CLASSES = [
    ("MLA full      [256tok,512] bf16", 256 * 512 * 2, torch.bfloat16),
    ("KPool pool    [64,128]    bf16", 64 * 256 * 2, torch.bfloat16),
    ("KPool tail    [2,4,128]   bf16", 2 * 4 * 256 * 2, torch.bfloat16),
    ("KDA conv      [3,3072]    bf16", 3 * 3072 * 2, torch.bfloat16),
    ("KDA recurrent [8,128,128] fp32", 8 * 128 * 128 * 4, torch.float32),
]


def run(desc, page_bytes, dtype, n_segs=3, nb=8, ids=(1, 3, 6)):
    assert page_bytes % 512 == 0, (desc, page_bytes)
    page_el = page_bytes // 4  # kernel 的 page_size_el 单位是 int32 字
    kv = torch.randint(1, 100, (n_segs, nb, page_el), dtype=torch.int32, device=dev)
    base = kv.data_ptr()
    addrs = torch.tensor(
        [base + s * nb * page_bytes for s in range(n_segs)], dtype=torch.int64, device=dev
    )
    bid = torch.tensor(list(ids), dtype=torch.int32, device=dev)
    ref = kv.clone()
    for b in ids:
        ref[:, b] = 0
    torch.dlc_vllm_zero_block_ids_(addrs, bid, len(ids), n_segs, page_el, nb)
    torch.dlc.synchronize()
    ok = bool(torch.equal(kv, ref))
    print(f"[{desc}] page_bytes={page_bytes} (%512={page_bytes % 512}) "
          f"page_el(int32)={page_el} n_segs={n_segs} ids={list(ids)} -> exact_match={ok}")
    return ok


def mixed_one_call():
    """反例：同一次调用里混用不同页大小的 segment。"""
    a = torch.randint(1, 100, (4, 8, 32768 // 4), dtype=torch.int32, device=dev)  # 32768B 页
    b = torch.randint(1, 100, (4, 8, 4096 // 4), dtype=torch.int32, device=dev)   # 4096B 页
    page_el = 32768 // 4
    addrs = torch.tensor([a.data_ptr(), b.data_ptr()], dtype=torch.int64, device=dev)
    bid = torch.tensor([1, 3], dtype=torch.int32, device=dev)
    ref_a, ref_b = a.clone(), b.clone()
    ref_a[:, [1, 3]] = 0
    ref_b[:, [1, 3]] = 0
    torch.dlc_vllm_zero_block_ids_(addrs, bid, 2, 2, page_el, 8)
    torch.dlc.synchronize()
    print(f"[mixed-page one call] seg0(32768B) match={bool(torch.equal(a, ref_a))} "
          f"seg1(4096B) match={bool(torch.equal(b, ref_b))}")


if __name__ == "__main__":
    results = [run(desc, nb_, dt) for desc, nb_, dt in PAGE_CLASSES]
    mixed_one_call()
    print("ALL_ALIGNED_AND_EXACT:", all(results))
