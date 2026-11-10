"""#05 复现：KDA 状态池上用通用 index_select / index_copy_ 的代价。

真机几何（GLM-5.3-Flash, TP8）：
    num_blocks=1168, recurrent state = [1168, 8, 128, 128] fp32 = 584 MiB
    该张量是 Mamba 共享页 ``[P, 1, 1, page_size_bytes]``（int8）里的一段 view：
    conv 段占前 18432 B，rec 段从 18432 开始、占 524288 B，
    因此 rec view 的 dim0 stride = 542720/4 = 135680（< 8*128*128 = 131072）→ **非连续**。

判据：
    * dense（连续池）与 padded（真实页内 view）执行同样的 gather/scatter；
    * 若 padded 比 dense 慢 20x 以上 ⇒ 复现「非连续池触发通用索引算子的病态路径」。

用法：
    python -u probe_state_pool_index_cost.py [dlc:N]        # 默认 dlc:0
退出码：
    0 = 复现（padded 至少比 dense 慢 20x）
    1 = 未复现
"""
import sys
import time

import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
torch.dlc.set_device(int(dev.split(":")[-1]))

NH = 8                        # TP8 下 local heads = 64/8
V = K = 128                   # head_dim = 128
P = 1168                      # GLM-5.3-Flash 实测 num_blocks
CONV_B, REC_B = 18432, 524288  # 18432 = 3072*3*2(bf16)；524288 = 8*128*128*4(fp32)
PAGE_B = CONV_B + REC_B
S = 8                         # num_seqs
ITERS = 5


def bench(fn, iters=ITERS):
    fn()
    torch.dlc.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.dlc.synchronize()
    return (time.perf_counter() - t0) / iters


idx = torch.arange(S, dtype=torch.int64, device=dev)
src = torch.randn(S, NH, V, K, dtype=torch.float32, device=dev)

dense = torch.zeros(P, NH, V, K, dtype=torch.float32, device=dev)
raw = torch.zeros(P, PAGE_B, dtype=torch.int8, device=dev)
padded = raw[:, CONV_B:CONV_B + REC_B].view(torch.float32).view(P, NH, V, K)

print(f"device={dev}  P={P}  state=[{P},{NH},{V},{K}] fp32 ({P*NH*V*K*4/2**20:.0f} MiB)")
print(f"  dense  stride={dense.stride()}  contiguous={dense.is_contiguous()}")
print(f"  padded stride={padded.stride()}  contiguous={padded.is_contiguous()}"
      f"  (DLC Mamba 共享页内 view)\n")

print(f"{'':>8} {'index_select+clone':>20} {'index_copy_':>14}")
timings = {}
for nm, st in (("dense", dense), ("padded", padded)):
    t_sel = bench(lambda: st.index_select(0, idx).clone())
    t_cp = bench(lambda: st.index_copy_(0, idx, src))
    timings[nm] = (t_sel, t_cp)
    print(f"{nm:>8} {t_sel*1e3:>17.3f}ms {t_cp*1e3:>12.3f}ms")

for nm, st in (("dense", dense), ("padded", padded)):
    ref = torch.zeros(P, NH, V, K, dtype=torch.float32, device="cpu")
    st.zero_()
    st.index_copy_(0, idx, src)
    ref[idx.cpu()] = src.cpu()
    print(f"{nm:>8} scatter 语义: {'OK' if bool(torch.equal(st.cpu(), ref)) else 'FAIL'}")

sel_slow, cp_slow = timings["padded"]
sel_fast, cp_fast = timings["dense"]
print(f"\n放大倍数：index_select {sel_slow/sel_fast:.0f}x，index_copy_ {cp_slow/cp_fast:.0f}x")
print(f"按 34 个 KDA 层累计（每层 1 次 gather + 1 次 scatter）：")
print(f"  dense  单步 {(sel_fast+cp_fast)*34*1e3:8.1f} ms")
print(f"  padded 单步 {(sel_slow+cp_slow)*34*1e3:8.1f} ms   ← 当前模型走的就是这条")

sys.exit(0 if (sel_slow / sel_fast) >= 20 and (cp_slow / cp_fast) >= 20 else 1)
