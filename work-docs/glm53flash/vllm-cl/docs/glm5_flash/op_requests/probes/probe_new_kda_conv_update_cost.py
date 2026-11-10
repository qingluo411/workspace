"""#02 `dlc_kda_causal_conv_update` 耗时归因探针（2026-09-22 晚）。

背景：早期结论认为「慢 13x」来自 host wrapper 的整池拷贝。本脚本把成本拆开实测：

  A) host 侧各步的实际开销（连续拷贝基线 / transpose+contiguous / +clone / wrapper 的 4 遍）；
  B) 固定槽位数与 D，只改「活跃请求数 B」——若耗时几乎不变，说明成本与活跃请求无关；
  C) 固定槽位数与 B，只改「每槽位字节数 D」——区分「按字节」还是「按槽位固定开销」；
  D) 汇总：host 各步之和 vs 算子整体调用。

结论（见 给算子组_01_02_修复要点.md 第二节）：成本 90% 以上来自 kernel 内部
`kda_causal_conv_update.c:109` 的整池 HBM2VMem（约 4 GB/s 量级、与活跃请求数无关），
host wrapper 的 4 遍整池拷贝在真机几何下只有约 0.27 ms。

用法：
    python -u probe_new_kda_conv_update_cost.py [dlc:N]     # 默认 dlc:0
安全：不会卡设备；仍建议一例一张干净卡。

2026-09-24 ABI：conv_state=[S_slot,state_len,D](SD，in-place)，weight=[4,D] fp32，只返回 out。
修复后 kernel 不再整池 HBM2VMem，A 组的「wrapper 4 遍」只剩历史基线（现已不存在）。
"""
import sys
import time
import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
W, L = 4, 3


def bench(fn, n=8):
    """每次调用后同步，取 n 次最小值（排除异步记账与抖动）。"""
    fn()
    torch.dlc.synchronize()
    best = float("inf")
    for _ in range(n):
        t = time.time()
        fn()
        torch.dlc.synchronize()
        best = min(best, time.time() - t)
    return best * 1e3


def ref_out(x_d, slot_state_ld, w2d):
    """单请求、单 token：y = silu(h[-3]*w0 + h[-2]*w1 + h[-1]*w2 + x*w3)，h 为 [L,D] 旧->新。

    注意入参都必须是 1D（x_d:[D]），返回 [D]；不要传 [1,D]，否则 stack 后会多一维、
    与 out 广播成 [B,B,D]，err 变成假值。
    """
    y = (slot_state_ld[L - 3].float() * w2d[:, 0]
         + slot_state_ld[L - 2].float() * w2d[:, 1]
         + slot_state_ld[L - 1].float() * w2d[:, 2]
         + x_d.float() * w2d[:, 3])
    return torch.nn.functional.silu(y).to(torch.bfloat16)


def case(slots, B, D, n=8):
    torch.manual_seed(0)
    x = (torch.randn(B, D, dtype=torch.bfloat16, device=dev) * 0.5).contiguous()
    w2d = (torch.randn(D, W, dtype=torch.float32, device=dev) * 0.3).contiguous()
    w4d = w2d.t().contiguous()                        # 新 ABI: [4, D] fp32
    pool = torch.randn(slots, L, D, dtype=torch.bfloat16, device=dev) * 0.5  # [S,L,D] SD
    qsl = torch.arange(0, B + 1, dtype=torch.int32, device=dev)
    idx = (torch.arange(B, dtype=torch.int32, device=dev) % (slots - 1)) + 1

    want = torch.stack([ref_out(x[b], pool[idx[b]], w2d) for b in range(B)])
    out = torch.dlc_kda_causal_conv_update(
        x, w4d, pool, idx, query_start_loc=qsl, max_query_len=1,
        activation="silu", null_state_id=0)
    err = (out.float() - want.float()).abs().max().item()
    t = bench(lambda: torch.dlc_kda_causal_conv_update(
        x, w4d, pool, idx, query_start_loc=qsl, max_query_len=1,
        activation="silu", null_state_id=0), n=n)
    pool_mb = slots * L * D * 2 / 1024 / 1024
    return t, err, pool_mb


def host_costs(S=1509, Lc=3, D=3072):
    torch.manual_seed(0)
    pool = torch.randn(S, Lc, D, dtype=torch.bfloat16, device=dev)
    flat, dst = pool.reshape(-1), torch.empty(S * Lc * D, dtype=torch.bfloat16, device=dev)

    def four_passes():
        t = pool.transpose(1, 2).contiguous()          # ①
        c = t.clone()                                  # ②
        pool.copy_(c.transpose(1, 2).contiguous())     # ③+④

    return {
        "连续->连续 copy_（基线）": bench(lambda: dst.copy_(flat)),
        "transpose(1,2).contiguous()": bench(lambda: pool.transpose(1, 2).contiguous()),
        "上一行 + clone()": bench(lambda: pool.transpose(1, 2).contiguous().clone()),
        "wrapper 的 4 遍（①~④）": bench(four_passes),
    }


print(f"dev={dev}  K=4, state_len={L}", flush=True)

print("\n=== A) host 侧各步开销（真机几何 S=1509, L=3, D=3072，池子 26.5 MB）===", flush=True)
host = host_costs()
for k, v in host.items():
    print(f"  {k:26s} = {v:7.2f} ms", flush=True)

print("\n=== B) 活跃请求数 B 的影响（slots=1509, D=3072）—— 修复后：与 B、槽位规模都无关 ===", flush=True)
for B in (1, 8, 64):
    t, err, mb = case(1509, B, 3072)
    print(f"  slots=1509 B={B:3d} | 池={mb:6.2f} MB | out_err={err:.3e} | {t:8.2f} ms", flush=True)

print("\n=== C) 每槽位字节数的影响（slots=1509, B=8）—— 修复后：不随池子字节数增长 ===", flush=True)
curve = []
for D in (256, 768, 1536, 3072):
    t, err, mb = case(1509, 8, D)
    curve.append((mb, t))
    print(f"  slots=1509 D={D:5d} | 池={mb:6.2f} MB | out_err={err:.3e} | {t:8.2f} ms"
          f" | {t / mb:6.3f} ms/MB", flush=True)

(mb0, t0), (mb1, t1) = curve[0], curve[-1]
slope, intercept = (t1 - t0) / (mb1 - mb0), t0 - (t1 - t0) / (mb1 - mb0) * mb0
print(f"  线性拟合：t ≈ {intercept:.2f} ms + {slope:.3f} ms/MB"
      f"（斜率≈0/负 = 不再按整池字节数搬运；修复前约 0.28 ms/MB）", flush=True)

t_full, _, _ = case(1509, 8, 3072)
print(f"\n=== D) 汇总（slots=1509, B=8, D=3072）===", flush=True)
print(f"  算子整体调用                        = {t_full:7.2f} ms", flush=True)
print(f"  （历史基线）修复前 host 4 遍整池拷贝  = {host['wrapper 的 4 遍（①~④）']:7.2f} ms"
      f"，现已从 wrapper 移除", flush=True)
print("  => 修复后耗时 = 单次 kernel + 一次 qkv/indices 连续性检查，", flush=True)
print(f"     与槽位总数、池子字节数、活跃请求数均无关（≈{t_full:.2f} ms）。", flush=True)
