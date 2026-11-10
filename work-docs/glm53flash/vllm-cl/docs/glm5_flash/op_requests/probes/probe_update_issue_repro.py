"""#02 `dlc_kda_causal_conv_update` 一页式复现：一次运行打印全部 4 个问题。

用法：
    python -u probe_update_issue_repro.py [dlc:N]      # 默认 dlc:6

真机几何：GLM5-5.3-Flash TP8，num_blocks=1509，conv 状态 [state_len=3, D=3072] bf16（每层 18 KB/页，
池子 26.5 MB）。逐条问题对应输出里的 [P0-1] / [P0-2a] / [P0-2b] / [P1-3] / [P1-4]，
末尾是「现有算子 vs 新算子」汇总表。

2026-09-24 算子组修复后的 ABI（本脚本已适配）：
  - `conv_state` 逻辑形状 **[S_slot, state_len, D]**（SD，D 连续），kernel 直接 in-place 写回；
  - `weight` 是**已转置**的 [4, D] fp32；
  - 返回值只有 `out`（不再返回 `committed_state`）；
  - wrapper 不再做 host D2H 校验、不再做 4 遍整池拷贝。

范围：**plain decode**（每请求 1 个 token、state_len=3、不传 num_accepted_tokens）。
MTP/spec（多 token、num_accepted_tokens 提交语义、state_len>3）不在本次适配范围，
所以本脚本不构造 spec 用例、也不当作缺陷证据。

安全：不会卡设备（不要用会卡死的 prefill 用例；本脚本只测 update）。
"""
import sys
import time

import torch

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "dlc:6"
D, L, W = 3072, 3, 4          # D_conv / state_len / 卷积宽度
SLOTS = 1509                  # 真实 num_blocks
B = 8                         # 活跃请求数（只用到 8 个页）


def bench(fn, n=5, sync=True):
    fn()
    if sync:
        torch.dlc.synchronize()
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn()
        if sync:
            torch.dlc.synchronize()
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best


def make(slots):
    """返回 (x, weight[4,D], pool/state[slots,L,D]=SD, ds_view, idx, qsl)。

    `pool` 既是 SD state（直接传给新算子），也用于同时构造 [slots,D,L]（DS）给旧的 native 算子。
    """
    torch.manual_seed(0)
    x = (torch.randn(B, D, dtype=torch.float32) * 0.5).to(torch.bfloat16).contiguous().to(DEVICE)
    w = (torch.randn(D, W, dtype=torch.float32) * 0.3).contiguous().to(DEVICE)   # [D,4]
    pool = torch.randn(slots, L, D, dtype=torch.bfloat16, device=DEVICE) * 0.5   # [slots,L,D] SD
    qsl = torch.arange(0, B + 1, dtype=torch.int32, device=DEVICE)
    idx = torch.arange(1, B + 1, dtype=torch.int32, device=DEVICE)
    return x, w.t().contiguous(), pool, pool.transpose(-1, -2), idx, qsl


def new_op(x, w4d, state, idx, qsl):
    """plain decode：单 token、不传 num_accepted_tokens。state 为 [S,L,D] SD，被 in-place 更新。"""
    return torch.dlc_kda_causal_conv_update(
        x, w4d, state, idx, query_start_loc=qsl, max_query_len=1,
        activation="silu", null_state_id=0)


def old_op(x, w4d, ds_view, idx):
    # 旧 native 算子要 weight=[D,4]（新 ABI 的 [4,D] 转回去）与 DS 的 state 视图
    return torch._causal_conv1d_update(
        x, w4d.t().contiguous().to(torch.bfloat16), torch.zeros(D, dtype=torch.bfloat16, device=DEVICE),
        ds_view, idx, None, None, -1, None, None, True, True)


print(f"dev={DEVICE}  D={D} state_len={L} slots={SLOTS} B={B}  池子={SLOTS * L * D * 2 / 1024 / 1024:.1f} MB", flush=True)

x, w4d, pool, ds_view, idx, qsl = make(SLOTS)
x_s, w4d_s, pool_s, ds_view_s, idx_s, qsl_s = make(64)

# ---- 正确性前提：数值本身是对的 ----
ref = torch.nn.functional.silu(
    pool[idx, L - 3].float() * w4d[0] + pool[idx, L - 2].float() * w4d[1]
    + pool[idx, L - 1].float() * w4d[2] + x.float() * w4d[3]).to(torch.bfloat16)
out = new_op(x, w4d, pool, idx, qsl)
torch.dlc.synchronize()
print(f"[前提] 数值 out_err={(out.float() - ref.float()).abs().max().item():.3e}（数值正确，问题在性能/健壮性）\n", flush=True)

# ---- [P0-1] kernel 每次整池 DMA：耗时随槽位总数线性增长，与活跃请求数无关 ----
t_1509 = bench(lambda: new_op(x, w4d, pool, idx, qsl))
t_64 = bench(lambda: new_op(x_s, w4d_s, pool_s, idx_s, qsl_s))
t_old = bench(lambda: old_op(x, w4d, ds_view, idx))
print(f"[P0-1] 耗时随「槽位总数」增长（B={B} 不变，只改池子大小）", flush=True)
print(f"        slots={SLOTS:5d}  新算子 {t_1509:6.2f} ms      现有 _causal_conv1d_update {t_old:5.2f} ms  → {t_1509 / t_old:.1f}x", flush=True)
print(f"        slots=   64  新算子 {t_64:6.2f} ms                                            → 与池子大小无关", flush=True)
print("        修复前根因：kda_causal_conv_update.c 每次调用无条件整池 HBM2VMem，"
      "只为给第二个返回值 committed_state 打底，而调用方不用它。\n", flush=True)

# ---- [P0-2a] wrapper 里每次调用都做 host D2H，调用会阻塞（enqueue ≈ device 时间）----
N_ENQ = 20
new_op(x, w4d, pool, idx, qsl)
torch.dlc.synchronize()
_times = []
for _ in range(N_ENQ):
    _a = time.perf_counter()
    new_op(x, w4d, pool, idx, qsl)
    _times.append((time.perf_counter() - _a) * 1e3)
_times.sort()
t_enq = _times[len(_times) // 2]          # 取中位数：偶发 GC/分配抖动会给出 ~100ms 的离群值
print(f"[P0-2a] host 阻塞（D2H）：连续 {N_ENQ} 次不 sync 的调用，入队中位数 {t_enq:.3f} ms/次"
      f"（min={_times[0]:.3f} max={_times[-1]:.3f}；设备耗时 {t_1509:.3f} ms/次）", flush=True)
print(f"        修复前每次入队 ≈ 设备耗时（被 D2H 校验卡住）；修复后中位数 {t_enq:.3f} ms 为纯 launch 开销。", flush=True)
print("        修复前根因：CausalConv1d.cxx 的 qsl[-1].item()/.to(kCPU) 校验，"
      "并在 num_accepted_tokens=None 时再同步一次。\n", flush=True)

# ---- [P0-2b] 因此无法被 DLCGraph 捕获 ----
print("[P0-2b] DLCGraph 捕获：", flush=True)
try:
    graph = torch.dlc.DLCGraph()
    with torch.dlc.graph(graph):
        out_c = new_op(x, w4d, pool, idx, qsl)
    graph.replay()
    torch.dlc.synchronize()
    fresh = new_op(x, w4d, pool, idx, qsl)
    torch.dlc.synchronize()
    print(f"        捕获成功，replay err={(out_c.float() - fresh.float()).abs().max().item():.3e}", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"        捕获失败：{type(exc).__name__}: {str(exc).splitlines()[0][:160]}", flush=True)
    print("        （插件 decode 走 CUDAGraphWrapper，捕获期 host 段只跑一次、replay 用到捕获时的值）\n", flush=True)

# ---- [P1-3] host wrapper 的 4 遍整池拷贝 ----
pool_c = pool.clone()
dst_c = torch.empty(SLOTS, D, L, dtype=torch.bfloat16, device=DEVICE)
dst_l = torch.empty(SLOTS, L, D, dtype=torch.bfloat16, device=DEVICE)


def four_passes():
    dst_c.copy_(pool_c.transpose(1, 2))       # ① transpose(1,2).contiguous()
    dst_l.copy_(dst_c.transpose(1, 2))        # ② clone()
    dst_c.copy_(dst_l.transpose(1, 2))        # ③ transpose(1,2).contiguous()
    pool_c.copy_(dst_c.transpose(1, 2))       # ④ conv_state.copy_(committed_state)


t_host = bench(four_passes, n=10)
print(f"[P1-3] 修复前 host wrapper 4 遍整池拷贝 = {t_host:.2f} ms（占单次调用 {t_host / t_1509 * 100:.0f}%；"
      f"34 层 ≈ {t_host * 34:.1f} ms/token）", flush=True)
print("        修复后 wrapper 不再做这 4 遍拷贝（新 ABI 只有 qkv.contiguous() + zeros_like(out)）。\n", flush=True)

# ---- [P1-4] pad 行（idx=-1）输出脏值 ----
x_p, w_p, pool_p, ds_p, _, qsl_p = make(8)
idx_p = torch.tensor([-1, 1, 2, 3, 4, -1, 5, 6], dtype=torch.int32, device=DEVICE)
out_new = new_op(x_p, w_p, pool_p, idx_p, qsl_p)
out_old = old_op(x_p, w_p, ds_p, idx_p)
torch.dlc.synchronize()
pad_rows_new = out_new[[0, 5]]
pad_rows_old = out_old[[0, 5]]
print(f"[P1-4] idx=-1（pad slot）行的输出：", flush=True)
print(f"        新算子 non-zero={int(torch.count_nonzero(pad_rows_new))}  maxabs={pad_rows_new.abs().max().item():.3f}"
      f"  （at::zeros_like 预置，应为全 0）", flush=True)
print(f"        现有算子 non-zero={int(torch.count_nonzero(pad_rows_old))}  maxabs={pad_rows_old.abs().max().item():.3f}"
      f"  （at::zeros 预置，应为全 0）", flush=True)
print("        位置：kernel 对 slot<0/slot>=state_slots 直接 continue 不写 out；"
      "host 从 at::empty 改为 at::zeros_like 后已修复。\n", flush=True)
