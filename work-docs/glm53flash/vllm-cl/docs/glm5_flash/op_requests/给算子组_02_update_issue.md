# [Operator Request] `dlc_kda_causal_conv_update`：整池 DMA 导致慢 28~36x + host D2H 阻断 graph + pad 行脏输出

> GLM-5.3-Flash 的 KDA 短卷积 **decode 更新**算子。**数值逐元素正确**（plain decode `state_err=0`），
> 问题全在性能与接口：真机几何（`num_blocks=1509`、`D_conv=3072`、池子 26.5 MB）下单次
> **7.2 ms vs 现有 `torch._causal_conv1d_update` 0.2~0.3 ms（≈28~36x）**，34 层 ⇒ 单步 decode 多花 ~240 ms/token。
>
> **本次范围（适配初期）**：只要求 **plain decode** —— 每请求 1 个 token、`state_len=3`、
> 不传 `num_accepted_tokens`、slot 取自 block table 首列。**MTP/spec（多 token、`num_accepted_tokens`
> 提交语义、`state_len>3`）本阶段不做**，不在本次要求内。
>
> 本文件只讲三件事：**问题 / 修法 / 复现**；背景与完整实测见 `给算子组_01_02_修复要点.md` 第二节。

## 一、有哪些问题

| # | 级别 | 问题 | 证据（真机几何：slots=1509、B=8、D=3072、state_len=3、单 token） |
|---|---|---|---|
| 1 | **P0** | **kernel 每次调用无条件整池 DMA**：`kda_causal_conv_update.c:109` 的 `HBM2VMem(hbm_conv_state, vmem_commit, state_slots*state_len*dim_bf16)` 把整个池（26.5 MB）搬进 VMEM，只为给第二个返回值 `committed_state`（全池快照）打底 —— 而**调用方根本不用它**（`vllm/models/glm5next/nvidia/kda.py:479` 只取 `out`）。卷积只读活跃 slot（`:131`）、回写也只写活跃 slot（`:235-237`） | 单次 **7.2 ms**（≈4 GB/s 整池搬运，`t ≈ 0.74 ms + 0.251 ms/MB`）；耗时随**槽位总数**线性增长（4→3072 槽位：0.61→14.73 ms），**与活跃请求数无关**（B=1/8/64 → 7.59/7.45/7.50 ms）；现有算子与槽位规模无关 |
| 2 | **P0** | **wrapper 每次调用都做 host D2H 同步**，调用会阻塞 host；也因此**无法被 `DLCGraph` 捕获**（插件 decode 走 `CUDAGraphWrapper`，`vllm_cl/worker/dlc_model_runner.py:5110`） | 2a：不 sync 地调用一次 = **7.34 ms**，与设备耗时 7.21 ms 相同（host 被卡住）；2b：`with torch.dlc.graph(g)` → `capture_end ... DLC error: Unknown synStatus`（捕获期 `.item()` 读到的值也不可用） |
| 3 | P1 | **wrapper 每次做 4 遍整池拷贝**：`transpose(1,2).contiguous()` → `clone()` → kernel → `transpose(1,2).contiguous()` → `conv_state.copy_()` | 0.18 ms/层（≈3%），34 层 ≈ 6 ms/token；不是 30x 的主因，但属纯浪费 |
| 4 | P1 | **pad 行（`idx=-1`）输出是脏值**：kernel 对 `slot<0 / slot>=state_slots` 直接 `continue` 不写 `out`，host 用 `at::empty` 申请输出 | 新算子 `maxabs=1.73`、6144 个非零；现有算子是 `at::zeros` 预置 → 全 0 |

## 二、该如何修

### 1. [P0] 去掉 kernel 的整池 DMA

```c
/* kda_causal_conv_update.c:109 —— 删掉这一行 */
HBM2VMem(hbm_conv_state, vmem_commit, state_slots * state_len * dim_bf16);
```

- 删掉后，唯一需要它的是「本次没有提交快照的 slot」（`accepted<=0`）—— 在 `:235` 的回写外加条件
  `if (accepted > 0)` 跳过即可（不提交 = state 不变），语义完全不变；
- `committed_state` 建议改为**只覆盖活跃 slot**，或做成可选输出 / 直接去掉（调用方不用）；
- host 侧 `at::empty({state_len, D, state_slots})`（每次新分配 26.5 MB）随之取消。

**验收**：1509 槽位单次 ≤ 1 ms；耗时不再随槽位总数 / 池子字节数增长。

### 2. [P0] 去掉 host D2H 同步

```cpp
// CausalConv1d.cxx:346
TORCH_CHECK(qsl[-1].item<int>() == qkv_flat.size(0), ...);   // D2H，删掉或放进开关
// CausalConv1d.cxx:349
auto qsl_cpu_check = qsl.to(at::kCPU);                       // D2H + 等设备队列排空
// CausalConv1d.cxx:376-378：不传 num_accepted_tokens 时（plain decode 就是这条）
auto qsl_cpu = qsl.to(at::kCPU); auto acc_cpu = accepted.to(at::kCPU);
... aptr[b] = qptr[b + 1] - qptr[b]; accepted.copy_(acc_cpu);  // host 算 + 拷回设备
```

- 这三处 host 校验放到 `validate_data` 开关下（默认关，debug 时才同步）；
- 不传 `num_accepted_tokens` 时改成设备侧张量运算 `accepted = qsl[1:] - qsl[:-1]`
  （或让 kernel 直接按 `qsl` 推每请求长度）。

**验收**：`validate_data` 关闭时不 sync 调用的 host 耗时 < 设备耗时的 20%；`DLCGraph` 能捕获，
且 replay 前改写输入缓冲后 replay 结果仍正确。

### 3. [P1] 去掉 wrapper 的 4 遍整池拷贝

位置：`CausalConv1d.cxx:401 / :402 / :414 / :415`。理想做法是像现有 `_causal_conv1d` 那样走
`make_flat_storage_view`（`CausalConv1d.cxx:60`）零拷贝 strided；**但前提是 kernel 支持行距元数据** ——
现在 `:131`/`:236` 硬编码 `slot*state_len*dim_bf16` 的紧凑寻址，直接零拷贝会把 prefill 的 state-stride
缺陷复制到 update 上。若不打算做 strided，至少去掉 `clone()` 与多余的 `transpose(1,2).contiguous()`。

**验收**：0.18 ms → 接近 0；池子行距用例仍 `state_err=0` 且行间段零改写。

### 4. [P1] pad 行输出预置零

host 用 `at::zeros` 申请 `out`（现有 wrapper 就是 `at::zeros`，`CausalConv1d.cxx:242`），
或在 kernel/host 里把被跳过的行显式写 0。

**验收**：`idx=-1` 那行输出全 0。

### 5. 其他接口约束（接入需要，供确认）

`qkv` 2D 时必须显式传 `query_start_loc` 且 `max_query_len ≥ 每请求长度`；`D_conv` 必须是 256 的倍数
（GLM5 = 3072 满足）；活跃请求的 state slot 必须互不重复；`conv_state` 为 `[S_slot,D,state_len]` bf16。

### 6. 本阶段不要求（留待后续 MTP/spec 时再对齐）

- 多 token（`T>1`）与 `num_accepted_tokens` 的提交语义（新算子 = 整窗滚动 + accepted 快照；
  旧 kernel = 读窗口偏移 + 整窗回写，`causal_conv1d_update.c:37-40,148-149,198`）；两者语义不同，
  但**适配初期不走 MTP/spec，这次不要求处理**；
- `state_len > 3`（= 3 + num_spec）的宽度支持；
- 变长 packed 输入（每请求长度不一致）的接口约定。

## 三、问题该如何复现

**脚本**：`vllm-cl/docs/glm5_flash/op_requests/probes/probe_update_issue_repro.py`
（一条命令跑完上面第 1~4 条，末尾给汇总表；只测 plain decode，不会卡设备）

```bash
cd /root/workspace/vllm-cl
/root/miniconda/envs/qingluo-dev1/bin/python -u \
    docs/glm5_flash/op_requests/probes/probe_update_issue_repro.py dlc:6
```

**实测输出**（dlc:2 / dlc:3 / dlc:6 一致）：

```
[前提] 数值 out_err=0.000e+00（数值正确，问题在性能/健壮性）

[P0-1] 耗时随「槽位总数」增长（B=8 不变，只改池子大小）
        slots= 1509  新算子   7.21 ms      现有 _causal_conv1d_update  0.26 ms  → 28.1x
        slots=   64  新算子   0.69 ms

[P0-2a] host 阻塞（D2H）：不 sync 地调一次   7.34 ms vs 设备耗时   7.21 ms

[P0-2b] DLCGraph 捕获：捕获失败：RuntimeError ... DLCGraph.cpp:160(capture_end) DLC error: Unknown synStatus

[P1-3] host wrapper 4 遍整池拷贝 = 0.18 ms（占单次调用 3%；34 层 ≈ 6.2 ms/token）

[P1-4] idx=-1（pad slot）行的输出：
        新算子 non-zero=6144  maxabs=1.734  （at::empty 残留脏值）
        现有算子 non-zero=0  maxabs=0.000  （at::zeros 预置，应为全 0）

=== 汇总（slots=1509, B=8, D=3072, 每 token 每层调用一次，共 34 层）===
  新算子单层   7.21 ms  → 34 层   245.1 ms/token
  现有算子单层   0.26 ms  → 34 层     8.7 ms/token
  其中 host 4 遍拷贝 0.18 ms/层，kernel 内整池 DMA ≈ 7.03 ms/层
```

**环境注意**

- 捕获失败时设备侧还会打印 `[DLCSynapse] sync API called during stream capture` —— 这是 host 同步 API 出现在
  捕获段里的直接证据（对应问题 2b）；
- 用 `dlc:1~7`；`dlc:0` 的 strided 整池拷贝异常慢（复位后依旧），会把现有算子的耗时也抬到 ~0.6 ms，
  倍数随之从 ~30x 变成 ~13x；
- 不会卡设备（只测 update；会把设备卡死的是 prefill 用例）；
- 其他相关脚本：`probe_new_kda_conv_update{,_window,_scaling,_cost,_stride}.py`；
  已固化的 pytest 验收回归：`vllm-cl/tests/test_glm5_kda_conv_update_op.py`
  （其中与本文件对应的 4 个 xfail = 上面第 1~4 条，修好后会自动 XPASS）。

<details>
<summary>附：复现脚本全文（probe_update_issue_repro.py，直接复制即可运行）</summary>

```python
"""#02 `dlc_kda_causal_conv_update` 一页式复现：一次运行打印全部 4 个问题。

用法：
    python -u probe_update_issue_repro.py [dlc:N]      # 默认 dlc:6

真机几何：GLM5-5.3-Flash TP8，num_blocks=1509，conv 状态 [state_len=3, D=3072] bf16（每层 18 KB/页，
池子 26.5 MB）。逐条问题对应输出里的 [P0-1] / [P0-2a] / [P0-2b] / [P1-3] / [P1-4]，
末尾是「现有算子 vs 新算子」汇总表。

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
    torch.manual_seed(0)
    x = (torch.randn(B, D, dtype=torch.float32) * 0.5).to(torch.bfloat16).contiguous().to(DEVICE)
    w = (torch.randn(D, W, dtype=torch.float32) * 0.3).contiguous().to(DEVICE)
    pool = torch.full((slots, L, D), 3.0, dtype=torch.bfloat16, device=DEVICE)
    pool[:, :, :] = (torch.randn(slots, L, D, dtype=torch.float32) * 0.4).to(torch.bfloat16).to(DEVICE)
    qsl = torch.arange(0, B + 1, dtype=torch.int32, device=DEVICE)
    idx = torch.arange(1, B + 1, dtype=torch.int32, device=DEVICE)
    return x, w, pool, pool.transpose(-1, -2), idx, qsl


def new_op(x, w, state, idx, qsl):
    """plain decode：单 token、不传 num_accepted_tokens。"""
    return torch.dlc_kda_causal_conv_update(
        x, w, state, idx, query_start_loc=qsl, max_query_len=1,
        activation="silu", null_state_id=0)


def old_op(x, w, state, idx):
    return torch._causal_conv1d_update(
        x, w.to(torch.bfloat16), torch.zeros(D, dtype=torch.bfloat16, device=DEVICE),
        state, idx, None, None, -1, None, None, True, True)


print(f"dev={DEVICE}  D={D} state_len={L} slots={SLOTS} B={B}  池子={SLOTS * L * D * 2 / 1024 / 1024:.1f} MB", flush=True)

x, w, pool, view, idx, qsl = make(SLOTS)
x_s, w_s, pool_s, view_s, idx_s, qsl_s = make(64)

# ---- 正确性前提：数值本身是对的 ----
ref = torch.nn.functional.silu(
    pool[idx, L - 3].float() * w[:, 0] + pool[idx, L - 2].float() * w[:, 1]
    + pool[idx, L - 1].float() * w[:, 2] + x.float() * w[:, 3]).to(torch.bfloat16)
out, _ = new_op(x, w, view, idx, qsl)
torch.dlc.synchronize()
print(f"[前提] 数值 out_err={(out.float() - ref.float()).abs().max().item():.3e}（数值正确，问题在性能/健壮性）\n", flush=True)

# ---- [P0-1] kernel 每次整池 DMA：耗时随槽位总数线性增长，与活跃请求数无关 ----
t_1509 = bench(lambda: new_op(x, w, view, idx, qsl))
t_64 = bench(lambda: new_op(x_s, w_s, view_s, idx_s, qsl_s))
t_old = bench(lambda: old_op(x, w, view, idx))
print(f"[P0-1] 耗时随「槽位总数」增长（B={B} 不变，只改池子大小）", flush=True)
print(f"        slots={SLOTS:5d}  新算子 {t_1509:6.2f} ms      现有 _causal_conv1d_update {t_old:5.2f} ms  → {t_1509 / t_old:.1f}x", flush=True)
print(f"        slots=   64  新算子 {t_64:6.2f} ms                                            → 池子越大越慢", flush=True)
print("        根因：kda_causal_conv_update.c:109 每次调用无条件整池 HBM2VMem（≈4 GB/s），"
      "只为给第二个返回值 committed_state 打底，而调用方不用它。\n", flush=True)

# ---- [P0-2a] wrapper 里每次调用都做 host D2H，调用会阻塞（enqueue ≈ device 时间）----
t_enq = bench(lambda: new_op(x, w, view, idx, qsl), sync=False)
print(f"[P0-2a] host 阻塞（D2H）：不 sync 地调一次 {t_enq:6.2f} ms vs 设备耗时 {t_1509:6.2f} ms", flush=True)
print("        根因：CausalConv1d.cxx:346/:349 的 qsl[-1].item()/.to(kCPU) 校验，"
      ":376-378 在 num_accepted_tokens=None 时再同步一次并把 host 算出的长度拷回设备。\n", flush=True)

# ---- [P0-2b] 因此无法被 DLCGraph 捕获 ----
print("[P0-2b] DLCGraph 捕获：", flush=True)
try:
    graph = torch.dlc.DLCGraph()
    with torch.dlc.graph(graph):
        out_c = new_op(x, w, view, idx, qsl)
    graph.replay()
    torch.dlc.synchronize()
    fresh, _ = new_op(x, w, view, idx, qsl)
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
print(f"[P1-3] host wrapper 4 遍整池拷贝 = {t_host:.2f} ms（占单次调用 {t_host / t_1509 * 100:.0f}%；"
      f"34 层 ≈ {t_host * 34:.1f} ms/token）", flush=True)
print("        位置：CausalConv1d.cxx:401/:402/:414/:415（不是 13x 的主因，但应一并去掉）\n", flush=True)

# ---- [P1-4] pad 行（idx=-1）输出脏值 ----
x_p, w_p, pool_p, view_p, _, qsl_p = make(8)
idx_p = torch.tensor([-1, 1, 2, 3, 4, -1, 5, 6], dtype=torch.int32, device=DEVICE)
out_new, _ = new_op(x_p, w_p, view_p, idx_p, qsl_p)
out_old = old_op(x_p, w_p, view_p.clone(), idx_p)
torch.dlc.synchronize()
pad_rows_new = out_new[[0, 5]]
pad_rows_old = out_old[[0, 5]]
print(f"[P1-4] idx=-1（pad slot）行的输出：", flush=True)
print(f"        新算子 non-zero={int(torch.count_nonzero(pad_rows_new))}  maxabs={pad_rows_new.abs().max().item():.3f}"
      f"  （at::empty 残留脏值）", flush=True)
print(f"        现有算子 non-zero={int(torch.count_nonzero(pad_rows_old))}  maxabs={pad_rows_old.abs().max().item():.3f}"
      f"  （at::zeros 预置，应为全 0）", flush=True)
print("        位置：kernel 对 slot<0/slot>=state_slots 直接 continue 不写 out，host 用 at::empty 申请输出。\n", flush=True)

# ---- 汇总 ----
print("=== 汇总（slots=1509, B=8, D=3072, 每 token 每层调用一次，共 34 层）===", flush=True)
print(f"  新算子单层 {t_1509:6.2f} ms  → 34 层 {t_1509 * 34:7.1f} ms/token", flush=True)
print(f"  现有算子单层 {t_old:6.2f} ms  → 34 层 {t_old * 34:7.1f} ms/token", flush=True)
print(f"  其中 host 4 遍拷贝 {t_host:.2f} ms/层，kernel 内整池 DMA ≈ {t_1509 - t_host:.2f} ms/层", flush=True)
```

</details>
