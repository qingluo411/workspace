# 算子需求参考实现（fp32，单文件、只依赖 torch）

本目录放**算子组对齐用的参考实现**；自检都跑在 CPU 上，不需要 DLC 设备：
`python kda_reference_dv_dk.py --selftest`、`python kpool_reference.py --selftest [--with-native]`。

| 算子 | 参考函数 | 需求单 |
|---|---|---|
| `torch.dlc_kda_chunk_prefill`（#03） | `kda_reference_dv_dk.py::chunk_prefill_ref()` | [`../03_dlc_kda_chunk_prefill.md`](../03_dlc_kda_chunk_prefill.md) |
| `torch.dlc_kda_recurrent_decode`（#04） | `kda_reference_dv_dk.py::recurrent_decode_ref()` | [`../04_dlc_kda_recurrent_decode.md`](../04_dlc_kda_recurrent_decode.md) |
| `torch.dlc_kpool_select_topk`（#09 扩展，含池读出） | `kpool_reference.py::kpool_pooled_read_ref()` + [`../09_dlc_kpool_mqa_logits.md`](../09_dlc_kpool_mqa_logits.md) 内 `kpool_select_topk_ref` | [`../09_dlc_kpool_mqa_logits.md`](../09_dlc_kpool_mqa_logits.md) |
| `torch.dlc_kpool_append_compress`（#07 扩展，含 tail 环写） | [`../07_dlc_kpool_compress.md`](../07_dlc_kpool_compress.md) 内 `kpool_append_compress_ref` | [`../07_dlc_kpool_compress.md`](../07_dlc_kpool_compress.md) |
| `torch.dlc_kda_state_gather_scatter`（#05） | `kda_state_gather_scatter_ref.py::kda_state_gather_ref()` / `kda_state_scatter_ref()` | [`../05_dlc_kda_state_gather_scatter.md`](../05_dlc_kda_state_gather_scatter.md) |

## KDA（#03 / #04）：state 布局 `[S, H, Dv, Dk]`

文件：`kda_reference_dv_dk.py`。

### 为什么是 `[S, H, Dv, Dk]`

2026-09-22 算子组同意把 recurrent state 的布局统一为 **`[S, H, Dv, Dk]`**（`S` = slot 数）：

- **行 = value 维（Dv），列 = key 维（Dk）**，衰减 `exp(gate)` 沿 Dk **列**乘；
- 与 fla（`q.new_empty(T, HV, V, K)`，`vllm/vllm/third_party/flash_linear_attention/ops/kda.py:59`）、
  插件 native（`vllm_cl/patch/patch_kda.py:101` 校验 `[pages,H,V,K]`、`:128` 沿 K 衰减）完全一致，
  **插件侧零转换**；#03 的 `initial_state` / `final_state` 同为 `[B,H,Dv,Dk]`；
- 交付版 kernel 原本用 `[S,H,Dk,Dv]`（行 = key），与上述三方互为转置。因为 GLM-5.3-Flash 的
  `Dk == Dv == 128`，两种布局形状都合法：**接错不会报错，只会静默把 key/value 轴用反**。
  这也是必须写死布局并落参考实现的原因。

kernel 内部若更习惯「Dv 在向量 lane」（更快地把 `proj`/`delta`/`out` 放满 128 lane），
可以在载入/写回时于 **VMEM 内做一次 128×128 转置**：搬运量不变（每 head `128*128` 个 fp32），
**不要额外走一遍 HBM**。

## 语义（两个算子共用）

```
gate[t,h,dk] = lower_bound * sigmoid(exp(A_log[h]) * (raw_g[t,h,dk] + g_bias[h,dk]))
q_hat = l2norm(q, eps=1e-6) * scale          # #03 的 query_scale 即 scale
k_hat = l2norm(k, eps=1e-6)
beta  = sigmoid(beta_logits)                 # 两个算子都收 raw beta logits

S      = S * exp(gate_t)                     # 沿 Dk 列衰减
proj   = sum_dk S[dv,dk] * k_hat[t,dk]
S     += k_hat[t][:,None] * (beta_t * (v_t - proj))[:, :, None]   # 外积
out_t  = sum_dk S[dv,dk] * q_hat[t,dk]       # 用**更新后**的 S
```

`chunk_prefill_ref()` 是同一语义的 chunk 版（chunk 内 `cumsum` + WY 形式：掩码下三角矩阵、
单位下三角前代 `(I+L)δ = rhs`、`K_to_end` 状态更新，与 kernel 的
`A / solve_tril / U,W / O_inter / K_to_end` 一一对应），并把尾块长度 / chunk_size 参数化。

## 用法

```bash
cd /root/workspace/vllm-cl/docs/glm5_flash/op_requests/refs
PY=/root/miniconda/envs/qingluo-dev1/bin/python

# 1) 纯 CPU 自检（默认动作，不占 DLC 设备）
$PY kda_reference_dv_dk.py --selftest
$PY kda_reference_dv_dk.py --selftest --heads 8 --dk 128 --dv 128   # 按 GLM5 几何

# 2) 与真机算子对拍（需要 DLC 设备；#03 未修复前会卡死设备，必须 --force-chunk 且用可复位的卡）
$PY kda_reference_dv_dk.py --compare-device dlc:6 --lens 4
$PY kda_reference_dv_dk.py --compare-device dlc:6 --lens 3,1,2 --chunk-size 64 --force-chunk
```

自检覆盖：

- `chunk_prefill_ref` 与 `recurrent_decode_ref`（同一布局下的逐 token 真值）等价：
  `lens = [1]/[2]/[4]/[7]/[16]/[1,1,1,1]/[5,2,1,3]` × `chunk_size = 1/2/8/64`，全部 `err < 1e-7`（fp32）；
- decode 的两个行为约定：`slot = 0`（PAD）→ out 全 0 且 state 不动；
  `num_accepted_tokens = [2]`（窗口 4）→ 写回的 state 等于「第 2 个 token 后的快照」，而不是整窗。

## 与算子对接时的注意点

- `g_bias` 两个算子公开签名不同：#04 是 `[H,Dk]`，**#03 是 `[Dk,H]`**（wrapper 内部 `permute({1,0})`
  成 `[H,Dk]`）。本参考统一收 `[H,Dk]`，传给 #03 时请 `g_bias.t().contiguous()`。
- `A_log` 收 1D `[H]` fp32（模型里是 `[1,1,H,1]`，注意 reshape）。
- 参考实现内部一律 fp32；算子返回的 `out` 是 bf16，比对时请 `.float()`（bf16 量化误差 ~1e-3 属正常）。
- `recurrent_decode_ref` 的 `state` 是**原地更新**（与算子一致）；`skip_slot_ids=(0,)` 复刻了当前 kernel
  「slot 0 跳过」的行为，要纯语义请传 `()`。
- #03 现状：host wrapper 必抛 `permute` 异常且 kernel 卡死设备，修复前不要在共享卡上跑 `--force-chunk`。

## 自检结果（2026-09-22，本机 CPU）

```
=== selftest: chunk_prefill_ref vs recurrent_decode_ref (H=8, Dk=128, Dv=128) ===
  [OK ] lens=[16]  chunk= 64  out_err=7.45e-09  state_err=2.76e-07
=== selftest 通过：chunk 版与逐 token 递推等价（fp32 容差 5e-05）===
=== selftest: recurrent_decode 语义（PAD slot / num_accepted_tokens）===
  [OK ] slot=0(PAD) -> out 全 0=True、state 未改动=True
  [OK ] acc=2/窗口=4 -> 与「2 个 token 后的快照」差 0.00e+00、与「整窗」差 6.89e-01
```

## KPool 池读出（已并入 #09 扩展，2026-10-09）

文件：`kpool_reference.py`（批量版；native golden 是**单请求 1D** 的
`NativeKPoolPagedCache.read_pools`，`vllm_cl/patch/models/kpool_indexer.py:359`）。

```python
kpool_pooled_read_ref(pool_ids,           # [Q, K] int32，每个 query 的 K 个逻辑池 id
                      block_table,        # [num_reqs, blocks] int32，逻辑块 -> 物理块
                      pool_cache,         # [num_blocks, pools_per_block, D] bf16
                      req_ids,            # [Q] int32
                      fill=0.0)           # -> [Q, K, D]，与 pool_cache 同 dtype
```

- 地址换算：`slot = block_table[req_ids[q], id // pools_per_block] * pools_per_block + id % pools_per_block`。
- **`pool_ids < 0`（不止 `-1`）→ 该位置输出 `fill`**；`req_ids` 越界 / 池 id 越界 / `block_table`
  指向 `-1` 或越界物理块 → 报错。
- 与 native golden 的**两处差异**（已在 issue 里写死，避免算子组照 native 实现）：
  1. native 是单请求 1D（`read_pools(request_id, pool_ids)`），本参考是批量 `[Q, K]`；
  2. native 对任何负 id **直接 raise `IndexError`**，本参考按「跳过 → 0」处理。

自检（CPU）：`python kpool_reference.py --selftest --with-native`
→ 批量版与 `NativeKPoolPagedCache.read_pools` 逐元素一致（`max_err=0`），并确认 native 对 `-1` 报错这一差异。
真机计时/对拍：`../probes/probe_kpool_pooled_read.py dlc:N`（交付后自动比对
`torch.dlc_kpool_pooled_read`）。

## KDA 状态 gather/scatter（#05）

`kda_state_gather_scatter_ref.py`：`state [P,H,Dv,Dk]`（**允许 `stride(0) != H*Dv*Dk`**）与
索引/mask 之间按页搬运的 golden reference。

- `kda_state_gather_ref(state, indices, has_initial_state, out=None)`：
  `indices[s] == -1` 或 `has_initial_state[s] == False` → 该序列整页为 0；否则整页拷贝 `state[indices[s]]`。
- `kda_state_scatter_ref(state, src, indices)`：原位写回；`indices[s] == -1` 跳过；
  `indices` 出现重复页时抛 `ValueError`；越界抛 `IndexError`。
- 纯 CPU、单文件，自带自检（构造 `stride(0)=page+3` 的页内 view）：

```bash
python refs/kda_state_gather_scatter_ref.py
# reference self-test OK  (strided state, stride0=43, page=40)
```
