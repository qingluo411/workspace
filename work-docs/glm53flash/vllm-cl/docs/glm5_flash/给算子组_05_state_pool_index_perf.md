# #05 KDA 状态池索引性能：真机 profile 证据（2026-09-28）

> 支撑 `op_requests/05_dlc_kda_state_gather_scatter.md`（P0）。本文件只放实测数据与机制结论，
> 需求与修法见该 issue。

## 1. 问题现象

GLM-5.3-Flash 在 8 卡（TP=8）上服务**能起来但极慢**：

- 一个 15 token 的 prefill 请求，**单步 ≈ 60 s**；
- decode ≈ 1~2 s/token；
- `top` 里 8 个 worker 各占 ~200% CPU，但 **DLC 芯片上几乎看不到计算**。

模型能正常 `Application startup complete`，KV cache 148,255 tokens，不是卡死。

## 2. py-spy 采样（TP0，prefill 步）

采样方法：`py-spy record --pid <Worker_TP0> --duration 60 --rate 50 --format speedscope`，
统计叶子帧（innermost frame）占比。

### 2.1 纯 prefill 窗口（15 token 的一步，>60 s）

| 叶子帧（含文件:行） | 占比 |
|---|---|
| `scatter_states_dlc (patch_kda.py:323)` | **50.1 %** |
| `_prefill_new_op (patch_kda.py:141)` | **35.2 %** |
| `gather_initial_states_dlc (patch_kda.py:316)` | **13.8 %** |
| 其余（`_run_kda` / kpool / mhc_pre / sparse_mla …） | 0.9 % |

→ 模型其余部分（注意力、MoE、线性层）合计不到 1 %。

### 2.2 混合窗口（decode 为主，仍有 prefill 在跑）

| 叶子帧 | 占比 | 说明 |
|---|---|---|
| `scatter_states_dlc (patch_kda.py:323)` | 30.7 % | prefill 路径 |
| `_prefill_new_op (patch_kda.py:141)` | 18.0 % | prefill 短卷积 |
| `_run_kda (patch_kda.py:363)` | 16.6 % | decode：native 逐 token Python 循环 |
| `_update_new_op (patch_kda.py:190)` | 15.3 % | decode：融合短卷积更新算子 |
| `_run_kda (patch_kda.py:351)` | 11.4 % | decode：`initial_state[idx].float().clone()` |
| `gather_initial_states_dlc (patch_kda.py:316)` | 7.4 % | prefill 路径 |

父帧分布：`_forward (kda.py:563)` 30.7 %、`fused_recurrent_kda_dlc` 28.1 %、
`causal_conv1d_fn_dlc` 18.0 %、`causal_conv1d_update_dlc` 15.3 %。

按层摊算（34 个 KDA 层）：`scatter` ≈ 0.9 s/层、`gather` ≈ 0.25 s/层。

## 3. 机制：为什么这么慢

### 3.1 张量几何

```
num_blocks                     = 1168
page_size_bytes                = 542720  (= 18432 conv + 524288 rec)
conv state per page            = [3, 3072] bf16        = 18432 B
recurrent state per page       = [8, 128, 128] fp32    = 524288 B
recurrent state pool (per layer) = [1168, 8, 128, 128] fp32 = 584 MiB   × 34 层
```

Mamba 页是**一个共享的 int8 buffer** `[P, 1, 1, page_size_bytes]`
（`vllm_cl/worker/mamba_cache.py:allocate_mamba_cache`），各 state 是页内按字节偏移切出来的 view。
rec 段从第 18432 B 开始、长 524288 B，所以：

```
padded(dim0) stride = 542720 / 4 = 135680   ≠   8*128*128 = 131072
```

→ **recurrent state view 非连续**（`contiguous=False`）。

### 3.2 直接测量（`probes/probe_state_pool_index_cost.py`，dlc:0）

同样只动 8 页（4 MiB），只把池子从 dense 换成真实页内 view：

| | `index_select(0,idx).clone()` | `index_copy_(0,idx,src)` |
|---|---|---|
| dense `[1168,8,128,128]` | 0.181 ms | 1.544 ms |
| padded（真实布局） | **579.7 ms** | **1873.0 ms** |
| 放大倍数 | **3196x** | **1213x** |

- padded 的等效搬运速率只有 ~0.3 GB/s，而 dense 是 ~100+ GB/s 量级 → **padded 没有走 DMA，
  是 host 参与的逐元素路径**，与「设备看不到计算、CPU 满载」的现象一致。
- 34 层累计：dense **58.7 ms/步** vs padded **83.4 s/步**，与实测 prefill 单步 ≈ 60 s 吻合。

### 3.3 即使连续，`index_copy_` 也是整池 DMA

池子规模扫描（dense，同样只写 8 页）：

| 池页数 | 池大小 | `index_copy_` |
|---|---|---|
| 8 | 4.0 MiB | 0.083 ms |
| 64 | 32.0 MiB | 0.143 ms |
| 256 | 128.0 MiB | 0.445 ms |
| 1168 | 584.0 MiB | 1.544 ms |

随页数线性增长 → 搬的是整个 `self`。根因在
`DLC_Custom_Kernel/dlc_kernels/index/index_copy.c` 的 `index_copy_vmask`：
`input0_length = h_input0 * w_input0`（整个 self）后整池 `dlc_dma`。

### 3.4 另有每层 2 次 host D2H

`vllm_cl/patch/patch_kda.py:315/321` 的 `torch.any(indices<0)` /
`torch.any(indices>=state.shape[0])` / `indices.unique()` 在 Python 里取真值，强制同步。

## 4. 结论

| 项 | 状态 |
|---|---|
| #05「可复用、无需动作」的原判定 | **推翻**：原判定基于 `[8,4,16,16]`（8 KiB）张量，未覆盖真实 584 MiB 页内 view |
| 当前瓶颈 | 非连续状态下通用索引算子走入 host 逐元素路径（占 prefill 单步 64 %） |
| 次要问题 | 连续池仍是整池 DMA；适配层每层 2 次 D2H 同步 |
| 修复出口 | 见 `op_requests/05_dlc_kda_state_gather_scatter.md`：按页 stride 寻址的融合 gather/scatter |

## 5. 复现

```bash
cd /root/workspace/vllm-cl
python -u docs/glm5_flash/op_requests/probes/probe_state_pool_index_cost.py dlc:0
```

脚本退出码 0 = 已复现（padded 比 dense 慢 ≥ 20x）。

---

## 6. 补充（2026-09-28 晚）：这条不是唯一瓶颈，也不是算子能修的

**同一个 `#01` prefill 算子、同一份数据**，只把 `conv_state` 从连续池换成同一张 padded 页的 view：

| `conv_state` `[1168,3,3072]` bf16 | stride0 | contiguous | 单次 `dlc_kda_causal_conv_prefill` |
|---|---|---|---|
| dense 独立池（20.5 MiB 存储） | 9216 | True | **0.093 ms** |
| padded（605 MiB 存储，页内 view） | 271360 | False | **1459.3 ms（15700x）** |

34 层累计 **4.3 ms → 48.4 s**，与模型内 `_prefill_new_op` 占 35%（≈21 s）同量级。

同一实验对 `#02` 的 update 算子也成立（`qkv=[8,1,3072]`、`max_query_len=1`）：

| op | dense `conv_state` | padded（页内 view） | 放大 |
|---|---|---|---|
| `dlc_kda_causal_conv_prefill` | 0.166 ms | **1396.1 ms** | 8400x |
| `dlc_kda_causal_conv_update` | 0.259 ms | **1339.3 ms** | 5200x |

**这解释了此前记录的「`_prefill_new_op` 模型内 0.6 s vs 隔离 0.06 ms，差 1 万倍」**：
隔离值是**连续池**上的、且当时比的是 update 算子；模型内拿到的是 padded 页 view。
两处都不是算子的锅。

> **未决**：update 算子隔离测 padded 是 1339 ms，若 decode 每步真跑 34 次，单步应 ≈45 s；
> 但从 profile 占比与 `/metrics` 推出来的 decode 步长明显没这么大。
> 说明微基准的输入与模型内仍不完全一致，或 decode 路径没有每层都走该算子。
> **需要在模型内加一次计时插桩确认**（决定「拆池」对 decode 的收益）。

**关键推论**：

1. `#01` 的 kernel 本身没有问题（连续池 0.093 ms），算子组**没有东西可修** ——
   这是**布局税**：只要张量是那张 padded 页的 view，任何搬运都会退化成
   约 O(整个存储)、~0.5 GB/s 的路径。
2. 所以 **#05 单独修好也不够**：gather/scatter 解决了 64%，剩下 35% 的 conv prefill
   仍然会停在 ~48 s/步。
3. 这条税的来源在 DLC 运行时/后端（非连续张量上的通用搬运路径），不在某个 kernel；
   与 `../给pytorch后端_dlc_strided_copy_bug.md` 记录的问题很可能是同一片区域。

**因此优先级需要重估**：插件侧「拆池」（消除非连续 view）覆盖所有被拖累的算子，
是当前性价比最高的修法；`#05` 从「阻断发布的 P0」降为「连续池上的进一步优化」
（`index_copy_` 1.554 ms → per-page 0.155 ms/层）。

---

## 7. 拆池落地后的实测（2026-09-28 晚，插件侧，无需算子组）

插件侧「拆池」已实现并真机验证（`DLC_MAMBA_SPLIT_POOL`，默认开；设计与改动点见
[`../拆池_mamba_state_pool.md`](../拆池_mamba_state_pool.md)）：

```
旧: kv_caches[layer] = int8 [1168, 1, 1, 542720]              # 打包页
新: kv_caches[layer] = (bf16 [1168, 3, 3072],                 # conv, stride0 = 9216
                        fp32 [1168, 8, 128, 128])             # rec,  stride0 = 131072
```

每层总字节不变（604.5 MiB），block id 语义不变，页仍是 512B 倍数。真机日志确认
`contiguous=[True, True]`（272 条 = 34 层 × 8 卡）。

### 7.1 算子级（TP7 插桩，13 token prefill + 8 decode）

| 项 | 拆池前 | 拆池后 | 倍数 |
|---|---|---|---|
| prefill host | 179 434 ms/step | **5 703 ms/step** | 31x |
| decode host | 119 962 ms/step | **2 579 ms/step** | 46x |
| `dlc_kda_causal_conv_prefill` | 1 941 ms/call | **0.40 ms/call** | 4 800x |
| `dlc_kda_causal_conv_update` | 1 835 ms/call | **0.19 ms/call** | 9 700x |
| `state gather` | 729 ms/call | **44 ms/call** | 17x |
| `state scatter` | 2 444 ms/call | **36 ms/call** | 68x |
| `kda_recurrent`（Python 参考实现） | 1 556 ms/call | **41 ms/call** | 38x |
| `kda_chunk`（Python 参考实现） | 17 ms/call | 20 ms/call | — |

端到端：13 token prompt + 8 output 的一个请求 **25.1 s**（拆池前光 prefill 一步就 ~179 s）。

### 7.2 结论：新瓶颈排序

1. **`#05` 仍是 prefill 侧第一项**：gather 1 495 ms + scatter 1 228 ms = 2.7 s/step，
   占 prefill 步的 47%，且远高于连续池微基准的 58.7 ms/step —— `index_select`/`index_copy_`
   在 584 MiB 池上仍是整池扫描。按本 issue 的「按页 stride 只搬选中页」实现即可收敛。
2. `#03`/`#04`（`kda_chunk` 0.69 s/step、`kda_recurrent` 1.41 s/step）成为 decode 侧主项。
3. 注意：`kda_recurrent` 的 41 ms/call 里，**算子本身只占约 0.03 ms**（设备微基准见
   `probes/probe_kda_recurrent_decode_cost.py`：`state[i].clone()` 0.026 ms、
   `.item()` 0.041 ms、`index_copy_` 0.011 ms）。差值来自适配层 `_run_kda` 每层多次
   host D2H（`cu_seqlens.item()` / `torch.any(...)` / `.unique()`）把 eager 队列排空，
   属于**插件侧自收敛项**，不在算子组范围。
