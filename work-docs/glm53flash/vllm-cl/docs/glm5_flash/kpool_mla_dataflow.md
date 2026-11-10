# GLM-5.3-Flash KPool MLA 数据流与算子覆盖

> 目的：从理论数据流角度把 KPool 选池链路（prefill / decode 同一套）逐段拆开，
> 每段标注对应的 native 实现位置、算子需求编号、当前接入状态，并指出**没有算子配上**的段。
> 相关：`op_requests/07_dlc_kpool_compress.md`（**扩展**，含 tail 环写入）、
> `09_dlc_kpool_mqa_logits.md`（**扩展**，含池读出 + topk + 展开）、
> `op_requests/19_dlc_kpool_slot_mapping.md`、
> `11_dlc_sparse_mla_forward_nope512.md`、
> `op_requests/接入清单_2026-09-28.md`、`性能优化点清单_2026-09-29.md`。
>
> **2026-10-09 范围重划**：原 #08 `dlc_kpool_append_update` **撤销**，
> 原 #10 `dlc_kpool_expand_topk` **退出主路径**；两单的语义分别并入 **07 扩展**
> 与 **09 扩展**（见下方 S2–S7 各行与 §6 覆盖汇总表）。

## 1. 为什么要 KPool

GLM-5.3-Flash 共 45 层 = 34 层 KDA（线性注意力）+ 11 层 MLA（`deepseek_sparse_attention`）。
只有 11 个 MLA 层带 indexer，每层在每个 query 上选 top-k token 做稀疏 attention。

朴素做法是 query 对全部历史 K 打分，成本 O(L)。KPool 先把历史按 `index_kpool=4` 个 token
压成 1 个 pool state，评分只在 pool 上做，选中的 pool 再展开回 4 个 token：

- 选池成本降到 O(L/4)；
- **真正的 attention 仍然在原始 token 上做**。

因此 KPool 只影响「选谁」（召回），不影响 attention 的数值。这是整套设计最重要的前提：
池化引入的误差是**选择误差**，不是注意力数值误差。

## 2. 几何与常量

| 量 | 值 | 说明 |
|---|---|---|
| MLA 层数 | 11 | 层号 3,7,...,43 |
| `index_n_heads` H | 32 | indexer 头数 |
| `index_head_dim` D | 128 | indexer head 维 |
| `index_kpool` | 4 | 一个池 = 4 个 token |
| `index_topk` | 2048 | 最终送进 attention 的 token 数 |
| `block_size` | 256 | 一个物理块 256 个 token |
| `pools_per_block` | 64 | 256 / 4 |
| 选池数 | 512 | 2048 / 4 |
| topk 输出宽 | 2051 | 2048 + 残尾 3 |
| topk buffer 宽 | 2176 | 2051 向上取到 128 的倍数（`vllm/models/glm5next/nvidia/model.py:596`） |

## 3. 三套 cache，两种职责

| cache | 形状 | 定义 | 职责 |
|---|---|---|---|
| pool cache | `[blocks, 64, 128]` bf16 | `vllm_cl/patch/models/glm5_indexer.py:67` | 压缩后的池向量，**只用于打分** |
| tail cache | `[blocks, 2, 4, 128]` bf16 | `glm5_indexer.py:103` | 平面 0=原始 K，平面 1=gate，只存未满 4 个的残池，**只用于续接压缩** |
| MLA latent cache | `[blocks, 256, 512]` bf16 | — | 真正被 attention 读的 KV（512 宽 NoPE） |

**KPool 的两套 cache 都不参与 attention**：它们只是选择器的状态。

## 4. 计算图总览

```
hidden_states [T, HID]
   │
   ├─ wq_b(qr)                    → q       [T, 32, 128]
   ├─ wk_weights_proj(h)          → fused   [T, 160]
   │     ├─ LayerNorm(fused[:,:128])  → k      [T, 128]   （池化 key 来源）
   │     └─ fp32 head 权重投影        → weights [T, 32]    （每头评分权重）
   └─ index_kpool_compress_gate·h → gate    [T, 128] bf16 （压缩门控）

native_kpool_indexer_step(q, k, weights, gate, ape, pool_cache, tail_cache, meta...)
   │
   ├─ S2 尾环写入      k/gate → tail_cache（环）
   ├─ S3 池化压缩      4 token --softmax(gate+ape)--> 1 池 → pool_cache
   ├─ S4 池读出        pool_cache + block_table → pooled_keys [P, 128]
   ├─ S5 池级打分      q, pooled_keys, weights → scores [T, P_visible]
   ├─ S6 top-k 选池    每行取 min(512, visible) 个池 id
   └─ S7 展开          pool id → 4 个 token id，截到 2048，追加残尾 0..3 个
                        → out [T, 2051] → topk_indices_buffer [T, 2176]
   │
   ├─ S8 逻辑位置 → latent cache 物理槽（convert_req_index_to_global_index）
   ├─ S9 latent cache 写入（concat_and_cache_mla，NoPE 512）
   └─ S10 稀疏 attention（dlc_sparse_attention_kv_cache，512-NoPE）
```

关键排序性质：`visible = sequence_length // pool_size`，一个 query 只能看见在它之前**已经闭合**的池。
同一 chunk 内新写的池对自己前面的行不可见，由 `pooled_keys[:visible]` 的切片天然保证
（`vllm_cl/patch/models/kpool_indexer.py:1074`），不需要额外同步。

## 5. 逐段数据流与算子映射

### S0 每步 metadata / slot mapping（worker 侧，每步一次）

一条算子 **#19 `torch.dlc_kpool_slot_mapping`**（双输出，见
`op_requests/19_dlc_kpool_slot_mapping.md`）承接，当前仍是两段 native 组合实现：

- **池槽**：`vllm_cl/patch/models/kpool_slot_mapping.py:18`
  (`get_compressed_slot_mapping_native`)。把 token 位置映射到 pool cache 物理槽，
  只有 `pos % 4 == 3` 的行拿到槽位，其余为 `-1`；槽位 = `物理块号 * 64 + 块内池号`，
  需按 `compressed_pos // 64` 做多块寻址。内部用 `searchsorted`（`:81`）。
- **tail 环槽**：`vllm_cl/patch/patch_mla_indexer.py:119`、
  `vllm_cl/kpool_worker_slots.py:28`。给每个活 token 分配 tail cache 的物理环槽
  （`tail 页表第 0 列 * 4 + pos % 4`，单块寻址），同样走 `searchsorted`
  （`patch_mla_indexer.py:162`、`kpool_worker_slots.py:59`）。

两段共享「由 `query_start_loc` 求每 token 的 request id + 每请求 `q_start/q_end/len`」这一整段中间量；
DLC 未注册 `searchsorted`，插件回退是把 `query_start_loc` 搬到 CPU 再搬回
（`vllm_cl/_searchsorted_utils.py:8`，实测 0.28–0.33 ms/次）。

计数（当前 V1 runner；V2 需要 Triton，未启用）：原本**每步 3 次** ——
① worker 侧 `BlockTable.compute_slot_mapping`（`patch_block_table.py:14` → `kpool_worker_slots.py:59`）；
② tail metadata builder（`patch_mla_indexer.py:191` → `:162`）；
③ pool metadata builder（`patch_mla_indexer.py:839` 钩子 → `kpool_slot_mapping.py:81`）。
① 与 ② 算的是**同一个** tail 映射，**✅ 2026-10-09 已在插件侧去重**（builder 改为复用 worker 写入的
block-table slot mapping，`DLC_KPOOL_TAIL_SLOT_REBUILD=1` 可回退到本地重建），**现为每步 2 次 ≈ 0.6 ms/步**。
合并成一条算子、一次算完、输出两个 `[T]` int32，是把这 2 次压到 0 的做法。

### S1 indexer 投影（框架侧）

- `q`：`glm5_indexer.py:284`；`k`：`:287`；`weights`：`:298`（fp32，为了排序精度）；
  `gate`：`:306`（投影后立刻 round 成 bf16，保证 chunk 边界前后同一个池的结果一致）。
- → **不需要专门算子**，由通用 matmul / LayerNorm / linear 承担（属框架侧）。

### S2 tail 环写入

- native：`kpool_indexer.py:1029-1035`（逐 token `.copy_()`）；
  fused：`_seed_kpool_fused` `kpool_indexer.py:651`，环写为一次 scatter（`:680`）。
- → 算子 **#07 扩展 `torch.dlc_kpool_append_compress`**（原 #08 的环更新并入 07）。
  状态：**未交付**。kernel 内三阶段（读 prior → 压缩本 chunk 所有闭合池 → 写尾池），
  `prior` 的 clone、布尔索引压紧槽位表、`cat`、`ape` 转换、`tail_start` 标量全部移入 kernel。

### S3 池化压缩

- 数学：`logits[i,d] = gate[i,d] + ape[i,d]`（`[4,128]`）→ `softmax(dim=0)` →
  `pooled[d] = Σ_i probs[i,d] * k[i,d]`。
- native：`compress_kpool` `kpool_indexer.py:474`；
  fused：`torch.dlc_kpool_compress` `kpool_indexer.py:695`。
- → 算子 **#07**（已接入）→ **#07 扩展**（把 S2 一起收进来，一次调用完成压缩 + 环写）。
  状态：基础形态**已接入**；扩展形态**未交付**。

### S4 池读出

- native：`kpool_indexer.py:1051-1052`，用 `block_table` 把逻辑池 id 换成物理页后直接 gather。
- → 算子 **#09 扩展 `torch.dlc_kpool_select_topk`**。状态：**未交付**；
  当前由普通 index 承担。扩展后 kernel 内按地址算术直接读
  `pool_cache[block_table[r, k // PPB], k % PPB]`，不再落地 `[P, D]` gather 张量
  （独立读出算子 `dlc_kpool_pooled_read` 不再需要，原 #08 可选读出侧一并撤销）。

### S5 池级打分

- 数学：`dots = einsum("nhd,pd->nhp", q, pooled_keys) / sqrt(128)`；
  `scores = Σ_h relu(dots) * (weights / sqrt(32))`。
- native：`kpool_scores` `kpool_indexer.py:554`；
  fused：`torch.dlc_kpool_mqa_logits` `kpool_indexer.py:716`（`_scores_kpool_fused` `:706`）。
- → 算子 **#09**（已接入）→ **#09 扩展**（与 S4/S6/S7 合成一次调用）。
  基础形态**已接入**；扩展形态**未交付**。注意**不能复用现有 MQA**：现有
  `torch.dlc_multi_query_attention` 是「先按 head 加权再点积」且无 ReLU，语义不等价。

### S6 top-512 选池

- native：`torch.topk` `kpool_indexer.py:1078`，`k = min(512, visible)`。
- → 算子 **#09 扩展**（`k_sel = min(topk_tokens // pool_size, visible)` 在 kernel 内选池，
  并列只要求集合一致，建议「降序 + 同分按池 id 升序」）。状态：**未交付**；
  当前保留 `torch.topk` 小算子。

### S7 展开 + 残尾拼接

- native：`expand_pool_topk` `kpool_indexer.py:583`；
  fused：`torch.dlc_vllm_kpool_expand_topk` `kpool_indexer.py:1084`。
- 语义：池 `p` → token `[4p, 4p+4)`，按该行自己的 `sequence_length` 截断并置 `-1`；
  再追加未闭合残尾 `[visible*4, sequence_length)`（0..3 个）。输出 `[T, 2051]`。
- → 算子 **#10**（已接入，**已退出主路径**）→ **并入 #09 扩展**：展开 + 残尾拼接由
  `torch.dlc_kpool_select_topk` 在选池之后直接写 `out` 整行（含对齐填充区 `-1`）。
  状态：**未交付**；#10 不再单独维护，`torch.dlc_vllm_kpool_expand_topk` 保留可用。

### S8 逻辑位置 → latent cache 物理槽

- 插件封装：`vllm_cl/attention/sparse_mla.py:349`，底层 `torch.convert_req_index_to_global_index`
  （`vllm_cl/_custom_ops.py:946`）。
- → **已有算子，无需新需求**（覆盖核查里的「可复用」）。

### S9 NoPE 512 latent cache 写入

- `dlc_vllm_concat_and_cache_mla_`，接线在 `vllm_cl/attention/sparse_mla.py`。
- → 算子 **#12**。状态：**已接入**。

### S10 稀疏 attention

- `dlc_ops.sparse_attention_nope(...)`：`vllm_cl/attention/sparse_mla.py:579`
  （512 宽 NoPE，直接读 paged cache；576 路径的算子不能服务该布局）。
- → 算子 **#11**。状态：**已接入**（`..._nope_bf16_{h2,h4,h8,h64}[_topk*]` 家族）。

### S11 KV / state 整页清零（链路外围）

- `vllm_cl/patch/patch_kv_block_zeroer.py`。
- → 算子 **#18**。状态：**已接入**。

## 6. 覆盖汇总表

| 段 | 功能 | native 位置 | 算子 | 状态 |
|---|---|---|---|---|
| S0 | pool + tail slot mapping（双输出） | `kpool_slot_mapping.py:18` / `patch_mla_indexer.py:119` / `kpool_worker_slots.py:28` | **#19** | 已提需求（P1），未交付 |
| S1 | indexer 投影 / LN / gate | `glm5_indexer.py:284-307` | 通用 matmul/LN | 框架侧，无需需求 |
| S2 | tail 环写入 | `kpool_indexer.py:680` / `:1030` | **#07 扩展**（原 #08 并入） | 未交付（原 #08 撤销） |
| S3 | 池化压缩 | `kpool_indexer.py:474` / `:695` | **#07 → #07 扩展** | 基础形态已接入；扩展未交付 |
| S4 | 池读出 | `kpool_indexer.py:1051` | **#09 扩展**（原 #08 读出侧并入） | 未交付（原 #08 撤销） |
| S5 | 池级打分 | `kpool_indexer.py:554` / `:716` | **#09 → #09 扩展** | 基础形态已接入；扩展未交付 |
| S6 | top-512 选池 | `kpool_indexer.py:1078` | **#09 扩展**（原「无算子」） | 未交付 |
| S7 | 展开 + 残尾 | `kpool_indexer.py:583` / `:1084` | **#09 扩展**（原 #10 并入） | 未交付（原 #10 退出主路径） |
| S8 | 逻辑→物理槽 | `_custom_ops.py:946` | 已有算子 | 覆盖 |
| S9 | latent 写入 | `sparse_mla.py` | **#12** | 已接入 |
| S10 | 稀疏 attention | `sparse_mla.py:579` | **#11** | 已接入 |
| S11 | 页清零 | `patch_kv_block_zeroer.py` | **#18** | 已接入 |

## 7. 没有算子配上的段

按「本条链路里还有哪一段没有算子承接 / 还在付 host 往返」清点，共 5 处；
重划之后 5 处**全部有归属**，其中 3 处（2/3/4）在插件侧或并入其他单据即已完成。

1. **S0 pool + tail slot mapping** —— 曾经完全没有需求覆盖（原 #08 的 ABI 只收 `req_ids`/
   `token_offsets`，没有 metadata，覆盖不到这次 CPU 往返）。**2026-10-09 已独立提需求
   `#19 torch.dlc_kpool_slot_mapping`（P1）**，双输出、一次算完，golden 已与两处生产实现逐位对拍。
   上游对照：池槽是 Triton kernel（`vllm/vllm/v1/attention/backends/mla/compressor_utils.py:151`，
   kernel `:172`），tail 环槽是纯 PyTorch + `searchsorted`
   （`vllm/vllm/v1/attention/backends/mla/indexer.py:558`）。
2. ~~**S0 的重复计算（插件侧可先去重）**~~ —— **✅ 2026-10-09 已去重**：尾映射原本在 worker
   （`patch_block_table.py:14`）与 tail builder（`patch_mla_indexer.py:191`）各算一遍，
   输入相同、结果逐位相同（dlc:0 六组几何实测），现在 builder 直接复用 worker 写好的 `slot_mapping`，
   省掉 1 次 searchsorted 往返 ≈ 0.3 ms/步。回退开关 `DLC_KPOOL_TAIL_SLOT_REBUILD=1`，
   回归 `pytest -k "kpool or indexer or glm5"` **171 passed, 1 xfailed**。
3. ~~**S4 池读出**~~ —— **✅ 2026-10-09 并入 #09 扩展**：kernel 内按 block table 地址算术
   直接读 `pool_cache`，不再需要独立的 `dlc_kpool_pooled_read`（原 #08 可选读出侧随之撤销）。
4. ~~**S6 top-512 选池**~~ —— **✅ 2026-10-09 并入 #09 扩展**：`k_sel = min(512, visible)`
   在 kernel 内与打分/展开同一次调用完成，插件里的 `torch.topk` 逐行小算子删除。
5. **S2 tail 环写入** —— 需求（原 #08）已**撤销**，**✅ 2026-10-09 并入 #07 扩展**。
   实测 dlc:0、D=128，
   `_seed_kpool_fused` 单层 decode 0.82 ms / chunk(n=512) 1.37 ms，其中环写
   0.40 / 0.55 ms（×11 层 ≈ 4.4 / 6.1 ms/步），压缩 0.44 / 0.78 ms。
   环写里又有一半以上是插件的 `torch.div(..., rounding_mode="floor")` 慢路径：
   **✅ 2026-10-09 已改为 `//` + `torch.remainder`**（`kpool_indexer.py:678`），
   单层 `_seed_kpool_fused` 实测 0.886 → 0.596 ms（decode）、1.319 → 1.022 ms（chunk n=512），
   即省 2.6–3.3 ms/步，且结果逐位一致。剩下的 0.23 ms/层（两次 scatter + 3 个小 op）是 #07 扩展的净收益，
   约 1.6 ms/步；但 07 扩展的主收益不是这 1.6 ms，而是**清掉 forward 内的 kernel 外编排与 D2H**，
   所以仍按 P0 排期。

也就是说：重划之后，**S0 → #19，S2/S3 → #07 扩展，S4–S7 → #09 扩展**，
插件里不再有「没有算子配上」的段（S1 属框架侧通用算子）；这四单交付后，
`native_kpool_indexer_step` 内的 kernel 外编排可以整体删除。

建议（待确认，不在本文档落地）：

- ~~插件侧先把 `_seed_kpool_fused` 的 `torch.div(..., rounding_mode="floor")` 改成 `//`~~
  **✅ 2026-10-09 已完成**（`kpool_indexer.py:678`），实测省 2.6–3.3 ms/步。
- ~~插件侧先去重尾映射~~ **✅ 2026-10-09 已完成**（`patch_mla_indexer.py:191`），省约 0.3 ms/步。
- ~~S6 维持 `torch.topk`，不单独提需求~~ **✅ 2026-10-09 改为并入 #09 扩展**（在 kernel 内选池）。

## 8. 参考实现索引

| 环节 | 参考实现 |
|---|---|
| 池槽 / tail 槽映射（#19 golden） | `kpool_slot_mapping.py:18`、`patch_mla_indexer.py:119`、`kpool_worker_slots.py:28` |
| 池化压缩 | `vllm_cl/patch/models/kpool_indexer.py:474` `compress_kpool` |
| 池级打分 | `kpool_indexer.py:554` `kpool_scores` |
| 展开 + 残尾 | `kpool_indexer.py:583` `expand_pool_topk` |
| 有状态 append/read（refs 对拍用） | `kpool_indexer.py:349` `NativeKPoolPagedCache`（`append` `:404`、`read_pools` `:442`） |
| 每层执行主体 | `kpool_indexer.py:775` `native_kpool_indexer_step` |
| 插件 forward | `vllm_cl/patch/models/glm5_indexer.py:276` |
| pool slot mapping（上游 Triton） | `vllm/vllm/v1/attention/backends/mla/compressor_utils.py:151` |
| tail slot mapping（上游 PyTorch） | `vllm/vllm/v1/attention/backends/mla/indexer.py:558` |
| 512 稀疏 attention | `vllm_cl/attention/sparse_mla.py:579` |
