# `_fused_operand` 归因与消解方案（2026-10-09）

> 范围：`vllm_cl/patch/patch_kda.py` 里 `_fused_operand` 这一处 view→独立张量的物化。
> 只覆盖 `#04 dlc_kda_recurrent_decode` 的 decode 路径（`_recurrent_decode_new_op`，`patch_kda.py:464`）；
> prefill 侧 `chunk_kda_with_fused_gate_dlc` 走的是 4-D ABI，另有自己的物化点，不在本文。
> 相关：`性能优化点清单_2026-09-29.md`（P0-3 剩余归因）、`接入清单_2026-09-28.md:132`、
> `op_requests/04_dlc_kda_recurrent_decode.md`。

## 0. 结论（TL;DR）

- `_fused_operand`（`patch_kda.py:441`）是一个**纯 Python 兜底**：`return tensor.contiguous() if not tensor.is_contiguous() else tensor.clone()`。
  带出 9 个操作数 × 34 个 KDA 层 = **decode 每步 306 次 device 侧物化/下发**，是 decode 侧
  `copy_after_async` 归因里的头号条目（1,438 次中占 306）。
- **根因不是「没 pin」「没 contiguous」**，而是 pytorch 侧 `dlc_kda_recurrent_decode` 的入口契约：
  `KdaRecurrentDecode.cpp:72-79` 要求所有操作数 `contiguous + stride(-1)==1 + storage_offset()==0`，
  `:104-107` 还要求 `q/k/v/raw_g/recurrent_state/out` 满足 `dlc_dim0_padded_ * dlc_layout_ == 512`
  （**一条逻辑行 = 一条独立 512B 物理行**，且 `:104-107` 读的是 **父 storage** 的描述符）。
  模型侧这些张量全是 view，storage 描述符继承父张量（整条 3HD / HD / 合并 GEMM 输出宽），
  物理行宽不为 512B → 必须物化成新张量。
- **9 个操作数并不都需要物化**（2026-10-09 真机实测，`op_requests/probes/probe_kda_decode_operand_views.py`）：
  - **必须物化（硬失败）**：`q`/`k`/`v`（`:72` 不连续 / `:76` offset）、`raw_g`（`:107` 物理行）、`beta`（`:72`/`:76`）；
  - **只是省一次隐式 staging（能算对）**：`query_start_loc`、`state_indices`（视图形态 launcher 每次
    `to_contiguous_dlc` 拷一份，B=64 实测 +11~19 µs/call）；
  - **完全不需要**：`A_log`、`g_bias`（根参数，已直传，见 §4-A）。
- **原先没有独立方案文档**：内容散落在 `接入清单_2026-09-28.md:132`（契约说明）与
  `性能优化点清单_2026-09-29.md:261-311`（逐操作数归因表 + A~F 落地办法）。本文把两处归并成正式方案，
  并补上给算子组的 ABI 扩展 prompt（§6）。
- **消解路径（按各项的归属分三类；频次都是 ×34 层/步）**
  | 类 | 操作数 | 归属 | 收益 | 状态 |
  |---|---|---|---|---|
  | 静态参数 | `A_log`、`g_bias` | 插件侧直传（不物化） | −68/步（306→238） | ✅ 已落地 |
  | 每步元数据 | `query_start_loc` | C：ABI（常驻 buffer + `batch` 标量） | −34/步（238→204） | ⏳ 待 ABI（§3.4） |
  | 每层激活 | `q`/`k`/`v`、`raw_g`、`beta` | D：#04 ABI（5 项一起，**不动 #02**） | −170/步（204→34） | ⏳ 待 ABI（§3.3） |
  | 每步元数据 | `state_indices` | B：插件侧「每步 pack 一次」（kernel/ABI 不动），连带省掉 `clamp_min` | −34/步（34→0） | ✅ 已落地（§3.5，2026-10-09） |

  **A~D 做完 `_fused_operand` 可整体删除**，decode 侧该条目从 306 归零。注：306 = 9 个操作数 × 34 层，
  只算 decode 这一处调用点；`#02 conv update` 侧那两个 `contiguous()` 与 `state_indices` 的 `clamp_min` 不计在内。

  > 2026-10-09 修正：旧版写的「B → 170、C~F → 0」两处对不上（238 − 68 = 170，但 C~F 只消 170 却要
  > 清掉 6×34=204 次）。正确的口径是 `query_start_loc`（C）和 `state_indices`（B）各 −34/步。

## 1. 为什么必须物化（根因）

### 1.1 算子入口的两层硬校验

`pytorch/aten/src/ATen/native/dlc/KdaRecurrentDecode.cpp`：

| 行 | 校验 | 影响 |
|---|---|---|
| `:72-79` | 全部操作数 `is_contiguous()`、`storage_offset()==0`、`stride(-1)==1` | split/unsqueeze 出的 view 带 offset 或非 unit stride 时直接 `TORCH_CHECK` 抛错 |
| `:101-102` | 全部 `layout()==kStrided && stride(-1)==1` | — |
| `:104-107` | `q/k/v/raw_g/recurrent_state/out`：`storage->dlc_dim0_padded_ * storage->dlc_layout_ == 512` | **关键**：要求每条逻辑行的物理行宽恰好 512B |

### 1.2 为什么「torch 侧看着合法」仍然不合法

- DLC 的 `DLCStorageImpl` 记录了两个 Python 侧不可见的字段：`dlc_dim0_padded_`（storage 最低维的物理宽度）
  与 `dlc_layout_`（元素字节数）。kernel 按 DMA 行描述符取数，**一行 = 一条 512B 物理行**。
- view（`split` / `reshape` / `unsqueeze` / 前缀切片）只是改 torch 的 `sizes/strides`，**不改底层 storage**。
  于是：
  - `q = qkv.split(...)`：`is_contiguous()` 可能为 True（N==1 时）、`storage_offset()==0`，但 `dlc_dim0_` 是整条 `3HD`；
  - `beta = beta_raw.unsqueeze(0)`：`storage_offset != 0`（父是合并 GEMM 输出）；
  - `raw_g = f_b_proj(f_a)[0].reshape(1,-1,H,D)`：`dlc_dim0_` 是 `H*D`。
- **Python 侧拿不到 `dlc_dim0_padded_/dlc_layout_`，无法可靠判断 dlc-contiguity**，所以 `_fused_operand`
  统一物化。这也是它写成「无条件 clone」而不是「按 `is_contiguous()` 判断」的原因。

## 2. 现有实现与调用点

| 位置 | 作用 |
|---|---|
| （已删除）`_materialize_operand` / `prime_static_operand` / `static_operand_fp32` / `_static_operand_cache` | 2026-10-09 整条静态操作数「预物化 + 缓存」路径已删除；`A_log` / `g_bias` 现在在 `_recurrent_decode_new_op`（`patch_kda.py:541`）与 `_chunk_prefill_new_op`（`:632`）里直接 `reshape` 成 ABI 形状传下去 |
| `patch_kda.py:441` `_fused_operand` | decode 路径的兜底物化（调用点见下三行） |
| `patch_kda.py:529` | `qsl = _fused_operand(cu_seqlens.reshape(-1).to(torch.int32))` |
| `patch_kda.py:532` | `idx = _fused_operand(ssm_state_indices.reshape(-1).to(torch.int32))`，随后 `idx.clamp_min(0)` |
| `patch_kda.py:535-536` | `_fused_operand(q/k/v)`、`_fused_operand(g)`、`_fused_operand(beta)` |

另有 `_recurrent_decode_new_op` 内部的 `reshape(-1)` / `.to(torch.int32)` 条目，属同一处契约
（profiler 里并到该函数首行，因此按行号检索时以函数名为准）。

## 3. 九个操作数逐个定源（含 2026-10-09 真机实测）

「直接传的结果」一列是 `op_requests/probes/probe_kda_decode_operand_views.py` 在 dlc:0 上的实测：
**FAIL = 入口 TORCH_CHECK 直接拒绝（必须物化）**；OK = 校验过得去、数值也正确。

| 操作数 | 前一步（模型侧产生处） | 当前形态 | 直接传的结果（实测） |
|---|---|---|---|
| `q`/`k`/`v` | `qkv_ns = causal_conv1d_update(...)`（#02 输出 `[T,3D]`）→ `qkv_ns.split(local_projection_size, dim=-1)` → `_rearr(x)=x.reshape(1,-1,H,D)`（`vllm/models/glm5next/nvidia/kda.py:487,493`） | 3 个 view，storage 是整条 `3HD` | **FAIL**：T>1 时 `:72`（split 切片 `stride[1]=3HD≠HD`，`is_contiguous()=False`）；T=1 时 `k`/`v` 带 offset，挂 `:76` |
| `raw_g`（g） | `g1 = self.f_b_proj(f_a)[0]` → `g1.reshape(1,-1,H,D)`（`kda.py:315-316`） | reshape view，torch 连续、offset 0，但 storage 是按 `[T, H*D]` 分配的 | **FAIL**：`:104-107` `unsupported physical row layout`（`dlc_dim0_padded_*layout ≠ 512`；校验读的是**父 storage**描述符） |
| `beta` | `qkv, beta_raw, f_a, g_a = projected.split([...])` → `beta = beta_raw.unsqueeze(0)`（`kda.py:299,314`） | 合并 GEMM 输出的切片，`storage_offset != 0` 且切片本身不连续 | **FAIL**：切片版挂 `:72`（不连续），reshape 版挂 `:76`（nonzero storage offset） |
| `query_start_loc` | `attn_metadata_narrowed.non_spec_query_start_loc` → `cu_seqlens = ...[...]`（`kda.py:360`）；上游 `CommonAttentionMetadata.query_start_loc` 每步 `async_tensor_h2d`，`[N+1]` int32 | 常驻 `[max_num_reqs+1]` 的前缀切片 | **OK**（offset 0、stride 1），但 `is_contiguous_dlc=false` → launcher 每次调用 `to_contiguous_dlc` 隐式拷一份；B=64 实测 +12.7 µs/call |
| `state_indices` | `attn_metadata.non_spec_state_indices_tensor` ← `block_table_tensor[:, 0]`；`block_table_tensor = block_table.gpu[:num_reqs]`，mamba `none` 模式行宽 = 1（`kv_cache_interface.py: MambaSpec.max_num_blocks_per_req`）→ 常驻 `[128,1]` int32 | 1-D 列视图（stride=(1,)）或 `[B,1]` 切片；storage `dim0=1 / dim0_padded=128`（int32 alignment=128 元素=512B，每个请求索引独占一条 512B 行） | **OK**，但同样隐式 staging：B=64 实测 1-D 列视图 +11.2 µs/call、`[B,1]` 切片 +19.1 µs/call。只有 `B % 128 == 0`（切片长度正好等于 alignment）时 `is_contiguous_dlc` 才为真、真正零拷贝 |
| `A_log` | 层参数（静态）：`nn.Parameter(torch.empty(1,1,H,1, fp32))`（`kda.py:249`） | 根张量 → `reshape(-1)` 得到 `[H]` 纯 view | kernel 按**基址平铺**读（`dlc_kda_recurrent_decode.cpp:316` `tensor_slice(address, (head/128)*512B)`），**view 即可用，无需物化**（也不走任何预物化/缓存路径） |
| `g_bias` | `dt_bias = nn.Parameter(torch.empty(H*D, fp32))`（`kda.py:205`） | 根张量 → `reshape(H,D)` 纯 view，每行 128 fp32 = 512B | launcher 走纯连续分支（`DLCKernelLaunch.h:916`），行距 = 末维 128 元素 = 512B，与物理布局一致，**view 即可用，无需物化**（也不走任何预物化/缓存路径） |

### 3.1 为什么「物化」就能过

- `contiguous()` / `clone()` 走的是**重新分配**：新 storage 按 `[1,T,H,D]`（或 `[H]`、`[H,D]`）创建，
  `dlc_dim0_` = 最内维 = `D`（bf16 pad 到 256 元素 = 512B）→ `:104-107` 通过；同时 `storage_offset`
  归零、stride 规范化 → `:72` / `:76` 通过。
- view 只改 torch 的 sizes/strides，**不改 storage**；而 `:104-107` 查的正是 `tensor.storage()`
  的描述符，所以「torch 连续」不够（`raw_g` 就是被这条拦下的）。
- 结论：这里的「连续化」同时干了两件事 —— 去掉 offset/非连续，以及**换一个物理行宽正确的 allocation**。

### 3.2 两个副产物（2026-10-09 实测）

- **`query_start_loc` / `state_indices` 的隐式 staging 成本**：两者不物化也能算出正确结果，但每次
  kernel 调用 launcher 会 `to_contiguous_dlc` 拷一份（B=64 实测：qsl 视图 +12.7 µs/call，
  `state_indices` 1-D 列视图 +11.2 µs/call、`[B,1]` 切片 +19.1 µs/call）。
  这**修正了 P0-3 的 B 项前提**：`block_table[:B]`（`[B,1]`）只有在 `B % 128 == 0`（切片长度正好
  等于 int32 alignment）时 `is_contiguous_dlc` 才为真、才是零拷贝；一般 batch 下它与 1-D 列视图
  一样会被 staging，「每 step 省 34 次下发」的预期不成立。
- **kernel 边界缺陷 `B == 128`**：`dlc_kda_recurrent_decode.cpp` 用
  `use_metadata_cache = B <= kVectorWords`（`kVectorWords = 128`）决定走 cache，
  但 cache 只 `hbm_to_vmem(..., kVectorWords)` 装了 128 个 int32，随后要读第 `B` 个（即 `B+1` 个值）
  → `B == 128` 时越界读到相邻 VMEM 区 → `Assert(query_start_loc must cover [0, N])` 后 launch failure。
  实测 B=126/127/129 正常，**只有 B==128 挂**。修法：条件改为 `B + 1 <= kVectorWords`。
  该缺陷随本方案的 ABI 需求一起提给算子组（见 §6 第 3 条）。

### 3.3 五项激活类操作数的消解办法：给 #04 加 `(base, offset, pitch)` 取数（2026-10-09 定稿）

**不改 `#02`**。曾评估过「让 `dlc_kda_causal_conv_update` 直接输出 3 个独立的 `[1,T,H,D]`」，
但那条路要求：

- 把 `-> Tensor` 改成 `-> (Tensor, Tensor, Tensor)`；
- 新增 `num_heads` 标量 —— `qkv[...,3*H*D]` / `weight[4,3*H*D]` / `conv_state[...,3*H*D]` 里
  只有 `D_total = 3*H*head_dim`，**推不出 `H`**，没有这个参数就分配不出 `[B,T,H,D]`（`head_dim` 可由
  `D_total/3/H` 反推）；
- kernel 输出行距从 `ALIGN256(D_total)` 改成 `ALIGN256(head_dim)`（`kda_causal_conv_update.c:129,289-291`），
  并要处理 3 份 `zeros` 的 null-slot 语义。

改动面明显更大，因此**把口子只开在 `#04`**：q/k/v/raw_g/beta 五项统一改成
「传父张量的 view + 每项一个取数描述」。

**机制**：`line_addr = base + offset + line * pitch`

- `base` = 父 storage 基址（`tensor.storage().data_ptr()` 语义）；
- `line = token * H + head`（`beta_logits` 用 `line = token`，`offset` 指向该 token 的 beta 段起点）；
- `offset` / `pitch` 以**元素**计；Python 侧直接给 `tensor.storage_offset()` 与 `tensor.stride(-2)`，
  不需要插件做任何额外推导。

现在 `#04` 把行距写死成描述符的 512B（`DLC_Custom_Kernel/dlc_kernels/attention/kda/dlc_kda_recurrent_decode.cpp:33-35`
的 `tensor_line`：`line * input->dim0_padded512 * 4`），而这五项在模型里全是父张量的 view，
行距由父张量决定 —— 这就是它们必须物化的直接原因。

**模型真实形态**（TP8：`H_local=8`、`head_dim=128`、`D_total=3072`；`T` = 本步 token 数）

| 操作数 | 父张量 | `offset`（元素） | `pitch`（元素） | 每 token 有效段 |
|---|---|---|---|---|
| `q` | `#02` 输出 `[T,3072]` | 0 | 3072 | `[0:1024]`，2048B 连续 |
| `k` | 同上 | 1024 | 3072 | `[1024:2048]`，2048B 连续 |
| `v` | 同上 | 2048 | 3072 | `[2048:3072]`，2048B 连续 |
| `raw_g` | `f_b_proj` 输出 `[T,1024]` | 0 | 1024 | 整行，2048B 连续 |
| `beta_logits` | `projected` `[T,3336]`（合并 GEMM：3072+8+128+128） | 3072 | 3336 | 前 8 个 bf16，16B |

**推荐 kernel 按「每 token 整段」读，而不是按 (token,head) 512B 行读**

- 每个 token 的 `[H*head_dim]` 段在父张量行内本来就是连续的 → 一次 2048B DMA 即可。
  q/k/v 每 token 的段起点分别是 0 / 2048B / 4096B，raw_g 是 0，**全部 512B 对齐**；
  段长 2048B 也是 512B 的整数倍 → 完全落在你们芯片友好的对齐上。
- 每 token 的这些 DMA 从 `4×H`（q/k/v/raw_g 各 8 次 = 32 次）降到 4 次，且读的全是有效字节。
- 若算子组坚持保留 (token,head) 行模型、只把 pitch 参数化，有两个坑要一起解决：
  ① q 段内第 2 个 head 起就落在 256B 边界（对非 512B 对齐敏感的芯片可能明显变慢）；
  ② DMA 仍读 512B，最后一个 (t,h) 会越过张量尾 256B —— `#02` 的输出是紧凑分配、没有尾部冗余，
  必须单独处理末行（或改小最后一次 DMA）。

### 3.4 `query_start_loc` 的消解办法：传常驻 buffer + `batch` 标量（2026-10-09 定稿）

**证据链：上游本来就有常驻 buffer，整条传下去是零拷贝**

| 位置 | 内容 |
|---|---|
| `vllm/vllm/v1/worker/gpu_model_runner.py:824` | `self.query_start_loc = self._make_buffer(self.max_num_reqs + 1, dtype=torch.int32)` —— 常驻设备 buffer |
| `gpu_model_runner.py:2490` | `CommonAttentionMetadata(query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1], ...)` —— 传下去的是它的**前缀切片** |
| `vllm/vllm/v1/attention/backends/gdn_attn.py:259` | `non_spec_query_start_loc = query_start_loc`（原样透传） |
| `vllm/models/glm5next/nvidia/kda.py:583` | `cu_seqlens = non_spec_query_start_loc[: num_decodes + 1]` |

view 链只有 narrow，所以 `cu_seqlens._base` 就是 `self.query_start_loc.gpu`。整条 buffer
**一定是 dlc-contiguous**：`sizes.back() == dlc_dim0_`（整条、零偏移），命中
`pytorch/aten/src/ATen/dlc/DLCTensorImpl.h:56-92` 里「`dlc_dim0_` 不对齐时退化要求
`sizes.back() == dlc_dim0_`」那一支。

**现在为什么还是拷**

- `patch_kda.py:528` `qsl = _fused_operand(cu_seqlens.reshape(-1).to(torch.int32))`：`.to(int32)`
  是空操作（上游 buffer 就是 int32），真正的开销是 `_fused_operand` 的 clone。
- 不 clone、直接传 view 的话，launcher 会隐式 staging（B=64 实测 +12.7 µs/call）——
  前缀切片 `[B+1]` 的 `sizes.back() != dlc_dim0_` → `is_contiguous_dlc == false`。
- 两种走法都是每层每步一次 ≈260B 的拷贝，×34 层 ≈ **432 µs/步**。

**改法（逐处）**

| 位置 | 改动 |
|---|---|
| ABI（`native_functions.yaml:465`，`dlc_kda_recurrent_decode`） | `query_start_loc` 允许传长度 ≥ `batch+1` 的常驻 buffer；新增 int 标量 `batch`（kernel 侧 `scalars_` 再加一个槽位） |
| kernel `dlc_kda_recurrent_decode.cpp:216` | `int B = query_start_loc_->shape[0] - 1;` → `B = scalars_[3]` |
| kernel `:236` | `use_metadata_cache = B <= kVectorWords` → `B + 1 <= kVectorWords`（**顺带修掉 `B == 128` 越界挂死**，见 §3.2） |
| kernel `B+1 > 128` 分支 | 现在走 `load_i32_scalar` 逐索引、每个索引一次 512B DMA（`request` 与 `request+1` 共 2B 次）；改成一次 `hbm_to_vmem(ALIGN128(B+1))` 装进 VMEM 再按索引取 |
| 插件 `patch_kda.py:528` | `qsl = cu_seqlens._base`（`_base is None` / 形状不符时回落 `_fused_operand`），并传 `batch = cu_seqlens.numel() - 1` |

**注意点**

- capture/padded 分支（`gdn_attn.py:489-494`）用的是另一条常驻 buffer
  `self.non_spec_query_start_loc[: batch_size + 1]`，同样有 `_base`；而且 `batch_size + 1` 就是整条长度，
  本来就是零拷贝，不需要特殊处理。
- 同一批里的同类拷贝：`patch_kda.py:532` 的 `idx = idx.clamp_min(0)`（PAD_SLOT_ID(-1) → 0 的归一化）
  会再分配一份。若 kernel 直接接受 `-1` 当 null slot（而不是触发 Assert），这个 copy 也能去掉
  → 并入同一条 ABI 需求（见 §6 第 4 条）。
- 收益：每层每步省一次 12.7 µs 级的下发/拷贝，34 层 ≈ 432 µs/步，同时消掉 `B == 128` 的挂死路径。

### 3.5 `state_indices` 的消解办法：把 pack 从「每层一次」提到「每步一次」（2026-10-09 定稿）

**kernel 侧的契约比 `query_start_loc` 宽松：batch 不看它的 shape**

- `dlc_kda_recurrent_decode.cpp:236-241`：`shape[1] == 1`（1-D）时走 cache —— 一次
  `hbm_to_vmem(..., kVectorWords)` 读 512B = 128 个 int32，之后 `load_i32_cached(vmem_indices_cache, request)`
  按 word 取。
- **batch 来自 `query_start_loc`**（`:216` `int B = query_start_loc_->shape[0] - 1`），不是从 `state_indices` 的
  shape 推的。所以 kernel 语义上「传一条比 B 更长的 1-D buffer、只读前 B 个 word」是合法的；
  能不能零拷贝取决于 launcher 的 contiguity 判定（见下面「落地要点」第一条）。
- 1-D 分支的语义就是「这条 512B 行里的 word i = 第 i 个请求的页号」—— 正是 pack 出来的紧凑布局。

**为什么现在还是要拷**

`patch_kda.py:536-544`：

```python
idx = _fused_operand(ssm_state_indices.reshape(-1).to(torch.int32))
idx = idx.clamp_min(0)
```

每个 KDA 层各做一次 clone + 一次 clamp（2 次下发 × 34 层/步 ≈ 750 µs/步）。源头
`non_spec_state_indices_tensor` 是**二维常驻 block table 的列视图**（`gdn_attn.py:257` 的
`block_table_tensor[:, 0]`；mamba `none` 模式下行宽 = 1，但每个元素物理上独占一条 512B 行），
直接传会被 launcher staging，所以现在只能拷。

**候选路与取舍**

| 路 | 做法 | 结论 |
|---|---|---|
| ① 插件侧「每步 pack 一次」 | 在每步只跑一次的钩子里（`vllm_cl/worker/dlc_model_runner.py` 的 `_build_attn_group_metadata`，`builder.build(...)` 之后）把 `block_table[:, 0]` 的 B 个页号拷进一条长度 = B 的 1-D int32 张量，顺带 `clamp_min` 一次；34 层共享 | ✅ **定稿**。不动 kernel、不动 ABI、不动 vllm core；每步 2 次下发替代 68 次 |
| ② 上游无条件 pack | 把 `gdn_attn.py:478-494` 那套（现在门控在 `use_full_cuda_graph`）对纯 decode 打开 | ⚠️ 动 vllm core、影响其他 mamba 模型；而且它传下来的是 `[:batch_size]` 前缀切片（`batch_size = m.num_reqs`，`:427`），仍会被 staging，还要再改成整条 |
| ③ kernel 直接读二维 block table | 传二维常驻 buffer + 行距 | ❌ kernel 的二维路径是**每请求一次 512B DMA 且带 sync**（`:262` `load_i32_scalar(tensor_line(...))`），每层 B 次串行 DMA，比一次 pack 贵得多 |

**落地要点（按 ①）**

- pack 出来的张量必须**正好是长度 B 的 1-D 连续张量**：`is_contiguous_dlc` 要求 `sizes.back() == dlc_dim0_`，
  所以不能传常驻 buffer 的前缀切片（`[:B]` 只在 `B % 128 == 0` 时才零拷贝，同 §3.2 的结论）。
  代价是每步一次分配（P1-2 图捕获时再改成常驻 + 整条直传）。
- pad 行本来就是 `NULL_BLOCK_ID(0)`：`dlc_model_runner.py:2285` 已经把
  `blk_table_tensor[num_reqs:num_reqs_padded]` 填成 `NULL_BLOCK_ID`，而 `B` 取自 `query_start_loc`
  （= `num_reqs_padded`，会覆盖到 pad 行），所以 pack 直接搬即可、**不需要**额外 fill；
  kernel 对 slot 0 输出全零、state 不写回。
- 这条**不需要**给 `state_indices` 加 `batch` 标量。
- **覆盖范围（2026-10-09 落地时核实）**：本次只覆盖 decode 的 `#04`（`patch_kda.py:536-544`），
  即每步 34 层那一处。同文件另两处同类拷贝**没有**一并省掉，各有原因，留作后续：
  - `:233-235`（`#02` conv update）：模型侧传的是切片 `non_spec_state_indices_tensor[:num_decodes]`
    （`kda.py:476-478`），**是新对象**、看不到挂在原张量上的标记；而 `_update_new_op` 又要求
    `idx.numel() == qkv.shape[0]`，不能改成直传整条。要省只能让模型侧不再切片（等于改 `#02` 的调用形态），
    与「不改 `#02`」的定稿冲突，故不动。
  - `:181-184`（`#01` conv prefill）：传的确实是同一个对象（`kda.py:469` 的
    `cache_indices=non_spec_state_indices_tensor`），但走的是 prefill 步，当前守卫 `num_prefills == 0`
    把它排除了。prefill 不是本次延迟瓶颈，且只在 prefill 步生效（34 层/步），暂不纳入；
    若要纳入，只需把守卫放宽成「非 spec」（保留 `num_spec_decodes == 0`）并在 `_chunk_prefill_new_op`
    里加同样的标记判断。
- 边界：`B > 128` 时 1-D cache 只有 128 个 word，kernel 会退回逐请求 `load_i32_scalar`（正确但慢）。
  是否把 cache 扩成多行，随 §6 一起提（注意 `:236` 那个 `use_metadata_cache` 同时管 `query_start_loc`
  和 `state_indices`：前者要 `B + 1 <= 128`，后者只要 `B <= 128`，请拆成两个条件）。

**收益**：每层少 3 次 host 下发（`_fused_operand` 的 clone、`clamp_min`、以及原先那句 `.to(torch.int32)`），
换成每步一次的 pack（一次分配 + 一次 d2d copy，无额外 fill）。
`test_prepacked_adapter_issues_fewer_tensor_dispatches` 的确定性计数（B = 2）：
兜底 `{clone: 7, clamp_min: 1, to: 2}` → 预 pack `{clone: 6, clamp_min: 0, to: 1}`。

时间上的收益**只在小队列/空载时能直接量到**：空载冷态单层约省 25–55 µs（`dlc:6`/`dlc:7`，B = 2），
34 层 ≈ 0.9–1.9 ms/步；一旦下发队列被前面几十个算子塞满、host 变成等待队列，每层那两三次
小算子的差别在 host 计时上就看不出来。因此端到端的瓶颈是否真的下移，以模型服务的 profiler
条目数为准（待复核）。

**落地记录（2026-10-09）**

| 文件 | 改动 |
|---|---|
| `vllm_cl/worker/dlc_model_runner.py` | 新增模块级 `_pack_kda_state_indices(attn_metadata_i)`；在 `_build_attn_group_metadata` 里 `builder.build(...)` 之后、写给该 group 各层之前调用，且用 `if not for_cudagraph_capture` 门控（图捕获时新分配地址会被固化进图，保持原样走逐层兜底） |
| `vllm_cl/patch/patch_kda.py` | 新增常量 `KDA_STATE_INDICES_PACKED_ATTR = "_kda_state_indices_packed"`；`_recurrent_decode_new_op` 改为「有标记就 `packed_idx.reshape(-1)` 透传，否则原路 `_fused_operand(...).clamp_min(0)`」 |
| `tests/test_kda_state_indices_pack.py` | 新增：打包内容/独立缓冲/幂等/长度、6 条守卫分支（非张量、非 dlc、空、prefill、spec、非 decode）、真机属性语义 |
| `tests/test_glm5_kda_recurrent_decode_op.py` | 新增 `test_adapter_passes_through_prepacked_state_indices`：预 pack 路径与逐层兜底**逐位一致**，且用计数版 `_fused_operand` 证明索引张量不再被 clone；新增 `test_prepacked_adapter_issues_fewer_tensor_dispatches` 确定性量化下发次数 |

实现要点/已核实：

- 标记挂在 `attn_metadata_i.non_spec_state_indices_tensor` 这个**张量对象**上，元数据字段本身不改写 —— native 兜底（`patch_kda.py:_run_kda` 的 `if idx < 0: continue # PAD_SLOT_ID`）和 prefill 侧读的是同一字段，需要保留负值语义。
- 判空必须写 `getattr(..., None) is not None`：属性值就是张量，多元素张量不能求布尔（写 `if getattr(..., False):` 会直接 `RuntimeError`）。
- 真机实测（`dlc:0`）：`is_dlc` 只读不能 setattr；自定义属性在同一对象上可见；`t[:n]` / `reshape(-1)` / `clone()` 都是新对象、**看不到**该属性。因此
  - 上游每条路径（`gdn_attn.py:257` 的 `block_table_tensor[:, 0]`、`:330` 的布尔掩码索引、`:478-494` 的常驻 buffer 前缀切片）每步都产生**新的 view 对象**，属性不可见 → 每步必然重新 pack，不会拿到陈旧索引；
  - GDN builder 的 `supports_update_block_table` 继承自 `AttentionMetadataBuilder`（`False`），所以 `cached_attn_metadata` 复用分支对 GDN 不生效，元数据每步重建。
  - 这条前提已用设备用例 `test_dlc_tensor_attr_semantics` 锁住。
- 未覆盖的边界（保持原样，不影响正确性）：图捕获步骤不 pack（仍走逐层兜底）；`B > 128` 时 kernel 退回逐请求 DMA（正确但慢，待 §6 的 cache 拆分）。
- **下发次数（确定性断言）**：每层 −3 次（`clone` / `clamp_min` / `to`），见上文计数。
- **时间（仅供参照，不做回归断言）**：空载冷态单层省 25–55 µs（B = 2）；队列拥塞时 host 计时区分不出。
  精确的 decode 侧条目数下降仍待模型服务 profiler 复核（需用户起服务）。

## 4. 落地办法 A~D

| # | 操作数 | 办法 | 归属 | 状态 |
|---|---|---|---|---|
| A | `A_log` / `g_bias` | **不做任何物化/缓存**：直接 `reshape` 成 ABI 形状（`[H]` / `[H,D]`）传下去 | 插件侧 | ✅ 已落地。**为什么不用物化**：这两个是根 `nn.Parameter`（`kda.py:205` dt_bias、`:249` A_log，`torch.empty` 分配 + `param.data.copy_` 原位载入），reshape 出的 view 本身就是 dlc-contiguous（dlc:0 上 heads=8/16/32/64/128 实测 view 直传与 clone 后直传的 out/state 逐位一致，见 §5 验收）；收益来自「每步每层不再走 `_fused_operand`」（306→238）。2026-10-09：`_materialize_operand` / `prime_static_operand` / `static_operand_fp32` / `_static_operand_cache` 整条路径已删除 |
| B | `state_indices` | 插件侧**每步 pack 一次**：在 `vllm_cl/worker/dlc_model_runner.py::_build_attn_group_metadata`（`builder.build(...)` 之后）把 `block_table[:, 0]` 的 B 个页号拷进长度 = B 的 1-D int32 张量、顺带 `clamp_min`；34 层共享，删除 `patch_kda.py:536-544` 的 `_fused_operand` + `clamp_min`（**仅 decode 的 `#04`**；`#01`/`#02` 两处的取舍见 §3.5 的「覆盖范围」） | 插件侧 | ✅ 已落地 2026-10-09（kernel/ABI 不动）。见 §3.5 |
| C | `query_start_loc` | `cu_seqlens._base` 取 `self.query_start_loc.gpu`（常驻 `[max_num_reqs+1]` int32），**整条**传下去 + kernel 新增 `batch` 标量（现在由 `size(0)-1` 推） | 算子组 ABI（#04）+ 插件侧切换 | ⏳ 待 ABI：已提单 [`op_requests/04b_dlc_kda_recurrent_decode_abi.md`](op_requests/04b_dlc_kda_recurrent_decode_abi.md)（2026-10-09）。证据链与逐处改法见 §3.4 |
| D | `q`/`k`/`v`/`raw_g`/`beta`（5 项合并成一条） | #04 按 `(base, offset, pitch)` 取数（`line_addr = base + offset + line*pitch`），调用方直接传父张量的 view；**不改 `#02`** | 算子组 ABI（#04）+ 插件侧切换 | ⏳ 待 ABI：已提单 [`op_requests/04b_dlc_kda_recurrent_decode_abi.md`](op_requests/04b_dlc_kda_recurrent_decode_abi.md)（2026-10-09）。机制、真实数值与推荐实现见 §3.3 |

补充说明：

- **B 的 pack 位置**：放「每步只跑一次」的钩子里（`_build_attn_group_metadata`），而不是模型层 `kda.py`
  （那是每层调用）。`state_indices` 在非 capture 路径上是 `block_table_tensor[:, 0]`（二维常驻 buffer 的列视图，
  `gdn_attn.py:257`），capture 路径上是 1-D 常驻 buffer的前缀切片（`gdn_attn.py:478-494`，且
  `batch_size = m.num_reqs`，`:427`）——后者同样只是前缀切片，直接传也会 staging，所以两条路都走同一个 pack。
  prefill 的 `gather_initial_states_dlc` / `scatter_states_dlc` 仍要求 1-D，不在本次范围（归 #05）。
- **C 的 `batch` 标量必须有**：传整条常驻 buffer 后无法再从 `size(0)` 推出有效长度。
  `cu_seqlens._base` 拿不到时（例如测试里自造的临时 int32 张量）仍退回 `_fused_operand`；
  capture/padded 分支用的是另一条常驻 buffer，同样有 `_base`，无需特殊处理（见 §3.4）。
- **D 不碰 `#02`**（2026-10-09 定稿）：让 `#02` 直接输出 3 个 `[1,T,H,D]` 需要新增 `num_heads` 标量
  （ABI 里推不出 `H`）并重写输出行距与 null-slot 置零语义，改动面更大；改成只在 `#04` 放开取数后，
  `#02` 的 ABI、kernel、模型侧调用全部不动。
- **D 的实现形态决定收益**：推荐「每 token 读整段 `H*head_dim`」（q/k/v/raw_g 各 1 次对齐 DMA）；
  若走 (token,head) 行模型 + pitch 参数化，会有 256B 起点与末行越读两个问题，见 §3.3。

## 5. 收益与验收

| 阶段 | decode 每步 `_fused_operand` 次数 | 说明 |
|---|---|---|
| 现状（A 已落地） | 238 | 7×34（`qsl`/`idx`/`q`/`k`/`v`/`raw_g`/`beta`） |
| + C | 204 | 再 −1 ×34（`query_start_loc`） |
| + D | 34 | 再 −5 ×34（`q`/`k`/`v`/`raw_g`/`beta`） |
| + B | **0** | 再 −1 ×34（`state_indices`，插件侧每步 pack 一次），`_fused_operand` 整体删除 |

验收方法：

- **B（已落地，2026-10-09）**：`tests/test_kda_state_indices_pack.py` 12 例 —— 打包内容/独立缓冲/幂等/长度、
  6 条守卫分支、以及真机属性语义（`view`/`reshape`/`clone` 不继承标记，保证每步都会重新 pack 而非读到陈旧索引）；
  `tests/test_glm5_kda_recurrent_decode_op.py::test_adapter_passes_through_prepacked_state_indices` ——
  预 pack 路径与逐层兜底**逐位一致**（`torch.equal`），并用计数版 `_fused_operand` 断言索引张量不再被 clone。
  早前的形状探针 `docs/glm5_flash/op_requests/probes/probe_kda_state_index_layout.py`（`clone(bt[:B,0])` /
  `bt[:B]` / `bt[:B,0]` 三种传法对 #04、#02 逐位一致）仍是 pack 布局的依据。
  仍待真机确认：`num_reqs_padded > num_reqs` 时 pad 行落到 slot 0（kernel 侧本来就输出全零、state 不写回）。
  运行：
  `cd /root/workspace/vllm-cl && DLC_TEST_DEVICE=dlc:0 /root/miniconda/envs/qingluo-dev1/bin/python -m pytest tests/test_kda_state_indices_pack.py tests/test_glm5_kda_recurrent_decode_op.py -q`
- **C / D**：`tests/test_glm5_kda_recurrent_decode_op.py` 现有 19 例全绿；新增「父张量 view 直传」用例
  （5 个操作数各按 §3.3 的真实 `(offset, pitch)` 形态构造），与 `_fused_operand` 版本逐位一致；
  profiler 里 decode 侧 `copy_after_async` 不再出现 `_recurrent_decode_new_op` 相关条目。
- 回归：`pytest tests/ -k "kda or glm5"`。

## 6. 给算子组的 ABI 扩展 prompt（#04，可复制）

> **本节的正式 issue 文档已单独成文**：[`op_requests/04b_dlc_kda_recurrent_decode_abi.md`](op_requests/04b_dlc_kda_recurrent_decode_abi.md)（2026-10-09）。
> 下面是原始 prompt，保留作对照；两者若不一致，以 issue 文档为准。

> 请在 `torch.dlc_kda_recurrent_decode` 上扩展操作数 ABI，使调用方可以直接传父张量的 view，
> 不需要在 Python 侧先物化成独立 512B 行张量。现状校验见
> `pytorch/aten/src/ATen/native/dlc/KdaRecurrentDecode.cpp:72-79`（`contiguous` + `stride(-1)==1`
> + `storage_offset()==0`）与 `:104-107`（`dlc_dim0_padded_ * dlc_layout_ == 512`，即「一条逻辑行
> = 一条独立 512B 物理行」）。**本需求不要求改动 `dlc_kda_causal_conv_update`**，它的 ABI 与输出形状
> 保持原样。
>
> 需要放开/处理以下四条：
>
> 1. **`q` / `k` / `v` / `raw_g` / `beta_logits`（5 项）**：接受父张量 view + 每项一个取数描述
>    `(base, offset, pitch)` —— kernel 内 `line_addr = base + offset + line * pitch`，
>    `line = token*H + head`（`beta_logits` 用 `line = token`，`offset` 指向该 token 的 beta 段起点）。
>    `offset` / `pitch` 以元素计，调用方直接给 `tensor.storage_offset()` 与 `tensor.stride(-2)`。
>    校验放宽为：保留 `stride(-1) == 1`，允许非零 `storage_offset`，并**删除这 5 项上的
>    「一行正好 512B」要求**（`:104-107` 对 `recurrent_state` / `out` 的校验保留）。
>
>    **推荐实现：按 token 一次读整段 `H*head_dim`，而不是按 (token,head) 512B 行读。**
>    模型侧每个 token 的 `[H*head_dim]` 段在父张量行内是连续的，q/k/v 的段起点分别是
>    0 / 2048B / 4096B，raw_g 是 0 —— 全部 512B 对齐，段长 2048B 也是 512B 整数倍；
>    这样每 token 的 DMA 从 `4×H`（=32 次）降到 4 次，且不会越过张量尾部。
>    若保留 (token,head) 行模型、只把 pitch 参数化：pitch 会变成 256B（第 2 个 head 起不在 512B 边界），
>    且最后一次 DMA 会越过张量尾 256B（父张量是紧凑分配，无尾部冗余），需要单独处理末行。
>
>    模型侧真实数值（TP8：`H_local=8`、`head_dim=128`、`D_total=3072`，`T` = 本步 token 数）：
>
>    | 操作数 | 父张量 | `offset`（元素） | `pitch`（元素） |
>    |---|---|---|---|
>    | `q` | `#02` 输出 `[T,3072]` | 0 | 3072 |
>    | `k` | 同上 | 1024 | 3072 |
>    | `v` | 同上 | 2048 | 3072 |
>    | `raw_g` | `f_b_proj` 输出 `[T,1024]` | 0 | 1024 |
>    | `beta_logits` | `projected` `[T,3336]` | 3072 | 3336 |
>
> 2. **`query_start_loc`**：允许传**完整的常驻 buffer**（长度 ≥ `batch+1`；调用方传的就是
>    `vllm/vllm/v1/worker/gpu_model_runner.py:824` 那条 `self.query_start_loc.gpu`，整条本身
>    dlc-contiguous、零拷贝），并新增一个 int 标量 `batch` 表示有效长度（有效值 = 前 `batch+1` 个元素），
>    替代现在从 `query_start_loc_->shape[0]-1` 推 batch 的做法。对应改动：
>    - `dlc_kda_recurrent_decode.cpp:216`：`int B = query_start_loc_->shape[0] - 1;` → 读新标量；
>    - `:236`：`use_metadata_cache = B <= kVectorWords` → `B + 1 <= kVectorWords`；
>    - `B+1 > 128` 分支：现在每个索引一次 512B DMA，请改成一次
>      `hbm_to_vmem(ALIGN128(B+1))` 装进 VMEM 再按索引取。
>    校验改为 `query_start_loc.size(0) >= batch + 1`。
>
> 3. **`use_metadata_cache` 那个条件要拆开、并修掉 `B == 128` 的越界**：`:236` 用同一个
>    `use_metadata_cache = B <= kVectorWords` 同时管 `query_start_loc` 和 `state_indices` 的 cache，
>    但两者的正确条件不同 —— `query_start_loc` 要读 `B+1` 个值（`B+1 <= kVectorWords`），
>    `state_indices` 只读 `B` 个（`B <= kVectorWords`）。现在的 `B <= kVectorWords` 让
>    `B == 128` 时越界读到相邻 VMEM 区 → `Assert(query_start_loc must cover [0, N])` 后 launch failure
>    （实测 B=126/127/129 正常，**只有 B==128 挂**）。请拆成两个条件，`query_start_loc` 用
>    `B + 1 <= kVectorWords`，`state_indices` 保持 `B <= kVectorWords`。
>    同时 `B+1 > 128` 时 query cache 要按 `ALIGN128(B+1)` 装（见第 2 条），`state_indices` 在
>    `B > 128` 时也要能多缓存几行（现在是退回逐请求 `load_i32_scalar`，正确但慢）。
>
> 4. **（可选）`state_indices` 接受 `-1`（PAD_SLOT_ID）当 null slot**：现在调用方为了避免 kernel 内 Assert，
>    必须先做一次 `idx.clamp_min(0)`（`vllm_cl/patch/patch_kda.py:532`）。本方案已把这次 `clamp_min`
>    收敛到每步一次（§3.5），所以这条**不阻塞**；若一并实现，插件侧可以直接传原始 `state_indices`。
>
> 说明：`state_indices` 的拷贝**不在本需求范围内** —— 按 §3.5，插件侧每步 pack 一次即可解决，
> kernel 与 ABI 都不用动（第 4 条只是可选优化）。
>
> 语义、数值结果必须与现有独立张量路径**逐位一致**；改完后 Python 侧 `_fused_operand`
> （`vllm_cl/patch/patch_kda.py:441`）整体删除。回归用例：
> `tests/test_glm5_kda_recurrent_decode_op.py`，以及按上表真实 `(offset, pitch)` 构造的 5 个操作数 view 形态各跑一遍；
> `query_start_loc` 另加「整条常驻 buffer + `batch`」与现有 `[:batch+1]` 前缀切片逐位一致的用例，
> 并覆盖 `batch = 128`（原挂死点）与 `batch > 128`（两行 cache / 整段 DMA 路径）。

## 7. 风险

- **C 的 `batch` 标量 + VMEM**：传整条常驻 buffer 后 kernel 无法再从 `size(0)` 推 batch，标量必须有；
  同时 query cache 要按 `ALIGN128(batch+1)` 分配（`max_num_reqs ≤ 255` 时最多 2 行 = 1KB），
  kernel 里的 `required_words` 断言要同步加上（见 §3.4）。
- **C 的 `_base` 兜底**：`cu_seqlens._base is None`（例如测试里自造的临时 int32 张量）或形状不符时仍退回
  `_fused_operand`，不能假定一定有常驻父张量。capture/padded 分支（`gdn_attn.py:489-494`）用的是
  `self.non_spec_query_start_loc[:batch_size+1]`，同样有 `_base`，且本来零拷贝，无需特殊处理（见 §3.4）。
- **B 的 pad 行**：pack 出来的 buffer 是长度 B（= `num_reqs_padded`）的紧凑数组，`[num_reqs, B)` 必须填
  `NULL_BLOCK_ID(0)`，否则 pad 行会写到别人的 state 池。每步 `fill_` 一次即可；也可以用
  `torch.zeros(B)` + `copy_`（多一次分配）。
- **B 的 per-step 分配**：方案 ① 每步新分配一条 `[B]` 张量（地址每步变），P1-2 图捕获时要改成常驻 buffer；
  届时若仍要零拷贝，就得走「整条常驻 buffer + `batch` 标量」，但 kernel 的 1-D 分支现在用
  `shape[0] != B` 判别（conv kernel `kda_causal_conv_update.c:79-83`）—— 传整条会被判成二维形态，
  所以到时候要连 kernel 的判别条件一起改。
- **B 的 cache 边界**：`B > 128` 时 1-D cache 只有 128 个 word，kernel 退回逐请求 `load_i32_scalar`
  （正确但慢）；且 `:236` 的 `use_metadata_cache` 同时管两个操作数，按 §6 第 3 条拆条件时不要顺手
  把 `state_indices` 卡到 `B + 1 <= 128`。
- **D 的 DMA 形态**：若算子组按 (token,head) 行模型 + pitch 参数化实现，会出现 256B 起点（对齐敏感，
  可能变慢）与末行越读 256B 两个问题；按「每 token 整段读」实现则两者都不存在。
  落地前用单层 micro-bench 对比两种形态的 kernel 时间。
- **D 的父张量生命周期**：`q/k/v` 的父张量是 `#02` 的输出、`beta` 的父张量是合并 GEMM 的输出，
  都是每步新分配的临时张量；传 view 后 kernel 只读不写，无需额外保活，但插件侧不能在调用前释放父张量。
- **`B == 128` 缺陷**：未修之前，batch 正好 128 会 poison 设备；插件侧暂不加入形状门槛，
  等算子组按 §6 第 3 条修复。
