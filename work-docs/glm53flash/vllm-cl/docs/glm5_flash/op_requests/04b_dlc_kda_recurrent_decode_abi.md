<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-10-09；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求类型：扩展已有算子能力（沿用现名，不新增算子、不改计算语义） -->
<!-- 关联：#04 04_dlc_kda_recurrent_decode.md（功能/精度已修复并接入，本条是它的 ABI 扩展）、
     _fused_operand_方案_2026-10-09.md §3.3 / §3.4 / §6、性能优化点清单 P0-3 -->
<!-- 一句话：让 #04 直接吃父张量的 view 与整条常驻 query_start_loc，删掉插件侧每层每步
     204 次仅为满足「独立 512B 行」校验而做的物化拷贝。 -->

# [Operator Request] dlc_kda_recurrent_decode —— 接受父张量 view + 常驻 query_start_loc（ABI 扩展）

## 需求类型
扩展已有算子能力

不改 `torch.dlc_kda_recurrent_decode` 的算子名、参数顺序、计算公式与输出形状；只做两类放宽：

1. 5 个激活类操作数（`q`/`k`/`v`/`raw_g`/`beta_logits`）接受**父张量的 view**，行地址按
   `(offset, pitch)` 计算，不再要求「一个逻辑行 = 一条独立 512B 物理行」；
2. `query_start_loc` 接受**整条常驻 buffer**（长度 ≥ `batch+1`），有效长度由参数给出。

同时修掉一个现存缺陷（`batch == 128` 越界 → launch failure / poison 设备）。

## 优先级
P0

**定级理由**：这是 P0-3「消除每步 9 千次 host↔device copy 与布局转换」里 KDA decode 侧的最后一段。
现状下 decode 每层每步要为这 5 项各 clone 一份，再加 `query_start_loc` 一次；34 层合计
**204 次物化下发/步**（方案 §5：238 − 34）。这是 KDA decode 侧 `_fused_operand` 这批里剩下的最后一段
（上游元数据 H2D 是另一批，见 [`../元数据H2D归因与合并方案_2026-10-09.md`](../元数据H2D归因与合并方案_2026-10-09.md)），
也是解除 `--enforce-eager`、打开图捕获（P1-2）的前置条件之一。

## 目标版本与期望时间
GLM-5.3-Flash 适配 Phase 2（性能阶段）。建议与 KPool 三件套（#07 扩展 / #09 扩展 / #19）同批提供，
具体里程碑待算子组排期确认。

## 算子名称
`torch.dlc_kda_recurrent_decode`（**沿用现名**，不新增算子名）

## 背景与功能说明

### 使用场景

GLM-5.3-Flash 的 KDA（Gated DeltaNet 类）层在 **plain decode** 每步对每个请求算 1 个 token；
单卡 TP8 下每层 H_local = 8、head_dim = 128、q/k/v 合并宽度 3072。34 个 KDA 层全部走本算子
（`vllm_cl/patch/patch_kda.py::_recurrent_decode_new_op`）。

### 当前问题：5 项操作数必须是「独立张量」，导致每层每步 6 次多余 clone

现有校验（都作用在**被传入的那个 view** 上）：

| 位置 | 内容 |
|---|---|
| `pytorch/aten/src/ATen/native/dlc/KdaRecurrentDecode.cpp:71-80` | 要求 `is_contiguous()` 且 `storage_offset() == 0` |
| 同上 `:104-107` | 要求 `dlc_dim0_padded_ * dlc_layout_ == 512`（一个逻辑行 = 一条独立 512B 物理行） |
| `DLC_Custom_Kernel/dlc_kernels/attention/kda/dlc_kda_recurrent_decode.cpp:29-31` | 行地址写死为 `line * dim0_padded512 * 4` |

而模型侧这 5 项**全是父张量的 view**（下表）。torch 语义下它们「看着连续、offset 也常为 0」，
但底层 storage 的行宽是父张量的，物理行距不是 512B，于是校验直接拒绝
（`raw_g` 报 `unsupported physical row layout`；`q/k/v/beta` 报 not contiguous / nonzero offset）。
调用方只能在 Python 侧先 `clone()` 成独立张量再进来 —— 每层每步 6 次（5 项 + `query_start_loc`）。

### 目标功能

1. 上述 5 项接受父张量 view，按 `(base, offset, pitch)` 取数（见「功能语义」）；
2. `query_start_loc` 接受整条常驻 buffer + 新增有效长度参数；
3. 修 `batch == 128` 越界（见「附带缺陷」）；
4. （可选）`state_indices` 直接接受 `-1`（PAD_SLOT_ID）当 null slot。

### 非目标

- **不改 `dlc_kda_causal_conv_update`（#02）**：它的 ABI、kernel、输出形状全部保持原样；
- 不改 state 布局（已是 `[S,H,Dv,Dk]`）、不改 gate/beta/state 的数学、不改输出形状与 dtype；
- `state_indices` 的逐层拷贝**不在本需求内** —— 已由插件侧「每步 pack 一次」解决
  （`docs/glm5_flash/_fused_operand_方案_2026-10-09.md` §3.5，已落地）；第 4 条仅为可选优化；
- `num_accepted_tokens` / spec / MTP 本阶段模型侧不启用，行为不变。

## 期望 API 与函数签名

```python
torch.dlc_kda_recurrent_decode(
    q, k, v, raw_g, beta_logits,          # 5 项：允许父张量 view
    A_log, g_bias,                        # 不变（根参数 reshape，已是规范布局）
    recurrent_state,                      # 不变：仍须 [S,H,Dv,Dk] 连续、512B 行
    state_indices,                        # 不变（1-D 或 2-D，行数 = batch）
    query_start_loc,                      # 放宽：允许长度 >= batch + 1 的常驻 buffer
    num_accepted_tokens=None,             # 不变
    *,
    lower_bound=-5.0,                     # 不变
    scale=0.08838834764831843,            # 不变
    batch=-1,                             # 新增，keyword-only，int；-1 = 退回 query_start_loc.size(0)-1
    out=None,                             # 不变
) -> Tensor
```

`batch=-1` 的默认值保证**旧调用方零改动**（现在的调用方传的就是 `[B+1]` 前缀切片）。
若算子组更愿意不出新参数：因为现有 ABI 已经强制 `state_indices.size(0) == batch`
（`KdaRecurrentDecode.cpp:65-68`），kernel 侧也可以直接取 `B = state_indices_->shape[0]`；
两种都接受，取哪种由算子组定。

对应 `native_functions.yaml:518` 需在 `dlc_kda_recurrent_decode` 的 `*,` 之后加 `int batch=-1`。

## 参数与返回值

只列本次改动相关的参数（其余同现有 ABI：bf16 激活、fp32 静态参数、in-place 写回 `recurrent_state`、
返回 `out` 或复用传入的 `out`）。

输入：

- `q`/`k`/`v`：`torch.Tensor`，逻辑形状 `[1, T, H, Dk]` bf16；**本次起允许是 `qkv` 合并张量的 view**
  （`qkv.split(D_total//3)` 再 `reshape(1, -1, H, Dk)`），行距由父张量决定。
- `raw_g`：`torch.Tensor`，逻辑形状 `[1, T, H, Dk]` bf16；允许是 `f_b_proj` 输出的 reshape view。
- `beta_logits`：`torch.Tensor`，逻辑形状 `[1, T, H]` bf16（未 sigmoid）；允许是合并 GEMM 输出的切片。
- `query_start_loc`：`torch.Tensor`，int32；**本次起允许长度大于 `batch+1`**，只读前 `batch+1` 个元素。
- `batch`：`int`，keyword-only，默认 `-1`；有效请求数（前 `batch+1` 个 `query_start_loc` 有效）。

返回值：

- `out`：`torch.Tensor`，`[1, T, H, Dv]` bf16；语义与形状不变。

### 模型侧真实 (offset, pitch)（TP8，H_local = 8，head_dim = 128，T = 本步 token 数）

| 操作数 | 父张量 | offset（元素） | token 行距（元素） | 每 token 有效段 |
|---|---|---|---|---|
| `q` | `#02` 输出 `[T, 3072]` | 0 | 3072 | `[0:1024]` → 2048 B，512B 对齐 |
| `k` | 同上 | 1024 | 3072 | 2048 B，512B 对齐 |
| `v` | 同上 | 2048 | 3072 | 2048 B，512B 对齐 |
| `raw_g` | `f_b_proj` 输出 `[T, 1024]` | 0 | 1024 | 整行 2048 B，512B 对齐 |
| `beta_logits` | 合并 GEMM 输出 `[T, 3336]` | 3072 | 3336 | 前 8 个 bf16 = 16 B，**起点非 512B 对齐** |

> `offset` 以**元素**计，等于该 view 的 `storage_offset()`；`token 行距` 等于
> `q/k/v/raw_g`（4-D view）的 `stride(-3)`、`beta_logits`（3-D view）的 `stride(-2)`。
> 表中的上游宽度是**逻辑**宽度；若上游 GEMM 输出行有 padding，实际行距会更大 ——
> **请从张量描述符取，不要写死 3072 / 1024 / 3336**。
> TP16 时 H_local = 4、q/k/v 各段 1024 B，其余同理；算子不应依赖具体数值。

## 功能语义

### 1. 5 项激活类操作数的取数规则

**契约**：把这 5 项当成「独立张量」读出来的**逻辑元素必须与调用方先物化、再传独立张量时逐位一致**。
现有校验（`is_contiguous` / `storage_offset()==0` / `dlc_dim0_padded_*dlc_layout_==512`）与 kernel 里写死的
行距 `line * dim0_padded512 * 4` 是仅有的障碍，请改为按**张量描述符的 `offset` 与 strides** 取数。

**推荐实现：按 token 一次读整段，而不是按 (token, head) 读 512B 行。**

```
token_addr(t) = base + offset + t * token_pitch      # 单位：元素
token_pitch   = stride(-3)   # q / k / v / raw_g（4-D view）
              = stride(-2)   # beta_logits（3-D view，即 token 行距）
```

- `base` = 父张量 storage 基址，`offset` = 该 view 的 `storage_offset()`；
- `q/k/v/raw_g`：从 `token_addr(t)` 起读 `H*Dk` 个**连续**元素（成立条件
  `stride(-1)==1 && stride(-2)==Dk && stride(-3)==H*Dk`，由调用方保证；可断言）；
- `beta_logits`：从 `token_addr(t)` 起读 `H` 个连续 bf16。

模型侧每个 token 的 `[H*Dk]` 段在父张量行内本来就是连续的，q/k/v 的段起点分别是
0 / 2048 B / 4096 B、`raw_g` 是 0 —— **全部 512B 对齐，段长 2048 B 也是 512B 整数倍**。
这样每 token 的 DMA 从 `4×H`（= 32 次）降到 4 次，且不会越过张量尾部。

**备选实现（保留 (token, head) 行模型，只把行距参数化）**：`line_addr = base + offset + (t*H + h) * stride(-2)`，
即行距用 `stride(-2)`（= `Dk` = 256 B）。可行，但有两个坑要一起处理：
① 第 2 个 head 起落在 256 B 边界（对非 512B 对齐敏感的硬件可能明显变慢）；
② 最后一次 DMA 会越过张量尾 256 B（父张量是紧凑分配、无尾部冗余），必须单独处理末行。

`beta_logits` 是唯一没有 512B 对齐起点的一项（`base + 6144 + token*6672` 字节，6672 不是 512 的倍数）：
请按「读 512B 对齐块 + 在 VMEM 内按偏移取数」或按 16 B/32 B 粒度直读实现，具体形式由算子组定，
只要求结果逐位一致。若该项实现成本过大，可先只放开 `q/k/v/raw_g` 四项（插件侧保留 `beta` 的物化），
收益降为 4/5。

### 2. `query_start_loc` + `batch`

- 允许传长度 `>= batch + 1` 的常驻 buffer（调用方传的是整条
  `vllm/vllm/v1/worker/gpu_model_runner.py:824` 的 `self.query_start_loc.gpu`，本身 dlc-contiguous、零拷贝）；
- 有效长度取前 `batch + 1` 个元素，`batch = -1` 时退回 `query_start_loc.size(0) - 1`；
- `batch` 同时用于替代 kernel 里从 shape 推 batch 的做法
  （`dlc_kda_recurrent_decode.cpp:216` `int B = query_start_loc_->shape[0] - 1;`）；
- 插件侧会传 `batch = cu_seqlens.numel() - 1`。

### 3. 附带缺陷：`batch == 128` 越界（请一并修）

`dlc_kda_recurrent_decode.cpp:244` 用同一个 `use_metadata_cache = B <= kVectorWords`（`kVectorWords = 128`）
同时管 `query_start_loc` 和 `state_indices` 的 VMEM cache，但两者需要读的 word 数不同：
`query_start_loc` 要读 `B+1` 个（`:246` 只装了 128 个 word，`:255` 却会读第 `B` 个，即第 129 个），
`state_indices` 只读 `B` 个。

实测：`batch = 126 / 127 / 129` 正常，**只有 `batch = 128` 挂**
（`Assert` 失败 → launch failure → poison 设备上下文，后续所有 kernel 下发失败）。
请拆成两个条件：`query_start_loc` 用 `B + 1 <= kVectorWords`，`state_indices` 保持 `B <= kVectorWords`；
`B+1 > 128` 时 `query_start_loc` 按 `ALIGN128(B+1)` 一次装入（`max_num_reqs ≤ 255` 时最多 2 行 = 1 KB），
`B > 128` 时 `state_indices` 也应支持多行 cache（现在退回逐请求 `load_i32_scalar`，正确但慢）。

### 4.（可选）`state_indices` 接受 `-1`

`state_indices` 行内 `-1`（PAD_SLOT_ID）当前会命中 `:277` 的
`Assert(slot >= 0 && slot < S_slot, ...)` → launch failure。调用方已把 `clamp_min(0)` 收敛到每步一次，
所以**这条不阻塞**；若一并支持（`-1` 视同 null slot：该行 `out` 全 0、state 不写回，与 slot 0 一致），
插件侧可以直接传原始 `state_indices`。

### 5. 可参考的既有先例

「按描述符 / 父张量 view 取数」在本仓库既有的 DLC 算子里已经跑通，**不需要新造机制**，请优先照抄下面两处同族实现。

**先例 A：`dlc_kda_state_gather_scatter`（#05 本族，已交付）**
- `state` / `src` / `out` 都是任意 view：只要求末三维连续、页间不重叠、`stride(0) >= 页大小`，
  **对 state 不做 `contiguous()`**；
- aten 侧：`pytorch/aten/src/ATen/native/dlc/KDAStateGatherScatter.cpp:10`
  （`storage_end = storage_offset + Σ(size-1)*stride`，`:23` 按此校验范围与 alias）、`:38`（只查页内连续）、
  `:43`（只查 `stride(0) >= 页大小`）；下发时用 `input_format` / `output_format`，
  即把张量描述符整体交给 kernel（`:99`）；
- kernel 侧：`DLC_Custom_Kernel/dlc_kernels/attention/kda/kda_state_pool.h:11` 的注释就是本需求的语义
  *"Logical offsets refer to the backing storage, whose row padding can differ from the view"*，
  寻址统一走 `kda_word_offset()`（`:25`，用 `t->storage_offset` 与 `t->stride[]` 折算，能跨带 padding 的行）。

**先例 B：`dlc_kda_causal_conv_prefill` / `dlc_kda_causal_conv_update`（#01/#02 本族，已交付）的 `conv_state`**
- `pytorch/aten/src/ATen/native/dlc/CausalConv1d.cxx:60` `make_flat_storage_view()`
  —— 把整块 storage 拍成 1-D `as_strided({storage_numel}, {1}, 0)`；
- `:90` `make_state_params()` —— 把 `size(2)` / `stride(1)` / `stride(2)` / `stride(0)` / `storage_offset`
  作为 **scalar** 一起下发给 kernel（`:250-270` 是实际的 `k.scalar(...)` 入参顺序），
  由 kernel 逐行取行首有效段（`:36-52` 的注释画了行布局）；
- 文件头注释（`:57`）明确写：*"conv_states is_contiguous() = False。此路径不做 contiguous() / clone()"*。
- **口径提示**：同一个文件里激活类操作数仍是物化路径 —— `:339` `auto qkv_work = qkv.contiguous();`、
  `:427` 同理。也就是说「父张量 view」这套做法在 state / pool 张量上已经落地，
  **激活类（q/k/v/g/beta）尚无先例**，本单正是把它推广过去；实现上建议直接复用 A 的「按描述符 stride/offset 寻址」
  或 B 的「flat view + metadata scalar」两种之一。

**框架层通道（已存在，只是本条目前把它堵住了）**
- `pytorch/aten/src/ATen/dlc/DLCKernelLaunch.cpp:155` `is_contiguous_dlc_with_metadata()`：
  非零 `storage_offset` + 带 padding 的行本来就放行，条件是
  `storage_offset % alignment == 0 && dim0 % alignment == 0 && last_dim % alignment == 0`（alignment = 512B / itemsize）；
- 同文件 `:232-242` 会把 `storage_offset` / `dim0` / `dim0_padded` / `dim0_padded512` 随 kernel 下发。
- 对比默认收紧路径：`is_dense_contiguous_dlc()`（`:30`）要求 `is_contiguous() && storage_offset()==0 && dlc_dim0_ % alignment == 0`。

**跨族先例（kernel 侧读 `storage_offset` / `stride[]` 是常规做法）**
- `DLC_Custom_Kernel/dlc_kernels/attention/paged_attention/paged_mla_bf16.cpp:282`
- `DLC_Custom_Kernel/dlc_kernels/kvcache/reshape_and_cache.hpp:123`（并且会断言 offset 与 cache dim0 对齐）
- `DLC_Custom_Kernel/dlc_kernels/hhp_tmp/copy/hhp_reshape_offset_f32_off0.cpp:289`
- `dlc_kernels/` 下共 46 个文件涉及 `storage_offset`。

### 边界与异常

- `batch == 0`、`T == 0`：保持现有行为（不新增语义）；
- 空 tensor、非法 `offset/pitch`（负 pitch、越界）：请显式报错，不要静默读错；
- `out`、`recurrent_state` 仍必须连续、512B 行、`out` 与 `recurrent_state` 不得 alias（校验不变）。

## 支持范围与约束

| 项 | 范围 |
|---|---|
| 激活 dtype | bf16（`q/k/v/raw_g/beta_logits`），与现状一致 |
| 静态参数 | `A_log` `[H]` fp32、`g_bias` `[H,Dk]` fp32，连续、offset 0（放宽范围内） |
| `recurrent_state` | `[S,H,Dv,Dk]` fp32，连续、每行 512B —— **不变** |
| `out` | `[1,T,H,Dv]` bf16，连续、512B 行 —— **不变** |
| 逻辑形状 | `T` 动态（decode 每请求 1 token，多 token 请求变长）；`H`、`Dk = Dv = 128` |
| `offset` / `token 行距` | 行距 > 0；`offset + (T-1)*行距 + 每 token 有效元素数 <= 父张量元素数`（调用方保证，算子可断言） |
| layout | 仍要求 `stride(-1) == 1`；**放宽 `is_contiguous()` / `storage_offset()==0` / 512B 物理行**（仅这 5 项） |
| device | 单卡 DLC；所有 tensor 同 device —— 不变 |

## 精度要求

- **Golden reference**：与本条目前「调用方先物化、再传独立张量」的路径**逐位一致**（`torch.equal`）。
  物化路径 = 现有已验收实现，故等价性等价于「没有回归」。
- reference 代码（可直接贴进单测；`logical()` 即 (offset, pitch) → 逻辑操作数的反解）：

```python
import torch


def logical(parent: torch.Tensor, offset: int, pitch: int, T: int, H: int, D: int):
    """按 (base, offset, pitch) 从父张量抽出逻辑操作数 [1,T,H,D]。

    行 t 的有效段 = parent.reshape(-1)[offset + t*pitch : offset + t*pitch + H*D]。
    这正是推荐取数规则 token_addr(t) = base + offset + t*token_pitch 的语义。
    """
    flat = parent.reshape(-1)
    rows = torch.arange(T).unsqueeze(1) * pitch + offset        # [T,1]
    cols = rows + torch.arange(H * D)                          # [T,H*D]
    return flat[cols].reshape(1, T, H, D).contiguous()


def logical_beta(parent: torch.Tensor, offset: int, pitch: int, T: int, H: int):
    """beta_logits：行 = token，行内取前 H 个 bf16。"""
    flat = parent.reshape(-1)
    cols = torch.arange(T).unsqueeze(1) * pitch + offset + torch.arange(H)
    return flat[cols].reshape(1, T, H).contiguous()


# 验收写法（view 路径 vs 物化路径）：
#   q_view = parent_qkv[:, : H * D].reshape(1, -1, H, D)   # 模型真实形态（qkv 的切片）
#   q_materialized = logical(parent_qkv, 0, 3072, T, H, D)
#   assert torch.equal(q_view.contiguous(), q_materialized)
#   然后两组输入分别调 dlc_kda_recurrent_decode，要求 out / recurrent_state 逐位相等。
```

- 序列递推（out / `recurrent_state`）的数值对齐参考实现：
  `docs/glm5_flash/op_requests/refs/kda_reference_dv_dk.py::recurrent_decode_ref`
  （state 布局 `[S,H,Dv,Dk]`，与 kernel 一致）。
- 比较方式：`atol = 0, rtol = 0`（同一 kernel、输入相同，必须逐位相同）；
  若要跨实现比对，用 bf16 量级 `atol = 2e-3` / state `atol = 2e-2`（现状回归的容差）。
- `batch = 128`（原挂死点）与 `batch > 128`（多行 cache 路径）必须与 `batch <= 127` 结果一致。

## 测试范围

- [x] 基本功能与 reference 对齐
- [ ] 多 dtype 精度（本算子激活只有 bf16，范围不变）
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [x] 非 contiguous、特殊 stride 或 layout
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式
- [ ] 长时间稳定性
- [x] 性能基准与性能回归

具体用例、shape 与预期：

1. **view 直传等价**：按上表 5 项的 `(offset, pitch)` 构造父张量 view，与 `logical()` 物化版本
   逐位一致；`T = 1 / 2 / 4 / 64`，`H = 8 / 16`。
2. **`query_start_loc` 整条 buffer**：`batch = 1 / 2 / 64 / 127 / 128 / 129 / 200`，
   整条 buffer（长度 `max_num_reqs+1`，尾部填 `last_query_loc`）与 `[:batch+1]` 前缀切片逐位一致；
   **`batch = 128` 必须不再 launch failure**。
3. **`beta_logits` 非对齐起点**：`offset = 3072`、行距 = `3336`（以及上游 GEMM 输出行被 padding 后的
   实际行距），与物化版本逐位一致。
4. **`state_indices`**：1-D 与 2-D（`next_n > 1`）两种形态、含 `-1` 行（若实现第 4 条）。
5. **异常**：行距 ≤ 0、越界（`offset + (T-1)*行距 + 每 token 有效元素数` 超父张量）、
   `stride(-1) != 1` → 明确报错，不静默读错。
6. **性能回归**：现状每层每步有 6 次物化下发（5 项 + `query_start_loc`），目标降到 **0**；
   单层单次调用耗时不高于现状。

## 验证环境与依赖版本

```text
DLC 硬件：8 卡，设备名 /dev/cltech0-7（TP=8，无 PP；单测逐张干净设备轮测）
vLLM commit：a07576686a
vLLM DLC（vllm-cl）commit：bab8586
PyTorch / torch-dlc：torch 2.5.0（DLC 构建），Python 3.10.16
关键环境变量：VLLM_USE_DLC_COL_MAJOR_MATMUL=1、DLC_KDA_RECURRENT_DECODE_OP=dlc（默认）
模型：GLM-5.3-Flash-Mix（TP8 → 每卡 H=8、head_dim=128、q/k/v 合并宽度 3072；34 层 KDA）
```

## 验收标准

- [ ] `q/k/v/raw_g/beta_logits` 接受父张量 view（torch 非连续 / 非零 `storage_offset`），
      结果与物化路径**逐位一致**
- [ ] `query_start_loc` 接受长度 ≥ `batch+1` 的常驻 buffer，`batch` 生效（含默认 `-1` 兼容旧调用）
- [ ] `batch == 128` 不再 launch failure、不再 poison 设备
- [ ] （若实现）`state_indices` 行内 `-1` 视同 null slot，与 slot 0 语义一致
- [ ] `recurrent_state` / `out` 的校验与语义保持不变（仍 512B 行、不得 alias）
- [ ] 上述测试范围逐项通过；`docs/glm5_flash/op_requests/refs/kda_reference_dv_dk.py` 对齐
- [ ] 算子组提供对应的 pytorch 侧 ABI/校验改动与单测（`pytorch/test_tpu/test_aten/.../test_kda_recurrent_decode.py`）

## Reference、日志与附件

- 方案与逐处改法：`docs/glm5_flash/_fused_operand_方案_2026-10-09.md`
  §3.3（5 项取数）、§3.4（`query_start_loc`）、§6（本需求的原始 prompt）
- 归因数据：`docs/glm5_flash/性能优化点清单_2026-09-29.md`（P0-3）
- 复现脚本（真机实测，含每项结论与 `batch == 128` 挂死复现）：
  - `docs/glm5_flash/op_requests/probes/probe_kda_decode_operand_views.py dlc:0`
    （三段：入口校验逐项 / 隐式 staging 成本 / `batch == 128` 挂死复现）
- 回归用例：`vllm-cl/tests/test_glm5_kda_recurrent_decode_op.py`（21 例，含 out 视图 / PAD slot /
  静态参数原形状直传等接入形态）
- reference 实现：`docs/glm5_flash/op_requests/refs/kda_reference_dv_dk.py`
- 相关单据：#04 `04_dlc_kda_recurrent_decode.md`（已完成并接入）、#02 `02_dlc_kda_causal_conv_update.md`
  （**本需求不改动它**）
