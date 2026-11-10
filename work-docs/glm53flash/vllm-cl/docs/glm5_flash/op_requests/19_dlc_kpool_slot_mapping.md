<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-10-09；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-10-09）：完全新算子 -->
<!-- 已有算子覆盖核查（2026-10-09）：未覆盖。DLC 无 searchsorted/bucketize，
     现有池 slot / tail slot 两处映射都是 CPU 往返（vllm_cl/_searchsorted_utils.py:8），
     批量一次约 0.28~0.33 ms/次。原本 V1 路径每步 3 次（worker 1 + tail builder 1 + pool builder 1），
     其中 tail builder 那次与 worker 重复，已由插件侧去重（2026-10-09），**现为每步 2 次**。
     原 08 号声称“顺带消除”，但其 ABI 只收
     req_ids/token_offsets，不含 metadata，实际覆盖不到，故独立成单。 -->
<!-- 2026-10-12 修订：tail 侧入参由 2-D `[R, NB_tail]` 改为 1-D 连续 `[R] int32`
     `tail_pages`。原因见「参数与返回值 → tail 页表形状约束」：DLC 对 int32 的
     dim0（最内维）按 128 元素 / 512 B 对齐，`[R, 1]` 会得到 128 倍 padding，
     且读写都要按 512 B 行距跳行。调用方每步只需一次 `[:, 0].contiguous()`（1 KB），
     并可并入每步 metadata 打包（见 docs/glm5_flash/元数据H2D归因与合并方案_2026-10-09.md）。 -->

# [Operator Request] dlc_kpool_slot_mapping（KPool 池槽 / tail 环槽 双输出映射）

## 需求类型
新增算子

**需求分类：完全新算子**
- 覆盖核查（2026-10-09）：DLC 无 `searchsorted` / `bucketize`，也无「按 request 边界把 token 映射到 cache 物理槽」的算子。
- 现有可复用项：无。`torch.convert_req_index_to_global_index` 做的是「逻辑 token 位置 → latent cache 槽」，
  输入已经是 per-token 的 request id 与 token 序号，**不含** `query_start_loc` 边界求 request id 这一步，语义不覆盖本条。
- 本条合并且只合并两处历史实现：
  - 池槽（pool slot）：`vllm_cl/patch/models/kpool_slot_mapping.py:18` `get_compressed_slot_mapping_native`
  - tail 环槽（tail slot）：`vllm_cl/patch/patch_mla_indexer.py:119` `_compute_kpool_tail_slot_mapping_native`、
    `vllm_cl/kpool_worker_slots.py:28` `compute_kpool_tail_slots_native`
- 两者共享「从 `query_start_loc` 求每 token 的 request id + 每请求的 q_start/q_end/len」这一整段中间量，
  各自做一遍就是两次 CPU 往返；再叠加 worker 侧的 tail 映射（`kpool_worker_slots.py:59`），
  当前 V1 路径每步共 **3 次** CPU 往返。合成一条算子、一次算完、输出两个张量，是把这 3 次压到 0 的做法。
  另注：worker 侧与 tail builder 侧算的是**同一个** tail 映射，插件侧本可先去重一次（见 kpool_mla_dataflow.md §7）。

## 优先级
P1

- 不阻断正确性：当前 native 组合实现数值正确，只是每步 2 次 host 同步（去重前为 3 次）。
- 收益量级：去重后仍是每步（不是每层）**2 次** CPU 往返，实测 0.28~0.33 ms/次 → 合计约 **0.6 ms/步**
  （去重那次已由插件侧拿回，见「背景」末段）。
- P0 门禁（长上下文性能验收）之前提供即可；若算子组排期紧张，可与 08 号合并排期。

## 目标版本与期望时间
GLM-5.3-Flash 适配 Phase 2（性能阶段）。建议在长上下文（≥32K prefill）性能验收前提供，具体里程碑待算子组排期确认。

## 算子名称
`torch.dlc_kpool_slot_mapping` —— 一个算子，**双输出**：

1. `pool_slot_mapping`：压缩池 cache 的物理槽（只有每 4 个 token 的最后一个有效）；
2. `tail_slot_mapping`：tail 环形 cache 的物理槽（每个活 token 一个）。

**请勿拆成两个算子**：两者的 request-id 搜索、`q_start`/`q_end`/`q_len` gather、位置推导完全共用，
拆开等于把这次的收益原样退回给插件。

## 背景与功能说明
使用场景：GLM-5.3-Flash 的 KPool 选池链路每步要产出两张「token → cache 物理槽」映射表，
分别喂给池 cache（`[blocks, 64, 128]`，256-token 块 / 4-token 池）与 tail 环 cache（`[blocks, 2, 4, 128]`）：

- 池槽：只有 `(pos + 1) % 4 == 0` 的 token 会写一个压缩池；槽位 = `物理块号 * 64 + 块内池号`，
  其中块号来自池 cache 的页表，需要按 `compressed_pos // 64` 做多块寻址。
- tail 环槽：每个活 token 都写；槽位 = `tail 页表第 0 列 * 4 + pos % 4`（每请求一条 4 槽的环），单块寻址。

当前问题：两处都先用 `torch.searchsorted(query_start_loc, arange(T), right=True) - 1` 求每 token 的 request id，
再 gather 每请求的边界与长度。DLC 未注册 `searchsorted`，插件里的回退是
**先把 `query_start_loc` 搬到 CPU、算完再搬回设备**（`vllm_cl/_searchsorted_utils.py:8`），
实测 0.28~0.33 ms/次（`probes` 见下），每步各做一次。

目标功能：单 kernel 内完成「request 边界搜索 → 每 token 位置推导 → 两组页表寻址」，
输出池槽与 tail 环槽两个 `[T]` int32 张量，全程无 host 往返，可被 CUDA graph 捕获。

非目标：池化压缩数学（07 号）；tail 环写入本身（08 号）；池级打分与 TopK（09/10 号）。
本条**不**负责把 `keys`/`gate` 写进 cache —— 那是 08 号，本条只产出地址。

## 期望 API 与函数签名
```python
def kpool_slot_mapping(
    query_start_loc: torch.Tensor,     # [R+1] int32/int64，请求 token 边界（含末元素）
    seq_lens: torch.Tensor,            # [R]   int32/int64，本 chunk 之后每请求的总序列长
    positions: torch.Tensor,           # [T]   int64，每 token 在其请求内的绝对位置
    pool_block_table: torch.Tensor,    # [R, NB_pool] int32，池 cache 页表
    tail_pages: torch.Tensor,          # 1-D int32，每请求 tail 页号（容量按 128 对齐，见下）
    live_mask: torch.Tensor,           # [T]   int32/bool，该 token 是否要写（padding 为 0）
    *,
    pool_size: int = 4,                # 一个池 = 4 个 token
    pools_per_block: int = 64,         # 池 cache 每物理块的池数 = block_size / pool_size
    num_tokens: int | None = None,     # 缺省用 positions.numel()
    req_indices: torch.Tensor | None = None,  # [R] int32，V2 表格的持久请求行重映射，可选
) -> tuple[torch.Tensor, torch.Tensor]:   # (pool_slots[T] int32, tail_slots[T] int32)
```

## 参数与返回值
输入：
- `query_start_loc`：长度 `R + 1`，单调不减；`R = num_reqs`。
- `seq_lens`：长度 `R`，值为该请求**本 chunk 之后**的总长度（含本 chunk）。
- `positions`：长度 `T`，每 token 的绝对位置（0-based），padding 行可填 0 并用 `live_mask` 屏蔽。
- `pool_block_table`：`[R, NB_pool]`，池 cache 的物理块号，`-1` 表示未分配。
- `tail_pages`：**1-D 连续 int32**，每请求的 tail 物理页号，`-1` 表示未分配。
  长度是**对齐后的容量** `ceil(max_num_reqs/128)*128`（当前 256），只读前 `R` 个元素；
  `R` 由 `seq_lens.numel()` 给出（见「tail 页表形状约束 → 容量与切片约束」）。
  行号语义：`tail_pages[r]` 是「batch 第 r 行」的页号；V2 的持久行重映射仍由 `req_indices` 完成
  （即 `page = tail_pages[req_indices[safe_req]]`），与 pool 侧一致。
  **不再接收 `[R, NB_tail]` 的 2-D 页表**，理由见下面「tail 页表形状约束」。
- `live_mask`：`[T]`，`1`/`True` = 该 token 需要写 cache。对应插件里 worker 侧已经算好的
  `slot_mapping >= 0` 语义（负槽位 = 无效 token，必须沿用，不能由本算子自行推断）。
- `req_indices`：可选，`[R]`，把「batch 行」映射到「持久请求行」。V1 不需要；V2 必须传。

属性：
- `pool_size`：默认 4。
- `pools_per_block`：默认 64。
- `num_tokens`：默认 `positions.numel()`。

返回值：
- `pool_slots`：`[T]` int32。越界 / 未分配 / 非池末 token 一律 `-1`。
- `tail_slots`：`[T]` int32。`live_mask == 0` 的行一律 `-1`。

边界：`T == 0` 或 `R == 0` 返回两个全 `-1`（长度为 `T`）的 `[T]` 张量，不得报错。

### tail 页表形状约束（2026-10-12 修订，**必须遵守**）

`tail_pages` 必须是 **1-D 连续 int32**（容量对齐到 128 元素），
不接受 `[R, 1]` / `[R, NB_tail]` 形状。依据：

1. tail 组在 vLLM 侧**每请求只有一个页**：`KpoolTailSpec.max_num_blocks_per_req()` 恒返回 1
   （`vllm/v1/kv_cache_interface.py:893`），且 `KpoolTailBackend.get_supported_kernel_block_sizes()`
   是 `[MultipleOf(1)]`（`vllm/v1/attention/backends/mla/indexer.py:277`），
   因此 `blocks_per_kv_block == 1`，worker 侧张量就是 `[max_num_reqs, 1]` int32。
2. DLC 的 int32 对齐粒度是 **dim0（最内维）= 128 元素 / 512 B**
   （`c10/dlc/kernel_arg_types_copy.h:70`，`dlc_storage_alignment(dlc_padding_int32) == 128`），
   而 `compute_storage_nbytes_contiguous_dlc` 会把 dim0 向上取整；
   `DLC_STORAGE_MIN_USAGE = 0.0`（`aten/src/ATen/dlc/DLCStorageImpl.h:8`）
   使「dim0 远小于对齐粒度时压平成 1-D」的分支永不生效。
   于是 `[max_num_reqs=256, 1]` int32 的物理布局是 `dim0_padded=128 × dim1=256`
   = **128 KB**，而逻辑数据只有 1 KB —— **128 倍 padding**。
   对照 pool 页表 `[256, 4096]`：dim0 = 4096 已是 128 的整数倍，无多余 padding，
   所以这个坏形状只出现在 tail 表上。
3. 访问代价：按 2-D 表访问时，内核要么按行搬 512 B（读 128 KB 用 1 KB），
   要么 4 B × R 次散射读；两者都是本芯片上已知的慢路径。
   改成 1-D `[R]` 后，当前几何 `R ≤ 256` → 256 元素 = 1 KB，
   dim0 天然 128 对齐，**零 padding、一次连续读**。

**并入每步 metadata 打包（2026-10-12 约定，接入时落地）**

打包框架**已经存在**：`vllm_cl/worker/metadata_packer.py`（C1 已落地）把
`input_ids` / `req_indices` / `query_pos` / `num_scheduled_tokens` 放进按 dtype 分桶的
常驻 pinned/device buffer，`mark()` + `flush()` 让每步每种 dtype 只发一次 H2D
（用法见 `vllm_cl/worker/dlc_model_runner.py:785`、`mark/flush` 在 `:2150-2153`）。

接入方式（三步，不新增任何 H2D 次数）：

1. 申请段：`self.kpool_tail_pages = self._metadata_packer.add("kpool_tail_pages", <容量>, torch.int32)`；
2. 每步在 host 侧直接填列：`self.kpool_tail_pages.np[:num_reqs] = tail_block_table.np[:num_reqs, 0]`
   —— `BlockTable` 的 host 侧本来就有这张表（`vllm/v1/worker/block_table.py:115` 的 `.np`），
   所以**不需要设备侧的 strided 拷贝**；随后 `mark("kpool_tail_pages", num_reqs)`，
   与 `num_scheduled_tokens` 共用既有的 `flush()` 点（`dlc_model_runner.py:2153`）；
3. device 侧把该段**整块**交给算子当 `tail_pages`（见下条容量约束），
   尾页号收敛带来的 1 KB 上传被完全摊进已有的 int32 桶。

容量与切片约束（**重要**）：打包段的视图与其它段共享一条带 padding 的 storage，
launcher 的 `at::detail::is_contiguous_dlc`（`pytorch/aten/src/ATen/dlc/DLCTensorImpl.h:56`）
对这种情况要求 `sizes.back() % align == 0`（int32 的 `align` = 128 元素 = 512 B）。
于是 `segment.gpu[:num_reqs]` 在 `num_reqs` 不是 128 的整数倍时（prefill 常态，例如 20）
会被判为非连续，launcher 自己再做一次连续化拷贝。
→ **约定**：段容量取 `ceil(max_num_reqs / 128) * 128`（当前 = 256，天然对齐、零浪费），
**整块**作为 `tail_pages` 传入，有效长度 `R` 由 `seq_lens.numel()` 给出；算子只读前 `R` 个元素，
不得依赖 `tail_pages.numel() == R`。

## 功能语义

```text
# 共用部分：每 token 的 request id 与位置
req[t]        = searchsorted_right_minus_one(query_start_loc[:R+1], t)
valid_req[t]  = 0 <= req[t] < R
q_start[t]    = query_start_loc[req[t]]
q_end[t]      = query_start_loc[req[t] + 1]
q_len[t]      = q_end[t] - q_start[t]
pos[t]        = positions[t]                          # 调用方已提供，不再从 seq_lens 反推

# ---- 输出 1：池槽（每 4 个 token 的最后一个有效）----
cpos[t]       = pos[t] // pool_size                   # 注意用 floor 语义
blk[t]        = cpos[t] // pools_per_block
in_blk[t]     = cpos[t] - blk[t] * pools_per_block
pool_blk[t]   = pool_block_table[req[t], blk[t]]
pool_slots[t] = pool_blk[t] * pools_per_block + in_blk[t]
pool_valid[t] = valid_req[t] & live_mask[t]
              & (t >= q_start[t]) & (t < q_end[t])    # 落在本请求的 token 窗口内
              & (pos[t] >= 0) & ((pos[t] + 1) == (cpos[t] + 1) * pool_size)   # 恰好是池末
              & (0 <= blk[t]) & (blk[t] < NB_pool)
              & (pool_blk[t] >= 0)
pool_slots[t] = pool_valid[t] ? pool_slots[t] : -1

# ---- 输出 2：tail 环槽（每个活 token 一个）----
off[t]        = pos[t] - (pos[t] // pool_size) * pool_size     # = pos % pool_size
tail_blk[t]   = tail_pages[req[t]]                # 1-D，行号 = batch 行（V2 经 req_indices 重映射）
tail_slots[t] = tail_blk[t] * pool_size + off[t]
tail_valid[t] = valid_req[t] & live_mask[t]
              & (t >= q_start[t]) & (t < q_end[t])
              & (pos[t] >= 0) & (tail_blk[t] >= 0)
tail_slots[t] = tail_valid[t] ? tail_slots[t] : -1
```

关键语义约束：
1. `searchsorted` 必须是 `right=True` 再减 1（`searchsorted_right_minus_one`），
   使零长度/边界上重复的请求行归属正确；`-1` 结果表示该 token 不属于任何请求。
2. 池槽只在 `(pos + 1) % pool_size == 0` 时有效；**不要**在其它位置给出「就近的池」。
3. `pool_size` 与 `pools_per_block` 语义不同，调用方容易搞混：
   池槽用的是 `pools_per_block`（默认 64，来自 256-token 块 / 4-token 池），
   tail 槽用的是 `pool_size`（4，即环长）。请在实现里显式区分，不要共用一个 `block_size`。
4. `live_mask` 必须参与有效性判定；无效 token 一律给 `-1`，不得写出可寻址槽位。
5. 负槽位的处理：**`-1` 行不得写入 cache**。当前插件在 08 号的环写里没有跳过 `-1`
   （`vllm_cl/patch/models/kpool_indexer.py:680` 对整窗无条件 scatter，`-1` 会落到 `tail_cache[-1]`），
   算子按本语义实现即顺带修掉这个隐患。
6. 不做任何 `set_`/`copy_` 之外的内存写；不修改任何输入张量。

broadcasting / mask：无。
in-place / alias：不修改输入；`out` 由调用方缓冲（见「支持范围」）。
空 / 零长度：`T == 0` 或 `R == 0` 返回全 `-1`。
异常：`query_start_loc.numel() != R + 1`、`seq_lens.numel() < R`、页表行数 `< R`、
`pool_size <= 0`、`pools_per_block <= 0` 报错。
确定性：无随机；整数运算，结果必须逐位确定。

## 支持范围与约束
支持的 dtype：页表与输出 int32（建议同时接受 int64 输入并内部收窄）；`live_mask` int32/bool。
支持的 shape：`R` 动态（覆盖 1~256）；`T` 动态（覆盖 1~32K+）；`pool_size=4`；
`pools_per_block=64`（`block_size=256`）；`NB_pool` 动态（当前部署 = `ceil(max_model_len/256)`
= 4096 @ 1M 上下文）；`tail_pages` 固定 1-D、容量按 128 元素对齐（当前 256）。
数值范围：页号可为 `-1`（未分配）；槽位为 int32 且必须能覆盖 `物理块数 * pools_per_block`。
layout：`tail_pages` 必须 1-D 连续；`pool_block_table` 允许非连续 **但需在文档中明确**
（当前调用方传入的是 `block_table[:, ::factor] // factor` 的结果，已是连续表）。
禁止把 `tail_pages` 展开成 `[R, 1]` / `[R, NB_tail]`（原因见「tail 页表形状约束」）。
device：DLC 单卡。
执行模式：**必须支持 CUDA graph**（decode 主路径），即全程无 host 同步、无动态 shape 依赖 host 值。
输出缓冲：建议提供 `out_pool` / `out_tail` 可选参数，允许调用方复用持久缓冲
（插件侧两张表都在 builder 里持有固定容量 buffer，见 `patch_mla_indexer.py:191`）。
其他限制：`DCP=1`、`PCP=1`（插件当前一致要求，见 `patch_mla_indexer.py:474`）。

## 精度要求
本条为整数地址运算，**不存在容差**：
- 要求输出与 reference **逐元素完全一致**（`torch.equal` 为真），不存在 atol/rtol。
- 定向覆盖：请求边界处的 token、零长度请求行、`pos = 0`、`pos % 4 == 3`（池末）、
  `pos % 4 != 3`、跨物理块边界（`cpos = 63/64/65`）、`block_table` 含 `-1`、`live_mask` 全 0 的行。
- NaN/Inf：不适用。
- 累加/中间精度：全部整数运算，中间量建议用 int64 再收窄，避免 `pos * pool_size` 溢出。

## 测试范围
- [ ] 基本功能与 reference 对齐
- [ ] 多 dtype 精度
- [ ] 静态 shape 与动态 shape
- [ ] 最小值、最大值、空 tensor 等边界
- [ ] 非 contiguous、特殊 stride 或 layout
- [ ] 非法参数和异常处理
- [ ] 单卡
- [ ] 多卡/分布式
- [ ] 长时间稳定性
- [ ] 性能基准与性能回归

具体测试用例、shape、数据范围和预期结果：
1. 功能：`R ∈ {1, 8, 64, 256}`，`T/R ∈ {1, 4, 512, 4096}`，`pool_size=4`，`pools_per_block=64`，
   与下方 reference 逐元素一致。
2. 边界：`T=0`、`R=0`、零长度请求行（`query_start_loc` 出现重复值）、`live_mask` 有 0 有 1。
3. 页表：`pool_block_table` / `tail_pages` 含 `-1`、乱序、恰好用满列数；
   另需覆盖 `R` 非 128 整数倍（如 `R = 1 / 8 / 64 / 200`）、以及
   **`R < tail_pages.numel()`（对齐容量 256）** 的 1-D 读法。
4. 块边界：`cpos` ∈ {63, 64, 65} 与 `NB_pool` 边界。
5. V2 行重映射：`req_indices` 非空且乱序。
6. 异常：少一个元素的 `query_start_loc`、页表行数不足、`pool_size=0`。
7. CUDA graph：捕获后重放，输出与 eager 一致；确认无 host 同步事件。
8. 性能：`R=64, T=64`（decode）与 `R=1, T=4096`（prefill）两种场景，目标单次 < 0.1 ms，
   且相对两次 `searchsorted` CPU 往返（合计 ~0.6 ms）有 ≥ 5× 的收益。

## 验证环境与依赖版本
DLC 硬件：8 卡，设备名 `/dev/cltech0-7`（TP=8，无 PP）
vLLM commit：`a07576686a`
vLLM DLC（vllm-cl）commit：`bab8586`
PyTorch / torch-dlc：torch 2.5.0（DLC 构建），Python 3.10.16
关键环境变量：`VLLM_USE_DLC_COL_MAJOR_MATMUL=1`
模型：GLM-5.3-Flash-Mix（AWQ 4-bit，`index_kpool=4`、`index_topk=2048`、`index_head_dim=128`、`index_n_heads=32`）

## 验收标准
- [ ] API、参数校验和错误行为符合约定
- [ ] 规定范围内的功能和精度测试通过（整数逐位一致）
- [ ] 需求中明确列出的特殊运行场景通过
- [ ] CUDA graph 下可捕获/重放，无 host 同步
- [ ] `tail_pages` 以 1-D 连续 int32 传入（拒收 `[R, 1]` / `[R, NB_tail]`），容量按 128 对齐、
      支持 `tail_pages.numel() > R`；`R` 非 128 整数倍时也无行距跳读、无 128 倍 padding 浪费
- [ ] 文档、测试和示例已提交
- [ ] 接入 vllm-cl 后，池槽与 tail 环槽两处 native 实现可整体删除，且
      `tests/test_kpool_metadata_native.py`、`tests/test_kpool_indexer_native.py` 全绿

## Reference、日志与附件

### Golden reference（直接可跑，来自插件现有生产实现）

两个函数都是当前插件在用的生产代码，算子必须与它们逐位一致；
`searchsorted_right_minus_one(starts, tokens) == torch.searchsorted(starts, tokens, right=True) - 1`。

```python
import torch

def _searchsorted_right_minus_one(starts, tokens):
    return torch.searchsorted(starts.cpu(), tokens.cpu(), right=True) - 1


def kpool_slot_mapping_ref(
    query_start_loc, seq_lens, positions,
    pool_block_table, tail_pages, live_mask,
    *, pool_size=4, pools_per_block=64, num_tokens=None, req_indices=None,
):
    """Return (pool_slots[T], tail_slots[T]) int32, matching production."""
    if num_tokens is None:
        num_tokens = positions.numel()
    R = seq_lens.numel()
    dev = positions.device
    pool_slots = torch.full((num_tokens,), -1, dtype=torch.int32, device=dev)
    tail_slots = torch.full((num_tokens,), -1, dtype=torch.int32, device=dev)
    if num_tokens == 0 or R == 0:
        return pool_slots, tail_slots

    starts = query_start_loc[:R + 1].to(torch.int64)
    tokens = torch.arange(num_tokens, dtype=torch.int64)
    req = _searchsorted_right_minus_one(starts, tokens)
    valid_req = (req >= 0) & (req < R)
    safe_req = req.clamp(min=0, max=R - 1)
    q_start = starts.index_select(0, safe_req)
    q_end = starts.index_select(0, safe_req + 1)
    q_len = q_end - q_start
    pos = positions[:num_tokens].to(torch.int64)
    in_window = valid_req & (tokens >= q_start) & (tokens < q_end) & (pos >= 0)

    # ---- pool slots ----
    cpos = torch.div(pos, pool_size, rounding_mode="floor")
    blk = torch.div(cpos, pools_per_block, rounding_mode="floor")
    in_blk = cpos - blk * pools_per_block
    safe_blk = blk.clamp(min=0, max=pool_block_table.shape[1] - 1)
    flat = safe_req * pool_block_table.shape[1] + safe_blk
    block_num = pool_block_table.reshape(-1).index_select(0, flat).to(torch.int64)
    slots = block_num * pools_per_block + in_blk
    valid = (in_window
             & ((pos + 1) == (cpos + 1) * pool_size)
             & (blk >= 0) & (blk < pool_block_table.shape[1])
             & (block_num >= 0)
             & (live_mask[:num_tokens].to(torch.bool)))
    pool_slots.copy_(torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int32))

    # ---- tail ring slots ----
    row = safe_req if req_indices is None else req_indices[:R].to(torch.int64).index_select(0, safe_req)
    row_ok = (row >= 0) & (row < tail_pages.numel())
    # 生产实现等价写法：tail_block_table[:, 0].to(torch.int64).index_select(0, row)
    page = tail_pages.to(torch.int64).index_select(0, row.clamp(min=0))
    off = pos - torch.div(pos, pool_size, rounding_mode="floor") * pool_size
    tslots = page * pool_size + off
    tvalid = (in_window & (page >= 0) & row_ok
              & (live_mask[:num_tokens].to(torch.bool)))
    tail_slots.copy_(torch.where(tvalid, tslots, torch.full_like(tslots, -1)).to(torch.int32))
    return pool_slots, tail_slots
```

说明：
- `tail_pages` 是「已收敛的 1-D 页号向量」，接入后由 metadata 打包段的 host 侧填入
  （`packer.np[:R] = block_table.np[:R, 0]`），device 侧直接是常驻段；
  与当前 native 的 `tail_block_table[:, 0].to(torch.int64)` 逐位等价（见「tail 页表形状约束」）。
- `_searchsorted_right_minus_one` 在 reference 里走 CPU 是为了可跑（DLC 无 `searchsorted`）；
  算子实现必须在 device 内完成，且结果与 CPU 版逐位一致。
- `seq_lens` 在 reference 中仅用于校验（生产实现用它反推 `pos` 当 `positions` 缺失）；
  算子 ABI 保留它是为了在 `positions` 缺失时可退化为
  `pos = seq_lens[req] - q_len + (t - q_start)`（与原实现一致）。

### 相关实现与脚本
- 池槽 native：`vllm_cl/patch/models/kpool_slot_mapping.py:18`、`:117`
- tail 槽 native：`vllm_cl/patch/patch_mla_indexer.py:119`、`vllm_cl/kpool_worker_slots.py:28`
- CPU 回退点：`vllm_cl/_searchsorted_utils.py:8`
- 调用点：`vllm_cl/patch/patch_mla_indexer.py:191`（tail builder）、`:482`（池 builder 钩子）、
  `vllm_cl/patch/patch_block_table.py:14`、`vllm_cl/worker/v2_ops.py:1616`、`:1701`
- tail 页号收敛（本单接入时替换这三处）：
  `vllm_cl/kpool_worker_slots.py:70`、`vllm_cl/patch/patch_mla_indexer.py:174`、
  `vllm_cl/worker/v2_ops.py:1613`，现在都是 `block_table[:, 0]`（stride(0)=512 B 行距），
  接入后统一改为每步一次 `block_table[:, 0].contiguous()` 并常驻/并入 metadata 打包
  （`docs/glm5_flash/元数据H2D归因与合并方案_2026-10-09.md`）。
- 上游对照（NVIDIA 路径）：池槽是 Triton kernel
  `vllm/vllm/v1/attention/backends/mla/compressor_utils.py:151`（kernel `:172`）；
  tail 槽是纯 PyTorch + `searchsorted` `vllm/vllm/v1/attention/backends/mla/indexer.py:558`
- 计时脚本：`/tmp/bench_searchsorted.py`（0.28~0.33 ms/次），本次新增
  `/tmp/bench_kpool_ring_alts.py`（环写替代写法对照：`div(floor)` 0.49→`//` 0.23 ms/层）
- 数据流上下文：`docs/glm5_flash/kpool_mla_dataflow.md` S0 段
- 相关文档：`08_dlc_kpool_append_update.md`（**2026-10-09 已撤销**；本单承接其「tail slot 内建」部分，
  其余部分并入 `07_dlc_kpool_compress.md` 与 `09_dlc_kpool_mqa_logits.md`）、
  `docs/glm5_flash/性能优化点清单_2026-09-29.md` P0-3 剩余归因
