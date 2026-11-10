<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22）：完全新算子 -->
<!-- 已有算子覆盖核查（2026-09-21，DLC 实测）：未覆盖（保留新增）
     证据：kernel 清单（kvcache/index/softmax 等）无 softmax 池化压缩；dlc_indexer_k_quant_and_cache 只做 fp8 量化并按 slot 写入
     建议：保留新增（P0） -->
<!-- 2026-10-09 修订：把 08_dlc_kpool_append_update.md 的 tail 环更新并入本单，ABI 从
     「单请求 + 标量 tail_start + 外部压紧槽位表」扩展为「整批 + 全张量 + kernel 内自读 prior 与自写尾池」。
     已交付的 torch.dlc_kpool_compress 保留为退化形态（不带 tail 环的纯压缩），语义不变。
     变更理由与逐条差异见「本次修订」一节。 -->

# [Operator Request] dlc_kpool_append_compress（KPool 池化压缩 + tail 环更新）

## 需求类型
扩展已有算子能力

**需求分类：扩展已有算子（把 08 号并入 07 号）**
- 覆盖核查（2026-09-21）：DLC 算子与 kernel 清单里没有任何「softmax 池化压缩」实现。最接近的两个都不是本语义：
  - `dlc_indexer_k_quant_and_cache`：把 k 量化成 fp8 并按 `slot_mapping` 写入 cache；
  - `dlc_indexer_k_cache_update_`：按 slot 写 indexer k cache。
  两者都不含「gate + APE 在 pool 内 softmax → 对 k 加权求和」的池化语义。故 09-21 判定为新增算子（P0）。
- 2026-10-09 复核：`torch.dlc_kpool_compress` 已交付并接入。本次不新增算子，而是**扩展它的边界**，
  把 08 号（tail 环形更新）吸收进来，并把 ABI 张量化，使整条 KPool 状态更新链路成为一次调用。
  08 号独立实现**不再需要**，见「本次修订」。

## 优先级
P0

**定级理由（2026-10-09）**：本单不再是「省 1.6 ms/步的环写」（那是 08 号降为 P1 的理由），
而是**消除模型 forward 内的 kernel 外编排与 D2H**，是解除 `--enforce-eager`、打开图捕获的前置条件之一
（该项在图捕获排查中被列为 `native_kpool_indexer_step` 内两处硬阻塞之一）。08 号因此并入本单并一并升为 P0。

## 目标版本与期望时间
GLM-5.3-Flash 适配 Phase 2（性能阶段）。建议在长上下文（≥32K prefill）性能验收与图捕获打开之前提供，
具体里程碑待算子组排期确认。

## 算子名称
**推荐**：`torch.dlc_kpool_append_compress`（`dlc_kpool_compress` 的扩展入口，语义为它的超集）

> 命名不影响需求内容：若算子组更希望沿用同名入口，可在 `torch.dlc_kpool_compress` 上以可选参数形式扩展，
> 语义与本篇一致即可。已交付的 `(keys, gate_scores, ape, slot_mapping, pool_cache, pool_size, tail_start)`
> 调用形态**保持可用**，作为不带 tail 环时的退化路径（现有单测与对照实现仍走它）。

## 本次修订（2026-10-09，实现前必读）

### 1. 为什么必须扩：当前"编排"都在 kernel 之外

插件现在把 S2（tail 环更新）+ S3（池化压缩）拆成 R 次（R = 请求数）Python 调用，每次调用前在 kernel 外做一堆搬运：

| 当前编排（`vllm_cl/patch/models/kpool_indexer.py`） | 位置 | 每请求每层代价 |
|---|---|---|
| 从环里读出 prior：`to(keys.dtype)` + `clone()` × 2 个平面 | `:1010-1011` | 2 次高级索引 + 2 次 cast + 2 次 clone |
| 环写：`tail_slot_mapping[...].long()` | `:678` | 1 次 int32→int64 cast + 1 次 alloc |
| 环写：`// pool_size` + `torch.remainder` 求 (block, offset) | `:679-680` | 2 次 elementwise |
| 环写：两次 3-D 高级索引 scatter（K 平面、gate 平面） | `:681-683` | 写满 T 行（重复覆盖同一批槽位，只有最后几次生效） |
| 压紧槽位表：`slot_rows[slot_rows != -1]` + `.to(int32).contiguous()` | `:687` | **布尔索引 → nonzero + D2H**，且输出长度数据相关 |
| 拼 prior：`torch.cat((prior_k, rows_k), 0)` × 2 | `:696-697` | 变长输入（长度 `phase + T'` 每步变化）+ 2 次 alloc |
| `ape.float().contiguous()` | `:701` | 每次调用重做 |
| `tail_start` 以标量传入，导致只能逐请求调用 | `:705` | 无法整批 |

这些 op 不产生任何计算价值，其中两项还是图捕获的硬阻塞：布尔索引的 `nonzero`（D2H）与变长 `cat`
（形状随步变化）。本单的目标就是让 kernel 直接吃张量、直接在 kernel 内完成环读写，从而把上表整段删掉。

### 2. 怎么扩：kernel 内三阶段，插件只传张量

对每个请求，kernel 内部按固定顺序做三件事（都在 tail 环的**同一页**上，≤ 4 个槽）：

1. **读 prior**：tail 环 offset `[0, phase)`，其中 `phase = (seq_len - query_len) % pool_size`；
2. **压缩本 chunk 内闭合的全部池**：对每个闭合池，4 个 token 中落在 prior 的从阶段 1 的寄存器读，
   落在本 chunk 的从输入行读；
3. **写新的未完成池**：把本 chunk 里属于新未完成池的 `m = min(s, T_live)` 行写到环 offset `[s-m, s)`，
   其中 `s = seq_len % pool_size`（`s == 0` 时不写）。

**顺序约束（必须显式实现）**：阶段 1 必须早于阶段 3。因为尾池写入可能覆盖 prior 占用的槽位
（例：`phase=2, T'=7` → `s=1, m=1`，写 offset 0，而 offset 0 正是 prior）。

**由此带来的两个简化**（相对现有实现）：
- 尾池写入只需写 `≤ pool_size-1` 行，且是「输入切片 → 环页连续槽位」的普通拷贝，
  不再需要 `//`、`remainder`、高级索引，也不需要写满 T 行；
- 不需要外部把 prior 和本 chunk 拼成一条：跨边界池只可能有一个，其 4 个 token 由 kernel 从两个来源直接取。

### 3. 与 08 号的关系

08 号（`dlc_kpool_append_update`）的剩余实质内容就是「tail 环形写入」，已被本篇阶段 1/3 覆盖，
其可选的池读出（S4）不属于本单，继续由普通 index 承担。**08 号请关闭/合并到本单，不要单独实现**
（避免出现「只做 scatter 的环写 kernel」+「只做压缩的 #07」两条需要插件粘合的路径）。

## 背景与功能说明
使用场景：GLM-5.3-Flash 的 11 层 KPool-DSA 索引器每 4 个 token 压缩成 1 个池 key 写入 pool cache；
未满 4 个的残池 token 保存在该请求的 tail 环页里，跨 chunk / 跨 step 续接。
池化是逐通道 softmax 加权求和，**不是**对 128 个通道做 softmax。

当前问题：
- 压缩部分（native `compress_kpool`，`vllm_cl/patch/models/kpool_indexer.py:474`）按池循环，每池一次 Python 迭代；
- 环更新与压缩被拆成 kernel 外的 R 次调用与上表所列的一堆搬运（含一次 D2H），无法进入图捕获。

目标功能：单次调用、整批完成「读 prior → 尾部环写 → 跨边界池与 chunk 内池的压缩写入」，输入输出全部为张量。

非目标：
- 池级打分与 TopK（09/10 号）；
- 池读出（S4，08 号的可选侧）；
- 池槽 / tail 环槽的映射计算本身（19 号；本算子直接消费它们）；
- FP8 index cache 与 Hadamard 旋转（后续独立需求）。

## 期望 API 与函数签名
```python
def dlc_kpool_append_compress(
    keys: torch.Tensor,               # [T, D] bf16，本 chunk 的 index key
    gate_scores: torch.Tensor,        # [T, D] bf16，本 chunk 的 gate score
    ape: torch.Tensor,                # [pool_size, D] fp32，池内位置嵌入
    query_start_loc: torch.Tensor,    # [R+1] int32，本 chunk 每请求的 token 边界
    seq_lens: torch.Tensor,           # [R] int32，处理完本 chunk 后每请求的总长度
    tail_slot_mapping: torch.Tensor,  # [T] int32，tail 环物理槽 page*pool_size + pos%pool_size；-1 = 非活 token
    pool_slot_mapping: torch.Tensor,  # [T] int32，pool cache 扁平物理槽；仅「闭合池的最后 token」有值，其余 -1
    tail_block_table: torch.Tensor,   # [R, B] int32，第 0 列为该请求的 tail 环页号
    tail_cache: torch.Tensor,         # [tail_blocks, 2, pool_size, D] bf16，原位读写
    pool_cache: torch.Tensor,         # [blocks, pools_per_block, D] bf16，原位写
    pool_size: int = 4,
) -> None:
```

## 参数与返回值
输入：
- `keys` / `gate_scores`：`[T, D]`，BF16，`D=128`。`T` 为本 step 的 token 数（可跨池边界、可含多请求）。
- `ape`：`[pool_size, D]`，FP32，按池内位置 `0..pool_size-1` 索引（checkpoint 的 `index_kpool_compress_ape`）。
- `query_start_loc`：`[R+1]`，int32，第 `r` 个请求占用行 `[query_start_loc[r], query_start_loc[r+1])`。
- `seq_lens`：`[R]`，int32，**处理完本 chunk 之后**该请求的总长度（不是 chunk 长度）。
  `query_len[r] = query_start_loc[r+1] - query_start_loc[r]`，`before_len[r] = seq_lens[r] - query_len[r]`。
- `tail_slot_mapping`：`[T]`，int32。活 token 的值为 tail 环物理槽号 `page*pool_size + pos%pool_size`；
  `-1` 表示非活 token（padding / 被 mask），kernel 必须跳过。非活行不参与环写、不触发压缩。
- `pool_slot_mapping`：`[T]`，int32。只有「闭合一个池的最后 token」（即 `pos % pool_size == pool_size-1`）
  才带 pool cache 扁平物理槽号；其余行为 `-1`。**注意：本参数按 token 行索引，不是按池号压紧的表。**
- `tail_block_table`：`[R, B]`，int32，`tail_block_table[r, 0]` 是该请求 tail 环页的池页号。
- `tail_cache`：`[tail_blocks, 2, pool_size, D]`，BF16，平面 0 存原始 K、平面 1 存 gate；**原位读写**。
- `pool_cache`：`[blocks, pools_per_block, D]`，BF16，**原位写**。

属性：
- `pool_size`：默认 4（`index_kpool=4`）。

返回值：无（原位写 `tail_cache` 与 `pool_cache`）。

边界：
- `T == 0`、`R == 0`、`B == 0`、`tail_block_table[r, 0] < 0`：该请求跳过，无操作；
- `query_start_loc[r] == query_start_loc[r+1]`：该请求本 chunk 无 token，跳过；
- `pool_slot_mapping` 在某行给出槽位但该行不是闭合 token：属调用方违约，可忽略或断言（见「异常」）。

## 功能语义
记号：请求 `r`、该请求第 `i` 个输入行（全局行号）、该行对应的全局位置 `pos = before_len[r] + (i - query_start_loc[r])`、
`phase = before_len[r] % pool_size`、`s = seq_lens[r] % pool_size`、环页 `page = tail_block_table[r, 0]`。

阶段 1（读 prior，必须在阶段 3 之前）：
```text
prior_k[r, t] = tail_cache[page, 0, t]     # t ∈ [0, phase)
prior_g[r, t] = tail_cache[page, 1, t]
```

阶段 2（压缩闭合池）：对所有 `pos % pool_size == pool_size-1` 且 `tail_slot_mapping[i] >= 0`
且 `pool_slot_mapping[i] >= 0` 的行，池内位置 `j ∈ [0, pool_size)` 的 token 取：
```text
q = pos - (pool_size - 1) + j
若 q < before_len[r]：  token = prior_*[r, q % pool_size]
否则：                  token = keys/gate_scores[i - (pos - q)]
logits[j, d] = gate[j, d] + ape[j, d]
probs[j, d]  = softmax_j(logits[j, d])            # softmax 沿池内位置 j，不是沿通道 d
pooled[d]    = Σ_j probs[j, d] * key[j, d]
pool_cache[pool_slot_mapping[i] // pools_per_block,
           pool_slot_mapping[i] %  pools_per_block] = pooled
```

阶段 3（写新未完成池）：
```text
若 s == 0：不写
否则：m = min(s, 本请求本 chunk 的活 token 数)
     取本请求最后 m 个活 token（按行序），第 t 个写到环 offset = s - m + t
     tail_cache[page, 0, s - m + t] = keys[row]
     tail_cache[page, 1, s - m + t] = gate_scores[row]
```
（`t ∈ [0, m)`，写入的 offset 区间为 `[s-m, s)`，与 `tail_slot_mapping` 在该窗口的取值必须一致。）

broadcasting：`ape` 沿池内位置广播到 `[pool_size, D]`。
in-place / alias：只写 `tail_cache` 与 `pool_cache`，不修改任何输入。
空 / 零长度：见「边界」。
异常：`keys.shape != gate_scores.shape`、`ape.shape != [pool_size, D]`、`query_start_loc.numel() != R+1`、
`tail_slot_mapping.numel() != T`、`pool_slot_mapping.numel() != T`、`tail_slot_mapping` 给出的槽位越界：报错。
确定性：无随机；softmax 与加权求和用 FP32，结果 cast 回 BF16 写入。
累加精度：FP32。

## 支持范围与约束
支持的 dtype：`keys/gate_scores/tail_cache/pool_cache` BF16；`ape` FP32；三个 mapping / 表 int32。内部 FP32。
支持的 shape：`T` 动态（覆盖到 32K）；`R` 动态（覆盖到最大 batch）；`D=128`（建议支持 64~256 泛化）；
`pool_size` 默认 4；`pools_per_block = block_size/pool_size`（GLM-5.3-Flash `block_size=256` → 64）；
`B` = tail 环页表的列数（GLM-5.3-Flash 只需第 0 列）。
数值范围：`gate_scores` BF16 常规范围；`ape` FP32 常规范围。
layout：`keys/gate_scores` 连续 `[T,D]`；`tail_cache` 最后一维连续、形状 `[tail_blocks, 2, pool_size, D]`；
`pool_cache` 最后一维连续；三个 mapping / `query_start_loc` / `seq_lens` 连续；`tail_block_table` 行连续。
device：DLC 单卡。
执行模式：**必须可在 DLCGraph 捕获内执行**（本单的核心目的之一）。
其他限制：
- 每个请求的 tail 环只占一页、只有 `pool_size` 个槽，**同一个请求的 prior 与新尾池共用这 4 个槽**，
  因此阶段 1 与阶段 3 的顺序不可交换；
- 调用方保证环页 offset `[0, phase)` 是该请求真实的 prior（调度侧会保留残池，见插件 docstring）；
- BF16 bring-up 阶段不需要 Hadamard 旋转与 FP8 量化（`dot(Hq,Hk)=dot(q,k)`）。

## 精度要求
Golden / reference：本文档「Reference、日志与附件」里的 `kpool_append_compress_ref`（纯 PyTorch，CPU 可跑）。
输入分布：`keys ~ N(0,1)`、`gate_scores ~ N(0,1)`、`ape ~ N(0,0.1)`；另含全相同 gate（softmax 退化为等权均值）与极端 gate 差。
样本数量：≥ 200 组，覆盖 `phase ∈ {0,1,2,3}` × `T % pool_size ∈ {0,1,2,3}` × 单请求/多请求。
比较 dtype：BF16 输出对 FP32 reference。
atol / rtol：`atol=1e-2, rtol=1e-2`。
必须逐元素一致：
- `pool_cache` 的每个被写入槽位与 reference 逐位一致；
- `tail_cache` **只比较活槽位**：调用后环页 offset `[0, s)` 必须与 reference 一致。
  offset `>= s` 属于陈旧值，**不要求**保持原值、也不要求清空（现有实现的 scatter 会写满 T 行、留下别的陈旧值，
  新实现只写 `[s-m, s)`，两者在这些槽位上必然不同，但都不会被后续读到）；
- softmax 沿**池内位置**（不是通道）这一语义必须与 reference 一致，用「gate 沿通道构造单调差异」的定向用例验证。
NaN / Inf / 极值：全 0 gate 时 pooled 等于 4 token 的等权均值。
累加精度：FP32。

## 测试范围
- [x] 基本功能与 reference 对齐
- [ ] 多 dtype 精度
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [ ] 非 contiguous、特殊 stride 或 layout
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式
- [ ] 长时间稳定性
- [x] 性能基准与性能回归
- [x] 图捕获（在 `torch.dlc.graph` 捕获上下文内执行并重放）

具体测试用例、shape、数据范围和预期结果：
1. 跨边界池：`phase ∈ {0,1,2,3}` 各取 `T ∈ {1,2,3,5,7}`，验证「prior + 本 chunk」拼出的池与 reference 一致。
2. 尾池写入：`seq_len % pool_size ∈ {0,1,2,3}`，验证 (a) `[s-m, s)` 写入本 chunk 最后 m 个活 token；
   (b) `[0, s-m)` 保持调用前的值（新未完成池的前缀来自 prior 或更早的写入，不能被覆盖）；
   `>= s` 的槽位不作要求。
3. 顺序约束：`phase=2`、chunk 恰好闭合一个池且 `s=1`（即 `T=7`）——此时尾池写入会覆盖 prior 的 offset 0，
   用于验证「先读后写」；若实现顺序颠倒，该用例必失败。
4. 多请求：`R ∈ {1, 8, 32}`，请求间长度不同、含零 token 请求、含 `tail_block_table[r,0] = -1` 的请求。
5. 非活 token：`T` 行中掺入 `tail_slot_mapping = -1` 的 padding 行，验证既不触发压缩也不写环。
6. 语义定向：沿通道单调差异 + 池内均匀 gate，验证 softmax 维度正确。
7. 边界：`T=0`、`R=0`、`D=128`、`pool_slot_mapping` 全 -1（无池闭合）。
8. 性能：`T=4096`、`R=1` 与 `T=32`、`R=32` 两种形态的单层耗时，对比插件现状（目标 ≥ 3×，
   且调用次数从 R 次降到 1 次）。
9. 图捕获：在 `torch.dlc.graph` 内捕获本算子并重放，输出与 eager 一致。

## 验证环境与依赖版本
DLC 硬件：8 卡，设备名 `/dev/cltech0-7`（TP=8，无 PP）
vLLM commit：`f5361fb09f`
vLLM DLC（vllm-cl）commit：`bab8586`
PyTorch / torch-dlc：torch 2.5.0（DLC 构建），Python 3.10.16
关键环境变量：`VLLM_USE_DLC_COL_MAJOR_MATMUL=1`
模型：GLM-5.3-Flash-Mix（`index_kpool=4`、`index_head_dim=128`、`block_size=256`、11 层带 indexer）

## 验收标准
- [ ] API、参数校验和错误行为符合约定
- [ ] 规定范围内的功能和精度测试通过
- [ ] 需求中明确列出的特殊运行场景通过（尤其：跨边界池、先读后写顺序、尾池只写 `[s-m, s)`）
- [ ] 可在 `torch.dlc.graph` 捕获内执行并重放
- [ ] 文档、测试和示例已提交
- [ ] 接入 vllm-cl 后，插件侧删除 `prior_*.clone()`、布尔索引压紧槽位表、`torch.cat`、逐请求调用与
      `tail_start` 标量，`native_kpool_indexer_step` 内不再有 kernel 外搬运
- [ ] GLM-5.3-Flash 在 DLC 上 prefill/decode 数值与 PyTorch reference 一致

## Reference、日志与附件

### Golden reference（直接可跑，CPU 即可）
```python
import torch


def kpool_append_compress_ref(
    keys,               # [T, D] bf16
    gate_scores,        # [T, D] bf16
    ape,                # [pool_size, D] fp32
    query_start_loc,    # [R+1] int32/int64
    seq_lens,           # [R] int32/int64，处理完本 chunk 之后的总长度
    tail_slot_mapping,  # [T] int32，-1 = 非活 token
    pool_slot_mapping,  # [T] int32，仅闭合池的最后 token 有值
    tail_block_table,   # [R, B] int32，第 0 列为 tail 环页号
    tail_cache,         # [tail_blocks, 2, pool_size, D] bf16，原位读写
    pool_cache,         # [blocks, pools_per_block, D] bf16，原位写
    pool_size=4,
):
    pools_per_block = pool_cache.shape[1]
    T = keys.shape[0]
    R = int(seq_lens.numel())
    assert keys.shape == gate_scores.shape
    assert tail_slot_mapping.numel() == T and pool_slot_mapping.numel() == T
    assert query_start_loc.numel() == R + 1
    assert ape.shape[0] == pool_size and ape.shape[1] == keys.shape[1]

    for r in range(R):
        start = int(query_start_loc[r])
        stop = int(query_start_loc[r + 1])
        if stop <= start:
            continue
        end_len = int(seq_lens[r])
        query_len = stop - start
        before_len = end_len - query_len
        phase = before_len % pool_size
        page = int(tail_block_table[r, 0])
        if page < 0:
            continue

        # 阶段 1：读 prior（环 offset [0, phase)），必须在阶段 3 之前
        prior_k = tail_cache[page, 0, :phase].clone()
        prior_g = tail_cache[page, 1, :phase].clone()

        # 阶段 2：压缩本 chunk 内闭合的全部池
        for i in range(start, stop):
            if int(tail_slot_mapping[i]) < 0:
                continue                       # 非活 token
            pos = before_len + (i - start)
            if pos < 0 or pos % pool_size != pool_size - 1:
                continue                       # 只有闭合池的最后 token 触发压缩
            slot = int(pool_slot_mapping[i])
            if slot < 0:
                continue
            tok_k, tok_g = [], []
            for j in range(pool_size):
                pos_j = pos - (pool_size - 1) + j          # 池内第 j 个 token 的全局位置
                if pos_j < before_len:                     # 落在 prior 里
                    tok_k.append(prior_k[pos_j % pool_size])
                    tok_g.append(prior_g[pos_j % pool_size])
                else:                                      # 落在本 chunk 输入里
                    row = i - (pos - pos_j)
                    tok_k.append(keys[row])
                    tok_g.append(gate_scores[row])
            k4 = torch.stack(tok_k).float()                # [pool_size, D]
            g4 = torch.stack(tok_g).float()
            probs = torch.softmax(g4 + ape.float(), dim=0)  # softmax 沿池内位置
            pooled = (k4 * probs).sum(0).to(pool_cache.dtype)
            block, off = divmod(slot, pools_per_block)
            pool_cache[block, off] = pooled

        # 阶段 3：写新的未完成池（环 offset [s-m, s)）
        s = end_len % pool_size
        if s:
            live = [i for i in range(start, stop) if int(tail_slot_mapping[i]) >= 0]
            m = min(s, len(live))
            for t, row in enumerate(live[-m:]):
                tail_cache[page, 0, s - m + t] = keys[row]
                tail_cache[page, 1, s - m + t] = gate_scores[row]

    return tail_cache, pool_cache
```

### reference 自身的自检结论（供算子组确认语义用）

上面的 reference 已用两种方式自检通过：

1. **单次调用**（用例脚本）：跨边界池、尾池写入范围、先读后写顺序三项断言全过。
2. **多步连续调用**：随机切分同一条序列成 1~10 个 chunk（600 组，`total ∈ [1,96]`，D=16），
   按 chunk 顺序逐次调用本 reference，最终 `pool_cache` 与「把整条序列按每 4 个 token 直接压缩」的
   独立实现**逐位一致（最大误差 0.0）**，每步后环的活槽位 `[0, s)` 也与该独立实现一致。

第 2 项说明「跨 chunk 续接」的语义是自洽的：只要按本文的三阶段实现，任何切分方式都得到同一个池化结果。

### 参考用例（覆盖跨边界池、尾池写入、先读后写）
```python
A, D, B, PB = 4, 8, 4, 3        # pool_size / head_dim / tail 页数 / pool_cache 每块槽数


def build(phase, chunk, prior_rows, page=1, pool_slot_base=5):
    """构造单个请求、单次调用的输入；tail 页其余槽位是哨兵值。"""
    T = len(chunk)
    end_len = phase + T
    before_len = end_len - T
    keys = chunk.clone()                       # bf16
    gates = (chunk * 0.3 + 0.7).to(torch.bfloat16)
    ape = torch.randn(A, D)
    tail = torch.randn(B, 2, A, D, dtype=torch.bfloat16)
    for t in range(phase):                     # 环里已存的 prior（按 bf16 落盘）
        tail[page, 0, t] = prior_rows[t]
        tail[page, 1, t] = prior_rows[t] * 0.5
    sentinel = tail.clone()                    # 调用前的环内容，用于取 prior 真值
    tail_slot = torch.zeros(T, dtype=torch.int32)
    pool_slot = torch.full((T,), -1, dtype=torch.int32)
    for i in range(T):
        pos = before_len + i
        tail_slot[i] = page * A + pos % A
        if pos % A == A - 1:
            pool_slot[i] = pool_slot_base + pos // A
    case = dict(
        keys=keys, gate_scores=gates, ape=ape,
        query_start_loc=torch.tensor([0, T], dtype=torch.int32),
        seq_lens=torch.tensor([end_len], dtype=torch.int32),
        tail_slot_mapping=tail_slot, pool_slot_mapping=pool_slot,
        tail_block_table=torch.tensor([[page]], dtype=torch.int32),
        tail_cache=tail, pool_cache=torch.zeros(4, PB, D, dtype=torch.bfloat16),
    )
    return case, sentinel


# 关键用例：phase=2、chunk=7 → s=1, m=1，尾池写入落在 offset 0，而 offset 0 正是 prior → 必须「先读后写」
chunk = torch.randn(7, D, dtype=torch.bfloat16)
prior = torch.randn(2, D, dtype=torch.bfloat16)
case, before = build(phase=2, chunk=chunk, prior_rows=prior)
kpool_append_compress_ref(**case, pool_size=A)
tail, pool = case["tail_cache"], case["pool_cache"]

# 检查 1：跨边界池（pos=3 闭合，槽位 5）的 4 个 token = prior[0:2]（环 offset 0/1）+ chunk 行 0/1（pos 2/3）
k4 = torch.stack([before[1, 0, 0], before[1, 0, 1], case["keys"][0], case["keys"][1]]).float()
g4 = torch.stack([before[1, 1, 0], before[1, 1, 1], case["gate_scores"][0], case["gate_scores"][1]]).float()
exp = (k4 * torch.softmax(g4 + case["ape"].float(), dim=0)).sum(0).to(pool.dtype)
assert torch.equal(pool[5 // PB, 5 % PB], exp)

# 检查 2：尾池只写 offset [s-m, s) = [0,1)；offset 1/2/3 不在契约内（陈旧值，可保留可覆盖）
assert torch.equal(tail[1, 0, 0:1], case["keys"][-1:])
```
### 相关实现与文档
- 现有已交付算子入口：`torch.dlc_kpool_compress`
  （`aten/src/ATen/native/dlc/kpool_compress.cpp`、schema 见 `native_functions.yaml:17108`；
  kernel 语义见 `DLC_Custom_Kernel/syntests/pooling/kpool_compress_write_test.cpp:56-66`——
  它按 `mapping[p]` 以池号索引、`keys[p*pool_size+r]` 按连续行分组，这正是本篇要扩展掉的「外部压紧 + 外部拼接」）。
- 插件调用点与现状：`vllm_cl/patch/models/kpool_indexer.py:651`（`_seed_kpool_fused`）、
  `:1005-1027`（逐请求调用）、`:474`（native `compress_kpool`）。
- cache 布局：pool `[blocks, 64, 128]`、tail `[blocks, 2, 4, 128]`（`vllm_cl/patch/models/glm5_indexer.py:67/:103`）。
- 相关需求与文档：`08_dlc_kpool_append_update.md`（并入本篇，作废）、`19_dlc_kpool_slot_mapping.md`（上游 slot 映射）、
  `../kpool_mla_dataflow.md` §5 S2/S3。
