<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22）：完全新算子 -->
<!-- 已有算子覆盖核查（2026-09-21，DLC 实测）：未覆盖（保留新增）
     证据：现有 dlc_multi_query_attention 是「先按 head 加权再点积」且无 ReLU，语义不等价 -->
<!-- 2026-10-09 修订：把 10_dlc_kpool_expand_topk.md 的「展开 + 残尾拼接」以及插件侧的
     「池 gather / valid_pools 构造 / 逐行 topk / 逐行调用」全部并入本单，ABI 从
     「单请求、只出 [N,P] logits」扩展为「整批、直接出 topk token buffer、kernel 内按 block table 读 cache」。
     已交付的 torch.dlc_kpool_mqa_logits（单请求 logits）保留为原型入口，语义不变。
     torch.dlc_vllm_kpool_expand_topk（10 号）退出主路径，保留给其他调用方。 -->

# [Operator Request] dlc_kpool_select_topk（KPool 池级打分 + TopK 选池 + 展开）

## 需求类型
扩展已有算子能力

**需求分类：扩展已有算子（把 10 号并入 09 号）**
- 覆盖核查（2026-09-21）：现有 `torch.dlc_multi_query_attention` 是「先按 head 加权再点积」且无 ReLU，
  与「逐 head 点积 → ReLU → 按 head 权重求和」不等价，故 09-21 判定为新增算子（P0）。
- 2026-10-09 复核：`torch.dlc_kpool_mqa_logits` 已交付并接入。本次不新增算子，而是**扩展它的边界**：
  输入侧吸收「按 block table 从 pool cache 直接读池」（不再由插件 gather），
  输出侧吸收 10 号的「展开 + 残尾拼接」，中间不再落地 `[T, P]` logits。10 号独立实现不再需要。

## 优先级
P0

**定级理由（2026-10-09）**：
- 现状每层每步约 **500 个小 op + 64 次 kernel 下发**（decode R=T=32），prefill（T=4096、R=1）
  约 **2.5 万个小 op + 4096 次 kernel 下发**，是 07 号之外最大的一块 kernel 外编排；
- 长上下文 prefill 会因 `[T, P]` fp32 中间量超限而**退回逐行 einsum**（见「本次修订」§3），当前就是最慢路径；
- 07 号扩展后插件侧的 per-request `plans` 消失，这一段将失去 host 标量驱动，必须同步改为张量驱动。

## 目标版本与期望时间
GLM-5.3-Flash 适配 Phase 2（性能阶段）。建议与 07 号的扩展同批排期（两者一起改完，
indexer 层的 kernel 外编排才能清零），具体里程碑待算子组排期确认。

## 算子名称
**推荐**：`torch.dlc_kpool_select_topk`（`dlc_kpool_mqa_logits` 的扩展入口，语义为它的超集）

> 命名不影响需求内容：也可在 `torch.dlc_kpool_mqa_logits` 上以可选参数形式扩展。
> 已交付的单请求原型 `(query, pooled_keys, head_weights, valid_pools, scale, head_weight_scale) -> [N, P]`
> **保持可用**（现有单测与对照实现仍走它）。

## 本次修订（2026-10-09，实现前必读）

### 1. 为什么必须扩：S4–S7 全是 kernel 外编排

| 当前编排（`vllm_cl/patch/models/kpool_indexer.py`） | 位置 | 每请求/每行代价 |
|---|---|---|
| 池读出：`arange` + `//` + `index_select` + `.long()` + `remainder` + 高级 gather | `:1052-1055` | 6 个小 op/请求，并落地 `[P, D]` gather 张量 |
| 构造 `valid_pools`：`arange` + add + `//` + `to(int32)` | `:711-714` | 4 个小 op/请求 |
| 逐行选池：`torch.topk(k = min(512, visible))`，k 是 host 标量 | `:1081` | 1 次小 kernel + 1 次 alloc/行 |
| 逐行展开：`#10` kernel（`tail_start` / `sequence_length` 是标量，只能逐行调） | `:1087` | 1 次 kernel 下发/行 |
| 逐行回写：`result[row, :width].copy_(expanded)` | `:1102` | 1 次 copy/行 |
| `out.fill_(-1)` | `:993` | 1 次全缓冲写/层 |

这些 op 不产生计算价值；其中「每行一次 kernel 下发」是下发耗时的主要来源。

### 2. 怎么扩：一个 kernel 从 q/weights 直接算到 topk buffer

kernel 内每行（token）做完整条链，**不落地任何中间张量**：

1. 由 `query_start_loc` / `seq_lens` 推该行的 `r`（请求）、`pos`（全局位置）、`sequence_length = pos + 1`；
2. `visible = sequence_length // pool_size`；
3. 对 `k ∈ [0, visible)`：**直接按地址算术读** `pool_cache[block_table[r, k // pools_per_block], k % pools_per_block]`，
   **不需要** 插件先 gather 成 `pooled_keys [P, D]`；
4. 打分（公式见「功能语义」）→ 取 `k_sel = min(topk_tokens // pool_size, visible)` 个最大的池；
5. 展开成 token id、按 `sequence_length` 截断，再拼残尾，直接写入 `out` 的该行。

### 3. 顺带解决的两个问题

- **`[T, P]` fp32 中间量消失**：现在 `_FUSED_SCORE_MAX_ELEMENTS = 32 * 1024 * 1024`（`:628`）限制了
  `query_rows × P ≤ 32M` 才走融合打分。4096-token prefill（P=1024）是 4.2M，走融合但落地 16 MB；
  32K prefill（P=8192）是 268M > 32M，**直接退回 `kpool_scores` 逐行 einsum**（`:1074`）。
  本扩展后 logits 只存在于寄存器/VMEM，长度不再受限，该门限可以删掉。
- **池 gather 张量消失**：地址算术即可寻址，省掉 `[P, D]` 的落地与读写。

### 4. 与 10 号的关系

10 号（`dlc_vllm_kpool_expand_topk`）的语义（池 id → 4 个 token id、按 `sequence_length` 截断、
追加残尾、无效槽 `-1`）被本篇第 5 步完整吸收。**10 号请标注为「退出主路径」**，
保留入口供其他调用方；不要再为「topk 之后再展开」单独维护一条需要插件粘合的路径。

## 背景与功能说明
使用场景：GLM-5.3-Flash 的 11 层 KPool-DSA 索引器，每层每步要为每个 query token 选出
最相关的 `index_topk = 2048` 个 token（先按池选 512 个池，再展开回 2048 个 token），
交给稀疏 MLA 做 attention。

当前问题：见「本次修订」§1 的编排清单；另外长上下文 prefill 会掉进逐行 einsum。

目标功能：单次调用、整批完成「按 block table 读池 cache → 池级打分 → TopK 选池 → 展开并拼残尾 → 写 topk buffer」。

非目标：
- 池化压缩与 tail 环更新（07 号扩展，`dlc_kpool_append_compress`）；
- 池槽 / tail 环槽映射（19 号）；
- 稀疏 MLA attention 本身（11 号）与 latent cache 写入（12 号）；
- DCP/PCP > 1 的分片语义（本阶段 DCP = PCP = 1）。

## 期望 API 与函数签名
```python
def dlc_kpool_select_topk(
    query: torch.Tensor,             # [T, H, D] bf16，indexer query
    head_weights: torch.Tensor,      # [T, H] fp32（或 bf16），每 token 的每 head 权重
    pool_cache: torch.Tensor,        # [blocks, pools_per_block, D] bf16，压缩后的池向量
    block_table: torch.Tensor,       # [R, B] int32，逻辑池块 → 物理池块（每块 pools_per_block 个池）
    query_start_loc: torch.Tensor,   # [R+1] int32，每请求的 token 边界
    seq_lens: torch.Tensor,          # [R] int32，处理完本 step 后每请求的总长度（未压缩的 token 数）
    out: torch.Tensor,               # [T, W] int32，W >= topk_tokens + pool_size - 1（GLM-5.3-Flash 用 2176）
    topk_tokens: int = 2048,
    pool_size: int = 4,
    scale: float | None = None,           # 默认 D ** -0.5
    head_weight_scale: float | None = None,  # 默认 H ** -0.5
) -> None:
```

## 参数与返回值
输入：
- `query`：`[T, H, D]`，BF16。`T` 为本 step 的 token 数（含多请求、可含 padding 行）。
- `head_weights`：`[T, H]`，FP32（GLM-5.3-Flash 生产为 fp32，为排序精度；也接受 BF16）。
- `pool_cache`：`[blocks, pools_per_block, D]`，BF16，只读。
- `block_table`：`[R, B]`，int32。第 `r` 行第 `j` 列是「该请求第 `j` 个池块」的物理块号；
  逻辑池 `pid` 的物理位置 = `block_table[r, pid // pools_per_block]`，块内偏移 `pid % pools_per_block`。
- `query_start_loc`：`[R+1]`，int32；`seq_lens`：`[R]`，int32（**处理完本 step 之后**的总长度，不是 chunk 长度）。
- `out`：`[T, W]`，int32，`W >= topk_tokens + pool_size - 1`。**原位写**。

属性：
- `topk_tokens`：默认 2048（`index_topk`）。
- `pool_size`：默认 4（`index_kpool`）。
- `scale` / `head_weight_scale`：默认 `D**-0.5` / `H**-0.5`。

返回值：无（原位写 `out`）。

边界：
- `T == 0`、`R == 0`：无操作；
- 请求间 `query_start_loc` 可重复（零 token 请求），不产生行；
- `visible == 0`（`sequence_length < pool_size`）：不选池，只写残尾；
- `out` 的整行 `[0, W)` 由 kernel 先填 `-1`，包括 `[topk_tokens + pool_size - 1, W)` 的对齐填充区
  （这样插件侧的 `out.fill_(-1)` 可以删掉）。

## 功能语义
记号：行 `i` 属于请求 `r`（`query_start_loc[r] <= i < query_start_loc[r+1]`），
`query_len = query_start_loc[r+1] - query_start_loc[r]`，`before_len = seq_lens[r] - query_len`，
`pos = before_len + (i - query_start_loc[r])`，`sequence_length = pos + 1`，`visible = sequence_length // pool_size`。

打分（沿用现有 `kpool_scores` / `dlc_kpool_mqa_logits` 的语义，逐 head ReLU 后加权求和，FP32 累加）：
```text
pooled[k] = pool_cache[block_table[r, k // pools_per_block], k % pools_per_block]   # k ∈ [0, visible)
dot[h, k] = Σ_d query[i, h, d] * pooled[k, d]
scores[k] = Σ_h relu(dot[h, k] * scale) * (head_weights[i, h] * head_weight_scale)
```

选池与展开：
```text
k_sel = min(topk_tokens // pool_size, visible)
sel   = scores 中最大的 k_sel 个池（降序；并列的处理见下）
out[i, 0 : W] = -1
for s in range(k_sel):
    for j in range(pool_size):
        idx = s * pool_size + j
        if idx >= topk_tokens: break
        tok = sel[s] * pool_size + j
        out[i, idx] = tok if 0 <= tok < sequence_length else -1
tail_len = sequence_length % pool_size          # 0..pool_size-1
out[i, topk_tokens : topk_tokens + tail_len] = arange(visible * pool_size, sequence_length)
```

说明：
- `sel[s] < visible` 保证其 4 个 token 都 `< sequence_length`，因此**选中池不会触发截断**；
  截断判断是为 `-1`/越界 id 兜底（本算子里不会出现，但保留判据以便与 reference 一致）。
- **并列（tie）处理**：分数并列时只要求「选出的池 id 集合一致」，顺序不作要求
  （下游稀疏 attention 把 topk 列表当集合用，顺序不影响数值）。
  若要逐位可复现，建议用「分数降序、同分按池 id 升序」——与本文 reference 一致。
  bf16 输入下并列会偶发（实测 300 组随机用例里 4 组），验收时请按上述规则区分。

broadcasting：`query` / `head_weights` 逐行独立；不同行的 `visible`、`k_sel` 不同。
in-place / alias：只写 `out`，不修改任何输入。
空 / 零长度：见「边界」。
异常：`query.ndim != 3`、`head_weights.shape != query.shape[:2]`、`pool_cache.shape[2] != D`、
`query_start_loc.numel() != R+1`、`seq_lens.numel() != R`、`out.shape[0] < T`、
`W < topk_tokens + pool_size - 1`、`block_table.shape[0] < R`：报错。
`block_table` 越界：建议按非法参数报错（不要静默读越界）。
确定性：无随机；打分累加 FP32。
累加精度：FP32（与现有实现一致，避免 bf16 近并列的排序错误）。

## 支持范围与约束
支持的 dtype：`query` / `pool_cache` BF16；`head_weights` FP32 或 BF16；`out` int32；`query_start_loc` / `seq_lens` / `block_table` int32。内部 FP32。
支持的 shape：`T` 动态（覆盖到 32K）；`R` 动态；`H = 32`（建议支持 16~64）；`D = 128`（建议支持 64~256）；
`topk_tokens = 2048`（建议支持任意 `pool_size` 的倍数）；`pool_size = 4`；
`pools_per_block = block_size / pool_size`（GLM-5.3-Flash `block_size=256` → 64）；`B` 动态。
数值范围：`sequence_length ≤ 1M`；`out` 中 `-1` 为无效哨兵。
layout：`query` 连续 `[T, H, D]`；`pool_cache` 最后一维连续；`out` 行连续；三个 metadata 张量连续。
device：DLC 单卡。
执行模式：需支持 DLCGraph 捕获与重放（本单目标之一：indexer 层从「每行一次下发」变成「每层一次下发」）。
其他限制：
- DCP/PCP 必须为 1（分片语义不在本单）；
- `block_table` 只在 `visible` 范围内被访问；`visible` 超过 `B * pools_per_block` 属调用方违约；
- 输出宽度：GLM-5.3-Flash 的 topk buffer 宽 2176（2051 向上对齐到 128），kernel 需要把整行填满（含对齐区 `-1`）。

## 精度要求
Golden / reference：本文档「Reference、日志与附件」里的 `kpool_select_topk_ref`（纯 PyTorch，CPU 可跑）。
输入分布：`query / pool_cache ~ N(0,1)`（bf16）、`head_weights ~ N(0,1)`（fp32）；
`sequence_length` 覆盖 `0 ~ 4K`（含 `< pool_size` 与各余数）；`visible` 覆盖 `< k_sel`、`== k_sel`、`> k_sel`。
样本数量：≥ 200 组；另加多请求（R ∈ {1,8,32}、含零 token 请求）与 `-1`/越界兜底用例。
比较 dtype：int32 逐元素。
atol / rtol：不适用；**要求 bit-exact**（并列导致的顺序差异除外，见「功能语义」的 tie 规则）。
必须逐元素一致：池 id 选择结果、token id 展开、残尾位置与长度、`-1` 的位置、整行（含对齐填充区）的填充。
NaN / Inf / 极值：`sequence_length < pool_size` 时只输出残尾；`visible == 0` 时 `[0, topk_tokens)` 全 `-1`。

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
1. 功能：`T ∈ {1, 64, 4096}`、`R ∈ {1, 8, 32}`，`sequence_length % 4 ∈ {0,1,2,3}`，与 reference 逐位一致。
2. 选池规模：`visible < k_sel`（未选满）、`visible == k_sel`、`visible > k_sel`（选满 512）三种都要覆盖。
3. 边界：`sequence_length < 4`、`sequence_length == 0`、`T=0`、`R=0`、零 token 请求。
4. 输出宽度：`W = 2051` 与 `W = 2176` 两种都测；校验 `[2051, 2176)` 也被填 `-1`。
5. 并列：构造分数并列的输入，校验「选出的池 id 集合一致」（顺序不作要求）；若实现采用
   「降序 + 同分按池 id 升序」，则该用例应逐位一致。
6. bit-exact 校验、异常、图捕获（捕获 + 重放与 eager 一致）。
7. 性能：`T=4096, R=1`（prefill）与 `T=32, R=32`（decode）两种形态的单层耗时，
   对比插件现状（目标 ≥ 5×，且 kernel 下发次数从 R+T 次降到 1 次）。

## 验证环境与依赖版本
DLC 硬件：8 卡，设备名 `/dev/cltech0-7`（TP=8，无 PP）
vLLM commit：`f5361fb09f`
vLLM DLC（vllm-cl）commit：`bab8586`
PyTorch / torch-dlc：torch 2.5.0（DLC 构建），Python 3.10.16
关键环境变量：`VLLM_USE_DLC_COL_MAJOR_MATMUL=1`
模型：GLM-5.3-Flash-Mix（`index_topk=2048`、`index_kpool=4`、`index_n_heads=32`、`index_head_dim=128`、11 层带 indexer）

## 验收标准
- [ ] API、参数校验和错误行为符合约定
- [ ] 规定范围内的功能和精度测试通过（含 `visible` 三种规模、残尾、对齐填充区、并列规则）
- [ ] 需求中明确列出的特殊运行场景通过
- [ ] 可在 `torch.dlc.graph` 捕获内执行并重放
- [ ] 文档、测试和示例已提交
- [ ] 接入 vllm-cl 后，插件侧删除池 gather、`valid_pools` 构造、逐行 `torch.topk`、逐行 `#10` 调用、
      逐行 `copy_` 与 `out.fill_(-1)`，`_FUSED_SCORE_MAX_ELEMENTS` 门限删除
- [ ] GLM-5.3-Flash 在 DLC 上 prefill/decode 数值与 PyTorch reference 一致

## Reference、日志与附件

### Golden reference（直接可跑，CPU 即可）
```python
import torch


def kpool_select_topk_ref(
    query,                # [T, H, D] bf16
    head_weights,         # [T, H] fp32（或 bf16）
    pool_cache,           # [blocks, pools_per_block, D] bf16
    block_table,          # [R, B] int32
    query_start_loc,      # [R+1] int32
    seq_lens,             # [R] int32，处理完本 step 之后的总长度
    out,                  # [T, W] int32，原位写
    topk_tokens=2048,
    pool_size=4,
    scale=None,
    head_weight_scale=None,
):
    T, H, D = query.shape
    PPB = pool_cache.shape[1]
    R = int(seq_lens.numel())
    if scale is None:
        scale = D ** -0.5
    if head_weight_scale is None:
        head_weight_scale = H ** -0.5
    select_k = topk_tokens // pool_size
    out.fill_(-1)                                  # 整行（含对齐填充区）先填 -1

    bounds = query_start_loc.to(torch.int64)
    for i in range(T):
        # 行 → 请求：与 19 号同一套 query_start_loc 解析
        r = int(torch.searchsorted(bounds, torch.tensor(i), right=True)) - 1
        if r < 0 or r >= R:
            continue
        start = int(bounds[r])
        if i < start or i >= int(bounds[r + 1]):
            continue
        qlen = int(bounds[r + 1]) - start
        before = int(seq_lens[r]) - qlen
        pos = before + (i - start)
        if pos < 0:
            continue                               # padding 行
        seql = pos + 1
        visible = seql // pool_size
        k_sel = min(select_k, visible)

        sel = []
        if k_sel:
            # 直接按地址算术读池，不落地 gather 张量
            pooled = torch.stack([
                pool_cache[int(block_table[r, k // PPB]), k % PPB] for k in range(visible)
            ])
            dots = torch.einsum("hd,pd->hp", query[i].float(), pooled.float()) * scale
            dots = torch.relu(dots)
            scores = (dots * (head_weights[i].float() * head_weight_scale).unsqueeze(-1)).sum(0)
            order = torch.argsort(-scores, stable=True)      # 同分时保持池 id 升序
            sel = [int(x) for x in order[:k_sel]]

        width = 0
        for s in sel:
            for j in range(pool_size):
                tok = s * pool_size + j
                out[i, width] = tok if 0 <= tok < seql else -1
                width += 1
                if width >= topk_tokens:
                    break
            if width >= topk_tokens:
                break

        tail_len = seql % pool_size                # 0..pool_size-1
        if tail_len:
            out[i, topk_tokens:topk_tokens + tail_len] = torch.arange(
                visible * pool_size, seql, dtype=out.dtype)
    return out
```

### reference 自身的自检结论（供算子组确认语义用）

上面的 reference 与**插件现有生产路径**（池 gather + `kpool_scores` + `torch.topk` + `expand_pool_topk`
逐行组合，即 `native_kpool_indexer_step` 的现有语义）做过对拍：300 组随机用例
（`R ∈ [1,5]`、`len ∈ [0,40)`、D=16、H=8、`topk_tokens=64`、`W=2176`）：

- 296 组**逐位一致**；
- 4 组只有「分数并列时选出的顺序」不同，**选出的 token 集合完全一致**（例：`scores` 里
  最大值与第 9 名相等且 `k_sel == visible`，两种排序都选中同一批池）；
- 集合不同的组数 = 0。

即：除并列顺序外，本 reference 与现有生产实现等价；插件接入本算子前后，数值行为不变。

### 相关实现与文档
- 现有已交付入口：`torch.dlc_kpool_mqa_logits`
  （`aten/src/ATen/native/dlc/KPoolMQALogits.cxx`、输入校验 `aten/src/ATen/native/KPoolMQALogits.h`）。
- 被吸收的 10 号：`torch.dlc_vllm_kpool_expand_topk`（语义见 `10_dlc_kpool_expand_topk.md`）。
- 插件调用点：`vllm_cl/patch/models/kpool_indexer.py:709`（`_scores_kpool_fused`）、`:1052-1102`（S4–S7）、
  `:554`（`kpool_scores`）、`:583`（`expand_pool_topk`）、`:628`（`_FUSED_SCORE_MAX_ELEMENTS`）。
- 相关需求与文档：`07_dlc_kpool_compress.md`（上游 07 号扩展；本单的前一段）、
  `10_dlc_kpool_expand_topk.md`（退出主路径）、`19_dlc_kpool_slot_mapping.md`（slot 映射）、
  `../kpool_mla_dataflow.md` §5 S4–S7。
