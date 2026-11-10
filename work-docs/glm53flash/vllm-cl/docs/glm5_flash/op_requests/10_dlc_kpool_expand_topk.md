<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22）：完全新算子 -->
<!-- 已有算子覆盖核查（2026-09-21，DLC 实测）：未覆盖（保留新增）
     证据：无 pool id→token id 展开算子；dlc_vllm_expand_kernel_ 只做值替换
     建议：保留新增（P1） -->
<!-- 2026-10-09 修订：本单**退出主路径**，不再单独实现。
     「池 id → token id 展开 + 按 sequence_length 截断 + 残尾拼接 + 整行填 -1」已并入
     09_dlc_kpool_mqa_logits.md 的扩展（torch.dlc_kpool_select_topk），
     由同一个 kernel 在选池之后直接写出 topk buffer，中间不再落地 [T,P] logits。
     已交付的 torch.dlc_vllm_kpool_expand_topk 保留可用（其他调用方 / 单测仍可走），
     但插件主路径不再调用它。以下原始需求内容仅作历史记录与语义参考。 -->

> **状态（2026-10-09）：退出主路径。** 语义已被
> [`09_dlc_kpool_mqa_logits.md`](09_dlc_kpool_mqa_logits.md) 的扩展入口
> `torch.dlc_kpool_select_topk` 完整吸收（见该篇「本次修订」§4）。不要再为
> 「先 topk、再展开」单独维护一条需要插件粘合的路径；如已在算子仓库建 issue，请留言说明并关闭。

# [Operator Request] dlc_kpool_expand_topk（池索引展开为 token 索引 + tail 拼接）

## 需求类型
新增算子

**需求分类：完全新算子**
- 核对结果（2026-09-21）：没有「池 id → token id 展开 + tail 拼接」的算子。相邻算子都不等价：
  - `dlc_vllm_expand_kernel_`：把某个值替换成另一个值（value expansion）；
  - `dlc_vllm_topk_per_row_prefill_` / `_decode_*`：只做 topk 选择，不做 `pool_id * pool_size + offset` 的索引生成，也不拼接不完整 tail。
- 因此本需求保持「新增算子」（P1）。
- 本次调整（2026-09-22）：需求类型与优先级（P1）均不变；覆盖核查后仍判定为全新算子。

## 优先级
P1

## 目标版本与期望时间
GLM-5.3-Flash 适配 Phase 2（性能阶段）。建议在长上下文（≥32K prefill）性能验收前提供，具体里程碑待算子组排期确认。

## 算子名称
`torch.dlc_kpool_expand_topk`

## 背景与功能说明
使用场景：KPool 选池后，要把选中的池 id 展开成 4 个 token id，并按序拼接尾部未完成池的 token，供稀疏 MLA 的 topk 索引使用。
当前问题：native `expand_pool_topk`（`vllm_cl/patch/models/kpool_indexer.py:500`）用 arange + 广播 + `torch.where` 组合，涉及多次中间张量和 boolean mask；且每个 query token 一行，属于高频小算子。
目标功能：单 kernel 完成「池 id → 4 个 token id → 截断到 topk_tokens → 尾部 tail token 追加 → 无效槽填 -1」。
非目标：选池（TopK 本身）；稀疏 MLA 计算（11 号）。

## 期望 API 与函数签名
```python
def kpool_expand_topk(
    pool_indices: torch.Tensor,   # [N, K] int32, 选中的池 id（-1 表示无效）
    pool_size: int,               # 4
    topk_tokens: int,             # 2048
    tail_start: int,              # 当前不完整池的起始 token 位置
    sequence_length: int,         # 该行对应的有效长度
    out: torch.Tensor | None = None,  # [N, topk_tokens + pool_size - 1] int32
) -> torch.Tensor:                # [N, 2051] int32（实现缓冲可对齐到 2176）
```

## 参数与返回值
输入：
- `pool_indices`：`[N, K]`，int32，TopK 选出的池 id，`-1` 为无效；K 应 ≥ `topk_tokens // pool_size`（GLM-5.3-Flash：2048/4 = 512）。
- `pool_size`：默认 4。
- `topk_tokens`：默认 2048。
- `tail_start`：当前不完整池的起始 token 位置（`sequence_length - sequence_length % pool_size`）。
- `sequence_length`：该行对应的有效 token 总数。

返回值：
- `[N, topk_tokens + pool_size - 1]`，int32，即 `[N, 2051]`；无效或越界槽填 `-1`。
- 上层稀疏 MLA tile 会把缓冲对齐到 2176，是否需要 padded 输出请算子组确认。

边界：`tail_start` 之后不足 4 个 token 时全部保留；`sequence_length < pool_size` 时 tail 全部保留。

## 功能语义
```text
select_k = topk_tokens // pool_size                       # 512
tok[n, i*4 + r] = pool_indices[n, i] * 4 + r              # i < select_k, r = 0..3
有效条件：tok < sequence_length
out[n, :topk_tokens] = valid ? tok : -1

tail_len = (sequence_length - tail_start < pool_size and sequence_length > tail_start)
           ? sequence_length - tail_start : 0
out[n, topk_tokens : topk_tokens + tail_len] = arange(tail_start, tail_start + tail_len)
其余位置 = -1
```

broadcasting：逐行独立，`sequence_length` / `tail_start` 按行不同。
in-place / alias：不修改输入；提供 `out` 时原位写。
空 / 零长度：`N == 0` 返回空；`K == 0` 时只输出 tail。
异常：形状不符报错。
确定性：无随机。

## 支持范围与约束
支持的 dtype：输入 / 输出 int32（若上层需要 int64 请明确）。
支持的 shape：`N` 动态；`K ∈ {512, 1024}`；`topk_tokens` 默认 2048（建议支持任意 4 的倍数）；输出宽度 `topk_tokens + pool_size - 1 = 2051`。
数值范围：token id < sequence_length ≤ 1M；`-1` 为无效哨兵。
layout：连续 2D。
device：DLC 单卡。
执行模式：需支持 CUDA graph。
其他限制：输出宽度可由算子组选择 2051 或对齐到 2176；若对齐，必须明确 padding 部分是否也用 `-1`（稀疏 MLA kernel 依赖该约定）。

## 精度要求
Golden / reference：`expand_pool_topk`（`vllm_cl/patch/models/kpool_indexer.py:500`）。
输入分布：随机池 id（含 `-1`、含越界 id）、`sequence_length` 覆盖 0~1M、`tail_start` 覆盖各余数。
样本数量：≥ 200 组。
比较 dtype：int32 逐元素。
atol / rtol：不适用；**要求 bit-exact**。
必须逐元素一致：无效槽严格为 `-1`；tail 段位置与长度与 reference 一致；输出长度必须是 2051（或按约定 2176 且 padding 语义明确）。
NaN / Inf / 极值：不适用。

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
1. 功能：N ∈ {1, 64, 4096}，K=512，覆盖 `sequence_length % 4 ∈ {0,1,2,3}`。
2. 边界：`sequence_length < 4`、`sequence_length == 0`、全部池 id 为 `-1`、池 id 越界。
3. 输出宽度：2051 / 2176 两种约定都测。
4. bit-exact 校验、异常、CUDA graph。
5. 性能：N=4096 下单次耗时；对比 native（目标 ≥ 2×）。

## 验证环境与依赖版本
DLC 硬件：8 卡，设备名 `/dev/cltech0-7`（TP=8，无 PP）
vLLM commit：`f5361fb09f`
vLLM DLC（vllm-cl）commit：`a14fc5b`
PyTorch / torch-dlc：torch 2.5.0（DLC 构建），Python 3.10.16
关键环境变量：`VLLM_USE_DLC_COL_MAJOR_MATMUL=1`
模型：GLM-5.3-Flash-Mix（AWQ 4-bit，`linear_num_heads=64`、`linear_head_dim=128`、`short_conv_kernel_size=4`、`gate_lower_bound=-5.0`）

## 验收标准
- [ ] API、参数校验和错误行为符合约定
- [ ] 规定范围内的功能和精度测试通过
- [ ] 需求中明确列出的特殊运行场景通过
- [ ] 文档、测试和示例已提交
- [ ] 接入 vllm-cl 后，GLM-5.3-Flash 在 DLC 上 prefill/decode 数值与 PyTorch reference 一致

## Reference、日志与附件
Reference 实现：`vllm_cl/patch/models/kpool_indexer.py:500`（`expand_pool_topk`）。
调用点：`native_kpool_indexer_step`（`:533`）。
输出宽度约定：`docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_适配研究.md` §6.2「256 token + ≤3 tail 缓冲对齐 2176」。
相关文档：`docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_适配研究.md` §6.5 算子 `dlc_expand_pool_topk`。
