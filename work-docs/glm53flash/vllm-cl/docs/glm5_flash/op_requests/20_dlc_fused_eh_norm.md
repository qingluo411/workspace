<!-- 按算子需求 Issue Form 字段编写，可直接粘贴到 GitHub Issue -->
<!-- 生成日期：2026-11-13 -->
<!-- 需求分类：完全新算子；优先级：P1；来源：GLM-5.3-Flash MTP -->

# [Operator Request] dlc_fused_eh_norm_bf16（MTP EH norm + concat 融合）

## 需求类型

新增算子

**需求分类：完全新算子**

- 现有 DLC RMSNorm 能力是面向单输入张量的 `custom_FusedRMSNorm_f32` /
  `custom_FusedAddRMSNorm_f32`，要求 FP32 输入，且不包含“position mask + 两路
  RMSNorm + concat”这一组合语义。
- 当前 GLM-5.3-Flash MTP 层使用 Python fallback `fused_eh_norm_dlc`，不是 DLC
  fused kernel。
- 本需求只新增一个 MTP 专用融合算子，不把 `embedding`、`eh_proj`、`shared_head`
  或 `lm_head` 一并塞进该算子。

## 优先级

P1

理由：该算子位于 MTP draft 每个 step 的入口，当前 fallback 包含 mask、两次
RMSNorm（含 FP32 转换/平方/均值/rsqrt/乘法）和 cat，多 kernel 且中间张量往返；
当前实测 `T=4 / H=4096` fallback 约 **0.58 ms**。它不是发布阻断项；如果最终
profile 显示它在 MTP step 中占比低于约 2%，可降为 P2。

## 目标版本与期望时间

vLLM DLC `v0.30.1rc0` / GLM-5.3-Flash MTP 性能优化里程碑；具体交付时间待算子组
排期确认。建议在 MTP 正式发布前完成，若性能收益不足可先保留 fallback。

## 算子名称

建议公开名：`torch.dlc_fused_eh_norm_bf16`

建议 kernel 名：`custom_fused_eh_norm_bf16`

模型侧接入 alias：`fused_eh_norm_dlc`（保持与上游 `fused_eh_norm` 相同的调用位置）

## 背景与功能说明

使用场景：

- GLM-5.3-Flash-Mix 第 45 层 MTP layer；
- 调用点 `Glm5NextMultiTokenPredictorLayer.forward()`，每个 MTP draft step 一次；
- 输入是 `inputs_embeds`、`previous_hidden_states`、两路 RMSNorm weight 和 `positions`；
- 输出 `[T, 2H]`，直接作为 `eh_proj` 的输入。

调用点等价代码：

```python
class Glm5NextMultiTokenPredictorLayer(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        eh_input = fused_eh_norm(
            positions,
            inputs_embeds,
            previous_hidden_states,
            self.enorm.weight,
            self.hnorm.weight,
            self.enorm.variance_epsilon,
        )
        hidden_states = self.eh_proj(eh_input)
        ...
```

当前 Python fallback 的完整实现：

```python
def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp32 = x.float()
    rrms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * rrms * weight.float()).to(x.dtype)


def fused_eh_norm_dlc(
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor,
    previous_hidden: torch.Tensor,
    enorm_w: torch.Tensor,
    hnorm_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """DLC fallback for the Triton MTP embed/norm/concat fusion."""
    if inputs_embeds.shape != previous_hidden.shape:
        raise ValueError("MTP embeds and previous hidden states must match")
    if positions.shape != (inputs_embeds.shape[0],):
        raise ValueError("MTP positions must be [tokens]")

    mask = (positions != 0).unsqueeze(-1)
    embeds = inputs_embeds * mask.to(inputs_embeds.dtype)
    return torch.cat(
        [
            _rms_norm(embeds, enorm_w, eps),
            _rms_norm(previous_hidden, hnorm_w, eps),
        ],
        dim=-1,
    )
```

其中 `rms_norm` 每次都先转 FP32，做 `mean(x*x)`、`rsqrt`、乘 weight，再 cast 回
BF16；`mask`、两次 norm 和 `cat` 会形成多次 kernel launch 和中间张量。

目标功能：

- 单 kernel 完成 `position != 0` mask、两路 RMSNorm 和 concat；
- 输出连续 `[T, 2H]` BF16，可直接喂给 `eh_proj`；
- 支持 DLC eager 和 Graph capture/replay；
- 数学语义、eps、FP32 累加和 BF16 输出与当前 fallback 一致。

非目标：

- 不在本算子中融合 embedding lookup；
- 不融合 `eh_proj` GEMM；
- 不融合 `shared_head.norm`、`lm_head` 或 local argmax；
- 不改变 MTP MoE、KDA、KPool 路径；
- 不做 MTP 权重量化。

预期收益：

- 将当前 mask + 2×RMSNorm + cat 的多次 launch/中间张量收敛为一次 kernel；
- 降低 MTP draft step 的固定开销，尤其是 `T` 较小的 decode 场景；
- 不改变 MTP greedy 输出和 acceptance 语义。

## 期望 API 与函数签名

```python
def dlc_fused_eh_norm_bf16(
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor,
    previous_hidden: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """返回 cat([enorm(masked embeds), hnorm(previous_hidden)]) -> [T, 2H]。"""
```

首期不提供 `out` 参数；输出由算子分配并返回。若算子组希望额外支持 `out`，必须
保证 graph-safe、地址稳定，并且行为与不传 `out` 完全一致。

## 参数与返回值

输入：

- `positions`：`[T]`，当前位置，首期要求 INT64；使用 `positions != 0` 作为
  embed mask。
- `inputs_embeds`：`[T, H]`，BF16；允许第 0 维 stride 非连续，但最后一维必须
  `stride(-1) == 1`。
- `previous_hidden`：`[T, H]`，BF16；布局要求同 `inputs_embeds`。
- `enorm_weight`：`[H]`，BF16，第一路 RMSNorm weight。
- `hnorm_weight`：`[H]`，BF16，第二路 RMSNorm weight。
- `eps`：float，GLM-5.3-Flash 使用 `1e-5`。

返回值：

- `[T, 2H]`，BF16，连续布局；前半为 `enorm(masked embeds)`，后半为
  `hnorm(previous_hidden)`。

行为约束：

- 不修改任何输入；
- 输入与输出不 alias；
- `T == 0` 返回形状正确的空张量；
- 不执行 host 侧 `.cpu()`、`.item()`、`.tolist()` 或 numpy 转换；
- 相同输入必须确定性输出。

## 功能语义

按 token 逐行计算，等价 PyTorch reference 如下：

```python
def rms_norm(x, weight, eps):
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)

masked = inputs_embeds * (positions != 0).unsqueeze(-1).to(inputs_embeds.dtype)
out = torch.cat([
    rms_norm(masked, enorm_weight, eps),
    rms_norm(previous_hidden, hnorm_weight, eps),
], dim=-1)
```

关键语义：

- `positions == 0` 的行，`masked embeds` 必须为全 0；因此输出前半必须为全 0，
  后半仍是正常的 `hnorm(previous_hidden)`；
- 两路 RMSNorm 都用 FP32 做平方和/均值/rsqrt，并乘 FP32 weight；
- 每路结果先 cast 到 BF16，再写入输出的对应半区，保持与 fallback 的舍入语义一致；
- 输出 `[T, 2H]` 行内连续，行间步长为 `2H`；
- 不使用随机数；不做 softmax、scale、bias 或额外 activation。

## 支持范围与约束

首期精确范围：

- `H = 4096`（GLM-5.3 MTP hidden size）；
- `T` 动态，覆盖 MTP 实际形状 `T = 2, 4, 8, 16, 32, 40, 64, 128`；
- `positions` INT64；
- `inputs_embeds`、`previous_hidden`、两路 weight、输出均为 BF16；
- 内部 RMSNorm 归约和 weight 乘法使用 FP32；
- DLC 单卡执行，不包含 TP collective；
- 支持 eager 和 DLC Graph capture/replay；
- 输入最后一维 stride=1；第 0 维 stride 可非连续（与上游 Triton `fused_eh_norm` 一致）；
- 不得依赖数据相关 host 分配；如 kernel 需要 scratch，使用持久 workspace 或
  调用方传入的预分配 buffer。

通用范围建议：

- `H` 为 128 的倍数；至少支持 1024/2048/4096 以便单元测试和后续复用；
- `T ∈ [0, max_num_batched_tokens]` 动态；
- `positions` 支持 INT64；如实现成本低，可额外支持 INT32；
- 不支持 FP16/FP32 输入（可另开需求）。

明确不支持：

- 非连续最后一维（`stride(-1) != 1`）；
- `positions` 长度与 `T` 不一致；
- weight 长度与 `H` 不一致；
- 多卡 collective、跨 rank 通信；
- 带 bias/scale/activation 的变体；
- 输入或输出 dtype 的隐式转换。

## 精度要求

Golden / reference：

- 以本文“Reference 实现”中的 `_reference` 为准；
- 同时对照上文“当前 Python fallback”的 `fused_eh_norm_dlc`。

输入分布与样本：

- `embeds`、`previous`、两路 weight 使用 `N(0, 1)` BF16；
- `positions` 覆盖全 0、全非 0、混合位置，且至少包含多个 position 0；
- 额外覆盖 embeds 全 0、weight 全 0、weight 全 1、极大值输入；
- `T` 至少覆盖 0、1、2、4、8、16、32、40、64、128；
- `H` 至少覆盖 1024、2048、4096；
- 至少 200 组固定 seed 随机输入。

比较方式：

- BF16 输出与 FP32 reference 比较；
- 建议 `atol=2e-2, rtol=2e-2`，与当前 DLC `fused_eh_norm` 回归保持一致；
- position 0 行的输出前半必须逐元素严格为 0；
- 不得出现 NaN/Inf；
- 不要求与 fallback 逐 bit 一致，但差异不得改变 MTP greedy token 序列。

端到端精度：

- 接入 MTP 后，eager 固定 greedy 请求至少 50/50 与无 MTP baseline 逐 token 一致；
- MTP + `FULL_DECODE_ONLY` graph 单请求至少 20/20；
- 长上下文 marker 检索与当前 MTP graph 基线一致。

## 测试范围

- [x] 基本功能与 reference 对齐
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [x] 非 contiguous 第 0 维 stride / 特殊 layout
- [x] 非法参数和异常处理
- [x] 单卡
- [x] 长时间稳定性
- [x] 性能基准与性能回归
- [ ] 多 dtype 精度（首期仅 BF16）
- [ ] 多卡/分布式（本算子无 collective）
- [x] DLC Graph capture/replay

具体测试用例、shape、数据范围和预期结果：

1. 功能：`T ∈ {0,1,2,4,8,16,32,40,64,128}`，`H ∈ {1024,2048,4096}`；
   position 0、全 0、混合位置各覆盖一次。
2. 精度：200 组固定 seed 随机输入，BF16 vs FP32 reference；输出前半 position 0
   严格为 0；无 NaN/Inf。
3. Layout：`inputs_embeds`/`previous_hidden` 的第 0 维 stride 分别使用 `H`、
   `H+1`、`2H` 等 view；最后一维 stride 必须为 1。
4. 异常：T 不一致、H 不一致、weight 长度错误、最后一维非连续、dtype 错误均报错。
5. Graph：至少 1000 次 capture/replay，输出与 eager 在容差内一致，无地址漂移。
6. 性能：对比 fallback，报告 `T=4/8/16/40, H=4096` 的 kernel 时间和额外内存；
   目标 `T=4` 端到端 ≤0.2 ms，且相对 fallback ≥3x；`T=40` 不得回退。
7. 集成：MTP eager 50/50，MTP graph 20/20，长上下文 marker 通过。

## 验证环境与依赖版本

DLC 硬件：8 × TYD，TP=8

模型：GLM-5.3-Flash-Mix，MTP layer 45

vLLM commit：`3eb6a0c681`（GDN spec padding 修复后的版本）

vLLM DLC commit：`1940b2e`（MTP graph 修复后的版本）

PyTorch / torch-dlc：torch 2.5.0（DLC 构建）

Runtime / driver：`Driver Version_v26.0.1`（发布验证组合；以启动日志为准）

Custom Kernel / LLVM：Custom Kernel `59e9f0788`、LLVM RTABI 3.2.0

关键环境变量：

```bash
export VLLM_PLUGINS=dlc,dlc_patches
export DLC_SKIP_VERSION_CHECK=0
export DLC_SYN_GRAPH_CACHE=1
export VLLM_USE_DLC_COL_MAJOR_MATMUL=1
export PYTORCH_DLC_ALLOC_CONF=expandable_segments:True
```

## 验收标准

- [ ] API、参数校验和错误行为符合约定
- [ ] 功能与 FP32 reference 在指定容差内一致
- [ ] position 0 mask、T=0、非连续行 stride 和异常输入测试通过
- [ ] eager 与 DLC Graph capture/replay 输出一致
- [ ] 性能达到约定目标，且不出现额外 host 同步或数据相关分配
- [ ] 接入 MTP 后 eager 50/50、graph 20/20、长上下文 marker 通过
- [ ] 文档、测试、reference 和性能报告已提交

## Reference、日志与附件

### 上游 Triton 实现（语义来源）

以下代码是当前 GPU 路径的语义参考；DLC 没有 Triton driver，不能直接运行，但其
计算顺序、stride 和输出布局应作为 DLC kernel 的对照：

```python
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rms_norm(x, w, eps, HIDDEN_SIZE: tl.constexpr):
    x = x.to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    rrms = tl.rsqrt(mean_sq + eps)
    w = w.to(tl.float32)
    return (x * rrms) * w


@triton.jit
def _fused_eh_norm_kernel(
    pos_ptr,
    embeds_ptr,
    embeds_stride,
    prev_ptr,
    prev_stride,
    enorm_w_ptr,
    hnorm_w_ptr,
    eps,
    out_ptr,
    out_stride,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """MTP input fusion: zero embeds at position 0, RMSNorm(embeds) with enorm
    and RMSNorm(prev_hidden) with hnorm, written side-by-side into ``out``
    ([N, 2H]) ready for the eh_proj GEMM. Replaces where + 2x RMSNorm + cat."""
    tok = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    mask = off < H

    pos = tl.load(pos_ptr + tok)
    e = tl.load(embeds_ptr + tok * embeds_stride + off, mask=mask, other=0.0)
    e = tl.where(pos == 0, 0.0, e.to(tl.float32))
    ew = tl.load(enorm_w_ptr + off, mask=mask)
    e_normed = _rms_norm(e, ew, eps, H)
    tl.store(out_ptr + tok * out_stride + off, e_normed, mask=mask)

    p = tl.load(prev_ptr + tok * prev_stride + off, mask=mask, other=0.0)
    hw = tl.load(hnorm_w_ptr + off, mask=mask)
    p_normed = _rms_norm(p, hw, eps, H)
    tl.store(out_ptr + tok * out_stride + H + off, p_normed, mask=mask)


def fused_eh_norm(
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor,
    previous_hidden: torch.Tensor,
    enorm_w: torch.Tensor,
    hnorm_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Returns cat([enorm(masked embeds), hnorm(prev_hidden)]) -> [N, 2H]."""
    n, h = inputs_embeds.shape
    out = torch.empty(n, 2 * h, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    _fused_eh_norm_kernel[(n,)](
        positions,
        inputs_embeds,
        inputs_embeds.stride(0),
        previous_hidden,
        previous_hidden.stride(0),
        enorm_w,
        hnorm_w,
        eps,
        out,
        out.stride(0),
        h,
        triton.next_power_of_2(h),
    )
    return out
```

### FP32 reference（精度对照）

```python
def _reference(positions, embeds, previous, enorm_w, hnorm_w, eps):
    def norm(x, weight):
        dtype = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
                * weight.float()).to(dtype)

    masked = embeds * (positions != 0).unsqueeze(-1).to(embeds.dtype)
    return torch.cat([norm(masked, enorm_w), norm(previous, hnorm_w)], dim=-1)
```

最小自包含校验示例：

```python
torch.manual_seed(7)
tokens, hidden = 5, 32
positions = torch.tensor([0, 1, 2, 0, 4], dtype=torch.int64)
embeds = torch.randn(tokens, hidden, dtype=torch.bfloat16)
previous = torch.randn(tokens, hidden, dtype=torch.bfloat16)
enorm = torch.randn(hidden, dtype=torch.bfloat16)
hnorm = torch.randn(hidden, dtype=torch.bfloat16)

expected = _reference(positions, embeds, previous, enorm, hnorm, 1e-6)
# got = dlc_fused_eh_norm_bf16(positions, embeds, previous, enorm, hnorm, 1e-6)
# torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)
```

### 已有验证数据

- MTP 层为第 45 层，hidden=4096，MTP 层在当前 checkpoint 中保持 BF16；
- 当前 fallback `T=4 / H=4096` 实测约 **0.58 ms**；
- 现有 MTP eager 固定 greedy 请求 **50/50** 与无 MTP baseline 逐 token 一致；
- 现有 MTP + `FULL_DECODE_ONLY` graph 单请求 **20/20**；
- 长上下文 marker 检索在 17,025 / 34,025 / 64,625 token 三个档位均通过。
