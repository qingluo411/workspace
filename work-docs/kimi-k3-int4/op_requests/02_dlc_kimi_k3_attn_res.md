<!-- GitHub Issue Form：算子需求字段 -->
<!-- 生成日期：2026-10-28 -->
<!-- 需求分类：新增算子；优先级：P1 -->

# [Operator Request] dlc_kimi_k3_attn_res（K3 Attention Residuals 融合）

## 需求类型

新增算子

## 优先级

P1

理由：K3 每个 decoder layer 在 attention 前和 MLP 前各调用一次 AttnRes，24 层切片前向
共 48 次。当前 native 实现包含 residual bank 写回、两次 RMSNorm 统计、softmax 和
`[T, num_blocks+1, 7168]` 的加权求和，是 MoE 之后最值得融合的非 GEMM 热点。

## 目标版本与期望时间

vLLM DLC `v0.30.1rc0` / Kimi-K3 TP8 性能里程碑；日期待算子组排期确认。

## 算子名称

`torch.dlc_kimi_k3_attn_res`

上游同语义 CUDA operator 名称为 `torch.ops._C.kimi_k3_attn_res`，可参照其参数和数值行为。

## 背景与功能说明

使用场景：

- Kimi-K3 的 Attention Residuals（AttnRes），`attn_res_block_size=12`；
- 完整 93 层模型有 8 个 residual block；当前 24 层切片有 2 个 block；
- 每个 layer 调用两次：
  1. attention 前：更新 prefix，并可能写 residual bank，再做 softmax mixing 和 input RMSNorm；
  2. MLP 前：使用 attention 输出作为 delta，再做一次 mixing 和 post-attention RMSNorm。

当前问题：

- 上游仅在 CUDA 13 / SM100 上提供 native fused kernel，其余路径使用 Triton；
- DLC 不能执行 Triton，当前由纯 PyTorch 组合实现完成；
- native 每个调用需要 bank 写回、平方和、rsqrt、点积、softmax、加权求和和输出 norm，
  产生多个 elementwise/reduction launch，且 full model 每 step 要执行约 186 次。

目标功能：

- 单 kernel 完成 AttnRes 的全部语义，包括 prefix 原位更新、可选 bank 写回、softmax 混合
  和可选输出 RMSNorm；
- 保持与 native 完全相同的 in-place/alias 行为。

非目标：

- 不做 attention/MLP GEMM；
- 不构造 residual bank 或 prefix；
- 不处理 mHC/Sinkhorn；K3 没有 mHC；
- 不包含后续 output gate 或 MoE。

## 期望 API 与函数签名

```python
def dlc_kimi_k3_attn_res(
    prefix: torch.Tensor,                 # [T, H], BF16，in-place 更新
    delta: torch.Tensor | None,            # [T, H], BF16 或 None
    blocks: torch.Tensor,                  # [T, B, H], BF16，in-place 可写
    norm_weight: torch.Tensor,             # [H], BF16
    qk_weight: torch.Tensor,               # [H], BF16
    output_norm_weight: torch.Tensor | None,  # [H], BF16 或 None
    *,
    num_blocks: int,
    block_write_idx: int = -1,
    eps: float = 1e-6,
    output_norm_eps: float = 1e-6,
    out: torch.Tensor | None = None,       # [T, H], BF16
) -> torch.Tensor:
    ...
```

## 参数与返回值

输入 / 属性：

- `prefix`：当前 residual prefix，`[T, H]`；若 `delta` 非空，先原地加 delta 并 cast 回 BF16。
- `delta`：attention/MLP 新增 residual；首层或 block 写层可为 `None`。
- `blocks`：residual bank，`[T, B, H]`；前 `num_blocks` 个 bank 参与 softmax。
- `norm_weight`：AttnRes 的 RMSNorm 权重。
- `qk_weight`：`[H]` 的 projection weight（上游为 `Linear(H,1)` squeeze 后）。
- `output_norm_weight`：可选，对混合结果再做一次 RMSNorm。
- `num_blocks`：`0..8`，切片内动态变化。
- `block_write_idx`：`-1` 表示不写 bank；`0..B-1` 表示写对应 bank。

返回值：

- `[T, H]` BF16；若提供 `out`，返回同一个 out tensor。

副作用：

- `delta` 非空时，`prefix` 被原位更新为 BF16 的 `prefix + delta`；
- `block_write_idx >= 0` 时，`blocks[:, block_write_idx, :]` 被写成更新后的 prefix；
- 其余输入不修改。

## 功能语义

```text
updated = prefix.float()
if delta is not None:
    updated = updated + delta.float()
    updated = updated.to(BF16).float()
    prefix.copy_(updated.to(BF16))

if block_write_idx >= 0:
    blocks[:, block_write_idx, :].copy_(updated.to(BF16))

if num_blocks == 0:
    mixed = updated
else:
    values = cat(blocks[:, :num_blocks, :].float(), updated[:, None, :], dim=1)
    rstd = rsqrt(mean(values.square(), dim=-1) + eps)
    score_weight = norm_weight.float() * qk_weight.float()
    logits = sum(values * score_weight, dim=-1) * rstd
    probs = softmax(logits, dim=-1)
    mixed = sum(probs[..., None] * values, dim=1)

if output_norm_weight is not None:
    out_rstd = rsqrt(mean(mixed.square(), dim=-1, keepdim=True) + output_norm_eps)
    mixed = mixed * out_rstd * output_norm_weight.float()

out = mixed.to(BF16)
```

边界：

- `T == 0`：返回 `[0, H]`，不修改输入；
- `num_blocks == 0`：softmax 退化为单路 prefix；
- `num_blocks > blocks.shape[1]`：报错；
- `block_write_idx` 越界：报错；
- `delta=None` 且 `block_write_idx=-1`：允许，用于纯 mixing。

确定性：无随机；归约 FP32。

## 支持范围与约束

首期精确范围：

- `H = 7168`；
- `B ∈ [1, 8]`，`num_blocks ∈ [0, B]`；
- `block_write_idx ∈ [-1, B-1]`；
- `T` 动态，至少覆盖 1、8、64、256、4096；
- `prefix/delta/blocks` BF16，`norm_weight/qk_weight/output_norm_weight` BF16；
- 所有 tensor 最后一维连续、stride=1；`out` 为独立 BF16 连续 tensor；
- 单卡 DLC；
- 支持 eager，目标支持 graph capture。
- `num_blocks` 和 `block_write_idx` 在 graph 捕获期间必须按 layer/模型状态固定；调用方应使用
  静态标量或预编译 variant，不能从 device tensor 读取这两个值。

非目标范围：

- FP16/FP32 输入首期可不要求；
- 不要求 `H != 7168` 的通用模板，但建议保留 H 为 128 倍数的可扩展路径；
- 不要求非连续 batch stride；调用方可先 contiguous。

## 精度要求

Golden / reference：

- 以本 issue 末尾的自包含 PyTorch reference 为准；
- 上游 Triton/CUDA 语义也由该 reference 完整表达。

输入分布：

- prefix/delta/blocks 为 `N(0,1)` 和低方差 `U(-0.1,0.1)`；
- weights 为 `U(0.5,1.5)`；
- 覆盖 `delta=None`、`output_norm_weight=None`、全 0 blocks 和全 0 prefix。

比较方式：

- BF16 输出统一转 FP32 与 native FP32 reference 比较；
- `atol=2e-2`、`rtol=2e-2`；
- 副作用必须逐元素比较：
  - `prefix` 更新后的 BF16 值与 reference bit-exact；
  - `blocks[:, block_write_idx]` 与 reference bit-exact。

累加精度：

- square、mean、rsqrt、点积、softmax 和 weighted sum 全部 FP32；
- 最终只 cast 一次 BF16。

## 测试范围

- [x] 基本功能与 reference 对齐
- [ ] 多 dtype 精度（首期只 BF16）
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [x] 非 contiguous、特殊 stride 或 layout
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式（按 token 本地执行，无跨卡通信）
- [x] 长时间稳定性
- [x] 性能基准与性能回归

具体测试用例、shape、数据范围和预期结果：

1. 基础：`T ∈ {1,8,64,256,4096}`，`H=7168`，`B=8`，`num_blocks ∈ {0,1,7,8}`。
2. block write：`block_write_idx ∈ {-1,0,7}`，`delta` 有/无。
3. 输出 norm：`output_norm_weight` 有/无。
4. 副作用：检查 prefix 和 blocks 的原位写回；确认 `delta=None` 时 prefix 不被改写。
5. 异常：`num_blocks > B`、`block_write_idx >= B`、shape 不匹配。
6. 性能：`T=1` 单次 ≤ 50 us；`T=4096` 单次 ≤ 10 ms；24 层前向 48 次调用的总成本
   相对 native 至少降低 5×。

## 验证环境与依赖版本

```text
DLC 硬件：8 x TYD，TP8；driver v26.0.1 / Api 26
vLLM：85fd014d5ff4
vLLM DLC（vllm-cl）：ab819da9a0f4 + K3 适配工作区改动
PyTorch / torch-dlc：2.5.0，9049914ec94a
DLC_Custom_Kernel：55b9aef7e0cb
模型：Kimi-K3-INT4-Slice8-TP8 / full Kimi-K3-INT4
```

## 验收标准

- [ ] API、参数校验和错误行为符合约定。
- [ ] `num_blocks=0/1/8` 与 native reference 一致，softmax 数值稳定。
- [ ] prefix 更新和 residual bank 写回与 reference bit-exact。
- [ ] `atol=2e-2, rtol=2e-2` 通过；无 NaN/Inf 传播异常。
- [ ] `T=1` ≤ 50 us、`T=4096` ≤ 10 ms（首期目标）。
- [ ] 可被 DLC Graph capture，不在 forward 内做 host D2H。
- [ ] 提供 caller `out` 时，输出地址在 capture/replay 期间保持稳定；prefix/blocks 的原位更新
      使用同一组静态 buffer。
- [ ] 文档、测试和示例已提交。

## Reference、日志与附件

### Self-contained PyTorch reference

```python
import torch


def attn_res_reference(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    if prefix.dim() != 2:
        raise ValueError("prefix must be 2D")
    tokens, hidden = prefix.shape
    if blocks.dim() != 3 or blocks.shape[0] != tokens or blocks.shape[2] != hidden:
        raise ValueError("blocks must be [tokens, banks, hidden]")
    if delta is not None and delta.shape != prefix.shape:
        raise ValueError("delta must match prefix")
    if num_blocks < 0 or num_blocks > blocks.shape[1]:
        raise ValueError("num_blocks is outside the residual bank")
    if block_write_idx < -1 or block_write_idx >= blocks.shape[1]:
        raise ValueError("block_write_idx is outside the residual bank")

    updated_prefix = prefix.float()
    if delta is not None:
        updated_prefix = (updated_prefix + delta.float()).to(prefix.dtype).float()
        prefix.copy_(updated_prefix.to(prefix.dtype))
    if block_write_idx >= 0:
        blocks[:, block_write_idx, :].copy_(updated_prefix.to(blocks.dtype))

    if num_blocks == 0:
        mixed = updated_prefix
    else:
        values = torch.cat(
            (blocks[:, :num_blocks, :].float(), updated_prefix.unsqueeze(1)),
            dim=1,
        )
        score_weight = norm_weight.float() * qk_weight.float()
        reciprocal_std = torch.rsqrt(values.square().mean(dim=-1) + float(eps))
        logits = (values * score_weight.view(1, 1, hidden)).sum(dim=-1)
        logits = logits * reciprocal_std
        probabilities = torch.softmax(logits, dim=-1)
        mixed = (probabilities.unsqueeze(-1) * values).sum(dim=1)

    if output_norm_weight is not None:
        output_std = torch.rsqrt(
            mixed.square().mean(dim=-1, keepdim=True) + float(output_norm_eps)
        )
        mixed = mixed * output_std * output_norm_weight.float().view(1, hidden)
    return mixed.to(prefix.dtype)
```

模型调用形态：

```python
hidden_states = attn_res_reference(
    prefix_sum,
    hidden_states,
    residual,
    self_attention_res_norm.weight,
    self_attention_res_proj.weight.squeeze(0),
    input_layernorm.weight,
    num_blocks=self.prev_valid_blocks,
    block_write_idx=self.block_write_idx if self.is_block_write_layer else -1,
    eps=self.self_attention_res_norm.variance_epsilon,
    output_norm_eps=self.input_layernorm.variance_epsilon,
)
```
