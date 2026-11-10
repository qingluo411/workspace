<!-- GitHub Issue Form：算子需求字段 -->
<!-- 生成日期：2026-10-28 -->
<!-- 需求分类：新增算子；优先级：P1 -->

# [Operator Request] dlc_kimi_k3_mla_decode_epilogue_bf16（K3 MLA decode q concat + RoPE + cache insert）

## 需求类型

新增算子

## 优先级

P1

理由：K3 每个 MLA 层的 decode 在 `forward_mqa` 之前都需要一次
`mqa_q=[ql_nope | rotated q_pe]` 拼接和 `[kv_c | rotated k_pe]` cache 写入。当前 native
实现由 GPT-J RoPE、concat、slot 索引和 `index_copy_` 串联，decode 每 step 触发多次 launch；
全模型 24 个 MLA 层累计后是除 MoE 外的重要固定开销。

## 目标版本与期望时间

vLLM DLC `v0.30.1rc0` / Kimi-K3 decode 性能里程碑；日期待算子组排期确认。

## 算子名称

`torch.dlc_kimi_k3_mla_decode_epilogue_bf16`

对应上游 CUDA fused op：
`torch.ops._C.fused_kimi_k3_mla_decode_q_concat_kv_cache_insert`。

## 背景与功能说明

使用场景：

- Kimi-K3 dense gated MLA decode；
- `ql_nope` 已由 absorbed query BMM 产生，形状 `[B,H,512]`；
- `q_pe` 形状 `[B,H,64]`；`k_pe` 为 `[B,64]` 或 `[B,1,64]`；
- `kv_c_normed` 形状 `[B,512]`；
- KV cache 为 BF16 `[num_blocks, page_size, 576]`。

当前问题：

- 上游只有 CUDA 融合 epilogue，DLC 无实现；
- native 需要独立执行：
  1. q_pe RoPE；
  2. k_pe RoPE；
  3. `mqa_q=cat(ql_nope,q_pe)`；
  4. latent concat 和 paged cache 写入；
- decode 是每 token 固定执行，kernel 数量比单次 FLOPs 更敏感。

目标功能：

- 单 kernel 完成 decode epilogue；
- 返回 `mqa_q=[ql_nope | rotated_q_pe]`；
- 原地写入 `kv_cache=[kv_c_normed | rotated_k_pe]`。

非目标：

- 不做 `forward_mqa` attention；
- 不做 q/kv projection、RMSNorm、DCP gather；
- 首期不支持 fp8 / fp8_ds_mla；
- 不包含 output gate。

## 期望 API 与函数签名

```python
def dlc_kimi_k3_mla_decode_epilogue_bf16(
    ql_nope: torch.Tensor,           # [B, H, 512], BF16
    q_pe: torch.Tensor,              # [B, H, 64], BF16
    kv_c_normed: torch.Tensor,       # [B, 512], BF16
    k_pe: torch.Tensor,              # [B, 1, 64] 或 [B, 64], BF16
    kv_cache: torch.Tensor,          # [num_blocks, page_size, 576], BF16
    slot_mapping: torch.Tensor,      # [B], int32/int64，-1 跳过
    positions: torch.Tensor | None = None,      # [B], int64
    cos_sin_cache: torch.Tensor | None = None,  # [max_pos, 64], FP32/BF16
    *,
    out: torch.Tensor | None = None, # [B, H, 576], BF16
) -> torch.Tensor:
    ...
```

## 参数与返回值

输入 / 属性：

- `ql_nope`：absorbed decode query latent，最后一维 512。
- `q_pe`：query RoPE 部分，最后一维 64。
- `kv_c_normed`：decode token 的 latent key 512 维。
- `k_pe`：单 head key RoPE 部分。
- `kv_cache`：paged latent cache，entry=576。
- `slot_mapping`：每个 decode token 的扁平 cache slot；`-1` 表示 pad。
- `positions/cos_sin_cache`：GPT-J RoPE；二者同时为空时走 NoPE。

返回值：

- `[B,H,576]` 的 `mqa_q`；若提供 `out`，原位写入并返回该 tensor。

副作用：

- 仅修改 `kv_cache` 的命中 slot；不修改 q/k 输入。

## 功能语义

```text
if positions is not None and cos_sin_cache is not None:
    q_pe_rot = gpt_j_rope(q_pe, positions, cos_sin_cache)
    k_pe_rot = gpt_j_rope(k_pe.reshape(B,1,64), positions, cos_sin_cache).squeeze(1)
else:
    q_pe_rot = q_pe
    k_pe_rot = k_pe.reshape(B,64)

mqa_q = cat((ql_nope, q_pe_rot), dim=-1)  # [B,H,576]
latent = cat((kv_c_normed, k_pe_rot), dim=-1)  # [B,576]

for b in range(B):
    if slot_mapping[b] >= 0:
        kv_cache_flat[slot_mapping[b]] = latent[b]
```

GPT-J RoPE 规则与 #03 完全一致。中间计算 FP32，输出一次 cast BF16。

边界：

- `B == 0`：返回 `[0,H,576]`；
- `slot_mapping=-1`：本 token 只计算 mqa_q，不写 cache；
- `positions` 越界、q/k shape 不匹配：报错；
- `H` 可为 1 或生产值 12。

确定性：无随机；同输入 bit 级确定性（同一 reduction 顺序）。

## 支持范围与约束

首期精确范围：

- `ql_nope` 512、`q_pe/k_pe` 64、cache entry 576；
- `H` 动态；TP8 生产值 12；
- `B` 动态，至少覆盖 1、8、32、64、256；
- BF16 q/k/cache；RoPE cache FP32（建议接受 BF16）；
- slot mapping int32/int64；
- 单卡 DLC；
- eager 必须通过，目标 graph capture；
- 首期不支持 fp8/ds_mla。
- `ql_nope` 可以是 bmm 输出经 transpose 得到的 strided view；kernel 必须直接读取，
  不能要求调用方先 contiguous。

layout：

- ql_nope/q_pe/kv_c 的最后一维连续；
- k_pe 允许 `[B,1,64]` 或 `[B,64]`；
- ql_nope 可有非连续 batch stride，但最后一维 stride=1；
- `out` 与 `mqa_q` 均为连续 BF16。

## 精度要求

Golden / reference：

- 以本 issue 末尾的自包含 PyTorch reference 为准。

输入分布：

- q/kv 为 `N(0,1)`；positions 覆盖 0、1、边界值和随机值；
- slot mapping 覆盖连续、乱序、含 -1；
- 至少 200 组。

比较方式：

- `mqa_q` 转 FP32：`atol=1e-2, rtol=1e-2`；
- cache 命中位置转 FP32：`atol=1e-2, rtol=1e-2`；
- 未命中 slot 保持 bit-exact；
- `slot_mapping=-1` 不改变 cache。

中间精度：

- RoPE 的 cos/sin 读取、乘加为 FP32；
- concat 不改数值；
- 最终 cast BF16 一次。

## 测试范围

- [x] 基本功能与 reference 对齐
- [ ] 多 dtype 精度（首期 BF16）
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [x] 非 contiguous、特殊 stride 或 layout
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式（每 rank 独立）
- [x] 长时间稳定性
- [x] 性能基准与性能回归

具体测试用例、shape、数据范围和预期结果：

1. `B ∈ {0,1,8,32,64,256}`，`H ∈ {1,12}`，576/512/64 实际维度。
2. RoPE / NoPE 两路；边界 positions。
3. slot mapping 连续/乱序/-1；cache 未命中 bit-exact。
4. 非连续 ql_nope batch view。
5. 性能：`B=1` 单次目标 ≤ 100 us；`B=256` ≤ 1 ms；24 层 decode 的 epilogue 总成本
   相对 native 至少降低 3×。

## 验证环境与依赖版本

```text
DLC 硬件：8 x TYD，TP8；driver v26.0.1 / Api 26
vLLM：85fd014d5ff4
vLLM DLC（vllm-cl）：ab819da9a0f4 + K3 适配工作区改动
PyTorch / torch-dlc：2.5.0，9049914ec94a
DLC_Custom_Kernel：55b9aef7e0cb
模型：Kimi-K3-INT4-Slice8-TP8 / full K3
```

## 验收标准

- [ ] API、参数校验和错误行为符合约定。
- [ ] mqa_q 与 cache 写入和 native reference 对齐。
- [ ] `slot_mapping=-1` / 未命中 slot 保持原值。
- [ ] BF16 输出满足 `atol=1e-2, rtol=1e-2`。
- [ ] `B=1` ≤ 100 us、`B=256` ≤ 1 ms（首期目标）。
- [ ] eager 无 host D2H；目标支持 graph capture。
- [ ] 使用 caller `out` 时，`mqa_q` 直接写入静态 buffer，forward_mqa 不再发生 q concat/copy。
- [ ] 接入 K3 graph 验收时，确认本算子是否仍位于 `@eager_break_during_capture` 段，并记录
      breakable-graph 与完整 graph 的实际差异。
- [ ] 文档、测试和示例已提交。

## Reference、日志与附件

### Self-contained PyTorch reference

```python
import torch


def apply_rope_gptj(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    cos_sin = cos_sin_cache.index_select(0, positions.to(torch.int64))
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(1).to(torch.float32)
    sin = sin.unsqueeze(1).to(torch.float32)
    x_fp32 = x.to(torch.float32)
    x_even = x_fp32[..., ::2]
    x_odd = x_fp32[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_odd * cos + x_even * sin
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(x.dtype)


def insert_latent_cache(
    kv_cache: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    latent = torch.cat((kv_c_normed, k_pe.reshape(k_pe.shape[0], -1)), dim=-1)
    flat_cache = kv_cache.view(-1, kv_cache.shape[-1])
    slots = slot_mapping.reshape(-1).to(torch.int64)
    valid = slots >= 0
    flat_cache.index_copy_(0, slots[valid], latent[valid])


def mla_decode_epilogue_reference(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    if positions is not None and cos_sin_cache is not None and k_pe.shape[0]:
        q_pe = apply_rope_gptj(q_pe, positions, cos_sin_cache)
        k_pe = apply_rope_gptj(
            k_pe.unsqueeze(1), positions, cos_sin_cache
        ).squeeze(1)

    insert_latent_cache(kv_cache, kv_c_normed, k_pe, slot_mapping)
    return torch.cat((ql_nope, q_pe), dim=-1)
```

调用形态：

```python
mqa_q = mla_decode_epilogue_reference(
    ql_nope,
    q_pe,
    kv_c_normed,
    k_pe,
    kv_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
)
# mqa_q 直接传给 paged MLA decode attention。
```
