<!-- GitHub Issue Form：算子需求字段 -->
<!-- 生成日期：2026-10-28 -->
<!-- 需求分类：新增算子；优先级：P1 -->

# [Operator Request] dlc_kimi_k3_mla_prefill_epilogue_bf16（K3 MLA prefill RoPE + cache insert）

## 需求类型

新增算子

## 优先级

P1

理由：K3 的 24 层切片有 6 个 MLA 层，完整模型有 24 个 MLA 层。当前 native prefill
epilogue 需要 q RoPE、k_pe RoPE、full-K concat、latent cache 写入和 slot 索引，多个
elementwise/concat kernel 串联；长 prefill 时 HBM 往返和 launch 次数会线性放大。

## 目标版本与期望时间

vLLM DLC `v0.30.1rc0` / Kimi-K3 TP8 功能与性能里程碑；日期待算子组排期确认。

## 算子名称

`torch.dlc_kimi_k3_mla_prefill_epilogue_bf16`

对应上游 CUDA fused op：
`torch.ops._C.fused_kimi_k3_mla_key_concat_kv_cache_insert`。

## 背景与功能说明

使用场景：

- Kimi-K3 dense gated MLA，qk_nope=128、qk_rope=64、v=128、kv_lora=512；
- TP8 每卡 MLA local heads=12；
- KV cache 为 BF16 `[num_blocks, page_size, 576]`，576=512 latent + 64 RoPE；
- prefill 每个 chunk 在 K/V projection 后调用一次。

当前问题：

- DLC 没有 K3 CUDA fused epilogue；
- native 实现需要：
  1. `index_select` cos/sin cache；
  2. 对 q 的 rope 尾部和 k_pe 分别执行 GPT-J RoPE；
  3. 将 k_pe 广播到所有 q heads 做 full-K concat；
  4. 将 `[kv_c_normed | rotated k_pe]` 写入 paged cache；
- 这对每个 MLA 层和每个 prefill chunk 产生多次 launch 和临时 tensor。

目标功能：

- 单 kernel 完成 prefill epilogue：
  - 对 `q[..., -64:]` 原地做 GPT-J RoPE；
  - 对 `k_pe` 做 GPT-J RoPE；
  - 返回 `k_out = [k_nope | rotated_k_pe]`；
  - 原地写 `kv_cache = [kv_c_normed | rotated_k_pe]`。

非目标：

- 不计算 attention；
- 不计算 q/kv projection 或 RMSNorm；
- 不做 fp8 / fp8_ds_mla 量化；首期仅 BF16；
- 不做 decode q concat；decode 见 #04。

**接入前置条件**：K3 prefill 的 V 是 `kv_b_proj` 输出中的 strided view；当前 DLC backend
会执行 `v.contiguous()`。DLC `custom_flash_attention_varlen_GQA_bf16` 源码已有
`v_hbm0->storage_offset == 128 / stride[1] == 256` 的 offset-V 分支，理论上可直接消费该
view；需要先用真实 shape 做 isolate probe。probe 通过后，删除 backend 的两次
`v.contiguous()`，本算子即可与 V view 直连；probe 不通过时才保留 V copy 或另立 attention
ABI 扩展。chunked-context 的 K concat 不在本算子范围内，严格零拷贝见 #07。

## 期望 API 与函数签名

```python
def dlc_kimi_k3_mla_prefill_epilogue_bf16(
    q: torch.Tensor,                 # [T, H, 192], BF16，rope 尾部原地更新
    k_nope: torch.Tensor,            # [T, H, 128], BF16
    k_pe: torch.Tensor,              # [T, 1, 64] 或 [T, 64], BF16
    kv_c_normed: torch.Tensor,       # [T, 512], BF16
    kv_cache: torch.Tensor,          # [num_blocks, page_size, 576], BF16
    slot_mapping: torch.Tensor,      # [T], int32/int64，-1 跳过
    positions: torch.Tensor | None = None,      # [T], int64
    cos_sin_cache: torch.Tensor | None = None,  # [max_pos, 64], FP32/BF16
    *,
    out: torch.Tensor | None = None, # [T, H, 192], BF16
) -> torch.Tensor:
    ...
```

## 参数与返回值

输入 / 属性：

- `q`：full query，最后一维 192；前 128 维为 nope，后 64 维为 RoPE。
- `k_nope`：per-head key nope 部分。
- `k_pe`：单 head 的 RoPE key；按 heads 广播。
- `kv_c_normed`：latent KV 512 维。
- `kv_cache`：paged latent cache；每 page 的最后一维为 576。
- `slot_mapping`：每个 token 写入的扁平 slot；`-1` 表示不写。
- `positions` / `cos_sin_cache`：GPT-J style RoPE；两者同时为空时走 NoPE 路径。

返回值：

- `[T, H, 192]` 的 full key；若提供 `out`，原位写入并返回该 tensor。

副作用：

- `q` 的 rope 尾部原地更新；
- `kv_cache` 命中 slot 的位置原地更新；
- 其余输入不修改。

## 功能语义

```text
rope_dim = 64

if positions is not None and cos_sin_cache is not None:
    q[..., -64:] = gpt_j_rope(q[..., -64:], positions, cos_sin_cache)
    k_pe_rot = gpt_j_rope(k_pe.reshape(T, 1, 64), positions, cos_sin_cache).squeeze(1)
else:
    k_pe_rot = k_pe.reshape(T, 64)

k_out = cat((k_nope, broadcast(k_pe_rot, H)), dim=-1)

latent = cat((kv_c_normed, k_pe_rot), dim=-1)  # [T,576]
for t in range(T):
    if slot_mapping[t] >= 0:
        kv_cache_flat[slot_mapping[t]] = latent[t]
```

GPT-J RoPE 规则（与 `torch.dlc_apply_gpt_j_rope` / K3 native 一致）：

```text
cos, sin = chunk(cos_sin_cache.index_select(0, positions), 2, dim=-1)
x_even, x_odd = x[..., ::2], x[..., 1::2]
out_even = x_even * cos - x_odd * sin
out_odd  = x_odd * cos + x_even * sin
out = interleave(out_even, out_odd)
```

边界：

- `T == 0`：返回 `[0,H,192]`，不写 cache；
- `slot_mapping=-1`：跳过写入；
- `positions` 超出 cache：报错；
- q/k_pe hidden 或 rope 维度不匹配：报错。

确定性：无随机；RoPE 中间计算 FP32，输出一次 cast BF16。

## 支持范围与约束

首期精确范围：

- q head dim = 192，nope = 128，rope = 64；
- 512 latent + 64 rope = 576 cache entry；
- `H` 动态；TP8 生产值为 12；
- `T` 动态，至少覆盖 1、8、64、256、4096；
- BF16 q/k/v/cache，RoPE cache FP32（建议同时接受 BF16）；
- `slot_mapping` int32/int64，1D；
- `kv_cache` 最后一维连续，slot 映射为扁平后的 `[num_blocks*page_size]`；
- 单卡 DLC；不支持 q/kv fp8。

layout 要求：

- q/k_nope 的 last dim 连续；
- k_pe 允许 `[T,1,64]` 或 `[T,64]`；
- 不要求 q/k 的 batch stride 完全连续，但要求最后一维连续且无负 stride。

执行模式：

- eager 必须通过；
- 目标支持 DLC Graph（避免 host 侧索引/分支）。
- 接入时先用 offset-V probe 决定是否移除 `v.contiguous()`；不能把未验证的 V copy
  算作本算子已经消除的中间拷贝。

## 精度要求

Golden / reference：

- 以本 issue 末尾的自包含 PyTorch reference 为准；
- 上游 CUDA kernel 语义由该 reference 完整表达。

输入分布：

- q/k_pe/kv_c 使用 `N(0,1)`；positions 覆盖 0、边界位置和随机位置；
- 至少 200 组，固定 seed。

比较方式：

- q 与 k_out 转 FP32 比较：`atol=1e-2, rtol=1e-2`；
- cache 写入位置与 reference 转 FP32 比较：`atol=1e-2, rtol=1e-2`；
- 未命中 slot 必须保持原值 bit-exact；
- `-1` slot 必须保持原值 bit-exact。

中间精度：

- RoPE cos/sin 和乘法使用 FP32；输出 cast BF16；
- 不允许在 BF16 中累加 `x*cos ± y*sin`。

## 测试范围

- [x] 基本功能与 reference 对齐
- [ ] 多 dtype 精度（首期 BF16）
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [x] 非 contiguous、特殊 stride 或 layout
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式（每 rank 独立处理本地 heads）
- [x] 长时间稳定性
- [x] 性能基准与性能回归

具体测试用例、shape、数据范围和预期结果：

1. `T ∈ {0,1,8,64,256,4096}`，`H=12`，真实 192/128/64/512/576 维度。
2. RoPE / NoPE 两条路径；positions 边界 0 和 max-1。
3. slot mapping：连续、乱序、含 -1；cache 未命中位置 bit-exact。
4. q 输入允许 strided batch view；输出及 k_out 连续。
5. 性能：`T=4096` 单次目标 ≤ 5 ms；相对 native 至少减少 4 次 launch。

## 验证环境与依赖版本

```text
DLC 硬件：8 x TYD，TP8；driver v26.0.1 / Api 26
vLLM：85fd014d5ff4
vLLM DLC（vllm-cl）：ab819da9a0f4 + K3 适配工作区改动
PyTorch / torch-dlc：2.5.0，9049914ec94a
DLC_Custom_Kernel：55b9aef7e0cb
模型：Kimi-K3-INT4-Slice8-TP8 / full K3；MLA qk_nope=128、qk_rope=64、kv_lora=512
```

## 验收标准

- [ ] API、参数校验和错误行为符合约定。
- [ ] q 原地更新、k_out 输出和 cache 写入与 native reference 对齐。
- [ ] `slot_mapping=-1` / 未命中 slot 保持原值。
- [ ] BF16 输出满足 `atol=1e-2, rtol=1e-2`。
- [ ] 支持 `T=4096`，单次 ≤ 5 ms（首期目标）。
- [ ] eager 无 host D2H；目标支持 graph capture。
- [ ] 真实 prefill shape 的 offset-V probe 通过，并删除 `v.contiguous()`；若 probe 失败，
      本条只算 MLA epilogue 优化完成，不算 prefill 全链路零拷贝完成。
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


def mla_prefill_epilogue_reference(
    q: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c_normed: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    if positions is not None and cos_sin_cache is not None and k_pe.shape[0]:
        rope_dim = k_pe.shape[-1]
        q[..., -rope_dim:] = apply_rope_gptj(
            q[..., -rope_dim:], positions, cos_sin_cache
        )
        k_pe = apply_rope_gptj(
            k_pe.unsqueeze(1), positions, cos_sin_cache
        ).squeeze(1)

    insert_latent_cache(kv_cache, kv_c_normed, k_pe, slot_mapping)
    return torch.cat(
        (
            k_nope,
            k_pe.unsqueeze(1).expand(-1, k_nope.shape[1], -1),
        ),
        dim=-1,
    )
```

调用形态：

```python
k = mla_prefill_epilogue_reference(
    q,
    k_nope,
    k_pe,
    kv_c_normed,
    kv_cache,
    slot_mapping,
    positions,
    cos_sin_cache,
)
# q 已被原地旋转；k 直接交给 prefill attention。
```
