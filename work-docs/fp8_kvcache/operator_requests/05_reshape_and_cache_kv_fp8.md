# [Operator Request] FP8 KV cache 融合量化写入算子

## 需求类型

扩展已有算子能力。

## 优先级

P0。

## 目标版本与期望时间

vLLM DLC FP8 KV cache 第一阶段；该算子阻塞 FP8 cache 的生产写入路径。具体交付版本与日期由算子组和接入方共同排期。

## 算子名称

- 建议 PyTorch 接口：`torch.dlc_reshape_and_cache_kv_fp8`。
- 建议 kernel 名称：`custom_reshape_and_cache_kv_fp8`。
- 本需求只包含一个融合算子：同时完成 BF16 K/V 量化、FP8 values 写入和 FP32 scales 写入；不包含 attention、gather 或 block copy。

## 背景与功能说明

使用场景：

- Qwen3 类 GQA 模型使用 `head_size=128`，KV head 在 TP4 下每 rank 为 2，`block_size=256`。
- 持久化 KV cache 采用分离布局：每个 token 保存 `[K heads | V heads]` 的 FP8 E4M3FN values，以及对应的逐 token、逐 KV head FP32 scale。
- 该布局每 token 使用 528 B，而 BF16 需要 1024 B；每个 256-token page 为 135168 B，而 BF16 page 为 262144 B。持久化容量约为 BF16 的 1.939 倍。

当前问题：

- 现有 DLC cache 写入路径只处理 BF16 K/V 和 BF16 cache，不能生成 FP8 values 或 FP32 scales。
- FP8 values 与 scales 是两个逻辑区域；scale 每 token 只有 16 B（TP4、2 个 KV head），不能使用普通按元素 Tensor 写入，否则会被 DLC 的小尾维 storage padding 放大。
- Python 逐 token 构造、量化和 copy 会引入大量 host/device 往返，无法用于 graph 或生产 decode。

目标功能：

- 一次 device launch 完成 BF16 K/V 按 group128 量化、FP8 E4M3FN values 生成、实际 FP32 scale 生成和 paged cache scatter。
- values 和 scales 直接写入同一 raw cache page 的对应区域，不产生额外的持久化 scale Tensor。
- 支持 `slot_mapping`、跨 block slot、`slot=-1` 跳过、非连续目标 slot 顺序和双 XYS。

非目标：

- 不实现 attention、cache gather/dequant、block copy、swap 或 zero。
- 不实现标准 MXFP8、group32、E5M2 或全局单 scale。
- 不支持 merged K/V cache、MLA cache 或 Mamba/KDA state。
- 首期不以 TP8 单 KV head 为发布阻塞项；接口和地址计算不得依赖每 token values 必须为 512 B。

## 期望 API 与函数签名

```python
def dlc_reshape_and_cache_kv_fp8(
    key: torch.Tensor,          # [T, H_kv, 128], bfloat16
    value: torch.Tensor,        # [T, H_kv, 128], bfloat16
    kv_cache: torch.Tensor,     # [num_blocks, page_bytes], float8_e4m3fn
    slot_mapping: torch.Tensor, # [T], int64
    block_size: int,
) -> None:
    ...
```

建议 C++ 签名为：

```cpp
void dlc_reshape_and_cache_kv_fp8(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& kv_cache,
    const at::Tensor& slot_mapping,
    int64_t block_size);
```

返回值为 `None`。`kv_cache` 必须标记为 mutation；`key`、`value` 和 `slot_mapping` 只读。

## 参数与返回值

输入：

- `key`：BF16 K，逻辑 shape `[T, H_kv, 128]`。
- `value`：BF16 V，逻辑 shape `[T, H_kv, 128]`。
- `slot_mapping`：INT64 一维 Tensor，shape `[T]`，每项为全局 slot。
- `block_size`：Python `int`，每个物理 page 包含的 token 数。
- `H_kv` 从 `key.size(1)` 获取，必须与 `value.size(1)` 相同。
- `num_blocks` 从 `kv_cache.size(0)` 获取；`page_bytes` 从 `kv_cache.size(1)` 获取。

输出：

- `kv_cache`：原地写入 FP8 values 与 FP32 scale 的原始字节。
- 函数返回 `None`。

各输入的 dtype、shape、layout 和 device 约束集中见“支持范围与约束”。

## 功能语义

### Cache page 布局

固定：

```text
head_size D = 128
group_size G = 128
B = block_size
H = H_kv

token_value_bytes  = 2 * H * 128
token_scale_bytes  = 2 * H * 4
values_region_bytes = B * token_value_bytes
scales_region_bytes = B * token_scale_bytes
page_bytes          = values_region_bytes + scales_region_bytes
```

一个 page 的字节布局为：

```text
[ values: B * token_value_bytes ][ scales: B * token_scale_bytes ]
```

page 内第 `s` 个 token（`0 <= s < B`）的有效字节为：

```text
values_base = s * token_value_bytes
K head h    = values_base + h * 128
V head h    = values_base + H * 128 + h * 128

scales_base = values_region_bytes + s * token_scale_bytes
sK(t,h)     = scales_base + h * 4
sV(t,h)     = scales_base + (H + h) * 4
```

逻辑顺序必须固定为：

```text
values: [K0, K1, ..., K(H-1), V0, V1, ..., V(H-1)]
scales: [sK0, sK1, ..., sK(H-1), sV0, sV1, ..., sV(H-1)]
```

对于 `B=256,H=2`：

```text
token_value_bytes   = 512
token_scale_bytes   = 16
values_region_bytes = 131072
scales_region_bytes = 4096
page_bytes          = 135168
```

### 量化语义

对每个 token、每个 K/V head 的 128 个元素独立执行 group128 量化。`x` 为 BF16 输入转 FP32 后的 128 个元素：

```text
finite_amax = max(abs(x_i))，仅统计 finite 元素
if 没有 finite 元素: finite_amax = 0
amax   = max(finite_amax, 1e-4)
scale  = FP32(amax * FP32(1.0 / 448.0))
q      = E4M3FN_RNE(clamp(x / scale, -448.0, 448.0))
```

- `scale` 必须是实际正 FP32 乘子，不是 E8M0 scale code，也不是全局 descale。
- `scale` 写入时必须保持 FP32 bit pattern，不得先转 BF16。
- 正负零保留符号；NaN 输出 canonical E4M3FN NaN `0x7f`；±Inf 不参与 amax，并在缩放后饱和为带符号 448。
- 有效 group 的 scale 不得为零。所有元素均为正零或负零的 group 使用 `amax=1e-4`、`scale=FP32(1e-4/448)`，values 保留对应带符号零。
- 没有 finite 元素的 group 同样使用 `amax=1e-4`、`scale=FP32(1e-4/448)`；其中的 NaN 和 ±Inf 仍按上一行的特殊值规则输出，不得被统一改写成全零。
- FP8 舍入必须与该 CPU reference 一致：先 FP32 除法和 clamp，再执行 E4M3FN round-to-nearest-even。

### 写入语义

对每个 `t in [0, T)`：

```text
slot = slot_mapping[t]
if slot == -1:
    跳过该 token，不写 values 和 scales
else:
    block  = slot // block_size
    in_blk = slot % block_size
    写入 K/V 的 2 * H * 128 个 FP8 bytes
    写入 2 * H 个 FP32 scale
```

- 只允许 `-1` 作为负 slot；其他负值非法。
- 同一调用中有效 slot 必须唯一；重复 slot 不在支持范围内，调用方必须保证不会发生。
- 不同 token 的写入必须并发安全，不得依赖 token 顺序。
- 未命中的 page 字节、其他 slot 的 values/scales 和 page padding 必须保持原值。
- 所有 page offset、slot offset 和字节地址计算必须使用 64 位整数，防止大 cache 上发生 32 位截断。
- `T=0` 时不启动实际工作，`kv_cache` 保持不变。

## 支持范围与约束

| 项目 | 首期范围 |
|---|---|
| 设备 | DLC/TYD，真机 |
| 输入 dtype | `key`、`value` 为 BF16 |
| cache dtype | 二维 raw `float8_e4m3fn` byte arena |
| head_size | 固定 128 |
| group_size | 固定 128 |
| KV heads | 首期强制验收 `H=2`；实现应可扩展到 `H∈{1,4,8}` |
| block_size | 首期强制验收 256；支持 256、512、1024 |
| token 数 | `0 <= T <= 65536` 动态 |
| slot dtype | INT64 |
| slot 范围 | `-1 <= slot < num_blocks * block_size` |
| page bytes | 必须等于 `block_size * (2*H*128 + 2*H*4)` |
| device 布局 | 所有 Tensor 位于同一 DLC device |
| Tensor 布局 | 全部 contiguous，`storage_offset=0` |
| 对齐 | cache 基址建议 512 B 对齐；不得逐 token 假设 512 B values 行 |
| 执行模式 | 普通 stream 与 graph capture/replay |
| 多卡 | 每个 TP rank 独立调用；不需要跨卡同步 |

约束：

- 不得通过 Python 循环逐 token 量化和 copy。
- 不得生成并持久化额外的 `[num_slots, heads]` FP32 scale Tensor。
- 不得依赖 FP8 Tensor 的小尾维自动 padding；cache 必须按 raw page byte 计算。
- `H=1` 时每 token values 为 256 B，仍必须按本布局正确写入，不能因为不足 512 B 而拒绝或扩成 512 B。
- 参数 shape、dtype、page_bytes、head 数和 block_size 不一致时必须在 launch 前报错。

## 精度要求

Golden/reference 实现：

- 使用本文末尾自包含 CPU reference。
- 参考输入必须先生成 BF16，再转 FP32 量化和计算 scale。
- FP8 values 与预期 raw bytes 逐字节一致。
- FP32 scales 与预期 FP32 bit pattern 逐位一致。
- 不允许仅比较量化后再反量化的数值误差代替 bytes/scale 检查。

特殊值：

- 覆盖正零、负零、全零 group、极小值、subnormal、448 附近、Inf、-Inf、NaN、单点 outlier 和混合 scale group。
- NaN 必须为 E4M3FN canonical NaN `0x7f`；NaN 不参与 finite amax。
- Inf 必须不参与 amax，并按带符号 448 饱和。
- 全零 group 的 scale 和 values 必须与 reference 完全一致。

完整 cache：

- 目标 slot 之外的既有字节、其他 page 和 padding 必须逐字节不变。
- `slot=-1` 不得触发任何写。
- 跨 page、非连续 slot 和乱序 slot 的最终 cache 必须与 reference 逐字节一致。

## 测试范围

- [x] 基本功能与 reference 对齐
- [x] 多 dtype 输入和 cache 类型检查
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [x] 非连续 slot 顺序、跨 page 与特殊 layout
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式（逐 rank 独立）
- [x] 长时间稳定性
- [x] 性能基准与性能回归

具体测试用例：

1. `T ∈ {0,1,17,127,128,255,256,257,4096}`；`H=2,B=256` 为必测基线。
2. `H ∈ {1,2,4,8}`；`B ∈ {256,512,1024}`；检查 page_bytes、token 偏移和 scale 偏移。
3. slot 覆盖 page 首尾、跨 page、逆序、随机乱序、重复 page 的不同 slot，以及 `-1` 与其他负值。
4. 输入分布覆盖全零、正负常量、标准正态、大值、小值、subnormal、Inf、NaN、每个 group 单 outlier。
5. K 与 V 使用不同 scale，K head 与 V head 使用不同 scale，避免所有 scale 相同掩盖 K/V 顺序错误。
6. cache 预填 poison，验证只修改目标 slot 的 values/scales，其他字节完全不变。
7. 连续调用至少 1000 次，交替不同 T、H 和 block_size，检查无语义漂移和只读输入不变。
8. stream、双 XYS、graph capture/replay 后 replay 新 slot 数据。
9. 构造大 block index 和接近地址上限的 page offset，核对所有地址、byte offset 和 DMA length 为 64 位。
10. 性能报告记录 launch 数、总耗时、DMA 覆盖率和 cache 有效带宽；首期不以速度作为阻断阈值，但要求一次调用完成整批写入且不存在逐 token host 循环。

## 验证环境与依赖版本

- DLC 硬件：TYD 真机，`DLC_TPU_VERSION=2`。
- 必须关闭模拟模式，并在报告中记录实际 device index。
- 验收报告记录 PyTorch、DLC_Custom_Kernel、LLVM、DLCSynapse、DLCsim 的完整 Git SHA 和工作区状态。
- 记录 driver、RTABI、kernel 共享库 SHA256、`CUSTOM_KERNEL_PATH`、`LD_LIBRARY_PATH` 和 `DLC_SYN_*` 实际值。
- 新增 Python/C++ 入口和 kernel 必须完成实际构建、安装、注册和加载，不能只凭源码或二进制文件存在判定通过。

## 验收标准

- [ ] Python/C++ API、mutation schema、默认值和参数校验符合约定。
- [ ] BF16 K/V 经一次调用即可写入 FP8 values 与 FP32 scales，无额外持久化 scale Tensor。
- [ ] `H=2,B=256` 时 page_bytes 严格为 135168：values 131072 B、scales 4096 B。
- [ ] `H=1` 时每 token values 256 B 的布局正确，不发生隐式 512 B 扩行。
- [ ] FP8 values 和 FP32 scales 与自包含 reference 逐字节/逐位一致。
- [ ] 所有 slot、page 边界、`slot=-1`、跨页、乱序、graph、stream 和双 XYS 用例通过。
- [ ] 未命中区域的 cache 内容逐字节不变；输入 Tensor 不被修改。
- [ ] 大地址计算无 32 位溢出，无逐 token host sync 或 Python 写循环。
- [ ] 自动测试、调用示例、参数错误说明和性能报告已提交。

## Reference、日志与附件

本需求不依赖外部代码、外部文档、日志或附件。以下 CPU reference 是完整、自包含的布局与量化 golden；仅依赖 CPU PyTorch，且不调用待测 DLC 算子。

```python
import torch


def reference_reshape_and_cache_kv_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    assert key.dtype == value.dtype == torch.bfloat16
    assert key.ndim == value.ndim == 3
    assert key.shape == value.shape
    assert key.shape[2] == 128
    assert kv_cache.dtype == torch.float8_e4m3fn
    assert kv_cache.ndim == 2
    assert slot_mapping.dtype == torch.int64

    tokens, heads, head_size = key.shape
    token_value_bytes = 2 * heads * head_size
    token_scale_bytes = 2 * heads * 4
    values_region_bytes = block_size * token_value_bytes
    page_bytes = values_region_bytes + block_size * token_scale_bytes

    assert head_size == 128
    assert slot_mapping.shape == (tokens,)
    assert kv_cache.shape[1] == page_bytes

    cache_bytes = kv_cache.view(torch.uint8)

    def quant_group(x_bf16: torch.Tensor):
        x = x_bf16.to(torch.float32)
        finite = torch.isfinite(x)
        finite_amax = x[finite].abs().max() if finite.any() else x.new_zeros(())
        amax = torch.maximum(finite_amax, x.new_tensor(1e-4))
        scale = amax * x.new_tensor(1.0 / 448.0)
        q = torch.clamp(x / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
        return q, scale

    for token in range(tokens):
        slot = int(slot_mapping[token].item())
        if slot == -1:
            continue
        assert slot >= 0

        block = slot // block_size
        in_block = slot % block_size
        value_parts = []
        scales = []

        for side in (key, value):
            for head in range(heads):
                q, scale = quant_group(side[token, head])
                value_parts.append(q.view(torch.uint8))
                scales.append(scale)

        value_bytes = torch.cat(value_parts)
        scale_bytes = torch.stack(scales).view(torch.uint8)
        value_start = in_block * token_value_bytes
        scale_start = values_region_bytes + in_block * token_scale_bytes

        cache_bytes[block, value_start:value_start + token_value_bytes] = value_bytes
        cache_bytes[block, scale_start:scale_start + token_scale_bytes] = scale_bytes
```
