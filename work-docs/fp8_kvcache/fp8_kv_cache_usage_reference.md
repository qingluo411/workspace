# FP8 KV Cache 使用参考（分离 FP8 values + FP32 scales）

日期：2026-10-27

状态：PyTorch 侧布局、参考写入/读取和容量验证已完成；融合 paged
writer/reader 尚未实现。本文描述当前 `torch.dlc.fp8_kv_cache` 的正确使用
方式，不把逐 token `copy_` 参考路径当成生产 decode 性能路径。

## 1. 适用范围

当前主线面向 GLM-5.3-Flash 的稀疏 MLA cache：

- `head_size = 512`
- `block_size = 256`
- group128 量化
- FP8 values + 每 token 4 个 FP32 group scale
- 每个逻辑 token 占用 `512B + 16B = 528B`

`torch.dlc.fp8_kv_cache` 也允许 `head_size` 为 512 的正整数倍，但模型接入
仍应以 GLM 的 512 宽度为验收目标。`group_size` 固定为 128。

## 2. 布局契约

### 2.1 两个独立 tensor

```text
values: [num_blocks, block_size, head_size]       torch.float8_e4m3fn
scales: [num_slots * num_groups]                  torch.float32

num_slots  = num_blocks * block_size
num_groups = head_size // 128
```

`values` 和 `scales` 必须同时创建、同时保存、同时传给 attention 层。不能把
两者拼成一个普通 FP8 tensor，也不能假定调用方仍然只接收一个 KV cache
tensor。

### 2.2 GLM 的具体几何

```text
values: [B, 256, 512]
scales: [B * 256 * 4]

一个 token:
  values: 512 * 1 = 512B
  scales:     4 * 4 =  16B
  total:              528B

一个 block:
  values: 256 * 512 = 131072B
  scales: 256 *  16 =   4096B
```

`values` 的物理 token pitch 必须是 512B，block pitch 必须是 `256 * 512B`。
`scales` 的 element pitch 必须是 4B，token pitch 必须是 16B。

### 2.3 禁止的布局

| 布局 | DLC 实际 token 占用 | 结果 |
|---|---:|---|
| `[B, 256, 512]` FP8 + flat `[slots * 4]` FP32 | 528B | 正确，主线 |
| `[slots, 4]` FP32 scales | 512B scales | 总占用回到 1024B/token |
| `[B, 256, 516]` FP8（legacy uint8 scale 内联） | 1024B | 无节省 |
| `[B, 256, 528]` FP8（FP32 scale 内联） | 1024B | 无节省 |
| 一维 FP8 raw 528B 自定义 cache | 528B | 理论可行，但需全套 raw ABI，当前不作主线 |

尤其注意：把 `[slots, 4]` 的 scale 再 `view(-1)` 成一维，逻辑 shape
虽然是 `[slots * 4]`，物理 token pitch 仍然是 512B；布局校验会通过
`data_ptr()` 检测并拒绝这种伪装。

### 2.4 真机地址步长确认：`[blocks, 256, 516]`

`[blocks, 256, 516]` 是“512B FP8 values + 4B uint8 scale”的 legacy
内联行宽；若内联当前 FP32 scale，逻辑行宽则是
`512 + 4 * 4 = 528B`。两者看起来都比 BF16 的 1024B/token 小，但 DLC
对一字节 tensor 按 512B tile 组织物理行，超过 512B 的最后维度会被补到
下一个 1024B 边界，因此它们都拿到相同的 1024B/token 实际 pitch。

在 TYD 真机上以 `B=4`、`block_size=256` 调用 `data_ptr()` 测量，结果如下：

| FP8 tensor | 逻辑 `stride()` | 逻辑 `nbytes()` | token pitch | block pitch | 按 pitch 计 4 blocks |
|---|---:|---:|---:|---:|---:|
| `[4,256,512]` | `(131072,512,1)` | 524,288 B | 512 B | 131,072 B | 524,288 B |
| `[4,256,516]` | `(132096,516,1)` | 528,384 B | 1024 B | 262,144 B | 1,048,576 B |
| `[4,256,528]` | `(135168,528,1)` | 540,672 B | 1024 B | 262,144 B | 1,048,576 B |

`[4,256,516]` 的逻辑 `nbytes()` 只有 528,384B，但地址 pitch 表明四块实际
占 1,048,576B，和 BF16 values 完全相同。`stride()` 以及
`untyped_storage().nbytes()` 只反映 PyTorch 逻辑视图，不能用它们判断
HBM 占用；这里的容量结论来自相邻元素的 `data_ptr()` 差值。

scale 也有同类问题。以 1024 slots 为例：

| FP32 scale tensor | 逻辑 `stride()` | 逻辑 `nbytes()` | 相邻 token pitch | 1024 slots 实际 pitch |
|---|---:|---:|---:|---:|
| `[1024,4]` | `(4,1)` | 16,384 B | 512 B | 524,288 B |
| `[4096]` | `(1,)` | 16,384 B | 16 B | 16,384 B |

把 `[1024,4]` 再 `view(-1)` 不会改变底层地址映射，第 4 个 scale 仍可能
落在下一个 512B 行。因此正确布局必须是两个独立 tensor：

- values 使用恰好 512B 的行宽，token pitch 为 512B；
- scales 从分配时就使用一维 `[slots * groups]`，token pitch 为 16B。

同样的步长结论在 `B=32`、8192 slots 的实际分配中复现：正确分离布局为
4,325,376B，BF16 与 `[slots,4]` scale、516/528B 内联行都是
8,388,608B。详见 5.2 节。

## 3. 快速开始

### 3.1 创建 cache

```python
import torch
from torch.dlc.fp8_kv_cache import (
    FP8KVCacheLayout,
    create_fp8_kv_cache,
)

B = 32
BLOCK_SIZE = 256
HEAD_SIZE = 512
DEVICE = "dlc:0"

layout = FP8KVCacheLayout(B, BLOCK_SIZE, HEAD_SIZE)
cache = create_fp8_kv_cache(
    B,
    BLOCK_SIZE,
    HEAD_SIZE,
    device=DEVICE,
)

print(cache.values.shape)       # torch.Size([32, 256, 512])
print(cache.values.stride())    # (131072, 512, 1)
print(cache.scales.shape)       # torch.Size([32768])
print(cache.scales.stride())     # (1,)
print(layout.bytes_per_token)    # 528
print(layout.storage_saving)     # 0.484375
```

`create_fp8_kv_cache(..., zero=True)` 会同时清零 FP8 values 和 FP32 scales。
如果调用方自己分配 tensor，可以直接接管并执行同一套物理布局校验：

```python
from torch.dlc.fp8_kv_cache import FP8KVCache

values = torch.zeros(
    (B, BLOCK_SIZE, HEAD_SIZE),
    dtype=torch.float8_e4m3fn,
    device=DEVICE,
)
scales = torch.zeros(
    (B * BLOCK_SIZE * (HEAD_SIZE // 128),),
    dtype=torch.float32,
    device=DEVICE,
)
cache = FP8KVCache(values, scales)
```

如果 `values`/`scales` 的 shape、stride、dtype、device 或真实地址 pitch 不
符合契约，构造 `FP8KVCache` 会直接抛出 `ValueError`。不建议只检查逻辑
`nbytes()`，因为 DLC 的一字节 tensor 和二维 FP32 scale 都可能发生物理
padding。

### 3.2 量化写入

`hidden_states` 是已经完成模型投影/归一化、准备写入 cache 的 BF16 token
行，shape 为 `[T, 512]`；`slot_mapping` 是 `[T]` 的 int64 展平 slot id。

```python
hidden = (torch.randn(8, 512) * 3).to(torch.bfloat16).to(DEVICE)
slot_mapping = torch.tensor(
    [0, 1, 255, 256, 257, 511, 512, -1],
    dtype=torch.int64,
    device=DEVICE,
)

cache.quantize_and_scatter(hidden, slot_mapping)
```

语义：

- 每个 128 维 group 独立计算 FP32 scale；
- 负 slot 跳过写入；
- `slot_mapping` 长度为 `T`；
- 默认 `scale_round=False`，保留原始 FP32 scale；
- 传 `scale_round=True` 时 scale 向上取 2 的幂，便于特定数值路径；
- 值和 scale 由同一参考 API 配对写入；当前实现内部为逐 token 设备
  `copy_`，不是融合 kernel。

### 3.3 Gather 与反量化

```python
values, scales = cache.gather_quantized(slot_mapping)
print(values.shape)   # (8, 512), float8_e4m3fn
print(scales.shape)   # (32,), float32

restored = cache.gather_dequantize(slot_mapping)
print(restored.shape) # (8, 512), bfloat16
```

`gather_quantized` 返回连续的 FP8 values 和扁平 FP32 scales，可直接送入
后续 attention kernel。`gather_dequantize` 方便做 golden、调试和不改
attention 的 bring-up。

负 slot 语义：

- `quantize_and_scatter`：任意负 slot 都跳过；
- `gather_quantized`/`gather_dequantize`：当前只接受 `-1` 作为无效 slot，
  返回零行；更小的负数会在调用前报错。

### 3.4 Block copy、prefix cache 与清零

```python
# 成对复制，src/dst 长度必须相同；重复 src 可做一对多。
cache.copy_blocks([0, 0], [3, 4])

# 清零指定 block 的 values 和 scales。
cache.zero_blocks([3, 4])

# 清零整个 cache。
cache.zero_()
```

`copy_blocks` 会同时复制 FP8 values 和 FP32 scales，避免两者失配。不要用
`cache.values[dst] = cache.values[src]` 这类高级索引写入进入生产路径；
当前 DLC 对 FP8 高级索引和 `index_copy_`/`scatter_` 仍可能落到 CPU 回退。

## 4. 直接使用底层算子

如果调用方已有自己的 cache 管理，只需要使用 quant/dequant 的底层 ABI：

```python
T = 8
quantized = torch.empty(
    (T, 512),
    dtype=torch.float8_e4m3fn,
    device=DEVICE,
)
token_scales = torch.empty(
    (T * 4,),
    dtype=torch.float32,
    device=DEVICE,
)

torch.dlc_quant_e4m3(hidden, quantized, token_scales, 128, False)
restored = torch.dlc_dequant_e4m3(quantized, token_scales, 128)
```

约束：

- `hidden`：2D contiguous BF16；
- `quantized`：2D contiguous E4M3FN；
- `token_scales`：1D contiguous FP32，长度 `T * ceil(512/128)`；
- `quantized` 和 `token_scales` 必须位于同一个 DLC device；
- 反量化输入必须是 2D contiguous FP8 values 和 1D contiguous FP32 scales。

## 5. 容量计算与显存节省

### 5.1 公式

令 `slots = num_blocks * block_size`，`groups = head_size / 128`：

```text
BF16 bytes       = slots * head_size * 2
FP8 values bytes = slots * head_size
FP32 scales bytes= slots * groups * 4
FP8 total bytes  = slots * (head_size + groups * 4)

saving = 1 - (head_size + groups * 4) / (head_size * 2)
```

GLM `head_size=512`、`groups=4`：

```text
saving = 1 - 528 / 1024 = 48.4375%
```

### 5.2 真机实测

测试条件：`B=32`、`block_size=256`、`slots=8192`，在新进程中用
`torch.dlc.memory_allocated()` 测量真实分配：

| 布局 | 实际分配 | 说明 |
|---|---:|---|
| BF16 values | `8,388,608 B`（8.000 MiB） | 基准 |
| FP8 values + flat FP32 scales | `4,325,376 B`（4.125 MiB） | values 4,194,304 + scales 131,072 |
| FP8 + `[slots,4]` scales | `8,388,608 B`（8.000 MiB） | scale 每 token 被补到 512B |
| FP8 inline `[B,256,528]` | `8,388,608 B`（8.000 MiB） | 每 token 被补到 1024B |

正确分离布局相对 BF16 实际节省 `4,063,232 B`（3.875 MiB），比例
`48.4375%`。反例的实际分配没有节省。

按 GLM 的 11 个 sparse MLA 层、每层 1411 blocks（即每层 361,216 slots）
估算：

```text
单层 BF16       = 352.750 MiB
单层 FP8+scale  = 181.887 MiB
单层节省        = 170.863 MiB

11 层 BF16      = 3880.250 MiB
11 层 FP8+scale = 2000.754 MiB
11 层共节省     = 1879.496 MiB（约 1.84 GiB）
```

这里假设每层使用相同的 1411-block 池；实际模型应按每层真实 allocator
配置重新计算。容量规划建议直接在目标设备上构造 `FP8KVCacheLayout`：

```python
layout = FP8KVCacheLayout(num_blocks, 256, 512)
bf16_mib = layout.num_slots * layout.bf16_bytes_per_token / 2**20
fp8_mib = layout.num_slots * layout.bytes_per_token / 2**20
print(bf16_mib, fp8_mib, layout.storage_saving)
```

## 6. 可运行示例与测试

### 6.1 使用示例

示例脚本：

[`work-tests/fp8_kvcache/example_fp8_kv_cache.py`](../../work-tests/fp8_kvcache/example_fp8_kv_cache.py)

运行：

```bash
export PATH=/root/workspace/.venv/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/chipltech/dlccl/lib:/usr/local/chipltech/simulator/lib:/usr/local/chipltech/synapse/lib:/usr/local/chipltech/bluejay/build:${LD_LIBRARY_PATH:-}

python work-tests/fp8_kvcache/example_fp8_kv_cache.py --device dlc:0
python work-tests/fp8_kvcache/example_fp8_kv_cache.py \
  --device dlc:0 --blocks 1411 --layers 11
```

脚本会实际比较 BF16、正确分离布局、`[slots,4]` scale 和 528B 内联行，并
演示量化写入、gather/dequant、负 slot、block copy 和 zero。

### 6.2 真机测试

测试文件：

[`pytorch/test/dlc_ops/test_fp8_kv_cache.py`](../../pytorch/test/dlc_ops/test_fp8_kv_cache.py)

```bash
export PATH=/root/workspace/.venv/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/chipltech/dlccl/lib:/usr/local/chipltech/simulator/lib:/usr/local/chipltech/synapse/lib:/usr/local/chipltech/bluejay/build:${LD_LIBRARY_PATH:-}

python -m pytest --noconftest -q pytorch/test/dlc_ops/test_fp8_kv_cache.py
```

当前结果：`6 passed`。覆盖：

- 精确 shape/stride/token pitch/block pitch 和 48.4375% 节省；
- 拒绝 516/528 内联行、`[slots,4]` scale 及其 flatten view；
- FP8 `zeros/fill_/zero_`、block copy/zero；
- BF16 -> FP8 + FP32 scale -> 跨 block/乱序/重复/负 slot -> gather -> dequant；
- 空 token 更新和越界 slot 的前置拒绝。

## 7. 当前限制

1. **参考写入/读取不是性能路径**  
   `quantize_and_scatter` 和 `gather_quantized` 当前按 token 执行设备
   `copy_`，用于正确性 bring-up。正式 decode 需要融合 paged writer/reader；
   不能把逐 token 多 launch 的实现直接放进 decode。

2. **不要使用 FP8 高级索引写入**  
   当前 DLC 的 FP8 `index_copy_`/`scatter_` 没有设备实现，部分高级索引和
   `index_select` 会静默走 CPU 回退。cache 管理请使用
   `copy_blocks`、`zero_blocks`，或者等融合 writer/reader。

3. **scale 必须整体分配**  
   一维 `scales` 应由 allocator 一次分配；不要先分配 `[slots,4]` 再
   `view(-1)`。逻辑 shape 正确不代表物理 pitch 正确。

4. **不要把 values 和 scales 拼成普通单 tensor**  
   当前 attention API 仍需要显式接收两个 cache；如果后续做 raw 528B
   backing，需要单独设计 allocator、block copy、zero 和 kernel ABI，不能
   直接把三维 `[B,256,528]` 当作省显存布局。

5. **当前没有 vLLM runner 接入**  
   本文只覆盖 PyTorch 侧使用；vLLM/vllm-cl 的 cache spec、allocator、
   graph writer/reader 仍需后续接入和验证。

## 8. 相关文档

- [GLM-5.3 Sparse MLA FP8 实施契约](vllm-cl/glm53_sparse_mla_fp8_plan_20261027.md)
- [FP8 KV cache 归档索引](README.md)
- [测试和示例入口](../../work-tests/fp8_kvcache/README.md)
- [PyTorch 布局 helper](../../pytorch/torch/dlc/fp8_kv_cache.py)
