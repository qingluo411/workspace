# FP8 KV Cache 算子数值与布局契约

本说明供 kernel 实现和测试使用，细化[适配设计](design.md)中的数值与存储契约。固定使用 `torch.float8_e4m3fn` cache、BF16 Q/K/V 输入和计算操作数、FP32 缩放与累加，D=128、P=256。本文给出目标算法和 reference，不表示这些新 KV 算子已经实现。

## 1 E4M3FN 的位定义

一个元素恰好一个 byte：

```text
bit      7     6 5 4 3     2 1 0
字段     S       E          M
宽度     1       4          3

S = (code >> 7) & 1
E = (code >> 3) & 15
M = code & 7
指数 bias = 7
```

解码规则：

| 条件 | 数值 |
|---|---|
| E=0，M=0 | 带符号的零，正零 0x00、负零 0x80 |
| E=0，M>0 | `(-1)^S * M * 2^-9`，subnormal，无隐含前导 1 |
| E>0，且不是 E=15/M=7 | `(-1)^S * (1 + M/8) * 2^(E-7)` |
| E=15，M=7 | NaN；0x7f 和 0xff 都是 NaN |

**E=15/M=0..6 仍是有限数，不能按 IEEE FP16/FP32 的规则全部当作 Inf/NaN。E4M3FN 没有 Inf 编码。** 本 KV writer 统一写 NaN 编码 0x7f；reader 接受两种 NaN 编码。

| 特征或值 | 数值 | 编码 |
|---|---:|---|
| 最小正 subnormal | 2^-9 = 0.001953125 | 0x01 |
| 最大正 subnormal | 7*2^-9 = 0.013671875 | 0x07 |
| 最小正 normal | 2^-6 = 0.015625 | 0x08 |
| 0.5 | 0.5 | 0x30 |
| 1 | 1 | 0x38 |
| 1.25 | 1.25 | 0x3a |
| -1.5 | -1.5 | 0xbc |
| 2 | 2 | 0x40 |
| 最大有限值 | 1.75*2^8 = 448 | 0x7e |
| 最小有限值 | -448 | 0xfe |

格式依据：[ONNX E4M3FN 规范](https://onnx.ai/onnx/technical/float8.html)、[本地 PyTorch 格式定义](/Users/qingluo/fp8/pytorch/c10/util/Float8_e4m3fn.h:3)。不要使用 E4M3FNUZ 的指数偏置或 NaN/零规则。

## 2 scale 和数值操作顺序

每层提供两个 FP32 scalar：`sK>0`、`sV>0`，必须有限。它们是 descale，即读取时乘的尺度；K 用 sK、V 用 sV。scale 不放在 FP8 byte 中，不按 token/head 动态生成，不要求是二次幂。算子只消费既定 scale；校准及缓存有效期间保持 scale 不变由框架负责。

定义 `RN32` 为舍入成 FP32，`RNE` 为最近偶数舍入。一个元素的规范过程为：

```text
量化写入：
  x32 = exact_bf16_to_fp32(x_bf16)
  y32 = RN32(x32 / scale32)
  code = E4M3FN_RNE_SATFINITE(y32)

反量化读取：
  d32 = exact_e4m3fn_decode(code)
  z32 = RN32(d32 * scale32)
  x_compute_bf16 = BF16_RNE(z32)
```

BF16→FP32 扩展是精确的：对有限值，BF16 的 16-bit bits 左移 16 位即对应 FP32 bits。FP8→FP32 解码也能精确表达所有有限 E4M3FN 值。量化损失来自写入的 FP8 舍入；读取乘 scale 后转 BF16还可能有额外舍入，不能恢复原始 BF16 的所有信息。

必须先在 FP32 中缩放，不能先把 scale 转成 BF16。也不能默认把 `RN32(x/s)` 改成 `RN32(x*RN32(1/s))` 或近似 reciprocal；中点附近的结果可能跨过 FP8 舍入边界。如果用倒数乘优化，需证明其编码满足本 reference，或另行明确数值契约，不能无声替换。

## 3 量化的具体计算

### 3.1 特殊值与饱和

对缩放后的 FP32 值 y：

1. 如果 y 是 NaN，返回 0x7f，不依赖不同 min/max 指令的 NaN 处理方式。
2. 保存 y 的 sign bit，包含 -0 的符号。
3. `a = min(abs(y), 448)`；±Inf 因此饱和到 ±448。
4. 把 a 舍入到最接近的非负 E4M3FN 有限值；等距时选择编码最低位为 0 的值。
5. 设置保存的符号位。

这是 KV writer 的饱和策略。通用 PyTorch `fill_` 应使用 PyTorch scalar 转换语义，不套用 KV scale 或这里的 SATFINITE 策略。

### 3.2 数学 reference

以下是描述算法的伪代码，不要求 kernel 实际计算 log2。`ilogb(a)` 表示从浮点指数求 `floor(log2(a))`，`RNE_int` 表示舍入到最近偶整数。

```text
quantize_scaled_fp32(y):
    if isnan(y):
        return 0x7f

    S = signbit(y)
    a = min(abs(y), 448)

    if a < 2^-6:
        c = RNE_int(a * 512)
        # c=0..7 为 zero/subnormal；c=8 自然进位成最小 normal 0x08。
    else:
        e = ilogb(a)
        n = RNE_int(a * 2^(3-e))
        # n 为包含隐含前导 1 的四位有效数字，范围 8..16。
        if n == 16:
            n = 8
            e = e + 1
        E = e + 7
        M = n - 8
        c = (E << 3) | M

    return (S << 7) | c
```

饱和到 448 后不会产生 E=15/M=7；其最大有限码为 0x7e。不要把 BF16 高字节直接截出来，也不要只把 mantissa 右移而不处理 bias、RNE 和 subnormal。

### 3.3 舍入测试向量

下表输入都是缩放完成后的 FP32 值：

| 输入 y | 输出 code | 解码值 | 原因 |
|---:|---|---:|---|
| +0 | 0x00 | +0 | 保留零 |
| -0 | 0x80 | -0 | 保留符号 |
| 1.0625 | 0x38 | 1 | 1 与 1.125 的中点，选末位偶数码 |
| 1.1875 | 0x3a | 1.25 | 1.125 与 1.25 的中点，选末位偶数码 |
| 2^-10 | 0x00 | +0 | 0 与最小 subnormal 的中点 |
| -2^-10 | 0x80 | -0 | 相同中点规则，保留符号 |
| 2^-9 | 0x01 | 2^-9 | 最小 subnormal 可精确表示 |
| 448 | 0x7e | 448 | 最大有限值 |
| 512 或 +Inf | 0x7e | 448 | 本 KV ABI 饱和 |
| -512 或 -Inf | 0xfe | -448 | 本 KV ABI 饱和 |
| NaN | 0x7f | NaN | 本 KV ABI canonical NaN |

已有 [`__dlc_fp8e4m3fn_from_fp32`](/Users/qingluo/fp8/DLC_Custom_Kernel/dlc_kernels/convert_element_type.h:69) 可复用 RNE 编码部分，但它不是 SATFINITE 转换，超范围会生成 NaN。必须在调用前按上述规则处理 NaN、clamp 和符号，并统一 NaN 码。

## 4 反量化的具体计算

先取 `S/E/M`。正常有限码可以直接构造 FP32 bits：

```text
fp32_bits = (S << 31) | ((E + 120) << 23) | (M << 20)
# 120 = FP32_bias(127) - E4M3FN_bias(7)
d32 = bitcast_float32(fp32_bits)
```

E=0 时必须另行处理：M=0 直接构造带符号零；M>0 的幅值为 `float32(M)*2^-9`。E=15/M=7 另行返回 NaN，不走 normal 构造。然后执行 `z32=RN32(d32*scale32)`，最后转 BF16。

BF16 RNE 的位级 reference 如下，所有加法按 uint32 处理：

```text
if isnan(z32):
    bf16_bits = 0x7fc0
else:
    u = bitcast_uint32(z32)
    bias = 0x7fff + ((u >> 16) & 1)
    bf16_bits = (u + bias) >> 16
```

例如 `d32=1`、`scale32=1.00390625` 时，乘积的 FP32 bits 为 0x3f808000，正好位于两个 BF16 值中间，RNE 输出 1.0、bits=0x3f80。若乘积为 1.01171875（0x3f818000），则输出 1.015625、bits=0x3f82。

现有 [`dequant_fp8e4m3fn_to_f32`](/Users/qingluo/fp8/DLC_Custom_Kernel/dlc_kernels/quantization/fp8_g128/dequant_fp8_g128_bf16_common.h:21) 可以复用解码数学；不能直接复用其 group128 main 的 UE8M0 scale 加载方式。BF16 reference 来自 [BFloat16.h](/Users/qingluo/fp8/pytorch/c10/util/BFloat16.h:66)。DLC `float_to_bfloat16` intrinsic 必须通过这些中点用例确认符合 RNE；不能仅凭函数名认定已验证。

## 5 一个有损量化例子

```text
BF16 K 输入 = 0.1640625
sK = 0.125

FP32 除 scale：0.1640625 / 0.125 = 1.3125
FP8 相邻值：1.25（0x3a）和 1.375（0x3b）
1.3125 是中点，RNE 选择 1.25，即 0x3a

读出解码：1.25
FP32 乘 scale：1.25 * 0.125 = 0.15625
BF16 计算值：0.15625，bits=0x3e20
```

原始 0.1640625 不会因“读取后用 BF16 算”而恢复。这里的误差来自 FP8 编码舍入，属于量化误差；相同 cache 的 attention 正确性测试应以 0.15625 作为该 K 元素的计算输入。

## 6 双 token 512 B tile 的精确布局

```text
cache.dtype = torch.float8_e4m3fn
cache.shape = [N, Hkv, 128, 512]

P=256 个 token/page，D=128 个元素/head
128 个 pair/page，每 pair 恰好 512 B
```

对物理 block b、KV head h、页内 pair j，tensor 的逻辑末维为：

```text
cache[b,h,j,  0:128] = K_fp8[2j,   0:128]
cache[b,h,j,128:256] = V_fp8[2j,   0:128]
cache[b,h,j,256:384] = K_fp8[2j+1, 0:128]
cache[b,h,j,384:512] = V_fp8[2j+1, 0:128]
```

这些是四个逻辑 plane。DLC HBM 中，每个 d 对应一个 32-bit word，按低位到高位为：

```text
bit 31..24        bit 23..16        bit 15..8         bit 7..0
V_odd_code[d]     K_odd_code[d]     V_even_code[d]    K_even_code[d]

word[d] = uint32(K_even_code[d])
        | (uint32(V_even_code[d]) << 8)
        | (uint32(K_odd_code[d])  << 16)
        | (uint32(V_odd_code[d])  << 24)
```

转换到足够宽的无符号整数后再移位，避免 signed char 扩展。一个 tile 是 128 个这样的 word。实际 HBM byte 顺序为 `K0[d],V0[d],K1[d],V1[d]`，不是整段 K0、整段 V0、整段 K1、整段 V1。CPU tensor 逻辑 plane 与物理 dump 要分别比较。[已有 DLC byte packing](/Users/qingluo/fp8/DLC_Custom_Kernel/dlc_kernels/kvcache/indexer_k_quant_and_cache_bf16.cpp:67)

## 7 完整数值和物理 word 例子

固定某一 head 的 d=0，`sK=0.125`、`sV=0.25`。同 pair 内两个 token 的其他 d 均为零：

| 元素 | 原始 BF16 数值 | scale | 除 scale 后 FP8 数值 | FP8 code | 恢复 BF16 bits |
|---|---:|---:|---:|---|---|
| K_even[0] | 0.125 | 0.125 | 1 | 0x38 | 0x3e00 |
| V_even[0] | -0.375 | 0.25 | -1.5 | 0xbc | 0xbec0 |
| K_odd[0] | 0.0625 | 0.125 | 0.5 | 0x30 | 0x3d80 |
| V_odd[0] | 0.5 | 0.25 | 2 | 0x40 | 0x3f00 |

```text
FP8 物理 word[0] = 0x4030bc38
其余 word[1..127] = 0x00000000

HBM 从低地址起的前四个 byte：38 bc 30 40
```

reader 解码并分别乘 K/V scale 后，按现有 BF16 计算布局重新组合：

```text
bf16_even_word[d] = K_even_bf16_bits[d] | (V_even_bf16_bits[d] << 16)
bf16_odd_word[d]  = K_odd_bf16_bits[d]  | (V_odd_bf16_bits[d]  << 16)

本例 bf16_even_word[0] = 0xbec03e00
本例 bf16_odd_word[0]  = 0x3f003d80
```

一个 512 B FP8 tile 在片上展开为 1024 B BF16 K/V。展开后的 tile 只在片上参与 BF16 QK/PV，不形成第二份 HBM BF16 cache。现有 BF16 merged 格式确实是低 16 bit K、高 16 bit V，见 [reshape_and_cache_kv_bf16.cpp](/Users/qingluo/fp8/DLC_Custom_Kernel/dlc_kernels/kvcache/reshape_and_cache_kv_bf16.cpp:3)。

## 8 分页寻址与读写步骤

写入地址按 slot mapping 计算：

```text
b = slot // 256
t = slot % 256
j = t // 2
r = t & 1
tile_id = ((b * Hkv + h) * 128 + j)

K_byte_offset = tile_id * 512 + 4*d + 2*r
V_byte_offset = K_byte_offset + 1
```

每个 head/page 为 65536 B，block stride 为 `Hkv*65536` B。DLC 的 `tensor` 地址以 128 B 为单位，所以 tile 指针是 `cache_base + tile_id*4`；`cache_base` 已经是 128 B 单位时不能再次除 128。DMA 一个 tile 的长度参数为 128 个 4 B word。大偏移使用足够宽的中间整数。

reader 根据 `block_table[request, logical_token//256]` 找物理 block；不能假设同一请求的 blocks 连续。Q/KV token、scale 和 causal 语义沿用主设计。

writer 的单 token 更新：

```text
1. BF16 K/V 扩展为 FP32，除各自 scale，按第 3 节得到 Kcode/Vcode。
2. DMA 整个旧 tile，等待完成。
3. pair[d] = (Kcode[d] & 0xff) | ((Vcode[d] & 0xff) << 8)
4. even: new_word[d] = (old_word[d] & 0xffff0000) | pair[d]
   odd:  new_word[d] = (old_word[d] & 0x0000ffff) | (pair[d] << 16)
5. DMA 整个 512 B tile 写回，等待完成。
```

第一版按 `h%2` 分配唯一 XYS owner，各 owner 顺序处理输入 slot。不能把同 pair 的 even/odd token 分给两个并发 writer 做完整 tile RMW，否则会覆盖对方更新。负 slot 不写，共享只读页先 COW，再更新独占目标页。

reader 的微步骤为：DMA tile → 拆四个 code plane → 对有效 token 解码为 FP32 → K/V 分别乘 scale → BF16 RNE → 合并成 BF16 K/V word → attention。序列尾部的无效另一 half 即使被物理 DMA 带入，也必须在 decode/scale/PV 前屏蔽成安全零，不能通过 `0*NaN` 消除；score 对无效位置置负无穷。

## 9 可以复用的原语与验证要求

| 位置 | 可以复用 | 必须改接或验证 |
|---|---|---|
| [FP32→E4M3FN helper](/Users/qingluo/fp8/DLC_Custom_Kernel/dlc_kernels/convert_element_type.h:69) | normal/subnormal 的 RNE 编码 | KV scale、外围 clamp、canonical NaN |
| [四 plane pack](/Users/qingluo/fp8/DLC_Custom_Kernel/dlc_kernels/convert_element_type.h:501) | 四个 byte 合成 lane word 的方式 | 输入 plane 顺序必须为 K_even/V_even/K_odd/V_odd |
| [E4M3FN decode](/Users/qingluo/fp8/DLC_Custom_Kernel/dlc_kernels/quantization/fp8_g128/dequant_fp8_g128_bf16_common.h:21) | E/S/M 解码公式 | 固定每层 FP32 scale 替换 UE8M0 group scale |
| [现有 BF16 转换调用](/Users/qingluo/fp8/DLC_Custom_Kernel/dlc_kernels/bf16.h:120) | 片上 FP32→BF16 packing 框架 | 用中点用例核对 intrinsic 的 RNE 行为 |

最小数值验收包括：254 个有限 FP8 编码在 scale=1 下的 decode/encode 往返（含 ±0）、两种 NaN 的识别、上述舍入中点、sK≠sV 且非 1、量化溢出饱和、只更新半 tile 的邻字节保护、逻辑 plane 和 HBM word 的一致性。reference 的编解码应独立于被测 kernel。

本文数值例子和有限码往返已通过主机侧枚举计算核对；这不替代 DLC intrinsic、DMA 或新算子的真机验收。代码实现仍按主设计中的工作包与依赖推进。
