# [Operator Request] 标准 MXFP8 E4M3 → BF16 反量化：G32 E8M0 与紧凑 scale tile

## 需求类型

新增算子。

## 优先级

P1。

## 目标版本与期望时间

FP8 KV cache 算子准备里程碑；与建议 PyTorch 接口 `torch.dlc_quant_mxfp8_e4m3`、kernel `custom_quant_mxfp8_e4m3` 配对交付，具体时间由算子组与接入方排期，当前无承诺日期。

## 算子名称

- 建议 PyTorch 接口：`torch.dlc_dequant_mxfp8_e4m3`。
- 建议 kernel 名称：`custom_dequant_mxfp8_e4m3_bf16`。
- 本 issue 对应一个反量化 kernel；group32固定于接口，不包含量化kernel或group128格式支持。

## 背景与功能说明

现有旧入口反量化只支持 group128，scale 使用逐 token 行布局，且 code255 的旧行为是 FP32 +Inf，不能直接作为标准 MXFP8 解码。
新增 group32、E4M3FN values 与标准 E8M0 scale 的反量化，读取与量化端一致的紧凑 scale tile，恢复普通 BF16 Tensor。
values 与 scale 是两个 Tensor；MX 数值编码按 OCP 标准，物理布局采用 TYD 本项目 ABI，不要求复现 NVIDIA 布局。
本批仅支持全 Tensor 转换；cache scatter/gather、逐 token append、跨请求并发更新 tile、attention 融合、values 区重排、E5M2 与模型质量验收均不在本 issue 范围。
现有旧 G128 接口/ABI 保持不变，不能静默切换其 group size 或 code255 语义。自定义group128即使采用标准E8M0的NaN语义，也不成为标准MXFP8；本接口固定group32。

## 期望 API 与函数签名

```python
def dlc_dequant_mxfp8_e4m3(
    input: torch.Tensor,
    scale_tiles: torch.Tensor,
    output: torch.Tensor,
) -> None:
    ...
```

建议 C++ 签名：`void dlc_dequant_mxfp8_e4m3(const Tensor& input, const Tensor& scale_tiles, Tensor& output)`。
调用方预分配 output，返回 `None`；schema 必须声明 output mutation，两个输入只读。固定 group32，无 `group_size` 或 `saturate` 参数；量化端的 SAT/OVF 已体现在输入编码中。

## 参数与返回值

- `input`：`torch.Tensor`，待恢复的 FP8 values；只读。
- `scale_tiles`：`torch.Tensor`，对应各组的 scale 编码；只读。
- `output`：`torch.Tensor`，调用方提供的 BF16 恢复缓冲区；写入。
- 所有参数无默认值，无其他属性；返回值为 `None`。dtype、shape、layout 和 device 约束集中见下文。

## 功能语义

`T` 表示token数量，`D` 表示每token的有效元素数；t、d、g分别为从0开始的token、元素和量化组索引。`ceil` 表示向上取整，`//` 表示整数向下除。

沿 D 维每 32 个元素共享 scale；尾 group 仅处理实际存在的元素。采用 `TYD_SCALE_T128_G4_V1`，values和scale是两个独立Tensor。定义 `G=ceil(D/32),Q=ceil(G/4),NT=ceil(T/128),N=NT*Q`，scale的二维逻辑形状为 `[N,512]`；T或D为0时N=0：

```text
g    = d // 32
tile = (t // 128) * Q + g // 4
k    = (t % 128) * 4 + g % 4
code = scale_tiles[tile, k]
physical_byte_offset = tile * 512 + 4 * (k % 128) + k // 128
```

`scale_tiles` 的每行逻辑序列为 128 个 token 各 4 个 group；物理偏移包含 TYD 原生 uint8 字节交织，不是 NVIDIA swizzle，不得重复重排。
对每个有效 `(t,d)` 执行以下语义；code255 必须优先判断，不能简单计算 `exp2(255-127)`：

```python
if code == 255:
    y = NaN                         # 整个有效 group，独立于 FP8 payload
else:
    s = float32(2.0 ** (int(code) - 127))
    q = decode_e4m3fn(input[t, d])   # 独立 raw-byte 解码，FP32
    y = float32(q * s)
output[t, d] = BF16_RNE(y)
```

E8M0 code `0…254` 表示 `2^(code-127)`，code0 是 `2^-127`，无零 scale、无 Inf scale；code255 表示 NaN scale，含该 scale 的整组均恢复 NaN，包括 q 为正负零的情况。
E4M3FN 的 `0x7f/0xff` 为 NaN，`0x00/0x80` 为正负零；有限 scale 下保留零符号。NaN 不承诺 payload 或符号，只比较 NaN mask。
FP32 中间计算按本项目 IEEE 约定处理：乘法溢出得到带符号 Inf；MX 标准对超过 FP32 范围的结果允许实现自定，本项目不把该策略表述为标准唯一要求。
最终 BF16 使用 RNE，溢出为带符号 Inf，保留可表示的 BF16 subnormal，不允许 FTZ；下溢舍入成零时保留符号。
无效 scale 槽一律不参与解码，padding 即使被填入 poison/255 也不得污染有效输出；values 行内 padding 同样忽略。
`T=0` 或 `D=0` 时，scale 必须为 `[0,512]`，完成元信息校验后 no-op、返回 `None`。不广播、不修改 input 或 scale，不产生随机结果。
三个 Tensor 不得共享 storage；禁止非零 offset、非 contiguous 或重叠 view。使用明确新入口识别 ABI，不得依据 shape 自动猜测 legacy/tiled 布局。
双 XYS 可共享只读 scale tile，但输出 DMA 范围必须独占，不能互相覆盖。调用期间其他 stream/请求不得修改输入；当前 stream 上按顺序异步执行，不做 host 数值扫描、隐式同步或 CPU fallback。

## 支持范围与约束

| 参数 | dtype | 逻辑 shape | 存储合同 |
|---|---|---|---|
| `input` | `float8_e4m3fn` | `[T,D]` | 当前普通 TYD FP8 Tensor，保留现有行 padding |
| `scale_tiles` | uint8 | `[NT*Q,512]`；任一维为零时 `[0,512]` | `TYD_SCALE_T128_G4_V1` |
| `output` | BF16 | `[T,D]` | 普通 TYD BF16 Tensor |

三个 Tensor 均为同一 TYD 卡上的二维 contiguous Tensor，`storage_offset=0`，storage 独立，形状严格匹配；错误 dtype/rank/shape/device/stride/offset/alias 在 launch 前明确拒绝。
支持动态 T、D，必须覆盖 `T=0…4096,D=0…8192` 的适用组合、尾 token tile 与尾 group。实际更大边界由显存、VMEM 分块及内核索引决定；先用 int64 检查乘法/对齐，超出 int32/VMEM 实现边界时早拒，不得溢出或静默截断。
支持全部 256 种 FP8 code 与全部 256 种 scale code；不能仅支持配套量化通常产生的码值子集，也不能将 subnormal 统一清零。
支持当前 DLC stream、非默认 stream 及 graph capture/replay；单卡独立执行、无 collective。不支持其他芯片、其他 FP8 编码和 group128 输入。
本需求压缩 scale 存储，未改变 values 逐行 padding；短 T 和非整 tile 容量必须按本文公式计算，不承诺所有 shape 均达到 48.4375%。

## 精度要求

下面的 CPU PyTorch reference 内嵌全部bitfield解码和tile寻址，不依赖项目文件，可直接复制运行。输入是逻辑CPU uint8矩阵；如果从DLC取回数据，应同dtype拷到CPU得到逻辑字节，不人为重复weave。独立解码不依赖量化producer，也不以往返抵消错误。

```python
import torch


def reference_dequant_mxfp8_e4m3(bits, scale_tiles):
    """Logical CPU uint8 bits[T,D], tiles[N,512] -> CPU BF16[T,D]."""
    for name, tensor in (("bits", bits), ("scale_tiles", scale_tiles)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(name + " must be a Tensor")
        if tensor.device.type != "cpu" or tensor.dtype != torch.uint8 or tensor.ndim != 2:
            raise ValueError(name + " must be a CPU uint8 matrix")
    T, D = bits.shape
    G = (D + 31) // 32
    Q = (G + 3) // 4
    N = ((T + 127) // 128) * Q
    if tuple(scale_tiles.shape) != (N, 512):
        raise ValueError("scale_tiles must have shape " + str((N, 512)))
    if T == 0 or D == 0:
        return torch.empty((T, D), dtype=torch.bfloat16)
    t = torch.arange(T)[:, None]
    g = torch.arange(G)[None, :]
    codes = scale_tiles[(t // 128) * Q + g // 4, (t % 128) * 4 + g % 4]
    scale_nan = codes == 255
    # Mask first; never evaluate 2**128 and mistake code255 for +Inf.
    safe_codes = torch.where(scale_nan, 127, codes)
    scales = torch.exp2(safe_codes.float() - 127.0)
    raw = bits.to(torch.int32)
    magnitude = raw & 127
    exponent = magnitude >> 3
    mantissa = magnitude & 7
    normal = (1.0 + mantissa.float() / 8.0) * torch.exp2(exponent.float() - 7.0)
    subnormal = mantissa.float() * (2.0 ** -9)
    values = torch.where(exponent == 0, subnormal, normal)
    values = torch.where((raw & 128) != 0, -values, values)
    values = torch.where(magnitude == 127, float("nan"), values)
    groups = torch.arange(D) // 32
    restored = values * scales[:, groups]  # FP32, signed overflow -> Inf
    restored = torch.where(scale_nan[:, groups], float("nan"), restored)
    return restored.to(torch.bfloat16)  # RNE, preserve signed zero/subnormal


# Fixed E8M0 semantics and minimal BF16 subnormal; padding is deliberately 255.
bits = torch.zeros((1, 64), dtype=torch.uint8)
bits[0, :4] = torch.tensor([0x00, 0x80, 0x08, 0x88], dtype=torch.uint8)
scales = torch.full((1, 512), 255, dtype=torch.uint8)
scales[0, 0] = 0
out = reference_dequant_mxfp8_e4m3(bits, scales)
assert out[0, :4].view(torch.int16).tolist() == [0, -32768, 1, -32767]
assert bool(torch.isnan(out[0, 32:]).all())
# Exhaust all 256 FP8 bytes x all 256 E8M0 codes. Native CPU decode is an
# independent check of the bitfield decoder above, not the implementation.
bits = torch.arange(256, dtype=torch.uint8).repeat(256, 1)
scales = torch.zeros((4, 512), dtype=torch.uint8)  # T=256,D=256 => Q=2,N=4
for t in range(256):
    for g in range(8):
        scales[(t // 128) * 2 + g // 4, (t % 128) * 4 + g % 4] = t
out = reference_dequant_mxfp8_e4m3(bits, scales)
native = bits.view(torch.float8_e4m3fn).float()
expected = (native * torch.exp2(torch.arange(256).clamp_max(254).float()
                              .sub(127))[:, None]).to(torch.bfloat16)
expected[255] = float("nan")
assert torch.equal(torch.isnan(out), torch.isnan(expected))
valid = ~torch.isnan(expected)
assert torch.equal(out.view(torch.int16)[valid], expected.view(torch.int16)[valid])
print("PASS: CPU MXFP8 dequant reference goldens; no DLC execution")
```

有限结果 BF16 raw bits 必须逐位一致，`atol=0,rtol=0`；正负零符号位一致，Inf 位置与符号一致，NaN 比较 mask，不比较 payload。中间计算 FP32、最终 BF16 RNE，不引入额外低精度中间量。
穷举 **256 FP8 code × 256 scale code**，在多 token/多 tile 中交错摆放，覆盖 `scale255 + q0/q负零/q有限/qNaN → NaN`，区分新G128/MXFP8的标准E8M0语义与旧G128 legacy code255行为。
最小 BF16 subnormal 必测：`q=0x08,scale=0 → 2^-133`，负号对称为 `q=0x88 → -2^-133`；补恰好半最小 subnormal、RNE tie、舍入至零、FP32/BF16 溢出边界，拒绝 FTZ。
固定 seed `20261015`；roundtrip时令x为原始BF16、y为恢复BF16，另报 `max_abs=max(abs(y-x))`、`RMSE=sqrt(mean((y-x)^2))`、`NRMSE=sqrt(sum((y-x)^2)/sum(x^2))`，零分母单独标记，并报平均/最大逐元素相对误差（排除x=0、说明near-zero）、新增零和NaN/Inf比例。SAT/OVF非有限结果先严格核对golden的mask并单独计数，有限子集指标同时报告覆盖数；不能过滤NaN后掩盖解码错误。有损量化误差仅是观测项，不能替代本反量化BF16 bits与NaN mask的一致性硬要求。
六类有限输入的可复现生成定义如下：每个 shape/分布重新创建 CPU `torch.Generator().manual_seed(20261015)`，先用 FP32 生成，最后转换 BF16。令 `j=t*D+d`、`g=d//32`、`sign=(-1)^j`：signed_zero 为按 j 交替的 `+0/-0`；constant 为按 j 交替的 `+1/-1`；normal 为 `N(0,1)`；tiny 为 `sign*2^(-126+(j%107))`；group_scales 为 `N(0,1)*2^((g%5)*6-12)`；outliers 为 `N(0,0.01²)`，并把每组首个有效元素覆盖为 `(-1)^(t+g)*16`。BF16最小subnormal、NaN/Inf及舍入边界另用下列固定golden和定向用例覆盖，不能仅靠随机分布。
配对量化 `torch.dlc_quant_mxfp8_e4m3` 的SAT/OVF输出均独立解码验收。其固定数值策略是：finite amax>0时 `e=clamp(floor(log2(amax))-8,-127,127)`，全零或没有finite非零元素时e=0；NaN/Inf排除amax，NaN输出canonical0x7f，SAT将Inf/溢出饱和为带符号448，OVF将Inf/舍入后溢出转NaN。该producer策略不限制本反量化接受全部合法编码，反量化没有SAT/OVF开关，已有溢出结果直接由FP8编码决定。

## 测试范围

- [x] 基本功能与 reference 对齐
- [x] 多 dtype 精度（FP8/uint8 输入、BF16 输出；其他 dtype 拒绝）
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [x] 非 contiguous、特殊 stride 或 layout（错误元信息拒绝、tile 布局验证）
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式（不需要 collective，不在本 issue 范围）
- [x] 长时间稳定性
- [x] 性能基准与性能回归

主矩阵：`T∈{1,17,127,128,129,255,256,257}` × `D∈{32,64,128,129,512,576,1024}`，配套上述 6 种分布；补 D=1/31/33/127/511/513/8192 和 T=4096 边界，覆盖 g=3/4、t=127/128、多个 Q、多 tile 和双 XYS。
空输入 `[0,576]`、`[2,0]` 必测；scale 形状必须 `[0,512]`。非法 shape、dtype、CPU/异卡、offset、stride、storage alias 逐项隔离验证，含元信息计算溢出早拒。
分别验证量化、反量化，不只验证 roundtrip；对任意 FP8/scale 编码做独立解码测试，不能让配对实现的相同错误相互抵消。
将无效 scale 槽和 values padding 填 poison 后结果不变；output 预填 poison 确认全部有效元素覆盖，input/scale 前后逻辑字节不变。
验证非默认 stream 异步依赖、graph capture/replay、双 XYS 输出不互覆盖；连续至少 1000 次、交替 shape 检查稳定性，不依赖 host 数值扫描同步。
性能基准记录独立 dequant 及 quant→dequant 的端到端时延、cycles、带宽和峰值 workspace，覆盖 T=1/128/129/4096、D=128/512/576；结果待实现后测量，不编造既有性能、不预设未经评估的硬阈值。

## 验证环境与依赖版本

验收使用 TYD 真机，编译期和运行期均为 `DLC_TPU_VERSION=2`；调用链为 PyTorch DLC fork → DLCSynapse/dlc_runtime → DLC_Custom_Kernel，kernel 由 DLC LLVM 工具链构建，并依赖匹配的 DLCsim CRT。使用工作区 Python 虚拟环境，确认新 PyTorch 入口已注册、实际 dispatch 到本 kernel，并加载本次交付的库。
日志必须记录：单卡 device index、真机/模拟模式、Python/PyTorch版本、PyTorch/DLC_Custom_Kernel/DLCSynapse/DLCsim/LLVM 的完整 Git SHA、构建选项、实际加载 `.so` 路径与 SHA256、driver 与 RTABI 版本，以及 `DLC_TPU_VERSION`、`DLC_SYN_USE_SIM`、`DLC_SYN_BLOCKING`、`LD_LIBRARY_PATH`、`CUSTOM_KERNEL_PATH`。撰写时环境记录 driver 为 `Driver Version_v23.0.1`、API 23、RTABI 3.0.0，仅作环境背景；交付时必须记录实际值，不能据此声称新算子已通过。
CPU reference、模拟器与真机结果分开标记；历史 group128 的 120 组往返和 13 项接口检查不能作为本 MXFP8 验收证据。vLLM/vllm-cl、W4A16 和模型端到端质量不参与本 issue 的必测链路，graph capture 必须使用实际 PyTorch/runtime 组合验证。

容量按当前 TYD values 布局完整核算。定义 `ALIGN_n(x)=ceil(x/n)*n`，FP8 values 为 `T*ALIGN_512(D)` B，scale 为 `N*512` B；压缩态合计 `T*ALIGN_512(D)+N*512` B，对应 BF16 为 `2*T*ALIGN_256(D)` B。空 Tensor 的有效数据容量为零，allocator 元信息/最小分配粒度另报。
`T=128,D=512`：FP8 为 65536 B、scale 为 2048 B，合计 **67584 B=66 KiB**，BF16 为 131072 B，节省 **48.4375%**。`T=128,D=128`：FP8 65536 B+scale512 B，高于 BF16 65536 B；values 逐行 padding 仍在。`T=1,D=512` 的 scale 仍需四个 tile、2048 B，压缩态2560 B高于BF16 1024 B。小 T、尾 tile 和窄 D 不承诺节省，跨 token 打包 values 不在本需求中。以上为布局容量公式，真实 HBM/allocator 分配需另测，不能用 `numel*element_size` 或 `memory_reserved` 代替。

## 验收标准

- [ ] 单 kernel、Python/C++ API、output mutation schema、参数校验、空 Tensor 和错误行为符合合同，旧 G128 接口/ABI 回归通过。
- [ ] 全部 raw code × scale code、正负零、NaN mask、BF16 subnormal/RNE/溢出、tail 和 poison 测试与独立 CPU reference 一致。
- [ ] 与 `torch.dlc_quant_mxfp8_e4m3` 的 SAT/OVF 量化完成独立精度与往返测试；当前 stream、异步非默认 stream、graph replay、双 XYS 和稳定性测试通过。
- [ ] `T=128,D=512` 时持久化 FP8 values 为 **65536 B**，scale 为 **2048 B**，总计 **67584 B**；相对 BF16 **131072 B** 节省 **48.4375%**，无旧逐行 scale 持久副本。
- [ ] 独立恢复的 BF16 output 和 workspace 另外统计峰值；显存分配证据核对真实 HBM allocation/allocator 粒度，不使用 `numel*element_size` 或 `memory_reserved` 代替实际占用，不声称全量还原时峰值也减半。
- [ ] 提交可复现测试、reference、调用示例、精度/容量/性能报告及首个失败坐标；明确尚未完成 cache writer、attention 融合与模型质量验收。

## Reference、日志与附件

- 本文“精度要求”中的完整 CPU reference、固定subnormal/NaN golden与256×256穷举检查可直接复制运行；不需要读取其他项目文件。本需求撰写时尚无本新kernel的设备日志或profile附件。
- 标准依据为 Open Compute Project《Microscaling Formats (MX) Specification》Version 1.0：§5.1允许values与scale分开或相邻存储、物理排列由实现选择；§5.2的表1规定MXFP8块长32、§5.2.1允许实现格式子集；§5.3.1规定元素编码；§5.4.1规定E8M0 code0…254代表 `2^(code-127)`、code255为NaN，§5.1规定NaN scale传播至整个有效block，code0不是零scale、标准没有Inf scale。上述规则已完整落实于本文解码合同。
- 元素格式依据 Open Compute Project《OCP 8-bit Floating Point Specification (OFP8)》Revision 1.0，2023-12-01勘误版：E4M3最大有限448，最小正常 `2^-6`，最小次正规 `2^-9`，0x00/0x80为正负零，0x7f/0xff为NaN，无Inf；本文明确FP32乘法后BF16 RNE，并为超FP32范围的乘法选择带符号Inf策略。
- 配对接口为 `torch.dlc_quant_mxfp8_e4m3(input, output, scale_tiles, saturate=True)`，kernel建议名 `custom_quant_mxfp8_e4m3`；完整Tensor输出使用本文相同tile映射。反量化仍必须接受该producer通常不会生成的scale255及全部其他合法编码。历史group128只作兼容回归，不能替代MXFP8验证。
- 要求交付：可复制测试命令及reference、全部环境指纹、各项通过/失败数、首个差异 `(t,d)`/`(tile,k)`、BF16 bits及NaN mask对比、容量/allocator证据、workspace与profile。日志区分CPU、模拟器与真机；不能引用未附上的日志或把未来验收写成既有通过结果。
