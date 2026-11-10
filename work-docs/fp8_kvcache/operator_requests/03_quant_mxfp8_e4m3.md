# [Operator Request] 新增 MXFP8(E4M3) group32 量化及紧凑 E8M0 scale 输出

## 需求类型

新增算子。

## 优先级

P1。

## 目标版本与期望时间

FP8 KV cache 算子准备里程碑；与建议 PyTorch 接口 `torch.dlc_dequant_mxfp8_e4m3`、kernel `custom_dequant_mxfp8_e4m3_bf16` 配对交付。具体日期由算子组与接入方排期，具体交付版本与日期待排期。

## 算子名称

- 建议 PyTorch 接口：`torch.dlc_quant_mxfp8_e4m3`。
- 建议 kernel 名：`custom_quant_mxfp8_e4m3`。
- 本 issue 对应一个量化 kernel；SAT/OVF 为该 kernel 的显式属性，不拆成不同数值格式，也不包含反量化 kernel。

## 背景与功能说明

现有旧入口量化仅支持 group128 的 E4M3FN + UE8M0，并采用 `ceil(log2(max(amax/448,1e-10)))`；不能直接视为 OCP MXFP8 group32。
本需求新增 OCP MXFP8 的 E4M3 格式子集：32 个元素共享一个标准 E8M0 scale，并使用本文完整定义的紧凑 scale tile 消除逐 token scale 行 padding。
数据与 scale 分为两个 Tensor，FP8 values 维持普通 TYD Tensor 布局。保留旧 G128 入口、返回值、布局及行为；本接口不按 shape 猜测或自动兼容 legacy scale。
自定义 group128 即使采用相同数值 recipe、SAT/OVF 和 E8M0 语义，也不成为标准 MXFP8；本接口固定 group32。
首批沿 D 维分组，BF16 输入、FP8 存储，供后续恢复 BF16 计算。E5M2、FP8 计算、KV cache 页/slot/scatter/逐 token append/共享 tile 并发更新、attention 融合、FP8 values 的跨 token 打包及端到端模型质量验收不在本 issue 范围。

## 期望 API 与函数签名

```python
def dlc_quant_mxfp8_e4m3(
    input: torch.Tensor,
    output: torch.Tensor,
    scale_tiles: torch.Tensor,
    saturate: bool = True,
) -> None:
    ...
```

建议 C++：`void dlc_quant_mxfp8_e4m3(const Tensor& input, Tensor& output, Tensor& scale_tiles, bool saturate=true)`。
调用方预分配输出；两个输出分别登记 mutation alias 集合，返回 `None`。名称固定 group32，不暴露可变 `group_size`。`saturate=True` 为 SAT，`False` 为 OVF。

## 参数与返回值

- `input`：`torch.Tensor`，待量化输入，只读。
- `output`：`torch.Tensor`，调用方提供的 FP8 编码结果缓冲区，写入。
- `scale_tiles`：`torch.Tensor`，调用方提供的 scale 编码缓冲区，写入。
- `saturate`：Python `bool`，默认 `True`，选择 FP8 结果发生溢出时的 SAT/OVF 行为。
- 返回 `None`。其余参数无默认值；dtype/shape/layout/device 集中见“支持范围与约束”。

## 功能语义

`T` 表示token数量，`D` 表示每token的有效元素数；t、d、g分别为从0开始的token、元素和量化组索引。`ceil` 表示向上取整，`//` 表示整数向下除。

本项目对有限块选择 OCP MX v1.0 §6.3 推荐支持的转换方法，E4M3 中最大可表示的 2 的幂为 `2^8=256`。对每个有效 token、沿 D 每 32 个有效输入元素为一组：

```python
x = FP32(input[t, 32*g : min(32*(g+1), D)])
# 以下 finite 归约及全零约定是本项目固定策略。
a = max((abs(x_i) for x_i in x if isfinite(x_i)), default=0.0)
if a == 0:
    e = 0
else:
    e = clamp(floor(log2(a)) - 8, -127, 127)
code = uint8(e + 127)
z = FP32(ldexp(x, -e))
q = E4M3_RNE(z, overflow_mode=('SAT' if saturate else 'OVF'))
output[t, group] = q
scale_tiles[tile, k] = code
```

指数选择必须准确识别 2 的幂和相邻 BF16 值，优先精确指数提取/`frexp`；不可用近似低精度 log2 将边界归错组。归约/归一化为 FP32，最终 FP8 RNE；不得使用 BF16 scale、`1e-10` 下限或旧 `ceil(log2(amax/448))` 代替本合同。

- 全零或无 finite 非零元素的组使用 code127、scale1。原始 ±0 的编码符号保留。
- NaN/Inf 不参与 amax；NaN 元素输出 canonical E4M3 NaN `0x7f`。
- SAT：±Inf 以及舍入后超出 E4M3 有限范围的元素转为带符号448。
- OVF：±Inf 以及舍入后溢出的元素转为 canonical NaN；不强制整组污染其他元素。
- FP8 正常值、次正规值、下溢至带符号零均按 RNE；溢出判断要符合 OFP8 的舍入后语义，尤其464中点不能误判。
- NaN/Inf 的 amax 排除规则和全零 scale 选择为本项目确定性扩展，不称为 OCP 唯一算法。其他 scale 选取算法需另立数值 recipe，不能在本接口中自动切换。

采用 `TYD_SCALE_T128_G4_V1`，values 和 scale 是两个独立 Tensor。定义 `G=ceil(D/32),Q=ceil(G/4),NT=ceil(T/128),N=NT*Q`，scale 的二维逻辑形状固定为 `[N,512]`；T或D为0时N=0：

```text
tile = (t // 128) * Q + g // 4
k    = (t % 128) * 4 + g % 4
scale_tiles[tile,k] = code
physical_byte_offset = tile*512 + 4*(k%128) + k//128
```

最后一行才是 TYD 原生字节交织后的物理偏移；CPU reference 只按逻辑 `(tile,k)` 写入，不重复 weave。tile 内逻辑行序不是 NVIDIA swizzle。量化的所有无效 token/scale 列槽清零，输入 padding 不参与 amax；最后不足32的 group 只处理有效元素。
`T=0` 或 `D=0` 时，scale `[0,512]`，校验后 no-op。输入不修改，输出与所有其他 Tensor 均不共享 storage。
调用方独占本次完整输出 tile；双 XYS 按完整128-token block及其所有通道tile分配唯一writer，或提供等效安全方案。VMEM 分批不得清零/覆盖先前 batch 的同一tile。不能产生旧逐行 scale 的常驻 HBM 副本，再靠 host 压缩。
本接口处理完整 Tensor，确定性执行，无随机数；当前 stream 异步执行，无 CPU fallback、无 host 数据扫描或隐式全设备同步。

## 支持范围与约束

| 参数 | dtype | 逻辑 shape | layout |
|---|---|---|---|
| `input` | BF16 | `[T,D]` | 普通 TYD BF16 |
| `output` | `float8_e4m3fn`，OCP E4M3 | `[T,D]` | 普通 TYD FP8，保留其 values padding |
| `scale_tiles` | uint8，承载标准 E8M0 code | `[NT*Q,512]`，T或D为0时 `[0,512]` | `TYD_SCALE_T128_G4_V1` |
| `saturate` | Python bool | 标量属性 | SAT/OVF |

首批动态 `0<=T<=4096,0<=D<=8192`，支持尾 token、尾 group；不要求 D 为32倍数，不做 broadcasting。大 shape 的 workspace 必须分块；int64先检查对齐/乘法，超过索引或设备容量的明确报错，不静默截断。
支持所有 BF16 编码类别：finite、±0、subnormal、±Inf、NaN；BF16 subnormal不得被无条件FTZ。最小非零BF16 `2^-133` 应能选择code0并编码为q=`2^-6`，恢复时仍非零。
本接口对非空有效组产生 scale code0…246（BF16有限amax决定上界，0为下界夹取），不生成255；这不减免配套反量化对全部合法E8M0码的支持。code255在标准中是NaN，不能沿用旧G128入口的Inf解释；新G128 packed同样采用NaN语义。
所有 Tensor 同一 TYD 卡、contiguous、storage_offset=0且互不alias；错误 dtype/rank/shape/device/stride/offset/alias 和非法属性在 launch 前拒绝。其他芯片、CPU/CUDA输入、E4M3FNUZ和E5M2不在本批。
支持非默认stream、graph capture/replay；输出tile生命周期及跨stream依赖由调用方保证。不要求分布式collective或跨卡scale同步。

## 精度要求

下面的 CPU PyTorch reference 完整定义本接口的数学结果和逻辑打包，可直接复制为一个 Python 程序运行；不依赖其他项目文件。环境需提供 CPU `torch.float8_e4m3fn` RNE 转换。reference 返回 uint8 FP8 编码与二维逻辑 scale tile，复制到 DLC 时按普通同 dtype 传输处理，不能再人为施加 TYD weave。

```python
import torch


def reference_quant_mxfp8_e4m3(src, saturate=True):
    """CPU BF16[T,D] -> logical uint8 bits[T,D], uint8 tiles[N,512]."""
    if not isinstance(src, torch.Tensor):
        raise TypeError("src must be a Tensor")
    if src.device.type != "cpu" or src.dtype != torch.bfloat16 or src.ndim != 2:
        raise ValueError("src must be CPU BF16[T,D]")
    if not isinstance(saturate, bool):
        raise TypeError("saturate must be bool")
    T, D = src.shape
    B = 32
    G = (D + B - 1) // B
    Q = (G + 3) // 4
    N = ((T + 127) // 128) * Q
    tiles = torch.zeros((N, 512), dtype=torch.uint8)
    if T == 0 or D == 0:
        return torch.empty((T, D), dtype=torch.uint8), tiles
    x = torch.nn.functional.pad(src.float(), (0, G * B - D)).reshape(T, G, B)
    finite_abs = torch.where(torch.isfinite(x), x.abs(), 0.0)
    amax = finite_abs.amax(dim=-1)
    _, power = torch.frexp(amax)  # amax=m*2**power, 0.5<=m<1 when nonzero
    e = torch.where(amax > 0, power - 1 - 8, 0).clamp(-127, 127)
    codes = (e + 127).to(torch.uint8)
    normalized = torch.ldexp(x, -e.unsqueeze(-1))
    if saturate:
        normalized = normalized.clamp(-448.0, 448.0)
    # CPU PyTorch E4M3FN conversion uses RNE; OVF preserves finite 464->448.
    bits = normalized.to(torch.float8_e4m3fn).view(torch.uint8)
    bits = torch.where(torch.isnan(normalized), 0x7f, bits)
    bits = torch.where((bits & 127) == 127, 0x7f, bits)
    bits = bits.reshape(T, G * B)[:, :D].contiguous()
    t = torch.arange(T)[:, None]
    g = torch.arange(G)[None, :]
    tiles[(t // 128) * Q + g // 4, (t % 128) * 4 + g % 4] = codes
    # These are logical bytes. CPU->DLC same-dtype copy applies native byte weave.
    return bits, tiles


# Handwritten goldens, independent of the implementation formula.
for value, sat_byte, ovf_byte in ((480.0, 0x7e, 0x7f),
                                 (464.0, 0x7e, 0x7e),
                                 (466.0, 0x7e, 0x7f)):
    x = torch.full((1, 32), value, dtype=torch.bfloat16)
    a, sa = reference_quant_mxfp8_e4m3(x, True)
    b, sb = reference_quant_mxfp8_e4m3(x, False)
    assert sa[0, 0].item() == 127 and torch.equal(sa, sb)
    assert bool((a == sat_byte).all()) and bool((b == ovf_byte).all())
x = torch.zeros((1, 32), dtype=torch.bfloat16)
x[0, :6] = torch.tensor([0., -0., 2.**-10, -2.**-10,
                         3*2.**-10, -3*2.**-10], dtype=torch.bfloat16)
x[0, -1] = 256.
a, s = reference_quant_mxfp8_e4m3(x)
assert a[0, :6].tolist() == [0, 128, 0, 128, 2, 130] and s[0, 0] == 127
x = torch.full((1, 32), 2.**-133, dtype=torch.bfloat16)
a, s = reference_quant_mxfp8_e4m3(x)
assert s[0, 0] == 0 and bool((a == 0x08).all())
print("PASS: CPU MXFP8 quant reference goldens; no DLC execution")
```

FP8有限编码逐字节一致、NaN统一为0x7f；所有有效scale code逐字节一致。`atol=0,rtol=0`，正负零符号位相同，不使用宽松float误差容忍掩盖编码错。固定 seed=20261015，记录样本数和shape。NaN/Inf混合块另测，不用于模型质量指标。
六类有限输入的可复现生成定义如下：每个 shape/分布重新创建 CPU `torch.Generator().manual_seed(20261015)`，先用 FP32 生成，最后转换 BF16。令 `j=t*D+d`、`g=d//32`、`sign=(-1)^j`：signed_zero 为按 j 交替的 `+0/-0`；constant 为按 j 交替的 `+1/-1`；normal 为 `N(0,1)`；tiny 为 `sign*2^(-126+(j%107))`；group_scales 为 `N(0,1)*2^((g%5)*6-12)`；outliers 为 `N(0,0.01²)`，并把每组首个有效元素覆盖为 `(-1)^(t+g)*16`。BF16最小subnormal、NaN/Inf及舍入边界另用下列固定golden和定向用例覆盖，不能仅靠随机分布。

定向golden必须包含：

- 组amax480：code127；SAT峰值448、OVF峰值NaN。与旧G128 code128明确区分。
- 组amax464：峰值RNE到448；466：SAT448、OVF NaN。两者在同一scale exponent0下验收。
- 由同组amax256固定scale1，检查 `2^-10` 舍入到±0、`3*2^-10` 舍入到±`2^-8`；遍历更多FP8中点及相邻BF16值。
- BF16最小subnormal `2^-133`：code0、q=0x08，配套恢复BF16原值；±0保符号；最大finite和2幂相邻值。
- NaN/Inf混合finite块、只有NaN/Inf的块、全零块，验证归约与SAT/OVF，重复执行一致。

量化损失另报 `max_abs=max(abs(y-x))`、`RMSE=sqrt(mean((y-x)^2))`、`NRMSE=sqrt(sum((y-x)^2)/sum(x^2))`，其中x为原始BF16、y为按本文格式恢复的BF16；零分母单独标记。另报平均/最大逐元素相对误差（排除原值0并说明near-zero）、新增零比例和NaN/Inf比例。SAT/OVF产生的非有限输出必须先按golden精确核对mask并单独计数，有限子集指标必须同时报告样本覆盖数，不能过滤NaN后掩盖编码错误。这些是观测项，精度硬门槛仍是与本文golden的bytes/codes一致，不预设真实KV质量结论。

## 测试范围

- [x] 基本功能与 reference 对齐
- [x] 多 dtype 精度（BF16/FP8/uint8接口合同；错误dtype拒绝）
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [x] 非 contiguous、特殊 stride 或 layout（错误输入拒绝，tile映射验证）
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式（本算子不含collective）
- [x] 长时间稳定性
- [x] 性能基准与性能回归

基础形状：`T∈{1,17,127,128,129,255,256,257}`，`D∈{32,64,128,129,512,576,1024}`；六种分布与SAT/OVF均测。补 `[0,576]`、`[2,0]`、D=1/31/33/127/511/513/8192、T=4096 的代表性组合，不要求全部极值笛卡尔积。
让相邻token/group的scale明显不同，检查 t127/128、g3/4、多个Q，避免重复内容掩盖错索引。scale输出预填poison后检查所有无效槽清零；输入padding毒化不影响归约。测试两XYS、VMEM多batch同tile、非默认stream和graph replay。
同一FP8 bytes、有效codes分别核对，再与配套dequant测试BF16结果；不能只跑roundtrip。至少1000次交替shape稳定性；旧G128 API回归仍通过。提交warmup/迭代数明确的时延、cycles、带宽、持久分配及峰值workspace报告，本批不硬编速度门限。

## 验证环境与依赖版本

验收使用 TYD 真机，编译期和运行期均为 `DLC_TPU_VERSION=2`；调用链为 PyTorch DLC fork → DLCSynapse/dlc_runtime → DLC_Custom_Kernel，kernel 由 DLC LLVM 工具链构建，并依赖匹配的 DLCsim CRT。使用工作区 Python 虚拟环境，确认新 PyTorch 入口已注册、实际 dispatch 到本 kernel，并加载本次交付的库。
日志必须记录：单卡 device index、真机/模拟模式、Python/PyTorch版本、PyTorch/DLC_Custom_Kernel/DLCSynapse/DLCsim/LLVM 的完整 Git SHA、构建选项、实际加载 `.so` 路径与 SHA256、driver 与 RTABI 版本，以及 `DLC_TPU_VERSION`、`DLC_SYN_USE_SIM`、`DLC_SYN_BLOCKING`、`LD_LIBRARY_PATH`、`CUSTOM_KERNEL_PATH`。撰写时环境记录 driver 为 `Driver Version_v23.0.1`、API 23、RTABI 3.0.0，仅作环境背景；交付时必须记录实际值，不能据此声称新算子已通过。
CPU reference、模拟器与真机结果分开标记；历史 group128 的 120 组往返和 13 项接口检查不能作为本 MXFP8 验收证据。vLLM/vllm-cl、W4A16 和模型端到端质量不参与本 issue 的必测链路，graph capture 必须使用实际 PyTorch/runtime 组合验证。

容量按当前 TYD values 布局完整核算。定义 `ALIGN_n(x)=ceil(x/n)*n`，FP8 values 为 `T*ALIGN_512(D)` B，scale 为 `N*512` B；压缩态合计 `T*ALIGN_512(D)+N*512` B，对应 BF16 为 `2*T*ALIGN_256(D)` B。空 Tensor 的有效数据容量为零，allocator 元信息/最小分配粒度另报。
`T=128,D=512`：FP8 为 65536 B、scale 为 2048 B，合计 **67584 B=66 KiB**，BF16 为 131072 B，节省 **48.4375%**。`T=128,D=128`：FP8 65536 B+scale512 B，高于 BF16 65536 B；values 逐行 padding 仍在。`T=1,D=512` 的 scale 仍需四个 tile、2048 B，压缩态2560 B高于BF16 1024 B。小 T、尾 tile 和窄 D 不承诺节省，跨 token 打包 values 不在本需求中。以上为布局容量公式，真实 HBM/allocator 分配需另测，不能用 `numel*element_size` 或 `memory_reserved` 代替。

## 验收标准

- [ ] 新API、mutation schema、SAT/OVF默认值、参数校验及空Tensor行为符合合同；旧G128入口/ABI不变。
- [ ] 指定数值、尾部、特殊值、FP8中点、BF16 subnormal全部通过；输出bytes和codes与golden一致，padding全部归零。
- [ ] 双XYS/VMEM分块无tile覆盖，stream/graph及重复稳定性测试通过；新量化直接产生packed scale，无常驻legacy scale副本或CPU fallback。
- [ ] T128、D512：FP8数据65536 B、scale2048 B，持久化合计67584 B=66 KiB，相对BF16 131072 B节省48.4375%；提供隔离分配的真实HBM/allocator证据。
- [ ] 不用numel或memory_reserved代替显存实测；持久输入和反量化output/workspace峰值分开，说明小T尾tile、D128数据padding的限制。
- [ ] 配套反量化、CPU reference、自动测试、示例和精度/容量/性能报告提交；明确这是E4M3子集，非E5M2或完整KV cache交付。

## Reference、日志与附件

- 本文“精度要求”内嵌的完整 CPU reference 和固定golden就是可复制的验收依据；交付测试不得依赖未随交付提供的文件。本需求撰写时尚无本新kernel的设备日志或profile附件。
- 标准依据为 Open Compute Project《Microscaling Formats (MX) Specification》Version 1.0：§5.1允许values与scale分开或相邻存储，具体物理布局由实现选择；§5.2的表1规定MXFP8块长32及E4M3/E5M2可选编码，§5.2.1允许实现格式子集，本接口只实现E4M3；§5.3.1规定元素格式；§5.4.1规定E8M0的code0…254为 `2^(code-127)`、code255为NaN，§5.1规定NaN scale传播至整个有效block；§6.3推荐支持基于块最大量级与元素格式最大2幂的scale转换方法，本接口固定采用 `floor(log2(finite_amax))-8` 并按本文规则处理非有限值与全零块，标准未强制所有实现只用这一种选scale算法。
- 元素编码与转换依据 Open Compute Project《OCP 8-bit Floating Point Specification (OFP8)》Revision 1.0，2023-12-01勘误版：E4M3最大有限数448、最小正常数 `2^-6`、最小次正规数 `2^-9`、无Inf编码、0x7f/0xff为NaN；RNE、SAT与OVF按照本文明确合同执行。
- 配对接口为 `torch.dlc_dequant_mxfp8_e4m3(input, scale_tiles, output)`，kernel建议名 `custom_dequant_mxfp8_e4m3_bf16`；独立验收其解码再做往返，不能用两端相同错误相互抵消。旧group128只用于兼容回归，不能作为MXFP8数值golden。
- 要求交付：可复制测试命令及reference、上述环境指纹、每项通过/失败数量、首个FP8 byte/scale code差异的 `(t,d)` 或 `(tile,k)`、最大误差及分布、真实HBM/allocator分配、workspace与profile。CPU、模拟、真机日志必须标记运行模式，不能把尚未执行的验收项标为通过。
