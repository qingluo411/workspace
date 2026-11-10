# [Operator Request] G128 E4M3FN → BF16 反量化：紧凑 scale tile 与标准 E8M0 解码语义

## 需求类型

扩展已有算子能力。

## 优先级

P1。

## 目标版本与期望时间

FP8 KV cache 算子准备里程碑；与建议配对量化入口 `torch.dlc_quant_e4m3_g128_packed` / `custom_quant_e4m3_g128_packed` 共同交付，具体时间由算子组与接入方排期。

## 算子名称

- 建议 PyTorch 接口：`torch.dlc_dequant_e4m3_g128_packed`。
- 建议 kernel 名称：`custom_dequant_e4m3_g128_packed_bf16`。
- 本 issue 仅对应上述一个反量化 kernel；不合并量化 kernel 的实现和验收范围。

## 背景与功能说明

已验证的 `custom_dequant_e4m3_g128_bf16` 读取逐 token、每行按 512 B 对齐的 uint8 scale Tensor。
新的 G128 量化将把多个 token 的 scale 紧凑写入 512 B tile；现有反量化寻址不能直接读取，必须提供配套入口。
本需求同时改变 scale 寻址与新 packed 入口的解码数值合同：采用标准 E8M0 的 code255=NaN 语义，继续按原 FP8 values Tensor 布局读入，FP32 解码/缩放后转换 BF16，输出普通 BF16 Tensor。
配套量化的推荐scale方法 `floor(log2(finite_amax))-8` 与旧 G128 的 `ceil(log2(max(amax/448,1e-10)))` 不同；本次建议新 packed 入口配套对齐数值行为，而不只变更存储布局。OCP MX v1.0 §6.3推荐支持该转换方法，并未把它规定为唯一算法。
values 与 scale 是两个 Tensor，不通过 dtype view 将旧 `[T,G]` 伪装成紧凑布局。
保留旧 `torch.dlc_dequant_e4m3`、`custom_dequant_e4m3_g128_bf16` 的名称、ABI、旧布局和数值语义，避免现有消费者回归。
旧入口code255=FP32 +Inf属于历史行为；新入口不能沿用。当前是待实现issue草案修订，不是已发布packed ABI升级；G128仍为采用OCP推荐数值方法的自定义group128格式，标准group32支持不属于本kernel范围。
不包括 cache gather/slot_mapping、逐 token append、跨请求并发 tile 更新、attention 融合、FP8 values 区重新打包或模型质量验收；本算子完成全 Tensor 转换。

## 期望 API 与函数签名

```python
def dlc_dequant_e4m3_g128_packed(
    input: torch.Tensor,
    scale_tiles: torch.Tensor,
    output: torch.Tensor,
) -> None:
    ...
```

建议 C++ 签名为 `void dlc_dequant_e4m3_g128_packed(const Tensor& input, const Tensor& scale_tiles, Tensor& output)`。
无可变 `group_size` 或 `saturate` 参数；量化的SAT/OVF选择已体现在输入编码中。调用方预分配 output，返回 `None`；PyTorch schema 标记 output mutation，两个输入只读。

## 参数与返回值

- `input`：`torch.Tensor`，待解码的 FP8 values；只读。
- `scale_tiles`：`torch.Tensor`，与 values 对应的 scale 编码；只读。
- `output`：`torch.Tensor`，调用方提供的恢复结果缓冲区；写入。
- 所有参数无默认值；无其他属性；返回值为 `None`。
- dtype、shape、layout、device 集中见“支持范围与约束”。

## 功能语义

`T` 表示token数量，`D` 表示每token的有效元素数；t、d、g分别为从0开始的token、元素和量化组索引。`ceil` 表示向上取整，`//` 表示整数向下除。

本项目的紧凑布局名称为 `TYD_SCALE_T128_G4_V1`，数值行为名称为 `OCP_E4M3_RNE_SAT_OVF_V1`，具体定义如下。values 与 scale 是两个独立 Tensor。令 `B=128,G=(D+B-1)//B,Q=(G+3)//4,NT=(T+127)//128,N=NT*Q`；其中 `//` 表示整数向下除，`ceil(x)` 表示向上取整。若 `T=0` 或 `D=0`，`N=0`。scale Tensor 的 shape 为 `[N,512]`，每个 tile 聚合最多128个token、每token4个scale。

```text
g    = d // 128
tile = (t // 128) * Q + g // 4
k    = (t % 128) * 4 + g % 4
code = scale_tiles[tile, k]
physical_byte_offset = tile * 512 + 4 * (k % 128) + k // 128
```

上述物理偏移含 TYD uint8 字节交织。tile 内序列与 NVIDIA swizzle 不相同；不能把该 uint8 Tensor 当普通 CPU row-major HBM 字节流直接访问。
对每个有效 `(t,d)`：

```python
if code == 255:
    y = NaN                        # 整个有效128-element group，不依赖q
else:
    s = float32(2.0 ** (int(code) - 127))
    q = decode_e4m3fn(input[t, d])   # FP32，保留有限值/正负零/NaN
    y = float32(q * s)
output[t, d] = BF16_RNE(y)
```

code `0…254` 对应 `2^(code-127)`，code0表示 `2^-127`，不是零scale；code254表示 `2^127`。E8M0没有零或Inf scale。
**code255必须先判为NaN scale，整个有效128元素group恢复NaN，包括q为正负零、有限非零或NaN的情况。** 禁止沿用 `exp2(255-127)` 得到Inf的旧路径；与旧有限非零q得到Inf的差异是本次新合同的预期行为。
配对量化对每个有效128元素group，取finite元素绝对值最大值a；a=0时e=0，否则 `e=clamp(floor(log2(a))-8,-127,127)`，code=e+127，以FP32归一化 `ldexp(x,-e)` 后E4M3 RNE。SAT将±Inf及溢出转带符号448；OVF将其转canonical NaN 0x7f；输入NaN始终转0x7f，NaN/Inf排除于amax，正负零保留，不使用1e-10下限。舍入后溢出使464中点到448而466在OVF下为NaN。此为完整配对数值合同；配对量化支持所有BF16编码类别，正常产生code0…246而不生成255；这不减免本反量化支持外部构造的全部256种scale code的义务。
E4M3FN 的 `0x7f/0xff` 为 NaN，`0x00/0x80` 是正负零，无 Inf payload；NaN payload 不作位级承诺。
有限scale下保留正负零符号；FP32乘法溢出得到带符号Inf，这是本项目明确选择，不能表述为MX标准唯一允许的行为。最终BF16采用RNE，溢出为带符号Inf、保留可表示subnormal，下溢舍入至零保留符号，不允许无条件FTZ。
尾 group 仅恢复有效 D，未使用 scale 槽全部忽略；即使 padding 槽含 poison，输出也不变。
`T=0` 或 `D=0` 时输入/输出同 shape、scale `[0,512]`，校验后 no-op、返回 `None`。
output 不得与任一输入共享 storage；输入之间也要求独立 storage，防止以重叠 view 解释两个独立布局。输入和 scale 均不得被修改。
双XYS可以共享只读tile，写output时必须独占各自完整values行的DMA范围；禁止由整块写回覆盖其他XYS或此前VMEM分块已写出的行。配对量化端对完整scale tile须有唯一writer；跨VMEM块累积完整后写回，避免覆盖。反量化无需改写tile，全部无效槽按mask忽略。调用期间不允许其他 stream/请求修改输入 tile。
确定性执行，无随机数、无 CPU fallback、无为 tensor 数值扫描而引入的 host 同步。

## 支持范围与约束

| 参数 | dtype | 逻辑 shape | 存储合同 |
|---|---|---|---|
| `input` | `float8_e4m3fn` | `[T,D]` | 现有普通 TYD FP8 Tensor |
| `scale_tiles` | uint8 | `[NT*Q,512]`，任一维为零时 `[0,512]` | `TYD_SCALE_T128_G4_V1` |
| `output` | BF16 | `[T,D]` | 普通 TYD BF16 Tensor |

支持动态 T、D、尾 token tile、尾 group，不做 broadcasting；首批必须覆盖 `T=0…4096,D=0…8192` 的适用组合。
大 shape 的实际边界由显存、VMEM 分块和内核索引范围决定；所有 shape/乘法/对齐检查先用 int64，超过实现边界在 launch 前明确拒绝，不得溢出。
支持全部 256 种 FP8 raw code 和全部 256 种 scale code。scale=255按上述NaN语义，不能只测试配套量化端通常产生的code0…246；code0和254均为合法有限scale。
FP32 乘法/exp2 的溢出、下溢及转换 BF16 的非有限结果按 IEEE/reference 验收；不对 BF16 有限性作无依据保证。
三 Tensor 必须同一 TYD 卡、contiguous、`storage_offset=0`，无 storage alias。错误 dtype、rank、shape、device、stride、offset 和 alias 立即给出明确异常。
支持当前 DLC stream、非默认 stream 和 graph capture/replay；无 host tensor 扫描/同步。单卡独立执行，不需要 collective；本次不支持其他 FP8 变体、group32 或其他芯片。
定义 `alignA(x)=ceil(x/A)*A`。在本 issue 保留的普通 values 布局下，非空 Tensor 的持久化容量为：

```text
FP8 values bytes = T * align512(D)
scale bytes      = N * 512
FP8 总容量       = T * align512(D) + N * 512
BF16 对照容量    = T * align256(D) * 2
```

任一维为零时容量按零计算；allocator 分配粒度另报。`T=128,D=512` 对应64 KiB values加512 B scale，共64.5 KiB；BF16为128 KiB，节省49.609375%。D=128等小D仍有values行padding，少量T仍有尾tile浪费，不能保证所有shape均节省或都接近减半。

## 精度要求

使用以下可直接运行的独立CPU reference作为golden。输入为CPU uint8 FP8 raw bytes `[T,D]` 及CPU uint8逻辑scale tiles `[N,512]`，返回BF16 `[T,D]`；函数内完整定义解码和解包，先屏蔽code255再计算有限scale，不调用待测设备实现。

```python
import torch

def reference_dequant_e4m3_g128(bits, scale_tiles):
    # CPU logical tensors; do not apply the physical native byte weave here.
    for x in (bits, scale_tiles):
        if x.device.type != "cpu" or x.dtype != torch.uint8 or x.ndim != 2:
            raise ValueError("bits and scale_tiles must be rank-2 CPU uint8")
    T, D = bits.shape
    G = (D + 127) // 128
    Q = (G + 3) // 4
    N = ((T + 127) // 128) * Q
    if tuple(scale_tiles.shape) != (N, 512):
        raise ValueError(f"scale_tiles must have shape {(N, 512)}")
    if T == 0 or D == 0:
        return torch.empty((T, D), dtype=torch.bfloat16)
    t = torch.arange(T)[:, None]
    g = torch.arange(G)[None, :]
    codes = scale_tiles[(t // 128) * Q + g // 4, (t % 128) * 4 + g % 4]
    raw = bits.to(torch.int32)
    mag = raw & 127
    exp, mant = mag >> 3, mag & 7
    normal = (1.0 + mant.float() / 8.0) * torch.exp2(exp.float() - 7.0)
    q = torch.where(exp == 0, mant.float() * (2.0 ** -9), normal)
    q = torch.where((raw & 128) != 0, -q, q)
    q = torch.where(mag == 127, float("nan"), q)
    nan_scale = codes == 255
    safe_codes = torch.where(nan_scale, 127, codes)
    scales = torch.exp2(safe_codes.float() - 127.0)
    groups = torch.arange(D) // 128
    y = q * scales[:, groups]
    y = torch.where(nan_scale[:, groups], float("nan"), y)
    return y.to(torch.bfloat16)
```
新输出与新golden比较，所有有限输出的BF16 raw bits一致，`atol=0,rtol=0`。旧入口只用于legacy回归与历史差异对照：原scale逐token行对齐512 B，以 `exp2(code-127)` 解码，code255变成FP32 +Inf；此时q=0得到NaN，有限非零q得到带符号Inf。不能把它作为新入口code255的golden。
相同逻辑payload在code0…254时可做新旧交叉核对；code255时必须分别符合各自合同，不能将“新旧所有输出逐位相同”设为验收目标。
正负零要求符号位一致；Inf 要求位置及符号一致；NaN 比较 mask，不比较 payload。FP32 中间值与最终 BF16 值的非有限分类也应符合 reference。
至少测试全部 256 FP8 code 与全部 256 scale code 的交叉组合，覆盖 code0、127、246、254、255 和 FP8 NaN/负零；特别验证code255搭配非零q时新输出NaN而legacy输出Inf。
最小BF16 subnormal定向验证：`q=0x08,code=0 → 2^-133`，负号对称为 `q=0x88 → -2^-133`；补半最小subnormal、RNE tie、舍入至零、FP32/BF16溢出边界，禁止FTZ。
往返误差报告与实现正确性分开；任意外部FP8/scale组合可还原溢出为Inf，新量化的SAT可能裁剪、OVF可能生成NaN，旧链路的BF16极值也可能往返为Inf，均需按各自golden报告。不能用 `equal_nan` 掩盖错误Inf、NaN位置或符号。新旧量化链路另报最大绝对误差、RMSE、NRMSE、饱和与NaN/Inf比例；不承诺采用推荐方法总能提高精度。
配套amax480定向例中，legacy scale2/q240恢复480，新recipe的SAT scale1/q448恢复448，OVF恢复NaN；amax464时scale1/q448恢复448，466时SAT恢复448、OVF恢复NaN。全零新scale code127，旧code94；最小BF16 2^-133新code0/q0x08恢复非零，旧算法量化归零。需报告这些预期数值差异，不能把它们误判为反量化寻址错误。
固定 seed，记录输入分布/shape/样本数；FP32 解码缩放，BF16 RNE 输出，不允许引入额外低精度中间量。

量化损失单独按原始值x与恢复值y统计：`max_abs=max(abs(y-x))`，`RMSE=sqrt(mean((y-x)^2))`，`NRMSE=sqrt(sum((y-x)^2)/sum(x^2))`。只对x、y均finite的样本计算这些实数指标，同时必须报告该子集的数量/占比，以及全样本NaN/Inf位置、符号和比例；OVF新增NaN不能被静默过滤成“精度通过”。分母为0时若误差也为0记NRMSE=0，否则记Inf。逐元素相对误差为 `abs(y-x)/abs(x)`，原值0单独报告，near-zero阈值与样本数须公开；cosine为有限子集内积除以两向量范数乘积，零范数单独标记。饱和比例按SAT中归一化绝对值大于448或原值为Inf的元素统计，并额外记录RNE后会溢出的比例；新增零比例指原值非零、恢复为零的元素占比。所有编码/scale/NaN mask仍须先与golden严格一致，损失指标不能替代实现正确性。

## 测试范围

- [x] 基本功能与 reference 对齐
- [x] 多 dtype 精度（FP8/uint8 输入与 BF16 输出；其他 dtype 拒绝）
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [x] 非 contiguous、特殊 stride 或 layout（拒绝错误输入，验证 tile 寻址）
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式（不在本 issue 范围）
- [x] 长时间稳定性
- [x] 性能基准与性能回归

固定 seed=20261015；每个 shape/分布开始时重置 CPU `torch.Generator().manual_seed(20261015)`，先生成FP32标准正态 `x=torch.randn((T,D),generator=generator)`，再按下列规则生成输入，最后统一转BF16：

- `signed_zero`：全置+0，奇数列置-0。
- `constant`：全置1，奇数token行置-1。
- `normal`：保留标准正态。
- `tiny`：列d的幅值为 `2^(-126+d%107)`，奇数列取负，所有token行相同。
- `group_scales`：将normal的列d乘以 `2^(((d//128)%5)*6-12)`。
- `outliers`：normal乘0.01；每128列的首元素置16，奇数token行相应首元素置-16。

上述6种分布与 `T∈{1,17,129,257},D∈{64,128,129,512,576}` 组合为120组基础输入；SAT/OVF分别验收。

量化、反量化分别与新reference对齐，不能只验证二者误差相互抵消的roundtrip；另测量化输入NaN/Inf策略、全零code127、amax480新旧差异及最小BF16 subnormal。
补 `T∈{0,127,128,255,256,4096},D∈{0,1,127,511,513,8192}` 的边界组合，特别检查 t=127/128、g=3/4、多个 token tile 与多个 Q。
FP8 all-code × scale all-code 用不规则 T/D 摆放，避免各 tile 内容相同掩盖错索引；修改无效 scale 槽及 values padding 为 poison，应不影响有效输出。
输入/scale 操作前后逐字节一致；output预填 poison 确认所有有效元素覆盖。错误 shape、dtype、CPU/异卡、非 contiguous、offset、alias 单独验证。
重复至少 1000 次、交替 shape；覆盖双 XYS、非默认 stream、graph replay 和旧 API 回归。记录时延、cycles、带宽及峰值 workspace；不设未经实测的速度阈值，异常退化须提供 profile。

## 验证环境与依赖版本

TYD 真机，`DLC_TPU_VERSION=2`；在指定空闲单卡验证并记录device index。使用PyTorch DLC fork，新增入口和kernel须完成构建、安装、实际注册与加载，不能仅凭源码或ELF blob存在认定可用。
验收报告记录PyTorch、DLC_Custom_Kernel、LLVM、DLCSynapse、DLCsim的实测完整Git SHA、工作区未提交修改、kernel共享库SHA256、driver/RTABI版本，以及 `LD_LIBRARY_PATH`、`CUSTOM_KERNEL_PATH`、`DLC_SYN_*`。driver 23.0.1、RTABI 3.0.0仅为当前历史基线版本，实际验收须重新记录。
vLLM/vllm-cl不参与本issue必测调用链。已有旧G128结果仅为legacy历史基线，不能代替新布局、新数值方法或显存测量结果。

## 验收标准

- [ ] Python/C++ 接口、mutation schema、参数错误、空 Tensor 与 alias 行为全部符合合同；旧 G128 API/ABI 不变且回归通过。
- [ ] 新输出与新G128 golden逐位一致；NaN按mask比较，code255优先传播整组NaN，全部raw code/scale交叉、subnormal/RNE、特殊值和尾边界用例通过；旧入口单独按legacy回归，不要求新旧code255输出相同。
- [ ] 与配套量化完成 120 cases 及有限 BF16 极值往返验证；反量化任意 FP8/scale 特殊值、padding poison、双 XYS/stream/graph 场景通过。
- [ ] 提交新旧链路最大绝对误差、RMSE、NRMSE、SAT饱和比例与OVF新增NaN比例，不预设未经模型验证的比例阈值。
- [ ] `T=128,D=512` 的持久化输入为 64 KiB values + 512 B scale = **64.5 KiB**，相对 BF16 缓存 128 KiB 节省 **49.609375%**；不存在旧逐行 scale 的持久化副本。
- [ ] 隔离分配测量 `memory_allocated` 差值并核对 storage、报告 allocator 粒度；不使用 `numel*element_size` 或 `memory_reserved` 代替显存证据。
- [ ] 容量收益不把独立恢复出的 BF16 output 算入持久化 FP8 缓存；反量化峰值包含 output/workspace，另行报告，不能声称 standalone 全量还原时总峰值同样减半。
- [ ] 可复现 reference、自动测试、调用示例、精度和性能报告提交；暂不支持 cache/attention 融合的限制写明。

## Reference、日志与附件

- CPU reference：上文内嵌 `reference_dequant_e4m3_g128` 可直接复制到CPU PyTorch环境运行，无其他项目文件依赖；输入是CPU逻辑uint8 raw bytes与逻辑scale tiles，函数内完整定义E4M3 bit-field解码和tile解包，不调用待测设备实现生成golden。FP32解码/缩放和BF16 RNE即本issue目标。
- 固定golden：code255搭配任意q必须NaN；`q=0x08,code=0` 恢复 `2^-133`，负号对称；`q=0x00/0x80,code=127` 为正/负零；配套480、464/466、最小BF16、全零例的预期结果已写入正文，应硬编码独立核对。
- 标准依据为 Open Compute Project《OCP Microscaling Formats (MX) Specification》Version 1.0，§5.4.1定义E8M0的code0…254为 `2^(code-127)`、code255为NaN，§5.1定义NaN scale传播至整个有效block；§6.3推荐支持基于最大元素指数的转换方法，并非唯一scale算法。本文B128为自定义格式，仅对齐所述数值方法和E8M0语义；超出FP32范围的乘积处理为本项目明确选择。
- 元素格式依据为 Open Compute Project《OCP 8-bit Floating Point Specification (OFP8)》Revision 1.0（2023-12-01勘误版），§5.1定义E4M3、正负零、次正规数、NaN且无Inf；§5.2.1定义RNE、SAT/OVF转换。本issue的FP32中间计算和最终BF16溢出/下溢规则已完整写入正文。
- 交付时在实现PR或issue附件提交可执行CPU reference与自动测试、全部case的shape/seed/分布、BF16 raw bits/NaN mask、全部FP8×scale编码交叉结果、新旧预期差异、NaN/Inf与量化损失报告、隔离显存/storage/workspace证据、环境版本、profile、调用示例和失败时首个差异坐标。现有报告仅作历史legacy基线，不算新验收。
