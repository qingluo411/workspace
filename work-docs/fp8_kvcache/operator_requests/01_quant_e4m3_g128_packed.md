# [Operator Request] G128 E4M3FN 量化：紧凑 scale tile 与 OCP 推荐数值方法对齐

## 需求类型

扩展已有算子能力。

## 优先级

P1。

## 目标版本与期望时间

FP8 KV cache 算子准备里程碑；交付时间由算子组与接入方共同排期，具体交付版本与日期待排期。

## 算子名称

- 建议 PyTorch 接口：`torch.dlc_quant_e4m3_g128_packed`。
- 建议 kernel 名称：`custom_quant_e4m3_g128_packed`。
- 本 issue 仅对应上述一个量化kernel；其scale输出供建议配对入口 `torch.dlc_dequant_e4m3_g128_packed` / `custom_dequant_e4m3_g128_packed_bf16` 读取，输出合同在本文完整定义。

## 背景与功能说明

现有 `custom_quant_e4m3_g128` 为每个 token、每 128 个通道生成一个 uint8 scale code，scale Tensor 的逻辑 shape 为 `[T,ceil(D/128)]`，物理存储逐行按 512 B 对齐。
例如 `T=128,D=512` 时，仅有 512 B 有效 scale，却占用 64 KiB，抵消 FP8 数据相对 BF16 的容量收益。
本需求在新 packed 入口同时引入跨 128 个 token 聚合的 scale tile，以及与 OCP MX v1.0 §6.3 推荐支持的方法一致的 scale 选择和标准 E8M0 特殊值语义；FP8 values 继续使用现有普通 Tensor 布局，values 与 scale 分为两个 Tensor。
现有量化采用 `ceil(log2(max(amax/448,1e-10)))`，与推荐方法 `floor(log2(amax))-8` 不同；现有反量化将 scale code255 解释为 FP32 +Inf，也与 E8M0 的 NaN 语义不同。本批建议在新入口一并对齐这些数值行为，不能只改布局后继续把历史行为当作新目标。
旧 `torch.dlc_quant_e4m3`、`custom_quant_e4m3_g128` 的名称、参数、返回值、布局与行为均保留，现有权重量化消费者无需迁移。
这是待实现 issue 草案的修订，不是已发布 packed ABI 的升级；新旧入口必须显式区分，不按 shape 自动猜测数值 recipe。
本批为全 Tensor 转换；不包括 cache `slot_mapping`/scatter、逐 token 原地 append、跨请求共享 tile 并发更新、attention 融合或 FP8 values 区重新打包。
这是采用 OCP 推荐数值方法与 E8M0 编码的自定义 group128 格式，仍不称为标准 MXFP8；标准group32支持不属于本kernel范围。OCP §6.3 推荐实现支持该转换方法，并不规定它是唯一合法的 scale 选择算法。

## 期望 API 与函数签名

```python
def dlc_quant_e4m3_g128_packed(
    input: torch.Tensor,
    output: torch.Tensor,
    scale_tiles: torch.Tensor,
    saturate: bool = True,
) -> None:
    ...
```

建议 C++ 签名为 `void dlc_quant_e4m3_g128_packed(const Tensor& input, Tensor& output, Tensor& scale_tiles, bool saturate=true)`。
无 `group_size` 可变参数，名称固定 G128。`saturate=True` 为 SAT，`False` 为 OVF，与 group32 MXFP8 量化接口一致。调用方预分配两个输出，算子写入后返回 `None`；PyTorch schema 必须标记两个输出的 mutation/alias 集合。

## 参数与返回值

- `input`：`torch.Tensor`，待量化数值；只读。
- `output`：`torch.Tensor`，调用方提供的量化结果缓冲区；写入。
- `scale_tiles`：`torch.Tensor`，调用方提供的 scale 编码缓冲区；写入。
- `saturate`：Python `bool`，默认 `True`，选择 FP8 结果发生溢出时的 SAT/OVF 行为。
- 其余参数均无默认值；无其他属性；返回值为 `None`。
- dtype、shape、layout 和 device 要求统一见“支持范围与约束”。

## 功能语义

`T` 表示token数量，`D` 表示每token的有效元素数；t、d、g分别为从0开始的token、元素和量化组索引。`ceil` 表示向上取整，`//` 表示整数向下除。

本项目采用 OCP MX v1.0 §6.3 推荐支持的 scale 选取方法，数值行为名称为 `OCP_E4M3_RNE_SAT_OVF_V1`，其完整规则如下，但 group 大小仍为128。E4M3 中最大可表示的 2 的幂是 `2^8=256`。对每个 token `t` 和沿 D 维的 group `g`，仅用有效元素计算：

```python
x = FP32(input[t, 128*g : min(128*(g+1), D)])
a = max((abs(x_i) for x_i in x if isfinite(x_i)), default=0.0)
if a == 0:
    e = 0
else:
    e = clamp(floor(log2(a)) - 8, -127, 127)
code = uint8(e + 127)
z = FP32(ldexp(x, -e))
output[t, group] = E4M3_RNE(z, overflow_mode=('SAT' if saturate else 'OVF'))
scale_tiles[tile, k] = code
```

指数选择必须准确识别 2 的幂及相邻 BF16 值，优先精确指数提取/`frexp`，不得因近似低精度 log2 归错组。归约、scale 和归一化为 FP32，最终 FP8 使用 RNE；新入口不得使用 `1e-10` 下限或旧 `ceil(log2(amax/448))` 作为本合同的替代。

- 全零或没有 finite 非零元素的 group 使用 `e=0,code=127,scale=1`；原始正负零保留符号。无效 tile 槽写0是 padding 约定，不表示有效全零 group 的 scale。
- NaN/Inf 不参与 finite amax；NaN 元素输出 canonical E4M3 NaN `0x7f`。
- SAT：±Inf 以及舍入后超出 E4M3 有限范围的元素转为带符号448。
- OVF：±Inf 以及舍入后溢出的元素转为 canonical NaN；其他元素独立编码，不因一个 NaN/Inf 强制整组污染。
- 正常值、次正规值与下溢至带符号零均按 RNE；溢出按 OFP8 的舍入后语义判断，特别是464中点必须舍入到448。
- E8M0 code0…254分别表示 `2^(code-127)`，code255为NaN，不能沿用旧 G128 的 Inf 解释；本量化 recipe 不生成255，但配套反量化必须支持它。

finite 归约时排除 NaN/Inf、全零 scale 选1是本项目固定的确定性策略，并非 OCP 规定的唯一策略。与 legacy 的差异需报告，数值对齐不等于保证所有输入误差都减小。不得为数值扫描增加 host 同步；接口仅对元信息可判断的非法参数立即报错。

本项目的紧凑布局名称为 `TYD_SCALE_T128_G4_V1`，数值行为名称为 `OCP_E4M3_RNE_SAT_OVF_V1`，具体定义如下。values 与 scale 是两个独立 Tensor。令 `B=128,G=(D+B-1)//B,Q=(G+3)//4,NT=(T+127)//128,N=NT*Q`；其中 `//` 表示整数向下除，`ceil(x)` 表示向上取整。若 `T=0` 或 `D=0`，`N=0`。scale Tensor 的 shape 为 `[N,512]`，每个 tile 聚合最多128个token、每token4个scale。

```text
tile = (t // 128) * Q + g // 4
k    = (t % 128) * 4 + g % 4
scale_tiles[tile, k] = code
physical_byte_offset = tile * 512 + 4 * (k % 128) + k // 128
```

`k` 是 uint8 Tensor 的逻辑列索引，最后一行公式才是 TYD native 字节交织后的物理偏移；不得直接按 `tile*512+k` 解释物理 HBM。
尾 group 只归约有效 D；尾 token 和尾 scale 列对应的未使用 tile 槽全部写 0，避免遗留数据。输入 padding 不得影响 amax。
`T=0` 或 `D=0` 时，`scale_tiles` 为 `[0,512]`，参数校验后 no-op，返回 `None`。
输出不与任意其他参数共享 storage，两个输出也不共享 storage；输入不得被修改。
调用方独占本次完整输出 tile。双XYS按完整128-token block分配其所有scale tiles与values行；每个完整输出tile只由一个writer写回。跨VMEM分块处理时，该writer必须在最终写回前保留或安全累积tile内已有结果，禁止后一个VMEM分块的整tile DMA覆盖前一个分块或另一个XYS已写出的scale；values行的DMA范围同样独占。
算子确定性执行，无随机数；不隐式回退 CPU，不生成旧的逐 token scale 临时 HBM Tensor 再由 host 压缩。

## 支持范围与约束

| 参数 | dtype | 逻辑 shape | 存储合同 |
|---|---|---|---|
| `input` | BF16 | `[T,D]` | 普通 TYD BF16 Tensor |
| `output` | `float8_e4m3fn` | `[T,D]` | 普通 TYD FP8 Tensor，保留原 values padding |
| `scale_tiles` | uint8 | `[NT*Q,512]`，任一维为零时 `[0,512]` | `TYD_SCALE_T128_G4_V1`，每行恰好 512 B |
| `saturate` | Python bool | 标量属性 | SAT/OVF |

支持动态 T、D、非 128 倍数的 D 和非 128 倍数的 T；无需 broadcasting。
首批必须覆盖 `T=0…4096,D=0…8192` 的适用组合；实际最大 shape 受设备可用显存、VMEM 分块与内核索引范围限制，超出限制需在 launch 前明确报错，禁止整数溢出或静默截断。
支持所有 BF16 编码类别：finite、±0、subnormal、±Inf、NaN；不得无条件 FTZ。最小非零 BF16 `2^-133` 必须选择code0并编码为 `q=0x08`，配套恢复仍非零。
有限 BF16 amax 决定本 recipe 生成 code0…246，全零为code127，不生成255；不减免反量化支持全部 E8M0 编码的义务。最大有限值可能在 SAT/OVF 产生不同的有损结果，按本 reference 验收，不保证恢复值始终有限。
对错误 dtype、rank、shape、device、stride、offset、storage alias 和非法属性，由 PyTorch 接口立即给出明确错误。
三 Tensor 必须同一 TYD 卡、contiguous、`storage_offset=0`；不支持 CPU/CUDA Tensor、任意 stride 或 view 别名。
支持当前 DLC stream 异步执行、非默认 stream 与 graph capture/replay；实现不得为数据检查引入 host 同步。单进程单卡执行，不依赖分布式 collective。
本布局为TYD执行布局，与NVIDIA scale swizzle不相同；对已有 `[T,G]` 做 `view(-1)` 不满足本合同。

定义 `alignA(x)=ceil(x/A)*A`。在本 issue 保留的普通 values 布局下，非空 Tensor 的持久化容量为：

```text
FP8 values bytes = T * align512(D)
scale bytes      = N * 512
FP8 总容量       = T * align512(D) + N * 512
BF16 对照容量    = T * align256(D) * 2
```

任一维为零时容量按零计算；allocator 分配粒度另报。`T=128,D=512` 对应64 KiB values加512 B scale，共64.5 KiB；BF16为128 KiB，节省49.609375%。D=128等小D仍有values行padding，少量T仍有尾tile浪费，不能保证所有shape均节省或都接近减半。

## 精度要求

使用以下可直接运行的独立CPU reference作为新合同的golden。输入为CPU BF16 `[T,D]`，返回CPU uint8 FP8 raw bytes `[T,D]` 和CPU uint8逻辑scale tiles `[N,512]`；CPU tiles不额外做物理字节交织。SAT/OVF分别对照对应参数；设备实现不得参与golden生成。

```python
import torch
import torch.nn.functional as F

def reference_quant_e4m3_g128(src, saturate=True):
    # CPU logical tensors; same-dtype CPU->DLC copy performs the native weave.
    if src.device.type != "cpu" or src.dtype != torch.bfloat16 or src.ndim != 2:
        raise ValueError("src must be a rank-2 CPU BF16 tensor")
    if not isinstance(saturate, bool):
        raise TypeError("saturate must be bool")
    T, D = src.shape
    G = (D + 127) // 128
    Q = (G + 3) // 4
    N = ((T + 127) // 128) * Q
    tiles = torch.zeros((N, 512), dtype=torch.uint8)
    if T == 0 or D == 0:
        return torch.empty((T, D), dtype=torch.uint8), tiles
    x = F.pad(src.float(), (0, G * 128 - D)).reshape(T, G, 128)
    finite_abs = torch.where(torch.isfinite(x), x.abs(), 0.0)
    amax = finite_abs.amax(dim=-1)
    _, exp = torch.frexp(amax)
    e = torch.where(amax > 0, exp - 1 - 8, 0).clamp(-127, 127)
    codes = (e + 127).to(torch.uint8)
    z = torch.ldexp(x, -e.unsqueeze(-1))
    if saturate:
        z = z.clamp(-448.0, 448.0)
    bits = z.to(torch.float8_e4m3fn).view(torch.uint8)
    bits = torch.where((bits & 127) == 127, 0x7f, bits)
    bits = bits.reshape(T, G * 128)[:, :D].contiguous()
    t = torch.arange(T)[:, None]
    g = torch.arange(G)[None, :]
    tiles[(t // 128) * Q + g // 4, (t % 128) * 4 + g % 4] = codes
    return bits, tiles
```
新量化 FP8 bytes、有效 scale codes 与新 golden 逐字节一致，NaN编码统一0x7f，正负零符号一致；`atol=0,rtol=0`，不接受放宽误差掩盖编码或布局错误。固定 seed=20261015，报告 seed、shape 与输入分布。
旧入口的 `e=ceil(log2(max(amax/448,1e-10)))`、`code=e+127`、RNE量化及逐行scale布局只用于legacy回归和新旧差异报告，不要求新旧FP8/scale逐字节相同。
验证零、负零、常量、normal、tiny、group_scales、outliers，并补 BF16 最大/最小正规值、次正规值及 FP8 舍入临界点。
定向验证：group amax480时，旧量化code128、峰值q240，恢复480；新量化code127、SAT峰值448并恢复448，OVF峰值NaN。此例明确说明推荐方法对齐不保证精度总是更好。amax464时峰值RNE到448，466时SAT448/OVF NaN。最小 BF16 `2^-133` 新code0/q0x08而旧算法归零；新全零code127而旧code94。
必须覆盖 NaN/Inf 混合有限块、只有 NaN/Inf 的块及全零块；验证 finite amax 排除、SAT/OVF、canonical NaN 和重复执行确定性。由同组amax256固定scale1，检查 `2^-10` RNE到±0、`3*2^-10` 到±`2^-8`。
中间归约/scale/归一化用 FP32，FP8 转换使用 RNE。新旧量化损失与差异分别报告 max_abs、RMSE、NRMSE、平均/最大逐元素相对误差、cosine、新增零比例、饱和比例、NaN/Inf比例；near-zero及原值0的相对误差口径单独说明。不得承诺此数值对齐在所有分布上提高精度，不把模型质量门限作为本 issue 的算子正确性判据。

量化损失单独按原始值x与恢复值y统计：`max_abs=max(abs(y-x))`，`RMSE=sqrt(mean((y-x)^2))`，`NRMSE=sqrt(sum((y-x)^2)/sum(x^2))`。只对x、y均finite的样本计算这些实数指标，同时必须报告该子集的数量/占比，以及全样本NaN/Inf位置、符号和比例；OVF新增NaN不能被静默过滤成“精度通过”。分母为0时若误差也为0记NRMSE=0，否则记Inf。逐元素相对误差为 `abs(y-x)/abs(x)`，原值0单独报告，near-zero阈值与样本数须公开；cosine为有限子集内积除以两向量范数乘积，零范数单独标记。饱和比例按SAT中归一化绝对值大于448或原值为Inf的元素统计，并额外记录RNE后会溢出的比例；新增零比例指原值非零、恢复为零的元素占比。所有编码/scale/NaN mask仍须先与golden严格一致，损失指标不能替代实现正确性。

## 测试范围

- [x] 基本功能与 reference 对齐
- [x] 多 dtype 精度（BF16 输入、FP8 输出、uint8 scale；其他 dtype 拒绝）
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [x] 非 contiguous、特殊 stride 或 layout（拒绝错误输入；验证 native tile 布局）
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

补 `T∈{0,127,128,255,256,4096}` 与 `D∈{0,1,127,511,513,8192}` 的边界组合。
覆盖 `G=1,4,5` 及多个 Q；特别检查 t=127/128、g=3/4 的跨 tile 边界。预填输出 tile 为非零 poison，验证所有无效槽清零；input padding 用 poison，验证有效结果不变。
重复至少 1000 次，交替小/大 shape；覆盖双 XYS 分工、非默认 stream、graph replay 和旧 API 回归。每个循环不要求额外 host 同步，正确的同步放在验收读取处。
固定 warmup/迭代数，记录旧/新 kernel 时延、双 XYS cycles、有效带宽和输出分配字节；此阶段不编造性能阈值，异常退化须给出 profile 解释。

## 验证环境与依赖版本

TYD 真机，`DLC_TPU_VERSION=2`；在指定空闲单卡验证并记录device index。使用PyTorch DLC fork，新增入口和kernel须完成构建、安装、实际注册与加载，不能仅凭源码或ELF blob存在认定可用。
验收报告记录PyTorch、DLC_Custom_Kernel、LLVM、DLCSynapse、DLCsim的实测完整Git SHA、工作区未提交修改、kernel共享库SHA256、driver/RTABI版本，以及 `LD_LIBRARY_PATH`、`CUSTOM_KERNEL_PATH`、`DLC_SYN_*`。driver 23.0.1、RTABI 3.0.0仅为当前历史基线版本，实际验收须重新记录。
vLLM/vllm-cl不参与本issue必测调用链。已有旧G128结果仅为legacy历史基线，不能代替新布局、新数值方法或显存测量结果。

## 验收标准

- [ ] 新 Python/C++ API、mutation schema、SAT/OVF属性默认值、参数校验与全部BF16类别行为符合约定；旧入口/ABI及原数值行为保持不变，其消费者独立回归通过。
- [ ] 规定测试全部通过；FP8 bytes及有效scale codes与新G128 golden逐字节一致，NaN/Inf/subnormal/RNE定向、padding、跨tile、双XYS与graph/stream场景通过；不以新旧逐字节一致作为目标。
- [ ] 分别记录SAT饱和比例和OVF新增NaN比例，并提交新旧最大绝对误差、RMSE、NRMSE差异；本阶段不设未经真实KV验证的比例阈值。
- [ ] `T=128,D=512` 的 FP8 values 为 64 KiB、scale storage 为 512 B，持久化两者共 64.5 KiB；相对 BF16 128 KiB 节省 **49.609375%**。
- [ ] 用独立分配测量 `memory_allocated` 差值并报告 allocator 粒度；同时核对实际 storage 容量。不得以 `numel*element_size` 或 `memory_reserved` 代替分配证据。
- [ ] 空间对比排除原输入、恢复 BF16 输出和临时 workspace；峰值/临时开销另报，禁止暗中保留旧逐行 scale。
- [ ] D=128 的 values 行 padding 尚存、尾 tile 开销等限制如实记录；提交性能、可复现reference、自动测试、调用示例及新旧绝对/RMSE/NRMSE/饱和/NaN差异报告，不宣称数值对齐必然提高精度。

## Reference、日志与附件

- CPU reference：上文内嵌 `reference_quant_e4m3_g128` 可直接复制到安装CPU PyTorch且支持 `float8_e4m3fn` 的Python环境运行；只有 `torch` 依赖，无其他项目文件依赖。它返回逻辑FP8 raw bytes和逻辑scale tiles；不在CPU端额外施加TYD物理字节交织。FP8 RNE编码使用CPU PyTorch，不调用待测设备实现生成golden。
- 固定golden：上文列出的480、464/466、最小BF16、全零、正负零和RNE临界点均须硬编码预期bytes/scale独立核对，避免只靠同一公式自证。
- 标准依据为 Open Compute Project《OCP Microscaling Formats (MX) Specification》Version 1.0，§5.4.1规定E8M0的code0…254为2的整数次幂、code255为NaN；§6.3推荐实现支持基于最大元素指数的转换方法，并非强制唯一scale算法。本issue的B128为自定义格式，数值方法采用该推荐规则。
- 元素格式依据为 Open Compute Project《OCP 8-bit Floating Point Specification (OFP8)》Revision 1.0（2023-12-01勘误版），§5.1定义E4M3编码，§5.2.1定义RNE及SAT/OVF转换：±0/subnormal保留，E4M3无Inf，SAT映射至带符号最大有限值，OVF溢出映射NaN。本issue另行固定NaN为0x7f、非有限值不参与amax的策略，完整规则已写入正文。
- 交付时在实现PR或issue附件提交可执行CPU reference与自动测试、全部case的shape/seed/分布/模式、FP8 bytes与有效scale对比、NaN/Inf/饱和及新旧量化损失报告、隔离显存分配和storage容量报告、workspace峰值、环境版本、profile、调用示例与失败时首个差异坐标。现有报告仅作历史legacy基线，不算新验收。
