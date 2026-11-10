# FP8 算子需求：紧凑 scale 与 MXFP8(E4M3)

状态：待提交的算子需求草案；日期：2026-10-15。本次修订建议新 G128 packed 入口同时对齐 OCP 推荐的数值方法与标准 E8M0 特殊值语义，替代此前“G128 仅改布局、沿用旧数值”的草案。本文约定拟交付行为，不代表新 kernel 已实现或通过真机验收。

归档说明：本目录为仓库外预研资料，不提交到组件仓库。临时 PyTorch 注册和 kernel 仓内测试改动已撤回；CPU reference 位于 work-tests，旧接口设备测试需正式或临时接口实现。

本批基于 [算子需求模板](../../../DLC_Custom_Kernel/.github/ISSUE_TEMPLATE/operator_request.yml)，每个独立 kernel 对应一份 issue；Markdown 的二级标题逐项对应模板的 14 个字段。可直接据此填写 GitHub 表单，本次仅生成本地文档。

| 顺序 | Issue 文档 | 建议 kernel | 作用 |
|---|---|---|---|
| 1 | [G128 量化](01_quant_e4m3_g128_packed.md) | `custom_quant_e4m3_g128_packed` | 紧凑 scale，并对齐 OCP 推荐数值方法 |
| 2 | [G128 反量化](02_dequant_e4m3_g128_packed.md) | `custom_dequant_e4m3_g128_packed_bf16` | 紧凑 scale，采用标准 E8M0 解码 |
| 3 | [MXFP8 量化](03_quant_mxfp8_e4m3.md) | `custom_quant_mxfp8_e4m3` | group32、E4M3、标准 E8M0、SAT/OVF |
| 4 | [MXFP8 反量化](04_dequant_mxfp8_e4m3.md) | `custom_dequant_mxfp8_e4m3_bf16` | 标准 E8M0 解码，输出 BF16 |
| 5 | [FP8 KV cache 量化写入](05_reshape_and_cache_kv_fp8.md) | `custom_reshape_and_cache_kv_fp8` | BF16 K/V 量化后写入分离式 FP8 values + FP32 scales 页 |
| 6 | [FP8 KV cache GQA Attention](06_flash_attn_varlen_gqa_fp8.md) | `custom_flash_attention_varlen_GQA_fp8` | 变长 GQA attention 直接消费分离式 FP8 values + FP32 scales 页 |

建议按 1+2、3+4 配对交付。两对共用本页的布局合同和 [CPU golden](../../../work-tests/fp8_kvcache/fp8_reference.py)。API/kernel 名为建议名称；改名须同步六份需求及接入测试，禁止仅凭 shape 自动识别布局。

05、06 使用当前 KV cache 的 FP32 scale 与同页分离 values/scales 布局，属于独立的写入与 attention 交付；其页布局和数值合同以两份 issue 正文为准，不能与 01–04 的 packed scale/MXFP8 布局互换。

## 交付范围与兼容性

- TYD 单卡、普通二维 Tensor、BF16 与 FP8 E4M3FN 之间的全 Tensor 转换；沿最后一维 D 分组。values 与 scale 是两个 Tensor。
- G128 的旧 `torch.dlc_quant_e4m3` / `torch.dlc_dequant_e4m3` 及其旧 kernel、逐行 scale ABI 和旧数值行为保留；新 G128 packed 入口同时修改布局与数值 recipe，避免静默影响权重量化及其他现有消费者。
- 新 G128 packed 与 group32 共用 OCP 推荐的 E4M3 scale 选取方法、RNE、SAT/OVF 和标准 E8M0 特殊值语义，只在量化分组大小上不同。G128 是采用这些数值规则的自定义 group128 格式，仍不能称为标准 MXFP8；标准 MXFP8 的 block size 为32。
- MXFP8 首批选择标准允许的 E4M3 格式子集；E5M2 不在本批内。支持 E4M3 的 RNE、SAT/OVF 转换与 E8M0 特殊值。不能仅把原 group128 参数改为 32 后宣称完成 MXFP8。
- 本批不实现 cache `slot_mapping`、scatter/gather、页 copy/COW、逐 token 原地追加、共享 tile 并发写、attention 融合或 FP8 values 数据区的重新打包。后续 KV cache writer/reader 必须另外约定页、head、slot 和所有权，不能直接将这四个转换算子视为完整 KV cache 接入。
- 首批要求支持动态 `0<=T<=4096, 0<=D<=8192`，包括尾 group/尾 tile。更大 shape 可以扩展，但应记录边界；元信息中的乘法/对齐先用 int64 校验，超出设备可用内存或实现索引上限应清楚报错。不得把所有 shape 的笛卡尔积都当成必测集合。

## 共享 scale 布局：TYD_SCALE_T128_G4_V1

这里的 `T128` 是 128 个 token 的存储块；`G4` 是每块 4 个 scale 列，不表示量化 group_size=4。量化粒度分别为 128 或 32。

```text
B  = group_size                     # G128:128；MXFP8:32
G  = ceil(D / B)                     # 每个 token 的逻辑 scale 数
Q  = ceil(G / 4)                     # 通道方向的 scale tile 数
NT = ceil(T / 128)
N  = NT * Q                         # T=0 或 D=0 时，N=0

values:      float8_e4m3fn[T,D]      # 普通 TYD Tensor，数据区布局不变
scale_tiles: uint8[N,512]            # 每一行是一个完整的 512 B scale tile

# 对有效 t in [0,T)、g in [0,G)：
tile = (t // 128) * Q + g // 4
k    = (t % 128) * 4 + g % 4
scale_tiles[tile,k] = scale_code[t,g]
```

`scale_code[t,g]` 是逻辑二维矩阵；`scale_tiles` 是实际传给新算子的紧凑容器。所有无效 token 或 scale 列的槽在量化输出时写 `0x00`，反量化依 T、D 忽略这些槽。`0x00` 是 padding 的存储值，不是数值零 scale；标准 E8M0 中它表示 `2^-127`。不用依靠该字节推断 token 是否有效。

本版本选择 tile 内的逻辑行序，未附加 NVIDIA 的 128×4 swizzle。TYD 普通 uint8 Tensor 自带字节交织，因此逻辑 `(tile,k)` 对应的真实 HBM 字节偏移是：

```text
physical_byte_offset = tile * 512 + 4 * (k % 128) + k // 128
```

依据为运行时的 uint8 原生映射，工作区源码位置：`/root/workspace/DLCSynapse/external_includes/kernel_arg_types.h:264`。CPU pack 只写逻辑 `scale_tiles[tile,k]`，H2D/D2H 负责原生布局转换，不可再次手工 weave。多 tile 在逻辑上顺序存放，每行末维恰为 512，消除原 `[T,G]` scale 逐 token 行 padding。

对 `T=128,D=512,B=128`，新缓冲区是 `[1,512]`。只分配 `[512]` 也能表示同样容量，但本 ABI 固定二维 `[N,512]`，方便表达多个 tile；对旧 `[128,4]` 做 `view(-1)` 不会形成该存储合同。

### 写入、并行与生命周期

量化调用独占完整输出 tile。每个 tile 必须有唯一 XYS writer；可按完整 128-token block 分工。若分多次 VMEM batch 处理同一 tile，不能反复清零或整块写回覆盖此前生成的 scale，须在唯一 owner 内正确累计、最后提交。不得让两个 XYS 各写一部分后相互覆盖。

本批量化每次生成完整 Tensor 的 scale_tiles，故可以统一写零 padding。后续逐 token append 需要单独的读改写/所有权设计，不能直接用本批“清零后写完整 tile”的 kernel 更新已有缓存。反量化可共享只读 tile，但禁止并发修改；普通 BF16 输出也须确保 DMA 写回范围不重叠。

所有接口使用调用方预分配输出、返回 `None`，正确登记 mutation schema 和 DLC dispatch；按本 fork 的 torchgen 规则完成绑定生成。错误 dtype/rank/shape/device/stride/storage offset/alias 在 launch 前拒绝；数据值检查不得偷偷触发 host 扫描。支持当前 stream、非默认 stream、graph capture/replay；跨 stream 依赖由调用方显式建立，禁止内部全设备同步替代依赖。

## 数值合同与标准边界

新 G128 packed 不再要求与 legacy 量化输出逐字节一致。建议对齐的内容作为本批新入口的实现与验收目标；旧入口独立做兼容回归。两种新入口采用数值 recipe `OCP_E4M3_RNE_SAT_OVF_V1`，物理 scale 布局仍是 `TYD_SCALE_T128_G4_V1`。物理布局标识不包含数值 recipe 或 group_size，调用方不能仅凭 shape/该标识推断它们；外部保存或交换缓存时，应同时记录group_size、layout版本及numeric_recipe，不能无标识跨入口复用。

| 项目 | 现有 G128 legacy，保留 | 新 G128 packed，建议目标 | MXFP8(E4M3)，group32 |
|---|---|---|---|
| 每个 scale 管理的元素 | 128 | 128，自定义格式 | 32，标准格式 |
| 有限非零块的指数 | `ceil(log2(max(amax/448,1e-10)))` | `clamp(floor(log2(amax))-8,-127,127)` | 与新 G128 相同 |
| 全零块 scale code | 94 | 127，保留各元素正负零 | 127，保留各元素正负零 |
| 量化 NaN/Inf 输入 | 原 API 行为保留，历史验收仅 finite | finite 元素归约 amax；NaN/Inf 逐元素按模式转换 | 与新 G128 相同 |
| scale=255 解码 | 现有 `exp2(128)=+Inf` 规则 | 整128元素组 NaN，先判断 | 整32元素组 NaN，先判断 |
| FP8 舍入/溢出 | 原实现行为 | RNE，SAT/OVF，默认SAT | 与新 G128 相同 |
| CPU golden | `legacy_g128_quant` / `legacy_g128_dequant` | `g128_quant` / `g128_dequant` | `mx_quant` / `mx_dequant` |

MXFP8 的数值格式采用 OCP MX v1.0 §5.2、§5.4.1，E4M3 编码采用 OFP8。两个新量化入口的有限块 scale 选取均采用 MX §6.3 推荐支持的转换语义：最大 2 的幂指数差；它不是标准唯一允许的 scale 算法。对极小 BF16 需将指数夹至 -127，移除旧 G128 的 `1e-10` 人为下限。支持 BF16 时量化实际产生code0…246，反量化仍需接受全部code0…255。

本项目为非有限块补充确定性约定：NaN/Inf 不参与 amax；全零或没有 finite 非零元素时 code127；NaN 元素输出 canonical `0x7f`；Inf 在 SAT 输出带符号448、在 OVF 输出NaN。有限值在 FP8 RNE 后按 SAT/OVF 处理溢出。这一归约扩展是本项目选择，不能说是 OCP 唯一指定。

必须用下列固定用例区分旧行为与新目标，并分别覆盖group128和group32：

| 用例 | 旧 G128 | 新 G128 packed / MXFP8 数值目标 |
|---|---|---|
| 组 `amax=480` | exponent1/code128，q240，恢复480 | exponent0/code127，SAT q448，OVF qNaN |
| 全零组 | code94 | code127，各元素±0符号保留 |
| 整组为 BF16 最小subnormal `2^-133` | scale下限使q归零 | code0、q=`2^-6`（0x08），恢复原BF16值 |
| 外部输入scale255、q=1 | 旧解码为+Inf | 整个有效group为NaN |
| 外部输入scale0/254 | 按旧实现独立回归 | 分别为有限scale `2^-127` / `2^127` |

这些修改统一数值合同，不保证所有输入的误差都下降。上表480用例中SAT会饱和而旧方法可精确恢复；必须分别报告旧G128、新G128、MXFP8的量化损失、饱和/新增零/NaN/Inf比例。新入口与对应新golden逐码比对；旧120用例可复用输入分布，但其原bytes、scale和误差统计不能直接作为新入口的期望值。

新 G128 和 MXFP8 的 scale0=`2^-127`、scale254=`2^127` 均为有限数；scale255=NaN。读取255时先将整个有效group标为NaN，不得先计算 `exp2(128)`，否则非零q会错误还原为Inf。其余code按 `BF16_RNE(FP32(decode(q) * scale))` 还原，保留subnormal、正负零。对外部任意FP8/scale组合超FP32/BF16可表示范围的行为，规范允许部分由实现确定，本项目约定IEEE溢出为带符号Inf；NaN验证mask，不强求payload。

## 容量验收口径

下列为默认 TYD 布局、非空二维 Tensor 的 storage 需求，allocator 分块/预留缓存可能另有开销：

```text
BF16 values bytes = T * ALIGN256(D) * 2
FP8 values bytes  = T * ALIGN512(D)
legacy scale bytes= T * ALIGN512(ceil(D / B))
packed scale bytes= ceil(T/128) * ceil(ceil(D/B)/4) * 512
```

| T、D | BF16 缓存 | G128 packed values+scale | MXFP8 packed values+scale |
|---|---:|---:|---:|
| 128、512 | 128 KiB | 64 KiB+512 B=64.5 KiB，省49.609375% | 64 KiB+2 KiB=66 KiB，省48.4375% |
| 1、512 | 1024 B | 512+512=1024 B，持平 | 512+2048=2560 B，增加 |
| 128、128 | 64 KiB | 64 KiB+512 B=64.5 KiB | 64 KiB+512 B=64.5 KiB |

此表说明：尾 tile 或小 D 可能抵消收益。D=128 的 FP8 values 本身逐行填充到512 B，本批不解决它。理论每32元素33 B的48.4375%收益只在完整、合适尺寸的紧凑存储上成立，不能推广为所有 shape 的真实峰值降幅。

验收应分别测 BF16 baseline 与 FP8 values+scale 的独立分配，固定相同 T、D、device，warmup 后同步读取 allocator 的 `memory_allocated` 差值，核对实际 DLC HBM storage 字节；不得只看逻辑 `storage.nbytes()`、`numel*element_size` 或 `memory_reserved`。排除原 BF16 输入、完整恢复输出及 workspace 后才能称为持久缓存占用；转换峰值和时延另报。新量化直接输出 packed scale，不得额外常驻旧逐行 scale 副本。性能需提交相同条件下的时延/cycles/带宽与 profile，本批不设无实测依据的速度门限。

## 验证环境

撰写时源码 HEAD 如下，只用于定位需求基线，不表示这些 SHA 的组合已经完成新算子验收；未来报告应保存完整 SHA 和实际加载 kernel 库 SHA。

| 组件 | HEAD |
|---|---|
| LLVM | `2fb2abd9e629` |
| DLCsim | `5bfa8963893a` |
| DLCSynapse | `370fb6f8cf8f` |
| DLC_Custom_Kernel | `548925372b9a` |
| pytorch | `b7da0c10d444` |
| vllm | `85fd014d5ff4` |
| vllm-cl | `e148df907df3` |

硬件 TYD，8 卡环境中选择一张空闲卡；driver `Driver Version_v23.0.1`，CRT RTABI `3.0.0`。验收另记 device index、真实/模拟模式、Python/torch 版本及构建安装状态。使用工作区 `.venv`；`DLC_TPU_VERSION=2`，`LLVM_PATH`、`LD_LIBRARY_PATH`、`CUSTOM_KERNEL_PATH`、`DLC_SYN_USE_SIM`/`DLC_SYN_BLOCKING` 实际值随报告记录，非交互 shell 不应假设已读 `.bashrc`。硬件验收必须确认未启用模拟；模拟结果单独标明。vLLM 不参与本批独立转换算子的必测链路。

已有 [legacy G128 报告副本](baseline_g128_report.json)：mode=`dlc:0`、120个往返+13个接口检查通过、failures=0。报告没有运行时模式/完整版本指纹，也没有显存实测；它仅证明该次旧入口测试的结果，不能替代 packed/MXFP8 验收。

## Reference 与检查

[fp8_reference.py](../../../work-tests/fp8_kvcache/fp8_reference.py) 只在 CPU 运行，提供 `pack_scales`、`unpack_scales`、独立 FP8 decoder、对齐后新 G128 的 `g128_quant`/`g128_dequant` 与 group32 的 `mx_quant`/`mx_dequant`；`legacy_g128_quant`/`legacy_g128_dequant` 单独保留原规则，用于历史回归与差异报告。其中 `bits` 是 CPU uint8 编码；设备接口的 values dtype 为 `float8_e4m3fn`。先解包后按逻辑位置比对，不能把量化和反量化互相抵消的往返结果当作唯一 oracle。

```bash
cd /root/workspace
uv run --no-sync --project /root/workspace /root/workspace/.venv/bin/python \
  work-tests/fp8_kvcache/fp8_reference.py --self-test
```

CPU golden 的 self-test 通过不代表设备实现交付。设备测试必须独立核对 FP8 bytes、有效 scale codes、BF16 bits/Inf sign/NaN mask、padding、双 XYS 所有权、graph/stream、legacy 回归以及真实 storage 分配。

官方依据：

- [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：§5.1不限定物理排布，§5.2格式子集与group32，§5.4.1 E8M0，§6.3转换。
- [OFP8 v1.0（2023-12-01勘误）](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1)：E4M3与RNE、SAT/OVF。
- [NVIDIA 128×4 scale layout](https://nvidia.github.io/cudnn-frontend/mxfp8-scale-factor-128x4-layout/)：仅参考“跨行聚合scale”的设计思想；本 TYD ABI 不宣称与 NVIDIA swizzle 字节互通。

本页与 01–04 四份转换算子 issue 明确取代早期固定 per-layer FP32 scale 草案中与本批转换算子冲突的假设；早期完整 KV cache/attention 方案仍需独立修订和验收，不能由这里的 scale 布局直接推导模型可用性。
