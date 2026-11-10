# GLM-5.3-Flash Sparse MLA FP8 KV Cache 预研与实施契约

日期：2026-10-27。状态：基础数值/ABI 与 PyTorch 分离布局已真机验证；
融合分页 writer/reader 和 vLLM 端到端接入尚未实现，不能把本文当作已完成
验收结论。

## 1. 结论先行

- 当前 `main` 已有 BF16 group128 量化到 E4M3FN 的算子和两套反量化 ABI；
  `main` 另已补上与本轮 FP32 scale 量化严格配对的
  `custom_dequant_e4m3_g128_scale_fp32_out_bf16`。
- FP8 元素的数制是标准 OCP `E4M3FN`，不是私有位格式：1 byte，
  S1/E4/M3，bias 7，RNE，无 Inf，最大有限值 448（`0x7e`），NaN 为
  `0x7f`，支持正负零。已有 device quant 结果与独立 CPU golden 逐字节一致。
- “自研架构特殊”主要不在元素编码，而在 group128 scale ABI、DLC 的
  一字节 tensor 物理 weave、以及缺少 paged KV writer/reader。
- 第一版范围应是 GLM-5.3 的 11 个 sparse MLA 层，cache 语义宽度为
  `D=512`、NoPE、block size 256、每个 token 4 个 FP32 group scale。
  KDA state、DCP/PCP、KV transfer 不作为第一版范围。
- **最终布局已确定为分离式双 tensor**：values
  `[num_blocks, block_size, 512]` FP8，scales 为一维
  `[num_blocks * block_size * 4]` FP32。不再以 528B raw flat 为主方案。
  一维 scale 必须作为一个整体分配，禁止 reshape/分配成 `[slots, 4]`。
- GLM 生产实现在 `release-glm53flash`，不在当前 `main`。数值、ABI 和
  allocator 基础可以先在 `main` 做，真正的模型闭环必须在 release 线对应
  `DLCMLASparseBackend` / `DLCMLASparseImpl` 上完成。

## 2. 当前算子与 ABI

### 2.1 BF16 -> E4M3FN

`torch.dlc_quant_e4m3(input, output, scale, group_size=128, scale_round=...)`
当前接受两种 scale ABI。

legacy uint8 scale：

- 输入 BF16 2D，输出 E4M3FN 2D，scale 为按 token 行的 uint8 code。
- `scale = 2**ceil(log2(max(abs(x)) / 448))`，下限 `1e-10`，code `0..255`。
- 反量化按 `2**(code-127)` 解码。
- scale 行存在 `ALIGN512(ceil(D/128))` 的物理组织开销，不适合作为新 KV
  cache 的主 ABI。

FP32 scale：

- 输入 BF16 2D，输出 E4M3FN 2D，scale 为扁平 FP32
  `[M * ceil(D/128)]`，没有逐 token 的 uint8 scale 行 padding。
- amax 排除 Inf，下限 `1e-4`；归一化值 clamp 到 `[-448, 448]`。
- `scale_round=True` 时 scale 向上取 2 的幂；`False` 时保留原始 FP32
  scale。两种模式都已逐 bit 验证。

### 2.2 E4M3FN -> BF16

`torch.dlc_dequant_e4m3(input, scales, group_size=128)` 和 `.out` 变体：

- input：2D contiguous `float8_e4m3fn`；
- scales：1D contiguous FP32 `[M * ceil(K/128)]`；
- output：与 input 同 shape 的 2D contiguous BF16；
- 与 FP32 scale quant 逐 token/group 配对，乘积和 BF16 舍入在真机
  bit-exact。

已有 legacy uint8 scale dequant 仍保留用于旧调用，不应与新 KV cache ABI
混用。

## 3. DLC 物理布局约束

int8/uint8/FP8 的 tensor 逻辑 dim0 按 512 个一字节元素对齐。一个 512B
tile 被拆成四个 128B plane，再按 32-bit word 交织：

```text
tile = logical_offset_in_row // 512
k = logical_offset_in_row % 512
physical_in_row = tile * 512 + 4 * (k % 128) + k // 128
row_pitch = ceil(dim0 / 512) * 512
```

PyTorch DLC 的逻辑 view/copy 会处理 weave；直接按 HBM 地址做 DMA 的 custom
kernel 必须自己遵守该式。已有 `indexer_k_quant_and_cache_bf16` 的
`store_u8_logical` / `store_fp8_group` 可以作为字节写回参考；该 kernel 的
`ALIGN512(cache_stride)` row stride 也确认了 516B 行在 HBM 中按 1024B
行距写入。

对 GLM 的 `D=512`，值 cache 的逻辑最后维正好是一整个 512B tile，因此：

```text
BF16 value: 512 * 2 = 1024 B/token
FP8 value:  512 * 1 =  512 B/token
FP32 scale: 4 groups * 4 = 16 B/token
FP8 total:  528 B/token
节省:       496 / 1024 = 48.44%
```

但这个收益只有在 scale 使用扁平布局时才成立。若把 FP32 scale 存成普通
2D `[slots, 4]`，DLC 会把 dim0=4 补到 128 个 FP32，即每 token 512B，
总占用反而达到 1024B/token，和 BF16 相同。

两种内联行宽必须区分：legacy uint8 scale 的完整逻辑行是 **516B**
（512B FP8 values + 4B uint8 scale）；若把当前 FP32 scale 内联，完整
逻辑行则是 **528B**（512B FP8 values + 16B FP32 scale）。把任一行宽直接建成
`[blocks, block_size, 516]` 或 `[blocks, block_size, 528]` 的一字节 tensor，
DLC 都会把最后一维从 512 以上的非对齐宽度补到 1024，即实际
1024B/token。516 只对应 legacy 调用，不能与 FP32 scale ABI 混用。

真机 `data_ptr()` 间隔已实测：`[blocks,256,512]` FP8 为 512B/token、
block stride 131072B；`[blocks,256,516]` 和 `[blocks,256,528]` FP8 均为
1024B/token、block stride 262144B（逻辑 `nbytes()` 分别是 528384B 和
540672B，但两者实际 HBM 都是 1048576B）；`[slots,4]` FP32 scale 为
512B/token；一维 FP8 和 `[4,slots]` FP32 scale 均为紧凑连续布局。最终
scale 采用 token-major 的一维 `[slots * 4]`，便于每个 token 的四个 group
scale 连续写入。

### 3.1 最终选择：分离式双 tensor

主线固定为两个逻辑 tensor：

```text
values: [num_blocks, block_size, 512]       torch.float8_e4m3fn
scales: [num_blocks * block_size * 4]       torch.float32
```

`values` 的 token pitch 为 512B，block pitch 为
`block_size * 512B`；`scales` 的 element pitch 为 4B、token pitch 为 16B。
GLM block size 为 256 时，每 token 共 528B，相对 BF16 的 1024B 节省
48.4375%。attention 层必须显式接收 values 和 scales 两个 cache，不能把
它伪装成普通 `MLAAttentionSpec` 单 tensor。

选择分离布局而不是 528B raw flat 的原因：

- 两个 tensor 的 shape/stride 都是 PyTorch 可表达的普通连续布局，allocator、
  zero、block copy、调试和按 dtype 校验都更直接；
- 一维 scale 已实测是紧凑布局，不需要自定义 raw ABI；
- raw 528B 需要 allocator/reshape、block zeroer、prefix-cache copy 和所有
  kernel 共同维护非标准 row stride，首版复杂度和误用风险更高。

528B raw flat 仅保留为容量极限备选，不作为当前实现目标。

不建议把 scale 存成 `[slots, 4]`，也不要直接复用 legacy uint8 scale 的
512B/token 行组织来宣称省了 48%。

## 4. 还缺的算子

### 4.1 Paged writer

建议接口（名称可调整）：

```text
bf16 token rows [T, 512]
    -> per-128 group FP8 values
    -> FP32 scales [T, 4]
    -> scatter to slot_mapping
```

约束：

- K 已是模型最终的 NoPE latent，writer 不再做 RoPE；
- slot mapping 为 int64，所有 `slot * row_bytes` 使用 64-bit 地址运算；
- 负 slot no-op；
- 支持 prefill 多 token、decode 单 token、尾部无效 token；
- scale 与值必须由同一个 kernel 写入，避免两者在异常退出时失配；
- 对 zero-size 和全负 slot 有明确 no-op 语义。

不能只组合现有 batch quant 再在 Python 中做 gather/scatter：这样会引入
临时 FP8/BF16 tensor、额外 H2D 元数据和多次 launch，在 decode 每步都不可
接受。现有 batch quant/dequant 适合 golden、离线转换和开发 bring-up。

### 4.2 Paged reader

有两条路径：

1. 正确性优先：paged gather + per-group dequant 到 BF16 workspace，再调用
   现有 BF16 sparse MLA kernel。实现简单，但每次 attention 都额外生成
   `[Q, topk, 512]` BF16，内存带宽和 VMEM 压力较大。
2. 性能路径：在现有 `dlc_sparse_attention_kv_cache` / `sparse_attention_nope`
   读取 top-k cache row 的位置直接解码 E4M3FN、乘 FP32 scale、转 BF16，
   然后进入原 QK/PV 流程。它需要覆盖 GLM 的 head 数、topk 2048/2176 和
   512 宽度组合，是第一版最终应走的路径。

reader 必须把 top-k 的 `-1`/越界项保持为 masked，不能因为读取 dummy slot
而引入 NaN 或污染 softmax。

## 5. vLLM DLC 接入点

当前 `vllm-cl/main`：

- `platform.py::supports_fp8()` 返回 `False`；
- sparse MLA 的 `supported_kv_cache_dtypes` 只有 `auto/bfloat16`；
- `concat_and_cache_mla_dlc` 忽略 `kv_cache_dtype` 与 scale；
- sparse MLA 只有 `_forward_bf16_kv`；
- `csrc/cache.cpp::reshape_and_cache` 强制 scale 为 1 且只走 BF16/FP32；
- `gather_and_maybe_dequant_cache` 固定 BF16 kernel。

实施顺序：

1. 先定义并测试分离式 cache ABI、writer 和 reader，不改 runner；
2. 为 GLM sparse MLA spec 增加专用 FP8 cache 类型/属性，不把平台所有 FP8
   compute 能力直接声明为 True；
3. 覆盖 KVCacheSpec 的 dtype、`real_page_size_bytes`/DLC padded page size、
   allocator reshape 和 block dim；
4. 覆盖初始化/生命周期：`zeros/zero_`、block zero、copy blocks、prefix
   cache、sleep/wake、KV connector、DCP 拒绝路径；
5. 接入 prefill eager writer 和 decode graph writer/reader；
6. 先在 release 线跑 GLM 端到端，再回移可复用基础。

## 6. 已完成的基础验证

- DLC FP8 `zeros/full/fill_/zero_`、rank-0、单元素、NaN、正负零、切片
  邻字节保护、转置 view、deterministic empty：`19/19 PASS`。
- DLC FP8/scale 物理地址步长：`7/7 PASS`，确认 legacy 516B 与 FP32
  528B 三维行都会被补到 1024B/token，并确认 `[slots,4]` scale 的
  padding 风险。
- FP32 scale quant -> dequant：50 个数值 case + 8 个 API case，
  共 `58/58 PASS`，FP8 字节、FP32 scale bits、BF16 输出 bits 独立检查。
- FP32 scale dequant syntest：`30/30 PASS`，0 ULP。
- PyTorch kernel test：`31/31 PASS`。
- PyTorch 分离布局 helper 与真机测试：`6/6 PASS`。新增
  `pytorch/torch/dlc/fp8_kv_cache.py` 和
  `pytorch/test/dlc_ops/test_fp8_kv_cache.py`，覆盖：
  - 精确 shape/stride/token pitch/block pitch，以及 528B/token、48.4375% 节省；
  - 拒绝 `[blocks, block_size, 516/528]` 内联行和
    `[slots, 4]`/其 flatten view 的 512B/token 物理 padding；
  - FP8 `zeros/fill_/zero_`、整 block copy/zero、逐 bit 校验；
  - BF16 -> FP32-scale FP8 quant -> 跨 block/乱序/重复/负 slot scatter ->
    gather -> dequant，与批量 golden 逐 bit 一致；
  - 空 token、越界 slot 在写入前拒绝。
- 通用 PyTorch FP8 `index_copy_`/`scatter_` 在当前 DLC 后端会落到不支持
  E4M3FN 的 CPU kernel；高级索引赋值和 `index_select` 虽可能通过 CPU
  回退跑通，但同样不能进入生产路径。分离布局 helper 因此不使用这些原语，
  当前参考 writer/reader 分别使用逐 token 设备 `copy_`，只用于正确性
  bring-up，不是 decode 性能路径。

这些结果只证明基础数值、PyTorch 接口和 FP8 初始化可用，不证明 paged
cache writer/reader、vLLM runner 或 GLM 服务闭环已完成。

## 7. 下一里程碑

最小可验收切片：

1. ~~`D=512`、block 256、16-byte/token FP32 scale 的分离布局 ABI 测试~~；
2. ~~BF16 -> FP8 + scale -> slot scatter -> gather -> dequant 的 device roundtrip~~；
3. ~~跨 block、负 slot、tail、重复 slot、block copy、cache zero 测试~~；
4. 实现融合 paged writer：一次 kernel 完成 group quant、values slot 写入和
   扁平 FP32 scale 写入，负 slot no-op，地址运算全程 64 位；
5. 实现 paged reader/fused dequant，替换当前逐 token `copy_` 参考路径；
6. 在 release 线用 11 个 sparse MLA 层跑单请求与 8 并发，并与 BF16 cache
   同输入比较输出；
7. 最后再测 graph replay、prefix cache 和容量/吞吐。
