# [Operator Request] FP8 KV cache GQA 变长 Flash Attention 算子

## 需求类型

扩展已有算子能力。

## 优先级

P0。

## 目标版本与期望时间

vLLM DLC FP8 KV cache 第一阶段；该算子阻塞 FP8 cache 的生产 attention 读取路径。具体交付版本与日期由算子组和接入方共同排期。

## 算子名称

- 建议 PyTorch 接口：`torch.dlc_flash_attn_varlen_gqa_fp8`。
- 建议 kernel 名称：`custom_flash_attention_varlen_GQA_fp8`。
- 本需求只包含一个 attention kernel：消费分离式 FP8 values + FP32 scales cache，完成变长 GQA attention；不包含 cache 写入、gather、block copy 或 block zero。

## 背景与功能说明

使用场景：

- Qwen3 类 GQA 模型，`head_size=128`，TP4 时每 rank 有 16 个 query heads 和 2 个 KV heads，每个 KV head 服务 8 个 query heads。
- 持久化 KV cache 使用分离布局：FP8 E4M3FN values 与逐 token、逐 KV head 的 FP32 scales 位于同一 raw page 的不同区域。
- `block_size=256`、2 个 KV heads 时，每个 page 为 135168 B：131072 B values 和 4096 B scales。
- decode 使用 query length 1；chunked prefill 使用 query length 大于 1 且可能小于累计 context length；一个批次可同时包含多个变长请求。

当前问题：

- 现有 DLC 变长 attention 只消费 BF16 K/V cache，不能读取 FP8 values 与独立 FP32 scales。
- 现有可选全局 descale 参数不能表示逐 token、逐 KV head 的 scale 数组，也无法表示 V scale 随 key/token 位置变化。
- 如果先用独立算子把整个序列 gather 并反量化到 BF16 HBM，再调用 attention，会产生与上下文长度成正比的额外 HBM 工作区和读写，不满足生产路径要求。

目标功能：

- 从一个 raw FP8 cache page 直接读取 values 和 scales，在设备内完成逐 token、逐 head 反量化并执行 GQA attention。
- 第一阶段允许把 K/V 在 VMEM 中反量化为 BF16，再复用现有 BF16 QK、softmax、PV 和输出更新数值路径；不得把完整 BF16 K/V 序列写回 HBM。
- 支持 varlen batch、decode、chunked prefill、causal、非连续物理 block table、`seqused_k`、可选 softmax LSE、双 XYS 和 graph replay。

非目标：

- 不要求原生 FP8 GEMM/FP8 MMA；若后续实现直接使用 FP8 或混合精度矩阵乘，必须保持本文数值语义并单独验收。
- 不量化 query，不引入 query scale，不改变 BF16 输出 dtype。
- 不支持 merged K/V cache、MLA、Mamba/KDA state、sliding window、ALiBi 或 logits softcap。
- 不实现 cache write、cache gather、block copy、swap 或 zero；这些由独立算子负责。

## 期望 API 与函数签名

```python
def dlc_flash_attn_varlen_gqa_fp8(
    q: torch.Tensor,              # [Tq, Hq, 128], bfloat16
    kv_cache: torch.Tensor,       # [num_blocks, page_bytes], float8_e4m3fn
    out: torch.Tensor,            # [Tq, Hq, 128], bfloat16
    softmax_lse: torch.Tensor,    # [Hq, Tq, 1], float32
    cu_seqlens_q: torch.Tensor,   # [B + 1], int32
    seqused_k: torch.Tensor,      # [B], int32
    block_table: torch.Tensor,    # [B, max_blocks], int32
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    block_size: int,
    num_kv_heads: int,
    return_softmax_lse: bool,
) -> None:
    ...
```

建议 C++ 签名为：

```cpp
void dlc_flash_attn_varlen_gqa_fp8(
    const at::Tensor& q,
    const at::Tensor& kv_cache,
    at::Tensor& out,
    at::Tensor& softmax_lse,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqused_k,
    const at::Tensor& block_table,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double softmax_scale,
    bool causal,
    int64_t block_size,
    int64_t num_kv_heads,
    bool return_softmax_lse);
```

返回值为 `None`。`out` 与 `softmax_lse` 标记为 mutation；`q`、`kv_cache` 和 metadata 只读。

## 参数与返回值

输入：

- `q`：BF16 query，逻辑 shape `[Tq, Hq, 128]`。
- `kv_cache`：二维 raw FP8 cache，逻辑 shape `[num_blocks, page_bytes]`，dtype 为 `float8_e4m3fn`。
- `cu_seqlens_q`：INT32，shape `[B + 1]`，从 0 开始且末尾等于 `Tq`。
- `seqused_k`：INT32，shape `[B]`，每请求有效 context token 数。
- `block_table`：INT32，shape `[B, max_blocks]`，每项为物理 block id。
- `max_seqlen_q`：Python `int`，本批次最大 query length。
- `max_seqlen_k`：Python `int`，本批次最大有效 context length。
- `softmax_scale`：FP32 标量，乘在 QK 点积结果上。
- `causal`：Python `bool`，是否应用 causal mask。
- `block_size`：Python `int`，每个 page 的逻辑 token 数。
- `num_kv_heads`：Python `int`，每 rank 的 KV head 数。
- `return_softmax_lse`：Python `bool`，是否写 `softmax_lse`。

输出：

- `out`：BF16 attention 输出，shape `[Tq, Hq, 128]`。
- `softmax_lse`：FP32 natural-log logsumexp，shape `[Hq, Tq, 1]`；仅在 `return_softmax_lse=True` 时写入。
- 函数返回 `None`。

## 功能语义

### Cache page 布局

固定：

```text
head_size D = 128
group_size G = 128
B = block_size
H = num_kv_heads

token_value_bytes   = 2 * H * 128
token_scale_bytes   = 2 * H * 4
values_region_bytes = B * token_value_bytes
scales_region_bytes = B * token_scale_bytes
page_bytes          = values_region_bytes + scales_region_bytes
```

一个 page 的字节布局：

```text
[ values: B * token_value_bytes ][ scales: B * token_scale_bytes ]
```

第 `s` 个 token（`0 <= s < B`）的 values 和 scales 地址：

```text
values_base = block_id * page_bytes + s * token_value_bytes
K head h    = values_base + h * 128
V head h    = values_base + H * 128 + h * 128

scales_base = block_id * page_bytes + values_region_bytes + s * token_scale_bytes
sK(t,h)     = scales_base + h * 4
sV(t,h)     = scales_base + (H + h) * 4
```

逻辑顺序固定为：

```text
values: [K0, K1, ..., K(H-1), V0, V1, ..., V(H-1)]
scales: [sK0, sK1, ..., sK(H-1), sV0, sV1, ..., sV(H-1)]
```

`B=256,H=2` 时：

```text
token_value_bytes   = 512
token_scale_bytes   = 16
values_region_bytes = 131072
scales_region_bytes = 4096
page_bytes          = 135168
```

### 数值语义

令 q head `g` 对应的 KV head 为：

```text
h = g // (Hq // H)
```

对请求 `b` 的第 `t` 个有效 key，从 `block_table[b, t // B]` 的 `t % B` slot 读取：

```text
K_f32(t,h,d) = decode_e4m3fn(K_fp8(t,h,d)) * sK(t,h)
V_f32(t,h,d) = decode_e4m3fn(V_fp8(t,h,d)) * sV(t,h)
```

第一阶段要求先把 K/V 按上述公式反量化为 BF16，再进入 QK、softmax 和 PV；反量化后的 BF16 转换必须使用 round-to-nearest-even。Q 保持 BF16。

attention 公式：

```text
score(i,t,g) = dot(Q(i,g), K(t,h)) * softmax_scale
score(i,t,g) = -Inf                                      if causal 且 t > query_position(i)
p(i,t,g)     = exp(score(i,t,g) - LSE(i,g))
LSE(i,g)     = logsumexp_t(score(i,t,g))
O(i,g,d)     = sum_t p(i,t,g) * V(t,h,d)
```

- 所有 score、max、exp、求和、LSE 和输出累加使用 FP32。
- K scale 在每个 key token 的 QK 结果中生效；V scale 在每个 key token 的 PV 贡献中生效。
- 不得把 V scale 移到整个输出之后当作全局因子；V scale 随 token 位置变化，不能从加权和中提出。
- 不得把 `softmax_scale` 当作 K scale 的替代；K scale 是逐 token、逐 KV head 的数组。
- `softmax_lse` 使用自然对数。
- 输出执行 FP32 到 BF16 round-to-nearest-even。

### Causal 与位置

对请求 `b`：

```text
q_len = cu_seqlens_q[b+1] - cu_seqlens_q[b]
k_len = seqused_k[b]
query_position(i) = k_len - q_len + i
```

- 每个请求必须满足 `q_len <= k_len`；不满足时在 launch 前报错。
- `causal=True` 时，key position `t` 仅当 `t <= query_position(i)` 时可见。
- `q_len=1` 时，该 query 可见全部 `k_len` 个 key，不需要额外 mask。
- `q_len=k_len` 时退化为标准下三角 causal mask。
- `q_len<k_len` 时对应 chunked prefill，只允许看到当前 query 及其之前的历史。
- `causal=False` 时全部有效 key 可见。
- `k_len=0` 或没有可见 key 的行，`out` 和 `softmax_lse` 必须为 NaN，不能静默写成 0。

### 分页读取

- 第 `t` 个逻辑 key 的物理 block 为 `block_table[b, t // block_size]`，page 内 slot 为 `t % block_size`。
- `block_table` 不要求连续、递增或从 0 开始；必须支持任意合法物理页顺序。
- 最后一个物理 page 只读取前 `k_len % block_size` 个 slot；整除时读取完整 page。
- 所有 page offset、slot offset、scale offset 和 DMA 地址计算必须使用 64 位整数。
- 不得在 HBM 中构造完整 BF16 K/V 序列，也不得逐请求启动独立 attention。
- 不得在 Python 或 host 端逐请求读取 block table、构造 mask 或做 host sync。

## 支持范围与约束

| 项目 | 首期范围 |
|---|---|
| 设备 | DLC/TYD，真机 |
| query dtype | BF16 |
| output dtype | BF16 |
| cache values | FP8 E4M3FN raw bytes |
| cache scales | 页内逐 token、逐 head FP32 |
| head_size | 固定 128 |
| group_size | 固定 128 |
| KV heads | 首期强制验收 `H=2`；实现支持 `H∈{1,2,4,8}` |
| query heads | 首期强制验收 `Hq=16,Hq/H=8`；支持整除关系 |
| block_size | 首期强制验收 256；支持 256、512、1024 |
| batch | `1 <= B <= 256`，动态 |
| query length | `0 <= q_len`，单请求最大 4096 为必测 |
| context length | 必测 1、255、256、257、8192、32768；ABI 与地址计算支持 131072 |
| 总 query token | `0 <= Tq <= 65536` |
| metadata dtype | `cu_seqlens_q`、`seqused_k`、`block_table` 为 INT32 |
| Tensor 布局 | 全部 contiguous，`storage_offset=0` |
| device 布局 | 所有 Tensor 位于同一 DLC device |
| 执行模式 | 普通 stream、graph capture/replay、双 XYS |
| 多卡 | 每个 TP rank 独立执行，不需要跨卡同步 |

约束：

- `Hq` 必须能被 `num_kv_heads` 整除，且 `1 <= Hq / H <= 8`。
- 每个请求必须满足 `q_len <= k_len`；该条件不满足时必须在 launch 前报错，不得进入 causal 或 non-causal 计算。
- `block_table` 的宽度必须至少为 `ceil(max_seqlen_k / block_size)`。
- `seqused_k[b]` 必须不超过 `block_table` 可表示的有效 capacity。
- `page_bytes` 必须严格等于 `block_size * (2*H*128 + 2*H*4)`。
- query/out shape、cache page、head 数、block_size 或 dtype 不一致时必须在 launch 前报错。
- `window_size`、ALiBi 和 softcap 不在本接口中；调用方不得依赖隐式支持。
- 首期不要求尾页填充区的 values/scales 为 0；只能读取 `[0, seqused_k[b])` 范围内的 slot。

## 精度要求

Golden/reference 实现：

- 使用本文末尾自包含 CPU reference。
- 先将 cache page 解码为 BF16 K/V，其中 FP8 解码、FP32 scale 乘法和 BF16 RNE 必须与写入算子约定一致。
- 使用 FP32 score、softmax、LSE 和 PV 累加得到 reference，再转 BF16 输出。
- 测试必须使用逐 token、逐 KV head 不同的 scales，避免相同 scale 掩盖索引错误。

比较标准：

- `out` 为 BF16：`atol=2e-2`、`rtol=2e-2`。
- 当 `return_softmax_lse=True` 时，LSE 为 FP32：`atol=1e-2`、`rtol=1e-2`。
- NaN 使用 mask 比较；完全 masked 行必须两边同为 NaN。
- 不允许通过忽略 NaN 或把 NaN 替换为 0 使测试通过。

数据分布：

- query 标准正态、全零、常量、大值和极小值。
- cache values 覆盖 FP8 正负零、subnormal、最小 normal、448、NaN code、随机有限值。
- scales 覆盖正常范围、跨数量级、极大正值，并确保乘积在 FP32 有限范围内。
- 至少 100 组随机 seed；固定 seed 保留为回归 golden。

中间精度：

- 反量化和 scale 乘法使用 FP32，随后 BF16 RNE。
- QK、online softmax、LSE、PV 和输出归一化使用 FP32 累加。
- 不允许把概率矩阵提前截断为 FP8。

## 测试范围

- [x] 基本功能与 reference 对齐
- [x] FP8 cache 与 BF16 中间精度
- [x] 静态 shape 与动态 shape
- [x] 最小值、最大值、空 tensor 等边界
- [x] 非连续物理 block 顺序、尾页与特殊 layout
- [x] 非法参数和异常处理
- [x] 单卡
- [ ] 多卡/分布式（逐 rank 独立）
- [x] 长时间稳定性
- [x] 性能基准与性能回归

具体测试用例：

1. `B ∈ {1,2,8,64,256}`；每请求 `q_len` 覆盖 0、1、2、127、255、256、257、4096。
2. `k_len` 覆盖 0、1、127、255、256、257、511、512、513、8192、32768；条件允许时增加 131072。
3. 同一批次混合 decode、full prefill 和 chunked prefill，例如 `q_len` 为 `[1,2,127,256]`、`k_len` 为 `[8192,257,1024,256]`。
4. `causal=True/False`；检查 `q_len=1`、`q_len=k_len`、`q_len<k_len` 和全 masked 行。
5. `block_table` 使用乱序、逆序、间隔和不连续物理页；最后一页分别覆盖 1、255、256、257 个有效 token。
6. GQA 比例覆盖 1、4、8；强制基线为 `Hq=16,H=2,numq_per_kv=8`。
7. 使用 one-hot query、单位 K 和刻意构造的 V，分别暴露 K scale、V scale、head 映射和 token 映射错误。
8. cache 尾页填充使用 poison，确认 attention 不读取 `seqused_k` 之后的 slot。
9. `return_softmax_lse=False` 时不写 LSE，也不得读取未分配的 LSE 缓冲区；`True` 时形状和数值正确。
10. graph capture/replay 后更新 `cu_seqlens_q`、`seqused_k`、`block_table` 和 cache 内容，必须使用最新 device 数据。
11. 非默认 stream、双 XYS，以及连续 1000 次交替 shape 的稳定性测试。
12. 非法 dtype、形状、page_bytes、head 比例、block_size、`q_len > k_len`、metadata 越界和 storage offset 必须有明确错误。
13. 性能报告记录 launch 数、decode 与 chunked-prefill 时延、双 XYS 使用、scale DMA 次数和相对 BF16 attention 的开销；首期不以绝对速度作为阻断阈值，但不得出现完整 BF16 cache 物化、host 循环或逐请求多次 launch。

## 验证环境与依赖版本

- DLC 硬件：8 卡 TYD 环境，选择空闲单卡验证并记录 device index。
- `DLC_TPU_VERSION=2`，必须使用真机，不得以模拟器结果代替。
- 验收报告记录 PyTorch、DLC_Custom_Kernel、LLVM、DLCSynapse、DLCsim 的完整 Git SHA 和工作区状态。
- 记录 driver、RTABI、kernel 共享库 SHA256、`CUSTOM_KERNEL_PATH`、`LD_LIBRARY_PATH` 和 `DLC_SYN_*` 实际值。
- 新增 Python/C++ 入口与 kernel 必须完成构建、安装、注册和加载。
- 集成环境使用 Qwen3-32B-Mix、TP4、`block_size=256`；算子单元测试不得依赖模型运行才能判定通过。

## 验收标准

- [ ] Python/C++ API、mutation schema、参数校验和错误行为符合约定。
- [ ] 正确读取分离式 FP8 values 与 FP32 scales，kv page、K/V 顺序、head 映射和 token 映射均正确。
- [ ] `H=2,B=256` 的 135168 B page 可被正确消费；131072 B values 和 4096 B scales 的偏移无歧义。
- [ ] decode、full prefill、chunked prefill、混合 varlen batch、causal 和非 causal 均与自包含 reference 对齐。
- [ ] 输出满足 `atol=2e-2, rtol=2e-2`；LSE 满足 `atol=1e-2, rtol=1e-2`；全 masked 行 NaN 语义一致。
- [ ] 乱序或非连续 block table、尾页、极端 scale、FP8 特殊值和不同 GQA 比例测试通过。
- [ ] 不在 HBM 中生成完整 BF16 K/V cache，不逐请求 launch，不触发 host sync。
- [ ] 支持 graph replay、非默认 stream 和双 XYS；大地址计算无 32 位溢出。
- [ ] 自动测试、调用示例、精度报告、性能报告和限制说明已提交。

## Reference、日志与附件

本需求不依赖外部代码、外部文档、日志或附件。以下 CPU reference 是完整、自包含的 cache 解码与 GQA attention golden；仅依赖 CPU PyTorch，且不调用待测 DLC 算子。

```python
import torch


def reference_flash_attn_varlen_gqa_fp8(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    block_size: int,
    num_kv_heads: int,
):
    del max_seqlen_q, max_seqlen_k
    assert q.dtype == torch.bfloat16
    assert q.ndim == 3 and q.shape[2] == 128
    assert kv_cache.dtype == torch.float8_e4m3fn and kv_cache.ndim == 2

    tokens, num_q_heads, head_size = q.shape
    num_kv_heads = int(num_kv_heads)
    block_size = int(block_size)
    assert head_size == 128
    assert num_q_heads % num_kv_heads == 0

    token_value_bytes = 2 * num_kv_heads * head_size
    token_scale_bytes = 2 * num_kv_heads * 4
    values_region_bytes = block_size * token_value_bytes
    page_bytes = values_region_bytes + block_size * token_scale_bytes
    assert kv_cache.shape[1] == page_bytes

    out = torch.empty_like(q)
    lse_out = torch.full((num_q_heads, tokens, 1), float("nan"), dtype=torch.float32)

    def decode_slot(block_id: int, slot: int):
        raw = kv_cache[block_id].view(torch.uint8)
        value_start = slot * token_value_bytes
        scale_start = values_region_bytes + slot * token_scale_bytes

        k_bytes = raw[value_start:value_start + num_kv_heads * head_size]
        v_start = value_start + num_kv_heads * head_size
        v_bytes = raw[v_start:v_start + num_kv_heads * head_size]
        scale_bytes = raw[scale_start:scale_start + token_scale_bytes]

        k = k_bytes.view(torch.float8_e4m3fn).to(torch.float32).reshape(
            num_kv_heads, head_size
        )
        v = v_bytes.view(torch.float8_e4m3fn).to(torch.float32).reshape(
            num_kv_heads, head_size
        )
        scales = scale_bytes.view(torch.float32)
        k = (k * scales[:num_kv_heads, None]).to(torch.bfloat16)
        v = (v * scales[num_kv_heads:, None]).to(torch.bfloat16)
        return k, v

    num_requests = int(seqused_k.numel())
    for request in range(num_requests):
        q_start = int(cu_seqlens_q[request].item())
        q_end = int(cu_seqlens_q[request + 1].item())
        q_len = q_end - q_start
        k_len = int(seqused_k[request].item())
        assert 0 <= q_len <= k_len
        if q_len == 0:
            continue

        keys = []
        values = []
        for token in range(k_len):
            block_id = int(block_table[request, token // block_size].item())
            key, value = decode_slot(block_id, token % block_size)
            keys.append(key)
            values.append(value)

        if k_len == 0:
            out[q_start:q_end].fill_(float("nan"))
            lse_out[:, q_start:q_end, :].fill_(float("nan"))
            continue

        keys = torch.stack(keys)
        values = torch.stack(values)
        q_base = k_len - q_len

        for q_head in range(num_q_heads):
            kv_head = q_head // (num_q_heads // num_kv_heads)
            q_f32 = q[q_start:q_end, q_head].to(torch.float32)
            k_f32 = keys[:, kv_head].to(torch.float32)
            v_f32 = values[:, kv_head].to(torch.float32)
            scores = torch.matmul(q_f32, k_f32.transpose(0, 1)) * float(softmax_scale)

            if causal:
                q_pos = torch.arange(q_len).view(-1, 1) + q_base
                k_pos = torch.arange(k_len).view(1, -1)
                scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

            row_lse = torch.logsumexp(scores, dim=-1)
            probabilities = torch.exp(scores - row_lse[:, None])
            attention = torch.matmul(probabilities, v_f32)
            out[q_start:q_end, q_head] = attention.to(torch.bfloat16)
            lse_out[q_head, q_start:q_end, 0] = row_lse

    return out, lse_out
```
