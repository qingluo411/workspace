# [Operator Request] GLM-5.3-Flash KDA prefill checkpoint state export（prefix cache + MTP）

> 本 issue 为自包含版本，可直接粘贴到 GitHub；正文中的接口、reference、测试数据和日志均已内嵌。

## 需求类型

扩展已有算子能力

## 优先级

P1

## 目标版本与期望时间

`DLC_Custom_Kernel` `release-glm53flash-3` 后续版本；期望 `2026-11-20`

本 issue 将普通 target prefill 与 MTP target prefill 的 KDA checkpoint 作为同一组
API 交付；MTP drafter 的第 45 层 sparse MLA/KPool cache 不在本 issue 范围内。

## 算子名称

- `torch.dlc_kda_chunk_prefill`
- `torch.dlc_kda_causal_conv_prefill`

## 背景与功能说明

GLM-5.3-Flash 的 KDA linear-attention 层在 prefix caching 下需要复用两类 state：

1. conv state：KDA q/k/v causal convolution 的滑动窗口状态；
2. recurrent state：KDA chunk recurrence 的 FP32 `[B,H,Dv,Dk]` 状态。

当前 `mamba_cache_mode=align` 只在一次 scheduler prefill step 的最后一个 token
且位置为 256-token block 边界时写 state。以 64K prompt、`block_size=256`、
`max_num_batched_tokens=4096` 为例，第一次请求的 chunk 末端是：

```text
4096, 8192, ..., 61440, 65536
```

最后一个可复用的 KDA state 边界是 `65280`，但它位于最后一次 chunk
`61440 -> 65536` 内部，不是 chunk 末端，因此第一次请求没有写出 65280
的 KDA checkpoint。重复请求需要按下面的顺序逐轮补边界：

```text
第 1 次：写入 61440
第 2 次：命中 61440，计算 61440 -> 65536，写入 65280
第 3 次：命中 65280，只计算 65280 -> 65536 的 256 token
```

当前两个算子的限制：

- `torch.dlc_kda_chunk_prefill` 只返回整段 query 的 `final_state`；
- `torch.dlc_kda_causal_conv_prefill` 只把最终 conv state 写到输入 state 的目标页；
- 两者都没有“在 query 内部指定 offset 输出 checkpoint”的能力。

当前框架用“在 checkpoint 边界把 prefill 拆成两次调用”的 Python fallback 做正确性验证：

```text
第一段 [start, checkpoint_offset] -> 得到 checkpoint state
第二段 [checkpoint_offset, end]   -> 得到 final state
```

这个 fallback 会重复启动 kernel，增加 host 调度和同步，且对 batch/后续 MTP 路径不友好，
不适合作为最终生产方案。

目标是在两个算子上增加可选的 checkpoint 输出：

- 每条序列指定 token offset 处输出 conv state 快照；
- 每条序列指定 token offset 处输出 recurrent state 快照；
- `checkpoint_offsets=None` 时保持现有调用、输出和性能行为；
- 支持 batch 内不同序列使用不同 offset；`0` 表示不导出。

### MTP + prefix cache 场景（2026-11-16 补充）

MTP 与 prefix cache 同时开启时，目标模型的 KDA prefill 还需要在 **query 内部** 导出
checkpoint。固定 4096-token prompt、`block_size=256`、
`max_num_batched_tokens=4096`、MTP `num_speculative_tokens=1`、prefix cache 开启时，
实际调度为：

```text
第 1 次请求：3840 + 256 token，没有导出 3584 checkpoint
第 2 次请求：3584 + 256 + 256 token，没有命中，但发现 shared-prefix junction
第 3 次请求：只计算最后 256 + 256 token，命中 3584
```

当前框架在 MTP 打开时不会给 KDA spec 设置 `num_prefill_checkpoint_blocks`，原因是
`get_kv_cache_spec()` 只要看到 `speculative_config is not None` 就直接返回原始 spec。
因此第一次请求没有 3584 的 recurrent/conv checkpoint；第二次请求只能依靠
full-attention 的 3584 前缀发现 shared-prefix junction，补写一次 checkpoint；第三次
才形成完整命中。这个时序不是 prefix cache 的随机行为，而是 MTP 下 checkpoint
能力缺失。

本 issue 的目标因此扩展为：**同一组 KDA prefill checkpoint 能力必须能用于 MTP
target 的 KDA prefill 路径**。MTP drafter（GLM5 的第 45 层 sparse MLA/KPool）的
KV cache、proposal/acceptance 逻辑不在本 issue 范围内。算子本身不需要新增
MTP 专用 kernel，但必须支持与 MTP target prefill 相同的 `query_start_loc`、
batch 行和 `checkpoint_offsets` 调用语义；draft token 不进入 KDA target prefill 的
query 行。

非目标：不改变 KDA 数值公式、不负责把 state 写入 paged KV cache、不负责 MTP
drafter 的 sparse MLA/KPool cache，也不负责 speculative acceptance/rejection。
目标模型 KDA 层在 MTP 路径中的 checkpoint 属于本 issue 范围。

## 现有算子接口（当前实现复制）

当前 KDA chunk prefill 的 Python 调用形式为：

```python
out, final_state = torch.dlc_kda_chunk_prefill(
    q,                  # bf16 [1, T, H, 128]
    k,                  # bf16 [1, T, H, 128]
    v,                  # bf16 [1, T, H, 128]
    raw_g,              # bf16 [1, T, H, 128]
    beta_logits,        # bf16 [1, T, H]
    a_log,              # fp32 [H]
    g_bias,             # fp32 [H, 128]
    initial_state,      # fp32 [B, H, 128, 128]
    query_start_loc,    # int32 [B+1]
    lower_bound,        # -5.0 for GLM5
    query_scale,        # 128 ** -0.5
    chunk_size,         # 64
)
```

当前 KDA causal conv prefill 的 Python 调用形式为：

```python
out = torch.dlc_kda_causal_conv_prefill(
    qkv,                # bf16 [D, T]
    weight,             # fp32 [4, D]
    state,              # conv state pool
    query_start_loc,    # int32 [B+1]
    cache_indices,      # int32 [B]
    has_initial_state,  # int32/bool [B]
    activation="silu",
    null_state_id=0,
)
```

## 期望 API 与函数签名

建议保持已有返回 arity，通过可选输入/输出参数扩展：

```python
out, final_state = torch.dlc_kda_chunk_prefill(
    q, k, v, raw_g, beta_logits, a_log, g_bias, initial_state,
    query_start_loc, lower_bound, query_scale, chunk_size,
    checkpoint_offsets=None,   # int32 [B], offset from query start; 0 = none
    checkpoint_state=None,     # optional FP32 [B, H, Dv, Dk] staging output
)
```

```python
out = torch.dlc_kda_causal_conv_prefill(
    qkv, weight, state, query_start_loc, cache_indices, has_initial_state,
    activation="silu", null_state_id=0,
    checkpoint_offsets=None,   # int32 [B], offset from query start; 0 = none
    checkpoint_state=None,     # optional state buffer for checkpoint snapshots
)
```

如实现更适合新增算子，可新增 `torch.dlc_kda_chunk_prefill_ckpt` 和
`torch.dlc_kda_causal_conv_prefill_ckpt`，但参数语义必须一致。

## 参数与返回值

### `dlc_kda_chunk_prefill`

- `q`, `k`, `v`, `raw_g`: BF16 `[1, T, H, Dk/Dv]`
- `beta_logits`: BF16 `[1, T, H]`
- `a_log`: FP32 `[H]`
- `g_bias`: FP32 `[H, Dk]`
- `initial_state`: FP32 `[B, H, Dv, Dk]`
- `query_start_loc`: int32 `[B + 1]`
- `checkpoint_offsets`: int32 `[B]`，相对每条 query 起点的 token offset，`0` 表示不导出
- `checkpoint_state`: 可选 FP32 `[B, H, Dv, Dk]`，输出 checkpoint recurrent state

输出 `out [1,T,H,Dv] BF16`、`final_state [B,H,Dv,Dk] FP32`；`checkpoint_state` 由调用者传入并被原地填充。

MTP 场景下 `B` 仍是本次 KDA target prefill 的有效 query 行数，`query_start_loc`
是这些行的权威边界；`checkpoint_offsets[b]` 相对该行 query start 计算。算子不得
依赖 `num_speculative_tokens`、draft token 或接受/拒绝结果来决定 checkpoint 语义。
同一次调用中允许部分行带 checkpoint、部分行 `offset=0`。

### `dlc_kda_causal_conv_prefill`

输入沿用现有参数；新增 `checkpoint_offsets` int32 `[B]` 和 `checkpoint_state`。
`checkpoint_state` 的形状与现有 conv state 单页槽位一致，建议 `[B, conv_dim, width]`
或 `[B, width, conv_dim]`，最终以现有 state layout 为准。

conv checkpoint 是 query 内部某 token offset 后的卷积滑动窗口快照；MTP target
prefill 与普通 target prefill 使用同一语义，不把 draft token 的 conv state 计入。

## 功能语义

对第 `b` 条序列，记：

```text
start = query_start_loc[b]
end   = query_start_loc[b + 1]
offset = checkpoint_offsets[b]
```

要求 `0 <= offset < end - start`。checkpoint state 表示消费
`start .. start + offset - 1` 后的状态，等价于两段调用的第一段 final state：

```text
chunk_prefill(start, start + offset) -> checkpoint_state
chunk_prefill(start + offset, end)   -> final_state
```

边界规则：

- `checkpoint_offsets=None` 时输出、最终 state、数值和 launch 行为与当前版本一致；
- `offset == 0` 表示没有 checkpoint，不读取或写入对应行；
- 建议要求 `offset % 64 == 0`，与 KDA 64-token 内部 chunk 对齐；
- 支持 batch 内混合 offset，包括部分行 0、部分行非 0；
- 空序列/零长度行跳过；非法参数必须明确报错，不能静默写错 state。

## 支持范围与约束

- TYD / DLC 真机；模拟器只要求可启动，不作为性能验收环境；
- `Dk == Dv == 128`、`chunk_size == 64`；
- `B` 至少覆盖 1、2、4、8；`T` 至少覆盖 1、64、256、4096、65536；
- `q/k/v/raw_g` BF16，recurrent state FP32；
- conv state dtype 与现有 cache dtype 一致（当前 GLM5 为 BF16，允许 FP32 变体）；
- `checkpoint_offsets` 为 int32；`checkpoint_state` 的布局、stride 和 512B 对齐要求需在 API 文档中明确；
- MTP target prefill：至少覆盖 `num_speculative_tokens=1`，KDA target query 的
  `checkpoint_offsets` 可非零；MTP draft token 不进入本算子；
- 不要求 MTP drafter 的 KV/KPool checkpoint；该部分由独立方案覆盖；
- MTP 场景下 `checkpoint_offsets=None` 仍必须保持 `out` / `final_state` 与当前
  target prefill 路径一致。

## 精度要求

recurrent state 的 reference 语义为：对前 `offset` 个 token 单独执行一次完整的
KDA chunk recurrence，取那次调用的 final state。可直接用现有
`torch.dlc_kda_chunk_prefill` 对 `[0, offset]` 前缀运行得到：

```python
_, ref_checkpoint_state = torch.dlc_kda_chunk_prefill(
    q_prefix, k_prefix, v_prefix, raw_g_prefix, beta_prefix,
    a_log, g_bias, initial_state_prefix, prefix_query_start_loc,
    lower_bound, query_scale, chunk_size,
)
```

conv state 的 reference 语义为：对 `[0, offset]` 前缀运行现有 conv prefill，
取中间 state 快照；等价于在 offset 处切开后在第二段继续使用该 state。

验收要求：

- checkpoint recurrent state 最大绝对误差 `<= 2e-2`，不得含 NaN/Inf；
- conv state 与两段 split 结果逐元素一致，或满足当前 conv 输出约定；
- `checkpoint_offsets=None` 时原有 `out/final_state` 对拍不得回退。

以上 `<= 2e-2` 只作为算子数值门槛，不能单独证明端到端 greedy 等价。最终验收还要
求 checkpoint state 与“只对前缀运行的同一算子 final state”逐 bit 或至多 1 个 FP32 ULP
一致；若暂时达不到逐 bit，至少必须同时满足：

- 4096-token prompt 的 `offset=3584` checkpoint 可由第一次请求稳定导出；
- 同一 prompt 的第二次请求能命中 3584，冷/热 greedy 输出完全一致；
- MTP `num_speculative_tokens=1` 在同一 prompt、同一 chunk 配置下，checkpoint 导出
  开启与关闭的 greedy 输出一致。

## 测试范围

- [ ] 基本功能与 reference 对齐
- [ ] 静态/动态 shape
- [ ] 单卡、batch 内混合 offset
- [ ] offset=0、非 checkpoint 路径
- [ ] 64-token 对齐、非对齐非法输入
- [ ] 4096/65536 长序列
- [ ] MTP target prefill，`num_speculative_tokens=1`，`offset=3584/65280`
- [ ] MTP prefix-cache 重复请求：首次导出、第二次命中、冷热输出一致
- [ ] 多卡/分布式（只要求 TP 编译通过）
- [ ] 性能基准与性能回归

建议用例：

```text
B=1, T=7168,  offset=6912
B=1, T=65536, offset=65280
B=2, lens=[4096, 7168], offsets=[3840, 6912]
B=4, lens=[7168, 4096, 2048, 1024], 混合 offset
B=1, T=256, offset=64
B=1, offset=0
MTP B=1, T=4096, offset=3584（num_speculative_tokens=1）
MTP B=1, T=65536, offset=65280（num_speculative_tokens=1）
MTP B=2, lens=[4096, 7168], offsets=[3584, 6912]
非法：offset<0、offset>=length、checkpoint_state shape/dtype 错误
```

MTP 端到端复现配置：

```text
max_model_len=8192
block_size=256
max_num_batched_tokens=4096
max_num_seqs=1
enable_prefix_caching=True
mamba_cache_mode=align
enable_mamba_shared_prefix_checkpoint=True
speculative_config={"method":"mtp","num_speculative_tokens":1}
```

期望在框架配套启用 MTP checkpoint gate 后：第一次 4096-token 请求导出 3584
checkpoint；第二次相同 prompt 直接命中 3584，不再经历
`3840+256 -> 3584+256+256 -> 256+256` 的三次收敛。

性能要求：

- 64K prefill 开启 checkpoint 时，不应重复执行整段 KDA prefill；
- 相对于无 checkpoint 路径，额外 launch 数不超过 1 个，额外 prefill 开销目标小于 5%；
- 必须记录 `B=1, T=65536` 的单次 kernel 耗时和峰值显存。

## 验证环境与依赖版本

```text
硬件：TYD 8 卡，/dev/cltech0..7
驱动：Driver Version_v26.0.1 / Api_Version_26
PyTorch：2.5.0
vLLM：release-glm53flash / 3eb6a0c681503401cde71b54da8b252a280084ca
vLLM-CL：release-glm53flash / 1940b2ef7f5fb8abfe460dba9b99f6b6807d6186
DLC_Custom_Kernel：release-glm53flash-3 / 59e9f0788995cf09bc757f7407925c08484691b2
关键环境变量：DLC_TPU_VERSION=2, VLLM_PLUGINS=dlc,dlc_patches, DLC_SYN_GRAPH_CACHE=1
```

## 验收标准

- [ ] API、参数校验和 offset=0 行为符合约定
- [ ] checkpoint conv state 与两段 split reference 对齐
- [ ] checkpoint recurrent state 与 prefix reference 对齐
- [ ] 无 checkpoint 时输出、final state 和性能不回退
- [ ] B=1、B>1、混合 offset、64K 长序列测试通过
- [ ] MTP target prefill 的 `offset=3584/65280` checkpoint 测试通过
- [ ] MTP prefix-cache 首次导出、第二次命中、冷热 greedy 输出一致
- [ ] 提供 `B=1,T=65536` 性能数据
- [ ] 提供 `MTP B=1,T=4096,offset=3584` 的单 kernel 与端到端数据
- [ ] 新增/更新文档、pytorch_test 和 syntest，并可由 CI 复现

## 当前框架侧临时验证（内嵌日志）

测试条件：`max_num_batched_tokens=4096`、`max_num_seqs=1`、
`enable_prefix_caching=True`、`mamba_cache_mode=align`、无 MTP、eager；
输出 `max_tokens=8`，TTFT 不受输出长度影响。

7K prompt（无前缀）：

```text
Iteration(0): 1 context requests, 4096 context tokens
Iteration(1): 1 context requests, 3072 context tokens
第 1 次 TTFT = 16.913 s

第 2 次 TTFT = 0.692 s
```

64K prompt（服务已有 7168-token 前缀）：

```text
Iteration(17): 1 context requests, 4096 context tokens
...
Iteration(30): 1 context requests, 4096 context tokens
Iteration(31): 1 context requests, 1024 context tokens
第 1 次 TTFT = 157.747 s

第 2 次 TTFT = 0.956 s
第 3 次 TTFT = 0.945 s
```

64K 首次请求经过了 `L-256` checkpoint 的 split 补写路径；第二次请求只重新计算
最后 256 token。该结果只验证 split fallback，不代表原生算子性能。

### MTP target 时序证据（2026-11-16）

同一台 TYD 8 卡机器，配置为 `max_model_len=8192`、`max_num_batched_tokens=4096`、
`block_size=256`、`max_num_seqs=1`、prefix cache 开启、`mamba_cache_mode=align`、
`enable_mamba_shared_prefix_checkpoint=True`、MTP
`num_speculative_tokens=1`、`FULL_DECODE_ONLY` graph。固定 4096-token prompt 连续发送
三次，观察到的 scheduler 日志为：

```text
第 1 次：Iteration(0): 3840 context tokens
          Iteration(1):  256 context tokens
          prefix cache hit: 0

第 2 次：Iteration(20): 3584 context tokens
          Iteration(21):  256 context tokens
          Iteration(22):  256 context tokens
          prefix cache hit: 0，但发现 shared-prefix junction=3584

第 3 次：Iteration(61):  256 context tokens
          Iteration(62):  256 context tokens
          prefix cache hit: 3584
```

代码侧原因是 MTP 打开时 `get_kv_cache_spec()` 不给 KDA spec 增加
`num_prefill_checkpoint_blocks`；因此第 1 次没有 3584 checkpoint，不能仅靠框架参数
消除时序。`--prefix-cache-retention-interval 256` 复核仍为 0/0/3584，不能替代
checkpoint 输出。算子 API 落地后，框架侧还需要把 MTP target 的 checkpoint gate
打开；建议本 issue 的验收同时覆盖“算子 API 可用”和“框架第一次请求可导出、第二次
请求可命中”两个层次。

## 备注

框架侧当前可先用 split fallback 验证 `L-256` checkpoint；本 issue 的目标是提供原生
checkpoint 输出，从而删除 split fallback、支持 batch，并避免重复 prefill。split fallback
只作为正确性参考和临时验证路径，不作为最终性能方案。MTP target KDA prefill 的
checkpoint 已纳入本 issue；MTP drafter 的 sparse MLA/KPool cache 仍需独立需求覆盖。
