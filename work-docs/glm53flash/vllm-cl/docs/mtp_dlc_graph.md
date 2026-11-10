# MTP DLC Graph Support

本文说明 DLC 下 MTP full-decode proposer 的 graph 化实现。目标场景是主模型使用
`FULL_DECODE_ONLY`，speculative method 为 `mtp` 或 `deepseek_mtp`。

## Scope

每个固定 shape 的 proposer graph 包含：

- first MTP drafter forward
- greedy sampler（`compute_logits().argmax()`）
- serial step 的 slot mapping 和 attention metadata device 更新
- 后续 MTP drafter forwards 和 greedy samplers
- 最终 draft token `stack`

first-pass attention metadata builder 保留在 graph 外。它可能执行 CPU shape 计算和 H2D
staging；每轮先更新持久 metadata buffer，随后 proposer graph 读取这些固定地址。serial
step 使用 device-resident metadata，不再重复 CPU prepare 或 H2D。

以下内容仍在 proposer graph 外：

- target model verify/sample
- first-pass metadata prepare
- scheduler bookkeeping 和按需 draft-token D2H

## Runtime Flow

运行时流程如下：

1. runner 完成一次 target batch prepare、target forward 和 sample。
2. `prepare_next_token_ids_padded` 和 `prepare_inputs_padded` 将小型动态结果写入 proposer
   持久 buffer，保证 capture/replay 地址一致。
3. first-pass attention metadata 在 graph 外构建，并更新持久 backend buffer。
4. 根据 `(first_pass_tokens, batch_size, num_speculative_tokens)` 查找 proposer graph。
5. 命中后只执行一次 `DLCGraph.replay()`，返回固定地址的 draft token tensor。
6. 最终结果继续复用 runner 原有的异步、按需 D2H 路径。

outer graph capture 时，内层 `drafter.model` 的 `CUDAGraphWrapper` 被强制设为 eager，避免
嵌套 graph。若 outer descriptor 没有预先 capture，则回退到原有逐-forward graph 路径，
不会在在线请求中 lazy capture。

## Descriptor

outer proposer descriptor 为：

```text
(first_pass_num_tokens, num_reqs, num_speculative_tokens,
 has_lora, num_active_loras)
```

以 `num_speculative_tokens=3`、capture sizes 规格化为 `4, 8, 12` 为例：

```text
(4,  1, 3)
(8,  2, 3)
(12, 3, 3)
```

旧的 per-forward descriptor（含 `draft_index`）仍保留，作为 outer graph 关闭或缺失时的
fallback。

## Capture Lifecycle

graph memory profiling 使用临时 allocator pool。outer proposer 不在该阶段 capture，避免
proposer 内部 lazy persistent buffer 引用临时 pool。正式 capture 前，runner 用最大 FULL
descriptor eager 跑一次完整 proposer，初始化 workspace 和持久 buffer；随后在主 graph
pool 中 capture 所有 outer descriptors。

默认启用条件：

- speculative method 包含 `mtp`
- `enforce_eager == False`
- compilation config 支持 full cudagraph
- 非 ubatching、非 multimodal、非 parallel drafting
- `DLC_MTP_DRAFTER_CUDAGRAPH` 未关闭
- `DLC_MTP_PROPOSER_CUDAGRAPH` 未关闭

可用以下开关恢复逐-forward graph，进行 A/B：

```bash
export DLC_MTP_PROPOSER_CUDAGRAPH=0
```

## Verification

debug 环境变量：

```bash
export DLC_MTP_DRAFTER_CUDAGRAPH_DEBUG=1
export DLC_MTP_DRAFTER_CUDAGRAPH_DEBUG_FILE=/tmp/mtp_graph_debug.log
```

outer graph 命中时会记录：

```text
dlc_mtp_proposer_graph replay tokens=4 reqs=1 steps=3
```

torch profiler trace 中，每个 decode iteration 应只有两个主要 replay：target graph 和 MTP
proposer graph；不应再出现三个独立 MTP drafter replay。

本次 batch-size 1、128 input、512 output 的直接 A/B：

```text
                         outer graph   per-forward graph
Mean TPOT                42.75 ms      43.44 ms
P90 TPOT                 44.65 ms      45.29 ms
Output throughput        22.98 tok/s   22.61 tok/s
```

Mean TPOT 改善约 `0.69 ms`（`1.6%`）。trace 中 outer proposer 已合并三个 forward、sampler
和 serial metadata 更新；剩余 MTP 时间主要是 graph 内计算和通信，所以进一步减少 Python
launch 的收益上限较小。

## Metadata Optimizations

### Native `next_n=4` indexer

vLLM 通用 indexer 默认仅将 `next_n=1,2` 视为 native shape，因为 CUDA DeepGEMM 的
`fp8_paged_mqa_logits` 只支持 `next_n <= 2`。DLC 的
`custom_paged_multi_query_attention_bf16` 和 `custom_topk_per_row_decode` 均按动态
`next_n` 实现，paged MQA 测试生成范围覆盖 1 到 8，top-k 测试也包含 8。因此 DLC patch
将 `DeepseekV32IndexerMetadataBuilder.natively_supported_next_n` 扩展为
`[1, 2, 4, 6, 8]`，覆盖常用的 MTP width。

对于 `num_speculative_tokens=3`，first MTP pass 保持为 `B x 4`，不再 flatten 成
`4B x 1`。这会消除 indexer builder 中的 `repeat_interleave`、expanded sequence lengths、
expanded block table 和多次 tiny D2D buffer copy。

### Device-resident sparse MLA request ids

MTP full decode 的 request-id mapping 是固定的：

```text
first pass:  [0,0,0,0, 1,1,1,1, ...]  (next_n=4)
serial pass: [0,1,2,...]
single req:  [0,0,...]
```

builder 初始化时预生成以上 device buffer。运行时在严格匹配
`fast_build + max_query_len + num_tokens` 的 MTP shape 下直接切片；单请求直接切片全零
buffer。prefill、mixed batch 和 irregular query lengths 仍使用原 NumPy/H2D fallback。
该优化在检测到 MTP speculative config 时自动启用，不再提供独立的
`DLC_MTP_SERIAL_METADATA_ON_DEVICE` 开关。它不依赖 outer proposer graph；关闭
`DLC_MTP_PROPOSER_CUDAGRAPH` 做逐-forward graph A/B 时仍然保留。

在优化前的
`/tmp/jinshu/glm5_0708_64k_topk256_trace_mtp_graph_tp8ep2` rank0 trace 中，
`sparse_mla.build` 平均 `0.880 ms`，其中 request-id `copy_` 累计 `16.489 ms/24次`，
约占 builder 时间的 `78%`。静态 device mapping 用于消除 MTP full-decode 热路径中的这次
tiny H2D。

## Reverted Experiment: Early Backup-token H2D

`prepare_next_token_ids_padded_dlc` 需要 backup token 处理 discarded request，以及
`sampled_token_ids` 整行无效但未 discard 的 fallback。原实现每轮在 target sample 后执行
CPU lookup 和 tiny H2D。旧 trace 中该 H2D 因 DLC async dependency 等待被放大为主要耗时。

实验曾把 backup token lookup/H2D 提前到 target `_prepare_inputs`，而 target sample 后只消费
已准备的 device buffer。数据依赖上这是成立的：仅 backup token 来自 iteration-start CPU
request state；drafter 本身仍严格依赖 target hidden states、logits 和 sampled token。

但端到端性能回退。final1526 trace：

```text
trace: /tmp/jinshu/glm5_0722_64k_topk256_trace_mtp_graph_tp8ep2_final1526

_prepare_inputs:
  20 calls, 9.272 ms/call
  early backup helper: 5.398 ms total

prepare_next_token_ids_padded_dlc:
  19 calls, 0.182 ms/call
```

该实验只是将 H2D 和 dependency bookkeeping 移到 target input preparation，并增加了每轮
input-preparation 工作，没有消除传输，因此已从工作树撤销。实验 patch 保存在仓库根目录
`backup_next_token_ids_early_h2d.patch`。后续若继续优化，应维护 device-resident committed
token state，真正移除每轮 backup H2D，而不是调整 copy 时序。

## Graph Memory Estimate

启动日志示例：

```text
CUDA graph pool memory: 0.03 GiB (actual), 0.53 GiB (estimated),
difference: 0.5 GiB (1969.2%).
```

estimated 和 actual 当前不是同口径测量：

- profiling 只采样每种 mode 的前两个 descriptor。第一个样本的全部 free-memory 下降都计为
  `first_capture`，包括 lazy kernel workspace、persistent graph input/output buffer、allocator
  pool growth 和首次 graph capture 开销。
- estimate 使用 `first_capture + per_graph * (num_graphs - 1)`；后续每张 graph 至少强制按
  `1 MiB` 估算。
- 正式 capture 在记录 `start_free_gpu_memory` 之前，会 eager 运行最大 FULL descriptor，初始化
  MTP proposer workspace 和 persistent buffer。因此这些一次性分配不进入 actual。
- actual 只计算正式 capture 区间的 `mem_get_info()` 差值，主要剩余共享 graph pool 增量。

所以 `0.53 GiB` 不是实际 graph pool 使用量，而是包含首次 lazy allocation 的保守预算；
`0.03 GiB` 更接近正式 graph capture 的新增 pool。百分比使用
`abs(estimate-actual) / actual`，actual 仅约 0.025 GiB 时，约 0.50 GiB 的绝对差会显示为接近
`2000%`。这主要是 estimator 口径问题，不表示运行时额外浪费了 20 倍 graph memory。

## Limitations

- 仅覆盖固定 padded batch 的 MTP full decode。
- capture sizes 未覆盖的 batch shape 会回退到逐-forward graph。
- multimodal、parallel drafting、ubatching 和动态 LoRA 组合尚未纳入 outer graph。
