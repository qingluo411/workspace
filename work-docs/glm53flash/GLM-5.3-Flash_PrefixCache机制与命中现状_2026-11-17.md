# GLM-5.3-Flash 在 vLLM 中的 Prefix Cache 机制、命中现状与 KDA Checkpoint 缺口

日期：2026-11-17  
适用模型：GLM-5.3-Flash-Mix（45 层，34 个 KDA linear-attention 层 + 11 个 sparse MLA 层）  
代码基线：`vllm` / `vllm-cl` 的 `release-glm53flash` 分支；本文行号按该分支本地
`7f953c0ca7 feat(glm5): materialize KDA prefill checkpoints for DLC prefix cache`
记录。若本地工作区检出于 `fp8-cache-v0.30.1`，该分支是另一条 FP8 cache 工作线，
不包含本文描述的 GLM5 prefix-cache 实现，复现前需先切回发布分支。

本文聚焦 target 模型 34 层 KDA 的 prefix-cache state；MTP drafter 的第 45 层
sparse MLA/KPool cache 不在本文讨论范围。

---

## 1. 结论摘要

1. GLM-5.3-Flash 的 prefix cache 不是单一的“token KV 前缀复用”，而是三组异构缓存
   的联合命中：34 个 KDA 层的 recurrent/conv state、11 个 sparse MLA 层的 latent
   KV 与 indexer K cache、以及不参与 prefix cache 的 KPool tail scratch。
2. 一次请求的有效命中长度是所有可缓存 group 命中长度的最小值。MLA 组可以在
   token 粒度上命中很长前缀，但只要 KDA 组在该边界没有 state，整条请求的有效命中
   就会被拉低，甚至为 0。因此 KDA state 是当前 GLM prefix cache 的限制项。
3. KDA state 只能按 256-token block 边界快照复用。`block_size=256` 时，4096-token
   prompt 的最大可复用边界本来只有 3840；MTP 还会因 EAGLE block drop 再退一个
   block，变成 3584。也就是说，MTP 下 4096 prompt 的结构性上限是 3584 / 4096 =
   87.5%，不可能达到 100%。
4. 当前 MTP + 4096 chunk + FULL_DECODE_ONLY graph 的 4K 精确 prompt 实测为
   **0 hit → 0 hit → 3584 hit**：第一次请求只写出 3840 和 4096 两个位置的 state，
   没有 3584 checkpoint；第二次请求依靠 MLA 组的 shared-prefix junction 停在 3584
   并补写 KDA state；第三次请求才命中 3584。这不是命中率统计 bug。
5. 512-token chunk 可以让 3584 正好落在 chunk 末端，因此第二次请求即可命中 3584；
   但这是用大量小 chunk 换命中时序，16K/32K/64K 冷请求 TTFT 分别达到约
   44.6 / 93.9 / 202.5 秒，不能作为生产推荐。
6. 没有完整的原生 KDA checkpoint 导出能力时，非 MTP 路径用 Python “在 checkpoint
   边界把 prefill 拆成两段”的 fallback：每层要做 host `.cpu().tolist()`、逐 row
   Python 循环、conv/recurrent 各两次 kernel 调用、`scatter_states` 和 `torch.cat`。它既增加
   launch/同步开销，也可能让冷/热请求走不同的数值路径，已在部分 prompt 上观察到
   greedy 输出翻转。
7. 截至 2026-11-17，`dlc_kda_chunk_prefill` 的 recurrent checkpoint 已由算子侧
   交付并通过单算子验证；但 `dlc_kda_causal_conv_prefill` 仍没有 checkpoint API，
   vLLM 也还没有把原生 checkpoint 接到模型路径上，MTP 的 KDA checkpoint spec gate
   也仍然关闭。因此“完整 KDA checkpoint 能力”尚未闭环。

---

## 2. 模型为什么需要特殊的 Prefix Cache

GLM-5.3-Flash 的主模型有 45 层，其中 34 层是 KDA linear attention，11 层是 sparse
MLA（layer 3、7、11、…、43）。两类层的“可复用状态”性质不同：

| 缓存组 | 层数 / spec 数 | `KVCacheSpec` | 缓存内容 | 复用单位 | `prefix_cacheable` |
|---|---:|---|---|---|---|
| KDA state | 34 层，34 个 `MambaSpec` | `MambaSpec` | conv state（BF16）+ recurrent state（FP32） | 每 256 token 一个 state 快照 | 是 |
| MLA latent KV | 11 层，11 个 `MLAAttentionSpec` | `MLAAttentionSpec` | MLA latent KV | 每 token | 是 |
| Indexer K cache | 11 层，11 个 `MLAAttentionSpec` | `MLAAttentionSpec(tokens_per_state=4)` | kpool 压缩后的 indexer K cache | 每 4 token 一个 state | 是 |
| KPool tail | 11 层，11 个 `KpoolTailSpec` | `KpoolTailSpec` | 未完成 pool 的 raw K + gate score 环形缓冲 | 每请求 1 个 block | **否** |

几个容易混淆的点：

- **KPool tail 不是 prefix cache 的一部分。** `KpoolTailSpec.prefix_cacheable` 返回
  `False`，它是每请求私有的 scratch，不参与 block hash 查找，也不拉低命中长度。
  当前命中为 0 时不要把 KPool tail 当成根因。
- **MLA 组命中长，不代表整条请求命中长。** 11 个 sparse MLA 层和 11 个 indexer
  cache 可以复用 token KV；但 KDA 组是 state-based，必须存在对应边界的状态快照。
  最终有效命中取各组最小值。
- **KDA state 不是普通 KV。** 单层 KDA state page 为 542,720 B：conv state
  `18,432 B`（BF16 `[3, 3072]`）+ recurrent state `524,288 B`（FP32
  `[8, 128, 128]`，TP=8）。一个覆盖 34 层的 state snapshot 约 **17.6 MiB /
  rank**；全局约 **140.8 MiB**。`align` 模式通常常驻 2 个以上 snapshot，实际
  KDA state 占用还要乘以 block 数。这是后文“checkpoint block 容量成本”的来源。

---

## 3. vLLM Prefix Cache 的基本流程

### 3.1 block hash 与 block size

请求进入调度器后，vLLM 按固定 token 粒度计算 `Request.block_hashes`。GLM 的 block
配置是 `--block-size 256`，但 hybrid cache 的 hash 粒度不一定等于 256：

- `scheduler_block_size`：所有 group 实际 block size 的 LCM，用于调度器的
  `num_computed_tokens` 对齐。
- `hash_block_size`：多 group 时默认取所有可缓存 group block size 的 GCD，也可以由
  `--prefix-match-unit` 显式覆盖。
- 如果存在 `MambaSpec` 且 `mamba_cache_mode != "align"`，hash 粒度会退回
  `scheduler_block_size`，细粒度前缀命中被关闭。

对应实现：`vllm/v1/core/kv_cache_utils.py:731 resolve_kv_cache_block_sizes()`。
GLM 的 KDA block 和 MLA block 都是 256，KPool tail 的 block 是 `index_kpool=4`
但不参与 hash，因此默认 `hash_block_size=256`。测试没有设置
`--prefix-match-unit`，所以本文所有命中长度都是 256 的倍数。

### 3.2 一次前缀查找做了什么

`KVCacheManager.get_computed_blocks()`（`vllm/v1/core/kv_cache_manager.py:264`）负责
一次请求的前缀查找：

1. 最大查找长度设为 `request.num_tokens - 1`。最后一个 token 必须重算以产生
   logits，因此即使全部命中，也不能把最后一个 token 直接复用。
2. 调用 `coordinator.find_longest_cache_hit(block_hashes, max_cache_hit_length)`。
3. 每个 cacheable group 各做一次从右向左的最长前缀查找。
4. 所有 group 的命中长度取最小值，得到最终 `num_new_computed_tokens`。
5. 如果某个 group（通常是 MLA）命中了比最终结果更长的前缀，多出来的部分记为
   `num_uncached_common_prefix_tokens`；`shared_prefix_boundary` 就是
   `num_new_computed_tokens + num_uncached`，即“某个稀疏保留 group 还没缓存、
   但其他 group 已经能证明共享”的 junction。

多 group 的归约在 `vllm/v1/core/kv_cache_coordinator.py:833
find_longest_cache_hit()` 中完成。它的核心不变式是：

```text
effective_hit = min(hit_MLA, hit_indexer, hit_KDA)
```

只要 KDA 组没有对应边界的状态，`effective_hit` 就会被压到 KDA 的前一个可用边界，
最坏为 0。

### 3.3 命中率是怎么算的

vLLM 的 `PrefixCacheStats` 按请求累计：

- `queries += request.num_tokens`（prompt token 数）；
- `hits += num_new_computed_tokens`（实际复用的 token 数）；
- 单请求命中率 = `hits / queries`；日志中的 hit rate 是最近一段请求的聚合值。

对应实现：`vllm/v1/metrics/stats.py:117`、`:132`；聚合 hit rate 在
`vllm/v1/metrics/stats.py:109`。因此：

- 第一次请求必然是 0%，这是正常的；
- 4096 prompt 命中 3584 时，单请求命中率是 87.5%，不是 100%；
- 如果日志显示第二、第三次仍然 0%，说明这两次请求的 KDA 组都没有命中，而不是
  统计公式把 3584 算成了 0。

---

## 4. KDA / Mamba align 模式的 State Checkpoint

### 4.1 为什么 KDA state 不能按 token 直接复用

MLA 的 KV 是“每个 token 一份”，可以按 token 截断；KDA 的 recurrent state 是
“消费到当前位置为止的累计状态”，不能从任意 token 的 KV 重新拼出来。因此 KDA
prefix cache 必须依赖**在精确边界上物化的 state 快照**。

`mamba_cache_mode=align` 的语义是：

- 只在 scheduler prefill step 的最后一个 token、且位置是 `block_size` 的整数倍时，
  写一个 KDA state；
- block table 的 row 是“按位置索引”的：第 `j` 个 slot 表示“消费了
  `(j+1) * block_size` 个 token 后的 state”；
- 每个请求实际只常驻 `2 + num_speculative_blocks + num_prefill_checkpoint_blocks`
  个 state block；更早的中间 state 会被 `remove_skipped_blocks()` 置为 null 并释放。

这带来两个直接后果：

1. 命中长度必须是 256 的倍数；如果请求前缀在 3584 和 3840 之间，KDA 只能回到
   3584。
2. 如果某个可复用边界落在一次 prefill chunk 的中间，而这次 chunk 又没有把它
   单独物化，那么即使 MLA 已经缓存了该边界，整条请求也无法命中。

### 4.2 checkpoint 位置和有效性

checkpoint 位置由 `get_mamba_prefill_checkpoint_position()` 计算
（`vllm/v1/kv_cache_interface.py:1094`）：

```python
checkpoint_position = (num_tokens - 1) // hash_block_size * hash_block_size
if drop_eagle_block:
    checkpoint_position -= hash_block_size
return max(checkpoint_position, 0)
```

当前默认配置下 MTP 的 `use_eagle_block_drop()` 为真，因此 4096 prompt 的位置是：

```text
no MTP: (4096 - 1) // 256 * 256          = 3840
MTP:    3840 - 256                       = 3584
```

`is_mamba_prefill_checkpoint_valid()`（`vllm/v1/kv_cache_interface.py:1106`）进一步
要求：

- query 起点按 hash block 对齐；
- checkpoint 必须严格落在本次 query 的 `(start, end)` 内；
- checkpoint 所在 block 必须比初始 state block 更靠后；
- checkpoint 到 query start 的偏移必须满足 `prefill_checkpoint_alignment`。

GLM 的 KDA spec 设置的是 `prefill_checkpoint_alignment=64`，因为 KDA kernel 内部
按 64-token chunk 计算；但 checkpoint 位置本身仍由 `hash_block_size=256` 决定。
不要把 64 误解成“可以按 64 token 命中前缀”。

### 4.3 当前 KDA spec 的开关

关键代码在 `vllm/models/glm5next/common/kda.py:185`：

```python
def get_kv_cache_spec(self, vllm_config: VllmConfig) -> MambaSpec:
    spec = super().get_kv_cache_spec(vllm_config)
    assert isinstance(spec, MambaSpec)
    if (
        current_platform.device_type != "dlc"
        or vllm_config.cache_config.mamba_cache_mode != "align"
        or vllm_config.speculative_config is not None
    ):
        return spec

    return replace(
        spec,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=64,
    )
```

也就是说，只有同时满足以下三个条件，KDA 才会分配一个内部 checkpoint block：

1. 运行在 DLC 真机；
2. `mamba_cache_mode=align`；
3. **没有 speculative config（即没有开 MTP）**。

MTP 打开时 `speculative_config is not None`，直接返回原始 `MambaSpec`，
`num_prefill_checkpoint_blocks=0`。随后：

- `MambaPrefillCheckpointBuilder.build()` 在
  `vllm/model_executor/layers/mamba/checkpoint.py:79` 直接返回 `None`；
- `GDNAttentionMetadata.checkpoint` 为 `None`
  （`vllm/v1/attention/backends/gdn_attn.py:574`）；
- `kda.py` 的 `checkpoint_prefill` 为 `False`
  （`vllm/models/glm5next/common/kda.py:500`）；
- 既不会走原生 checkpoint，也不会走 Python split fallback。

这是当前 MTP + prefix cache 命中时序问题的直接代码原因。

### 4.4 shared-prefix junction 如何让缓存“逐轮收敛”

即使 MTP 下没有 KDA checkpoint，调度器仍有一个补偿机制：当 MLA 组命中了更长的
前缀、而 KDA 组没有时，`get_computed_blocks()` 会把这段更长的共享前缀记录为
`shared_prefix_boundary`。调度器的 `_mamba_block_aligned_split()`
（`vllm/v1/core/sched/scheduler.py:405`）会在 junction 处提前结束一个 chunk，
让 KDA 在 chunk 末端写出该边界的 state。

以 4096-token prompt、chunk=4096、MTP、graph 为例：

| 请求 | 调度 | 命中 | 发生了什么 |
|---|---:|---:|---|
| 第 1 次 | `3840 + 256` | 0 | 只写出 3840/4096 的 KDA state，没有 3584 |
| 第 2 次 | `3584 + 256 + 256` | 0 | MLA 能证明 3584 共享前缀，KDA 不能；调度器停在 3584，补写 KDA state |
| 第 3 次 | `256 + 256` | 3584 | KDA 和 MLA 都能在 3584 命中，只重算尾部 |

注意第 2 次请求的 `cache_hit_delta` 仍然是 0：它确实从 0 开始计算了 3584 个 token，
只是在 3584 处结束并写下了 state。第 3 次才真正复用。这就是“前两次 0、第三次
3584”的完整解释。

这里的 junction 是 256 对齐的，因此不依赖 `--enable-mamba-shared-prefix-checkpoint`；
该开关只在 `prefix_match_unit < block_size`、需要停在亚 block junction，且 MTP/EAGLE
group 等条件满足时才起作用。
本文测试没有设置 `--prefix-match-unit`，所以第 2 次请求的 3584 stop 是 block-aligned
junction 的自然结果。

### 4.5 checkpoint block 的容量成本

`MambaSpec.max_memory_usage_bytes()` 在 `align` 模式下是：

```text
page_size_bytes * (2 + num_speculative_blocks + num_prefill_checkpoint_blocks)
```

对 34 层 KDA：

- 每层 state page = 542,720 B；
- 34 层合计 = 18,452,480 B ≈ 17.6 MiB / rank；
- 全局（TP=8）约 140.8 MiB / 请求（一个 snapshot）。

本文测试为 TP=8、PP=1，34 层在同一个 Mamba group；若启用 PP，mamba group 会被
切分，checkpoint block 的容量需要按各 group 分别计算。

因此 `num_prefill_checkpoint_blocks=1` 不是“零成本”的：每个在途请求要多保留一个
覆盖 34 层的 state block。MTP 下当前直接关掉 checkpoint，省下了这块容量，但代价是
命中时序退化；这是一笔不划算的交换。

---

## 5. 没有原生 KDA Checkpoint 算子时，vLLM 实际怎么跑

注意：下面的 Python split fallback 只在 KDA spec 分配了 checkpoint block 时触发，
即当前 **DLC + align + 非 MTP** 的组合。MTP 下 spec gate 直接把
`num_prefill_checkpoint_blocks` 置为 0，根本不会进入本节的 fallback，而是走 4.4 的
shared-prefix junction 逐轮补写。

### 5.1 通用 metadata 已经存在，但 DLC 没有原生 exporter

vLLM 已经有 backend-neutral 的 checkpoint metadata：

- `compute_mamba_prefill_checkpoints()`：`vllm/model_executor/layers/mamba/checkpoint.py:21`
- `MambaPrefillCheckpointBuilder`：`vllm/model_executor/layers/mamba/checkpoint.py:70`
- KDA 的 CUDA FlashKDA exporter：`vllm/model_executor/layers/mamba/kda_checkpoint.py:16`

但 `FlashKDAPrefillCheckpointExporter` 走的是 Triton kernel 路径，DLC 平台没有
对应的 DLC exporter。DLC 实际走的是 `vllm/models/glm5next/common/kda.py` 中的
Python fallback：

- `_causal_conv1d_with_checkpoint()`：`kda.py:803`
- `_chunk_prefill_with_checkpoint()`：`kda.py:900`

### 5.2 fallback 的具体执行路径

当 `checkpoint_prefill=True` 时，KDA 前向不再调用一次完整的
`causal_conv1d_fn` / `chunk_kda_with_fused_gate`，而是：

1. 用 `.detach().cpu().tolist()` 把 `query_start_loc`、`state_indices`、
   `checkpoint_offsets`、`checkpoint_state_indices` 拉到 host；
2. 按 batch row 做 Python 循环；
3. 对每个需要 checkpoint 的 row，在 `split_at = start + checkpoint_offset` 处切成
   两段；
4. conv 路径调用两次 `causal_conv1d_fn`，把第一段结束时的 conv state 复制到
   checkpoint state block；
5. recurrent 路径调用两次 `chunk_kda_with_fused_gate`，第一段返回
   `checkpoint_state`，用 `scatter_states` 写到 checkpoint block，第二段从该
   state 继续；
6. 用 `torch.cat` 把两段输出拼回去。

对应代码片段（`kda.py:822`、`kda.py:916`）：

```python
starts = query_start_loc.detach().cpu().tolist()
cache_indices = state_indices.detach().cpu().tolist()
offsets = checkpoint.checkpoint_offsets.detach().cpu().tolist()
checkpoint_indices = checkpoint.state_indices.detach().cpu().tolist()
```

这不是一个“只多一个 checkpoint 写回”的实现，而是把一段 prefill 拆成两段独立
kernel sequence，再在 Python 侧拼接。

### 5.3 这条 fallback 的成本

按 34 个 KDA 层、每个 checkpoint 所在 chunk（单请求单行）计算：

| 成本项 | 当前 fallback | 原生 checkpoint 的目标 |
|---|---|---|
| device → host 同步 | conv/recurrent 两个 fallback 各 4 次 `.cpu().tolist()`，每层共 8 次；34 层累计 272 次 | 0 次 |
| kernel launch | conv 2 次 + recurrent 2 次，每层合计 4 次；相比无 checkpoint 的 1+1 次，每层多 2 次 | conv 1 次 + recurrent 1 次，每层合计 2 次；checkpoint 在 kernel 内条件写回 |
| Python 控制流 | 按 batch row 循环，row 内再切片、建 `cu_seqlens` | 无 |
| 输出拼接 | `torch.cat(parts)` / `torch.cat(outputs)` | kernel 直接写连续输出 |
| batch 混合 offset | 每行单独处理，难以批量化 | 单次 kernel 支持 `[B]` offsets |
| 数值路径 | 两段 kernel 的 shape/归约顺序与单段不同 | 单段 kernel 内快照，路径一致性更好 |

这也是为什么 512-token chunk 能把第二次请求的命中提前：3584 正好是 chunk 末端，
第一次请求按正常 chunk-end state 写出，不需要在 chunk 内 split。它不是消除了
fallback，而是绕开了 fallback 的触发条件。

---

## 6. 当前测试命中现状

### 6.1 测试配置

除特别说明外，测试环境为 TYD 8 卡、TP=8、BF16、`block_size=256`、prefix cache
开启、`mamba_cache_mode=align`、单请求、`max_num_seqs=1`。关键配置分四组：

| 编号 | 配置 | MTP | chunk | graph | 用途 |
|---|---|---|---:|---|---|
| A | 4096-graph-exact4096 | 1 spec token | 4096 | FULL_DECODE_ONLY | 精确 4K 命中时序 |
| B | 512-prefix-on-eager-exact4096 | 1 spec token | 512 | eager | 512 chunk 对照 |
| C | no-MTP，eager | 无 | 4096 | eager | 排除 MTP 影响 |
| D | 512-prefix-on-eager-mtp | 1 spec token | 512 | eager | 16K/32K/64K 长上下文 |

### 6.2 精确 4096 prompt，MTP + 4096 chunk + graph

结果文件：`work-tests/mtp/results/prefix_cache_20261116/mtp_prefix_4096_graph_exact_hit_timing_20261116.json`

| 请求 | 命中 token | 单请求命中率 | 端到端耗时（含 32 token 输出） |
|---|---:|---:|---:|
| 第 1 次 | 0 | 0% | 12.363 s |
| 第 2 次 | 0 | 0% | 12.446 s |
| 第 3 次 | 3584 | 87.5% | 3.536 s |
| 第 4 次 | 3584 | 87.5% | 3.535 s |
| 第 5 次 | 3584 | 87.5% | 3.536 s |

该 prompt 的 5 次输出 token 完全一致（`all_same_tokens=true`）。但这只说明该
prompt 对微小数值差异不敏感，不能证明冷/热数值路径等价；见 6.4（非 MTP）与
7.2（MTP）的反例。

补充对照：MTP eager、4096 chunk 的
`mtp_prefix_cache_ckpt_eager_4096_5runs.json` 同样是 0 / 0 / 3584
（12.351 / 12.361 / 3.640 s），且冷/热输出翻转（`all_same_tokens=false`）。
说明“两次冷请求、第三次才命中”不是 graph 独有，而是 4096 chunk 下 KDA
checkpoint 缺失的共性行为。

### 6.3 精确 4096 prompt，MTP + 512 chunk + eager

结果文件：`work-tests/mtp/results/prefix_cache_20261116/mtp_prefix_512_exact4096_hit_timing_20261116.json`

| 请求 | 命中 token | 单请求命中率 | 端到端耗时 |
|---|---:|---:|---:|
| 第 1 次 | 0 | 0% | 12.988 s |
| 第 2 次 | 3584 | 87.5% | 3.768 s |
| 第 3 次 | 3584 | 87.5% | 3.761 s |
| 第 4 次 | 3584 | 87.5% | 3.777 s |

输出 4 次完全一致（`all_same_tokens=true`）。原因是 512 chunk 下 3584 是第 7 个
chunk 的末端，第一次请求就直接写出了该 state，不需要 chunk 内 split。
补充：`mtp_prefix_cache_ckpt_512_graph_4096_5runs.json` 在 512 chunk + graph
下同样为 0 / 3584，且冷/热输出一致，说明该结论不只适用于 eager。

### 6.4 无 MTP 对照，4096 chunk + eager

结果文件：`work-tests/mtp/results/prefix_cache_20261116/nomtp_prefix_cache_ckpt_eager_4096_5runs.json`

| 请求 | 命中 token | 单请求命中率 | 端到端耗时 | 输出前 7 token |
|---|---:|---:|---:|---|
| 第 1 次 | 0 | 0% | 11.993 s | `197,197,322,13,455,6890,198` |
| 第 2 次 | 3840 | 93.75% | 3.158 s | `197,197,322,197,197,322,197` |
| 第 3 次 | 3840 | 93.75% | 3.199 s | `197,197,322,197,197,322,197` |
| 第 4 次 | 3840 | 93.75% | 3.114 s | `197,197,322,197,197,322,197` |
| 第 5 次 | 3840 | 93.75% | 3.158 s | `197,197,322,197,197,322,197` |

这是最关键的对照：MTP 关掉后，命中时序变成“第一次 0、第二次 3840”，但**冷/热输出
在第 4 个 token 处翻转**（`all_same_tokens=false`）。说明输出翻转不是 MTP 独有，
而是 split fallback 冷路径与 checkpoint-resume 热路径的数值路径差异。

### 6.5 4000 prompt 的 prefix-on / prefix-off 对照

测试用固定 prompt `P`（4000 token），共享前缀为 `P[:3584]`，两个不同 4-token
suffix `A`、`B`。结果文件：

- `mtp_prefix_512_on_20261116.json`
- `mtp_prefix_512_shared_prefix_20261116.json`
- `mtp_prefix_512_off_20261116.json`

| 请求 | prefix-on 命中 | prefix-on 耗时 | 与 prefix-off 输出 |
|---|---:|---:|---|
| `P` 首次 | 0 | 12.377 s | 一致 |
| `P` 重复 | 3584 | 3.113 s | 一致 |
| `P[:3584]+A` 首次 | 3072 | 3.530 s | 一致 |
| `P[:3584]+A` 重复 | 3328 | 2.853 s | 一致 |
| `P[:3584]+B` | 3328 | 2.535 s | 一致 |

这里 shared-prefix 请求第一次只命中 3072，重复后才提升到 3328；但输出始终与
prefix-off 基线一致。说明在 512 chunk、checkpoint 边界与 chunk 末端对齐时，
prefix cache 的数值路径是稳定的。

### 6.6 512 chunk 长上下文（MTP，eager）

结果文件：`work-tests/mtp/results/prefix_cache_20261116/mtp_prefix_512_longctx_eager_20261116.json`  
配置：`max_model_len=66560`、`max_num_batched_tokens=512`、MTP、prefix cache 开启、
graph 关闭；每档同一 prompt 冷/热各一次，输出 4 token。

| prompt tokens | cold TTFT | cold total | hot TTFT | hot total | hot 命中 | hot 命中率 |
|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 44.639 s | 45.001 s | 1.659 s | 2.002 s | 15,872 | 96.875% |
| 32,768 | 93.884 s | 94.087 s | 1.847 s | 2.050 s | 32,256 | 98.4375% |
| 65,536 | 202.453 s | 202.638 s | 2.221 s | 2.417 s | 65,024 | 99.21875% |

热点命中长度都是 `prompt - 512`：一个 256 block 用于“最后一个 token 必须重算”的
向下取整，另一个 256 block 用于 MTP/EAGLE 的 block drop。长上下文下命中率看起来
接近 100%，但冷请求的绝对耗时仍然被 prefill 支配，512 chunk 的代价非常明显。

### 6.7 对命中现状的归纳

1. **0% 不一定是统计错误。** 第一次请求必然 0%；MTP + 4K + 4096 chunk 的第二次
   请求也是 0%，因为 KDA 组没有 3584 state，MLA 的 3584 命中被多组最小值归零。
2. **最大命中长度有结构性上限。** 4096 prompt 在 MTP 下是 3584（87.5%），无 MTP
   是 3840（93.75%）；16K/32K/64K 在 MTP 下是 `prompt - 512`。
3. **512 chunk 能提前命中，但不能提升冷性能。** 它把 3584 变成 chunk 末端，
   第二次请求即可命中；代价是 16K/32K/64K 冷 TTFT 约 44.6/93.9/202.5 秒。
4. **冷/热输出可能翻转。** 4096 chunk 的 split 路径在部分 prompt 上出现 greedy
   token 翻转；512 chunk 的对照未出现。这个问题不能靠“某个 prompt 的
   `all_same_tokens=true`”来证明不存在。

---

## 7. 没有完整原生 KDA Checkpoint 能力的后果

### 7.1 命中收敛变慢：重复请求也要先付两次冷成本

在 MTP + 4096 chunk + 4K prompt 的场景，前缀缓存不是“第一次请求完就能复用”，
而是：

```text
第 1 次：0 hit，12.363 s
第 2 次：0 hit，12.446 s
第 3 次：3584 hit，3.536 s
```

前两次都是完整 prefill，累计约 24.8 秒；第三次才降到 3.5 秒。如果业务是“同一
prompt 只发一两次”，prefix cache 基本没有收益；如果业务是“同一 prompt 高频重复”，
也要先经历两轮冷启动。这个收敛时序对在线服务不友好，尤其是长上下文和 MTP 同时
开启时。

512 chunk 可以把收敛压缩到“第一次冷、第二次命中”，但它把冷请求本身变慢：
64K 冷 TTFT 202.5 秒，不能作为生产推荐。

### 7.2 正确性风险：冷/热请求走不同的数值路径

没有完整原生 checkpoint 时，冷/热请求无法保证走同一条 KDA 数值路径：

- 非 MTP 的冷请求：在 checkpoint 边界把一段 prefill 拆成两段 kernel，第二段从
  第一段返回的 `checkpoint_state` 继续；热请求从 paged cache 读取 checkpoint
  state，只计算剩余 token。
- MTP 的冷请求：按 `3840 + 256` 计算，没有 3584 checkpoint；热请求从 3584 的
  cached state 续算，只计算剩余 512 token。

两条路径的 kernel shape、FP32 归约顺序、chunk 边界和 state 物化/拷贝方式不完全
相同，因此不能保证与“单段完整 prefill 的中间 state”逐 bit 一致。实测已经看到
greedy 输出翻转：

| 场景 | 冷请求输出 | 热请求输出 | 是否一致 |
|---|---|---|---|
| 无 MTP，4096 chunk | `197,197,322,13,455,6890,198` | `197,197,322,197,197,322,197` | **否** |
| MTP，4096 chunk | `197,197,322,197,197,322,197` | `197,197,322,13,455,6890,198` | **否** |
| MTP，512 chunk | `197,197,322,13,455,6890,198` | 同左 | 是 |

翻转是否出现取决于 prompt。`mtp_prefix_4096_graph_exact_hit_timing_20261116.json`
中的那条 prompt 5 次输出一致，但 `mtp_prefix_cache_4096_5runs.json` 的另一条固定
prompt 出现翻转；因此不能用单个 prompt 的 `all_same_tokens=true` 证明冷/热路径等价。

这不是“模型精度变差”，而是同一 prompt 在 cache cold/hot 两种路径下产生不同的
greedy 结果。对评测、回归和线上一致性来说，这比单纯的性能损失更严重：prefix cache
命中率会改变答案。当前只能逐 prompt 验证，不能假设所有 prompt 都稳定。

需要强调：原生 checkpoint 算子能消除“Python 两段 split”带来的路径差异，但并不能
自动保证与完整 prefill 逐 bit 一致。算子 issue 的最终验收要求仍然是“第一次请求
稳定导出 3584、第二次请求命中 3584、冷/热 greedy 输出完全一致”，而不是只看单算子
state 的 `2e-2` 误差。

### 7.3 性能开销：host 同步、双 kernel、Python 拼接

fallback 的成本是结构性的，不是一次性的：

- 单请求单行下，conv/recurrent 两个 fallback 各 4 次 `.cpu().tolist()`，每层共
  8 次，34 层累计 272 次 device→host 同步；
- checkpoint 所在 chunk 内，conv 和 recurrent 各调用两次 kernel，每层合计 4 次，
  较无 checkpoint 的 1+1 次多 2 次；
- 每个 batch row 单独切片、构造 `cu_seqlens`、`scatter_states`、`torch.cat`；
- batch 内不同 offset 只能逐行处理，无法利用打包 kernel 的批量能力；
- 这些操作发生在 34 个 KDA 层上，且 prefill 是长上下文的主要成本。

512 chunk 虽然绕开了 split，但把长上下文拆成大量 512-token prefill step；16K/32K/64K
冷 TTFT 约 44.6/93.9/202.5 秒，说明“用小 chunk 换命中”在长上下文下不可接受。

### 7.4 容量成本：checkpoint block 不是免费的

`align` 模式下每个请求常驻 `2 + num_speculative_blocks + num_prefill_checkpoint_blocks`
个 KDA state block。一个 checkpoint block 覆盖 34 层，约 17.6 MiB / rank /
请求（全局约 140.8 MiB）。这意味着：

- 开 checkpoint：命中时序更好，但每个在途请求多占一块 state；
- 关 checkpoint（当前 MTP）：省下这块容量，但前两次请求无法命中。

当前 MTP 选择了后者，因此 4K 场景出现两次冷请求；这是容量与命中时序之间的被动
权衡，不是最优设计。

### 7.5 工程与运维影响

- MTP + prefix cache + FULL_DECODE_ONLY graph + 4096 chunk 的组合目前不能宣称
  “生产就绪”：命中时序和冷/热一致性都还有缺口。
- 512 chunk workaround 只适合做对照或临时验证，不适合作为生产配置；它会把长上下文
  冷启动拉到分钟级。
- 需要把“第一次请求导出 checkpoint、第二次请求命中、冷/热 greedy 一致”作为
  独立验收项，而不能只看服务能启动、请求能返回。
- 当前观察到的 graph capture 卡死（`custom_allreduce_2x4tpu4root_simple_bf16_rdma`、
  `custom_gather_tensor bad_vld_base`）是另一类时序/竞态问题，不应归因于 KDA
  checkpoint；本文不把它们混入 prefix-cache 结论。

### 7.6 哪些现象不是这个缺口造成的

- **第一次请求 0%**：正常，任何 prefix cache 第一次都无缓存。
- **KPool tail 不命中**：设计如此，`KpoolTailSpec.prefix_cacheable=False`。
- **MLA 组命中很长但整条请求命中很短**：多组最小值归约的正常结果，限制项是 KDA。
- **只有唯一 prompt、无共享前缀时命中率长期为 0**：prefix cache 本来就不适用；
  它不是“缓存坏了”。

---

## 8. 原生 KDA Checkpoint 的进展与剩余缺口

### 8.1 截至 2026-11-17 的状态

| 能力 | 状态 | 证据 / 说明 |
|---|---|---|
| `dlc_kda_chunk_prefill` recurrent checkpoint | **已交付** | kernel `7894c5129`（来源 `830835ae8`）、pytorch `ec181b5ff5`（来源 `3446511336`）；静态 20/20、动态 4000/4000、syntest 5/5；4K/64K 补充验证通过 |
| `dlc_kda_causal_conv_prefill` checkpoint | **未交付** | 当前 API 仍无 `checkpoint_offsets` / `checkpoint_state`；conv checkpoint 仍只能靠 Python 两段调用 |
| vLLM 模型路径接线 | **未完成** | `vllm-cl/vllm_cl/patch/patch_kda.py:568` 调用 `torch.dlc_kda_chunk_prefill` 时没有 checkpoint 参数；`vllm/models/glm5next/common/kda.py` 仍走 split fallback |
| MTP KDA checkpoint gate | **仍关闭** | `kda.py:185` 在 `speculative_config is not None` 时直接返回基础 `MambaSpec`，`num_prefill_checkpoint_blocks=0` |
| 端到端验收 | **未完成** | 尚未验证“第一次 4K 请求导出 3584、第二次命中、冷/热 greedy 一致”；64K 65280 同样未做端到端 |

### 8.2 完整落地后应该变成什么样

以 4096 prompt、MTP、chunk=4096 为例，目标行为是：

```text
第 1 次：3840 + 256，但在 3584 处由原生 kernel 导出 checkpoint
第 2 次：直接命中 3584，只计算 3584 -> 4096 的 512 token
```

不再需要：

```text
第 1 次：3840 + 256，无 3584 checkpoint
第 2 次：3584 + 256 + 256，补写 3584
第 3 次：256 + 256，命中 3584
```

64K 场景对应：

```text
第 1 次：chunk 末尾 61440，但在 65280 处导出 checkpoint
第 2 次：命中 65280，只计算 65280 -> 65536 的 256 token
```

同时，原生实现应把 checkpoint 写在主 kernel 内，不再有：

- 每层 8 次 host `.cpu().tolist()`（conv/recurrent 各 4 次）；
- Python 按 row 循环；
- conv/recurrent 各两次 prefill kernel；
- `torch.cat` 拼接；
- batch 内混合 offset 的逐行处理。

### 8.3 还不能直接宣布完成的原因

1. **conv checkpoint 缺失。** KDA 的 conv state 和 recurrent state 必须同时可复用；
   只交付 recurrent 不足以覆盖完整 KDA state。
2. **vLLM 还没接线。** 即使算子 API 可用，模型侧仍然走 fallback；需要把
   `checkpoint_offsets` / `checkpoint_state` 接到 `patch_kda.py` 和
   `kda.py` 的 DLC 路径。
3. **MTP gate 还没打开。** 需要在 MTP target prefill 下分配 checkpoint block，并
   验证 MTP 的 draft/verify 不会污染 checkpoint 语义。
4. **数值一致性还没验收。** 单算子 state 误差达标不等于端到端 greedy 一致；
   必须做冷/热 A/B。
5. **性能还没正式量化。** issue 要求 64K prefill 额外 launch 不超过 1、额外开销
   小于 5%，目前还没有标准 warmup + 多次重复的数据。

---

## 9. 复现与证据

### 9.1 服务启动参数

MTP + 4096 chunk + FULL_DECODE_ONLY graph：

```bash
PORT=18005 MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.85 \
VLLM_USE_DLC_COL_MAJOR_MATMUL=1 PYTORCH_DLC_ALLOC_CONF=expandable_segments:True \
DLC_SYN_GRAPH_CACHE=1 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
DLC_MTP_DRAFTER_CUDAGRAPH=1 DLC_MTP_PROPOSER_CUDAGRAPH=1 \
  bash start_glm5_flash_vllm_cl.sh \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 1 \
    --no-async-scheduling \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --kernel-config '{"enable_jit_warmup": false}'
```

512 chunk eager 对照：把 `--max-num-batched-tokens` 改为 `512`，加
`--enforce-eager`。无 MTP 对照：去掉 `--speculative-config`。

### 9.2 结果文件

目录：`work-tests/mtp/results/prefix_cache_20261116/`

| 文件 | 内容 |
|---|---|
| `mtp_prefix_4096_graph_exact_hit_timing_20261116.json` | MTP + 4096 chunk + graph，精确 4K，0/0/3584 |
| `mtp_prefix_4096_retention256_20261116.json` | 同场景 + retention 256，仍 0/0/3584 |
| `mtp_prefix_512_exact4096_hit_timing_20261116.json` | MTP + 512 chunk + eager，0/3584/3584 |
| `nomtp_prefix_cache_ckpt_eager_4096_5runs.json` | 无 MTP 对照，0/3840，冷热输出翻转 |
| `mtp_prefix_cache_4096_5runs.json` | 4K 冷热输出翻转（MTP，4096 chunk） |
| `mtp_prefix_cache_ckpt_eager_4096_5runs.json` | MTP eager 4096 chunk，0/0/3584，冷热输出翻转 |
| `mtp_prefix_cache_ckpt_512_4096_5runs.json` | 512 chunk，冷热一致 |
| `mtp_prefix_cache_ckpt_512_graph_4096_5runs.json` | 512 chunk + graph，冷热一致 |
| `mtp_prefix_512_on_20261116.json` | 4000 prompt prefix-on，0/3584 |
| `mtp_prefix_512_shared_prefix_20261116.json` | 共享前缀 A/B，3072/3328 |
| `mtp_prefix_512_off_20261116.json` | prefix-off 基线 |
| `mtp_prefix_512_longctx_eager_20261116.json` | 16K/32K/64K 冷热 TTFT 与命中 |

### 9.3 关键代码路径

| 作用 | 路径:行 |
|---|---|
| KDA spec 开关 | `vllm/models/glm5next/common/kda.py:185` |
| checkpoint 位置计算 | `vllm/v1/kv_cache_interface.py:1094` |
| checkpoint 有效性判断 | `vllm/v1/kv_cache_interface.py:1106` |
| checkpoint metadata builder | `vllm/model_executor/layers/mamba/checkpoint.py:21`、`:70` |
| GDN metadata 挂载 checkpoint | `vllm/v1/attention/backends/gdn_attn.py:574` |
| scheduler 的 Mamba chunk 切分 | `vllm/v1/core/sched/scheduler.py:405` |
| Mamba manager | `vllm/v1/core/single_type_kv_cache_manager.py:1445` |
| prefix lookup | `vllm/v1/core/kv_cache_manager.py:264` |
| 多 group 命中归约 | `vllm/v1/core/kv_cache_coordinator.py:833` |
| block size 解析 | `vllm/v1/core/kv_cache_utils.py:731` |
| DLC conv fallback | `vllm/models/glm5next/common/kda.py:803` |
| DLC recurrent fallback | `vllm/models/glm5next/common/kda.py:900` |
| DLC conv 原生调用 | `vllm-cl/vllm_cl/patch/patch_kda.py:122` |
| DLC chunk 原生调用 | `vllm-cl/vllm_cl/patch/patch_kda.py:568` |
| 命中率统计 | `vllm/v1/metrics/stats.py:109`、`:117`、`:132` |

---

## 10. 结论

GLM-5.3-Flash 的 prefix cache 难点不在 MLA，而在 KDA：MLA 可以按 token 复用，
KDA 必须在精确的 256-token 边界上物化 conv + recurrent state。当前完整原生
checkpoint 能力尚未闭环，导致 MTP 下 4K 精确 prompt 需要两次冷请求、第三次才
命中 3584；非 MTP 路径则依赖 Python split fallback，带来 host 同步、双 kernel、
Python 拼接和冷/热输出翻转风险。512 chunk 可以让第二次请求命中，但冷启动代价
不可接受。下一步应以“第一次请求导出 checkpoint、第二次请求命中、冷/热 greedy
一致、64K 65280 可复用、额外 prefill 开销 < 5%”作为完整验收标准，而不是只验证
单算子 state 误差或服务能启动。

---

## 11. 2026-11-19 冷热一致性测试补充

> 本节为最新测试结论。当前代码已把原生 KDA checkpoint 接入模型路径，并移除了
> MTP 下关闭 checkpoint 的 spec gate；前文涉及“MTP 仍无法生成 checkpoint”的
> 段落属于 2026-11-17 的历史状态。

新增测试脚本：

`work-tests/mtp/run_prefix_cache_consistency.py`

测试环境：

- 确定性 MoE（固定 expert 到 XYS 的分核）
- `USE_CPU_FOR_COLLECT_OP=True`
- eager、prefix cache 开启、`mamba_cache_mode=align`
- `block_size=256`、`max_num_batched_tokens=4096`、`max_model_len=8192`
- 单请求；并发用例为 4 路
- `--enable-prompt-tokens-details`

### 11.1 MTP + prefix cache

| 用例 | 结果 | `cached_tokens` | 输出一致性 |
|---|---|---|---:|
| 1024 完全重复 3 次 | PASS | 0 / 512 / 512 | 一致 |
| 3584 完全重复 3 次 | **FAIL** | 0 / 3072 / 3072 | 冷/热不一致 |
| 4096 完全重复 3 次 | **FAIL** | 0 / 3584 / 3584 | 冷/热不一致 |
| 1024 + 不同 suffix | PASS | 0 / 768 / 768 / 768 | suffix 热请求一致 |
| 3584 + 不同 suffix | PASS | 0 / 3328 / 3328 / 3328 | suffix 热请求一致 |
| 1024 四路并发 | PASS | 4 路均为 512 | 4 路一致 |

3584 冷/热输出：

```text
cold: [197, 197, 532, 197, 197, 532, 197, 197]
hot:  [197, 532, 197, 532, 197, 532, 197, 532]
```

4096 冷/热输出：

```text
cold: [197, 197, 6, 197, 6, 197, 6, 197]
hot:  [197, 3, 197, 3, 197, 3, 197, 3]
```

### 11.2 no-MTP + prefix cache 对照

| 用例 | 结果 | `cached_tokens` | 输出一致性 |
|---|---|---|---:|
| 1024 完全重复 3 次 | PASS | 0 / 768 / 768 | 一致 |
| 3584 完全重复 3 次 | **FAIL** | 0 / 3328 / 3328 | 冷/热不一致 |
| 4096 完全重复 3 次 | **FAIL** | 0 / 3840 / 3840 | 冷/热不一致 |
| 1024 + 不同 suffix | PASS | 0 / 1024 / 1024 / 1024 | suffix 热请求一致 |
| 3584 + 不同 suffix | PASS | 0 / 3584 / 3584 / 3584 | suffix 热请求一致 |
| 1024 四路并发 | **FAIL** | 4 路均为 768 | 1/4 路输出不同 |

no-MTP 的 3584/4096 冷/热翻转与 MTP 完全同型，说明该问题不是 MTP 特有。
由于测试已经使用确定性 MoE 和 CPU communication，当前结果不能再用 MoE 动态分核
或 DLCCL 累加顺序解释。后续应单独排查 KDA checkpoint 冷/热数值路径，以及并发
batch 下的调度/归约一致性。

### 11.3 测试结果文件

- `work-tests/mtp/results/prefix_cache_consistency_moe_cpu_20261119.json`
- `work-tests/mtp/results/prefix_cache_consistency_nomtp_moe_cpu_20261119.json`

### 11.4 当前遗留问题

1. 3584/4096 相同 prompt 冷/热输出不一致，MTP 与 no-MTP 均可复现。
2. no-MTP 4 路并发相同 prompt 有 1/4 输出不同；MTP 4 路并发当前一致。
3. 根因暂未处理，先保留为 prefix-cache 冷热一致性已知问题；后续修复前不得用
   `512` chunk 或单个 prompt 的 `all_same_tokens=true` 代替完整回归。
