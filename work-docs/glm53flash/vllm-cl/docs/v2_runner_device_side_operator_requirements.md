# DLC V2 + Spec Decode 融合算子 Issue 清单

## 1. 范围与符号

本文覆盖当前 `V2 runner + DFlash/DSpark/MTP + GLM-5.2` 路径。每一节对应一个可独立
提交给算子组的公开算子 issue，算子使用各节标题给出的完整 `torch.*` 名称；通用新
算子默认使用 `torch.vllm_v2_*`，已有 DLC 专用 ABI 使用
`torch.dlc_vllm_v2_*`。

当前启动命令没有设置 `draft_sample_method`，因此使用上游默认值 `greedy`：

- DSpark `_sample_sequential` 走 argmax，不调用 `gumbel_sample`；
- `gumbel_sample` 只用于 target 无 draft token 时的采样，通常是 prefill 结束后的
  第一个 token；
- rejection sampler 按 one-hot draft 分布执行，`draft_logits is None`。

`draft_sample_method="probabilistic"` 是必须支持的扩展场景。该模式下 DSpark 每轮
proposal 会调用 `S` 次 `gumbel_sample`，且这些调用处于 DSpark FULL graph capture
范围内。

当前典型配置：

- `R`: 当前 request 数，`1..3`
- `Rmax`: request-state 容量，`3`
- `S`: DSpark draft token 数，`7`
- `M`: autoregressive MTP draft token 数，当前典型值 `5`
- `Q`: DSpark query token 数，`S + 1 = 8`
- `L`: target logits 行数，decode 典型上限 `R * (S + 1) = 24`
- `V`: target vocab size，`154880`
- `Tmax`: max batched tokens，`4096`
- `Bmax`: `max_model_len / block_size`，当前典型值 `8192 / 256 = 32`
- `Bg`: 当前 KV cache group 的 block-table 宽度；主模型 DCP=8 时典型 `4`，
  draft DCP=1 时典型 `32`
- `Hidx`: indexer heads，`32`
- `Didx`: indexer head dim，`128`
- `Kidx`: indexer top-k，`2048`

PyTorch native 实现允许 `.item()` 和 Python 循环，仅作为语义参考。DLC kernel 必须在
设备侧读取动态 metadata，不允许 D2H copy、tensor truth-value 同步或 stream/device
synchronize。

### 1.1 独立 issue 与公共 helper

公开 issue 之间不能存在运行时依赖。例如 `torch.vllm_v2_rejection_sample_` 不能在
kernel 外再调用一次 `torch.vllm_v2_gumbel_sample_`，否则 rejection 不再是单次融合
调用。

stateless random 和 Gumbel argmax 是 reference helper，不是新的公开 `torch.*` 算子：

- 每个 issue 的 native reference 如果调用 helper，提交 issue 时必须把 helper 一起
  放入 issue 正文；
- `gumbel_sample` issue 附带 `gumbel_block_argmax`；`rejection_sample` issue 附带
  `tl_rand32` 和 `gumbel_block_argmax`，保证两个 issue 可以独立实现和验收；
- 算子组落地时应在 C++/DLC kernel 源码中共享或 inline 同一份内部 helper，避免两套
  Philox/Gumbel 语义；
- 如果算子组希望单独跟踪公共实现，可以建立一个“不新增 Python API”的内部子任务，
  但两个公开算子 issue 的 ABI 和交付不能依赖额外 kernel launch。

## 2. 算子总览

| 优先级 | 新算子 | 当前 vllm-cl 位置 | 调用条件 / 历史 profile 次数 |
| --- | --- | --- | ---: |
| P0 | `torch.vllm_v2_prepare_pos_seq_lens_` | `v2_ops.py:1116` | 16 |
| P0 | `torch.vllm_v2_combine_sampled_and_draft_tokens_` | `v2_ops.py:1162` | 16 |
| P0 | `torch.vllm_v2_prepare_block_table_` | `v2_ops.py:1532,1545` | 16 |
| P0 | `torch.vllm_v2_compute_slot_mappings_` | `v2_ops.py:1611` | MTP profile 60 次；15 轮 proposal、`M=5` |
| P0 | `torch.vllm_v2_prepare_autoregressive_prefill_inputs_` | `v2_ops.py:935` | MTP profile 15 次 |
| P0 | `torch.vllm_v2_prepare_autoregressive_decode_inputs_` | `v2_ops.py:1021` | MTP profile 15 次 |
| P0 | `torch.vllm_v2_expand_idx_mapping_` | `v2_ops.py:1220` | 15 |
| P0 | `torch.vllm_v2_post_update_` | `v2_ops.py:1271` | 15 |
| P0 | `torch.dlc_vllm_v2_prepare_dflash_inputs_` | `v2_ops.py:1653` | 15 |
| P0 | `torch.vllm_v2_rejection_sample_` | `v2_ops.py:507` | 每轮 verification |
| P0 | `torch.vllm_v2_flatten_sampled_` | `v2_ops.py:826` | 每轮 rejection |
| 条件 P0 | `torch.vllm_v2_apply_logit_bias_` | `v2_ops.py:168` | `min_tokens`/allowed tokens/logit bias 启用时每轮 sampling |
| P1 | `torch.vllm_v2_gumbel_sample_` | `v2_ops.py:461` | greedy DSpark 下通常 1 次；probabilistic 时被 capture |
| P1 | `torch.vllm_v2_prepare_prefill_inputs_` | `v2_ops.py:896` | 1 |
| P2 | `torch.vllm_v2_pack_ragged` | `sparse_attn_indexer.py:110` | eager model path |
| P2 | `torch.vllm_v2_unpack_ragged` | `sparse_attn_indexer.py:150` | eager model path |

`get_num_sampled_and_rejected` 当前已经是纯 device tensor 表达式，不需要为了消除 CPU
同步新增算子。top-k/top-p 已接入 DLC 实现，也不在本文重复提 issue。

本清单根据 MTP profile 纳入三个 MTP 热路径：slot mapping、autoregressive prefill 和
autoregressive decode。penalties、bad words、logprobs、grammar 等可选采样功能不在
当前 benchmark 热路径；它们仍是 Python fallback，但不应混入本文这些独立算子 issue，
后续按功能和 profile 另行提单。

`logit bias` 是条件热路径：不设置 `min_tokens`、`allowed_token_ids` 和 `logit_bias` 时，
上游 `LogitBiasState` 的 host guard 会直接跳过；当前固定 1024 输出的 profile 请求设置
了 `min_tokens=1024`，因此它在每轮 sampling 执行，按条件 P0 独立提单。

### 2.1 Issue reference 依赖矩阵

| 公开算子 issue | 必须附带的 reference helper | 提交要求 |
| --- | --- | --- |
| `prepare_pos_seq_lens_` | 无 | scalar reference 直接写 `int(t.item())` |
| `combine_sampled_and_draft_tokens_` | 无 | 同上 |
| `prepare_block_table_` | 无 | 同上；同时附 DCP slot 公式 |
| `compute_slot_mappings_` | 无 | 同上；只计算 slot，不得附带 block-table gather |
| `prepare_autoregressive_prefill_inputs_` | 无 | 同上；issue 必须列出拆平后的全部 tensor ABI |
| `prepare_autoregressive_decode_inputs_` | 无 | 同上；接收完整二维 draft buffer，不接收跨行 column view |
| `expand_idx_mapping_` | 无 | 同上 |
| `post_update_` | 无 | 同上 |
| `dlc_vllm_v2_prepare_dflash_inputs_` | 无 | 同上 |
| `gumbel_sample_` | `gumbel_block_argmax`；后者内部使用 RNG 语义 | 必须附 3.7.1 全部代码 |
| `rejection_sample_` | `tl_rand32`、`gumbel_block_argmax` | 必须附 3.7.1 全部代码，不得调用公开 `gumbel_sample_` |
| `flatten_sampled_` | 无 | scalar reference 直接写 `int(t.item())` |
| `apply_logit_bias_` | 无 | scalar reference 直接写 `int(t.item())` |
| `prepare_prefill_inputs_` | 无 | 同上 |
| `pack_ragged` / `unpack_ragged` | 无公共 helper | native 中的 `.item()` 只用于表达动态 shape 语义 |

native reference 中的 `int(t.item())` 只用于表达控制流。所有设备算子都必须直接读取
tensor，不能把 `.item()` 对应成 host callback。

## 3. Decode 算子（P0/P1）

### 3.1 `torch.vllm_v2_prepare_pos_seq_lens_`

**PyTorch native**

```python
def vllm_v2_prepare_pos_seq_lens_native(
    idx_mapping: torch.Tensor,          # int32 [R], typical [1..3]
    query_start_loc: torch.Tensor,      # int32 [R+1], typical [2..4]
    num_computed_tokens: torch.Tensor,  # int32 [Rmax], typical [3]
    pos: torch.Tensor,                  # int64 [Tmax], in-place output, typical [4096]
    seq_lens: torch.Tensor,             # int32 [Rmax], in-place output, typical [3]
) -> None:
    num_reqs = idx_mapping.shape[0]
    active_seq_lens = (
        num_computed_tokens[idx_mapping.long()].to(seq_lens.dtype)
        + query_start_loc[1:num_reqs + 1]
        - query_start_loc[:num_reqs]
    )
    seq_lens.copy_(
        torch.cat(
            (
                active_seq_lens,
                active_seq_lens.new_zeros(seq_lens.shape[0] - num_reqs),
            )
        )
    )
    for batch_idx, req_state_idx_t in enumerate(idx_mapping):
        req_state_idx = int(req_state_idx_t.item())
        computed = int(num_computed_tokens[req_state_idx].item())
        start = int(query_start_loc[batch_idx].item())
        end = int(query_start_loc[batch_idx + 1].item())
        query_len = end - start
        pos[start:end].copy_(
            torch.arange(
                computed,
                computed + query_len,
                dtype=pos.dtype,
                device=pos.device,
            )
        )
```

**vllm-cl 接入位置**

- 替换 `/home/workspace/vllm-cl/vllm_cl/worker/v2_ops.py:1116`
  `prepare_pos_seq_lens`。
- `install_v2_fallbacks()` 中保持对上游 `model_runner.prepare_pos_seq_lens` 的 patch，
  wrapper 内改为调用新算子。

### 3.2 `torch.vllm_v2_combine_sampled_and_draft_tokens_`

**PyTorch native**

```python
def vllm_v2_combine_sampled_and_draft_tokens_native(
    input_ids: torch.Tensor,           # int32 [Tmax], in-place, typical [4096]
    idx_mapping: torch.Tensor,         # int32 [R], typical [1..3]
    last_sampled_tokens: torch.Tensor, # int64 [Rmax,1], typical [3,1]
    query_start_loc: torch.Tensor,     # int32 [R+1], typical [2..4]
    seq_lens: torch.Tensor,            # int32 [R], typical [1..3]
    prefill_len: torch.Tensor,         # int32 [Rmax], typical [3]
    draft_tokens: torch.Tensor,        # int64 [Rmax,S], typical [3,7]
    cu_num_logits: torch.Tensor,       # int32 [R+1], typical [2..4]
    num_logits: int,                   # host int, output length L, typical 1..24
    num_new_sampled_tokens: int = 1,   # host int, DSpark typical 1
) -> torch.Tensor:
    logits_indices = torch.empty(
        num_logits,
        dtype=torch.int64,
        device=input_ids.device,
    )
    for batch_idx, req_state_idx_t in enumerate(idx_mapping):
        req_state_idx = int(req_state_idx_t.item())
        logits_out_start = int(cu_num_logits[batch_idx].item())
        logits_out_end = int(cu_num_logits[batch_idx + 1].item())
        req_num_logits = logits_out_end - logits_out_start
        num_drafts = req_num_logits - num_new_sampled_tokens
        query_end = int(query_start_loc[batch_idx + 1].item())
        logits_start = query_end - req_num_logits

        logits_indices[logits_out_start:logits_out_end].copy_(
            torch.arange(
                logits_start,
                query_end,
                dtype=torch.int64,
                device=input_ids.device,
            )
        )

        seq_len = int(seq_lens[batch_idx].item())
        prompt_len = int(prefill_len[req_state_idx].item())
        if seq_len <= prompt_len:
            continue
        first_logit_seq_pos = seq_len - req_num_logits
        if num_new_sampled_tokens and first_logit_seq_pos >= prompt_len:
            input_ids[logits_start].copy_(last_sampled_tokens[req_state_idx, 0])
        if num_drafts:
            input_ids[query_end - num_drafts:query_end].copy_(
                draft_tokens[req_state_idx, :num_drafts].to(input_ids.dtype)
            )
    return logits_indices
```

**vllm-cl 接入位置**

- 替换 `vllm_cl/worker/v2_ops.py:1162`。
- wrapper 与当前 fallback 一样，按 host-known `num_logits` 创建并返回
  `logits_indices`。

### 3.3 `torch.vllm_v2_prepare_block_table_`

单次调用处理一个 KV cache group，并融合 block-table gather 与 slot mapping。
该 native 是 `DLCBlockTables.gather_block_tables` 和 `compute_slot_mappings` 两段现有
fallback 的机械合并；除了把两段共同的 request loop 合并外，不改变二者输出语义。

主模型和 draft model 可以属于不同 KV cache group，并使用不同的 DCP size。例如主模型
group 使用 DCP=8，而不支持 DCP 的 DSpark/DFlash draft attention group 使用 DCP=1。
因此 `cp_size` 和 `cp_rank` 是当前 `group_id` 的参数，不能从全局 DCP 配置统一处理所有
group。

`BlockTables.slot_mappings` 是持久化的二维 `[G,Tmax]` buffer。`slot_mappings[group_id]`
虽然 contiguous，但当 `group_id > 0` 时带有非零 `storage_offset`，DLC custom kernel
不能正确处理这种 tensor view。因此算子 ABI 必须接收完整二维 buffer 和 `group_id`，由
kernel 按 `group_id * slot_mappings.stride(0)` 定位输出行，不能由 Python wrapper 先取
一维 row view 再传入算子。

`BlockTables.num_blocks.gpu` 同样是二维 `[G,Rmax]` buffer，必须完整传入算子并在
kernel 内读取 `num_blocks[group_id, req_state_idx]`；不能传
`num_blocks[group_id]` row view。

**PyTorch native**

```python
def vllm_v2_prepare_block_table_native(
    input_block_table: torch.Tensor,   # int32 [Rmax,Bg], in-place output; target [3,4], draft [3,32]
    slot_mappings: torch.Tensor,       # int64 [G,Tmax], in-place output, typical [2,4096]
    group_id: int,                     # host int, current KV cache group, typical 0..G-1
    idx_mapping: torch.Tensor,         # int32 [R], typical [1..3]
    query_start_loc: torch.Tensor,     # int32 [R+1], typical [2..4]
    positions: torch.Tensor,           # int64 [Tpad], typical [8..24]
    source_block_table: torch.Tensor,  # int32 [Rmax,Bg]; target [3,4], draft [3,32]
    num_blocks: torch.Tensor,          # int32 [G,Rmax], typical [2,3]
    num_reqs_padded: int,              # host int, typical 1..3
    num_tokens_padded: int,            # host int, typical 8..24, max 4096
    block_size: int,                   # host int, typical 256
    cp_size: int,                      # host int, current group DCP size; target typical 8, draft 1
    cp_rank: int,                      # host int, current group rank; 0 when cp_size=1
    cp_interleave: int,                # host int, vLLM DCP interleave
    pad_slot_id: int = -1,             # host int
) -> tuple[torch.Tensor, torch.Tensor]:
    assert 0 <= group_id < slot_mappings.shape[0]
    assert 0 <= num_tokens_padded <= slot_mappings.shape[1]
    input_block_table[:num_reqs_padded].zero_()
    slot_mapping = slot_mappings[group_id]
    slot_mapping.fill_(pad_slot_id)

    for batch_idx, req_state_idx_t in enumerate(idx_mapping):
        req_state_idx = int(req_state_idx_t.item())
        req_num_blocks = int(num_blocks[group_id, req_state_idx].item())
        input_block_table[batch_idx, :req_num_blocks].copy_(
            source_block_table[req_state_idx, :req_num_blocks]
        )

        start = int(query_start_loc[batch_idx].item())
        end = int(query_start_loc[batch_idx + 1].item())
        req_positions = positions[start:end]
        cp_block_size = block_size * cp_size
        block_indices = req_positions // cp_block_size
        block_offsets = req_positions % cp_block_size
        block_numbers = source_block_table[
            req_state_idx, block_indices.long()
        ]
        if cp_size == 1:
            slots = block_numbers * block_size + block_offsets
        else:
            is_local = (
                block_offsets // cp_interleave % cp_size == cp_rank
            )
            rounds = block_offsets // (cp_interleave * cp_size)
            remainder = block_offsets % cp_interleave
            local_offsets = rounds * cp_interleave + remainder
            slots = block_numbers * block_size + local_offsets
            slots = torch.where(
                is_local,
                slots,
                torch.full_like(slots, pad_slot_id),
            )
        slot_mapping[start:end].copy_(slots)
    return input_block_table, slot_mappings
```

**vllm-cl 接入位置**

- 替换 `vllm_cl/worker/v2_ops.py:1532` 中
  `DLCBlockTables.gather_block_tables` 和 `compute_slot_mappings`。
- 在 `vllm_cl/worker/dlc_model_runner_v2.py:DLCModelRunnerV2` override
  `prepare_attn`，一次调用融合 method，不再先后执行两个 Python fallback。
- 多 KV cache group 由 wrapper 按 group 各调用一次。每次传完整
  `self.slot_mappings` 和当前 `group_id`，禁止传
  `self.slot_mappings[group_id]` view。
- 每次调用还必须传完整 `self.num_blocks.gpu`，禁止传
  `self.num_blocks.gpu[group_id]` view。
- wrapper 必须按 group 传入 `self.group_dcp_sizes[group_id]`；对应 `cp_rank` 为
  `self.cp_rank if cp_size > 1 else 0`。禁止直接对所有 group 使用全局
  `self.cp_size/self.cp_rank`。
- kernel 必须通过二维 buffer 的 stride 定位当前 group 的输出行，并保持整个
  `slot_mappings` 持久化地址不变，以兼容 graph capture。
- 直接复用 `BlockTables.num_blocks.gpu` 的 persistent int32 `[G,Rmax]` tensor；不能在
  kernel 调用前从 `BlockTables.num_blocks.np` 临时构造或同步。

### 3.4 `torch.vllm_v2_expand_idx_mapping_`

**PyTorch native**

```python
def vllm_v2_expand_idx_mapping_native(
    idx_mapping: torch.Tensor,      # int32 [R], typical [1..3]
    total_num_logits: int,          # host int, L, typical 1..24
    cu_num_logits: torch.Tensor,    # int32 [R+1], typical [2..4]
    max_expand_len: int,            # host int, typical S+1=8
) -> tuple[torch.Tensor, torch.Tensor]:
    del max_expand_len
    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    expanded_local_pos = torch.empty(
        total_num_logits,
        dtype=torch.int32,
        device=idx_mapping.device,
    )
    for req_idx, req_state_idx in enumerate(idx_mapping):
        start = int(cu_num_logits[req_idx].item())
        end = int(cu_num_logits[req_idx + 1].item())
        expanded_idx_mapping[start:end] = req_state_idx
        expanded_local_pos[start:end].copy_(
            torch.arange(
                end - start,
                dtype=torch.int32,
                device=idx_mapping.device,
            )
        )
    return expanded_idx_mapping, expanded_local_pos
```

**vllm-cl 接入位置**

- 替换 `vllm_cl/worker/v2_ops.py:1220`。
- 与当前 fallback 一样，按 host-known `total_num_logits` 创建两个 output tensor。

### 3.5 `torch.vllm_v2_post_update_`

**PyTorch native**

```python
def vllm_v2_post_update_native(
    idx_mapping: torch.Tensor,            # int32 [R], may contain -1, typical [1..3]
    num_computed_tokens: torch.Tensor,    # int32 [Rmax], in-place, typical [3]
    last_sampled_tokens: torch.Tensor,    # int64 [Rmax,1], in-place, typical [3,1]
    output_bin_counts: torch.Tensor | None,  # int32 [Rmax,V] or None
    sampled_tokens: torch.Tensor,         # int64 [R,S+1], typical [1..3,8]
    num_sampled: torch.Tensor,            # int32 [R], typical [1..3]
    num_rejected: torch.Tensor,           # int32 [R], typical [1..3]
    query_start_loc: torch.Tensor | None, # int32 [R+1] or None
    all_token_ids: torch.Tensor,          # int32 [Rmax,max_model_len], in-place, [3,8192]
    total_len: torch.Tensor,              # int32 [Rmax], in-place, typical [3]
) -> None:
    valid_requests: list[tuple[int, int]] = []
    for batch_idx, req_state_idx_tensor in enumerate(idx_mapping):
        req_state_idx = int(req_state_idx_tensor.item())
        if req_state_idx >= 0:
            valid_requests.append((batch_idx, req_state_idx))
    if not valid_requests:
        return

    batch_indices = torch.tensor(
        [batch_idx for batch_idx, _ in valid_requests],
        dtype=torch.long,
        device=idx_mapping.device,
    )
    req_state_indices = torch.tensor(
        [req_state_idx for _, req_state_idx in valid_requests],
        dtype=torch.long,
        device=idx_mapping.device,
    )
    old_totals = total_len[req_state_indices]
    sampled_counts = num_sampled[batch_indices]

    for request_idx, (batch_idx, req_state_idx) in enumerate(valid_requests):
        old_total = int(old_totals[request_idx].item())
        sampled_count = int(sampled_counts[request_idx].item())
        if sampled_count:
            tokens = sampled_tokens[batch_idx, :sampled_count]
            all_token_ids[
                req_state_idx, old_total:old_total + sampled_count
            ].copy_(tokens.to(all_token_ids.dtype))
            if output_bin_counts is not None:
                output_bin_counts[req_state_idx].scatter_add_(
                    0,
                    tokens.long(),
                    torch.ones_like(tokens, dtype=output_bin_counts.dtype),
                )

    new_totals = old_totals + sampled_counts.to(old_totals.dtype)
    total_len.copy_(
        total_len.scatter(0, req_state_indices, new_totals)
    )

    last_offsets = (sampled_counts.long() - 1).clamp_min(0)
    last_tokens = sampled_tokens[batch_indices].gather(
        1, last_offsets.unsqueeze(1)
    )
    current_last = last_sampled_tokens[req_state_indices]
    updated_last = torch.where(
        sampled_counts.unsqueeze(1) > 0,
        last_tokens.to(current_last.dtype),
        current_last,
    )
    last_sampled_tokens.copy_(
        last_sampled_tokens.scatter(
            0,
            req_state_indices.unsqueeze(1).expand_as(updated_last),
            updated_last,
        )
    )

    if query_start_loc is None:
        query_lens = torch.zeros_like(sampled_counts)
    else:
        query_lens = (
            query_start_loc[batch_indices + 1] - query_start_loc[batch_indices]
        )
    computed_deltas = query_lens - num_rejected[batch_indices].to(query_lens.dtype)
    current_computed = num_computed_tokens[req_state_indices]
    num_computed_tokens.copy_(
        num_computed_tokens.scatter(
            0,
            req_state_indices,
            current_computed + computed_deltas.to(current_computed.dtype),
        )
    )
```

**vllm-cl 接入位置**

- 替换 `vllm_cl/worker/v2_ops.py:1271`。
- 保持对 `model_runner.post_update` 的 patch；所有 state tensor 原地更新。

### 3.6 `torch.dlc_vllm_v2_prepare_dflash_inputs_`

该算子同时覆盖 DFlash 与 DSpark 共用的 `prepare_dflash_inputs` 入口。
`sample_from_anchor=True` 表示 anchor 本身参与采样，使用 `N` 个 query slot；
`sample_from_anchor=False` 表示 anchor 是独立 bonus token，使用 `1+N` 个 query
slot。调用方必须按实际 draft checkpoint 传入该参数，并保证
`num_query_per_req` 与对应布局一致。

**PyTorch native**

```python
def dlc_vllm_v2_prepare_dflash_inputs_native(
    out_input_ids: torch.Tensor,          # int32 [Tmax], in-place output, typical [4096]
    out_query_positions: torch.Tensor,    # int64 [Tmax], in-place output, typical [4096]
    out_query_start_loc: torch.Tensor,    # int32 [Rmax+1], output, typical [4]
    out_seq_lens: torch.Tensor,           # int32 [Rmax], output, typical [3]
    out_query_slots: torch.Tensor,        # int64 [Tmax], output, typical [4096]
    out_context_positions: torch.Tensor,  # int64 [Tmax], output, typical [4096]
    out_context_slots: torch.Tensor,      # int64 [Tmax], output, typical [4096]
    out_sample_indices: torch.Tensor,     # int64 [Rmax*S], output, typical [21]
    out_sample_pos: torch.Tensor,         # int64 [Rmax*S], output, typical [21]
    out_sample_idx: torch.Tensor,         # int32 [Rmax*S], output, typical [21]
    target_positions: torch.Tensor,       # int64 [Ttarget_pad], typical [8..24]
    target_query_start_loc: torch.Tensor, # int32 [R+1], typical [2..4]
    idx_mapping: torch.Tensor,            # int32 [R], typical [1..3]
    num_sampled: torch.Tensor,            # int32 [R], typical [1..3]
    num_rejected: torch.Tensor,           # int32 [R], typical [1..3]
    last_sampled: torch.Tensor,           # int64 [Rmax,1], typical [3,1]
    next_prefill_tokens: torch.Tensor,    # int32 [Rmax], typical [3]
    block_table: torch.Tensor,            # int32 [Rpad,Bmax], typical [3,32]
    block_size: int,                      # host int, typical 256
    mask_token_id: int,                   # host int, DFlash/DSpark parallel draft token id
    num_query_per_req: int,               # host int, Q=7 (anchor) or 8 (bonus anchor)
    num_speculative_steps: int,           # host int, S=7
    max_num_reqs: int,                    # host int, Rmax=3
    max_num_tokens: int,                  # host int, Tmax=4096
    max_model_len: int,                   # host int, 8192
    sample_from_anchor: bool,              # host bool; True: N slots, False: 1+N slots
    pad_slot_id: int = -1,                # host int
) -> None:
    num_reqs = idx_mapping.shape[0]
    query_count = num_reqs * num_query_per_req
    out_query_slots[query_count:max_num_tokens].fill_(pad_slot_id)
    out_query_start_loc.copy_(
        torch.arange(
            max_num_reqs + 1,
            dtype=torch.int32,
            device=idx_mapping.device,
        ).clamp(max=num_reqs) * num_query_per_req
    )
    pad_sample_start = num_reqs * num_speculative_steps
    pad_sample_end = max_num_reqs * num_speculative_steps
    out_sample_indices[pad_sample_start:pad_sample_end].zero_()
    out_sample_pos[pad_sample_start:pad_sample_end].zero_()
    out_sample_idx[pad_sample_start:pad_sample_end].fill_(-1)

    for req_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[req_idx].item())
        ctx_start = int(target_query_start_loc[req_idx].item())
        ctx_end = int(target_query_start_loc[req_idx + 1].item())
        valid_ctx_end = ctx_end - int(num_rejected[req_idx].item())
        bonus = (
            last_sampled[req_state_idx, 0]
            if int(num_sampled[req_idx].item()) > 0
            else next_prefill_tokens[req_state_idx]
        )

        ctx_pos = target_positions[ctx_start:ctx_end]
        out_context_positions[ctx_start:ctx_end].copy_(ctx_pos)
        ctx_block_idx = (ctx_pos // block_size).clamp(max=block_table.shape[1] - 1)
        ctx_block_num = block_table[req_idx, ctx_block_idx.long()]
        out_context_slots[ctx_start:ctx_end].copy_(
            ctx_block_num * block_size + ctx_pos % block_size
        )

        last_valid_pos = int(target_positions[valid_ctx_end - 1].item())
        query_base = req_idx * num_query_per_req
        query_end = query_base + num_query_per_req
        query_pos = torch.arange(
            last_valid_pos + 1,
            last_valid_pos + 1 + num_query_per_req,
            dtype=torch.int64,
            device=idx_mapping.device,
        )
        out_input_ids[query_base:query_end].fill_(mask_token_id)
        out_input_ids[query_base] = bonus
        out_query_positions[query_base:query_end].copy_(
            query_pos.clamp(max=max_model_len - 1)
        )
        q_block_idx = (query_pos // block_size).clamp(max=block_table.shape[1] - 1)
        q_block_num = block_table[req_idx, q_block_idx.long()]
        out_query_slots[query_base:query_end].copy_(
            q_block_num * block_size + query_pos % block_size
        )

        sample_start = req_idx * num_speculative_steps
        sample_end = sample_start + num_speculative_steps
        sample_offset = 0 if sample_from_anchor else 1
        query_sample_start = query_base + sample_offset
        query_sample_end = query_sample_start + num_speculative_steps
        out_sample_indices[sample_start:sample_end].copy_(
            torch.arange(
                query_sample_start,
                query_sample_end,
                dtype=torch.int64,
                device=idx_mapping.device,
            )
        )
        predicted_pos = query_pos[
            sample_offset : sample_offset + num_speculative_steps
        ]
        if sample_from_anchor:
            predicted_pos = predicted_pos + 1
        out_sample_pos[sample_start:sample_end].copy_(predicted_pos)
        out_sample_idx[sample_start:sample_end].fill_(req_state_idx)

    valid_ctx_ends = (
        target_query_start_loc[1:num_reqs + 1] - num_rejected[:num_reqs]
    )
    active_seq_lens = (
        target_positions[(valid_ctx_ends - 1).long()].to(out_seq_lens.dtype)
        + 1
        + num_query_per_req
    )
    out_seq_lens.copy_(
        torch.cat(
            (
                active_seq_lens,
                active_seq_lens.new_zeros(max_num_reqs - num_reqs),
            )
        )
    )
```

**vllm-cl 接入位置**

- 替换 `vllm_cl/worker/v2_ops.py:1653` `prepare_dflash_inputs`。
- `install_v2_fallbacks()` 中 DFlash 与 DSpark 都通过上游
  `dflash.prepare_dflash_inputs` 入口调用；DLC wrapper 必须把运行时的
  `sample_from_anchor` 原样传给算子，不能根据模型类型写死。
- 算子测试必须同时覆盖 `sample_from_anchor=True/False`，并分别检查
  `out_sample_indices` 使用 `query_base` 和 `query_base + 1` 作为起点。

### 3.7 `torch.vllm_v2_gumbel_sample_`

当前 greedy DSpark 不在 proposal 中调用该函数；它仍用于 target 无 draft token 的
采样。启用 `draft_sample_method="probabilistic"` 后，它会进入 DSpark FULL graph，
因此设备实现必须动态读取 seed、position、temperature 和 mapping。

#### 3.7.1 必须随 issue 提交的公共 reference helper

下面三个 helper 共同定义 RNG/Gumbel 语义：`gumbel_block_argmax` 被 3.7 和 3.8
共用，`tl_rand32` 只被 3.8 的接受判定使用，`philox_randint` 是
`tl.randint(seed, position)` 的 CPU reference。它们不是独立的公开算子，也不能在
运行时形成额外 kernel launch。提交 issue 时按 2.1 的依赖矩阵附带相应 reference。

```python
U32_MASK = 0xFFFFFFFF


def philox_randint(seed: int, position: int) -> int:
    # CPU scalar equivalent of tl.randint(seed, position).
    counter = (position & U32_MASK, position >> 32, 0, 0)
    key = (seed & U32_MASK, (seed >> 32) & U32_MASK)
    for round_idx in range(10):
        product0 = 0xD2511F53 * counter[0]
        product1 = 0xCD9E8D57 * counter[2]
        counter = (
            ((product1 >> 32) ^ counter[1] ^ key[0]) & U32_MASK,
            product1 & U32_MASK,
            ((product0 >> 32) ^ counter[3] ^ key[1]) & U32_MASK,
            product0 & U32_MASK,
        )
        if round_idx != 9:
            key = (
                (key[0] + 0x9E3779B9) & U32_MASK,
                (key[1] + 0xBB67AE85) & U32_MASK,
            )
    return counter[0]


def tl_rand32(
    seed: int,                  # host value read from int64 [Rmax]
    offset: torch.Tensor,       # int32/int64 scalar logical token position
    includes_zero: bool,        # host bool
) -> torch.Tensor:
    position = int(offset.item())
    generator = torch.Generator(device=offset.device)
    generator.manual_seed(seed)
    # DLC random requires Generator offsets to be multiples of four. This is
    # backend alignment; the logical vLLM RNG offset is still `position`.
    generator.set_offset(position * 4)
    uniform = torch.rand(
        (),
        dtype=torch.float32,
        device=offset.device,
        generator=generator,
    )
    if not includes_zero:
        uniform.clamp_min_(4.6566127342e-10)
    return uniform


def gumbel_block_argmax(
    logits: torch.Tensor,       # bf16/fp32 [V], typical [154880]
    temperature: float,         # host value read from fp32 [Rmax]
    seed: int,                  # host value read from int64 [Rmax]
    pos: torch.Tensor,          # int32/int64 scalar logical token position
    use_fp64: bool,             # host bool
) -> torch.Tensor:
    row = logits.float()
    if temperature != 0.0:
        position = int(pos.item())
        gumbel_seed = philox_randint(seed, position)

        generator = torch.Generator(device=row.device)
        generator.manual_seed(gumbel_seed)
        generator.set_offset(0)
        sample_dtype = torch.float64 if use_fp64 else torch.float32
        uniform = torch.rand(
            row.shape[0],
            dtype=sample_dtype,
            device=row.device,
            generator=generator,
        )
        min_uniform = (
            torch.finfo(torch.float64).tiny
            if use_fp64
            else 4.6566127342e-10
        )
        uniform.clamp_min_(min_uniform)
        if use_fp64:
            gumbel_noise = -torch.log(-torch.log(uniform))
            row = row.double()
        else:
            gumbel_noise = -torch.log(-torch.log1p(-uniform))
        row = row + gumbel_noise
    return row.argmax()
```

设备实现要求：

- `tl_rand32(seed, pos, ...)` 的逻辑 key 是 `(seed, pos)`；`pos * 4`
  只是 DLC random 的 offset 对齐方式，不能在 ABI 层把 position 改成四倍；
- Gumbel seed 必须由 `(seed, pos)` 在设备侧派生，随后按 vocab offset 生成噪声；
- FP32 使用 flipped Gumbel `-log(-log1p(-u))`，下限为
  `4.6566127342e-10`；FP64 使用 `-log(-log(u))`；
- 设备实现可以共享 Philox/Gumbel 内部 utility，但不得暴露或调用额外 Python op。

#### 3.7.2 PyTorch native

```python
def vllm_v2_gumbel_sample_native(
    logits: torch.Tensor,               # bf16/fp32 [L,V], typical [1..24,154880]
    expanded_idx_mapping: torch.Tensor, # int32 [L], typical [1..24]
    temperature: torch.Tensor,          # fp32 [Rmax], typical [3]
    seed: torch.Tensor,                 # int64 [Rmax], typical [3]
    pos: torch.Tensor,                  # int32/int64 [L], typical [1..24]
    apply_temperature: bool,            # host bool
    output_processed_logits: torch.Tensor | None = None, # fp32 [Rmax,S,V] or None
    output_processed_logits_col: torch.Tensor | None = None, # int32 scalar/[L]/None
    use_fp64: bool = False,             # host bool
) -> torch.Tensor:
    num_tokens, vocab_size = logits.shape
    sampled = torch.zeros(num_tokens, dtype=torch.int64, device=logits.device)

    for token_idx in range(num_tokens):
        req_state_idx = int(expanded_idx_mapping[token_idx].item())
        if req_state_idx < 0:
            continue
        row = logits[token_idx].float()
        temp = float(temperature[req_state_idx].item())
        if apply_temperature and temp != 0.0:
            row = row / temp

        if output_processed_logits is not None:
            if output_processed_logits_col is None:
                col = 0
            elif output_processed_logits_col.ndim == 0:
                col = int(output_processed_logits_col.item())
            else:
                col = int(output_processed_logits_col[token_idx].item())
            output_processed_logits[req_state_idx].view(-1, vocab_size)[col].copy_(
                row.to(output_processed_logits.dtype)
            )

        sampled[token_idx] = gumbel_block_argmax(
            row,
            temp,
            int(seed[req_state_idx].item()),
            pos[token_idx],
            use_fp64,
        )
    return sampled
```

**vllm-cl 接入位置**

- 替换 `vllm_cl/worker/v2_ops.py:461`。
- patch 入口包括 `sampler.gumbel_sample`、`speculator.gumbel_sample` 和
  `dspark.gumbel_sample`。
- 当前 fallback helper 位于 `vllm_cl/worker/v2_ops.py:403` 和 `:425`。

### 3.8 `torch.vllm_v2_rejection_sample_`

这是单一融合算子，不再区分 `_rejection_sample_dlc` 和
`_rejection_sample_torch`。它的 ABI 与上游
`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:rejection_sample` 一致，覆盖
standard rejection、synthetic acceptance、FP64 Gumbel 和 block verification。

下面的 reference 对齐当前 `vllm_cl/worker/v2_ops.py:507`。本 issue 必须同时附带
3.7.1 的 `tl_rand32` 和 `gumbel_block_argmax`。它们只是完整描述 RNG/Gumbel
语义的 reference helper，不是另一个待新增设备算子；设备实现必须将它们和
rejection 主体融合为一次调用。

greedy/random 仍是逐 request 的必要语义，由
`temperature[idx_mapping] == 0` 在设备侧选择。不能再通过 Python/CPU 计算
`all_greedy`、`all_random`，也不能拆成多个旧 DLC rejection op。仅当
`draft_sample_method="probabilistic"` 时 DSpark 才传入 `draft_logits`；其中保存的是
已经应用 temperature 的 processed logits，新算子不得再次除以 temperature。当前
greedy DSpark 传入 `draft_logits=None`，按 one-hot draft 分布处理。

`draft_sampled` 的真实运行时 dtype 是 `int32`。Model Runner V2 直接通过
`input_batch.input_ids[input_batch.logits_indices]` 构造该张量，而
`InputBuffers.input_ids` 原生以 `torch.int32` 分配。设备算子必须直接接受这个
`int32 [L]` 输入；若内部索引或比较需要 int64，应在算子内部提升，不能要求
vLLM-DLC 在每轮 verification 前执行 `.long()` 或额外 copy。这里的输入
`draft_sampled` 不要与返回的 int64 `sampled` 混淆。

**PyTorch native**

```python
import math


def vllm_v2_rejection_sample_native(
    target_logits: torch.Tensor,        # in: bf16/fp32 [L,V], typical [1..24,154880]
    draft_logits: torch.Tensor | None, # in: bf16/fp32 [Rmax,S,V] or None, [3,7,154880]
    draft_sampled: torch.Tensor,        # in: int32 [L], typical [1..24]
    cu_num_logits: torch.Tensor,        # in: int32 [R+1], typical [2..4]
    pos: torch.Tensor,                  # in: int32/int64 [L], typical [1..24]
    idx_mapping: torch.Tensor,          # in: int32 [R], typical [1..3]
    expanded_idx_mapping: torch.Tensor,# in: int32 [L], typical [1..24]
    expanded_local_pos: torch.Tensor,  # in: int32 [L], typical [1..24]
    temperature: torch.Tensor,          # in: fp32 [Rmax], typical [3]
    seed: torch.Tensor,                 # in: int64 [Rmax], typical [3]
    num_speculative_steps: int,         # in: host int, typical S=7
    synthetic_conditional_rates: torch.Tensor | None = None, # in: fp32 [S] or None
    use_fp64: bool = False,             # in: host bool
    use_block_verification: bool = False, # in: host bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_block_verification and synthetic_conditional_rates is not None:
        raise AssertionError(
            "Block verification is incompatible with synthetic acceptance rates."
        )

    num_reqs = cu_num_logits.shape[0] - 1
    vocab_size = target_logits.shape[-1]
    if draft_logits is not None:
        vocab_size = min(vocab_size, draft_logits.shape[-1])

    sampled = torch.full(
        (num_reqs, num_speculative_steps + 1),
        -1,
        dtype=torch.int64,
        device=target_logits.device,
    )
    num_sampled = torch.empty(
        num_reqs,
        dtype=torch.int32,
        device=target_logits.device,
    )

    for req_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[req_idx].item())
        start = int(cu_num_logits[req_idx].item())
        end = int(cu_num_logits[req_idx + 1].item())
        num_drafts = end - start - 1
        req_seed = int(seed[req_state_idx].item())
        temp = float(temperature[req_state_idx].item())
        greedy = temp == 0.0

        target_log_probs: list[torch.Tensor] = []
        draft_log_probs: list[torch.Tensor] | None = (
            [] if draft_logits is not None else None
        )
        if not greedy:
            for draft_step in range(num_drafts):
                logit_idx = start + draft_step
                target_log_probs.append(
                    target_logits[logit_idx, :vocab_size]
                    .float()
                    .log_softmax(dim=-1)
                )
                if draft_log_probs is not None:
                    draft_req_idx = int(
                        expanded_idx_mapping[logit_idx].item()
                    )
                    draft_local_pos = int(
                        expanded_local_pos[logit_idx].item()
                    )
                    # DSpark stores temperature-processed draft logits.
                    draft_log_probs.append(
                        draft_logits[
                            draft_req_idx,
                            draft_local_pos,
                            :vocab_size,
                        ]
                        .float()
                        .log_softmax(dim=-1)
                    )

        cumulative_log_p: list[float] | None = None
        if use_block_verification and not greedy:
            cumulative_log_p = []
            log_p = 0.0
            for draft_step in range(num_drafts):
                logit_idx = start + draft_step
                draft_token = int(draft_sampled[logit_idx + 1].item())
                if not 0 <= draft_token < vocab_size:
                    log_p = float("-inf")
                else:
                    target_lp = float(
                        target_log_probs[draft_step][draft_token].item()
                    )
                    draft_lp = (
                        float(
                            draft_log_probs[draft_step][draft_token].item()
                        )
                        if draft_log_probs is not None
                        else 0.0
                    )
                    log_p = min(log_p + target_lp - draft_lp, 0.0)
                cumulative_log_p.append(log_p)

        if use_block_verification and not greedy:
            assert cumulative_log_p is not None
            accepted = 0
            for draft_step in range(num_drafts):
                logit_idx = start + draft_step
                draft_token = int(draft_sampled[logit_idx + 1].item())
                valid_draft = 0 <= draft_token < vocab_size
                sampled[req_idx, draft_step] = draft_token

                uniform = float(
                    tl_rand32(
                        req_seed,
                        pos[logit_idx],
                        includes_zero=False,
                    ).item()
                )

                prefix_joint_ratio = math.exp(
                    cumulative_log_p[draft_step]
                )
                if draft_step == num_drafts - 1:
                    threshold = prefix_joint_ratio
                else:
                    next_target_probs = target_log_probs[
                        draft_step + 1
                    ].exp()
                    if draft_log_probs is None:
                        next_logit_idx = logit_idx + 1
                        next_draft_token = int(
                            draft_sampled[next_logit_idx + 1].item()
                        )
                        if 0 <= next_draft_token < vocab_size:
                            residual_mass = prefix_joint_ratio * (
                                1.0
                                - float(
                                    next_target_probs[
                                        next_draft_token
                                    ].item()
                                )
                            )
                        else:
                            residual_mass = prefix_joint_ratio
                    else:
                        next_draft_probs = draft_log_probs[
                            draft_step + 1
                        ].exp()
                        residual_mass = float(
                            (
                                prefix_joint_ratio
                                * next_target_probs
                                - next_draft_probs
                            )
                            .clamp_min(0.0)
                            .sum()
                            .item()
                        )
                    denominator = (
                        residual_mass + 1.0 - prefix_joint_ratio
                    )
                    threshold = (
                        residual_mass / denominator
                        if denominator > 0.0
                        else 1.0
                    )

                if valid_draft and uniform <= threshold:
                    accepted = draft_step + 1

            resample_step = accepted
        else:
            accepted = 0
            greedy_rejected = False
            for draft_step in range(num_drafts):
                logit_idx = start + draft_step
                draft_token = int(draft_sampled[logit_idx + 1].item())
                valid_draft = 0 <= draft_token < vocab_size

                uniform = float(
                    tl_rand32(
                        req_seed,
                        pos[logit_idx],
                        includes_zero=False,
                    ).item()
                )

                if greedy:
                    target_argmax = int(
                        target_logits[
                            logit_idx, :vocab_size
                        ].argmax().item()
                    )
                    if synthetic_conditional_rates is None:
                        accept = (
                            valid_draft
                            and draft_token == target_argmax
                        )
                    else:
                        accept = (
                            valid_draft
                            and uniform
                            < float(
                                synthetic_conditional_rates[
                                    draft_step
                                ].item()
                            )
                        )
                    sampled[req_idx, draft_step] = (
                        draft_token if accept else target_argmax
                    )
                    if not accept:
                        greedy_rejected = True
                        break
                else:
                    if synthetic_conditional_rates is not None:
                        accept = (
                            valid_draft
                            and uniform
                            < float(
                                synthetic_conditional_rates[
                                    draft_step
                                ].item()
                            )
                        )
                    elif valid_draft:
                        target_lp = float(
                            target_log_probs[
                                draft_step
                            ][draft_token].item()
                        )
                        draft_lp = (
                            float(
                                draft_log_probs[
                                    draft_step
                                ][draft_token].item()
                            )
                            if draft_log_probs is not None
                            else 0.0
                        )
                        accept = (
                            target_lp
                            > math.log(max(uniform, 1.0e-38))
                            + draft_lp
                        )
                    else:
                        accept = False

                    if not accept:
                        break
                    sampled[req_idx, draft_step] = draft_token

                accepted += 1

            if greedy_rejected:
                num_sampled[req_idx] = accepted + 1
                continue
            resample_step = accepted

        resample_logit_idx = start + resample_step
        is_bonus = resample_step == num_drafts
        if is_bonus:
            residual_logits = target_logits[
                resample_logit_idx, :vocab_size
            ]
        elif draft_log_probs is None:
            residual_logits = target_logits[
                resample_logit_idx, :vocab_size
            ].float().clone()
            rejected_draft_token = int(
                draft_sampled[resample_logit_idx + 1].item()
            )
            if 0 <= rejected_draft_token < vocab_size:
                residual_logits[rejected_draft_token] = float("-inf")
        else:
            target_lp = target_log_probs[resample_step]
            if (
                use_block_verification
                and cumulative_log_p is not None
                and resample_step > 0
            ):
                target_lp = (
                    target_lp + cumulative_log_p[resample_step - 1]
                )
            draft_lp = draft_log_probs[resample_step]
            ratio = torch.exp(draft_lp - target_lp)
            residual_logits = torch.where(
                ratio < 1.0,
                target_lp + torch.log1p(-ratio),
                torch.full_like(target_lp, float("-inf")),
            )

        sampled[req_idx, resample_step] = gumbel_block_argmax(
            residual_logits,
            temp,
            req_seed,
            pos[resample_logit_idx],
            use_fp64,
        )
        num_sampled[req_idx] = resample_step + 1

    return sampled, num_sampled

```

返回值：

- `sampled`: int64 `[R,S+1]`，典型 `[1..3,8]`。
- `num_sampled`: int32 `[R]`，典型 `[1..3]`。

**vllm-cl 接入位置**

- 当前 native reference 位于 `vllm_cl/worker/v2_ops.py:507`，并已是唯一
  `rejection_sample` 运行路径。
- 新设备算子落地后，该函数只保留一次
  `torch.vllm_v2_rejection_sample_(...)` 调用。
- `draft_sampled` 必须直接消费上游 `input_batch.input_ids` 派生的 int32 张量，
  不得在 Python 接入层转换为 int64；算子测试必须覆盖真实的 int32 输入。
- 新算子直接消费 device-side seed/position/temperature，不执行 `.item()`、
  `.cpu().tolist()` 或 Python request/token 循环。
- 不得调用 `torch.dlc_vllm_rejection_greedy_sample_`、
  `torch.sample_recovered_token` 或 `torch.rejection_random_sample_`。
- standard、synthetic、FP64 Gumbel、block verification 必须共享上述同一个 ABI；
  `use_block_verification=True` 与 synthetic rates 同时启用时按上游抛错。

### 3.9 `torch.vllm_v2_flatten_sampled_`

**PyTorch native**

```python
def vllm_v2_flatten_sampled_native(
    flat_sampled: torch.Tensor,  # int64 [L], in-place output, typical [1..24]
    sampled: torch.Tensor,       # int64 [R,S+1], typical [1..3,8]
    num_sampled: torch.Tensor,   # int32 [R], typical [1..3]
    cu_num_logits: torch.Tensor, # int32 [R+1], typical [2..4]
) -> None:
    for req_idx in range(num_sampled.shape[0]):
        start = int(cu_num_logits[req_idx].item())
        count = int(num_sampled[req_idx].item())
        flat_sampled[start:start + count].copy_(sampled[req_idx, :count])
```

**vllm-cl 接入位置**

- 替换 `vllm_cl/worker/v2_ops.py:826` `_FlattenSampledKernel`。
- 在 `install_v2_fallbacks()` 中继续赋给上游 rejection sampler 的
  `_flatten_sampled_kernel` wrapper。

### 3.10 `torch.vllm_v2_apply_logit_bias_`

该算子原地实现上游 logit-bias kernel 的三组独立语义：

- `allowed_token_ids` 非空时，只保留允许 token 的原始 logit，其余位置写为 `-inf`；
- `logit_bias` 非空时，对指定 token 的 logit 原地加 bias；
- 生成位置尚未达到 `min_lens` 时，将 stop token 的 logit 写为 `-inf`。

三组功能可同时启用，执行顺序必须与下面 reference 一致。

**PyTorch native**

```python
def vllm_v2_apply_logit_bias_native(
    logits: torch.Tensor,               # fp32 [L,V], in-place, typical [1..24,154880]
    expanded_idx_mapping: torch.Tensor, # int32 [L], typical [1..24]
    pos: torch.Tensor,                  # int64 [L], typical [1..24]
    num_allowed_token_ids: torch.Tensor,# int32 [Rmax], typical [3]
    allowed_token_ids: torch.Tensor,    # int32 [Rmax,1024], typical [3,1024]
    num_logit_bias: torch.Tensor,       # int32 [Rmax], typical [3]
    logit_bias_token_ids: torch.Tensor, # int32 [Rmax,1024], typical [3,1024]
    logit_bias: torch.Tensor,           # fp32 [Rmax,1024], typical [3,1024]
    min_lens: torch.Tensor,             # int32 [Rmax], typical [3]
    num_stop_token_ids: torch.Tensor,   # int32 [Rmax], typical [3]
    stop_token_ids: torch.Tensor,       # int32 [Rmax,128], typical [3,128]
) -> None:
    for token_idx, req_state_idx_t in enumerate(expanded_idx_mapping):
        req_state_idx = int(req_state_idx_t.item())
        row = logits[token_idx]

        num_allowed = int(num_allowed_token_ids[req_state_idx].item())
        if num_allowed:
            token_ids = allowed_token_ids[
                req_state_idx, :num_allowed
            ].long()
            saved_logits = row[token_ids].clone()
            row.fill_(float("-inf"))
            row[token_ids] = saved_logits

        num_bias = int(num_logit_bias[req_state_idx].item())
        if num_bias:
            token_ids = logit_bias_token_ids[
                req_state_idx, :num_bias
            ].long()
            row[token_ids] += logit_bias[req_state_idx, :num_bias]

        num_stop = int(num_stop_token_ids[req_state_idx].item())
        if (
            num_stop
            and int(pos[token_idx].item()) + 1
            < int(min_lens[req_state_idx].item())
        ):
            token_ids = stop_token_ids[req_state_idx, :num_stop].long()
            row[token_ids] = float("-inf")
```

**调用条件**

- 上游 `LogitBiasState.apply_logit_bias()` 保留现有 host guard；所有 active request 的
  `min_tokens`、`allowed_token_ids` 和 `logit_bias` 均未启用时，不调用该设备算子。
- `min_tokens > 0` 且请求存在 stop token 时，即使没有显式 `logit_bias`，也会调用该
  算子。当前固定输出 profile 的 `min_tokens=1024` 属于此场景。
- `expanded_idx_mapping` 决定每一行 logits 对应的 request-state 行；kernel 必须逐行读取
  对应的动态计数和 metadata，不能假设所有行属于同一请求。

**vllm-cl 接入位置**

- 当前 native fallback 位于 `vllm_cl/worker/v2_ops.py:168`，并在
  `install_v2_fallbacks()` 中替换上游 module-level `apply_logit_bias`。
- 新算子接入时保留该 native fallback，在函数开头优先调用一次
  `torch.vllm_v2_apply_logit_bias_(...)` 后返回，便于模型外逐元素对照。
- 算子原地修改 `logits`，不新增 output tensor，不允许 Python request/token 循环、
  `.item()`、D2H copy 或 synchronize。
- 测试必须分别覆盖 allowed-only、bias-only、min-length stop mask，以及三者同时启用；
  还需覆盖同一 graph replay 中不同 request 行使用不同动态计数。

### 3.11 `torch.vllm_v2_compute_slot_mappings_`

该算子只计算 autoregressive MTP 后续 draft step 使用的 slot mapping，不 gather 或改写
`input_block_tables`。不能通过调用 `torch.vllm_v2_prepare_block_table_` 实现，因为后者
还会清零并复制完整 block table，不属于当前 helper 的语义，并会在每个 MTP draft step
引入额外写流量。

单次调用处理一个 KV cache group。与 `prepare_block_table_` 一样，ABI 必须接收完整
二维 `slot_mappings` 和 `group_id`，由 kernel 定位输出行；禁止把带非零
`storage_offset` 的 `slot_mappings[group_id]` view 传给 DLC custom kernel。

**PyTorch native**

```python
def vllm_v2_compute_slot_mappings_native(
    slot_mappings: torch.Tensor,       # int64 [G,Tmax], in-place output, typical [1..2,4096]
    group_id: int,                     # host int, current KV cache group, typical 0..G-1
    idx_mapping: torch.Tensor,         # int32 [R], MTP typical [1..3]
    query_start_loc: torch.Tensor,     # int32 [R+1], MTP typical [2..4]
    positions: torch.Tensor,           # int64 [Tpad], decode typical [1..3]
    source_block_table: torch.Tensor,  # int32 [Rmax,Bg], MTP draft typical [3,32]
    block_size: int,                   # host int, typical 256
    cp_size: int,                      # host int, current group DCP size, typical 1 or 8
    cp_rank: int,                      # host int, 0 when cp_size=1
    cp_interleave: int,                # host int, vLLM DCP interleave, typical 1
    pad_slot_id: int = -1,             # host int
) -> None:
    assert 0 <= group_id < slot_mappings.shape[0]
    slot_mapping = slot_mappings[group_id]
    slot_mapping.fill_(pad_slot_id)

    for batch_idx, req_state_idx_t in enumerate(idx_mapping):
        req_state_idx = int(req_state_idx_t.item())
        start = int(query_start_loc[batch_idx].item())
        end = int(query_start_loc[batch_idx + 1].item())
        req_positions = positions[start:end]
        cp_block_size = block_size * cp_size
        block_indices = req_positions // cp_block_size
        block_offsets = req_positions % cp_block_size
        block_numbers = source_block_table[
            req_state_idx, block_indices.long()
        ]

        if cp_size == 1:
            slots = block_numbers * block_size + block_offsets
        else:
            is_local = (
                block_offsets // cp_interleave % cp_size == cp_rank
            )
            rounds = block_offsets // (cp_interleave * cp_size)
            remainder = block_offsets % cp_interleave
            local_offsets = rounds * cp_interleave + remainder
            slots = block_numbers * block_size + local_offsets
            slots = torch.where(
                is_local,
                slots,
                torch.full_like(slots, pad_slot_id),
            )
        slot_mapping[start:end].copy_(slots)
```

**vllm-cl 接入位置**

- 替换 `vllm_cl/worker/v2_ops.py:1611` 的
  `DLCBlockTables.compute_slot_mappings` Python 实现。
- wrapper 按 KV cache group 各调用一次，传完整 `self.slot_mappings`、当前
  `group_id`、`self.block_tables[group_id].gpu` 和当前 group 的
  `block_size/cp_size/cp_rank`。
- 每个 group 的 `cp_size` 必须取 `self.group_dcp_sizes[group_id]`；`cp_rank` 为
  `self.cp_rank if cp_size > 1 else 0`，以支持主模型与 draft model 混合 DCP。
- 所有 group 完成后，wrapper 仍返回
  `self.slot_mappings[:, :num_tokens_padded]`；`num_tokens_padded` 是 host-known 的
  Python view 长度，不需要成为 kernel 参数。
- kernel 必须先把当前 group 的整个 `[Tmax]` 输出行写为 `pad_slot_id`，再写 active
  token，避免 graph replay 读取到上一次调用残留的 slot。
- 当前 MTP profile 中该 method 在 15 轮 proposal 内调用 60 次，正好对应
  `15 * (M - 1)`，其中 `M=5`。每次调用都必须是单个 device kernel launch，不能有
  `.item()`、D2H copy 或 synchronize。

### 3.12 `torch.vllm_v2_prepare_autoregressive_prefill_inputs_`

这里的 `prefill` 指每轮 autoregressive proposal 的第一个 draft pass，并不只指请求的
初始 prompt prefill。该 helper 在每轮 MTP decode proposal 都会执行：根据本轮 rejected
token 数修正 query 长度，将 target 输入左移一位，补入最新 sampled token 或 chunked
prefill 的下一个 token，并准备 draft model 的 position、query-start、seq-len 和 padding
metadata。

公开算子 ABI 只接收 tensor 和 host scalar，不能把上游 `InputBuffers` 或 `InputBatch`
Python 对象传给 C++ binding。

**PyTorch native**

```python
def vllm_v2_prepare_autoregressive_prefill_inputs_native(
    last_token_indices: torch.Tensor,     # int64 [Rmax], in-place output, typical [3]
    current_draft_step: torch.Tensor,     # int64 scalar [], in-place output
    draft_input_ids: torch.Tensor,        # int32 [Tmax], in-place output, typical [4096]
    draft_positions: torch.Tensor,        # int64 [Tmax], in-place output, typical [4096]
    draft_query_start_loc: torch.Tensor,  # int32 [Rmax+1], in-place output, typical [4]
    draft_seq_lens: torch.Tensor,         # int32 [Rmax], in-place output, typical [3]
    target_input_ids: torch.Tensor,       # int32 [Tpad], decode typical [6..18], max [4096]
    target_positions: torch.Tensor,       # int64 [Tpad], decode typical [6..18], max [4096]
    idx_mapping: torch.Tensor,            # int32 [R], typical [1..3]
    target_query_start_loc: torch.Tensor, # int32 [R+1], typical [2..4]
    target_seq_lens: torch.Tensor,        # int32 [R], typical [1..3]
    num_sampled: torch.Tensor,            # int32 [R], typical [1..3]
    num_rejected: torch.Tensor,           # int32 [R], typical [1..3]
    last_sampled: torch.Tensor,           # int64 [Rmax,1], typical [3,1]
    next_prefill_tokens: torch.Tensor,    # int32 [Rmax], typical [3]
    max_num_reqs: int,                    # host int, Rmax, typical 3
) -> torch.Tensor:
    num_reqs = idx_mapping.shape[0]
    final_query_end = 0
    for req_idx in range(num_reqs):
        req_state_idx = int(idx_mapping[req_idx].item())
        query_start = int(target_query_start_loc[req_idx].item())
        query_end = int(target_query_start_loc[req_idx + 1].item())
        query_len = query_end - query_start - int(num_rejected[req_idx].item())
        final_query_end = query_end

        if query_len > 1:
            draft_input_ids[
                query_start : query_start + query_len - 1
            ].copy_(
                target_input_ids[
                    query_start + 1 : query_start + query_len
                ]
            )

        last_token_idx = query_start + query_len - 1
        last_token_indices[req_idx] = last_token_idx
        if int(num_sampled[req_idx].item()) > 0:
            next_token = last_sampled[req_state_idx, 0]
        else:
            next_token = next_prefill_tokens[req_state_idx]
        draft_input_ids[last_token_idx].copy_(next_token)
        draft_positions[
            query_start : query_start + query_len
        ].copy_(
            target_positions[query_start : query_start + query_len]
        )
        draft_query_start_loc[req_idx].copy_(
            target_query_start_loc[req_idx]
        )
        draft_seq_lens[req_idx].copy_(target_seq_lens[req_idx])

    current_draft_step.zero_()
    draft_query_start_loc[num_reqs : max_num_reqs + 1].fill_(
        final_query_end
    )
    draft_seq_lens[num_reqs:max_num_reqs].zero_()
    last_token_indices[num_reqs:max_num_reqs].zero_()
    return last_token_indices
```

**vllm-cl 接入位置**

- 替换 `vllm_cl/worker/v2_ops.py:935` 的
  `prepare_autoregressive_prefill_inputs` Python 实现，并保留该函数作为对象拆包
  wrapper。
- wrapper 将 `input_buffers` 拆成 `input_ids/positions/query_start_loc/seq_lens`，将
  `input_batch` 拆成 `input_ids/positions/idx_mapping/query_start_loc/seq_lens`，然后只
  launch 一次新算子。
- `last_token_indices`、`current_draft_step` 和四个 draft input buffer 都是持久化
  in-place output；kernel 不得分配或返回替代 buffer，以保持 graph 地址稳定。
- 当前 MTP profile 中该 helper 调用 15 次，总耗时约 27.86 ms；融合后不得保留
  request loop、`.item()`、D2H copy 或 synchronize。
- 测试必须覆盖纯 decode（`num_sampled > 0`）、chunked prefill
  （`num_sampled == 0`）、存在 rejected token、多 request 以及 `R < Rmax` 时三类
  metadata padding。

### 3.13 `torch.vllm_v2_prepare_autoregressive_decode_inputs_`

该算子准备 autoregressive MTP 每轮 proposal 中后续 draft decode step 的输入。它把每个
request 的第一个 draft token 写入 decode input，推进 position，根据 target sequence
length 和 rejected token 数更新 draft sequence length，并补齐 graph replay 使用的
`query_start_loc` 和 inactive request metadata。

上游调用点
`vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:263` 当前传入
`self.draft_tokens[:num_reqs, 0]`。这是带跨行 stride 的一维 column view；上游 Triton
kernel 通过单独的 `draft_tokens_stride` 支持该布局，但 DLC custom kernel 不应接收这种
输入。vllm-cl wrapper 必须先按下面的方式恢复原始连续二维 buffer，过程只创建 tensor
view，不做 device copy：

```python
num_reqs = draft_token_column.shape[0]
assert draft_token_column.ndim == 1
assert draft_token_column.storage_offset() == 0
draft_row_stride = draft_token_column.stride(0)
full_draft_tokens = draft_token_column.as_strided(
    (max_num_reqs, draft_row_stride),
    (draft_row_stride, 1),
)
```

在当前上游分配中，原始 `self.draft_tokens` 为连续的 `int64 [Rmax,M]`，因此
`draft_row_stride == M`，恢复后的 `full_draft_tokens` 也是连续的 `int64 [Rmax,M]`。
融合算子只读取第 0 列；传完整二维 tensor 是为了避免把 storage layout 语义隐藏在一维
view 和额外 stride 参数中。

**PyTorch native**

```python
def vllm_v2_prepare_autoregressive_decode_inputs_native(
    input_ids: torch.Tensor,        # int32 [Tmax], in-place output, typical [4096]
    positions: torch.Tensor,        # int64 [Tmax], in-place output, typical [4096]
    query_start_loc: torch.Tensor,  # int32 [Rmax+1], in-place output, typical [4]
    seq_lens: torch.Tensor,         # int32 [Rmax], in-place output, typical [3]
    draft_tokens: torch.Tensor,     # int64 [Rmax,M], contiguous input, typical [3,5]
    target_seq_lens: torch.Tensor,  # int32 [R], input, typical [1..3]
    num_rejected: torch.Tensor,     # int32 [R], input, typical [1..3]
    num_reqs: int,                  # host int, R, typical 1..3
    max_model_len: int,             # host int, typical 8192..65536
    advance_draft_positions: bool,  # host bool, GLM MTP is True
) -> None:
    input_ids[:num_reqs].copy_(draft_tokens[:num_reqs, 0])

    if advance_draft_positions:
        positions[:num_reqs].add_(1).clamp_(max=max_model_len - 1)
        seq_lens[:num_reqs].copy_(
            (
                target_seq_lens[:num_reqs]
                - num_rejected[:num_reqs]
                + 1
            ).clamp(max=max_model_len)
        )

    query_start_loc.copy_(
        torch.arange(
            query_start_loc.shape[0],
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        ).clamp_(max=num_reqs)
    )
    seq_lens[num_reqs:].zero_()
```

**vllm-cl 接入位置**

- `install_v2_fallbacks()` 已通过
  `autoregressive.prepare_decode_inputs = prepare_autoregressive_decode_inputs`
  patch 上游 module-level function，不需要修改 vllm 仓库。
- Python reference 直接保留在
  `vllm_cl/worker/v2_ops.py:prepare_autoregressive_decode_inputs` 的融合算子调用和
  `return` 后面，不额外定义 `_native` helper。wrapper 按上述方式从上游传入的
  column view 无拷贝恢复完整 `full_draft_tokens`，再调用一次新算子。
- 新算子参数依次传 `input_buffers` 的四个持久化 buffer、完整
  `full_draft_tokens`、`target_seq_lens`、`num_rejected` 以及三个 host scalar；不得接收
  `InputBuffers` Python 对象。
- 当前 MTP profile 中该 helper 调用 15 次，总耗时约 18.43 ms；融合后不得保留
  `torch.arange`、多个 pointwise kernel 或额外 tensor allocation。
- 测试必须覆盖 `R=1..Rmax`、存在 rejected token、position/sequence length 接近
  `max_model_len` 的 clamp，以及 `advance_draft_positions=False`。graph replay 必须验证
  active request 数变化后 `query_start_loc` 和 inactive `seq_lens` 不残留旧值。

## 4. P1 Prefill 算子

### 4.1 `torch.vllm_v2_prepare_prefill_inputs_`

**PyTorch native**

```python
def vllm_v2_prepare_prefill_inputs_native(
    input_ids: torch.Tensor,           # int32 [Tmax], in-place output, typical [4096]
    next_prefill_tokens: torch.Tensor, # int32 [Rmax], in-place output, typical [3]
    idx_mapping: torch.Tensor,         # int32 [R], typical [1..3]
    query_start_loc: torch.Tensor,     # int32 [R+1], typical [2..4]
    all_token_ids: torch.Tensor,       # int32 [Rmax,max_model_len], typical [3,8192]
    prefill_len: torch.Tensor,         # int32 [Rmax], typical [3]
    num_computed_tokens: torch.Tensor, # int32 [Rmax], typical [3]
) -> None:
    for batch_idx, req_state_idx_t in enumerate(idx_mapping):
        req_state_idx = int(req_state_idx_t.item())
        computed = int(num_computed_tokens[req_state_idx].item())
        prompt_len = int(prefill_len[req_state_idx].item())
        if computed >= prompt_len:
            continue
        query_start = int(query_start_loc[batch_idx].item())
        query_end = int(query_start_loc[batch_idx + 1].item())
        query_len = query_end - query_start
        input_ids[query_start:query_end].copy_(
            all_token_ids[req_state_idx, computed : computed + query_len]
        )
        next_pos = computed + query_len
        if next_pos < prompt_len:
            next_prefill_tokens[req_state_idx].copy_(
                all_token_ids[req_state_idx, next_pos]
            )
```

**vllm-cl 接入位置**

- 替换 `vllm_cl/worker/v2_ops.py:896`。
- 当前 profile 只在首次 prefill 调用一次，因此优先级低于 decode P0 算子。

## 5. P2 GLM Sparse Indexer Eager 算子

这两个算子不属于 DSpark 控制流，但当前 GLM target model 的 eager forward 会调用。Graph
模式下它们位于 model graph 内，因此排在 P2。

### 5.1 `torch.vllm_v2_pack_ragged`

该 native 与当前 `pack_seq_pytorch` 一致。它的动态 output allocation 是该 issue 需要由
算子组进一步评估的 graph 限制，不能在 native reference 中擅自改成另一套 ABI。

**PyTorch native**

```python
def vllm_v2_pack_ragged_native(
    packed: torch.Tensor,   # bf16 [N,Hidx,Didx], input, typical [1..24,32,128]
    lengths: torch.Tensor,  # int32 [B], input, typical [1..3]
    pad_value: float = -float("inf"), # host float
) -> torch.Tensor:
    batch_size = lengths.numel()
    max_len = int(lengths.max().item())
    output = torch.full(
        (batch_size, max_len) + packed.shape[1:],
        pad_value,
        dtype=packed.dtype,
        device=packed.device,
    )
    starts = torch.zeros(
        batch_size + 1,
        dtype=torch.long,
        device=lengths.device,
    )
    starts[1:] = lengths.cumsum(0)
    for batch_idx in range(lengths.shape[0]):
        length = int(lengths[batch_idx].item())
        start = int(starts[batch_idx].item())
        output[batch_idx, :length].copy_(packed[start:start + length])
    return output
```

**vllm-cl 接入位置**

- 替换 `vllm_cl/ops/sparse_attn_indexer.py:110` `pack_seq_pytorch`。
- 调用点为同文件约 `381` 行的 decode padding 路径。

### 5.2 `torch.vllm_v2_unpack_ragged`

**PyTorch native**

```python
def vllm_v2_unpack_ragged_native(
    padded: torch.Tensor,   # int32 [B,Nmax,Kidx], input, typical [3,8,2048]
    lengths: torch.Tensor,  # int32 [B], input, typical [1..3]
) -> torch.Tensor:
    batch_size = lengths.numel()
    total_tokens = int(lengths.sum().item())
    output = torch.empty(
        (total_tokens,) + padded.shape[2:],
        dtype=padded.dtype,
        device=padded.device,
    )
    starts = torch.zeros(
        batch_size + 1,
        dtype=torch.long,
        device=lengths.device,
    )
    starts[1:] = lengths.cumsum(0)
    for batch_idx in range(lengths.shape[0]):
        length = int(lengths[batch_idx].item())
        start = int(starts[batch_idx].item())
        output[start:start + length].copy_(padded[batch_idx, :length])
    return output
```

**vllm-cl 接入位置**

- 替换 `vllm_cl/ops/sparse_attn_indexer.py:150` `unpack_seq_pytorch`。
- 调用点为同文件约 `482` 行的 decode top-k unpack 路径。

## 6. 通用验收要求

每个 issue 都必须满足：

1. native reference 与 DLC kernel 输出逐元素一致。
2. eager 与 graph 均支持，输出使用 persistent/preallocated buffer。
3. graph replay 更换 request 顺序、ragged lengths、acceptance length 后读取新值。
4. kernel 内无 D2H copy、`.item()` 等价路径或 synchronize。
5. 对应 vllm-cl Python fallback 不再出现在 decode profiler 热路径。
6. GLM-5.2 + DSpark 回答保持正确，acceptance metrics 与当前正确基线一致。
7. issue 正文不存在未定义的 reference helper；按 2.1 附上所有依赖代码。
8. helper 只能作为同一公开算子内部的 device utility/inline 逻辑，不能增加公开
   `torch.*` ABI 或额外 kernel launch。
9. Gumbel/rejection 必须用固定 seed 和 position 验证 RNG 结果，并额外验证 graph replay
   更新 seed、position、temperature 后不会复用 capture 时的常量。
10. `gumbel_sample_` 除当前 greedy DSpark 基线外，还必须使用
    `draft_sample_method="probabilistic"` 验证 DSpark FULL graph 路径。
