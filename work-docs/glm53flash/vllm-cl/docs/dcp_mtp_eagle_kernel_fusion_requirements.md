# DCP + MTP EAGLE 元数据算子融合需求

## 当前临时实现

`vllm_cl/patch/patch_eagle.py` 在 vLLM EAGLE draft loop 外增加了两个
PyTorch correctness baseline：

- `_localize_eagle_step_slot_mapping_dcp_native()`：将 global position 转成当前
  CP rank 的 local slot，非 owner rank 写 `-1`；
- `_update_dcp_local_seq_lens_native()`：从更新后的 global `seq_lens` 计算当前
  DCP rank 的 local length。

默认 EAGLE 路径会先把 length metadata 复制到 drafter 的持久化 buffer。每个
draft step 更新 position 后修正 local slot，并在构建 draft attention metadata
之前刷新 local length。该顺序覆盖 rejection rollback、正常 position 增长和
constant-position draft 路径。

临时实现用于验证正确性，会产生额外 PyTorch kernel 和临时 tensor。正式实现
应把这些计算融合进 DLC custom op。

## Draft-step custom op 新接口

当前接口：

```python
torch.eagle_step_slot_mapping_metadata_(
    seq_lens,
    positions,
    block_table,
    out_clamped_positions,
    out_slot_mapping,
    block_size,
    max_model_len,
    batch_size,
    input_batch_size,
)
```

建议新增：

```python
dcp_local_seq_lens
total_cp_world_size
total_cp_rank
dcp_world_size
dcp_rank
cp_kv_cache_interleave_size
```

建议的完整语义为：

1. position 增长并执行现有 max-length 处理；
2. 原地更新 global `seq_lens`；
3. 计算 rank-local slot；
4. 从更新后的 global length 计算并写入 `dcp_local_seq_lens`。

非 DCP 应保留原接口或使用独立 specialization，不能增加额外 kernel launch、
除法或取模开销。

## Local slot 公式

设：

```text
p = clamped_position
B = block_size
W = total_cp_world_size
R = total_cp_rank
I = cp_kv_cache_interleave_size
```

计算：

```text
virtual_block_size = B * W
block_index = p // virtual_block_size
virtual_offset = p % virtual_block_size
owner_rank = (virtual_offset // I) % W
local_offset = (virtual_offset // (W * I)) * I + virtual_offset % I
slot = block_table[req_idx, block_index] * B + local_offset
```

只有 `owner_rank == R` 且 position 有效时写入 `slot`，其他 rank 以及 CUDA
graph padding row 写 `-1`。

## Local length 公式

设更新后的 global length 为 `L`：

```text
N = dcp_world_size
r = dcp_rank
I = cp_kv_cache_interleave_size
rounds = L // (N * I)
remainder = L % (N * I)
rank_remainder = clamp(remainder - r * I, 0, I)
local_length = rounds * I + rank_remainder
```

draft-step custom op 应在写回 global `seq_lens` 的同一个 thread 中写回 local
length，不增加第二次 launch。

## Rejection rollback

step 0 forward 后，vLLM 当前执行：

```python
seq_lens -= num_rejected_tokens_gpu
```

正式版本需要让 rollback 同时更新 local length。建议提供原地融合接口：

```python
eagle_rejection_rollback_metadata_(
    seq_lens,
    dcp_local_seq_lens,
    num_rejected_tokens,
    num_reqs,
    dcp_world_size,
    dcp_rank,
    cp_kv_cache_interleave_size,
)
```

该算子必须在 step 0 forward 后执行，不能提前融合进
`eagle_prepare_inputs_padded_kernel`，否则会改变 step 0 的 padded drafter 语义。

## 约束和验收

- 所有输出使用调用方提供的持久化 buffer，支持 CUDA graph capture；
- 不申请动态 tensor，不执行 host sync、D2H copy 或 collective；
- slot ownership 使用 total CP，local length 使用 DCP；
- `seq_lens` 和 `dcp_local_seq_lens` 不允许 alias；
- 覆盖 DCP size 2/4/8、所有 rank、interleave 1 和大于 1；
- 覆盖 virtual block 边界、max model length 和 graph padding；
- 覆盖零 rejection、部分 rejection、全部 rejection和多 request；
- 连续执行相同请求时，输出不受上一次请求的 KV 内容影响；
- DCP draft step 不新增 launch，rollback 最多维持当前的一次 length-update
  launch。
