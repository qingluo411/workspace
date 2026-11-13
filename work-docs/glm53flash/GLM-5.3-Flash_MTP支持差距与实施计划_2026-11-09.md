# GLM-5.3-Flash MTP 支持差距与实施计划

日期：2026-11-09  
目标：评估在 `release-glm53flash` 上启用 GLM-5.3-Flash MTP speculative decoding
所需工作，并给出可执行的实施顺序与验收口径。

## 1. 结论

**2026-11-13 状态**：MTP eager 与 `FULL_DECODE_ONLY` graph 均已打通；单请求
greedy、in-flight 1/8/16/32/64 并发和 17K/34K/64.6K 长上下文语义验证通过。
当前主要剩余 AIME26、正式 acceptance/性能验收和 `num_speculative_tokens>1`。
详细结果见第 13 节。

最初评估时发布路径没有启用 MTP，但并不是“从零实现”。上游 vLLM 已有
`Glm5NextMTPModel`、MTP 配置探测、Eagle/MTP proposer、rejection sampler 和
decode graph 基础设施；KDA/KPool 的底层 DLC 算子也有相当一部分已经覆盖
spec 语义。真正需要补的是 **GLM-5.3 特有的模型/缓存事务接线**，集中在：

1. 文本-only GLM5 MTP 的加载兼容性；
2. MTP 层 `fused_eh_norm` 的 DLC fallback；
3. KDA spec verify 的插件包装层；
4. KPool 在 speculative verify 后的 request 级回滚/重放；
5. MTP 层自己的 KPool cache 与 `index_share_for_mtp_iteration`；
6. spec-aware graph、长上下文、并发和验收。

其中 **KPool 事务回滚是最大风险项**；KDA 本身主要不是算子问题，而是当前
`vllm-cl` 适配层显式拦截了 spec 参数。

## 2. 当前仓库状态

本地已切换到可用的 GLM release 线：

| 仓库 | 当前分支 | 说明 |
|---|---|---|
| `DLC_Custom_Kernel` | `release-glm53flash` | 本地分支存在，但跟踪关系仍指向 `origin/main` |
| `pytorch` | `release-glm53flash` | 同上 |
| `vllm` | `release-glm53flash` | 跟踪 `origin/release-glm53flash` |
| `vllm-cl` | `release-glm53flash` | 比远端多一个本地提交 `b3cb855` |
| `LLVM` / `CLsim` / `DLCSynapse` / `DLC_CL` | `main` | 远端没有 `release-glm53flash` 分支 |

模拟器已切换到 `CLsim`；`DLCsim` 仅保留仓库和旧安装目录用于回滚，不参与当前 MTP 工作。

## 3. 已有基础

### 3.1 模型与配置

- checkpoint 声明 `num_hidden_layers=45`、`num_nextn_predict_layers=1`。
- 第 45 层是 MTP 层，共 889 个权重 key、约 13.84 GiB，目标模型量化配置通过
  `.layers.45.` 排除项将其保持为 BF16。
- 第 45 层不是 KDA 层，也不是 mHC 层；它是 MLA/KPool indexer + 288-expert MoE +
  `eh_proj`/`enorm`/`hnorm`/`shared_head`。
- MTP 层 checkpoint 明确包含 `self_attn.indexer.*` 权重，因此不能把 MTP 层简化成
  纯 dense MLA；否则会丢 KPool 选池语义。
- vLLM 已有：
  - `Glm5NextMTPModel` 注册：`vllm/vllm/model_executor/models/registry.py:692`
  - `glm5_next -> glm5_next_mtp` 配置探测：
    `vllm/vllm/config/speculative.py:1058`
  - `Glm5NextMultiTokenPredictor`、`Glm5NextMTP`：
    `vllm/vllm/models/glm5next/common/mtp.py`

### 3.2 通用 MTP 运行时

- V1 runner 已有 MTP proposer、投机 token D2H、rejection correction、异步 spec
  scheduling 和 MTP outer graph。
- `vllm-cl` 已有 `patch_eagle.py`、`patch_rejection_sampler.py`、
  `patch_speculative_config.py`、`patch_mtp_input_ids.py` 等 DLC 适配。
- `work-docs/glm53flash/vllm-cl/docs/mtp_dlc_graph.md` 已经记录了 MTP
  `FULL_DECODE_ONLY` proposer graph 的实现与验证边界。
- `patch_mla_indexer.py` 已把 DLC 的 native next-n 支持扩展到 `1, 2, 4, 6, 8`。

### 3.3 KDA 算子层已有 spec 语义

底层算子已经具备，并且已有冻结用例：

- `torch.dlc_kda_causal_conv_update` 支持多 token、`query_start_loc`、
  `num_accepted_tokens` 和 `state_len = 3 + num_spec`。
- `torch.dlc_kda_recurrent_decode` 支持 `state_indices [B, next_n]` 和
  `num_accepted_tokens` 快照提交。
- `vllm-cl/tests/test_glm5_kda_conv_update_op.py` 已有 accepted=0/部分/全部接受用例。
- `vllm-cl/tests/test_glm5_kda_recurrent_decode_op.py` 已有 accepted 快照用例。

结论：KDA 第一阶段不需要新写设备 kernel，重点是解除插件包装层拦截并接入正确
的布局/参数。

## 4. 已实测阻断

使用最小 eager 服务启动 MTP：

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

配置探测成功，进程已经加载 target 和 MTP 权重，但在 draft proposer 的
`load_model()` 中退出：

```text
AttributeError: 'Glm5NextConfig' object has no attribute 'image_token_index'.
Did you mean: 'image_token_id'?
```

根因在上游的通用 MTP 多模态兼容分支：

- `vllm/vllm/v1/spec_decode/llm_base_proposer.py:1352` 对
  `supports_multimodal(target_model)` 进入多模态处理；
- GLM5 的配置字段名是 `image_token_id`，但没有被列入该函数前面的兼容名单，
  最终走到 `image_token_index` fallback。

这是一个小的加载兼容修复。修复后，MTP 模型已经完成 23.36 GiB/rank 的加载、
lm_head/shared_head 共享和 worker 初始化，但继续在 KV cache 初始化处遇到第二个阻断：

```text
AssertionError: All drafting layers should belong to the same kv cache group
```

对应 `vllm/vllm/v1/spec_decode/llm_base_proposer.py:1687`。GLM5 MTP 层同时拥有
MLA KV、KPool pool cache 和 KPool tail cache 三类 cache 语义，不能沿用上游“所有
draft 层共用一个 KV cache group/AttentionMetadata”的假设。该问题不是算子 kernel
问题，但会阻止 MTP 请求进入 forward，需要把 draft proposer 扩展为多 KV group /
多 metadata。

日志：

- 首个加载阻断：`/root/workspace/log/mtp_smoke_20261109.log`
- 修复加载后的 KV group 阻断：`/root/workspace/log/mtp_smoke_20261109_2.log`

## 5. 工作拆分

| 优先级 | 工作项 | 预估 | 主要落点 |
|---|---|---:|---|
| P0 | GLM5 文本-only MTP 加载兼容 | S | `vllm_cl` upstream compat patch |
| P0 | MTP `fused_eh_norm` DLC fallback | S/M | `common/mtp.py` 的全局函数重绑定 |
| P0 | KDA conv/recurrent spec 包装层 | M | `vllm_cl/patch/patch_kda.py` |
| P0 | KPool verify 事务与 accepted 重放 | L | `vllm_cl` runner + KPool transaction |
| P0 | MTP 层 KPool/index-share 正确性 | L | MTP layer、KPool cache、draft runner |
| P1 | spec-aware graph 与 FULL decode | M | `dlc_model_runner.py` / graph patch |
| P1 | 权重、容量、长上下文和并发验证 | M | tests + benchmark |
| P2 | 性能调优（acceptance、num_spec、KPool） | L | 后续优化阶段 |

## 6. 详细实施项

### 6.1 GLM5 文本-only MTP 加载兼容（已实现，已验证）

需要把 `Glm5NextForConditionalGeneration` 纳入 `image_token_id` 兼容分支，或在
`vllm-cl` 的 upstream compatibility patch 中为 GLM5 target 跳过
`image_token_index` fallback。

验收：

- `--language-model-only` + `--speculative-config method=mtp` 能完成 target/draft
  模型构建；
- 不改变非 MTP 文本服务路径；
- 不把视觉 token 字段硬编码进 MTP draft 模型。

实现：`vllm_cl/patch/models/patch_glm5_mtp.py` 在 proposer `load_model()` 前为缺失
`image_token_index` 的配置补齐 `image_token_id` 别名。

### 6.2 MTP `fused_eh_norm` DLC fallback（已实现，已验证）

`Glm5NextMultiTokenPredictorLayer.forward()` 直接调用
`vllm.models.glm5next.nvidia.ops.fused_eh_norm`，该实现是 Triton kernel。
当前 DLC 明确没有可用 Triton driver，因此第一个 MTP forward 会在该处失败。

第一版应提供等价 PyTorch fallback：

```text
mask = positions != 0
embed = inputs_embeds * mask
eh_input = cat(enorm(embed), hnorm(previous_hidden))
```

之后可再评估用 DLC fused norm/cat 或单独 kernel 优化。验收要求：

- position 0 的 embed 必须清零；
- `enorm` / `hnorm` 的 eps、dtype、FP32 累加语义与参考一致；
- 输出 `[N, 2H]` 连续，可直接喂给 `eh_proj`；
- eager 和 graph 捕获结果 bit-exact/容差一致。

当前实现：`patch_glm5_mtp.fused_eh_norm_dlc`，已覆盖 position 0 mask、CPU reference
和真实 DLC 设备执行。当前版本是功能 fallback，尚未做融合性能优化。

### 6.3 KDA spec verify（wrapper 已实现，已验证）

`vllm-cl/patch/patch_kda.py` 当前有三处显式拦截：

- conv update：有 `num_accepted_tokens` 或 `query_start_loc` 就抛
  `NotImplementedError`；
- recurrent decode：有 `num_accepted_tokens` 就抛；
- recurrent decode：`ssm_state_indices.dim() != 1` 就抛。

需要改为真实接入：

1. conv spec 路径按上游语义接收 flattened `[num_tokens, D]` 与
   `query_start_loc`，不能错误地把它当成 `[B, D, 1]`。
2. `num_accepted_tokens` 透传给 `dlc_kda_causal_conv_update`。
3. recurrent decode 允许 `[B, next_n]` state indices，保留第 0 列选页语义。
4. recurrent decode 透传 `num_accepted_tokens`，确保 state 写回的是 accepted 快照。
5. 保留 plain decode 的 packed 1-D fast path；spec path 不能复用被
   `_pack_kda_state_indices` 无条件压缩的索引。
6. 增加 wrapper 级测试，而不只是底层算子测试。

验收用例至少覆盖：

- `num_spec=1`：全接受、拒绝、accepted=0；
- `num_spec=3`：部分接受；
- 单请求和多请求 mixed spec/non-spec；
- `state_len=3+num_spec`；
- KDA state page 的 null/padding slot；
- 与上游 Triton/GDN reference 的 out 和 state 对齐。

当前结果：`patch_kda.py` 已允许 spec conv/recurrent 参数，conv 支持 flattened
`[tokens,D] + query_start_loc + num_accepted_tokens`，recurrent 支持
`[B,next_n]` state indices 和 accepted 快照。新增 wrapper 用例和 graph replay 用例
均已通过。

### 6.4 KPool speculative transaction（已接入，端到端仍有残余漂移）

> 2026-11-10 更新：transaction 已接入 runner，并修复了 prefill 提交范围和
> pool-row 快照语义；但 MTP greedy 端到端仍存在间歇性 target 错 token。最新排查见
> `GLM-5.3-Flash_MTP端到端排查记录_2026-11-10.md`。

这是 MTP 的核心正确性风险。当前 target verify 的 KPool 路径在一次 forward 中：

1. 对候选 token 执行 `#19` slot mapping；
2. 执行 `#07 append_compress`，写 raw tail 和已完成的 pool cache；
3. 执行 `#09 select_topk`；
4. 之后 rejection sampler 才知道哪些 candidate 被接受。

`#07` 在采样前已把全部候选 token 写入 cache；KDA 有 accepted 快照，但 KPool
没有等价的提交/回滚机制。旧 native 路径曾有 `vllm_cl/kpool_speculative.py`
journal，但在 `560a93f` 随 native fallback 一并删除。当前 fused 路径没有可用的
request 级回滚。

必须补齐至少一个方案：

- **方案 A（优先）**：verify 前对每个请求的 raw-tail 行和可能跨过的 pool page
  做小范围快照；采样后恢复，再按 accepted prefix 重放 `#07`；
- **方案 B**：把 `#07` 拆成 append/provisional 和 commit 两阶段，accepted 后只
  commit 被接受的前缀；
- **方案 C**：为 KPool 写 request-granular journal，记录 touched tail rows/
  pool pages，支持部分接受回滚。

需要特别验证：

- 一次 verify 跨 pool boundary；
- accepted=0、部分接受、全部接受；
- 同一 step 多请求不同 accepted 数；
- eager 与 graph replay；
- 回滚后的下一次 decode 与不开 MTP 的 greedy 输出完全一致。

当前结果：已用真实 `#07` 复现“拒绝候选污染 pool/tail cache”，并验证
“snapshot -> restore -> accepted-prefix replay”与纯 committed 路径逐字节一致。
这说明第一版不需要改 `#07` ABI；剩余工作是把这个事务接入 V1 runner，并做
request/page 粒度快照和性能评估。

### 6.5 MTP 层自己的 KPool / index-share（算子路径已验证）

第 45 层 checkpoint 有完整的 `self_attn.indexer.*` 权重，MTP layer 实际会创建
KPool indexer。当前 `patch_glm5next.py` 的注释把 MTP 描述成 dense MLA，和
checkpoint 不一致；当前代码只是因为 `DLCGlm5NextMLAAttention` 默认
`is_mtp_layer=False` 才实际创建了 indexer，属于“隐式正确”，需要补显式测试，
防止后续重构把 indexer 删掉。

`index_share_for_mtp_iteration=true` 的流程是：

- step 0：MTP layer 计算自己的 KPool top-k；
- step 1+：跳过 indexer，复用 step 0 的 top-k；
- `compact_topk_indices` 把每个请求最后一个 query row 的选择结果压到 buffer 前部。

需要验证：

- MTP KPool buffer 与 target buffer 的所有权/共享关系；
- `set_skip_topk` 对 DLC MLA wrapper 是否生效；
- `compact_topk_indices` 在 DLC int32 buffer 上的高级索引行为；
- MTP draft cache 的 KPool tail/pool 在拒绝后是否需要同样的 journal；
- `num_spec=1/2/4` 时 `next_n` metadata、block table 和 query_start_loc 一致。

当前结果：

- `#09` 在 `next_n=4/8` 的小形状测试与稳定排序 reference 逐位一致；
- `#19` 在 4-token spec window 下给出正确的 pool/tail slot；
- `compact_topk_indices` 在 DLC top-k buffer 上结果正确；
- 2-D state indices 与 accepted 快照已由 KDA 测试覆盖。

生产几何补充：

- 使用真实 16K capture 的 `pool_cache=[1569,64,128]` 和 block table；
- `next_n=4` 平均 `3.84 ms`，`next_n=8` 平均 `7.58 ms`；
- 批量 next-n 输出与逐行单 query kernel 完全一致；
- 与 CPU FP32 reference 相比，只有 2 个 pool 的排列顺序不同（共 8 个 int 元素），
  selected pool 集合完全一致，属于近并列分数的归约顺序差异，不是 next-n 批处理错误；
- `#19` spec window 为 `0.22 ms`，pool/tail slot 均精确匹配；
- `compact_topk_indices` 在生产宽度 2176 下为 `0.15 ms`，结果逐位一致。

仍需在模型级验证 MTP draft cache 的拒绝回滚和 graph 地址稳定性。

### 6.6 Runner 与 graph（算子级已验，模型级待解 KV group）

通用 MTP runner 和 proposer graph 已经存在，但 GLM5 需要确认：

- V1 DLC runner 的 target verify metadata 能覆盖 spec batch；
- KDA `num_accepted_tokens` 和 state indices 在 full graph 中是持久地址；
- KPool journal 的 snapshot/restore/replay 不能进入被捕获的 graph，必须在 graph
  replay 前后由 runner 执行；
- MTP outer graph 对 GLM5 的 KPool buffer、draft metadata、slot mapping 地址稳定；
- graph capture 尺寸覆盖 batch 与 `num_spec` 组合；
- graph replay 与 eager 的 token 输出、acceptance count、KDA state 对齐。

建议顺序是先 `--enforce-eager` 打通功能，再打开
`FULL_DECODE_ONLY` 和 `DLC_MTP_PROPOSER_CUDAGRAPH`。

算子级已经验证 KDA spec 和 KPool `next_n` 可 graph capture/replay；当前模型级首先
需要解决 draft 层多 KV cache group，之后才能进入完整 MTP graph。

### 6.7 权重与容量

MTP 会增加：

- 第 45 层 BF16 权重约 13.84 GiB，TP8 下约 1.73 GiB/rank；
- 一个 MTP draft cache group（KPool pool + tail）；
- KDA conv state 宽度从 3 增加到 `3+num_spec`，每层每 page 的 conv state 增加
  `num_spec × D_local × 2 bytes`。

必须重新核算：

- `gpu_memory_utilization` 下的 KV/state block 数；
- 1M 声明上限对应的实际可用 context；
- 64 并发下的 graph bucket 和 state page 容量；
- MTP 开启前后的最大并发/上下文回退比例。

## 7. 建议里程碑

### M0：能加载，不接请求

- 修 `image_token_index`；
- MTP 权重加载审计；
- 启动日志出现 `Glm5NextMTPModel` 的完成构建。

### M1：eager、num_spec=1、单请求正确

- DLC `fused_eh_norm` fallback；
- KDA wrapper spec 接入；
- KPool target/draft transaction；
- greedy 下 MTP 输出与 baseline token-for-token 一致。

### M2：多请求与 spec depth

- `num_spec=1/2/3/4`；
- mixed spec/non-spec；
- accepted=0/部分/全部；
- KPool cross-boundary rollback；
- 采样指标：draft acceptance、accepted length、TPOT。

### M3：graph 与生产参数

- `FULL_DECODE_ONLY` + MTP proposer graph；
- 1/8/16/32/64 并发；
- 4K/16K/32K 长上下文；
- AIME26 和固定 greedy hash 回归；
- 与无 MTP 服务的 TTFT/TPOT/吞吐对比。

## 8. 验收矩阵

| 类别 | 必须通过 |
|---|---|
| 加载 | 文本-only target + MTP draft，无 missing/unexpected key |
| 数值 | greedy 输出与无 MTP baseline 完全一致 |
| KDA | accepted=0/部分/全部；out/state 与 reference 对齐 |
| KPool | verify 跨 pool boundary 后回滚并重放 accepted prefix |
| 请求 | 单请求、多请求、mixed spec/non-spec |
| Graph | eager vs FULL_DECODE_ONLY 输出一致 |
| 深度 | num_spec=1/2/3/4 |
| 上下文 | 4K/16K/32K，至少一次 128K 容量验证 |
| 并发 | 1/8/16/32/64 |
| 性能 | 明确 acceptance rate、TPOT、TTFT、吞吐和显存占用 |
| 回归 | AIME26、NaN 检查、prefix cache、KPool top-k 稳定性 |

## 9. 风险与待确认项

1. **KPool 事务是正确性阻塞项**：不补回滚，长上下文跨 pool boundary 的 MTP
   结果不可信。
2. **MTP draft cache 是否也需要回滚**尚未实测；不能只处理 target KPool。
3. **`index_share_for_mtp_iteration` 与 DLC topk buffer 的地址稳定性**需要 graph
   级验证。
4. **MTP layer BF16 MoE 性能**可能成为 proposer 的主要成本，需要在功能正确后
   单独 profile。
5. **当前 MTP 首次失败只验证到加载兼容层**；后续还需要逐层打开功能，不能把
   “服务能启动”当作“MTP 已支持”。

## 10. 建议下一步

先完成 M0 到 M1 的最小闭环，按以下顺序提交：

1. GLM5 `image_token_index` 兼容 patch；
2. `fused_eh_norm` DLC fallback + 单元测试；
3. KDA spec wrapper + 真实设备回归；
4. KPool target transaction + greedy baseline 对比；
5. MTP draft KPool/index-share 验证；
6. eager 单请求通过后，再进入 graph 和性能阶段。

## 11. 本轮验证结果（2026-11-09）

真实 DLC 设备上的结果：

| 验证项 | 结果 |
|---|---|
| KDA conv/recurrent 全量回归 + 新增 spec wrapper/graph 用例 | `37 passed` |
| MTP `fused_eh_norm` CPU + DLC | `2 passed` |
| KPool #09 next_n=4/8、#19 spec window、compact top-k | `4 passed` |
| rejection sampler + 288-expert BF16 MoE（T=4） | `2 passed` |
| KPool rejected-token 污染与 restore/accepted replay | `1 passed` |
| 已有 MLA indexer MTP buffer / hybrid MTP config | `8 passed` |

补充生产相关验证：

- 真实 MTP MoE 几何 `T=4, H=4096, I=2048, E=288, topk=8` 已跑通，输出
  `[4,4096]` BF16；与常数权重解析参考的最大绝对误差 `0.125`，相对峰值
  `0.37%`，属于 BF16 舍入量级；
- 11 层 KPool 事务原型（每层 1 个 touched pool page + 1 个 tail page，
  snapshot + restore + accepted-prefix replay）中位约 `2.50 ms`。该数字包含
  临时输入分配和 22 次层内调用，后续可改为持久 buffer/批量 copy 优化。

生产事务复测（每层 2 个可能跨过的 pool page + 1 个 tail page）：

- 11 层总耗时中位 `2.78 ms`；
- layer 0 的两个 touched pool page 和 tail page 均与直接 committed 路径逐字节一致。

最终生产形状记录（`work-tests/mtp/results/operator_level.json`）：

| 项目 | 结果 |
|---|---|
| KPool #09 next_n=4 | `3.84 ms`，rowwise exact，selected set equal |
| KPool #09 next_n=8 | `7.58 ms`，rowwise exact，selected set equal |
| KPool #19 spec window | `0.22 ms`，pool/tail exact |
| compact top-k width=2176 | `0.15 ms`，exact |
| 11-layer KPool transaction | `2.78 ms`，2 pool + 1 tail page/layer，state exact |
| BF16 MoE `T=4/H=4096/I=2048/E=288/topk=8` | `4.59 ms`，max abs `0.015625`，相对峰值 `0.48%`，allocated `13.50 GiB` |
| `fused_eh_norm` fallback `T=4/H=4096` | `0.58 ms`，position 0 masked |

结论：

- KDA spec 算子契约已经可用，不需要先写新 kernel；
- KPool `#09/#19`、compact top-k、rejection sampler、BF16 MoE 的算子路径可用；
- KPool transaction 已接入 runner，并修复 prefill 提交范围与 pool-row 快照；
  端到端仍存在间歇性 target 错 token，详见 2026-11-10 排查记录；
- `fused_eh_norm` 功能 fallback 可用，性能融合尚未做；
- draft 多 KV group 已完成接入，不再是阻断项；
- 生产几何 `H=4096 / I=2048 / 288 experts` 的 MoE 仍需在模型级做性能和峰值显存验证，
  当前小形状已证明 288-expert dispatch 路径正确。

## 12. 2026-11-12 状态更新

本轮已完成 MTP eager 单请求 greedy 精度闭环：固定 8-token prompt、32-token
输出重复运行 **50/50 与无 MTP baseline 逐 token 一致**；无 MTP
`FULL_DECODE_ONLY` graph 回归 **20/20**。KPool transaction、KDA spec 和高 state
slot 相关定向测试共 **77 passed**（41 + 36）。

根因最终收敛到 **`g_a` strided slice 的模型侧 DLC 物化/staging 路径**：`g_a`
是融合投影的 `[T,128]` 末分片，交给 `g_b_proj` 前由 launcher 隐式复制时，失败
样本可反解出前 32 个 BF16 元素仍是旧值，导致 GEMM row 0 的全部输出列受影响。
`projected`、`g_a`、KDA core 和权重均已排除。精确 shape 的独立
`custom_slice_16bit` syntest 与 2000 次连续下发 stress 均通过；但
`work-tests/mtp/run_slice_staging_compare.py` 的两阶段耦合测例（phase A 用
`index_select` 保存正确结果，phase B 用 split view 触发 staging）已经稳定复现：
T=2 原始 view 约 211/1000 失败，显式 `.contiguous()` 约 282/1000 失败，而
`index_select` 0/1000 失败；T=1/8 未复现，T=3..7 也可触发。显式 contiguous
的 `g_a_input` 仍有 32 个错误元素，且输出预清零不能修复，说明问题在 slice/copy
物化本身，而不是 GEMM 输出 buffer。首个失败样本的错误 32 个 BF16 元素与
producer 输入 `hidden[0:32]` 逐字节相同，说明 staging 首段很可能残留了前序
in_proj 使用过的 VMEM 数据。

已在 `DLC_Custom_Kernel/dlc_kernels/copy/slice_16bit.cpp` 的 branch 5 将
`bf16_offmemcpy(CMEM)` 改为独立的 `bf16_v2v_withoff(VMEM) -> vmem_to_hbm` 路径；
修复库已安装并验证：耦合测例 T=2 的 view/contiguous 1000 次、T=1..8 shape 矩阵、
`slice_bf16` 43 项、`slice_mtp_stress` 2000 次均通过，MTP eager 为 50/50。

模型侧随后关闭了 `index_select`，让 KDA 直接使用旧 `g_a` split view；修复后的
kernel 在真实 MTP 模型上再次通过 **50/50**，因此已移除该 workaround，生产路径
直接走修复后的 `custom_slice_16bit`。详细证据、日志与验证见：

`GLM-5.3-Flash_MTP端到端排查记录_2026-11-10.md` 第 10.6 节。

结论：MTP eager 单请求精度阻断已关闭；`custom_slice_16bit` staging 根修已完成，
并通过耦合/单元回归及真实 split-view MTP 50/50。MTP + `FULL_DECODE_ONLY` graph
已完成单请求、in-flight 1/8/16/32/64 并发和 17K/34K/64.6K 长上下文语义验证。
发布前仍需完成 AIME26、正式 acceptance/性能验收和更深 spec depth 验证。

2026-11-13 又在最终修复库上重跑无 MTP `FULL_DECODE_ONLY` graph 固定请求，
结果为 **20/20**，日志：
`/root/workspace/log/baseline_slicefix_graph2_20261113.log`。该轮使用
`GPU_MEMORY_UTILIZATION=0.80`、`max-num-batched-tokens=1024`、
`max-num-seqs=1`，未启用 MTP；测试后服务已停止。

## 13. 2026-11-13 MTP graph / 并发 / 长上下文验证

### 13.1 本轮修复

| 文件 | 修复 |
|---|---|
| `vllm-cl/vllm_cl/worker/dlc_model_runner.py` | `_dummy_run()` 为 MTP verify warmup 填充 dummy `num_decode_draft_tokens` / `num_accepted_tokens`，避免被 GDN 归类为 prefill |
| `vllm-cl/vllm_cl/patch/patch_kda.py` | `gather_initial_states_dlc()` / `scatter_states_dlc()` 将非连续或非 int32 indices 归一化为同设备连续 int32 |
| `vllm/vllm/v1/attention/backends/gdn_attn.py` | spec graph 的 `spec_query_start_loc` 尾部覆盖 padded token；修复 batch 缩小时 KDA `synErrorLaunchFailure` |
| `vllm/tests/v1/attention/test_gdn_metadata_builder.py` | 新增 `test_full_cudagraph_spec_metadata_covers_padded_tokens` |

### 13.2 验证结果

- MTP + `FULL_DECODE_ONLY` graph 单请求固定 greedy：**20/20**。
- 并发语义（每请求唯一 marker）：in-flight 1/8/16/32/64 两轮全部通过。
- 长上下文 marker 检索：17,025 / 34,025 / 64,625 token 全部命中，MTP graph 与
  MTP eager 耗时基本一致。
- 4K 配置实际同时 `Running` 峰值约 20（KDA state/KV 容量限制）；32/64 档为
  in-flight/排队执行，不等价于 32/64 同时 decode。proposer graph 已确认在
  reqs=1/4/8/12/16/20 上 replay。
- 随机 token 长 prompt 的 greedy 输出在 no-MTP baseline 自身也会跨服务分叉，
  因此长上下文改用 needle marker 语义验收，不采用逐 token 全等。

详细数据、日志和边界说明见
`GLM-5.3-Flash_MTP端到端排查记录_2026-11-10.md` 第 10.7 节。

### 13.3 尚未完成

- AIME26 端到端精度；
- 正式 acceptance rate、TTFT/TPOT、吞吐和显存峰值对比；
- `num_speculative_tokens>1`；
- 若要求真正 32/64 同时 decode，需要先降低每请求 KDA state/KV 占用。
