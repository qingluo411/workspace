# GLM-5.3-Flash 在 vLLM 中的 MTP 机制与代码现状

日期：2026-11-18  
适用模型：GLM-5.3-Flash-Mix  
代码基线：`vllm` / `vllm-cl` `release-glm53flash`；`DLC_Custom_Kernel`
`release-glm53flash-3`；`pytorch` `main`。当前生产路径为 **V1 Model Runner**，
`num_speculative_tokens=1` 已验证。

---

## 1. 结论摘要

GLM-5.3-Flash 的 MTP 不是独立的 draft 模型，而是 **checkpoint 自带的第 45 层
Multi-Token Predictor**。它复用 target 模型同一个 checkpoint，通过 vLLM 的
EAGLE/MTP proposer 在每次 target decode 后生成 1 个 draft token，再由 target
模型在下一步做 `1 + num_spec` 行 verify。当前链路已经打通：

- MTP eager：固定 128-token prompt，greedy 50/50 通过；
- MTP + `FULL_DECODE_ONLY` graph：固定 128-token prompt，20/20 通过；
- 长上下文 marker：17,025 / 34,025 / 64,625 token 三档全部命中；
- `fused_eh_norm_bf16` 已接入 DLC 原生 op，单算子验收通过；
- KDA 的 speculative conv/recurrent、KPool verify 事务、多 draft KV group 和
  MTP graph 均已有 DLC 适配。

当前仍开放的主要问题：

- 8/64-token 短 prompt 的 MTP 输出与 no-MTP baseline 分叉；8-token 在 Python
  EH norm fallback 下也分叉，说明不是 `fused_eh_norm` 引入，重点在 MTP target/
  KDA 小 T 路径；
- `num_speculative_tokens > 1` 未做模型级验收；
- async scheduling、MTP + prefix cache + graph 长上下文仍待验证；
- MTP 层 BF16 MoE 未量化，仍是 proposer 的主要成本之一；
- MTP 的 acceptance rate、AIME26、正式 TTFT/TPOT/吞吐/显存指标尚未归档。

---

## 2. 模型结构：第 45 层 MTP

### 2.1 基本参数

GLM-5.3-Flash-Mix 的文本配置：

| 项目 | 值 |
|---|---|
| `num_hidden_layers` | 45 |
| `num_nextn_predict_layers` | 1 |
| KDA 层 | 34 层（layer types 为 `linear_attention`） |
| sparse MLA 层 | 11 层（layer types 为 `deepseek_sparse_attention`） |
| MTP 层 | 第 45 层（`num_hidden_layers` 之后的 1 层） |
| `index_kpool` / `index_topk` / `index_n_heads` | 4 / 2048 / 32 |
| MTP hidden | 4096 |

第 45 层不是 KDA 层，也不是 mHC 层。它是一个 **MLA/KPool indexer + 288-expert
MoE + EH norm/proj + shared head** 的 decoder block。代码在
`vllm/models/glm5next/common/model.py:303` 明确：

```python
is_kda_layer = not is_mtp_layer and config.is_kda_layer(layer_idx)
```

因此 MTP 层走 MLA 路径，不创建 KDA self-attention。

### 2.2 MTP 层内部结构

`Glm5NextMultiTokenPredictorLayer`（`vllm/models/glm5next/common/mtp.py:39`）包含：

- `enorm`：对 `inputs_embeds` 做 RMSNorm；
- `hnorm`：对 `previous_hidden_states` 做 RMSNorm；
- `eh_proj`：`Linear(2H, H, bias=False)`；
- `mtp_block`：一个 `Glm5NextDecoderLayer(is_mtp_layer=True)`，内部是
  sparse MLA + MoE；
- `shared_head`：final RMSNorm + LM head；
- `topk_indices_buffer`：MTP 自己的 KPool indexer top-k scratch，可配置与 target
  共享。

forward 语义（`mtp.py:85`）：

```python
eh_input = fused_eh_norm(
    positions,
    inputs_embeds,
    previous_hidden_states,
    enorm.weight,
    hnorm.weight,
    enorm.variance_epsilon,
)
hidden_states = eh_proj(eh_input)
hidden_states, residual, _, _ = mtp_block(
    positions=positions,
    hidden_states=hidden_states,
    residual=None,
)
hidden_states, _ = shared_head.norm(hidden_states, residual=residual)
return hidden_states, hidden_states
```

关键点：

- `positions == 0` 的 embedding 会被清零；这是 MTP 训练语义要求，不是可选的
  数值装饰。
- `enorm` / `hnorm` 都是 FP32 累加、BF16 输出；`fused_eh_norm` 把 mask、
  两路 RMSNorm 和 concat 收成一个 kernel。
- MTP block 返回 `(hidden_states, residual)` 的未求和 pair，交给
  `shared_head.norm` 做 fused residual-add + RMSNorm。
- `shared_head.head` 是 target `lm_head` 的共享引用；`compute_logits()` 只做
  LM head，不重复做一次 RMSNorm。

### 2.3 权重加载与共享

MTP 权重来自同一个 checkpoint 的第 45 层：

- `get_spec_layer_idx_from_weight_name()` 识别
  `model.layers.45.*` / `layers.45.*`；
- `_rewrite_spec_layer_name()` 把 `enorm`、`hnorm`、`eh_proj`、`shared_head`
  映射到 MTP 结构，其余 decoder 权重映射到 `mtp_block`；
- target 模型加载时跳过 spec layer；MTP 模型只加载第 45 层和共享的
  `embed_tokens` / `shared_head`。

proposer 加载时还会：

- 把 target `lm_head` 共享给 MTP `shared_head.head`；
- 把 target `topk_indices_buffer` 共享给 MTP 的 MLA/indexer 模块；
- 打开 `index_share_for_mtp_iteration` 时，step 0 计算 top-k，step 1+ 复用。

当前 checkpoint 配置中 `index_share_for_mtp_iteration=true`；`num_speculative_tokens=1`
不会真正进入 step 1+，所以该复用逻辑目前只是代码路径已就绪，尚未在模型级
多 draft step 中验收。

规模上，第 45 层约 889 个权重 key、约 13.84 GiB BF16；TP8 下约 1.73 GiB/rank。
MTP 层当前保持 BF16，不参与 target 的 AWQ/量化路径。

### 2.4 缓存与容量影响

- Target 模型本身有三类 cache group：34 层 KDA state、11 层 MLA + 11 层
  indexer K cache、11 层 KPool tail scratch。
- MTP drafter 还会增加自己的 sparse MLA/indexer 与 KPool tail cache group；
  `13b663fd09 feat(mtp): support multiple draft KV cache groups` 已把
  draft 多 cache group 纳入框架。
- KDA conv state 宽度从 `3` 增加到 `3 + num_spec`，为 speculative verify 的
  accepted-token 滑动窗口预留空间。
- MTP 权重约 13.84 GiB（BF16，TP8 约 1.73 GiB/rank），且 MTP layer 的
  288-expert MoE 仍是 proposer 的主要固定成本。

---

## 3. vLLM 如何识别并构造 MTP

启动参数：

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

关键配置流程：

1. `SpeculativeConfig.hf_config_override()` 看到 `model_type == "glm5_next"`，
   改写为 `glm5_next_mtp`，设置 `n_predict = num_nextn_predict_layers`，
   architecture 改为 `Glm5NextMTPModel`
   （`vllm/config/speculative.py:1059`）。
2. `method="mtp"` 且没有显式 draft model 时，draft model 直接使用 target
   checkpoint 路径（同一模型、同一权重文件）。
3. registry 将 `Glm5NextMTPModel` 解析为
   `vllm.models.glm5next.Glm5NextMTP`。
4. V1 runner 创建 `EagleProposer`；GLM5 MTP 继承
   `SpecDecodeBaseProposer`，使用 `pass_hidden_states_to_model=True`。
5. MTP 的 `num_speculative_tokens` 默认取 checkpoint 的 `n_predict`（本模型为 1）。
   如果显式设置更大的值，必须满足 `num_speculative_tokens % n_predict == 0`
   的多模块复用约束。

> 当前生产配置只验证了 `num_speculative_tokens=1`。代码路径支持
> `num_speculative_tokens>1`（`use_multi_module_mtp()`、serial proposer steps、
> `num_prefill_lookahead_tokens > 1`），但没有完成模型级验收。

当前 DLC 生产路径是 V1 Model Runner；V2 runner 依赖 Triton driver，本环境没有
可用 Triton，启动时会回落到 V1。因此本文描述的 MTP proposer/graph 行为以
`vllm-cl` 的 V1 runner 为准。

---

## 4. 一轮完整的 MTP draft/verify 流程

下面以 `num_speculative_tokens=1`、单请求 decode 为例。

### 4.1 Target forward + sample

1. Scheduler 给请求分配 `1 + num_spec` 个 query slot（MTP 的
   `max_num_new_slots_for_drafting` 为 0，target query 本身被复用）。
2. Target 模型执行一次 forward，输出每个 token 的 hidden states。
3. Target sampler 采样当前 token；同时保留 target hidden states 供 MTP proposer
   使用。

### 4.2 Proposer 准备 draft 输入

`SpecDecodeBaseProposer.set_inputs_first_pass()`
（`vllm/v1/spec_decode/llm_base_proposer.py:820`）做 EAGLE 风格的输入重排：

```text
target tokens: [a1, b1, b2, c1, c2, c3]
draft  input:  [b1, b2, c1, c2, c3, c3]
next token:    把每个请求最后 sampled token 放到对应 sample slot
```

同时：

- `positions` 从 target positions 复制；
- `hidden_states` 缓冲填入 target hidden states；
- `next_token_ids` 通过 padded buffer 写入固定地址；
- `num_rejected_tokens_gpu` / `token_indices_to_sample` 为下一步的拒绝修正准备。

这一步的 CPU staging/H2D 在 MTP outer graph 之外执行；graph 只读取持久 buffer。

### 4.3 MTP drafter forward

`Glm5NextMTP.forward()` 调用 `Glm5NextMultiTokenPredictorLayer.forward()`：

1. `embed_tokens(input_ids)` 得到 shifted token embedding；
2. `fused_eh_norm(positions, embeds, target_hidden, enorm_w, hnorm_w, eps)`；
3. `eh_proj` 把 `[N, 2H]` 投到 `[N, H]`；
4. `mtp_block` 执行 sparse MLA + MoE；
5. `shared_head.norm` + shared LM head 得到 draft logits；
6. greedy 路径取 argmax；如启用 `use_local_argmax_reduction`，走
   vocab-parallel `get_top_tokens()`，避免 full-vocab logits all-gather。

对 `num_speculative_tokens=1`，一次 proposer 只做一次 drafter forward。  
对 `num_speculative_tokens>1`，后续 draft step 会用上一步的 hidden state 和
新的 draft token embedding 继续串行 forward；MTP layer 按
`spec_step_idx % num_mtp_layers` 复用模块。

### 4.4 Scheduler 接收 draft 并组织 verify

worker 将 draft token 异步 D2H 后交给 scheduler：

- `Scheduler.update_draft_token_ids()`（`scheduler.py:2407`）写入
  `request.spec_token_ids`；
- 下一步 scheduler 把 `1 + num_spec` 行放入
  `scheduled_spec_decode_tokens`；
- token budget 使用 `num_spec_tokens` / `draft_slots` 计算；
- MTP 的 `num_lookahead_tokens = num_speculative_tokens`；
- 多模块 MTP 的 `num_prefill_lookahead_tokens = num_speculative_tokens`，
  scheduler 不会在 prefill chunk 中切进 drafter 还要读取的 lookahead 窗口。

### 4.5 Target verify 与 KDA/KPool 状态

Target 模型对 `1 + num_spec` 行做一次 verify forward：

- `GDNAttentionMetadata` 把 query 分成 `spec_query_start_loc` 与
  `non_spec_query_start_loc`；`spec_state_indices_tensor` 形状为
  `[batch, num_spec]`（`gdn_attn.py:51`）；
- KDA conv update 接收 flattened `[tokens, D]`、`query_start_loc`、
  `num_accepted_tokens` 和 `max_query_len`；
- KDA recurrent decode 接收 `[B, next_n]` state indices，并按
  `num_accepted_tokens` 提交 accepted 快照；
- GDN spec metadata 对 padding token 做了尾部覆盖，避免 graph 下 padded token
  越界（`3eb6a0c681` 的修复）。

采样后，`sampler` 用 `-1` 标记被拒绝的 draft 行；`gpu_model_runner` 的
`_update_states_after_model_execute()`（`gpu_model_runner.py:1564`）统计每个请求
接受的 token 数，并驱动：

- `postprocess_mamba_align_gpu()`（`mamba_cache_mode=align`）或
  `postprocess_mamba_all()`；
- KDA conv/recurrent state 从 running slot 迁到目标 aligned block；
- DLC `patch_mamba_state_copy.py::_postprocess_mamba_compat()`（`:128`）的旧版
  V1 兼容路径。

### 4.6 KPool 的 speculative transaction

KPool 在 target verify 中会先执行：

1. `#19` slot mapping；
2. `#07` append/compress（写 raw tail 和完成的 pool page）；
3. `#09` select/top-k。

但 rejection sampler 此时还不知道哪些 candidate 被接受，因此 DLC runner 使用
`vllm_cl/kpool_speculative.py` 做 request/page 粒度事务：

- verify 前记录 `tail_cache` / `pool_cache` 的 touched rows；
- 采样后按 accepted prefix 回滚并重放 `#07`；
- accepted=0、部分接受、全部接受都走同一事务；
- transaction 在 graph replay 外执行，不能把 snapshot/restore 放进被捕获的 graph。

runner 接入点在 `dlc_model_runner.py:4302`
（`capture_kpool_speculative_step`）和 `:4468`
（`finish_kpool_speculative_v1`）。默认由
`DLC_KPOOL_SPECULATIVE_TRANSACTION=1` 开启；关闭只用于定位问题，不应作为生产配置。

---

## 5. DLC 侧的关键适配

### 5.1 代码地图

| 作用 | 路径 / 行号 |
|---|---|
| MTP 模型结构 | `vllm/models/glm5next/common/mtp.py:39`、`:114`、`:211` |
| MTP forward / EH norm / shared head | `vllm/models/glm5next/common/mtp.py:85`、`:95`、`:103`、`:110` |
| MTP 权重加载 | `vllm/models/glm5next/common/mtp.py:292` |
| MTP 配置探测 | `vllm/config/speculative.py:1059` |
| MTP slot / lookahead | `vllm/config/speculative.py:1885`、`:1947`、`:1976` |
| proposer 主流程 | `vllm/v1/spec_decode/llm_base_proposer.py:515` |
| draft 输入重排 | `vllm/v1/spec_decode/llm_base_proposer.py:820` |
| padded verify 输入 | `vllm/v1/spec_decode/llm_base_proposer.py:1106` |
| KDA accepted state | `vllm/v1/worker/gpu_model_runner.py:1564`、`:1589` |
| scheduler draft/lookahead | `vllm/v1/core/sched/scheduler.py:532`、`:2407` |
| GDN spec metadata | `vllm/v1/attention/backends/gdn_attn.py:51`、`:103` |
| EH norm 接入 / 原生 op | `vllm-cl/vllm_cl/patch/models/patch_glm5_mtp.py:35`、`:51` |
| MTP drafter/proposer graph | `vllm-cl/vllm_cl/patch/patch_eagle.py:72`、`:87`、`:103`、`:349` |
| KPool 事务 | `vllm-cl/vllm_cl/kpool_speculative.py:28`、`:209`、`:265` |
| KDA spec wrapper | `vllm-cl/vllm_cl/patch/patch_kda.py:162`、`:250` |
| Mamba state postprocess | `vllm-cl/vllm_cl/patch/patch_mamba_state_copy.py:128` |
| DLC graph wrapper | `vllm-cl/vllm_cl/compilation/dlc_graph.py:43` |

### 5.2 `fused_eh_norm`

- `patch_glm5_mtp.py::_patch_mtp_glm5_image_token_alias()` 在 proposer
  `load_model()` 前把 `image_token_id` 补成 `image_token_index`，绕过上游
  text-only GLM5 的 MTP 加载兼容分支。
- `Glm5NextMultiTokenPredictorLayer.forward()` 调用上游
  `vllm.models.glm5next.nvidia.ops.fused_eh_norm`；DLC 没有 Triton，因此
  `patch_glm5_mtp.py` 将 `glm5_mtp.fused_eh_norm` 重绑定到
  `fused_eh_norm_dlc`。
- 当前实现优先调用 `torch.dlc_fused_eh_norm_bf16`；满足 DLC + BF16 + INT64
  positions + `H % 256 == 0` 时走原生 op，否则保留 Python fallback。
- `VLLM_DLC_NATIVE_EH_NORM=0` 仅用于 A/B 强制回退，默认走原生 op。
- 单算子验收：静态 6/6、动态 4000/4000、覆盖率 91.56%；`T=4,H=4096`
  native 约 0.0207 ms，相对 Python fallback 22.45x。

### 5.3 KDA spec 适配

`patch_kda.py` 的 speculative 路径：

- pytorch `main` 必须包含 `51b4648508`（14-arg KDA chunk checkpoint API）；
  缺少时 kernel 与 aten schema 参数数不一致，服务会在
  `custom_dlc_kda_chunk_prefill` 卡死。
- conv update 支持 `[tokens, D] + query_start_loc +
  num_accepted_tokens + max_query_len`；
- recurrent decode 支持 `[B, next_n]` state indices 和 accepted 快照；
- KDA state shape 的 conv 宽度使用 `conv_kernel_size - 1 + num_spec`
  （`model.py:992`），为 speculative verify 预留 accepted token 滑动窗口；
- `_postprocess_mamba_compat()` 根据 `num_accepted_tokens` 计算 running state
  的 aligned target block，再执行 state copy。

### 5.4 KPool / draft cache

- target KPool 使用 `kpool_speculative.py` 的 snapshot/restore/replay；
- MTP drafter 也有自己的 sparse MLA/KPool cache group；
- 当前 `num_speculative_tokens=1` 已验证 target verify 和 MTP drafter 路径；
  多模块 MTP、drafter rejection 回滚和 drafter cache 的 prefix reuse 仍需单独验收；
- `DLC_KPOOL_SPECULATIVE_TRANSACTION` 默认开启。

### 5.5 DCP / PCP

- `patch_draft_dcp.py` 对 draft attention 关闭 DCP，避免 draft 层沿 target 的
  DCP 布局走错；
- MTP 与 `cp_kv_cache_interleave_size > 1` 的组合有显式 guard；
- 当前生产验证是 DCP=1、PCP=1；DCP/PCP 多卡组合未纳入本轮 MTP 验收。

---

## 6. Graph：target graph + MTP drafter/proposer graph

### 6.1 target graph

生产配置：

```bash
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

prefill / chunked prefill 保持 eager；decode 的 target forward 被捕获为 graph。
MTP 的 verify step 仍由 target graph 执行 `1 + num_spec` 行。

### 6.2 MTP drafter graph

`patch_eagle.py` 在 `FULL_DECODE_ONLY`、非 eager、非 parallel drafting、
非 multimodal 且 `DLC_MTP_DRAFTER_CUDAGRAPH != 0` 时启用：

- 单个 MTP drafter forward 的 graph；
- 首次 capture 前用最大 FULL descriptor eager 跑一次，初始化 workspace/
  persistent buffer；
- graph 记录输入地址，replay 时校验地址稳定。

### 6.3 MTP proposer outer graph

`DLC_MTP_PROPOSER_CUDAGRAPH != 0` 且条件满足时，outer graph 把以下内容合并为
一次 replay：

- first MTP drafter forward；
- greedy draft sampler；
- serial step 的 slot mapping / attention metadata device 更新；
- 后续 MTP drafter forward 和 sampler；
- 最终 draft token stack。

descriptor 为：

```text
(first_pass_num_tokens, num_reqs, num_speculative_tokens,
 has_lora, num_active_loras)
```

first-pass metadata 仍在 graph 外构建；outer graph capture 时内层 drafter 的
`CUDAGraphWrapper` 被强制设为 eager，避免嵌套 graph。descriptor 没有预先 capture
时回退到逐-forward graph，不会在在线请求中 lazy capture。

常用开关：

```bash
export DLC_MTP_PROPOSER_CUDAGRAPH=0   # 回退逐-forward graph，做 A/B
export DLC_MTP_DRAFTER_CUDAGRAPH=0    # 关闭 drafter 侧 graph
export DLC_MTP_DRAFTER_CUDAGRAPH_DEBUG=1
export DLC_MTP_DRAFTER_CUDAGRAPH_DEBUG_FILE=/tmp/mtp_graph_debug.log
```

---

## 7. 当前代码现状与验证结果

### 7.1 已验证

以下 MTP 端到端结果均为 `prefix cache 关闭`、`mamba-cache-mode none`、
`num_speculative_tokens=1` 的配置。

| 项目 | 结果 |
|---|---:|
| MTP eager，固定 128-token prompt，32-token greedy，50 次 | **50/50 PASS** |
| MTP + `FULL_DECODE_ONLY` graph，固定 128-token prompt，20 次 | **20/20 PASS** |
| 长上下文 marker 17,025 token | PASS，44.323 s |
| 长上下文 marker 34,025 token | PASS，91.216 s |
| 长上下文 marker 64,625 token | PASS，183.623 s |
| `fused_eh_norm_bf16` 静态/动态 | 6/6、4000/4000 |
| KDA chunk prefill 静态 | 20/20 |
| MTP eager 128-token 单次耗时 | 2.344–2.473 s |
| MTP graph 128-token 单次耗时 | 2.033–2.161 s |

说明：当前 66560 graph + 长上下文验证是在 **prefix cache 关闭** 下完成的；
MTP + prefix cache 的 graph 长上下文仍属于开放项。

### 7.2 支持矩阵

| 能力 | 当前状态 |
|---|---|
| `num_speculative_tokens=1` | 已验证（eager/graph） |
| `num_speculative_tokens>1` | `2/4/8` 已完成 eager、8/64-token 基础正确性验收；graph、长上下文、prefix-cache 和正式性能未验证 |
| MTP eager | 已验证 |
| MTP `FULL_DECODE_ONLY` | 已验证 |
| MTP outer proposer graph | 已验证 128-token / 66560 graph 场景 |
| async scheduling | eager 1/8/16 路已通过；`async + FULL_DECODE_ONLY graph` 失败，`num_spec=2` 同步 graph 也同样失败 |
| MTP + prefix cache | 部分验证；存在 checkpoint gate、命中时序和冷热翻转问题 |
| MTP + prefix cache + graph 长上下文 | 未打通 |
| MTP drafter 独立 cache / prefix reuse | 未单独验证 |
| 多模态 MTP | 不支持/未纳入当前文本-only路径 |
| parallel drafting / ubatching + MTP graph | 不纳入 outer graph |
| DCP/PCP 多卡 MTP | 当前只验证 DCP=1、PCP=1 |

### 7.3 已知问题

1. **8/64-token 短 prompt 分叉**：在确定性 MoE + CPU communication 下已不再
   复现（`num_spec=1/2/4/8` 均 10/10 PASS）。原始 DLCCL 通信仍需等确定性
   all-reduce 落地后复验。
2. **`num_speculative_tokens>1` 部分验收**：`2/4/8` 已通过 eager、8/64-token
   基础正确性；`graph`、长上下文、prefix-cache、并发和 rejection 回滚仍未验证。
3. **MTP + prefix cache checkpoint gate**：MTP 下关闭 KDA checkpoint 的 spec gate
   已移除，原生 checkpoint 已接入；但 2026-11-19 测试仍发现 3584/4096 冷/热
   输出不一致，需继续排查。
4. **MTP + prefix cache 冷热翻转**：4096 chunk 下，3584/4096 prompt 的冷/热
   greedy 输出仍可能不同；no-MTP 对照组也能复现。512 chunk 可绕过，第二次
   请求命中 3584；但长上下文 cold TTFT 约 44.6/93.9/202.5 s（16K/32K/64K）。
5. **MTP + prefix cache + graph**：4096 chunk capture 曾卡在 `0/1`；
   66560 graph 当前是在 prefix cache 关闭时通过，组合路径仍需单独验证。
6. **async scheduling**：MTP eager 下 8/64-token 和 1/8/16 路并发已通过；
   `async + FULL_DECODE_ONLY graph` 在 capture 阶段失败，错误为
   `kpool_slot_mapping: query_start_loc[0] must be 0, got 16`。同步调度下
   `num_spec=2 + graph` 也复现同一错误，说明 graph 阻断不是 async 独有。
7. **MTP 层 BF16 MoE 成本**：MTP 层没有量化，288-expert BF16 MoE 在 proposer
   固定开销中占比高；当前只融合了 `fused_eh_norm`，不是完整 MTP 层融合。
8. **drafter cache 未单独验证**：MTP 第 45 层有独立 sparse MLA/KPool cache group，
   其 prefix 命中、KPool tail 生命周期和 rejection 回滚尚未单独验收。
9. **graph/collective 偶发卡死**：曾出现
   `custom_allreduce_2x4tpu4root_simple_bf16_rdma`、`custom_gather_tensor
   bad_vld_base`；倾向 capture 时序/竞态，尚未根因确认。
10. **真正 32/64 同时 decode 受限**：受 KDA state/KV 容量限制，当前 4K 配置实际
    Running 峰值约 20；32/64 档主要是排队/缩批，不是真正同时 decode。
11. **`postprocess_mamba` 兼容层测试不足**：DLC V1 的
    `_postprocess_mamba_compat()` 仍缺 batch、多 accepted token、CoW 的独立回归。
12. **正式指标未完成**：acceptance rate、AIME26、TTFT/TPOT、吞吐、显存峰值
    尚未形成发布级报告。

### 7.4 可选扩展：hybrid MTP + suffix

`vllm-cl` 还实现了 `hybrid_mtp_suffix`：MTP 与 suffix decoding 各生成候选，
每个 request 选择一个 winner 再交给 target verifier。它只支持 V1 runner，
会强制关闭 async scheduling，不属于当前生产 MTP 默认路径，也未纳入本轮验收。

### 7.5 2026-11-19：`num_speculative_tokens > 1` 基础验证

验证条件：eager、prefix cache 关闭、`mamba_cache_mode=none`、单请求、
`max_model_len=4096`、确定性 MoE（固定 expert 到 XYS 分核）+
`USE_CPU_FOR_COLLECT_OP=True`。耗时只用于正确性阶段对比，不代表生产性能。

| `num_spec` | 8-token greedy | 64-token greedy | 8-token 单次耗时 | 64-token 单次耗时 | 观测 acceptance |
|---:|---|---|---:|---:|---|
| 1 | 已通过 | 已通过 | 7.0–7.8 s | 8.8–9.3 s | mean 2.1–2.8，约 50%–88% |
| 2 | 10/10 PASS | 10/10 PASS | 5.5–5.9 s | 8.4–8.8 s | mean 2.0–2.8，约 50%–88% |
| 3 | 10/10 PASS | 10/10 PASS | 5.1–5.6 s | 8.5–9.0 s | 8-token mean 3.0–3.3；64-token mean 2.1–2.35 |
| 4 | 10/10 PASS | 10/10 PASS | 5.1–5.2 s | 9.3–9.7 s | mean 2.2–3.8，约 30%–69% |
| 8 | 10/10 PASS | 10/10 PASS | 14.1–14.9 s | 29.0–30.1 s | mean 1.75–2.6，约 9%–20% |

结论：

- `num_speculative_tokens=2/3/4/8` 均能启动，并与 no-MTP 8/64-token 输出完全一致。
- `2/3/4` 是当前更合理的候选；`8` 的 acceptance 明显下降，耗时显著增加。
- 发布目标默认值设 `num_speculative_tokens=2`；但在 graph 阻断修复前，
  `FULL_DECODE_ONLY` 生产路径仍只能使用已验证的 `num_spec=1`，eager 路径可用 `2`。
- `3` 仅作为短请求可选档，待 graph、长上下文和 prefix-cache 复验后再考虑上浮。
- 本轮没有验证 `2/3/4/8` + graph、长上下文、prefix-cache、并发和 rejection 回滚。

结果文件：

- `work-tests/mtp/results/num_spec2_8_10_20261119.json`
- `work-tests/mtp/results/num_spec2_64_10_20261119.json`
- `work-tests/mtp/results/num_spec3_8_10_20261119.json`
- `work-tests/mtp/results/num_spec3_64_10_20261119.json`
- `work-tests/mtp/results/num_spec4_8_10_20261119.json`
- `work-tests/mtp/results/num_spec4_64_10_20261119.json`
- `work-tests/mtp/results/num_spec8_8_10_20261119.json`
- `work-tests/mtp/results/num_spec8_64_10_20261119.json`

### 7.6 2026-11-19：async scheduling 验证

验证条件：`num_speculative_tokens=2`、eager、prefix cache 关闭、
确定性 MoE + `USE_CPU_FOR_COLLECT_OP=True`。

| 场景 | 结果 |
|---|---|
| 8-token greedy vs no-MTP，10 次 | **10/10 PASS** |
| 64-token greedy vs no-MTP，10 次 | **10/10 PASS** |
| 并发语义 1/8/16 路 | **1/1、8/8、16/16 PASS** |
| async + `FULL_DECODE_ONLY graph` | **启动失败** |

async + graph 失败栈：

```text
RuntimeError: kpool_slot_mapping: query_start_loc[0] must be 0, got 16
```

补充对照：同步调度下的 `num_spec=2 + FULL_DECODE_ONLY graph` 也在 capture
阶段以同一错误退出。因此当前 graph 阻断与 async scheduling 无直接关系，后续
需要单独修复 `num_spec=2 + graph` 的 KPool slot metadata。

结果文件：

- `work-tests/mtp/results/async_numspec2_8_10_20261119.json`
- `work-tests/mtp/results/async_numspec2_64_10_20261119.json`
- `work-tests/mtp/results/async_numspec2_concurrency_20261119.json`

### 7.7 2026-11-19：`num_spec=2 + graph` 排查进展

现象：`num_spec=2 + FULL_DECODE_ONLY graph` 在 graph capture 阶段失败：

```text
kpool_slot_mapping: query_start_loc[0] must be 0, got 16
```

调用链收敛到：

```text
DLCModelRunner.capture_model
  -> EagleProposer.propose
  -> build_per_group_and_layer_attn_metadata
  -> DeepseekV32IndexerMetadataBuilder.build_for_drafting
  -> GLM5 fused KPool slot mapping
  -> torch.dlc_kpool_slot_mapping
```

已确认的关键事实：

1. graph capture 中同时存在两组 metadata：
   - target/verify：`num_reqs=16, num_tokens=48`，qsl 为 `[0,3,6,...,48]`；
   - draft：`num_reqs=16, num_tokens=16`，CPU qsl 为 `[0,1,2,...,16]`。
2. draft 的 **CPU qsl 始终正确**，但同一时刻 GPU qsl 可能已经变成
   `[1,1,...,0]`、随机 int32 或 `[3,3,...,-48]`。
3. 在 `_dummy_run` 中重置 qsl[0]、填充 padding 并 `copy_to_gpu()` 后，GPU qsl
   仍可能被后续异步 H2D 覆盖；capture 中直接 `torch.dlc.synchronize()` 会报
   `synErrorStreamCaptureUnsupported`，不可用。
4. 用 CPU 真值 qsl 构造 capture 专用持久 buffer 后，Python 侧打印已正确，但
   kernel 仍可能在实际消费时读到旧值 `16`，说明 qsl buffer 的写入/复用与 kernel
   消费之间缺少 graph-safe 的生命周期隔离。

结论：根因不在 KPool 的参数校验，也不在 #19 kernel 本身，而在 **graph capture
期间 draft `query_start_loc` 的异步 H2D staging 与 buffer 复用**。`num_spec=1`
时路径恰好不暴露该竞态；`num_spec=2` 的 target/draft metadata 形状和调用次数
增加后，stale qsl 被 #19 读到。

后续修复方向：

- 在 capture 前预先准备 graph bucket 对应的 qsl，并使用 capture 生命周期内不会被
  其他 metadata 覆盖的独立 persistent buffer；
- 或把 draft qsl 直接构造成 device buffer，在 capture stream 上用事件 fencing，
  避免 Python 侧 `async_tensor_h2d` 临时 buffer 竞争；
- 或为 graph capture 使用静态 qsl（同一 bucket 的 query 边界固定），只在 replay
  前通过受控的 device copy 更新；
- 修复后需重跑 `num_spec=2/3` + graph 的 8/64-token、并发和长上下文。

本轮实验性 patch 和调试打印已撤回；`num_spec=2 + graph` 仍为已知阻断项。

补充验证（2026-11-19）：

- 设置既有 PyTorch DLC 开关 `DLC_SYN_COPY_ASYNC=O0` 后，仍报
  `query_start_loc[0] must be 0, got 16`，说明不是简单 copy 异步等级问题。
- 设置 `DLC_KPOOL_SLOT_MAPPING_CHECK=0` 后 graph capture 可以完成，但
  `num_spec=2` 的 8-token 输出从第 2 个 token 起与 no-MTP 不同；因此不能关闭
  校验作为修复，实际 KPool qsl 仍然是错的。
- 结论进一步收敛为：必须在进入 graph capture 前预置所有 target/draft qsl，
  并从 `CommonAttentionMetadata` 构造处使用这些 persistent buffer；在
  KPool metadata builder 内临时生成/拷贝已经太晚。

---

## 8. 关键限制与后续工作

### 8.1 不建议直接宣称的能力

- 不建议宣称“MTP + prefix cache 生产就绪”：4096 chunk 的命中时序和冷热翻转
  仍存在。
- 不建议宣称“`num_speculative_tokens>1` 已完整支持”：2/4/8 只完成了 eager、
  8/64-token 基础正确性；graph、长上下文、prefix-cache 和正式性能仍缺。
- 不建议宣称“MTP 在所有 prompt 长度和通信配置下与 baseline 完全一致”：
  当前一致性结果依赖确定性 MoE + CPU communication；原始 DLCCL 通信仍需复验。
- 不建议把 512 chunk 当作生产 workaround：它只是绕过非末端 checkpoint split，
  长上下文冷启动代价过高。

### 8.2 建议顺序

1. 先修 KDA causal-conv checkpoint 和 MTP checkpoint gate，解决 prefix-cache
   命中时序和冷热翻转；
2. 再排查 8/64-token MTP/KDA 小 T 分叉；
3. `num_speculative_tokens=2/4/8` 已完成基础正确性；继续补 graph、长上下文、
   prefix-cache、并发和 draft cache rejection 回滚验收；
4. 打通 MTP + prefix cache + graph 4096/66560，再验证 async scheduling；
5. 最后补 acceptance、AIME26、TTFT/TPOT、吞吐和显存峰值。

---

## 9. 证据与结果文件

MTP 端到端结果：

- `work-tests/mtp/results/fused_eh_norm_baseline_8_20261118.json`
- `work-tests/mtp/results/fused_eh_norm_baseline_64_20261118.json`
- `work-tests/mtp/results/fused_eh_norm_baseline_128_20261118.json`
- `work-tests/mtp/results/fused_eh_norm_mtp_eager_128_50_20261118.json`
- `work-tests/mtp/results/fused_eh_norm_mtp_graph_128_20_20261118.json`
- `work-tests/mtp/results/fused_eh_norm_mtp_graph_longctx_20261118.json`
- `work-tests/mtp/results/fused_eh_norm_mtp_eager_fallback_10_20261118.json`

Prefix cache 结果：

- `work-tests/mtp/results/prefix_cache_20261116/`

关键日志：

- `/root/workspace/log/fused_eh_norm_mtp_eager_native2_20261118.log`
- `/root/workspace/log/fused_eh_norm_mtp_graph66560_20261118.log`
- `/root/workspace/log/fused_eh_norm_nomtp_eager_20261118.log`
- `/root/workspace/log/fused_eh_norm_mtp_eager_fallback_20261118.log`

