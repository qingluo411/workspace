# GLM-5.3-Flash MTP + Graph：KPool 投机事务导致长序列分叉分析

日期：2026-11-26
对象：`GLM-5.3-Flash-Mix`、MTP3、TP4、`FULL_DECODE_ONLY` graph、prefix cache
环境：`/root/workspace`，驱动 `Driver Version_v26.0.1`，RTABI 3.2.0

## 1. 结论

问题不是采样随机性，也不是 MoE/通信的普通数值抖动，而是 **MTP verify 写 KPool 后，被拒绝 draft token 在 graph replay 中没有回滚**：

1. target verify 一次向 KPool 写入 `1 + num_speculative_tokens = 4` 行；
2. sampler 只接受其中一部分，剩余 draft 行本应回滚；
3. eager 路径由 `vllm_cl/kpool_speculative.py` 的 Python 事务完成
   snapshot/restore/replay；
4. 但 `FULL_DECODE_ONLY` 使用 monolithic `CUDAGraphWrapper`，KPool Python
   路径在 replay 时不执行，事务记录为空；
5. 被拒绝的候选 token 在关闭 pool page 时污染 KPool pool cache，后续
   sparse-MLA indexer 选错 top-k，最终表现为约 160 token 后输出分叉。

修复后，graph + MTP3 的热重复请求在 500/1200 token 上逐 token 一致；4096-token
prompt 的 prefix-cache 热命中路径也一致。

## 2. 复现与 dump 方法

复现配置为测试收敛后的 MTP3/TP4/graph 配置（`max_model_len=16384`）。使用
349-token AIME chat prompt，温度 0、greedy、`min_tokens=max_tokens`：

```bash
python work-tests/mtp/run_repeat_token_dump.py \
  --port 18005 \
  --prompt-token-ids work-tests/mtp/results/20261126_divergence/prompt_aime_q1_chat_long.json \
  --runs 2 --warmup-tokens 1200 --max-tokens 1200 \
  --output work-tests/mtp/results/20261126_divergence/final_1200.json
```

hash dump 使用 `VLLM_DUMP_MODEL_MODE=hash`，只记录 tensor hash；比较工具为
`work-tests/mtp/compare_dump_hashes.py` 和 `compare_full_dumps.py`。

## 3. 首个真实分叉点

修复前 graph hash dump（`graph_cache_hash/`）中：

- warmup/run 在相对 step 0..49 的 `hidden_states`、`sample_hidden_states`、
  `logits` 全部一致；
- 相对 step 50 的 **输入 token 仍一致**，但 `hidden_states`、`logits` 开始不同；
- layer 3 indexer 的 `tail_cache` 在 step 50 首次不同；
- eager 完整 dump 的首个差异是 layer 3 `topk_indices`（`[4,2176]`，1064 个元素
  不同，最大差 408），对应生成 token 160。

因此分叉发生在 target verify 内部，而不是 sampler 或 tokenizer。

## 4. 根因证据

- `DLC_KPOOL_SPECULATIVE_TRANSACTION=0` + graph：仍然在 token 160 分叉；
- `DLC_KPOOL_SPECULATIVE_TRANSACTION=1` + eager：500/1200 token 一致；
- 关闭 MTP：一致；
- 关闭 `DLC_SYN_GRAPH_CACHE`、关闭 drafter/proposer graph、关闭 dirty
  block-table skip：均仍分叉；
- 因此问题收敛到 **target FULL graph 内的 KPool 事务失效**。

代码层面：

- `FULL_DECODE_ONLY` 走完整 `CUDAGraphWrapper`，不是 breakable/PIECEWISE；
- replay 不重新执行 `DLCGlm5NextIndexer.forward()` 的 Python；
- `record_kpool_append_compress()` 因此不会在新 step 记录 `_Call`；
- `finish_kpool_speculative_v1()` 看到空 calls，直接跳过 rollback/replay。

KPool 的 pool page 是跨 step 可见的；单个被拒绝且带非负 pool slot 的 draft
token 会留下错误 pool，直到后续 `#09` select/top-k 读到它才显性化，这与
“前 159 token 一致、约第 160 token 分叉”的现象吻合。

## 5. 修复

### 5.1 graph-replay-safe KPool transaction

文件：`vllm-cl/vllm_cl/kpool_speculative.py`

- 在 graph capture 期间记录 KPool 调用的静态 tensor 引用；
- 按 `(num_tokens, num_reqs)` 保存 capture template；
- replay 前安装 template，并在 graph replay 前对当前 pool/tail mapping 命中的
  cache 行做 device-side snapshot；
- sampler 返回后，若拒绝行关闭了 pool page：
  - 恢复 snapshot；
  - 按 accepted prefix 重放 `#07` `dlc_kpool_append_compress`；
- 只选择 target model 的 KPool cache：抓取日志显示同一 key 同时包含 target 11
  个 cache 和 drafter 12 个 cache，必须过滤，避免 rollback 污染 drafter。

文件：`vllm-cl/vllm_cl/worker/dlc_model_runner.py`

- 向 `capture_kpool_speculative_step()` 传入 graph template key 和 target KPool
  cache data_ptr 集合；
- 在 graph replay 的 batch descriptor 上复用 capture template；
- dump 输入 metadata 仅在 `VLLM_DUMP_MODEL_OUTPUTS=1` 时构造，避免生产路径常驻
  debug 开销。

### 5.2 顺带修复

- `vllm-cl/vllm_cl/patch/patch_kv_block_zeroer.py`：KPool pool cache 使用 32 KB
  物理页大小清零，修复逻辑页 16 KB/物理页 32 KB 时只清半页的问题；
- `vllm-cl/vllm_cl/attention/sparse_mla.py`：NoPE-512 支持 `heads=16/32`，
  使 TP4 的 sparse MLA 配置可启动；
- `vllm-cl/vllm_cl/dump_model.py`：新增 hash-only dump、step/module 过滤和
  graph runtime tensor hash，用于定位 graph replay 内部首个差异。

## 6. 验证结果

修复后主要结果：

| 场景 | 结果 |
|---|---|
| graph + MTP3，500 token，短 decode 预热后 warm/run | `all_same=True` |
| graph + MTP3，1200 token，短 decode 预热后 warm/run | `all_same=True` |
| graph + MTP3，1200 token 连续重复 3 组 | 全部 `all_same=True` |
| 4096-token prompt，prefix-cache 热命中，1200 token | run1/run2 `all_same=True`，prefix hit rate 约 32.6%/47.5% |
| eager + MTP3，500 token | `all_same=True` |
| 关闭 MTP | 一致 |

结果文件：

- `work-tests/mtp/results/20261126_divergence/final4_1200.json`
- `work-tests/mtp/results/20261126_divergence/final4b_1200.json`
- `work-tests/mtp/results/20261126_divergence/final4c_1200.json`
- `work-tests/mtp/results/20261126_divergence/final4_4096p_1200.json`
- `work-tests/mtp/results/20261126_divergence/graphtxn3_1200.json`
- 修复前对照：`final_1200.json`、`final2_1200.json`（token 160 分叉）

### 6.1 logits 级严格复核

补充使用 4096-token prompt、`--enable-prompt-tokens-details` 和 hash dump，
先做 cold warmup，再连续发送 4 个相同 hot 请求，每个请求生成 1200 token。
4 个 hot 请求的 `cached_tokens` 均为 `3584`，每个请求都产生 409 个模型 step。

比较范围是每个 step 的：

- `hidden_states`
- `sample_hidden_states`
- `logits`

两两比较结果：

| 比较 | runtime hash records | mismatch |
|---|---:|---:|
| run1 vs run2 | 1227 | 0 |
| run2 vs run3 | 1227 | 0 |
| run3 vs run4 | 1227 | 0 |
| run1 vs run4 | 1227 | 0 |

因此在这组都命中 `3584` prompt cache token 的 hot 请求中，logits 是逐 step
bit-exact 一致的，不只是 token 一致。示例 `logits` hash 在相对 step
0/50/160/408 四次请求均完全相同。

进一步把输出长度从 1200 延长到 **3000 token**（4096-token prompt，cold
warmup 后 3 个 hot 请求）：

- 3 次 hot 请求 `cached_tokens` 均为 `3584`；
- 每次生成 3000 token，单次耗时约 111 s；
- 每个请求 962 个模型 step；
- 每个请求 2886 个 runtime hash record；
- run1↔run2、run2↔run3、run1↔run3 均为 **0 mismatch**；
- 示例 `logits` hash 在相对 step 0/160/500/961 三次请求完全相同。

3000-token 结果文件：

- `work-tests/mtp/results/20261126_divergence/logits_long_3000.json`
- `work-tests/mtp/results/20261126_divergence/logits_long_3000.log`
- `work-tests/mtp/results/20261126_divergence/logits_long_run1_vs_run2.json`
- `work-tests/mtp/results/20261126_divergence/logits_long_run2_vs_run3.json`
- `work-tests/mtp/results/20261126_divergence/logits_long_run1_vs_run3.json`

注意：cold warmup 与 hot run 不能直接逐 iteration 对齐比较，因为 prefill
chunk 数和首 token 分布不同；本结论只针对“都命中相同 prompt cache 的 hot
请求”。

结果文件：

- dump：`work-tests/mtp/results/20261126_divergence/logits_hot_20261126_v2/`
- 请求输出：`work-tests/mtp/results/20261126_divergence/logits_hot_repeat.json`
- 比较：`logits_run1_vs_run2.json`、`logits_run2_vs_run3.json`、
  `logits_run3_vs_run4.json`、`logits_run1_vs_run4.json`

## 7. 已知边界与建议

- **热重复已修复**，但当前证据显示：全新服务启动后，如果第一条请求就是
  1200-token 的长 cold request，仍观察到一次 token 160 分叉；先发一条 32-token
  decode 请求预热后，后续 500/1200 hot request 稳定一致。
- 该 cold-start 残余更像 graph capture/第一条真实 decode 的缓存初始化问题，
  与已修复的 replay 事务不是同一层；建议生产启动流程增加一次短 decode
  warmup，或在 graph capture 完成后显式跑一次真实短 decode 事务。
- 修复默认开启（`DLC_KPOOL_SPECULATIVE_TRANSACTION=1`）；`=0` 仅用于回归对照。

## 8. 复现命令

```bash
export DLC_SYN_GRAPH_CACHE=1
export VLLM_USE_DLC_COL_MAJOR_MATMUL=1
export PYTORCH_DLC_ALLOC_CONF=expandable_segments:True

vllm serve /data/models/GLM-5.3-Flash-Mix \
  --served-model-name GLM-5.3-Flash-Mix --language-model-only \
  --dtype bfloat16 --max-model-len 16384 --max-num-batched-tokens 4096 \
  --max-num-seqs 1 --block-size 256 --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 --trust-remote-code --port 18005 \
  --enable-prefix-caching --prefix-cache-retention-interval 256 \
  --mamba-prefix-checkpoint-max-per-request 32 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kernel-config '{"enable_jit_warmup": false}' \
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm47
```

调试时可用 `DLC_KPOOL_TEMPLATE_DEBUG=1` 打印 capture/attach/rollback 摘要；默认关闭。
