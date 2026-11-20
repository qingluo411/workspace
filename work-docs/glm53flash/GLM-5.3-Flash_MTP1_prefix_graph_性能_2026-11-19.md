# GLM-5.3-Flash MTP=1 + Prefix Cache + Graph 性能结果

> 测试时间：2026-11-19T09:26:31+00:00（UTC）  
> 状态：测试完成  
> 测试配置：`method=mtp`、`num_speculative_tokens=1`、prefix cache 开启、`FULL_DECODE_ONLY` graph、正常 DLCCL 通信。  
> 数据口径：`vllm bench serve` 客户端实测；**不包含 KPool 优化后的理想值，也未替换生产路径中的该算子**。

## 1. 结论摘要

- 计划场景 `6` 个：通过 `6`、失败/部分失败 `0`、跳过 `0`。
- 短上下文单请求：TTFT P50 `402.21 ms`，TPOT P50 `59.29 ms`，输出吞吐 `15.91 tok/s`。
- 短上下文 8 并发：TTFT P50 `1,984.67 ms`，TPOT P50 `96.30 ms`，输出吞吐 `66.48 tok/s`。
- 4K prompt 单请求：TTFT P50 `993.82 ms`，E2EL 中位数 `1,423.42 ms`。
- 所有时延和吞吐均为客户端侧统计，不含服务端 profiler 插桩。

## 2. 服务与测试口径

### 2.1 服务配置

| 项目 | 值 |
|---|---|
| API | `http://127.0.0.1:18005` |
| Model | `GLM-5.3-Flash-Mix` |
| 模型路径 | `/data/models/GLM-5.3-Flash-Mix` |
| Tokenizer | `/data/models/GLM-5.3-Flash-Mix` |
| `max_model_len` | `8,192` |
| `max_num_batched_tokens` | `4,096` |
| `max_num_seqs` | `16` |
| `speculative_config` | `method=mtp`, `num_speculative_tokens=1` |
| `enable_prefix_caching` | `true` |
| `mamba_cache_mode` | `align` |
| `async_scheduling` | `false` |
| H2D copy | `DLC_SYN_COPY_ASYNC=O0`（同步） |
| Collective | 正常 DLCCL，未设置 `USE_CPU_FOR_COLLECT_OP` |
| TP | `8` |
| `block_size` | `256` |
| `gpu_memory_utilization` | `0.85` |
| Graph | `开启` |
| Graph mode | `FULL_DECODE_ONLY` |
| `DLC_SYN_GRAPH_CACHE` | `1` |

### 2.2 压测方法

- 客户端：`vllm bench serve --backend openai-chat`。
- 数据集：`random`，固定 `--seed`；`--temperature 0`、`--ignore-eos`，使输出长度可重复。
- 每个场景首轮先做 `1` 个不计时 warmup，随后正式重复 `3` 次；表中数值为正式运行的跨 run 中位数。
- 由于 prefix cache 开启，重复 run 会复用已缓存前缀；表中数值是服务预热后的稳定态，
  不代表所有请求都从冷前缀开始。`medium_decode_c8` 的首个正式 run 仍包含部分冷请求，
  其 TTFT 和输出吞吐明显差于后两轮，聚合值按跨 run 中位数统计。
- 时延指标来自客户端请求时间戳：TTFT 为首 token 时延，TPOT 排除首 token，ITL 为 token 间隔，E2EL 为请求端到端时延。
- 未启用服务端 profiler / kernel log；未在测试期间运行其他压测。

### 2.3 构建标记

- Kernel 仓库 revision：`4de635bbb`；工作区是否 dirty：`开启`。
- 已安装算子库：`/usr/local/chipltech/synapse/lib/libcustom_dlc_perf_lib.so`，大小 `1,616,131,856` B，mtime `2026-11-19T04:26:50`。

## 3. 吞吐结果

| 场景 | 名义输入/输出 token | 请求数 | 并发 | 有效/计划 run | 实际平均输入/输出 token | 输出 tok/s | 请求/s | 完成/失败 | 峰值并发 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 短上下文 decode / 单请求 (`short_decode_c1`) | 49/96 | 1 | 1 | 3/3 | 61.0/96.0 | 15.91 | 0.17 | 1/0 | 1 |
| 短上下文 decode / 8 并发 (`short_decode_c8`) | 49/96 | 8 | 8 | 3/3 | 61.0/96.0 | 66.48 | 0.69 | 8/0 | 8 |
| 中等 prompt decode / 单请求 (`medium_decode_c1`) | 860/64 | 1 | 1 | 3/3 | 872.0/64.0 | 12.45 | 0.19 | 1/0 | 1 |
| 中等 prompt decode / 8 并发 (`medium_decode_c8`) | 860/64 | 8 | 8 | 3/3 | 872.0/64.0 | 36.75 | 0.57 | 8/0 | 8 |
| 1K prompt TTFT / 单请求 (`ttft_1217_c1`) | 1217/8 | 1 | 1 | 3/3 | 1,229.0/8.0 | 4.61 | 0.58 | 1/0 | 1 |
| 4K prompt / 单请求 (`prefill_4k_c1`) | 3610/8 | 1 | 1 | 3/3 | 3,622.0/8.0 | 5.62 | 0.70 | 1/0 | 1 |

## 4. 时延结果

| 场景 | TTFT mean | TTFT P50 | TTFT P90 | TPOT mean | TPOT P50 | ITL P50 | E2EL mean | E2EL P50 | E2EL P90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 短上下文 decode / 单请求 (`short_decode_c1`) | 402.21 | 402.21 | 402.21 | 59.29 | 59.29 | 109.84 | 6,031.76 | 6,031.76 | 6,031.76 |
| 短上下文 decode / 8 并发 (`short_decode_c8`) | 1,787.87 | 1,984.67 | 1,985.42 | 97.25 | 96.30 | 178.77 | 11,026.55 | 11,133.12 | 11,396.27 |
| 中等 prompt decode / 单请求 (`medium_decode_c1`) | 1,075.38 | 1,075.38 | 1,075.38 | 64.61 | 64.61 | 109.75 | 5,141.95 | 5,141.95 | 5,141.95 |
| 中等 prompt decode / 8 并发 (`medium_decode_c8`) | 6,797.75 | 7,614.22 | 7,615.28 | 109.47 | 98.52 | 176.07 | 13,694.38 | 13,750.88 | 13,854.58 |
| 1K prompt TTFT / 单请求 (`ttft_1217_c1`) | 1,305.03 | 1,305.03 | 1,305.03 | 60.91 | 60.91 | 106.44 | 1,733.94 | 1,733.94 | 1,733.94 |
| 4K prompt / 单请求 (`prefill_4k_c1`) | 993.82 | 993.82 | 993.82 | 61.20 | 61.20 | 107.60 | 1,423.42 | 1,423.42 | 1,423.42 |

> 时延单位均为 ms。TTFT/P90 仅反映对应请求形状的观测值；不同输入和输出长度不应直接混作同一 SLA。

## 5. 重复性

| 场景 | 有效 run | 输出 tok/s (min / median / max) | TTFT P50 ms (min / median / max) | TPOT P50 ms (min / median / max) |
|---|---:|---:|---:|---:|
| 短上下文 decode / 单请求 (`short_decode_c1`) | 3 | 15.74 / 15.91 / 15.96 | 398.77 / 402.21 / 403.34 | 59.09 / 59.29 / 59.96 |
| 短上下文 decode / 8 并发 (`short_decode_c8`) | 3 | 66.46 / 66.48 / 68.38 | 1,961.54 / 1,984.67 / 2,080.26 | 95.08 / 96.30 / 97.51 |
| 中等 prompt decode / 单请求 (`medium_decode_c1`) | 3 | 12.35 / 12.45 / 12.57 | 1,071.21 / 1,075.38 / 1,088.35 | 63.77 / 64.61 / 64.97 |
| 中等 prompt decode / 8 并发 (`medium_decode_c8`) | 3 | 22.73 / 36.75 / 37.52 | 7,456.51 / 7,614.22 / 11,630.87 | 96.52 / 98.52 / 167.58 |
| 1K prompt TTFT / 单请求 (`ttft_1217_c1`) | 3 | 4.61 / 4.61 / 4.63 | 1,302.47 / 1,305.03 / 1,309.29 | 60.66 / 60.91 / 61.51 |
| 4K prompt / 单请求 (`prefill_4k_c1`) | 3 | 5.59 / 5.62 / 5.64 | 991.01 / 993.82 / 995.03 | 60.83 / 61.20 / 62.56 |

## 6. 配置相关观测

- 服务启动日志确认 `FULL_DECODE_ONLY`、`Graph capturing finished`，并且 MTP
  proposer/drafter 均使用 graph。
- MTP `SpecDecoding metrics` 共 29 个统计窗口：Mean acceptance length
  `1.59 / 1.88 / 2.00`（min / median / max），平均 draft acceptance rate
  `58.8% / 88.0% / 100.0%`。
- Prefix cache hit rate 从首次请求的 `0%` 逐步升到 `52.3%`；本次压测使用固定
  random 数据集 seed，因此该数值包含跨 run 的重复前缀复用。

## 7. 本次阻断修复

初次复测 `medium_decode_c8` 时，服务在 KDA prefill 路径崩溃：

```text
dlc_kda_causal_conv_prefill: checkpoint_offsets must be int32 [B]
```

诊断表明实际传入的 `checkpoint_offsets` 是 `int32`，但形状为 `(6,)`，而该次
KDA prefill 的 batch 为 `5`。根因是 MTP 混合 batch 中 checkpoint builder 使用
`m.num_reqs`，把 spec/draft 行也计入了 checkpoint request rows。修正为只对
非 spec 行生成 checkpoint 后，`medium_decode_c8` 恢复 `3/3 PASS`，最终矩阵全部通过。

## 8. 与旧无 MTP 基线参考对比

以下基线来自 `GLM-5.3-Flash_最终性能结果_2026-11-02.md`，是**不同配置的参考对照**：
旧基线无 MTP，`max_model_len=4096`、`max_num_seqs=64`；本次为 MTP=1、`max_model_len=8192`、
`max_num_seqs=16`、prefix cache 开启。因此不能把差异全部归因于 MTP。

| 场景 | 旧输出 tok/s | 本次输出 tok/s | 旧 TTFT P50 | 本次 TTFT P50 | 旧 TPOT P50 | 本次 TPOT P50 |
|---|---:|---:|---:|---:|---:|---:|
| 短上下文单请求 | 14.12 | 15.91 | 446.05 ms | 402.21 ms | 66.83 ms | 59.29 ms |
| 短上下文 8 并发 | 56.71 | 66.48 | 2,754.25 ms | 1,984.67 ms | 113.54 ms | 96.30 ms |
| 中等 prompt 单请求 | 10.73 | 12.45 | 1,204.50 ms | 1,075.38 ms | 75.53 ms | 64.61 ms |
| 中等 prompt 8 并发 | 27.16 | 36.75 | 8,680.97 ms | 7,614.22 ms | 161.39 ms | 98.52 ms |
| 1K prompt 单请求 | 2.46 | 4.61 | 2,691.95 ms | 1,305.03 ms | 80.49 ms | 60.91 ms |
| 4K prompt 单请求 | 3.88 | 5.62 | 1,218.15 ms | 993.82 ms | 120.44 ms | 61.20 ms |

## 9. 状态与限制

- 完成时间：2026-11-19T09:33:22+00:00（UTC）。
- 本次表格对应 Graph `开启`、`DLC_SYN_GRAPH_CACHE=1` 的服务路径。
- 4K 及更长 prompt 的 E2EL 主要受 prefill 支配；decode 吞吐应以对应输出长度的结果解读。
- 64 并发可能落入非整除 graph bucket；该成本已包含在实测值中，不另行剔除。
- **本表没有采用 KPool 优化后的性能模型或估算数据**。如需评估优化收益，应另列对照表。
- 本次没有测试 cold-prefix 的逐请求延迟分布；prefix cache 的冷热一致性问题仍需按
  `GLM-5.3-Flash_PrefixCache机制与命中现状_2026-11-17.md` 单独验收。

### 失败项

- 无。

## 10. 原始数据

- 结果目录：`work-tests/mtp/results/20261119_mtp1_prefix_graph_final/20261119_mtp1_prefix_graph_final`
- 汇总 JSON：`work-tests/mtp/results/20261119_mtp1_prefix_graph_final/20261119_mtp1_prefix_graph_final/summary.json`
- 每个场景的 JSON 和日志位于 `work-tests/mtp/results/20261119_mtp1_prefix_graph_final/20261119_mtp1_prefix_graph_final/<case_key>/`。

