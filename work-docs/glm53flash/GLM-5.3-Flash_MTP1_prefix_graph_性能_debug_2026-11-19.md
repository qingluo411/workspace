# GLM-5.3-Flash 最终性能结果

> 测试时间：2026-11-19T09:17:50+00:00（UTC）  
> 状态：测试完成  
> 数据口径：`vllm bench serve` 客户端实测；当前发布路径；**不包含 KPool 优化后的理想值，也未替换生产路径中的该算子**。

## 1. 结论摘要

- 计划场景 `1` 个：通过 `0`、失败/部分失败 `1`、跳过 `0`。
- 尚无有效场景结果。
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
| TP | `8` |
| `block_size` | `256` |
| `gpu_memory_utilization` | `0.85` |
| Graph | `开启` |
| Graph mode | `FULL_DECODE_ONLY` |
| `DLC_SYN_GRAPH_CACHE` | `1` |

### 2.2 压测方法

- 客户端：`vllm bench serve --backend openai-chat`。
- 数据集：`random`，固定 `--seed`；`--temperature 0`、`--ignore-eos`，使输出长度可重复。
- 每个场景首轮先做 `0` 个不计时 warmup，随后正式重复 `1` 次；表中数值为正式运行的跨 run 中位数。
- 时延指标来自客户端请求时间戳：TTFT 为首 token 时延，TPOT 排除首 token，ITL 为 token 间隔，E2EL 为请求端到端时延。
- 未启用服务端 profiler / kernel log；未在测试期间运行其他压测。

### 2.3 构建标记

- Kernel 仓库 revision：`4de635bbb`；工作区是否 dirty：`开启`。
- 已安装算子库：`/usr/local/chipltech/synapse/lib/libcustom_dlc_perf_lib.so`，大小 `1,616,131,856` B，mtime `2026-11-19T04:26:50`。

## 3. 吞吐结果

| 场景 | 名义输入/输出 token | 请求数 | 并发 | 有效/计划 run | 实际平均输入/输出 token | 输出 tok/s | 请求/s | 完成/失败 | 峰值并发 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 中等 prompt decode / 8 并发 (`medium_decode_c8`) | 860/64 | 8 | 8 | 0/1 | — | — | — | —/— | — |

## 4. 时延结果

| 场景 | TTFT mean | TTFT P50 | TTFT P90 | TPOT mean | TPOT P50 | ITL P50 | E2EL mean | E2EL P50 | E2EL P90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 中等 prompt decode / 8 并发 (`medium_decode_c8`) | — | — | — | — | — | — | — | — | — |

> 时延单位均为 ms。TTFT/P90 仅反映对应请求形状的观测值；不同输入和输出长度不应直接混作同一 SLA。

## 5. 重复性

| 场景 | 有效 run | 输出 tok/s (min / median / max) | TTFT P50 ms (min / median / max) | TPOT P50 ms (min / median / max) |
|---|---:|---:|---:|---:|
| 中等 prompt decode / 8 并发 (`medium_decode_c8`) | 0 | — | — | — |

## 6. 状态与限制

- 完成时间：2026-11-19T09:18:06+00:00（UTC）。
- 本次表格对应 Graph `开启`、`DLC_SYN_GRAPH_CACHE=1` 的服务路径。
- 4K 及更长 prompt 的 E2EL 主要受 prefill 支配；decode 吞吐应以对应输出长度的结果解读。
- 64 并发可能落入非整除 graph bucket；该成本已包含在实测值中，不另行剔除。
- **本表没有采用 KPool 优化后的性能模型或估算数据**。如需评估优化收益，应另列对照表。

### 失败项

- `medium_decode_c8` run 1: vLLM reported 7 failed request(s) (log: `work-tests/mtp/results/20261119_mtp1_prefix_graph_debug/20261119_mtp1_prefix_graph_debug/medium_decode_c8/medium_decode_c8.run00.log`)

## 7. 原始数据

- 结果目录：`work-tests/mtp/results/20261119_mtp1_prefix_graph_debug/20261119_mtp1_prefix_graph_debug`
- 汇总 JSON：`work-tests/mtp/results/20261119_mtp1_prefix_graph_debug/20261119_mtp1_prefix_graph_debug/summary.json`
- 每个场景的 JSON 和日志位于 `work-tests/mtp/results/20261119_mtp1_prefix_graph_debug/20261119_mtp1_prefix_graph_debug/<case_key>/`。

