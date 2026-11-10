# GLM-5.3-Flash 发布前功能与性能验证记录

日期：2026-10-22

## 1. 验证范围与代码状态

| 仓库 | 分支 | HEAD |
|---|---|---|
| vllm | `release-glm53flash` | `2309b38c8d` |
| vllm-cl | `release-glm53flash` | `f971a82` |
| DLC_Custom_Kernel | `release-glm53flash` | `92a36d525` |
| pytorch | `release-glm53flash` | `ab7c0354ec` |

运行配置：

- 8 卡 TYD，TP=8，`--enforce-eager`，`block_size=256`，`max_model_len=131072`；
- `max_num_batched_tokens=4096`，chunked prefill 开启；
- 模型：`/data/models/GLM-5.3-Flash-Mix`；
- 服务端口：`18005`；
- 所有 DLC 进程设置 `DLC_SKIP_VERSION_CHECK=1`；
- 快路径开关：`DLC_KPOOL_OP=fast`、`DLC_KPOOL_SELECT_TOPK_OP=fast`、
  `DLC_KPOOL_VALIDATE=0`。

本轮只保留用户要求的正式代码路径；NaN 排查用的 payload dump、逐层 hook 和
临时 kernel 日志均未常驻。性能数字分别标注了 profiler / kernel-log 开销。

## 2. 本轮代码修正

### 2.1 #09 prefill 自动回退

`vllm-cl/vllm_cl/patch/models/kpool_indexer.py` 的
`_can_use_select_topk_op()` 增加纯形状门禁：

```python
query.shape[0] == num_reqs
```

只有每个请求恰好一个 query token 的批次才启用 #09 fast，也就是 decode。多 token
的 prefill 批次继续使用 #07 更新缓存，但 top-k/展开走 native。该判断不读取设备
标量，不引入 D2H 同步。

原因是原有 #09 fast 在 prefill 多 token 批次上严重退化：4K 从 native 的约 24.7 s
退化到约 98.7 s；8K 会触发 `sample_tokens` RPC 超时和 `EngineDeadError`。

> 后续决策（同日）：由于该模型人员投入减少、不再继续优化 #09 kernel，已移除上述
> decode-only 门禁，prefill 和 decode 统一走 `#07 + #09`。本报告 4.2 节的 kernel
> 命中矩阵是拆除门禁前的快照；拆除后的性能代价见
> `KPool_09_select_topk_性能量化_2026-10-22.md` 第 8 节。

### 2.2 #19 真实服务路径

此前 V1 eager 路径会命中 `use_simple_cpu_slot_mapping`，导致 #19 没有进入真实
模型路径。已在 `vllm-cl/vllm_cl/worker/dlc_model_runner.py` 增加门禁：存在 KPool
tail 表时禁用简单 CPU slot-mapping 分支；同时修正 `patch_block_table.py` 中
`MultiGroupBlockTable` 的几何读取。

### 2.3 分析脚本

- `barrier_wait.py` / `rank_sweep.py` 改为按每个 rank 的 torch trace 解析 worker
  PID，不再假设 PID 连续；
- 脚本会动态标出 allreduce root rank。root 的 sync duration 天然接近 0，不能用它
  判断 barrier 等待；
- `host_stall.py` 保留 per-rank PID 映射。

## 3. 功能正确性

使用 `#07=fast`、`#19=fast`、`#09=native` 的完整矩阵结果如下；加入 prefill 门禁后，
prefill 仍然走同一 native #09 路径，因此该矩阵的 prefill 结论不变。

| 场景 | 结果 |
|---|---|
| 单请求非流式 | 200 OK，48 completion tokens |
| 4 并发 | 4/4 成功，wall 14.669 s |
| 128-token streaming decode | TTFT 0.266 s，total 22.792 s，约 5.64 tok/s |
| 16K chunked prefill | prompt 16038，TTFT 97.215 s，正常 |
| 32K 长上下文 | prompt 32037，TTFT 194.745 s，正常 |
| prefill→decode 边界 | 3 个 streaming case 均非空，usage 正常 |

完整结果：

- `log/validation_2026-10-22/functional_matrix_slot19.json`
- `log/validation_2026-10-22/functional_matrix_slot19.log`

## 4. 快路径命中

### 4.1 早先短请求

一个短请求产生 1 个 prefill step + 1 个 decode step，8 个 rank 完全一致：

- `custom_kpool_slot_mapping`：2 次/rank（每 step 一次）；
- `custom_kpool_append_compress`：22 次/rank（2 step × 11 indexer 层）；
- `custom_vllm_kpool_select_topk`：22 次/rank。

说明 #07/#09/#19 均进入真实模型路径，没有 dtype/contiguous/ABI 静默回退。

### 4.2 门禁后的 4K 唯一请求

prompt 3854 tokens，实际产生 2 个 prefill chunk + 1 个 decode step。请求前后差分
（8 rank 合计，折算每 rank）：

| kernel | 总增量 | 每 rank | 结论 |
|---|---:|---:|---|
| `custom_kpool_slot_mapping` (#19) | +24 | +3 | 3 个 step 各一次 |
| `custom_kpool_append_compress` (#07) | +88 | +11 | 仅 decode 一步 |
| `custom_vllm_kpool_select_topk` (#09) | +88 | +11 | 仅 decode 一步 |
| `custom_kpool_compress` | +176 | +22 | 2 个 prefill chunk × 11 层 |
| `custom_kpool_mqa_logits` | +176 | +22 | 2 个 prefill chunk × 11 层 |
| `custom_vllm_kpool_expand_topk` | +339152 | +42394 | native prefill 展开 |

这组数据证明：prefill 没有再进入 #09 fast，decode 仍保留 #07/#09 fast。

实测：

- 4K 唯一请求（带 kernel-log）：**29.911 s**，200 OK；
- 8K 唯一请求（带 kernel-log）：**58.156 s**，200 OK；
- 对照：fast-prefill 时 4K 约 98.7 s，8K 超时/EngineDead。

证据：

- `log/validation_2026-10-22/gate_4k_unique_request.json`
- `log/validation_2026-10-22/gate_4k_unique_before.txt`
- `log/validation_2026-10-22/gate_4k_unique_after.txt`
- `log/validation_2026-10-22/gate_8k_unique_request.json`
- `log/validation_2026-10-22/gate_8k_unique_response.txt`

## 5. 性能指标

profiler 窗口来自
`log/validation_2026-10-22/profile_allfast/`，包含 1 个 22-token prefill 和 3 个
decode step。除特别说明外，取 decode step2（span 140.6 ms，rank0）。

| 指标 | 旧基线 | 本轮结果 | 结论 |
|---|---:|---:|---|
| M1 `_fused_operand` | 204 次/step | **0** | 达标 |
| M2 H2D | 约 70 次小拷贝 | 44 次显式 H2D；另有 336 次 runtime host-device async | 次数未到 1，耗时已很小 |
| M3 D2H | 两处长阻塞，约 223 ms/step | 40 次 `.tolist()` 合计 0.2 ms；4 次 copy-back 合计 0.427 ms；无 >0.5 ms 阻塞 D2H | 长阻塞消除，次数未到 1 |
| M4 kernel 数 | 3888 | **1982** | -49.0% |
| M5 设备空洞 | 389.5 ms | **29.1 ms**，1937 holes，mean 15 µs，max 0.695 ms | -92.5% |
| M6 barrier arrival spread | mean 2.534 ms，sum 220.5 ms | **mean 1.531 ms，max 10.86 ms，sum 90.4 ms** | -59.0%，但仍未到 ≤20 ms |
| M7 有效设备计算 | 94.3–95.9 ms | 单请求 decode：device busy union 110.6 ms，其中 sync 88.2 ms、有效 22.4 ms | 旧数字与本轮 batch/workload 不同，不能直接定稿 |
| M8 decode 步长 | profiled 602.6 ms | profiled 140.6 ms；无 profiler **96.9 ms** | 达标，且优于 150–200 ms 目标 |

补充：

- M1 对应目标已明确完成；trace 中仍有 `aten::to` 767、`aten::copy_` 568、
  `aten::cat` 11、`aten::contiguous` 1 次/decode step，其中多数是 GEMM 输出/
  dtype 转换，不应与 `_fused_operand` 物化混为一谈；
- M2 的 44 次是显式 `handle_h2d_copy`，336 次是 runtime 的
  `driver.CopyHostDeviceAsync` 事务；后者包含内部参数上传，不能简单当作模型层
  metadata H2D 次数；
- M3 的 208 次 `aten::item` / `_local_scalar_dense` 不再对应 >0.5 ms 的阻塞 D2H；
  采样和 host 标量读取仍是后续可优化项；
- M6 的 root rank 是 rank3。其 sync duration 约 2 µs 是正常的“root 放行”行为，
  但 root 的到达时间仍在关键路径上；90.4 ms 是当前最大单项瓶颈。

分析产物：

- `log/validation_2026-10-22/profile_metrics/barrier_wait_step2_fixed.txt`
- `log/validation_2026-10-22/profile_metrics/rank_sweep_fixed.txt`
- `log/validation_2026-10-22/profile_metrics/host_stall_step2.txt`
- `log/validation_2026-10-22/profile_metrics/stream_union.txt`

## 6. 干净路径 decode

无 profiler、无 kernel-log，单请求 64 个输出 token（`ignore_eos=true`）：

| 指标 | 结果 |
|---|---:|
| TTFT | 0.341 s |
| total | 6.448 s |
| decode window | 6.107 s |
| decode throughput | **10.32 tok/s** |
| 平均 token 间隔 | **96.93 ms** |
| p50 / p90 | 96.35 / 102.91 ms |
| min / max | 89.58 / 131.09 ms |

结果：

- `log/validation_2026-10-22/clean_decode64.json`
- `log/validation_2026-10-22/service_clean_perf.log`

## 7. 结论与后续

1. 功能矩阵、16K/32K 长上下文、并发和 prefill→decode 边界均通过；
2. #07/#09/#19 在真实服务路径命中；本报告采集时 prefill 通过形状门禁回退
   native，该门禁随后已拆除；
3. M4、M5、M8 已完成主要目标；M8 干净路径约 96.9 ms/step；
4. 发布前剩余的主要性能项是 **M6 barrier root-critical 等待**，其次是 M2/M3
   的调用次数和标量读取；
5. 16K/32K 完整矩阵是在 #09=native 下采集的；拆除门禁后 prefill 也走 #09，
   这些长上下文数字不再代表最终开关，发布前必须用
   `DLC_KPOOL_SELECT_TOPK_OP=fast` 重跑 16K/32K smoke。

