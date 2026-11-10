# GLM-5.3-Flash Graph 捕获验证（2026-10-22）

> 2026-10-26 补充：本文的历史测试未显式设置 `DLC_SYN_GRAPH_CACHE`，Synapse
> 按默认值 `0` 运行，即未启用 kernel-batch cached replay。生产配置现已设为
> `DLC_SYN_GRAPH_CACHE=1`。同日已完成 cache=0/cache=1 单并发复测：decode gap
> 降低约 9%–10%，TTFT 基本不变；本文较早的 Graph/eager 数字仍作为未启用缓存时的
> 基线。

## 结论

Graph 路径已可启用，当前采用 `FULL_DECODE_ONLY`：decode 使用 FULL graph，
prefill/chunked prefill 保持 eager。8 卡 TP=8 环境下已完成 1、2、8、16、24、32
并发验证，32-token 固定输入下 graph 与 eager 输出逐字一致。

## 启用方式

```bash
MAX_MODEL_LEN=4096 PORT=18005 GPU_MEMORY_UTILIZATION=0.80 \
bash work-docs/glm53flash/vllm-cl/docs/glm5_flash/start_glm5_flash_vllm_cl.sh \
  --graph --max-num-seqs 32 --max-num-batched-tokens 1024
```

`--graph` 会移除 `--enforce-eager`，并追加：

```text
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
--kernel-config '{"enable_jit_warmup": false}'
```

默认（不加 `--graph`）仍为 eager。

## 本次修复

### 1. Graph wrapper 路径

`vllm_cl/platform.py` 的 `get_static_graph_wrapper_cls()` 由不存在的
`vllm.compilation.dlc_graph.CUDAGraphWrapper` 改为
`vllm_cl.compilation.dlc_graph.CUDAGraphWrapper`。

### 2. Graph profiling 的 KV cache groups

`vllm_cl/patch/patch_kv_cache_utils.py` 中，`_init_minimal_kv_cache_for_profiling()`
原先直接调用上游 `get_kv_cache_groups`，绕过了 DLC GLM-5 grouping，导致 profiling
阶段出现 Mamba page 大于 MLA page。现已保留上游函数别名，并将模块级
`kv_cache_utils.get_kv_cache_groups` 替换为 DLC 版本。

### 3. KDA padded-decode 的 query_start_loc

首个 8 并发失败点：

```text
dlc_kda_recurrent_decode ... query_start_loc must cover [0, N]
DLC error: synErrorLaunchFailure
at::dlc::DLCGraph::replay(bool)
```

根因不是 KPool graph，而是 FULL graph 的 token padding：

* 实际只调度 6 个 decode token，但 graph bucket 为 8；
* `q` 被补齐到 8 行，state_indices 尾部为 `NULL`；
* GDN metadata 原先把 padding 的 `query_start_loc` 都填成 6，且模型侧按
  `num_decodes + 1` 截断，kernel 看到最后一个边界 6 != padded N=8。

修复分两处：

* `vllm/v1/attention/backends/gdn_attn.py`：识别真实 1-token decode，把剩余
  PAD token 逐个分配给尾部 NULL state 槽。6 个真实请求 + 2 个 padding 时，
  boundary 变为 `[0,1,2,3,4,5,6,7,8]`，state indices 为
  `[s0..s5,-1,-1]`。
* `vllm/models/glm5next/common/kda.py`：不再把 `cu_seqlens` 截成
  `num_decodes + 1`，直接传完整 persistent buffer，符合 #04b ABI。

新增离线 CPU 回归：

```text
tests/v1/attention/test_gdn_metadata_builder.py::test_full_cudagraph_decode_metadata_covers_padded_tokens
```

## 验证矩阵

| 场景 | 配置 | 结果 |
|---|---|---|
| Decode graph 捕获 | `max-num-seqs=8` | 捕获 4 个 bucket，约 2 s |
| 1 / 2 并发 | `max-num-seqs=8` | HTTP 200 |
| 8 并发 | `max-num-seqs=8` | HTTP 200，8/8 |
| 长 prompt chunked prefill | 1495 prompt tokens | HTTP 200，prefill→decode 正常 |
| 混合 prefill/decode | 4 长 + 4 短并发 | 8/8 HTTP 200 |
| 16 并发 | `max-num-seqs=32` | 16/16 HTTP 200 |
| 24 并发（非 2 次幂 bucket） | `max-num-seqs=32` | 24/24 HTTP 200 |
| 32 并发 | `max-num-seqs=32` | 32/32 HTTP 200 |
| Graph vs eager | 相同 prompt，`temperature=0`，32 tokens | completion 逐字一致 |

相关日志：

```text
/tmp/glm5_graph_smoke6.log     # max-num-seqs=8
/tmp/glm5_graph_smoke7.log     # max-num-seqs=32
/tmp/glm5_eager_compare.log    # eager 基线
```

日志中未再出现 `query_start_loc must cover`、`synErrorLaunchFailure`、
`Assertion on XYS` 或 worker died。

## 当前边界

* 仅 decode 走 graph；prefill/chunked prefill 仍是 eager。
* DLC 当前把 `torch.compile` 做成直通实现，piecewise graph 暂无收益，因此使用
  `FULL_DECODE_ONLY`。
* graph 模式必须关闭 JIT warmup，否则模型构造会导入未安装的 NVIDIA `tilelang`。

## Graph / eager 性能量化

### 测量口径

两种模式使用完全相同的服务和请求，顺序为 graph 跑完并优雅停止，再启动 eager 重跑：

```text
TP=8, max_model_len=4096, max_num_batched_tokens=4096,
max_num_seqs=64, gpu_memory_utilization=0.80,
block_size=256, prefix caching 开启, 无 profiler / 无 kernel-log
```

客户端使用流式请求、`temperature=0`、`ignore_eos=true`，每档重复 3 次取中位数；
64 并发档额外重复 5 次复核。主指标：

* `TTFT`：首 token 到达时间；
* `gap_p50`：相邻流式 token 间隔的中位数，代表单请求 decode 步长；
* `aggregate_tok_s`：总 completion tokens / 并发墙钟。

复现脚本：`probes/bench_graph_vs_eager.py`。

### 结果

| 场景 | 并发 | graph gap p50 | eager gap p50 | graph 吞吐 | eager 吞吐 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 49-token prompt + 96 out | 1 | 67.64 ms | 110.27 ms | 14.11 tok/s | 8.83 tok/s | graph 快 39%，吞吐 +60% |
| 49-token prompt + 96 out | 8 | 103.49 ms | 112.86 ms | 65.38 tok/s | 59.95 tok/s | graph 快 8%，吞吐 +9% |
| 49-token prompt + 64 out | 32 | 222.04 ms | 222.88 ms | 100.08 tok/s | 100.55 tok/s | 基本持平 |
| 49-token prompt + 64 out | 64 | 277.89 ms | 263.59 ms | 92.10 tok/s | 99.12 tok/s | graph 慢约 5%（见下方归因） |
| 860-token prompt + 64 out | 1 | 73.34 ms | 96.03 ms | 11.27 tok/s | 8.99 tok/s | graph 快 24% |
| 860-token prompt + 64 out | 8 | 148.58 ms | 144.84 ms | 30.46 tok/s | 31.22 tok/s | 基本持平 |
| 1217-token prompt + 8 out | 1 | 83.90 ms | 98.70 ms | 2.60 tok/s | 2.50 tok/s | TTFT 2.502 s vs 2.498 s，prefill 无损 |

TTFT 在全部可对比场景保持一致（差异 <3%），说明 `FULL_DECODE_ONLY` 没有影响
prefill / chunked prefill 路径。

### Synapse graph replay cache 单并发复测（2026-10-26）

固定配置：TP=8、`max_model_len=4096`、`max_num_batched_tokens=4096`、
`max_num_seqs=64`、`gpu_memory_utilization=0.80`、`block_size=256`、
`FULL_DECODE_ONLY`、`enable_jit_warmup=false`。客户端 `temperature=0`、
`ignore_eos=true`、流式返回；每档重复 3 次取中位数。两组服务仅
`DLC_SYN_GRAPH_CACHE` 不同。

| 场景 | cache=0 gap p50 | cache=1 gap p50 | gap 变化 | cache=0 tok/s | cache=1 tok/s | 吞吐变化 | TTFT 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 49-token prompt + 96 out | 63.78 ms | 57.58 ms | -9.7% | 14.90 | 16.53 | +10.9% | 321 ms → 316 ms |
| 860-token prompt + 64 out | 74.42 ms | 67.61 ms | -9.2% | 11.13 | 12.13 | +8.9% | 1016 ms → 1009 ms |
| 1217-token prompt + 8 out | 80.13 ms | 72.31 ms | -9.8% | 2.61 | 2.67 | +2.3% | 2493 ms → 2486 ms |

结论：cache=1 的收益集中在 graph replay 的 decode 下发路径，三档 `gap_p50`
均下降约 9%–10%；prefill TTFT 差异小于 1%，长 prompt 场景的总吞吐主要受 prefill
时间支配，因此只提升约 2.3%。两组测试均出现一次性的首次长 prompt 冷路径开销
（860-token 首次 TTFT 约 7.44 s，1217-token 首次约 5.39 s），后续重复稳定，
正式压测前应先预热各输入长度。原始结果保存在
`/root/workspace/log/glm5_graph_cache0_single.json` 和
`/root/workspace/log/glm5_graph_cache1_single.json`。

### 64 并发反向差异归因

64 并发并不是 64 个请求同时 decode。服务日志显示实际调度为：

```text
Running: 38 reqs, Waiting: 26 reqs, DLC KV cache usage: 99.7%
```

FULL graph 的 capture sizes 为
`[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]`。38 个真实请求会选择 40 的 bucket，
多算 `40 - 38 = 2` 个 padding 槽；若该 batch 已进入计算受限区，理论额外成本
`2 / 38 ≈ 5.3%`，与实测 token 间隔高约 4.5%、聚合吞吐低约 7% 吻合。

32 并发正好命中 32 bucket，无 padding，因此 graph / eager 基本持平；单请求和 8
并发则因为 graph 消除了大量 host 下发开销，收益明显。这个结果说明：

1. graph 的收益主要来自小/中 batch 的 host 开销消除；
2. 大 batch 已接近计算受限，收益被摊薄；
3. 非标准并发数可能因向上取 bucket 产生 padding 成本，后续可评估增加
   `38/39` 等业务热点 bucket，或结合 profiler 确认是否值得。

### 产物

```text
/tmp/glm5_graph_bench.json       # graph 主矩阵
/tmp/glm5_graph_bench.log
/tmp/glm5_eager_bench.json       # eager 主矩阵
/tmp/glm5_eager_bench.log
/tmp/glm5_graph_c64_recheck.json # 64 并发 5 次复核
/tmp/glm5_graph_c64_recheck.log
/tmp/glm5_graph_perf2.log        # graph 64 并发服务日志（含 38 running + 26 waiting）
```

### 结论

* 单请求 decode 的 p50 步长由约 110 ms 降到 68 ms，是当前 graph 的主要收益；
* 8 并发有约 9% 聚合吞吐收益；
* 32 并发无收益也无回退；
* 64 并发因 `38 -> 40` bucket padding 约回退 5%，不是功能问题；
* prefill / 长 prompt TTFT 保持不变，发布配置可保持 `FULL_DECODE_ONLY`。
