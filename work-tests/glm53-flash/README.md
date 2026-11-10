# GLM-5.3-Flash 最终性能测试

`run_final_vllm_bench.py` 是纯客户端压测脚本：服务必须先由人工启动，脚本只负责健康检查、运行 `vllm bench serve`、解析 JSON、聚合重复结果并生成 Markdown。它不会启动、重启或修改服务，也不会替换 KPool 算子。

## 1. 启动服务

按服务指南推荐的 4K 生产配置启动 graph 服务。`start_glm5_flash_vllm_cl.sh` 默认开启 `FULL_DECODE_ONLY` graph 和 `DLC_SYN_GRAPH_CACHE=1`；额外参数会透传给 `vllm serve`：

```bash
cd /root/workspace/work-docs/glm53flash/vllm-cl/docs/glm5_flash
PORT=18005 \
MAX_MODEL_LEN=4096 \
GPU_MEMORY_UTILIZATION=0.80 \
./start_glm5_flash_vllm_cl.sh \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64
```

确认服务就绪：

```bash
curl -fsS http://127.0.0.1:18005/health
curl -fsS http://127.0.0.1:18005/v1/models
```

## 2. 运行最终测试

默认连接 `18005`，自动从 `/v1/models` 读取模型名和实际 `max_model_len`：

```bash
cd /root/workspace
/root/workspace/.venv/bin/python \
  work-tests/glm53-flash/run_final_vllm_bench.py
```

如果服务端口不是 `18005`：

```bash
BASE_URL=http://127.0.0.1:8000 \
/root/workspace/.venv/bin/python \
  work-tests/glm53-flash/run_final_vllm_bench.py
```

也可以覆盖模型名或结果文档路径：

```bash
/root/workspace/.venv/bin/python \
  work-tests/glm53-flash/run_final_vllm_bench.py \
  --model GLM-5.3-Flash-Mix \
  --doc /root/workspace/work-docs/glm53flash/GLM-5.3-Flash_最终性能结果_2026-11-02.md
```

先用 dry-run 检查命令矩阵：

```bash
/root/workspace/.venv/bin/python \
  work-tests/glm53-flash/run_final_vllm_bench.py --dry-run
```

## 3. 默认矩阵

默认 `--matrix core` 覆盖以下场景；每个场景首轮先跑 1 次不计时 warmup，正式重复 3 次，表中取跨 run 中位数：

| case | Prompt/Output | 请求数 | 并发 | 用途 |
|---|---:|---:|---:|---|
| `short_decode_c1` | 49 / 96 | 1 | 1 | 单请求 decode |
| `short_decode_c8` | 49 / 96 | 8 | 8 | 8 并发 decode |
| `short_decode_c32` | 49 / 64 | 32 | 32 | 32 并发 decode |
| `short_decode_c64` | 49 / 64 | 64 | 64 | 64 并发 decode |
| `medium_decode_c1` | 860 / 64 | 1 | 1 | 中等 prompt 单请求 |
| `medium_decode_c8` | 860 / 64 | 8 | 8 | 中等 prompt 8 并发 |
| `ttft_1217_c1` | 1217 / 8 | 1 | 1 | TTFT |
| `prefill_4k_c1` | 3610 / 8 | 1 | 1 | 4K prefill / 长上下文 |
| `guide_baseline_c8` | 512 / 64 | 16 | 8 | 服务指南随机基线 |

`--matrix extended` 额外加入 8K/16K/32K TTFT；脚本会按服务实际 `max_model_len` 自动跳过超限场景。只有确认服务上下文容量足够时才使用：

```bash
/root/workspace/.venv/bin/python \
  work-tests/glm53-flash/run_final_vllm_bench.py \
  --matrix extended
```

单独指定场景、减少重复次数或增加单次超时：

```bash
/root/workspace/.venv/bin/python \
  work-tests/glm53-flash/run_final_vllm_bench.py \
  --cases short_decode_c1,prefill_4k_c1 \
  --repeats 1 --timeout-sec 3600
```

## 4. 产物和口径

- 原始 JSON、日志和 `summary.json`：`work-tests/glm53-flash/results/<timestamp>/`
- 默认 Markdown：`work-docs/glm53flash/GLM-5.3-Flash_最终性能结果_<date>.md`
- 指标：TTFT、TPOT、ITL、E2EL、输出吞吐、请求吞吐、失败请求数
- 客户端命令固定 `--backend openai-chat`、`random` 数据集、`--temperature 0`、`--ignore-eos`、固定 seed
- 本测试是当前发布路径的实测值，**不包含 KPool 优化后的估算值**

脚本会在每个 case 前后检查 `/health`。若服务退出、graph 被检测为 eager、或检测到 `DLC_SYN_GRAPH_CACHE=0`，默认直接停止，避免生成口径错误的结果表。若机器无法读取 `/proc` 导致服务参数无法探测，脚本会继续运行但把相应配置标为“未采集”。
