# Kimi-K3-INT4 TP8 真实切片实测基线（2026-10-28）

## 环境

- 模型：`/data/models/Kimi-K3-INT4-Slice8-TP8`
- 24 层：1 dense + 23 MoE；18 KDA + 6 MLA；2 个 AttnRes block 边界
- TP8，8 x 64 GiB TYD
- eager，`max_model_len=128`，`max_num_batched_tokens=128`，`max_num_seqs=1`
- vLLM：`85fd014d5ff4`，`release-kimi-k3-int4`
- vllm-cl：`ab819da9a0f4` + K3 适配工作区改动
- PyTorch DLC：2.5.0，`9049914ec94a`
- DLC_Custom_Kernel：`55b9aef7e0cb`

## 启动

| 项目 | 结果 |
|---|---:|
| 25 个 safetensors shard 加载 | 373.49 s |
| 每 rank 权重显存 | 47.56 GiB |
| profile + KV cache + warmup | 17.69 s |
| 可用 KV cache | 8,429 tokens / 461 blocks |

profile dummy forward 中，native MoE 使用零输出快速路径；真实请求仍使用真实权重。

## 真实请求

| 请求 | prompt tokens | completion tokens | 端到端耗时 | 输出 |
|---|---:|---:|---:|---|
| `Hello` | 1 | 1 | 41.57 s | 有效 token |
| `Hello` | 1 | 2 | 25.00 s | `和监督展` |
| `Hello` | 1 | 2（重复） | 23.73 s | `和监督展`，与上次逐字一致 |
| `Hello` | 1 | 4 | 47.87 s | `和监督展 evacuating` |
| `The quick brown fox jumps over the lazy dog.` | 10 | 1 | 102.83 s | `inton` |

日志未出现 NaN、ERROR、DLC 队列超时或 `!` 输出。当前耗时由 native routed-expert
MoE 主导；expert 反量化 cache 会显著影响后续请求耗时。

## 说明

这是 24 层前缀切片的实测数据，不代表完整 93 层模型延迟或输出精度。它用于：

1. 证明 KDA/MLA/MoE/AttnRes 的真实权重路径可运行；
2. 为算子需求提供当前 native 回退的性能基线；
3. 作为 #01 MoE 算子 P0 排序的直接依据。

原始服务日志：`/tmp/kimi_real_start_log`（临时文件，重要结论已摘录到本文）。
