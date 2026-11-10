# Qwen3 FP8 KV cache 低利用率容量与精度验收

日期：2026-10-27

## 1. 目的与范围

本轮不把 decode 性能作为验收门槛，优先确认以下三点：

1. 相同显存上限下，FP8 KV cache 能否容纳更多 block、token 和长上下文请求。
2. Qwen3-32B-Mix 的短请求输出是否与 BF16 KV cache 有实质差异。
3. 真实 8192-token prompt 下，跨多个物理页的写入、gather、反量化是否保持正确。

测试分支为 `vllm-cl:fp8-kvcache-main`。只修改 `vllm-cl`，不修改主 `vllm` 仓库。

## 2. 配置

两个实例除 KV cache 类型外参数完全一致：

```text
模型                 /data/models/Qwen3-32B-Mix
TP                   4
block size           256
max model len        40960
max batched tokens   4096
max sequences        16
GPU memory util      0.25
dtype                bfloat16
attention            eager reference path
prefix caching       disabled
```

FP8 使用 `--kv-cache-dtype fp8_per_token_head`；BF16 使用
`--kv-cache-dtype auto`。两者均从 `/root` 启动，避免 `/tmp/vllm` 的 Python
namespace 污染。

## 3. 容量结果

| cache | available KV memory | blocks | tokens | 40960-token 并发上限 |
|---|---:|---:|---:|---:|
| BF16 | 11.12 GiB | 711 | 182,016 | 4.44x |
| FP8 values + FP32 scales | 11.12 GiB | 1,380 | 353,280 | 8.62x |

FP8/BF16 block 容量比为 `1.941x`，与理论页大小比一致：

```text
BF16 page = 256 * 2 * 2 * 128 * 2 = 262,144 B
FP8 page  = 256 * 512 + 256 * 16 = 135,168 B
ratio     = 1.9394x
```

在相同 0.25 显存上限下，FP8 多容纳 `669 blocks / 171,264 tokens`。

## 4. 短请求精度

测试包含 8 个固定 prompt，覆盖事实、算术、中文、代码和简短推理；每项用
temperature 0 生成 32 token。另对 3 段中英文文本计算 prompt-token NLL。

| 指标 | 结果 |
|---|---:|
| 32-token 序列完全一致 | 4/8 |
| 其余 4 项首个分叉位置 | 30、14、12、0 |
| 三段文本 NLL 差（FP8 - BF16） | -0.00354、-0.02152、-0.00319 |
| 平均绝对 logprob 偏差 | 0.068、0.050、0.057 |

未完全一致的序列中，多数在 12-30 token 后因极小的概率差发生正常贪心分叉；
唯一首 token 不同的算术 prompt 是 BF16 基线被训练语料模式带到无关网页文本，
FP8 输出为正确的 `40`。三段文本的 FP8 NLL 均没有变差。

原始报告：

- `work-tests/fp8_kvcache/reports/qwen3_accuracy_fp8_lowutil_20261027.json`
- `work-tests/fp8_kvcache/reports/qwen3_accuracy_bf16_lowutil_20261027.json`
- `work-tests/fp8_kvcache/reports/qwen3_accuracy_fp8_vs_bf16_lowutil_20261027.json`

## 5. 8192-token 精度

使用相同 tokenizer 和相同 8192 个 token ID，在文本中部放入
`The secret access code is ORANGE-7319.`，末尾要求只回答 access code。

| cache | 输出 | prompt NLL | 耗时 |
|---|---|---:|---:|
| BF16 | `The secret access code is ORANGE-7319.` | 0.214328 | 4.56 s |
| FP8 | `The secret access code is ORANGE-7319.` | 0.211847 | 147.62 s |

FP8/BF16 的 16 个输出 token 完全一致。8191 个 prompt 位置的统计为：

- 平均 NLL 差：`-0.002480`
- 平均绝对 logprob 偏差：`0.004971`
- 最大单 token 绝对偏差：`0.995879`

这说明当前分离布局在跨 32 个物理页的 8192-token 场景下没有出现索引、
scale、dequant 或页边界错误。

原始报告：

- `work-tests/fp8_kvcache/reports/qwen3_long8192_bf16_lowutil_20261027.json`
- `work-tests/fp8_kvcache/reports/qwen3_long8192_fp8_lowutil_20261027.json`
- `work-tests/fp8_kvcache/reports/qwen3_long8192_fp8_vs_bf16_lowutil_20261027.json`

## 6. 慢路径根因与修正

此前 5740-block cache 的 5-token 微基准为 scatter 约 3.9 s、gather 约
0.55 s。进一步拆分后，512 B FP8 value copy 仅约 0.08 ms；慢的是在大
storage 上对 16 B scale 子区先做 `.view(torch.float32)` 再 copy。

现在 scale 全程按 uint8 原始字节搬运，写完后仍以 FP32 解释。5740-block
单层微基准从秒级降到：

```text
scatter  1.68 ms
gather   0.65 ms
```

8192-token 端到端 prefill/decode 仍需 147.62 s，这是 Python per-token
reference writer/reader 加 eager 逐请求 SDPA 的开销，不是容量本身造成的。
生产性能仍需 fused scatter/gather 和原生 FP8 attention。

## 7. 结论与限制

- 相同显存预算下，FP8 KV cache 提供 `1.941x` block/token 容量。
- 短上下文未观察到有意义的精度退化；8192-token needle 回答完全一致。
- 分离 values/scales 布局通过跨页正确性验证。
- 当前仍是 correctness/reference path，必须 `--enforce-eager`；没有覆盖
  graph、prefix caching、滑动窗口、TP8 一 KV head/rank 或高并发性能。
