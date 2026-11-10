# KPool `#09 dlc_vllm_kpool_select_topk` 性能量化（2026-10-22）

## 1. 结论

本轮在 TYD 单卡上隔离测量了 `#09 dlc_vllm_kpool_select_topk` 与当前 native
回退路径，得到两个独立结论：

1. **prefill 的交叉点约为每请求 180 个 query token（单 chunk）。**
   - 128 token 时 `#09` 约快 30%；
   - 180 token 附近基本持平；
   - 184–200 token 起 native 开始更快；
   - 256 token 时 `#09` 慢约 1.3 倍；
   - 1K 时慢约 4.6 倍；
   - 2K 时慢约 9.1 倍；
   - 4K 时慢约 17.8 倍。

2. **decode 不是无条件命中 `#09` 更快；交叉点取决于 batch 与已缓存上下文长度。**
   - 单请求 decode：约 448 token 上下文后 native 更快；
   - 32 请求并发 decode：约 800 token/请求后 native 更快；
   - 32 请求、2K 上下文时 `#09` 已慢约 2.3 倍；
   - 32 请求、32K 上下文时慢约 7.0 倍。

测量时的 dispatch 只看

```python
query.shape[0] == num_reqs
```

不足以覆盖长上下文 decode：该条件能挡住普通多 token prefill，但不能挡住
“每请求一个 token、上下文很长、batch 很大”的 decode。该门禁已按第 8 节的
最终决策拆除，prefill 和 decode 现在统一走 `#07 + #09`。

## 2. 测试范围

本文只测 **KPool indexer 的 selection 部分**：

- fast：一次 `torch.dlc_vllm_kpool_select_topk`；
- native：当前 `native_kpool_indexer_step` 的池打分/topk/expand 回退路径。

不计入：

- `#07 dlc_kpool_append_compress` 与 native 旧 compress 的差异；
- 模型其他层、MoE、Attention、KDA、集合通信；
- 采样和 D2H 输出。

模型共有 11 个 indexer 层。下文的单层耗时可粗略乘以 11，用于估算整模 selection
成本；但不要把这个乘积当成完整 decode step 或完整 TTFT。

## 3. 环境与代码状态

| 项目 | 值 |
|---|---|
| 日期 | 2026-10-22 |
| 芯片 | TYD，8 卡；测试使用 `dlc:1` |
| 驱动 | `Driver Version_v24.0.1` |
| 测试时状态 | 8 卡空闲，`dlc:1` 为 `Halted/PASS` |
| DLC_Custom_Kernel | `release-glm53flash` / `92a36d525` |
| pytorch | `release-glm53flash` / `ab7c0354ec` |
| vllm-cl | `release-glm53flash` / `f971a82` |
| 环境变量 | `DLC_SKIP_VERSION_CHECK=1`、`VLLM_PLUGINS=''` |

关键代码：

- fast 入口：`vllm-cl/vllm_cl/_custom_ops.py:1338`
- dispatch 门禁：`vllm-cl/vllm_cl/patch/models/kpool_indexer.py:689`
- native plan：`vllm-cl/vllm_cl/patch/models/kpool_indexer.py:893`
- native 池 gather：`vllm-cl/vllm_cl/patch/models/kpool_indexer.py:1279`
- native 逐行 topk：`vllm-cl/vllm_cl/patch/models/kpool_indexer.py:1310`
- native 逐行 expand：`vllm-cl/vllm_cl/patch/models/kpool_indexer.py:1316`
- `#09` kernel：`DLC_Custom_Kernel/dlc_kernels/vllm/vllm_kpool_select_topk.cpp:80`

## 4. 输入几何与测量方法

使用 GLM-5.3-Flash 生产几何：

| 参数 | 值 |
|---|---:|
| `index_n_heads` / `H` | 32 |
| `index_head_dim` / `D` | 128 |
| `pool_size` | 4 |
| `pools_per_block` | 64 |
| `topk_tokens` | 2048 |
| 输出宽度 | 2176 |
| `query` / `pool_cache` dtype | BF16 |
| `head_weights` dtype | FP32（生产路径） |
| `block_table` / metadata dtype | int32 |

测量方式：

- 同一组输入分别运行 fast 和 native；
- 每个 case 预热 1–2 次；
- 取 2–7 次 wall time 中位数；
- 每次调用后执行 `torch.dlc.synchronize()`；
- 无 profiler、无 kernel log；
- 使用随机数据，检查过 `T=16/64/192/512` 以及 `R=32, ctx=1024` 的 fast/native
  输出逐元素完全一致。

## 5. Prefill 结果

单请求，所有 prompt token 在同一个 prefill chunk 中；`context = query tokens`。

| query tokens | #09 (ms) | native (ms) | #09/native | 更快 |
|---:|---:|---:|---:|---|
| 64 | 3.07 | 6.70 | 0.46 | #09 |
| 128 | 9.86 | 13.85 | 0.71 | #09 |
| 176 | 17.49 | 17.89 | 0.98 | #09 |
| 180 | 18.24 | 19.30 | 0.95 | 约持平 |
| 184 | 19.18 | 18.86 | 1.02 | native |
| 192 | 20.64 | 19.59 | 1.05 | native |
| 200 | 22.24 | 20.39 | 1.09 | native |
| 224 | 27.09 | 22.83 | 1.19 | native |
| 256 | 34.97 | 27.25 | 1.28 | native |
| 384 | 74.73 | 42.36 | 1.76 | native |
| 512 | 129.9 | 53.7 | 2.42 | native |
| 768 | 285.2 | 81.9 | 3.48 | native |
| 1K | 500.7 | 108.8 | 4.60 | native |
| 1.5K | 1112 | 160.1 | 6.95 | native |
| 2K | 1965 | 215.7 | 9.11 | native |
| 3K | 4384 | 324.8 | 13.50 | native |
| 4K | 7745 | 434.7 | 17.82 | native |

### 5.1 Prefill 结论

- **实际交叉点约 180–184 token/请求/chunk。**
- 180 token 内的优势也不大，176 token 仅约 2%；真正明显领先的区间是
  ≤128 token。
- 1K 以上已不适合使用当前 `#09` prefill。
- 若整模按 11 个 indexer 层估算，4K prefill 的 selection-only 时间为：
  - `#09`：约 `7.745 s × 11 = 85.2 s`；
  - native：约 `0.435 s × 11 = 4.8 s`；
  - 差值约 80 s。

这与 2026-10-22 整模实测中“4K fast-prefill 约 98.7 s、native 约 24.7 s”的
量级一致。

## 6. Decode 结果

### 6.1 单请求 decode

每请求 1 个 query token；`context` 为已缓存序列长度。

| context | #09 (ms) | native (ms) | #09/native | 更快 |
|---:|---:|---:|---:|---|
| 64 | 0.182 | 0.495 | 0.37 | #09 |
| 128 | 0.264 | 0.500 | 0.53 | #09 |
| 256 | 0.433 | 0.648 | 0.67 | #09 |
| 384 | 0.594 | 0.647 | 0.92 | #09 |
| 448 | 0.675 | 0.632 | 1.07 | native |
| 512 | 0.760 | 0.623 | 1.22 | native |
| 1K | 1.397 | 0.513 | 2.73 | native |
| 4K | 5.23 | 0.681 | 7.68 | native |
| 16K | 20.69 | 1.66 | 12.45 | native |

单请求 decode 的交叉点约在 **448 token 上下文**。

### 6.2 32 请求并发 decode

每请求 1 个 query token，32 个请求上下文长度相同。

| context/req | #09 (ms) | native (ms) | #09/native | 更快 |
|---:|---:|---:|---:|---|
| 512 | 11.05 | 16.42 | 0.67 | #09 |
| 768 | 16.38 | 17.18 | 0.95 | #09 |
| 832 | 17.68 | 17.13 | 1.03 | native |
| 1K | 21.41 | 16.90 | 1.27 | native |
| 2K | 42.09 | 18.56 | 2.27 | native |
| 4K | 82.53 | 22.91 | 3.60 | native |
| 8K | 163.7 | 32.6 | 5.03 | native |
| 16K | 325.8 | 52.9 | 6.15 | native |
| 32K | 650.1 | 92.3 | 7.04 | native |

32 请求并发时，交叉点约在 **800 token/请求**。

### 6.3 整模换算

按 11 个 indexer 层估算 selection-only 成本：

| 场景 | #09 | native | 额外成本 |
|---|---:|---:|---:|
| 单请求，4K context decode | 57.5 ms/step | 7.5 ms/step | +50.0 ms/step |
| 32 请求，2K context decode | 463 ms/step | 204 ms/step | +259 ms/step |
| 32 请求，32K context decode | 7151 ms/step | 1015 ms/step | +6.1 s/step |

这不是完整 decode step；它只说明：长上下文或大批量 decode 时，`#09` 的
selection 成本本身就可能成为主要瓶颈。

## 7. 原因分析

### 7.1 `#09` kernel 的并行结构

`vllm_kpool_select_topk.cpp` 主要结构是：

```text
for each query token T_i:                 // 两 XYS 仅按 T 切成两份
  for each visible pool p:                // 串行扫描全部可见池
    load pool[p]
    for each head h:
      for d in 128:
        dot(q[T_i,h,:], pool[p,:])
  sort all visible pools
  write expanded top-k + tail
```

问题在：

- 只按 token 数 `T` 在 XYS0/XYS1 之间切分；
- 每个 XYS 内部仍逐 token 串行；
- 每个 token 都重新扫描并加载所有可见 pool；
- 没有 query tile × pool tile；
- 没有相邻 query 复用同一批 pool 数据；
- `T=1` 的 decode 只会用到 `xys0`，`xys1` 空闲；
- 每行都按 2048 宽 top-k + tail 输出，仍有固定写回成本。

prefill 时每行的可见池数随位置增长，总池扫描量近似
`Σ_i floor(i / pool_size)`，因此 4K 相对 512 token 呈明显二次增长。

### 7.2 native 为什么在长上下文更快

native 回退不是“纯逐行点积”：

- 先用 `torch.dlc_kpool_mqa_logits` 对一个请求的所有 query row × visible pool
  做批量打分；
- 再逐行 `torch.topk`；
- 再逐行 `dlc_vllm_kpool_expand_topk`。

逐行 topk/expand 有固定 kernel 下发成本，所以在短序列时不如一次完成的 `#09`；
但长序列时批量打分明显优于 `#09` 的逐 token/逐池串行扫描。

## 8. fast-path 门禁的最终决策（2026-10-22 更新）

> 后续决策：由于该模型人员投入减少，不再继续优化 `#09` kernel。已移除
> `query.shape[0] == num_reqs` 的 decode-only 门禁；prefill 和 decode 都直接
> 走 `#07`（更新 tail/pool）加 `#09`（打分、top-k、展开）的融合算子。

拆除门禁后：

- 所有通过 ABI、dtype、contiguous 和输出宽度检查的 BF16 DLC batch 都走
  `#07 → #09`，包括多 token prefill 和 chunked prefill；
- 4K prefill 的 selection 层会承担约 17.8 倍的单层开销（11 层约多 80 s）；
- 32 请求、32K context 的 decode 会承担约 7 倍的单层开销；
- `DLC_KPOOL_SELECT_TOPK_OP=native` 或 `DLC_KPOOL_OP=native` 仍保留为逃生开关，
  可用于功能对照和紧急回退。

上文测得的 180/448/800 token 交叉点仍然有效，可作为将来若恢复 kernel 优化时
重新设计门禁的标定依据；当前发布路径不再使用这些阈值。

## 9. Kernel 优化方向

当前 `#09` 不应只做局部调优，建议按以下方向重写 prefill/长上下文变体：

1. query row tile × pool tile 并行，而不是只按 token 切两个 XYS；
2. 每个 query tile 共享一批已加载的 pool，减少 HBM 重读；
3. 批量计算 `[query_rows, visible_pools]` 分数，再按行做 top-k；
4. 允许 `T=1` 时按 pool tile 使用两个 XYS，避免 xys1 空闲；
5. 将 expand 与 top-k 融合，避免每行独立写回整个宽缓冲；
6. 对超宽输出按实际 `sequence_length` 限定写回范围；
7. 优化后用同一组 prefill/decode 数据重新测交叉点，再决定 fast gate。

## 10. 复现要点

最小复现参数：

```text
device=dlc:1
query=[T, 32, 128], bfloat16
head_weights=[T, 32], float32
pool_cache=[ceil(ceil(context/4)/64), 64, 128], bfloat16
block_table=[requests, ceil(ceil(context/4)/64)], int32
query_start_loc=int32 prefix sum of query lengths
seq_lens=[requests], int32
topk_tokens=2048
pool_size=4
out=[T, 2176], int32
```

fast：

```python
torch.dlc_vllm_kpool_select_topk(
    query, head_weights, pool_cache, block_table,
    query_start_loc, seq_lens,
    2048, 4, None, None, out=out,
)
```

native：

```text
按请求 gather pooled_keys
  -> torch.dlc_kpool_mqa_logits(query[start:stop], pooled_keys,
                                 head_weights[start:stop], valid_pools)
  -> torch.topk(scores[row, :visible], k=min(512, visible)).indices
  -> torch.dlc_vllm_kpool_expand_topk(...)
  -> out[row, :2051].copy_(expanded)
```

## 11. 注意事项

- 本文是单卡、单层、随机数据的隔离微基准；不同卡、时钟状态和后台负载会带来
  数个百分点的波动。
- 结论是 selection-only；整模仍受 MoE、Attention、KDA 和集合通信影响。
- prefill 的 180 token 阈值是“每请求、每 chunk”的 query token 数；chunked
  prefill 时按实际 chunk 内每请求 token 数理解。
- decode 的 448/800 token 是 R=1 与 R=32 的实测锚点，不应直接外推到所有
  batch 组合；生产门禁最好按 batch 和总可见池数动态估算。
