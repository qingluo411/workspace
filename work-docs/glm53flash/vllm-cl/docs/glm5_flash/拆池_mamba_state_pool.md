# DLC Mamba 状态池拆分（split pool）方案

> 2026-09-28。替代方案：#05（`dlc_kda_state_gather_scatter`）只覆盖 gather/scatter 的 64%，
> 剩下的 conv prefill/update 在共享页布局下同样慢 3~4 个数量级，只有拆池能治。
> 证据与微基准见 `给算子组_05_state_pool_index_perf.md`。

## 1. 问题：打包页让每个 state 都变成非连续张量

旧布局：每层一个 `[num_blocks, 1, 1, page_size_bytes]` int8 页，
`MambaBase.bind_kv_cache` 按字节偏移切出 conv / recurrent 两个 view：

```
page_size_bytes = 542720 = 18432(conv bf16 [3,3072]) + 524288(rec fp32 [8,128,128])
conv view dim0 stride = 542720 / 2 = 271360 元素   ≠ 9216   (自身紧凑 stride)
rec  view dim0 stride = 542720 / 4 = 135680 元素   ≠ 131072
```

两个 view 都 `contiguous=False`。DLC 后端对这类张量会掉出 DMA 路径、走 host 逐元素
搬运（实测等效带宽 ~0.3–0.5 GB/s，且耗时与**整个 storage** 相关，与实际动了几页无关）：

| 操作（仅动 8 页 / 4 MiB） | dense 连续池 | padded 页内 view | 放大 |
|---|---|---|---|
| `index_select`+clone | 0.181 ms | 579.7 ms | 3196x |
| `index_copy_` | 1.544 ms | 1873.0 ms | 1213x |
| `dlc_kda_causal_conv_prefill` | 0.166 ms | 1396.1 ms | 8400x |
| `dlc_kda_causal_conv_update` | 0.259 ms | 1339.3 ms | 5200x |

模型内 profile（prefill 一步 179 s）：`scatter` 83.1 s + `conv_prefill` 65.0 s +
`gather` 24.8 s ≈ 173 s（96%）；decode 一步 120 s：`conv_update` 62.4 s +
`kda_recurrent` 52.9 s ≈ 96%。

## 2. 方案：每个 state 一个独立连续池

```
旧:  kv_caches[layer] = int8 [P, 1, 1, 542720]           # 打包页
新:  kv_caches[layer] = MambaStatePool([bf16 [P, 3, 3072],   # conv，stride0 = 9216
                                        fp32 [P, 8, 128, 128]]) # rec，stride0 = 131072
```

- **总字节不变**（`page_size_padded is None` 时严格相等），block id 语义不变，
  每页大小仍是 512B 倍数（conv 18432、rec 524288），没有引入新的对齐问题。
- 层内仍是一个「每层一个 cache 条目」，只是条目从张量变成 2 个张量的容器；
  模型侧 `(conv_state, recurrent_state) = self.kv_cache` 不变。
- KV cache 分组/计费仍按 `MambaSpec.page_size_bytes`，`num_blocks` 不变。

## 3. 改动点

| 文件 | 改动 |
|---|---|
| `vllm_cl/worker/mamba_cache.py` | 新增 `MambaStatePool` 与拆池分配；`DLC_MAMBA_SPLIT_POOL=0` 回退旧打包页；带 page padding 或 state 页非 512B 倍数时自动回退 |
| `vllm_cl/patch/patch_mamba_state_pool.py`（新） | 补丁 `MambaBase.bind_kv_cache`：`MambaStatePool` 直接安装（校验形状/dtype/连续性），打包页仍走上游按字节切分 |
| `vllm_cl/patch/__init__.py` | 注册上面的补丁 |
| `vllm_cl/patch/patch_kv_block_zeroer.py` | 按 storage 是否「大于自身」判断是否为打包页；拆池后各 state 走通用分支按自身页大小登记（#18 的整页清零算子因此也能覆盖 KDA state） |
| `vllm_cl/worker/dlc_model_runner.py` | `init_fp8_kv_scales` 支持容器型条目 |
| `vllm_cl/worker/dlc_model_runner_v2.py` | KV cache 分配日志不再假设第一个条目是张量 |

开关：`DLC_MAMBA_SPLIT_POOL`（默认 `1` 开启；`0`/`false`/`off`/`native` 回退打包页）。

## 4. 其它说明

- `runner_kv_caches` 现在会存入容器对象。上游签名本来就是
  `list[torch.Tensor | list[torch.Tensor]]`；DLC 两个 runner 对 `self.kv_caches`
  的使用只有「退出时置空」与（量化 KV 时的）`zero_()`，均已适配。
- DLC runner 不走 `copy_kv_cache_blocks_inplace`（`mamba_cache_mode="none"`，没有
  `kv_cache_block_copies`），因此容器条目不会被误当张量切片。
- 只有 KV connector（`has_kv_transfer_group()`）会把 `kv_caches` 交给
  `register_kv_caches`，该场景本阶段不使用。
