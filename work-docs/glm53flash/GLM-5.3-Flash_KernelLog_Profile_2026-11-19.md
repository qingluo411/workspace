# GLM-5.3-Flash 最新 kernel-log profile（2026-11-19）

> 配置：MTP `num_speculative_tokens=1`、prefix cache 开启、`FULL_DECODE_ONLY` graph、TP8。  
> 采集方式：`start_glm5_flash_vllm_cl.sh --kernel-log=...`；rank0 为 `syn_1408198.ansi`。  
> Workload：同一 4096-token prompt，输出 8 token；第一发冷请求，第二发热请求。  
> Kernel log 本身会显著增加执行时间，本文只比较 kernel cycle 和占比，不把请求耗时当作正常性能。

## 1. 请求窗口

| 请求 | cached_tokens | 请求耗时 | rank0 日志窗口 |
|---|---:|---:|---:|
| 冷请求 | 0 | 11.611 s | 51.34 MB |
| 同 prompt 热请求 | 3,584 | 2.707 s | 36.35 MB |

两次 greedy 输出均为：

```text
[196, 197, 198, 199, 200, 201, 202, 203]
```

## 2. rank0 总 cycle

按 `custom_dlc_kda_recurrent_decode` 切分，所有时间按 1400 MHz 折算：

| 阶段 | events | cycles | 折算时间 |
|---|---:|---:|---:|
| 冷请求 prefill | 16,216 | 14,659,155,612 | 10,470.83 ms |
| 冷请求 decode | 10,737 | 295,781,349 | 211.27 ms |
| 热请求 prefill | 8,723 | 2,368,163,438 | 1,691.55 ms |
| 热请求 decode | 10,737 | 303,998,256 | 217.14 ms |

冷/热 prefill 的 cycle 差距约 **6.19x**，与 3584-token prefix 命中一致；冷/热 decode
基本持平（约 `211 ms` 对 `217 ms`），说明本次 prefix 命中主要削减 prefill。

## 3. 冷请求 prefill 热点

| 算子 | calls | cycles | 占比 |
|---|---:|---:|---:|
| `custom_dlc_kda_chunk_prefill` | 34 | 4,378,145,359 | 29.87% |
| `custom_vllm_kpool_select_topk` | 12 | 2,888,739,573 | 19.71% |
| `custom_mhc_pre` | 90 | 2,603,010,428 | 17.76% |
| `custom_vllm_fused_moe_w4a16_col_major_g128_align` | 42 | 1,818,743,798 | 12.41% |
| `custom_sparse_attention_kv_cache_nope_bf16_h8_topk2176_n1048576` | 12 | 1,116,805,626 | 7.62% |
| `custom_mhc_post` | 90 | 574,855,176 | 3.92% |
| `custom_sync_all_machine` | 97 | 267,178,950 | 1.82% |
| `custom_allreduce_2x4tpu4root_blockcyclic_simple_bf16_rdma` | 94 | 177,020,603 | 1.21% |
| `custom_kda_causal_conv_prefill` | 34 | 138,680,142 | 0.95% |

冷 prefill 的前五项合计约 **87.4%**。KPool `select_topk` 仍是单算子第二大热点，
在 4096-token 冷 prefill 中占 **19.71%**。

## 4. 热请求 prefill 热点

| 算子 | calls | cycles | 占比 |
|---|---:|---:|---:|
| `custom_vllm_fused_moe_w4a16_col_major_g128_align` | 84 | 564,223,670 | 23.83% |
| `custom_dlc_kda_chunk_prefill` | 68 | 547,674,062 | 23.13% |
| `custom_vllm_kpool_select_topk` | 24 | 367,808,663 | 15.53% |
| `custom_mhc_pre` | 180 | 341,553,632 | 14.42% |
| `custom_sparse_attention_kv_cache_nope_bf16_h8_topk2176_n1048576` | 24 | 142,578,898 | 6.02% |
| `custom_allreduce_2x4tpu4root_simple_bf16_rdma` | 188 | 116,794,689 | 4.93% |
| `custom_mhc_post` | 180 | 76,199,020 | 3.22% |

热 prefill 中，KPool `select_topk` 占比降到 **15.53%**，但仍是第三大热点；
MoE 占比升高到 23.83%，主要因为 prefill 总 cycle 大幅缩小后固定权重 GEMM 占比上升。

## 5. decode 热点

冷请求 decode：

| 算子 | calls | cycles | 占比 |
|---|---:|---:|---:|
| `custom_mhc_pre` | 359 | 51,236,703 | 17.32% |
| `custom_vllm_fused_moe_w4a16_col_major_g128_align_decode` | 168 | 45,626,021 | 15.43% |
| `custom_allreduce_2x4tpu4root_simple_bf16_rdma` | 372 | 43,614,550 | 14.75% |
| `custom_mm_ah_8_rhsT_bf16_col_major` | 1,142 | 33,310,691 | 11.26% |
| `custom_dlc_kda_recurrent_decode` | 136 | 22,099,712 | 7.47% |
| `custom_reshape_offset` | 616 | 13,524,490 | 4.57% |
| `custom_reshape_offset_f32_off0` | 2,365 | 13,075,292 | 4.42% |
| `custom_vllm_kpool_select_topk` | 47 | 6,776,933 | 2.29% |

热请求 decode 与冷请求结构基本一致，总 cycle 约高 2.8%；主要热点仍是 mHC、
MoE decode、DLCCL allreduce、8-token matmul 和 KDA recurrent decode。两个
reshape/offset 算子合计约 **8.7%**，仍是值得继续压缩的布局开销。

## 6. 结论

1. prefix cache 命中 3584 token 后，prefill cycle 从 `14.66B` 降到 `2.37B`，约 **6.19x**；
   decode cycle 基本不变。
2. 冷 prefill 的首要热点是 KDA chunk prefill、KPool `select_topk`、mHC pre、W4A16 MoE
   和 sparse attention；其中 KPool 仍占 `19.71%`。
3. 热 prefill 中 KPool 仍占 `15.53%`，说明命中后剩余 prefill 的固定成本仍不可忽视。
4. decode 的主要成本集中在 mHC、MoE decode、DLCCL allreduce 和 KDA recurrent decode；
   reshape/offset 类算子合计约 8.7%，支持继续做布局融合或消除中间物化。
5. `custom_sync_all_machine` 是 kernel-log 采集时额外打开的集合通信同步算子；其 cycle
   应和 `custom_allreduce_*` 一起看，不能单独当作模型正常路径成本。

## 7. 原始数据

- rank0 kernel log：`/root/workspace/log/klog_mtp1_prefix_graph_20261119_v1/syn_1408198.ansi`
- 冷请求精确窗口与解析：`work-docs/glm53flash/vllm-cl/docs/glm5_flash/measurements/2026-11-19_kernel_log/cold_exact/`
- 热请求精确窗口与解析：`work-docs/glm53flash/vllm-cl/docs/glm5_flash/measurements/2026-11-19_kernel_log/hot_exact/`
- 请求/cache 元数据：`/root/workspace/log/klog_mtp1_prefix_graph_20261119_v1/windows_20261119.json`

解析命令：

```bash
python helper/tool.py cold_exact/prefill.ansi
python helper/tool.py cold_exact/decode.ansi
python helper/tool.py hot_exact/prefill.ansi
python helper/tool.py hot_exact/decode.ansi
```

解析诊断：冷 prefill/decode 为 `16,216/10,737` events，热 prefill/decode 为 `8,723/10,737`
events；`tool.py` 无 `name_mismatches`、`cycle_mismatches`，未匹配 launch 为 0。
