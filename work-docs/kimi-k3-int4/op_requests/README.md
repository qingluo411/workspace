# Kimi-K3-INT4 DLC 算子需求（Issue 草稿）

本目录保存 Kimi-K3-INT4 在 DLC 上从“PyTorch native 回退”迁移到 DLC 融合算子的
Issue 草稿。每份正文按
`DLC_Custom_Kernel/.github/ISSUE_TEMPLATE/operator_request.yml` 的 14 个字段顺序编写，
可直接用于填写 GitHub「算子需求」表单；本次只生成本地文档，尚未提交 issue。

## 当前基线

- 真实模型：`/data/models/Kimi-K3-INT4-Slice8-TP8`，24 层 TP8 切片。
- 真实权重已加载：每 rank 47.56 GiB；单请求 prefill/decode 已跑通。
- 当前主要瓶颈不是功能，而是 native routed-expert MoE。真实 24 层切片上：
  - `Hello` 1-token：41.57 s；
  - `Hello` 2-token：25.00 s；
  - `Hello` 4-token：47.87 s；
  - 10-token prompt + 1-token decode：102.83 s。
- profile dummy forward 已临时绕过 native MoE，否则 128-token profiling 会触发 DLC
  600 秒队列超时。

## 链式数据流审核结论

六算子覆盖了主要 native 计算，但**不能仅凭实现这六个算子就得到严格的全路径零拷贝单 graph**：
MLA prefill 仍有 `v.contiguous()`，KDA decode 仍有两处后处理拷贝，KDA/MLA 当前还处于
`eager_break_during_capture` 段。完整审核和需要补齐的接入动作见
[`00b_链式数据流审核_2026-10-28.md`](00b_链式数据流审核_2026-10-28.md)。

按该结论推进的可行目标改造清单、文件级修改点和验收门槛见
[`00c_可行目标改造方案_2026-10-28.md`](00c_可行目标改造方案_2026-10-28.md)。

## 本批需求

| # | 算子 | 类型 | 优先级 | 主要收益 |
|---|---|---|---|---|
| 01 | [`dlc_vllm_fused_moe_w4a16_prerouted_col_major`](01_dlc_vllm_fused_moe_w4a16_prerouted_col_major.md) | 扩展现有 W4A16 MoE | **P0** | 替换逐 token/逐 expert native 循环，是当前第一性能瓶颈 |
| 02 | [`dlc_kimi_k3_attn_res`](02_dlc_kimi_k3_attn_res.md) | 新增 | P1 | 每层 2 次 residual softmax mixing，当前 24 层前向共 48 次 native 调用 |
| 03 | [`dlc_kimi_k3_mla_prefill_epilogue_bf16`](03_dlc_kimi_k3_mla_prefill_epilogue_bf16.md) | 新增 | P1 | 合并 q/k RoPE、K concat、latent cache 写入 |
| 04 | [`dlc_kimi_k3_mla_decode_epilogue_bf16`](04_dlc_kimi_k3_mla_decode_epilogue_bf16.md) | 新增 | P1 | 合并 decode q concat、RoPE 和 latent cache 写入 |
| 05 | [`dlc_situ_and_mul`](05_dlc_situ_and_mul.md) | 新增 | P1 | 覆盖 dense MLP、shared experts 和 routed experts 的 SiTU-GLU |
| 06 | [`dlc_kimi_k3_kda_packed_decode`](06_dlc_kimi_k3_kda_packed_decode.md) | 扩展现有 packed decode | P2 | 消除 KDA decode 的 q/k/v split、reshape 和临时 cu_seqlens |
| 07 | [`dlc_kimi_k3_mla_context_attention_bf16`](07_dlc_kimi_k3_mla_context_attention_bf16.md) | 新增（strict follow-up） | P2 | 直接消费 `k_nope/k_pe/v`，移除 chunked-context 的 native K concat；不阻断 V1 |

建议排期：**01 先行**；02/03/04/05 可并行；06 在 01 后按 decode profile 决定是否立即启用；
07 是严格零拷贝长 context 的后续项，不进入 V1 阻断路径。

## 已有算子覆盖结论

详细核查见 [`00_已有算子覆盖核查.md`](00_已有算子覆盖核查.md)。核心结论：

- KDA conv prefill/update、chunk prefill、recurrent decode core、gated RMSNorm 已有 DLC 算子，不重复提单。
- `dlc_vllm_concat_and_cache_mla_` 已有，但只做 cache 写入，不含 K3 的 q concat 和 RoPE；
  因此 MLA epilogue 仍需要 03/04。
- `dlc_apply_gpt_j_rope`、`_fused_rms_norm` 等可以拆开覆盖 MLA q/k RoPE 和 q/kv RMSNorm，
  但前者会退化为多次 launch，后者两次调用已足够，本批不单提 RMSNorm。
- KDA state gather/scatter 在当前 split state pool 下由连续 tensor 的 native 索引覆盖，
  暂不作为性能需求提交；等 TP32/长上下文 profile 再决定。
- Latent MoE tail、MLA output gate、MoonViT3d 暂不在本批，先等待 profile 和视觉路径验证。

## 使用方式

1. 按优先级复制对应 Markdown 正文到 GitHub Issue Form 的相同字段。
2. 表单中的代码块、checkbox 和验证环境可直接保留。
3. 提交前确认日志、模型路径和 benchmark 数据允许公开。
4. 每份文档的「Reference、日志与附件」列出当前适配代码和实测报告。

## 相关文档

- [`../算子移植与原生回退_2026-10-28.md`](../算子移植与原生回退_2026-10-28.md)
- [`../切片模型_TP8_2026-10-28.md`](../切片模型_TP8_2026-10-28.md)
- [`reports/real_slice_perf_2026-10-28.md`](reports/real_slice_perf_2026-10-28.md)
