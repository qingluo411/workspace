# Hy4-preview-Mix DLC 算子需求索引

算子需求按 `DLC_Custom_Kernel/.github/ISSUE_TEMPLATE/operator_request.yml`
的字段组织，一份文档对应一个可独立提交、独立验收的公开算子。

| 优先级 | 文档 | 算子 | 类型 | 阻塞点 |
|---|---|---|---|---|
| P0 | [01_ihc_pre.md](01_ihc_pre.md) | `torch.dlc_ihc_pre` | 新增 | iHC attention/MLP 入口 |
| P0 | [02_ihc_post.md](02_ihc_post.md) | `torch.dlc_ihc_post` | 新增 | iHC 子块输出回写 |
| P1 | [03_ihc_head.md](03_ihc_head.md) | `torch.dlc_ihc_head` | 新增 | 最终 residual 通道合并 |
| P0 | [04_sparse_mla_sink.md](04_sparse_mla_sink.md) | `torch.dlc_sparse_mla_apply_sink_` | 新增 | learnable attention sink |
| P1 | [05_gated_mla_epilogue.md](05_gated_mla_epilogue.md) | `torch.dlc_gated_mla_epilogue` | 新增 | gated MLA 生产性能 |
| P0 | [06_fused_moe_awq_hy4.md](06_fused_moe_awq_hy4.md) | `torch.dlc_vllm_fused_moe_w4a16_awq_col_major` | 扩展/修复 | 256 experts、FP32 router、clamp 10 |

建议提交顺序：

1. `ihc_pre` 与 `ihc_post`：可并行，二者是 iHC decode 的主要 launch 开销。
2. `sparse_mla_apply_sink_`：P0 正确性缺口，不能静默退化为无 sink。
3. `fused_moe_awq_hy4`：先修 vllm-cl 接入，再补 256-expert/TP8/EP8 回归。
4. `ihc_head`：功能可先走 eager，生产性能验收前完成。
5. `gated_mla_epilogue`：不阻塞正确性，进入性能优化阶段后完成。

说明：

- 首批不要求新增 indexer 算子；DLC 先复用 BF16 indexer cache 路径。
- `sparse_mla_apply_sink_` 是独立 epilogue，便于复用到 DCP combine 之后；
  后续可再评估把 sink 融合进 sparse attention 主 kernel。
- iHC `post + next pre` 跨层融合是后续 P2 优化，不在本索引中。

