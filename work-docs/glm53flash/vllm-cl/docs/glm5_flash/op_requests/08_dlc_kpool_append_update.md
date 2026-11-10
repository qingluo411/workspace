<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22）：完全新算子（P0） -->
<!-- 2026-10-09 撤销：本单不再作为独立需求提交，原有范围已全部并入其他单据：
     S2 tail 环写入（环更新）→ #07 扩展 07_dlc_kpool_compress.md（dlc_kpool_append_compress）
     S4 池读出            → #09 扩展 09_dlc_kpool_mqa_logits.md（dlc_kpool_select_topk，kernel 内按 block table 直读 pool cache）
     tail 环槽映射         → #19 19_dlc_kpool_slot_mapping.md（dlc_kpool_slot_mapping）
     若本单已经在算子仓库建过 issue，请在该 issue 下留言说明并关闭。
     本文件 2026-10-09 之前的完整内容（ABI、参考实现、对拍数据）见 git 历史。 -->

# [Operator Request] dlc_kpool_append_update（已撤销，不再单独实现）

## 状态：作废（2026-10-09）

本单最初的范围是「tail 环形更新 + 整池落盘 + 池读出」，其中还包含「内建 tail slot 计算、
消除 `searchsorted` CPU 往返」。2026-09 之后插件侧改成无状态实现，范围逐块被拆分，
现在**没有任何一块还留给本单**：

| 原属本单的部分 | 去向 | 备注 |
|---|---|---|
| 池化压缩（原 S3，`dlc_kpool_compress` 已交付并接入） | **#07**（已交付） | `_seed_kpool_fused`，`kpool_indexer.py:695` |
| tail 环形写入 / 环更新（原 S2） | **#07 扩展** | 并入 07 后改叫 `dlc_kpool_append_compress`，ABI 张量化、整批一次调用 |
| 池读出（原 S4，可选侧 `dlc_kpool_pooled_read`） | **#09 扩展** | 并入 09 后在 kernel 内按 block table 地址算术直接读 `pool_cache`，不再需要独立读出算子 |
| tail 环槽映射 / 消除 `searchsorted` CPU 往返 | **#19**（新提，P1） | 池槽 + tail 环槽双输出，一次算完 |
| kernel 内维护 `tail_lengths` / `sequence_lengths` 的有状态设计 | **作废** | 插件已改无状态：`phase = (seq_len - query_len) % pool_size` 每步由 metadata 推导，请求换序 / 块回收无需恢复状态 |

## 实现要求

不要按本单实现任何算子。请以实现以下三篇为准：

- `07_dlc_kpool_compress.md` —— `torch.dlc_kpool_append_compress`（池化压缩 + tail 环更新）；
- `09_dlc_kpool_mqa_logits.md` —— `torch.dlc_kpool_select_topk`（按 block table 读池 + 打分 + topk + 展开）；
- `19_dlc_kpool_slot_mapping.md` —— `torch.dlc_kpool_slot_mapping`（池槽 + tail 环槽映射）。

## 对插件侧的影响

- `08_dlc_kpool_append_update.md` 里原先承诺的「S2 环写」收益（`_seed_kpool_fused` 单层
  0.23 ms/层 ≈ 1.6 ms/步，dlc:0、D=128）统计口径随之并入上述三单；
- 本单不再出现在 `README.md` 的提交清单与 `接入清单_2026-09-28.md` 中。
