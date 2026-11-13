# GLM-5.3-Flash 融合算子需求（Issue 草稿）

本目录保存 GLM-5.3-Flash 在 DLC 上从「纯 PyTorch native 组合实现」迁移到「融合算子」的
算子需求 issue 草稿，共 **18 个**，按 `docs/glm5_flash/op_request.yml` 的 Issue Form 字段顺序编写。

> **2026-11-13 追加（MTP）**：新增 **#20 `dlc_fused_eh_norm_bf16`**（P1）。
> 这是目前唯一确认的 MTP 专属融合算子：当前 `fused_eh_norm_dlc` 是 mask + 两次
> RMSNorm + cat 的 Python fallback，`T=4/H=4096` 约 0.58 ms。详见
> [`20_dlc_fused_eh_norm.md`](20_dlc_fused_eh_norm.md)。

## 背景

当前 GLM-5.3-Flash 适配在 DLC 上全部走 native PyTorch 组合实现（见各文件的「背景与功能说明」），
可以跑通功能，但长 prefill 与 decode 的 kernel 数量、HBM 往返和 Python 循环开销都不可接受。
本目录把每个 native 组合实现整理成一条算子需求，供算子组评审排期。

> **2026-09-21 更新（已有算子覆盖核查）**：已在 DLC 上逐条实测现有算子。结论：18 条里
> **5 条已被现有算子覆盖（建议撤回）、5 条应改写为「扩展已有算子能力」、2 条应改写为「修复算子功能或精度」、
> 6 条确实是新增算子**（其中 #02 的判定已于 09-22 修正，见下条）。完整证据、约束与待算子组确认的问题见
> [`00_已有算子覆盖核查.md`](00_已有算子覆盖核查.md)，实测脚本见 `probes/`。
>
> **2026-09-28 更新（#05 重新提为 P0，推翻「撤回」）**：模型首次真机跑通后发现 prefill 单步 ≈ 60 s、
> host CPU 满载而 DLC 无计算。profile 定位到 **#05 的 `index_select` / `index_copy_` 才是当前第一瓶颈**
> （占 prefill 单步 64%）。原因是 KDA recurrent state 是 Mamba 共享页里的**非连续 view**
> （`[1168,8,128,128] fp32`，dim0 stride 135680 ≠ 131072），通用索引算子在该布局下掉进 host 逐元素路径：
> 同样只搬 8 页，`index_select` 0.181 ms → **579.7 ms（3196x）**、`index_copy_` 1.544 ms → **1873.0 ms（1213x）**；
> 34 层累计 58.7 ms → **83.4 s/步**。2026-09-21 的「已覆盖、融合收益 <5%」结论基于 `[8,4,16,16]`（8 KiB）验证张量，
> **不成立，作废**。→ **#05 重新提为 P0 新增算子**，见 [`05_dlc_kda_state_gather_scatter.md`](05_dlc_kda_state_gather_scatter.md)
> 与 [`../给算子组_05_state_pool_index_perf.md`](../给算子组_05_state_pool_index_perf.md)、
> 复现脚本 `probes/probe_state_pool_index_cost.py`。

> **2026-09-22 更新（需求分类定稿）**：按覆盖核查结果把 18 条需求分成四类，并逐条调整了
> `需求类型` / `优先级` / `算子名称`（每份文件顶部新增「需求分类（2026-09-22）」注释，
> `## 需求类型` 之后新增「需求分类」段，写明现有算子、实测值、缺口、落地方式与本次调整）：
> **完全新算子 7 条（06/07/08/09/10/15/16）、需扩展已有算子能力 3 条（03/04/13）、
> 需修复已有算子问题 1 条（11）、可复用已有算子 7 条（01/02/05/12/14/17/18，撤回不提交）**。
> （2026-09-28 起 05 不再属于「可复用」，已重新提为 P0 新增算子；见上方更新。）
> 实际需要提给算子组的共 **11 条**（2026-09-22 三次复核后为 **10 条**，见上方「新算子实测」），
> 逐条明细见下一节「需求分类总表」。
>
> 2026-09-22 结论修正：**#02 的 page 0 不是缺陷** —— page 0 是 vLLM 保留的 null block（`NULL_BLOCK_ID=0`），
> kernel 跳过 0（null block）与 -1（pad slot）属设计行为，CUDA graph 的 padding 行正依赖该语义；
> 初版探针的 reference 未实现该约定，误报为缺陷。故 #02 由「需修复」改为「可复用（撤回）」。
>
> 2026-09-22 结论修正（#18）：**GLM-5.3-Flash 不存在非 512B 对齐的页** —— 5 类页
> （MLA full 262,144 B / KPool pool 32,768 B / KPool tail 4,096 B / KDA conv 18,432 B /
> KDA recurrent 524,288 B）全部是 512B 的整数倍，且按真实页几何多段调用实测 bit-exact
> （`probes/probe_zero_block_ids_glm5_geometry.py`）。故 #18 由「需扩展（以接入为主）」
> 改为「可复用（撤回）」：现有 `dlc_vllm_zero_block_ids_` 已够用，只需插件接入。
>
> 2026-09-22 二次复核（扩展 vs 新增的判据）：**#13 `dlc_mhc_pre` 维持「扩展」**（`norm_weight`/`norm_eps`
> 是上游 `MHCPreOp` 在所有后端都有的可选参数，`None` 时行为不变，属补齐签名）；
> **#16 `dlc_silu_and_mul_with_clamp` 由「扩展」改为「新增独立算子」**（上游即独立 op，`silu_and_mul` 是所有模型
> FFN 共享热路径且有量化融合 pattern 依赖其签名）。两者合计当时为提交 11 条；
> 2026-09-22 三次复核后再减 1 条（#06 撤回、#03/#04 改判为「需修复」），即提交 10 条。
>
> **2026-09-22 新算子实测（算子组交付 5 个新算子后的复核）**：算子组已在 pytorch `test-glm53flash`
> 分支（HEAD `10e44cfff0a`）提交 5 个 KDA 专用算子 —— `dlc_kda_causal_conv_prefill`、
> `dlc_kda_causal_conv_update`、`dlc_kda_chunk_prefill`、`dlc_kda_recurrent_decode`、
> `dlc_kda_rmsnorm_sigmoid_gate`，已在 DLC 上逐一实测（脚本 `probes/probe_new_kda_*.py`）：
> **#06 完全满足（撤回）；#02 数值逐元素正确但每次调用搬一遍整池（1509 槽位 7.49ms vs 现有
> 0.59ms ≈13×，且耗时随槽位总数线性增长；**归因已修正：94% 在 kernel 内的整池 `HBM2VMem`（`kda_causal_conv_update.c:109`），
> host wrapper 的 4 遍整池拷贝只占 6%/0.47ms**）；#01 数值正确但 N≥3072 卡死设备（无 token 分块、整条序列
> 驻留 16MB VMEM）；#03 每次调用必抛异常且 kernel 卡死设备；#04 state 数值正确但 `out` 逐位恒为 0
> （已定位到 `dlc_kda_recurrent_decode.cpp:189` 把常量 0.0f 写进 torch 回读的那半；state 与插件 native
> 互为转置（**2026-09-22 算子组已同意统一为 `[S,H,Dv,Dk]`**，新参考实现 `refs/kda_reference_dv_dk.py`）、
> `num_accepted_tokens` 未生效）。**
> **2026-09-22 晚最终决定（适配组）：4 个 KDA 新算子都采用，由算子组修复**：#01~#04 全部由
> 「撤回 / 需扩展」改为「**需修复（提交）**」，**#06 撤回**（已满足，插件接入）。
> 分类变为 **完全新算子 6 / 需扩展 1 / 需修复 5 / 可复用（撤回）6 → 实际提交 12 条**
> （01、02、03、04、07、08、09、10、11、13、15、16）。01~04 的可粘贴修复单见
> [`给算子组_01_02_修复要点.md`](给算子组_01_02_修复要点.md) 与
> [`给算子组_03_04_修复要点.md`](给算子组_03_04_修复要点.md)。
> #02（update）另有一份**可直接贴的精简版**：[`给算子组_02_update_issue.md`](给算子组_02_update_issue.md)
> （只讲「问题 / 修法 / 复现」，含一条命令跑完 4 个问题的脚本）。

> **2026-09-28 修复后复测（#01/#02 已可接入）**：算子组的修复
> （`DLC_Custom_Kernel` `fix: #7949 resolve kda_causal_conv_prefill stuck` / `perf: #7950 optimize
> kda_causal_conv_update, del big dma`；`pytorch` `remove copy transpose in kda_causal_conv_*`）已
> cherry-pick 到本地 `test-glm53flash` 并复测：
> **#01（prefill）** N=4096（修复前卡死）/8192/16384/32768、B=16 批、非连续 state 视图全部通过
> （`out_err≈3.9e-03` bf16 级、`state_err=0`、行间段零改写）；**#02（update）** 1509 槽单次
> **0.06 ms**（修复前 7.49 ms，随槽位线性增长）且与槽位数无关、host 无 D2H、可被 `DLCGraph` 捕获、
> pad 行输出全 0，验收回归 `tests/test_glm5_kda_conv_update_op.py` **11/11 PASS**。
> 新 ABI：`conv_state=[S_slot,state_len,D]`(SD，in-place 写回)、`weight=[4,D]` fp32、
> update 只返回 `out`。复测脚本（probes）与 #01/#02 文档已同步更新；详见两份文档的
> 「2026-09-28 修复后复测」小节。**遗留**：kernel 不读 `conv_state.stride`，非连续视图靠 torch 侧
> 自动「连续副本+回写」兜底（≈0.05 ms/层）；prefill kernel 对 `idx=-1` 仍是断言拒绝。


## 使用方法

1. 打开算子组仓库的 Issue 页面，选择模板「算子需求」。
2. 每个 `.md` 文件对应一个 issue：文件里的 `## xxx` 小节与模板字段一一对应，逐字段粘贴即可。
3. 标题使用文件中的一级标题（已含 `[Operator Request]` 前缀）。
4. **只提交「需求分类总表」里标注「提交」的 13 条**；标注「可复用（撤回）」的 5 条
   （06/12/14/17/18）仅作内部记录与插件侧动作清单，不要提单。
   （2026-09-28：05 由「可复用（撤回）」改为「提交 · P0」，见上方更新。）
5. 待确认字段：`目标版本与期望时间` 目前填的是建议里程碑，需要按算子组实际排期替换。
6. **#03 / #04 两条 P0 已整理成可直接粘贴的修复要点**：见
   [`给算子组_03_04_修复要点.md`](给算子组_03_04_修复要点.md)（含复现步骤、实测日志、根因代码行、验收标准，
   以及可直接复制的复现脚本全文）。

## 需求清单（初稿，2026-09-21）

（下表保留最初的 native 实现清单，用于追溯「哪段 Python 组合实现对应哪条需求」；
`类型` / `优先级` 是初稿值，**2026-09-22 定稿后的分类、需求类型与优先级以「需求分类总表」为准**）

| # | 算子 | 类型 | 优先级 | native 实现位置 | 说明 |
|---|---|---|---|---|---|
| 01 | `dlc_kda_causal_conv_prefill` | 新增 | P0 | `vllm_cl/patch/patch_kda.py:21/31` | 34 层 KDA 短卷积 prefill，当前 Python 逐 token 循环 |
| 02 | `dlc_kda_causal_conv_update` | 新增 | P0 | `vllm_cl/patch/patch_kda.py:48` | KDA 短卷积 decode / MTP verify 更新 |
| 03 | `dlc_kda_chunk_prefill` | 新增 | P0 | `vllm_cl/patch/patch_kda.py:93/139` | KDA chunked prefill；当前直接复用逐 token 递推 |
| 04 | `dlc_kda_recurrent_decode` | 新增 | P0 | `vllm_cl/patch/patch_kda.py:134` | KDA decode 递推；当前显式不支持 speculative |
| 05 | `dlc_kda_state_gather_scatter` | 新增 | P1 | `vllm_cl/patch/patch_kda.py:81/87` | 递归状态 gather/scatter |
| 06 | `dlc_rmsnorm_sigmoid_gate` | 新增 | P1 | `vllm_cl/patch/patch_kda.py:145` | KDA 输出 gated RMSNorm |
| 07 | `dlc_kpool_compress` | 新增 → **扩展** | P0 | `vllm_cl/patch/models/kpool_indexer.py:391` | 4-token 逐通道 softmax 池化压缩；**2026-10-09 扩为 `dlc_kpool_append_compress`，把 08 的 tail 环写入并进来** |
| 08 | `dlc_kpool_append_update` | ~~新增~~ **撤销** | ~~P0~~ | `vllm_cl/patch/models/kpool_indexer.py:321/359` | **2026-10-09 撤销**：S2 环写入 → #07 扩展；S4 池读出 → #09 扩展；tail 环槽 → #19 |
| 09 | `dlc_kpool_mqa_logits` | 新增 → **扩展** | P0 | `vllm_cl/patch/models/kpool_indexer.py:471` | 池级打分（逐 head dot → ReLU → 加权）；**2026-10-09 扩为 `dlc_kpool_select_topk`，把池读出（S4）+ topk（S6）+ 展开（S7，原 #10）并进来** |
| 10 | `dlc_kpool_expand_topk` | 新增 → **退出主路径** | P1 | `vllm_cl/patch/models/kpool_indexer.py:500` | 池 id 展开为 token id + tail 拼接；**2026-10-09 语义并入 #09 扩展**，入口保留可用 |
| 11 | `dlc_sparse_mla_forward_nope512` | 扩展 | P0 | `vllm_cl/attention/sparse_mla.py:569` | 512 维 NoPE 稀疏 MLA 前向 |
| 12 | `dlc_concat_and_cache_mla_nope512` | 扩展 | P1 | `vllm_cl/attention/sparse_mla.py:60` | 512 维 NoPE MLA cache 写入 |
| 13 | `dlc_mhc_pre` | 新增 | P1 | `vllm/models/glm5next/nvidia/model.py` + `mhc_pre_torch` | mHC 前置四路残差混合（含 20 次 Sinkhorn） |
| 14 | `dlc_mhc_post` | 新增 | P1 | `mhc_post_torch` | mHC 后置残差合并 |
| 15 | `dlc_mhc_fused_post_pre` | 新增 | P2 | `MHCFusedPostPreOp` | mHC post+pre 融合，减少残差流往返 |
| 16 | `dlc_silu_and_mul_with_clamp` | 扩展 | P1 | `SiluAndMulWithClamp.forward_native` | 带 `swiglu_limit=10.0` 的 SwiGLU（dense/shared/视觉） |
| 17 | `dlc_fused_q_kv_rmsnorm` | 新增 | P2 | `vllm_cl/patch/patch_fused_qk_rmsnorm.py` | 视觉塔 Q/K RMSNorm |
| 18 | `dlc_kv_cache_zero_blocks` | 新增 | P0 | `vllm_cl/patch/patch_kv_block_zeroer.py:193` | KV cache 物理块清零；当前 `zero_()` 被注释禁用，须恢复语义并消除 view 布局转换 |
| 19 | `dlc_kpool_slot_mapping` | 新增 | P1 | `vllm_cl/patch/models/kpool_slot_mapping.py:18`、`vllm_cl/patch/patch_mla_indexer.py:119`、`vllm_cl/kpool_worker_slots.py:28` | 池槽 + tail 环槽双输出；DLC 无 `searchsorted`，现为每步 2 次 CPU 往返（2026-10-09 新增，承接 08 号的 tail slot 部分） |
| 20 | `dlc_fused_eh_norm_bf16` | 新增 | P1 | `vllm_cl/patch/models/patch_glm5_mtp.py:33` | MTP EH norm 融合：position mask + 两路 RMSNorm + concat；当前 fallback `T=4/H=4096` 约 0.58 ms（2026-11-13 新增） |

## 需求分类总表（2026-09-22 定稿）

| # | 需求（初稿算子名） | 需求分类 | 需求类型 | 优先级 | 覆盖核查判定与建议动作 |
|---|---|---|---|---|---|
| 01 | `dlc_kda_causal_conv_prefill` | 需修复 | 修复算子功能或精度 | P0 | **采用新算子**（2026-09-22 晚定稿）。数值正确但 **N≥3072 卡死设备**（无 token 分块、整条序列驻留 16MB VMEM）→ 提交（点名修复 P0；另需放行 `-1`(PAD_SLOT_ID)、去掉 host 侧 D2H 同步。经源码核对，`state_indices` 1D 与 fp32 `weight` **不是缺陷**，旧结论已作废）；修复期间插件用 `torch._causal_conv1d` 兜底 |
| 02 | `dlc_kda_causal_conv_update` | 需修复 | 修复算子功能或精度 | P0 | **采用新算子**（2026-09-22 晚定稿）。数值**逐元素正确**（含 spec/L=4,6，state_err=0），但每次调用搬一遍整池（1509 槽位 **7.49ms vs 0.59ms ≈13×**，耗时随槽位总数线性增长；**归因：kernel 内 `kda_causal_conv_update.c:109` 整池 `HBM2VMem` 占 94%，host wrapper 4 遍拷贝只占 6%**）、`num_accepted_tokens=None` 时多一次 D2H 同步、`idx=-1` 行输出脏值 → 提交（4 项点名修复 + 澄清 spec 提交语义）；修复期间插件用 `torch._causal_conv1d_update` 兜底 |
| 03 | `dlc_kda_chunk_prefill` | 需修复 | 修复算子功能或精度 | P0 | 算子组已交付同名算子，但**每次调用必抛 `permute` 异常**（5 维张量用 4 阶 permute，`FusedOps.cpp:845`）、**launch 出的 kernel 卡死设备**、并就地改写输入 `beta_logits` → 提交（点名修复）；state 布局统一为 `[B,H,Dv,Dk]` |
| 04 | `dlc_kda_recurrent_decode` | 需修复 | 修复算子功能或精度 | P0 | 算子组已交付同名算子，**state 数值正确（与 fla 语义逐项对齐，`state_err≈3e-5`）但 `out` 逐位恒为 0**（kernel `dlc_kda_recurrent_decode.cpp:189` 把常量 0.0f 写进 torch 回读的那半；二次调用 state 仍在更新）；state 布局统一为 `[S,H,Dv,Dk]`（算子组 2026-09-22 确认，与插件 native `[H,V,K]` 一致、插件零转换；新参考实现 `refs/kda_reference_dv_dk.py`）、`num_accepted_tokens` 未生效 → 提交（点名修复） |
| 05 | `dlc_kda_state_gather_scatter` | 新增算子 | 新增算子 | **P0** | **2026-09-28 推翻原「可复用」结论**：真机几何（`[1168,8,128,128] fp32`、Mamba 页内非连续 view、dim0 stride 135680）下通用索引算子掉进 host 逐元素路径 —— `index_select` 0.181→579.7 ms（3196x）、`index_copy_` 1.544→1873.0 ms（1213x），34 层累计 58.7 ms→83.4 s/步，占 prefill 单步 64% → **提交（P0）**，按页 stride 寻址只搬被选页 |
| 06 | `dlc_rmsnorm_sigmoid_gate` | 可复用（撤回） | 其他 | P2 | 算子组已交付 `dlc_kda_rmsnorm_sigmoid_gate`，实测 sigmoid 语义正确（err 0.34%，3D/4D、bf16/fp32 weight、T=0 均通过）→ 插件接入，不提交 |
| 07 | `dlc_kpool_compress` → `dlc_kpool_append_compress` | 完全新算子 → **扩展已有算子能力** | 扩展已有算子能力 | P0 | 2026-09-28 已交付 `torch.dlc_kpool_compress` 并接入；**2026-10-09 扩边界**：吸收 08 的 tail 环写入，ABI 张量化（整批、kernel 内读 prior/写尾池）→ 提交扩展 |
| 08 | `dlc_kpool_append_update` | **撤销** | — | — | **2026-10-09 撤销**，范围全部并入其他单据（S2→#07 扩展、S4→#09 扩展、tail 环槽→#19）；已在算子仓库建 issue 的请关闭 |
| 09 | `dlc_kpool_mqa_logits` → `dlc_kpool_select_topk` | 完全新算子 → **扩展已有算子能力** | 扩展已有算子能力 | P0 | 2026-09-28 已交付并接入；**2026-10-09 扩边界**：吸收池读出（S4，kernel 内按 block table 直读 cache）、topk（S6）、展开 + 残尾（S7，原 #10）→ 提交扩展 |
| 10 | `dlc_kpool_expand_topk` | 完全新算子 → **退出主路径** | — | — | **2026-10-09 语义并入 #09 扩展**（`dlc_kpool_select_topk` 直接写 topk buffer）；入口保留给其他调用方与单测 |
| 11 | `dlc_sparse_mla_forward_nope512` | 需修复 + 扩展 | 修复算子功能或精度 | P0 | 512-NoPE NaN/卡死 + topk 2051 分派 → 提交（点名修复 `dlc_sparse_attention_kv_cache`） |
| 12 | `dlc_concat_and_cache_mla_nope512` | 可复用（撤回） | 其他 | P2 | `dlc_vllm_concat_and_cache_mla_` 在 kpe=0 亦 bit-exact → 插件接入，不提交 |
| 13 | `dlc_mhc_pre` | 需扩展 | 扩展已有算子能力 | P1 | 已覆盖，缺 `norm_weight`/`norm_eps` → 提交（加参数） |
| 14 | `dlc_mhc_post` | 可复用（撤回） | 其他 | P2 | 已完全覆盖（err 3.90e-03）→ 不提交 |
| 15 | `dlc_mhc_fused_post_pre` | 完全新算子 | 新增算子 | P2 | 未覆盖；建议先接入 13/14 再评估收益 → 提交（P2） |
| 16 | `dlc_silu_and_mul_with_clamp` | 完全新算子 | 新增算子 | P1 | 无 clamp 变体；上游也是独立算子 `_C.silu_and_mul_with_clamp`，`silu_and_mul` 是全模型共享 FFN 热路径 → 新增独立算子，提交 |
| 17 | `dlc_fused_q_kv_rmsnorm` | 可复用（撤回） | 其他 | P2 | 两次 `dlc_fused_rms_norm_f32_bf16` 即等价 → 插件接入，不提交 |
| 18 | `dlc_kv_cache_zero_blocks` | 可复用（撤回） | 其他 | P2 | 5 类页（4,096~524,288 B）全为 512B 整数倍，真实页几何实测 bit-exact → 插件接入，不提交 |
| 19 | `dlc_kpool_slot_mapping` | 完全新算子 | 新增算子 | P1 | 未覆盖（DLC 无 `searchsorted`/`bucketize`；`convert_req_index_to_global_index` 不含边界求 request id 这一步）→ 提交。pool 槽 + tail 环槽合并为一条双输出算子，省掉每步 2 次 CPU 往返（V1：worker + pool builder；tail builder 那次已由插件侧去重） |
| 20 | `dlc_fused_eh_norm_bf16` | 完全新算子 | 新增算子 | P1 | 现有 DLC RMSNorm 仅单输入 FP32，无 position mask + 两路 RMSNorm + concat；MTP 专属融合，提交 |

分类统计（2026-09-22 晚最终，见上方「新算子实测」）：完全新算子 6 / 需扩展 1 / **需修复 5** /
可复用（撤回）6 → **实际提交 12 条**：01、02、03、04、07、08、09、10、11、13、15、16。

**2026-10-09 重划后的清单状态（19 条 → 撤销 08 后 18 条）**：

- 撤销 1 条：**08**（范围并入 07 / 09 / 19）；
- 由「新增算子」改为「扩展已有算子」2 条：**07**（吸收 08 环写入）、**09**（吸收 #10 + 原 S4/S6）；
- 退出主路径 1 条：**10**（语义并入 09，入口保留）；
- 新增 1 条：**19**（`dlc_kpool_slot_mapping`，池槽 + tail 环槽双输出）；
- 其余不变：01、02、03、04、05、06、11、12、13、14、15、16、17、18。

即当前提交给算子组的清单为：**01、02、03、04、05、07（扩展）、09（扩展）、11、13、15、16、19**；
12、14、17、18 由已有算子覆盖（撤回）；06 已由算子组交付的算子覆盖。

**2026-10-09 修订**：
- 新增 **#19 `dlc_kpool_slot_mapping`**（P1，双输出：池槽 + tail 环槽）。原先散落在 #08 里的
  「内建 tail slot / 消除 `searchsorted` CPU 往返」由本单承接 —— #08 的 ABI 只收 `req_ids`/`token_offsets`，
  没有 metadata，覆盖不到这部分。
- **#08 `dlc_kpool_append_update` 撤销**：池化压缩已由 #07 覆盖，tail 环写入并入 **#07 扩展**，
  池读出并入 **#09 扩展**，tail 环槽移交 **#19**，有状态设计作废（插件已改无状态）→ 本单不再单独实现。
  若已在算子仓库建 issue，请留言说明并关闭。详见
  [`08_dlc_kpool_append_update.md`](08_dlc_kpool_append_update.md)。
- **#07 / #09 的边界扩展**：07 吸收 tail 环写入、09 吸收池读出 + topk + 展开（原 #10），
  两单仍是 P0，但定级理由由「省小算子耗时」改为**「清掉 forward 内的 kernel 外编排与 D2H，
  是解除 `--enforce-eager`、打开图捕获的前置」**。详见
  [`07_dlc_kpool_compress.md`](07_dlc_kpool_compress.md)、
  [`09_dlc_kpool_mqa_logits.md`](09_dlc_kpool_mqa_logits.md) 与
  [`../kpool_mla_dataflow.md`](../kpool_mla_dataflow.md)。
- **新增 #04 的 ABI 扩展单**：**#04 本身（`04_dlc_kda_recurrent_decode.md`）已完成并接入**，
  但它在 decode 里要求 5 个激活类操作数各是「独立 512B 行」张量，导致插件侧每层每步 6 次物化拷贝
  （34 层 ≈ 204 次/步）。新的扩展单
  [`04b_dlc_kda_recurrent_decode_abi.md`](04b_dlc_kda_recurrent_decode_abi.md)（P0）把这 5 项改成
  「父张量 view + `(offset, pitch)` 取数」、`query_start_loc` 改成「整条常驻 buffer + `batch`」，
  并附带修掉 `batch == 128` 的越界挂死。它是 P0-3 消解 `_fused_operand` 的最后一段，
  方案见 [`../_fused_operand_方案_2026-10-09.md`](../_fused_operand_方案_2026-10-09.md) §3.3/§3.4/§6。
  **不改 `#02`**（`dlc_kda_causal_conv_update` 的 ABI 与输出形状全部保持原样）。

## 依赖与排期建议

分类定稿（2026-09-22）后的排期与提交建议：

- **实际提给算子组的 12 条（2026-09-22 口径）**：01、02、03、04、07、08、09、10、11、13、15、16
  （06 已由算子组交付的 `dlc_kda_rmsnorm_sigmoid_gate` 覆盖，撤回）。
  12、14、17、18 可由现有算子覆盖，**撤回不提交**，对应改动全部落在插件侧。
  （05 已按 2026-09-28 结论重新提为 P0，不再撤回。）
  **2026-10-09 重划后**：08 撤销、10 退出主路径、07/09 由「新增」改为「扩展」、新增 19 →
  当前在跟 12 条变为：**01、02、03、04、05、07（扩展）、09（扩展）、11、13、15、16、19**。
  **2026-10-09 追加**：**04b（#04 的 ABI 扩展）**，见下一条。
- **KDA 线（01~04、06）——算子组的工作量集中在 4 条「修新算子」**（均 P0，建议一起排）：
  算子组已交付 5 个专用算子，实测后**适配组决定全部采用**，请算子组按下面 4 条修复：
  01 prefill 需 token 分块（现在 N≥3072 卡死设备，P0）+ 放行 `-1`(PAD_SLOT_ID) + 去掉 host 侧 D2H 同步；
  02 update 需去掉 **kernel 内的整池 DMA**（`kda_causal_conv_update.c:109`，占 13× 的 94%）与 host 的 D2H 同步
  （现在 1509 槽位 7.49ms vs 0.59ms，且随槽位线性增长；host wrapper 的 4 遍拷贝只占 0.47ms/6%）
  + pad 行输出写 0 + 澄清 spec 提交语义；
  03 chunk_prefill 需修返回处的 4 阶 permute（5 维张量）+ kernel 卡死设备 + `beta_logits` 就地改写
  + state 布局统一为 `[B,H,Dv,Dk]`；
  04 recurrent_decode 需修 `dlc_kda_recurrent_decode.cpp:189` 的输出写回（out 恒 0）+ state 布局统一为
  `[S,H,Dv,Dk]`（已确认）+ 实现 `num_accepted_tokens`。
  修复期间插件用现有算子兜底（开关默认关），修好后逐个切换并重跑数值对齐；
  06 已由 `dlc_kda_rmsnorm_sigmoid_gate` 直接覆盖，撤回转插件接入。
  可粘贴的完整修复单：[`给算子组_01_02_修复要点.md`](给算子组_01_02_修复要点.md)、
  [`给算子组_03_04_修复要点.md`](给算子组_03_04_修复要点.md)；
  #02 的精简版（问题 / 修法 / 复现）：[`给算子组_02_update_issue.md`](给算子组_02_update_issue.md)；
  **#03/#04 的 fp32 参考实现（state 布局 `[S,H,Dv,Dk]`）**：[`refs/kda_reference_dv_dk.py`](refs/kda_reference_dv_dk.py)
  （`chunk_prefill_ref` / `recurrent_decode_ref`，`--selftest` 纯 CPU 自检，用法见 [`refs/README.md`](refs/README.md)）。
- **KPool 线（07 扩展、09 扩展、19）P0/P1**：它们决定 DSA 稀疏选池能否跑起来，也决定 indexer 层
  能否开图。07 扩展把 tail 环写入收进 kernel（原 08 撤销），09 扩展把池读出 + topk + 展开
  收进 kernel（原 10 退出主路径），19 消除每步 `searchsorted` 的 CPU 往返（当前 2 次 ≈ 0.6 ms/步）；
  三单交付后 `native_kpool_indexer_step` 里的 kernel 外编排可整体删除。
  参考实现：`refs/kpool_reference.py`（池化压缩 / 池读出）、
  `07_dlc_kpool_compress.md` 与 `09_dlc_kpool_mqa_logits.md` 内附的 golden reference；
  核对/计时脚本 `probes/probe_kpool_pooled_read.py`。
- **11 号的 P0 性质是「修 bug」**：现有 `dlc_sparse_attention_kv_cache` 在 512-NoPE 下 NaN/卡死，
  且 topk 宽度不支持 2051；这条不修，11 层 DSA 会一直停留在 native 回退甚至崩溃。
- **13 号是「加参数」级的小改动**（`dlc_mhc_pre` 增加 `norm_weight`/`norm_eps`），性价比高；
  15 号等 13/14 接入后再实测残差流往返占比（< 15% 可放弃融合）。
- **16 号影响所有前馈路径**，建议与 KDA 扩展并行推进；它不是「扩展 `silu_and_mul`」而是**新增独立算子**
  （保持全模型共享的 `silu_and_mul` 签名与热路径不变，与上游 `_C.silu_and_mul_with_clamp` 同名同语义）；
  routed expert 侧已由 `dlc_vllm_fused_moe_*` 的 `swiglu_limit` 参数覆盖，不在本条目范围。
- **18 号不用提单**：GLM5 的 5 类页全部 512B 对齐（芯片侧只做 512B 对齐算子也够用），
  只需插件按「每块字节数」分组接入 `dlc_vllm_zero_block_ids_`；接入时若发现某类页非 512B 倍数，
  再按「kernel 支持按字节寻址」重新评估（本模型已确认不会出现）。
- **优先级调整**：01、02、18（均 P0→P2）、12、14（均 P1→P2）、06（P1→P2）随撤回降级，
  其余条目优先级不变；05 由 P2 升为 **P0**（2026-09-28），17 原本已是 P2。

## 本目录未覆盖的内容

- **fused MoE（AWQ）内 expert 激活**：已由 DLC kernel 的 `swiglustep` 覆盖（`vllm_cl/ops/awq_marlin.py`），不需要新算子。
- **FP32 router logits**：属于后续接口工作（当前首阶段接受 BF16 routing），不在本轮。
- **FP8 index cache / Hadamard 旋转**：BF16 bring-up 阶段不需要（`dot(Hq,Hk)=dot(q,k)`），作为后续独立需求。
- **块表 / slot mapping 的通用算子**：~~属于插件侧张量计算，不构成算子组需求~~ —— 2026-10-09 修订：
  KPool 的**池槽 + tail 环槽双输出**已单独提为 **#19 `dlc_kpool_slot_mapping`**
  （DLC 无 `searchsorted`/`bucketize`，每步 2 次 CPU 往返），
  其余通用块表映射仍留在插件侧。
- **KV cache 清零**：已由 `dlc_vllm_zero_block_ids_` 覆盖（见 18 号），本轮只做接入。
- **KDA 短卷积 / decode 递推 / MLA cache 写入 / mHC pre-post / 状态 gather-scatter**：分别由
  `_causal_conv1d(_update)`、`fused_recurrent_gated_delta_rule`、`dlc_vllm_concat_and_cache_mla_`、
  `dlc_mhc_pre/post`、通用 indexing 算子覆盖，2026-09-21 核查后不再作为新算子提单。

## 参考文档

- `docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_适配研究.md`：模型结构账本与各子系统建议算子清单（§5.4 / §6.5 / §7.3 / §8.1）。
- `docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_阻断点2-7.md`：阻断点状态与代码位置。
- `docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_联调问题与修复记录.md`：真机联调暴露的问题与处置。
- `vllm-cl/tests/test_kda_native.py`、`tests/test_glm5_kv_grouping_dlc.py`：可用于生成 reference 数值。
- `op_requests/接入清单_2026-09-28.md`：19 条需求的「可接入 / 可复用 / 等算子组」现状与接线点；2026-09-28 已接入 #01、#02、#06、#12、#18（含开关与回归测试）。
- `op_requests/00_已有算子覆盖核查.md`：19 条需求的既有算子覆盖结论、实测数据与约束。
- `op_requests/probes/`：覆盖核查用的实测脚本（每脚本自带 reference 与误差打印）。
