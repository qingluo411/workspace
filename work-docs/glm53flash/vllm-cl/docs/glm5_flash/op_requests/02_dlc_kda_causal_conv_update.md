<!-- 目标仓库 Issue Form：op_request.yml（算子需求） -->
<!-- 精简版（只讲问题 / 修法 / 复现，建议直接贴这个）：给算子组_02_update_issue.md -->
<!-- 生成日期：2026-09-21；来源：vllm-cl GLM-5.3-Flash 适配 -->
<!-- 需求分类（2026-09-22 晚最终定稿）：需修复 —— 采用算子组交付的新算子，本条提交、点名修复
     决定：适配组确认采用 dlc_kda_causal_conv_update（数值逐元素正确），由算子组修 host wrapper 的
     整池拷贝 / None 时的 D2H 同步 / pad 行脏输出后接入（**MTP/spec 本阶段不做，不要求澄清 spec 语义**）；
     修复期间插件用 torch._causal_conv1d_update 兜底。
     完整修复单（可直接粘贴）见 给算子组_01_02_修复要点.md -->
<!-- 2026-09-22 晚实测（加 warm-up 重测）：1509 槽位 7.49ms vs 现有 0.59ms（≈13×），且耗时随槽位总数
     线性增长（4→3072 槽位：0.61→14.73ms），与活跃请求数无关。
     2026-09-22 深夜修正归因：实测（probe_new_kda_conv_update_cost.py）成本 94% 在 kernel 内部
     （kda_causal_conv_update.c:109 每次调用整池 HBM2VMem，≈4 GB/s），host wrapper 的 4 遍整池拷贝只有 0.47ms/6%；
     「成本来自 wrapper 整池拷贝」的旧结论是按流量推算、非实测，已作废。 -->
<!-- 已有算子覆盖核查（2026-09-22 修正版：DLC 实测 + kernel/上游源码比对）：已被 torch._causal_conv1d_update 完全覆盖
     结论：KDA decode / MTP verify 需要的「多 token 卷积 + state 原位回写 + 每请求物理页索引 + num_accepted_tokens 回滚」现有算子全部支持，无需新增算子，也无需修算子。
     page 0 说明：page 0 是 vLLM 的保留 null block（NULL_BLOCK_ID=0），kernel 跳过 0 与 -1（PAD_SLOT_ID）是设计行为——CUDA graph 的 padding 行正是靠这个语义保持 state 不被污染。
     2026-09-21 的探针 reference 没有实现该约定，误报为「page 0 缺陷」；探针已于 2026-09-22 修正。
     备查（非请求、无需算子组动作）：prefill kernel（causal_conv1d.c:63,156）只跳过 -1、不跳过 0，与 update kernel 的空槽约定不同；
     但 prefill 的 cache_indices 同样取自 block table 首列（真实请求 >=1），padded 行在行号判断之前就被 seqlen<=0 跳过，故 vLLM 路径不会把 page 0 交给 prefill kernel。 -->

# [Operator Request] dlc_kda_causal_conv_update（KDA 短卷积 decode 更新）

## 需求类型
修复算子功能或精度

**需求分类：需修复（本条提交，点名修复）**
- **2026-09-22 晚最终决定：采用算子组交付的新算子**，由算子组修复下述缺陷后接入；
  修复完成前插件暂用现有算子兜底（迁移开关 `KDA_CONV_UPDATE_USE_NEW_OP` 计划加在 `vllm_cl/patch/patch_kda.py`，默认 False，**目前尚未落到代码里**，插件现在仍走 native 组合实现）。
  完整、可直接粘贴的修复单见 [`给算子组_01_02_修复要点.md`](给算子组_01_02_修复要点.md)。

- 现有算子：`torch._causal_conv1d_update(x, weight, bias, conv_state, conv_state_indices, num_accepted_tokens=None, query_start_loc=None, max_query_len=-1, block_idx_last_scheduled_token=None, initial_state_idx=None, has_bias=True, silu_activation=True)`（注册：`pytorch/aten/src/ATen/native/native_functions.yaml:427`；DLC 实现 `causal_conv1d_update_dlc`，wrapper `aten/src/ATen/native/dlc/CausalConv1d.cxx:170`，kernel `DLC_Custom_Kernel/dlc_kernels/conv/causal_conv1d_update.c`）。
- 功能覆盖（对照 KDA decode / MTP verify 的需求逐项核对）：单 token decode（`[B, dim]`）、多 token / spec（`[B, dim, T]` 与 packed + `query_start_loc`）、conv_state 原位滚动回写、每请求物理页索引、`num_accepted_tokens`（只提交被接受的 token）、silu 激活、`out` 复用 —— 全部具备，**不需要新算子**。
- 实测（2026-09-21，dlc:0；dim=256 / K=4 / bf16）：除 page 0 以外的索引逐元素一致（err `0.00e+00`：`[1]`、`[2]`、`[3]`，`[0,2]` 的第 2 行，`[1,0]` 的第 1 行）。
- page 0 的真相（2026-09-22 修正）：kernel 硬编码 `null_block_id = 0` 与 `pad_slot_id = -1` 并跳过这两行（`causal_conv1d_update.c:65-66,145,293,346`）；输出张量由 wrapper 预置零（`CausalConv1d.cxx:242` 的 `at::zeros`），被跳过行的 state 不动。这与上游约定完全一致：`vllm/v1/attention/backends/utils.py:46` 的 `NULL_BLOCK_ID = 0`、`vllm/v1/core/block_pool.py:188-191` 把 0 号块标记 `is_null` 且永不分配，CUDA graph 的 padded 行由上游显式填 `NULL_BLOCK_ID`（`vllm/v1/attention/backends/short_conv_attn.py:467`，另见 `:202-205` 注释）。因此 **page 0 不是缺陷，而是必须保留的语义**。
- 落地方式：插件把 `vllm-cl/vllm_cl/patch/patch_kda.py:45` 的 `causal_conv1d_update_dlc`（Python 逐请求循环）切换为 `torch._causal_conv1d_update`；`conv_state_indices` 继续沿用 block table 首列（`short_conv_attn.py:371`），不要自造从 0 开始的页号。注意 native 版 `_page_index` 对 `idx < 0` 会抛错，而 kernel 对 `-1` 是跳过，接入时按 kernel 语义处理 padding。
- 备查（非请求、无算子组动作）：prefill kernel 只跳过 `-1`、不跳过 0（`DLC_Custom_Kernel/dlc_kernels/conv/causal_conv1d.c:63,156,279,371`），与 update kernel 的空槽约定不同。但 prefill 用的也是 block table 首列的索引（`short_conv_attn.py:371/183`，真实请求 ≥ 1），且 padded 行在行号判断之前就被 `seqlen <= 0` 跳过（两个 kernel 都是先判 seqlen、再判 row），因此 vLLM 路径不会把 page 0 交给 prefill kernel —— 只有将来手工构造索引时才需要留意。
- 复现脚本：`probes/probe_causal_conv1d_state_index_zero.py`（2026-09-22 修正版，含 null block / pad slot 断言与 state 不变校验）。
- 本次调整（2026-09-22）：需求类型「新增算子」→「其他（复用已有算子）」；优先级 P0 → P2；算子名称改为复用 `torch._causal_conv1d_update`；page 0 一项由「算子缺陷」更正为「null block 设计行为」。
- 本次调整（2026-09-22 晚，最终）：**由「可复用（撤回）」→「需修复（提交）」**，优先级 P2→**P0**：采用算子组交付的新算子，请其修复 wrapper 整池拷贝等问题（详见 `给算子组_01_02_修复要点.md` 第二节）；修复期间插件用 `torch._causal_conv1d_update` 兜底。


## 2026-09-22 新算子实测（算子组已交付 `dlc_kda_causal_conv_update`）

算子组已提交专用算子 `dlc_kda_causal_conv_update`（同文件 `CausalConv1d.cxx`，
kernel `custom_kda_causal_conv_update`，返回 `(out, committed_state)` 并就地更新 `conv_state`）。
实测脚本：`probes/probe_new_kda_conv_update.py`、`probes/probe_new_kda_conv_update_window.py`、
`probes/probe_new_kda_conv_update_scaling.py`。

**实测结论：数值完全正确（plain decode `state_err=0`；spec/L>3 仅作背景参考），但在真机几何下比现有算子慢约
28~36 倍（7.21 ms vs 0.2~0.3 ms），修复前不建议上热路径。**（`dlc:0` 的 strided 拷贝异常慢，会把现有算子也拖到
0.59 ms、显示成 ~13 倍；初版「慢 100~250 倍」是按流量推算的估计值。）

**范围（2026-09-22 明确）**：适配初期只要求 **plain decode**（每请求 1 token、`state_len=3`、不传 `num_accepted_tokens`）。
**MTP/spec 相关（多 token、`num_accepted_tokens` 提交语义、`state_len>3`）本阶段不做**，本文档中相关段落仅作背景留档。

| 用例 | 新算子 | 已有 `_causal_conv1d_update` |
|---|---|---|
| plain decode B=4，含 null slot 0 | `out_err=0`、`state_err=0`；null 行输出全 0、state 不变 | 一致（null 行输出全 0、state 不变） |
| spec T=4 `accepted=[0,2,4,1]`；T=6 `[3,0]` | `out_err=0`、`state_err=0` | **不吻合**：`out_err=1.07`、`state_err=2.85` |
| L=4 / L=6（num_spec=1/3），含 `accepted=0` | 全部 `state_err=0`（`out_err≤9.8e-04`） | 全部不吻合（`state_err≈2.6~3.0`） |
| 3D qkv `[B,T,D]`、`state_indices` 2D `[B,T]`（取第 0 列） | 支持 | 不支持 2D |
| `D=320`（非 256 倍数） | 报错：`V1 BF16 kernel requires D_conv to be a multiple of 256` | 支持 |

**spec 语义不一致（本阶段不要求处理，留待启用 MTP 时再对齐）**：以「滑窗递推 + 处理完 `num_accepted_tokens` 个 token 后提交」
为 reference 时，新算子逐元素精确一致，而 `_causal_conv1d_update` 的 out/state 都对不上。
旧 kernel 的注释写的是另一套约定：`offset = na<0 ? 0 : na-1`、`shift = na<0 ? seqlen : 1`
（`DLC_Custom_Kernel/dlc_kernels/conv/causal_conv1d_update.c:37-40,148-149,198`），
即把 `num_accepted_tokens` 当作**读窗口偏移 + 整窗回写**，而不是「回滚到 accepted 处的快照」。
两者不能互换；**适配初期不走 MTP/spec，本条本次不需要算子组处理**，后续启用 MTP 时再确认以哪套为契约。

**性能（决定性，D=3072 真实几何 `num_blocks=1509`）**：

（2026-09-22 晚**重测**：每档先各跑一次 warm-up 再计时，排除首次 kernel 加载的一次性开销；
上一版的 70.3 ms 含首次初始化，已作废）

| 槽位数 | 新算子（plain B=8 单 token） | 已有 `_causal_conv1d_update` | 新算子（spec T=4, acc=3） | 已有 |
|---|---|---|---|---|
| 4（退化：idx 会重复） | 0.61 ms | 0.86 ms | 0.34 ms | 0.72 ms |
| 256 | 1.75 ms | 0.61 ms | — | — |
| 1024 | 5.17 ms | 0.57 ms | 5.26 ms | 0.73 ms |
| **1509（真机 `num_blocks`）** | **7.49 ms** | **0.59 ms（≈13×）** | **7.41 ms** | 0.75 ms |
| 2048 | 9.99 ms | 0.60 ms | — | — |
| 3072 | 14.73 ms | 0.72 ms | — | — |

**关键特征：新算子耗时随「槽位总数」线性增长（4→3072 槽位：0.61→14.73 ms），
而现有算子与槽位规模无关（0.57~0.86 ms）** —— 成本来自对整个 state 池的无条件搬运（与活跃请求数无关，B=8 只用到 8 个页）。

**成本归因（2026-09-22 深夜实测，`probe_new_kda_conv_update_cost.py`，S=1509/D=3072，池子 26.5 MB）**：

| 构成 | 实测 | 占比 |
|---|---|---|
| **kernel 内整池 `HBM2VMem`（`kda_causal_conv_update.c:109`）** | **≈ 6.9 ms** | **≈ 94%** |
| host wrapper 的 4 遍整池拷贝（`CausalConv1d.cxx:401-415`） | 0.47 ms | ≈ 6% |

- 成本 ∝ **池子字节数**（`t ≈ 0.74 ms + 0.251 ms/MB`，即 ≈ 4 GB/s 的整池搬运带宽），**与活跃请求数无关**
  （B=1/8/64 → 7.59/7.45/7.50 ms）；
- 因此 13× 的主因是 **kernel 侧的整池 DMA**，不是 host wrapper（旧结论是按流量推算的，非实测，已作废）。

kernel 侧（真正的主因）—— `kda_causal_conv_update.c:109` 每次调用无条件把整个池搬进 VMEM，
只为给第二个返回值 `committed_state`（全池快照）打底，而**调用方根本不用这个返回值**
（上游 `vllm/models/glm5next/nvidia/kda.py:445` 只取 `out`）：

```c
/* vmem_commit = all slot snapshots */
HBM2VMem(hbm_conv_state, vmem_commit, state_slots * state_len * dim_bf16);   /* :109 整池 DMA */
```

host 侧（次要，0.47 ms/次，34 层 ≈ 16 ms/token）—— wrapper 每次调用做 4 遍整池拷贝：
`transpose(1,2).contiguous()`（①）→ `clone()`（②）→ kernel → `transpose(1,2).contiguous()`（③）→
`conv_state.copy_(committed_state)`（④）；另外 `num_accepted_tokens=None` 分支里 `qsl.to(at::kCPU)`
构造 accepted，**每次调用都会强制一次 D2H 同步**（`CausalConv1d.cxx:349/:376-378`）。
注意 ① 恰好把池子视图拉平，kernel 内部 `slot*state_len*dim_bf16` 的紧凑寻址才成立 ——
所以「改零拷贝」必须同时给 kernel 加行距元数据（`probe_new_kda_conv_update_stride.py` 证明当前 update kernel 走 host 拉平后是对的）。

而 `_causal_conv1d_update` 走 `make_flat_storage_view` 零拷贝、只按 `cache_indices` 读写活跃页。
按 34 层、每 token 一次算，新算子单步 decode 要多花约 **34 × 7.6 ms ≈ 260 ms/token**，仍然不可接受
（修复目标：1509 槽位 ≤ ~1 ms，与现有算子同量级）。

**其他行为差异（若后续仍要接入需注意）**：
1. `idx=-1`（pad slot）时**输出不写**（`at::empty` 残留脏值，实测 `out_maxabs=1.71`），
   而现有算子返回全 0；vLLM 当前 padding 用 0（null block）故不触发，但属健壮性缺口；
2. 2D `qkv` 必须显式传 `query_start_loc`（现有算子可省），插件需额外 `arange(B+1)` 且 `max_query_len≥1`；
3. 要求活跃请求的 state slot **互不重复**（否则结果未定义）。

**处置（2026-09-22 晚最终定稿）**：本条改为「**提交**」——采用新算子，请算子组修：
（a）**去掉 kernel 的整池 DMA**（`kda_causal_conv_update.c:109`；`accepted<=0` 的 slot 直接跳过回写即可保持语义），
   并把 `committed_state` 改为只覆盖活跃 slot / 可选输出；顺带去 wrapper 的 4 遍整池拷贝
    （要走零拷贝 strided 访问，须同时支持行距元数据，否则会重演 #01 的 state stride 缺陷），
（b）不传 `num_accepted_tokens` 时（plain decode 就是这条）不要在 host 侧 `qsl.to(kCPU)`，
（c）pad 行（`idx=-1`）输出写 0（现在是 `at::empty` 的脏值），
（d）【本阶段不要求】spec 提交语义留待启用 MTP 时再对齐（新算子与「整窗滚动 + accepted 快照」参考一致；
旧 `_causal_conv1d_update` 在 `acc<T` 时不吻合，`state_err≈2.6~3.0`）；
修复期间热路径（本轮只有 plain decode）用 `torch._causal_conv1d_update` 兜底，修好后切换并重跑对齐。
- 复现脚本：`probes/probe_new_kda_conv_update.py`、`..._window.py`、`..._scaling.py`、
  `..._cost.py`（成本归因）、`..._stride.py`（池子行距正确性）。
- **验收回归测试**：`vllm-cl/tests/test_glm5_kda_conv_update_op.py`
  （`DLC_TEST_DEVICE=dlc:6 python -m pytest tests/test_glm5_kda_conv_update_op.py -v`；当前 7 passed / 4 xfailed，
  4 个 xfail 即 pad 行、无 host 同步、graph 捕获、槽位无关成本这四条待修标准）。
- 2026-09-22 晚复核：三个脚本在干净设备上重跑，结论不变（新算子 plain/spec/L=4/L=6 全部 `state_err=0`；
  `idx=-1` 行输出为脏值 `out_maxabs≈1.05~1.71`；`D=320` 报「需 256 倍数」；
  1509 槽位 **7.49ms vs 0.59ms（≈13×，加 warm-up 重测）**，且耗时随槽位总数线性增长）。
  2026-09-22 深夜进一步归因（`probe_new_kda_conv_update_cost.py`）：13× 里 **94% 在 kernel 内部的整池
  `HBM2VMem`（`:109`）**，host wrapper 的 4 遍拷贝只有 0.47ms（6%）；成本 ∝ 池子字节数、与活跃请求数无关。
  主脚本 `probe_new_kda_conv_update.py` 初版在 `L>3` 时只滚动最后 3 行、把前 `L-3` 行原样留在快照里，
  会误报 `L=6 state_err=2.695`；已按 `..._window.py` 的「整窗滚动」语义修正，修正后 `L=6` 同样 `state_err=0`。
  即：**新算子在 L=3/4/6 的提交语义与窗口参考一致，不存在 L=6 的精度缺陷**。

## 2026-09-28 修复后复测（结论：4 个问题全部修复，验收用例 11/11 通过）

算子组在 `DLC_Custom_Kernel` `jinxing_perf_7950_260924`（commit `a6adf5922`，删除整池 DMA）
+ `pytorch` `jinxing_kda_causal`（commit `b791208401f`，删除 host 拷贝/D2H、改 ABI）修复；
已 cherry-pick 到本地 `test-glm53flash`（`DLC_Custom_Kernel 907ad482e`、`pytorch a18fc46ddcd`）。

新 ABI（与修复前不同）：
- 返回值从 `(out, committed_state)` 改为**单个 `out`**；
- `conv_state` 逻辑形状 **`[S_slot, state_len, D]`**（SD），由 kernel **直接 in-place 写回**（不再返回整池快照）；
- `weight` 为**已转置**的 `[4, D]` fp32（`[D,4]` 会被明确报错）；
- `out` 由 `at::zeros_like` 预置（pad 行天然为 0）。

| 原问题 | 修复前实测 | 修复后实测（dlc:1/2/3/6） |
|---|---|---|
| P0-1 kernel 每次整池 `HBM2VMem` | 1509 槽 7.49 ms，随槽位总数线性增长 | **0.06 ms**；slots=64/1509/3072、B=1/8/64 基本不变；现有 `_causal_conv1d_update` 0.26 ms → 新算子 **0.2×** |
| P0-2a wrapper 内 host D2H 同步 | 每次入队 ≈ 设备耗时（被 `.item()`/`.to(kCPU)` 卡住） | 入队中位数 **0.05 ms**（纯 launch；偶发 ~100 ms 为 GC/分配抖动，按中位数统计） |
| P0-2b 无法被 DLCGraph 捕获 | 捕获报错 | **捕获成功**，replay 与参考一致（`test_op_is_graph_capturable_and_replays_correctly` PASS） |
| P1-3 host 4 遍整池拷贝 | ≈0.23 ms/层（34 层 ≈7.8 ms/token） | wrapper 已无任何 state 拷贝（只剩 `qkv.contiguous()` + `zeros_like(out)`） |
| P1-4 `idx=-1` 行输出脏值 | `at::empty` 残留 | `at::zeros_like` 预置 → pad 行全 0（`test_pad_slot_output_is_zero` PASS） |

回归用例：`tests/test_glm5_kda_conv_update_op.py` **11/11 PASS**（正确性 7 条 + pad 行 / 无 host 同步 /
graph replay / 槽位无关成本 4 条），原先 4 条 `xfail` 已改为普通用例。
数值：plain decode / spec / `state_len=3,4,6` / 池子非连续视图：`out_err` 0~9.8e-4（bf16 级）、`state_err=0`。

遗留（不影响本轮接入）：kernel 仍按 `slot*state_len*D` 紧凑寻址、不读 `conv_state.stride`；真实分页池里
conv 段与 recurrent 段同页（非连续视图）靠 DLC 的 `dlc_dlc_contiguous` 自动「连续副本 + 回写」，
结果正确、代价约 0.05 ms/层。若要完全零拷贝需 kernel 支持 stride 寻址。

## 优先级
P0

## 目标版本与期望时间
GLM-5.3-Flash 适配 Phase 2（性能阶段）。建议与 #01、#03、#04 一起在长上下文（≥32K prefill）性能验收前给出，具体里程碑待算子组排期确认。

## 算子名称
修复 `torch.dlc_kda_causal_conv_update`（采用新算子；修复 kernel 整池 DMA、host D2H 同步、pad 行输出写 0）

## 背景与功能说明
使用场景：GLM-5.3-Flash 有 34 层 KDA 线性注意力，每层的 q/k/v 短卷积（kernel=4 深度可分离卷积）在 decode 阶段对每个请求的新 token 做卷积，并把历史原位滚动写回 conv state 页。
（后续启用 MTP/spec 时还需要按 `num_accepted_tokens` 只提交被接受的 token，避免 rejected draft token 污染 state —— **本阶段不做**。）
当前问题：插件仍走 native 组合实现（`vllm_cl/patch/patch_kda.py:45` 的 `causal_conv1d_update_dlc`），在 Python 里逐请求循环，34 层 × batch 会产生大量 mini kernel，且无法进入 CUDA graph。
目标功能：直接把该路径切到已有的 `torch._causal_conv1d_update`，语义与布局按「支持范围与约束」对齐。
非目标：prefill 变长路径（见 `01_dlc_kda_causal_conv_prefill.md`）；卷积权重本身的生成；APC（现有算子未实现，见约束）。

## 期望 API 与函数签名
```python
# 现有算子，无需改动。KDA decode / MTP verify 的调用形态：
torch._causal_conv1d_update(
    x,                                    # [B, dim] 或 [B, dim, T] bf16
    weight,                               # [dim, K] bf16（KDA 的 [dim, 1, K] 需 squeeze(1)）
    bias,                                 # [dim] bf16 或 None
    conv_state,                           # [num_pages, dim, K-1] bf16，原位更新（通道维 stride == 1）
    conv_state_indices,                   # [B] int32，block_table[:, 0]；0=null block、-1=pad slot 会被跳过
    num_accepted_tokens=None,             # [B] int32，spec 路径
    query_start_loc=None,                 # [num_seqs+1] int32，packed / varlen 路径
    max_query_len=-1,
    block_idx_last_scheduled_token=None,  # 必须为 None（APC 未实现）
    initial_state_idx=None,               # 必须为 None（APC 未实现）
    has_bias=True,
    silu_activation=True,
)  # -> [B, dim]（x.dim()==2）或 [B, dim, T]（x.dim()==3），bf16
```

## 参数与返回值
输入：
- `x`：`[B, dim]`（每请求 1 个 token）或 `[B, dim, T]`（多 token / spec）；packed 变长时配合 `query_start_loc` 使用。
- `weight`：`[dim, K]`，BF16，深度卷积权重（每通道独立）；KDA 侧为 `[dim, 1, K]`，需 `squeeze(1)`。
- `bias`：`[dim]`，BF16 或 None（wrapper 内部用零向量占位）。
- `conv_state`：`[num_pages, dim, K-1]`，BF16/FP32，**原位更新**。
- `conv_state_indices`：`[B]`，int32（wrapper 会 `.to(kInt).contiguous()`）。
- `num_accepted_tokens`：`[B]`，int32；未提供时等价于全 -1。
- `query_start_loc`：`[num_seqs+1]`，int32；packed/varlen 输入时的序列边界。

属性：
- `max_query_len`：int，默认 -1，仅 `query_start_loc` 存在时参与确定 seqlen。
- `block_idx_last_scheduled_token` / `initial_state_idx`：**必须为 None**（wrapper 显式 `TORCH_CHECK` 拒绝，APC 未实现）。
- `has_bias` / `silu_activation`：bool，默认 True。

返回值：
- 与 `x` 同形同 dtype（`x.dim()==3` 时内部 reshape 后转置回来），输出缓冲区由 wrapper 以 `at::zeros` 分配。

边界与特殊行：
- `conv_state_indices[b] == 0`（null block）或 `== -1`（pad slot）：该行**输出全零、`conv_state` 不变**。
- 全部行都被跳过时 kernel 直接返回，输出仍为全零（提前 zeros 分配所致）。

## 功能语义
现有 kernel 的行为（每个请求独立，页号由 `conv_state_indices` 指定）：

```text
for b in range(B):
    row = conv_state_indices[b]
    if row == 0 or row == -1:        # null block / pad slot
        out[b] = 0                   # 输出零，不改 state
        continue
    hist = conv_state[row]           # [dim, K-1]
    for t in range(seqlen_b):
        win  = concat(hist, x[b, :, t])
        y[b, :, t] = silu(sum(win * weight, -1) + bias)
        hist = concat(hist[:, 1:], x[b, :, t])
    conv_state[row] = hist           # 原位写回
```

- spec 路径（**本阶段不要求**）：`num_accepted_tokens[b] = na` 时，以 `offset = max(na-1, 0)` 为读取基准，只把被接受的 token 提交进 state，被拒绝的 token 仍计算输出但不落 state。
- in-place / alias：`conv_state` 原位写；不修改 `x`；返回新张量。
- 空 / 零长度：`B == 0` 或 `seqlen == 0` 的行跳过。
- 确定性：无随机、无原子序依赖。
- 累加精度：内部按输入 dtype（bf16/fp32）计算，KDA 侧要求 fp32 累加可由算子组确认。

## 支持范围与约束
支持的 dtype：`x`/`conv_state` 为 BF16 或 FP32（两者 dtype 必须一致）；索引 int32。
支持的 shape：`B` 动态；`dim` = 3×H_local×128（TP8 下 3×8×128 = 3072）；`T` 支持 1~8（MTP draft 深度）；`K` 由 `weight` 形状决定，本需求只覆盖 `K=4`（KDA 的 short_conv_kernel_size）。
layout / stride 要求（wrapper 的 `validate_state_metadata` 强制，实测均命中过）：
- `conv_state` 必须 3D，且**通道维 `stride(1) == 1`**（否则报 `conv state dim axis must be contiguous`）；
- BF16 下 `conv_state.storage_offset()` 必须为偶数；
- `weight` 必须 2D；`x` 必须 2D 或 3D；`bias` 必须 1D 且长度 == `dim`；
- 所有尺寸/stride/offset 必须在 int32 范围内。
device：DLC 单卡（TP=8 时每卡各跑自己的分片）。
执行模式：支持 CUDA graph 捕获（decode 主路径）；graph 内 shape 静态，padded 行由上游填 `NULL_BLOCK_ID`。
其他限制：APC 参数（`block_idx_last_scheduled_token`、`initial_state_idx`）必须为 `None`；索引必须来自 block table（真实请求的页号 ≥ 1）。

## 精度要求
Golden / reference：`vllm-cl/vllm_cl/patch/patch_kda.py:45`（`causal_conv1d_update_dlc`）的 fp32 累加实现；探针 `probes/probe_causal_conv1d_state_index_zero.py` 自带同一 reference，并额外显式建模 null block / pad slot 约定。
输入数据分布与随机种子：`x ~ N(0,1)`、state ~ `N(0,0.5)`，`torch.manual_seed(0)`；覆盖 spec 全接受 / 部分接受 / 全拒绝。
样本数量：每类 shape ≥ 200 组。
比较 dtype：BF16 输出 + 写回后的 state。
atol / rtol：`atol=2e-2, rtol=2e-2`（实测真实页 `0.00e+00`）。
必须逐元素一致：
- `conv_state_indices` 为 0 / -1 的行：输出全零、`conv_state` 与调用前 **bit-exact** 相同（CUDA graph padding 依赖此行为）；
- `num_accepted_tokens = 0` 时：`conv_state` 与调用前 bit-exact 相同。
NaN / Inf / 极值：不崩溃、不污染 state。
累加精度：FP32。

## 测试范围
本条不需要算子组开发，下面是**插件接入回归**的用例清单（供接入时使用，也便于算子组复核现有行为）：
- [ ] 基本功能与 reference 对齐（`B ∈ {1,8,64,256}` 单 token decode）
- [ ] 多 dtype 精度（bf16 / fp32 各一遍）
- [ ] 静态 shape 与动态 shape（含 CUDA graph 回放路径）
- [ ] 边界：`B=0`、`T=0`、`K=2`
- [ ] 非 contiguous / 特殊 stride（`conv_state` 由 `[pages, K-1, dim]` `permute(0,2,1)` 得到，stride(1)==1）
- [ ] 非法参数与异常：`conv_state` 通道维不连续、bf16 奇数 `storage_offset`、APC 参数非 None 必须报错；
      `conv_state_indices` 越界目前**没有校验**（kernel 会按页号直接寻址），必须由插件侧保证索引合法
- [ ] 单卡；多卡/分布式（TP8 各分片独立）
- [ ] 长时间稳定性（decode 循环不泄漏、不污染 state）
- [ ] 性能基准：34 层 × `B=64` 单步 decode 总耗时（目标 < 1ms）与 native 对比

具体测试用例、shape、数据范围和预期结果：
1. 真实页：`conv_state_indices=[1] / [2] / [3] / [1,2,3]` → 输出与 reference 一致，state 正确滚动。
2. null block：`[0] / [0,2] / [1,0]` → 含 0 的行输出全零、state 不变，其余行结果正确（这是预期行为，不是缺陷）。
3. pad slot：`[-1] / [-1,1]` → 含 -1 的行输出全零、state 不变。
4. spec（**本阶段不要求**，供后续）：`T ∈ {1,2,4,6,8}`，`num_accepted_tokens` 覆盖 0 / 部分 / 全部。
5. 异常：APC 参数非 None → `causal_conv1d_update: APC not implemented`；`conv_state` 通道维不连续 →
   `conv state dim axis must be contiguous`；bf16 奇数 `storage_offset` →
   `bf16 conv state storage_offset must be even`（后两条来自共享的 `validate_state_metadata`，报错前缀为 `causal_conv1d:`）。

## 验证环境与依赖版本
DLC 硬件：8 卡，设备名 `/dev/cltech0-7`（TP=8，无 PP）
vLLM commit：`f5361fb09f`
vLLM DLC（vllm-cl）commit：`a14fc5b`
PyTorch / torch-dlc：torch 2.5.0（DLC 构建），Python 3.10.16
关键环境变量：`VLLM_USE_DLC_COL_MAJOR_MATMUL=1`
模型：GLM-5.3-Flash-Mix（AWQ 4-bit，`linear_num_heads=64`、`linear_head_dim=128`、`short_conv_kernel_size=4`、`gate_lower_bound=-5.0`）

## 验收标准
- [ ] 插件侧接入后，**KDA decode（plain，本轮范围）** 数值与 native reference 一致（atol/rtol 2e-2，实测 0.00e+00；MTP verify 后续再验收）
- [ ] 真实请求使用的 state 页号全部来自 block table，且从不为 0
- [ ] `conv_state_indices` 为 0 / -1 的行：输出全零、state bit-exact 不变（CUDA graph padding 依赖）
- [ ] `conv_state` 通道维 stride、bf16 storage_offset、int32 索引等约束在插件侧被满足（必要时提前断言）
- [ ] 插件侧自查：真实请求使用的 `conv_state_indices` 全部来自 block table（≥ 1），不存在 page 0 落到真实 state 的情况

## Reference、日志与附件
Reference 实现：`vllm_cl/patch/patch_kda.py:45`（`causal_conv1d_update_dlc`，fp32 累加）。
算子实现：kernel `DLC_Custom_Kernel/dlc_kernels/conv/causal_conv1d_update.c`（null/pad 跳过见 `:65-66,145,293,346`）；wrapper `pytorch/aten/src/ATen/native/dlc/CausalConv1d.cxx:170`（`at::zeros` 见 `:242`，`validate_state_metadata` 见 `:65-85`）。
上游约定：`vllm/v1/attention/backends/utils.py:46`（`NULL_BLOCK_ID = 0`）、`vllm/v1/core/block_pool.py:188-191`（null block 永不分配）、`vllm/v1/attention/backends/short_conv_attn.py:371/467`（索引取自 block table 首列、padding 行填 `NULL_BLOCK_ID`）。
空槽约定差异（备查、无实际影响）：prefill kernel `DLC_Custom_Kernel/dlc_kernels/conv/causal_conv1d.c:63,156,279,371` 只跳过 `-1`；由于 page 0 是保留 null block，vLLM 路径不会给 prefill 传 0。
调用点：`vllm/models/glm5next/nvidia/kda.py:445`（spec）、`:479`（plain decode）。
复现脚本：`probes/probe_causal_conv1d_state_index_zero.py`（2026-09-22 修正版）、`probes/probe_causal_conv1d_prefill_update.py`。
相关文档：`docs/glm5_flash/GLM-5.3-Flash_vLLM-CL_适配研究.md` §5.4 算子 `dlc_kda_causal_conv_update`。
